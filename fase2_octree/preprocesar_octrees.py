"""Genera y valida octrees para el objetivo especifico 1.

La ejecucion parte exclusivamente de las mallas ``.off`` de ModelNet40,
muestrea una sola nube por modelo y usa esa misma nube para las
resoluciones 32^3 y 64^3. Cada salida incluye un octree v1 validado y un
manifiesto trazable. Admite una muestra controlada para aceptar la
implementacion sin procesar todavia ModelNet40 completo. No se ejecutan HCE,
clasificadores ni Net5.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from octree import (  # noqa: E402
    construir_grid_octree,
    leer_off,
    muestrear_superficie_con_normales,
    normalizar_malla,
    profundidad_de,
)
from octree_real import (  # noqa: E402
    OCTREE_FORMAT_VERSION,
    carga_binaria_sin_comprimir,
    carga_npz_sin_comprimir,
    construir_octree,
    guardar_octree_disperso,
    medir_memoria_real_python,
    octree_a_grid_denso,
    reconstruir_octree_desde_npz,
)
from validacion_objetivo1 import (  # noqa: E402
    guardar_json_atomico,
    manifiesto_base,
    metricas_estructura,
    ruta_portable,
    semilla_estable_modelo,
    sha256_archivo,
    validar_equivalencia_denso_octree,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - solo afecta la barra visual
    def tqdm(iterable, **_kwargs):
        return iterable


CLASES_MODELNET40 = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle",
    "bowl", "car", "chair", "cone", "cup", "curtain", "desk", "door",
    "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person", "piano",
    "plant", "radio", "range_hood", "sink", "sofa", "stairs", "stool",
    "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox",
]
CLASE_A_INDICE = {clase: indice for indice, clase in enumerate(CLASES_MODELNET40)}
CONTEOS_OFICIALES = {"train": 9843, "test": 2468}


def _validar_resolucion(resolucion: int) -> None:
    if resolucion < 2 or resolucion & (resolucion - 1):
        raise ValueError(f"La resolucion debe ser potencia de dos: {resolucion}")


def recolectar_modelos(dataset_root: Path, splits: list[str]) -> list[dict]:
    """Enumera ModelNet40 en un orden canonico e independiente del sistema."""
    modelos = []
    for split in splits:
        for categoria in CLASES_MODELNET40:
            directorio = dataset_root / categoria / split
            for ruta in sorted(directorio.glob("*.off")):
                relativa = ruta.relative_to(dataset_root).as_posix()
                modelos.append({
                    "ruta": str(ruta),
                    "ruta_relativa": relativa,
                    "model_id": ruta.stem,
                    "categoria": categoria,
                    "etiqueta": CLASE_A_INDICE[categoria],
                    "split": split,
                })
    return modelos


def seleccionar_muestra_controlada(
    modelos: list[dict], n_por_categoria_split: int,
) -> list[dict]:
    """Selecciona deterministamente hasta ``n`` modelos por categoria y split."""
    if n_por_categoria_split <= 0:
        raise ValueError("--muestra-por-categoria-split debe ser positivo")
    conteos: dict[tuple[str, str], int] = {}
    seleccion = []
    for modelo in modelos:
        clave = (modelo["categoria"], modelo["split"])
        usados = conteos.get(clave, 0)
        if usados < n_por_categoria_split:
            seleccion.append(modelo)
            conteos[clave] = usados + 1
    return seleccion


def _guardar_npz_atomico(raiz, destino: Path, etiqueta: int,
                         profundidad_max: int, metadatos: dict) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporal = tempfile.mkstemp(
        prefix=f".{destino.stem}.", suffix=".npz", dir=destino.parent,
    )
    os.close(descriptor)
    try:
        guardar_octree_disperso(
            raiz, temporal, etiqueta, profundidad_max, metadatos=metadatos,
        )
        os.replace(temporal, destino)
    except Exception:
        try:
            os.unlink(temporal)
        except FileNotFoundError:
            pass
        raise


def _tamano_npz_denso_temporal(grid: np.ndarray, directorio: Path) -> int:
    directorio.mkdir(parents=True, exist_ok=True)
    descriptor, temporal = tempfile.mkstemp(suffix=".npz", dir=directorio)
    os.close(descriptor)
    try:
        np.savez_compressed(temporal, grid=grid)
        return Path(temporal).stat().st_size
    finally:
        try:
            os.unlink(temporal)
        except FileNotFoundError:
            pass


def procesar_modelo(tarea: dict) -> dict:
    """Procesa un modelo de forma autocontenida; funcion segura para workers."""
    ruta = Path(tarea["ruta"])
    output_root = Path(tarea["output_root"])
    manifest_root = Path(tarea["manifest_root"])
    resoluciones = tarea["resoluciones"]
    semilla_modelo = semilla_estable_modelo(
        tarea["semilla_base"], tarea["ruta_relativa"],
    )
    sha_origen = sha256_archivo(ruta)

    t0 = time.perf_counter()
    vertices, caras = leer_off(str(ruta))
    vertices = normalizar_malla(vertices)
    rng = np.random.default_rng(semilla_modelo)
    puntos, normales = muestrear_superficie_con_normales(
        vertices, caras, tarea["n_puntos"], rng,
    )
    tiempo_preparacion_ms = (time.perf_counter() - t0) * 1000.0

    manifiesto = manifiesto_base(
        model_id=tarea["model_id"],
        categoria=tarea["categoria"],
        split=tarea["split"],
        ruta_origen_relativa=tarea["ruta_relativa"],
        sha256_origen=sha_origen,
        semilla_base=tarea["semilla_base"],
        semilla_muestreo=semilla_modelo,
        n_puntos_muestreo=tarea["n_puntos"],
    )
    manifiesto["tiempo_preparacion_nube_ms"] = tiempo_preparacion_ms

    filas = []
    for resolucion in resoluciones:
        _validar_resolucion(resolucion)
        profundidad_max = profundidad_de(resolucion)

        tracemalloc.start()
        t0 = time.perf_counter()
        raiz = construir_octree(puntos, normales, profundidad_max)
        tiempo_construccion_ms = (time.perf_counter() - t0) * 1000.0
        _, memoria_pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        estructura = metricas_estructura(raiz, resolucion, profundidad_max)
        memoria_python = medir_memoria_real_python(raiz)
        carga_binaria = carga_binaria_sin_comprimir(raiz)

        t0 = time.perf_counter()
        grid_directo = construir_grid_octree(puntos, normales, resolucion)
        tiempo_denso_ms = (time.perf_counter() - t0) * 1000.0
        grid_arbol = octree_a_grid_denso(raiz, resolucion)
        equivalencia = validar_equivalencia_denso_octree(
            puntos, normales, raiz, resolucion,
            grid_directo=grid_directo, grid_arbol=grid_arbol,
        )

        destino = (
            output_root / f"octrees_{resolucion}" / tarea["categoria"]
            / tarea["split"] / f"{tarea['model_id']}.npz"
        )
        if destino.exists() and not tarea["sobrescribir"]:
            raise FileExistsError(
                f"Ya existe {destino}; use --sobrescribir para regenerarlo"
            )

        metadatos_npz = {
            "model_id": tarea["model_id"],
            "categoria": tarea["categoria"],
            "split": tarea["split"],
            "semilla_muestreo": semilla_modelo,
            "n_puntos_muestreo": tarea["n_puntos"],
            "archivo_origen_sha256": sha_origen,
        }
        t0 = time.perf_counter()
        _guardar_npz_atomico(
            raiz, destino, tarea["etiqueta"], profundidad_max, metadatos_npz,
        )
        tiempo_guardado_ms = (time.perf_counter() - t0) * 1000.0

        # Verificacion del archivo realmente escrito: cargar, reconstruir y
        # volver a comparar contra la rejilla construida desde los puntos.
        raiz_cargada, etiqueta_cargada, profundidad_cargada = (
            reconstruir_octree_desde_npz(str(destino))
        )
        if etiqueta_cargada != tarea["etiqueta"]:
            raise AssertionError("La etiqueta cambio durante la persistencia")
        if profundidad_cargada != profundidad_max:
            raise AssertionError("La profundidad cambio durante la persistencia")
        equivalencia_persistida = validar_equivalencia_denso_octree(
            puntos, normales, raiz_cargada, resolucion,
            grid_directo=grid_directo,
        )

        tamano_npz = destino.stat().st_size
        carga_npz = carga_npz_sin_comprimir(destino)
        tamano_npz_denso = _tamano_npz_denso_temporal(
            grid_directo, output_root / ".temp_referencia_densa",
        )
        salida_relativa = destino.relative_to(output_root).as_posix()
        datos_resolucion = {
            "resolucion": resolucion,
            "profundidad_max": profundidad_max,
            **estructura,
            "tiempo_construccion_ms": tiempo_construccion_ms,
            "tiempo_guardado_ms": tiempo_guardado_ms,
            "memoria_pico_construccion_bytes": int(memoria_pico),
            "memoria_estructura_python_bytes": memoria_python["memoria_python_bytes"],
            "carga_binaria_nodos_sin_comprimir_bytes": carga_binaria["total_bytes"],
            "carga_binaria_sin_comprimir_bytes": carga_npz["total_bytes"],
            "bytes_binarios_por_nodo": carga_binaria["bytes_por_nodo"],
            "tamano_npz_bytes": tamano_npz,
            "archivo_npz": salida_relativa,
            "archivo_npz_sha256": sha256_archivo(destino),
            "referencia_densa": {
                "forma": list(grid_directo.shape),
                "dtype": str(grid_directo.dtype),
                "tiempo_construccion_ms": tiempo_denso_ms,
                "carga_sin_comprimir_bytes": int(grid_directo.nbytes),
                "tamano_npz_comprimido_bytes": int(tamano_npz_denso),
            },
            "equivalencia_antes_guardado": equivalencia,
            "equivalencia_despues_carga": equivalencia_persistida,
        }
        manifiesto["salidas"][str(resolucion)] = datos_resolucion

        filas.append({
            "model_id": tarea["model_id"],
            "categoria": tarea["categoria"],
            "split": tarea["split"],
            "semilla_muestreo": semilla_modelo,
            **{clave: valor for clave, valor in datos_resolucion.items()
               if clave not in {"referencia_densa", "equivalencia_antes_guardado",
                                "equivalencia_despues_carga", "nodos_por_nivel"}},
            "nodos_por_nivel": "|".join(map(str, estructura["nodos_por_nivel"])),
            "densa_carga_sin_comprimir_bytes": int(grid_directo.nbytes),
            "densa_tamano_npz_comprimido_bytes": int(tamano_npz_denso),
            "equivalencia": True,
        })

    ruta_manifiesto = (
        manifest_root / tarea["categoria"] / tarea["split"]
        / f"{tarea['model_id']}.json"
    )
    guardar_json_atomico(manifiesto, ruta_manifiesto)
    return {
        "ok": True,
        "model_id": tarea["model_id"],
        "categoria": tarea["categoria"],
        "split": tarea["split"],
        "manifiesto": str(ruta_manifiesto),
        "filas": filas,
    }


def _escribir_csv_atomico(filas: list[dict], destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporal = tempfile.mkstemp(
        prefix=f".{destino.name}.", suffix=".tmp", dir=destino.parent,
    )
    os.close(descriptor)
    try:
        campos = sorted({clave for fila in filas for clave in fila})
        with open(temporal, "w", newline="", encoding="utf-8") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=campos)
            escritor.writeheader()
            escritor.writerows(filas)
        os.replace(temporal, destino)
    except Exception:
        try:
            os.unlink(temporal)
        except FileNotFoundError:
            pass
        raise


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera y valida los octrees definitivos de ModelNet40",
    )
    parser.add_argument(
        "--dataset-root", type=Path,
        default=RAIZ_PROYECTO / "Dataset" / "ModelNet40",
        help="Directorio que contiene las 40 carpetas de ModelNet40",
    )
    parser.add_argument(
        "--output-root", type=Path, default=RAIZ_PROYECTO / "data",
        help="Directorio para octrees_<R>/ y manifests/",
    )
    parser.add_argument(
        "--resultados-dir", type=Path,
        default=RAIZ_PROYECTO / "resultados" / "objetivo1",
    )
    parser.add_argument("--resoluciones", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--splits", nargs="+", choices=["train", "test"],
                        default=["train", "test"])
    parser.add_argument(
        "--categorias", nargs="+", choices=CLASES_MODELNET40, default=None,
        help="Categorias incluidas; por defecto se consideran las 40",
    )
    parser.add_argument("--n-puntos", type=int, default=20000)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--procesos", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument(
        "--limite", type=int, default=None,
        help="Limite global para una prueba tecnica rapida",
    )
    parser.add_argument(
        "--muestra-por-categoria-split", type=int, default=None,
        help=(
            "Seleccion determinista de N modelos por cada categoria y split; "
            "recomendado para la validacion controlada del objetivo 1"
        ),
    )
    parser.add_argument("--sobrescribir", action="store_true")
    return parser


def main() -> int:
    args = construir_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    resultados_dir = args.resultados_dir.resolve()
    manifest_root = output_root / "manifests"

    for resolucion in args.resoluciones:
        _validar_resolucion(resolucion)
    if args.n_puntos <= 0 or args.procesos <= 0:
        raise ValueError("--n-puntos y --procesos deben ser positivos")
    if args.limite is not None and args.muestra_por_categoria_split is not None:
        raise ValueError(
            "--limite y --muestra-por-categoria-split son mutuamente excluyentes"
        )
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"No se encontro ModelNet40 en {dataset_root}. "
            "Indique la ubicacion con --dataset-root."
        )

    modelos_encontrados = recolectar_modelos(dataset_root, args.splits)
    conteos_encontrados = {
        split: sum(modelo["split"] == split for modelo in modelos_encontrados)
        for split in args.splits
    }
    modelos = modelos_encontrados
    if args.categorias is not None:
        categorias = set(args.categorias)
        modelos = [m for m in modelos if m["categoria"] in categorias]
    if args.muestra_por_categoria_split is not None:
        modelos = seleccionar_muestra_controlada(
            modelos, args.muestra_por_categoria_split,
        )
    if args.limite is not None:
        modelos = modelos[:args.limite]
    if not modelos:
        raise RuntimeError("No se encontraron archivos .off en la estructura esperada")

    print("=" * 72)
    print("OBJETIVO ESPECIFICO 1 — GENERACION VALIDADA DE OCTREES")
    print(f"Dataset      : {dataset_root}")
    print(f"Modelos      : {len(modelos)}")
    print(f"Resoluciones : {args.resoluciones}")
    print(f"Formato      : {OCTREE_FORMAT_VERSION}")
    print(f"Procesos     : {args.procesos}")
    print("=" * 72)

    tareas = []
    for modelo in modelos:
        tareas.append({
            **modelo,
            "output_root": str(output_root),
            "manifest_root": str(manifest_root),
            "resoluciones": list(args.resoluciones),
            "n_puntos": args.n_puntos,
            "semilla_base": args.semilla,
            "sobrescribir": args.sobrescribir,
        })

    resultados = []
    errores = []
    inicio = time.perf_counter()

    if args.procesos == 1:
        iterador = tqdm(tareas, total=len(tareas), desc="Modelos", ncols=88)
        for tarea in iterador:
            try:
                resultados.append(procesar_modelo(tarea))
            except Exception as exc:  # continuar y consolidar todos los fallos
                errores.append({
                    "model_id": tarea["model_id"],
                    "categoria": tarea["categoria"],
                    "split": tarea["split"],
                    "error": f"{type(exc).__name__}: {exc}",
                })
    else:
        with ProcessPoolExecutor(max_workers=args.procesos) as executor:
            futuros = {executor.submit(procesar_modelo, tarea): tarea for tarea in tareas}
            for futuro in tqdm(
                as_completed(futuros), total=len(futuros), desc="Modelos", ncols=88,
            ):
                tarea = futuros[futuro]
                try:
                    resultados.append(futuro.result())
                except Exception as exc:
                    errores.append({
                        "model_id": tarea["model_id"],
                        "categoria": tarea["categoria"],
                        "split": tarea["split"],
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    duracion_s = time.perf_counter() - inicio
    filas = [fila for resultado in resultados for fila in resultado["filas"]]
    if args.muestra_por_categoria_split is not None:
        nombre_metricas = "metricas_muestra_controlada.csv"
    elif args.limite is not None or args.categorias is not None:
        nombre_metricas = "metricas_prueba_parcial.csv"
    else:
        nombre_metricas = "metricas_modelnet40.csv"
    _escribir_csv_atomico(filas, resultados_dir / nombre_metricas)

    corrida_completa = (
        args.limite is None
        and args.muestra_por_categoria_split is None
        and args.categorias is None
        and set(args.splits) == {"train", "test"}
        and set(args.resoluciones) == {32, 64}
        and conteos_encontrados == CONTEOS_OFICIALES
        and len(resultados) == sum(CONTEOS_OFICIALES.values())
        and not errores
    )
    if args.muestra_por_categoria_split is not None:
        alcance = "muestra_controlada"
    elif args.limite is not None or args.categorias is not None:
        alcance = "prueba_parcial"
    else:
        alcance = "modelnet40_completo" if corrida_completa else "incompleto"
    resumen = {
        "octree_format_version": OCTREE_FORMAT_VERSION,
        "alcance_ejecucion": alcance,
        "dataset_root": ruta_portable(dataset_root, RAIZ_PROYECTO),
        "output_root": ruta_portable(output_root, RAIZ_PROYECTO),
        "resoluciones": args.resoluciones,
        "n_puntos_muestreo": args.n_puntos,
        "semilla_base": args.semilla,
        "categorias_solicitadas": args.categorias or CLASES_MODELNET40,
        "muestra_por_categoria_split": args.muestra_por_categoria_split,
        "limite_global": args.limite,
        "conteos_encontrados_dataset": conteos_encontrados,
        "archivo_metricas": nombre_metricas,
        "n_modelos_solicitados": len(modelos),
        "n_modelos_exitosos": len(resultados),
        "n_modelos_fallidos": len(errores),
        "duracion_total_s": duracion_s,
        "corrida_completa_modelnet40": corrida_completa,
        "errores": errores,
    }
    guardar_json_atomico(resumen, resultados_dir / "resumen_ejecucion.json")

    print(f"Exitosos : {len(resultados)}")
    print(f"Fallidos : {len(errores)}")
    print(f"Alcance  : {alcance.replace('_', ' ').upper()}")
    print(f"Resumen  : {resultados_dir / 'resumen_ejecucion.json'}")
    return 0 if not errores else 1


if __name__ == "__main__":
    raise SystemExit(main())
