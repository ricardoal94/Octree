"""Barrido reproducible de ``num_workers`` para el smoke test Net5-OctNet.

Ejecuta las combinaciones R32/R64 con 0, 4 y 8 trabajadores, valida cada
resumen y consolida las metricas en JSON y CSV. Los artefactos se escriben,
por defecto, fuera del repositorio para que ``--exigir-git-limpio`` siga
siendo valido durante las seis corridas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
SCRIPT_ENTRENAMIENTO = RAIZ_PROYECTO / "fase3_net5" / "fase3_net5_entrenamiento.py"
PREFIJO_MODELO = "net5_octree"
SCHEMA_RESUMEN = "net5-octree-native"
SCHEMA_RESUMEN_VERSION = "1.2.0"
SCHEMA_BARRIDO = "net5-octree-worker-benchmark"
SCHEMA_BARRIDO_VERSION = "1.0.0"


class ErrorValidacionSmoke(ValueError):
    """Indica que una corrida no cumple el contrato del barrido."""


def _valor_numerico_positivo(valor) -> bool:
    return (
        isinstance(valor, (int, float))
        and not isinstance(valor, bool)
        and math.isfinite(float(valor))
        and float(valor) > 0.0
    )


def _agregar_error(errores: list[str], condicion: bool, mensaje: str) -> None:
    if not condicion:
        errores.append(mensaje)


def ruta_resumen(resultados_dir: Path, resolucion: int, workers: int) -> Path:
    """Devuelve la ruta determinista del resumen para una combinacion."""

    return (
        resultados_dir
        / f"resumen_{PREFIJO_MODELO}_R{resolucion}_smoke_w{workers}.json"
    )


def construir_comando(
    *,
    python: str,
    resolucion: int,
    workers: int,
    limite_train: int,
    limite_val: int,
    limite_test: int,
    epochs: int,
    batch_size: int,
    lotes_perfil: int,
    data_root: Path,
    particion_manifest: Path,
    checkpoints_dir: Path,
    logs_dir: Path,
    resultados_dir: Path,
) -> list[str]:
    """Construye el comando oficial de una corrida del barrido."""

    return [
        python,
        str(SCRIPT_ENTRENAMIENTO),
        "--resolucion", str(resolucion),
        "--backend", "octree_native",
        "--tag", f"_smoke_w{workers}",
        "--limite_train", str(limite_train),
        "--limite_val", str(limite_val),
        "--limite_test", str(limite_test),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--num-workers", str(workers),
        "--lotes-perfil", str(lotes_perfil),
        "--data-root", str(data_root),
        "--particion-manifest", str(particion_manifest),
        "--checkpoints-dir", str(checkpoints_dir),
        "--logs-dir", str(logs_dir),
        "--resultados-dir", str(resultados_dir),
        "--exigir-git-limpio",
        "--exigir-git-publicado",
    ]


def cargar_resumen(ruta: Path) -> dict:
    try:
        with ruta.open("r", encoding="utf-8") as archivo:
            resumen = json.load(archivo)
    except FileNotFoundError as exc:
        raise ErrorValidacionSmoke(f"No existe el resumen esperado: {ruta}") from exc
    except json.JSONDecodeError as exc:
        raise ErrorValidacionSmoke(f"JSON invalido en {ruta}: {exc}") from exc
    if not isinstance(resumen, dict):
        raise ErrorValidacionSmoke(f"El resumen {ruta} no contiene un objeto JSON")
    return resumen


def validar_resumen(
    resumen: dict,
    *,
    resolucion: int,
    workers: int,
    limite_train: int,
    limite_val: int,
    limite_test: int,
    epochs: int,
    batch_size: int,
    lotes_perfil: int,
) -> None:
    """Valida trazabilidad, configuracion y perfil de una corrida."""

    errores: list[str] = []
    configuracion = resumen.get("configuracion") or {}
    particion = resumen.get("particion") or {}
    perfil = resumen.get("perfil_rendimiento") or {}
    detalle = perfil.get("detalle_backend") or {}

    _agregar_error(
        errores, resumen.get("schema_name") == SCHEMA_RESUMEN,
        f"schema_name debe ser {SCHEMA_RESUMEN!r}",
    )
    _agregar_error(
        errores, resumen.get("schema_version") == SCHEMA_RESUMEN_VERSION,
        f"schema_version debe ser {SCHEMA_RESUMEN_VERSION}",
    )
    _agregar_error(
        errores, resumen.get("backend") == "octree_native",
        "backend debe ser octree_native",
    )
    _agregar_error(
        errores, resumen.get("alcance") == "PARCIAL_SMOKE",
        "alcance debe ser PARCIAL_SMOKE",
    )
    _agregar_error(
        errores, resumen.get("valido_como_resultado_objetivo3") is False,
        "un smoke limitado no debe marcarse como resultado final del Objetivo 3",
    )
    _agregar_error(
        errores, resumen.get("resolucion") == resolucion,
        f"resolucion debe ser {resolucion}",
    )
    _agregar_error(
        errores, resumen.get("git_dirty") is False,
        "git_dirty debe ser false",
    )
    _agregar_error(
        errores, resumen.get("git_commit_publicado") is True,
        "git_commit_publicado debe ser true",
    )
    _agregar_error(
        errores, isinstance(resumen.get("git_commit"), str)
        and len(resumen["git_commit"]) >= 7,
        "git_commit debe estar registrado",
    )

    esperados_config = {
        "epochs_max": epochs,
        "batch_size": batch_size,
        "num_workers": workers,
        "lotes_perfil": lotes_perfil,
        "exigir_git_limpio": True,
        "exigir_git_publicado": True,
        "precalcular_planes": True,
    }
    for clave, esperado in esperados_config.items():
        _agregar_error(
            errores, configuracion.get(clave) == esperado,
            f"configuracion.{clave} debe ser {esperado!r}",
        )
    prefetch_esperado = 1 if workers > 0 else None
    _agregar_error(
        errores, configuracion.get("prefetch_factor") == prefetch_esperado,
        f"configuracion.prefetch_factor debe ser {prefetch_esperado!r}",
    )

    for clave, esperado in (
        ("n_train_usado", limite_train),
        ("n_val_usado", limite_val),
        ("n_test_usado", limite_test),
    ):
        _agregar_error(
            errores, particion.get(clave) == esperado,
            f"particion.{clave} debe ser {esperado}",
        )

    for clave in (
        "tiempo_inferencia_promedio_ms",
        "tiempo_pipeline_promedio_ms",
        "vram_pico_mb",
        "tamano_modelo_mb",
    ):
        _agregar_error(
            errores, _valor_numerico_positivo(resumen.get(clave)),
            f"{clave} debe ser finito y mayor que cero",
        )
    if (
        _valor_numerico_positivo(resumen.get("tiempo_inferencia_promedio_ms"))
        and _valor_numerico_positivo(resumen.get("tiempo_pipeline_promedio_ms"))
    ):
        _agregar_error(
            errores,
            float(resumen.get("tiempo_pipeline_promedio_ms", 0.0))
            >= float(resumen["tiempo_inferencia_promedio_ms"]),
            "el tiempo integral no puede ser menor que el forward",
        )

    for clave in (
        "carga_lote_ms_por_muestra",
        "forward_total_ms_por_muestra",
        "planes_backend_ms_por_muestra",
        "resto_forward_ms_por_muestra",
        "memoria_planes_cpu_mb_por_muestra",
        "interacciones_convolucion_por_muestra",
    ):
        _agregar_error(
            errores, _valor_numerico_positivo(perfil.get(clave)),
            f"perfil_rendimiento.{clave} debe ser finito y mayor que cero",
        )
    _agregar_error(
        errores, perfil.get("lotes_medidos") == lotes_perfil,
        f"perfil_rendimiento.lotes_medidos debe ser {lotes_perfil}",
    )
    _agregar_error(
        errores, perfil.get("muestras_medidas") == lotes_perfil * batch_size,
        "perfil_rendimiento.muestras_medidas no coincide con lotes y batch",
    )
    for clave in (
        "n_planes_convolucion",
        "n_interacciones_convolucion",
        "bytes_planes_convolucion",
        "n_planes_pooling",
        "n_asignaciones_pooling",
        "bytes_planes_pooling",
        "n_mapas_finales",
        "bytes_mapas_finales",
    ):
        _agregar_error(
            errores, _valor_numerico_positivo(detalle.get(clave)),
            f"perfil_rendimiento.detalle_backend.{clave} debe ser mayor que cero",
        )

    if errores:
        combinacion = f"R{resolucion}, workers={workers}"
        detalle_errores = "\n".join(f"- {error}" for error in errores)
        raise ErrorValidacionSmoke(
            f"La corrida {combinacion} no cumple el contrato:\n{detalle_errores}"
        )


def validar_conjunto(
    ejecuciones: Iterable[dict],
    *,
    resoluciones: Iterable[int],
    workers: Iterable[int],
) -> None:
    """Comprueba cobertura y equivalencia experimental de todo el barrido."""

    ejecuciones = list(ejecuciones)
    resoluciones = tuple(resoluciones)
    workers = tuple(workers)
    combinaciones_esperadas = {
        (int(resolucion), int(n_workers))
        for resolucion in resoluciones
        for n_workers in workers
    }
    combinaciones = {
        (
            int(resumen["resolucion"]),
            int(resumen["configuracion"]["num_workers"]),
        )
        for resumen in ejecuciones
    }
    errores = []
    _agregar_error(
        errores, len(ejecuciones) == len(combinaciones_esperadas),
        "el numero de ejecuciones no coincide con el barrido solicitado",
    )
    _agregar_error(
        errores, combinaciones == combinaciones_esperadas,
        "faltan combinaciones o existen combinaciones duplicadas",
    )

    campos_comunes = {
        "commit": {resumen.get("git_commit") for resumen in ejecuciones},
        "branch": {resumen.get("git_branch") for resumen in ejecuciones},
        "seed": {resumen.get("seed") for resumen in ejecuciones},
        "manifest": {
            (resumen.get("particion") or {}).get("manifest")
            for resumen in ejecuciones
        },
        "metodo_particion": {
            (resumen.get("particion") or {}).get("metodo")
            for resumen in ejecuciones
        },
        "parametros_entrenables": {
            resumen.get("parametros_entrenables") for resumen in ejecuciones
        },
        "backend_version": {
            resumen.get("backend_version") for resumen in ejecuciones
        },
    }
    for campo, valores in campos_comunes.items():
        _agregar_error(
            errores, len(valores) == 1 and None not in valores,
            f"todas las corridas deben compartir {campo}",
        )

    if errores:
        raise ErrorValidacionSmoke(
            "El conjunto del barrido no es comparable:\n"
            + "\n".join(f"- {error}" for error in errores)
        )


COLUMNAS_CSV = [
    "resolucion",
    "num_workers",
    "git_commit",
    "git_branch",
    "n_train_usado",
    "n_val_usado",
    "n_test_usado",
    "tiempo_total_min",
    "forward_ms_por_muestra",
    "pipeline_ms_por_muestra",
    "carga_lote_ms_por_muestra",
    "transferencia_atributos_ms_por_muestra",
    "planes_backend_ms_por_muestra",
    "resto_forward_ms_por_muestra",
    "memoria_planes_cpu_mb_por_muestra",
    "interacciones_convolucion_por_muestra",
    "vram_pico_mb",
    "tamano_modelo_mb",
    "test_acc_smoke",
]


def fila_comparacion(resumen: dict) -> dict:
    configuracion = resumen["configuracion"]
    particion = resumen["particion"]
    perfil = resumen["perfil_rendimiento"]
    return {
        "resolucion": resumen["resolucion"],
        "num_workers": configuracion["num_workers"],
        "git_commit": resumen["git_commit"],
        "git_branch": resumen["git_branch"],
        "n_train_usado": particion["n_train_usado"],
        "n_val_usado": particion["n_val_usado"],
        "n_test_usado": particion["n_test_usado"],
        "tiempo_total_min": resumen["tiempo_total_min"],
        "forward_ms_por_muestra": resumen["tiempo_inferencia_promedio_ms"],
        "pipeline_ms_por_muestra": resumen["tiempo_pipeline_promedio_ms"],
        "carga_lote_ms_por_muestra": perfil["carga_lote_ms_por_muestra"],
        "transferencia_atributos_ms_por_muestra": perfil[
            "transferencia_atributos_ms_por_muestra"
        ],
        "planes_backend_ms_por_muestra": perfil[
            "planes_backend_ms_por_muestra"
        ],
        "resto_forward_ms_por_muestra": perfil[
            "resto_forward_ms_por_muestra"
        ],
        "memoria_planes_cpu_mb_por_muestra": perfil[
            "memoria_planes_cpu_mb_por_muestra"
        ],
        "interacciones_convolucion_por_muestra": perfil[
            "interacciones_convolucion_por_muestra"
        ],
        "vram_pico_mb": resumen["vram_pico_mb"],
        "tamano_modelo_mb": resumen["tamano_modelo_mb"],
        "test_acc_smoke": resumen["test_acc"],
    }


def escribir_consolidado(
    ejecuciones: Iterable[dict],
    *,
    salida_dir: Path,
    resoluciones: Iterable[int],
    workers: Iterable[int],
) -> tuple[Path, Path]:
    """Escribe el informe JSON y la tabla CSV del barrido validado."""

    resoluciones = tuple(resoluciones)
    workers = tuple(workers)
    filas = sorted(
        (fila_comparacion(resumen) for resumen in ejecuciones),
        key=lambda fila: (fila["resolucion"], fila["num_workers"]),
    )
    salida_dir.mkdir(parents=True, exist_ok=True)
    ruta_json = salida_dir / "comparacion_workers_octnet.json"
    ruta_csv = salida_dir / "comparacion_workers_octnet.csv"
    temporal_json = ruta_json.with_suffix(".json.tmp")
    temporal_csv = ruta_csv.with_suffix(".csv.tmp")

    consolidado = {
        "schema_name": SCHEMA_BARRIDO,
        "schema_version": SCHEMA_BARRIDO_VERSION,
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "estado": "VALIDADO",
        "resoluciones": list(resoluciones),
        "num_workers": list(workers),
        "n_ejecuciones": len(filas),
        "git_commit": filas[0]["git_commit"],
        "git_branch": filas[0]["git_branch"],
        "criterio": (
            "Seleccionar por tiempo de pipeline; forward, memoria de planes y "
            "VRAM se conservan como diagnosticos y restricciones."
        ),
        "seleccion_workers": {
            str(resolucion): min(
                (
                    fila for fila in filas
                    if fila["resolucion"] == resolucion
                ),
                key=lambda fila: fila["pipeline_ms_por_muestra"],
            )["num_workers"]
            for resolucion in resoluciones
        },
        "resultados": filas,
    }
    with temporal_json.open("w", encoding="utf-8", newline="\n") as archivo:
        json.dump(consolidado, archivo, indent=2, ensure_ascii=False)
        archivo.write("\n")
    os.replace(temporal_json, ruta_json)

    with temporal_csv.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_CSV)
        escritor.writeheader()
        escritor.writerows(filas)
    os.replace(temporal_csv, ruta_csv)
    return ruta_json, ruta_csv


def _mostrar_comando(comando: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(comando)
    return shlex.join(comando)


def resolver_particion_manifest(
    ruta_solicitada: Path | None,
    logs_dir: Path,
) -> Path:
    """Mantiene el manifiesto fuera del repositorio salvo ruta explicita."""

    if ruta_solicitada is not None:
        return ruta_solicitada.resolve()
    return (logs_dir / "particion_objetivo2_modelos.json").resolve()


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta y valida el barrido de num_workers para Net5-OctNet."
        )
    )
    parser.add_argument("--resoluciones", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 4, 8])
    parser.add_argument("--limite-train", type=int, default=40)
    parser.add_argument("--limite-val", type=int, default=40)
    parser.add_argument("--limite-test", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lotes-perfil", type=int, default=3)
    parser.add_argument("--data-root", type=Path, default=RAIZ_PROYECTO / "data")
    parser.add_argument(
        "--particion-manifest", type=Path,
        default=None,
        help=(
            "Ruta del manifiesto; por defecto se guarda con los artefactos "
            "externos para conservar limpio el repositorio"
        ),
    )
    parser.add_argument(
        "--artefactos-dir", type=Path,
        default=RAIZ_PROYECTO.parent / f"{RAIZ_PROYECTO.name}_smoke_workers",
        help="Directorio externo para checkpoints, logs y resultados",
    )
    parser.add_argument(
        "--solo-validar", action="store_true",
        help="No ejecuta entrenamientos; valida resultados ya existentes",
    )
    parser.add_argument(
        "--continuar", dest="continuar_existentes", action="store_true",
        help="Reutiliza una combinacion existente solo si supera la validacion",
    )
    parser.add_argument(
        "--mostrar-comandos", action="store_true",
        help="Imprime los comandos sin ejecutarlos ni validar resultados",
    )
    return parser.parse_args()


def _validar_argumentos(args: argparse.Namespace) -> None:
    if not args.resoluciones or set(args.resoluciones) - {32, 64}:
        raise ValueError("--resoluciones solo admite 32 y 64")
    if len(set(args.resoluciones)) != len(args.resoluciones):
        raise ValueError("--resoluciones no admite valores duplicados")
    if not args.workers or any(valor < 0 for valor in args.workers):
        raise ValueError("--workers requiere enteros no negativos")
    if len(set(args.workers)) != len(args.workers):
        raise ValueError("--workers no admite valores duplicados")
    for nombre in (
        "limite_train", "limite_val", "limite_test", "epochs",
        "batch_size", "lotes_perfil",
    ):
        if getattr(args, nombre) < 1:
            raise ValueError(f"--{nombre.replace('_', '-')} debe ser positivo")
    if args.solo_validar and args.mostrar_comandos:
        raise ValueError("--solo-validar y --mostrar-comandos son excluyentes")


def main() -> int:
    args = _argumentos()
    _validar_argumentos(args)
    artefactos_dir = args.artefactos_dir.resolve()
    checkpoints_dir = artefactos_dir / "checkpoints"
    logs_dir = artefactos_dir / "logs"
    resultados_dir = artefactos_dir / "resultados"
    particion_manifest = resolver_particion_manifest(
        args.particion_manifest, logs_dir,
    )
    comandos = []
    for resolucion in args.resoluciones:
        for workers in args.workers:
            comandos.append((
                resolucion,
                workers,
                construir_comando(
                    python=sys.executable,
                    resolucion=resolucion,
                    workers=workers,
                    limite_train=args.limite_train,
                    limite_val=args.limite_val,
                    limite_test=args.limite_test,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lotes_perfil=args.lotes_perfil,
                    data_root=args.data_root.resolve(),
                    particion_manifest=particion_manifest,
                    checkpoints_dir=checkpoints_dir,
                    logs_dir=logs_dir,
                    resultados_dir=resultados_dir,
                ),
            ))

    if args.mostrar_comandos:
        for _, _, comando in comandos:
            print(_mostrar_comando(comando))
        return 0

    for directorio in (checkpoints_dir, logs_dir, resultados_dir):
        directorio.mkdir(parents=True, exist_ok=True)

    ejecuciones = []
    for indice, (resolucion, workers, comando) in enumerate(comandos, start=1):
        ruta = ruta_resumen(resultados_dir, resolucion, workers)
        reutilizado = False
        if args.continuar_existentes and ruta.exists():
            try:
                resumen_existente = cargar_resumen(ruta)
                validar_resumen(
                    resumen_existente,
                    resolucion=resolucion,
                    workers=workers,
                    limite_train=args.limite_train,
                    limite_val=args.limite_val,
                    limite_test=args.limite_test,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lotes_perfil=args.lotes_perfil,
                )
                reutilizado = True
                print(f"[{indice}/{len(comandos)}] Reutilizando {ruta.name}")
            except ErrorValidacionSmoke as exc:
                print(f"[Aviso] Se repetira {ruta.name}: {exc}")

        if not args.solo_validar and not reutilizado:
            print(
                f"[{indice}/{len(comandos)}] R{resolucion}, "
                f"num_workers={workers}"
            )
            print(_mostrar_comando(comando))
            subprocess.run(comando, cwd=RAIZ_PROYECTO, check=True)

        resumen = cargar_resumen(ruta)
        validar_resumen(
            resumen,
            resolucion=resolucion,
            workers=workers,
            limite_train=args.limite_train,
            limite_val=args.limite_val,
            limite_test=args.limite_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lotes_perfil=args.lotes_perfil,
        )
        ejecuciones.append(resumen)

    validar_conjunto(
        ejecuciones, resoluciones=args.resoluciones, workers=args.workers,
    )
    ruta_json, ruta_csv = escribir_consolidado(
        ejecuciones,
        salida_dir=resultados_dir,
        resoluciones=args.resoluciones,
        workers=args.workers,
    )
    print("Barrido validado correctamente.")
    print(f"JSON: {ruta_json}")
    print(f"CSV : {ruta_csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ErrorValidacionSmoke, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
