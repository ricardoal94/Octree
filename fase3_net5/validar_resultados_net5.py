"""Valida y consolida los resultados oficiales de Net5-OctNet.

El cierre del Objetivo 3 requiere exactamente dos corridas nativas completas,
R=32 y R=64, sobre las mismas particiones, el mismo commit y el mismo entorno.
Este modulo no importa PyTorch, por lo que tambien puede ejecutarse en CI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from particion_objetivo2 import CLASES_MODELNET40

SCHEMA_RESUMEN = "net5-octree-native"
SCHEMA_RESUMEN_VERSION = "1.4.0"
SCHEMA_VALIDACION = "net5-objective3-validation"
SCHEMA_VALIDACION_VERSION = "1.0.0"
N_TRAIN_TOTAL = 9843
N_TRAIN = 8858
N_VAL = 985
N_TEST = 2468
RESOLUCIONES = (32, 64)
VARIABLES_HILOS_CPU = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _constante_json_invalida(valor: str):
    raise ValueError(f"Constante JSON no valida: {valor}")


def cargar_json_estricto(ruta: Path) -> dict:
    try:
        with ruta.open("r", encoding="utf-8") as archivo:
            contenido = json.load(
                archivo, parse_constant=_constante_json_invalida,
            )
    except FileNotFoundError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"JSON invalido en {ruta}: {exc}") from exc
    if not isinstance(contenido, dict):
        raise TypeError(f"{ruta} no contiene un objeto JSON")
    return contenido


def _positivo(valor: object) -> bool:
    return (
        isinstance(valor, (int, float))
        and not isinstance(valor, bool)
        and math.isfinite(float(valor))
        and float(valor) > 0
    )


def _probabilidad(valor: object) -> bool:
    return (
        isinstance(valor, (int, float))
        and not isinstance(valor, bool)
        and math.isfinite(float(valor))
        and 0 <= float(valor) <= 1
    )


def _sha256_valido(valor: object) -> bool:
    return (
        isinstance(valor, str)
        and len(valor) == 64
        and all(caracter in "0123456789abcdef" for caracter in valor)
    )


def _agregar(errores: list[str], condicion: bool, mensaje: str) -> None:
    if not condicion:
        errores.append(mensaje)


def _validar_entorno(entorno: object, errores: list[str]) -> None:
    _agregar(errores, isinstance(entorno, dict), "entorno_ejecucion no es objeto")
    if not isinstance(entorno, dict):
        return
    for seccion in ("sistema_operativo", "python", "pytorch", "hardware"):
        _agregar(
            errores, isinstance(entorno.get(seccion), dict),
            f"entorno_ejecucion.{seccion} no es objeto",
        )
    hardware = entorno.get("hardware") or {}
    pytorch = entorno.get("pytorch") or {}
    if not isinstance(hardware, dict) or not isinstance(pytorch, dict):
        return
    _agregar(
        errores, hardware.get("dispositivo") == "cuda",
        "la corrida oficial debe ejecutarse en CUDA",
    )
    gpu = hardware.get("gpu")
    _agregar(errores, isinstance(gpu, dict), "hardware.gpu no esta registrado")
    if isinstance(gpu, dict):
        _agregar(
            errores, isinstance(gpu.get("nombre"), str) and gpu["nombre"].strip(),
            "hardware.gpu.nombre no esta registrado",
        )
        _agregar(
            errores, _positivo(gpu.get("vram_total_gib")),
            "hardware.gpu.vram_total_gib debe ser positivo",
        )
    _agregar(
        errores,
        isinstance(pytorch.get("version"), str) and pytorch["version"].strip(),
        "pytorch.version no esta registrado",
    )
    _agregar(
        errores,
        isinstance(pytorch.get("cuda_compilacion"), str)
        and pytorch["cuda_compilacion"].strip(),
        "pytorch.cuda_compilacion no esta registrado",
    )


def _validar_matriz_y_reporte(resumen: dict, errores: list[str]) -> None:
    matriz = resumen.get("matriz_confusion")
    es_matriz = (
        isinstance(matriz, list)
        and len(matriz) == len(CLASES_MODELNET40)
        and all(
            isinstance(fila, list) and len(fila) == len(CLASES_MODELNET40)
            for fila in matriz
        )
    )
    _agregar(errores, es_matriz, "matriz_confusion debe tener forma 40x40")
    if not es_matriz:
        return
    valores = [valor for fila in matriz for valor in fila]
    _agregar(
        errores,
        all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in valores),
        "matriz_confusion debe contener enteros no negativos",
    )
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in valores):
        return
    total = sum(valores)
    _agregar(errores, total == N_TEST, f"matriz_confusion suma {total}, no {N_TEST}")
    exactitud_matriz = sum(matriz[i][i] for i in range(len(matriz))) / N_TEST
    if _probabilidad(resumen.get("test_acc")):
        _agregar(
            errores, abs(exactitud_matriz - float(resumen["test_acc"])) <= 1e-6,
            "test_acc no coincide con la matriz_confusion",
        )

    reporte = resumen.get("reporte_clasificacion")
    _agregar(
        errores, isinstance(reporte, dict),
        "reporte_clasificacion debe ser un objeto",
    )
    if not isinstance(reporte, dict):
        return
    for indice, clase in enumerate(CLASES_MODELNET40):
        metricas = reporte.get(clase)
        _agregar(
            errores, isinstance(metricas, dict),
            f"reporte_clasificacion no contiene {clase}",
        )
        if isinstance(metricas, dict):
            soporte = metricas.get("support")
            _agregar(
                errores,
                isinstance(soporte, (int, float))
                and math.isfinite(float(soporte))
                and int(soporte) == sum(matriz[indice]),
                f"support de {clase} no coincide con la matriz",
            )
            for metrica in ("precision", "recall", "f1-score"):
                _agregar(
                    errores, _probabilidad(metricas.get(metrica)),
                    f"{metrica} de {clase} no es una probabilidad",
                )
    _agregar(
        errores,
        _probabilidad(reporte.get("accuracy"))
        and abs(float(reporte["accuracy"]) - float(resumen.get("test_acc", -1)))
        <= 1e-6,
        "accuracy del reporte no coincide con test_acc",
    )


def validar_resumen_oficial(
    resumen: dict,
    resolucion: int,
    *,
    workers: int,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    lotes_perfil: int,
) -> list[str]:
    errores: list[str] = []
    particion = resumen.get("particion") or {}
    configuracion = resumen.get("configuracion") or {}
    entrenamiento = resumen.get("entrenamiento") or {}
    perfil = resumen.get("perfil_rendimiento") or {}

    _agregar(errores, resumen.get("schema_name") == SCHEMA_RESUMEN, "schema_name invalido")
    _agregar(
        errores, resumen.get("schema_version") == SCHEMA_RESUMEN_VERSION,
        f"schema_version debe ser {SCHEMA_RESUMEN_VERSION}",
    )
    _agregar(errores, resumen.get("backend") == "octree_native", "backend no es nativo")
    _agregar(errores, resumen.get("alcance") == "COMPLETO", "alcance no es COMPLETO")
    _agregar(
        errores, resumen.get("valido_como_resultado_objetivo3") is True,
        "valido_como_resultado_objetivo3 debe ser true",
    )
    _agregar(errores, resumen.get("motivo_no_valido") is None, "motivo_no_valido debe ser null")
    _agregar(errores, resumen.get("resolucion") == resolucion, "resolucion incorrecta")
    _agregar(
        errores, resumen.get("profundidad_octree") == (5 if resolucion == 32 else 6),
        "profundidad_octree incorrecta",
    )
    _agregar(errores, resumen.get("seed") == 42, "seed debe ser 42")
    _agregar(errores, resumen.get("git_dirty") is False, "git_dirty debe ser false")
    _agregar(
        errores, resumen.get("git_commit_publicado") is True,
        "git_commit_publicado debe ser true",
    )
    _agregar(
        errores,
        isinstance(resumen.get("git_commit"), str)
        and len(resumen["git_commit"]) == 40,
        "git_commit debe contener el SHA completo",
    )
    _agregar(errores, _positivo(resumen.get("parametros_entrenables")), "parametros_entrenables invalido")
    _validar_entorno(resumen.get("entorno_ejecucion"), errores)

    esperados_particion = {
        "n_train_total": N_TRAIN_TOTAL,
        "n_train_usado": N_TRAIN,
        "n_val_usado": N_VAL,
        "n_test_usado": N_TEST,
    }
    for clave, esperado in esperados_particion.items():
        _agregar(
            errores, particion.get(clave) == esperado,
            f"particion.{clave} debe ser {esperado}",
        )
    _agregar(
        errores, isinstance(particion.get("manifest"), str),
        "particion.manifest no esta registrado",
    )

    esperados_config = {
        "epochs_max": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "num_workers": workers,
        "lr": lr,
        "weight_decay": 1e-4,
        "scheduler": "StepLR(step_size=20, gamma=0.7)",
        "lotes_perfil": lotes_perfil,
        "exigir_git_limpio": True,
        "exigir_git_publicado": True,
        "precalcular_planes": True,
        "prefetch_factor": 1 if workers > 0 else None,
    }
    for clave, esperado in esperados_config.items():
        _agregar(
            errores, configuracion.get(clave) == esperado,
            f"configuracion.{clave} debe ser {esperado!r}",
        )
    _agregar(
        errores,
        configuracion.get("hilos_bibliotecas_cpu")
        == {variable: "1" for variable in VARIABLES_HILOS_CPU},
        "hilos_bibliotecas_cpu no aplica la politica de un hilo",
    )
    _agregar(
        errores,
        configuracion.get("persistent_workers")
        == {"train": False, "val": False, "test": workers > 0},
        "persistent_workers no coincide con el protocolo",
    )

    epocas_completadas = entrenamiento.get("epocas_completadas")
    _agregar(
        errores,
        isinstance(epocas_completadas, int)
        and not isinstance(epocas_completadas, bool)
        and 1 <= epocas_completadas <= epochs,
        "entrenamiento.epocas_completadas esta fuera de rango",
    )
    if isinstance(epocas_completadas, int) and epocas_completadas < epochs:
        _agregar(
            errores, entrenamiento.get("detenido_por_early_stopping") is True,
            "una corrida corta debe haber terminado por early stopping",
        )
    _agregar(
        errores, _sha256_valido(entrenamiento.get("particion_sha256")),
        "entrenamiento.particion_sha256 no es valido",
    )
    _agregar(
        errores,
        isinstance(entrenamiento.get("historial_json"), str)
        and isinstance(entrenamiento.get("historial_csv"), str),
        "los historiales no estan registrados",
    )

    for clave in ("mejor_val_acc", "test_acc"):
        _agregar(errores, _probabilidad(resumen.get(clave)), f"{clave} no es probabilidad")
    _agregar(
        errores,
        isinstance(resumen.get("mejor_epoca"), int)
        and isinstance(epocas_completadas, int)
        and 1 <= resumen["mejor_epoca"] <= epocas_completadas,
        "mejor_epoca esta fuera del historial",
    )
    _agregar(
        errores,
        isinstance(resumen.get("test_loss"), (int, float))
        and math.isfinite(float(resumen["test_loss"]))
        and float(resumen["test_loss"]) >= 0,
        "test_loss no es finito o es negativo",
    )
    for clave in (
        "tiempo_total_min", "tamano_modelo_mb", "vram_pico_mb",
        "tiempo_inferencia_total_s", "tiempo_inferencia_promedio_ms",
        "tiempo_pipeline_total_s", "tiempo_pipeline_promedio_ms",
    ):
        _agregar(errores, _positivo(resumen.get(clave)), f"{clave} debe ser positivo")
    _agregar(
        errores, resumen.get("n_muestras_test") == N_TEST,
        f"n_muestras_test debe ser {N_TEST}",
    )
    _agregar(
        errores, resumen.get("repeticiones_inferencia") == 3,
        "repeticiones_inferencia debe ser 3",
    )
    if _positivo(resumen.get("tiempo_inferencia_promedio_ms")) and _positivo(
        resumen.get("tiempo_pipeline_promedio_ms")
    ):
        _agregar(
            errores,
            resumen["tiempo_pipeline_promedio_ms"]
            >= resumen["tiempo_inferencia_promedio_ms"],
            "el tiempo de pipeline no puede ser menor que el forward",
        )
    for clave in (
        "carga_lote_ms_por_muestra", "forward_total_ms_por_muestra",
        "planes_backend_ms_por_muestra", "resto_forward_ms_por_muestra",
        "memoria_planes_cpu_mb_por_muestra",
        "interacciones_convolucion_por_muestra",
    ):
        _agregar(
            errores, _positivo(perfil.get(clave)),
            f"perfil_rendimiento.{clave} debe ser positivo",
        )
    _agregar(
        errores, perfil.get("lotes_medidos") == lotes_perfil,
        f"perfil_rendimiento.lotes_medidos debe ser {lotes_perfil}",
    )
    _validar_matriz_y_reporte(resumen, errores)
    return errores


def _campos_comunes(resumen: dict) -> dict:
    return {
        "git_commit": resumen.get("git_commit"),
        "git_branch": resumen.get("git_branch"),
        "backend_version": resumen.get("backend_version"),
        "seed": resumen.get("seed"),
        "parametros_entrenables": resumen.get("parametros_entrenables"),
        "manifest": (resumen.get("particion") or {}).get("manifest"),
        "metodo_particion": (resumen.get("particion") or {}).get("metodo"),
        "particion_sha256": (resumen.get("entrenamiento") or {}).get(
            "particion_sha256"
        ),
        "entorno_ejecucion": resumen.get("entorno_ejecucion"),
    }


def validar_conjunto_oficial(
    resumenes: dict[int, dict],
    *,
    barrido: dict | None = None,
) -> list[str]:
    errores: list[str] = []
    _agregar(
        errores, set(resumenes) == set(RESOLUCIONES),
        "deben existir exactamente los resultados R32 y R64",
    )
    if set(resumenes) != set(RESOLUCIONES):
        return errores
    campos = [_campos_comunes(resumenes[r]) for r in RESOLUCIONES]
    for clave in campos[0]:
        _agregar(
            errores, campos[0][clave] == campos[1][clave],
            f"R32 y R64 no comparten {clave}",
        )
    if barrido is not None:
        _agregar(
            errores, barrido.get("schema_name") == "net5-octree-worker-benchmark",
            "el JSON de workers tiene schema_name invalido",
        )
        _agregar(
            errores, barrido.get("schema_version") == "1.2.0",
            "el JSON de workers tiene schema_version invalido",
        )
        _agregar(errores, barrido.get("estado") == "VALIDADO", "el barrido no esta VALIDADO")
        _agregar(
            errores, barrido.get("git_commit") == campos[0]["git_commit"],
            "el barrido y los entrenamientos usan commits distintos",
        )
        _agregar(
            errores,
            barrido.get("entorno_ejecucion") == campos[0]["entorno_ejecucion"],
            "el barrido y los entrenamientos usan entornos distintos",
        )
        seleccion = barrido.get("seleccion_workers") or {}
        for resolucion in RESOLUCIONES:
            workers = (resumenes[resolucion].get("configuracion") or {}).get(
                "num_workers"
            )
            _agregar(
                errores, seleccion.get(str(resolucion)) == workers,
                f"R{resolucion} no usa los workers seleccionados por el barrido",
            )
    return errores


COLUMNAS_CSV = [
    "resolucion", "git_commit", "num_workers", "epocas_completadas",
    "mejor_epoca", "mejor_val_acc", "test_acc", "test_loss",
    "tiempo_total_min", "forward_ms_por_muestra", "pipeline_ms_por_muestra",
    "vram_pico_mb", "tamano_modelo_mb", "gpu_nombre", "pytorch_version",
    "cuda_version",
]


def _fila(resumen: dict) -> dict:
    entorno = resumen["entorno_ejecucion"]
    gpu = entorno["hardware"]["gpu"]
    return {
        "resolucion": resumen["resolucion"],
        "git_commit": resumen["git_commit"],
        "num_workers": resumen["configuracion"]["num_workers"],
        "epocas_completadas": resumen["entrenamiento"]["epocas_completadas"],
        "mejor_epoca": resumen["mejor_epoca"],
        "mejor_val_acc": resumen["mejor_val_acc"],
        "test_acc": resumen["test_acc"],
        "test_loss": resumen["test_loss"],
        "tiempo_total_min": resumen["tiempo_total_min"],
        "forward_ms_por_muestra": resumen["tiempo_inferencia_promedio_ms"],
        "pipeline_ms_por_muestra": resumen["tiempo_pipeline_promedio_ms"],
        "vram_pico_mb": resumen["vram_pico_mb"],
        "tamano_modelo_mb": resumen["tamano_modelo_mb"],
        "gpu_nombre": gpu["nombre"],
        "pytorch_version": entorno["pytorch"]["version"],
        "cuda_version": entorno["pytorch"]["cuda_compilacion"],
    }


def escribir_consolidado(
    resumenes: dict[int, dict],
    validaciones: list[dict],
    errores_conjunto: list[str],
    *,
    salida_dir: Path,
    barrido: dict | None,
) -> tuple[Path, Path, bool]:
    salida_dir.mkdir(parents=True, exist_ok=True)
    valido = not errores_conjunto and all(
        not validacion["errores"] for validacion in validaciones
    )
    resoluciones_validas = {
        validacion.get("resolucion")
        for validacion in validaciones
        if validacion.get("valido") is True
        and not validacion.get("errores")
    }
    filas = [
        _fila(resumenes[resolucion])
        for resolucion in RESOLUCIONES
        if resolucion in resumenes and resolucion in resoluciones_validas
    ]
    informe = {
        "schema_name": SCHEMA_VALIDACION,
        "schema_version": SCHEMA_VALIDACION_VERSION,
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "estado": "VALIDADO" if valido else "INVALIDO",
        "validacion_completa": valido,
        "resoluciones": list(RESOLUCIONES),
        "validaciones": validaciones,
        "errores_conjunto": errores_conjunto,
        "barrido_workers": (
            {
                "git_commit": barrido.get("git_commit"),
                "seleccion_workers": barrido.get("seleccion_workers"),
            }
            if barrido is not None
            else None
        ),
        "resultados": filas,
    }
    ruta_json = salida_dir / "validacion_objetivo3_net5.json"
    ruta_csv = salida_dir / "resumen_objetivo3_net5.csv"
    temporal_json = ruta_json.with_suffix(".json.tmp")
    temporal_csv = ruta_csv.with_suffix(".csv.tmp")
    with temporal_json.open("w", encoding="utf-8", newline="\n") as archivo:
        json.dump(informe, archivo, indent=2, ensure_ascii=False)
        archivo.write("\n")
    os.replace(temporal_json, ruta_json)
    with temporal_csv.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_CSV)
        escritor.writeheader()
        escritor.writerows(filas)
    os.replace(temporal_csv, ruta_csv)
    return ruta_json, ruta_csv, valido


def validar_archivos(
    resultados_dir: Path,
    *,
    workers_por_resolucion: dict[int, int],
    epochs: int = 200,
    patience: int = 20,
    batch_size: int = 1,
    lr: float = 0.001,
    lotes_perfil: int = 3,
    barrido: dict | None = None,
) -> tuple[dict[int, dict], list[dict], list[str]]:
    resumenes: dict[int, dict] = {}
    validaciones = []
    for resolucion in RESOLUCIONES:
        ruta = resultados_dir / f"resumen_net5_octree_R{resolucion}.json"
        errores = []
        try:
            resumen = cargar_json_estricto(ruta)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            resumen = None
            errores.append(str(exc))
        if resumen is not None:
            resumenes[resolucion] = resumen
            errores.extend(validar_resumen_oficial(
                resumen,
                resolucion,
                workers=workers_por_resolucion[resolucion],
                epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                lr=lr,
                lotes_perfil=lotes_perfil,
            ))
        validaciones.append({
            "resolucion": resolucion,
            "archivo": ruta.name,
            "valido": not errores,
            "errores": errores,
        })
    errores_conjunto = validar_conjunto_oficial(resumenes, barrido=barrido)
    return resumenes, validaciones, errores_conjunto


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida los resultados oficiales R32/R64 de Net5-OctNet.",
    )
    parser.add_argument("--resultados-dir", type=Path, required=True)
    parser.add_argument("--salida-dir", type=Path, default=None)
    parser.add_argument("--barrido-json", type=Path, default=None)
    parser.add_argument("--workers-r32", type=int, default=8)
    parser.add_argument("--workers-r64", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lotes-perfil", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = _argumentos()
    barrido = (
        cargar_json_estricto(args.barrido_json)
        if args.barrido_json is not None
        else None
    )
    resumenes, validaciones, errores_conjunto = validar_archivos(
        args.resultados_dir.resolve(),
        workers_por_resolucion={
            32: args.workers_r32,
            64: args.workers_r64,
        },
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        lotes_perfil=args.lotes_perfil,
        barrido=barrido,
    )
    salida_dir = (args.salida_dir or args.resultados_dir).resolve()
    ruta_json, ruta_csv, valido = escribir_consolidado(
        resumenes,
        validaciones,
        errores_conjunto,
        salida_dir=salida_dir,
        barrido=barrido,
    )
    print(f"JSON: {ruta_json}")
    print(f"CSV : {ruta_csv}")
    if not valido:
        print("La validacion del Objetivo 3 fallo.")
        return 1
    print("Resultados oficiales del Objetivo 3 validados correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
