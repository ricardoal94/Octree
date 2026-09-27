"""Ejecuta de forma controlada los entrenamientos oficiales R32 y R64.

El flujo exige un barrido de workers valido del mismo commit y entorno,
escribe todos los artefactos fuera del repositorio, permite continuar una
ejecucion interrumpida y valida los dos resultados antes de consolidarlos.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from comparar_workers_octnet import (
    SCHEMA_BARRIDO,
    SCHEMA_BARRIDO_VERSION,
    cargar_resumen,
    construir_entorno_workers,
)
from comparar_workers_octnet import (
    ruta_resumen as ruta_resumen_smoke,
)
from comparar_workers_octnet import (
    validar_conjunto as validar_conjunto_smoke,
)
from comparar_workers_octnet import (
    validar_resumen as validar_resumen_smoke,
)
from trazabilidad_git import (
    capturar_estado_git,
    exigir_commit_git_publicado,
    exigir_estado_git_limpio,
)
from validar_resultados_net5 import (
    RESOLUCIONES,
    cargar_json_estricto,
    escribir_consolidado,
    validar_archivos,
    validar_resumen_oficial,
)

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
SCRIPT_ENTRENAMIENTO = RAIZ_PROYECTO / "fase3_net5" / "fase3_net5_entrenamiento.py"
WORKERS_BARRIDO = (0, 4, 8)
LIMITES_BARRIDO = {"train": 40, "val": 40, "test": 40}
EPOCHS_BARRIDO = 1
BATCH_BARRIDO = 1
LOTES_PERFIL_BARRIDO = 3


class ErrorObjetivo3(ValueError):
    """Indica que el protocolo oficial no puede continuar."""


def _mostrar_comando(comando: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(comando)
    return shlex.join(comando)


def ruta_resumen_oficial(resultados_dir: Path, resolucion: int) -> Path:
    return resultados_dir / f"resumen_net5_octree_R{resolucion}.json"


def ruta_checkpoint_ultimo(checkpoints_dir: Path, resolucion: int) -> Path:
    return checkpoints_dir / f"net5_octree_ultimo_R{resolucion}.pth"


def construir_comando_oficial(
    *,
    python: str,
    resolucion: int,
    workers: int,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    lotes_perfil: int,
    data_root: Path,
    particion_manifest: Path,
    checkpoints_dir: Path,
    logs_dir: Path,
    resultados_dir: Path,
    reanudar: bool,
) -> list[str]:
    comando = [
        python,
        str(SCRIPT_ENTRENAMIENTO),
        "--resolucion", str(resolucion),
        "--backend", "octree_native",
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
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
    if reanudar:
        comando.append("--reanudar")
    return comando


def cargar_y_validar_barrido(
    barrido_dir: Path,
    *,
    commit_actual: str | None,
) -> tuple[dict, dict[int, int]]:
    resultados_dir = barrido_dir / "resultados"
    ejecuciones = []
    for resolucion in RESOLUCIONES:
        for workers in WORKERS_BARRIDO:
            resumen = cargar_resumen(
                ruta_resumen_smoke(resultados_dir, resolucion, workers)
            )
            validar_resumen_smoke(
                resumen,
                resolucion=resolucion,
                workers=workers,
                limite_train=LIMITES_BARRIDO["train"],
                limite_val=LIMITES_BARRIDO["val"],
                limite_test=LIMITES_BARRIDO["test"],
                epochs=EPOCHS_BARRIDO,
                batch_size=BATCH_BARRIDO,
                lotes_perfil=LOTES_PERFIL_BARRIDO,
            )
            ejecuciones.append(resumen)
    validar_conjunto_smoke(
        ejecuciones,
        resoluciones=RESOLUCIONES,
        workers=WORKERS_BARRIDO,
    )
    ruta_consolidado = resultados_dir / "comparacion_workers_octnet.json"
    consolidado = cargar_json_estricto(ruta_consolidado)
    errores = []
    if consolidado.get("schema_name") != SCHEMA_BARRIDO:
        errores.append("schema_name del barrido es invalido")
    if consolidado.get("schema_version") != SCHEMA_BARRIDO_VERSION:
        errores.append("schema_version del barrido es invalido")
    if consolidado.get("estado") != "VALIDADO":
        errores.append("el barrido no esta VALIDADO")
    if consolidado.get("n_ejecuciones") != 6:
        errores.append("el barrido no contiene las seis ejecuciones")
    seleccion_calculada = {
        resolucion: min(
            (
                resumen for resumen in ejecuciones
                if resumen["resolucion"] == resolucion
            ),
            key=lambda resumen: resumen["tiempo_pipeline_promedio_ms"],
        )["configuracion"]["num_workers"]
        for resolucion in RESOLUCIONES
    }
    seleccion_json = consolidado.get("seleccion_workers") or {}
    for resolucion, workers in seleccion_calculada.items():
        if seleccion_json.get(str(resolucion)) != workers:
            errores.append(
                f"la seleccion de workers R{resolucion} no coincide con "
                "las mediciones"
            )
    commit_barrido = ejecuciones[0].get("git_commit")
    if consolidado.get("git_commit") != commit_barrido:
        errores.append("el commit consolidado no coincide con los resúmenes")
    if commit_actual is not None and commit_barrido != commit_actual:
        errores.append(
            "el barrido pertenece a otro commit; debe regenerarse despues "
            "de publicar el flujo oficial"
        )
    entorno = ejecuciones[0].get("entorno_ejecucion")
    if consolidado.get("entorno_ejecucion") != entorno:
        errores.append("el entorno consolidado no coincide con los resúmenes")
    if errores:
        raise ErrorObjetivo3(
            "El barrido de workers no cumple el protocolo:\n- "
            + "\n- ".join(errores)
        )
    return consolidado, seleccion_calculada


def validar_entorno_actual_con_barrido(barrido: dict) -> None:
    # Importacion diferida: la CI liviana puede probar este modulo sin PyTorch.
    from entorno_ejecucion import capturar_entorno_ejecucion
    from net5_modelo import get_device

    entorno_actual = capturar_entorno_ejecucion(get_device())
    if entorno_actual != barrido.get("entorno_ejecucion"):
        raise ErrorObjetivo3(
            "El entorno actual no coincide exactamente con el barrido de "
            "workers. Regenera el barrido en este equipo antes de entrenar."
        )


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta y valida los entrenamientos oficiales Net5 R32/R64.",
    )
    parser.add_argument(
        "--barrido-dir",
        type=Path,
        default=RAIZ_PROYECTO.parent / f"{RAIZ_PROYECTO.name}_smoke_workers",
        help="Carpeta externa generada por comparar_workers_octnet.py",
    )
    parser.add_argument(
        "--artefactos-dir",
        type=Path,
        default=RAIZ_PROYECTO.parent / f"{RAIZ_PROYECTO.name}_objetivo3_oficial",
        help="Carpeta externa para checkpoints, logs y resultados oficiales",
    )
    parser.add_argument("--data-root", type=Path, default=RAIZ_PROYECTO / "data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lotes-perfil", type=int, default=3)
    parser.add_argument(
        "--continuar",
        action="store_true",
        help=(
            "Omite resultados ya validos y reanuda desde la ultima epoca "
            "cuando existe un checkpoint compatible"
        ),
    )
    parser.add_argument(
        "--solo-validar", action="store_true",
        help="No entrena; valida y consolida los resultados existentes",
    )
    parser.add_argument(
        "--mostrar-comandos", action="store_true",
        help="Imprime los comandos previstos sin ejecutar ni crear carpetas",
    )
    return parser.parse_args()


def _validar_argumentos(args: argparse.Namespace) -> None:
    for nombre in ("epochs", "patience", "batch_size", "lotes_perfil"):
        if getattr(args, nombre) < 1:
            raise ErrorObjetivo3(
                f"--{nombre.replace('_', '-')} debe ser positivo"
            )
    if args.lr <= 0:
        raise ErrorObjetivo3("--lr debe ser positivo")
    if args.solo_validar and args.mostrar_comandos:
        raise ErrorObjetivo3(
            "--solo-validar y --mostrar-comandos son excluyentes"
        )


def _validar_resumen_individual(
    ruta: Path,
    resolucion: int,
    *,
    workers: int,
    args: argparse.Namespace,
) -> dict:
    resumen = cargar_json_estricto(ruta)
    errores = validar_resumen_oficial(
        resumen,
        resolucion,
        workers=workers,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        lotes_perfil=args.lotes_perfil,
    )
    if errores:
        raise ErrorObjetivo3(
            f"El resultado R{resolucion} no es valido:\n- "
            + "\n- ".join(errores)
        )
    return resumen


def main() -> int:
    args = _argumentos()
    _validar_argumentos(args)
    barrido_dir = args.barrido_dir.resolve()
    artefactos_dir = args.artefactos_dir.resolve()
    checkpoints_dir = artefactos_dir / "checkpoints"
    logs_dir = artefactos_dir / "logs"
    resultados_dir = artefactos_dir / "resultados"
    particion_manifest = barrido_dir / "logs" / "particion_objetivo2_modelos.json"

    estado_git = capturar_estado_git(RAIZ_PROYECTO)
    if not args.mostrar_comandos:
        exigir_estado_git_limpio(estado_git)
        exigir_commit_git_publicado(estado_git)
    barrido, seleccion_workers = cargar_y_validar_barrido(
        barrido_dir,
        commit_actual=(
            None if args.mostrar_comandos else estado_git.get("git_commit")
        ),
    )
    if not args.mostrar_comandos:
        validar_entorno_actual_con_barrido(barrido)

    comandos = []
    for resolucion in RESOLUCIONES:
        ultimo = ruta_checkpoint_ultimo(checkpoints_dir, resolucion)
        reanudar = bool(args.continuar and ultimo.is_file())
        comandos.append((
            resolucion,
            construir_comando_oficial(
                python=sys.executable,
                resolucion=resolucion,
                workers=seleccion_workers[resolucion],
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                lr=args.lr,
                lotes_perfil=args.lotes_perfil,
                data_root=args.data_root.resolve(),
                particion_manifest=particion_manifest,
                checkpoints_dir=checkpoints_dir,
                logs_dir=logs_dir,
                resultados_dir=resultados_dir,
                reanudar=reanudar,
            ),
        ))

    if args.mostrar_comandos:
        for _, comando in comandos:
            print(_mostrar_comando(comando))
        return 0

    if not particion_manifest.is_file():
        raise ErrorObjetivo3(
            f"No existe el manifiesto validado del barrido: {particion_manifest}"
        )
    for directorio in (checkpoints_dir, logs_dir, resultados_dir):
        directorio.mkdir(parents=True, exist_ok=True)

    if not args.continuar and not args.solo_validar:
        existentes = [
            ruta for ruta in artefactos_dir.rglob("*") if ruta.is_file()
        ]
        if existentes:
            raise ErrorObjetivo3(
                "La carpeta de artefactos ya contiene archivos. Usa otra "
                "carpeta o --continuar para proteger la evidencia existente."
            )

    entorno_workers = construir_entorno_workers()
    if not args.solo_validar:
        for indice, (resolucion, comando) in enumerate(comandos, start=1):
            ruta = ruta_resumen_oficial(resultados_dir, resolucion)
            if args.continuar and ruta.is_file():
                try:
                    _validar_resumen_individual(
                        ruta,
                        resolucion,
                        workers=seleccion_workers[resolucion],
                        args=args,
                    )
                except (ErrorObjetivo3, TypeError, ValueError):
                    pass
                else:
                    print(
                        f"[{indice}/2] R{resolucion} ya esta validado; se omite."
                    )
                    continue
            print(f"[{indice}/2] Ejecutando R{resolucion} oficial")
            print(_mostrar_comando(comando))
            subprocess.run(
                comando,
                cwd=RAIZ_PROYECTO,
                check=True,
                env=entorno_workers,
            )
            _validar_resumen_individual(
                ruta,
                resolucion,
                workers=seleccion_workers[resolucion],
                args=args,
            )

    resumenes, validaciones, errores_conjunto = validar_archivos(
        resultados_dir,
        workers_por_resolucion=seleccion_workers,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        lotes_perfil=args.lotes_perfil,
        barrido=barrido,
    )
    ruta_json, ruta_csv, valido = escribir_consolidado(
        resumenes,
        validaciones,
        errores_conjunto,
        salida_dir=resultados_dir,
        barrido=barrido,
    )
    print(f"JSON: {ruta_json}")
    print(f"CSV : {ruta_csv}")
    if not valido:
        raise ErrorObjetivo3(
            "Los dos entrenamientos aun no satisfacen el contrato del Objetivo 3"
        )
    print("Objetivo 3: resultados oficiales R32/R64 validados.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ErrorObjetivo3,
        FileNotFoundError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
