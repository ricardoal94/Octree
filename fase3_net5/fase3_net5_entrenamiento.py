"""Entrenamiento de Net5-Octree y de su referencia densa diagnostica.

El backend predeterminado ``octree_native`` opera directamente sobre el
grid-octree. ``dense_reference`` se conserva solo como diagnostico y sus
resultados nunca son oficiales.

Caracteristicas:
  - CPU o GPU via PyTorch
  - Early Stopping basado en val accuracy (segun metodologia, sec. 6.3)
  - Optimizador Adam + CrossEntropy
  - Log CSV y JSON por epoca
  - Checkpoint del mejor modelo
  - Medicion de VRAM, tiempo de entrenamiento e inferencia (Fase 4)
  - Sin aumento de datos (criterio de equivalencia, sec. 6.6)

Uso:
    python fase3_net5/fase3_net5_entrenamiento.py --resolucion 32 \
      --backend dense_reference --tag _smoke --limite_train 80 \
      --limite_val 40 --limite_test 80
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "fase1_modelnet40"))
sys.path.insert(0, str(Path(__file__).parent))

from entorno_ejecucion import capturar_entorno_ejecucion
from fase1_setup import set_global_seed
from net5_dataset import CLASES, crear_dataloaders_densos_referencia
from net5_dataset_octree import crear_dataloaders_octree
from net5_modelo import crear_modelo, crear_modelo_denso_referencia, get_device
from particion_objetivo2 import (
    cargar_o_crear_particion,
    listar_muestras_octree,
    seleccionar_subconjunto_balanceado,
)
from trazabilidad_git import (
    capturar_estado_git,
    exigir_commit_git_publicado,
    exigir_estado_git_limpio,
)

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
SEED = 42
SCHEMA_RESUMEN_VERSION = "1.4.0"
SCHEMA_CHECKPOINT_ULTIMO = "net5-training-checkpoint"
SCHEMA_CHECKPOINT_ULTIMO_VERSION = "1.0.0"
VARIABLES_HILOS_CPU = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _sincronizar_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _ruta_portable(ruta: Path) -> str:
    ruta = ruta.resolve()
    try:
        return ruta.relative_to(RAIZ_PROYECTO.resolve()).as_posix()
    except ValueError:
        return ruta.name


def _sha256_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _guardar_checkpoint_atomico(datos: dict, ruta: Path) -> None:
    temporal = ruta.with_name(f"{ruta.name}.tmp")
    torch.save(datos, temporal)
    os.replace(temporal, ruta)


def _cargar_checkpoint(ruta: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(
            ruta, map_location=device, weights_only=False,
        )
    except TypeError:  # PyTorch anterior a ``weights_only``.
        checkpoint = torch.load(ruta, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"El checkpoint {ruta} no contiene un diccionario")
    return checkpoint


def _capturar_estado_rng(loader_train) -> dict:
    generador = getattr(loader_train, "generator", None)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "dataloader": (
            generador.get_state() if generador is not None else None
        ),
    }


def _restaurar_estado_rng(estado: dict, loader_train) -> None:
    requeridos = {"python", "numpy", "torch_cpu", "torch_cuda", "dataloader"}
    faltantes = sorted(requeridos - set(estado))
    if faltantes:
        raise ValueError(
            "El checkpoint no contiene todo el estado aleatorio: "
            + ", ".join(faltantes)
        )
    random.setstate(estado["python"])
    np.random.set_state(estado["numpy"])
    torch.set_rng_state(estado["torch_cpu"])
    if estado["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError(
                "El checkpoint contiene estado CUDA, pero CUDA no esta disponible"
            )
        torch.cuda.set_rng_state_all(estado["torch_cuda"])
    generador = getattr(loader_train, "generator", None)
    if estado["dataloader"] is not None:
        if generador is None:
            raise ValueError("El DataLoader no expone el generador guardado")
        generador.set_state(estado["dataloader"])


def _validar_contrato_reanudacion(guardado: object, actual: dict) -> None:
    if not isinstance(guardado, dict):
        raise TypeError("El checkpoint no contiene contrato de reanudacion")
    diferencias = [
        clave for clave, valor in actual.items()
        if guardado.get(clave) != valor
    ]
    extras = sorted(set(guardado) - set(actual))
    if diferencias or extras:
        detalle = ", ".join(sorted(diferencias + extras))
        raise ValueError(
            "El checkpoint no corresponde a la corrida actual; difieren: "
            f"{detalle}"
        )


def _reescribir_logs_historial(
    historial: list[dict], csv_path: Path, json_path: Path,
) -> None:
    columnas = [
        "epoca", "train_loss", "train_acc", "val_loss", "val_acc", "lr",
        "tiempo_s",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(historial)
    with json_path.open("w", encoding="utf-8", newline="\n") as archivo:
        json.dump(historial, archivo, indent=2, ensure_ascii=False)
        archivo.write("\n")


# ──────────────────────────────────────────────────────────────
# ENTRENAMIENTO DE UNA EPOCA
# ──────────────────────────────────────────────────────────────

def entrenar_epoca(modelo, loader, criterio, optimizador, device):
    modelo.train()
    perdida_total, correctos, total = 0.0, 0, 0

    barra = tqdm(loader, desc="  Train", leave=False, ncols=80)
    for grids, etiquetas in barra:
        grids     = grids.to(device, non_blocking=True)
        etiquetas = etiquetas.to(device, non_blocking=True)

        optimizador.zero_grad()
        logits  = modelo(grids)
        perdida = criterio(logits, etiquetas)
        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item() * grids.size(0)
        correctos     += (logits.argmax(1) == etiquetas).sum().item()
        total         += grids.size(0)
        barra.set_postfix(loss=f"{perdida.item():.4f}")

    return perdida_total / total, correctos / total


# ──────────────────────────────────────────────────────────────
# EVALUACION
# ──────────────────────────────────────────────────────────────

def evaluar(modelo, loader, criterio, device, desc="Val"):
    modelo.eval()
    perdida_total, correctos, total = 0.0, 0, 0

    with torch.no_grad():
        barra = tqdm(loader, desc=f"  {desc}", leave=False, ncols=80)
        for grids, etiquetas in barra:
            grids     = grids.to(device, non_blocking=True)
            etiquetas = etiquetas.to(device, non_blocking=True)
            logits    = modelo(grids)
            perdida   = criterio(logits, etiquetas)
            perdida_total += perdida.item() * grids.size(0)
            correctos     += (logits.argmax(1) == etiquetas).sum().item()
            total         += grids.size(0)

    return perdida_total / total, correctos / total


# ──────────────────────────────────────────────────────────────
# MEDIR TIEMPO DE INFERENCIA (Fase 4 de la metodologia)
# ──────────────────────────────────────────────────────────────

def medir_tiempo_inferencia(modelo, loader, device, n_repeticiones: int = 3) -> dict:
    """
    Mide el tiempo promedio de clasificar una sola muestra sobre el
    conjunto de test completo, promediando N repeticiones.
    Cumple con la Fase 4: "tiempo promedio requerido para clasificar
    una sola muestra, calculado sobre el total del conjunto de prueba".
    """
    modelo.eval()
    tiempos = []
    tiempos_pipeline = []
    total_ultima_repeticion = 0

    for _ in range(n_repeticiones):
        tiempo_modelo = 0.0
        total = 0
        _sincronizar_cuda(device)
        t0_pipeline = time.perf_counter()
        with torch.no_grad():
            for grids, _ in loader:
                grids = grids.to(device, non_blocking=True)
                _sincronizar_cuda(device)
                t0 = time.perf_counter()
                modelo(grids)
                _sincronizar_cuda(device)
                tiempo_modelo += time.perf_counter() - t0
                total += grids.size(0)
        _sincronizar_cuda(device)
        tiempos_pipeline.append(time.perf_counter() - t0_pipeline)
        tiempos.append(tiempo_modelo)
        total_ultima_repeticion = total

    tiempo_total_prom = np.mean(tiempos)
    tiempo_por_muestra_ms = (tiempo_total_prom / total_ultima_repeticion) * 1000
    tiempo_pipeline_total_prom = np.mean(tiempos_pipeline)
    tiempo_pipeline_por_muestra_ms = (
        tiempo_pipeline_total_prom / total_ultima_repeticion
    ) * 1000

    return {
        "tiempo_inferencia_total_s":       round(float(tiempo_total_prom), 4),
        "tiempo_inferencia_promedio_ms":    round(float(tiempo_por_muestra_ms), 4),
        "tiempo_pipeline_total_s": round(
            float(tiempo_pipeline_total_prom), 4,
        ),
        "tiempo_pipeline_promedio_ms": round(
            float(tiempo_pipeline_por_muestra_ms), 4,
        ),
        "n_muestras_test":                 int(total_ultima_repeticion),
        "protocolo_tiempo": (
            "forward con preparacion y transferencia de planes del backend; "
            "excluye carga del lote y transferencia inicial de atributos"
        ),
        "protocolo_tiempo_pipeline": (
            "extremo a extremo; incluye carga, preparacion geometrica, "
            "transferencia CPU-GPU y forward"
        ),
        "repeticiones_inferencia": int(n_repeticiones),
    }


def medir_perfil_rendimiento(modelo, loader, device, n_lotes: int = 3) -> dict:
    """Separa carga, transferencia, planes del backend y resto del forward."""

    modelo.eval()
    tiempos = {
        "carga_lote_s": 0.0,
        "transferencia_atributos_s": 0.0,
        "forward_total_s": 0.0,
    }
    perfil_backend = {}
    total_muestras = 0
    lotes_medidos = 0
    iterador = iter(loader)

    with torch.no_grad():
        while lotes_medidos < n_lotes:
            t0 = time.perf_counter()
            try:
                grids, _ = next(iterador)
            except StopIteration:
                break
            tiempos["carga_lote_s"] += time.perf_counter() - t0

            _sincronizar_cuda(device)
            t0 = time.perf_counter()
            grids = grids.to(device, non_blocking=True)
            _sincronizar_cuda(device)
            tiempos["transferencia_atributos_s"] += time.perf_counter() - t0

            if hasattr(grids, "activar_perfil"):
                grids.activar_perfil()
            _sincronizar_cuda(device)
            t0 = time.perf_counter()
            modelo(grids)
            _sincronizar_cuda(device)
            tiempos["forward_total_s"] += time.perf_counter() - t0

            if hasattr(grids, "resumen_perfil"):
                for clave, valor in grids.resumen_perfil().items():
                    perfil_backend[clave] = perfil_backend.get(clave, 0) + valor
            total_muestras += grids.size(0)
            lotes_medidos += 1

    claves_tiempo_plan = (
        "plan_convolucion_cpu_s",
        "transferencia_plan_convolucion_s",
        "plan_pooling_cpu_s",
        "transferencia_plan_pooling_s",
        "mapa_final_cpu_s",
        "transferencia_mapa_final_s",
    )
    tiempos_plan_s = sum(
        float(perfil_backend.get(clave, 0.0))
        for clave in claves_tiempo_plan
    )
    resto_forward_s = max(0.0, tiempos["forward_total_s"] - tiempos_plan_s)
    divisor = max(total_muestras, 1)
    bytes_planes = sum(
        int(valor)
        for clave, valor in perfil_backend.items()
        if clave.startswith("bytes_")
    )
    interacciones = int(
        perfil_backend.get("n_interacciones_convolucion", 0)
    )
    return {
        "lotes_medidos": lotes_medidos,
        "muestras_medidas": total_muestras,
        "carga_lote_ms_por_muestra": round(
            tiempos["carga_lote_s"] * 1000 / divisor, 4,
        ),
        "transferencia_atributos_ms_por_muestra": round(
            tiempos["transferencia_atributos_s"] * 1000 / divisor, 4,
        ),
        "forward_total_ms_por_muestra": round(
            tiempos["forward_total_s"] * 1000 / divisor, 4,
        ),
        "planes_backend_ms_por_muestra": round(
            tiempos_plan_s * 1000 / divisor, 4,
        ),
        "resto_forward_ms_por_muestra": round(
            resto_forward_s * 1000 / divisor, 4,
        ),
        "memoria_planes_cpu_mb_por_muestra": round(
            bytes_planes / 1e6 / divisor, 4,
        ),
        "interacciones_convolucion_por_muestra": round(
            interacciones / divisor, 2,
        ),
        "detalle_backend": {
            (
                clave.replace("_s", "_ms_por_muestra")
                if clave.endswith("_s")
                else clave
            ): (
                round(float(valor) * 1000 / divisor, 4)
                if clave.endswith("_s")
                else int(valor)
            )
            for clave, valor in perfil_backend.items()
        },
        "nota": (
            "Perfil diagnostico sobre un subconjunto de lotes. Si el "
            "precalculo esta activo, su costo aparece en carga_lote; dentro "
            "del forward se miden el ensamble del lote y las transferencias."
        ),
    }


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolucion", type=int, default=32, choices=[32, 64])
    parser.add_argument(
        "--backend", choices=["octree_native", "dense_reference"],
        default="octree_native",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping: epocas sin mejora antes de parar")
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Por defecto: 1 para octree_native y 16 para dense_reference",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--limite_train", type=int, default=None,
                        help="Usar N muestras balanceadas de train "
                             "(smoke test rapido, no afecta el uso normal)")
    parser.add_argument("--limite_val", type=int, default=None,
                        help="Usar N muestras balanceadas de val")
    parser.add_argument("--limite_test", type=int, default=None,
                        help="Usar N muestras balanceadas de test")
    parser.add_argument("--tag", type=str, default="",
                        help="Sufijo para checkpoint/logs/resultados, para no "
                             "sobreescribir una corrida completa (ej. '_smoke')")
    parser.add_argument("--data-root", type=Path, default=RAIZ_PROYECTO / "data")
    parser.add_argument(
        "--particion-manifest", type=Path,
        default=RAIZ_PROYECTO / "logs" / "particion_objetivo2_modelos.json",
    )
    parser.add_argument(
        "--checkpoints-dir", type=Path, default=RAIZ_PROYECTO / "checkpoints",
    )
    parser.add_argument("--logs-dir", type=Path, default=RAIZ_PROYECTO / "logs")
    parser.add_argument(
        "--resultados-dir", type=Path,
        default=RAIZ_PROYECTO / "resultados" / "objetivo3",
    )
    parser.add_argument(
        "--reanudar", action="store_true",
        help=(
            "Continua desde el checkpoint de ultima epoca. Rechaza cambios "
            "de codigo, entorno, datos o hiperparametros."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--lotes-perfil", type=int, default=3,
        help="Numero de lotes de test usados para perfilar el backend",
    )
    parser.add_argument(
        "--exigir-git-limpio", action="store_true",
        help="Abortar si la corrida no parte de un commit sin cambios locales",
    )
    parser.add_argument(
        "--exigir-git-publicado", action="store_true",
        help="Abortar si HEAD no aparece en una referencia remota conocida",
    )
    parser.add_argument(
        "--sin-precalcular-planes",
        dest="precalcular_planes",
        action="store_false",
        help=(
            "Construir los planes dentro del forward (solo para comparar el "
            "rendimiento; por defecto se preparan en el DataLoader)"
        ),
    )
    parser.set_defaults(precalcular_planes=True)
    args = parser.parse_args()
    if args.lotes_perfil < 1:
        raise ValueError("--lotes-perfil debe ser al menos 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers no puede ser negativo")
    if args.epochs < 1:
        raise ValueError("--epochs debe ser al menos 1")
    if args.patience < 1:
        raise ValueError("--patience debe ser al menos 1")
    if args.lr <= 0:
        raise ValueError("--lr debe ser positivo")

    estado_git_inicial = capturar_estado_git(RAIZ_PROYECTO)
    if args.exigir_git_limpio:
        exigir_estado_git_limpio(estado_git_inicial)
    if args.exigir_git_publicado:
        exigir_commit_git_publicado(estado_git_inicial)

    R = args.resolucion
    if args.batch_size is None:
        args.batch_size = 1 if args.backend == "octree_native" else 16
    if args.batch_size < 1:
        raise ValueError("--batch_size debe ser al menos 1")

    if args.backend == "dense_reference" and not args.tag:
        raise ValueError(
            "La referencia densa es solo diagnostica y requiere --tag "
            "(por ejemplo, --tag _smoke_dense)."
        )

    print("=" * 60)
    print(
        "  Git            : "
        f"{estado_git_inicial.get('git_branch') or 'desconocida'} @ "
        f"{(estado_git_inicial.get('git_commit') or 'desconocido')[:12]} | "
        f"limpio={estado_git_inicial.get('git_dirty') is False} | "
        f"publicado={estado_git_inicial.get('git_commit_publicado') is True}"
    )
    nombre_backend = (
        "NET5-OCTREE NATIVO"
        if args.backend == "octree_native"
        else "DIAGNOSTICO DENSO TABLA 5"
    )
    print(f"  {nombre_backend} — Resolucion {R}^3")
    print("=" * 60)

    set_global_seed(SEED)
    device = get_device()
    entorno_ejecucion = capturar_entorno_ejecucion(device)

    for d in (args.checkpoints_dir, args.logs_dir, args.resultados_dir):
        d.mkdir(parents=True, exist_ok=True)

    raiz_resolucion = args.data_root / f"octrees_{R}"
    idx_train, idx_val, idx_test, particion = cargar_o_crear_particion(
        raiz_resolucion=raiz_resolucion,
        ruta_manifest=args.particion_manifest,
        seed=SEED,
        val_split=0.10,
    )
    _, etiquetas_train_total, _ = listar_muestras_octree(
        raiz_resolucion, "train",
    )
    _, etiquetas_test, _ = listar_muestras_octree(raiz_resolucion, "test")
    idx_train = seleccionar_subconjunto_balanceado(
        idx_train, etiquetas_train_total, args.limite_train, SEED,
    )
    idx_val = seleccionar_subconjunto_balanceado(
        idx_val, etiquetas_train_total, args.limite_val, SEED,
    )
    idx_test = seleccionar_subconjunto_balanceado(
        idx_test, etiquetas_test, args.limite_test, SEED,
    )

    argumentos_loader = {
        "raiz_resolucion": str(raiz_resolucion),
        "resolucion": R,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": SEED,
    }
    if args.backend == "octree_native":
        loader_train, loader_val, loader_test = crear_dataloaders_octree(
            **argumentos_loader,
            precalcular_planes=args.precalcular_planes,
        )
        modelo, device = crear_modelo(resolucion=R, device=device)
    else:
        loader_train, loader_val, loader_test = (
            crear_dataloaders_densos_referencia(**argumentos_loader)
        )
        modelo, device = crear_modelo_denso_referencia(
            resolucion=R, device=device,
        )
    criterio    = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler   = optim.lr_scheduler.StepLR(optimizador, step_size=20, gamma=0.7)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Archivos de log
    prefijo = "net5_octree" if args.backend == "octree_native" else "dense_tabla5"
    csv_path = args.logs_dir / f"{prefijo}_historial_R{R}{args.tag}.csv"
    json_path = args.logs_dir / f"{prefijo}_historial_R{R}{args.tag}.json"
    ckpt_path = args.checkpoints_dir / f"{prefijo}_mejor_R{R}{args.tag}.pth"
    ckpt_ultimo_path = (
        args.checkpoints_dir / f"{prefijo}_ultimo_R{R}{args.tag}.pth"
    )
    contrato_reanudacion = {
        "git_commit": estado_git_inicial.get("git_commit"),
        "backend": args.backend,
        "backend_version": "1.1.1",
        "resolucion": R,
        "seed": SEED,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": 1e-4,
        "scheduler": "StepLR(step_size=20, gamma=0.7)",
        "precalcular_planes": args.precalcular_planes,
        "limite_train": args.limite_train,
        "limite_val": args.limite_val,
        "limite_test": args.limite_test,
        "tag": args.tag,
        "particion_sha256": _sha256_archivo(args.particion_manifest),
        "n_train_usado": len(loader_train.dataset),
        "n_val_usado": len(loader_val.dataset),
        "n_test_usado": len(loader_test.dataset),
        "entorno_ejecucion": entorno_ejecucion,
    }

    historial: list[dict] = []
    mejor_val_acc = float("-inf")
    mejor_epoca = 0
    epocas_sin_mejora = 0
    epoca_inicial = 1
    reanudado_desde_epoca = None
    tiempo_entrenamiento_acumulado_s = 0.0
    vram_pico_acumulada_mb = 0.0

    if args.reanudar:
        if not ckpt_ultimo_path.is_file():
            raise FileNotFoundError(
                "No existe el checkpoint requerido para reanudar: "
                f"{ckpt_ultimo_path}"
            )
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                "No existe el checkpoint del mejor modelo: "
                f"{ckpt_path}"
            )
        ultimo = _cargar_checkpoint(ckpt_ultimo_path, device)
        if ultimo.get("schema_name") != SCHEMA_CHECKPOINT_ULTIMO:
            raise ValueError("El checkpoint de reanudacion tiene schema_name invalido")
        if ultimo.get("schema_version") != SCHEMA_CHECKPOINT_ULTIMO_VERSION:
            raise ValueError("El checkpoint de reanudacion tiene version incompatible")
        _validar_contrato_reanudacion(
            ultimo.get("contrato_reanudacion"), contrato_reanudacion,
        )
        modelo.load_state_dict(ultimo["model_state"])
        optimizador.load_state_dict(ultimo["optimizer_state"])
        scheduler.load_state_dict(ultimo["scheduler_state"])
        historial = ultimo.get("historial")
        if not isinstance(historial, list):
            raise ValueError("El checkpoint no contiene un historial valido")
        ultima_epoca = int(ultimo.get("epoca", 0))
        if [fila.get("epoca") for fila in historial] != list(
            range(1, ultima_epoca + 1)
        ):
            raise ValueError("El historial del checkpoint no es consecutivo")
        mejor_val_acc = float(ultimo["mejor_val_acc"])
        mejor_epoca = int(ultimo["mejor_epoca"])
        epocas_sin_mejora = int(ultimo["epocas_sin_mejora"])
        tiempo_entrenamiento_acumulado_s = float(
            ultimo["tiempo_entrenamiento_acumulado_s"]
        )
        vram_pico_acumulada_mb = float(
            ultimo.get("vram_pico_acumulada_mb", 0.0)
        )
        _restaurar_estado_rng(ultimo["estado_rng"], loader_train)
        mejor_guardado = _cargar_checkpoint(ckpt_path, device)
        if (
            int(mejor_guardado.get("epoca", 0)) != mejor_epoca
            or float(mejor_guardado.get("mejor_val_acc", -1.0))
            != mejor_val_acc
        ):
            raise ValueError(
                "El checkpoint del mejor modelo no coincide con la ultima "
                "epoca cerrada; no se puede reanudar de forma segura"
            )
        epoca_inicial = ultima_epoca + 1
        reanudado_desde_epoca = ultima_epoca
        print(
            f"  Reanudacion    : desde epoca {ultima_epoca}; "
            f"siguiente={epoca_inicial}"
        )

    _reescribir_logs_historial(historial, csv_path, json_path)

    print(f"\n  Epocas max   : {args.epochs}")
    print(f"  Early stop   : patience={args.patience}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Data workers : {args.num_workers}")
    print(f"  LR inicial   : {args.lr}")
    if args.backend == "octree_native":
        print(f"  Precalc. plan: {args.precalcular_planes}")
    print(f"  Train batches: {len(loader_train)}")
    print(f"  Val   batches: {len(loader_val)}\n")

    epocas_ejecutadas_esta_invocacion = 0
    for epoca in range(epoca_inicial, args.epochs + 1):
        if epocas_sin_mejora >= args.patience:
            break
        _sincronizar_cuda(device)
        t_ep = time.perf_counter()

        train_loss, train_acc = entrenar_epoca(modelo, loader_train, criterio, optimizador, device)
        val_loss,   val_acc   = evaluar(modelo, loader_val, criterio, device, "Val")

        scheduler.step()
        lr_actual = scheduler.get_last_lr()[0]
        _sincronizar_cuda(device)
        t_ep = time.perf_counter() - t_ep

        # Checkpoint si mejora
        es_mejor = val_acc > mejor_val_acc
        if es_mejor:
            mejor_val_acc = val_acc
            mejor_epoca = epoca
            epocas_sin_mejora = 0
            _guardar_checkpoint_atomico({
                "schema_name": "net5-best-checkpoint",
                "schema_version": "1.0.0",
                "epoca": epoca,
                "model_state": modelo.state_dict(),
                "mejor_val_acc": mejor_val_acc,
                "resolucion": R,
                "contrato_reanudacion": contrato_reanudacion,
            }, ckpt_path)
            marca = " <- mejor"
        else:
            epocas_sin_mejora += 1
            marca = ""

        print(
            f"  Ep {epoca:03d}/{args.epochs} | "
            f"Train {train_acc*100:.2f}% loss {train_loss:.4f} | "
            f"Val {val_acc*100:.2f}% loss {val_loss:.4f} | "
            f"LR {lr_actual:.6f} | {t_ep:.1f}s{marca}"
        )

        tiempo_entrenamiento_acumulado_s += t_ep
        epocas_ejecutadas_esta_invocacion += 1
        if device.type == "cuda":
            vram_actual_mb = torch.cuda.max_memory_allocated(device) / 1e6
            vram_pico_acumulada_mb = max(
                vram_pico_acumulada_mb, vram_actual_mb,
            )
            if epocas_ejecutadas_esta_invocacion == 1:
                print(f"  [GPU] VRAM pico: {vram_actual_mb:.1f} MB")

        historial.append({
            "epoca": epoca,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr": round(lr_actual, 8),
            "tiempo_s": round(t_ep, 2),
        })
        _reescribir_logs_historial(historial, csv_path, json_path)
        _guardar_checkpoint_atomico({
            "schema_name": SCHEMA_CHECKPOINT_ULTIMO,
            "schema_version": SCHEMA_CHECKPOINT_ULTIMO_VERSION,
            "contrato_reanudacion": contrato_reanudacion,
            "epoca": epoca,
            "model_state": modelo.state_dict(),
            "optimizer_state": optimizador.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "mejor_val_acc": mejor_val_acc,
            "mejor_epoca": mejor_epoca,
            "epocas_sin_mejora": epocas_sin_mejora,
            "historial": historial,
            "tiempo_entrenamiento_acumulado_s": (
                tiempo_entrenamiento_acumulado_s
            ),
            "vram_pico_acumulada_mb": vram_pico_acumulada_mb,
            "estado_rng": _capturar_estado_rng(loader_train),
        }, ckpt_ultimo_path)

        # Early stopping
        if epocas_sin_mejora >= args.patience:
            print(f"\n  [Early Stopping] Sin mejora en {args.patience} epocas. "
                  f"Deteniendo en epoca {epoca}.")
            break

    _sincronizar_cuda(device)
    t_total = tiempo_entrenamiento_acumulado_s / 60

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            "No se genero el checkpoint del mejor modelo; no es posible evaluar"
        )

    # Evaluacion final en test con el mejor modelo
    print("\n[Test] Cargando mejor modelo y evaluando...")
    ckpt = _cargar_checkpoint(ckpt_path, device)
    modelo.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluar(modelo, loader_test, criterio, device, "Test")

    # Matriz de confusion y reporte por clase
    print("[Test] Generando matriz de confusion...")
    from sklearn.metrics import classification_report, confusion_matrix
    todas_pred, todas_real = [], []
    modelo.eval()
    with torch.no_grad():
        for grids, etiquetas in loader_test:
            grids = grids.to(device, non_blocking=True)
            logits = modelo(grids)
            todas_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            todas_real.extend(etiquetas.numpy().tolist())
    etiquetas_40 = list(range(len(CLASES)))
    matriz_conf = confusion_matrix(todas_real, todas_pred, labels=etiquetas_40).tolist()
    reporte_cls = classification_report(todas_real, todas_pred,
                                        labels=etiquetas_40,
                                        target_names=CLASES,
                                        output_dict=True, zero_division=0)

    # Medir tiempo de inferencia (Fase 4)
    print("[Fase 4] Midiendo tiempo de inferencia...")
    metricas_inf = medir_tiempo_inferencia(modelo, loader_test, device)
    print("[Diagnostico] Perfilando carga, planes y ejecucion...")
    perfil_rendimiento = medir_perfil_rendimiento(
        modelo, loader_test, device, n_lotes=args.lotes_perfil,
    )

    # Tamano del modelo en MB
    tamano_mb = ckpt_path.stat().st_size / 1e6

    # VRAM pico total
    vram_pico_mb = (
        max(
            vram_pico_acumulada_mb,
            torch.cuda.max_memory_allocated(device) / 1e6,
        )
        if device.type == "cuda"
        else 0.0
    )

    print("\n" + "=" * 60)
    print(f"  {nombre_backend} COMPLETADO — R={R}^3")
    print("=" * 60)
    print(f"  Tiempo total    : {t_total:.1f} min")
    print(f"  Mejor Val Acc   : {mejor_val_acc*100:.2f}%  (ep {ckpt['epoca']})")
    print(f"  Test  Acc final : {test_acc*100:.2f}%")
    print(f"  Inf/muestra     : {metricas_inf['tiempo_inferencia_promedio_ms']:.3f} ms")
    print(f"  Pipeline/muestra: {metricas_inf['tiempo_pipeline_promedio_ms']:.3f} ms")
    print(f"  VRAM pico       : {vram_pico_mb:.1f} MB")
    print(f"  Tamano modelo   : {tamano_mb:.2f} MB")
    print(
        "  Perfil forward  : "
        f"{perfil_rendimiento['forward_total_ms_por_muestra']:.1f} ms/muestra "
        f"(planes {perfil_rendimiento['planes_backend_ms_por_muestra']:.1f}; "
        f"resto {perfil_rendimiento['resto_forward_ms_por_muestra']:.1f})"
    )
    print(
        "  Carga/transfer. : "
        f"{perfil_rendimiento['carga_lote_ms_por_muestra']:.1f} / "
        f"{perfil_rendimiento['transferencia_atributos_ms_por_muestra']:.1f} "
        "ms/muestra"
    )
    print("=" * 60)

    limites_activos = any(
        limite is not None
        for limite in (args.limite_train, args.limite_val, args.limite_test)
    )
    es_nativo = args.backend == "octree_native"
    test_oficial_completo = len(loader_test.dataset) == 2468
    resultado_oficial = bool(
        es_nativo
        and not limites_activos
        and test_oficial_completo
        and estado_git_inicial.get("git_dirty") is False
        and estado_git_inicial.get("git_commit_publicado") is True
    )
    if resultado_oficial:
        motivo_no_valido = None
    elif not es_nativo:
        motivo_no_valido = (
            "Usa nn.Conv3d sobre una rejilla densa; no implementa las "
            "operaciones OctNet sobre el grid-octree."
        )
    elif limites_activos:
        motivo_no_valido = (
            "Corrida nativa limitada; sirve como smoke test, no como "
            "evaluacion completa."
        )
    elif estado_git_inicial.get("git_dirty") is not False:
        motivo_no_valido = (
            "La corrida no parte de un arbol de trabajo Git limpio."
        )
    elif estado_git_inicial.get("git_commit_publicado") is not True:
        motivo_no_valido = (
            "El commit usado no aparece en una referencia remota conocida."
        )
    else:
        motivo_no_valido = (
            "El test no contiene las 2.468 muestras oficiales completas."
        )
    resumen = {
        "schema_name": (
            "net5-octree-native" if es_nativo else "dense-table5-diagnostic"
        ),
        "schema_version": SCHEMA_RESUMEN_VERSION,
        "backend_version": "1.1.1",
        "alcance": (
            "PARCIAL_SMOKE"
            if limites_activos
            else (
                "COMPLETO"
                if resultado_oficial
                else ("INCOMPLETO" if es_nativo else "COMPLETO_DIAGNOSTICO")
            )
        ),
        "valido_como_resultado_objetivo3": resultado_oficial,
        "motivo_no_valido": motivo_no_valido,
        "backend": args.backend,
        "entorno_ejecucion": entorno_ejecucion,
        **estado_git_inicial,
        "resolucion": R,
        "profundidad_octree": 5 if R == 32 else 6,
        "parametros_entrenables": sum(
            p.numel() for p in modelo.parameters() if p.requires_grad
        ),
        "seed": SEED,
        "particion": {
            "manifest": _ruta_portable(args.particion_manifest),
            "raiz_datos": _ruta_portable(raiz_resolucion),
            "metodo": particion["metodo"],
            "n_train_total": particion["n_train_total"],
            "n_train_usado": len(loader_train.dataset),
            "n_val_usado": len(loader_val.dataset),
            "n_test_usado": len(loader_test.dataset),
        },
        "configuracion": {
            "epochs_max": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": (
                1
                if args.backend == "octree_native" and args.num_workers > 0
                else None
            ),
            "hilos_bibliotecas_cpu": {
                variable: os.environ.get(variable)
                for variable in VARIABLES_HILOS_CPU
            },
            "persistent_workers": (
                {
                    "train": False,
                    "val": False,
                    "test": args.num_workers > 0,
                }
                if es_nativo
                else None
            ),
            "lr": args.lr,
            "weight_decay": 1e-4,
            "scheduler": "StepLR(step_size=20, gamma=0.7)",
            "lotes_perfil": args.lotes_perfil,
            "exigir_git_limpio": args.exigir_git_limpio,
            "exigir_git_publicado": args.exigir_git_publicado,
            "precalcular_planes": (
                args.precalcular_planes if es_nativo else None
            ),
        },
        "entrenamiento": {
            "reanudar_solicitado": args.reanudar,
            "reanudado_desde_epoca": reanudado_desde_epoca,
            "epocas_ejecutadas_esta_invocacion": (
                epocas_ejecutadas_esta_invocacion
            ),
            "epocas_completadas": len(historial),
            "detenido_por_early_stopping": (
                epocas_sin_mejora >= args.patience
            ),
            "checkpoint_mejor": ckpt_path.name,
            "checkpoint_ultimo": ckpt_ultimo_path.name,
            "historial_csv": csv_path.name,
            "historial_json": json_path.name,
            "particion_sha256": contrato_reanudacion["particion_sha256"],
        },
        "mejor_val_acc":  round(float(mejor_val_acc), 6),
        "mejor_epoca":    int(ckpt["epoca"]),
        "test_acc":       round(float(test_acc), 6),
        "test_loss":      round(float(test_loss), 6),
        "tiempo_total_min": round(t_total, 2),
        "tamano_modelo_mb": round(tamano_mb, 2),
        "vram_pico_mb":     round(vram_pico_mb, 1),
        "perfil_rendimiento": perfil_rendimiento,
        "matriz_confusion":          matriz_conf,
        "reporte_clasificacion":     reporte_cls,
        **metricas_inf,
    }

    salida = args.resultados_dir / f"resumen_{prefijo}_R{R}{args.tag}.json"
    with salida.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(resumen, f, indent=2)
        f.write("\n")

    return resumen


if __name__ == "__main__":
    main()
