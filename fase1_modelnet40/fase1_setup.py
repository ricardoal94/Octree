"""
Fase 1: Definicion del caso de estudio y control de estocasticidad
=================================================================
Dataset   : ModelNet40 (particion oficial 9.843 train / 2.468 test)
Validacion: 10% del train reservado para validacion interna
Semilla   : seed=42 para pesos, shuffle y bosque aleatorio
Registro  : config.yaml + experiment_log.json generados automaticamente
"""

import os
import sys
import json
import random
import platform
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# 1.  SEMILLA GLOBAL  (seed = 42)
# ──────────────────────────────────────────────────────────────

SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """
    Fija seed en Python, NumPy, PyTorch (CPU y GPU) y CUDNN.
    Garantiza reproducibilidad completa del experimento.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # multi-GPU

    # Reproducibilidad CUDNN (a costa de velocidad)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # Variable de entorno para operaciones atomicas de CUDA
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[Reproducibilidad] Semilla global fijada: seed={seed}")


# ──────────────────────────────────────────────────────────────
# 2.  CARGA DE CONFIGURACION YAML
# ──────────────────────────────────────────────────────────────

def cargar_config(ruta: str = "config.yaml") -> dict:
    """Lee el archivo YAML y retorna el diccionario de configuracion."""
    with open(ruta, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[Config] Cargado: {ruta}")
    return cfg


# ──────────────────────────────────────────────────────────────
# 3.  DETECCION DE HARDWARE Y VERSIONES
# ──────────────────────────────────────────────────────────────

def detectar_hardware() -> dict:
    """Recopila informacion del hardware y versiones de librerias."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_info = {}

    if device == "cuda":
        gpu_info = {
            "nombre": torch.cuda.get_device_name(0),
            "memoria_total_GB": round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            ),
            "cuda_version": torch.version.cuda,
            "num_gpus": torch.cuda.device_count(),
        }

    # Versiones de librerias instaladas
    versiones = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }

    # Intentar obtener versiones opcionales
    for lib in ["sklearn", "yaml", "tqdm", "open3d"]:
        try:
            mod = __import__(lib)
            versiones[lib] = getattr(mod, "__version__", "instalado")
        except ImportError:
            versiones[lib] = "no instalado"

    hardware = {
        "sistema_operativo": platform.system(),
        "arquitectura": platform.machine(),
        "procesador": platform.processor(),
        "dispositivo_entrenamiento": device,
        "gpu": gpu_info,
        "versiones_librerias": versiones,
    }

    return hardware


# ──────────────────────────────────────────────────────────────
# 4.  PARTICION DE DATOS  (train / val / test)
# ──────────────────────────────────────────────────────────────

def particionar_dataset(
    n_train_total: int = 9843,
    n_test: int = 2468,
    val_split: float = 0.10,
    seed: int = SEED,
) -> dict:
    """
    Crea los indices de particion del dataset ModelNet40.

    Parametros
    ----------
    n_train_total : total de muestras de entrenamiento oficiales
    n_test        : total de muestras de prueba oficiales
    val_split     : fraccion del train reservada para validacion
    seed          : semilla para el shuffle reproducible

    Retorna
    -------
    dict con indices y conteos de cada particion
    """
    rng = np.random.default_rng(seed)          # generador NumPy reproducible

    # Indices del conjunto de entrenamiento oficial
    indices_train_total = np.arange(n_train_total)
    rng.shuffle(indices_train_total)           # barajado reproducible

    # Calculo del corte
    n_val = int(np.floor(val_split * n_train_total))
    n_train = n_train_total - n_val

    idx_val   = indices_train_total[:n_val]
    idx_train = indices_train_total[n_val:]
    idx_test  = np.arange(n_test)

    particion = {
        "train": {
            "indices": idx_train.tolist(),
            "n_muestras": int(n_train),
            "porcentaje": round(n_train / n_train_total * 100, 1),
        },
        "val": {
            "indices": idx_val.tolist(),
            "n_muestras": int(n_val),
            "porcentaje": round(n_val / n_train_total * 100, 1),
        },
        "test": {
            "indices": idx_test.tolist(),
            "n_muestras": int(n_test),
            "porcentaje": 100.0,
        },
    }

    print(
        f"\n[Dataset] Particion ModelNet40 (seed={seed}):\n"
        f"  Train  : {n_train:>5} muestras  ({particion['train']['porcentaje']}%)\n"
        f"  Val    : {n_val:>5} muestras  ({particion['val']['porcentaje']}%)\n"
        f"  Test   : {n_test:>5} muestras  (conjunto oficial)\n"
        f"  Total  : {n_train + n_val + n_test:>5} muestras\n"
    )

    return particion


# ──────────────────────────────────────────────────────────────
# 5.  VERIFICACION DE INTEGRIDAD
# ──────────────────────────────────────────────────────────────

def verificar_particion(particion: dict) -> bool:
    """
    Comprueba que no haya fugas de datos entre train y val.
    Retorna True si la particion es valida.
    """
    idx_train = set(particion["train"]["indices"])
    idx_val   = set(particion["val"]["indices"])

    solapamiento = idx_train & idx_val
    valido = len(solapamiento) == 0

    estado = "VALIDA" if valido else f"ERROR - {len(solapamiento)} indices solapados"
    print(f"[Verificacion] Integridad de particion: {estado}")
    return valido


# ──────────────────────────────────────────────────────────────
# 6.  REGISTRO DEL EXPERIMENTO  (JSON)
# ──────────────────────────────────────────────────────────────

def guardar_registro(
    cfg: dict,
    hardware: dict,
    particion: dict,
    ruta_log: str = "logs/experiment_log.json",
) -> None:
    """
    Genera un archivo JSON con toda la informacion del experimento:
    configuracion, hardware, versiones y particion de datos.
    """
    Path(ruta_log).parent.mkdir(parents=True, exist_ok=True)

    registro = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "version_experimento": "1.0.0",
            "fase": 1,
        },
        "configuracion": cfg,
        "hardware": hardware,
        "particion": {
            "seed_utilizada": SEED,
            "val_split": cfg["dataset"]["val_split"],
            "train_n": particion["train"]["n_muestras"],
            "val_n":   particion["val"]["n_muestras"],
            "test_n":  particion["test"]["n_muestras"],
        },
    }

    # Guardar JSON con indentacion legible
    with open(ruta_log, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)

    print(f"[Registro] Experimento guardado en: {ruta_log}")


# ──────────────────────────────────────────────────────────────
# 7.  DIRECTORIOS DEL PROYECTO
# ──────────────────────────────────────────────────────────────

def crear_estructura_directorios(cfg: dict) -> None:
    """Crea las carpetas necesarias del proyecto si no existen."""
    rutas = cfg.get("rutas", {})
    for nombre, ruta in rutas.items():
        Path(ruta).mkdir(parents=True, exist_ok=True)
    print(f"[Estructura] Directorios creados: {list(rutas.values())}")


# ──────────────────────────────────────────────────────────────
# 8.  PUNTO DE ENTRADA PRINCIPAL
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FASE 1: DEFINICION DEL CASO DE ESTUDIO")
    print("  Dataset: ModelNet40  |  Seed: 42")
    print("=" * 60)

    # 1. Fijar semilla global
    set_global_seed(SEED)

    # 2. Cargar configuracion
    cfg_path = Path(__file__).parent / "config.yaml"
    cfg = cargar_config(str(cfg_path))

    # 3. Detectar hardware y versiones
    hardware = detectar_hardware()
    print(
        f"\n[Hardware] Dispositivo: {hardware['dispositivo_entrenamiento'].upper()}"
    )
    if hardware["gpu"]:
        print(f"           GPU: {hardware['gpu']['nombre']}")
    print(f"           Python {hardware['versiones_librerias']['python']} | "
          f"PyTorch {hardware['versiones_librerias']['torch']}")

    # 4. Crear estructura de carpetas
    crear_estructura_directorios(cfg)

    # 5. Particionar dataset
    particion = particionar_dataset(
        n_train_total=cfg["dataset"]["train_total"],
        n_test=cfg["dataset"]["test_total"],
        val_split=cfg["dataset"]["val_split"],
        seed=SEED,
    )

    # 6. Verificar integridad
    assert verificar_particion(particion), "Error en la particion del dataset"

    # 7. Guardar registro JSON
    log_path = Path(cfg["rutas"]["logs"]) / "experiment_log.json"
    guardar_registro(cfg, hardware, particion, str(log_path))

    # 8. Guardar indices de particion para reproducibilidad
    particion_path = Path(cfg["rutas"]["logs"]) / "particion_indices.npz"
    np.savez(
        str(particion_path),
        idx_train=np.array(particion["train"]["indices"]),
        idx_val=np.array(particion["val"]["indices"]),
        idx_test=np.array(particion["test"]["indices"]),
    )
    print(f"[Registro] Indices guardados en: {particion_path}")

    print("\n" + "=" * 60)
    print("  Fase 1 completada exitosamente.")
    print("  Proxima fase: Implementacion de Net5 (PointNet)")
    print("=" * 60)

    return cfg, hardware, particion


if __name__ == "__main__":
    cfg, hardware, particion = main()
