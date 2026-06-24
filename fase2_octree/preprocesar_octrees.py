"""
preprocesar_octrees.py - Fase 2
=================================
Convierte TODO el dataset ModelNet40 (train + test) a representaciones
de octree en ambas resoluciones (32^3 y 64^3) y las guarda en disco como
archivos .npy comprimidos, para que la Fase 3 (entrenamiento) no tenga
que reconstruir octrees en cada epoca.

Estructura de salida:
    data/octrees_32/<clase>/<split>/<archivo>.npz
    data/octrees_64/<clase>/<split>/<archivo>.npz

Cada .npz contiene:
    grid     : array (4, R, R, R) float32
    etiqueta : int (indice de clase)

Uso:
    python preprocesar_octrees.py
"""

import sys
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from octree import malla_a_octree, profundidad_de

# ── Configuracion ──────────────────────────────────────────────
RAIZ_DATASET = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\Dataset\ModelNet40")
RAIZ_SALIDA  = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\data")
RESOLUCIONES = [32, 64]
N_PUNTOS_MUESTREO = 20000
SEED = 42
N_PROCESOS = 10  # Ryzen 7 5700X: 8 nucleos/16 hilos. Dejamos algo de margen para el sistema.

CLASES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf",
    "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person",
    "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent",
    "toilet", "tv_stand", "vase", "wardrobe", "xbox",
]
CLASE2IDX = {c: i for i, c in enumerate(CLASES)}


# ──────────────────────────────────────────────────────────────
# RECOLECCION DE ARCHIVOS
# ──────────────────────────────────────────────────────────────

def recolectar_archivos(raiz: Path, split: str) -> list:
    """Retorna lista de (ruta_off, etiqueta_int, nombre_archivo)."""
    muestras = []
    for clase in CLASES:
        carpeta = raiz / clase / split
        if not carpeta.exists():
            continue
        etiqueta = CLASE2IDX[clase]
        for archivo in sorted(carpeta.glob("*.off")):
            muestras.append((str(archivo), etiqueta, archivo.stem, clase))
    return muestras


# ──────────────────────────────────────────────────────────────
# PROCESAMIENTO DE UNA MUESTRA (ejecutado en worker process)
# ──────────────────────────────────────────────────────────────

def procesar_una_muestra(args: tuple) -> tuple:
    """
    Procesa un archivo .off: genera grids para todas las resoluciones
    y los guarda en disco. Disenado para ejecutarse en un ProcessPoolExecutor.

    Retorna (nombre_archivo, exito: bool, error: str|None)
    """
    ruta_off, etiqueta, nombre, clase, split, idx_global = args

    try:
        for R in RESOLUCIONES:
            grid = malla_a_octree(
                ruta_off, resolucion=R,
                n_puntos_muestreo=N_PUNTOS_MUESTREO,
                seed=SEED, idx_muestra=idx_global,
            )

            dir_salida = RAIZ_SALIDA / f"octrees_{R}" / clase / split
            dir_salida.mkdir(parents=True, exist_ok=True)
            ruta_salida = dir_salida / f"{nombre}.npz"

            np.savez_compressed(ruta_salida, grid=grid, etiqueta=etiqueta)

        return (nombre, True, None)

    except Exception as e:
        return (nombre, False, str(e))


# ──────────────────────────────────────────────────────────────
# MAIN: procesar train y test en paralelo
# ──────────────────────────────────────────────────────────────

def procesar_split(split: str):
    print(f"\n{'='*60}")
    print(f"  Procesando split: {split.upper()}")
    print(f"{'='*60}")

    muestras = recolectar_archivos(RAIZ_DATASET, split)
    print(f"  Archivos encontrados: {len(muestras)}")

    if len(muestras) == 0:
        print(f"  ADVERTENCIA: no se encontraron archivos para '{split}'")
        return [], []

    # Preparar argumentos con indice global (para seed reproducible)
    tareas = [
        (ruta, etiqueta, nombre, clase, split, i)
        for i, (ruta, etiqueta, nombre, clase) in enumerate(muestras)
    ]

    exitosos = []
    fallidos  = []

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_PROCESOS) as executor:
        futuros = {executor.submit(procesar_una_muestra, t): t for t in tareas}

        barra = tqdm(as_completed(futuros), total=len(futuros), ncols=80, desc=f"  {split}")
        for futuro in barra:
            nombre, exito, error = futuro.result()
            if exito:
                exitosos.append(nombre)
            else:
                fallidos.append((nombre, error))

    t1 = time.time()

    print(f"\n  Completado en {(t1-t0)/60:.1f} min")
    print(f"  Exitosos : {len(exitosos)}")
    print(f"  Fallidos : {len(fallidos)}")

    if fallidos:
        print("\n  Primeros errores:")
        for nombre, error in fallidos[:5]:
            print(f"    {nombre}: {error}")

    return exitosos, fallidos


def main():
    print("=" * 60)
    print("  FASE 2: PREPROCESAMIENTO DE OCTREES (32^3 y 64^3)")
    print("=" * 60)
    print(f"  Dataset origen : {RAIZ_DATASET}")
    print(f"  Salida         : {RAIZ_SALIDA}")
    print(f"  Resoluciones   : {RESOLUCIONES}")
    print(f"  Puntos muestreo: {N_PUNTOS_MUESTREO}")
    print(f"  Procesos       : {N_PROCESOS}")

    resumen = {}
    for split in ["train", "test"]:
        exitosos, fallidos = procesar_split(split)
        resumen[split] = {"exitosos": len(exitosos), "fallidos": len(fallidos)}

    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    for split, datos in resumen.items():
        print(f"  {split:10s}: {datos['exitosos']} exitosos, {datos['fallidos']} fallidos")
    print(f"\n  Archivos guardados en: {RAIZ_SALIDA}")
    print("  Estructura: data/octrees_<R>/<clase>/<split>/<archivo>.npz")
    print("=" * 60)


if __name__ == "__main__":
    main()
