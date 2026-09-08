"""
medir_metricas_off.py
======================
Procesa todos los archivos .off de ModelNet40 y calcula las metricas
de ocupacion, tiempo de procesamiento y memoria requeridas para la
tabla del capitulo de metodologia.

Metricas calculadas por objeto:
  - Numero de vertices y caras de la malla original
  - Tiempo de normalizacion + muestreo de superficie (ms)
  - Tiempo de cuantizacion a grid 32^3 (ms)
  - Tiempo de cuantizacion a grid 64^3 (ms)
  - Celdas ocupadas en 32^3 (absoluto y porcentaje)
  - Celdas ocupadas en 64^3 (absoluto y porcentaje)
  - Memoria del grid 32^3 (KB, float32)
  - Memoria del grid 64^3 (KB, float32)

Salida:
  - resultados/metricas_off_completo.csv   (todas las muestras)
  - resultados/metricas_off_resumen.csv    (estadisticas por clase)
  - resultados/metricas_off_global.json    (estadisticas globales)

Uso:
    python medir_metricas_off.py
    python medir_metricas_off.py --n_muestras 100   # muestra rapida
    python medir_metricas_off.py --split train
"""

import sys
import csv
import json
import time
import argparse
import tracemalloc
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

RAIZ_DATASET = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\Dataset\ModelNet40")
DIR_RESULTADOS = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\resultados")
N_PUNTOS_MUESTREO = 20000
N_PROCESOS = 10
SEED = 42

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


# ──────────────────────────────────────────────────────────────
# PIPELINE GEOMETRICO (sin dependencias externas)
# ──────────────────────────────────────────────────────────────

def leer_off(ruta: str) -> tuple:
    with open(ruta) as f:
        lineas = f.read().splitlines()
    inicio = 1 if lineas[0].strip().upper() == "OFF" else 0
    if lineas[0].strip().upper().startswith("OFF") and lineas[0].strip().upper() != "OFF":
        lineas[0] = lineas[0][3:].strip(); inicio = 0
    n_v, n_c, _ = map(int, lineas[inicio].split())
    verts = np.array([list(map(float, lineas[inicio+1+i].split()[:3]))
                      for i in range(n_v)], dtype=np.float32)
    caras = []
    for i in range(n_c):
        t = list(map(int, lineas[inicio+1+n_v+i].split()))
        if len(t) >= 4:
            caras.append(t[1:4])
    caras = np.array(caras, dtype=np.int64) if caras else None
    # Saneo indices fuera de rango
    if caras is not None:
        validas = ((caras >= 0) & (caras < n_v)).all(axis=1)
        caras = caras[validas]
    return verts, caras, n_v, n_c


def normalizar(verts: np.ndarray) -> np.ndarray:
    v = verts - verts.mean(axis=0)
    s = np.max(np.abs(v))
    return v / s if s > 0 else v


def muestrear_superficie(verts, caras, n, seed):
    if caras is None or len(caras) == 0:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(verts), n, replace=len(verts) < n)
        return verts[idx]
    rng = np.random.default_rng(seed)
    v0, v1, v2 = verts[caras[:,0]], verts[caras[:,1]], verts[caras[:,2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)
    areas = np.nan_to_num(np.clip(areas, 0, None))
    s = areas.sum()
    if s <= 0:
        return verts[:n]
    probs = areas / s
    idx = rng.choice(len(caras), n, p=probs)
    r1 = rng.random(n).astype(np.float32)
    r2 = rng.random(n).astype(np.float32)
    sq = np.sqrt(r1)
    return ((1-sq)[:,None]*v0[idx] + (sq*(1-r2))[:,None]*v1[idx]
            + (sq*r2)[:,None]*v2[idx])


def voxelizar(pts, R):
    idx = np.clip(((pts + 1.0) * 0.5 * R).astype(np.int64), 0, R-1)
    g = np.zeros((R, R, R), dtype=np.float32)
    g[idx[:,0], idx[:,1], idx[:,2]] = 1.0
    return g


# ──────────────────────────────────────────────────────────────
# PROCESAMIENTO DE UN ARCHIVO
# ──────────────────────────────────────────────────────────────

def procesar_archivo(args: tuple) -> dict:
    ruta, clase, split = args
    nombre = Path(ruta).stem

    try:
        # Lectura
        verts_raw, caras, n_v, n_c = leer_off(ruta)

        # Normalizacion + muestreo
        t0 = time.perf_counter()
        verts = normalizar(verts_raw)
        pts = muestrear_superficie(verts, caras, N_PUNTOS_MUESTREO, SEED)
        t_preproceso_ms = (time.perf_counter() - t0) * 1000

        # Voxelizacion 32^3
        t0 = time.perf_counter()
        g32 = voxelizar(pts, 32)
        t_vox32_ms = (time.perf_counter() - t0) * 1000

        # Voxelizacion 64^3
        t0 = time.perf_counter()
        g64 = voxelizar(pts, 64)
        t_vox64_ms = (time.perf_counter() - t0) * 1000

        # Metricas de ocupacion
        ocup32 = int(g32.sum())
        ocup64 = int(g64.sum())
        pct32 = round(100 * ocup32 / g32.size, 4)
        pct64 = round(100 * ocup64 / g64.size, 4)

        # Memoria de los grids (float32 = 4 bytes por celda)
        mem32_kb = round(g32.size * 4 / 1024, 2)   # 32^3 * 4 = 131 KB
        mem64_kb = round(g64.size * 4 / 1024, 2)   # 64^3 * 4 = 1024 KB = 1 MB

        return {
            "nombre": nombre,
            "clase": clase,
            "split": split,
            "n_vertices": n_v,
            "n_caras": n_c,
            "t_preproceso_ms": round(t_preproceso_ms, 3),
            "t_vox32_ms":      round(t_vox32_ms, 3),
            "t_vox64_ms":      round(t_vox64_ms, 3),
            "t_total_ms":      round(t_preproceso_ms + t_vox32_ms + t_vox64_ms, 3),
            "ocup32_abs":  ocup32,
            "ocup32_pct":  pct32,
            "ocup64_abs":  ocup64,
            "ocup64_pct":  pct64,
            "mem32_kb":    mem32_kb,
            "mem64_kb":    mem64_kb,
            "error":       None,
        }

    except Exception as e:
        return {
            "nombre": nombre, "clase": clase, "split": split,
            "error": str(e),
        }


# ──────────────────────────────────────────────────────────────
# RECOLECTAR ARCHIVOS
# ──────────────────────────────────────────────────────────────

def recolectar(split: str, n_muestras: int = None) -> list:
    tareas = []
    for clase in CLASES:
        carpeta = RAIZ_DATASET / clase / split
        if not carpeta.exists():
            continue
        archivos = sorted(carpeta.glob("*.off"))
        if n_muestras:
            # Tomar proporcional al tamano de la clase
            archivos = archivos[:max(1, n_muestras // len(CLASES))]
        for f in archivos:
            tareas.append((str(f), clase, split))
    return tareas


# ──────────────────────────────────────────────────────────────
# ESTADISTICAS RESUMEN
# ──────────────────────────────────────────────────────────────

def calcular_resumen(resultados: list) -> dict:
    """Calcula estadisticas globales sobre todos los resultados validos."""
    validos = [r for r in resultados if r.get("error") is None]
    if not validos:
        return {}

    campos_numericos = [
        "n_vertices", "n_caras", "t_preproceso_ms", "t_vox32_ms",
        "t_vox64_ms", "t_total_ms", "ocup32_abs", "ocup32_pct",
        "ocup64_abs", "ocup64_pct",
    ]

    resumen = {"n_total": len(validos), "n_errores": len(resultados) - len(validos)}
    for campo in campos_numericos:
        vals = np.array([r[campo] for r in validos])
        resumen[campo] = {
            "media":   round(float(vals.mean()), 4),
            "mediana": round(float(np.median(vals)), 4),
            "min":     round(float(vals.min()), 4),
            "max":     round(float(vals.max()), 4),
            "std":     round(float(vals.std()), 4),
        }

    # Memoria fija (igual para todos)
    resumen["mem32_kb_por_objeto"] = round(32**3 * 4 / 1024, 2)
    resumen["mem64_kb_por_objeto"] = round(64**3 * 4 / 1024, 2)
    resumen["mem32_dataset_mb"] = round(len(validos) * 32**3 * 4 / 1024**2, 1)
    resumen["mem64_dataset_mb"] = round(len(validos) * 64**3 * 4 / 1024**2, 1)

    return resumen


def calcular_resumen_por_clase(resultados: list) -> dict:
    """Calcula metricas promedio por clase para la tabla del capitulo."""
    validos = [r for r in resultados if r.get("error") is None]
    por_clase = {}
    for clase in CLASES:
        muestras = [r for r in validos if r["clase"] == clase]
        if not muestras:
            continue
        por_clase[clase] = {
            "n": len(muestras),
            "vertices_media":   round(np.mean([r["n_vertices"] for r in muestras]), 0),
            "caras_media":      round(np.mean([r["n_caras"]    for r in muestras]), 0),
            "t_total_media_ms": round(np.mean([r["t_total_ms"] for r in muestras]), 2),
            "ocup32_pct_media": round(np.mean([r["ocup32_pct"] for r in muestras]), 3),
            "ocup64_pct_media": round(np.mean([r["ocup64_pct"] for r in muestras]), 3),
        }
    return por_clase


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test", "ambos"])
    parser.add_argument("--n_muestras", type=int, default=None,
                        help="Limitar el total de muestras (para prueba rapida)")
    args = parser.parse_args()

    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    splits = ["train", "test"] if args.split == "ambos" else [args.split]

    print("=" * 60)
    print("  METRICAS DE PROCESAMIENTO .off -> Voxel Grid")
    print("=" * 60)

    todos_resultados = []

    for split in splits:
        tareas = recolectar(split, args.n_muestras)
        print(f"\n[{split.upper()}] {len(tareas)} archivos a procesar...")

        t_inicio = time.time()
        with ProcessPoolExecutor(max_workers=N_PROCESOS) as executor:
            futuros = {executor.submit(procesar_archivo, t): t for t in tareas}
            barra = tqdm(as_completed(futuros), total=len(futuros), ncols=80)
            for futuro in barra:
                todos_resultados.append(futuro.result())
        t_total = time.time() - t_inicio

        n_ok  = sum(1 for r in todos_resultados if r.get("error") is None and r["split"] == split)
        n_err = sum(1 for r in todos_resultados if r.get("error") is not None and r["split"] == split)
        print(f"  Completado en {t_total:.1f}s | OK: {n_ok} | Errores: {n_err}")

    # Guardar CSV completo
    validos = [r for r in todos_resultados if r.get("error") is None]
    csv_completo = DIR_RESULTADOS / "metricas_off_completo.csv"
    campos = ["nombre","clase","split","n_vertices","n_caras",
              "t_preproceso_ms","t_vox32_ms","t_vox64_ms","t_total_ms",
              "ocup32_abs","ocup32_pct","ocup64_abs","ocup64_pct",
              "mem32_kb","mem64_kb"]
    with open(csv_completo, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for r in validos:
            w.writerow({k: r.get(k,"") for k in campos})
    print(f"\n[CSV] Guardado: {csv_completo.name} ({len(validos)} filas)")

    # Guardar CSV resumen por clase
    resumen_clase = calcular_resumen_por_clase(todos_resultados)
    csv_resumen = DIR_RESULTADOS / "metricas_off_resumen.csv"
    with open(csv_resumen, "w", newline="", encoding="utf-8") as f:
        campos_r = ["clase","n","vertices_media","caras_media",
                    "t_total_media_ms","ocup32_pct_media","ocup64_pct_media"]
        w = csv.DictWriter(f, fieldnames=campos_r)
        w.writeheader()
        for clase, datos in resumen_clase.items():
            w.writerow({"clase": clase, **datos})
    print(f"[CSV] Guardado: {csv_resumen.name}")

    # Guardar JSON global
    resumen_global = calcular_resumen(todos_resultados)
    json_global = DIR_RESULTADOS / "metricas_off_global.json"
    with open(json_global, "w") as f:
        json.dump(resumen_global, f, indent=2)
    print(f"[JSON] Guardado: {json_global.name}")

    # Imprimir resumen en consola
    print("\n" + "=" * 60)
    print("  RESUMEN GLOBAL")
    print("=" * 60)
    print(f"  Objetos procesados      : {resumen_global.get('n_total', 0):,}")
    print(f"  Errores                 : {resumen_global.get('n_errores', 0)}")

    if "n_vertices" in resumen_global:
        print(f"\n  Vertices por malla      : "
              f"media={resumen_global['n_vertices']['media']:.0f}, "
              f"min={resumen_global['n_vertices']['min']:.0f}, "
              f"max={resumen_global['n_vertices']['max']:.0f}")
        print(f"  Ocupacion 32^3 (%)      : "
              f"media={resumen_global['ocup32_pct']['media']:.2f}%, "
              f"min={resumen_global['ocup32_pct']['min']:.2f}%, "
              f"max={resumen_global['ocup32_pct']['max']:.2f}%")
        print(f"  Ocupacion 64^3 (%)      : "
              f"media={resumen_global['ocup64_pct']['media']:.2f}%, "
              f"min={resumen_global['ocup64_pct']['min']:.2f}%, "
              f"max={resumen_global['ocup64_pct']['max']:.2f}%")
        print(f"  Tiempo total/objeto (ms): "
              f"media={resumen_global['t_total_ms']['media']:.1f}, "
              f"max={resumen_global['t_total_ms']['max']:.1f}")
        print(f"\n  Memoria 32^3/objeto     : {resumen_global['mem32_kb_por_objeto']} KB")
        print(f"  Memoria 64^3/objeto     : {resumen_global['mem64_kb_por_objeto']} KB")
        print(f"  Memoria 32^3 dataset    : {resumen_global['mem32_dataset_mb']} MB")
        print(f"  Memoria 64^3 dataset    : {resumen_global['mem64_dataset_mb']} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
