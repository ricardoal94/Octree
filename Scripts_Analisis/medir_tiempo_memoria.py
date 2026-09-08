"""
medir_tiempo_memoria.py
========================
Mide el tiempo de construccion y memoria estimada de cada etapa
del pipeline de conversion .off -> octree, para un archivo .off dado.

Etapas medidas:
  1. Lectura del archivo .off
  2. Normalizacion geometrica
  3. Muestreo de superficie (area-weighted)
  4. Cuantizacion a grid 32^3
  5. Cuantizacion a grid 64^3
  6. Pipeline completo (1+2+3+4+5)

Memoria medida:
  - Memoria del array de vertices (bytes)
  - Memoria del array de caras (bytes)
  - Memoria de la nube de puntos muestreada (bytes)
  - Memoria del grid 32^3 float32 (bytes)
  - Memoria del grid 64^3 float32 (bytes)
  - Memoria pico del proceso durante cada etapa (tracemalloc)

Salida:
  - Tabla en consola con todas las mediciones
  - JSON con los resultados (para incluir en la tesis)

Uso:
    python medir_tiempo_memoria.py --off chair_0001.off
    python medir_tiempo_memoria.py --off chair_0001.off --repeticiones 20
"""

import argparse
import json
import sys
import time
import tracemalloc
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# PIPELINE GEOMETRICO
# ──────────────────────────────────────────────────────────────

def leer_off(ruta: str) -> tuple:
    with open(ruta) as f:
        lineas = f.read().splitlines()
    inicio = 1 if lineas[0].strip().upper() == "OFF" else 0
    if lineas[0].strip().upper().startswith("OFF") and lineas[0].strip().upper() != "OFF":
        lineas[0] = lineas[0][3:].strip(); inicio = 0
    n_v, n_c, _ = map(int, lineas[inicio].split())
    verts = np.array(
        [list(map(float, lineas[inicio + 1 + i].split()[:3])) for i in range(n_v)],
        dtype=np.float32,
    )
    caras = []
    for i in range(n_c):
        t = list(map(int, lineas[inicio + 1 + n_v + i].split()))
        if len(t) >= 4:
            caras.append(t[1:4])
    caras = np.array(caras, dtype=np.int64) if caras else np.zeros((0, 3), dtype=np.int64)
    validas = ((caras >= 0) & (caras < n_v)).all(axis=1)
    caras = caras[validas]
    return verts, caras, n_v, len(caras)


def normalizar(verts: np.ndarray) -> np.ndarray:
    v = verts - verts.mean(axis=0)
    s = np.max(np.abs(v))
    return v / s if s > 0 else v


def muestrear_superficie(verts, caras, n: int, seed: int = 42) -> np.ndarray:
    if len(caras) == 0:
        rng = np.random.default_rng(seed)
        return verts[rng.choice(len(verts), n, replace=len(verts) < n)]
    rng = np.random.default_rng(seed)
    v0, v1, v2 = verts[caras[:, 0]], verts[caras[:, 1]], verts[caras[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    areas = np.nan_to_num(np.clip(areas, 0, None))
    s = areas.sum()
    if s <= 0:
        return verts[:n]
    probs = areas / s
    idx = rng.choice(len(caras), n, p=probs)
    r1 = rng.random(n).astype(np.float32)
    r2 = rng.random(n).astype(np.float32)
    sq = np.sqrt(r1)
    return (
        (1 - sq)[:, None] * v0[idx]
        + (sq * (1 - r2))[:, None] * v1[idx]
        + (sq * r2)[:, None] * v2[idx]
    ).astype(np.float32)


def voxelizar(pts: np.ndarray, R: int) -> np.ndarray:
    idx = np.clip(((pts + 1.0) * 0.5 * R).astype(np.int64), 0, R - 1)
    g = np.zeros((R, R, R), dtype=np.float32)
    g[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return g


# ──────────────────────────────────────────────────────────────
# MEDICION
# ──────────────────────────────────────────────────────────────

def medir_etapa(fn, *args, repeticiones: int = 10) -> tuple:
    """
    Ejecuta fn(*args) `repeticiones` veces y retorna:
      (resultado, tiempo_medio_ms, tiempo_min_ms, tiempo_max_ms, mem_pico_kb)
    El tiempo se mide con time.perf_counter (alta resolucion).
    La memoria pico se mide con tracemalloc en la primera ejecucion.
    """
    # Medicion de memoria en la primera ejecucion
    tracemalloc.start()
    resultado = fn(*args)
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_pico_kb = pico / 1024

    # Medicion de tiempo (multiples repeticiones para estabilidad)
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        fn(*args)
        tiempos.append((time.perf_counter() - t0) * 1000)

    return (
        resultado,
        round(float(np.mean(tiempos)), 4),
        round(float(np.min(tiempos)), 4),
        round(float(np.max(tiempos)), 4),
        round(mem_pico_kb, 2),
    )


def memoria_array_kb(arr: np.ndarray) -> float:
    """Memoria real ocupada por un array numpy (KB)."""
    return round(arr.nbytes / 1024, 2)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mide tiempo y memoria del pipeline .off -> octree"
    )
    parser.add_argument("--off", type=str, required=True,
                        help="Ruta al archivo .off")
    parser.add_argument("--n_puntos", type=int, default=30000,
                        help="Puntos a muestrear sobre la superficie (default: 30000)")
    parser.add_argument("--repeticiones", type=int, default=10,
                        help="Repeticiones para estabilizar mediciones de tiempo (default: 10)")
    parser.add_argument("--salida", type=str, default=None,
                        help="Ruta JSON donde guardar los resultados (opcional)")
    args = parser.parse_args()

    ruta = args.off
    N    = args.n_puntos
    REP  = args.repeticiones

    print("=" * 65)
    print(f"  MEDICION DE TIEMPO Y MEMORIA — {Path(ruta).name}")
    print("=" * 65)
    print(f"  Puntos de muestreo : {N:,}")
    print(f"  Repeticiones       : {REP}")
    print()

    resultados = {}

    # ── Etapa 1: Lectura ──────────────────────────────────────
    (verts_raw, caras, n_v, n_c), t_med, t_min, t_max, mem_pico = \
        medir_etapa(leer_off, ruta, repeticiones=REP)

    mem_verts_kb = memoria_array_kb(verts_raw)
    mem_caras_kb = memoria_array_kb(caras)

    resultados["lectura"] = {
        "descripcion":   "Lectura del archivo .off",
        "n_vertices":    n_v,
        "n_caras":       n_c,
        "t_media_ms":    t_med,
        "t_min_ms":      t_min,
        "t_max_ms":      t_max,
        "mem_pico_kb":   mem_pico,
        "mem_vertices_kb": mem_verts_kb,
        "mem_caras_kb":    mem_caras_kb,
    }

    # ── Etapa 2: Normalizacion ────────────────────────────────
    verts_norm, t_med, t_min, t_max, mem_pico = \
        medir_etapa(normalizar, verts_raw, repeticiones=REP)

    resultados["normalizacion"] = {
        "descripcion": "Normalizacion al cubo [-1,1]^3",
        "t_media_ms":  t_med,
        "t_min_ms":    t_min,
        "t_max_ms":    t_max,
        "mem_pico_kb": mem_pico,
        "mem_salida_kb": memoria_array_kb(verts_norm),
    }

    # ── Etapa 3: Muestreo de superficie ───────────────────────
    pts, t_med, t_min, t_max, mem_pico = \
        medir_etapa(muestrear_superficie, verts_norm, caras, N,
                    repeticiones=REP)

    mem_pts_kb = memoria_array_kb(pts)

    resultados["muestreo"] = {
        "descripcion":   f"Muestreo area-weighted ({N:,} puntos)",
        "n_puntos":      N,
        "t_media_ms":    t_med,
        "t_min_ms":      t_min,
        "t_max_ms":      t_max,
        "mem_pico_kb":   mem_pico,
        "mem_nube_kb":   mem_pts_kb,
    }

    # ── Etapa 4: Voxelizacion 32^3 ────────────────────────────
    g32, t_med, t_min, t_max, mem_pico = \
        medir_etapa(voxelizar, pts, 32, repeticiones=REP)

    ocup32 = int(g32.sum())
    pct32  = round(100 * ocup32 / g32.size, 3)
    mem_g32_kb = memoria_array_kb(g32)

    resultados["voxel_32"] = {
        "descripcion":    "Cuantizacion a grid 32^3",
        "resolucion":     32,
        "total_celdas":   g32.size,
        "celdas_ocup":    ocup32,
        "pct_ocupacion":  pct32,
        "t_media_ms":     t_med,
        "t_min_ms":       t_min,
        "t_max_ms":       t_max,
        "mem_pico_kb":    mem_pico,
        "mem_grid_kb":    mem_g32_kb,
        "mem_grid_mb":    round(mem_g32_kb / 1024, 4),
    }

    # ── Etapa 5: Voxelizacion 64^3 ────────────────────────────
    g64, t_med, t_min, t_max, mem_pico = \
        medir_etapa(voxelizar, pts, 64, repeticiones=REP)

    ocup64 = int(g64.sum())
    pct64  = round(100 * ocup64 / g64.size, 3)
    mem_g64_kb = memoria_array_kb(g64)

    resultados["voxel_64"] = {
        "descripcion":    "Cuantizacion a grid 64^3",
        "resolucion":     64,
        "total_celdas":   g64.size,
        "celdas_ocup":    ocup64,
        "pct_ocupacion":  pct64,
        "t_media_ms":     t_med,
        "t_min_ms":       t_min,
        "t_max_ms":       t_max,
        "mem_pico_kb":    mem_pico,
        "mem_grid_kb":    mem_g64_kb,
        "mem_grid_mb":    round(mem_g64_kb / 1024, 4),
    }

    # ── Pipeline completo ────────────────────────────────────
    def pipeline_completo(ruta, N):
        v, c, _, _ = leer_off(ruta)
        v = normalizar(v)
        p = muestrear_superficie(v, c, N)
        voxelizar(p, 32)
        voxelizar(p, 64)

    _, t_med, t_min, t_max, mem_pico = \
        medir_etapa(pipeline_completo, ruta, N, repeticiones=REP)

    resultados["pipeline_completo"] = {
        "descripcion": "Pipeline completo (lectura + norm + muestreo + vox32 + vox64)",
        "t_media_ms":  t_med,
        "t_min_ms":    t_min,
        "t_max_ms":    t_max,
        "mem_pico_kb": mem_pico,
        "mem_pico_mb": round(mem_pico / 1024, 3),
    }

    # ── Imprimir tabla ────────────────────────────────────────
    print(f"  Archivo : {Path(ruta).name}")
    print(f"  Vertices: {n_v:,}   Caras: {n_c:,}")
    print()

    # Tabla de tiempos
    ancho = 45
    sep   = "-" * 75
    print(f"  {'ETAPA':<{ancho}} {'T.media':>9} {'T.min':>9} {'T.max':>9}  {'Mem.pico':>10}")
    print(f"  {'':.<{ancho}} {'(ms)':>9} {'(ms)':>9} {'(ms)':>9}  {'(KB)':>10}")
    print("  " + sep)

    etapas_print = [
        ("1. Lectura .off",              resultados["lectura"]),
        ("2. Normalizacion [-1,1]^3",    resultados["normalizacion"]),
        (f"3. Muestreo superficie ({N:,} pts)", resultados["muestreo"]),
        ("4. Cuantizacion 32^3",         resultados["voxel_32"]),
        ("5. Cuantizacion 64^3",         resultados["voxel_64"]),
        ("   PIPELINE COMPLETO",         resultados["pipeline_completo"]),
    ]

    for nombre, datos in etapas_print:
        sep_line = "  " + sep if nombre.startswith("   PIPELINE") else ""
        if sep_line:
            print(sep_line)
        print(
            f"  {nombre:<{ancho}} "
            f"{datos['t_media_ms']:>9.3f} "
            f"{datos['t_min_ms']:>9.3f} "
            f"{datos['t_max_ms']:>9.3f}  "
            f"{datos['mem_pico_kb']:>10.1f}"
        )

    # Tabla de memoria de los datos
    print()
    print(f"  MEMORIA DE ESTRUCTURAS DE DATOS")
    print("  " + sep)
    print(f"  {'Estructura':<{ancho}} {'Tamano (KB)':>12} {'Tamano (MB)':>12}")
    print("  " + sep)

    estructuras = [
        ("Vertices (float32, 3 coords)",   mem_verts_kb),
        ("Caras (int64, 3 indices)",        mem_caras_kb),
        (f"Nube de puntos ({N:,} pts)",     mem_pts_kb),
        ("Grid voxel 32^3 (float32)",       mem_g32_kb),
        ("Grid voxel 64^3 (float32)",       mem_g64_kb),
    ]
    for nombre, kb in estructuras:
        print(f"  {nombre:<{ancho}} {kb:>12.2f} {kb/1024:>12.4f}")

    # Datos de ocupacion
    print()
    print(f"  OCUPACION DE LOS GRIDS")
    print("  " + sep)
    print(f"  {'Grid':<{ancho}} {'Celdas ocup.':>13} {'Total celdas':>13} {'Ocupacion (%)':>14}")
    print("  " + sep)
    for R, n_ocup, pct, n_total in [
        ("32^3", ocup32, pct32, g32.size),
        ("64^3", ocup64, pct64, g64.size),
    ]:
        print(f"  {R:<{ancho}} {n_ocup:>13,} {n_total:>13,} {pct:>13.3f}%")

    print()
    print("=" * 65)

    # ── Guardar JSON ──────────────────────────────────────────
    resultados["archivo"] = Path(ruta).name
    resultados["n_puntos_muestreo"] = N
    resultados["repeticiones"] = REP

    if args.salida:
        ruta_json = Path(args.salida)
    else:
        ruta_json = Path(f"metricas_{Path(ruta).stem}.json")

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"  Resultados guardados en: {ruta_json.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
