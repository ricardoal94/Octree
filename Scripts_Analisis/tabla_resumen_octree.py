"""
tabla_resumen_octree.py
========================
Lee el CSV completo generado por medir_metricas_off.py y produce:

  1. Tabla resumen por clase (consola) con:
       - n modelos procesados y errores
       - ocupacion media en 32^3 y 64^3
       - nodos medios del octree en 32^3 y 64^3
       - tiempo medio de procesamiento por objeto
       - almacenamiento total estimado

  2. Tabla resumen global (train + test) citables en el documento

  3. Figura PNG/SVG de la tabla lista para incluir en la tesis

  4. JSON con los valores para citar en el texto

Uso:
    python tabla_resumen_octree.py
    python tabla_resumen_octree.py --split train
    python tabla_resumen_octree.py --split test
"""

import csv
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

DIR_RESULTADOS = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\resultados")

# Memoria fija por objeto (float32, 4 bytes/celda)
MEM_32_KB  = 32**3 * 4 / 1024          # 128 KB
MEM_64_KB  = 64**3 * 4 / 1024          # 1024 KB = 1 MB
MEM_32_MB  = MEM_32_KB / 1024
MEM_64_MB  = MEM_64_KB / 1024

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
# CARGA DEL CSV
# ──────────────────────────────────────────────────────────────

def cargar_csv(split_filtro: str = None) -> list:
    ruta = DIR_RESULTADOS / "metricas_off_completo.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro {ruta.name}.\n"
            f"Corre primero: python medir_metricas_off.py"
        )
    filas = []
    with open(ruta, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            if split_filtro and fila["split"] != split_filtro:
                continue
            filas.append(fila)
    return filas


# ──────────────────────────────────────────────────────────────
# CALCULO DE ESTADISTICAS
# ──────────────────────────────────────────────────────────────

def calcular_estadisticas(filas: list) -> tuple:
    """
    Retorna (stats_por_clase, stats_global).
    stats_por_clase : dict clase -> dict de metricas
    stats_global    : dict de metricas globales
    """
    por_clase = defaultdict(list)
    for fila in filas:
        por_clase[fila["clase"]].append(fila)

    stats_clase = {}
    for clase in CLASES:
        muestras = por_clase.get(clase, [])
        if not muestras:
            continue
        n = len(muestras)

        def media(campo):
            vals = [float(m[campo]) for m in muestras if m.get(campo)]
            return round(np.mean(vals), 4) if vals else 0.0

        def suma(campo):
            vals = [float(m[campo]) for m in muestras if m.get(campo)]
            return round(sum(vals), 2) if vals else 0.0

        stats_clase[clase] = {
            "n":                n,
            "ocup32_pct_media": media("ocup32_pct"),
            "ocup64_pct_media": media("ocup64_pct"),
            "nodos32_media":    round(media("ocup32_abs"), 0),
            "nodos64_media":    round(media("ocup64_abs"), 0),
            "t_total_media_ms": media("t_total_ms"),
            "mem32_total_mb":   round(n * MEM_32_MB, 2),
            "mem64_total_mb":   round(n * MEM_64_MB, 2),
        }

    # Estadisticas globales
    n_total  = len(filas)
    n_clases = len(stats_clase)

    def media_global(campo):
        vals = [float(f[campo]) for f in filas if f.get(campo)]
        return round(np.mean(vals), 4) if vals else 0.0

    def std_global(campo):
        vals = [float(f[campo]) for f in filas if f.get(campo)]
        return round(np.std(vals), 4) if vals else 0.0

    stats_global = {
        "n_total":              n_total,
        "n_clases":             n_clases,
        "n_errores":            0,   # los errores no llegan al CSV

        # Ocupacion
        "ocup32_pct_media":     media_global("ocup32_pct"),
        "ocup32_pct_std":       std_global("ocup32_pct"),
        "ocup64_pct_media":     media_global("ocup64_pct"),
        "ocup64_pct_std":       std_global("ocup64_pct"),

        # Nodos del octree
        "nodos32_media":        round(media_global("ocup32_abs"), 0),
        "nodos32_std":          round(std_global("ocup32_abs"), 0),
        "nodos64_media":        round(media_global("ocup64_abs"), 0),
        "nodos64_std":          round(std_global("ocup64_abs"), 0),

        # Tiempo
        "t_total_media_ms":     media_global("t_total_ms"),
        "t_total_std_ms":       std_global("t_total_ms"),
        "t_total_dataset_min":  round(n_total * media_global("t_total_ms") / 60000, 1),

        # Almacenamiento total (dataset completo)
        "mem32_total_dataset_mb":  round(n_total * MEM_32_MB, 1),
        "mem32_total_dataset_gb":  round(n_total * MEM_32_MB / 1024, 3),
        "mem64_total_dataset_mb":  round(n_total * MEM_64_MB, 1),
        "mem64_total_dataset_gb":  round(n_total * MEM_64_MB / 1024, 3),
        "mem_por_objeto_32_kb":    round(MEM_32_KB, 1),
        "mem_por_objeto_64_kb":    round(MEM_64_KB, 1),
    }

    return stats_clase, stats_global


# ──────────────────────────────────────────────────────────────
# IMPRESION EN CONSOLA
# ──────────────────────────────────────────────────────────────

def imprimir_tabla_consola(stats_clase: dict, stats_global: dict, split: str):
    print("\n" + "=" * 95)
    print(f"  TABLA RESUMEN DEL OCTREE — ModelNet40  ({split.upper()})")
    print("=" * 95)

    enc = ["Clase", "N", "Ocup.32³(%)", "Ocup.64³(%)",
           "Nodos 32³", "Nodos 64³", "T.media(ms)", "Mem.32³(MB)", "Mem.64³(MB)"]
    anchos = [14, 5, 12, 12, 10, 10, 13, 12, 12]
    header = "  " + "  ".join(f"{h:<{w}}" for h, w in zip(enc, anchos))
    print(header)
    print("  " + "-" * 93)

    for clase in CLASES:
        if clase not in stats_clase:
            continue
        s = stats_clase[clase]
        vals = [
            clase, str(s["n"]),
            f"{s['ocup32_pct_media']:.3f}",
            f"{s['ocup64_pct_media']:.3f}",
            f"{s['nodos32_media']:.0f}",
            f"{s['nodos64_media']:.0f}",
            f"{s['t_total_media_ms']:.2f}",
            f"{s['mem32_total_mb']:.2f}",
            f"{s['mem64_total_mb']:.2f}",
        ]
        print("  " + "  ".join(f"{v:<{w}}" for v, w in zip(vals, anchos)))

    print("  " + "=" * 93)
    g = stats_global
    vals_g = [
        "GLOBAL", str(g["n_total"]),
        f"{g['ocup32_pct_media']:.3f}±{g['ocup32_pct_std']:.3f}",
        f"{g['ocup64_pct_media']:.3f}±{g['ocup64_pct_std']:.3f}",
        f"{g['nodos32_media']:.0f}",
        f"{g['nodos64_media']:.0f}",
        f"{g['t_total_media_ms']:.2f}±{g['t_total_std_ms']:.2f}",
        f"{g['mem32_total_dataset_mb']:.1f} MB",
        f"{g['mem64_total_dataset_mb']:.1f} MB",
    ]
    print("  " + "  ".join(f"{v:<{w}}" for v, w in zip(vals_g, anchos)))
    print("=" * 95)

    print(f"\n  Resumen citables en el documento:")
    print(f"  - Objetos procesados           : {g['n_total']:,}")
    print(f"  - Ocupacion media 32³          : {g['ocup32_pct_media']:.3f}% "
          f"(±{g['ocup32_pct_std']:.3f}%)")
    print(f"  - Ocupacion media 64³          : {g['ocup64_pct_media']:.3f}% "
          f"(±{g['ocup64_pct_std']:.3f}%)")
    print(f"  - Nodos medios octree 32³      : {g['nodos32_media']:.0f} "
          f"(±{g['nodos32_std']:.0f})")
    print(f"  - Nodos medios octree 64³      : {g['nodos64_media']:.0f} "
          f"(±{g['nodos64_std']:.0f})")
    print(f"  - Tiempo medio por objeto      : {g['t_total_media_ms']:.2f} ms "
          f"(±{g['t_total_std_ms']:.2f} ms)")
    print(f"  - Tiempo total dataset         : ~{g['t_total_dataset_min']:.1f} min")
    print(f"  - Almacenamiento 32³ (dataset) : {g['mem32_total_dataset_mb']:.1f} MB "
          f"({g['mem32_total_dataset_gb']:.3f} GB)")
    print(f"  - Almacenamiento 64³ (dataset) : {g['mem64_total_dataset_mb']:.1f} MB "
          f"({g['mem64_total_dataset_gb']:.3f} GB)")
    print(f"  - Memoria por objeto 32³       : {g['mem_por_objeto_32_kb']:.0f} KB")
    print(f"  - Memoria por objeto 64³       : {g['mem_por_objeto_64_kb']:.0f} KB")


# ──────────────────────────────────────────────────────────────
# FIGURA: tabla imagen para la tesis
# ──────────────────────────────────────────────────────────────

def graficar_tabla_resumen(stats_clase: dict, stats_global: dict,
                            split: str, R: int = None):
    """
    Genera una tabla imagen con las estadisticas por clase,
    lista para incluir directamente en el documento de tesis.
    """
    columnas = [
        "Clase", "N", f"Ocup.\n32³ (%)", f"Ocup.\n64³ (%)",
        f"Nodos\n32³", f"Nodos\n64³", "T. media\n(ms)",
        "Almacen.\n32³ (MB)", "Almacen.\n64³ (MB)",
    ]

    clases_presentes = [c for c in CLASES if c in stats_clase]
    celdas = []
    for clase in clases_presentes:
        s = stats_clase[clase]
        celdas.append([
            clase,
            str(s["n"]),
            f"{s['ocup32_pct_media']:.3f}",
            f"{s['ocup64_pct_media']:.3f}",
            f"{s['nodos32_media']:.0f}",
            f"{s['nodos64_media']:.0f}",
            f"{s['t_total_media_ms']:.2f}",
            f"{s['mem32_total_mb']:.2f}",
            f"{s['mem64_total_mb']:.2f}",
        ])

    # Fila de totales globales
    g = stats_global
    celdas.append([
        "TOTAL / MEDIA",
        str(g["n_total"]),
        f"{g['ocup32_pct_media']:.3f}",
        f"{g['ocup64_pct_media']:.3f}",
        f"{g['nodos32_media']:.0f}",
        f"{g['nodos64_media']:.0f}",
        f"{g['t_total_media_ms']:.2f}",
        f"{g['mem32_total_dataset_mb']:.1f}",
        f"{g['mem64_total_dataset_mb']:.1f}",
    ])

    alto = max(8, 0.42 * len(celdas) + 1.5)
    fig, ax = plt.subplots(figsize=(18, alto))
    ax.axis("off")

    tabla = ax.table(
        cellText=celdas, colLabels=columnas,
        loc="center", cellLoc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8.5)
    tabla.scale(1.05, 1.65)

    # Colorear encabezado y fila de totales
    for (row, col), cell in tabla.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif row == len(celdas):   # fila de totales
            cell.set_facecolor("#CFD8DC")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F5F5")

    ax.set_title(
        f"Tabla de ocupación y costo del pipeline de octree — "
        f"ModelNet40 ({split})\n"
        f"Valores medios por clase | Memoria = float32, 4 bytes/celda",
        fontsize=11, pad=18, fontweight="bold",
    )

    sufijo = f"_{split}"
    for ext in ("png", "svg"):
        salida = DIR_RESULTADOS / f"tabla_resumen_octree{sufijo}.{ext}"
        plt.savefig(salida, dpi=150 if ext == "png" else None,
                    format=ext, bbox_inches="tight")
        print(f"[Guardado] {salida.name}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="ambos",
                        choices=["train", "test", "ambos"])
    args = parser.parse_args()

    splits = ["train", "test"] if args.split == "ambos" else [args.split]

    for split in splits:
        print(f"\nCargando CSV para split={split}...")
        try:
            filas = cargar_csv(split_filtro=None if split == "ambos" else split)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return

        if not filas:
            print(f"  [Aviso] No hay datos para split={split}")
            continue

        print(f"  Filas cargadas: {len(filas):,}")

        stats_clase, stats_global = calcular_estadisticas(filas)
        imprimir_tabla_consola(stats_clase, stats_global, split)

        # Guardar JSON con valores citables
        json_salida = DIR_RESULTADOS / f"resumen_octree_{split}.json"
        with open(json_salida, "w", encoding="utf-8") as f:
            json.dump({
                "split": split,
                "global": stats_global,
                "por_clase": stats_clase,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n[JSON] Guardado: {json_salida.name}")

        print("\nGenerando tabla imagen...")
        graficar_tabla_resumen(stats_clase, stats_global, split)

    print(f"\nTodo guardado en: {DIR_RESULTADOS}")


if __name__ == "__main__":
    main()
