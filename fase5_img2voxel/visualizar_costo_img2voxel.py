"""
visualizar_costo_img2voxel.py
================================
Compara el costo computacional de Img2Voxel (experimento adicional)
contra los 3 enfoques de la metodologia principal (SVM, Random Forest,
Net5-Octree), para una resolucion dada.

IMPORTANTE: Img2Voxel resuelve una tarea distinta (reconstruccion,
no clasificacion), asi que su "accuracy" no es directamente comparable.
Esta comparacion se limita estrictamente al COSTO COMPUTACIONAL:
tiempo de inferencia, tamano del modelo y VRAM, que si son medidas
homogeneas entre las 4 implementaciones.

Uso:
    python visualizar_costo_img2voxel.py --resolucion 32
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DIR_RESULTADOS = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\resultados")

COLORES = {
    "SVM":          "#2196F3",
    "RandomForest": "#FF9800",
    "Net5":         "#4CAF50",
    "Img2Voxel":    "#9C27B0",
}


def cargar_todos(R: int) -> dict:
    """Carga los 3 resumenes existentes + el de Img2Voxel (si existe)."""
    datos = {}

    with open(DIR_RESULTADOS / f"resumen_hce_R{R}.json") as f:
        hce = json.load(f)
    datos["SVM"] = hce["svm"]
    datos["RandomForest"] = hce["random_forest"]

    with open(DIR_RESULTADOS / f"resumen_net5_R{R}.json") as f:
        datos["Net5"] = json.load(f)

    ruta_img2voxel = DIR_RESULTADOS / f"resumen_img2voxel_R{R}.json"
    if ruta_img2voxel.exists():
        with open(ruta_img2voxel) as f:
            datos["Img2Voxel"] = json.load(f)
    else:
        print(f"[Aviso] No se encontro {ruta_img2voxel.name}. "
             f"Corre entrenar_img2voxel.py --resolucion {R} primero.")

    return datos


def graficar_costo_computacional(datos: dict, R: int):
    """Genera 3 subplots: tiempo de inferencia, tamano de modelo, VRAM."""
    modelos = list(datos.keys())
    colores = [COLORES[m] for m in modelos]

    tiempo_ms = [datos[m]["tiempo_inferencia_promedio_ms"] for m in modelos]
    tamano_mb = [datos[m]["tamano_modelo_mb"] for m in modelos]
    vram_mb   = [datos[m].get("vram_pico_mb", 0) for m in modelos]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, valores, titulo, unidad in zip(
        axes,
        [tiempo_ms, tamano_mb, vram_mb],
        ["Tiempo de inferencia\npor muestra", "Tamano del modelo\nguardado",
         "VRAM pico\n(GPU, 0 si no aplica)"],
        ["ms", "MB", "MB"],
    ):
        bars = ax.bar(modelos, valores, color=colores)
        for b, v in zip(bars, valores):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                   f"{v:.2f} {unidad}", ha="center", va="bottom", fontsize=9)
        ax.set_title(titulo, fontsize=11)
        ax.set_ylabel(unidad)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        f"Costo Computacional — Resolucion {R}^3\n"
        f"(Img2Voxel resuelve una tarea distinta: reconstruccion vs clasificacion; "
        f"solo el COSTO es comparable, no el accuracy)",
        fontsize=12,
    )
    plt.tight_layout()
    salida = DIR_RESULTADOS / f"costo_img2voxel_comparativo_R{R}.png"
    plt.savefig(salida, dpi=150, bbox_inches="tight")
    print(f"[Guardado] {salida.name}")
    plt.close()


def graficar_tabla_costo(datos: dict, R: int):
    """Tabla resumen con todas las metricas de costo, como imagen."""
    columnas = ["Metodo", "Tarea", "Inf/muestra (ms)", "Tamano (MB)", "VRAM (MB)"]
    tareas = {
        "SVM": "Clasificacion", "RandomForest": "Clasificacion",
        "Net5": "Clasificacion", "Img2Voxel": "Reconstruccion",
    }

    filas = []
    for nombre, d in datos.items():
        filas.append([
            nombre,
            tareas[nombre],
            f"{d['tiempo_inferencia_promedio_ms']:.3f}",
            f"{d['tamano_modelo_mb']:.2f}",
            f"{d.get('vram_pico_mb', 0):.1f}",
        ])

    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.5 * len(filas)))
    ax.axis("off")
    tabla = ax.table(cellText=filas, colLabels=columnas, loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.2, 1.7)

    for (row, col), cell in tabla.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F5F5")

    ax.set_title(f"Resumen de Costo Computacional — Resolucion {R}^3",
                fontsize=12, pad=15, fontweight="bold")
    plt.tight_layout()
    salida = DIR_RESULTADOS / f"tabla_costo_img2voxel_R{R}.png"
    plt.savefig(salida, dpi=150, bbox_inches="tight")
    print(f"[Guardado] {salida.name}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolucion", type=int, default=32, choices=[32, 64])
    args = parser.parse_args()
    R = args.resolucion

    print("=" * 60)
    print(f"  COSTO COMPUTACIONAL COMPARATIVO — Resolucion {R}^3")
    print("=" * 60)

    datos = cargar_todos(R)

    if "Img2Voxel" not in datos:
        print("\nNo se puede generar la comparativa completa sin Img2Voxel.")
        return

    print("\nGenerando graficas...")
    graficar_costo_computacional(datos, R)
    graficar_tabla_costo(datos, R)

    print(f"\nGuardado en: {DIR_RESULTADOS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
