"""
verificar_octree_visual.py - Fase 2
=====================================
Herramienta de verificacion visual: convierte una malla .off a octree
y la grafica en 3D con matplotlib para confirmar que el pipeline
(normalizacion -> muestreo -> cuantizacion -> codificacion) funciona
correctamente antes de procesar todo el dataset.

Uso:
    python verificar_octree_visual.py "ruta\al\archivo.off"
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from octree import (
    leer_off, normalizar_malla, muestrear_superficie_con_normales,
    construir_grid_octree, profundidad_de,
)


def graficar_pipeline(ruta_off: str, resolucion: int = 32, seed: int = 42):
    """Genera una figura con 3 paneles: malla normalizada, nube de puntos
    muestreada, y grid de octree (celdas ocupadas)."""

    rng = np.random.default_rng(seed)

    # 1. Cargar y normalizar
    vertices, caras = leer_off(ruta_off)
    vertices_norm = normalizar_malla(vertices)

    # 2. Muestrear superficie con normales
    puntos, normales = muestrear_superficie_con_normales(
        vertices_norm, caras, n_puntos=20000, rng=rng,
    )

    # 3. Cuantizar a grid de octree
    grid = construir_grid_octree(puntos, normales, resolucion)
    ocupacion = grid[0]

    # Coordenadas de celdas ocupadas (para graficar como scatter)
    idx_ocupadas = np.argwhere(ocupacion > 0)
    R = resolucion
    coords_centro = (idx_ocupadas + 0.5) / R * 2 - 1   # volver a escala [-1,1]

    fig = plt.figure(figsize=(15, 5))

    # Panel 1: vertices originales normalizados
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(vertices_norm[:, 0], vertices_norm[:, 1], vertices_norm[:, 2],
                s=0.5, alpha=0.3, color="steelblue")
    ax1.set_title(f"1. Malla normalizada\n({len(vertices_norm)} vertices)")
    ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 1); ax1.set_zlim(-1, 1)

    # Panel 2: nube de puntos muestreada sobre la superficie
    ax2 = fig.add_subplot(132, projection="3d")
    sub = rng.choice(len(puntos), min(3000, len(puntos)), replace=False)
    ax2.scatter(puntos[sub, 0], puntos[sub, 1], puntos[sub, 2],
                s=1, alpha=0.4, color="darkorange")
    ax2.set_title(f"2. Muestreo superficie\n(20000 pts, mostrando 3000)")
    ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1); ax2.set_zlim(-1, 1)

    # Panel 3: grid de octree (celdas ocupadas)
    ax3 = fig.add_subplot(133, projection="3d")
    pct = 100 * len(idx_ocupadas) / ocupacion.size
    ax3.scatter(coords_centro[:, 0], coords_centro[:, 1], coords_centro[:, 2],
                s=8, alpha=0.6, color="seagreen", marker="s")
    ax3.set_title(f"3. Octree {R}^3 (L={profundidad_de(R)})\n"
                  f"{len(idx_ocupadas):,} celdas ocupadas ({pct:.1f}%)")
    ax3.set_xlim(-1, 1); ax3.set_ylim(-1, 1); ax3.set_zlim(-1, 1)

    plt.tight_layout()

    nombre_salida = f"verificacion_octree_{Path(ruta_off).stem}_{resolucion}.png"
    plt.savefig(nombre_salida, dpi=120, bbox_inches="tight")
    print(f"\n[Guardado] Figura: {nombre_salida}")
    plt.show()

    return grid


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Uso: python verificar_octree_visual.py "ruta\\al\\archivo.off" [resolucion]')
        sys.exit(0)

    ruta = sys.argv[1]
    resolucion = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    print("=" * 60)
    print("  VERIFICACION VISUAL DEL PIPELINE DE OCTREE")
    print("=" * 60)
    print(f"  Archivo    : {ruta}")
    print(f"  Resolucion : {resolucion}^3")

    grid = graficar_pipeline(ruta, resolucion=resolucion)

    print(f"\n  Grid shape final : {grid.shape}")
    print(f"  Canales          : [ocupacion, normal_x, normal_y, normal_z]")
    print("\nVerificacion completada.")
