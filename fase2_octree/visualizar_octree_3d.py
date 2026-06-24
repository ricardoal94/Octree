"""
visualizar_octree_3d.py
=========================
Visualizacion 3D de los grids de octree generados en la Fase 2.

Genera 3 tipos de vistas:
  1. Voxel plot solido (matplotlib ax.voxels) — el objeto como bloques 3D
  2. Scatter de celdas ocupadas coloreadas por vector normal (RGB = normal)
  3. Comparacion lado a lado 32^3 vs 64^3 del mismo objeto

Uso:
    python visualizar_octree_3d.py --clase airplane --split train --indice 0
    python visualizar_octree_3d.py --archivo "ruta\al\archivo.off"
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "fase2_octree"))
from octree import malla_a_octree, profundidad_de

RAIZ_DATASET = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\Dataset\ModelNet40")
RAIZ_DATA    = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\data")


# ──────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────────────────────

def cargar_grid_desde_npz(clase: str, split: str, indice: int, resolucion: int) -> tuple:
    """Carga un grid ya preprocesado desde data/octrees_<R>/<clase>/<split>/."""
    carpeta = RAIZ_DATA / f"octrees_{resolucion}" / clase / split
    archivos = sorted(carpeta.glob("*.npz"))
    if indice >= len(archivos):
        raise IndexError(f"Indice {indice} fuera de rango ({len(archivos)} archivos disponibles)")

    data = np.load(archivos[indice])
    return data["grid"], archivos[indice].stem


def cargar_grid_desde_off(ruta_off: str, resolucion: int) -> tuple:
    """Genera el grid en el momento a partir de un .off arbitrario."""
    grid = malla_a_octree(ruta_off, resolucion=resolucion, n_puntos_muestreo=20000, seed=42)
    return grid, Path(ruta_off).stem


# ──────────────────────────────────────────────────────────────
# 1. VOXEL PLOT SOLIDO
# ──────────────────────────────────────────────────────────────

def graficar_voxels_solidos(grid: np.ndarray, titulo: str, ax=None):
    """
    Dibuja los voxels ocupados como cubos solidos, coloreados segun
    su vector normal mapeado a RGB: color = (normal + 1) / 2
    (asi normal en [-1,1] se mapea a color en [0,1]).
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

    ocupacion = grid[0] > 0
    normales  = grid[1:4]   # (3, R, R, R)

    # Mapear normal [-1,1] -> color RGB [0,1]
    color_rgb = (np.transpose(normales, (1, 2, 3, 0)) + 1.0) / 2.0
    alpha = np.ones(grid.shape[1:]) * 0.9
    colores = np.concatenate([color_rgb, alpha[..., None]], axis=-1)

    ax.voxels(ocupacion, facecolors=colores, edgecolor="k", linewidth=0.1)
    ax.set_title(titulo)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    return ax


# ──────────────────────────────────────────────────────────────
# 2. SCATTER DE CELDAS OCUPADAS (mas rapido para R=64)
# ──────────────────────────────────────────────────────────────

def graficar_scatter_normales(grid: np.ndarray, titulo: str, ax=None):
    """
    Alternativa mas ligera al voxel plot: dibuja un punto por cada
    celda ocupada, coloreado segun su normal (RGB).
    Mucho mas rapido para R=64 (262144 celdas posibles).
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

    R = grid.shape[1]
    ocupacion = grid[0]
    normales  = grid[1:4]

    idx = np.argwhere(ocupacion > 0)
    coords = (idx + 0.5) / R * 2 - 1   # escala [-1, 1]

    vecs_normal = normales[:, idx[:, 0], idx[:, 1], idx[:, 2]].T   # (N, 3)
    colores = (vecs_normal + 1.0) / 2.0   # RGB en [0,1]
    colores = np.clip(colores, 0, 1)

    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
              c=colores, s=15, marker="s", alpha=0.85)
    ax.set_title(titulo)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    return ax


# ──────────────────────────────────────────────────────────────
# 3. COMPARACION 32^3 vs 64^3
# ──────────────────────────────────────────────────────────────

def comparar_resoluciones(ruta_off: str = None, clase: str = None,
                          split: str = "train", indice: int = 0,
                          usar_voxels: bool = False):
    """Genera una figura con 32^3 y 64^3 del mismo objeto, lado a lado."""

    fig = plt.figure(figsize=(16, 8))

    grids = {}
    nombre_objeto = None

    for R in (32, 64):
        if ruta_off:
            grid, nombre_objeto = cargar_grid_desde_off(ruta_off, R)
        else:
            grid, nombre_objeto = cargar_grid_desde_npz(clase, split, indice, R)
        grids[R] = grid

    for i, R in enumerate((32, 64)):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ocupacion = grids[R][0]
        n_ocup = int((ocupacion > 0).sum())
        pct = 100 * n_ocup / ocupacion.size
        titulo = f"Resolucion {R}^3 (L={profundidad_de(R)})\n{n_ocup:,} celdas ocupadas ({pct:.1f}%)"

        if usar_voxels:
            graficar_voxels_solidos(grids[R], titulo, ax=ax)
        else:
            graficar_scatter_normales(grids[R], titulo, ax=ax)

    fig.suptitle(f"Objeto: {nombre_objeto}", fontsize=14)
    plt.tight_layout()

    salida = Path(f"comparacion_3d_{nombre_objeto}.png")
    plt.savefig(salida, dpi=130, bbox_inches="tight")
    print(f"\n[Guardado] {salida}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualizacion 3D de octrees")
    parser.add_argument("--archivo", type=str, default=None,
                        help="Ruta directa a un archivo .off")
    parser.add_argument("--clase", type=str, default="airplane",
                        help="Clase de ModelNet40 (si no se usa --archivo)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--indice", type=int, default=0,
                        help="Indice de la muestra dentro de la clase")
    parser.add_argument("--voxels", action="store_true",
                        help="Usar voxel plot solido en vez de scatter (mas lento, mas realista)")
    args = parser.parse_args()

    print("=" * 60)
    print("  VISUALIZACION 3D — Octree 32^3 vs 64^3")
    print("=" * 60)

    if args.archivo:
        print(f"  Archivo: {args.archivo}")
    else:
        print(f"  Clase  : {args.clase}")
        print(f"  Split  : {args.split}")
        print(f"  Indice : {args.indice}")

    comparar_resoluciones(
        ruta_off=args.archivo,
        clase=args.clase,
        split=args.split,
        indice=args.indice,
        usar_voxels=args.voxels,
    )

    print("\nVisualizacion completada.")


if __name__ == "__main__":
    main()
