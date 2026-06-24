"""
renderizar_vistas.py - Experimento adicional: Reconstruccion 3D desde imagen 2D
==================================================================================
Genera N vistas 2D (renders) de cada malla .off de ModelNet40, rotando la
camara alrededor del objeto en azimuts uniformemente espaciados. Estas
imagenes seran la entrada del encoder 2D del modelo Img2Voxel.

IMPORTANTE: este es un experimento adicional, fuera de la metodologia
original de tesis (HCE vs Net5 como clasificadores). Su proposito es
explorar reconstruccion 3D aproximada desde una imagen 2D de un objeto.

Metodo de renderizado: se usa matplotlib (Axes3D.voxels / scatter) para
rasterizar la malla ya convertida a octree (reutilizando octree.py de la
Fase 2), evitando dependencias pesadas como pyrender/open3d que ya
descartamos por incompatibilidad con Python 3.12.

Salida: imagenes PNG en escala de grises (mas liviano, suficiente para
silueta + sombreado por profundidad), tamano 128x128 por defecto.

Estructura de salida:
    data/renders/<clase>/<split>/<archivo>_v00.png
    data/renders/<clase>/<split>/<archivo>_v01.png
    ...
    data/renders/<clase>/<split>/<archivo>_v05.png   (6 vistas por defecto)

Uso:
    python renderizar_vistas.py --n_vistas 6 --resolucion 128
"""

import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend sin GUI, mucho mas rapido para render masivo
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "fase2_octree"))
from octree import leer_off, normalizar_malla

# ── Configuracion ──────────────────────────────────────────────
RAIZ_DATASET = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\Dataset\ModelNet40")
RAIZ_SALIDA  = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\data\renders")
N_PROCESOS   = 10   # Ryzen 7 5700X

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
# RENDERIZADO DE UNA VISTA
# ──────────────────────────────────────────────────────────────

def renderizar_vista(vertices: np.ndarray, caras: np.ndarray, azimut: float,
                     elevacion: float, resolucion: int = 128) -> np.ndarray:
    """
    Renderiza una vista 2D de la malla usando matplotlib, vista desde un
    angulo (azimut, elevacion) fijo. Usa Poly3DCollection para un sombreado
    aproximado basado en la normal de cada triangulo (mas realista que un
    scatter de puntos).

    Retorna un array (resolucion, resolucion) en escala de grises [0, 255].
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(resolucion / 100, resolucion / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()

    if caras is not None and len(caras) > 0:
        triangulos = vertices[caras]   # (M, 3, 3)

        # Sombreado simple tipo "flat shading": intensidad segun el angulo
        # entre la normal del triangulo y una luz direccional fija
        v0, v1, v2 = triangulos[:, 0], triangulos[:, 1], triangulos[:, 2]
        normales = np.cross(v1 - v0, v2 - v0)
        normas = np.linalg.norm(normales, axis=1, keepdims=True)
        normas = np.clip(normas, 1e-12, None)
        normales = normales / normas

        luz = np.array([0.5, 0.5, 1.0])
        luz = luz / np.linalg.norm(luz)
        intensidad = np.clip(normales @ luz, 0.15, 1.0)   # piso de 0.15 (ambient)

        colores = np.stack([intensidad] * 3, axis=1)   # gris segun intensidad

        coleccion = Poly3DCollection(triangulos, facecolor=colores,
                                     edgecolor="none", alpha=1.0)
        ax.add_collection3d(coleccion)
    else:
        # Fallback: si no hay caras, dibujar los vertices como scatter
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                  s=1, color="gray")

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.view_init(elev=elevacion, azim=azimut)
    ax.set_box_aspect([1, 1, 1])

    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    imagen_rgb = buffer[:, :, :3]
    imagen_gris = imagen_rgb.mean(axis=2).astype(np.uint8)

    plt.close(fig)

    return imagen_gris


# ──────────────────────────────────────────────────────────────
# PROCESAMIENTO DE UNA MUESTRA (multiples vistas)
# ──────────────────────────────────────────────────────────────

def procesar_una_muestra(args: tuple) -> tuple:
    """
    Genera N vistas de un archivo .off y las guarda como PNG.
    Disenado para ejecutarse en ProcessPoolExecutor.
    """
    ruta_off, nombre, clase, split, n_vistas, resolucion = args

    try:
        vertices, caras = leer_off(ruta_off)
        vertices = normalizar_malla(vertices)

        dir_salida = RAIZ_SALIDA / clase / split
        dir_salida.mkdir(parents=True, exist_ok=True)

        # Azimuts uniformemente espaciados alrededor del objeto (360 grados)
        azimuts = np.linspace(0, 360, n_vistas, endpoint=False)
        elevacion = 20.0   # angulo fijo de elevacion (vista "de 3/4")

        for i, az in enumerate(azimuts):
            img = renderizar_vista(vertices, caras, az, elevacion, resolucion)
            ruta_salida = dir_salida / f"{nombre}_v{i:02d}.png"
            plt.imsave(ruta_salida, img, cmap="gray", vmin=0, vmax=255)

        return (nombre, True, None)

    except Exception as e:
        return (nombre, False, str(e))


# ──────────────────────────────────────────────────────────────
# RECOLECCION Y PROCESAMIENTO MASIVO
# ──────────────────────────────────────────────────────────────

def recolectar_archivos(raiz: Path, split: str) -> list:
    muestras = []
    for clase in CLASES:
        carpeta = raiz / clase / split
        if not carpeta.exists():
            continue
        for archivo in sorted(carpeta.glob("*.off")):
            muestras.append((str(archivo), archivo.stem, clase))
    return muestras


def procesar_split(split: str, n_vistas: int, resolucion: int):
    print(f"\n{'='*60}")
    print(f"  Renderizando split: {split.upper()}")
    print(f"{'='*60}")

    muestras = recolectar_archivos(RAIZ_DATASET, split)
    print(f"  Archivos encontrados: {len(muestras)}")
    print(f"  Vistas por objeto: {n_vistas}  ->  {len(muestras) * n_vistas} imagenes totales")

    if len(muestras) == 0:
        return [], []

    tareas = [
        (ruta, nombre, clase, split, n_vistas, resolucion)
        for ruta, nombre, clase in muestras
    ]

    exitosos, fallidos = [], []
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_vistas", type=int, default=6,
                        help="Numero de vistas (azimuts) por objeto")
    parser.add_argument("--resolucion", type=int, default=128,
                        help="Resolucion en pixeles de cada render (cuadrado)")
    args = parser.parse_args()

    print("=" * 60)
    print("  RENDERIZADO DE VISTAS 2D — Experimento Img2Voxel")
    print("=" * 60)
    print(f"  Dataset origen : {RAIZ_DATASET}")
    print(f"  Salida         : {RAIZ_SALIDA}")
    print(f"  Vistas/objeto  : {args.n_vistas}")
    print(f"  Resolucion img : {args.resolucion}x{args.resolucion}")
    print(f"  Procesos       : {N_PROCESOS}")

    resumen = {}
    for split in ["train", "test"]:
        exitosos, fallidos = procesar_split(split, args.n_vistas, args.resolucion)
        resumen[split] = {"exitosos": len(exitosos), "fallidos": len(fallidos)}

    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    for split, datos in resumen.items():
        print(f"  {split:10s}: {datos['exitosos']} exitosos, {datos['fallidos']} fallidos")
    print(f"\n  Imagenes guardadas en: {RAIZ_SALIDA}")
    print("=" * 60)


if __name__ == "__main__":
    main()
