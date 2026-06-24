"""
visualizar_reconstruccion.py - Experimento adicional: Img2Voxel
=================================================================
Muestra lado a lado:
  Panel 1: imagen 2D de entrada (el render que el modelo recibe)
  Panel 2: voxel grid REAL del objeto (ground truth)
  Panel 3: voxel grid RECONSTRUIDO por Img2Voxel

Permite evaluar visualmente la calidad de la reconstruccion y
encontrar casos buenos/malos para incluir en la tesis.

Uso:
    # Una clase especifica
    python visualizar_reconstruccion.py --clase airplane --indice 0 --resolucion 64

    # Muestra aleatoria
    python visualizar_reconstruccion.py --aleatorio --resolucion 64

    # Generar una cuadricula de N ejemplos (bueno para figuras de tesis)
    python visualizar_reconstruccion.py --cuadricula 6 --resolucion 64
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F_nn
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "fase1_modelnet40"))
sys.path.insert(0, str(Path(__file__).parent))

from img2voxel_modelo import Img2Voxel, get_device

RAIZ_DATA    = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\data")
RAIZ_RENDERS = RAIZ_DATA / "renders"
DIR_CKPT     = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\checkpoints")
DIR_RESULTADOS = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\resultados")

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
N_VISTAS = 8


# ──────────────────────────────────────────────────────────────
# CARGA
# ──────────────────────────────────────────────────────────────

def cargar_modelo(R: int, device: torch.device) -> Img2Voxel:
    modelo = Img2Voxel(resolucion=R).to(device)
    ckpt = torch.load(DIR_CKPT / f"img2voxel_mejor_R{R}.pth", map_location=device)
    modelo.load_state_dict(ckpt["model_state"])
    modelo.eval()
    return modelo


def cargar_muestra(clase: str, indice: int, R: int, vista: int = 0) -> tuple:
    """
    Retorna (img_tensor, voxel_real_dilatado, normales_raw, nombre, etiqueta).
    Las normales se guardan del grid sin dilatar para colorear correctamente.
    """
    raiz_octrees = RAIZ_DATA / f"octrees_{R}"
    carpeta = raiz_octrees / clase / "test"
    archivos = sorted(carpeta.glob("*.npz"))
    archivo = archivos[indice % len(archivos)]
    nombre = archivo.stem

    # Cargar imagen
    ruta_img = RAIZ_RENDERS / clase / "test" / f"{nombre}_v{vista:02d}.png"
    img = np.asarray(Image.open(ruta_img).convert("L"), dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0)   # (1, 128, 128)

    # Cargar voxel + dilatacion (mismo proceso que el dataset)
    data = np.load(archivo)
    grid_raw = data["grid"].copy()

    # Guardar normales originales (antes de dilatar) para colorear bien en 3D
    normales_raw = torch.from_numpy(grid_raw[1:4]).float()   # (3, R, R, R)

    ocup = torch.from_numpy(grid_raw[0]).float().unsqueeze(0).unsqueeze(0)
    for _ in range(4):
        ocup = F_nn.max_pool3d(ocup, kernel_size=3, stride=1, padding=1)
    grid_tensor = torch.from_numpy(grid_raw).float()
    grid_tensor[0] = ocup.squeeze()

    return img_tensor, grid_tensor, normales_raw, nombre, int(data["etiqueta"])


def reconstruir(modelo, img_tensor: torch.Tensor, device: torch.device,
                umbral: float = 0.3) -> torch.Tensor:
    """Pasa la imagen por el modelo y retorna el voxel grid binarizado.
    Umbral 0.3 (en lugar de 0.5) produce reconstrucciones mas completas
    que representan mejor la forma general del objeto."""
    x = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = modelo(x)
    prob_ocup = torch.sigmoid(logits[0, 0]).cpu()
    return (prob_ocup > umbral).float()


# ──────────────────────────────────────────────────────────────
# VISUALIZACION 3D DE UN VOXEL GRID
# ──────────────────────────────────────────────────────────────

def scatter_voxel(ax, voxel: np.ndarray, normales: np.ndarray = None,
                  color_fijo: str = None, titulo: str = "", alpha: float = 0.8):
    """
    Dibuja un voxel grid como scatter 3D.
    Si hay normales, colorea por ellas. Si no, usa color_fijo.
    """
    R = voxel.shape[0]
    idx = np.argwhere(voxel > 0.5)
    if len(idx) == 0:
        ax.text(0, 0, 0, "Sin celdas\nocupadas", ha="center", va="center")
        ax.set_title(titulo, fontsize=10)
        return

    coords = (idx + 0.5) / R * 2 - 1

    if normales is not None and color_fijo is None:
        vecs = normales[:, idx[:, 0], idx[:, 1], idx[:, 2]].T
        normas = np.linalg.norm(vecs, axis=1, keepdims=True)
        # Si las normales estan disponibles en esas celdas, colorear por ellas
        # Si no (celdas agregadas por dilatacion), usar color neutro gris-azul
        tiene_normal = (normas > 0.1).squeeze()
        colores = np.full((len(idx), 3), 0.5)   # gris por defecto
        if tiene_normal.any():
            colores[tiene_normal] = np.clip(
                (vecs[tiene_normal] / normas[tiene_normal] + 1.0) / 2.0, 0, 1
            )
    else:
        colores = color_fijo or "steelblue"

    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
              c=colores, s=6, alpha=alpha, marker="s")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(titulo, fontsize=10)


# ──────────────────────────────────────────────────────────────
# FIGURA PRINCIPAL: imagen + real + reconstruccion
# ──────────────────────────────────────────────────────────────

def graficar_reconstruccion(img_tensor, voxel_real, normales_raw, voxel_recon,
                             nombre, etiqueta, R, iou, salida=None):
    """
    3 paneles:
      1. Imagen 2D de entrada
      2. Voxel real (ground truth), coloreado por normales originales
      3. Voxel reconstruido por Img2Voxel
    """
    fig = plt.figure(figsize=(14, 5))

    # Panel 1: imagen 2D
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_tensor.squeeze().numpy(), cmap="gray")
    ax1.set_title("Imagen 2D de entrada\n(render desde 1 vista)", fontsize=10)
    ax1.axis("off")

    # Panel 2: voxel real (con normales originales sin dilatar)
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    scatter_voxel(ax2, voxel_real[0].numpy(), normales=normales_raw.numpy(),
                  titulo=f"Voxel real (ground truth)\n{R}^3 con dilatacion")

    # Panel 3: voxel reconstruido
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    scatter_voxel(ax3, voxel_recon.numpy(), color_fijo="#9C27B0",
                  titulo=f"Reconstruccion Img2Voxel\nIoU = {iou:.3f}")

    clase_nombre = CLASES[etiqueta]
    fig.suptitle(
        f"Reconstruccion 3D desde imagen 2D  |  Objeto: {nombre}  |  "
        f"Clase: {clase_nombre}  |  Resolucion: {R}^3",
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if salida is None:
        DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
        salida = DIR_RESULTADOS / f"reconstruccion_{nombre}_R{R}.png"

    plt.savefig(salida, dpi=140, bbox_inches="tight")
    print(f"[Guardado] {salida}")
    plt.show()
    plt.close()
    return iou


def calcular_iou(voxel_pred: torch.Tensor, voxel_real: torch.Tensor) -> float:
    pred = (voxel_pred > 0.5).float()
    real = (voxel_real[0] > 0.5).float()
    interseccion = (pred * real).sum()
    union = ((pred + real) > 0).float().sum()
    return float(interseccion / union.clamp(min=1.0))


# ──────────────────────────────────────────────────────────────
# CUADRICULA: N ejemplos en una sola figura
# ──────────────────────────────────────────────────────────────

def graficar_cuadricula(modelo, device, R: int, n: int = 6, umbral: float = 0.3):
    """
    Genera una figura con N ejemplos aleatorios de diferentes clases,
    mostrando imagen + reconstruccion (sin el real, para ahorrar espacio).
    Ideal para figuras de tesis que muestran diversidad de resultados.
    """
    rng = np.random.default_rng(42)
    clases_elegidas = rng.choice(CLASES, n, replace=False)

    fig = plt.figure(figsize=(4 * n, 8))

    ious = []
    for col, clase in enumerate(clases_elegidas):
        carpeta = RAIZ_DATA / f"octrees_{R}" / clase / "test"
        archivos = sorted(carpeta.glob("*.npz"))
        if not archivos:
            continue
        archivo = archivos[0]
        nombre = archivo.stem

        try:
            img_tensor, voxel_real, normales_raw, nombre, etiqueta = cargar_muestra(clase, 0, R)
            voxel_recon = reconstruir(modelo, img_tensor, device, umbral=umbral)
            iou = calcular_iou(voxel_recon, voxel_real)
            ious.append(iou)

            # Imagen 2D
            ax_img = fig.add_subplot(2, n, col + 1)
            ax_img.imshow(img_tensor.squeeze().numpy(), cmap="gray")
            ax_img.set_title(f"{clase}\n(imagen)", fontsize=8)
            ax_img.axis("off")

            # Reconstruccion 3D
            ax_3d = fig.add_subplot(2, n, n + col + 1, projection="3d")
            scatter_voxel(ax_3d, voxel_recon.numpy(), color_fijo="#9C27B0",
                         titulo=f"IoU={iou:.3f}", alpha=0.7)

        except Exception as e:
            print(f"  [Aviso] Error en {clase}: {e}")

    iou_promedio = np.mean(ious) if ious else 0
    fig.suptitle(
        f"Reconstrucciones Img2Voxel — Resolucion {R}^3  |  "
        f"IoU promedio: {iou_promedio:.3f}",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    salida = DIR_RESULTADOS / f"cuadricula_reconstrucciones_R{R}.png"
    plt.savefig(salida, dpi=140, bbox_inches="tight")
    print(f"\n[Guardado] {salida}")
    plt.show()
    plt.close()

    print(f"IoU promedio en {len(ious)} ejemplos: {iou_promedio:.4f}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolucion", type=int, default=64, choices=[32, 64])
    parser.add_argument("--clase", type=str, default="airplane")
    parser.add_argument("--indice", type=int, default=0)
    parser.add_argument("--vista", type=int, default=0,
                        help="Que vista usar (0 a N_VISTAS-1)")
    parser.add_argument("--aleatorio", action="store_true")
    parser.add_argument("--cuadricula", type=int, default=0,
                        help="Si >0, genera cuadricula con ese numero de ejemplos")
    parser.add_argument("--umbral", type=float, default=0.3,
                        help="Umbral de binarizacion de ocupacion (default 0.3)")
    args = parser.parse_args()

    R = args.resolucion

    print("=" * 60)
    print(f"  VISUALIZACION DE RECONSTRUCCIONES — Img2Voxel {R}^3")
    print("=" * 60)

    device = get_device()
    print("\n[1/2] Cargando modelo Img2Voxel...")
    modelo = cargar_modelo(R, device)
    print("  OK")

    # Modo cuadricula
    if args.cuadricula > 0:
        print(f"\n[2/2] Generando cuadricula con {args.cuadricula} ejemplos...")
        graficar_cuadricula(modelo, device, R, n=args.cuadricula, umbral=args.umbral)
        return

    # Modo muestra individual
    if args.aleatorio:
        rng = np.random.default_rng()
        clase = rng.choice(CLASES)
        indice = int(rng.integers(0, 50))
        vista = int(rng.integers(0, N_VISTAS))
    else:
        clase = args.clase
        indice = args.indice
        vista = args.vista

    print(f"\n[2/2] Procesando: clase={clase}, indice={indice}, vista={vista}")

    img_tensor, voxel_real, normales_raw, nombre, etiqueta = cargar_muestra(clase, indice, R, vista)
    voxel_recon = reconstruir(modelo, img_tensor, device, umbral=args.umbral)
    iou = calcular_iou(voxel_recon, voxel_real)

    print(f"  IoU: {iou:.4f}  (umbral={args.umbral})")
    print(f"  Celdas reales       : {(voxel_real[0] > 0.5).sum().item():,}")
    print(f"  Celdas reconstruidas: {(voxel_recon > 0.5).sum().item():,}")

    graficar_reconstruccion(img_tensor, voxel_real, normales_raw, voxel_recon,
                            nombre, etiqueta, R, iou)


if __name__ == "__main__":
    main()
