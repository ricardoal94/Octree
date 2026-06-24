"""
gradcam_3d.py
===============
IMPORTANTE — que es y que NO es esta herramienta:

  Net5-Octree es un clasificador. Durante el forward pass colapsa toda
  la geometria del objeto a un vector de 512 numeros (via Global Average
  Pooling) antes de predecir la clase. Esa compresion es deliberada y
  pierde informacion espacial detallada de forma irreversible: el modelo
  NO puede "reconstruir" ni "generar" el objeto 3D a partir de ese vector,
  porque la informacion para hacerlo ya no esta disponible.

  Lo que SI se puede hacer sin entrenar nada nuevo es Grad-CAM 3D: un
  mapa de calor que muestra que regiones del objeto tuvieron mas
  influencia sobre la decision final de clasificacion, usando los
  gradientes de la clase predicha respecto a las activaciones de la
  ultima capa convolucional con resolucion espacial (antes del pooling
  global). Esto es una "atencion" o "saliencia", NO una reconstruccion
  geometrica ni una generacion de forma nueva.

  Todas las figuras generadas por este script dejan esto explicito en
  su titulo para evitar interpretaciones erroneas.

Uso:
    python gradcam_3d.py --clase airplane --indice 0 --resolucion 32
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "fase2_octree"))
sys.path.insert(0, str(Path(__file__).parent.parent / "fase3_net5"))

from octree      import malla_a_octree, profundidad_de
from net5_modelo import Net5Octree, get_device

RAIZ_DATA      = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\data")
DIR_CKPT       = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\checkpoints")
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


# ──────────────────────────────────────────────────────────────
# GRAD-CAM 3D
# ──────────────────────────────────────────────────────────────

class GradCAM3D:
    """
    Implementacion de Grad-CAM adaptada a convoluciones 3D.

    Engancha hooks en el penultimo bloque jerarquico de Net5Octree
    (la ultima capa que aun conserva resolucion espacial > 1x1x1,
    justo antes del Global Average Pooling), captura sus activaciones
    y gradientes durante un forward+backward pass, y calcula un mapa
    de calor ponderado por la importancia de cada canal para la clase
    predicha.
    """

    def __init__(self, modelo: Net5Octree):
        self.modelo = modelo
        self.activaciones = None
        self.gradientes = None

        # Penultimo bloque jerarquico: ultimo con resolucion espacial >1
        capa_objetivo = self.modelo.bloques[-2]

        capa_objetivo.register_forward_hook(self._hook_forward)
        capa_objetivo.register_full_backward_hook(self._hook_backward)

    def _hook_forward(self, module, entrada, salida):
        self.activaciones = salida.detach()

    def _hook_backward(self, module, grad_entrada, grad_salida):
        self.gradientes = grad_salida[0].detach()

    def generar_mapa(self, x: torch.Tensor, clase_objetivo: int = None) -> tuple:
        """
        x : tensor (1, 4, R, R, R)
        clase_objetivo : indice de clase a explicar. Si None, usa la
                          prediccion del modelo.

        Retorna (mapa_calor (r,r,r) en [0,1], clase_predicha, confianza)
        """
        self.modelo.eval()
        x = x.clone().requires_grad_(False)

        logits = self.modelo(x)
        probs  = F.softmax(logits, dim=1)

        if clase_objetivo is None:
            clase_objetivo = int(logits.argmax(dim=1).item())
        confianza = float(probs[0, clase_objetivo].item())

        # Backward solo respecto al logit de la clase objetivo
        self.modelo.zero_grad()
        logits[0, clase_objetivo].backward()

        # Pesos de importancia: promedio global de gradientes por canal
        pesos = self.gradientes.mean(dim=(2, 3, 4), keepdim=True)   # (1, C, 1, 1, 1)

        # Combinacion ponderada de activaciones + ReLU (Grad-CAM estandar)
        mapa = (pesos * self.activaciones).sum(dim=1, keepdim=True)  # (1, 1, r, r, r)
        mapa = F.relu(mapa)

        # Reescalar al tamano original del grid de entrada
        R = x.shape[-1]
        mapa = F.interpolate(mapa, size=(R, R, R), mode="trilinear", align_corners=False)
        mapa = mapa.squeeze().cpu().numpy()

        # Normalizar a [0, 1]
        if mapa.max() > mapa.min():
            mapa = (mapa - mapa.min()) / (mapa.max() - mapa.min())
        else:
            mapa = np.zeros_like(mapa)

        return mapa, clase_objetivo, confianza


# ──────────────────────────────────────────────────────────────
# CARGA DE MODELO Y MUESTRA
# ──────────────────────────────────────────────────────────────

def cargar_modelo_net5(R: int, device: torch.device) -> Net5Octree:
    modelo = Net5Octree(resolucion=R, num_clases=40).to(device)
    ckpt = torch.load(DIR_CKPT / f"net5_mejor_R{R}.pth", map_location=device)
    modelo.load_state_dict(ckpt["model_state"])
    modelo.eval()
    return modelo


def obtener_muestra(args, R: int) -> tuple:
    if args.archivo:
        grid = malla_a_octree(args.archivo, resolucion=R, n_puntos_muestreo=20000, seed=42)
        return grid, None, Path(args.archivo).stem

    carpeta = RAIZ_DATA / f"octrees_{R}" / args.clase / "test"
    archivos = sorted(carpeta.glob("*.npz"))
    if args.indice >= len(archivos):
        raise IndexError(f"Indice fuera de rango ({len(archivos)} disponibles)")
    archivo = archivos[args.indice]
    data = np.load(archivo)
    return data["grid"], int(data["etiqueta"]), archivo.stem


# ──────────────────────────────────────────────────────────────
# VISUALIZACION: objeto + mapa de calor superpuesto
# ──────────────────────────────────────────────────────────────

def graficar_gradcam(
    grid: np.ndarray, mapa_calor: np.ndarray, nombre_muestra: str,
    etiqueta_real, clase_predicha: int, confianza: float, R: int,
):
    """
    Dos paneles lado a lado:
      Izquierda : objeto original coloreado por su normal de superficie
      Derecha   : mismas celdas ocupadas, coloreadas por el mapa de
                  calor Grad-CAM (rojo = mayor influencia en la decision)
    """
    ocupacion = grid[0]
    idx = np.argwhere(ocupacion > 0)
    coords = (idx + 0.5) / R * 2 - 1

    fig = plt.figure(figsize=(16, 8))

    # Panel 1: objeto original (color = normal de superficie)
    ax1 = fig.add_subplot(121, projection="3d")
    normales = grid[1:4]
    vecs_normal = normales[:, idx[:, 0], idx[:, 1], idx[:, 2]].T
    colores_normal = np.clip((vecs_normal + 1.0) / 2.0, 0, 1)
    ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=colores_normal, s=14, marker="s", alpha=0.85)
    ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 1); ax1.set_zlim(-1, 1)
    ax1.set_title("Objeto original\n(color = vector normal de superficie)", fontsize=11)

    # Panel 2: mapa de calor Grad-CAM
    ax2 = fig.add_subplot(122, projection="3d")
    valores_calor = mapa_calor[idx[:, 0], idx[:, 1], idx[:, 2]]
    sc = ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                    c=valores_calor, cmap="jet", s=14, marker="s",
                    alpha=0.85, vmin=0, vmax=1)
    ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1); ax2.set_zlim(-1, 1)
    ax2.set_title("Grad-CAM 3D\n(regiones que mas influyen en la prediccion)", fontsize=11)
    plt.colorbar(sc, ax=ax2, fraction=0.04, pad=0.1, label="Importancia (0=baja, 1=alta)")

    clase_real_txt = CLASES[etiqueta_real] if etiqueta_real is not None else "desconocida"
    clase_pred_txt = CLASES[clase_predicha]
    correcto = ""
    if etiqueta_real is not None:
        correcto = " ✓ CORRECTO" if clase_predicha == etiqueta_real else " ✗ INCORRECTO"

    fig.suptitle(
        f"Muestra: {nombre_muestra}  |  Clase real: {clase_real_txt}  |  "
        f"Prediccion Net5: {clase_pred_txt} ({confianza*100:.1f}%){correcto}\n"
        f"NOTA: Grad-CAM muestra regiones de influencia en la clasificacion, "
        f"NO es una reconstruccion ni generacion del objeto",
        fontsize=11,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    salida = DIR_RESULTADOS / f"gradcam3d_{nombre_muestra}_R{R}.png"
    plt.savefig(salida, dpi=140, bbox_inches="tight")
    print(f"\n[Guardado] Ruta absoluta: {salida.resolve()}")
    print(f"[Guardado] Archivo existe: {salida.exists()}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM 3D para Net5-Octree")
    parser.add_argument("--resolucion", type=int, default=32, choices=[32, 64])
    parser.add_argument("--clase", type=str, default="airplane")
    parser.add_argument("--indice", type=int, default=0)
    parser.add_argument("--archivo", type=str, default=None,
                        help="Ruta a un .off externo (ignora --clase/--indice)")
    args = parser.parse_args()

    R = args.resolucion

    print("=" * 60)
    print(f"  GRAD-CAM 3D — Net5-Octree Resolucion {R}^3")
    print("=" * 60)
    print("\n  NOTA: esto muestra regiones de influencia en la decision")
    print("  del clasificador, NO una reconstruccion/generacion 3D.\n")

    device = get_device()

    print("[1/3] Cargando Net5-Octree...")
    modelo = cargar_modelo_net5(R, device)

    print("[2/3] Obteniendo muestra...")
    grid, etiqueta_real, nombre_muestra = obtener_muestra(args, R)
    print(f"  Muestra: {nombre_muestra}")

    print("[3/3] Calculando Grad-CAM 3D...")
    gradcam = GradCAM3D(modelo)
    x = torch.from_numpy(grid).float().unsqueeze(0).to(device)
    mapa_calor, clase_predicha, confianza = gradcam.generar_mapa(x)

    print(f"\n  Prediccion: {CLASES[clase_predicha]} ({confianza*100:.1f}%)")
    if etiqueta_real is not None:
        print(f"  Clase real: {CLASES[etiqueta_real]}")

    graficar_gradcam(grid, mapa_calor, nombre_muestra, etiqueta_real,
                     clase_predicha, confianza, R)

    print("\nGrad-CAM 3D completado.")


if __name__ == "__main__":
    main()
