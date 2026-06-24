"""
comparar_predicciones.py
===========================
Carga los 3 modelos entrenados (SVM, Random Forest, Net5-Octree) y
compara sus predicciones sobre la misma muestra: ya sea una muestra
del conjunto de test, o un archivo .off arbitrario.

Uso:
    # Probar con una muestra del test set por clase + indice
    python comparar_predicciones.py --clase airplane --indice 0 --resolucion 32

    # Probar con un archivo .off cualquiera (no necesita estar en ModelNet40)
    python comparar_predicciones.py --archivo "C:/ruta/al/modelo.off" --resolucion 32

    # Probar con una muestra aleatoria del test set
    python comparar_predicciones.py --aleatorio --resolucion 32
"""

import sys
import argparse
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "fase2_octree"))
sys.path.insert(0, str(Path(__file__).parent.parent / "fase3_hce"))
sys.path.insert(0, str(Path(__file__).parent.parent / "fase3_net5"))

from octree           import malla_a_octree, profundidad_de
from hce_extraccion   import extraer_descriptores_hce
from net5_modelo      import Net5Octree, get_device

# ── Rutas ──────────────────────────────────────────────────────
RAIZ_DATASET = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\Dataset\ModelNet40")
RAIZ_DATA    = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\data")
DIR_CKPT     = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\checkpoints")

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
# CARGA DE MODELOS
# ──────────────────────────────────────────────────────────────

def cargar_modelos_hce(R: int) -> tuple:
    """Carga SVM, Random Forest y el scaler guardados."""
    svm    = joblib.load(DIR_CKPT / f"hce_svm_R{R}.joblib")
    rf     = joblib.load(DIR_CKPT / f"hce_rf_R{R}.joblib")
    scaler = joblib.load(DIR_CKPT / f"hce_scaler_R{R}.joblib")
    return svm, rf, scaler


def cargar_modelo_net5(R: int, device: torch.device) -> torch.nn.Module:
    """Carga el checkpoint de Net5-Octree y lo pone en modo evaluacion."""
    modelo = Net5Octree(resolucion=R, num_clases=40).to(device)
    ckpt = torch.load(DIR_CKPT / f"net5_mejor_R{R}.pth", map_location=device)
    modelo.load_state_dict(ckpt["model_state"])
    modelo.eval()
    return modelo


# ──────────────────────────────────────────────────────────────
# OBTENER LA MUESTRA (desde .off directo o desde .npz precomputado)
# ──────────────────────────────────────────────────────────────

def obtener_grid_y_etiqueta(args, R: int) -> tuple:
    """
    Retorna (grid, etiqueta_real, nombre_muestra).
    etiqueta_real puede ser None si se usa un archivo .off externo
    sin clase conocida.
    """
    if args.archivo:
        print(f"\n[Carga] Generando octree desde archivo externo: {args.archivo}")
        grid = malla_a_octree(args.archivo, resolucion=R, n_puntos_muestreo=20000, seed=42)
        return grid, None, Path(args.archivo).stem

    if args.aleatorio:
        rng = np.random.default_rng()
        clase = rng.choice(CLASES)
        carpeta = RAIZ_DATA / f"octrees_{R}" / clase / "test"
        archivos = sorted(carpeta.glob("*.npz"))
        archivo = rng.choice(archivos)
    else:
        carpeta = RAIZ_DATA / f"octrees_{R}" / args.clase / "test"
        archivos = sorted(carpeta.glob("*.npz"))
        if args.indice >= len(archivos):
            raise IndexError(f"Indice {args.indice} fuera de rango "
                            f"({len(archivos)} muestras disponibles para '{args.clase}')")
        archivo = archivos[args.indice]

    data = np.load(archivo)
    grid = data["grid"]
    etiqueta_real = int(data["etiqueta"])

    return grid, etiqueta_real, archivo.stem


# ──────────────────────────────────────────────────────────────
# PREDICCIONES DE CADA MODELO
# ──────────────────────────────────────────────────────────────

def predecir_svm(svm, scaler, grid: np.ndarray, R: int) -> tuple:
    L = profundidad_de(R)
    feats = extraer_descriptores_hce(grid, L).reshape(1, -1)
    feats_s = scaler.transform(feats)

    pred = svm.predict(feats_s)[0]

    # SVC con probability=False no tiene predict_proba; usamos decision_function
    decision = svm.decision_function(feats_s)[0]
    # Convertir a pseudo-confianza via softmax sobre decision_function
    exp = np.exp(decision - decision.max())
    probs = exp / exp.sum()
    confianza = probs[pred]

    return int(pred), float(confianza), probs


def predecir_rf(rf, grid: np.ndarray, R: int) -> tuple:
    L = profundidad_de(R)
    feats = extraer_descriptores_hce(grid, L).reshape(1, -1)

    pred = rf.predict(feats)[0]
    probs = rf.predict_proba(feats)[0]
    confianza = probs[pred]

    return int(pred), float(confianza), probs


def predecir_net5(modelo, grid: np.ndarray, device: torch.device) -> tuple:
    x = torch.from_numpy(grid).float().unsqueeze(0).to(device)   # (1, 4, R, R, R)

    with torch.no_grad():
        logits = modelo(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred = int(probs.argmax())
    confianza = float(probs[pred])

    return pred, confianza, probs


# ──────────────────────────────────────────────────────────────
# VISUALIZACION 3D DEL OBJETO CON LAS PREDICCIONES
# ──────────────────────────────────────────────────────────────

def graficar_objeto_3d_con_predicciones(
    grid: np.ndarray, nombre_muestra: str, etiqueta_real,
    resultados: dict, R: int, usar_voxels: bool = False,
):
    """
    Dibuja el objeto 3D (coloreado por su vector normal, igual que en
    visualizar_octree_3d.py) y agrega las predicciones de los 3 modelos
    como texto superpuesto, indicando con un check/cruz si acertaron.
    """
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    ocupacion = grid[0]
    normales  = grid[1:4]

    if usar_voxels:
        ocup_bool = ocupacion > 0
        color_rgb = (np.transpose(normales, (1, 2, 3, 0)) + 1.0) / 2.0
        alpha = np.ones(grid.shape[1:]) * 0.9
        colores = np.concatenate([color_rgb, alpha[..., None]], axis=-1)
        ax.voxels(ocup_bool, facecolors=colores, edgecolor="k", linewidth=0.1)
    else:
        idx = np.argwhere(ocupacion > 0)
        coords = (idx + 0.5) / R * 2 - 1
        vecs_normal = normales[:, idx[:, 0], idx[:, 1], idx[:, 2]].T
        colores = np.clip((vecs_normal + 1.0) / 2.0, 0, 1)
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                  c=colores, s=14, marker="s", alpha=0.85)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    # Titulo principal
    clase_real_txt = CLASES[etiqueta_real] if etiqueta_real is not None else "desconocida"
    titulo = f"Muestra: {nombre_muestra}  |  Clase real: {clase_real_txt}  |  R={R}^3"
    ax.set_title(titulo, fontsize=11, pad=15)

    # Texto con las predicciones de los 3 modelos, debajo de la figura
    lineas_pred = []
    for nombre_modelo, (pred, confianza, _) in resultados.items():
        clase_pred = CLASES[pred]
        if etiqueta_real is not None:
            marca = "✓" if pred == etiqueta_real else "✗"
        else:
            marca = "—"
        lineas_pred.append(f"{marca} {nombre_modelo}: {clase_pred} ({confianza*100:.1f}%)")

    texto_predicciones = "\n".join(lineas_pred)
    fig.text(0.5, 0.02, texto_predicciones, ha="center", va="bottom",
             fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.9))

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    # Ruta absoluta, en la carpeta resultados/ del proyecto (consistente
    # con el resto de las visualizaciones generadas en las fases previas)
    DIR_RESULTADOS = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\resultados")
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    salida = DIR_RESULTADOS / f"prediccion_3d_{nombre_muestra}_R{R}.png"

    plt.savefig(salida, dpi=140, bbox_inches="tight")
    print(f"\n[Guardado] Ruta absoluta: {salida.resolve()}")
    print(f"[Guardado] El archivo existe: {salida.exists()}  |  Tamano: "
         f"{salida.stat().st_size / 1024:.1f} KB" if salida.exists() else "[ERROR] El archivo NO se creo")

    plt.show()


# ──────────────────────────────────────────────────────────────
# REPORTE COMPARATIVO
# ──────────────────────────────────────────────────────────────

def imprimir_comparativa(nombre_muestra, etiqueta_real, resultados: dict, R: int):
    print("\n" + "=" * 65)
    print(f"  COMPARATIVA DE PREDICCIONES — Resolucion {R}^3")
    print("=" * 65)
    print(f"  Muestra      : {nombre_muestra}")
    if etiqueta_real is not None:
        print(f"  Clase real   : {CLASES[etiqueta_real]}")
    else:
        print(f"  Clase real   : desconocida (archivo externo)")
    print("-" * 65)

    print(f"  {'Modelo':<15}{'Prediccion':<18}{'Confianza':<12}{'Correcto?'}")
    print("-" * 65)

    for nombre_modelo, (pred, confianza, _) in resultados.items():
        clase_predicha = CLASES[pred]
        if etiqueta_real is not None:
            correcto = "✓ SI" if pred == etiqueta_real else "✗ NO"
        else:
            correcto = "—"
        print(f"  {nombre_modelo:<15}{clase_predicha:<18}{confianza*100:>6.2f}%     {correcto}")

    print("=" * 65)

    # Top-3 de cada modelo
    print("\n  Top-3 predicciones por modelo:")
    for nombre_modelo, (pred, confianza, probs) in resultados.items():
        top3_idx = np.argsort(probs)[::-1][:3]
        top3_str = ", ".join(f"{CLASES[i]} ({probs[i]*100:.1f}%)" for i in top3_idx)
        print(f"    {nombre_modelo:<15}: {top3_str}")

    print()


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comparar predicciones SVM vs RF vs Net5")
    parser.add_argument("--resolucion", type=int, default=32, choices=[32, 64])
    parser.add_argument("--clase", type=str, default="airplane",
                        help="Clase de ModelNet40 (usado si no se pasa --archivo ni --aleatorio)")
    parser.add_argument("--indice", type=int, default=0,
                        help="Indice de la muestra dentro de la clase (conjunto test)")
    parser.add_argument("--archivo", type=str, default=None,
                        help="Ruta a un archivo .off externo (ignora --clase/--indice)")
    parser.add_argument("--aleatorio", action="store_true",
                        help="Elegir una muestra aleatoria del test set")
    parser.add_argument("--voxels", action="store_true",
                        help="Usar voxel plot solido en vez de scatter (mas lento, mas realista)")
    parser.add_argument("--sin-grafica", action="store_true",
                        help="Omitir la visualizacion 3D, solo mostrar texto")
    args = parser.parse_args()

    R = args.resolucion

    print("=" * 65)
    print(f"  CARGANDO MODELOS — Resolucion {R}^3")
    print("=" * 65)

    device = get_device()

    print("\n[1/3] Cargando SVM y Random Forest (HCE)...")
    svm, rf, scaler = cargar_modelos_hce(R)
    print("  OK")

    print("\n[2/3] Cargando Net5-Octree...")
    net5 = cargar_modelo_net5(R, device)
    print("  OK")

    print("\n[3/3] Obteniendo muestra a clasificar...")
    grid, etiqueta_real, nombre_muestra = obtener_grid_y_etiqueta(args, R)
    print(f"  Muestra: {nombre_muestra}")

    # Predicciones
    resultados = {}
    resultados["SVM"]           = predecir_svm(svm, scaler, grid, R)
    resultados["Random Forest"] = predecir_rf(rf, grid, R)
    resultados["Net5-Octree"]   = predecir_net5(net5, grid, device)

    imprimir_comparativa(nombre_muestra, etiqueta_real, resultados, R)

    if not args.sin_grafica:
        print("[Visualizacion] Generando figura 3D...")
        try:
            graficar_objeto_3d_con_predicciones(
                grid, nombre_muestra, etiqueta_real, resultados, R,
                usar_voxels=args.voxels,
            )
        except Exception as e:
            import traceback
            print(f"\n[ERROR] Fallo al generar la figura 3D: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()