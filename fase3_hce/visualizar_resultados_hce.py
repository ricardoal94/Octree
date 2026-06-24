"""
visualizar_resultados_hce.py
==============================
Genera graficas a partir de los resultados guardados por
fase3_hce_entrenamiento.py (resumen_hce_R<R>.json):

  1. Matriz de confusion (SVM y Random Forest) — mapa de calor
  2. Comparacion de accuracy entre SVM y Random Forest
  3. Comparacion de tiempo de inferencia y tamano de modelo
  4. Feature importance del Random Forest (grafico de barras)
  5. Precision/Recall/F1 por clase (Random Forest)

Uso:
    python visualizar_resultados_hce.py --resolucion 32
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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


def cargar_resumen(resolucion: int) -> dict:
    ruta = DIR_RESULTADOS / f"resumen_hce_R{resolucion}.json"
    with open(ruta, "r") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────
# 1. MATRIZ DE CONFUSION
# ──────────────────────────────────────────────────────────────

def graficar_matriz_confusion(resumen: dict, resolucion: int):
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    for ax, modelo_key, titulo in zip(
        axes, ["svm", "random_forest"], ["SVM (RBF)", "Random Forest"]
    ):
        matriz = np.array(resumen[modelo_key]["matriz_confusion"])
        matriz_norm = matriz.astype(float) / matriz.sum(axis=1, keepdims=True).clip(min=1)

        im = ax.imshow(matriz_norm, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASES)))
        ax.set_yticks(range(len(CLASES)))
        ax.set_xticklabels(CLASES, rotation=90, fontsize=6)
        ax.set_yticklabels(CLASES, fontsize=6)
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Clase real")
        acc = resumen[modelo_key]["test_acc"] * 100
        ax.set_title(f"{titulo} — Matriz de Confusion (Test Acc: {acc:.2f}%)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    salida = DIR_RESULTADOS / f"matriz_confusion_hce_R{resolucion}.png"
    plt.savefig(salida, dpi=150, bbox_inches="tight")
    print(f"[Guardado] {salida}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# 2. COMPARACION DE ACCURACY
# ──────────────────────────────────────────────────────────────

def graficar_comparacion_accuracy(resumen: dict, resolucion: int):
    modelos = ["SVM", "Random Forest"]
    val_acc  = [resumen["svm"]["val_acc"] * 100, resumen["random_forest"]["val_acc"] * 100]
    test_acc = [resumen["svm"]["test_acc"] * 100, resumen["random_forest"]["test_acc"] * 100]

    x = np.arange(len(modelos))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    barras1 = ax.bar(x - ancho/2, val_acc, ancho, label="Validacion", color="steelblue")
    barras2 = ax.bar(x + ancho/2, test_acc, ancho, label="Test", color="darkorange")

    for barras in (barras1, barras2):
        for b in barras:
            altura = b.get_height()
            ax.annotate(f"{altura:.1f}%", (b.get_x() + b.get_width()/2, altura),
                       ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Comparacion de Accuracy — HCE Resolucion {resolucion}^3")
    ax.set_xticks(x)
    ax.set_xticklabels(modelos)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    salida = DIR_RESULTADOS / f"comparacion_accuracy_hce_R{resolucion}.png"
    plt.savefig(salida, dpi=150, bbox_inches="tight")
    print(f"[Guardado] {salida}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# 3. COSTO COMPUTACIONAL: tiempo de inferencia y tamano de modelo
# ──────────────────────────────────────────────────────────────

def graficar_costo_computacional(resumen: dict, resolucion: int):
    modelos = ["SVM", "Random Forest"]
    tiempo_ms = [
        resumen["svm"]["tiempo_inferencia_promedio_ms"],
        resumen["random_forest"]["tiempo_inferencia_promedio_ms"],
    ]
    tamano_mb = [
        resumen["svm"]["tamano_modelo_mb"],
        resumen["random_forest"]["tamano_modelo_mb"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    colores = ["steelblue", "darkorange"]

    axes[0].bar(modelos, tiempo_ms, color=colores)
    axes[0].set_ylabel("Tiempo (ms)")
    axes[0].set_title("Tiempo de inferencia por muestra")
    for i, v in enumerate(tiempo_ms):
        axes[0].text(i, v, f"{v:.3f} ms", ha="center", va="bottom")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(modelos, tamano_mb, color=colores)
    axes[1].set_ylabel("Tamano (MB)")
    axes[1].set_title("Tamano del modelo guardado")
    for i, v in enumerate(tamano_mb):
        axes[1].text(i, v, f"{v:.2f} MB", ha="center", va="bottom")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Costo Computacional — HCE Resolucion {resolucion}^3")
    plt.tight_layout()
    salida = DIR_RESULTADOS / f"costo_computacional_hce_R{resolucion}.png"
    plt.savefig(salida, dpi=150, bbox_inches="tight")
    print(f"[Guardado] {salida}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# 4. FEATURE IMPORTANCE (Random Forest)
# ──────────────────────────────────────────────────────────────

def graficar_feature_importance(resumen: dict, resolucion: int):
    importancias = resumen["random_forest"]["feature_importances"]
    nombres = [it["feature"] for it in importancias]
    valores = [it["importancia"] for it in importancias]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(nombres))

    ax.barh(y_pos, valores, color="seagreen")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(nombres, fontsize=9)
    ax.invert_yaxis()   # el mas importante arriba
    ax.set_xlabel("Importancia")
    ax.set_title(f"Importancia de Features (Random Forest) — Resolucion {resolucion}^3")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    salida = DIR_RESULTADOS / f"feature_importance_hce_R{resolucion}.png"
    plt.savefig(salida, dpi=150, bbox_inches="tight")
    print(f"[Guardado] {salida}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# 5. PRECISION / RECALL / F1 POR CLASE
# ──────────────────────────────────────────────────────────────

def graficar_metricas_por_clase(resumen: dict, resolucion: int, modelo_key="random_forest"):
    reporte = resumen[modelo_key]["reporte_clasificacion"]

    precision = [reporte[c]["precision"] for c in CLASES]
    recall    = [reporte[c]["recall"] for c in CLASES]
    f1        = [reporte[c]["f1-score"] for c in CLASES]

    x = np.arange(len(CLASES))
    ancho = 0.27

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(x - ancho, precision, ancho, label="Precision", color="steelblue")
    ax.bar(x,          recall,    ancho, label="Recall", color="darkorange")
    ax.bar(x + ancho,  f1,        ancho, label="F1-score", color="seagreen")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASES, rotation=90, fontsize=8)
    ax.set_ylabel("Puntuacion")
    nombre_modelo = "Random Forest" if modelo_key == "random_forest" else "SVM"
    ax.set_title(f"Precision / Recall / F1 por clase — {nombre_modelo} R{resolucion}^3")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    salida = DIR_RESULTADOS / f"metricas_por_clase_{modelo_key}_R{resolucion}.png"
    plt.savefig(salida, dpi=150, bbox_inches="tight")
    print(f"[Guardado] {salida}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolucion", type=int, default=32, choices=[32, 64])
    args = parser.parse_args()
    R = args.resolucion

    print("=" * 60)
    print(f"  VISUALIZACION DE RESULTADOS HCE — Resolucion {R}^3")
    print("=" * 60)

    resumen = cargar_resumen(R)

    print("\nGenerando graficas...")
    graficar_matriz_confusion(resumen, R)
    graficar_comparacion_accuracy(resumen, R)
    graficar_costo_computacional(resumen, R)
    graficar_feature_importance(resumen, R)
    graficar_metricas_por_clase(resumen, R, "random_forest")
    graficar_metricas_por_clase(resumen, R, "svm")

    print(f"\nTodas las graficas guardadas en: {DIR_RESULTADOS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
