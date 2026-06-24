"""
visualizar_features_3d.py
============================
Visualizaciones 3D del espacio de features extraidos por HCE y del
comportamiento del SVM, para complementar las graficas 2D ya generadas.

Genera 2 tipos de figuras:

  1. Embedding 3D del espacio de features (PCA o t-SNE a 3 componentes),
     coloreado por clase real. Muestra si los 17/18 descriptores HCE
     separan bien las 40 categorias de ModelNet40.

  2. Superficie de decision aproximada del SVM, proyectando a los primeros
     3 componentes de PCA y re-entrenando un SVM auxiliar SOLO sobre esos
     3 componentes (para poder graficar la frontera). Esto es una
     APROXIMACION ILUSTRATIVA, no el modelo real de 17/18 dimensiones,
     y se etiqueta explicitamente como tal en el titulo de la grafica.

Uso:
    python visualizar_features_3d.py --resolucion 32 --metodo pca
    python visualizar_features_3d.py --resolucion 32 --metodo tsne
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

DIR_LOGS       = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\logs")
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


def cargar_features(resolucion: int) -> tuple:
    """Carga el cache de features guardado por fase3_hce_entrenamiento.py."""
    ruta = DIR_LOGS / f"hce_features_R{resolucion}.npz"
    data = np.load(ruta)
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]


# ──────────────────────────────────────────────────────────────
# 1. EMBEDDING 3D DEL ESPACIO DE FEATURES
# ──────────────────────────────────────────────────────────────

def graficar_embedding_3d(X: np.ndarray, y: np.ndarray, resolucion: int,
                          metodo: str = "pca", n_muestras_max: int = 3000,
                          seed: int = 42):
    """
    Reduce los features a 3 dimensiones (PCA o t-SNE) y grafica un
    scatter 3D coloreado por clase real.
    """
    rng = np.random.default_rng(seed)

    # Submuestrear si hay demasiados puntos (t-SNE es costoso)
    if len(X) > n_muestras_max:
        idx = rng.choice(len(X), n_muestras_max, replace=False)
        X_sub, y_sub = X[idx], y[idx]
    else:
        X_sub, y_sub = X, y

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    if metodo == "pca":
        reductor = PCA(n_components=3, random_state=seed)
        X_3d = reductor.fit_transform(X_scaled)
        varianza_explicada = reductor.explained_variance_ratio_.sum() * 100
        subtitulo = f"PCA — varianza explicada: {varianza_explicada:.1f}%"
    else:
        reductor = TSNE(n_components=3, random_state=seed, perplexity=30, init="pca")
        X_3d = reductor.fit_transform(X_scaled)
        subtitulo = "t-SNE (perplexity=30)"

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Colormap con 40 colores distintos
    cmap = plt.get_cmap("tab20")
    colores_clase = [cmap(i % 20) if i < 20 else plt.get_cmap("tab20b")(i % 20)
                     for i in range(40)]

    for clase_idx in np.unique(y_sub):
        mask = y_sub == clase_idx
        ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                  s=12, alpha=0.6, color=colores_clase[clase_idx],
                  label=CLASES[clase_idx])

    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.set_zlabel("Componente 3")
    ax.set_title(
        f"Embedding 3D del espacio de features HCE — Resolucion {resolucion}^3\n"
        f"{subtitulo}  |  {len(X_sub)} muestras, 40 clases"
    )

    # Leyenda compacta fuera del plot (40 clases es mucho para una leyenda normal)
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=6, ncol=2)

    plt.tight_layout()
    salida = DIR_RESULTADOS / f"embedding3d_{metodo}_R{resolucion}.png"
    plt.savefig(salida, dpi=140, bbox_inches="tight")
    print(f"[Guardado] {salida}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# 2. SUPERFICIE DE DECISION DEL SVM (APROXIMACION ILUSTRATIVA)
# ──────────────────────────────────────────────────────────────

def graficar_superficie_decision_svm_3d(
    X: np.ndarray, y: np.ndarray, resolucion: int,
    clases_a_mostrar: list = None, seed: int = 42,
):
    """
    IMPORTANTE: esta es una aproximacion ilustrativa. El SVM real opera
    sobre 17/18 dimensiones; aqui se proyectan los datos a 3 componentes
    PCA y se entrena un SVM AUXILIAR solo sobre esas 3 dimensiones, unica-
    mente para poder visualizar una frontera de decision. La exactitud de
    este SVM auxiliar NO debe reportarse como resultado del experimento
    real (eso ya esta en resumen_hce_R<R>.json).

    Para mantener la figura legible, se muestran solo 4 clases elegidas
    (por defecto, las primeras 4 mas representativas).
    """
    if clases_a_mostrar is None:
        clases_a_mostrar = [0, 1, 7, 8]   # airplane, bathtub, car, chair

    mask = np.isin(y, clases_a_mostrar)
    X_sub, y_sub = X[mask], y[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    pca = PCA(n_components=3, random_state=seed)
    X_3d = pca.fit_transform(X_scaled)
    varianza_explicada = pca.explained_variance_ratio_.sum() * 100

    # Re-mapear etiquetas a 0..3 para el SVM auxiliar
    mapa_etiquetas = {c: i for i, c in enumerate(clases_a_mostrar)}
    y_mapeado = np.array([mapa_etiquetas[c] for c in y_sub])

    svm_aux = SVC(kernel="rbf", C=10, gamma="scale", random_state=seed)
    svm_aux.fit(X_3d, y_mapeado)
    acc_aux = svm_aux.score(X_3d, y_mapeado)

    # Crear malla 3D para evaluar la superficie de decision
    margen = 0.5
    x_min, x_max = X_3d[:, 0].min() - margen, X_3d[:, 0].max() + margen
    y_min, y_max = X_3d[:, 1].min() - margen, X_3d[:, 1].max() + margen
    z_min, z_max = X_3d[:, 2].min() - margen, X_3d[:, 2].max() + margen

    n_grid = 18   # resolucion de la malla 3D (cubica: n_grid^3 evaluaciones)
    xx, yy, zz = np.meshgrid(
        np.linspace(x_min, x_max, n_grid),
        np.linspace(y_min, y_max, n_grid),
        np.linspace(z_min, z_max, n_grid),
    )
    puntos_malla = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    predicciones = svm_aux.predict(puntos_malla)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab10")

    # Puntos de la malla, coloreados por la clase predicha (transparencia baja)
    # para insinuar las regiones de decision sin saturar la figura
    for clase_local in np.unique(predicciones):
        m = predicciones == clase_local
        ax.scatter(puntos_malla[m, 0], puntos_malla[m, 1], puntos_malla[m, 2],
                  s=3, alpha=0.03, color=cmap(clase_local))

    # Puntos de datos reales, marcados con borde negro y color solido
    for clase_local, clase_global in enumerate(clases_a_mostrar):
        m = y_mapeado == clase_local
        ax.scatter(X_3d[m, 0], X_3d[m, 1], X_3d[m, 2],
                  s=35, alpha=0.9, color=cmap(clase_local),
                  edgecolor="k", linewidth=0.4,
                  label=CLASES[clase_global])

    ax.set_xlabel("PCA-1"); ax.set_ylabel("PCA-2"); ax.set_zlabel("PCA-3")
    ax.set_title(
        f"Aproximacion ilustrativa de fronteras del SVM (proyeccion PCA-3D)\n"
        f"SOLO {len(clases_a_mostrar)} clases, SVM auxiliar entrenado en 3D "
        f"(acc en esta proyeccion: {acc_aux*100:.1f}%)\n"
        f"Varianza explicada por PCA: {varianza_explicada:.1f}%  "
        f"— NO representa el modelo real de {X.shape[1]} dimensiones",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    salida = DIR_RESULTADOS / f"svm_frontera_ilustrativa_R{resolucion}.png"
    plt.savefig(salida, dpi=140, bbox_inches="tight")
    print(f"[Guardado] {salida}")
    plt.close()

    return acc_aux, varianza_explicada


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolucion", type=int, default=32, choices=[32, 64])
    parser.add_argument("--metodo", type=str, default="pca", choices=["pca", "tsne"])
    args = parser.parse_args()
    R = args.resolucion

    print("=" * 60)
    print(f"  VISUALIZACION 3D DE FEATURES Y SVM — Resolucion {R}^3")
    print("=" * 60)

    print("\n[1/2] Cargando features cacheados...")
    X_train, y_train, X_test, y_test = cargar_features(R)
    print(f"  X_train: {X_train.shape}")

    print(f"\n[2/2] Generando embedding 3D ({args.metodo.upper()})...")
    graficar_embedding_3d(X_train, y_train, R, metodo=args.metodo)

    print("\n[3/3] Generando aproximacion ilustrativa de frontera SVM...")
    acc_aux, var_exp = graficar_superficie_decision_svm_3d(X_train, y_train, R)
    print(f"  (Esta cifra es solo de la proyeccion 3D ilustrativa, NO el resultado real)")
    print(f"  Acc en proyeccion 3D: {acc_aux*100:.1f}% | Varianza explicada: {var_exp:.1f}%")

    print(f"\nGraficas guardadas en: {DIR_RESULTADOS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
