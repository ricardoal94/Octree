"""Visualiza un resumen de clasificación sin alterar su alcance declarado.

El título distingue resultados OctNet válidos de diagnósticos densos para
evitar que una figura provisional se presente como evidencia del Objetivo 3.

Uso:
    python fase3_net5/visualizar_matriz_confusion_net5.py RESUMEN.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO / "fase2_octree"))
from preprocesar_octrees import CLASES_MODELNET40  # noqa: E402

CLASES = CLASES_MODELNET40


def cargar_resumen(ruta: Path) -> dict:
    with ruta.open(encoding="utf-8") as archivo:
        resumen = json.load(archivo)
    requeridos = {
        "resolucion", "matriz_confusion", "reporte_clasificacion",
        "test_acc", "mejor_val_acc", "mejor_epoca",
    }
    faltantes = sorted(requeridos - resumen.keys())
    if faltantes:
        raise ValueError(f"Faltan campos en {ruta}: {', '.join(faltantes)}")
    matriz = np.asarray(resumen["matriz_confusion"])
    if matriz.shape != (len(CLASES), len(CLASES)):
        raise ValueError(f"Matriz de confusión inválida: {matriz.shape}")
    return resumen


def etiqueta_alcance(resumen: dict) -> str:
    if resumen.get("valido_como_resultado_objetivo3") is True:
        return "OctNet nativo — resultado del Objetivo 3"
    return "REFERENCIA DENSA — DIAGNÓSTICO, NO RESULTADO DEL OBJETIVO 3"


def graficar_matriz_confusion(resumen: dict, salida_dir: Path, prefijo: str) -> None:
    matriz = np.asarray(resumen["matriz_confusion"])
    matriz_norm = matriz.astype(float) / matriz.sum(axis=1, keepdims=True).clip(min=1)
    resolucion = int(resumen["resolucion"])

    fig, ax = plt.subplots(figsize=(13, 11))
    imagen = ax.imshow(matriz_norm, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(
        imagen, ax=ax, fraction=0.035, pad=0.03,
        label="Tasa de clasificación por clase real",
    )
    ax.set_xticks(range(len(CLASES)), labels=CLASES, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(CLASES)), labels=CLASES, fontsize=6.5)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(
        f"{etiqueta_alcance(resumen)} — R={resolucion}³\n"
        f"Test Acc: {resumen['test_acc'] * 100:.2f}% | "
        f"Val Acc: {resumen['mejor_val_acc'] * 100:.2f}% | "
        f"Mejor época: {resumen['mejor_epoca']}",
    )
    plt.tight_layout()
    for extension in ("png", "svg"):
        destino = salida_dir / f"{prefijo}_matriz_confusion.{extension}"
        plt.savefig(
            destino,
            dpi=150 if extension == "png" else None,
            format=extension,
            bbox_inches="tight",
        )
        print(f"[Guardado] {destino}")
    plt.close(fig)


def graficar_metricas_por_clase(
    resumen: dict,
    salida_dir: Path,
    prefijo: str,
) -> None:
    reporte = resumen["reporte_clasificacion"]
    precision = np.asarray([reporte.get(c, {}).get("precision", 0) for c in CLASES])
    recall = np.asarray([reporte.get(c, {}).get("recall", 0) for c in CLASES])
    f1 = np.asarray([reporte.get(c, {}).get("f1-score", 0) for c in CLASES])
    orden = np.argsort(f1)[::-1]
    x = np.arange(len(CLASES))
    ancho = 0.27

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(x - ancho, precision[orden], ancho, label="Precisión", color="#2196F3")
    ax.bar(x, recall[orden], ancho, label="Recall", color="#FF9800")
    ax.bar(x + ancho, f1[orden], ancho, label="F1-score", color="#4CAF50")
    ax.set_xticks(x, labels=[CLASES[i] for i in orden], rotation=90, fontsize=8)
    ax.set_ylabel("Puntuación")
    ax.set_ylim(0, 1.08)
    ax.axhline(
        resumen["test_acc"], color="red", linestyle="--", linewidth=1.2,
        label=f"Test Acc global ({resumen['test_acc'] * 100:.2f}%)",
    )
    ax.set_title(etiqueta_alcance(resumen))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for extension in ("png", "svg"):
        destino = salida_dir / f"{prefijo}_metricas_por_clase.{extension}"
        plt.savefig(
            destino,
            dpi=150 if extension == "png" else None,
            format=extension,
            bbox_inches="tight",
        )
        print(f"[Guardado] {destino}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("resumen", type=Path, nargs="+")
    parser.add_argument("--salida-dir", type=Path)
    args = parser.parse_args()

    for ruta in args.resumen:
        resumen = cargar_resumen(ruta)
        salida_dir = args.salida_dir or ruta.resolve().parent
        salida_dir.mkdir(parents=True, exist_ok=True)
        graficar_matriz_confusion(resumen, salida_dir, ruta.stem)
        graficar_metricas_por_clase(resumen, salida_dir, ruta.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
