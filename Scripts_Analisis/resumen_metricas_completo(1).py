"""
resumen_metricas_completo.py
==============================
Lee los JSON de resultados de HCE (SVM, Random Forest) y Net5-Octree
para ambas resoluciones (32^3 y 64^3) y genera una tabla comparativa
completa con todas las metricas de la Fase 4 de la metodologia:

  - Tiempo de entrenamiento
  - Tiempo de inferencia por muestra
  - Memoria pico (VRAM/RAM)
  - Tamano del modelo guardado (MB)
  - Exactitud de validacion
  - Exactitud de test
  - Epoca optima (solo Net5)

Salida:
  - Tabla en consola
  - CSV exportable para LaTeX
  - PNG con la tabla como figura

Uso:
    python resumen_metricas_completo.py
"""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

DIR_RESULTADOS = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\resultados")
DIR_LOGS       = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\logs")


# ──────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────────────────────

def cargar_json(ruta: Path) -> dict:
    if not ruta.exists():
        print(f"  [Aviso] No encontrado: {ruta.name}")
        return {}
    with open(ruta) as f:
        return json.load(f)


def extraer_tiempo_entrenamiento_hce(resumen: dict, modelo: str) -> str:
    """HCE guarda el tiempo de busqueda de hiperparametros para SVM
    y el tiempo de entrenamiento para RF."""
    datos = resumen.get(modelo, {})
    if modelo == "svm":
        t = datos.get("tiempo_busqueda_hiperparam_s", None)
        return f"{t:.1f} s" if t else "N/D"
    else:
        t = datos.get("tiempo_entrenamiento_s", None)
        return f"{t:.1f} s" if t else "N/D"


def cargar_todas_metricas() -> list:
    """
    Retorna una lista de dicts, uno por modelo/resolucion, con todas
    las metricas comparables.
    """
    filas = []

    for R in [32, 64]:
        hce = cargar_json(DIR_RESULTADOS / f"resumen_hce_R{R}.json")
        net5 = cargar_json(DIR_RESULTADOS / f"resumen_net5_R{R}.json")

        if hce:
            # SVM
            svm = hce.get("svm", {})
            filas.append({
                "Modelo":         f"SVM (HCE)",
                "Resolucion":     f"{R}³",
                "t_entreno":      extraer_tiempo_entrenamiento_hce(hce, "svm"),
                "t_inf_ms":       svm.get("tiempo_inferencia_promedio_ms", "N/D"),
                "mem_pico_mb":    "CPU / RAM",
                "tamano_mb":      svm.get("tamano_modelo_mb", "N/D"),
                "val_acc_pct":    round(svm.get("val_acc", 0) * 100, 2),
                "test_acc_pct":   round(svm.get("test_acc", 0) * 100, 2),
                "mejor_epoca":    "N/A",
                "hiperparametros": str(svm.get("mejores_hiperparametros", "")),
            })

            # Random Forest
            rf = hce.get("random_forest", {})
            filas.append({
                "Modelo":         f"Random Forest (HCE)",
                "Resolucion":     f"{R}³",
                "t_entreno":      extraer_tiempo_entrenamiento_hce(hce, "random_forest"),
                "t_inf_ms":       rf.get("tiempo_inferencia_promedio_ms", "N/D"),
                "mem_pico_mb":    "CPU / RAM",
                "tamano_mb":      rf.get("tamano_modelo_mb", "N/D"),
                "val_acc_pct":    round(rf.get("val_acc", 0) * 100, 2),
                "test_acc_pct":   round(rf.get("test_acc", 0) * 100, 2),
                "mejor_epoca":    "N/A",
                "hiperparametros": f"n_estimators=100",
            })

        if net5:
            # Tiempo total de entrenamiento
            t_min = net5.get("tiempo_total_min", None)
            t_str = f"{t_min:.1f} min" if t_min else "N/D"

            filas.append({
                "Modelo":         f"Net5-Octree",
                "Resolucion":     f"{R}³",
                "t_entreno":      t_str,
                "t_inf_ms":       net5.get("tiempo_inferencia_promedio_ms", "N/D"),
                "mem_pico_mb":    net5.get("vram_pico_mb", "N/D"),
                "tamano_mb":      net5.get("tamano_modelo_mb", "N/D"),
                "val_acc_pct":    round(net5.get("mejor_val_acc", 0) * 100, 2),
                "test_acc_pct":   round(net5.get("test_acc", 0) * 100, 2),
                "mejor_epoca":    net5.get("mejor_epoca", "N/D"),
                "hiperparametros": "Adam, lr=0.001, StepLR",
            })

    return filas


# ──────────────────────────────────────────────────────────────
# TABLA EN CONSOLA
# ──────────────────────────────────────────────────────────────

def imprimir_tabla(filas: list):
    print("\n" + "=" * 100)
    print("  RESUMEN COMPLETO DE METRICAS — HCE y Net5-Octree")
    print("=" * 100)

    encabezados = [
        "Modelo", "Res.", "T.Entreno", "T.Inf(ms)",
        "Mem.Pico(MB)", "Tamano(MB)", "Val Acc(%)", "Test Acc(%)", "Ep.Optima"
    ]
    anchos = [22, 5, 12, 11, 14, 12, 11, 12, 10]

    # Encabezado
    header = "  " + "  ".join(f"{h:<{w}}" for h, w in zip(encabezados, anchos))
    print(header)
    print("  " + "-" * 98)

    sep_anterior = None
    for fila in filas:
        res = fila["Resolucion"]
        if sep_anterior and sep_anterior != res:
            print("  " + "-" * 98)
        sep_anterior = res

        valores = [
            fila["Modelo"],
            fila["Resolucion"],
            str(fila["t_entreno"]),
            str(fila["t_inf_ms"]),
            str(fila["mem_pico_mb"]),
            str(fila["tamano_mb"]),
            str(fila["val_acc_pct"]),
            str(fila["test_acc_pct"]),
            str(fila["mejor_epoca"]),
        ]
        linea = "  " + "  ".join(f"{v:<{w}}" for v, w in zip(valores, anchos))
        print(linea)

    print("=" * 100)


# ──────────────────────────────────────────────────────────────
# EXPORTAR CSV
# ──────────────────────────────────────────────────────────────

def exportar_csv(filas: list):
    ruta = DIR_RESULTADOS / "resumen_metricas_completo.csv"
    campos = [
        "Modelo", "Resolucion", "t_entreno", "t_inf_ms",
        "mem_pico_mb", "tamano_mb", "val_acc_pct", "test_acc_pct",
        "mejor_epoca", "hiperparametros"
    ]
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"\n[CSV] Guardado: {ruta}")


# ──────────────────────────────────────────────────────────────
# FIGURA: tabla como imagen + grafico de barras comparativo
# ──────────────────────────────────────────────────────────────

def graficar_comparativa(filas: list):
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    colores_modelo = {
        "SVM (HCE)":          "#2196F3",
        "Random Forest (HCE)":"#FF9800",
        "Net5-Octree":        "#4CAF50",
    }

    etiquetas = [f"{f['Modelo']}\n{f['Resolucion']}" for f in filas]
    colores   = [colores_modelo.get(f["Modelo"], "gray") for f in filas]

    def safe_float(v):
        try: return float(v)
        except: return 0.0

    # ── 1. Test Accuracy ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    vals = [safe_float(f["test_acc_pct"]) for f in filas]
    bars = ax1.bar(etiquetas, vals, color=colores)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax1.set_title("Exactitud en Test (%)", fontsize=11)
    ax1.set_ylim(60, 100)
    ax1.set_ylabel("%")
    ax1.tick_params(axis="x", labelsize=7)
    ax1.grid(axis="y", alpha=0.3)

    # ── 2. Validacion Accuracy ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    vals = [safe_float(f["val_acc_pct"]) for f in filas]
    bars = ax2.bar(etiquetas, vals, color=colores)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax2.set_title("Exactitud en Validación (%)", fontsize=11)
    ax2.set_ylim(60, 100)
    ax2.set_ylabel("%")
    ax2.tick_params(axis="x", labelsize=7)
    ax2.grid(axis="y", alpha=0.3)

    # ── 3. Tiempo de inferencia ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    vals = [safe_float(f["t_inf_ms"]) for f in filas]
    bars = ax3.bar(etiquetas, vals, color=colores)
    for b, v in zip(bars, vals):
        ax3.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax3.set_title("Tiempo de Inferencia por Muestra (ms)", fontsize=11)
    ax3.set_ylabel("ms")
    ax3.tick_params(axis="x", labelsize=7)
    ax3.grid(axis="y", alpha=0.3)

    # ── 4. Tamano del modelo ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    vals = [safe_float(f["tamano_mb"]) for f in filas]
    bars = ax4.bar(etiquetas, vals, color=colores)
    for b, v in zip(bars, vals):
        ax4.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax4.set_title("Tamaño del Modelo (MB)", fontsize=11)
    ax4.set_ylabel("MB")
    ax4.tick_params(axis="x", labelsize=7)
    ax4.grid(axis="y", alpha=0.3)

    # Leyenda global
    from matplotlib.patches import Patch
    leyenda = [Patch(color=c, label=m) for m, c in colores_modelo.items()]
    fig.legend(handles=leyenda, loc="upper center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 1.01))

    fig.suptitle(
        "Comparativa de Métricas — HCE (SVM + Random Forest) vs Net5-Octree\n"
        "Resoluciones 32³ y 64³",
        fontsize=13, y=1.04,
    )

    for ext in ("png", "svg"):
        salida = DIR_RESULTADOS / f"comparativa_metricas_completa.{ext}"
        plt.savefig(salida, dpi=150 if ext == "png" else None,
                    format=ext, bbox_inches="tight")
        print(f"[Guardado] {salida.name}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# TABLA IMAGEN (para incluir directamente en la tesis)
# ──────────────────────────────────────────────────────────────

def graficar_tabla_imagen(filas: list):
    columnas = [
        "Modelo", "Res.", "T. Entren.", "T. Inf.\n(ms)",
        "Mem. Pico\n(MB)", "Tamaño\n(MB)", "Val Acc\n(%)", "Test Acc\n(%)", "Ép.\nÓptima"
    ]

    celdas = []
    for f in filas:
        celdas.append([
            f["Modelo"],
            f["Resolucion"],
            str(f["t_entreno"]),
            str(f["t_inf_ms"]),
            str(f["mem_pico_mb"]),
            str(f["tamano_mb"]),
            str(f["val_acc_pct"]),
            str(f["test_acc_pct"]),
            str(f["mejor_epoca"]),
        ])

    fig, ax = plt.subplots(figsize=(18, 0.55 * len(filas) + 1.5))
    ax.axis("off")

    tabla = ax.table(
        cellText=celdas, colLabels=columnas,
        loc="center", cellLoc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1.1, 1.9)

    colores_fila = {
        "SVM (HCE)":           "#E3F2FD",
        "Random Forest (HCE)": "#FFF3E0",
        "Net5-Octree":         "#E8F5E9",
    }

    for (row, col), cell in tabla.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif row <= len(filas):
            modelo = filas[row - 1]["Modelo"]
            cell.set_facecolor(colores_fila.get(modelo, "white"))

    ax.set_title(
        "Resumen de Métricas Experimentales — HCE vs Net5-Octree",
        fontsize=13, pad=20, fontweight="bold",
    )

    for ext in ("png", "svg"):
        salida = DIR_RESULTADOS / f"tabla_metricas_completa.{ext}"
        plt.savefig(salida, dpi=150 if ext == "png" else None,
                    format=ext, bbox_inches="tight")
        print(f"[Guardado] {salida.name}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("Cargando metricas de todos los modelos...")
    filas = cargar_todas_metricas()

    if not filas:
        print("[ERROR] No se encontraron archivos de resultados.")
        print(f"  Verifica que existan JSON en: {DIR_RESULTADOS}")
        return

    imprimir_tabla(filas)
    exportar_csv(filas)

    print("\nGenerando figuras...")
    graficar_comparativa(filas)
    graficar_tabla_imagen(filas)

    print(f"\nTodo guardado en: {DIR_RESULTADOS}")


if __name__ == "__main__":
    main()
