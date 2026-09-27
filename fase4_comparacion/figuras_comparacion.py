"""Genera las figuras de la comparacion a partir de resultados/.

Uso (desde la raiz del repositorio):
    python fase4_comparacion/figuras_comparacion.py [--carpeta RUTA]
Requiere haber ejecutado antes scripts/evaluar.py (todas las etapas).
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CARPETA_DEFECTO = Path(__file__).resolve().parent.parent.parent / "Objetivos_4_y_5_comparacion"
RES = CARPETA_DEFECTO / "resultados"
FIG = CARPETA_DEFECTO / "figuras"


def configurar_carpeta() -> None:
    """Lee --carpeta y fija las rutas de entrada y salida."""
    import argparse
    global RES, FIG
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--carpeta", type=Path, default=CARPETA_DEFECTO,
                        help="Carpeta de la comparacion (con resultados/)")
    carpeta = parser.parse_args().carpeta.resolve()
    RES, FIG = carpeta / "resultados", carpeta / "figuras"

# Paleta validada (slots 1-2 categoricos) y tinta del grafico
COLOR_R = {32: "#2a78d6", 64: "#eb6834"}
SUPERFICIE, TINTA, TINTA2, APAGADO = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
REJILLA, EJE = "#e1e0d9", "#c3c2b7"
AZULES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
METODOS = [("svm", "SVM"), ("rf", "Bosque aleatorio"), ("net5", "Net5-Octree")]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10, "axes.facecolor": SUPERFICIE, "figure.facecolor": SUPERFICIE,
    "axes.edgecolor": EJE, "axes.labelcolor": TINTA2, "xtick.color": APAGADO,
    "ytick.color": TINTA2, "axes.grid": True, "grid.color": REJILLA,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "text.color": TINTA,
    "axes.titleweight": "semibold", "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none",  # texto editable en el SVG, no trazos
})


def cargar(nombre):
    return json.loads((RES / nombre).read_text(encoding="utf-8"))


def guardar(fig, nombre):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{nombre}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{nombre}.svg", bbox_inches="tight")
    plt.close(fig)


def leyenda_resolucion(ax, **kw):
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COLOR_R[R], label=f"R{R}")
                       for R in (32, 64)], frameon=False, **kw)


# 1. Aciertos: exactitud, exactitud balanceada y F1 macro (dumbbell R32 -> R64)
def fig_aciertos(an):
    met = an["metricas"]
    paneles = [("exactitud", "Exactitud"), ("exactitud_balanceada", "Exactitud balanceada"),
               ("f1_macro", "F1 macro")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for ax, (clave, titulo) in zip(axes, paneles):
        for fila, (m, nombre) in enumerate(METODOS):
            v32, v64 = met[f"{m}_R32"][clave], met[f"{m}_R64"][clave]
            ax.plot([v32 * 100, v64 * 100], [fila, fila], color=EJE, lw=2, zorder=1)
            for R, v in ((32, v32), (64, v64)):
                if clave == "exactitud":
                    lo, hi = met[f"{m}_R{R}"]["exactitud_ic95"]
                    ax.plot([lo * 100, hi * 100], [fila, fila], color=COLOR_R[R], lw=1,
                            alpha=0.55, zorder=2)
                ax.scatter(v * 100, fila, s=70, color=COLOR_R[R], edgecolor=SUPERFICIE,
                           linewidth=2, zorder=3)
            derecha = max(v32, v64) * 100
            if clave == "exactitud":
                derecha = max(met[f"{m}_R{R}"]["exactitud_ic95"][1] for R in (32, 64)) * 100
            ax.text(derecha + 0.6, fila, f"{v32*100:.1f} → {v64*100:.1f}", va="center",
                    fontsize=8.5, color=TINTA2)
        ax.set_title(titulo, loc="left")
        ax.set_xlabel("%")
        ax.set_yticks(range(3), [n for _, n in METODOS])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(left=min(ax.get_xlim()[0], 60), right=ax.get_xlim()[1] + 5)
    leyenda_resolucion(axes[0], loc="lower left", fontsize=9)
    fig.suptitle("Aciertos en test (2468 objetos). Líneas finas en Exactitud: IC 95 % bootstrap",
                 x=0.01, ha="left", fontsize=10, color=TINTA2)
    guardar(fig, "fig1_aciertos")


# 2. Tiempo por objeto desglosado (medias, un proceso, batch 1)
def fig_tiempos(ti):
    comp = [("Carga del octree", "#2a78d6"), ("Descriptores / grid-octree y planes", "#eb6834"),
            ("Clasificador / red", "#1baf7a")]
    casos = [("SVM", "hce_carga_ms", "hce_descriptores_ms", "svm_clasificador_ms", "svm_total_ms"),
             ("Bosque aleatorio", "hce_carga_ms", "hce_descriptores_ms", "rf_clasificador_ms", "rf_total_ms"),
             ("Net5 (GPU)", "net5_carga_ms", "net5_preparacion_ms", "net5_forward_gpu_ms", "net5_gpu_total_ms"),
             ("Net5 (CPU, 1 hilo)", "net5_carga_ms", "net5_preparacion_ms", "net5_forward_cpu_ms", "net5_cpu_total_ms")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, R in zip(axes, (32, 64)):
        t = ti["resoluciones"][str(R)]["tiempos_ms"]
        for fila, (nombre, *claves, total) in enumerate(casos):
            izq = 0
            for (etq, color), clave in zip(comp, claves):
                v = t[clave]["media"]
                ax.barh(fila, v, left=izq, color=color, height=0.55, edgecolor=SUPERFICIE,
                        linewidth=1.5, label=etq if fila == 0 else None)
                izq += v
            ax.text(izq + ax.get_xlim()[1] * 0.01, fila,
                    f"{izq:.0f} ms (mediana {t[total]['mediana']:.0f})",
                    va="center", fontsize=8.5, color=TINTA2)
        ax.set_title(f"R{R}", loc="left")
        ax.set_xlabel("ms por objeto (media)")
        ax.set_yticks(range(len(casos)), [c[0] for c in casos])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.35)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.0, -0.42), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Latencia de inferencia por objeto desde el archivo del octree "
                 "(un proceso, sin workers, batch 1)", x=0.01, ha="left", fontsize=10, color=TINTA2)
    guardar(fig, "fig2_tiempos")


# 3. Recursos: RAM de inferencia, VRAM y tamano en disco
def fig_recursos(me, an):
    casos = me["casos"]
    filas = [("SVM", "svm"), ("Bosque aleatorio", "rf"), ("Net5 (GPU)", "net5_gpu"),
             ("Net5 (CPU)", "net5_cpu")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8),
                             gridspec_kw={"width_ratios": [1.7, 1, 1]})
    alto = 0.36

    def barras(ax, valores, titulo, xlabel, etiquetas):
        for i, _ in enumerate(etiquetas):
            for j, R in enumerate((32, 64)):
                v = valores[i][j]
                if v is None:
                    continue
                y = i + (j - 0.5) * alto
                ax.barh(y, v, height=alto - 0.04, color=COLOR_R[R])
                ax.text(v, y, f" {v:,.1f}", va="center", fontsize=8, color=TINTA2)
        ax.set_yticks(range(len(etiquetas)), etiquetas)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(titulo, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.25)

    # RAM: base del proceso (gris) + RAM adicional (color); la barra completa es la RAM total
    ax = axes[0]
    for i, (_, c) in enumerate(filas):
        for j, R in enumerate((32, 64)):
            caso = casos[f"{c}_R{R}"]
            total = caso["ram_pico_proceso_mb"]
            adicional = caso["ram_pico_incremento_mb"]
            base = total - adicional
            y = i + (j - 0.5) * alto
            ax.barh(y, base, height=alto - 0.04, color="#e1e0d9",
                    label="Base del proceso (intérprete y bibliotecas)" if i == j == 0 else None)
            ax.barh(y, adicional, left=base, height=alto - 0.04, color=COLOR_R[R],
                    edgecolor=SUPERFICIE, linewidth=1)
            ax.text(total, y, f" +{adicional:,.0f} → {total:,.0f}", va="center",
                    fontsize=8, color=TINTA2)
    ax.set_yticks(range(len(filas)), [f for f, _ in filas])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title("RAM en inferencia: adicional (color) y total (barra completa)", loc="left")
    ax.set_xlabel("MiB (pico del working set)")
    ax.set_xlim(0, ax.get_xlim()[1] * 1.28)
    barras(axes[1], [[casos[f"net5_gpu_R{R}"]["vram_pico_asignada_mb"] for R in (32, 64)]],
           "VRAM pico (Net5 en GPU)", "MiB asignados por PyTorch", ["Net5 (GPU)"])
    mod = an["modelos"]
    barras(axes[2], [[mod[str(R)][m]["archivo_mb"] for R in (32, 64)] for m, _ in METODOS],
           "Tamaño del modelo en disco", "MB", [n for _, n in METODOS])
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color="#e1e0d9", label="Base del proceso"),
                            Line2D([], [], marker="s", ls="", color=COLOR_R[32], label="Adicional R32"),
                            Line2D([], [], marker="s", ls="", color=COLOR_R[64], label="Adicional R64")],
                   frameon=False, fontsize=8.5, loc="upper center",
                   bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    guardar(fig, "fig3_recursos")


# 4. Compromiso exactitud vs latencia
def fig_compromiso(an, ti):
    casos = [("SVM", "svm", "svm_total_ms", "o"), ("Bosque aleatorio", "rf", "rf_total_ms", "s"),
             ("Net5 GPU", "net5", "net5_gpu_total_ms", "^"),
             ("Net5 CPU", "net5", "net5_cpu_total_ms", "D")]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for nombre, m, clave, marca in casos:
        pts = []
        for R in (32, 64):
            x = ti["resoluciones"][str(R)]["tiempos_ms"][clave]["mediana"]
            y = an["metricas"][f"{m}_R{R}"]["exactitud"] * 100
            pts.append((x, y))
            ax.scatter(x, y, marker=marca, s=90, color=COLOR_R[R], edgecolor=SUPERFICIE,
                       linewidth=2, zorder=3)
        ax.annotate("", xy=pts[1], xytext=pts[0],
                    arrowprops=dict(arrowstyle="->", color=EJE, lw=1.2), zorder=1)
        dy = {"SVM": -0.35, "Bosque aleatorio": 0.3}.get(nombre, 0.0)
        ax.text(pts[1][0] * 1.08, pts[1][1] + dy, nombre, va="center", fontsize=9, color=TINTA)
    ax.set_xscale("log")
    ax.set_xlabel("Latencia por objeto, mediana (ms, escala log)")
    ax.set_ylabel("Exactitud en test (%)")
    ax.set_title("Compromiso exactitud–tiempo (flecha: R32 → R64)", loc="left")
    leyenda_resolucion(ax, loc="lower right", fontsize=9)
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 2.2)
    guardar(fig, "fig4_compromiso")


# 5. Recall por clase (40 clases x 6 modelos)
def fig_por_clase():
    with (RES / "recall_por_clase.csv").open(encoding="utf-8") as f:
        filas = list(csv.reader(f))
    cab, datos = filas[0][1:], filas[1:]
    orden = ["svm_R32", "svm_R64", "rf_R32", "rf_R64", "net5_R32", "net5_R64"]
    idx = [cab.index(c) for c in orden]
    clases = [d[0] for d in datos]
    M = np.asarray([[float(d[1 + i]) for i in idx] for d in datos])
    orden_f = np.argsort(-M.mean(axis=1))
    M, clases = M[orden_f], [clases[i] for i in orden_f]
    cmap = LinearSegmentedColormap.from_list("azul", ["#f0efec"] + AZULES)
    fig, ax = plt.subplots(figsize=(6.4, 11))
    im = ax.imshow(M * 100, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j] * 100
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if v > 65 else TINTA)
    etiquetas = ["SVM\nR32", "SVM\nR64", "Bosque\nR32", "Bosque\nR64", "Net5\nR32", "Net5\nR64"]
    ax.set_xticks(range(6), etiquetas, fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(clases)), clases, fontsize=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Recall por clase (%)")
    cb.outline.set_visible(False)
    ax.set_title("Recall por clase en test (ordenado por la media)", loc="left", pad=34)
    guardar(fig, "fig5_recall_por_clase")


# 6. Acuerdo entre metodos
def fig_acuerdo(an):
    cats = [("los_3_aciertan", "Los 3 aciertan", "#184f95"), ("2_aciertan", "2 aciertan", "#3987e5"),
            ("1_acierta", "1 acierta", "#86b6ef"), ("ninguno_acierta", "Ninguno acierta", "#c3c2b7")]
    fig, ax = plt.subplots(figsize=(10, 2.2))
    for fila, R in enumerate((32, 64)):
        a = an["acuerdo_entre_metodos"][str(R)]
        izq = 0
        for clave, etq, color in cats:
            v = a[clave]
            ax.barh(fila, v, left=izq, color=color, height=0.55, edgecolor=SUPERFICIE,
                    linewidth=2, label=etq if fila == 0 else None)
            if v > 60:
                ax.text(izq + v / 2, fila, f"{v}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if color in ("#184f95", "#3987e5") else TINTA)
            izq += v
    ax.set_yticks([0, 1], ["R32", "R64"])
    ax.invert_yaxis()
    ax.set_xlim(0, 2468)
    ax.set_xlabel("Objetos de test")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.75), frameon=False, fontsize=9)
    ax.set_title("¿Cuántos de los tres métodos clasifican bien cada objeto?", loc="left")
    guardar(fig, "fig6_acuerdo")


# 7. Curvas de entrenamiento de Net5
def fig_entrenamiento():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), sharey=True)
    for ax, (col, titulo) in zip(axes, (("train_acc", "Exactitud de entrenamiento"),
                                        ("val_acc", "Exactitud de validación"))):
        for R in (32, 64):
            with (RES / "oficiales" / f"net5_octree_historial_R{R}.csv").open(encoding="utf-8") as f:
                filas = list(csv.DictReader(f))
            ep = [int(r["epoca"]) for r in filas]
            v = [float(r[col]) * 100 for r in filas]
            ax.plot(ep, v, color=COLOR_R[R], lw=2, label=f"R{R}")
            if col == "val_acc":
                mejor = int(np.argmax(v))
                ax.scatter(ep[mejor], v[mejor], s=60, color=COLOR_R[R], edgecolor=SUPERFICIE,
                           linewidth=2, zorder=3)
                ax.annotate(f"mejor: época {ep[mejor]}", (ep[mejor], v[mejor]),
                            xytext=(0, 10 if R == 64 else -16), textcoords="offset points",
                            ha="center", fontsize=8, color=TINTA2)
        ax.set_title(titulo, loc="left")
        ax.set_xlabel("Época")
        ax.set_ylabel("%")
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    guardar(fig, "fig7_entrenamiento_net5")


def main():
    configurar_carpeta()
    an, ti, me = cargar("analisis.json"), cargar("tiempos.json"), cargar("memoria.json")
    fig_aciertos(an)
    fig_tiempos(ti)
    fig_recursos(me, an)
    fig_compromiso(an, ti)
    fig_por_clase()
    fig_acuerdo(an)
    fig_entrenamiento()
    print("Figuras en", FIG)


if __name__ == "__main__":
    main()
