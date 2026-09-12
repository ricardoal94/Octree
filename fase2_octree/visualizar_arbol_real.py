"""
Visualiza el octree REAL como cajas de subdivision (wireframe), a
diferencia de las figuras anteriores que mostraban una rejilla densa
uniforme. Aqui cada caja dibujada es un nodo que REALMENTE EXISTE en
el arbol (no fue podado), y su tamano varia segun la profundidad --
la caracteristica visual distintiva de un octree adaptativo real.
"""
import sys
sys.path.insert(0, '/mnt/user-data/outputs/fase2_octree')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from octree import leer_off, normalizar_malla, muestrear_superficie_con_normales
from octree_real import construir_octree, NodoOctree


def obtener_nodos_en_profundidad(raiz: NodoOctree, profundidad_objetivo: int) -> list:
    """Recolecta (centro, tamano) de todos los nodos EXISTENTES (no
    podados) en una profundidad especifica del arbol."""
    nodos = []

    def _rec(nodo):
        if nodo is None:
            return
        if nodo.profundidad == profundidad_objetivo:
            nodos.append((nodo.centro, nodo.tamano))
            return  # no seguir bajando mas alla de la profundidad pedida
        if not nodo.es_hoja:
            for hijo in nodo.hijos:
                _rec(hijo)
        # Si es hoja pero a menor profundidad que la pedida, no hay
        # nodos mas profundos en esta rama (ya no se subdividio mas).

    _rec(raiz)
    return nodos


def dibujar_caja_wireframe(ax, centro, tamano, color="steelblue", alpha=0.6, lw=0.6):
    """Dibuja las 12 aristas de un cubo centrado en `centro` con lado `tamano`."""
    h = tamano / 2.0
    # 8 vertices del cubo
    signos = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
              (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
    verts = [centro + h*np.array(s) for s in signos]

    # 12 aristas (pares de indices de vertices)
    aristas_idx = [
        (0,1),(1,2),(2,3),(3,0),   # cara inferior
        (4,5),(5,6),(6,7),(7,4),   # cara superior
        (0,4),(1,5),(2,6),(3,7),   # verticales
    ]
    segmentos = [[verts[i], verts[j]] for i, j in aristas_idx]

    coleccion = Line3DCollection(segmentos, colors=color, linewidths=lw, alpha=alpha)
    ax.add_collection3d(coleccion)


def graficar_niveles_arbol_real(raiz, profundidad_max, R_hoja, salida_base):
    """Genera una figura multi-panel mostrando las cajas REALES del
    arbol en cada nivel, ilustrando la poda (menos cajas, de tamano
    variable, concentradas donde hay geometria)."""

    niveles_a_mostrar = list(range(1, profundidad_max + 1))  # omitir L=0 (trivial, 1 caja)
    n_paneles = len(niveles_a_mostrar)

    fig = plt.figure(figsize=(4 * n_paneles, 4.5))

    for col, d in enumerate(niveles_a_mostrar):
        ax = fig.add_subplot(1, n_paneles, col + 1, projection="3d")
        nodos = obtener_nodos_en_profundidad(raiz, d)

        for centro, tamano in nodos:
            dibujar_caja_wireframe(ax, centro, tamano)

        max_posible = 8 ** d
        pct = 100 * len(nodos) / max_posible

        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f"$L={d}$\n{len(nodos)} / {max_posible} nodos ({pct:.1f}%)",
                    fontsize=10)
        ax.view_init(elev=22, azim=-55)
        ax.set_box_aspect([1, 1, 1])

    fig.text(0.5, -0.02,
             f"Figura: Octree REAL — cajas de subdivision existentes por nivel "
             f"(sin poda no se muestran; hoja en $L={profundidad_max}$, "
             f"resolucion equivalente ${R_hoja}^3$)",
             ha="center", va="top", fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{salida_base}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{salida_base}.svg", format="svg", bbox_inches="tight")
    print(f"[Guardado] {salida_base}.png / .svg")
    plt.close()


# ── Generar para chair_0001.off ──
verts, caras = leer_off("/mnt/user-data/uploads/chair_0001.off")
verts = normalizar_malla(verts)
rng = np.random.default_rng(42)
pts, norms = muestrear_superficie_con_normales(verts, caras, 20000, rng)

for R, L in [(32, 5), (64, 6)]:
    raiz = construir_octree(pts, norms, profundidad_max=L)
    graficar_niveles_arbol_real(raiz, L, R, f"/home/claude/svg_final/octree_real_niveles_R{R}")

print("\nListo.")
