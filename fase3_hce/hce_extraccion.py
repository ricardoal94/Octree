"""
hce_extraccion.py - Fase 3 (Enfoque clasico: HCE)
====================================================
Hand-Crafted Extraction: calcula descriptores manuales a partir de un
OCTREE REAL (arbol con nodo raiz, subdivision recursiva y poda de ramas
vacias -- ver octree_real.py), sin usar deep learning.

IMPORTANTE - migracion desde la version anterior: esta version opera
sobre la ESTRUCTURA DE ARBOL REAL (NodoOctree), no sobre una rejilla
densa (R,R,R). La version anterior calculaba ocupacion_por_nivel()
mediante max-pool sobre un array denso que reservaba memoria para las
R^3 celdas sin importar si estaban ocupadas; eso NO era un octree real
(ver discusion tecnica que motivo esta migracion). Los valores
numericos resultantes son equivalentes (mismo criterio de ocupacion),
pero ahora se derivan de un recorrido real del arbol con poda.

Convencion de profundidad (raiz = L=0, un solo nodo, R=2^L en cada
nivel, consistente con octree_real.py):
    32^3 -> L=5 (hoja)
    64^3 -> L=6 (hoja)

Descriptores extraidos (segun metodologia: "ocupacion de nodos por nivel
y momentos geometricos de la estructura del arbol"):

  A) Ocupacion jerarquica por nivel:
     Para cada profundidad d = 0 (raiz) .. L (hoja), se calcula el %
     de nodos que REALMENTE EXISTEN en el arbol (no podados) respecto
     al maximo posible en ese nivel (8^d). Ver
     octree_real.py::ocupacion_por_nivel_arbol().
     Produce L+1 valores escalares: 6 para R=32 (L=5), 7 para R=64 (L=6).

  B) Momentos geometricos globales (sobre los CENTROS de las hojas
     ocupadas del arbol, tratados como una nube de puntos discreta):
     - Centroide (3 valores: cx, cy, cz)
     - Varianza por eje (3 valores: vx, vy, vz)
     - Dispersion radial promedio (1 valor)
     - Asimetria (skewness) por eje (3 valores)

  C) Estadisticas del vector normal promedio (sobre las hojas ocupadas):
     - Norma promedio de las normales (coherencia de superficie)
     - Varianza de la norma (rugosidad/variabilidad)

Total de features: (L+1) [ocupacion por nivel, incluyendo la raiz] +
10 (momentos geometricos) + 2 (estadisticas de normales)
  Para R=32 (L=5, niveles L=0..L=5): 6 + 12 = 18 features
  Para R=64 (L=6, niveles L=0..L=6): 7 + 12 = 19 features
"""

import numpy as np
from pathlib import Path

from octree_real import (
    NodoOctree, recolectar_hojas, ocupacion_por_nivel_arbol,
)


# ──────────────────────────────────────────────────────────────
# A. OCUPACION JERARQUICA POR NIVEL (via arbol real)
# ──────────────────────────────────────────────────────────────

def ocupacion_por_nivel(raiz: NodoOctree, profundidad_max: int) -> np.ndarray:
    """
    Wrapper delgado sobre ocupacion_por_nivel_arbol() de octree_real.py,
    mantenido en este modulo para no romper el resto de la interfaz de
    extraccion HCE. Ver esa funcion para el detalle del calculo.

    Retorna array de tamaño (profundidad_max + 1): indice 0 = raiz,
    indice profundidad_max = hoja.
    """
    return ocupacion_por_nivel_arbol(raiz, profundidad_max)


# ──────────────────────────────────────────────────────────────
# B. MOMENTOS GEOMETRICOS GLOBALES (sobre centros de hojas del arbol)
# ──────────────────────────────────────────────────────────────

def momentos_geometricos(hojas: list) -> np.ndarray:
    """
    Calcula momentos geometricos sobre los CENTROS de las hojas
    ocupadas del octree real (en lugar de indices de una rejilla densa).

    Parametros
    ----------
    hojas : lista de NodoOctree (salida de recolectar_hojas())

    Retorna 10 valores:
        [cx, cy, cz, vx, vy, vz, dispersion_radial, skew_x, skew_y, skew_z]
    """
    if len(hojas) == 0:
        return np.zeros(10, dtype=np.float32)

    coords = np.array([h.centro for h in hojas], dtype=np.float32)  # (N, 3)

    centroide = coords.mean(axis=0)
    diff = coords - centroide
    varianza = diff.var(axis=0)

    dist_radial = np.linalg.norm(diff, axis=1)
    dispersion_radial = dist_radial.mean()

    std = np.sqrt(np.clip(varianza, 1e-12, None))
    skew = np.mean(diff ** 3, axis=0) / (std ** 3 + 1e-12)

    features = np.concatenate([
        centroide, varianza, [dispersion_radial], skew,
    ]).astype(np.float32)

    return features


# ──────────────────────────────────────────────────────────────
# C. ESTADISTICAS DEL VECTOR NORMAL (sobre hojas del arbol)
# ──────────────────────────────────────────────────────────────

def estadisticas_normales(hojas: list) -> np.ndarray:
    """
    A partir de la normal promedio almacenada en cada hoja ocupada,
    calcula:
      - norma promedio de las normales (coherencia de superficie)
      - varianza de la norma (rugosidad/variabilidad)

    Parametros
    ----------
    hojas : lista de NodoOctree (salida de recolectar_hojas())

    Retorna array de 2 valores.
    """
    if len(hojas) == 0:
        return np.zeros(2, dtype=np.float32)

    vecs = np.array([h.normal_promedio for h in hojas], dtype=np.float32)
    normas = np.linalg.norm(vecs, axis=1)

    return np.array([normas.mean(), normas.var()], dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# EXTRACTOR COMPLETO: arbol real -> vector de features
# ──────────────────────────────────────────────────────────────

def extraer_descriptores_hce(raiz: NodoOctree, profundidad_max: int) -> np.ndarray:
    """
    Pipeline completo de extraccion HCE para un octree REAL.

    Parametros
    ----------
    raiz            : NodoOctree raiz (salida de construir_octree())
    profundidad_max : L de la hoja (5 para R=32, 6 para R=64)

    Retorna
    -------
    vector de features 1D, tamaño = (profundidad_max + 1) + 10 + 2
    """
    hojas = recolectar_hojas(raiz)

    feats_nivel  = ocupacion_por_nivel(raiz, profundidad_max)   # L+1 valores
    feats_mom    = momentos_geometricos(hojas)                    # 10 valores
    feats_normal = estadisticas_normales(hojas)                   # 2 valores

    return np.concatenate([feats_nivel, feats_mom, feats_normal]).astype(np.float32)


def nombres_features(profundidad_max: int) -> list:
    """
    Retorna los nombres descriptivos de cada feature, en el mismo
    orden que produce extraer_descriptores_hce(). Util para interpretar
    feature_importances_ del Random Forest.

    Convencion: ocupacion_por_nivel() ahora recorre el arbol real desde
    la raiz (d=0) hasta la hoja (d=profundidad_max), en ese orden
    ascendente. Por lo tanto:
        indice 0                -> L = 0 (raiz)
        indice profundidad_max  -> L = profundidad_max (hoja)
    Para 32^3 (profundidad_max=5): L=0 (raiz) ... L=5 (hoja).
    Para 64^3 (profundidad_max=6): L=0 (raiz) ... L=6 (hoja).
    """
    nombres = [f"ocupacion_L{d}" for d in range(profundidad_max + 1)]
    nombres += ["centroide_x", "centroide_y", "centroide_z",
                "varianza_x", "varianza_y", "varianza_z",
                "dispersion_radial",
                "skew_x", "skew_y", "skew_z"]
    nombres += ["normal_norma_media", "normal_norma_varianza"]
    return nombres


# ──────────────────────────────────────────────────────────────
# TEST RAPIDO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "fase2_octree"))
    from octree_real import construir_octree

    print("=" * 60)
    print("  TEST: Extraccion de descriptores HCE (arbol real)")
    print("=" * 60)

    # Nube de puntos sintetica: esfera hueca (superficie), no solido
    rng = np.random.default_rng(42)
    n_pts = 5000
    theta = rng.uniform(0, np.pi, n_pts)
    phi = rng.uniform(0, 2 * np.pi, n_pts)
    radio = 0.7
    x = radio * np.sin(theta) * np.cos(phi)
    y = radio * np.sin(theta) * np.sin(phi)
    z = radio * np.cos(theta)
    puntos = np.stack([x, y, z], axis=1).astype(np.float32)
    normales = puntos / radio

    for R, L in [(32, 5), (64, 6)]:
        print(f"\n--- Resolucion {R}^3 (L={L}, hoja) ---")
        raiz = construir_octree(puntos, normales, profundidad_max=L)

        feats = extraer_descriptores_hce(raiz, L)
        nombres = nombres_features(L)

        print(f"  Total features: {len(feats)}  (esperado: {L+1+12})")
        assert len(feats) == L + 1 + 12
        assert len(nombres) == len(feats)

        for nombre, valor in zip(nombres, feats):
            print(f"    {nombre:25s}: {valor:.4f}")

    print("\n  Test completado correctamente (arbol real, sin rejilla densa).")
