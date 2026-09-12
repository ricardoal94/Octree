"""
hce_extraccion.py - Fase 3 (Enfoque clasico: HCE)
====================================================
Hand-Crafted Extraction: calcula descriptores manuales a partir de un
OCTREE REAL (arbol con nodo raiz, subdivision recursiva y poda de ramas
vacias -- ver octree_real.py), sin usar deep learning.

Dos formas de uso:
  1. extraer_descriptores_hce(raiz, profundidad_max) -- cuando se tiene
     el arbol NodoOctree recien construido (ej. pruebas, visualizacion).
  2. extraer_descriptores_hce_desde_npz(ruta_npz) -- cuando se carga
     el archivo disperso ya persistido por preprocesar_octrees.py
     (uso normal en el entrenamiento de SVM/Random Forest). Esta
     variante NO reconstruye el arbol completo: calcula la ocupacion
     por nivel directamente desde los centros de las hojas guardadas
     (ver octree_real.py::ocupacion_por_nivel_desde_hojas, verificada
     numericamente equivalente a recorrer el arbol).

Convencion de profundidad (raiz = L=0, un solo nodo, R=2^L en cada
nivel, consistente con octree_real.py):
    32^3 -> L=5 (hoja)
    64^3 -> L=6 (hoja)

Descriptores extraidos (segun metodologia: "ocupacion de nodos por nivel
y momentos geometricos de la estructura del arbol"):

  A) Ocupacion jerarquica por nivel: % de nodos que REALMENTE EXISTEN
     en el arbol (no podados) en cada profundidad d=0 (raiz) .. L (hoja),
     respecto al maximo posible en ese nivel (8^d). Produce L+1 valores.

  B) Momentos geometricos globales (sobre los CENTROS de las hojas
     ocupadas, tratados como una nube de puntos discreta):
     centroide (3), varianza por eje (3), dispersion radial (1),
     skewness por eje (3) = 10 valores.

  C) Estadisticas del vector normal promedio (sobre las hojas ocupadas):
     norma media (1) y varianza de la norma (1) = 2 valores.

Total de features: (L+1) + 10 + 2
  Para R=32 (L=5): 6 + 12 = 18 features
  Para R=64 (L=6): 7 + 12 = 19 features
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "fase2_octree"))
from octree_real import (
    NodoOctree, recolectar_hojas, ocupacion_por_nivel_arbol,
    ocupacion_por_nivel_desde_hojas, cargar_octree_disperso,
)


# ──────────────────────────────────────────────────────────────
# B. MOMENTOS GEOMETRICOS GLOBALES (sobre arrays crudos de centros)
# ──────────────────────────────────────────────────────────────

def momentos_geometricos(coords: np.ndarray) -> np.ndarray:
    """
    Calcula momentos geometricos sobre un array (N, 3) de coordenadas
    (centros de hojas ocupadas del octree real).

    Retorna 10 valores:
        [cx, cy, cz, vx, vy, vz, dispersion_radial, skew_x, skew_y, skew_z]
    """
    if len(coords) == 0:
        return np.zeros(10, dtype=np.float32)

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
# C. ESTADISTICAS DEL VECTOR NORMAL (sobre array crudo de normales)
# ──────────────────────────────────────────────────────────────

def estadisticas_normales(normales: np.ndarray) -> np.ndarray:
    """
    A partir de un array (N, 3) de normales promedio de cada hoja,
    calcula norma media y varianza de la norma.

    Retorna array de 2 valores.
    """
    if len(normales) == 0:
        return np.zeros(2, dtype=np.float32)

    normas = np.linalg.norm(normales, axis=1)
    return np.array([normas.mean(), normas.var()], dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# EXTRACTOR 1: desde un arbol NodoOctree recien construido
# ──────────────────────────────────────────────────────────────

def extraer_descriptores_hce(raiz: NodoOctree, profundidad_max: int) -> np.ndarray:
    """
    Extraccion HCE a partir de un arbol NodoOctree ya construido en
    memoria (uso tipico: pruebas, visualizacion, scripts que acaban
    de llamar a construir_octree() y no pasan por disco).
    """
    hojas = recolectar_hojas(raiz)
    centros = np.array([h.centro for h in hojas], dtype=np.float32) if hojas else np.zeros((0,3), dtype=np.float32)
    normales = np.array([h.normal_promedio for h in hojas], dtype=np.float32) if hojas else np.zeros((0,3), dtype=np.float32)

    feats_nivel  = ocupacion_por_nivel_arbol(raiz, profundidad_max)
    feats_mom    = momentos_geometricos(centros)
    feats_normal = estadisticas_normales(normales)

    return np.concatenate([feats_nivel, feats_mom, feats_normal]).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# EXTRACTOR 2: directamente desde el .npz disperso persistido
# (uso normal en fase3_hce_entrenamiento.py -- NO reconstruye el arbol)
# ──────────────────────────────────────────────────────────────

def extraer_descriptores_hce_desde_npz(ruta_npz: str) -> np.ndarray:
    """
    Extraccion HCE directamente desde el archivo disperso guardado por
    preprocesar_octrees.py, SIN reconstruir el arbol completo. La
    ocupacion por nivel se calcula desde los centros de hoja (ver
    octree_real.py::ocupacion_por_nivel_desde_hojas, verificada
    numericamente equivalente a recorrer el arbol real).

    Este es el metodo usado en produccion por fase3_hce_entrenamiento.py.
    """
    d = cargar_octree_disperso(ruta_npz)
    centros = d["centros_hoja"]
    normales = d["normales_hoja"]
    profundidad_max = d["profundidad_max"]

    feats_nivel  = ocupacion_por_nivel_desde_hojas(centros, profundidad_max)
    feats_mom    = momentos_geometricos(centros)
    feats_normal = estadisticas_normales(normales)

    return np.concatenate([feats_nivel, feats_mom, feats_normal]).astype(np.float32)


def nombres_features(profundidad_max: int) -> list:
    """
    Nombres descriptivos de cada feature, mismo orden que producen
    extraer_descriptores_hce() / extraer_descriptores_hce_desde_npz().

    indice 0 -> L=0 (raiz) ... indice profundidad_max -> L=profundidad_max (hoja).
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
    from octree_real import construir_octree, guardar_octree_disperso

    print("=" * 60)
    print("  TEST: extraccion HCE -- arbol en memoria vs .npz disperso")
    print("=" * 60)

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
        print(f"\n--- Resolucion {R}^3 (L={L}) ---")
        raiz = construir_octree(puntos, normales, profundidad_max=L)

        feats_memoria = extraer_descriptores_hce(raiz, L)

        ruta_tmp = f"/tmp/test_hce_R{R}.npz"
        guardar_octree_disperso(raiz, ruta_tmp, etiqueta=0, profundidad_max=L)
        feats_disco = extraer_descriptores_hce_desde_npz(ruta_tmp)

        print(f"  Features (memoria) == Features (disco): "
              f"{np.allclose(feats_memoria, feats_disco)}")
        assert np.allclose(feats_memoria, feats_disco), "Inconsistencia memoria vs disco"
        assert len(feats_memoria) == L + 1 + 12

    print("\n  Test completado: ambos extractores dan resultados identicos.")
