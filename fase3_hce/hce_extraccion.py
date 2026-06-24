"""
hce_extraccion.py - Fase 3 (Enfoque clasico: HCE)
====================================================
Hand-Crafted Extraction: calcula descriptores manuales a partir de los
grids de octree precomputados (Fase 2), sin usar deep learning.

Descriptores extraidos (segun metodologia: "ocupacion de nodos por nivel
y momentos geometricos de la estructura del arbol"):

  A) Ocupacion jerarquica por nivel:
     El grid denso de resolucion R (nivel hoja, profundidad L) se puede
     reconstruir hacia niveles superiores del octree agregando bloques de
     2x2x2 (max-pool de ocupacion). Para cada nivel l = 1..L se calcula
     el porcentaje de nodos ocupados respecto al total de ese nivel.
     Esto produce L valores escalares (5 para resolucion 32, 6 para 64).

  B) Momentos geometricos globales (sobre las celdas ocupadas del nivel
     hoja, tratando sus coordenadas como una nube de puntos discreta):
     - Centroide (3 valores: cx, cy, cz)
     - Varianza por eje (3 valores: vx, vy, vz)
     - Momento de inercia / dispersion radial promedio (1 valor)
     - Asimetria (skewness) por eje (3 valores)

  C) Estadisticas del vector normal promedio:
     - Norma promedio de los vectores normales en celdas ocupadas
       (mide cuan "plana"/coherente es la superficie capturada)
     - Varianza de las normales (mide rugosidad/variabilidad)

Total de features: L (ocupacion por nivel) + 10 (momentos geometricos)
+ 2 (estadisticas de normales) = L + 12
  Para R=32 (L=5): 17 features
  Para R=64 (L=6): 18 features
"""

import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# A. OCUPACION JERARQUICA POR NIVEL
# ──────────────────────────────────────────────────────────────

def ocupacion_por_nivel(ocupacion: np.ndarray, profundidad: int) -> np.ndarray:
    """
    Reconstruye la ocupacion en cada nivel del octree (desde la raiz
    hasta el nivel hoja) agregando bloques 2x2x2 sucesivamente, y calcula
    el porcentaje de nodos ocupados en cada nivel.

    Un nodo padre esta "ocupado" si al menos uno de sus 8 hijos lo esta
    (definicion estandar de octree: un nodo interno existe si contiene
    alguna superficie en su subarbol).

    Parametros
    ----------
    ocupacion   : grid binario (R, R, R), nivel hoja del octree
    profundidad : L, numero de niveles (5 para R=32, 6 para R=64)

    Retorna
    -------
    array de tamaño `profundidad` con el % de ocupacion en cada nivel,
    ordenado de la hoja (nivel L) hacia la raiz (nivel 1)
    """
    porcentajes = []
    nivel_actual = ocupacion.copy()

    for _ in range(profundidad):
        pct = 100.0 * nivel_actual.sum() / nivel_actual.size
        porcentajes.append(pct)

        R_actual = nivel_actual.shape[0]
        if R_actual == 1:
            break

        # Agregar bloques 2x2x2: un bloque esta ocupado si algun hijo lo esta
        R_mitad = R_actual // 2
        bloque = nivel_actual.reshape(R_mitad, 2, R_mitad, 2, R_mitad, 2)
        nivel_actual = (bloque.max(axis=(1, 3, 5)) > 0).astype(np.float32)

    return np.array(porcentajes, dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# B. MOMENTOS GEOMETRICOS GLOBALES
# ──────────────────────────────────────────────────────────────

def momentos_geometricos(ocupacion: np.ndarray) -> np.ndarray:
    """
    Calcula momentos geometricos sobre las coordenadas de las celdas
    ocupadas (tratadas como una nube de puntos discreta en [-1,1]^3).

    Retorna 10 valores:
        [cx, cy, cz, vx, vy, vz, dispersion_radial, skew_x, skew_y, skew_z]
    """
    R = ocupacion.shape[0]
    idx = np.argwhere(ocupacion > 0)   # (N, 3) indices enteros

    if len(idx) == 0:
        return np.zeros(10, dtype=np.float32)

    # Convertir indices a coordenadas continuas en [-1, 1]
    coords = (idx + 0.5) / R * 2 - 1   # (N, 3)

    centroide = coords.mean(axis=0)               # (3,)
    diff = coords - centroide
    varianza = diff.var(axis=0)                     # (3,)

    # Dispersion radial promedio (momento de inercia simplificado)
    dist_radial = np.linalg.norm(diff, axis=1)
    dispersion_radial = dist_radial.mean()

    # Asimetria (skewness) por eje: E[(x-mu)^3] / std^3
    std = np.sqrt(np.clip(varianza, 1e-12, None))
    skew = np.mean(diff ** 3, axis=0) / (std ** 3 + 1e-12)

    features = np.concatenate([
        centroide, varianza, [dispersion_radial], skew,
    ]).astype(np.float32)

    return features


# ──────────────────────────────────────────────────────────────
# C. ESTADISTICAS DEL VECTOR NORMAL
# ──────────────────────────────────────────────────────────────

def estadisticas_normales(grid: np.ndarray) -> np.ndarray:
    """
    A partir de los canales 1:4 del grid (nx, ny, nz) en celdas ocupadas,
    calcula:
      - norma promedio de las normales (coherencia de superficie)
      - varianza de la norma (rugosidad/variabilidad)

    Retorna array de 2 valores.
    """
    ocupacion = grid[0]
    normales  = grid[1:4]   # (3, R, R, R)

    idx = np.argwhere(ocupacion > 0)
    if len(idx) == 0:
        return np.zeros(2, dtype=np.float32)

    vecs = normales[:, idx[:, 0], idx[:, 1], idx[:, 2]].T   # (N, 3)
    normas = np.linalg.norm(vecs, axis=1)

    return np.array([normas.mean(), normas.var()], dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# EXTRACTOR COMPLETO: grid -> vector de features
# ──────────────────────────────────────────────────────────────

def extraer_descriptores_hce(grid: np.ndarray, profundidad: int) -> np.ndarray:
    """
    Pipeline completo de extraccion HCE para un grid de octree.

    Parametros
    ----------
    grid        : array (4, R, R, R) -- salida de octree.py
    profundidad : L (5 para R=32, 6 para R=64)

    Retorna
    -------
    vector de features 1D, tamaño = profundidad + 10 + 2
    """
    ocupacion = grid[0]

    feats_nivel  = ocupacion_por_nivel(ocupacion, profundidad)     # L valores
    feats_mom    = momentos_geometricos(ocupacion)                  # 10 valores
    feats_normal = estadisticas_normales(grid)                      # 2 valores

    return np.concatenate([feats_nivel, feats_mom, feats_normal]).astype(np.float32)


def nombres_features(profundidad: int) -> list:
    """Retorna los nombres descriptivos de cada feature, en el mismo
    orden que produce extraer_descriptores_hce(). Util para interpretar
    feature_importances_ del Random Forest."""
    nombres = [f"ocupacion_nivel_{i+1}" for i in range(profundidad)]
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
    print("=" * 60)
    print("  TEST: Extraccion de descriptores HCE")
    print("=" * 60)

    # Grid sintetico: esfera ocupada en el centro (simula un objeto solido)
    R = 32
    L = 5
    grid = np.zeros((4, R, R, R), dtype=np.float32)

    centro = R // 2
    radio = R // 4
    for x in range(R):
        for y in range(R):
            for z in range(R):
                if (x-centro)**2 + (y-centro)**2 + (z-centro)**2 <= radio**2:
                    grid[0, x, y, z] = 1.0
                    # Normal sintetica apuntando hacia afuera del centro
                    v = np.array([x-centro, y-centro, z-centro], dtype=np.float32)
                    norma = np.linalg.norm(v)
                    if norma > 0:
                        grid[1:4, x, y, z] = v / norma

    feats = extraer_descriptores_hce(grid, L)
    nombres = nombres_features(L)

    print(f"\n  Total features: {len(feats)}")
    for nombre, valor in zip(nombres, feats):
        print(f"    {nombre:25s}: {valor:.4f}")

    assert len(feats) == L + 12, "Numero de features inesperado"
    print("\n  Test completado correctamente.")
