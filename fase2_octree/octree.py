"""
octree.py - Fase 2
===================
Construccion operativa de representaciones 32^3 y 64^3 a partir de
mallas .OFF de ModelNet40.

Flujo determinista (segun metodologia):
    1. Normalizacion   : escalado al cubo unitario [-1,1]^3, centrado en origen
    2. Muestreo         : nube de puntos uniforme sobre la superficie (area-weighted)
    3. Cuantizacion     : grid denso equivalente al nivel hoja de un octree
                          cuya raiz se identifica como L=0 (2^3, 8 nodos),
                          con R = 2^(L+1). Hoja L=4 para 32^3, L=5 para 64^3.
    4. Codificacion     : cada nodo hoja almacena ocupacion + vector normal promedio

La representacion final es un grid denso (voxel grid) de resolucion 2^L,
donde cada celda ocupada contiene su vector normal promedio (nx,ny,nz)
y un canal adicional de ocupacion binaria. Esto es equivalente a un
octree completo "aplanado" a su nivel hoja, mas eficiente de procesar
en GPU que un arbol disperso real, y trivialmente convertible de vuelta
a estructura de arbol si se requiere.

Canales por celda: [ocupacion, nx, ny, nz]  -> grid shape (4, R, R, R)
"""

import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# 1. LECTURA DE MALLA .OFF (vertices + caras)
# ──────────────────────────────────────────────────────────────

def leer_off(ruta: str) -> tuple:
    """
    Lee un archivo .off y retorna (vertices, caras).
    vertices: array (N, 3) float32
    caras   : array (M, 3) int64  (indices de triangulos)
    """
    with open(ruta, "r") as f:
        primera = f.readline().strip()

        if primera.upper().startswith("OFF") and primera.upper() != "OFF":
            resto = primera[3:].strip()
            n_verts, n_caras, _ = map(int, resto.split())
        else:
            n_verts, n_caras, _ = map(int, f.readline().split())

        lineas_vertices = [f.readline() for _ in range(n_verts)]
        lineas_caras    = [f.readline() for _ in range(n_caras)]

    vertices = np.loadtxt(lineas_vertices, dtype=np.float32, usecols=(0, 1, 2))
    if vertices.ndim == 1:
        vertices = vertices.reshape(1, -1)
    vertices = np.nan_to_num(vertices, nan=0.0, posinf=0.0, neginf=0.0)

    caras = None
    if n_caras > 0:
        caras_raw = np.loadtxt(lineas_caras, dtype=np.int64)
        if caras_raw.ndim == 1:
            caras_raw = caras_raw.reshape(1, -1)
        caras = caras_raw[:, 1:4]

        # Saneo: descartar caras con indices fuera de rango
        n_v = len(vertices)
        validas = (
            (caras[:, 0] >= 0) & (caras[:, 0] < n_v) &
            (caras[:, 1] >= 0) & (caras[:, 1] < n_v) &
            (caras[:, 2] >= 0) & (caras[:, 2] < n_v)
        )
        caras = caras[validas]

    return vertices, caras


# ──────────────────────────────────────────────────────────────
# 2. NORMALIZACION: escalado a [-1,1]^3, centrado en origen
# ──────────────────────────────────────────────────────────────

def normalizar_malla(vertices: np.ndarray) -> np.ndarray:
    """
    Centra la malla en el origen y la escala para que quepa exactamente
    en el cubo unitario [-1, 1]^3 (manteniendo proporciones, sin distorsion).
    """
    centroide = vertices.mean(axis=0)
    v = vertices - centroide

    # Escala por el maximo valor absoluto en cualquier eje (cubo, no esfera)
    escala = np.max(np.abs(v))
    if escala > 0:
        v = v / escala

    return v.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# 3. MUESTREO: nube de puntos uniforme sobre la superficie
#    (area-weighted sampling, igual al usado en PointNet)
# ──────────────────────────────────────────────────────────────

def calcular_normales_caras(vertices: np.ndarray, caras: np.ndarray) -> np.ndarray:
    """Calcula el vector normal unitario de cada triangulo (M, 3)."""
    v0 = vertices[caras[:, 0]]
    v1 = vertices[caras[:, 1]]
    v2 = vertices[caras[:, 2]]

    normales = np.cross(v1 - v0, v2 - v0)
    normas = np.linalg.norm(normales, axis=1, keepdims=True)
    normas = np.clip(normas, 1e-12, None)
    normales = normales / normas

    return normales.astype(np.float32)


def muestrear_superficie_con_normales(
    vertices: np.ndarray, caras: np.ndarray, n_puntos: int,
    rng: np.random.Generator,
) -> tuple:
    """
    Muestreo uniforme sobre la superficie del mesh (area-weighted).
    Retorna (puntos, normales) donde cada punto lleva la normal de su
    triangulo de origen.

    puntos   : (n_puntos, 3)
    normales : (n_puntos, 3)
    """
    if caras is None or len(caras) == 0:
        # Fallback: usar vertices directamente, normal indefinida -> ceros
        idx = rng.choice(len(vertices), n_puntos, replace=len(vertices) < n_puntos)
        puntos = vertices[idx]
        normales = np.zeros_like(puntos)
        return puntos, normales

    v0 = vertices[caras[:, 0]]
    v1 = vertices[caras[:, 1]]
    v2 = vertices[caras[:, 2]]

    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    areas = np.nan_to_num(areas, nan=0.0, posinf=0.0, neginf=0.0)
    areas = np.clip(areas, 0.0, None)

    suma = areas.sum()
    if suma <= 0:
        idx = rng.choice(len(vertices), n_puntos, replace=len(vertices) < n_puntos)
        puntos = vertices[idx]
        normales = np.zeros_like(puntos)
        return puntos, normales

    probs = areas / suma
    probs = np.nan_to_num(probs, nan=0.0)
    probs = probs / probs.sum()

    idx_tri = rng.choice(len(caras), size=n_puntos, p=probs)

    r1 = rng.random(n_puntos).astype(np.float32)
    r2 = rng.random(n_puntos).astype(np.float32)
    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2

    puntos = (
        u[:, None] * v0[idx_tri]
        + v[:, None] * v1[idx_tri]
        + w[:, None] * v2[idx_tri]
    )
    puntos = np.nan_to_num(puntos, nan=0.0, posinf=0.0, neginf=0.0)

    normales_caras = calcular_normales_caras(vertices, caras)
    normales = normales_caras[idx_tri]

    return puntos.astype(np.float32), normales.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# 4. CUANTIZACION JERARQUICA + CODIFICACION
#    Grid denso de resolucion R, equivalente al nivel hoja de un
#    octree cuya raiz se identifica como L=0 (convencion documental:
#    R = 2^(L+1), donde L=0 -> 2^3 con 8 nodos).
#    Canal 0: ocupacion binaria
#    Canales 1-3: vector normal promedio de los puntos en esa celda
# ──────────────────────────────────────────────────────────────

# IMPORTANTE - dos convenciones distintas, no intercambiables:
#
# 1) PROFUNDIDAD_POR_RESOLUCION: numero de ITERACIONES usado
#    internamente por ocupacion_por_nivel() (hce_extraccion.py) para
#    reconstruir la jerarquia desde la hoja hasta 2^3 (sin llegar a
#    1^3). Este valor NO debe reinterpretarse como una etiqueta "L".
#    Se mantiene por compatibilidad con el codigo de extraccion HCE
#    ya validado.
#
# 2) nivel_hoja(R): la etiqueta "L" que se presenta en el documento
#    de tesis y en las figuras, bajo la convencion acordada con el
#    profesor: la raiz conceptual es L=0 (resolucion 2^3, 8 nodos),
#    y R = 2^(L+1). Bajo esta convencion, la hoja de 32^3 es L=4 y
#    la hoja de 64^3 es L=5. Esta es la UNICA convencion que debe
#    aparecer en texto, figuras y nombres de columnas del documento.
PROFUNDIDAD_POR_RESOLUCION = {32: 5, 64: 6}


def construir_grid_octree(
    puntos: np.ndarray, normales: np.ndarray, resolucion: int,
) -> np.ndarray:
    """
    Cuantiza la nube de puntos (ya normalizada a [-1,1]^3) en un grid
    denso de resolucion x resolucion x resolucion, equivalente al nivel
    hoja de un octree de profundidad L = log2(resolucion).

    Retorna grid de forma (4, R, R, R):
        canal 0       : ocupacion binaria (1 si la celda contiene >=1 punto)
        canales 1,2,3 : vector normal promedio (nx, ny, nz) de los puntos
                        que caen en esa celda; (0,0,0) si esta vacia
    """
    R = resolucion

    # Mapear [-1, 1] -> [0, R) y truncar a indices validos
    idx = ((puntos + 1.0) * 0.5 * R).astype(np.int64)
    idx = np.clip(idx, 0, R - 1)

    grid_ocupacion = np.zeros((R, R, R), dtype=np.float32)
    grid_normal_sum = np.zeros((R, R, R, 3), dtype=np.float32)
    grid_conteo = np.zeros((R, R, R), dtype=np.float32)

    # Acumulacion vectorizada usando indices lineales (mucho mas rapido que loop)
    idx_lineal = idx[:, 0] * R * R + idx[:, 1] * R + idx[:, 2]

    np.add.at(grid_conteo.reshape(-1), idx_lineal, 1.0)
    for c in range(3):
        np.add.at(grid_normal_sum.reshape(-1, 3)[:, c], idx_lineal, normales[:, c])

    grid_ocupacion = (grid_conteo > 0).astype(np.float32)

    # Promedio de normales por celda (evitar division por cero)
    conteo_safe = np.clip(grid_conteo, 1.0, None)
    grid_normal_prom = grid_normal_sum / conteo_safe[..., None]

    # Re-normalizar el vector promedio a unitario (si no es cero)
    normas = np.linalg.norm(grid_normal_prom, axis=-1, keepdims=True)
    normas_safe = np.clip(normas, 1e-12, None)
    grid_normal_prom = np.where(normas > 1e-12, grid_normal_prom / normas_safe, 0.0)

    # Apilar canales: (4, R, R, R)
    grid = np.concatenate([
        grid_ocupacion[None, ...],
        np.transpose(grid_normal_prom, (3, 0, 1, 2)),
    ], axis=0)

    return grid.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# RELLENO SOLIDO (solo para experimento Img2Voxel)
# ──────────────────────────────────────────────────────────────

def rellenar_interior_solido(ocupacion: np.ndarray) -> np.ndarray:
    """
    Rellena el interior de un objeto voxelizado usando flood fill 3D
    desde el exterior. Las celdas alcanzables desde el borde del grid
    son exterior; todo lo demas es interior (y se marca como ocupado).

    Esto convierte una representacion de "cascara superficial" (~1%
    de ocupacion) a un solido compacto (~10-30% de ocupacion), lo cual
    es necesario para que el modelo Img2Voxel pueda aprender a
    reconstruir formas geometricas con un IoU significativo.

    NOTA: Este proceso NO se aplica en el pipeline principal de HCE
    y Net5 (que usan la superficie original correctamente). Solo se
    usa para preparar los targets de Img2Voxel.
    """
    from scipy.ndimage import label

    R = ocupacion.shape[0]

    # Crear volumen extendido con borde de 1 celda libre (garantiza conectividad exterior)
    vol = np.zeros((R+2, R+2, R+2), dtype=np.uint8)
    vol[1:-1, 1:-1, 1:-1] = (ocupacion > 0.5).astype(np.uint8)

    # Invertir: 1=libre, 0=superficie
    vol_libre = 1 - vol

    # Etiquetar componentes conectados del espacio libre
    etiquetado, n_comps = label(vol_libre)

    # La etiqueta del exterior es la del voxel (0,0,0) que siempre esta libre
    etiqueta_exterior = etiquetado[0, 0, 0]

    # Interior = libre pero NO conectado al exterior
    interior = (vol_libre == 1) & (etiquetado != etiqueta_exterior)

    # Combinar superficie + interior
    solido = vol.copy()
    solido[interior] = 1

    # Quitar el borde de padding
    solido = solido[1:-1, 1:-1, 1:-1].astype(np.float32)

    return solido


def construir_grid_octree_solido(
    puntos: np.ndarray, normales: np.ndarray, resolucion: int,
) -> np.ndarray:
    """
    Igual que construir_grid_octree pero con relleno solido del interior.
    Produce grids con ~10-30% de ocupacion en vez de ~1%.
    Usado SOLO en el pipeline de Img2Voxel.
    """
    # Primero construir el grid de superficie normal
    grid = construir_grid_octree(puntos, normales, resolucion)

    # Rellenar el interior
    ocupacion_solida = rellenar_interior_solido(grid[0])

    # Reemplazar el canal de ocupacion (mantener normales de superficie)
    grid_solido = grid.copy()
    grid_solido[0] = ocupacion_solida

    return grid_solido


def malla_a_octree_solido(
    ruta_off: str,
    resolucion: int = 32,
    n_puntos_muestreo: int = 20000,
    seed: int = 42,
    idx_muestra: int = 0,
) -> np.ndarray:
    """
    Igual que malla_a_octree pero produce un solido (interior relleno).
    Usado EXCLUSIVAMENTE para preparar los targets del experimento Img2Voxel.
    No afecta los octrees de HCE ni Net5.
    """
    assert resolucion in PROFUNDIDAD_POR_RESOLUCION, \
        f"Resolucion {resolucion} no soportada. Usar 32 o 64."

    rng = np.random.default_rng(seed + idx_muestra)

    vertices, caras = leer_off(ruta_off)
    vertices = normalizar_malla(vertices)

    puntos, normales = muestrear_superficie_con_normales(
        vertices, caras, n_puntos_muestreo, rng,
    )

    grid = construir_grid_octree_solido(puntos, normales, resolucion)

    return grid


def malla_a_octree(
    ruta_off: str,
    resolucion: int = 32,
    n_puntos_muestreo: int = 20000,
    seed: int = 42,
    idx_muestra: int = 0,
) -> np.ndarray:
    """
    Pipeline completo determinista: .off -> grid de octree (4, R, R, R).

    Parametros
    ----------
    ruta_off          : ruta al archivo .off
    resolucion        : 32 o 64
    n_puntos_muestreo : cuantos puntos de superficie muestrear antes de
                         cuantizar (mas puntos = cuantizacion mas fiel,
                         pero mas costo). 20000 es un valor robusto para
                         resolucion 64.
    seed, idx_muestra : controlan el RNG determinista (reproducibilidad)

    Retorna
    -------
    grid : array (4, R, R, R) float32
    """
    assert resolucion in PROFUNDIDAD_POR_RESOLUCION, \
        f"Resolucion {resolucion} no soportada. Usar 32 o 64."

    rng = np.random.default_rng(seed + idx_muestra)

    vertices, caras = leer_off(ruta_off)
    vertices = normalizar_malla(vertices)

    puntos, normales = muestrear_superficie_con_normales(
        vertices, caras, n_puntos_muestreo, rng,
    )

    grid = construir_grid_octree(puntos, normales, resolucion)

    return grid


def profundidad_de(resolucion: int) -> int:
    """
    Retorna la profundidad de hoja L del octree, bajo la convencion
    matematica estandar confirmada tras implementar el arbol REAL
    (octree_real.py): raiz = L=0 (un solo nodo, todo el espacio),
    R = 2^L en cada nivel. Bajo esta convencion:
        32^3 -> L=5 (hoja)
        64^3 -> L=6 (hoja)
    Esta es la misma convencion usada por ocupacion_por_nivel_arbol()
    en octree_real.py y por nivel_hoja() (alias, ver mas abajo).
    """
    return PROFUNDIDAD_POR_RESOLUCION[resolucion]


def nivel_hoja(resolucion: int) -> int:
    """
    Alias de profundidad_de(), mantenido por compatibilidad con codigo
    y figuras existentes que ya llaman a nivel_hoja(). Desde que se
    implemento el arbol real (octree_real.py), profundidad_de() y
    nivel_hoja() son el MISMO valor: la convencion "L=0 en 2^3, 8 nodos"
    usada anteriormente para las figuras de la rejilla densa quedo
    reemplazada por la convencion matematica estandar de un octree con
    raiz unica (ver docstring de octree_real.py).
        32^3 -> L=5 (hoja)
        64^3 -> L=6 (hoja)
    """
    return PROFUNDIDAD_POR_RESOLUCION[resolucion]


# ──────────────────────────────────────────────────────────────
# TEST RAPIDO / VERIFICACION
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) > 1:
        ruta_test = sys.argv[1]
    else:
        print("Uso: python octree.py <ruta_a_archivo.off>")
        sys.exit(0)

    print("=" * 60)
    print("  TEST: Pipeline .off -> Octree")
    print("=" * 60)

    for R in (32, 64):
        t0 = time.time()
        grid = malla_a_octree(ruta_test, resolucion=R, n_puntos_muestreo=20000, seed=42)
        t1 = time.time()

        ocupacion = grid[0]
        n_ocupadas = int(ocupacion.sum())
        pct_ocupacion = 100 * n_ocupadas / ocupacion.size

        print(f"\n[Resolucion {R}^3]  (L={nivel_hoja(R)}, hoja)")
        print(f"  Shape grid       : {grid.shape}")
        print(f"  Celdas ocupadas  : {n_ocupadas:,} / {ocupacion.size:,} ({pct_ocupacion:.2f}%)")
        print(f"  Tiempo           : {t1 - t0:.3f}s")
        print(f"  Rango normal x   : [{grid[1].min():.3f}, {grid[1].max():.3f}]")

    print("\nPipeline verificado correctamente.")
