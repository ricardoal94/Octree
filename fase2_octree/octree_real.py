"""
octree_real.py
================
Implementacion de un octree REAL: arbol con nodo raiz unico, subdivision
recursiva en 8 octantes, y poda explicita de ramas vacias (no se crea
ningun nodo para regiones sin geometria).

Esto reemplaza la aproximacion anterior (construir_grid_octree en
octree.py), que en realidad era una rejilla densa multicanal, NO un
octree: reservaba memoria para las R^3 celdas sin importar si estaban
ocupadas o no.

Convencion de profundidad (matematicamente estandar para octrees):
    - Raiz: profundidad d=0, UN SOLO NODO, cubre todo el espacio [-1,1]^3.
    - Cada subdivision reparte el nodo en 8 octantes (hijos).
    - A profundidad d, la resolucion equivalente (si todo estuviera
      subdividido) es R = 2^d.
    - Hoja de 32^3  -> d=5.
    - Hoja de 64^3  -> d=6.

IMPORTANTE: esta convencion (root=d=0 como UN nodo, R=2^d) es la
definicion matematica estandar de un octree y es la que se usa para
la LOGICA DE CONSTRUCCION del arbol. Es distinta de la convencion
"documental" L=0->2^3 acordada anteriormente para las FIGURAS de la
rejilla densa (que no era un arbol real). Con un arbol real construido
de verdad, se recomienda usar esta convencion matematica estandar
(root=1 nodo) en el documento, en lugar de la convencion anterior, ya
que ahora si existe una raiz unica que puede mostrarse como tal.
"""

import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# ESTRUCTURA DEL NODO
# ──────────────────────────────────────────────────────────────

class NodoOctree:
    """
    Nodo de un octree real.

    Atributos
    ---------
    centro       : np.ndarray (3,) -- centro del cubo que representa este nodo
    tamano       : float -- longitud de arista del cubo
    profundidad  : int -- 0 para la raiz, aumenta 1 por cada subdivision
    ocupado      : bool -- True si este nodo (o algun descendiente) contiene puntos
    es_hoja      : bool -- True si no tiene hijos (por profundidad maxima o por
                   no requerir mas subdivision)
    hijos        : list de 8 elementos, cada uno NodoOctree o None.
                   None significa rama PODADA (no se crea el nodo, no se
                   reserva memoria para esa region vacia).
    normal_promedio : np.ndarray (3,) o None -- solo definido en hojas ocupadas
    n_puntos     : int -- cantidad de puntos de la nube que caen en este nodo
                   (diagnostico / verificacion, no se persiste en disco)
    """

    __slots__ = ("centro", "tamano", "profundidad", "ocupado", "es_hoja",
                "hijos", "normal_promedio", "n_puntos")

    def __init__(self, centro: np.ndarray, tamano: float, profundidad: int):
        self.centro = centro
        self.tamano = tamano
        self.profundidad = profundidad
        self.ocupado = False
        self.es_hoja = True
        self.hijos = [None] * 8
        self.normal_promedio = None
        self.n_puntos = 0


# ──────────────────────────────────────────────────────────────
# CONSTRUCCION RECURSIVA CON PODA
# ──────────────────────────────────────────────────────────────

def _octante_de_puntos(puntos: np.ndarray, centro: np.ndarray) -> np.ndarray:
    """
    Determina a que octante (0-7) pertenece cada punto, segun su
    posicion relativa al centro del nodo actual en cada eje.
    Codificacion de bits: bit0=eje X, bit1=eje Y, bit2=eje Z.
    """
    bits_x = (puntos[:, 0] >= centro[0]).astype(np.int64)
    bits_y = (puntos[:, 1] >= centro[1]).astype(np.int64)
    bits_z = (puntos[:, 2] >= centro[2]).astype(np.int64)
    return bits_x + bits_y * 2 + bits_z * 4


def _centro_hijo(centro: np.ndarray, tamano: float, octante: int) -> np.ndarray:
    """Calcula el centro del octante hijo indicado (0-7)."""
    cuarto = tamano / 4.0
    dx = cuarto if (octante & 1) else -cuarto
    dy = cuarto if (octante & 2) else -cuarto
    dz = cuarto if (octante & 4) else -cuarto
    return centro + np.array([dx, dy, dz], dtype=np.float32)


def construir_octree(
    puntos: np.ndarray,
    normales: np.ndarray,
    profundidad_max: int,
    centro: np.ndarray = None,
    tamano: float = 2.0,
    profundidad: int = 0,
) -> NodoOctree:
    """
    Construye recursivamente un octree real a partir de una nube de
    puntos ya normalizada al cubo [-1,1]^3.

    PODA: si un octante no contiene ningun punto, su hijo correspondiente
    se deja como None -- NO se crea el nodo, NO se reserva memoria, y NO
    se continua la recursion en esa rama. Esta es la propiedad central
    que distingue a un octree real de una rejilla densa.

    Parametros
    ----------
    puntos          : (N, 3) nube de puntos normalizada
    normales        : (N, 3) normal de superficie de cada punto
    profundidad_max : profundidad de la hoja (5 para resolucion 32,
                      6 para resolucion 64; ver convencion en el docstring
                      del modulo)
    centro, tamano, profundidad : usados internamente por la recursion,
                      no deben pasarse en la llamada inicial

    Retorna
    -------
    NodoOctree raiz (profundidad=0)
    """
    if centro is None:
        centro = np.zeros(3, dtype=np.float32)

    nodo = NodoOctree(centro, tamano, profundidad)
    n = len(puntos)
    nodo.n_puntos = n

    if n == 0:
        nodo.ocupado = False
        nodo.es_hoja = True
        return nodo

    nodo.ocupado = True

    if profundidad >= profundidad_max:
        nodo.es_hoja = True
        normal_prom = normales.mean(axis=0)
        norma = np.linalg.norm(normal_prom)
        nodo.normal_promedio = (
            (normal_prom / norma).astype(np.float32)
            if norma > 1e-12 else np.zeros(3, dtype=np.float32)
        )
        return nodo

    nodo.es_hoja = False
    octantes = _octante_de_puntos(puntos, centro)

    for i in range(8):
        mask = octantes == i
        if not mask.any():
            nodo.hijos[i] = None   # PODA: no se crea nodo ni se recursa
            continue
        centro_hijo = _centro_hijo(centro, tamano, i)
        nodo.hijos[i] = construir_octree(
            puntos[mask], normales[mask], profundidad_max,
            centro=centro_hijo, tamano=tamano / 2.0, profundidad=profundidad + 1,
        )

    return nodo


# ──────────────────────────────────────────────────────────────
# RECORRIDO DEL ARBOL
# ──────────────────────────────────────────────────────────────

def recolectar_hojas(raiz: NodoOctree) -> list:
    """Retorna la lista de nodos hoja OCUPADOS del arbol (recorrido DFS)."""
    hojas = []

    def _rec(nodo):
        if nodo is None:
            return
        if nodo.es_hoja:
            if nodo.ocupado:
                hojas.append(nodo)
        else:
            for hijo in nodo.hijos:
                _rec(hijo)

    _rec(raiz)
    return hojas


def contar_nodos_por_profundidad(raiz: NodoOctree, profundidad_max: int) -> np.ndarray:
    """
    Cuenta cuantos nodos REALMENTE EXISTEN (no podados) en cada
    profundidad del arbol, de 0 (raiz) a profundidad_max (hojas).
    """
    conteo = np.zeros(profundidad_max + 1, dtype=np.int64)

    def _rec(nodo):
        if nodo is None:
            return
        conteo[nodo.profundidad] += 1
        if not nodo.es_hoja:
            for hijo in nodo.hijos:
                _rec(hijo)

    _rec(raiz)
    return conteo


def ocupacion_por_nivel_arbol(raiz: NodoOctree, profundidad_max: int) -> np.ndarray:
    """
    Calcula el % de ocupacion en cada nivel del arbol: nodos que
    REALMENTE EXISTEN (no podados) respecto al maximo posible en ese
    nivel (8^profundidad). Version basada en el arbol real -- reemplaza
    la funcion anterior que operaba via max-pool sobre una rejilla densa.

    Retorna un array de tamaño (profundidad_max + 1): indice 0 = raiz,
    indice profundidad_max = nivel hoja.
    """
    conteo = contar_nodos_por_profundidad(raiz, profundidad_max)
    porcentajes = np.zeros(profundidad_max + 1, dtype=np.float32)
    for d in range(profundidad_max + 1):
        total_posible = 8 ** d
        porcentajes[d] = 100.0 * conteo[d] / total_posible
    return porcentajes


def contar_nodos_totales(raiz: NodoOctree) -> int:
    """Cuenta el total de nodos reales existentes en el arbol."""
    total = [0]

    def _rec(nodo):
        if nodo is None:
            return
        total[0] += 1
        if not nodo.es_hoja:
            for hijo in nodo.hijos:
                _rec(hijo)

    _rec(raiz)
    return total[0]


# ──────────────────────────────────────────────────────────────
# MATERIALIZACION A GRID DENSO (solo para alimentar Net5-Octree,
# que por diseño de Conv3d/ConvTranspose3d requiere tensores densos)
# ──────────────────────────────────────────────────────────────

def octree_a_grid_denso(raiz: NodoOctree, resolucion: int) -> np.ndarray:
    """
    Convierte el arbol disperso a un grid denso (4, R, R, R), SOLO en
    el momento de alimentar la red neuronal. El almacenamiento en disco
    (ver guardar_octree_disperso) NUNCA usa este formato denso.
    """
    grid = np.zeros((4, resolucion, resolucion, resolucion), dtype=np.float32)
    hojas = recolectar_hojas(raiz)

    if len(hojas) == 0:
        return grid

    centros = np.array([h.centro for h in hojas], dtype=np.float32)
    normales = np.array([h.normal_promedio for h in hojas], dtype=np.float32)

    idx = np.clip(((centros + 1.0) * 0.5 * resolucion).astype(np.int64),
                 0, resolucion - 1)

    grid[0, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    grid[1, idx[:, 0], idx[:, 1], idx[:, 2]] = normales[:, 0]
    grid[2, idx[:, 0], idx[:, 1], idx[:, 2]] = normales[:, 1]
    grid[3, idx[:, 0], idx[:, 1], idx[:, 2]] = normales[:, 2]

    return grid


# ──────────────────────────────────────────────────────────────
# SERIALIZACION DISPERSA (ahorro de memoria real)
# ──────────────────────────────────────────────────────────────

def guardar_octree_disperso(raiz: NodoOctree, ruta_npz: str, etiqueta: int,
                            profundidad_max: int) -> None:
    """
    Guarda SOLO las hojas ocupadas del arbol: sus centros y normales.
    A diferencia del formato denso anterior (R^3 celdas siempre
    reservadas), este archivo pesa proporcionalmente al numero de
    hojas ocupadas, tipicamente 1-3% de R^3.
    """
    hojas = recolectar_hojas(raiz)
    n = len(hojas)

    centros = np.zeros((n, 3), dtype=np.float32)
    normales = np.zeros((n, 3), dtype=np.float32)
    for i, h in enumerate(hojas):
        centros[i] = h.centro
        normales[i] = h.normal_promedio

    np.savez_compressed(
        ruta_npz,
        centros_hoja=centros,
        normales_hoja=normales,
        etiqueta=etiqueta,
        profundidad_max=profundidad_max,
    )


def cargar_octree_disperso(ruta_npz: str) -> dict:
    """Carga el archivo disperso y retorna sus componentes crudos."""
    data = np.load(ruta_npz)
    return {
        "centros_hoja": data["centros_hoja"],
        "normales_hoja": data["normales_hoja"],
        "etiqueta": int(data["etiqueta"]),
        "profundidad_max": int(data["profundidad_max"]),
    }


def cargar_y_materializar(ruta_npz: str, resolucion: int) -> tuple:
    """
    Carga un octree disperso desde disco y lo materializa a un grid
    denso (4, R, R, R) SOLO en memoria RAM, en el momento de usarlo
    (p. ej. para alimentar Net5-Octree). El archivo en disco permanece
    disperso; nunca se guarda un grid denso completo.

    Retorna (grid_denso, etiqueta).
    """
    d = cargar_octree_disperso(ruta_npz)
    centros = d["centros_hoja"]
    normales = d["normales_hoja"]

    grid = np.zeros((4, resolucion, resolucion, resolucion), dtype=np.float32)
    if len(centros) > 0:
        idx = np.clip(((centros + 1.0) * 0.5 * resolucion).astype(np.int64),
                     0, resolucion - 1)
        grid[0, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
        grid[1, idx[:, 0], idx[:, 1], idx[:, 2]] = normales[:, 0]
        grid[2, idx[:, 0], idx[:, 1], idx[:, 2]] = normales[:, 1]
        grid[3, idx[:, 0], idx[:, 1], idx[:, 2]] = normales[:, 2]

    return grid, d["etiqueta"]


def ocupacion_por_nivel_desde_hojas(centros_hoja: np.ndarray, profundidad_max: int) -> np.ndarray:
    """
    Calcula el % de ocupacion por nivel DIRECTAMENTE desde los centros
    de las hojas persistidas en disco (ver guardar_octree_disperso),
    sin necesidad de reconstruir el arbol completo.

    Equivalencia matematica (verificada numericamente): en este octree,
    un nodo a profundidad d existe (no fue podado) si y solo si al
    menos una hoja ocupada cae dentro de su region. Como todas las
    hojas ocupadas estan exactamente a profundidad_max, basta con
    contar cuantas celdas UNICAS de resolucion 2^d contienen al menos
    un centro de hoja -- esto da identicamente el mismo resultado que
    contar_nodos_por_profundidad() sobre el arbol reconstruido.

    Util quirurgicamente para extraer features HCE (hce_extraccion.py)
    directamente desde el .npz disperso, sin reconstruir NodoOctree.

    Parametros
    ----------
    centros_hoja    : (N, 3) centros de las hojas ocupadas
    profundidad_max : L de la hoja (5 para R=32, 6 para R=64)

    Retorna
    -------
    array de tamaño (profundidad_max + 1): indice 0 = raiz, indice
    profundidad_max = hoja.
    """
    porcentajes = np.zeros(profundidad_max + 1, dtype=np.float32)
    if len(centros_hoja) == 0:
        return porcentajes

    for d in range(profundidad_max + 1):
        R_d = 2 ** d
        idx = np.clip(((centros_hoja + 1.0) * 0.5 * R_d).astype(np.int64),
                     0, R_d - 1)
        celdas_unicas = np.unique(idx, axis=0)
        porcentajes[d] = 100.0 * len(celdas_unicas) / (8 ** d)

    return porcentajes


# ──────────────────────────────────────────────────────────────
# COMPARATIVA DE MEMORIA: arbol disperso vs rejilla densa
# ──────────────────────────────────────────────────────────────

def comparar_memoria(raiz: NodoOctree, resolucion: int, profundidad_max: int) -> dict:
    """
    Calcula el ahorro de memoria REAL del octree disperso frente a la
    rejilla densa equivalente, para reportar honestamente en la tesis.
    """
    n_hojas = len(recolectar_hojas(raiz))
    n_nodos_totales = contar_nodos_totales(raiz)

    # Memoria dispersa: cada hoja guarda 3 floats (centro) + 3 floats
    # (normal) = 6 floats de 4 bytes = 24 bytes/hoja.
    bytes_por_hoja = 6 * 4
    memoria_dispersa_bytes = n_hojas * bytes_por_hoja

    # Memoria densa equivalente (formato anterior):
    # 4 canales x R^3 celdas x 4 bytes/celda (float32)
    memoria_densa_bytes = 4 * (resolucion ** 3) * 4

    return {
        "n_hojas_ocupadas": n_hojas,
        "n_nodos_totales_arbol": n_nodos_totales,
        "memoria_dispersa_kb": round(memoria_dispersa_bytes / 1024, 3),
        "memoria_densa_kb": round(memoria_densa_bytes / 1024, 3),
        "factor_ahorro": round(memoria_densa_bytes / max(memoria_dispersa_bytes, 1), 2),
        "pct_ocupacion_hoja": round(100 * n_hojas / (resolucion ** 3), 4),
    }


# ──────────────────────────────────────────────────────────────
# TEST RAPIDO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  TEST: Octree real con poda de ramas vacias")
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

    for R, prof_max in [(32, 5), (64, 6)]:
        print(f"\n--- Resolucion {R}^3 (profundidad hoja d={prof_max}) ---")

        raiz = construir_octree(puntos, normales, profundidad_max=prof_max)

        n_hojas = len(recolectar_hojas(raiz))
        n_nodos = contar_nodos_totales(raiz)
        ocup_por_nivel = ocupacion_por_nivel_arbol(raiz, prof_max)

        print(f"  Nodos totales en el arbol : {n_nodos:,}")
        print(f"  Hojas ocupadas            : {n_hojas:,}")
        print(f"  Ocupacion por nivel (%)   : {np.round(ocup_por_nivel, 2)}")

        mem = comparar_memoria(raiz, R, prof_max)
        print(f"  Memoria dispersa          : {mem['memoria_dispersa_kb']:.2f} KB")
        print(f"  Memoria densa (antes)     : {mem['memoria_densa_kb']:.2f} KB")
        print(f"  Factor de ahorro          : {mem['factor_ahorro']:.1f}x")

        grid = octree_a_grid_denso(raiz, R)
        n_ocup_grid = int(grid[0].sum())
        assert n_ocup_grid == n_hojas, "Inconsistencia hojas vs grid materializado"
        print(f"  Verificacion grid denso   : {n_ocup_grid} celdas (coincide con hojas) OK")

    print("\n" + "=" * 60)
    print("  Test completado: la poda de ramas vacias funciona")
    print("  correctamente y el arbol es un octree real.")
    print("=" * 60)
