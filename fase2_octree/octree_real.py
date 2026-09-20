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

import sys
from pathlib import Path

import numpy as np


# Formato unico aprobado para los artefactos del objetivo especifico 1.
# Cualquier cambio incompatible debe incrementar esta version y disponer
# de una migracion explicita; los lectores rechazan silenciosamente los
# formatos antiguos para evitar mezclar resultados no comparables.
OCTREE_FORMAT_NAME = "octree_adaptativo_modelnet40"
OCTREE_FORMAT_VERSION = "1.0.0"

_METADATA_DEFAULTS = {
    "model_id": "",
    "categoria": "",
    "split": "",
    "semilla_muestreo": -1,
    "n_puntos_muestreo": -1,
    "archivo_origen_sha256": "",
}


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
    normal_coherencia : float o None -- MAGNITUD del promedio de normales
                   ANTES de normalizar a vector unitario (observacion de
                   Andres Gonzalez). Equivale a la "longitud resultante
                   media" de estadistica direccional: cercano a 1 si las
                   normales de los puntos de la hoja apuntan casi todas
                   en la misma direccion (superficie plana/coherente);
                   cercano a 0 si apuntan en direcciones dispersas que
                   se cancelan parcialmente (superficie rugosa, curva
                   pronunciada, o una hoja que abarca una arista/esquina).
                   normal_promedio, en cambio, SI se normaliza a vector
                   unitario (se conserva asi para materializar el grid
                   denso con una direccion normal consistente); por eso
                   la norma de normal_promedio es siempre ~1 o exactamente
                   0, y no debe usarse como medida de coherencia.
    """

    __slots__ = ("centro", "tamano", "profundidad", "ocupado", "es_hoja",
                "hijos", "normal_promedio", "normal_coherencia", "n_puntos")

    def __init__(self, centro: np.ndarray, tamano: float, profundidad: int):
        self.centro = centro
        self.tamano = tamano
        self.profundidad = profundidad
        self.ocupado = False
        self.es_hoja = True
        self.hijos = [None] * 8
        self.normal_promedio = None
        self.normal_coherencia = None
        self.n_puntos = 0


# ──────────────────────────────────────────────────────────────
# CONSTRUCCION RECURSIVA CON PODA
# ──────────────────────────────────────────────────────────────

def _indice_celda_final(puntos: np.ndarray, resolucion: int) -> np.ndarray:
    """
    Indice de celda (0..R-1 por eje) de cada punto en la rejilla final,
    calculado con la MISMA formula que ``construir_grid_octree`` (grid
    denso de referencia).
    """
    idx = ((puntos + 1.0) * 0.5 * resolucion).astype(np.int64)
    return np.clip(idx, 0, resolucion - 1)


def _octante_de_bits(idx_leaf: np.ndarray, profundidad_max: int, profundidad: int) -> np.ndarray:
    """
    Determina el octante (0-7) de cada punto en el nivel ``profundidad``
    a partir del indice de celda final ya calculado (``idx_leaf``), en
    vez de comparar la posicion del punto contra el centro del nodo.

    Esto garantiza que la particion recursiva del octree sea bit a bit
    identica a la cuantizacion directa del grid denso: usar comparaciones
    de punto flotante contra un ``centro`` acumulado a lo largo de varios
    niveles de recursion puede clasificar de forma distinta un punto muy
    cercano a un limite de celda que la formula directa de cuantizacion
    (division redondeada a la baja en un solo paso), produciendo una
    ocupacion identica pero con puntos repartidos entre celdas vecinas
    de forma distinta -- y por lo tanto normales promedio distintas.
    """
    bit = profundidad_max - 1 - profundidad
    bits_x = (idx_leaf[:, 0] >> bit) & 1
    bits_y = (idx_leaf[:, 1] >> bit) & 1
    bits_z = (idx_leaf[:, 2] >> bit) & 1
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
    idx_leaf: np.ndarray = None,
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
    if idx_leaf is None:
        idx_leaf = _indice_celda_final(puntos, 2 ** profundidad_max)

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
        # float64: coherente con la acumulacion de construir_grid_octree
        # (evita divergencias de redondeo float32 que se amplifican al
        # normalizar hojas con normales casi canceladas).
        normal_prom = normales.astype(np.float64).mean(axis=0)
        norma = np.linalg.norm(normal_prom)
        # CORRECCION (observacion de Andres Gonzalez): guardar la
        # magnitud ANTES de normalizar. Esta es la medida de coherencia
        # angular real -- normal_promedio (abajo) se normaliza a vector
        # unitario y pierde esta informacion (su norma queda siempre
        # ~1 o exactamente 0).
        nodo.normal_coherencia = float(norma)
        nodo.normal_promedio = (
            (normal_prom / norma).astype(np.float32)
            if norma > 1e-12 else np.zeros(3, dtype=np.float32)
        )
        return nodo

    nodo.es_hoja = False
    octantes = _octante_de_bits(idx_leaf, profundidad_max, profundidad)

    for i in range(8):
        mask = octantes == i
        if not mask.any():
            nodo.hijos[i] = None   # PODA: no se crea nodo ni se recursa
            continue
        centro_hijo = _centro_hijo(centro, tamano, i)
        nodo.hijos[i] = construir_octree(
            puntos[mask], normales[mask], profundidad_max,
            centro=centro_hijo, tamano=tamano / 2.0, profundidad=profundidad + 1,
            idx_leaf=idx_leaf[mask],
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

# ──────────────────────────────────────────────────────────────
# SERIALIZACION CON ESTRUCTURA JERARQUICA COMPLETA
# ──────────────────────────────────────────────────────────────
#
# CORRECCION (observacion de Andres Gonzalez): la version anterior de
# esta serializacion guardaba UNICAMENTE los centros y normales de las
# hojas ocupadas. Aunque eso permite recalcular estadisticas de
# ocupacion por nivel (ver ocupacion_por_nivel_desde_hojas, verificada
# matematicamente equivalente), NO preserva la estructura del arbol:
# al cargar el archivo se recuperaba una simple lista de celdas, sin
# relaciones padre-hijo ni informacion de que nodos internos existen.
#
# La version corregida serializa TODOS los nodos existentes del arbol
# (internos y hojas, nunca los podados) mediante un recorrido DFS
# (pre-orden), guardando para cada nodo:
#   - profundidad (uint8)
#   - mascara de hijos (uint8, bits 0-7): bit i =1 si hijos[i] existe.
#     Un nodo hoja tiene mascara=0 (por construccion, ver
#     construir_octree: un nodo solo dejar de subdividirse si es hoja).
#   - normal promedio (3 x float32): solo tiene significado en hojas;
#     se guarda como ceros en nodos internos.
#
# El centro y tamaño de cada nodo NO se guardan explicitamente: se
# reconstruyen de forma deterministica durante la carga, replicando la
# misma logica de subdivision (_centro_hijo) que uso construir_octree(),
# siguiendo el orden exacto de descenso indicado por las mascaras de
# hijos. Esto reduce el tamaño del archivo sin perder informacion.
#
# Con este formato, cargar_octree_disperso() reconstruye el NodoOctree
# completo (con todas las relaciones padre-hijo reales) y luego deriva
# de el las hojas -- por lo que el resto del pipeline (net5_dataset.py,
# hce_extraccion.py) sigue funcionando sin cambios, ahora respaldado
# por una persistencia que si preserva la jerarquia real.

def _mascara_hijos(nodo: NodoOctree) -> int:
    """Codifica en un entero de 8 bits cuales de los 8 hijos existen."""
    mascara = 0
    if not nodo.es_hoja:
        for i in range(8):
            if nodo.hijos[i] is not None:
                mascara |= (1 << i)
    return mascara


def _serializar_dfs(raiz: NodoOctree) -> tuple:
    """
    Recorre el arbol en pre-orden (DFS) y produce 4 arrays paralelos,
    uno por cada nodo EXISTENTE (internos y hojas, nunca podados):
        profundidades : (M,) uint8
        mascaras      : (M,) uint8 -- 0 para hojas
        normales      : (M, 3) float32 -- direccion UNITARIA, solo valida si mascara==0
        coherencias   : (M,) float32 -- magnitud PRE-normalizacion (observacion
                        de Andres Gonzalez), solo valida si mascara==0
    """
    profundidades = []
    mascaras = []
    normales = []
    coherencias = []

    def _rec(nodo):
        if nodo is None:
            return
        profundidades.append(nodo.profundidad)
        m = _mascara_hijos(nodo)
        mascaras.append(m)
        if nodo.es_hoja:
            normales.append(nodo.normal_promedio
                           if nodo.normal_promedio is not None
                           else np.zeros(3, dtype=np.float32))
            coherencias.append(
                nodo.normal_coherencia if nodo.normal_coherencia is not None else 0.0
            )
        else:
            normales.append(np.zeros(3, dtype=np.float32))
            coherencias.append(0.0)
            for i in range(8):
                if nodo.hijos[i] is not None:
                    _rec(nodo.hijos[i])

    _rec(raiz)

    return (
        np.array(profundidades, dtype=np.uint8),
        np.array(mascaras, dtype=np.uint8),
        np.array(normales, dtype=np.float32),
        np.array(coherencias, dtype=np.float32),
    )


def _validar_arrays_serializados(
    profundidades: np.ndarray,
    mascaras: np.ndarray,
    normales: np.ndarray,
    coherencias: np.ndarray,
    profundidad_max: int,
) -> None:
    """Valida la forma y las invariantes del formato antes de reconstruir."""
    n_nodos = len(profundidades)
    if n_nodos == 0:
        raise ValueError("El archivo de octree no contiene nodos")
    if profundidades.ndim != 1 or mascaras.ndim != 1 or coherencias.ndim != 1:
        raise ValueError("Profundidades, mascaras y coherencias deben ser vectores")
    if mascaras.shape != (n_nodos,) or coherencias.shape != (n_nodos,):
        raise ValueError("Los arrays serializados no tienen la misma cantidad de nodos")
    if normales.shape != (n_nodos, 3):
        raise ValueError("El array de normales debe tener forma (n_nodos, 3)")
    if int(profundidades[0]) != 0:
        raise ValueError("El primer nodo serializado debe ser la raiz (profundidad 0)")
    if int(profundidades.max()) != profundidad_max:
        raise ValueError("La profundidad maxima declarada no coincide con los nodos")
    if np.any(profundidades > profundidad_max):
        raise ValueError("Se encontraron nodos por debajo de la profundidad maxima")
    if np.any((mascaras == 0) & (profundidades != profundidad_max)):
        raise ValueError("Todas las hojas ocupadas deben estar en la profundidad maxima")
    if not np.isfinite(normales).all() or not np.isfinite(coherencias).all():
        raise ValueError("El archivo contiene normales o coherencias no finitas")
    if np.any(coherencias < -1e-6) or np.any(coherencias > 1.0 + 1e-5):
        raise ValueError("La coherencia normal debe pertenecer al intervalo [0, 1]")


def _reconstruir_dfs(profundidades: np.ndarray, mascaras: np.ndarray,
                     normales: np.ndarray, coherencias: np.ndarray) -> NodoOctree:
    """
    Reconstruye el arbol NodoOctree completo (topologia identica al
    original: mismos nodos, mismas relaciones padre-hijo, mismas
    profundidades) a partir de los 4 arrays paralelos producidos por
    _serializar_dfs(). El centro y tamaño de cada nodo se derivan
    deterministicamente de la ruta de descenso, replicando la misma
    formula usada durante la construccion original (_centro_hijo).
    """
    puntero = [0]   # indice mutable de lectura sobre los arrays planos

    def _rec(centro, tamano, profundidad_esperada):
        if puntero[0] >= len(profundidades):
            raise ValueError("Estructura serializada truncada")
        i = puntero[0]
        profundidad = int(profundidades[i])
        if profundidad != profundidad_esperada:
            raise ValueError(
                "Estructura serializada inconsistente: profundidad inesperada"
            )
        mascara = int(mascaras[i])
        normal = normales[i]
        coherencia = float(coherencias[i])
        puntero[0] += 1

        nodo = NodoOctree(centro, tamano, profundidad)
        nodo.ocupado = True

        if mascara == 0:
            # Nodo hoja (por construccion, un nodo interno siempre
            # tiene al menos un hijo -- ver construir_octree)
            nodo.es_hoja = True
            nodo.normal_promedio = normal
            nodo.normal_coherencia = coherencia
        else:
            nodo.es_hoja = False
            for bit in range(8):
                if mascara & (1 << bit):
                    centro_hijo = _centro_hijo(centro, tamano, bit)
                    nodo.hijos[bit] = _rec(
                        centro_hijo, tamano / 2.0, profundidad_esperada + 1,
                    )
                else:
                    nodo.hijos[bit] = None
        return nodo

    raiz = _rec(centro=np.zeros(3, dtype=np.float32), tamano=2.0, profundidad_esperada=0)
    if puntero[0] != len(profundidades):
        raise ValueError("Estructura serializada inconsistente: sobraron nodos")
    return raiz


def guardar_octree_disperso(raiz: NodoOctree, ruta_npz: str, etiqueta: int,
                            profundidad_max: int, metadatos: dict | None = None) -> None:
    """
    Guarda la ESTRUCTURA JERARQUICA COMPLETA del arbol (todos los nodos
    existentes, internos y hojas, con sus relaciones padre-hijo), no
    solo las hojas. El archivo sigue pesando proporcionalmente al
    numero de nodos reales del arbol (internos + hojas), tipicamente
    muy por debajo de una rejilla densa R^3 (ver comparar_memoria()).
    """
    profundidades, mascaras, normales, coherencias = _serializar_dfs(raiz)
    _validar_arrays_serializados(
        profundidades, mascaras, normales, coherencias, profundidad_max,
    )

    metadata = dict(_METADATA_DEFAULTS)
    if metadatos:
        desconocidos = set(metadatos) - set(metadata)
        if desconocidos:
            raise ValueError(
                "Metadatos no reconocidos: " + ", ".join(sorted(desconocidos))
            )
        metadata.update(metadatos)

    resolucion = 2 ** int(profundidad_max)

    np.savez_compressed(
        ruta_npz,
        format_name=np.asarray(OCTREE_FORMAT_NAME),
        format_version=np.asarray(OCTREE_FORMAT_VERSION),
        profundidades=profundidades,
        mascaras=mascaras,
        normales=normales,
        coherencias=coherencias,
        etiqueta=np.asarray(etiqueta, dtype=np.int16),
        profundidad_max=np.asarray(profundidad_max, dtype=np.uint8),
        resolucion=np.asarray(resolucion, dtype=np.uint16),
        model_id=np.asarray(str(metadata["model_id"])),
        categoria=np.asarray(str(metadata["categoria"])),
        split=np.asarray(str(metadata["split"])),
        semilla_muestreo=np.asarray(metadata["semilla_muestreo"], dtype=np.int64),
        n_puntos_muestreo=np.asarray(metadata["n_puntos_muestreo"], dtype=np.int64),
        archivo_origen_sha256=np.asarray(str(metadata["archivo_origen_sha256"])),
    )


def _leer_npz_validado(ruta_npz: str) -> tuple[dict, dict]:
    """Lee un NPZ v1, valida su esquema y retorna arrays y metadatos."""
    campos_obligatorios = {
        "format_name", "format_version", "profundidades", "mascaras",
        "normales", "coherencias", "etiqueta", "profundidad_max",
        "resolucion", "model_id", "categoria", "split",
        "semilla_muestreo", "n_puntos_muestreo", "archivo_origen_sha256",
    }
    with np.load(ruta_npz, allow_pickle=False) as data:
        faltantes = campos_obligatorios - set(data.files)
        if faltantes:
            raise ValueError(
                "Formato de octree antiguo o incompleto; faltan campos: "
                + ", ".join(sorted(faltantes))
            )

        format_name = str(data["format_name"].item())
        format_version = str(data["format_version"].item())
        if format_name != OCTREE_FORMAT_NAME:
            raise ValueError(f"Formato de octree no reconocido: {format_name!r}")
        if format_version != OCTREE_FORMAT_VERSION:
            raise ValueError(
                f"Version no compatible: {format_version!r}; "
                f"se requiere {OCTREE_FORMAT_VERSION!r}"
            )

        arrays = {
            "profundidades": np.asarray(data["profundidades"], dtype=np.uint8),
            "mascaras": np.asarray(data["mascaras"], dtype=np.uint8),
            "normales": np.asarray(data["normales"], dtype=np.float32),
            "coherencias": np.asarray(data["coherencias"], dtype=np.float32),
        }
        metadatos = {
            "format_name": format_name,
            "format_version": format_version,
            "etiqueta": int(data["etiqueta"].item()),
            "profundidad_max": int(data["profundidad_max"].item()),
            "resolucion": int(data["resolucion"].item()),
            "model_id": str(data["model_id"].item()),
            "categoria": str(data["categoria"].item()),
            "split": str(data["split"].item()),
            "semilla_muestreo": int(data["semilla_muestreo"].item()),
            "n_puntos_muestreo": int(data["n_puntos_muestreo"].item()),
            "archivo_origen_sha256": str(data["archivo_origen_sha256"].item()),
        }

    _validar_arrays_serializados(
        arrays["profundidades"], arrays["mascaras"], arrays["normales"],
        arrays["coherencias"], metadatos["profundidad_max"],
    )
    if metadatos["resolucion"] != 2 ** metadatos["profundidad_max"]:
        raise ValueError("Resolucion y profundidad maxima son incompatibles")
    return arrays, metadatos


def reconstruir_octree_desde_npz(ruta_npz: str) -> tuple:
    """
    Carga el archivo y reconstruye el arbol NodoOctree COMPLETO, con
    la topologia identica al arbol original (mismos nodos internos,
    mismas relaciones padre-hijo, mismas hojas con sus normales y su
    coherencia angular pre-normalizacion).

    Retorna (raiz: NodoOctree, etiqueta: int, profundidad_max: int).
    """
    arrays, metadatos = _leer_npz_validado(ruta_npz)
    raiz = _reconstruir_dfs(
        arrays["profundidades"], arrays["mascaras"], arrays["normales"],
        arrays["coherencias"],
    )
    etiqueta = metadatos["etiqueta"]
    profundidad_max = metadatos["profundidad_max"]
    return raiz, etiqueta, profundidad_max


def leer_metadatos_octree(ruta_npz: str) -> dict:
    """Retorna los metadatos validados sin exponer arrays internos."""
    _, metadatos = _leer_npz_validado(ruta_npz)
    return metadatos


def carga_binaria_sin_comprimir(raiz: NodoOctree) -> dict:
    """Calcula el payload exacto de los cuatro arrays estructurales."""
    profundidades, mascaras, normales, coherencias = _serializar_dfs(raiz)
    componentes = {
        "profundidades_bytes": int(profundidades.nbytes),
        "mascaras_bytes": int(mascaras.nbytes),
        "normales_bytes": int(normales.nbytes),
        "coherencias_bytes": int(coherencias.nbytes),
    }
    componentes["total_bytes"] = sum(componentes.values())
    componentes["bytes_por_nodo"] = 18
    return componentes


def carga_npz_sin_comprimir(ruta_npz: str | Path) -> dict:
    """Suma los bytes de todos los arrays contenidos en un NPZ validado.

    Esta magnitud incluye estructura y metadatos, pero excluye el overhead
    del contenedor ZIP. Se reporta separada del tamaño comprimido real.
    """
    _leer_npz_validado(str(ruta_npz))
    with np.load(ruta_npz, allow_pickle=False) as data:
        por_campo = {campo: int(data[campo].nbytes) for campo in data.files}
    return {
        "por_campo_bytes": por_campo,
        "total_bytes": int(sum(por_campo.values())),
    }


def cargar_octree_disperso(ruta_npz: str) -> dict:
    """
    Interfaz de compatibilidad con el resto del pipeline (net5_dataset.py,
    hce_extraccion.py): reconstruye el arbol completo desde el archivo
    (ver reconstruir_octree_desde_npz) y deriva de el las hojas ocupadas,
    devolviendo el mismo diccionario que la version anterior, mas el
    nuevo campo 'coherencias_hoja'.
    """
    arrays, metadatos = _leer_npz_validado(ruta_npz)
    raiz = _reconstruir_dfs(
        arrays["profundidades"], arrays["mascaras"], arrays["normales"],
        arrays["coherencias"],
    )
    etiqueta = metadatos["etiqueta"]
    profundidad_max = metadatos["profundidad_max"]
    hojas = recolectar_hojas(raiz)

    n = len(hojas)
    centros = np.zeros((n, 3), dtype=np.float32)
    normales = np.zeros((n, 3), dtype=np.float32)
    coherencias = np.zeros(n, dtype=np.float32)
    for i, h in enumerate(hojas):
        centros[i] = h.centro
        normales[i] = h.normal_promedio
        coherencias[i] = h.normal_coherencia if h.normal_coherencia is not None else 0.0

    return {
        "centros_hoja": centros,
        "normales_hoja": normales,
        "coherencias_hoja": coherencias,
        "etiqueta": etiqueta,
        "profundidad_max": profundidad_max,
        "metadatos": metadatos,
    }


def cargar_y_materializar(ruta_npz: str, resolucion: int) -> tuple:
    """
    Carga un octree disperso desde disco (reconstruyendo su jerarquia
    completa) y lo materializa a un grid denso (4, R, R, R) SOLO en
    memoria RAM, en el momento de usarlo (p. ej. para alimentar
    Net5-Octree). El archivo en disco permanece disperso con su
    estructura jerarquica completa; nunca se guarda un grid denso.

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
# MEDICION REAL DE MEMORIA (objetos Python) vs REJILLA DENSA
# ──────────────────────────────────────────────────────────────
#
# CORRECCION (observacion de Andres Gonzalez): la version anterior de
# comparar_memoria() calculaba la memoria como "24 bytes por hoja"
# (3 floats de centro + 3 floats de normal), IGNORANDO por completo:
#   - los nodos INTERNOS del arbol (solo contaba hojas)
#   - la lista de 8 punteros a hijos de cada nodo
#   - la sobrecarga real de los objetos Python (incluso con __slots__,
#     cada instancia y cada array de numpy tiene overhead propio)
#
# La version corregida MIDE (no estima) la memoria real en RAM
# recorriendo el arbol completo con sys.getsizeof() sobre cada objeto
# NodoOctree, su lista de hijos, y sus arrays de numpy (centro y
# normal). Esto captura el costo real de TODOS los nodos existentes,
# internos y hojas, tal como estan efectivamente representados en
# memoria durante la ejecucion.

def medir_memoria_real_python(raiz: NodoOctree) -> dict:
    """
    Mide la memoria REAL en RAM ocupada por el arbol de objetos Python,
    recorriendo TODOS los nodos existentes (internos y hojas, nunca los
    podados) y sumando sys.getsizeof() de:
      - el objeto NodoOctree en si (usa __slots__, sin __dict__)
      - la lista de 8 elementos hijos (aunque varios sean None)
      - el array numpy del centro (3 floats)
      - el array numpy de la normal promedio (solo en hojas ocupadas)

    A diferencia de una formula teorica simplificada, esto es una
    MEDICION real del costo en memoria de cada objeto tal como Python
    los representa, incluyendo su overhead individual.

    Retorna
    -------
    dict con n_nodos_medidos, memoria_real_bytes, memoria_real_kib
    (KiB = kibibytes, 1024 bytes; ver nota de unidades en el modulo)
    """
    n_nodos = [0]
    total_bytes = [0]
    vistos = set()

    def _sumar_una_vez(objeto):
        if objeto is None:
            return
        identidad = id(objeto)
        if identidad in vistos:
            return
        vistos.add(identidad)
        total_bytes[0] += sys.getsizeof(objeto)

    def _rec(nodo):
        if nodo is None:
            return
        n_nodos[0] += 1

        # Se recorre el grafo de objetos retenido por el arbol y cada
        # objeto se cuenta una sola vez. Ademas del nodo, la lista y los
        # arrays, se incluyen los escalares referenciados por __slots__;
        # la version anterior omitia, entre otros, normal_coherencia.
        _sumar_una_vez(nodo)
        _sumar_una_vez(nodo.hijos)
        _sumar_una_vez(nodo.centro)
        _sumar_una_vez(nodo.tamano)
        _sumar_una_vez(nodo.profundidad)
        _sumar_una_vez(nodo.ocupado)
        _sumar_una_vez(nodo.es_hoja)
        _sumar_una_vez(nodo.normal_promedio)
        _sumar_una_vez(nodo.normal_coherencia)
        _sumar_una_vez(nodo.n_puntos)

        if not nodo.es_hoja:
            for hijo in nodo.hijos:
                _rec(hijo)

    _rec(raiz)

    return {
        "n_nodos_medidos": n_nodos[0],
        "memoria_python_bytes": total_bytes[0],
        "memoria_python_kib": round(total_bytes[0] / 1024, 3),
    }


def comparar_memoria(raiz: NodoOctree, resolucion: int, profundidad_max: int) -> dict:
    """
    Compara tres magnitudes de memoria, claramente diferenciadas y con
    nombres que no deben confundirse entre si:

    1. memoria_python_kib: memoria ocupada por la REPRESENTACION EN
       PYTHON del arbol completo (internos + hojas) -- objetos
       NodoOctree, listas de 8 hijos, y arrays de numpy de centro y
       normal -- medida via sys.getsizeof() recorriendo cada nodo
       (ver medir_memoria_real_python()). Distinta del pico transitorio
       durante la construccion (medido aparte con tracemalloc en
       medir_tiempo_memoria.py) y del almacenamiento en disco (tamaño
       real del archivo .npz).

    2. memoria_binaria_estimada_kib: estimacion TEORICA (no medida) del
       tamaño minimo si se serializara el arbol en un formato binario
       compacto (1 byte profundidad + 1 byte mascara de hijos + 12
       bytes de normal + 4 bytes de coherencia por CADA nodo existente,
       interno u hoja). Esta
       cifra NO es memoria de objetos Python; es una cota inferior de
       referencia para comparar contra el archivo .npz real (que ademas
       incluye compresion y overhead del formato .npz).

    3. memoria_densa_kib: memoria ocupada por la REPRESENTACION EN
       PYTHON de la rejilla densa equivalente (formato descartado tras
       la observacion de Andres Gonzalez), medida con el MISMO criterio
       que memoria_python_kib (sys.getsizeof sobre el array materializado,
       sin sumar nbytes por separado -- ver correccion de doble conteo).

    Retorna un dict con las tres magnitudes y los factores de ahorro
    correspondientes.
    """
    n_hojas = len(recolectar_hojas(raiz))
    n_nodos_totales = contar_nodos_totales(raiz)

    medicion_python = medir_memoria_real_python(raiz)

    # Payload exacto del formato v1 sin compresion: 1 byte de profundidad,
    # 1 byte de mascara, 12 bytes de normal y 4 bytes de coherencia.
    # La version anterior omitia los 4 bytes de coherencia por nodo.
    carga_binaria = carga_binaria_sin_comprimir(raiz)
    memoria_binaria_estimada_bytes = carga_binaria["total_bytes"]

    # Memoria de la representacion densa equivalente EN PYTHON (formato
    # anterior, descartado). CORRECCION: antes se sumaba una formula
    # (4*R^3*4 bytes, el tamaño crudo de los datos) que no correspondia
    # al mismo criterio de medicion que memoria_python_kib. Ahora se
    # materializa el array y se mide con sys.getsizeof(), igual que el
    # arbol, para una comparacion consistente.
    grid_denso_temp = octree_a_grid_denso(raiz, resolucion)
    memoria_densa_bytes = sys.getsizeof(grid_denso_temp)

    return {
        "n_hojas_ocupadas": n_hojas,
        "n_nodos_totales_arbol": n_nodos_totales,
        "memoria_python_kib": medicion_python["memoria_python_kib"],
        "memoria_binaria_estimada_kib": round(memoria_binaria_estimada_bytes / 1024, 3),
        "carga_binaria_sin_comprimir_bytes": memoria_binaria_estimada_bytes,
        "bytes_binarios_por_nodo": carga_binaria["bytes_por_nodo"],
        "memoria_densa_kib": round(memoria_densa_bytes / 1024, 3),
        "factor_ahorro_python_vs_densa": round(
            memoria_densa_bytes / max(medicion_python["memoria_python_bytes"], 1), 2
        ),
        "factor_ahorro_binario_vs_densa": round(
            memoria_densa_bytes / max(memoria_binaria_estimada_bytes, 1), 2
        ),
        "pct_ocupacion_hoja": round(100 * n_hojas / (resolucion ** 3), 4),
    }


# ──────────────────────────────────────────────────────────────
# TEST RAPIDO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import sys as sys_module
    from pathlib import Path as PathModule

    parser = argparse.ArgumentParser(
        description="Test de octree_real.py: poda de ramas vacias y "
                    "comparativa de memoria. Sin --off, usa una esfera "
                    "sintetica; con --off, usa la malla real indicada."
    )
    parser.add_argument("--off", type=str, default=None,
                        help="Ruta a un archivo .off real (opcional)")
    parser.add_argument("--salida", type=str, default=None,
                        help="Ruta JSON donde guardar los resultados (opcional)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TEST: Octree real con poda de ramas vacias")
    print("=" * 60)

    nombre_objeto = "esfera_sintetica"

    if args.off:
        # Usar una malla real: reutiliza el pipeline de octree.py para
        # leer, normalizar y muestrear la superficie con normales.
        sys_module.path.insert(0, str(PathModule(__file__).parent))
        from octree import (
            leer_off, normalizar_malla, muestrear_superficie_con_normales,
        )

        print(f"\n  Archivo: {args.off}")
        verts, caras = leer_off(args.off)
        verts = normalizar_malla(verts)
        rng = np.random.default_rng(42)
        puntos, normales = muestrear_superficie_con_normales(verts, caras, 20000, rng)
        nombre_objeto = PathModule(args.off).stem
        print(f"  Vertices: {len(verts)}   Puntos muestreados: {len(puntos)}")
    else:
        print("\n  (Sin --off: usando esfera sintetica de prueba)")
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

    resultados_json = {"archivo": nombre_objeto}

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
        print(f"  Memoria Python (arbol)    : {mem['memoria_python_kib']:.2f} KiB")
        print(f"  Memoria densa (referencia): {mem['memoria_densa_kib']:.2f} KiB")
        print(f"  Factor de ahorro (Python) : {mem['factor_ahorro_python_vs_densa']:.1f}x")

        grid = octree_a_grid_denso(raiz, R)
        n_ocup_grid = int(grid[0].sum())
        assert n_ocup_grid == n_hojas, "Inconsistencia hojas vs grid materializado"
        print(f"  Verificacion grid denso   : {n_ocup_grid} celdas (coincide con hojas) OK")

        resultados_json[f"R{R}"] = {
            "profundidad_max": prof_max,
            "n_nodos_totales": n_nodos,
            "n_hojas_ocupadas": n_hojas,
            "ocupacion_por_nivel_pct": [round(float(v), 4) for v in ocup_por_nivel],
            "memoria_python_kib": mem["memoria_python_kib"],
            "memoria_binaria_estimada_kib": mem["memoria_binaria_estimada_kib"],
            "memoria_densa_kib": mem["memoria_densa_kib"],
            "factor_ahorro_python_vs_densa": mem["factor_ahorro_python_vs_densa"],
            "factor_ahorro_binario_vs_densa": mem["factor_ahorro_binario_vs_densa"],
        }

    print("\n" + "=" * 60)
    print("  Test completado: la poda de ramas vacias funciona")
    print("  correctamente y el arbol es un octree real.")
    print("=" * 60)

    if args.off:
        ruta_salida = PathModule(args.salida) if args.salida else \
                     PathModule(f"octree_real_{nombre_objeto}.json")
        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(resultados_json, f, indent=2, ensure_ascii=False)
        print(f"\n  Resultados guardados en: {ruta_salida.resolve()}")
