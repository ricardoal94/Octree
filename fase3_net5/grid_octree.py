"""Representacion geometrica dispersa del grid-octree de OctNet.

El formato del Objetivo 1 guarda un unico octree global de profundidad 5
(R=32) o 6 (R=64), podando las ramas vacias. OctNet, en cambio, usa una
rejilla regular de octrees superficiales de profundidad maxima 3. Este modulo
realiza la conversion exacta sin materializar un volumen ``R^3``:

* la rejilla tiene ``(R / 8)^3`` arboles superficiales;
* una rama podada se recupera como una hoja vacia implicita;
* una hoja ocupada conserva ``[ocupacion, nx, ny, nz]``;
* cada hoja se describe por su origen entero y su arista en voxeles finos.

Tambien construye planes dispersos para convolucion 3x3x3 y max-pooling 2.
El plan de convolucion implementa el equivalente matematico de
``ten2oc(conv(oc2ten(O)))`` de las ecuaciones 4--8 del articulo, usando el
volumen de interseccion entre hojas. No crea tensores densos de alta
resolucion y no depende de PyTorch, por lo que su geometria puede probarse de
forma aislada.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import product

import numpy as np


PROFUNDIDAD_OCTREE_SUPERFICIAL = 3
ARISTA_OCTREE_SUPERFICIAL = 2 ** PROFUNDIDAD_OCTREE_SUPERFICIAL


@dataclass(frozen=True)
class PlanConvolucion:
    """Aristas de la operacion dispersa para un kernel 3x3x3.

    Cada entrada representa una contribucion
    ``coeficiente * W[kernel] @ x[entrada]`` a la hoja ``salida``. El indice
    del kernel sigue el orden C de ``(kx, ky, kz)`` y usa correlacion cruzada,
    la misma convencion de ``torch.nn.Conv3d``.
    """

    salida: np.ndarray
    entrada: np.ndarray
    kernel: np.ndarray
    coeficiente: np.ndarray
    offsets_kernel: np.ndarray

    @property
    def nbytes(self) -> int:
        """Memoria ocupada por los arreglos del plan."""

        return int(sum(
            arreglo.nbytes
            for arreglo in (
                self.salida,
                self.entrada,
                self.kernel,
                self.coeficiente,
                self.offsets_kernel,
            )
        ))


@dataclass(frozen=True)
class PlanPooling:
    """Geometria de salida y correspondencia hoja-entrada -> hoja-salida."""

    geometria_salida: "GeometriaGridOctree"
    entrada_a_salida: np.ndarray

    @property
    def nbytes(self) -> int:
        """Memoria adicional del mapeo de pooling."""

        return int(self.entrada_a_salida.nbytes)


@dataclass(frozen=True)
class GeometriaGridOctree:
    """Particion dispersa de un volumen mediante hojas cubicas.

    ``origenes`` y ``tamanos`` estan expresados en coordenadas enteras de la
    resolucion efectiva. Las hojas no se solapan y cubren todo el dominio,
    aunque solo se guarda un registro por hoja (no uno por voxel).
    """

    resolucion: int
    origenes: np.ndarray
    tamanos: np.ndarray

    def __post_init__(self) -> None:
        resolucion = int(self.resolucion)
        origenes = np.asarray(self.origenes, dtype=np.int32)
        tamanos = np.asarray(self.tamanos, dtype=np.int16)

        if resolucion < ARISTA_OCTREE_SUPERFICIAL:
            raise ValueError("La resolucion efectiva debe ser al menos 8")
        if resolucion % ARISTA_OCTREE_SUPERFICIAL != 0:
            raise ValueError("La resolucion debe ser multiplo de 8")
        if origenes.ndim != 2 or origenes.shape[1] != 3:
            raise ValueError("Los origenes deben tener forma (n_hojas, 3)")
        if tamanos.shape != (len(origenes),):
            raise ValueError("Debe existir un tamano por hoja")
        if len(origenes) == 0:
            raise ValueError("Un grid-octree debe contener al menos una hoja")
        if np.any(tamanos <= 0) or np.any(tamanos > ARISTA_OCTREE_SUPERFICIAL):
            raise ValueError("Las aristas de hoja deben pertenecer a [1, 8]")
        if np.any((tamanos & (tamanos - 1)) != 0):
            raise ValueError("Las aristas de hoja deben ser potencias de dos")
        if np.any(origenes < 0) or np.any(origenes + tamanos[:, None] > resolucion):
            raise ValueError("Se encontraron hojas fuera del dominio")
        if np.any(origenes % tamanos[:, None] != 0):
            raise ValueError("Los origenes deben estar alineados con su hoja")

        volumen = int(np.sum(tamanos.astype(np.int64) ** 3))
        if volumen != resolucion ** 3:
            raise ValueError(
                "Las hojas no cubren exactamente el volumen: "
                f"{volumen} != {resolucion ** 3}"
            )

        # Congela copias contiguas para que los planes cacheados no queden
        # invalidados por mutaciones externas.
        origenes = np.ascontiguousarray(origenes)
        tamanos = np.ascontiguousarray(tamanos)
        origenes.setflags(write=False)
        tamanos.setflags(write=False)
        object.__setattr__(self, "resolucion", resolucion)
        object.__setattr__(self, "origenes", origenes)
        object.__setattr__(self, "tamanos", tamanos)

    @property
    def n_hojas(self) -> int:
        return len(self.tamanos)

    @property
    def forma_rejilla(self) -> tuple[int, int, int]:
        lado = self.resolucion // ARISTA_OCTREE_SUPERFICIAL
        return (lado, lado, lado)

    @cached_property
    def hojas_por_arbol(self) -> dict[tuple[int, int, int], np.ndarray]:
        grupos: dict[tuple[int, int, int], list[int]] = {}
        coords = self.origenes // ARISTA_OCTREE_SUPERFICIAL
        for indice, coord in enumerate(coords):
            clave = tuple(int(v) for v in coord)
            grupos.setdefault(clave, []).append(indice)
        return {
            clave: np.asarray(indices, dtype=np.int64)
            for clave, indices in grupos.items()
        }

    @cached_property
    def plan_convolucion(self) -> PlanConvolucion:
        return construir_plan_convolucion(self)

    @cached_property
    def plan_pooling(self) -> PlanPooling:
        return construir_plan_pooling(self)

    @cached_property
    def _mapa_voxel_a_hoja_final(self) -> np.ndarray:
        """Construye una sola vez el mapa plano de la salida final 8^3."""

        if self.resolucion != ARISTA_OCTREE_SUPERFICIAL:
            raise ValueError(
                "Solo se permite expandir indices en la salida final 8^3"
            )
        mapa = np.empty((8, 8, 8), dtype=np.int64)
        for indice, (origen, tamano) in enumerate(zip(self.origenes, self.tamanos)):
            x, y, z = (int(v) for v in origen)
            s = int(tamano)
            mapa[x:x + s, y:y + s, z:z + s] = indice
        mapa = mapa.reshape(-1)
        mapa.setflags(write=False)
        return mapa

    def indices_voxel_a_hoja(self) -> np.ndarray:
        """Retorna el mapa plano voxel->hoja para la salida final 8^3.

        Se restringe deliberadamente a resolucion 8: este mapa se usa solo
        para alimentar la capa totalmente conectada de la Tabla 5 y nunca
        permite densificar las entradas R=32/R=64.
        """

        return self._mapa_voxel_a_hoja_final


@dataclass(frozen=True)
class MuestraGridOctree:
    """Geometria y atributos iniciales de una muestra."""

    geometria: GeometriaGridOctree
    atributos: np.ndarray

    def __post_init__(self) -> None:
        atributos = np.asarray(self.atributos, dtype=np.float32)
        if atributos.shape != (self.geometria.n_hojas, 4):
            raise ValueError("Los atributos deben tener forma (n_hojas, 4)")
        if not np.isfinite(atributos).all():
            raise ValueError("Los atributos contienen valores no finitos")
        atributos = np.ascontiguousarray(atributos)
        atributos.setflags(write=False)
        object.__setattr__(self, "atributos", atributos)


def precalcular_planes_net5(
    geometria: GeometriaGridOctree,
) -> tuple[GeometriaGridOctree, ...]:
    """Materializa los planes geometricos requeridos por Net5.

    La operacion no crea atributos densos ni modifica la geometria. Su
    objetivo es permitir que un ``DataLoader`` con trabajadores construya
    los planes en paralelo antes de que el lote llegue al hilo que controla
    la GPU. Los objetos quedan almacenados por ``cached_property`` y se
    reutilizan en todas las convoluciones de la misma escala.
    """

    resolucion = int(geometria.resolucion)
    if resolucion < ARISTA_OCTREE_SUPERFICIAL:
        raise ValueError("La resolucion debe ser al menos 8")
    cociente = resolucion // ARISTA_OCTREE_SUPERFICIAL
    if cociente * ARISTA_OCTREE_SUPERFICIAL != resolucion:
        raise ValueError("La resolucion debe ser multiplo de 8")
    n_pool = cociente.bit_length() - 1
    if 2 ** n_pool != cociente:
        raise ValueError("La resolucion efectiva debe ser potencia de dos")

    etapas = []
    actual = geometria
    for _ in range(n_pool):
        _ = actual.plan_convolucion
        etapas.append(actual)
        actual = actual.plan_pooling.geometria_salida
    _ = actual.indices_voxel_a_hoja()
    etapas.append(actual)
    return tuple(etapas)


def _descender_a_raiz_superficial(raiz, coord: tuple[int, int, int],
                                  profundidad_rejilla: int):
    nodo = raiz
    gx, gy, gz = coord
    for nivel in range(profundidad_rejilla):
        if nodo is None:
            return None
        desplazamiento = profundidad_rejilla - nivel - 1
        octante = (
            ((gx >> desplazamiento) & 1)
            | (((gy >> desplazamiento) & 1) << 1)
            | (((gz >> desplazamiento) & 1) << 2)
        )
        nodo = nodo.hijos[octante]
    return nodo


def convertir_a_grid_octree(raiz, resolucion: int) -> MuestraGridOctree:
    """Convierte el ``NodoOctree`` global del proyecto al grid-octree OctNet."""

    resolucion = int(resolucion)
    if resolucion not in (32, 64):
        raise ValueError("El protocolo oficial solo admite R=32 o R=64")
    profundidad_total = resolucion.bit_length() - 1
    if 2 ** profundidad_total != resolucion:
        raise ValueError("La resolucion debe ser una potencia de dos")
    if int(raiz.profundidad) != 0:
        raise ValueError("La conversion requiere el nodo raiz global")

    profundidad_rejilla = profundidad_total - PROFUNDIDAD_OCTREE_SUPERFICIAL
    lado_rejilla = 2 ** profundidad_rejilla
    origenes: list[tuple[int, int, int]] = []
    tamanos: list[int] = []
    atributos: list[tuple[float, float, float, float]] = []

    def emitir(nodo, origen: tuple[int, int, int], tamano: int,
               profundidad_local: int) -> None:
        if nodo is None:
            origenes.append(origen)
            tamanos.append(tamano)
            atributos.append((0.0, 0.0, 0.0, 0.0))
            return

        if profundidad_local == PROFUNDIDAD_OCTREE_SUPERFICIAL:
            if not nodo.es_hoja:
                raise ValueError("El arbol excede la profundidad declarada")
            normal = (
                np.asarray(nodo.normal_promedio, dtype=np.float32)
                if nodo.normal_promedio is not None
                else np.zeros(3, dtype=np.float32)
            )
            origenes.append(origen)
            tamanos.append(tamano)
            atributos.append(
                (1.0, float(normal[0]), float(normal[1]), float(normal[2]))
            )
            return

        if nodo.es_hoja:
            raise ValueError(
                "Una hoja ocupada aparecio antes de la resolucion maxima"
            )

        mitad = tamano // 2
        for octante in range(8):
            desplazamiento = (
                mitad if (octante & 1) else 0,
                mitad if (octante & 2) else 0,
                mitad if (octante & 4) else 0,
            )
            origen_hijo = tuple(
                origen[eje] + desplazamiento[eje] for eje in range(3)
            )
            emitir(
                nodo.hijos[octante], origen_hijo, mitad,
                profundidad_local + 1,
            )

    for gx, gy, gz in product(range(lado_rejilla), repeat=3):
        nodo = _descender_a_raiz_superficial(
            raiz, (gx, gy, gz), profundidad_rejilla,
        )
        origen = (
            gx * ARISTA_OCTREE_SUPERFICIAL,
            gy * ARISTA_OCTREE_SUPERFICIAL,
            gz * ARISTA_OCTREE_SUPERFICIAL,
        )
        emitir(nodo, origen, ARISTA_OCTREE_SUPERFICIAL, 0)

    geometria = GeometriaGridOctree(
        resolucion=resolucion,
        origenes=np.asarray(origenes, dtype=np.int32),
        tamanos=np.asarray(tamanos, dtype=np.int16),
    )
    return MuestraGridOctree(
        geometria=geometria,
        atributos=np.asarray(atributos, dtype=np.float32),
    )


def construir_plan_convolucion(geometria: GeometriaGridOctree) -> PlanConvolucion:
    """Construye el stencil disperso exacto de una convolucion 3x3x3."""

    bloques_salida: list[np.ndarray] = []
    bloques_entrada: list[np.ndarray] = []
    bloques_kernel: list[np.ndarray] = []
    bloques_coeficiente: list[np.ndarray] = []
    resolucion = geometria.resolucion
    origenes = geometria.origenes.astype(np.int64, copy=False)
    tamanos = geometria.tamanos.astype(np.int64, copy=False)
    codigo_tamano = {1: 0, 2: 1, 4: 2, 8: 3}

    def codificar(cajas: np.ndarray, tamano: int) -> np.ndarray:
        codigo = codigo_tamano[tamano]
        return (
            ((codigo * resolucion + cajas[:, 0]) * resolucion + cajas[:, 1])
            * resolucion
            + cajas[:, 2]
        )

    claves_hoja = np.empty(geometria.n_hojas, dtype=np.int64)
    for tamano in (1, 2, 4, 8):
        mascara = tamanos == tamano
        claves_hoja[mascara] = codificar(origenes[mascara], tamano)
    orden_claves = np.argsort(claves_hoja)
    claves_ordenadas = claves_hoja[orden_claves]

    for tamano_salida in (1, 2, 4, 8):
        indices_salida = np.flatnonzero(tamanos == tamano_salida)
        if len(indices_salida) == 0:
            continue
        origen_salida = origenes[indices_salida]
        volumen_salida = float(tamano_salida ** 3)

        for kx, ky, kz in product(range(3), repeat=3):
            indice_kernel = kx * 9 + ky * 3 + kz
            if indice_kernel == 13:
                bloques_salida.append(indices_salida)
                bloques_entrada.append(indices_salida)
                bloques_kernel.append(np.full(
                    len(indices_salida), indice_kernel, dtype=np.int8,
                ))
                bloques_coeficiente.append(np.ones(
                    len(indices_salida), dtype=np.float32,
                ))
                continue

            inferior = origen_salida + np.asarray(
                [kx - 1, ky - 1, kz - 1], dtype=np.int64,
            )
            superior = inferior + tamano_salida

            for tamano_entrada in (1, 2, 4, 8):
                inicio = np.floor_divide(inferior, tamano_entrada) * tamano_entrada
                n_eje = (tamano_salida + tamano_entrada - 1) // tamano_entrada + 1
                offsets = np.asarray(
                    list(product(range(n_eje), repeat=3)), dtype=np.int64,
                ) * tamano_entrada
                candidatos = inicio[:, None, :] + offsets[None, :, :]
                longitudes = (
                    np.minimum(superior[:, None, :], candidatos + tamano_entrada)
                    - np.maximum(inferior[:, None, :], candidatos)
                )
                validos = (
                    np.all(longitudes > 0, axis=2)
                    & np.all(candidatos >= 0, axis=2)
                    & np.all(candidatos + tamano_entrada <= resolucion, axis=2)
                )
                if not np.any(validos):
                    continue

                filas, columnas = np.nonzero(validos)
                cajas = candidatos[filas, columnas]
                claves = codificar(cajas, tamano_entrada)
                posiciones = np.searchsorted(claves_ordenadas, claves)
                existen = posiciones < len(claves_ordenadas)
                if np.any(existen):
                    posiciones_seguras = np.minimum(
                        posiciones, len(claves_ordenadas) - 1,
                    )
                    existen &= claves_ordenadas[posiciones_seguras] == claves
                if not np.any(existen):
                    continue

                filas = filas[existen]
                columnas = columnas[existen]
                posiciones = posiciones[existen]
                volumenes = np.prod(
                    longitudes[filas, columnas], axis=1, dtype=np.int64,
                )
                bloques_salida.append(indices_salida[filas])
                bloques_entrada.append(orden_claves[posiciones])
                bloques_kernel.append(np.full(
                    len(filas), indice_kernel, dtype=np.int8,
                ))
                bloques_coeficiente.append(
                    (volumenes / volumen_salida).astype(np.float32)
                )

    salida = np.concatenate(bloques_salida).astype(np.int64, copy=False)
    entrada = np.concatenate(bloques_entrada).astype(np.int64, copy=False)
    kernel = np.concatenate(bloques_kernel)
    coeficiente = np.concatenate(bloques_coeficiente)
    orden = np.argsort(kernel, kind="stable")
    kernel = kernel[orden]
    conteos = np.bincount(kernel, minlength=27)
    offsets_kernel = np.concatenate([
        np.asarray([0], dtype=np.int64),
        np.cumsum(conteos, dtype=np.int64),
    ])
    return PlanConvolucion(
        salida=salida[orden],
        entrada=entrada[orden],
        kernel=kernel,
        coeficiente=coeficiente[orden],
        offsets_kernel=offsets_kernel,
    )


def construir_plan_pooling(geometria: GeometriaGridOctree) -> PlanPooling:
    """Construye el max-pooling 2x2x2 directo sobre las hojas.

    Las hojas con arista mayor que uno se copian reduciendo su arista a la
    mitad. Las hojas finas de arista uno que convergen en el mismo voxel de
    salida quedan asociadas a una unica hoja, sobre la que el backend aplica
    el maximo por canal. Es la construccion de la ecuacion 10 de OctNet.
    """

    if geometria.resolucion <= ARISTA_OCTREE_SUPERFICIAL:
        raise ValueError("No se puede aplicar pooling por debajo de 8^3")
    if (geometria.resolucion // ARISTA_OCTREE_SUPERFICIAL) % 2 != 0:
        raise ValueError("La rejilla de octrees debe tener lado par")

    claves: list[tuple[int, int, int, int]] = []
    for origen, tamano_np in zip(geometria.origenes, geometria.tamanos):
        tamano = int(tamano_np)
        origen_salida = tuple(int(v) // 2 for v in origen)
        tamano_salida = max(1, tamano // 2)
        claves.append((*origen_salida, tamano_salida))

    claves_unicas = sorted(set(claves))
    clave_a_indice = {clave: i for i, clave in enumerate(claves_unicas)}
    entrada_a_salida = np.asarray(
        [clave_a_indice[clave] for clave in claves], dtype=np.int64,
    )
    origenes = np.asarray([clave[:3] for clave in claves_unicas], dtype=np.int32)
    tamanos = np.asarray([clave[3] for clave in claves_unicas], dtype=np.int16)
    geometria_salida = GeometriaGridOctree(
        resolucion=geometria.resolucion // 2,
        origenes=origenes,
        tamanos=tamanos,
    )
    return PlanPooling(
        geometria_salida=geometria_salida,
        entrada_a_salida=entrada_a_salida,
    )
