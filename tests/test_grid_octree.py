"""Pruebas geometricas del backend OctNet sin dependencia de PyTorch."""

import pickle
from itertools import product

import numpy as np
import pytest

import grid_octree
from cuantizacion import cuantizar_indices_octree
from grid_octree import (
    construir_plan_convolucion,
    convertir_a_grid_octree,
    precalcular_planes_net5,
)
from octree_real import construir_octree


def _nube_controlada():
    puntos = np.asarray([
        [-0.91, -0.77, -0.63],
        [-0.12, 0.18, 0.37],
        [0.43, 0.61, 0.82],
        [0.88, -0.71, 0.09],
    ], dtype=np.float32)
    normales = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32)
    normales /= np.linalg.norm(normales, axis=1, keepdims=True)
    return puntos, normales


def _expandir_hojas(geometria, atributos):
    canales = atributos.shape[1]
    volumen = np.empty(
        (canales, geometria.resolucion, geometria.resolucion,
         geometria.resolucion),
        dtype=np.float32,
    )
    for indice, (origen, tamano) in enumerate(
        zip(geometria.origenes, geometria.tamanos)
    ):
        x, y, z = (int(v) for v in origen)
        s = int(tamano)
        volumen[:, x:x + s, y:y + s, z:z + s] = atributos[indice, :, None, None, None]
    return volumen


@pytest.mark.parametrize("resolucion,profundidad", [(32, 5), (64, 6)])
def test_conversion_preserva_ocupacion_y_reconstruye_vacios(
    resolucion, profundidad,
):
    puntos, normales = _nube_controlada()
    raiz = construir_octree(puntos, normales, profundidad)
    muestra = convertir_a_grid_octree(raiz, resolucion)
    geometria = muestra.geometria

    assert geometria.forma_rejilla == (resolucion // 8,) * 3
    assert len(geometria.hojas_por_arbol) == (resolucion // 8) ** 3
    assert np.sum(geometria.tamanos.astype(np.int64) ** 3) == resolucion ** 3
    assert np.any((muestra.atributos[:, 0] == 0) & (geometria.tamanos > 1))

    ocupadas = muestra.atributos[:, 0] == 1
    assert np.all(geometria.tamanos[ocupadas] == 1)
    indices_esperados = cuantizar_indices_octree(puntos, resolucion)
    assert set(map(tuple, geometria.origenes[ocupadas])) == set(
        map(tuple, indices_esperados)
    )


@pytest.mark.parametrize("resolucion,profundidad", [(32, 5), (64, 6)])
def test_convolucion_dispersa_equivale_a_referencia_densa(
    resolucion, profundidad,
):
    puntos, normales = _nube_controlada()
    muestra = convertir_a_grid_octree(
        construir_octree(puntos, normales, profundidad_max=profundidad),
        resolucion,
    )
    geometria = muestra.geometria
    rng = np.random.default_rng(42)
    entrada = rng.normal(size=(geometria.n_hojas, 2)).astype(np.float32)
    pesos = rng.normal(size=(3, 2, 3, 3, 3)).astype(np.float32)
    bias = rng.normal(size=3).astype(np.float32)

    plan = geometria.plan_convolucion
    salida_dispersa = np.broadcast_to(
        bias, (geometria.n_hojas, 3),
    ).copy()
    pesos_planos = pesos.reshape(3, 2, 27)
    for kernel in range(27):
        mascara = plan.kernel == kernel
        contribucion = (
            entrada[plan.entrada[mascara]] @ pesos_planos[:, :, kernel].T
        ) * plan.coeficiente[mascara, None]
        np.add.at(salida_dispersa, plan.salida[mascara], contribucion)

    volumen = _expandir_hojas(geometria, entrada)
    acolchado = np.pad(volumen, ((0, 0), (1, 1), (1, 1), (1, 1)))
    salida_densa = np.broadcast_to(
        bias[:, None, None, None],
        (3, resolucion, resolucion, resolucion),
    ).copy()
    for kx in range(3):
        for ky in range(3):
            for kz in range(3):
                ventana = acolchado[
                    :,
                    kx:kx + resolucion,
                    ky:ky + resolucion,
                    kz:kz + resolucion,
                ]
                salida_densa += np.einsum(
                    "oi,ixyz->oxyz", pesos[:, :, kx, ky, kz], ventana,
                )

    esperada = np.empty_like(salida_dispersa)
    for indice, (origen, tamano) in enumerate(
        zip(geometria.origenes, geometria.tamanos)
    ):
        x, y, z = (int(v) for v in origen)
        s = int(tamano)
        esperada[indice] = salida_densa[
            :, x:x + s, y:y + s, z:z + s,
        ].mean(axis=(1, 2, 3))
    np.testing.assert_allclose(salida_dispersa, esperada, rtol=2e-5, atol=2e-5)


def test_pooling_disperso_equivale_a_maxpool_denso():
    puntos, normales = _nube_controlada()
    geometria = convertir_a_grid_octree(
        construir_octree(puntos, normales, profundidad_max=5), 32,
    ).geometria
    rng = np.random.default_rng(7)
    entrada = rng.normal(size=(geometria.n_hojas, 3)).astype(np.float32)
    plan = geometria.plan_pooling

    salida = np.full((plan.geometria_salida.n_hojas, 3), -np.inf, np.float32)
    np.maximum.at(salida, plan.entrada_a_salida, entrada)

    volumen = _expandir_hojas(geometria, entrada)
    denso_pool = volumen.reshape(3, 16, 2, 16, 2, 16, 2).max(
        axis=(2, 4, 6),
    )
    for indice, (origen, tamano) in enumerate(zip(
        plan.geometria_salida.origenes,
        plan.geometria_salida.tamanos,
    )):
        x, y, z = (int(v) for v in origen)
        s = int(tamano)
        bloque = denso_pool[:, x:x + s, y:y + s, z:z + s]
        esperado = bloque[:, 0, 0, 0]
        np.testing.assert_allclose(
            bloque,
            np.broadcast_to(esperado[:, None, None, None], bloque.shape),
        )
        np.testing.assert_allclose(salida[indice], esperado)


@pytest.mark.parametrize(
    "resolucion,profundidad,resoluciones_esperadas",
    [(32, 5, (32, 16, 8)), (64, 6, (64, 32, 16, 8))],
)
def test_precalculo_net5_conserva_planes_por_escala(
    resolucion, profundidad, resoluciones_esperadas,
):
    puntos, normales = _nube_controlada()
    geometria = convertir_a_grid_octree(
        construir_octree(puntos, normales, profundidad_max=profundidad),
        resolucion,
    ).geometria

    etapas = precalcular_planes_net5(geometria)

    assert tuple(etapa.resolucion for etapa in etapas) == resoluciones_esperadas
    for etapa in etapas[:-1]:
        assert "plan_convolucion" in etapa.__dict__
        assert "plan_pooling" in etapa.__dict__
        assert etapa.plan_convolucion.nbytes > 0
        assert etapa.plan_pooling.nbytes > 0
    assert "_mapa_voxel_a_hoja_final" in etapas[-1].__dict__
    assert etapas[-1].indices_voxel_a_hoja().shape == (8 ** 3,)


def test_planes_precalculados_sobreviven_transferencia_entre_procesos():
    puntos, normales = _nube_controlada()
    geometria = convertir_a_grid_octree(
        construir_octree(puntos, normales, profundidad_max=5), 32,
    ).geometria
    precalcular_planes_net5(geometria)

    restaurada = pickle.loads(pickle.dumps(geometria))

    assert "plan_convolucion" in restaurada.__dict__
    assert "plan_pooling" in restaurada.__dict__
    assert restaurada.plan_convolucion.nbytes == geometria.plan_convolucion.nbytes
    assert (
        "plan_convolucion"
        in restaurada.plan_pooling.geometria_salida.__dict__
    )


def _nube_densa(n_puntos: int = 3000, semilla: int = 3):
    """Superficie esferica con hojas de todas las aristas (1, 2, 4 y 8)."""

    rng = np.random.default_rng(semilla)
    normales = rng.normal(size=(n_puntos, 3)).astype(np.float32)
    normales /= np.linalg.norm(normales, axis=1, keepdims=True)
    return (0.85 * normales).astype(np.float32), normales


def _plan_convolucion_referencia(geometria):
    """Algoritmo original de enumeracion exhaustiva de candidatos.

    Se conserva aqui como referencia: el plan optimizado debe producir
    exactamente las mismas aristas, coeficientes y orden.
    """

    bloques_salida, bloques_entrada = [], []
    bloques_kernel, bloques_coeficiente = [], []
    resolucion = geometria.resolucion
    origenes = geometria.origenes.astype(np.int64, copy=False)
    tamanos = geometria.tamanos.astype(np.int64, copy=False)
    codigo_tamano = {1: 0, 2: 1, 4: 2, 8: 3}

    def codificar(cajas, tamano):
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
                inicio = (
                    np.floor_divide(inferior, tamano_entrada) * tamano_entrada
                )
                n_eje = (
                    (tamano_salida + tamano_entrada - 1) // tamano_entrada + 1
                )
                offsets = np.asarray(
                    list(product(range(n_eje), repeat=3)), dtype=np.int64,
                ) * tamano_entrada
                candidatos = inicio[:, None, :] + offsets[None, :, :]
                longitudes = (
                    np.minimum(superior[:, None, :],
                               candidatos + tamano_entrada)
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
                claves = codificar(candidatos[filas, columnas], tamano_entrada)
                posiciones = np.searchsorted(claves_ordenadas, claves)
                seguras = np.minimum(posiciones, len(claves_ordenadas) - 1)
                existen = claves_ordenadas[seguras] == claves
                if not np.any(existen):
                    continue
                filas = filas[existen]
                columnas = columnas[existen]
                volumenes = np.prod(
                    longitudes[filas, columnas], axis=1, dtype=np.int64,
                )
                bloques_salida.append(indices_salida[filas])
                bloques_entrada.append(orden_claves[posiciones[existen]])
                bloques_kernel.append(np.full(
                    len(filas), indice_kernel, dtype=np.int8,
                ))
                bloques_coeficiente.append(
                    (volumenes / volumen_salida).astype(np.float32)
                )

    kernel = np.concatenate(bloques_kernel)
    orden = np.argsort(kernel, kind="stable")
    kernel = kernel[orden]
    offsets_kernel = np.concatenate([
        np.asarray([0], dtype=np.int64),
        np.cumsum(np.bincount(kernel, minlength=27), dtype=np.int64),
    ])
    return {
        "salida": np.concatenate(bloques_salida).astype(np.int64)[orden],
        "entrada": np.concatenate(bloques_entrada).astype(np.int64)[orden],
        "kernel": kernel,
        "coeficiente": np.concatenate(bloques_coeficiente)[orden],
        "offsets_kernel": offsets_kernel,
    }


@pytest.mark.parametrize("backend", ["publico", "numpy", "numba"])
@pytest.mark.parametrize("nube", ["controlada", "densa"])
@pytest.mark.parametrize("resolucion,profundidad", [(32, 5), (64, 6)])
def test_plan_convolucion_identico_a_enumeracion_exhaustiva(
    backend, nube, resolucion, profundidad,
):
    if backend == "numba":
        pytest.importorskip("numba")
    construir = {
        "publico": construir_plan_convolucion,
        "numpy": grid_octree._construir_plan_convolucion_numpy,
        "numba": grid_octree._construir_plan_convolucion_numba,
    }[backend]
    puntos, normales = (
        _nube_controlada() if nube == "controlada" else _nube_densa()
    )
    geometria = convertir_a_grid_octree(
        construir_octree(puntos, normales, profundidad_max=profundidad),
        resolucion,
    ).geometria
    assert set(np.unique(geometria.tamanos)) == {1, 2, 4, 8}

    while geometria.resolucion > 8:
        plan = construir(geometria)
        esperado = _plan_convolucion_referencia(geometria)
        for campo, valor in esperado.items():
            obtenido = getattr(plan, campo)
            assert obtenido.dtype == valor.dtype, campo
            np.testing.assert_array_equal(obtenido, valor, err_msg=campo)
        geometria = geometria.plan_pooling.geometria_salida
