"""Criterios de aceptacion del backend PyTorch OctNet.

Se omiten en entornos sin PyTorch. No requieren ModelNet40: construyen
grid-octrees sinteticos con el mismo formato jerarquico del proyecto.
"""

import io
import inspect

import numpy as np
import pytest


torch = pytest.importorskip("torch", reason="El backend nativo requiere PyTorch")
import torch.nn as nn

from grid_octree import convertir_a_grid_octree, precalcular_planes_net5
from net5_modelo import Net5Octree
from octnet_backend import LoteGridOctree
from octree_real import construir_octree


def _lote_sintetico(resolucion: int, batch_size: int = 1):
    profundidad = {32: 5, 64: 6}[resolucion]
    puntos_base = np.asarray([
        [-0.91, -0.77, -0.63],
        [0.43, 0.61, 0.82],
    ], dtype=np.float32)
    normales = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    muestras = []
    for i in range(batch_size):
        puntos = puntos_base.copy()
        puntos[:, 0] += np.float32(i * 0.001)
        raiz = construir_octree(puntos, normales, profundidad)
        muestras.append(convertir_a_grid_octree(raiz, resolucion))
    return LoteGridOctree.desde_muestras(muestras)


@pytest.mark.parametrize("resolucion", [32, 64])
def test_forward_backward_y_ausencia_de_conv3d(resolucion):
    torch.manual_seed(42)
    modelo = Net5Octree(resolucion=resolucion, num_clases=40, dropout=0.0)
    assert not any(isinstance(modulo, nn.Conv3d) for modulo in modelo.modules())

    lote = _lote_sintetico(resolucion)
    logits = modelo(lote)
    assert logits.shape == (1, 40)
    assert torch.isfinite(logits).all()

    perdida = logits.square().mean()
    perdida.backward()
    parametros = [p for p in modelo.parameters() if p.requires_grad]
    assert parametros
    assert all(p.grad is not None for p in parametros)
    assert all(torch.isfinite(p.grad).all() for p in parametros)


def test_checkpoint_reproduce_logits_en_eval():
    torch.manual_seed(42)
    lote = _lote_sintetico(32)
    modelo = Net5Octree(resolucion=32, dropout=0.5).eval()
    with torch.no_grad():
        esperado = modelo(lote)

    buffer = io.BytesIO()
    torch.save(modelo.state_dict(), buffer)
    buffer.seek(0)
    recargado = Net5Octree(resolucion=32, dropout=0.5).eval()
    recargado.load_state_dict(torch.load(buffer, map_location="cpu"))
    with torch.no_grad():
        obtenido = recargado(lote)
    assert torch.allclose(esperado, obtenido, atol=1e-6)


def test_capacidad_fija_y_guardas_sin_expansion_r3():
    modelo_32 = Net5Octree(resolucion=32)
    modelo_64 = Net5Octree(resolucion=64)
    assert modelo_32.contar_parametros() == modelo_64.contar_parametros()

    lote = _lote_sintetico(32)
    with pytest.raises(ValueError, match="salida final 8"):
        lote.geometrias[0].indices_voxel_a_hoja()

    fuente = inspect.getsource(type(modelo_32))
    assert "Conv3d" not in fuente


def test_perfil_backend_registra_planes_sin_alterar_salida():
    torch.manual_seed(42)
    lote = _lote_sintetico(32).activar_perfil()
    modelo = Net5Octree(resolucion=32, dropout=0.0).eval()

    with torch.no_grad():
        logits = modelo(lote)

    perfil = lote.resumen_perfil()
    assert logits.shape == (1, 40)
    assert perfil["n_planes_convolucion"] > 0
    assert perfil["n_planes_pooling"] > 0
    assert perfil["n_mapas_finales"] == 1
    assert perfil["plan_convolucion_cpu_s"] >= 0.0
    assert perfil["n_interacciones_convolucion"] > 0
    assert perfil["bytes_planes_convolucion"] > 0
    assert perfil["bytes_planes_pooling"] > 0
    assert perfil["bytes_mapas_finales"] > 0


def test_precalculo_y_ruta_batch_uno_conservan_logits():
    torch.manual_seed(17)
    modelo = Net5Octree(resolucion=32, dropout=0.0).eval()
    lote_referencia = _lote_sintetico(32)
    lote_precalculado = _lote_sintetico(32)
    etapas = precalcular_planes_net5(lote_precalculado.geometrias[0])

    plan = etapas[0].plan_convolucion
    salida, entrada, coeficiente, offsets = (
        lote_precalculado._plan_convolucion_numpy()
    )
    assert salida is plan.salida
    assert entrada is plan.entrada
    assert coeficiente is plan.coeficiente
    assert offsets is plan.offsets_kernel

    with torch.no_grad():
        esperado = modelo(lote_referencia)
        obtenido = modelo(lote_precalculado)
    assert torch.allclose(esperado, obtenido, atol=1e-6)


def test_ensamble_multimuestra_conserva_primera_prediccion():
    torch.manual_seed(23)
    modelo = Net5Octree(resolucion=32, dropout=0.0).eval()
    lote_uno = _lote_sintetico(32, batch_size=1)
    lote_dos = _lote_sintetico(32, batch_size=2)

    with torch.no_grad():
        logits_uno = modelo(lote_uno)
        logits_dos = modelo(lote_dos)

    assert logits_dos.shape == (2, 40)
    assert torch.allclose(logits_uno[0], logits_dos[0], atol=1e-6)
