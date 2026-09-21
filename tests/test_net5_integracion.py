"""
test_net5_integracion.py - Prueba de integracion corta de Net5 (Objetivo 3)
=============================================================================
No entrena Net5 completo. Verifica, para R=32 y R=64, usando las MISMAS
particiones y semilla (seed=42) del Objetivo 2 (logs/particion_indices.npz,
generado por fase1_modelnet40/fase1_setup.py::particionar_dataset):

  1. Carga de datos reales (octrees dispersos -> grid denso (4,R,R,R))
  2. Dimensiones de entrada/salida del modelo
  3. Propagacion hacia adelante (forward)
  4. Propagacion hacia atras (backward / gradientes)
  5. Calculo de la perdida (CrossEntropyLoss)
  6. Guardado (y recarga) del checkpoint del modelo

Ejecucion:
    pytest tests/test_net5_integracion.py -v
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from net5_dataset import OctreeDataset
from net5_modelo import Net5Octree, crear_modelo

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
PARTICION_PATH = ROOT / "logs" / "particion_indices.npz"

N_TRAIN_MUESTRA = 8
N_VAL_MUESTRA = 4


def _cargar_indices_muestra():
    """Toma un subconjunto pequeno de la particion real (seed=42,
    Objetivo 2) para mantener la prueba de integracion rapida."""
    assert PARTICION_PATH.exists(), (
        f"No se encontro la particion de Objetivo 2 en {PARTICION_PATH}. "
        "Ejecute fase1_modelnet40/fase1_setup.py primero."
    )
    particion = np.load(str(PARTICION_PATH))
    idx_train = particion["idx_train"][:N_TRAIN_MUESTRA]
    idx_val = particion["idx_val"][:N_VAL_MUESTRA]
    return idx_train, idx_val


@pytest.fixture(scope="module")
def indices_muestra():
    return _cargar_indices_muestra()


@pytest.mark.parametrize("resolucion", [32, 64])
def test_integracion_net5_corta(resolucion, indices_muestra, tmp_path):
    idx_train, idx_val = indices_muestra
    raiz_resolucion = DATA_ROOT / f"octrees_{resolucion}"
    assert raiz_resolucion.exists(), f"No existe {raiz_resolucion}"

    torch.manual_seed(42)

    # 1. Carga de datos reales (octree disperso -> grid denso)
    ds_train = OctreeDataset(str(raiz_resolucion), resolucion, "train", idx_train)
    ds_val = OctreeDataset(str(raiz_resolucion), resolucion, "val", idx_val)
    assert len(ds_train) == N_TRAIN_MUESTRA
    assert len(ds_val) == N_VAL_MUESTRA

    grid, etiqueta = ds_train[0]
    assert grid.shape == (4, resolucion, resolucion, resolucion)
    assert grid.dtype == torch.float32
    assert 0 <= int(etiqueta) < 40

    lote_train = torch.stack([ds_train[i][0] for i in range(len(ds_train))])
    etq_train = torch.stack([ds_train[i][1] for i in range(len(ds_train))])
    lote_val = torch.stack([ds_val[i][0] for i in range(len(ds_val))])
    etq_val = torch.stack([ds_val[i][1] for i in range(len(ds_val))])

    # 2. Dimensiones del modelo
    device = torch.device("cpu")
    modelo, _ = crear_modelo(resolucion=resolucion, num_clases=40, device=device)
    assert isinstance(modelo, Net5Octree)

    # 3. Propagacion hacia adelante
    logits_train = modelo(lote_train)
    assert logits_train.shape == (N_TRAIN_MUESTRA, 40)
    assert torch.isfinite(logits_train).all()

    modelo.eval()
    with torch.no_grad():
        logits_val = modelo(lote_val)
    assert logits_val.shape == (N_VAL_MUESTRA, 40)
    modelo.train()

    # 5. Calculo de la perdida
    criterio = nn.CrossEntropyLoss()
    perdida = criterio(logits_train, etq_train)
    assert torch.isfinite(perdida)
    assert perdida.item() > 0

    # 4. Propagacion hacia atras
    optimizador = optim.Adam(modelo.parameters(), lr=1e-3)
    optimizador.zero_grad()
    perdida.backward()

    parametros_con_grad = [p for p in modelo.parameters() if p.requires_grad]
    assert len(parametros_con_grad) > 0
    assert all(p.grad is not None for p in parametros_con_grad), \
        "Algun parametro no recibio gradiente en el backward"
    norma_grad_total = sum(p.grad.norm().item() for p in parametros_con_grad)
    assert norma_grad_total > 0, "El gradiente total es cero"

    optimizador.step()

    # 6. Guardado y recarga del checkpoint
    ckpt_path = tmp_path / f"net5_test_R{resolucion}.pth"
    torch.save({
        "epoca": 1,
        "model_state": modelo.state_dict(),
        "optimizer_state": optimizador.state_dict(),
        "mejor_val_acc": 0.0,
        "resolucion": resolucion,
    }, ckpt_path)
    assert ckpt_path.exists()
    assert ckpt_path.stat().st_size > 0

    modelo_recargado, _ = crear_modelo(resolucion=resolucion, num_clases=40, device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    modelo_recargado.load_state_dict(ckpt["model_state"])

    modelo.eval()
    modelo_recargado.eval()
    with torch.no_grad():
        logits_original = modelo(lote_train)
        logits_recargado = modelo_recargado(lote_train)
    assert torch.allclose(logits_original, logits_recargado, atol=1e-6), \
        "Los logits del modelo recargado no coinciden con el original"


def test_capacidad_fija_entre_resoluciones():
    """La Tabla 4 de OctNet garantiza el mismo numero de parametros
    independientemente de la resolucion de entrada (32^3 o 64^3)."""
    modelo_32 = Net5Octree(resolucion=32, num_clases=40)
    modelo_64 = Net5Octree(resolucion=64, num_clases=40)
    assert modelo_32.contar_parametros() == modelo_64.contar_parametros()
