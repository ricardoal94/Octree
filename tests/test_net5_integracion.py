"""Integración local de la referencia densa de la Tabla 5.

Estas pruebas requieren PyTorch y los octrees reales. Validan el cableado
diagnóstico, no el backend OctNet ni el cierre del Objetivo 3.
"""

from pathlib import Path

import pytest


torch = pytest.importorskip(
    "torch",
    reason="La integración densa opcional requiere PyTorch",
)
import torch.nn as nn
import torch.optim as optim

from net5_dataset import DenseOctreeDataset
from net5_modelo import DenseTabla5Reference, crear_modelo_denso_referencia
from particion_objetivo2 import (
    cargar_o_crear_particion,
    listar_muestras_octree,
    seleccionar_subconjunto_balanceado,
)


pytestmark = pytest.mark.dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
N_TRAIN_MUESTRA = 2
N_VAL_MUESTRA = 2


@pytest.fixture(scope="module")
def manifest_temporal(tmp_path_factory):
    for resolucion in (32, 64):
        if not (DATA_ROOT / f"octrees_{resolucion}").exists():
            pytest.skip("No están disponibles los octrees reales R32 y R64")
    return tmp_path_factory.mktemp("particion") / "objetivo2.json"


def _indices_muestra(raiz_resolucion: Path, manifest_path: Path):
    idx_train, idx_val, _, _ = cargar_o_crear_particion(
        raiz_resolucion,
        manifest_path,
        seed=42,
        val_split=0.10,
    )
    _, etiquetas, _ = listar_muestras_octree(raiz_resolucion, "train")
    return (
        seleccionar_subconjunto_balanceado(
            idx_train, etiquetas, N_TRAIN_MUESTRA, seed=42,
        ),
        seleccionar_subconjunto_balanceado(
            idx_val, etiquetas, N_VAL_MUESTRA, seed=42,
        ),
    )


@pytest.mark.parametrize("resolucion", [32, 64])
def test_integracion_densa_corta(resolucion, manifest_temporal, tmp_path):
    raiz_resolucion = DATA_ROOT / f"octrees_{resolucion}"
    idx_train, idx_val = _indices_muestra(
        raiz_resolucion, manifest_temporal,
    )
    torch.manual_seed(42)

    ds_train = DenseOctreeDataset(
        str(raiz_resolucion), resolucion, "train", idx_train,
    )
    ds_val = DenseOctreeDataset(
        str(raiz_resolucion), resolucion, "val", idx_val,
    )
    assert len(ds_train) == N_TRAIN_MUESTRA
    assert len(ds_val) == N_VAL_MUESTRA

    grid, etiqueta = ds_train[0]
    assert grid.shape == (4, resolucion, resolucion, resolucion)
    assert grid.dtype == torch.float32
    assert 0 <= int(etiqueta) < 40

    lote_train = torch.stack([ds_train[i][0] for i in range(len(ds_train))])
    etq_train = torch.stack([ds_train[i][1] for i in range(len(ds_train))])
    lote_val = torch.stack([ds_val[i][0] for i in range(len(ds_val))])

    device = torch.device("cpu")
    modelo, _ = crear_modelo_denso_referencia(
        resolucion=resolucion,
        num_clases=40,
        device=device,
    )
    assert isinstance(modelo, DenseTabla5Reference)

    logits_train = modelo(lote_train)
    assert logits_train.shape == (N_TRAIN_MUESTRA, 40)
    assert torch.isfinite(logits_train).all()

    modelo.eval()
    with torch.no_grad():
        logits_val = modelo(lote_val)
    assert logits_val.shape == (N_VAL_MUESTRA, 40)
    modelo.train()

    criterio = nn.CrossEntropyLoss()
    perdida = criterio(logits_train, etq_train)
    assert torch.isfinite(perdida)
    assert perdida.item() > 0

    optimizador = optim.Adam(modelo.parameters(), lr=1e-3)
    optimizador.zero_grad()
    perdida.backward()
    parametros = [p for p in modelo.parameters() if p.requires_grad]
    assert parametros
    assert all(p.grad is not None for p in parametros)
    assert sum(p.grad.norm().item() for p in parametros) > 0
    optimizador.step()

    ckpt_path = tmp_path / f"dense_tabla5_R{resolucion}.pth"
    torch.save({"model_state": modelo.state_dict()}, ckpt_path)
    modelo_recargado, _ = crear_modelo_denso_referencia(
        resolucion=resolucion,
        num_clases=40,
        device=device,
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    modelo_recargado.load_state_dict(checkpoint["model_state"])

    modelo.eval()
    modelo_recargado.eval()
    with torch.no_grad():
        logits_original = modelo(lote_train)
        logits_recargado = modelo_recargado(lote_train)
    assert torch.allclose(logits_original, logits_recargado, atol=1e-6)


def test_referencia_densa_tiene_capacidad_fija():
    """La topología de la Tabla 5 conserva el conteo de parámetros."""
    modelo_32 = DenseTabla5Reference(resolucion=32, num_clases=40)
    modelo_64 = DenseTabla5Reference(resolucion=64, num_clases=40)
    assert modelo_32.contar_parametros() == modelo_64.contar_parametros()


def test_constructor_oficial_crea_octnet_nativo():
    from net5_modelo import Net5Octree, crear_modelo

    modelo, _ = crear_modelo(resolucion=32, device=torch.device("cpu"))
    assert isinstance(modelo, Net5Octree)
    assert not any(isinstance(modulo, nn.Conv3d) for modulo in modelo.modules())
