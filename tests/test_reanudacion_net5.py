import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fase3_net5_entrenamiento import (
    _capturar_estado_rng,
    _cargar_checkpoint,
    _guardar_checkpoint_atomico,
    _restaurar_estado_rng,
    _validar_contrato_reanudacion,
)


class _LoaderFalso:
    def __init__(self):
        self.generator = torch.Generator()
        self.generator.manual_seed(42)


def test_contrato_reanudacion_rechaza_cambios():
    contrato = {"git_commit": "a" * 40, "resolucion": 32, "epochs": 200}
    _validar_contrato_reanudacion(dict(contrato), contrato)

    modificado = dict(contrato, resolucion=64)
    with pytest.raises(ValueError, match="resolucion"):
        _validar_contrato_reanudacion(modificado, contrato)


def test_estado_aleatorio_se_restaura_en_limite_de_epoca():
    loader = _LoaderFalso()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    estado = _capturar_estado_rng(loader)
    esperado = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(1).item()),
        torch.rand(1, generator=loader.generator).item(),
    )

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    loader.generator.manual_seed(7)
    _restaurar_estado_rng(estado, loader)
    obtenido = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(1).item()),
        torch.rand(1, generator=loader.generator).item(),
    )

    assert obtenido == pytest.approx(esperado)


def test_checkpoint_atomico_puede_recargarse(tmp_path):
    ruta = tmp_path / "ultimo.pth"
    datos = {"epoca": 3, "model_state": {"peso": torch.tensor([1.0])}}

    _guardar_checkpoint_atomico(datos, ruta)
    recuperado = _cargar_checkpoint(ruta, torch.device("cpu"))

    assert recuperado["epoca"] == 3
    assert torch.equal(recuperado["model_state"]["peso"], torch.tensor([1.0]))
    assert not (tmp_path / "ultimo.pth.tmp").exists()
