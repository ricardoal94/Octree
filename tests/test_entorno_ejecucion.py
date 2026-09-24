import json

import torch
from entorno_ejecucion import capturar_entorno_ejecucion


def test_captura_entorno_cpu_es_completa_y_serializable():
    entorno = capturar_entorno_ejecucion(torch.device("cpu"))

    assert entorno["sistema_operativo"]["sistema"]
    assert entorno["sistema_operativo"]["arquitectura"]
    assert entorno["python"]["version"]
    assert entorno["pytorch"]["version"] == str(torch.__version__)
    assert entorno["hardware"]["cpu_modelo"]
    assert entorno["hardware"]["cpu_nucleos_logicos"] > 0
    assert entorno["hardware"]["ram_total_bytes"] > 0
    assert entorno["hardware"]["ram_total_gib"] > 0
    assert entorno["hardware"]["dispositivo"] == "cpu"
    assert entorno["hardware"]["gpu"] is None
    json.dumps(entorno)
