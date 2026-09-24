import csv
import json

import pytest
from particion_objetivo2 import CLASES_MODELNET40
from validar_resultados_net5 import (
    cargar_json_estricto,
    escribir_consolidado,
    validar_archivos,
    validar_conjunto_oficial,
    validar_resumen_oficial,
)


def _entorno():
    return {
        "sistema_operativo": {
            "sistema": "Windows",
            "release": "11",
            "version": "10.0.26100",
            "arquitectura": "AMD64",
        },
        "python": {"version": "3.12.7", "implementacion": "CPython"},
        "pytorch": {
            "version": "2.9.0+cu128",
            "cuda_compilacion": "12.8",
            "cudnn_version": 91002,
        },
        "hardware": {
            "cpu_modelo": "CPU de prueba",
            "cpu_nucleos_logicos": 16,
            "ram_total_bytes": 34_359_738_368,
            "ram_total_gib": 32.0,
            "dispositivo": "cuda",
            "gpu": {
                "nombre": "NVIDIA GeForce RTX 5070",
                "indice": 0,
                "cantidad_dispositivos": 1,
                "capacidad_cuda": "12.0",
                "vram_total_bytes": 12_884_901_888,
                "vram_total_gib": 12.0,
            },
        },
    }


def _matriz_y_reporte():
    soportes = [62] * 28 + [61] * 12
    matriz = [[0] * 40 for _ in range(40)]
    reporte = {}
    for indice, (clase, soporte) in enumerate(zip(CLASES_MODELNET40, soportes)):
        matriz[indice][indice] = soporte
        reporte[clase] = {
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
            "support": float(soporte),
        }
    reporte["accuracy"] = 1.0
    for promedio in ("macro avg", "weighted avg"):
        reporte[promedio] = {
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
            "support": 2468.0,
        }
    return matriz, reporte


def _resumen_oficial(resolucion=32, workers=8, commit="a" * 40):
    matriz, reporte = _matriz_y_reporte()
    return {
        "schema_name": "net5-octree-native",
        "schema_version": "1.4.0",
        "backend_version": "1.1.1",
        "alcance": "COMPLETO",
        "valido_como_resultado_objetivo3": True,
        "motivo_no_valido": None,
        "backend": "octree_native",
        "entorno_ejecucion": _entorno(),
        "git_commit": commit,
        "git_branch": "feature/octnet-native-backend",
        "git_dirty": False,
        "git_commit_publicado": True,
        "resolucion": resolucion,
        "profundidad_octree": 5 if resolucion == 32 else 6,
        "parametros_entrenables": 8_547_456,
        "seed": 42,
        "particion": {
            "manifest": "particion_objetivo2_modelos.json",
            "raiz_datos": f"octrees_{resolucion}",
            "metodo": "sklearn.train_test_split(stratify=y, shuffle=True)",
            "n_train_total": 9843,
            "n_train_usado": 8858,
            "n_val_usado": 985,
            "n_test_usado": 2468,
        },
        "configuracion": {
            "epochs_max": 200,
            "patience": 20,
            "batch_size": 1,
            "num_workers": workers,
            "prefetch_factor": 1,
            "hilos_bibliotecas_cpu": {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            "persistent_workers": {
                "train": False,
                "val": False,
                "test": True,
            },
            "lr": 0.001,
            "weight_decay": 1e-4,
            "scheduler": "StepLR(step_size=20, gamma=0.7)",
            "lotes_perfil": 3,
            "exigir_git_limpio": True,
            "exigir_git_publicado": True,
            "precalcular_planes": True,
        },
        "entrenamiento": {
            "reanudar_solicitado": True,
            "reanudado_desde_epoca": 12,
            "epocas_ejecutadas_esta_invocacion": 18,
            "epocas_completadas": 30,
            "detenido_por_early_stopping": True,
            "checkpoint_mejor": f"net5_octree_mejor_R{resolucion}.pth",
            "checkpoint_ultimo": f"net5_octree_ultimo_R{resolucion}.pth",
            "historial_csv": f"net5_octree_historial_R{resolucion}.csv",
            "historial_json": f"net5_octree_historial_R{resolucion}.json",
            "particion_sha256": "c" * 64,
        },
        "mejor_val_acc": 1.0,
        "mejor_epoca": 10,
        "test_acc": 1.0,
        "test_loss": 0.01,
        "tiempo_total_min": 300.0,
        "tamano_modelo_mb": 102.6,
        "vram_pico_mb": 1800.0,
        "perfil_rendimiento": {
            "lotes_medidos": 3,
            "muestras_medidas": 3,
            "carga_lote_ms_por_muestra": 800.0,
            "transferencia_atributos_ms_por_muestra": 0.2,
            "forward_total_ms_por_muestra": 74.0,
            "planes_backend_ms_por_muestra": 8.0,
            "resto_forward_ms_por_muestra": 66.0,
            "memoria_planes_cpu_mb_por_muestra": 48.0,
            "interacciones_convolucion_por_muestra": 123_456.0,
            "detalle_backend": {},
        },
        "matriz_confusion": matriz,
        "reporte_clasificacion": reporte,
        "tiempo_inferencia_total_s": 185.0,
        "tiempo_inferencia_promedio_ms": 75.0,
        "tiempo_pipeline_total_s": 2221.0,
        "tiempo_pipeline_promedio_ms": 900.0,
        "n_muestras_test": 2468,
        "repeticiones_inferencia": 3,
        "protocolo_tiempo": "solo forward",
        "protocolo_tiempo_pipeline": "extremo a extremo",
    }


def _validar(resumen, resolucion=32, workers=8):
    return validar_resumen_oficial(
        resumen,
        resolucion,
        workers=workers,
        epochs=200,
        patience=20,
        batch_size=1,
        lr=0.001,
        lotes_perfil=3,
    )


def test_resumen_oficial_completo_es_valido():
    assert _validar(_resumen_oficial()) == []


@pytest.mark.parametrize(
    ("mutacion", "mensaje"),
    [
        (lambda r: r.update(alcance="PARCIAL_SMOKE"), "alcance"),
        (lambda r: r["particion"].update(n_test_usado=40), "n_test_usado"),
        (lambda r: r["configuracion"].update(num_workers=4), "num_workers"),
        (lambda r: r["matriz_confusion"][0].__setitem__(0, 61), "suma 2467"),
        (
            lambda r: r["entrenamiento"].update(
                epocas_completadas=30, detenido_por_early_stopping=False,
            ),
            "early stopping",
        ),
    ],
)
def test_validador_detecta_resultados_no_oficiales(mutacion, mensaje):
    resumen = _resumen_oficial()
    mutacion(resumen)
    assert any(mensaje in error for error in _validar(resumen))


def test_conjunto_exige_mismo_commit_entorno_y_barrido():
    resumenes = {32: _resumen_oficial(32), 64: _resumen_oficial(64)}
    barrido = {
        "schema_name": "net5-octree-worker-benchmark",
        "schema_version": "1.2.0",
        "estado": "VALIDADO",
        "git_commit": "a" * 40,
        "entorno_ejecucion": _entorno(),
        "seleccion_workers": {"32": 8, "64": 8},
    }
    assert validar_conjunto_oficial(resumenes, barrido=barrido) == []

    resumenes[64]["git_commit"] = "b" * 40
    errores = validar_conjunto_oficial(resumenes, barrido=barrido)
    assert any("git_commit" in error for error in errores)


def test_archivos_y_consolidado_generan_evidencia(tmp_path):
    resultados = tmp_path / "resultados"
    resultados.mkdir()
    for resolucion in (32, 64):
        (resultados / f"resumen_net5_octree_R{resolucion}.json").write_text(
            json.dumps(_resumen_oficial(resolucion)), encoding="utf-8",
        )
    resumenes, validaciones, errores = validar_archivos(
        resultados,
        workers_por_resolucion={32: 8, 64: 8},
    )
    ruta_json, ruta_csv, valido = escribir_consolidado(
        resumenes,
        validaciones,
        errores,
        salida_dir=resultados,
        barrido=None,
    )

    assert valido is True
    informe = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert informe["estado"] == "VALIDADO"
    with ruta_csv.open(encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    assert [fila["resolucion"] for fila in filas] == ["32", "64"]


def test_consolidado_invalido_tolera_resumen_malformado(tmp_path):
    validaciones = [
        {
            "resolucion": 32,
            "archivo": "resumen_net5_octree_R32.json",
            "valido": False,
            "errores": ["entorno_ejecucion no es objeto"],
        },
        {
            "resolucion": 64,
            "archivo": "resumen_net5_octree_R64.json",
            "valido": False,
            "errores": ["archivo inexistente"],
        },
    ]

    ruta_json, ruta_csv, valido = escribir_consolidado(
        {32: {"resolucion": 32}},
        validaciones,
        ["deben existir exactamente los resultados R32 y R64"],
        salida_dir=tmp_path,
        barrido=None,
    )

    assert valido is False
    informe = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert informe["estado"] == "INVALIDO"
    assert informe["resultados"] == []
    with ruta_csv.open(encoding="utf-8", newline="") as archivo:
        assert list(csv.DictReader(archivo)) == []


def test_json_estricto_rechaza_nan(tmp_path):
    ruta = tmp_path / "resultado.json"
    ruta.write_text('{"test_acc": NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="Constante JSON no valida"):
        cargar_json_estricto(ruta)
