import csv
import json

import pytest
from comparar_workers_octnet import (
    VARIABLES_HILOS_CPU,
    ErrorValidacionSmoke,
    construir_comando,
    construir_entorno_workers,
    escribir_consolidado,
    resolver_particion_manifest,
    ruta_resumen,
    validar_conjunto,
    validar_resumen,
)


def test_manifest_por_defecto_queda_fuera_del_repositorio(tmp_path):
    logs_dir = tmp_path / "artefactos" / "logs"

    assert resolver_particion_manifest(None, logs_dir) == (
        logs_dir / "particion_objetivo2_modelos.json"
    ).resolve()

    ruta_explicita = tmp_path / "particion_existente.json"
    assert resolver_particion_manifest(ruta_explicita, logs_dir) == (
        ruta_explicita.resolve()
    )


def test_entorno_workers_limita_hilos_y_conserva_otras_variables():
    entorno = construir_entorno_workers({
        "OPENBLAS_NUM_THREADS": "32",
        "VARIABLE_AJENA": "se_conserva",
    })

    assert entorno["VARIABLE_AJENA"] == "se_conserva"
    assert all(entorno[variable] == "1" for variable in VARIABLES_HILOS_CPU)


def _resumen_valido(resolucion=32, workers=4, commit="a" * 40):
    return {
        "schema_name": "net5-octree-native",
        "schema_version": "1.3.0",
        "backend_version": "1.1.1",
        "alcance": "PARCIAL_SMOKE",
        "valido_como_resultado_objetivo3": False,
        "backend": "octree_native",
        "entorno_ejecucion": {
            "sistema_operativo": {
                "sistema": "Windows",
                "release": "11",
                "version": "10.0.26100",
                "arquitectura": "AMD64",
            },
            "python": {
                "version": "3.14.0",
                "implementacion": "CPython",
            },
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
        },
        "git_commit": commit,
        "git_branch": "feature/octnet-native-backend",
        "git_dirty": False,
        "git_commit_publicado": True,
        "resolucion": resolucion,
        "parametros_entrenables": 8_547_456,
        "seed": 42,
        "particion": {
            "manifest": "logs/particion_objetivo2_modelos.json",
            "metodo": "sklearn.train_test_split(stratify=y, shuffle=True)",
            "n_train_usado": 40,
            "n_val_usado": 40,
            "n_test_usado": 40,
        },
        "configuracion": {
            "epochs_max": 1,
            "batch_size": 1,
            "num_workers": workers,
            "prefetch_factor": 1 if workers > 0 else None,
            "hilos_bibliotecas_cpu": {
                variable: "1" for variable in VARIABLES_HILOS_CPU
            },
            "persistent_workers": {
                "train": False,
                "val": False,
                "test": workers > 0,
            },
            "lotes_perfil": 3,
            "exigir_git_limpio": True,
            "exigir_git_publicado": True,
            "precalcular_planes": True,
        },
        "test_acc": 0.025,
        "tiempo_total_min": 1.0,
        "tiempo_inferencia_promedio_ms": 75.0,
        "tiempo_pipeline_promedio_ms": 900.0,
        "vram_pico_mb": 1800.0,
        "tamano_modelo_mb": 102.6,
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
            "detalle_backend": {
                "n_planes_convolucion": 9,
                "n_interacciones_convolucion": 370_368,
                "bytes_planes_convolucion": 42_000_000,
                "n_planes_pooling": 9,
                "n_asignaciones_pooling": 12_000,
                "bytes_planes_pooling": 2_000_000,
                "n_mapas_finales": 3,
                "bytes_mapas_finales": 100_000,
            },
        },
    }


def _validar(resumen, resolucion=32, workers=4):
    validar_resumen(
        resumen,
        resolucion=resolucion,
        workers=workers,
        limite_train=40,
        limite_val=40,
        limite_test=40,
        epochs=1,
        batch_size=1,
        lotes_perfil=3,
    )


def test_comando_usa_etiqueta_directorios_y_trazabilidad(tmp_path):
    comando = construir_comando(
        python="python",
        resolucion=64,
        workers=8,
        limite_train=40,
        limite_val=40,
        limite_test=40,
        epochs=1,
        batch_size=1,
        lotes_perfil=3,
        data_root=tmp_path / "data",
        particion_manifest=tmp_path / "particion.json",
        checkpoints_dir=tmp_path / "checkpoints",
        logs_dir=tmp_path / "logs",
        resultados_dir=tmp_path / "resultados",
    )

    assert comando[0] == "python"
    assert comando[comando.index("--resolucion") + 1] == "64"
    assert comando[comando.index("--num-workers") + 1] == "8"
    assert comando[comando.index("--tag") + 1] == "_smoke_w8"
    assert "--exigir-git-limpio" in comando
    assert "--exigir-git-publicado" in comando
    assert ruta_resumen(tmp_path, 64, 8).name.endswith("R64_smoke_w8.json")


def test_validador_acepta_un_resumen_completo():
    _validar(_resumen_valido())


@pytest.mark.parametrize(
    ("mutacion", "mensaje"),
    [
        (lambda r: r["configuracion"].update(num_workers=0), "num_workers"),
        (
            lambda r: r["configuracion"].update(exigir_git_publicado=False),
            "exigir_git_publicado",
        ),
        (
            lambda r: r["configuracion"].update(hilos_bibliotecas_cpu={}),
            "hilos_bibliotecas_cpu",
        ),
        (
            lambda r: r["configuracion"].update(persistent_workers={}),
            "persistent_workers",
        ),
        (
            lambda r: r["perfil_rendimiento"].update(
                memoria_planes_cpu_mb_por_muestra=0.0
            ),
            "memoria_planes_cpu",
        ),
        (
            lambda r: r["perfil_rendimiento"]["detalle_backend"].update(
                n_interacciones_convolucion=0
            ),
            "n_interacciones_convolucion",
        ),
        (
            lambda r: r.pop("entorno_ejecucion"),
            "entorno_ejecucion",
        ),
        (
            lambda r: r["entorno_ejecucion"]["hardware"].update(gpu=None),
            "hardware.gpu",
        ),
        (
            lambda r: r["entorno_ejecucion"].update(hardware=[]),
            "hardware debe ser un objeto",
        ),
    ],
)
def test_validador_rechaza_configuracion_o_perfil_incompleto(mutacion, mensaje):
    resumen = _resumen_valido()
    mutacion(resumen)

    with pytest.raises(ErrorValidacionSmoke, match=mensaje):
        _validar(resumen)


def test_conjunto_exige_mismo_commit_y_todas_las_combinaciones():
    ejecuciones = [
        _resumen_valido(resolucion, workers)
        for resolucion in (32, 64)
        for workers in (0, 4, 8)
    ]
    validar_conjunto(ejecuciones, resoluciones=[32, 64], workers=[0, 4, 8])

    ejecuciones[-1]["git_commit"] = "b" * 40
    with pytest.raises(ErrorValidacionSmoke, match="commit"):
        validar_conjunto(
            ejecuciones, resoluciones=[32, 64], workers=[0, 4, 8],
        )


def test_consolidado_es_ordenado_y_auditable(tmp_path):
    ejecuciones = [
        _resumen_valido(64, 4),
        _resumen_valido(32, 8),
        _resumen_valido(32, 0),
    ]
    ruta_json, ruta_csv = escribir_consolidado(
        ejecuciones,
        salida_dir=tmp_path,
        resoluciones=[32, 64],
        workers=[0, 4, 8],
    )

    consolidado = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert consolidado["schema_name"] == "net5-octree-worker-benchmark"
    assert consolidado["schema_version"] == "1.2.0"
    assert consolidado["estado"] == "VALIDADO"
    assert consolidado["entorno_ejecucion"]["hardware"]["gpu"]["nombre"] == (
        "NVIDIA GeForce RTX 5070"
    )
    assert consolidado["seleccion_workers"] == {"32": 0, "64": 4}
    assert [
        (fila["resolucion"], fila["num_workers"])
        for fila in consolidado["resultados"]
    ] == [(32, 0), (32, 8), (64, 4)]

    with ruta_csv.open(encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    assert len(filas) == 3
    assert filas[0]["resolucion"] == "32"
    assert filas[0]["num_workers"] == "0"
    assert filas[0]["gpu_nombre"] == "NVIDIA GeForce RTX 5070"


def test_conjunto_exige_el_mismo_entorno_en_todas_las_corridas():
    ejecuciones = [
        _resumen_valido(resolucion, workers)
        for resolucion in (32, 64)
        for workers in (0, 4, 8)
    ]
    ejecuciones[-1]["entorno_ejecucion"]["python"]["version"] = "3.14.1"

    with pytest.raises(ErrorValidacionSmoke, match="entorno_ejecucion"):
        validar_conjunto(
            ejecuciones, resoluciones=[32, 64], workers=[0, 4, 8],
        )
