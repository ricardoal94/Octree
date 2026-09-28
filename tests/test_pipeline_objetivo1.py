import json
import shutil
import sys
from pathlib import Path

from octree_real import OCTREE_FORMAT_VERSION, leer_metadatos_octree
from preprocesar_octrees import main, procesar_modelo, seleccionar_muestra_controlada


def test_muestra_controlada_es_balanceada_y_determinista():
    modelos = [
        {"model_id": f"{categoria}_{split}_{indice}",
         "categoria": categoria, "split": split}
        for split in ("train", "test")
        for categoria in ("airplane", "chair")
        for indice in range(3)
    ]
    muestra = seleccionar_muestra_controlada(modelos, 1)

    assert [modelo["model_id"] for modelo in muestra] == [
        "airplane_train_0", "chair_train_0",
        "airplane_test_0", "chair_test_0",
    ]


def test_pipeline_genera_npz_manifiesto_y_metricas(tmp_path: Path):
    fixture = Path(__file__).parent / "fixtures" / "tetrahedron.off"
    output_root = tmp_path / "data"
    tarea = {
        "ruta": str(fixture),
        "ruta_relativa": "chair/train/chair_control.off",
        "model_id": "chair_control",
        "categoria": "chair",
        "etiqueta": 8,
        "split": "train",
        "output_root": str(output_root),
        "manifest_root": str(output_root / "manifests"),
        "resoluciones": [32, 64],
        "n_puntos": 1000,
        "semilla_base": 42,
        "sobrescribir": False,
    }

    resultado = procesar_modelo(tarea)
    assert resultado["ok"] is True
    with Path(resultado["manifiesto"]).open(encoding="utf-8") as archivo:
        manifiesto = json.load(archivo)
    assert manifiesto["octree_format_version"] == OCTREE_FORMAT_VERSION
    assert set(manifiesto["salidas"]) == {"32", "64"}

    for resolucion, datos in manifiesto["salidas"].items():
        assert datos["equivalencia_antes_guardado"]["equivalente"] is True
        assert datos["equivalencia_despues_carga"]["equivalente"] is True
        assert len(datos["nodos_por_nivel"]) == (6 if resolucion == "32" else 7)
        assert datos["bytes_binarios_por_nodo"] == 18
        assert (
            datos["carga_binaria_sin_comprimir_bytes"]
            > datos["carga_binaria_nodos_sin_comprimir_bytes"]
        )
        ruta_npz = output_root / datos["archivo_npz"]
        assert ruta_npz.is_file()
        metadatos = leer_metadatos_octree(ruta_npz)
        assert metadatos["model_id"] == "chair_control"
        assert metadatos["resolucion"] == int(resolucion)


def test_main_admite_dataset_y_salida_fuera_del_proyecto(
    tmp_path: Path, monkeypatch,
):
    fixture = Path(__file__).parent / "fixtures" / "tetrahedron.off"
    dataset_root = tmp_path / "datos_externos" / "ModelNet40"
    modelo = dataset_root / "chair" / "train" / "chair_control.off"
    modelo.parent.mkdir(parents=True)
    shutil.copy2(fixture, modelo)

    output_root = tmp_path / "salidas_externas" / "octrees"
    resultados_dir = tmp_path / "resultados_externos" / "objetivo1"
    monkeypatch.setattr(sys, "argv", [
        "preprocesar_octrees.py",
        "--dataset-root", str(dataset_root),
        "--output-root", str(output_root),
        "--resultados-dir", str(resultados_dir),
        "--resoluciones", "32",
        "--splits", "train",
        "--categorias", "chair",
        "--limite", "1",
        "--n-puntos", "100",
        "--procesos", "1",
    ])

    assert main() == 0
    with (resultados_dir / "resumen_ejecucion.json").open(
        encoding="utf-8",
    ) as archivo:
        resumen = json.load(archivo)

    assert resumen["dataset_root"] == "ModelNet40"
    assert resumen["output_root"] == "octrees"
    assert resumen["n_modelos_exitosos"] == 1
    assert resumen["n_modelos_fallidos"] == 0
