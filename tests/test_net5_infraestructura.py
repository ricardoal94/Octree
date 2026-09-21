"""Pruebas sin PyTorch para la infraestructura compartida del Objetivo 3."""

from collections import Counter

import numpy as np

from fase3_hce_entrenamiento import particionar_train_val
from particion_objetivo2 import (
    cargar_o_crear_particion,
    crear_manifest_particion,
    listar_muestras_octree,
    seleccionar_subconjunto_balanceado,
)
from preprocesar_octrees import CLASES_MODELNET40


def _crear_dataset_falso(raiz):
    clases = CLASES_MODELNET40[:4]
    for resolucion in (32, 64):
        raiz_resolucion = raiz / f"octrees_{resolucion}"
        for clase in clases:
            train = raiz_resolucion / clase / "train"
            test = raiz_resolucion / clase / "test"
            train.mkdir(parents=True)
            test.mkdir(parents=True)
            for i in range(10):
                (train / f"{clase}_{i:04d}.npz").touch()
            for i in range(2):
                (test / f"{clase}_{i:04d}.npz").touch()
    return clases


def test_manifest_reproduce_particion_hce_y_es_portable(tmp_path):
    clases = _crear_dataset_falso(tmp_path)
    raiz = tmp_path / "octrees_32"
    manifest = crear_manifest_particion(raiz, seed=42, val_split=0.10)

    X = np.arange(40, dtype=np.int64).reshape(-1, 1)
    y = np.repeat(np.arange(4, dtype=np.int64), 10)
    X_train, _, X_val, _ = particionar_train_val(
        X, y, val_split=0.10, seed=42,
    )
    posiciones = {
        f"{clase}/{clase}_{i:04d}": clase_idx * 10 + i
        for clase_idx, clase in enumerate(clases)
        for i in range(10)
    }

    assert [posiciones[x] for x in manifest["train_ids"]] == X_train[:, 0].tolist()
    assert [posiciones[x] for x in manifest["val_ids"]] == X_val[:, 0].tolist()
    assert manifest["n_train"] == 36
    assert manifest["n_val"] == 4
    assert manifest["n_test"] == 8
    assert Counter(x.split("/")[0] for x in manifest["val_ids"]) == {
        clase: 1 for clase in clases
    }
    assert str(tmp_path) not in repr(manifest)


def test_un_manifest_mapea_los_mismos_objetos_en_r32_y_r64(tmp_path):
    clases = _crear_dataset_falso(tmp_path)
    ruta_manifest = tmp_path / "logs" / "particion.json"

    idx_train_32, idx_val_32, idx_test_32, manifest_32 = cargar_o_crear_particion(
        tmp_path / "octrees_32", ruta_manifest, seed=42, val_split=0.10,
    )
    idx_train_64, idx_val_64, idx_test_64, manifest_64 = cargar_o_crear_particion(
        tmp_path / "octrees_64", ruta_manifest, seed=42, val_split=0.10,
    )

    assert manifest_32 == manifest_64
    assert np.array_equal(idx_train_32, idx_train_64)
    assert np.array_equal(idx_val_32, idx_val_64)
    assert np.array_equal(idx_test_32, idx_test_64)

    _, etiquetas, ids = listar_muestras_octree(tmp_path / "octrees_32", "train")
    muestra = seleccionar_subconjunto_balanceado(
        idx_train_32, etiquetas, limite=4, seed=42,
    )
    assert {ids[i].split("/")[0] for i in muestra} == set(clases)
