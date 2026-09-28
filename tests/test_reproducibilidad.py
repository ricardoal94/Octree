import numpy as np

from octree_real import _serializar_dfs, construir_octree
from validacion_objetivo1 import ruta_portable, semilla_estable_modelo


def test_semilla_no_depende_del_sistema_operativo():
    semilla_a = semilla_estable_modelo(42, "chair/train/chair_0001.off")
    semilla_b = semilla_estable_modelo(42, r"chair\train\chair_0001.off")
    assert semilla_a == semilla_b


def test_construccion_repetida_es_identica():
    rng = np.random.default_rng(42)
    puntos = rng.uniform(-1, 1, size=(1000, 3)).astype(np.float32)
    normales = rng.normal(size=(1000, 3)).astype(np.float32)
    normales /= np.clip(np.linalg.norm(normales, axis=1, keepdims=True), 1e-12, None)

    serializaciones = []
    for _ in range(2):
        raiz = construir_octree(puntos, normales, profundidad_max=5)
        serializaciones.append(_serializar_dfs(raiz))

    for array_a, array_b in zip(*serializaciones):
        np.testing.assert_array_equal(array_a, array_b)


def test_ruta_portable_admite_ubicaciones_externas(tmp_path):
    raiz_proyecto = tmp_path / "repositorio"
    ruta_interna = raiz_proyecto / "Dataset" / "ModelNet40"
    ruta_externa = tmp_path / "datos_externos" / "ModelNet40"

    assert ruta_portable(ruta_interna, raiz_proyecto) == "Dataset/ModelNet40"
    assert ruta_portable(ruta_externa, raiz_proyecto) == "ModelNet40"
