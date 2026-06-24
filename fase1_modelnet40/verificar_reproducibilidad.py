"""
verificar_reproducibilidad.py
=====================================================
Herramienta de control: ejecuta dos veces la particion
con la misma seed y confirma que los resultados son
identicos. Si los hashes coinciden, la reproducibilidad
esta garantizada.
"""

import hashlib
import json
import numpy as np
from fase1_setup import set_global_seed, particionar_dataset

SEED = 42


def hash_particion(particion: dict) -> str:
    """Calcula SHA-256 de los indices de particion."""
    datos = json.dumps(
        {k: v["indices"] for k, v in particion.items() if "indices" in v},
        sort_keys=True,
    )
    return hashlib.sha256(datos.encode()).hexdigest()


def verificar_reproducibilidad(n_ejecuciones: int = 3) -> bool:
    """
    Ejecuta la particion N veces y verifica que el hash sea identico.
    Retorna True si todas las ejecuciones producen el mismo resultado.
    """
    print(f"\n[Verificacion] Probando reproducibilidad ({n_ejecuciones} ejecuciones)...\n")

    hashes = []
    for i in range(n_ejecuciones):
        set_global_seed(SEED)
        particion = particionar_dataset(seed=SEED)
        h = hash_particion(particion)
        hashes.append(h)
        print(f"  Ejecucion {i+1}: SHA-256 = {h[:16]}...{h[-8:]}")

    todos_iguales = len(set(hashes)) == 1

    print(
        f"\n  Resultado: {'REPRODUCIBLE' if todos_iguales else 'NO REPRODUCIBLE'}"
    )
    if todos_iguales:
        print("  Todas las particiones son identicas. Control de estocasticidad OK.")
    else:
        print("  ADVERTENCIA: Se detectaron diferencias entre ejecuciones.")

    return todos_iguales


def verificar_sin_fuga_datos(particion: dict) -> dict:
    """
    Verifica que no haya data leakage entre los conjuntos.
    Retorna un resumen de la verificacion.
    """
    idx_train = set(particion["train"]["indices"])
    idx_val   = set(particion["val"]["indices"])
    idx_test  = set(particion["test"]["indices"])

    resultados = {
        "train_val_overlap":   len(idx_train & idx_val),
        "train_test_overlap":  len(idx_train & idx_test),
        "val_test_overlap":    len(idx_val & idx_test),
        "total_train":         len(idx_train),
        "total_val":           len(idx_val),
        "total_test":          len(idx_test),
    }

    print("\n[Data Leakage] Verificacion de solapamiento entre conjuntos:")
    print(f"  Train ∩ Val  : {resultados['train_val_overlap']} muestras")
    print(f"  Train ∩ Test : {resultados['train_test_overlap']} muestras")
    print(f"  Val   ∩ Test : {resultados['val_test_overlap']} muestras")

    sin_fuga = all(v == 0 for v in [
        resultados["train_val_overlap"],
        resultados["train_test_overlap"],
        resultados["val_test_overlap"],
    ])
    print(f"\n  Estado: {'SIN DATA LEAKAGE' if sin_fuga else 'ADVERTENCIA: HAY FUGA DE DATOS'}")

    return resultados


if __name__ == "__main__":
    print("=" * 60)
    print("  VERIFICACION DE REPRODUCIBILIDAD - FASE 1")
    print("=" * 60)

    # Test de reproducibilidad
    reproducible = verificar_reproducibilidad(n_ejecuciones=3)

    # Test de data leakage con una ejecucion de referencia
    set_global_seed(SEED)
    particion_ref = particionar_dataset(seed=SEED)
    resultados_fuga = verificar_sin_fuga_datos(particion_ref)

    # Resumen final
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    print(f"  Reproducibilidad garantizada : {'SI' if reproducible else 'NO'}")
    sin_fuga = all(v == 0 for v in [
        resultados_fuga["train_val_overlap"],
        resultados_fuga["train_test_overlap"],
        resultados_fuga["val_test_overlap"],
    ])
    print(f"  Sin data leakage             : {'SI' if sin_fuga else 'NO'}")
    print(f"  Train / Val / Test           : "
          f"{resultados_fuga['total_train']} / "
          f"{resultados_fuga['total_val']} / "
          f"{resultados_fuga['total_test']}")
    print("=" * 60)
