"""Particion reproducible compartida por HCE y el enfoque profundo.

El objetivo 2 separa el ``train`` oficial de ModelNet40 con
``train_test_split(..., stratify=y, random_state=42)``. Este modulo reproduce
exactamente esa operacion sobre el orden canonico de los NPZ y persiste los
identificadores de modelo, no indices posicionales dependientes del equipo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO / "fase2_octree"))

from preprocesar_octrees import CLASES_MODELNET40  # noqa: E402


SCHEMA_NAME = "modelnet40-objective2-split"
SCHEMA_VERSION = "1.0.0"


def _validar_manifest(manifest: dict) -> None:
    requeridos = {
        "schema_name", "schema_version", "seed", "val_split", "metodo",
        "orden_clases", "n_train_total", "n_train", "n_val", "n_test",
        "train_ids", "val_ids", "test_ids",
    }
    faltantes = sorted(requeridos - manifest.keys())
    if faltantes:
        raise ValueError(
            "Faltan campos en el manifiesto: " + ", ".join(faltantes)
        )
    if manifest["schema_name"] != SCHEMA_NAME:
        raise ValueError("El archivo no es un manifiesto de particion reconocido")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Version incompatible del manifiesto de particion")
    if manifest["orden_clases"] != list(CLASES_MODELNET40):
        raise ValueError("El orden de clases del manifiesto es incompatible")

    for clave in ("train_ids", "val_ids", "test_ids"):
        ids = manifest[clave]
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            raise ValueError(f"{clave} debe ser una lista de identificadores")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{clave} contiene identificadores duplicados")

    if set(manifest["train_ids"]) & set(manifest["val_ids"]):
        raise ValueError("Train y validacion se solapan en el manifiesto")
    conteos = {
        "n_train": len(manifest["train_ids"]),
        "n_val": len(manifest["val_ids"]),
        "n_test": len(manifest["test_ids"]),
    }
    for clave, observado in conteos.items():
        if manifest[clave] != observado:
            raise ValueError(f"Conteo inconsistente en {clave}")
    if manifest["n_train_total"] != conteos["n_train"] + conteos["n_val"]:
        raise ValueError("Conteo inconsistente en n_train_total")


def listar_muestras_octree(
    raiz_resolucion: str | Path,
    split: str,
) -> tuple[list[Path], np.ndarray, list[str]]:
    """Enumera NPZ en el mismo orden canonico utilizado por HCE."""
    if split not in {"train", "test"}:
        raise ValueError("split debe ser 'train' o 'test'")

    raiz = Path(raiz_resolucion)
    rutas: list[Path] = []
    etiquetas: list[int] = []
    identificadores: list[str] = []

    for etiqueta, clase in enumerate(CLASES_MODELNET40):
        carpeta = raiz / clase / split
        if not carpeta.exists():
            continue
        for ruta in sorted(carpeta.glob("*.npz")):
            rutas.append(ruta)
            etiquetas.append(etiqueta)
            identificadores.append(f"{clase}/{ruta.stem}")

    if len(set(identificadores)) != len(identificadores):
        raise ValueError("Se encontraron identificadores de modelo duplicados")

    return rutas, np.asarray(etiquetas, dtype=np.int64), identificadores


def crear_manifest_particion(
    raiz_resolucion: str | Path,
    seed: int = 42,
    val_split: float = 0.10,
) -> dict:
    """Crea la misma particion estratificada y determinista del objetivo 2."""
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split debe estar entre 0 y 1")

    _, y_train_total, ids_train_total = listar_muestras_octree(
        raiz_resolucion, "train",
    )
    _, _, ids_test = listar_muestras_octree(raiz_resolucion, "test")
    if not ids_train_total or not ids_test:
        raise ValueError(
            "No se encontraron los octrees de train y test en la raiz indicada"
        )

    indices = np.arange(len(ids_train_total), dtype=np.int64)
    idx_train, idx_val = train_test_split(
        indices,
        test_size=val_split,
        random_state=seed,
        shuffle=True,
        stratify=y_train_total,
    )

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "val_split": float(val_split),
        "metodo": "sklearn.train_test_split(stratify=y, shuffle=True)",
        "orden_clases": list(CLASES_MODELNET40),
        "n_train_total": len(ids_train_total),
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "n_test": len(ids_test),
        "train_ids": [ids_train_total[i] for i in idx_train],
        "val_ids": [ids_train_total[i] for i in idx_val],
        "test_ids": ids_test,
    }


def guardar_manifest_particion(manifest: dict, ruta: str | Path) -> None:
    _validar_manifest(manifest)
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8", newline="\n") as archivo:
        json.dump(manifest, archivo, indent=2, ensure_ascii=False)
        archivo.write("\n")


def cargar_manifest_particion(ruta: str | Path) -> dict:
    with Path(ruta).open("r", encoding="utf-8") as archivo:
        manifest = json.load(archivo)
    _validar_manifest(manifest)
    return manifest


def _indices_desde_ids(ids_actuales: list[str], ids_particion: list[str]) -> np.ndarray:
    posicion = {modelo_id: i for i, modelo_id in enumerate(ids_actuales)}
    faltantes = [modelo_id for modelo_id in ids_particion if modelo_id not in posicion]
    if faltantes:
        muestra = ", ".join(faltantes[:5])
        raise ValueError(f"Faltan modelos de la particion en los octrees: {muestra}")
    return np.asarray([posicion[modelo_id] for modelo_id in ids_particion], dtype=np.int64)


def cargar_o_crear_particion(
    raiz_resolucion: str | Path,
    ruta_manifest: str | Path,
    seed: int = 42,
    val_split: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Carga la particion por ID o la crea una unica vez de forma reproducible."""
    ruta_manifest = Path(ruta_manifest)
    if ruta_manifest.exists():
        manifest = cargar_manifest_particion(ruta_manifest)
        if manifest["seed"] != seed or manifest["val_split"] != val_split:
            raise ValueError("Semilla o proporcion de validacion incompatibles")
    else:
        manifest = crear_manifest_particion(raiz_resolucion, seed, val_split)
        guardar_manifest_particion(manifest, ruta_manifest)

    _, _, ids_train_actuales = listar_muestras_octree(raiz_resolucion, "train")
    _, _, ids_test_actuales = listar_muestras_octree(raiz_resolucion, "test")

    esperados_train = set(manifest["train_ids"]) | set(manifest["val_ids"])
    if set(ids_train_actuales) != esperados_train:
        raise ValueError("Los modelos train actuales no coinciden con el manifiesto")
    if set(ids_test_actuales) != set(manifest["test_ids"]):
        raise ValueError("Los modelos test actuales no coinciden con el manifiesto")

    idx_train = _indices_desde_ids(ids_train_actuales, manifest["train_ids"])
    idx_val = _indices_desde_ids(ids_train_actuales, manifest["val_ids"])
    idx_test = _indices_desde_ids(ids_test_actuales, manifest["test_ids"])
    return idx_train, idx_val, idx_test, manifest


def seleccionar_subconjunto_balanceado(
    indices: np.ndarray,
    etiquetas_globales: np.ndarray,
    limite: int | None,
    seed: int = 42,
) -> np.ndarray:
    """Selecciona una muestra diagnostica con la mayor cobertura de clases posible."""
    indices = np.asarray(indices, dtype=np.int64)
    if limite is None or limite >= len(indices):
        return indices
    if limite <= 0:
        raise ValueError("El limite de muestras debe ser positivo")

    rng = np.random.default_rng(seed)
    por_clase: dict[int, list[int]] = {}
    for indice in indices:
        por_clase.setdefault(int(etiquetas_globales[indice]), []).append(int(indice))
    for valores in por_clase.values():
        rng.shuffle(valores)

    seleccion: list[int] = []
    clases = sorted(por_clase)
    ronda = 0
    while len(seleccion) < limite:
        agregados = 0
        for clase in clases:
            valores = por_clase[clase]
            if ronda < len(valores):
                seleccion.append(valores[ronda])
                agregados += 1
                if len(seleccion) == limite:
                    break
        if agregados == 0:
            break
        ronda += 1
    return np.asarray(seleccion, dtype=np.int64)
