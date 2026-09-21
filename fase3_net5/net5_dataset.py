"""Cargador provisional para la referencia densa de la Tabla 5.

Este modulo materializa el octree en un tensor ``(4, R, R, R)`` y, por ello,
NO constituye el backend OctNet requerido por el objetivo 3. Se conserva
unicamente para pruebas diagnosticas de la topologia de red. El backend
definitivo debe consumir la jerarquia del octree sin expandirla a ``R^3``.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from functools import partial
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "fase2_octree"))
from octree_real import cargar_octree_disperso
from particion_objetivo2 import CLASES_MODELNET40, listar_muestras_octree


CLASES = CLASES_MODELNET40


def _materializar_grid_denso(centros: np.ndarray, normales: np.ndarray,
                             resolucion: int) -> np.ndarray:
    """
    Construye un grid denso (4, R, R, R) a partir de los centros y
    normales de las hojas ocupadas. Se ejecuta en __getitem__, es decir,
    en cada carga de muestra -- no se persiste en disco.
    """
    grid = np.zeros((4, resolucion, resolucion, resolucion), dtype=np.float32)
    if len(centros) == 0:
        return grid
    idx = np.clip(((centros + 1.0) * 0.5 * resolucion).astype(np.int64),
                 0, resolucion - 1)
    grid[0, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    grid[1, idx[:, 0], idx[:, 1], idx[:, 2]] = normales[:, 0]
    grid[2, idx[:, 0], idx[:, 1], idx[:, 2]] = normales[:, 1]
    grid[3, idx[:, 0], idx[:, 1], idx[:, 2]] = normales[:, 2]
    return grid


class DenseOctreeDataset(Dataset):
    """
    Carga octrees dispersos y los materializa para la referencia densa.

    Parametros
    ----------
    raiz_resolucion : carpeta data/octrees_<R>/
    resolucion      : R (32 o 64) -- necesario para saber a que tamaño
                      materializar el grid denso desde las hojas dispersas
    split           : 'train', 'val' o 'test'
    idx_subset      : indices para sub-particion train/val (opcional)
    """

    def __init__(self, raiz_resolucion: str, resolucion: int,
                split: str = "train", idx_subset: np.ndarray = None):
        super().__init__()
        self.resolucion = resolucion
        carpeta_split = "train" if split in ("train", "val") else "test"

        rutas, etiquetas, _ = listar_muestras_octree(
            raiz_resolucion, carpeta_split,
        )
        self.muestras = rutas
        self.etiquetas = etiquetas

        if idx_subset is not None:
            indices = np.asarray(idx_subset, dtype=np.int64)
            if indices.size and (indices.min() < 0 or indices.max() >= len(rutas)):
                raise IndexError("La particion contiene indices fuera del dataset")
            self.muestras = [rutas[i] for i in indices]
            self.etiquetas = etiquetas[indices]

        print(f"[Dataset] Octree disperso '{split}': {len(self.muestras)} muestras "
             f"(materializacion a {resolucion}^3 al vuelo)")

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, idx: int):
        d = cargar_octree_disperso(str(self.muestras[idx]))
        etiqueta_esperada = int(self.etiquetas[idx])
        if d["etiqueta"] != etiqueta_esperada:
            raise ValueError("La etiqueta del NPZ no coincide con su carpeta")
        grid_np = _materializar_grid_denso(
            d["centros_hoja"], d["normales_hoja"], self.resolucion,
        )
        grid = torch.from_numpy(grid_np).float()    # (4, R, R, R)
        etiqueta = torch.tensor(d["etiqueta"], dtype=torch.long)
        return grid, etiqueta


def _worker_init_fn(worker_id: int, seed: int = 42) -> None:
    np.random.seed(seed + worker_id)


def crear_dataloaders_densos_referencia(
    raiz_resolucion: str,
    resolucion: int,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray = None,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple:
    """Crea DataLoaders para la referencia densa, no para OctNet.

    idx_test=None (por defecto) usa el conjunto de prueba oficial completo;
    se puede pasar un subconjunto de indices para pruebas rapidas (smoke
    tests) de la propia rutina de entrenamiento."""

    ds_train = DenseOctreeDataset(raiz_resolucion, resolucion, "train", idx_train)
    ds_val = DenseOctreeDataset(raiz_resolucion, resolucion, "val", idx_val)
    ds_test = DenseOctreeDataset(raiz_resolucion, resolucion, "test", idx_test)

    g = torch.Generator()
    g.manual_seed(seed)
    init_fn = partial(_worker_init_fn, seed=seed)

    loader_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=init_fn, generator=g,
    )
    loader_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    loader_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return loader_train, loader_val, loader_test
