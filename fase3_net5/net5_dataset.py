"""
net5_dataset.py - Fase 3 (Enfoque profundo)
=============================================
Dataset PyTorch que carga los octrees REALES en formato DISPERSO (.npz
con solo hojas ocupadas, ver octree_real.py y preprocesar_octrees.py) y
los MATERIALIZA a un grid denso (4, R, R, R) unicamente en memoria RAM,
en el momento de entregar cada muestra al DataLoader.

El archivo en disco permanece disperso en todo momento (proporcional al
numero de hojas ocupadas, tipicamente 1-3% de R^3). Solo se construye
un tensor denso transitorio porque Conv3d/ConvTranspose3d de PyTorch
requieren tensores densos -- esto es una materializacion "justo a
tiempo" (just-in-time), no un cambio en el formato de almacenamiento.

Restriccion metodologica (seccion 6.6): NO se aplica aumento de datos
en ningun enfoque, para observar puramente el efecto de la resolucion.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from functools import partial
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "fase2_octree"))
from octree_real import cargar_octree_disperso


CLASES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf",
    "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person",
    "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent",
    "toilet", "tv_stand", "vase", "wardrobe", "xbox",
]


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


class OctreeDataset(Dataset):
    """
    Carga octrees dispersos (.npz) desde la estructura generada en
    Fase 2, y los materializa a grid denso al vuelo para Net5Octree.

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
        raiz = Path(raiz_resolucion)
        self.resolucion = resolucion
        carpeta_split = "train" if split in ("train", "val") else "test"

        self.muestras = []
        for clase in CLASES:
            carpeta = raiz / clase / carpeta_split
            if not carpeta.exists():
                continue
            for archivo in sorted(carpeta.glob("*.npz")):
                self.muestras.append(str(archivo))

        if idx_subset is not None:
            self.muestras = [self.muestras[i] for i in idx_subset
                             if i < len(self.muestras)]

        print(f"[Dataset] Octree disperso '{split}': {len(self.muestras)} muestras "
             f"(materializacion a {resolucion}^3 al vuelo)")

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, idx: int):
        d = cargar_octree_disperso(self.muestras[idx])
        grid_np = _materializar_grid_denso(
            d["centros_hoja"], d["normales_hoja"], self.resolucion,
        )
        grid = torch.from_numpy(grid_np).float()    # (4, R, R, R)
        etiqueta = torch.tensor(d["etiqueta"], dtype=torch.long)
        return grid, etiqueta


def _worker_init_fn(worker_id: int, seed: int = 42) -> None:
    np.random.seed(seed + worker_id)


def crear_dataloaders_octree(
    raiz_resolucion: str,
    resolucion: int,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray = None,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple:
    """Crea DataLoaders de train, val y test para Net5Octree, a partir
    del formato disperso (materializacion a denso ocurre en __getitem__).

    idx_test=None (por defecto) usa el conjunto de prueba oficial completo;
    se puede pasar un subconjunto de indices para pruebas rapidas (smoke
    tests) de la propia rutina de entrenamiento."""

    ds_train = OctreeDataset(raiz_resolucion, resolucion, "train", idx_train)
    ds_val   = OctreeDataset(raiz_resolucion, resolucion, "val",   idx_val)
    ds_test  = OctreeDataset(raiz_resolucion, resolucion, "test",  idx_test)

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