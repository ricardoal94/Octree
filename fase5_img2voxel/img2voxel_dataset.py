import numpy as np
import torch
import torch.nn.functional as F_nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from functools import partial

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

class Img2VoxelDataset(Dataset):
    def __init__(self, raiz_renders, raiz_octrees, split="train",
                idx_subset=None, n_vistas=8, modo_vista="aleatoria", seed=42):
        super().__init__()
        self.raiz_renders = Path(raiz_renders)
        self.raiz_octrees = Path(raiz_octrees)
        self.n_vistas = n_vistas
        self.modo_vista = modo_vista
        self.seed = seed
        carpeta_split = "train" if split in ("train", "val") else "test"
        self.muestras = []
        for clase in CLASES:
            carpeta_octree = self.raiz_octrees / clase / carpeta_split
            if not carpeta_octree.exists():
                continue
            for archivo_npz in sorted(carpeta_octree.glob("*.npz")):
                self.muestras.append((archivo_npz.stem, clase))
        if idx_subset is not None:
            self.muestras = [self.muestras[i] for i in idx_subset if i < len(self.muestras)]
        self.split = carpeta_split
        print(f"[Dataset] Img2Voxel '{split}': {len(self.muestras)} objetos x {n_vistas} vistas")

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, idx):
        nombre, clase = self.muestras[idx]
        if self.modo_vista == "aleatoria":
            rng = np.random.default_rng(self.seed + idx + hash(nombre) % 10000)
            vista_idx = int(rng.integers(0, self.n_vistas))
        else:
            vista_idx = 0
        ruta_imagen = self.raiz_renders / clase / self.split / f"{nombre}_v{vista_idx:02d}.png"
        ruta_octree = self.raiz_octrees / clase / self.split / f"{nombre}.npz"
        img = Image.open(ruta_imagen).convert("L")
        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0)
        data = np.load(ruta_octree)
        grid_raw = data["grid"].copy()
        ocup = torch.from_numpy(grid_raw[0]).float().unsqueeze(0).unsqueeze(0)
        for _ in range(4):
            ocup = F_nn.max_pool3d(ocup, kernel_size=3, stride=1, padding=1)
        grid_tensor = torch.from_numpy(grid_raw).float()
        grid_tensor[0] = ocup.squeeze()
        etiqueta = torch.tensor(int(data["etiqueta"]), dtype=torch.long)
        return img_tensor, grid_tensor, etiqueta

def _worker_init_fn(worker_id, seed=42):
    np.random.seed(seed + worker_id)

def crear_dataloaders_img2voxel(raiz_renders, raiz_octrees, idx_train, idx_val,
                                 n_vistas=8, batch_size=16, num_workers=4, seed=42):
    ds_train = Img2VoxelDataset(raiz_renders, raiz_octrees, "train", idx_train, n_vistas, "aleatoria", seed)
    ds_val   = Img2VoxelDataset(raiz_renders, raiz_octrees, "val",   idx_val,   n_vistas, "fija",      seed)
    ds_test  = Img2VoxelDataset(raiz_renders, raiz_octrees, "test",  None,      n_vistas, "fija",      seed)
    g = torch.Generator()
    g.manual_seed(seed)
    init_fn = partial(_worker_init_fn, seed=seed)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0),
        worker_init_fn=init_fn, generator=g)
    loader_val  = DataLoader(ds_val,  batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
    return loader_train, loader_val, loader_test
