"""
dataset.py - Fase 2
===================
Carga el dataset ModelNet40 desde la estructura de carpetas oficial.

Estructura esperada:
    Dataset/
        ModelNet40/
            airplane/
                train/  *.off o *.txt
                test/   *.off o *.txt
            bathtub/
            ...

Soporta archivos .off y .txt (nubes de puntos pre-muestreadas).
"""

import os
import re
import numpy as np
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# LECTURA DE ARCHIVOS
# ──────────────────────────────────────────────────────────────

def leer_off(ruta: str) -> tuple:
    """
    Lee un archivo .off y retorna (vertices, caras).
    vertices: array (N, 3)
    caras   : array (M, 3) de indices enteros (asume triangulos)
    """
    with open(ruta, "r") as f:
        primera = f.readline().strip()

        # Manejar 'OFF' pegado al numero de vertices: 'OFF1234 ...'
        if primera.upper().startswith("OFF") and primera.upper() != "OFF":
            resto = primera[3:].strip()
            n_verts, n_caras, _ = map(int, resto.split())
        else:
            n_verts, n_caras, _ = map(int, f.readline().split())

        # Leer vertices
        lineas_vertices = [f.readline() for _ in range(n_verts)]
        # Leer caras
        lineas_caras = [f.readline() for _ in range(n_caras)]

    vertices = np.loadtxt(lineas_vertices, dtype=np.float32, usecols=(0, 1, 2))
    if vertices.ndim == 1:
        vertices = vertices.reshape(1, -1)

    # Saneo: algunos .off de ModelNet40 tienen vertices corruptos (NaN/inf)
    vertices = np.nan_to_num(vertices, nan=0.0, posinf=0.0, neginf=0.0)

    # Las caras en .off empiezan con el numero de vertices por cara (3 para triangulos)
    caras = None
    if n_caras > 0:
        caras_raw = np.loadtxt(lineas_caras, dtype=np.int64)
        if caras_raw.ndim == 1:
            caras_raw = caras_raw.reshape(1, -1)
        caras = caras_raw[:, 1:4]   # descartar la primera columna (conteo), tomar 3 indices

    return vertices, caras


def leer_txt(ruta: str) -> np.ndarray:
    """Lee nube de puntos desde .txt (x,y,z,nx,ny,nz) o solo (x,y,z)."""
    data = np.loadtxt(ruta, delimiter=",", dtype=np.float32)
    return data[:, :3]  # solo coordenadas xyz


def cargar_malla(ruta: str) -> tuple:
    """
    Dispatcher: lee .off o .txt segun extension.
    Retorna (vertices, caras). caras es None para .txt (ya es nube de puntos).
    """
    ext = Path(ruta).suffix.lower()
    if ext == ".off":
        return leer_off(ruta)
    elif ext == ".txt":
        return leer_txt(ruta), None
    else:
        raise ValueError(f"Formato no soportado: {ext}")


# ──────────────────────────────────────────────────────────────
# MUESTREO DE PUNTOS
# ──────────────────────────────────────────────────────────────

def muestrear_superficie(
    vertices: np.ndarray, caras: np.ndarray, n_puntos: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Muestreo uniforme sobre la superficie del mesh (area-weighted),
    igual al usado en el paper original de PointNet.

    Para cada triangulo se calcula su area; la probabilidad de elegir
    un triangulo es proporcional a su area. Luego se genera un punto
    aleatorio dentro del triangulo elegido usando coordenadas baricentricas.

    Robusto ante: indices de cara fuera de rango, vertices con NaN/inf,
    triangulos degenerados (area cero) y mallas sin caras validas.
    """
    n_verts = len(vertices)

    # Filtrar caras con indices fuera de rango (datos corruptos en el .off)
    caras_validas = (
        (caras[:, 0] >= 0) & (caras[:, 0] < n_verts) &
        (caras[:, 1] >= 0) & (caras[:, 1] < n_verts) &
        (caras[:, 2] >= 0) & (caras[:, 2] < n_verts)
    )
    caras = caras[caras_validas]

    if len(caras) == 0:
        # Sin caras validas: usar vertices directamente como fallback
        return muestrear_puntos(vertices, n_puntos, rng)

    v0 = vertices[caras[:, 0]]
    v1 = vertices[caras[:, 1]]
    v2 = vertices[caras[:, 2]]

    # Area de cada triangulo via producto cruz
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    # Sanear: reemplazar NaN/inf (vertices corruptos) por area 0
    areas = np.nan_to_num(areas, nan=0.0, posinf=0.0, neginf=0.0)
    areas = np.clip(areas, 0.0, None)

    suma_areas = areas.sum()

    if suma_areas <= 0:
        # Todas las areas son cero/invalidas: fallback a vertices
        return muestrear_puntos(vertices, n_puntos, rng)

    probs = areas / suma_areas
    # Saneo final por seguridad numerica
    probs = np.nan_to_num(probs, nan=0.0)
    probs = probs / probs.sum()

    # Elegir triangulos proporcional al area
    idx_tri = rng.choice(len(caras), size=n_puntos, p=probs)

    # Coordenadas baricentricas aleatorias (muestreo uniforme en triangulo)
    r1 = rng.random(n_puntos).astype(np.float32)
    r2 = rng.random(n_puntos).astype(np.float32)
    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2

    puntos = (
        u[:, None] * v0[idx_tri]
        + v[:, None] * v1[idx_tri]
        + w[:, None] * v2[idx_tri]
    )

    # Saneo final de puntos (por si algun vertice de origen tenia NaN)
    puntos = np.nan_to_num(puntos, nan=0.0, posinf=0.0, neginf=0.0)

    return puntos.astype(np.float32)


def muestrear_puntos(vertices: np.ndarray, n_puntos: int, rng: np.random.Generator) -> np.ndarray:
    """
    Muestrea exactamente n_puntos del array de vertices.
    FALLBACK: solo se usa cuando no hay caras disponibles (ej. .txt).
    Si hay menos vertices, repite con reemplazo.
    """
    n = len(vertices)
    if n >= n_puntos:
        idx = rng.choice(n, n_puntos, replace=False)
    else:
        idx = rng.choice(n, n_puntos, replace=True)
    return vertices[idx]


def normalizar_nube(puntos: np.ndarray) -> np.ndarray:
    """Centra en el origen y escala al radio unitario."""
    centroide = puntos.mean(axis=0)
    puntos = puntos - centroide
    radio = np.max(np.linalg.norm(puntos, axis=1))
    if radio > 0:
        puntos = puntos / radio
    return puntos


# ──────────────────────────────────────────────────────────────
# AUMENTACION DE DATOS
# ──────────────────────────────────────────────────────────────

def aumentar_nube(puntos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Aumentacion estandar para nubes de puntos 3D:
    - Rotacion aleatoria en eje Y
    - Jitter gaussiano
    - Escala aleatoria
    """
    # Rotacion en Y
    angulo = rng.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angulo), np.sin(angulo)
    R = np.array([
        [ cos_a, 0, sin_a],
        [     0, 1,     0],
        [-sin_a, 0, cos_a],
    ], dtype=np.float32)
    puntos = puntos @ R.T

    # Jitter
    puntos += rng.normal(0, 0.02, puntos.shape).astype(np.float32)
    puntos = np.clip(puntos, -1.0, 1.0)

    # Escala
    escala = rng.uniform(0.8, 1.2)
    puntos *= escala

    return puntos


# ──────────────────────────────────────────────────────────────
# DATASET CLASS
# ──────────────────────────────────────────────────────────────

class ModelNet40Dataset(Dataset):
    """
    Dataset PyTorch para ModelNet40.

    Parametros
    ----------
    raiz        : ruta a la carpeta que contiene las 40 subcarpetas de clases
    split       : 'train', 'val' o 'test'
    n_puntos    : puntos a muestrear por modelo (default 1024)
    aumentar    : aplicar data augmentation (solo en train)
    idx_subset  : indices opcionales para sub-particion (train/val split)
    seed        : semilla para reproducibilidad
    """

    # Lista oficial de las 40 clases de ModelNet40
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

    def __init__(
        self,
        raiz: str,
        split: str = "train",
        n_puntos: int = 1024,
        aumentar: bool = False,
        idx_subset: np.ndarray = None,
        seed: int = 42,
    ):
        super().__init__()
        self.raiz     = Path(raiz)
        self.split    = split          # 'train' o 'test'
        self.n_puntos = n_puntos
        self.aumentar = aumentar
        self.seed     = seed
        self.clase2idx = {c: i for i, c in enumerate(self.CLASES)}

        # Carpeta fuente: train usa 'train', val y test usan 'test'
        carpeta_split = "train" if split in ("train", "val") else "test"

        # Recolectar todos los archivos
        self.muestras = []   # lista de (ruta_archivo, etiqueta_int)
        for clase in self.CLASES:
            carpeta = self.raiz / clase / carpeta_split
            if not carpeta.exists():
                continue
            etiqueta = self.clase2idx[clase]
            for archivo in sorted(carpeta.glob("*.off")) or sorted(carpeta.glob("*.txt")):
                self.muestras.append((str(archivo), etiqueta))
            # Buscar .txt si no habia .off
            if not list(carpeta.glob("*.off")):
                for archivo in sorted(carpeta.glob("*.txt")):
                    self.muestras.append((str(archivo), etiqueta))

        # Eliminar duplicados manteniendo orden
        vistos = set()
        unicos = []
        for item in self.muestras:
            if item[0] not in vistos:
                vistos.add(item[0])
                unicos.append(item)
        self.muestras = unicos

        # Aplicar sub-particion (train / val)
        if idx_subset is not None:
            self.muestras = [self.muestras[i] for i in idx_subset if i < len(self.muestras)]

        print(
            f"[Dataset] ModelNet40 '{split}': "
            f"{len(self.muestras)} modelos | "
            f"{n_puntos} pts/modelo | "
            f"augment={aumentar}"
        )

    def __len__(self) -> int:
        return len(self.muestras)

    def __getitem__(self, idx: int):
        ruta, etiqueta = self.muestras[idx]

        # RNG determinista por muestra (seed + idx para reproducibilidad)
        rng = np.random.default_rng(self.seed + idx)

        # Cargar y procesar
        vertices, caras = cargar_malla(ruta)

        if caras is not None and len(caras) > 0:
            # Muestreo uniforme sobre la superficie (metodo correcto, igual a PointNet)
            puntos = muestrear_superficie(vertices, caras, self.n_puntos, rng)
        else:
            # Fallback: muestrear directamente de los vertices (.txt o mesh sin caras)
            puntos = muestrear_puntos(vertices, self.n_puntos, rng)

        puntos = normalizar_nube(puntos)

        if self.aumentar:
            puntos = aumentar_nube(puntos, rng)

        # Transponer a (3, N) para Conv1D de PointNet
        puntos_tensor = torch.from_numpy(puntos.T).float()   # (3, 1024)
        etiqueta_tensor = torch.tensor(etiqueta, dtype=torch.long)

        return puntos_tensor, etiqueta_tensor


# ──────────────────────────────────────────────────────────────
# FACTORY: crea los tres DataLoaders listos para usar
# ──────────────────────────────────────────────────────────────

def _worker_init_fn(worker_id: int, seed: int = 42) -> None:
    """Funcion nombrada (picklable) para inicializar la seed de cada worker."""
    np.random.seed(seed + worker_id)


def crear_dataloaders(
    raiz_dataset: str,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    n_puntos: int = 1024,
    batch_size: int = 32,
    num_workers: int = 4,      # paralelizar carga (CPU prepara mientras GPU entrena)
    seed: int = 42,
) -> tuple:
    """
    Crea DataLoaders de train, val y test.

    Retorna
    -------
    (loader_train, loader_val, loader_test)
    """
    ds_train = ModelNet40Dataset(
        raiz=raiz_dataset, split="train",
        n_puntos=n_puntos, aumentar=True,
        idx_subset=idx_train, seed=seed,
    )
    ds_val = ModelNet40Dataset(
        raiz=raiz_dataset, split="val",
        n_puntos=n_puntos, aumentar=False,
        idx_subset=idx_val, seed=seed,
    )
    ds_test = ModelNet40Dataset(
        raiz=raiz_dataset, split="test",
        n_puntos=n_puntos, aumentar=False,
        idx_subset=None, seed=seed,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    init_fn = partial(_worker_init_fn, seed=seed)

    loader_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=init_fn,
        generator=g,
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


# ──────────────────────────────────────────────────────────────
# TEST RAPIDO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "fase1_modelnet40"))
    from fase1_setup import set_global_seed, particionar_dataset

    RAIZ = r"C:\Users\ricar\Documents\Codigos\Tesis\Dataset\ModelNet40"
    SEED = 42

    set_global_seed(SEED)
    particion = particionar_dataset(seed=SEED)

    idx_train = np.array(particion["train"]["indices"])
    idx_val   = np.array(particion["val"]["indices"])

    loader_train, loader_val, loader_test = crear_dataloaders(
        raiz_dataset=RAIZ,
        idx_train=idx_train,
        idx_val=idx_val,
        n_puntos=1024,
        batch_size=16,
        num_workers=0,
        seed=SEED,
    )

    # Verificar un batch
    puntos, etiquetas = next(iter(loader_train))
    print(f"\n[Test] Batch shape  : {puntos.shape}")     # (16, 3, 1024)
    print(f"[Test] Labels shape : {etiquetas.shape}")    # (16,)
    print(f"[Test] Clases en batch: {etiquetas.tolist()}")
    print("\nDataset cargado correctamente.")