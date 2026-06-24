"""
test_dilatacion.py
===================
Test independiente que NO depende de img2voxel_dataset.py.
Carga un .npz directamente, aplica dilatacion y muestra el resultado.
Corre esto para confirmar que la dilatacion funciona en tu maquina.

Uso:
    python test_dilatacion.py --resolucion 64
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

RAIZ_DATA = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\data")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolucion", type=int, default=64, choices=[32, 64])
    args = parser.parse_args()
    R = args.resolucion

    # Buscar el primer .npz disponible
    raiz_octrees = RAIZ_DATA / f"octrees_{R}"
    npz_path = None
    for clase_dir in sorted(raiz_octrees.iterdir()):
        for split_dir in sorted(clase_dir.iterdir()):
            for f in sorted(split_dir.glob("*.npz")):
                npz_path = f
                break
        if npz_path:
            break

    if npz_path is None:
        print(f"[ERROR] No se encontraron archivos .npz en {raiz_octrees}")
        return

    print(f"Archivo cargado: {npz_path}")

    data = np.load(npz_path)
    grid_raw = data["grid"]   # (4, R, R, R)
    ocup_raw = grid_raw[0]

    n_ocup_raw = (ocup_raw > 0.5).sum()
    n_total = ocup_raw.size
    print(f"\nAntes de dilatar:")
    print(f"  Celdas ocupadas: {n_ocup_raw:,} / {n_total:,} ({100*n_ocup_raw/n_total:.2f}%)")

    # Aplicar dilatacion con max_pool3d
    ocup_tensor = torch.from_numpy(ocup_raw).float().unsqueeze(0).unsqueeze(0)
    ocup_dilatada = F.max_pool3d(ocup_tensor, kernel_size=3, stride=1, padding=1)
    ocup_dilatada_np = ocup_dilatada.squeeze().numpy()

    n_ocup_dil = (ocup_dilatada_np > 0.5).sum()
    print(f"\nDespues de dilatar (kernel=3):")
    print(f"  Celdas ocupadas: {n_ocup_dil:,} / {n_total:,} ({100*n_ocup_dil/n_total:.2f}%)")
    print(f"  Factor de aumento: {n_ocup_dil/max(n_ocup_raw,1):.1f}x")

    # Verificar que el contenido del npz tiene sentido
    print(f"\nCanales del grid:")
    for i, nombre in enumerate(["ocupacion", "normal_x", "normal_y", "normal_z"]):
        canal = grid_raw[i]
        print(f"  Canal {i} ({nombre}): min={canal.min():.3f}, max={canal.max():.3f}, "
              f"mean={canal.mean():.4f}")

    # Cuantos npz hay en total?
    total_npz = sum(1 for _ in raiz_octrees.rglob("*.npz"))
    print(f"\nTotal archivos .npz en octrees_{R}/: {total_npz:,}")

    print("\n[OK] Test de dilatacion completado.")
    print("Si el factor de aumento es >= 5x, la dilatacion funciona correctamente.")


if __name__ == "__main__":
    main()
