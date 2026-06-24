"""
diagnostico_img2voxel.py
==========================
Verifica paso a paso por que el modelo colapsa a IoU=0.
Corre esto ANTES de volver a entrenar.

Uso:
    python diagnostico_img2voxel.py --resolucion 64
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "fase1_modelnet40"))
sys.path.insert(0, str(Path(__file__).parent))

from fase1_setup       import set_global_seed, particionar_dataset
from img2voxel_modelo  import Img2Voxel, get_device
from img2voxel_dataset import Img2VoxelDataset

RAIZ_DATA    = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\data")
RAIZ_RENDERS = RAIZ_DATA / "renders"
DIR_LOGS     = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\logs")
SEED = 42
N_VISTAS = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolucion", type=int, default=32, choices=[32, 64])
    args = parser.parse_args()
    R = args.resolucion

    set_global_seed(SEED)
    device = get_device()

    print("=" * 60)
    print(f"  DIAGNOSTICO IMG2VOXEL — Resolucion {R}^3")
    print("=" * 60)

    # ── 1. Verificar que los pares imagen/voxel se cargan correctamente ──
    print("\n[1/5] Verificando carga del dataset...")
    raiz_octrees = RAIZ_DATA / f"octrees_{R}"
    ds = Img2VoxelDataset(
        str(RAIZ_RENDERS), str(raiz_octrees), split="train",
        n_vistas=N_VISTAS, modo_vista="fija",
    )
    print(f"  Muestras en dataset: {len(ds)}")

    img, voxel, etiqueta = ds[0]
    print(f"  Imagen shape: {img.shape}, dtype: {img.dtype}")
    print(f"  Imagen rango: [{img.min():.3f}, {img.max():.3f}]")
    print(f"  Voxel shape:  {voxel.shape}, dtype: {voxel.dtype}")
    print(f"  Voxel canal 0 (ocupacion) rango: [{voxel[0].min():.3f}, {voxel[0].max():.3f}]")

    n_ocupadas = (voxel[0] > 0.5).sum().item()
    n_total = voxel[0].numel()
    pct_ocup = 100 * n_ocupadas / n_total
    print(f"  Celdas ocupadas en muestra 0: {n_ocupadas:,} / {n_total:,} ({pct_ocup:.1f}%)")
    print(f"  (Con dilatacion aplicada en el dataset)")

    if n_ocupadas == 0:
        print("\n  [ERROR CRITICO] El voxel grid esta completamente VACIO.")
        return

    # Verificar con mas muestras
    print("\n  Verificando ocupacion en 10 muestras aleatorias...")
    rng = np.random.default_rng(0)
    indices = rng.choice(min(len(ds), 500), 10, replace=False)
    ocupaciones = []
    for i in indices:
        _, v, _ = ds[i]
        pct = 100 * (v[0] > 0.5).sum().item() / v[0].numel()
        ocupaciones.append(pct)
    print(f"  Ocupacion promedio: {np.mean(ocupaciones):.1f}%")
    print(f"  Ocupacion min/max:  {np.min(ocupaciones):.1f}% / {np.max(ocupaciones):.1f}%")

    if np.mean(ocupaciones) < 1.0:
        print("\n  [ERROR] Ocupacion promedio < 1% — los voxels estan casi vacios.")
        print("  Causa probable: los octrees preprocesados tienen un problema.")
        return

    # ── 2. Verificar salida inicial del modelo (antes de entrenar) ──
    print("\n[2/5] Verificando salida inicial del modelo (pesos aleatorios)...")
    modelo = Img2Voxel(resolucion=R).to(device)
    modelo.eval()

    img_batch = img.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = modelo(img_batch)

    logits_ocup = logits[0, 0]
    prob_ocup = torch.sigmoid(logits_ocup)
    print(f"  Logits ocupacion rango: [{logits_ocup.min():.3f}, {logits_ocup.max():.3f}]")
    print(f"  Probs ocupacion rango:  [{prob_ocup.min():.3f}, {prob_ocup.max():.3f}]")
    print(f"  Probs > 0.5 (predicciones positivas): {(prob_ocup > 0.5).sum().item()}")

    # ── 3. Verificar pos_weight aplicado ──
    print("\n[3/5] Verificando pos_weight...")
    voxel_batch = voxel.unsqueeze(0).to(device)
    ocup_real = voxel_batch[:, 0]
    n_ocup_batch = ocup_real.sum().clamp(min=1.0)
    n_total_batch = ocup_real.numel()
    n_vacias_batch = n_total_batch - n_ocup_batch
    pw = (n_vacias_batch / n_ocup_batch).clamp(max=50.0)
    print(f"  Celdas ocupadas en batch: {n_ocup_batch.item():.0f}")
    print(f"  pos_weight calculado:     {pw.item():.2f}")

    perdida_con_pw = F.binary_cross_entropy_with_logits(
        logits_ocup.unsqueeze(0), ocup_real, pos_weight=pw,
    )
    perdida_sin_pw = F.binary_cross_entropy_with_logits(
        logits_ocup.unsqueeze(0), ocup_real,
    )
    print(f"  Loss BCE sin pos_weight:  {perdida_sin_pw.item():.4f}")
    print(f"  Loss BCE con pos_weight:  {perdida_con_pw.item():.4f}")

    # ── 4. Verificar que el IoU mejora cuando el modelo predice bien ──
    print("\n[4/5] Verificando IoU con prediccion perfecta...")
    logits_perfectos = voxel_batch[:, 0].clone() * 10 - 5   # +10 donde ocupado, -5 donde vacio
    prob_perfectos = torch.sigmoid(logits_perfectos)
    pred_bin = (prob_perfectos > 0.5).float()
    real_bin = voxel_batch[:, 0]
    interseccion = (pred_bin * real_bin).sum()
    union = ((pred_bin + real_bin) > 0).float().sum()
    iou_perfecto = interseccion / union.clamp(min=1.0)
    print(f"  IoU con logits perfectos: {iou_perfecto.item():.4f} (debe ser ~1.0)")

    # ── 5. Verificar emparejamiento imagen/voxel ──
    print("\n[5/5] Verificando emparejamiento imagen-voxel...")
    clase_esperada = ds.muestras[0][1]
    print(f"  Muestra 0: clase={clase_esperada}, imagen={img.shape}, voxel={voxel.shape}")
    print(f"  La imagen tiene contenido (no toda negra): {img.mean().item():.3f} (>0 es buena senal)")

    print("\n" + "=" * 60)
    print("  DIAGNOSTICO COMPLETADO")
    print("=" * 60)
    print("\n  Si llegaste aqui sin errores criticos, el dataset esta bien.")
    print("  El problema del IoU=0 es de entrenamiento, no de datos.")
    print("  Prueba correr con LR mas bajo: --lr 0.0001")


if __name__ == "__main__":
    main()
