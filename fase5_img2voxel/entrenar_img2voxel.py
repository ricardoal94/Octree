"""
entrenar_img2voxel.py - Experimento adicional: Img2Voxel
============================================================
Entrena el modelo Img2Voxel (imagen 2D -> voxel grid 3D) y mide su
costo computacional con el mismo protocolo usado en HCE y Net5
(Fase 4 de la metodologia principal), para que sea comparable:

  - Tiempo de entrenamiento total hasta convergencia (early stopping)
  - Tiempo de inferencia promedio por muestra
  - VRAM pico (GPU)
  - Tamano del modelo guardado (MB)
  - Metricas de reconstruccion: IoU de ocupacion (Intersection over Union)

Uso:
    python entrenar_img2voxel.py --resolucion 32
    python entrenar_img2voxel.py --resolucion 64
"""

import sys
import csv
import json
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "fase1_modelnet40"))
sys.path.insert(0, str(Path(__file__).parent))

from fase1_setup        import set_global_seed, particionar_dataset
from img2voxel_modelo    import crear_modelo, get_device, perdida_reconstruccion
from img2voxel_dataset   import crear_dataloaders_img2voxel

# ── Rutas ──────────────────────────────────────────────────────
RAIZ_DATA      = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\data")
RAIZ_RENDERS   = RAIZ_DATA / "renders"
DIR_CKPT       = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\checkpoints")
DIR_LOGS       = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\logs")
DIR_RESULTADOS = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\resultados")
SEED = 42
N_VISTAS = 8   # debe coincidir con lo usado en renderizar_vistas.py


# ──────────────────────────────────────────────────────────────
# METRICA: IoU de ocupacion (Intersection over Union)
# ──────────────────────────────────────────────────────────────

def calcular_iou(voxel_logits: torch.Tensor, voxel_real: torch.Tensor,
                 umbral: float = 0.5) -> float:
    """
    IoU entre la prediccion binarizada de ocupacion y la ocupacion real.
    Es la metrica estandar para evaluar calidad de reconstruccion
    volumetrica (usada en papers como 3D-R2N2, Pix2Vox).
    """
    pred_prob = torch.sigmoid(voxel_logits[:, 0])
    pred_bin  = (pred_prob > umbral).float()
    real_bin  = voxel_real[:, 0]

    interseccion = (pred_bin * real_bin).sum(dim=(1, 2, 3))
    union = ((pred_bin + real_bin) > 0).float().sum(dim=(1, 2, 3))

    iou = interseccion / union.clamp(min=1.0)
    return iou.mean().item()


# ──────────────────────────────────────────────────────────────
# ENTRENAMIENTO DE UNA EPOCA
# ──────────────────────────────────────────────────────────────

def entrenar_epoca(modelo, loader, optimizador, device):
    modelo.train()
    perdida_acum, iou_acum, n_batches = 0.0, 0.0, 0

    barra = tqdm(loader, desc="  Train", leave=False, ncols=80)
    for imagenes, voxels_reales, _ in barra:
        imagenes      = imagenes.to(device, non_blocking=True)
        voxels_reales = voxels_reales.to(device, non_blocking=True)

        optimizador.zero_grad()
        voxel_logits = modelo(imagenes)
        perdidas = perdida_reconstruccion(voxel_logits, voxels_reales)
        perdidas["perdida_total"].backward()
        optimizador.step()

        iou = calcular_iou(voxel_logits, voxels_reales)

        perdida_acum += perdidas["perdida_total"].item()
        iou_acum     += iou
        n_batches    += 1
        barra.set_postfix(loss=f"{perdidas['perdida_total'].item():.4f}", iou=f"{iou:.3f}")

    return perdida_acum / n_batches, iou_acum / n_batches


# ──────────────────────────────────────────────────────────────
# EVALUACION
# ──────────────────────────────────────────────────────────────

def evaluar(modelo, loader, device, desc="Val"):
    modelo.eval()
    perdida_acum, iou_acum, n_batches = 0.0, 0.0, 0

    with torch.no_grad():
        barra = tqdm(loader, desc=f"  {desc}", leave=False, ncols=80)
        for imagenes, voxels_reales, _ in barra:
            imagenes      = imagenes.to(device, non_blocking=True)
            voxels_reales = voxels_reales.to(device, non_blocking=True)

            voxel_logits = modelo(imagenes)
            perdidas = perdida_reconstruccion(voxel_logits, voxels_reales)
            iou = calcular_iou(voxel_logits, voxels_reales)

            perdida_acum += perdidas["perdida_total"].item()
            iou_acum     += iou
            n_batches    += 1

    return perdida_acum / n_batches, iou_acum / n_batches


# ──────────────────────────────────────────────────────────────
# MEDICION DE COSTO COMPUTACIONAL (mismo protocolo que HCE/Net5)
# ──────────────────────────────────────────────────────────────

def medir_tiempo_inferencia(modelo, loader, device, n_repeticiones: int = 3) -> dict:
    modelo.eval()
    tiempos = []

    for _ in range(n_repeticiones):
        t0 = time.time()
        total = 0
        with torch.no_grad():
            for imagenes, _, _ in loader:
                imagenes = imagenes.to(device, non_blocking=True)
                modelo(imagenes)
                total += imagenes.size(0)
        tiempos.append(time.time() - t0)

    tiempo_total_prom = float(np.mean(tiempos))
    tiempo_por_muestra_ms = (tiempo_total_prom / total) * 1000

    return {
        "tiempo_inferencia_total_s":     round(tiempo_total_prom, 4),
        "tiempo_inferencia_promedio_ms": round(tiempo_por_muestra_ms, 4),
        "n_muestras_test":               int(total),
    }


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolucion", type=int, default=32, choices=[32, 64])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    R = args.resolucion

    print("=" * 60)
    print(f"  EXPERIMENTO ADICIONAL: IMG2VOXEL — Resolucion {R}^3")
    print("=" * 60)
    print("  NOTA: Esto reconstruye una APROXIMACION del voxel grid")
    print("  a partir de UNA imagen 2D. No forma parte de la metodologia")
    print("  principal de tesis (HCE vs Net5 clasificadores).")
    print("=" * 60)

    set_global_seed(SEED)
    device = get_device()

    for d in (DIR_CKPT, DIR_LOGS, DIR_RESULTADOS):
        d.mkdir(parents=True, exist_ok=True)

    # Particion train/val (reutiliza la misma de toda la tesis)
    npz_path = DIR_LOGS / "particion_indices.npz"
    if npz_path.exists():
        data      = np.load(str(npz_path))
        idx_train = data["idx_train"]
        idx_val   = data["idx_val"]
    else:
        particion = particionar_dataset(seed=SEED)
        idx_train = np.array(particion["train"]["indices"])
        idx_val   = np.array(particion["val"]["indices"])

    # DataLoaders
    raiz_octrees = RAIZ_DATA / f"octrees_{R}"
    loader_train, loader_val, loader_test = crear_dataloaders_img2voxel(
        raiz_renders=str(RAIZ_RENDERS),
        raiz_octrees=str(raiz_octrees),
        idx_train=idx_train, idx_val=idx_val,
        n_vistas=N_VISTAS, batch_size=args.batch_size,
        num_workers=4, seed=SEED,
    )

    # Modelo y optimizador
    modelo, device = crear_modelo(resolucion=R, device=device)
    optimizador = optim.Adam(modelo.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler   = optim.lr_scheduler.StepLR(optimizador, step_size=20, gamma=0.7)

    # Logs
    csv_path  = DIR_LOGS / f"img2voxel_historial_R{R}.csv"
    ckpt_path = DIR_CKPT / f"img2voxel_mejor_R{R}.pth"

    historial = []
    mejor_val_iou = 0.0
    epocas_sin_mejora = 0

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoca", "train_loss", "train_iou", "val_loss", "val_iou", "lr", "tiempo_s"]
        )

    print(f"\n  Epocas max   : {args.epochs}")
    print(f"  Early stop   : patience={args.patience}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Vistas/objeto: {N_VISTAS}\n")

    t_inicio = time.time()

    for epoca in range(1, args.epochs + 1):
        t_ep = time.time()

        train_loss, train_iou = entrenar_epoca(modelo, loader_train, optimizador, device)
        val_loss,   val_iou   = evaluar(modelo, loader_val, device, "Val")

        scheduler.step()
        lr_actual = scheduler.get_last_lr()[0]
        t_ep = time.time() - t_ep

        es_mejor = val_iou > mejor_val_iou
        if es_mejor:
            mejor_val_iou = val_iou
            epocas_sin_mejora = 0
            torch.save({
                "epoca": epoca, "model_state": modelo.state_dict(),
                "mejor_val_iou": mejor_val_iou, "resolucion": R,
            }, ckpt_path)
            marca = " ← mejor"
        else:
            epocas_sin_mejora += 1
            marca = ""

        print(
            f"  Ep {epoca:03d}/{args.epochs} | "
            f"Train loss {train_loss:.4f} IoU {train_iou:.3f} | "
            f"Val loss {val_loss:.4f} IoU {val_iou:.3f} | "
            f"LR {lr_actual:.6f} | {t_ep:.1f}s{marca}"
        )

        if device.type == "cuda" and epoca == 1:
            vram_mb = torch.cuda.max_memory_allocated(device) / 1e6
            print(f"  [GPU] VRAM pico: {vram_mb:.1f} MB")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoca, round(train_loss,6), round(train_iou,6),
                                    round(val_loss,6), round(val_iou,6),
                                    round(lr_actual,8), round(t_ep,2)])
        historial.append({"epoca": epoca, "train_loss": round(train_loss,6),
                          "train_iou": round(train_iou,6), "val_loss": round(val_loss,6),
                          "val_iou": round(val_iou,6), "tiempo_s": round(t_ep,2)})

        if epocas_sin_mejora >= args.patience:
            print(f"\n  [Early Stopping] Sin mejora en {args.patience} epocas. "
                 f"Deteniendo en epoca {epoca}.")
            break

    t_total = (time.time() - t_inicio) / 60

    # Evaluacion final en test
    print("\n[Test] Cargando mejor modelo y evaluando...")
    ckpt = torch.load(ckpt_path, map_location=device)
    modelo.load_state_dict(ckpt["model_state"])
    test_loss, test_iou = evaluar(modelo, loader_test, device, "Test")

    print("[Costo computacional] Midiendo tiempo de inferencia...")
    metricas_inf = medir_tiempo_inferencia(modelo, loader_test, device)

    tamano_mb = ckpt_path.stat().st_size / 1e6
    vram_pico_mb = torch.cuda.max_memory_allocated(device) / 1e6 \
                   if device.type == "cuda" else 0.0

    print("\n" + "=" * 60)
    print(f"  ENTRENAMIENTO COMPLETADO — Img2Voxel R={R}^3")
    print("=" * 60)
    print(f"  Tiempo total    : {t_total:.1f} min")
    print(f"  Mejor Val IoU   : {mejor_val_iou:.4f}  (ep {ckpt['epoca']})")
    print(f"  Test  IoU final : {test_iou:.4f}")
    print(f"  Inf/muestra     : {metricas_inf['tiempo_inferencia_promedio_ms']:.3f} ms")
    print(f"  VRAM pico       : {vram_pico_mb:.1f} MB")
    print(f"  Tamano modelo   : {tamano_mb:.2f} MB")
    print("=" * 60)

    resumen = {
        "experimento": "img2voxel (adicional, fuera de metodologia principal)",
        "resolucion": R,
        "n_vistas_entrenamiento": N_VISTAS,
        "seed": SEED,
        "mejor_val_iou": round(float(mejor_val_iou), 6),
        "mejor_epoca":   int(ckpt["epoca"]),
        "test_iou":      round(float(test_iou), 6),
        "test_loss":     round(float(test_loss), 6),
        "tiempo_total_min": round(t_total, 2),
        "tamano_modelo_mb": round(tamano_mb, 2),
        "vram_pico_mb":     round(vram_pico_mb, 1),
        **metricas_inf,
    }

    with open(DIR_RESULTADOS / f"resumen_img2voxel_R{R}.json", "w") as f:
        json.dump(resumen, f, indent=2)

    print(f"\n  Resultados guardados en: resultados/resumen_img2voxel_R{R}.json")

    return resumen


if __name__ == "__main__":
    main()
