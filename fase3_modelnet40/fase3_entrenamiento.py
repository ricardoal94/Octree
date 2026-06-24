"""
fase3_entrenamiento.py
======================
Entrena Net5 sobre ModelNet40 con:
  - GPU (RTX 5070) via CUDA
  - Scheduler StepLR
  - Checkpointing del mejor modelo
  - Log de metricas por epoca en CSV y JSON
  - Barra de progreso con tqdm

Uso:
    python fase3_entrenamiento.py
"""

import sys
import csv
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

# Modulos propios
sys.path.insert(0, str(Path(__file__).parent.parent / "fase1_modelnet40"))
sys.path.insert(0, str(Path(__file__).parent.parent / "fase2_modelnet40"))

from fase1_setup      import set_global_seed, particionar_dataset, cargar_config
from dataset          import crear_dataloaders
from modelo           import crear_modelo, get_device, perdida_ortogonalidad

# ── Rutas ──────────────────────────────────────────────────────
RAIZ_DATASET  = r"C:\Users\ricar\Documents\Codigos\Tesis\Dataset\ModelNet40"
RAIZ_CONFIG   = Path(__file__).parent.parent / "fase1_modelnet40" / "config.yaml"
DIR_CKPT      = Path(__file__).parent.parent / "checkpoints"
DIR_LOGS      = Path(__file__).parent.parent / "logs"
DIR_RESULTADOS= Path(__file__).parent.parent / "resultados"
SEED          = 42
PESO_ORTOGONAL= 0.001   # peso del termino de regularizacion del feature T-Net


# ──────────────────────────────────────────────────────────────
# ENTRENAMIENTO DE UNA EPOCA
# ──────────────────────────────────────────────────────────────

def entrenar_epoca(modelo, loader, criterio, optimizador, device, peso_ortogonal=0.001):
    modelo.train()
    perdida_total = 0.0
    correctos     = 0
    total         = 0

    barra = tqdm(loader, desc="  Train", leave=False, ncols=80)
    for puntos, etiquetas in barra:
        puntos    = puntos.to(device)
        etiquetas = etiquetas.to(device)

        optimizador.zero_grad()
        logits, trans_feat = modelo(puntos, retornar_trans_feat=True)

        perdida_clasif = criterio(logits, etiquetas)
        perdida_ortog  = perdida_ortogonalidad(trans_feat)
        perdida = perdida_clasif + peso_ortogonal * perdida_ortog

        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item() * puntos.size(0)
        correctos     += (logits.argmax(dim=1) == etiquetas).sum().item()
        total         += puntos.size(0)

        barra.set_postfix(loss=f"{perdida.item():.4f}")

    return perdida_total / total, correctos / total


# ──────────────────────────────────────────────────────────────
# EVALUACION
# ──────────────────────────────────────────────────────────────

def evaluar(modelo, loader, criterio, device, desc="Val"):
    modelo.eval()
    perdida_total = 0.0
    correctos     = 0
    total         = 0

    with torch.no_grad():
        barra = tqdm(loader, desc=f"  {desc} ", leave=False, ncols=80)
        for puntos, etiquetas in barra:
            puntos    = puntos.to(device)
            etiquetas = etiquetas.to(device)

            logits  = modelo(puntos)
            perdida = criterio(logits, etiquetas)

            perdida_total += perdida.item() * puntos.size(0)
            correctos     += (logits.argmax(dim=1) == etiquetas).sum().item()
            total         += puntos.size(0)

    return perdida_total / total, correctos / total


# ──────────────────────────────────────────────────────────────
# GUARDAR / CARGAR CHECKPOINT
# ──────────────────────────────────────────────────────────────

def guardar_checkpoint(modelo, optimizador, scheduler, epoca, metrica, ruta):
    torch.save({
        "epoca":           epoca,
        "model_state":     modelo.state_dict(),
        "optimizer_state": optimizador.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "mejor_val_acc":   metrica,
    }, ruta)


# ──────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FASE 3: ENTRENAMIENTO Net5 — ModelNet40")
    print("=" * 60)

    # 1. Semilla y dispositivo
    set_global_seed(SEED)
    device = get_device()

    # 2. Configuracion
    cfg        = cargar_config(str(RAIZ_CONFIG))
    n_puntos   = cfg["dataset"]["num_points"]
    batch_size = cfg["entrenamiento"]["batch_size"]
    epochs     = cfg["entrenamiento"]["epochs"]
    lr         = cfg["entrenamiento"]["learning_rate"]
    step_size  = cfg["entrenamiento"]["step_size"]
    gamma      = cfg["entrenamiento"]["gamma"]
    wd         = cfg["entrenamiento"]["weight_decay"]
    num_clases = cfg["modelo"]["num_clases"]
    dropout    = cfg["modelo"]["dropout"]

    # 3. Particion
    npz_path = DIR_LOGS / "particion_indices.npz"
    if npz_path.exists():
        data      = np.load(str(npz_path))
        idx_train = data["idx_train"]
        idx_val   = data["idx_val"]
    else:
        particion = particionar_dataset(seed=SEED)
        idx_train = np.array(particion["train"]["indices"])
        idx_val   = np.array(particion["val"]["indices"])

    # 4. DataLoaders
    loader_train, loader_val, loader_test = crear_dataloaders(
        raiz_dataset=RAIZ_DATASET,
        idx_train=idx_train,
        idx_val=idx_val,
        n_puntos=n_puntos,
        batch_size=batch_size,
        num_workers=4,
        seed=SEED,
    )

    # 5. Modelo, criterio, optimizador, scheduler
    modelo, device = crear_modelo(num_clases=num_clases, dropout=dropout, device=device)
    criterio       = nn.CrossEntropyLoss()
    optimizador    = optim.Adam(modelo.parameters(), lr=lr, weight_decay=wd)
    scheduler      = optim.lr_scheduler.StepLR(optimizador, step_size=step_size, gamma=gamma)

    # 6. Preparar archivos de log
    DIR_CKPT.mkdir(parents=True, exist_ok=True)
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    csv_path  = DIR_LOGS / "historial_entrenamiento.csv"
    json_path = DIR_LOGS / "historial_entrenamiento.json"
    ckpt_mejor= DIR_CKPT / "mejor_modelo.pth"
    ckpt_ultimo=DIR_CKPT / "ultimo_modelo.pth"

    historial = []
    mejor_val_acc = 0.0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoca", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "tiempo_s"])

    print(f"\n  Epocas     : {epochs}")
    print(f"  Batch size : {batch_size}")
    print(f"  LR inicial : {lr}")
    print(f"  Dispositivo: {device}\n")

    # 7. Loop de entrenamiento
    t_inicio = time.time()

    for epoca in range(1, epochs + 1):
        t_ep = time.time()

        train_loss, train_acc = entrenar_epoca(modelo, loader_train, criterio, optimizador, device, peso_ortogonal=PESO_ORTOGONAL)
        val_loss,   val_acc   = evaluar(modelo, loader_val,   criterio, device, "Val ")

        scheduler.step()
        lr_actual = scheduler.get_last_lr()[0]
        t_ep = time.time() - t_ep

        # Guardar mejor modelo
        es_mejor = val_acc > mejor_val_acc
        if es_mejor:
            mejor_val_acc = val_acc
            guardar_checkpoint(modelo, optimizador, scheduler, epoca, mejor_val_acc, ckpt_mejor)
            marca = " ← mejor"
        else:
            marca = ""

        # Guardar ultimo checkpoint
        guardar_checkpoint(modelo, optimizador, scheduler, epoca, mejor_val_acc, ckpt_ultimo)

        # Consola
        print(
            f"  Ep {epoca:03d}/{epochs} | "
            f"Train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
            f"Val loss {val_loss:.4f} acc {val_acc*100:.2f}% | "
            f"LR {lr_actual:.6f} | {t_ep:.1f}s{marca}"
        )

        # Log CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoca, round(train_loss,6), round(train_acc,6),
                             round(val_loss,6), round(val_acc,6),
                             round(lr_actual,8), round(t_ep,2)])

        # Log JSON
        historial.append({
            "epoca": epoca, "train_loss": round(train_loss,6),
            "train_acc": round(train_acc,6), "val_loss": round(val_loss,6),
            "val_acc": round(val_acc,6), "lr": round(lr_actual,8),
            "tiempo_s": round(t_ep,2),
        })
        with open(json_path, "w") as f:
            json.dump(historial, f, indent=2)

    # 8. Evaluacion final en test
    print("\n[Test] Evaluando mejor modelo en conjunto de test...")
    ckpt = torch.load(ckpt_mejor, map_location=device)
    modelo.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluar(modelo, loader_test, criterio, device, "Test")

    t_total = (time.time() - t_inicio) / 60

    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO COMPLETADO")
    print(f"  Tiempo total     : {t_total:.1f} min")
    print(f"  Mejor Val Acc    : {mejor_val_acc*100:.2f}%  (ep {ckpt['epoca']})")
    print(f"  Test  Acc final  : {test_acc*100:.2f}%")
    print(f"  Checkpoint       : {ckpt_mejor}")
    print(f"  Log CSV          : {csv_path}")
    print("=" * 60)

    # Guardar resumen final
    resumen = {
        "mejor_val_acc":  round(mejor_val_acc, 6),
        "mejor_epoca":    int(ckpt["epoca"]),
        "test_acc":       round(test_acc, 6),
        "test_loss":      round(test_loss, 6),
        "tiempo_total_min": round(t_total, 2),
        "epochs":         epochs,
        "batch_size":     batch_size,
        "seed":           SEED,
        "dispositivo":    str(device),
    }
    with open(DIR_RESULTADOS / "resumen_fase3.json", "w") as f:
        json.dump(resumen, f, indent=2)

    return modelo, resumen


if __name__ == "__main__":
    main()