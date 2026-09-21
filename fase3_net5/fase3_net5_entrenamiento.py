"""
fase3_net5_entrenamiento.py - Fase 3 (Enfoque profundo)
=========================================================
Entrenamiento de Net5Octree sobre los grids de octree precomputados.

Caracteristicas:
  - GPU (RTX 5070) via CUDA
  - Early Stopping basado en val accuracy (segun metodologia, sec. 6.3)
  - Optimizador Adam + CrossEntropy
  - Log CSV y JSON por epoca
  - Checkpoint del mejor modelo
  - Medicion de VRAM, tiempo de entrenamiento e inferencia (Fase 4)
  - Sin aumento de datos (criterio de equivalencia, sec. 6.6)

Uso:
    python fase3_net5_entrenamiento.py --resolucion 32
    python fase3_net5_entrenamiento.py --resolucion 64
"""

import sys
import csv
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "fase1_modelnet40"))
sys.path.insert(0, str(Path(__file__).parent))

from fase1_setup      import set_global_seed, particionar_dataset, cargar_config
from net5_modelo      import crear_modelo, get_device
from net5_dataset     import crear_dataloaders_octree

# ── Rutas ──────────────────────────────────────────────────────
RAIZ_DATA      = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\data")
RAIZ_CONFIG    = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\fase1_modelnet40\config.yaml")
DIR_CKPT       = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\checkpoints")
DIR_LOGS       = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\logs")
DIR_RESULTADOS = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\resultados")
SEED = 42


# ──────────────────────────────────────────────────────────────
# ENTRENAMIENTO DE UNA EPOCA
# ──────────────────────────────────────────────────────────────

def entrenar_epoca(modelo, loader, criterio, optimizador, device):
    modelo.train()
    perdida_total, correctos, total = 0.0, 0, 0

    barra = tqdm(loader, desc="  Train", leave=False, ncols=80)
    for grids, etiquetas in barra:
        grids     = grids.to(device, non_blocking=True)
        etiquetas = etiquetas.to(device, non_blocking=True)

        optimizador.zero_grad()
        logits  = modelo(grids)
        perdida = criterio(logits, etiquetas)
        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item() * grids.size(0)
        correctos     += (logits.argmax(1) == etiquetas).sum().item()
        total         += grids.size(0)
        barra.set_postfix(loss=f"{perdida.item():.4f}")

    return perdida_total / total, correctos / total


# ──────────────────────────────────────────────────────────────
# EVALUACION
# ──────────────────────────────────────────────────────────────

def evaluar(modelo, loader, criterio, device, desc="Val"):
    modelo.eval()
    perdida_total, correctos, total = 0.0, 0, 0

    with torch.no_grad():
        barra = tqdm(loader, desc=f"  {desc}", leave=False, ncols=80)
        for grids, etiquetas in barra:
            grids     = grids.to(device, non_blocking=True)
            etiquetas = etiquetas.to(device, non_blocking=True)
            logits    = modelo(grids)
            perdida   = criterio(logits, etiquetas)
            perdida_total += perdida.item() * grids.size(0)
            correctos     += (logits.argmax(1) == etiquetas).sum().item()
            total         += grids.size(0)

    return perdida_total / total, correctos / total


# ──────────────────────────────────────────────────────────────
# MEDIR TIEMPO DE INFERENCIA (Fase 4 de la metodologia)
# ──────────────────────────────────────────────────────────────

def medir_tiempo_inferencia(modelo, loader, device, n_repeticiones: int = 3) -> dict:
    """
    Mide el tiempo promedio de clasificar una sola muestra sobre el
    conjunto de test completo, promediando N repeticiones.
    Cumple con la Fase 4: "tiempo promedio requerido para clasificar
    una sola muestra, calculado sobre el total del conjunto de prueba".
    """
    modelo.eval()
    tiempos = []

    for _ in range(n_repeticiones):
        t0 = time.time()
        with torch.no_grad():
            total = 0
            for grids, _ in loader:
                grids = grids.to(device, non_blocking=True)
                modelo(grids)
                total += grids.size(0)
        tiempos.append(time.time() - t0)

    tiempo_total_prom = np.mean(tiempos)
    tiempo_por_muestra_ms = (tiempo_total_prom / total) * 1000

    return {
        "tiempo_inferencia_total_s":       round(float(tiempo_total_prom), 4),
        "tiempo_inferencia_promedio_ms":    round(float(tiempo_por_muestra_ms), 4),
        "n_muestras_test":                 int(total),
    }


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolucion", type=int, default=32, choices=[32, 64])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping: epocas sin mejora antes de parar")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--limite_train", type=int, default=None,
                        help="Usar solo las primeras N muestras de train "
                             "(smoke test rapido, no afecta el uso normal)")
    parser.add_argument("--limite_val", type=int, default=None,
                        help="Usar solo las primeras N muestras de val")
    parser.add_argument("--limite_test", type=int, default=None,
                        help="Usar solo las primeras N muestras de test")
    parser.add_argument("--tag", type=str, default="",
                        help="Sufijo para checkpoint/logs/resultados, para no "
                             "sobreescribir una corrida completa (ej. '_smoke')")
    args = parser.parse_args()

    R = args.resolucion

    print("=" * 60)
    print(f"  FASE 3: NET5-OCTREE — Resolucion {R}^3")
    print("=" * 60)

    set_global_seed(SEED)
    device = get_device()

    for d in (DIR_CKPT, DIR_LOGS, DIR_RESULTADOS):
        d.mkdir(parents=True, exist_ok=True)

    # Particion train/val
    npz_path = DIR_LOGS / "particion_indices.npz"
    if npz_path.exists():
        data      = np.load(str(npz_path))
        idx_train = data["idx_train"]
        idx_val   = data["idx_val"]
    else:
        particion = particionar_dataset(seed=SEED)
        idx_train = np.array(particion["train"]["indices"])
        idx_val   = np.array(particion["val"]["indices"])

    idx_test = None
    if args.limite_train is not None:
        idx_train = idx_train[:args.limite_train]
    if args.limite_val is not None:
        idx_val = idx_val[:args.limite_val]
    if args.limite_test is not None:
        idx_test = np.arange(args.limite_test)

    # DataLoaders (formato disperso, materializacion a denso al vuelo)
    raiz_resolucion = RAIZ_DATA / f"octrees_{R}"
    loader_train, loader_val, loader_test = crear_dataloaders_octree(
        raiz_resolucion=str(raiz_resolucion),
        resolucion=R,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        batch_size=args.batch_size,
        num_workers=4,
        seed=SEED,
    )

    # Modelo, criterio, optimizador, scheduler
    modelo, device = crear_modelo(resolucion=R, device=device)
    criterio    = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler   = optim.lr_scheduler.StepLR(optimizador, step_size=20, gamma=0.7)

    # Archivos de log
    csv_path  = DIR_LOGS  / f"net5_historial_R{R}{args.tag}.csv"
    json_path = DIR_LOGS  / f"net5_historial_R{R}{args.tag}.json"
    ckpt_path = DIR_CKPT  / f"net5_mejor_R{R}{args.tag}.pth"

    historial = []
    mejor_val_acc    = 0.0
    epocas_sin_mejora = 0

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoca", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "tiempo_s"]
        )

    print(f"\n  Epocas max   : {args.epochs}")
    print(f"  Early stop   : patience={args.patience}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  LR inicial   : {args.lr}")
    print(f"  Train batches: {len(loader_train)}")
    print(f"  Val   batches: {len(loader_val)}\n")

    t_inicio = time.time()

    for epoca in range(1, args.epochs + 1):
        t_ep = time.time()

        train_loss, train_acc = entrenar_epoca(modelo, loader_train, criterio, optimizador, device)
        val_loss,   val_acc   = evaluar(modelo, loader_val, criterio, device, "Val")

        scheduler.step()
        lr_actual = scheduler.get_last_lr()[0]
        t_ep = time.time() - t_ep

        # Checkpoint si mejora
        es_mejor = val_acc > mejor_val_acc
        if es_mejor:
            mejor_val_acc = val_acc
            epocas_sin_mejora = 0
            torch.save({
                "epoca": epoca, "model_state": modelo.state_dict(),
                "optimizer_state": optimizador.state_dict(),
                "mejor_val_acc": mejor_val_acc,
                "resolucion": R,
            }, ckpt_path)
            marca = " <- mejor"
        else:
            epocas_sin_mejora += 1
            marca = ""

        print(
            f"  Ep {epoca:03d}/{args.epochs} | "
            f"Train {train_acc*100:.2f}% loss {train_loss:.4f} | "
            f"Val {val_acc*100:.2f}% loss {val_loss:.4f} | "
            f"LR {lr_actual:.6f} | {t_ep:.1f}s{marca}"
        )

        # VRAM
        if device.type == "cuda" and epoca == 1:
            vram_mb = torch.cuda.max_memory_allocated(device) / 1e6
            print(f"  [GPU] VRAM pico: {vram_mb:.1f} MB")

        # Log
        fila = [epoca, round(train_loss,6), round(train_acc,6),
                round(val_loss,6), round(val_acc,6), round(lr_actual,8), round(t_ep,2)]
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(fila)

        historial.append({
            "epoca": epoca, "train_loss": round(train_loss,6),
            "train_acc": round(train_acc,6), "val_loss": round(val_loss,6),
            "val_acc": round(val_acc,6), "lr": round(lr_actual,8),
            "tiempo_s": round(t_ep,2),
        })
        with open(json_path, "w") as f:
            json.dump(historial, f, indent=2)

        # Early stopping
        if epocas_sin_mejora >= args.patience:
            print(f"\n  [Early Stopping] Sin mejora en {args.patience} epocas. "
                  f"Deteniendo en epoca {epoca}.")
            break

    t_total = (time.time() - t_inicio) / 60

    # Evaluacion final en test con el mejor modelo
    print("\n[Test] Cargando mejor modelo y evaluando...")
    ckpt = torch.load(ckpt_path, map_location=device)
    modelo.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluar(modelo, loader_test, criterio, device, "Test")

    # Matriz de confusion y reporte por clase
    print("[Test] Generando matriz de confusion...")
    from sklearn.metrics import confusion_matrix, classification_report
    CLASES = [
        "airplane","bathtub","bed","bench","bookshelf","bottle","bowl","car",
        "chair","cone","cup","curtain","desk","door","dresser","flower_pot",
        "glass_box","guitar","keyboard","lamp","laptop","mantel","monitor",
        "night_stand","person","piano","plant","radio","range_hood","sink",
        "sofa","stairs","stool","table","tent","toilet","tv_stand","vase",
        "wardrobe","xbox",
    ]
    todas_pred, todas_real = [], []
    modelo.eval()
    with torch.no_grad():
        for grids, etiquetas in loader_test:
            grids = grids.to(device, non_blocking=True)
            logits = modelo(grids)
            todas_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            todas_real.extend(etiquetas.numpy().tolist())
    etiquetas_40 = list(range(len(CLASES)))
    matriz_conf = confusion_matrix(todas_real, todas_pred, labels=etiquetas_40).tolist()
    reporte_cls = classification_report(todas_real, todas_pred,
                                        labels=etiquetas_40,
                                        target_names=CLASES,
                                        output_dict=True, zero_division=0)

    # Medir tiempo de inferencia (Fase 4)
    print("[Fase 4] Midiendo tiempo de inferencia...")
    metricas_inf = medir_tiempo_inferencia(modelo, loader_test, device)

    # Tamano del modelo en MB
    tamano_mb = ckpt_path.stat().st_size / 1e6

    # VRAM pico total
    vram_pico_mb = torch.cuda.max_memory_allocated(device) / 1e6 \
                   if device.type == "cuda" else 0.0

    print("\n" + "=" * 60)
    print(f"  ENTRENAMIENTO COMPLETADO — Net5 R={R}^3")
    print("=" * 60)
    print(f"  Tiempo total    : {t_total:.1f} min")
    print(f"  Mejor Val Acc   : {mejor_val_acc*100:.2f}%  (ep {ckpt['epoca']})")
    print(f"  Test  Acc final : {test_acc*100:.2f}%")
    print(f"  Inf/muestra     : {metricas_inf['tiempo_inferencia_promedio_ms']:.3f} ms")
    print(f"  VRAM pico       : {vram_pico_mb:.1f} MB")
    print(f"  Tamano modelo   : {tamano_mb:.2f} MB")
    print("=" * 60)

    resumen = {
        "resolucion": R,
        "profundidad_octree": 5 if R == 32 else 6,
        "seed": SEED,
        "mejor_val_acc":  round(float(mejor_val_acc), 6),
        "mejor_epoca":    int(ckpt["epoca"]),
        "test_acc":       round(float(test_acc), 6),
        "test_loss":      round(float(test_loss), 6),
        "tiempo_total_min": round(t_total, 2),
        "tamano_modelo_mb": round(tamano_mb, 2),
        "vram_pico_mb":     round(vram_pico_mb, 1),
        "matriz_confusion":          matriz_conf,
        "reporte_clasificacion":     reporte_cls,
        **metricas_inf,
    }

    with open(DIR_RESULTADOS / f"resumen_net5_R{R}{args.tag}.json", "w") as f:
        json.dump(resumen, f, indent=2)

    return resumen


if __name__ == "__main__":
    main()