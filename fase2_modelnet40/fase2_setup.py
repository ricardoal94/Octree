"""
fase2_setup.py
==============
Punto de entrada de la Fase 2: carga el dataset ModelNet40,
instancia Net5 y verifica que todo funciona antes de entrenar.

Uso:
    python fase2_setup.py
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Importar modulos de la Fase 1
sys.path.insert(0, str(Path(__file__).parent.parent / "fase1_modelnet40"))
from fase1_setup import set_global_seed, particionar_dataset, cargar_config

# Importar modulos de la Fase 2
from dataset import crear_dataloaders
from modelo  import crear_modelo, get_device

# ── Configuracion ──────────────────────────────────────────────
RAIZ_DATASET = r"C:\Users\ricar\Documents\Codigos\Tesis\Dataset\ModelNet40"
RAIZ_CONFIG  = Path(__file__).parent.parent / "fase1_modelnet40" / "config.yaml"
SEED         = 42


def main():
    print("=" * 60)
    print("  FASE 2: DATASET + MODELO Net5")
    print("=" * 60)

    # 1. Semilla global
    set_global_seed(SEED)

    # 2. Detectar dispositivo
    device = get_device()

    # 3. Cargar configuracion
    cfg = cargar_config(str(RAIZ_CONFIG))
    n_puntos   = cfg["dataset"]["num_points"]
    batch_size = 32   # RTX 5070 puede manejar batch completo

    # 4. Recuperar particion reproducible
    log_path = Path(__file__).parent.parent / "fase1_modelnet40" / "logs" / "particion_indices.npz"
    if log_path.exists():
        data      = np.load(str(log_path))
        idx_train = data["idx_train"]
        idx_val   = data["idx_val"]
        print(f"[Particion] Indices cargados desde: {log_path}")
    else:
        particion = particionar_dataset(seed=SEED)
        idx_train = np.array(particion["train"]["indices"])
        idx_val   = np.array(particion["val"]["indices"])
        print("[Particion] Indices generados (archivo .npz no encontrado)")

    # 5. Crear DataLoaders
    print("\n[Dataset] Inicializando DataLoaders...")
    loader_train, loader_val, loader_test = crear_dataloaders(
        raiz_dataset=RAIZ_DATASET,
        idx_train=idx_train,
        idx_val=idx_val,
        n_puntos=n_puntos,
        batch_size=batch_size,
        num_workers=0,   # 0 en Windows
        seed=SEED,
    )

    # 6. Verificar un batch — mover a GPU
    print("\n[Verificacion] Cargando primer batch de train...")
    puntos, etiquetas = next(iter(loader_train))
    puntos    = puntos.to(device)
    etiquetas = etiquetas.to(device)
    print(f"  Puntos shape  : {puntos.shape}")
    print(f"  Dispositivo   : {puntos.device}")

    # 7. Crear modelo Net5 en GPU
    modelo, device = crear_modelo(
        num_clases=cfg["modelo"]["num_clases"],
        dropout=cfg["modelo"]["dropout"],
        device=device,
    )

    # 8. Forward pass de prueba en GPU
    print("\n[Verificacion] Forward pass en GPU...")
    modelo.eval()
    with torch.no_grad():
        logits = modelo(puntos)
    predicciones = logits.argmax(dim=1)
    print(f"  Logits shape  : {logits.shape}")
    print(f"  Predicciones  : {predicciones.cpu().tolist()}")

    # 9. Mostrar memoria GPU usada
    if device.type == "cuda":
        mem_usada = torch.cuda.memory_allocated(device) / 1e6
        mem_total = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"\n[GPU] Memoria usada : {mem_usada:.1f} MB / {mem_total:.1f} GB")

    # 10. Resumen
    print("\n" + "=" * 60)
    print("  Fase 2 completada exitosamente.")
    print(f"  Dispositivo   : {device}")
    print(f"  Train batches : {len(loader_train)}")
    print(f"  Val   batches : {len(loader_val)}")
    print(f"  Test  batches : {len(loader_test)}")
    print("  Proxima fase  : Entrenamiento de Net5")
    print("=" * 60)

    return modelo, device, loader_train, loader_val, loader_test


if __name__ == "__main__":
    main()
