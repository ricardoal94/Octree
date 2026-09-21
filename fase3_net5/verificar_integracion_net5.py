"""
verificar_integracion_net5.py - Fase 3 (Enfoque profundo) - Objetivo 3
==========================================================================
Prueba de integracion corta de Net5, para R=32 y R=64, que documenta con
valores concretos (no solo pass/fail) cada uno de los puntos requeridos:

  1. Carga de datos reales (octree disperso -> grid denso)
  2. Dimensiones de entrada/salida del modelo
  3. Propagacion hacia adelante (forward)
  4. Propagacion hacia atras (backward / gradientes)
  5. Calculo de la perdida (CrossEntropyLoss)
  6. Actualizacion de pesos (paso del optimizador)
  7. Guardado y recarga del checkpoint del modelo

NO ejecuta un entrenamiento completo: usa un subconjunto pequeno (por
defecto 8 train + 4 val) tomado de la MISMA particion y semilla (seed=42)
del Objetivo 2 (logs/particion_indices.npz).

Genera un informe JSON versionable con todos los valores medidos, para
citar directamente en la documentacion del Objetivo 3.

Uso:
    python fase3_net5/verificar_integracion_net5.py
    python fase3_net5/verificar_integracion_net5.py --n-train 8 --n-val 4
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO / "fase2_octree"))
sys.path.insert(0, str(Path(__file__).parent))

from net5_dataset import OctreeDataset, CLASES
from net5_modelo import Net5Octree, crear_modelo, get_device

PARTICION_PATH = RAIZ_PROYECTO / "logs" / "particion_indices.npz"
DATA_ROOT = RAIZ_PROYECTO / "data"


def _cargar_indices_muestra(n_train: int, n_val: int) -> tuple:
    if not PARTICION_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro la particion del Objetivo 2 en {PARTICION_PATH}. "
            "Ejecute fase1_modelnet40/fase1_setup.py primero."
        )
    particion = np.load(str(PARTICION_PATH))
    return particion["idx_train"][:n_train], particion["idx_val"][:n_val]


def verificar_resolucion(resolucion: int, n_train: int, n_val: int,
                         device: torch.device) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  Verificacion de integracion Net5 -- R={resolucion}^3")
    print(f"{'=' * 60}")

    reporte = {"resolucion": resolucion, "device": str(device)}
    torch.manual_seed(42)

    # ── 1. Carga de datos reales ────────────────────────────────
    t0 = time.time()
    idx_train, idx_val = _cargar_indices_muestra(n_train, n_val)
    raiz_resolucion = DATA_ROOT / f"octrees_{resolucion}"
    if not raiz_resolucion.exists():
        raise FileNotFoundError(f"No existe {raiz_resolucion}")

    ds_train = OctreeDataset(str(raiz_resolucion), resolucion, "train", idx_train)
    ds_val = OctreeDataset(str(raiz_resolucion), resolucion, "val", idx_val)

    grid0, etiqueta0 = ds_train[0]
    lote_train = torch.stack([ds_train[i][0] for i in range(len(ds_train))]).to(device)
    etq_train = torch.stack([ds_train[i][1] for i in range(len(ds_train))]).to(device)
    lote_val = torch.stack([ds_val[i][0] for i in range(len(ds_val))]).to(device)
    etq_val = torch.stack([ds_val[i][1] for i in range(len(ds_val))]).to(device)
    tiempo_carga = time.time() - t0

    reporte["carga_datos"] = {
        "origen_particion": str(PARTICION_PATH),
        "seed": 42,
        "n_train_muestra": len(ds_train),
        "n_val_muestra": len(ds_val),
        "n_clases_dataset": len(CLASES),
        "forma_grid_una_muestra": list(grid0.shape),
        "dtype_grid": str(grid0.dtype),
        "canales": "ocupacion, normal_x, normal_y, normal_z",
        "etiqueta_ejemplo": int(etiqueta0),
        "rango_etiquetas_valido": [0, len(CLASES) - 1],
        "tiempo_carga_s": round(tiempo_carga, 4),
    }
    print(f"[1. Carga de datos] train={len(ds_train)} val={len(ds_val)} "
         f"grid={tuple(grid0.shape)} dtype={grid0.dtype} "
         f"({tiempo_carga:.3f}s)")

    # ── 2. Dimensiones del modelo ───────────────────────────────
    modelo, _ = crear_modelo(resolucion=resolucion, num_clases=40, device=device)
    n_parametros = modelo.contar_parametros()
    reporte["modelo"] = {
        "arquitectura": "Net5Octree (OctNet, Riegler et al. CVPR 2017, "
                        "Tabla 4 -- capacidad fija -- del material suplementario)",
        "resolucion": resolucion,
        "maxpools_aplicados": modelo.n_pool,
        "canales_entrada": 4,
        "num_clases": 40,
        "dropout": modelo.dropout.p,
        "parametros_totales": int(n_parametros),
        "dim_entrada_fc": modelo.dim_fc,
    }
    print(f"[2. Dimensiones]   parametros={n_parametros:,} "
         f"maxpools={modelo.n_pool} dim_fc={modelo.dim_fc}")

    # ── 3. Forward ───────────────────────────────────────────────
    t0 = time.time()
    logits_train = modelo(lote_train)
    tiempo_forward = time.time() - t0

    modelo.eval()
    with torch.no_grad():
        logits_val = modelo(lote_val)
    modelo.train()

    assert logits_train.shape == (len(ds_train), 40)
    assert torch.isfinite(logits_train).all()

    reporte["forward"] = {
        "forma_entrada": list(lote_train.shape),
        "forma_salida": list(logits_train.shape),
        "logits_finitos": bool(torch.isfinite(logits_train).all()),
        "logits_min": round(float(logits_train.detach().min()), 6),
        "logits_max": round(float(logits_train.detach().max()), 6),
        "logits_media": round(float(logits_train.detach().mean()), 6),
        "forma_salida_val": list(logits_val.shape),
        "tiempo_forward_s": round(tiempo_forward, 4),
    }
    print(f"[3. Forward]       entrada={tuple(lote_train.shape)} -> "
         f"salida={tuple(logits_train.shape)} "
         f"logits[min={reporte['forward']['logits_min']}, "
         f"max={reporte['forward']['logits_max']}] ({tiempo_forward:.4f}s)")

    # ── 5. Perdida (se calcula antes del backward que la consume) ─
    criterio = nn.CrossEntropyLoss()
    perdida = criterio(logits_train, etq_train)
    assert torch.isfinite(perdida)
    reporte["perdida"] = {
        "criterio": "CrossEntropyLoss",
        "valor": round(float(perdida.item()), 6),
        "finita": bool(torch.isfinite(perdida)),
    }
    print(f"[5. Perdida]       CrossEntropyLoss={perdida.item():.6f}")

    # ── 4. Backward ─────────────────────────────────────────────
    optimizador = optim.Adam(modelo.parameters(), lr=1e-3)
    optimizador.zero_grad()
    t0 = time.time()
    perdida.backward()
    tiempo_backward = time.time() - t0

    parametros = list(modelo.parameters())
    parametros_con_grad = [p for p in parametros if p.grad is not None]
    norma_grad_total = sum(p.grad.norm().item() ** 2 for p in parametros_con_grad) ** 0.5

    reporte["backward"] = {
        "n_parametros_totales": len(parametros),
        "n_parametros_con_gradiente": len(parametros_con_grad),
        "todos_los_parametros_recibieron_gradiente": len(parametros_con_grad) == len(parametros),
        "norma_gradiente_total_l2": round(norma_grad_total, 6),
        "tiempo_backward_s": round(tiempo_backward, 4),
    }
    print(f"[4. Backward]      {len(parametros_con_grad)}/{len(parametros)} "
         f"parametros con gradiente, ||grad||={norma_grad_total:.4f} "
         f"({tiempo_backward:.4f}s)")

    # ── 6. Paso del optimizador (verifica que los pesos cambian) ──
    pesos_antes = {n: p.detach().clone() for n, p in modelo.named_parameters()}
    optimizador.step()
    cambios = [
        (p.detach() - pesos_antes[n]).norm().item()
        for n, p in modelo.named_parameters()
    ]
    reporte["actualizacion_pesos"] = {
        "optimizador": "Adam",
        "lr": 1e-3,
        "cambio_promedio_norma_l2": round(float(np.mean(cambios)), 8),
        "n_tensores_actualizados": int(sum(c > 0 for c in cambios)),
        "n_tensores_totales": len(cambios),
    }
    print(f"[6. Optimizador]   {reporte['actualizacion_pesos']['n_tensores_actualizados']}"
         f"/{len(cambios)} tensores de pesos cambiaron tras Adam.step()")

    # ── 7. Guardado y recarga del checkpoint ───────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / f"net5_verificacion_R{resolucion}.pth"
        torch.save({
            "epoca": 1,
            "model_state": modelo.state_dict(),
            "optimizer_state": optimizador.state_dict(),
            "mejor_val_acc": 0.0,
            "resolucion": resolucion,
        }, ckpt_path)
        tamano_mb = ckpt_path.stat().st_size / 1e6

        modelo_recargado, _ = crear_modelo(resolucion=resolucion, num_clases=40, device=device)
        ckpt = torch.load(ckpt_path, map_location=device)
        modelo_recargado.load_state_dict(ckpt["model_state"])

        modelo.eval()
        modelo_recargado.eval()
        with torch.no_grad():
            logits_original = modelo(lote_train)
            logits_recargado = modelo_recargado(lote_train)
        modelo.train()
        logits_coinciden = torch.allclose(logits_original, logits_recargado, atol=1e-6)

    reporte["checkpoint"] = {
        "tamano_mb": round(tamano_mb, 2),
        "recarga_exitosa": True,
        "logits_identicos_tras_recarga": bool(logits_coinciden),
    }
    print(f"[7. Checkpoint]    tamano={tamano_mb:.2f} MB, "
         f"logits identicos tras recarga={logits_coinciden}")

    assert logits_coinciden, "Los logits no coinciden tras recargar el checkpoint"
    reporte["estado"] = "OK"
    return reporte


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prueba de integracion corta de Net5 (Objetivo 3), "
                    "con reporte JSON detallado para documentacion",
    )
    parser.add_argument("--n-train", type=int, default=8)
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--resoluciones", type=int, nargs="+", default=[32, 64],
                        choices=[32, 64])
    parser.add_argument(
        "--salida", type=Path,
        default=RAIZ_PROYECTO / "resultados" / "verificacion_integracion_net5.json",
    )
    args = parser.parse_args()

    device = get_device()
    reportes = []
    for resolucion in args.resoluciones:
        reportes.append(verificar_resolucion(resolucion, args.n_train, args.n_val, device))

    parametros_por_resolucion = {r["resolucion"]: r["modelo"]["parametros_totales"]
                                 for r in reportes}
    capacidad_fija = len(set(parametros_por_resolucion.values())) <= 1 \
        if len(parametros_por_resolucion) > 1 else None

    informe = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "objetivo": 3,
            "descripcion": "Prueba de integracion corta de Net5 (sin entrenamiento "
                           "completo): carga de datos, dimensiones, forward, "
                           "backward, perdida y guardado/recarga del modelo.",
            "n_train_muestra": args.n_train,
            "n_val_muestra": args.n_val,
            "particion_origen": str(PARTICION_PATH),
            "seed": 42,
        },
        "verificaciones": reportes,
        "capacidad_fija_entre_resoluciones": {
            "parametros_por_resolucion": {str(k): v for k, v in parametros_por_resolucion.items()},
            "identico": capacidad_fija,
        },
        "todas_las_verificaciones_ok": all(r["estado"] == "OK" for r in reportes),
    }

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(informe, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n{'=' * 60}")
    print(f"  RESUMEN")
    print(f"{'=' * 60}")
    for r in reportes:
        print(f"  R={r['resolucion']:>3}^3 : {r['estado']} "
             f"({r['modelo']['parametros_totales']:,} parametros)")
    if capacidad_fija is not None:
        print(f"  Capacidad fija entre resoluciones: "
             f"{'SI' if capacidad_fija else 'NO'}")
    print(f"  Informe guardado en: {args.salida}")

    return 0 if informe["todas_las_verificaciones_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
