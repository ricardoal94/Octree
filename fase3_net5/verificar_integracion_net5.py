"""Verificacion de la referencia densa de la Tabla 5.

Esta prueba documenta el flujo provisional, pero no valida el objetivo 3:
``nn.Conv3d`` no implementa las operaciones OctNet sobre el grid-octree.
Para R=32 y R=64 documenta con
valores concretos (no solo pass/fail) cada uno de los puntos requeridos:

  1. Carga de datos reales (octree disperso -> grid denso)
  2. Dimensiones de entrada/salida del modelo
  3. Propagacion hacia adelante (forward)
  4. Propagacion hacia atras (backward / gradientes)
  5. Calculo de la perdida (CrossEntropyLoss)
  6. Actualizacion de pesos (paso del optimizador)
  7. Guardado y recarga del checkpoint del modelo

NO ejecuta un entrenamiento completo: usa un subconjunto pequeno (por
defecto 2 train + 2 val) tomado de la MISMA particion estratificada y
semilla (seed=42) del Objetivo 2.

Genera un informe JSON diagnostico que no debe citarse como resultado final.

Uso:
    python fase3_net5/verificar_integracion_net5.py
    python fase3_net5/verificar_integracion_net5.py --n-train 2 --n-val 2
"""

import argparse
import json
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

from net5_dataset import DenseOctreeDataset, CLASES
from net5_modelo import crear_modelo_denso_referencia, get_device
from particion_objetivo2 import (
    cargar_o_crear_particion,
    listar_muestras_octree,
    seleccionar_subconjunto_balanceado,
)

PARTICION_PATH = RAIZ_PROYECTO / "logs" / "particion_objetivo2_modelos.json"
DATA_ROOT = RAIZ_PROYECTO / "data"


def _sincronizar_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _ruta_portable(ruta: Path) -> str:
    ruta = ruta.resolve()
    try:
        return ruta.relative_to(RAIZ_PROYECTO.resolve()).as_posix()
    except ValueError:
        return ruta.name


def _cargar_indices_muestra(
    raiz_resolucion: Path,
    particion_path: Path,
    n_train: int,
    n_val: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    idx_train, idx_val, _, manifest = cargar_o_crear_particion(
        raiz_resolucion, particion_path, seed=42, val_split=0.10,
    )
    _, etiquetas, _ = listar_muestras_octree(raiz_resolucion, "train")
    idx_train = seleccionar_subconjunto_balanceado(
        idx_train, etiquetas, n_train, seed=42,
    )
    idx_val = seleccionar_subconjunto_balanceado(
        idx_val, etiquetas, n_val, seed=42,
    )
    return idx_train, idx_val, manifest


def verificar_resolucion(
    resolucion: int,
    n_train: int,
    n_val: int,
    device: torch.device,
    data_root: Path,
    particion_path: Path,
) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  Diagnostico denso de la Tabla 5 -- R={resolucion}^3")
    print(f"{'=' * 60}")

    reporte = {"resolucion": resolucion, "device": str(device)}
    torch.manual_seed(42)

    # ── 1. Carga de datos reales ────────────────────────────────
    t0 = time.perf_counter()
    raiz_resolucion = data_root / f"octrees_{resolucion}"
    if not raiz_resolucion.exists():
        raise FileNotFoundError(f"No existe {raiz_resolucion}")
    idx_train, idx_val, manifest = _cargar_indices_muestra(
        raiz_resolucion, particion_path, n_train, n_val,
    )

    ds_train = DenseOctreeDataset(
        str(raiz_resolucion), resolucion, "train", idx_train,
    )
    ds_val = DenseOctreeDataset(
        str(raiz_resolucion), resolucion, "val", idx_val,
    )

    grid0, etiqueta0 = ds_train[0]
    lote_train = torch.stack([ds_train[i][0] for i in range(len(ds_train))]).to(device)
    etq_train = torch.stack([ds_train[i][1] for i in range(len(ds_train))]).to(device)
    lote_val = torch.stack([ds_val[i][0] for i in range(len(ds_val))]).to(device)
    tiempo_carga = time.perf_counter() - t0

    reporte["carga_datos"] = {
        "origen_particion": _ruta_portable(particion_path),
        "metodo_particion": manifest["metodo"],
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
    modelo, _ = crear_modelo_denso_referencia(
        resolucion=resolucion, num_clases=40, device=device,
    )
    n_parametros = modelo.contar_parametros()
    reporte["modelo"] = {
        "arquitectura": "Referencia densa de la topologia de la Tabla 5",
        "backend": "dense_reference",
        "valido_como_resultado_objetivo3": False,
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
    _sincronizar_cuda(device)
    t0 = time.perf_counter()
    logits_train = modelo(lote_train)
    _sincronizar_cuda(device)
    tiempo_forward = time.perf_counter() - t0

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
    _sincronizar_cuda(device)
    t0 = time.perf_counter()
    perdida.backward()
    _sincronizar_cuda(device)
    tiempo_backward = time.perf_counter() - t0

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
        ckpt_path = Path(tmp) / f"dense_tabla5_verificacion_R{resolucion}.pth"
        torch.save({
            "epoca": 1,
            "model_state": modelo.state_dict(),
            "optimizer_state": optimizador.state_dict(),
            "mejor_val_acc": 0.0,
            "resolucion": resolucion,
        }, ckpt_path)
        tamano_mb = ckpt_path.stat().st_size / 1e6

        modelo_recargado, _ = crear_modelo_denso_referencia(
            resolucion=resolucion, num_clases=40, device=device,
        )
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
        description="Diagnostico corto de la referencia densa de la Tabla 5; "
                    "no constituye una validacion del Objetivo 3",
    )
    parser.add_argument("--n-train", type=int, default=2)
    parser.add_argument("--n-val", type=int, default=2)
    parser.add_argument("--resoluciones", type=int, nargs="+", default=[32, 64],
                        choices=[32, 64])
    parser.add_argument(
        "--salida", type=Path,
        default=(RAIZ_PROYECTO / "resultados" / "objetivo3" /
                 "verificacion_dense_tabla5.json"),
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--particion-manifest", type=Path, default=PARTICION_PATH,
    )
    args = parser.parse_args()

    device = get_device()
    reportes = []
    for resolucion in args.resoluciones:
        reportes.append(verificar_resolucion(
            resolucion,
            args.n_train,
            args.n_val,
            device,
            args.data_root,
            args.particion_manifest,
        ))

    parametros_por_resolucion = {r["resolucion"]: r["modelo"]["parametros_totales"]
                                 for r in reportes}
    capacidad_fija = len(set(parametros_por_resolucion.values())) <= 1 \
        if len(parametros_por_resolucion) > 1 else None

    informe = {
        "metadata": {
            "schema_name": "dense-table5-diagnostic",
            "timestamp": datetime.now().isoformat(),
            "objetivo": 3,
            "descripcion": "Diagnostico de la referencia densa de la Tabla 5: "
                           "carga, dimensiones, forward, backward, perdida y "
                           "guardado/recarga del modelo.",
            "backend": "dense_reference",
            "valido_como_resultado_objetivo3": False,
            "motivo_no_valido": (
                "nn.Conv3d materializa una grilla densa y no implementa las "
                "operaciones nativas sobre el grid-octree de OctNet."
            ),
            "n_train_muestra": args.n_train,
            "n_val_muestra": args.n_val,
            "particion_origen": _ruta_portable(args.particion_manifest),
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
