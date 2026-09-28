"""Evaluacion comparativa de SVM, Bosque Aleatorio y Net5-Octree (R32 y R64).

Usa los modelos ya entrenados de ``modelos/`` y los octrees de test del
repositorio. Etapas (se pueden ejecutar por separado):

  predicciones  Predice los 2468 objetos de test con los seis modelos.
  tiempos       Latencia por objeto (un proceso, sin workers) sobre una
                muestra estratificada del test, con el mismo protocolo para
                todos los metodos.
  memoria       Memoria RAM (y VRAM para Net5) de la inferencia, cada caso en
                un proceso aislado.
  analisis      Metricas, intervalos de confianza, McNemar, errores y costo
                de los datos por resolucion.

Los modelos, datos y resultados se buscan en la carpeta
de la comparacion (``--carpeta``, por defecto ``resultados/Objetivos_4_y_5_comparacion``
del repositorio), con esta estructura:
    modelos/hce/, modelos/net5/, datos/, resultados/oficiales/

Uso (desde la raiz del repositorio, con su venv):
    python fase4_comparacion/evaluar_comparacion.py
    python fase4_comparacion/evaluar_comparacion.py --etapas tiempos
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import platform
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

# Fijar los hilos antes de importar bibliotecas numericas.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np

RAIZ_REPO = Path(__file__).resolve().parent.parent
CARPETA_DEFECTO = RAIZ_REPO / "resultados" / "Objetivos_4_y_5_comparacion"
CARPETA = CARPETA_DEFECTO
MODELOS = CARPETA / "modelos"
DATOS = CARPETA / "datos"
RESULTADOS = CARPETA / "resultados"
RESOLUCIONES = (32, 64)
SEMILLA = 42


# ──────────────────────────────────────────────────────────────
# Utilidades comunes
# ──────────────────────────────────────────────────────────────

def configurar_carpeta(carpeta: Path) -> None:
    global CARPETA, MODELOS, DATOS, RESULTADOS
    CARPETA = carpeta
    MODELOS, DATOS, RESULTADOS = carpeta / "modelos", carpeta / "datos", carpeta / "resultados"
    for requerida in (MODELOS / "hce", MODELOS / "net5", DATOS, RESULTADOS / "oficiales"):
        if not requerida.is_dir():
            raise FileNotFoundError(f"Falta la carpeta {requerida}")


def configurar_repo(repo: Path) -> None:
    for sub in ("fase1_modelnet40", "fase2_octree", "fase3_hce", "fase3_net5"):
        sys.path.insert(0, str(repo / sub))


def guardar_json(ruta: Path, datos) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")


def muestras_test(repo: Path, R: int):
    from particion_objetivo2 import listar_muestras_octree

    return listar_muestras_octree(repo / "data" / f"octrees_{R}", "test")


def subconjunto_estratificado(etiquetas: np.ndarray, por_clase: int) -> np.ndarray:
    """Hasta ``por_clase`` objetos de test por clase, con semilla fija."""
    rng = np.random.default_rng(SEMILLA)
    elegidos = []
    for clase in np.unique(etiquetas):
        indices = np.flatnonzero(etiquetas == clase)
        elegidos.extend(rng.choice(indices, min(por_clase, len(indices)), replace=False))
    return np.sort(np.asarray(elegidos))


def cargar_hce(R: int, clave: str, un_hilo: bool = False):
    import joblib

    modelo = joblib.load(MODELOS / "hce" / f"hce_{clave}_R{R}.joblib")
    if un_hilo and hasattr(modelo, "n_jobs"):
        modelo.n_jobs = 1  # latencia por objeto; no cambia las predicciones
    return modelo


def cargar_net5(R: int, dispositivo):
    import torch
    from net5_modelo import Net5Octree

    ckpt = torch.load(MODELOS / "net5" / f"net5_octree_mejor_R{R}.pth",
                      map_location="cpu", weights_only=False)
    modelo = Net5Octree(resolucion=R)
    modelo.load_state_dict(ckpt["model_state"])
    return modelo.to(dispositivo).eval()


def entorno() -> dict:
    info = {
        "sistema": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "nucleos_logicos": os.cpu_count(),
        "python": platform.python_version(),
    }
    try:
        import sklearn
        import torch
        info.update({
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "hilos_torch_cpu": torch.get_num_threads(),
        })
    except ImportError:
        pass
    return info


# ──────────────────────────────────────────────────────────────
# Etapa 1: predicciones sobre todo el test
# ──────────────────────────────────────────────────────────────

def etapa_predicciones(repo: Path) -> None:
    import torch
    from torch.utils.data import DataLoader
    from net5_dataset_octree import OctreeNativeDataset, collate_grid_octree

    discrepancias = []
    oficiales = RESULTADOS / "oficiales"
    for R in RESOLUCIONES:
        _, y, ids = muestras_test(repo, R)
        feats = np.load(DATOS / f"hce_features_R{R}.npz")
        if not np.array_equal(feats["y_test"], y):
            raise RuntimeError("El orden de las caracteristicas HCE no coincide con el test")
        pred = {
            "svm": cargar_hce(R, "svm").predict(feats["X_test"]),
            "rf": cargar_hce(R, "rf").predict(feats["X_test"]),
        }

        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        modelo = cargar_net5(R, dispositivo)
        ds = OctreeNativeDataset(repo / "data" / f"octrees_{R}", R, "test", None)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=8,
                            collate_fn=collate_grid_octree, prefetch_factor=1)
        salidas = []
        with torch.no_grad():
            for lote, _ in loader:
                salidas.append(int(modelo(lote.to(dispositivo)).argmax(1).item()))
        pred["net5"] = np.asarray(salidas)

        with (RESULTADOS / f"predicciones_R{R}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "real", "svm", "rf", "net5"])
            for fila in zip(ids, y, pred["svm"], pred["rf"], pred["net5"]):
                w.writerow([fila[0], *(int(v) for v in fila[1:])])

        of_hce = json.loads((oficiales / f"resumen_hce_R{R}.json").read_text(encoding="utf-8"))
        of_net = json.loads((oficiales / f"resumen_net5_octree_R{R}.json").read_text(encoding="utf-8"))
        ref = {"svm": of_hce["svm"]["test_acc"], "rf": of_hce["random_forest"]["test_acc"],
               "net5": of_net["test_acc"]}
        for clave in ("svm", "rf", "net5"):
            acc = float(np.mean(pred[clave] == y))
            marca = "OK" if abs(acc - ref[clave]) < 5e-7 else "DIFERENTE"
            if marca != "OK":
                discrepancias.append(f"R{R} {clave}")
            print(f"R{R} {clave:5s} test acc {acc:.6f} | oficial {ref[clave]:.6f} [{marca}]")

    if discrepancias:
        raise RuntimeError("Predicciones distintas del resultado oficial: " + ", ".join(discrepancias))

# ──────────────────────────────────────────────────────────────
# Etapa 2: latencia por objeto (protocolo comun)
# ──────────────────────────────────────────────────────────────

def etapa_tiempos(repo: Path, por_clase: int) -> None:
    import torch
    from contrato_hce import aplicar_contrato_hce, cargar_contrato_hce
    from grid_octree import convertir_a_grid_octree, precalcular_planes_net5
    from hce_extraccion import extraer_descriptores_hce_desde_datos
    from octnet_backend import LoteGridOctree
    from octree_real import cargar_octree_disperso, reconstruir_octree_desde_npz

    if not torch.cuda.is_available():
        raise RuntimeError("La comparacion de tiempos CPU/GPU requiere CUDA.")
    gpu = torch.device("cuda")
    cpu = torch.device("cpu")
    resumen = {"protocolo": {
        "objetos_por_clase": por_clase,
        "descripcion": (
            "Un solo proceso, sin DataLoader ni workers, un objeto a la vez "
            "(batch 1). Cada objeto parte de su archivo .npz del octree. "
            "Se descartan 10 objetos de calentamiento."
        ),
    }, "entorno": entorno(), "resoluciones": {}}

    for R in RESOLUCIONES:
        rutas, y, ids = muestras_test(repo, R)
        sel = subconjunto_estratificado(y, por_clase)
        calentamiento = [i for i in range(len(rutas)) if i not in set(sel)][:10]
        contrato = cargar_contrato_hce(DATOS / f"contrato_hce_R{R}.json",
                                       resolucion=R, exigir_autorizado=True)
        svm = cargar_hce(R, "svm", un_hilo=True)
        rf = cargar_hce(R, "rf", un_hilo=True)
        net_gpu = cargar_net5(R, gpu)
        net_cpu = cargar_net5(R, cpu)

        filas = []
        for n, i in enumerate(list(calentamiento) + list(sel)):
            ruta = str(rutas[i])
            # HCE: carga + descriptores + contrato, luego cada clasificador
            t0 = time.perf_counter()
            datos = cargar_octree_disperso(ruta)
            t1 = time.perf_counter()
            x = np.asarray(aplicar_contrato_hce(
                extraer_descriptores_hce_desde_datos(datos), contrato,
            ), dtype=np.float64)[None, :]
            t2 = time.perf_counter()
            p_svm = int(svm.predict(x)[0])
            t3 = time.perf_counter()
            p_rf = int(rf.predict(x)[0])
            t4 = time.perf_counter()
            # Net5: carga + grid-octree + planes, luego forward GPU y CPU
            raiz, _, _ = reconstruir_octree_desde_npz(ruta)
            t5 = time.perf_counter()
            muestra = convertir_a_grid_octree(raiz, R)
            precalcular_planes_net5(muestra.geometria)
            t6 = time.perf_counter()
            with torch.no_grad():
                lote = LoteGridOctree.desde_muestras([muestra])
                t7 = time.perf_counter()
                p_gpu = int(net_gpu(lote.to(gpu)).argmax(1).item())  # .item() sincroniza
                t8 = time.perf_counter()
                p_cpu = int(net_cpu(LoteGridOctree.desde_muestras([muestra])).argmax(1).item())
                t9 = time.perf_counter()
            if n < len(calentamiento):
                continue
            ms = lambda a, b: (b - a) * 1000.0
            filas.append({
                "id": ids[i], "real": int(y[i]), "n_hojas": muestra.geometria.n_hojas,
                "bytes_npz": os.path.getsize(ruta),
                "hce_carga_ms": ms(t0, t1), "hce_descriptores_ms": ms(t1, t2),
                "svm_clasificador_ms": ms(t2, t3), "rf_clasificador_ms": ms(t3, t4),
                "net5_carga_ms": ms(t4, t5), "net5_preparacion_ms": ms(t5, t6),
                "net5_forward_gpu_ms": ms(t7, t8), "net5_forward_cpu_ms": ms(t8, t9),
                "pred_svm": p_svm, "pred_rf": p_rf, "pred_net5_gpu": p_gpu,
                "pred_net5_cpu": p_cpu,
            })
            if len(filas) % 50 == 0:
                print(f"  R{R}: {len(filas)}/{len(sel)} objetos", flush=True)

        for f in filas:
            base_hce = f["hce_carga_ms"] + f["hce_descriptores_ms"]
            base_net = f["net5_carga_ms"] + f["net5_preparacion_ms"]
            f["svm_total_ms"] = base_hce + f["svm_clasificador_ms"]
            f["rf_total_ms"] = base_hce + f["rf_clasificador_ms"]
            f["net5_gpu_total_ms"] = base_net + f["net5_forward_gpu_ms"]
            f["net5_cpu_total_ms"] = base_net + f["net5_forward_cpu_ms"]
        with (RESULTADOS / f"tiempos_por_objeto_R{R}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)

        estad = {}
        for clave in filas[0]:
            if clave.endswith("_ms"):
                v = np.asarray([f[clave] for f in filas])
                estad[clave] = {"mediana": float(np.median(v)), "media": float(v.mean()),
                                "p95": float(np.percentile(v, 95))}
        coherencia = float(np.mean([f["pred_net5_gpu"] == f["pred_net5_cpu"] for f in filas]))
        resumen["resoluciones"][str(R)] = {
            "n_objetos": len(filas), "tiempos_ms": estad,
            "net5_gpu_igual_cpu": coherencia,
            "n_hojas_media": float(np.mean([f["n_hojas"] for f in filas])),
        }
        print(f"R{R}: {len(filas)} objetos | mediana total SVM "
              f"{estad['svm_total_ms']['mediana']:.1f} ms, RF {estad['rf_total_ms']['mediana']:.1f}, "
              f"Net5 GPU {estad['net5_gpu_total_ms']['mediana']:.1f}, "
              f"Net5 CPU {estad['net5_cpu_total_ms']['mediana']:.1f}")
    guardar_json(RESULTADOS / "tiempos.json", resumen)


# ──────────────────────────────────────────────────────────────
# Etapa 3: memoria (cada caso en un proceso aislado)
# ──────────────────────────────────────────────────────────────

class _Memoria(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


if sys.platform == "win32":
    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _KERNEL32.GetCurrentProcess.restype = wintypes.HANDLE
    _PSAPI = ctypes.WinDLL("psapi", use_last_error=True)
    _PSAPI.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    _PSAPI.GetProcessMemoryInfo.restype = wintypes.BOOL


def ram_mb() -> float:
    """Working set actual del proceso (Windows)."""
    if sys.platform != "win32":
        raise RuntimeError("La medicion de working set requiere Windows; use --etapas predicciones analisis en otros sistemas.")
    m = _Memoria()
    m.cb = ctypes.sizeof(m)
    if not _PSAPI.GetProcessMemoryInfo(_KERNEL32.GetCurrentProcess(), ctypes.byref(m), m.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return m.WorkingSetSize / 2**20


class Muestreador:
    """Registra el maximo del working set cada 2 ms en un hilo aparte."""

    def __init__(self):
        self.maximo = ram_mb()
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

    def _bucle(self):
        while self._activo:
            self.maximo = max(self.maximo, ram_mb())
            time.sleep(0.002)

    def detener(self) -> float:
        self._activo = False
        self._hilo.join()
        return max(self.maximo, ram_mb())


def medir_memoria_caso(repo: Path, metodo: str, R: int, n_objetos: int) -> dict:
    """Se ejecuta en un proceso nuevo: base -> cargar modelo -> inferencia."""
    rutas, y, _ = muestras_test(repo, R)
    sel = subconjunto_estratificado(y, 40)[:n_objetos]
    if metodo in ("svm", "rf"):
        from contrato_hce import aplicar_contrato_hce, cargar_contrato_hce
        from hce_extraccion import extraer_descriptores_hce_desde_datos
        from octree_real import cargar_octree_disperso
        contrato = cargar_contrato_hce(DATOS / f"contrato_hce_R{R}.json", resolucion=R)
        base = ram_mb()
        mon = Muestreador()
        modelo = cargar_hce(R, metodo, un_hilo=True)
        tras_carga = ram_mb()
        for i in sel:
            datos = cargar_octree_disperso(str(rutas[i]))
            x = np.asarray(aplicar_contrato_hce(
                extraer_descriptores_hce_desde_datos(datos), contrato))[None, :]
            modelo.predict(x)
        pico = mon.detener()
        return {"ram_base_mb": base, "ram_modelo_mb": tras_carga - base,
                "ram_pico_incremento_mb": pico - base, "ram_pico_proceso_mb": pico}

    import torch
    from grid_octree import convertir_a_grid_octree, precalcular_planes_net5
    from octnet_backend import LoteGridOctree
    from octree_real import reconstruir_octree_desde_npz
    dispositivo = torch.device("cuda" if metodo == "net5_gpu" else "cpu")
    base = ram_mb()
    mon = Muestreador()
    if dispositivo.type == "cuda":
        torch.cuda.init()
        torch.zeros(1, device=dispositivo)
    tras_contexto = ram_mb()
    modelo = cargar_net5(R, dispositivo)
    tras_carga = ram_mb()
    if dispositivo.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        vram_modelo = torch.cuda.memory_allocated() / 2**20
    with torch.no_grad():
        for i in sel:
            raiz, _, _ = reconstruir_octree_desde_npz(str(rutas[i]))
            muestra = convertir_a_grid_octree(raiz, R)
            precalcular_planes_net5(muestra.geometria)
            modelo(LoteGridOctree.desde_muestras([muestra]).to(dispositivo)).argmax(1).item()
    pico = mon.detener()
    salida = {"ram_base_mb": base, "ram_contexto_cuda_mb": tras_contexto - base,
              "ram_modelo_mb": tras_carga - tras_contexto,
              "ram_pico_incremento_mb": pico - base, "ram_pico_proceso_mb": pico}
    if dispositivo.type == "cuda":
        salida.update({
            "vram_modelo_mb": vram_modelo,
            "vram_pico_asignada_mb": torch.cuda.max_memory_allocated() / 2**20,
            "vram_pico_reservada_mb": torch.cuda.max_memory_reserved() / 2**20,
        })
    return salida


def etapa_memoria(repo: Path, n_objetos: int) -> None:
    if sys.platform != "win32":
        raise RuntimeError("La etapa memoria requiere Windows.")
    casos = {}
    for R in RESOLUCIONES:
        for metodo in ("svm", "rf", "net5_gpu", "net5_cpu"):
            proc = subprocess.run(
                [sys.executable, __file__, "--repo", str(repo), "--carpeta", str(CARPETA),
                 "--caso-memoria",
                 metodo, str(R), "--objetos-memoria", str(n_objetos)],
                capture_output=True, text=True, check=True,
            )
            ultima = [l for l in proc.stdout.splitlines() if l.startswith("{")][-1]
            casos[f"{metodo}_R{R}"] = json.loads(ultima)
            print(f"  {metodo} R{R}: {casos[f'{metodo}_R{R}']}", flush=True)
    guardar_json(RESULTADOS / "memoria.json", {
        "protocolo": (
            "Cada caso en un proceso nuevo. 'ram_base_mb' es el working set tras "
            "importar las bibliotecas; los incrementos se miden sobre esa base. "
            f"Inferencia de {n_objetos} objetos de test, uno a uno, desde el .npz. "
            "El pico de RAM se muestrea cada 2 ms."
        ),
        "entorno": entorno(), "casos": casos,
    })


# ──────────────────────────────────────────────────────────────
# Etapa 4: analisis de aciertos, errores y costos
# ──────────────────────────────────────────────────────────────

def mcnemar_exacto(a_ok: np.ndarray, b_ok: np.ndarray) -> dict:
    from scipy.stats import binomtest

    solo_a = int(np.sum(a_ok & ~b_ok))
    solo_b = int(np.sum(~a_ok & b_ok))
    n = solo_a + solo_b
    p = binomtest(solo_a, n, 0.5).pvalue if n else 1.0
    return {"solo_primero_acierta": solo_a, "solo_segundo_acierta": solo_b, "p_valor": float(p)}


def etapa_analisis(repo: Path) -> None:
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
    from particion_objetivo2 import CLASES_MODELNET40 as CLASES

    rng = np.random.default_rng(SEMILLA)
    metricas, por_clase, confusiones, acuerdo, pruebas = {}, {}, {}, {}, []
    preds = {}
    for R in RESOLUCIONES:
        with (RESULTADOS / f"predicciones_R{R}.csv").open(encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
        y = np.asarray([int(r["real"]) for r in filas])
        for m in ("svm", "rf", "net5"):
            p = np.asarray([int(r[m]) for r in filas])
            preds[(m, R)] = (y, p)
            ok = p == y
            boot = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(2000)]
            metricas[f"{m}_R{R}"] = {
                "aciertos": int(ok.sum()), "errores": int((~ok).sum()), "n": len(ok),
                "exactitud": float(ok.mean()),
                "exactitud_ic95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "exactitud_balanceada": float(balanced_accuracy_score(y, p)),
                "f1_macro": float(f1_score(y, p, average="macro")),
                "f1_ponderado": float(f1_score(y, p, average="weighted")),
            }
            por_clase[f"{m}_R{R}"] = recall_score(y, p, average=None, labels=range(40)).tolist()
            pares = {}
            for real, pr in zip(y[~ok], p[~ok]):
                pares[(real, pr)] = pares.get((real, pr), 0) + 1
            confusiones[f"{m}_R{R}"] = [
                {"real": CLASES[a], "predicha": CLASES[b], "n": n}
                for (a, b), n in sorted(pares.items(), key=lambda t: -t[1])[:8]
            ]
        ok = {m: preds[(m, R)][1] == y for m in ("svm", "rf", "net5")}
        k = ok["svm"].astype(int) + ok["rf"] + ok["net5"]
        acuerdo[str(R)] = {
            "los_3_aciertan": int(np.sum(k == 3)), "2_aciertan": int(np.sum(k == 2)),
            "1_acierta": int(np.sum(k == 1)), "ninguno_acierta": int(np.sum(k == 0)),
            "solo_net5_acierta": int(np.sum(ok["net5"] & ~ok["svm"] & ~ok["rf"])),
            "solo_hce_acierta": int(np.sum(~ok["net5"] & (ok["svm"] | ok["rf"]))),
            "oraculo_alguno_acierta": float(np.mean(k > 0)),
        }
        for a, b in (("net5", "svm"), ("net5", "rf"), ("rf", "svm")):
            pruebas.append({"comparacion": f"{a} vs {b}", "resolucion": R,
                            **mcnemar_exacto(ok[a], ok[b])})
    for m in ("svm", "rf", "net5"):
        y64, p64 = preds[(m, 64)]
        y32, p32 = preds[(m, 32)]
        if not np.array_equal(y32, y64):
            raise RuntimeError("R32 y R64 no tienen el mismo orden de test")
        pruebas.append({"comparacion": f"{m}: R64 vs R32", "resolucion": "64 vs 32",
                        **mcnemar_exacto(p64 == y64, p32 == y32)})

    # Costo de los datos por resolucion (octrees de todo ModelNet40)
    datos = {}
    for R in RESOLUCIONES:
        tam = [p.stat().st_size for p in (repo / "data" / f"octrees_{R}").rglob("*.npz")]
        datos[str(R)] = {"archivos": len(tam), "total_mb": sum(tam) / 1e6,
                         "media_kb": float(np.mean(tam)) / 1024}

    # Tamano y complejidad de los modelos
    import joblib
    import torch
    modelos = {}
    for R in RESOLUCIONES:
        svm = joblib.load(MODELOS / "hce" / f"hce_svm_R{R}.joblib")
        rf = joblib.load(MODELOS / "hce" / f"hce_rf_R{R}.joblib")
        ck = torch.load(MODELOS / "net5" / f"net5_octree_mejor_R{R}.pth",
                        map_location="cpu", weights_only=False)
        modelos[str(R)] = {
            "svm": {"archivo_mb": (MODELOS / "hce" / f"hce_svm_R{R}.joblib").stat().st_size / 1e6,
                    "vectores_soporte": int(svm.named_steps["svm"].n_support_.sum()),
                    "n_caracteristicas": int(svm.n_features_in_)},
            "rf": {"archivo_mb": (MODELOS / "hce" / f"hce_rf_R{R}.joblib").stat().st_size / 1e6,
                   "arboles": len(rf.estimators_),
                   "nodos": int(sum(e.tree_.node_count for e in rf.estimators_)),
                   "n_caracteristicas": int(rf.n_features_in_)},
            "net5": {"archivo_mb": (MODELOS / "net5" / f"net5_octree_mejor_R{R}.pth").stat().st_size / 1e6,
                     "parametros": int(sum(v.numel() for v in ck["model_state"].values()))},
        }

    guardar_json(RESULTADOS / "analisis.json", {
        "metricas": metricas, "acuerdo_entre_metodos": acuerdo, "mcnemar": pruebas,
        "confusiones_principales": confusiones, "datos_por_resolucion": datos,
        "modelos": modelos,
    })
    with (RESULTADOS / "recall_por_clase.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        claves = list(por_clase)
        w.writerow(["clase", *claves])
        for c, nombre in enumerate(CLASES):
            w.writerow([nombre, *(round(por_clase[k][c], 4) for k in claves)])
    for k, v in metricas.items():
        print(f"{k:10s} acc {v['exactitud']:.4f} IC95 [{v['exactitud_ic95'][0]:.4f}, "
              f"{v['exactitud_ic95'][1]:.4f}] | bal {v['exactitud_balanceada']:.4f} | "
              f"F1 macro {v['f1_macro']:.4f}")


# ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", type=Path, default=RAIZ_REPO,
                        help="Raiz del repositorio Tesis (con data/octrees_32 y _64)")
    parser.add_argument("--carpeta", type=Path, default=CARPETA_DEFECTO,
                        help="Carpeta de la comparacion (modelos/, datos/, resultados/)")
    parser.add_argument("--etapas", nargs="+",
                        default=["predicciones", "tiempos", "memoria", "analisis"],
                        choices=["predicciones", "tiempos", "memoria", "analisis"])
    parser.add_argument("--objetos-por-clase", type=int, default=10)
    parser.add_argument("--objetos-memoria", type=int, default=100)
    parser.add_argument("--caso-memoria", nargs=2, metavar=("METODO", "R"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    configurar_carpeta(args.carpeta.resolve())
    configurar_repo(repo)
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    if args.caso_memoria:
        metodo, R = args.caso_memoria
        print(json.dumps(medir_memoria_caso(repo, metodo, int(R), args.objetos_memoria)))
        return 0

    RESULTADOS.mkdir(exist_ok=True)
    for etapa in args.etapas:
        print(f"\n=== Etapa: {etapa}", flush=True)
        if etapa == "predicciones":
            etapa_predicciones(repo)
        elif etapa == "tiempos":
            etapa_tiempos(repo, args.objetos_por_clase)
        elif etapa == "memoria":
            etapa_memoria(repo, args.objetos_memoria)
        else:
            etapa_analisis(repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
