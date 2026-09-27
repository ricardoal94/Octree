"""Genera las tablas de la comparacion (CSV y Markdown) a partir de resultados/.

Uso (desde la raiz del repositorio):
    python fase4_comparacion/tablas_comparacion.py [--carpeta RUTA]
"""

import csv
import json
from pathlib import Path

CARPETA_DEFECTO = Path(__file__).resolve().parent.parent / "resultados" / "Objetivos_4_y_5_comparacion"
RES = CARPETA_DEFECTO / "resultados"
TAB = CARPETA_DEFECTO / "tablas"


def configurar_carpeta() -> None:
    """Lee --carpeta y fija las rutas de entrada y salida."""
    import argparse
    global RES, TAB
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--carpeta", type=Path, default=CARPETA_DEFECTO,
                        help="Carpeta de la comparacion (con resultados/)")
    carpeta = parser.parse_args().carpeta.resolve()
    RES, TAB = carpeta / "resultados", carpeta / "tablas"
NOMBRE = {"svm": "SVM", "rf": "Bosque aleatorio", "net5": "Net5-Octree"}


def cargar(nombre):
    return json.loads((RES / nombre).read_text(encoding="utf-8"))


def escribir(nombre, cabecera, filas):
    TAB.mkdir(exist_ok=True)
    with (TAB / f"{nombre}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cabecera)
        w.writerows(filas)
    lineas = ["| " + " | ".join(cabecera) + " |", "|" + "---|" * len(cabecera)]
    lineas += ["| " + " | ".join(str(c) for c in fila) + " |" for fila in filas]
    (TAB / f"{nombre}.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main():
    configurar_carpeta()
    an, ti, me = cargar("analisis.json"), cargar("tiempos.json"), cargar("memoria.json")
    of = {R: {"hce": cargar(f"oficiales/resumen_hce_R{R}.json"),
              "net5": cargar(f"oficiales/resumen_net5_octree_R{R}.json")} for R in (32, 64)}

    # 1. Aciertos
    filas = []
    for m in ("svm", "rf", "net5"):
        for R in (32, 64):
            x = an["metricas"][f"{m}_R{R}"]
            lo, hi = x["exactitud_ic95"]
            filas.append([NOMBRE[m], f"R{R}", x["aciertos"], x["errores"],
                          f"{x['exactitud']*100:.2f}", f"[{lo*100:.1f}, {hi*100:.1f}]",
                          f"{x['exactitud_balanceada']*100:.2f}", f"{x['f1_macro']*100:.2f}",
                          f"{x['f1_ponderado']*100:.2f}"])
    escribir("t1_aciertos", ["Método", "Res.", "Aciertos", "Errores", "Exactitud (%)",
                             "IC 95 %", "Exactitud balanceada (%)", "F1 macro (%)",
                             "F1 ponderado (%)"], filas)

    # 2. McNemar
    filas = []
    for p in an["mcnemar"]:
        a, b = p["comparacion"].split(" vs ") if ":" not in p["comparacion"] else (None, None)
        if a:
            nombre = f"{NOMBRE[a]} vs {NOMBRE[b.strip()]} (R{p['resolucion']})"
        else:
            m = p["comparacion"].split(":")[0]
            nombre = f"{NOMBRE[m]}: R64 vs R32"
        filas.append([nombre, p["solo_primero_acierta"], p["solo_segundo_acierta"],
                      f"{p['p_valor']:.2g}", "sí" if p["p_valor"] < 0.05 else "no"])
    escribir("t2_mcnemar", ["Comparación (A vs B)", "Solo A acierta", "Solo B acierta",
                            "p (McNemar exacto)", "Diferencia significativa (α = 0.05)"], filas)

    # 3. Tiempos por objeto
    casos = [("SVM", "hce_carga_ms", "hce_descriptores_ms", "svm_clasificador_ms", "svm_total_ms"),
             ("Bosque aleatorio", "hce_carga_ms", "hce_descriptores_ms", "rf_clasificador_ms", "rf_total_ms"),
             ("Net5-Octree (GPU)", "net5_carga_ms", "net5_preparacion_ms", "net5_forward_gpu_ms", "net5_gpu_total_ms"),
             ("Net5-Octree (CPU, 1 hilo)", "net5_carga_ms", "net5_preparacion_ms", "net5_forward_cpu_ms", "net5_cpu_total_ms")]
    filas = []
    for nombre, carga, prep, clf, total in casos:
        for R in (32, 64):
            t = ti["resoluciones"][str(R)]["tiempos_ms"]
            filas.append([nombre, f"R{R}", f"{t[carga]['mediana']:.1f}", f"{t[prep]['mediana']:.1f}",
                          f"{t[clf]['mediana']:.2f}", f"{t[total]['mediana']:.1f}",
                          f"{t[total]['p95']:.1f}"])
    escribir("t3_tiempos", ["Método", "Res.", "Carga (ms)", "Descriptores / preparación (ms)",
                            "Clasificador / red (ms)", "Total mediana (ms)", "Total p95 (ms)"], filas)

    # 4. Memoria y tamano
    filas = []
    for nombre, caso, m in (("SVM", "svm", "svm"), ("Bosque aleatorio", "rf", "rf"),
                            ("Net5-Octree (GPU)", "net5_gpu", "net5"), ("Net5-Octree (CPU)", "net5_cpu", "net5")):
        for R in (32, 64):
            c = me["casos"][f"{caso}_R{R}"]
            mod = an["modelos"][str(R)][m]
            complejidad = {"svm": f"{mod.get('vectores_soporte', '')} vectores de soporte",
                           "rf": f"{mod.get('arboles', '')} árboles, {mod.get('nodos', 0):,} nodos",
                           "net5": f"{mod.get('parametros', 0):,} parámetros"}[m]
            filas.append([nombre, f"R{R}", f"{mod['archivo_mb']:.1f}", complejidad,
                          f"{c['ram_modelo_mb']:.0f}", f"{c['ram_pico_incremento_mb']:.0f}",
                          f"{c['ram_pico_proceso_mb']:.0f}",
                          f"{c['vram_pico_asignada_mb']:.0f} / {c['vram_pico_reservada_mb']:.0f}"
                          if "vram_pico_asignada_mb" in c else "—"])
    escribir("t4_memoria_tamano", ["Método", "Res.", "Archivo (MB)", "Complejidad",
                                   "RAM adicional al cargar el modelo (MiB)",
                                   "RAM adicional pico en inferencia (MiB)",
                                   "RAM total pico del proceso (MiB)",
                                   "VRAM pico asignada / reservada (MiB)"], filas)

    # 5. Costo de entrenamiento y de datos
    filas = []
    for R in (32, 64):
        h, n = of[R]["hce"], of[R]["net5"]
        extr = ti["resoluciones"][str(R)]["tiempos_ms"]
        est_min = (extr["hce_carga_ms"]["media"] + extr["hce_descriptores_ms"]["media"]) * 9843 / 60000
        d = an["datos_por_resolucion"][str(R)]
        filas.append([f"R{R}", f"{d['total_mb']:.0f} MB ({d['media_kb']:.1f} KB/objeto)",
                      f"~{est_min:.1f} min (estimado)",
                      f"{h['svm']['tiempo_busqueda_hiperparam_s']:.1f} s",
                      f"{h['random_forest']['tiempo_entrenamiento_s']:.2f} s",
                      f"{n['tiempo_total_min']:.0f} min ({n['entrenamiento']['epocas_completadas']} épocas, mejor {n['mejor_epoca']})",
                      f"{n['vram_pico_mb']:.0f} MB"])
    escribir("t5_entrenamiento_datos", ["Res.", "Octrees de ModelNet40 en disco",
                                        "Extracción HCE del train", "SVM (búsqueda de hiperparámetros)",
                                        "Bosque aleatorio (ajuste)", "Net5-Octree (entrenamiento)",
                                        "VRAM pico entrenamiento Net5"], filas)

    # 6. Acuerdo entre metodos
    filas = []
    for R in (32, 64):
        a = an["acuerdo_entre_metodos"][str(R)]
        filas.append([f"R{R}", a["los_3_aciertan"], a["2_aciertan"], a["1_acierta"],
                      a["ninguno_acierta"], a["solo_net5_acierta"], a["solo_hce_acierta"],
                      f"{a['oraculo_alguno_acierta']*100:.1f}"])
    escribir("t6_acuerdo", ["Res.", "Los 3 aciertan", "2 aciertan", "1 acierta", "Ninguno",
                            "Solo Net5 acierta", "Net5 falla y algún HCE acierta",
                            "Al menos uno acierta (%)"], filas)

    # 7. Confusiones principales
    filas = []
    for clave, lista in an["confusiones_principales"].items():
        m, R = clave.rsplit("_", 1)
        filas.append([NOMBRE[m], R, "; ".join(f"{c['real']}→{c['predicha']} ({c['n']})" for c in lista[:4])])
    escribir("t7_confusiones", ["Método", "Res.", "Confusiones más frecuentes (real→predicha, n)"], filas)
    print("Tablas en", TAB)


if __name__ == "__main__":
    main()
