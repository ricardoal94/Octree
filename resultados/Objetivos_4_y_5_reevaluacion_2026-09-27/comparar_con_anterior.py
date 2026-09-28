"""Compara una reevaluacion con los resultados publicados.

Por defecto compara esta carpeta (resultados/Objetivos_4_y_5_reevaluacion_2026-09-27)
con la publicada (resultados/Objetivos_4_y_5_comparacion). No modifica ninguna de
las dos; escribe diferencias.json y diferencias.md en --salida.

Uso (desde la raiz del repositorio):
    python resultados/Objetivos_4_y_5_reevaluacion_2026-09-27/comparar_con_anterior.py

En la ejecucion original se uso con la disposicion externa:
    --nuevo carpeta --anterior anterior_publicado --salida .
"""

import argparse
import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
NUEVO = BASE
ANTERIOR = BASE.parent / "Objetivos_4_y_5_comparacion"
SALIDA = BASE
METODOS = ("svm", "rf", "net5")


def leer_json(carpeta, nombre):
    return json.loads((carpeta / nombre).read_text(encoding="utf-8"))


def leer_csv(carpeta, nombre):
    with (carpeta / nombre).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    global NUEVO, ANTERIOR, SALIDA
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nuevo", type=Path, default=NUEVO, help="Carpeta de la reevaluacion")
    parser.add_argument("--anterior", type=Path, default=ANTERIOR, help="Carpeta publicada")
    parser.add_argument("--salida", type=Path, default=SALIDA, help="Donde escribir diferencias.*")
    args = parser.parse_args()
    NUEVO, ANTERIOR, SALIDA = args.nuevo.resolve(), args.anterior.resolve(), args.salida.resolve()

    salida = {"predicciones": {}, "analisis_identico": {}, "tiempos": {}, "memoria": {}}

    # 1. Predicciones objeto por objeto
    for R in (32, 64):
        viejo = {f["id"]: f for f in leer_csv(ANTERIOR, f"resultados/predicciones_R{R}.csv")}
        nuevo = {f["id"]: f for f in leer_csv(NUEVO, f"resultados/predicciones_R{R}.csv")}
        salida["predicciones"][str(R)] = {
            "mismos_ids": set(viejo) == set(nuevo),
            "n": len(nuevo),
            "diferencias": {m: sum(viejo[i][m] != nuevo[i][m] for i in nuevo) for m in METODOS},
        }

    # 2. Analisis: debe ser identico (mismos modelos, mismos datos, semilla fija)
    an_v, an_n = leer_json(ANTERIOR, "resultados/analisis.json"), leer_json(NUEVO, "resultados/analisis.json")
    for clave in an_v:
        salida["analisis_identico"][clave] = an_v[clave] == an_n.get(clave)
    rc_v = leer_csv(ANTERIOR, "resultados/recall_por_clase.csv")
    rc_n = leer_csv(NUEVO, "resultados/recall_por_clase.csv")
    salida["analisis_identico"]["recall_por_clase"] = rc_v == rc_n

    # 3. Tiempos: misma muestra y comparacion de medianas y p95
    ti_v, ti_n = leer_json(ANTERIOR, "resultados/tiempos.json"), leer_json(NUEVO, "resultados/tiempos.json")
    for R in (32, 64):
        ids_v = [f["id"] for f in leer_csv(ANTERIOR, f"resultados/tiempos_por_objeto_R{R}.csv")]
        ids_n = [f["id"] for f in leer_csv(NUEVO, f"resultados/tiempos_por_objeto_R{R}.csv")]
        por_campo = {}
        for campo, v in ti_v["resoluciones"][str(R)]["tiempos_ms"].items():
            n = ti_n["resoluciones"][str(R)]["tiempos_ms"][campo]
            por_campo[campo] = {
                "mediana_anterior": v["mediana"], "mediana_nueva": n["mediana"],
                "cambio_mediana_pct": 100 * (n["mediana"] - v["mediana"]) / v["mediana"],
                "p95_anterior": v["p95"], "p95_nuevo": n["p95"],
                "cambio_p95_pct": 100 * (n["p95"] - v["p95"]) / v["p95"],
            }
        salida["tiempos"][str(R)] = {"misma_muestra": ids_v == ids_n, "campos": por_campo}
    salida["tiempos"]["entorno_anterior"] = ti_v.get("entorno")
    salida["tiempos"]["entorno_nuevo"] = ti_n.get("entorno")

    # 4. Memoria
    me_v, me_n = leer_json(ANTERIOR, "resultados/memoria.json"), leer_json(NUEVO, "resultados/memoria.json")
    for caso, v in me_v["casos"].items():
        n = me_n["casos"][caso]
        salida["memoria"][caso] = {
            k: {"anterior": v[k], "nuevo": n[k], "diferencia": n[k] - v[k]} for k in v
        }

    (SALIDA / "diferencias.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")

    # Resumen legible
    lineas = ["# Diferencias entre la reevaluación y los resultados publicados", ""]
    lineas += ["## Predicciones (2,468 objetos de test por resolución)", "",
               "| Resolución | Mismos IDs | SVM | Bosque | Net5 |", "|---|---|---|---|---|"]
    for R, p in salida["predicciones"].items():
        d = p["diferencias"]
        lineas.append(f"| R{R} | {'sí' if p['mismos_ids'] else 'no'} | {d['svm']} | {d['rf']} | {d['net5']} |")
    lineas += ["", "## Análisis de aciertos (debe ser idéntico)", "", "| Componente | Idéntico |", "|---|---|"]
    lineas += [f"| {k} | {'sí' if v else '**no**'} |" for k, v in salida["analisis_identico"].items()]
    nombres = {"svm_total_ms": "SVM total", "rf_total_ms": "Bosque total",
               "net5_gpu_total_ms": "Net5 GPU total", "net5_cpu_total_ms": "Net5 CPU total",
               "hce_carga_ms": "HCE carga", "hce_descriptores_ms": "HCE descriptores",
               "svm_clasificador_ms": "SVM clasificador", "rf_clasificador_ms": "Bosque clasificador",
               "net5_carga_ms": "Net5 carga", "net5_preparacion_ms": "Net5 preparación",
               "net5_forward_gpu_ms": "Net5 forward GPU", "net5_forward_cpu_ms": "Net5 forward CPU"}
    lineas += ["", "## Tiempos por objeto (mediana, ms)", "",
               "| Componente | R32 anterior | R32 nuevo | Cambio | R64 anterior | R64 nuevo | Cambio |",
               "|---|---|---|---|---|---|---|"]
    for campo, nombre in nombres.items():
        a, b = salida["tiempos"]["32"]["campos"][campo], salida["tiempos"]["64"]["campos"][campo]
        lineas.append(f"| {nombre} | {a['mediana_anterior']:.2f} | {a['mediana_nueva']:.2f} | "
                      f"{a['cambio_mediana_pct']:+.1f} % | {b['mediana_anterior']:.2f} | "
                      f"{b['mediana_nueva']:.2f} | {b['cambio_mediana_pct']:+.1f} % |")
    lineas.append("")
    lineas.append(f"Misma muestra de 400 objetos: R32 {'sí' if salida['tiempos']['32']['misma_muestra'] else 'no'}, "
                  f"R64 {'sí' if salida['tiempos']['64']['misma_muestra'] else 'no'}.")
    lineas += ["", "## Memoria (MiB)", "",
               "| Caso | RAM adicional anterior | RAM adicional nueva | RAM total anterior | RAM total nueva | VRAM asignada anterior | VRAM asignada nueva |",
               "|---|---|---|---|---|---|---|"]
    for caso, v in salida["memoria"].items():
        vram_a = v.get("vram_pico_asignada_mb", {}).get("anterior")
        vram_n = v.get("vram_pico_asignada_mb", {}).get("nuevo")
        lineas.append(
            f"| {caso} | {v['ram_pico_incremento_mb']['anterior']:.0f} | {v['ram_pico_incremento_mb']['nuevo']:.0f} | "
            f"{v['ram_pico_proceso_mb']['anterior']:.0f} | {v['ram_pico_proceso_mb']['nuevo']:.0f} | "
            f"{'—' if vram_a is None else f'{vram_a:.0f}'} | {'—' if vram_n is None else f'{vram_n:.0f}'} |")
    (SALIDA / "diferencias.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
