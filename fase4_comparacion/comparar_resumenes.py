"""Compara resultados publicados sin cargar pesos ni volver a entrenar.

Uso: python fase4_comparacion/comparar_resumenes.py
Solo usa la biblioteca estandar. Los tiempos se conservan con su alcance;
no calcula aceleraciones entre protocolos de medicion distintos.
"""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verificar_metricas(resultado):
    matriz = resultado["matriz_confusion"]
    if len(matriz) != 40 or any(len(fila) != 40 for fila in matriz):
        raise ValueError("La matriz debe tener 40 filas y columnas")
    if any(type(x) is not int or x < 0 for fila in matriz for x in fila):
        raise ValueError("La matriz debe contener conteos enteros no negativos")
    soportes = [sum(fila) for fila in matriz]
    total = sum(soportes)
    if total != 2468:
        raise ValueError(f"Test incompleto: {total} muestras")
    aciertos = sum(matriz[i][i] for i in range(40))
    f1 = []
    for i, soporte in enumerate(soportes):
        predichos = sum(fila[i] for fila in matriz)
        f1.append(2 * matriz[i][i] / (soporte + predichos)
                  if soporte + predichos else 0.0)
    macro = sum(f1) / len(f1)
    if abs(aciertos / total - resultado["test_acc"]) > 0.00000051:
        raise ValueError("Accuracy incompatible con la matriz")
    if abs(macro - resultado["reporte_clasificacion"]["macro avg"]["f1-score"]) > 1e-10:
        raise ValueError("F1 macro incompatible con la matriz")
    return total, aciertos, macro, soportes


def generar():
    filas, fuentes, particiones = [], {}, []
    soporte_referencia = None
    for resolucion in (32, 64):
        ruta_hce = ROOT / f"resultados/objetivo2/resumen_hce_R{resolucion}.json"
        ruta_net = ROOT / f"resultados/objetivo3/resultados/resumen_net5_octree_R{resolucion}.json"
        hce, net = (json.loads(p.read_text()) for p in (ruta_hce, ruta_net))
        for ruta in (ruta_hce, ruta_net):
            fuentes[ruta.relative_to(ROOT).as_posix()] = hashlib.sha256(ruta.read_bytes()).hexdigest()
        conteos = (hce["n_train"], hce["n_val"], hce["n_test"])
        if conteos != tuple(net["particion"][k] for k in ("n_train_usado", "n_val_usado", "n_test_usado")):
            raise ValueError("Difieren los tamanos de particion")
        if hce["seed"] != net["seed"]:
            raise ValueError("Difieren las semillas")
        particiones.append(net["entrenamiento"]["particion_sha256"])
        historial = ROOT / f"resultados/objetivo3/logs/net5_octree_historial_R{resolucion}.csv"
        with historial.open(newline="") as f:
            epocas = list(csv.DictReader(f))
        mejor = max(epocas, key=lambda e: float(e["val_acc"]))
        if (int(mejor["epoca"]) != net["mejor_epoca"]
                or float(mejor["val_acc"]) != net["mejor_val_acc"]
                or len(epocas) != net["entrenamiento"]["epocas_completadas"]):
            raise ValueError("El historial no coincide con el resumen")
        if len(epocas) - int(mejor["epoca"]) != net["configuracion"]["patience"]:
            raise ValueError("Parada temprana inesperada en estas corridas")
        for nombre, resultado in (("SVM-HCE", hce["svm"]),
                                  ("RandomForest-HCE", hce["random_forest"]),
                                  ("Net5-Octree", net)):
            total, aciertos, macro, soportes = verificar_metricas(resultado)
            if soporte_referencia is not None and soportes != soporte_referencia:
                raise ValueError("Difieren los conteos por clase")
            soporte_referencia = soportes
            filas.append({
                "modelo": nombre, "resolucion": resolucion,
                "n_test": total, "aciertos": aciertos,
                "accuracy_pct": round(aciertos / total * 100, 6),
                "f1_macro_pct": round(macro * 100, 6),
                "inferencia_reportada_ms": resultado["tiempo_inferencia_promedio_ms"],
                "alcance_tiempo": ("forward incl. planes; GPU; batch=1" if nombre == "Net5-Octree"
                                   else "predict sobre descriptores precalculados; CPU; lote completo"),
                "pipeline_ms": resultado.get("tiempo_pipeline_promedio_ms", ""),
                "tamano_modelo_mb": resultado["tamano_modelo_mb"],
            })
    if len(set(particiones)) != 1:
        raise ValueError("R32 y R64 no registran la misma particion")
    destino = ROOT / "resultados/objetivo4"
    destino.mkdir(parents=True, exist_ok=True)
    with (destino / "comparacion_publicada.csv").open("w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    evidencia = {
        "alcance": "Auditoria de registros; no reejecuta modelos ni valida identidad de muestras",
        "fuentes_sha256": fuentes,
        "particion_sha256_net5": particiones[0],
        "filas": filas,
        "limitaciones": [
            "El manifiesto por ID y los checkpoints no estan versionados en el PR.",
            "Coincidir en conteos, semilla y algoritmo no prueba identidad de todos los archivos de datos.",
            "Los tiempos HCE y Net5 usan alcances, lotes y dispositivos distintos.",
            "Una corrida por resolucion no cuantifica variacion entre semillas.",
        ],
    }
    (destino / "auditoria_comparacion.json").write_text(
        json.dumps(evidencia, indent=2, ensure_ascii=False) + "\n")
    print("Verificadas 6 matrices, exactitudes, F1 macro y los dos historiales.")
    for fila in filas:
        print(f"R{fila['resolucion']} {fila['modelo']}: {fila['accuracy_pct']:.2f}% / F1 {fila['f1_macro_pct']:.2f}%")


if __name__ == "__main__":
    generar()
