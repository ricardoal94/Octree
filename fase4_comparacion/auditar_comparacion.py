"""Comprueba evidencia publicada; no sustituye la inferencia con modelos reales."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def auditar(carpeta, exigir_modelos=False):
    def leer(nombre):
        return json.loads((carpeta / nombre).read_text(encoding='utf-8'))

    def filas(nombre):
        with (carpeta / nombre).open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))

    def comprobar(condicion, mensaje):
        if not condicion:
            raise ValueError(mensaje)

    manifiesto = leer('datos/particion_objetivo2_modelos.json')
    grupos = [set(manifiesto[k + '_ids']) for k in ('train', 'val', 'test')]
    for k, ids in zip(('train', 'val', 'test'), grupos):
        comprobar(len(ids) == len(manifiesto[k + '_ids']) == manifiesto['n_' + k], 'Conteos/duplicados: ' + k)
    comprobar(not any(grupos[i] & grupos[j] for i in range(3) for j in range(i)), 'Particiones solapadas')
    analisis = leer('resultados/analisis.json')
    tiempos = leer('resultados/tiempos.json')
    predicciones = {}
    for r in (32, 64):
        datos = filas(f'resultados/predicciones_R{r}.csv')
        por_id = {x['id']: x for x in datos}
        comprobar(len(datos) == len(por_id) == 2468 and set(por_id) == grupos[2], f'IDs test R{r}')
        comprobar(all(manifiesto['orden_clases'][int(x['real'])] == x['id'].split('/')[0] for x in datos), f'Clases R{r}')
        predicciones[r] = por_id
        hce = leer(f'resultados/oficiales/resumen_hce_R{r}.json')
        net = leer(f'resultados/oficiales/resumen_net5_octree_R{r}.json')
        for metodo, oficial in [('svm', hce['svm']), ('rf', hce['random_forest']), ('net5', net)]:
            cm = np.zeros((40, 40), dtype=int)
            for x in datos:
                cm[int(x['real']), int(x[metodo])] += 1
            comprobar(np.array_equal(cm, oficial['matriz_confusion']), f'Matriz {metodo} R{r}')
            aciertos = int(np.trace(cm))
            met = analisis['metricas'][f'{metodo}_R{r}']
            comprobar(aciertos == met['aciertos'] and met['errores'] == 2468 - aciertos, f'Aciertos {metodo} R{r}')
            comprobar(np.isclose(aciertos / 2468, oficial['test_acc'], atol=5e-7, rtol=0), f'Exactitud oficial {metodo} R{r}')
            f1 = np.mean(2 * cm.diagonal() / (cm.sum(0) + cm.sum(1)))
            comprobar(np.isclose(f1, met['f1_macro'], atol=1e-12, rtol=0), f'F1 {metodo} R{r}')
        registros = filas(f'resultados/tiempos_por_objeto_R{r}.csv')
        comprobar(len(registros) == len({x['id'] for x in registros}) == tiempos['resoluciones'][str(r)]['n_objetos'], f'Muestra tiempos R{r}')
        for x in registros:
            comprobar(x['id'] in por_id, f'ID de tiempos fuera del test R{r}')
            ref = por_id[x['id']]
            comprobar(all(x['pred_' + m] == ref[m] for m in ('svm', 'rf')) and x['pred_net5_gpu'] == x['pred_net5_cpu'] == ref['net5'], f'Prediccion tiempos {x["id"]}')
        for campo, resumen in tiempos['resoluciones'][str(r)]['tiempos_ms'].items():
            valores = np.array([float(x[campo]) for x in registros])
            comprobar(np.all(np.isfinite(valores)) and np.all(valores >= 0), f'Tiempos invalidos {campo}')
            comprobar(np.allclose([np.median(valores), valores.mean(), np.percentile(valores, 95)], [resumen['mediana'], resumen['media'], resumen['p95']], atol=1e-8, rtol=1e-10), f'Resumen tiempos {campo} R{r}')
    for prueba in analisis['mcnemar']:
        r = prueba['resolucion']
        if isinstance(r, int):
            a, b = prueba['comparacion'].split(' vs ')
            pares = [(x[a] == x['real'], x[b] == x['real']) for x in predicciones[r].values()]
        else:
            m = prueba['comparacion'].split(':')[0]
            pares = [(x[m] == x['real'], predicciones[32][ident][m] == x['real']) for ident, x in predicciones[64].items()]
        b = sum(x and not y for x, y in pares)
        c = sum(y and not x for x, y in pares)
        p = min(1, 2 * sum(math.comb(b + c, k) for k in range(min(b, c) + 1)) / 2 ** (b + c))
        comprobar(b == prueba['solo_primero_acierta'] and c == prueba['solo_segundo_acierta'] and math.isclose(p, prueba['p_valor'], rel_tol=1e-10, abs_tol=1e-15), 'McNemar: ' + prueba['comparacion'])
    ausentes = []
    for linea in (carpeta / 'SHA256SUMS.txt').read_text().splitlines():
        esperado, nombre = linea.split(' *', 1)
        archivo = carpeta / nombre
        if not archivo.is_file():
            comprobar(nombre.startswith('modelos/'), 'Dato ausente: ' + nombre)
            ausentes.append(nombre)
            continue
        comprobar(hashlib.sha256(archivo.read_bytes()).hexdigest() == esperado, 'Hash distinto: ' + nombre)
    comprobar(not exigir_modelos or not ausentes, 'Modelos ausentes: ' + ', '.join(ausentes))
    return {'estado': 'OK', 'matrices_verificadas': 6, 'mcnemar_verificados': len(analisis['mcnemar']), 'modelos_ausentes': ausentes, 'inferencia_repetida': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--carpeta', type=Path, default=Path(__file__).resolve().parents[1] / 'resultados/Objetivos_4_y_5_comparacion')
    parser.add_argument('--exigir-modelos', action='store_true')
    args = parser.parse_args()
    print(json.dumps(auditar(args.carpeta, args.exigir_modelos), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
