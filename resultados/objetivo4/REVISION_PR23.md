# Revisión técnica del PR #23

## Versiones examinadas

- PR original: `c6fa862c50ddb2e88554f20cb82929cb38f83419`.
- Código registrado por las dos corridas: `8164428f30ee997e31fa690d59a81bc7ad12beb2`.
- Fecha de revisión: 25 de septiembre de 2026 (Colombia).

## Hallazgo bloqueante y corrección

El PR original incorpora las optimizaciones y resultados sobre una base que
todavía contiene el modelo denso diagnóstico. Faltan el dataset nativo,
orquestador, validador y soporte de trazabilidad; `net5_modelo.py` tampoco
contiene la implementación nativa necesaria.

Reproducción en el PR original, con PyTorch instalado:
`python -m pytest tests/test_octnet_backend.py --collect-only -q` falla con
`ModuleNotFoundError: No module named 'net5_dataset_octree'`.

Se integró en una rama de revisión el commit exacto `8164428` utilizado en los
entrenamientos. Los cuatro conflictos add/add correspondían a archivos iguales
salvo finales de línea; se conservaron las versiones del commit de ejecución.
Se mantuvieron intactos los resultados e historiales originales.

El CI ligero anterior no instalaba PyTorch y podía omitir las pruebas del
backend. Se añadió un trabajo CPU con PyTorch y Numba para comprobarlas.

## Pruebas ejecutadas

- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -m 'not dataset and not gpu' -q`: **99 passed, 4 deselected**, 37.38 s.
- Incluye equivalencia exacta de los planes NumPy/Numba con enumeración
  exhaustiva en R32/R64, equivalencia de convolución fusionada y gradientes
  en float64, recarga de checkpoints sintéticos y pruebas de partición.
- Auditoría independiente de seis matrices: sus totales, exactitudes y F1
  macro concuerdan con los reportes; cada una contiene 2468 muestras.
- Los dos historiales concuerdan con la mejor época, validación y paciencia.
- Entorno local: Python 3.12, PyTorch 2.14.0+cu130, Numba 0.67.0, NumPy 2.3.5;
  CUDA no disponible. No se reprodujeron entrenamiento ni tiempos de GPU.
- Las cuatro pruebas excluidas requieren dataset/GPU; el resultado local no
  demuestra reproducción de las predicciones con los pesos reales.

## Comparación de los registros publicados

| Modelo | Resolución | Accuracy test (%) | F1 macro (%) |
|---|---:|---:|---:|
| SVM-HCE | 32 | 75.93 | 65.82 |
| Random Forest-HCE | 32 | 76.38 | 67.72 |
| Net5-Octree | 32 | 82.21 | 74.93 |
| SVM-HCE | 64 | 77.19 | 68.47 |
| Random Forest-HCE | 64 | 77.19 | 68.53 |
| Net5-Octree | 64 | 83.23 | 77.23 |

Net5 obtiene 144 aciertos adicionales en R32 frente a RF y 149 en R64 frente
a cualquiera de los dos clasificadores clásicos: +5.8347 y +6.0373 puntos
porcentuales respectivamente. Es una comparación descriptiva de estas corridas;
no establece significancia estadística ni variación entre semillas.

La semilla, los tamaños y el algoritmo de partición coinciden. Los dos reportes
Net5 registran el mismo hash de manifiesto. Falta el manifiesto real para
comprobar los identificadores; los resúmenes HCE no incluyen ese hash.

## Selección del modelo y tiempos

El código guarda el checkpoint cuando mejora `val_acc`, espera 20 épocas sin
mejora y recarga el mejor checkpoint antes de evaluar test. R32 selecciona la
época 25 y termina en 45; R64 selecciona la 19 y termina en 39. La implementación
es coherente con los historiales.

Los promedios de época son 3.8514 min (R32) y 12.2271 min (R64). El campo
`tiempo_total_min` suma los tiempos medidos de las épocas de train/validación;
no es el tiempo de pared de toda la invocación: excluye, entre otras cosas,
escritura de checkpoints y evaluación final.

El tiempo llamado «solo forward» incluye preparación/transferencia de planes
dentro de `modelo(grids)`. `LoteGridOctree.to()` solo transfiere atributos.
Se corrigió la descripción para futuras ejecuciones; los valores históricos
no se modificaron. Las tres repeticiones recorren test completo y sincronizan
CUDA; no se guardan los tiempos individuales ni su dispersión.

HCE mide una llamada `predict(X_test)` sobre descriptores ya calculados, en CPU
y con el conjunto como lote. Net5 usa GPU y batch 1; su pipeline incluye carga,
conversión/preparación de octrees y transferencias desde los NPZ. No incluye
la generación inicial de NPZ desde mallas. Por ello no deben publicarse cocientes
de velocidad HCE/Net5 como una comparación de extremo a extremo equivalente.

## Pendientes concretos para Ricardo

1. Compartir los mejores checkpoints R32/R64 (`net5_octree_mejor_R32.pth` y
   `net5_octree_mejor_R64.pth`) mediante un enlace persistente con SHA-256.
   Los últimos checkpoints son útiles si se quiere reanudar, pero no son
   necesarios para reproducir la evaluación del mejor modelo.
2. Compartir `particion_objetivo2_modelos.json` y el JSON del barrido de workers
   usado para seleccionar 8 workers. Los archivos actuales solo resumen ese barrido.
3. Para reproducir la comparación completa, identificar también los modelos
   HCE, sus contratos de features y cualquier transformador requerido.
4. Antes del cierre del análisis de costo, medir protocolos equivalentes:
   predicción con representación lista y pipeline desde una entrada común,
   documentando lote, hardware, calentamiento y repeticiones. Esto requiere
   evaluar modelos guardados, no volver a entrenarlos.

El script legado `fase4_comparacion/comparar_predicciones.py` conserva rutas
Windows absolutas, un nombre antiguo de checkpoint y entrada densa; no debe
usarse para validar este backend nativo sin adaptarlo. La comparación de esta
revisión usa exclusivamente `comparar_resumenes.py` y los registros publicados.

## Reproducir esta comparación

```bash
python fase4_comparacion/comparar_resumenes.py
```

Genera `comparacion_publicada.csv` y `auditoria_comparacion.json`, incluyendo
hashes de los archivos fuente. No carga pesos ni modifica resultados originales.
