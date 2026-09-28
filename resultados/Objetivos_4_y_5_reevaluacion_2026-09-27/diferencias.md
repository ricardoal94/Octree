# Diferencias entre la reevaluación y los resultados publicados

## Predicciones (2,468 objetos de test por resolución)

| Resolución | Mismos IDs | SVM | Bosque | Net5 |
|---|---|---|---|---|
| R32 | sí | 0 | 0 | 0 |
| R64 | sí | 0 | 0 | 0 |

## Análisis de aciertos (debe ser idéntico)

| Componente | Idéntico |
|---|---|
| metricas | sí |
| acuerdo_entre_metodos | sí |
| mcnemar | sí |
| confusiones_principales | sí |
| datos_por_resolucion | sí |
| modelos | sí |
| recall_por_clase | sí |

## Tiempos por objeto (mediana, ms)

| Componente | R32 anterior | R32 nuevo | Cambio | R64 anterior | R64 nuevo | Cambio |
|---|---|---|---|---|---|---|
| SVM total | 14.32 | 13.61 | -4.9 % | 52.74 | 51.53 | -2.3 % |
| Bosque total | 19.66 | 18.69 | -4.9 % | 57.68 | 56.78 | -1.6 % |
| Net5 GPU total | 52.09 | 48.30 | -7.3 % | 230.10 | 217.52 | -5.5 % |
| Net5 CPU total | 148.54 | 153.20 | +3.1 % | 549.10 | 587.05 | +6.9 % |
| HCE carga | 8.32 | 7.93 | -4.7 % | 27.56 | 27.29 | -1.0 % |
| HCE descriptores | 5.02 | 4.87 | -3.1 % | 23.04 | 22.76 | -1.2 % |
| SVM clasificador | 0.82 | 0.82 | -0.9 % | 0.82 | 0.84 | +2.2 % |
| Bosque clasificador | 5.88 | 5.83 | -0.9 % | 5.77 | 5.79 | +0.4 % |
| Net5 carga | 7.30 | 6.91 | -5.3 % | 24.91 | 24.54 | -1.5 % |
| Net5 preparación | 37.55 | 35.48 | -5.5 % | 187.90 | 180.22 | -4.1 % |
| Net5 forward GPU | 5.91 | 5.59 | -5.5 % | 11.81 | 9.90 | -16.2 % |
| Net5 forward CPU | 102.65 | 108.93 | +6.1 % | 334.87 | 379.50 | +13.3 % |

Misma muestra de 400 objetos: R32 sí, R64 sí.

## Memoria (MiB)

| Caso | RAM adicional anterior | RAM adicional nueva | RAM total anterior | RAM total nueva | VRAM asignada anterior | VRAM asignada nueva |
|---|---|---|---|---|---|---|
| svm_R32 | 40 | 40 | 161 | 161 | — | — |
| rf_R32 | 268 | 268 | 388 | 389 | — | — |
| net5_gpu_R32 | 729 | 748 | 1309 | 1328 | 138 | 138 |
| net5_cpu_R32 | 272 | 274 | 852 | 854 | — | — |
| svm_R64 | 42 | 46 | 163 | 167 | — | — |
| rf_R64 | 266 | 266 | 387 | 387 | — | — |
| net5_gpu_R64 | 821 | 765 | 1402 | 1345 | 421 | 421 |
| net5_cpu_R64 | 734 | 730 | 1314 | 1310 | — | — |
