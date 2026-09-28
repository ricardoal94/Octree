| Método | Res. | Carga (ms) | Descriptores / preparación (ms) | Clasificador / red (ms) | Total mediana (ms) | Total p95 (ms) |
|---|---|---|---|---|---|---|
| SVM | R32 | 8.3 | 5.0 | 0.82 | 14.3 | 34.6 |
| SVM | R64 | 27.6 | 23.0 | 0.82 | 52.7 | 118.8 |
| Bosque aleatorio | R32 | 8.3 | 5.0 | 5.88 | 19.7 | 40.8 |
| Bosque aleatorio | R64 | 27.6 | 23.0 | 5.77 | 57.7 | 123.8 |
| Net5-Octree (GPU) | R32 | 7.3 | 37.5 | 5.91 | 52.1 | 105.4 |
| Net5-Octree (GPU) | R64 | 24.9 | 187.9 | 11.81 | 230.1 | 462.9 |
| Net5-Octree (CPU, 1 hilo) | R32 | 7.3 | 37.5 | 102.65 | 148.5 | 312.4 |
| Net5-Octree (CPU, 1 hilo) | R64 | 24.9 | 187.9 | 334.87 | 549.1 | 1090.2 |
