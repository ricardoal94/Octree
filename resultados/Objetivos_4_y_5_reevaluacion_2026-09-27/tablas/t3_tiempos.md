| Método | Res. | Carga (ms) | Descriptores / preparación (ms) | Clasificador / red (ms) | Total mediana (ms) | Total p95 (ms) |
|---|---|---|---|---|---|---|
| SVM | R32 | 7.9 | 4.9 | 0.82 | 13.6 | 35.0 |
| SVM | R64 | 27.3 | 22.8 | 0.84 | 51.5 | 121.5 |
| Bosque aleatorio | R32 | 7.9 | 4.9 | 5.83 | 18.7 | 39.8 |
| Bosque aleatorio | R64 | 27.3 | 22.8 | 5.79 | 56.8 | 126.3 |
| Net5-Octree (GPU) | R32 | 6.9 | 35.5 | 5.59 | 48.3 | 91.5 |
| Net5-Octree (GPU) | R64 | 24.5 | 180.2 | 9.90 | 217.5 | 441.9 |
| Net5-Octree (CPU, 1 hilo) | R32 | 6.9 | 35.5 | 108.93 | 153.2 | 310.3 |
| Net5-Octree (CPU, 1 hilo) | R64 | 24.5 | 180.2 | 379.50 | 587.1 | 1218.8 |
