# Aceleración y corrección del aprendizaje de Net5-Octree

Rama `feature/octnet-native-backend`, commits `9e908a7a8`, `41d5f5cbd` y
`8164428f3` (sobre `28a069e88`). Todos están publicados.

## Resumen

| Problema | Causa | Solución | Efecto |
|---|---|---|---|
| El modelo no aprendía: la accuracy de val quedaba fija en 9.04 % | LR 1e-3 demasiado alto para `fc1` (8.4 M parámetros) | `--lr 0.0001` (argumento; sin cambio de código) | Aprende: 47 % de val tras 1 época de prueba |
| Época R32 de ~34 min, GPU al ~8 % | Cálculo de planes en CPU y convolución con 27 bucles | Tres optimizaciones, con resultados idénticos | Época R32 ~4-5 min, R64 ~12-15 min |

Ninguna optimización cambia el experimento: el plan de convolución es idéntico bit a bit, y la convolución fusionada es matemáticamente equivalente (verificado en `float64`).

---

## 1. Aprendizaje: learning rate (sin cambios de código)

### Síntoma

En el entrenamiento completo de R32 con los valores por defecto (batch 1, LR 1e-3), la accuracy de validación quedó en **9.04 % en todas las épocas**. Ese valor coincide aproximadamente con la frecuencia de la clase más común (chair), y la loss de val se mantuvo en ~3.45. En una prueba con clases balanceadas, la loss de val quedó en **3.688 ≈ ln(40)**: una predicción uniforme.

### Diagnóstico

Se descartó un fallo de arquitectura con un script de diagnóstico sobre el modelo recién inicializado:

- La escala de las activaciones decae suavemente (std 0.40 → 0.07) y no hay capas muertas.
- Todas las capas reciben gradiente de orden similar.
- El modelo **sobreajusta 8 muestras al 100 %** en 10-25 pasos, así que la arquitectura puede aprender.

La pista fue que, con LR 1e-3, la loss **saltó de 3.7 a 7.3** en el paso 10, incluso con solo 8 muestras. `fc1` recibe 16,384 entradas (32 × 8³) y concentra casi todos los parámetros. Con Adam, un paso de 1e-3 en tantos pesos cambia demasiado su salida. Con gradientes ruidosos (batch 1) y dropout, esos saltos se repiten, y la hipótesis es que acaban apagando las ReLU de `fc1`. La salida queda entonces determinada solo por el bias, que aprende la distribución de clases, y eso encaja con los dos síntomas.

### Solución y evidencia

Se usa `--lr 0.0001`, manteniendo batch 1 y el resto del protocolo. Prueba corta en R32 (200 muestras de train, 3 épocas):

| LR | Val época 1 | Val época 2 | Val época 3 | Test |
|---|---|---|---|---|
| 1e-3 | 3 % | 2 % | 3 % (loss 3.688) | — |
| **1e-4** | 15 % | 21 % | **37 %** (loss 2.61) | 32.5 % |

`StepLR(step_size=20, gamma=0.7)` sigue reduciendo el LR durante el entrenamiento. El orquestador recibe `--lr 0.0001` y valida los resultados oficiales con ese valor. El barrido de workers no depende del LR, porque solo mide velocidad.

---

## 2. Aceleración

### Dónde se iba el tiempo (medido)

| Etapa por muestra, R32 | Tiempo |
|---|---|
| Cálculo de planes de convolución (CPU, workers) | ~500-600 ms |
| Forward + backward en GPU, batch 1 | ~168 ms, casi todo lanzamiento de operaciones pequeñas |
| Cálculo real en GPU | ~6-22 ms |

La GPU estaba casi siempre esperando a la CPU. El cálculo de planes era el **98 %** del coste de carga, y generaba **3.6 M candidatos para quedarse con 118 k aristas útiles**, el 3 %.

### 2.1 Plan de convolución sin candidatos imposibles (`9e908a7a8`)

**Archivo:** `fase3_net5/grid_octree.py`, función `construir_plan_convolucion`

**Qué cambia:** las hojas del grid-octree forman una partición, así que una celda que se solapa con la hoja de salida **no puede ser otra hoja** (solo la propia hoja). El algoritmo anterior evaluaba igualmente todas las celdas del interior. El nuevo:

- Por eje, distingue las celdas interiores de la única celda exterior posible, la franja de 1 vóxel en la dirección del desplazamiento del kernel.
- Evalúa solo combinaciones con al menos un eje exterior, más la propia hoja cuando las aristas coinciden.
- Procesa los 26 kernels no centrales a la vez y reproduce el orden oficial con una única ordenación final.

**Resultado:** R32 616 → 186 ms por muestra (×3.3), R64 3,853 → 1,109 ms (×3.5).

### 2.2 Convolución con los 27 kernels fusionados (`41d5f5cbd`)

**Archivo:** `fase3_net5/octnet_backend.py`, `ConvolucionOctree3x3.forward` y el nuevo `LoteGridOctree.plan_convolucion_fusionado_torch`

**Qué cambia:** antes, cada convolución hacía 27 iteraciones, cada una con `index_select`, matmul, multiplicación e `index_add_`: unas 108 operaciones de GPU solo en el forward. Ahora:

1. Una sola matmul proyecta cada hoja con los 27 kernels: `Z = X @ W`, con forma (hojas, 27 × C_out).
2. Un `index_select` toma, para cada arista, la fila `Z[entrada × 27 + kernel]`.
3. Una multiplicación por el coeficiente y un único `index_add_`.

**Resultado:** paso de entrenamiento con batch 1: R32 168 → 12 ms por muestra (×14.5), R64 149 → 15 ms (×9.9).

### 2.3 Plan de convolución compilado con numba (`8164428f3`)

**Archivos:** `fase3_net5/grid_octree.py` y `requirements.txt`

**Qué cambia:** el plan se construye con un bucle compilado que recorre las aristas **directamente en el orden oficial** (kernel, arista de salida, arista de entrada, hoja, x, y, z). Aplica la misma regla de descarte, no necesita ordenar al final y hace una sola pasada.

- **Respaldo automático:** si numba no está instalado se usa la versión numpy de 2.1. La CI ligera (`requirements-objetivo2.txt`) no necesita la dependencia.
- El backend activo queda en `grid_octree.BACKEND_PLAN_CONVOLUCION` (`"numba"` o `"numpy"`).
- `requirements.txt` añade `numba>=0.67.0`. Versión instalada: 0.67.0, con numpy 2.4.6 y Python 3.14.

**Resultado:** R32 144 → 29 ms por muestra (×5), R64 1,050 → 172 ms (×6).

### Alternativas evaluadas y descartadas

| Alternativa | Motivo |
|---|---|
| Caché de planes en disco o RAM | Ocuparía ~51 GB para R32 y ~250 GB para R64 |
| Más workers | 4 → 8 solo daba un 10 %; la CPU tiene 8 núcleos físicos |
| Batch 4 | Tras la fusión solo gana ~15 % y cambia el protocolo |
| Planes en GPU (proceso principal) | Plan idéntico, pero limitado por el coste de lanzar ~1,200 operaciones en Windows; sin ganancia en R32 |
| Planes en GPU (dentro de los workers) | Cada worker crea un contexto CUDA: la VRAM llegó a 11.1 de 12 GB y el proceso se bloqueó |
| Mapa denso vóxel→hoja | Contradice el contrato de no densificar R32/R64 |

---

## 3. Verificación

| Comprobación | Resultado |
|---|---|
| Suite de tests | 103 pasan |
| Plan (numpy y numba) frente a la enumeración exhaustiva original | Idéntico en valores, dtype y orden (R32 y R64, 2 nubes, todas las escalas) |
| Plan numba frente a numpy con datos reales | 390/390 idénticos (train y test, R32 y R64) |
| Convolución fusionada frente a la versión por kernel | Salida y gradientes iguales en `float64` con tolerancia 1e-12 |
| Respaldo sin numba | Se activa la versión numpy |
| Accuracy tras 1 época de prueba (R32, 1100 muestras) | 46-47 % antes y después de las optimizaciones |

Tests añadidos:

- `tests/test_grid_octree.py`: `test_plan_convolucion_identico_a_enumeracion_exhaustiva` (conserva el algoritmo original como referencia) y la nube `_nube_densa`, con hojas de arista 1, 2, 4 y 8.
- `tests/test_octnet_backend.py`: `test_convolucion_fusionada_equivale_a_kernel_por_kernel`.

---

## 4. Impacto en tiempos

Época de prueba de R32 (1000 de train + 100 de val, batch 1, LR 1e-4, 8 workers):

| Versión | Tiempo |
|---|---|
| Original | 229.8 s |
| + plan sin candidatos imposibles | 192.0 s |
| + convolución fusionada | 67.7 s |
| + numba | **28.7 s** |

Barrido de workers en `8164428f3` (pipeline con 8 workers): **R32 14 ms por muestra, R64 64 ms**, frente a 63 ms y 422 ms en `41d5f5cbd`.

| Época completa (estimada) | Original | Actual |
|---|---|---|
| R32 | ~34 min | ~4-5 min |
| R64 | más de 2 h | ~12-15 min |

---

## 5. Archivos modificados

| Archivo | Commits | Cambio |
|---|---|---|
| `fase3_net5/grid_octree.py` | `9e908a7a8`, `8164428f3` | Plan sin candidatos imposibles; versión numba con respaldo numpy |
| `fase3_net5/octnet_backend.py` | `41d5f5cbd` | Convolución con los 27 kernels fusionados |
| `tests/test_grid_octree.py` | `9e908a7a8`, `8164428f3` | Equivalencia del plan con el algoritmo original (numpy, numba y API pública) |
| `tests/test_octnet_backend.py` | `41d5f5cbd` | Equivalencia de la convolución fusionada (salida y gradientes) |
| `requirements.txt` | `8164428f3` | Añade `numba>=0.67.0` |

Cambios fuera del código:

- `numba 0.67.0` instalado en el venv.
- La carpeta `logs/` sin trackear se movió a `..\Tesis_logs_sueltos_20260924\` para dejar el árbol limpio.
- Barrido regenerado y validado en `..\Tesis_smoke_workers_8164428` (8 workers para R32 y R64).
