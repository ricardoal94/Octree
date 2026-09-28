# Reevaluación de los modelos entrenados con los octrees originales (27-09-2026)

Reevaluación de los seis modelos ya entrenados (SVM, Bosque aleatorio y Net5-Octree, en R32 y R64) en el equipo de los autores, con los octrees originales de ModelNet40. Se siguieron las instrucciones de la PR #26 (`../Objetivos_4_y_5_comparacion/README.md`, sección 2.2) **sin modificar ninguno de sus programas**.

La nueva ejecución se guarda en **una carpeta separada** para conservar los resultados anteriores y poder compararlos:

| Carpeta | Contenido |
|---|---|
| `resultados/Objetivos_4_y_5_comparacion/` | Resultados publicados en la PR #26, **sin modificar** |
| `resultados/Objetivos_4_y_5_reevaluacion_2026-09-27/` (esta) | Nueva ejecución del 27-09-2026, su registro y la comparación con la anterior |

**Alcance:** no se reentrenó ningún modelo y **no se declara cerrado el objetivo general**. Se documentan los límites del cronometraje y de la muestra de memoria (sección 5).

## Resumen

- **Aciertos y errores: reproducidos exactamente.** Las 2,468 predicciones de cada uno de los 6 modelos coinciden con las publicadas (0 diferencias), y también las métricas, los intervalos de confianza, las pruebas de McNemar, el acuerdo entre métodos, las confusiones y el recall por clase.
- **Tiempos: varían entre −7.3 % y +6.9 %** en la latencia total por objeto (mediana). Se mantienen las proporciones entre métodos y resoluciones que sostienen las conclusiones del informe.
- **Memoria: estable, con una excepción.** La RAM adicional de Net5 en GPU pasa de +13 % a +2 % entre R32 y R64. Esa diferencia está dentro de la variación entre ejecuciones y no debe presentarse como un efecto de la resolución.
- Se encontraron dos problemas de **portabilidad a Windows** en los programas de auditoría, y uno de **acceso** al enlace de Drive (sección 4).

## 1. Qué se ejecutó

| Aspecto | Valor |
|---|---|
| Fecha | 2026-09-27, 20:49-21:01 (UTC−5) |
| Rama y commit | `codex/cierre-comparacion` @ `8147a6e018243f351c97dca25d3e2d5621a2294b` (cabeza de la PR #26) |
| Estado de git | Sin cambios en archivos versionados. Una carpeta sin seguimiento ajena a la evaluación (`respaldo_fase4/`). |
| Equipo | Windows 11 Pro 10.0.26100, AMD Ryzen 7 5700X (8 núcleos y 16 hilos), 31.9 GB de RAM, NVIDIA RTX 5070 12 GB (controlador 617.14) |
| Software | Python 3.14.4, PyTorch 2.11.0+cu128 (CUDA 12.8), numpy 2.4.6, scikit-learn 1.9.0, numba 0.67.0, scipy 1.17.1, joblib 1.5.3, matplotlib 3.11.0 |
| Octrees | `data/octrees_32` y `data/octrees_64` originales, 12,311 archivos cada uno |
| Carpeta evaluada | Copia externa de `resultados/Objetivos_4_y_5_comparacion`, exportada de git con finales de línea LF (ver 4.1), más los seis modelos en `modelos/` |

Pasos, ejecutados con `ejecutar_reevaluacion.ps1`, que reproduce la sección 2.2 del README pasando `--carpeta` a los cuatro programas:

| Paso | Comando | Resultado | Duración |
|---|---|---|---|
| 1 | `auditar_comparacion.py --exigir-modelos` | OK: 6 matrices, 9 McNemar, 0 modelos ausentes | < 1 s |
| 2 | `evaluar_comparacion.py` | 6/6 predicciones `[OK]` frente a las oficiales | 12.1 min |
| 3 | `auditar_comparacion.py --exigir-modelos` | OK sobre las salidas nuevas | < 1 s |
| 4 | `tablas_comparacion.py` | 7 tablas | < 1 s |
| 5 | `figuras_comparacion.py` | 7 figuras (PNG y SVG) | 4 s |

La salida completa de cada paso, con marcas de tiempo, está en `registro_ejecucion.log`.

Antes de la evaluación también se revisó la PR:

- **Suite de pruebas completa** (`python -m pytest`, con un hilo por biblioteca): **103 pruebas superadas**, incluidas las 4 que requieren dataset real y GPU, que no se habían podido ejecutar en la revisión anterior.
- **`comparar_resumenes.py`:** métricas publicadas consistentes (ver 4.1 sobre el archivo que reescribe).
- **`auditar_comparacion.py` sobre la carpeta publicada** (exportada con LF): OK, con los 6 modelos ausentes como único aviso.

## 2. Acceso a los modelos

- **Enlace de Google Drive:** responde y muestra dos subcarpetas, `hce` y `net5`, modificadas el 26-09, pero **pide iniciar sesión** para ver o descargar los archivos. Sin credenciales no fue posible comprobar la descarga desde ese enlace.
- **Modelos usados:** las copias locales de los autores. Sus SHA-256 coinciden con los seis de `SHA256SUMS.txt` (paso 1, `modelos_ausentes: []`), que son los mismos que `VERIFICACION_MODELOS.json` registra para los archivos descargados de Drive. Son, por tanto, los mismos modelos.
- Los modelos HCE se cargaron con scikit-learn 1.9.0, la misma versión con que se guardaron, sin advertencias de versión.

## 3. Diferencias con los resultados publicados

Detalle completo en `diferencias.md` y `diferencias.json`. Se regeneran comparando las dos carpetas, desde la raíz del repositorio:

```powershell
python resultados/Objetivos_4_y_5_reevaluacion_2026-09-27/comparar_con_anterior.py
```

Para comparar otras dos ejecuciones: `--nuevo RUTA --anterior RUTA --salida RUTA`.

### 3.1 Aciertos y errores: sin diferencias

| Elemento | Resultado |
|---|---|
| Predicciones R32 y R64 (2,468 objetos cada una, mismos IDs) | 0 diferencias en SVM, Bosque y Net5 |
| Métricas, IC 95 %, McNemar, acuerdo, confusiones, costo de datos, complejidad de modelos | Idénticos |
| Recall por clase | Idéntico |
| Tablas t1 (aciertos), t2 (McNemar), t6 (acuerdo) y t7 (confusiones) | Idénticas |

### 3.2 Tiempos por objeto (mediana, ms; misma muestra de 400 objetos)

| Método | R32 publicado | R32 nuevo | Cambio | R64 publicado | R64 nuevo | Cambio |
|---|---|---|---|---|---|---|
| SVM | 14.32 | 13.61 | −4.9 % | 52.74 | 51.53 | −2.3 % |
| Bosque aleatorio | 19.66 | 18.69 | −4.9 % | 57.68 | 56.78 | −1.6 % |
| Net5 GPU | 52.09 | 48.30 | −7.3 % | 230.10 | 217.52 | −5.5 % |
| Net5 CPU (1 hilo) | 148.54 | 153.20 | +3.1 % | 549.10 | 587.05 | +6.9 % |

Por componentes, el mayor cambio es el forward de Net5: −16.2 % en GPU y +13.3 % en CPU, ambos en R64. Por eso cambian las tablas t3 y t5; en la t5, solo el tiempo *estimado* de extracción HCE de R64 (9.2 → 9.0 min), que se deriva de estos tiempos.

**Las proporciones que sostienen las conclusiones se mantienen:**

| Proporción | Publicado | Nuevo |
|---|---|---|
| Net5 GPU frente a SVM y Bosque (latencia) | 2.7-4.4 veces | 2.6-4.2 veces |
| Latencia R64 / R32: SVM, Bosque, Net5 GPU | ×3.7, ×2.9, ×4.4 | ×3.8, ×3.0, ×4.5 |
| Net5 CPU frente a SVM | ×10.4 | ×11.3-11.4 |

### 3.3 Memoria (MiB)

| Caso | RAM adicional publicada | RAM adicional nueva | RAM total publicada | RAM total nueva | VRAM asignada |
|---|---|---|---|---|---|
| SVM R32 | 40 | 40 | 161 | 161 | — |
| SVM R64 | 42 | 46 | 163 | 167 | — |
| Bosque R32 | 268 | 268 | 388 | 389 | — |
| Bosque R64 | 266 | 266 | 387 | 387 | — |
| Net5 GPU R32 | 729 | 748 | 1,309 | 1,328 | 138 (igual) |
| Net5 GPU R64 | 821 | 765 | 1,402 | 1,345 | 421 (igual) |
| Net5 CPU R32 | 272 | 274 | 852 | 854 | — |
| Net5 CPU R64 | 734 | 730 | 1,314 | 1,310 | — |

La VRAM es idéntica y los casos de CPU varían menos de 5 MiB. La RAM adicional de **Net5 en GPU** varía más (+19 MiB en R32 y −56 MiB en R64). Con ello, la diferencia R32 → R64 pasa de **+13 %** (publicado) a **+2 %** (nuevo). **La afirmación del informe publicado "RAM adicional de Net5 en GPU: +13 % en R64" no es robusta:** está dentro de la variación entre ejecuciones. El aumento en CPU (×2.7) sí se reproduce.

## 4. Hallazgos para revisar

### 4.1 Finales de línea en Windows (programas de auditoría)

Con `core.autocrlf=true`, que es la configuración habitual de git en Windows, los archivos de texto se escriben con CRLF al hacer checkout, y sus bytes dejan de coincidir con los hashes calculados sobre LF:

- `auditar_comparacion.py` **falla** en una copia de trabajo de Windows con `Hash distinto: datos/particion_objetivo2_modelos.json`. El contenido es idéntico: sin los CR, el hash coincide con `SHA256SUMS.txt` y con el blob de git.
- `comparar_resumenes.py` **reescribe** `resultados/objetivo4/auditoria_comparacion.json` con otros hashes de los resúmenes de Net5, por la misma causa. El cambio se revirtió y no forma parte de esta entrega.

En el CI de Linux no ocurre. Posibles soluciones (no aplicadas): un `.gitattributes` que fije `eol=lf` para `resultados/**`, o calcular los hashes sobre el contenido con los finales de línea normalizados. Para esta reevaluación se usó una exportación de git con LF (`git -c core.autocrlf=false archive`), que representa fielmente lo publicado.

### 4.2 Acceso a Drive

El enlace publicado exige iniciar sesión. Si debe servir a terceros, conviene darle acceso de lectura a quien tenga el enlace.

## 5. Límites del protocolo (sin cambios respecto a la PR #26)

Se mantienen los límites que documenta el informe publicado. Se precisan dos:

- **Cronometraje de Net5 en GPU:** la latencia omite el armado del lote (`LoteGridOctree.desde_muestras`) previo al forward. En CPU sí se incluye, así que las dos mediciones no son simétricas.
- **Muestra de memoria:** los 100 objetos resultan de truncar una lista ordenada por clase, y son exactamente **40 de airplane, 40 de bathtub y 20 de bed**. El pico de memoria **representa solo esas tres clases**, no el conjunto de test.
- Los tiempos parten del `.npz` del octree y excluyen su generación desde la malla. El tiempo de extracción HCE de entrenamiento sigue siendo una estimación.
- Un solo entrenamiento por modelo (semilla 42) y un solo equipo de medición.

## 6. Archivos de esta carpeta

| Archivo | Contenido |
|---|---|
| `REEVALUACION.md` | Este informe |
| `registro_ejecucion.log` | Salida completa de los cinco pasos, con entorno y marcas de tiempo |
| `ejecutar_reevaluacion.ps1` | Script que ejecutó los pasos (rutas del equipo de los autores) |
| `comparar_con_anterior.py` | Compara esta carpeta con `../Objetivos_4_y_5_comparacion/` |
| `diferencias.md`, `diferencias.json` | Comparación detallada |
| `resultados/` | Salidas de `evaluar_comparacion.py` de esta ejecución |
| `tablas/`, `figuras/` | Tablas y figuras regeneradas con esta ejecución |

La evaluación se ejecutó en una carpeta externa al repositorio (`Reevaluacion_objetivos_4_5_20260927/`), con dos subcarpetas: `carpeta/` (la evaluada, con los modelos) y `anterior_publicado/` (copia sin modificar de lo publicado), para que el evaluador no sobrescribiera los resultados publicados. Después se copiaron aquí sus salidas.

`comparar_con_anterior.py` se ejecutó primero sobre esas dos subcarpetas y después desde el repositorio, sobre estas dos carpetas. Las dos veces produjo los mismos `diferencias.md` y `diferencias.json`.

Los modelos no se incluyen (`.gitignore`), y los datos de entrada son los ya publicados en `../Objetivos_4_y_5_comparacion/datos/`.
