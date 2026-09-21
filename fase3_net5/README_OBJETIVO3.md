# Objetivo específico 3 - Net5 (enfoque profundo basado en octrees)

Este flujo implementa, entrena y configura Net5 -- la variante de **capacidad
fija** de OctNet (Riegler, Ulusoy, Geiger, *"OctNet: Learning Deep 3D
Representations at High Resolutions"*, CVPR 2017), documentada en la **Tabla
4** del material suplementario del paper -- adaptada a 4 canales de entrada
(ocupación + normal promedio) y 40 clases de salida (ModelNet40), bajo las
mismas particiones de datos y semilla (`seed=42`) del Objetivo 2.

Ver el docstring de [`net5_modelo.py`](net5_modelo.py) para el detalle
completo de la arquitectura y su correspondencia con la Tabla 4 del paper.

## Prerrequisito

Los octrees definitivos de `data/octrees_32/` y `data/octrees_64/` deben
existir (generados en la Fase 2), y la partición Objetivo 1/2 debe estar
disponible en `logs/particion_indices.npz` (generada por
`fase1_modelnet40/fase1_setup.py`, `seed=42`).

## 1. Prueba de integración corta (recomendada antes de cualquier entrenamiento)

```bash
python -m pytest tests/test_net5_integracion.py -v
```

Para R=32 y R=64, sobre un subconjunto pequeño (8 muestras `train` + 4
`val`) tomado de la partición real del Objetivo 2, verifica:

1. Carga de datos reales (octree disperso `.npz` -> grid denso `(4,R,R,R)`).
2. Dimensiones de entrada/salida del modelo.
3. Propagación hacia adelante (`forward`).
4. Propagación hacia atrás (`backward`, gradientes no nulos en todos los
   parámetros).
5. Cálculo de la pérdida (`CrossEntropyLoss`).
6. Guardado y recarga del checkpoint (los logits del modelo recargado
   coinciden con los del original).
7. Igualdad del número de parámetros entre R=32 y R=64 (propiedad de
   capacidad fija de la Tabla 4).

Evidencia de la última ejecución: `resultados/evidencia_prueba_integracion_net5.txt`.

### 1a. Informe detallado con valores medidos (para documentar el Objetivo 3)

`pytest` solo informa pass/fail. Para un informe con los valores concretos
de cada punto (dimensiones, número de parámetros, valor de la pérdida,
norma del gradiente, tamaño del checkpoint en MB, etc.), ejecute:

```bash
python fase3_net5/verificar_integracion_net5.py
```

Genera `resultados/verificacion_integracion_net5.json` con, para R=32 y
R=64: origen y semilla de la partición usada, forma y dtype del grid
cargado, arquitectura y número de parámetros del modelo, forma y rango de
los logits de salida, valor de `CrossEntropyLoss`, número de parámetros
que recibieron gradiente y su norma L2 total, verificación de que
`optimizer.step()` efectivamente modificó los pesos, y tamaño en MB +
verificación de que los logits coinciden tras recargar el checkpoint.
También reporta si el número de parámetros es idéntico entre R=32 y R=64
(propiedad de capacidad fija de la Tabla 4 de OctNet).

Argumentos opcionales: `--n-train`, `--n-val`, `--resoluciones`, `--salida`.

### 1b. Smoke test del script de entrenamiento real (opcional)

Para ejercitar `fase3_net5_entrenamiento.py` de punta a punta (loop de
entrenamiento, scheduler, early stopping, matriz de confusión, medición de
tiempo de inferencia y guardado de checkpoint) sin correr las ~9 800 muestras
completas, use `--limite_train/--limite_val/--limite_test` con `--tag` para
no pisar los archivos de una corrida real:

```bash
python fase3_net5/fase3_net5_entrenamiento.py --resolucion 32 \
  --limite_train 10 --limite_val 4 --limite_test 6 \
  --epochs 2 --batch_size 4 --patience 5 --tag _smoke

python fase3_net5/fase3_net5_entrenamiento.py --resolucion 64 \
  --limite_train 10 --limite_val 4 --limite_test 6 \
  --epochs 2 --batch_size 4 --patience 5 --tag _smoke
```

Genera `checkpoints/net5_mejor_R{32,64}_smoke.pth`,
`logs/net5_historial_R{32,64}_smoke.{csv,json}` y
`resultados/resumen_net5_R{32,64}_smoke.json` (no versionados: son solo para
verificación local). Ambas resoluciones corrieron exitosamente en la GPU de
desarrollo (RTX 5070) en ~5 s por corrida.

## 2. Entrenamiento completo

```bash
python fase3_net5/fase3_net5_entrenamiento.py --resolucion 32
python fase3_net5/fase3_net5_entrenamiento.py --resolucion 64
```

Argumentos opcionales: `--epochs`, `--patience` (early stopping sobre
`val_acc`), `--batch_size`, `--lr`.

Usa la partición train/val/test del Objetivo 2 (`logs/particion_indices.npz`,
`seed=42`), Adam + `CrossEntropyLoss`, *early stopping* según val accuracy, y
no aplica aumento de datos (criterio de equivalencia, sección 6.6 de la
metodología). Cada ejecución guarda:

```text
checkpoints/net5_mejor_R32.pth
checkpoints/net5_mejor_R64.pth
logs/net5_historial_R32.csv / .json
logs/net5_historial_R64.csv / .json
resultados/resumen_net5_R32.json
resultados/resumen_net5_R64.json
```

El resumen incluye exactitud de validación/prueba, matriz de confusión y
reporte de clasificación por clase (40 clases), tiempo de entrenamiento e
inferencia, VRAM pico y tamaño del modelo en disco -- insumos para la Fase 4
(medición de costo computacional) y la comparación con el enfoque HCE del
Objetivo 2.
