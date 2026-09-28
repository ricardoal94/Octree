> **Comparación final R32/R64:** [informe y reproducción](resultados/Objetivos_4_y_5_comparacion/README.md). Auditoría sin GPU: `python fase4_comparacion/auditar_comparacion.py`.

# Octree clásico y Net5 — ModelNet40

Esta versión integra los experimentos de clasificación R32/R64 y su comparación. Las secciones siguientes conservan el procedimiento de construcción y validación del objetivo 1.

Este repositorio contiene el flujo reproducible para construir, persistir y
comparar dos representaciones tridimensionales de ModelNet40:

1. una rejilla densa de referencia;
2. un octree adaptativo real, con nodo raíz, hasta ocho hijos por nodo y poda
   explícita de regiones vacías.

El alcance de esta versión termina en la validación de las representaciones.
**HCE, SVM, Random Forest, Net5 e Img2Voxel están fuera del objetivo específico
1 y no forman parte de este procedimiento de aceptación.** Sus directorios se
conservan como trabajo posterior, pero no deben ejecutarse hasta aprobar esta
etapa.

## Estado de aceptación

El código implementa el formato definitivo, las pruebas y la generación de
métricas. La implementación correspondiente al objetivo específico 1 se valida
con tres niveles de evidencia: pruebas sintéticas autocontenidas, la prueba
controlada de `chair_0001` y una muestra pequeña de objetos reales de distintas
categorías y particiones. **No es necesario procesar ModelNet40 completo para
aceptar esta implementación.**

La regeneración de los 12 311 modelos se realizará posteriormente, antes de la
extracción de características y del entrenamiento de los clasificadores. Los
resultados antiguos de `resultados/` no sustituyen las nuevas pruebas
controladas.

## Componentes relevantes

```text
Octree/
├── fase2_octree/
│   ├── octree.py                    # lectura, normalización y rejilla densa
│   ├── octree_real.py               # árbol adaptativo y formato NPZ v1
│   ├── validacion_objetivo1.py      # equivalencia y manifiestos
│   ├── preprocesar_octrees.py       # muestra controlada o generación global
│   ├── validar_chair_0001.py        # prueba controlada independiente
│   └── visualizar_arbol_real.py     # visualización opcional con rutas CLI
├── Scripts_Analisis/
│   ├── medir_tiempo_memoria.py      # medición controlada de un modelo
│   └── medir_metricas_off.py        # verificación de muestra o conjunto global
├── tests/                            # pruebas sintéticas autocontenidas
├── requirements-objetivo1.txt
├── pytest.ini
└── README.md
```

No hay rutas absolutas en los componentes del objetivo 1. Todas las
ubicaciones se resuelven desde el repositorio o se reciben como argumentos.

## Instalación desde un clon limpio

Requisitos: Git y Python 3.10 o posterior.

### Windows PowerShell

```powershell
git clone https://github.com/ricardoal94/Octree.git
cd Octree
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-objetivo1.txt
```

### Linux o macOS

```bash
git clone https://github.com/ricardoal94/Octree.git
cd Octree
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-objetivo1.txt
```

Las dependencias de aprendizaje automático no son necesarias para este
objetivo.

## Ubicación de ModelNet40

El dataset no se versiona en Git. La estructura esperada es la partición
oficial en formato OFF:

```text
ModelNet40/
├── airplane/
│   ├── train/*.off
│   └── test/*.off
├── chair/
│   ├── train/chair_0001.off
│   └── test/*.off
└── ... 38 categorías adicionales
```

Puede colocarse en `Dataset/ModelNet40/` o indicarse mediante
`--dataset-root`. No es necesario editar ningún archivo Python.

## 1. Pruebas automatizadas

Desde la raíz del repositorio:

```bash
python -m pytest
```

Las pruebas son autocontenidas y no dependen de octrees o métricas generados
previamente. Verifican:

- correspondencia celda por celda entre rejilla densa y hojas del octree en
  32³ y 64³;
- casos sobre fronteras del dominio y planos de subdivisión;
- persistencia y reconstrucción completa de la jerarquía;
- rechazo explícito de archivos antiguos sin versión;
- inclusión de la coherencia normal en la carga binaria;
- determinismo de semillas y de la estructura serializada;
- generación integrada de NPZ, manifiesto y métricas.

## 2. Prueba controlada con `chair_0001`

Esta prueba vuelve a crear todo desde la malla original; no consume resultados
anteriores:

```bash
python fase2_octree/validar_chair_0001.py \
  --off Dataset/ModelNet40/chair/train/chair_0001.off
```

En Windows PowerShell puede escribirse en una sola línea. La salida por defecto
es:

```text
resultados/objetivo1/validacion_chair_0001.json
```

La prueba debe superar tanto 32³ como 64³ antes de procesar el dataset.

## 3. Prueba técnica con un solo modelo

Para comprobar rápidamente la integración del preprocesador:

```bash
python fase2_octree/preprocesar_octrees.py \
  --dataset-root Dataset/ModelNet40 \
  --limite 1 \
  --procesos 1 \
  --sobrescribir
```

El resumen indicará `Alcance: PRUEBA PARCIAL`. Esto no es un fallo ni pretende
ser la evidencia principal de aceptación.

## 4. Muestra controlada para aceptar el objetivo 1

Después de validar `chair_0001`, se procesa una muestra determinista pequeña.
El siguiente ejemplo toma un objeto de `train` y uno de `test` para cinco
categorías de geometría diferente, siempre en 32³ y 64³:

```bash
python fase2_octree/preprocesar_octrees.py \
  --dataset-root Dataset/ModelNet40 \
  --output-root data/objetivo1_muestra \
  --resultados-dir resultados/objetivo1/muestra_controlada \
  --categorias airplane car chair sofa table \
  --muestra-por-categoria-split 1 \
  --resoluciones 32 64 \
  --n-puntos 20000 \
  --semilla 42 \
  --procesos 4 \
  --sobrescribir
```

La selección es canónica e independiente del sistema operativo. El resumen
debe indicar `Alcance: MUESTRA CONTROLADA`, cero modelos fallidos y resultados
de equivalencia verdaderos antes y después de la recarga.

Para verificar y consolidar esta muestra:

```bash
python Scripts_Analisis/medir_metricas_off.py \
  --manifests-dir data/objetivo1_muestra/manifests \
  --output-root data/objetivo1_muestra \
  --resultados-dir resultados/objetivo1/muestra_controlada \
  --permitir-incompleto
```

La opción `--permitir-incompleto` significa aquí que se está verificando una
muestra declarada, no que se toleren errores de formato, integridad,
equivalencia o metadatos.

## 5. Generación completa futura

La siguiente ejecución **no forma parte de la aceptación inmediata del
objetivo específico 1**. Se conserva para generar las entradas definitivas
antes de comenzar la extracción de características y los clasificadores:

```bash
python fase2_octree/preprocesar_octrees.py \
  --dataset-root Dataset/ModelNet40 \
  --output-root data \
  --resultados-dir resultados/objetivo1 \
  --resoluciones 32 64 \
  --n-puntos 20000 \
  --semilla 42 \
  --procesos 8 \
  --sobrescribir
```

El número de procesos debe ajustarse a la memoria y a los núcleos disponibles.
En esa ejecución futura deben omitirse `--limite`, `--categorias` y
`--muestra-por-categoria-split`.

Para cada malla se realiza el siguiente procedimiento:

1. lectura y normalización al cubo `[-1, 1]³`;
2. muestreo de 20 000 puntos con una semilla estable por modelo;
3. construcción del octree de profundidad 5 y 6 usando la misma nube;
4. construcción independiente de la rejilla densa de 32³ y 64³;
5. comparación exacta de ocupación y comparación numérica de normales;
6. guardado del octree, recarga y reconstrucción de la jerarquía;
7. repetición de la prueba de equivalencia después de cargar;
8. escritura atómica del manifiesto y las métricas.

Una muestra controlada nunca se presenta como ModelNet40 completo. De igual
forma, una salida global parcial o con errores nunca se marca como completa.

## Formato definitivo del octree

La única versión admitida es:

```text
format_name    = octree_adaptativo_modelnet40
format_version = 1.0.0
```

Cada NPZ conserva:

| Campo | Tipo | Significado |
|---|---:|---|
| `profundidades` | `uint8[M]` | Profundidad de cada nodo en DFS preorden |
| `mascaras` | `uint8[M]` | Bits que indican cuáles de los ocho hijos existen |
| `normales` | `float32[M,3]` | Normal unitaria; cero en nodos internos |
| `coherencias` | `float32[M]` | Magnitud de la normal media antes de normalizar |
| `etiqueta` | `int16` | Índice de categoría ModelNet40 |
| `profundidad_max` | `uint8` | 5 para 32³ y 6 para 64³ |
| `resolucion` | `uint16` | Resolución espacial equivalente |
| `model_id` | texto | Identificador de la malla |
| `categoria` | texto | Categoría ModelNet40 |
| `split` | texto | `train` o `test` |
| `semilla_muestreo` | `int64` | Semilla específica del modelo |
| `n_puntos_muestreo` | `int64` | Número de puntos utilizados |
| `archivo_origen_sha256` | texto | Huella de la malla original |

El centro y el tamaño de cada nodo se reconstruyen de forma determinista a
partir de su ruta en el árbol. Los lectores rechazan archivos antiguos o
incompatibles; no existe una conversión silenciosa.

La carga estructural sin comprimir es de 18 bytes por nodo:

```text
1 profundidad + 1 máscara + 12 normal + 4 coherencia
```

Además se reporta la suma sin comprimir de todos los campos del NPZ, incluidos
los metadatos.

## Manifiesto por modelo

Los manifiestos se generan en:

```text
data/manifests/<categoria>/<split>/<model_id>.json
```

Cada uno registra:

- identificador, categoría, partición y SHA-256 de la malla;
- semilla base, semilla derivada, algoritmo de derivación y puntos muestreados;
- versión del formato;
- archivos NPZ, tamaños y SHA-256;
- métricas completas para 32³ y 64³;
- resultados de equivalencia antes del guardado y después de la carga.

La semilla específica se deriva mediante SHA-256 de la ruta relativa. Por
ello no depende del orden de los procesos, del sistema operativo ni del número
de trabajadores.

## Métricas reportadas por modelo y resolución

| Métrica | Definición |
|---|---|
| Nodos por nivel | Nodos existentes desde la raíz hasta la profundidad máxima |
| Nodos totales | Nodos internos más hojas ocupadas |
| Hojas ocupadas | Celdas ocupadas en el nivel hoja |
| Ocupación | `100 × hojas / R³` |
| Tiempo de construcción | Tiempo de `construir_octree`, en milisegundos |
| Memoria pico | Pico transitorio medido con `tracemalloc`, en bytes |
| Estructura Python | Grafo de objetos retenido por el árbol, sin doble conteo |
| Binario sin comprimir | Payload de todos los arrays del NPZ antes de ZIP |
| NPZ real | Tamaño del archivo comprimido en disco |
| Referencia densa | Tensor `float32` de forma `(4,R,R,R)` y su NPZ comparable |

Las magnitudes no se mezclan: memoria pico, estructura residente, payload
binario y archivo comprimido se conservan como mediciones distintas.

## 6. Auditoría futura de ModelNet40 completo

Después de generar todos los objetos en la etapa futura:

```bash
python Scripts_Analisis/medir_metricas_off.py \
  --manifests-dir data/manifests \
  --output-root data \
  --resultados-dir resultados/objetivo1
```

El consolidador comprueba:

- 9 843 modelos de entrenamiento y 2 468 de prueba;
- exactamente un manifiesto por modelo;
- dos resoluciones por manifiesto;
- equivalencia antes y después de cargar;
- existencia y SHA-256 de cada NPZ;
- coincidencia entre metadatos del NPZ y del manifiesto.

Produce:

```text
resultados/objetivo1/metricas_modelnet40_verificadas.csv
resultados/objetivo1/resumen_metricas_modelnet40.json
```

Este modo estricto audita los conteos oficiales y no usa
`--permitir-incompleto`.

## Criterio final de aprobación

La implementación correspondiente al objetivo específico 1 puede aprobarse si:

- `python -m pytest` termina sin fallos;
- `chair_0001` supera 32³ y 64³;
- la muestra controlada incluye objetos de distintas categorías y las
  particiones `train` y `test`;
- no hay modelos fallidos en la muestra;
- todas las equivalencias son verdaderas antes del guardado y después de la
  carga;
- cada archivo conserva la jerarquía completa, la versión y los metadatos;
- las métricas estructurales y de almacenamiento quedan registradas;
- el código y los resultados consolidados quedan asociados a un commit
  identificable.

Después de aceptar esta lista se planifica la regeneración completa de
ModelNet40. La extracción HCE, la clasificación y el aprendizaje profundo solo
podrán iniciarse cuando esa generación global haya terminado y haya sido
auditada con el modo estricto.

El procedimiento reproducible para iniciar el objetivo específico 2 está en
[`fase3_hce/README_OBJETIVO2.md`](fase3_hce/README_OBJETIVO2.md). Incluye la
auditoría exclusiva de `train`, la generación del contrato versionado de
características y el bloqueo automático del entrenamiento mientras la evidencia
sea incompleta.
