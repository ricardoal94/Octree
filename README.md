# Octree: Clasificación 3D en ModelNet40 con Enfoques Clásicos y Profundos

Trabajo de grado que compara un enfoque clásico basado en descriptores manuales
extraídos de una estructura de octree (HCE + SVM/Random Forest) contra una red
convolucional jerárquica (Net5-Octree), evaluando ambos en dos resoluciones
espaciales equivalentes: **32³** y **64³**, sobre el dataset **ModelNet40**.

Repositorio: [https://github.com/ricardoal94/Octree](https://github.com/ricardoal94/Octree)

---

## Tabla de contenido

1. [Estructura del repositorio](#estructura-del-repositorio)
2. [Iteración metodológica: enfoque descartado (PointNet)](#iteración-metodológica-enfoque-descartado-pointnet)
3. [Requisitos e instalación](#requisitos-e-instalación)
4. [Dataset ModelNet40](#dataset-modelnet40)
5. [Reproducibilidad: semillas y configuración](#reproducibilidad-semillas-y-configuración)
6. [Fase 1 — Configuración y partición](#fase-1--configuración-y-partición)
7. [Fase 2 — Construcción de octrees](#fase-2--construcción-de-octrees)
8. [Fase 3a — Enfoque clásico (HCE + SVM + Random Forest)](#fase-3a--enfoque-clásico-hce--svm--random-forest)
9. [Fase 3b — Enfoque profundo (Net5-Octree)](#fase-3b--enfoque-profundo-net5-octree)
10. [Fase 4 — Comparación y visualización](#fase-4--comparación-y-visualización)
11. [Experimento adicional — Img2Voxel](#experimento-adicional--img2voxel)
12. [Herramientas de análisis y figuras](#herramientas-de-análisis-y-figuras)
13. [Especificaciones técnicas exactas](#especificaciones-técnicas-exactas)
14. [Registros y resultados](#registros-y-resultados)

---

## Estructura del repositorio

```
Octree/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── fase1_modelnet40/            # Configuración global y partición del dataset
│   ├── config.yaml
│   ├── fase1_setup.py
│   └── verificar_reproducibilidad.py
│
├── fase2_octree/                # Pipeline .off -> octree (32³ y 64³)
│   ├── octree.py
│   ├── preprocesar_octrees.py
│   └── visualizar_octree_3d.py
│
├── fase3_hce/                   # Enfoque clásico: descriptores + SVM/RF
│   ├── hce_extraccion.py
│   ├── fase3_hce_entrenamiento.py
│   ├── visualizar_resultados_hce.py
│   └── visualizar_features_3d.py
│
├── fase3_net5/                  # Enfoque profundo: Net5-Octree (3D-CNN)
│   ├── net5_modelo.py
│   ├── net5_dataset.py
│   ├── fase3_net5_entrenamiento.py
│   ├── visualizar_comparativa.py
│   └── visualizar_matriz_confusion_net5.py
│
├── fase4_comparacion/           # Comparación cualitativa e interpretabilidad
│   ├── comparar_predicciones.py
│   └── gradcam_3d.py
│
├── fase5_img2voxel/             # Experimento adicional (fuera de la metodología principal)
│   ├── renderizar_vistas.py
│   ├── img2voxel_modelo.py
│   ├── img2voxel_dataset.py
│   ├── entrenar_img2voxel.py
│   ├── visualizar_reconstruccion.py
│   └── visualizar_costo_img2voxel.py
│
├── explorado_descartado/         # Iteración metodológica previa (ver sección dedicada)
│   └── fase2_modelnet40_pointnet/
│       ├── dataset.py
│       └── modelo.py
│
├── Scripts_Analisis/             # Herramientas de medición y figuras del capítulo de metodología
│   ├── generar_figuras_metodologia.py
│   ├── medir_tiempo_memoria.py
│   ├── medir_metricas_off.py
│   ├── tabla_resumen_octree.py
│   └── resumen_metricas_completo.py
│
├── Dataset/                      # (NO versionado) ModelNet40 descargado localmente
├── data/                         # (NO versionado) Octrees y renders preprocesados
├── checkpoints/                  # (NO versionado) Pesos de modelos entrenados
├── logs/                         # (parcialmente versionado) Historiales de entrenamiento
└── Registros/                    # (versionado) CSV/JSON de métricas y resultados finales
    ├── fase1/
    ├── fase2_octree/
    ├── fase3_hce/
    ├── fase3_net5/
    └── fase5_img2voxel/
```

> **Nota:** las carpetas `Dataset/`, `data/` y `checkpoints/` están excluidas del
> control de versiones (ver `.gitignore`) porque contienen el dataset original
> (que no debe redistribuirse) y pesos de modelos regenerables mediante los
> scripts de entrenamiento. La carpeta `Registros/` sí se versiona y contiene
> únicamente los archivos ligeros de métricas (CSV/JSON) necesarios para
> analizar los resultados sin tener que re-entrenar nada.

---

## Iteración metodológica: enfoque descartado (PointNet)

> ⚠️ **Este código NO forma parte de la metodología final de la tesis.** Se
> conserva únicamente por transparencia respecto al proceso de desarrollo y
> no debe usarse para reproducir los resultados reportados en el documento.

Antes de adoptar la representación de octree (grilla voxelizada de 32³/64³
con descriptores HCE y Net5-Octree), la primera aproximación al problema
implementó una red tipo **PointNet** (con módulos T-Net de alineación
espacial) que clasificaba directamente **nubes de puntos** muestreadas sobre
la superficie de cada malla, sin ningún paso de voxelización ni construcción
de octree.

Esa implementación se encuentra archivada en `explorado_descartado/fase2_modelnet40_pointnet/`
y contiene:

- `dataset.py` — carga de nubes de puntos con aumento de datos (rotación,
  jitter, escala aleatoria).
- `modelo.py` — arquitectura Net5 con T-Net de entrada (3×3) y T-Net de
  características (64×64), siguiendo el diseño original de PointNet.

### Por qué se descartó

El primer objetivo específico de la tesis exige explícitamente un enfoque
basado en **octree** evaluado en resoluciones equivalentes de 32³ y 64³. La
representación de nube de puntos de PointNet no construye ninguna estructura
jerárquica ni voxelizada, por lo que **no satisface el objetivo tal como está
formulado**. El proyecto se reorientó completamente hacia el pipeline
`.off → normalización → muestreo de superficie → voxelización → octree`
documentado en la [Fase 2](#fase-2--construcción-de-octrees) de este README,
que sí permite evaluar el efecto de la resolución espacial sobre el
desempeño de clasificación — la variable independiente central del diseño
experimental.

### Diferencias clave frente a la metodología final

| Aspecto | Enfoque descartado (PointNet) | Metodología final (Octree) |
|---|---|---|
| Representación de entrada | Nube de puntos `(3, N)` | Grid voxelizado `(4, R, R, R)` |
| Resolución espacial evaluada | No aplica (sin discretización) | 32³ y 64³ |
| Aumento de datos | Sí (rotación, jitter, escala) | No, por criterio de equivalencia experimental |
| Descriptores clásicos comparables | No implementados | HCE (ocupación por nivel + momentos geométricos) |
| Satisface el objetivo específico de octree | No | Sí |

No se recomienda ejecutar los scripts de `explorado_descartado/` como parte
del flujo de reproducción del proyecto; se incluyen exclusivamente como
evidencia documental del proceso iterativo de diseño metodológico.

---

## Requisitos e instalación

- Python 3.10 – 3.12
- Sistema operativo probado: Windows 11 (PowerShell)
- GPU NVIDIA opcional pero recomendada para las Fases 3b y 5 (probado en RTX 5070)

```bash
# 1. Clonar el repositorio
git clone https://github.com/ricardoal94/Octree.git
cd Octree

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Instalación de PyTorch con soporte GPU (opcional pero recomendado)

Si se dispone de GPU NVIDIA, instalar la variante CUDA de PyTorch **antes** de
`pip install -r requirements.txt`, o reinstalar después si ya se instaló la
versión CPU:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verificar la instalación:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Dataset ModelNet40

### Descarga

El dataset **no se distribuye en este repositorio**. Debe descargarse manualmente
desde la fuente oficial:

- **Fuente oficial:** [https://modelnet.cs.princeton.edu](https://modelnet.cs.princeton.edu)
- **Archivo esperado:** `ModelNet40.zip` (mallas en formato `.off`, partición oficial)

### Ubicación esperada

Descomprimir el dataset de manera que quede en la siguiente ruta relativa a la
raíz del repositorio:

```
Octree/
└── Dataset/
    └── ModelNet40/
        ├── airplane/
        │   ├── train/
        │   │   ├── airplane_0001.off
        │   │   ├── airplane_0002.off
        │   │   └── ...
        │   └── test/
        │       ├── airplane_0627.off
        │       └── ...
        ├── bathtub/
        ├── bed/
        ├── ...
        └── xbox/
```

### Estructura esperada por el código

- 40 subcarpetas de clase (una por categoría de ModelNet40).
- Cada clase contiene dos subcarpetas: `train/` y `test/`, siguiendo la
  **partición oficial** del dataset (9 843 modelos de entrenamiento y
  2 468 de prueba en total).
- Los archivos son mallas en formato `.off` (vértices + caras triangulares).

Si la ruta local difiere de `Dataset/ModelNet40/`, debe actualizarse la
constante `RAIZ_DATASET` (o equivalente) al inicio de cada script que lea
directamente del dataset (`preprocesar_octrees.py`, `renderizar_vistas.py`,
`medir_metricas_off.py`, `generar_figuras_metodologia.py`).

### Cómo se distinguen train y test

La distinción entre entrenamiento y prueba se hereda directamente de la
estructura de carpetas oficial de ModelNet40 (`.../<clase>/train/` y
`.../<clase>/test/`), **no** se recalcula ni se mezcla en ningún punto del
pipeline. Del subconjunto `train/` se reserva adicionalmente un 10 % para
validación interna (ver Fase 1), manteniendo el conjunto `test/` oficial
intacto y usado únicamente para la evaluación final reportada.

---

## Reproducibilidad: semillas y configuración

Todas las operaciones estocásticas del proyecto usan una **semilla única y
fija: `seed = 42`**. Esto incluye:

| Operación                                   | Dónde se fija                          |
|----------------------------------------------|-----------------------------------------|
| Barajado de datos / partición train-val       | `fase1_setup.py::particionar_dataset`   |
| Inicialización de pesos (Net5, Img2Voxel)     | `set_global_seed()` (Python, NumPy, PyTorch CPU/GPU, CUDNN) |
| Partición del Random Forest                   | `random_state=42` en `RandomForestClassifier` |
| Muestreo de superficie (área-weighted)        | `np.random.default_rng(seed + idx)` por muestra |
| Selección de vista en Img2Voxel (train)       | `np.random.default_rng(seed + idx + hash(nombre))` |

`fase1_setup.py::set_global_seed()` fija adicionalmente `cudnn.deterministic = True`
y `cudnn.benchmark = False` para maximizar la reproducibilidad en GPU (a costa
de cierta velocidad de entrenamiento).

`fase1_modelnet40/verificar_reproducibilidad.py` ejecuta la partición del
dataset 3 veces con la misma semilla y compara los hashes SHA-256 resultantes,
confirmando reproducibilidad bit a bit.

---

## Fase 1 — Configuración y partición

Define la semilla global, la partición train/val/test y registra la
configuración del hardware y las versiones de librerías utilizadas.

```bash
cd fase1_modelnet40
python fase1_setup.py
```

**Salidas:**
- `logs/experiment_log.json` — configuración completa, hardware, versiones.
- `logs/particion_indices.npz` — índices de partición train/val (reutilizados
  por todas las fases posteriores para garantizar consistencia).

**Verificar reproducibilidad:**
```bash
python verificar_reproducibilidad.py
```

**Parámetros clave (`config.yaml`):**
- `dataset.train_total`: 9843
- `dataset.test_total`: 2468
- `dataset.val_split`: 0.10 (10 % del train reservado para validación)
- `reproducibilidad.seed`: 42

---

## Fase 2 — Construcción de octrees

Convierte cada malla `.off` en dos representaciones voxelizadas densas
(equivalentes a un octree aplanado a su nivel hoja): resolución **32³**
(profundidad `L = 5`) y **64³** (profundidad `L = 6`).

### Flujo determinista

1. **Normalización:** centrado en el origen y escalado al cubo `[-1, 1]³`.
2. **Muestreo de superficie:** muestreo *area-weighted* de puntos sobre los
   triángulos de la malla (no sobre los vértices), preservando la
   representatividad geométrica de la superficie.
3. **Cuantización jerárquica:** los puntos muestreados se asignan a celdas de
   una grilla densa de resolución `R = 2^L`.
4. **Codificación:** cada celda ocupada almacena un valor binario de ocupación
   y el vector normal promedio de los puntos que caen en ella.

### Comandos

```bash
cd fase2_octree

# Procesar el dataset completo (train + test, ambas resoluciones)
python preprocesar_octrees.py

# Verificación visual de una muestra individual
python visualizar_octree_3d.py --clase airplane --split train --indice 0
```

**Salidas:** `data/octrees_32/<clase>/<split>/<archivo>.npz` y
`data/octrees_64/<clase>/<split>/<archivo>.npz`, cada uno con:
- `grid`: array `float32` de forma `(4, R, R, R)` — canal 0 = ocupación,
  canales 1-3 = vector normal `(nx, ny, nz)`.
- `etiqueta`: entero de clase (0-39).

**Parámetros de muestreo:**
- `N_PUNTOS_MUESTREO = 20000` puntos por objeto.
- `SEED = 42`.
- `N_PROCESOS = 10` (paralelización vía `ProcessPoolExecutor`, ajustar según
  núcleos de CPU disponibles).

---

## Fase 3a — Enfoque clásico (HCE + SVM + Random Forest)

### Descriptores manuales (HCE)

`hce_extraccion.py` calcula, a partir del grid de octree, un vector de
**17 características** (resolución 32³) o **18** (resolución 64³):

| Grupo                          | Cantidad | Descripción                                                     |
|---------------------------------|----------|-------------------------------------------------------------------|
| Ocupación jerárquica por nivel  | `L` (5 o 6) | % de nodos ocupados en cada nivel del octree, de la raíz a la hoja |
| Momentos geométricos globales   | 10       | Centroide (3), varianza por eje (3), dispersión radial (1), skewness por eje (3) |
| Estadísticas del vector normal  | 2        | Norma media y varianza de las normales en celdas ocupadas          |

### Entrenamiento

```bash
cd fase3_hce
python fase3_hce_entrenamiento.py --resolucion 32
python fase3_hce_entrenamiento.py --resolucion 64
```

**Hiperparámetros:**
- **SVM:** kernel RBF, búsqueda en grilla sobre `C ∈ {0.1, 1, 10, 100}` y
  `gamma ∈ {"scale", 0.001, 0.01, 0.1}`, validación cruzada `cv=3`.
- **Random Forest:** `n_estimators=100`, `max_depth=None`, `random_state=42`.
- Features escaladas con `StandardScaler` antes del SVM (Random Forest usa
  las features sin escalar).

**Salidas:**
- `checkpoints/hce_svm_R{32,64}.joblib`
- `checkpoints/hce_rf_R{32,64}.joblib`
- `checkpoints/hce_scaler_R{32,64}.joblib`
- `resultados/resumen_hce_R{32,64}.json` (accuracy, matriz de confusión,
  reporte por clase, tiempos de inferencia, tamaño de modelo)
- `logs/hce_features_R{32,64}.npz` (caché de features extraídas)

### Visualización

```bash
python visualizar_resultados_hce.py --resolucion 32
python visualizar_features_3d.py --resolucion 32 --metodo pca
```

---

## Fase 3b — Enfoque profundo (Net5-Octree)

Red convolucional 3D jerárquica que opera directamente sobre el grid de
octree (4 canales), con un bloque por nivel de resolución (`stride=2` en cada
bloque, imitando el ascenso de nivel en el octree).

```bash
cd fase3_net5
python fase3_net5_entrenamiento.py --resolucion 32
python fase3_net5_entrenamiento.py --resolucion 64 --batch_size 8
```

**Hiperparámetros:**
- Optimizador **Adam**, `lr=0.001`, `weight_decay=1e-4`.
- Scheduler `StepLR` (`step_size=20`, `gamma=0.7`).
- **Early stopping** con `patience=20` épocas sin mejora en validación.
- `batch_size=16` por defecto (ajustar según VRAM disponible).
- Sin aumento de datos (restricción de equivalencia experimental, ver
  sección de Criterios de equivalencia más abajo).

**Salidas:**
- `checkpoints/net5_mejor_R{32,64}.pth`
- `resultados/resumen_net5_R{32,64}.json` (incluye matriz de confusión,
  reporte de clasificación, tiempos, VRAM pico, tamaño de modelo)
- `logs/net5_historial_R{32,64}.csv` / `.json` (curvas de entrenamiento)

### Visualización

```bash
python visualizar_comparativa.py --resolucion 32
python visualizar_matriz_confusion_net5.py --resolucion 32
```

---

## Fase 4 — Comparación y visualización

Comparación cualitativa de predicciones entre los tres modelos (SVM, Random
Forest, Net5) sobre una misma muestra, con visualización 3D del objeto:

```bash
cd fase4_comparacion
python comparar_predicciones.py --clase airplane --indice 0 --resolucion 32
```

Interpretabilidad de Net5 mediante Grad-CAM 3D (regiones del objeto con mayor
influencia en la decisión de clasificación; **no** es una reconstrucción
geométrica):

```bash
python gradcam_3d.py --clase airplane --indice 0 --resolucion 32
```

---

## Experimento adicional — Img2Voxel

> **Nota metodológica:** este experimento está **fuera del alcance de la
> metodología principal** de la tesis (que compara HCE vs. Net5 como
> clasificadores). Se incluye como exploración adicional de reconstrucción
> 3D aproximada a partir de una única imagen 2D, y se documenta por
> transparencia y trazabilidad.

### Renderizado de vistas 2D

```bash
cd fase5_img2voxel
python renderizar_vistas.py --n_vistas 8 --resolucion 128
```

**Salidas:** `data/renders/<clase>/<split>/<archivo>_v{00..07}.png`
(imágenes en escala de grises, 128×128 px, 8 vistas por objeto en azimuts
uniformemente espaciados).

### Entrenamiento

```bash
python entrenar_img2voxel.py --resolucion 64 --batch_size 16
```

**Arquitectura:** encoder CNN 2D (128×128 → vector latente de 512) + decoder
`ConvTranspose3D` (vector latente → grid `(4, R, R, R)`).

**Función de pérdida:** BCE ponderada (`pos_weight` automático por batch,
compensa el desbalance entre celdas ocupadas y vacías) + MSE de normales
enmascarado solo en celdas ocupadas.

**Métrica:** IoU (Intersection over Union) sobre el canal de ocupación —
métrica estándar en reconstrucción volumétrica (no comparable directamente
con el accuracy de clasificación de HCE/Net5).

**Salidas:**
- `checkpoints/img2voxel_mejor_R{32,64}.pth`
- `resultados/resumen_img2voxel_R{32,64}.json`

### Visualización

```bash
python visualizar_reconstruccion.py --clase airplane --indice 0 --resolucion 64
python visualizar_costo_img2voxel.py --resolucion 64
```

---

## Herramientas de análisis y figuras

Scripts de soporte para el capítulo de metodología (no forman parte del
pipeline experimental, documentan y validan el proceso geométrico):

```bash
cd Scripts_Analisis

# Genera las 6 figuras vectoriales del pipeline .off -> octree
# (malla normalizada, nube de puntos, voxel 32^3, voxel 64^3,
#  niveles jerarquicos L=0..4 y L=0..5)
python generar_figuras_metodologia.py --off "ruta\a\chair_0001.off" --salida figuras_tesis

# Mide tiempo y memoria de cada etapa del pipeline para un archivo especifico
python medir_tiempo_memoria.py --off "ruta\a\chair_0001.off" --repeticiones 10

# Procesa TODO el dataset y calcula metricas de ocupacion/tiempo/memoria por objeto
python medir_metricas_off.py --split ambos

# Genera la tabla resumen (errores, ocupacion media, nodos medios, tiempo medio, almacenamiento)
python tabla_resumen_octree.py

# Tabla comparativa final: SVM vs Random Forest vs Net5 (tiempo, memoria, tamaño, accuracy, epoca optima)
python resumen_metricas_completo.py
```

---

## Especificaciones técnicas exactas

Esta sección documenta explícitamente los criterios y formatos técnicos del
proyecto, para eliminar ambigüedad al reproducir o auditar los resultados.

### Criterio de ocupación de un vóxel

Una celda del grid se considera **ocupada** (`ocupación = 1.0`) si al menos
uno de los puntos muestreados sobre la superficie de la malla cae dentro de
sus límites espaciales al cuantizar las coordenadas continuas `[-1, 1]³` a
índices enteros `[0, R)`. No se aplica ningún umbral de densidad mínima en la
Fase 2; el criterio es binario y determinista dado el conjunto de puntos
muestreado.

En el experimento adicional Img2Voxel, dado que las mallas de ModelNet40 son
superficies huecas (no sólidos), la ocupación resultante es muy baja
(≈1 % en 64³). Para ese experimento específico se aplica una dilatación
morfológica (`max_pool3d`, kernel 3, 4 pasadas) sobre el canal de ocupación
únicamente durante el entrenamiento del decoder, para evitar el colapso del
modelo a predecir "todo vacío". Esta dilatación **no se aplica** en las Fases
2, 3a ni 3b.

### Tipo de dato

Todos los grids de octree y voxel se almacenan como `numpy.float32`
(4 bytes por celda), tanto para el canal de ocupación (valores `{0.0, 1.0}`)
como para los canales de normales (valores en `[-1.0, 1.0]`).

### Estructura guardada por objeto

Cada archivo `.npz` en `data/octrees_{32,64}/` contiene dos arrays:

```python
{
    "grid":     np.ndarray,  # float32, shape (4, R, R, R)
    "etiqueta": int,         # indice de clase, 0-39
}
```

### Nombres y rutas de archivos de salida

| Fase | Patrón de archivo | Contenido |
|------|--------------------|-----------|
| 1 | `logs/experiment_log.json` | Config., hardware, versiones |
| 1 | `logs/particion_indices.npz` | Índices train/val |
| 2 | `data/octrees_{R}/<clase>/<split>/<nombre>.npz` | Grid + etiqueta |
| 3a | `resultados/resumen_hce_R{R}.json` | Métricas SVM + RF |
| 3a | `checkpoints/hce_{svm,rf,scaler}_R{R}.joblib` | Modelos entrenados |
| 3b | `resultados/resumen_net5_R{R}.json` | Métricas Net5 |
| 3b | `checkpoints/net5_mejor_R{R}.pth` | Pesos del mejor modelo |
| 3b | `logs/net5_historial_R{R}.csv` | Curvas de entrenamiento |
| 5  | `data/renders/<clase>/<split>/<nombre>_v{00-07}.png` | Vistas 2D |
| 5  | `resultados/resumen_img2voxel_R{R}.json` | Métricas Img2Voxel |

### Semilla usada en cada fase

`seed = 42` en todas las fases sin excepción (ver sección de
[Reproducibilidad](#reproducibilidad-semillas-y-configuración) para el
detalle de dónde se aplica en cada caso).

### Parámetros de muestreo

- Puntos muestreados por objeto (Fases 2, 3a, 3b): **20 000**.
- Puntos muestreados por objeto (Fase 5, renderizado): **20 000** (mismo
  muestreo, reutilizado para generar los renders).
- Método: muestreo *area-weighted* (probabilidad de elegir un triángulo
  proporcional a su área), coordenadas baricéntricas uniformes dentro del
  triángulo elegido.

### Parámetros de HCE

Ver tabla de descriptores en la sección [Fase 3a](#fase-3a--enfoque-clásico-hce--svm--random-forest).
Total de features: `L + 12`, donde `L` es la profundidad del octree
(`L=5` para 32³, `L=6` para 64³).

### Hiperparámetros de SVM y Random Forest

| Modelo | Hiperparámetro | Valor / rango de búsqueda |
|--------|------------------|------------------------------|
| SVM | kernel | RBF |
| SVM | `C` | búsqueda en `{0.1, 1, 10, 100}` |
| SVM | `gamma` | búsqueda en `{"scale", 0.001, 0.01, 0.1}` |
| SVM | validación cruzada | `cv=3` (`GridSearchCV`) |
| Random Forest | `n_estimators` | 100 |
| Random Forest | `max_depth` | `None` (sin límite) |
| Random Forest | `random_state` | 42 |

### Criterios de equivalencia experimental

Para que la comparación entre enfoques sea válida:
- Todas las pruebas de inferencia se ejecutan en el mismo equipo (CPU/GPU
  documentado en `logs/experiment_log.json`).
- **No se aplica aumento de datos** en ningún enfoque (HCE ni Net5), para
  aislar el efecto de la resolución del octree.
- La exactitud se reporta como promedio global sobre el conjunto de prueba
  oficial de ModelNet40 (2 468 muestras), nunca sobre el conjunto de
  validación interno.

---

## Registros y resultados

Los archivos de métricas y resultados finales (ligeros, en formato CSV/JSON)
se organizan en la carpeta `Registros/` para su consulta directa sin
necesidad de re-ejecutar el pipeline completo:

```
Registros/
├── fase1/
│   └── experiment_log.json
├── fase2_octree/
│   ├── metricas_off_completo.csv
│   ├── metricas_off_resumen.csv
│   ├── metricas_off_global.json
│   ├── resumen_octree_train.json
│   └── resumen_octree_test.json
├── fase3_hce/
│   ├── resumen_hce_R32.json
│   └── resumen_hce_R64.json
├── fase3_net5/
│   ├── resumen_net5_R32.json
│   ├── resumen_net5_R64.json
│   ├── net5_historial_R32.csv
│   └── net5_historial_R64.csv
└── fase5_img2voxel/
    ├── resumen_img2voxel_R32.json
    └── resumen_img2voxel_R64.json
```

Estos archivos son la fuente de verdad para las tablas y figuras reportadas
en el documento de tesis, y permiten auditar cualquier cifra citada sin
necesidad de acceso al dataset completo ni a los pesos de los modelos.
