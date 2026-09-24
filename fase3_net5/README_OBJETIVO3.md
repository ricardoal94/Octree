# Objetivo específico 3 — estado del enfoque profundo

## Estado actual

Ya existe un **backend OctNet nativo de referencia** que opera directamente
sobre hojas jerárquicas y no usa `torch.nn.Conv3d` ni materializa el volumen
de entrada `(4, R, R, R)`. La implementación está conectada a la topología de
capacidad fija de la Tabla 5 y al script de entrenamiento.

El núcleo geométrico y su equivalencia matemática con convolución y pooling
densos están cubiertos por pruebas independientes de PyTorch. Todavía faltan
dos validaciones operativas para cerrar el objetivo:

- ejecutar en el equipo con PyTorch las pruebas de forward, backward y
  checkpoint del backend completo;
- medir su rendimiento y entrenar R=32/R=64 sobre ModelNet40 completo.

La antigua referencia densa se conserva únicamente para diagnóstico y nunca
debe denominarse Net5/OctNet en los resultados.

El contrato que debe cumplir ese backend está en
[`CONTRATO_BACKEND_OCTNET.md`](CONTRATO_BACKEND_OCTNET.md).

## Arquitectura de referencia

La variante de capacidad fija está descrita en la **Tabla 5**, no en la Tabla
4, del material suplementario de *OctNet: Learning Deep 3D Representations at
High Resolutions*. La adaptación diagnóstica usa cuatro canales de entrada
(ocupación y normal promedio) y 40 logits para ModelNet40.

`net5_modelo.py` expone dos rutas separadas:

- `crear_modelo()` construye `Net5Octree` sobre el backend disperso;
- `crear_modelo_denso_referencia()` construye la referencia diagnóstica con
  `Conv3d`.

## Conversión al grid-octree

`grid_octree.py` convierte el octree global del Objetivo 1 a la estructura
híbrida del artículo:

1. R=32 se divide en una rejilla de `4 x 4 x 4` octrees; R=64, en una de
   `8 x 8 x 8`.
2. Cada octree de la rejilla tiene profundidad máxima 3 y cubre `8^3`
   posiciones de la resolución efectiva.
3. Las ramas podadas del NPZ se recuperan como hojas vacías implícitas; las
   hojas ocupadas conservan `[ocupación, nx, ny, nz]`.
4. La convolución 3x3x3 usa intersecciones entre hojas para implementar el
   promedio de la ecuación 8 sin `oc2ten` materializado.
5. El max-pooling transforma la jerarquía según la ecuación 10. Solo al llegar
   a la salida final 8^3 se expande ese pequeño tensor para la capa FC de la
   Tabla 5.

## Partición experimental

`particion_objetivo2.py` reproduce la partición empleada realmente por HCE:

```python
train_test_split(
    indices,
    test_size=0.10,
    random_state=42,
    shuffle=True,
    stratify=y,
)
```

El manifiesto `logs/particion_objetivo2_modelos.json` guarda identificadores
portables `clase/modelo`, no posiciones dependientes del equipo. El mismo
manifiesto debe validarse en R=32 y R=64, de modo que ambos enfoques usen
exactamente los mismos objetos en train, validación y test.

## Pruebas

Las pruebas generales, que no requieren PyTorch ni el dataset local, se
ejecutan con:

```bash
python -m pytest
```

Las pruebas del backend PyTorch se omiten automáticamente si PyTorch no está
instalado:

```bash
python -m pytest tests/test_octnet_backend.py -v
```

La integración densa histórica con los octrees reales sigue siendo local y
diagnóstica. Si PyTorch o `data/octrees_{32,64}` no están disponibles, pytest
la omite:

```bash
python -m pytest -m dataset tests/test_net5_integracion.py -v
```

También puede generarse un JSON diagnóstico con valores de forward, backward
y recarga de checkpoint:

```bash
python fase3_net5/verificar_integracion_net5.py
```

La salida predeterminada es
`resultados/objetivo3/verificacion_dense_tabla5.json` y queda marcada con
`valido_como_resultado_objetivo3: false`.

## Smoke test de la referencia densa

La ejecución exige `--backend dense_reference` y un `--tag` para evitar que
se confunda con una corrida oficial:

```bash
python fase3_net5/fase3_net5_entrenamiento.py \
  --resolucion 32 --backend dense_reference --tag _smoke_dense \
  --limite_train 80 --limite_val 40 --limite_test 80 \
  --epochs 2 --batch_size 4 --num-workers 0
```

Los límites seleccionan muestras con cobertura balanceada de clases; no se
toman simplemente los primeros objetos del test. El informe resultante usa
el esquema `dense-table5-diagnostic` y nunca debe incorporarse a las tablas
comparativas del Objetivo 3.

## Smoke test del backend nativo

Antes del entrenamiento completo debe ejecutarse una muestra corta en el
equipo con PyTorch y los NPZ:

```bash
python fase3_net5/fase3_net5_entrenamiento.py \
  --resolucion 32 --backend octree_native --tag _smoke_native \
  --limite_train 40 --limite_val 40 --limite_test 40 \
  --epochs 1 --batch_size 1 --num-workers 4 \
  --exigir-git-limpio --exigir-git-publicado --lotes-perfil 3
```

Una corrida limitada queda marcada como `PARCIAL_SMOKE`. Solo una corrida
nativa sin límites puede marcarse como resultado completo del Objetivo 3.
El backend usa `batch_size=1` de forma predeterminada porque los planes
dispersos dependen de la topología de cada objeto. Antes de aumentarlo se debe
comprobar la VRAM con R=64.

Las opciones de trazabilidad capturan la rama, el commit, el upstream y el
estado del repositorio antes de crear resultados; la corrida se detiene si
existen cambios locales o si el commit no está publicado. El resumen incluye
además un bloque `entorno_ejecucion` con sistema operativo, versiones de
Python, PyTorch, CUDA y cuDNN, modelo de CPU, núcleos lógicos, RAM total y,
cuando corresponde, nombre, capacidad de cómputo y VRAM de la GPU. No se debe
completar esta información manualmente. También incluye un
`perfil_rendimiento` que separa carga del lote, transferencia de atributos,
preparación y transferencia de planes, y el resto del `forward`.
Para R=64 se repite el mismo comando cambiando `--resolucion 32` por
`--resolucion 64`.

La construcción vectorizada de planes evita escanear volúmenes R³. De forma
predeterminada, los planes se preparan dentro del `DataLoader`; con
`num_workers > 0` pueden construirse en paralelo antes de que el lote llegue
al proceso que controla la GPU. Para `batch_size=1`, el backend reutiliza
directamente los arreglos de cada geometría y evita concatenaciones completas.
El barrido limita a uno los hilos internos de OpenBLAS, OMP, MKL y NumExpr para
evitar que cada trabajador multiplique el consumo de memoria. Los trabajadores
de entrenamiento y validación se cierran al terminar cada recorrido; solo los
de test permanecen activos porque ese cargador se reutiliza en las mediciones.

El resumen separa dos medidas: `tiempo_inferencia_promedio_ms` incluye solo el
`forward`, mientras `tiempo_pipeline_promedio_ms` incluye carga, preparación
geométrica, transferencia y `forward`. También registra memoria e
interacciones de los planes. La opción `--sin-precalcular-planes` se reserva
para comparar contra la ruta anterior y no se recomienda para las corridas
completas.

La trazabilidad distingue un árbol limpio de un commit publicado. Para una
corrida verificable deben usarse conjuntamente `--exigir-git-limpio` y
`--exigir-git-publicado`; una evaluación completa queda marcada como no válida
si el commit no aparece en una referencia remota conocida.

Antes de programar las corridas completas se debe comparar el pipeline con
`num_workers=0`, `4` y `8`. Si la preparación geométrica continúa dominando el
tiempo aun en paralelo, el siguiente paso técnico será trasladar esa fase a
una extensión C++/CUDA, sin cambiar la semántica ya validada.

El barrido completo de smoke tests se ejecuta con un solo comando desde la
raíz del repositorio:

```bash
python fase3_net5/comparar_workers_octnet.py
```

El script ejecuta las seis combinaciones R32/R64 por 0/4/8 trabajadores con
etiquetas independientes, exige un commit limpio y publicado, y comprueba la
configuración, la trazabilidad y las métricas del backend. Los checkpoints,
logs y resultados se escriben en la carpeta hermana `<repositorio>_smoke_workers`
para que la primera corrida no ensucie el repositorio e invalide las demás.
Al finalizar genera `comparacion_workers_octnet.json` y
`comparacion_workers_octnet.csv`; ambos incorporan el entorno experimental y
el JSON registra también el número de trabajadores con menor tiempo integral
para cada resolución. El validador exige que las seis corridas pertenezcan al
mismo equipo y entorno de software, de modo que los resúmenes con el contrato
anterior deben regenerarse y no pueden mezclarse con los nuevos.

Si una ejecución se interrumpe, puede continuarse sin repetir las combinaciones
que ya sean válidas:

```bash
python fase3_net5/comparar_workers_octnet.py --continuar
```

Para auditar resultados existentes sin entrenar de nuevo se usa
`--solo-validar`; para revisar los seis comandos sin ejecutarlos se usa
`--mostrar-comandos`.

## Pendiente para cerrar el Objetivo 3

1. Ejecutar las pruebas PyTorch y el smoke test nativo en la máquina de
   entrenamiento.
2. Medir tiempo de `forward`, tiempo integral, memoria de planes y VRAM con
   distintos números de trabajadores; si es necesario, optimizar el plan
   disperso antes de las corridas completas.
3. Entrenar R=32 y R=64 completos con el mismo manifiesto del Objetivo 2.
4. Evaluar las 2.468 muestras del test oficial y generar evidencia trazable.

El unpooling descrito por el artículo no forma parte de Net5 de clasificación
y por eso no se incluye en esta ruta. Será necesario únicamente si se adopta
una arquitectura de decodificación o segmentación.
