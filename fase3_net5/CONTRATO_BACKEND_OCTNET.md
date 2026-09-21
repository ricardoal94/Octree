# Contrato mínimo del backend OctNet

Este contrato separa una implementación OctNet válida de una CNN 3D densa.

## Representación y operaciones

- La entrada debe conservar la estructura jerárquica del octree y sus hojas.
- Convolución y pooling deben operar sobre esa estructura dispersa.
- Ningún paso del backend oficial puede materializar un tensor completo
  `(C, R, R, R)` ni delegar la convolución principal a `torch.nn.Conv3d`.
- Debe documentarse la conversión entre el octree serializado del proyecto y
  el `grid-octree` esperado por la biblioteca o implementación elegida.

## Arquitectura

- Usar la topología de capacidad fija de la Tabla 5 del material
  suplementario de OctNet.
- Mantener el mismo número de parámetros entrenables en R=32 y R=64.
- Adaptar la entrada a cuatro atributos por hoja: ocupación, `nx`, `ny`, `nz`.
- Producir 40 logits; `CrossEntropyLoss` aplica internamente la normalización.

## Protocolo experimental

- Semilla global: 42.
- Train/validación: partición estratificada 90/10 usada por el Objetivo 2.
- Test: las 2.468 muestras oficiales completas.
- Identidad de objetos compartida entre R=32, R=64 y HCE mediante
  `logs/particion_objetivo2_modelos.json`.
- Sin aumento de datos, para preservar la equivalencia experimental.

## Criterios automáticos de aceptación

1. Forward con salida `(B, 40)` para R=32 y R=64.
2. Backward con gradientes finitos en todos los parámetros entrenables.
3. Guardado y recarga de checkpoint con logits reproducibles en modo eval.
4. Igualdad del conteo de parámetros entre las dos resoluciones.
5. Prueba que rechace o detecte cualquier expansión densa `R³` en el camino
   oficial.
6. Medición CUDA sincronizada de tiempo y memoria pico.
7. Resultados con esquema, commit, configuración, conteos y rutas portables.

Hasta satisfacer estos puntos, cualquier ejecución de `DenseTabla5Reference`
es exclusivamente diagnóstica y no cierra el Objetivo 3.
