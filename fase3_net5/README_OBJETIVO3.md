# Objetivo específico 3 — estado del enfoque profundo

## Estado actual

El backend **OctNet nativo todavía no está implementado**. El código incluido
en esta carpeta permite probar la topología de capacidad fija de la Tabla 5
del material suplementario de Riegler, Ulusoy y Geiger (CVPR 2017), pero lo
hace con `torch.nn.Conv3d` sobre una rejilla densa `(4, R, R, R)`.

Por tanto:

- la referencia densa no debe denominarse Net5/OctNet en los resultados;
- sus métricas no son evidencia válida del Objetivo 3;
- los entrenamientos oficiales deben esperar al backend que opere
  directamente sobre la jerarquía `grid-octree`.

El contrato que debe cumplir ese backend está en
[`CONTRATO_BACKEND_OCTNET.md`](CONTRATO_BACKEND_OCTNET.md).

## Arquitectura de referencia

La variante de capacidad fija está descrita en la **Tabla 5**, no en la Tabla
4, del material suplementario de *OctNet: Learning Deep 3D Representations at
High Resolutions*. La adaptación diagnóstica usa cuatro canales de entrada
(ocupación y normal promedio) y 40 logits para ModelNet40.

`net5_modelo.py` conserva esa topología únicamente como referencia densa. La
función genérica `crear_modelo()` falla de forma explícita para impedir que
esa CNN se presente accidentalmente como OctNet.

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

La integración densa con los octrees reales es una prueba local y
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

## Pendiente para cerrar el Objetivo 3

1. Implementar o integrar operaciones de convolución, pooling y unpooling
   directamente sobre el `grid-octree`.
2. Conectar la topología de capacidad fija de la Tabla 5 a ese backend.
3. Verificar forward, backward, checkpoint y consumo de memoria sin expandir
   las entradas a `R³`.
4. Entrenar R=32 y R=64 completos con el mismo manifiesto del Objetivo 2.
5. Evaluar las 2.468 muestras del test oficial y generar evidencia trazable.
