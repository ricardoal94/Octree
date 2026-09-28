# Evidencia para el cierre de los objetivos 4 y 5

Revisión del 27 de septiembre de 2026 (Colombia), posterior a la reevaluación de los autores. Este documento reúne la evidencia experimental y el texto propuesto para el informe; la aprobación académica corresponde al director.

## Correspondencia con la propuesta

| Objetivo | Evidencia y alcance | Estado |
|---|---|---|
| 4. Definir y ejecutar un protocolo reproducible, con particiones, semillas, configuraciones y registro de resultados | Partición por objetos y semilla 42; programas y backend integrados; modelos identificados por SHA-256; entorno, versión del código, comandos y registro de reevaluación. | Evidencia experimental reunida y verificada. |
| 5. Comparar exactitud, matrices de confusión, tiempos, memoria, tamaño y escalabilidad por resolución | Seis casos (tres métodos, dos resoluciones), 2.468 objetos de test; métricas globales y por clase; nueve comparaciones de McNemar; tiempos de entrenamiento e inferencia, memoria, tamaños, tablas y figuras. | Comparación realizada; conclusiones válidas dentro de los límites declarados. |

Fuentes: [informe de comparación](README.md), [reevaluación](../Objetivos_4_y_5_reevaluacion_2026-09-27/REEVALUACION.md), [registro de ejecución](../Objetivos_4_y_5_reevaluacion_2026-09-27/registro_ejecucion.log) y [diferencias entre ejecuciones](../Objetivos_4_y_5_reevaluacion_2026-09-27/diferencias.md).

## Qué se comprobó

Los autores ejecutaron la evaluación con el código del commit `8147a6e018243f351c97dca25d3e2d5621a2294b` y conservaron la nueva salida en una carpeta separada. La revisión posterior verificó que los CSV completos de predicciones R32 y R64 coinciden con los anteriores: mismos 2.468 objetos, etiquetas y predicciones de los tres métodos, sin diferencias.

La auditoría de las salidas nuevas aprobó las seis matrices de confusión, nueve pruebas de McNemar, resúmenes de tiempos y hashes, usando los datos y modelos ya verificados. La comparación `diferencias.json` también se regeneró y coincidió con la publicada. Esto es una revisión de la evidencia entregada; no se afirma haber repetido localmente la inferencia Net5 en el equipo del revisor.

Net5 recorrió los octrees originales de todo el test en ambas resoluciones. SVM y Bosque Aleatorio reutilizaron las características publicadas para el test completo; en la etapa de tiempos se regeneraron descriptores desde 400 octrees por resolución y se contrastaron sus predicciones. Por tanto, la evidencia no equivale a regenerar todas las características clásicas desde las mallas u octrees.

Se añadió una regla de finales de línea LF para la evidencia textual. Una prueba de checkout con `core.autocrlf=true` conservó los hashes y aprobó la auditoría; no se modificaron los hashes ni se relajó su verificación. Las copias antiguas con CRLF requieren actualizarse según el README.

## Resultados principales

| Método | Exactitud R32 | Exactitud R64 | F1 macro R32 | F1 macro R64 |
|---|---:|---:|---:|---:|
| SVM | 75,93 % | 77,19 % | 65,82 % | 68,47 % |
| Bosque Aleatorio | 76,38 % | 77,19 % | 67,72 % | 68,53 % |
| Net5 | 82,21 % | 83,23 % | 74,93 % | 77,23 % |

Fuente: [tabla de métricas](tablas/t1_aciertos.md). Estas métricas se conservaron en la reevaluación.

## Texto propuesto para las conclusiones del informe

En las configuraciones evaluadas sobre ModelNet40, Net5 alcanzó mayor exactitud y F1 macro que SVM y Bosque Aleatorio en ambas resoluciones. Frente al mejor clasificador clásico de cada resolución, la ventaja de exactitud fue de 5,83 puntos porcentuales en R32 y 6,04 en R64. Las pruebas pareadas de McNemar respaldaron las diferencias entre Net5 y los métodos clásicos. La reevaluación conservó todas las predicciones de los seis casos, lo que aporta evidencia de repetibilidad de la clasificación bajo el protocolo declarado.

La mayor exactitud de Net5 estuvo acompañada de un mayor costo computacional. En la reevaluación, la mediana de inferencia fue de 13,6 y 51,5 ms para SVM, 18,7 y 56,8 ms para Bosque Aleatorio, y 48,3 y 217,5 ms para Net5 en GPU, en R32 y R64 respectivamente. Estos valores corresponden al equipo y al alcance del cronometraje documentados, no al procesamiento completo desde las mallas. SVM presentó el menor tamaño de archivo y constituye una alternativa favorable cuando se priorizan rapidez y almacenamiento.

El aumento de resolución produjo ganancias de exactitud entre 0,81 y 1,26 puntos porcentuales. En Net5, R64 añadió 25 aciertos sobre 2.468 objetos, pero la prueba utilizada no detectó una diferencia significativa respecto a R32. Este resultado no demuestra equivalencia. Net5 R32 ofrece un compromiso favorable entre desempeño y costo, mientras Net5 R64 obtiene la mayor exactitud observada. Decidir si la resolución adicional compensa requiere valorar los aciertos y las clases de interés frente al costo de ejecución.

Los resultados permiten describir la escalabilidad entre R32 y R64 para estas implementaciones, sin extrapolar a resoluciones mayores. La comparación aporta evidencia favorable al aprendizaje de características geométricas en el experimento realizado; no establece una superioridad universal de Net5 ni cuantifica la variabilidad entre entrenamientos.

## Límites que deben conservarse en el informe final

- Se utilizó una sola semilla de entrenamiento. Los intervalos del test no representan variabilidad entre entrenamientos. Los valores p corresponden a pruebas individuales sin ajuste por comparaciones múltiples.
- Los tiempos excluyen la generación de octrees desde las mallas. En Net5 GPU también se omite el armado del lote previo al forward; en CPU se incluye. No presentar esas latencias como tiempos simétricos de extremo a extremo.
- La muestra de memoria contiene 40 objetos de airplane, 40 de bathtub y 20 de bed. Sus picos describen ese subconjunto, no todas las clases. La RAM adicional de Net5 GPU aumentó aproximadamente 13 % entre resoluciones en la corrida original y 2 % en la nueva: no hay un porcentaje estable establecido.
- La extracción HCE de entrenamiento tiene un costo estimado. La suma de tiempos de épocas de Net5 no incluye necesariamente todo el tiempo transcurrido de la sesión.
- Las métricas clásicas del test completo parten de características guardadas. La regeneración desde octrees se verificó en la muestra de tiempos, no en todo el test.
- Se midió en un solo equipo. Tanto los valores absolutos como las proporciones entre métodos pueden variar con hardware, hilos e implementación.

## Cierre documental

No se requieren nuevos entrenamientos ni repetir la evaluación para sostener las conclusiones limitadas a este protocolo. La integración de los cambios debe conservar ambas ejecuciones. El paso documental restante es incorporar estas comparaciones, conclusiones y limitaciones en la versión vigente del documento de grado y someterla a la revisión del director. Este archivo no modifica por sí mismo el proyecto de Overleaf ni declara aprobado el objetivo general.
