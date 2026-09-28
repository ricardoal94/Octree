# Comparación reproducible HCE–Net5

El [informe único y las instrucciones de reproducción](../resultados/Objetivos_4_y_5_comparacion/README.md) contienen resultados, límites y rutas.

Desde la raíz del repositorio:

```powershell
python fase4_comparacion/auditar_comparacion.py
python fase4_comparacion/tablas_comparacion.py
python fase4_comparacion/figuras_comparacion.py
```

Para evaluar los modelos reales, seguir los requisitos del informe. Los programas antiguos `comparar_predicciones.py` y `gradcam_3d.py` no son la ruta de reproducción del backend nativo.
