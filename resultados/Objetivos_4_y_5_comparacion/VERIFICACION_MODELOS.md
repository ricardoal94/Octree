# Verificación de modelos descargados — 27 de septiembre de 2026

Se descargaron los seis modelos del enlace de Drive publicado por Ricardo, con autorización del director. Los seis SHA-256 coinciden con `SHA256SUMS.txt`.

| Modelo | Resolución | Exactitud reproducida | Diferencias frente al CSV publicado |
|---|---|---:|---:|
| SVM | R32 | 75,9319 % | 0 de 2468 |
| Bosque Aleatorio | R32 | 76,3776 % | 0 de 2468 |
| SVM | R64 | 77,1880 % | 0 de 2468 |
| Bosque Aleatorio | R64 | 77,1880 % | 0 de 2468 |

La inferencia HCE utilizó `X_test` de los NPZ de características publicados. Se comprobaron las etiquetas contra los CSV; estos NPZ no almacenan IDs por fila, de modo que esta comprobación no sustituye la regeneración de características desde los octrees.

Los checkpoints Net5 R32/R64 se cargaron estrictamente en el backend integrado y contienen pesos finitos. Son idénticos a los checkpoints previamente compartidos. No se repitió la inferencia Net5 por no disponer aquí de los octrees originales.

La verificación local utilizó scikit-learn 1.8.0 y los modelos indican 1.9.0: se registraron las advertencias de versión, aunque las cuatro salidas coinciden exactamente. La reproducción definitiva debe usar el entorno de los autores. Versiones, hashes y advertencias completas: `VERIFICACION_MODELOS.json`.

No se midieron nuevamente tiempos ni memoria, ni se ejecutaron entrenamientos. Pendiente: evaluación desde los octrees originales y registro de ejecución en Windows/CUDA, según el README.
