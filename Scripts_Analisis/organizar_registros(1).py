"""
organizar_registros.py
========================
Copia (no mueve) los archivos de metricas y resultados ya generados por
las distintas fases hacia una carpeta Registros/ con estructura clara,
lista para versionar en git. No modifica los archivos originales en
logs/ ni resultados/, por lo que es seguro correrlo en cualquier momento.

Uso:
    python organizar_registros.py
"""

import shutil
from pathlib import Path

RAIZ = Path(r"C:\Users\ricar\Documents\Codigos\Tesis")
DIR_LOGS = RAIZ / "logs"
DIR_RESULTADOS = RAIZ / "resultados"
DIR_REGISTROS = RAIZ / "Registros"

# Mapa: (carpeta_origen, patron_glob) -> subcarpeta_destino
MAPA = [
    (DIR_LOGS,       "experiment_log.json",          "fase1"),
    (DIR_LOGS,       "particion_indices.npz",         "fase1"),

    (DIR_RESULTADOS, "metricas_off_*.csv",            "fase2_octree"),
    (DIR_RESULTADOS, "metricas_off_*.json",           "fase2_octree"),
    (DIR_RESULTADOS, "resumen_octree_*.json",         "fase2_octree"),
    (DIR_RESULTADOS, "tabla_resumen_octree*.png",     "fase2_octree"),
    (DIR_RESULTADOS, "tabla_resumen_octree*.svg",     "fase2_octree"),

    (DIR_RESULTADOS, "resumen_hce_R*.json",           "fase3_hce"),
    (DIR_LOGS,       "hce_features_R*.npz",           "fase3_hce"),
    (DIR_RESULTADOS, "*hce*.png",                     "fase3_hce"),
    (DIR_RESULTADOS, "*hce*.svg",                     "fase3_hce"),

    (DIR_RESULTADOS, "resumen_net5_R*.json",          "fase3_net5"),
    (DIR_LOGS,       "net5_historial_R*.csv",         "fase3_net5"),
    (DIR_LOGS,       "net5_historial_R*.json",        "fase3_net5"),
    (DIR_RESULTADOS, "*net5*.png",                    "fase3_net5"),
    (DIR_RESULTADOS, "*net5*.svg",                    "fase3_net5"),

    (DIR_RESULTADOS, "resumen_img2voxel_R*.json",     "fase5_img2voxel"),
    (DIR_LOGS,       "img2voxel_historial_R*.csv",    "fase5_img2voxel"),
    (DIR_RESULTADOS, "*img2voxel*.png",                "fase5_img2voxel"),

    (DIR_RESULTADOS, "comparativa_metricas_completa.*", "comparativa_final"),
    (DIR_RESULTADOS, "tabla_metricas_completa.*",       "comparativa_final"),
    (DIR_RESULTADOS, "resumen_metricas_completo.csv",   "comparativa_final"),
]


def main():
    print("=" * 60)
    print("  ORGANIZANDO REGISTROS")
    print("=" * 60)

    total_copiados = 0
    total_no_encontrados = 0

    for carpeta_origen, patron, subcarpeta in MAPA:
        destino = DIR_REGISTROS / subcarpeta
        destino.mkdir(parents=True, exist_ok=True)

        if not carpeta_origen.exists():
            continue

        archivos = list(carpeta_origen.glob(patron))
        if not archivos:
            total_no_encontrados += 1
            continue

        for archivo in archivos:
            destino_archivo = destino / archivo.name
            shutil.copy2(archivo, destino_archivo)
            print(f"  [OK] {archivo.name}  ->  Registros/{subcarpeta}/")
            total_copiados += 1

    print("\n" + "=" * 60)
    print(f"  Archivos copiados     : {total_copiados}")
    print(f"  Patrones sin coincidencia: {total_no_encontrados} (normal si aun no se corrio esa fase)")
    print(f"  Carpeta destino       : {DIR_REGISTROS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
