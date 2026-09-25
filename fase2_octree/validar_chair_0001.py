"""Prueba controlada e independiente con chair_0001 en 32^3 y 64^3."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from octree import (  # noqa: E402
    leer_off,
    muestrear_superficie_con_normales,
    normalizar_malla,
    profundidad_de,
)
from octree_real import (  # noqa: E402
    OCTREE_FORMAT_VERSION,
    construir_octree,
    guardar_octree_disperso,
    reconstruir_octree_desde_npz,
)
from validacion_objetivo1 import (  # noqa: E402
    guardar_json_atomico,
    metricas_estructura,
    ruta_portable,
    sha256_archivo,
    validar_equivalencia_denso_octree,
)


def validar(ruta_off: Path, n_puntos: int, semilla: int) -> dict:
    if ruta_off.stem != "chair_0001":
        raise ValueError("La prueba controlada requiere el modelo chair_0001.off")

    vertices, caras = leer_off(str(ruta_off))
    vertices = normalizar_malla(vertices)
    rng = np.random.default_rng(semilla)
    puntos, normales = muestrear_superficie_con_normales(
        vertices, caras, n_puntos, rng,
    )
    sha_origen = sha256_archivo(ruta_off)

    resultado = {
        "prueba": "chair_0001_equivalencia_y_persistencia",
        "octree_format_version": OCTREE_FORMAT_VERSION,
        "archivo": ruta_portable(ruta_off, RAIZ_PROYECTO),
        "archivo_sha256": sha_origen,
        "semilla": semilla,
        "n_puntos": n_puntos,
        "resoluciones": {},
    }

    with tempfile.TemporaryDirectory(prefix="chair_0001_octree_") as temporal:
        for resolucion in (32, 64):
            profundidad = profundidad_de(resolucion)
            raiz = construir_octree(puntos, normales, profundidad)
            equivalencia = validar_equivalencia_denso_octree(
                puntos, normales, raiz, resolucion,
            )

            ruta_npz = Path(temporal) / f"chair_0001_R{resolucion}.npz"
            guardar_octree_disperso(
                raiz, ruta_npz, etiqueta=8, profundidad_max=profundidad,
                metadatos={
                    "model_id": "chair_0001",
                    "categoria": "chair",
                    "split": "train",
                    "semilla_muestreo": semilla,
                    "n_puntos_muestreo": n_puntos,
                    "archivo_origen_sha256": sha_origen,
                },
            )
            raiz_cargada, etiqueta, profundidad_cargada = (
                reconstruir_octree_desde_npz(ruta_npz)
            )
            equivalencia_cargada = validar_equivalencia_denso_octree(
                puntos, normales, raiz_cargada, resolucion,
            )
            if etiqueta != 8 or profundidad_cargada != profundidad:
                raise AssertionError("Los metadatos cambiaron durante el round-trip")

            resultado["resoluciones"][str(resolucion)] = {
                **metricas_estructura(raiz, resolucion, profundidad),
                "equivalencia_directa": equivalencia,
                "equivalencia_despues_carga": equivalencia_cargada,
            }

    resultado["prueba_superada"] = True
    return resultado


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--off", type=Path,
        default=RAIZ_PROYECTO / "Dataset" / "ModelNet40" / "chair" / "train"
        / "chair_0001.off",
    )
    parser.add_argument("--n-puntos", type=int, default=20000)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument(
        "--salida", type=Path,
        default=RAIZ_PROYECTO / "resultados" / "objetivo1"
        / "validacion_chair_0001.json",
    )
    args = parser.parse_args()
    if not args.off.is_file():
        raise FileNotFoundError(
            f"No se encontro {args.off}. Use --off para indicar chair_0001.off."
        )

    resultado = validar(args.off, args.n_puntos, args.semilla)
    guardar_json_atomico(resultado, args.salida)
    print("Prueba chair_0001 superada para 32^3 y 64^3")
    print(f"Resultados: {args.salida.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
