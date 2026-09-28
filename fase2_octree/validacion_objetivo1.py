"""Validaciones y utilidades comunes del objetivo especifico 1.

Este modulo no depende de resultados preexistentes. Las comparaciones
parten siempre de una misma nube de puntos y construyen de manera
independiente la rejilla densa y el octree adaptativo.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

try:  # Ejecucion como paquete
    from .cuantizacion import cuantizar_indices_octree
    from .octree import construir_grid_octree
    from .octree_real import (
        OCTREE_FORMAT_VERSION,
        contar_nodos_por_profundidad,
        contar_nodos_totales,
        octree_a_grid_denso,
        recolectar_hojas,
    )
except ImportError:  # Ejecucion directa desde fase2_octree/
    from cuantizacion import cuantizar_indices_octree
    from octree import construir_grid_octree
    from octree_real import (
        OCTREE_FORMAT_VERSION,
        contar_nodos_por_profundidad,
        contar_nodos_totales,
        octree_a_grid_denso,
        recolectar_hojas,
    )


MANIFEST_FORMAT_VERSION = "1.0.0"
SEED_ALGORITHM = "sha256_ruta_relativa_u32_mas_semilla_base"


def ruta_portable(ruta: str | Path, raiz_proyecto: str | Path) -> str:
    """Registra rutas sin depender de la ubicacion local del repositorio.

    Las rutas contenidas en el proyecto se guardan relativas a su raiz. Si el
    dataset o la salida estan fuera del clon, solo se conserva el nombre del
    archivo o directorio para evitar rutas absolutas especificas del equipo.
    """
    ruta_resuelta = Path(ruta).resolve()
    raiz_resuelta = Path(raiz_proyecto).resolve()
    try:
        return ruta_resuelta.relative_to(raiz_resuelta).as_posix()
    except ValueError:
        return ruta_resuelta.name


def sha256_archivo(ruta: str | Path, bloque: int = 1024 * 1024) -> str:
    """Calcula SHA-256 sin cargar el archivo completo en memoria."""
    digest = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for fragmento in iter(lambda: archivo.read(bloque), b""):
            digest.update(fragmento)
    return digest.hexdigest()


def semilla_estable_modelo(semilla_base: int, ruta_relativa: str) -> int:
    """Deriva una semilla uint32 estable e independiente del orden de ejecucion."""
    clave = ruta_relativa.replace("\\", "/").encode("utf-8")
    desplazamiento = int.from_bytes(hashlib.sha256(clave).digest()[:4], "little")
    return int((int(semilla_base) + desplazamiento) % (2**32))


def celdas_ocupadas_desde_puntos(puntos: np.ndarray, resolucion: int) -> np.ndarray:
    """Cuantiza directamente los puntos y retorna indices unicos ordenados."""
    indices = cuantizar_indices_octree(puntos, resolucion)
    return np.unique(indices, axis=0)


def celdas_ocupadas_desde_octree(raiz, resolucion: int) -> np.ndarray:
    """Retorna las celdas hoja ocupadas del arbol como indices ordenados."""
    hojas = recolectar_hojas(raiz)
    if not hojas:
        return np.empty((0, 3), dtype=np.int64)
    centros = np.asarray([hoja.centro for hoja in hojas], dtype=np.float32)
    indices = cuantizar_indices_octree(centros, resolucion)
    return np.unique(indices, axis=0)


def validar_equivalencia_denso_octree(
    puntos: np.ndarray,
    normales: np.ndarray,
    raiz,
    resolucion: int,
    atol_normales: float = 1e-6,
    grid_directo: np.ndarray | None = None,
    grid_arbol: np.ndarray | None = None,
) -> dict:
    """Compara, celda por celda, dos construcciones independientes.

    La rejilla se cuantiza directamente desde ``puntos``. El grid del
    octree se materializa desde las hojas del arbol. La equivalencia de
    ocupacion es exacta; las normales se comparan con tolerancia numerica.
    """
    celdas_directas = celdas_ocupadas_desde_puntos(puntos, resolucion)
    celdas_arbol = celdas_ocupadas_desde_octree(raiz, resolucion)

    celdas_iguales = np.array_equal(celdas_directas, celdas_arbol)
    if grid_directo is None:
        grid_directo = construir_grid_octree(puntos, normales, resolucion)
    if grid_arbol is None:
        grid_arbol = octree_a_grid_denso(raiz, resolucion)
    ocupacion_igual = np.array_equal(grid_directo[0], grid_arbol[0])

    mascara = grid_directo[0].astype(bool)
    if np.any(mascara):
        diferencias = np.abs(grid_directo[1:, mascara] - grid_arbol[1:, mascara])
        max_error_normal = float(diferencias.max(initial=0.0))
    else:
        max_error_normal = 0.0
    normales_iguales = bool(
        np.allclose(
            grid_directo[1:, mascara], grid_arbol[1:, mascara],
            rtol=1e-6, atol=atol_normales,
        )
    )

    resultado = {
        "resolucion": int(resolucion),
        "n_celdas_denso": int(len(celdas_directas)),
        "n_hojas_octree": int(len(celdas_arbol)),
        "celdas_iguales": bool(celdas_iguales),
        "ocupacion_grid_igual": bool(ocupacion_igual),
        "normales_iguales": normales_iguales,
        "max_error_absoluto_normales": max_error_normal,
        "equivalente": bool(celdas_iguales and ocupacion_igual and normales_iguales),
    }
    if not resultado["equivalente"]:
        raise AssertionError(
            "La rejilla densa y el octree no son equivalentes en "
            f"resolucion {resolucion}: {resultado}"
        )
    return resultado


def metricas_estructura(raiz, resolucion: int, profundidad_max: int) -> dict:
    """Calcula las metricas estructurales exigidas, sin medir tiempos."""
    nodos_por_nivel = contar_nodos_por_profundidad(raiz, profundidad_max)
    n_nodos = contar_nodos_totales(raiz)
    n_hojas = len(recolectar_hojas(raiz))
    return {
        "nodos_por_nivel": [int(valor) for valor in nodos_por_nivel],
        "n_nodos_totales": int(n_nodos),
        "n_hojas_ocupadas": int(n_hojas),
        "porcentaje_ocupacion_hoja": float(100.0 * n_hojas / (resolucion**3)),
    }


def guardar_json_atomico(datos: dict, ruta: str | Path) -> None:
    """Escribe JSON de forma atomica para no dejar manifiestos parciales."""
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporal = tempfile.mkstemp(
        prefix=f".{destino.name}.", suffix=".tmp", dir=destino.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as archivo:
            json.dump(datos, archivo, indent=2, ensure_ascii=False, sort_keys=True)
            archivo.write("\n")
        os.replace(temporal, destino)
    except Exception:
        try:
            os.unlink(temporal)
        except FileNotFoundError:
            pass
        raise


def manifiesto_base(
    *,
    model_id: str,
    categoria: str,
    split: str,
    ruta_origen_relativa: str,
    sha256_origen: str,
    semilla_base: int,
    semilla_muestreo: int,
    n_puntos_muestreo: int,
) -> dict:
    """Crea la parte comun y versionada de un manifiesto por modelo."""
    return {
        "manifest_format_version": MANIFEST_FORMAT_VERSION,
        "octree_format_version": OCTREE_FORMAT_VERSION,
        "model_id": model_id,
        "categoria": categoria,
        "split": split,
        "archivo_origen": ruta_origen_relativa.replace("\\", "/"),
        "archivo_origen_sha256": sha256_origen,
        "muestreo": {
            "n_puntos": int(n_puntos_muestreo),
            "semilla_base": int(semilla_base),
            "semilla_modelo": int(semilla_muestreo),
            "algoritmo_semilla": SEED_ALGORITHM,
        },
        "salidas": {},
    }
