"""Captura reproducible del entorno usado por los experimentos Net5."""

from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path

import torch


def _texto_o_desconocido(valor: object) -> str:
    texto = str(valor).strip() if valor is not None else ""
    return texto or "desconocido"


def _modelo_cpu() -> str:
    candidatos = (
        platform.processor(),
        os.environ.get("PROCESSOR_IDENTIFIER"),
        platform.uname().processor,
    )
    for candidato in candidatos:
        if candidato and candidato.strip():
            return candidato.strip()

    cpuinfo = Path("/proc/cpuinfo")
    try:
        for linea in cpuinfo.read_text(encoding="utf-8").splitlines():
            if linea.lower().startswith("model name"):
                return linea.split(":", maxsplit=1)[1].strip()
    except (OSError, IndexError):
        pass
    return "desconocido"


def _ram_total_bytes() -> int | None:
    """Devuelve la memoria fisica total sin depender de paquetes externos."""

    if os.name == "nt":
        class EstadoMemoria(ctypes.Structure):
            _fields_ = [
                ("longitud", ctypes.c_ulong),
                ("carga_memoria", ctypes.c_ulong),
                ("total_fisica", ctypes.c_ulonglong),
                ("fisica_disponible", ctypes.c_ulonglong),
                ("total_paginacion", ctypes.c_ulonglong),
                ("paginacion_disponible", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("virtual_disponible", ctypes.c_ulonglong),
                ("virtual_extendida_disponible", ctypes.c_ulonglong),
            ]

        estado = EstadoMemoria()
        estado.longitud = ctypes.sizeof(EstadoMemoria)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(  # type: ignore[attr-defined]
                ctypes.byref(estado)
            ):
                return int(estado.total_fisica)
        except (AttributeError, OSError):
            return None

    try:
        paginas = int(os.sysconf("SC_PHYS_PAGES"))
        tamano_pagina = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    total = paginas * tamano_pagina
    return total if total > 0 else None


def capturar_entorno_ejecucion(device: torch.device) -> dict:
    """Obtiene software y hardware relevantes en un objeto serializable."""

    ram_bytes = _ram_total_bytes()
    gpu = None
    if device.type == "cuda":
        indice = device.index
        if indice is None:
            indice = torch.cuda.current_device()
        propiedades = torch.cuda.get_device_properties(indice)
        capacidad = torch.cuda.get_device_capability(indice)
        gpu = {
            "nombre": torch.cuda.get_device_name(indice),
            "indice": int(indice),
            "cantidad_dispositivos": int(torch.cuda.device_count()),
            "capacidad_cuda": f"{capacidad[0]}.{capacidad[1]}",
            "vram_total_bytes": int(propiedades.total_memory),
            "vram_total_gib": round(
                propiedades.total_memory / (1024 ** 3), 2
            ),
        }

    version_cudnn = torch.backends.cudnn.version()
    return {
        "sistema_operativo": {
            "sistema": _texto_o_desconocido(platform.system()),
            "release": _texto_o_desconocido(platform.release()),
            "version": _texto_o_desconocido(platform.version()),
            "arquitectura": _texto_o_desconocido(platform.machine()),
        },
        "python": {
            "version": platform.python_version(),
            "implementacion": platform.python_implementation(),
        },
        "pytorch": {
            "version": str(torch.__version__),
            "cuda_compilacion": torch.version.cuda,
            "cudnn_version": (
                int(version_cudnn) if version_cudnn is not None else None
            ),
        },
        "hardware": {
            "cpu_modelo": _modelo_cpu(),
            "cpu_nucleos_logicos": os.cpu_count(),
            "ram_total_bytes": ram_bytes,
            "ram_total_gib": (
                round(ram_bytes / (1024 ** 3), 2)
                if ram_bytes is not None
                else None
            ),
            "dispositivo": device.type,
            "gpu": gpu,
        },
    }
