"""Backend PyTorch disperso para operaciones sobre grid-octrees.

Las convoluciones usan los planes geometricos de :mod:`grid_octree` y son
completamente diferenciables. No se invoca ``Conv3d`` ni se materializa el
volumen de entrada. Solo se expande la salida final de resolucion 8^3, que es
la entrada explicita de la capa totalmente conectada de la Tabla 5.

Esta implementacion prioriza trazabilidad y equivalencia matematica. Los
planes se construyen en CPU y se almacenan por lote; las multiplicaciones y
la reduccion se ejecutan en el dispositivo de los atributos (CPU o CUDA).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from grid_octree import GeometriaGridOctree, MuestraGridOctree


@dataclass
class LoteGridOctree:
    """Lote de hojas concatenadas con geometria independiente por muestra."""

    atributos: torch.Tensor
    geometrias: tuple[GeometriaGridOctree, ...]
    offsets: np.ndarray
    _cache: dict = field(default_factory=dict, repr=False)
    _perfil: dict | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.atributos.ndim != 2:
            raise ValueError("Los atributos del lote deben tener forma (N, C)")
        self.offsets = np.asarray(self.offsets, dtype=np.int64)
        if self.offsets.shape != (len(self.geometrias) + 1,):
            raise ValueError("Los offsets no corresponden al numero de muestras")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != len(self.atributos):
            raise ValueError("Los offsets no cubren todas las hojas")
        esperados = np.asarray(
            [geometria.n_hojas for geometria in self.geometrias], dtype=np.int64,
        )
        if not np.array_equal(np.diff(self.offsets), esperados):
            raise ValueError("Los offsets no coinciden con las geometrias")
        resoluciones = {geometria.resolucion for geometria in self.geometrias}
        if len(resoluciones) != 1:
            raise ValueError("Todas las muestras de un lote deben tener igual resolucion")

    @classmethod
    def desde_muestras(cls, muestras: Iterable[MuestraGridOctree]) -> "LoteGridOctree":
        muestras = tuple(muestras)
        if not muestras:
            raise ValueError("No se puede construir un lote vacio")
        geometrias = tuple(muestra.geometria for muestra in muestras)
        cantidades = [geometria.n_hojas for geometria in geometrias]
        offsets = np.concatenate([
            np.asarray([0], dtype=np.int64),
            np.cumsum(cantidades, dtype=np.int64),
        ])
        atributos_np = np.concatenate(
            [muestra.atributos for muestra in muestras], axis=0,
        )
        return cls(
            atributos=torch.from_numpy(atributos_np.copy()),
            geometrias=geometrias,
            offsets=offsets,
        )

    @property
    def resolucion(self) -> int:
        return self.geometrias[0].resolucion

    @property
    def batch_size(self) -> int:
        return len(self.geometrias)

    def size(self, dim: int | None = None):
        forma = (self.batch_size, self.atributos.shape[1])
        return forma if dim is None else forma[dim]

    def to(self, *args, **kwargs) -> "LoteGridOctree":
        return LoteGridOctree(
            atributos=self.atributos.to(*args, **kwargs),
            geometrias=self.geometrias,
            offsets=self.offsets,
            _cache=self._cache,
            _perfil=self._perfil,
        )

    def pin_memory(self) -> "LoteGridOctree":
        return LoteGridOctree(
            atributos=self.atributos.pin_memory(),
            geometrias=self.geometrias,
            offsets=self.offsets,
            _cache=self._cache,
            _perfil=self._perfil,
        )

    def con_atributos(self, atributos: torch.Tensor) -> "LoteGridOctree":
        return LoteGridOctree(
            atributos=atributos,
            geometrias=self.geometrias,
            offsets=self.offsets,
            _cache=self._cache,
            _perfil=self._perfil,
        )

    def activar_perfil(self) -> "LoteGridOctree":
        """Activa contadores compartidos por todas las capas del lote."""

        self._perfil = {
            "plan_convolucion_cpu_s": 0.0,
            "plan_geometria_cpu_s": 0.0,
            "ensamble_plan_lote_cpu_s": 0.0,
            "transferencia_plan_convolucion_s": 0.0,
            "plan_pooling_cpu_s": 0.0,
            "transferencia_plan_pooling_s": 0.0,
            "mapa_final_cpu_s": 0.0,
            "transferencia_mapa_final_s": 0.0,
            "n_planes_convolucion": 0,
            "n_interacciones_convolucion": 0,
            "bytes_planes_convolucion": 0,
            "n_planes_pooling": 0,
            "n_asignaciones_pooling": 0,
            "bytes_planes_pooling": 0,
            "n_mapas_finales": 0,
            "bytes_mapas_finales": 0,
        }
        return self

    def resumen_perfil(self) -> dict:
        """Devuelve una copia de los contadores internos del backend."""

        return dict(self._perfil or {})

    def _registrar_perfil(self, clave: str, valor: float) -> None:
        if self._perfil is not None:
            self._perfil[clave] += valor

    def _plan_convolucion_numpy(self) -> tuple[np.ndarray, ...]:
        clave = "conv_numpy"
        if clave not in self._cache:
            t0_total = time.perf_counter() if self._perfil is not None else None
            t0_geometria = (
                time.perf_counter() if self._perfil is not None else None
            )
            planes = tuple(
                geometria.plan_convolucion for geometria in self.geometrias
            )
            if t0_geometria is not None:
                self._registrar_perfil(
                    "plan_geometria_cpu_s", time.perf_counter() - t0_geometria,
                )

            t0_ensamble = (
                time.perf_counter() if self._perfil is not None else None
            )
            if self.batch_size == 1:
                plan = planes[0]
                self._cache[clave] = (
                    plan.salida,
                    plan.entrada,
                    plan.coeficiente,
                    plan.offsets_kernel,
                )
            else:
                salidas_por_kernel = [[] for _ in range(27)]
                entradas_por_kernel = [[] for _ in range(27)]
                coeficientes_por_kernel = [[] for _ in range(27)]
                for offset, plan in zip(self.offsets[:-1], planes):
                    for kernel in range(27):
                        inicio = int(plan.offsets_kernel[kernel])
                        fin = int(plan.offsets_kernel[kernel + 1])
                        salidas_por_kernel[kernel].append(
                            plan.salida[inicio:fin] + int(offset)
                        )
                        entradas_por_kernel[kernel].append(
                            plan.entrada[inicio:fin] + int(offset)
                        )
                        coeficientes_por_kernel[kernel].append(
                            plan.coeficiente[inicio:fin]
                        )
                salidas = []
                entradas = []
                coeficientes = []
                offsets_kernel = [0]
                for kernel in range(27):
                    salida_kernel = np.concatenate(salidas_por_kernel[kernel])
                    entrada_kernel = np.concatenate(entradas_por_kernel[kernel])
                    coeficiente_kernel = np.concatenate(
                        coeficientes_por_kernel[kernel]
                    )
                    salidas.append(salida_kernel)
                    entradas.append(entrada_kernel)
                    coeficientes.append(coeficiente_kernel)
                    offsets_kernel.append(
                        offsets_kernel[-1] + len(salida_kernel)
                    )
                self._cache[clave] = (
                    np.concatenate(salidas),
                    np.concatenate(entradas),
                    np.concatenate(coeficientes),
                    np.asarray(offsets_kernel, dtype=np.int64),
                )
            if t0_ensamble is not None:
                self._registrar_perfil(
                    "ensamble_plan_lote_cpu_s",
                    time.perf_counter() - t0_ensamble,
                )
            if t0_total is not None:
                self._registrar_perfil(
                    "plan_convolucion_cpu_s", time.perf_counter() - t0_total,
                )
                self._perfil["n_planes_convolucion"] += 1
                salida, entrada, coeficiente, offsets = self._cache[clave]
                self._perfil["n_interacciones_convolucion"] += len(salida)
                bytes_geometrias = sum(plan.nbytes for plan in planes)
                bytes_ensamble = 0 if self.batch_size == 1 else sum(
                    arreglo.nbytes
                    for arreglo in (salida, entrada, coeficiente, offsets)
                )
                self._perfil["bytes_planes_convolucion"] += (
                    bytes_geometrias + bytes_ensamble
                )
        return self._cache[clave]

    def plan_convolucion_torch(self) -> tuple:
        dispositivo = self.atributos.device
        clave = ("conv_torch", dispositivo.type, dispositivo.index)
        if clave not in self._cache:
            salida, entrada, coeficiente, offsets = (
                self._plan_convolucion_numpy()
            )
            t0 = time.perf_counter() if self._perfil is not None else None
            self._cache[clave] = (
                torch.as_tensor(salida, dtype=torch.long, device=dispositivo),
                torch.as_tensor(entrada, dtype=torch.long, device=dispositivo),
                torch.as_tensor(
                    coeficiente,
                    dtype=self.atributos.dtype,
                    device=dispositivo,
                ),
                offsets,
            )
            if t0 is not None and dispositivo.type == "cuda":
                torch.cuda.synchronize(dispositivo)
            if t0 is not None:
                self._registrar_perfil(
                    "transferencia_plan_convolucion_s",
                    time.perf_counter() - t0,
                )
        return self._cache[clave]

    def plan_convolucion_fusionado_torch(self) -> tuple:
        """Plan para aplicar los 27 kernels con una sola proyeccion.

        Devuelve ``(salida, fila_proyeccion, coeficiente)``. La convolucion
        proyecta cada hoja con los 27 kernels a la vez, ``Z = X @ W``, y la
        arista ``e`` toma la fila ``entrada[e] * 27 + kernel[e]`` de ``Z``.
        """

        dispositivo = self.atributos.device
        clave = ("conv_fusion", dispositivo.type, dispositivo.index)
        if clave not in self._cache:
            salida, entrada, coeficiente, offsets = self.plan_convolucion_torch()
            conteos = torch.as_tensor(
                np.diff(np.asarray(offsets, dtype=np.int64)),
                device=dispositivo,
            )
            kernel = torch.repeat_interleave(
                torch.arange(27, device=dispositivo), conteos,
            )
            self._cache[clave] = (salida, entrada * 27 + kernel, coeficiente)
        return self._cache[clave]

    def max_pool2(self) -> "LoteGridOctree":
        t0 = time.perf_counter() if self._perfil is not None else None
        planes = tuple(geometria.plan_pooling for geometria in self.geometrias)
        geometrias_salida = tuple(
            plan.geometria_salida for plan in planes
        )
        cantidades_salida = [g.n_hojas for g in geometrias_salida]
        offsets_salida = np.concatenate([
            np.asarray([0], dtype=np.int64),
            np.cumsum(cantidades_salida, dtype=np.int64),
        ])
        if self.batch_size == 1:
            mapa_np = planes[0].entrada_a_salida
        else:
            mapeos = []
            for offset_salida, plan in zip(offsets_salida[:-1], planes):
                mapeos.append(
                    plan.entrada_a_salida + int(offset_salida)
                )
            mapa_np = np.concatenate(mapeos)
        if t0 is not None:
            self._registrar_perfil(
                "plan_pooling_cpu_s", time.perf_counter() - t0,
            )
            self._perfil["n_planes_pooling"] += 1
            self._perfil["n_asignaciones_pooling"] += len(mapa_np)
            bytes_geometrias = sum(plan.nbytes for plan in planes)
            bytes_ensamble = 0 if self.batch_size == 1 else mapa_np.nbytes
            self._perfil["bytes_planes_pooling"] += (
                bytes_geometrias + bytes_ensamble
            )
        t0 = time.perf_counter() if self._perfil is not None else None
        mapa = torch.as_tensor(
            mapa_np, dtype=torch.long, device=self.atributos.device,
        )
        if t0 is not None and self.atributos.device.type == "cuda":
            torch.cuda.synchronize(self.atributos.device)
        if t0 is not None:
            self._registrar_perfil(
                "transferencia_plan_pooling_s", time.perf_counter() - t0,
            )
        indice = mapa[:, None].expand(-1, self.atributos.shape[1])
        salida = torch.full(
            (int(offsets_salida[-1]), self.atributos.shape[1]),
            -torch.inf,
            dtype=self.atributos.dtype,
            device=self.atributos.device,
        )
        salida.scatter_reduce_(
            0, indice, self.atributos, reduce="amax", include_self=True,
        )
        return LoteGridOctree(
            atributos=salida,
            geometrias=geometrias_salida,
            offsets=offsets_salida,
            _perfil=self._perfil,
        )

    def tensor_final_8(self) -> torch.Tensor:
        """Expande unicamente la salida final 8^3 para la capa FC."""

        if self.resolucion != 8:
            raise ValueError("La capa FC requiere una salida de resolucion 8^3")
        t0 = time.perf_counter() if self._perfil is not None else None
        if self.batch_size == 1:
            mapa_np = self.geometrias[0].indices_voxel_a_hoja()
        else:
            mapas = []
            for offset, geometria in zip(self.offsets[:-1], self.geometrias):
                mapas.append(geometria.indices_voxel_a_hoja() + int(offset))
            mapa_np = np.concatenate(mapas)
        if t0 is not None:
            self._registrar_perfil(
                "mapa_final_cpu_s", time.perf_counter() - t0,
            )
            self._perfil["n_mapas_finales"] += 1
            bytes_geometrias = sum(
                geometria.indices_voxel_a_hoja().nbytes
                for geometria in self.geometrias
            )
            bytes_ensamble = 0 if self.batch_size == 1 else mapa_np.nbytes
            self._perfil["bytes_mapas_finales"] += (
                bytes_geometrias + bytes_ensamble
            )
        t0 = time.perf_counter() if self._perfil is not None else None
        indices = torch.tensor(
            mapa_np,
            dtype=torch.long,
            device=self.atributos.device,
        )
        if t0 is not None and self.atributos.device.type == "cuda":
            torch.cuda.synchronize(self.atributos.device)
        if t0 is not None:
            self._registrar_perfil(
                "transferencia_mapa_final_s", time.perf_counter() - t0,
            )
        # El mapa se genero en orden (x,y,z). Se devuelve (B,C,X,Y,Z).
        denso = self.atributos.index_select(0, indices)
        denso = denso.reshape(self.batch_size, 8, 8, 8, -1)
        return denso.permute(0, 4, 1, 2, 3).contiguous()


class ConvolucionOctree3x3(nn.Module):
    """Convolucion 3x3x3 exacta sobre hojas, sin ``torch.nn.Conv3d``."""

    def __init__(self, canales_entrada: int, canales_salida: int,
                 bias: bool = True):
        super().__init__()
        self.canales_entrada = int(canales_entrada)
        self.canales_salida = int(canales_salida)
        self.weight = nn.Parameter(torch.empty(
            canales_salida, canales_entrada, 3, 3, 3,
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(canales_salida))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(
            self.weight, mode="fan_out", nonlinearity="relu",
        )
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, lote: LoteGridOctree) -> LoteGridOctree:
        if lote.atributos.shape[1] != self.canales_entrada:
            raise ValueError(
                f"Se esperaban {self.canales_entrada} canales y llegaron "
                f"{lote.atributos.shape[1]}"
            )
        salida_idx, fila_proyeccion, coeficiente = (
            lote.plan_convolucion_fusionado_torch()
        )
        n_hojas = lote.atributos.shape[0]
        if self.bias is None:
            salida = lote.atributos.new_zeros((n_hojas, self.canales_salida))
        else:
            salida = self.bias.unsqueeze(0).expand(n_hojas, -1).clone()

        # (C_out, C_in, 27) -> (C_in, 27 * C_out): la columna k * C_out + o
        # corresponde al kernel k y al canal de salida o.
        pesos = self.weight.reshape(
            self.canales_salida, self.canales_entrada, 27,
        ).permute(1, 2, 0).reshape(self.canales_entrada, -1)
        proyeccion = (lote.atributos @ pesos).view(
            n_hojas * 27, self.canales_salida,
        )
        contribucion = (
            proyeccion.index_select(0, fila_proyeccion)
            * coeficiente[:, None]
        )
        salida.index_add_(0, salida_idx, contribucion)
        return lote.con_atributos(salida)


class MaxPoolOctree2(nn.Module):
    """Max-pooling 2x2x2 que transforma directamente la jerarquia."""

    def forward(self, lote: LoteGridOctree) -> LoteGridOctree:
        return lote.max_pool2()
