"""Net5-Octree nativo y referencia densa de la Tabla 5.

``Net5Octree`` ejecuta convolucion y pooling sobre la estructura dispersa
grid-octree mediante :mod:`octnet_backend`. ``DenseTabla5Reference`` conserva
la misma topologia con ``torch.nn.Conv3d`` exclusivamente para diagnosticos y
no puede producir resultados oficiales del objetivo 3.

Fuente de la arquitectura
--------------------------
Net5 corresponde a la variante de **capacidad fija** ("keep the capacity
of the model, i.e., the number of parameters, constant") de OctNet
(Riegler, Ulusoy, Geiger, "OctNet: Learning Deep 3D Representations at
High Resolutions", CVPR 2017), documentada en la **Tabla 5** ("Network
Architectures ModelNet10 Classification") del material suplementario del
paper: https://www.cvlibs.net/publications/Riegler2017CVPR_supplementary.pdf

La Tabla 5 define 5 bloques de dos convoluciones 3^3 cada uno, con
canales FIJOS independientes de la resolucion de entrada:

    Bloque 1: conv(Cin, 8)  -> conv(8, 14)
    Bloque 2: conv(14, 14)  -> conv(14, 20)
    Bloque 3: conv(20, 20)  -> conv(20, 26)
    Bloque 4: conv(26, 26)  -> conv(26, 32)
    Bloque 5: conv(32, 32)  -> conv(32, 32)

seguidos de Dropout(0.5) -> FC(512) -> FC(num_clases) -> SoftMax.

Lo que varia segun la resolucion de entrada R es *cuantos* de esos 5
bloques terminan en maxpool(2): la tabla aplica maxpool "eliminando
capas de pooling desde el inicio de la red" para resoluciones menores,
de forma que TODAS las resoluciones terminan en 8^3 antes del
clasificador FC y el numero de parametros de la red permanece
constante. El numero de maxpool necesarios es:

    N_pool(R) = log2(R / 8)

y se aplican en los ULTIMOS N_pool bloques:

    R=32^3 (N_pool=2): bloques 1-3 sin pool, pool tras bloques 4 y 5
                       32 -> 32 -> 32 -> 32 -> 16 -> 8
    R=64^3 (N_pool=3): bloques 1-2 sin pool, pool tras bloques 3, 4 y 5
                       64 -> 64 -> 64 -> 32 -> 16 -> 8

Adaptaciones del proyecto respecto al paper original:
  - Canales de entrada: 4 (ocupacion + normal promedio nx, ny, nz) en
    vez de 1 (solo ocupacion binaria), acorde a la codificacion de hojas
    definida en la Fase 2 (preprocesar_octrees.py / octree_real.py).
  - Capa de salida: 40 clases (ModelNet40) en vez de 10 (ModelNet10).
  - La capa SoftMax final se omite porque nn.CrossEntropyLoss ya la
    aplica internamente sobre los logits.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .octnet_backend import (
        ConvolucionOctree3x3,
        LoteGridOctree,
        MaxPoolOctree2,
    )
except ImportError:  # Ejecucion directa desde fase3_net5/
    from octnet_backend import (
        ConvolucionOctree3x3,
        LoteGridOctree,
        MaxPoolOctree2,
    )


# ──────────────────────────────────────────────────────────────
# BLOQUE BASICO: conv(Cin, Cout) 3^3, stride 1 + ReLU (notacion Tabla 5)
# ──────────────────────────────────────────────────────────────

class ConvReLU3D(nn.Module):
    """conv(C_in, C_out): Conv3D 3x3x3, stride 1, padding 'same' + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x))


class BloqueConv3DDenso(nn.Module):
    """
    Un bloque de la Tabla 5: dos conv(3^3) consecutivas seguidas de un
    maxpool(2) opcional. El maxpool se omite en los bloques iniciales
    para las resoluciones de entrada menores, manteniendo el numero de
    parametros de la red fijo independientemente de R (ver docstring
    del modulo).
    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, con_pool: bool):
        super().__init__()
        self.conv_a = ConvReLU3D(in_ch, mid_ch)
        self.conv_b = ConvReLU3D(mid_ch, out_ch)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2) if con_pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_a(x)
        x = self.conv_b(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


# Canales (mid, out) de los 5 bloques de la Tabla 5 -- fijos, no dependen de R
_CANALES_BLOQUES = [
    (8, 14),
    (14, 20),
    (20, 26),
    (26, 32),
    (32, 32),
]


# ──────────────────────────────────────────────────────────────
# REFERENCIA DENSA DE LA TOPOLOGIA DE CAPACIDAD FIJA (TABLA 5)
# ──────────────────────────────────────────────────────────────

class DenseTabla5Reference(nn.Module):
    """
    CNN 3D densa con la topologia de capacidad fija de la Tabla 5.

    No debe denominarse Net5/OctNet en reportes experimentales porque no
    ejecuta convoluciones ni pooling sobre los nodos del grid-octree.

    Parametros
    ----------
    resolucion    : 32 o 64
    num_clases    : 40 para ModelNet40
    dropout       : tasa de dropout antes del clasificador FC (0.5 en la Tabla 5)
    in_channels   : 4 (ocupacion + nx + ny + nz)
    """

    RESOLUCION_BASE = 8  # resolucion espacial final antes del clasificador FC

    def __init__(self, resolucion: int = 32, num_clases: int = 40,
                 dropout: float = 0.5, in_channels: int = 4):
        super().__init__()

        assert resolucion in (32, 64), "Resolucion debe ser 32 o 64"
        self.resolucion = resolucion
        self.num_clases = num_clases

        n_bloques = len(_CANALES_BLOQUES)
        n_pool = int(round(math.log2(resolucion / self.RESOLUCION_BASE)))
        assert 0 <= n_pool <= n_bloques, (
            f"Resolucion {resolucion} requeriria {n_pool} maxpools, "
            f"pero la red solo tiene {n_bloques} bloques"
        )
        self.n_pool = n_pool

        bloques = []
        canal_in = in_channels
        for i, (mid, out) in enumerate(_CANALES_BLOQUES):
            bloque_idx = i + 1  # 1-indexado
            con_pool = bloque_idx > (n_bloques - n_pool)
            bloques.append(BloqueConv3DDenso(canal_in, mid, out, con_pool))
            canal_in = out
        self.bloques = nn.Sequential(*bloques)

        canal_final = _CANALES_BLOQUES[-1][1]  # 32
        self.dim_fc = canal_final * (self.RESOLUCION_BASE ** 3)

        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(self.dim_fc, 512)
        self.fc2 = nn.Linear(512, num_clases)

        self._init_pesos()

    def _init_pesos(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 4, R, R, R)
        Retorna logits (B, num_clases). La normalizacion SoftMax se
        omite: nn.CrossEntropyLoss la aplica internamente.
        """
        x = self.bloques(x)             # (B, 32, 8, 8, 8)
        x = x.flatten(1)                # (B, 32*8*8*8)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))         # (B, 512)
        x = self.fc2(x)                 # (B, num_clases)
        return x

    def contar_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BloqueConvOctree(nn.Module):
    """Dos convoluciones OctNet con ReLU y max-pooling opcional."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, con_pool: bool):
        super().__init__()
        self.conv_a = ConvolucionOctree3x3(in_ch, mid_ch)
        self.conv_b = ConvolucionOctree3x3(mid_ch, out_ch)
        self.pool = MaxPoolOctree2() if con_pool else None

    def forward(self, lote: LoteGridOctree) -> LoteGridOctree:
        lote = self.conv_a(lote)
        lote = lote.con_atributos(F.relu(lote.atributos))
        lote = self.conv_b(lote)
        lote = lote.con_atributos(F.relu(lote.atributos))
        if self.pool is not None:
            lote = self.pool(lote)
        return lote


class Net5Octree(nn.Module):
    """Net5 de capacidad fija sobre el backend grid-octree disperso.

    La topologia y los canales corresponden a la Tabla 5. R=32 aplica
    pooling tras los bloques 4 y 5; R=64, tras los bloques 3, 4 y 5. Ambas
    variantes llegan a 8^3 y tienen exactamente los mismos parametros.
    """

    RESOLUCION_BASE = 8

    def __init__(self, resolucion: int = 32, num_clases: int = 40,
                 dropout: float = 0.5, in_channels: int = 4):
        super().__init__()
        if resolucion not in (32, 64):
            raise ValueError("Resolucion debe ser 32 o 64")
        self.resolucion = int(resolucion)
        self.num_clases = int(num_clases)
        self.n_pool = int(round(math.log2(resolucion / self.RESOLUCION_BASE)))

        bloques = []
        canal_in = in_channels
        n_bloques = len(_CANALES_BLOQUES)
        for i, (mid, out) in enumerate(_CANALES_BLOQUES):
            bloque_idx = i + 1
            con_pool = bloque_idx > (n_bloques - self.n_pool)
            bloques.append(BloqueConvOctree(canal_in, mid, out, con_pool))
            canal_in = out
        self.bloques = nn.ModuleList(bloques)

        canal_final = _CANALES_BLOQUES[-1][1]
        self.dim_fc = canal_final * (self.RESOLUCION_BASE ** 3)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(self.dim_fc, 512)
        self.fc2 = nn.Linear(512, num_clases)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, lote: LoteGridOctree) -> torch.Tensor:
        if not isinstance(lote, LoteGridOctree):
            raise TypeError("Net5Octree requiere un LoteGridOctree disperso")
        if lote.resolucion != self.resolucion:
            raise ValueError(
                f"El modelo R={self.resolucion} recibio un lote "
                f"R={lote.resolucion}"
            )
        for bloque in self.bloques:
            lote = bloque(lote)
        x = lote.tensor_final_8()
        x = x.flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def contar_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        nombre = torch.cuda.get_device_name(0)
        memoria = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Dispositivo] GPU: {nombre} ({memoria:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[Dispositivo] CPU")
    return device


def crear_modelo_denso_referencia(
    resolucion: int = 32,
    num_clases: int = 40,
    dropout: float = 0.5,
    device: torch.device = None,
) -> tuple:
    if device is None:
        device = get_device()
    modelo = DenseTabla5Reference(
        resolucion=resolucion, num_clases=num_clases, dropout=dropout,
    ).to(device)
    params = modelo.contar_parametros()
    print("\n[Modelo] Referencia densa creada (topologia de la Tabla 5):")
    print(f"  Resolucion    : {resolucion}^3  (maxpools aplicados={modelo.n_pool})")
    print(f"  Clases        : {num_clases}")
    print(f"  Dropout       : {dropout}")
    print(f"  Parametros    : {params:,}")
    print(f"  Dispositivo   : {device}")
    return modelo, device


def crear_modelo(
    resolucion: int = 32,
    num_clases: int = 40,
    dropout: float = 0.5,
    device: torch.device = None,
) -> tuple:
    """Crea el modelo oficial con operaciones directas sobre grid-octrees."""

    if device is None:
        device = get_device()
    modelo = Net5Octree(
        resolucion=resolucion,
        num_clases=num_clases,
        dropout=dropout,
    ).to(device)
    params = modelo.contar_parametros()
    print("\n[Modelo] Net5-Octree nativo creado:")
    print(f"  Resolucion    : {resolucion}^3  (maxpools={modelo.n_pool})")
    print(f"  Clases        : {num_clases}")
    print(f"  Dropout       : {dropout}")
    print(f"  Parametros    : {params:,}")
    print(f"  Dispositivo   : {device}")
    return modelo, device


# ──────────────────────────────────────────────────────────────
# TEST RAPIDO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    params_r32 = params_r64 = None
    for R in (32, 64):
        modelo, device = crear_modelo_denso_referencia(resolucion=R)
        x = torch.randn(2, 4, R, R, R).to(device)
        logits = modelo(x)
        print(f"  Input {tuple(x.shape)} -> Output {tuple(logits.shape)}\n")
        if R == 32:
            params_r32 = modelo.contar_parametros()
        else:
            params_r64 = modelo.contar_parametros()

    assert params_r32 == params_r64, (
        "El numero de parametros deberia ser identico para R=32 y R=64 "
        "(topologia de capacidad fija, Tabla 5)"
    )
    print(f"[OK] Parametros identicos en R=32 y R=64: {params_r32:,}")
