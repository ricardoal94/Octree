"""
img2voxel_modelo.py - Experimento adicional: Img2Voxel
==========================================================
Arquitectura encoder-decoder para reconstruccion 3D aproximada desde
una sola imagen 2D.

IMPORTANTE: este modelo es independiente de Net5 (el clasificador de
tu metodologia principal). Es un experimento adicional que entrena
desde cero con un objetivo distinto: reconstruccion, no clasificacion.

Arquitectura:
    Encoder 2D  : imagen (1, 128, 128) -> vector latente (512,)
                  CNN 2D con 5 bloques Conv2D + stride, GAP final
    Decoder 3D  : vector latente (512,) -> voxel grid (4, R, R, R)
                  FC inicial -> reshape a (C, 2, 2, 2) -> ConvTranspose3D
                  sucesivas hasta alcanzar resolucion R (32 o 64)

Salida: 4 canales [ocupacion, nx, ny, nz], igual formato que octree.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# ENCODER 2D: imagen -> vector latente
# ──────────────────────────────────────────────────────────────

class ConvBnRelu2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Encoder2D(nn.Module):
    """
    CNN 2D que comprime una imagen 128x128 en escala de grises a un
    vector latente de 512 dimensiones.

    128 -> 64 -> 32 -> 16 -> 8 -> 4  (5 bloques con stride=2)
    Luego Global Average Pooling -> FC -> vector latente
    """
    def __init__(self, dim_latente: int = 512, in_channels: int = 1):
        super().__init__()
        self.bloques = nn.Sequential(
            ConvBnRelu2D(in_channels, 32),    # 128 -> 64
            ConvBnRelu2D(32, 64),              # 64  -> 32
            ConvBnRelu2D(64, 128),             # 32  -> 16
            ConvBnRelu2D(128, 256),            # 16  -> 8
            ConvBnRelu2D(256, 512),            # 8   -> 4
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(512, dim_latente)
        self.bn_fc = nn.BatchNorm1d(dim_latente)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 128, 128) -> retorna (B, dim_latente)"""
        x = self.bloques(x)
        x = self.gap(x).flatten(1)
        x = F.relu(self.bn_fc(self.fc(x)))
        return x


# ──────────────────────────────────────────────────────────────
# DECODER 3D: vector latente -> voxel grid
# ──────────────────────────────────────────────────────────────

class ConvTransposeBnRelu3D(nn.Module):
    """ConvTranspose3D(stride=2) -> BatchNorm3D -> ReLU. Duplica la resolucion espacial."""
    def __init__(self, in_ch, out_ch, ultima_capa=False):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4,
                                       stride=2, padding=1, bias=False)
        self.ultima_capa = ultima_capa
        if not ultima_capa:
            self.bn = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        if self.ultima_capa:
            return x   # sin BN/ReLU en la ultima capa (logits crudos)
        return F.relu(self.bn(x))


class Decoder3D(nn.Module):
    """
    Decoder que expande un vector latente a un voxel grid de resolucion
    R (32 o 64), con 4 canales de salida [ocupacion, nx, ny, nz].

    El vector latente se proyecta primero a un cubo pequeno (512, 2, 2, 2)
    y luego se aplican ConvTranspose3D sucesivas duplicando la resolucion:
        R=32: 2 -> 4 -> 8 -> 16 -> 32        (4 deconvoluciones)
        R=64: 2 -> 4 -> 8 -> 16 -> 32 -> 64  (5 deconvoluciones)
    """
    def __init__(self, resolucion: int = 32, dim_latente: int = 512,
                out_channels: int = 4):
        super().__init__()
        assert resolucion in (32, 64)
        self.resolucion = resolucion
        n_capas = 4 if resolucion == 32 else 5

        # Proyeccion inicial: vector -> cubo (512, 2, 2, 2)
        self.fc_inicial = nn.Linear(dim_latente, 512 * 2 * 2 * 2)
        self.bn_inicial = nn.BatchNorm1d(512 * 2 * 2 * 2)

        canales = [512, 256, 128, 64, 32, 16]
        capas = []
        for i in range(n_capas):
            in_ch  = canales[i]
            out_ch = canales[i + 1]
            es_ultima = (i == n_capas - 1)
            if es_ultima:
                capas.append(ConvTransposeBnRelu3D(in_ch, out_channels, ultima_capa=True))
            else:
                capas.append(ConvTransposeBnRelu3D(in_ch, out_ch))
        self.capas = nn.Sequential(*capas)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, dim_latente) -> retorna (B, 4, R, R, R) logits crudos"""
        B = z.size(0)
        x = F.relu(self.bn_inicial(self.fc_inicial(z)))
        x = x.view(B, 512, 2, 2, 2)
        x = self.capas(x)
        return x


# ──────────────────────────────────────────────────────────────
# MODELO COMPLETO: Img2Voxel
# ──────────────────────────────────────────────────────────────

class Img2Voxel(nn.Module):
    """
    Modelo completo encoder-decoder: imagen 2D -> voxel grid 3D.

    La salida tiene 4 canales con logits crudos (sin activacion):
        canal 0    : logit de ocupacion (aplicar sigmoid para probabilidad)
        canales1-3: vector normal sin restringir (se puede normalizar
                     post-hoc si se desea un vector unitario)
    """
    def __init__(self, resolucion: int = 32, dim_latente: int = 512):
        super().__init__()
        self.resolucion = resolucion
        self.encoder = Encoder2D(dim_latente=dim_latente, in_channels=1)
        self.decoder = Decoder3D(resolucion=resolucion, dim_latente=dim_latente,
                                 out_channels=4)
        self._init_pesos()

    def _init_pesos(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imagen: torch.Tensor) -> torch.Tensor:
        """imagen: (B, 1, 128, 128) -> retorna voxel_logits (B, 4, R, R, R)"""
        z = self.encoder(imagen)
        voxel_logits = self.decoder(z)
        return voxel_logits

    def contar_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────
# FUNCION DE PERDIDA DE RECONSTRUCCION
# ──────────────────────────────────────────────────────────────

def perdida_reconstruccion(voxel_logits: torch.Tensor, voxel_real: torch.Tensor,
                           peso_normales: float = 1.0) -> dict:
    """
    Calcula la perdida de reconstruccion combinando:
      - BCE (Binary Cross-Entropy) para el canal de ocupacion
      - MSE para los canales de normales, PERO SOLO en las celdas
        realmente ocupadas (no tiene sentido penalizar normales en
        espacio vacio, donde el valor real es (0,0,0) por definicion)

    Parametros
    ----------
    voxel_logits : (B, 4, R, R, R) salida cruda del modelo
    voxel_real   : (B, 4, R, R, R) grid real (ocupacion + normales)
    peso_normales: factor de ponderacion del termino de normales

    Retorna
    -------
    dict con 'perdida_total', 'perdida_ocupacion', 'perdida_normales'
    """
    ocup_logits = voxel_logits[:, 0]       # (B, R, R, R)
    ocup_real   = voxel_real[:, 0]

    normales_pred = voxel_logits[:, 1:4]    # (B, 3, R, R, R)
    normales_real = voxel_real[:, 1:4]

    # BCE con logits (mas estable numericamente que sigmoid + BCE separados)
    perdida_ocup = F.binary_cross_entropy_with_logits(ocup_logits, ocup_real)

    # MSE de normales, enmascarado solo donde hay ocupacion real
    mascara = (ocup_real > 0.5).unsqueeze(1).float()   # (B, 1, R, R, R)
    n_ocupadas = mascara.sum().clamp(min=1.0)

    diff_normales = (normales_pred - normales_real) * mascara
    perdida_normales = (diff_normales ** 2).sum() / (n_ocupadas * 3)

    perdida_total = perdida_ocup + peso_normales * perdida_normales

    return {
        "perdida_total":      perdida_total,
        "perdida_ocupacion":  perdida_ocup,
        "perdida_normales":   perdida_normales,
    }


# ──────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        nombre  = torch.cuda.get_device_name(0)
        memoria = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Dispositivo] GPU: {nombre} ({memoria:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[Dispositivo] CPU")
    return device


def crear_modelo(resolucion: int = 32, dim_latente: int = 512,
                 device: torch.device = None) -> tuple:
    if device is None:
        device = get_device()
    modelo = Img2Voxel(resolucion=resolucion, dim_latente=dim_latente).to(device)
    params = modelo.contar_parametros()
    print(f"\n[Modelo] Img2Voxel creado:")
    print(f"  Resolucion salida : {resolucion}^3")
    print(f"  Dim. latente      : {dim_latente}")
    print(f"  Parametros        : {params:,}")
    print(f"  Dispositivo       : {device}")
    return modelo, device


# ──────────────────────────────────────────────────────────────
# TEST RAPIDO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    for R in (32, 64):
        print(f"\n{'='*50}\nTest resolucion {R}\n{'='*50}")
        modelo, device = crear_modelo(resolucion=R)

        x_img = torch.randn(2, 1, 128, 128).to(device)
        voxel_out = modelo(x_img)
        print(f"  Input  imagen : {x_img.shape}")
        print(f"  Output voxel  : {voxel_out.shape}")
        assert voxel_out.shape == (2, 4, R, R, R), "Shape de salida incorrecto"

        # Test de la funcion de perdida
        voxel_real = torch.zeros(2, 4, R, R, R).to(device)
        voxel_real[:, 0] = (torch.rand(2, R, R, R) > 0.9).float().to(device)  # ~10% ocupado
        perdidas = perdida_reconstruccion(voxel_out, voxel_real)
        print(f"  Perdida total      : {perdidas['perdida_total'].item():.4f}")
        print(f"  Perdida ocupacion  : {perdidas['perdida_ocupacion'].item():.4f}")
        print(f"  Perdida normales   : {perdidas['perdida_normales'].item():.4f}")

    print("\nTodos los tests pasaron correctamente.")
