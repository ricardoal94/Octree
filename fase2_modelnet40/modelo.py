"""
modelo.py - Fase 2
==================
Implementacion de Net5 basada en la arquitectura PointNet,
ahora con T-Net (modulos de alineacion espacial).

Arquitectura:
    Entrada       : (B, 3, N)  — batch de nubes de puntos
    Input T-Net   : aprende una matriz 3x3 para alinear la nube de entrada
    Feature T-Net : aprende una matriz 64x64 para alinear features intermedias
    Salida        : (B, 40)    — logits por clase

Capas:
    T-Net(3x3) → Conv1D x2 → T-Net(64x64) → Conv1D x3 → MaxPool global → FC x3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# BLOQUE BASICO: Conv1D + BN + ReLU
# ──────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv1D → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class LinBnRelu(nn.Module):
    """Linear → BatchNorm → ReLU."""

    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=False)
        self.bn = nn.BatchNorm1d(out_f)

    def forward(self, x):
        return F.relu(self.bn(self.fc(x)))


# ──────────────────────────────────────────────────────────────
# T-NET — RED DE ALINEACION ESPACIAL
# ──────────────────────────────────────────────────────────────

class TNet(nn.Module):
    """
    Red de transformacion (T-Net) del paper PointNet.

    Aprende una matriz de transformacion k x k que se aplica a la entrada
    para alinearla canonicamente, haciendo al modelo mas robusto a
    rotaciones y variaciones de pose.

    Parametros
    ----------
    k : dimension de la matriz de transformacion (3 para input, 64 para features)
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

        # Extractor compartido de features (igual estilo a PointNet)
        self.conv1 = ConvBnRelu(k,   64)
        self.conv2 = ConvBnRelu(64,  128)
        self.conv3 = ConvBnRelu(128, 1024)

        self.fc1 = LinBnRelu(1024, 512)
        self.fc2 = LinBnRelu(512,  256)
        self.fc3 = nn.Linear(256, k * k)

        # Inicializar la transformacion final como identidad
        nn.init.constant_(self.fc3.weight, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : tensor (B, k, N)
        Retorna la matriz de transformacion (B, k, k)
        """
        batch_size = x.size(0)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.max(x, dim=2)[0]   # (B, 1024)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)              # (B, k*k)

        # Sumar la matriz identidad para que la transformacion inicial
        # sea cercana a "no hacer nada" (estabiliza el entrenamiento)
        identidad = torch.eye(self.k, device=x.device, dtype=x.dtype).flatten()
        x = x + identidad.unsqueeze(0).repeat(batch_size, 1)
        x = x.view(batch_size, self.k, self.k)

        return x


# ──────────────────────────────────────────────────────────────
# NET5 — ARQUITECTURA PRINCIPAL CON T-NET
# ──────────────────────────────────────────────────────────────

class Net5(nn.Module):
    """
    PointNet simplificado con 5 capas convolucionales (Net5) + T-Net.

    Parametros
    ----------
    num_clases : numero de categorias de salida (40 para ModelNet40)
    dropout    : tasa de dropout en las capas FC
    usar_tnet_features : si aplicar tambien la alineacion de features (64x64).
                          Recomendado True, pero se puede desactivar para
                          ahorrar computo si es necesario.
    """

    def __init__(self, num_clases: int = 40, dropout: float = 0.3,
                 usar_tnet_features: bool = True):
        super().__init__()
        self.usar_tnet_features = usar_tnet_features

        # ── T-Net de entrada: alinea la nube de puntos (3x3) ──
        self.input_tnet = TNet(k=3)

        # ── Primeras 2 capas Conv1D (antes del feature T-Net) ──
        self.conv1 = ConvBnRelu(3,  64)
        self.conv2 = ConvBnRelu(64, 64)

        # ── T-Net de features: alinea el espacio de features (64x64) ──
        if usar_tnet_features:
            self.feature_tnet = TNet(k=64)

        # ── Resto de capas Conv1D (5 en total) ──
        self.conv3 = ConvBnRelu(64,  128)
        self.conv4 = ConvBnRelu(128, 256)
        self.conv5 = ConvBnRelu(256, 1024)

        # ── Clasificador global (3 FC) ──
        self.fc1 = LinBnRelu(1024, 512)
        self.fc2 = LinBnRelu(512,  256)
        self.fc3 = nn.Linear(256, num_clases)

        self.dropout = nn.Dropout(p=dropout)

        # Inicializar pesos (con seed ya fijada globalmente)
        self._init_pesos()

    def _init_pesos(self):
        """
        Inicializacion Kaiming para Conv y Xavier para FC.
        Excepcion: la ultima capa fc3 de cada TNet ya fue inicializada
        manualmente como identidad (ver clase TNet) y no debe sobreescribirse.
        """
        # IDs de las capas fc3 de los T-Net, para excluirlas del loop generico
        capas_tnet_excluir = set()
        capas_tnet_excluir.add(id(self.input_tnet.fc3))
        if self.usar_tnet_features:
            capas_tnet_excluir.add(id(self.feature_tnet.fc3))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                if id(m) not in capas_tnet_excluir:
                    nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, retornar_trans_feat: bool = False):
        """
        Parametros
        ----------
        x : tensor (B, 3, N)
        retornar_trans_feat : si True, tambien retorna la matriz de
                               transformacion de features (para regularizacion)

        Retorna
        -------
        logits : tensor (B, num_clases)
        trans_feat (opcional) : tensor (B, 64, 64)
        """
        # ── Input T-Net: alinear la nube de puntos ──
        trans_input = self.input_tnet(x)               # (B, 3, 3)
        x = x.transpose(2, 1)                            # (B, N, 3)
        x = torch.bmm(x, trans_input)                     # aplicar transformacion
        x = x.transpose(2, 1)                             # (B, 3, N)

        # ── Primeras capas conv ──
        x = self.conv1(x)   # (B, 64, N)
        x = self.conv2(x)   # (B, 64, N)

        # ── Feature T-Net: alinear el espacio de features ──
        trans_feat = None
        if self.usar_tnet_features:
            trans_feat = self.feature_tnet(x)              # (B, 64, 64)
            x = x.transpose(2, 1)                            # (B, N, 64)
            x = torch.bmm(x, trans_feat)                      # aplicar transformacion
            x = x.transpose(2, 1)                             # (B, 64, N)

        # ── Resto de capas conv ──
        x = self.conv3(x)   # (B, 128,  N)
        x = self.conv4(x)   # (B, 256,  N)
        x = self.conv5(x)   # (B, 1024, N)

        # Agregacion global: max pooling sobre todos los puntos
        x = torch.max(x, dim=2)[0]   # (B, 1024)

        # Clasificador
        x = self.dropout(self.fc1(x))   # (B, 512)
        x = self.dropout(self.fc2(x))   # (B, 256)
        x = self.fc3(x)                  # (B, 40)

        if retornar_trans_feat:
            return x, trans_feat
        return x

    def contar_parametros(self) -> int:
        """Retorna el total de parametros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────
# REGULARIZACION DE ORTOGONALIDAD (estabiliza el feature T-Net)
# ──────────────────────────────────────────────────────────────

def perdida_ortogonalidad(trans_feat: torch.Tensor) -> torch.Tensor:
    """
    Penaliza que la matriz de transformacion de features (64x64) se
    aleje de ser ortogonal (A @ A^T ≈ I). Esto evita que el T-Net de
    features colapse o distorsione el espacio de forma arbitraria.

    Se usa como termino adicional en la funcion de perdida:
        loss_total = loss_clasificacion + 0.001 * perdida_ortogonalidad(trans_feat)
    """
    if trans_feat is None:
        return torch.tensor(0.0, device="cpu")

    batch_size, k, _ = trans_feat.shape
    identidad = torch.eye(k, device=trans_feat.device).unsqueeze(0).repeat(batch_size, 1, 1)
    diff = torch.bmm(trans_feat, trans_feat.transpose(2, 1)) - identidad
    return torch.mean(torch.norm(diff, dim=(1, 2)))


# ──────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Detecta y retorna el dispositivo disponible (CUDA o CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        nombre  = torch.cuda.get_device_name(0)
        memoria = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Dispositivo] GPU: {nombre} ({memoria:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[Dispositivo] CPU (CUDA no disponible)")
    return device


def crear_modelo(num_clases: int = 40, dropout: float = 0.3,
                 device: torch.device = None,
                 usar_tnet_features: bool = True) -> tuple:
    """Instancia Net5 (con T-Net), la mueve al dispositivo y muestra resumen."""
    if device is None:
        device = get_device()
    modelo = Net5(
        num_clases=num_clases, dropout=dropout,
        usar_tnet_features=usar_tnet_features,
    ).to(device)
    params = modelo.contar_parametros()
    print(f"\n[Modelo] Net5 + T-Net creado:")
    print(f"  Clases        : {num_clases}")
    print(f"  Dropout       : {dropout}")
    print(f"  Feature T-Net : {usar_tnet_features}")
    print(f"  Parametros    : {params:,}")
    print(f"  Dispositivo   : {device}")
    return modelo, device


# ──────────────────────────────────────────────────────────────
# TEST RAPIDO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")

    torch.manual_seed(42)

    modelo, device = crear_modelo(num_clases=40, dropout=0.3)

    # Batch simulado en el mismo dispositivo que el modelo
    x_dummy = torch.randn(4, 3, 1024).to(device)
    logits, trans_feat = modelo(x_dummy, retornar_trans_feat=True)

    print(f"\n[Test] Input  shape       : {x_dummy.shape}")
    print(f"[Test] Output shape       : {logits.shape}")        # (4, 40)
    print(f"[Test] Trans feat shape   : {trans_feat.shape}")    # (4, 64, 64)
    print(f"[Test] Perdida ortogonal  : {perdida_ortogonalidad(trans_feat).item():.6f}")
    print(f"[Test] Logits muestra     : {logits[0, :5].detach().cpu().numpy().round(3)}")
    print("\nModelo Net5 + T-Net verificado correctamente.")
