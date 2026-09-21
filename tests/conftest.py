import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FASE1 = ROOT / "fase1_modelnet40"
FASE2 = ROOT / "fase2_octree"
FASE3_HCE = ROOT / "fase3_hce"
FASE3_NET5 = ROOT / "fase3_net5"
sys.path.insert(0, str(FASE1))
sys.path.insert(0, str(FASE2))
sys.path.insert(0, str(FASE3_HCE))
sys.path.insert(0, str(FASE3_NET5))
