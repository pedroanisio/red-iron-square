"""Expected Free Energy bounded context.

Provides an EFE-based decision engine as an alternative to the utility
dot-product, decomposing action value into epistemic (curiosity) and
pragmatic (preference alignment) components.
"""

from src.efe.c_vector import CVector
from src.efe.engine import EFEEngine
from src.efe.params import EFEParams

__all__ = ["CVector", "EFEEngine", "EFEParams"]
