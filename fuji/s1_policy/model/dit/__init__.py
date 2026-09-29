"""1D flow-matching decoders for DP-Flow.

``ConditionalDiT1D`` is the default (AdaLN + RoPE, per-position time for RTC);
``ConditionalUnet1D`` is a FiLM alternative without per-position time / RTC.
"""

from fuji.s1_policy.model.dit.conditional_dit1d import ConditionalDiT1D
from fuji.s1_policy.model.dit.conditional_unet1d import ConditionalUnet1D

__all__ = ["ConditionalDiT1D", "ConditionalUnet1D"]
