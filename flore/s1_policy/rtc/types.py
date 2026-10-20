"""Shared data structures for S1 action-policy inference.

Ported verbatim from FAR-lab42_vr rfm_heavy inference/vla/utils.py (itself
from FAR-pi rfmpi/inference). Kept as a standalone module so the RTC layer
has no dependency on the model package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Observation:
    """Observation with a global index for tracking."""

    images: np.ndarray
    state: np.ndarray
    prompt: str | None
    obs_idx: int


@dataclass
class ActionChunk:
    """Action chunk with the observation index it was generated from."""

    actions: np.ndarray
    obs_idx: int
