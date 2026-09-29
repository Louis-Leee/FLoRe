"""Normalizer loading and observation clipping utilities.

Shared between sim2sim (headless eval) and sim2real
(holosoma_wbt.py, holosoma_maskedmimic.py) to ensure identical
observation pre-processing.

The ONNX model has normalizer weights baked in. We pre-clip the raw
observations so that the baked-in normalizer produces values in
[-NORM_CLIP, NORM_CLIP], preventing extreme outliers from masked or
out-of-distribution entries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Must match EmpiricalNormalization in ppo.py
NORM_CLIP = 3.0
NORM_EPS = 1e-2


def load_normalizer_from_path(model_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load normalizer mean and std+eps from checkpoint files.

    Searches (in order):
    1. ONNX metadata (``normalizer_mean``/``normalizer_std``) — self-contained, no extra files
    2. ``<model_base>_normalizer.npz`` — lightweight, no torch needed
    3. ``<model_base>.pt`` — full checkpoint, needs torch

    Returns (mean, std_with_eps) or (None, None) if not found.
    The returned std already has NORM_EPS added.
    """
    base = str(Path(model_path).with_suffix(""))

    # Try ONNX metadata first (self-contained, no extra files needed)
    onnx_path = base + ".onnx" if not model_path.endswith(".onnx") else model_path
    if Path(onnx_path).is_file():
        try:
            import onnx

            model = onnx.load(onnx_path)
            meta = {p.key: p.value for p in model.metadata_props}
            if "normalizer_mean" in meta and "normalizer_std" in meta:
                mean = np.array([float(x) for x in meta["normalizer_mean"].split(",")], dtype=np.float32)
                std_eps = np.array([float(x) for x in meta["normalizer_std"].split(",")], dtype=np.float32) + NORM_EPS
                return mean, std_eps
        except Exception:
            pass

    # Try pre-extracted NPZ (works on Jetson without torch)
    npz_path = base + "_normalizer.npz"
    if Path(npz_path).is_file():
        try:
            data = np.load(npz_path)
            mean = data["mean"].flatten().astype(np.float32)
            std_eps = data["std"].flatten().astype(np.float32) + NORM_EPS
            return mean, std_eps
        except Exception:
            pass

    # Try .pt checkpoint (legacy fallback)
    pt_path = base + ".pt"
    if Path(pt_path).is_file():
        try:
            import torch

            ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)

            # Format 1: dense WBT checkpoints
            norm_sd = ckpt.get("actor_obs_normalizer_state_dict")
            if norm_sd and "_mean" in norm_sd:
                mean = norm_sd["_mean"].numpy().flatten().astype(np.float32)
                std_eps = norm_sd["_std"].numpy().flatten().astype(np.float32) + NORM_EPS
                return mean, std_eps

            # Format 2: MaskedMimic / holosoma checkpoints (running_mean/running_var)
            norm_sd = ckpt.get("obs_normalizer")
            if norm_sd and "running_mean" in norm_sd:
                mean = norm_sd["running_mean"].numpy().flatten().astype(np.float32)
                var = norm_sd["running_var"].numpy().flatten().astype(np.float32)
                std_eps = np.sqrt(var + 1e-5).astype(np.float32) + NORM_EPS
                return mean, std_eps
        except Exception:
            pass

    return None, None


def clip_normalized_obs(
    obs: np.ndarray, norm_mean: np.ndarray, norm_std_eps: np.ndarray, clip: float = NORM_CLIP
) -> np.ndarray:
    """Pre-clip obs so the ONNX baked-in normalizer produces clipped values.

    The ONNX model computes: ``normalized = (obs - mean) / std_eps``
    We want:                 ``clipped = clip(normalized, -clip, clip)``
    So we feed:              ``obs' = clipped * std_eps + mean``
    Then ONNX normalizer produces the clipped result.
    """
    normalized = (obs - norm_mean) / norm_std_eps
    clipped = np.clip(normalized, -clip, clip)
    return (clipped * norm_std_eps + norm_mean).astype(np.float32)


def clip_normalized_obs_inplace(
    obs: np.ndarray, scratch: np.ndarray, norm_mean: np.ndarray, norm_std_eps: np.ndarray, clip: float = NORM_CLIP
) -> None:
    """In-place version of :func:`clip_normalized_obs`.

    Uses *scratch* buffer (same shape as obs) to avoid allocation.
    Result is written back into *obs*.
    """
    np.subtract(obs, norm_mean, out=scratch)
    np.divide(scratch, norm_std_eps, out=scratch)
    np.clip(scratch, -clip, clip, out=scratch)
    np.multiply(scratch, norm_std_eps, out=obs)
    np.add(obs, norm_mean, out=obs)
