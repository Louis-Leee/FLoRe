"""Quantile (q01/q99) normalization statistics — GCL convention.

Mirrors GCL ``data/transforms.py`` ``normalize``/``unnormalize`` (use_quantiles path)
and ``data_types.py`` ``NormStats``::

    normalize:   (x - q01) / (max(q99 - q01, 0.02) + 1e-6) * 2 - 1     -> ~[-1, 1]
    unnormalize: (x + 1) / 2 * (max(q99 - q01, 0.02) + 1e-6) + q01

Stats are per-key (e.g. ``"actions"``, ``"state"``), per-dimension. Saved/loaded as
JSON so the trainer can bake them into a checkpoint alongside ``model.safetensors``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_QUANTILE_MIN_RANGE = 0.02
_EPS = 1e-6


@dataclass
class NormStats:
    """Per-dimension normalization statistics for one key."""

    q01: np.ndarray
    q99: np.ndarray
    mean: np.ndarray | None = None
    std: np.ndarray | None = None


def normalize_quantile(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Map to ~[-1, 1] using q01/q99 (GCL convention). Extra trailing dims pass through."""
    q01, q99 = stats.q01, stats.q99
    d = q01.shape[-1]
    rng = np.maximum(q99 - q01, _QUANTILE_MIN_RANGE)
    if x.shape[-1] > d:
        head = (x[..., :d] - q01) / (rng + _EPS) * 2.0 - 1.0
        return np.concatenate([head, x[..., d:]], axis=-1)
    return (x - q01) / (rng + _EPS) * 2.0 - 1.0


def unnormalize_quantile(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Inverse of :func:`normalize_quantile`. Extra trailing dims pass through."""
    q01, q99 = stats.q01, stats.q99
    d = q01.shape[-1]
    rng = np.maximum(q99 - q01, _QUANTILE_MIN_RANGE)
    if x.shape[-1] > d:
        head = (x[..., :d] + 1.0) / 2.0 * (rng + _EPS) + q01
        return np.concatenate([head, x[..., d:]], axis=-1)
    return (x + 1.0) / 2.0 * (rng + _EPS) + q01


def compute_quantile_stats(samples: dict[str, np.ndarray]) -> dict[str, NormStats]:
    """Compute per-key q01/q99 (+ mean/std) from stacked sample arrays.

    Parameters
    ----------
    samples : dict[str, np.ndarray]
        ``{key: array}`` where the last axis is the feature dim and all leading
        axes are flattened as samples (e.g. actions stacked as (N*H, action_dim)).
    """
    stats: dict[str, NormStats] = {}
    for key, arr in samples.items():
        flat = np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1])
        stats[key] = NormStats(
            q01=np.percentile(flat, 1, axis=0),
            q99=np.percentile(flat, 99, axis=0),
            mean=flat.mean(axis=0),
            std=flat.std(axis=0),
        )
    return stats


def save_norm_stats(
    stats: dict[str, NormStats], path: str | Path, meta: dict | None = None
) -> None:
    """Save stats to JSON (lists, so it is diff-friendly and language-agnostic)."""
    payload = {
        key: {
            "q01": s.q01.tolist(),
            "q99": s.q99.tolist(),
            "mean": None if s.mean is None else s.mean.tolist(),
            "std": None if s.std is None else s.std.tolist(),
        }
        for key, s in stats.items()
    }
    if meta is not None:
        payload[_META_KEY] = meta
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


#: Top-level JSON keys that are PROVENANCE, not per-key statistics. ``_meta`` carries
#: the layout mode a file was fitted for (see scripts/fit_hl_norm_stats.py): two modes
#: can be byte-identical in every width the trainer compares, so the width assertion
#: alone cannot tell them apart.
_META_KEY = "_meta"


def load_norm_stats_meta(path: str | Path) -> dict:
    """Provenance block from a stats file (``{}`` for files written before it existed)."""
    return parse_norm_stats_meta(Path(path).read_text())


def parse_norm_stats_meta(text: str) -> dict:
    """Parse provenance from an already-frozen normalization JSON payload."""
    return dict(json.loads(text).get(_META_KEY) or {})


def load_norm_stats(path: str | Path) -> dict[str, NormStats]:
    """Load stats saved by :func:`save_norm_stats`. Ignores the provenance block."""
    return parse_norm_stats(Path(path).read_text())


def parse_norm_stats(text: str) -> dict[str, NormStats]:
    """Parse statistics from an already-frozen normalization JSON payload."""
    raw = json.loads(text)
    out: dict[str, NormStats] = {}
    for key, vals in raw.items():
        if key == _META_KEY:
            continue
        out[key] = NormStats(
            q01=np.asarray(vals["q01"], dtype=np.float64),
            q99=np.asarray(vals["q99"], dtype=np.float64),
            mean=None if vals.get("mean") is None else np.asarray(vals["mean"], dtype=np.float64),
            std=None if vals.get("std") is None else np.asarray(vals["std"], dtype=np.float64),
        )
    return out
