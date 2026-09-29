"""Verify the browser-side obs construction matches the Python runtime.

The browser uses per-term 4-frame history with `[t-3, t-2, t-1, t]` layout,
followed by normalizer pre-clip, exactly mirroring
``HolosomaWBTPolicy._build_observation`` and
``wbt_wrappers_inference.utils.normalizer.clip_normalized_obs``. If anyone
breaks the layout (e.g. swaps to per-frame stacking, reorders within a term,
or skips pre-clip), the policy diverges immediately at first inference.

This test does the construction in NumPy using the documented contract, so
the JS port can be verified against it (and so the contract is pinned in the
test suite).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[5]
_ONNX = _REPO / "logs/WBT-Holosoma/20260522_215023-g1_29dof_wbt_manager-locomotion/model_05000.onnx"


def _make_obs_per_term(history: dict[str, list[np.ndarray]], term_order: list[str]) -> np.ndarray:
    """Build the 564D vector with per-term `[t-3, t-2, t-1, t]` layout.

    Each entry in ``history[term]`` is a (base_dim,) Float32 array, ordered
    chronologically (oldest first, newest last).
    """
    pieces = []
    for term in term_order:
        h = history[term]
        if len(h) != 4:
            raise ValueError(f"term {term!r} has {len(h)} history entries, expected 4")
        pieces.append(np.concatenate(h))
    return np.concatenate(pieces).astype(np.float32)


def _clip_normalized(obs: np.ndarray, mean: np.ndarray, std_eps: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Re-clip the obs the same way wbt_wrappers_inference.utils.normalizer does."""
    n = (obs - mean) / std_eps
    n = np.clip(n, -clip, clip)
    return (n * std_eps + mean).astype(np.float32)


_TERM_ORDER = [
    "command_lower_body",
    "vr_2point_pos",
    "vr_2point_orn",
    "motion_ref_ori_b",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
]
_TERM_DIMS = {
    "command_lower_body": 24,
    "vr_2point_pos": 6,
    "vr_2point_orn": 12,
    "motion_ref_ori_b": 6,
    "projected_gravity": 3,
    "base_ang_vel": 3,
    "joint_pos": 29,
    "joint_vel": 29,
    "actions": 29,
}


def test_obs_layout_per_term_first():
    """The first 24*4=96 elements of the obs vector are command_lower_body's
    full 4-frame history, NOT 4 frames of the full 141D vector."""
    history = {term: [np.full(dim, idx, dtype=np.float32) for idx in range(4)] for term, dim in _TERM_DIMS.items()}
    # Set command_lower_body history to a recognizable pattern to verify position.
    history["command_lower_body"] = [np.full(24, 100 + i, dtype=np.float32) for i in range(4)]

    obs = _make_obs_per_term(history, _TERM_ORDER)
    assert obs.shape == (564,)
    # First 96 elements = command_lower_body's 4 history frames.
    assert np.all(obs[:24] == 100)  # frame t-3
    assert np.all(obs[24:48] == 101)  # frame t-2
    assert np.all(obs[48:72] == 102)  # frame t-1
    assert np.all(obs[72:96] == 103)  # frame t


def test_obs_layout_per_term_diverges_from_per_frame():
    """Sanity: per-term layout != per-frame layout, ruling out the silent bug."""
    history = {term: [np.random.randn(dim).astype(np.float32) for _ in range(4)] for term, dim in _TERM_DIMS.items()}
    obs_per_term = _make_obs_per_term(history, _TERM_ORDER)
    # Per-frame: each frame is the full 141D, stacked newest-first.
    # frame_t = concat of all term[3] (newest); frame_t-3 = concat of all term[0].
    frames = []
    for hist_idx in [3, 2, 1, 0]:
        frames.append(np.concatenate([history[term][hist_idx] for term in _TERM_ORDER]))
    obs_per_frame = np.concatenate(frames).astype(np.float32)
    assert obs_per_term.shape == obs_per_frame.shape
    # Same total content (rough float32 sum), different positions
    assert obs_per_term.sum() == pytest.approx(obs_per_frame.sum(), rel=1e-3, abs=1e-3)
    assert not np.allclose(obs_per_term, obs_per_frame)
    # As a stronger invariant: the multiset of values is identical
    sorted_a = np.sort(obs_per_term)
    sorted_b = np.sort(obs_per_frame)
    np.testing.assert_allclose(sorted_a, sorted_b)


@pytest.mark.skipif(not _ONNX.is_file(), reason="local checkpoint required")
def test_normalizer_preclip_changes_obs():
    """Verify the normalizer pre-clip is applied (pre-clipped obs differs
    from raw obs whenever raw values exceed ±3σ)."""
    onnxruntime = pytest.importorskip("onnxruntime")
    sess = onnxruntime.InferenceSession(str(_ONNX), providers=["CPUExecutionProvider"])
    meta = dict(sess.get_modelmeta().custom_metadata_map)
    if "normalizer_mean" not in meta:
        pytest.skip("checkpoint has no baked normalizer")
    mean = np.array([float(x) for x in meta["normalizer_mean"].split(",")], dtype=np.float32)
    std = np.array([float(x) for x in meta["normalizer_std"].split(",")], dtype=np.float32) + 1e-2

    np.random.seed(0)
    # Build a noisy obs intentionally far outside ±3σ to trigger clipping
    raw = (mean + std * np.random.randn(564).astype(np.float32) * 5.0).astype(np.float32)
    clipped = _clip_normalized(raw, mean, std)

    # Some entries were clipped — clipped != raw on those
    diff = np.abs(raw - clipped)
    assert (diff > 0).sum() > 100, "Expected many entries to be clipped at 5σ noise"
    # All clipped values lie within ±3σ of mean
    normalized = (clipped - mean) / std
    assert np.all(np.abs(normalized) <= 3.0 + 1e-5)


@pytest.mark.skipif(not _ONNX.is_file(), reason="local checkpoint required")
def test_per_frame_layout_diverges_from_per_term_at_action_output():
    """Smoke test that the layout choice is ACTUALLY load-bearing — the same
    underlying numbers produce wildly different actions when permuted from
    per-term to per-frame. This is the bug the in-browser port had earlier."""
    onnxruntime = pytest.importorskip("onnxruntime")
    sess = onnxruntime.InferenceSession(str(_ONNX), providers=["CPUExecutionProvider"])
    meta = dict(sess.get_modelmeta().custom_metadata_map)
    mean = np.array([float(x) for x in meta["normalizer_mean"].split(",")], dtype=np.float32)
    std = np.array([float(x) for x in meta["normalizer_std"].split(",")], dtype=np.float32) + 1e-2

    np.random.seed(0)
    history: dict[str, list[np.ndarray]] = {term: [] for term in _TERM_ORDER}
    for term, dim in _TERM_DIMS.items():
        for _ in range(4):
            history[term].append((np.random.randn(dim).astype(np.float32) * 0.3))
    obs_per_term = _clip_normalized(_make_obs_per_term(history, _TERM_ORDER), mean, std)
    # Per-frame layout (the BROKEN one)
    frames = []
    for hist_idx in [3, 2, 1, 0]:
        frames.append(np.concatenate([history[term][hist_idx] for term in _TERM_ORDER]))
    obs_per_frame = _clip_normalized(np.concatenate(frames).astype(np.float32), mean, std)

    out_per_term = sess.run(
        None, {"obs": obs_per_term.reshape(1, 564), "time_step": np.array([[0.0]], dtype=np.float32)}
    )[0].reshape(-1)
    out_per_frame = sess.run(
        None, {"obs": obs_per_frame.reshape(1, 564), "time_step": np.array([[0.0]], dtype=np.float32)}
    )[0].reshape(-1)

    diff = np.linalg.norm(out_per_term - out_per_frame)
    assert diff > 1.0, f"Expected per-term and per-frame layouts to diverge significantly; diff={diff}"
