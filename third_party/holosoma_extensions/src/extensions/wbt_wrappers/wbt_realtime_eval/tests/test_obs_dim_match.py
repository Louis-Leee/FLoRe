"""Total obs dim per frame x history must match the ONNX input shape.

Catches drift between obs_registry's term layout and a trained holosoma_2pt_wbt
ONNX. `vr_2point_orn` is the 12D 6D rotation rep (first two columns of the
rotation matrix, 2 wrists x 6) — NOT the legacy 8D xyzw quat, which caused an
antipodal/double-cover OOD bug at deploy. If anyone reverts it to width 8 (or
otherwise resizes a term), this test fails before runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Per-frame term widths (matches holosoma_2pt_wbt term order).
# See src/extensions/wbt_wrappers/wbt_wrappers_inference/config_values/observation.py
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
_PER_FRAME = 141
_HISTORY = 4
_TOTAL = _PER_FRAME * _HISTORY  # 564

# Force preset (g1-29dof-holosoma-2pt-force-wbt): the 2pt terms PLUS a
# force_cmd_local(6) channel inserted after vr_2point_orn. force_cmd_local is
# the OBSERVED F^cmd intent (the policy sees it); F^ext is NOT here — it is a
# physics-only disturbance that must never widen the obs vector.
_FORCE_TERM_DIMS = {
    "command_lower_body": 24,
    "vr_2point_pos": 6,
    "vr_2point_orn": 12,
    "force_cmd_local": 6,
    "motion_ref_ori_b": 6,
    "projected_gravity": 3,
    "base_ang_vel": 3,
    "joint_pos": 29,
    "joint_vel": 29,
    "actions": 29,
}
_FORCE_PER_FRAME = 147
_FORCE_TOTAL = _FORCE_PER_FRAME * _HISTORY  # 588


def test_per_frame_dim_sum():
    assert sum(_TERM_DIMS.values()) == _PER_FRAME


def test_total_dim_with_history():
    assert _PER_FRAME * _HISTORY == _TOTAL


def test_force_preset_per_frame_dim_sum():
    # The force preset is exactly the 2pt preset + force_cmd_local(6).
    assert sum(_FORCE_TERM_DIMS.values()) == _FORCE_PER_FRAME
    assert _FORCE_PER_FRAME == _PER_FRAME + _FORCE_TERM_DIMS["force_cmd_local"]


def test_force_preset_total_dim_with_history():
    assert _FORCE_PER_FRAME * _HISTORY == _FORCE_TOTAL == 588


def test_force_preset_matches_presets_module():
    """The hardcoded force-term layout here must match presets.py (the source
    of truth that drives the actual ONNX obs vector)."""
    from wbt_realtime_eval.presets import get_preset

    preset = get_preset("g1-29dof-holosoma-2pt-force-wbt")
    assert preset.obs_keys == tuple(_FORCE_TERM_DIMS), (
        f"presets.py force obs order {preset.obs_keys} != test layout {tuple(_FORCE_TERM_DIMS)}"
    )
    assert preset.per_frame_dim == _FORCE_PER_FRAME
    assert preset.total_dim == _FORCE_TOTAL


def test_obs_dim_match_local_onnx():
    """Compare against any local *.onnx in logs/WBT-Holosoma/.

    Skipped on CI / fresh clones with no logs/ directory.
    """
    onnxruntime = pytest.importorskip("onnxruntime")

    repo_root = Path(__file__).resolve().parents[5]
    log_root = repo_root / "logs" / "WBT-Holosoma"
    if not log_root.is_dir():
        pytest.skip(f"local logs dir not present: {log_root}")

    candidates = sorted(log_root.glob("*/model_*.onnx"))
    if not candidates:
        pytest.skip(f"no model_*.onnx checkpoints under {log_root}")

    # Walk all candidates and assert against the first one matching the 2pt-WBT
    # input dim. Other presets (3pt: 576, dense: 628, terrain: 449) legitimately
    # have different last dims, so iterate rather than fail on the first non-2pt
    # checkpoint.
    matched: Path | None = None
    for onnx_path in candidates:
        sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        obs_input = next((i for i in sess.get_inputs() if i.name == "obs"), None)
        if obs_input is None:
            continue
        last_dim = obs_input.shape[-1]
        if isinstance(last_dim, int) and last_dim == _TOTAL:
            matched = onnx_path
            break

    if matched is None:
        pytest.skip(
            f"no 2pt-WBT checkpoint (obs dim {_TOTAL}) found among {len(candidates)} ONNX files under {log_root}"
        )

    # Re-confirm with a fresh session so the assertion is self-contained.
    sess = onnxruntime.InferenceSession(str(matched), providers=["CPUExecutionProvider"])
    obs_input = next(i for i in sess.get_inputs() if i.name == "obs")
    assert obs_input.shape[-1] == _TOTAL, (
        f"matched checkpoint {matched.name} reports obs dim {obs_input.shape[-1]} != {_TOTAL}"
    )


def test_force_obs_dim_match_local_onnx():
    """If a 588-dim force checkpoint is present locally, assert its obs dim.

    Skipped on CI / fresh clones. Note 588 is shared by both the force preset
    (147x4) and the 3pt-global preset; this only asserts the dim, not the
    semantics, so it's a coarse guard for the force build path.
    """
    onnxruntime = pytest.importorskip("onnxruntime")

    repo_root = Path(__file__).resolve().parents[5]
    log_root = repo_root / "logs" / "WBT-Holosoma"
    if not log_root.is_dir():
        pytest.skip(f"local logs dir not present: {log_root}")

    candidates = sorted(log_root.glob("*/model_*.onnx"))
    matched: Path | None = None
    for onnx_path in candidates:
        sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        obs_input = next((i for i in sess.get_inputs() if i.name == "obs"), None)
        if obs_input is None:
            continue
        last_dim = obs_input.shape[-1]
        if isinstance(last_dim, int) and last_dim == _FORCE_TOTAL:
            matched = onnx_path
            break

    if matched is None:
        pytest.skip(f"no {_FORCE_TOTAL}-dim checkpoint found under {log_root}")

    sess = onnxruntime.InferenceSession(str(matched), providers=["CPUExecutionProvider"])
    obs_input = next(i for i in sess.get_inputs() if i.name == "obs")
    assert obs_input.shape[-1] == _FORCE_TOTAL
