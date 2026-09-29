"""Tests for the opt-in hand-frame F^cmd mode in the realtime (browser) viz.

Two concerns, mirroring the headless --F_on_hand_frame ablation:

1. NumPy parity: the rotation the browser's WbtForceCmdLocal applies in the
   hand-frame path —
       F_world = R(q_pelvis_world) @ F_pelvis
       F_hand  = R(q_wrist_world)^-1 @ F_world
   must equal what the policy was trained on (force_tracking.py rotates the WORLD
   F^cmd buffer by the inverse wrist quat) AND what the inference deploy path
   computes (holosoma_wbt.py: R_rel^-1 @ F_pelvis, R_rel = wrist-in-pelvis). It
   must also send F^cmd=0 -> 0 in every frame (the F=0 invariant).

2. Builder wiring: setup_builder(..., force_cmd_hand_frame=True) must inject
   `hand_frame: True` + `wrist_body_names` into the force_cmd_local component
   ONLY, and force_cmd_hand_frame=False (the default) must leave the components
   dict byte-identical to before (so every existing pelvis bundle is unaffected).

The parity test is pure NumPy and always runs. The builder test needs a local
force-preset (obs dim 588) checkpoint and is skipped if none is present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[5]
_LOG_ROOT = _REPO / "logs/WBT-Holosoma"
_MOTIONS = _REPO / "src/extensions/wbt_wrappers/wbt_training/motions"
_FORCE_OBS_DIM = 588  # 147 per frame x 4 history


# --------------------------------------------------------------------------- #
# 1. NumPy parity (always runs)                                               #
# --------------------------------------------------------------------------- #
def _qnorm(q):
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def _qinv(q):  # conjugate of a unit quat (wxyz)
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _qrot(q, v):  # forward R(q)@v — mirrors WbtObservations.ts wbtQuatRotate
    qn = _qnorm(q)
    qi = _qinv(qn)
    r = _qmul(_qmul(qn, np.array([0.0, *v])), qi)
    return r[1:]


def _qrot_inv(q, v):  # inverse R(q)^-1@v — mirrors wbtQuatRotateInverse
    qn = _qnorm(q)
    qi = _qinv(qn)
    r = _qmul(_qmul(qi, np.array([0.0, *v])), qn)
    return r[1:]


def _R(q):  # rotation matrix from wxyz quat (mirrors matrix_from_quat)
    w, x, y, z = _qnorm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def test_browser_hand_frame_matches_training_and_inference():
    """Browser quat form == training world-matrix form == inference R_rel form."""
    rng = np.random.default_rng(12345)
    e_world = e_rel = e_zero = 0.0
    for _ in range(5000):
        q_p = _qnorm(rng.standard_normal(4))
        q_w = _qnorm(rng.standard_normal(4))
        f_p = rng.standard_normal(3) * 25.0

        # Browser path: world-rotate then inverse-wrist-rotate (quat ops).
        f_hand_browser = _qrot_inv(q_w, _qrot(q_p, f_p))

        # Training path: R(q_wrist_world)^-1 @ R(q_pelvis_world) @ F_pelvis.
        f_hand_train = _R(q_w).T @ (_R(q_p) @ f_p)
        e_world = max(e_world, float(np.max(np.abs(f_hand_browser - f_hand_train))))

        # Inference path (holosoma_wbt.py): R_rel = R_p^-1 @ R_w; out = R_rel.T @ F_p.
        r_rel = _R(q_p).T @ _R(q_w)
        f_hand_infer = r_rel.T @ f_p
        e_rel = max(e_rel, float(np.max(np.abs(f_hand_browser - f_hand_infer))))

        # F=0 invariant.
        e_zero = max(e_zero, float(np.max(np.abs(_qrot_inv(q_w, _qrot(q_p, np.zeros(3)))))))

    assert e_world < 1e-10, f"browser vs training mismatch: {e_world}"
    assert e_rel < 1e-10, f"browser vs inference mismatch: {e_rel}"
    assert e_zero == 0.0, f"F=0 invariant violated: {e_zero}"


# --------------------------------------------------------------------------- #
# 2. Builder wiring (needs a local force checkpoint)                          #
# --------------------------------------------------------------------------- #
def _find_force_checkpoint() -> Path | None:
    if not _LOG_ROOT.is_dir():
        return None
    onnxruntime = pytest.importorskip("onnxruntime")
    for onnx_path in sorted(_LOG_ROOT.glob("*/model_*.onnx")):
        try:
            sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        except Exception:
            continue
        obs_input = next((i for i in sess.get_inputs() if i.name == "obs"), None)
        if obs_input is not None and obs_input.shape[-1] == _FORCE_OBS_DIM:
            return onnx_path
    return None


def _force_cmd_component(policy_config_path: Path) -> dict:
    cfg = json.loads(policy_config_path.read_text())
    obs_group = next(iter(cfg["observations"].values()))
    comps = obs_group["components"]
    return next(c for c in comps if c["name"] == "WbtForceCmdLocal")


def _build(tmp_path: Path, model: Path, *, hand_frame: bool) -> Path:
    from wbt_realtime_eval.main import setup_builder

    out = tmp_path / ("hf" if hand_frame else "pelvis")
    setup_builder(
        model_path=model,
        motion_dir=_MOTIONS,
        output_dir=out,
        preset="g1-29dof-holosoma-2pt-force-wbt",
        force_cmd_hand_frame=hand_frame,
    )
    return out.parent / f"{out.name}_policy_config.json"


@pytest.mark.skipif(not _MOTIONS.is_dir(), reason="local motions dir absent")
def test_flag_injects_hand_frame_into_force_cmd_only(tmp_path, monkeypatch):
    model = _find_force_checkpoint()
    if model is None:
        pytest.skip("no local force-preset (588D) checkpoint")
    monkeypatch.setenv("MJSWAN_NO_LAUNCH", "1")

    cfg_off = _build(tmp_path, model, hand_frame=False)
    cfg_on = _build(tmp_path, model, hand_frame=True)

    off = json.loads(cfg_off.read_text())["observations"]
    on = json.loads(cfg_on.read_text())["observations"]
    off_comps = next(iter(off.values()))["components"]
    on_comps = next(iter(on.values()))["components"]

    # ON: force_cmd_local carries hand_frame + wrist_body_names.
    fc_on = next(c for c in on_comps if c["name"] == "WbtForceCmdLocal")
    assert fc_on.get("hand_frame") is True
    assert fc_on.get("wrist_body_names") == ["left_wrist_yaw_link", "right_wrist_yaw_link"]

    # OFF: force_cmd_local has NO hand-frame fields (pelvis default).
    fc_off = next(c for c in off_comps if c["name"] == "WbtForceCmdLocal")
    assert "hand_frame" not in fc_off
    assert "wrist_body_names" not in fc_off

    # Every OTHER component is byte-identical between the two builds (the flag
    # touches force_cmd_local ONLY -> existing pelvis bundles are unaffected).
    def _strip_force(comps):
        return [c for c in comps if c["name"] != "WbtForceCmdLocal"]

    assert _strip_force(off_comps) == _strip_force(on_comps)
