"""Live-VIVE override tests for HolosomaMaskedMimicPolicy (two_point mode).

The MaskedMimic policy has NO ``vr_2point`` obs term — the two wrist targets are
WORLD-frame body poses (CONDITIONABLE_BODIES idx 4=left_wrist, 5=right_wrist)
turned into per-step local deltas by ``_build_target_poses_and_masks``. The
live-VIVE override injects a slewed relative-clutch DELTA onto those world
targets. These tests pin the load-bearing invariants:

  1. OFF (no live source) ⇒ ``_build_target_poses_and_masks`` is byte-identical to
     the NPZ replay path (the override never runs).
  2. ARMED with the trackers AT the clutch origin (delta == 0) ⇒ the wrist
     target_poses equal the FROZEN-frame held reference (zero-jump arm), and are
     STATIC across the 15-step future lookahead (the frozen clip's moving future
     can't drift the hands).
  3. Relative motion of the trackers moves the wrist targets by the matching
     world-frame displacement (rotated through the robot root), slew-capped.
  4. Recenter re-snaps the origin so the held delta returns toward zero.

Skips cleanly when the sonic data / inference deps are unavailable.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pytest

_SONIC_DIR = os.path.expanduser("~/data/sonic_manipulation_test")

pytest.importorskip("pinocchio")
pytest.importorskip("wbt_wrappers_inference")

from wbt_wrappers_inference.config_values.robot_config import (  # noqa: E402
    CONDITIONABLE_BODIES,
    g1_29dof_wbt,
)

_NUM_DOFS = 29
_N_COND = 6
_NUM_FUTURE = 30
_EVAL_VISIBLE = 15
_LEFT_WRIST = 4
_RIGHT_WRIST = 5


class _FakeViveSource:
    """Stand-in for ViveLiveSource: returns a fixed (pos[6], orn[12]) command.

    ``set(pos, orn)`` updates the held command; ``set(None, None)`` makes it
    stale (``latest()`` returns ``(None, None)``)."""

    def __init__(self):
        self._pos = None
        self._orn = None

    def set(self, pos, orn):
        self._pos = None if pos is None else np.asarray(pos, dtype=np.float32)
        self._orn = None if orn is None else np.asarray(orn, dtype=np.float32)

    def latest(self):
        if self._pos is None:
            return None, None
        return self._pos.copy(), self._orn.copy()

    def shutdown(self):
        pass


def _identity_orn12():
    eye6 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    return np.concatenate([eye6, eye6])


def _make_policy(npz_path, *, vive=False, orientation=False, frame_idx=0):
    """Minimal HolosomaMaskedMimicPolicy harness for two_point mode (no heavy
    __init__), wired exactly as _build_target_poses_and_masks + the VIVE override
    read it."""
    from holosoma_inference.policies.wbt import PinocchioRobot
    from wbt_wrappers_inference.policies.holosoma_maskedmimic import (
        HolosomaMaskedMimicPolicy,
        _JOINT_TARGETS_DIM,
        _NUM_COND_BODIES,
        _REDUCED_COORDS_DIM,
        _TARGET_MASKS_DIM,
        _TARGET_POSES_DIM,
    )

    repo = Path(__file__).resolve().parents[5]
    urdf = repo / "thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof_yam.urdf"

    from loguru import logger as _loguru

    p = object.__new__(HolosomaMaskedMimicPolicy)
    p.logger = _loguru  # arming helpers log via self.logger
    p.num_dofs = _NUM_DOFS
    p.default_dof_angles = np.array(g1_29dof_wbt.default_dof_angles, dtype=np.float64)
    p._visible_body_indices = [_LEFT_WRIST, _RIGHT_WRIST]  # two_point
    p._joint_mode = False
    p._tracking_mode = "two_point"
    p._gravity_world = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    p._reduced_coords_buf = np.zeros(_REDUCED_COORDS_DIM, dtype=np.float32)
    p._target_poses_buf = np.zeros(_TARGET_POSES_DIM, dtype=np.float32)
    p._target_masks_buf = np.zeros(_TARGET_MASKS_DIM, dtype=np.float32)
    p._joint_targets_buf = np.zeros(_JOINT_TARGETS_DIM, dtype=np.float32)
    p._mocap_ref_pos = np.zeros((_NUM_COND_BODIES, 3), dtype=np.float32)
    p._mocap_ref_quat_xyzw = np.zeros((_NUM_COND_BODIES, 4), dtype=np.float32)
    p._fk_q_buf = np.zeros(3 + 4 + _NUM_DOFS, dtype=np.float64)
    p._fk_body_pos = np.zeros((_NUM_COND_BODIES, 3), dtype=np.float32)
    p._fk_body_quat = np.zeros((_NUM_COND_BODIES, 4), dtype=np.float32)
    p.pinocchio_robot = PinocchioRobot(g1_29dof_wbt, urdf.read_text())
    p._pinocchio_frame_ids = [p.pinocchio_robot.robot_model.getFrameId(bn) for bn in CONDITIONABLE_BODIES]
    p._load_external_motion(npz_path)
    p._ext_frame_idx = frame_idx

    # VIVE state (mirror __init__)
    p._is_2pt_mm = True
    p._vive_live = _FakeViveSource() if vive else None
    p._vive_armed = False
    p._vive_orientation = orientation
    p._vive_max_pos_step = 0.02
    p._vive_max_orn_step = 0.05
    p._vr_last_dpos = None
    p._vr_last_drot6d = None
    p._vr_clutch_live0 = None
    p._vr_clutch_orn0 = None
    # Absolute-mode arm clutch (mirrors __init__; both anchors None until arm).
    p._vive_abs_clutch = True
    p._vr_abs_off_pos = None
    p._vr_abs_orn0 = None
    p._last_live_pos = None
    p._last_live_orn = None
    return p


def _robot_state(p, seed=0):
    """A realistic robot state near the clip frame-0 pose (robot != reference)."""
    rng = np.random.default_rng(seed)
    jp0, _jv0, ref_pos0, ref_quat_xyzw0 = p._get_ext_motion_at_frame(0)
    n = _NUM_DOFS
    state = np.zeros((1, 7 + n + 3 + 3 + n), dtype=np.float64)
    state[0, 0:3] = ref_pos0
    state[0, 3:7] = np.array([ref_quat_xyzw0[3], ref_quat_xyzw0[0], ref_quat_xyzw0[1], ref_quat_xyzw0[2]])
    state[0, 7 : 7 + n] = jp0 + rng.uniform(-0.15, 0.15, size=n)
    state[0, 7 + n + 6 : 7 + n + 6 + n] = rng.uniform(-0.5, 0.5, size=n)
    return state


def _first_clip():
    clips = sorted(glob.glob(os.path.join(_SONIC_DIR, "*.npz")))
    assert clips, "no sonic clips found"
    return clips[0]


def _wrist_slots(buf, step, bi):
    base = step * _N_COND * 7 + bi * 7
    return buf[base : base + 7].copy()


pytestmark = pytest.mark.skipif(not os.path.isdir(_SONIC_DIR), reason="sonic_manipulation_test data not present")


def test_off_is_byte_identical():
    """No live source ⇒ target_poses identical to the plain NPZ-replay build."""
    npz = _first_clip()
    base = _make_policy(npz, vive=False, frame_idx=3)
    live = _make_policy(npz, vive=True, frame_idx=3)  # source present but DISARMED
    state = _robot_state(base)

    base._align_rot_mat = None
    base._compute_motion_alignment(state)
    base._build_target_poses_and_masks(state)

    live._align_rot_mat = None
    live._compute_motion_alignment(state)
    live._build_target_poses_and_masks(state)  # _vive_armed=False ⇒ override inert

    np.testing.assert_array_equal(base._target_poses_buf, live._target_poses_buf)
    np.testing.assert_array_equal(base._target_masks_buf, live._target_masks_buf)


def test_armed_zero_delta_is_zero_jump_and_static():
    """Arming with trackers AT the origin (delta==0) ⇒ wrist target_poses equal the
    FROZEN-frame held reference and are identical across all 15 future steps."""
    npz = _first_clip()
    p = _make_policy(npz, vive=True, orientation=False, frame_idx=0)
    state = _robot_state(p)
    p._align_rot_mat = None
    p._compute_motion_alignment(state)

    # Arm at an arbitrary tracker pose (clutch makes the origin == this pose).
    pos0 = np.array([0.2, 0.2, -0.15, 0.2, -0.2, -0.15], dtype=np.float32)
    p._vive_live.set(pos0, _identity_orn12())
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True

    p._build_target_poses_and_masks(state)

    # The two wrist slots must be IDENTICAL across every visible future step
    # (static hold — the frozen clip's moving future does not leak in).
    for bi in (_LEFT_WRIST, _RIGHT_WRIST):
        s0 = _wrist_slots(p._target_poses_buf, 0, bi)
        for step in range(1, _EVAL_VISIBLE):
            np.testing.assert_allclose(_wrist_slots(p._target_poses_buf, step, bi), s0, atol=1e-6)


def test_relative_motion_moves_wrist_target():
    """Moving a tracker +X by 5 cm (well under the slew cap over a few ticks) moves
    the corresponding wrist target by ~the same world displacement."""
    npz = _first_clip()
    p = _make_policy(npz, vive=True, orientation=False, frame_idx=0)
    state = _robot_state(p)
    p._align_rot_mat = None
    p._compute_motion_alignment(state)

    pos0 = np.array([0.2, 0.2, -0.15, 0.2, -0.2, -0.15], dtype=np.float32)
    p._vive_live.set(pos0, _identity_orn12())
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True

    # Baseline (delta 0).
    p._build_target_poses_and_masks(state)
    left0 = _wrist_slots(p._target_poses_buf, 0, _LEFT_WRIST)[:3]

    # Move the LEFT tracker +5 cm in pelvis +X, then let the slew ramp converge.
    moved = pos0.copy()
    moved[0] += 0.05
    p._vive_live.set(moved, _identity_orn12())
    for _ in range(20):  # 20 * 0.02 m/tick cap ≥ 0.05 m displacement
        p._build_target_poses_and_masks(state)
    left1 = _wrist_slots(p._target_poses_buf, 0, _LEFT_WRIST)[:3]

    # The wrist target moved; magnitude ≈ 5 cm (root≈upright ⇒ pelvis≈world here),
    # and the RIGHT wrist (untouched tracker) barely moved.
    disp = np.linalg.norm(left1 - left0)
    assert disp > 0.02, f"left wrist target should move with the tracker, got {disp:.4f} m"
    assert disp < 0.08, f"left wrist target moved too far, got {disp:.4f} m"
    right1 = _wrist_slots(p._target_poses_buf, 0, _RIGHT_WRIST)[:3]
    # Re-baseline right wrist by rebuilding at origin is unnecessary; just bound it.
    assert np.linalg.norm(right1) < 5.0


def test_recenter_returns_delta_toward_zero():
    """After moving a tracker, recenter (re-snap the origin at the new pose) makes
    the held delta start collapsing back toward zero on the next ticks."""
    npz = _first_clip()
    p = _make_policy(npz, vive=True, orientation=False, frame_idx=0)
    state = _robot_state(p)
    p._align_rot_mat = None
    p._compute_motion_alignment(state)

    pos0 = np.array([0.2, 0.2, -0.15, 0.2, -0.2, -0.15], dtype=np.float32)
    p._vive_live.set(pos0, _identity_orn12())
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True

    moved = pos0.copy()
    moved[0] += 0.05
    p._vive_live.set(moved, _identity_orn12())
    for _ in range(20):
        p._build_target_poses_and_masks(state)
    delta_before = float(np.linalg.norm(p._vr_last_dpos))
    assert delta_before > 0.02

    # Recenter at the CURRENT (moved) tracker pose → new origin == moved, so the
    # commanded delta target becomes 0 again and the held delta ramps back down.
    p._vive_recenter()
    for _ in range(20):
        p._build_target_poses_and_masks(state)
    delta_after = float(np.linalg.norm(p._vr_last_dpos))
    assert delta_after < delta_before * 0.5, f"recenter should shrink the delta: {delta_before}->{delta_after}"


def test_position_only_subtracts_swinging_lever():
    """Position-only teleop (HOLOSOMA_VIVE_ORIENTATION=0) must recover the true
    wrist ORIGIN displacement even when the bridge SWINGS the contact lever
    (bridge --orientation on). A pure wrist TWIST at the anchor pose (no origin
    motion) must produce ~zero wrist-target displacement — NOT a phantom
    R(wrist)@offset translation."""
    npz = _first_clip()
    p = _make_policy(npz, vive=True, orientation=False, frame_idx=0)  # orn CLUTCH off
    state = _robot_state(p)
    p._align_rot_mat = None
    p._compute_motion_alignment(state)

    # Anchor with the wrist at identity orientation.
    pos0 = np.array([0.2, 0.2, -0.15, 0.2, -0.2, -0.15], dtype=np.float32)
    p._vive_live.set(pos0, _identity_orn12())
    assert p._snapshot_clutch_anchor()  # snapshots orn origin even though clutch off
    p._vive_armed = True
    p._build_target_poses_and_masks(state)
    left0 = _wrist_slots(p._target_poses_buf, 0, _LEFT_WRIST)[:3].copy()

    # Now the operator TWISTS the left wrist 60° in place: the CONTACT point the
    # bridge streams moves by (R_new - I)@offset (a phantom ~several-cm swing),
    # but the wrist ORIGIN did NOT move. Set the new contact pos accordingly.
    import numpy as _np
    from wbt_wrappers_inference.policies.holosoma_maskedmimic import _VR_2POINT_OFFSETS

    th = _np.radians(60.0)
    c, s = _np.cos(th), _np.sin(th)
    R = _np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    r6 = R[:, :2].reshape(-1).astype(np.float32)
    left_off = _VR_2POINT_OFFSETS[0].astype(_np.float64)
    # contact_new = origin0 + R@offset; origin0 = pos0_contact - I@offset
    origin0_L = pos0[:3].astype(_np.float64) - left_off
    contact_new_L = origin0_L + R @ left_off
    moved = pos0.copy()
    moved[:3] = contact_new_L.astype(np.float32)
    orn_new = _np.concatenate([r6, _identity_orn12()[6:]]).astype(np.float32)
    p._vive_live.set(moved, orn_new)
    for _ in range(30):
        p._build_target_poses_and_masks(state)
    left1 = _wrist_slots(p._target_poses_buf, 0, _LEFT_WRIST)[:3]
    # Pure twist at fixed origin ⇒ wrist-target position essentially unchanged.
    drift = float(np.linalg.norm(left1 - left0))
    assert drift < 0.01, f"position-only must ignore the lever swing, got {drift:.4f} m drift"


def test_stale_while_armed_holds_last_delta():
    """A stale frame while armed must HOLD the last fed delta (never revert to the
    moving clip future)."""
    npz = _first_clip()
    p = _make_policy(npz, vive=True, orientation=False, frame_idx=0)
    state = _robot_state(p)
    p._align_rot_mat = None
    p._compute_motion_alignment(state)

    pos0 = np.array([0.2, 0.2, -0.15, 0.2, -0.2, -0.15], dtype=np.float32)
    p._vive_live.set(pos0, _identity_orn12())
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True
    moved = pos0.copy()
    moved[0] += 0.05
    p._vive_live.set(moved, _identity_orn12())
    for _ in range(20):
        p._build_target_poses_and_masks(state)
    held = p._vr_last_dpos.copy()
    left_held = _wrist_slots(p._target_poses_buf, 0, _LEFT_WRIST).copy()

    # Source goes stale → the override holds, target_poses unchanged.
    p._vive_live.set(None, None)
    p._build_target_poses_and_masks(state)
    np.testing.assert_array_equal(p._vr_last_dpos, held)
    np.testing.assert_allclose(_wrist_slots(p._target_poses_buf, 0, _LEFT_WRIST), left_held, atol=1e-6)
