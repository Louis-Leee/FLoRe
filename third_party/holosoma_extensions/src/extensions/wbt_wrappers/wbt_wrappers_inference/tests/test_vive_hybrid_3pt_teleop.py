"""Live-VIVE teleop on the 3-POINT HYBRID position-force controller
(exp:g1-wbt-3pt-hybrid).

The 3pt-hybrid actor obs is 168D/frame × 4 history = 672D, and it carries
``vr_3point_pos`` (9D = L,R,head) / ``vr_3point_orn`` (12D = 3 bodies × QUATERNION
xyzw, NOT the 2pt rot6d) → ``detect_obs_mode`` returns ``"3pt"`` →
``HolosomaWBTPolicy._is_3pt`` is True. The live-VIVE override (originally built
2pt-only) was generalized to fire for the 3pt family too; force/servo stay
two-hand (the head has no F/T sensor).

These tests run the REAL ``_compute_obs_terms`` on the REAL 3pt-hybrid obs config
with a real Pinocchio yam model + a fake VIVE source, and pin the load-bearing
invariants for the 3pt-hybrid + VIVE combination:

  1. Assembled per-frame obs is exactly 168D in the trained hybrid term order.
  2. Arming VIVE and moving the trackers changes ONLY the ``vr_3point_pos`` slot
     — it never leaks into the force / servo slots, which stay zero when no live
     F/T source and servo are configured.
  3. OFF (no VIVE source) ⇒ the vr_3point slots are the plain NPZ-FK reference
     (byte-identical to the non-teleop path).
  4. Relative-clutch arming is zero-jump (trackers AT the anchor ⇒ vr_3point ==
     the held FK reference), and only relative tracker motion moves the points,
     slew-capped. The 9D position delta drives all three points (incl. head).
  5. A live head displacement moves the HEAD slot (slot index 2 of vr_3point_pos)
     — the genuinely new 3pt degree of freedom.

Skips cleanly when pinocchio or the yam URDF is unavailable.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("wbt_wrappers_inference")

_REPO = Path(__file__).resolve().parents[5]
_YAM_URDF = _REPO / "thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof_yam.urdf"

_HYBRID_3PT_PRESET = "g1-29dof-holosoma-3pt-hybrid-wbt"

# Per-frame (168D) slot offsets — see observation.holosoma_3pt_hybrid_wbt. Locked
# here so a silent term-order/width change trips the test rather than the robot.
# Delta vs 2pt-hybrid: vr_pos 6→9, vr_orn stays 12 (but quaternion), everything
# after shifts by +3.
_SLOTS = {
    "command_lower_body": (0, 24),
    "vr_3point_pos": (24, 33),
    "vr_3point_orn": (33, 45),
    "force_cmd_local": (45, 51),
    "force_meas_local": (51, 57),
    "integrator_state": (57, 63),
    "x_tgt_shift": (63, 69),
    "motion_ref_ori_b": (69, 75),
    "projected_gravity": (75, 78),
    "base_ang_vel": (78, 81),
    "joint_pos": (81, 110),
    "joint_vel": (110, 139),
    "actions": (139, 168),
}
_PER_FRAME = 168


def _identity_orn12_quat():
    """Three identity xyzw quaternions [0,0,0,1] — the 3pt vr_3point_orn rep."""
    return np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), 3)


class _FakeViveSource:
    """Stand-in for ViveLiveSource: returns a fixed (pos[9], orn[12]) 3pt command.

    ``set(pos, orn)`` updates the held command; ``set(None, None)`` makes it stale
    (``latest()`` returns ``(None, None)``)."""

    def __init__(self):
        self._pos = None
        self._orn = None

    def set(self, pos, orn):
        self._pos = None if pos is None else np.asarray(pos, dtype=np.float32)
        self._orn = None if orn is None else np.asarray(orn, dtype=np.float32)

    def latest(self):
        if self._pos is None:
            return None, None
        # The real bridge ALWAYS sends orn alongside pos (identity quats when
        # position-only); the policy asserts orn is non-None when pos is.
        orn = _identity_orn12_quat() if self._orn is None else self._orn.copy()
        return self._pos.copy(), orn


def _stub_hybrid_3pt_policy(*, vive=False):
    """Minimal HolosomaWBTPolicy wired exactly as ``_compute_obs_terms`` reads it
    for the 3pt-hybrid preset: the REAL 3pt-hybrid obs config + a real Pinocchio
    yam model + the VIVE/force caches. No heavy __init__ (object.__new__)."""
    from loguru import logger as _loguru

    from holosoma_inference.policies.wbt_utils import PinocchioRobot

    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.config_values.robot_config import VR_3POINT_BODIES, g1_29dof_wbt
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    if not _YAM_URDF.is_file():
        pytest.skip(f"yam URDF not present: {_YAM_URDF}")

    obs_cfg = DEFAULTS[_HYBRID_3PT_PRESET].observation
    terms = list(obs_cfg.obs_dict["actor_obs"])
    hist = obs_cfg.history_length_dict.get("actor_obs", 4)

    p = object.__new__(HolosomaWBTPolicy)
    p.logger = _loguru
    p.num_dofs = 29
    p.config = SimpleNamespace(
        robot=g1_29dof_wbt, observation=obs_cfg, task=SimpleNamespace(rl_rate=50.0, headless_sim_eval=False)
    )
    p.default_dof_angles = np.array(g1_29dof_wbt.default_dof_angles, dtype=np.float64)
    p.pinocchio_robot = PinocchioRobot(g1_29dof_wbt, _YAM_URDF.read_text())

    # obs assembly state — 3pt this time.
    p._obs_terms = terms
    p._obs_mode = "3pt"
    p._is_2pt = False
    p._is_3pt = True
    p._vr_orn_6d = False  # plain 3pt trains the QUATERNION orn rep (setup_policy sets this)
    # Hand geometry, derived from the preset's terms as setup_policy does. False here (the
    # base 3pt-hybrid preset keeps `vr_3point_pos` / the 0.18 m VR lever); asserting it via
    # the same expression rather than a literal keeps this stub honest if the preset moves.
    p._vr_pos_tcp = any("vr_3point_pos_tcp" in t for t in terms)
    p._history_length = hist
    p._obs_history = {t: deque(maxlen=hist) for t in terms}
    p._last_action = np.zeros(29, dtype=np.float32)
    p._ref_body_is_pelvis = True
    p._heading_initialized = True
    p.robot_yaw_offset = 0.0
    p.motion_yaw_offset = 0.0
    p._align_rot_mat = np.eye(3)

    # VR FK frame ids (3pt bodies: L wrist, R wrist, torso).
    model = p.pinocchio_robot.robot_model
    p._vr_frame_ids = [model.getFrameId(b) for b in VR_3POINT_BODIES]

    ref_dof = p.default_dof_angles.astype(np.float32)
    p._motion_command_t = np.concatenate([ref_dof.reshape(1, -1), np.zeros((1, 29), dtype=np.float32)], axis=1)
    p._ref_pos_xyz_t = np.array([[0.0, 0.0, 0.78]], dtype=np.float32)
    p._ref_quat_xyzw_t = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    p._hold_dof_override = None

    # Force / servo caches (all zero → the four hybrid slots read zero).
    p._force_live = None
    p._force_meas_pelvis = np.zeros((2, 3), dtype=np.float64)
    p._servo_integ = np.zeros((2, 3), dtype=np.float64)
    p._servo_delta_x = np.zeros((2, 3), dtype=np.float64)
    p._servo_dq = np.zeros(29, dtype=np.float64)
    p._hybrid_servo = False
    p._force_cmd_pelvis_raw = None
    p._force_cmd_local_scaled = None
    p._force_cmd_hand_frame = False
    p._force_cmd_sensor_frame = False
    p._force_cmd_keys = None

    # VIVE state (relative/clutch, orientation off — position-only 3pt teleop).
    p._vive_live = _FakeViveSource() if vive else None
    p._vive_armed = False
    p._vive_relative = True
    p._vive_orientation = False
    p._vive_max_pos_step = 0.02
    p._vive_max_orn_step = 0.05
    p._vr_last_pos = None
    p._vr_last_orn = None
    p._vr_clutch_live0 = None
    p._vr_clutch_orn0 = None
    # Absolute-mode arm clutch (mirrors __init__; both anchors None until arm).
    p._vive_abs_clutch = True
    p._vr_abs_off_pos = None
    p._vr_abs_orn0 = None
    return p


def _robot_state(p):
    n = p.num_dofs
    row = np.zeros((1, 7 + n + 6 + n), dtype=np.float32)
    row[0, 0:3] = [0.0, 0.0, 0.78]
    row[0, 3:7] = [1.0, 0.0, 0.0, 0.0]  # wxyz identity
    row[0, 7 : 7 + n] = p.default_dof_angles
    return row


def _per_frame(obs, term_offsets, hist):
    """Extract the LATEST per-frame 168D vector from the assembled (1, 672) obs."""
    obs = obs.reshape(-1)
    out = np.zeros(_PER_FRAME, dtype=np.float32)
    cursor = 0
    for _name, (lo, hi) in term_offsets.items():
        w = hi - lo
        block = obs[cursor : cursor + w * hist]
        out[lo:hi] = block[(hist - 1) * w : hist * w]  # newest frame
        cursor += w * hist
    return out


def _slot(vec168, name):
    lo, hi = _SLOTS[name]
    return vec168[lo:hi]


def test_hybrid_3pt_obs_is_672d_and_3pt():
    """The 3pt-hybrid preset is 3pt (VIVE-eligible) and assembles to 672D."""
    p = _stub_hybrid_3pt_policy(vive=False)
    assert p._is_3pt, "3pt-hybrid must detect as 3pt so the VIVE override fires"
    obs = p._build_observation(_robot_state(p))
    assert obs.shape == (1, _PER_FRAME * p._history_length), obs.shape


def test_hybrid_3pt_orn_is_unit_quaternions():
    """vr_3point_orn is 3 unit xyzw quaternions (12D) at the frozen standing ref —
    NOT rot6d. Each 4-vector must be unit norm."""
    p = _stub_hybrid_3pt_policy(vive=False)
    frame = _per_frame(p._build_observation(_robot_state(p)), _SLOTS, p._history_length)
    orn = _slot(frame, "vr_3point_orn").reshape(3, 4)
    norms = np.linalg.norm(orn, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"3pt orn must be 3 unit quats, got norms {norms}"


def test_hybrid_3pt_force_slots_zero_without_live_source():
    """No live F/T + servo off ⇒ the four hybrid-only slots read exactly zero."""
    p = _stub_hybrid_3pt_policy(vive=False)
    frame = _per_frame(p._build_observation(_robot_state(p)), _SLOTS, p._history_length)
    for name in ("force_cmd_local", "force_meas_local", "integrator_state", "x_tgt_shift"):
        assert np.allclose(_slot(frame, name), 0.0), f"{name} must be zero with no live source"


def test_vive_3pt_moves_only_vr_slots_not_force_slots():
    """Arming VIVE + moving the trackers changes ONLY the vr_3point_pos slot —
    NEVER the force / servo slots (which have no head/tracker input)."""
    p = _stub_hybrid_3pt_policy(vive=True)
    rs = _robot_state(p)

    base = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)
    ref_vr_pos = _slot(base, "vr_3point_pos").copy()  # 9D [L,R,head]

    # Arm the clutch AT the current FK reference (delta 0), then push all 3 points.
    live0 = ref_vr_pos.copy()
    p._vive_live.set(live0, None)
    p._vr_clutch_live0 = live0.astype(np.float32).copy()
    p._vive_armed = True
    moved = live0.copy()
    moved[0] += 0.30  # left hand +x
    moved[3] += 0.30  # right hand +x
    moved[6] += 0.20  # head +x
    p._vive_live.set(moved, None)

    frame = None
    for _ in range(80):
        frame = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)

    vr_pos = _slot(frame, "vr_3point_pos")
    assert vr_pos[0] > ref_vr_pos[0] + 0.05, "left hand x should track the +x push"
    assert vr_pos[3] > ref_vr_pos[3] + 0.05, "right hand x should track the +x push"
    assert vr_pos[6] > ref_vr_pos[6] + 0.05, "HEAD x should track the +x push (new 3pt DOF)"
    for name in ("force_cmd_local", "force_meas_local", "integrator_state", "x_tgt_shift"):
        assert np.allclose(_slot(frame, name), 0.0), f"VIVE must not leak into {name}"


def test_vive_3pt_arm_is_zero_jump():
    """Relative-clutch arming with the trackers AT the anchor ⇒ the first armed
    vr_3point_pos equals the held FK reference (no jump), for all 3 points."""
    p = _stub_hybrid_3pt_policy(vive=True)
    rs = _robot_state(p)
    ref_vr_pos = _slot(_per_frame(p._build_observation(rs), _SLOTS, p._history_length), "vr_3point_pos").copy()

    live0 = ref_vr_pos.copy()
    p._vive_live.set(live0, None)
    p._vr_clutch_live0 = live0.astype(np.float32).copy()
    p._vive_armed = True
    armed = _slot(_per_frame(p._build_observation(rs), _SLOTS, p._history_length), "vr_3point_pos")
    assert np.allclose(armed, ref_vr_pos, atol=1e-5), "zero-delta 3pt arm must be zero-jump"


def test_vive_3pt_off_is_byte_identical_vr_slot():
    """No VIVE source ⇒ the vr_3point_pos slot is the plain NPZ-FK reference,
    independent of the (absent) live source — the byte-identical baseline."""
    p_off = _stub_hybrid_3pt_policy(vive=False)
    p_on = _stub_hybrid_3pt_policy(vive=True)  # source present but NOT armed
    rs = _robot_state(p_off)
    off = _slot(_per_frame(p_off._build_observation(rs), _SLOTS, p_off._history_length), "vr_3point_pos")
    on_unarmed = _slot(_per_frame(p_on._build_observation(rs), _SLOTS, p_on._history_length), "vr_3point_pos")
    assert np.allclose(off, on_unarmed), "unarmed live source must not change the 3pt obs"
