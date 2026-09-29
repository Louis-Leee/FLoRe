"""Live-VIVE teleop on the 3-POINT HYBRID GLOBAL-VEL position-force controller
(exp:g1-wbt-3pt-hybrid-global-vel, canonical run bqlzkwok).

The 183D/frame × 4 history = 732D actor obs carries ``vr_3point_pos`` (9D =
L,R,head), ``vr_3point_orn_6d`` (18D = 3 bodies × rot6d — NOT the plain-3pt
quaternion), the exp:g1-wbt-global-vel root-state trio ``motion_ref_pos_b`` /
``motion_ref_vel_b`` / ``base_lin_vel`` (walk-enabled), and the four hybrid
force/servo terms. ``detect_obs_mode`` returns ``"3pt_global"`` →
``HolosomaWBTPolicy._is_3pt`` is True (VIVE-eligible) and ``_vr_orn_6d`` is True
(6D FK + quat→rot6d VIVE ingest).

These tests run the REAL ``_compute_obs_terms`` on the REAL global-vel obs
config with a real Pinocchio yam model + a fake VIVE source, and pin the
load-bearing invariants:

  1. Assembled per-frame obs is exactly 183D in the trained term order (732D
     stacked), and the orn slot is 3 VALID rot6d blocks (orthonormal columns).
  2. Arming VIVE and moving the trackers changes ONLY the ``vr_3point_pos``
     slot — never the force/servo slots NOR the root-state trio (which stays
     whatever the motion/odometry dictate).
  3. OFF (no VIVE source) ⇒ vr slots are the plain NPZ-FK reference.
  4. Relative-clutch arming is zero-jump for BOTH pos (9D) and orn (18D, via
     the real ``_snapshot_clutch_anchor`` quat→rot6d ingest), and a live
     tracker ROTATION moves the orn slot while keeping every block a valid
     rotation (Gram-Schmidt reprojection after the slew).
  5. The root-state trio reads exactly what training defines: ref-vs-robot
     root position error in base frame, ref root velocity in base frame, and
     the robot's own base-frame linear velocity.

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

_GV_PRESET = "g1-29dof-holosoma-3pt-hybrid-global-vel-wbt"

# Per-frame (183D) slot offsets — see observation.holosoma_3pt_hybrid_global_vel_wbt.
# Locked here so a silent term-order/width change trips the test rather than the
# robot. Deltas vs 3pt-hybrid (168D): vr orn 12→18 (rot6d), + motion_ref_pos_b(3)
# + motion_ref_vel_b(3) after the orn block, + base_lin_vel(3) after base_ang_vel.
_SLOTS = {
    "command_lower_body": (0, 24),
    "vr_3point_pos": (24, 33),
    "vr_3point_orn_6d": (33, 51),
    "motion_ref_pos_b": (51, 54),
    "motion_ref_vel_b": (54, 57),
    "force_cmd_local": (57, 63),
    "force_meas_local": (63, 69),
    "integrator_state": (69, 75),
    "x_tgt_shift": (75, 81),
    "motion_ref_ori_b": (81, 87),
    "projected_gravity": (87, 90),
    "base_ang_vel": (90, 93),
    "base_lin_vel": (93, 96),
    "joint_pos": (96, 125),
    "joint_vel": (125, 154),
    "actions": (154, 183),
}
_PER_FRAME = 183


def _identity_orn12_quat():
    """Three identity xyzw quaternions [0,0,0,1] — the 3pt BRIDGE wire rep (the
    bridge keeps streaming quats; the policy converts to rot6d at ingest)."""
    return np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), 3)


class _FakeViveSource:
    """Stand-in for ViveLiveSource: returns a fixed (pos[9], orn[12]) 3pt command.

    ``set(pos, orn)`` updates the held command; ``set(None, None)`` makes it stale
    (``latest()`` returns ``(None, None)``). orn is ALWAYS the 12D quat wire rep —
    exactly what the real 3pt bridge sends, so the ingest conversion is exercised."""

    def __init__(self):
        self._pos = None
        self._orn = None

    def set(self, pos, orn):
        self._pos = None if pos is None else np.asarray(pos, dtype=np.float32)
        self._orn = None if orn is None else np.asarray(orn, dtype=np.float32)

    def latest(self):
        if self._pos is None:
            return None, None
        orn = _identity_orn12_quat() if self._orn is None else self._orn.copy()
        return self._pos.copy(), orn


def _stub_gv_policy(*, vive=False, orientation=False):
    """Minimal HolosomaWBTPolicy wired exactly as ``_compute_obs_terms`` reads it
    for the 3pt-hybrid-global-vel preset: the REAL 183D obs config + a real
    Pinocchio yam model + the VIVE/force caches. No heavy __init__."""
    from loguru import logger as _loguru

    from holosoma_inference.policies.wbt_utils import PinocchioRobot

    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.config_values.robot_config import VR_3POINT_BODIES, g1_29dof_wbt
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    if not _YAM_URDF.is_file():
        pytest.skip(f"yam URDF not present: {_YAM_URDF}")

    obs_cfg = DEFAULTS[_GV_PRESET].observation
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

    # obs assembly state — 3pt_global + the 6D orn rep this time.
    p._obs_terms = terms
    p._obs_mode = "3pt_global"
    p._is_2pt = False
    p._is_3pt = True
    p._vr_orn_6d = True
    # Hand geometry, derived from the preset's terms as setup_policy does. False here (the
    # global-vel preset keeps `vr_3point_pos` / the 0.18 m VR lever) — only the LOCAL
    # unified twins moved to the TCP.
    p._vr_pos_tcp = any("vr_3point_pos_tcp" in t for t in terms)
    p._history_length = hist
    p._obs_history = {t: deque(maxlen=hist) for t in terms}
    p._last_action = np.zeros(29, dtype=np.float32)
    p._ref_body_is_pelvis = True
    p._heading_initialized = True
    p.robot_yaw_offset = 0.0
    p.motion_yaw_offset = 0.0
    p._align_rot_mat = np.eye(3)
    p._align_pos_offset = np.zeros(3)

    # VR FK frame ids (3pt bodies: L wrist, R wrist, torso).
    model = p.pinocchio_robot.robot_model
    p._vr_frame_ids = [model.getFrameId(b) for b in VR_3POINT_BODIES]

    ref_dof = p.default_dof_angles.astype(np.float32)
    p._motion_command_t = np.concatenate([ref_dof.reshape(1, -1), np.zeros((1, 29), dtype=np.float32)], axis=1)
    p._ref_pos_xyz_t = np.array([[0.0, 0.0, 0.78]], dtype=np.float32)
    p._ref_quat_xyzw_t = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    p._ref_vel_xyz_t = np.zeros((1, 3), dtype=np.float32)
    p._hold_dof_override = None
    # Teleop/STAND default: the clip is FROZEN (motion_ref_vel_b then commands
    # zero — the frozen-clip guard). Tests that exercise the PLAYING branch flip
    # this to True explicitly.
    p.motion_clip_progressing = False

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

    # VIVE state (relative/clutch; orientation per test).
    p._vive_live = _FakeViveSource() if vive else None
    p._vive_armed = False
    p._vive_relative = True
    p._vive_orientation = orientation
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
    p._orn_mismatch_warned = False
    return p


def _robot_state(p, *, root_pos=(0.0, 0.0, 0.78), base_lin_vel=(0.0, 0.0, 0.0)):
    n = p.num_dofs
    row = np.zeros((1, 7 + n + 6 + n), dtype=np.float32)
    row[0, 0:3] = root_pos
    row[0, 3:7] = [1.0, 0.0, 0.0, 0.0]  # wxyz identity
    row[0, 7 : 7 + n] = p.default_dof_angles
    row[0, 7 + n : 7 + n + 3] = base_lin_vel  # world lin vel (FK odometry / mocap slice)
    return row


def _per_frame(obs, term_offsets, hist):
    """Extract the LATEST per-frame 183D vector from the assembled (1, 732) obs."""
    obs = obs.reshape(-1)
    out = np.zeros(_PER_FRAME, dtype=np.float32)
    cursor = 0
    for _name, (lo, hi) in term_offsets.items():
        w = hi - lo
        block = obs[cursor : cursor + w * hist]
        out[lo:hi] = block[(hist - 1) * w : hist * w]  # newest frame
        cursor += w * hist
    return out


def _slot(vec, name):
    lo, hi = _SLOTS[name]
    return vec[lo:hi]


def _assert_valid_rot6d_blocks(orn18, atol=1e-4):
    for b in range(3):
        m = orn18[b * 6 : b * 6 + 6].reshape(3, 2)
        assert abs(np.linalg.norm(m[:, 0]) - 1.0) < atol, f"body {b} col0 not unit"
        assert abs(np.linalg.norm(m[:, 1]) - 1.0) < atol, f"body {b} col1 not unit"
        assert abs(float(m[:, 0] @ m[:, 1])) < atol, f"body {b} cols not orthogonal"


# --- assembly ------------------------------------------------------------------
def test_gv_obs_is_732d_and_3pt_global():
    p = _stub_gv_policy(vive=False)
    assert p._is_3pt and p._vr_orn_6d
    obs = p._build_observation(_robot_state(p))
    assert obs.shape == (1, _PER_FRAME * p._history_length), obs.shape


def test_gv_orn_slot_is_valid_rot6d():
    """vr_3point_orn_6d is 3 valid rot6d blocks (18D) at the frozen standing ref —
    NOT quaternions (the plain-3pt rep)."""
    p = _stub_gv_policy(vive=False)
    frame = _per_frame(p._build_observation(_robot_state(p)), _SLOTS, p._history_length)
    _assert_valid_rot6d_blocks(_slot(frame, "vr_3point_orn_6d"))


def test_gv_force_slots_zero_without_live_source():
    p = _stub_gv_policy(vive=False)
    frame = _per_frame(p._build_observation(_robot_state(p)), _SLOTS, p._history_length)
    for name in ("force_cmd_local", "force_meas_local", "integrator_state", "x_tgt_shift"):
        assert np.allclose(_slot(frame, name), 0.0), f"{name} must be zero with no live source"


# --- root-state trio (the walk-enabling delta) -----------------------------------
def test_gv_motion_ref_pos_b_is_ref_minus_robot_in_base_frame():
    """Identity root orientation ⇒ motion_ref_pos_b == ref_pos − robot_root_pos."""
    p = _stub_gv_policy(vive=False)
    rs = _robot_state(p, root_pos=(0.3, -0.2, 0.78))
    frame = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)
    expected = np.array([0.0 - 0.3, 0.0 - (-0.2), 0.78 - 0.78], dtype=np.float32)
    assert np.allclose(_slot(frame, "motion_ref_pos_b"), expected, atol=1e-5)


def test_gv_motion_ref_vel_b_reads_ref_velocity_while_clip_plays():
    """While the clip PLAYS, the ref root velocity (world) lands in
    motion_ref_vel_b (identity root ⇒ pass-through)."""
    p = _stub_gv_policy(vive=False)
    p.motion_clip_progressing = True
    p._ref_vel_xyz_t = np.array([[0.5, 0.1, 0.0]], dtype=np.float32)
    frame = _per_frame(p._build_observation(_robot_state(p)), _SLOTS, p._history_length)
    assert np.allclose(_slot(frame, "motion_ref_vel_b"), [0.5, 0.1, 0.0], atol=1e-5)


def test_gv_motion_ref_vel_b_zero_while_clip_frozen():
    """FROZEN-CLIP GUARD: while the clip is frozen (STAND hold / VIVE-armed),
    motion_ref_vel_b must command ZERO even if the frozen frame's stored root
    velocity is nonzero (arming mid-clip / a clip trimmed mid-stride) — the
    reference root is NOT advancing, so a nonzero v would contradict the static
    motion_ref_pos_b hold and can keep a walk-enabled policy leaning/stepping."""
    p = _stub_gv_policy(vive=False)
    p.motion_clip_progressing = False
    p._ref_vel_xyz_t = np.array([[0.5, 0.1, 0.0]], dtype=np.float32)  # frozen mid-walk frame
    frame = _per_frame(p._build_observation(_robot_state(p)), _SLOTS, p._history_length)
    assert np.allclose(_slot(frame, "motion_ref_vel_b"), 0.0, atol=1e-6)


def test_gv_base_lin_vel_reads_odometry_slice():
    """base_lin_vel = the robot-state world lin vel rotated into the base frame
    (identity root ⇒ pass-through). This is the slice FK odometry / mocap fills —
    all-zero without a wrapper, which is why --task.fk-odometry true is required."""
    p = _stub_gv_policy(vive=False)
    frame = _per_frame(p._build_observation(_robot_state(p, base_lin_vel=(0.4, 0.0, -0.1))), _SLOTS, p._history_length)
    assert np.allclose(_slot(frame, "base_lin_vel"), [0.4, 0.0, -0.1], atol=1e-5)


# --- VIVE position teleop --------------------------------------------------------
def test_vive_gv_moves_only_vr_pos_slot():
    """Arming VIVE + moving the trackers changes ONLY the vr_3point_pos slot —
    NEVER the force/servo slots nor the root-state trio."""
    p = _stub_gv_policy(vive=True)
    rs = _robot_state(p)

    base = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)
    ref_vr_pos = _slot(base, "vr_3point_pos").copy()

    live0 = ref_vr_pos.copy()
    p._vive_live.set(live0, None)
    assert p._snapshot_clutch_anchor()
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
    assert vr_pos[6] > ref_vr_pos[6] + 0.05, "HEAD x should track the +x push"
    for name in ("force_cmd_local", "force_meas_local", "integrator_state", "x_tgt_shift"):
        assert np.allclose(_slot(frame, name), 0.0), f"VIVE must not leak into {name}"
    for name in ("motion_ref_pos_b", "motion_ref_vel_b", "base_lin_vel"):
        assert np.allclose(_slot(frame, name), _slot(base, name), atol=1e-6), f"VIVE must not leak into {name}"
    # Position-only teleop: orn slot holds the FK reference (still valid rot6d).
    assert np.allclose(_slot(frame, "vr_3point_orn_6d"), _slot(base, "vr_3point_orn_6d"), atol=1e-5)


def test_vive_gv_arm_is_zero_jump():
    p = _stub_gv_policy(vive=True)
    rs = _robot_state(p)
    ref_vr_pos = _slot(_per_frame(p._build_observation(rs), _SLOTS, p._history_length), "vr_3point_pos").copy()

    p._vive_live.set(ref_vr_pos.copy(), None)
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True
    armed = _slot(_per_frame(p._build_observation(rs), _SLOTS, p._history_length), "vr_3point_pos")
    assert np.allclose(armed, ref_vr_pos, atol=1e-5), "zero-delta arm must be zero-jump"


def test_vive_gv_off_is_byte_identical_vr_slot():
    p_off = _stub_gv_policy(vive=False)
    p_on = _stub_gv_policy(vive=True)  # source present but NOT armed
    rs = _robot_state(p_off)
    off = _per_frame(p_off._build_observation(rs), _SLOTS, p_off._history_length)
    on_unarmed = _per_frame(p_on._build_observation(rs), _SLOTS, p_on._history_length)
    assert np.allclose(off, on_unarmed), "unarmed live source must not change the obs"


# --- VIVE orientation teleop (the quat→rot6d ingest path) -------------------------
def test_vive_gv_orn_clutch_zero_jump_via_real_anchor_snapshot():
    """Arm with the ORN clutch on (bridge sends quats): _snapshot_clutch_anchor
    must store the anchor CONVERTED to rot6d, and an unmoved tracker ⇒ the orn
    slot equals the FK reference (zero-jump), still valid rot6d."""
    p = _stub_gv_policy(vive=True, orientation=True)
    rs = _robot_state(p)
    base = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)
    ref_orn = _slot(base, "vr_3point_orn_6d").copy()
    ref_pos = _slot(base, "vr_3point_pos").copy()

    live_quats = _identity_orn12_quat()
    p._vive_live.set(ref_pos.copy(), live_quats)
    assert p._snapshot_clutch_anchor()
    assert p._vr_clutch_orn0 is not None and p._vr_clutch_orn0.size == 18, (
        "anchor must be stored in the 18D rot6d obs rep (quat wire → rot6d ingest)"
    )
    p._vive_armed = True

    frame = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)
    armed_orn = _slot(frame, "vr_3point_orn_6d")
    assert np.allclose(armed_orn, ref_orn, atol=1e-4), "zero-delta orn arm must be zero-jump"
    _assert_valid_rot6d_blocks(armed_orn)


def test_vive_gv_orn_clutch_tracks_live_rotation_and_stays_on_manifold():
    """Rotating the LEFT tracker by 20° about z moves the LEFT orn block toward
    R_d·R_ref (slew-ramped over ticks); every block stays a valid rotation after
    the slew + Gram-Schmidt reprojection; the untouched bodies hold the ref."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    p = _stub_gv_policy(vive=True, orientation=True)
    rs = _robot_state(p)
    base = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)
    ref_orn = _slot(base, "vr_3point_orn_6d").copy()
    ref_pos = _slot(base, "vr_3point_pos").copy()

    live0_quats = _identity_orn12_quat()
    p._vive_live.set(ref_pos.copy(), live0_quats)
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True

    # Rotate the LEFT tracker 20° about z (world/pelvis) — wire rep stays quat.
    ang = np.deg2rad(20.0)
    c, s = np.cos(ang), np.sin(ang)
    R_d = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    q_left = P._R_to_quat_xyzw(R_d)  # identity live0 ⇒ live = R_d
    live_quats = _identity_orn12_quat()
    live_quats[0:4] = q_left
    p._vive_live.set(ref_pos.copy(), live_quats)

    frame = None
    for _ in range(120):
        frame = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)
    orn = _slot(frame, "vr_3point_orn_6d")
    _assert_valid_rot6d_blocks(orn)
    # LEFT block converged to R_d @ R_ref.
    expected_left = R_d @ P._rot6d_to_R(ref_orn[0:6])
    assert np.allclose(P._rot6d_to_R(orn[0:6]), expected_left, atol=1e-3), "left orn must track the live rotation"
    # RIGHT + HEAD blocks hold the reference.
    assert np.allclose(orn[6:12], ref_orn[6:12], atol=1e-4)
    assert np.allclose(orn[12:18], ref_orn[12:18], atol=1e-4)


def test_vive_gv_orn_clutch_cross_family_frame_holds_ref_orientation():
    """A 2pt live frame (pos[6] + orn[12] = 2×rot6d — e.g. the bridge left in
    MODE=2) reaching a 3pt-6D hold with the orn clutch ON must HOLD the FK
    reference orientation — the 2pt rot6d bytes must NEVER be misread as
    quaternions. The position clutch still drives the two hand points (positions
    are rep-compatible xyz), and the head position holds the ref."""
    p = _stub_gv_policy(vive=True, orientation=True)
    rs = _robot_state(p)
    base = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)
    ref_orn = _slot(base, "vr_3point_orn_6d").copy()
    ref_pos = _slot(base, "vr_3point_pos").copy()

    # 2pt wire frame: pos[6] = the two hands at the ref, orn[12] = 2×identity rot6d.
    live0_pos6 = ref_pos[:6].copy()
    identity_rot6d = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    live_orn12_rot6d = np.tile(identity_rot6d, 2)
    p._vive_live.set(live0_pos6, live_orn12_rot6d)
    assert p._snapshot_clutch_anchor()
    # pos[6] ⇒ NOT a genuine 3pt frame ⇒ the anchor must stay UNCONVERTED (12D).
    assert p._vr_clutch_orn0 is not None and p._vr_clutch_orn0.size == 12
    p._vive_armed = True

    moved = live0_pos6.copy()
    moved[0] += 0.20  # left hand +x
    p._vive_live.set(moved, live_orn12_rot6d)
    frame = None
    for _ in range(60):
        frame = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)

    # Orientation held at the FK reference (the cross-family guard), still valid.
    orn = _slot(frame, "vr_3point_orn_6d")
    assert np.allclose(orn, ref_orn, atol=1e-5), "cross-family orn must hold the FK ref"
    _assert_valid_rot6d_blocks(orn)
    # Left hand position tracked; head held at ref.
    vr_pos = _slot(frame, "vr_3point_pos")
    assert vr_pos[0] > ref_pos[0] + 0.05
    assert np.allclose(vr_pos[6:9], ref_pos[6:9], atol=1e-5)


def test_vive_gv_stale_source_holds_last_fed_target():
    """Mid-armed staleness (latest() → (None, None)) must HOLD the last fed
    target (never zeros, never a jump); a later fresh frame resumes the ramp."""
    p = _stub_gv_policy(vive=True)
    rs = _robot_state(p)
    base = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)
    ref_pos = _slot(base, "vr_3point_pos").copy()

    p._vive_live.set(ref_pos.copy(), None)
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True
    moved = ref_pos.copy()
    moved[0] += 0.10
    p._vive_live.set(moved, None)
    for _ in range(30):
        frame = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)
    tracked = _slot(frame, "vr_3point_pos").copy()
    assert tracked[0] > ref_pos[0] + 0.05

    # Go STALE: the fed target must stay exactly where it was (no zeros, no snap-back).
    p._vive_live.set(None, None)
    for _ in range(5):
        frame = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)
    held = _slot(frame, "vr_3point_pos")
    assert np.allclose(held, tracked, atol=1e-6), "stale source must hold the last fed target"
