"""Live-VIVE teleop on the HYBRID position-force controller (exp:g1-wbt-2pt-hybrid).

The hybrid actor obs is 165D/frame × 4 history = 660D, and it carries
``vr_2point_pos`` / ``vr_2point_orn`` terms → ``detect_obs_mode`` returns ``"2pt"``
→ ``HolosomaWBTPolicy._is_2pt`` is True. So the *existing* live-VIVE override
(built for the 2pt / 2pt-force deploys) fires on the hybrid checkpoint with no
code change. The one genuinely NEW interaction on this controller is that the
per-frame obs now also carries four hybrid-only slots that the earlier presets
did not have — ``force_cmd_local`` / ``force_meas_local`` / ``integrator_state``
/ ``x_tgt_shift`` (offsets 42:66 within the 165D frame).

These tests run the REAL ``_compute_obs_terms`` on the REAL hybrid obs config
with a real Pinocchio yam model + a fake VIVE source, and pin the load-bearing
invariants for the hybrid + VIVE combination:

  1. Assembled per-frame obs is exactly 165D in the trained hybrid term order.
  2. Arming VIVE and moving the trackers changes ONLY the ``vr_2point_pos`` slot
     (24:30) — it never leaks into the force / servo slots (42:66), which stay
     zero when no live F/T source and servo are configured.
  3. OFF (no VIVE source) ⇒ the vr_2point slots are the plain NPZ-FK reference
     (byte-identical to the non-teleop path).
  4. Relative-clutch arming is zero-jump (trackers AT the anchor ⇒ vr_2point ==
     the held FK reference), and only relative tracker motion moves the hands,
     slew-capped.

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

_HYBRID_PRESET = "g1-29dof-holosoma-2pt-hybrid-wbt"

# Per-frame (165D) slot offsets — see observation.holosoma_2pt_hybrid_wbt. Locked
# here so a silent term-order/width change trips the test rather than the robot.
_SLOTS = {
    "command_lower_body": (0, 24),
    "vr_2point_pos": (24, 30),
    "vr_2point_orn": (30, 42),
    "force_cmd_local": (42, 48),
    "force_meas_local": (48, 54),
    "integrator_state": (54, 60),
    "x_tgt_shift": (60, 66),
    "motion_ref_ori_b": (66, 72),
    "projected_gravity": (72, 75),
    "base_ang_vel": (75, 78),
    "joint_pos": (78, 107),
    "joint_vel": (107, 136),
    "actions": (136, 165),
}
_PER_FRAME = 165


def _identity_orn12():
    """Two identity rot6d columns [r00,r01,r10,r11,r20,r21] = [1,0,0,1,0,0]×2."""
    eye6 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    return np.concatenate([eye6, eye6])


class _FakeViveSource:
    """Stand-in for ViveLiveSource: returns a fixed (pos[6], orn[12]) command.

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
        # The real bridge ALWAYS sends orn alongside pos (identity when
        # position-only); the policy asserts orn is non-None when pos is.
        orn = _identity_orn12() if self._orn is None else self._orn.copy()
        return self._pos.copy(), orn


def _stub_hybrid_policy(*, vive=False):
    """Minimal HolosomaWBTPolicy wired exactly as ``_compute_obs_terms`` reads it
    for the hybrid 2pt preset: the REAL hybrid obs config + a real Pinocchio yam
    model + the VIVE/force caches. No heavy __init__ (object.__new__)."""
    from loguru import logger as _loguru

    from holosoma_inference.policies.wbt_utils import PinocchioRobot

    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.config_values.robot_config import VR_2POINT_BODIES, g1_29dof_wbt
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    if not _YAM_URDF.is_file():
        pytest.skip(f"yam URDF not present: {_YAM_URDF}")

    obs_cfg = DEFAULTS[_HYBRID_PRESET].observation
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

    # obs assembly state
    p._obs_terms = terms
    p._obs_mode = "2pt"
    p._is_2pt = True
    p._is_3pt = False
    p._vr_orn_6d = True  # the 2pt family trains the rot6d orn rep (setup_policy sets this)
    # Hand geometry: the 2pt family has no TCP variant, so _vr_offsets() ignores this and
    # always returns the 0.18 m 2pt lever. Set it anyway so the stub matches the real
    # attribute surface (a bare object.__new__ has none) and never AttributeErrors.
    p._vr_pos_tcp = False
    p._history_length = hist
    p._obs_history = {t: deque(maxlen=hist) for t in terms}
    p._last_action = np.zeros(29, dtype=np.float32)
    p._ref_body_is_pelvis = True
    # Pre-seed heading as captured (offsets 0): the robot + reference are at
    # identity yaw here, so the yaw strip is a no-op. This also avoids the
    # _capture_robot_yaw_offset_holosoma path, which reads self.interface (absent
    # in this obs-only stub).
    p._heading_initialized = True
    p.robot_yaw_offset = 0.0
    p.motion_yaw_offset = 0.0
    p._align_rot_mat = np.eye(3)  # standing, identity yaw ⇒ no motion alignment shift

    # VR FK frame ids (resolved as setup_policy does).
    model = p.pinocchio_robot.robot_model
    p._vr_frame_ids = [model.getFrameId(b) for b in VR_2POINT_BODIES]

    # Frozen-clip reference: a standing pose (default angles), root at a plausible
    # standing height + identity yaw. This is what _compute_vr_fk FKs to the hand
    # reference target while the clip is frozen (VIVE hold).
    ref_dof = p.default_dof_angles.astype(np.float32)
    p._motion_command_t = np.concatenate([ref_dof.reshape(1, -1), np.zeros((1, 29), dtype=np.float32)], axis=1)
    p._ref_pos_xyz_t = np.array([[0.0, 0.0, 0.78]], dtype=np.float32)
    p._ref_quat_xyzw_t = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    p._hold_dof_override = None  # ready-pose off for this obs-slot test (isolate VIVE)

    # Force / servo caches (all zero → the four hybrid slots read zero; the
    # byte-identical no-live-source path). No servo, no live F/T here — the point
    # is to prove VIVE does not disturb these slots.
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

    # VIVE state (mirrors __init__ defaults; relative/clutch, orientation off).
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
    # Absolute-mode arm clutch (default ON in __init__); both anchors None until arm.
    p._vive_abs_clutch = True
    p._vr_abs_off_pos = None
    p._vr_abs_orn0 = None
    return p


def _robot_state(p):
    """A robot_state_data row: [root_pos(3), root_quat_wxyz(4), dof(29), lin(3),
    ang(3), dofvel(29)]. Root at standing height + identity ⇒ has_mocap True."""
    n = p.num_dofs
    row = np.zeros((1, 7 + n + 6 + n), dtype=np.float32)
    row[0, 0:3] = [0.0, 0.0, 0.78]
    row[0, 3:7] = [1.0, 0.0, 0.0, 0.0]  # wxyz identity
    row[0, 7 : 7 + n] = p.default_dof_angles
    return row


def _per_frame(obs, term_offsets, hist):
    """Extract the LATEST per-frame 165D vector from the assembled (1, 660) obs.

    _build_observation lays out obs as [term0 × hist, term1 × hist, ...]; the
    newest frame is the LAST history slot of each term. Reassemble it here."""
    obs = obs.reshape(-1)
    out = np.zeros(_PER_FRAME, dtype=np.float32)
    cursor = 0
    for _name, (lo, hi) in term_offsets.items():
        w = hi - lo
        block = obs[cursor : cursor + w * hist]
        out[lo:hi] = block[(hist - 1) * w : hist * w]  # newest frame
        cursor += w * hist
    return out


def _slot(vec165, name):
    lo, hi = _SLOTS[name]
    return vec165[lo:hi]


def test_hybrid_obs_is_660d_and_2pt():
    """The hybrid preset is 2pt (VIVE-eligible) and assembles to 660D."""
    p = _stub_hybrid_policy(vive=False)
    assert p._is_2pt, "hybrid must detect as 2pt so the VIVE override fires"
    obs = p._build_observation(_robot_state(p))
    assert obs.shape == (1, _PER_FRAME * p._history_length), obs.shape


def test_hybrid_force_slots_zero_without_live_source():
    """No live F/T + servo off ⇒ the four hybrid-only slots read exactly zero
    (the byte-identical pure-2pt-tracking path on a hybrid checkpoint)."""
    p = _stub_hybrid_policy(vive=False)
    frame = _per_frame(p._build_observation(_robot_state(p)), _SLOTS, p._history_length)
    for name in ("force_cmd_local", "force_meas_local", "integrator_state", "x_tgt_shift"):
        assert np.allclose(_slot(frame, name), 0.0), f"{name} must be zero with no live source"


def test_vive_moves_only_vr_slots_not_force_slots():
    """The genuinely NEW hybrid interaction: arming VIVE + moving the trackers must
    change ONLY the vr_2point_pos slot — NEVER the force / servo slots."""
    p = _stub_hybrid_policy(vive=True)
    rs = _robot_state(p)

    # Baseline (not armed): vr_2point = NPZ-FK reference; force slots zero.
    base = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)
    ref_vr_pos = _slot(base, "vr_2point_pos").copy()

    # Arm the clutch AT the current FK reference (delta 0), then push both hands.
    live0 = ref_vr_pos.copy()  # trackers coincide with the held hand ref
    p._vive_live.set(live0, None)
    p._vr_clutch_live0 = live0.astype(np.float32).copy()
    p._vive_armed = True
    # Push both hands forward (pelvis +x) far beyond one slew step.
    moved = live0.copy()
    moved[0] += 0.30
    moved[3] += 0.30
    p._vive_live.set(moved, None)

    # Several ticks so the slew ramps the target in.
    frame = None
    for _ in range(60):
        frame = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)

    vr_pos = _slot(frame, "vr_2point_pos")
    # vr_2point_pos advanced along +x for BOTH hands (unscaled: obs_scale 1.0).
    assert vr_pos[0] > ref_vr_pos[0] + 0.05, "left hand x should track the +x push"
    assert vr_pos[3] > ref_vr_pos[3] + 0.05, "right hand x should track the +x push"
    # The four hybrid force/servo slots are UNTOUCHED by the VIVE override.
    for name in ("force_cmd_local", "force_meas_local", "integrator_state", "x_tgt_shift"):
        assert np.allclose(_slot(frame, name), 0.0), f"VIVE must not leak into {name}"


def test_vive_arm_is_zero_jump():
    """Relative-clutch arming with the trackers AT the anchor ⇒ the first armed
    vr_2point_pos equals the held FK reference (no arm jump)."""
    p = _stub_hybrid_policy(vive=True)
    rs = _robot_state(p)
    ref_vr_pos = _slot(_per_frame(p._build_observation(rs), _SLOTS, p._history_length), "vr_2point_pos").copy()

    live0 = ref_vr_pos.copy()
    p._vive_live.set(live0, None)
    p._vr_clutch_live0 = live0.astype(np.float32).copy()
    p._vive_armed = True
    # Trackers held exactly at the anchor (delta 0) ⇒ target == reference.
    armed = _slot(_per_frame(p._build_observation(rs), _SLOTS, p._history_length), "vr_2point_pos")
    assert np.allclose(armed, ref_vr_pos, atol=1e-5), "zero-delta arm must be zero-jump"


def test_vive_off_is_byte_identical_vr_slot():
    """No VIVE source ⇒ the vr_2point_pos slot is the plain NPZ-FK reference,
    independent of the (absent) live source — the byte-identical baseline."""
    p_off = _stub_hybrid_policy(vive=False)
    p_on = _stub_hybrid_policy(vive=True)  # source present but NOT armed
    rs = _robot_state(p_off)
    off = _slot(_per_frame(p_off._build_observation(rs), _SLOTS, p_off._history_length), "vr_2point_pos")
    on_unarmed = _slot(_per_frame(p_on._build_observation(rs), _SLOTS, p_on._history_length), "vr_2point_pos")
    assert np.allclose(off, on_unarmed), "unarmed live source must not change the obs"


def test_absolute_arm_clutch_capture_via_toggle():
    """ABSOLUTE mode (HOLOSOMA_VIVE_RELATIVE=0): pressing 'start' must capture the arm
    clutch offset = FK(reference) − wire, so the target math can hold the arm pose.

    Drives the REAL arm path (_toggle_vive_teleop) with the wire deliberately OFF the
    clip-ref FK (the standing steady-state gap that produced the reported "LEFT hand jerks
    up-left on start"). Asserts the captured offset equals FK(ref) − wire and that it lands
    the first armed target back on FK(ref) — i.e. wire + off == FK(ref)."""
    p = _stub_hybrid_policy(vive=True)
    p._vive_relative = False  # ABSOLUTE mode — the path the fix touches
    p._vive_arm_tol = 0.07  # matches __init__ default; the 3 cm gap below passes it
    p.use_policy_action = True  # A (START) already pressed
    rs = _robot_state(p)
    ref_vr_pos = _slot(_per_frame(p._build_observation(rs), _SLOTS, p._history_length), "vr_2point_pos").copy()

    # Published wire = the MEASURED pose, a few cm off the clip-ref FK (within the gate).
    wire = ref_vr_pos.copy()
    wire[0] += 0.03  # left hand +x
    wire[2] += 0.03  # left hand +z (the "up" in "up-left")
    p._vive_live.set(wire, None)

    p._toggle_vive_teleop()  # press 'start' — arms + captures the clutch offset
    assert p._vive_armed, "absolute arm should succeed (gap within tol)"
    assert p._vr_abs_off_pos is not None, "arm must capture the absolute clutch offset"
    # off == FK(ref) − wire, per point; adding it back to the wire returns FK(ref).
    assert np.allclose(p._vr_abs_off_pos, ref_vr_pos - wire, atol=1e-5), (
        f"offset must be FK(ref)−wire (got {p._vr_abs_off_pos[:3]}, want {(ref_vr_pos - wire)[:3]})")
    assert np.allclose(wire + p._vr_abs_off_pos, ref_vr_pos, atol=1e-5), "wire+off must == FK(ref)"


def test_absolute_arm_clutch_zero_jump_then_tracks_wire_delta():
    """ABSOLUTE mode arm clutch: the first armed tick holds the ARM POSE (zero jump even
    when the wire is off the clip-ref FK), and a subsequent wire move is tracked as a pure
    DELTA from the arm pose — no persistent reach bias.

    This is the regression for the reported "LEFT hand jerks up-left the instant I press
    start". Pre-fix, absolute mode fed the wire (== the robot's measured pose, a few cm off
    the arm-zeroed ready reference) as the target directly, so the obs target stepped from
    FK(ref) → wire on arm and the whole-body policy re-applied its steady-state offset to
    the new target → the lurch. The clutch adds the captured offset FK(ref)−wire so the
    target == FK(ref) at arm, and target == FK(ref) + (wire − wire_arm) afterwards."""
    p = _stub_hybrid_policy(vive=True)
    p._vive_relative = False  # ABSOLUTE mode
    rs = _robot_state(p)
    ref_vr_pos = _slot(_per_frame(p._build_observation(rs), _SLOTS, p._history_length), "vr_2point_pos").copy()

    # Arm with the wire 3 cm off the clip-ref FK. Emulate the capture the toggle performs
    # (FK(ref) − wire) so this test isolates the target-APPLICATION path in _compute_obs_terms.
    wire_arm = ref_vr_pos.copy()
    wire_arm[0] += 0.03  # left +x
    wire_arm[2] += 0.03  # left +z ("up-left")
    p._vive_live.set(wire_arm, None)
    p._vr_abs_off_pos = (ref_vr_pos - wire_arm).astype(np.float32)
    p._vive_armed = True

    # First armed tick: target must equal FK(ref) (ZERO jump), NOT the raw wire.
    armed = _slot(_per_frame(p._build_observation(rs), _SLOTS, p._history_length), "vr_2point_pos")
    assert np.allclose(armed, ref_vr_pos, atol=1e-5), (
        f"arm-clutch first tick must hold FK(ref) (got {armed[:3]}, want {ref_vr_pos[:3]})")
    assert not np.allclose(armed, wire_arm, atol=1e-3), (
        "regression: first armed tick snapped to the raw wire (the arm-jerk bug)")

    # Now the wire moves +x (Execute drives it). The target must follow the DELTA from the
    # arm pose = FK(ref) + (wire − wire_arm), with the offset unchanged (no bias, no drift).
    wire_moved = wire_arm.copy()
    wire_moved[0] += 0.30  # push far beyond one slew step so the ramp fully catches up
    p._vive_live.set(wire_moved, None)
    frame = None
    for _ in range(60):
        frame = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)
    vr_pos = _slot(frame, "vr_2point_pos")
    expected = ref_vr_pos + (wire_moved - wire_arm)  # delta tracking, no persistent offset
    assert np.allclose(vr_pos, expected, atol=1e-3), (
        f"armed target must track wire delta from the arm pose (got {vr_pos[:3]}, want {expected[:3]})")
    # The captured offset must be untouched by ticking (only disarm/stop clears it).
    assert np.allclose(p._vr_abs_off_pos, ref_vr_pos - wire_arm, atol=1e-6), "offset must persist while armed"


def _rotz_rot6d(deg):
    """A single rot6d block (first two COLUMNS of Rz, column-major as _rot6d_to_R reads it)."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    t = np.deg2rad(deg)
    rz = np.array(
        [[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return HolosomaWBTPolicy._R_to_rot6d(rz), rz


def test_absolute_arm_clutch_orientation_is_zero_jump_then_tracks_orn_delta():
    """ABSOLUTE mode arm clutch, ORIENTATION axis: the wrist orn target must hold the FK
    reference at arm (zero twist) even though the wire orn differs from it, and then follow
    only the wire's rotation SINCE the arm anchor.

    The absolute branch has no frozen-ref orn hold (unlike relative), so pre-fix the orn
    target stepped FK(ref) -> wire_orn at arm = an uncommanded wrist twist (the rotational
    analogue of the position lurch). The clutch composes R_target = (R_wire . R_wire0^T) . R_ref
    via the same helpers the relative clutch uses."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    p = _stub_hybrid_policy(vive=True)
    p._vive_relative = False  # ABSOLUTE mode
    rs = _robot_state(p)
    base = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)
    ref_vr_pos = _slot(base, "vr_2point_pos").copy()
    ref_vr_orn = _slot(base, "vr_2point_orn").copy()  # 2×rot6d = 12D, the FK ref orn

    # Wire orn at arm = the MEASURED wrist orn, a real 20 deg rotation off the FK ref (the
    # rotational steady-state offset). Both hands get the same twist for simplicity.
    blk20, rz20 = _rotz_rot6d(20.0)
    wire_orn_arm = np.concatenate([blk20, blk20]).astype(np.float32)
    p._vive_live.set(ref_vr_pos, wire_orn_arm)
    # Arm: position offset is zero here (wire pos == ref) so this isolates orientation.
    p._vr_abs_off_pos = np.zeros_like(ref_vr_pos, dtype=np.float32)
    p._vr_abs_orn0 = wire_orn_arm.copy()
    p._vive_armed = True

    # First armed tick: orn target must be the FK REF (zero twist), NOT the wire orn.
    armed = _slot(_per_frame(p._build_observation(rs), _SLOTS, p._history_length), "vr_2point_orn")
    assert np.allclose(armed, ref_vr_orn, atol=1e-5), (
        f"arm-clutch must hold the FK-ref orientation at arm (got {armed[:6]}, want {ref_vr_orn[:6]})")
    assert not np.allclose(armed, wire_orn_arm, atol=1e-3), (
        "regression: orn target snapped to the raw wire orn (the wrist-twist bug)")

    # Now the wire rotates a further 10 deg (total 30). The target must be that DELTA (10 deg)
    # composed onto the FK ref: R_target = R_delta . R_ref, with R_delta = R_wire . R_wire0^T.
    blk30, rz30 = _rotz_rot6d(30.0)
    p._vive_live.set(ref_vr_pos, np.concatenate([blk30, blk30]).astype(np.float32))
    frame = None
    for _ in range(80):  # let the orn slew fully ramp in
        frame = _per_frame(p._build_observation(rs), _SLOTS, p._history_length)
    got = _slot(frame, "vr_2point_orn")
    r_delta = rz30 @ rz20.T  # the wire's rotation since the arm anchor (10 deg about z)
    for h in range(2):
        s = slice(h * 6, h * 6 + 6)
        r_ref_h = HolosomaWBTPolicy._rot6d_to_R(ref_vr_orn[s])
        expected = HolosomaWBTPolicy._R_to_rot6d(r_delta @ r_ref_h)
        assert np.allclose(got[s], expected, atol=1e-3), (
            f"hand {h}: orn must track the wire delta composed on the FK ref "
            f"(got {got[s]}, want {expected})")
    # The orn anchor must persist while armed (only disarm/stop clears it).
    assert np.allclose(p._vr_abs_orn0, wire_orn_arm, atol=1e-6), "orn anchor must persist while armed"
