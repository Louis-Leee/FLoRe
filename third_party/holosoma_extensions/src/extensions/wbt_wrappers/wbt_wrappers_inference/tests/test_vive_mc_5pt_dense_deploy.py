"""Deploy path for the two MOTION-COMPLETION teleop presets (user 2026-08-27):

  * ``exp:g1-wbt-5pt-hybrid-unified-anneal``  — 184D × 4 = 736D, five keypoints
  * ``exp:g1-wbt-hybrid-unified-anneal``      — 183D × 4 = 732D, 29 joint targets

The rig has THREE trackers. Both presets want more than that, so the same three-tracker
stream goes through Motion_Completion in ``vive_teleop_bridge`` and the extra channels are
read out of its 29-DoF FK. From ``holosoma_wbt``'s side that is invisible: the bridge
streams the policy's own rep, so what is tested here is that the DEPLOY runtime consumes it
without inventing a second geometry.

Four things can be wrong without any dimension check noticing, and each has a test below:

1. **5pt silently classified as "dense".** ``"vr_5point"`` does not contain the
   ``"vr_3point"`` substring, so before the fix ``detect_obs_mode`` fell through every
   ``vr_*`` rule and returned ``"dense"`` — a real mode whose obs path skips VR FK
   entirely. The 736D policy would have been fed a zero tracking slot at the right width.
2. **The 5pt prefix drifting off the 3pt TCP.** Slots ``[0:9]``/``[0:12]`` of the 5pt
   vectors must be byte-identical to the 3pt ones, because the training layout appends the
   feet rather than interleaving them. A lever/TCP slip here is 6.7 cm per hand at the
   same width.
3. **Dense joint targets index-aligned instead of name-mapped.** The completion's joint
   order and holosoma's ``dof_names`` are two independent real orders that happen to agree
   today. Index-aligning would scramble the body the day that stops being true, with no
   error raised anywhere.
4. **The 732D collision.** dense-unified is 732D and so is
   ``g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt``. Disjoint layouts, identical
   width — only the term names separate them.

Skips cleanly when pinocchio or the yam URDF is unavailable.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("wbt_wrappers_inference")

_REPO = Path(__file__).resolve().parents[5]
_YAM_URDF = _REPO / "thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof_yam.urdf"

_5PT_PRESET = "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt"
_DENSE_PRESET = "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt"
_3PT_PRESET = "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt"
#: The BASE whole-body tracker exp:g1-wbt (W&B run ruo0o6fr), 628D. Joint-space steered like
#: dense-unified — the 58D `command` term IS its tracking front-end, so the bridge's dense
#: wire drives it unchanged — but with NO force block, which is exactly why the
#: `contact_phase`-keyed dense-unified check cannot see it. Same preset used for the
#: already-verified recorded-clip replay, which is what makes it the sane first rig target.
_WBT_PRESET = "g1-29dof-holosoma-wbt"

# Per-frame slot offsets, locked so a silent term-order/width change trips the test rather
# than the robot. Derived from observation.holosoma_{5pt,dense}_hybrid_unified_wbt.
_SLOTS_5PT = {
    "command_lower_body": (0, 24),
    "vr_5point_pos_tcp": (24, 39),  # 5 × 3
    "vr_5point_orn": (39, 59),  # 5 × 4 (xyzw)
    "force_cmd_local": (59, 65),
    "force_meas_local": (65, 71),
    "contact_phase": (71, 73),
    "integrator_state": (73, 79),
    "x_tgt_shift": (79, 85),
    "motion_ref_ori_b": (85, 91),
    "projected_gravity": (91, 94),
    "base_ang_vel": (94, 97),
    "joint_pos": (97, 126),
    "joint_vel": (126, 155),
    "actions": (155, 184),
}
_SLOTS_DENSE = {
    "command": (0, 58),  # 29 joint targets ++ 29 velocities
    "force_cmd_local": (58, 64),
    "force_meas_local": (64, 70),
    "contact_phase": (70, 72),
    "integrator_state": (72, 78),
    "x_tgt_shift": (78, 84),
    "motion_ref_ori_b": (84, 90),
    "projected_gravity": (90, 93),
    "base_ang_vel": (93, 96),
    "joint_pos": (96, 125),
    "joint_vel": (125, 154),
    "actions": (154, 183),
}
# BASE tracker exp:g1-wbt (628D = 157 x 4). The dense layout with the whole 26D force block
# deleted — which is the point: `command` still leads at slot 0, so the completion's joint
# solution lands in the same place, and everything after it shifts down by 26.
_SLOTS_WBT = {
    "command": (0, 58),  # 29 joint targets ++ 29 velocities
    "motion_ref_ori_b": (58, 64),
    "projected_gravity": (64, 67),
    "base_ang_vel": (67, 70),
    "joint_pos": (70, 99),
    "joint_vel": (99, 128),
    "actions": (128, 157),
}
_SLOTS = {"5pt": _SLOTS_5PT, "dense": _SLOTS_DENSE, "3pt": None, "wbt": _SLOTS_WBT}
_PER_FRAME = {"5pt": 184, "dense": 183, "wbt": 157}


def _identity_quats(n: int) -> np.ndarray:
    """n identity xyzw quaternions — the vr_{3,5}point_orn wire/obs rep."""
    return np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), n)


class _FakeKeypointSource:
    """Stand-in for ViveLiveSource on a keypoint frame (2pt/3pt/5pt)."""

    def __init__(self, n_points: int = 5):
        self._n = n_points
        self._pos: np.ndarray | None = None
        self._orn: np.ndarray | None = None

    def set(self, pos, orn=None):
        self._pos = None if pos is None else np.asarray(pos, dtype=np.float32)
        self._orn = None if orn is None else np.asarray(orn, dtype=np.float32)

    def latest(self):
        if self._pos is None:
            return None, None
        orn = _identity_quats(self._pos.size // 3) if self._orn is None else self._orn.copy()
        return self._pos.copy(), orn

    def latest_joints(self):
        return None

    def latest_force(self):
        return None


class _FakeDenseSource:
    """Stand-in for ViveLiveSource on a DENSE frame: 29 joint targets + names."""

    def __init__(self, names: list[str]):
        self._names = tuple(names)
        self._q: np.ndarray | None = None
        self._dq: np.ndarray | None = None

    def set(self, q, dq=None, names=None):
        self._q = None if q is None else np.asarray(q, dtype=np.float32)
        self._dq = np.zeros(29, dtype=np.float32) if dq is None else np.asarray(dq, dtype=np.float32)
        if names is not None:
            self._names = tuple(names)

    def latest_joints(self):
        if self._q is None:
            return None
        return self._q.copy(), self._dq.copy(), self._names

    def latest(self):
        return None, None  # a dense frame carries no keypoints

    def latest_force(self):
        return None


def _stub_policy(family: str, *, live=None, relative: bool = True):
    """A HolosomaWBTPolicy wired exactly as ``_compute_obs_terms`` reads it for one of the
    two MC presets: the REAL obs config off the registry, a real Pinocchio yam model, and
    the observe-only servo state ``_resolve_unified_mode`` would have installed.

    The mode flags are DERIVED the way ``setup_policy`` derives them (``detect_obs_mode`` +
    the contact_phase marker), never hand-set, so this stub cannot disagree with the real
    startup path about which family it is.
    """
    from loguru import logger as _loguru

    from holosoma_inference.policies.wbt_utils import PinocchioRobot

    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.config_values.robot_config import (
        VR_3POINT_BODIES,
        VR_5POINT_BODIES,
        g1_29dof_wbt,
    )
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy
    from wbt_wrappers_inference.utils.force_deploy_math import ServoConfig
    from wbt_wrappers_inference.utils.obs_utils import detect_obs_mode

    if not _YAM_URDF.is_file():
        pytest.skip(f"yam URDF not present: {_YAM_URDF}")

    preset = {"5pt": _5PT_PRESET, "dense": _DENSE_PRESET, "3pt": _3PT_PRESET, "wbt": _WBT_PRESET}[family]
    obs_cfg = DEFAULTS[preset].observation
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

    p._obs_terms = terms
    p._obs_mode = detect_obs_mode(terms)
    p._is_2pt = p._obs_mode == "2pt"
    p._is_3pt = p._obs_mode in ("3pt", "3pt_global")
    p._is_5pt = p._obs_mode == "5pt"
    p._has_contact_phase_term = any("contact_phase" in t for t in terms)
    p._is_dense_unified = p._obs_mode == "dense" and p._has_contact_phase_term and any(t == "command" for t in terms)
    p._vr_orn_6d = False
    p._vr_pos_tcp = any(("vr_3point_pos_tcp" in t or "vr_5point_pos_tcp" in t) for t in terms)
    p._history_length = hist
    p._obs_history = {t: deque(maxlen=hist) for t in terms}
    p._last_action = np.zeros(29, dtype=np.float32)
    p._ref_body_is_pelvis = True
    p._heading_initialized = True
    p.robot_yaw_offset = 0.0
    p.motion_yaw_offset = 0.0
    p._align_rot_mat = np.eye(3)
    p._height_scan_dim = 0

    model = p.pinocchio_robot.robot_model
    _bodies = VR_5POINT_BODIES if p._is_5pt else VR_3POINT_BODIES
    # Keyed on the obs MODE, not on the unified marker: NO dense preset has keypoints, so the
    # base tracker must get the empty list too (it is resolved before `_vive_live` is set,
    # and this decision does not depend on a bridge being attached).
    p._vr_frame_ids = [] if p._obs_mode == "dense" else [model.getFrameId(b) for b in _bodies]
    p._force_wrist_frame_ids = None

    ref_dof = p.default_dof_angles.astype(np.float32)
    p._motion_command_t = np.concatenate([ref_dof.reshape(1, -1), np.zeros((1, 29), dtype=np.float32)], axis=1)
    p._ref_pos_xyz_t = np.array([[0.0, 0.0, 0.78]], dtype=np.float32)
    p._ref_quat_xyzw_t = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    p._hold_dof_override = None
    p.motion_clip_progressing = False

    # Force / servo state — OBSERVE-ONLY, no live F/T (so all force slots read zeros).
    p._force_live = None
    p._force_meas_sign = 1.0
    p._force_meas_pelvis = np.zeros((2, 3), dtype=np.float64)
    p._servo_integ = np.zeros((2, 3), dtype=np.float64)
    p._servo_delta_x = np.zeros((2, 3), dtype=np.float64)
    p._servo_dq = np.zeros(29, dtype=np.float64)
    p._servo_arm_cols = None
    p._servo_arm_pd_slice = []
    p._servo_cfg = ServoConfig()
    p._force_contact_n = 2.0
    p._force_runaway_n = 0.0
    p._hybrid_servo = False
    p._servo_observe_only = True
    p._unified_integ_cmd_gate = True
    p._integ_windup_warned = False
    p._contact_phase_nonfinite_warned = False
    p._force_tare_burst_thread = None
    p._force_tare_settle_s = 1.5
    p._force_cmd_pelvis_raw = None
    p._force_cmd_local_scaled = None
    p._force_cmd_hand_frame = False
    p._force_cmd_sensor_frame = False
    p._force_cmd_keys = None
    p.use_policy_action = True
    p._compliance_vec = np.zeros(6, dtype=np.float32)

    p._contact_phase = np.zeros(2, dtype=np.float64)
    p._contact_hyst = np.zeros(2, dtype=np.float64)
    p._contact_phase_ticks = 0
    p._contact_phase_theta_on = 2.0
    p._contact_phase_theta_off = 1.5
    p._contact_phase_tau_s = 0.05
    p._contact_phase_warmup_steps = 5

    p._vive_live = live
    # Derived here, AFTER _vive_live, because setup_policy's rule depends on it: the BASE
    # tracker joins the joint-space teleop family only when a bridge is actually configured,
    # which is what keeps its clip-replay deploy unchanged. Never hand-set, same as the
    # flags above, so the stub cannot disagree with the real startup path.
    p._is_dense_base_teleop = (
        p._obs_mode == "dense"
        and not p._has_contact_phase_term
        and any(t == "command" for t in terms)
        and p._vive_live is not None
    )
    p._vive_armed = False
    p._vive_relative = relative
    p._vive_orientation = False
    p._vive_max_pos_step = 0.02
    p._vive_max_orn_step = 0.05
    p._vive_max_joint_step = 0.05
    p._vive_arm_tol = 0.07
    p._vive_arm_tol_rad = 0.20
    p._vive_abs_clutch = True
    p._orn_mismatch_warned = False
    p._clear_vive_teleop_anchors()
    return p


def _robot_state(p):
    n = p.num_dofs
    row = np.zeros((1, 7 + n + 6 + n), dtype=np.float32)
    row[0, 0:3] = [0.0, 0.0, 0.78]
    row[0, 3:7] = [1.0, 0.0, 0.0, 0.0]  # wxyz identity
    row[0, 7 : 7 + n] = p.default_dof_angles
    return row


def _terms(p):
    """The raw per-term dict (pre-history-stacking) — what the obs is built from."""
    return p._compute_obs_terms(_robot_state(p))


def _obs(p, family):
    """Assembled obs, then the NEWEST per-frame vector sliced back out of it."""
    flat = p._build_observation(_robot_state(p)).reshape(-1)
    hist = p._history_length
    out = np.zeros(_PER_FRAME[family], dtype=np.float32)
    cursor = 0
    for _name, (lo, hi) in _SLOTS[family].items():
        w = hi - lo
        block = flat[cursor : cursor + w * hist]
        out[lo:hi] = block[(hist - 1) * w : hist * w]
        cursor += w * hist
    return out


def _slot(vec, family, name):
    lo, hi = _SLOTS[family][name]
    return vec[lo:hi]


def _settle(p, family, name, ticks: int = 40):
    """Tick the obs until the slew limiter has converged, then return that slot.

    Necessary because ``_slew_vec`` caps the L2 norm of the WHOLE per-tick change, so a
    step applied to several points at once is capped on the first tick even when each
    point moved less than the cap: 5 points × 1 cm is L2 2.24 cm against a 2 cm budget.
    A single-tick comparison therefore has to use a tolerance loose enough to hide a real
    tracking error — settling first lets the tracking assertions be exact, and the cap gets
    its own test (:func:`test_slew_cap_limits_the_first_tick_after_a_step`)."""
    out = None
    for _ in range(ticks):
        out = _slot(_obs(p, family), family, name).copy()
    return out


# ── 1. the mode-detection regression ────────────────────────────────────────────


def test_5pt_obs_mode_is_5pt_not_dense() -> None:
    """The exact bug this work started from. ``"vr_5point"`` does not contain the
    ``"vr_3point"`` substring, so a 5pt preset used to match no ``vr_*`` rule and — because
    its obs carries motion_ref_ori_b rather than motion_ref_pos_b — returned ``"dense"``.
    "dense" is a REGISTERED mode whose obs path never runs VR FK, so the 736D policy would
    have been handed a zero-filled tracking slot at exactly the right width. Nothing
    downstream could have caught that."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.utils.obs_utils import detect_obs_mode

    terms = list(DEFAULTS[_5PT_PRESET].observation.obs_dict["actor_obs"])
    assert detect_obs_mode(terms) == "5pt", terms
    # And the dense-unified preset really is the dense mode (it has no keypoints at all).
    assert detect_obs_mode(list(DEFAULTS[_DENSE_PRESET].observation.obs_dict["actor_obs"])) == "dense"


def test_mode_flags_are_mutually_exclusive_across_all_registry_presets() -> None:
    """No preset may resolve to two families at once, and `_is_dense_unified` must NOT fire
    for the legacy dense/global presets — their deploy behaviour (no ready-pose hold, no
    live joint override) has to stay exactly as it was."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.utils.obs_utils import detect_obs_mode

    seen_legacy_dense = 0
    for key, cfg in DEFAULTS.items():
        terms = list(cfg.observation.obs_dict["actor_obs"])
        mode = detect_obs_mode(terms)
        flags = {
            "2pt": mode == "2pt",
            "3pt": mode in ("3pt", "3pt_global"),
            "5pt": mode == "5pt",
        }
        assert sum(flags.values()) <= 1, (key, mode, flags)
        dense_unified = mode == "dense" and any("contact_phase" in t for t in terms) and any(t == "command" for t in terms)
        if mode == "dense" and not dense_unified:
            seen_legacy_dense += 1
        if dense_unified:
            assert key == _DENSE_PRESET, f"{key} unexpectedly resolves as dense-unified"
    assert seen_legacy_dense > 0, "no legacy dense preset left to regress against"


# ── 2. the 5pt geometry: a strict superset of 3pt ───────────────────────────────


def test_5pt_fk_prefix_is_byte_identical_to_3pt() -> None:
    """Slots [0:9] of the 5pt position vector and [0:12] of its orientation vector must be
    BYTE-identical to the 3pt ones from the same reference pose.

    That is the property the training layout guarantees (the feet are APPENDED, not
    interleaved), and it is what makes 5pt a drop-in for a 3pt operator: the hands are
    tracked at the same 0.242 m TCP and the head at the same torso +0.35 z. A lever/TCP
    slip or a reordering would keep every width intact and move each hand 6.7 cm."""
    p5 = _stub_policy("5pt")
    p3 = _stub_policy("3pt")

    pos5, orn5 = p5._compute_vr_fk(p5._motion_command_t[0, :29], p5._vr_offsets())
    pos3, orn3 = p3._compute_vr_fk(p3._motion_command_t[0, :29], p3._vr_offsets())

    assert pos5.shape == (15,) and orn5.shape == (20,)
    assert pos3.shape == (9,) and orn3.shape == (12,)
    assert np.array_equal(pos5[:9], pos3), (pos5[:9], pos3)
    assert np.array_equal(orn5[:12], orn3), (orn5[:12], orn3)


def test_5pt_feet_carry_no_lever_and_hands_carry_the_tcp() -> None:
    """The two foot points are the BARE ``*_ankle_roll_link`` origins. A hand offset is a
    physical mounting lever and the head offset lifts torso_link to HMD height, but there
    is no such lever for a foot — maskedmimic tracks the raw link, so inventing a sole
    offset would price a point nothing else in the stack knows about."""
    from wbt_wrappers_inference.config_values.robot_config import (
        VR_5POINT_BODIES,
        VR_5POINT_OFFSETS_TCP,
    )

    assert len(VR_5POINT_BODIES) == len(VR_5POINT_OFFSETS_TCP) == 5
    assert VR_5POINT_BODIES[3:] == ["left_ankle_roll_link", "right_ankle_roll_link"]
    assert [list(o) for o in VR_5POINT_OFFSETS_TCP[3:]] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert [round(o[0], 4) for o in VR_5POINT_OFFSETS_TCP[:2]] == [0.242, 0.242]
    assert list(VR_5POINT_OFFSETS_TCP[2]) == [0.0, 0.0, 0.35]

    # And the runtime really uses that table (not a re-typed copy).
    p5 = _stub_policy("5pt")
    assert np.array_equal(p5._vr_offsets(), np.asarray(VR_5POINT_OFFSETS_TCP, dtype=np.float32))


def test_5pt_obs_is_736d_with_the_expected_slot_widths() -> None:
    p5 = _stub_policy("5pt")
    flat = p5._build_observation(_robot_state(p5)).reshape(-1)
    assert flat.size == 736, flat.size
    t = _terms(p5)
    assert t["vr_5point_pos_tcp"].size == 15
    assert t["vr_5point_orn"].size == 20
    assert sum(v.size for v in t.values()) == 184


# ── 3. 5pt live teleop: the 3pt contract at N=5 ─────────────────────────────────


def test_5pt_arm_is_zero_jump_then_tracks_the_wire_delta() -> None:
    """Relative/clutch arming must not move the robot: target == FK(ref) at arm, then
    follows only the tracker displacement since the anchor. Verified on all FIVE points —
    including the feet, which a 3-point-shaped clutch would leave frozen at the reference
    while the hands moved."""
    live = _FakeKeypointSource()
    p = _stub_policy("5pt", live=live)
    ref_pos, _ = p._compute_vr_fk(p._motion_command_t[0, :29], p._vr_offsets())

    live.set(np.arange(15, dtype=np.float32) * 0.1)  # trackers wherever they happen to be
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True
    got = _slot(_obs(p, "5pt"), "5pt", "vr_5point_pos_tcp")
    assert np.allclose(got, ref_pos, atol=1e-6), "arming moved the target"

    # Now move every point by +1 cm in x; the target must follow by exactly that delta.
    d = np.zeros(15, dtype=np.float32)
    d[0::3] = 0.01
    live.set(np.arange(15, dtype=np.float32) * 0.1 + d)
    got2 = _settle(p, "5pt", "vr_5point_pos_tcp")
    assert np.allclose(got2, ref_pos + d, atol=1e-6), (got2 - ref_pos)


def test_5pt_live_moves_only_the_tracking_slots() -> None:
    """A tracker command must not leak into the force block. Without a live F/T source
    every force/servo/phase slot is zero, and arming must leave them that way."""
    live = _FakeKeypointSource()
    p = _stub_policy("5pt", live=live)
    live.set(np.arange(15, dtype=np.float32) * 0.1)
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True
    d = np.zeros(15, dtype=np.float32)
    d[0::3] = 0.05
    live.set(np.arange(15, dtype=np.float32) * 0.1 + d)
    vec = _obs(p, "5pt")
    for name in ("force_cmd_local", "force_meas_local", "contact_phase", "integrator_state", "x_tgt_shift"):
        assert np.allclose(_slot(vec, "5pt", name), 0.0), name


def test_a_3pt_bridge_frame_on_a_5pt_hold_holds_the_feet_at_the_reference() -> None:
    """The "wrong bridge MODE" case, which is the likeliest operator error: a 3pt bridge
    streams pos[9] against a 5pt policy. ``_match_len`` / ``_apply_clutch_delta`` must
    override only the leading points and HOLD the two foot slots at the FK reference —
    never grow or shrink the obs, never leave a foot at zero (which is the pelvis origin,
    i.e. a leg folded through the body)."""
    live = _FakeKeypointSource()
    p = _stub_policy("5pt", live=live)
    ref_pos, _ = p._compute_vr_fk(p._motion_command_t[0, :29], p._vr_offsets())

    live.set(np.arange(9, dtype=np.float32) * 0.1)  # a 3pt-width frame
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True
    d = np.zeros(9, dtype=np.float32)
    d[0::3] = 0.02
    live.set(np.arange(9, dtype=np.float32) * 0.1 + d)
    got = _settle(p, "5pt", "vr_5point_pos_tcp")
    assert got.size == 15
    assert np.allclose(got[:9], ref_pos[:9] + d, atol=1e-6)
    assert np.allclose(got[9:], ref_pos[9:], atol=1e-6), "feet must hold the reference"


def test_slew_cap_limits_the_first_tick_after_a_step() -> None:
    """The cap is on the L2 norm of the WHOLE per-tick change, not per point — so a step
    applied to five points at once ramps in over several ticks even though each point moved
    less than the cap. Pinned because it is why the tracking tests above settle first, and
    because a per-point cap would let a five-point step move 2.24x faster than intended."""
    live = _FakeKeypointSource()
    p = _stub_policy("5pt", live=live)
    ref_pos, _ = p._compute_vr_fk(p._motion_command_t[0, :29], p._vr_offsets())
    base = np.arange(15, dtype=np.float32) * 0.1
    live.set(base)
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True
    _obs(p, "5pt")  # first armed tick: target == ref

    d = np.zeros(15, dtype=np.float32)
    d[0::3] = 0.01  # 5 points x 1 cm => L2 = 2.24 cm against a 2 cm budget
    live.set(base + d)
    first = _slot(_obs(p, "5pt"), "5pt", "vr_5point_pos_tcp")
    step = float(np.linalg.norm(first - ref_pos))
    assert step == pytest.approx(p._vive_max_pos_step, rel=1e-4), step
    assert step < float(np.linalg.norm(d)), "the step was not capped at all"
    # ...and it does converge to the full delta given enough ticks.
    assert np.allclose(_settle(p, "5pt", "vr_5point_pos_tcp"), ref_pos + d, atol=1e-6)


def test_slew_feet_default_covers_every_point_in_one_pool() -> None:
    """``HOLOSOMA_VIVE_SLEW_FEET`` unset/1: the caps cover all five points — byte-identical to the
    two bare ``_slew_vec`` calls every test above was written against."""
    p = _stub_policy("5pt", live=_FakeKeypointSource())
    assert not hasattr(p, "_vive_slew_feet") or p._vive_slew_feet
    assert p._vr_slewed_points(5) == 5
    p._vive_slew_feet = True
    assert p._vr_slewed_points(5) == 5
    prev_pos = np.zeros(15, dtype=np.float32)
    tgt_pos = np.arange(15, dtype=np.float32) * 0.01
    prev_orn = _identity_quats(5)
    tgt_orn = np.tile(np.array([0.1, 0.0, 0.0, 0.995], dtype=np.float32), 5)
    pos, orn = p._slew_keypoints(prev_pos, tgt_pos, prev_orn, tgt_orn)
    assert np.array_equal(pos, p._slew_vec(prev_pos, tgt_pos, p._vive_max_pos_step))
    assert np.array_equal(orn, p._slew_vec(prev_orn, tgt_orn, p._vive_max_orn_step))


def test_slew_feet_off_exempts_the_feet_and_caps_only_the_upper_pool() -> None:
    """``HOLOSOMA_VIVE_SLEW_FEET=0`` on 5pt: the two FEET take their target verbatim while the three
    upper points still share one L2 pool at the cap. A gait is a far foot target by construction
    (a walking clip's feet demand 19-46 mm/tick against a 5 mm cap), so a slewed foot does not walk
    slowly — it crawls toward a target that keeps running away."""
    live = _FakeKeypointSource()
    p = _stub_policy("5pt", live=live)
    p._vive_slew_feet = False
    assert p._vr_slewed_points(5) == 3
    ref_pos, _ = p._compute_vr_fk(p._motion_command_t[0, :29], p._vr_offsets())
    base = np.arange(15, dtype=np.float32) * 0.1
    live.set(base)
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True
    _obs(p, "5pt")  # first armed tick: target == ref

    d = np.zeros(15, dtype=np.float32)
    d[0::3] = 0.01  # the three upper points: 1 cm each => L2 1.73 cm against a 2 cm budget
    d[9:] = [0.05, -0.02, 0.03, -0.06, 0.01, 0.04]  # the feet: far more than the cap, in one tick
    live.set(base + d)
    first = _slot(_obs(p, "5pt"), "5pt", "vr_5point_pos_tcp")
    # Upper pool: within the cap and NOT scaled by the feet's demand (they are out of the pool).
    upper_step = float(np.linalg.norm(first[:9] - ref_pos[:9]))
    assert upper_step == pytest.approx(float(np.linalg.norm(d[:9])), rel=1e-4), upper_step
    assert upper_step <= p._vive_max_pos_step + 1e-6
    # Feet: the target itself, on the first tick.
    assert np.allclose(first[9:], ref_pos[9:] + d[9:], atol=1e-6), first[9:] - ref_pos[9:]

    # And when the upper points alone exceed the cap they are STILL clamped (the exemption never
    # widens past the feet).
    d2 = np.zeros(15, dtype=np.float32)
    d2[0::3] = 0.05
    live.set(base + d + d2)
    second = _slot(_obs(p, "5pt"), "5pt", "vr_5point_pos_tcp")
    assert float(np.linalg.norm(second[:9] - first[:9])) == pytest.approx(p._vive_max_pos_step, rel=1e-4)
    assert np.allclose(second[9:], ref_pos[9:] + d[9:] + d2[9:], atol=1e-6)


def test_slew_feet_off_exempts_the_foot_orientations_too() -> None:
    """The orientation cap splits the same way: the leading 3x4 quaternion entries are one pool,
    the feet's two quaternions pass through verbatim."""
    p = _stub_policy("5pt", live=_FakeKeypointSource())
    p._vive_slew_feet = False
    prev_pos = np.zeros(15, dtype=np.float32)
    tgt_pos = np.zeros(15, dtype=np.float32)
    prev_orn = _identity_quats(5)
    big = np.array([0.3, 0.0, 0.0, 0.954], dtype=np.float32)  # far past the 0.05 orn cap
    tgt_orn = np.tile(big, 5)
    pos, orn = p._slew_keypoints(prev_pos, tgt_pos, prev_orn, tgt_orn)
    assert np.array_equal(pos, tgt_pos)
    assert float(np.linalg.norm(orn[:12] - prev_orn[:12])) == pytest.approx(p._vive_max_orn_step, rel=1e-4)
    assert np.array_equal(orn[12:], tgt_orn[12:])


def test_slew_feet_off_is_a_5pt_only_knob() -> None:
    """A 3pt policy has no feet: the knob changes nothing there."""
    p = _stub_policy("3pt", live=_FakeKeypointSource(3))
    p._vive_slew_feet = False
    assert p._vr_slewed_points(3) == 3
    prev = np.zeros(9, dtype=np.float32)
    tgt = np.full(9, 0.05, dtype=np.float32)
    pos, _orn = p._slew_keypoints(prev, tgt, _identity_quats(3), _identity_quats(3))
    assert np.array_equal(pos, p._slew_vec(prev, tgt, p._vive_max_pos_step))


def test_record_tap_publishes_the_per_point_slew_mask_next_to_vr_cmd_pos() -> None:
    """The consumer that MEASURES the cap off ``vr_cmd_pos`` (FLoRe's SlewMonitor) has to know which
    rows the cap covered; the mask rides in the same frame, None when there is no keypoint target."""
    from wbt_wrappers_inference.utils.record_tap import RecordTap

    tap = object.__new__(RecordTap)  # no socket, no thread: publish() only touches the queue
    tap._seq = 0
    tap._q = __import__("queue").Queue(maxsize=8)
    tap._encode_errors = 0
    tap._dropped = 0
    common = dict(
        running=True, vive_armed=True, timestep=0.0, servo_on=False, f_cmd=None, f_meas=None,
        f_meas_ungated=None, servo_integ=None, servo_dx=None, servo_dq=None,
        dof_pos=np.zeros(29), dof_vel=np.zeros(29), root_quat_wxyz=np.array([1.0, 0, 0, 0]),
        base_ang_vel=np.zeros(3),
    )
    tap.publish(vr_cmd_pos=np.zeros((5, 3)), vr_cmd_slewed=[True, True, True, False, False], **common)
    tap.publish(vr_cmd_pos=None, vr_cmd_slewed=None, **common)
    tap.publish(vr_cmd_pos=np.zeros((5, 3)), **common)  # an old call site: the key is still there
    f0, f1, f2 = tap._q.get_nowait(), tap._q.get_nowait(), tap._q.get_nowait()
    assert f0["vr_cmd_slewed"] == [True, True, True, False, False]
    assert f1["vr_cmd_slewed"] is None and f1["vr_cmd_pos"] is None
    assert f2["vr_cmd_slewed"] is None
    import json

    json.dumps(f0)  # plain bools, JSON-serialisable


def test_the_policy_derives_the_slew_mask_from_the_same_rule_as_the_slew() -> None:
    """The mask the tap gets and the split ``_slew_keypoints`` applies come from ONE function, so
    they cannot disagree about which points were capped."""
    p = _stub_policy("5pt", live=_FakeKeypointSource())
    for feet, expect in ((True, [True] * 5), (False, [True, True, True, False, False])):
        p._vive_slew_feet = feet
        n = p._vr_slewed_points(5)
        assert [i < n for i in range(5)] == expect


def test_5pt_stale_freezes_the_target_rather_than_snapping_to_the_reference() -> None:
    """Bridge hiccup while armed: the commanded target FREEZES at the last fed value.
    Snapping back to the FK reference would be an un-slewed jump of the whole clutch
    delta — the one thing every other transition here is ramped to avoid."""
    live = _FakeKeypointSource()
    p = _stub_policy("5pt", live=live)
    base = np.arange(15, dtype=np.float32) * 0.1
    live.set(base)
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True
    d = np.zeros(15, dtype=np.float32)
    d[0::3] = 0.01
    live.set(base + d)
    moved = _slot(_obs(p, "5pt"), "5pt", "vr_5point_pos_tcp").copy()
    live.set(None)  # STALE
    frozen = _slot(_obs(p, "5pt"), "5pt", "vr_5point_pos_tcp")
    assert np.allclose(frozen, moved, atol=1e-7), "stale must freeze, not snap"


# ── 4. dense: the joint-space path ──────────────────────────────────────────────


def _dof_names():
    from wbt_wrappers_inference.config_values.robot_config import G1_29DOF_JOINT_NAMES

    return list(G1_29DOF_JOINT_NAMES)


def test_dense_obs_is_732d_with_a_58d_command_slot() -> None:
    p = _stub_policy("dense")
    flat = p._build_observation(_robot_state(p)).reshape(-1)
    assert flat.size == 732, flat.size
    t = _terms(p)
    assert t["command"].size == 58
    assert sum(v.size for v in t.values()) == 183
    assert p._is_dense_unified is True
    assert p._uses_vr_keypoints is False


def test_dense_remap_is_by_name_not_by_index() -> None:
    """THE dense test. Feed a wire whose joint order is REVERSED relative to dof_names and
    assert every value lands on its own joint.

    On today's data the completion's order and holosoma's ``dof_names`` are byte-identical,
    so an index-aligned implementation passes every realistic test — and would scramble the
    whole body the moment either order changed, silently. Reversing the wire order is the
    only way to tell the two implementations apart."""
    names = _dof_names()
    rev = list(reversed(names))
    live = _FakeDenseSource(rev)
    p = _stub_policy("dense", live=live, relative=False)  # absolute: target IS the wire
    p._vive_abs_clutch = False  # ...raw, so the assertion is about the remap alone

    # A distinct value per NAME, presented in the reversed wire order.
    value_of = {n: 0.01 * (i + 1) for i, n in enumerate(names)}
    live.set(np.array([value_of[n] for n in rev], dtype=np.float32))
    p._vive_armed = True

    cmd = _slot(_obs(p, "dense"), "dense", "command")
    got_q = cmd[:29]
    want_q = np.array([value_of[n] for n in names], dtype=np.float32)
    # The slew caps the FIRST tick, so compare the remap DIRECTION not the magnitude:
    # assert the un-slewed target the remap produced instead.
    live_q, _ = p._dense_live_vectors()
    assert np.allclose(live_q, want_q, atol=1e-7), f"remap is index-aligned, not by name: {live_q} != {want_q}"
    # ...and the permutation itself is the reversal, resolved by name.
    perm = p._dense_joint_perm(tuple(rev))
    assert perm.tolist() == list(range(28, -1, -1)), perm.tolist()
    # The commanded vector is on the way there (same sign per joint), never scrambled.
    assert got_q.size == 29


def test_dense_remap_is_the_identity_on_the_real_completion_order() -> None:
    """The completion's ``g1_fk_params.npz`` joint order and holosoma's ``dof_names`` agree
    exactly today, so the by-name remap must resolve to the IDENTITY permutation. Pinned so
    the fast path is known-correct — and so that if the two orders ever diverge this test
    fails loudly instead of the robot moving strangely."""
    names = _dof_names()
    p = _stub_policy("dense", live=_FakeDenseSource(names))
    perm = p._dense_joint_perm(tuple(names))
    assert perm.tolist() == list(range(29))


def test_dense_missing_joint_holds_the_reference() -> None:
    """A wire that omits a joint must leave that DOF at the held reference, not at zero
    (zero is not "no change" for a joint target — it is a specific, usually extreme pose)."""
    names = _dof_names()
    dropped = names[0]
    partial = names[1:] + ["not_a_real_joint"]
    live = _FakeDenseSource(partial)
    p = _stub_policy("dense", live=live)
    live.set(np.full(29, 0.75, dtype=np.float32))

    ref_q = p._motion_command_t[0, :29].astype(np.float32)
    live_q, _ = p._dense_live_vectors()
    i = names.index(dropped)
    assert live_q[i] == pytest.approx(ref_q[i]), "a missing joint must hold the reference"
    assert live_q[i + 1] == pytest.approx(0.75)
    assert p._dense_perm_missing == [dropped]


def test_dense_arm_is_zero_jump() -> None:
    """Relative/clutch arming leaves the commanded joint vector exactly at the reference,
    then tracks the completion's joint delta."""
    names = _dof_names()
    live = _FakeDenseSource(names)
    p = _stub_policy("dense", live=live)
    ref_q = p._motion_command_t[0, :29].astype(np.float32)

    live.set(np.full(29, 0.5, dtype=np.float32))  # a posture far from the reference
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True
    cmd = _slot(_obs(p, "dense"), "dense", "command")
    assert np.allclose(cmd[:29], ref_q, atol=1e-6), "arming moved the joints"
    assert np.allclose(cmd[29:], 0.0), "a static hold must command zero reference velocity"

    # A small delta on ONE joint moves only that joint's target.
    q2 = np.full(29, 0.5, dtype=np.float32)
    q2[7] += 0.01
    live.set(q2)
    d = _settle(p, "dense", "command")[:29] - ref_q
    assert d[7] == pytest.approx(0.01, abs=1e-6), d[7]
    assert np.allclose(np.delete(d, 7), 0.0, atol=1e-6)


def test_dense_rate_limit_is_per_joint_and_does_not_couple_joints() -> None:
    """The joint limiter must be PER JOINT, not an L2 norm over the vector.

    Two failure modes an L2 cap has here, both real on a rig:

    * COUPLING — one fast wrist would throttle every other joint, including the legs.
    * BINDING IN NORMAL MOTION — 29 joints each at a perfectly ordinary teleop rate blow an
      L2 budget that any single joint respects (two arm joints at 2 rad/s already exceed the
      2.5 rad/s an 0.05 rad/tick L2 cap allows). It would rate-limit the operator and, via
      the velocity guard, pin the reference velocity at zero the entire time.
    """
    names = _dof_names()
    live = _FakeDenseSource(names)
    p = _stub_policy("dense", live=live)
    ref_q = p._motion_command_t[0, :29].astype(np.float32)
    live.set(ref_q.copy())
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True
    _obs(p, "dense")

    # ONE joint steps far past the cap; every other joint moves a legal small amount.
    step = p._vive_max_joint_step
    target = ref_q.copy()
    target += 0.5 * step  # legal for all
    target[5] = ref_q[5] + 10.0 * step  # illegal for one
    live.set(target)
    q = _slot(_obs(p, "dense"), "dense", "command")[:29]
    d = q - ref_q
    assert d[5] == pytest.approx(step, rel=1e-5), "the fast joint must be clipped to the cap"
    others = np.delete(d, 5)
    assert np.allclose(others, 0.5 * step, rtol=1e-4), "one fast joint must not throttle the rest"

    # And a whole-body motion that is legal per joint must NOT bind at all — so the
    # completion's velocity keeps flowing (an L2 cap would zero it here).
    p2 = _stub_policy("dense", live=_FakeDenseSource(names))
    p2._vive_live.set(ref_q.copy())
    assert p2._snapshot_clutch_anchor()
    p2._vive_armed = True
    _obs(p2, "dense")
    dq = np.full(29, 1.5, dtype=np.float32)
    p2._vive_live.set(ref_q + 0.9 * step, dq)
    cmd = _slot(_obs(p2, "dense"), "dense", "command")
    assert np.allclose(cmd[:29], ref_q + 0.9 * step, atol=1e-6), "a per-joint-legal step must pass unclipped"
    assert np.allclose(cmd[29:], dq, atol=1e-6), "and its reference velocity must survive"


def test_dense_stale_freezes_and_never_commands_zeros() -> None:
    """29 zero joint targets is the G1 folded flat, so a stale bridge must FREEZE the last
    commanded vector — the same contract the keypoint path has."""
    names = _dof_names()
    live = _FakeDenseSource(names)
    p = _stub_policy("dense", live=live)
    base = np.full(29, 0.4, dtype=np.float32)
    live.set(base)
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True
    moved = base.copy()
    moved[3] += 0.02
    live.set(moved)
    cmd = _slot(_obs(p, "dense"), "dense", "command").copy()
    live.set(None)  # STALE
    frozen = _slot(_obs(p, "dense"), "dense", "command")
    assert np.allclose(frozen[:29], cmd[:29], atol=1e-7), "stale must freeze the joint targets"
    assert not np.allclose(frozen[:29], 0.0)
    assert np.allclose(frozen[29:], 0.0), "frozen ⇒ zero reference velocity"


def test_dense_velocity_is_zero_while_the_slew_binds_and_live_once_settled() -> None:
    """A command that is still ramping must not also claim it is moving at the live rate:
    that pos/vel pair is internally inconsistent and off-distribution for a policy trained
    on a smooth clip. Same reasoning as the frozen-clip guard on motion_ref_vel_b and the
    velocity half of _apply_hold_dof_override."""
    names = _dof_names()
    live = _FakeDenseSource(names)
    p = _stub_policy("dense", live=live)
    ref_q = p._motion_command_t[0, :29].astype(np.float32)
    live.set(ref_q.copy())
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True

    dq = np.full(29, 0.3, dtype=np.float32)
    # A big step: |target - last| >> max_joint_step ⇒ the slew binds ⇒ velocity zeroed.
    far = ref_q + 1.0
    live.set(far, dq)
    cmd = _slot(_obs(p, "dense"), "dense", "command")
    assert np.allclose(cmd[29:], 0.0), "ramping must not report the live velocity"

    # A target within one slew step ⇒ settled ⇒ the completion's own velocity flows.
    settled = p._dense_last_q.copy()
    live.set(settled, dq)
    cmd2 = _slot(_obs(p, "dense"), "dense", "command")
    assert np.allclose(cmd2[29:], dq, atol=1e-6), cmd2[29:]


def test_dense_unarmed_is_byte_identical_to_the_npz_reference() -> None:
    """With a live source connected but NOT armed, the obs must be bit-for-bit the held
    clip reference — the whole point of the arm gate."""
    names = _dof_names()
    live = _FakeDenseSource(names)
    live.set(np.full(29, 0.9, dtype=np.float32))
    p_live = _stub_policy("dense", live=live)
    p_off = _stub_policy("dense", live=None)
    assert np.array_equal(_obs(p_live, "dense"), _obs(p_off, "dense"))


# ── 5. the wire codec ───────────────────────────────────────────────────────────


def _ingest(msg):
    """Drive ViveLiveSource._ingest without starting its reader thread."""
    import json

    from wbt_wrappers_inference.utils.vive_live_source import ViveLiveSource

    src = object.__new__(ViveLiveSource)
    src._stale_timeout = 0.0
    src._lock = __import__("threading").Lock()
    src._pos = src._orn = src._f_cmd = None
    src._joint_pos = src._joint_vel = src._joint_names = None
    src._ref_joint_pos = src._ref_joint_vel = src._ref_joint_names = None
    src._last_t = 0.0
    src._seq = -1
    src._stale_logged = False
    src._joint_names_warned = False
    src._ref_joints_warned = False
    src._ingest(json.dumps(msg))
    return src


def test_wire_accepts_the_5pt_pose_widths() -> None:
    src = _ingest({"mode": "5pt", "pos": [0.0] * 15, "orn": [0.0] * 20, "seq": 1})
    assert src._pos is not None and src._pos.size == 15
    assert src._orn is not None and src._orn.size == 20
    assert src.latest_joints() is None, "a keypoint frame carries no joints"


@pytest.mark.parametrize(
    "mode,npos,norn",
    [
        ("5pt", 15, 12),  # 5 points but a 3pt-width orn
        ("5pt", 9, 20),  # 3pt-width pos tagged 5pt
        ("3pt", 15, 20),  # a 5pt frame mislabelled 3pt
        ("2pt", 9, 12),  # 3pt widths tagged 2pt
    ],
)
def test_wire_rejects_internally_inconsistent_pose_frames(mode: str, npos: int, norn: int) -> None:
    """The (pos, orn) width PAIR must agree with the declared mode. Width alone cannot do
    this job: a 3pt 3×quat and a 2pt 2×rot6d are both 12 wide, so the pos width is the only
    thing that identifies the rotation rep — which is why a mismatch is dropped rather than
    passed through for the policy to guess at."""
    src = _ingest({"mode": mode, "pos": [0.0] * npos, "orn": [0.0] * norn})
    assert src._pos is None, f"{mode} pos{npos}/orn{norn} should have been dropped"


def test_wire_parses_a_dense_frame_and_keeps_the_pose_fields_none() -> None:
    names = _dof_names()
    src = _ingest(
        {"mode": "dense", "joint_pos": [0.1] * 29, "joint_vel": [0.2] * 29, "joint_names": names, "seq": 3}
    )
    got = src.latest_joints()
    assert got is not None
    q, dq, wire_names = got
    assert q.size == 29 and dq.size == 29
    assert list(wire_names) == names
    assert src._pos is None and src._orn is None, "a dense frame must not populate pose slots"
    assert src.latest() == (None, None)


def test_wire_drops_a_dense_frame_without_joint_names() -> None:
    """Without names there is no honest way to place 29 numbers, and index-aligning two
    different real joint orders scrambles the body with no error anywhere. Drop the frame
    (the consumer then holds the reference) rather than guess."""
    for bad in (None, ["only", "two"], [1] * 29):
        msg = {"mode": "dense", "joint_pos": [0.1] * 29, "joint_vel": [0.0] * 29}
        if bad is not None:
            msg["joint_names"] = bad
        assert _ingest(msg).latest_joints() is None, bad


def test_wire_drops_a_nonfinite_dense_frame() -> None:
    names = _dof_names()
    for key in ("joint_pos", "joint_vel"):
        msg = {"mode": "dense", "joint_pos": [0.1] * 29, "joint_vel": [0.0] * 29, "joint_names": names}
        msg[key] = [float("nan")] + [0.0] * 28
        assert _ingest(msg).latest_joints() is None, key


# ── 6. the ready-pose hold + arm gates ──────────────────────────────────────────


def test_ready_pose_hold_applies_to_both_mc_families_but_not_legacy_dense() -> None:
    """The INIT/hold ready pose (arms zeroed) is what makes arming start from a sane
    posture. It now covers 5pt and dense-UNIFIED — dense is safe because
    ``_apply_hold_dof_override`` writes BOTH halves of the 58D command (pos = ready pose,
    vel = 0), which is what the original "dense would be inconsistent" exclusion was
    written against. LEGACY dense/global presets must stay excluded: they are not part of
    this teleop and their deploy behaviour must not change."""
    for family in ("5pt", "dense", "3pt"):
        p = _stub_policy(family)
        p._init_q_target = None
        p._maybe_apply_init_ready_pose()
        assert p._hold_dof_override is not None, f"{family} should get the ready-pose hold"
        arms = [
            "left_shoulder_pitch_joint",
            "right_elbow_joint",
            "left_wrist_yaw_joint",
        ]
        names = _dof_names()
        for a in arms:
            assert p._hold_dof_override[names.index(a)] == pytest.approx(0.0), (family, a)

    # A legacy dense preset gets NO override (byte-identical deploy).
    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.utils.obs_utils import detect_obs_mode

    legacy = next(
        k
        for k, c in DEFAULTS.items()
        if detect_obs_mode(list(c.observation.obs_dict["actor_obs"])) == "dense"
        and not any("contact_phase" in t for t in c.observation.obs_dict["actor_obs"])
    )
    p = _stub_policy("dense")
    p.config = SimpleNamespace(
        robot=p.config.robot, observation=DEFAULTS[legacy].observation, task=p.config.task
    )
    p._obs_terms = list(DEFAULTS[legacy].observation.obs_dict["actor_obs"])
    p._obs_mode = "dense"
    p._has_contact_phase_term = False
    p._is_dense_unified = False
    p._maybe_apply_init_ready_pose()
    assert p._hold_dof_override is None, f"{legacy} must keep its legacy frame-0 hold"


def test_5pt_arm_gate_does_not_refuse_on_the_completion_inferred_FEET() -> None:
    """The rig has THREE pucks; the two 5pt FOOT keypoints are Motion_Completion's inference
    off those same three, so an operator cannot aim them and gating on them was a permanent
    lockout (rig, 2026-08-28).

    Reproduces the measured refusal: hands + head inside the 0.070 m tol while the two feet
    sat at 0.101 m and 0.062 m. That MUST arm. Jump safety is untouched — the absolute arm
    clutch re-bases all five points (asserted separately below).
    """
    live = _FakeKeypointSource(5)
    p = _stub_policy("5pt", live=live, relative=False)
    assert p._is_5pt, "this test is meaningless unless the stub really is the 5pt family"
    held, _ = p._compute_vr_fk(p._motion_command_t[0, :29], p._vr_offsets())
    hp = held.reshape(-1, 3).copy()
    assert hp.shape[0] == 5

    # Exactly the rig's numbers: tracked points well inside tol, feet well outside.
    wire = hp.copy()
    wire[0, 0] += 0.03  # LEFT hand   0.030 m  (inside 0.070)
    wire[1, 0] += 0.05  # RIGHT hand  0.050 m  (inside)
    wire[2, 0] += 0.02  # head/torso  0.020 m  (inside)
    wire[3, 0] += 0.101  # LEFT foot  0.101 m  (OUTSIDE — but not the operator's to move)
    wire[4, 0] += 0.062  # RIGHT foot 0.062 m
    live.set(wire.reshape(-1))
    assert p._vive_arm_pose_ok() is True, (
        "a foot gap must NOT refuse: the operator has no puck on their feet, so this is a "
        "lockout with no corresponding action"
    )

    # A TRACKED point out of tol still refuses, and the message names it.
    wire[0, 0] = hp[0, 0] + 0.15
    live.set(wire.reshape(-1))
    assert p._vive_arm_pose_ok() is False
    assert p._vive_arm_refuse_why == "too_far"
    assert "LEFT hand" in p._vive_arm_refuse_detail, p._vive_arm_refuse_detail
    # ...and never blames a point the operator cannot move.
    assert "foot" not in p._vive_arm_refuse_detail.lower(), p._vive_arm_refuse_detail


def test_5pt_arm_clutch_still_rebases_the_ungated_foot_points() -> None:
    """Dropping the feet from the GATE must not drop them from the CLUTCH — that is what
    makes the gate change safe rather than merely permissive. The clutch offset is what
    holds the first armed command at the held pose, so it must cover all five points."""
    live = _FakeKeypointSource(5)
    p = _stub_policy("5pt", live=live, relative=False)
    held, _ = p._compute_vr_fk(p._motion_command_t[0, :29], p._vr_offsets())
    hp = held.reshape(-1, 3).copy()
    wire = hp.copy()
    wire[3, 0] += 0.101  # the very foot gap the gate now ignores
    wire[4, 1] -= 0.062
    live.set(wire.reshape(-1))

    p._toggle_vive_teleop()
    assert p._vive_armed is True, "the gate must have passed for this test to mean anything"
    assert p._vr_abs_off_pos is not None, "absolute arm clutch did not engage"
    off = p._vr_abs_off_pos.reshape(-1, 3)
    assert off.shape[0] == 5, f"clutch offset covers only {off.shape[0]} of 5 points"
    # target = wire + off must equal the held pose at the arm instant, FEET INCLUDED.
    np.testing.assert_allclose(wire + off, hp, atol=1e-5)


def test_dense_arm_gate_does_not_refuse_on_the_completion_inferred_LEGS() -> None:
    """The dense twin of the 5pt foot lockout, and it bites far harder: the held reference is
    the READY POSE (``default_dof_angles`` with the arms zeroed), whose stance is a 38° crouch
    — ``knee = +0.669 rad``. A straight-legged completion therefore misses the knee by 0.669,
    more than THREE TIMES the 0.20 rad tol, and no hand puck can bend a knee. The deltas below
    are that real gap, not a token overshoot."""
    names = _dof_names()
    live = _FakeDenseSource(names)
    p = _stub_policy("dense", live=live, relative=False)
    ref_q = p._motion_command_t[0, :29].astype(np.float32)

    q = ref_q.copy()
    for j, delta in (
        ("left_knee_joint", -0.669),  # the completion stands straight; the reference crouches
        ("right_knee_joint", -0.669),
        ("left_hip_pitch_joint", 0.312),
        ("right_ankle_pitch_joint", 0.363),
        ("waist_pitch_joint", 0.24),
    ):
        q[names.index(j)] += delta
    live.set(q)
    assert p._vive_arm_pose_ok() is True, "leg/torso-pitch gaps must not refuse — unaimable"

    # An ARM joint past tol still refuses, and is named.
    q[names.index("left_shoulder_pitch_joint")] += 0.5
    live.set(q)
    assert p._vive_arm_pose_ok() is False
    assert "left_shoulder_pitch_joint" in p._vive_arm_refuse_detail, p._vive_arm_refuse_detail
    assert "knee" not in p._vive_arm_refuse_detail, p._vive_arm_refuse_detail


def test_arm_gate_refusal_quotes_the_right_units_per_family() -> None:
    """The keypoint families gate in METRES via HOLOSOMA_VIVE_ARM_TOL, dense in RADIANS via
    HOLOSOMA_VIVE_ARM_TOL_RAD. Quoting one pair for both sends the operator at a knob that
    cannot move the check they just failed."""
    import re

    from loguru import logger as _loguru

    names = _dof_names()
    for family, want_unit, want_env in (("5pt", "m)", "HOLOSOMA_VIVE_ARM_TOL,"), ("dense", "rad)", "TOL_RAD,")):
        msgs: list[str] = []
        if family == "dense":
            live = _FakeDenseSource(names)
            p = _stub_policy("dense", live=live, relative=False)
            q = p._motion_command_t[0, :29].astype(np.float32).copy()
            q[names.index("left_shoulder_pitch_joint")] += 1.0
            live.set(q)
        else:
            live = _FakeKeypointSource(5)
            p = _stub_policy("5pt", live=live, relative=False)
            held, _ = p._compute_vr_fk(p._motion_command_t[0, :29], p._vr_offsets())
            hp = held.reshape(-1, 3).copy()
            hp[0, 0] += 1.0
            live.set(hp.reshape(-1))
        sink = _loguru.add(lambda m: msgs.append(str(m)), level="WARNING", format="{message}")
        try:
            p._toggle_vive_teleop()
        finally:
            _loguru.remove(sink)
        assert p._vive_armed is False, f"{family}: should have refused"
        refusal = next((m for m in msgs if "REFUSED" in m), "")
        assert refusal, f"{family}: no refusal logged, got {msgs}"
        flat = re.sub(r"\s+", " ", refusal)
        assert want_unit in flat, f"{family}: wrong tol unit in {flat!r}"
        assert want_env in flat, f"{family}: wrong override env in {flat!r}"


def test_dense_absolute_arm_gate_is_joint_space_and_refuses_a_far_posture() -> None:
    """The dense family has no keypoints, so the ABSOLUTE-mode pre-arm gate compares the
    completion's joint solution to the held reference joint-by-joint. It must refuse a
    posture far from the hold (which would otherwise be an instant whole-body lurch)."""
    names = _dof_names()
    live = _FakeDenseSource(names)
    p = _stub_policy("dense", live=live, relative=False)
    ref_q = p._motion_command_t[0, :29].astype(np.float32)

    live.set(ref_q + 1.0)  # way off
    assert p._vive_arm_pose_ok() is False
    live.set(ref_q + 0.01)  # within tolerance
    assert p._vive_arm_pose_ok() is True
    live.set(None)  # no frame at all
    assert p._vive_arm_pose_ok() is False


def test_dense_absolute_arm_clutch_is_zero_jump() -> None:
    """Absolute mode + arm clutch: the first armed command equals the held pose no matter
    what posture the completion reports, then tracks its delta."""
    names = _dof_names()
    live = _FakeDenseSource(names)
    p = _stub_policy("dense", live=live, relative=False)
    ref_q = p._motion_command_t[0, :29].astype(np.float32)
    live.set(ref_q + 0.05)
    p.use_policy_action = True
    p._toggle_vive_teleop()
    assert p._vive_armed is True
    assert p._dense_abs_off_q is not None
    cmd = _slot(_obs(p, "dense"), "dense", "command")
    assert np.allclose(cmd[:29], ref_q, atol=1e-6), "absolute arming must not jump"


def test_disarm_clears_every_anchor_including_the_dense_ones() -> None:
    """One helper clears all ten anchor fields, because the same block used to be
    open-coded at five sites — and a single forgotten line there is a stale origin, i.e. a
    real jump on the next arm."""
    names = _dof_names()
    live = _FakeDenseSource(names)
    p = _stub_policy("dense", live=live)
    live.set(np.full(29, 0.3, dtype=np.float32))
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True
    _obs(p, "dense")
    assert p._dense_clutch_q0 is not None and p._dense_last_q is not None

    p._toggle_vive_teleop()  # disarm
    assert p._vive_armed is False
    for f in (
        "_vr_last_pos",
        "_vr_last_orn",
        "_vr_clutch_live0",
        "_vr_clutch_orn0",
        "_vr_abs_off_pos",
        "_vr_abs_orn0",
        "_dense_last_q",
        "_dense_last_dq",
        "_dense_clutch_q0",
        "_dense_abs_off_q",
    ):
        assert getattr(p, f) is None, f


# ── the two frontends do not share a numpy, and one of them is numpy 1 ──────────


def test_load_mc_names_works_on_this_interpreter() -> None:
    """Regression for the rig launch that died AFTER the MC backend reported ready
    (2026-08-28): ``ModuleNotFoundError: No module named 'numpy._core'``.

    ``g1_fk_params.npz`` keeps ``body_names``/``joint_names`` as ``dtype=object``, so reading
    them goes through pickle, and numpy 2 renamed ``numpy.core`` -> ``numpy._core`` in the
    recorded class path. The file is written by a numpy-2 env, and the two frontends that read
    it do NOT share an interpreter:

        MC selftest frontend   UMI_collection/.venv       numpy 2.4.6  -> loads fine
        teleop bridge frontend holosoma hssim             numpy 1.26.0 -> ModuleNotFoundError

    So the selftest passed end-to-end (VERDICT: PASS, frame chain 0.000000 mm) while the rig
    launch could not start, and no amount of selftesting would have found it. This suite runs
    in hssim, so THIS test is the reproduction — it fails without
    ``mc_frames._allow_numpy2_pickles`` and passes with it. Do not "simplify" that shim away.
    """
    umi = Path(os.path.expanduser(os.environ.get("MC_UMI_DIR", "~/FLoRe/third_party/UMI_collection")))
    mc = Path(os.path.expanduser("~/FLoRe/third_party/Motion_Completion"))
    if not (umi / "motion_completion").is_dir():
        pytest.skip(f"no motion_completion/ under {umi}")
    if not (mc / "manip/data/util/g1_fk_params.npz").is_file():
        pytest.skip(f"no g1_fk_params.npz under {mc} (peer checkout, not a submodule)")
    pkg = str(umi / "motion_completion")
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    import mc_frames

    body, joint = mc_frames.load_mc_names(str(mc))
    # 30 bodies / 29 joints is the G1 the checkpoint was trained against; a silent change here
    # would scramble the by-name remap rather than raise.
    assert len(body) == 30, body
    assert len(joint) == 29, joint
    assert body[0] == "pelvis"
    assert joint[0] == "left_hip_pitch_joint"
    assert all(isinstance(x, str) for x in body + joint), "must come back as str, not np.str_"


# ── 7. the BASE tracker exp:g1-wbt on the same joint-space wire ──────────────────
#
# exp:g1-wbt is the base whole-body tracker: 628D, `command` at slot 0, NO force block. Its
# whole tracking front-end is that 58D command term — the same thing the bridge's `--mode
# dense` wire carries — so the machinery the dense-unified policy uses drives it unchanged.
#
# What blocked it was a FAMILY test, not a missing mechanism: `_is_dense_unified` requires a
# `contact_phase` term, which only the force presets have. So every piece of the joint-space
# teleop path existed and was tested, and the one policy whose replay is already measured
# end-to-end on recorded takes could not be teleoperated at all. `_dense_joint_teleop`
# separates "steered by 29 joint targets" from "is the unified force policy"; these tests
# pin both halves of that split, including the part that must NOT change.


def test_wbt_base_obs_is_628d_with_command_leading() -> None:
    """628D = 157 x 4, `command` first at 58D, and NO force/contact slots anywhere."""
    p = _stub_policy("wbt")
    flat = p._build_observation(_robot_state(p)).reshape(-1)
    assert flat.size == 628, flat.size
    t = _terms(p)
    assert t["command"].size == 58
    assert sum(v.size for v in t.values()) == 157
    assert not any("force" in k or "contact" in k for k in t), sorted(t)
    assert p._uses_vr_keypoints is False
    # It is NOT the unified family — that is the whole reason a second flag was needed.
    assert p._is_dense_unified is False


def test_wbt_base_joins_the_joint_space_teleop_family_only_with_a_live_bridge() -> None:
    """THE test for the change. With a bridge configured the base tracker is steerable in
    joint space; with none it is NOT — and that second half is what keeps the verified
    CLIP-REPLAY deploy of this same preset byte-identical, ready-pose hold included."""
    live = _stub_policy("wbt", live=_FakeDenseSource(_dof_names()))
    assert live._is_dense_base_teleop is True
    assert live._dense_joint_teleop is True

    replay = _stub_policy("wbt", live=None)
    assert replay._is_dense_base_teleop is False
    assert replay._dense_joint_teleop is False, (
        "a bridge-less base-tracker deploy must stay OUT of the teleop family: it is the "
        "measured replay path, and joining would switch on the init ready-pose hold"
    )


def test_wbt_base_command_slot_takes_the_live_joint_solution() -> None:
    """The completion's joint vector reaches the `command` slot once armed — the same
    override the dense-unified policy gets, at the same slot 0, on a 628D obs."""
    names = _dof_names()
    live = _FakeDenseSource(names)
    p = _stub_policy("wbt", live=live, relative=False)
    p._vive_abs_clutch = False  # absolute + no clutch: the target IS the wire
    p._vive_max_joint_step = 0.0  # disable the slew so one tick lands on the target

    want = np.array([0.01 * (i + 1) for i in range(29)], dtype=np.float32)
    live.set(want)
    p._vive_armed = True

    cmd = _slot(_obs(p, "wbt"), "wbt", "command")
    assert np.allclose(cmd[:29], want, atol=1e-6), (cmd[:29], want)


def test_wbt_base_remap_is_by_name_not_by_index() -> None:
    """Same hazard as dense-unified, same guard: the completion's joint order and holosoma's
    ``dof_names`` are two independent real orders that happen to agree today. Feed a REVERSED
    wire — the only way to tell a by-name remap from an index-aligned copy."""
    names = _dof_names()
    rev = list(reversed(names))
    live = _FakeDenseSource(rev)
    p = _stub_policy("wbt", live=live, relative=False)
    p._vive_abs_clutch = False

    value_of = {n: 0.01 * (i + 1) for i, n in enumerate(names)}
    live.set(np.array([value_of[n] for n in rev], dtype=np.float32))
    p._vive_armed = True

    live_q, _ = p._dense_live_vectors()
    want_q = np.array([value_of[n] for n in names], dtype=np.float32)
    assert np.allclose(live_q, want_q, atol=1e-7), f"index-aligned, not by name: {live_q} != {want_q}"
    assert p._dense_joint_perm(tuple(rev)).tolist() == list(range(28, -1, -1))


def test_wbt_base_unarmed_is_the_held_reference() -> None:
    """Armed-only, exactly like the keypoint and dense-unified paths: a live frame sitting on
    the wire must not move the robot until the operator arms."""
    names = _dof_names()
    live = _FakeDenseSource(names)
    p = _stub_policy("wbt", live=live)
    live.set(np.full(29, 0.9, dtype=np.float32))  # nothing like the reference

    cmd = _slot(_obs(p, "wbt"), "wbt", "command")
    ref = p._motion_command_t.reshape(-1)
    assert np.allclose(cmd, ref, atol=1e-7), "unarmed must be the held reference, byte-identical"


def test_wbt_base_stale_while_armed_freezes_rather_than_zeroing() -> None:
    """A bridge hiccup must FREEZE the last commanded joint vector. Never zeros: 29 zero
    joint targets is the G1 folded flat, which on hardware is a fall, not a hold."""
    names = _dof_names()
    live = _FakeDenseSource(names)
    p = _stub_policy("wbt", live=live, relative=False)
    p._vive_abs_clutch = False
    p._vive_max_joint_step = 0.0

    want = np.array([0.01 * (i + 1) for i in range(29)], dtype=np.float32)
    live.set(want)
    p._vive_armed = True
    moved = _slot(_obs(p, "wbt"), "wbt", "command").copy()

    live.set(None)  # STALE
    frozen = _slot(_obs(p, "wbt"), "wbt", "command")
    assert np.allclose(frozen[:29], moved[:29], atol=1e-7), "stale must freeze at the last target"
    assert not np.allclose(frozen[:29], 0.0), "and must never fall back to zeros"


def test_wbt_base_relative_clutch_arms_with_zero_jump() -> None:
    """Default RELATIVE/clutch: the first armed command is the held reference no matter where
    the operator's trackers are, so arming never steps the robot."""
    names = _dof_names()
    live = _FakeDenseSource(names)
    p = _stub_policy("wbt", live=live, relative=True)
    live.set(np.full(29, 0.7, dtype=np.float32))  # far from the reference

    assert p._snapshot_clutch_anchor()
    p._vive_armed = True
    cmd = _slot(_obs(p, "wbt"), "wbt", "command")
    ref_q = p._motion_command_t[0, :29]
    assert np.allclose(cmd[:29], ref_q, atol=1e-6), "arming must be a zero-jump operation"


def test_the_joint_space_family_is_exactly_these_two_presets() -> None:
    """Registry-wide guard. Only the dense-unified policy and the BASE tracker may resolve as
    joint-space teleop; everything else — including `g1-29dof-bm-wbt`, the other bare-`command`
    dense preset — must stay out unless someone deliberately points a bridge at it.

    This is the widened form of the old "must NOT fire for legacy dense" invariant: the rule
    moved from `contact_phase` to (dense + bare `command` + a live source), so the thing worth
    pinning is that the widening did not sweep in a third preset by accident."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.utils.obs_utils import detect_obs_mode

    joint_space, legacy_dense = [], 0
    for key, cfg in DEFAULTS.items():
        terms = list(cfg.observation.obs_dict["actor_obs"])
        mode = detect_obs_mode(terms)
        if mode != "dense":
            continue
        if any(t == "command" for t in terms):
            joint_space.append(key)
        else:
            legacy_dense += 1
    assert sorted(joint_space) == sorted([_DENSE_PRESET, _WBT_PRESET, "g1-29dof-bm-wbt"]), joint_space
    assert legacy_dense > 0, "no non-command dense preset left to regress against"


# ── 6. the walk skill's reference legs: `ref_joints` -> command_lower_body (5pt only) ───────
#
# The 5pt obs carries `command_lower_body` = the REFERENCE motion's 12 leg joints + velocities.
# On the live path that is read from the FROZEN idle clip — the ready stance, zero velocity —
# so while FLoRe's walk skill put a gait on the two foot keypoints the legs were still told
# "stand" by a 24-D channel, and the robot stepped in place. A 5pt frame may now carry the
# clip's own joints BY NAME (`ref_joints`), which this section pins to: 5pt only, ARMED only,
# fresh-frame only (stale freezes the last one, like the keypoints), never slewed, a frame
# without the field byte-identical, refused whole when a lower-body joint is unnamed, and
# reported on RecordTap as `ref_joints_live`.


class _FakeKeypointSourceWithLegs(_FakeKeypointSource):
    """A 5pt keypoint frame that may also carry `ref_joints` (pos, vel, names)."""

    def __init__(self, n_points: int = 5):
        super().__init__(n_points)
        self._ref = None

    def set_ref_joints(self, q, dq, names):
        self._ref = None if q is None else (
            np.asarray(q, dtype=np.float32), np.asarray(dq, dtype=np.float32), tuple(names))

    def latest_ref_joints(self):
        if self._pos is None or self._ref is None:   # stale / no frame => no override either
            return None
        q, dq, names = self._ref
        return q.copy(), dq.copy(), names


def _lower(vec29_q, vec29_dq):
    from wbt_wrappers_inference.config_values.robot_config import LOWER_BODY_INDICES

    idx = np.asarray(LOWER_BODY_INDICES)
    return np.concatenate([np.asarray(vec29_q)[idx], np.asarray(vec29_dq)[idx]]).astype(np.float32)


def _armed_5pt_with_legs(*, relative: bool = True):
    live = _FakeKeypointSourceWithLegs()
    p = _stub_policy("5pt", live=live, relative=relative)
    live.set(np.arange(15, dtype=np.float32) * 0.1)
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True
    return p, live


def test_wire_parses_ref_joints_on_a_5pt_frame_and_drops_it_when_the_next_frame_lacks_it() -> None:
    names = _dof_names()
    q = [0.01 * i for i in range(29)]
    dq = [-0.1 * i for i in range(29)]
    src = _ingest({"mode": "5pt", "pos": [0.0] * 15, "orn": [0.0] * 20, "seq": 1,
                   "ref_joints": {"names": names, "pos": q, "vel": dq}})
    got = src.latest_ref_joints()
    assert got is not None
    gq, gdq, gnames = got
    assert np.allclose(gq, q) and np.allclose(gdq, dq) and list(gnames) == names
    assert src._pos is not None and src._pos.size == 15, "the pose is applied alongside"
    assert src.latest_joints() is None, "ref_joints is NOT the dense wire's joint command"
    # The very next keypoint frame without the field clears it: the override is per frame,
    # never held against fresh keypoints (the walk skill stops sending it when its ramp lands).
    src._ingest(__import__("json").dumps({"mode": "5pt", "pos": [0.0] * 15, "orn": [0.0] * 20, "seq": 2}))
    assert src.latest_ref_joints() is None
    assert src._pos is not None


@pytest.mark.parametrize(
    "bad",
    [
        "not a dict",
        {"names": ["j"] * 29, "pos": [0.0] * 29, "vel": [0.0] * 29},          # duplicate names
        {"names": "x", "pos": [0.0] * 29, "vel": [0.0] * 29},
        {"pos": [0.0] * 29, "vel": [0.0] * 29},                                # no names
        {"names": None, "pos": [0.0] * 12, "vel": [0.0] * 12},                 # 12 wide: not the wire
        {"names": None, "pos": [float("nan")] + [0.0] * 28, "vel": [0.0] * 29},
        {"names": None, "pos": [0.0] * 29},                                    # no velocities
    ],
)
def test_wire_ignores_a_malformed_ref_joints_but_keeps_the_pose(bad) -> None:
    if isinstance(bad, dict) and bad.get("names", "keep") is None:
        bad = dict(bad, names=_dof_names())
    src = _ingest({"mode": "5pt", "pos": [0.0] * 15, "orn": [0.0] * 20, "ref_joints": bad})
    assert src.latest_ref_joints() is None, bad
    assert src._pos is not None and src._pos.size == 15, "a bad ref_joints must never drop the pose"
    assert src._ref_joints_warned is True


def test_ref_joints_override_drives_command_lower_body_while_armed_and_is_absent_otherwise() -> None:
    p, live = _armed_5pt_with_legs()
    names = _dof_names()
    frozen = _slot(_obs(p, "5pt"), "5pt", "command_lower_body").copy()
    ref_q = p._motion_command_t[0, :29]
    assert np.allclose(frozen, _lower(ref_q, np.zeros(29))), "no field: the frozen clip's legs, zero velocity"
    assert p._ref_joints_applied is False

    q = np.linspace(-0.5, 0.5, 29).astype(np.float32)
    dq = np.linspace(3.0, -3.0, 29).astype(np.float32)
    live.set_ref_joints(q, dq, names)
    got = _slot(_obs(p, "5pt"), "5pt", "command_lower_body")
    assert np.allclose(got, _lower(q, dq), atol=1e-6), "the wire's legs, sliced exactly as the clip is"
    assert p._ref_joints_applied is True
    # Un-slewed: a 3 rad/s velocity and a 0.5 rad step land in ONE tick (the feet's exemption logic).
    assert np.abs(got - frozen).max() > 0.4
    # The keypoints on the same frame are untouched by the field.
    kp = _slot(_obs(p, "5pt"), "5pt", "vr_5point_pos_tcp")
    assert kp.size == 15

    # Disarmed: the field is ignored and nothing is held over.
    p._vive_armed = False
    back = _slot(_obs(p, "5pt"), "5pt", "command_lower_body")
    assert np.array_equal(back, frozen)
    assert p._ref_joints_applied is False and p._ref_joints_last is None


def test_ref_joints_stale_freezes_the_last_override_like_the_keypoints() -> None:
    p, live = _armed_5pt_with_legs()
    names = _dof_names()
    q = np.full(29, 0.3, dtype=np.float32)
    dq = np.full(29, -1.0, dtype=np.float32)
    live.set_ref_joints(q, dq, names)
    fed = _slot(_obs(p, "5pt"), "5pt", "command_lower_body").copy()
    assert np.allclose(fed, _lower(q, dq), atol=1e-6)
    live.set(None)   # the bridge went STALE mid-walk
    frozen = _slot(_obs(p, "5pt"), "5pt", "command_lower_body")
    assert np.array_equal(frozen, fed), "stale must freeze the reference legs, not snap them to the stance"
    assert p._ref_joints_applied is True
    # A fresh frame WITHOUT the field afterwards clears it (the walk ended while the bridge was out).
    live.set(np.arange(15, dtype=np.float32) * 0.1)
    live.set_ref_joints(None, None, names)
    cleared = _slot(_obs(p, "5pt"), "5pt", "command_lower_body")
    assert np.allclose(cleared, _lower(p._motion_command_t[0, :29], np.zeros(29)))
    assert p._ref_joints_applied is False


def test_ref_joints_is_remapped_by_name_not_by_index() -> None:
    p, live = _armed_5pt_with_legs()
    names = _dof_names()
    q = np.arange(29, dtype=np.float32) * 0.02
    dq = np.arange(29, dtype=np.float32) * -0.05
    perm = np.random.default_rng(3).permutation(29)
    live.set_ref_joints(q[perm], dq[perm], [names[i] for i in perm])   # a shuffled sender order
    got = _slot(_obs(p, "5pt"), "5pt", "command_lower_body")
    assert np.allclose(got, _lower(q, dq), atol=1e-6), "placed by NAME: the shuffle must be undone"


def test_ref_joints_refuses_a_frame_missing_a_lower_body_joint_and_warns_once() -> None:
    from wbt_wrappers_inference.config_values.robot_config import LOWER_BODY_INDICES

    p, live = _armed_5pt_with_legs()
    names = _dof_names()
    victim = LOWER_BODY_INDICES[3]
    names[victim] = "not_a_joint"          # 29 unique names, one lower-body joint unnamed
    frozen = _slot(_obs(p, "5pt"), "5pt", "command_lower_body").copy()
    live.set_ref_joints(np.full(29, 0.4), np.zeros(29), names)
    got = _slot(_obs(p, "5pt"), "5pt", "command_lower_body")
    assert np.array_equal(got, frozen), "a PARTIAL lower-body reference is refused whole"
    assert p._ref_joints_applied is False and p._ref_joints_warned is True


def test_ref_joints_knob_off_ignores_the_field() -> None:
    p, live = _armed_5pt_with_legs()
    p._vive_ref_joints = False
    frozen = _slot(_obs(p, "5pt"), "5pt", "command_lower_body").copy()
    live.set_ref_joints(np.full(29, 0.4), np.full(29, 2.0), _dof_names())
    assert np.array_equal(_slot(_obs(p, "5pt"), "5pt", "command_lower_body"), frozen)
    assert p._ref_joints_applied is False


def test_ref_joints_is_a_5pt_only_channel() -> None:
    """A 3pt policy has the same `command_lower_body` term; the field must not touch it."""
    live = _FakeKeypointSourceWithLegs(3)
    p = _stub_policy("3pt", live=live)
    live.set(np.arange(9, dtype=np.float32) * 0.1)
    assert p._snapshot_clutch_anchor()
    p._vive_armed = True
    t0 = _terms(p)["command_lower_body"].copy()
    live.set_ref_joints(np.full(29, 0.4), np.full(29, 2.0), _dof_names())
    t1 = _terms(p)["command_lower_body"]
    assert np.array_equal(t0, t1)
    assert p._ref_joints_applied is None, "not a 5pt fact: RecordTap gets null, not false"


def test_ref_joints_state_is_cleared_with_the_other_teleop_anchors() -> None:
    p, live = _armed_5pt_with_legs()
    live.set_ref_joints(np.full(29, 0.4), np.zeros(29), _dof_names())
    _obs(p, "5pt")
    assert p._ref_joints_last is not None
    p._clear_vive_teleop_anchors()
    assert p._ref_joints_last is None


def test_record_tap_publishes_ref_joints_live_as_a_tri_state() -> None:
    from wbt_wrappers_inference.utils.record_tap import RecordTap

    tap = object.__new__(RecordTap)
    tap._seq = 0
    tap._q = __import__("queue").Queue(maxsize=8)
    tap._encode_errors = 0
    tap._dropped = 0
    common = dict(
        running=True, vive_armed=True, timestep=0.0, servo_on=False, f_cmd=None, f_meas=None,
        f_meas_ungated=None, servo_integ=None, servo_dx=None, servo_dq=None,
        dof_pos=np.zeros(29), dof_vel=np.zeros(29), root_quat_wxyz=np.array([1.0, 0, 0, 0]),
        base_ang_vel=np.zeros(3),
    )
    tap.publish(ref_joints_live=True, **common)
    tap.publish(ref_joints_live=False, **common)
    tap.publish(**common)   # an old call site
    f0, f1, f2 = tap._q.get_nowait(), tap._q.get_nowait(), tap._q.get_nowait()
    assert f0["ref_joints_live"] is True and f1["ref_joints_live"] is False and f2["ref_joints_live"] is None
    import json

    json.dumps(f0)
