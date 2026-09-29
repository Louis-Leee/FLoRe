"""Live-VIVE teleop + live-F/T sim2real on the LOCAL C1 UNIFIED single-policy
presets (``exp:g1-wbt-3pt-hybrid-unified`` and its ``-anneal`` twin).

The unified actor obs is 170D/frame × 4 history = 680D: the pelvis-local 168D
3pt-hybrid layout with a 2D soft ``contact_phase`` label inserted right after
``force_meas_local``. Two deploy behaviours are genuinely new versus the
servo-ON 3pt-hybrid deploy, and both are exercised here against the REAL
``_compute_obs_terms`` / ``_update_force_meas_and_servo`` with a real Pinocchio
yam model and fake VIVE + F/T sources:

  1. **contact_phase is computed LIVE from the Bota reading** — hysteresis on
     ‖F_meas‖ (2.0 N on / 1.5 N off, HOLDing in the dead band) → 50 ms LPF, per
     hand, with a post-START warmup. Training derives it from the *simulated* F/T
     magnitude precisely so the real sensor reproduces it.
  2. **The servo is OBSERVE-ONLY** — the contact-gated integrator still winds, so
     ``integrator_state`` (I) and ``x_tgt_shift`` (Δx) are live actor inputs, but
     ``Δq_arm`` is NEVER injected into the PD path (the policy is the sole
     executor, exactly as in training).

Plus the invariants inherited from the 3pt VIVE port (zero-jump clutch arming,
tracker motion confined to the vr_3point_pos slot, head DOF) re-pinned at 170D.

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

_UNIFIED_PRESET = "g1-29dof-holosoma-3pt-hybrid-unified-wbt"

# Per-frame (170D) slot offsets — see observation.holosoma_3pt_hybrid_unified_wbt.
# Locked here so a silent term-order/width change trips the test, not the robot.
# Delta vs 3pt-hybrid (168D): contact_phase(2) after force_meas_local; everything
# after shifts by +2.
_SLOTS = {
    "command_lower_body": (0, 24),
    "vr_3point_pos": (24, 33),
    "vr_3point_orn": (33, 45),
    "force_cmd_local": (45, 51),
    "force_meas_local": (51, 57),
    "contact_phase": (57, 59),
    "integrator_state": (59, 65),
    "x_tgt_shift": (65, 71),
    "motion_ref_ori_b": (71, 77),
    "projected_gravity": (77, 80),
    "base_ang_vel": (80, 83),
    "joint_pos": (83, 112),
    "joint_vel": (112, 141),
    "actions": (141, 170),
}
_PER_FRAME = 170


def _identity_orn12_quat():
    """Three identity xyzw quaternions [0,0,0,1] — the 3pt vr_3point_orn rep."""
    return np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), 3)


class _FakeViveSource:
    """Stand-in for ViveLiveSource: returns a fixed (pos[9], orn[12]) 3pt command."""

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


class _FakeForceSource:
    """Stand-in for ForceLiveSource: per-hand SENSOR-frame Newtons (3-vec each)."""

    def __init__(self, left=(0.0, 0.0, 0.0), right=(0.0, 0.0, 0.0)):
        self.set(left, right)

    def set(self, left, right):
        self._l = np.asarray(left, dtype=np.float64)
        self._r = np.asarray(right, dtype=np.float64)

    def latest(self):
        return self._l.copy(), self._r.copy()


def _stub_unified_policy(*, vive=False, force_source=None):
    """Minimal HolosomaWBTPolicy wired exactly as ``_compute_obs_terms`` +
    ``_update_force_meas_and_servo`` read it for the UNIFIED preset: the REAL 170D
    obs config, a real Pinocchio yam model, and the observe-only servo state that
    ``_resolve_unified_mode`` would have installed. No heavy __init__."""
    from loguru import logger as _loguru

    from holosoma_inference.policies.wbt_utils import PinocchioRobot

    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.config_values.robot_config import VR_3POINT_BODIES, g1_29dof_wbt
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy
    from wbt_wrappers_inference.utils.force_deploy_math import ServoConfig

    if not _YAM_URDF.is_file():
        pytest.skip(f"yam URDF not present: {_YAM_URDF}")

    obs_cfg = DEFAULTS[_UNIFIED_PRESET].observation
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

    # obs assembly state (plain "3pt": quaternion orn, no root trio).
    p._obs_terms = terms
    p._obs_mode = "3pt"
    p._is_2pt = False
    p._is_3pt = True
    p._vr_orn_6d = False
    # TCP-unified hand geometry, derived from the preset's own terms exactly as
    # setup_policy does — so this stub tracks the 0.242 m TCP iff the real preset does,
    # and cannot silently drift to the 0.18 m lever if the preset changes.
    p._vr_pos_tcp = any("vr_3point_pos_tcp" in t for t in terms)
    p._history_length = hist
    p._obs_history = {t: deque(maxlen=hist) for t in terms}
    p._last_action = np.zeros(29, dtype=np.float32)
    p._ref_body_is_pelvis = True
    p._heading_initialized = True
    p.robot_yaw_offset = 0.0
    p.motion_yaw_offset = 0.0
    p._align_rot_mat = np.eye(3)

    model = p.pinocchio_robot.robot_model
    p._vr_frame_ids = [model.getFrameId(b) for b in VR_3POINT_BODIES]
    p._force_wrist_frame_ids = None

    ref_dof = p.default_dof_angles.astype(np.float32)
    p._motion_command_t = np.concatenate([ref_dof.reshape(1, -1), np.zeros((1, 29), dtype=np.float32)], axis=1)
    p._ref_pos_xyz_t = np.array([[0.0, 0.0, 0.78]], dtype=np.float32)
    p._ref_quat_xyzw_t = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    p._hold_dof_override = None

    # Force / servo state — OBSERVE-ONLY, exactly what _resolve_unified_mode leaves.
    p._force_live = force_source
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
    p._hybrid_servo = False  # forced OFF for unified — Δq is NEVER injected
    p._servo_observe_only = True
    p._unified_integ_cmd_gate = True  # the default; _resolve_unified_mode-independent
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

    # contact_phase state (mirrors __init__ / training HybridContactConfig).
    p._contact_phase = np.zeros(2, dtype=np.float64)
    p._contact_hyst = np.zeros(2, dtype=np.float64)
    p._contact_phase_ticks = 0
    p._contact_phase_theta_on = 2.0
    p._contact_phase_theta_off = 1.5
    p._contact_phase_tau_s = 0.05
    p._contact_phase_warmup_steps = 5

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


def _per_frame(obs, hist):
    """Extract the LATEST per-frame 170D vector from the assembled (1, 680) obs."""
    obs = obs.reshape(-1)
    out = np.zeros(_PER_FRAME, dtype=np.float32)
    cursor = 0
    for _name, (lo, hi) in _SLOTS.items():
        w = hi - lo
        block = obs[cursor : cursor + w * hist]
        out[lo:hi] = block[(hist - 1) * w : hist * w]  # newest frame
        cursor += w * hist
    return out


def _slot(vec, name):
    lo, hi = _SLOTS[name]
    return vec[lo:hi]


def _code_of(fn) -> str:
    """Source of `fn` with comments and string literals stripped, so a source-level pin
    cannot be satisfied by a comment or docstring that merely mentions the symbol."""
    import inspect
    import io
    import textwrap
    import tokenize

    text = textwrap.dedent(inspect.getsource(fn))
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        return "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())
    return " ".join(out)


def _tick(p, rs, n=1):
    frame = None
    for _ in range(n):
        frame = _per_frame(p._build_observation(rs), p._history_length)
    return frame


def _set_force_cmd(p, left, right):
    """Set the applied (slewed) F^cmd exactly as _poll_force_cmd_keys would: the raw
    (2,3) pelvis-frame Newtons the gate/servo read AND the pre-scaled 6D obs vector
    (×0.04 = 1/25 N) the force_cmd_local term returns."""
    raw = np.array([left, right], dtype=np.float64)
    p._force_cmd_pelvis_raw = raw
    p._force_cmd_local_scaled = (raw.reshape(-1) * 0.04).astype(np.float32)
    return raw


# --- obs geometry --------------------------------------------------------------
def test_unified_obs_is_680d_and_3pt():
    p = _stub_unified_policy()
    assert p._is_3pt, "unified-local must detect as 3pt so the VIVE override fires"
    obs = p._build_observation(_robot_state(p))
    assert obs.shape == (1, _PER_FRAME * p._history_length), obs.shape


def test_unified_orn_is_unit_quaternions():
    """vr_3point_orn stays the 3×xyzw QUATERNION rep (the local family) — 3 unit
    4-vectors, not rot6d."""
    p = _stub_unified_policy()
    frame = _tick(p, _robot_state(p))
    norms = np.linalg.norm(_slot(frame, "vr_3point_orn").reshape(3, 4), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), norms


def test_all_force_slots_zero_without_live_source():
    """No live F/T ⇒ force_cmd / force_meas / contact_phase / I / Δx all read exactly
    zero, so a unified checkpoint runs pure 3pt position tracking headless."""
    p = _stub_unified_policy(force_source=None)
    frame = _tick(p, _robot_state(p))
    for name in ("force_cmd_local", "force_meas_local", "contact_phase", "integrator_state", "x_tgt_shift"):
        assert np.allclose(_slot(frame, name), 0.0), name


# --- contact_phase (the new deployable label) ----------------------------------
def test_contact_phase_rises_above_theta_on_after_warmup():
    """A sustained 6 N contact drives φ→1 on that hand only, and only AFTER the
    post-START warmup has elapsed."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 0.0])
    p = _stub_unified_policy(force_source=src)
    rs = _robot_state(p)

    # Warmup: the first `warmup_steps` ticks force the hysteresis OFF even at 6 N.
    for _ in range(p._contact_phase_warmup_steps):
        frame = _tick(p, rs)
        assert np.allclose(_slot(frame, "contact_phase"), 0.0), "warmup must suppress the label"

    frame = _tick(p, rs, n=60)  # >> tau/dt ⇒ converged
    phi = _slot(frame, "contact_phase")
    assert phi[0] > 0.95, phi
    assert phi[1] < 1e-6, "the free hand must stay at 0"
    assert 0.0 <= phi.min() and phi.max() <= 1.0


def test_contact_phase_lpf_matches_analytic_step_response():
    """φ must follow the exact first-order step response the policy trained on:
    φ_n = 1 − (1 − β)^n with β = 1 − exp(−dt/τ), dt = one control tick."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 0.0])
    p = _stub_unified_policy(force_source=src)
    rs = _robot_state(p)
    _tick(p, rs, n=p._contact_phase_warmup_steps)  # clear the warmup (phi still 0)
    assert np.allclose(p._contact_phase, 0.0)

    beta = 1.0 - np.exp(-(1.0 / 50.0) / 0.05)
    for n in range(1, 9):
        frame = _tick(p, rs)
        expected = 1.0 - (1.0 - beta) ** n
        # The internal state is float64; the obs slot is cast to float32 (as the
        # ONNX consumes it), so compare at float32 resolution.
        assert abs(float(_slot(frame, "contact_phase")[0]) - expected) < 1e-6, n
        assert abs(float(p._contact_phase[0]) - expected) < 1e-12, n


def test_contact_phase_hysteresis_holds_in_dead_band():
    """Once latched ON at >2.0 N, a drop into the 1.5-2.0 N dead band must HOLD the
    latch (φ stays ≈1); only dropping BELOW 1.5 N releases it."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 0.0])
    p = _stub_unified_policy(force_source=src)
    rs = _robot_state(p)
    _tick(p, rs, n=80)
    assert _slot(_tick(p, rs), "contact_phase")[0] > 0.95

    src.set([0.0, 0.0, 1.7], [0.0, 0.0, 0.0])  # inside the dead band
    frame = _tick(p, rs, n=40)
    assert _slot(frame, "contact_phase")[0] > 0.95, "dead band must HOLD the latch"

    src.set([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])  # below theta_off → release
    frame = _tick(p, rs, n=80)
    assert _slot(frame, "contact_phase")[0] < 0.05, "below theta_off must release"


def test_contact_phase_dead_band_does_not_latch_from_free_space():
    """Symmetrically: a reading that only ever sits in the dead band must NEVER
    latch ON (it starts released), so sensor noise around 1.6 N cannot fake contact."""
    src = _FakeForceSource(left=[0.0, 0.0, 1.7], right=[0.0, 0.0, 1.9])
    p = _stub_unified_policy(force_source=src)
    frame = _tick(p, _robot_state(p), n=100)
    assert np.allclose(_slot(frame, "contact_phase"), 0.0), "dead band must not latch from 0"


def test_contact_phase_zeroed_when_not_running():
    """Not driving motors (pre-START / stopped) ⇒ φ is zero and the warmup re-arms,
    mirroring training's zeroed reset buffer."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 6.0])
    p = _stub_unified_policy(force_source=src)
    rs = _robot_state(p)
    _tick(p, rs, n=80)
    assert p._contact_phase.max() > 0.9

    p.use_policy_action = False
    frame = _tick(p, rs)
    assert np.allclose(_slot(frame, "contact_phase"), 0.0)
    assert p._contact_phase_ticks == 0, "warmup must re-arm for the next START"


def test_contact_phase_reset_on_restart():
    """_reset_motion_state (STOP → re-START) must clear a latched φ so the new arm
    starts in free space with the warmup re-armed."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 6.0])
    p = _stub_unified_policy(force_source=src)
    _tick(p, _robot_state(p), n=80)
    assert p._contact_phase.max() > 0.9
    p._reset_contact_phase_state()
    assert np.allclose(p._contact_phase, 0.0)
    assert np.allclose(p._contact_hyst, 0.0)
    assert p._contact_phase_ticks == 0


def test_contact_phase_consistent_with_gate_on_cmd():
    """With GATE_CMD on and no F^cmd, the hand's F_meas obs is zeroed — φ MUST be
    zeroed too, or the actor would see the never-trained pair (φ=1, F_meas=0)."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 0.0])
    p = _stub_unified_policy(force_source=src)
    p._force_meas_gate_on_cmd = True
    p._force_meas_gate_cmd_n = 1e-6
    _set_force_cmd(p, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])  # no command on either hand
    frame = _tick(p, _robot_state(p), n=100)
    assert np.allclose(_slot(frame, "force_meas_local"), 0.0), "gate must zero F_meas"
    assert np.allclose(_slot(frame, "contact_phase"), 0.0), "phi must follow the gated F_meas"


def test_contact_phase_input_magnitude_is_frame_invariant():
    """φ keys off ‖F_meas‖, and sensor→pelvis is a pure ROTATION, so the same sensor
    magnitude on any axis yields the same decision variable. Asserting on the
    saturated φ would be vacuous (any force in (θ_on, ∞) converges to the same
    trajectory), so assert on ‖_force_meas_pelvis‖ — the quantity actually claimed
    invariant — and only then that φ latched."""
    mags, phis = [], []
    for axis in range(3):
        f = [0.0, 0.0, 0.0]
        f[axis] = 6.0
        p = _stub_unified_policy(force_source=_FakeForceSource(left=f, right=[0.0, 0.0, 0.0]))
        frame = _tick(p, _robot_state(p), n=100)
        mags.append(float(np.linalg.norm(p._force_meas_pelvis[0])))
        phis.append(float(_slot(frame, "contact_phase")[0]))
    assert max(mags) - min(mags) < 1e-9, mags
    assert all(abs(m - 6.0) < 1e-9 for m in mags), mags  # rotation preserves magnitude
    assert all(v > 0.95 for v in phis), phis


# --- observe-only servo --------------------------------------------------------
def test_observe_only_publishes_wound_integrator_but_no_dq():
    """THE unified invariant: with a real F^cmd and a real contact the integrator
    winds (I and Δx become nonzero in the obs) yet Δq stays EXACTLY zero."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 6.0])
    p = _stub_unified_policy(force_source=src)
    _set_force_cmd(p, [5.0, 0.0, 0.0], [5.0, 0.0, 0.0])
    rs = _robot_state(p)
    frame = _tick(p, rs, n=20)

    assert np.linalg.norm(_slot(frame, "integrator_state")) > 1e-3, "I must wind (observe-only ≠ servo-off)"
    assert np.linalg.norm(_slot(frame, "x_tgt_shift")) > 1e-6, "Δx must be published"
    assert np.allclose(p._servo_dq, 0.0), "observe-only must NEVER produce a Δq"


def test_observe_only_delta_x_equals_task_wrench_over_k():
    """Δx = (F^cmd + I)/K — the same reference shift training fed the Eq.2 target."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 0.0])
    p = _stub_unified_policy(force_source=src)
    _set_force_cmd(p, [4.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    _tick(p, _robot_state(p), n=10)
    expected = (p._force_cmd_pelvis_raw + p._servo_integ) / p._servo_cfg.stiffness_K
    assert np.allclose(p._servo_delta_x, expected, atol=1e-12)


def test_observe_only_integrator_frozen_in_free_space():
    """The contact gate still applies: below HOLOSOMA_FORCE_CONTACT_N the integrator
    is frozen at zero even with a nonzero F^cmd (no free-space windup).

    ALSO pins the README's central safety number: with I frozen at 0, the published
    setpoint shift is Δx = F^cmd/K, i.e. 5 N / 200 N·m⁻¹ = 2.5 cm — nonzero IN FREE
    SPACE. That is why observe-only does not make the run safe: the policy is trained
    to chase this shift, so the F^cmd keys still move the arms with no contact at all.
    """
    src = _FakeForceSource(left=[0.0, 0.0, 0.5], right=[0.0, 0.0, 0.0])  # < 2 N gate
    p = _stub_unified_policy(force_source=src)
    _set_force_cmd(p, [5.0, 0.0, 0.0], [5.0, 0.0, 0.0])
    frame = _tick(p, _robot_state(p), n=30)
    assert np.allclose(p._servo_integ, 0.0), "contact gate must freeze I off-contact"
    assert np.allclose(p._servo_dq, 0.0)
    # Δx = (F^cmd + 0)/K on both hands, published to the actor's x_tgt_shift slot.
    expected = p._force_cmd_pelvis_raw / p._servo_cfg.stiffness_K
    assert np.allclose(p._servo_delta_x, expected, atol=1e-12)
    assert abs(float(p._servo_delta_x[0][0]) - 0.025) < 1e-12, "5 N should shift the setpoint 2.5 cm"
    assert np.allclose(_slot(frame, "x_tgt_shift"), expected.reshape(-1), atol=1e-7)


def test_observe_only_zeroes_integrator_when_not_running():
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 6.0])
    p = _stub_unified_policy(force_source=src)
    _set_force_cmd(p, [5.0, 0.0, 0.0], [5.0, 0.0, 0.0])
    _tick(p, _robot_state(p), n=20)
    assert np.linalg.norm(p._servo_integ) > 1e-3
    p.use_policy_action = False
    _tick(p, _robot_state(p))
    assert np.allclose(p._servo_integ, 0.0)
    assert np.allclose(p._servo_delta_x, 0.0)


def test_runaway_valve_holds_the_integrator_in_observe_only():
    """In observe-only there is no Δq to cut, so a runaway must HOLD the last good
    I/Δx rather than zero them: zeroing would hand the SOLE controller
    {F_meas huge, φ=1, I=0, Δx=0}, a pairing training never produces (a large
    F_meas always comes with a wound integrator)."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 0.0])
    p = _stub_unified_policy(force_source=src)
    _set_force_cmd(p, [5.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    _tick(p, _robot_state(p), n=20)
    integ_before = p._servo_integ.copy()
    dx_before = p._servo_delta_x.copy()
    assert np.linalg.norm(integ_before) > 1e-3

    p._force_runaway_n = 5.0  # the 6 N reading now trips it
    frame = _tick(p, _robot_state(p))
    assert np.allclose(p._servo_integ, integ_before), "observe-only runaway must HOLD I"
    assert np.allclose(p._servo_delta_x, dx_before), "observe-only runaway must HOLD Δx"
    assert np.allclose(p._servo_dq, 0.0), "still no Δq"
    assert np.linalg.norm(_slot(frame, "force_meas_local")) > 0.0, "F_meas obs still reports"


def test_runaway_valve_still_zeroes_the_integrator_when_servo_would_inject():
    """REGRESSION: for a real servo-ON (non-unified) deploy the valve keeps its
    original meaning — freeze the servo by zeroing I/Δx/Δq."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 0.0])
    p = _stub_unified_policy(force_source=src)
    p._servo_observe_only = False
    p._hybrid_servo = True  # pretend a servo-ON 3pt-hybrid deploy
    _set_force_cmd(p, [5.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    for _ in range(20):
        p._update_force_meas_and_servo(p.default_dof_angles.copy())
    assert np.linalg.norm(p._servo_integ) > 1e-3
    p._force_runaway_n = 5.0
    p._update_force_meas_and_servo(p.default_dof_angles.copy())
    assert np.allclose(p._servo_integ, 0.0), "servo-ON runaway must FREEZE (zero) the servo"
    assert np.allclose(p._servo_delta_x, 0.0)
    assert np.allclose(p._servo_dq, 0.0)


def test_contact_phase_suppressed_for_the_whole_tare_window():
    """The "reading not yet trusted" window (burst + settle, default 2.0+1.5 s ≫ the
    5-tick parity warmup) is the load-bearing suppression: the pre-tare reading is
    untrusted and the hysteresis LATCHES, so a >θ_on residual bias must not stick φ at
    ≈1 in free space for the whole window on every take."""
    import time as _time

    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 6.0])  # untared bias
    p = _stub_unified_policy(force_source=src)
    p._arm_force_trust_window(0.5)
    rs = _robot_state(p)
    frame = _tick(p, rs, n=120)  # >> the 5-tick warmup, still inside the window
    assert np.allclose(_slot(frame, "contact_phase"), 0.0), "the tare window must suppress φ"
    # Window expires → φ is free to latch on a genuine contact.
    while _time.monotonic() < p._force_trust_deadline:
        _time.sleep(0.02)
    frame = _tick(p, rs, n=80)
    assert _slot(frame, "contact_phase")[0] > 0.95, "φ must go live once the tare has landed"


def test_contact_phase_warmup_helper_is_stub_safe():
    """_contact_phase_warmup_active must not raise on a bare object.__new__ stub
    (no counters, no tare-burst attribute) — the force-deploy test idiom."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    bare = object.__new__(P)
    assert P._contact_phase_warmup_active(bare) is True  # ticks 0 <= warmup 5


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_contact_phase_nonfinite_is_scrubbed_before_the_obs():
    """φ is fed STRAIGHT to the ONNX with no NaN guard downstream (np.clip passes
    NaN, and BasePolicy adds the scaled action to the PD target unchecked), so a
    corrupted state must be scrubbed to 0 = free space."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 0.0])
    p = _stub_unified_policy(force_source=src)
    rs = _robot_state(p)
    _tick(p, rs, n=20)
    p._contact_phase[:] = [np.nan, np.inf]
    frame = _tick(p, rs)
    phi = _slot(frame, "contact_phase")
    assert np.all(np.isfinite(phi)), phi
    assert 0.0 <= phi.min() and phi.max() <= 1.0, phi


def test_record_tap_field_is_gated_on_the_obs_term_not_observe_only():
    """The RECORD φ field keys off the obs term actually being PRESENT (so a future
    non-observe-only contact_phase preset still records it), and `_as_list(None)`
    keeps the field null for every pre-existing preset."""
    from wbt_wrappers_inference.utils.record_tap import _as_list

    assert _as_list(None) is None
    p = _stub_unified_policy()
    p._obs_terms = list(p.config.observation.obs_dict["actor_obs"])
    p._hybrid_servo = False
    p._servo_observe_only = False
    p._has_contact_phase_term = False
    assert p._resolve_unified_mode() is True
    assert p._has_contact_phase_term is True


def test_servo_off_non_unified_still_zeroes_integrator():
    """REGRESSION: the pre-existing Layer-A path is untouched — a NON-unified stub
    with servo off and observe-only off keeps I/Δx/Δq at zero."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 6.0])
    p = _stub_unified_policy(force_source=src)
    p._servo_observe_only = False  # pretend a plain 3pt-hybrid Layer-A deploy
    _set_force_cmd(p, [5.0, 0.0, 0.0], [5.0, 0.0, 0.0])
    p._update_force_meas_and_servo(p.default_dof_angles.copy())
    for _ in range(20):
        p._update_force_meas_and_servo(p.default_dof_angles.copy())
    assert np.allclose(p._servo_integ, 0.0)
    assert np.allclose(p._servo_delta_x, 0.0)
    assert np.allclose(p._servo_dq, 0.0)
    assert np.linalg.norm(p._force_meas_pelvis) > 0.0, "F_meas obs is still live (Layer A)"


# --- VIVE teleop at 170D -------------------------------------------------------
def test_vive_moves_only_vr_slots_not_force_slots():
    """Arming VIVE + moving all three trackers changes ONLY vr_3point_pos — never
    the force / phase / servo slots."""
    p = _stub_unified_policy(vive=True)
    rs = _robot_state(p)
    base = _tick(p, rs)
    ref_vr_pos = _slot(base, "vr_3point_pos").copy()

    live0 = ref_vr_pos.copy()
    p._vive_live.set(live0, None)
    p._vr_clutch_live0 = live0.astype(np.float32).copy()
    p._vive_armed = True
    moved = live0.copy()
    moved[0] += 0.30  # left hand +x
    moved[3] += 0.30  # right hand +x
    moved[6] += 0.20  # head +x
    p._vive_live.set(moved, None)

    frame = _tick(p, rs, n=80)
    vr_pos = _slot(frame, "vr_3point_pos")
    assert vr_pos[0] > ref_vr_pos[0] + 0.05
    assert vr_pos[3] > ref_vr_pos[3] + 0.05
    assert vr_pos[6] > ref_vr_pos[6] + 0.05, "HEAD x should track (the 3pt DOF)"
    for name in ("force_cmd_local", "force_meas_local", "contact_phase", "integrator_state", "x_tgt_shift"):
        assert np.allclose(_slot(frame, name), 0.0), f"VIVE must not leak into {name}"


def test_vive_arm_is_zero_jump():
    p = _stub_unified_policy(vive=True)
    rs = _robot_state(p)
    ref_vr_pos = _slot(_tick(p, rs), "vr_3point_pos").copy()
    live0 = ref_vr_pos.copy()
    p._vive_live.set(live0, None)
    p._vr_clutch_live0 = live0.astype(np.float32).copy()
    p._vive_armed = True
    armed = _slot(_tick(p, rs), "vr_3point_pos")
    assert np.allclose(armed, ref_vr_pos, atol=1e-5), "zero-delta arm must be zero-jump"


def test_vive_teleop_and_live_force_coexist():
    """The combined deploy mode: tracker-driven hands AND a live contact. The VR slot
    tracks the tracker while contact_phase / F_meas track the sensor, independently."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 0.0])
    p = _stub_unified_policy(vive=True, force_source=src)
    rs = _robot_state(p)
    base = _tick(p, rs)
    ref_vr_pos = _slot(base, "vr_3point_pos").copy()

    live0 = ref_vr_pos.copy()
    p._vive_live.set(live0, None)
    p._vr_clutch_live0 = live0.astype(np.float32).copy()
    p._vive_armed = True
    moved = live0.copy()
    moved[0] += 0.20
    p._vive_live.set(moved, None)

    frame = _tick(p, rs, n=100)
    assert _slot(frame, "vr_3point_pos")[0] > ref_vr_pos[0] + 0.05, "tracker still drives the hand"
    phi = _slot(frame, "contact_phase")
    assert phi[0] > 0.95 and phi[1] < 1e-6, phi
    assert np.linalg.norm(_slot(frame, "force_meas_local")) > 0.0
    assert np.allclose(p._servo_dq, 0.0), "still no Δq — observe-only"


def test_unarmed_vive_is_byte_identical_vr_slot():
    p_off = _stub_unified_policy(vive=False)
    p_on = _stub_unified_policy(vive=True)  # source present but NOT armed
    rs = _robot_state(p_off)
    off = _slot(_tick(p_off, rs), "vr_3point_pos")
    on_unarmed = _slot(_tick(p_on, rs), "vr_3point_pos")
    assert np.allclose(off, on_unarmed)


# --- offline recorder plumbing for the new label ---------------------------------
def _record_teleop_mod():
    """Import the out-of-process recorder (it lives in sandbox/, outside the package).

    The README promises the φ signal is verifiable offline, so the recorder's column
    layout is part of this port's contract, not an unrelated script.
    """
    import importlib.util
    import sys

    path = _REPO / "force_deploy/record_teleop.py"
    if not path.is_file():
        pytest.skip(f"recorder not present: {path}")
    spec = importlib.util.spec_from_file_location("_record_teleop_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses needs the module registered
    spec.loader.exec_module(mod)
    return mod


def _blank_policy_frame():
    return dict(
        f_cmd=[[0, 0, 0], [0, 0, 0]],
        f_meas=[[0, 0, 0], [0, 0, 0]],
        f_meas_ungated=[[0, 0, 0], [0, 0, 0]],
        servo_integ=[[0, 0, 0], [0, 0, 0]],
        servo_dx=[[0, 0, 0], [0, 0, 0]],
        servo_dq=[0.0] * 29,
        dof_pos=[0.0] * 29,
        dof_vel=[0.0] * 29,
        root_quat_wxyz=[1, 0, 0, 0],
        base_ang_vel=[0, 0, 0],
    )


def test_recorder_stores_contact_phase_columns():
    """The recorder must persist φ as `contact_phase_{L,R}` — that is the offline
    check ("did φ only rise while I was actually touching something?") the README
    promises. Row width must stay equal to POLICY_COLS."""
    m = _record_teleop_mod()
    cols = list(m.POLICY_COLS)
    assert cols[-2:] == ["contact_phase_L", "contact_phase_R"], cols[-4:]
    row = m._flat_policy_row({**_blank_policy_frame(), "contact_phase": [0.4, 0.9]})
    assert row is not None and len(row) == len(cols)
    i = cols.index("contact_phase_L")
    assert row[i] == 0.4 and row[i + 1] == 0.9


def test_recorder_contact_phase_is_nan_for_non_unified_takes():
    """A frame with no φ (every pre-existing preset, and pre-φ takes) must still
    produce a full-width row with NaN in the two new slots — never a dropped or
    ragged row, since the DOF-width gate is the only thing allowed to drop a row."""
    m = _record_teleop_mod()
    cols = list(m.POLICY_COLS)
    row = m._flat_policy_row(_blank_policy_frame())
    assert row is not None and len(row) == len(cols)
    i = cols.index("contact_phase_L")
    assert np.isnan(row[i]) and np.isnan(row[i + 1])
    # A null on the wire (the tap's default) behaves the same as an absent key.
    row_null = m._flat_policy_row({**_blank_policy_frame(), "contact_phase": None})
    assert np.isnan(row_null[i]) and np.isnan(row_null[i + 1])


# --- integrator windup: the phantom-bias hazard the cmd gate closes ---------------
def test_free_space_bias_does_not_wind_the_integrator_by_default():
    """THE hazard: the deploy integrator gate keys on ‖F_meas‖ (training uses the
    privileged depth>0), so a sustained free-space offset above the 2 N contact gate —
    a drifted tare, RNEA gravity-model error, a grasped object — looks like contact.
    With F^cmd=0 the error is e=F_meas, so I would ramp to ±I_max (40 N) and Δx to the
    0.2 m clip: a 20 cm commanded setpoint shift that the unified policy CHASES.

    The default command gate must prevent this entirely, so the documented bring-up run
    (every F^cmd at 0) is genuinely equivalent to Layer A."""
    src = _FakeForceSource(left=[0.0, 0.0, 3.0], right=[0.0, 0.0, 0.0])  # 3 N phantom
    p = _stub_unified_policy(force_source=src)
    _set_force_cmd(p, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])  # nothing commanded
    frame = _tick(p, _robot_state(p), n=1500)  # 30 s at 50 Hz

    assert np.allclose(p._servo_integ, 0.0), f"I wound in free space: {p._servo_integ}"
    assert np.allclose(p._servo_delta_x, 0.0), f"Δx nonzero with no F^cmd: {p._servo_delta_x}"
    assert np.allclose(_slot(frame, "integrator_state"), 0.0)
    assert np.allclose(_slot(frame, "x_tgt_shift"), 0.0)
    # The policy must still SEE the force and the contact label (that is the whole
    # difference from GATE_CMD, which blinds it).
    assert np.linalg.norm(_slot(frame, "force_meas_local")) > 0.0
    assert _slot(frame, "contact_phase")[0] > 0.95


def test_cmd_gate_off_reproduces_the_unbounded_windup():
    """Sanity-check that the guard above is load-bearing, not vacuous: with the gate
    explicitly disabled (the fidelity A/B setting) the same 3 N phantom DOES ramp I to
    the anti-windup clip and Δx to the 0.2 m clip."""
    src = _FakeForceSource(left=[0.0, 0.0, 3.0], right=[0.0, 0.0, 0.0])
    p = _stub_unified_policy(force_source=src)
    p._unified_integ_cmd_gate = False
    _set_force_cmd(p, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    _tick(p, _robot_state(p), n=1500)
    assert np.linalg.norm(p._servo_integ[0]) > 0.9 * p._servo_cfg.integrator_max
    assert np.linalg.norm(p._servo_delta_x[0]) > 0.15  # → the 0.2 m clip


def test_cmd_gate_lets_a_commanded_hand_wind_normally():
    """The gate must not disable the feature: a hand carrying F^cmd runs the full
    trained loop (I winds, Δx = (F^cmd+I)/K), while the OTHER hand stays gated."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 6.0])
    p = _stub_unified_policy(force_source=src)
    _set_force_cmd(p, [5.0, 0.0, 0.0], [0.0, 0.0, 0.0])  # left commanded only
    _tick(p, _robot_state(p), n=40)
    assert np.linalg.norm(p._servo_integ[0]) > 1e-3, "commanded hand must wind"
    assert np.allclose(p._servo_integ[1], 0.0), "uncommanded hand must stay gated"
    expected = (p._force_cmd_pelvis_raw + p._servo_integ) / p._servo_cfg.stiffness_K
    assert np.allclose(p._servo_delta_x, expected, atol=1e-12)


def test_servo_presets_keep_the_original_contact_gate():
    """REGRESSION: the extra observe-only conditions must NOT touch the servo path. A
    non-unified stub with a 3 N free-space reading and no F^cmd still winds I exactly
    as before (that behaviour is what the servo-ON checkpoints were validated with)."""
    src = _FakeForceSource(left=[0.0, 0.0, 3.0], right=[0.0, 0.0, 0.0])
    p = _stub_unified_policy(force_source=src)
    p._servo_observe_only = False
    p._hybrid_servo = True
    _set_force_cmd(p, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    for _ in range(200):
        p._update_force_meas_and_servo(p.default_dof_angles.copy())
    assert np.linalg.norm(p._servo_integ[0]) > 1e-3, "servo path must be byte-identical"
    assert p._servo_contact_gate(0) == 1.0
    assert p._servo_contact_gate(1) == 0.0  # below the threshold either way


def test_integrator_gate_held_during_the_tare_window():
    """The same "untrusted reading" argument that suppresses φ must also hold the
    integrator: winding I off a pre-tare bias is exactly the phantom-Δx hazard."""
    import time as _time

    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 6.0])
    p = _stub_unified_policy(force_source=src)
    _set_force_cmd(p, [5.0, 0.0, 0.0], [5.0, 0.0, 0.0])  # commanded, so only the window gates
    p._arm_force_trust_window(0.5)
    _tick(p, _robot_state(p), n=120)
    assert np.allclose(p._servo_integ, 0.0), "I must not wind on an un-tared reading"
    assert np.allclose(p._contact_phase, 0.0)
    while _time.monotonic() < p._force_trust_deadline:
        _time.sleep(0.02)
    _tick(p, _robot_state(p), n=40)
    assert np.linalg.norm(p._servo_integ) > 1e-3, "must go live once the tare has landed"


def test_windup_alarm_fires_once_and_rearms():
    """With no soft abort the operator's only defence is seeing it, so ‖I‖ crossing
    half of I_max must warn — once per crossing, not at 50 Hz."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 0.0])
    p = _stub_unified_policy(force_source=src)
    _set_force_cmd(p, [5.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert p._integ_windup_warned is False
    _tick(p, _robot_state(p), n=1500)
    assert np.linalg.norm(p._servo_integ[0]) > 0.5 * p._servo_cfg.integrator_max
    assert p._integ_windup_warned is True, "the alarm latch must have fired"
    # Unwinding re-arms it (so a second episode of windup is reported again).
    p._servo_integ[:] = 0.0
    p._warn_on_integrator_windup()
    assert p._integ_windup_warned is False


def test_manual_tare_clears_the_latched_phase_and_integrator():
    """A re-tare is the documented cure for a latched φ / wound I, so it must clear
    both immediately rather than waiting ~4τ for the LPF."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 0.0])
    p = _stub_unified_policy(force_source=src)
    _set_force_cmd(p, [5.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    _tick(p, _robot_state(p), n=80)
    assert p._contact_phase[0] > 0.9 and np.linalg.norm(p._servo_integ) > 1e-3

    tare_calls = []
    p._force_live = SimpleNamespace(latest=src.latest, request_tare=lambda: tare_calls.append(1))
    p._trigger_force_tare()
    assert tare_calls == [1]
    assert np.allclose(p._contact_phase, 0.0) and np.allclose(p._contact_hyst, 0.0)
    assert np.allclose(p._servo_integ, 0.0) and np.allclose(p._servo_delta_x, 0.0)
    assert p._contact_phase_ticks == 0, "the suppression window must re-arm after a re-tare"


def test_reset_motion_state_zeroes_fcmd_in_observe_only():
    """STOP → re-START must actually stop the pushing. On a unified preset a stale
    F^cmd makes x_tgt_shift = F/K nonzero on the very FIRST tick of the new arm
    (un-slewed) and the policy chases it, so `_reset_motion_state` zeroes the applied
    F^cmd — while the SERVO presets keep the pre-existing carry-over behaviour."""
    for observe_only, expect_zero in ((True, True), (False, False)):
        p = _stub_unified_policy()
        p._servo_observe_only = observe_only
        _set_force_cmd(p, [15.0, 0.0, 0.0], [15.0, 0.0, 0.0])
        # The extra state _reset_motion_state touches (it is the shared reset funnel).
        p._motion_command_0 = p._motion_command_t.copy()
        p._ref_quat_xyzw_0 = p._ref_quat_xyzw_t.copy()
        p._ref_pos_xyz_0 = p._ref_pos_xyz_t.copy()
        p._ref_vel_xyz_t = np.zeros((1, 3), dtype=np.float32)
        p._ref_vel_xyz_0 = p._ref_vel_xyz_t.copy()
        p._ext_frame_idx = 3
        p._action_ema_state = None
        p._force_cmd_target_raw = p._force_cmd_pelvis_raw.copy()
        p.use_policy_action = False  # the STOP / restore path

        p._reset_motion_state()

        if expect_zero:
            assert np.allclose(p._force_cmd_target_raw, 0.0), "the slew TARGET must be zeroed"
            assert np.allclose(p._force_cmd_pelvis_raw, 0.0), "a STOP must hard-zero the applied F^cmd"
            assert np.allclose(p._force_cmd_local_scaled, 0.0), "the pre-scaled obs vector too"
        else:
            assert np.allclose(p._force_cmd_pelvis_raw[0], [15.0, 0.0, 0.0]), "servo path must be unchanged"
            assert np.allclose(p._force_cmd_target_raw[0], [15.0, 0.0, 0.0]), "servo slew target unchanged"
        # Shared resets apply either way.
        assert np.allclose(p._servo_integ, 0.0) and np.allclose(p._contact_phase, 0.0)


def test_start_burst_arms_the_trust_window_including_the_settle():
    """`request_tare()` only sets an Event; the bridge then averages ~1 s while STILL
    republishing the frozen pre-tare value (so it never looks stale). The window must
    therefore cover the burst PLUS a settle interval, and must be armed BEFORE the
    thread starts so the take's very first tick already sees it open."""
    import time as _time

    p = _stub_unified_policy()
    requests = []
    p._force_live = SimpleNamespace(request_tare=lambda: requests.append(_time.monotonic()))
    p._force_tare_on_start_n = 2
    p._force_tare_on_start_s = 0.2
    p._force_tare_settle_s = 0.4
    p._force_tare_burst_thread = None
    p._force_trust_deadline = 0.0

    t0 = _time.monotonic()
    p._start_force_tare_burst()
    # Armed synchronously, covering burst + settle (not just the requests).
    assert p._force_trust_deadline - t0 >= 0.2 + 0.4 - 1e-6
    assert p._contact_phase_warmup_active() is True
    p._contact_phase_ticks = p._contact_phase_warmup_steps + 1  # past the parity window
    assert p._contact_phase_warmup_active() is True, "the tare window must still hold it"
    p._force_tare_burst_thread.join(timeout=3.0)
    assert len(requests) == 2
    while _time.monotonic() < p._force_trust_deadline:
        _time.sleep(0.02)
    assert p._contact_phase_warmup_active() is False


def test_manual_tare_arms_the_trust_window():
    """A manual re-tare (R2 / T) is the documented cure for a latched φ, so it must get
    the SAME settle window — otherwise the latch re-asserts ~120 ms later off the
    still-frozen pre-tare reading and the cure silently does nothing."""
    import time as _time

    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 0.0])
    p = _stub_unified_policy(force_source=src)
    p._force_tare_settle_s = 0.4
    _set_force_cmd(p, [5.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    rs = _robot_state(p)
    _tick(p, rs, n=80)
    assert p._contact_phase[0] > 0.9 and np.linalg.norm(p._servo_integ) > 1e-3

    p._force_live = SimpleNamespace(latest=src.latest, request_tare=lambda: None)
    p._trigger_force_tare()
    assert p._force_trust_deadline > _time.monotonic()
    # The bias is STILL being reported (the bridge has not applied the tare yet), but φ
    # and I must stay down for the whole window — that is the point.
    frame = _tick(p, rs, n=15)
    assert np.allclose(_slot(frame, "contact_phase"), 0.0), "φ must not re-latch inside the window"
    assert np.allclose(p._servo_integ, 0.0), "I must not re-wind inside the window"
    while _time.monotonic() < p._force_trust_deadline:
        _time.sleep(0.02)
    frame = _tick(p, rs, n=80)
    assert _slot(frame, "contact_phase")[0] > 0.95, "and must recover afterwards"


def test_trust_window_never_shortens():
    """`_arm_force_trust_window` must extend, never truncate, an open window (a short
    manual tare during a long START burst must not shrink the burst's protection)."""
    p = _stub_unified_policy()
    p._arm_force_trust_window(5.0)
    long_deadline = p._force_trust_deadline
    p._arm_force_trust_window(0.1)
    assert p._force_trust_deadline == long_deadline
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        p._arm_force_trust_window(bad)
        assert p._force_trust_deadline == long_deadline, bad


def test_settle_default_covers_the_bridge_tare():
    """Pin the default settle interval: the bridge averages ~1 s after the request, so a
    default of 0 (or anything below it) re-opens the un-tared-latch hole this window
    exists to close. FT_ALGO=bota needs ≳3 s — documented, operator-set."""
    import os

    import wbt_wrappers_inference.policies.holosoma_wbt as hw

    assert "HOLOSOMA_FORCE_TARE_SETTLE_SECONDS" not in os.environ
    assert hw._FORCE_TARE_SETTLE_S >= 1.0, (
        "the settle window must cover the bridge's ~1 s RNEA tare average; a smaller "
        "default re-opens the un-tared-latch hole"
    )
    # And __init__ must actually use the constant as its default.
    assert "_FORCE_TARE_SETTLE_S" in _code_of(hw.HolosomaWBTPolicy.__init__)


def test_windup_alarm_threshold_is_half_of_integrator_max():
    """Pin the alarm threshold. It is the ONLY defence for the residual case the command
    gate cannot cover (a phantom offset on a hand that IS commanded), so a silently
    raised threshold would leave that case unreported."""
    import wbt_wrappers_inference.policies.holosoma_wbt as hw

    src = _code_of(hw.HolosomaWBTPolicy._warn_on_integrator_windup)
    assert "0.5" in src and "integrator_max" in src
    # Behaviourally: just over half of I_max must warn, just under must not.
    p = _stub_unified_policy(force_source=_FakeForceSource())
    half = 0.5 * p._servo_cfg.integrator_max
    p._servo_integ[0] = [0.0, 0.0, half * 0.99]
    p._warn_on_integrator_windup()
    assert p._integ_windup_warned is False, "must not fire below half of I_max"
    p._servo_integ[0] = [0.0, 0.0, half * 1.01]
    p._warn_on_integrator_windup()
    assert p._integ_windup_warned is True, "must fire just above half of I_max"


def test_windup_alarm_threshold_override_and_off():
    """``HOLOSOMA_UNIFIED_WINDUP_WARN_N`` (the launcher's ``WINDUP_WARN_N``) replaces the
    half-of-I_max default — read in ``__init__`` through ``parse_unified_windup_warn_n``, so
    the stub sets the attribute the same method reads. 200 N (the launcher default since
    2026-09-20) can never fire under the 40 N/axis clip, 0 is OFF, ``None`` is the reviewed
    default, and a small number fires early. Print-only: no command changes either way."""
    import wbt_wrappers_inference.policies.holosoma_wbt as hw

    assert "parse_unified_windup_warn_n" in _code_of(hw.HolosomaWBTPolicy.__init__)
    p = _stub_unified_policy(force_source=_FakeForceSource())
    imax = float(p._servo_cfg.integrator_max)
    # 10x: even the clip itself on every axis (‖I‖ = 40·√3 ≈ 69 N) stays silent.
    p._integ_windup_warn_n = 200.0
    p._servo_integ[0] = [imax, imax, imax]
    p._warn_on_integrator_windup()
    assert p._integ_windup_warned is False, "200 N is unreachable under the per-axis clip"
    # 0 = off, whatever I does.
    p._integ_windup_warn_n = 0.0
    p._warn_on_integrator_windup()
    assert p._integ_windup_warned is False
    # A small threshold fires where the default would not.
    p._integ_windup_warn_n = 5.0
    p._servo_integ[0] = [0.0, 0.0, 6.0]
    p._warn_on_integrator_windup()
    assert p._integ_windup_warned is True
    # None = the reviewed default (half of I_max): 6 N is below it, so a fresh latch stays quiet …
    p._integ_windup_warned = False
    p._integ_windup_warn_n = None
    p._warn_on_integrator_windup()
    assert p._integ_windup_warned is False
    # … and just above half of I_max fires exactly as before the knob existed.
    p._servo_integ[0] = [0.0, 0.0, 0.5 * imax * 1.01]
    p._warn_on_integrator_windup()
    assert p._integ_windup_warned is True


def test_midrun_reset_ramps_fcmd_instead_of_stepping_it():
    """`select+A` (play clip) runs the same reset funnel while the policy is LIVE. There
    the applied F^cmd must be left to the 2 N/tick slew rather than hard-zeroed, so the
    force_cmd_local / Δx feed-forward does not step in a single tick — the exact thing
    the slew limiter exists to prevent. Only the TARGET goes to zero."""
    p = _stub_unified_policy()
    _set_force_cmd(p, [15.0, 0.0, 0.0], [15.0, 0.0, 0.0])
    p._motion_command_0 = p._motion_command_t.copy()
    p._ref_quat_xyzw_0 = p._ref_quat_xyzw_t.copy()
    p._ref_pos_xyz_0 = p._ref_pos_xyz_t.copy()
    p._ref_vel_xyz_t = np.zeros((1, 3), dtype=np.float32)
    p._ref_vel_xyz_0 = p._ref_vel_xyz_t.copy()
    p._ext_frame_idx = 0
    p._action_ema_state = None
    p._force_cmd_target_raw = p._force_cmd_pelvis_raw.copy()
    p.use_policy_action = True  # still driving motors (mid-run clip start)

    p._reset_motion_state()

    assert np.allclose(p._force_cmd_target_raw, 0.0), "the target must go to zero"
    assert np.allclose(p._force_cmd_pelvis_raw[0], [15.0, 0.0, 0.0]), (
        "the APPLIED F^cmd must be left to the slew mid-run, not stepped"
    )
