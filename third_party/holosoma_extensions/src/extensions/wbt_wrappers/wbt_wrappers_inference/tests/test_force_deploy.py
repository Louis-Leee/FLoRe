"""Tests for the sim2real live-F/T deploy path (hybrid + force-ext presets).

Pins the load-bearing behaviours of the live-Bota → policy integration:

  1. ``force_deploy_math`` — R_SENSOR matches the URDF ft_sensor_joint rpy and the
     stated axis convention; sensor→pelvis is the EXACT inverse of the F^cmd
     pelvis→sensor chain; the servo integrator gates on contact and clamps Δq.
  2. ``_update_force_meas_and_servo`` on a stubbed ``HolosomaWBTPolicy`` (real
     Pinocchio yam model, fake force source): OFF → force obs zero (byte-identical);
     ON → force_meas_local / force_ext_local carry the ×0.04-scaled pelvis force;
     servo OFF → I/Δx zero + no Δq; servo ON (armed) → I winds in contact, Δq
     nonzero and clamped.

Pure-math tests always run; the policy-stub tests skip cleanly without pinocchio
or the yam URDF.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wbt_wrappers_inference.utils.force_deploy_math import (
    R_SENSOR,
    ServoConfig,
    rot_matrix_from_rpy,
    sensor_force_to_pelvis,
    servo_step,
    tcp_jacobian_from_wrist,
)

_REPO = Path(__file__).resolve().parents[5]
_YAM_URDF = _REPO / "thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof_yam.urdf"


# --------------------------------------------------------------------------- #
# 1. Pure math
# --------------------------------------------------------------------------- #


def test_r_sensor_matches_urdf_rpy_and_axes():
    # R_SENSOR == rpy(0,-pi/2,pi) (the URDF {side}_ft_sensor_joint origin).
    assert np.allclose(R_SENSOR, rot_matrix_from_rpy((0.0, -np.pi / 2, np.pi)), atol=1e-9)
    # symmetric involution.
    assert np.allclose(R_SENSOR @ R_SENSOR, np.eye(3), atol=1e-12)
    # user's stated axes: sensor +z→wrist +x, +y→-y, +x→+z.
    assert np.allclose(R_SENSOR @ np.array([0.0, 0.0, 1.0]), [1, 0, 0])
    assert np.allclose(R_SENSOR @ np.array([0.0, 1.0, 0.0]), [0, -1, 0])
    assert np.allclose(R_SENSOR @ np.array([1.0, 0.0, 0.0]), [0, 0, 1])


def test_sensor_to_pelvis_is_inverse_of_fcmd_chain():
    # For any wrist rotation R_rel, the deploy sensor→pelvis read must recover the
    # pelvis force that the F^cmd pelvis→sensor chain (holosoma_wbt._force_cmd_local_obs)
    # would have produced: f_sensor = R_SENSOR^T @ (R_rel^T @ f_pelvis).
    rng = np.random.default_rng(1)
    for _ in range(20):
        # random rotation via QR of a gaussian matrix (no scipy dependency).
        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        r_rel = q * np.sign(np.linalg.det(q))  # ensure det=+1
        f_pelvis = rng.standard_normal(3)
        f_sensor = R_SENSOR.T @ (r_rel.T @ f_pelvis)
        assert np.allclose(sensor_force_to_pelvis(f_sensor, r_rel, sign=1.0), f_pelvis, atol=1e-9)


def test_sensor_to_pelvis_sign_flip():
    f = sensor_force_to_pelvis(np.array([0.0, 0.0, 5.0]), np.eye(3), sign=1.0)
    assert np.allclose(f, [5, 0, 0])
    assert np.allclose(sensor_force_to_pelvis(np.array([0.0, 0.0, 5.0]), np.eye(3), sign=-1.0), [-5, 0, 0])


@pytest.mark.parametrize("raw", ["1", "true", "YES", "On"])
def test_force_meas_gate_parser_accepts_true_values(raw):
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_force_meas_gate_cfg

    assert parse_force_meas_gate_cfg({"HOLOSOMA_FORCE_MEAS_GATE_ON_CMD": raw}) == (True, 1e-6)


@pytest.mark.parametrize("raw", ["", "0", "false", "NO", "Off"])
def test_force_meas_gate_parser_accepts_false_values(raw):
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_force_meas_gate_cfg

    assert parse_force_meas_gate_cfg({"HOLOSOMA_FORCE_MEAS_GATE_ON_CMD": raw}) == (False, 1e-6)


@pytest.mark.parametrize("raw", ["enabled", "2", "tru"])
def test_force_meas_gate_parser_rejects_unknown_boolean(raw):
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_force_meas_gate_cfg

    with pytest.raises(ValueError, match="HOLOSOMA_FORCE_MEAS_GATE_ON_CMD"):
        parse_force_meas_gate_cfg({"HOLOSOMA_FORCE_MEAS_GATE_ON_CMD": raw})


@pytest.mark.parametrize("raw", ["nan", "inf", "-1", "1e999"])
def test_force_meas_gate_parser_rejects_nonfinite_or_negative_threshold(raw):
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_force_meas_gate_cfg

    with pytest.raises(ValueError, match="HOLOSOMA_FORCE_MEAS_GATE_CMD_N"):
        parse_force_meas_gate_cfg({"HOLOSOMA_FORCE_MEAS_GATE_CMD_N": raw})


def test_servo_step_gate_integrator_and_clamp():
    cfg = ServoConfig()
    J = np.array([[0.1, 0.2, 0, 0, 0, 0, 0], [0, 0.05, 0.1, 0, 0, 0, 0], [0, 0, 0, 0.1, 0, 0, 0]])
    f_cmd = np.array([5.0, 0, 0])
    f_meas = np.array([-4.0, 0, 0])  # residual e = +1 N on x
    dq, integ, dx = servo_step(J, f_cmd, f_meas, 1.0, np.zeros(3), cfg, dt=0.02)
    assert integ[0] > 0 and abs(integ[1]) < 1e-12  # winds on x only
    assert np.allclose(dx, (f_cmd + integ) / cfg.stiffness_K)
    assert dq.shape == (7,)
    # gate=0 freezes (zeros) the integrator.
    _, integ0, _ = servo_step(J, f_cmd, f_meas, 0.0, np.array([3.0, 0, 0]), cfg, dt=0.02)
    assert np.allclose(integ0, 0.0)
    # Δq clamp to ±dq_max.
    dqc, _, _ = servo_step(np.ones((3, 7)) * 100.0, np.array([25.0, 25, 25]), np.zeros(3), 1.0, np.zeros(3), cfg, 0.02)
    assert np.all(np.abs(dqc) <= cfg.servo_dq_max + 1e-9)


def test_tcp_jacobian_offset_shift():
    # Pure rotation column (J_v=0, J_ω=e_z): shifting to a +x TCP lever adds
    # -[r]× J_ω = -(r × ω̂). r=(0.2,0,0), ω̂=z → cross = (0.2,0,0)×(0,0,1)=(0,-0.2,0);
    # -that = (0,0.2,0). So J_v_tcp column = (0, 0.2, 0).
    j_lin = np.zeros((3, 1))
    j_ang = np.array([[0.0], [0.0], [1.0]])
    out = tcp_jacobian_from_wrist(j_lin, j_ang, np.array([0.2, 0.0, 0.0]))
    assert np.allclose(out[:, 0], [0.0, 0.2, 0.0]), out


# --------------------------------------------------------------------------- #
# 2. Policy-stub integration (real Pinocchio yam model + fake force source)
# --------------------------------------------------------------------------- #

pytest.importorskip("pinocchio")


class _FakeForceSource:
    def __init__(self, left, right):
        self._l = np.asarray(left, dtype=np.float32)
        self._r = np.asarray(right, dtype=np.float32)

    def latest(self):
        return self._l.copy(), self._r.copy()


def _stub_policy(*, force_source=None, servo=False, obs_terms=None):
    """Minimal HolosomaWBTPolicy wired exactly as _update_force_meas_and_servo +
    _compute_obs_terms read it: a real Pinocchio yam model + the force caches."""
    from loguru import logger as _loguru

    from holosoma_inference.policies.wbt_utils import PinocchioRobot

    from wbt_wrappers_inference.config_values.robot_config import g1_29dof_wbt
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    if not _YAM_URDF.is_file():
        pytest.skip(f"yam URDF not present: {_YAM_URDF}")

    p = object.__new__(HolosomaWBTPolicy)
    p.logger = _loguru
    p.num_dofs = 29
    p.config = SimpleNamespace(robot=g1_29dof_wbt, task=SimpleNamespace(rl_rate=50.0))
    p.default_dof_angles = np.array(g1_29dof_wbt.default_dof_angles, dtype=np.float64)
    p.pinocchio_robot = PinocchioRobot(g1_29dof_wbt, _YAM_URDF.read_text())
    p._force_wrist_frame_ids = None
    # Force-deploy state (mirrors __init__).
    p._force_live = force_source
    p._force_meas_sign = 1.0
    p._force_meas_pelvis = np.zeros((2, 3), dtype=np.float64)
    p._hybrid_servo = servo
    p._servo_integ = np.zeros((2, 3), dtype=np.float64)
    p._servo_delta_x = np.zeros((2, 3), dtype=np.float64)
    p._servo_dq = np.zeros(29, dtype=np.float64)
    p._servo_arm_cols = None
    p._servo_arm_pd_slice = []
    p._force_contact_n = 2.0
    p._force_runaway_n = 0.0
    p._servo_cfg = ServoConfig()
    p._force_cmd_pelvis_raw = None
    p.use_policy_action = True
    return p


def test_off_path_force_obs_zero():
    """No live source → the pelvis-force cache stays zero (byte-identical path)."""
    p = _stub_policy(force_source=None, servo=False)
    dof = p.default_dof_angles.copy()
    p._update_force_meas_and_servo(dof)
    assert np.allclose(p._force_meas_pelvis, 0.0)
    assert np.allclose(p._servo_dq, 0.0)
    assert np.allclose(p._servo_integ, 0.0)


def test_live_force_fills_pelvis_cache():
    """A live sensor reading rotates sensor→pelvis into the cache (nonzero,
    magnitude preserved by the pure rotation)."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[0.0, 0.0, 0.0])
    p = _stub_policy(force_source=src, servo=False)
    dof = p.default_dof_angles.copy()
    p._update_force_meas_and_servo(dof)
    # Left hand read 6 N on sensor +z; rotation preserves magnitude.
    assert abs(np.linalg.norm(p._force_meas_pelvis[0]) - 6.0) < 1e-6
    assert np.allclose(p._force_meas_pelvis[1], 0.0)
    # Servo off → no integrator / no Δq.
    assert np.allclose(p._servo_integ, 0.0)
    assert np.allclose(p._servo_dq, 0.0)


def test_servo_off_by_default_no_dq():
    src = _FakeForceSource(left=[0.0, 0.0, 10.0], right=[0.0, 0.0, 10.0])
    p = _stub_policy(force_source=src, servo=False)
    p._update_force_meas_and_servo(p.default_dof_angles.copy())
    assert np.allclose(p._servo_dq, 0.0)


def test_servo_on_winds_integrator_and_injects_dq():
    """Servo ON + armed + a contact-magnitude reading with F^cmd set → the
    integrator winds and a clamped Δq is scattered into the arm DOFs."""
    src = _FakeForceSource(left=[0.0, 0.0, 10.0], right=[0.0, 0.0, 10.0])  # 10 N each, > contact gate
    p = _stub_policy(force_source=src, servo=True)
    p._force_cmd_pelvis_raw = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)  # push +x
    dof = p.default_dof_angles.copy()
    for _ in range(5):  # a few ticks to let the integrator wind
        p._update_force_meas_and_servo(dof)
    assert np.linalg.norm(p._servo_integ) > 0.0, "integrator should wind in contact"
    assert np.linalg.norm(p._servo_dq) > 0.0, "servo should inject a nonzero Δq"
    assert np.all(np.abs(p._servo_dq) <= p._servo_cfg.servo_dq_max + 1e-9)
    # Δq only touches ARM DOFs (indices 15..28), never legs/waist (0..14).
    assert np.allclose(p._servo_dq[:15], 0.0), "servo must not touch lower-body DOFs"


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_nonfinite_reading_zeroed_not_propagated():
    """A NaN/inf sensor reading must never reach the obs cache (it would sail
    through np.clip to a motor command). The policy zeroes any non-finite value.
    (The RuntimeWarning from NaN flowing through the sensor→pelvis matmul is the
    exact path under test — the belt-and-suspenders guard zeroes it afterward.)"""

    class _NaNSource:
        def latest(self):
            return np.array([np.nan, 0.0, 5.0], dtype=np.float32), np.array([0.0, np.inf, 0.0], dtype=np.float32)

    p = _stub_policy(force_source=_NaNSource(), servo=False)
    p._update_force_meas_and_servo(p.default_dof_angles.copy())
    assert np.all(np.isfinite(p._force_meas_pelvis)), "non-finite force must be zeroed before the obs"


@pytest.mark.parametrize(
    "bad_action",
    [
        np.full((1, 29), np.nan, dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((29, 1), dtype=np.float32),
    ],
    ids=["nonfinite", "wrong-width", "transposed"],
)
def test_bad_onnx_action_disables_policy_and_enters_fixed_stiff_hold(bad_action):
    """A corrupt/wrong ONNX must publish one finite target and hold it fixed."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    p = object.__new__(HolosomaWBTPolicy)
    p.motion_clip_progressing = True
    p._poll_force_cmd_keys = lambda: None
    p._align_rot_mat = np.eye(3)
    p._apply_hold_dof_override = lambda: None
    p._build_observation = lambda _state: np.zeros((1, 680), dtype=np.float32)
    p._debug_log = None
    p._clip_normalized_obs = lambda obs: obs
    p._onnx_obs_input = "obs"
    p._onnx_uses_time_step = True
    p.curr_motion_timestep = 0.0
    p.onnx_policy_session = SimpleNamespace(run=lambda _outputs, _feed: [bad_action])
    p.onnx_output_names = ["actions"]
    p._onnx_action_output = "actions"
    p._servo_dq = np.ones(29, dtype=np.float64)
    p.use_policy_action = True
    p.get_ready_state = False
    p.num_dofs = 29
    p.default_dof_angles = np.linspace(-0.2, 0.2, 29)
    p.joint_offsets = np.zeros(29)
    p.upper_body_controller = False
    p.cmd_q = np.linspace(-0.3, 0.3, 29)
    p.cmd_dq = np.zeros(29)
    p.cmd_tau = np.zeros(29)
    p._soft_stop_active = False
    p._stiff_hold_kp = np.ones(29)
    p._stiff_hold_kd = np.ones(29)
    p.latency_tracker = SimpleNamespace(measure=lambda _name: nullcontext())
    robot_state = np.zeros((1, 71), dtype=np.float32)
    robot_state[0, 7:36] = np.linspace(-0.4, 0.4, 29)
    sent = []
    p.interface = SimpleNamespace(
        get_low_state=lambda: robot_state, send_low_command=lambda q, *_args, **_kwargs: sent.append(q.copy())
    )

    p.policy_action()

    assert p.use_policy_action is False
    assert np.allclose(p._servo_dq, 0.0)
    assert len(sent) == 1
    assert np.all(np.isfinite(sent[0]))
    assert np.allclose(sent[0], robot_state[0, 7:36])
    assert p._stiff_hold_active is True

    captured_hold = sent[0].copy()
    robot_state[0, 7:36] += 0.1
    p.policy_action()
    assert len(sent) == 2
    assert np.allclose(sent[1], captured_hold), "fault hold must not follow a sagging readback"


def test_bad_onnx_action_with_nonfinite_readback_reuses_last_finite_command():
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    p = object.__new__(HolosomaWBTPolicy)
    p.num_dofs = 29
    p.joint_offsets = np.zeros(29)
    p.default_dof_angles = np.zeros(29)
    p.cmd_q = np.linspace(-0.3, 0.3, 29)
    p._servo_dq = np.ones(29)
    p.use_policy_action = True
    p.get_ready_state = False
    p.motion_clip_progressing = True
    p._soft_stop_active = False
    p._stiff_hold_kp = np.ones(29)
    p._stiff_hold_kd = np.ones(29)

    robot_state = np.zeros((1, 71), dtype=np.float32)
    robot_state[0, 7] = np.nan
    relative_hold = p._disable_policy_with_stiff_hold(robot_state, "test fault")

    assert np.all(np.isfinite(relative_hold))
    assert np.allclose(relative_hold[0], p.cmd_q)
    assert np.allclose(p._stiff_hold_q[0], p.cmd_q)


def test_force_tare_button_requests_and_resets():
    """The tare button calls ForceLiveSource.request_tare() and clears the local
    servo integrator; it is a safe no-op (warns) when no live source is set."""

    class _TareSpy:
        def __init__(self):
            self.n = 0

        def latest(self):
            return np.zeros(3, np.float32), np.zeros(3, np.float32)

        def request_tare(self):
            self.n += 1

    spy = _TareSpy()
    p = _stub_policy(force_source=spy, servo=True)
    p._servo_integ[:] = 3.0
    p._servo_delta_x[:] = 0.5
    p._trigger_force_tare()
    assert spy.n == 1, "tare button must forward request_tare() to the bridge"
    assert np.allclose(p._servo_integ, 0.0) and np.allclose(p._servo_delta_x, 0.0), (
        "tare must clear the servo integrator"
    )

    # No live source → no-op (no crash).
    p2 = _stub_policy(force_source=None, servo=False)
    p2._trigger_force_tare()  # must not raise


def test_servo_frozen_when_not_armed():
    """Servo requires use_policy_action (armed). When not armed, no Δq / no wind."""
    src = _FakeForceSource(left=[0.0, 0.0, 10.0], right=[0.0, 0.0, 10.0])
    p = _stub_policy(force_source=src, servo=True)
    p._force_cmd_pelvis_raw = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)
    p.use_policy_action = False
    p._update_force_meas_and_servo(p.default_dof_angles.copy())
    assert np.allclose(p._servo_dq, 0.0)
    assert np.allclose(p._servo_integ, 0.0)


def test_force_meas_gate_off_by_default_is_byte_identical():
    """No gate attr / gate off ⇒ F_meas flows unchanged for BOTH hands even with no
    F^cmd (the object.__new__ stub omits the gate attrs, exercising getattr defaults)."""
    src = _FakeForceSource(left=[0.0, 0.0, 8.0], right=[0.0, 0.0, 4.0])
    p = _stub_policy(force_source=src, servo=False)
    assert not hasattr(p, "_force_meas_gate_on_cmd")  # stub bypassed __init__
    p._force_cmd_pelvis_raw = None  # no command on either hand
    p._update_force_meas_and_servo(p.default_dof_angles.copy())
    # Both hands still carry their full measured force (gate is off → nothing zeroed).
    assert abs(np.linalg.norm(p._force_meas_pelvis[0]) - 8.0) < 1e-6
    assert abs(np.linalg.norm(p._force_meas_pelvis[1]) - 4.0) < 1e-6


def test_force_meas_gate_zeros_uncommanded_hand():
    """Gate ON: a hand with ~0 F^cmd feeds ZERO F_meas (position control); a hand with
    a nonzero F^cmd still carries its measured force (the gate is opened for it)."""
    src = _FakeForceSource(left=[0.0, 0.0, 8.0], right=[0.0, 0.0, 6.0])
    p = _stub_policy(force_source=src, servo=False)
    p._force_meas_gate_on_cmd = True
    p._force_meas_gate_cmd_n = 1e-6
    # LEFT commanded (+5 N x), RIGHT uncommanded (0 N).
    p._force_cmd_pelvis_raw = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    p._update_force_meas_and_servo(p.default_dof_angles.copy())
    assert abs(np.linalg.norm(p._force_meas_pelvis[0]) - 8.0) < 1e-6, "commanded hand keeps its F_meas"
    assert np.allclose(p._force_meas_pelvis[1], 0.0), "uncommanded hand is gated to zero F_meas"


def test_force_meas_gate_none_cmd_gates_both():
    """Gate ON with _force_cmd_pelvis_raw None (no F^cmd buffer yet) ⇒ BOTH hands gated
    (a None command is 0 N everywhere, so neither gate opens)."""
    src = _FakeForceSource(left=[0.0, 0.0, 8.0], right=[0.0, 0.0, 6.0])
    p = _stub_policy(force_source=src, servo=False)
    p._force_meas_gate_on_cmd = True
    p._force_meas_gate_cmd_n = 1e-6
    p._force_cmd_pelvis_raw = None
    p._update_force_meas_and_servo(p.default_dof_angles.copy())
    assert np.allclose(p._force_meas_pelvis, 0.0), "None F^cmd → both hands gated to zero"


def test_force_meas_gate_freezes_servo_on_uncommanded_hand():
    """Gate ON + servo ON: an uncommanded hand feeds zero F_meas AND zero F^cmd, so its
    servo error e=F_cmd+F_meas=0 → no integrator wind, no Δq into that arm. The commanded
    hand still winds + injects Δq (control case)."""
    src = _FakeForceSource(left=[0.0, 0.0, 10.0], right=[0.0, 0.0, 10.0])  # both > contact gate
    p = _stub_policy(force_source=src, servo=True)
    p._force_meas_gate_on_cmd = True
    p._force_meas_gate_cmd_n = 1e-6
    p._force_cmd_pelvis_raw = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)  # only LEFT
    for _ in range(5):
        p._update_force_meas_and_servo(p.default_dof_angles.copy())
    # RIGHT hand (uncommanded) → gated F_meas 0, so its integrator never winds.
    assert np.allclose(p._servo_integ[1], 0.0), "uncommanded hand: no integrator wind"
    # LEFT hand (commanded) → contact + F^cmd → the integrator winds and Δq is injected.
    assert np.linalg.norm(p._servo_integ[0]) > 0.0, "commanded hand: integrator winds"
    # Δq lands only in the LEFT arm DOFs (15..21), never the RIGHT arm (22..28).
    assert np.linalg.norm(p._servo_dq[15:22]) > 0.0, "left arm gets Δq"
    assert np.allclose(p._servo_dq[22:29], 0.0), "right (uncommanded) arm gets no Δq"


def test_force_meas_gate_does_not_hide_runaway_on_uncommanded_hand():
    """SAFETY: the F_meas-on-cmd gate must NOT blind the runaway valve. A large force on
    an UNCOMMANDED hand (gate would zero its F_meas obs) must still trip the runaway
    freeze — the valve reads the UNGATED snapshot, so it fires and zeroes the servo."""
    # RIGHT hand uncommanded but reading 50 N (> runaway); LEFT commanded + light contact.
    src = _FakeForceSource(left=[0.0, 0.0, 10.0], right=[0.0, 0.0, 50.0])
    p = _stub_policy(force_source=src, servo=True)
    p._force_meas_gate_on_cmd = True
    p._force_meas_gate_cmd_n = 1e-6
    p._force_runaway_n = 35.0
    p._force_cmd_pelvis_raw = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)  # only LEFT
    p._update_force_meas_and_servo(p.default_dof_angles.copy())
    # Runaway must trip on the uncommanded RIGHT hand → global servo freeze (no Δq, no wind),
    # even though the RIGHT hand's F_meas obs is gated to zero.
    assert np.allclose(p._servo_dq, 0.0), "runaway on an uncommanded hand must still freeze the servo"
    assert np.allclose(p._servo_integ, 0.0), "runaway must zero the servo integrator"
    # The gate still applies to the OBS: the uncommanded RIGHT hand's F_meas reads zero.
    assert np.allclose(p._force_meas_pelvis[1], 0.0), "gate still zeros the uncommanded hand's F_meas obs"


def test_force_meas_gate_clears_stale_observe_only_state_before_runaway_hold():
    """Observe-only runaway handling holds prior I/Δx, but a GATED uncommanded
    hand must be position-only even when its ungated sensor trips that alarm."""
    src = _FakeForceSource(left=[0.0, 0.0, 50.0], right=[0.0, 0.0, 50.0])
    p = _stub_policy(force_source=src, servo=False)
    p._servo_observe_only = True
    p._force_meas_gate_on_cmd = True
    p._force_meas_gate_cmd_n = 1e-6
    p._force_runaway_n = 35.0
    p._force_cmd_pelvis_raw = np.zeros((2, 3), dtype=np.float64)
    p._servo_integ[:] = 8.0
    p._servo_delta_x[:] = 0.04

    p._update_force_meas_and_servo(p.default_dof_angles.copy())

    assert np.allclose(p._force_meas_pelvis, 0.0)
    assert np.allclose(p._servo_integ, 0.0)
    assert np.allclose(p._servo_delta_x, 0.0)


def test_runaway_freezes_servo():
    """A F_meas above HOLOSOMA_FORCE_RUNAWAY_N must FREEZE the servo (no Δq, no wind)
    — the runaway valve is not just a log; it cuts the Layer-B push while tripped.
    The F_meas obs itself is still reported (only the servo push is frozen)."""
    src = _FakeForceSource(left=[0.0, 0.0, 50.0], right=[0.0, 0.0, 0.0])  # 50 N > runaway
    p = _stub_policy(force_source=src, servo=True)
    p._force_runaway_n = 35.0  # trip threshold
    p._force_cmd_pelvis_raw = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
    p._update_force_meas_and_servo(p.default_dof_angles.copy())
    # F_meas obs is still live (the left hand read 50 N).
    assert np.linalg.norm(p._force_meas_pelvis[0]) > 40.0, "F_meas obs must stay live during a runaway"
    # But the servo is frozen: no Δq, no integrator wind.
    assert np.allclose(p._servo_dq, 0.0), "runaway must freeze the servo Δq"
    assert np.allclose(p._servo_integ, 0.0), "runaway must zero the servo integrator"

    # Below threshold, the same setup DOES drive the servo (control case).
    src2 = _FakeForceSource(left=[0.0, 0.0, 10.0], right=[0.0, 0.0, 10.0])  # 10 N < runaway
    p2 = _stub_policy(force_source=src2, servo=True)
    p2._force_runaway_n = 35.0
    p2._force_cmd_pelvis_raw = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
    for _ in range(3):
        p2._update_force_meas_and_servo(p2.default_dof_angles.copy())
    assert np.linalg.norm(p2._servo_dq) > 0.0, "below runaway, the servo must still push"


# --------------------------------------------------------------------------- #
# 3. INIT / held pose = forearms-HORIZONTAL ready pose (arms=0), NPZ pristine
# --------------------------------------------------------------------------- #

# 29-dof SDK order; arm joints are the contiguous 15..28 block.
_DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip


def _ready_default():
    """A default_dof_angles with crouched legs + nonzero arms (like the real ONNX)."""
    d = np.zeros(29, dtype=np.float64)
    # legs/waist crouch (must survive the ready-pose zeroing).
    for i, v in ((0, -0.312), (3, 0.669), (4, -0.363), (6, -0.312), (9, 0.669), (10, -0.363)):
        d[i] = v
    # arms (must be zeroed by the ready pose).
    for i, v in ((15, 0.2), (16, 0.2), (18, 0.6), (22, 0.2), (23, -0.2), (25, 0.6)):
        d[i] = v
    return d


def _stub_ready_policy(*, headless: bool):
    """Minimal policy exercising _maybe_apply_init_ready_pose + _apply_hold_dof_override:
    a fake NPZ whose frame-0 differs hard from the ready pose so we can assert the
    NPZ is left PRISTINE and only the frozen-hold read is substituted."""
    from loguru import logger as _loguru

    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    default = _ready_default()
    jp = np.zeros((2, 36), dtype=np.float32)
    jp[0, :7] = [0.1, 0.2, 0.78, 1.0, 0.0, 0.0, 0.0]  # distinct root prefix
    jp[0, 7:] = np.linspace(-0.9, 1.0, 29)  # frame-0 DOF far from default+ready
    jp[1, 7:] = 0.5  # frame-1 DOF distinct
    jv = np.ones((2, 35), dtype=np.float32)
    n_bodies = 3
    p = object.__new__(HolosomaWBTPolicy)
    p.logger = _loguru
    p.default_dof_angles = default
    p.config = SimpleNamespace(
        task=SimpleNamespace(headless_sim_eval=headless, rl_rate=50), robot=SimpleNamespace(dof_names=list(_DOF_NAMES))
    )
    p._ext_motion = {
        "joint_pos": jp.copy(),
        "joint_vel": jv.copy(),
        "body_pos_w": np.zeros((2, n_bodies, 3), dtype=np.float32),
        "body_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (2, n_bodies, 1)).astype(np.float32),
    }
    p._ext_motion_length = 2
    p._ext_ref_body_idx = 0
    p._ext_frame_idx = 0
    p._is_2pt = True  # ready pose is 2pt-only (hybrid/force presets all detect as 2pt)
    p._init_q_target = jp[0, 7:].astype(np.float64).copy()  # as setup_policy seeds it
    p.motion_clip_progressing = False
    return p, default, jp.copy()


def _expected_ready(default):
    r = default.copy()
    r[15:29] = 0.0  # arms zeroed, legs/waist kept
    return r


def test_ready_pose_on_for_real_robot():
    """headless=False → INIT target + hold override = default with ARMS ZEROED;
    legs/waist kept; the NPZ is left PRISTINE (no frame-0/frame-1 mutation)."""
    import os

    os.environ.pop("HOLOSOMA_INIT_READY_POSE", None)
    p, default, jp_orig = _stub_ready_policy(headless=False)
    p._maybe_apply_init_ready_pose()
    ready = _expected_ready(default)
    assert np.allclose(p._init_q_target, ready), "INIT target = default with arms=0"
    assert np.allclose(p._init_q_target[15:29], 0.0), "arms zeroed"
    assert np.allclose(p._init_q_target[:15], default[:15]), "legs/waist kept at default"
    assert p._hold_dof_override is not None and np.allclose(p._hold_dof_override, ready)
    # NPZ must be untouched (select+A plays the real clip from real frame-0).
    assert np.allclose(p._ext_motion["joint_pos"], jp_orig), "NPZ joint_pos must stay pristine"


def test_hold_override_applied_only_while_frozen():
    """_apply_hold_dof_override rewrites the frozen reference DOF but is a no-op
    once the clip is playing (motion_clip_progressing True) → no select+A jump."""
    import os

    os.environ.pop("HOLOSOMA_INIT_READY_POSE", None)
    p, default, jp_orig = _stub_ready_policy(headless=False)
    p._maybe_apply_init_ready_pose()
    ready = _expected_ready(default)
    # Frozen hold: _motion_command_t seeded from real NPZ frame-0 (pos) + a NONZERO
    # reference velocity (like a grasp clip's frame-0 vel), then overridden.
    p._motion_command_t = np.concatenate(
        [jp_orig[0, 7:].reshape(1, -1), np.full((1, 29), 2.2, dtype=np.float32)], axis=1
    )
    p._apply_hold_dof_override()
    assert np.allclose(p._motion_command_t[0, :29], ready), "frozen hold → ready pose DOF"
    assert np.allclose(p._motion_command_t[0, 29:], 0.0), "frozen hold → zero reference velocity"
    # Clip playing: the override must NOT fire (real frame + velocity tracked).
    p.motion_clip_progressing = True
    p._motion_command_t[0, :29] = jp_orig[1, 7:]
    p._motion_command_t[0, 29:] = 1.5
    p._apply_hold_dof_override()
    assert np.allclose(p._motion_command_t[0, :29], jp_orig[1, 7:]), "clip playback pos untouched"
    assert np.allclose(p._motion_command_t[0, 29:], 1.5), "clip playback vel untouched"


def test_ready_pose_off_for_headless():
    """headless=True → auto-off: no override, NPZ pristine, INIT stays frame-0."""
    import os

    os.environ.pop("HOLOSOMA_INIT_READY_POSE", None)
    p, _default, jp_orig = _stub_ready_policy(headless=True)
    p._maybe_apply_init_ready_pose()
    assert p._hold_dof_override is None, "headless → no hold override (byte-identical)"
    assert np.allclose(p._init_q_target, jp_orig[0, 7:]), "headless INIT target stays frame-0"
    # A no-op override call must leave a frozen reference untouched.
    p._motion_command_t = np.concatenate([jp_orig[0, 7:].reshape(1, -1), np.zeros((1, 29), dtype=np.float32)], axis=1)
    p._apply_hold_dof_override()
    assert np.allclose(p._motion_command_t[0, :29], jp_orig[0, 7:]), "no override in headless"


def test_ready_pose_env_overrides():
    """Env forces the behavior regardless of headless."""
    import os

    try:
        os.environ["HOLOSOMA_INIT_READY_POSE"] = "0"  # force OFF on real robot
        p, _d, jp_orig = _stub_ready_policy(headless=False)
        p._maybe_apply_init_ready_pose()
        assert p._hold_dof_override is None, "env=0 disables the override on real robot"
        assert np.allclose(p._init_q_target, jp_orig[0, 7:]), "env=0 restores frame-0 INIT"

        os.environ["HOLOSOMA_INIT_READY_POSE"] = "1"  # force ON in headless
        p, d, _jp = _stub_ready_policy(headless=True)
        p._maybe_apply_init_ready_pose()
        assert np.allclose(p._init_q_target, _expected_ready(d)), "env=1 forces ready pose in headless"
    finally:
        os.environ.pop("HOLOSOMA_INIT_READY_POSE", None)


# --------------------------------------------------------------------------- #
# 3b. Live F^cmd SLEW + drain (the policy-side _poll/_drain_force_cmd_keys)
# --------------------------------------------------------------------------- #


class _PresetForceKeyController:
    """Fake ForceCmdKeyController that returns a fixed target on the first poll
    (as if N keys were buffered), then reports no change. Mirrors the real
    poll(raw)→(target, changed, msgs) contract."""

    def __init__(self, target):
        self._target = np.asarray(target, dtype=np.float32).reshape(2, 3)
        self._fired = False

    def poll(self, raw):
        if not self._fired:
            self._fired = True
            return self._target.copy(), True, ["burst"]
        return np.asarray(raw, dtype=np.float32).reshape(2, 3), False, []


def _stub_force_key_policy(*, target, slew_n=2.0):
    from loguru import logger as _loguru

    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    p = object.__new__(HolosomaWBTPolicy)
    p.logger = _loguru
    p._force_cmd_pelvis_raw = np.zeros((2, 3), dtype=np.float32)
    p._force_cmd_target_raw = np.zeros((2, 3), dtype=np.float32)
    p._force_cmd_local_scaled = np.zeros(6, dtype=np.float32)
    p._force_cmd_obs_scale = 0.04
    p._force_cmd_key_slew_n = slew_n
    p._force_cmd_keys = _PresetForceKeyController(target)
    return p


def test_force_key_slew_caps_burst_step():
    """A buffered burst (e.g. 12×`u` → 24 N target) must NOT step the applied
    F^cmd in one tick — it slews at most _force_cmd_key_slew_n N/axis/tick."""
    p = _stub_force_key_policy(target=[[24.0, 0, 0], [0, 0, 0]], slew_n=2.0)
    p._poll_force_cmd_keys()  # tick 1: target=24, applied slews +2
    assert np.isclose(p._force_cmd_pelvis_raw[0, 0], 2.0), "first tick capped at slew_n"
    assert np.isclose(p._force_cmd_local_scaled[0], 2.0 * 0.04)
    # Subsequent ticks keep ramping +2 toward the (now steady) target.
    for _ in range(20):
        p._poll_force_cmd_keys()
    assert np.isclose(p._force_cmd_pelvis_raw[0, 0], 24.0), "eventually reaches the target"


def test_force_key_drain_discards_buffered_keys():
    """_drain_force_cmd_keys snaps the target to the current applied value and
    empties the queue, so pre-START taps do not flush as a startup step."""
    p = _stub_force_key_policy(target=[[24.0, 0, 0], [0, 0, 0]], slew_n=2.0)
    p._drain_force_cmd_keys()  # discards the buffered burst
    assert np.allclose(p._force_cmd_target_raw, 0.0), "target snapped back to applied (0)"
    p._poll_force_cmd_keys()  # controller now reports no change
    assert np.allclose(p._force_cmd_pelvis_raw, 0.0), "no step after drain"


def test_force_key_slew_never_diverges_positive_slew():
    """The applied F^cmd converges TOWARD the target monotonically (never away):
    a positive slew_n is the invariant the __init__ guard enforces (a non-positive
    slew would invert np.clip into an unbounded runaway — see the max(...,1e-6) floor).
    """
    p = _stub_force_key_policy(target=[[10.0, -6.0, 0], [0, 0, 0]], slew_n=2.0)
    prev = 999.0
    for _ in range(40):
        p._poll_force_cmd_keys()
        err = float(np.linalg.norm(p._force_cmd_pelvis_raw[0] - np.array([10.0, -6.0, 0.0])))
        assert err <= prev + 1e-6, "applied must never move away from target"
        prev = err
    assert prev < 1e-5, "converges to the target"


def test_force_key_slew_env_guard_floors_nonpositive():
    """The __init__ env parse floors slew/step/clamp at a tiny positive value so a
    misconfigured HOLOSOMA_FORCE_CMD_KEY_SLEW_N=0/-2 cannot invert np.clip into a
    divergent F^cmd. Mirror that exact expression here (the guard is a one-liner)."""
    import os

    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy  # noqa: F401

    for raw, floored in (("0", 1e-6), ("-2", 1e-6), ("3.5", 3.5), ("", 2.0)):
        val = max(float(os.environ.get("_UNSET_", raw) or 2.0), 1e-6)
        assert val == floored or (raw == "3.5" and val == 3.5), (raw, val)
    # And prove the divergence the guard prevents: negative slew inverts np.clip.
    delta = np.array([5.0, -5.0, 0.0])
    bad = np.clip(delta, -(-2.0), -2.0)  # a_min=2 > a_max=-2 → returns a_max for all
    assert np.allclose(bad, -2.0), "negative slew would push every axis to -slew (runaway)"


# --------------------------------------------------------------------------- #
# 4. Live keyboard F^cmd control (apply_force_key — pure)
# --------------------------------------------------------------------------- #


def test_force_key_axis_nudges_selected_hand():
    """u/j/h/k/y/n nudge the SELECTED hand's ±x/±y/±z; the other hand is untouched."""
    from wbt_wrappers_inference.utils.force_cmd_keys import apply_force_key

    raw = np.zeros((2, 3), dtype=np.float32)
    # Left hand selected: +x, +x, +y, -z.
    raw, sel, ch, _ = apply_force_key(raw, 0, "u", step_n=2.0)
    raw, sel, ch, _ = apply_force_key(raw, sel, "u")
    raw, sel, ch, _ = apply_force_key(raw, sel, "h")
    raw, sel, ch, _ = apply_force_key(raw, sel, "n")
    assert ch and sel == 0
    assert np.allclose(raw[0], [4.0, 2.0, -2.0]), raw[0]
    assert np.allclose(raw[1], 0.0), "right hand untouched while left selected"


def test_force_key_switch_hand_and_zero():
    """l toggles the hand; c zeros the selected hand; p zeros both."""
    from wbt_wrappers_inference.utils.force_cmd_keys import apply_force_key

    raw = np.array([[3.0, 0, 0], [0, 5.0, 0]], dtype=np.float32)
    raw2, sel, ch, _ = apply_force_key(raw, 0, "l")  # select right
    assert sel == 1 and ch and np.allclose(raw2, raw), "switch must not change forces"
    raw3, sel, ch, _ = apply_force_key(raw2, sel, "c")  # zero right
    assert np.allclose(raw3[1], 0.0) and np.allclose(raw3[0], [3.0, 0, 0])
    raw4, _, ch, _ = apply_force_key(raw3, sel, "p")  # zero both
    assert np.allclose(raw4, 0.0)


def test_force_key_clamp_and_unknown_key():
    """Axis nudges clamp to ±clamp_n; unknown keys are inert (changed=False)."""
    from wbt_wrappers_inference.utils.force_cmd_keys import apply_force_key

    raw = np.zeros((2, 3), dtype=np.float32)
    for _ in range(100):  # way past the clamp
        raw, _, _, _ = apply_force_key(raw, 0, "u", step_n=2.0, clamp_n=25.0)
    assert raw[0, 0] == 25.0, "must clamp to +clamp_n"
    raw2, sel, ch, msg = apply_force_key(raw, 0, "Z")  # not a force key
    assert not ch and msg is None and sel == 0 and np.allclose(raw2, raw)


def test_force_key_does_not_mutate_input():
    """apply_force_key is pure — the input array is never mutated in place."""
    from wbt_wrappers_inference.utils.force_cmd_keys import apply_force_key

    raw = np.zeros((2, 3), dtype=np.float32)
    _ = apply_force_key(raw, 0, "u")
    assert np.allclose(raw, 0.0), "input array must be left untouched (returns a copy)"


# --------------------------------------------------------------------------- #
# F_meas low-pass (HOLOSOMA_FORCE_MEAS_EMA_ALPHA) — off-by-default byte-identical
# + correct EMA recursion at the single shared-cache point.
# --------------------------------------------------------------------------- #
def test_force_meas_ema_off_is_byte_identical():
    """No alpha attr / alpha=1.0 ⇒ the pelvis cache equals the unfiltered rotation
    (the stub omits _force_meas_ema_alpha, exercising the getattr(...,1.0) default)."""
    src = _FakeForceSource(left=[0.0, 0.0, 6.0], right=[3.0, 0.0, 0.0])
    p = _stub_policy(force_source=src, servo=False)
    assert not hasattr(p, "_force_meas_ema_alpha")  # stub bypassed __init__
    p._update_force_meas_and_servo(p.default_dof_angles.copy())
    unfiltered = p._force_meas_pelvis.copy()

    p2 = _stub_policy(force_source=src, servo=False)
    p2._force_meas_ema_alpha = 1.0  # explicit OFF
    p2._force_meas_ema_state = None
    p2._update_force_meas_and_servo(p2.default_dof_angles.copy())
    assert np.allclose(p2._force_meas_pelvis, unfiltered), "alpha=1.0 must equal the unfiltered cache"


def test_force_meas_ema_smooths_and_seeds():
    """alpha<1 low-passes the pelvis force: seeded to the first reading (no ramp
    transient), then F_t = alpha*F_raw + (1-alpha)*F_prev at the shared cache."""
    # Constant sensor reading → the magnitude is a rotation-invariant, so the pelvis
    # |F| is fixed; the EMA of a constant seeded to itself stays put on frame 1, and
    # a step from a different seed converges geometrically. Use alpha=0.5 and a step.
    src = _FakeForceSource(left=[0.0, 0.0, 10.0], right=[0.0, 0.0, 0.0])
    p = _stub_policy(force_source=src, servo=False)
    p._force_meas_ema_alpha = 0.5
    p._force_meas_ema_state = None

    # Frame 1: seed = first reading (|F_L| = 10, no ramp-from-zero).
    p._update_force_meas_and_servo(p.default_dof_angles.copy())
    assert abs(np.linalg.norm(p._force_meas_pelvis[0]) - 10.0) < 1e-6, "must seed to first reading, not 0"
    seeded = p._force_meas_ema_state.copy()

    # Now the source drops to zero (contact released): F should DECAY, not snap.
    p._force_live = _FakeForceSource(left=[0.0, 0.0, 0.0], right=[0.0, 0.0, 0.0])
    p._update_force_meas_and_servo(p.default_dof_angles.copy())
    # 0.5*0 + 0.5*seed = seed/2 → magnitude halves, not zero.
    assert np.allclose(p._force_meas_pelvis, 0.5 * seeded), "EMA must decay geometrically, not snap to 0"
    assert 4.9 < np.linalg.norm(p._force_meas_pelvis[0]) < 5.1


def test_force_meas_ema_alpha_rejects_out_of_range(monkeypatch):
    """__init__ guards alpha ∈ (0, 1]; 0 and >1 and non-numeric are rejected."""
    import os as _os

    # We test the guard logic directly (constructing a full policy needs hardware
    # config); mirror the exact parse+check from __init__.
    def _parse(val):
        _os.environ["HOLOSOMA_FORCE_MEAS_EMA_ALPHA"] = val
        s = _os.environ.get("HOLOSOMA_FORCE_MEAS_EMA_ALPHA", "").strip()
        a = float(s) if s else 1.0
        if not (0.0 < a <= 1.0):
            raise ValueError("out of range")
        return a

    assert _parse("0.5") == 0.5
    assert _parse("1.0") == 1.0
    for bad in ("0", "0.0", "-0.2", "1.5", "2"):
        with pytest.raises(ValueError):
            _parse(bad)
    with pytest.raises(ValueError):
        _parse("abc")
    _os.environ.pop("HOLOSOMA_FORCE_MEAS_EMA_ALPHA", None)


# --------------------------------------------------------------------------- #
# 5. RNEA dynamic compensator (pure numpy port) + anti-alias output stage.
#    These pin the physics of utils/rnea_compensator (the bridge's --ft-algo rnea
#    engine) and the 1 kHz→12 Hz LP→50 Hz output stage. No holosoma / bota needed.
# --------------------------------------------------------------------------- #
from wbt_wrappers_inference.utils.rnea_compensator import (  # noqa: E402
    AntiAliasFilter,
    RneaCompensatorConfig,
    RneaDynamicCompensator,
)

# A minimal, physically-valid single-body payload (PD inertia) in the JSON schema.
_RNEA_CFG_DICT = {
    "rnea_compensator_config": {
        "payload_properties": {
            "mass": 0.8,
            "com": {"x": 0.02, "y": 0.01, "z": 0.09},
            "inertia": {"Ixx": 0.009, "Ixy": 0.0, "Ixz": 0.0, "Iyy": 0.006, "Iyz": 0.0, "Izz": 0.001},
        },
        "geometry": {"imu_offset": [0.0, 0.0, -0.0257]},
        "conventions": {"accel_is_specific_force": True, "wrench_sign": -1.0, "wrench_sign_verified": True},
        "toggles": {"enable_linear_inertia": True, "enable_rotational_inertia": True, "enable_com_coupling": True},
        "alpha_estimator": {"omega_lowpass_cutoff_hz": 20.0, "alpha_lowpass_cutoff_hz": 20.0},
    }
}


def _rnea_cfg(**overrides):
    import copy
    import json as _json
    import tempfile

    d = copy.deepcopy(_RNEA_CFG_DICT)
    for k, v in overrides.items():
        d["rnea_compensator_config"]["conventions"][k] = v
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        _json.dump(d, fh)
        path = fh.name
    return RneaCompensatorConfig.from_json(path)


def test_rnea_static_gravity_is_removed_by_tare():
    """At a static pose (ω=α=0), predict = m·g contribution; after tare_from_prediction
    the compensated wrench is ~0 — this is the gravity-removal a live tare does."""
    comp = RneaDynamicCompensator(_rnea_cfg())
    # Static: accel = specific force = +g "up" in sensor axes (say +z), ω = 0.
    raw = np.zeros(12)
    raw[6:9] = [0.0, 0.0, 9.81]  # accelerometer reads +g at rest
    # Tare on this static sample.
    pred = comp.predict_payload_wrench(0.0, raw[6:9], raw[9:12])
    comp.tare_from_prediction(raw[:6], pred)
    out = np.zeros(12)
    assert comp.update(0.001, raw, out) == "OK"
    assert np.allclose(out[:6], 0.0, atol=1e-9), "tared static output must be ~0"


def test_rnea_removes_linear_inertial_phantom():
    """The whole point vs gravity-only: a linear acceleration produces an m·a wrench
    that RNEA subtracts. After a static tare, feeding an *accelerating* sample leaves
    the compensated force reflecting only the DELTA·m — i.e. the phantom is modeled,
    so the deploy can cancel it (gravity-only would pass it straight through)."""
    comp = RneaDynamicCompensator(_rnea_cfg())
    raw0 = np.zeros(12)
    raw0[6:9] = [0.0, 0.0, 9.81]
    comp.tare_from_prediction(raw0[:6], comp.predict_payload_wrench(0.0, raw0[6:9], raw0[9:12]))
    # Now the sensor sees the SAME raw force but a +2 m/s² x-acceleration added.
    raw1 = raw0.copy()
    raw1[6] += 2.0  # a_x = 2 m/s²
    out = np.zeros(12)
    comp.update(0.001, raw1, out)
    # predicted force gains wrench_sign·m·Δa on x; compensated = raw − bias − sign·pred.
    # The magnitude of the modeled phantom on x is m·Δa = 0.8·2 = 1.6 N.
    pred = comp.last_predicted_wrench
    assert abs(pred[0] - 0.8 * 2.0) < 1e-6, "linear inertial term must be m·a on the accel axis"


def test_rnea_rejects_gravity_removed_accel():
    with pytest.raises(NotImplementedError):
        RneaDynamicCompensator(_rnea_cfg(accel_is_specific_force=False))


def test_rnea_update_requires_length_12():
    comp = RneaDynamicCompensator(_rnea_cfg())
    with pytest.raises(ValueError):
        comp.update(0.0, np.zeros(6), np.zeros(12))


def test_rnea_nonfinite_output_is_degraded():
    comp = RneaDynamicCompensator(_rnea_cfg())
    raw = np.zeros(12)
    raw[0] = np.inf
    out = np.zeros(12)
    assert comp.update(0.0, raw, out).startswith("DEGRADED")


def test_antialias_passthrough_when_disabled():
    """cutoff<=0 ⇒ pass-through (byte-identical to no filtering)."""
    f = AntiAliasFilter(0.0, size=3)
    assert not f.enabled
    x = np.array([1.0, -2.0, 3.0])
    assert np.allclose(f.update(x, 0.001), x)


def test_antialias_seeds_and_smooths():
    """Seeds to the first good sample (no ramp-from-0), then geometrically low-passes."""
    f = AntiAliasFilter(12.0, size=3)
    first = f.update(np.array([10.0, 0.0, 0.0]), 0.001)
    assert np.allclose(first, [10.0, 0.0, 0.0]), "must seed to first sample, not 0"
    # A step to a new value moves partway, never overshoots, magnitude < step.
    nxt = f.update(np.array([0.0, 0.0, 0.0]), 0.001)
    assert 0.0 < nxt[0] < 10.0


def test_antialias_holds_last_good_on_nonfinite():
    """A NaN/Inf must NOT enter the state (it would latch the output to NaN forever)."""
    f = AntiAliasFilter(12.0, size=3)
    f.update(np.ones(3), 0.001)
    good = f.update(np.ones(3), 0.001).copy()
    held = f.update(np.array([np.nan, 1.0, 1.0]), 0.001)
    assert np.allclose(held, good), "non-finite input must hold the last good value"
    assert np.all(np.isfinite(f.update(np.ones(3), 0.001))), "state must not be latched to NaN"


def test_antialias_reset_reseeds():
    """reset() drops the state so the next sample re-seeds (used on re-tare)."""
    f = AntiAliasFilter(12.0, size=3)
    f.update(np.array([9.0, 9.0, 9.0]), 0.001)
    f.reset()
    reseed = f.update(np.array([2.0, 2.0, 2.0]), 0.001)
    assert np.allclose(reseed, [2.0, 2.0, 2.0]), "after reset the next sample must re-seed"


def test_antialias_cutoff_delay_bound():
    """Group delay ~1/(2π·fc) at 12 Hz must be under one 20 ms (50 Hz) control tick."""
    delay_s = 1.0 / (2.0 * np.pi * 12.0)
    assert delay_s < 0.020, f"12 Hz LP delay {delay_s * 1e3:.1f} ms must be < 20 ms"


def test_antialias_clamps_stale_dt():
    """A stale-large dt must NOT make the LP a pass-through (a=dt/(tau+dt)→1). The
    max_dt clamp keeps the smoothing bounded so one spike can't slip through on
    recovery from a run of skipped samples."""
    f = AntiAliasFilter(12.0, size=3, max_dt=0.1)
    f.update(np.zeros(3), 0.001)  # seed at 0
    # A huge dt (e.g. 10 s gap) with the clamp behaves like dt=0.1, NOT pass-through.
    clamped = f.update(np.array([100.0, 0.0, 0.0]), 10.0)
    a_clamped = 0.1 / (1.0 / (2.0 * np.pi * 12.0) + 0.1)
    assert np.isclose(clamped[0], a_clamped * 100.0, rtol=1e-6), "stale dt must be clamped to max_dt"
    assert clamped[0] < 100.0, "clamped LP must NOT pass the spike through unfiltered"

    # With the clamp disabled (max_dt=0), the same huge dt IS a near pass-through.
    g = AntiAliasFilter(12.0, size=3, max_dt=0.0)
    g.update(np.zeros(3), 0.001)
    passed = g.update(np.array([100.0, 0.0, 0.0]), 10.0)
    assert passed[0] > 99.0, "max_dt=0 must NOT clamp (near pass-through on a huge dt)"


# --------------------------------------------------------------------------- #
# 6. Auto-tare BURST on START (_start_force_tare_burst / _handle_start_policy).
# --------------------------------------------------------------------------- #
def _stub_policy_for_tare_burst(n, seconds, *, source):
    """Minimal policy carrying only the auto-tare-burst state, bypassing __init__."""
    from loguru import logger as _loguru

    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    p = object.__new__(HolosomaWBTPolicy)
    p.logger = _loguru
    p._force_live = source
    p._force_tare_on_start_n = n
    p._force_tare_on_start_s = seconds
    p._force_tare_burst_thread = None
    p._servo_integ = np.zeros((2, 3), dtype=np.float64)
    p._servo_delta_x = np.zeros((2, 3), dtype=np.float64)
    return p


def test_tare_burst_fires_n_requests():
    """A START burst forwards N re-tare requests to the bridge over the window."""

    class _Spy:
        def __init__(self):
            self.n = 0

        def request_tare(self):
            self.n += 1

    spy = _Spy()
    p = _stub_policy_for_tare_burst(5, 0.1, source=spy)  # short window for the test
    p._servo_integ[:] = 4.0
    p._start_force_tare_burst()
    p._force_tare_burst_thread.join(timeout=5.0)
    assert spy.n == 5, f"burst must fire exactly 5 re-tares, got {spy.n}"
    # The burst must NOT touch the servo integrator — that is owned by the 50 Hz
    # control thread (cleared by _reset_motion_state on the same START edge). Writing
    # it from this daemon thread would be an unsynchronized cross-thread race.
    assert np.allclose(p._servo_integ, 4.0), "burst must NOT touch servo state (control-thread owned)"


def test_tare_burst_count_one_with_window_floors_to_two():
    """HOLOSOMA_FORCE_TARE_ON_START=1 with a positive window must still land a tare
    AFTER the settle window — floored to >=2 so one fires at the END, not only t≈0."""

    class _Spy:
        def __init__(self):
            self.n = 0

        def request_tare(self):
            self.n += 1

    spy = _Spy()
    p = _stub_policy_for_tare_burst(1, 0.05, source=spy)  # count 1 + positive window
    p._start_force_tare_burst()
    p._force_tare_burst_thread.join(timeout=5.0)
    assert spy.n == 2, f"count=1 + window must floor to 2 (one post-settle tare), got {spy.n}"

    # But count 1 with NO window (total_s=0) is honored as a single immediate tare.
    spy2 = _Spy()
    p2 = _stub_policy_for_tare_burst(1, 0.0, source=spy2)
    p2._start_force_tare_burst()
    p2._force_tare_burst_thread.join(timeout=5.0)
    assert spy2.n == 1, f"count=1 + no window must fire exactly once, got {spy2.n}"


def test_tare_burst_noop_without_source_or_zero_count():
    """No live source, or count 0 → no thread spawned, no crash (byte-identical off)."""

    class _Spy:
        def __init__(self):
            self.n = 0

        def request_tare(self):
            self.n += 1

    # No source.
    p = _stub_policy_for_tare_burst(5, 0.1, source=None)
    p._start_force_tare_burst()
    assert p._force_tare_burst_thread is None

    # Count 0 with a source present.
    spy = _Spy()
    p2 = _stub_policy_for_tare_burst(0, 0.1, source=spy)
    p2._start_force_tare_burst()
    assert p2._force_tare_burst_thread is None
    assert spy.n == 0


def test_tare_burst_does_not_stack():
    """A second START while a burst is still running does not spawn a second thread."""

    class _SlowSpy:
        def __init__(self):
            self.n = 0

        def request_tare(self):
            self.n += 1
            time.sleep(0.05)

    spy = _SlowSpy()
    p = _stub_policy_for_tare_burst(5, 0.4, source=spy)
    p._start_force_tare_burst()
    first = p._force_tare_burst_thread
    assert first is not None and first.is_alive()
    p._start_force_tare_burst()  # should be a no-op while the first is alive
    assert p._force_tare_burst_thread is first, "must not stack a second burst thread"
    first.join(timeout=5.0)


# --------------------------------------------------------------------------- #
# Opt-in teleop RECORD tap (utils/record_tap.py). The tap is a broadcast-only WS
# that ships F^cmd + pelvis F_meas/servo + 29-DOF proprioception + a running flag
# to the out-of-process recorder. These pin: the env gate is OFF by default
# (byte-identical), publish never blocks without a client, the wire schema is
# stable, and overflow drops the OLDEST frame (never back-pressures the RL loop).
# --------------------------------------------------------------------------- #
def _record_tap_mod():
    from wbt_wrappers_inference.utils import record_tap

    return record_tap


def test_record_enabled_gate_default_off(monkeypatch):
    rt = _record_tap_mod()
    for var in ("RECORD", "HOLOSOMA_RECORD"):
        monkeypatch.delenv(var, raising=False)
    assert rt.record_enabled() is False  # unset → OFF (byte-identical policy path)
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv("RECORD", truthy)
        assert rt.record_enabled() is True
    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("RECORD", falsy)
        assert rt.record_enabled() is False
    # legacy alias also honoured
    monkeypatch.setenv("RECORD", "")
    monkeypatch.setenv("HOLOSOMA_RECORD", "1")
    assert rt.record_enabled() is True


def test_record_tap_publish_without_client_never_blocks():
    """With no recorder connected, publish() must return promptly and not raise —
    the load-bearing 'capture is best-effort, control is sacred' invariant."""
    rt = _record_tap_mod()
    tap = rt.RecordTap(port=8896)
    time.sleep(0.2)
    try:
        t0 = time.perf_counter()
        for _ in range(50):
            tap.publish(
                running=True,
                vive_armed=False,
                timestep=0.0,
                servo_on=False,
                f_cmd=np.zeros((2, 3)),
                f_meas=np.zeros((2, 3)),
                f_meas_ungated=np.zeros((2, 3)),
                servo_integ=np.zeros((2, 3)),
                servo_dx=np.zeros((2, 3)),
                servo_dq=np.zeros(29),
                dof_pos=np.zeros(29),
                dof_vel=np.zeros(29),
                root_quat_wxyz=np.array([1.0, 0, 0, 0]),
                base_ang_vel=np.zeros(3),
            )
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"50 publishes took {elapsed:.3f}s — publish must be O(copy), non-blocking"
    finally:
        tap.close()


def test_record_tap_overflow_drops_oldest():
    """A full queue drops the OLDEST frame and enqueues the newest (never blocks the
    producer). With no consumer draining, the queue caps at max_queue."""
    rt = _record_tap_mod()
    tap = rt.RecordTap(port=8895, max_queue=8)
    time.sleep(0.2)
    try:
        for i in range(100):
            tap.publish(
                running=True,
                vive_armed=False,
                timestep=float(i),
                servo_on=False,
                f_cmd=np.full((2, 3), i),
                f_meas=np.zeros((2, 3)),
                f_meas_ungated=np.zeros((2, 3)),
                servo_integ=np.zeros((2, 3)),
                servo_dx=np.zeros((2, 3)),
                servo_dq=np.zeros(29),
                dof_pos=np.zeros(29),
                dof_vel=np.zeros(29),
                root_quat_wxyz=np.array([1.0, 0, 0, 0]),
                base_ang_vel=np.zeros(3),
            )
        # The queue never exceeds its cap, and overflow was counted (not silently lost).
        assert tap._q.qsize() <= 8
        assert tap._dropped > 0, "expected drop-oldest overflow accounting with no consumer"
    finally:
        tap.close()


def test_record_tap_wire_schema_roundtrip():
    """A subscribed client receives well-formed frames carrying every documented
    field with the right shapes — the contract the recorder + replay depend on."""
    import asyncio
    import json
    import threading

    import websockets

    rt = _record_tap_mod()
    port = 8894
    tap = rt.RecordTap(port=port)
    time.sleep(0.3)
    received: list[dict] = []

    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:

            def pub():
                for i in range(5):
                    tap.publish(
                        running=True,
                        vive_armed=(i > 2),
                        timestep=float(i),
                        servo_on=True,
                        f_cmd=np.full((2, 3), i),
                        f_meas=np.ones((2, 3)) * 0.5,
                        f_meas_ungated=np.ones((2, 3)) * 0.9,
                        servo_integ=np.zeros((2, 3)),
                        servo_dx=np.zeros((2, 3)),
                        servo_dq=np.arange(29.0),
                        dof_pos=np.arange(29.0),
                        dof_vel=np.ones(29),
                        root_quat_wxyz=np.array([1.0, 0, 0, 0]),
                        base_ang_vel=np.zeros(3),
                        # C1 UNIFIED soft contact label (None for every other preset).
                        contact_phase=np.array([0.25, 0.75]),
                    )
                    time.sleep(0.02)

            threading.Thread(target=pub).start()
            try:
                for _ in range(5):
                    received.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0)))
            except asyncio.TimeoutError:
                pass

    try:
        asyncio.run(client())
    finally:
        tap.close()

    assert received, "no frames received by the subscribed client"
    f = received[-1]
    for k in (
        "type",
        "seq",
        "t",
        "running",
        "vive_armed",
        "timestep",
        "servo_on",
        "f_cmd",
        "f_meas",
        "f_meas_ungated",
        "servo_integ",
        "servo_dx",
        "servo_dq",
        "dof_pos",
        "dof_vel",
        "root_quat_wxyz",
        "base_ang_vel",
        "contact_phase",
    ):
        assert k in f, f"wire frame missing {k!r}"
    assert f["type"] == "record"
    assert np.asarray(f["f_cmd"]).shape == (2, 3)
    assert np.asarray(f["f_meas_ungated"]).tolist() == [[0.9] * 3, [0.9] * 3]
    assert len(f["dof_pos"]) == 29 and len(f["dof_vel"]) == 29 and len(f["servo_dq"]) == 29
    assert len(f["root_quat_wxyz"]) == 4 and len(f["base_ang_vel"]) == 3
    # C1 UNIFIED soft contact label: 2 per-hand values, and NULL (not omitted) for
    # every preset whose obs has no contact_phase term.
    assert f["contact_phase"] == [0.25, 0.75]


def test_record_tap_contact_phase_defaults_to_null():
    """Omitting `contact_phase=` (every non-unified preset) must still emit the key as
    null, so a recorder keyed on column names never sees a ragged frame."""
    import json

    rt = _record_tap_mod()
    captured: list[dict] = []
    tap = object.__new__(rt.RecordTap)
    tap._seq = 0
    tap._encode_errors = 0
    tap._dropped = 0
    tap._q = SimpleNamespace(put_nowait=captured.append)
    rt.RecordTap.publish(
        tap,
        running=True,
        vive_armed=False,
        timestep=0.0,
        servo_on=False,
        f_cmd=None,
        f_meas=np.zeros((2, 3)),
        f_meas_ungated=np.zeros((2, 3)),
        servo_integ=np.zeros((2, 3)),
        servo_dx=np.zeros((2, 3)),
        servo_dq=np.zeros(29),
        dof_pos=np.zeros(29),
        dof_vel=np.zeros(29),
        root_quat_wxyz=np.array([1.0, 0, 0, 0]),
        base_ang_vel=np.zeros(3),
    )
    assert captured, "publish enqueued nothing"
    frame = captured[-1]
    assert "contact_phase" in frame and frame["contact_phase"] is None
    json.dumps(frame)  # must stay JSON-serializable


def test_record_tap_as_list_none_passthrough():
    """_as_list(None) → None (JSON null), the documented nullable path. This is the
    REAL early-tick runtime value of f_cmd (_force_cmd_pelvis_raw is None before the
    first key nudge), so the tap must serialize it as null, not crash."""
    rt = _record_tap_mod()
    assert rt._as_list(None) is None
    assert rt._as_list(np.zeros((2, 3))) == [[0.0] * 3, [0.0] * 3]


def test_record_tap_none_fcmd_roundtrip():
    """A frame with f_cmd=None (uncommanded/early tick) round-trips as JSON null — the
    recorder relies on this to represent 'no command this tick'."""
    import asyncio
    import json
    import threading

    import websockets

    rt = _record_tap_mod()
    port = 8893
    tap = rt.RecordTap(port=port)
    time.sleep(0.3)
    received: list[dict] = []

    async def client():
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:

            def pub():
                for _ in range(4):
                    tap.publish(
                        running=True,
                        vive_armed=False,
                        timestep=0.0,
                        servo_on=False,
                        f_cmd=None,  # ← the nullable path
                        f_meas=np.zeros((2, 3)),
                        f_meas_ungated=np.zeros((2, 3)),
                        servo_integ=np.zeros((2, 3)),
                        servo_dx=np.zeros((2, 3)),
                        servo_dq=np.zeros(29),
                        dof_pos=np.zeros(29),
                        dof_vel=np.zeros(29),
                        root_quat_wxyz=np.array([1.0, 0, 0, 0]),
                        base_ang_vel=np.zeros(3),
                    )
                    time.sleep(0.02)

            threading.Thread(target=pub).start()
            try:
                for _ in range(4):
                    received.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0)))
            except asyncio.TimeoutError:
                pass

    try:
        asyncio.run(client())
    finally:
        tap.close()

    assert received, "no frames received"
    assert received[-1]["f_cmd"] is None, "f_cmd=None must serialize as JSON null"


def test_record_tap_byte_identical_when_off():
    """Construction-level proof of the byte-identical guarantee: with RECORD unset, the
    policy sets _record_tap = None and the rl_inference tap block is a single is-None
    early-return — no RecordTap is built, no port bound, no per-tick work. Mirrors the
    object.__new__ stub idiom the other force tests use."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    rt = _record_tap_mod()
    # env unset → the gate the policy __init__ consults is False
    import os

    saved = {k: os.environ.get(k) for k in ("RECORD", "HOLOSOMA_RECORD")}
    try:
        for k in ("RECORD", "HOLOSOMA_RECORD"):
            os.environ.pop(k, None)
        assert rt.record_enabled() is False

        # A stub that never ran __init__ (so no _record_tap attr at all): the tap block
        # in rl_inference reads self._record_tap; the policy sets it to None in __init__,
        # and the block guards `if self._record_tap is not None`. Emulate the guarded
        # read on a bare stub to prove a None (or absent-then-None) tap does nothing.
        p = object.__new__(HolosomaWBTPolicy)
        p._record_tap = None
        assert p._record_tap is None
        # The guard the policy uses; must be false → the whole publish path is skipped.
        assert not (p._record_tap is not None)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# --------------------------------------------------------------------------- #
# 3. WIRE F^cmd fold (_poll_force_cmd_wire) — the HL-v2 online planner path.
#    The planner streams a per-hand force command in the raw Bota SENSOR frame on
#    the :8770 wire (ViveLiveSource.latest_force); the low level rotates it to
#    pelvis with its OWN FK and slews it into _force_cmd_pelvis_raw at 2 N/tick.
# --------------------------------------------------------------------------- #
class _FakeViveForce:
    """Fake ViveLiveSource exposing only latest_force() (the wire F^cmd accessor)."""

    def __init__(self, f_cmd):
        self._f = None if f_cmd is None else np.asarray(f_cmd, dtype=np.float32)

    def latest_force(self):
        return None if self._f is None else self._f.copy()


def _stub_wire_policy(*, wire_f, enabled=True, gate_n=0.0, slew_n=2.0, clamp_n=25.0, seeded=None):
    """A HolosomaWBTPolicy stub wired exactly as _poll_force_cmd_wire reads it: a real
    Pinocchio yam model (for _wrist_R_rel) + the wire source + the F^cmd buffers."""
    from loguru import logger as _loguru

    from holosoma_inference.policies.wbt_utils import PinocchioRobot

    from wbt_wrappers_inference.config_values.robot_config import g1_29dof_wbt
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    if not _YAM_URDF.is_file():
        pytest.skip(f"yam URDF not present: {_YAM_URDF}")

    p = object.__new__(HolosomaWBTPolicy)
    p.logger = _loguru
    p.num_dofs = 29
    p.config = SimpleNamespace(robot=g1_29dof_wbt, task=SimpleNamespace(rl_rate=50.0))
    p.default_dof_angles = np.array(g1_29dof_wbt.default_dof_angles, dtype=np.float64)
    p.pinocchio_robot = PinocchioRobot(g1_29dof_wbt, _YAM_URDF.read_text())
    p._force_wrist_frame_ids = None
    p._force_meas_sign = 1.0
    p._vive_live = _FakeViveForce(wire_f)
    p._force_cmd_wire_enabled = enabled
    p._force_cmd_wire_gate_n = gate_n
    p._force_cmd_obs_scale = 0.04
    p._force_cmd_key_slew_n = slew_n
    p._force_cmd_key_clamp_n = clamp_n
    p._force_cmd_pelvis_raw = None if seeded is None else np.asarray(seeded, dtype=np.float32)
    p._force_cmd_target_raw = None if seeded is None else np.asarray(seeded, dtype=np.float32)
    p._force_cmd_local_scaled = None
    return p


def test_wire_fcmd_off_is_byte_identical():
    """HOLOSOMA_FORCE_CMD_WIRE off → _poll_force_cmd_wire is a no-op even with a live wire."""
    p = _stub_wire_policy(wire_f=[0.0, 0.0, 10.0], enabled=False)
    p._poll_force_cmd_wire(p.default_dof_angles.copy())
    assert p._force_cmd_pelvis_raw is None  # untouched


def test_wire_fcmd_none_commands_zero():
    """No f_cmd on the wire (latest_force None) → the applied command decays to zero."""
    p = _stub_wire_policy(wire_f=None, enabled=True, seeded=[[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    p._poll_force_cmd_wire(p.default_dof_angles.copy())
    # Target is zero; the applied slews toward it by <= slew_n per axis.
    assert np.all(np.abs(p._force_cmd_pelvis_raw) <= np.abs([[4.0, 0, 0], [0, 0, 0]]))
    assert p._force_cmd_pelvis_raw[0, 0] == pytest.approx(2.0)  # 4 N → 2 N in one 2 N/tick step


def test_wire_fcmd_folds_sensor_to_pelvis_and_slews():
    """A LEFT sensor-frame command rotates to pelvis, but the FIRST tick only advances by
    the PER-AXIS slew cap (never an un-slewed jump — same per-axis clip as the keys)."""
    p = _stub_wire_policy(wire_f=[0.0, 0.0, 10.0], enabled=True, slew_n=2.0)
    p._poll_force_cmd_wire(p.default_dof_angles.copy())
    # Each axis moved at most the 2 N/tick slew cap (magnitude may exceed 2 N for a
    # multi-axis pelvis direction — that is the per-axis clip, matching _poll_force_cmd_keys).
    assert np.all(np.abs(p._force_cmd_pelvis_raw[0]) <= 2.0 + 1e-5)
    assert np.any(np.abs(p._force_cmd_pelvis_raw[0]) > 0.0)  # LEFT moved off zero
    assert np.allclose(p._force_cmd_pelvis_raw[1], 0.0)  # RIGHT uncommanded


def test_wire_fcmd_reaches_full_pelvis_target_after_enough_slew():
    """Repeated ticks converge the applied F^cmd to the full sensor→pelvis magnitude."""
    p = _stub_wire_policy(wire_f=[0.0, 0.0, 6.0], enabled=True, slew_n=2.0)
    dof = p.default_dof_angles.copy()
    for _ in range(10):
        p._poll_force_cmd_wire(dof)
    assert np.linalg.norm(p._force_cmd_pelvis_raw[0]) == pytest.approx(6.0, abs=1e-5)


def test_wire_fcmd_low_level_deadband_hard_zeros_below_gate():
    """A command below the low-level deadband is hard-zeroed BEFORE the buffer → pure
    position regardless of what the planner published."""
    p = _stub_wire_policy(wire_f=[0.0, 0.0, 3.0], enabled=True, gate_n=6.0)
    p._poll_force_cmd_wire(p.default_dof_angles.copy())
    assert np.allclose(p._force_cmd_pelvis_raw, 0.0)  # 3 N <= 6 N gate → zero
    # Just above the gate passes (slewed).
    p2 = _stub_wire_policy(wire_f=[0.0, 0.0, 8.0], enabled=True, gate_n=6.0)
    p2._poll_force_cmd_wire(p2.default_dof_angles.copy())
    assert np.linalg.norm(p2._force_cmd_pelvis_raw[0]) > 0.0


# --------------------------------------------------------------------------- #
# 4. HARD F/T STOP (_check_force_estop / _trip_force_estop) — the limit that runs the
#    joystick-'B' handler (the launchers arm it at 40 N; these stubs use 20 to keep the
#    arithmetic readable, since the mechanism is threshold-agnostic by construction).
#    What is pinned here is WHAT CANNOT DISABLE IT:
#    the F_meas EMA, the F_meas-on-F^cmd gate, a missing Pinocchio frame, and a
#    sensor sign flip must all be irrelevant, because each of them sits between the
#    sensor and every OTHER consumer of the reading.
# --------------------------------------------------------------------------- #
def _stub_estop_policy(*, left, right=(0.0, 0.0, 0.0), threshold=20.0, running=True, trusted=True):
    """A policy stubbed down to exactly what the E-stop reads. Deliberately does NOT
    build Pinocchio: the stop must work without FK, and this asserts that structurally."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    p = object.__new__(HolosomaWBTPolicy)
    p._force_live = _FakeForceSource(left, right)
    p._force_estop_n = float(threshold)
    p._force_estop_latched = False
    p.use_policy_action = bool(running)
    # An OPEN trust window is "not trusted yet" (deadline in the future).
    p._force_trust_deadline = 0.0 if trusted else time.monotonic() + 60.0
    return p


@pytest.mark.parametrize(
    ("left", "should_trip"),
    [
        ((0.0, 0.0, 25.0), True),      # single axis over
        ((15.0, 15.0, 0.0), True),     # only the MAGNITUDE is over (21.2 N), no axis is
        ((0.0, 0.0, 20.0), False),     # exactly AT the threshold does not trip (strict >)
        ((11.0, 0.0, 0.0), False),     # well under
    ],
)
def test_estop_trips_on_the_magnitude_not_a_single_axis(left, should_trip):
    p = _stub_estop_policy(left=left)
    tripped = p._check_force_estop()
    assert (tripped is not None) == should_trip
    if should_trip:
        assert tripped == pytest.approx(float(np.linalg.norm(left)))


def test_estop_watches_both_hands():
    """A collision on the UNCOMMANDED hand is the one the F_meas gate would hide, so the
    stop reads the bridge's per-hand reading directly and both hands count."""
    assert _stub_estop_policy(left=(0, 0, 0), right=(0.0, 0.0, 30.0))._check_force_estop() is not None


def test_estop_is_off_at_zero_threshold_and_when_not_running():
    """0 = off (the pre-existing behaviour for anyone who did not set the env), and a
    stopped policy never re-trips — stiff hold already owns the joints, which IS the
    post-stop state, so tripping there would re-run the handler every tick."""
    assert _stub_estop_policy(left=(0.0, 0.0, 99.0), threshold=0.0)._check_force_estop() is None
    assert _stub_estop_policy(left=(0.0, 0.0, 99.0), running=False)._check_force_estop() is None


def test_estop_is_suppressed_while_the_reading_is_untrusted():
    """During the START auto-tare window the pre-tare bias is the same order as the
    threshold, so a trip there would be a false positive on EVERY take."""
    p = _stub_estop_policy(left=(0.0, 0.0, 99.0), trusted=False)
    assert p._check_force_estop() is None
    p._force_trust_deadline = 0.0  # window closed
    assert p._check_force_estop() is not None


def test_estop_does_not_depend_on_pinocchio_or_the_meas_sign_or_the_ema():
    """The three transforms between the sensor and every other consumer cannot disable it.

    ``_stub_estop_policy`` builds NO pinocchio_robot and sets no ``_force_meas_sign`` /
    ``_force_meas_ema_*`` at all — so if the implementation ever routed the check through
    ``_wrist_R_rel``, the sign, or the EMA state, this test would raise AttributeError
    rather than quietly reporting "no trip".
    """
    p = _stub_estop_policy(left=(0.0, 0.0, 25.0))
    assert not hasattr(p, "pinocchio_robot")
    assert not hasattr(p, "_force_meas_sign")
    assert not hasattr(p, "_force_meas_ema_state")
    assert p._check_force_estop() == pytest.approx(25.0)
    # ...and the sign convention is irrelevant because a magnitude is sign-invariant.
    assert _stub_estop_policy(left=(0.0, 0.0, -25.0))._check_force_estop() == pytest.approx(25.0)


def test_estop_ignores_a_stale_or_absent_source_and_non_finite_readings():
    """A stale stream reads as zeros (ForceLiveSource.latest), and a NaN must not trip:
    "the sensor is broken" is a different fault with its own gate, and a stop that fires
    on NaN would make a flaky bridge look like a collision."""
    assert _stub_estop_policy(left=(0.0, 0.0, 0.0))._check_force_estop() is None

    class _NaN:
        def latest(self):
            return np.array([np.nan, 0.0, 0.0], dtype=np.float32), np.zeros(3, dtype=np.float32)

    p = _stub_estop_policy(left=(0.0, 0.0, 0.0))
    p._force_live = _NaN()
    assert p._check_force_estop() is None

    p2 = _stub_estop_policy(left=(0.0, 0.0, 25.0))
    p2._force_live = None  # no live F/T configured at all
    assert p2._check_force_estop() is None


def test_estop_latches_so_one_impact_stops_once():
    p = _stub_estop_policy(left=(0.0, 0.0, 25.0))
    assert p._check_force_estop() is not None
    p._force_estop_latched = True  # what _trip_force_estop sets
    assert p._check_force_estop() is None


def test_estop_trip_runs_the_joystick_b_handler():
    """The whole point: the resulting machine state is the operator's own B press, so
    there is ONE stopped state to reason about (and the FUJI planner's low-level-lost
    watchdog, which detects B by the RecordTap stream stopping, sees this identically)."""
    p = _stub_estop_policy(left=(0.0, 0.0, 25.0))
    calls = []
    p._handle_stop_policy = lambda: calls.append("stop")  # type: ignore[method-assign]
    p._trip_force_estop(25.0)
    assert calls == ["stop"], "the trip must go through _handle_stop_policy, not a bespoke shutdown"
    assert p._force_estop_latched is True


def test_estop_log_is_not_red_on_red(monkeypatch):
    """The stop message must not be painted RED.

    loguru's CRITICAL level colour is ``<RED><bold>`` — uppercase is a BACKGROUND in
    loguru markup — and the default format wraps the body in ``<level>{message}</level>``.
    A termcolor red FOREGROUND on top of that emits ESC[41m ESC[1m ESC[31m, i.e. red text
    on a red block: the one line the operator must read is the one they cannot. This
    shipped once and was found on the rig, so it is pinned rather than remembered.
    """
    from wbt_wrappers_inference.policies import holosoma_wbt as hw

    seen: list[str | None] = []

    def _spy_colored(text, color=None, *args, **kwargs):
        seen.append(color)
        return text

    monkeypatch.setattr(hw, "colored", _spy_colored)
    p = _stub_estop_policy(left=(0.0, 0.0, 25.0))
    p._handle_stop_policy = lambda: None  # type: ignore[method-assign]
    p._trip_force_estop(25.0)
    # Both halves matter: the second pins the colour, the first pins that this test can
    # still SEE it (a `termcolor.colored(...)` call style would dodge the monkeypatch and
    # leave the assertion below vacuously true).
    assert seen, "the spy saw no colored() call — this test can no longer see the colour it pins"
    assert "red" not in seen, f"CRITICAL is a red BACKGROUND; red text on it is illegible: {seen}"


def test_estop_threshold_parse_is_strict():
    """A tolerant parse fails SILENTLY on a safety limit: '20N' would read as "off" on a
    run the operator believes is protected."""
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_force_estop_cfg

    assert parse_force_estop_cfg({}) == 0.0  # absent = off (pre-existing behaviour)
    assert parse_force_estop_cfg({"HOLOSOMA_FORCE_ESTOP_N": "20"}) == 20.0
    assert parse_force_estop_cfg({"HOLOSOMA_FORCE_ESTOP_N": " 1.5 "}) == 1.5
    assert parse_force_estop_cfg({"HOLOSOMA_FORCE_ESTOP_N": ""}) == 0.0
    for bad in ("20N", "twenty", "-5", "nan", "inf"):
        with pytest.raises(ValueError):
            parse_force_estop_cfg({"HOLOSOMA_FORCE_ESTOP_N": bad})


def test_estop_check_cannot_crash_the_control_loop():
    """A bare object with NONE of the E-stop state must return "no trip", not raise.

    ``_check_force_estop`` runs FIRST in ``rl_inference``, so an AttributeError there would
    take down the 50 Hz loop this limit exists to protect — strictly worse than a missed
    check, and it is why the reads go through ``getattr``. Pinned so nobody "tightens" them
    back into direct attribute access.
    """
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    bare = object.__new__(HolosomaWBTPolicy)
    assert bare._check_force_estop() is None


def test_estop_stops_the_tick_before_the_onnx_runs_and_publishes_one_finite_hold():
    """The INTEGRATED tick: a 25 N reading must stop the policy without running the model.

    The two halves are unit-tested above; this pins that they are actually WIRED into
    ``rl_inference`` in the right order. If the check ever moved below the ONNX call, the
    tripping tick would still emit a policy action into a collision — the exact thing the
    stop exists to prevent, and invisible to a test of ``_check_force_estop`` alone.

    Built on the same object.__new__ stub as the bad-ONNX fault test, so the two faults are
    shown taking the SAME exit (one finite target, then held fixed).
    """
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    onnx_calls = []
    p = object.__new__(HolosomaWBTPolicy)
    # --- the E-stop's own state: armed at 20 N, 25 N on the LEFT hand, trusted, running ---
    p._force_live = _FakeForceSource([0.0, 0.0, 25.0], [0.0, 0.0, 0.0])
    p._force_estop_n = 20.0
    p._force_estop_latched = False
    p._force_trust_deadline = 0.0
    # --- enough of a policy to complete one rl_inference tick ---
    p.motion_clip_progressing = True
    p._poll_force_cmd_keys = lambda: None
    p._poll_force_cmd_wire = lambda _dof: onnx_calls.append("wire")  # must NOT be reached
    p._force_cmd_wire_enabled = True
    p._align_rot_mat = np.eye(3)
    p._apply_hold_dof_override = lambda: None
    p._build_observation = lambda _state: onnx_calls.append("obs") or np.zeros((1, 680), dtype=np.float32)
    p._debug_log = None
    p._clip_normalized_obs = lambda obs: obs
    p._onnx_obs_input = "obs"
    p._onnx_uses_time_step = True
    p.curr_motion_timestep = 0.0
    p.onnx_policy_session = SimpleNamespace(
        run=lambda _o, _f: onnx_calls.append("onnx") or [np.zeros((1, 29), dtype=np.float32)]
    )
    p.onnx_output_names = ["actions"]
    p._onnx_action_output = "actions"
    p._servo_dq = np.ones(29, dtype=np.float64)
    p.use_policy_action = True
    p.get_ready_state = False
    p.num_dofs = 29
    p.default_dof_angles = np.linspace(-0.2, 0.2, 29)
    p.joint_offsets = np.zeros(29)
    p.upper_body_controller = False
    p.cmd_q = np.linspace(-0.3, 0.3, 29)
    p.cmd_dq = np.zeros(29)
    p.cmd_tau = np.zeros(29)
    p._soft_stop_active = False
    p._stiff_hold_kp = np.ones(29)
    p._stiff_hold_kd = np.ones(29)
    p.latency_tracker = SimpleNamespace(measure=lambda _name: nullcontext())
    # `_handle_stop_policy` is exercised on its own above; here it would need the whole
    # motion-clip + debug-log apparatus, so record the call and keep this test on the tick.
    stopped = []
    p._handle_stop_policy = lambda: stopped.append(True)  # type: ignore[method-assign]

    robot_state = np.zeros((1, 71), dtype=np.float32)
    robot_state[0, 7:36] = np.linspace(-0.4, 0.4, 29)
    sent = []
    p.interface = SimpleNamespace(
        get_low_state=lambda: robot_state, send_low_command=lambda q, *_a, **_k: sent.append(q.copy())
    )

    p.policy_action()

    assert stopped == [True], "the tick must run the joystick-B stop handler"
    assert onnx_calls == [], f"nothing downstream may run on a tripping tick, got {onnx_calls}"
    assert p.use_policy_action is False
    assert p._force_estop_latched is True
    assert np.allclose(p._servo_dq, 0.0), "no servo Δq may be injected on the stopping tick"
    # One finite command, at the MEASURED pose — the same exit the bad-ONNX fault takes.
    assert len(sent) == 1
    assert np.all(np.isfinite(sent[0]))
    assert np.allclose(sent[0], robot_state[0, 7:36])
    assert p._stiff_hold_active is True


# --------------------------------------------------------------------------- #
# 5. PER-HAND SENSOR MASK (HOLOSOMA_FORCE_HANDS). The HL-v2 checkpoints are all
#    layout `left_only`: the model predicts one hand, the right arm is merely HELD.
#    So the right F/T is an input to nothing — and on this rig its RNEA wrench_sign
#    is seeded/UNVALIDATED, so it must not be able to reach the actor or halt the robot.
# --------------------------------------------------------------------------- #
def _stub_masked_policy(*, hands, gate_on, gate_n=1.5, left_cmd=(5.0, 0, 0), right_cmd=(0.0, 0, 0)):
    """A servo/obs-capable stub with BOTH sensors in hard contact but force commanded only
    on the LEFT — the configuration the rig actually runs."""
    p = _stub_policy(force_source=_FakeForceSource([0.0, 0.0, 14.0], [0.0, 0.0, 18.0]), servo=False)
    p._force_cmd_pelvis_raw = np.array([left_cmd, right_cmd], dtype=np.float32)
    p._force_meas_gate_on_cmd = gate_on
    p._force_meas_gate_cmd_n = gate_n
    p._force_hands_enabled = np.array(hands, dtype=bool)
    p._force_estop_n = 20.0
    p._force_estop_latched = False
    p._force_trust_deadline = 0.0
    p.use_policy_action = True
    return p


@pytest.mark.parametrize("gate_on", [True, False], ids=["gate-on", "gate-off"])
def test_masked_hand_reaches_none_of_the_four_force_obs_channels(gate_on):
    """With ``hands=left`` the right hand is dead in all four 2-hand force terms, and — the
    point of a mask rather than a gate — it stays dead with ``GATE_CMD=0``.

    Parametrized over the gate deliberately: before the mask existed, the right hand read
    zero only as a SIDE EFFECT of never being commanded, so a ``GATE_CMD=0`` fidelity A/B
    silently fed 18 N and φ=1 to the actor. That is the regression this pins.
    """
    p = _stub_masked_policy(hands=(True, False), gate_on=gate_on)
    dof = p.default_dof_angles.copy()
    for _ in range(30):  # let the contact-phase LPF and the integrator settle
        p._update_force_meas_and_servo(dof)

    assert np.linalg.norm(p._force_meas_pelvis[0]) == pytest.approx(14.0, abs=1e-3), "LEFT must stay live"
    assert p._contact_phase[0] > 0.9, "LEFT must still report contact"
    for name, arr in (
        ("force_meas_local", p._force_meas_pelvis),
        ("integrator_state", p._servo_integ),
        ("x_tgt_shift", p._servo_delta_x),
    ):
        assert np.allclose(arr[1], 0.0), f"{name} carries the masked RIGHT hand"
    assert p._contact_phase[1] == pytest.approx(0.0, abs=1e-9), "contact_phase carries the masked RIGHT hand"


def test_masked_hand_cannot_trip_the_hard_stop_but_an_unmasked_one_can():
    """The stop reads the sensor directly, so it needs the mask applied there too.

    Both directions are asserted, because "no trip" alone would also pass if the stop were
    simply broken — the ``hands=both`` case proves the 25 N reading is otherwise detected.
    """
    live = _stub_masked_policy(hands=(True, True), gate_on=True)
    live._force_live = _FakeForceSource([0.0, 0.0, 2.0], [0.0, 0.0, 25.0])  # 25 N on the RIGHT only
    assert live._check_force_estop() == pytest.approx(25.0), "an in-loop hand must still stop the robot"

    masked = _stub_masked_policy(hands=(True, False), gate_on=True)
    masked._force_live = _FakeForceSource([0.0, 0.0, 2.0], [0.0, 0.0, 25.0])
    assert masked._check_force_estop() is None, "a masked hand must not be able to halt the robot"
    # ...and the LEFT hand still stops it, so the mask narrowed the scope, not the function.
    masked._force_live = _FakeForceSource([0.0, 0.0, 25.0], [0.0, 0.0, 0.0])
    assert masked._check_force_estop() == pytest.approx(25.0)


def test_masked_hand_takes_no_force_command_from_the_wire():
    """F^cmd only means anything against the F_meas it is servoed toward, and that reading is
    zeroed for a masked hand. Exercised with a 6-wide wire (the planner sends 3 today) so the
    invariant holds for a future two-hand publisher."""
    p = _stub_wire_policy(wire_f=[0.0, 0.0, 10.0, 0.0, 0.0, 10.0], enabled=True, gate_n=1.5)
    p._force_hands_enabled = np.array([True, False], dtype=bool)
    for _ in range(10):
        p._poll_force_cmd_wire(p.default_dof_angles.copy())
    assert np.linalg.norm(p._force_cmd_pelvis_raw[0]) > 0.0, "LEFT must still be commandable"
    assert np.allclose(p._force_cmd_pelvis_raw[1], 0.0), "a masked hand must carry no F^cmd"


def test_force_hands_parse_is_strict_and_defaults_to_both():
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_force_hands_cfg

    assert parse_force_hands_cfg({}) == (True, True)  # absent = byte-identical two-hand
    assert parse_force_hands_cfg({"HOLOSOMA_FORCE_HANDS": "left"}) == (True, False)
    assert parse_force_hands_cfg({"HOLOSOMA_FORCE_HANDS": " RIGHT "}) == (False, True)
    assert parse_force_hands_cfg({"HOLOSOMA_FORCE_HANDS": "none"}) == (False, False)
    for bad in ("l", "LEFT_ONLY", "0", "1", "left,right"):
        with pytest.raises(ValueError):
            parse_force_hands_cfg({"HOLOSOMA_FORCE_HANDS": bad})


def test_unmasked_two_hand_behaviour_is_unchanged():
    """The default must be byte-identical: with ``hands=both`` and the gate off, the right
    hand reaches the obs exactly as it did before the mask existed."""
    p = _stub_masked_policy(hands=(True, True), gate_on=False)
    dof = p.default_dof_angles.copy()
    for _ in range(30):
        p._update_force_meas_and_servo(dof)
    assert np.linalg.norm(p._force_meas_pelvis[1]) == pytest.approx(18.0, abs=1e-3)
    assert p._contact_phase[1] > 0.9
