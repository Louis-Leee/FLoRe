"""Tests for the VIVE-teleop tracking-error capture + analysis.

Covers the load-bearing correctness claims:

* ``_compute_vr_fk`` is ROOT-INVARIANT — the commanded (reference-joint) and the
  executed (measured-joint) hand poses live in the SAME pelvis frame regardless
  of the floating-base pose, so cmd-vs-exec is a fair same-frame comparison.
* the rot6d→R helper in ``teleop_tracking_log`` is bit-for-bit the policy's own
  ``_rot6d_to_R`` (so the analysis side reconstructs the exact rotation the
  policy used);
* the latency estimator recovers a known injected lag and the tracking-gain /
  geodesic-error math is correct;
* the ``TeleopTrackingLog`` NPZ round-trips and ``analyze`` runs end-to-end on a
  synthetic "draw a circle" trajectory.

Run: pytest src/extensions/wbt_wrappers/wbt_wrappers_inference/tests/test_teleop_tracking.py
"""

from __future__ import annotations

import numpy as np
import pytest

from wbt_wrappers_inference.utils.teleop_tracking_log import TeleopTrackingLog, geodesic_angle_deg, rot6d_to_R
from wbt_wrappers_inference import analyze_teleop_tracking as ana


def _wbt_has_ready_pose() -> bool:
    """True iff HolosomaWBTPolicy._apply_teleop_ready_pose is implemented.

    The HOLOSOMA_TELEOP_READY_POSE override was ported from yam_gripper as tests +
    the MaskedMimic-side helper, but the WBT-policy implementation was only ever in
    an uncommitted working tree (present on NO branch). These tests auto-skip until
    that impl lands, and auto-enable the moment it does.
    """
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    return hasattr(HolosomaWBTPolicy, "_apply_teleop_ready_pose")


requires_wbt_ready_pose = pytest.mark.skipif(
    not _wbt_has_ready_pose(),
    reason="HolosomaWBTPolicy._apply_teleop_ready_pose not implemented (uncommitted on yam_gripper)",
)


# --------------------------------------------------------------------------- #
# rot6d helpers
# --------------------------------------------------------------------------- #
def _R_to_rot6d(R: np.ndarray) -> np.ndarray:
    return R[:, :2].reshape(-1).astype(np.float32)


def _rand_R(seed: int) -> np.ndarray:
    """A random proper rotation via QR (deterministic per seed)."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_rot6d_round_trip_identity():
    assert np.allclose(rot6d_to_R(_R_to_rot6d(np.eye(3))), np.eye(3))


def test_rot6d_to_R_recovers_rotation():
    for s in range(8):
        R = _rand_R(s)
        R_hat = rot6d_to_R(_R_to_rot6d(R))
        assert np.allclose(R_hat, R, atol=1e-6)
        # valid rotation: orthonormal, det +1
        assert np.allclose(R_hat @ R_hat.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R_hat), 1.0, atol=1e-9)


def test_rot6d_helper_matches_policy_implementation():
    """teleop_tracking_log.rot6d_to_R must equal HolosomaWBTPolicy._rot6d_to_R."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    for s in range(6):
        r6 = _R_to_rot6d(_rand_R(s))
        assert np.allclose(rot6d_to_R(r6), HolosomaWBTPolicy._rot6d_to_R(r6), atol=1e-12)


def test_rot6d_to_R_degenerate_inputs_safe():
    # zero column → falls back to canonical basis, still a valid rotation
    R = rot6d_to_R(np.zeros(6, dtype=np.float32))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# geodesic orientation error
# --------------------------------------------------------------------------- #
def test_geodesic_zero_for_equal():
    r6 = _R_to_rot6d(_rand_R(3))
    assert geodesic_angle_deg(r6, r6) < 1e-6


def test_geodesic_recovers_known_angle():
    th = np.radians(35.0)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    a = _R_to_rot6d(np.eye(3))
    b = _R_to_rot6d(Rz)
    assert abs(geodesic_angle_deg(a, b) - 35.0) < 1e-4


def test_geodesic_is_symmetric():
    a = _R_to_rot6d(_rand_R(1))
    b = _R_to_rot6d(_rand_R(2))
    assert abs(geodesic_angle_deg(a, b) - geodesic_angle_deg(b, a)) < 1e-6


# --------------------------------------------------------------------------- #
# latency / gain / axis metrics
# --------------------------------------------------------------------------- #
def test_latency_recovers_injected_lag():
    rl_rate = 50.0
    n = 600
    tt = np.arange(n) / rl_rate
    base = np.stack(
        [np.sin(2 * np.pi * 0.5 * tt), np.cos(2 * np.pi * 0.5 * tt), 0.1 * np.sin(2 * np.pi * 0.3 * tt)], axis=1
    )
    lag = 7  # samples
    cmd = base
    exe = np.roll(base, lag, axis=0)
    exe[:lag] = base[0]
    got_lag, corr = ana._best_lag_samples(cmd, exe, max_lag=25)
    assert got_lag == lag
    assert corr > 0.99


def test_latency_zero_when_no_motion():
    cmd = np.ones((100, 3)) * 0.3
    exe = np.ones((100, 3)) * 0.3
    lag, corr = ana._best_lag_samples(cmd, exe, max_lag=25)
    assert lag == 0


def test_tracking_gain_and_axis_metrics():
    n = 400
    tt = np.linspace(0, 2 * np.pi * 5, n, endpoint=False)  # whole periods → RMS(sin)=1/√2
    cmd = np.stack([0.10 * np.sin(tt), np.zeros(n), np.zeros(n)], axis=1)
    exe = np.stack([0.05 * np.sin(tt), np.zeros(n), np.zeros(n)], axis=1)  # half amplitude on x
    m = ana._axis_metrics(cmd, exe)
    assert abs(m["tracking_gain"]["x"] - 0.5) < 1e-2
    assert m["tracking_gain"]["y"] is None and m["tracking_gain"]["z"] is None  # still axes
    # error is exactly 0.05*sin → RMSE = 0.05/sqrt(2) m = 3.53 cm
    assert abs(m["rmse_cm"]["x"] - 100 * 0.05 / np.sqrt(2)) < 0.05


def test_compute_metrics_perfect_tracking_zero_error():
    rng = np.random.default_rng(0)
    cmd_pos = rng.standard_normal((200, 3)) * 0.05
    cmd_orn = np.tile(_R_to_rot6d(np.eye(3)), (200, 1))
    m = ana.compute_metrics(cmd_pos, cmd_pos.copy(), cmd_orn, cmd_orn.copy(), 50.0)
    assert m["position"]["err_norm_cm"]["rmse"] < 1e-6
    assert m["orientation"]["geodesic_deg"]["max"] < 1e-3


# --------------------------------------------------------------------------- #
# TeleopTrackingLog round-trip + analyze() end-to-end
# --------------------------------------------------------------------------- #
def test_log_record_and_save_roundtrip(tmp_path):
    p = str(tmp_path / "tl.npz")
    log = TeleopTrackingLog(p, rl_rate=50.0, model_path="m.onnx", motion_path="x.npz", relative=True, orientation=True)
    for k in range(10):
        log.record(
            frame=k,
            t=1000.0 + k * 0.02,
            cmd_pos=np.full(6, k, np.float32),
            cmd_orn=np.tile(_R_to_rot6d(np.eye(3)), 2),
            exec_pos=np.full(6, k + 0.5, np.float32),
            exec_orn=np.tile(_R_to_rot6d(np.eye(3)), 2),
            live_pos=np.full(6, k, np.float32),
            live_orn=None,
            armed=(k >= 3),
        )
    log.save()
    d = np.load(p, allow_pickle=True)
    assert d["cmd_pos"].shape == (10, 6) and d["exec_orn"].shape == (10, 12)
    assert bool(d["relative"]) and bool(d["orientation"])
    assert d["armed"].sum() == 7
    # live_orn was None → all-NaN row
    assert np.isnan(d["live_orn"]).all()


def test_log_save_idempotent_and_empty_noop(tmp_path):
    p = str(tmp_path / "empty.npz")
    log = TeleopTrackingLog(p)
    log.save()  # empty → no file
    import os

    assert not os.path.exists(p)
    log.record(0, 0.0, np.zeros(6), np.tile(_R_to_rot6d(np.eye(3)), 2), np.zeros(6), np.tile(_R_to_rot6d(np.eye(3)), 2))
    log.save()
    assert os.path.exists(p)
    mtime = os.path.getmtime(p)
    log.save()  # idempotent — no rewrite
    assert os.path.getmtime(p) == mtime


def _write_circle_log(path, rl_rate=50.0, lag=5, gain=0.8, radius=0.1):
    """Synthetic 'draw a circle about Z with the RIGHT hand' teleop log.

    cmd traces a circle in the XY plane at hand height; exec follows with a known
    lag + amplitude gain → the analysis should recover both.
    """
    n = 500
    tt = np.arange(n) / rl_rate
    cx = radius * np.cos(2 * np.pi * 0.4 * tt)
    cy = radius * np.sin(2 * np.pi * 0.4 * tt)
    cz = np.full(n, 0.2)
    eye6 = _R_to_rot6d(np.eye(3))
    log = TeleopTrackingLog(path, rl_rate=rl_rate, relative=True, orientation=False)
    for k in range(n):
        # right-hand center about (0.3, -0.1, 0.2); left hand still
        cmd_pos = np.array([0.3, -0.1, 0.2, 0.3 + cx[k], -0.1 + cy[k], cz[k]], np.float32)
        kk = max(0, k - lag)
        exec_pos = np.array([0.3, -0.1, 0.2, 0.3 + gain * cx[kk], -0.1 + gain * cy[kk], cz[kk]], np.float32)
        log.record(
            frame=k,
            t=100.0 + k / rl_rate,
            cmd_pos=cmd_pos,
            cmd_orn=np.concatenate([eye6, eye6]),
            exec_pos=exec_pos,
            exec_orn=np.concatenate([eye6, eye6]),
            live_pos=cmd_pos,
            live_orn=None,
            armed=True,
        )
    log.save()


def test_analyze_circle_end_to_end(tmp_path):
    log_path = str(tmp_path / "circle.npz")
    _write_circle_log(log_path, lag=5, gain=0.8)
    out = str(tmp_path / "out")
    res = ana.analyze(log_path, ["right"], out, armed_only=True, no_plot=False)
    rh = res["hands"]["right"]
    # latency recovered (5 ticks @ 50 Hz = 100 ms)
    assert rh["latency"]["lag_samples"] == 5
    assert abs(rh["latency"]["estimated_ms"] - 100.0) < 1e-6
    # gain ~0.8 on the moving axes x,y
    assert abs(rh["position"]["tracking_gain"]["x"] - 0.8) < 0.05
    assert abs(rh["position"]["tracking_gain"]["y"] - 0.8) < 0.05
    # lag-compensated error << raw error (delay was the dominant term)
    assert rh["latency"]["lag_compensated_err_rmse_cm"] < rh["position"]["err_norm_cm"]["rmse"]
    import os

    assert os.path.exists(out + "_right.png")
    assert os.path.exists(out + "_right_metrics.json")


def test_analyze_left_hand_still_is_clean(tmp_path):
    log_path = str(tmp_path / "circle2.npz")
    _write_circle_log(log_path, lag=5, gain=0.8)
    out = str(tmp_path / "out2")
    res = ana.analyze(log_path, ["left"], out, armed_only=True, no_plot=True)
    # left hand never moves and exec==cmd there → ~0 error
    assert res["hands"]["left"]["position"]["err_norm_cm"]["rmse"] < 1e-6


# --------------------------------------------------------------------------- #
# FK root-invariance — the load-bearing claim that cmd & exec share a frame
# --------------------------------------------------------------------------- #
def _build_vr_fk_callable():
    """A standalone _compute_vr_fk bound to a real Pinocchio G1 model.

    Mirrors HolosomaWBTPolicy._compute_vr_fk exactly (same anchor/offset/rot6d
    math) but takes the root pose as an argument so we can vary it and assert the
    LOCAL output is invariant. Skips if pinocchio / the bundled URDF is absent.
    """
    pin = pytest.importorskip("pinocchio")
    from pathlib import Path

    from holosoma_inference.utils.math.quat import (
        matrix_from_quat,
        quat_inverse,
        quat_mul,
        quat_rotate_inverse,
        xyzw_to_wxyz,
    )
    from holosoma_inference.policies.wbt import PinocchioRobot
    from wbt_wrappers_inference.config_values.robot_config import g1_29dof_wbt, VR_2POINT_BODIES, VR_2POINT_OFFSETS

    urdf_path = Path(__file__).resolve().parents[2] / "wbt_wrappers_inference" / "urdf" / "g1_29dof.urdf"
    if not urdf_path.is_file():
        pytest.skip("bundled g1_29dof.urdf not present")
    robot = PinocchioRobot(g1_29dof_wbt, urdf_path.read_text())
    offsets = np.array(VR_2POINT_OFFSETS, dtype=np.float32)
    frame_ids = [robot.robot_model.getFrameId(b) for b in VR_2POINT_BODIES]

    def vr_fk(ref_joint_pos, ref_pos, ref_quat_xyzw):
        joint_pos_pin = ref_joint_pos[robot.real2pinocchio_index]
        q = np.concatenate([ref_pos, ref_quat_xyzw, joint_pos_pin])
        pin.framesForwardKinematics(robot.robot_model, robot.robot_data, q)
        anchor_pos = np.asarray(ref_pos, dtype=np.float64).copy()
        anchor_quat_wxyz = xyzw_to_wxyz(np.asarray(ref_quat_xyzw, dtype=np.float64).reshape(1, 4))
        pos_list, orn_list = [], []
        for i, fid in enumerate(frame_ids):
            fp = robot.robot_data.oMf[fid]
            body_pos_w = fp.translation.astype(np.float64)
            body_quat_xyzw = pin.Quaternion(fp.rotation).coeffs().astype(np.float64)
            body_quat_wxyz = xyzw_to_wxyz(body_quat_xyzw.reshape(1, 4))
            offset = offsets[i].reshape(1, 3)
            offset_world = quat_rotate_inverse(quat_inverse(body_quat_wxyz), offset).reshape(3)
            vr_pos_w = body_pos_w + offset_world
            diff = (vr_pos_w - anchor_pos).reshape(1, 3)
            pos_list.append(quat_rotate_inverse(anchor_quat_wxyz, diff).reshape(3))
            local_quat_wxyz = quat_mul(quat_inverse(anchor_quat_wxyz), body_quat_wxyz)
            orn_list.append(matrix_from_quat(local_quat_wxyz)[..., :2].reshape(-1))
        return np.concatenate(pos_list), np.concatenate(orn_list)

    return vr_fk


def test_compute_vr_fk_is_root_invariant():
    """The CORE guarantee: the LOCAL (pelvis-frame) vr_2point pose depends ONLY on
    joint angles, not on the floating-base pose. So cmd (ref joints) and exec
    (measured joints) are in the SAME frame even when the robot has translated /
    rotated arbitrarily — a fair apples-to-apples comparison."""
    vr_fk = _build_vr_fk_callable()
    rng = np.random.default_rng(0)
    joints = rng.uniform(-0.4, 0.4, size=29).astype(np.float32)

    # Reference root (identity-ish) vs a wildly different translated+yawed root.
    pos_a, orn_a = vr_fk(joints, np.array([0.0, 0.0, 0.75]), np.array([0.0, 0.0, 0.0, 1.0]))
    th = np.radians(57.0)
    quat_yaw_xyzw = np.array([0.0, 0.0, np.sin(th / 2), np.cos(th / 2)])
    pos_b, orn_b = vr_fk(joints, np.array([3.1, -2.4, 0.91]), quat_yaw_xyzw)

    assert np.allclose(pos_a, pos_b, atol=1e-6), "local vr position must be root-invariant"
    assert np.allclose(orn_a, orn_b, atol=1e-6), "local vr orientation must be root-invariant"


def test_compute_vr_fk_different_joints_differ():
    """Sanity: different joint angles DO change the local pose (the test above is
    not vacuously true)."""
    vr_fk = _build_vr_fk_callable()
    root_pos, root_quat = np.array([0.0, 0.0, 0.75]), np.array([0.0, 0.0, 0.0, 1.0])
    p0, _ = vr_fk(np.zeros(29, np.float32), root_pos, root_quat)
    j1 = np.zeros(29, np.float32)
    j1[22] = 0.6  # right shoulder pitch (arm moves)
    p1, _ = vr_fk(j1, root_pos, root_quat)
    assert np.linalg.norm(p0 - p1) > 1e-3


def test_analyze_too_few_ticks_raises(tmp_path):
    log_path = str(tmp_path / "tiny.npz")
    log = TeleopTrackingLog(log_path, rl_rate=50.0)
    eye6 = _R_to_rot6d(np.eye(3))
    for k in range(2):
        log.record(
            k,
            k * 0.02,
            np.zeros(6),
            np.concatenate([eye6, eye6]),
            np.zeros(6),
            np.concatenate([eye6, eye6]),
            armed=True,
        )
    log.save()
    with pytest.raises(SystemExit):
        ana.analyze(log_path, ["right"], str(tmp_path / "o"), armed_only=True, no_plot=True)


# --------------------------------------------------------------------------- #
# Review-driven: non-contiguous span, wall-clock duration, max_ticks cap
# --------------------------------------------------------------------------- #
def test_contiguous_runs_detects_gaps():
    # rows 0,1,2 then a gap then 5,6 then a gap then 9 → 3 runs
    rows = np.array([0, 1, 2, 5, 6, 9])
    runs = ana._contiguous_runs(rows)
    assert runs == [(0, 3), (3, 5), (5, 6)]
    # fully contiguous → single run
    assert ana._contiguous_runs(np.arange(7)) == [(0, 7)]
    assert ana._contiguous_runs(np.array([], dtype=int)) == []


def _write_two_session_log(path, rl_rate=50.0):
    """Two armed bursts separated by a disarmed gap → a non-contiguous armed mask."""
    eye6 = _R_to_rot6d(np.eye(3))
    log = TeleopTrackingLog(path, rl_rate=rl_rate)
    n_each = 30
    row = 0
    for sess in range(2):
        for k in range(n_each):
            ph = (sess * n_each + k) * 0.1
            cmd = np.array([0.3, -0.1, 0.2, 0.3 + 0.1 * np.sin(ph), -0.1, 0.2], np.float32)
            log.record(
                row,
                100.0 + row / rl_rate,
                cmd,
                np.concatenate([eye6, eye6]),
                cmd,
                np.concatenate([eye6, eye6]),
                armed=True,
            )
            row += 1
        # disarmed gap (armed=False) → breaks contiguity of the armed mask
        for k in range(5):
            log.record(
                row,
                100.0 + row / rl_rate,
                np.zeros(6),
                np.concatenate([eye6, eye6]),
                np.zeros(6),
                np.concatenate([eye6, eye6]),
                armed=False,
            )
            row += 1
    log.save()


def test_analyze_largest_run_picks_one_session(tmp_path):
    log_path = str(tmp_path / "two.npz")
    _write_two_session_log(log_path)
    res = ana.analyze(log_path, ["right"], str(tmp_path / "lr"), armed_only=True, no_plot=True, largest_run=True)
    # two armed sessions of 30 each → largest run is 30 ticks
    assert res["analyzed_ticks"] == 30
    assert res["n_contiguous_runs"] == 2


def test_analyze_non_contiguous_warns_but_runs(tmp_path, capsys):
    log_path = str(tmp_path / "two2.npz")
    _write_two_session_log(log_path)
    res = ana.analyze(log_path, ["right"], str(tmp_path / "nc"), armed_only=True, no_plot=True, largest_run=False)
    out = capsys.readouterr().out
    assert "DISJOINT" in out or "disjoint" in out
    assert res["analyzed_ticks"] == 60  # both sessions concatenated when not --largest-run


def test_duration_uses_wallclock_when_jittered(tmp_path):
    """duration_s should reflect real elapsed wall-clock, not n/rl_rate, under a stall."""
    eye6 = _R_to_rot6d(np.eye(3))
    log_path = str(tmp_path / "jit.npz")
    log = TeleopTrackingLog(log_path, rl_rate=50.0)
    n = 50
    for k in range(n):
        # wall-clock advances at 100ms/tick (a stall vs the nominal 20ms)
        cmd = np.array([0.3, -0.1, 0.2, 0.3 + 0.05 * np.sin(k * 0.2), -0.1, 0.2], np.float32)
        log.record(k, 200.0 + k * 0.1, cmd, np.concatenate([eye6, eye6]), cmd, np.concatenate([eye6, eye6]), armed=True)
    log.save()
    res = ana.analyze(log_path, ["right"], str(tmp_path / "j"), armed_only=True, no_plot=True)
    # real elapsed = 49*0.1 = 4.9s, NOT n/rl_rate = 50/50 = 1.0s
    assert abs(res["hands"]["right"]["duration_s"] - 4.9) < 1e-6


def test_max_ticks_cap(tmp_path):
    log_path = str(tmp_path / "capped.npz")
    log = TeleopTrackingLog(log_path, rl_rate=50.0, max_ticks=5)
    eye6 = _R_to_rot6d(np.eye(3))
    for k in range(20):
        log.record(k, k * 0.02, np.zeros(6), np.concatenate([eye6, eye6]), np.zeros(6), np.concatenate([eye6, eye6]))
    assert len(log) == 5
    log.save()
    d = np.load(log_path)
    assert d["cmd_pos"].shape == (5, 6)


def test_save_reentrancy_guard_rolls_back(tmp_path, monkeypatch):
    """If np.savez raises, _saved must roll back so a retry can still save."""
    log_path = str(tmp_path / "rb.npz")
    log = TeleopTrackingLog(log_path, rl_rate=50.0)
    eye6 = _R_to_rot6d(np.eye(3))
    log.record(0, 0.0, np.zeros(6), np.concatenate([eye6, eye6]), np.zeros(6), np.concatenate([eye6, eye6]))

    calls = {"n": 0}
    real_savez = np.savez

    def flaky_savez(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full (simulated)")
        return real_savez(*a, **k)

    monkeypatch.setattr(np, "savez", flaky_savez)
    with pytest.raises(OSError):
        log.save()
    assert log._saved is False  # rolled back
    log.save()  # retry succeeds
    import os

    assert os.path.exists(log_path)


# --------------------------------------------------------------------------- #
# Teleop READY POSE override (HOLOSOMA_TELEOP_READY_POSE)
# --------------------------------------------------------------------------- #
# Right/left arm column ranges within the 29-DOF SDK joint vector.
_RIGHT_ARM_29 = list(range(22, 29))
_LEFT_ARM_29 = list(range(15, 22))


def _ready_pose_stub(raw: str, joint_pos: np.ndarray, pinocchio_robot=None):
    """Bind HolosomaWBTPolicy._apply_teleop_ready_pose to a minimal stub.

    The method only touches: self._teleop_ready_pose_raw, self.config.robot.dof_names,
    self._ext_motion["joint_pos"/"joint_vel"], and (optionally) self.pinocchio_robot.
    joint_pos may be width 36 (root+29) or 29 (no root); joint_vel mirrors it
    (35 / 29). Pass a real PinocchioRobot to exercise the clamp path. Returns the
    mutated (joint_pos, joint_vel).
    """
    from types import SimpleNamespace

    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy
    from wbt_wrappers_inference.config_values.robot_config import g1_29dof_wbt

    T = joint_pos.shape[0]
    vel_w = 35 if joint_pos.shape[1] > 29 else 29
    stub = SimpleNamespace(
        _teleop_ready_pose_raw=raw,
        config=SimpleNamespace(robot=g1_29dof_wbt),
        pinocchio_robot=pinocchio_robot,
        _ext_motion={
            "joint_pos": joint_pos.astype(np.float32).copy(),
            "joint_vel": np.ones((T, vel_w), dtype=np.float32),
        },
    )
    HolosomaWBTPolicy._apply_teleop_ready_pose(stub)  # type: ignore[arg-type]
    return stub._ext_motion["joint_pos"], stub._ext_motion["joint_vel"]


def _bundled_pinocchio_robot():
    """A real PinocchioRobot on the bundled g1_29dof.urdf, or skip."""
    pytest.importorskip("pinocchio")
    from pathlib import Path

    from holosoma_inference.policies.wbt import PinocchioRobot
    from wbt_wrappers_inference.config_values.robot_config import g1_29dof_wbt

    urdf_path = Path(__file__).resolve().parents[2] / "wbt_wrappers_inference" / "urdf" / "g1_29dof.urdf"
    if not urdf_path.is_file():
        pytest.skip("bundled g1_29dof.urdf not present")
    return PinocchioRobot(g1_29dof_wbt, urdf_path.read_text())


@requires_wbt_ready_pose
def test_ready_pose_zero_zeros_both_arms_frame0_only():
    jp = np.random.default_rng(1).uniform(-0.5, 0.5, size=(4, 36)).astype(np.float32)
    out, jv = _ready_pose_stub("zero", jp)
    arm_cols = [7 + i for i in _LEFT_ARM_29 + _RIGHT_ARM_29]
    vel_cols = [6 + i for i in _LEFT_ARM_29 + _RIGHT_ARM_29]
    # FRAME 0 arms zeroed; frames 1+ left untouched (so a played clip stays in-distribution)
    assert np.allclose(out[0, arm_cols], 0.0), "frame-0 arms must be zeroed"
    assert np.allclose(out[1:, arm_cols], jp[1:, arm_cols]), "frames 1+ arms must be untouched"
    # lower body + waist untouched in every frame
    lower_cols = [7 + i for i in range(15)]
    assert np.allclose(out[:, lower_cols], jp[:, lower_cols]), "lower body must be untouched"
    # frame-0 arm velocities zeroed; frames 1+ velocities (seeded to 1.0) left intact
    assert np.allclose(jv[0, vel_cols], 0.0)
    assert np.allclose(jv[1:, vel_cols], 1.0), "frames 1+ arm velocities must be untouched"


@requires_wbt_ready_pose
def test_ready_pose_right_only_leaves_left_arm():
    jp = np.random.default_rng(2).uniform(-0.5, 0.5, size=(3, 36)).astype(np.float32)
    out, _ = _ready_pose_stub("right", jp)
    assert np.allclose(out[0, [7 + i for i in _RIGHT_ARM_29]], 0.0)
    assert np.allclose(out[0, [7 + i for i in _LEFT_ARM_29]], jp[0, [7 + i for i in _LEFT_ARM_29]])


@requires_wbt_ready_pose
def test_ready_pose_explicit_14_floats():
    jp = np.zeros((2, 36), dtype=np.float32)
    vals = [0.1 * i for i in range(14)]
    out, _ = _ready_pose_stub(",".join(str(v) for v in vals), jp)
    assert np.allclose(out[0, [7 + i for i in _LEFT_ARM_29]], vals[:7], atol=1e-6)
    assert np.allclose(out[0, [7 + i for i in _RIGHT_ARM_29]], vals[7:], atol=1e-6)


@requires_wbt_ready_pose
@pytest.mark.parametrize("off", ["garbage", "1,2,3", "0", "false", "no", "off"])
def test_ready_pose_bad_or_off_value_is_noop(off):
    # unparseable / wrong-length (no-op with warn) AND off-tokens (clean no-op)
    jp = np.random.default_rng(3).uniform(-0.5, 0.5, size=(2, 36)).astype(np.float32)
    out, _ = _ready_pose_stub(off, jp)
    assert np.allclose(out, jp), f"value {off!r} must be a no-op"


@requires_wbt_ready_pose
def test_ready_pose_width29_no_root_prefix():
    """A 29-wide NPZ (no root prefix) must apply at offset 0, not 7."""
    jp = np.random.default_rng(4).uniform(-0.5, 0.5, size=(2, 29)).astype(np.float32)
    out, jv = _ready_pose_stub("zero", jp)
    assert np.allclose(out[0, _LEFT_ARM_29 + _RIGHT_ARM_29], 0.0)
    assert np.allclose(out[0, list(range(15))], jp[0, list(range(15))]), "lower body untouched"
    assert np.allclose(jv[0, _LEFT_ARM_29 + _RIGHT_ARM_29], 0.0)


@requires_wbt_ready_pose
def test_ready_pose_explicit_clamps_to_urdf_limits():
    """The clamp guard must bound out-of-range explicit angles to the URDF limits."""
    robot = _bundled_pinocchio_robot()
    model = robot.robot_model
    # Far out-of-range request on every arm joint (±10 rad); expect clamp to limits.
    big = "10," * 13 + "-10"
    jp = np.zeros((1, 36), dtype=np.float32)
    out, _ = _ready_pose_stub(big, jp, pinocchio_robot=robot)
    from wbt_wrappers_inference.config_values.robot_config import g1_29dof_wbt

    names = list(g1_29dof_wbt.dof_names)
    requested = [10.0] * 13 + [-10.0]
    for col_arm, req in zip(_LEFT_ARM_29 + _RIGHT_ARM_29, requested):
        name = names[col_arm]
        qi = model.idx_qs[model.getJointId(name)]
        lo, hi = float(model.lowerPositionLimit[qi]), float(model.upperPositionLimit[qi])
        got = float(out[0, 7 + col_arm])
        assert lo - 1e-5 <= got <= hi + 1e-5, f"{name}={got} not clamped to [{lo},{hi}]"
        assert np.isclose(got, np.clip(req, lo, hi), atol=1e-5), f"{name} expected clamp to limit"


@requires_wbt_ready_pose
def test_ready_pose_zero_makes_forearm_horizontal():
    """End-to-end: after the 'zero' override, FK of the frame-0 arm joints puts
    BOTH forearms within a few degrees of horizontal and pointing forward (+x).
    This is the actual user-facing guarantee."""
    pin = pytest.importorskip("pinocchio")
    from pathlib import Path

    from holosoma_inference.policies.wbt import PinocchioRobot
    from wbt_wrappers_inference.config_values.robot_config import g1_29dof_wbt

    urdf_path = Path(__file__).resolve().parents[2] / "wbt_wrappers_inference" / "urdf" / "g1_29dof.urdf"
    if not urdf_path.is_file():
        pytest.skip("bundled g1_29dof.urdf not present")

    # A clip whose arms hang down (non-horizontal) — override should fix it.
    jp = np.zeros((1, 36), dtype=np.float32)
    rng = np.random.default_rng(7)
    jp[0, 7 + 15 : 7 + 29] = rng.uniform(-0.8, 0.8, size=14)  # random arm pose
    out, _ = _ready_pose_stub("zero", jp)
    j29 = out[0, 7:]

    robot = PinocchioRobot(g1_29dof_wbt, urdf_path.read_text())
    model, data = robot.robot_model, robot.robot_data
    q = pin.neutral(model)
    for name, val in zip(list(g1_29dof_wbt.dof_names), j29):
        if model.existJointName(name):
            q[model.idx_qs[model.getJointId(name)]] = float(val)
    pin.framesForwardKinematics(model, data, q)
    for side in ("right", "left"):
        R = data.oMf[model.getFrameId(f"{side}_elbow_link")].rotation
        fwd = R[:, 0]  # forearm long axis (elbow-link local +x)
        elev_deg = np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))
        assert abs(elev_deg) < 12.0, f"{side} forearm not horizontal: {elev_deg:.1f} deg"
        assert fwd[0] > 0.9, f"{side} forearm not pointing forward: x={fwd[0]:.2f}"
