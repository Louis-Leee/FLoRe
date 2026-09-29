"""Headless unit tests for force_target_math (spec §6).

Run in the hssim env:
    ~/.holosoma_deps/miniconda3/envs/hssim/bin/python -m pytest \
        src/extensions/wbt_wrappers/wbt_wrappers_inference/tests/test_force_target_math.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from wbt_wrappers_inference import force_target_math as ftm

# Real clips present on the box (spec §5); IK tests skip if data absent.
_NPZ = "/data/dataset/sonic_manipulation_holosoma/avoid_bump_push_R_003__A460_M.npz"
_NPZ2 = "/data/dataset/sonic_manipulation_holosoma/Neutral_throw_ball_001__A057_M.npz"
_URDF = (
    "thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof_yam.urdf"
)


# --- quaternion helpers ----------------------------------------------------
def test_quat_identity_is_identity():
    R = ftm.quat_to_mat_wxyz(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(R, np.eye(3))


def test_quat_apply_z_90deg():
    # 90° about +z (wxyz): x-axis → y-axis.
    q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    out = ftm.quat_apply_wxyz(q, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(out, [0.0, 1.0, 0.0], atol=1e-7)


def test_adjoint_moment_and_force_channels():
    # The adjoint is IMPORTED from the validated playground (not re-derived). A pure
    # lateral force measured at the SENSOR, with zero sensor moment, maps to the TCP
    # as: force rotated by R_sensor (sensor +y → wrist -y), plus a lever-INDUCED
    # moment m_T = -lever × f_T (this is the 6-D coupling — the adjoint INTRODUCES
    # the 0.16 m·f moment (2026-07-10, was 0.133) when reducing a sensor force to the TCP; it does NOT
    # cancel it — cancellation happens in the OTHER direction, when a force applied
    # AT the TCP shows up as a phantom sensor moment). Assert BOTH channels so a
    # dropped lever term or a sign flip fails the test.
    w_sensor = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    w_tcp = ftm.adjoint_sensor_to_tcp(w_sensor)
    # R_sensor maps sensor +y → wrist -y: f_T = (0, -10, 0).
    assert np.allclose(w_tcp[:3], [0.0, -10.0, 0.0], atol=1e-6)
    # m_T = -lever × f_T, lever=(0.16,0,0): -(0.16,0,0)×(0,-10,0) = (0,0,1.6). (2026-07-10, was 0.133/1.33)
    assert np.allclose(w_tcp[3:6], [0.0, 0.0, ftm.FT_TCP_LEVER_X * 10.0], atol=1e-6)
    # Force magnitude preserved (rotation only).
    assert np.isclose(np.linalg.norm(w_tcp[:3]), 10.0, atol=1e-6)


# --- pulse -----------------------------------------------------------------
def test_pulse_phases():
    spec = ftm.PulseSpec(ramp=50, hold=200, idle=50)
    assert ftm.pulse_scale(0, spec) == 0.0
    assert ftm.pulse_scale(50, spec) == pytest.approx(1.0)  # end of ramp-up
    assert ftm.pulse_scale(150, spec) == pytest.approx(1.0)  # mid hold
    assert ftm.pulse_scale(250, spec) == pytest.approx(1.0)  # end of hold
    assert ftm.pulse_scale(300, spec) == pytest.approx(0.0, abs=1e-9)  # end of ramp-down
    assert ftm.pulse_scale(320, spec) == 0.0  # idle
    assert ftm.pulse_phase(25, spec) == "ramp-up"
    assert ftm.pulse_phase(150, spec) == "hold"
    assert ftm.pulse_phase(280, spec) == "ramp-down"
    assert ftm.pulse_phase(320, spec) == "idle"


def test_pulse_ramp_is_linear_and_bounded():
    spec = ftm.PulseSpec(ramp=50, hold=200, idle=50)
    assert ftm.pulse_scale(25, spec) == pytest.approx(0.5)
    for t in range(spec.period * 2):
        s = ftm.pulse_scale(t, spec)
        assert 0.0 <= s <= 1.0


# --- force profiles + Eq.2 shift ------------------------------------------
def test_20N_gives_10cm_shift():
    # 20 N pure command, no external → shift = 20/200 = 0.10 m.
    spec = ftm.PulseSpec()
    f_cmd = ftm.f_cmd_pelvis(150, np.array([1.0, 0.0, 0.0]), 20.0, spec)  # mid-hold → full
    assert np.isclose(np.linalg.norm(f_cmd), 20.0)
    shift = ftm.target_shift_world(f_cmd, np.zeros(3))
    assert np.isclose(np.linalg.norm(shift), 0.10, atol=1e-9)


def test_scenario_A_zero_ext():
    spec = ftm.PulseSpec()
    f_cmd = ftm.f_cmd_pelvis(150, np.array([1.0, 0, 0]), 20.0, spec)
    assert np.allclose(ftm.f_ext_pelvis(150, f_cmd, spec, "A"), 0.0)


def test_scenario_B_relaxes_to_negative_fcmd():
    spec = ftm.PulseSpec(ramp=50, hold=200, idle=50)
    f_cmd = ftm.f_cmd_pelvis(250, np.array([1.0, 0, 0]), 20.0, spec)  # end of hold → full
    f_ext = ftm.f_ext_pelvis(250, f_cmd, spec, "B")  # gate fully engaged
    assert np.allclose(f_ext, -f_cmd, atol=1e-6)
    # Equilibrium: F_cmd + F_ext → 0 ⇒ target returns to contact (shift ≈ 0).
    assert np.allclose(ftm.target_shift_world(f_cmd, f_ext), 0.0, atol=1e-6)


def test_scenario_B_engages_across_hold():
    spec = ftm.PulseSpec(ramp=50, hold=200, idle=50)
    f_cmd = np.array([20.0, 0.0, 0.0])
    # Start of hold (t=51): gate ~0 → F_ext ~0. Mid hold: partial. End: full.
    assert np.linalg.norm(ftm.f_ext_pelvis(51, f_cmd, spec, "B")) < 1.0
    assert np.allclose(ftm.f_ext_pelvis(250, f_cmd, spec, "B"), -f_cmd, atol=1e-6)


def test_bad_scenario_raises():
    with pytest.raises(ValueError):
        ftm.f_ext_pelvis(0, np.zeros(3), ftm.PulseSpec(), "Z")


# --- sensor vs TCP anchor (the core question) ------------------------------
def test_sensor_vs_tcp_shift_equal_for_pure_force():
    # For a pure 3-D force the shift VECTOR is anchor-independent (rotation-only);
    # the two target paths differ ONLY by the fixed base-point offset.
    q_wrist = np.array([1.0, 0.0, 0.0, 0.0])  # level wrist
    p_wrist = np.array([0.3, -0.1, 1.0])
    p_tcp = ftm.contact_point_world(p_wrist, q_wrist, ftm.TCP_OFFSET_X)
    p_ft = ftm.contact_point_world(p_wrist, q_wrist, ftm.FT_OFFSET_X)
    offset = p_tcp - p_ft
    # Level wrist +x → world +x: the base-point offset is exactly the lever (0.16 m, 2026-07-10).
    assert np.isclose(np.linalg.norm(offset), ftm.FT_TCP_LEVER_X, atol=1e-9)
    assert np.isclose(offset[0], ftm.FT_TCP_LEVER_X, atol=1e-9)

    f_cmd_w = np.array([20.0, 0.0, 0.0])
    tgt_tcp = ftm.eq2_target_world(p_tcp, f_cmd_w, np.zeros(3))
    tgt_ft = ftm.eq2_target_world(p_ft, f_cmd_w, np.zeros(3))
    # Same shift applied → the two targets differ by exactly the base-point offset.
    assert np.allclose((tgt_tcp - tgt_ft), offset, atol=1e-9)


def test_offset_is_pure_x_translation_under_wrist_rotation():
    # Under a wrist yaw the anchor offset stays magnitude 0.16 (2026-07-10) but rotates in world.
    q_wrist = np.array([np.cos(np.pi / 6), 0.0, 0.0, np.sin(np.pi / 6)])  # 60° yaw
    p_wrist = np.zeros(3)
    p_tcp = ftm.contact_point_world(p_wrist, q_wrist, ftm.TCP_OFFSET_X)
    p_ft = ftm.contact_point_world(p_wrist, q_wrist, ftm.FT_OFFSET_X)
    assert np.isclose(np.linalg.norm(p_tcp - p_ft), ftm.FT_TCP_LEVER_X, atol=1e-9)


# --- F_cmd/F_ext pelvis->world rotation IS applied in frame_targets (spec §6) --
def _synthetic_motion(n=300, pelvis_yaw=0.9, wrist_yaw=0.4):
    """A tiny in-memory Motion with a non-identity pelvis + wrist yaw, so the
    frame-rotation tests run WITHOUT the dataset (no false-green on a data-less box)."""
    T = n
    jp = np.zeros((T, 36))
    qp = np.array([np.cos(pelvis_yaw / 2), 0.0, 0.0, np.sin(pelvis_yaw / 2)])  # pelvis yaw
    qw = np.array([np.cos(wrist_yaw / 2), 0.0, 0.0, np.sin(wrist_yaw / 2)])  # wrist yaw
    jp[:, 3:7] = qp
    body_names = ["pelvis"] + [f"b{i}" for i in range(1, 32)]
    body_names[24] = "left_wrist_yaw_link"
    body_names[31] = "right_wrist_yaw_link"
    bp = np.zeros((T, 32, 3))
    bq = np.zeros((T, 32, 4))
    bq[:, :, 0] = 1.0  # identity everywhere by default
    bq[:, 0, :] = qp  # pelvis matches base quat (npz redundancy contract)
    bp[:, 31, :] = [0.3, -0.2, 1.0]
    bq[:, 31, :] = qw
    bp[:, 24, :] = [0.3, 0.2, 1.0]
    bq[:, 24, :] = qw
    return ftm.Motion(
        fps=50, joint_pos=jp, body_pos_w=bp, body_quat_w=bq,
        body_names=body_names, joint_names=[f"j{i}" for i in range(29)],
    )


def test_fcmd_pelvis_to_world_rotation_applied():
    # If frame_targets forgot the pelvis->world rotation (used the pelvis-frame
    # force as world), this would fail. Pelvis yawed 0.9 rad, F_cmd along pelvis +x.
    m = _synthetic_motion(pelvis_yaw=0.9)
    spec = ftm.PulseSpec()
    ft = ftm.frame_targets(m, 150, "right", np.array([1.0, 0, 0]), 12.0, spec, "A")
    R = ftm.quat_to_mat_wxyz(m.pelvis_quat(150))
    expect_world = R @ np.array([12.0, 0.0, 0.0])  # full at mid-hold
    assert np.allclose(ft.f_cmd_world, expect_world, atol=1e-9)
    # And it is NOT the raw pelvis-frame vector (rotation really happened).
    assert not np.allclose(ft.f_cmd_world, [12.0, 0.0, 0.0], atol=1e-3)


def test_fext_world_rotation_and_scenarioB_in_frame_targets():
    # Scenario B through the full frame_targets path: F_ext_world = R_pelvis @ (-gate*F_cmd_p),
    # and at end-of-hold the shift returns to ~0 (equilibrium), so x_target ≈ contact.
    m = _synthetic_motion(pelvis_yaw=0.9)
    spec = ftm.PulseSpec(ramp=50, hold=200, idle=50)
    ft = ftm.frame_targets(m, 250, "right", np.array([1.0, 0, 0]), 12.0, spec, "B")
    assert np.allclose(ft.f_cmd_world + ft.f_ext_world, 0.0, atol=1e-6)
    assert np.allclose(ft.shift, 0.0, atol=1e-6)
    assert np.allclose(ft.x_target_tcp, ft.p_tcp, atol=1e-6)


def test_frame_out_of_range_raises():
    m = _synthetic_motion(n=8)
    with pytest.raises(IndexError):
        m.wrist_pose(8, "right")
    with pytest.raises(IndexError):
        m.dof(-1)
    with pytest.raises(ValueError):
        m.wrist_pose(0, "middle")  # type: ignore[arg-type]


# --- data-backed frame_targets + IK (skip if data/urdf absent) -------------
def _have(path: str) -> bool:
    import os

    return os.path.exists(path)


@pytest.mark.skipif(not _have(_NPZ), reason="dataset not present")
def test_load_motion_and_frame_targets():
    m = ftm.load_motion(_NPZ)
    assert m.n_frames > 10 and m.fps == 50
    # Base pose equals pelvis body pose (verified npz redundancy).
    assert np.allclose(m.base_quat(0), m.pelvis_quat(0), atol=1e-6)
    ft = ftm.frame_targets(m, 5, "right", np.array([1.0, 0, 0]), 20.0, ftm.PulseSpec(), "A")
    # TCP is 0.16 m further along wrist +x than the FT anchor (2026-07-10, was 0.133).
    assert np.isclose(np.linalg.norm(ft.p_tcp - ft.p_ft), ftm.FT_TCP_LEVER_X, atol=1e-6)


def _solved_tcp_world(ik, m, t, side, res):
    """Rebuild the solved config, FK, and return the ghost's TCP in world coords."""
    import pinocchio as pin

    q = ik._q0_from_reference(m, m.dof(t))
    for jn, val in res.q_arm.items():
        q[ik._q_idx[jn]] = val
    pin.forwardKinematics(ik.model, ik.data, q)
    pin.updateFramePlacements(ik.model, ik.data)
    tcp_pelvis = ik.data.oMf[ik._tcp_frame_id[side]].translation
    R_pelvis = ftm.quat_to_mat_wxyz(m.pelvis_quat(t))
    return R_pelvis @ tcp_pelvis + m.base_pos(t)


@pytest.mark.skipif(not (_have(_NPZ) and _have(_URDF)), reason="dataset/urdf not present")
def test_ik_zero_shift_returns_reference():
    m = ftm.load_motion(_NPZ)
    ik = ftm.ArmIK(_URDF)
    t, side = 20, "right"
    # Zero shift → TCP target = reference contact → IK should reproduce the reference
    # arm angles (the seed already satisfies both position and orientation, so the
    # hierarchical solver's first step is ~0). Residual ~0, angles ≈ reference.
    ft = ftm.frame_targets(m, t, side, np.array([1.0, 0, 0]), 0.0, ftm.PulseSpec(), "A")
    res = ik.solve(m, t, side, ft.x_target_tcp)
    assert res.residual_m < 1e-3, f"zero-shift IK residual {res.residual_m}"
    name_to_val = dict(zip(m.joint_names, m.dof(t)))
    for jn, val in res.q_arm.items():
        assert np.isclose(val, name_to_val[jn], atol=5e-3), f"{jn}: {val} vs {name_to_val[jn]}"


@pytest.mark.skipif(not (_have(_NPZ) and _have(_URDF)), reason="dataset/urdf not present")
@pytest.mark.parametrize("side", ["right", "left"])
def test_ik_small_shift_tcp_reaches_world_target(side):
    # A 12 N (6 cm) forward command at mid-hold — a modest reachable shift — for BOTH
    # hands. The hierarchical IK drives the TCP FRAME directly, so the ghost's TCP must
    # land on the world target regardless of the (soft) orientation drift.
    m = ftm.load_motion(_NPZ)
    ik = ftm.ArmIK(_URDF)
    t = 20
    ft = ftm.frame_targets(m, t, side, np.array([1.0, 0, 0]), 12.0, ftm.PulseSpec(), "A")
    res = ik.solve(m, t, side, ft.x_target_tcp)
    assert res.converged, f"[{side}] IK did not converge: residual {res.residual_m} m"
    assert res.residual_m < 2e-2
    tcp_world = _solved_tcp_world(ik, m, t, side, res)
    assert np.allclose(tcp_world, ft.x_target_tcp, atol=2e-3), (
        f"[{side}] solved TCP {tcp_world} != world target {ft.x_target_tcp}"
    )


@pytest.mark.skipif(not (_have(_NPZ) and _have(_URDF)), reason="dataset/urdf not present")
def test_ik_soft_orientation_avoids_wrist_counter_curl():
    # The bug the hierarchical IK fixes: a +z (up) command used to make the hard
    # 6-DOF orientation lock counter-rotate wrist_pitch the "wrong" way (Δ≈+0.56 rad).
    # With soft orientation the wrist relaxes toward its natural sign (small/negative),
    # while the TCP still reaches the target and orientation follows within a few deg.
    m = ftm.load_motion(_NPZ)
    ik = ftm.ArmIK(_URDF)
    t, side = 120, "right"
    ft = ftm.frame_targets(m, t, side, np.array([0.0, 0.0, 1.0]), 12.0, ftm.PulseSpec(), "A")
    res = ik.solve(m, t, side, ft.x_target_tcp)
    assert res.converged and res.residual_m < 5e-3
    ref = dict(zip(m.joint_names, m.dof(t)))
    d_wpitch = res.q_arm[f"{side}_wrist_pitch_joint"] - ref[f"{side}_wrist_pitch_joint"]
    # No large positive counter-curl (the old lock gave +0.56 rad; soft-follow keeps
    # |Δ| well under that and not the "wrong" sign of the old artifact).
    assert d_wpitch < 0.3, f"wrist_pitch still counter-curling: Δ={d_wpitch:+.3f} rad"
    # Orientation still softly followed (not wildly off) — a few tens of degrees at most.
    assert res.ori_err_rad < np.radians(45), f"orientation drifted too far: {np.degrees(res.ori_err_rad):.1f} deg"


@pytest.mark.skipif(not (_have(_NPZ2) and _have(_URDF)), reason="second dataset/urdf not present")
def test_ik_on_second_clip_throw():
    # The fast throw clip (spec §5) stresses IK differently; every converged frame's
    # TCP must reach the world target and unconverged frames must flag (not crash).
    m = ftm.load_motion(_NPZ2)
    ik = ftm.ArmIK(_URDF)
    spec = ftm.PulseSpec()
    n_conv = 0
    for t in range(0, m.n_frames, 5):
        ft = ftm.frame_targets(m, t, "right", np.array([1.0, 0, 0]), 12.0, spec, "A")
        res = ik.solve(m, t, "right", ft.x_target_tcp)
        assert np.isfinite(res.residual_m)
        if res.converged:
            n_conv += 1
            # The ghost TCP reaches the world target to within the convergence
            # threshold (res.residual_m, reported in the pelvis frame == world dist).
            tcp_world = _solved_tcp_world(ik, m, t, "right", res)
            assert np.linalg.norm(tcp_world - ft.x_target_tcp) <= res.residual_m + 1e-4
    assert n_conv > 0, "no frames converged on the throw clip"
