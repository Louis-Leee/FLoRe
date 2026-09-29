"""The ghost-robot arm IK's ANALYTIC Jacobian must equal the finite-difference one it replaced.

``scripts/hl_serving.py`` swapped the ghost preview's numeric Jacobian for a closed-form geometric
one (7.6x cheaper: the numeric version re-ran ``yourdfpy.update_cfg`` — 0.382 ms — seven extra
times per iteration, so 88% of the solve cost was the derivative). That is only a safe swap if the
two agree: the ghost is the operator's preview of what the arm will do, so a subtly different
Jacobian means a preview that converges somewhere else than the one that was reviewed on the rig.

The IK lives inside ``_run_viser``'s closure and cannot be imported, so this reimplements BOTH
Jacobians against the real URDF and compares them. That is the property under test — the formula,
not the plumbing — and it is exactly what a wrong axis/frame convention would break.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

_URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "UMI_collection", "robot_assets", "g1_yam", "g1_29dof_yam.urdf")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(_URDF), reason="yam URDF not present (UMI_collection submodule not init'd)")


def _so3_log(R: np.ndarray) -> np.ndarray:
    """Copy of hl_serving._so3_log (closure-local there)."""
    R = np.asarray(R, dtype=np.float64)
    cos_t = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta / (2.0 * np.sin(theta)) * axis


@pytest.fixture(scope="module")
def rig():
    yourdfpy = pytest.importorskip("yourdfpy")
    fk = yourdfpy.URDF.load(_URDF, load_meshes=False, build_scene_graph=True)
    names = list(fk.actuated_joint_names)
    arm = [n for n in names
           if n.startswith("left_") and any(k in n for k in ("shoulder", "elbow", "wrist"))]
    idx = np.array([names.index(n) for n in arm], dtype=np.intp)
    jm = fk.joint_map
    return {
        "fk": fk, "names": names, "arm": arm, "idx": idx,
        "axes": [np.asarray(jm[n].axis, dtype=np.float64) for n in arm],
        "childs": [jm[n].child for n in arm],
        "lo": np.array([float(jm[n].limit.lower) for n in arm]),
        "hi": np.array([float(jm[n].limit.upper) for n in arm]),
    }


def _wrist_fk(rig, cfg):
    rig["fk"].update_cfg(cfg)
    T = np.asarray(rig["fk"].get_transform(frame_to="left_wrist_yaw_link", frame_from="pelvis"))
    return T[:3, 3], T[:3, :3]


def _j_analytic(rig, p_wrist):
    """The shipped formula: for revolute joint i, [z x (p_w - p_i); z]. Assumes fk is AT cfg."""
    J = np.empty((6, len(rig["childs"])))
    for i, (child, axis) in enumerate(zip(rig["childs"], rig["axes"])):
        T_i = np.asarray(rig["fk"].get_transform(frame_to=child, frame_from="pelvis"))
        z = T_i[:3, :3] @ axis
        J[:3, i] = np.cross(z, p_wrist - T_i[:3, 3])
        J[3:, i] = z
    return J


def _j_finite_diff(rig, cfg, q, d=1e-5):
    """The REPLACED implementation, verbatim in structure."""
    cfg = cfg.copy()
    idx = rig["idx"]
    cfg[idx] = q
    p, R = _wrist_fk(rig, cfg)
    J = np.zeros((6, idx.size))
    for i in range(idx.size):
        cfg[idx] = q
        cfg[idx[i]] = q[i] + d
        p2, R2 = _wrist_fk(rig, cfg)
        J[:3, i] = (p2 - p) / d
        J[3:, i] = (R @ _so3_log(R.T @ R2)) / d
    return J


def test_all_arm_joints_are_revolute(rig):
    """The analytic columns assume revolute joints; hl_serving refuses to build the IK otherwise,
    so pin the assumption against the URDF that is actually shipped."""
    jm = rig["fk"].joint_map
    assert [str(jm[n].type) for n in rig["arm"]] == ["revolute"] * 7


def test_analytic_jacobian_matches_finite_differences(rig):
    """The swap's correctness condition, over the reachable joint range."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(25):
        q = rng.uniform(np.maximum(rig["lo"], -1.2), np.minimum(rig["hi"], 1.2))
        cfg = np.zeros(len(rig["names"]))
        cfg[rig["idx"]] = q
        p, _ = _wrist_fk(rig, cfg)        # leaves fk AT cfg, as the shipped caller does
        J_a = _j_analytic(rig, p)
        J_f = _j_finite_diff(rig, cfg, q)
        worst = max(worst, float(np.abs(J_a - J_f).max()))
    # The finite-difference reference itself carries O(d) truncation error at d=1e-5, so this
    # bound is the agreement of the two estimators, not the analytic one's own accuracy.
    assert worst < 1e-4, f"analytic vs finite-difference Jacobian disagree by {worst:.2e}"


def test_analytic_jacobian_predicts_real_motion(rig):
    """Independent of the differencing scheme: J @ dq must predict where the wrist actually goes.
    Catches a self-consistent-but-wrong convention (e.g. a transposed rotation) that comparing two
    implementations of the same mistake would miss."""
    rng = np.random.default_rng(1)
    cfg = np.zeros(len(rig["names"]))
    q = rng.uniform(-0.6, 0.6, 7)
    cfg[rig["idx"]] = q
    p0, R0 = _wrist_fk(rig, cfg)
    J = _j_analytic(rig, p0)
    dq = rng.uniform(-1e-4, 1e-4, 7)
    cfg2 = cfg.copy()
    cfg2[rig["idx"]] = q + dq
    p1, R1 = _wrist_fk(rig, cfg2)
    pred = J @ dq
    assert np.allclose(pred[:3], p1 - p0, atol=1e-7), "position block mispredicts real motion"
    assert np.allclose(pred[3:], R0 @ _so3_log(R0.T @ R1), atol=1e-6), \
        "orientation block mispredicts real motion"


def test_dls_step_still_converges_with_the_analytic_jacobian(rig):
    """End-to-end on the SOLVER, not just the derivative: the damped least-squares loop the ghost
    runs must still reach a reachable target. A cheaper Jacobian that needed more iterations would
    hand back the stutter it was meant to remove."""
    idx = rig["idx"]
    cfg = np.zeros(len(rig["names"]))
    q0 = np.zeros(7)
    cfg[idx] = q0
    p_start, R_start = _wrist_fk(rig, cfg)
    # A target 3 cm away with the same orientation — the scale of a real per-chunk ghost step.
    p_tgt = p_start + np.array([0.03, -0.02, 0.01])
    R_tgt = R_start
    q = q0.copy()
    for _ in range(14):
        cfg[idx] = q
        p, R = _wrist_fk(rig, cfg)
        e_p = p_tgt - p
        e_r = R @ _so3_log(R.T @ R_tgt)
        if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
            break
        J = _j_analytic(rig, p)
        e = np.concatenate([e_p, e_r])
        dq = J.T @ np.linalg.solve(J @ J.T + 0.05 * np.eye(6), e)
        q = np.clip(q + dq, rig["lo"], rig["hi"])
    cfg[idx] = q
    p_end, _ = _wrist_fk(rig, cfg)
    err_mm = float(np.linalg.norm(p_tgt - p_end) * 1000.0)
    assert err_mm < 5.0, f"DLS did not converge within the ghost's iteration budget ({err_mm:.1f} mm)"
