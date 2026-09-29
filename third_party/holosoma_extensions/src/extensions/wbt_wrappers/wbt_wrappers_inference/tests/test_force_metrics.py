"""Regression tests for the UniFP force-tracking eval metrics.

Covers the g1-wbt-2pt-force benchmark's Tier-1 (free-space, displacement-implied)
force-realization metrics added to `eval_utils.MetricsLogger`, plus the Eval-3
`MuJoCoInterface` external-force injection.

The metric identities under test (all in the ROOT/PELVIS frame, which is the
frame the training reward and the F_cmd obs both use):

    F_hat   = K * (hand_pol - hand_ref)            realized force (Hookean)
    e_F     = || F_cmd - F_hat ||  = K * r_pos     force-realization error
    r_pos   = || hand_pol - (hand_ref + F_cmd/K) ||  Eq.2 position residual

Run: pytest src/extensions/wbt_wrappers/wbt_wrappers_inference/tests/test_force_metrics.py
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
import pytest

from wbt_wrappers_inference.utils import eval_utils as eu
from wbt_wrappers_inference.utils.eval_utils import resolve_xml, _load_model, MetricsLogger
from wbt_wrappers_inference.utils.headless_sim_factory import MuJoCoInterface

ISA = eu.ISAAC_TO_MUJOCO
K = eu.FORCE_STIFFNESS_K

_TEST_NPZ = (
    "/data/dataset/sonic_manipulation_holosoma_test/"
    "big_heavy_one_hand_behind_high_to_behind_low_R_001__A524.npz"
)


@pytest.fixture(scope="module")
def mj_model():
    return _load_model(resolve_xml(None), terrain_obj=None)


@pytest.fixture(scope="module")
def ref_npz():
    if not os.path.exists(_TEST_NPZ):
        pytest.skip(f"test motion not present: {_TEST_NPZ}")
    return np.load(_TEST_NPZ, allow_pickle=True)


def _set_qpos_to_ref(mj_model, mj_data, ref, t):
    """Place the robot exactly on the reference frame t (perfect tracking)."""
    mj_data.qpos[0:3] = ref["body_pos_w"][t, 0]
    mj_data.qpos[3:7] = ref["body_quat_w"][t, 0]
    if ref["joint_pos"].shape[1] == 36:
        mj_data.qpos[7:36] = ref["joint_pos"][t, 7:]
    else:
        mj_data.qpos[7:36] = ref["joint_pos"][t][ISA]
    mujoco.mj_forward(mj_model, mj_data)


def test_quat_to_mat_is_rotation():
    q = np.array([np.cos(0.37), 0.0, np.sin(0.37), 0.0])  # pitch
    R = eu._quat_to_mat_wxyz(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_parse_force_cmd_env(monkeypatch):
    monkeypatch.delenv("HOLOSOMA_FORCE_CMD_LOCAL", raising=False)
    assert np.allclose(MetricsLogger._parse_force_cmd_env(), np.zeros((2, 3)))
    monkeypatch.setenv("HOLOSOMA_FORCE_CMD_LOCAL", "0,0,0,5,0,0")
    fc = MetricsLogger._parse_force_cmd_env()
    assert fc.shape == (2, 3)
    assert np.allclose(fc[1], [5, 0, 0]) and np.allclose(fc[0], 0)
    monkeypatch.setenv("HOLOSOMA_FORCE_CMD_LOCAL", "1,2,3")  # wrong length
    with pytest.raises(ValueError):
        MetricsLogger._parse_force_cmd_env()


def test_f0_perfect_replay_zero_force(mj_model, ref_npz, monkeypatch):
    """F_cmd=0 + exact reference replay → drift≈0, e_F≈0 (F=0 invariant floor)."""
    monkeypatch.delenv("HOLOSOMA_FORCE_CMD_LOCAL", raising=False)
    ml = MetricsLogger(ref_npz, mj_model)
    assert ml._force_enabled
    data = mujoco.MjData(mj_model)
    T = ref_npz["joint_pos"].shape[0]
    for t in range(T):
        _set_qpos_to_ref(mj_model, data, ref_npz, t)
        ml.log_frame(t, data)
    fs = ml.finalize(T)["force_summary"]
    # ~1e-6 cm = FK float64 round-trip noise; require << 1 micron-scale, i.e. 1e-3 cm.
    for tag in ("left", "right"):
        assert fs[f"force_drift_median_cm_{tag}"] < 1e-3
        assert fs[f"force_realization_err_median_N_{tag}"] < 1e-3


def test_ignored_command_reports_full_error(mj_model, ref_npz, monkeypatch):
    """F_cmd=5N R+x but hand replays the zero-force ref (ignores cmd) →
    target_err == intended displacement (2.5cm) and e_F == |F_cmd| (5N)."""
    monkeypatch.setenv("HOLOSOMA_FORCE_CMD_LOCAL", "0,0,0,5,0,0")
    ml = MetricsLogger(ref_npz, mj_model)
    data = mujoco.MjData(mj_model)
    T = ref_npz["joint_pos"].shape[0]
    for t in range(T):
        _set_qpos_to_ref(mj_model, data, ref_npz, t)
        ml.log_frame(t, data)
    fs = ml.finalize(T)["force_summary"]
    assert fs["force_drift_median_cm_right"] < 1e-3
    assert abs(fs["force_target_err_median_cm_right"] - 2.5) < 1e-3
    assert abs(fs["force_realization_err_median_N_right"] - 5.0) < 1e-3


def test_perfect_realization_zero_error(mj_model, ref_npz, monkeypatch):
    """F_cmd=5N R+x and the right hand displaced by exactly F/K=2.5cm in the
    pelvis +x direction → drift≈2.5cm, e_F≈0, realized≈[5,0,0] N. Also robust to
    root drift (we read the live pelvis pose each frame)."""
    monkeypatch.setenv("HOLOSOMA_FORCE_CMD_LOCAL", "0,0,0,5,0,0")
    ml = MetricsLogger(ref_npz, mj_model)
    _, rmj, _ = ml._force_hand_ids[1]
    data = mujoco.MjData(mj_model)
    T = ref_npz["joint_pos"].shape[0]
    for t in range(T):
        _set_qpos_to_ref(mj_model, data, ref_npz, t)
        R_root = eu._quat_to_mat_wxyz(data.qpos[3:7].copy())
        data.xpos[rmj] = data.xpos[rmj] + R_root @ np.array([0.025, 0.0, 0.0])
        ml.log_frame(t, data)
    fs = ml.finalize(T)["force_summary"]
    assert abs(fs["force_drift_median_cm_right"] - 2.5) < 1e-2
    assert fs["force_target_err_median_cm_right"] < 1e-2
    assert fs["force_realization_err_median_N_right"] < 1e-2
    fr = fs["force_realized_median_pelvis_N_right"]
    assert abs(fr[0] - 5.0) < 1e-2 and abs(fr[1]) < 1e-2 and abs(fr[2]) < 1e-2


def test_build_metrics_output_surfaces_force_and_per_body(mj_model, ref_npz, monkeypatch):
    """build_metrics_output must no longer drop per-body arrays and must carry
    the force block to disk."""
    monkeypatch.setenv("HOLOSOMA_FORCE_CMD_LOCAL", "0,0,0,5,0,0")
    ml = MetricsLogger(ref_npz, mj_model)
    data = mujoco.MjData(mj_model)
    T = ref_npz["joint_pos"].shape[0]
    for t in range(T):
        _set_qpos_to_ref(mj_model, data, ref_npz, t)
        ml.log_frame(t, data)
    out = eu.build_metrics_output(ml.finalize(T), "clip", "model")
    assert out["per_frame_body_pos_error_per_body_cm"]
    assert "left_wrist_yaw_link" in out["body_names_matched"]
    assert out["stiffness_K"] == K
    assert out["force_cmd_local_N"] == [0, 0, 0, 5, 0, 0]
    assert "force_summary" in out and out["force_summary"]


def test_force_ext_injection_nonzero_and_noop(mj_model):
    """MuJoCoInterface injects F_ext at the hand point (nonzero qfrc) when set,
    and is a clean no-op when force_ext_pelvis is None/zero."""
    data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, data)
    kp = np.ones(29) * 100.0
    kd = np.ones(29) * 2.0
    tl = np.ones(29) * 100.0

    fe = np.array([[0, 0, 0], [10.0, 0, 0]])
    iface = MuJoCoInterface(mj_model, data, None, kp, kd, tl, steps_per_control=4, force_ext_pelvis=fe)
    assert iface._force_ext_hand_ids == [25, 32]
    data.qfrc_applied[:] = 0
    iface._apply_force_ext()
    assert np.linalg.norm(data.qfrc_applied) > 1.0  # ~10 N + moment arm

    iface0 = MuJoCoInterface(mj_model, data, None, kp, kd, tl, steps_per_control=4, force_ext_pelvis=None)
    data.qfrc_applied[:] = 0
    iface0._apply_force_ext()
    assert np.linalg.norm(data.qfrc_applied) == 0.0


# ---------------------------------------------------------------------------
# Force-arrow video overlay (orange=F_ext real, purple=F_cmd intent)
# ---------------------------------------------------------------------------
def test_parse_force_local_env_none_and_zero(monkeypatch):
    """parse_force_local_env → None when unset OR all-zero (so no arrow is
    drawn), and the (2,3) array otherwise. Distinct from the metrics parser
    which returns zeros for the F=0 baseline."""
    monkeypatch.delenv("HOLOSOMA_FORCE_EXT_LOCAL", raising=False)
    assert eu.parse_force_local_env("HOLOSOMA_FORCE_EXT_LOCAL") is None
    monkeypatch.setenv("HOLOSOMA_FORCE_EXT_LOCAL", "0,0,0,0,0,0")
    assert eu.parse_force_local_env("HOLOSOMA_FORCE_EXT_LOCAL") is None  # all-zero → None
    monkeypatch.setenv("HOLOSOMA_FORCE_EXT_LOCAL", "0,0,0,10,0,0")
    arr = eu.parse_force_local_env("HOLOSOMA_FORCE_EXT_LOCAL")
    assert arr.shape == (2, 3) and np.allclose(arr[1], [10, 0, 0]) and np.allclose(arr[0], 0)
    monkeypatch.setenv("HOLOSOMA_FORCE_EXT_LOCAL", "1,2,3")  # wrong length
    with pytest.raises(ValueError):
        eu.parse_force_local_env("HOLOSOMA_FORCE_EXT_LOCAL")


def _mini_scene_with_hands():
    """Minimal MuJoCo model exposing the two wrist_yaw bodies the arrows attach
    to, so we can drive xpos/xmat/qpos directly and inspect the drawn geom."""
    xml = """<mujoco><worldbody>
      <geom type="plane" size="2 2 0.1"/>
      <body name="left_wrist_yaw_link" pos="-0.2 0 0.9"><freejoint/><geom type="sphere" size="0.04"/></body>
      <body name="right_wrist_yaw_link" pos="0.2 0 0.9"><freejoint/><geom type="sphere" size="0.04"/></body>
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _arrow_dir(geom):
    """Recover the arrow's world direction from its connector pose.

    mjv_connector orients a geom's local +z axis along (to-from); column 2 of
    the geom rotation matrix is that unit direction."""
    return np.asarray(geom.mat).reshape(3, 3)[:, 2]


def test_arrow_direction_matches_world_force():
    """The arrow points along R_pelvis @ F_pelvis — the SAME world transform
    _apply_force_ext / _log_force_frame use. With identity pelvis, a pelvis +x
    force draws a world +x arrow; with a 90°-yaw pelvis it draws world +y."""
    m, d = _mini_scene_with_hands()
    scn = mujoco.MjvScene(m, maxgeom=100)
    # identity pelvis (qpos[3:7] = wxyz = 1,0,0,0 by default for the freejoint)
    scn.ngeom = 0
    eu._add_hand_force_arrows(scn, m, d, force_cmd_pelvis=None,
                              force_ext_pelvis=np.array([[0, 0, 0], [10.0, 0, 0]]))
    assert scn.ngeom == 1  # only right hand non-zero
    assert np.allclose(_arrow_dir(scn.geoms[0]), [1, 0, 0], atol=1e-6)  # world +x
    # rotate the RIGHT hand body's free-joint quat by +90° about z (wxyz)
    rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    # set the WORLD root quat used by _pelvis_world_R == qpos[3:7] (the FIRST
    # free joint). The first body in qpos order is left; ensure we rotate qpos[3:7].
    d.qpos[3:7] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]  # +90° yaw
    mujoco.mj_forward(m, d)
    scn.ngeom = 0
    eu._add_hand_force_arrows(scn, m, d, force_cmd_pelvis=None,
                              force_ext_pelvis=np.array([[0, 0, 0], [10.0, 0, 0]]))
    # pelvis +x rotated by +90° yaw → world +y
    assert np.allclose(_arrow_dir(scn.geoms[0]), [0, 1, 0], atol=1e-6)
    _ = rid


def test_arrow_sign_and_both_hands_and_skip_zero():
    """F_ext +x and F_cmd -x on the right hand draw ANTIPARALLEL arrows (the
    Eval-3 cancel case). Both hands draw when both non-zero. Zero vectors add
    no geom."""
    m, d = _mini_scene_with_hands()
    scn = mujoco.MjvScene(m, maxgeom=100)
    scn.ngeom = 0
    eu._add_hand_force_arrows(
        scn, m, d,
        force_cmd_pelvis=np.array([[0, 0, 0], [-10.0, 0, 0]]),
        force_ext_pelvis=np.array([[0, 0, 0], [10.0, 0, 0]]),
    )
    # right hand: one F_ext (+x) + one F_cmd (-x) = 2 geoms; left all-zero = 0
    assert scn.ngeom == 2
    d_ext = _arrow_dir(scn.geoms[0])  # F_ext added first
    d_cmd = _arrow_dir(scn.geoms[1])
    assert np.allclose(d_ext, [1, 0, 0], atol=1e-6)
    assert np.allclose(d_cmd, [-1, 0, 0], atol=1e-6)  # antiparallel
    assert np.dot(d_ext, d_cmd) < -0.99
    # both hands non-zero → 2 arrows (one each)
    scn.ngeom = 0
    eu._add_hand_force_arrows(
        scn, m, d,
        force_cmd_pelvis=None,
        force_ext_pelvis=np.array([[0, 0, 10.0], [0, 0, 10.0]]),
    )
    assert scn.ngeom == 2  # left + right
    # None/None and all-zero → nothing drawn
    scn.ngeom = 0
    eu._add_hand_force_arrows(scn, m, d, None, None)
    assert scn.ngeom == 0
    eu._add_hand_force_arrows(scn, m, d, force_cmd_pelvis=np.zeros((2, 3)), force_ext_pelvis=None)
    assert scn.ngeom == 0


def test_arrow_anchor_at_rubber_hand_point():
    """The arrow STARTS at the rubber_hand point xpos[wrist]+R_wrist@offset_b —
    the same point mj_applyFT pushes on — using the correct per-hand offset."""
    m, d = _mini_scene_with_hands()
    scn = mujoco.MjvScene(m, maxgeom=100)
    scn.ngeom = 0
    rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    expected = d.xpos[rid] + d.xmat[rid].reshape(3, 3) @ eu.FORCE_HAND_OFFSETS_B[1]
    eu._add_hand_force_arrows(scn, m, d, force_cmd_pelvis=None,
                              force_ext_pelvis=np.array([[0, 0, 0], [10.0, 0, 0]]))
    # For mjGEOM_ARROW, mjv_connector places geom.pos at the BASE ("from"), i.e.
    # the anchor itself (unlike capsule connectors, whose pos is the midpoint).
    assert np.allclose(np.asarray(scn.geoms[0].pos), expected, atol=1e-6)


def test_arrow_colors_orange_ext_purple_cmd():
    """Color contract: F_ext arrow is orange, F_cmd arrow is purple."""
    assert eu.FORCE_ARROW_RGBA_EXT[0] > 0.9 and eu.FORCE_ARROW_RGBA_EXT[2] < 0.1  # orange
    assert eu.FORCE_ARROW_RGBA_CMD[2] > 0.8 and eu.FORCE_ARROW_RGBA_CMD[1] < 0.4  # purple
    m, d = _mini_scene_with_hands()
    scn = mujoco.MjvScene(m, maxgeom=100)
    scn.ngeom = 0
    eu._add_hand_force_arrows(
        scn, m, d,
        force_cmd_pelvis=np.array([[0, 0, 0], [0, 0, 5.0]]),
        force_ext_pelvis=np.array([[0, 0, 0], [10.0, 0, 0]]),
    )
    assert np.allclose(scn.geoms[0].rgba, eu.FORCE_ARROW_RGBA_EXT)  # F_ext first → orange
    assert np.allclose(scn.geoms[1].rgba, eu.FORCE_ARROW_RGBA_CMD)  # F_cmd second → purple
