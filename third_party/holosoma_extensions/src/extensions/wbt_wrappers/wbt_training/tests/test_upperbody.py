"""Unit tests for the g1-wbt-2pt-force-upperbody preset (relative-frame TCP + composition).

All tests use MagicMock for env/simulator — no IsaacSim runtime needed. Reuses the
force-tracking test fixtures (same package) for the env/motion-command scaffolding.
"""

import torch

from wbt_training.tests.test_force_tracking import (
    _make_env_with_motion_command,
    _make_env_with_simulator,  # noqa: F401  (kept for parity / future sim-level tests)
)


def _prep_tcp_env(num_envs=2):
    """Env with motion_command + force buffers + a live body_quat_w/pos_w on the
    simulator. Reuses the force-tracking fixture, then guarantees the live wrist
    pose tensors exist (identity quats, zero positions) for the TCP FK path."""
    from wbt_training.config_values.force_tracking import HandForceConfig

    cfg = HandForceConfig()
    env, mc = _make_env_with_motion_command(num_envs=num_envs, cfg=cfg)
    # Live wrist FK inputs (sim ids 17/24 exist in the 50-body arrays).
    bp = torch.zeros(num_envs, 50, 3)
    bq = torch.zeros(num_envs, 50, 4)
    bq[..., 0] = 1.0  # identity wxyz
    env.simulator._robot.data.body_pos_w = bp
    env.simulator._robot.data.body_quat_w = bq
    return env, mc, cfg


# ---------------------------------------------------------------------------
# Task 1: relative-frame TCP reward funcs
# ---------------------------------------------------------------------------


def test_tcp_pos_relative_returns_num_envs_and_reads_relative_fields():
    from wbt_training.config_values.upperbody import tcp_track_pos_exp_relative

    env, mc, cfg = _prep_tcp_env()
    env._wbt_force_cmd_w.zero_()
    env._wbt_force_ext_w.zero_()
    r = tcp_track_pos_exp_relative(env, cfg=cfg)
    assert r.shape == (env.num_envs,)
    # F=0 and cur wrist at origin with identity quat: ref uses body_pos_relative_w
    # (zeros) + relative quat (identity) rotated offset; cur uses live quat (identity)
    # rotated offset. Same offset both sides → residual 0 → reward 1.
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-5)


def test_tcp_pos_relative_f_zero_is_pure_fingertip_tracking():
    """With F=0 the Eq.2 shift is 0, so target == raw relative fingertip ref."""
    from wbt_training.config_values.upperbody import tcp_track_pos_exp_relative

    env, mc, cfg = _prep_tcp_env()
    env._wbt_force_cmd_w.zero_()
    env._wbt_force_ext_w.zero_()
    # Move the live LEFT wrist 0.05 m in +x from its ref (both quats identity, both
    # positions zero anchored) → residual 0.05 on left hand only → err_l1 = 0.05.
    env.simulator._robot.data.body_pos_w[:, 17, :] = torch.tensor([0.05, 0.0, 0.0])
    r = tcp_track_pos_exp_relative(env, cfg=cfg)
    expected = torch.exp(torch.tensor(-0.05 * 2.0 / 0.15))
    assert torch.allclose(r, torch.full_like(r, expected.item()), atol=1e-4)


def test_tcp_pos_relative_shift_by_F_over_K():
    """World-frame F^cmd_w = (30,0,0) on left → target shift 30/200 = 0.15 m.
    Place live left wrist 0.15 m fwd of ref → residual 0 → reward 1."""
    from wbt_training.config_values.upperbody import tcp_track_pos_exp_relative

    env, mc, cfg = _prep_tcp_env()
    env._wbt_force_cmd_w.zero_()
    env._wbt_force_ext_w.zero_()
    env._wbt_force_cmd_w[:, 0, 0] = 30.0  # left hand, world +x
    env.simulator._robot.data.body_pos_w[:, 17, :] = torch.tensor([0.15, 0.0, 0.0])
    r = tcp_track_pos_exp_relative(env, cfg=cfg)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-4)


def test_tcp_pos_relative_reads_relative_not_absolute_anchor():
    """Frame trap for POSITION (mutation-verified): set body_pos_relative_w != body_pos_w,
    place the live wrists at the RELATIVE ref, and require reward 1. If the func read the
    absolute body_pos_w (the falcon global bug), the residual would be ~0.8 m and reward ~0.
    Without this test the three tests above are vacuous — the fixture leaves relative==absolute."""
    from wbt_training.config_values.upperbody import tcp_track_pos_exp_relative

    env, mc, cfg = _prep_tcp_env()
    env._wbt_force_cmd_w.zero_()
    env._wbt_force_ext_w.zero_()
    # relative ref = 0.10 m fwd on both wrists (motion idx 12,13); absolute ref = 0.90 m.
    mc.body_pos_relative_w = torch.zeros(env.num_envs, 14, 3)
    mc.body_pos_relative_w[:, 12, 0] = 0.10
    mc.body_pos_relative_w[:, 13, 0] = 0.10
    mc.body_pos_w = torch.zeros(env.num_envs, 14, 3)
    mc.body_pos_w[:, 12, 0] = 0.90
    mc.body_pos_w[:, 13, 0] = 0.90
    # live wrists (sim ids 17/24) placed AT the relative ref → residual 0 iff relative.
    env.simulator._robot.data.body_pos_w[:, 17, :] = torch.tensor([0.10, 0.0, 0.0])
    env.simulator._robot.data.body_pos_w[:, 24, :] = torch.tensor([0.10, 0.0, 0.0])
    r = tcp_track_pos_exp_relative(env, cfg=cfg)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-5)


def test_tcp_orn_relative_reads_relative_and_robot_quats():
    from wbt_training.config_values.upperbody import tcp_track_orn_exp_relative

    env, mc, cfg = _prep_tcp_env()
    # Fixture sets mc.body_quat_relative_w to identity (xyzw). Set the live robot quat.
    mc.robot_body_quat_w = torch.zeros(env.num_envs, 14, 4)
    mc.robot_body_quat_w[..., 3] = 1.0  # identity xyzw
    r = tcp_track_orn_exp_relative(env, cfg=cfg)
    assert r.shape == (env.num_envs,)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-5)


def test_tcp_orn_relative_does_not_read_absolute_body_quat_w():
    """Guard against reusing the GLOBAL falcon variant: the relative orn func must
    read body_quat_relative_w, not the absolute body_quat_w. We set the two to
    DIFFERENT values and assert the reward tracks the RELATIVE one."""
    from wbt_training.config_values.upperbody import tcp_track_orn_exp_relative

    env, mc, cfg = _prep_tcp_env()
    mc.robot_body_quat_w = torch.zeros(env.num_envs, 14, 4)
    mc.robot_body_quat_w[..., 3] = 1.0  # identity
    # relative == identity (matches robot) → reward 1; absolute set to a 180° flip.
    mc.body_quat_relative_w = torch.zeros(env.num_envs, 14, 4)
    mc.body_quat_relative_w[..., 3] = 1.0
    mc.body_quat_w = torch.zeros(env.num_envs, 14, 4)
    mc.body_quat_w[..., 0] = 1.0  # xyzw (1,0,0,0) = 180° about x — a DIFFERENT quat
    r = tcp_track_orn_exp_relative(env, cfg=cfg)
    # If the func wrongly read body_quat_w it would see a large error (r far below 1).
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-5)


def test_tcp_orn_relative_kernel_and_sigma():
    """Pin the orn kernel exp(-θ²/σ²) and σ=0.25 with a known small rotation so the
    identity-only tests above don't leave the kernel form / σ unverified (at θ=0 any
    kernel gives 1.0)."""
    from wbt_training.config_values.upperbody import tcp_track_orn_exp_relative

    theta = 0.1  # rad about z, applied to the live wrist only
    env, mc, cfg = _prep_tcp_env()
    mc.body_quat_relative_w = torch.zeros(env.num_envs, 14, 4)
    mc.body_quat_relative_w[..., 3] = 1.0  # identity (xyzw)
    mc.robot_body_quat_w = torch.zeros(env.num_envs, 14, 4)
    mc.robot_body_quat_w[..., 2] = torch.sin(torch.tensor(theta / 2))  # z (xyzw)
    mc.robot_body_quat_w[..., 3] = torch.cos(torch.tensor(theta / 2))  # w
    r = tcp_track_orn_exp_relative(env, cfg=cfg)
    # quat_error_magnitude gives θ per hand; mean over 2 hands = θ; exp(-θ²/σ²).
    expected = torch.exp(torch.tensor(-(theta**2) / 0.25**2))
    assert torch.allclose(r, torch.full_like(r, expected.item()), atol=1e-4)


# ---------------------------------------------------------------------------
# Task 2: assembled reward + termination cfgs
# ---------------------------------------------------------------------------


def test_upperbody_reward_adds_tcp_terms_only():
    """Reward == inherited 2pt-force terms + exactly {tcp_track_pos, tcp_track_orn}."""
    from wbt_training.config_values.force_tracking import g1_29dof_wbt_2pt_force_reward
    from wbt_training.config_values.upperbody import g1_29dof_wbt_2pt_force_upperbody_reward as rw

    inherited = set(g1_29dof_wbt_2pt_force_reward.terms.keys())
    new = set(rw.terms.keys())
    assert new - inherited == {"tcp_track_pos", "tcp_track_orn"}
    assert inherited - new == set()  # nothing removed
    # New term wiring: correct funcs, weights, sigmas.
    tp = rw.terms["tcp_track_pos"]
    to = rw.terms["tcp_track_orn"]
    assert tp.func.endswith(":tcp_track_pos_exp_relative") and tp.weight == 2.0
    assert tp.params.get("sigma") == 0.15
    assert to.func.endswith(":tcp_track_orn_exp_relative") and to.weight == 1.5
    assert to.params.get("sigma") == 0.25
    # Eq.2 unchanged at 6.0 (matches 2pt-force).
    assert rw.terms["force_track_pos_eq2"].weight == 6.0


def test_upperbody_termination_drops_ankles_keeps_wrists():
    from wbt_training.config_values.upperbody import g1_29dof_wbt_upperbody_termination as tm

    bt = tm.terms["bad_tracking"]
    names = tuple(bt.params["bad_motion_body_pos_body_names"])
    assert names == ("left_wrist_yaw_link", "right_wrist_yaw_link")
    # Thresholds and the timeout term are preserved.
    assert bt.params["bad_ref_pos_threshold"] == 0.5
    assert bt.params["bad_ref_ori_threshold"] == 0.8
    assert "timeout" in tm.terms  # timeout term preserved
    # body_names_to_track is preserved (14 bodies incl. both wrists) — required for
    # bad_motion_body_pos_body_names to remain a valid subset and for the runtime
    # assert (motion_cfg.body_names_to_track == term.body_names_to_track) to pass.
    tracked = tuple(bt.params["body_names_to_track"])
    assert len(tracked) == 14
    assert "left_wrist_yaw_link" in tracked and "right_wrist_yaw_link" in tracked


def test_upperbody_cfgs_do_not_mutate_bases():
    """replace() returns copies — the inherited base cfgs are untouched (object identity)."""
    from holosoma.config_values.wbt.g1.termination import g1_29dof_wbt_termination
    from wbt_training.config_values.force_tracking import g1_29dof_wbt_2pt_force_reward
    from wbt_training.config_values.upperbody import (
        g1_29dof_wbt_2pt_force_upperbody_reward,
        g1_29dof_wbt_upperbody_termination,
    )

    # Base reward has no TCP terms; base termination still gates the ankles.
    assert "tcp_track_pos" not in g1_29dof_wbt_2pt_force_reward.terms
    base_bt = g1_29dof_wbt_termination.terms["bad_tracking"]
    assert "left_ankle_roll_link" in tuple(base_bt.params["bad_motion_body_pos_body_names"])
    # The new cfgs are distinct objects.
    assert g1_29dof_wbt_2pt_force_upperbody_reward is not g1_29dof_wbt_2pt_force_reward
    assert g1_29dof_wbt_upperbody_termination is not g1_29dof_wbt_termination


# ---------------------------------------------------------------------------
# Task 3: preset registration
# ---------------------------------------------------------------------------


def test_preset_registered_and_composed():
    from wbt_training.config_values.experiment import DEFAULTS, g1_wbt_2pt_force, g1_wbt_2pt_force_upperbody

    # Registered under the exact key → CLI exp:g1-wbt-2pt-force-upperbody.
    assert DEFAULTS["g1_wbt_2pt_force_upperbody"] is g1_wbt_2pt_force_upperbody
    # Yam URDF baked in.
    assert g1_wbt_2pt_force_upperbody.robot.asset.urdf_file == "g1/g1_29dof_yam.urdf"
    # Reward = upperbody reward (has TCP terms); termination = ankle-relaxed.
    assert "tcp_track_pos" in g1_wbt_2pt_force_upperbody.reward.terms
    assert tuple(
        g1_wbt_2pt_force_upperbody.termination.terms["bad_tracking"].params["bad_motion_body_pos_body_names"]
    ) == ("left_wrist_yaw_link", "right_wrist_yaw_link")
    # Obs / command / curriculum / algo inherited UNCHANGED (same object identities).
    assert g1_wbt_2pt_force_upperbody.observation is g1_wbt_2pt_force.observation
    assert g1_wbt_2pt_force_upperbody.command is g1_wbt_2pt_force.command
    assert g1_wbt_2pt_force_upperbody.curriculum is g1_wbt_2pt_force.curriculum
    assert g1_wbt_2pt_force_upperbody.algo is g1_wbt_2pt_force.algo


def test_existing_presets_unmutated_by_upperbody():
    """Non-invasion: the parent and siblings stay object-identical after building
    the new preset. Guards against accidental in-place mutation."""
    from wbt_training.config_values.experiment import DEFAULTS, g1_wbt_2pt, g1_wbt_2pt_force

    # Parent 2pt-force keeps its own (non-yam-overridden) reward/termination.
    assert "tcp_track_pos" not in g1_wbt_2pt_force.reward.terms
    assert g1_wbt_2pt_force.termination is not DEFAULTS["g1_wbt_2pt_force_upperbody"].termination
    # 2pt (pre-force) has no force terms at all.
    assert "force_track_pos_eq2" not in g1_wbt_2pt.reward.terms
    # The new preset did not overwrite the parent's DEFAULTS entry.
    assert DEFAULTS["g1_wbt_2pt_force"] is g1_wbt_2pt_force


def test_cli_subcommand_name_maps():
    """The DEFAULTS key underscores→hyphens gives the exp: subcommand."""
    from wbt_training.config_values.experiment import DEFAULTS

    key = "g1_wbt_2pt_force_upperbody"
    assert key in DEFAULTS
    assert f"exp:{key.replace('_', '-')}" == "exp:g1-wbt-2pt-force-upperbody"
