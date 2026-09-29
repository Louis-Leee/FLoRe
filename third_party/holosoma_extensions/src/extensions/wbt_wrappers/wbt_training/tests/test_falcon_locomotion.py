"""Unit tests for the FALCON lean lower-body locomotion pieces (Step 5).

MagicMock env/simulator — no IsaacSim runtime. Covers the WBT-safe reward terms
and the stance-mask refresh logic of FalconVelocityCommand (tested in isolation
via __new__ to avoid the full command-manager lifecycle)."""

from unittest.mock import MagicMock

import torch


def _eye_quat(n):
    q = torch.zeros(n, 4)
    q[:, 3] = 1.0  # xyzw identity (holosoma root quat convention is xyzw / w_last)
    return q


def _make_loco_env(num_envs=3):
    env = MagicMock()
    env.num_envs = num_envs
    env.device = torch.device("cpu")
    sim = MagicMock()
    # robot_root_states: [px,py,pz, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
    root = torch.zeros(num_envs, 13)
    root[:, 6] = 1.0  # identity quat (xyzw)
    sim.robot_root_states = root
    sim.base_quat = _eye_quat(num_envs)
    env.simulator = sim
    return env


def test_upright_orientation_zero_when_level():
    from wbt_training.config_values.falcon_locomotion import upright_orientation

    env = _make_loco_env()
    r = upright_orientation(env)
    assert r.shape == (3,)
    assert torch.allclose(r, torch.zeros(3), atol=1e-6), "level base ⇒ zero orientation penalty"


def test_upright_orientation_positive_when_tilted():
    from wbt_training.config_values.falcon_locomotion import upright_orientation

    env = _make_loco_env(num_envs=1)
    # Tilt 90° about x (roll): projected gravity gains a y-component.
    s2 = (2**0.5) / 2.0
    env.simulator.base_quat = torch.tensor([[s2, 0.0, 0.0, s2]])  # xyzw, 90° about x
    r = upright_orientation(env)
    assert r.item() > 0.5, r.item()


def test_base_ang_vel_xy_penalty():
    from wbt_training.config_values.falcon_locomotion import base_ang_vel_xy_penalty

    env = _make_loco_env(num_envs=2)
    env.simulator.robot_root_states[:, 10] = 3.0  # roll rate
    env.simulator.robot_root_states[:, 11] = 4.0  # pitch rate
    env.simulator.robot_root_states[:, 12] = 9.0  # yaw rate (must be ignored)
    r = base_ang_vel_xy_penalty(env)
    assert torch.allclose(r, torch.full((2,), 25.0), atol=1e-5), "3²+4²=25, yaw excluded"


def test_alive_is_constant_one():
    from wbt_training.config_values.falcon_locomotion import alive

    env = _make_loco_env(num_envs=4)
    assert torch.allclose(alive(env), torch.ones(4))


def test_velocity_command_reads_command_manager():
    from wbt_training.config_values.falcon_locomotion import velocity_command

    env = _make_loco_env()
    cmds = torch.tensor([[0.5, 0.0, 0.1], [0.0, 0.0, 0.0], [1.0, -0.3, 0.2]])
    env.command_manager = MagicMock()
    env.command_manager.commands = cmds
    out = velocity_command(env)
    assert torch.allclose(out, cmds)


def test_stance_mask_refresh_marks_zero_command_envs():
    """FalconVelocityCommand._refresh_stance_mask must set stance=True exactly where
    the commanded velocity is ~0 (the stand envs)."""
    from wbt_training.config_values.falcon_locomotion import FalconVelocityCommand

    term = FalconVelocityCommand.__new__(FalconVelocityCommand)
    env = _make_loco_env(num_envs=4)
    env.stance_mask = torch.zeros(4, dtype=torch.bool)
    term.env = env
    # commands: env0 walk, env1 stand(0), env2 walk, env3 stand(0)
    cmds = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    # Patch the `commands` property to return our tensor.
    type(term).commands = property(lambda self: cmds)
    try:
        term._refresh_stance_mask(None)
        assert env.stance_mask.tolist() == [False, True, False, True]
    finally:
        del type(term).commands


def test_stance_mask_refresh_partial_env_ids():
    """Refreshing a subset of env_ids updates only those entries."""
    from wbt_training.config_values.falcon_locomotion import FalconVelocityCommand

    term = FalconVelocityCommand.__new__(FalconVelocityCommand)
    env = _make_loco_env(num_envs=4)
    env.stance_mask = torch.tensor([True, True, True, True])
    term.env = env
    cmds = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.0, 0.0]])
    type(term).commands = property(lambda self: cmds)
    try:
        # Only refresh envs {0, 2}; envs {1,3} keep their prior True.
        term._refresh_stance_mask(torch.tensor([0, 2]))
        assert env.stance_mask.tolist() == [False, True, False, True]
    finally:
        del type(term).commands
