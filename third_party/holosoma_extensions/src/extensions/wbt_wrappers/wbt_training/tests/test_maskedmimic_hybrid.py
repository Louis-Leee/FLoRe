"""Unit tests for the MaskedMimic + HYBRID position-force preset (exp:g1-maskmimic-hybrid).

All tests use MagicMock for env/simulator — no IsaacSim runtime. They cover the parts NEW to the
maskmimic-hybrid port:
  - the GLOBAL-frame servo Eq.2 rewards (force_track_pos_eq2_hybrid_global /
    force_track_eq2_servo_global), incl. the 6-D F_ext slice and the servo-inclusive ref+Δx branch;
  - the ACTOR joint-velocity dropout (per-episode / per-step / eval_zero / disabled), and that the
    CRITIC path is never dropped;
  - the obs layout (maskmimic prefix + obs90.. force/servo terms, lexicographic-safe, fixed width);
  - command/curriculum/termination wiring;
  - the servo-off variant (servo_enabled=False, identical reward KEYS + obs width);
  - preset registration (both exp names resolve, yam URDF, mm algo inherited);
  - NON-INVASION of g1_wbt_2pt_hybrid, g1_wbt_maskedmimic, g1_wbt_maskedmimic_force.

The shared hybrid physics (injector, servo, sensor, estop) and force machinery are covered by
test_hybrid.py + test_force_tracking.py and reused verbatim — not re-tested here.
"""

import types
from unittest.mock import MagicMock

import torch


# ===========================================================================
# Fixtures (mirror test_maskedmimic_force.py, but the HYBRID injector's F_ext buffer is 6-D).
# ===========================================================================


def _make_env_with_motion_command(num_envs=2, cfg=None):
    """Env with simulator + motion_command (global body_pos_w) + the HYBRID 6-D force buffers.

    The global reward reads mc.body_pos_w (absolute). body_pos_relative_w is a 99.0 decoy so a
    regression to the relative anchor is caught. _wbt_force_ext_w is 6-D (hybrid), _wbt_delta_x
    is the servo setpoint shift."""
    from wbt_training.config_values.force_tracking import HandForceConfig

    cfg = cfg or HandForceConfig()
    env = MagicMock()
    n, h = num_envs, len(cfg.body_names)
    env.num_envs = num_envs
    env.device = torch.device("cpu")
    env.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)

    sim = MagicMock()
    sim.body_ids = list(range(50))
    # Resolve wrist names → isaac indices 17/24 (identity through body_ids, as in the
    # maskmimic-force fixture) so _force_cur_pos_w / _force_ref_pos_w_global read the placed hands.
    sim.find_rigid_body_indice = MagicMock(
        side_effect=lambda name: {"left_wrist_yaw_link": 17, "right_wrist_yaw_link": 24}[name]
    )
    sim._robot = MagicMock()
    sim._robot.data = types.SimpleNamespace(
        body_quat_w=torch.zeros(num_envs, 50, 4),
        root_quat_w=torch.zeros(num_envs, 4),
        body_pos_w=torch.zeros(num_envs, 50, 3),
    )
    sim._robot.data.body_quat_w[..., 0] = 1.0
    sim._robot.data.root_quat_w[..., 0] = 1.0
    env.simulator = sim

    # Force buffers (HYBRID widths: cmd 3-D, ext 6-D, meas 6-D, integ/delta 3-D).
    env._wbt_force_cmd_w = torch.zeros(n, h, 3)
    env._wbt_force_ext_w = torch.zeros(n, h, 6)
    env._wbt_force_meas_w = torch.zeros(n, h, 6)
    env._wbt_integ_state = torch.zeros(n, h, 3)
    env._wbt_delta_x = torch.zeros(n, h, 3)

    # motion_command with 14 tracked bodies; wrists at idx 12/13 (as in maskmimic-force fixture).
    mc = MagicMock()
    body_names_list = ["pelvis"] + [f"x{i}" for i in range(11)] + ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    mc.motion_cfg = types.SimpleNamespace(body_names_to_track=body_names_list)
    mc.body_pos_w = torch.zeros(num_envs, 14, 3)
    mc.robot_body_pos_w = torch.zeros(num_envs, 14, 3)
    mc.body_pos_relative_w = torch.full((num_envs, 14, 3), 99.0)  # decoy
    cm = MagicMock()
    cm.get_state = MagicMock(side_effect=lambda name: mc if name == "motion_command" else None)
    env.command_manager = cm
    return env, mc


def _place_hands(env, left_xyz, right_xyz):
    bp = torch.zeros(env.num_envs, 50, 3)
    bp[:, 17, :] = torch.tensor(left_xyz)  # left_wrist_yaw isaac idx (fixture convention)
    bp[:, 24, :] = torch.tensor(right_xyz)  # right_wrist_yaw
    env.simulator._robot.data.body_pos_w = bp


# ===========================================================================
# 1. GLOBAL-frame servo Eq.2 rewards.
# ===========================================================================


def test_servo_off_global_eq2_equals_layerA_when_servo_disabled():
    """force_track_eq2_servo_global with servo_enabled=False delegates to
    force_track_pos_eq2_hybrid_global (byte-identical result)."""
    from wbt_training.config_values.force_tracking import HandForceConfig
    from wbt_training.config_values.hybrid import HybridContactConfig
    from wbt_training.config_values.maskedmimic_hybrid import (
        force_track_eq2_servo_global,
        force_track_pos_eq2_hybrid_global,
    )

    cfg = HandForceConfig()
    env, _ = _make_env_with_motion_command(cfg=cfg)
    env._wbt_hybrid_cfg = HybridContactConfig(servo_enabled=False)
    env._wbt_force_cmd_w.uniform_(-3, 3)
    env._wbt_force_ext_w[..., :3].uniform_(-3, 3)
    _place_hands(env, [0.1, 0.0, 0.0], [0.0, 0.2, 0.0])
    r_servo = force_track_eq2_servo_global(env, cfg=cfg)
    r_layerA = force_track_pos_eq2_hybrid_global(env, cfg=cfg)
    assert torch.allclose(r_servo, r_layerA, atol=1e-6)


def test_servo_on_global_eq2_uses_ref_plus_delta_x():
    """servo_enabled=True → target = ref_global + Δx. Δx=(0.1,0,0) on left hand, hand placed
    exactly there → r=1. Confirms the servo path reads _wbt_delta_x, not (F_cmd+F_ext)/K."""
    from wbt_training.config_values.force_tracking import HandForceConfig
    from wbt_training.config_values.hybrid import HybridContactConfig
    from wbt_training.config_values.maskedmimic_hybrid import force_track_eq2_servo_global

    cfg = HandForceConfig()
    env, mc = _make_env_with_motion_command(cfg=cfg)
    env._wbt_hybrid_cfg = HybridContactConfig(servo_enabled=True)
    env._wbt_force_cmd_w.uniform_(-5, 5)  # ignored on the servo path
    env._wbt_force_ext_w.uniform_(-5, 5)  # ignored on the servo path
    env._wbt_delta_x.zero_()
    env._wbt_delta_x[:, 0, 0] = 0.10  # left hand setpoint shift +x 0.10 m
    _place_hands(env, [0.10, 0.0, 0.0], [0.0, 0.0, 0.0])
    r = force_track_eq2_servo_global(env, cfg=cfg)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-4)


def test_layerA_global_eq2_F_zero_is_pure_tracking():
    """F=0 ⇒ target = ref_global ⇒ hand at ref → r=1 (the F=0⇒tracking invariant)."""
    from wbt_training.config_values.force_tracking import HandForceConfig
    from wbt_training.config_values.maskedmimic_hybrid import force_track_pos_eq2_hybrid_global

    cfg = HandForceConfig()
    env, mc = _make_env_with_motion_command(cfg=cfg)
    env._wbt_force_cmd_w.zero_()
    env._wbt_force_ext_w.zero_()
    _place_hands(env, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    r = force_track_pos_eq2_hybrid_global(env, cfg=cfg)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-5)


def test_layerA_global_eq2_anchors_on_body_pos_w_not_relative():
    """Global variant must anchor on mc.body_pos_w (absolute), NOT the 99.0 body_pos_relative_w
    decoy. Hands placed at the non-trivial global ref → r=1; a relative-anchor regression → r≈0."""
    from wbt_training.config_values.force_tracking import HandForceConfig
    from wbt_training.config_values.maskedmimic_hybrid import force_track_pos_eq2_hybrid_global

    cfg = HandForceConfig()
    env, mc = _make_env_with_motion_command(cfg=cfg)
    env._wbt_force_cmd_w.zero_()
    env._wbt_force_ext_w.zero_()
    mc.body_pos_w[:, 12, :] = torch.tensor([1.0, 2.0, 3.0])
    mc.body_pos_w[:, 13, :] = torch.tensor([4.0, 5.0, 6.0])
    _place_hands(env, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    r = force_track_pos_eq2_hybrid_global(env, cfg=cfg)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-4), (
        f"got {r}; if << 1 the global reward regressed to the 99.0 body_pos_relative_w decoy."
    )


def test_layerA_global_eq2_slices_6d_ext_buffer():
    """The hybrid F_ext buffer is 6-D; the reward must slice [:3]. Fill the MOMENT channels
    ([3:]) with garbage — a correct reward ignores them (r unchanged); a non-slicing reward would
    broadcast-fail (3-D cmd + 6-D ext) or corrupt the target."""
    from wbt_training.config_values.force_tracking import HandForceConfig
    from wbt_training.config_values.maskedmimic_hybrid import force_track_pos_eq2_hybrid_global

    cfg = HandForceConfig(stiffness_K=200.0)
    env, mc = _make_env_with_motion_command(cfg=cfg)
    env._wbt_force_cmd_w.zero_()
    env._wbt_force_ext_w.zero_()
    env._wbt_force_ext_w[:, 0, 0] = 20.0  # left hand +x 20 N force → target +0.10 m
    env._wbt_force_ext_w[..., 3:] = 123.0  # MOMENT garbage — must be sliced off
    _place_hands(env, [0.10, 0.0, 0.0], [0.0, 0.0, 0.0])
    r = force_track_pos_eq2_hybrid_global(env, cfg=cfg)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-4)


def test_global_eq2_l1_sum_and_2_over_sigma_scale():
    """Each hand 0.1 m off in x, no force → err_l1 = 0.2 (sum over hands & xyz),
    r = exp(-0.2·2/0.3). Catches L2/σ² and mean-vs-sum drift."""
    from wbt_training.config_values.force_tracking import HandForceConfig
    from wbt_training.config_values.maskedmimic_hybrid import force_track_pos_eq2_hybrid_global

    cfg = HandForceConfig()
    env, mc = _make_env_with_motion_command(cfg=cfg)
    env._wbt_force_cmd_w.zero_()
    env._wbt_force_ext_w.zero_()
    _place_hands(env, [0.10, 0.0, 0.0], [0.10, 0.0, 0.0])
    r = force_track_pos_eq2_hybrid_global(env, cfg=cfg)
    expected = torch.exp(torch.tensor(-0.2 * 2.0 / 0.3))
    assert torch.allclose(r, torch.full_like(r, expected.item()), atol=1e-4)


# ===========================================================================
# 2. ACTOR joint-velocity dropout.
# ===========================================================================


class _FakeProvider:
    """Minimal stand-in exposing just what _apply_jv_dropout / _ensure_jv_mask / reset touch."""

    def __init__(self, num_envs, device, jv_cfg):
        self.num_envs = num_envs
        self.device = device
        self.jv_cfg = jv_cfg
        self._jv_dropout_mask = None


def _bind(provider):
    """Bind the real provider methods onto the fake (they only use self.num_envs/device/jv_cfg/
    _jv_dropout_mask), so we can unit-test the dropout math without an IsaacSim env."""
    from wbt_training.config_values.maskedmimic_hybrid import _MaskedMimicHybridObservationProvider as P

    provider._ensure_jv_mask = types.MethodType(P._ensure_jv_mask, provider)
    provider._apply_jv_dropout = types.MethodType(P._apply_jv_dropout, provider)
    return provider


def test_jv_dropout_disabled_is_identity():
    from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig

    p = _bind(_FakeProvider(4, torch.device("cpu"), JointVelDropoutConfig(enabled=False)))
    jv = torch.randn(4, 29)
    out = p._apply_jv_dropout(jv)
    assert torch.equal(out, jv)


def test_jv_dropout_eval_zero_zeros_everything():
    from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig

    p = _bind(_FakeProvider(4, torch.device("cpu"), JointVelDropoutConfig(enabled=True, eval_zero=True)))
    jv = torch.randn(4, 29)
    out = p._apply_jv_dropout(jv)
    assert torch.count_nonzero(out) == 0


def test_jv_dropout_per_episode_zeros_exactly_masked_envs():
    from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig

    p = _bind(_FakeProvider(6, torch.device("cpu"), JointVelDropoutConfig(enabled=True, per_episode=True)))
    # Force a deterministic mask: envs 1,3,5 dropped.
    p._jv_dropout_mask = torch.tensor([False, True, False, True, False, True])
    jv = torch.ones(6, 29)
    out = p._apply_jv_dropout(jv)
    dropped = out.abs().sum(dim=-1) == 0
    assert torch.equal(dropped, torch.tensor([False, True, False, True, False, True]))
    # Non-dropped envs are unchanged (no noise_scale).
    assert torch.equal(out[0], torch.ones(29))


def test_jv_dropout_prob_1_drops_all_prob_0_drops_none():
    from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig

    p1 = _bind(_FakeProvider(8, torch.device("cpu"), JointVelDropoutConfig(enabled=True, dropout_prob=1.0)))
    out1 = p1._apply_jv_dropout(torch.ones(8, 29))
    assert torch.count_nonzero(out1) == 0
    p0 = _bind(_FakeProvider(8, torch.device("cpu"), JointVelDropoutConfig(enabled=True, dropout_prob=0.0)))
    out0 = p0._apply_jv_dropout(torch.ones(8, 29))
    assert torch.equal(out0, torch.ones(8, 29))


def test_jv_dropout_lazy_mask_is_sampled_not_zero_filled():
    """First access allocates the per-episode mask by SAMPLING (rand<prob), not zero-fill —
    so episode-0 envs get real coverage even if reset never fired. With prob=1.0 the lazy mask
    must be all-True (all dropped)."""
    from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig

    p = _bind(_FakeProvider(16, torch.device("cpu"), JointVelDropoutConfig(enabled=True, dropout_prob=1.0)))
    mask = p._ensure_jv_mask()
    assert mask.dtype == torch.bool and mask.shape == (16,)
    assert bool(mask.all()), "prob=1.0 lazy mask must be all-True (sampled, not zero-filled)."


def test_jv_dropout_per_step_varies():
    """per_step mode samples a fresh Bernoulli each call — two calls with prob=0.5 over many
    envs should (almost surely) differ."""
    from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig

    p = _bind(_FakeProvider(2048, torch.device("cpu"), JointVelDropoutConfig(enabled=True, per_episode=False)))
    a = p._apply_jv_dropout(torch.ones(2048, 29)).abs().sum(dim=-1) == 0
    b = p._apply_jv_dropout(torch.ones(2048, 29)).abs().sum(dim=-1) == 0
    assert not torch.equal(a, b), "per_step masks should differ across calls (prob 0.5, 2048 envs)."


def test_jv_dropout_reduced_coords_actor_only_critic_clean():
    """The dropout hook lives in the apply_noise=True (actor) branch of _reduced_coords_obs.
    Build a fake provider, call the REAL _reduced_coords_obs for both branches with prob=1.0, and
    assert the actor's joint_vel slice is zeroed while the critic's is intact."""
    from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig
    from wbt_training.config_values.maskedmimic_hybrid import _MaskedMimicHybridObservationProvider as P

    n, ndof = 4, 29
    p = _FakeProvider(n, torch.device("cpu"), JointVelDropoutConfig(enabled=True, dropout_prob=1.0, per_episode=True))
    # Stub the base sub-obs the copied _reduced_coords_obs calls.
    jp = torch.randn(n, ndof)
    jv = torch.randn(n, ndof).abs() + 1.0  # strictly nonzero so "zeroed" is unambiguous
    pg = torch.randn(n, 3)
    bav = torch.randn(n, 3)
    p._joint_pos_rel = lambda: jp
    p._joint_vel_rel = lambda: jv
    p._projected_gravity_obs = lambda: pg
    p._base_ang_vel_obs = lambda: bav
    p.obs_cfg = types.SimpleNamespace(include_base_lin_vel=False)
    p._s2r_enabled = False
    p._ensure_jv_mask = types.MethodType(P._ensure_jv_mask, p)
    p._apply_jv_dropout = types.MethodType(P._apply_jv_dropout, p)
    reduced_obs = types.MethodType(P._reduced_coords_obs, p)

    actor_reduced, _ = reduced_obs(apply_noise=True)
    critic_reduced, _ = reduced_obs(apply_noise=False)
    # layout: [joint_pos(29), joint_vel(29), proj_grav(3), base_ang_vel(3)]
    actor_jv = actor_reduced[:, ndof : 2 * ndof]
    critic_jv = critic_reduced[:, ndof : 2 * ndof]
    assert torch.count_nonzero(actor_jv) == 0, "actor joint_vel must be fully dropped (prob=1.0)."
    assert torch.count_nonzero(critic_jv) > 0, "critic joint_vel must be untouched (apply_noise=False)."
    # And the non-joint_vel components of the actor match the base (proj_grav/base_ang_vel present).
    assert critic_reduced[:, :ndof].shape == (n, ndof)


def _parity_check_reduced_coords(include_base_lin_vel: bool):
    """Assert the subclass _reduced_coords_obs (dropout disabled) reproduces the base method
    EXACTLY (layout + values) on the same stubs, for a given include_base_lin_vel. Guards the
    verbatim copy against silent drift if the base method changes."""
    from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig
    from wbt_training.config_values.maskedmimic_hybrid import _MaskedMimicHybridObservationProvider as P
    from wbt_training.config_values.maskedmimic_observations import _MaskedMimicObservationProvider as BaseP

    n, ndof = 3, 29
    jp, jv, pg, bav, blv = (
        torch.randn(n, ndof),
        torch.randn(n, ndof),
        torch.randn(n, 3),
        torch.randn(n, 3),
        torch.randn(n, 3),
    )
    sub = _FakeProvider(n, torch.device("cpu"), JointVelDropoutConfig(enabled=False))
    sub._joint_pos_rel = lambda: jp
    sub._joint_vel_rel = lambda: jv
    sub._projected_gravity_obs = lambda: pg
    sub._base_ang_vel_obs = lambda: bav
    sub._base_lin_vel_obs = lambda: blv
    sub.obs_cfg = types.SimpleNamespace(include_base_lin_vel=include_base_lin_vel)
    sub._s2r_enabled = False
    sub._ensure_jv_mask = types.MethodType(P._ensure_jv_mask, sub)
    sub._apply_jv_dropout = types.MethodType(P._apply_jv_dropout, sub)
    sub_reduced = types.MethodType(P._reduced_coords_obs, sub)
    base_reduced = types.MethodType(BaseP._reduced_coords_obs, sub)  # base method on same stubs
    # apply_noise=False (deterministic, no random noise) → exact parity.
    r_sub, _ = sub_reduced(apply_noise=False)
    r_base, _ = base_reduced(apply_noise=False)
    assert torch.equal(r_sub, r_base), (
        f"subclass _reduced_coords_obs diverged from base when disabled (include_base_lin_vel={include_base_lin_vel})."
    )
    return r_sub


def test_jv_dropout_reduced_coords_parity_when_disabled():
    """Parity with the base method must hold for BOTH include_base_lin_vel values — production
    uses True (the maskmimic obs config), so the copy must not diverge on that path either."""
    r_no_blv = _parity_check_reduced_coords(include_base_lin_vel=False)
    r_blv = _parity_check_reduced_coords(include_base_lin_vel=True)
    # base_lin_vel adds a 3-D block: with-blv width must exceed without-blv by exactly 3.
    assert r_blv.shape[-1] == r_no_blv.shape[-1] + 3


# ===========================================================================
# 2c. Teleport-aware command: re-seed the hybrid causal state at motion boundaries.
# ===========================================================================


def _make_teleport_cmd(num_envs=4):
    """Build a MaskedMimicHybridMultiMotionCommand via __new__ (skip the heavy __init__) with just
    the fields step() touches: time_steps, motion_ids, motion.motion_end_idx, device, and an _env
    whose command_manager.get_state returns a spy injector. super().step() is stubbed so we test
    ONLY the re-seed logic (not the parent teleport)."""
    from unittest.mock import MagicMock

    from wbt_training.config_values.maskedmimic_hybrid import MaskedMimicHybridMultiMotionCommand

    cmd = MaskedMimicHybridMultiMotionCommand.__new__(MaskedMimicHybridMultiMotionCommand)
    cmd.device = torch.device("cpu")
    cmd.time_steps = torch.zeros(num_envs, dtype=torch.long)
    cmd.motion_ids = torch.zeros(num_envs, dtype=torch.long)
    cmd.motion = types.SimpleNamespace(motion_end_idx=torch.tensor([10]))  # single motion, ends at step 10
    injector = MagicMock()
    env = MagicMock()
    env.command_manager.get_state = MagicMock(
        side_effect=lambda name: injector if name == "hybrid_contact_injector" else None
    )
    cmd._env = env
    return cmd, injector


def test_teleport_reseeds_injector_for_ended_envs():
    """When an env's clip ends this step, the hybrid command must call injector.reset() with those
    env ids AFTER the parent teleport (so the plane/K_env/bias/integrator re-seed at the new pose)."""
    cmd, injector = _make_teleport_cmd(num_envs=4)
    # envs 1 and 3 are at the boundary: time_steps+1 (=10) >= motion_end (10).
    cmd.time_steps = torch.tensor([0, 9, 5, 9])
    # Stub the parent step() so we isolate the re-seed logic (base does the teleport + pulse restart).
    from wbt_training.config_values.maskedmimic_commands import MaskedMimicMultiMotionCommand

    orig = MaskedMimicMultiMotionCommand.step
    try:
        MaskedMimicMultiMotionCommand.step = lambda self: None
        cmd.step()
    finally:
        MaskedMimicMultiMotionCommand.step = orig
    assert injector.reset.call_count == 1
    called_ids = injector.reset.call_args[0][0]
    assert sorted(called_ids.tolist()) == [1, 3], f"expected envs [1,3] re-seeded, got {called_ids.tolist()}"


def test_teleport_no_reseed_when_no_env_ended():
    """No env at a boundary → injector.reset() must NOT be called (no spurious re-seed)."""
    cmd, injector = _make_teleport_cmd(num_envs=4)
    cmd.time_steps = torch.tensor([0, 1, 2, 3])  # none reach motion_end-1
    from wbt_training.config_values.maskedmimic_commands import MaskedMimicMultiMotionCommand

    orig = MaskedMimicMultiMotionCommand.step
    try:
        MaskedMimicMultiMotionCommand.step = lambda self: None
        cmd.step()
    finally:
        MaskedMimicMultiMotionCommand.step = orig
    injector.reset.assert_not_called()


def test_teleport_reseed_missing_injector_is_noop():
    """If the injector is somehow absent (get_state → None), step() must not crash."""
    cmd, injector = _make_teleport_cmd(num_envs=2)
    cmd._env.command_manager.get_state = MagicMock(return_value=None)
    cmd.time_steps = torch.tensor([9, 9])
    from wbt_training.config_values.maskedmimic_commands import MaskedMimicMultiMotionCommand

    orig = MaskedMimicMultiMotionCommand.step
    try:
        MaskedMimicMultiMotionCommand.step = lambda self: None
        cmd.step()  # must not raise
    finally:
        MaskedMimicMultiMotionCommand.step = orig


# ===========================================================================
# 3. Preset structure: obs layout, reward, command, curriculum, termination.
# ===========================================================================


def test_obs_layout_actor_prefix_and_force_terms():
    from wbt_training.config_values.maskedmimic_hybrid import g1_29dof_wbt_observation_maskedmimic_hybrid as obs

    a = sorted(obs.groups["actor_obs"].terms.keys())
    # maskmimic actor block (obs00..obs07) is a strict prefix; then obs90..93.
    assert a[:8] == [
        f"obs0{i}_" + s
        for i, s in enumerate(
            [
                "noisy_reduced_coords_obs",
                "masked_mimic_target_poses",
                "masked_mimic_target_masks",
                "joint_space_target_poses",
                "joint_space_mode_mask",
                "historical_pose_obs",
                "previous_actions",
                "end_effector_masses",
            ]
        )
    ]
    assert a[8:] == ["obs90_force_cmd_local", "obs91_force_meas_local", "obs92_integrator_state", "obs93_x_tgt_shift"]
    assert obs.groups["actor_obs"].history_length == 1
    # The privileged critic-only sim ground-truth must NEVER be in the actor obs (breaks sim2real).
    for priv in ("contact_gate", "force_ext_local", "k_env_privileged", "penetration_privileged"):
        assert not any(priv in k for k in a), f"privileged term {priv} leaked into the ACTOR obs."


def test_obs_layout_critic_has_privileged_terms():
    from wbt_training.config_values.maskedmimic_hybrid import g1_29dof_wbt_observation_maskedmimic_hybrid as obs

    c = sorted(obs.groups["critic_obs"].terms.keys())
    # 18 base maskmimic critic terms (obs00..obs17) then obs90..97.
    assert c[:18] == [k for k in c if k < "obs90"]
    assert c[18:] == [
        "obs90_force_cmd_local",
        "obs91_force_meas_local",
        "obs92_integrator_state",
        "obs93_x_tgt_shift",
        "obs94_contact_gate",
        "obs95_force_ext_local",
        "obs96_k_env_privileged",
        "obs97_penetration_privileged",
    ]
    assert obs.groups["critic_obs"].history_length == 1


def test_obs_lexicographic_prefix_invariant():
    """obs17 < obs90 so the maskmimic block stays a strict prefix under sorted-key concat."""
    assert "obs17_critic_end_effector_masses" < "obs90_force_cmd_local"
    assert "obs07_end_effector_masses" < "obs90_force_cmd_local"


def test_reward_swaps_and_weights():
    from wbt_training.config_values.maskedmimic_hybrid import g1_29dof_wbt_maskedmimic_hybrid_reward as rew

    keys = rew.terms.keys()
    assert "motion_global_body_pos" not in keys, "full-body global pos must be swapped for no-wrist."
    assert "motion_global_wrist_pos" not in keys, "dedicated wrist-pos must be dropped (Eq.2 owns it)."
    assert "motion_global_body_pos_no_wrist" in keys
    assert "force_track_pos_eq2" in keys and rew.terms["force_track_pos_eq2"].weight == 6.0
    assert rew.terms["force_track_pos_eq2"].func.endswith(":force_track_eq2_servo_global")
    assert "force_error_preintegrator" in keys and rew.terms["force_error_preintegrator"].weight == 1.0
    assert "feet_slip" in keys and rew.terms["feet_slip"].weight == -2.0
    # KEPT wrist ori/vel (orthogonal to Eq.2's position target).
    for k in ("motion_global_wrist_ori", "motion_global_wrist_lin_vel", "motion_global_wrist_ang_vel"):
        assert k in keys, f"{k} must be kept (orthogonal to Eq.2)."


def test_command_wires_injector_setup_and_reset_not_step():
    from wbt_training.config_values.maskedmimic_hybrid import g1_29dof_wbt_maskedmimic_hybrid_command as cmd

    assert "hybrid_contact_injector" in cmd.setup_terms
    assert "hybrid_contact_injector" in cmd.reset_terms
    assert "hybrid_contact_injector" not in cmd.step_terms, "injector advances on the substep hook, not step()."
    assert "motion_command" in cmd.step_terms


def test_command_motion_term_repointed_to_teleport_aware_subclass():
    """motion_command must use the teleport-aware hybrid subclass in ALL of setup/reset/step
    (so the mid-episode teleport re-seed fires), with the motion_config params preserved."""
    from wbt_training.config_values.maskedmimic_hybrid import g1_29dof_wbt_maskedmimic_hybrid_command as cmd

    for grp in (cmd.setup_terms, cmd.reset_terms, cmd.step_terms):
        mc = grp["motion_command"]
        assert mc.func.endswith(":MaskedMimicHybridMultiMotionCommand"), mc.func
    assert "motion_config" in cmd.setup_terms["motion_command"].params, "motion_config params must survive repoint."


def test_curriculum_has_force_alpha_and_termination_has_estop():
    from wbt_training.config_values.maskedmimic_hybrid import (
        g1_29dof_wbt_maskedmimic_hybrid_curriculum as cur,
        g1_29dof_wbt_maskedmimic_hybrid_termination as term,
    )

    assert "force_alpha" in cur.setup_terms
    assert "force_runaway" in term.terms
    assert "timeout" in term.terms and "bad_tracking" in term.terms


# ===========================================================================
# 4. servo-off variant.
# ===========================================================================


def test_servo_off_variant_disables_servo_same_keys():
    from wbt_training.config_values.maskedmimic_hybrid import (
        g1_29dof_wbt_maskedmimic_hybrid_command_servo_off as cmd_off,
        g1_29dof_wbt_maskedmimic_hybrid_reward as rew_on,
        g1_29dof_wbt_maskedmimic_hybrid_reward_servo_off as rew_off,
    )

    hc = cmd_off.setup_terms["hybrid_contact_injector"].params["hybrid_config"]
    assert hc.servo_enabled is False
    # Identical reward KEYS (checkpoint/critic lineage) — only the cfg-carriers differ.
    assert set(rew_on.terms.keys()) == set(rew_off.terms.keys())
    # Same weights too (6.0 in both — the servo-off phase must not be under-weighted).
    assert rew_off.terms["force_track_pos_eq2"].weight == 6.0


# ===========================================================================
# 5. Preset registration + NON-INVASION.
# ===========================================================================


def test_presets_registered_and_resolve():
    from wbt_training.config_values.experiment import DEFAULTS

    for k in ("g1_maskmimic_hybrid", "g1_maskmimic_hybrid_servo_off"):
        assert k in DEFAULTS, f"{k} not registered."
    exp = DEFAULTS["g1_maskmimic_hybrid"]
    assert exp.robot.asset.urdf_file == "g1/g1_29dof_yam.urdf"
    # mm algo inherited (low LR + adaptive KL).
    assert exp.algo.config.actor_learning_rate == 2e-5
    assert exp.algo.config.schedule == "adaptive"
    # tyro exp-name derivation.
    names = {f"exp:{k.replace('_', '-')}" for k in DEFAULTS}
    assert "exp:g1-maskmimic-hybrid" in names
    assert "exp:g1-maskmimic-hybrid-servo-off" in names


def test_non_invasion_of_other_presets():
    from wbt_training.config_values.experiment import DEFAULTS

    # 2pt-hybrid untouched (servo still ON, still relative-frame Eq.2 func).
    two = DEFAULTS["g1_wbt_2pt_hybrid"]
    hc = two.command.setup_terms["hybrid_contact_injector"].params["hybrid_config"]
    assert hc.servo_enabled is True
    assert two.reward.terms["force_track_pos_eq2"].func.endswith(":force_track_eq2_servo"), (
        "2pt-hybrid must keep the RELATIVE-frame Eq.2 func, not the global one."
    )
    # base maskmimic untouched (still has both body-pos and dedicated wrist-pos).
    mm = DEFAULTS["g1_wbt_maskedmimic"]
    assert "motion_global_body_pos" in mm.reward.terms
    assert "motion_global_wrist_pos" in mm.reward.terms
    assert "force_track_pos_eq2" not in mm.reward.terms
    # maskmimic-force untouched (Eq.2 global at 6.0, no servo/preintegrator terms).
    mf = DEFAULTS["g1_wbt_maskedmimic_force"]
    assert mf.reward.terms["force_track_pos_eq2"].func.endswith(":force_track_pos_eq2_global")
    assert "force_error_preintegrator" not in mf.reward.terms
    assert "integrator_state" not in [t.split("_", 1)[-1] for t in mf.observation.groups["actor_obs"].terms]


def test_actor_obs_no_servo_state_leak_into_force_only_presets():
    """maskmimic-force actor obs must NOT gain the hybrid servo terms (I / Δx)."""
    from wbt_training.config_values.experiment import DEFAULTS

    mf_actor = DEFAULTS["g1_wbt_maskedmimic_force"].observation.groups["actor_obs"].terms
    for bad in ("integrator_state", "x_tgt_shift", "force_meas_local"):
        assert not any(bad in k for k in mf_actor), f"{bad} leaked into maskmimic-force actor obs."
