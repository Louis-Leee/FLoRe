"""Unit tests for the force-tracking variant.

All tests use MagicMock for env/simulator — no IsaacSim runtime needed.
"""

import torch
import types
from unittest.mock import MagicMock


def test_hand_force_config_defaults_match_gh_v2():
    from wbt_training.config_values.force_tracking import HandForceConfig

    cfg = HandForceConfig()
    # Pulse magnitudes (UniFP `force_period_min/max=[3.5, 9.0]s`, ramp `[1.0, 3.0]s` @ 50 Hz)
    assert cfg.f_cmd_max == 25.0
    assert cfg.f_ext_max == 25.0
    # K=200 N/m: UniFP `force_kp` nominal (Math Foundation §6).
    assert cfg.stiffness_K == 200.0
    # σ=0.3 m: UniFP single-hand 0.5 → double-hand sum 0.25 → round to 0.3.
    assert cfg.sigma == 0.3
    # Penalty threshold 0.6 m: 2.4× legal max offset (F_max/K = 50/200 = 0.25 m).
    assert cfg.pen_threshold == 0.6
    assert cfg.body_names == ("left_wrist_yaw_link", "right_wrist_yaw_link")
    # Curriculum DISABLED by default — full force from training step 0.
    # `alpha_step=0.0` makes ForceAlphaCurriculum.reset() a no-op even though it
    # remains registered (so `_wbt_force_alpha` is still allocated and
    # `_advance_force_pulses` keeps multiplying by alpha = init = 1.0).
    # Re-enable by setting force_alpha_init=0.0 + alpha_step=0.01.
    assert cfg.force_alpha_init == 1.0
    assert cfg.force_alpha_max == 1.0
    assert cfg.alpha_step == 0.0
    # Threshold knobs are kept (defensive only); their relative ordering still
    # holds so re-enabling the curriculum just by flipping init/step is safe.
    assert 0 < cfg.level_down_threshold < cfg.level_up_threshold


def _eye_quat(num_envs):
    """Identity quaternion (wxyz) for num_envs."""
    q = torch.zeros(num_envs, 4)
    q[:, 0] = 1.0
    return q


def _make_env_min(num_envs=4):
    env = MagicMock()
    env.num_envs = num_envs
    env.device = torch.device("cpu")
    env.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    # _advance_force_pulses now rotates root-frame F -> world via root_quat_w,
    # so even the minimal env must expose a real identity root quaternion.
    env.simulator._robot.data = types.SimpleNamespace(root_quat_w=_eye_quat(num_envs))
    return env


def _allocate_force_buffers(env, cfg):
    """Mirror what HandForceInjector.setup() will do in Task 3.
    Tests for Task 2 use this directly so the obs functions and pulse helper
    can be unit-tested without spinning up the full injector class."""
    from wbt_training.config_values.force_tracking import _allocate_pulse_state

    n, h = env.num_envs, len(cfg.body_names)
    env._wbt_force_alpha = torch.tensor(cfg.force_alpha_init, dtype=torch.float)
    env._wbt_force_cmd_b = torch.zeros(n, h, 3)
    env._wbt_force_ext_b = torch.zeros(n, h, 3)
    # World-frame buffers (root-rotated): single source of truth for PhysX + reward.
    env._wbt_force_cmd_w = torch.zeros(n, h, 3)
    env._wbt_force_ext_w = torch.zeros(n, h, 3)
    env._wbt_force_state = _allocate_pulse_state(env, cfg)
    # Ensure an identity root quat is present (root->world rotation in advance).
    data = getattr(env.simulator._robot, "data", None)
    if not (isinstance(data, types.SimpleNamespace) and isinstance(getattr(data, "root_quat_w", None), torch.Tensor)):
        env.simulator._robot.data = types.SimpleNamespace(root_quat_w=_eye_quat(n))


def test_force_cmd_local_is_pure_reader():
    """force_cmd_local must NOT advance the pulse — it's a pure read of
    env._wbt_force_cmd_b. Buffers must be pre-allocated by HandForceInjector.setup()
    (here we allocate manually). See Math Foundation §8 single-writer rule."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_cmd_local

    env = _make_env_min(num_envs=3)
    cfg = HandForceConfig()
    _allocate_force_buffers(env, cfg)
    # Mutate the buffer directly; obs must read it back unchanged (no advance).
    env._wbt_force_cmd_b[:, 0, 0] = 7.0
    out = force_cmd_local(env, cfg=cfg)
    assert out.shape == (3, 6)
    assert out[0, 0].item() == 7.0, "obs function advanced the pulse — must be pure reader"


# ===========================================================================
# --F_on_hand_frame ablation: actor-only hand-frame F^cmd obs encoding.
# Only the OBS ENCODING of F^cmd changes (still pelvis-sampled + pelvis-applied);
# physics (F^ext injection) and the world-frame Eq.2 reward are untouched.
# ===========================================================================


def test_force_cmd_local_hand_frame_default_is_pelvis_path():
    """hand_frame defaults to False → byte-identical to the legacy pelvis path
    (returns env._wbt_force_cmd_b). Guards every existing caller/checkpoint."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_cmd_local

    env = _make_env_with_simulator(num_envs=3)
    cfg = HandForceConfig()
    _allocate_force_buffers(env, cfg)
    env._wbt_force_cmd_b[:, 0, 0] = 7.0
    env._wbt_force_cmd_w[:, 1, 1] = 9.0  # must be IGNORED on the pelvis path
    out = force_cmd_local(env, cfg=cfg)  # hand_frame omitted → False
    assert torch.equal(out, env._wbt_force_cmd_b.reshape(3, -1)), "default path must read _wbt_force_cmd_b (pelvis)"
    out2 = force_cmd_local(env, cfg=cfg, hand_frame=False)
    assert torch.equal(out2, env._wbt_force_cmd_b.reshape(3, -1))


def test_force_cmd_local_hand_frame_rotates_world_into_wrist_frame():
    """hand_frame=True → R(q_wrist_yaw)^-1 @ _wbt_force_cmd_w (world→hand), the
    SAME inverse-quat op the injector uses for F^ext. With the wrist rotated 90°
    about z, world (x,y,z) maps to hand (y,-x,z)."""
    from wbt_training.config_values.force_tracking import HandForceConfig, _quat_apply_inverse_wxyz, force_cmd_local

    cfg = HandForceConfig()
    env = _make_env_with_simulator(num_envs=2)
    _allocate_force_buffers(env, cfg)
    # Wrist bodies (ids 17, 24) rotated 90° about z (wxyz).
    s2 = (2**0.5) / 2.0
    bq = env.simulator._robot.data.body_quat_w
    bq[:, 17, 0] = s2
    bq[:, 17, 3] = s2
    bq[:, 24, 0] = s2
    bq[:, 24, 3] = s2
    # WORLD-frame command: left +x 20 N, right +y 10 N.
    env._wbt_force_cmd_w[:, 0, :] = torch.tensor([20.0, 0.0, 0.0])
    env._wbt_force_cmd_w[:, 1, :] = torch.tensor([0.0, 10.0, 0.0])

    out = force_cmd_local(env, cfg=cfg, hand_frame=True).reshape(2, 2, 3)
    # R(z90)^-1: (x,y,z) -> (y,-x,z). Left (20,0,0)->(0,-20,0); right (0,10,0)->(10,0,0).
    assert torch.allclose(out[0, 0], torch.tensor([0.0, -20.0, 0.0]), atol=1e-5), out[0, 0]
    assert torch.allclose(out[0, 1], torch.tensor([10.0, 0.0, 0.0]), atol=1e-5), out[0, 1]
    # Equals the injector's world->body-local op on the same buffer.
    expected = _quat_apply_inverse_wxyz(bq[:, [17, 24], :], env._wbt_force_cmd_w)
    assert torch.allclose(out, expected, atol=1e-6)


def test_force_cmd_local_hand_frame_uses_world_not_pelvis_buffer():
    """hand_frame must rotate the WORLD buffer (_wbt_force_cmd_w), NOT the pelvis
    buffer (_wbt_force_cmd_b) — rotating _b would double-count the pelvis rotation.
    Identity wrist quats → output equals _wbt_force_cmd_w, regardless of _b."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_cmd_local

    cfg = HandForceConfig()
    env = _make_env_with_simulator(num_envs=2)
    _allocate_force_buffers(env, cfg)  # identity wrist quats from fixture
    env._wbt_force_cmd_w[:, 0, :] = torch.tensor([3.0, 4.0, 5.0])
    env._wbt_force_cmd_b[:, 0, :] = torch.tensor([-1.0, -1.0, -1.0])  # decoy, must be ignored
    out = force_cmd_local(env, cfg=cfg, hand_frame=True).reshape(2, 2, 3)
    assert torch.allclose(out[:, 0, :], env._wbt_force_cmd_w[:, 0, :], atol=1e-6), "must read _wbt_force_cmd_w"


def test_force_cmd_local_hand_frame_zero_maps_to_zero():
    """F^cmd=0 ⇒ R^-1·0 = 0: the F=0⇒2pt collapse + idle phases hold bit-for-bit
    even with a non-identity wrist rotation."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_cmd_local

    cfg = HandForceConfig()
    env = _make_env_with_simulator(num_envs=2)
    _allocate_force_buffers(env, cfg)
    s2 = (2**0.5) / 2.0
    bq = env.simulator._robot.data.body_quat_w
    bq[:, 17, 0] = s2
    bq[:, 17, 3] = s2
    bq[:, 24, 0] = s2
    bq[:, 24, 3] = s2
    env._wbt_force_cmd_w.zero_()
    out = force_cmd_local(env, cfg=cfg, hand_frame=True)
    assert torch.count_nonzero(out) == 0


def test_force_cmd_local_hand_frame_is_pure_reader():
    """The hand-frame branch must NOT mutate any force buffer or body_quat_w
    (single-writer rule, Math Foundation §8)."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_cmd_local

    cfg = HandForceConfig()
    env = _make_env_with_simulator(num_envs=2)
    _allocate_force_buffers(env, cfg)
    env._wbt_force_cmd_w.uniform_(-5, 5)
    env.episode_length_buf += 1
    snap_w = env._wbt_force_cmd_w.clone()
    snap_b = env._wbt_force_cmd_b.clone()
    snap_q = env.simulator._robot.data.body_quat_w.clone()
    _ = force_cmd_local(env, cfg=cfg, hand_frame=True)
    assert torch.equal(env._wbt_force_cmd_w, snap_w)
    assert torch.equal(env._wbt_force_cmd_b, snap_b)
    assert torch.equal(env.simulator._robot.data.body_quat_w, snap_q)


def test_handframe_preset_flips_actor_only_critic_untouched():
    """g1_29dof_wbt_observation_2pt_force_handframe = actor force_cmd_local with
    params hand_frame=True, critic force_cmd_local untouched (strict actor-only).
    Layout, ordering, scale, noise, and the base preset are all preserved."""
    from wbt_training.config_values.force_tracking import (
        g1_29dof_wbt_observation_2pt_force as base,
        g1_29dof_wbt_observation_2pt_force_handframe as hf,
    )

    a, c = hf.groups["actor_obs"].terms, hf.groups["critic_obs"].terms
    base_a = base.groups["actor_obs"].terms
    afc = next(k for k, v in a.items() if v.func.endswith(":force_cmd_local"))
    cfc = next(k for k, v in c.items() if v.func.endswith(":force_cmd_local"))
    # Actor flipped, critic NOT.
    assert a[afc].params.get("hand_frame") is True, "actor force_cmd_local must be hand_frame"
    assert c[cfc].params.get("hand_frame", False) is False, "critic force_cmd_local must stay pelvis (actor-only)"
    # Base preset untouched (non-invasion).
    assert base_a[afc].params.get("hand_frame", None) is None, "base actor preset must NOT be mutated"
    # Layout/scale/noise preserved on the flipped term and overall ordering.
    assert a[afc].scale == base_a[afc].scale
    assert a[afc].noise == base_a[afc].noise
    assert list(a.keys()) == list(base_a.keys()), "actor term order/layout must be preserved (147D)"
    # force_ext_local is never hand-framed.
    for grp in (a, c):
        for k, v in grp.items():
            if v.func.endswith(":force_ext_local"):
                assert v.params.get("hand_frame", False) is False


def test_with_force_cmd_hand_frame_noop_on_non_force_preset():
    """The helper returns the SAME object for an obs cfg with no force_cmd_local
    actor term (so the CLI flag is a harmless no-op on non-force presets)."""
    from wbt_training.config_values.force_tracking import with_force_cmd_hand_frame
    from wbt_training.config_values.observation import g1_29dof_wbt_observation_2pt

    out = with_force_cmd_hand_frame(g1_29dof_wbt_observation_2pt)
    assert out is g1_29dof_wbt_observation_2pt, "no force_cmd_local actor term → unchanged identity"


def test_cli_flag_pop_and_apply_actor_only():
    """train_agent --F_on_hand_frame: argv pop (bare + value forms) and applying
    it to the real g1_wbt_2pt_force flips the actor term only, leaving the
    original preset and command/reward/curriculum/algo intact."""
    import sys

    import wbt_training.train_agent as ta
    from wbt_training.config_values.experiment import g1_wbt_2pt, g1_wbt_2pt_force

    # --- argv pop: bare flag ---
    saved = sys.argv
    try:
        sys.argv = ["p", "exp:g1-wbt-2pt-force", "--training.num-envs", "32768", "--F_on_hand_frame", "--x", "1"]
        assert ta._pop_f_on_hand_frame_flag() is True
        assert "--F_on_hand_frame" not in sys.argv
        assert sys.argv == ["p", "exp:g1-wbt-2pt-force", "--training.num-envs", "32768", "--x", "1"]
        # value forms
        sys.argv = ["p", "--F_on_hand_frame", "False"]
        assert ta._pop_f_on_hand_frame_flag() is False
        assert sys.argv == ["p"]
        sys.argv = ["p", "--F_on_hand_frame", "True"]
        assert ta._pop_f_on_hand_frame_flag() is True
        # absent → False, argv untouched
        sys.argv = ["p", "exp:g1-wbt"]
        assert ta._pop_f_on_hand_frame_flag() is False
        assert sys.argv == ["p", "exp:g1-wbt"]
    finally:
        sys.argv = saved

    # --- apply: off → same object ---
    assert ta._apply_f_on_hand_frame(g1_wbt_2pt_force, False) is g1_wbt_2pt_force
    # --- apply: on, force preset → actor flipped, original intact ---
    new = ta._apply_f_on_hand_frame(g1_wbt_2pt_force, True)
    assert new is not g1_wbt_2pt_force
    a = new.observation.groups["actor_obs"].terms
    c = new.observation.groups["critic_obs"].terms
    afc = next(k for k, v in a.items() if v.func.endswith(":force_cmd_local"))
    cfc = next(k for k, v in c.items() if v.func.endswith(":force_cmd_local"))
    assert a[afc].params.get("hand_frame") is True
    assert c[cfc].params.get("hand_frame", False) is False
    # Original preset NOT mutated.
    assert g1_wbt_2pt_force.observation.groups["actor_obs"].terms[afc].params.get("hand_frame", None) is None
    # Non-obs config shared (only observation rebuilt).
    assert new.command is g1_wbt_2pt_force.command
    assert new.reward is g1_wbt_2pt_force.reward
    assert new.curriculum is g1_wbt_2pt_force.curriculum
    assert new.algo is g1_wbt_2pt_force.algo
    # --- apply: on, NON-force preset → unchanged (harmless no-op) ---
    assert ta._apply_f_on_hand_frame(g1_wbt_2pt, True) is g1_wbt_2pt


def test_force_ext_local_is_pure_reader():
    from wbt_training.config_values.force_tracking import HandForceConfig, force_ext_local

    env = _make_env_min()
    cfg = HandForceConfig()
    _allocate_force_buffers(env, cfg)
    env._wbt_force_ext_b[:, 1, 2] = -3.5
    out = force_ext_local(env, cfg=cfg)
    assert out.shape == (env.num_envs, 6)
    assert out[0, 5].item() == -3.5  # right-hand z-component


def test_force_obs_functions_do_not_mutate_buffers():
    """Calling the obs functions must NEVER change env._wbt_force_*_b. If a
    future refactor makes them advance the pulse, this test fails immediately."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_cmd_local, force_ext_local

    cfg = HandForceConfig()
    env = _make_env_min()
    _allocate_force_buffers(env, cfg)
    env._wbt_force_cmd_b.uniform_(-5, 5)
    env._wbt_force_ext_b.uniform_(-5, 5)
    snap_cmd = env._wbt_force_cmd_b.clone()
    snap_ext = env._wbt_force_ext_b.clone()
    # Even with episode_length_buf "advancing", obs must not advance pulse.
    env.episode_length_buf += 1
    _ = force_cmd_local(env, cfg=cfg)
    _ = force_ext_local(env, cfg=cfg)
    assert torch.equal(env._wbt_force_cmd_b, snap_cmd)
    assert torch.equal(env._wbt_force_ext_b, snap_ext)


def test_advance_force_pulses_drives_hold_plateau():
    """Direct exercise of the pulse helper (called only by HandForceInjector
    in production). With alpha=1 and tight ramp/hold, magnitudes should reach
    f_max but never exceed."""
    from wbt_training.config_values.force_tracking import HandForceConfig, _advance_force_pulses

    cfg = HandForceConfig(force_alpha_init=1.0, ramp_steps=(2, 2), hold_steps=(4, 4), idle_steps=(0, 0))
    env = _make_env_min(num_envs=2)
    _allocate_force_buffers(env, cfg)
    env._wbt_force_alpha.fill_(1.0)
    # Drive into hold plateau.
    for _ in range(5):
        env.episode_length_buf += 1
        _advance_force_pulses(env, cfg)
    assert env._wbt_force_cmd_b.abs().max() > 1.0
    assert env._wbt_force_ext_b.abs().max() > 1.0
    # Magnitudes never exceed f_max (with idle=0 there's no auto-resample inflation).
    assert env._wbt_force_cmd_b.abs().max() <= cfg.f_cmd_max + 1e-5
    assert env._wbt_force_ext_b.abs().max() <= cfg.f_ext_max + 1e-5


def test_advance_force_pulses_idempotent_within_step():
    """Calling the helper twice with the same episode_length_buf must not
    advance state twice. This is the contract HandForceInjector relies on
    to make substeps 1..N a no-op."""
    from wbt_training.config_values.force_tracking import HandForceConfig, _advance_force_pulses

    cfg = HandForceConfig(force_alpha_init=1.0, ramp_steps=(2, 2), hold_steps=(4, 4), idle_steps=(0, 0))
    env = _make_env_min()
    _allocate_force_buffers(env, cfg)
    env._wbt_force_alpha.fill_(1.0)
    env.episode_length_buf.fill_(5)
    _advance_force_pulses(env, cfg)
    snapshot_cmd = env._wbt_force_cmd_b.clone()
    _advance_force_pulses(env, cfg)  # same step id → no-op
    assert torch.equal(env._wbt_force_cmd_b, snapshot_cmd)


def test_advance_force_pulses_populates_world_buffers():
    """Root-frame pulse must be rotated into world buffers each advance.
    With identity root_quat, world == root values; with a 90deg-about-z root
    quat, the world vector is the root vector rotated by 90deg about z."""
    from wbt_training.config_values.force_tracking import HandForceConfig, _advance_force_pulses

    cfg = HandForceConfig(force_alpha_init=1.0, ramp_steps=(2, 2), hold_steps=(4, 4), idle_steps=(0, 0))
    env = _make_env_min(num_envs=2)
    _allocate_force_buffers(env, cfg)
    env._wbt_force_alpha.fill_(1.0)
    # 90deg about z, wxyz: w=cos45, z=sin45.
    s2 = (2**0.5) / 2.0
    rq = torch.zeros(2, 4)
    rq[:, 0] = s2
    rq[:, 3] = s2
    env.simulator._robot.data.root_quat_w = rq
    for _ in range(5):
        env.episode_length_buf += 1
        _advance_force_pulses(env, cfg)
    # world = R(z90) @ root: (x, y, z) -> (-y, x, z).
    rb = env._wbt_force_cmd_b
    expected_w = torch.stack([-rb[..., 1], rb[..., 0], rb[..., 2]], dim=-1)
    assert torch.allclose(env._wbt_force_cmd_w, expected_w, atol=1e-5)
    assert torch.allclose(
        env._wbt_force_ext_w,
        torch.stack(
            [-env._wbt_force_ext_b[..., 1], env._wbt_force_ext_b[..., 0], env._wbt_force_ext_b[..., 2]], dim=-1
        ),
        atol=1e-5,
    )


def _make_env_with_simulator(num_envs=2):
    env = _make_env_min(num_envs=num_envs)
    sim = MagicMock()
    sim.find_rigid_body_indice = MagicMock(
        side_effect=lambda name: {"left_wrist_yaw_link": 17, "right_wrist_yaw_link": 24}[name]
    )
    sim.body_ids = list(range(50))
    sim._robot = MagicMock()
    sim._robot.set_external_force_and_torque = MagicMock()
    sim._robot.data = types.SimpleNamespace(
        body_quat_w=torch.zeros(num_envs, 50, 4), root_quat_w=torch.zeros(num_envs, 4)
    )
    sim._robot.data.body_quat_w[..., 0] = 1.0  # identity
    sim._robot.data.root_quat_w[..., 0] = 1.0
    env.simulator = sim
    env._apply_force_in_physics_step = MagicMock()
    return env


def _setup_injector(env, cfg):
    """Helper: instantiate + setup HandForceInjector, also pre-allocate
    _wbt_force_alpha (which is owned by ForceAlphaCurriculum.setup() in
    production but is needed for the injector's pulse advance to work)."""
    from wbt_training.config_values.force_tracking import HandForceInjector

    cfg_obj = MagicMock()
    cfg_obj.params = {"hand_force_config": cfg}
    inj = HandForceInjector(cfg_obj, env)
    inj.setup()
    # ForceAlphaCurriculum.setup() owns this in production.
    if not hasattr(env, "_wbt_force_alpha"):
        env._wbt_force_alpha = torch.tensor(cfg.force_alpha_init, dtype=torch.float)
    return inj


def test_hand_force_injector_setup_allocates_buffers_and_patches():
    from wbt_training.config_values.force_tracking import HandForceConfig

    env = _make_env_with_simulator()
    original = env._apply_force_in_physics_step
    inj = _setup_injector(env, HandForceConfig())

    # Buffers allocated by setup().
    assert hasattr(env, "_wbt_force_cmd_b") and env._wbt_force_cmd_b.shape == (env.num_envs, 2, 3)
    assert hasattr(env, "_wbt_force_ext_b") and env._wbt_force_ext_b.shape == (env.num_envs, 2, 3)
    assert hasattr(env, "_wbt_force_state") and "cmd" in env._wbt_force_state
    # Body indices resolved.
    assert inj.body_indices.tolist() == [17, 24]
    # Patched.
    assert env._apply_force_in_physics_step is not original
    assert inj._original_apply_force is original


def test_hand_force_injector_writes_zero_wrench_when_alpha_zero():
    from wbt_training.config_values.force_tracking import HandForceConfig

    env = _make_env_with_simulator()
    _setup_injector(env, HandForceConfig())  # alpha_init = 0

    env._apply_force_in_physics_step()  # triggers patched path; advances pulse but alpha=0
    assert env.simulator._robot.set_external_force_and_torque.called
    forces_arg = env.simulator._robot.set_external_force_and_torque.call_args.kwargs["forces"]
    assert torch.allclose(forces_arg, torch.zeros_like(forces_arg))


def test_hand_force_injector_writes_nonzero_wrench_when_active():
    from wbt_training.config_values.force_tracking import HandForceConfig

    cfg = HandForceConfig(force_alpha_init=1.0, ramp_steps=(2, 2), hold_steps=(4, 4), idle_steps=(0, 0))
    env = _make_env_with_simulator()
    _setup_injector(env, cfg)
    # Drive pulse to mid-hold by stepping the patched hook itself.
    for _ in range(5):
        env.episode_length_buf += 1
        env._apply_force_in_physics_step()

    forces_arg = env.simulator._robot.set_external_force_and_torque.call_args.kwargs["forces"]
    assert forces_arg.abs().max() > 1.0
    body_ids_arg = env.simulator._robot.set_external_force_and_torque.call_args.kwargs["body_ids"]
    assert sorted(body_ids_arg.tolist()) == [17, 24]


def test_inject_force_ext_converts_world_to_body_local():
    """World-frame contract: the buffer fed to PhysX is the WORLD force
    converted to body-local via quat_apply_inverse(body_quat_w, force_ext_w),
    matching holosoma push.py (IsaacLab is_global=False). With identity body
    AND root quats the injected force equals force_ext_w == force_ext_b; with a
    90deg body rotation the injected force is rotated by the inverse."""
    from wbt_training.config_values.force_tracking import HandForceConfig, _quat_apply_inverse_wxyz

    cfg = HandForceConfig(force_alpha_init=1.0, ramp_steps=(2, 2), hold_steps=(4, 4), idle_steps=(0, 0))
    env = _make_env_with_simulator()
    # Non-identity body quats on the two wrist bodies (90deg about z, wxyz).
    s2 = (2**0.5) / 2.0
    bq = env.simulator._robot.data.body_quat_w
    bq[:, 17, 0] = s2
    bq[:, 17, 3] = s2
    bq[:, 24, 0] = s2
    bq[:, 24, 3] = s2
    _setup_injector(env, cfg)
    for _ in range(5):
        env.episode_length_buf += 1
        env._apply_force_in_physics_step()

    forces_arg = env.simulator._robot.set_external_force_and_torque.call_args.kwargs["forces"]
    body_quat = env.simulator._robot.data.body_quat_w[:, [17, 24], :]
    expected = _quat_apply_inverse_wxyz(body_quat, env._wbt_force_ext_w)
    assert forces_arg.shape == env._wbt_force_ext_w.shape
    assert torch.allclose(forces_arg, expected, atol=1e-5), (
        "forces must be world F_ext converted to body-local via "
        "quat_apply_inverse(body_quat_w, force_ext_w), like push.py."
    )


def test_hand_force_injector_reset_restarts_pulse_per_env():
    """reset(env_ids) must restart the force pulse for the reset envs (GH
    `_force_track.sample_init` parity, motion_tracking.py:1412-1434): zero the
    root- AND world-frame force buffers for those envs, resample their pulse
    phase, and clear their idempotency marker so the next advance re-fires.

    Without this, an env that resets mid-hold-plateau begins its NEW episode
    with a held ~25 N already on the hand while the robot has just been snapped
    back to its reference pose — a start-of-episode force shock that does not
    exist in GH. Other (non-reset) envs must be left untouched.
    """
    from wbt_training.config_values.force_tracking import HandForceConfig

    # Long hold so every env is mid-plateau (non-zero force) when we reset.
    cfg = HandForceConfig(force_alpha_init=1.0, ramp_steps=(2, 2), hold_steps=(50, 50), idle_steps=(0, 0))
    env = _make_env_with_simulator(num_envs=4)
    inj = _setup_injector(env, cfg)

    # Drive all envs into the hold plateau (non-zero F^cmd and F^ext).
    for _ in range(5):
        env.episode_length_buf += 1
        env._apply_force_in_physics_step()
    assert env._wbt_force_cmd_b.abs().max() > 1.0
    assert env._wbt_force_ext_b.abs().max() > 1.0

    # Snapshot the envs we are NOT resetting (must be preserved).
    keep = [1, 3]
    keep_cmd = env._wbt_force_cmd_b[keep].clone()
    keep_ext = env._wbt_force_ext_b[keep].clone()
    keep_t_cmd = env._wbt_force_state["cmd"]["t"][keep].clone()

    # Reset envs 0 and 2 (mirrors command_manager.reset(env_ids)).
    reset_ids = torch.tensor([0, 2], dtype=torch.long)
    inj.reset(reset_ids)

    # Reset envs: force buffers (root AND world) zeroed.
    for buf in ("_wbt_force_cmd_b", "_wbt_force_ext_b", "_wbt_force_cmd_w", "_wbt_force_ext_w"):
        b = getattr(env, buf)
        assert torch.equal(b[reset_ids], torch.zeros_like(b[reset_ids])), f"{buf} not zeroed on reset"
    # Reset envs: pulse phase restarted (t back to 0) and idempotency marker cleared.
    assert torch.equal(
        env._wbt_force_state["cmd"]["t"][reset_ids], torch.zeros_like(env._wbt_force_state["cmd"]["t"][reset_ids])
    )
    assert torch.equal(
        env._wbt_force_state["ext"]["t"][reset_ids], torch.zeros_like(env._wbt_force_state["ext"]["t"][reset_ids])
    )
    assert (env._wbt_force_state["last_step"][reset_ids] == -1).all(), "last_step not cleared on reset"

    # Non-reset envs: untouched.
    assert torch.equal(env._wbt_force_cmd_b[keep], keep_cmd), "reset leaked into non-reset envs"
    assert torch.equal(env._wbt_force_ext_b[keep], keep_ext), "reset leaked into non-reset envs"
    assert torch.equal(env._wbt_force_state["cmd"]["t"][keep], keep_t_cmd)


def test_hand_force_injector_reset_then_advance_starts_from_ramp():
    """After reset, the very next advance for a reset env must start the pulse
    near zero (ramp-up), not resume the previous hold plateau — i.e. the reset
    actually re-phases the schedule rather than just zeroing one frame."""
    from wbt_training.config_values.force_tracking import HandForceConfig, _advance_force_pulses

    cfg = HandForceConfig(force_alpha_init=1.0, ramp_steps=(10, 10), hold_steps=(50, 50), idle_steps=(0, 0))
    env = _make_env_with_simulator(num_envs=2)
    inj = _setup_injector(env, cfg)

    for _ in range(15):  # well into the hold plateau
        env.episode_length_buf += 1
        env._apply_force_in_physics_step()
    plateau_mag = env._wbt_force_cmd_b[0].abs().max().item()
    assert plateau_mag > 1.0

    # Reset env 0; its episode_length_buf restarts at 0 (mirrors wbt_manager).
    inj.reset(torch.tensor([0], dtype=torch.long))
    env.episode_length_buf[0] = 0
    _advance_force_pulses(env, cfg)  # first advance of the new episode

    # Env 0's force at the first ramp step is far below the plateau it left.
    assert env._wbt_force_cmd_b[0].abs().max().item() < plateau_mag, (
        "reset env resumed the old hold plateau instead of restarting the ramp"
    )


def test_obs_and_reward_never_advance_pulse_after_episode_buf_increment():
    """Critical anti-regression: episode_length_buf increments BEFORE reward/obs
    in _post_physics_step. If a future refactor makes obs/reward call
    _advance_force_pulses, they'll see a new step id and advance to F(k+1) while
    cur_pos_w was integrated with F(k). This test catches that desync at unit level."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_cmd_local, force_ext_local

    cfg = HandForceConfig(force_alpha_init=1.0, ramp_steps=(2, 2), hold_steps=(4, 4), idle_steps=(0, 0))
    env = _make_env_with_simulator()
    _setup_injector(env, cfg)
    # One full env-step's substeps drive pulse to F(k=1).
    env.episode_length_buf += 1
    for _ in range(4):
        env._apply_force_in_physics_step()
    snap_cmd = env._wbt_force_cmd_b.clone()
    snap_ext = env._wbt_force_ext_b.clone()

    # Now simulate _post_physics_step: increment buf, then run obs/reward reads.
    env.episode_length_buf += 1  # k → k+1, mirrors base_task.py:429
    _ = force_cmd_local(env, cfg=cfg)
    _ = force_ext_local(env, cfg=cfg)
    # Buffers MUST still hold F(k); obs/reward did NOT advance to F(k+1).
    assert torch.equal(env._wbt_force_cmd_b, snap_cmd), "obs advanced pulse — desync bug"
    assert torch.equal(env._wbt_force_ext_b, snap_ext), "obs advanced pulse — desync bug"


def _make_env_with_motion_command(num_envs=2, cfg=None):
    """Build a fully-allocated env: simulator + motion_command + force buffers.
    Force buffers are pre-allocated (mirrors HandForceInjector.setup()) so
    reward functions can be tested as pure readers without driving physics."""
    from wbt_training.config_values.force_tracking import HandForceConfig

    cfg = cfg or HandForceConfig()
    env = _make_env_with_simulator(num_envs=num_envs)
    _allocate_force_buffers(env, cfg)
    mc = MagicMock()
    body_names_list = ["pelvis"] + [f"x{i}" for i in range(11)] + ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    # Real MotionCommand exposes the body-name list at instance.motion_cfg
    # (NOT instance.cfg.motion_config — verified at wbt.py:516-521).
    mc.motion_cfg = types.SimpleNamespace(body_names_to_track=body_names_list)
    # `_force_ref_pos_w` reads `body_pos_relative_w` / `body_quat_relative_w`
    # (yaw-relocated to robot's current root xy + yaw, with z absolute) — see
    # the function docstring. Tests set the relative_w properties directly so
    # we exercise the same field the production code reads.
    mc.body_pos_relative_w = torch.zeros(num_envs, 14, 3)
    mc.body_quat_relative_w = torch.zeros(num_envs, 14, 4)
    # `body_quat_relative_w` is xyzw (wbt.py:812 uses `quat_mul(..., w_last=True)`
    # which preserves the xyzw convention from `body_quat_w`). Identity
    # quaternion in xyzw is (0, 0, 0, 1).
    mc.body_quat_relative_w[..., 3] = 1.0  # identity (xyzw)
    # Keep the raw `body_pos_w` / `body_quat_w` defined too — some tests still
    # poke them as a sanity reference, and downstream baseline reward terms
    # (not under test here) read other properties on the same MagicMock.
    mc.body_pos_w = torch.zeros(num_envs, 14, 3)
    mc.body_quat_w = torch.zeros(num_envs, 14, 4)
    mc.body_quat_w[..., 3] = 1.0
    cm = MagicMock()
    cm.get_state = MagicMock(side_effect=lambda name: mc if name == "motion_command" else None)
    env.command_manager = cm
    return env, mc


def test_force_track_pos_eq2_reward_one_when_hand_at_target():
    """When cur = ref + (F^cmd + F^ext)/K, err_l1 = 0 → r = exp(0) = 1."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_track_pos_eq2

    cfg = HandForceConfig()
    env, mc = _make_env_with_motion_command(cfg=cfg)
    env._wbt_force_alpha.fill_(0.0)
    # motion_ref body_pos_w defaults to zeros → ref = (0, 0, 0); with alpha=0,
    # target = ref. Place wrist_yaw at the origin too → err_l1 = 0 → r = 1.
    env.simulator._robot.data.body_pos_w = torch.zeros(env.num_envs, 50, 3)

    r = force_track_pos_eq2(env, cfg=cfg)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-5)


def test_force_track_pos_eq2_target_shifts_by_F_over_K():
    """World-frame F^cmd_w = (20, 0, 0) on left hand -> target shift 20/200 =
    0.10 m forward in world. Hand placed at the shifted target -> reward 1."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_track_pos_eq2

    cfg = HandForceConfig(stiffness_K=200.0)
    env, mc = _make_env_with_motion_command(cfg=cfg)
    env._wbt_force_alpha.fill_(1.0)
    env._wbt_force_cmd_w.zero_()
    env._wbt_force_ext_w.zero_()
    env._wbt_force_cmd_w[:, 0, 0] = 20.0  # world-frame 20 N +x on left hand
    # Live wrist quats are identity (fixture) -> offset cancels; place the live
    # left wrist 0.10 m forward of its motion ref (both at origin), right at ref.
    bp = torch.zeros(env.num_envs, 50, 3)
    bp[:, 17, :] = torch.tensor([0.10, 0.0, 0.0])
    bp[:, 24, :] = torch.tensor([0.0, 0.0, 0.0])
    env.simulator._robot.data.body_pos_w = bp

    r = force_track_pos_eq2(env, cfg=cfg)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-4)


def test_force_track_pos_eq2_l1_norm_and_2_over_sigma_scale():
    """Each hand 0.1 m off in x with no force → err_l1 = 0.2 (sum over both hands & xyz),
    r = exp(-0.2 · 2 / 0.3) = exp(-1.333) ≈ 0.264. Catches L2/σ² drift."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_track_pos_eq2

    cfg = HandForceConfig()
    env, mc = _make_env_with_motion_command(cfg=cfg)
    env._wbt_force_alpha.fill_(0.0)
    # Both wrists 0.1 m off in +x from the ref at origin.
    bp = torch.zeros(env.num_envs, 50, 3)
    bp[:, 17, :] = torch.tensor([0.10, 0.0, 0.0])
    bp[:, 24, :] = torch.tensor([0.10, 0.0, 0.0])
    env.simulator._robot.data.body_pos_w = bp

    r = force_track_pos_eq2(env, cfg=cfg)
    expected = torch.exp(torch.tensor(-0.2 * 2.0 / 0.3))
    assert torch.allclose(r, torch.full_like(r, expected.item()), atol=1e-4), (
        f"got {r[0].item():.4f}, expected {expected.item():.4f}. If you got ~0.801, you used L2/σ² instead of L1·2/σ."
    )


def test_force_track_pos_eq2_offset_cancels_under_wrist_rotation():
    """FK-offset bug fix: the rubber_hand offset must be rotated by the SAME
    live wrist quaternion on the cur and ref sides, so it cancels in the
    residual regardless of wrist orientation. With NO force and the live wrist
    rotated 90deg about z, cur==ref -> reward 1. The old code rotated the offset
    by the motion quat on the ref side and the live quat on the cur side, so a
    rotated wrist injected ~2 cm phantom error and the reward dropped below 1."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_track_pos_eq2

    cfg = HandForceConfig(stiffness_K=200.0)
    env, mc = _make_env_with_motion_command(cfg=cfg)
    env._wbt_force_alpha.fill_(0.0)
    env._wbt_force_cmd_w.zero_()
    env._wbt_force_ext_w.zero_()
    # Live wrist quats rotated 90deg about z (wxyz). Motion ref quats stay
    # identity (fixture), which is exactly the cur != ref-quat condition the old
    # code mishandled.
    s2 = (2**0.5) / 2.0
    bq = torch.zeros(env.num_envs, 50, 4)
    bq[..., 0] = 1.0
    bq[:, 17, 0] = s2
    bq[:, 17, 3] = s2
    bq[:, 24, 0] = s2
    bq[:, 24, 3] = s2
    env.simulator._robot.data.body_quat_w = bq
    # Live wrist positions equal the motion-ref wrist positions (both at origin),
    # so cur_rubber == ref_rubber once the offset cancels.
    env.simulator._robot.data.body_pos_w = torch.zeros(env.num_envs, 50, 3)

    r = force_track_pos_eq2(env, cfg=cfg)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-4), (
        f"got {r[0].item():.4f}; if < 1 the offset is not cancelling — ref and "
        f"cur are rotating the rubber_hand offset by different quaternions."
    )


def test_force_track_pos_eq2_uses_sum_not_mean():
    """Hand-L aligned, hand-R 0.2 m off → err_l1 = 0.2 (sum, not mean to 0.1).
    r = exp(-0.2 · 2 / 0.3) ≈ 0.264. Catches mean(exp(per-hand)) drift."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_track_pos_eq2

    cfg = HandForceConfig()
    env, mc = _make_env_with_motion_command(cfg=cfg)
    env._wbt_force_alpha.fill_(0.0)
    # Left aligned with ref (origin); right 0.2 m off in +x.
    bp = torch.zeros(env.num_envs, 50, 3)
    bp[:, 17, :] = torch.tensor([0.0, 0.0, 0.0])
    bp[:, 24, :] = torch.tensor([0.20, 0.0, 0.0])
    env.simulator._robot.data.body_pos_w = bp

    r = force_track_pos_eq2(env, cfg=cfg)
    expected = torch.exp(torch.tensor(-0.2 * 2.0 / 0.3))
    assert torch.allclose(r, torch.full_like(r, expected.item()), atol=1e-4), (
        f"got {r[0].item():.4f}, expected {expected.item():.4f}. "
        f"If you got ~0.632, the reduction is mean(exp(...)) — must be exp(sum(...))."
    )


def test_force_track_pos_pen_returns_positive_indicator():
    """Penalty term returns POSITIVE [0, 1] (graded mean over hands). Sign comes
    from the RewardTermCfg weight (default -6.0). Returning negative AND wiring
    weight=-6.0 would double-negate to positive reward — a serious bug."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_track_pos_pen

    cfg = HandForceConfig()
    env, _ = _make_env_with_motion_command(cfg=cfg)

    # Both wrists 0.7 m off ref → drift > 0.6 → r = 1.0
    bp = torch.zeros(env.num_envs, 50, 3)
    bp[:, 17, :] = torch.tensor([0.7, 0.0, 0.0])
    bp[:, 24, :] = torch.tensor([0.7, 0.0, 0.0])
    env.simulator._robot.data.body_pos_w = bp
    r = force_track_pos_pen(env, cfg=cfg)
    assert torch.allclose(r, torch.ones(env.num_envs))

    # Both wrists 0.4 m off ref → drift < 0.6 → r = 0
    bp[:, 17, :] = torch.tensor([0.4, 0.0, 0.0])
    bp[:, 24, :] = torch.tensor([0.4, 0.0, 0.0])
    env.simulator._robot.data.body_pos_w = bp
    r = force_track_pos_pen(env, cfg=cfg)
    assert torch.allclose(r, torch.zeros(env.num_envs))


def test_force_track_pos_pen_graded_mean_over_hands():
    """One hand bad → r = 0.5 (mean across [bad=1, ok=0]); both hands bad → r = 1.0."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_track_pos_pen

    cfg = HandForceConfig()
    env, _ = _make_env_with_motion_command(cfg=cfg)
    bp = torch.zeros(env.num_envs, 50, 3)
    bp[:, 17, :] = torch.tensor([0.7, 0.0, 0.0])  # left bad
    bp[:, 24, :] = torch.tensor([0.0, 0.0, 0.0])  # right ok
    env.simulator._robot.data.body_pos_w = bp
    r = force_track_pos_pen(env, cfg=cfg)
    assert torch.allclose(r, torch.full_like(r, 0.5))


def test_force_track_pos_pen_uses_raw_drift_not_eq2_residual():
    """Penalty must read ‖cur - ref‖, NOT ‖cur - (ref + F/K)‖.
    With strong force and hand at the shifted target, raw drift > 0.6 should still trigger r=1."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_track_pos_pen

    cfg = HandForceConfig(stiffness_K=200.0)
    env, _ = _make_env_with_motion_command(cfg=cfg)
    env._wbt_force_alpha.fill_(1.0)
    env._wbt_force_cmd_b.zero_()
    env._wbt_force_ext_b.zero_()
    env._wbt_force_cmd_b[:, :, 0] = 140.0
    # Hands at the shifted Eq.2 target (140/200 = 0.7 m). Eq.2 residual ≈ 0,
    # but raw drift = 0.7 > 0.6 → penalty must still trigger.
    bp = torch.zeros(env.num_envs, 50, 3)
    bp[:, 17, :] = torch.tensor([0.70, 0.0, 0.0])
    bp[:, 24, :] = torch.tensor([0.70, 0.0, 0.0])
    env.simulator._robot.data.body_pos_w = bp

    r = force_track_pos_pen(env, cfg=cfg)
    assert torch.allclose(r, torch.ones(env.num_envs)), "Penalty must trigger on raw drift, not on Eq.2 residual."


def test_inject_force_ext_passes_urdf_offset_to_physx():
    """Selection-D contract: PhysX wrench is applied at the rubber_hand offset
    within the wrist_yaw local frame, not at the wrist_yaw COM. This is
    rigid-body-equivalent to applying force on the rubber_hand link itself.
    Verifies `positions=` kwarg is set with the URDF offsets."""
    from wbt_training.config_values.force_tracking import HandForceConfig

    cfg = HandForceConfig(force_alpha_init=1.0, ramp_steps=(2, 2), hold_steps=(4, 4), idle_steps=(0, 0))
    env = _make_env_with_simulator()
    _setup_injector(env, cfg)
    # Drive into hold so wrench is non-zero.
    for _ in range(5):
        env.episode_length_buf += 1
        env._apply_force_in_physics_step()

    call_kwargs = env.simulator._robot.set_external_force_and_torque.call_args.kwargs
    assert "positions" in call_kwargs, "positions= kwarg missing — wrench is at wrist_yaw COM"
    positions = call_kwargs["positions"]
    # Expected yam URDF offsets (wrist→ft_sensor 0.072 + palm 0.0095): (0.0815, +0.003, 0) and (0.0815, -0.003, 0).
    expected = torch.tensor([[[0.0815, +0.003, 0.0], [0.0815, -0.003, 0.0]]] * env.num_envs)
    assert torch.allclose(positions, expected, atol=1e-6), (
        f"positions don't match URDF offsets — got {positions[0]}, expected {expected[0]}"
    )


def test_force_ref_pos_w_rotates_offset_by_live_wrist_quat():
    """FK contract (offset-cancel fix): ref_pos_w = motion_wrist_pos_relative_w
    + R(live_wrist_quat_wxyz) @ offset_b. The offset is rotated by the LIVE wrist
    quaternion (not the motion quaternion) so it cancels against _force_cur_pos_w,
    which rotates the same offset by the same live quaternion."""
    from wbt_training.config_values.force_tracking import HandForceConfig, _force_ref_pos_w

    cfg = HandForceConfig()
    env, mc = _make_env_with_motion_command(cfg=cfg)
    # Motion-ref wrist positions (yaw-relocated), non-trivial.
    mc.body_pos_relative_w[:, 12, :] = torch.tensor([1.0, 2.0, 3.0])  # left_wrist_yaw
    mc.body_pos_relative_w[:, 13, :] = torch.tensor([4.0, 5.0, 6.0])  # right_wrist_yaw
    # Identity live wrist quats (fixture _robot.data.body_quat_w) -> R @ offset = offset.
    ref_w = _force_ref_pos_w(env, cfg)
    assert ref_w.shape == (env.num_envs, 2, 3)
    expected_left = torch.tensor([1.0 + 0.0815, 2.0 + 0.003, 3.0])
    expected_right = torch.tensor([4.0 + 0.0815, 5.0 - 0.003, 6.0])
    assert torch.allclose(ref_w[0, 0], expected_left, atol=1e-5)
    assert torch.allclose(ref_w[0, 1], expected_right, atol=1e-5)

    # Now rotate the live wrist 90deg about z (wxyz): offset (0.0815, +-0.003, 0)
    # rotates to (-+0.003, 0.0815, 0). Position anchor (body_pos_relative_w) is
    # unchanged because only the OFFSET rotation source changed.
    s2 = (2**0.5) / 2.0
    bq = torch.zeros(env.num_envs, 50, 4)
    bq[..., 0] = 1.0
    bq[:, 17, 0] = s2
    bq[:, 17, 3] = s2
    bq[:, 24, 0] = s2
    bq[:, 24, 3] = s2
    env.simulator._robot.data.body_quat_w = bq
    ref_w2 = _force_ref_pos_w(env, cfg)
    # left offset (0.0815, +0.003, 0) -> R(z90) -> (-0.003, 0.0815, 0)
    expected_left2 = torch.tensor([1.0 - 0.003, 2.0 + 0.0815, 3.0])
    # right offset (0.0815, -0.003, 0) -> R(z90) -> (0.003, 0.0815, 0)
    expected_right2 = torch.tensor([4.0 + 0.003, 5.0 + 0.0815, 6.0])
    assert torch.allclose(ref_w2[0, 0], expected_left2, atol=1e-5)
    assert torch.allclose(ref_w2[0, 1], expected_right2, atol=1e-5)


def test_force_cur_pos_w_applies_fk_with_wxyz_quat():
    """Selection-D FK contract: cur_pos_w = wrist_yaw_live_pos + R(wxyz_quat) @ offset_b.
    _robot.data.body_quat_w is wxyz (IsaacSim native), so the helper applies wxyz directly."""
    from wbt_training.config_values.force_tracking import HandForceConfig, _force_cur_pos_w

    cfg = HandForceConfig()
    env, _ = _make_env_with_motion_command(cfg=cfg)
    # _robot.data.body_quat_w defaults to identity wxyz (1, 0, 0, 0) from fixture.
    bp = torch.zeros(env.num_envs, 50, 3)
    bp[:, 17, :] = torch.tensor([1.0, 2.0, 3.0])  # live left_wrist_yaw
    bp[:, 24, :] = torch.tensor([4.0, 5.0, 6.0])  # live right_wrist_yaw
    env.simulator._robot.data.body_pos_w = bp

    cur_w = _force_cur_pos_w(env, cfg)
    assert cur_w.shape == (env.num_envs, 2, 3)
    expected_left = torch.tensor([1.0 + 0.0815, 2.0 + 0.003, 3.0])
    expected_right = torch.tensor([4.0 + 0.0815, 5.0 - 0.003, 6.0])
    assert torch.allclose(cur_w[0, 0], expected_left, atol=1e-5)
    assert torch.allclose(cur_w[0, 1], expected_right, atol=1e-5)


def test_force_ref_pos_w_uses_relative_not_raw_motion_world_pose():
    """Yaw-mismatch regression: when raw body_pos_w (motion file absolute frame)
    diverges from body_pos_relative_w (yaw-relocated to robot's current root),
    the reward must read body_pos_relative_w. This locks in the parity with
    GH `reward_keypoints_w` (motion_tracking.py:443-446) and with holosoma's
    own baseline `motion_relative_body_position_error_exp`. See plan §F frame
    audit for the rationale.

    If a refactor accidentally reverts to raw body_pos_w, the residual will
    pick up the diverging absolute position and this test will catch it."""
    from wbt_training.config_values.force_tracking import HandForceConfig, _force_ref_pos_w

    cfg = HandForceConfig()
    env, mc = _make_env_with_motion_command(cfg=cfg)
    # Raw body_pos_w (motion-file absolute, NOT yaw-aligned to robot): place
    # wrists somewhere far away. If the helper still reads this, the test
    # asserts will fail with these large numbers.
    mc.body_pos_w[:, 12, :] = torch.tensor([100.0, 100.0, 100.0])
    mc.body_pos_w[:, 13, :] = torch.tensor([100.0, 100.0, 100.0])
    # Yaw-relocated reference (the field production reads): identity quat,
    # known wrist positions.
    mc.body_pos_relative_w[:, 12, :] = torch.tensor([1.0, 2.0, 3.0])
    mc.body_pos_relative_w[:, 13, :] = torch.tensor([4.0, 5.0, 6.0])

    ref_w = _force_ref_pos_w(env, cfg)
    expected_left = torch.tensor([1.0 + 0.0815, 2.0 + 0.003, 3.0])
    expected_right = torch.tensor([4.0 + 0.0815, 5.0 - 0.003, 6.0])
    assert torch.allclose(ref_w[0, 0], expected_left, atol=1e-5), (
        f"ref_w left = {ref_w[0, 0]} — if it's near (100, 100, 100), the helper "
        f"regressed to reading raw body_pos_w (the absolute motion frame)."
    )
    assert torch.allclose(ref_w[0, 1], expected_right, atol=1e-5)


def test_force_alpha_curriculum_disabled_by_default():
    """With the default config the curriculum is disabled — alpha starts at 1.0
    (full force from training step 0) and reset() is a no-op because
    alpha_step=0.0. Re-enable by passing force_alpha_init=0.0 + alpha_step=0.01,
    which is what the per-test cases below exercise."""
    from wbt_training.config_values.force_tracking import HandForceConfig, ForceAlphaCurriculum

    env = _make_env_min()
    env.curriculum_manager = None  # take the test fallback path
    env.average_episode_length = torch.tensor(0.0)
    env.log_dict = {}
    cfg_obj = MagicMock()
    cfg_obj.params = {"hand_force_config": HandForceConfig()}
    cur = ForceAlphaCurriculum(cfg_obj, env)
    cur.setup()
    assert float(env._wbt_force_alpha) == 1.0
    # reset() must be a no-op when alpha_step is zero, even with an
    # average_episode_length far below level_down_threshold.
    cur.reset(env_ids=torch.arange(env.num_envs))
    assert float(env._wbt_force_alpha) == 1.0


def test_force_alpha_ramps_up_when_episodes_long():
    from wbt_training.config_values.force_tracking import HandForceConfig, ForceAlphaCurriculum

    cfg = HandForceConfig(level_up_threshold=500.0, alpha_step=0.1, force_alpha_max=1.0)
    env = _make_env_min()
    # Force the unit-test fallback (production path is covered separately).
    env.curriculum_manager = None
    env.average_episode_length = torch.tensor(800.0)  # well past level_up
    env.log_dict = {}
    cfg_obj = MagicMock()
    cfg_obj.params = {"hand_force_config": cfg}
    cur = ForceAlphaCurriculum(cfg_obj, env)
    cur.setup()
    for _ in range(15):
        cur.reset(env_ids=torch.arange(env.num_envs))
    assert float(env._wbt_force_alpha) == 1.0  # clamped at max
    assert "force_alpha" in env.log_dict


def test_force_alpha_ramps_when_avg_is_float_property():
    """Test the unit-test fallback path: env.average_episode_length set
    directly as a Tensor or float (MagicMock convenience). This is NOT the
    production path — see test_force_alpha_reads_from_curriculum_tracker
    below for the real WBT-shaped scenario."""
    from wbt_training.config_values.force_tracking import HandForceConfig, ForceAlphaCurriculum

    cfg = HandForceConfig(level_up_threshold=300.0, alpha_step=0.1, force_alpha_max=1.0)
    env = _make_env_min()
    # Disable the production curriculum_manager path so the fallback fires.
    env.curriculum_manager = None
    env.average_episode_length = 500.0  # plain float
    env.log_dict = {}
    cfg_obj = MagicMock()
    cfg_obj.params = {"hand_force_config": cfg}
    cur = ForceAlphaCurriculum(cfg_obj, env)
    cur.setup()
    cur.reset(env_ids=torch.arange(env.num_envs))
    assert float(env._wbt_force_alpha) > 0.0, (
        "alpha must ramp when env.average_episode_length is a plain float (unit-test fallback path)."
    )
    assert "force_alpha" in env.log_dict, "log_dict must be written for W&B"


def test_force_alpha_reads_from_curriculum_tracker():
    """Production-shape regression test. WholeBodyTrackingManager(BaseTask)
    does NOT inherit from LocomotionManager and has no
    `average_episode_length` property — only the
    AverageEpisodeLengthTracker registered on curriculum_manager. The
    curriculum MUST read from that tracker, not from a missing attribute on
    env. Earlier code did `getattr(env, 'average_episode_length', None)` →
    `None` in production → curriculum silently no-op'd, which is exactly
    the bug observed in W&B run hhzucnne (no force_alpha key in 60 logged
    metrics, despite avg_ep_len climbing to 213).
    """
    from wbt_training.config_values.force_tracking import HandForceConfig, ForceAlphaCurriculum

    cfg = HandForceConfig(level_up_threshold=300.0, alpha_step=0.1, force_alpha_max=1.0)
    env = _make_env_min()
    # Mimic production: NO env.average_episode_length attribute.
    # Use a real spec to disable MagicMock auto-attribute behavior.
    if hasattr(env, "average_episode_length"):
        del env.average_episode_length
    env.log_dict = {}
    # Production-shape: curriculum_manager.get_term("average_episode_tracker")
    # returns a tracker with `.get_average()` returning a 0-dim Tensor.
    tracker = MagicMock()
    tracker.get_average = MagicMock(return_value=torch.tensor(500.0))
    env.curriculum_manager = MagicMock()
    env.curriculum_manager.get_term = MagicMock(
        side_effect=lambda name: tracker if name == "average_episode_tracker" else None
    )

    cfg_obj = MagicMock()
    cfg_obj.params = {"hand_force_config": cfg}
    cur = ForceAlphaCurriculum(cfg_obj, env)
    cur.setup()
    cur.reset(env_ids=torch.arange(env.num_envs))

    # Verify the tracker was actually read (proves the production path).
    assert tracker.get_average.called, (
        "Curriculum did NOT read from curriculum_manager.get_term tracker — "
        "it must in production where env has no .average_episode_length attr."
    )
    assert float(env._wbt_force_alpha) > 0.0, (
        "Alpha must ramp when tracker reports avg=500 > level_up_threshold=300. "
        "If alpha stayed at 0, the curriculum is no-op'd in production."
    )
    assert "force_alpha" in env.log_dict, "log_dict must be written for W&B"


def test_force_alpha_ramps_down_when_episodes_short():
    from wbt_training.config_values.force_tracking import HandForceConfig, ForceAlphaCurriculum

    cfg = HandForceConfig(force_alpha_init=0.5, level_down_threshold=200.0, alpha_step=0.1)
    env = _make_env_min()
    # Force the unit-test fallback (production path is covered separately).
    env.curriculum_manager = None
    env.average_episode_length = torch.tensor(50.0)  # well below level_down
    env.log_dict = {}
    cfg_obj = MagicMock()
    cfg_obj.params = {"hand_force_config": cfg}
    cur = ForceAlphaCurriculum(cfg_obj, env)
    cur.setup()
    for _ in range(15):
        cur.reset(env_ids=torch.arange(env.num_envs))
    assert float(env._wbt_force_alpha) == 0.0  # clamped at zero


def test_g1_wbt_2pt_force_registered_in_defaults():
    from wbt_training.config_values.experiment import DEFAULTS, g1_wbt_2pt, g1_wbt_2pt_force

    assert "g1_wbt_2pt_force" in DEFAULTS
    assert DEFAULTS["g1_wbt_2pt_force"] is g1_wbt_2pt_force
    # Inherits algo from g1_wbt_2pt.
    assert g1_wbt_2pt_force.algo.config.num_learning_iterations == g1_wbt_2pt.algo.config.num_learning_iterations
    # Has the new force-side cfg objects wired in.
    assert "hand_force_injector" in g1_wbt_2pt_force.command.setup_terms
    # REGRESSION GUARD: the injector MUST also be in reset_terms, otherwise
    # CommandManager never dispatches HandForceInjector.reset() and the
    # trapezoidal pulse + F^ext buffer leak across episode boundaries (the
    # start-of-episode force-shock bug). setup_terms alone gives the term a
    # `stages={"setup"}` set, and CommandManager.reset() skips it. See the
    # comment on g1_29dof_wbt_force_command in force_tracking.py.
    assert "hand_force_injector" in g1_wbt_2pt_force.command.reset_terms
    # motion_command must remain wired in reset_terms (not clobbered by the
    # injector addition).
    assert "motion_command" in g1_wbt_2pt_force.command.reset_terms
    assert "force_alpha" in g1_wbt_2pt_force.curriculum.setup_terms
    assert "force_track_pos_eq2" in g1_wbt_2pt_force.reward.terms
    # `force_track_pos_pen` is intentionally NOT wired in this preset (see the
    # comment on `g1_29dof_wbt_2pt_force_reward`). The function still exists in
    # force_tracking.py and has its own unit tests so it can be re-enabled by
    # adding one entry to the preset.
    assert "force_track_pos_pen" not in g1_wbt_2pt_force.reward.terms


def test_force_obs_terms_use_sequential_keys():
    from wbt_training.config_values.force_tracking import g1_29dof_wbt_observation_2pt_force

    actor_keys = list(g1_29dof_wbt_observation_2pt_force.groups["actor_obs"].terms.keys())
    # Sequential numeric prefix, no obs3a/obs3b splicing.
    import re

    assert all(re.match(r"^obs\d+_", k) for k in actor_keys), actor_keys
    # Actor sees F^cmd only (UniFP asymmetry — see motion_tracking.py:1655).
    assert any("force_cmd_local" in k for k in actor_keys)
    assert not any("force_ext_local" in k for k in actor_keys), (
        "force_ext_local must NOT be actor-visible — UniFP requires the actor to "
        "infer F^ext from proprioception (see motion_tracking.py:1626-1657)."
    )


def test_force_critic_obs_includes_both_forces():
    """UniFP critic privilege: actor sees F^cmd; critic sees F^cmd + F^ext.
    Mirrors gentle_humanoid `force_track_priv` at motion_tracking.py:1701-1707
    where last_force_ext_b AND last_force_cmd_b are concatenated for the critic."""
    from wbt_training.config_values.force_tracking import g1_29dof_wbt_observation_2pt_force

    critic_keys = list(g1_29dof_wbt_observation_2pt_force.groups["critic_obs"].terms.keys())
    assert "force_cmd_local" in critic_keys, "critic must see F^cmd (UniFP asymmetry)"
    assert "force_ext_local" in critic_keys, "critic must see F^ext (UniFP privilege)"


def test_tyro_alias_resolves():
    from wbt_training.config_values.experiment import DEFAULTS

    aliases = {f"exp:{name.replace('_', '-')}" for name in DEFAULTS}
    assert "exp:g1-wbt-2pt-force" in aliases


# ===========================================================================
# exp:g1-wbt-2pt-force-ext — F^ext-in-actor, first-commit reward recipe.
# This preset is a deliberate REVERT of g1_wbt_2pt_force to its original design
# (F^ext back in the actor + full-body tracking + Eq.2@4.0 + pen@-6.0), so these
# tests assert exactly the ways it must DIFFER from the shipped g1_wbt_2pt_force.
# ===========================================================================


def test_g1_wbt_2pt_force_ext_registered_and_yam_urdf():
    from wbt_training.config_values.experiment import DEFAULTS, g1_wbt_2pt_force_ext

    assert "g1_wbt_2pt_force_ext" in DEFAULTS
    assert DEFAULTS["g1_wbt_2pt_force_ext"] is g1_wbt_2pt_force_ext
    # tyro alias.
    aliases = {f"exp:{name.replace('_', '-')}" for name in DEFAULTS}
    assert "exp:g1-wbt-2pt-force-ext" in aliases
    # yam URDF baked in (matches the other force presets).
    assert g1_wbt_2pt_force_ext.robot.asset.urdf_file == "g1/g1_29dof_yam.urdf"
    # Force machinery inherited from the 2pt-force base (command + curriculum unchanged).
    assert "hand_force_injector" in g1_wbt_2pt_force_ext.command.setup_terms
    assert "hand_force_injector" in g1_wbt_2pt_force_ext.command.reset_terms
    assert "force_alpha" in g1_wbt_2pt_force_ext.curriculum.setup_terms


def test_2pt_force_ext_actor_sees_both_forces():
    """The whole point of this variant: F^ext is ADDED BACK to the actor obs
    (drops the UniFP actor/critic asymmetry), so the actor sees BOTH F^cmd and
    F^ext. Contrast test_force_obs_terms_use_sequential_keys, which asserts the
    OPPOSITE for the shipped g1_wbt_2pt_force."""
    import re

    from wbt_training.config_values.force_tracking import g1_29dof_wbt_observation_2pt_force_ext

    actor_keys = list(g1_29dof_wbt_observation_2pt_force_ext.groups["actor_obs"].terms.keys())
    assert all(re.match(r"^obs\d+_", k) for k in actor_keys), actor_keys
    assert any("force_cmd_local" in k for k in actor_keys), "actor must see F^cmd"
    assert any("force_ext_local" in k for k in actor_keys), (
        "actor MUST see F^ext in this variant (that is the whole point of -force-ext)"
    )
    # F^ext lands right after F^cmd (first-commit layout).
    cmd_idx = next(i for i, k in enumerate(actor_keys) if "force_cmd_local" in k)
    ext_idx = next(i for i, k in enumerate(actor_keys) if "force_ext_local" in k)
    assert ext_idx == cmd_idx + 1, actor_keys
    # Critic obs unchanged from 2pt-force (still F^cmd + F^ext, pelvis frame).
    critic_keys = list(g1_29dof_wbt_observation_2pt_force_ext.groups["critic_obs"].terms.keys())
    assert "force_cmd_local" in critic_keys and "force_ext_local" in critic_keys


def test_2pt_force_ext_actor_obs_is_153d():
    """141D 2pt baseline + 6D F^cmd + 6D F^ext = 153D per frame (before history).
    Compares term-for-term against the shipped 147D g1_wbt_2pt_force actor obs to
    prove the ONLY per-frame delta is the added 6D force_ext_local term."""
    from wbt_training.config_values.force_tracking import force_cmd_local, force_ext_local
    from wbt_training.config_values.experiment import g1_wbt_2pt, g1_wbt_2pt_force, g1_wbt_2pt_force_ext

    # The two force obs terms are 6D each (2 hands × 3 xyz); reuse the pure-reader
    # helpers on a minimal env to confirm the per-term width rather than hard-coding.
    cfg_none_env = _make_env_min(num_envs=1)
    _allocate_force_buffers(
        cfg_none_env,
        __import__("wbt_training.config_values.force_tracking", fromlist=["HandForceConfig"]).HandForceConfig(),
    )
    assert force_cmd_local(cfg_none_env).shape[-1] == 6
    assert force_ext_local(cfg_none_env).shape[-1] == 6

    ext_actor = g1_wbt_2pt_force_ext.observation.groups["actor_obs"].terms
    base_actor = g1_wbt_2pt.observation.groups["actor_obs"].terms
    force_actor = g1_wbt_2pt_force.observation.groups["actor_obs"].terms
    # -ext actor = 2pt baseline (9 terms) + F^cmd + F^ext = 11 terms.
    assert len(ext_actor) == len(base_actor) + 2
    # vs the shipped 2pt-force actor (10 terms = baseline + F^cmd only), -ext adds exactly one.
    assert len(ext_actor) == len(force_actor) + 1
    # The extra term is force_ext_local.
    ext_funcs = [t.func for t in ext_actor.values()]
    force_funcs = [t.func for t in force_actor.values()]
    added = [f for f in ext_funcs if f not in force_funcs or ext_funcs.count(f) > force_funcs.count(f)]
    assert any(f.endswith(":force_ext_local") for f in added)


def test_2pt_force_ext_reward_is_first_commit_recipe():
    """First-commit ('最最开始') reward: FULL 14-body position tracking KEPT (no
    no_wrist swap), Eq.2 @ 4.0 (NOT the later 6.0 bump), pen @ -6.0 (restored),
    and NO feet_slip. This is the exact set that differs from g1_wbt_2pt_force."""
    from wbt_training.config_values.experiment import g1_wbt_2pt_force, g1_wbt_2pt_force_ext

    rt = g1_wbt_2pt_force_ext.reward.terms
    # Force terms at the original weights.
    assert rt["force_track_pos_eq2"].weight == 4.0, "Eq.2 must be the first-commit 4.0, not 6.0"
    assert "force_track_pos_pen" in rt, "the -6.0 safety penalty is restored in this variant"
    assert rt["force_track_pos_pen"].weight == -6.0
    # FULL-body position tracking KEPT — the whole body is tracked, wrist included
    # (no no_wrist swap), so it is NOT 'over-tracking the wrist alone'.
    assert "motion_relative_body_position_error_exp" in rt
    assert "motion_relative_body_position_error_exp_no_wrist" not in rt
    assert rt["motion_relative_body_position_error_exp"].weight == 1.0
    # Orientation term inherited unchanged (full 14 bodies, weight 1.0).
    assert rt["motion_relative_body_orientation_error_exp"].weight == 1.0
    # No feet_slip in the first-commit recipe.
    assert "feet_slip" not in rt
    # CONTRAST: the shipped g1_wbt_2pt_force is the LATER recipe (Eq.2@6.0, no_wrist,
    # feet_slip, no pen). Assert we differ from it on every one of those axes.
    ft = g1_wbt_2pt_force.reward.terms
    assert ft["force_track_pos_eq2"].weight == 6.0
    assert "motion_relative_body_position_error_exp_no_wrist" in ft
    assert "feet_slip" in ft
    assert "force_track_pos_pen" not in ft


def test_2pt_force_ext_no_fmeas_no_servo():
    """Pure UniFP position-overshoot: this variant must NOT pull in any hybrid
    (F_meas / servo / integrator) machinery — it is built off force_tracking, not
    hybrid. Guards against an accidental import of the sensor/servo obs terms."""
    from wbt_training.config_values.experiment import g1_wbt_2pt_force_ext

    actor = g1_wbt_2pt_force_ext.observation.groups["actor_obs"].terms
    critic = g1_wbt_2pt_force_ext.observation.groups["critic_obs"].terms
    all_funcs = [t.func for t in {**actor, **critic}.values()]
    for banned in ("force_meas_local", "integrator_state", "x_tgt_shift", "contact_gate"):
        assert not any(banned in f for f in all_funcs), f"{banned} must not appear (no F_meas/servo)"
    # Reward has no servo/pre-integrator terms. NOTE: check the func PATHS by substring
    # (the reward func is a full dotted path like "…hybrid:force_track_eq2_servo", so a
    # bare-name `not in {…funcs}` set check would be vacuously true — match the obs-side
    # substring style instead so a wired servo reward func actually trips this).
    assert "force_error_preintegrator" not in g1_wbt_2pt_force_ext.reward.terms
    reward_funcs = [t.func for t in g1_wbt_2pt_force_ext.reward.terms.values()]
    for banned in ("force_track_eq2_servo", "force_error_reward_preintegrator"):
        assert not any(banned in f for f in reward_funcs), f"{banned} must not appear (no servo/pre-integrator)"
    # And the Eq.2 reward must be the PLAIN force_tracking one, not the hybrid servo variant.
    assert any(f.endswith("force_tracking:force_track_pos_eq2") for f in reward_funcs)


def test_existing_experiments_unmutated_by_force_preset():
    """Non-Invasion Contract: every existing g1_wbt_* preset must remain
    *byte-identical* after this PR. Importing experiment.py adds the new
    `g1_wbt_2pt_force` preset; assert (a) the parent g1_wbt_2pt was not
    mutated by `replace(...)` (object identity preserved), (b) DEFAULTS
    gained exactly one new key, and (c) every protected preset still
    resolves to a non-None cfg.
    """
    from wbt_training.config_values import experiment as exp
    from wbt_training.config_values.experiment import DEFAULTS

    # Subset of the protected list (sufficient to catch identity violations).
    protected = {
        "g1_wbt",
        "g1_wbt_fast_sac",
        "g1_wbt_3pt",
        "g1_wbt_2pt",
        "g1_wbt_1pt",
        "g1_wbt_chip_local",
        "g1_wbt_chip_global",
        "g1_wbt_global",
        "g1_wbt_global_vel",
        "g1_wbt_maskedmimic",
        "g1_wbt_3pt_mm",
        "g1_wbt_global_mm",
        "g1_wbt_terrain_ref",
    }
    for k in protected:
        assert k in DEFAULTS, f"missing {k}"
        assert DEFAULTS[k] is getattr(exp, k), f"{k} re-mapped — non-invasion violated"

    # Parent of new preset: replace() returns a copy; original must be intact.
    assert exp.g1_wbt_2pt.observation is not exp.g1_wbt_2pt_force.observation
    assert exp.g1_wbt_2pt.command is not exp.g1_wbt_2pt_force.command
    assert exp.g1_wbt_2pt.reward is not exp.g1_wbt_2pt_force.reward
    assert exp.g1_wbt_2pt.curriculum is not exp.g1_wbt_2pt_force.curriculum

    # Strict assertion on the force preset inventory (see the force_keys check below).
    assert "g1_wbt_2pt_force" in DEFAULTS
    # `g1_wbt_2pt_force_falcon` (FALCON dual-agent), `g1_wbt_2pt_force_upperbody`
    # (single-actor upper-body precision), and `g1_wbt_2pt_force_ext` (F^ext-in-actor
    # first-commit revert) are intentional additive siblings built by replace() off the
    # 2pt-force / maskedmimic-force bases; none may be the SAME object as g1_wbt_2pt_force
    # nor mutate it.
    force_keys = {k for k in DEFAULTS if "2pt_force" in k}
    assert force_keys == {
        "g1_wbt_2pt_force",
        "g1_wbt_2pt_force_ext",
        "g1_wbt_2pt_force_falcon",
        "g1_wbt_2pt_force_upperbody",
    }, f"unexpected force keys: {force_keys}"
    assert DEFAULTS["g1_wbt_2pt_force_falcon"] is not DEFAULTS["g1_wbt_2pt_force"]
    assert DEFAULTS["g1_wbt_2pt_force_upperbody"] is not DEFAULTS["g1_wbt_2pt_force"]
    assert DEFAULTS["g1_wbt_2pt_force_ext"] is not DEFAULTS["g1_wbt_2pt_force"]
    # The _ext sibling must NOT mutate the 2pt-force parent's obs/reward objects.
    assert DEFAULTS["g1_wbt_2pt_force_ext"].observation is not DEFAULTS["g1_wbt_2pt_force"].observation
    assert DEFAULTS["g1_wbt_2pt_force_ext"].reward is not DEFAULTS["g1_wbt_2pt_force"].reward
    # The FALCON preset uses the dual-agent algo + env; the 2pt-force preset does not.
    assert "PPODecoupled" in DEFAULTS["g1_wbt_2pt_force_falcon"].algo._target_
    assert "PPODecoupled" not in DEFAULTS["g1_wbt_2pt_force"].algo._target_
    assert "FalconDecoupled" not in DEFAULTS["g1_wbt_2pt_force"].env_class
    # The upperbody preset is single-actor (stock PPO, no dual-agent algo/env) and only
    # adds TCP reward terms + a yam URDF; it must not touch the 2pt-force parent.
    assert "PPODecoupled" not in DEFAULTS["g1_wbt_2pt_force_upperbody"].algo._target_
    assert "tcp_track_pos" not in DEFAULTS["g1_wbt_2pt_force"].reward.terms
    assert "tcp_track_pos" in DEFAULTS["g1_wbt_2pt_force_upperbody"].reward.terms


# ===========================================================================
# motion_relative_body_position_error_exp_no_wrist:
# 14-link relative position-tracking reward with wrist excluded so Eq.2 owns
# the rubber_hand position target alone (no 1.0/4.0 tug-of-war on the wrist).
# ===========================================================================


def _make_env_with_motion_bodies(body_names, num_envs=2):
    """Minimal env exposing motion_command.body_pos_relative_w / robot_body_pos_w
    for an arbitrary body-name list, plus motion_cfg.body_names_to_track."""
    env = MagicMock()
    env.num_envs = num_envs
    env.device = torch.device("cpu")

    mc = MagicMock()
    mc.motion_cfg = MagicMock()
    mc.motion_cfg.body_names_to_track = list(body_names)
    n = num_envs
    b = len(body_names)
    mc.body_pos_relative_w = torch.zeros(n, b, 3)
    mc.robot_body_pos_w = torch.zeros(n, b, 3)
    # Orientation buffers (xyzw); identity = (0, 0, 0, 1).
    mc.body_quat_relative_w = torch.zeros(n, b, 4)
    mc.body_quat_relative_w[..., 3] = 1.0
    mc.robot_body_quat_w = torch.zeros(n, b, 4)
    mc.robot_body_quat_w[..., 3] = 1.0

    cm = MagicMock()
    cm.get_state = MagicMock(side_effect=lambda name: mc if name == "motion_command" else None)
    env.command_manager = cm
    return env, mc


def test_motion_relative_body_position_error_exp_no_wrist_perfect_match_returns_one():
    """ref == robot for all bodies → r = exp(0) = 1."""
    from wbt_training.config_values.force_tracking import motion_relative_body_position_error_exp_no_wrist

    body_names = ["pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    env, mc = _make_env_with_motion_bodies(body_names)
    mc.body_pos_relative_w.fill_(0.5)
    mc.robot_body_pos_w.fill_(0.5)
    r = motion_relative_body_position_error_exp_no_wrist(env, sigma=0.3)
    assert r.shape == (env.num_envs,)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-6)


def test_motion_relative_body_position_error_exp_no_wrist_excludes_wrist():
    """Move ONLY the wrist bodies; reward should still be 1.0 because wrist is
    excluded from the per-body mean. Move pelvis instead and reward drops."""
    from wbt_training.config_values.force_tracking import motion_relative_body_position_error_exp_no_wrist

    body_names = ["pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    env, mc = _make_env_with_motion_bodies(body_names)

    # Wrist-only displacement: 1m apart on the two wrist bodies, but they are
    # excluded → kept indices are pelvis, torso → all-zero diff → r=1.
    mc.body_pos_relative_w.zero_()
    mc.robot_body_pos_w.zero_()
    mc.body_pos_relative_w[:, 2, 0] = 1.0  # left_wrist_yaw_link x = 1m on motion side
    mc.body_pos_relative_w[:, 3, 0] = 1.0  # right_wrist_yaw_link x = 1m
    r_wrist_only = motion_relative_body_position_error_exp_no_wrist(env, sigma=0.3)
    assert torch.allclose(r_wrist_only, torch.ones(env.num_envs), atol=1e-6), (
        f"wrist-only displacement should be ignored, got {r_wrist_only}"
    )

    # Reset the per-env cache before re-running with a different displacement
    # pattern (body_names_to_track is the same → kept indices unchanged, but
    # the cache survived from above).
    # pelvis displacement: kept includes pelvis (idx 0), so this DOES count.
    env2, mc2 = _make_env_with_motion_bodies(body_names)
    mc2.body_pos_relative_w[:, 0, 0] = 0.6  # pelvis x = 0.6m
    # err per body: pelvis = 0.36, torso = 0; mean over kept (2 bodies) = 0.18
    # r = exp(-0.18 / 0.09) = exp(-2.0) ≈ 0.1353
    r_pelvis = motion_relative_body_position_error_exp_no_wrist(env2, sigma=0.3)
    expected = torch.exp(torch.tensor(-0.18 / 0.09))
    assert torch.allclose(r_pelvis, torch.full_like(r_pelvis, expected.item()), atol=1e-5), (
        f"pelvis displacement should drop reward to {expected.item():.4f}, got {r_pelvis[0].item():.4f}"
    )


def test_motion_relative_body_position_error_exp_no_wrist_does_not_mutate_state():
    """Pure reader contract."""
    from wbt_training.config_values.force_tracking import motion_relative_body_position_error_exp_no_wrist

    body_names = ["pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    env, mc = _make_env_with_motion_bodies(body_names)
    mc.body_pos_relative_w.uniform_(-1, 1)
    mc.robot_body_pos_w.uniform_(-1, 1)
    snap_motion = mc.body_pos_relative_w.clone()
    snap_robot = mc.robot_body_pos_w.clone()

    _ = motion_relative_body_position_error_exp_no_wrist(env)

    assert torch.equal(mc.body_pos_relative_w, snap_motion)
    assert torch.equal(mc.robot_body_pos_w, snap_robot)


def test_motion_relative_body_orientation_error_exp_no_wrist_perfect_match_returns_one():
    """ref quat == robot quat for all bodies -> r = exp(0) = 1."""
    from wbt_training.config_values.force_tracking import motion_relative_body_orientation_error_exp_no_wrist

    body_names = ["pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    env, mc = _make_env_with_motion_bodies(body_names)
    r = motion_relative_body_orientation_error_exp_no_wrist(env, sigma=0.4)
    assert r.shape == (env.num_envs,)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-6)


def test_motion_relative_body_orientation_error_exp_no_wrist_excludes_wrist():
    """Rotating ONLY the wrist bodies leaves r = 1 (wrist excluded); rotating a
    kept body (pelvis) by 90deg about z drops the reward by the exp kernel."""
    import math

    from wbt_training.config_values.force_tracking import motion_relative_body_orientation_error_exp_no_wrist

    body_names = ["pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    # Wrist-only rotation: 90deg about z on both wrist bodies (xyzw).
    env, mc = _make_env_with_motion_bodies(body_names)
    s2 = (2**0.5) / 2.0
    for idx in (2, 3):  # the two wrist bodies
        mc.body_quat_relative_w[:, idx, :] = torch.tensor([0.0, 0.0, s2, s2])
    r_wrist = motion_relative_body_orientation_error_exp_no_wrist(env, sigma=0.4)
    assert torch.allclose(r_wrist, torch.ones(env.num_envs), atol=1e-5), (
        f"wrist-only rotation should be ignored, got {r_wrist}"
    )

    # Pelvis (kept) rotated 90deg about z; torso identity. quat_error_magnitude
    # returns pi/2 rad; squared = (pi/2)^2; mean over kept (pelvis, torso) =
    # (pi/2)^2 / 2; r = exp(-mean / 0.4^2).
    env2, mc2 = _make_env_with_motion_bodies(body_names)
    mc2.body_quat_relative_w[:, 0, :] = torch.tensor([0.0, 0.0, s2, s2])  # pelvis
    r_pelvis = motion_relative_body_orientation_error_exp_no_wrist(env2, sigma=0.4)
    expected = math.exp(-((math.pi / 2) ** 2 / 2.0) / 0.4**2)
    assert torch.allclose(r_pelvis, torch.full_like(r_pelvis, expected), atol=1e-4), (
        f"pelvis rotation should drop reward to {expected:.5f}, got {r_pelvis[0].item():.5f}"
    )


def test_motion_relative_body_orientation_error_exp_no_wrist_does_not_mutate_state():
    """Pure reader contract."""
    from wbt_training.config_values.force_tracking import motion_relative_body_orientation_error_exp_no_wrist

    body_names = ["pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    env, mc = _make_env_with_motion_bodies(body_names)
    snap_motion = mc.body_quat_relative_w.clone()
    snap_robot = mc.robot_body_quat_w.clone()
    _ = motion_relative_body_orientation_error_exp_no_wrist(env)
    assert torch.equal(mc.body_quat_relative_w, snap_motion)
    assert torch.equal(mc.robot_body_quat_w, snap_robot)


def test_orientation_keeps_baseline_14body_incl_wrist_in_preset():
    """g1_29dof_wbt_2pt_force_reward must KEEP the baseline 14-body relative
    orientation term UNCHANGED (incl. left/right_wrist_yaw_link).

    rubber_hand is rigidly fixed to wrist_yaw, so wrist_yaw orientation == hand
    orientation — the contact-pose DOF manipulation must track. It is orthogonal
    to Eq.2 (the FK offset cancels, so Eq.2 rewards wrist-ORIGIN translation, not
    rotation). The wrist-excluded orientation variant must NOT be wired here
    (the function still exists for reuse, but the preset uses the full 14-body
    baseline term)."""
    from wbt_training.config_values.force_tracking import g1_29dof_wbt_2pt_force_reward

    terms = g1_29dof_wbt_2pt_force_reward.terms
    # Baseline 14-body orientation term is kept (inherited unchanged).
    assert "motion_relative_body_orientation_error_exp" in terms, (
        "baseline 14-body orientation term (incl. wrist) must be kept — wrist_yaw "
        "orientation == rubber_hand orientation, the manipulation contact pose"
    )
    # The wrist-excluded orientation variant must NOT be wired into the preset.
    assert "motion_relative_body_orientation_error_exp_no_wrist" not in terms, (
        "no-wrist orientation variant must NOT be wired — that was the bug that "
        "left the hand contact orientation unsupervised"
    )
    term = terms["motion_relative_body_orientation_error_exp"]
    assert term.weight == 1.0
    assert term.params["sigma"] == 0.4


def test_no_wrist_reward_replaces_baseline_in_preset():
    """g1_29dof_wbt_2pt_force_reward must replace the baseline 14-link
    `motion_relative_body_position_error_exp` with the no-wrist variant, and
    must NOT include joint_pos_tracking / joint_vel_tracking (removed as
    redundant with the 14-link world-pose tracking already in baseline)."""
    from wbt_training.config_values.force_tracking import g1_29dof_wbt_2pt_force_reward

    terms = g1_29dof_wbt_2pt_force_reward.terms
    assert "motion_relative_body_position_error_exp" not in terms, (
        "baseline 14-link position term must be removed (replaced by no-wrist variant)"
    )
    assert "motion_relative_body_position_error_exp_no_wrist" in terms, "no-wrist variant must be wired in"
    assert "joint_pos_tracking" not in terms, "joint_pos_tracking removed as redundant"
    assert "joint_vel_tracking" not in terms, "joint_vel_tracking removed as redundant"

    no_wrist = terms["motion_relative_body_position_error_exp_no_wrist"]
    assert no_wrist.weight == 1.0
    assert no_wrist.func.endswith(":motion_relative_body_position_error_exp_no_wrist")
    assert no_wrist.params["sigma"] == 0.3
    assert tuple(no_wrist.params["exclude_body_names"]) == ("left_wrist_yaw_link", "right_wrist_yaw_link")


# ===========================================================================
# feet_slip — port of GH `feet_slip` reward.
# ===========================================================================


def _make_env_with_feet(num_envs=2, num_bodies=10, history_length=3):
    """Minimal env exposing simulator.body_names + contact_forces_history +
    _rigid_body_vel. Two ankle bodies at indices 5 (left) and 8 (right)."""
    env = MagicMock()
    env.num_envs = num_envs
    env.device = torch.device("cpu")

    sim = MagicMock()
    sim.body_names = [
        "pelvis",
        "torso",
        "left_hip_roll",
        "left_knee",
        "left_ankle_pitch",
        "left_ankle_roll_link",  # idx 5
        "right_hip_roll",
        "right_knee",
        "right_ankle_roll_link",  # idx 8
        "right_ankle_pitch",
    ]
    assert len(sim.body_names) == num_bodies
    sim.contact_forces_history = torch.zeros(num_envs, history_length, num_bodies, 3)
    sim._rigid_body_vel = torch.zeros(num_envs, num_bodies, 3)
    env.simulator = sim
    return env


def test_feet_slip_zero_when_feet_stationary_in_contact():
    """In contact + zero velocity → slip = 0. No false-positive while standing."""
    from wbt_training.config_values.force_tracking import feet_slip

    env = _make_env_with_feet()
    # Both ankle indices (5, 8) get a 50 N normal force in the latest history slot.
    env.simulator.contact_forces_history[:, :, 5, 2] = 50.0
    env.simulator.contact_forces_history[:, :, 8, 2] = 50.0
    # Velocity stays zero.
    r = feet_slip(env)
    assert r.shape == (env.num_envs,)
    assert torch.allclose(r, torch.zeros(env.num_envs), atol=1e-6)


def test_feet_slip_zero_when_feet_moving_but_not_in_contact():
    """Feet flying through air (v_xy = 1 m/s, contact forces = 0) → slip = 0.
    Catches the bug where the in_contact gate is dropped or inverted."""
    from wbt_training.config_values.force_tracking import feet_slip

    env = _make_env_with_feet()
    # contact_forces_history stays at zero.
    env.simulator._rigid_body_vel[:, 5, 0] = 1.0  # left foot moving forward in world
    env.simulator._rigid_body_vel[:, 8, 1] = 2.0  # right foot moving sideways
    r = feet_slip(env)
    assert torch.allclose(r, torch.zeros(env.num_envs), atol=1e-6), (
        f"feet_slip > 0 with no contact: {r}. Did the in_contact gate get dropped?"
    )


def test_feet_slip_value_matches_sum_of_squared_xy_speeds():
    """Both feet in contact, left foot v_xy = (3, 4) → ‖v‖² = 25.
    Right foot v_xy = (1, 0) → ‖v‖² = 1. Total per env = 26 (sum, not mean)."""
    from wbt_training.config_values.force_tracking import feet_slip

    env = _make_env_with_feet()
    env.simulator.contact_forces_history[:, :, 5, 2] = 50.0
    env.simulator.contact_forces_history[:, :, 8, 2] = 50.0
    env.simulator._rigid_body_vel[:, 5, 0] = 3.0
    env.simulator._rigid_body_vel[:, 5, 1] = 4.0
    env.simulator._rigid_body_vel[:, 8, 0] = 1.0
    r = feet_slip(env)
    assert torch.allclose(r, torch.full((env.num_envs,), 26.0), atol=1e-5)


def test_feet_slip_ignores_vertical_velocity():
    """v_z must NOT count — only xy components contribute. Foot jumping up from
    ground is not 'sliding'. Catches accidentally including z in the norm."""
    from wbt_training.config_values.force_tracking import feet_slip

    env = _make_env_with_feet()
    env.simulator.contact_forces_history[:, :, 5, 2] = 50.0
    env.simulator.contact_forces_history[:, :, 8, 2] = 50.0
    # Pure vertical velocity on both feet.
    env.simulator._rigid_body_vel[:, 5, 2] = 5.0
    env.simulator._rigid_body_vel[:, 8, 2] = -5.0
    r = feet_slip(env)
    assert torch.allclose(r, torch.zeros(env.num_envs), atol=1e-6), (
        f"got {r}, expected 0 — vertical velocity should not count toward slip."
    )


def test_feet_slip_returns_positive_magnitude():
    """Holosoma convention: func returns POSITIVE, weight in cfg is negative.
    Verify never returns a negative value (would double-negate with weight=-2)."""
    from wbt_training.config_values.force_tracking import feet_slip

    env = _make_env_with_feet()
    env.simulator.contact_forces_history[:, :, 5, 2] = 50.0
    env.simulator.contact_forces_history[:, :, 8, 2] = 50.0
    env.simulator._rigid_body_vel[:, 5, :2] = 2.0
    env.simulator._rigid_body_vel[:, 8, :2] = -3.0
    r = feet_slip(env)
    assert (r >= 0).all(), f"feet_slip returned negative value: {r}. Holosoma convention violated."
    assert (r > 0).all(), "expected positive slip with non-zero feet velocity in contact"


def test_feet_slip_contact_threshold_filters_noise():
    """Contact-force noise (< 1 N) should not register as in-contact."""
    from wbt_training.config_values.force_tracking import feet_slip

    env = _make_env_with_feet()
    # Below threshold (0.5 N noise floor) → should be filtered.
    env.simulator.contact_forces_history[:, :, 5, 2] = 0.5
    env.simulator.contact_forces_history[:, :, 8, 2] = 0.5
    env.simulator._rigid_body_vel[:, 5, :2] = 5.0
    env.simulator._rigid_body_vel[:, 8, :2] = 5.0
    r = feet_slip(env)
    assert torch.allclose(r, torch.zeros(env.num_envs)), (
        f"contact threshold not enforced: forces below 1 N triggered slip: {r}"
    )


def test_feet_slip_uses_max_over_history():
    """Contact in *any* history slot should count. If a foot touched ground at
    t-1 but not t, current_contact_time-style logic still flags it."""
    from wbt_training.config_values.force_tracking import feet_slip

    env = _make_env_with_feet(history_length=3)
    # No contact at t=2 (latest), but contact at t=0 and t=1.
    env.simulator.contact_forces_history[:, 0, 5, 2] = 50.0  # left foot in contact at t=0
    env.simulator.contact_forces_history[:, 1, 5, 2] = 50.0  # at t=1
    # t=2 is zero.
    env.simulator._rigid_body_vel[:, 5, :2] = 1.0  # |v_xy|² = 2
    r = feet_slip(env)
    # Only left foot contributed; right foot never touched ground.
    assert torch.allclose(r, torch.full((env.num_envs,), 2.0), atol=1e-5)


def test_feet_slip_does_not_mutate_state():
    """Pure reader contract: must not touch contact_forces_history or _rigid_body_vel."""
    from wbt_training.config_values.force_tracking import feet_slip

    env = _make_env_with_feet()
    env.simulator.contact_forces_history.uniform_(0, 100)
    env.simulator._rigid_body_vel.uniform_(-3, 3)
    snap_cf = env.simulator.contact_forces_history.clone()
    snap_v = env.simulator._rigid_body_vel.clone()
    _ = feet_slip(env)
    assert torch.equal(env.simulator.contact_forces_history, snap_cf)
    assert torch.equal(env.simulator._rigid_body_vel, snap_v)


def test_feet_slip_caches_indices_on_env():
    """Repeated calls must reuse the cached body indices (not re-resolve names)."""
    from wbt_training.config_values.force_tracking import feet_slip

    env = _make_env_with_feet()
    _ = feet_slip(env)
    assert hasattr(env, "_wbt_feet_slip_body_indices")
    assert env._wbt_feet_slip_body_indices.tolist() == [5, 8]


def test_feet_slip_wired_into_reward_preset_with_negative_weight():
    """Reward preset must use weight=-2.0 (sign-correctness gate).

    GH yaml has `feet_slip: {weight: 2.0, ...}` because GH func returns
    negative magnitude. Holosoma func returns positive magnitude, so the
    cfg must carry the negative sign — never both, never neither.
    """
    from wbt_training.config_values.force_tracking import g1_29dof_wbt_2pt_force_reward

    terms = g1_29dof_wbt_2pt_force_reward.terms
    assert "feet_slip" in terms, "feet_slip missing from reward preset"
    term = terms["feet_slip"]
    assert term.weight == -2.0, (
        f"feet_slip weight is {term.weight}, expected -2.0. "
        f"Holosoma convention: penalty func returns positive, cfg weight is negative."
    )
    assert term.func.endswith(":feet_slip")
    assert term.params == {"body_names": ("left_ankle_roll_link", "right_ankle_roll_link")}


# ===========================================================================
# --F_on_sensor_frame ablation: actor-only ft_sensor-frame F^cmd obs encoding.
# The sensor frame differs from wrist_yaw by the CONSTANT R_sensor (ft_sensor_joint
# rpy="0 -pi/2 pi"): F^cmd_sensor = R_sensor^T @ (R(q_wrist)^-1 @ F_world).
# ===========================================================================


def test_rot_matrix_from_rpy_sensor_is_symmetric_involution():
    """R_sensor from rpy=(0,-pi/2,pi) is a symmetric involution (det +1) that maps
    sensor +x→wrist +z, +y→wrist -y, +z→wrist +x (columns = sensor axes in wrist)."""
    from wbt_training.config_values.force_tracking import HandForceConfig, _rot_matrix_from_rpy

    R = _rot_matrix_from_rpy(HandForceConfig().sensor_rpy)
    expected = torch.tensor([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    assert torch.allclose(R, expected, atol=1e-6), R
    assert torch.allclose(R, R.T, atol=1e-6), "R_sensor must be symmetric"
    assert torch.allclose(R @ R, torch.eye(3), atol=1e-6), "R_sensor must be an involution"
    assert abs(torch.det(R).item() - 1.0) < 1e-6, "R_sensor must be a proper rotation (det +1)"


def test_force_cmd_local_sensor_frame_default_is_pelvis_path():
    """sensor_frame defaults to False → byte-identical to the legacy pelvis path."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_cmd_local

    cfg = HandForceConfig()
    env = _make_env_with_simulator(num_envs=2)
    _allocate_force_buffers(env, cfg)
    env._wbt_force_cmd_b[:, 0, 0] = 7.0
    env._wbt_force_cmd_w[:, 1, 1] = 9.0  # must be IGNORED on the pelvis path
    out = force_cmd_local(env, cfg=cfg)  # sensor_frame omitted → False
    assert torch.allclose(out, env._wbt_force_cmd_b.reshape(2, -1))


def test_force_cmd_local_sensor_frame_composes_constant_after_wrist():
    """sensor_frame=True → R_sensor^T @ (R(q_wrist)^-1 @ F_world). With identity wrist
    quats, the wrist rotation is identity so the result is exactly R_sensor^T @ F_world:
    world (Fx,Fy,Fz) → sensor (Fz,-Fy,Fx)."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_cmd_local

    cfg = HandForceConfig()
    env = _make_env_with_simulator(num_envs=2)  # fixture gives identity wrist quats
    _allocate_force_buffers(env, cfg)
    env._wbt_force_cmd_w[:, 0, :] = torch.tensor([20.0, 3.0, 5.0])  # left
    env._wbt_force_cmd_w[:, 1, :] = torch.tensor([0.0, 10.0, 0.0])  # right
    out = force_cmd_local(env, cfg=cfg, sensor_frame=True).reshape(2, 2, 3)
    # R_sensor^T @ (x,y,z) = (z,-y,x) for this symmetric involution.
    assert torch.allclose(out[0, 0], torch.tensor([5.0, -3.0, 20.0]), atol=1e-5), out[0, 0]
    assert torch.allclose(out[0, 1], torch.tensor([0.0, -10.0, 0.0]), atol=1e-5), out[0, 1]


def test_force_cmd_local_sensor_frame_equals_handframe_left_mult_by_Rt():
    """sensor-frame output == R_sensor^T @ hand-frame output, for a NON-identity
    wrist rotation (proves the composition order, not just the identity case)."""
    from wbt_training.config_values.force_tracking import HandForceConfig, _rot_matrix_from_rpy, force_cmd_local

    cfg = HandForceConfig()
    env = _make_env_with_simulator(num_envs=2)
    _allocate_force_buffers(env, cfg)
    # Wrist bodies (ids 17, 24) rotated 90° about z (wxyz).
    s2 = (2**0.5) / 2.0
    bq = env.simulator._robot.data.body_quat_w
    for bid in (17, 24):
        bq[:, bid, 0] = s2
        bq[:, bid, 3] = s2
    env._wbt_force_cmd_w[:, 0, :] = torch.tensor([20.0, 0.0, 0.0])
    env._wbt_force_cmd_w[:, 1, :] = torch.tensor([0.0, 10.0, 7.0])

    hand = force_cmd_local(env, cfg=cfg, hand_frame=True).reshape(2, 2, 3)
    sensor = force_cmd_local(env, cfg=cfg, sensor_frame=True).reshape(2, 2, 3)
    R = _rot_matrix_from_rpy(cfg.sensor_rpy)
    expected = torch.einsum("ji,nhj->nhi", R, hand)  # R_sensor^T @ hand
    assert torch.allclose(sensor, expected, atol=1e-6)


def test_force_cmd_local_sensor_frame_zero_maps_to_zero():
    """F^cmd=0 ⇒ 0 in the sensor frame too (F=0⇒2pt collapse preserved)."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_cmd_local

    cfg = HandForceConfig()
    env = _make_env_with_simulator(num_envs=2)
    _allocate_force_buffers(env, cfg)
    s2 = (2**0.5) / 2.0
    bq = env.simulator._robot.data.body_quat_w
    for bid in (17, 24):
        bq[:, bid, 0] = s2
        bq[:, bid, 3] = s2
    env._wbt_force_cmd_w.zero_()
    out = force_cmd_local(env, cfg=cfg, sensor_frame=True)
    assert torch.count_nonzero(out) == 0


def test_force_cmd_local_sensor_frame_is_pure_reader():
    """The sensor-frame branch must NOT mutate any force buffer or body_quat_w."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_cmd_local

    cfg = HandForceConfig()
    env = _make_env_with_simulator(num_envs=2)
    _allocate_force_buffers(env, cfg)
    env._wbt_force_cmd_w.uniform_(-5, 5)
    env.episode_length_buf += 1
    snap_w = env._wbt_force_cmd_w.clone()
    snap_b = env._wbt_force_cmd_b.clone()
    snap_q = env.simulator._robot.data.body_quat_w.clone()
    _ = force_cmd_local(env, cfg=cfg, sensor_frame=True)
    assert torch.equal(env._wbt_force_cmd_w, snap_w)
    assert torch.equal(env._wbt_force_cmd_b, snap_b)
    assert torch.equal(env.simulator._robot.data.body_quat_w, snap_q)


def test_force_cmd_local_hand_and_sensor_frame_mutually_exclusive():
    """Passing both hand_frame and sensor_frame must assert (guards CLI misuse)."""
    import pytest

    from wbt_training.config_values.force_tracking import HandForceConfig, force_cmd_local

    cfg = HandForceConfig()
    env = _make_env_with_simulator(num_envs=2)
    _allocate_force_buffers(env, cfg)
    with pytest.raises(AssertionError):
        force_cmd_local(env, cfg=cfg, hand_frame=True, sensor_frame=True)


def test_sensorframe_preset_flips_actor_only_critic_untouched():
    """g1_29dof_wbt_observation_2pt_force_sensorframe = actor force_cmd_local with
    params sensor_frame=True, critic untouched (strict actor-only). Layout, ordering,
    scale, noise, and the base preset are all preserved."""
    from wbt_training.config_values.force_tracking import (
        g1_29dof_wbt_observation_2pt_force as base,
        g1_29dof_wbt_observation_2pt_force_sensorframe as sf,
    )

    a, c = sf.groups["actor_obs"].terms, sf.groups["critic_obs"].terms
    base_a = base.groups["actor_obs"].terms
    afc = next(k for k, v in a.items() if v.func.endswith(":force_cmd_local"))
    cfc = next(k for k, v in c.items() if v.func.endswith(":force_cmd_local"))
    assert a[afc].params.get("sensor_frame") is True, "actor force_cmd_local must be sensor_frame"
    assert c[cfc].params.get("sensor_frame", False) is False, "critic force_cmd_local must stay pelvis (actor-only)"
    # Base preset untouched (non-invasion).
    assert base_a[afc].params.get("sensor_frame", None) is None, "base actor preset must NOT be mutated"
    assert a[afc].scale == base_a[afc].scale
    assert a[afc].noise == base_a[afc].noise
    assert list(a.keys()) == list(base_a.keys()), "actor term order/layout must be preserved"
    # force_ext_local is never sensor-framed.
    for grp in (a, c):
        for k, v in grp.items():
            if v.func.endswith(":force_ext_local"):
                assert v.params.get("sensor_frame", False) is False


def test_with_force_cmd_sensor_frame_noop_on_non_force_preset():
    """The helper returns the SAME object for an obs cfg with no force_cmd_local
    actor term (so the CLI flag is a harmless no-op on non-force presets)."""
    from wbt_training.config_values.force_tracking import with_force_cmd_sensor_frame
    from wbt_training.config_values.observation import g1_29dof_wbt_observation_2pt

    out = with_force_cmd_sensor_frame(g1_29dof_wbt_observation_2pt)
    assert out is g1_29dof_wbt_observation_2pt, "no force_cmd_local actor term → unchanged identity"


def test_cli_flag_pop_and_apply_sensor_frame_actor_only():
    """train_agent --F_on_sensor_frame: argv pop (bare + value forms) and applying
    it to the real g1_wbt_2pt_force flips the actor term only."""
    import sys

    import wbt_training.train_agent as ta
    from wbt_training.config_values.experiment import g1_wbt_2pt, g1_wbt_2pt_force

    saved = sys.argv
    try:
        sys.argv = ["p", "exp:g1-wbt-2pt-force", "--F_on_sensor_frame", "--x", "1"]
        assert ta._pop_f_on_sensor_frame_flag() is True
        assert "--F_on_sensor_frame" not in sys.argv
        assert sys.argv == ["p", "exp:g1-wbt-2pt-force", "--x", "1"]
        sys.argv = ["p", "--F_on_sensor_frame", "False"]
        assert ta._pop_f_on_sensor_frame_flag() is False
        assert sys.argv == ["p"]
        sys.argv = ["p", "exp:g1-wbt"]
        assert ta._pop_f_on_sensor_frame_flag() is False
        assert sys.argv == ["p", "exp:g1-wbt"]
    finally:
        sys.argv = saved

    assert ta._apply_f_on_sensor_frame(g1_wbt_2pt_force, False) is g1_wbt_2pt_force
    new = ta._apply_f_on_sensor_frame(g1_wbt_2pt_force, True)
    assert new is not g1_wbt_2pt_force
    a = new.observation.groups["actor_obs"].terms
    c = new.observation.groups["critic_obs"].terms
    afc = next(k for k, v in a.items() if v.func.endswith(":force_cmd_local"))
    cfc = next(k for k, v in c.items() if v.func.endswith(":force_cmd_local"))
    assert a[afc].params.get("sensor_frame") is True
    assert c[cfc].params.get("sensor_frame", False) is False
    # Original preset NOT mutated; non-obs config shared.
    assert g1_wbt_2pt_force.observation.groups["actor_obs"].terms[afc].params.get("sensor_frame", None) is None
    assert new.command is g1_wbt_2pt_force.command
    assert new.reward is g1_wbt_2pt_force.reward
    # NON-force preset → unchanged.
    assert ta._apply_f_on_sensor_frame(g1_wbt_2pt, True) is g1_wbt_2pt
