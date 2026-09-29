"""Unit tests for exp:g1-wbt-3pt-hybrid (the 2pt-hybrid controller ported to 3pt tracking).

Covers the ONLY two genuinely-new pieces (everything else is reused verbatim from hybrid.py):
  - the 168D 3pt-hybrid ACTOR observation layout (vr_3point + force/servo terms, zero-padded
    sorted keys obs00..obs12) and its FIXED width across servo on/off + dropout on/off;
  - the ACTOR joint-velocity dropout obs term (disabled / eval_zero / per_episode / per_step /
    lazy-mask / reset re-sample), and that it is ACTOR-only (the critic keeps a clean dof_vel);
  - the experiment presets (yam URDF, servo shipped ON + estop, servo-off two-phase parity);
  - the inference + mjswan obs configs agree with the training layout (168D, same term order).

All tests use MagicMock / lightweight fakes for env/simulator — no IsaacSim runtime needed
(mirrors tests/test_hybrid.py + tests/test_maskedmimic_hybrid.py)."""

from unittest.mock import MagicMock

import torch

# ===========================================================================
# 1. 168D actor observation layout.
# ===========================================================================

_EXPECTED_ACTOR = {
    "obs00_command_lower_body": 24,
    "obs01_vr_3point_pos": 9,
    "obs02_vr_3point_orn": 12,
    "obs03_force_cmd_local": 6,
    "obs04_force_meas_local": 6,
    "obs05_integrator_state": 6,
    "obs06_x_tgt_shift": 6,
    "obs07_motion_ref_ori_b": 6,
    "obs08_projected_gravity": 3,
    "obs09_base_ang_vel": 3,
    "obs10_dof_pos": 29,
    "obs11_dof_vel": 29,
    "obs12_actions": 29,
}


def test_actor_obs_keys_and_order():
    from wbt_training.config_values.hybrid_3pt import g1_29dof_wbt_observation_3pt_hybrid as obs

    actor = obs.groups["actor_obs"]
    assert list(actor.terms.keys()) == list(_EXPECTED_ACTOR.keys())
    # The obs manager concatenates in sorted(keys()) order — insertion order MUST equal
    # sorted order (zero-padded prefixes), else the vector scrambles (obs10 vs obs1/obs2).
    assert list(actor.terms.keys()) == sorted(actor.terms.keys())
    assert actor.history_length == 4
    assert obs.groups["critic_obs"].history_length == 10


def test_actor_per_frame_dim_is_168():
    # Widths are the documented single-frame dims; their sum is the 168D contract.
    assert sum(_EXPECTED_ACTOR.values()) == 168


def test_actor_uses_vr_3point_not_2point():
    from wbt_training.config_values.hybrid_3pt import g1_29dof_wbt_observation_3pt_hybrid as obs

    terms = obs.groups["actor_obs"].terms
    assert "vr_3point_local_target" in terms["obs01_vr_3point_pos"].func
    assert "vr_3point_local_orn_target" in terms["obs02_vr_3point_orn"].func
    # No 2pt VR term leaked in from the source 2pt-hybrid layout.
    assert not any("vr_2point" in t.func for t in terms.values())


def test_force_and_servo_terms_present_with_right_scales():
    from wbt_training.config_values.hybrid_3pt import g1_29dof_wbt_observation_3pt_hybrid as obs

    terms = obs.groups["actor_obs"].terms
    assert terms["obs03_force_cmd_local"].scale == 0.04
    assert terms["obs04_force_meas_local"].scale == 0.04
    assert terms["obs05_integrator_state"].scale == 1.0
    assert terms["obs06_x_tgt_shift"].scale == 1.0
    # Force/servo terms carry no manager noise (F^meas already carries the D4 sensor model;
    # F^cmd/I/Δx are internal), matching the 2pt-hybrid actor.
    for k in ("obs03_force_cmd_local", "obs04_force_meas_local", "obs05_integrator_state", "obs06_x_tgt_shift"):
        assert terms[k].noise == 0.0
    assert "hybrid:force_meas_local" in terms["obs04_force_meas_local"].func
    assert "hybrid:integrator_state" in terms["obs05_integrator_state"].func
    assert "hybrid:x_tgt_shift" in terms["obs06_x_tgt_shift"].func


def test_dof_vel_term_is_the_dropout_term_with_zero_manager_noise():
    from wbt_training.config_values.hybrid_3pt import g1_29dof_wbt_observation_3pt_hybrid as obs

    dv = obs.groups["actor_obs"].terms["obs11_dof_vel"]
    # The actor dof_vel is the stateful JointVelDropout term (which applies the base 0.5 noise
    # INTERNALLY), so the ObsTermCfg noise MUST be 0.0 — else the manager double-applies noise.
    assert dv.func.endswith("hybrid_3pt:JointVelDropout")
    assert dv.noise == 0.0
    assert dv.scale == 1.0


def test_critic_reused_from_2pt_hybrid_verbatim():
    from wbt_training.config_values.hybrid import g1_29dof_wbt_observation_2pt_hybrid as hy2
    from wbt_training.config_values.hybrid_3pt import g1_29dof_wbt_observation_3pt_hybrid as hy3

    # The hybrid critic is tracking-agnostic; the 3pt preset reuses the SAME term objects.
    # (ObsGroupCfg shallow-copies the terms dict into a fresh container, so compare identity
    # per-term rather than the container object.)
    c2 = hy2.groups["critic_obs"].terms
    c3 = hy3.groups["critic_obs"].terms
    assert set(c2) == set(c3)
    assert all(c3[k] is c2[k] for k in c2), "every critic term must be the SAME object (verbatim reuse)."
    # Critic still carries a clean (non-dropout) dof_vel term.
    assert "dof_vel" in c3
    assert c3["dof_vel"].func.endswith("wbt:dof_vel")


# ===========================================================================
# 2. Experiment presets (yam URDF, servo ON + estop, servo-off two-phase parity).
# ===========================================================================


def test_presets_registered_with_yam_urdf():
    from wbt_training.config_values.experiment import DEFAULTS

    for name in ("g1_wbt_3pt_hybrid", "g1_wbt_3pt_hybrid_servo_off"):
        assert name in DEFAULTS, name
        assert DEFAULTS[name].robot.asset.urdf_file == "g1/g1_29dof_yam.urdf"


def test_preset_ships_servo_enabled_with_estop():
    """The shipped exp:g1-wbt-3pt-hybrid reuses hybrid.py's servo-ON command + force_runaway
    estop termination — verify both landed (force tracking on launch, deploy-safety valve)."""
    from wbt_training.config_values.experiment import DEFAULTS

    on = DEFAULTS["g1_wbt_3pt_hybrid"]
    inj = on.command.setup_terms["hybrid_contact_injector"]
    assert inj.params["hybrid_config"].servo_enabled is True
    assert "force_runaway" in on.termination.terms


def test_servo_off_variant_flips_only_servo_and_keeps_obs_and_reward_keys():
    from wbt_training.config_values.experiment import DEFAULTS

    on = DEFAULTS["g1_wbt_3pt_hybrid"]
    off = DEFAULTS["g1_wbt_3pt_hybrid_servo_off"]
    # servo flag flips
    assert off.command.setup_terms["hybrid_contact_injector"].params["hybrid_config"].servo_enabled is False
    # obs is the SAME manager (fixed 168D → checkpoint transfers with no surgery)
    assert on.observation is off.observation
    # reward KEY set identical (only the cfg-carriers differ) → critic/checkpoint lineage intact
    assert set(on.reward.terms.keys()) == set(off.reward.terms.keys())
    # estop still wired in the servo-off phase (keeps the termination manager identical)
    assert "force_runaway" in off.termination.terms


def test_preset_reuses_2pt_hybrid_reward_and_command_verbatim():
    """The hybrid stack is tracking-agnostic: the 3pt preset reuses hybrid.py's reward /
    command / curriculum / termination objects directly (no re-wrap in hybrid_3pt.py)."""
    from wbt_training.config_values import hybrid as H
    from wbt_training.config_values.experiment import DEFAULTS

    on = DEFAULTS["g1_wbt_3pt_hybrid"]
    assert on.reward is H.g1_29dof_wbt_2pt_hybrid_reward
    assert on.command is H.g1_29dof_wbt_hybrid_command
    assert on.curriculum is H.g1_29dof_wbt_hybrid_curriculum
    assert on.termination is H.g1_29dof_wbt_hybrid_termination


# ===========================================================================
# 3. ACTOR joint-velocity dropout obs term.
# ===========================================================================


class _FakeEnv:
    def __init__(self, num_envs=6, device="cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)


def _make_term(jv_cfg, num_envs=6, base_noise=0.0):
    """Build a JointVelDropout term with a MagicMock ObsTermCfg carrying the jv_cfg + base_noise
    params. base_noise defaults to 0 so the dropout math is exact (no injected uniform noise)."""
    from wbt_training.config_values.hybrid_3pt import JointVelDropout

    cfg = MagicMock()
    cfg.params = {"jv_cfg": jv_cfg, "base_noise": base_noise}
    return JointVelDropout(cfg, _FakeEnv(num_envs))


def test_module_default_dropout_is_disabled():
    """The 3pt-hybrid preset ships with the dropout OFF (user 2026-07-16), unlike maskmimic."""
    import wbt_training.config_values.hybrid_3pt as H

    assert H._DEFAULT_JV_DROPOUT_CFG.enabled is False


def test_jv_dropout_disabled_is_identity():
    from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig

    t = _make_term(JointVelDropoutConfig(enabled=False))
    jv = torch.randn(6, 29)
    assert torch.equal(t._apply_jv_dropout(jv), jv)


def test_jv_dropout_eval_zero_zeros_everything():
    from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig

    t = _make_term(JointVelDropoutConfig(enabled=True, eval_zero=True))
    assert torch.count_nonzero(t._apply_jv_dropout(torch.randn(6, 29))) == 0


def test_jv_dropout_per_episode_zeros_exactly_masked_envs():
    from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig

    t = _make_term(JointVelDropoutConfig(enabled=True, per_episode=True))
    t._jv_dropout_mask = torch.tensor([False, True, False, True, False, True])
    out = t._apply_jv_dropout(torch.ones(6, 29))
    dropped = out.abs().sum(dim=-1) == 0
    assert torch.equal(dropped, torch.tensor([False, True, False, True, False, True]))
    assert torch.equal(out[0], torch.ones(29))  # survivor unchanged (no noise_scale)


def test_jv_dropout_prob_edges():
    from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig

    t1 = _make_term(JointVelDropoutConfig(enabled=True, dropout_prob=1.0), num_envs=8)
    assert torch.count_nonzero(t1._apply_jv_dropout(torch.ones(8, 29))) == 0
    t0 = _make_term(JointVelDropoutConfig(enabled=True, dropout_prob=0.0), num_envs=8)
    assert torch.equal(t0._apply_jv_dropout(torch.ones(8, 29)), torch.ones(8, 29))


def test_jv_dropout_lazy_mask_is_sampled_not_zero_filled():
    from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig

    t = _make_term(JointVelDropoutConfig(enabled=True, dropout_prob=1.0, per_episode=True), num_envs=16)
    mask = t._ensure_jv_mask()
    assert mask.dtype == torch.bool and mask.shape == (16,)
    assert bool(mask.all()), "prob=1.0 lazy mask must be all-True (sampled, not zero-filled)."


def test_jv_dropout_per_step_varies():
    from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig

    t = _make_term(JointVelDropoutConfig(enabled=True, per_episode=False), num_envs=2048)
    a = t._apply_jv_dropout(torch.ones(2048, 29)).abs().sum(dim=-1) == 0
    b = t._apply_jv_dropout(torch.ones(2048, 29)).abs().sum(dim=-1) == 0
    assert not torch.equal(a, b), "per_step masks should differ across calls (prob 0.5, 2048 envs)."


def test_jv_dropout_reset_resamples_per_episode_mask():
    from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig

    t = _make_term(JointVelDropoutConfig(enabled=True, dropout_prob=1.0, per_episode=True))
    t.reset(torch.tensor([0, 2, 4]))
    assert t._jv_dropout_mask[[0, 2, 4]].all()  # prob=1.0 → the reset envs are dropped
    # per_step reset is a no-op (no persistent mask); must not raise.
    t2 = _make_term(JointVelDropoutConfig(enabled=True, per_episode=False))
    t2.reset(None)


def test_jv_dropout_call_applies_base_noise_then_dropout(monkeypatch):
    """__call__ pulls the core dof_vel, applies the base 0.5 uniform noise, THEN the dropout.
    With dropout disabled the output equals dof_vel + base-noise (noise present); with prob=1.0
    every env is zeroed regardless of the noise."""
    import wbt_training.config_values.hybrid_3pt as H
    from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig

    fake_dv = torch.ones(6, 29) * 3.0
    monkeypatch.setattr(H, "_core_dof_vel", lambda env: fake_dv.clone())

    # Disabled: base noise applied → not exactly equal to the clean dof_vel, but same shape.
    t = _make_term(JointVelDropoutConfig(enabled=False), base_noise=0.5)
    out = t(_FakeEnv(6))
    assert out.shape == (6, 29)
    assert not torch.equal(out, fake_dv), "base 0.5 noise should perturb the disabled output."
    assert (out - fake_dv).abs().max() <= 0.5 + 1e-6

    # prob=1.0: dropped to exactly 0 even with base noise upstream.
    t2 = _make_term(JointVelDropoutConfig(enabled=True, dropout_prob=1.0), base_noise=0.5)
    assert torch.count_nonzero(t2(_FakeEnv(6))) == 0


# ===========================================================================
# 4. Inference + mjswan obs configs agree with training (168D, same order).
# ===========================================================================


def test_inference_obs_config_matches_training_layout():
    from wbt_wrappers_inference.config_values.observation import holosoma_3pt_hybrid_wbt as obs

    keys = obs.obs_dict["actor_obs"]
    # inference lists BARE names (command_lower_body, vr_3point_pos, …) in the training sorted order
    expected = [k.split("_", 1)[1] for k in _EXPECTED_ACTOR]
    # obs00_command_lower_body → command_lower_body ; obs11_dof_vel → dof_vel (inference calls it joint_vel)
    expected = [{"dof_pos": "joint_pos", "dof_vel": "joint_vel"}.get(name, name) for name in expected]
    assert keys == expected, (keys, expected)
    assert sum(obs.obs_dims[k] for k in keys) == 168
    assert obs.history_length_dict["actor_obs"] == 4
    assert obs.obs_scales["force_cmd_local"] == 0.04 and obs.obs_scales["force_meas_local"] == 0.04


def test_inference_registry_maps_both_name_keys_to_same_config():
    from wbt_wrappers_inference.config_values import inference as I

    reg = next(v for v in vars(I).values() if isinstance(v, dict) and "g1-29dof-holosoma-3pt-hybrid-wbt" in v)
    assert reg["g1-29dof-holosoma-3pt-hybrid-wbt"] is I.g1_29dof_holosoma_3pt_hybrid_wbt
    assert reg["g1-29dof-holosoma-3pt-hybrid-layerA-wbt"] is I.g1_29dof_holosoma_3pt_hybrid_wbt


def test_mjswan_preset_matches_inference_order_and_dim():
    from wbt_realtime_eval.presets import get_preset
    from wbt_wrappers_inference.config_values.observation import holosoma_3pt_hybrid_wbt as obs

    for nm in ("g1-29dof-holosoma-3pt-hybrid-wbt", "g1-29dof-holosoma-3pt-hybrid-layerA-wbt"):
        p = get_preset(nm)
        assert p.vr_family == "3pt"
        assert p.history_length == 4
        assert p.per_frame_dim == 168
        assert list(p.obs_keys) == obs.obs_dict["actor_obs"], (nm, list(p.obs_keys))
