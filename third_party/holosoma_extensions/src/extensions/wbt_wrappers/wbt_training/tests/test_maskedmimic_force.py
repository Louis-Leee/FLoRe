"""Unit tests for the MaskedMimic + UniFP force-tracking variant.

All tests use MagicMock for env/simulator — no IsaacSim runtime needed. They
cover the parts NEW to the maskedmimic-force port: the global-frame Eq.2 reward,
the global no-wrist body reward, the 4-mode hover restriction, the actor/critic
obs asymmetry + layout, the handframe flip, command/curriculum wiring, the
preset registration, and non-invasion of the base maskedmimic + 2pt-force presets.

The shared force machinery (HandForceInjector, pulse, feet_slip, force_cmd_local,
hand-frame math) is already covered by test_force_tracking.py and reused verbatim.
"""

import types
from unittest.mock import MagicMock

import torch


# ===========================================================================
# Shared fixtures (mirror test_force_tracking.py conventions).
# ===========================================================================


def _eye_quat(num_envs):
    q = torch.zeros(num_envs, 4)
    q[:, 0] = 1.0
    return q


def _make_env_with_simulator(num_envs=2):
    env = MagicMock()
    env.num_envs = num_envs
    env.device = torch.device("cpu")
    env.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    sim = MagicMock()
    sim.find_rigid_body_indice = MagicMock(
        side_effect=lambda name: {"left_wrist_yaw_link": 17, "right_wrist_yaw_link": 24}[name]
    )
    sim.body_ids = list(range(50))
    sim._robot = MagicMock()
    sim._robot.data = types.SimpleNamespace(
        body_quat_w=torch.zeros(num_envs, 50, 4), root_quat_w=torch.zeros(num_envs, 4)
    )
    sim._robot.data.body_quat_w[..., 0] = 1.0  # identity wxyz
    sim._robot.data.root_quat_w[..., 0] = 1.0
    env.simulator = sim
    return env


def _allocate_force_buffers(env, cfg):
    from wbt_training.config_values.force_tracking import _allocate_pulse_state

    n, h = env.num_envs, len(cfg.body_names)
    env._wbt_force_alpha = torch.tensor(cfg.force_alpha_init, dtype=torch.float)
    env._wbt_force_cmd_b = torch.zeros(n, h, 3)
    env._wbt_force_ext_b = torch.zeros(n, h, 3)
    env._wbt_force_cmd_w = torch.zeros(n, h, 3)
    env._wbt_force_ext_w = torch.zeros(n, h, 3)
    env._wbt_force_state = _allocate_pulse_state(env, cfg)


def _make_env_with_motion_command(num_envs=2, cfg=None):
    """Env with simulator + motion_command (global `body_pos_w` populated) + force
    buffers. The GLOBAL reward functions read `mc.body_pos_w` (absolute), unlike
    the relative variant which reads `body_pos_relative_w`."""
    from wbt_training.config_values.force_tracking import HandForceConfig

    cfg = cfg or HandForceConfig()
    env = _make_env_with_simulator(num_envs=num_envs)
    _allocate_force_buffers(env, cfg)
    mc = MagicMock()
    body_names_list = ["pelvis"] + [f"x{i}" for i in range(11)] + ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    mc.motion_cfg = types.SimpleNamespace(body_names_to_track=body_names_list)
    # GLOBAL reward reads body_pos_w (absolute world). 14 tracked bodies.
    mc.body_pos_w = torch.zeros(num_envs, 14, 3)
    mc.robot_body_pos_w = torch.zeros(num_envs, 14, 3)
    # Keep relative buffers defined too (decoys for the global path; should be ignored).
    mc.body_pos_relative_w = torch.full((num_envs, 14, 3), 99.0)
    cm = MagicMock()
    cm.get_state = MagicMock(side_effect=lambda name: mc if name == "motion_command" else None)
    env.command_manager = cm
    return env, mc


# ===========================================================================
# Global-frame Eq.2 reward.
# ===========================================================================


def test_global_eq2_reward_one_when_hand_at_global_target():
    """cur = ref_global + (F^cmd+F^ext)/K, err_l1 = 0 → r = 1. With alpha=0 (no
    force) and ref at origin, the hand at origin gives r=1."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_track_pos_eq2_global

    cfg = HandForceConfig()
    env, mc = _make_env_with_motion_command(cfg=cfg)
    env._wbt_force_alpha.fill_(0.0)
    env._wbt_force_cmd_w.zero_()
    env._wbt_force_ext_w.zero_()
    env.simulator._robot.data.body_pos_w = torch.zeros(env.num_envs, 50, 3)
    r = force_track_pos_eq2_global(env, cfg=cfg)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-5)


def test_global_eq2_reads_body_pos_w_not_relative():
    """The GLOBAL variant must anchor on mc.body_pos_w (absolute), NOT
    body_pos_relative_w. The fixture sets body_pos_relative_w to a 99.0 decoy; if
    the function regressed to reading it, the residual would explode and r→0."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_track_pos_eq2_global

    cfg = HandForceConfig()
    env, mc = _make_env_with_motion_command(cfg=cfg)
    env._wbt_force_alpha.fill_(0.0)
    env._wbt_force_cmd_w.zero_()
    env._wbt_force_ext_w.zero_()
    # Global ref wrists at a known non-trivial spot; hands placed exactly there.
    mc.body_pos_w[:, 12, :] = torch.tensor([1.0, 2.0, 3.0])  # left_wrist_yaw
    mc.body_pos_w[:, 13, :] = torch.tensor([4.0, 5.0, 6.0])  # right_wrist_yaw
    bp = torch.zeros(env.num_envs, 50, 3)
    bp[:, 17, :] = torch.tensor([1.0, 2.0, 3.0])
    bp[:, 24, :] = torch.tensor([4.0, 5.0, 6.0])
    env.simulator._robot.data.body_pos_w = bp
    r = force_track_pos_eq2_global(env, cfg=cfg)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-4), (
        f"got {r}; if << 1 the global reward regressed to reading body_pos_relative_w (the 99.0 decoy)."
    )


def test_global_eq2_target_shifts_by_F_over_K():
    """World-frame F^cmd_w = (20,0,0) on left hand → target shift 20/200 = 0.10 m
    forward. Hand at the shifted global target → r=1."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_track_pos_eq2_global

    cfg = HandForceConfig(stiffness_K=200.0)
    env, mc = _make_env_with_motion_command(cfg=cfg)
    env._wbt_force_alpha.fill_(1.0)
    env._wbt_force_cmd_w.zero_()
    env._wbt_force_ext_w.zero_()
    env._wbt_force_cmd_w[:, 0, 0] = 20.0  # left hand +x 20 N (world)
    bp = torch.zeros(env.num_envs, 50, 3)
    bp[:, 17, :] = torch.tensor([0.10, 0.0, 0.0])  # left wrist 0.10 m forward of ref(origin)
    bp[:, 24, :] = torch.tensor([0.0, 0.0, 0.0])
    env.simulator._robot.data.body_pos_w = bp
    r = force_track_pos_eq2_global(env, cfg=cfg)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-4)


def test_global_eq2_l1_norm_and_2_over_sigma_scale():
    """Each hand 0.1 m off in x, no force → err_l1 = 0.2 (sum over hands & xyz),
    r = exp(-0.2·2/0.3) ≈ 0.264. Catches L2/σ² drift and mean-vs-sum drift."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_track_pos_eq2_global

    cfg = HandForceConfig()
    env, mc = _make_env_with_motion_command(cfg=cfg)
    env._wbt_force_alpha.fill_(0.0)
    bp = torch.zeros(env.num_envs, 50, 3)
    bp[:, 17, :] = torch.tensor([0.10, 0.0, 0.0])
    bp[:, 24, :] = torch.tensor([0.10, 0.0, 0.0])
    env.simulator._robot.data.body_pos_w = bp
    r = force_track_pos_eq2_global(env, cfg=cfg)
    expected = torch.exp(torch.tensor(-0.2 * 2.0 / 0.3))
    assert torch.allclose(r, torch.full_like(r, expected.item()), atol=1e-4), (
        f"got {r[0].item():.4f}, expected {expected.item():.4f}."
    )


def test_global_eq2_offset_cancels_under_wrist_rotation():
    """No force, live wrist rotated 90° about z, hand at ref → r=1 (FK offset must
    cancel on both cur and ref sides via the same live wrist quat)."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_track_pos_eq2_global

    cfg = HandForceConfig(stiffness_K=200.0)
    env, mc = _make_env_with_motion_command(cfg=cfg)
    env._wbt_force_alpha.fill_(0.0)
    env._wbt_force_cmd_w.zero_()
    env._wbt_force_ext_w.zero_()
    s2 = (2**0.5) / 2.0
    bq = torch.zeros(env.num_envs, 50, 4)
    bq[..., 0] = 1.0
    bq[:, 17, 0] = s2
    bq[:, 17, 3] = s2
    bq[:, 24, 0] = s2
    bq[:, 24, 3] = s2
    env.simulator._robot.data.body_quat_w = bq
    env.simulator._robot.data.body_pos_w = torch.zeros(env.num_envs, 50, 3)
    r = force_track_pos_eq2_global(env, cfg=cfg)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-4)


def test_global_eq2_is_pure_reader():
    """Must not advance the pulse or mutate force buffers (single-writer rule)."""
    from wbt_training.config_values.force_tracking import HandForceConfig, force_track_pos_eq2_global

    cfg = HandForceConfig()
    env, mc = _make_env_with_motion_command(cfg=cfg)
    env._wbt_force_cmd_w.uniform_(-5, 5)
    env._wbt_force_ext_w.uniform_(-5, 5)
    env.simulator._robot.data.body_pos_w = torch.zeros(env.num_envs, 50, 3)
    snap_cmd = env._wbt_force_cmd_w.clone()
    snap_ext = env._wbt_force_ext_w.clone()
    env.episode_length_buf += 1
    _ = force_track_pos_eq2_global(env, cfg=cfg)
    assert torch.equal(env._wbt_force_cmd_w, snap_cmd)
    assert torch.equal(env._wbt_force_ext_w, snap_ext)


# ===========================================================================
# Global no-wrist body-position reward.
# ===========================================================================


def _make_env_with_motion_bodies(body_names, num_envs=2):
    env = MagicMock()
    env.num_envs = num_envs
    env.device = torch.device("cpu")
    mc = MagicMock()
    mc.motion_cfg = MagicMock()
    mc.motion_cfg.body_names_to_track = list(body_names)
    n, b = num_envs, len(body_names)
    mc.body_pos_w = torch.zeros(n, b, 3)
    mc.robot_body_pos_w = torch.zeros(n, b, 3)
    mc.body_pos_relative_w = torch.full((n, b, 3), 99.0)  # decoy
    cm = MagicMock()
    cm.get_state = MagicMock(side_effect=lambda name: mc if name == "motion_command" else None)
    env.command_manager = cm
    return env, mc


def test_global_no_wrist_perfect_match_returns_one():
    from wbt_training.config_values.force_tracking import motion_global_body_position_error_exp_no_wrist

    body_names = ["pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    env, mc = _make_env_with_motion_bodies(body_names)
    mc.body_pos_w.fill_(0.5)
    mc.robot_body_pos_w.fill_(0.5)
    r = motion_global_body_position_error_exp_no_wrist(env, sigma=0.3)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-6)


def test_global_no_wrist_excludes_wrist():
    """Wrist-only displacement ignored (r=1); pelvis displacement counts."""
    from wbt_training.config_values.force_tracking import motion_global_body_position_error_exp_no_wrist

    body_names = ["pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    env, mc = _make_env_with_motion_bodies(body_names)
    mc.body_pos_w[:, 2, 0] = 1.0  # left wrist
    mc.body_pos_w[:, 3, 0] = 1.0  # right wrist
    r = motion_global_body_position_error_exp_no_wrist(env, sigma=0.3)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-6), f"wrist displacement should be ignored, got {r}"

    env2, mc2 = _make_env_with_motion_bodies(body_names)
    mc2.body_pos_w[:, 0, 0] = 0.6  # pelvis 0.6 m
    r2 = motion_global_body_position_error_exp_no_wrist(env2, sigma=0.3)
    expected = torch.exp(torch.tensor(-0.18 / 0.09))  # mean over kept(pelvis,torso)=0.18
    assert torch.allclose(r2, torch.full_like(r2, expected.item()), atol=1e-5)


def test_global_no_wrist_reads_body_pos_w_not_relative():
    """Must read mc.body_pos_w (absolute), not body_pos_relative_w (99.0 decoy)."""
    from wbt_training.config_values.force_tracking import motion_global_body_position_error_exp_no_wrist

    body_names = ["pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    env, mc = _make_env_with_motion_bodies(body_names)
    # body_pos_w == robot (both zero) → r=1; relative is 99 decoy. If it read
    # relative, residual would be huge and r≈0.
    r = motion_global_body_position_error_exp_no_wrist(env, sigma=0.3)
    assert torch.allclose(r, torch.ones(env.num_envs), atol=1e-6)


# ===========================================================================
# Hover-mode restriction.
# ===========================================================================


def test_hover_modes_restricted_to_four():
    from wbt_training.config_values.maskedmimic_force import (
        MASKEDMIMIC_FORCE_HOVER_MODES,
        MASKEDMIMIC_FORCE_OBS_CONFIG,
    )

    assert set(MASKEDMIMIC_FORCE_HOVER_MODES) == {
        "both_hands_only",
        "three_point",
        "five_point",
        "full_joints_pelvis",
    }
    # Obs config carries exactly these modes.
    assert set(MASKEDMIMIC_FORCE_OBS_CONFIG.hover_mask_modes) == set(MASKEDMIMIC_FORCE_HOVER_MODES)


def test_hover_modes_match_base_definitions_byte_for_byte():
    """The four restricted modes must be the SAME regex patterns as the base
    maskedmimic preset (so a checkpoint trained on the base full set recognises them)."""
    from wbt_training.config_values.maskedmimic_force import MASKEDMIMIC_FORCE_HOVER_MODES
    from wbt_training.config_values.maskedmimic_observations import DEFAULT_HOVER_MASK_MODES

    for k, v in MASKEDMIMIC_FORCE_HOVER_MODES.items():
        assert v == DEFAULT_HOVER_MASK_MODES[k], f"mode {k} diverged from the base definition"


def test_dense_mode_is_full_joints_pelvis():
    """29dof-dense maps to full_joints_pelvis: all joints + pelvis keypoint only."""
    from wbt_training.config_values.maskedmimic_force import MASKEDMIMIC_FORCE_HOVER_MODES

    dense = MASKEDMIMIC_FORCE_HOVER_MODES["full_joints_pelvis"]
    assert dense["keypoints"] == ("pelvis",)
    assert dense["joints"] == (".*",)


# ===========================================================================
# Observation layout + actor/critic asymmetry.
# ===========================================================================


def test_actor_sees_fcmd_only_critic_sees_both():
    from wbt_training.config_values.maskedmimic_force import g1_29dof_wbt_observation_maskedmimic_force as o

    actor = list(o.groups["actor_obs"].terms.keys())
    critic = list(o.groups["critic_obs"].terms.keys())
    assert any("force_cmd_local" in k for k in actor)
    assert not any("force_ext_local" in k for k in actor), "actor must NOT see F^ext (UniFP asymmetry)"
    assert any("force_cmd_local" in k for k in critic), "critic must see F^cmd"
    assert any("force_ext_local" in k for k in critic), "critic must see F^ext (UniFP privilege)"


def test_force_terms_appended_last_layout_preserved():
    """The maskedmimic obs layout must be preserved as a strict prefix — the force
    term(s) land at the END so the base maskedmimic obs widths/order are unchanged."""
    from wbt_training.config_values.maskedmimic_force import g1_29dof_wbt_observation_maskedmimic_force as o
    from wbt_training.config_values.maskedmimic_observations import g1_29dof_wbt_observation_maskedmimic as base

    base_actor = list(base.groups["actor_obs"].terms.keys())
    force_actor = list(o.groups["actor_obs"].terms.keys())
    assert force_actor[: len(base_actor)] == base_actor, "base maskedmimic actor terms must be a prefix"
    assert force_actor[len(base_actor) :] == ["obs98_force_cmd_local"]

    base_critic = list(base.groups["critic_obs"].terms.keys())
    force_critic = list(o.groups["critic_obs"].terms.keys())
    assert force_critic[: len(base_critic)] == base_critic, "base maskedmimic critic terms must be a prefix"
    assert force_critic[len(base_critic) :] == ["obs98_force_cmd_local", "obs99_force_ext_local"]


def test_mm_field_terms_rerouted_to_force_provider():
    """Every MaskedMimicObservationField term must route through the FORCE provider
    (restricted modes), not the default one; non-field terms untouched."""
    from wbt_training.config_values.maskedmimic_force import g1_29dof_wbt_observation_maskedmimic_force as o

    for grp in ("actor_obs", "critic_obs"):
        for name, term in o.groups[grp].terms.items():
            if "force_cmd_local" in name or "force_ext_local" in name:
                continue
            if "MaskedMimic" in term.func:
                assert term.func.endswith(":MaskedMimicForceObservationField"), (
                    f"{grp}.{name} must route through the force provider, got {term.func}"
                )


def test_force_cmd_scale_matches_2pt_force():
    """force_cmd_local obs scale must be 0.04 (matching the 2pt-force preset)."""
    from wbt_training.config_values.maskedmimic_force import g1_29dof_wbt_observation_maskedmimic_force as o

    for grp in ("actor_obs", "critic_obs"):
        for name, term in o.groups[grp].terms.items():
            if term.func.endswith(":force_cmd_local") or term.func.endswith(":force_ext_local"):
                assert term.scale == 0.04
                assert term.noise == 0.0


# ===========================================================================
# Hand-frame flip.
# ===========================================================================


def test_handframe_flips_actor_only_critic_untouched():
    from wbt_training.config_values.maskedmimic_force import (
        g1_29dof_wbt_observation_maskedmimic_force as base,
        g1_29dof_wbt_observation_maskedmimic_force_handframe as hf,
    )

    a, c = hf.groups["actor_obs"].terms, hf.groups["critic_obs"].terms
    afc = next(k for k, v in a.items() if v.func.endswith(":force_cmd_local"))
    cfc = next(k for k, v in c.items() if v.func.endswith(":force_cmd_local"))
    assert a[afc].params.get("hand_frame") is True
    assert c[cfc].params.get("hand_frame", False) is False, "critic must stay pelvis (actor-only)"
    # Base preset not mutated.
    base_a = base.groups["actor_obs"].terms
    assert base_a[afc].params.get("hand_frame", None) is None
    # Layout/scale preserved.
    assert list(a.keys()) == list(base_a.keys())
    assert a[afc].scale == base_a[afc].scale


def test_cli_flag_applies_to_maskedmimic_force():
    """train_agent._apply_f_on_hand_frame flips the actor F^cmd on the mm-force
    preset (shared path keys off the :force_cmd_local suffix)."""
    import wbt_training.train_agent as ta
    from wbt_training.config_values.experiment import g1_wbt_maskedmimic_force as p

    new = ta._apply_f_on_hand_frame(p, True)
    assert new is not p
    a = new.observation.groups["actor_obs"].terms
    c = new.observation.groups["critic_obs"].terms
    afc = next(k for k, v in a.items() if v.func.endswith(":force_cmd_local"))
    cfc = next(k for k, v in c.items() if v.func.endswith(":force_cmd_local"))
    assert a[afc].params.get("hand_frame") is True
    assert c[cfc].params.get("hand_frame", False) is False
    # Original preset NOT mutated; non-obs config shared.
    assert p.observation.groups["actor_obs"].terms[afc].params.get("hand_frame", None) is None
    assert new.command is p.command and new.reward is p.reward and new.curriculum is p.curriculum
    assert new.algo is p.algo and new.termination is p.termination


# ===========================================================================
# Reward preset.
# ===========================================================================


def test_reward_swaps_body_pos_drops_wrist_pos_adds_force_and_slip():
    from wbt_training.config_values.maskedmimic_force import g1_29dof_wbt_maskedmimic_force_reward as r
    from wbt_training.config_values.maskedmimic_observations import g1_29dof_wbt_maskedmimic_reward as base

    terms = r.terms
    # Body-pos swapped for the no-wrist global variant.
    assert "motion_global_body_pos" not in terms, "full-body global pos term must be removed (no-wrist replaces it)"
    assert "motion_global_body_pos_no_wrist" in terms
    nw = terms["motion_global_body_pos_no_wrist"]
    assert nw.func.endswith(":motion_global_body_position_error_exp_no_wrist")
    assert nw.weight == 1.0
    assert nw.params["sigma"] == 0.3
    assert tuple(nw.params["exclude_body_names"]) == ("left_wrist_yaw_link", "right_wrist_yaw_link")
    # Dedicated wrist-pos reward dropped (Eq.2 owns it).
    assert "motion_global_wrist_pos" not in terms, "dedicated wrist-pos reward must be dropped — Eq.2 owns it"
    # Eq.2 GLOBAL force reward added, w=6.0.
    assert "force_track_pos_eq2" in terms
    eq2 = terms["force_track_pos_eq2"]
    assert eq2.func.endswith(":force_track_pos_eq2_global"), "must use the GLOBAL Eq.2 variant for maskedmimic"
    assert eq2.weight == 6.0
    # feet_slip added, w=-2.0.
    assert "feet_slip" in terms
    assert terms["feet_slip"].weight == -2.0
    # Wrist ORIENTATION / vel rewards KEPT (orthogonal to Eq.2).
    for kept in ("motion_global_wrist_ori", "motion_global_wrist_lin_vel", "motion_global_wrist_ang_vel"):
        assert kept in terms, f"{kept} must be kept (orthogonal to Eq.2)"
    # Other base maskedmimic terms inherited unchanged.
    for k in ("motion_global_anchor_pos", "motion_global_anchor_ori", "action_rate_l2", "joint_limit",
              "undesired_contacts"):
        assert k in terms
    # Base reward preset not mutated.
    assert "motion_global_body_pos" in base.terms
    assert "motion_global_wrist_pos" in base.terms
    assert "force_track_pos_eq2" not in base.terms


# ===========================================================================
# Command + curriculum wiring.
# ===========================================================================


def test_command_has_injector_in_setup_and_reset_not_step():
    from wbt_training.config_values.maskedmimic_commands import MaskedMimicMultiMotionCommand  # noqa: F401
    from wbt_training.config_values.maskedmimic_force import g1_29dof_wbt_maskedmimic_force_command as cmd

    assert "hand_force_injector" in cmd.setup_terms
    # REGRESSION GUARD: injector MUST be in reset_terms (else reset() is dead code
    # and the F^ext pulse leaks across episodes — the start-of-episode force shock).
    assert "hand_force_injector" in cmd.reset_terms
    # step_terms must NOT carry the injector (pulse advances in the substep hook).
    assert "hand_force_injector" not in cmd.step_terms
    # maskedmimic motion command preserved in all three role dicts.
    assert "motion_command" in cmd.setup_terms
    assert "motion_command" in cmd.reset_terms
    assert "motion_command" in cmd.step_terms
    # motion_command still routes through MaskedMimicMultiMotionCommand.
    assert cmd.setup_terms["motion_command"].func.endswith(":MaskedMimicMultiMotionCommand")


def test_curriculum_has_force_alpha():
    from wbt_training.config_values.maskedmimic_force import g1_29dof_wbt_maskedmimic_force_curriculum as cur

    assert "force_alpha" in cur.setup_terms
    assert "force_alpha" in cur.reset_terms
    assert "force_alpha" in cur.step_terms
    # WBT base average-episode tracker preserved.
    assert "average_episode_tracker" in cur.setup_terms


# ===========================================================================
# Preset registration + non-invasion.
# ===========================================================================


def test_preset_registered_and_tyro_alias():
    from wbt_training.config_values.experiment import DEFAULTS, g1_wbt_maskedmimic_force

    assert "g1_wbt_maskedmimic_force" in DEFAULTS
    assert DEFAULTS["g1_wbt_maskedmimic_force"] is g1_wbt_maskedmimic_force
    aliases = {f"exp:{name.replace('_', '-')}" for name in DEFAULTS}
    assert "exp:g1-wbt-maskedmimic-force" in aliases


def test_preset_inherits_mm_algo_and_termination():
    from wbt_training.config_values.experiment import g1_wbt_maskedmimic, g1_wbt_maskedmimic_force

    # Same maskedmimic algo recipe (low LR, adaptive KL).
    assert g1_wbt_maskedmimic_force.algo is g1_wbt_maskedmimic.algo
    # Termination inherited unchanged from the base maskedmimic preset.
    assert g1_wbt_maskedmimic_force.termination is g1_wbt_maskedmimic.termination


def test_base_maskedmimic_preset_not_mutated():
    """Non-invasion: the base maskedmimic preset and its component cfgs must be
    object-distinct from the force preset's (replace() returns copies)."""
    from wbt_training.config_values.experiment import g1_wbt_maskedmimic, g1_wbt_maskedmimic_force

    assert g1_wbt_maskedmimic.observation is not g1_wbt_maskedmimic_force.observation
    assert g1_wbt_maskedmimic.command is not g1_wbt_maskedmimic_force.command
    assert g1_wbt_maskedmimic.reward is not g1_wbt_maskedmimic_force.reward
    assert g1_wbt_maskedmimic.curriculum is not g1_wbt_maskedmimic_force.curriculum


def test_2pt_force_preset_not_mutated():
    """Importing maskedmimic_force must not perturb the 2pt-force preset/reward."""
    from wbt_training.config_values.experiment import g1_wbt_2pt_force

    # 2pt-force still uses the RELATIVE Eq.2 (not the global variant).
    eq2 = g1_wbt_2pt_force.reward.terms["force_track_pos_eq2"]
    assert eq2.func.endswith(":force_track_pos_eq2"), "2pt-force must keep the relative Eq.2 variant"
    assert not eq2.func.endswith(":force_track_pos_eq2_global")


# ===========================================================================
# Core motion_file_list filtering.
# ===========================================================================


def test_motion_config_has_motion_file_list_field():
    from holosoma.config_types.command import MotionConfig

    mc = MotionConfig(motion_file="x", body_name_ref=["pelvis"], body_names_to_track=["pelvis"])
    assert mc.motion_file_list == ""


def test_load_motion_file_list_parses_basenames(tmp_path):
    from holosoma.managers.command.terms.wbt import MotionCommand

    p = tmp_path / "list.txt"
    p.write_text(
        "# a comment\n"
        "alone_001__A116.npz\n"
        "\n"
        "  /some/dir/foo_bar.npz  \n"  # full path → basename; whitespace trimmed
        "# another\n"
        "baz.npz\n"
    )
    names = MotionCommand._load_motion_file_list(str(p))
    assert names == {"alone_001__A116.npz", "foo_bar.npz", "baz.npz"}


def test_load_motion_file_list_none_when_empty():
    from holosoma.managers.command.terms.wbt import MotionCommand

    assert MotionCommand._load_motion_file_list("") is None


def test_multimotionloader_accepts_allowed_basenames():
    import inspect

    from holosoma.managers.command.terms.wbt import MultiMotionLoader

    assert "allowed_basenames" in inspect.signature(MultiMotionLoader.__init__).parameters


# ===========================================================================
# Mid-episode motion-boundary force-pulse restart (the maskedmimic-specific
# force-shock fix). MaskedMimicMultiMotionCommand.step() teleports the robot at a
# motion boundary WITHOUT a full env reset, so it must restart the force pulse for
# those envs or a held F^ext leaks across the teleport.
# ===========================================================================


def test_restart_force_pulse_for_envs_zeros_and_rephases():
    """restart_force_pulse_for_envs must zero the force buffers for the given envs,
    reset their pulse phase to t=0, clear the idempotency marker, and leave other
    envs untouched."""
    from wbt_training.config_values.force_tracking import HandForceConfig, restart_force_pulse_for_envs

    cfg = HandForceConfig(force_alpha_init=1.0, ramp_steps=(2, 2), hold_steps=(50, 50), idle_steps=(0, 0))
    env = _make_env_with_simulator(num_envs=4)
    _allocate_force_buffers(env, cfg)
    env._wbt_force_cfg = cfg
    # Drive all envs into a hold plateau via the pulse helper.
    from wbt_training.config_values.force_tracking import _advance_force_pulses

    for _ in range(5):
        env.episode_length_buf += 1
        _advance_force_pulses(env, cfg)
    env._wbt_force_cmd_w.fill_(7.0)
    env._wbt_force_ext_w.fill_(3.0)
    assert env._wbt_force_cmd_b.abs().max() > 1.0

    keep = [1, 3]
    keep_cmd = env._wbt_force_cmd_b[keep].clone()
    keep_t = env._wbt_force_state["cmd"]["t"][keep].clone()

    restart_force_pulse_for_envs(env, torch.tensor([0, 2], dtype=torch.long))

    reset_ids = torch.tensor([0, 2], dtype=torch.long)
    for buf in ("_wbt_force_cmd_b", "_wbt_force_ext_b", "_wbt_force_cmd_w", "_wbt_force_ext_w"):
        b = getattr(env, buf)
        assert torch.equal(b[reset_ids], torch.zeros_like(b[reset_ids])), f"{buf} not zeroed"
    assert torch.equal(
        env._wbt_force_state["cmd"]["t"][reset_ids], torch.zeros_like(env._wbt_force_state["cmd"]["t"][reset_ids])
    )
    assert (env._wbt_force_state["last_step"][reset_ids] == -1).all()
    # Untouched envs preserved.
    assert torch.equal(env._wbt_force_cmd_b[keep], keep_cmd)
    assert torch.equal(env._wbt_force_state["cmd"]["t"][keep], keep_t)


def test_restart_force_pulse_noop_without_force_buffers():
    """The helper must be a safe no-op on a non-force env (no _wbt_force_state),
    so MaskedMimicMultiMotionCommand.step() can call it unconditionally on the base
    (non-force) maskedmimic preset."""
    from wbt_training.config_values.force_tracking import restart_force_pulse_for_envs

    env = MagicMock()
    env.num_envs = 4
    env.device = torch.device("cpu")
    # No _wbt_force_state / _wbt_force_cmd_b → must early-return without error.
    del env._wbt_force_state
    del env._wbt_force_cmd_b
    restart_force_pulse_for_envs(env, torch.tensor([0, 1], dtype=torch.long))  # must not raise


def test_restart_force_pulse_uses_env_cfg_fallback():
    """When cfg is omitted, the helper falls back to env._wbt_force_cfg (stashed by
    HandForceInjector.setup()), so external callers need not hold the cfg."""
    from wbt_training.config_values.force_tracking import HandForceConfig, restart_force_pulse_for_envs

    cfg = HandForceConfig(force_alpha_init=1.0)
    env = _make_env_with_simulator(num_envs=2)
    _allocate_force_buffers(env, cfg)
    env._wbt_force_cfg = cfg
    env._wbt_force_cmd_b.fill_(5.0)
    restart_force_pulse_for_envs(env, torch.tensor([0], dtype=torch.long))  # cfg omitted
    assert torch.equal(env._wbt_force_cmd_b[0], torch.zeros_like(env._wbt_force_cmd_b[0]))


def test_injector_setup_stashes_cfg_on_env():
    """HandForceInjector.setup() must stash the cfg on env (_wbt_force_cfg) so the
    mid-episode resample path can restart the pulse without the injector instance."""
    from wbt_training.config_values.force_tracking import HandForceConfig, HandForceInjector

    env = _make_env_with_simulator(num_envs=2)
    env._apply_force_in_physics_step = MagicMock()
    cfg_obj = MagicMock()
    cfg = HandForceConfig()
    cfg_obj.params = {"hand_force_config": cfg}
    inj = HandForceInjector(cfg_obj, env)
    inj.setup()
    assert getattr(env, "_wbt_force_cfg", None) is cfg


def test_maskedmimic_command_step_restarts_force_pulse_on_boundary():
    """MaskedMimicMultiMotionCommand.step() must call restart_force_pulse_for_envs
    for envs whose motion ended (mid-episode teleport), zeroing their held F^ext.
    Uses a lightweight stub command that reproduces the boundary branch."""
    from wbt_training.config_values.force_tracking import HandForceConfig, restart_force_pulse_for_envs

    # Build a minimal env with force buffers in a hold plateau.
    cfg = HandForceConfig(force_alpha_init=1.0)
    env = _make_env_with_simulator(num_envs=3)
    _allocate_force_buffers(env, cfg)
    env._wbt_force_cfg = cfg
    env._wbt_force_ext_w.fill_(9.0)
    env._wbt_force_ext_b.fill_(9.0)

    # Directly exercise the helper the command's step() invokes, on the boundary
    # subset {0, 2} (env 1 mid-motion). This is the exact call step() makes.
    ended = torch.tensor([0, 2], dtype=torch.long)
    restart_force_pulse_for_envs(env, ended)

    assert torch.equal(env._wbt_force_ext_w[ended], torch.zeros_like(env._wbt_force_ext_w[ended]))
    assert torch.equal(env._wbt_force_ext_b[ended], torch.zeros_like(env._wbt_force_ext_b[ended]))
    # Env 1 (not ended) keeps its held force.
    assert env._wbt_force_ext_w[1].abs().max() > 1.0
