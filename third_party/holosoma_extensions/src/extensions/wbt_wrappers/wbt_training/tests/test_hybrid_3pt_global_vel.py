"""Unit tests for exp:g1-wbt-3pt-hybrid-global-vel (3pt-hybrid + global root pos/vel + 6D orn).

Covers the genuinely-new pieces (everything else is reused verbatim from hybrid.py / hybrid_3pt.py):
  - the 183D actor observation layout (168D 3pt-hybrid + motion_ref_pos_b/vel_b + base_lin_vel,
    with vr_3point_orn swapped quat(12)→6D(18)), zero-padded sorted keys obs00..obs15, and its
    FIXED width across servo on/off + dropout on/off;
  - the ONE new obs func vr_3point_local_orn_target_6d = 3 bodies × 6 = 18D via the shared
    return_6d helper (the quaternion vr_3point_orn stays 12D — 3pt-hybrid untouched);
  - the root-state trio reused verbatim from exp:g1-wbt-3pt-global-vel's obs terms;
  - the ACTOR joint-velocity dropout term reused verbatim from hybrid_3pt (default OFF);
  - the experiment presets (yam URDF, servo shipped ON + estop, servo-off two-phase parity,
    reward/command/curriculum/termination reused verbatim from hybrid.py);
  - the inference + mjswan obs configs agree with the training layout (183D, same term order).

All tests use MagicMock / lightweight fakes — no IsaacSim runtime (mirrors test_hybrid_3pt.py)."""

from unittest.mock import MagicMock

import torch

# ===========================================================================
# 1. 183D actor observation layout.
# ===========================================================================

_EXPECTED_ACTOR = {
    "obs00_command_lower_body": 24,
    "obs01_vr_3point_pos": 9,
    "obs02_vr_3point_orn_6d": 18,
    "obs03_motion_ref_pos_b": 3,
    "obs04_motion_ref_vel_b": 3,
    "obs05_force_cmd_local": 6,
    "obs06_force_meas_local": 6,
    "obs07_integrator_state": 6,
    "obs08_x_tgt_shift": 6,
    "obs09_motion_ref_ori_b": 6,
    "obs10_projected_gravity": 3,
    "obs11_base_ang_vel": 3,
    "obs12_base_lin_vel": 3,
    "obs13_dof_pos": 29,
    "obs14_dof_vel": 29,
    "obs15_actions": 29,
}


def test_actor_obs_keys_and_order():
    from wbt_training.config_values.hybrid_3pt_global_vel import g1_29dof_wbt_observation_3pt_hybrid_global_vel as obs

    actor = obs.groups["actor_obs"]
    assert list(actor.terms.keys()) == list(_EXPECTED_ACTOR.keys())
    # The obs manager concatenates in sorted(keys()) order — insertion order MUST equal
    # sorted order (zero-padded prefixes), else the vector scrambles (obs10 vs obs1/obs2).
    assert list(actor.terms.keys()) == sorted(actor.terms.keys())
    assert actor.history_length == 4
    assert obs.groups["critic_obs"].history_length == 10


def test_actor_per_frame_dim_is_183():
    # The 183D contract, checked against the REAL obs dims (not just the test's own
    # literal): the training actor keys must equal _EXPECTED_ACTOR's keys AND the
    # inference obs_dims (the deploy contract, one dim per matching term) must sum to
    # 183 over those same terms — so this fails if either layer's real widths drift.
    from wbt_training.config_values.hybrid_3pt_global_vel import g1_29dof_wbt_observation_3pt_hybrid_global_vel as obs
    from wbt_wrappers_inference.config_values.observation import holosoma_3pt_hybrid_global_vel_wbt as iobs

    assert list(obs.groups["actor_obs"].terms.keys()) == list(_EXPECTED_ACTOR.keys())
    # inference obs_dims are the real per-term widths the ONNX consumes; map the
    # training prefixed key → the bare inference name (dof_pos/dof_vel → joint_pos/joint_vel).
    rename = {"dof_pos": "joint_pos", "dof_vel": "joint_vel"}
    real_total = 0
    for k in _EXPECTED_ACTOR:
        bare = k.split("_", 1)[1]
        bare = rename.get(bare, bare)
        assert iobs.obs_dims[bare] == _EXPECTED_ACTOR[k], (k, bare, iobs.obs_dims[bare])
        real_total += iobs.obs_dims[bare]
    assert real_total == 183


def test_orientation_is_6d_not_quaternion():
    from wbt_training.config_values.hybrid_3pt_global_vel import g1_29dof_wbt_observation_3pt_hybrid_global_vel as obs

    orn = obs.groups["actor_obs"].terms["obs02_vr_3point_orn_6d"]
    # 6D func (distinct name from the 12D quaternion vr_3point_orn) — 3 bodies × 6 = 18D.
    assert orn.func.endswith("hybrid_3pt_global_vel:vr_3point_local_orn_target_6d")
    assert orn.scale == 1.0 and orn.noise == 0.0


def test_root_terms_present_and_reused_from_global_vel():
    from wbt_training.config_values.hybrid_3pt_global_vel import g1_29dof_wbt_observation_3pt_hybrid_global_vel as obs
    from wbt_training.config_values.observation import _actor_obs_3pt_global_vel_terms as GV

    terms = obs.groups["actor_obs"].terms
    # The three global-vel root signals are the SAME ObsTermCfg objects as exp:g1-wbt-3pt-global-vel.
    assert terms["obs03_motion_ref_pos_b"] is GV["obs03_motion_ref_pos_b"]
    assert terms["obs04_motion_ref_vel_b"] is GV["obs04_motion_ref_vel_b"]
    assert terms["obs12_base_lin_vel"] is GV["obs08_base_lin_vel"]
    assert "motion_anchor_pos_b" in terms["obs03_motion_ref_pos_b"].func
    assert "motion_anchor_vel_b" in terms["obs04_motion_ref_vel_b"].func
    assert "base_lin_vel_b" in terms["obs12_base_lin_vel"].func


def test_force_and_servo_terms_reused_from_3pt_hybrid():
    from wbt_training.config_values.hybrid_3pt import _actor_obs_3pt_hybrid_terms as H3
    from wbt_training.config_values.hybrid_3pt_global_vel import g1_29dof_wbt_observation_3pt_hybrid_global_vel as obs

    terms = obs.groups["actor_obs"].terms
    # Force/servo channels are the SAME ObsTermCfg objects as the 3pt-hybrid layout.
    assert terms["obs05_force_cmd_local"] is H3["obs03_force_cmd_local"]
    assert terms["obs06_force_meas_local"] is H3["obs04_force_meas_local"]
    assert terms["obs07_integrator_state"] is H3["obs05_integrator_state"]
    assert terms["obs08_x_tgt_shift"] is H3["obs06_x_tgt_shift"]
    assert terms["obs05_force_cmd_local"].scale == 0.04
    assert terms["obs06_force_meas_local"].scale == 0.04


def test_dof_vel_term_is_the_dropout_term():
    from wbt_training.config_values.hybrid_3pt_global_vel import g1_29dof_wbt_observation_3pt_hybrid_global_vel as obs

    dv = obs.groups["actor_obs"].terms["obs14_dof_vel"]
    # Reused verbatim from hybrid_3pt: the stateful JointVelDropout term (base 0.5 noise
    # applied INTERNALLY → ObsTermCfg noise MUST be 0.0). Default OFF.
    assert dv.func.endswith("hybrid_3pt:JointVelDropout")
    assert dv.noise == 0.0
    assert dv.scale == 1.0


def test_existing_quaternion_vr_3point_orn_untouched():
    """The 6D preset must NOT change the existing 3pt-hybrid quaternion orientation term."""
    from wbt_training.config_values.hybrid_3pt import g1_29dof_wbt_observation_3pt_hybrid as h3

    orn = h3.groups["actor_obs"].terms["obs02_vr_3point_orn"]
    assert orn.func.endswith("observation:vr_3point_local_orn_target")  # still the 12D quat func


def test_critic_reused_from_2pt_hybrid_verbatim():
    from wbt_training.config_values.hybrid import g1_29dof_wbt_observation_2pt_hybrid as hy2
    from wbt_training.config_values.hybrid_3pt_global_vel import g1_29dof_wbt_observation_3pt_hybrid_global_vel as hyGV

    c2 = hy2.groups["critic_obs"].terms
    cGV = hyGV.groups["critic_obs"].terms
    assert set(c2) == set(cGV)
    assert all(cGV[k] is c2[k] for k in c2), "every critic term must be the SAME object (verbatim reuse)."
    # The critic already carries the privileged root signals (so no critic change is needed
    # to add the actor root obs — mirrors exp:g1-wbt-global-vel).
    assert "motion_ref_pos_b" in cGV
    assert "base_lin_vel" in cGV
    # Critic keeps a clean (non-dropout) dof_vel term.
    assert cGV["dof_vel"].func.endswith("wbt:dof_vel")


# ===========================================================================
# 2. The one new obs func: 6D 3-point orientation = 18D.
# ===========================================================================


def test_vr_3point_orn_6d_returns_18d(monkeypatch):
    """vr_3point_local_orn_target_6d delegates to the shared return_6d=True helper with the
    3 VR_3POINT_BODIES → 3 × 6 = 18D; identity anchor/bodies → [1,0,0,1,0,0] per body."""
    import wbt_training.config_values.hybrid_3pt_global_vel as G
    import wbt_training.config_values.observation as O
    from wbt_training.config_values.robot_config import VR_3POINT_BODIES

    n = 4

    class MC:
        motion_cfg = MagicMock(body_names_to_track=list(VR_3POINT_BODIES))
        ref_quat_w = torch.tensor([[1.0, 0, 0, 0]] * n)
        body_quat_w = torch.tensor([[[1.0, 0, 0, 0]] * 3] * n)

    env = MagicMock()
    env.num_envs = n
    env.command_manager.get_state.return_value = MC()
    # Make _get_motion_command's isinstance(mc, MotionCommand) pass.
    monkeypatch.setattr(O, "MotionCommand", MC)

    out = G.vr_3point_local_orn_target_6d(env)
    assert out.shape == (n, 18)
    assert out[0, :6].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # first two cols of identity


# ===========================================================================
# 3. Experiment presets (yam URDF, servo ON + estop, servo-off two-phase parity).
# ===========================================================================


def test_presets_registered_with_yam_urdf():
    from wbt_training.config_values.experiment import DEFAULTS

    for name in ("g1_wbt_3pt_hybrid_global_vel", "g1_wbt_3pt_hybrid_global_vel_servo_off"):
        assert name in DEFAULTS, name
        assert DEFAULTS[name].robot.asset.urdf_file == "g1/g1_29dof_yam.urdf"


def test_preset_ships_servo_enabled_with_estop():
    from wbt_training.config_values.experiment import DEFAULTS

    on = DEFAULTS["g1_wbt_3pt_hybrid_global_vel"]
    inj = on.command.setup_terms["hybrid_contact_injector"]
    assert inj.params["hybrid_config"].servo_enabled is True
    assert "force_runaway" in on.termination.terms


def test_servo_off_variant_flips_only_servo_and_keeps_obs_and_reward_keys():
    from wbt_training.config_values.experiment import DEFAULTS

    on = DEFAULTS["g1_wbt_3pt_hybrid_global_vel"]
    off = DEFAULTS["g1_wbt_3pt_hybrid_global_vel_servo_off"]
    assert off.command.setup_terms["hybrid_contact_injector"].params["hybrid_config"].servo_enabled is False
    # obs is the SAME manager (fixed 183D → checkpoint transfers with no surgery)
    assert on.observation is off.observation
    assert set(on.reward.terms.keys()) == set(off.reward.terms.keys())
    assert "force_runaway" in off.termination.terms


def test_preset_reuses_hybrid_reward_command_curriculum_termination_verbatim():
    """Root pos/vel is an OBSERVATION signal, not a reward change: the global-vel preset reuses
    hybrid.py's reward / command / curriculum / termination objects directly (like 3pt-hybrid)."""
    from wbt_training.config_values import hybrid as H
    from wbt_training.config_values.experiment import DEFAULTS

    on = DEFAULTS["g1_wbt_3pt_hybrid_global_vel"]
    assert on.reward is H.g1_29dof_wbt_2pt_hybrid_reward
    assert on.command is H.g1_29dof_wbt_hybrid_command
    assert on.curriculum is H.g1_29dof_wbt_hybrid_curriculum
    assert on.termination is H.g1_29dof_wbt_hybrid_termination


def test_preset_only_delta_vs_3pt_hybrid_is_observation():
    """The preset is replace(g1_wbt_3pt_hybrid, observation=...): everything but the obs is shared."""
    from wbt_training.config_values.experiment import DEFAULTS

    base = DEFAULTS["g1_wbt_3pt_hybrid"]
    gv = DEFAULTS["g1_wbt_3pt_hybrid_global_vel"]
    assert gv.observation is not base.observation  # obs differs (168D → 183D)
    assert gv.reward is base.reward
    assert gv.command is base.command
    assert gv.curriculum is base.curriculum
    assert gv.termination is base.termination
    assert gv.algo is base.algo


# ===========================================================================
# 4. Inference + mjswan obs configs agree with training (183D, same order).
# ===========================================================================


def test_inference_obs_config_matches_training_layout():
    from wbt_wrappers_inference.config_values.observation import holosoma_3pt_hybrid_global_vel_wbt as obs

    keys = obs.obs_dict["actor_obs"]
    # inference lists BARE names in the training sorted order; dof_pos/dof_vel → joint_pos/joint_vel.
    expected = [k.split("_", 1)[1] for k in _EXPECTED_ACTOR]
    expected = [{"dof_pos": "joint_pos", "dof_vel": "joint_vel"}.get(name, name) for name in expected]
    assert keys == expected, (keys, expected)
    assert sum(obs.obs_dims[k] for k in keys) == 183
    assert obs.obs_dims["vr_3point_orn_6d"] == 18
    assert obs.history_length_dict["actor_obs"] == 4
    assert obs.obs_scales["force_cmd_local"] == 0.04 and obs.obs_scales["force_meas_local"] == 0.04


def test_inference_detect_obs_mode_is_3pt_global():
    """detect_obs_mode must classify the 183D config as 3pt_global (has vr_3point + motion_ref_pos_b),
    so _is_3pt is True (FK runs on 3 bodies) and the runtime emits the 6D orn (via _vr_orn_6d)."""
    from wbt_wrappers_inference.config_values.observation import holosoma_3pt_hybrid_global_vel_wbt as obs
    from wbt_wrappers_inference.utils.obs_utils import detect_obs_mode

    assert detect_obs_mode(obs.obs_dict["actor_obs"]) == "3pt_global"


def test_inference_registry_maps_both_name_keys_to_same_config():
    from wbt_wrappers_inference.config_values import inference as I

    reg = I.DEFAULTS
    tgt = I.g1_29dof_holosoma_3pt_hybrid_global_vel_wbt
    assert reg["g1-29dof-holosoma-3pt-hybrid-global-vel-wbt"] is tgt
    assert reg["g1-29dof-holosoma-3pt-hybrid-global-vel-layerA-wbt"] is tgt


def test_mjswan_preset_matches_inference_order_and_dim():
    from wbt_realtime_eval.presets import get_preset
    from wbt_wrappers_inference.config_values.observation import holosoma_3pt_hybrid_global_vel_wbt as obs

    for nm in ("g1-29dof-holosoma-3pt-hybrid-global-vel-wbt", "g1-29dof-holosoma-3pt-hybrid-global-vel-layerA-wbt"):
        p = get_preset(nm)
        assert p.vr_family == "3pt"
        assert p.history_length == 4
        assert p.per_frame_dim == 183
        assert list(p.obs_keys) == obs.obs_dict["actor_obs"], (nm, list(p.obs_keys))


def test_mjswan_vr_3point_orn_6d_term_is_18d():
    from wbt_realtime_eval.presets import _TERMS

    assert _TERMS["vr_3point_orn_6d"].base_dim == 18
    assert _TERMS["vr_3point_orn_6d"].ts_name == "WbtVr3PointOrn6d"
