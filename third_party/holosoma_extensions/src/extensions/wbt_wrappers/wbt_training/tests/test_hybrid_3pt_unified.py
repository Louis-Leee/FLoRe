"""Unit tests for the LOCAL C1 unified twins (`exp:g1-wbt-3pt-hybrid-unified[-anneal]`).

The local twins = the R2/R1 unified recipe (observe-only integrator / servo-authority
anneal, soft contact_phase, unified reward, force-alpha ramp) applied to the ORIGINAL
168D pelvis-local `exp:g1-wbt-3pt-hybrid` base — NO root pos/vel trio, quaternion
vr_3point_orn (not 6D). Actor obs: 168D + contact_phase(2) = 170D.

Everything except the obs layout is the SAME carrier objects as the global-vel unified
presets (hybrid_unified.py) — the hybrid stack is tracking-point- and obs-agnostic.
All config/identity assertions — no IsaacSim runtime.
"""

import os

import pytest

# `WBT_HAND_MASS_DR=1` (phase 2 of the 2026-08-05 TCP precision work) deliberately swaps the
# local unified twins' randomization / robot / cfg carriers, so the phase-1 identity
# assertions below do not hold there. Skip rather than leave phase 2 with a red suite.
_phase1_only = pytest.mark.skipif(
    os.environ.get("WBT_HAND_MASS_DR", "0").strip().lower() in ("1", "true", "yes"),
    reason="phase-1 invariant; WBT_HAND_MASS_DR=1 intentionally changes these",
)

# ---------------------------------------------------------------------------
# 170D actor obs layout (168D 3pt-hybrid + contact_phase after force_meas_local;
# slots ≥05 shift +1). Zero-padded sorted keys obs00..obs13 (sorted-concat contract).
# ---------------------------------------------------------------------------
_EXPECTED_LOCAL_UNIFIED_ACTOR = {
    "obs00_command_lower_body": 24,
    # TCP-unified hand point (user 2026-08-03): both HANDS tracked at the 0.242 m
    # fingertip TCP (the force anchor) instead of the 0.18 m VR lever, so position and
    # force share ONE point. Distinct term name (not a re-parameterised vr_3point_pos)
    # so a checkpoint's config records which geometry it trained on.
    "obs01_vr_3point_pos_tcp": 9,
    "obs02_vr_3point_orn": 12,  # quaternion (3 bodies x 4) — NOT the 6D global-vel rep
    "obs03_force_cmd_local": 6,
    "obs04_force_meas_local": 6,
    "obs05_contact_phase": 2,
    "obs06_integrator_state": 6,
    "obs07_x_tgt_shift": 6,
    "obs08_motion_ref_ori_b": 6,
    "obs09_projected_gravity": 3,
    "obs10_base_ang_vel": 3,
    "obs11_dof_pos": 29,
    "obs12_dof_vel": 29,
    "obs13_actions": 29,
}  # = 170


def test_local_unified_actor_obs_layout_170d():
    from wbt_training.config_values.hybrid_unified import g1_29dof_wbt_observation_3pt_unified_hybrid as obs

    actor = obs.groups["actor_obs"]
    assert list(actor.terms.keys()) == list(_EXPECTED_LOCAL_UNIFIED_ACTOR.keys())
    assert list(actor.terms.keys()) == sorted(actor.terms.keys())  # sorted-concat contract
    assert sum(_EXPECTED_LOCAL_UNIFIED_ACTOR.values()) == 170
    assert actor.history_length == 4
    cp = actor.terms["obs05_contact_phase"]
    assert cp.func == "wbt_training.config_values.hybrid:contact_phase"
    assert cp.scale == 1.0 and cp.noise == 0.0
    # contact_phase is the SAME object as the 185D layout's (a value-equal duplicate
    # could silently diverge if the term ever gains params — review 2026-07-24).
    from wbt_training.config_values.hybrid_unified import g1_29dof_wbt_observation_3pt_unified_hybrid_global_vel

    assert cp is g1_29dof_wbt_observation_3pt_unified_hybrid_global_vel.groups["actor_obs"].terms["obs07_contact_phase"]


def test_local_unified_reuses_3pt_hybrid_terms_verbatim():
    from wbt_training.config_values.hybrid_3pt import _actor_obs_3pt_hybrid_terms as base
    from wbt_training.config_values.hybrid_unified import g1_29dof_wbt_observation_3pt_unified_hybrid as obs

    actor = obs.groups["actor_obs"].terms
    # every non-contact_phase, non-TCP-position term is the SAME ObsTermCfg object as the
    # 168D layout. obs01 is DELIBERATELY not shared: it is the TCP-unified position term
    # (0.242 m hands) while the base keeps the 0.18 m VR lever — asserted separately in
    # test_local_unified_hand_point_is_the_force_tcp.
    remap = {  # local-unified key → 168D key (slots shift by +1 after obs04)
        "obs00_command_lower_body": "obs00_command_lower_body",
        "obs02_vr_3point_orn": "obs02_vr_3point_orn",
        "obs03_force_cmd_local": "obs03_force_cmd_local",
        "obs04_force_meas_local": "obs04_force_meas_local",
        "obs06_integrator_state": "obs05_integrator_state",
        "obs07_x_tgt_shift": "obs06_x_tgt_shift",
        "obs08_motion_ref_ori_b": "obs07_motion_ref_ori_b",
        "obs09_projected_gravity": "obs08_projected_gravity",
        "obs10_base_ang_vel": "obs09_base_ang_vel",
        "obs11_dof_pos": "obs10_dof_pos",
        "obs12_dof_vel": "obs11_dof_vel",  # the JointVelDropout term (default OFF), verbatim
        "obs13_actions": "obs12_actions",
    }
    for uk, bk in remap.items():
        assert actor[uk] is base[bk], (uk, bk)
    # the dropout wrapper stays the hybrid_3pt term (same env knobs, same default OFF)
    assert actor["obs12_dof_vel"].func == "wbt_training.config_values.hybrid_3pt:JointVelDropout"


def test_tcp_unified_offset_matches_hybrid_tcp():
    """The VR_3POINT_OFFSETS_TCP hand offset MUST equal hybrid.TCP_OFFSET_X.

    robot_config MIRRORS the constant rather than importing it (importing would close the
    cycle robot_config → hybrid → force_tracking → observation → robot_config), so this
    test is the only thing keeping the two in sync. If someone retunes the gripper and
    moves TCP_OFFSET_X, this fails instead of silently splitting position from force again.
    """
    from wbt_training.config_values.hybrid import TCP_OFFSET_X
    from wbt_training.config_values.robot_config import TCP_UNIFIED_HAND_OFFSET_X, VR_3POINT_OFFSETS_TCP

    assert TCP_UNIFIED_HAND_OFFSET_X == TCP_OFFSET_X
    # Both HANDS sit exactly on the force TCP: (tcp_offset_x, 0, 0) — the SAME vector
    # HybridContactInjector.setup writes into env._wbt_force_offsets_b. No lateral term:
    # keeping the old ∓0.025 would leave the points 2.5 cm apart, i.e. not unified.
    assert VR_3POINT_OFFSETS_TCP[0] == [TCP_OFFSET_X, 0.0, 0.0]
    assert VR_3POINT_OFFSETS_TCP[1] == [TCP_OFFSET_X, 0.0, 0.0]


def test_tcp_unified_head_point_unchanged():
    """Only the HANDS move; the head (torso_link +0.35 z) is carried over verbatim."""
    from wbt_training.config_values.robot_config import VR_3POINT_OFFSETS, VR_3POINT_OFFSETS_TCP

    assert VR_3POINT_OFFSETS_TCP[2] == list(VR_3POINT_OFFSETS[2])
    assert len(VR_3POINT_OFFSETS_TCP) == len(VR_3POINT_OFFSETS) == 3


def test_default_vr_offsets_untouched():
    """The SHARED 0.18 m VR offsets must NOT have moved — every other 3pt/2pt preset and
    every already-trained checkpoint depends on them. This is the blast-radius guard."""
    from wbt_training.config_values.robot_config import VR_1POINT_OFFSETS, VR_2POINT_OFFSETS, VR_3POINT_OFFSETS

    assert VR_3POINT_OFFSETS == [[0.18, -0.025, 0.0], [0.18, +0.025, 0.0], [0.0, 0.0, 0.35]]
    assert VR_2POINT_OFFSETS == [[0.18, -0.025, 0.0], [0.18, +0.025, 0.0]]
    assert VR_1POINT_OFFSETS == [[0.18, +0.025, 0.0]]


def test_local_unified_hand_point_is_the_force_tcp():
    """The unified obs01 slot uses the TCP position func; the base 3pt-hybrid keeps the
    0.18 m one. Guards against the two ever being collapsed back into one term."""
    from wbt_training.config_values.hybrid_3pt import _actor_obs_3pt_hybrid_terms as base
    from wbt_training.config_values.hybrid_unified import g1_29dof_wbt_observation_3pt_unified_hybrid as obs

    term = obs.groups["actor_obs"].terms["obs01_vr_3point_pos_tcp"]
    assert term.func == "wbt_training.config_values.observation:vr_3point_local_target_tcp"
    assert term.scale == 1.0 and term.noise == 0.0
    # the base preset is NOT dragged along
    assert base["obs01_vr_3point_pos"].func == "wbt_training.config_values.observation:vr_3point_local_target"


def test_vr_offset_cache_key_separates_the_two_geometries():
    """`_resolve_offsets` caches on env; the two 3pt terms share body names but differ in
    offsets, so the cache key must include the offsets or whichever term runs first
    poisons the other (a silent 6.2 cm error in one of the two)."""
    import torch
    from wbt_training.config_values.observation import _resolve_offsets

    class _Env:
        device = "cpu"
        num_envs = 1

    env = _Env()
    bodies = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
    a = _resolve_offsets(env, bodies, [[0.18, -0.025, 0.0], [0.18, 0.025, 0.0], [0.0, 0.0, 0.35]])
    b = _resolve_offsets(env, bodies, [[0.242, 0.0, 0.0], [0.242, 0.0, 0.0], [0.0, 0.0, 0.35]])
    assert not torch.allclose(a, b), "cache collision: the TCP term got the 0.18 m offsets"
    assert float(b[0, 0, 0]) == pytest.approx(0.242)  # float32 tensor
    # and the first entry is still intact after the second call (no overwrite)
    assert float(a[0, 0, 0]) == pytest.approx(0.18)


def test_inference_tcp_offsets_match_training():
    """Deploy-side mirror must be byte-identical to the training constant — the two live in
    separate packages with no shared import, so only this test couples them."""
    from wbt_training.config_values.robot_config import VR_3POINT_OFFSETS_TCP as train_tcp

    from wbt_wrappers_inference.config_values.robot_config import VR_3POINT_OFFSETS_TCP as infer_tcp

    assert [list(map(float, o)) for o in infer_tcp] == [list(map(float, o)) for o in train_tcp]


def test_local_unified_critic_is_hybrid_critic_verbatim():
    from wbt_training.config_values.hybrid import g1_29dof_wbt_observation_2pt_hybrid
    from wbt_training.config_values.hybrid_unified import g1_29dof_wbt_observation_3pt_unified_hybrid as obs

    c2 = g1_29dof_wbt_observation_2pt_hybrid.groups["critic_obs"].terms
    cu = obs.groups["critic_obs"].terms
    assert set(cu.keys()) == set(c2.keys())
    assert all(cu[k] is c2[k] for k in c2), "every critic term must be the SAME object (verbatim reuse)"
    assert obs.groups["critic_obs"].history_length == 10


# ---------------------------------------------------------------------------
# Experiment presets (R2 local unified + R1 local unified-anneal)
# ---------------------------------------------------------------------------
@_phase1_only
def test_local_unified_presets_registered_and_wired():
    from wbt_training.config_values.experiment import DEFAULTS
    from wbt_training.config_values.hybrid_unified import (
        _UNIFIED_ANNEAL_CFG,
        _UNIFIED_CFG,
        g1_29dof_wbt_observation_3pt_unified_hybrid,
    )

    p = DEFAULTS["g1_wbt_3pt_hybrid_unified"]
    a = DEFAULTS["g1_wbt_3pt_hybrid_unified_anneal"]
    assert p.observation is g1_29dof_wbt_observation_3pt_unified_hybrid
    assert a.observation is g1_29dof_wbt_observation_3pt_unified_hybrid  # same 170D both
    # R2 injector carries observe-only; R1 carries the full-servo cfg
    p_inj = p.command.setup_terms["hybrid_contact_injector"].params["hybrid_config"]
    a_inj = a.command.setup_terms["hybrid_contact_injector"].params["hybrid_config"]
    assert p_inj is _UNIFIED_CFG and p_inj.servo_observe_only is True and p_inj.servo_enabled is True
    assert a_inj is _UNIFIED_ANNEAL_CFG and a_inj.servo_observe_only is False and a_inj.servo_enabled is True
    # rewards: unified terms present, preintegrator gone
    for exp in (p, a):
        assert "force_error_direct" in exp.reward.terms
        assert "contact_init_bonus" in exp.reward.terms
        assert "approach_guidance" in exp.reward.terms
        assert "force_rate_penalty" in exp.reward.terms
        assert "force_error_preintegrator" not in exp.reward.terms
    # curriculum: force-alpha ramp ENABLED on both; servo-authority ONLY on the anneal twin
    p_fa = p.curriculum.setup_terms["force_alpha"].params["hand_force_config"]
    a_fa = a.curriculum.setup_terms["force_alpha"].params["hand_force_config"]
    for fa in (p_fa, a_fa):
        assert fa.force_alpha_init == 0.0 and fa.alpha_step == 0.01
    assert "servo_authority" not in p.curriculum.setup_terms
    assert "servo_authority" in a.curriculum.setup_terms
    assert "servo_authority" in a.curriculum.reset_terms
    # termination: force_runaway estop wired with the matching cfg
    assert p.termination.terms["force_runaway"].params["cfg"] is _UNIFIED_CFG
    assert a.termination.terms["force_runaway"].params["cfg"] is _UNIFIED_ANNEAL_CFG


@_phase1_only
def test_local_unified_shares_carriers_with_global_vel_unified():
    """Deliberate design: the hybrid command/curriculum/termination carriers are
    tracking-point- and obs-agnostic, so the local twins reuse the EXACT global-vel
    unified objects (mirrors g1_wbt_3pt_hybrid reusing hybrid.py's 2pt carriers).

    ⚠️ `reward` is the ONE deliberate exception since 2026-08-05: the TCP precision terms
    (tcp_precision.py) were added to the LOCAL twins only, because only they track both
    hands at the 0.242 m TCP — the global-vel family still uses the 0.18 m VR lever, where
    a 0.242 m reward FK would be pricing a point the obs never mentions. So the deltas
    between the two families are now `observation` AND `reward`; everything else is shared.
    See test_local_unified_reward_adds_only_the_tcp_terms for what that delta contains.
    """
    from wbt_training.config_values.experiment import DEFAULTS

    lp, gp = DEFAULTS["g1_wbt_3pt_hybrid_unified"], DEFAULTS["g1_wbt_3pt_hybrid_global_vel_unified"]
    la, ga = DEFAULTS["g1_wbt_3pt_hybrid_unified_anneal"], DEFAULTS["g1_wbt_3pt_hybrid_global_vel_unified_anneal"]
    for local, gvel in ((lp, gp), (la, ga)):
        assert local.command is gvel.command
        assert local.curriculum is gvel.curriculum
        assert local.termination is gvel.termination
        assert local.observation is not gvel.observation  # 170D local vs 185D global-vel
        assert local.reward is not gvel.reward  # local-only TCP precision terms


@_phase1_only
def test_local_unified_reward_adds_only_the_tcp_terms():
    """The local↔global-vel reward delta must be EXACTLY the two TCP precision terms, with
    every shared term still the same object — so the local twins have not silently drifted
    into a different reward recipe (2026-08-05)."""
    from wbt_training.config_values.experiment import DEFAULTS

    for lk, gk in (
        ("g1_wbt_3pt_hybrid_unified", "g1_wbt_3pt_hybrid_global_vel_unified"),
        ("g1_wbt_3pt_hybrid_unified_anneal", "g1_wbt_3pt_hybrid_global_vel_unified_anneal"),
    ):
        loc, gv = DEFAULTS[lk].reward.terms, DEFAULTS[gk].reward.terms
        assert set(loc) - set(gv) == {"tcp_position_error_exp", "tcp_orientation_error_exp"}
        assert not set(gv) - set(loc)
        for k, v in gv.items():
            assert loc[k] is v, f"shared reward term {k} diverged"


@_phase1_only
def test_local_unified_inherits_3pt_hybrid_base():
    from wbt_training.config_values.experiment import DEFAULTS

    base = DEFAULTS["g1_wbt_3pt_hybrid"]
    p = DEFAULTS["g1_wbt_3pt_hybrid_unified"]
    a = DEFAULTS["g1_wbt_3pt_hybrid_unified_anneal"]
    assert p.robot is base.robot  # yam URDF etc. inherited from the LOCAL 3pt-hybrid base
    assert p.algo is base.algo
    assert a.robot is base.robot
    # local: NO root-state trio, quaternion orientation (the point of this preset family)
    actor = p.observation.groups["actor_obs"].terms
    assert "obs02_vr_3point_orn" in actor
    assert not any("orn_6d" in k for k in actor)
    assert not any("motion_ref_pos_b" in k or "motion_ref_vel_b" in k or "base_lin_vel" in k for k in actor)


def test_local_unified_does_not_disturb_base_3pt_hybrid():
    """The base exp:g1-wbt-3pt-hybrid (train + eval) must be untouched: same 168D obs
    object, same servo-ON command/reward/termination as before the local-unified fork."""
    from wbt_training.config_values.experiment import DEFAULTS
    from wbt_training.config_values.hybrid import (
        g1_29dof_wbt_2pt_hybrid_reward,
        g1_29dof_wbt_hybrid_command,
        g1_29dof_wbt_hybrid_termination,
    )
    from wbt_training.config_values.hybrid_3pt import g1_29dof_wbt_observation_3pt_hybrid

    base = DEFAULTS["g1_wbt_3pt_hybrid"]
    assert base.observation is g1_29dof_wbt_observation_3pt_hybrid  # 168D untouched
    assert base.command is g1_29dof_wbt_hybrid_command
    assert base.reward is g1_29dof_wbt_2pt_hybrid_reward
    assert base.termination is g1_29dof_wbt_hybrid_termination
    hc = base.command.setup_terms["hybrid_contact_injector"].params["hybrid_config"]
    assert hc.servo_enabled is True and hc.servo_observe_only is False
    assert "force_error_preintegrator" in base.reward.terms
    fa = base.curriculum.setup_terms["force_alpha"].params["hand_force_config"]
    assert fa.force_alpha_init == 1.0 and fa.alpha_step == 0.0
    # the 168D actor layout itself gained no term
    assert len(base.observation.groups["actor_obs"].terms) == 13
    assert "obs05_contact_phase" not in base.observation.groups["actor_obs"].terms


def test_local_unified_does_not_disturb_global_vel_unified():
    """Adding the local twins must leave the 185D global-vel unified presets byte-identical."""
    from wbt_training.config_values.experiment import DEFAULTS
    from wbt_training.config_values.hybrid_unified import g1_29dof_wbt_observation_3pt_unified_hybrid_global_vel

    gp = DEFAULTS["g1_wbt_3pt_hybrid_global_vel_unified"]
    assert gp.observation is g1_29dof_wbt_observation_3pt_unified_hybrid_global_vel
    actor = gp.observation.groups["actor_obs"].terms
    assert len(actor) == 17  # 185D layout unchanged
    assert "obs07_contact_phase" in actor


# ---------------------------------------------------------------------------
# Inference configs (170D, training↔inference agreement)
# ---------------------------------------------------------------------------
def test_inference_config_matches_training_170d():
    from wbt_wrappers_inference.config_values.observation import holosoma_3pt_hybrid_unified_wbt as iobs

    rename = {"dof_pos": "joint_pos", "dof_vel": "joint_vel"}
    total = 0
    for k, dim in _EXPECTED_LOCAL_UNIFIED_ACTOR.items():
        bare = rename.get(k.split("_", 1)[1], k.split("_", 1)[1])
        assert iobs.obs_dims[bare] == dim, (k, bare)
        total += iobs.obs_dims[bare]
    assert total == 170
    assert list(iobs.obs_dict["actor_obs"]) == [
        rename.get(k.split("_", 1)[1], k.split("_", 1)[1]) for k in _EXPECTED_LOCAL_UNIFIED_ACTOR
    ]
    # ALL scales must match the training ObsTermCfg.scale term-for-term (review
    # 2026-07-24: a 3-term spot check let an integrator_state 1.0→0.5 mutation survive).
    from wbt_training.config_values.hybrid_unified import g1_29dof_wbt_observation_3pt_unified_hybrid as tobs

    for k, term in tobs.groups["actor_obs"].terms.items():
        bare = rename.get(k.split("_", 1)[1], k.split("_", 1)[1])
        assert iobs.obs_scales[bare] == term.scale, (bare, iobs.obs_scales[bare], term.scale)
    assert iobs.history_length_dict["actor_obs"] == 4


def test_inference_presets_registered():
    from wbt_wrappers_inference.config_values.inference import DEFAULTS as INF_DEFAULTS

    a = INF_DEFAULTS["g1-29dof-holosoma-3pt-hybrid-unified-wbt"]
    b = INF_DEFAULTS["g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt"]
    assert a is b  # same 170D obs serves both training twins
    # base 168D inference preset untouched
    base = INF_DEFAULTS["g1-29dof-holosoma-3pt-hybrid-wbt"]
    assert base is not a
    assert "contact_phase" not in base.observation.obs_dims
    # and the 185D global-vel unified preset is a DIFFERENT config (deep-keyed)
    gvel = INF_DEFAULTS["g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt"]
    assert gvel is not a
    assert "vr_3point_orn" in a.observation.obs_dims  # quat, local family
    assert "vr_3point_orn_6d" not in a.observation.obs_dims
    assert "base_lin_vel" not in a.observation.obs_dims
