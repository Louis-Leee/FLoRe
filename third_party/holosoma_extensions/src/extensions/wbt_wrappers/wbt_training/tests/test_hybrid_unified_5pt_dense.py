"""Unit tests for the 5-POINT and DENSE twins of the LOCAL R1 anneal preset.

    exp:g1-wbt-5pt-hybrid-unified-anneal   184D actor  (170D 3pt layout + the two FEET)
    exp:g1-wbt-hybrid-unified-anneal       183D actor  (dense 29-DOF tracking, no keypoints)

Both are a ONE-FIELD `replace(g1_wbt_3pt_hybrid_unified_anneal, observation=...)`, so the
load-bearing claim these tests defend is "the force handling is identical to
exp:g1-wbt-3pt-hybrid-unified-anneal" — asserted by OBJECT IDENTITY, not by comparing
weights, because identity is what makes drift impossible.

All config/identity assertions plus a small monkeypatched numeric check of the two new obs
term functions — no IsaacSim runtime.
"""

import os
from types import SimpleNamespace

import pytest
import torch

# `WBT_HAND_MASS_DR=1` (phase 2 of the TCP precision work) deliberately swaps the unified
# twins' randomization / robot / cfg carriers. The new presets inherit that swap, so the
# assertions keyed to the phase-1 objects do not hold there — skip rather than go red.
_phase1_only = pytest.mark.skipif(
    os.environ.get("WBT_HAND_MASS_DR", "0").strip().lower() in ("1", "true", "yes"),
    reason="phase-1 invariant; WBT_HAND_MASS_DR=1 intentionally changes these",
)

# The 14 tracked bodies (robot_config.wbt_body_names), in order. Both ankle_roll links MUST
# be in here or `observation._resolve_body_indices` raises at env build for the 5pt preset.
_TRACKED = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

# ---------------------------------------------------------------------------
# 184D 5-point actor obs. Slot KEYS + order are the 170D layout's, byte-for-byte; only
# obs01/obs02 change (name AND width: 9→15 position, 12→20 quaternion orientation).
# ---------------------------------------------------------------------------
_EXPECTED_5PT_ACTOR = {
    "obs00_command_lower_body": 24,
    "obs01_vr_5point_pos_tcp": 15,  # 5 bodies x 3; hands still on the 0.242 m force TCP
    "obs02_vr_5point_orn": 20,  # 5 bodies x 4 (QUATERNION, like the local 3pt family)
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
}  # = 184

# ---------------------------------------------------------------------------
# 183D DENSE actor obs. `motion_command` (58D = 29 joint pos + 29 joint vel) replaces the
# whole keypoint block AND `command_lower_body` (which it subsumes); the 26D force block
# sits right after it, where hybrid.py puts it in the 2pt layout.
# ---------------------------------------------------------------------------
_EXPECTED_DENSE_ACTOR = {
    "obs00_motion_command": 58,
    "obs01_force_cmd_local": 6,
    "obs02_force_meas_local": 6,
    "obs03_contact_phase": 2,
    "obs04_integrator_state": 6,
    "obs05_x_tgt_shift": 6,
    "obs06_motion_ref_ori_b": 6,
    "obs07_projected_gravity": 3,
    "obs08_base_ang_vel": 3,
    "obs09_dof_pos": 29,
    "obs10_dof_vel": 29,
    "obs11_actions": 29,
}  # = 183


# ---------------------------------------------------------------------------
# Obs layouts
# ---------------------------------------------------------------------------
def test_5pt_actor_obs_layout_184d():
    from wbt_training.config_values.hybrid_unified_5pt_dense import g1_29dof_wbt_observation_5pt_unified_hybrid as obs

    actor = obs.groups["actor_obs"]
    assert list(actor.terms.keys()) == list(_EXPECTED_5PT_ACTOR.keys())
    assert list(actor.terms.keys()) == sorted(actor.terms.keys())  # sorted-concat contract
    assert sum(_EXPECTED_5PT_ACTOR.values()) == 184
    assert actor.history_length == 4
    assert actor.enable_noise is True


def test_dense_actor_obs_layout_183d():
    from wbt_training.config_values.hybrid_unified_5pt_dense import g1_29dof_wbt_observation_dense_unified_hybrid as obs

    actor = obs.groups["actor_obs"]
    assert list(actor.terms.keys()) == list(_EXPECTED_DENSE_ACTOR.keys())
    assert list(actor.terms.keys()) == sorted(actor.terms.keys())
    assert sum(_EXPECTED_DENSE_ACTOR.values()) == 183
    assert actor.history_length == 4
    assert actor.enable_noise is True


def test_both_critics_are_the_hybrid_critic_verbatim():
    """The critic is tracking-point-agnostic (`robot_body_pos_b`/`_ori_b` already cover all
    14 tracked bodies, feet included), so both presets reuse hybrid.py's object graph."""
    from wbt_training.config_values.hybrid import g1_29dof_wbt_observation_2pt_hybrid
    from wbt_training.config_values.hybrid_unified_5pt_dense import (
        g1_29dof_wbt_observation_5pt_unified_hybrid,
        g1_29dof_wbt_observation_dense_unified_hybrid,
    )

    base = g1_29dof_wbt_observation_2pt_hybrid.groups["critic_obs"].terms
    for obs in (g1_29dof_wbt_observation_5pt_unified_hybrid, g1_29dof_wbt_observation_dense_unified_hybrid):
        critic = obs.groups["critic_obs"]
        # `ObsGroupCfg` copies the terms dict on construction, so compare per-term identity
        # (same convention as test_hybrid_3pt_unified.py's critic test).
        assert set(critic.terms) == set(base)
        assert all(critic.terms[k] is base[k] for k in base), "every critic term must be the SAME object"
        assert critic.history_length == 10
        assert critic.enable_noise is False


def test_5pt_reuses_the_170d_terms_verbatim():
    """Every slot except the two keypoint terms must be the SAME ObsTermCfg object as the
    170D local-unified layout — including `obs05_contact_phase` and the `obs12` dropout
    wrapper. A value-equal duplicate would silently diverge the moment a term gains params."""
    from wbt_training.config_values.hybrid_unified import _actor_obs_local_unified_terms as base
    from wbt_training.config_values.hybrid_unified_5pt_dense import g1_29dof_wbt_observation_5pt_unified_hybrid as obs

    actor = obs.groups["actor_obs"].terms
    shared = [k for k in _EXPECTED_5PT_ACTOR if k not in ("obs01_vr_5point_pos_tcp", "obs02_vr_5point_orn")]
    assert len(shared) == 12
    for k in shared:
        assert actor[k] is base[k], k  # slot keys are identical, so no remap table needed
    # the actor-only joint-velocity dropout (default OFF) comes along unchanged
    assert actor["obs12_dof_vel"].func == "wbt_training.config_values.hybrid_3pt:JointVelDropout"
    # and the two new terms are NOT shared with the 3pt layout (different geometry + width)
    assert actor["obs01_vr_5point_pos_tcp"] is not base["obs01_vr_3point_pos_tcp"]


def test_dense_reuses_tml_and_force_terms_verbatim():
    """Dense: the non-force slots are the SAME objects as `exp:g1-wbt`'s 157D layout and the
    force slots the SAME objects as the 170D unified layout — the dense preset introduces no
    new obs term at all. `dof_vel` is the one deliberate substitution (dropout wrapper)."""
    from wbt_training.config_values.hybrid_unified import _actor_obs_local_unified_terms as unified
    from wbt_training.config_values.hybrid_unified_5pt_dense import g1_29dof_wbt_observation_dense_unified_hybrid as obs
    from wbt_training.config_values.observation import _actor_obs_terms as tml

    actor = obs.groups["actor_obs"].terms
    for dense_key, tml_key in (
        ("obs00_motion_command", "obs0_motion_command"),
        ("obs06_motion_ref_ori_b", "obs1_motion_ref_ori_b"),
        ("obs07_projected_gravity", "obs2_projected_gravity"),
        ("obs08_base_ang_vel", "obs3_base_ang_vel"),
        ("obs09_dof_pos", "obs4_dof_pos"),
        ("obs11_actions", "obs6_actions"),
    ):
        assert actor[dense_key] is tml[tml_key], (dense_key, tml_key)
    for dense_key, unified_key in (
        ("obs01_force_cmd_local", "obs03_force_cmd_local"),
        ("obs02_force_meas_local", "obs04_force_meas_local"),
        ("obs03_contact_phase", "obs05_contact_phase"),
        ("obs04_integrator_state", "obs06_integrator_state"),
        ("obs05_x_tgt_shift", "obs07_x_tgt_shift"),
        ("obs10_dof_vel", "obs12_dof_vel"),
    ):
        assert actor[dense_key] is unified[unified_key], (dense_key, unified_key)
    # dof_vel is the dropout wrapper, NOT the plain TML term (behaviourally identical while
    # disabled — it applies the same base 0.5 noise internally)
    assert actor["obs10_dof_vel"] is not tml["obs5_dof_vel"]
    assert actor["obs10_dof_vel"].func == "wbt_training.config_values.hybrid_3pt:JointVelDropout"
    assert actor["obs10_dof_vel"].noise == 0.0 and tml["obs5_dof_vel"].noise == 0.5


def test_dense_carries_no_keypoint_term():
    """The whole point of the dense variant: joint targets, not keypoints."""
    from wbt_training.config_values.hybrid_unified_5pt_dense import g1_29dof_wbt_observation_dense_unified_hybrid as obs

    keys = list(obs.groups["actor_obs"].terms)
    assert not any("vr_" in k for k in keys)
    assert not any("command_lower_body" in k for k in keys)  # subsumed by motion_command
    assert "obs00_motion_command" in keys


def test_both_are_root_invariant_like_the_local_3pt_family():
    """No `motion_ref_pos_b` / `motion_ref_vel_b` / `base_lin_vel` in the actor — these track
    in place and do not walk, exactly like `exp:g1-wbt-3pt-hybrid-unified[-anneal]`. The 6D
    orientation rep also stays out (it belongs to the `-global-vel` family)."""
    from wbt_training.config_values.hybrid_unified_5pt_dense import (
        g1_29dof_wbt_observation_5pt_unified_hybrid,
        g1_29dof_wbt_observation_dense_unified_hybrid,
    )

    for obs in (g1_29dof_wbt_observation_5pt_unified_hybrid, g1_29dof_wbt_observation_dense_unified_hybrid):
        keys = list(obs.groups["actor_obs"].terms)
        assert not any(("motion_ref_pos_b" in k or "motion_ref_vel_b" in k or "base_lin_vel" in k) for k in keys)
        assert not any("orn_6d" in k for k in keys)


# ---------------------------------------------------------------------------
# The five points: provenance, geometry
# ---------------------------------------------------------------------------
def test_five_points_are_maskedmimic_five_point_after_alias_resolution():
    """The user's spec was "look at exp:g1-wbt-maskedmimic's 5pt to see where the points
    are". This pins that we used exactly that set: maskedmimic's `five_point` hover mode,
    with its own alias table applied (torso→torso_link, left/right_hand→*_wrist_yaw_link).
    If someone retunes the hover catalogue, this fails instead of silently disagreeing."""
    from wbt_training.config_values.hybrid_unified_5pt_dense import VR_5POINT_BODIES
    from wbt_training.config_values.maskedmimic_observations import DEFAULT_HOVER_MASK_MODES

    alias = {"torso": "torso_link", "left_hand": "left_wrist_yaw_link", "right_hand": "right_wrist_yaw_link"}
    mm5 = DEFAULT_HOVER_MASK_MODES["five_point"]["keypoints"]
    mm3 = DEFAULT_HOVER_MASK_MODES["three_point"]["keypoints"]
    assert set(alias.get(b, b) for b in mm5) == set(VR_5POINT_BODIES)
    # and 5pt really is 3pt + the two feet, in maskedmimic's own catalogue too
    assert set(mm5) - set(mm3) == {"left_ankle_roll_link", "right_ankle_roll_link"}


def test_5pt_is_the_3pt_list_plus_the_two_feet():
    from wbt_training.config_values.hybrid_unified_5pt_dense import (
        VR_5POINT_BODIES,
        VR_5POINT_FEET_BODIES,
        VR_5POINT_OFFSETS_TCP,
    )
    from wbt_training.config_values.robot_config import VR_3POINT_BODIES, VR_3POINT_OFFSETS_TCP

    assert VR_5POINT_BODIES[:3] == list(VR_3POINT_BODIES)
    assert VR_5POINT_BODIES[3:] == list(VR_5POINT_FEET_BODIES)
    assert len(VR_5POINT_BODIES) == len(VR_5POINT_OFFSETS_TCP) == 5
    # the 3pt prefix of the OFFSETS is the 3pt constant, so a gripper respec that moves
    # VR_3POINT_OFFSETS_TCP moves the 5pt preset with it (no second literal to forget)
    assert VR_5POINT_OFFSETS_TCP[:3] == [list(o) for o in VR_3POINT_OFFSETS_TCP]


def test_5pt_hands_sit_on_the_force_tcp():
    from wbt_training.config_values.hybrid import TCP_OFFSET_X
    from wbt_training.config_values.hybrid_unified_5pt_dense import VR_5POINT_BODIES, VR_5POINT_OFFSETS_TCP

    for i in (0, 1):
        assert VR_5POINT_BODIES[i].endswith("_wrist_yaw_link")
        assert VR_5POINT_OFFSETS_TCP[i] == [TCP_OFFSET_X, 0.0, 0.0]


def test_5pt_head_unchanged_and_feet_have_no_offset():
    """Head keeps torso_link +0.35 z. Feet are the bare ankle_roll_link origin: the hand and
    head offsets model physical levers (tracker mount / head height) and no such lever exists
    for a foot — maskedmimic tracks the raw body, and BODY_NAME_TO_MJ_IDX even aliases
    `left_foot_contact_point` onto the same index as `left_ankle_roll_link`."""
    from wbt_training.config_values.hybrid_unified_5pt_dense import VR_5POINT_FOOT_OFFSET, VR_5POINT_OFFSETS_TCP
    from wbt_training.config_values.robot_config import BODY_NAME_TO_MJ_IDX, VR_3POINT_OFFSETS

    assert VR_5POINT_OFFSETS_TCP[2] == list(VR_3POINT_OFFSETS[2]) == [0.0, 0.0, 0.35]
    assert VR_5POINT_FOOT_OFFSET == [0.0, 0.0, 0.0]
    assert VR_5POINT_OFFSETS_TCP[3] == VR_5POINT_OFFSETS_TCP[4] == [0.0, 0.0, 0.0]
    for side in ("left", "right"):
        assert BODY_NAME_TO_MJ_IDX[f"{side}_foot_contact_point"] == BODY_NAME_TO_MJ_IDX[f"{side}_ankle_roll_link"]


def test_all_five_bodies_are_tracked_by_the_motion_command():
    """`observation._resolve_body_indices` does `body_names_to_track.index(name)`, so a body
    missing from the tracked list is a ValueError at env build — this is the cheap guard."""
    from wbt_training.config_values.command import motion_config
    from wbt_training.config_values.hybrid_unified_5pt_dense import VR_5POINT_BODIES

    tracked = list(motion_config.body_names_to_track)
    assert tracked == _TRACKED  # pin the 14-body list this preset family assumes
    for body in VR_5POINT_BODIES:
        assert body in tracked, body


# ---------------------------------------------------------------------------
# The two new obs term functions (numeric, monkeypatched motion command)
# ---------------------------------------------------------------------------
def _fake_env_and_mc(num_envs: int = 3):
    """A pelvis-at-origin, identity-attitude motion command with distinguishable body
    positions. Identity ref quat + zero ref pos makes the pelvis-local transform a no-op, so
    the position term reduces to `body_pos_w + R·offset` and the assertions stay readable."""
    nb = len(_TRACKED)
    ident = torch.tensor([0.0, 0.0, 0.0, 1.0])  # XYZW, as the whole codebase uses
    mc = SimpleNamespace(
        motion_cfg=SimpleNamespace(body_names_to_track=_TRACKED),
        ref_quat_w=ident.repeat(num_envs, 1),
        ref_pos_w=torch.zeros(num_envs, 3),
        body_pos_w=torch.arange(num_envs * nb * 3, dtype=torch.float).reshape(num_envs, nb, 3),
        body_quat_w=ident.repeat(num_envs, nb, 1),
    )
    return SimpleNamespace(device="cpu", num_envs=num_envs), mc


def test_5pt_obs_functions_widths_and_3pt_prefix(monkeypatch):
    """15D / 20D, and slots [0:9] / [0:12] are the 3pt vectors UNCHANGED — the feet are
    appended, not interleaved, so a 3pt-trained intuition (and any hand-written probe or
    W&B panel) still reads the first three points the same way."""
    import wbt_training.config_values.observation as O
    from wbt_training.config_values.hybrid_unified_5pt_dense import (
        vr_5point_local_orn_target,
        vr_5point_local_target_tcp,
    )

    env, mc = _fake_env_and_mc()
    monkeypatch.setattr(O, "_get_motion_command", lambda _env: mc)

    pos5, orn5 = vr_5point_local_target_tcp(env), vr_5point_local_orn_target(env)
    pos3, orn3 = O.vr_3point_local_target_tcp(env), O.vr_3point_local_orn_target(env)
    assert pos5.shape == (env.num_envs, 15) and orn5.shape == (env.num_envs, 20)
    assert torch.allclose(pos5[:, :9], pos3)
    assert torch.allclose(orn5[:, :12], orn3)


def test_5pt_foot_slots_are_the_bare_ankle_reference_positions(monkeypatch):
    """Zero offset ⇒ the foot slots are exactly the reference ankle_roll_link positions
    (tracked indices 3 and 6). This is what pins "no sole offset" numerically, not just as
    a constant."""
    import wbt_training.config_values.observation as O
    from wbt_training.config_values.hybrid_unified_5pt_dense import vr_5point_local_target_tcp

    env, mc = _fake_env_and_mc()
    monkeypatch.setattr(O, "_get_motion_command", lambda _env: mc)

    pos5 = vr_5point_local_target_tcp(env)
    assert torch.allclose(pos5[:, 9:12], mc.body_pos_w[:, _TRACKED.index("left_ankle_roll_link"), :])
    assert torch.allclose(pos5[:, 12:15], mc.body_pos_w[:, _TRACKED.index("right_ankle_roll_link"), :])


def test_5pt_offset_cache_does_not_collide_with_the_3pt_terms():
    """`_resolve_offsets` caches on the env keyed by bodies AND offsets. The 5pt list is a
    superset of the 3pt one, so verify neither term can serve the other a stale tensor (the
    same class of bug the 0.18-vs-0.242 cache key guards against)."""
    from wbt_training.config_values.hybrid_unified_5pt_dense import VR_5POINT_BODIES, VR_5POINT_OFFSETS_TCP
    from wbt_training.config_values.observation import _resolve_offsets
    from wbt_training.config_values.robot_config import VR_3POINT_BODIES, VR_3POINT_OFFSETS, VR_3POINT_OFFSETS_TCP

    env = SimpleNamespace(device="cpu", num_envs=1)
    five = _resolve_offsets(env, VR_5POINT_BODIES, VR_5POINT_OFFSETS_TCP)
    tcp3 = _resolve_offsets(env, list(VR_3POINT_BODIES), [list(o) for o in VR_3POINT_OFFSETS_TCP])
    vr3 = _resolve_offsets(env, list(VR_3POINT_BODIES), [list(o) for o in VR_3POINT_OFFSETS])
    assert five.shape == (1, 5, 3) and tcp3.shape == (1, 3, 3) and vr3.shape == (1, 3, 3)
    assert torch.allclose(five[:, :3, :], tcp3)  # same geometry for the shared three
    assert not torch.allclose(tcp3, vr3)  # and the 0.18 m term is still its own thing
    assert float(vr3[0, 0, 0]) == pytest.approx(0.18)


# ---------------------------------------------------------------------------
# Experiment presets: the "observation is the ONLY delta" contract
# ---------------------------------------------------------------------------
def test_new_presets_registered_under_the_expected_cli_names():
    from wbt_training.config_values.experiment import DEFAULTS

    for key, cli in (
        ("g1_wbt_5pt_hybrid_unified_anneal", "exp:g1-wbt-5pt-hybrid-unified-anneal"),
        ("g1_wbt_hybrid_unified_anneal", "exp:g1-wbt-hybrid-unified-anneal"),
    ):
        assert key in DEFAULTS
        assert f"exp:{key.replace('_', '-')}" == cli  # the exact transform AnnotatedExperimentConfig applies


@_phase1_only
def test_observation_is_the_only_delta_vs_the_3pt_anneal_twin():
    """THE load-bearing test. Everything that decides force behaviour — the injector cfg
    (R1 anneal, not R2 observe-only), the 17-term reward incl. the two TCP precision terms,
    the servo-authority curriculum, the force-runaway termination, the yam robot, the algo,
    the randomization — must be the SAME OBJECT as the 3pt anneal twin's. Identity, not
    equality: a value-equal copy is exactly how "identical force handling" rots."""
    from wbt_training.config_values.experiment import DEFAULTS

    ref = DEFAULTS["g1_wbt_3pt_hybrid_unified_anneal"]
    fields = (
        "command",
        "reward",
        "curriculum",
        "termination",
        "randomization",
        "robot",
        "algo",
        "action",
        "training",
        "logger",
        "simulator",
        "env_class",
    )
    for key in ("g1_wbt_5pt_hybrid_unified_anneal", "g1_wbt_hybrid_unified_anneal"):
        exp = DEFAULTS[key]
        for f in fields:
            assert getattr(exp, f) is getattr(ref, f), f"{key}.{f} is not the 3pt anneal twin's object"
        assert exp.observation is not ref.observation


@_phase1_only
def test_new_presets_are_the_R1_anneal_twin_not_R2():
    """They were asked for as `-anneal`: live Jᵀ servo whose authority anneals 1→0. So the
    injector must carry `servo_observe_only=False` and the curriculum the servo-authority
    term — the R2 observe-only config would train a different controller under the same name."""
    from wbt_training.config_values.experiment import DEFAULTS
    from wbt_training.config_values.hybrid_unified import _UNIFIED_ANNEAL_CFG

    for key in ("g1_wbt_5pt_hybrid_unified_anneal", "g1_wbt_hybrid_unified_anneal"):
        exp = DEFAULTS[key]
        inj = exp.command.setup_terms["hybrid_contact_injector"].params["hybrid_config"]
        assert inj is _UNIFIED_ANNEAL_CFG
        assert inj.servo_enabled is True and inj.servo_observe_only is False
        assert "servo_authority" in exp.curriculum.setup_terms
        assert "servo_authority" in exp.curriculum.reset_terms
        assert exp.termination.terms["force_runaway"].params["cfg"] is _UNIFIED_ANNEAL_CFG
        # force-alpha ramp from 0 ("position-first inside one run"), as in the 3pt twin
        fa = exp.curriculum.setup_terms["force_alpha"].params["hand_force_config"]
        assert fa.force_alpha_init == 0.0 and fa.alpha_step == 0.01


@_phase1_only
def test_new_presets_carry_the_full_unified_plus_tcp_reward():
    """The four unified force terms, no preintegrator, and BOTH TCP precision terms at the
    weights the 3pt anneal twin ships — this is the "make the two hands' TCP tracking
    better" requirement, satisfied by inheritance rather than a retuned copy."""
    from wbt_training.config_values.experiment import DEFAULTS
    from wbt_training.config_values.tcp_precision import g1_29dof_wbt_unified_anneal_tcp_reward

    for key in ("g1_wbt_5pt_hybrid_unified_anneal", "g1_wbt_hybrid_unified_anneal"):
        rw = DEFAULTS[key].reward
        assert rw is g1_29dof_wbt_unified_anneal_tcp_reward
        for term in ("force_error_direct", "contact_init_bonus", "approach_guidance", "force_rate_penalty"):
            assert term in rw.terms
        assert "force_error_preintegrator" not in rw.terms
        assert rw.terms["tcp_position_error_exp"].weight == 4.0
        assert rw.terms["tcp_position_error_exp"].params["sigma"] == 0.10
        assert rw.terms["tcp_orientation_error_exp"].weight == 2.0
        assert rw.terms["tcp_orientation_error_exp"].params["sigma"] == 0.25


def test_new_widths_cannot_collide_with_an_existing_unified_checkpoint():
    """Every trained unified checkpoint is 170D (or 185D for the global-vel family). 184D and
    183D differ from both, so a stale `--training.checkpoint` fails on a shape mismatch
    instead of loading silently against the wrong geometry — which is the failure mode the
    launcher's TCP preflight exists to catch for same-width presets."""
    from wbt_training.config_values.experiment import DEFAULTS

    dims = {
        "command_lower_body": 24,
        "motion_command": 58,
        "vr_3point_pos_tcp": 9,
        "vr_3point_orn": 12,
        "vr_3point_pos": 9,
        "vr_3point_orn_6d": 18,
        "vr_5point_pos_tcp": 15,
        "vr_5point_orn": 20,
        "motion_ref_pos_b": 3,
        "motion_ref_vel_b": 3,
        "force_cmd_local": 6,
        "force_meas_local": 6,
        "contact_phase": 2,
        "integrator_state": 6,
        "x_tgt_shift": 6,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "base_lin_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    }

    def width(key):
        terms = DEFAULTS[key].observation.groups["actor_obs"].terms
        return sum(dims[k.split("_", 1)[1]] for k in terms)

    assert width("g1_wbt_3pt_hybrid_unified_anneal") == 170
    assert width("g1_wbt_3pt_hybrid_global_vel_unified_anneal") == 185
    assert width("g1_wbt_5pt_hybrid_unified_anneal") == 184
    assert width("g1_wbt_hybrid_unified_anneal") == 183
    assert len({170, 185, 184, 183}) == 4


# ---------------------------------------------------------------------------
# Blast radius: nothing else may move
# ---------------------------------------------------------------------------
# The 170D local-unified layout, pinned here so "we did not disturb the 3pt twins" is
# checked against a literal rather than against the object we might have edited.
_EXPECTED_LOCAL_170D = (
    "obs00_command_lower_body",
    "obs01_vr_3point_pos_tcp",
    "obs02_vr_3point_orn",
    "obs03_force_cmd_local",
    "obs04_force_meas_local",
    "obs05_contact_phase",
    "obs06_integrator_state",
    "obs07_x_tgt_shift",
    "obs08_motion_ref_ori_b",
    "obs09_projected_gravity",
    "obs10_base_ang_vel",
    "obs11_dof_pos",
    "obs12_dof_vel",
    "obs13_actions",
)


def test_the_3pt_unified_twins_are_untouched():
    from wbt_training.config_values.experiment import DEFAULTS
    from wbt_training.config_values.hybrid_unified import g1_29dof_wbt_observation_3pt_unified_hybrid

    for key in ("g1_wbt_3pt_hybrid_unified", "g1_wbt_3pt_hybrid_unified_anneal"):
        obs = DEFAULTS[key].observation
        assert obs is g1_29dof_wbt_observation_3pt_unified_hybrid
        actor = obs.groups["actor_obs"].terms
        assert list(actor) == list(_EXPECTED_LOCAL_170D)
        assert (
            actor["obs01_vr_3point_pos_tcp"].func == "wbt_training.config_values.observation:vr_3point_local_target_tcp"
        )


def test_the_dense_and_3pt_base_presets_are_untouched():
    """`exp:g1-wbt` (the 157D dense base the new dense preset forks) and
    `exp:g1-wbt-3pt-hybrid` must keep their own obs objects and force config."""
    from wbt_training.config_values.experiment import DEFAULTS
    from wbt_training.config_values.hybrid import g1_29dof_wbt_hybrid_command
    from wbt_training.config_values.hybrid_3pt import g1_29dof_wbt_observation_3pt_hybrid
    from wbt_training.config_values.observation import g1_29dof_wbt_observation_tml

    dense = DEFAULTS["g1_wbt"]
    assert dense.observation is g1_29dof_wbt_observation_tml
    assert len(dense.observation.groups["actor_obs"].terms) == 7  # still the plain 157D layout
    assert "hybrid_contact_injector" not in dense.command.setup_terms  # no force stack on the base

    base3 = DEFAULTS["g1_wbt_3pt_hybrid"]
    assert base3.observation is g1_29dof_wbt_observation_3pt_hybrid
    assert base3.command is g1_29dof_wbt_hybrid_command
    assert len(base3.observation.groups["actor_obs"].terms) == 13


def test_shared_vr_offsets_and_tcp_constants_untouched():
    """The blast-radius guard: the 0.18 m VR offsets are shared by every other 3pt/2pt
    preset and every already-trained checkpoint, and the 3pt TCP list is what the live
    `pfl96opy` lineage trained on. Adding the 5pt lists must not have edited either."""
    from wbt_training.config_values.hybrid import TCP_OFFSET_X
    from wbt_training.config_values.robot_config import (
        TCP_UNIFIED_HAND_OFFSET_X,
        VR_1POINT_OFFSETS,
        VR_2POINT_OFFSETS,
        VR_3POINT_BODIES,
        VR_3POINT_OFFSETS,
        VR_3POINT_OFFSETS_TCP,
    )

    assert VR_3POINT_OFFSETS == [[0.18, -0.025, 0.0], [0.18, +0.025, 0.0], [0.0, 0.0, 0.35]]
    assert VR_2POINT_OFFSETS == [[0.18, -0.025, 0.0], [0.18, +0.025, 0.0]]
    assert VR_1POINT_OFFSETS == [[0.18, +0.025, 0.0]]
    assert VR_3POINT_OFFSETS_TCP == [[0.242, 0.0, 0.0], [0.242, 0.0, 0.0], [0.0, 0.0, 0.35]]
    assert VR_3POINT_BODIES == ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
    assert TCP_UNIFIED_HAND_OFFSET_X == TCP_OFFSET_X == 0.242


def test_new_module_adds_no_obs_terms_to_any_other_preset():
    """No preset other than the two new ones may reference the new obs term functions."""
    from wbt_training.config_values.experiment import DEFAULTS

    new = ("vr_5point_local_target_tcp", "vr_5point_local_orn_target")
    expected = {"g1_wbt_5pt_hybrid_unified_anneal"}
    users = set()
    for key, exp in DEFAULTS.items():
        groups = getattr(getattr(exp, "observation", None), "groups", None) or {}
        for group in groups.values():
            if any(term.func.endswith(f":{n}") for term in group.terms.values() for n in new):
                users.add(key)
    assert users == expected, users


def test_no_new_wbt_robot_config_fields():
    """The 5pt lists are module constants, NOT `WBTRobotConfig` fields — adding fields there
    would appear in every preset's serialized config / W&B diff."""
    from dataclasses import fields

    from wbt_training.config_values.robot_config import WBTRobotConfig

    names = {f.name for f in fields(WBTRobotConfig)}
    assert not any("5point" in n or "five" in n for n in names)
