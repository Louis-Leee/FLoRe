"""Two more tracking front-ends for the LOCAL C1 unified-anneal stack (user 2026-08-27).

  * `exp:g1-wbt-5pt-hybrid-unified-anneal`  — 184D actor: the 170D 3pt layout with the
    two FEET added as tracked points (5-point tracking).
  * `exp:g1-wbt-hybrid-unified-anneal`      — 183D actor: DENSE 29-DOF tracking (the
    `exp:g1-wbt` 157D obs) instead of keypoints, plus the same force channels.

WHY THIS MODULE IS ONLY OBSERVATIONS. The whole unified stack — the causal K_env contact
spring, the simulated F/T sensor, the Jᵀ servo + `ServoAuthorityCurriculum` anneal, the
soft `contact_phase`, `force_error_direct` / `contact_init_bonus` / `approach_guidance` /
`force_rate_penalty`, the force-alpha ramp, the force-runaway estop, and the two
`tcp_precision` TCP terms — reads the `_wbt_*` buffers and the motion command, NEVER the
obs layout. So it is tracking-point-agnostic, exactly as `hybrid_3pt.py` and
`hybrid_3pt_global_vel.py` already demonstrate. Both new presets are therefore a ONE-FIELD
`replace(g1_wbt_3pt_hybrid_unified_anneal, observation=...)` in experiment.py, and the
force handling is *the same objects*, not a copy that could drift.

NON-INVASION: this module only READS `observation.py` / `hybrid.py` / `hybrid_3pt.py` /
`hybrid_unified.py` / `robot_config.py`. Every shared obs slot below is the SAME ObsTermCfg
object the 170D layout uses, so a change there propagates here instead of diverging. No
existing preset, obs term, or constant is touched — in particular `VR_3POINT_OFFSETS`
(0.18 m, shared by every other 3pt/2pt preset) and `VR_3POINT_OFFSETS_TCP` are read-only
here, and the 5-point body/offset lists are NEW module constants rather than new
`WBTRobotConfig` fields (adding fields there would show up in every preset's W&B config).

---------------------------------------------------------------------------
WHERE THE FIVE POINTS COME FROM
---------------------------------------------------------------------------
`exp:g1-wbt-maskedmimic`'s hover-mask catalogue (maskedmimic_observations.py
DEFAULT_HOVER_MASK_MODES) defines them:

    "three_point": {"keypoints": ("torso", "left_hand", "right_hand")}
    "five_point":  {"keypoints": ("torso", "left_hand", "right_hand",
                                  "left_ankle_roll_link", "right_ankle_roll_link")}

and its alias table resolves `torso`→`torso_link`, `left_hand`→`left_wrist_yaw_link`,
`right_hand`→`right_wrist_yaw_link`. So maskedmimic's `three_point` is exactly this repo's
`VR_3POINT_BODIES`, and `five_point` = that plus the two `*_ankle_roll_link` — i.e. 5pt is
3pt + two FEET, which is what the user asked for. The two feet are appended (not
interleaved) so slots 0–2 of the position/orientation vectors stay byte-compatible in
MEANING with the 3pt layout, which keeps W&B plots and any hand-written probe readable.

FOOT OFFSETS ARE ZERO, deliberately. The hand offsets are a physical mounting lever (a VR
tracker 0.18 m out, or the 0.242 m fingertip TCP) and the head offset lifts torso_link to
head height; there is no such lever for a foot — maskedmimic tracks the raw
`*_ankle_roll_link` body, and `robot_config.BODY_NAME_TO_MJ_IDX` even maps
`left_foot_contact_point` onto the same MuJoCo index as `left_ankle_roll_link`. A nonzero
sole offset would price a point no other term in the stack knows about.

HANDS STAY ON THE 0.242 m TCP. This is the *unified* family: `VR_3POINT_OFFSETS_TCP` put
both hands on the force anchor so position and force are supervised at ONE point (see
`observation.vr_3point_local_target_tcp` and `tcp_precision.py`). The 5-point offsets are
built FROM that constant, so if the gripper is respecced and `hybrid.TCP_OFFSET_X` moves,
the 3pt and 5pt presets move together — there is no second literal to forget.

---------------------------------------------------------------------------
THE DENSE VARIANT
---------------------------------------------------------------------------
`exp:g1-wbt`'s actor obs is NOT keypoints at all: `motion_command` is
`cat(joint_pos, joint_vel)` = 58D, i.e. all 29 joint targets. That already subsumes the
24D `command_lower_body` slice the keypoint presets carry, so the dense layout drops it and
carries no `vr_*point_*` term whatsoever. Force handling is unchanged: F^cmd, F^meas,
`contact_phase`, I and Δx are inserted as one block right after the tracking block, exactly
where `hybrid.py` puts them in the 2pt layout.

The two `tcp_precision` reward terms still apply and still matter here — they read the
motion command and the robot data views directly, never the obs — so the dense policy is
still explicitly paid for putting each hand's 0.242 m TCP where the reference wants it, in
both translation (w=4.0, σ=0.10 m) and attitude (w=2.0, σ=0.25 rad). That is the "make the
two hands' TCP tracking better" requirement, satisfied by the SAME reward object the 3pt
anneal twin uses rather than by a re-tuned copy.

---------------------------------------------------------------------------
CAVEATS WORTH SAYING OUT LOUD
---------------------------------------------------------------------------
* Both are NEW obs widths (184D / 183D), so NO existing checkpoint warm-starts into them —
  `pfl96opy` and friends are 170D. These presets train from scratch.
* Orientation stays a QUATERNION (4D per body), matching the pelvis-local unified family.
  The 6D rep only lives in the `-global-vel` family. The known quaternion double-cover OOD
  at deploy therefore applies here exactly as it does to `exp:g1-wbt-3pt-hybrid-unified`;
  this is inherited-by-design, not a new decision.
* Like the 3pt local twins these are root-invariant (no `motion_ref_pos_b`/`_vel_b`/
  `base_lin_vel` in the actor), so they track in place and do not walk.
* Deploy is NOT wired: `wbt_wrappers_inference` has no 184D/183D preset and the mjswan viz
  scripts are keyed to the 170D ones. Training only, for now.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.config_types.observation import ObsGroupCfg, ObservationManagerCfg, ObsTermCfg

from wbt_training.config_values.hybrid import g1_29dof_wbt_observation_2pt_hybrid
from wbt_training.config_values.hybrid_unified import _actor_obs_local_unified_terms
from wbt_training.config_values.observation import (
    _actor_obs_terms as _actor_obs_dense_terms,
    _vr_local_orn_target,
    _vr_local_target,
)
from wbt_training.config_values.robot_config import VR_3POINT_BODIES, VR_3POINT_OFFSETS_TCP

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


# ---------------------------------------------------------------------------
# 5-point tracked bodies + offsets (TCP-unified hands, unchanged head, bare feet).
#
# Order = the 3pt list VERBATIM, then the two feet appended. Spelled as
# `[*VR_3POINT_BODIES, ...]` rather than re-typing the three names so the two lists cannot
# drift, and so the 3pt prefix property is structural instead of a comment.
# ---------------------------------------------------------------------------
VR_5POINT_FEET_BODIES: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")

VR_5POINT_BODIES: list[str] = [*VR_3POINT_BODIES, *VR_5POINT_FEET_BODIES]

# Feet are tracked at the ankle_roll_link ORIGIN — no lever (see the module docstring).
VR_5POINT_FOOT_OFFSET: list[float] = [0.0, 0.0, 0.0]

VR_5POINT_OFFSETS_TCP: list[list[float]] = [
    *[list(o) for o in VR_3POINT_OFFSETS_TCP],  # L hand @ TCP, R hand @ TCP, head @ +0.35 z
    list(VR_5POINT_FOOT_OFFSET),
    list(VR_5POINT_FOOT_OFFSET),
]


def vr_5point_local_target_tcp(env: WholeBodyTrackingManager) -> torch.Tensor:
    """5-point position target in the pelvis-local reference frame, 15D.

    `vr_3point_local_target_tcp` (hands on the 0.242 m force TCP, head at torso_link
    +0.35 z) with the two `*_ankle_roll_link` origins appended, so slots [0:9] are the 3pt
    vector unchanged and [9:15] are left foot then right foot.

    A DISTINCT term name — not a re-parameterised `vr_3point_pos_tcp` — for the same reason
    the TCP term itself is distinct (observation.py:759): the term name recorded in a
    checkpoint's config is what states which geometry it trained on. Width differs too
    (15 vs 9), so a mix-up would fail loudly rather than silently; the naming is belt AND
    braces.

    PURE READER (delegates to `observation._vr_local_target`, which caches the offset
    tensor keyed on bodies AND offsets — so this term cannot poison the 3pt term's cache).
    """
    return _vr_local_target(env, VR_5POINT_BODIES, VR_5POINT_OFFSETS_TCP)


def vr_5point_local_orn_target(env: WholeBodyTrackingManager) -> torch.Tensor:
    """5-point orientation target in the pelvis-local reference frame, 20D = 5 bodies × 4.

    QUATERNION, matching the pelvis-local unified family's `vr_3point_orn` (the 6D rep is
    the `-global-vel` family's). Slots [0:12] are the 3pt vector unchanged.

    Offsets are irrelevant to orientation (a rigid offset does not rotate the body), so no
    offsets argument — same as `vr_3point_local_orn_target`. PURE READER.
    """
    return _vr_local_orn_target(env, VR_5POINT_BODIES)


# ---------------------------------------------------------------------------
# 184D 5-point actor obs = the 170D local-unified layout with vr_3point_{pos_tcp,orn}
# (9+12) swapped for vr_5point_{pos_tcp,orn} (15+20). Slot KEYS and their order are
# otherwise identical to the 170D layout (obs00..obs13), and every other slot is the SAME
# ObsTermCfg object — including obs05_contact_phase and the obs12 JointVelDropout wrapper
# (default OFF, `WBT_JV_DROPOUT_*`).
#
# Zero-padded two-digit keys: the obs manager concatenates in `sorted(keys())` ALPHABETICAL
# order, so a one-digit scheme would string-sort "obs10" between "obs1" and "obs2".
#
# ⚠️ FIXED WIDTH 184D regardless of servo mode / dropout: the I and Δx channels are present
# in every run and zeroed when the servo is off (the injector allocates unconditionally).
# ---------------------------------------------------------------------------
_L = _actor_obs_local_unified_terms

_VR_5POINT_POS_TCP_TERM = ObsTermCfg(
    func="wbt_training.config_values.hybrid_unified_5pt_dense:vr_5point_local_target_tcp", scale=1.0, noise=0.0
)
_VR_5POINT_ORN_TERM = ObsTermCfg(
    func="wbt_training.config_values.hybrid_unified_5pt_dense:vr_5point_local_orn_target", scale=1.0, noise=0.0
)

_actor_obs_5pt_unified_terms = {
    "obs00_command_lower_body": _L["obs00_command_lower_body"],
    "obs01_vr_5point_pos_tcp": _VR_5POINT_POS_TCP_TERM,
    "obs02_vr_5point_orn": _VR_5POINT_ORN_TERM,
    "obs03_force_cmd_local": _L["obs03_force_cmd_local"],
    "obs04_force_meas_local": _L["obs04_force_meas_local"],
    "obs05_contact_phase": _L["obs05_contact_phase"],
    "obs06_integrator_state": _L["obs06_integrator_state"],
    "obs07_x_tgt_shift": _L["obs07_x_tgt_shift"],
    "obs08_motion_ref_ori_b": _L["obs08_motion_ref_ori_b"],
    "obs09_projected_gravity": _L["obs09_projected_gravity"],
    "obs10_base_ang_vel": _L["obs10_base_ang_vel"],
    "obs11_dof_pos": _L["obs11_dof_pos"],
    "obs12_dof_vel": _L["obs12_dof_vel"],  # JointVelDropout wrapper, default OFF
    "obs13_actions": _L["obs13_actions"],
}  # 24+15+20+6+6+2+6+6+6+3+3+29+29+29 = 184D actor (736D w/ history 4)

# Critic reused from the 2pt-hybrid preset VERBATIM — it is tracking-point-agnostic
# (`robot_body_pos_b` / `robot_body_ori_b` already cover all 14 tracked bodies, feet
# included) and never subject to the actor's dropout.
_critic_obs_hybrid_terms = g1_29dof_wbt_observation_2pt_hybrid.groups["critic_obs"].terms

g1_29dof_wbt_observation_5pt_unified_hybrid = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_5pt_unified_terms
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=10, terms=_critic_obs_hybrid_terms
        ),
    }
)


# ---------------------------------------------------------------------------
# 183D DENSE actor obs = `exp:g1-wbt`'s 157D TML layout (motion_command 58D = all 29 joint
# targets + velocities — no keypoints, no command_lower_body) with the SAME force block the
# keypoint layouts carry (F^cmd, F^meas, contact_phase, I, Δx = 26D) inserted right after
# the tracking term, i.e. the position `hybrid.py` uses in the 2pt layout.
#
# Every non-force slot is the SAME ObsTermCfg object as `observation._actor_obs_terms`
# EXCEPT dof_vel, which becomes the JointVelDropout wrapper so the dense preset inherits
# the 3pt-hybrid lineage's actor-only dof_vel robustness knob (default OFF; width-neutral
# and behaviourally identical when off — the wrapper applies the same base 0.5 noise
# internally that `_actor_obs_terms["obs5_dof_vel"]` declares).
#
# 183D happens to equal `exp:g1-wbt-3pt-hybrid-global-vel`'s width. That is a coincidence of
# arithmetic, not a shared layout — the term names differ completely, so nothing can load
# across them by accident.
# ---------------------------------------------------------------------------
_D = _actor_obs_dense_terms

_actor_obs_dense_unified_terms = {
    "obs00_motion_command": _D["obs0_motion_command"],
    "obs01_force_cmd_local": _L["obs03_force_cmd_local"],
    "obs02_force_meas_local": _L["obs04_force_meas_local"],
    "obs03_contact_phase": _L["obs05_contact_phase"],
    "obs04_integrator_state": _L["obs06_integrator_state"],
    "obs05_x_tgt_shift": _L["obs07_x_tgt_shift"],
    "obs06_motion_ref_ori_b": _D["obs1_motion_ref_ori_b"],
    "obs07_projected_gravity": _D["obs2_projected_gravity"],
    "obs08_base_ang_vel": _D["obs3_base_ang_vel"],
    "obs09_dof_pos": _D["obs4_dof_pos"],
    "obs10_dof_vel": _L["obs12_dof_vel"],  # JointVelDropout wrapper, default OFF
    "obs11_actions": _D["obs6_actions"],
}  # 58+6+6+2+6+6+6+3+3+29+29+29 = 183D actor (732D w/ history 4)

g1_29dof_wbt_observation_dense_unified_hybrid = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_dense_unified_terms
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=10, terms=_critic_obs_hybrid_terms
        ),
    }
)


__all__ = [
    "VR_5POINT_BODIES",
    "VR_5POINT_FEET_BODIES",
    "VR_5POINT_FOOT_OFFSET",
    "VR_5POINT_OFFSETS_TCP",
    "vr_5point_local_target_tcp",
    "vr_5point_local_orn_target",
    "g1_29dof_wbt_observation_5pt_unified_hybrid",
    "g1_29dof_wbt_observation_dense_unified_hybrid",
]
