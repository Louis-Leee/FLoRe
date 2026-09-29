"""3-point hybrid position-force controller + global root pos/vel + 6D orientation
(`exp:g1-wbt-3pt-hybrid-global-vel`).

Takes `exp:g1-wbt-3pt-hybrid` (the tracking-agnostic hybrid Layer A + B controller on
the 3-point base) and, exactly the way `exp:g1-wbt-global-vel` extends `exp:g1-wbt` over
`exp:g1-wbt-global`, folds the ROOT-STATE signals into the actor observation:

  * ``motion_ref_pos_b`` (3D) — the base-frame target-vs-current root position error,
  * ``motion_ref_vel_b`` (3D) — the reference root linear velocity in the robot base frame,
  * ``base_lin_vel``     (3D) — the robot's own base linear velocity,

so the policy can command the base to TRANSLATE through the world (plain 3pt-hybrid is
root-invariant / pelvis-local and cannot walk). Plus the ONE representation change the
user asked for: the 3-point tracked-body ORIENTATION is a 6D rotation rep
(``vr_3point_orn_6d`` = 3 bodies × 6 = 18D) instead of the plain-3pt quaternion (3 × 4 =
12D). 6D has no double-cover / antipodal OOD branch, so it round-trips train↔deploy
cleanly — the same reason 2pt-hybrid uses 6D for ``vr_2point_orn`` (observation.py:756).

Net actor obs: 168D (3pt-hybrid) → 183D. Breakdown vs 3pt-hybrid:
  + motion_ref_pos_b (3) + motion_ref_vel_b (3) + base_lin_vel (3)   = +9  (root pos/vel)
  + vr_3point_orn 12→18                                              = +6  (quat → 6D)
  = 168 + 15 = 183D actor (732D with history_length=4).

WHY THIS PORT IS TINY (identical philosophy to hybrid_3pt.py):
  * The whole hybrid physics/reward/command/curriculum/termination stack in `hybrid.py`
    is tracking-agnostic AND observation-agnostic — root pos/vel is an OBSERVATION signal,
    NOT a reward change (verified: `exp:g1-wbt-global` / `-global-vel` add these to the obs
    only and leave the reward/command untouched; the absolute-world anchor reward
    `motion_global_ref_position_error_exp` @ 0.5 is already in the core reward the hybrid
    reward inherits). So the SHIPPED servo-ON command/reward/curriculum/termination (Layer
    A + B, force servo ON) are reused VERBATIM from `hybrid.py` by experiment.py — this
    preset is `replace(g1_wbt_3pt_hybrid, observation=<the 183D obs below>)`.
  * F^cmd / F^meas / F^ext all stay in the hybrid injector's root frame (hybrid.py) — the
    force/servo channels are byte-identical to 3pt-hybrid.
  * The ACTOR joint-velocity dropout (`JointVelDropout`, default OFF) is reused from
    `hybrid_3pt.py` VERBATIM (same class, same `WBT_JV_DROPOUT_*` env knobs, same default).

NON-INVASION: this module only READS observation.py / hybrid.py / hybrid_3pt.py. It never
edits `hybrid.py` / `force_tracking.py` / `observation.py` (hard rule). The only genuinely
new code is `vr_3point_local_orn_target_6d` (one line delegating to the existing 6D helper).
The existing quaternion `vr_3point_orn` (used by `exp:g1-wbt-3pt-hybrid`, 3pt, 3pt-global)
is left byte-for-byte untouched — the 6D term is a DISTINCT name (`vr_3point_orn_6d`), so
exp:g1-wbt-3pt-hybrid and its shipped inference/mjswan configs do not shift.
"""

from __future__ import annotations

import torch

from holosoma.config_types.observation import ObsGroupCfg, ObservationManagerCfg, ObsTermCfg

from wbt_training.config_values.hybrid import g1_29dof_wbt_observation_2pt_hybrid
from wbt_training.config_values.hybrid_3pt import _actor_obs_3pt_hybrid_terms
from wbt_training.config_values.observation import _actor_obs_3pt_global_vel_terms, _vr_local_orn_target
from wbt_training.config_values.robot_config import VR_3POINT_BODIES

# Re-export the reused dropout term so it and its default config are importable from this
# module too (the actor obs11→obs14 dof_vel term reuses the hybrid_3pt JointVelDropout class
# by dotted path — see below). Keeps `WBT_JV_DROPOUT_*` behaviour identical to 3pt-hybrid.
from wbt_training.config_values.hybrid_3pt import JointVelDropout  # noqa: F401


# ---------------------------------------------------------------------------
# The ONE genuinely-new obs function: 6D 3-point tracked-body orientation.
# ---------------------------------------------------------------------------
def vr_3point_local_orn_target_6d(env) -> torch.Tensor:
    """3-point tracked-body orientation in the REFERENCE-MOTION anchor (pelvis) frame as a
    continuous 6D rotation rep — 3 bodies × 6 = 18D (vs the plain-3pt quaternion 3 × 4 = 12D).

    Delegates to the SAME shared helper the 2-point 6D term uses
    (observation._vr_local_orn_target with return_6d=True — the first-two-columns rep,
    observation.py:720-731), just with VR_3POINT_BODIES. 6D removes the quaternion
    double-cover / antipodal OOD branch that bites at deploy (observation.py:756). PURE
    READER; frame is unchanged from the quaternion term (reference-motion pelvis frame)."""
    return _vr_local_orn_target(env, VR_3POINT_BODIES, return_6d=True)


# ---------------------------------------------------------------------------
# 183D 3pt-hybrid-global-vel actor observation + reused hybrid critic.
#
# = the 3pt-hybrid 168D layout with (a) vr_3point_orn swapped quat(12)→6D(18) and (b) the
# root-state trio motion_ref_pos_b / motion_ref_vel_b / base_lin_vel inserted at the SAME
# positions exp:g1-wbt-global-vel uses (pos/vel right after the tracking orientation; robot
# base_lin_vel right after base_ang_vel). Force/servo terms (F^cmd, F^meas, I, Δx) keep
# their 3pt-hybrid slots; dof_vel is the reused JointVelDropout term (default OFF).
#
# ⚠️ Zero-padded sorted keys obs00..obs15 — the obs manager concatenates in sorted(keys())
# ALPHABETICAL order, so the numeric prefixes MUST be two-digit (a 1-digit scheme sorts
# "obs10" between "obs1" and "obs2" and scrambles the vector).
#
# ⚠️ FIXED WIDTH — ALWAYS 183D regardless of servo_enabled (I/Δx buffers allocated
# unconditionally in the hybrid injector, zeroed when servo off) AND regardless of dropout
# on/off (obs14 stays a 29D dof_vel term either way) → one network + checkpoint lineage for
# servo on/off and the `-servo-off` two-phase resume.
#
# EVERY reused term is the SAME ObsTermCfg object as its source (verbatim reuse, so a diff
# is trivially auditable): the plain / force / servo / dropout terms come from
# hybrid_3pt._actor_obs_3pt_hybrid_terms, and the root trio comes from observation's
# _actor_obs_3pt_global_vel_terms (identical scale/noise as exp:g1-wbt-3pt-global-vel).
# ---------------------------------------------------------------------------
_actor_obs_3pt_hybrid_global_vel_terms = {
    "obs00_command_lower_body": _actor_obs_3pt_hybrid_terms["obs00_command_lower_body"],
    "obs01_vr_3point_pos": _actor_obs_3pt_hybrid_terms["obs01_vr_3point_pos"],
    # 6D orientation (18D) — the one genuinely-new term. DISTINCT name from the quaternion
    # vr_3point_orn so exp:g1-wbt-3pt-hybrid (quat) is untouched.
    "obs02_vr_3point_orn_6d": ObsTermCfg(
        func="wbt_training.config_values.hybrid_3pt_global_vel:vr_3point_local_orn_target_6d", scale=1.0, noise=0.0
    ),
    # Root position + reference root velocity (exp:g1-wbt-global-vel signals), reused verbatim.
    "obs03_motion_ref_pos_b": _actor_obs_3pt_global_vel_terms["obs03_motion_ref_pos_b"],
    "obs04_motion_ref_vel_b": _actor_obs_3pt_global_vel_terms["obs04_motion_ref_vel_b"],
    # Force / servo channels (F^cmd, F^meas, I, Δx) — reused verbatim from 3pt-hybrid.
    "obs05_force_cmd_local": _actor_obs_3pt_hybrid_terms["obs03_force_cmd_local"],
    "obs06_force_meas_local": _actor_obs_3pt_hybrid_terms["obs04_force_meas_local"],
    "obs07_integrator_state": _actor_obs_3pt_hybrid_terms["obs05_integrator_state"],
    "obs08_x_tgt_shift": _actor_obs_3pt_hybrid_terms["obs06_x_tgt_shift"],
    "obs09_motion_ref_ori_b": _actor_obs_3pt_hybrid_terms["obs07_motion_ref_ori_b"],
    "obs10_projected_gravity": _actor_obs_3pt_hybrid_terms["obs08_projected_gravity"],
    "obs11_base_ang_vel": _actor_obs_3pt_hybrid_terms["obs09_base_ang_vel"],
    # Robot base linear velocity (exp:g1-wbt-global-vel signal), reused verbatim.
    "obs12_base_lin_vel": _actor_obs_3pt_global_vel_terms["obs08_base_lin_vel"],
    "obs13_dof_pos": _actor_obs_3pt_hybrid_terms["obs10_dof_pos"],
    # ACTOR-only joint-velocity dropout (default OFF), reused verbatim from hybrid_3pt.
    "obs14_dof_vel": _actor_obs_3pt_hybrid_terms["obs11_dof_vel"],
    "obs15_actions": _actor_obs_3pt_hybrid_terms["obs12_actions"],
}  # 24+9+18+3+3+6+6+6+6+6+3+3+3+29+29+29 = 183D actor (732D w/ history_length=4)

# Critic reused from the 2pt-hybrid preset VERBATIM (tracking-agnostic; already carries the
# privileged motion_ref_pos_b + base_lin_vel critic terms via critic_obs_shared_terms, so
# NO critic change is needed to add the actor root signals — mirrors exp:g1-wbt-global-vel,
# whose critic is likewise critic_obs_tml unchanged). NEVER subject to the dropout.
_critic_obs_3pt_hybrid_global_vel_terms = g1_29dof_wbt_observation_2pt_hybrid.groups["critic_obs"].terms

g1_29dof_wbt_observation_3pt_hybrid_global_vel = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_3pt_hybrid_global_vel_terms
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=10, terms=_critic_obs_3pt_hybrid_global_vel_terms
        ),
    }
)

__all__ = ["vr_3point_local_orn_target_6d", "g1_29dof_wbt_observation_3pt_hybrid_global_vel"]
