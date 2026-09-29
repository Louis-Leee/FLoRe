"""Experiment presets for G1 29-DOF whole-body tracking.

These reuse the holosoma core WBT reward and termination configs,
customising the command, observation, and algorithm configs to match
TML-BeyondMimic-Generalist so that trained checkpoints are directly
deployable with ``wbt_wrappers_inference``.

Key changes vs. the core holosoma WBT defaults:

- **Observation**: adds ``projected_gravity`` (154D -> 157D), actor
  history_length=4, critic history_length=10 (matching TML).
- **Network**: PPO actor/critic hidden dims [2048, 2048, 1024, 1024, 512, 512]
  (matching TML's ~50x larger architecture).
- **Hyperparameters**: PPO learning rate 1e-3, Adam optimizer (no weight
  decay), 5 learning epochs, 30k iterations, save every 500 steps.
- **FastSAC**: larger hidden dims (actor 1024, critic 1536) to better match
  the TML-scale capacity, while keeping FastSAC-specific algo settings.
"""

from __future__ import annotations

from dataclasses import replace

import tyro
from typing_extensions import Annotated

from holosoma.config_types.algo import LayerConfig, ModuleConfig, OptimizerConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import DEFAULTS as CORE_DEFAULTS
from holosoma.config_values.logger import wandb as wandb_logger
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt, g1_29dof_wbt_fast_sac
from holosoma.config_values.robot import g1_29dof as _g1_29dof_robot
from holosoma.config_values import simulator

try:
    from holosoma.config_values.wbt.g1.randomization import (
        g1_29dof_wbt_randomization_aggressive,
        g1_29dof_wbt_randomization_ee_mass,
    )
except ImportError:
    from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization as _base_rand

    g1_29dof_wbt_randomization_aggressive = _base_rand
    g1_29dof_wbt_randomization_ee_mass = _base_rand
from wbt_training.config_values.terrain import (
    g1_29dof_joint_pos_implicit,
    g1_29dof_wbt_terrain_command,
    g1_29dof_wbt_terrain_command_pelvis,
    g1_29dof_wbt_terrain_observation,
    g1_29dof_wbt_terrain_observation_no_heightscan,
    g1_29dof_wbt_terrain_randomization,
    g1_29dof_wbt_terrain_reward,
    g1_29dof_wbt_terrain_termination,
)
from wbt_training.config_values import command
from wbt_training.config_values.maskedmimic_observations import (
    g1_29dof_wbt_maskedmimic_reward,
    g1_29dof_wbt_maskedmimic_termination,
    g1_29dof_wbt_observation_maskedmimic,
)
from wbt_training.config_values.observation import (
    g1_29dof_wbt_observation_1pt,
    g1_29dof_wbt_observation_1pt_fastsac,
    g1_29dof_wbt_observation_2pt,
    g1_29dof_wbt_observation_2pt_fastsac,
    g1_29dof_wbt_observation_2pt_global,
    g1_29dof_wbt_observation_3pt,
    g1_29dof_wbt_observation_3pt_fastsac,
    g1_29dof_wbt_observation_3pt_global,
    g1_29dof_wbt_observation_3pt_global_vel,
    g1_29dof_wbt_observation_chip_global,
    g1_29dof_wbt_observation_chip_local,
    g1_29dof_wbt_observation_fastsac,
    g1_29dof_wbt_observation_global,
    g1_29dof_wbt_observation_global_fastsac,
    g1_29dof_wbt_observation_global_vel,
    g1_29dof_wbt_observation_global_vel_fastsac,
    g1_29dof_wbt_observation_tml,
    g1_29dof_wbt_observation_wo_se,
    g1_29dof_wbt_observation_wo_se_fastsac,
)

# ---------------------------------------------------------------------------
# TML-aligned network architecture for PPO
# ---------------------------------------------------------------------------
_tml_hidden_dims = [2048, 2048, 1024, 1024, 512, 512]

_tml_actor_module = ModuleConfig(
    type="MLP",
    input_dim=["actor_obs"],
    output_dim=["robot_action_dim"],
    layer_config=LayerConfig(hidden_dims=_tml_hidden_dims, activation="ELU"),
)

_tml_critic_module = ModuleConfig(
    type="MLP",
    input_dim=["critic_obs"],
    output_dim=[1],
    layer_config=LayerConfig(hidden_dims=_tml_hidden_dims, activation="ELU"),
)

_adam_no_wd = OptimizerConfig(_target_="torch.optim.Adam", weight_decay=0.0)

# ---------------------------------------------------------------------------
# PPO preset aligned with TML-BeyondMimic-Generalist
# ---------------------------------------------------------------------------
g1_wbt = replace(
    g1_29dof_wbt,
    training=replace(g1_29dof_wbt.training, project="WBT-Holosoma"),
    command=command.g1_29dof_wbt_command,
    observation=g1_29dof_wbt_observation_tml,
    logger=wandb_logger,
    algo=replace(
        g1_29dof_wbt.algo,
        config=replace(
            g1_29dof_wbt.algo.config,
            # TML-aligned network architecture
            module_dict=replace(
                g1_29dof_wbt.algo.config.module_dict, actor=_tml_actor_module, critic=_tml_critic_module
            ),
            # TML-aligned hyperparameters
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            actor_optimizer=_adam_no_wd,
            critic_optimizer=_adam_no_wd,
            num_learning_epochs=5,
            num_learning_iterations=30000,
            save_interval=500,
        ),
    ),
)

# ---------------------------------------------------------------------------
# FastSAC preset with larger networks and TML observation
# ---------------------------------------------------------------------------
g1_wbt_fast_sac = replace(
    g1_29dof_wbt_fast_sac,
    training=replace(g1_29dof_wbt_fast_sac.training, project="WBT-Holosoma"),
    command=command.g1_29dof_wbt_command,
    observation=g1_29dof_wbt_observation_fastsac,
    logger=wandb_logger,
    algo=replace(
        g1_29dof_wbt_fast_sac.algo,
        config=replace(
            g1_29dof_wbt_fast_sac.algo.config,
            # Larger hidden dims to better match TML-scale capacity
            actor_hidden_dim=1024,
            critic_hidden_dim=1536,
        ),
    ),
)

# ---------------------------------------------------------------------------
# 3-point tracking presets (TML-aligned)
# ---------------------------------------------------------------------------
# PPO 3pt: same algo as g1_wbt but with 3-point actor observation (144D)
g1_wbt_3pt = replace(g1_wbt, observation=g1_29dof_wbt_observation_3pt)

# FastSAC 3pt: same algo as g1_wbt_fast_sac but with 3-point actor observation
g1_wbt_3pt_fast_sac = replace(g1_wbt_fast_sac, observation=g1_29dof_wbt_observation_3pt_fastsac)

# ---------------------------------------------------------------------------
# 3-point tracking + root position (147D actor obs)
# ---------------------------------------------------------------------------
g1_wbt_3pt_global = replace(g1_wbt, observation=g1_29dof_wbt_observation_3pt_global)

# ---------------------------------------------------------------------------
# 3-point tracking + root position + root velocity (153D actor obs)
# ---------------------------------------------------------------------------
g1_wbt_3pt_global_vel = replace(g1_wbt, observation=g1_29dof_wbt_observation_3pt_global_vel)

# ---------------------------------------------------------------------------
# 2-point tracking presets (left wrist + right wrist, 141D actor obs)
# ---------------------------------------------------------------------------
g1_wbt_2pt = replace(g1_wbt, observation=g1_29dof_wbt_observation_2pt)

g1_wbt_2pt_global = replace(g1_wbt, observation=g1_29dof_wbt_observation_2pt_global)

g1_wbt_2pt_fast_sac = replace(g1_wbt_fast_sac, observation=g1_29dof_wbt_observation_2pt_fastsac)

# ---------------------------------------------------------------------------
# 2-point + per-hand UniFP-style force-tracking preset (147D actor obs = 141D
# 2pt baseline + 6D force_cmd_local).
#
# Adds vs g1_wbt_2pt:
#   - HandForceInjector (CommandTerm): substep PhysX F^ext wrench at the
#     *_rubber_hand contact point (applied on *_wrist_yaw_link + positions=offset).
#   - obs: force_cmd_local (6D) on the ACTOR (root frame); force_cmd_local +
#     force_ext_local (6D each) on the CRITIC only (UniFP actor/critic asymmetry).
#   - reward: force_track_pos_eq2 (w=6.0) + feet_slip (w=-2.0); the baseline
#     14-link relative-POSITION term is swapped for a no_wrist (12-body) variant
#     (wrist position → Eq.2). The 14-link relative-ORIENTATION term is KEPT
#     (incl. wrist_yaw == hand orientation). force_track_pos_pen is defined but
#     NOT wired.
#   - ForceAlphaCurriculum: REGISTERED but DISABLED BY DEFAULT (force_alpha_init
#     =1.0, alpha_step=0.0 → full force from step 0). To enable the 0→1 ramp set
#     force_alpha_init=0.0 + alpha_step=0.01 in HandForceConfig.
# ---------------------------------------------------------------------------
from wbt_training.config_values.force_tracking import (  # noqa: E402
    g1_29dof_wbt_2pt_force_ext_reward,
    g1_29dof_wbt_2pt_force_reward,
    g1_29dof_wbt_force_command,
    g1_29dof_wbt_force_curriculum,
    g1_29dof_wbt_observation_2pt_force,
    g1_29dof_wbt_observation_2pt_force_ext,
)

g1_wbt_2pt_force = replace(
    g1_wbt_2pt,
    command=g1_29dof_wbt_force_command,
    observation=g1_29dof_wbt_observation_2pt_force,
    reward=g1_29dof_wbt_2pt_force_reward,
    curriculum=g1_29dof_wbt_force_curriculum,
)

# ---------------------------------------------------------------------------
# F^ext-in-actor variant (exp:g1-wbt-2pt-force-ext).
#
# = g1_wbt_2pt_force reverted to its ORIGINAL first-commit design (the version
# behind wandb run 09wvlz16), as a SEPARATE preset so g1_wbt_2pt_force and
# g1_wbt_2pt_hybrid are both left untouched. Two deltas vs g1_wbt_2pt_force:
#   (1) obs — F^ext is ADDED BACK to the ACTOR (actor sees F^cmd + F^ext, 153D;
#       drops the UniFP actor/critic asymmetry). Critic obs unchanged.
#   (2) reward — the earliest ("最最开始") weights: FULL 14-body position tracking
#       kept (no no_wrist swap → the whole body is tracked, not just the wrist),
#       + force_track_pos_eq2 @ 4.0 (not the later 6.0) + force_track_pos_pen @
#       -6.0. No feet_slip.
# Plus the yam URDF (g1/g1_29dof_yam.urdf) baked in (matches the other force
# presets). Command / curriculum / algo / force machinery inherited unchanged
# (K=200, σ=0.3, curriculum off → full force from step 0). Pure UniFP position-
# overshoot: NO F_meas, NO servo. See config_values/force_tracking.py.
# ---------------------------------------------------------------------------
g1_wbt_2pt_force_ext = replace(
    g1_wbt_2pt_force,
    robot=replace(
        g1_wbt_2pt_force.robot, asset=replace(g1_wbt_2pt_force.robot.asset, urdf_file="g1/g1_29dof_yam.urdf")
    ),
    observation=g1_29dof_wbt_observation_2pt_force_ext,
    reward=g1_29dof_wbt_2pt_force_ext_reward,
)

# ---------------------------------------------------------------------------
# Single-actor upper-body-precision preset (exp:g1-wbt-2pt-force-upperbody).
#
# = g1_wbt_2pt_force with three deltas: (1) yam URDF baked in; (2) reward adds the
# relative-frame TCP fingertip precision terms (upperbody.py); (3) termination drops
# the two ankles from the body-position gate. The lower body stands by KEEPING the
# inherited clip-leg-tracking (standing-manip clips => the clip is a valid standing
# pose), so no action mask / env subclass / stand reward is needed. Obs / command /
# curriculum / algo / force machinery inherited unchanged (Eq.2=6.0, curriculum off).
# See docs/superpowers/plans/2026-07-06-g1-wbt-2pt-force-upperbody.md.
# ---------------------------------------------------------------------------
from wbt_training.config_values.upperbody import (  # noqa: E402
    g1_29dof_wbt_2pt_force_upperbody_reward,
    g1_29dof_wbt_upperbody_termination,
)

g1_wbt_2pt_force_upperbody = replace(
    g1_wbt_2pt_force,
    robot=replace(
        g1_wbt_2pt_force.robot, asset=replace(g1_wbt_2pt_force.robot.asset, urdf_file="g1/g1_29dof_yam.urdf")
    ),
    reward=g1_29dof_wbt_2pt_force_upperbody_reward,
    termination=g1_29dof_wbt_upperbody_termination,
)

# ---------------------------------------------------------------------------
# Hybrid position-force controller v1: causal K_env contact + simulated F/T sensor
# (F_meas in the actor obs), forked from the PLAIN 2pt preset (not 2pt-force) with a
# torso-stable rebalanced reward (force_track_pos_eq2 3.0 not 6.0; rel-orientation 2.0;
# ang-vel sigma 2.0). yam URDF baked in. Layer B (servo) adds `termination=` in Task 11.
# See docs/superpowers/plans/2026-07-09-g1-wbt-2pt-hybrid-v1.md.
# ---------------------------------------------------------------------------
from wbt_training.config_values.hybrid import (  # noqa: E402
    g1_29dof_wbt_2pt_hybrid_reward,
    g1_29dof_wbt_2pt_hybrid_reward_servo_off,
    g1_29dof_wbt_hybrid_command,
    g1_29dof_wbt_hybrid_command_servo_off,
    g1_29dof_wbt_hybrid_curriculum,
    g1_29dof_wbt_hybrid_termination,
    g1_29dof_wbt_hybrid_termination_servo_off,
    g1_29dof_wbt_observation_2pt_hybrid,
)

g1_wbt_2pt_hybrid = replace(
    g1_wbt_2pt,
    robot=replace(g1_wbt_2pt.robot, asset=replace(g1_wbt_2pt.robot.asset, urdf_file="g1/g1_29dof_yam.urdf")),
    command=g1_29dof_wbt_hybrid_command,
    observation=g1_29dof_wbt_observation_2pt_hybrid,
    reward=g1_29dof_wbt_2pt_hybrid_reward,
    curriculum=g1_29dof_wbt_hybrid_curriculum,
    termination=g1_29dof_wbt_hybrid_termination,  # core + force_runaway estop (servo shipped ON)
)

# Phase-1 (servo-OFF) variant for the two-phase curriculum (user 2026-07-10): train the stand +
# position tracking to convergence here, then resume the SAME checkpoint under g1_wbt_2pt_hybrid
# (servo ON) via `--training.checkpoint <phase1.pt>`. Obs is fixed 165D in BOTH so the network
# transfers with no surgery; only servo_enabled flips. See config_values/hybrid.py.
g1_wbt_2pt_hybrid_servo_off = replace(
    g1_wbt_2pt_hybrid,
    command=g1_29dof_wbt_hybrid_command_servo_off,
    reward=g1_29dof_wbt_2pt_hybrid_reward_servo_off,
    termination=g1_29dof_wbt_hybrid_termination_servo_off,
)

# ---------------------------------------------------------------------------
# 3-point hybrid position-force controller (exp:g1-wbt-3pt-hybrid).
#
# = g1_wbt_3pt (3-point tracking base) + the SAME hybrid stack as g1_wbt_2pt_hybrid. Because
# g1_wbt_2pt_hybrid and g1_wbt_3pt both fork g1_wbt with the SAME command/reward/curriculum/
# termination, the whole hybrid controller (causal K_env contact, sim F/T sensor with F_meas
# in the actor, Jᵀ servo shipped ON, force-runaway estop) is tracking-point-agnostic and
# reused VERBATIM from hybrid.py — the ONLY delta is the actor observation (vr_3point instead
# of vr_2point, 168D) + an ACTOR joint-velocity dropout (WBT_JV_DROPOUT_* env knobs, DEFAULT
# OFF; hybrid_3pt.py). yam URDF baked in. See docs/superpowers/specs/2026-07-16-g1-wbt-3pt-hybrid-design.md.
# ---------------------------------------------------------------------------
from wbt_training.config_values.hybrid_3pt import g1_29dof_wbt_observation_3pt_hybrid  # noqa: E402

g1_wbt_3pt_hybrid = replace(
    g1_wbt_3pt,
    robot=replace(g1_wbt_3pt.robot, asset=replace(g1_wbt_3pt.robot.asset, urdf_file="g1/g1_29dof_yam.urdf")),
    command=g1_29dof_wbt_hybrid_command,
    observation=g1_29dof_wbt_observation_3pt_hybrid,
    reward=g1_29dof_wbt_2pt_hybrid_reward,
    curriculum=g1_29dof_wbt_hybrid_curriculum,
    termination=g1_29dof_wbt_hybrid_termination,  # core + force_runaway estop (servo shipped ON)
)

# Phase-1 (servo-OFF) variant for the two-phase curriculum: train the stand + position
# tracking to convergence here, then resume the SAME checkpoint under g1_wbt_3pt_hybrid
# (servo ON) via `--training.checkpoint <phase1.pt>`. Obs is fixed 168D in BOTH so the
# network transfers with no surgery; only servo_enabled flips.
g1_wbt_3pt_hybrid_servo_off = replace(
    g1_wbt_3pt_hybrid,
    command=g1_29dof_wbt_hybrid_command_servo_off,
    reward=g1_29dof_wbt_2pt_hybrid_reward_servo_off,
    termination=g1_29dof_wbt_hybrid_termination_servo_off,
)

# ---------------------------------------------------------------------------
# 3-point hybrid + global root pos/vel + 6D orientation (exp:g1-wbt-3pt-hybrid-global-vel).
#
# = g1_wbt_3pt_hybrid with the ROOT-STATE observation signals folded in exactly as
# exp:g1-wbt-global-vel extends exp:g1-wbt: motion_ref_pos_b + motion_ref_vel_b +
# base_lin_vel added to the actor obs so the base can TRANSLATE through the world (plain
# 3pt-hybrid is root-invariant and cannot walk), PLUS the 3-point tracked-body orientation
# switched quaternion→6D (no double-cover OOD at deploy). 168D → 183D actor. Because
# root pos/vel is an OBSERVATION signal (NOT a reward change — verified: global/global-vel
# add it to the obs only; the absolute-world anchor reward is already in the inherited core
# reward), the SHIPPED servo-ON hybrid command/reward/curriculum/termination (Layer A + B,
# force servo ON, force-runaway estop) are reused VERBATIM — the ONLY delta is `observation`.
# yam URDF + ACTOR joint-vel dropout (default OFF) inherited from g1_wbt_3pt_hybrid.
# See docs/superpowers/specs/2026-07-20-g1-wbt-3pt-hybrid-global-vel-design.md.
# ---------------------------------------------------------------------------
from wbt_training.config_values.hybrid_3pt_global_vel import (  # noqa: E402
    g1_29dof_wbt_observation_3pt_hybrid_global_vel,
)

g1_wbt_3pt_hybrid_global_vel = replace(g1_wbt_3pt_hybrid, observation=g1_29dof_wbt_observation_3pt_hybrid_global_vel)

# Phase-1 (servo-OFF) two-phase twin, mirroring g1_wbt_3pt_hybrid_servo_off: swap only the
# three servo-off cfg carriers; obs stays a fixed 183D so the checkpoint transfers phase→phase.
g1_wbt_3pt_hybrid_global_vel_servo_off = replace(
    g1_wbt_3pt_hybrid_global_vel,
    command=g1_29dof_wbt_hybrid_command_servo_off,
    reward=g1_29dof_wbt_2pt_hybrid_reward_servo_off,
    termination=g1_29dof_wbt_hybrid_termination_servo_off,
)

# ---------------------------------------------------------------------------
# C1 unified single-policy presets (exp:g1-wbt-3pt-hybrid-global-vel-unified[-anneal]).
# Design: docs/superpowers/specs/2026-07-22-unified-hybrid-policy-design.md.
#
# R2 mainline (`-unified`): OBSERVE-ONLY integrator — Layer B keeps computing I/Δx
# (obs + Eq.2 reward target) but never injects Δq_arm; the policy is the sole executor.
# Force-alpha ramp ENABLED from 0 ("position-first" happens inside ONE run — the
# literature-backed alternative to a hard two-phase resume). 185D actor obs (183D +
# soft contact_phase label, hysteresis 3/1.5 N + 50 ms LPF).
#
# R1 anneal twin (`-unified-anneal`): full Jᵀ servo whose ACTUATION authority anneals
# 1→0 performance-gated (ServoAuthorityCurriculum; contact force-error EMA < 2 N per
# −0.02 step). Integrator/obs/reward keep full authority throughout — only the injected
# Δq_arm is scaled. Terminal α=0 config ≡ the R2 preset → the two runs A/B directly.
#
# Both forks change ONLY observation/command/reward/curriculum/termination via replace();
# the base g1_wbt_3pt_hybrid_global_vel (train + eval) is byte-identical to before.
# ---------------------------------------------------------------------------
from wbt_training.config_values.hybrid_unified import (  # noqa: E402
    g1_29dof_wbt_observation_3pt_unified_hybrid_global_vel,
    g1_29dof_wbt_unified_anneal_command,
    g1_29dof_wbt_unified_anneal_curriculum,
    g1_29dof_wbt_unified_anneal_reward,
    g1_29dof_wbt_unified_anneal_termination,
    g1_29dof_wbt_unified_command,
    g1_29dof_wbt_unified_curriculum,
    g1_29dof_wbt_unified_reward,
    g1_29dof_wbt_unified_termination,
)

g1_wbt_3pt_hybrid_global_vel_unified = replace(
    g1_wbt_3pt_hybrid_global_vel,
    observation=g1_29dof_wbt_observation_3pt_unified_hybrid_global_vel,
    command=g1_29dof_wbt_unified_command,
    reward=g1_29dof_wbt_unified_reward,
    curriculum=g1_29dof_wbt_unified_curriculum,
    termination=g1_29dof_wbt_unified_termination,
)
g1_wbt_3pt_hybrid_global_vel_unified_anneal = replace(
    g1_wbt_3pt_hybrid_global_vel_unified,
    command=g1_29dof_wbt_unified_anneal_command,
    reward=g1_29dof_wbt_unified_anneal_reward,
    curriculum=g1_29dof_wbt_unified_anneal_curriculum,
    termination=g1_29dof_wbt_unified_anneal_termination,
)

# ---------------------------------------------------------------------------
# LOCAL C1 unified twins (exp:g1-wbt-3pt-hybrid-unified[-anneal], user 2026-07-24).
#
# = the SAME R2/R1 unified recipe on the ORIGINAL pelvis-local 168D 3pt-hybrid base:
# no root pos/vel trio, quaternion vr_3point_orn (root-invariant, cannot walk — like
# the base exp:g1-wbt-3pt-hybrid). The unified command/reward/curriculum/termination
# carriers are tracking-point- AND obs-agnostic (they read the `_wbt_*` buffers, never
# the obs layout), so they are reused VERBATIM from the global-vel unified presets —
# the ONLY delta vs those is `observation` (170D = 168D + soft contact_phase), and the
# ONLY delta vs the base g1_wbt_3pt_hybrid is the unified stack. Both base presets
# (g1_wbt_3pt_hybrid + g1_wbt_3pt_hybrid_global_vel_unified) stay byte-identical.
# ---------------------------------------------------------------------------
from wbt_training.config_values.hybrid_unified import (  # noqa: E402
    g1_29dof_wbt_observation_3pt_unified_hybrid,
)

g1_wbt_3pt_hybrid_unified = replace(
    g1_wbt_3pt_hybrid,
    observation=g1_29dof_wbt_observation_3pt_unified_hybrid,
    command=g1_29dof_wbt_unified_command,
    reward=g1_29dof_wbt_unified_reward,
    curriculum=g1_29dof_wbt_unified_curriculum,
    termination=g1_29dof_wbt_unified_termination,
)
g1_wbt_3pt_hybrid_unified_anneal = replace(
    g1_wbt_3pt_hybrid_unified,
    command=g1_29dof_wbt_unified_anneal_command,
    reward=g1_29dof_wbt_unified_anneal_reward,
    curriculum=g1_29dof_wbt_unified_anneal_curriculum,
    termination=g1_29dof_wbt_unified_anneal_termination,
)

# ---------------------------------------------------------------------------
# TCP PRECISION (user 2026-08-05, plan docs/2026-08-05-tcp-precision-translation-orientation-plan.md).
#
# Applied IN PLACE to the two local unified twins above — the user asked for these presets
# to carry the fix, not for a `-tcp` fork. Purely additive on `reward`:
#   + tcp_position_error_exp    (w=4.0, σ=0.10 m) — prices the 0.242 m TCP POSITION, which
#     no existing term does: Eq.2's FK offset cancels algebraically in `cur − ref` (both
#     sides rotate it by the same live quat), and the 14-body position term excludes both
#     wrists. Its ref side rotates the offset by the REF attitude, so the orientation lever
#     (2·d·sin(θ/2): 4.8 cm at 0.2 rad) is priced as the position error it physically is.
#     Chases the SAME servo-shifted setpoint Eq.2 does (ref + Δx, 0 in free space), so the
#     two never disagree on where the hand belongs — no contact gate, so no reward hack.
#   + tcp_orientation_error_exp (w=2.0, σ=0.25 rad) — wrist attitude undiluted (mean over
#     2 hands not 14 bodies, σ 0.25 not 0.4 ⇒ ~36× the gradient of the status quo).
#
# ⚠️ OBSERVATION IS UNTOUCHED (still 170D, still the same ObservationManagerCfg object), so
# `--training.checkpoint` from pfl96opy warm-starts with NO network surgery: the actor,
# critic, and obs normalizer all keep their shapes. Reward changes affect only the gradient
# the checkpoint continues under.
#
# WBT_HAND_MASS_DR=1 additionally turns on phase 2 (gripper-mass DR + the m_distal sync);
# DEFAULT OFF so phase 1 measures the precision ceiling without DR confounding it. All of
# the phase logic lives in tcp_precision.unified_tcp_preset_overrides.
#
# The global-vel unified twins, the base 3pt-hybrid, and every other preset are unchanged.
# ---------------------------------------------------------------------------
from wbt_training.config_values.tcp_precision import (  # noqa: E402
    HAND_MASS_BODY_NAMES,
    HAND_MASS_DR_ENABLED,
    unified_tcp_preset_overrides,
)

g1_wbt_3pt_hybrid_unified = replace(g1_wbt_3pt_hybrid_unified, **unified_tcp_preset_overrides(anneal=False))
g1_wbt_3pt_hybrid_unified_anneal = replace(
    g1_wbt_3pt_hybrid_unified_anneal, **unified_tcp_preset_overrides(anneal=True)
)

if HAND_MASS_DR_ENABLED:
    # Phase 2 only: narrow mass DR to the two merged wrist bodies (94.1% gripper by mass).
    # `randomize_link_body_names` is REPLACED, not extended — the robot default is 9
    # pelvis/hip/knee links, which would scatter the DR across the legs and never touch the
    # hand. Nothing is lost today: `randomize_mass_startup` is that field's ONLY runtime
    # consumer (locomotion.py:741/789/825) and no preset here carried that term before phase
    # 2, so the leg list was dead config. But it IS a loaded gun — if anyone later enables
    # mass DR on these twins expecting whole-body coverage, they get wrists-only silently.
    # Hence: derive PER PRESET (never share one RobotConfig across both twins — that would
    # clobber an anneal-specific `robot=` the moment anyone adds one), and assert the
    # baseline we are overriding is the untouched robot default so a future change to it
    # cannot pass unnoticed. Review 2026-08-05.
    def _hand_mass_only_robot(exp):
        base = list(exp.robot.randomize_link_body_names)
        assert base == list(_g1_29dof_robot.randomize_link_body_names), (
            "expected the untouched robot default here; this preset already customises "
            f"randomize_link_body_names ({base}) and narrowing it to the wrists would discard that"
        )
        return replace(exp, robot=replace(exp.robot, randomize_link_body_names=list(HAND_MASS_BODY_NAMES)))

    g1_wbt_3pt_hybrid_unified = _hand_mass_only_robot(g1_wbt_3pt_hybrid_unified)
    g1_wbt_3pt_hybrid_unified_anneal = _hand_mass_only_robot(g1_wbt_3pt_hybrid_unified_anneal)

# ---------------------------------------------------------------------------
# 5-POINT and DENSE twins of the LOCAL R1 anneal preset (user 2026-08-27):
#   exp:g1-wbt-5pt-hybrid-unified-anneal  — 184D actor: the 170D 3pt layout with the two
#     FEET added as tracked points. The five points are maskedmimic's `five_point` hover
#     mode (maskedmimic_observations.DEFAULT_HOVER_MASK_MODES): torso + both hands + both
#     `*_ankle_roll_link`, i.e. exactly this repo's 3pt set plus the feet. Hands stay on the
#     0.242 m force TCP (offsets derived FROM `VR_3POINT_OFFSETS_TCP`, so a gripper respec
#     moves 3pt and 5pt together); head unchanged; feet at the ankle_roll_link origin.
#   exp:g1-wbt-hybrid-unified-anneal      — 183D actor: DENSE 29-DOF tracking, i.e.
#     `exp:g1-wbt`'s 157D obs (motion_command = all 29 joint pos+vel, no keypoints) plus the
#     same 26D force block.
#
# Both are a ONE-FIELD `replace(..., observation=...)` on g1_wbt_3pt_hybrid_unified_anneal,
# deliberately: the ENTIRE force stack — observe-only-vs-anneal injector cfg, the four
# unified reward terms, `ServoAuthorityCurriculum`, the force-alpha ramp, the force-runaway
# estop, the `WBT_HAND_MASS_DR` phase gating, AND the two `tcp_precision` TCP terms that
# price each hand's 0.242 m TCP in translation (w=4.0, σ=0.10 m) and attitude (w=2.0,
# σ=0.25 rad) — is inherited as the SAME OBJECTS. There is no copy to drift, and "force
# handling identical to exp:g1-wbt-3pt-hybrid-unified-anneal" is enforced structurally
# rather than by review. This works because the unified carriers read the `_wbt_*` buffers
# and the motion command, never the obs layout (the same property `hybrid_3pt.py` and
# `hybrid_3pt_global_vel.py` already rely on).
#
# Placed AFTER the HAND_MASS_DR block on purpose, so phase 2 (`WBT_HAND_MASS_DR=1`) is
# inherited too — including the wrists-only `randomize_link_body_names`. `_hand_mass_only_robot`
# is NOT re-applied (its assert expects the untouched robot default, which by here it isn't).
#
# NEW obs widths ⇒ NO existing checkpoint warm-starts into either (every unified checkpoint
# is 170D); these train from scratch. Deploy is not wired: `wbt_wrappers_inference` has no
# 184D/183D preset. Every other preset, obs term, and constant is untouched.
# ---------------------------------------------------------------------------
from wbt_training.config_values.hybrid_unified_5pt_dense import (  # noqa: E402
    g1_29dof_wbt_observation_5pt_unified_hybrid,
    g1_29dof_wbt_observation_dense_unified_hybrid,
)

g1_wbt_5pt_hybrid_unified_anneal = replace(
    g1_wbt_3pt_hybrid_unified_anneal, observation=g1_29dof_wbt_observation_5pt_unified_hybrid
)
g1_wbt_hybrid_unified_anneal = replace(
    g1_wbt_3pt_hybrid_unified_anneal, observation=g1_29dof_wbt_observation_dense_unified_hybrid
)

# ---------------------------------------------------------------------------
# 1-point tracking presets (right wrist only, 130D actor obs)
# ---------------------------------------------------------------------------
g1_wbt_1pt = replace(g1_wbt, observation=g1_29dof_wbt_observation_1pt)

g1_wbt_1pt_fast_sac = replace(g1_wbt_fast_sac, observation=g1_29dof_wbt_observation_1pt_fastsac)

# ---------------------------------------------------------------------------
# Without state estimation presets (TML-aligned)
# ---------------------------------------------------------------------------
# Removes motion_ref_pos_b and base_lin_vel from critic obs, making the
# policy deployable without a linear velocity state estimator.
# Matches TML G1FlatWoStateEstimationEnvCfg.
g1_wbt_wo_se = replace(g1_wbt, observation=g1_29dof_wbt_observation_wo_se)

g1_wbt_wo_se_fast_sac = replace(g1_wbt_fast_sac, observation=g1_29dof_wbt_observation_wo_se_fastsac)

# ---------------------------------------------------------------------------
# CHIP presets (3pt + compliance + selective 10-frame history)
# ---------------------------------------------------------------------------
# CHIP local: 984D actor obs (matches TML local_chip)
g1_wbt_chip_local = replace(g1_wbt, observation=g1_29dof_wbt_observation_chip_local)

# CHIP global: 987D actor obs (matches TML global_chip, adds motion_anchor_pos_b)
g1_wbt_chip_global = replace(g1_wbt, observation=g1_29dof_wbt_observation_chip_global)

# ---------------------------------------------------------------------------
# Global tracking presets (dense + root position/velocity)
# ---------------------------------------------------------------------------
# Global: 160D actor obs (157D TML + 3D root position in body frame)
g1_wbt_global = replace(g1_wbt, observation=g1_29dof_wbt_observation_global)

g1_wbt_global_fast_sac = replace(g1_wbt_fast_sac, observation=g1_29dof_wbt_observation_global_fastsac)

# Global + velocity: 166D actor obs (157D TML + 3D root pos + 3D ref root vel + 3D base lin vel)
g1_wbt_global_vel = replace(g1_wbt, observation=g1_29dof_wbt_observation_global_vel)

g1_wbt_global_vel_fast_sac = replace(g1_wbt_fast_sac, observation=g1_29dof_wbt_observation_global_vel_fastsac)

# ---------------------------------------------------------------------------
# MaskedMimic preset
# Uses its own PPO hyperparams (lr 2e-5, entropy, adaptive KL)
# with maskedmimic-specific observation, reward, termination, and command.
# ---------------------------------------------------------------------------
# Shared maskedmimic-style algo config (low LR, adaptive KL, entropy, 2M iters)
_mm_algo = replace(
    g1_wbt.algo,
    config=replace(
        g1_wbt.algo.config,
        init_noise_std=0.05,
        actor_learning_rate=2e-5,
        critic_learning_rate=1e-4,
        min_actor_learning_rate=1e-5,
        max_actor_learning_rate=2e-4,
        min_critic_learning_rate=1e-4,
        max_critic_learning_rate=1e-4,
        entropy_coef=0.013,
        max_grad_norm=0.1,
        schedule="adaptive",
        desired_kl=0.01,
        num_learning_iterations=30000,
        save_interval=500,
    ),
)

g1_wbt_maskedmimic = replace(
    g1_wbt,
    command=command.g1_29dof_wbt_maskedmimic_command,
    observation=g1_29dof_wbt_observation_maskedmimic,
    reward=g1_29dof_wbt_maskedmimic_reward,
    termination=g1_29dof_wbt_maskedmimic_termination,
    algo=_mm_algo,
)

# ---------------------------------------------------------------------------
# MaskedMimic + per-hand UniFP force-tracking preset.
#
# = g1_wbt_maskedmimic with the force machinery from exp:g1-wbt-2pt-force ported
# onto the maskedmimic recipe:
#   - command : maskedmimic multi-motion command + HandForceInjector (F^ext pulse).
#   - obs     : maskedmimic obs (hover modes restricted to {2pt, 3pt, 5pt, 29dof
#               -dense}) + F^cmd on the actor; + F^cmd + F^ext on the critic.
#   - reward  : maskedmimic global rewards with the body-pos term swapped for the
#               no-wrist GLOBAL variant + the dedicated wrist-pos reward dropped
#               (Eq.2 owns the wrist), plus force_track_pos_eq2_global (w=6.0) +
#               feet_slip (w=-2.0).
#   - curriculum : force_alpha ramp (disabled by default, full force from step 0).
# termination is INHERITED from g1_wbt_maskedmimic (unchanged). Honour the
# --F_on_hand_frame flag via the shared train_agent path (keys off force_cmd_local).
# ---------------------------------------------------------------------------
from wbt_training.config_values.maskedmimic_force import (  # noqa: E402
    g1_29dof_wbt_maskedmimic_force_command,
    g1_29dof_wbt_maskedmimic_force_curriculum,
    g1_29dof_wbt_maskedmimic_force_reward,
    g1_29dof_wbt_observation_maskedmimic_force,
)

g1_wbt_maskedmimic_force = replace(
    g1_wbt_maskedmimic,
    command=g1_29dof_wbt_maskedmimic_force_command,
    observation=g1_29dof_wbt_observation_maskedmimic_force,
    reward=g1_29dof_wbt_maskedmimic_force_reward,
    curriculum=g1_29dof_wbt_maskedmimic_force_curriculum,
)

# ---------------------------------------------------------------------------
# FALCON-style DUAL-AGENT UniFP force-tracking preset (exp:g1-wbt-2pt-force-falcon).
#
# = g1_wbt_maskedmimic_force (multi-motion loader + global-frame Eq.2 + force machinery)
# with the FALCON upper/lower decoupling bolted on ADDITIVELY:
#   - algo      : PPODecoupled — TWO actors + TWO critics (lower 15 = 12 leg + 3 waist,
#                 upper 14 = arms) over ONE shared obs, per-agent GAE/LR/optimizers.
#   - env_class : FalconDecoupledManager — swaps in DualGroupRewardManager (per-group
#                 lower/upper reward streams), exposes the action-slice maps + stance_mask.
#   - command   : maskedmimic multi-motion + HandForceInjector (both unchanged) + a
#                 FalconVelocityCommand (walk/stand) driving the lower agent.
#   - obs       : maskedmimic-force obs with the actor F^cmd flipped to the ft_sensor
#                 frame (req 1) + a velocity-command term for the lower agent.
#   - reward    : maskedmimic-force terms TAGGED lower/upper + lean velocity-command
#                 locomotion (lower) + TCP precision (upper). Eq.2 / all UniFP force
#                 math UNCHANGED (req 7); F^ext still on wrist_yaw (req 6).
# curriculum inherited from g1_wbt_maskedmimic_force (force_alpha, disabled by default).
# ---------------------------------------------------------------------------
from wbt_training.config_values.falcon_force_preset import (  # noqa: E402
    g1_29dof_wbt_falcon_force_command,
    g1_29dof_wbt_falcon_force_reward,
    g1_29dof_wbt_observation_falcon_force,
)
from wbt_training.config_values.ppo_decoupled_algo import g1_29dof_wbt_falcon_force_algo  # noqa: E402

g1_wbt_2pt_force_falcon = replace(
    g1_wbt_maskedmimic_force,
    env_class="wbt_training.envs.falcon_decoupled_manager.FalconDecoupledManager",
    algo=g1_29dof_wbt_falcon_force_algo,
    command=g1_29dof_wbt_falcon_force_command,
    observation=g1_29dof_wbt_observation_falcon_force,
    reward=g1_29dof_wbt_falcon_force_reward,
)

# ---------------------------------------------------------------------------
# MaskedMimic + HYBRID position-force controller (exp:g1-maskmimic-hybrid).
#
# = g1_wbt_maskedmimic (multi-motion loader, 4 conditioning modes {2pt,3pt,5pt,29dof-dense},
# absolute-world-frame tracking, low-LR adaptive-KL PPO) with the exp:g1-wbt-2pt-hybrid HYBRID
# controller ported on: causal K_env contact spring + simulated wrist F/T sensor (F_meas in the
# actor) + Jᵀ contact-gated force servo (shipped ON) + force-runaway estop. The whole hybrid
# physics stack (HybridContactInjector, servo, sensor, estop) is frame-agnostic and reused
# verbatim; only the position-anchored Eq.2 reward gets a GLOBAL-frame twin (anchored on
# mc.body_pos_w to match maskmimic's global tracking). Also adds an ACTOR joint-velocity dropout
# (WBT_JV_DROPOUT_* env knobs) so a corrupted real F/T reading can't destabilise a velocity-reliant
# policy (the actor learns to lean off its own dof_vel, zeroed at deploy). yam URDF baked in.
# See docs/superpowers/plans/2026-07-15-g1-maskmimic-hybrid.md.
# ---------------------------------------------------------------------------
from wbt_training.config_values.maskedmimic_hybrid import (  # noqa: E402
    g1_29dof_wbt_maskedmimic_hybrid_command,
    g1_29dof_wbt_maskedmimic_hybrid_command_servo_off,
    g1_29dof_wbt_maskedmimic_hybrid_curriculum,
    g1_29dof_wbt_maskedmimic_hybrid_reward,
    g1_29dof_wbt_maskedmimic_hybrid_reward_servo_off,
    g1_29dof_wbt_maskedmimic_hybrid_termination,
    g1_29dof_wbt_maskedmimic_hybrid_termination_servo_off,
    g1_29dof_wbt_observation_maskedmimic_hybrid,
)

g1_maskmimic_hybrid = replace(
    g1_wbt_maskedmimic,  # inherit the mm algo (_mm_algo: lr 2e-5, adaptive KL, entropy, 30k iters)
    robot=replace(
        g1_wbt_maskedmimic.robot, asset=replace(g1_wbt_maskedmimic.robot.asset, urdf_file="g1/g1_29dof_yam.urdf")
    ),
    command=g1_29dof_wbt_maskedmimic_hybrid_command,
    observation=g1_29dof_wbt_observation_maskedmimic_hybrid,
    reward=g1_29dof_wbt_maskedmimic_hybrid_reward,
    curriculum=g1_29dof_wbt_maskedmimic_hybrid_curriculum,
    termination=g1_29dof_wbt_maskedmimic_hybrid_termination,  # + force_runaway estop (servo shipped ON)
)

# Phase-1 (servo-OFF) variant for the two-phase curriculum: train the stand + position tracking to
# convergence here, then resume the SAME checkpoint under g1_maskmimic_hybrid (servo ON) via
# `--training.checkpoint <phase1.pt>`. Obs width is identical in BOTH (servo I/Δx buffers present +
# zeroed when off) so the network transfers with no surgery; only servo_enabled flips.
g1_maskmimic_hybrid_servo_off = replace(
    g1_maskmimic_hybrid,
    command=g1_29dof_wbt_maskedmimic_hybrid_command_servo_off,
    reward=g1_29dof_wbt_maskedmimic_hybrid_reward_servo_off,
    termination=g1_29dof_wbt_maskedmimic_hybrid_termination_servo_off,
)

# ---------------------------------------------------------------------------
# Dense/3pt presets with maskedmimic training recipe (low LR, adaptive KL)
# Same observation/reward/termination as the standard presets, only algo differs.
# ---------------------------------------------------------------------------
g1_wbt_global_mm = replace(g1_wbt_global, algo=_mm_algo)
g1_wbt_3pt_mm = replace(g1_wbt_3pt, algo=_mm_algo)

# ---------------------------------------------------------------------------
# MaskedMimic with FastSAC
# Same maskedmimic observation/reward/termination/command but FastSAC algorithm.
# Maskedmimic already uses history_length=1 so replay buffer is compatible.
# ---------------------------------------------------------------------------
g1_wbt_maskedmimic_fast_sac = replace(
    g1_wbt_fast_sac,
    command=command.g1_29dof_wbt_maskedmimic_command,
    observation=g1_29dof_wbt_observation_maskedmimic,
    reward=g1_29dof_wbt_maskedmimic_reward,
    termination=g1_29dof_wbt_maskedmimic_termination,
)

_ee_mass_body_names = list(_g1_29dof_robot.randomize_link_body_names) + [
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

# ---------------------------------------------------------------------------
# End-effector mass DR: moderate mass randomization on all links + EE
# ---------------------------------------------------------------------------
g1_wbt_maskedmimic_ee_mass = replace(
    g1_wbt_maskedmimic,
    randomization=g1_29dof_wbt_randomization_ee_mass,
    robot=replace(g1_wbt_maskedmimic.robot, randomize_link_body_names=_ee_mass_body_names),
)

# ---------------------------------------------------------------------------
# Aggressive all-DR: mass + friction + COM + PD gains + action delay + pushes
# ---------------------------------------------------------------------------
g1_wbt_maskedmimic_aggressive_dr = replace(
    g1_wbt_maskedmimic,
    randomization=g1_29dof_wbt_randomization_aggressive,
    robot=replace(g1_wbt_maskedmimic.robot, randomize_link_body_names=_ee_mass_body_names),
)

# ---------------------------------------------------------------------------
# Terrain presets. Start from the stock `g1_29dof_wbt` base (ref-matching);
# terrain-specific command/observation/reward/termination presets live in
# `wbt_training.config_values.terrain`.
# ---------------------------------------------------------------------------
_ref_adamw_no_wd = OptimizerConfig(_target_="torch.optim.AdamW", weight_decay=0.0)

import json as _json  # noqa: E402

g1_wbt_terrain_ref = replace(
    g1_29dof_wbt,  # ref base, NOT our local TML-tuned g1_wbt
    training=replace(
        g1_29dof_wbt.training,
        project="WBT-Holosoma",
        name="g1_29dof_wbt_terrain_manager",
        num_envs=4096,
        # Application-specific preprocess: resolves training.registry_name via
        # wandb (or file://), samples per-env on-path obstacles, splices the
        # combined terrain OBJ + motion NPZ into the config before env build.
        preprocess_hook="wbt_training.config_values.terrain:apply_terrain_preprocess",
        preprocess_hook_kwargs=_json.dumps(
            {
                "add_onpath_obstacle": True,
                # num_variants=64 matches ref dck4z60w / mdlzasjz (obstacle
                # variants << num_envs, so many envs share each obstacle layout).
                "num_variants": 64,
            }
        ),
    ),
    logger=wandb_logger,
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(simulator.isaacsim.config.sim, max_episode_length_s=10.0),
            scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0),
        ),
    ),
    # g1_29dof_wbt base already has actor/critic hidden_dims=[512,256,128] — no module override needed.
    algo=replace(
        g1_29dof_wbt.algo,
        config=replace(
            g1_29dof_wbt.algo.config,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            num_learning_epochs=5,
            num_learning_iterations=20000,
            save_interval=500,
            entropy_coef=0.005,
            init_noise_std=1.0,
            init_at_random_ep_len=False,
            use_symmetry=False,
            empirical_normalization=False,
            actor_optimizer=_ref_adamw_no_wd,
            critic_optimizer=_ref_adamw_no_wd,
        ),
    ),
    robot=replace(
        g1_29dof_wbt.robot,
        control=replace(
            g1_29dof_wbt.robot.control,
            # Ref values verified via wandb config of dck4z60w.
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
            control_mode="implicit_position_target",
            implicit_actuator_builder="wbt_training.config_values.robot_config:build_g1_implicit_actuators",
        ),
        asset=replace(g1_29dof_wbt.robot.asset, urdf_file="g1/main_mesh_arm_collision_halfspherehand.urdf"),
    ),
    # Ref routes position targets to IsaacSim's built-in PD drive via
    # JointPositionTargetActionTerm (not the explicit-PD JointPositionActionTerm
    # inherited from g1_29dof_wbt).
    action=g1_29dof_joint_pos_implicit,
    observation=g1_29dof_wbt_terrain_observation,
    termination=g1_29dof_wbt_terrain_termination,
    randomization=g1_29dof_wbt_terrain_randomization,
    command=g1_29dof_wbt_terrain_command,
    reward=g1_29dof_wbt_terrain_reward,
)

# Terrain preset without the height_scan oracle: actor/critic drop those 289
# ray readings. Same obstacle injection + motion set as g1_wbt_terrain_ref;
# bumped to 60k iterations since training has to solve terrain traversal
# without the heightmap sensor.
g1_wbt_terrain_no_heightscan = replace(
    g1_wbt_terrain_ref,
    observation=g1_29dof_wbt_terrain_observation_no_heightscan,
    algo=replace(
        g1_wbt_terrain_ref.algo, config=replace(g1_wbt_terrain_ref.algo.config, num_learning_iterations=60000)
    ),
)

# Ref-intent variant: train with effective unit scaling (action_scale=1.0,
# action_scales_by_effort_limit_over_p_gain=False). The original FAR-Holosoma
# terrain preset set these same values on the *action term* (not robot.control);
# our JointPositionTargetActionTerm now applies control.action_scale, so to
# match the ref math we set the control-level knobs to unit here.
g1_wbt_terrain_ref_noscale = replace(
    g1_wbt_terrain_ref,
    robot=replace(
        g1_wbt_terrain_ref.robot,
        control=replace(
            g1_wbt_terrain_ref.robot.control, action_scale=1.0, action_scales_by_effort_limit_over_p_gain=False
        ),
    ),
)

g1_wbt_terrain_no_heightscan_noscale = replace(
    g1_wbt_terrain_no_heightscan,
    robot=replace(
        g1_wbt_terrain_no_heightscan.robot,
        control=replace(
            g1_wbt_terrain_no_heightscan.robot.control,
            action_scale=1.0,
            action_scales_by_effort_limit_over_p_gain=False,
        ),
    ),
)

g1_wbt_terrain_ref_pelvis = replace(g1_wbt_terrain_ref, command=g1_29dof_wbt_terrain_command_pelvis)

g1_wbt_terrain_no_heightscan_pelvis = replace(g1_wbt_terrain_no_heightscan, command=g1_29dof_wbt_terrain_command_pelvis)

DEFAULTS = {
    **CORE_DEFAULTS,
    # Dense tracking (all joints + 6pt keypoints, 157D actor obs, PPO)
    "g1_wbt": g1_wbt,
    # Dense tracking with FastSAC algorithm
    "g1_wbt_fast_sac": g1_wbt_fast_sac,
    # 3-point tracking: pelvis + left/right hand keypoints (144D actor obs, PPO)
    "g1_wbt_3pt": g1_wbt_3pt,
    # 3-point tracking with FastSAC algorithm
    "g1_wbt_3pt_fast_sac": g1_wbt_3pt_fast_sac,
    # 3-point tracking + root position in body frame (147D actor obs)
    "g1_wbt_3pt_global": g1_wbt_3pt_global,
    # 3-point tracking + root position + root velocity (153D actor obs)
    "g1_wbt_3pt_global_vel": g1_wbt_3pt_global_vel,
    # 2-point tracking: left/right wrist only (141D actor obs, PPO)
    "g1_wbt_2pt": g1_wbt_2pt,
    # 2-point tracking + root position (144D actor obs, PPO)
    "g1_wbt_2pt_global": g1_wbt_2pt_global,
    # 2-point tracking with FastSAC algorithm
    "g1_wbt_2pt_fast_sac": g1_wbt_2pt_fast_sac,
    # 2-point tracking + per-hand UniFP-style force command (147D actor obs)
    "g1_wbt_2pt_force": g1_wbt_2pt_force,
    # 2pt-force with F^ext ADDED BACK to the actor (153D) + first-commit reward
    # weights (full 14-body tracking + Eq.2@4.0 + pen@-6.0); yam URDF baked in.
    "g1_wbt_2pt_force_ext": g1_wbt_2pt_force_ext,
    # Single-actor upper-body precision: stands by clip-leg-tracking + relative-frame
    # TCP fingertip precision on top of 2pt + Eq.2 force (yam URDF baked in).
    "g1_wbt_2pt_force_upperbody": g1_wbt_2pt_force_upperbody,
    # Hybrid position-force controller v1: causal K_env contact + sim F/T sensor
    # (F_meas in actor) + torso-stable reward; Layer B Jᵀ servo behind servo_enabled.
    "g1_wbt_2pt_hybrid": g1_wbt_2pt_hybrid,
    # Phase-1 servo-OFF variant (train the stand first, then resume checkpoint under servo-ON).
    "g1_wbt_2pt_hybrid_servo_off": g1_wbt_2pt_hybrid_servo_off,
    # 3-point hybrid position-force controller: SAME hybrid stack as 2pt-hybrid on the 3pt
    # tracking base (168D actor obs, servo shipped ON) + ACTOR joint-vel dropout (default OFF).
    "g1_wbt_3pt_hybrid": g1_wbt_3pt_hybrid,
    # Phase-1 servo-OFF variant of the 3pt-hybrid preset (two-phase curriculum, fixed 168D obs).
    "g1_wbt_3pt_hybrid_servo_off": g1_wbt_3pt_hybrid_servo_off,
    # 3pt-hybrid + global root pos/vel + 6D orientation (183D actor obs): the 3pt-hybrid
    # Layer A + B controller (servo shipped ON) with motion_ref_pos_b/vel_b + base_lin_vel
    # added to the obs (so the base can walk) and vr_3point_orn switched quat→6D.
    "g1_wbt_3pt_hybrid_global_vel": g1_wbt_3pt_hybrid_global_vel,
    # Phase-1 servo-OFF variant of the 3pt-hybrid-global-vel preset (fixed 183D obs, two-phase).
    "g1_wbt_3pt_hybrid_global_vel_servo_off": g1_wbt_3pt_hybrid_global_vel_servo_off,
    # C1 unified single-policy R2 mainline: observe-only integrator (reference generator,
    # policy is the executor) + force-alpha ramp from 0 + soft contact_phase label (185D).
    "g1_wbt_3pt_hybrid_global_vel_unified": g1_wbt_3pt_hybrid_global_vel_unified,
    # R1 anneal twin: full servo, actuation authority annealed 1→0 performance-gated;
    # terminal config ≡ the unified preset → direct A/B.
    "g1_wbt_3pt_hybrid_global_vel_unified_anneal": g1_wbt_3pt_hybrid_global_vel_unified_anneal,
    # LOCAL C1 unified R2: the unified recipe (observe-only integrator + contact_phase +
    # force-alpha ramp) on the pelvis-local 168D 3pt-hybrid base → 170D actor obs.
    "g1_wbt_3pt_hybrid_unified": g1_wbt_3pt_hybrid_unified,
    # LOCAL R1 anneal twin: full servo annealed 1→0 on the same 170D local base.
    "g1_wbt_3pt_hybrid_unified_anneal": g1_wbt_3pt_hybrid_unified_anneal,
    # 5-POINT twin of the LOCAL R1 anneal preset (184D): the 170D layout with the two FEET
    # (*_ankle_roll_link, from maskedmimic's `five_point` hover mode) added as tracked
    # points. Hands still on the 0.242 m TCP; force stack + TCP precision reward inherited
    # as the SAME objects — `observation` is the only delta.
    "g1_wbt_5pt_hybrid_unified_anneal": g1_wbt_5pt_hybrid_unified_anneal,
    # DENSE twin of the LOCAL R1 anneal preset (183D): exp:g1-wbt's 29-DOF joint-target obs
    # (no keypoints) + the same 26D force block; same one-field delta.
    "g1_wbt_hybrid_unified_anneal": g1_wbt_hybrid_unified_anneal,
    # 1-point tracking: right wrist only (130D actor obs, PPO)
    "g1_wbt_1pt": g1_wbt_1pt,
    # 1-point tracking with FastSAC algorithm
    "g1_wbt_1pt_fast_sac": g1_wbt_1pt_fast_sac,
    # Dense tracking without state estimation (no base_lin_vel in critic)
    "g1_wbt_wo_se": g1_wbt_wo_se,
    # Dense tracking without state estimation, FastSAC
    "g1_wbt_wo_se_fast_sac": g1_wbt_wo_se_fast_sac,
    # CHIP local: 3pt + compliance + selective 10-frame history (984D actor obs)
    "g1_wbt_chip_local": g1_wbt_chip_local,
    # CHIP global: 3pt + compliance + history + root anchor (987D actor obs)
    "g1_wbt_chip_global": g1_wbt_chip_global,
    # Dense tracking + root position in body frame (160D actor obs)
    "g1_wbt_global": g1_wbt_global,
    # Dense tracking + root position, FastSAC
    "g1_wbt_global_fast_sac": g1_wbt_global_fast_sac,
    # Dense tracking + root position + root velocity (166D actor obs)
    "g1_wbt_global_vel": g1_wbt_global_vel,
    # Dense tracking + root position + root velocity, FastSAC
    "g1_wbt_global_vel_fast_sac": g1_wbt_global_vel_fast_sac,
    # MaskedMimic: hover-masked keypoints + joints (low LR, adaptive KL, entropy)
    "g1_wbt_maskedmimic": g1_wbt_maskedmimic,
    # MaskedMimic + per-hand UniFP force tracking (hover modes restricted to
    # {2pt, 3pt, 5pt, 29dof-dense}; actor F^cmd, critic F^cmd+F^ext; global Eq.2)
    "g1_wbt_maskedmimic_force": g1_wbt_maskedmimic_force,
    # FALCON-style dual-agent (2 actor + 2 critic; lower velocity-command locomotion /
    # upper 2pt + sensor-frame F^cmd + TCP manipulation) on the maskedmimic-force base.
    "g1_wbt_2pt_force_falcon": g1_wbt_2pt_force_falcon,
    # MaskedMimic + HYBRID position-force controller: causal K_env contact + sim F/T sensor
    # (F_meas in actor) + Jᵀ force servo (shipped ON) + actor joint-vel dropout; global-frame Eq.2.
    "g1_maskmimic_hybrid": g1_maskmimic_hybrid,
    # Phase-1 servo-OFF variant (train the stand + position tracking first, then resume servo-ON).
    "g1_maskmimic_hybrid_servo_off": g1_maskmimic_hybrid_servo_off,
    # MaskedMimic with FastSAC algorithm
    "g1_wbt_maskedmimic_fast_sac": g1_wbt_maskedmimic_fast_sac,
    # Dense global tracking with maskedmimic training recipe (low LR, adaptive KL)
    "g1_wbt_global_mm": g1_wbt_global_mm,
    # 3-point tracking with maskedmimic training recipe (low LR, adaptive KL)
    "g1_wbt_3pt_mm": g1_wbt_3pt_mm,
    # MaskedMimic + end-effector mass randomization (link scaling 0.8-1.5, base ±1-3 kg)
    "g1_wbt_maskedmimic_ee_mass": g1_wbt_maskedmimic_ee_mass,
    # MaskedMimic + aggressive all-DR (mass + friction + COM + PD gains + delay + pushes)
    "g1_wbt_maskedmimic_aggressive_dr": g1_wbt_maskedmimic_aggressive_dr,
    # Terrain preset for dense WBT with on-path obstacle injection
    "g1_wbt_terrain_ref": g1_wbt_terrain_ref,
    # Terrain preset without the oracle height_scan; 60k iters
    "g1_wbt_terrain_no_heightscan": g1_wbt_terrain_no_heightscan,
    # Same terrain preset but with effective unit action scaling (ref-intent).
    "g1_wbt_terrain_ref_noscale": g1_wbt_terrain_ref_noscale,
    "g1_wbt_terrain_no_heightscan_noscale": g1_wbt_terrain_no_heightscan_noscale,
    # Pelvis-anchored terrain variants (native holosoma convention).
    "g1_wbt_terrain_ref_pelvis": g1_wbt_terrain_ref_pelvis,
    "g1_wbt_terrain_no_heightscan_pelvis": g1_wbt_terrain_no_heightscan_pelvis,
}

AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{name.replace('_', '-')}": cfg for name, cfg in DEFAULTS.items()}
        )
    ),
]
