"""Inference configurations for TML and BM WBT wrapper policies."""

from __future__ import annotations

from dataclasses import replace

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values import robot

from wbt_wrappers_inference.config_values import observation, task

# fmt: off

# Shared robot config override for TML/BM policies (custom KP/KD and stiff startup).
_g1_29dof_wbt_wrapper_robot = replace(
    robot.g1_29dof,
    motor_kp=(
        40.1792, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012,  # left leg
        40.1792, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012,  # right leg
        40.1792, 28.5012, 28.5012,                              # waist
        14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783,  # left arm
        14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783,  # right arm
    ),
    motor_kd=(
        2.5579, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144,  # left leg
        2.5579, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144,  # right leg
        2.5579, 1.8144, 1.8144,                            # waist
        0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,  # left arm
        0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,  # right arm
    ),
    stiff_startup_pos=(
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
        0.0, 0.0, 0.0,                            # waist
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,       # left arm
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # right arm
    ),
    stiff_startup_kp=(
        350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
        350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
        200.0, 200.0, 200.0,
        40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
    ),
    stiff_startup_kd=(
        5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
        5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
        5.0, 5.0, 5.0,
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
    ),
)
# fmt: on

g1_29dof_tml_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.tml_wbt, task=task.tml_wbt
)

g1_29dof_bm_wbt = InferenceConfig(robot=_g1_29dof_wbt_wrapper_robot, observation=observation.bm_wbt, task=task.bm_wbt)

# TEMPORARY preset for legacy TML-trained CHIP checkpoints.
g1_29dof_tml_chip_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.tml_chip, task=task.tml_chip
)

g1_29dof_holosoma_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_wbt, task=task.holosoma_wbt
)

g1_29dof_holosoma_3pt_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_3pt_wbt, task=task.holosoma_3pt_wbt
)

g1_29dof_holosoma_2pt_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_2pt_wbt, task=task.holosoma_3pt_wbt
)

g1_29dof_holosoma_2pt_force_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_2pt_force_wbt, task=task.holosoma_3pt_wbt
)

g1_29dof_holosoma_2pt_force_ext_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_2pt_force_ext_wbt, task=task.holosoma_3pt_wbt
)

# Hybrid position-force controller (exp:g1-wbt-2pt-hybrid). Servo-ON (31wei7by)
# and Layer-A/servo-OFF (6y8314vq) share the SAME 165D obs — one InferenceConfig
# serves both; the browser preset name / experiment_config.servo_enabled decides
# whether the Jᵀ servo runs. Headless eval fills the force terms with zeros.
g1_29dof_holosoma_2pt_hybrid_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_2pt_hybrid_wbt, task=task.holosoma_3pt_wbt
)

# 3pt hybrid position-force controller (exp:g1-wbt-3pt-hybrid). Servo-ON and
# Layer-A/servo-OFF share the SAME 168D obs — one InferenceConfig serves both; the
# browser preset name / experiment_config.servo_enabled decides whether the Jᵀ servo
# runs. task = holosoma_3pt_wbt (the 3-point tracking task, like every 3pt preset).
# Headless eval fills the force terms with zeros → pure 3pt tracking on a hybrid ckpt.
g1_29dof_holosoma_3pt_hybrid_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_3pt_hybrid_wbt, task=task.holosoma_3pt_wbt
)

# 3pt hybrid + global root pos/vel + 6D orn (exp:g1-wbt-3pt-hybrid-global-vel,
# canonical run bqlzkwok). Servo-ON and Layer-A/servo-OFF share the SAME 183D obs —
# one InferenceConfig serves both; HOLOSOMA_HYBRID_SERVO (deploy) decides whether
# the Jᵀ servo runs. task = holosoma_3pt_wbt like every 3pt preset. The root-state
# trio (motion_ref_pos_b/motion_ref_vel_b/base_lin_vel) makes detect_obs_mode
# return "3pt_global" → first-tick motion alignment; base_lin_vel needs
# --task.fk-odometry true (or mocap) on hardware. Headless eval fills the force
# terms with zeros → pure walk-enabled 3pt tracking on a hybrid ckpt.
g1_29dof_holosoma_3pt_hybrid_global_vel_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot,
    observation=observation.holosoma_3pt_hybrid_global_vel_wbt,
    task=task.holosoma_3pt_wbt,
)

# C1 unified single-policy preset (exp:g1-wbt-3pt-hybrid-global-vel-unified[-anneal]).
# 185D = the hybrid-global-vel obs + the soft contact_phase label (2D). One config
# serves both training twins (R2 observe-only / R1 anneal) — identical obs, and at
# deploy neither injects a servo (observe-only by construction / α annealed to 0).
g1_29dof_holosoma_3pt_hybrid_global_vel_unified_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot,
    observation=observation.holosoma_3pt_hybrid_global_vel_unified_wbt,
    task=task.holosoma_3pt_wbt,
)

# LOCAL C1 unified single-policy (exp:g1-wbt-3pt-hybrid-unified, canonical run
# m3s0pw9i) and its R1 anneal twin (exp:g1-wbt-3pt-hybrid-unified-anneal, current run
# iss0qs71). 170D = the pelvis-local 168D 3pt-hybrid obs + the soft contact_phase
# label (2D) — no root pos/vel trio, quaternion vr_3point_orn, so detect_obs_mode
# returns plain "3pt" (root-invariant; does NOT walk). ONE InferenceConfig serves
# BOTH twins: the obs is identical and the R2/R1 split is a training-time
# curriculum (observe-only from the start vs servo authority annealed 1→0).
#
# At deploy NEITHER twin injects a Jᵀ servo: holosoma_wbt.py keys "observe-only"
# off the PRESENCE of the contact_phase term, so it holds even for an R1
# checkpoint (whose ONNX bakes servo_observe_only=False and whose annealed
# terminal α is never serialized). NOTE the mjswan browser runtime in this tree reads
# `servo_observe_only` straight from the checkpoint instead, so it does NOT agree with
# this decision for an R1 ckpt — do not use the browser as an R1 pre-flight. The integrator still winds, so integrator_state (I) /
# x_tgt_shift (Δx) stay live actor inputs; only Δq_arm is suppressed.
# contact_phase is computed LIVE from the Bota ‖F_meas‖ (hysteresis 2/1.5 N +
# 50 ms LPF); with no live F/T source all force/servo/phase terms read zeros →
# pure 3pt position tracking on a unified checkpoint.
g1_29dof_holosoma_3pt_hybrid_unified_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot,
    observation=observation.holosoma_3pt_hybrid_unified_wbt,
    task=task.holosoma_3pt_wbt,
)

# 5-POINT unified anneal (exp:g1-wbt-5pt-hybrid-unified-anneal, W&B run dlcxc59d).
# 184D x 4 = 736D = the 170D 3pt unified obs with the tracking front-end widened from 3 to
# 5 points (both feet appended). Everything else — the whole force block, the scales, the
# history — is the 3pt config's, so this deploys through exactly the same path:
# `detect_obs_mode` returns "5pt", `_vr_offsets()` returns the TCP triple + two zero foot
# offsets, and `_compute_vr_fk` loops over 5 frame IDs instead of 3.
#
# TELEOP NOTE: the rig has THREE trackers, so the two foot targets cannot come from
# hardware. They come from a Motion_Completion pass in `vive_teleop_bridge --mode 5pt`,
# read out of its 29-DoF FK at the SAME 0.242 m TCP the hands use. See
# `wbt_realtime_eval/vive/mc_teleop_source.py`. Without a live bridge this preset simply
# replays the NPZ clip's 5-point FK, like every other tracking preset.
g1_29dof_holosoma_5pt_hybrid_unified_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot,
    observation=observation.holosoma_5pt_hybrid_unified_wbt,
    task=task.holosoma_3pt_wbt,
)

# DENSE unified anneal (exp:g1-wbt-hybrid-unified-anneal, W&B run 7r54008a).
# 183D x 4 = 732D. NO keypoints at all: the tracking front-end is the 58D `command` term
# (29 absolute joint targets ++ 29 velocities, which already subsumes command_lower_body),
# plus the same 26D force block the keypoint unified configs carry.
#
# ⚠️ 732D COLLIDES with g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt (also 732D).
# The layouts are disjoint — the obs-dim check alone CANNOT tell them apart. What does:
# `_validate_obs_term_order` compares the checkpoint's own actor-obs term names against
# this config, so a swap is refused by NAME.
#
# TELEOP: `vive_teleop_bridge --mode dense` streams the completion's 29 joint targets and
# `holosoma_wbt._live_dense_command` remaps them onto dof_names BY NAME. Without a live
# bridge this replays the NPZ clip's motion_command, i.e. the legacy dense behaviour.
g1_29dof_holosoma_dense_hybrid_unified_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot,
    observation=observation.holosoma_dense_hybrid_unified_wbt,
    task=task.holosoma_wbt,
)

g1_29dof_holosoma_2pt_global_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_2pt_global_wbt, task=task.holosoma_wbt
)

g1_29dof_holosoma_3pt_global_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_3pt_global_wbt, task=task.holosoma_3pt_wbt
)

g1_29dof_holosoma_global_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_global_wbt, task=task.holosoma_wbt
)

g1_29dof_holosoma_global_vel_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_global_vel_wbt, task=task.holosoma_wbt
)

# Terrain presets: pair a `*_motion.npz` with its `*_terrain.npy` via
# `--task.headless-terrain` for obstacle-aware sim2sim.
g1_29dof_holosoma_terrain_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_terrain_wbt, task=task.holosoma_wbt
)

g1_29dof_holosoma_terrain_no_heightscan_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot,
    observation=observation.holosoma_terrain_wbt_no_heightscan,
    task=task.holosoma_wbt,
)

g1_29dof_holosoma_chip_local_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot,
    observation=observation.holosoma_chip_local_wbt,
    task=task.holosoma_chip_local_wbt,
)

g1_29dof_holosoma_chip_global_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot,
    observation=observation.holosoma_chip_global_wbt,
    task=task.holosoma_chip_global_wbt,
)

g1_29dof_holosoma_maskedmimic_3pt_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_maskedmimic, task=task.holosoma_maskedmimic_3pt
)

g1_29dof_holosoma_maskedmimic_3pt_head_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot,
    observation=observation.holosoma_maskedmimic,
    task=task.holosoma_maskedmimic_3pt_head,
)

g1_29dof_holosoma_maskedmimic_3pt_real_head_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot,
    observation=observation.holosoma_maskedmimic,
    task=task.holosoma_maskedmimic_3pt_real_head,
)

g1_29dof_holosoma_maskedmimic_pelvis_wrists_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot,
    observation=observation.holosoma_maskedmimic,
    task=task.holosoma_maskedmimic_pelvis_wrists,
)

g1_29dof_holosoma_maskedmimic_4pt_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_maskedmimic, task=task.holosoma_maskedmimic_4pt
)

g1_29dof_holosoma_maskedmimic_5pt_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_maskedmimic, task=task.holosoma_maskedmimic_5pt
)

g1_29dof_holosoma_maskedmimic_full_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_maskedmimic, task=task.holosoma_maskedmimic_full
)

g1_29dof_holosoma_maskedmimic_2pt_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_wrapper_robot, observation=observation.holosoma_maskedmimic, task=task.holosoma_maskedmimic_2pt
)

DEFAULTS = {
    "g1-29dof-tml-wbt": g1_29dof_tml_wbt,
    "g1-29dof-bm-wbt": g1_29dof_bm_wbt,
    "g1-29dof-tml-chip-wbt": g1_29dof_tml_chip_wbt,
    "g1-29dof-holosoma-wbt": g1_29dof_holosoma_wbt,
    "g1-29dof-holosoma-3pt-wbt": g1_29dof_holosoma_3pt_wbt,
    "g1-29dof-holosoma-2pt-wbt": g1_29dof_holosoma_2pt_wbt,
    "g1-29dof-holosoma-2pt-force-wbt": g1_29dof_holosoma_2pt_force_wbt,
    "g1-29dof-holosoma-2pt-force-ext-wbt": g1_29dof_holosoma_2pt_force_ext_wbt,
    # Both hybrid presets map to the SAME 165D obs config (servo on/off is a
    # runtime/browser flag, not an obs difference).
    "g1-29dof-holosoma-2pt-hybrid-wbt": g1_29dof_holosoma_2pt_hybrid_wbt,
    "g1-29dof-holosoma-2pt-hybrid-layerA-wbt": g1_29dof_holosoma_2pt_hybrid_wbt,
    # 3pt hybrid: both name keys map to the SAME 168D obs config (servo on/off is a
    # runtime/browser flag, not an obs difference), mirroring the 2pt-hybrid pair.
    "g1-29dof-holosoma-3pt-hybrid-wbt": g1_29dof_holosoma_3pt_hybrid_wbt,
    "g1-29dof-holosoma-3pt-hybrid-layerA-wbt": g1_29dof_holosoma_3pt_hybrid_wbt,
    # 3pt hybrid + global root pos/vel + 6D orientation: both name keys map to the SAME
    # 183D obs config (servo on/off is a runtime/browser flag), mirroring the 2pt/3pt pairs.
    "g1-29dof-holosoma-3pt-hybrid-global-vel-wbt": g1_29dof_holosoma_3pt_hybrid_global_vel_wbt,
    "g1-29dof-holosoma-3pt-hybrid-global-vel-layerA-wbt": g1_29dof_holosoma_3pt_hybrid_global_vel_wbt,
    # C1 unified single-policy: both training twins (R2 observe-only / R1 anneal) share
    # the SAME 185D obs config — obs is identical; only the training-time curriculum differs.
    "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt": g1_29dof_holosoma_3pt_hybrid_global_vel_unified_wbt,
    "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-anneal-wbt": g1_29dof_holosoma_3pt_hybrid_global_vel_unified_wbt,
    # LOCAL C1 unified twins (exp:g1-wbt-3pt-hybrid-unified[-anneal]): both name keys
    # map to the SAME 170D obs config (the pelvis-local 168D layout + contact_phase).
    # The R2/R1 split is a TRAINING curriculum, not an obs difference, and at deploy
    # neither injects a servo — the runtime forces observe-only off the contact_phase
    # term, so an R1 checkpoint's baked servo_observe_only=False cannot re-enable Δq.
    # There is deliberately NO "-layerA-" alias here. Note this is a NAMING decision,
    # not a functional guard: for the other hybrid families the "-layerA-" key maps to
    # the very same InferenceConfig and only the operator's HOLOSOMA_HYBRID_SERVO
    # differs, and the runtime keys observe-only off the contact_phase OBS TERM, so a
    # hypothetical "...-unified-layerA-wbt" would behave identically (I/Δx live, no
    # Δq). The name is simply a lie for this family — "Layer A" means "sensing only,
    # servo off", whereas the unified integrator is fully live as a reference
    # generator — so we do not offer it.
    "g1-29dof-holosoma-3pt-hybrid-unified-wbt": g1_29dof_holosoma_3pt_hybrid_unified_wbt,
    "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt": g1_29dof_holosoma_3pt_hybrid_unified_wbt,
    # The two MOTION-COMPLETION teleop families (user 2026-08-27). Only the "-anneal-"
    # spelling exists for each: unlike the 3pt family there is no R2 twin — both runs
    # (dlcxc59d / 7r54008a) train from scratch, because a 184D/183D actor cannot warm-start
    # off any 170D unified checkpoint. A non-anneal key would name a checkpoint that does
    # not exist, so it is deliberately absent rather than aliased.
    "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt": g1_29dof_holosoma_5pt_hybrid_unified_wbt,
    "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt": g1_29dof_holosoma_dense_hybrid_unified_wbt,
    "g1-29dof-holosoma-2pt-global-wbt": g1_29dof_holosoma_2pt_global_wbt,
    "g1-29dof-holosoma-3pt-global-wbt": g1_29dof_holosoma_3pt_global_wbt,
    "g1-29dof-holosoma-global-wbt": g1_29dof_holosoma_global_wbt,
    "g1-29dof-holosoma-global-vel-wbt": g1_29dof_holosoma_global_vel_wbt,
    "g1-29dof-holosoma-terrain-wbt": g1_29dof_holosoma_terrain_wbt,
    "g1-29dof-holosoma-terrain-no-heightscan-wbt": g1_29dof_holosoma_terrain_no_heightscan_wbt,
    "g1-29dof-holosoma-chip-local-wbt": g1_29dof_holosoma_chip_local_wbt,
    "g1-29dof-holosoma-chip-global-wbt": g1_29dof_holosoma_chip_global_wbt,
    "g1-29dof-holosoma-maskedmimic-3pt-wbt": g1_29dof_holosoma_maskedmimic_3pt_wbt,
    "g1-29dof-holosoma-maskedmimic-3pt-head-wbt": g1_29dof_holosoma_maskedmimic_3pt_head_wbt,
    "g1-29dof-holosoma-maskedmimic-3pt-real-head-wbt": g1_29dof_holosoma_maskedmimic_3pt_real_head_wbt,
    "g1-29dof-holosoma-maskedmimic-pelvis-wrists-wbt": g1_29dof_holosoma_maskedmimic_pelvis_wrists_wbt,
    "g1-29dof-holosoma-maskedmimic-4pt-wbt": g1_29dof_holosoma_maskedmimic_4pt_wbt,
    "g1-29dof-holosoma-maskedmimic-5pt-wbt": g1_29dof_holosoma_maskedmimic_5pt_wbt,
    "g1-29dof-holosoma-maskedmimic-full-wbt": g1_29dof_holosoma_maskedmimic_full_wbt,
    "g1-29dof-holosoma-maskedmimic-2pt-wbt": g1_29dof_holosoma_maskedmimic_2pt_wbt,
}
