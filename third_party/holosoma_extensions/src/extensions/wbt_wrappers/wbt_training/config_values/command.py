"""Command presets for G1 29-DOF whole-body tracking.

The body tracking list matches the 14-body configuration used by
TML-BeyondMimic-Generalist so that trained checkpoints are directly
deployable with ``wbt_wrappers_inference``.
"""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_values.command import DEFAULTS as CORE_DEFAULTS
from holosoma.config_values.wbt.g1.command import g1_29dof_wbt_command as _core_g1_wbt_cmd
from holosoma.config_values.wbt.g1.command import init_pose_config as g1_init_pose_config
from holosoma.config_types.command import CommandTermCfg, MotionConfig
from wbt_training.config_values.maskedmimic_commands import TMLMaskedMimicMotionConfig
from wbt_training.config_values.robot_config import G1_29DOF_WBT_BODY_NAMES

motion_config = MotionConfig(
    # Same default as core holosoma WBT. Override at the CLI, e.g.:
    #   --command.setup-terms.motion-command.params.motion-config.motion-file /path/to/motion.npz
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz",
    body_names_to_track=G1_29DOF_WBT_BODY_NAMES,
    body_name_ref=["pelvis"],
    use_adaptive_timesteps_sampler=True,
    noise_to_initial_pose=g1_init_pose_config,
)

g1_29dof_wbt_command = replace(
    _core_g1_wbt_cmd,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand", params={"motion_config": motion_config}
        )
    },
)

maskedmimic_motion_config = TMLMaskedMimicMotionConfig(
    motion_file=motion_config.motion_file,
    body_names_to_track=G1_29DOF_WBT_BODY_NAMES,
    body_name_ref=["pelvis"],
    use_adaptive_timesteps_sampler=True,
    noise_to_initial_pose=g1_init_pose_config,
    default_pose_transition_strategy="interpolation",
    default_pose_prepend_duration_s=2.0,
    start_at_timestep_zero_prob=0.3,
    pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
    },
    velocity_range={
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.2, 0.2),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
    },
    joint_position_range=(-0.1, 0.1),
)

g1_29dof_wbt_maskedmimic_command = replace(
    _core_g1_wbt_cmd,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="wbt_training.config_values.maskedmimic_commands:MaskedMimicMultiMotionCommand",
            params={"motion_config": maskedmimic_motion_config},
        )
    },
    reset_terms={
        "motion_command": CommandTermCfg(
            func="wbt_training.config_values.maskedmimic_commands:MaskedMimicMultiMotionCommand"
        )
    },
    step_terms={
        "motion_command": CommandTermCfg(
            func="wbt_training.config_values.maskedmimic_commands:MaskedMimicMultiMotionCommand"
        )
    },
)

terrain_motion_config = replace(motion_config, enable_default_pose_prepend=False, enable_default_pose_append=False)

g1_29dof_wbt_terrain_command = replace(
    _core_g1_wbt_cmd,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand", params={"motion_config": terrain_motion_config}
        )
    },
)

DEFAULTS = {
    **CORE_DEFAULTS,
    "g1_wbt": g1_29dof_wbt_command,
    "g1_wbt_maskedmimic": g1_29dof_wbt_maskedmimic_command,
    "g1_wbt_terrain": g1_29dof_wbt_terrain_command,
}
