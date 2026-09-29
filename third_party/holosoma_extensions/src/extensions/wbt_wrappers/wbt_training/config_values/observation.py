"""Observation presets for G1 29-DOF whole-body tracking.

The core holosoma WBT actor observation omits ``projected_gravity`` (3D),
producing a 154D observation vector.  The TML-BeyondMimic-Generalist
training pipeline includes it, yielding the 157D vector that the
``wbt_wrappers_inference`` TML policy expects:

    command (58) + motion_ref_ori_b (6) + projected_gravity (3)
    + base_ang_vel (3) + dof_pos (29) + dof_vel (29) + actions (29) = 157

This module overrides the actor observation to add ``projected_gravity``
in the correct position so that checkpoints trained here are deployable
with ``wbt_wrappers_inference``.

**Observation history (TML-aligned):**  Actor observations use
``history_length=4`` and critic observations use ``history_length=10``,
matching TML-BeyondMimic-Generalist's temporal context stacking.

**Important — observation ordering:**  Holosoma's observation manager
concatenates terms in *dict insertion order* (manager.py:139-142).  Term keys
below use numeric prefixes (``obs0_`` … ``obs6_``) and are listed in that
order, so insertion order == the exact TML concatenation order (and also ==
alphabetical order, which is why the prefixes are chosen this way).  The
``func`` field is unaffected — it still points to the correct holosoma
observation function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wbt_training.config_values.robot_config import (
    LOWER_BODY_INDICES as LOWER_JOINT_ISAACLAB_INDICES,
    VR_1POINT_BODIES,
    VR_1POINT_OFFSETS,
    VR_2POINT_BODIES,
    VR_2POINT_OFFSETS,
    VR_3POINT_BODIES,
    VR_3POINT_OFFSETS,
    VR_3POINT_OFFSETS_TCP,
)

import torch

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_values.wbt.g1.observation import critic_obs_shared_terms, critic_obs_w_object_terms
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import quat_apply, quat_inverse, quat_mul, quaternion_to_matrix

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager

# Actor observation terms in TML-compatible order (157D total).
#
# Holosoma's observation manager concatenates terms in DICT INSERTION ORDER
# (see holosoma/managers/observation/manager.py:139-142). The numeric prefixes
# + listing the keys in order ensure insertion order == TML order (== alphabetical):
#   obs0 (58) + obs1 (6) + obs2 (3) + obs3 (3) + obs4 (29) + obs5 (29) + obs6 (29) = 157
_actor_obs_terms = {
    "obs0_motion_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_command", scale=1.0, noise=0.0
    ),
    "obs1_motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b", scale=1.0, noise=0.05
    ),
    "obs2_projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:projected_gravity", scale=1.0, noise=0.01
    ),
    "obs3_base_ang_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_ang_vel", scale=1.0, noise=0.2),
    "obs4_dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos", scale=1.0, noise=0.01),
    "obs5_dof_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel", scale=1.0, noise=0.5),
    "obs6_actions": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:actions", scale=1.0, noise=0.0),
}

actor_obs_tml = ObsGroupCfg(concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_terms)

critic_obs_tml = ObsGroupCfg(concatenate=True, enable_noise=False, history_length=10, terms=critic_obs_shared_terms)

# --- TML-aligned observation (PPO): actor history=4, critic history=10 ---
g1_29dof_wbt_observation_tml = ObservationManagerCfg(groups={"actor_obs": actor_obs_tml, "critic_obs": critic_obs_tml})

# --- FastSAC observation: TML actor obs but history=1 (core holosoma default)
# to avoid replay buffer OOM (buffer_size × num_envs × obs_dim is huge with
# long history).
_actor_obs_fastsac = ObsGroupCfg(concatenate=True, enable_noise=True, history_length=1, terms=_actor_obs_terms)

g1_29dof_wbt_observation_fastsac = ObservationManagerCfg(
    groups={
        "actor_obs": _actor_obs_fastsac,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_shared_terms
        ),
    }
)

# ===========================================================================
# 3-point tracking observation (TML-aligned)
# ===========================================================================
#
# Actor obs layout (144D):
#   command_lower_body (24) + vr_3point_pos (9) + vr_3point_orn (12)
#   + motion_ref_ori_b (6) + projected_gravity (3) + base_ang_vel (3)
#   + dof_pos (29) + dof_vel (29) + actions (29) = 144
#
# Matches TML tracking_env_cfg_3pt.py ObservationsCfg.PolicyCfg.
_actor_obs_3pt_terms = {
    "obs0_command_lower_body": ObsTermCfg(
        func="wbt_training.config_values.observation:command_lower_body", scale=1.0, noise=0.0
    ),
    "obs1_vr_3point_pos": ObsTermCfg(
        func="wbt_training.config_values.observation:vr_3point_local_target", scale=1.0, noise=0.0
    ),
    "obs2_vr_3point_orn": ObsTermCfg(
        func="wbt_training.config_values.observation:vr_3point_local_orn_target", scale=1.0, noise=0.0
    ),
    "obs3_motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b", scale=1.0, noise=0.05
    ),
    "obs4_projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:projected_gravity", scale=1.0, noise=0.01
    ),
    "obs5_base_ang_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_ang_vel", scale=1.0, noise=0.2),
    "obs6_dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos", scale=1.0, noise=0.01),
    "obs7_dof_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel", scale=1.0, noise=0.5),
    "obs8_actions": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:actions", scale=1.0, noise=0.0),
}

actor_obs_3pt = ObsGroupCfg(concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_3pt_terms)

# PPO 3pt observation: actor history=4, critic history=10
g1_29dof_wbt_observation_3pt = ObservationManagerCfg(groups={"actor_obs": actor_obs_3pt, "critic_obs": critic_obs_tml})

# FastSAC 3pt observation: history=1
g1_29dof_wbt_observation_3pt_fastsac = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(concatenate=True, enable_noise=True, history_length=1, terms=_actor_obs_3pt_terms),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_shared_terms
        ),
    }
)


# ===========================================================================
# 3-point tracking + root position (147D)
# ===========================================================================
#
# Actor obs layout (147D):
#   command_lower_body (24) + vr_3point_pos (9) + vr_3point_orn (12)
#   + motion_ref_pos_b (3) + motion_ref_ori_b (6) + projected_gravity (3)
#   + base_ang_vel (3) + dof_pos (29) + dof_vel (29) + actions (29) = 147
_actor_obs_3pt_global_terms = {
    "obs00_command_lower_body": ObsTermCfg(
        func="wbt_training.config_values.observation:command_lower_body", scale=1.0, noise=0.0
    ),
    "obs01_vr_3point_pos": ObsTermCfg(
        func="wbt_training.config_values.observation:vr_3point_local_target", scale=1.0, noise=0.0
    ),
    "obs02_vr_3point_orn": ObsTermCfg(
        func="wbt_training.config_values.observation:vr_3point_local_orn_target", scale=1.0, noise=0.0
    ),
    "obs03_motion_ref_pos_b": ObsTermCfg(
        func="wbt_training.config_values.observation:motion_anchor_pos_b", scale=1.0, noise=0.0
    ),
    "obs04_motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b", scale=1.0, noise=0.05
    ),
    "obs05_projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:projected_gravity", scale=1.0, noise=0.01
    ),
    "obs06_base_ang_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_ang_vel", scale=1.0, noise=0.2),
    "obs07_dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos", scale=1.0, noise=0.01),
    "obs08_dof_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel", scale=1.0, noise=0.5),
    "obs09_actions": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:actions", scale=1.0, noise=0.0),
}

g1_29dof_wbt_observation_3pt_global = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_3pt_global_terms
        ),
        "critic_obs": critic_obs_tml,
    }
)

# ===========================================================================
# 3-point tracking + root position + root velocity (153D)
# ===========================================================================
#
# Actor obs layout (153D):
#   command_lower_body (24) + vr_3point_pos (9) + vr_3point_orn (12)
#   + motion_ref_pos_b (3) + motion_ref_vel_b (3) + motion_ref_ori_b (6)
#   + projected_gravity (3) + base_ang_vel (3) + base_lin_vel (3)
#   + dof_pos (29) + dof_vel (29) + actions (29) = 153
_actor_obs_3pt_global_vel_terms = {
    "obs00_command_lower_body": ObsTermCfg(
        func="wbt_training.config_values.observation:command_lower_body", scale=1.0, noise=0.0
    ),
    "obs01_vr_3point_pos": ObsTermCfg(
        func="wbt_training.config_values.observation:vr_3point_local_target", scale=1.0, noise=0.0
    ),
    "obs02_vr_3point_orn": ObsTermCfg(
        func="wbt_training.config_values.observation:vr_3point_local_orn_target", scale=1.0, noise=0.0
    ),
    "obs03_motion_ref_pos_b": ObsTermCfg(
        func="wbt_training.config_values.observation:motion_anchor_pos_b", scale=1.0, noise=0.0
    ),
    "obs04_motion_ref_vel_b": ObsTermCfg(
        func="wbt_training.config_values.observation:motion_anchor_vel_b", scale=1.0, noise=0.0
    ),
    "obs05_motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b", scale=1.0, noise=0.05
    ),
    "obs06_projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:projected_gravity", scale=1.0, noise=0.01
    ),
    "obs07_base_ang_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_ang_vel", scale=1.0, noise=0.2),
    "obs08_base_lin_vel": ObsTermCfg(
        func="wbt_training.config_values.observation:base_lin_vel_b", scale=1.0, noise=0.1
    ),
    "obs09_dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos", scale=1.0, noise=0.01),
    "obs10_dof_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel", scale=1.0, noise=0.5),
    "obs11_actions": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:actions", scale=1.0, noise=0.0),
}

g1_29dof_wbt_observation_3pt_global_vel = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_3pt_global_vel_terms
        ),
        "critic_obs": critic_obs_tml,
    }
)


# ===========================================================================
# 2-point tracking + root position (left wrist + right wrist, global)
# ===========================================================================
#
# Actor obs layout (144D):
#   command_lower_body (24) + vr_2point_pos (6) + vr_2point_orn (12, 6D rot rep)
#   + motion_ref_pos_b (3) + motion_ref_ori_b (6) + projected_gravity (3) + base_ang_vel (3)
#   + dof_pos (29) + dof_vel (29) + actions (29) = 144
_actor_obs_2pt_global_terms = {
    "obs00_command_lower_body": ObsTermCfg(
        func="wbt_training.config_values.observation:command_lower_body", scale=1.0, noise=0.0
    ),
    "obs01_vr_2point_pos": ObsTermCfg(
        func="wbt_training.config_values.observation:vr_2point_local_target", scale=1.0, noise=0.0
    ),
    "obs02_vr_2point_orn": ObsTermCfg(
        func="wbt_training.config_values.observation:vr_2point_local_orn_target", scale=1.0, noise=0.0
    ),
    "obs03_motion_ref_pos_b": ObsTermCfg(
        func="wbt_training.config_values.observation:motion_anchor_pos_b", scale=1.0, noise=0.0
    ),
    "obs04_motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b", scale=1.0, noise=0.05
    ),
    "obs05_projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:projected_gravity", scale=1.0, noise=0.01
    ),
    "obs06_base_ang_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_ang_vel", scale=1.0, noise=0.2),
    "obs07_dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos", scale=1.0, noise=0.01),
    "obs08_dof_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel", scale=1.0, noise=0.5),
    "obs09_actions": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:actions", scale=1.0, noise=0.0),
}

g1_29dof_wbt_observation_2pt_global = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_2pt_global_terms
        ),
        "critic_obs": critic_obs_tml,
    }
)


# ===========================================================================
# 2-point tracking (left wrist + right wrist)
# ===========================================================================
#
# Actor obs layout (141D):
#   command_lower_body (24) + vr_2point_pos (6) + vr_2point_orn (12, 6D rot rep)
#   + motion_ref_ori_b (6) + projected_gravity (3) + base_ang_vel (3)
#   + dof_pos (29) + dof_vel (29) + actions (29) = 141
_actor_obs_2pt_terms = {
    "obs0_command_lower_body": ObsTermCfg(
        func="wbt_training.config_values.observation:command_lower_body", scale=1.0, noise=0.0
    ),
    "obs1_vr_2point_pos": ObsTermCfg(
        func="wbt_training.config_values.observation:vr_2point_local_target", scale=1.0, noise=0.0
    ),
    "obs2_vr_2point_orn": ObsTermCfg(
        func="wbt_training.config_values.observation:vr_2point_local_orn_target", scale=1.0, noise=0.0
    ),
    "obs3_motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b", scale=1.0, noise=0.05
    ),
    "obs4_projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:projected_gravity", scale=1.0, noise=0.01
    ),
    "obs5_base_ang_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_ang_vel", scale=1.0, noise=0.2),
    "obs6_dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos", scale=1.0, noise=0.01),
    "obs7_dof_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel", scale=1.0, noise=0.5),
    "obs8_actions": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:actions", scale=1.0, noise=0.0),
}

# PPO 2pt observation: actor history=4, critic history=10
g1_29dof_wbt_observation_2pt = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_2pt_terms),
        "critic_obs": critic_obs_tml,
    }
)

# FastSAC 2pt observation: history=1
g1_29dof_wbt_observation_2pt_fastsac = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(concatenate=True, enable_noise=True, history_length=1, terms=_actor_obs_2pt_terms),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_shared_terms
        ),
    }
)

# ===========================================================================
# 1-point tracking (right wrist only)
# ===========================================================================
#
# Actor obs layout (130D):
#   command_lower_body (24) + vr_1point_pos (3) + vr_1point_orn (4)
#   + motion_ref_ori_b (6) + projected_gravity (3) + base_ang_vel (3)
#   + dof_pos (29) + dof_vel (29) + actions (29) = 130
_actor_obs_1pt_terms = {
    "obs0_command_lower_body": ObsTermCfg(
        func="wbt_training.config_values.observation:command_lower_body", scale=1.0, noise=0.0
    ),
    "obs1_vr_1point_pos": ObsTermCfg(
        func="wbt_training.config_values.observation:vr_1point_local_target", scale=1.0, noise=0.0
    ),
    "obs2_vr_1point_orn": ObsTermCfg(
        func="wbt_training.config_values.observation:vr_1point_local_orn_target", scale=1.0, noise=0.0
    ),
    "obs3_motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b", scale=1.0, noise=0.05
    ),
    "obs4_projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:projected_gravity", scale=1.0, noise=0.01
    ),
    "obs5_base_ang_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_ang_vel", scale=1.0, noise=0.2),
    "obs6_dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos", scale=1.0, noise=0.01),
    "obs7_dof_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel", scale=1.0, noise=0.5),
    "obs8_actions": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:actions", scale=1.0, noise=0.0),
}

# PPO 1pt observation: actor history=4, critic history=10
g1_29dof_wbt_observation_1pt = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_1pt_terms),
        "critic_obs": critic_obs_tml,
    }
)

# FastSAC 1pt observation: history=1
g1_29dof_wbt_observation_1pt_fastsac = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(concatenate=True, enable_noise=True, history_length=1, terms=_actor_obs_1pt_terms),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_shared_terms
        ),
    }
)

# ===========================================================================
# Without state estimation (TML-aligned)
# ===========================================================================
# Removes motion_ref_pos_b (3D) and base_lin_vel (3D) from critic obs.
# These terms require a linear velocity state estimator; removing them makes
# the policy fully deployable without one.
# Matches TML G1FlatWoStateEstimationEnvCfg.
_critic_obs_wo_se_terms = {
    k: v for k, v in critic_obs_shared_terms.items() if k not in ("motion_ref_pos_b", "base_lin_vel")
}

critic_obs_wo_se = ObsGroupCfg(concatenate=True, enable_noise=False, history_length=10, terms=_critic_obs_wo_se_terms)

# PPO w/o state estimation: same actor obs (157D), reduced critic obs
g1_29dof_wbt_observation_wo_se = ObservationManagerCfg(
    groups={"actor_obs": actor_obs_tml, "critic_obs": critic_obs_wo_se}
)

# FastSAC w/o state estimation: history=1
g1_29dof_wbt_observation_wo_se_fastsac = ObservationManagerCfg(
    groups={
        "actor_obs": _actor_obs_fastsac,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=_critic_obs_wo_se_terms
        ),
    }
)

# ===========================================================================
# CHIP local (3pt + compliance + selective 10-frame history, 984D)
# ===========================================================================
#
# Obs layout (984D):
#   --- single-frame (54D) ---
#   command_lower_body (24) + vr_3point_pos (9) + vr_3point_orn (12)
#   + motion_ref_ori_b (6) + compliance (3)
#   --- 10-frame history (930D) ---
#   [projected_gravity (3) + base_ang_vel (3) + dof_pos (29)
#    + dof_vel (29) + actions (29)] * 10
#
# Total: 54 + 930 = 984
_actor_obs_chip_local_terms = {
    "obs00_command_lower_body": ObsTermCfg(
        func="wbt_training.config_values.observation:command_lower_body", scale=1.0, noise=0.0
    ),
    "obs01_vr_3point_pos": ObsTermCfg(
        func="wbt_training.config_values.observation:vr_3point_local_target", scale=1.0, noise=0.0
    ),
    "obs02_vr_3point_orn": ObsTermCfg(
        func="wbt_training.config_values.observation:vr_3point_local_orn_target", scale=1.0, noise=0.0
    ),
    "obs03_motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b", scale=1.0, noise=0.05
    ),
    "obs04_projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:projected_gravity", scale=1.0, noise=0.01
    ),
    "obs05_base_ang_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_ang_vel", scale=1.0, noise=0.2),
    "obs06_dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos", scale=1.0, noise=0.01),
    "obs07_dof_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel", scale=1.0, noise=0.5),
    "obs08_actions": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:actions", scale=1.0, noise=0.0),
    "obs09_compliance": ObsTermCfg(func="wbt_training.config_values.observation:compliance", scale=1.0, noise=0.0),
}

_CHIP_LOCAL_NO_HISTORY = {
    "obs00_command_lower_body",
    "obs01_vr_3point_pos",
    "obs02_vr_3point_orn",
    "obs03_motion_ref_ori_b",
    "obs09_compliance",
}

actor_obs_chip_local = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=10, terms=_actor_obs_chip_local_terms
)
object.__setattr__(actor_obs_chip_local, "_no_history_terms", _CHIP_LOCAL_NO_HISTORY)  # type: ignore[attr-defined]

g1_29dof_wbt_observation_chip_local = ObservationManagerCfg(
    groups={"actor_obs": actor_obs_chip_local, "critic_obs": critic_obs_tml}
)

# ===========================================================================
# CHIP global (local_chip + motion_anchor_pos_b, 987D)
# ===========================================================================
#
# Adds motion_ref_pos_b (3D, no history) for root-state tracking.
# Total: 54 + 3 + 930 = 987
_actor_obs_chip_global_terms = {
    "obs00_command_lower_body": ObsTermCfg(
        func="wbt_training.config_values.observation:command_lower_body", scale=1.0, noise=0.0
    ),
    "obs01_vr_3point_pos": ObsTermCfg(
        func="wbt_training.config_values.observation:vr_3point_local_target", scale=1.0, noise=0.0
    ),
    "obs02_vr_3point_orn": ObsTermCfg(
        func="wbt_training.config_values.observation:vr_3point_local_orn_target", scale=1.0, noise=0.0
    ),
    "obs03_motion_ref_pos_b": ObsTermCfg(
        func="wbt_training.config_values.observation:motion_anchor_pos_b", scale=1.0, noise=0.0
    ),
    "obs04_motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b", scale=1.0, noise=0.05
    ),
    "obs05_projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:projected_gravity", scale=1.0, noise=0.01
    ),
    "obs06_base_ang_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_ang_vel", scale=1.0, noise=0.2),
    "obs07_dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos", scale=1.0, noise=0.01),
    "obs08_dof_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel", scale=1.0, noise=0.5),
    "obs09_actions": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:actions", scale=1.0, noise=0.0),
    "obs10_compliance": ObsTermCfg(func="wbt_training.config_values.observation:compliance", scale=1.0, noise=0.0),
}

_CHIP_GLOBAL_NO_HISTORY = {
    "obs00_command_lower_body",
    "obs01_vr_3point_pos",
    "obs02_vr_3point_orn",
    "obs03_motion_ref_pos_b",
    "obs04_motion_ref_ori_b",
    "obs10_compliance",
}

actor_obs_chip_global = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=10, terms=_actor_obs_chip_global_terms
)
object.__setattr__(actor_obs_chip_global, "_no_history_terms", _CHIP_GLOBAL_NO_HISTORY)  # type: ignore[attr-defined]

g1_29dof_wbt_observation_chip_global = ObservationManagerCfg(
    groups={"actor_obs": actor_obs_chip_global, "critic_obs": critic_obs_tml}
)

# ===========================================================================
# Global tracking — dense (TML obs + root position, 160D)
# ===========================================================================
#
# Adds motion_anchor_pos_b (3D) to the standard 157D TML obs for root
# position tracking.  The policy receives both local orientation and global
# root position, enabling it to track absolute motion trajectories.
#
# Obs layout (160D):
#   motion_command (58) + motion_ref_ori_b (6) + motion_ref_pos_b (3)
#   + projected_gravity (3) + base_ang_vel (3) + dof_pos (29)
#   + dof_vel (29) + actions (29) = 160
_actor_obs_global_terms = {
    "obs0_motion_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_command", scale=1.0, noise=0.0
    ),
    "obs1_motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b", scale=1.0, noise=0.05
    ),
    "obs2_motion_ref_pos_b": ObsTermCfg(
        func="wbt_training.config_values.observation:motion_anchor_pos_b", scale=1.0, noise=0.0
    ),
    "obs3_projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:projected_gravity", scale=1.0, noise=0.01
    ),
    "obs4_base_ang_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_ang_vel", scale=1.0, noise=0.2),
    "obs5_dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos", scale=1.0, noise=0.01),
    "obs6_dof_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel", scale=1.0, noise=0.5),
    "obs7_actions": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:actions", scale=1.0, noise=0.0),
}

actor_obs_global = ObsGroupCfg(concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_global_terms)

g1_29dof_wbt_observation_global = ObservationManagerCfg(
    groups={"actor_obs": actor_obs_global, "critic_obs": critic_obs_tml}
)

# FastSAC global: history=1
g1_29dof_wbt_observation_global_fastsac = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(concatenate=True, enable_noise=True, history_length=1, terms=_actor_obs_global_terms),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_shared_terms
        ),
    }
)

# ===========================================================================
# Global tracking — dense + root velocity (TML obs + root pos + root vel, 166D)
# ===========================================================================
#
# Adds both root position (3D) and reference root velocity (3D) plus robot
# base linear velocity (3D) for full global tracking.
#
# Obs layout (166D):
#   motion_command (58) + motion_ref_ori_b (6) + motion_ref_pos_b (3)
#   + motion_ref_vel_b (3) + projected_gravity (3) + base_ang_vel (3)
#   + base_lin_vel (3) + dof_pos (29) + dof_vel (29) + actions (29) = 166
_actor_obs_global_vel_terms = {
    "obs00_motion_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_command", scale=1.0, noise=0.0
    ),
    "obs01_motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b", scale=1.0, noise=0.05
    ),
    "obs02_motion_ref_pos_b": ObsTermCfg(
        func="wbt_training.config_values.observation:motion_anchor_pos_b", scale=1.0, noise=0.0
    ),
    "obs03_motion_ref_vel_b": ObsTermCfg(
        func="wbt_training.config_values.observation:motion_anchor_vel_b", scale=1.0, noise=0.0
    ),
    "obs04_projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:projected_gravity", scale=1.0, noise=0.01
    ),
    "obs05_base_ang_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_ang_vel", scale=1.0, noise=0.2),
    "obs06_base_lin_vel": ObsTermCfg(
        func="wbt_training.config_values.observation:base_lin_vel_b", scale=1.0, noise=0.1
    ),
    "obs07_dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos", scale=1.0, noise=0.01),
    "obs08_dof_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel", scale=1.0, noise=0.5),
    "obs09_actions": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:actions", scale=1.0, noise=0.0),
}

actor_obs_global_vel = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_global_vel_terms
)

g1_29dof_wbt_observation_global_vel = ObservationManagerCfg(
    groups={"actor_obs": actor_obs_global_vel, "critic_obs": critic_obs_tml}
)

# FastSAC global+vel: history=1
g1_29dof_wbt_observation_global_vel_fastsac = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=1, terms=_actor_obs_global_vel_terms
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_shared_terms
        ),
    }
)

# ===========================================================================
# Terrain observation (dense + root pos + base_lin_vel + height_scan)
# ===========================================================================
#
# Alphabetical concatenation order (449D per frame, history_length=1):
#   actions(29) + base_ang_vel(3) + base_lin_vel(3) + dof_pos(29) + dof_vel(29)
#   + height_scan(289) + motion_command(58) + motion_ref_ori_b(6) + motion_ref_pos_b(3)
_height_scan_term = ObsTermCfg(func="wbt_training.config_values.observation:height_scan", scale=1.0, noise=0.0)

_actor_obs_terrain_terms = {
    "actions": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:actions", scale=1.0, noise=0.0),
    "base_ang_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_ang_vel", scale=1.0, noise=0.2),
    "base_lin_vel": ObsTermCfg(func="wbt_training.config_values.observation:base_lin_vel_b", scale=1.0, noise=0.5),
    "dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos", scale=1.0, noise=0.01),
    "dof_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel", scale=1.0, noise=0.5),
    "height_scan": _height_scan_term,
    "motion_command": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:motion_command", scale=1.0, noise=0.0),
    "motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b", scale=1.0, noise=0.05
    ),
    "motion_ref_pos_b": ObsTermCfg(
        func="wbt_training.config_values.observation:motion_anchor_pos_b", scale=1.0, noise=0.25
    ),
}

_critic_obs_terrain_terms = {**critic_obs_shared_terms, "height_scan": _height_scan_term}

g1_29dof_wbt_observation_terrain = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(concatenate=True, enable_noise=True, history_length=1, terms=_actor_obs_terrain_terms),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=_critic_obs_terrain_terms
        ),
    }
)

# ===========================================================================
# Full WBT with object (existing)
# ===========================================================================
g1_29dof_wbt_observation_tml_w_object = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_tml,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=10, terms=critic_obs_w_object_terms
        ),
    }
)


# ---------------------------------------------------------------------------
# Observation term runtime functions
# (merged from observations.py)
# ---------------------------------------------------------------------------


def _get_motion_command(env: WholeBodyTrackingManager) -> MotionCommand:
    mc = env.command_manager.get_state("motion_command")
    assert isinstance(mc, MotionCommand), f"Expected MotionCommand, got {type(mc)}"
    return mc


def _resolve_body_indices(mc: MotionCommand, body_names: list[str]) -> list[int]:
    tracked = mc.motion_cfg.body_names_to_track
    return [tracked.index(name) for name in body_names]


def _resolve_offsets(
    env: WholeBodyTrackingManager, body_names: list[str], body_offsets: list[list[float]]
) -> torch.Tensor:
    # Key on the OFFSETS as well as the bodies: `vr_3point_local_target` and
    # `vr_3point_local_target_tcp` share the same three body names but carry different
    # offsets (0.18 VR lever vs 0.242 force TCP). Keying on bodies alone would let
    # whichever term ran first poison the other with its offsets.
    key = f"_wbt_vr_offsets_{'_'.join(body_names)}_{hash(tuple(map(tuple, body_offsets)))}"
    if not hasattr(env, key):
        offsets = torch.tensor(body_offsets, dtype=torch.float32, device=env.device).unsqueeze(0)
        setattr(env, key, offsets)
    return getattr(env, key)


def _vr_local_target(
    env: WholeBodyTrackingManager, body_names: list[str], body_offsets: list[list[float]]
) -> torch.Tensor:
    mc = _get_motion_command(env)
    n = len(body_names)
    body_indices = _resolve_body_indices(mc, body_names)
    offsets = _resolve_offsets(env, body_names, body_offsets)

    ref_quat = mc.ref_quat_w
    ref_pos = mc.ref_pos_w

    vr_pos_w = mc.body_pos_w[:, body_indices, :]
    vr_quat_w = mc.body_quat_w[:, body_indices, :]

    vr_pos_w = vr_pos_w + quat_apply(vr_quat_w, offsets.expand(env.num_envs, -1, -1), w_last=True)

    ref_quat_expanded = ref_quat.unsqueeze(1).expand(-1, n, -1)
    diff = vr_pos_w - ref_pos.unsqueeze(1)
    local_pos = quat_apply(quat_inverse(ref_quat_expanded, w_last=True), diff, w_last=True)

    return local_pos.reshape(env.num_envs, -1)


def _vr_local_orn_target(env: WholeBodyTrackingManager, body_names: list[str], return_6d: bool = False) -> torch.Tensor:
    mc = _get_motion_command(env)
    n = len(body_names)
    body_indices = _resolve_body_indices(mc, body_names)

    ref_quat = mc.ref_quat_w
    vr_quat_w = mc.body_quat_w[:, body_indices, :]

    ref_quat_expanded = ref_quat.unsqueeze(1).expand(-1, n, -1)
    local_quat = quat_mul(quat_inverse(ref_quat_expanded, w_last=True), vr_quat_w, w_last=True)

    if return_6d:
        # Continuous 6D rotation rep = first two columns of the rotation matrix
        # (per body: [r00, r01, r10, r11, r20, r21], C-order). Mirrors the
        # baseline `motion_ref_ori_b` term EXACTLY (holosoma observation terms
        # wbt.py:149-158: quaternion_to_matrix(q, w_last=True)[..., :2].reshape).
        # Unlike the raw quaternion this has no double cover, so it round-trips
        # cleanly train<->deploy: at inference the wrist quat is rebuilt via
        # Eigen `pin.Quaternion(R).coeffs()` which picks an inconsistent sign
        # per frame (q ~ -q), an antipode the quaternion-trained policy never
        # saw -> OOD -> the observed left-wrist blow-up. 6D removes that branch.
        mat = quaternion_to_matrix(local_quat, w_last=True)
        return mat[..., :2].reshape(env.num_envs, -1)

    return local_quat.reshape(env.num_envs, -1)


def command_lower_body(env: WholeBodyTrackingManager) -> torch.Tensor:
    mc = _get_motion_command(env)
    joint_pos = mc.joint_pos[:, LOWER_JOINT_ISAACLAB_INDICES]
    joint_vel = mc.joint_vel[:, LOWER_JOINT_ISAACLAB_INDICES]
    return torch.cat([joint_pos, joint_vel], dim=1)


def vr_3point_local_target(env: WholeBodyTrackingManager) -> torch.Tensor:
    return _vr_local_target(env, VR_3POINT_BODIES, VR_3POINT_OFFSETS)


def vr_3point_local_target_tcp(env: WholeBodyTrackingManager) -> torch.Tensor:
    """3-point position target with the two HAND points moved onto the force TCP.

    Identical to `vr_3point_local_target` in every respect (same bodies, same
    wrist-local→world→pelvis-local transform chain, same 9D layout) except the hand
    offsets: `(0.242, 0, 0)` — the fingertip grasp midpoint the hybrid force stack
    already acts on — instead of the default `(0.18, ∓0.025, 0)` VR-tracker lever.
    So POSITION and FORCE supervise the SAME physical point.

    Used ONLY by `exp:g1-wbt-3pt-hybrid-unified[-anneal]` (user 2026-08-03). A DISTINCT
    obs-term name (not a re-parameterised `vr_3point_pos`) so the term name recorded in
    each checkpoint's config states which geometry it trained on: the pre-2026-08-03
    unified checkpoints carry `vr_3point_pos` (0.18) and cannot be silently deployed
    against the 0.242 FK, or vice-versa. The head point is unchanged.
    """
    return _vr_local_target(env, VR_3POINT_BODIES, VR_3POINT_OFFSETS_TCP)


def vr_3point_local_orn_target(env: WholeBodyTrackingManager) -> torch.Tensor:
    return _vr_local_orn_target(env, VR_3POINT_BODIES)


def vr_2point_local_target(env: WholeBodyTrackingManager) -> torch.Tensor:
    return _vr_local_target(env, VR_2POINT_BODIES, VR_2POINT_OFFSETS)


def vr_2point_local_orn_target(env: WholeBodyTrackingManager) -> torch.Tensor:
    # 6D rotation rep (12D = 2 wrists x 6) instead of quaternion (8D) to avoid
    # the antipodal/double-cover OOD bug at deploy. 1pt/3pt stay on quaternion.
    return _vr_local_orn_target(env, VR_2POINT_BODIES, return_6d=True)


def vr_1point_local_target(env: WholeBodyTrackingManager) -> torch.Tensor:
    return _vr_local_target(env, VR_1POINT_BODIES, VR_1POINT_OFFSETS)


def vr_1point_local_orn_target(env: WholeBodyTrackingManager) -> torch.Tensor:
    return _vr_local_orn_target(env, VR_1POINT_BODIES)


def compliance(env: WholeBodyTrackingManager) -> torch.Tensor:
    key = "_wbt_compliance"
    ctr_key = "_wbt_compliance_counter"

    if not hasattr(env, key):
        val = torch.zeros(env.num_envs, 3, device=env.device)
        val[:, 0] = 0.5
        val[:, 1] = 0.5
        setattr(env, key, val)
        setattr(env, ctr_key, torch.randint(100, 200, (env.num_envs,), device=env.device))

    counter: torch.Tensor = getattr(env, ctr_key)
    counter -= 1
    resample = counter <= 0
    if resample.any():
        n = int(resample.sum().item())
        val: torch.Tensor = getattr(env, key)
        val[resample, 0] = torch.rand(n, device=env.device)
        val[resample, 1] = torch.rand(n, device=env.device)
        val[resample, 2] = 0.0
        counter[resample] = torch.randint(100, 200, (n,), device=env.device)

    return getattr(env, key)


def motion_anchor_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    from holosoma.managers.observation.terms.wbt import motion_ref_pos_b

    return motion_ref_pos_b(env)


def motion_anchor_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    mc = _get_motion_command(env)
    ref_vel_w = mc.body_lin_vel_w[:, 0, :]
    robot_quat = env.base_quat
    ref_vel_b = quat_apply(quat_inverse(robot_quat, w_last=True), ref_vel_w, w_last=True)
    return ref_vel_b


def base_lin_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    from holosoma.managers.observation.terms.wbt import base_lin_vel

    return base_lin_vel(env)


# Single canonical implementation lives in `wbt_training.config_values.terrain`;
# re-export here so both dotted paths (observation:height_scan and
# terrain:height_scan) resolve to the same function.
from wbt_training.config_values.terrain import height_scan  # noqa: E402,F401
