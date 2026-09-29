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
from wbt_training.config_values import command
from wbt_training.config_values.observation import (
    g1_29dof_wbt_observation_1pt,
    g1_29dof_wbt_observation_1pt_fastsac,
    g1_29dof_wbt_observation_2pt,
    g1_29dof_wbt_observation_2pt_fastsac,
    g1_29dof_wbt_observation_3pt,
    g1_29dof_wbt_observation_3pt_fastsac,
    g1_29dof_wbt_observation_fastsac,
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
# 2-point tracking presets (left wrist + right wrist, 141D actor obs)
# ---------------------------------------------------------------------------
g1_wbt_2pt = replace(g1_wbt, observation=g1_29dof_wbt_observation_2pt)

g1_wbt_2pt_fast_sac = replace(g1_wbt_fast_sac, observation=g1_29dof_wbt_observation_2pt_fastsac)

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

DEFAULTS = {
    **CORE_DEFAULTS,
    "g1_wbt": g1_wbt,
    "g1_wbt_fast_sac": g1_wbt_fast_sac,
    "g1_wbt_3pt": g1_wbt_3pt,
    "g1_wbt_3pt_fast_sac": g1_wbt_3pt_fast_sac,
    "g1_wbt_2pt": g1_wbt_2pt,
    "g1_wbt_2pt_fast_sac": g1_wbt_2pt_fast_sac,
    "g1_wbt_1pt": g1_wbt_1pt,
    "g1_wbt_1pt_fast_sac": g1_wbt_1pt_fast_sac,
    "g1_wbt_wo_se": g1_wbt_wo_se,
    "g1_wbt_wo_se_fast_sac": g1_wbt_wo_se_fast_sac,
}

AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{name.replace('_', '-')}": cfg for name, cfg in DEFAULTS.items()}
        )
    ),
]
