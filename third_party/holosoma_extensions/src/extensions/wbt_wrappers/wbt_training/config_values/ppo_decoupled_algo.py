"""Algo preset value for the FALCON dual-agent PPO (``PPODecoupled``).

Builds a ``PPODecoupledAlgoConfig`` whose ``_target_`` points at the extension's
``PPODecoupled`` class. Per-agent networks reuse the TML-scale MLP dims (matching the
monolithic WBT presets) so capacity is comparable; the two agents get independent
init-noise / LR / KL. Registered as the ``algo`` field of the new preset.
"""

from __future__ import annotations

from holosoma.config_types.algo import LayerConfig, ModuleConfig, OptimizerConfig, PPOModuleDictConfig

from wbt_training.config_types.ppo_decoupled import (
    DecoupledAgentConfig,
    PPODecoupledAlgoConfig,
    PPODecoupledConfig,
)

_TML_HIDDEN = [2048, 2048, 1024, 1024, 512, 512]
_ADAM_NO_WD = OptimizerConfig(_target_="torch.optim.Adam", weight_decay=0.0)


def _actor_module() -> ModuleConfig:
    # output_dim ["robot_action_dim"] is resolved to THIS agent's slice width
    # (15 / 14) by PPOActor._process_module_config at build time.
    return ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=["robot_action_dim"],
        layer_config=LayerConfig(hidden_dims=_TML_HIDDEN, activation="ELU"),
    )


def _critic_module() -> ModuleConfig:
    return ModuleConfig(
        type="MLP",
        input_dim=["critic_obs"],
        output_dim=[1],
        layer_config=LayerConfig(hidden_dims=_TML_HIDDEN, activation="ELU"),
    )


def _agent() -> DecoupledAgentConfig:
    return DecoupledAgentConfig(
        actor=_actor_module(),
        critic=_critic_module(),
        init_noise_std=0.8,
        actor_learning_rate=1.0e-3,
        critic_learning_rate=1.0e-3,
        desired_kl=0.01,
        entropy_coef=0.01,
        actor_optimizer=_ADAM_NO_WD,
        critic_optimizer=_ADAM_NO_WD,
    )


# Throwaway single-agent module_dict: required by the inherited PPOConfig base field
# but UNUSED by PPODecoupled (it builds nets from `lower`/`upper`). Kept minimal.
_DUMMY_MODULE_DICT = PPOModuleDictConfig(actor=_actor_module(), critic=_critic_module())

g1_29dof_wbt_falcon_force_algo = PPODecoupledAlgoConfig(
    _target_="wbt_training.agents.ppo_decoupled.PPODecoupled",
    _recursive_=False,
    config=PPODecoupledConfig(
        module_dict=_DUMMY_MODULE_DICT,
        lower=_agent(),
        upper=_agent(),
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        max_grad_norm=1.0,
        schedule="adaptive",
        num_steps_per_env=24,
        save_interval=500,
        num_learning_iterations=30000,
        init_at_random_ep_len=True,
        empirical_normalization=False,
        use_symmetry=False,
    ),
)


__all__ = ["g1_29dof_wbt_falcon_force_algo"]
