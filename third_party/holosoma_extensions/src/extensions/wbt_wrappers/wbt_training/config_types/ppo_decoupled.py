"""Config dataclasses for the FALCON-style dual-agent PPO (``PPODecoupled``).

Additive, extension-local. To satisfy the CORE ``ExperimentConfig.algo`` field type
``Union[PPOAlgoConfig, FastSACAlgoConfig]`` WITHOUT editing the core Union, these
SUBCLASS the core ``PPOConfig`` / ``PPOAlgoConfig`` — pydantic's Union validation
accepts a subclass instance (verified) and preserves the extra per-agent fields, so no
core submodule edit is needed.

``PPODecoupledConfig`` adds two ``DecoupledAgentConfig`` (lower / upper) on top of
``PPOConfig``. The inherited single-agent fields (``module_dict``, ``init_noise_std``,
``actor_learning_rate``, …) are NOT used by ``PPODecoupled`` (it reads ``lower``/``upper``
instead); ``module_dict`` is still required by the base dataclass, so the preset passes a
throwaway one. ``num_learning_iterations`` / ``num_steps_per_env`` are inherited from
``PPOConfig`` (nightly/eval + logging read them).
"""

from __future__ import annotations

from dataclasses import field

from pydantic.dataclasses import dataclass

from holosoma.config_types.algo import ModuleConfig, OptimizerConfig, PPOAlgoConfig, PPOConfig


@dataclass(frozen=True)
class DecoupledAgentConfig:
    """Per-agent network + noise/LR/KL config (one per body group)."""

    actor: ModuleConfig
    """Actor MLP config (output_dim ``["robot_action_dim"]`` resolves to THIS agent's
    action-slice width at build time — not the full 29)."""

    critic: ModuleConfig
    """Critic MLP config (scalar value head; consumes the shared critic_obs)."""

    init_noise_std: float = 0.8
    actor_learning_rate: float = 1.0e-3
    critic_learning_rate: float = 1.0e-3
    desired_kl: float = 0.01
    entropy_coef: float = 0.01
    max_actor_learning_rate: float | None = None
    min_actor_learning_rate: float | None = None
    max_critic_learning_rate: float | None = None
    min_critic_learning_rate: float | None = None

    actor_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    critic_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))


@dataclass(frozen=True)
class PPODecoupledConfig(PPOConfig):
    """PPOConfig + two per-agent configs. Subclasses PPOConfig so it validates against
    the core ``AlgoInitConfig``/``AlgoConfig`` Union without a core edit.

    NOTE: the inherited single-agent fields (``module_dict`` etc.) are unused by
    ``PPODecoupled`` — it reads ``lower`` / ``upper``. ``module_dict`` is still a required
    base field, so the preset supplies a throwaway one. Fields WITHOUT defaults must
    precede those with defaults in the *combined* dataclass; PPOConfig's ``module_dict``
    is the only no-default base field, and ``lower``/``upper`` here also lack defaults, so
    all three are keyword-supplied by the preset (no ordering issue for kw construction).
    """

    lower: DecoupledAgentConfig = None  # type: ignore[assignment]  # required in practice; see preset
    upper: DecoupledAgentConfig = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.lower is None or self.upper is None:
            raise ValueError("PPODecoupledConfig requires both `lower` and `upper` DecoupledAgentConfig.")


@dataclass(frozen=True)
class PPODecoupledAlgoConfig(PPOAlgoConfig):
    """Outer wrapper. Subclasses PPOAlgoConfig; ``config`` carries a PPODecoupledConfig
    (a PPOConfig subclass, so the base field type still validates)."""

    config: PPODecoupledConfig  # narrows PPOAlgoConfig.config: PPOConfig
