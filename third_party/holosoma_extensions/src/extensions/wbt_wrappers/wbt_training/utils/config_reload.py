"""Checkpoint-config reload shim for the FALCON dual-agent preset.

WHY THIS EXISTS
---------------
The dual-agent algo config ``PPODecoupledAlgoConfig`` / ``PPODecoupledConfig`` are
SUBCLASSES of the core ``PPOAlgoConfig`` / ``PPOConfig`` (config_types/ppo_decoupled.py)
so they pass the core ``ExperimentConfig.algo`` field type
``Union[PPOAlgoConfig, FastSACAlgoConfig]`` WITHOUT a core edit. That works when the
config is validated from an INSTANCE (training / training-resume: the config comes from
the live in-memory tyro preset).

But the eval / deploy / benchmark / ONNX-export-from-checkpoint path reloads the config
from a serialized DICT — ``ExperimentConfig(**checkpoint["experiment_config"])`` in
``holosoma.utils.eval_utils`` (``_load_config_from_checkpoint`` / the wandb-yaml loader).
On DICT validation, pydantic's un-discriminated left-to-right Union matches the FIRST
member ``PPOAlgoConfig`` and constructs a PLAIN ``PPOAlgoConfig`` whose ``.config`` is a
plain ``PPOConfig`` — SILENTLY DROPPING the ``lower`` / ``upper`` fields while keeping
``_target_ = ...PPODecoupled``. ``PPODecoupled._init_config`` then crashes with
``AttributeError: 'PPOConfig' object has no attribute 'lower'`` (100% of FALCON
eval/deploy runs).

FIX (extension-local, no core edit)
------------------------------------
Monkeypatch the two ``eval_utils`` config loaders so that, when the serialized algo's
``_target_`` is the dual-agent class, we rebuild the ``PPODecoupledAlgoConfig`` INSTANCE
(instance path preserves ``lower`` / ``upper``, verified) and inject it into the
``ExperimentConfig`` instead of letting the dict Union downcast it. Any other preset is
untouched (the patch is a no-op unless the algo ``_target_`` matches).

``install_falcon_config_reload_patch()`` is idempotent; call it from the eval/deploy
entrypoints (or at extension import) before any checkpoint config is loaded.
"""

from __future__ import annotations

from typing import Any

_FALCON_ALGO_TARGET = "wbt_training.agents.ppo_decoupled.PPODecoupled"
_PATCHED_FLAG = "_wbt_falcon_config_reload_patched"


def _rebuild_falcon_algo_if_needed(config_data: dict) -> dict:
    """If ``config_data['algo']._target_`` is the dual-agent class, replace the algo
    dict with a rebuilt ``PPODecoupledAlgoConfig`` INSTANCE so the subclass fields
    survive ``ExperimentConfig(**config_data)``. Returns a shallow-copied dict (never
    mutates the caller's) with only the ``algo`` entry swapped; no-op otherwise."""
    algo = config_data.get("algo")
    if not isinstance(algo, dict) or algo.get("_target_") != _FALCON_ALGO_TARGET:
        return config_data
    from wbt_training.config_types.ppo_decoupled import PPODecoupledAlgoConfig, PPODecoupledConfig

    inner = algo.get("config")
    # Instance path preserves lower/upper (verified). Build inner first, then wrapper.
    cfg_inst = inner if not isinstance(inner, dict) else PPODecoupledConfig(**inner)
    algo_inst = PPODecoupledAlgoConfig(
        _target_=algo["_target_"],
        _recursive_=algo.get("_recursive_", False),
        config=cfg_inst,
    )
    out = dict(config_data)
    out["algo"] = algo_inst
    return out


def install_falcon_config_reload_patch() -> None:
    """Idempotently wrap holosoma.utils.eval_utils config loaders so FALCON dual-agent
    checkpoints reload with lower/upper intact. No-op for all other presets."""
    try:
        from holosoma.utils import eval_utils
    except ImportError:
        return
    if getattr(eval_utils, _PATCHED_FLAG, False):
        return

    import functools

    import torch
    import yaml

    _orig_from_ckpt = eval_utils._load_config_from_checkpoint
    _orig_saved = eval_utils.load_saved_experiment_config
    ExperimentConfig = eval_utils.ExperimentConfig
    _wandb_prefix = getattr(eval_utils, "_WANDB_PREFIX", "wandb://")
    config_name = getattr(eval_utils, "CONFIG_NAME", "config.yaml")

    @functools.wraps(_orig_from_ckpt)
    def _patched_from_checkpoint(checkpoint_path: Any):
        checkpoint_contents = torch.load(checkpoint_path, map_location="cpu")
        config_data = _rebuild_falcon_algo_if_needed(checkpoint_contents["experiment_config"])
        return ExperimentConfig(**config_data), checkpoint_contents.get("wandb_run_path")

    @functools.wraps(_orig_saved)
    def _patched_saved(checkpoint_cfg: Any):
        # Mirror load_saved_experiment_config, inserting the FALCON algo rebuild at both
        # the local-checkpoint and wandb-yaml validation sites. Non-FALCON checkpoints
        # take the identical path (rebuild is a no-op unless _target_ matches).
        from pathlib import Path

        checkpoint = getattr(checkpoint_cfg, "checkpoint", None)
        if checkpoint is None:
            raise ValueError("No checkpoint provided")
        checkpoint_str = str(checkpoint)
        if not checkpoint_str.startswith(_wandb_prefix):
            return _patched_from_checkpoint(Path(checkpoint_str).expanduser())
        wandb_run_path, _ = eval_utils._parse_wandb_reference(checkpoint_str)
        config_uri = f"{_wandb_prefix}{wandb_run_path}/{config_name}"
        cached_config_path = eval_utils.get_cached_file_path(config_uri)
        with open(cached_config_path) as f:
            config_data = _rebuild_falcon_algo_if_needed(yaml.safe_load(f))
        return ExperimentConfig(**config_data), wandb_run_path

    eval_utils._load_config_from_checkpoint = _patched_from_checkpoint
    eval_utils.load_saved_experiment_config = _patched_saved
    setattr(eval_utils, _PATCHED_FLAG, True)
