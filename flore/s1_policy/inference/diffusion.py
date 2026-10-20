"""Offline HL-v2 inference session for a DP-Diffusion checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from flore.s1_policy.data.norm_stats import unnormalize_quantile
from flore.s1_policy.inference.common import (
    HLSampleSource,
    RolloutResult,
    build_rollout_result,
    configure_deterministic_inference,
    dataclass_config,
    load_norm_bundle,
    prepare_model_batch,
    read_checkpoint_config,
    resolve_inference_device,
    strict_load_weights,
    validate_sampler_steps,
    validate_model_layout,
)
from flore.s1_policy.model.dp_diffusion import DPDiffusionConfig, DPDiffusionModel


class DiffusionInferenceSession:
    """A strict, deterministic offline inference session for one DDIM checkpoint."""

    algorithm = "diffusion"

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        zarr_paths: tuple[str, ...] | None = None,
        device: str = "cuda",
        num_steps: int | None = None,
    ) -> None:
        checkpoint, raw = read_checkpoint_config(checkpoint_dir, self.algorithm)
        source = HLSampleSource(raw["hl_data"], zarr_paths=zarr_paths)
        try:
            config_values = raw.get("diffusion")
            if not isinstance(config_values, dict):
                raise ValueError("diffusion checkpoint is missing its diffusion config block")
            config = dataclass_config(DPDiffusionConfig, config_values)
            validate_model_layout(config, raw["hl_data"], source.layout)
            norm_stats = load_norm_bundle(raw, checkpoint, source)
            if config.clip_sample and "actions" not in norm_stats:
                raise ValueError("diffusion clip_sample=True requires bundled action normalization")

            resolved_device = resolve_inference_device(device)
            configure_deterministic_inference(resolved_device)
            model = DPDiffusionModel(config)
            strict_load_weights(model, checkpoint, raw["_provenance"]["model_sha256"])
            model.to(resolved_device).eval()
        except Exception:
            source.close()
            raise

        self.checkpoint = checkpoint
        self.raw_config = raw
        self.config = config
        self.source = source
        self.norm_stats = norm_stats
        self.device = resolved_device
        self.model = model
        self.max_num_steps = int(config.num_train_timesteps)
        self._closed = False
        try:
            self.num_steps = validate_sampler_steps(
                num_steps if num_steps is not None else config.num_inference_steps,
                self.max_num_steps,
            )
        except ValueError:
            self.close()
            raise

    def infer(self, sample_index: int, *, seed: int = 0, num_steps: int | None = None) -> RolloutResult:
        if self._closed:
            raise RuntimeError("DiffusionInferenceSession is closed")
        steps = self.num_steps if num_steps is None else validate_sampler_steps(num_steps, self.max_num_steps)
        ref, sample = self.source.get(sample_index)
        batch = prepare_model_batch(sample, self.norm_stats, self.device)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.model.reset()
        with torch.inference_mode():
            sampled = self.model.sample_actions(batch, num_steps=steps, generator=generator)
        prediction = sampled[0].detach().float().cpu().numpy()
        if "actions" in self.norm_stats:
            prediction = unnormalize_quantile(prediction, self.norm_stats["actions"])
        return build_rollout_result(
            algorithm=self.algorithm,
            num_steps=steps,
            seed=seed,
            ref=ref,
            sample=sample,
            predicted_actions=np.asarray(prediction),
            source=self.source,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.source.close()
        if self.device.type == "cuda":
            self.model.to("cpu")
            scheduler = self.model.noise_scheduler
            for name, value in vars(scheduler).items():
                if isinstance(value, torch.Tensor):
                    setattr(scheduler, name, value.cpu())

    def __enter__(self) -> "DiffusionInferenceSession":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
