"""Offline HL-v2 inference session for a DP-ACT checkpoint.

The third sibling of :mod:`fuji.s1_policy.inference.flow` and
:mod:`fuji.s1_policy.inference.diffusion`, with the same strict loader contract: the
checkpoint's own ``hl_data`` block rebuilds the exact training :class:`HLDataset`, the
split/normalization provenance is cryptographically validated against the on-disk
dataset, and weights load ``strict=True`` behind a sha256 check.

ONE difference, and it is structural rather than a policy choice: ACT is a single-pass
CVAE, so there is no sampler loop. ``num_steps`` is therefore pinned to 1 and both
``max_num_steps`` and the model refuse anything else. The viewer's "Sampler steps" spinner
consequently has a single legal value — see :attr:`ACTInferenceSession.max_num_steps`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fuji.s1_policy.data.norm_stats import unnormalize_quantile
from fuji.s1_policy.inference.common import (
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
    validate_model_layout,
    validate_sampler_steps,
)
from fuji.s1_policy.model.dp_act import DPACTConfig, DPACTModel


class ACTInferenceSession:
    """A strict, deterministic offline inference session for one ACT checkpoint."""

    algorithm = "act"

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
            config_values = raw.get("act")
            if not isinstance(config_values, dict):
                raise ValueError("ACT checkpoint is missing its act config block")
            config = dataclass_config(DPACTConfig, config_values)
            validate_model_layout(config, raw["hl_data"], source.layout)
            norm_stats = load_norm_bundle(raw, checkpoint, source)

            resolved_device = resolve_inference_device(device)
            configure_deterministic_inference(resolved_device)
            model = DPACTModel(config)
            strict_load_weights(model, checkpoint, raw["_provenance"]["model_sha256"])
            # .eval() is load-bearing for ACT specifically, not just conventional: the DETR
            # transformer carries dropout=0.1, so in train() mode two identical rollouts
            # differ by O(1) in normalized units even with the latent pinned at z=0.
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
        #: 1, not None. The viewer reads this as its spinner's ``max``; ``None`` (the flow
        #: session's value, meaning "unbounded Euler steps") would advertise a knob that
        #: :meth:`DPACTModel.sample_actions` refuses.
        self.max_num_steps = 1
        self._closed = False
        try:
            self.num_steps = validate_sampler_steps(
                1 if num_steps is None else num_steps, self.max_num_steps)
        except ValueError:
            self.close()
            raise

    def infer(self, sample_index: int, *, seed: int = 0, num_steps: int | None = None) -> RolloutResult:
        """One deterministic rollout against one exact training anchor.

        ``seed`` is accepted for signature parity with the two siblings and recorded in the
        result, but with the default ``sample_latent_at_inference=False`` it cannot change
        the output: ``z`` is the prior MEAN (exactly 0) and the model is in eval mode. Two
        infers at different seeds are bit-identical, which is the honest behavior of a
        deterministic decoder — not a bug to work around.
        """
        if self._closed:
            raise RuntimeError("ACTInferenceSession is closed")
        steps = self.num_steps if num_steps is None else validate_sampler_steps(
            num_steps, self.max_num_steps)
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

    def __enter__(self) -> "ACTInferenceSession":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
