"""DDPM/DDIM diffusion VLA base — the equal-level sibling of ``FlowMatchingVLABase``.

This is the diffusion counterpart of :class:`flore.s1_policy.model.base.FlowMatchingVLABase`.
It owns the *objective* (the forward-noising process and the reverse sampler) while a
subclass implements :meth:`_encode_global_cond` to map observations to a flat conditioning
vector — exactly the same contract the flow base uses, so the two are drop-in siblings
selected by ``TrainConfig.algo``.

Where it differs from flow, and why
-----------------------------------
The flow base integrates a *continuous* rectified-flow ODE (Beta-sampled time in
``[0, 1]``, linear interpolant, Euler steps). This base runs the *discrete* DDPM/DDIM
process instead, using ``diffusers``' battle-tested :class:`DDIMScheduler` for the noise
schedule and the reverse step — the SAME solver UMI-on-Air's ``DiffusionUnetTimmPolicy``
uses, so the numerics match a well-understood reference rather than a hand-rolled one.

    training  (``compute_loss``): sample an INTEGER timestep ``t ∈ [0, num_train_timesteps)``,
              build ``x_t = scheduler.add_noise(actions, noise, t)``, and regress the model
              output onto the scheduler's target (ε for ``prediction_type='epsilon'``).
    sampling  (``sample_actions``): ``x_T = randn``; for each of ``num_inference_steps``
              scheduler timesteps, predict, then ``x_{t-1} = scheduler.step(...).prev_sample``.

Two tricks carried over from UMI (both default-on, both optional):

* **Input perturbation** (``input_pertub``, DDPM-IP): the noise ADDED to the trajectory is
  ``noise + input_pertub * randn``, but the regression target stays the ORIGINAL ``noise``.
  Alleviates exposure bias. https://github.com/forever208/DDPM-IP
* **Multi-draw** (``num_draws``, UMI's ``train_diffusion_n_samples``): run the encoder once
  and share its conditioning across K independent (noise, timestep) draws — the identical
  amortization the flow base does.

Masking is IDENTICAL to the flow base (same four masks, same composition, same reason for
zeroing unsupervised action entries before noising), so the HL-v2 per-(step, dim) validity
contract is honored the same way on both objectives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flore.s1_policy.model.loss_metrics import squared_error_sums


class DiffusionVLABase(ABC, nn.Module):
    """Shared DDPM-train / DDIM-sample loop for a diffusion action decoder.

    Subclasses implement :meth:`_encode_global_cond` (same signature as the flow base) and
    MUST, in ``__init__`` after setting ``self.config`` and building ``self.decoder``, call
    :meth:`_build_noise_scheduler`. This base handles noise/timestep sampling, the masked
    ε-prediction loss, and the DDIM reverse sampler.

    Expected config attributes (set by the subclass config):
        action_dim, action_horizon, num_train_timesteps, beta_start, beta_end,
        beta_schedule, clip_sample, prediction_type, num_inference_steps, input_pertub,
        num_draws.
    """

    @abstractmethod
    def _encode_global_cond(self, batch_dict: dict) -> Tensor:
        """Encode observations into a flat global conditioning vector — ``[B, cond_dim]``."""
        ...

    # ------------------------------------------------------------------
    # Scheduler + dtype plumbing (mirrors the flow base's small helpers).
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_dtype(dtype_str: str) -> torch.dtype:
        if isinstance(dtype_str, torch.dtype):
            if not dtype_str.is_floating_point:
                raise ValueError(f"unsupported floating-point precision {dtype_str}")
            return dtype_str
        dtype_str = dtype_str.replace("torch.", "").replace("bf16", "bfloat16").replace("fp32", "float32")
        dtype = getattr(torch, dtype_str, None)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError(f"unsupported floating-point precision {dtype_str!r}")
        return dtype

    @property
    def dtype(self) -> torch.dtype:
        return next(self.decoder.parameters()).dtype

    def _build_noise_scheduler(self) -> None:
        """Construct the ``diffusers`` DDIM scheduler from ``self.config``.

        Held as a plain attribute (NOT a buffer): it owns only the deterministic noise
        schedule (``betas``/``alphas_cumprod``), fully reproduced from the config, so it
        never needs to ride in the checkpoint. Imported lazily so a flow-only install path
        that never touches diffusion pays nothing.
        """
        from diffusers import DDIMScheduler

        c = self.config
        if c.steps_offset != 0:
            raise ValueError(
                f"diffusion.steps_offset must be 0, got {c.steps_offset!r}; nonzero offsets can produce "
                "out-of-range or negative DDIM timesteps"
            )
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=c.num_train_timesteps,
            beta_start=c.beta_start,
            beta_end=c.beta_end,
            beta_schedule=c.beta_schedule,
            clip_sample=c.clip_sample,
            set_alpha_to_one=c.set_alpha_to_one,
            steps_offset=c.steps_offset,
            prediction_type=c.prediction_type,
        )

    @property
    def num_draws(self) -> int:
        """Independent (noise, timestep) draws per encoder forward (UMI's
        ``train_diffusion_n_samples``). Defaults to 1."""
        return max(1, getattr(self.config, "num_draws", 1))

    def _sample_noise(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def _sample_timesteps(self, count: int, device: torch.device) -> Tensor:
        """One uniform integer DDPM timestep per row — ``[0, num_train_timesteps)``.

        A named method rather than an inline ``torch.randint`` purely so the objective's two
        stochastic inputs (this and :meth:`_sample_noise`) are both PINNABLE by name. The DDP
        gradient-parity test needs ``(epsilon, t)`` to be a function of the GLOBAL row index,
        so that a 2-rank step and a single-process step over the concatenated batch see the
        same draws and any difference in the gradient is attributable to the reduction. With
        the draw inlined, the only way to pin it was to patch ``torch.randint`` globally,
        which also hits the scheduler and the dataloader.
        """
        return torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (count,), device=device
        ).long()

    # ------------------------------------------------------------------
    # Loss masking — copied verbatim from the flow base so the two objectives honor the
    # HL-v2 validity contract identically (the ONLY change is the docstring wording).
    # ------------------------------------------------------------------
    def _compute_diffusion_loss(
        self,
        pred: Tensor,
        target: Tensor,
        action_is_pad: Tensor | None,
        action_dim_mask: Tensor | None = None,
        action_mask: Tensor | None = None,
    ) -> Tensor:
        """Masked MSE between the model output and the scheduler's target.

        Three masks compose multiplicatively (there is no RTC ``prefix_mask`` here — the
        diffusion sampler is plain, see :meth:`sample_actions`):
          - ``action_is_pad`` (B, T): True on temporal-padding timesteps.
          - ``action_dim_mask`` (B, action_dim): False on the F.pad zero-padding added to
            reach ``config.action_dim`` (so unused slots are never supervised toward zero).
          - ``action_mask`` (B, T, action_dim): the HL-v2 per-(step, dim) validity — strictly
            richer than the (T,) + (D,) pair (a step may be supervised for pose while its
            force bin is unmeasured, and a reserved-but-sensorless column is invalid at
            every step).
        """
        sums = squared_error_sums(
            pred,
            target,
            action_is_pad=action_is_pad,
            action_dim_mask=action_dim_mask,
            action_mask=action_mask,
        )["overall"]
        return sums[0] / sums[1].clamp(min=1.0)

    @torch.no_grad()
    def _extra_train_metrics(
        self,
        pred: Tensor,
        target: Tensor,
        action_is_pad: Tensor | None,
        action_dim_mask: Tensor | None,
        action_mask: Tensor | None = None,
    ) -> dict[str, float]:
        """Optional per-group scalar metrics (logging only). Base returns {}."""
        return {}

    # ------------------------------------------------------------------
    # RTC state parity: a diffusion sampler with no RTC still keeps the cache so serving's
    # reset()/episode-boundary calls behave identically to the flow model.
    # ------------------------------------------------------------------
    def _init_rtc_state(self) -> None:
        AH = self.config.action_horizon
        AD = self.config.action_dim
        self.register_buffer("_last_actions", torch.zeros(1, AH, AD), persistent=False)
        self._last_actions_valid: bool = False

    def reset(self) -> None:
        """Reset cached actions at an episode boundary (mirrors the flow base)."""
        self._last_actions_valid = False

    def forward(self, batch_dict: dict) -> dict:
        """Training forward pass with DDPM ε-prediction (or the scheduler's target).

        When ``num_draws > 1`` the encoder runs once and K independent (noise, timestep)
        pairs share the same conditioning. Returns a plain dict:
        ``{"loss": Tensor, "metrics": {"diffusion_loss": float, ...}, "aux": {...}}``.
        """
        actions = batch_dict["actions"].to(self.dtype)
        device = actions.device
        K = self.num_draws

        global_cond = self._encode_global_cond(batch_dict)

        if K > 1:
            global_cond = global_cond.repeat_interleave(K, dim=0)
            actions = actions.repeat_interleave(K, dim=0)

        # HL-v2 per-(step, dim) validity. Fetched before noising because it gates the
        # decoder INPUT (via zeroing below) as well as the loss weight.
        action_mask = batch_dict.get("action_mask")
        if action_mask is not None and K > 1:
            action_mask = action_mask.repeat_interleave(K, dim=0)

        action_is_pad = batch_dict.get("action_is_pad")
        if action_is_pad is not None and K > 1:
            action_is_pad = action_is_pad.repeat_interleave(K, dim=0)

        action_dim_mask = batch_dict.get("action_dim_mask")
        if action_dim_mask is not None and K > 1:
            action_dim_mask = action_dim_mask.repeat_interleave(K, dim=0)

        # Zero the UNSUPERVISED entries before noising — same rationale as the flow base:
        # masking only the loss WEIGHT is not enough, because the noised trajectory is the
        # decoder INPUT and the decoder attends across the whole chunk, so a stale-held or
        # zero-filled target at one step measurably shifts the prediction at another. A
        # zeroed clean action noises to pure scaled noise there, exactly what inference
        # sees, making the loss provably independent of non-measurements. (v2 path only —
        # the ε target is unaffected; masked entries are dropped from the loss regardless.)
        if action_mask is not None:
            actions = actions * action_mask.to(actions.dtype)

        noise = self._sample_noise(actions.shape, device).to(self.dtype)
        # Input perturbation (DDPM-IP): perturb the noise that is ADDED, keep the ORIGINAL
        # noise as the ε target below.
        input_pertub = float(getattr(self.config, "input_pertub", 0.0))
        if input_pertub > 0.0:
            noise_added = noise + input_pertub * self._sample_noise(actions.shape, device).to(self.dtype)
        else:
            noise_added = noise

        timesteps = self._sample_timesteps(actions.shape[0], device)

        noisy = self.noise_scheduler.add_noise(actions, noise_added, timesteps)

        pred = self.decoder(noisy, timesteps, local_cond=None, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = actions
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(actions, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction_type {pred_type!r}")

        loss = self._compute_diffusion_loss(pred, target, action_is_pad, action_dim_mask, action_mask)

        metrics = {}
        if not batch_dict.get("_skip_scalar_metrics", False):
            metrics = {"diffusion_loss": loss.item()}
            metrics.update(self._extra_train_metrics(
                pred, target, action_is_pad, action_dim_mask, action_mask))
        group_defs = self._metric_group_defs() if hasattr(self, "_metric_group_defs") else {}
        with torch.no_grad():
            metric_sums = squared_error_sums(
                pred,
                target,
                action_is_pad=action_is_pad,
                action_dim_mask=action_dim_mask,
                action_mask=action_mask,
                group_dims=group_defs,
            )

        return {
            "loss": loss,
            "metrics": metrics,
            "metric_sums": metric_sums,
            "aux": {"pred": pred, "noise": noise, "timesteps": timesteps},
        }

    @torch.no_grad()
    def sample_actions(
        self,
        batch_dict: dict,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
        policy_smooth_lam: float = 0.0,
        **rtc_kwargs,
    ) -> Tensor:
        """Generate actions via the DDIM reverse process (plain, no RTC prefix).

        ``num_steps`` defaults to ``config.num_inference_steps``. ``rtc_kwargs`` accepts —
        and, being a plain sampler, IGNORES with a hard error on any non-trivial value —
        the flow base's ``steps_executed`` / ``delay_estimate``, so a caller wiring RTC by
        mistake fails loudly instead of silently getting un-chunked actions. Wire DDIM
        frozen-prefix inpainting before enabling RTC here.
        """
        delay = rtc_kwargs.get("delay_estimate", 0)
        delay = delay.item() if isinstance(delay, torch.Tensor) else delay
        if delay:
            raise NotImplementedError(
                "the diffusion sampler is plain (no Real-Time Chunking): got "
                f"delay_estimate={delay}. RTC frozen-prefix inpainting is not implemented "
                "for the diffusion model yet — sample with delay_estimate=0.")

        if not hasattr(self, "_last_actions"):
            self._init_rtc_state()
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        B = batch_dict["images"][0].shape[0]
        device = batch_dict["images"][0].device
        AH = self.config.action_horizon
        AD = self.config.action_dim

        global_cond = self._encode_global_cond(batch_dict)

        # torch.randn requires generator.device == tensor device. A caller doing
        # reproducible sampling naturally builds a CPU torch.Generator() (its default),
        # which would crash against a CUDA model. Relocate the seed onto a device-matched
        # generator so reproducible GPU sampling works instead of raising. Compare the FULL
        # device (not just .type): `device` resolves to a concrete indexed device, so a
        # cuda:0 generator against a cuda:1 model must relocate too — a `.type`-only check
        # would skip that and torch.randn would still raise on the index axis.
        if generator is not None and generator.device != device:
            seed = int(
                torch.randint(
                    0,
                    torch.iinfo(torch.int64).max,
                    (),
                    device=generator.device,
                    generator=generator,
                ).item()
            )
            generator = torch.Generator(device=device).manual_seed(seed)

        x_t = torch.randn((B, AH, AD), dtype=self.dtype, device=device, generator=generator)

        self.noise_scheduler.set_timesteps(num_steps, device=device)
        for t in self.noise_scheduler.timesteps:
            pred = self.decoder(x_t, t, local_cond=None, global_cond=global_cond)
            x_t = self.noise_scheduler.step(pred, t, x_t, generator=generator).prev_sample

        if policy_smooth_lam > 0.0:
            from flore.s1_policy.model._action_smoothing import smooth_chunk_in_place

            x_t = smooth_chunk_in_place(x_t, policy_smooth_lam)

        self._last_actions = x_t.clone()
        self._last_actions_valid = True
        return x_t.float()

    # ------------------------------------------------------------------
    # State conditioning — identical helpers to the flow base, so a subclass can share the
    # zero-pad / learned-projection contract without depending on the flow model.
    # ------------------------------------------------------------------
    def _pad_state(self, state: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        """Zero-pad robot state to filled_state_dim, flattened over the To obs frames.

        Accepts a single-frame ``(B, AD)`` or a history ``(B, To, AD)`` state; each frame is
        right-padded to ``filled_state_dim`` and the To frames concatenated ->
        ``(B, To * filled_state_dim)``. Pads each frame INDEPENDENTLY (never flatten-then-pad,
        which would truncate ``AD * (To - 1)`` real dims once To > 1)."""
        To = max(1, getattr(self.config, "obs_history_len", 1))
        fsd = self.config.filled_state_dim
        if state is None:
            return torch.zeros(batch_size, To * fsd, device=device, dtype=self.dtype)
        state = state.to(self.dtype)
        if state.shape[-1] > fsd:
            raise ValueError(f"state width {state.shape[-1]} exceeds filled_state_dim={fsd}")
        if state.dim() == 2:  # (B, AD) single frame
            return F.pad(state, (0, fsd - state.shape[1]), "constant", 0)
        padded = F.pad(state, (0, fsd - state.shape[2]), "constant", 0)
        return padded.reshape(state.shape[0], -1)

    def _build_state_encoder(self) -> int:
        """Set up state conditioning and return its PER-FRAME width in ``global_cond``.

        Call from the subclass ``__init__`` BEFORE computing ``global_cond_dim``. With
        ``config.state_projection`` True, build a learned
        ``nn.Linear(action_dim, filled_state_dim * state_proj_mult)`` and return its width;
        otherwise leave ``self.state_proj = None`` and return ``config.filled_state_dim``.
        ``state_in_dim`` is ``action_dim`` (the data pipeline right-zero-pads state to the
        model's action_dim), NOT ``config.state_dim``.
        """
        if getattr(self.config, "state_projection", False):
            in_dim = self.config.action_dim
            proj_mult = getattr(self.config, "state_proj_mult", 4)
            out_dim = self.config.filled_state_dim * proj_mult
            self.state_proj = nn.Linear(in_dim, out_dim)
            self._state_proj_in_dim = in_dim
            return out_dim
        self.state_proj = None
        return self.config.filled_state_dim

    def _project_state(self, state: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        """State contribution to ``global_cond``: learned projection or zero-pad.

        Mirrors :meth:`_pad_state` when no projection is configured. When a projection
        exists, the SAME ``nn.Linear`` is applied to each of the To obs frames and the
        results concatenated -> ``(B, To * out_dim)``. The per-frame input is defensively
        padded/truncated to the layer's ``in_features`` so a width drift never breaks the
        matmul.
        """
        if self.state_proj is None:
            return self._pad_state(state, batch_size, device)

        To = max(1, getattr(self.config, "obs_history_len", 1))
        in_dim = self._state_proj_in_dim
        if state is None:
            state = torch.zeros(batch_size, To, in_dim, device=device, dtype=self.dtype)
        else:
            state = state.to(self.dtype)
            if state.dim() == 2:  # (B, AD) -> (B, 1, AD)
                state = state.unsqueeze(1)
            if state.shape[2] != in_dim:
                if state.shape[2] < in_dim:
                    state = F.pad(state, (0, in_dim - state.shape[2]), "constant", 0)
                else:
                    state = state[:, :, :in_dim]
        projected = self.state_proj(state)  # (B, To, out_dim)
        return projected.reshape(projected.shape[0], -1)
