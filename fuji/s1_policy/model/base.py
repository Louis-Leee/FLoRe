"""Flow-matching VLA base — shared training loop and Euler/RTC inference.

Ported from GCL (``covariant/models/gam/gcl/policies/networks/base.py``). Strips the
GCL ``TrainResult`` / metric-collector plumbing: ``forward`` returns a plain dict of
tensors instead. The flow-matching math and the Real-Time Chunking (RTC) masking are
kept bit-for-bit — porting those exactly is the point of this module.

Subclasses implement ``_encode_global_cond()`` to map observations to a flat
conditioning vector; this base handles noise/time sampling, the flow-matching loss
(with temporal-pad + RTC-prefix + action-dim masking), and Euler ODE integration
(with RTC frozen-prefix inpainting) at inference time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fuji.s1_policy.model.loss_metrics import squared_error_sums


def apply_rtc_time_masking(
    time: torch.Tensor, rtc_delay_range: tuple[int, int], action_horizon: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Train-Time RTC (Real-Time Chunking) time masking.

    See https://arxiv.org/abs/2512.05964 for details.

    Parameters
    ----------
    time : torch.Tensor
        Initial time tensor of shape (batch_size,).
    rtc_delay_range : tuple[int, int]
        (min_delay, max_delay) half-open range for prefix length — delay is
        sampled from [min_delay, max_delay) (max_delay exclusive, same as GCL).
        E.g. (0, 5) -> delay ∈ {0, 1, 2, 3, 4}.
    action_horizon : int
        Action horizon length.

    Returns
    -------
    time_masked : torch.Tensor
        Shape (batch_size, action_horizon). Prefix positions have time=0.0.
    prefix_mask : torch.Tensor
        Boolean (batch_size, action_horizon). True = prefix (excluded from loss).
    """
    batch_size = time.shape[0]
    device = time.device
    min_delay, max_delay = rtc_delay_range
    delay = torch.randint(min_delay, max_delay, (batch_size,), device=device)
    prefix_mask = torch.arange(action_horizon, device=device)[None, :] < delay[:, None]
    time_masked = torch.where(prefix_mask, torch.tensor(0.0, device=device), time[:, None])
    return time_masked, prefix_mask


class FlowMatchingVLABase(ABC, nn.Module):
    """Shared flow-matching training loop and Euler ODE inference.

    Subclasses implement ``_encode_global_cond()`` to map observations to a flat
    conditioning vector. This base handles noise sampling, flow-matching loss,
    action masking, and Euler integration at inference time.

    Expected config attributes (set by subclass):
        action_dim, action_horizon, inference_steps, flow_beta_a, flow_beta_b.
    """

    @abstractmethod
    def _encode_global_cond(self, batch_dict: dict) -> Tensor:
        """Encode observations into a flat global conditioning vector.

        Returns:
            [B, global_cond_dim] tensor.
        """
        ...

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

    def _sample_noise(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if generator is not None and generator.device != device:
            generator = torch.Generator(device=device).manual_seed(generator.initial_seed())
        return torch.randn(shape, dtype=torch.float32, device=device, generator=generator)

    def _sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        alpha = torch.as_tensor(self.config.flow_beta_a, dtype=torch.float32, device=device)
        beta = torch.as_tensor(self.config.flow_beta_b, dtype=torch.float32, device=device)
        time_beta = torch.distributions.Beta(alpha, beta).sample((batch_size,))
        return (time_beta * 0.999 + 0.001).to(dtype=torch.float32, device=device)

    def _compute_flow_loss(
        self,
        v_t: Tensor,
        u_t: Tensor,
        action_is_pad: Tensor | None,
        prefix_mask: Tensor | None = None,
        action_dim_mask: Tensor | None = None,
        action_mask: Tensor | None = None,
    ) -> Tensor:
        """MSE loss between predicted and target velocity, with optional masking.

        Four independent masks compose multiplicatively:
          - ``action_is_pad`` (B, T): True on temporal-padding timesteps.
          - ``prefix_mask``  (B, T): True on RTC prefix positions to skip.
          - ``action_dim_mask`` (B, action_dim): False on the F.pad zero-padding
            added to reach ``config.action_dim``. Without it, cotraining a 10D
            single-arm card alongside 20D bimanual ones supervises the unused
            slots toward zero on every single-arm sample, and dilutes the
            real-dim signal by (real_dim / action_dim) on every sample.
          - ``action_mask`` (B, T, action_dim): the HL-v2 per-(step, dim) validity.
            Strictly richer than the (T,) + (D,) pair, which can only express an
            outer product: a step may be supervised for pose while its force bin is
            unmeasured, and a reserved-but-sensorless column is invalid at every step.
        """
        sums = squared_error_sums(
            v_t,
            u_t,
            action_is_pad=action_is_pad,
            prefix_mask=prefix_mask,
            action_dim_mask=action_dim_mask,
            action_mask=action_mask,
        )["overall"]
        return sums[0] / sums[1].clamp(min=1.0)

    @torch.no_grad()
    def _extra_train_metrics(
        self,
        v_t: Tensor,
        u_t: Tensor,
        action_is_pad: Tensor | None,
        prefix_mask: Tensor | None,
        action_dim_mask: Tensor | None,
        action_mask: Tensor | None = None,
    ) -> dict[str, float]:
        """Optional per-group scalar metrics for logging only (no gradient).

        The base class computes nothing extra; a subclass that knows its action-dim
        layout (e.g. keypoint / quantity blocks) can override this to break the
        single ``flow_loss`` into sub-losses. Implementations MUST use the same
        mask convention as :meth:`_compute_flow_loss` — including ``action_mask`` —
        so the sub-losses are directly comparable to ``flow_loss``. Returns an empty
        dict here.
        """
        return {}

    def _init_rtc_state(self) -> None:
        """Initialize RTC inference state. Call from subclass __init__."""
        AH = self.config.action_horizon
        AD = self.config.action_dim
        self.register_buffer("_last_actions", torch.zeros(1, AH, AD), persistent=False)
        self._last_actions_valid: bool = False

    def reset(self) -> None:
        """Reset RTC state. Call at episode boundaries."""
        self._last_actions_valid = False

    @property
    def num_draws(self) -> int:
        """Number of independent (noise, timestep) draws per encoder forward.

        Multi-draw amortizes the cost of an expensive encoder by sampling K
        independent flow-matching targets against the same conditioning.
        Defaults to 1 (standard single-draw) unless config.num_draws is set.
        """
        return getattr(self.config, "num_draws", 1)

    def forward(self, batch_dict: dict) -> dict:
        """Training forward pass with flow matching.

        When num_draws > 1, the encoder runs once and K independent
        (noise, timestep) pairs share the same conditioning vector.
        Supports RTC when config.rtc_delay_range is set.

        Returns a plain dict: ``{"loss": Tensor, "metrics": {"flow_loss": float},
        "aux": {"v_t", "noise", "time"}}``.
        """
        actions = batch_dict["actions"].to(self.dtype)
        B = actions.shape[0]
        device = actions.device
        K = self.num_draws

        global_cond = self._encode_global_cond(batch_dict)

        if K > 1:
            global_cond = global_cond.repeat_interleave(K, dim=0)
            actions = actions.repeat_interleave(K, dim=0)

        # HL-v2 per-(step, dim) validity. Fetched HERE, before the interpolant, because it
        # gates the decoder INPUT as well as the loss weight (see below). Repeat-interleaved
        # like the other masks so the K multi-draw copies of a sample share it.
        action_mask = batch_dict.get("action_mask")
        if action_mask is not None and K > 1:
            action_mask = action_mask.repeat_interleave(K, dim=0)

        noise = self._sample_noise(actions.shape, device).to(self.dtype)
        time = self._sample_time(B * K, device).to(self.dtype)

        rtc_delay_range = getattr(self.config, "rtc_delay_range", None)
        if rtc_delay_range is not None:
            AH = actions.shape[1]
            time_per_pos, prefix_mask = apply_rtc_time_masking(time, rtc_delay_range, AH)
            time_expanded = time_per_pos.unsqueeze(-1)  # (B*K, AH, 1)
            decoder_time = time_per_pos  # (B*K, AH)
        else:
            prefix_mask = None
            time_expanded = time[:, None, None]
            decoder_time = time  # (B*K,)

        # Zero the UNSUPERVISED entries before building the interpolant. Masking only the
        # loss WEIGHT is not enough: `actions` also feeds x_t, which is the decoder's
        # INPUT, and the decoder attends across the whole chunk — so a stale-held or
        # zero-filled target at step 15 measurably shifts the predicted velocity at step
        # 0 (verified: changing only masked entries moved the loss, and v_t on VALID
        # entries by up to 0.24). At inference the chunk is generated from pure noise, so
        # the model never sees a target there anyway; zeroing makes the loss provably
        # independent of values that are, by definition, not measurements.
        #
        # v2 path only (`action_mask`). The legacy action_is_pad/action_dim_mask path is
        # deliberately left byte-identical so the already-deployed checkpoint stays
        # reproducible.
        if action_mask is not None:
            actions = actions * action_mask.to(actions.dtype)

        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        v_t = self.decoder(x_t, decoder_time, local_cond=None, global_cond=global_cond)

        action_is_pad = batch_dict.get("action_is_pad")
        if action_is_pad is not None and K > 1:
            action_is_pad = action_is_pad.repeat_interleave(K, dim=0)

        action_dim_mask = batch_dict.get("action_dim_mask")
        if action_dim_mask is not None and K > 1:
            action_dim_mask = action_dim_mask.repeat_interleave(K, dim=0)

        loss = self._compute_flow_loss(v_t, u_t, action_is_pad, prefix_mask, action_dim_mask,
                                       action_mask)

        # Direct model callers retain the historical scalar metrics API. The Trainer sets
        # this internal flag and aggregates metric_sums on device, avoiding per-step
        # .item()/.tolist() GPU synchronizations between log intervals.
        metrics = {}
        if not batch_dict.get("_skip_scalar_metrics", False):
            metrics = {"flow_loss": loss.item()}
            metrics.update(self._extra_train_metrics(v_t, u_t, action_is_pad, prefix_mask,
                                                      action_dim_mask, action_mask))
        group_defs = self._metric_group_defs() if hasattr(self, "_metric_group_defs") else {}
        with torch.no_grad():
            metric_sums = squared_error_sums(
                v_t,
                u_t,
                action_is_pad=action_is_pad,
                prefix_mask=prefix_mask,
                action_dim_mask=action_dim_mask,
                action_mask=action_mask,
                group_dims=group_defs,
            )

        return {
            "loss": loss,
            "metrics": metrics,
            "metric_sums": metric_sums,
            "aux": {"v_t": v_t, "noise": noise, "time": time},
        }

    @torch.no_grad()
    def sample_actions(
        self,
        batch_dict: dict,
        num_steps: int | None = None,
        steps_executed: int | torch.Tensor = 0,
        delay_estimate: int | torch.Tensor = 0,
        policy_smooth_lam: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate actions via flow-matching Euler integration.

        Supports RTC when ``delay_estimate > 0`` and cached actions are
        available from a previous call. The first ``prefix_len`` rows of the
        returned chunk are the (shifted) cached actions held exactly frozen
        (time=0, dt=0) through every Euler step — the RTC hard-mask invariant.
        """
        if not hasattr(self, "_last_actions"):
            self._init_rtc_state()

        if num_steps is None:
            num_steps = self.config.inference_steps

        B = batch_dict["images"][0].shape[0]
        device = batch_dict["images"][0].device
        AH = self.config.action_horizon
        AD = self.config.action_dim

        global_cond = self._encode_global_cond(batch_dict)

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

        actions_shape = (B, AH, AD)
        noise = self._sample_noise(actions_shape, device, generator=generator).to(self.dtype)

        # RTC setup
        delay_val = delay_estimate.item() if isinstance(delay_estimate, torch.Tensor) else delay_estimate
        use_rtc = getattr(self, "_last_actions_valid", False) and delay_val > 0

        if use_rtc:
            steps_exec = steps_executed.item() if isinstance(steps_executed, torch.Tensor) else steps_executed
            remaining = AH - steps_exec
            prefix_len = min(delay_val, remaining)
            prefix_mask = torch.arange(AH, device=device)[None, :] < prefix_len
            prefix_mask = prefix_mask.expand(B, -1)
            indices = torch.arange(AH, device=device)[None, :, None] + steps_exec
            indices = indices.clamp(0, AH - 1).expand(B, AH, AD)
            last_gathered = torch.gather(self._last_actions.expand(B, -1, -1).to(device), 1, indices)
            x_t = torch.where(prefix_mask.unsqueeze(-1), last_gathered, noise)
        else:
            prefix_mask = None
            x_t = noise

        dt = -1.0 / num_steps
        time_val = 1.0
        for _ in range(num_steps):
            if use_rtc:
                t_per_pos = torch.where(
                    prefix_mask,
                    torch.zeros(1, device=device, dtype=self.dtype),
                    torch.full((1,), time_val, device=device, dtype=self.dtype),
                )
                v_t = self.decoder(x_t, t_per_pos, local_cond=None, global_cond=global_cond)
                dt_mask = torch.where(
                    prefix_mask.unsqueeze(-1),
                    torch.zeros(1, device=device, dtype=self.dtype),
                    torch.tensor(dt, device=device, dtype=self.dtype),
                )
                x_t = x_t + dt_mask * v_t
            else:
                t_tensor = torch.full((B,), time_val, dtype=self.dtype, device=device)
                v_t = self.decoder(x_t, t_tensor, local_cond=None, global_cond=global_cond)
                x_t = x_t + dt * v_t
            time_val += dt

        if policy_smooth_lam > 0.0:
            from fuji.s1_policy.model._action_smoothing import smooth_chunk_in_place

            if use_rtc:
                # Smoothing the full chunk would MOVE the frozen prefix rows and
                # break the RTC hard-mask invariant (prefix must stay bit-exact
                # equal to the shifted previous chunk — it is already being
                # executed on the robot). Smooth, then restore the prefix rows.
                # The suffix still blends toward the (unsmoothed) prefix values,
                # which is the correct boundary condition: the executed actions
                # are immutable facts.
                prefix_vals = torch.where(prefix_mask.unsqueeze(-1), x_t, torch.zeros_like(x_t))
                x_t = smooth_chunk_in_place(x_t, policy_smooth_lam)
                x_t = torch.where(prefix_mask.unsqueeze(-1), prefix_vals, x_t)
            else:
                x_t = smooth_chunk_in_place(x_t, policy_smooth_lam)

        self._last_actions = x_t.clone()
        self._last_actions_valid = True
        return x_t.float()

    def _pad_state(self, state: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        """Zero-pad robot state to filled_state_dim, flattened over the To obs frames.

        Accepts either a single-frame ``(B, AD)`` state or a history ``(B, To, AD)``
        state. Each frame is right-padded to ``filled_state_dim`` and the To frames
        are concatenated -> ``(B, To * filled_state_dim)``, matching the To-scaled
        ``state_cond_width`` the model sized ``global_cond`` with. NOTE: this pads each
        frame independently — never flatten-then-pad (that would silently truncate
        ``AD * (To - 1)`` real dims once To > 1)."""
        To = max(1, getattr(self.config, "obs_history_len", 1))
        fsd = self.config.filled_state_dim
        if state is None:
            return torch.zeros(batch_size, To * fsd, device=device, dtype=self.dtype)
        state = state.to(self.dtype)
        if state.shape[-1] > fsd:
            raise ValueError(f"state width {state.shape[-1]} exceeds filled_state_dim={fsd}")
        if state.dim() == 2:  # (B, AD) single frame
            return F.pad(state, (0, fsd - state.shape[1]), "constant", 0)
        # (B, To, AD) -> pad each frame's AD to fsd -> flatten to (B, To*fsd)
        padded = F.pad(state, (0, fsd - state.shape[2]), "constant", 0)
        return padded.reshape(state.shape[0], -1)

    # ------------------------------------------------------------------
    # State conditioning: either zero-pad (default) or a learned projection.
    # ------------------------------------------------------------------
    def _build_state_encoder(self) -> int:
        """Set up state conditioning and return its width in ``global_cond``.

        Call from a subclass ``__init__`` BEFORE computing ``global_cond_dim``.

        When ``config.state_projection`` is True, build a learned
        ``nn.Linear(action_dim, filled_state_dim * state_proj_mult)`` (state gets
        its own embedding instead of sitting in a zero-padded slot) and return the
        projected width. Otherwise leave ``self.state_proj = None`` and return
        ``config.filled_state_dim`` (the unchanged zero-pad path).

        ``state_in_dim`` is the runtime state width the model actually receives,
        which is ``action_dim`` (the data pipeline right-zero-pads state to the
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

        Mirrors ``_pad_state`` when no projection is configured. When a projection
        exists, the SAME ``nn.Linear`` is applied to each of the To obs frames and the
        results concatenated -> ``(B, To * out_dim)``. Accepts a single-frame
        ``(B, AD)`` or a history ``(B, To, AD)`` state; a zero vector of the expected
        width is used when ``state`` is None. The per-frame input is defensively
        padded/truncated to the layer's ``in_features`` so a width drift never breaks
        the matmul.
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
        # Apply the shared projection to every frame, then flatten the To frames.
        projected = self.state_proj(state)  # (B, To, out_dim)
        return projected.reshape(projected.shape[0], -1)
