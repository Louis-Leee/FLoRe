"""1D Conditional DiT for flow-matching action prediction.

Ported from GCL (``covariant/models/gam/gcl/policies/networks/_dit``), originally
from FAR-pi/rfmpi/dp_rl. Uses AdaLN conditioning with RoPE/learned pos embeddings.

This is the DEFAULT DP-Flow decoder. Unlike :class:`ConditionalUnet1D`, it accepts a
per-position ``timestep`` of shape ``(B, T)`` (in addition to scalar / ``(B,)``),
which is what Real-Time Chunking (RTC) frozen-prefix inference needs — RTC holds
``time=0`` on the prefix positions and the diffusion timestep on the rest within a
single forward. See ``fuji.s1_policy.model.base.FlowMatchingVLABase.sample_actions``.
"""

from __future__ import annotations

import logging
import math

import einops
import torch
from torch import nn

logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=x.dtype) * -emb)
        emb = x.unsqueeze(-1) * emb
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class RotaryEmbedding1D(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return freqs[None, :, None, :]


def apply_rotary_emb(x, freqs):
    x_rot = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x_rot.unbind(dim=-1)
    freqs_cos = freqs.cos()
    freqs_sin = freqs.sin()
    x_out = torch.stack(
        [
            x1 * freqs_cos - x2 * freqs_sin,
            x1 * freqs_sin + x2 * freqs_cos,
        ],
        dim=-1,
    )
    return x_out.flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, n_heads=8, context_dim=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim or dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim or dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, context=None, rope_emb=None):
        context = x if context is None else context
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b n h d", h=self.n_heads),
            (q, k, v),
        )

        if rope_emb is not None and context is x:
            q = apply_rotary_emb(q, rope_emb)
            k = apply_rotary_emb(k, rope_emb)

        q, k, v = map(
            lambda t: einops.rearrange(t, "b n h d -> b h n d"),
            (q, k, v),
        )

        attn = torch.softmax(torch.einsum("bhid,bhjd->bhij", q, k) * self.scale, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class DiTBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_mult=4):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = Attention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = FeedForward(dim, mlp_mult)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=False),
        )

    def forward(self, x, cond, rope_emb=None):
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=-1)

        x = x + gate_sa * self.self_attn(self.norm1(x) * (1 + scale_sa) + shift_sa, rope_emb=rope_emb)

        x = x + gate_mlp * self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp)

        return x


class ConditionalDiT1D(nn.Module):
    def __init__(
        self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        model_channels=512,
        num_blocks=12,
        n_heads=16,
        mlp_mult=4,
        max_seq_len=64,
        pos_emb_type="rope",
    ):
        super().__init__()
        self.pos_emb_type = pos_emb_type

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, model_channels),
        )

        self.global_cond_encoder = None
        if global_cond_dim is not None:
            self.global_cond_encoder = nn.Sequential(
                nn.SiLU(),
                nn.Linear(global_cond_dim, model_channels, bias=False),
            )

        self.input_proj = nn.Linear(input_dim, model_channels, bias=False)

        self.local_cond_encoder = None
        if local_cond_dim is not None:
            self.local_cond_encoder = nn.Linear(local_cond_dim, model_channels, bias=False)

        if pos_emb_type == "rope":
            head_dim = model_channels // n_heads
            self.rope = RotaryEmbedding1D(head_dim, max_seq_len)
            self.pos_emb = None
        elif pos_emb_type == "learned":
            self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, model_channels) * 0.02)
            self.rope = None
        else:
            raise ValueError(f"pos_emb_type must be 'rope' or 'learned', got {pos_emb_type}")

        self.blocks = nn.ModuleList([DiTBlock(model_channels, n_heads, mlp_mult) for _ in range(num_blocks)])

        self.final_norm = RMSNorm(model_channels)
        self.final_linear = nn.Linear(model_channels, input_dim, bias=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=False),
        )

        logger.info("ConditionalDiT1D params: %e", sum(p.numel() for p in self.parameters()))

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor | float, local_cond=None, global_cond=None, **kwargs
    ):
        """
        sample: (B, T, input_dim)
        timestep: (B,) scalar per-batch OR (B, T) per-position
        local_cond: (B, T, local_cond_dim)
        global_cond: (B, global_cond_dim)
        output: (B, T, input_dim)
        """
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        if timestep.ndim == 1:
            timestep = timestep.expand(sample.shape[0])

        t_emb = self.diffusion_step_encoder(timestep)

        if global_cond is not None and self.global_cond_encoder is not None:
            g_emb = self.global_cond_encoder(global_cond)
            if t_emb.ndim == 3 and g_emb.ndim == 2:
                g_emb = g_emb.unsqueeze(1)
            cond = t_emb + g_emb
        else:
            cond = t_emb

        x = self.input_proj(sample)

        if local_cond is not None and self.local_cond_encoder is not None:
            x = x + self.local_cond_encoder(local_cond)

        if self.pos_emb_type == "rope":
            rope_emb = self.rope(x.shape[1], x.device)
        elif self.pos_emb_type == "learned":
            x = x + self.pos_emb[:, : x.shape[1], :]
            rope_emb = None

        for block in self.blocks:
            x = block(x, cond, rope_emb)

        if cond.ndim == 2:
            cond_3d = cond.unsqueeze(1)
        else:
            cond_3d = cond
        shift, scale = self.final_modulation(cond_3d).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        x = self.final_linear(x)

        return x
