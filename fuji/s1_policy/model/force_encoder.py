"""Causal force-history encoder for the HL-v2 policy.

The observation carries a short wrench trajectory — ``(L, C)`` with ``L`` causal
anti-aliased bins over the second preceding ``t0`` (``L=8``, ``C=6`` per F/T-equipped
hand). This module turns it into a fixed-width vector for ``global_cond``.

Why a small conv stack rather than a flatten+Linear:

* **The temporal profile IS the signal.** In a contact task, "3 N and rising fast"
  and "3 N and releasing" demand opposite actions, and they are identical under any
  reduction that ignores order. A conv over the time axis sees the slope and the
  curvature; a flatten+Linear can in principle learn them but has to rediscover
  locality from 48 unstructured inputs.
* **It keeps ``state`` small.** Flattening a two-hand history into ``state`` would be
  96 dims — larger than ``filled_state_dim`` (64), where ``_pad_state`` would silently
  truncate it via a negative ``F.pad`` width.
* **Causal padding** means the encoder can be evaluated identically online, on a
  buffer that only ever grows to the left.

Validity gating
---------------
``force_history_valid`` is a per-sample bool from the dataset: False when the window
was not actually measured (sensor gap, tare transient, or an anchor too close to the
episode start, where the converter's bins hold a stale value). An invalid window must
NOT be presented as a measurement of zero force — a free-space-tared wrench is ≈0 by
construction, so "no contact" and "no data" are numerically indistinguishable and the
model would learn to trust a stale hold. Instead, an invalid sample's encoding is
replaced by a LEARNED "no force data" embedding, the same device UVA-style models use
for absent modalities. The model then knows the difference.

Two sensors: ``groups``
-----------------------
``groups=n`` builds ``n`` INDEPENDENT copies of the stack above, each consuming its own
``channels // n`` slice of the window and each with its own ``invalid_embed``, and
concatenates their outputs (so ``global_cond`` receives ``n * out_dim``). This is how the
3-tracker rig's two Botas enter the model, and it is not merely an implementation
convenience:

* **Per-hand validity is only expressible this way.** The first conv mixes channels, so
  after one shared stack there is no per-hand quantity left to gate — one hand's sensor
  gap would have to blank both hands' encoding. Measured on ``0830_1411_clean32`` this is
  currently worth **+0.17%** of windows (left valid 0.9700, right 0.9733, AND 0.9700): the
  two Botas tare on the same keypress and did not drop independently there. It is kept for
  the whole-take failure — one of six bus devices dropping out — not for that number.
* **The two wrenches are in DIFFERENT frames.** Each is raw in its own Bota sensor frame
  (``attrs.hl_force_frame == "bota_sensor"``), and nothing in the window says which frame
  a channel belongs to. A shared first layer would have to learn the cross-frame relation
  from unlabelled channel position; separate stacks let the mixing happen in the decoder's
  FiLM/AdaLN conditioning, where vision and state are also available.
* **The left branch stays structurally identical to the shipped single-hand encoder**, so a
  ``left_only`` checkpoint's force weights are a legitimate initialization for group 0.

Group 0 lives DIRECTLY on this module under the historical names (``conv.*``,
``readout.*``, ``invalid_embed``) and groups 1.. live in ``extra_groups``. So ``groups=1``
has a bit-identical ``state_dict`` to every shipped checkpoint, and in a two-hand model the
LEFT hand's weights are still named exactly what a ``left_only`` checkpoint calls them.
(Attaching group 0 through a wrapper submodule AND aliasing the names would register both
paths — duplicate keys with shared storage, which ``safetensors`` refuses to save.)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _build_conv(channels: int, width: int) -> nn.Sequential:
    """Two causal conv layers (left-pad by kernel-1, no right context).

    Kernel 3 twice gives a receptive field of 5 bins = 625 ms of the v2 1 s window.
    """
    return nn.Sequential(
        nn.ConstantPad1d((2, 0), 0.0),
        nn.Conv1d(channels, width, kernel_size=3),
        nn.GELU(),
        nn.ConstantPad1d((2, 0), 0.0),
        nn.Conv1d(width, width, kernel_size=3),
        nn.GELU(),
    )


def _encode_group(conv: nn.Module, readout: nn.Module, invalid_embed: Tensor,
                  window: Tensor, valid: Tensor | None) -> Tensor:
    """``(B, L, c)`` -> ``(B, out_dim)`` for ONE sensor. ``valid`` is ``(B,)`` or None.

    A free function rather than a method so group 0 (attached directly to
    :class:`ForceHistoryEncoder` for checkpoint-name compatibility) and groups 1..
    (:class:`_ForceGroup` submodules) run byte-identical code.
    """
    x = window.to(conv[1].weight.dtype).transpose(1, 2)   # (B, c, L)
    h = conv(x)                                           # (B, W, L)
    # Readout concatenates the final timestep — the bin ending at t0 — with a mean over
    # time, so both the instantaneous state and the window average reach global_cond.
    feat = torch.cat([h[:, :, -1], h.mean(dim=2)], dim=1)  # (B, 2W)
    out = readout(feat)                                    # (B, out_dim)
    if valid is not None:
        # Substitute the learned "no data" embedding wherever the window was not
        # measured. Done AFTER the conv (rather than by zeroing the input) so a
        # stale-hold window cannot masquerade as a real zero-force measurement.
        keep = valid.reshape(-1, 1).to(out.dtype)
        out = keep * out + (1.0 - keep) * invalid_embed.to(out.dtype).expand(out.shape[0], -1)
    return out


class _ForceGroup(nn.Module):
    """One NON-PRIMARY sensor's stack. Group 0 lives on the parent (see the docstring)."""

    def __init__(self, channels: int, width: int, out_dim: int) -> None:
        super().__init__()
        self.conv = _build_conv(channels, width)
        self.readout = nn.Linear(2 * width, out_dim)
        # Learned stand-in for "this window was not measured". Zero-init so an
        # untrained model starts out treating an invalid window as a neutral
        # contribution rather than injecting a random bias.
        self.invalid_embed = nn.Parameter(torch.zeros(out_dim))

    def forward(self, window: Tensor, valid: Tensor | None) -> Tensor:
        return _encode_group(self.conv, self.readout, self.invalid_embed, window, valid)


class ForceHistoryEncoder(nn.Module):
    """``(B, L, C)`` causal wrench window -> ``(B, groups * out_dim)``.

    Parameters
    ----------
    hist_len : int
        L, the number of time bins (8 in the HL-v2 contract, 1 in v3).
    channels : int
        C, wrench channels = 6 per F/T-equipped hand.
    width : int
        Hidden width of each conv stack.
    out_dim : int
        Output width contributed to ``global_cond`` PER GROUP.
    groups : int
        Number of independent sensors packed into ``channels`` (default 1). Must divide
        ``channels``; group ``i`` reads ``channels[i*c : (i+1)*c]`` and is gated by
        ``valid[:, i]``. See the module docstring for why this is not one wide conv.
    """

    def __init__(self, hist_len: int, channels: int, width: int = 128, out_dim: int = 64,
                 groups: int = 1) -> None:
        super().__init__()
        if hist_len < 1 or channels < 1:
            raise ValueError(f"ForceHistoryEncoder needs hist_len>=1 and channels>=1, "
                             f"got {hist_len} and {channels}")
        groups = int(groups)
        if groups < 1:
            raise ValueError(f"ForceHistoryEncoder groups must be >= 1, got {groups}")
        if channels % groups:
            raise ValueError(
                f"ForceHistoryEncoder channels={channels} is not divisible by "
                f"groups={groups}; each sensor contributes an equal 6-DoF block, so an "
                f"uneven split would silently route half of one hand's wrench into the "
                f"other hand's encoder.")
        self.hist_len = int(hist_len)
        self.channels = int(channels)
        self.out_dim = int(out_dim)
        self.groups = groups
        #: Channels per group, i.e. per F/T sensor (6 in practice).
        self.group_channels = self.channels // groups
        #: Total width this module contributes to ``global_cond``.
        self.total_out_dim = self.out_dim * groups

        # GROUP 0 on this module under the historical names, so a one-group encoder's
        # state_dict is bit-identical to every shipped checkpoint's and a two-hand model's
        # LEFT branch is still named what a left_only checkpoint calls it.
        self.conv = _build_conv(self.group_channels, width)
        self.readout = nn.Linear(2 * width, self.out_dim)
        self.invalid_embed = nn.Parameter(torch.zeros(self.out_dim))
        # Groups 1.. Empty ModuleList at groups==1 contributes no state_dict entries.
        self.extra_groups = nn.ModuleList(
            _ForceGroup(self.group_channels, width, self.out_dim) for _ in range(groups - 1))

    def forward(self, force_history: Tensor, valid: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        force_history : (B, L, C) — oldest bin first, newest (ending at t0) last.
        valid : (B,) or (B, groups) bool, or None to treat every sample as measured.
            A ``(B,)`` flag with ``groups > 1`` is broadcast to every group (that is the
            conservative reading: "the whole window is/isn't a measurement").

        Returns
        -------
        (B, groups * out_dim)
        """
        if force_history.dim() != 3:
            raise ValueError(f"force_history must be (B, L, C), got {tuple(force_history.shape)}")
        b, length, ch = force_history.shape
        if length != self.hist_len or ch != self.channels:
            raise ValueError(
                f"force_history is (B, {length}, {ch}) but this encoder was built for "
                f"(B, {self.hist_len}, {self.channels}). A window/channel mismatch "
                f"mis-times or mis-routes every force input.")
        if valid is not None:
            if valid.dim() == 1:
                per_group = [valid] * self.groups
            elif valid.dim() == 2 and valid.shape[1] == self.groups:
                per_group = [valid[:, i] for i in range(self.groups)]
            else:
                raise ValueError(
                    f"force_history_valid must be (B,) or (B, {self.groups}), got "
                    f"{tuple(valid.shape)}. A width mismatch here would gate the wrong "
                    f"sensor — refused rather than broadcast.")
        else:
            per_group = [None] * self.groups

        c = self.group_channels
        outs = [_encode_group(self.conv, self.readout, self.invalid_embed,
                              force_history[:, :, :c], per_group[0])]
        for i, g in enumerate(self.extra_groups, start=1):
            outs.append(g(force_history[:, :, i * c:(i + 1) * c], per_group[i]))
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
