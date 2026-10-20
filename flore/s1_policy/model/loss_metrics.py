"""Exact masked error sums shared by the Flow, Diffusion and ACT objectives.

:func:`squared_error_sums` (order 2) is what Flow and Diffusion report; ACT regresses
actions under an **L1** objective, so it reports :func:`absolute_error_sums` (order 1)
instead. Both delegate to one masking implementation, because the masks — not the
error function — are the load-bearing part: a group's reported number must be the SAME
numerator/denominator ratio the optimizer sees, restricted to that group's dims.
"""

from __future__ import annotations

import torch
from torch import Tensor


def masked_error_sums(
    pred: Tensor,
    target: Tensor,
    *,
    order: int = 2,
    action_is_pad: Tensor | None = None,
    prefix_mask: Tensor | None = None,
    action_dim_mask: Tensor | None = None,
    action_mask: Tensor | None = None,
    group_dims: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Return un-clamped ``(sum error, valid element count)`` per group.

    ``order`` selects the elementwise error: 2 = squared (Flow/Diffusion, both of which
    regress a Gaussian-ish target under MSE), 1 = absolute (ACT's L1 reconstruction).
    Nothing else differs — in particular the mask composition is identical, so an
    ``order=1`` "overall" sum is exactly the masked mean ``|pred - target|`` that ACT's
    ``l1`` loss term optimizes.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 (absolute) or 2 (squared), got {order!r}")
    include = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
    if action_is_pad is not None:
        include &= ~action_is_pad
    if prefix_mask is not None:
        include &= ~prefix_mask
    weight = include.unsqueeze(-1).to(torch.float32).expand_as(pred)
    if action_dim_mask is not None:
        weight = weight * action_dim_mask.unsqueeze(1).to(weight.dtype)
    if action_mask is not None:
        weight = weight * action_mask.to(weight.dtype)

    residual = (pred - target).to(torch.float32)
    sq_err = residual.square() if order == 2 else residual.abs()
    per_dim_num = (sq_err * weight).sum(dim=(0, 1))
    per_dim_den = weight.sum(dim=(0, 1))
    width = pred.shape[-1]

    groups = dict(group_dims or {})
    groups = {name: tuple(d for d in dims if d < width) for name, dims in groups.items()}
    force_dims = set(groups.get("force", ()))
    groups["overall"] = tuple(range(width))
    groups["action_nonforce"] = tuple(d for d in range(width) if d not in force_dims)

    out: dict[str, tuple[Tensor, Tensor]] = {}
    for name, dims in groups.items():
        if dims:
            index = torch.as_tensor(dims, dtype=torch.long, device=pred.device)
            out[name] = (per_dim_num[index].sum(), per_dim_den[index].sum())
        else:
            zero = per_dim_num.new_zeros(())
            out[name] = (zero, zero.clone())
    return out


def squared_error_sums(pred: Tensor, target: Tensor, **kwargs) -> dict[str, tuple[Tensor, Tensor]]:
    """``masked_error_sums`` with ``order=2``. The Flow/Diffusion entry point.

    Kept as a named wrapper rather than replaced by callers passing ``order=2``: it is the
    spelling both already-trained objectives use, and a silent signature change on a
    shared loss primitive is exactly the kind of edit that makes a deployed checkpoint's
    metrics quietly incomparable to its own history.
    """
    return masked_error_sums(pred, target, order=2, **kwargs)


def absolute_error_sums(pred: Tensor, target: Tensor, **kwargs) -> dict[str, tuple[Tensor, Tensor]]:
    """``masked_error_sums`` with ``order=1``. ACT's L1 entry point.

    ACT's objective is ``L1 + kl_weight * KL`` (Zhao et al. 2023); the reported
    ``act_loss/<group>`` numbers are this L1 term restricted to a group's dims, so they
    are directly comparable to the optimized reconstruction loss. NOTE that an ACT
    ``overall`` is therefore NOT comparable to a flow/diffusion ``overall``: different
    order, and different regression target (actions vs velocity/epsilon).
    """
    return masked_error_sums(pred, target, order=1, **kwargs)
