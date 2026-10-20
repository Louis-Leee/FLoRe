"""In-place penalized smoothing for the final action chunk.

Vendored from GCL (``policies/networks/_paligemma/_action_smoothing.py``). Used by
``FlowMatchingVLABase.sample_actions`` when ``policy_smooth_lam > 0``. The smoother
runs after the Euler loop, just before ``self._last_actions`` is written, so the
chunk returned to the caller AND the RTC cache both hold the smoothed values.

Implementation
--------------
Whittaker-Henderson smoothing with a 2nd-difference penalty:

    minimize  ||y - x||^2 + lam * ||D2 x||^2
    closed form:  x = (I + lam * D2^T D2)^-1 y

with D2 the (H-2, H) second-difference matrix. The system matrix is symmetric
positive definite and pentadiagonal, tiny for typical H~50, solved densely with
``torch.linalg.solve`` (torch-native, no scipy, no graph break on the hot path).

Only the xyz + rot6d columns (the first 9 of the action vector) are smoothed;
gripper (column 9) and any padding pass through untouched so binary open/close
transitions stay sharp.
"""

from __future__ import annotations

import torch
from torch import Tensor

_POSE_COLS_END = 9  # xyz (0:3) + rot6d (3:9)


def smooth_chunk_in_place(chunk: Tensor, lam: float) -> Tensor:
    """Return a smoothed copy of ``chunk`` (xyz + rot6d columns only).

    Parameters
    ----------
    chunk : Tensor
        Action chunk of shape ``(B, H, action_dim)`` in NORMALIZED model space.
        Any dtype/device is accepted; the smoother solves in float32 and casts
        back to the original dtype on the same device.
    lam : float
        Whittaker-Henderson second-difference penalty. ``0`` returns the chunk
        unchanged. Typical 0.05-1.0; higher = smoother.
    """
    if lam <= 0.0:
        return chunk
    if chunk.dim() != 3:
        return chunk
    B, H, D = chunk.shape
    if H < 4 or D < 1:
        return chunk

    cols_end = min(_POSE_COLS_END, D)
    if cols_end <= 0:
        return chunk

    device = chunk.device
    in_dtype = chunk.dtype

    # Build I + lam * D2^T D2 once. Penalty matrix only depends on H.
    eye = torch.eye(H, device=device, dtype=torch.float32)
    d2 = eye[:-2] - 2.0 * torch.roll(eye, -1, dims=0)[:-2] + torch.roll(eye, -2, dims=0)[:-2]
    a_mat = eye + float(lam) * (d2.T @ d2)  # (H, H), SPD pentadiagonal

    pose = chunk[..., :cols_end].to(torch.float32)  # (B, H, cols_end)
    # Reshape pose to (H, B * cols_end) so torch.linalg.solve treats every
    # (B, axis) pair as an independent RHS column.
    rhs = pose.permute(1, 0, 2).reshape(H, -1)  # (H, B * cols_end)
    sol = torch.linalg.solve(a_mat, rhs)  # (H, B * cols_end)
    pose_smoothed = sol.reshape(H, B, cols_end).permute(1, 0, 2)  # (B, H, cols_end)

    out = chunk.clone()
    out[..., :cols_end] = pose_smoothed.to(in_dtype)
    return out
