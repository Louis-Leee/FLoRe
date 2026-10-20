"""Aggregation and physical trajectory metrics for S1 training diagnostics."""

from __future__ import annotations

import math

import numpy as np
import torch

from flore.s1_policy.data.transforms import GRIPPER_WIDTH_REF_M
from flore.utils.rot6d import rot6d_to_rotmat


class MetricAccumulator:
    """Accumulate exact numerators/denominators without averaging batch means."""

    def __init__(self) -> None:
        self.sums: dict[str, list[float]] = {}

    def add(self, values: dict[str, tuple[torch.Tensor | float, torch.Tensor | float]]) -> None:
        for name, (num, den) in values.items():
            n = float(num.detach().cpu()) if torch.is_tensor(num) else float(num)
            d = float(den.detach().cpu()) if torch.is_tensor(den) else float(den)
            current = self.sums.setdefault(name, [0.0, 0.0])
            current[0] += n
            current[1] += d

    def means(self) -> dict[str, float]:
        return {
            name: (num / den if den > 0 else math.nan)
            for name, (num, den) in self.sums.items()
        }


def physical_trajectory_sums(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    layout,
) -> dict[str, tuple[float, float]]:
    """Mask-aware sums/counts for deploy-space trajectory errors.

    Inputs must already be de-normalized to the dataset action convention: metres,
    rot6d, normalized gripper opening, newtons, and newton-metres — or, for the
    ``wholebody`` layout, radians in place of the pose columns.

    Which keys come back follows the LAYOUT, not a fixed list: a keypoint mode reports
    ``position_cm``/``rotation_geodesic_deg`` and a whole-body mode reports ``joint_deg``
    instead. ⚠️ The pose keys are OMITTED rather than reported as zero for a mode that has no
    pose columns, and that is the whole point of the guard below: ``mask[..., []].all(axis=-1)``
    is ``True`` everywhere and a norm over 0 columns is exactly 0.0, so the arithmetic would
    have logged a PERFECT 0.00 cm position error for a policy that never predicted a position.
    """
    pred = np.asarray(pred.detach().cpu() if torch.is_tensor(pred) else pred, dtype=np.float64)
    target = np.asarray(target.detach().cpu() if torch.is_tensor(target) else target, dtype=np.float64)
    mask = np.asarray(mask.detach().cpu() if torch.is_tensor(mask) else mask, dtype=bool)
    out = {
        "gripper_mm": [0.0, 0.0],
        "force_n": [0.0, 0.0],
        "torque_nm": [0.0, 0.0],
    }
    joint_dims = np.asarray(layout.joint_dims(), dtype=np.int64)
    if joint_dims.size:
        out["joint_deg"] = [0.0, 0.0]
    else:
        out["position_cm"] = [0.0, 0.0]
        out["rotation_geodesic_deg"] = [0.0, 0.0]

    def add(name: str, values: np.ndarray, valid: np.ndarray) -> None:
        if valid.any():
            out[name][0] += float(values[valid].sum())
            out[name][1] += float(valid.sum())

    if joint_dims.size:
        # PER JOINT, not a norm over all 29: the mean absolute joint error in degrees is the
        # quantity the low level's 0.20 rad (11.5 deg) pre-arm gate and its 0.0833 rad/step
        # slew budget are expressed in, so it reads directly against them. A 29-dim L2 norm
        # would grow with the joint count and compare to nothing.
        add("joint_deg",
            np.degrees(np.abs(pred[..., joint_dims] - target[..., joint_dims])).mean(axis=-1),
            mask[..., joint_dims].all(axis=-1))

    for role in layout.roles:
        slots = layout.slots(role)
        if slots.pos.stop > slots.pos.start:
            pos_dims = np.arange(slots.pos.start, slots.pos.stop)
            valid = mask[..., pos_dims].all(axis=-1)
            add("position_cm", np.linalg.norm(pred[..., pos_dims] - target[..., pos_dims], axis=-1) * 100.0, valid)

        if slots.rot6d.stop > slots.rot6d.start:
            rot_dims = np.arange(slots.rot6d.start, slots.rot6d.stop)
            valid = mask[..., rot_dims].all(axis=-1)
            rp = rot6d_to_rotmat(pred[..., rot_dims])
            rt = rot6d_to_rotmat(target[..., rot_dims])
            rel = np.einsum("...ij,...kj->...ik", rp, rt)
            cosine = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
            add("rotation_geodesic_deg", np.degrees(np.arccos(cosine)), valid)

        if slots.gripper is not None:
            d = slots.gripper
            add("gripper_mm", np.abs(pred[..., d] - target[..., d]) * GRIPPER_WIDTH_REF_M * 1000.0,
                mask[..., d])

        if slots.force is not None:
            force_dims = np.arange(slots.force.start, slots.force.start + 3)
            torque_dims = np.arange(slots.force.start + 3, slots.force.stop)
            f_valid = mask[..., force_dims].all(axis=-1)
            t_valid = mask[..., torque_dims].all(axis=-1)
            add("force_n", np.linalg.norm(pred[..., force_dims] - target[..., force_dims], axis=-1), f_valid)
            add("torque_nm", np.linalg.norm(pred[..., torque_dims] - target[..., torque_dims], axis=-1), t_valid)
    return {name: (vals[0], vals[1]) for name, vals in out.items()}
