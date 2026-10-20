"""Data transforms for the DP-Flow S1 policy: rotation lift, absolute->relative
EE conversion, and gripper normalization.

Rotation conventions come from ``flore.utils.rot6d`` (the single source of truth for
the whole repo) — rot6d here is the FIRST-TWO-ROWS flavor, axis-angle (rotvec) is the
on-disk UMI zarr rotation. We do NOT duplicate that math.

Absolute->relative EE conversion copies GCL's ``ActionType.RELATIVE`` SE(3) semantics
(``covariant/models/gam/gcl/data/transforms.py:to_model_actions``): each future pose is
expressed in the body frame of a reference (current-EE) pose::

    R_rel = R_ref^T @ R_a
    p_rel = R_ref^T @ (p_a - p_ref)

DEVIATION from GCL: GCL's ``to_model_actions`` takes a flat action vector and slices
paired xyz/rot6d blocks by config. Here the caller passes pos and rot6d arrays
explicitly (cleaner for the fixed UMI 3-point layout); the underlying per-pose math is
identical. ``delta_ee_to_absolute`` is the exact inverse (used by serving / tests).
"""

from __future__ import annotations

import numpy as np

from flore.utils.rot6d import (
    rot6d_to_rotmat,
    rotmat_to_rot6d,
    rotmat_to_rotvec,
    rotvec_to_rotmat,
)

# Reference full-open gripper width in metres; width_m / this -> [0, 1].
GRIPPER_WIDTH_REF_M = 0.095


# ---------------------------------------------------------------------------
# Rotation lift: axis-angle (on-disk) <-> rot6d (model space)
# ---------------------------------------------------------------------------
def rotvec_to_rot6d(rotvec: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 6) rot6d (first-two-rows)."""
    return rotmat_to_rot6d(rotvec_to_rotmat(np.asarray(rotvec, dtype=np.float64)))


def rot6d_to_rotvec(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) rot6d -> (..., 3) axis-angle."""
    return rotmat_to_rotvec(rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64)))


def rot6d_relative(rot6d: np.ndarray, ref_rot6d: np.ndarray) -> np.ndarray:
    """Orientation of ``rot6d`` expressed in the frame of ``ref_rot6d``: ``R_ref^T R``.

    The ROTATION-ONLY half of :func:`absolute_to_delta_ee` — no translation. Used for
    the ``wrt_start`` observation channel (a keypoint's current orientation relative to
    its episode-start orientation), mirroring UMI-on-Air's
    ``robot0_eef_rot_axis_angle_wrt_start``. Frame-invariant to the world origin, so it
    is safe to feed the model where absolute pose would be out-of-distribution.

    Parameters
    ----------
    rot6d : (..., 6) current rotation (rot6d).
    ref_rot6d : (6,) or broadcastable reference rotation (episode-start).
    """
    r = rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64))          # (..., 3, 3)
    r_ref = rot6d_to_rotmat(np.asarray(ref_rot6d, dtype=np.float64))  # (3, 3) typically
    r_rel = np.einsum("ji,...jk->...ik", r_ref, r)                    # R_ref^T @ R
    return rotmat_to_rot6d(r_rel)


# ---------------------------------------------------------------------------
# Absolute EE trajectory -> relative-to-reference (GCL RELATIVE SE(3) math)
# ---------------------------------------------------------------------------
def absolute_to_delta_ee(
    pos: np.ndarray,
    rot6d: np.ndarray,
    ref_pos: np.ndarray,
    ref_rot6d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express absolute EE poses in the body frame of a reference pose.

    Copies GCL ``to_model_actions`` RELATIVE math per pose:
        ``R_rel = R_ref^T @ R_a``  and  ``p_rel = R_ref^T @ (p_a - p_ref)``.

    Parameters
    ----------
    pos : (..., 3) absolute positions.
    rot6d : (..., 6) absolute rotations (rot6d).
    ref_pos : (3,) or broadcastable reference position (current EE).
    ref_rot6d : (6,) or broadcastable reference rotation.

    Returns
    -------
    (rel_pos, rel_rot6d) : each same leading shape as ``pos`` / ``rot6d``.
    """
    pos = np.asarray(pos, dtype=np.float64)
    rot6d = np.asarray(rot6d, dtype=np.float64)
    r_a = rot6d_to_rotmat(rot6d)  # (..., 3, 3)
    r_ref = rot6d_to_rotmat(np.asarray(ref_rot6d, dtype=np.float64))  # (3, 3) typically
    ref_pos = np.asarray(ref_pos, dtype=np.float64)

    # R_rel = R_ref^T @ R_a
    r_rel = np.einsum("ji,...jk->...ik", r_ref, r_a)
    rel_rot6d = rotmat_to_rot6d(r_rel)

    # p_rel = R_ref^T @ (p_a - p_ref)
    dp = pos - ref_pos
    rel_pos = np.einsum("ji,...j->...i", r_ref, dp)
    return rel_pos.astype(np.float64), rel_rot6d.astype(np.float64)


def delta_ee_to_absolute(
    rel_pos: np.ndarray,
    rel_rot6d: np.ndarray,
    ref_pos: np.ndarray,
    ref_rot6d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`absolute_to_delta_ee`.

    ``R_a = R_ref @ R_rel``  and  ``p_a = R_ref @ p_rel + p_ref``.
    """
    rel_pos = np.asarray(rel_pos, dtype=np.float64)
    rel_rot6d = np.asarray(rel_rot6d, dtype=np.float64)
    r_rel = rot6d_to_rotmat(rel_rot6d)
    r_ref = rot6d_to_rotmat(np.asarray(ref_rot6d, dtype=np.float64))
    ref_pos = np.asarray(ref_pos, dtype=np.float64)

    r_a = np.einsum("ij,...jk->...ik", r_ref, r_rel)
    abs_rot6d = rotmat_to_rot6d(r_a)
    abs_pos = np.einsum("ij,...j->...i", r_ref, rel_pos) + ref_pos
    return abs_pos.astype(np.float64), abs_rot6d.astype(np.float64)


# ---------------------------------------------------------------------------
# Shared reference-frame resolution (HoMMI-style gripper-centric frame)
# ---------------------------------------------------------------------------
# rot6d of the identity rotation (first two rows of I3). rot6d_to_rotmat of this
# is exactly I3 (verified), so an identity base is a no-op reference pose.
IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)

# 32-dim state/action layout, kept in sync with flore.s1_policy.data.umi_dataset:
#   left [0:3]pos + [3:9]rot6d, right [10:13]+[13:19], head [20:23]+[23:29].
_REF_FRAME_SLOTS = {
    "left": (slice(0, 3), slice(3, 9)),
    "right": (slice(10, 13), slice(13, 19)),
    "head": (slice(20, 23), slice(23, 29)),
}
REF_FRAME_CHOICES = ("per_keypoint", "left", "right", "head", "world")


def resolve_base(state32: np.ndarray, ref_frame: str) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the shared reference base pose ``(base_pos[3], base_rot6d[6])`` for a
    32-dim ABSOLUTE anchor ``state`` under a given ``ref_frame``.

    ``left``/``right``/``head`` read that keypoint's absolute anchor pose out of the
    fixed 32-dim slot layout — one shared SE(3) frame for every keypoint (HoMMI's
    gripper-centric frame). ``world`` returns the identity base (pos 0, rot6d
    identity), i.e. keep absolute world poses. ``per_keypoint`` has NO single shared
    base (each keypoint is relative to its own anchor) and MUST be handled by the
    caller, so requesting a base for it is a programming error.

    TRAIN-SIDE ONLY today: this is used by the training dataset
    (``UMIDataset.__getitem__``). Serving (``serving/server.py::_chunk_to_absolute``)
    still un-relativizes every keypoint against its OWN anchor (i.e. hardcoded
    ``per_keypoint``) and does NOT yet call this helper. A model trained with
    ``ref_frame != per_keypoint`` therefore CANNOT be correctly deployed until serving
    resolves the SAME shared base here — otherwise the absolute reconstruction is
    silently wrong. Keep this helper the single source of truth when serving is wired.
    """
    if ref_frame == "world":
        return np.zeros(3, dtype=np.float64), IDENTITY_ROT6D.copy()
    if ref_frame == "per_keypoint":
        raise ValueError("resolve_base: 'per_keypoint' has no single shared base; "
                         "handle it per-keypoint in the caller")
    slots = _REF_FRAME_SLOTS.get(ref_frame)
    if slots is None:
        raise ValueError(f"resolve_base: unknown ref_frame {ref_frame!r}; "
                         f"expected one of {REF_FRAME_CHOICES}")
    pos_sl, rot_sl = slots
    state32 = np.asarray(state32, dtype=np.float64)
    return state32[pos_sl].copy(), state32[rot_sl].copy()


# ---------------------------------------------------------------------------
# Gripper width normalization: metres <-> [0, 1]
# ---------------------------------------------------------------------------
def gripper_width_to_norm(width_m: np.ndarray, ref_m: float = GRIPPER_WIDTH_REF_M) -> np.ndarray:
    """Absolute gripper width in metres -> normalized [0, 1] (clipped)."""
    return np.clip(np.asarray(width_m, dtype=np.float64) / ref_m, 0.0, 1.0)


def gripper_norm_to_width(norm: np.ndarray, ref_m: float = GRIPPER_WIDTH_REF_M) -> np.ndarray:
    """Normalized [0, 1] gripper -> absolute width in metres."""
    return np.asarray(norm, dtype=np.float64) * ref_m
