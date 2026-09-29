"""Rotation conversions shared by training, serving, and the client.

Conventions (single source of truth for the whole repo):

- **rot6d** = the FIRST TWO ROWS of a 3x3 rotation matrix, flattened row-major to
  6 floats, reconstructed with Gram-Schmidt. This matches both the GCL training
  transforms and the FAR-pi WBT-VLA wire format.
- **Quaternions on the S0 wire (:8770 ``vr_3point_orn``)** are **xyzw** (scalar-last),
  matching holosoma's 3pt bridge. libsurvive/pysurvive and the UMI recorder use
  **wxyz** (scalar-first) — convert at exactly one place (the boundary you're on).
- Axis-angle (rotvec) is the on-disk UMI zarr rotation representation.
"""

from __future__ import annotations

import numpy as np


def rotmat_to_rot6d(rotmat: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrix -> (..., 6) first-two-rows rot6d."""
    return rotmat[..., :2, :].reshape(*rotmat.shape[:-2], 6)


def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) rot6d -> (..., 3, 3) rotation matrix via Gram-Schmidt.

    Rows are orthonormalized: r0 = normalize(a0); r1 = normalize(a1 - <a1,r0> r0);
    r2 = r0 x r1.
    """
    a = rot6d.reshape(*rot6d.shape[:-1], 2, 3)
    r0 = a[..., 0, :]
    r0 = r0 / np.linalg.norm(r0, axis=-1, keepdims=True).clip(1e-8)
    a1 = a[..., 1, :]
    r1 = a1 - np.sum(a1 * r0, axis=-1, keepdims=True) * r0
    r1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True).clip(1e-8)
    r2 = np.cross(r0, r1)
    return np.stack([r0, r1, r2], axis=-2)


def rotvec_to_rotmat(rotvec: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrix (Rodrigues)."""
    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    small = theta[..., 0] < 1e-8
    k = np.where(theta > 1e-8, rotvec / theta.clip(1e-8), 0.0)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    K = np.stack(
        [
            np.stack([zero, -kz, ky], axis=-1),
            np.stack([kz, zero, -kx], axis=-1),
            np.stack([-ky, kx, zero], axis=-1),
        ],
        axis=-2,
    )
    eye = np.broadcast_to(np.eye(3), K.shape).copy()
    st = np.sin(theta)[..., None]
    ct = np.cos(theta)[..., None]
    R = eye + st * K + (1.0 - ct) * (K @ K)
    R[small] = np.eye(3)
    return R


def rotmat_to_rotvec(rotmat: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrix -> (..., 3) axis-angle."""
    from scipy.spatial.transform import Rotation

    shape = rotmat.shape[:-2]
    return Rotation.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(*shape, 3)


def rotmat_to_quat_xyzw(rotmat: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) xyzw quaternion (S0 wire convention)."""
    from scipy.spatial.transform import Rotation

    shape = rotmat.shape[:-2]
    return Rotation.from_matrix(rotmat.reshape(-1, 3, 3)).as_quat().reshape(*shape, 4)


def quat_xyzw_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw quaternion -> (..., 3, 3)."""
    from scipy.spatial.transform import Rotation

    shape = quat.shape[:-1]
    return Rotation.from_quat(quat.reshape(-1, 4)).as_matrix().reshape(*shape, 3, 3)


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """libsurvive/UMI-recorder wxyz -> wire xyzw."""
    return np.concatenate([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], axis=-1)


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """wire xyzw -> wxyz."""
    return np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], axis=-1)
