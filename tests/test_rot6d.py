"""Round-trip tests for the rotation conversions — the load-bearing math of the
S1 action -> S0 vr_3point translation."""

import numpy as np
from scipy.spatial.transform import Rotation

from flore.utils.rot6d import (
    quat_wxyz_to_xyzw,
    quat_xyzw_to_rotmat,
    quat_xyzw_to_wxyz,
    rot6d_to_rotmat,
    rotmat_to_quat_xyzw,
    rotmat_to_rot6d,
    rotmat_to_rotvec,
    rotvec_to_rotmat,
)


def _random_rotmats(n: int, seed: int = 0) -> np.ndarray:
    return Rotation.random(n, random_state=seed).as_matrix()


def test_rot6d_roundtrip():
    R = _random_rotmats(100)
    np.testing.assert_allclose(rot6d_to_rotmat(rotmat_to_rot6d(R)), R, atol=1e-6)


def test_rot6d_gram_schmidt_repairs_noise():
    R = _random_rotmats(10)
    noisy = rotmat_to_rot6d(R) + np.random.default_rng(0).normal(0, 1e-3, (10, 6))
    R2 = rot6d_to_rotmat(noisy)
    # Result must be a valid rotation matrix.
    np.testing.assert_allclose(R2 @ np.swapaxes(R2, -1, -2), np.broadcast_to(np.eye(3), R2.shape), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(R2), 1.0, atol=1e-6)


def test_rotvec_roundtrip():
    R = _random_rotmats(100)
    np.testing.assert_allclose(rotvec_to_rotmat(rotmat_to_rotvec(R)), R, atol=1e-6)


def test_rotvec_identity():
    np.testing.assert_allclose(rotvec_to_rotmat(np.zeros(3)), np.eye(3), atol=1e-12)


def test_quat_xyzw_roundtrip():
    R = _random_rotmats(100)
    np.testing.assert_allclose(quat_xyzw_to_rotmat(rotmat_to_quat_xyzw(R)), R, atol=1e-6)


def test_wxyz_xyzw_roundtrip():
    q = Rotation.random(50, random_state=1).as_quat()  # xyzw
    np.testing.assert_allclose(quat_wxyz_to_xyzw(quat_xyzw_to_wxyz(q)), q)


def test_rot6d_to_quat_chain():
    """The exact chain used by action_to_3pt: rot6d -> matrix -> quat xyzw."""
    R = _random_rotmats(50)
    q = rotmat_to_quat_xyzw(rot6d_to_rotmat(rotmat_to_rot6d(R)))
    q_ref = Rotation.from_matrix(R).as_quat()
    # Quaternions are sign-ambiguous: compare |dot| ~ 1.
    dots = np.abs(np.sum(q * q_ref, axis=-1))
    np.testing.assert_allclose(dots, 1.0, atol=1e-6)
