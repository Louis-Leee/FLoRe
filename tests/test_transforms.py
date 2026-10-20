"""Data transforms: rotvec<->rot6d lift, absolute<->relative EE SE(3) roundtrip,
gripper normalization, and the GCL quantile norm/unnorm roundtrip."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from flore.s1_policy.data.norm_stats import (
    NormStats,
    compute_quantile_stats,
    normalize_quantile,
    unnormalize_quantile,
)
from flore.s1_policy.data.transforms import (
    absolute_to_delta_ee,
    delta_ee_to_absolute,
    gripper_norm_to_width,
    gripper_width_to_norm,
    rot6d_to_rotvec,
    rotvec_to_rot6d,
)


def test_rotvec_rot6d_roundtrip():
    rotvec = Rotation.random(50, random_state=0).as_rotvec()
    back = rot6d_to_rotvec(rotvec_to_rot6d(rotvec))
    # Compare via rotation matrices (rotvec is not unique near pi).
    r0 = Rotation.from_rotvec(rotvec).as_matrix()
    r1 = Rotation.from_rotvec(back).as_matrix()
    np.testing.assert_allclose(r0, r1, atol=1e-6)


def test_absolute_delta_ee_roundtrip():
    rng = np.random.default_rng(0)
    pos = rng.normal(0, 0.2, (30, 3))
    rot6d = rotvec_to_rot6d(Rotation.random(30, random_state=1).as_rotvec())
    ref_pos = rng.normal(0, 0.2, 3)
    ref_rot6d = rotvec_to_rot6d(Rotation.random(1, random_state=2).as_rotvec())[0]

    rel_pos, rel_rot6d = absolute_to_delta_ee(pos, rot6d, ref_pos, ref_rot6d)
    abs_pos, abs_rot6d = delta_ee_to_absolute(rel_pos, rel_rot6d, ref_pos, ref_rot6d)

    np.testing.assert_allclose(abs_pos, pos, atol=1e-6)
    # rot6d equality via matrices
    r_a = Rotation.from_matrix  # noqa: F841 (readability)
    np.testing.assert_allclose(abs_rot6d, rot6d, atol=1e-6)


def test_delta_ee_relative_to_self_is_identity():
    """A pose relative to itself is zero translation + identity rotation."""
    pos = np.array([[1.0, 2.0, 3.0]])
    rot6d = rotvec_to_rot6d(np.array([[0.3, -0.2, 0.5]]))
    rel_pos, rel_rot6d = absolute_to_delta_ee(pos, rot6d, pos[0], rot6d[0])
    np.testing.assert_allclose(rel_pos[0], 0.0, atol=1e-9)
    np.testing.assert_allclose(rel_rot6d[0], [1, 0, 0, 0, 1, 0], atol=1e-9)


def test_gripper_norm_roundtrip():
    w = np.array([0.0, 0.0475, 0.095])
    norm = gripper_width_to_norm(w)
    np.testing.assert_allclose(norm, [0.0, 0.5, 1.0], atol=1e-9)
    np.testing.assert_allclose(gripper_norm_to_width(norm), w, atol=1e-9)
    # clipping
    assert gripper_width_to_norm(np.array([0.2]))[0] == 1.0


def test_quantile_norm_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2, (1000, 4))
    stats = compute_quantile_stats({"a": x})["a"]
    xn = normalize_quantile(x, stats)
    xr = unnormalize_quantile(xn, stats)
    np.testing.assert_allclose(xr, x, atol=1e-6)


def test_quantile_norm_passthrough_extra_dims():
    """Extra trailing dims beyond the stats width pass through untouched."""
    stats = NormStats(q01=np.zeros(3), q99=np.ones(3))
    x = np.concatenate([np.full((5, 3), 0.5), np.full((5, 2), 7.0)], axis=-1)
    xn = normalize_quantile(x, stats)
    np.testing.assert_allclose(xn[:, 3:], 7.0)  # untouched
    np.testing.assert_allclose(xn[:, :3], 0.0, atol=1e-4)  # 0.5 in [0,1] -> ~0
