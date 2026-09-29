"""Observation mode detection and motion alignment utilities.

Shared between sim2sim (headless eval) and sim2real
(holosoma_wbt.py, holosoma_maskedmimic.py).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def detect_obs_mode(term_names: Sequence[str]) -> str:
    """Detect observation mode from term names.

    Returns one of: ``"dense"``, ``"global"``, ``"global_vel"``, ``"5pt"``,
    ``"3pt"``, ``"3pt_global"``, ``"2pt"``, ``"1pt"``, ``"maskedmimic"``.
    """
    names = set(term_names)
    has_ref_pos = any("motion_ref_pos_b" in n for n in names)
    # 5-POINT must be tested BEFORE the fall-through: "vr_5point" does not contain the
    # "vr_3point" substring, so without this branch a 5pt preset matched NO vr_* rule and
    # silently returned "dense" (its obs carries motion_ref_ori_b, not motion_ref_pos_b).
    # "dense" is a real, registered mode whose obs path skips VR FK entirely, so the
    # 736-D policy would have been fed a zero-filled tracking slot at the right width.
    if any("vr_5point" in n for n in names):
        return "5pt"
    if any("vr_3point" in n for n in names):
        return "3pt_global" if has_ref_pos else "3pt"
    if any("vr_2point" in n for n in names):
        return "2pt"
    if any("vr_1point" in n for n in names):
        return "1pt"
    if any("noisy_reduced_coords" in n or "masked_mimic_target" in n for n in names):
        return "maskedmimic"
    has_ref_vel = any("motion_ref_vel_b" in n for n in names)
    if has_ref_pos and has_ref_vel:
        return "global_vel"
    if has_ref_pos:
        return "global"
    return "dense"


def compute_motion_alignment(
    robot_pos: np.ndarray, robot_quat_wxyz: np.ndarray, ref_pos_0: np.ndarray, ref_quat_wxyz_0: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute yaw + position transform from NPZ frame to mocap frame.

    Aligns the reference trajectory so that frame 0 matches the robot's
    current position and heading. Only xy-position and yaw are aligned;
    z uses motion data directly.

    Args:
        robot_pos: Robot pelvis world position (3,).
        robot_quat_wxyz: Robot pelvis quaternion wxyz (4,).
        ref_pos_0: Reference frame-0 pelvis world position (3,).
        ref_quat_wxyz_0: Reference frame-0 pelvis quaternion wxyz (4,).

    Returns:
        (align_rot_mat [3,3], align_pos_offset [3,], yaw_offset_rad)
    """
    robot_yaw = _quat_yaw(robot_quat_wxyz)
    ref_yaw_0 = _quat_yaw(ref_quat_wxyz_0)
    yaw_offset = robot_yaw - ref_yaw_0

    c, s = float(np.cos(yaw_offset)), float(np.sin(yaw_offset))
    align_rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    ref_pos_0_rotated = align_rot_mat @ ref_pos_0
    align_pos_offset = np.asarray(robot_pos, dtype=np.float64) - ref_pos_0_rotated

    return align_rot_mat, align_pos_offset, float(yaw_offset)


def _quat_yaw(quat_wxyz: np.ndarray) -> float:
    """Extract yaw angle from a wxyz quaternion."""
    q = np.asarray(quat_wxyz).flatten()
    w, x, y, z = q[0], q[1], q[2], q[3]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2)))
