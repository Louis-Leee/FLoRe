"""FK-based odometry: estimate root position from joint kinematics + foot contact.

Two implementations:
- FKOdometry (Pinocchio): hardware-deployable, uses only joint angles + IMU orientation.
- MuJoCoFKOdometry: sim-only, uses MuJoCo data.xpos for ground-truth body positions.
  Useful as a diagnostic upper bound for contact-anchored odometry.

The stance foot (lowest foot) is anchored on the ground, and root position
is derived from:  root_pos = stance_anchor_world - foot_relative_to_root.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from holosoma_inference.config.config_types.robot import RobotConfig


_FOOT_BODIES = ("left_foot_contact_point", "right_foot_contact_point")

# Hysteresis: only switch stance foot if the other foot is lower by this margin.
# Prevents chattering when both feet are near the ground, which causes
# anchor resets → position discontinuities → velocity spikes.
_STANCE_SWITCH_MARGIN = 0.02  # 2 cm

# EMA smoothing for velocity (0 = full smoothing, 1 = no smoothing).
_VEL_EMA_ALPHA = 0.2

# Ground-plane height for z-clamping. Prevents z-drift accumulation.
_GROUND_Z = 0.0


def _select_stance_foot(foot_positions, stance_anchor):
    """Select stance foot with hysteresis to prevent chattering."""
    z_values = [fp[2] for fp in foot_positions]
    lowest_i = int(np.argmin(z_values))
    if stance_anchor is None:
        return lowest_i
    current_i = stance_anchor[0]
    if lowest_i == current_i:
        return current_i
    z_diff = z_values[current_i] - z_values[lowest_i]
    if z_diff > _STANCE_SWITCH_MARGIN:
        return lowest_i
    return current_i


def _anchor_update(root_pos, foot_rel, stance_anchor, best_i):
    """Update root position from stance anchor, or set new anchor on foot change."""
    if stance_anchor[0] == best_i:
        root_pos[:2] = stance_anchor[1][:2] - foot_rel[:2]
        root_pos[2] = _GROUND_Z - foot_rel[2]
    else:
        anchor = root_pos + foot_rel
        anchor[2] = _GROUND_Z
        stance_anchor = (best_i, anchor)
    return stance_anchor


def _ema_velocity(root_pos, prev_root_pos, prev_vel, dt):
    """EMA-smoothed velocity from position delta."""
    raw_vel = (root_pos - prev_root_pos) / dt
    return _VEL_EMA_ALPHA * raw_vel + (1.0 - _VEL_EMA_ALPHA) * prev_vel


class FKOdometry:
    """Contact-anchored FK odometry using Pinocchio (hardware-deployable)."""

    def __init__(self, robot_cfg: RobotConfig, urdf_text: str, control_dt: float):
        import pinocchio as pin
        from holosoma_inference.policies.wbt import PinocchioRobot

        self._pin = PinocchioRobot(robot_cfg, urdf_text)
        self._dt = control_dt

        model = self._pin.robot_model
        self._foot_fids = []
        for name in _FOOT_BODIES:
            fid = model.getFrameId(name, pin.FrameType.BODY)
            if fid >= model.nframes:
                logger.warning(f"FKOdometry: body '{name}' not found in URDF")
            self._foot_fids.append(fid)
        logger.info(f"FKOdometry: foot frame IDs {dict(zip(_FOOT_BODIES, self._foot_fids))}")

        self._q_buf = np.zeros(model.nq, dtype=np.float64)
        self._root_pos = None
        self._prev_root_pos = None
        self._stance_anchor = None
        self._vel = np.zeros(3, dtype=np.float64)

    def _run_fk(self, root_pos, imu_quat_wxyz, joint_pos_sdk):
        import pinocchio as pin

        q = self._q_buf
        q[:3] = root_pos
        q[3] = imu_quat_wxyz[1]  # x
        q[4] = imu_quat_wxyz[2]  # y
        q[5] = imu_quat_wxyz[3]  # z
        q[6] = imu_quat_wxyz[0]  # w
        q[7:] = joint_pos_sdk[self._pin.real2pinocchio_index]
        pin.framesForwardKinematics(self._pin.robot_model, self._pin.robot_data, q)
        return [self._pin.robot_data.oMf[fid].translation.copy() for fid in self._foot_fids]

    def update(
        self, imu_quat_wxyz: np.ndarray, joint_pos_sdk: np.ndarray, init_pos: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Call each control step. Returns (root_pos_estimate, root_lin_vel_estimate)."""
        if self._root_pos is None:
            if init_pos is not None:
                self._root_pos = init_pos.copy()
            else:
                self._root_pos = np.array([0.0, 0.0, 0.75])
            self._prev_root_pos = self._root_pos.copy()
            foot_positions = self._run_fk(self._root_pos, imu_quat_wxyz, joint_pos_sdk)
            best_i = int(np.argmin([fp[2] for fp in foot_positions]))
            self._stance_anchor = (best_i, foot_positions[best_i].copy())
            return self._root_pos.copy(), self._vel.copy()

        foot_positions = self._run_fk(self._root_pos, imu_quat_wxyz, joint_pos_sdk)
        best_i = _select_stance_foot(foot_positions, self._stance_anchor)
        foot_rel = foot_positions[best_i] - self._root_pos
        self._stance_anchor = _anchor_update(self._root_pos, foot_rel, self._stance_anchor, best_i)
        self._vel = _ema_velocity(self._root_pos, self._prev_root_pos, self._vel, self._dt)
        self._prev_root_pos = self._root_pos.copy()
        return self._root_pos.copy(), self._vel.copy()


class MuJoCoFKOdometry:
    """Contact-anchored FK odometry using MuJoCo data.xpos (sim-only diagnostic)."""

    def __init__(self, mj_model, mj_data, control_dt: float):
        import mujoco

        self._mj_data = mj_data
        self._dt = control_dt

        self._foot_bids = []
        for name in _FOOT_BODIES:
            bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                logger.warning(f"MuJoCoFKOdometry: body '{name}' not found in XML")
            self._foot_bids.append(bid)
        self._pelvis_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        logger.info(f"MuJoCoFKOdometry: foot body IDs {dict(zip(_FOOT_BODIES, self._foot_bids))}")

        self._root_pos = None
        self._prev_root_pos = None
        self._stance_anchor = None
        self._vel = np.zeros(3, dtype=np.float64)

    def update(self, init_pos: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Call each control step. Returns (root_pos_estimate, root_lin_vel_estimate)."""
        foot_positions = [self._mj_data.xpos[bid].copy() for bid in self._foot_bids]
        pelvis_pos = self._mj_data.xpos[self._pelvis_bid].copy()

        if self._root_pos is None:
            self._root_pos = init_pos.copy() if init_pos is not None else pelvis_pos.copy()
            self._prev_root_pos = self._root_pos.copy()
            best_i = int(np.argmin([fp[2] for fp in foot_positions]))
            self._stance_anchor = (best_i, foot_positions[best_i].copy())
            return self._root_pos.copy(), self._vel.copy()

        best_i = _select_stance_foot(foot_positions, self._stance_anchor)
        foot_rel = foot_positions[best_i] - pelvis_pos
        self._stance_anchor = _anchor_update(self._root_pos, foot_rel, self._stance_anchor, best_i)
        self._vel = _ema_velocity(self._root_pos, self._prev_root_pos, self._vel, self._dt)
        self._prev_root_pos = self._root_pos.copy()
        return self._root_pos.copy(), self._vel.copy()


class FKOdometryWrapper:
    """Interface wrapper that injects Pinocchio FK-estimated position/velocity into get_low_state().

    Drop-in replacement for MocapInterfaceWrapper when no external mocap is available.
    Works on real hardware (only needs joint angles + IMU orientation).
    """

    def __init__(
        self,
        inner_interface,
        robot_cfg: RobotConfig,
        urdf_text: str,
        control_dt: float,
        num_joints: int = 29,
        init_pos: np.ndarray | None = None,
    ):
        self._inner = inner_interface
        self._odom = FKOdometry(robot_cfg, urdf_text, control_dt)
        self._num_joints = num_joints
        self._lin_vel_start = 7 + num_joints
        self._init_pos = init_pos

    def get_low_state(self) -> np.ndarray:
        state = self._inner.get_low_state()
        imu_quat_wxyz = state[0, 3:7]
        joint_pos = state[0, 7 : 7 + self._num_joints]
        pos, vel = self._odom.update(imu_quat_wxyz, joint_pos, init_pos=self._init_pos)
        self._init_pos = None
        state[0, 0:3] = pos
        state[0, self._lin_vel_start : self._lin_vel_start + 3] = vel
        return state

    def __getattr__(self, name):
        return getattr(self._inner, name)


class MuJoCoFKOdometryWrapper:
    """Interface wrapper using MuJoCo ground-truth body positions for FK odometry.

    Sim-only diagnostic tool. Same interface as FKOdometryWrapper.
    """

    def __init__(
        self,
        inner_interface,
        mj_model,
        mj_data,
        control_dt: float,
        num_joints: int = 29,
        init_pos: np.ndarray | None = None,
    ):
        self._inner = inner_interface
        self._odom = MuJoCoFKOdometry(mj_model, mj_data, control_dt)
        self._num_joints = num_joints
        self._lin_vel_start = 7 + num_joints
        self._init_pos = init_pos

    def get_low_state(self) -> np.ndarray:
        state = self._inner.get_low_state()
        pos, vel = self._odom.update(init_pos=self._init_pos)
        self._init_pos = None
        state[0, 0:3] = pos
        state[0, self._lin_vel_start : self._lin_vel_start + 3] = vel
        return state

    def __getattr__(self, name):
        return getattr(self._inner, name)
