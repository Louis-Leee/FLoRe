#!/usr/bin/env python3
"""Standalone viser replay using the correct retargeting pipeline from dev/csf/gmr.

Runs the SMPLRetargeter + TML ONNX tracker in MuJoCo, displays:
  1. Raw PICO skeleton (magenta)
  2. Retargeted G1 (orange) — IK output, fixed-base FK + root offset
  3. TML tracker G1 (green) — policy rollout in MuJoCo sim

Usage:
    python viser_teleop_replay.py \
        --pkl .../pico_body_recording.pkl \
        --urdf .../g1_29dof.urdf \
        --xml .../scene_g1_29dof_wbt_plane.xml \
        --tml_onnx .../tml_general.onnx
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import viser
from scipy.spatial.transform import Rotation
from wbt_training.config_values.robot_config import (
    ISAACLAB_TO_MUJOCO_DOF as ISAAC_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB_DOF as MUJOCO_TO_ISAAC,
)

from teleop_inference.retargeting.smpl_retargeter import SMPLRetargeter, _COORD_ROT_MAT

# SMPL kinematic tree parent indices for 24 joints
_SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def _quat_inverse_wxyz(q):
    return np.concatenate((q[:, 0:1], -q[:, 1:]), axis=1)


def _quat_mul_wxyz(a, b):
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.stack([w, x, y, z]).reshape(a.shape)


def _quat_rotate_inverse_wxyz(q, v):
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0
    return a - b + c


def _matrix_from_quat_wxyz(q):
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two_s = 2.0 / (q * q).sum(-1)
    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))


def _quat_to_rpy(q_wxyz):
    w, x, y, z = q_wxyz
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = np.sign(sinp) * (np.pi / 2) if abs(sinp) >= 1 else np.arcsin(sinp)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _rpy_to_quat_wxyz(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def precompute_retarget(retargeter, poses, skip_first=1):
    """Run retargeter on all frames, returning joints, root pos/quat, pico skeletons.

    PICO skeleton processing matches teleop_wbt.py lines 395-430:
      1. coord transform (Y-up → Z-up)
      2. scale + ground anchor
      3. remove heading (yaw only) + align pico +Y fwd to URDF +X fwd
      4. anchor to retargeter root position
    """
    poses = poses[skip_first:]
    N = len(poses)
    all_joints = []
    all_root_pos = []
    all_root_quat_wxyz = []
    all_pico_local = []
    all_root_body_quat_xyzw = []

    for i in range(N):
        q_joints, dq, root_orn_wxyz = retargeter.retarget(poses[i])
        root_pos = retargeter._prev_q[:3].copy()
        root_body_quat_xyzw = retargeter._prev_q[3:7].copy()

        all_joints.append(q_joints.copy())
        all_root_pos.append(root_pos)
        all_root_quat_wxyz.append(root_orn_wxyz.copy())
        all_root_body_quat_xyzw.append(root_body_quat_xyzw.copy())

        # Pico skeleton: coord transform + scale + ground anchor
        pico_zup = poses[i][:, :3] @ _COORD_ROT_MAT.T
        pico_scaled = retargeter._scale_positions(pico_zup)
        pico_grounded = SMPLRetargeter._anchor_to_ground(pico_scaled)

        # Remove heading (yaw only) + align pico frame (+Y fwd) to URDF frame (+X fwd)
        # Matches teleop_wbt.py lines 416-422
        yaw = Rotation.from_quat(root_body_quat_xyzw).as_euler("ZYX")[0]
        yaw_mat = Rotation.from_euler("z", yaw + np.pi / 2).as_matrix()
        pico_relative = pico_grounded - pico_grounded[0:1]  # pelvis-relative
        pico_local = pico_relative @ yaw_mat
        all_pico_local.append(pico_local)

        if (i + 1) % 100 == 0 or i == N - 1:
            print(f"  Retargeted {i + 1}/{N}")

    return all_joints, all_root_pos, all_root_quat_wxyz, all_pico_local, all_root_body_quat_xyzw


def _load_mujoco_scene(xml_path):
    """Load MuJoCo model from scene XML, handling asset directory resolution."""
    abs_xml = os.path.abspath(xml_path)
    xml_dir = os.path.dirname(abs_xml)
    search = xml_dir
    for _ in range(5):
        if os.path.isdir(os.path.join(search, "assets")):
            break
        search = os.path.dirname(search)
    orig = os.getcwd()
    os.chdir(search)
    model = mujoco.MjModel.from_xml_path(abs_xml)
    os.chdir(orig)
    return model


def run_tml_policy(onnx_path, pico_poses, retargeter, xml_path, policy_fps=50.0):
    """Run TML policy in MuJoCo, matching teleop_wbt.py exactly.

    Reproduces the full original flow:
      1. Init phase: 500-step interpolation to default pose (get_init_target)
      2. Policy warmup: ~50 steps with static reference (frame 0, motion not progressing)
      3. Motion clip: frame_idx advances by 1 per policy step, retarget at 50Hz
    """
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map

    default_pos_isaac = np.array([float(x) for x in meta["default_joint_pos"].split(",")])
    action_scale_isaac = np.array([float(x) for x in meta["action_scale"].split(",")])
    kp_isaac = np.array([float(x) for x in meta["joint_stiffness"].split(",")])
    kd_isaac = np.array([float(x) for x in meta["joint_damping"].split(",")])

    default_q_mj = default_pos_isaac[ISAAC_TO_MUJOCO]
    action_scale_mj = action_scale_isaac[ISAAC_TO_MUJOCO]
    kp_mj = kp_isaac[ISAAC_TO_MUJOCO]
    kd_mj = kd_isaac[ISAAC_TO_MUJOCO]

    # MuJoCo scene model (free joint) — holosoma overrides XML timestep to 1/fps
    model = _load_mujoco_scene(xml_path)
    data = mujoco.MjData(model)

    sim_dt = 1.0 / 200.0  # holosoma fps=200
    model.opt.timestep = sim_dt
    control_decimation = 4  # holosoma control_decimation=4 → policy at 50Hz
    steps_per_control = control_decimation

    torque_limit = np.array([model.actuator_ctrlrange[i][1] for i in range(model.nu)])

    # Initial pose (matching holosoma robot config: pos=[0,0,0.8], rot=identity)
    data.qpos[2] = 0.8
    data.qpos[3:7] = [1, 0, 0, 0]  # wxyz identity
    data.qpos[7:36] = default_q_mj
    mujoco.mj_forward(model, data)

    # Reset retargeter so it starts fresh for the policy run
    retargeter.reset()

    # --- Phase 1: Quick settle with no gravity, then enable ---
    # PD alone can't support the robot (torque limits). Place robot at correct
    # height with zero gravity for a few steps to establish contacts, then
    # immediately start policy which provides active balance.
    print("  Init phase: zero-gravity settle...")
    orig_gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0.0
    for _ in range(50):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
    data.qvel[:] = 0.0
    model.opt.gravity[:] = orig_gravity
    print(f"  Init done: height={data.qpos[2]:.3f}")

    # --- Phase 2: Policy warmup with static reference (frame 0) ---
    # Matches the gap between pressing ']' (start policy) and 'm' (start motion)
    print("  Policy warmup: 50 steps with static ref...")
    robot_heading_init_inv = None
    ref_heading_init_inv = None
    last_action = np.zeros(29)
    target_q = default_q_mj.copy()

    # Retarget frame 0 once for warmup reference
    warmup_poses = pico_poses[0]
    q_ref_warmup, dq_ref_warmup, ref_orn_warmup = retargeter.retarget(warmup_poses)

    for warmup_step in range(50):
        # Re-retarget same frame (velocity = 0 after first call, matching original)
        q_ref_mujoco, dq_ref_mujoco, ref_root_orn_wxyz = retargeter.retarget(warmup_poses)
        ref_q_isaac = q_ref_mujoco[MUJOCO_TO_ISAAC]
        ref_dq_isaac = dq_ref_mujoco[MUJOCO_TO_ISAAC]

        for sub in range(steps_per_control):
            if sub == 0:
                base_quat_wxyz = data.qpos[3:7].copy()
                if np.dot(base_quat_wxyz, base_quat_wxyz) < 1e-8:
                    base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
                dof_pos_mj = data.qpos[7:36].copy()
                ang_vel = data.qvel[3:6].copy()
                dof_vel_mj = data.qvel[6:35].copy()

                if robot_heading_init_inv is None:
                    _, _, yaw = _quat_to_rpy(base_quat_wxyz)
                    heading_quat = _rpy_to_quat_wxyz(0, 0, yaw).reshape(1, 4)
                    robot_heading_init_inv = _quat_inverse_wxyz(heading_quat)
                if ref_heading_init_inv is None:
                    ref_heading_init_inv = _quat_inverse_wxyz(ref_root_orn_wxyz.reshape(1, 4))

                root_orn = _quat_mul_wxyz(robot_heading_init_inv, base_quat_wxyz.reshape(1, 4))
                rel_ref_orn = _quat_mul_wxyz(ref_heading_init_inv, ref_root_orn_wxyz.reshape(1, 4))
                root_inv = _quat_inverse_wxyz(root_orn)
                relative_quat = _quat_mul_wxyz(root_inv, rel_ref_orn)
                rot_matrix = _matrix_from_quat_wxyz(relative_quat)
                anchor_ori = rot_matrix[..., :2].reshape(-1)
                gravity = _quat_rotate_inverse_wxyz(root_orn, np.array([[0, 0, -1.0]])).flatten()

                command = np.concatenate([ref_q_isaac, ref_dq_isaac])
                dof_pos = dof_pos_mj[MUJOCO_TO_ISAAC] - default_pos_isaac
                dof_vel = dof_vel_mj[MUJOCO_TO_ISAAC]
                obs = (
                    np.concatenate([command, anchor_ori, gravity, ang_vel, dof_pos, dof_vel, last_action])
                    .reshape(1, -1)
                    .astype(np.float32)
                )

                action_isaac = session.run(None, {"obs": obs})[0].flatten()
                last_action = action_isaac.copy()
                target_q = default_q_mj + action_isaac[ISAAC_TO_MUJOCO] * action_scale_mj

            tau = (target_q - data.qpos[7:36]) * kp_mj - data.qvel[6:35] * kd_mj
            data.ctrl[:] = np.clip(tau, -torque_limit, torque_limit)
            mujoco.mj_step(model, data)

    # --- Phase 3: Motion clip (frame_idx advances 1 per policy step) ---
    N_pico = len(pico_poses)
    results = []

    for frame_idx in range(N_pico):
        joint_poses = pico_poses[min(frame_idx, N_pico - 1)]

        # Retarget at policy rate (same as original teleop_wbt.py)
        q_ref_mujoco, dq_ref_mujoco, ref_root_orn_wxyz = retargeter.retarget(joint_poses)
        ref_q_isaac = q_ref_mujoco[MUJOCO_TO_ISAAC]
        ref_dq_isaac = dq_ref_mujoco[MUJOCO_TO_ISAAC]

        for sub in range(steps_per_control):
            if sub == 0:
                base_quat_wxyz = data.qpos[3:7].copy()
                if np.dot(base_quat_wxyz, base_quat_wxyz) < 1e-8:
                    base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
                dof_pos_mj = data.qpos[7:36].copy()
                ang_vel = data.qvel[3:6].copy()
                dof_vel_mj = data.qvel[6:35].copy()

                # Heading-corrected orientations (heading already initialized in warmup)
                root_orn = _quat_mul_wxyz(robot_heading_init_inv, base_quat_wxyz.reshape(1, 4))
                rel_ref_orn = _quat_mul_wxyz(ref_heading_init_inv, ref_root_orn_wxyz.reshape(1, 4))

                root_inv = _quat_inverse_wxyz(root_orn)
                relative_quat = _quat_mul_wxyz(root_inv, rel_ref_orn)
                rot_matrix = _matrix_from_quat_wxyz(relative_quat)
                anchor_ori = rot_matrix[..., :2].reshape(-1)

                gravity = _quat_rotate_inverse_wxyz(root_orn, np.array([[0, 0, -1.0]])).flatten()

                command = np.concatenate([ref_q_isaac, ref_dq_isaac])
                dof_pos = dof_pos_mj[MUJOCO_TO_ISAAC] - default_pos_isaac
                dof_vel = dof_vel_mj[MUJOCO_TO_ISAAC]

                obs = (
                    np.concatenate([command, anchor_ori, gravity, ang_vel, dof_pos, dof_vel, last_action])
                    .reshape(1, -1)
                    .astype(np.float32)
                )

                action_isaac = session.run(None, {"obs": obs})[0].flatten()
                last_action = action_isaac.copy()
                target_q = default_q_mj + action_isaac[ISAAC_TO_MUJOCO] * action_scale_mj

                results.append((data.qpos[:3].copy(), data.qpos[3:7].copy(), data.qpos[7:36].copy()))

            tau = (target_q - data.qpos[7:36]) * kp_mj - data.qvel[6:35] * kd_mj
            data.ctrl[:] = np.clip(tau, -torque_limit, torque_limit)
            mujoco.mj_step(model, data)

            if data.qpos[2] < 0.3:
                break

        if data.qpos[2] < 0.3:
            print(f"  TML fell at step {frame_idx}")
            break

    return results


def _precompute_ref_body_fk(retarget_joints, retarget_root_pos, retarget_root_quat_wxyz, xml_path):
    """Run MuJoCo FK on retargeted joints to get body positions/quaternions for all frames."""
    model = _load_mujoco_scene(xml_path)
    data = mujoco.MjData(model)

    N = len(retarget_joints)
    all_body_pos = []  # [N, nbody, 3]
    all_body_quat = []  # [N, nbody, 4] wxyz

    for i in range(N):
        data.qpos[:3] = retarget_root_pos[i]
        # Convert root quat from wxyz to MuJoCo wxyz (same convention)
        data.qpos[3:7] = retarget_root_quat_wxyz[i]
        data.qpos[7:36] = retarget_joints[i]
        mujoco.mj_forward(model, data)
        all_body_pos.append(data.xpos.copy())
        all_body_quat.append(data.xquat.copy())  # wxyz

    return np.array(all_body_pos), np.array(all_body_quat), model


def run_maskedmimic_policy(
    onnx_dir,
    retarget_joints,
    retarget_root_pos,
    retarget_root_quat_wxyz,
    xml_path,
    ref_body_pos,
    ref_body_quat,
    ref_model,
    policy_fps=50.0,
    pico_fps=5.0,
):
    """Run MaskedMimic policy (2409D) in MuJoCo using retargeted reference with body FK."""
    import json
    import torch
    import glob as _glob

    onnx_files = sorted(_glob.glob(os.path.join(onnx_dir, "*.onnx")))
    if not onnx_files:
        print(f"  No ONNX files in {onnx_dir}")
        return []
    onnx_path = onnx_files[-1]

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    exp_config = json.loads(meta["experiment_config"])

    dof_names = exp_config["robot"]["dof_names"]
    default_angles = exp_config["robot"]["init_state"]["default_joint_angles"]
    default_q = np.array([default_angles[name] for name in dof_names])
    action_scale = np.array([float(x) for x in meta["action_scale"].strip("[]").split(",")])
    kp = np.array([float(x) for x in meta["kp"].strip("[]").split(",")])
    kd = np.array([float(x) for x in meta["kd"].strip("[]").split(",")])

    clip_obs = exp_config["observation"].get("clip_observations", 100.0)
    term_names = sorted(exp_config["observation"]["groups"]["actor_obs"].get("terms", {}).keys())

    sim_fps = exp_config["simulator"]["config"]["sim"]["fps"]
    decimation = exp_config["simulator"]["config"]["sim"]["control_decimation"]
    sim_dt = 1.0 / sim_fps
    control_dt = decimation / sim_fps
    steps_per_control = max(1, int(control_dt / sim_dt))

    # Load normalizer from .pt
    _NORM_EPS = 1e-5
    _NORM_CLIP = 5.0
    pt_name = os.path.splitext(os.path.basename(onnx_path))[0] + ".pt"
    pt_path = os.path.join(onnx_dir, pt_name)
    norm_mean, norm_std = None, None
    if os.path.exists(pt_path):
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
        norm_sd = ckpt.get("actor_obs_normalizer_state_dict")
        if norm_sd is not None and "_mean" in norm_sd:
            norm_mean = norm_sd["_mean"].numpy().flatten()
            norm_std = norm_sd["_std"].numpy().flatten()
            print(f"  Loaded normalizer from {pt_name}")
    if norm_mean is None:
        print("  WARNING: no normalizer found — MaskedMimic will likely fail!")

    # MuJoCo scene model
    model = _load_mujoco_scene(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = sim_dt

    torque_limit = np.array([model.actuator_ctrlrange[i][1] for i in range(model.nu)])

    # Body indices for MaskedMimic targets
    _MM_BODIES = [
        "pelvis",
        "torso_link",
        "right_ankle_roll_link",
        "left_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]
    mm_body_mj_bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn) for bn in _MM_BODIES]
    # Same body IDs in the ref FK model (same XML)
    mm_ref_bids = [mujoco.mj_name2id(ref_model, mujoco.mjtObj.mjOBJ_BODY, bn) for bn in _MM_BODIES]
    _MM_NUM_FUTURE = 30
    _MM_EVAL_VISIBLE_STEPS = 15

    # Initial pose: standard training init (matching holosoma robot config exactly)
    data.qpos[2] = 0.8
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:36] = default_q
    mujoco.mj_forward(model, data)

    # --- Phase 1: Zero-gravity settle (same fix as TML) ---
    print("  Init phase: zero-gravity settle...")
    orig_gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0.0
    for _ in range(50):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
    data.qvel[:] = 0.0
    model.opt.gravity[:] = orig_gravity
    print(f"  Init done: height={data.qpos[2]:.3f}")

    last_action = np.zeros(29)
    target_q = default_q.copy()
    ticker = 0

    # Heading alignment for reference body FK (matching TML/WBT pattern)
    # In training, robot and ref start at same heading. In eval they don't.
    mm_heading_align = None  # Rotation to align ref heading to robot heading
    mm_ref_root_pos_init = None
    mm_robot_pos_init = None

    N_retarget = len(retarget_joints)
    N_policy = int(N_retarget * policy_fps / pico_fps)
    results = []

    for policy_step in range(N_policy):
        retarget_idx = min(int(policy_step * pico_fps / policy_fps), N_retarget - 1)

        for sub in range(steps_per_control):
            if sub == 0:
                q_isaac = data.qpos[7:36].copy()
                dq_isaac = data.qvel[6:35].copy()
                robot_quat_xyzw = data.qpos[3:7][[1, 2, 3, 0]]
                omega = data.qvel[3:6].copy()
                robot_rot = Rotation.from_quat(robot_quat_xyzw)
                robot_rot_inv = robot_rot.inv()

                # Initialize heading alignment on first step
                if mm_heading_align is None:
                    robot_yaw = robot_rot.as_euler("ZYX")[0]
                    ref_root_quat_xyzw = retarget_root_quat_wxyz[retarget_idx][[1, 2, 3, 0]]
                    ref_yaw = Rotation.from_quat(ref_root_quat_xyzw).as_euler("ZYX")[0]
                    mm_heading_align = Rotation.from_euler("z", robot_yaw - ref_yaw)
                    mm_ref_root_pos_init = retarget_root_pos[retarget_idx].copy()
                    mm_robot_pos_init = data.qpos[:3].copy()

                terms = {}
                for name in term_names:
                    if "noisy_reduced_coords" in name:
                        joint_pos_rel = q_isaac - default_q
                        proj_grav = robot_rot_inv.apply(np.array([0, 0, -1.0]))
                        base_lin_vel_b = robot_rot_inv.apply(data.qvel[:3])
                        terms[name] = np.clip(
                            np.concatenate([joint_pos_rel, dq_isaac, proj_grav, base_lin_vel_b, omega]),
                            -clip_obs,
                            clip_obs,
                        )
                    elif "masked_mimic_target_poses" in name and "masks" not in name:
                        robot_bpos = np.array([data.xpos[bid].copy() for bid in mm_body_mj_bids])
                        robot_bquat_xyzw = np.array([data.xquat[bid][[1, 2, 3, 0]].copy() for bid in mm_body_mj_bids])
                        robot_brots_inv = [Rotation.from_quat(q).inv() for q in robot_bquat_xyzw]
                        future_poses = []
                        for step in range(_MM_NUM_FUTURE):
                            if step < _MM_EVAL_VISIBLE_STEPS:
                                ft_pico = min(
                                    retarget_idx + int((step + 1) * pico_fps / policy_fps + 0.5), N_retarget - 1
                                )
                                for bi in range(len(mm_ref_bids)):
                                    ref_bpos_raw = ref_body_pos[ft_pico, mm_ref_bids[bi]]
                                    ref_bquat_wxyz = ref_body_quat[ft_pico, mm_ref_bids[bi]]
                                    ref_bquat_xyzw = np.array(
                                        [ref_bquat_wxyz[1], ref_bquat_wxyz[2], ref_bquat_wxyz[3], ref_bquat_wxyz[0]]
                                    )
                                    # Heading-align reference body: rotate about ref root, translate to robot
                                    ref_bpos = (
                                        mm_heading_align.apply(ref_bpos_raw - mm_ref_root_pos_init) + mm_robot_pos_init
                                    )
                                    ref_brot_aligned = mm_heading_align * Rotation.from_quat(ref_bquat_xyzw)
                                    dp = robot_brots_inv[bi].apply(ref_bpos - robot_bpos[bi])
                                    dq = (robot_brots_inv[bi] * ref_brot_aligned).as_quat()
                                    if dq[3] < 0:
                                        dq = -dq
                                    future_poses.append(dp)
                                    future_poses.append(dq)
                            else:
                                future_poses.append(np.zeros(7 * len(mm_ref_bids)))
                        terms[name] = np.clip(np.concatenate(future_poses), -clip_obs, clip_obs)
                    elif "masked_mimic_target_masks" in name:
                        n_bodies = len(mm_ref_bids)
                        masks = np.zeros(_MM_NUM_FUTURE * n_bodies)
                        masks[: _MM_EVAL_VISIBLE_STEPS * n_bodies] = 1.0
                        terms[name] = masks
                    elif "joint_space_target_poses" in name:
                        jt = []
                        for step in range(_MM_NUM_FUTURE):
                            if step < _MM_EVAL_VISIBLE_STEPS:
                                ft_pico = min(
                                    retarget_idx + int((step + 1) * pico_fps / policy_fps + 0.5), N_retarget - 1
                                )
                                jt.append(retarget_joints[ft_pico])
                            else:
                                jt.append(np.zeros(29))
                        terms[name] = np.clip(np.concatenate(jt), -clip_obs, clip_obs)
                    elif "joint_space_mode_mask" in name:
                        terms[name] = np.array([1.0])
                    elif "historical_pose" in name:
                        terms[name] = np.zeros(0)
                    elif "end_effector_masses" in name:
                        terms[name] = np.zeros(2)
                    elif "previous_actions" in name or "actions" in name:
                        terms[name] = np.clip(last_action.copy(), -clip_obs, clip_obs)
                    else:
                        terms[name] = np.zeros(1)

                obs = np.concatenate([terms[n] for n in term_names]).reshape(1, -1).astype(np.float32)

                if norm_mean is not None:
                    normalized = (obs - norm_mean) / (norm_std + _NORM_EPS)
                    clipped = np.clip(normalized, -_NORM_CLIP, _NORM_CLIP)
                    obs = (clipped * (norm_std + _NORM_EPS) + norm_mean).astype(np.float32)

                inputs = {"obs": obs, "time_step": np.array([[float(ticker)]], dtype=np.float32)}
                ticker += 1

                action_isaac = session.run(None, inputs)[0].flatten()
                last_action = action_isaac.copy()
                target_q = default_q + action_isaac * action_scale

                results.append((data.qpos[:3].copy(), data.qpos[3:7].copy(), data.qpos[7:36].copy()))

            tau = (target_q - data.qpos[7:36]) * kp - data.qvel[6:35] * kd
            data.ctrl[:] = np.clip(tau, -torque_limit, torque_limit)
            mujoco.mj_step(model, data)

            if data.qpos[2] < 0.3:
                break

        if data.qpos[2] < 0.3:
            print(f"  MaskedMimic fell at step {policy_step}")
            break

    return results


def run_wbt_policy(
    onnx_path, retarget_joints, retarget_root_pos, retarget_root_quat_wxyz, xml_path, policy_fps=50.0, pico_fps=5.0
):
    """Run Dense WBT policy (628D, history=4) in MuJoCo using retargeted reference."""
    import json

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    exp_config = json.loads(meta["experiment_config"])

    dof_names = exp_config["robot"]["dof_names"]
    default_angles = exp_config["robot"]["init_state"]["default_joint_angles"]
    default_q = np.array([default_angles[name] for name in dof_names])
    action_scale = np.array([float(x) for x in meta["action_scale"].strip("[]").split(",")])
    kp = np.array([float(x) for x in meta["kp"].strip("[]").split(",")])
    kd = np.array([float(x) for x in meta["kd"].strip("[]").split(",")])
    # NOTE: Scene XML, URDF, and Isaac dof_names all share the same joint ordering.
    # No ISAAC_TO_MUJOCO conversion needed for holosoma-trained models.

    obs_cfg = exp_config["observation"]["groups"]["actor_obs"]
    history_length = obs_cfg.get("history_length", 1)
    clip_obs = exp_config["observation"].get("clip_observations", 100.0)
    term_names = sorted(obs_cfg.get("terms", {}).keys())

    sim_fps = exp_config["simulator"]["config"]["sim"]["fps"]
    decimation = exp_config["simulator"]["config"]["sim"]["control_decimation"]
    sim_dt = 1.0 / sim_fps
    control_dt = decimation / sim_fps
    steps_per_control = max(1, int(control_dt / sim_dt))

    # MuJoCo scene model
    model = _load_mujoco_scene(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = sim_dt

    torque_limit = np.array([model.actuator_ctrlrange[i][1] for i in range(model.nu)])

    # Initial pose: standard training init (matching holosoma robot config exactly)
    # In training: pos=[0,0,0.8], rot=identity, joints=default
    data.qpos[2] = 0.8
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:36] = default_q
    mujoco.mj_forward(model, data)

    # --- Phase 1: Zero-gravity settle (same fix as TML) ---
    # PD alone can't support the robot with torque limits. Settle with no gravity
    # to establish contacts, then restore gravity when policy provides active balance.
    print("  Init phase: zero-gravity settle...")
    orig_gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0.0
    for _ in range(50):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
    data.qvel[:] = 0.0
    model.opt.gravity[:] = orig_gravity
    print(f"  Init done: height={data.qpos[2]:.3f}")

    input_names = [inp.name for inp in session.get_inputs()]
    has_time_step = "time_step" in input_names

    history = {n: deque(maxlen=history_length) for n in term_names}
    last_action = np.zeros(29)
    target_q = default_q.copy()
    ticker = 0

    # Heading alignment: match TML approach — remove initial heading from BOTH
    # robot and reference independently, so both are in a heading-free frame.
    # In training, both start at heading=0, so heading_init_inv = identity.
    # In eval, robot starts at ~0 but ref has retargeter heading.
    robot_heading_init_inv = None  # inverse of robot's initial heading-only quat (xyzw)
    ref_heading_init_inv = None  # inverse of ref's initial full orientation (xyzw)
    ref_pos_init = None  # Initial reference root position (for global pos offset)
    robot_pos_init = None

    N_retarget = len(retarget_joints)
    N_policy = int(N_retarget * policy_fps / pico_fps)
    results = []

    for policy_step in range(N_policy):
        retarget_idx = min(int(policy_step * pico_fps / policy_fps), N_retarget - 1)

        for sub in range(steps_per_control):
            if sub == 0:
                ref_dof_pos = retarget_joints[retarget_idx]  # already in Isaac/scene order
                if retarget_idx > 0:
                    ref_dof_vel = (ref_dof_pos - retarget_joints[retarget_idx - 1]) * pico_fps
                else:
                    ref_dof_vel = np.zeros(29)
                ref_root_orn_wxyz = retarget_root_quat_wxyz[retarget_idx]

                # Robot state (scene XML qpos is already in Isaac order)
                q_isaac = data.qpos[7:36].copy()
                dq_isaac = data.qvel[6:35].copy()
                robot_quat_xyzw = data.qpos[3:7][[1, 2, 3, 0]]
                omega = data.qvel[3:6].copy()

                robot_rot = Rotation.from_quat(robot_quat_xyzw)
                _robot_rot_inv = robot_rot.inv()  # noqa: F841
                ref_quat_xyzw = ref_root_orn_wxyz[[1, 2, 3, 0]]
                ref_rot = Rotation.from_quat(ref_quat_xyzw)

                # Initialize heading alignment on first step (matching TML approach)
                if robot_heading_init_inv is None:
                    robot_yaw = robot_rot.as_euler("ZYX")[0]
                    robot_heading_init_inv = Rotation.from_euler("z", -robot_yaw)
                if ref_heading_init_inv is None:
                    ref_heading_init_inv = ref_rot.inv()
                    ref_pos_init = retarget_root_pos[retarget_idx].copy()
                    robot_pos_init = data.qpos[:3].copy()

                # Remove initial headings from both (matching TML pattern exactly)
                root_orn = robot_heading_init_inv * robot_rot
                rel_ref_orn = ref_heading_init_inv * ref_rot
                # motion_ref_ori_b: relative rotation in heading-free frame
                rel_rot = root_orn.inv() * rel_ref_orn

                # Reference root position/velocity for global models
                ref_root_pos = retarget_root_pos[retarget_idx]
                # Offset ref so it starts at robot's initial position
                ref_root_pos_aligned = ref_root_pos - ref_pos_init + robot_pos_init
                if retarget_idx > 0:
                    ref_root_vel_w = (ref_root_pos - retarget_root_pos[retarget_idx - 1]) * pico_fps
                else:
                    ref_root_vel_w = np.zeros(3)

                robot_pos = data.qpos[:3].copy()
                robot_lin_vel = data.qvel[:3].copy()

                # Build terms — use heading-removed root_orn for gravity (matching TML).
                # In training, heading_init = identity so root_orn = robot_rot.
                terms = {}
                for name in term_names:
                    if "motion_command" in name:
                        terms[name] = np.clip(np.concatenate([ref_dof_pos, ref_dof_vel]), -clip_obs, clip_obs)
                    elif "motion_ref_vel_b" in name:
                        terms[name] = np.clip(root_orn.inv().apply(ref_root_vel_w), -clip_obs, clip_obs)
                    elif "motion_ref_pos_b" in name:
                        terms[name] = np.clip(
                            root_orn.inv().apply(ref_root_pos_aligned - robot_pos), -clip_obs, clip_obs
                        )
                    elif "motion_ref_ori_b" in name:
                        terms[name] = np.clip(rel_rot.as_matrix()[:, :2].flatten(), -clip_obs, clip_obs)
                    elif "projected_gravity" in name:
                        terms[name] = np.clip(root_orn.inv().apply(np.array([0, 0, -1.0])), -clip_obs, clip_obs)
                    elif "base_lin_vel" in name:
                        terms[name] = np.clip(root_orn.inv().apply(robot_lin_vel), -clip_obs, clip_obs)
                    elif "base_ang_vel" in name:
                        terms[name] = np.clip(omega, -clip_obs, clip_obs)
                    elif "dof_pos" in name:
                        terms[name] = np.clip(q_isaac - default_q, -clip_obs, clip_obs)
                    elif "dof_vel" in name:
                        terms[name] = np.clip(dq_isaac, -clip_obs, clip_obs)
                    elif "actions" in name:
                        terms[name] = np.clip(last_action.copy(), -clip_obs, clip_obs)
                    else:
                        terms[name] = np.zeros(1)

                # Build obs with history
                parts = []
                for name in term_names:
                    buf = history[name]
                    buf.append(terms[name].copy())
                    h = list(buf)
                    if len(h) < history_length:
                        h = [np.zeros_like(h[0])] * (history_length - len(h)) + h
                    parts.append(np.concatenate(h[-history_length:]))
                obs = np.concatenate(parts).reshape(1, -1).astype(np.float32)

                inputs = {"obs": obs}
                if has_time_step:
                    inputs["time_step"] = np.array([[float(ticker)]], dtype=np.float32)
                ticker += 1

                action_isaac = session.run(None, inputs)[0].flatten()
                last_action = action_isaac.copy()
                target_q = default_q + action_isaac * action_scale

                results.append((data.qpos[:3].copy(), data.qpos[3:7].copy(), data.qpos[7:36].copy()))

            # PD control
            tau = (target_q - data.qpos[7:36]) * kp - data.qvel[6:35] * kd
            data.ctrl[:] = np.clip(tau, -torque_limit, torque_limit)
            mujoco.mj_step(model, data)

            if data.qpos[2] < 0.3:
                break

        if data.qpos[2] < 0.3:
            print(f"  WBT fell at step {policy_step}")
            break

    return results


def main():
    parser = argparse.ArgumentParser(description="Viser teleop replay (standalone)")
    parser.add_argument("--pkl", default="holosoma_extensions/src/extensions/teleop/data/pico_body_recording.pkl")
    parser.add_argument(
        "--urdf", default="holosoma_extensions/src/extensions/wbt_wrappers/wbt_wrappers_inference/urdf/g1_29dof.urdf"
    )
    parser.add_argument(
        "--xml",
        default="holosoma_extensions/thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1/scenes/scene_g1_29dof_wbt_plane.xml",
    )
    parser.add_argument(
        "--tml_onnx",
        default="holosoma_extensions/src/extensions/wbt_wrappers/wbt_wrappers_inference/models/tml_general.onnx",
    )
    parser.add_argument("--wbt_onnx", default="/tmp/ckpt/model_29999.onnx")
    parser.add_argument("--wbt_local_onnx", default="/tmp/ckpt_local_lafan1/model_10500.onnx")
    parser.add_argument("--wbt_global_onnx", default="/tmp/ckpt_global_lafan1/model_10500.onnx")
    parser.add_argument("--mm_onnx_dir", default="/tmp/mm_lafan1_1n_ckpt", help="Dir with MaskedMimic .onnx + .pt")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--skip_first", type=int, default=1, help="Skip first N frames (T-pose calibration)")
    args = parser.parse_args()

    # Load pico recording
    with open(args.pkl, "rb") as f:
        rec = pickle.load(f)
    all_poses = rec["body_joint_poses"]
    ts = rec["timestamps_ns"]
    N_total = len(ts)
    pico_duration = (ts[-1] - ts[0]) / 1e9
    pico_fps = (N_total - 1) / pico_duration if pico_duration > 0 else 5.0
    print(f"PICO recording: {N_total} frames, {pico_duration:.1f}s, {pico_fps:.1f} fps")

    # Step 1: Retarget all frames
    print("\nRetargeting...")
    retargeter = SMPLRetargeter(args.urdf, dt=1.0 / pico_fps)
    all_joints, all_root_pos, all_root_quat, all_pico, all_root_body_quat = precompute_retarget(
        retargeter, all_poses, skip_first=args.skip_first
    )
    N_retarget = len(all_joints)
    print(f"  height_ratio={retargeter._height_ratio:.4f}")

    # Step 2: Run TML policy — retarget at policy rate (50Hz), matching original teleop_wbt.py
    policy_fps = 50.0
    pico_poses_for_tml = all_poses[args.skip_first :]  # same frames retargeter saw
    tml_retargeter = SMPLRetargeter(args.urdf, dt=1.0 / policy_fps)
    print(f"\nRunning TML policy: {args.tml_onnx}")
    tml_results = run_tml_policy(args.tml_onnx, pico_poses_for_tml, tml_retargeter, args.xml, policy_fps=policy_fps)
    print(f"  TML: {len(tml_results)} frames")

    # Step 2b: Run Dense WBT policy (SONIC)
    wbt_results = []
    if args.wbt_onnx and os.path.exists(args.wbt_onnx):
        print(f"\nRunning Dense WBT (SONIC): {args.wbt_onnx}")
        wbt_results = run_wbt_policy(
            args.wbt_onnx, all_joints, all_root_pos, all_root_quat, args.xml, policy_fps=50.0, pico_fps=pico_fps
        )
        print(f"  WBT-SONIC: {len(wbt_results)} frames")
    else:
        print(f"\nSkipping Dense WBT-SONIC (no onnx at {args.wbt_onnx})")

    # Step 2c: Run Dense WBT Local (LAFAN1)
    wbt_local_results = []
    if args.wbt_local_onnx and os.path.exists(args.wbt_local_onnx):
        print(f"\nRunning Dense WBT Local (LAFAN1): {args.wbt_local_onnx}")
        wbt_local_results = run_wbt_policy(
            args.wbt_local_onnx, all_joints, all_root_pos, all_root_quat, args.xml, policy_fps=50.0, pico_fps=pico_fps
        )
        print(f"  WBT-Local: {len(wbt_local_results)} frames")
    else:
        print(f"\nSkipping Dense WBT-Local (no onnx at {args.wbt_local_onnx})")

    # Step 2d: Run Dense WBT Global (LAFAN1)
    wbt_global_results = []
    if args.wbt_global_onnx and os.path.exists(args.wbt_global_onnx):
        print(f"\nRunning Dense WBT Global (LAFAN1): {args.wbt_global_onnx}")
        wbt_global_results = run_wbt_policy(
            args.wbt_global_onnx, all_joints, all_root_pos, all_root_quat, args.xml, policy_fps=50.0, pico_fps=pico_fps
        )
        print(f"  WBT-Global: {len(wbt_global_results)} frames")
    else:
        print(f"\nSkipping Dense WBT-Global (no onnx at {args.wbt_global_onnx})")

    # Step 2c: Run MaskedMimic policy
    mm_results = []
    if args.mm_onnx_dir and os.path.isdir(args.mm_onnx_dir):
        import glob as _glob

        if _glob.glob(os.path.join(args.mm_onnx_dir, "*.onnx")):
            print("\nPrecomputing reference body FK for MaskedMimic...")
            ref_body_pos, ref_body_quat, ref_fk_model = _precompute_ref_body_fk(
                all_joints, all_root_pos, all_root_quat, args.xml
            )
            print(f"  Body FK: {ref_body_pos.shape[0]} frames, {ref_body_pos.shape[1]} bodies")

            print(f"\nRunning MaskedMimic policy: {args.mm_onnx_dir}")
            mm_results = run_maskedmimic_policy(
                args.mm_onnx_dir,
                all_joints,
                all_root_pos,
                all_root_quat,
                args.xml,
                ref_body_pos,
                ref_body_quat,
                ref_fk_model,
                policy_fps=50.0,
                pico_fps=pico_fps,
            )
            print(f"  MaskedMimic: {len(mm_results)} frames")
        else:
            print(f"\nSkipping MaskedMimic (no onnx in {args.mm_onnx_dir})")
    else:
        print(f"\nSkipping MaskedMimic (no dir at {args.mm_onnx_dir})")

    # Step 3: Setup viser
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=20.0, height=20.0, plane="xy")

    # URDF mesh robots via viser.extras.ViserUrdf
    from viser.extras import ViserUrdf as _ViserUrdf

    urdf_path = Path(args.urdf)

    # Y offsets for separation — 7 columns
    y_pico = -4.5
    y_retarget = -3.0
    y_tml = -1.5
    y_wbt = 0.0
    y_wbt_local = 1.5
    y_wbt_global = 3.0
    y_mm = 4.5

    # Create root frames for each robot mesh
    retarget_frame = server.scene.add_frame("/retarget", show_axes=False, position=(0.0, y_retarget, 0.0))
    tml_frame = server.scene.add_frame("/tml", show_axes=False, position=(0.0, y_tml, 0.0))
    wbt_frame = server.scene.add_frame("/wbt", show_axes=False, position=(0.0, y_wbt, 0.0))
    wbt_local_frame = server.scene.add_frame("/wbt_local", show_axes=False, position=(0.0, y_wbt_local, 0.0))
    wbt_global_frame = server.scene.add_frame("/wbt_global", show_axes=False, position=(0.0, y_wbt_global, 0.0))
    mm_frame = server.scene.add_frame("/mm", show_axes=False, position=(0.0, y_mm, 0.0))

    # Load URDF meshes
    retarget_urdf = _ViserUrdf(server, urdf_or_path=urdf_path, root_node_name="/retarget")
    tml_urdf = _ViserUrdf(server, urdf_or_path=urdf_path, root_node_name="/tml")
    wbt_urdf = _ViserUrdf(server, urdf_or_path=urdf_path, root_node_name="/wbt")
    wbt_local_urdf = _ViserUrdf(server, urdf_or_path=urdf_path, root_node_name="/wbt_local")
    wbt_global_urdf = _ViserUrdf(server, urdf_or_path=urdf_path, root_node_name="/wbt_global")
    mm_urdf = _ViserUrdf(server, urdf_or_path=urdf_path, root_node_name="/mm")

    # Get joint name list (same order as MuJoCo qpos for this URDF)
    urdf_joint_names = list(retarget_urdf.get_actuated_joint_limits().keys())
    print(f"  URDF joints: {len(urdf_joint_names)}")

    # Labels
    server.scene.add_label("/pico_label", "PICO Raw", position=(0.0, y_pico, 1.8))
    server.scene.add_label("/retarget_label", "Retargeted (IK)", position=(0.0, y_retarget, 1.8))
    server.scene.add_label("/tml_label", "TML Tracker", position=(0.0, y_tml, 1.8))
    server.scene.add_label("/wbt_label", "WBT-SONIC", position=(0.0, y_wbt, 1.8))
    server.scene.add_label("/wbt_local_label", "WBT-Local-LAFAN1", position=(0.0, y_wbt_local, 1.8))
    server.scene.add_label("/wbt_global_label", "WBT-Global-LAFAN1", position=(0.0, y_wbt_global, 1.8))
    server.scene.add_label("/mm_label", "MaskedMimic", position=(0.0, y_mm, 1.8))

    # TML now has 1:1 with PICO frames (retargeted at policy rate, one frame per step)
    # WBT/MM still use pre-computed retarget at pico_fps, mapped to policy fps
    N_frames = N_retarget
    wbt_per_pico = policy_fps / pico_fps  # ~10 WBT/MM frames per PICO frame
    print(f"\nViser ready at http://localhost:{args.port}")
    print(f"Total frames: {N_frames} (PICO/retarget/TML)")
    print(
        f"  WBT-SONIC: {len(wbt_results)}, WBT-Local: {len(wbt_local_results)}, WBT-Global: {len(wbt_global_results)}, MM: {len(mm_results)}"
    )

    # GUI controls
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=N_frames - 1, step=1, initial_value=0)
        playing_cb = server.gui.add_checkbox("Play", initial_value=False)
        speed_slider = server.gui.add_slider("Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0)

    def _draw_pico(positions, y_offset):
        for i in range(24):
            pos = positions[i].copy()
            pos[1] += y_offset
            server.scene.add_icosphere(f"/pico/j{i}", radius=0.025, color=(200, 100, 255), position=tuple(pos))
            pid = _SMPL_PARENTS[i]
            if pid >= 0:
                pp = positions[pid].copy()
                pp[1] += y_offset
                server.scene.add_spline_catmull_rom(
                    f"/pico/l{i}", positions=(tuple(pp), tuple(pos)), color=(200, 100, 255), line_width=2.0
                )

    def update_frame(idx):
        """idx is in PICO frame space (0 to N_retarget-1)."""
        idx = max(0, min(idx, N_frames - 1))

        ref_root_pos = all_root_pos[idx]

        # 1. PICO skeleton — heading-removed, anchored to retarget root
        # all_pico[idx] is pelvis-relative + heading-removed (from precompute_retarget)
        pico_display = all_pico[idx] + ref_root_pos  # anchor to retarget root
        _draw_pico(pico_display, y_pico)

        # 2. Retargeted G1 mesh — yaw removed from root quat (matching original)
        root_body_quat_xyzw = all_root_body_quat[idx]
        euler_zyx = Rotation.from_quat(root_body_quat_xyzw).as_euler("ZYX")
        euler_zyx[0] = 0.0  # zero out yaw
        noyaw_quat_xyzw = Rotation.from_euler("ZYX", euler_zyx).as_quat()
        noyaw_quat_wxyz = (noyaw_quat_xyzw[3], noyaw_quat_xyzw[0], noyaw_quat_xyzw[1], noyaw_quat_xyzw[2])
        retarget_frame.position = (
            float(ref_root_pos[0]),
            float(ref_root_pos[1]) + y_retarget,
            float(ref_root_pos[2]) - 0.05,
        )  # compensate URDF FK offset
        retarget_frame.wxyz = noyaw_quat_wxyz
        retarget_urdf.update_cfg(all_joints[idx].tolist())

        # 3. TML policy mesh — 1:1 with PICO (retargeted at policy rate)
        if idx < len(tml_results):
            t_pos, t_quat, t_joints = tml_results[idx]
            tml_frame.position = (float(t_pos[0]), float(t_pos[1]) + y_tml, float(t_pos[2]))
            tml_frame.wxyz = (1.0, 0.0, 0.0, 0.0)
            tml_urdf.update_cfg(t_joints.tolist())

        # 4. Dense WBT-SONIC policy mesh
        wbt_idx = int(idx * wbt_per_pico)
        if wbt_idx < len(wbt_results):
            w_pos, w_quat, w_joints = wbt_results[wbt_idx]
            wbt_frame.position = (float(w_pos[0]), float(w_pos[1]) + y_wbt, float(w_pos[2]))
            wbt_frame.wxyz = (1.0, 0.0, 0.0, 0.0)
            wbt_urdf.update_cfg(w_joints.tolist())

        # 5. Dense WBT-Local-LAFAN1 policy mesh
        if wbt_idx < len(wbt_local_results):
            w_pos, w_quat, w_joints = wbt_local_results[wbt_idx]
            wbt_local_frame.position = (float(w_pos[0]), float(w_pos[1]) + y_wbt_local, float(w_pos[2]))
            wbt_local_frame.wxyz = (1.0, 0.0, 0.0, 0.0)
            wbt_local_urdf.update_cfg(w_joints.tolist())

        # 6. Dense WBT-Global-LAFAN1 policy mesh
        if wbt_idx < len(wbt_global_results):
            w_pos, w_quat, w_joints = wbt_global_results[wbt_idx]
            wbt_global_frame.position = (float(w_pos[0]), float(w_pos[1]) + y_wbt_global, float(w_pos[2]))
            wbt_global_frame.wxyz = (1.0, 0.0, 0.0, 0.0)
            wbt_global_urdf.update_cfg(w_joints.tolist())

        # 7. MaskedMimic policy mesh
        mm_idx = int(idx * wbt_per_pico)
        if mm_idx < len(mm_results):
            m_pos, m_quat, m_joints = mm_results[mm_idx]
            mm_frame.position = (float(m_pos[0]), float(m_pos[1]) + y_mm, float(m_pos[2]))
            mm_frame.wxyz = (1.0, 0.0, 0.0, 0.0)
            mm_urdf.update_cfg(m_joints.tolist())

    # Main loop
    accumulator = 0.0
    last_time = time.time()
    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now

            if playing_cb.value:
                accumulator += dt * speed_slider.value * pico_fps
                if accumulator >= 1.0:
                    steps = int(accumulator)
                    accumulator -= steps
                    new_val = min(frame_slider.value + steps, N_frames - 1)
                    frame_slider.value = new_val

            update_frame(int(frame_slider.value))
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
