#!/usr/bin/env python3
"""Viser replay: raw PICO + retargeted G1 + TML WBT + Dense WBT side-by-side.

Uses ViserUrdf for full G1 robot mesh rendering.
Runs two ONNX policies (TML general and Dense WBT) in MuJoCo to produce
tracked joint trajectories, then replays everything in viser.

Usage:
    python viser_pico_replay.py \
        --pkl /tmp/pico_body_recording.pkl \
        --urdf .../g1_29dof.urdf \
        --xml .../scene_g1_29dof_wbt_plane.xml \
        --tml_onnx .../tml_general.onnx \
        --dense_onnx_dir /tmp/ckpt
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import deque
from pathlib import Path

import glob as _glob
import mujoco
import numpy as np
import onnxruntime as ort
import viser
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf
from wbt_training.config_values.robot_config import (
    ISAACLAB_TO_MUJOCO_DOF as ISAAC_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB_DOF as MUJOCO_TO_ISAAC,
)

# SMPL kinematic tree parent indices for 24 joints
_SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# Pico Y-up to robot Z-up
_YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

_NORM_CLIP = 3.0


def _load_normalizer_stats(onnx_dir, onnx_basename):
    """Load normalizer mean/std from .pt file if available."""
    pt_name = os.path.splitext(onnx_basename)[0] + ".pt"
    pt_path = os.path.join(onnx_dir, pt_name)
    if not os.path.exists(pt_path):
        return None, None
    import torch

    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    if "running_mean_obs" in state and "running_var_obs" in state:
        mean = state["running_mean_obs"].numpy().flatten()
        var = state["running_var_obs"].numpy().flatten()
        std = np.sqrt(var + 1e-8)
        return mean, std
    return None, None


def _yaw(qpos):
    q = qpos[3:7]
    return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def _quat_to_rpy(q_wxyz):
    """wxyz quaternion -> (roll, pitch, yaw)."""
    w, x, y, z = q_wxyz
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _rpy_to_quat_wxyz(roll, pitch, yaw):
    """(roll, pitch, yaw) -> wxyz quaternion."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def _quat_inv_wxyz(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_rotate_inv_wxyz(q_wxyz, v):
    """Rotate vector v by inverse of quaternion q (wxyz)."""
    q_inv = _quat_inv_wxyz(q_wxyz)
    r = Rotation.from_quat([q_inv[1], q_inv[2], q_inv[3], q_inv[0]])
    return r.apply(v)


def _subtract_frame_transforms_wxyz(q1_wxyz, q2_wxyz):
    """q1^-1 * q2 (relative rotation from frame 1 to frame 2)."""
    return _quat_mul_wxyz(_quat_inv_wxyz(q1_wxyz), q2_wxyz)


def _matrix_from_quat_wxyz(q_wxyz):
    """wxyz quaternion -> 3x3 rotation matrix."""
    r = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return r.as_matrix()


def run_tml_rollout(onnx_path, ref_npz_path, xml_path, max_frames=600):
    """Run TML WBT ONNX policy in MuJoCo.

    Starts upright with reference first-frame joints. Uses simple relative
    rotation for anchor_ori (no heading removal) and MuJoCo-order command.
    """
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map

    default_pos_isaac = np.array([float(x) for x in meta["default_joint_pos"].split(",")])
    action_scale_isaac = np.array([float(x) for x in meta["action_scale"].split(",")])
    kp = np.array([float(x) for x in meta["joint_stiffness"].split(",")])
    kd = np.array([float(x) for x in meta["joint_damping"].split(",")])

    default_q_mj = default_pos_isaac[ISAAC_TO_MUJOCO]
    action_scale_mj = action_scale_isaac[ISAAC_TO_MUJOCO]
    kp_mj = kp[ISAAC_TO_MUJOCO]
    kd_mj = kd[ISAAC_TO_MUJOCO]

    ref = np.load(ref_npz_path)
    motion_length = ref["joint_pos"].shape[0]
    ref_dof_pos = ref["joint_pos"][:, 7:]
    ref_dof_vel = ref["joint_vel"][:, 6:]
    ref_pelvis_quat_wxyz = ref["body_quat_w"][:, 0]

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    sim_dt = 0.005
    model.opt.timestep = sim_dt
    control_dt = 0.02
    steps_per_control = int(control_dt / sim_dt)

    torque_limit = np.array([model.actuator_forcerange[i][1] for i in range(model.nu)])
    if np.all(torque_limit == 0):
        torque_limit = np.ones(29) * 200.0

    # Start upright with reference first-frame joints
    data.qpos[0:3] = [0.0, 0.0, 0.793]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:36] = ref_dof_pos[0]
    mujoco.mj_forward(model, data)

    last_action = np.zeros(29)
    target_q = default_q_mj.copy()
    results = []

    for ticker in range(max_frames if max_frames > 0 else motion_length):
        mid = min(ticker, motion_length - 1)

        q_mj = data.qpos[7:36]
        dq_mj = data.qvel[6:35]
        base_quat_wxyz = data.qpos[3:7]
        omega = data.qvel[3:6]

        # Relative rotation from robot to reference (no heading removal)
        base_quat_xyzw = base_quat_wxyz[[1, 2, 3, 0]]
        robot_rot = Rotation.from_quat(base_quat_xyzw)
        robot_rot_inv = robot_rot.inv()
        ref_quat_xyzw = ref_pelvis_quat_wxyz[mid][[1, 2, 3, 0]]
        ref_rot = Rotation.from_quat(ref_quat_xyzw)
        rel_rot = robot_rot_inv * ref_rot
        anchor_ori = rel_rot.as_matrix()[:, :2].flatten()

        gravity = robot_rot_inv.apply(np.array([0, 0, -1.0]))
        command = np.concatenate([ref_dof_pos[mid], ref_dof_vel[mid]])

        dof_pos = q_mj[MUJOCO_TO_ISAAC] - default_pos_isaac
        dof_vel = dq_mj[MUJOCO_TO_ISAAC]

        obs = (
            np.concatenate([command, anchor_ori, gravity, omega, dof_pos, dof_vel, last_action])
            .reshape(1, -1)
            .astype(np.float32)
        )

        action_isaac = session.run(None, {"obs": obs})[0].flatten()
        last_action = action_isaac.copy()
        target_q = default_q_mj + action_isaac[ISAAC_TO_MUJOCO] * action_scale_mj

        results.append((data.qpos[0:3].copy(), data.qpos[3:7].copy(), data.qpos[7:36].copy()))

        if data.qpos[2] < 0.3:
            break

        for _ in range(steps_per_control):
            tau = (target_q - data.qpos[7:36]) * kp_mj - data.qvel[6:35] * kd_mj
            data.ctrl[:] = np.clip(tau, -torque_limit, torque_limit)
            mujoco.mj_step(model, data)

    return results


def run_policy_rollout(onnx_dir, ref_npz_path, xml_path, max_frames=0):
    """Run a holosoma WBT ONNX policy in MuJoCo (has experiment_config metadata).

    Returns: list of (root_pos(3), root_quat_wxyz(4), joint_angles(29)) per control step.
    """
    onnx_files = sorted(_glob.glob(os.path.join(onnx_dir, "*.onnx")))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX files in {onnx_dir}")
    onnx_path = onnx_files[-1]
    onnx_basename = os.path.basename(onnx_path)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    exp_config = json.loads(meta["experiment_config"])

    norm_mean, norm_std = _load_normalizer_stats(onnx_dir, onnx_basename)

    kp = np.array([float(x) for x in meta["kp"].strip("[]").split(",")])
    kd = np.array([float(x) for x in meta["kd"].strip("[]").split(",")])
    torque_limit = np.array(exp_config["robot"]["dof_effort_limit_list"], dtype=np.float64)

    default_angles = exp_config["robot"]["init_state"]["default_joint_angles"]
    dof_names = exp_config["robot"]["dof_names"]
    default_q = np.array([default_angles[name] for name in dof_names])
    action_scale = np.array([float(x) for x in meta["action_scale"].strip("[]").split(",")])

    sim_fps = exp_config["simulator"]["config"]["sim"]["fps"]
    decimation = exp_config["simulator"]["config"]["sim"]["control_decimation"]
    sim_dt = 1.0 / sim_fps
    control_dt = decimation / sim_fps

    input_names = [inp.name for inp in session.get_inputs()]
    has_time_step = "time_step" in input_names
    _obs_dim = session.get_inputs()[0].shape[1]  # noqa: F841

    obs_cfg = exp_config["observation"]["groups"]["actor_obs"]
    history_length = obs_cfg.get("history_length", 1)
    clip_obs = exp_config["observation"].get("clip_observations", 100.0)

    # Load reference motion
    ref = np.load(ref_npz_path)
    motion_length = ref["joint_pos"].shape[0]
    ref_dof_pos = ref["joint_pos"][:, 7:]
    ref_dof_vel = ref["joint_vel"][:, 6:]
    _ref_root_pos = ref["joint_pos"][:, :3]  # noqa: F841
    _ref_root_quat_wxyz = ref["joint_pos"][:, 3:7]  # noqa: F841
    ref_pelvis_quat_wxyz = ref["body_quat_w"][:, 0]

    # Setup MuJoCo
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = sim_dt
    steps_per_control = max(1, int(control_dt / sim_dt))

    # Initial pose: upright standing at origin
    data.qpos[7:36] = default_q.copy()
    data.qpos[0:3] = [0.0, 0.0, 0.793]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # upright, identity quat
    mujoco.mj_forward(model, data)

    term_names = sorted(obs_cfg.get("terms", {}).keys())
    if not term_names:
        term_names = [
            "obs0_motion_command",
            "obs1_motion_ref_ori_b",
            "obs2_projected_gravity",
            "obs3_base_ang_vel",
            "obs4_dof_pos",
            "obs5_dof_vel",
            "obs6_actions",
        ]
    history = {n: deque(maxlen=history_length) for n in term_names}
    last_action = np.zeros(29)

    ticker = 0
    video_fps = int(1.0 / control_dt)
    effective_max = motion_length if max_frames <= 0 else max_frames
    total_steps = int(effective_max / video_fps * (1.0 / sim_dt))

    results = []  # list of (root_pos, root_quat_wxyz, joints_29)

    for sim_step in range(total_steps):
        if sim_step % steps_per_control == 0:
            q_isaac = data.qpos[7:36]
            dq_isaac = data.qvel[6:35]
            robot_quat_xyzw = data.qpos[3:7][[1, 2, 3, 0]]
            omega = data.qvel[3:6]

            mid = min(ticker, motion_length - 1)
            robot_rot = Rotation.from_quat(robot_quat_xyzw)
            robot_rot_inv = robot_rot.inv()

            ref_quat_wxyz = ref_pelvis_quat_wxyz[mid]
            ref_quat_xyzw = ref_quat_wxyz[[1, 2, 3, 0]]
            ref_rot = Rotation.from_quat(ref_quat_xyzw)
            rel_rot = robot_rot_inv * ref_rot

            terms = {}
            for name in term_names:
                if "motion_command" in name:
                    terms[name] = np.clip(np.concatenate([ref_dof_pos[mid], ref_dof_vel[mid]]), -clip_obs, clip_obs)
                elif "motion_ref_ori_b" in name:
                    terms[name] = np.clip(rel_rot.as_matrix()[:, :2].flatten(), -clip_obs, clip_obs)
                elif "projected_gravity" in name:
                    terms[name] = np.clip(robot_rot_inv.apply(np.array([0, 0, -1.0])), -clip_obs, clip_obs)
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

            parts = []
            for name in term_names:
                buf = history[name]
                buf.append(terms[name].copy())
                h = list(buf)
                if len(h) < history_length:
                    h = [np.zeros_like(h[0])] * (history_length - len(h)) + h
                parts.append(np.concatenate(h[-history_length:]))
            obs = np.concatenate(parts).reshape(1, -1).astype(np.float32)

            if norm_mean is not None:
                obs = np.clip((obs - norm_mean) / norm_std, -_NORM_CLIP, _NORM_CLIP).astype(np.float32)

            inputs = {"obs": obs}
            if has_time_step:
                inputs["time_step"] = np.array([[float(ticker % motion_length)]], dtype=np.float32)

            ort_outs = session.run(None, inputs)
            action = ort_outs[0].flatten()
            last_action = action.copy()
            ticker += 1

            target_q = default_q + action * action_scale

            # Save state
            results.append(
                (
                    data.qpos[0:3].copy(),
                    data.qpos[3:7].copy(),  # wxyz
                    data.qpos[7:36].copy(),
                )
            )

            if max_frames > 0 and len(results) >= max_frames:
                break

            if data.qpos[2] < 0.3:
                break

        # PD control
        tau = (target_q - data.qpos[7:36]) * kp - data.qvel[6:35] * kd
        data.ctrl[:] = np.clip(tau, -torque_limit, torque_limit)
        mujoco.mj_step(model, data)

    return results


def main():
    parser = argparse.ArgumentParser(description="Viser: PICO + retarget + TML + Dense WBT")
    parser.add_argument("--pkl", required=True, help="PICO recording pickle")
    parser.add_argument("--urdf", required=True, help="Fixed-base G1 URDF")
    parser.add_argument("--xml", required=True, help="MuJoCo scene XML (free-joint)")
    parser.add_argument("--tml_onnx", required=True, help="TML general ONNX file")
    parser.add_argument("--dense_onnx_dir", required=True, help="Dense WBT checkpoint dir")
    parser.add_argument("--npz", default=None, help="Pre-converted holosoma NPZ (skip retargeting)")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument("--max_frames", type=int, default=600, help="Max policy frames")
    parser.add_argument("--skip-first", type=int, default=1, help="Skip N PICO calibration frames")
    args = parser.parse_args()

    # ---- Step 1: Retarget PICO to holosoma NPZ ----
    npz_path = args.npz
    if npz_path is None:
        from retargeting.smpl_retargeter import SMPLRetargeter
        import pico_to_holosoma_npz as p2h

        with open(args.pkl, "rb") as f:
            rec = pickle.load(f)
        all_poses = rec["body_joint_poses"]
        ts = rec["timestamps_ns"]
        N = len(ts)
        duration = (ts[-1] - ts[0]) / 1e9
        input_fps = (N - 1) / duration if duration > 0 else 5.0
        print(f"PICO: {N} frames, {duration:.1f}s, {input_fps:.1f} FPS")

        print("Retargeting PICO → holosoma NPZ...")
        q_joints, root_pos, root_quat_wxyz = p2h.retarget_sequence(all_poses, args.xml, input_fps, urdf_path=args.urdf)

        root_pos, root_quat_wxyz, q_joints, T = p2h.interpolate_to_fps(
            root_pos, root_quat_wxyz, q_joints, input_fps, 50
        )
        print(f"  {T} frames at 50 FPS")

        npz_path = "/tmp/pico_viser_retargeted.npz"
        p2h.run_fk_and_save(root_pos, root_quat_wxyz, q_joints, 50, args.xml, npz_path)
    print(f"Using NPZ: {npz_path}")

    # ---- Step 2: Load retarget data for PICO skeleton display ----
    from retargeting.smpl_retargeter import SMPLRetargeter

    with open(args.pkl, "rb") as f:
        rec = pickle.load(f)
    all_poses = rec["body_joint_poses"][args.skip_first :]
    ts = rec["timestamps_ns"][args.skip_first :]
    N_pico = len(ts)
    pico_duration = (ts[-1] - ts[0]) / 1e9
    pico_fps = (N_pico - 1) / pico_duration if pico_duration > 0 else 5.0

    retargeter = SMPLRetargeter(args.urdf, dt=1.0 / pico_fps)

    print("Pre-computing PICO skeletons + retarget...")
    pico_scaled_all = []
    retarget_joints_all = []
    retarget_root_pos_all = []
    retarget_root_quat_wxyz_all = []

    for i in range(N_pico):
        frame = all_poses[i]
        q_j, dq, root_orn_wxyz = retargeter.retarget(frame)
        root_pos = retargeter._prev_q[:3].copy()
        root_quat_xyzw = retargeter._prev_q[3:7].copy()
        root_quat_wxyz = np.array([root_quat_xyzw[3], root_quat_xyzw[0], root_quat_xyzw[1], root_quat_xyzw[2]])

        retarget_joints_all.append(q_j.copy())
        retarget_root_pos_all.append(root_pos)
        retarget_root_quat_wxyz_all.append(root_quat_wxyz)

        pico_zup = frame[:, :3] @ _YUP_TO_ZUP.T
        pico_scaled = retargeter._scale_positions(pico_zup)
        pico_scaled = SMPLRetargeter._anchor_to_ground(pico_scaled)
        pico_scaled_all.append(pico_scaled)

        if (i + 1) % 100 == 0 or i == N_pico - 1:
            print(f"  {i + 1}/{N_pico}")

    print(f"Retargeter: g1_leg={retargeter._g1_leg:.3f}m, height_ratio={retargeter._height_ratio:.4f}")

    # ---- Step 3: Run TML policy ----
    print(f"\nRunning TML policy: {args.tml_onnx}")
    tml_results = run_tml_rollout(args.tml_onnx, npz_path, args.xml, args.max_frames)
    print(f"  TML: {len(tml_results)} frames")

    # ---- Step 4: Run Dense WBT policy ----
    print(f"\nRunning Dense WBT policy: {args.dense_onnx_dir}")
    dense_results = run_policy_rollout(args.dense_onnx_dir, npz_path, args.xml, args.max_frames)
    print(f"  Dense: {len(dense_results)} frames")

    # ---- Step 5: Setup Viser ----
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=12.0, height=12.0, plane="xy")

    urdf_path = Path(args.urdf).resolve()

    # Y-offsets for 4 views
    y_pico = -3.0
    y_retarget = -1.0
    y_tml = 1.0
    y_dense = 3.0

    # Create ViserUrdf instances for 3 robots (retarget, TML, Dense)
    retarget_root = server.scene.add_frame("/retarget", show_axes=False)
    retarget_urdf = ViserUrdf(server, urdf_or_path=urdf_path, root_node_name="/retarget")

    tml_root = server.scene.add_frame("/tml", show_axes=False)
    tml_urdf = ViserUrdf(server, urdf_or_path=urdf_path, root_node_name="/tml")

    dense_root = server.scene.add_frame("/dense", show_axes=False)
    dense_urdf = ViserUrdf(server, urdf_or_path=urdf_path, root_node_name="/dense")

    urdf_joint_names = list(retarget_urdf.get_actuated_joint_limits().keys())
    urdf_joint_to_idx = {n: i for i, n in enumerate(urdf_joint_names)}

    # MuJoCo joint names for mapping
    mj_model = mujoco.MjModel.from_xml_path(str(urdf_path))
    mj_joint_names = []
    for j in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name:
            mj_joint_names.append(name)
    mj_to_urdf = [urdf_joint_to_idx.get(jn, -1) for jn in mj_joint_names]

    # MuJoCo scene model joint names (free-joint, 36 qpos)
    scene_model = mujoco.MjModel.from_xml_path(args.xml)
    scene_dof_names = []
    for j in range(scene_model.njnt):
        name = mujoco.mj_id2name(scene_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name:
            scene_dof_names.append(name)
    # For policy results: qpos[7:36] maps to scene joints (skip free joint)
    scene_to_urdf = [urdf_joint_to_idx.get(n, -1) for n in scene_dof_names[1:]]  # skip root_joint

    # Labels
    server.scene.add_label("/pico_label", "PICO Raw", position=(0.0, y_pico, 1.8))
    server.scene.add_label("/retarget_label", "Retargeted (IK)", position=(0.0, y_retarget, 1.8))
    server.scene.add_label("/tml_label", "TML WBT", position=(0.0, y_tml, 1.8))
    server.scene.add_label("/dense_label", "Dense WBT (ours)", position=(0.0, y_dense, 1.8))

    # Determine frame count — use max so all views play their full length
    policy_fps = 50.0
    N_frames = max(len(tml_results), len(dense_results), N_pico)
    print(f"\nTotal frames for replay: {N_frames}")
    print(f"  PICO: {N_pico} | Retarget: {N_pico} | TML: {len(tml_results)} | Dense: {len(dense_results)}")

    # GUI controls
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=N_frames - 1, step=1, initial_value=0)
        playing_cb = server.gui.add_checkbox("Play", initial_value=False)
        speed_slider = server.gui.add_slider("Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0)

    with server.gui.add_folder("Info"):
        server.gui.add_text(
            "Stats",
            initial_value=(
                f"PICO: {N_pico}f/{pico_duration:.1f}s | "
                f"TML: {len(tml_results)}f | Dense: {len(dense_results)}f | "
                f"h_ratio={retargeter._height_ratio:.3f}"
            ),
            disabled=True,
        )

    def _update_urdf(root_handle, viser_urdf_obj, root_pos, root_quat_wxyz, joints_29, y_offset, joint_map):
        """Update a ViserUrdf skeleton."""
        pos = root_pos.copy()
        pos[1] += y_offset
        root_handle.position = tuple(pos.tolist())
        root_handle.wxyz = tuple(root_quat_wxyz.tolist())

        cfg = [0.0] * len(urdf_joint_names)
        for src_idx, urdf_idx in enumerate(joint_map):
            if urdf_idx >= 0 and src_idx < len(joints_29):
                cfg[urdf_idx] = float(joints_29[src_idx])
        viser_urdf_obj.update_cfg(cfg)

    def _remove_yaw(quat_wxyz):
        """Remove yaw from quaternion so robot faces fixed direction."""
        rot = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        euler = rot.as_euler("ZYX")
        euler[0] = 0.0
        noyaw_xyzw = Rotation.from_euler("ZYX", euler).as_quat()
        return np.array([noyaw_xyzw[3], noyaw_xyzw[0], noyaw_xyzw[1], noyaw_xyzw[2]])

    def update_frame(idx):
        idx = max(0, min(idx, N_frames - 1))

        # Map policy frame index (50 fps) to PICO frame index (variable fps)
        pico_idx = min(int(idx * pico_fps / policy_fps), N_pico - 1)

        # 1. Raw PICO skeleton (magenta stickman)
        pico_pos = pico_scaled_all[pico_idx].copy()
        for i in range(24):
            pos = pico_pos[i].copy()
            pos[1] += y_pico
            server.scene.add_icosphere(f"/pico/j{i}", radius=0.025, color=(200, 100, 255), position=tuple(pos))
            pid = _SMPL_PARENTS[i]
            if pid >= 0:
                pp = pico_pos[pid].copy()
                pp[1] += y_pico
                server.scene.add_spline_catmull_rom(
                    f"/pico/l{i}", positions=(tuple(pp), tuple(pos)), color=(200, 100, 255), line_width=2.0
                )

        # 2. Retargeted G1 (full mesh, from IK)
        r_joints = retarget_joints_all[pico_idx]
        r_pos = retarget_root_pos_all[pico_idx]
        r_quat = _remove_yaw(retarget_root_quat_wxyz_all[pico_idx])
        _update_urdf(retarget_root, retarget_urdf, r_pos, r_quat, r_joints, y_retarget, mj_to_urdf)

        # 3. TML WBT (full mesh, from policy rollout)
        if idx < len(tml_results):
            t_pos, t_quat, t_joints = tml_results[idx]
            t_quat_noyaw = _remove_yaw(t_quat)
            _update_urdf(tml_root, tml_urdf, t_pos, t_quat_noyaw, t_joints, y_tml, scene_to_urdf)

        # 4. Dense WBT (full mesh, from policy rollout)
        if idx < len(dense_results):
            d_pos, d_quat, d_joints = dense_results[idx]
            d_quat_noyaw = _remove_yaw(d_quat)
            _update_urdf(dense_root, dense_urdf, d_pos, d_quat_noyaw, d_joints, y_dense, scene_to_urdf)

    update_frame(0)

    print(f"\nViser server running at http://localhost:{args.port}")
    print("Open in browser. Press Ctrl+C to exit.")

    @frame_slider.on_update
    def _(_evt):
        update_frame(int(frame_slider.value))

    last_time = time.perf_counter()
    try:
        while True:
            if playing_cb.value:
                now = time.perf_counter()
                dt = now - last_time
                last_time = now
                advance = dt * policy_fps * float(speed_slider.value)
                new_val = float(frame_slider.value) + advance
                if new_val >= N_frames:
                    new_val = 0.0
                frame_slider.value = int(new_val)
                update_frame(int(frame_slider.value))
            else:
                last_time = time.perf_counter()
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
