#!/usr/bin/env python3
"""Convert ASAP/PBHC-format .pkl motion files to holosoma-format .npz via MuJoCo FK.

These pkl files (from ASAP, KungfuBot/PBHC) use joblib serialization and contain:
  - root_trans_offset: [T, 3]  root position
  - root_rot: [T, 4]          root quaternion (xyzw)
  - dof: [T, 23]              joint positions (23 DOF = 29 DOF minus 6 wrist joints)
  - fps: int                   framerate (typically 30)

The 23-DOF representation removes 6 wrist joints from the full 29-DOF G1:
  - left_wrist_roll, left_wrist_pitch, left_wrist_yaw (indices 19,20,21 in Isaac order)
  - right_wrist_roll, right_wrist_pitch, right_wrist_yaw (indices 26,27,28 in Isaac order)

This converter:
  1. Loads pkl via joblib
  2. Maps 23 DOF -> 29 DOF (zeros for missing wrist joints)
  3. Interpolates from input FPS to 50 FPS
  4. Runs MuJoCo FK for body poses
  5. Outputs holosoma-format NPZ (32 bodies)

Usage:
    python pkl_to_holosoma_npz.py /path/to/motions.pkl output_dir/
    python pkl_to_holosoma_npz.py input_dir/ output_dir/
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from wbt_training.config_values.robot_config import (
    BODY_NAME_TO_MJ_IDX,
    DOF23_TO_DOF29_INDICES,
    G1_29DOF_JOINT_NAMES as G1_JOINT_NAMES_29,
    G1_32BODY_NAMES as G1_BODY_NAMES,
)

MUJOCO_SCENE_XML = "assets/g1/scene_29dof.xml"


def expand_23dof_to_29dof(dof_23: np.ndarray) -> np.ndarray:
    """Expand 23-DOF joint positions to 29-DOF by inserting zeros for wrist joints."""
    T = dof_23.shape[0]
    dof_29 = np.zeros((T, 29), dtype=dof_23.dtype)
    dof_29[:, DOF23_TO_DOF29_INDICES] = dof_23
    return dof_29


def interpolate_to_fps(root_pos, root_quat_xyzw, joint_pos_isaac, input_fps=30, output_fps=50):
    """Interpolate motion from input_fps to output_fps."""
    T_in = root_pos.shape[0]
    duration = (T_in - 1) / input_fps
    T_out = int(duration * output_fps) + 1

    t_in = np.linspace(0, duration, T_in)
    t_out = np.linspace(0, duration, T_out)

    root_pos_out = np.zeros((T_out, 3))
    for j in range(3):
        root_pos_out[:, j] = np.interp(t_out, t_in, root_pos[:, j])

    n_joints = joint_pos_isaac.shape[1]
    joint_pos_out = np.zeros((T_out, n_joints))
    for j in range(n_joints):
        joint_pos_out[:, j] = np.interp(t_out, t_in, joint_pos_isaac[:, j])

    rots_in = Rotation.from_quat(root_quat_xyzw)
    slerp = Slerp(t_in, rots_in)
    rots_out = slerp(t_out)
    root_quat_xyzw_out = rots_out.as_quat()

    return root_pos_out, root_quat_xyzw_out, joint_pos_out, T_out


def load_mujoco_model(deploy_dir: str):
    """Load the MuJoCo G1 model."""
    orig_cwd = os.getcwd()
    os.chdir(deploy_dir)
    try:
        with open(MUJOCO_SCENE_XML) as f:
            model = mujoco.MjModel.from_xml_string(f.read())
    finally:
        os.chdir(orig_cwd)
    return model


def convert_single_motion(
    name: str, motion_data: dict, model: mujoco.MjModel, output_path: str, output_fps: int = 50
) -> None:
    """Convert a single motion clip to holosoma NPZ."""
    root_pos = np.array(motion_data["root_trans_offset"], dtype=np.float64)
    root_rot = np.array(motion_data["root_rot"], dtype=np.float64)  # xyzw
    dof = np.array(motion_data["dof"], dtype=np.float64)
    input_fps = int(motion_data.get("fps", 30))

    _T_in = root_pos.shape[0]  # noqa: F841
    n_dof = dof.shape[1]

    # Expand 23 DOF -> 29 DOF if needed
    if n_dof == 23:
        joint_pos_isaac = expand_23dof_to_29dof(dof)
    elif n_dof == 29:
        joint_pos_isaac = dof
    else:
        print(f"  WARNING: {name} has {n_dof} DOF (expected 23 or 29), skipping")
        return

    # Interpolate to output FPS
    root_pos, root_quat_xyzw, joint_pos_isaac, T = interpolate_to_fps(
        root_pos, root_rot, joint_pos_isaac, input_fps, output_fps
    )

    # NOTE: Despite the variable name, joint_pos_isaac is already in MuJoCo
    # body-grouped order (the PKL data and expand_23dof_to_29dof use body-grouped
    # indexing). Do NOT apply ISAAC_TO_MUJOCO — that would double-permute and garble FK.
    joint_pos_mujoco = joint_pos_isaac

    # Run MuJoCo FK
    data = mujoco.MjData(model)
    nb = model.nbody

    body_pos_w_all = np.zeros((T, nb, 3))
    body_quat_w_all = np.zeros((T, nb, 4))

    for t in range(T):
        data.qpos[0:3] = root_pos[t]
        qx, qy, qz, qw = root_quat_xyzw[t]
        data.qpos[3:7] = [qw, qx, qy, qz]
        data.qpos[7:36] = joint_pos_mujoco[t]

        mujoco.mj_kinematics(model, data)

        body_pos_w_all[t] = data.xpos.copy()
        body_quat_w_all[t] = data.xquat.copy()

    # Extract 32 holosoma bodies
    body_indices = [BODY_NAME_TO_MJ_IDX[n] for n in G1_BODY_NAMES]
    body_pos_w = body_pos_w_all[:, body_indices, :]
    body_quat_w = body_quat_w_all[:, body_indices, :]

    # Velocities via finite differences
    dt = 1.0 / output_fps
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0)

    body_ang_vel_w = np.zeros_like(body_pos_w)
    for b in range(len(G1_BODY_NAMES)):
        quats_wxyz = body_quat_w[:, b, :]
        quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
        rots = Rotation.from_quat(quats_xyzw)
        for t in range(1, T - 1):
            drot = rots[t - 1].inv() * rots[t + 1]
            body_ang_vel_w[t, b] = drot.as_rotvec() / (2 * dt)
        body_ang_vel_w[0, b] = body_ang_vel_w[1, b]
        body_ang_vel_w[-1, b] = body_ang_vel_w[-2, b]

    root_lin_vel = body_lin_vel_w[:, 0, :]
    root_ang_vel = body_ang_vel_w[:, 0, :]
    joint_vel_isaac = np.gradient(joint_pos_isaac, dt, axis=0)

    # Build holosoma format
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    joint_pos_h = np.concatenate([root_pos, root_quat_wxyz, joint_pos_isaac], axis=1)
    joint_vel_h = np.concatenate([root_lin_vel, root_ang_vel, joint_vel_isaac], axis=1)

    out = {
        "fps": np.array(output_fps),
        "joint_pos": joint_pos_h.astype(np.float64),
        "joint_vel": joint_vel_h.astype(np.float64),
        "body_pos_w": body_pos_w.astype(np.float64),
        "body_quat_w": body_quat_w.astype(np.float64),
        "body_lin_vel_w": body_lin_vel_w.astype(np.float64),
        "body_ang_vel_w": body_ang_vel_w.astype(np.float64),
        "joint_names": np.array(G1_JOINT_NAMES_29),
        "body_names": np.array(G1_BODY_NAMES),
    }

    np.savez(output_path, **out)
    print(
        f"  {name} -> {os.path.basename(output_path)}  "
        f"({T} frames, {T / output_fps:.1f}s, {n_dof}dof->{29}dof, {len(G1_BODY_NAMES)} bodies)"
    )


def convert_pkl_file(pkl_path: str, output_dir: str, model: mujoco.MjModel, output_fps: int = 50) -> int:
    """Convert all motions in a single pkl file to holosoma NPZ files.

    Each pkl contains a dict of {motion_name: motion_data} pairs.
    Returns the number of motions converted.
    """
    data = joblib.load(pkl_path)

    if not isinstance(data, dict):
        print(f"  WARNING: {pkl_path} is not a dict, skipping")
        return 0

    count = 0
    for motion_name, motion_data in data.items():
        if not isinstance(motion_data, dict) or "dof" not in motion_data:
            print(f"  WARNING: {motion_name} missing 'dof' key, skipping")
            continue

        # Clean up name for filename
        safe_name = motion_name.replace("/", "_").replace(" ", "_")
        out_path = os.path.join(output_dir, f"{safe_name}.npz")
        convert_single_motion(motion_name, motion_data, model, out_path, output_fps)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Convert ASAP/PBHC pkl motions to holosoma NPZ via MuJoCo FK")
    parser.add_argument("input", help="Input pkl file or directory of pkl files")
    parser.add_argument("output", help="Output directory for NPZ files")
    parser.add_argument("--output-fps", type=int, default=50, help="Output NPZ framerate (default: 50)")
    parser.add_argument("--deploy-dir", default=None, help="Path to TML-Humanoid-Deploy (for MuJoCo XML)")
    args = parser.parse_args()

    # Find TML-Humanoid-Deploy
    deploy_dir = args.deploy_dir
    if deploy_dir is None:
        candidates = [
            os.path.expanduser("~/git/TML-Humanoid-Deploy"),
            os.path.join(os.path.dirname(__file__), "../../../../TML-Humanoid-Deploy"),
        ]
        for c in candidates:
            if os.path.isdir(c) and os.path.exists(os.path.join(c, MUJOCO_SCENE_XML)):
                deploy_dir = os.path.abspath(c)
                break
    if deploy_dir is None or not os.path.exists(os.path.join(deploy_dir, MUJOCO_SCENE_XML)):
        print(f"ERROR: Cannot find TML-Humanoid-Deploy with {MUJOCO_SCENE_XML}. Use --deploy-dir.")
        sys.exit(1)

    print(f"Using MuJoCo model from: {deploy_dir}/{MUJOCO_SCENE_XML}")
    model = load_mujoco_model(deploy_dir)
    print(f"Model: {model.nbody} bodies, {model.njnt} joints")

    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.input):
        pkl_files = sorted(Path(args.input).glob("*.pkl"))
        if not pkl_files:
            print(f"No pkl files found in {args.input}")
            sys.exit(1)
        print(f"Found {len(pkl_files)} pkl files")
        total = 0
        for pkl_file in pkl_files:
            print(f"\nProcessing {pkl_file.name}...")
            total += convert_pkl_file(str(pkl_file), args.output, model, args.output_fps)
        print(f"\nDone! Converted {total} motions to {args.output}")
    else:
        print(f"\nProcessing {args.input}...")
        count = convert_pkl_file(args.input, args.output, model, args.output_fps)
        print(f"\nDone! Converted {count} motions to {args.output}")


if __name__ == "__main__":
    main()
