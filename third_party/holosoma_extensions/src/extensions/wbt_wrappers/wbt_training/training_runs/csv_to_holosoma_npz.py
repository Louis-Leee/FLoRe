#!/usr/bin/env python3
"""Batch-convert LAFAN1 CSV motions to holosoma-format .npz via MuJoCo FK.

This replaces the TML csv_to_npz.py (which requires IsaacSim) with a
standalone MuJoCo-based converter. For each CSV it:
  1. Reads root pose + 29 joint positions (Isaac ordering, 30fps)
  2. Interpolates to 50fps
  3. Uses MuJoCo forward kinematics to compute body poses
  4. Outputs holosoma-format NPZ (root DOFs prepended, 32 body names)

CSV format (LAFAN1 retargeted to G1):
  Columns: root_x, root_y, root_z, quat_x, quat_y, quat_z, quat_w, j0..j28
  Joint order: Isaac ordering (same as holosoma)
  FPS: 30

Usage:
    python csv_to_holosoma_npz.py /path/to/motion_data/lafan1_csv/g1/ /path/to/motion_data/lafan1_holosoma/
    python csv_to_holosoma_npz.py /path/to/motion_data/lafan1_csv/g1/walk1_subject1.csv output.npz
"""

import argparse
import os
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from wbt_training.config_values.robot_config import (
    BODY_NAME_TO_MJ_IDX,
    G1_29DOF_JOINT_NAMES as G1_JOINT_NAMES,
    G1_32BODY_NAMES as G1_BODY_NAMES,
)

MUJOCO_SCENE_XML = "assets/g1/scene_29dof.xml"


def _interpolate_to_50fps(root_pos, root_quat_xyzw, joint_pos_isaac, input_fps=30, output_fps=50):
    """Interpolate motion from input_fps to output_fps."""
    T_in = root_pos.shape[0]
    duration = (T_in - 1) / input_fps
    T_out = int(duration * output_fps) + 1

    t_in = np.linspace(0, duration, T_in)
    t_out = np.linspace(0, duration, T_out)

    # Linear interpolation for positions and joint angles
    root_pos_out = np.zeros((T_out, 3))
    joint_pos_out = np.zeros((T_out, 29))
    for j in range(3):
        root_pos_out[:, j] = np.interp(t_out, t_in, root_pos[:, j])
    for j in range(29):
        joint_pos_out[:, j] = np.interp(t_out, t_in, joint_pos_isaac[:, j])

    # Slerp for quaternions
    rots_in = Rotation.from_quat(root_quat_xyzw)  # scipy uses xyzw
    slerp = Slerp(t_in, rots_in)
    rots_out = slerp(t_out)
    root_quat_xyzw_out = rots_out.as_quat()  # xyzw

    return root_pos_out, root_quat_xyzw_out, joint_pos_out, T_out


def _load_mujoco_model(deploy_dir: str):
    """Load the MuJoCo G1 model."""
    orig_cwd = os.getcwd()
    os.chdir(deploy_dir)
    try:
        with open(MUJOCO_SCENE_XML) as f:
            model = mujoco.MjModel.from_xml_string(f.read())
    finally:
        os.chdir(orig_cwd)
    return model


def convert_csv_to_holosoma(
    csv_path: str, output_path: str, model: mujoco.MjModel, input_fps: int = 30, output_fps: int = 50
) -> None:
    """Convert a single LAFAN1 CSV to holosoma-format NPZ."""
    # Load CSV
    raw = np.loadtxt(csv_path, delimiter=",")
    assert raw.shape[1] == 36, f"Expected 36 columns, got {raw.shape[1]}"

    root_pos = raw[:, :3]
    root_quat_xyzw = raw[:, 3:7]  # CSV is xyzw
    joint_pos_isaac = raw[:, 7:]  # Isaac ordering

    # Interpolate to output_fps
    root_pos, root_quat_xyzw, joint_pos_isaac, T = _interpolate_to_50fps(
        root_pos, root_quat_xyzw, joint_pos_isaac, input_fps, output_fps
    )

    # NOTE: Despite the variable name, joint_pos_isaac is already in MuJoCo
    # body-grouped order (the CSV comes from holosoma retargeting, not TML).
    # Do NOT apply ISAAC_TO_MUJOCO — that would double-permute and garble FK.
    joint_pos_mujoco = joint_pos_isaac

    # Run MuJoCo FK for each frame
    data = mujoco.MjData(model)
    nb = model.nbody  # 31 (including world)

    body_pos_w_all = np.zeros((T, nb, 3))
    body_quat_w_all = np.zeros((T, nb, 4))  # MuJoCo uses wxyz

    for t in range(T):
        # Set root state: qpos[0:3] = xyz, qpos[3:7] = wxyz quaternion
        data.qpos[0:3] = root_pos[t]
        # Convert xyzw to wxyz for MuJoCo
        qx, qy, qz, qw = root_quat_xyzw[t]
        data.qpos[3:7] = [qw, qx, qy, qz]
        # Set joint positions (MuJoCo ordering)
        data.qpos[7:36] = joint_pos_mujoco[t]

        mujoco.mj_kinematics(model, data)

        body_pos_w_all[t] = data.xpos.copy()
        body_quat_w_all[t] = data.xquat.copy()  # wxyz

    # Extract the 32 holosoma bodies from MuJoCo's body list
    body_indices = [BODY_NAME_TO_MJ_IDX[name] for name in G1_BODY_NAMES]
    body_pos_w = body_pos_w_all[:, body_indices, :]
    body_quat_w = body_quat_w_all[:, body_indices, :]  # already wxyz

    # Compute velocities via finite differences
    dt = 1.0 / output_fps
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0)

    # Angular velocity: approximate from quaternion differences
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

    # Root velocities (from body 0 = pelvis)
    root_lin_vel = body_lin_vel_w[:, 0, :]
    root_ang_vel = body_ang_vel_w[:, 0, :]

    # Joint velocities via finite differences
    joint_vel_isaac = np.gradient(joint_pos_isaac, dt, axis=0)

    # Build holosoma-format arrays
    # joint_pos_h: [root_xyz(3), root_quat_wxyz(4), joint_pos_isaac(29)] = 36
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    joint_pos_h = np.concatenate([root_pos, root_quat_wxyz, joint_pos_isaac], axis=1)
    # joint_vel_h: [root_lin_vel(3), root_ang_vel(3), joint_vel_isaac(29)] = 35
    joint_vel_h = np.concatenate([root_lin_vel, root_ang_vel, joint_vel_isaac], axis=1)

    out = {
        "fps": np.array(output_fps),
        "joint_pos": joint_pos_h.astype(np.float64),
        "joint_vel": joint_vel_h.astype(np.float64),
        "body_pos_w": body_pos_w.astype(np.float64),
        "body_quat_w": body_quat_w.astype(np.float64),
        "body_lin_vel_w": body_lin_vel_w.astype(np.float64),
        "body_ang_vel_w": body_ang_vel_w.astype(np.float64),
        "joint_names": np.array(G1_JOINT_NAMES),
        "body_names": np.array(G1_BODY_NAMES),
    }

    np.savez(output_path, **out)
    name = os.path.basename(csv_path)
    print(
        f"  {name} -> {os.path.basename(output_path)}  ({T} frames, {T / output_fps:.1f}s, {len(G1_BODY_NAMES)} bodies)"
    )


def main():
    parser = argparse.ArgumentParser(description="Batch convert LAFAN1 CSV to holosoma NPZ via MuJoCo FK")
    parser.add_argument("input", help="Input CSV file or directory of CSVs")
    parser.add_argument("output", help="Output NPZ file or directory")
    parser.add_argument("--input-fps", type=int, default=30, help="Input CSV framerate (default: 30)")
    parser.add_argument("--output-fps", type=int, default=50, help="Output NPZ framerate (default: 50)")
    parser.add_argument(
        "--deploy-dir", default=None, help="Path to TML-Humanoid-Deploy (for MuJoCo XML). Auto-detected if not set."
    )
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
    model = _load_mujoco_model(deploy_dir)
    print(f"Model: {model.nbody} bodies, {model.njnt} joints")

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        csv_files = sorted(f for f in os.listdir(args.input) if f.endswith(".csv"))
        if not csv_files:
            print(f"No CSV files found in {args.input}")
            sys.exit(1)
        print(f"Converting {len(csv_files)} CSV files...")
        for f in csv_files:
            out_name = f.replace(".csv", ".npz")
            convert_csv_to_holosoma(
                os.path.join(args.input, f), os.path.join(args.output, out_name), model, args.input_fps, args.output_fps
            )
        print(f"\nDone! Converted {len(csv_files)} files to {args.output}")
    else:
        convert_csv_to_holosoma(args.input, args.output, model, args.input_fps, args.output_fps)


if __name__ == "__main__":
    main()
