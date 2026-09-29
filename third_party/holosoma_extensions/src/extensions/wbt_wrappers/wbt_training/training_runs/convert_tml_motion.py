#!/usr/bin/env python3
"""Convert TML-format motion .npz to holosoma-format .npz.

TML format (from csv_to_npz.py / TML-BeyondMimic-Generalist):
  - joint_pos: [T, 29]  (joint-only, no root DOFs)
  - joint_vel: [T, 29]
  - body_pos_w: [T, 30, 3]  (30 bodies for G1 29-DOF)
  - body_quat_w: [T, 30, 4]  (wxyz)
  - No joint_names, body_names arrays

Holosoma format (expected by holosoma.managers.command.terms.wbt):
  - joint_pos: [T, 36]  (7 root DOFs prepended: xyz + wxyz)
  - joint_vel: [T, 35]  (6 root DOFs prepended: lin_vel + ang_vel)
  - body_pos_w: [T, N, 3]
  - body_quat_w: [T, N, 4]  (wxyz)
  - joint_names: [29]  (string array)
  - body_names: [N]  (string array)

Usage:
    python convert_tml_motion.py input.npz output.npz
    python convert_tml_motion.py input_dir/ output_dir/   # batch convert
"""

import argparse
import os
import sys

import numpy as np

from wbt_training.config_values.robot_config import (
    G1_29DOF_JOINT_NAMES as G1_JOINT_NAMES,
    G1_32BODY_NAMES as G1_BODY_NAMES,
)


def _pad_bodies_to_32(body_pos_w, body_quat_w, body_lin_vel_w=None, body_ang_vel_w=None):
    """Pad 30-body TML data to 32 bodies expected by holosoma robot config.

    TML captures 30 bodies (ending at right_wrist_roll_link, index 29).
    The robot config expects 32 bodies: the 2 extra are right_wrist_pitch_link
    and right_wrist_yaw_link (child links of right_wrist_roll_link).
    We duplicate right_wrist_roll_link's pose for these.
    """
    # right_wrist_roll_link is at index 29 (last body in 30-body data)
    parent_idx = 29
    pad_pos = body_pos_w[:, parent_idx : parent_idx + 1, :]  # [T, 1, 3]
    pad_quat = body_quat_w[:, parent_idx : parent_idx + 1, :]  # [T, 1, 4]

    body_pos_w = np.concatenate([body_pos_w, pad_pos, pad_pos], axis=1)  # [T, 32, 3]
    body_quat_w = np.concatenate([body_quat_w, pad_quat, pad_quat], axis=1)  # [T, 32, 4]

    if body_lin_vel_w is not None:
        pad_lv = body_lin_vel_w[:, parent_idx : parent_idx + 1, :]
        body_lin_vel_w = np.concatenate([body_lin_vel_w, pad_lv, pad_lv], axis=1)
    if body_ang_vel_w is not None:
        pad_av = body_ang_vel_w[:, parent_idx : parent_idx + 1, :]
        body_ang_vel_w = np.concatenate([body_ang_vel_w, pad_av, pad_av], axis=1)

    return body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w


def convert_tml_to_holosoma(input_path: str, output_path: str) -> None:
    """Convert a single TML-format .npz to holosoma format."""
    data = np.load(input_path)

    joint_pos = data["joint_pos"]  # [T, 29]
    joint_vel = data["joint_vel"]  # [T, 29]
    body_pos_w = data["body_pos_w"]  # [T, N, 3]
    body_quat_w = data["body_quat_w"]  # [T, N, 4] wxyz

    T = joint_pos.shape[0]
    num_joints = joint_pos.shape[1]
    num_bodies = body_pos_w.shape[1]

    # Validate
    assert num_joints == 29, f"Expected 29 joints, got {num_joints}"

    # Prepend root DOFs from body_pos_w and body_quat_w (pelvis = body 0)
    root_pos = body_pos_w[:, 0, :]  # [T, 3] xyz
    root_quat_wxyz = body_quat_w[:, 0, :]  # [T, 4] wxyz

    # joint_pos_holosoma = [root_xyz, root_quat_wxyz, joint_pos] -> [T, 36]
    joint_pos_h = np.concatenate([root_pos, root_quat_wxyz, joint_pos], axis=1)

    # joint_vel_holosoma = [root_lin_vel, root_ang_vel, joint_vel] -> [T, 35]
    body_lin_vel_w = data["body_lin_vel_w"] if "body_lin_vel_w" in data else None
    body_ang_vel_w = data["body_ang_vel_w"] if "body_ang_vel_w" in data else None

    if body_lin_vel_w is not None and body_ang_vel_w is not None:
        root_lin_vel = body_lin_vel_w[:, 0, :]  # [T, 3]
        root_ang_vel = body_ang_vel_w[:, 0, :]  # [T, 3]
    else:
        # Approximate from finite differences
        dt = 1.0 / float(data["fps"])
        root_lin_vel = np.gradient(root_pos, dt, axis=0)
        # Approximate angular velocity (simplified)
        root_ang_vel = np.zeros((T, 3))

    joint_vel_h = np.concatenate([root_lin_vel, root_ang_vel, joint_vel], axis=1)

    # Pad 30-body TML data to 32 bodies (robot config expects 32)
    if num_bodies == 30:
        body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w = _pad_bodies_to_32(
            body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
        )
        body_names = G1_BODY_NAMES  # all 32
        num_bodies = 32
    elif num_bodies == 32:
        body_names = G1_BODY_NAMES
    else:
        body_names = G1_BODY_NAMES[: min(num_bodies, len(G1_BODY_NAMES))]
        if num_bodies > len(G1_BODY_NAMES):
            body_names += [f"body_{i}" for i in range(len(G1_BODY_NAMES), num_bodies)]

    # Build output
    out = {
        "fps": data["fps"],
        "joint_pos": joint_pos_h.astype(np.float64),
        "joint_vel": joint_vel_h.astype(np.float64),
        "body_pos_w": body_pos_w.astype(np.float64),
        "body_quat_w": body_quat_w.astype(np.float64),
        "joint_names": np.array(G1_JOINT_NAMES),
        "body_names": np.array(body_names),
    }

    # Carry over velocity arrays if present
    if body_lin_vel_w is not None:
        out["body_lin_vel_w"] = body_lin_vel_w.astype(np.float64)
    if body_ang_vel_w is not None:
        out["body_ang_vel_w"] = body_ang_vel_w.astype(np.float64)

    np.savez(output_path, **out)
    print(f"Converted: {input_path} -> {output_path}")
    print(f"  joint_pos: {joint_pos.shape} -> {joint_pos_h.shape}")
    print(f"  joint_vel: {joint_vel.shape} -> {joint_vel_h.shape}")
    print(f"  bodies: {num_bodies}, frames: {T}")


def main():
    parser = argparse.ArgumentParser(description="Convert TML motion .npz to holosoma format")
    parser.add_argument("input", help="Input .npz file or directory of .npz files")
    parser.add_argument("output", help="Output .npz file or directory")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        npz_files = [f for f in os.listdir(args.input) if f.endswith(".npz")]
        if not npz_files:
            print(f"No .npz files found in {args.input}")
            sys.exit(1)
        for f in sorted(npz_files):
            convert_tml_to_holosoma(os.path.join(args.input, f), os.path.join(args.output, f))
        print(f"\nConverted {len(npz_files)} files.")
    else:
        convert_tml_to_holosoma(args.input, args.output)


if __name__ == "__main__":
    main()
