#!/usr/bin/env python3
"""Convert PICO body tracking recording (.pkl) to holosoma-format NPZ via retargeting + MuJoCo FK.

Pipeline:
  1. Load PICO pickle (24-joint SMPL, Y-up, variable FPS ~5-30Hz)
  2. Run SMPLRetargeter (mink IK) → G1 29-DOF joint angles + root orientation
  3. Interpolate to 50 FPS
  4. Run MuJoCo FK → 32 body poses
  5. Save as holosoma NPZ

Requires: mink, loguru, mujoco, scipy, numpy

Usage:
    python pico_to_holosoma_npz.py pico_recording.pkl output.npz
    python pico_to_holosoma_npz.py pico_recording.pkl output_dir/
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from wbt_training.config_values.robot_config import (
    BODY_NAME_TO_MJ_IDX,
    G1_29DOF_JOINT_NAMES as G1_JOINT_NAMES_29,
    G1_32BODY_NAMES as G1_BODY_NAMES,
)

# Coordinate transform: Pico Y-up → robot Z-up
_COORD_ROT_MAT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def load_pico_recording(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        rec = pickle.load(f)
    _body_poses = rec["body_joint_poses"]  # noqa: F841  (N, 24, 7)
    ts = rec["timestamps_ns"]
    N = len(ts)
    dt = (ts[-1] - ts[0]) / 1e9
    fps = (N - 1) / dt if dt > 0 else 5.0
    print(f"Loaded PICO recording: {N} frames, {dt:.1f}s, {fps:.1f} FPS")
    return rec, fps


def _compute_g1_height_ratio(xml_path):
    """Compute proper height ratio: G1 pelvis-to-ankle / human average."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    # Set robot to default pose (pelvis at proper height)
    data.qpos[:] = 0
    data.qpos[2] = 0.793  # G1 default pelvis height
    data.qpos[3] = 1.0  # identity quaternion w
    mujoco.mj_forward(model, data)
    pelvis_z = data.xpos[1][2]  # pelvis body
    ankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    ankle_z = data.xpos[ankle_id][2]
    g1_leg = pelvis_z - ankle_z
    return g1_leg  # ~0.757m


def retarget_sequence(body_joint_poses, xml_path, input_fps, urdf_path=None):
    """Run SMPLRetargeter on all frames, returning G1 joint angles + root poses."""
    from retargeting.smpl_retargeter import SMPLRetargeter

    # Use fixed-base URDF for retargeter (no free joint → correct FK leg length)
    # The scene XML with free joint gives wrong neutral pose.
    retarget_model = urdf_path if urdf_path else xml_path
    dt = 1.0 / input_fps
    retargeter = SMPLRetargeter(retarget_model, dt=dt)

    # Compute proper height ratio using full-body FK on the scene XML
    g1_leg_length = _compute_g1_height_ratio(xml_path)

    N = body_joint_poses.shape[0]
    all_q_joints = []
    all_root_pos = []
    all_root_quat_wxyz = []

    # Compute human leg length from first frame
    frame0 = body_joint_poses[0]
    pelvis_yup = frame0[0, :3]
    ankle_yup = frame0[7, :3]  # left_ankle in SMPL
    human_leg = np.linalg.norm((_COORD_ROT_MAT @ pelvis_yup) - (_COORD_ROT_MAT @ ankle_yup))
    height_ratio = g1_leg_length / max(human_leg, 0.01)
    print(f"  Height ratio: {height_ratio:.3f} (G1 leg={g1_leg_length:.3f}m, human leg={human_leg:.3f}m)")

    for i in range(N):
        frame = body_joint_poses[i]  # (24, 7)
        q_joints_29, dq, root_orn_wxyz = retargeter.retarget(frame)

        # Root position from retargeter's internal state (already in Z-up, scaled)
        root_pos = retargeter._prev_q[:3].copy()
        # Anchor z so feet are near ground (G1 default pelvis height ~0.793)
        if i == 0:
            z_offset = 0.793 - root_pos[2]
        root_pos[2] += z_offset

        all_q_joints.append(q_joints_29)
        all_root_pos.append(root_pos)
        all_root_quat_wxyz.append(root_orn_wxyz)

        if (i + 1) % 50 == 0 or i == N - 1:
            print(f"  Retargeted {i + 1}/{N} frames")

    return (np.array(all_q_joints), np.array(all_root_pos), np.array(all_root_quat_wxyz))


def interpolate_to_fps(root_pos, root_quat_wxyz, joint_pos_mj, input_fps, output_fps=50):
    """Interpolate from input FPS to output FPS."""
    T_in = root_pos.shape[0]
    duration = (T_in - 1) / input_fps
    T_out = max(int(duration * output_fps) + 1, 2)

    t_in = np.linspace(0, duration, T_in)
    t_out = np.linspace(0, duration, T_out)

    # Root position — linear interp
    root_pos_out = np.zeros((T_out, 3))
    for j in range(3):
        root_pos_out[:, j] = np.interp(t_out, t_in, root_pos[:, j])

    # Joint angles — linear interp
    n_joints = joint_pos_mj.shape[1]
    joint_pos_out = np.zeros((T_out, n_joints))
    for j in range(n_joints):
        joint_pos_out[:, j] = np.interp(t_out, t_in, joint_pos_mj[:, j])

    # Root orientation — SLERP
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    rots_in = Rotation.from_quat(root_quat_xyzw)
    slerp = Slerp(t_in, rots_in)
    rots_out = slerp(t_out)
    root_quat_wxyz_out = rots_out.as_quat(scalar_first=True)

    return root_pos_out, root_quat_wxyz_out, joint_pos_out, T_out


def run_fk_and_save(root_pos, root_quat_wxyz, joint_pos_mj, output_fps, xml_path, output_path):
    """Run MuJoCo FK and save holosoma NPZ."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    nb = model.nbody
    T = root_pos.shape[0]

    body_pos_w_all = np.zeros((T, nb, 3))
    body_quat_w_all = np.zeros((T, nb, 4))

    for t in range(T):
        data.qpos[0:3] = root_pos[t]
        data.qpos[3:7] = root_quat_wxyz[t]  # wxyz for MuJoCo
        data.qpos[7:36] = joint_pos_mj[t]
        mujoco.mj_kinematics(model, data)
        body_pos_w_all[t] = data.xpos.copy()
        body_quat_w_all[t] = data.xquat.copy()  # wxyz

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
    joint_vel = np.gradient(joint_pos_mj, dt, axis=0)

    # Build holosoma format
    joint_pos_h = np.concatenate([root_pos, root_quat_wxyz, joint_pos_mj], axis=1)
    joint_vel_h = np.concatenate([root_lin_vel, root_ang_vel, joint_vel], axis=1)

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
    print(f"Saved: {output_path} ({T} frames, {T / output_fps:.1f}s, 32 bodies)")


def main():
    parser = argparse.ArgumentParser(description="Convert PICO body tracking pkl to holosoma NPZ")
    parser.add_argument("input", help="Input PICO pickle file")
    parser.add_argument("output", help="Output NPZ file or directory")
    parser.add_argument("--xml", default=None, help="MuJoCo scene XML (auto-resolved)")
    parser.add_argument("--urdf", default=None, help="Fixed-base URDF for retargeter (auto-resolved)")
    parser.add_argument("--output-fps", type=int, default=50, help="Output FPS (default: 50)")
    args = parser.parse_args()

    # Auto-resolve XML
    xml_path = args.xml
    if xml_path is None:
        xml_path = os.path.join(
            os.path.dirname(__file__),
            "../../../../thirdparty/holosoma/src/holosoma/holosoma/data",
            "robots/g1/scenes/scene_g1_29dof_wbt_plane.xml",
        )
    xml_path = os.path.abspath(xml_path)
    if not os.path.exists(xml_path):
        print(f"Cannot find XML at {xml_path}", file=sys.stderr)
        sys.exit(1)

    # Auto-resolve URDF (fixed-base, no free joint — needed for correct retargeter FK)
    urdf_path = args.urdf
    if urdf_path is None:
        urdf_path = os.path.join(
            os.path.dirname(__file__), "../../../wbt_wrappers/wbt_wrappers_inference/urdf/g1_29dof.urdf"
        )
    urdf_path = os.path.abspath(urdf_path)
    if not os.path.exists(urdf_path):
        print(f"Warning: URDF not found at {urdf_path}, falling back to XML", file=sys.stderr)
        urdf_path = None

    # Determine output path
    output_path = args.output
    if os.path.isdir(output_path):
        stem = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(output_path, f"{stem}.npz")

    # Load PICO recording
    rec, input_fps = load_pico_recording(args.input)
    body_poses = rec["body_joint_poses"]

    # Retarget all frames
    print(f"\nRetargeting {body_poses.shape[0]} frames...")
    q_joints, root_pos, root_quat_wxyz = retarget_sequence(body_poses, xml_path, input_fps, urdf_path=urdf_path)

    # Interpolate to output FPS
    print(f"\nInterpolating {input_fps:.1f} → {args.output_fps} FPS...")
    root_pos, root_quat_wxyz, q_joints, T = interpolate_to_fps(
        root_pos, root_quat_wxyz, q_joints, input_fps, args.output_fps
    )
    print(f"  {T} output frames ({T / args.output_fps:.1f}s)")

    # Run FK and save
    print("\nRunning FK and saving...")
    run_fk_and_save(root_pos, root_quat_wxyz, q_joints, args.output_fps, xml_path, output_path)


if __name__ == "__main__":
    main()
