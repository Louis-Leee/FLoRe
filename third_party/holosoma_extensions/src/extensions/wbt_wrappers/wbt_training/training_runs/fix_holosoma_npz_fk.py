#!/usr/bin/env python3
"""Recompute body FK in holosoma-format NPZ files using the correct MuJoCo model.

The original converters (csv_to_holosoma_npz, pkl_to_holosoma_npz) applied
ISAAC_TO_MUJOCO to joint data that was ALREADY in MuJoCo order, producing
garbled body_pos_w and body_quat_w. This script fixes existing NPZ files
by recomputing FK from the (correct) joint_pos data.

Usage:
    # Fix a single file
    python fix_holosoma_npz_fk.py /path/to/file.npz

    # Fix all NPZ files in a directory
    python fix_holosoma_npz_fk.py /path/to/motion_data/lafan1_holosoma/

    # Fix multiple directories
    python fix_holosoma_npz_fk.py /path/to/motion_data/lafan1_holosoma/ /path/to/motion_data/asap_holosoma/ /path/to/motion_data/pbhc_holosoma/
"""

import argparse
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wbt_training.config_values.robot_config import BODY_NAME_TO_MJ_IDX, G1_32BODY_NAMES as G1_BODY_NAMES

MUJOCO_SCENE_XML = "assets/g1/scene_29dof.xml"


def load_mujoco_model(deploy_dir: str):
    orig_cwd = os.getcwd()
    os.chdir(deploy_dir)
    try:
        with open(MUJOCO_SCENE_XML) as f:
            model = mujoco.MjModel.from_xml_string(f.read())
    finally:
        os.chdir(orig_cwd)
    return model


def fix_single_npz(npz_path: str, model: mujoco.MjModel) -> bool:
    """Recompute body FK for a single holosoma NPZ file. Returns True on success."""
    d = dict(np.load(npz_path, allow_pickle=True))

    joint_pos = d["joint_pos"]
    T = joint_pos.shape[0]

    if joint_pos.shape[1] != 36:
        print(f"  SKIP {os.path.basename(npz_path)}: joint_pos has {joint_pos.shape[1]} cols (expected 36)")
        return False

    # Extract root and joint data from holosoma format
    # joint_pos[:, 0:3] = root xyz, joint_pos[:, 3:7] = root quat wxyz, joint_pos[:, 7:] = joints (MuJoCo order)
    root_pos = joint_pos[:, :3]
    root_quat_wxyz = joint_pos[:, 3:7]
    joint_pos_mujoco = joint_pos[:, 7:]  # already in MuJoCo body-grouped order

    # Run MuJoCo FK
    data = mujoco.MjData(model)
    nb = model.nbody

    body_pos_w_all = np.zeros((T, nb, 3))
    body_quat_w_all = np.zeros((T, nb, 4))  # wxyz

    for t in range(T):
        data.qpos[0:3] = root_pos[t]
        data.qpos[3:7] = root_quat_wxyz[t]
        data.qpos[7:36] = joint_pos_mujoco[t]
        mujoco.mj_kinematics(model, data)
        body_pos_w_all[t] = data.xpos.copy()
        body_quat_w_all[t] = data.xquat.copy()  # wxyz

    # Extract 32 holosoma bodies
    body_indices = [BODY_NAME_TO_MJ_IDX[name] for name in G1_BODY_NAMES]
    body_pos_w = body_pos_w_all[:, body_indices, :]
    body_quat_w = body_quat_w_all[:, body_indices, :]

    # Recompute velocities
    fps = float(d["fps"])
    dt = 1.0 / fps
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

    # Update
    d["body_pos_w"] = body_pos_w.astype(np.float64)
    d["body_quat_w"] = body_quat_w.astype(np.float64)
    d["body_lin_vel_w"] = body_lin_vel_w.astype(np.float64)
    d["body_ang_vel_w"] = body_ang_vel_w.astype(np.float64)

    # Write to temp file first then move (NFS doesn't support seek on open files)
    import shutil
    import tempfile

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".npz")
    os.close(tmp_fd)
    try:
        np.savez(tmp_path, **d)
        shutil.move(tmp_path, npz_path)
    except Exception:
        os.unlink(tmp_path)
        raise
    return True


def main():
    parser = argparse.ArgumentParser(description="Recompute MuJoCo FK in holosoma NPZ files")
    parser.add_argument("inputs", nargs="+", help="NPZ file(s) or directory(ies)")
    parser.add_argument("--deploy-dir", default=None, help="Path to TML-Humanoid-Deploy (for MuJoCo XML)")
    args = parser.parse_args()

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
    if deploy_dir is None:
        print("ERROR: Cannot find TML-Humanoid-Deploy. Use --deploy-dir.")
        sys.exit(1)

    print(f"Using MuJoCo model from: {deploy_dir}/{MUJOCO_SCENE_XML}")
    model = load_mujoco_model(deploy_dir)

    # Collect all NPZ files
    npz_files = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            npz_files.extend(sorted(str(p) for p in Path(inp).glob("*.npz")))
        elif os.path.isfile(inp):
            npz_files.append(inp)
        else:
            print(f"WARNING: {inp} not found, skipping")

    if not npz_files:
        print("No NPZ files found.")
        sys.exit(1)

    print(f"Fixing FK in {len(npz_files)} files...")
    fixed = 0
    for i, f in enumerate(npz_files):
        ok = fix_single_npz(f, model)
        if ok:
            fixed += 1
        if (i + 1) % 10 == 0 or i == len(npz_files) - 1:
            print(f"  Progress: {i + 1}/{len(npz_files)} (fixed={fixed})")

    print(f"\nDone! Fixed {fixed}/{len(npz_files)} files.")


if __name__ == "__main__":
    main()
