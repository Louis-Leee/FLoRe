#!/usr/bin/env python3
"""Convert BeyondMimic-format NPZ motion files to holosoma-format NPZ via MuJoCo FK.

BeyondMimic NPZ format (e.g., SONIC_data_filtered):
    fps            : [1]         scalar in array
    joint_pos      : [T, 29]    joint positions (29 DOF, Isaac ordering)
    joint_vel      : [T, 29]    joint velocities
    body_pos_w     : [T, 30, 3] body positions (30 bodies, no foot_contact_points)
    body_quat_w    : [T, 30, 4] body quaternions (wxyz)
    body_lin_vel_w : [T, 30, 3] body linear velocities
    body_ang_vel_w : [T, 30, 3] body angular velocities

Holosoma NPZ format:
    fps            : scalar
    joint_pos      : [T, 36]    root_xyz(3) + root_quat_wxyz(4) + joints(29)
    joint_vel      : [T, 35]    root_lin_vel(3) + root_ang_vel(3) + joints(29)
    body_pos_w     : [T, 32, 3] 32 bodies (with foot_contact_points)
    body_quat_w    : [T, 32, 4] wxyz
    body_lin_vel_w : [T, 32, 3]
    body_ang_vel_w : [T, 32, 3]
    joint_names    : [29]       string array
    body_names     : [32]       string array

This converter:
  1. Extracts root state (pos, quat) from the pelvis body (index 0)
  2. Re-runs MuJoCo FK to get body poses in the correct holosoma ordering (32 bodies)
  3. Computes velocities from body data and finite differences
  4. Outputs holosoma-format NPZ

Usage:
    # Convert single file
    python beyondmimic_to_holosoma_npz.py input.npz output.npz

    # Batch convert directory
    python beyondmimic_to_holosoma_npz.py /nfs/.../SONIC_data_filtered/ /nfs/.../sonic_filtered_holosoma/

    # Limit number of files (for testing)
    python beyondmimic_to_holosoma_npz.py input_dir/ output_dir/ --max-files 100
"""

import argparse
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wbt_training.config_values.robot_config import (
    BODY_NAME_TO_MJ_IDX,
    G1_29DOF_JOINT_NAMES as G1_JOINT_NAMES_MUJOCO,
    G1_32BODY_NAMES as G1_BODY_NAMES,
    ISAACLAB_TO_MUJOCO_DOF as ISAAC_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB_DOF as MUJOCO_TO_ISAAC,
)

G1_JOINT_NAMES_ISAAC = [G1_JOINT_NAMES_MUJOCO[i] for i in MUJOCO_TO_ISAAC]

MUJOCO_SCENE_XML = "assets/g1/scene_29dof.xml"


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


def detect_quat_convention(
    body_quat_w: np.ndarray, joint_pos_isaac: np.ndarray, body_pos_w: np.ndarray, model: mujoco.MjModel
) -> str:
    """Auto-detect whether body_quat_w is wxyz or xyzw by running FK and comparing.

    Tests the first frame with both conventions and returns the one that
    produces body positions closest to the input body_pos_w.
    """
    data = mujoco.MjData(model)
    root_pos = body_pos_w[0, 0, :]
    q = body_quat_w[0, 0, :]
    joint_pos_mj = joint_pos_isaac[0, ISAAC_TO_MUJOCO]  # correct: BeyondMimic IS Isaac order

    errors = {}
    for convention, quat_wxyz in [("wxyz", q), ("xyzw", np.array([q[3], q[0], q[1], q[2]]))]:
        data.qpos[0:3] = root_pos
        data.qpos[3:7] = quat_wxyz
        data.qpos[7:36] = joint_pos_mj
        mujoco.mj_kinematics(model, data)
        # Compare pelvis position (should match exactly)
        _pelvis_err = np.linalg.norm(data.xpos[1] - root_pos)  # noqa: F841
        # Compare a few other bodies
        fk_positions = data.xpos[1 : model.nbody]  # 30 bodies
        bm_positions = body_pos_w[0]  # 30 bodies
        # Use mean position error of first few bodies as metric
        n = min(len(fk_positions), len(bm_positions))
        pos_err = np.mean(np.linalg.norm(fk_positions[:n] - bm_positions[:n], axis=1))
        errors[convention] = pos_err

    best = min(errors, key=errors.get)
    return best


def convert_beyondmimic_to_holosoma(
    input_path: str, output_path: str, model: mujoco.MjModel, quat_convention: str | None = None
) -> bool:
    """Convert a single BeyondMimic NPZ to holosoma-format NPZ.

    Returns True on success, False on skip.
    """
    d = np.load(input_path, allow_pickle=True)

    # Extract fields
    fps_val = d["fps"]
    if fps_val.ndim > 0:
        fps_val = int(fps_val.item() if fps_val.size == 1 else fps_val[0])
    else:
        fps_val = int(fps_val)

    joint_pos_isaac = d["joint_pos"].astype(np.float64)  # [T, 29]
    joint_vel_isaac = d["joint_vel"].astype(np.float64)  # [T, 29]
    bm_body_pos = d["body_pos_w"].astype(np.float64)  # [T, 30, 3]
    bm_body_quat = d["body_quat_w"].astype(np.float64)  # [T, 30, 4]

    T = joint_pos_isaac.shape[0]
    if T < 2:
        return False

    # Root state from pelvis (body index 0)
    root_pos = bm_body_pos[:, 0, :]  # [T, 3]
    root_quat_raw = bm_body_quat[:, 0, :]  # [T, 4]

    # Auto-detect quat convention if not specified
    if quat_convention is None:
        quat_convention = detect_quat_convention(bm_body_quat, joint_pos_isaac, bm_body_pos, model)

    # Convert to wxyz for MuJoCo
    if quat_convention == "wxyz":
        root_quat_wxyz = root_quat_raw
    else:  # xyzw
        root_quat_wxyz = root_quat_raw[:, [3, 0, 1, 2]]

    # Run MuJoCo FK for each frame
    data = mujoco.MjData(model)
    nb = model.nbody

    body_pos_w_all = np.zeros((T, nb, 3))
    body_quat_w_all = np.zeros((T, nb, 4))  # wxyz

    for t in range(T):
        data.qpos[0:3] = root_pos[t]
        data.qpos[3:7] = root_quat_wxyz[t]
        data.qpos[7:36] = joint_pos_isaac[t, ISAAC_TO_MUJOCO]  # correct: BeyondMimic IS Isaac order
        mujoco.mj_kinematics(model, data)
        body_pos_w_all[t] = data.xpos.copy()
        body_quat_w_all[t] = data.xquat.copy()

    # Extract 32 holosoma bodies
    body_indices = [BODY_NAME_TO_MJ_IDX[name] for name in G1_BODY_NAMES]
    body_pos_w = body_pos_w_all[:, body_indices, :]
    body_quat_w = body_quat_w_all[:, body_indices, :]

    # Velocities via finite differences
    dt = 1.0 / fps_val
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

    # Build holosoma-format arrays (joints must be in MuJoCo body-grouped order)
    joint_pos_h = np.concatenate([root_pos, root_quat_wxyz, joint_pos_isaac[:, ISAAC_TO_MUJOCO]], axis=1)  # [T, 36]
    joint_vel_h = np.concatenate([root_lin_vel, root_ang_vel, joint_vel_isaac[:, ISAAC_TO_MUJOCO]], axis=1)  # [T, 35]

    out = {
        "fps": np.array(fps_val),
        "joint_pos": joint_pos_h,
        "joint_vel": joint_vel_h,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "joint_names": np.array(G1_JOINT_NAMES_MUJOCO),
        "body_names": np.array(G1_BODY_NAMES),
    }

    np.savez(output_path, **out)
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert BeyondMimic NPZ to holosoma NPZ via MuJoCo FK")
    parser.add_argument("input", help="Input NPZ file or directory")
    parser.add_argument("output", help="Output NPZ file or directory")
    parser.add_argument("--deploy-dir", default=None, help="Path to TML-Humanoid-Deploy (for MuJoCo XML)")
    parser.add_argument(
        "--quat-convention",
        choices=["wxyz", "xyzw", "auto"],
        default="auto",
        help="Quaternion convention in input data (default: auto-detect)",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Max files to convert (0=all, for testing)")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip files that already exist in output directory"
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
    model = load_mujoco_model(deploy_dir)

    quat_conv = None if args.quat_convention == "auto" else args.quat_convention

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        npz_files = sorted(Path(args.input).glob("*.npz"))
        if not npz_files:
            print(f"No NPZ files found in {args.input}")
            sys.exit(1)

        if args.max_files > 0:
            npz_files = npz_files[: args.max_files]

        print(f"Converting {len(npz_files)} NPZ files...")

        # Auto-detect quat convention from first file
        if quat_conv is None:
            d = np.load(str(npz_files[0]), allow_pickle=True)
            quat_conv = detect_quat_convention(
                d["body_quat_w"].astype(np.float64),
                d["joint_pos"].astype(np.float64),
                d["body_pos_w"].astype(np.float64),
                model,
            )
            print(f"Auto-detected quaternion convention: {quat_conv}")

        converted = 0
        skipped = 0
        for i, f in enumerate(npz_files):
            out_path = os.path.join(args.output, f.name)
            if args.skip_existing and os.path.exists(out_path):
                skipped += 1
                continue
            try:
                ok = convert_beyondmimic_to_holosoma(str(f), out_path, model, quat_conv)
                if ok:
                    converted += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  ERROR converting {f.name}: {e}")
                skipped += 1

            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i + 1}/{len(npz_files)} (converted={converted}, skipped={skipped})")

        print(f"\nDone! Converted {converted} files, skipped {skipped}, to {args.output}")
    else:
        ok = convert_beyondmimic_to_holosoma(args.input, args.output, model, quat_conv)
        if ok:
            print(f"Converted {args.input} -> {args.output}")
        else:
            print(f"Skipped {args.input} (too short)")


if __name__ == "__main__":
    main()
