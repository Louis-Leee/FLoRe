#!/usr/bin/env python3
"""Visualize holosoma or TML format NPZ motion sequences in MuJoCo.

Replays joint_pos data through MuJoCo FK and renders to mp4.
Supports both holosoma format (36 cols: 7 root + 29 joints) and
TML format (29 cols: joints only, with body_pos_w/body_quat_w for root).
This script is part of the FAR-pi repository.

Usage:
    conda activate hssim  # or whichever env has mujoco + numpy
    python visualize_motion.py /path/to/motion.npz --xml /path/to/scene.xml -o output.mp4
    python visualize_motion.py /path/to/motion.npz --xml /path/to/scene.xml -o output.mp4 --max_frames 500
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import platform

os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Darwin" else "egl")

import mujoco
import numpy as np

try:
    from wbt_training.config_values.robot_config import ISAACLAB_TO_MUJOCO_DOF as ISAAC_TO_MUJOCO
except ImportError:
    ISAAC_TO_MUJOCO = None

try:
    from wbt_wrappers_inference.utils.eval_utils import inject_terrain_mesh
except ImportError:
    inject_terrain_mesh = None


def load_model(xml_path: str, terrain_obj: str | None = None) -> mujoco.MjModel:
    abs_xml = os.path.abspath(xml_path)
    xml_dir = os.path.dirname(abs_xml)
    search = xml_dir
    found_assets = False
    for _ in range(5):
        if os.path.isdir(os.path.join(search, "assets")):
            found_assets = True
            break
        parent = os.path.dirname(search)
        if parent == search:
            break
        search = parent
    if not found_assets:
        search = xml_dir
    orig_cwd = os.getcwd()
    os.chdir(search)
    try:
        with open(abs_xml) as f:
            xml_str = f.read()
        if terrain_obj is not None:
            if inject_terrain_mesh is None:
                raise ImportError("Terrain support requires wbt_wrappers_inference.utils.eval_utils")
            xml_str = inject_terrain_mesh(xml_str, os.path.abspath(terrain_obj))
            print(f"  Terrain mesh injected: {terrain_obj}")
        model = mujoco.MjModel.from_xml_string(xml_str)
    finally:
        os.chdir(orig_cwd)
    return model


def render_motion(
    npz_path: str,
    xml_path: str,
    output_path: str,
    max_frames: int | None = None,
    width: int = 1280,
    height: int = 720,
    speedup: int = 1,
    terrain_obj: str | None = None,
):
    d = np.load(npz_path, allow_pickle=True)
    joint_pos = d["joint_pos"]
    fps = int(np.atleast_1d(d["fps"]).flatten()[0])
    T = joint_pos.shape[0]
    ncols = joint_pos.shape[1]

    # Detect format
    if ncols == 36:
        fmt = "holosoma"
    elif ncols == 29:
        fmt = "tml"
        if "body_pos_w" not in d or "body_quat_w" not in d:
            print("  WARNING: TML format (29 cols) but missing body_pos_w/body_quat_w. Skipping.")
            return False
        if ISAAC_TO_MUJOCO is None:
            print("  WARNING: TML format requires robot_config (ISAACLAB_TO_MUJOCO_DOF). Skipping.")
            return False
    else:
        print(f"  WARNING: joint_pos has {ncols} cols, expected 36 or 29. Skipping.")
        return False

    if max_frames and T > max_frames:
        T = max_frames

    model = load_model(xml_path, terrain_obj=terrain_obj)
    data = mujoco.MjData(model)

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, height=height, width=width)

    BASE_AZIMUTH = 135.0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1  # pelvis
    cam.distance = 3.0
    cam.elevation = -20.0
    cam.azimuth = BASE_AZIMUTH

    duration = T / fps
    print(f"  Format: {fmt}, rendering {T} frames ({duration:.1f}s) at {fps}fps, speedup={speedup}x ...")

    # For TML format, extract root from body data and remap joints.
    # All beyondmimic/TML format NPZ files use wxyz quaternion convention.
    # Do NOT auto-detect — the heuristic fails when root has large yaw rotation
    # (e.g. SONIC kick/jump where w≈0.63 and z≈0.77, mis-detected as xyzw).
    if fmt == "tml":
        body_pos_w = d["body_pos_w"]  # [T, 30, 3]
        body_quat_w = d["body_quat_w"]  # [T, 30, 4] wxyz convention

    # Extract initial yaw for camera heading tracking
    if fmt == "holosoma":
        w0, x0, y0, z0 = joint_pos[0, 3:7]
    else:
        w0, x0, y0, z0 = body_quat_w[0, 0]
    initial_yaw = np.arctan2(2 * (w0 * z0 + x0 * y0), 1 - 2 * (y0**2 + z0**2))

    # Collect frames
    frames = []
    for t in range(0, T, speedup):
        if fmt == "holosoma":
            # Holosoma format: joint_pos has 7 root DOFs + 29 joints in MuJoCo
            # body-grouped order (same order as TML XML), so no remapping needed.
            data.qpos[0:3] = joint_pos[t, :3]
            data.qpos[3:7] = joint_pos[t, 3:7]
            data.qpos[7:36] = joint_pos[t, 7:36]
        else:
            # TML: root from body_pos_w/body_quat_w (pelvis = body 0, wxyz)
            data.qpos[0:3] = body_pos_w[t, 0]
            data.qpos[3:7] = body_quat_w[t, 0]  # already wxyz
            # TML joint_pos is in Isaac order, convert to MuJoCo order
            data.qpos[7:36] = joint_pos[t][ISAAC_TO_MUJOCO]

        mujoco.mj_forward(model, data)

        # Track robot heading: rotate camera azimuth to follow yaw changes
        w, x, y, z = data.qpos[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        cam.azimuth = BASE_AZIMUTH + np.degrees(yaw - initial_yaw)

        renderer.update_scene(data, camera=cam)
        frame = renderer.render().copy()
        frames.append(frame)

    renderer.close()

    # Save with ffmpeg
    out_fps = fps // speedup
    if out_fps < 1:
        out_fps = 1
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        str(out_fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        print(f"  ffmpeg error: {proc.stderr.read().decode()[:200]}")
        return False

    print(f"  Saved: {output_path} ({len(frames)} frames, {out_fps}fps, {len(frames) / out_fps:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize holosoma NPZ motion in MuJoCo (FAR-pi)")
    parser.add_argument("npz", help="Path to motion NPZ file")
    parser.add_argument("--xml", required=True, help="Path to MuJoCo scene XML")
    parser.add_argument("-o", "--output", required=True, help="Output mp4 path")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to render")
    parser.add_argument("--speedup", type=int, default=1, help="Speedup factor (skip frames)")
    parser.add_argument("--terrain", default=None, metavar="OBJ", help="Terrain OBJ mesh (replaces flat floor)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    print(f"[FAR-pi] Visualizing: {args.npz}")
    ok = render_motion(
        args.npz,
        args.xml,
        args.output,
        max_frames=args.max_frames,
        speedup=args.speedup,
        width=args.width,
        height=args.height,
        terrain_obj=args.terrain,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
