"""Turn an .h2r motion view (qpos only) back into the holosoma WBT NPZ that bake_motions reads.

The view's manifest says joint_pos = MuJoCo qpos [root xyz, root wxyz, 29 joints] at 50 Hz. The
fields its converter dropped are recomputed: joint velocities by finite differences, and the
per-body world poses (body_pos_w / body_quat_w, plus their velocities) by MuJoCo forward
kinematics on the scene the browser bundle uses, in that scene's body order. MuJoCo's `world`
body is left out, as in the training clips: the bundle builder wants one body list for all of
its motions.

    python h2r_view_to_holosoma.py <view_dir> <scene.xml> <out_dir> <motion> [<motion> ...]
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def angular_velocity(quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
    """World-frame angular velocity from consecutive orientations (T, B, 4) -> (T, B, 3)."""
    q = quat_wxyz / np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
    conj = q * np.array([1, -1, -1, -1])
    dq = quat_mul(q[1:], conj[:-1])                       # rotation from frame t to t+1, world frame
    dq *= np.sign(dq[..., :1] + 1e-12)                    # shortest arc
    angle = 2 * np.arctan2(np.linalg.norm(dq[..., 1:], axis=-1), dq[..., 0])
    axis = dq[..., 1:] / np.maximum(np.linalg.norm(dq[..., 1:], axis=-1, keepdims=True), 1e-12)
    w = axis * (angle * fps)[..., None]
    return np.concatenate([w, w[-1:]], axis=0)            # the last frame repeats the one before


def convert(src: Path, dst: Path, fps: float, joint_names: list[str], model: mujoco.MjModel) -> None:
    qpos = np.load(src)["qpos"].astype(np.float64)
    data = mujoco.MjData(model)
    T = qpos.shape[0]
    pos = np.zeros((T, model.nbody - 1, 3))
    quat = np.zeros((T, model.nbody - 1, 4))
    for t in range(T):
        data.qpos[:] = model.qpos0
        data.qpos[: qpos.shape[1]] = qpos[t]
        mujoco.mj_kinematics(model, data)
        pos[t], quat[t] = data.xpos[1:], data.xquat[1:]   # body 0 is the world
    lin = np.gradient(pos, 1.0 / fps, axis=0)
    ang = angular_velocity(quat, fps)
    dof_vel = np.gradient(qpos[:, 7:], 1.0 / fps, axis=0)
    body_names = [model.body(i).name for i in range(1, model.nbody)]
    np.savez(dst, joint_pos=qpos.astype(np.float32), joint_vel=dof_vel.astype(np.float32),
             fps=np.array([fps], dtype=np.float32), joint_names=np.array(joint_names),
             body_names=np.array(body_names), body_pos_w=pos.astype(np.float32),
             body_quat_w=quat.astype(np.float32), body_lin_vel_w=lin.astype(np.float32),
             body_ang_vel_w=ang.astype(np.float32))


def main() -> None:
    view_dir, scene, out_dir, *names = sys.argv[1:]
    manifest = json.loads((Path(view_dir) / "manifest.json").read_text())
    fps = 1.0 / float(manifest["timestep"])
    joint_names = manifest["qpos_names"][7:]
    model = mujoco.MjModel.from_xml_path(scene)
    model_joints = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    if model_joints[: len(joint_names)] != joint_names:
        raise SystemExit(f"joint order of {scene} differs from the view's qpos_names")
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0] != 0:
        raise SystemExit(f"{scene}: the robot's free joint is expected first in qpos")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for n in names:
        convert(Path(view_dir) / manifest["motions_subdir"] / f"{n}.npz", Path(out_dir) / f"{n}.npz", fps, joint_names, model)
        print(f"{n}: {fps:g} Hz, {len(joint_names)} joints, {model.nbody - 1} bodies")


if __name__ == "__main__":
    main()
