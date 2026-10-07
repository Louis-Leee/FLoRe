#!/usr/bin/env python3
"""Copy holosoma motion clips into a demo motion directory under readable names, each turned to
face the demo's camera.

The demo's camera keeps one bearing while its look-at point follows the pelvis, so a clip is seen
from wherever its first frame happens to face. Each clip is turned about the vertical axis so that
its pelvis starts at the origin facing +x, as the builder's ``idle_default`` stands; the camera of
``build_demo.py`` (azimuth 45°, i.e. placed 45° to the robot's left of straight ahead) then sees
it from the front at three quarters. Turning a whole clip changes nothing the controller sees,
since its targets are expressed relative to the robot. ``fps``, the joint and body names and the
joint angles are copied unchanged.

    python scripts/prepare_demo_motions.py <out_dir> <name>=<clip.npz> [<name>=<clip.npz> ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

VECTORS = ("body_lin_vel_w", "body_ang_vel_w")


def yaw_of(q_wxyz: np.ndarray) -> float:
    w, x, y, z = q_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def turn(src: Path, dst: Path) -> float:
    """Write `src` turned and moved so its pelvis starts at the origin facing +x; returns the turn."""
    d = dict(np.load(src))
    names = [str(n) for n in d["body_names"]]
    pelvis = names.index("pelvis")
    a = (np.pi - yaw_of(d["body_quat_w"][0, pelvis])) % (2 * np.pi) - np.pi   # in [-pi, pi)
    c, s = np.cos(a), np.sin(a)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    q_turn = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
    origin = d["body_pos_w"][0, pelvis] * np.array([1.0, 1.0, 0.0])

    def place(p: np.ndarray) -> np.ndarray:
        return ((p - origin) @ rot.T).astype(p.dtype)

    d["body_pos_w"] = place(d["body_pos_w"])
    d["body_quat_w"] = quat_mul(q_turn, d["body_quat_w"]).astype(d["body_quat_w"].dtype)
    for k in VECTORS:
        d[k] = (d[k] @ rot.T).astype(d[k].dtype)
    jp, jv = d["joint_pos"].copy(), d["joint_vel"].copy()
    jp[:, :3] = place(jp[:, :3])                                    # free joint: position, then wxyz
    jp[:, 3:7] = quat_mul(q_turn, jp[:, 3:7])
    if jv.shape[1] == jp.shape[1] - 1:                              # [lin, ang (both world), joints]
        jv[:, :3] = jv[:, :3] @ rot.T
        jv[:, 3:6] = jv[:, 3:6] @ rot.T
    d["joint_pos"], d["joint_vel"] = jp, jv
    np.savez(dst, **d)
    return float(np.degrees(a))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("clips", nargs="+", help="<name>=<clip.npz>; the name is what the demo lists")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.clips:
        name, _, src = spec.partition("=")
        if not name or not src:
            raise SystemExit(f"expected <name>=<clip.npz>, got {spec!r}")
        print(f"{name}: turned by {turn(Path(src), args.out_dir / f'{name}.npz'):+.1f} deg")


if __name__ == "__main__":
    main()
