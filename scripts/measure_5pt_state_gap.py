#!/usr/bin/env python3
"""Measure how far a 5pt dataset's OBSERVATION sits from the pose the robot starts in.

    uv run scripts/measure_5pt_state_gap.py <5pt zarr> [--mode fivepoint_geom_xy3_rpy]

``scripts/hl_serving_5pt.py`` feeds ``diffusion_umi_5pt`` the pelvis-frame geometry of the last
COMMANDED keypoints (``obs_mode=last_target``), and the very first observation of a run is the
G1's READY POSE — the pose the low level holds after the operator presses **A**. Whether that
first observation is something the policy has seen is a property of the DATASET's placement
(``attrs.wb_cond_mode``: ``absolute --dz-anchor hand`` for 0902/rotate_box, ``home`` for
push_cart/push_gantry), not of the checkpoint, and it decides whether the first chunk is a
prediction or an extrapolation. So it is measured per dataset rather than assumed from the last
one — this is the ``fivepoint`` twin of the wholebody_geom table in
``flore/s1_policy/data/wb_geom.py``'s module docstring (the 0.28-0.58 sigma numbers).

Three tables, all computed with the SAME functions training and deploy use (``KeypointFK`` +
``wb_geom.geom_from_bodies``; the ready pose from ``ready_pose.resolve_deploy_anchor``):

1. **The state at EPISODE STARTS vs the ready pose**, per column: ready value, dataset mean, std,
   gap in sigmas, and whether the ready value lies inside the whole dataset's ``[q01, q99]`` —
   the range ``norm_stats.json`` maps onto ``[-1, 1]`` (a column outside it is clamped by DDIM's
   ``clip_sample`` on every step).
2. **The RPY wrap risk**, per Euler channel: the fraction of covered frames past 150 deg (the
   ``wb_geom`` warning threshold) and 170 deg. A channel that lives near +-180 deg jumps by 2 pi
   between neighbouring frames, in training AND at deploy.
3. **The keypoints at episode start vs the ready-pose wire**, in the pelvis frame (mm): where the
   DATA's step 0 would land relative to the low level's pre-arm gate reference under
   ``--post-anchor off`` (the 70 mm gate on the three upper points). Under the default
   ``--post-anchor command`` this gap never reaches the wire, but it still says how far the model's
   world is from the robot's.

Needs the viz extra (yourdfpy) and both submodules, exactly like training this preset does.
Read-only; prints a report and exits 0 (2 on a refused input).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from flore.s1_policy.data import wb_geom  # noqa: E402
from flore.s1_policy.data.hl_dataset import _register_image_codecs  # noqa: E402
from flore.s1_policy.data.layout import get_layout  # noqa: E402
from flore.s1_policy.inference import ready_pose  # noqa: E402
from flore.s1_policy.serving.hl_5pt import WIRE_ROLES  # noqa: E402

#: The wire's five slots -> the URDF body whose FK is that keypoint (hands carry the fingertip
#: lever, the chest its own, the feet none) — ``ready_pose.ROLE_BODY`` plus the two ankles.
_BODY_OF_ROLE = {
    "left": wb_geom.LEFT_BODY, "right": wb_geom.RIGHT_BODY, "head": wb_geom.CHEST_BODY,
    "left_foot": "left_ankle_roll_link", "right_foot": "right_ankle_roll_link",
}


def _load(path: str):
    import zarr  # noqa: PLC0415

    _register_image_codecs()
    with zarr.ZipStore(path, mode="r") as store:
        root = zarr.group(store=store)
        attrs = dict(root.attrs)
        if not attrs.get("wb_windows"):
            raise SystemExit(f"{path}: no wb_* arrays (attrs.wb_windows absent) — not a completed 5pt file")
        joints = np.asarray(root["data"]["wb_state_joints"][:], dtype=np.float64)
        valid = np.asarray(root["data"]["wb_state_valid"][:], dtype=np.float64).reshape(joints.shape[0], -1)[:, 0] > 0.5
        ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    return attrs, joints, valid, ends


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zarr", help="a completed (wbhome/_5pt) zarr.zip")
    ap.add_argument("--mode", default="fivepoint_geom_xy3_rpy", help="the layout whose state to measure")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="FK at most this many covered frames for the range/wrap tables (0 = all; "
                         "episode starts are always all measured)")
    args = ap.parse_args(argv)

    layout = get_layout(args.mode)
    if not layout.state_is_geometric or layout.is_wholebody:
        print(f"--mode {args.mode!r} has no geometric keypoint state; this measures fivepoint_geom_* modes",
              file=sys.stderr)
        return 2
    blocks = tuple(layout.state_geom)
    names = tuple(n for b in blocks for n in wb_geom.GEOM_BLOCK_NAMES[b])

    fk, why = ready_pose.KeypointFK.load()
    if fk is None:
        print(f"cannot load the keypoint FK: {why}", file=sys.stderr)
        return 2
    anchor, why = ready_pose.resolve_deploy_anchor(WIRE_ROLES)
    if anchor is None:
        print(f"cannot resolve the G1 ready pose: {why}", file=sys.stderr)
        return 2

    attrs, joints, valid, ends = _load(args.zarr)
    starts = np.concatenate([[0], ends[:-1]]).astype(int)
    starts = starts[(starts >= 0) & (starts < joints.shape[0])]
    start_ok = starts[valid[starts]]
    print(f"{os.path.basename(args.zarr)}: {len(ends)} episodes / {joints.shape[0]} frames, "
          f"wb_cond_mode={attrs.get('wb_cond_mode')!r}, {valid.sum()} covered frames, "
          f"{start_ok.size} valid episode starts")
    print(f"ready pose: {anchor.describe()}")
    print(f"state blocks {blocks} -> {len(names)} columns")

    # --- the ready pose's own observation: the planner's _geom_from_anchor, byte for byte ------
    body_pos = {_BODY_OF_ROLE[r]: np.asarray(anchor.pos[r], dtype=np.float64) for r in ("left", "right", "head")}
    body_rot = {_BODY_OF_ROLE[r]: np.asarray(anchor.rotmat[r], dtype=np.float64) for r in ("left", "right", "head")}
    ready_geom = wb_geom.geom_from_bodies(body_pos, body_rot, blocks)

    # --- FK of the dataset --------------------------------------------------------------------
    idx = np.nonzero(valid)[0]
    if args.max_frames and idx.size > args.max_frames:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, size=args.max_frames, replace=False))
    idx = np.union1d(idx, start_ok)
    feats = np.zeros((joints.shape[0], len(names)))
    kp_pel = np.zeros((joints.shape[0], len(WIRE_ROLES), 3))
    for n, f in enumerate(idx):
        pos, rot = fk.bodies(joints[f])
        feats[f] = wb_geom.geom_from_bodies(pos, rot, blocks)
        for k, role in enumerate(WIRE_ROLES):
            kp_pel[f, k] = pos[_BODY_OF_ROLE[role]]
        if (n + 1) % 10000 == 0:
            print(f"  FK {n + 1}/{idx.size}", flush=True)

    q01 = np.percentile(feats[idx], 1, axis=0)
    q99 = np.percentile(feats[idx], 99, axis=0)
    at_start = feats[start_ok]
    mean, std = at_start.mean(axis=0), at_start.std(axis=0)

    # --- table 1 --------------------------------------------------------------------------------
    print("\n1. STATE at the ready pose vs the dataset's EPISODE STARTS "
          f"({start_ok.size} starts; range = [q01, q99] over {idx.size} covered frames)")
    print(f"{'column':<12}{'ready':>10}{'start mean':>12}{'start std':>11}{'gap/std':>9}{'q01':>10}{'q99':>10}  in range?")
    n_out = 0
    for j, name in enumerate(names):
        unit = 180.0 / np.pi if name.endswith(("roll", "pitch", "yaw")) else 1.0
        gap_sig = abs(ready_geom[j] - mean[j]) / max(std[j], 1e-9)
        inside = q01[j] <= ready_geom[j] <= q99[j]
        n_out += 0 if inside else 1
        print(f"{name:<12}{ready_geom[j] * unit:>10.3f}{mean[j] * unit:>12.3f}{std[j] * unit:>11.3f}"
              f"{gap_sig:>9.2f}{q01[j] * unit:>10.3f}{q99[j] * unit:>10.3f}  {'yes' if inside else 'NO'}")
    print(f"  (metres for *_dx/*_dy, degrees for roll/pitch/yaw)  -> {n_out} of {len(names)} ready-pose "
          f"columns fall OUTSIDE the training [q01, q99] range")

    # --- table 2 --------------------------------------------------------------------------------
    if "rpy" in blocks:
        print("\n2. RPY WRAP RISK over the covered frames (|angle| past 150 / 170 deg)")
        for j, name in enumerate(names):
            if not name.endswith(("roll", "pitch", "yaw")):
                continue
            a = np.abs(feats[idx, j])
            print(f"{name:<12} worst {np.degrees(a.max()):7.1f} deg · >150: {100 * (a > np.radians(150)).mean():6.2f}% "
                  f"· >170: {100 * (a > np.radians(170)).mean():6.2f}%")

    # --- table 3 --------------------------------------------------------------------------------
    print("\n3. KEYPOINTS at episode start vs the ready-pose wire, PELVIS frame (mm; the data's step 0 "
          "under --post-anchor off)")
    lift = np.array([0.0, 0.0, float(anchor.pelvis_z)])
    print(f"{'slot':<12}{'ready xyz (m)':>28}{'start mean xyz (m)':>30}{'|gap| mean':>11}{'p90':>8}{'max':>8}")
    for k, role in enumerate(WIRE_ROLES):
        ready = np.asarray(anchor.pos[role], dtype=np.float64) - lift
        d = np.linalg.norm(kp_pel[start_ok, k] - ready, axis=1) * 1e3
        print(f"{role:<12}{np.array2string(ready, precision=3):>28}"
              f"{np.array2string(kp_pel[start_ok, k].mean(axis=0), precision=3):>30}"
              f"{d.mean():>11.1f}{np.percentile(d, 90):>8.1f}{d.max():>8.1f}")
    print("  (the low level gates the first THREE at 70 mm and only warns on the feet at 300 mm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
