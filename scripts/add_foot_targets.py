#!/usr/bin/env python3
"""Add the ``fivepoint_*`` family's FOOT keypoint targets to a whole-body zarr.

``diffusion_umi_5pt`` predicts five keypoint poses; the rig tracks three. The two feet are
therefore not a recording, they are a DERIVATION of the joint target the ``wholebody`` preset
already trains on. Eight arrays, four per foot::

    wb_<side>_foot_action_pose   (T, H, 6)  chunk target: [pos 3][rotvec 3], pelvis frame
    wb_<side>_foot_action_valid  (T, H)     1.0 where that step is a trustworthy target
    wb_<side>_foot_pos           (T, 3)     ANCHOR pose at t0 (camera grid)
    wb_<side>_foot_rot_axis_angle(T, 3)

The chunk entries are ``KeypointFK(wb_action_joints[t, k])``'s ``*_ankle_roll_link`` pose; the
anchors are ``KeypointFK(wb_state_joints[t])``'s. Five consequences of that definition, all
deliberate:

* **One FK, the deploy one.** :class:`~fuji.s1_policy.inference.ready_pose.KeypointFK` is the
  same class ``hl_dense``'s command anchor, the ready-pose frame and ``wb_geom``'s state
  reduction use, off the same ``g1_29dof_yam.urdf``. The ankles get NO lever (holosoma's
  ``VR_5POINT_OFFSETS_TCP`` is zero there and ``mc_frames.KEYPOINT_OFFSETS`` agrees), so the
  keypoint IS the body origin. A second FK that disagreed by a lever is exactly what
  ``ready_pose._cross_check_bake`` exists to catch, and there is no second FK here.
* **PELVIS frame, no root pose.** ``wb_root_pose`` is not read. The foot pose is a pure function
  of the 29 stored joint angles, which makes it a STANCE ("where the foot sits relative to the
  body") rather than a footstep — a take where the operator walks produces no walking foot
  target. ``docs/RUNBOOK_diffusion_umi_5pt.md`` holds the measurement that makes that the right
  trade (chest net XY 37.9 mm median / 111.9 p90 over 167 episodes; 86% of episodes never leave
  a 100 mm ball). It is safe for the ACTION because ``absolute_to_delta_ee`` is covariant under
  a rigid transform applied to both the pose and its anchor, and ``ref_frame='per_keypoint'``
  gives each keypoint its own anchor — so a foot in the pelvis frame composes correctly beside a
  hand in the recorded world frame.
* **FK of the STORED target, not an interpolation of per-frame feet.** ``wb_action_joints[t,k]``
  is ``np.interp``'d in TIME onto ``t0 + k/hz``, and this dataset's camera grid has a 4.9%
  dt-outlier fraction — so step ``k`` of anchor ``t`` is NOT frame ``t+k`` (measured: 0.55 rad
  worst at every index offset in [-2, +2], while step 0 matches its own state to 0.000e+00).
  FK'ing the per-frame joints and interpolating the RESULT would be ~200x cheaper and a
  DIFFERENT quantity (``FK(interp(q)) != interp(FK(q))``). This way the foot target is by
  construction the FK of the joint target, so the 5-point and the 29-joint families cannot end
  up describing two different bodies.
* **Validity is inherited and INTERSECTED, never invented.** ``wb_<side>_foot_action_valid`` is
  ``wb_action_valid AND wb_state_valid[t]``: a step the completion never reached is zero-filled
  (0 rad is a legal posture, so the values cannot say so), and a step whose ANCHOR frame was
  never covered would be relativized against a fabricated frame — a zero rotvec is the identity
  rotation, i.e. a perfectly legal wrong answer. Both are dropped here rather than downstream.
* **Step 0 is exact.** ``wb_action_joints[t,0] == wb_state_joints[t]`` to 0.000e+00 (the
  exporter's own property, re-measured on this file), so the foot chunk's row 0 IS its anchor
  and ``absolute_to_delta_ee`` returns identity there — the same relationship the hands have.

Cost: 0.655 ms per FK call single-threaded, times the valid ``(T, H)`` pairs — 80 min for a
36820 x 200 file on one core, so the FK is spread over a process pool (``--jobs``, default all
cores). Output goes to a NEW file via ``zarr.copy_store`` (chunk-for-chunk, so no JXL stream is
re-encoded), because a ``.zarr.zip`` cannot gain arrays in place.

Usage::

    uv run scripts/add_foot_targets.py <wb>.zarr.zip -o <wb>_5pt.zarr.zip
    uv run scripts/add_foot_targets.py <wb>_5pt.zarr.zip --check     # verify, no FK
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fuji.s1_policy.data.layout import FOOT_ACTION_KEYS, JOINT_DIM, _FOOT_ZARR_KEYS

#: The two roles, and the URDF body each reads. Same bodies as
#: ``ready_pose._FOOT_BODIES`` and holosoma's ``VR_5POINT_FEET_BODIES``.
FOOT_ROLES: tuple[str, ...] = ("left_foot", "right_foot")
FOOT_BODY: dict[str, str] = {"left_foot": "left_ankle_roll_link",
                             "right_foot": "right_ankle_roll_link"}
#: Attr that marks a file as carrying the eight arrays. The discriminator ``HLDataset`` checks
#: for the ``fivepoint_*`` family, exactly as ``wb_windows`` is the one for ``wholebody``.
FOOT_ATTR = "wb_foot_targets"

#: Worker-global FK, built once per process. A ``KeypointFK`` holds a ``yourdfpy.URDF`` with a
#: scene graph, and ``bodies()`` MUTATES that model's configuration — so it is not thread-safe
#: and each worker builds its own. Tasks are contiguous row ranges for the same reason.
_FK = None


def _worker_init(urdf_path: str | None, anchor_path: str | None) -> None:
    global _FK  # noqa: PLW0603 - one per process, by design (see _FK)
    from fuji.s1_policy.data.wb_geom import load_geom_fk

    fk, why = load_geom_fk(urdf_path=urdf_path, home_anchor_path=anchor_path)
    if fk is None:
        raise RuntimeError(f"worker could not load the keypoint FK: {why}")
    _FK = fk


def _fk_poses(joints: np.ndarray, valid: np.ndarray, fk=None) -> dict[str, np.ndarray]:
    """``role -> (..., 6)`` ankle poses for a ``(..., 29)`` joint array.

    ``valid`` has ``joints``' leading shape; a False entry is left as zeros. Returns
    ``[pos 3][rotvec 3]`` — axis-angle because that is what every other pose array in a UMI zarr
    stores (``robot0_eef_rot_axis_angle``, ``head_rot_axis_angle``) and what ``HLDataset``
    already feeds through ``rotvec_to_rot6d``. A fifth rotation convention in the same file
    would be one more thing to get backwards.
    """
    from scipy.spatial.transform import Rotation

    solver = fk if fk is not None else _FK
    joints = np.asarray(joints, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    lead = joints.shape[:-1]
    if valid.shape != lead:
        raise ValueError(f"valid {valid.shape} must match joints' leading shape {lead}")
    flat_q = joints.reshape(-1, JOINT_DIM)
    flat_v = valid.reshape(-1)
    out = {role: np.zeros((flat_q.shape[0], 6), dtype=np.float32) for role in FOOT_ROLES}
    for i in np.flatnonzero(flat_v):
        pos, rot = solver.bodies(flat_q[i])
        for role in FOOT_ROLES:
            body = FOOT_BODY[role]
            out[role][i, :3] = pos[body]
            out[role][i, 3:] = Rotation.from_matrix(rot[body]).as_rotvec()
    return {role: arr.reshape(*lead, 6) for role, arr in out.items()}


def _fk_rows(payload: tuple) -> tuple:
    """One pool task: ``(lo, joints (n,H,29), valid (n,H))`` -> ``(lo, {role: (n,H,6)})``."""
    lo, joints, valid = payload
    return lo, _fk_poses(joints, valid)


def build(src_path: str, out_path: str, *, jobs: int, rows_per_task: int,
          urdf_path: str | None, anchor_path: str | None, overwrite: bool) -> int:
    import zarr

    from fuji.s1_policy.data.hl_dataset import _register_image_codecs
    from fuji.s1_policy.data.wb_geom import load_geom_fk

    # The SAME registration HLDataset uses (`imagecodecs.numcodecs`; UMI's vendored
    # `imagecodecs_numcodecs` lives in the submodule and is not importable from FUJI's venv).
    # Must run before the first array-metadata load or the read raises
    # "codec not available: 'imagecodecs_jpegxl'".
    _register_image_codecs()
    if os.path.exists(out_path) and not overwrite:
        print(f"[feet] {out_path} exists; pass --overwrite to replace it", file=sys.stderr)
        return 2
    src = zarr.open(src_path, "r")
    attrs = dict(src.attrs)
    if not attrs.get("wb_windows"):
        print(f"[feet] {os.path.basename(src_path)} has no attrs.wb_windows. The foot targets "
              f"are FK'd from `wb_action_joints`, so run UMI_collection's "
              f"run_mc_to_hl_wholebody.sh on this file FIRST.", file=sys.stderr)
        return 2
    data = src["data"]
    missing = [k for k in ("wb_action_joints", "wb_action_valid", "wb_state_joints",
                           "wb_state_valid") if k not in data]
    if missing:
        print(f"[feet] missing {missing}", file=sys.stderr)
        return 2

    aj = data["wb_action_joints"]
    n_rows, horizon = int(aj.shape[0]), int(aj.shape[1])
    act_valid = np.asarray(data["wb_action_valid"][:], dtype=np.float32) > 0.5
    st_valid = np.asarray(data["wb_state_valid"][:], dtype=np.float32).reshape(-1) > 0.5
    # The intersection is the whole validity story -- see the module docstring's fourth point.
    valid = act_valid & st_valid[:, None]
    print(f"[feet] {os.path.basename(src_path)}: T={n_rows} H={horizon} "
          f"cond_mode={attrs.get('wb_cond_mode')} dz_anchor={attrs.get('wb_dz_anchor')}")
    print(f"[feet] valid (t,k): action {100.0 * act_valid.mean():.2f}% AND covered anchor "
          f"-> {100.0 * valid.mean():.2f}% ({int(valid.sum())} FK calls)")

    # --- ANCHOR poses: one FK per frame, in-process (T calls ~ 24 s, not worth a pool) -----
    fk, why = load_geom_fk(urdf_path=urdf_path, home_anchor_path=anchor_path)
    if fk is None:
        print(f"[feet] {why}", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    anchor = _fk_poses(np.asarray(data["wb_state_joints"][:], dtype=np.float64), st_valid, fk=fk)
    print(f"[feet] anchors: {int(st_valid.sum())} frames in {time.perf_counter() - t0:.0f}s")

    # --- CHUNK poses: (T, H) FK over a process pool ---------------------------------------
    chunk = {role: np.zeros((n_rows, horizon, 6), dtype=np.float32) for role in FOOT_ROLES}
    step = max(1, int(rows_per_task))
    ranges = [(lo, min(lo + step, n_rows)) for lo in range(0, n_rows, step)]
    t0, done = time.perf_counter(), 0
    with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init,
                             initargs=(urdf_path, anchor_path)) as pool:
        futures = [pool.submit(_fk_rows, (lo, np.asarray(aj[lo:hi], dtype=np.float64),
                                          valid[lo:hi])) for lo, hi in ranges]
        for fut in futures:
            lo, got = fut.result()
            n = got[FOOT_ROLES[0]].shape[0]
            for role in FOOT_ROLES:
                chunk[role][lo:lo + n] = got[role]
            done += n
            if done % max(1, n_rows // 10) < n:
                el = time.perf_counter() - t0
                print(f"[feet] {done}/{n_rows} rows · {el:.0f}s · "
                      f"eta {el * (n_rows - done) / max(done, 1):.0f}s", flush=True)
    print(f"[feet] chunk FK done in {time.perf_counter() - t0:.0f}s on {jobs} workers")

    # ROW 0 IS THE ANCHOR, and this is the one relationship a shape cannot check: the hands get
    # it from the converter, the feet must get it from the exporter's own
    # `wb_action_joints[t,0] == wb_state_joints[t]` property. If it ever stops holding, every
    # foot chunk is relativized against a frame it does not start at.
    rows = np.flatnonzero(valid[:, 0])
    if rows.size:
        worst = max(float(np.abs(chunk[r][rows, 0] - anchor[r][rows]).max()) for r in FOOT_ROLES)
        print(f"[feet] row0 == anchor: worst {worst:.3e} over {rows.size} anchors")
        if worst > 1e-6:
            print(f"[feet] [ERR] chunk row 0 is not the anchor pose (worst {worst:.3e}); the "
                  f"exporter's action[:,0] == state property no longer holds", file=sys.stderr)
            return 1

    print(f"[feet] copying -> {os.path.basename(out_path)} (chunk-for-chunk; no image "
          f"re-encoded)")
    dst_store = zarr.ZipStore(out_path, mode="w")
    try:
        # EXCLUDE the root `.zattrs` from the copy and write the MERGED dict once, instead of
        # copying it and then calling `attrs.update`. A ZipStore APPENDS: updating an attr after
        # copying it emits a second `.zattrs` member under the same name (the visible symptom is
        # `UserWarning: Duplicate name: '.zattrs'`), zarr then reads whichever the zipfile index
        # happens to resolve first, and a later `zarr.copy_store` OUT of this archive fails with
        # `CopyError: key '.zattrs' exists in destination`. Copying through a MemoryStore would
        # also fix it and would need the whole multi-GB file in RAM.
        zarr.copy_store(src.store, dst_store, excludes=[r"^\.zattrs$"])
        dst = zarr.open(dst_store, "a")
        for role in FOOT_ROLES:
            pose_key, valid_key = FOOT_ACTION_KEYS[role]
            pos_key, rot_key, _ = _FOOT_ZARR_KEYS[role]
            for key, arr in ((pose_key, chunk[role]),
                             (valid_key, valid.astype(np.float32)),
                             (pos_key, anchor[role][:, :3].copy()),
                             (rot_key, anchor[role][:, 3:].copy())):
                dst["data"].create_dataset(key, data=arr, chunks=(1,) + arr.shape[1:],
                                           overwrite=True)
        # `put`, not `update`: the root `.zattrs` was excluded from the copy above, so this is
        # the ONE write of the whole merged dict.
        dst.attrs.put({
            **attrs,
            FOOT_ATTR: True,
            "wb_foot_bodies": [FOOT_BODY[r] for r in FOOT_ROLES],
            "wb_foot_frame": "pelvis_stance_no_root",
            "wb_foot_source": "KeypointFK(wb_action_joints)",
            "wb_foot_tool": "scripts/add_foot_targets.py",
        })
    finally:
        dst_store.close()
    return report(out_path)


def report(path: str) -> int:
    """Verify + describe an already-written file. No FK, no GPU, no model."""
    import zarr

    from fuji.s1_policy.data.hl_dataset import _register_image_codecs

    _register_image_codecs()
    g = zarr.open(path, "r")
    attrs, data = dict(g.attrs), g["data"]
    print(f"\n{os.path.basename(path)}")
    if not attrs.get(FOOT_ATTR):
        print(f"  [ERR] attrs.{FOOT_ATTR} is not set — this file has no foot targets",
              file=sys.stderr)
        return 1
    t, h = (int(x) for x in data["wb_action_joints"].shape[:2])
    bad = 0
    for role in FOOT_ROLES:
        pose_key, valid_key = FOOT_ACTION_KEYS[role]
        pos_key, rot_key, _ = _FOOT_ZARR_KEYS[role]
        for key, want in ((pose_key, (t, h, 6)), (valid_key, (t, h)),
                          (pos_key, (t, 3)), (rot_key, (t, 3))):
            got = tuple(int(x) for x in data[key].shape) if key in data else None
            ok = got == want
            bad += 0 if ok else 1
            print(f"  [{'ok ' if ok else 'ERR'}] {key:32s} {got} (want {want})")
    if bad:
        return 1
    print(f"  frame: {attrs.get('wb_foot_frame')} · source: {attrs.get('wb_foot_source')} · "
          f"bodies: {attrs.get('wb_foot_bodies')}")
    for role in FOOT_ROLES:
        pose_key, valid_key = FOOT_ACTION_KEYS[role]
        v = np.asarray(data[valid_key][:], dtype=np.float32) > 0.5
        rows = np.flatnonzero(v.any(axis=1))[:1500]
        if not rows.size:
            print(f"  [ERR] {role}: no valid step anywhere", file=sys.stderr)
            return 1
        p = np.asarray(data[pose_key][rows], dtype=np.float64)[v[rows]]
        print(f"  {role:11s} valid {100.0 * v.mean():5.2f}%  "
              f"x {p[:, 0].min():+.3f}..{p[:, 0].max():+.3f}  "
              f"y {p[:, 1].min():+.3f}..{p[:, 1].max():+.3f}  "
              f"z {p[:, 2].min():+.3f}..{p[:, 2].max():+.3f} m (pelvis frame)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="a whole-body zarr (attrs.wb_windows set)")
    ap.add_argument("-o", "--out", default=None, help="output path (default: <source>_5pt)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="verify SOURCE's foot arrays and exit; no FK, no copy")
    ap.add_argument("--jobs", type=int, default=0,
                    help="FK worker processes (default: os.cpu_count())")
    ap.add_argument("--rows-per-task", type=int, default=256,
                    help="anchor rows per pool task; larger = less pickling, coarser progress")
    ap.add_argument("--urdf-path", default=None)
    ap.add_argument("--home-anchor-path", default=None)
    args = ap.parse_args(argv)

    if args.check:
        return report(args.source)
    out = args.out
    if not out:
        for suffix in (".zarr.zip", ".zarr"):
            if args.source.endswith(suffix):
                out = f"{args.source[: -len(suffix)]}_5pt{suffix}"
                break
        if not out:
            print("[feet] cannot derive --out from a non-.zarr path; pass -o", file=sys.stderr)
            return 2
    return build(args.source, out, jobs=max(1, args.jobs or (os.cpu_count() or 1)),
                 rows_per_task=args.rows_per_task, urdf_path=args.urdf_path,
                 anchor_path=args.home_anchor_path, overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
