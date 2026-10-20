#!/usr/bin/env python3
"""Headless acceptance for the MC teleop source: the frame chain, and the LATENCY.

Answers the two questions that decide whether 3-tracker teleop of the 5pt / dense
policies is usable, WITHOUT trackers and WITHOUT a robot:

  1. IS THE CHAIN RIGHT?  Feed the source the conditioning that
     `run_replay_zarr_viz.sh --motion-completion` computes for a real take and check that
     what reaches the backend is the SAME thing. This is exact, not approximate -- see
     "the identity trick" below.
  2. IS IT REAL TIME?  Replay that take through the source at 30 Hz on the WALL CLOCK,
     one window at a time on the worker thread, and report the measured per-window
     compute, the duty cycle, and the command lag actually achieved.

THE IDENTITY TRICK. `CondEpisode.wrist_*` is the take's hand pose already re-based onto
MC's home (`cond_mode="home"`), and `left_pos/left_quat` is that same pose with the
gripper mount applied. The source re-bases internally with the SAME `home_anchor_traj`
onto the SAME `home_wrist_pose`. Feeding it `wrist_*` therefore makes its frame-0 equal
its own home, and `p_home + (p - p0)` with `p0 == p_home` is the identity -- so the
source's output must reproduce `left_pos/left_quat` BIT-for-BIT (to float round-off).
Any error in the mount, the home, or the multiplication side shows up immediately, and it
does so without this test restating a single formula.

Two envs, as everywhere in this pipeline: this script needs ZARR, which lives in
UMI_collection's `.venv`, not in hssim. The module under test is numpy-only on purpose so
either interpreter can import it.

    ~/FLoRe/third_party/UMI_collection/.venv/bin/python \
        .../wbt_realtime_eval/vive/mc_teleop_selftest.py <take>.zarr.zip [--episode 0]

Run as a PLAIN SCRIPT, not `python -m wbt_realtime_eval.vive....`: the `-m` form executes
`wbt_realtime_eval/__init__.py` first, which pulls in the whole bundle builder (onnx,
mujoco, mjswan) — none of which exists in the UMI venv, and none of which this test needs.
For the same reason `mc_teleop_source` is imported off `vive/`'s own directory rather than
through the package. `scripts/_mc_teleop_selftest.sh` does all of that for you.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_UMI_DEFAULT = os.path.expanduser("~/FLoRe/third_party/UMI_collection")

# Import the module under test off THIS directory, not through the package: importing
# `wbt_realtime_eval.vive.mc_teleop_source` would execute the top-level
# `wbt_realtime_eval/__init__.py`, which pulls in onnx/mujoco/mjswan — absent from the UMI
# venv this script must run in, and irrelevant to the frame chain.
sys.path.insert(0, _HERE)

from mc_teleop_source import (  # noqa: E402
    MC_MIN_SMOOTH_LAG_FRAMES,
    MC_OVERLAP,
    MC_RATE_HZ,
    MC_STRIDE,
    MC_WINDOW,
    MotionCompletionSource,
)


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zarr", help="a converted 3pt session (.zarr.zip) with BOTH hands + the chest")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--umi-dir", default=os.environ.get("MC_UMI_DIR") or _UMI_DEFAULT)
    ap.add_argument("--mode", default="5pt", choices=["5pt", "dense"])
    ap.add_argument("--lag-frames", type=int, default=MC_STRIDE)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = the whole episode")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="wall-clock replay speed; 1.0 = real time (what the latency claim means)")
    ap.add_argument("--foot-ik", action="store_true",
                    help="enable the completion's causal leg IK (foot_fix=inertialize). OFF by "
                         "default: measured 1937 ms/window p50 vs the 667 ms a window settles = 291%% duty, "
                         "which blows the command lag out from a constant 700 ms to 6100 ms p50 / 8867 ms "
                         "max. This flag exists so that number stays reproducible rather than quoted.")
    ap.add_argument("--no-wrist-ik", action="store_true",
                    help="disable the wrist IK too (the ~20 ms of the ~45 ms window budget it "
                         "costs) -- for isolating where the time goes, not for driving anything")
    args = ap.parse_args(argv)

    # mc_episode reaches for `replay_zarr_viz` (for the TESTED TCP->wrist inverse
    # `wrist_pose_from_stored`), which lives in UMI_collection/replay/ -- so both dirs go
    # on the path, exactly as run_replay_zarr_viz.sh's interpreter sees them.
    for sub in ("motion_completion", "replay"):
        d = os.path.join(args.umi_dir, sub)
        if d not in sys.path:
            sys.path.insert(0, d)
    import mc_episode  # noqa: PLC0415
    import mc_frames as F  # noqa: PLC0415
    import replay_zarr_viz as rzv  # noqa: PLC0415

    print(f"=== take: {args.zarr}  episode {args.episode} ===")
    # `CondEpisode.from_zarr` wants the viewer's `ZarrEpisodes` view, not a raw zarr group:
    # it reads `z.has_robot1` / `z.has_head` (per-device presence flags derived from
    # `data/`'s keys) and slices on `meta/episode_ends`. Using the same wrapper the
    # verified `--motion-completion` path uses is also what keeps this test honest.
    z = rzv.ZarrEpisodes(args.zarr)
    ep = mc_episode.CondEpisode.from_zarr(z, args.episode, cond_mode="home")
    print(ep.describe())
    for w in ep.warnings():
        print(f"  ! {w}")
    n = ep.n if args.max_frames <= 0 else min(ep.n, args.max_frames)
    print(f"replaying {n} / {ep.n} frames at {MC_RATE_HZ:g} Hz x {args.speed:g}")

    src = MotionCompletionSource(
        keypoint_mode=args.mode, hand_point="tcp", lag_frames=args.lag_frames,
        wrist_ik=not args.no_wrist_ik, foot_fix="inertialize" if args.foot_ik else "none",
        # `body`, NOT the live default `puck`: the head slot below is fed `ep.root_pos/quat`,
        # which `CondEpisode` built from the zarr's `head_*` -- ALREADY carrying the
        # converter's chest relabel. Relabelling again would not cancel (order 3) and would
        # silently turn the yaw the identity trick is measuring against.
        head_frame="body",
        umi_dir=args.umi_dir, episode_frames=n + MC_WINDOW,
    )
    ok, reason = src.available()
    if not ok:
        _fail(f"motion completion unusable: {reason}")
    if not src.start():
        _fail("backend did not start")

    # ---- the tracker stand-in -------------------------------------------------------
    # Feed the ALREADY-home-anchored WRIST pose with world_yaw=0. See "the identity
    # trick": the source's own re-basing then collapses to the identity, so its cond must
    # equal ep.left_pos/left_quat exactly. The head slot gets the take's root cond, whose
    # rotation is already yaw-only -- root_cond_from_chest is idempotent on that.
    # Tap the seam where the source hands a conditioning frame to the backend, and record
    # WHICH SAMPLE it was built from. Do not assume "cond frame j came from sample j": the
    # source resamples the caller's rate onto MC's 30 Hz with a zero-order hold, so a slow
    # host legitimately repeat-fills and the two indices diverge. Pairing them here is what
    # makes the comparison below a statement about the CHAIN and not about the clock.
    pushed_cond: list[tuple[int, tuple]] = []
    cursor = [0]
    _orig = src._cond_frame_locked  # noqa: SLF001 - the selftest's whole point is this seam

    def _tap(world_yaw, left, right, head):
        out = _orig(world_yaw, left, right, head)
        if out is not None:
            pushed_cond.append((cursor[0], out))
        return out

    src._cond_frame_locked = _tap  # noqa: SLF001

    dt = 1.0 / (MC_RATE_HZ * max(args.speed, 1e-6))
    lags: list[float] = []
    outs: list[dict] = []
    t_start = time.time()
    for i in range(n):
        cursor[0] = i
        target = t_start + i * dt
        slack = target - time.time()
        if slack > 0:
            time.sleep(slack)
        src.push(
            0.0,
            (ep.wrist_left_pos[i], ep.wrist_left_quat[i]),
            (ep.wrist_right_pos[i], ep.wrist_right_quat[i]),
            (ep.root_pos[i], ep.root_quat[i]),
            now=target,
        )
        got = src.latest(now=target)
        if got is not None:
            lags.append(got["lag_s"])
            outs.append(got)
    # Stop the clock HERE. Everything after this is the tail drain, which is not part of
    # the real-time claim -- folding it in was worth a bogus 3.4x-too-slow reading.
    wall = time.time() - t_start
    # Let the queued windows finish so the tail is scored too. The stop condition is "no
    # window can still run", NOT "newest >= n-1": the last frames of the take belong to a
    # window whose cond runs past the end, so it never becomes runnable and waiting for it
    # would always burn the full timeout.
    n_runnable = sum(1 for st0 in F.window_starts(n + MC_WINDOW, MC_WINDOW, MC_OVERLAP)
                     if st0 + MC_WINDOW <= n)
    deadline = time.time() + 30.0
    while time.time() < deadline and src.stats()["windows"] < n_runnable:
        time.sleep(0.05)
    st = src.stats()
    drain = time.time() - t_start - wall

    # ---- (1) the frame chain --------------------------------------------------------
    if not pushed_cond:
        _fail("no conditioning frames were built")
    idx = np.asarray([j for j, _ in pushed_cond], dtype=int)
    k = idx.size
    got_lp = np.stack([c[0] for _, c in pushed_cond])
    got_lq = np.stack([c[1] for _, c in pushed_cond])
    got_rp = np.stack([c[2] for _, c in pushed_cond])
    got_rq = np.stack([c[3] for _, c in pushed_cond])
    got_root = np.stack([c[4] for _, c in pushed_cond])
    n_repeat = n - k
    e_lp = np.abs(got_lp - ep.left_pos[idx]).max()
    e_rp = np.abs(got_rp - ep.right_pos[idx]).max()
    e_lq = np.abs(np.abs((got_lq * ep.left_quat[idx]).sum(1)) - 1.0).max()
    e_rq = np.abs(np.abs((got_rq * ep.right_quat[idx]).sum(1)) - 1.0).max()
    e_root = np.abs(got_root - ep.root_pos[idx]).max()
    print("\n=== (1) FRAME CHAIN vs the verified offline path (mc_episode, cond_mode=home) ===")
    print(f"  left  gripper pos  max |err| = {e_lp * 1e3:.6f} mm     quat 1-|dot| = {e_lq:.2e}")
    print(f"  right gripper pos  max |err| = {e_rp * 1e3:.6f} mm     quat 1-|dot| = {e_rq:.2e}")
    print(f"  root (chest) pos   max |err| = {e_root * 1e3:.6f} mm")
    print(f"  cond frames built: {k} from {n} samples "
          f"({n_repeat} sample(s) folded by the 30 Hz zero-order hold)")
    chain_ok = max(e_lp, e_rp, e_root) < 1e-6 and max(e_lq, e_rq) < 1e-9
    print(f"  -> {'OK (bit-exact)' if chain_ok else 'MISMATCH'}")

    # ---- (1b) the loop closes: FK'd wrist vs the wrist we commanded -----------------
    print("\n=== (1b) LOOP CLOSES: FK'd left_wrist_yaw_link vs the commanded wrist ===")
    with src._lock:  # noqa: SLF001
        valid = np.flatnonzero(src._valid[:n])  # noqa: SLF001
        fk_pos = src._fk_pos.copy()  # noqa: SLF001
    if valid.size == 0:
        _fail("no window settled -- nothing to score")
    bi = src._body_names.index("left_wrist_yaw_link")  # noqa: SLF001
    d = np.linalg.norm(fk_pos[valid, bi] - ep.wrist_left_pos[valid], axis=1)
    print(f"  frames scored: {valid.size}   p50 = {np.median(d) * 1e3:.3f} mm   "
          f"p95 = {np.percentile(d, 95) * 1e3:.3f} mm   max = {d.max() * 1e3:.3f} mm")
    print("  (sub-mm = the chain is right. ~57 mm = the gripper mount was applied twice or")
    print("   skipped; ~242 mm = the TCP->wrist lever was missed. mc_selftest names these too.)")

    # ---- (2) real-time behaviour ----------------------------------------------------
    print("\n=== (2) REAL TIME (wall clock) ===")
    rt = wall / max(n / MC_RATE_HZ, 1e-9)
    print(f"  replayed {n} frames in {wall:.2f} s  (motion is {n / MC_RATE_HZ:.2f} s) "
          f"-> {rt:.3f} x real time {'OK' if rt < 1.05 else 'SLOW'}")
    print(f"  tail drain after the stream ended: {drain:.2f} s "
          f"({n_runnable} runnable windows, {st['windows']} run)")
    print(f"  windows run: {st['windows']}   errors: {st['errors']}"
          + (f"  last: {st['last_error']}" if st["errors"] else ""))
    print(f"  per-window compute: p50 {st['win_ms_p50']:.1f} ms   max {st['win_ms_max']:.1f} ms")
    print(f"  duty cycle: {st['duty_pct']:.1f} %   "
          f"(compute / the {MC_STRIDE / MC_RATE_HZ * 1e3:.0f} ms of motion a window settles)")
    if lags:
        print(f"  command lag: p50 {np.median(lags) * 1e3:.0f} ms   p95 "
              f"{np.percentile(lags, 95) * 1e3:.0f} ms   max {max(lags) * 1e3:.0f} ms")
        print(f"  configured lag: {st['lag_frames']} frames = {st['lag_ms']:.0f} ms "
              f"(floor for gap-free playback: {MC_MIN_SMOOTH_LAG_FRAMES} frames = "
              f"{MC_MIN_SMOOTH_LAG_FRAMES / MC_RATE_HZ * 1e3:.0f} ms)")
    print(f"  commands produced: {len(outs)} / {n} frames")

    # ---- (3) the wire payload -------------------------------------------------------
    print(f"\n=== (3) WIRE PAYLOAD ({args.mode}) ===")
    if not outs:
        _fail("no command was ever produced")
    last = outs[-1]
    if args.mode == "5pt":
        pos = np.asarray(last["pos"], dtype=np.float64)
        orn = np.asarray(last["orn"], dtype=np.float64)
        if pos.shape != (15,) or orn.shape != (20,):
            _fail(f"5pt payload must be pos[15]/orn[20], got {pos.shape}/{orn.shape}")
        print(f"  names: {last['names']}")
        for j, nm in enumerate(last["names"]):
            print(f"    {nm:24s} pos={np.round(pos[j * 3:j * 3 + 3], 4).tolist()}  "
                  f"quat_xyzw={np.round(orn[j * 4:j * 4 + 4], 4).tolist()}")
        nrm = np.linalg.norm(orn.reshape(5, 4), axis=1)
        print(f"  quat norms: {np.round(nrm, 6).tolist()}  (all 1.0 -> unit)")
        if np.abs(nrm - 1.0).max() > 1e-6:
            _fail("a 5pt orientation slot is not a unit quaternion")
    else:
        jp = np.asarray(last["joint_pos"], dtype=np.float64)
        jv = np.asarray(last["joint_vel"], dtype=np.float64)
        if jp.shape != (29,) or jv.shape != (29,):
            _fail(f"dense payload must be 29+29, got {jp.shape}/{jv.shape}")
        print(f"  joint_pos: max |q| = {np.abs(jp).max():.3f} rad "
              f"({np.degrees(np.abs(jp).max()):.1f} deg)")
        print(f"  joint_vel: max |dq| = {np.abs(jv).max():.3f} rad/s")
        print(f"  joint_names[:4] = {last['joint_names'][:4]} ... (MC order; remap BY NAME)")
    print(f"  frame {last['frame']}  lag {last['lag_s'] * 1e3:.0f} ms")

    src.stop()

    verdict = (chain_ok and st["errors"] == 0 and st["duty_pct"] < 100.0 and bool(outs)
               and rt < 1.05 and k == n)
    print(f"\n=== VERDICT: {'PASS' if verdict else 'ATTENTION'} ===")
    if not chain_ok:
        print("  - the frame chain does NOT reproduce the offline path")
    if st["errors"]:
        print(f"  - {st['errors']} window(s) failed")
    if st["duty_pct"] >= 100.0:
        print(f"  - duty {st['duty_pct']:.0f} % >= 100 %: NOT real time")
    if rt >= 1.05:
        print(f"  - the stream itself ran at {rt:.2f} x real time: the source is blocking its caller")
    if k != n:
        print(f"  - {n - k} sample(s) were folded on a 1:1 feed: the frame clock is drifting")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
