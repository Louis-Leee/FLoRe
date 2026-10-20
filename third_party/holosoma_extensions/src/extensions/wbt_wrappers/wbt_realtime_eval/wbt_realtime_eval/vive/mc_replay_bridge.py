#!/usr/bin/env python3
"""A `:8770` bridge for --mode 5pt / dense driven by a RECORDED take. No trackers, no robot.

    bash scripts/_mc_replay_bridge.sh <take>.zarr.zip --mode 5pt          # this module
    bash scripts/_vive_teleop5_build.sh --mode 5pt                        # the sim consumer

WHY THIS EXISTS
===============
The 5pt / dense browser sim (`main_vive5.py`, the MuJoCo/WASM twin of the rig's
`MODE=5pt-unified-anneal`) has existed since `697569d`, but the only thing that could FEED it
was the live bridge — i.e. three VIVE pucks, a calibration pose, and an operator. So the one
thing you actually want to check in sim before touching a physical G1 (does the completion map
my motion onto the robot the way I expect?) needed the rig. This drops in as the producer:
same `:8770`, same payload (`mc_teleop_source.wire_payload`, the ONE definition — a sim fed a
different shape proves nothing), same `MotionCompletionSource` with the same window schedule
and the same 667 ms lag floor. The only thing replaced is where the tracker poses come from.

It is also the only way to make the readout-frame question REPEATABLE. `--walk` injects a
synthetic operator translation on top of the take, so `--ref-frame frozen` vs `live` is an A/B
you can watch in the browser instead of a thing you argue about:

    --ref-frame frozen  (default)  a 0.30 m walk moves the commanded points ~300 mm
    --ref-frame live               the same walk moves them ~3 mm  (the pre-2026-08-29 bug)

WHICH INTERPRETER
=================
UMI_collection's `.venv`, not hssim: reading the take needs zarr, which hssim has none of.
`mc_teleop_source` and `mc_frames` are numpy-only precisely so either side imports the one
copy, and this module adds only `websockets` (present there). Run it as a PLAIN SCRIPT — the
`-m` form executes `wbt_realtime_eval/__init__.py`, which pulls in onnx/mujoco/mjswan. The
launcher does that for you.

EXPECT THE WINDOW TIME TO INFLATE WHEN THE BROWSER IS ON THE SAME HOST
======================================================================
Measured 2026-08-29: this driver alone runs the completion at **46-50 ms/window (7% duty)**,
which is the documented figure; with the MuJoCo/WASM + ONNX page running beside it on the same
workstation the same windows measure **~300 ms p50 (46% duty)**. That is host contention, not
the model and not this driver — the lag it reports stays at the 667-700 ms floor either way. On
a rig the policy runs elsewhere, so the 7% figure is the one that matters.

PICK THE EPISODE FROM THE BROWSER
=================================
`--episode` is only the STARTING take. This bridge also READS its socket, so the page's
episode panel (`custom_observations.ts`) can ask for any take in the dataset without a
restart — which is the whole point on a 128-episode session, where restarting the bridge per
take means re-spawning the completion backend every time. Two control messages, both already
in the browser's `sendCmd` vocabulary:

    {"cmd": "episode", "index": N}   switch to take N (0-based, dataset order)
    {"cmd": "recalibrate"}           restart the CURRENT take

A switch is a reload + `src.reset()` + a fresh home anchor, so it costs the same ~1 s of
window fill as startup — during which the LAST command keeps being published, i.e. the robot
holds instead of snapping to HOME. The request is queued and applied by the feed task, never
by the socket task: loading an episode mutates the completion source, and doing that between
a `push()` and its `latest()` would score new cond frames against the old window schedule.

WHAT IT DOES NOT DO
===================
No calibration, no world yaw, no staleness gate, no HOME fallback: the take is already
home-anchored (`cond_mode="home"`) and always fresh. So this is NOT a substitute for
rehearsing the operator's calibration flow — it exercises everything DOWNSTREAM of it. And it
publishes no COMMAND until the first window has settled, exactly like the live bridge, so the
consumer still sees the ~1 s of nothing at startup (what it does publish in that gap is the
episode-list metadata below, which carries no pose and so cannot move anything).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_UMI_DEFAULT = os.path.expanduser("~/FLoRe/third_party/UMI_collection")

# Import the source off THIS directory rather than through the package, for the reason in the
# docstring: `wbt_realtime_eval/__init__.py` reaches for onnx/mujoco/mjswan.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mc_teleop_source import (  # noqa: E402
    MC_DEFAULT_REF_FRAME,
    MC_RATE_HZ,
    MC_REF_FRAMES,
    MC_STRIDE,
    MC_WINDOW,
    MotionCompletionSource,
    wire_payload,
)


def _load(umi_dir: str):
    """`mc_episode` + `replay_zarr_viz` out of the UMI checkout.

    `mc_episode` reaches for `replay_zarr_viz` (for the TESTED TCP->wrist inverse
    `wrist_pose_from_stored`), so BOTH dirs go on the path — exactly as
    `run_replay_zarr_viz.sh`'s interpreter sees them, and exactly as `mc_teleop_selftest`
    does it.
    """
    for sub in ("motion_completion", "replay"):
        d = os.path.join(umi_dir, sub)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"no {sub}/ under {umi_dir} (set MC_UMI_DIR / --umi-dir)")
        if d not in sys.path:
            sys.path.insert(0, d)
    if importlib.util.find_spec("zarr") is None:
        raise ImportError(
            "no zarr in this interpreter -- run this under UMI_collection's .venv, not hssim "
            "(scripts/_mc_replay_bridge.sh does)"
        )
    import mc_episode
    import replay_zarr_viz

    return mc_episode, replay_zarr_viz


def walk_offsets(n: int, dx: float, start: int, ramp: int) -> np.ndarray:
    """(n, 3) synthetic operator translation: 0, then a linear ramp to `dx` in +x.

    A RAMP, not a step: a teleport is a 9 m/s tracker velocity, which is a dropout as far as
    the completion is concerned, so the thing being demonstrated would be confounded by the
    thing being avoided. `ramp` frames at 30 Hz over `dx` metres is the commanded speed —
    the default 30/0.30 is 0.30 m/s, an ordinary walking pace.
    """
    out = np.zeros((n, 3), dtype=np.float64)
    if abs(dx) > 0.0:
        out[:, 0] = dx * np.clip((np.arange(n) - start) / max(ramp, 1), 0.0, 1.0)
    return out


def apply_control(state: dict, msg) -> str:
    """Apply one control message to `state`; returns the line to log ("" = nothing to say).

    Bookkeeping ONLY: it queues `state["want_episode"]` and never touches the completion
    source, so the feed task stays the single owner of the episode (see `_serve`). Module-level
    and pure so it can be tested without a socket — the range check is the part worth a test,
    because an out-of-range index reaching the feed task would take the driver down mid-run,
    from a click in a browser.
    """
    if not isinstance(msg, dict):
        return f"ignoring non-object control message {msg!r}"
    cmd = str(msg.get("cmd") or "")
    if cmd == "episode":
        try:
            idx = int(msg.get("index"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return f"ignoring malformed episode request {msg!r}"
        n_ep = int(state["n_episodes"])
        if not 0 <= idx < n_ep:
            return f"ignoring episode {idx}: this dataset has {n_ep} (0..{n_ep - 1})"
        state["want_episode"] = idx
        return f"<- episode {idx} requested"
    if cmd == "recalibrate":
        # The page's Recalibrate button. A replay has no operator pose to re-capture, so the
        # honest equivalent is restarting the take — which IS a fresh home anchor, the same
        # thing --loop does at the end of every lap.
        state["want_episode"] = int(state["episode"])
        return f"<- recalibrate: restarting episode {state['episode']}"
    return f"ignoring unknown control message {msg!r}"


async def _serve(state: dict, host: str, port: int, rate_hz: float, stop: asyncio.Event):
    """Broadcast the newest payload at `rate_hz` and read control messages, per client.

    Same shape as the live bridge's server: a client that stalls does not back-pressure the
    completion, because each connection only ever sends the LATEST frame. The READ half is
    what the browser's episode panel talks to — the page has had `sendCmd` since the live
    Recalibrate button, and a bridge with no reader swallowed every one of those messages
    silently (a click that does nothing, with nothing logged anywhere).
    """
    import websockets

    async def pump(ws):
        """Send-only half: the newest frame, or the metadata-only hello until one exists.

        The hello carries the episode list and NO pose, which is what lets the picker appear
        during the ~1 s window fill instead of after it — the browser's `ingest` guards every
        pose field on its own length, so a frame without them updates nothing but the panel.
        """
        period = 1.0 / max(rate_hz, 1e-6)
        while not stop.is_set():
            frame = state.get("frame") or state.get("hello")
            if frame is not None:
                await ws.send(json.dumps(frame))
            await asyncio.sleep(period)

    async def control(ws):
        """Receive-only half. QUEUES a request; never switches the episode itself.

        The feed task owns the episode because loading one mutates the completion source
        (reset + a fresh home anchor). Doing that from here could land between a `push()` and
        its `latest()`, i.e. score the new take's cond frames against the old window
        schedule — a seam that would look like a completion artifact.
        """
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except (TypeError, ValueError):
                print(f"[mc-replay] ignoring unparseable control message {raw!r}", flush=True)
                continue
            line = apply_control(state, msg)
            if line:
                print(f"[mc-replay] {line}", flush=True)

    async def handler(ws):
        state["clients"] += 1
        print(f"[mc-replay] client connected (clients={state['clients']})", flush=True)
        tasks = [asyncio.create_task(pump(ws)), asyncio.create_task(control(ws))]
        try:
            # Whichever half ends first (a dropped client ends both), tear the other down.
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
            # Retrieve the ConnectionClosed the loser raised: an un-awaited task exception
            # prints a traceback at GC time, and a client going away is not an error.
            await asyncio.gather(*tasks, return_exceptions=True)
            state["clients"] -= 1
            print(f"[mc-replay] client gone (clients={state['clients']})", flush=True)

    async with websockets.serve(handler, host, port):
        print(f"[mc-replay] serving ws://{host}:{port} at {rate_hz:g} Hz", flush=True)
        await stop.wait()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("zarr", help="a converted 3pt session (.zarr.zip) with BOTH hands + chest")
    ap.add_argument("--mode", default="5pt", choices=["5pt", "dense"])
    ap.add_argument("--episode", type=int, default=0,
                    help="the STARTING episode. The browser's episode panel switches takes "
                         "live over the same socket (or send {\"cmd\":\"episode\",\"index\":N} "
                         "yourself), so this no longer decides what you can watch — which "
                         "matters on a 100+ episode session, where a restart per take means "
                         "re-spawning the completion backend every time.")
    ap.add_argument("--loop", action="store_true",
                    help="restart the episode when it ends (a new MC episode + home anchor "
                         "each time, i.e. the same thing a Recalibrate does) instead of "
                         "holding the last command")
    ap.add_argument("--ref-frame", default=None, choices=list(MC_REF_FRAMES),
                    help=f"which pelvis frame the 5pt keypoints are read out in (default "
                         f"{MC_DEFAULT_REF_FRAME}, or $MC_REF_FRAME). Inert for dense.")
    ap.add_argument("--walk", type=float, default=0.0, metavar="DX",
                    help="inject a synthetic operator walk of DX metres in +x on ALL THREE "
                         "pucks (default 0 = replay the take as recorded). This is the "
                         "readout-frame A/B: with --ref-frame frozen the commanded points "
                         "move by DX, with live they barely move at all.")
    ap.add_argument("--walk-start", type=int, default=45, help="MC frame the walk starts on")
    ap.add_argument("--walk-ramp", type=int, default=30, help="frames the walk takes (30 = 1 s)")
    ap.add_argument("--publish-mc-qpos", action="store_true",
                    help="BOTH modes: also ship the whole completed body (29 joints + names "
                         "+ its root pose in the readout frame) for a viewer to draw — "
                         "vive/mc_teleop_gate_viz.py in viser, or the browser sim's "
                         "light-purple reference ghost. Additive, off by default; pass it "
                         "whenever you want to see what the completion asked for next to "
                         "what the low-level controller actually did.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1 -- loopback, like every server in "
                         "this repo). A non-loopback bind is allowed but warns.")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--rate-hz", type=float, default=50.0, help="wire rate (the rig's is 50)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="wall-clock replay speed. 1.0 = real time, which is what the lag "
                         "numbers mean; anything else is for inspection only.")
    ap.add_argument("--mc-lag-frames", type=int, default=MC_STRIDE)
    ap.add_argument("--mc-foot-ik", action="store_true",
                    help="OFFLINE ONLY (291%% duty). Here purely so the number stays "
                         "reproducible; it will make the command fall seconds behind.")
    ap.add_argument("--umi-dir", default=os.environ.get("MC_UMI_DIR") or _UMI_DEFAULT)
    args = ap.parse_args(argv)

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[mc-replay] ⚠️  binding NON-LOOPBACK ({args.host}:{args.port})",
              file=sys.stderr, flush=True)
    mc_episode, rzv = _load(args.umi_dir)

    z = rzv.ZarrEpisodes(args.zarr)
    n_ep = int(z.n_ep)
    if not 0 <= args.episode < n_ep:
        print(f"[mc-replay] ERROR: --episode {args.episode} is out of range; this dataset "
              f"has {n_ep} episode(s) (0..{n_ep - 1})", file=sys.stderr)
        return 2

    def load_episode(idx: int):
        """One take's cond stream. Cheap — pose arrays only, no camera decode."""
        got = mc_episode.CondEpisode.from_zarr(z, idx, cond_mode="home")
        print(f"[mc-replay] take   : {args.zarr}  episode {idx}/{n_ep - 1}")
        print(f"[mc-replay] {got.describe()}")
        for w in got.warnings():
            print(f"[mc-replay]   ! {w}")
        return got

    ep = load_episode(args.episode)
    n = int(ep.n)
    # Size the completion's buffers for the LONGEST take in the dataset, not for this one:
    # `MotionCompletionSource` fixes them at construction (`reset()` reallocates at the same
    # size), so a per-episode size would truncate any longer take the browser switches to —
    # silently, as a motion that stops partway. Over-allocating is what the +MC_WINDOW below
    # already did; the buffers are ~250 kB per 600 frames.
    max_frames = int((z.ends - z.starts).max()) if n_ep else n
    if max_frames > n:
        print(f"[mc-replay] episode buffers sized for the LONGEST take ({max_frames} frames) "
              f"so a switch to any of the {n_ep} episodes fits")

    src = MotionCompletionSource(
        keypoint_mode=args.mode, hand_point="tcp", lag_frames=args.mc_lag_frames,
        wrist_ik=True, foot_fix="inertialize" if args.mc_foot_ik else "none",
        ref_frame=args.ref_frame,
        # `body`, NOT the live default `puck`: `CondEpisode` builds its root out of the zarr's
        # `head_*`, which the CONVERTER already relabelled into the chest body frame
        # (`attrs.T_tracker_head`). The relabel has order 3, so a second application would
        # not cancel -- it would turn the whole take ~120 deg and read as a model failure.
        head_frame="body",
        umi_dir=args.umi_dir,
        episode_frames=max_frames + MC_WINDOW, log=lambda m: print(m, flush=True),
    )
    ok, why = src.available()
    if not ok:
        print(f"[mc-replay] ERROR: motion completion unusable: {why}", file=sys.stderr)
        return 2
    if not src.start():
        print("[mc-replay] ERROR: backend did not start", file=sys.stderr)
        return 2

    shift = walk_offsets(n, args.walk, args.walk_start, args.walk_ramp)
    if args.walk:
        print(f"[mc-replay] synthetic WALK +{args.walk:.3f} m in x over frames "
              f"{args.walk_start}..{args.walk_start + args.walk_ramp} "
              f"({args.walk / (args.walk_ramp / MC_RATE_HZ):.2f} m/s)")

    state = {
        "frame": None,
        "clients": 0,
        "episode": int(args.episode),
        "n_episodes": n_ep,
        # Published until the first command settles, and re-published never after. No `mode`
        # key on purpose: the browser routes the COMMAND arrays on that tag, and it also uses
        # it to warn about a missing `mc_qpos` ghost — a tagged pose-less frame would trip
        # that warning before the first real frame ever arrived.
        "hello": {"seq": 0, "replay": True, "episode": int(args.episode),
                  "n_episodes": n_ep, "episode_frames": n, "episode_frame": 0},
        "want_episode": None,
    }
    stop = asyncio.Event()

    async def feed():
        """Push the take at 30 Hz on the wall clock and publish whatever has settled."""
        nonlocal ep, n, shift
        seq = 0
        i = 0
        dt = 1.0 / (MC_RATE_HZ * max(args.speed, 1e-6))
        t0 = time.time()
        last_log = 0.0
        while not stop.is_set():
            want = state.get("want_episode")
            if want is not None:
                # The page picked a take (or asked for a restart). `state["frame"]` is left
                # ALONE: the consumer keeps getting the last command through the ~1 s window
                # fill, so the robot holds its pose instead of snapping back to HOME. The
                # frame's own `episode` field therefore stays the OLD one until a new command
                # settles — which is the truth, and is what the panel's status line reports.
                state["want_episode"] = None
                print(f"[mc-replay] switching to episode {want} "
                      f"(hold the last command for ~1 s while its first window fills)",
                      flush=True)
                ep = load_episode(want)
                n = int(ep.n)
                shift = walk_offsets(n, args.walk, args.walk_start, args.walk_ramp)
                src.reset()
                state["episode"] = int(want)
                i, t0 = 0, time.time()
            if i >= n:
                if not args.loop:
                    if state["frame"] is not None:
                        await asyncio.sleep(0.2)
                        continue
                    break
                print("[mc-replay] episode end -> restarting (new home anchor)", flush=True)
                # Blocks the event loop for one backend round trip (tens of ms), so the
                # broadcast hiccups once per lap. Left synchronous on purpose: a reset that
                # raced the feed could push cond frames into the OLD episode, and this is an
                # inspection tool where one visible hitch beats a subtle seam.
                src.reset()
                i, t0 = 0, time.time()
            target = t0 + i * dt
            slack = target - time.time()
            if slack > 0:
                await asyncio.sleep(slack)
            d = shift[i]
            src.push(
                0.0,
                (ep.wrist_left_pos[i] + d, ep.wrist_left_quat[i]),
                (ep.wrist_right_pos[i] + d, ep.wrist_right_quat[i]),
                (ep.root_pos[i] + d, ep.root_quat[i]),
                now=target,
            )
            got = src.latest(now=target)
            if got is not None:
                seq += 1
                frame = wire_payload(
                    args.mode, got, seq=seq, t=target, orientation=True,
                    publish_qpos=bool(args.publish_mc_qpos),
                )
                # Replay-only telemetry, attached HERE and not inside `wire_payload`: that is
                # the ONE wire definition this shares with the live bridge (a sim fed a
                # different shape proves nothing), and a live rig has no episodes. The browser
                # ignores keys it does not know, and its episode panel only mounts once
                # `n_episodes` arrives — which is what keeps the picker off the rig's page.
                frame["replay"] = True
                frame["episode"] = int(state["episode"])
                frame["n_episodes"] = n_ep
                frame["episode_frames"] = n
                frame["episode_frame"] = i
                state["frame"] = frame
            now = time.time()
            if now - last_log > 5.0:
                last_log = now
                st = src.stats()
                print(f"[mc-replay] ep {state['episode']} frame {i}/{n} seq={seq} "
                      f"clients={state['clients']} "
                      f"windows={st['windows']} err={st['errors']} "
                      f"win_p50={st['win_ms_p50']:.0f} ms duty={st['duty_pct']:.0f}% "
                      f"lag={got['lag_s'] * 1e3:.0f} ms" if got else
                      f"[mc-replay] ep {state['episode']} frame {i}/{n} filling the first "
                      f"window (no command published yet — this is the 1 s of the MC schedule)",
                      flush=True)
            i += 1
        stop.set()

    async def run():
        await asyncio.gather(
            _serve(state, args.host, args.port, args.rate_hz, stop),
            feed(),
        )

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        src.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
