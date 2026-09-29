#!/usr/bin/env python3
"""Drive ``scripts/hl_offline_drive.py`` the way the BROWSER does — with no browser.

The mjswan page is a WebSocket CLIENT of ``:8770`` that reads pose frames and sends
``{"cmd": ...}`` back. So is this script. That makes the whole path verifiable in CI-ish
conditions on a workstation with no MuJoCo/WASM build, no ONNX, no display and no rig — which
matters because the alternative (a headless Chrome + CDP session, see
``docs/RUNBOOK_hl_serving_mc.md``) takes ~60 s of WASM warm-up per attempt and cannot assert on
numbers.

WHAT IT PROVES, in order, and each is a real failure mode that has a shape here:

1. the wire's ``mode`` and widths are what the family claims  — a mismatch makes the low level
   drop every frame, which reads as a hung robot;
2. :data:`~fuji.s1_policy.serving.hl_offline.PAGE_GATE_KEY` is present  — without it the page
   never grows the FUJI panel and there is no way to drive anything;
3. Predict STAGES without publishing  — the command frame must not move on a Predict;
4. the violet ghost ANIMATES while staged  — the preview is a separate channel from the command,
   so this is what says "previewing is not publishing";
5. Run MOVES the command  — and by how much per step, against the low level's slew budget;
6. a sample switch RESETS  — the staging is dropped and the wire returns to a hold pose;
7. STOP interrupts within a step.

Usage (with the driver already running, or --spawn to start one)::

    ./scripts/smoke_test_hl_offline.py --checkpoint <wholebody ckpt> --spawn
    ./scripts/smoke_test_hl_offline.py            # against a driver you started yourself
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_FUJI_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _FUJI_ROOT not in sys.path:
    sys.path.insert(0, _FUJI_ROOT)


def _p(msg: str) -> None:
    print(msg, flush=True)


class Client:
    """The browser's half of the contract: read the newest frame, send control verbs."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.latest: dict | None = None

    async def pump(self) -> None:
        async for raw in self.ws:
            try:
                self.latest = json.loads(raw)
            except (TypeError, ValueError):
                continue

    async def send(self, action: str, **kw) -> None:
        await self.ws.send(json.dumps({"cmd": "fuji", "action": action, **kw}))

    async def do(self, action: str, timeout: float = 120.0, **kw) -> dict:
        """Send a verb and wait until the planner has ACTED on it (``fuji_ack`` advanced).

        ⚠️ Every step of this test needs this rather than ``wait_for(state == ...)``. The planner
        answers a verb on its own loop, so a state predicate matches the frame that is ALREADY on
        the wire and reads the previous state as the answer: the first version of this file
        "failed" a working Predict that way, in under a millisecond.
        """
        base = int((self.latest or {}).get("fuji_ack", 0))
        await self.send(action, **kw)
        return await self.wait_for(lambda m: int(m.get("fuji_ack", 0)) > base,
                                   f"the planner to act on {action!r}", timeout=timeout)

    async def wait_for(self, pred, what: str, timeout: float = 30.0) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            f = self.latest
            if f is not None and pred(f):
                return f
            await asyncio.sleep(0.02)
        raise AssertionError(f"timed out after {timeout:g}s waiting for {what} "
                             f"(last state={None if self.latest is None else self.latest.get('fuji_state')!r}"
                             f", msg={None if self.latest is None else self.latest.get('fuji_msg')!r})")


async def _settle(cli: Client, timeout: float) -> None:
    """Wait out a ``--hold sample`` RAMP, then a beat for the wire to stop changing.

    The ramp is a real move on the wire (0.5 rad/s over a measured 0.514 rad gap ≈ 1 s), so
    "Predict published nothing" can only be asserted once it has arrived — otherwise the ramp's
    own progress reads as the Predict having moved the command.
    """
    await cli.wait_for(lambda m: "ramping" not in str(m.get("fuji_msg", "")),
                       "the hold ramp to finish", timeout=timeout)
    await asyncio.sleep(0.4)


async def _run(args) -> int:
    import numpy as np
    import websockets

    from fuji.s1_policy.serving.hl_offline import PAGE_GATE_KEY

    url = f"ws://{args.host}:{args.port}"
    _p(f"[smoke] connecting to {url} as the low level + the page …")
    deadline = time.time() + args.connect_timeout
    ws = None
    while time.time() < deadline:
        try:
            ws = await websockets.connect(url)
            break
        except OSError:
            await asyncio.sleep(0.5)
    if ws is None:
        _p(f"[smoke] FAIL: nothing listening on {url} after {args.connect_timeout:g}s")
        return 2
    cli = Client(ws)
    pump = asyncio.create_task(cli.pump())
    try:
        # 1 + 2. the wire and the page gate.
        f = await cli.wait_for(lambda m: PAGE_GATE_KEY in m, f"the first frame carrying {PAGE_GATE_KEY}")
        mode = f.get("mode")
        n_samples = int(f[PAGE_GATE_KEY])
        _p(f"[smoke] mode={mode!r} · {n_samples} anchors · {f.get('fuji_n_episodes')} episode(s) "
           f"· layout {f.get('fuji_layout')!r}")
        assert n_samples > 0, "the driver reports no anchors"
        if mode == "dense":
            assert len(f["joint_pos"]) == 29 and len(f["joint_names"]) == 29, "dense width"
            def cmd(m):  # noqa: E306
                return np.asarray(m["joint_pos"], dtype=float)
            unit, budget = "rad", 0.0833
        elif mode == "3pt":
            assert len(f["pos"]) == 9 and len(f["orn"]) == 12, "3pt width"
            def cmd(m):  # noqa: E306
                return np.asarray(m["pos"], dtype=float)
            # ⚠️ 0.005, the value `3_run_policy_vive_force.sh` actually pins as an inline prefix —
            # NOT holosoma_wbt's 0.02 code default, which under-reports the headroom 4x. The mjswan
            # target this smoke test drives implements no slew at all, so nothing here is clamped;
            # the number exists to keep the printed headroom honest about the REAL 3pt low level.
            unit, budget = "m", 0.005
        else:
            _p(f"[smoke] FAIL: unexpected wire mode {mode!r}")
            return 2

        # The startup hold is a RAMP under --hold sample, so wait for it to settle before
        # asserting that a Predict does not move the command.
        await _settle(cli, args.hold_timeout)
        before = cmd(cli.latest)

        # 3. Predict STAGES; the command must not move.
        f = await cli.do("predict", timeout=args.predict_timeout)
        assert f.get("fuji_state") == "staged", f"Predict did not stage: {f.get('fuji_msg')!r}"
        n_chunk = int(f["fuji_chunk_steps"])
        _p(f"[smoke] staged {n_chunk} step(s): {f.get('fuji_msg')}")
        moved = float(np.abs(cmd(cli.latest) - before).max())
        assert moved < 1e-9, f"Predict MOVED the command by {moved:g} {unit} — it must only stage"
        _p(f"[smoke] ok: Predict published nothing (command moved {moved:g} {unit})")

        # 4. the ghost animates while staged (dense only — a keypoint layout has no body).
        if mode == "dense":
            g0 = cli.latest.get("mc_qpos")
            assert g0 is not None, "no mc_qpos on the wire — the page would draw no violet body"
            assert len(cli.latest.get("mc_joint_names", [])) == len(g0), "ghost name/value width"
            steps = {int(cli.latest["fuji_ghost_step"])}
            g_seen = [np.asarray(g0, dtype=float)]
            for _ in range(40):
                await asyncio.sleep(0.05)
                steps.add(int(cli.latest["fuji_ghost_step"]))
                g_seen.append(np.asarray(cli.latest["mc_qpos"], dtype=float))
            span = float(np.abs(np.stack(g_seen) - g_seen[0]).max())
            assert len(steps) > 3, f"the ghost did not advance (steps seen: {sorted(steps)})"
            _p(f"[smoke] ok: the violet ghost animated over {len(steps)} step(s), "
               f"spanning {span:.4f} rad — and the command still has not moved "
               f"({float(np.abs(cmd(cli.latest) - before).max()):g} {unit})")
            assert float(np.abs(cmd(cli.latest) - before).max()) < 1e-9, \
                "the ghost animation leaked onto the command wire"

        # 5. Run moves the command, one step per 30 Hz tick. Sampled while it drips, so `send`
        # (not `do`): `do` would wait for the ack, which lands only AFTER the whole drip.
        await cli.send("run")
        seen: list = []
        t0 = time.time()
        while time.time() - t0 < args.run_timeout:
            m = cli.latest
            if m is not None and int(m.get("fuji_step", -1)) >= 0:
                seen.append((int(m["fuji_step"]), cmd(m)))
            if m is not None and m.get("fuji_state") in ("hold", "stopped") and seen:
                break
            await asyncio.sleep(0.005)
        assert seen, "Run never reported a step"
        uniq: dict = {}
        for i, v in seen:
            uniq.setdefault(i, v)
        order = sorted(uniq)
        travelled = float(np.abs(uniq[order[-1]] - before).max())
        per_step = max((float(np.abs(uniq[b] - uniq[a]).max())
                        for a, b in zip(order, order[1:], strict=False)), default=0.0)
        _p(f"[smoke] ok: Run executed steps {order[0]}..{order[-1]} · command travelled "
           f"{travelled:.4f} {unit} · worst per-step {per_step:.4f} {unit} "
           f"(low level's budget {budget:g} {unit}/step)")
        assert travelled > 1e-6, "Run did not move the command at all"
        if per_step > budget:
            _p(f"[smoke] ⚠ worst per-step {per_step:.4f} exceeds the low level's {budget:g} "
               f"{unit} slew budget — it will lag, not jump, but the chunk is fast for this wire")

        # 6. a sample switch resets.
        want = min(n_samples - 1, int(cli.latest["fuji_sample"]) + 500)
        f = await cli.do("sample", index=want)
        assert int(f["fuji_sample"]) == want, \
            f"cursor is at {f['fuji_sample']}, asked for {want}: {f.get('fuji_msg')!r}"
        assert int(f["fuji_chunk_steps"]) == 0, "a sample switch left a chunk staged"
        _p(f"[smoke] ok: cursor -> anchor {want} (episode {f.get('fuji_episode')} frame "
           f"{f.get('fuji_episode_frame')}, {f.get('fuji_split')}) and the staging was dropped")

        # 7. STOP interrupts a RUNNING chunk. Asserted on the state and the step it reached,
        # not through `do`: an ack can only land once the drip returns, so waiting for one here
        # would accept the RUN's own ack as the answer (which is how the first version of this
        # step passed a driver whose STOP could not interrupt anything).
        await _settle(cli, args.hold_timeout)
        await cli.do("predict", timeout=args.predict_timeout)
        n_exec = int(cli.latest["fuji_exec_steps"])
        await cli.send("run")
        await cli.wait_for(lambda m: m.get("fuji_state") == "running", "the drip to start")
        await asyncio.sleep(0.08)
        await cli.send("stop")
        f = await cli.wait_for(lambda m: m.get("fuji_state") == "stopped", "STOP to land",
                              timeout=args.run_timeout)
        at = int(f["fuji_step"])
        assert at < n_exec - 1, (
            f"STOP landed at step {at} of {n_exec} executing — the chunk ran to completion, so "
            "the stop did not interrupt anything")
        _p(f"[smoke] ok: STOP interrupted at step {at} of {n_exec} executing "
           f"(chunk is {n_chunk}) — {f.get('fuji_msg')}")

        _p("[smoke] PASS")
        return 0
    except AssertionError as exc:
        _p(f"[smoke] FAIL: {exc}")
        return 1
    finally:
        pump.cancel()
        await ws.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--spawn", action="store_true",
                    help="start scripts/hl_offline_drive.py first (needs --checkpoint)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra flags forwarded to the spawned driver")
    ap.add_argument("--connect-timeout", type=float, default=180.0,
                    help="how long to wait for the driver to bind (a cold model load is slow)")
    ap.add_argument("--hold-timeout", type=float, default=60.0)
    ap.add_argument("--predict-timeout", type=float, default=120.0)
    ap.add_argument("--run-timeout", type=float, default=30.0)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    proc = None
    if args.spawn:
        if not args.checkpoint:
            ap.error("--spawn needs --checkpoint")
        cmd = [sys.executable, os.path.join(_HERE, "hl_offline_drive.py"),
               "--checkpoint", args.checkpoint, "--device", args.device,
               "--s0-ws-port", str(args.port), *args.extra]
        _p(f"[smoke] spawning: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd)
    try:
        return asyncio.run(_run(args))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
