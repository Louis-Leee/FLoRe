#!/usr/bin/env python3
"""OFFLINE-SAMPLE DRIVER: a dataset anchor -> one chunk -> a REAL low level, steered from mjswan.

The third corner of a triangle whose other two the repo already had::

    scripts/infer_s1_wholebody.py     sample -> chunk -> a viser DRAWING           (no controller)
    scripts/hl_serving_wholebody.py   sensors -> chunk -> the :8770 wire            (no ground truth)
    THIS SCRIPT                       sample -> chunk -> the :8770 wire, whose other end is the
                                      mjswan MuJoCo/WASM low level running in a browser tab.

So it answers the one question neither of those can: **what does the whole-body controller
actually DO with a chunk we already know is good?** — repeatably, because the observation is a
stored sample rather than whatever the cameras happened to see, and with no rig at all: no
trackers, no Botas, no D405s, no gantry, no G1.

⚠️ OPEN LOOP, BY CONSTRUCTION — read this before reading a result.
   The observation is the DATASET's, at the cursor. It is NOT the robot's. So the chunk cannot
   react to what the controller did with the last one, and pressing Predict twice at one cursor
   returns the identical chunk (same seed, deterministic session). ``auto`` therefore advances
   the cursor by the executed step count and re-infers, which is an open-loop REHEARSAL of a
   recorded observation sequence — the right test for "does the low level track this", and the
   wrong one for "does the policy recover". Closing the loop needs the live planners
   (``hl_serving_wholebody.py`` / ``hl_serving_mc.py``) and their sensors.

TWO FAMILIES, ONE SCRIPT, picked from the checkpoint's own layout::

    hl_data.mode = wholebody              29 joints  -> DenseWire      -> mode="dense"
      (diffusion_umi_wholebody)                                           MODE=dense-unified-anneal
      browser: scripts/_vive_teleop5_build.sh --mode dense

    hl_data.mode = both_hands_head[_dual] 3 EE poses -> ThreePointWire -> mode="3pt"
      (diffusion_umi_3tracker)                                            MODE=3pt-unified-anneal
      browser: scripts/_vive_teleop3_build.sh

The 3-tracker route here is DIRECT: no motion completion, because a 3-keypoint policy and a
3-keypoint low level want the same three poses. That is the whole reason it is worth having next
to ``hl_serving_mc.py``, which must insert a completion (a second model, a second interpreter,
~80 ms/chunk and a 0.40 rad null-space choice to clutch away) only because the 5pt/dense low
levels have no keypoint front-end. See ``hl_offline.ThreePointWire``.

EVERY CONTROL IS IN THE BROWSER PAGE
===================================
``S0Driver`` binds ``:8770`` as the server and the page is its client, so the page's existing
``sendCmd`` channel is the control path — the same one the MC replay bridge's episode picker
uses. This script therefore has NO viser window and needs no display; its stdout is a log, not a
UI. The panel appears in the page automatically (it is gated on the ``fuji_n_samples`` wire key,
which nothing else sends), and offers: the sample / episode picker, Predict, Run, ▶▶ auto, Stop,
Reset. See ``docs/RUNBOOK_hl_offline_drive.md``.

  {"cmd":"fuji","action":"predict"}                {"cmd":"fuji","action":"run"}
  {"cmd":"fuji","action":"reset"}                  {"cmd":"fuji","action":"stop"}
  {"cmd":"fuji","action":"sample","index":N}       {"cmd":"fuji","action":"episode","index":N}
  {"cmd":"fuji","action":"auto"}

That vocabulary is also all a 20-line ``websockets`` client needs, which is what
``scripts/smoke_test_hl_offline.py`` is: the whole path is verifiable with no browser.

WHAT THE PAGE DRAWS
===================
* **wholebody** — the LIGHT-PURPLE translucent G1, animated through the staged chunk, on the
  page's existing ``mc_qpos`` ghost channel (joints keyed BY NAME; a wholebody chunk IS 29 joint
  angles). It is a channel SEPARATE from the command, so previewing is not publishing.
* **3pt** — the page's own commanded-target triads (magenta L TCP / yellow R TCP / green head),
  which move when the chunk RUNS. No body: a keypoint policy predicts three poses and no
  posture, and drawing one would mean asking a second model to invent it.

TWO HOLD POSES, and the measured reason the choice exists
=========================================================
``--hold sample`` (the default here) ramps the wire to the SAMPLE's own step-0 pose first, so the
robot stands where the demonstration stood before anything is predicted — which is what makes the
prediction legible. For ``wholebody`` that pose is in the dataset's ``cond_mode=home`` posture
space, measured **0.514 rad at the knee** from the ready crouch the low level holds after **A**,
so it is RAMPED at ``--ramp-rad-s`` rather than stepped (see ``hl_offline.ramp_step``), and the
clutch is then re-based there so step 0 of the chunk is still a bit-exact no-op.

``--hold ready`` is the deploy-faithful alternative: keep the ready pose and clutch the chunk
onto it, exactly as ``hl_serving_wholebody.py`` does. Use it when the question is "would this
chunk be safe on the rig", not "what does this chunk look like".
(For ``3pt`` the two coincide and the flag says so: that chunk is decoded against the ready-pose
anchor, so its own step 0 IS the ready pose.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_FUJI_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _FUJI_ROOT not in sys.path:
    sys.path.insert(0, _FUJI_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# The gripper CALIBRATION, the jaw state machine and the bridge WS client are imported from the
# left_only planner rather than copied — the same rule scripts/hl_serving_mc.py follows, and for
# the same reason: each has its own review history (the closing overshoot, the run-length confirm,
# the tightest-width latch) and a fix must land in every planner at once. Importing the module is
# side-effect-free: it binds nothing and parses no args until its own main().
import hl_serving as HL  # noqa: E402

from fuji.client.s0_driver import S0Driver  # noqa: E402
from fuji.s1_policy.data.layout import (  # noqa: E402
    ACTION_DT,
    ACTION_HZ,
    WB_JOINT_NAMES,
)
from fuji.s1_policy.inference import ready_pose as RP  # noqa: E402
from fuji.s1_policy.serving import hl_dense, hl_offline, hl_wholebody  # noqa: E402
from fuji.s1_policy.serving.hl_online import decode_chunk  # noqa: E402
from fuji.s1_policy.serving.live_robot import G1_29DOF_SDK_DOF_NAMES  # noqa: E402
from fuji.utils.rot6d import rotmat_to_quat_xyzw, rot6d_to_rotmat, rotmat_to_rot6d  # noqa: E402

#: The two low-level families this script can drive, and the browser build that serves each.
FAMILIES = {
    "dense": ("MODE=dense-unified-anneal", "scripts/_vive_teleop5_build.sh --mode dense"),
    "3pt": ("MODE=3pt-unified-anneal", "scripts/_vive_teleop3_build.sh"),
}

#: Which family a layout's chunk belongs on. Derived, never guessed: the width of a wholebody
#: action and a 3-tracker one are both plausible and the wire tag is what routes them.
FAMILY_BY_LAYOUT = {
    "wholebody": "dense",
    "both_hands_head": "3pt",
    "both_hands_head_dual": "3pt",
}

HOLD_MODES = ("sample", "ready")

#: The ghost animates at the ACTION grid's rate, not the heartbeat's: it is showing the chunk as
#: it would execute, so a faster loop would make a 50-step chunk look like a twitch.
GHOST_HZ = ACTION_HZ


def _log(msg: str) -> None:
    print(f"[hl-offline] {msg}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="an HL-v2 checkpoint dir (wholebody, or a 3-tracker keypoint one)")
    p.add_argument("--algo", default=None, choices=["diffusion", "flow", "act"],
                   help="which algorithm block the checkpoint carries (default: read from "
                        "config.json — the launcher never has to pass it)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--zarr-paths", nargs="*", default=None,
                   help="override the checkpoint's recorded zarr paths (same order). Normally "
                        "unnecessary: they are relocated by basename, and provenance is still "
                        "sha256-checked, so this cannot swap in a different dataset silently.")
    p.add_argument("--num-steps", type=int, default=None,
                   help="sampler steps (default: the checkpoint's own)")
    p.add_argument("--seed", type=int, default=0,
                   help="base sampler seed. Predict N uses seed + N, so re-pressing Predict at "
                        "one cursor shows the SAMPLER's spread; --fixed-seed pins it instead.")
    p.add_argument("--fixed-seed", action="store_true",
                   help="use --seed for every Predict, so one cursor is bit-reproducible")

    g = p.add_argument_group("the low level")
    g.add_argument("--low-level", default=None, choices=sorted(FAMILIES),
                   help="which wire family to publish (default: derived from the checkpoint's "
                        "layout). Passing the wrong one is refused by name — the low level's own "
                        "width guard would just drop every frame, which reads as a hung robot.")
    g.add_argument("--s0-ws-host", default="127.0.0.1")
    g.add_argument("--s0-ws-port", type=int, default=8770)
    g.add_argument("--sample", type=int, default=0,
                   help="the STARTING anchor. The page's picker moves it live, so this only "
                        "decides where it opens.")
    g.add_argument("--episode", type=int, default=None,
                   help="start at this episode's first anchor instead of --sample")

    c = p.add_argument_group("cadence")
    c.add_argument("--step-hz", type=float, default=ACTION_HZ,
                   help="chunk playback rate; MUST be the training grid's 30 Hz")
    c.add_argument("--drip-hz", type=float, default=60.0,
                   help="heartbeat / re-send rate. Only has to beat the low level's 0.25 s stale "
                        "gate; S0Driver stops re-broadcasting after 0.5 s of silence.")
    c.add_argument("--exec-steps", type=int, default=16,
                   help="how many of the chunk's steps one Run executes (live on the page)")
    c.add_argument("--auto-max-seconds", type=float, default=20.0,
                   help="wall-clock ceiling for the page's ▶▶ auto run. Nothing else bounds it: "
                        "the cursor walks to the end of the dataset otherwise.")

    d = p.add_argument_group("the hold pose + the clutch")
    d.add_argument("--hold", default="sample", choices=HOLD_MODES,
                   help="'sample' (default) ramps the wire to the SAMPLE's own step-0 pose before "
                        "predicting, so the body stands where the demo did. 'ready' keeps the G1 "
                        "ready pose, which is what the rig deploy does. Inert for --low-level 3pt "
                        "(that chunk's step 0 IS the ready pose).")
    d.add_argument("--ramp-rad-s", type=float, default=0.5, metavar="RAD_S",
                   help="[--hold sample, dense] how fast the wire walks to the hold pose. The gap "
                        "is measured 0.514 rad at the knee, so at the default this is a ~1 s move "
                        "instead of a step the low level would spend 7+ ticks slewing.")
    d.add_argument("--wb-clutch", default="joint", choices=list(hl_dense.CLUTCH_MODES),
                   help="[dense] 'joint' (default) publishes the chunk's CHANGE from the hold "
                        "pose, so step 0 is a bit-exact no-op. 'none' publishes the model's "
                        "absolute angles.")
    d.add_argument("--legs", default="predict", choices=list(hl_wholebody.LEG_MODES),
                   help="[dense] what reaches the 12 leg joints. 'hold' pins them to the clutch "
                        "base; it needs --wb-clutch joint.")
    d.add_argument("--post-anchor", default="off", choices=list(hl_dense.POST_ANCHOR_MODES),
                   help="where the NEXT chunk's wire continues from. 'off' (default) re-reads the "
                        "hold pose; 'command' carries the last executed step forward. With no "
                        "RecordTap behind a browser sim a carried anchor FREEZES rather than "
                        "integrating blind, which is the safe direction and is reported.")
    d.add_argument("--no-zero-arms", dest="zero_arms", action="store_false",
                   help="build the ready pose from the raw crouch default_qpos instead of the "
                        "arms-zeroed one. Only correct against HOLOSOMA_INIT_READY_POSE=0.")
    d.set_defaults(zero_arms=True)

    f = p.add_argument_group("force")
    # OFF by default here, unlike the deploy planners. A browser sim's contacts are not the
    # task's and its servo is inert in free space (measured: F meas 0.0 N throughout a 3pt sim
    # session), so a force command would be published into nothing and read back as a success.
    f.add_argument("--enable-force-cmd", action="store_true",
                   help="publish the chunk's predicted F^cmd as well. OFF by default: a browser "
                        "sim is free space, so this proves nothing there. The low level also "
                        "needs FORCE_CMD_WIRE=1.")
    f.add_argument("--force-gate-n", type=float, default=1.5,
                   help="below this |F| the command is hard-zeroed (pure position)")

    j = p.add_argument_group("the jaws (an OUT-OF-BAND channel, NOT the :8770 wire)")
    j.add_argument("--command-gripper", dest="command_gripper", action="store_true",
                   help="drive the real jaws from the chunk's own predicted widths, over each "
                        "gripper bridge's WebSocket. OFF by default. ⚠️ :8770 carries NO gripper "
                        "field in either wire family, so this is a SECOND channel that reaches "
                        "real CAN hardware even when the low level is a browser sim (mjswan has "
                        "no jaw) — it is the one part of this script that can move something "
                        "physical without a robot being involved in the wire at all.")
    j.add_argument("--gripper-ws", default="ws://127.0.0.1:8784", help="LEFT jaw bridge")
    j.add_argument("--gripper-right-ws", default="ws://127.0.0.1:8785", help="RIGHT jaw bridge")
    # Defaults identical to hl_serving / hl_serving_mc: these are measured rig properties (the
    # recorded "closed" mode is 42-48 mm, never 0), not a per-script preference.
    j.add_argument("--grip-overshoot-mm", type=float, default=10.0,
                   help="drive a CONFIRMED close this far past the predicted width, so the jaw "
                        "stalls into the object and the position error becomes grip force")
    j.add_argument("--grip-close-below-mm", type=float, default=70.0,
                   help="predicted widths under this are the 'closing' mode (the recorded signal "
                        "is bimodal: closed 42-48 mm, open 84-94 mm, nothing between)")
    j.add_argument("--grip-confirm-steps", type=int, default=3,
                   help="consecutive agreeing steps before the overshoot engages or releases")
    return p


# ---------------------------------------------------------------------------
# the two sessions
# ---------------------------------------------------------------------------
def _read_algo(checkpoint: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        return str(json.loads((Path(checkpoint) / "config.json").read_text()).get("algo"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"could not read algo from {checkpoint}/config.json: {exc}") from None


def _read_layout_mode(checkpoint: str) -> str:
    try:
        raw = json.loads((Path(checkpoint) / "config.json").read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"could not read {checkpoint}/config.json: {exc}") from None
    return str((raw.get("hl_data") or {}).get("mode", ""))


def _open_session(checkpoint: str, algo: str, layout_mode: str, *, device: str,
                  zarr_paths, num_steps):
    """The offline inference session for this layout. Two classes, one call site.

    Both are the SAME strict, sha256-checked, provenance-bound loader the offline viewers use —
    which is the point of reusing them rather than building a third: a checkpoint/dataset
    mismatch is refused here by exactly the code that refuses it there.
    """
    paths = tuple(zarr_paths) if zarr_paths else None
    if layout_mode == "wholebody":
        from fuji.s1_policy.inference.wholebody import WholeBodyInferenceSession

        return WholeBodyInferenceSession(checkpoint, algorithm=algo, zarr_paths=paths,
                                         device=device, num_steps=num_steps)
    if algo == "diffusion":
        from fuji.s1_policy.inference.diffusion import DiffusionInferenceSession as S
    elif algo == "flow":
        from fuji.s1_policy.inference.flow import FlowInferenceSession as S
    elif algo == "act":
        from fuji.s1_policy.inference.act import ACTInferenceSession as S
    else:
        raise SystemExit(f"unknown algo {algo!r}")
    return S(checkpoint, zarr_paths=paths, device=device, num_steps=num_steps)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    algo = _read_algo(args.checkpoint, args.algo)
    layout_mode = _read_layout_mode(args.checkpoint)
    family = args.low_level or FAMILY_BY_LAYOUT.get(layout_mode)
    if family is None:
        raise SystemExit(
            f"checkpoint layout {layout_mode!r} has no low-level family mapping here (known: "
            f"{sorted(FAMILY_BY_LAYOUT)}). A left_only / both_hands checkpoint predicts a subset "
            "of the 3 points and belongs on scripts/hl_serving.py, which HOLDS the rest.")
    want = FAMILY_BY_LAYOUT.get(layout_mode)
    if want is not None and family != want:
        raise SystemExit(
            f"--low-level {family} does not match layout {layout_mode!r} (which predicts a "
            f"{want} wire). Nothing downstream would say so: the low level validates only the "
            "frame WIDTH, so a mismatch drops every frame and reads as a hung robot.")

    _log(f"loading {algo} checkpoint {args.checkpoint} ({layout_mode}) on {args.device} …")
    session = _open_session(args.checkpoint, algo, layout_mode, device=args.device,
                            zarr_paths=args.zarr_paths, num_steps=args.num_steps)
    try:
        return _run(args, session, algo, layout_mode, family)
    finally:
        session.close()


def _run(args, session, algo: str, layout_mode: str, family: str) -> int:
    layout = session.source.layout
    cursor = hl_offline.SampleCursor(session.source.refs)
    if args.episode is not None:
        if not cursor.set_episode(int(args.episode)):
            raise SystemExit(
                f"--episode {args.episode} is outside [0, {cursor.n_episodes - 1}]")
    elif not cursor.set_index(int(args.sample)) and int(args.sample) != 0:
        raise SystemExit(f"--sample {args.sample} is outside [0, {len(cursor) - 1}]")
    _log(f"dataset: {len(cursor)} anchors in {cursor.n_episodes} episode(s) — "
         f"{', '.join(Path(p).name for p in session.source.resolved_paths)}")
    _log(f"cursor opens at {cursor.at().describe()}")
    if abs(args.step_hz - 1.0 / ACTION_DT) > 0.5:
        raise SystemExit(
            f"--step-hz {args.step_hz} is not the training grid's {1.0 / ACTION_DT:.0f} Hz — the "
            "chunk is published one step per tick, so any other rate plays it back at the wrong "
            "speed.")

    # --- the ready pose ---------------------------------------------------------------
    # For `dense` it is the clutch base + the idle wire. For `3pt` it is ALSO the DECODE ANCHOR,
    # which makes it a physical claim rather than a rendering choice: the chunk is
    # anchor-relative, so what it is decoded against decides where the hands actually go.
    anchor, why = RP.resolve_deploy_anchor(("left", "right", "head"), zero_arms=args.zero_arms)
    if anchor is None or anchor.dof is None:
        raise SystemExit(f"cannot resolve the G1 ready pose: {why}")
    _log(f"ready pose: {anchor.describe()}")
    for note in anchor.notes:
        _log(f"  ⚠️  {note}")

    ll_mode, browser = FAMILIES[family]
    driver = (_DenseDriver if family == "dense" else _ThreePointDriver)(
        args, session, layout, anchor, algo)
    _log(f"wire: mode={driver.wire.mode!r} · the low level MUST be {ll_mode} · "
         f"its pre-arm gate is {driver.wire.gate_note}")
    _log(f"browser low level: bash third_party/holosoma_extensions/{browser}")
    _log(f"layout: {layout_mode} · action {layout.action_dim}-D · state {layout.state_dim}-D · "
         f"H={driver.horizon} @ {ACTION_HZ:g} Hz · sampler {session.num_steps} step(s)")

    # --- the jaws (a second channel; see _Jaws) --------------------------------------
    jaws = None
    if args.command_gripper:
        if not layout.gripper_roles:
            raise SystemExit(
                f"--command-gripper: layout {layout_mode!r} predicts no gripper column, so there "
                "is nothing to command. (A width of 0.0 is a legal fully-closed target, which is "
                "why this is refused rather than published as a zero.)")
        jaws = _Jaws(args, layout.gripper_roles)
        jaws.start()
        _log(f"jaws COMMANDED for {list(jaws.roles)}: "
             + ", ".join(f"{r}={jaws.urls[r]}" for r in jaws.roles)
             + f" · overshoot {args.grip_overshoot_mm:g} mm below "
               f"{args.grip_close_below_mm:g} mm, confirmed over {args.grip_confirm_steps} step(s)")
        _log("  ⚠️  this is NOT the :8770 wire — it is the gripper bridge's own socket, so it "
             "reaches REAL CAN hardware even while the low level is a browser sim.")
    else:
        _log(f"jaws READ-ONLY (no --command-gripper): the chunk's {len(layout.gripper_roles)} "
             "predicted width(s) are decoded and discarded; each bridge sits at target_frac=freeze.")

    # --- the page protocol ----------------------------------------------------------
    ctl_lock = threading.Lock()
    ctl = {"n_samples": len(cursor), "n_episodes": cursor.n_episodes,
           "sample": cursor.index, "req": None, "want_sample": None, "want_episode": None,
           "stop": False}
    stop_flag = threading.Event()          # the operator's STOP, latched

    def _on_control(msg: dict) -> None:
        # Runs on the WS event loop, which also owns the broadcaster: BOOKKEEPING ONLY, with the
        # ONE exception below.
        with ctl_lock:
            line = hl_offline.apply_control(ctl, msg)
            want_stop = bool(ctl.get("stop"))
        if want_stop:
            # ⚠️ STOP IS ACTED ON HERE, not in the control loop. That loop BLOCKS for the whole
            # drip inside `_do_run` (533 ms for a 16-step chunk at 30 Hz), so a STOP routed
            # through it cannot interrupt the motion it exists to interrupt — measured on the
            # first end-to-end run, the chunk ran to completion with the STOP already queued.
            # `Event.set()` is non-blocking and thread-safe, which is what makes this the one
            # verb the socket thread may touch; the loop still consumes the latch, for the page
            # message and to retire an auto run.
            stop_flag.set()
        if line:
            _log(line)

    # CONSTRUCTED here, STARTED after the hold pose is resolved: `start()` binds the port, and a
    # frame published before the startup reset has run would carry a placeholder readout. A page
    # (or scripts/smoke_test_hl_offline.py) reading that first frame as the state is exactly how
    # the first end-to-end run "settled" before the ramp had even been requested.
    s0 = S0Driver(host=args.s0_ws_host, port=args.s0_ws_port, on_control=_on_control)

    dripping = threading.Event()
    busy = threading.Lock()
    wire_lock = threading.Lock()
    page_lock = threading.Lock()
    shutdown = threading.Event()
    last_wire = {"frame": driver.wire.idle(), "fcmd": None}
    staged: dict = {"out": None}
    # `ack` counts control requests this loop has FINISHED acting on. It is the only field of
    # `page` that is a protocol rather than a readout — see hl_offline.page_extras.
    page = {"state": "hold", "msg": f"waiting for the low level on :{args.s0_ws_port}",
            "chunk_steps": 0, "step": -1, "ghost": 0, "ack": 0}
    seed_ctl = {"n": 0}
    ramp = {"target": None}                # a pending move to the hold pose (dense only)

    def _set_wire(frame, fcmd) -> None:
        with wire_lock:
            last_wire["frame"] = tuple(np.asarray(a, dtype=np.float64).copy() for a in frame)
            last_wire["fcmd"] = None if fcmd is None else np.asarray(fcmd, np.float64).copy()

    def _set_page(**kw) -> None:
        with page_lock:
            page.update(kw)

    def _extras() -> dict:
        """The page's telemetry + (dense) one animated ghost step. Rebuilt per publish.

        The ghost is advanced HERE rather than in the drip so it plays while the chunk is merely
        STAGED — which is the whole point of a preview — and it is a separate wire channel from
        the command, so a page showing violet motion is not a robot being moved.
        """
        with page_lock:
            snap = dict(page)
        out = hl_offline.page_extras(
            cursor=cursor.at(), n_samples=len(cursor), n_episodes=cursor.n_episodes,
            state=snap["state"], message=snap["msg"],
            layout_note=f"{layout_mode} {layout.action_dim}-D · H={driver.horizon} · {algo}",
            chunk_steps=int(snap["chunk_steps"]), exec_steps=int(args.exec_steps),
            step=int(snap["step"]), ghost_step=int(snap["ghost"]), ack=int(snap["ack"]))
        ghost = driver.ghost(staged["out"], int(snap["ghost"]))
        if ghost is not None:
            out.update(ghost)
        return out

    def _send(frame, fcmd) -> bool:
        ok, reason = driver.wire.publish(s0, frame, fcmd, extra=_extras())
        if not ok:
            _log(f"REFUSED a {driver.wire.mode} frame: {reason} — sending nothing; the low "
                 "level's stale gate holds.")
        return ok

    # --- the heartbeat --------------------------------------------------------------
    def _heartbeat() -> None:
        period = 1.0 / max(args.drip_hz, 1.0)
        ghost_period = 1.0 / max(GHOST_HZ, 1.0)
        next_ghost = 0.0
        while not shutdown.is_set():
            try:
                now = time.monotonic()
                if now >= next_ghost:
                    next_ghost = now + ghost_period
                    with page_lock:
                        n = int(page["chunk_steps"])
                        page["ghost"] = (int(page["ghost"]) + 1) % n if n > 0 else 0
                if not dripping.is_set():
                    # THE RAMP, on the heartbeat: --hold sample is a MOVE, not a target, and it
                    # must keep progressing while the operator does nothing. It is abandoned the
                    # instant a chunk starts dripping (that chunk was clutched to the ramp's
                    # destination, so continuing to slide the base underneath it would make the
                    # first executed step a jump of exactly the remainder).
                    tgt = ramp["target"]
                    if tgt is not None:
                        with wire_lock:
                            cur = last_wire["frame"]
                        nxt, done = driver.ramp(cur, tgt, args.ramp_rad_s / max(args.drip_hz, 1.0))
                        _set_wire(nxt, None)
                        # ⚠️ REBASE EVERY STEP, not on arrival. The clutch base must equal the
                        # frame actually on the wire at all times, or a Predict that lands
                        # mid-ramp stages a chunk whose step 0 is the base — i.e. a jump of
                        # exactly the ramp's remaining distance, published the instant Run is
                        # pressed. Measured that way on the first end-to-end run: 0.15 rad.
                        driver.rebase(nxt)
                        if done:
                            ramp["target"] = None
                            _log(f"hold pose reached: {driver.wire.status(nxt)}")
                            _set_page(msg=f"holding: {driver.wire.status(nxt)}")
                    with wire_lock:
                        frame = tuple(a.copy() for a in last_wire["frame"])
                        fc = None if last_wire["fcmd"] is None else last_wire["fcmd"].copy()
                    _send(frame, fc if args.enable_force_cmd else None)
            except Exception as exc:  # noqa: BLE001 - the heartbeat must never die
                _log(f"heartbeat error: {exc!r}")
            time.sleep(period)

    # --- predict / run / reset ------------------------------------------------------
    def _predict() -> str:
        seed_ctl["n"] += 1
        seed = int(args.seed) if args.fixed_seed else int(args.seed) + seed_ctl["n"]
        _set_page(state="predicting", msg="predicting …")
        t0 = time.monotonic()
        try:
            result = session.infer(cursor.index, seed=seed, num_steps=args.num_steps)
        except Exception as exc:  # noqa: BLE001 - a failed sample must not publish
            _set_page(state="hold", msg=f"SAMPLE FAILED: {type(exc).__name__}: {exc}")
            return f"SAMPLE FAILED: {type(exc).__name__}: {exc} — nothing staged."
        ms = (time.monotonic() - t0) * 1e3
        note = driver.wire.sync(None, None)
        try:
            out = driver.stage(result)
        except Exception as exc:  # noqa: BLE001
            _set_page(state="hold", msg=f"DECODE FAILED: {type(exc).__name__}: {exc}")
            return f"DECODE FAILED: {type(exc).__name__}: {exc} — nothing staged."
        staged["out"] = out
        _set_page(state="staged", chunk_steps=int(out["plan"]["n"]), ghost=0, step=-1,
                  msg=f"staged {out['plan']['n']} step(s) in {ms:.0f} ms · {out['note']}")
        return (f"staged: anchor {cursor.index} seed {seed} · sample {ms:.0f} ms · "
                f"{out['plan']['n']} step(s) · base {note} · {out['note']}")

    def _drip(out) -> int:
        """Publish the first ``--exec-steps`` steps, one per 30 Hz step."""
        plan = out["plan"]
        n_steps = max(1, min(int(args.exec_steps), int(plan["n"])))
        step_dt = 1.0 / max(args.step_hz, 1.0)
        drip_dt = 1.0 / max(args.drip_hz, 1.0)
        last = 0
        ramp["target"] = None      # a Run supersedes an unfinished move to the hold pose
        dripping.set()
        _set_page(state="running")
        try:
            for i in range(n_steps):
                if stop_flag.is_set():
                    break
                frame = driver.wire.frame(plan, i)
                fcmd = driver.fcmd(out, i)
                _set_wire(frame, fcmd)
                # The jaws go out ONCE per action step, not per re-send: `_send` below repeats
                # the same frame at --drip-hz to beat the low level's stale gate, and a jaw
                # command is a state-machine input — replaying it would advance the confirmation
                # run several times per step and engage the overshoot early.
                if jaws is not None:
                    jaws.command(driver.grip_widths(out, i))
                _set_page(step=i, ghost=i)
                last = i
                t_step = time.time()
                while True:
                    _send(frame, fcmd if args.enable_force_cmd else None)
                    if stop_flag.is_set():
                        break
                    remaining = step_dt - (time.time() - t_step)
                    if remaining <= 0:
                        break
                    time.sleep(min(drip_dt, remaining))
        finally:
            dripping.clear()
        # Commit to the LAST EXECUTED step, not the chunk end: the next chunk must branch from
        # where the body actually stopped. Under --post-anchor off this is the within-chunk
        # advance and the next Predict's sync overwrites it.
        driver.wire.commit(driver.wire.frame(plan, last))
        _set_page(state="stopped" if stop_flag.is_set() else "hold", step=last)
        if jaws is not None:
            _log(f"drip finished at step {last} · {jaws.note()}")
        return last + 1

    def _reset(why: str) -> str:
        stop_flag.clear()
        staged["out"] = None
        # OPEN THE JAWS, and only if we ever closed them. A reset abandons the chunk, and an
        # abandoned grasp left clamped is the one piece of state on this path that outlives the
        # process — the wire goes back to the ready pose on its own, a squeezed jaw does not.
        if jaws is not None:
            jaws.open_all(why)
        # `state` too, not just the counters: a reset DROPS the staging, so leaving the state at
        # 'staged' told the page there was a chunk to Run when there was not (seen in the browser
        # after a sample switch, with chunk_steps already back to 0).
        with page_lock:
            page.update(state="hold", chunk_steps=0, step=-1, ghost=0)
        target = driver.hold_target(cursor)
        if target is None:
            _set_wire(driver.wire.idle(), None)
            driver.reset(None)
            ramp["target"] = None
            return f"{why}: wire + clutch back at the ready pose"
        # RAMPED, never stepped (hl_offline.ramp_step), and the clutch is re-based to WHERE THE
        # WIRE IS rather than to where it is going — the heartbeat keeps the two equal on every
        # ramp step. Setting the base to `target` here instead is what made the first end-to-end
        # run jump 0.15 rad on a Predict that landed mid-ramp.
        #
        # It also ramps from wherever the wire currently is, not from the ready pose: switching
        # samples is then a move between two demonstration postures rather than a trip home.
        ramp["target"] = target
        return (f"{why}: ramping to the sample's own step-0 pose at {args.ramp_rad_s:g} rad/s "
                f"({driver.hold_gap_note(target)})")

    # --- the control loop -----------------------------------------------------------
    # THE STARTUP HOLD FIRST, then bind, then heartbeat: see the S0Driver construction above.
    _startup = _reset("startup")
    _set_page(msg=_startup)
    _log(_startup)
    s0.start()
    threading.Thread(target=_heartbeat, daemon=True, name="hb").start()
    _log("")
    _log("READY. Everything is driven from the browser page — this process has no window.")
    _log(f"  1. start the low level:  bash third_party/holosoma_extensions/{browser}")
    _log(f"  2. it connects to :{args.s0_ws_port} and the FUJI panel appears in the page")
    _log("  3. pick a sample → Predict → Run  (or ▶▶ auto)")
    _log("")
    auto = {"until": 0.0}
    last_status = 0.0
    n_client_seen = -1
    try:
        while True:
            with ctl_lock:
                req = ctl.pop("req", None)
                ctl["req"] = None
                want_s = ctl.get("want_sample")
                want_e = ctl.get("want_episode")
                ctl["want_sample"] = ctl["want_episode"] = None
                if ctl.get("stop"):
                    ctl["stop"] = False
                    stop_flag.set()
                    auto["until"] = 0.0
            if req == "sample" and want_s is not None:
                moved = cursor.set_index(int(want_s))
                _set_page(msg=_reset(f"sample {cursor.index}"
                                     + ("" if moved else " (already there)")))
                _log(f"cursor -> {cursor.at().describe()}")
            elif req == "episode" and want_e is not None:
                cursor.set_episode(int(want_e))
                _set_page(msg=_reset(f"episode {cursor.at().episode}"))
                _log(f"cursor -> {cursor.at().describe()}")
            elif req == "reset":
                _set_page(msg=_reset("reset"))
            elif req == "stop":
                _set_page(state="stopped", msg="STOP — motion interrupted, staging dropped")
                staged["out"] = None
                # The squeeze LATCH only, not an open: Stop is an interrupt, and forcing the jaw
                # open would drop whatever it is holding onto the table. The next grasp still has
                # to earn its confirmation run.
                if jaws is not None:
                    jaws.release()
            elif req == "predict":
                if busy.acquire(blocking=False):
                    try:
                        stop_flag.clear()
                        _log(_predict())
                    finally:
                        busy.release()
                else:
                    _set_page(msg="busy")
            elif req == "run":
                _set_page(msg=_do_run(args, s0, staged, busy, stop_flag, _drip, _set_page,
                                      driver))
            elif req == "auto":
                if auto["until"] > 0.0:
                    _set_page(msg="an auto run is already active — press Stop")
                elif s0.num_clients == 0:
                    _set_page(msg=f"the low level is NOT connected to :{args.s0_ws_port}")
                else:
                    stop_flag.clear()
                    auto["until"] = time.time() + float(args.auto_max_seconds)
                    _set_page(msg=f"auto run: predict -> run, until Stop or "
                                  f"{args.auto_max_seconds:g} s")
                    _log(f"auto run: START (ceiling {args.auto_max_seconds:g} s)")
            # ACKNOWLEDGE, after the verb has been ACTED on and never before. A sender polling
            # `fuji_ack` can then distinguish "not handled yet" from "handled, and this is the
            # answer"; without it every reader races this loop and reads the previous state.
            if req is not None:
                with page_lock:
                    page["ack"] = int(page["ack"]) + 1

            # THE AUTO LOOP, one chunk per pass of this loop rather than its own thread: every
            # iteration is the SAME _predict -> _drip pair the buttons call, and running it here
            # is what lets a Stop / a sample change / a Reset land BETWEEN chunks with no extra
            # mutex. It advances the cursor by the executed steps, which is what makes it an
            # open-loop rehearsal of the recorded observation sequence (see the module docstring).
            if auto["until"] > 0.0 and not stop_flag.is_set():
                if time.time() >= auto["until"]:
                    auto["until"] = 0.0
                    _set_page(state="hold", msg="auto run finished — wall-clock ceiling")
                elif s0.num_clients == 0:
                    auto["until"] = 0.0
                    _set_page(state="hold",
                              msg=f"auto run STOPPED — the low level left :{args.s0_ws_port}")
                    _log("AUTO: the low level disconnected — run ended.")
                elif busy.acquire(blocking=False):
                    try:
                        note = _predict()
                        out = staged["out"]
                        if out is None:
                            auto["until"] = 0.0
                            _set_page(state="hold", msg=f"auto run STOPPED — {note}")
                            _log(f"AUTO: stopped, nothing staged — {note}")
                        else:
                            n = _drip(out)
                            staged["out"] = None
                            if not cursor.step(n):
                                auto["until"] = 0.0
                                _set_page(state="hold",
                                          msg="auto run finished — end of the dataset")
                            if stop_flag.is_set():
                                auto["until"] = 0.0
                    finally:
                        busy.release()

            now = time.monotonic()
            if now - last_status > 1.0:
                last_status = now
                if s0.num_clients != n_client_seen:
                    n_client_seen = s0.num_clients
                    _log(f"low level: {n_client_seen} client(s) on :{args.s0_ws_port}"
                         + ("" if n_client_seen else " — start the browser build"))
                with wire_lock:
                    frame = tuple(a.copy() for a in last_wire["frame"])
                with page_lock:
                    snap = dict(page)
                if snap["state"] != "hold" or ramp["target"] is not None:
                    _log(f"[{snap['state']}] {driver.wire.status(frame)} · {snap['msg']}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        _log("interrupted")
        return 0
    finally:
        shutdown.set()
        if jaws is not None:
            jaws.open_all("shutdown")
            jaws.stop()
        s0.stop()


class _Jaws:
    """The chunk's predicted gripper widths -> ``target_frac`` on each jaw's own bridge.

    ⚠️ **A SECOND CHANNEL, and the protocol is why.** ``:8770`` has no gripper field in EITHER
    wire family, so a jaw cannot be commanded on the wire this script publishes; it is reached
    over the gripper bridge's own WebSocket (``:8784`` / ``:8785``), exactly as the three deploy
    planners do it. Two consequences that follow from that and not from a choice made here:

      * it reaches **real CAN hardware even when the low level is a browser sim** — mjswan has no
        jaw, so a `--command-gripper` run against the browser moves a physical gripper and
        nothing else. That is legitimate (the widths ARE the policy's prediction) but it is the
        one part of this open-loop rehearsal that is not a simulation, hence off by default.
      * with no bridge listening, the sends are swallowed by ``_WSSubscriber``'s reconnect loop
        rather than raising — so "the jaws did nothing" never fails a Run, and the launch log
        says which URLs were assumed.

    ONE state machine over ONE control dict **per hand** (``HL.new_grip_ctl``), for the reason
    that constructor documents: ``build_grip_command``'s run-length state (``engaged`` / ``run``
    / ``held_frac``) lives in the dict it is handed, not in the closure, so a shared dict lets the
    LEFT jaw's confirm run engage the RIGHT jaw's overshoot. The constructor is also what keeps
    the units straight — the CLI flags are millimetres, the keys it writes are metres.
    """

    def __init__(self, args, roles) -> None:
        self.roles = tuple(roles)
        urls = {"left": args.gripper_ws, "right": args.gripper_right_ws}
        missing = [r for r in self.roles if r not in urls]
        if missing:
            raise SystemExit(
                f"--command-gripper: no jaw bridge URL for role(s) {missing}. This script knows "
                f"about {sorted(urls)}; a new gripper role needs its own flag rather than a "
                "silently unmapped hand.")
        self.calib = HL.GripperCalib()
        self._lock = threading.Lock()
        self._ctl = {r: HL.new_grip_ctl(overshoot_mm=float(args.grip_overshoot_mm),
                                        close_below_mm=float(args.grip_close_below_mm),
                                        confirm_steps=int(args.grip_confirm_steps))
                     for r in self.roles}
        self._cmd = {r: HL.build_grip_command(self.calib, self._ctl[r], self._lock)
                     for r in self.roles}
        self.subs = {r: HL._WSSubscriber(urls[r], f"grip_{r}") for r in self.roles}
        self.urls = {r: urls[r] for r in self.roles}
        self._ever = False
        self.last: dict[str, float] = dict.fromkeys(self.roles, float("nan"))

    def start(self) -> None:
        for sub in self.subs.values():
            sub.start()

    def stop(self) -> None:
        for sub in self.subs.values():
            sub.stop()

    def command(self, widths: dict) -> None:
        """One step's ``{role: width_m}`` -> one ``target_frac`` per hand.

        A non-finite or absent width is SKIPPED rather than sent as a zero: the chunk's gripper
        column is masked out for a role with no physical jaw, and 0.0 is a legal fully-closed
        target, so a fabricated value is invisible in the numbers.
        """
        for role in self.roles:
            w = widths.get(role)
            if w is None or not np.isfinite(w):
                continue
            frac, _squeezing = self._cmd[role][0](float(w))
            if not np.isfinite(frac):
                continue
            self.subs[role].send({"target_frac": float(frac)})
            self.last[role] = float(frac)
            self._ever = True

    def release(self) -> None:
        """Drop every hand's squeeze latch, so the next grasp has to earn its confirmation run."""
        for role in self.roles:
            self._cmd[role][1]()

    def open_all(self, why: str) -> None:
        """Command every jaw fully OPEN. Reset and shutdown only, never mid-chunk.

        Inert until something has actually been commanded: with the switch off the jaws were
        never driven and are left exactly where the bridge has them (``keyboard_gripper
        --initial-frac 1.0`` brings them up open), which is the same contract ``hl_serving``'s
        shutdown path keeps.
        """
        if not self._ever:
            return
        self.release()
        for role in self.roles:
            try:
                self.subs[role].send({"target_frac": 1.0})
            except Exception as exc:  # noqa: BLE001 - a jaw release must not mask the caller
                _log(f"{why}: could not open the {role} jaw ({exc!r})")
        self.last = dict.fromkeys(self.roles, 1.0)
        _log(f"{why}: commanded {len(self.roles)} jaw(s) OPEN")

    def note(self) -> str:
        return "jaws " + ", ".join(
            f"{r[0].upper()} {'--' if not np.isfinite(self.last[r]) else f'{self.last[r] * 100:.0f}%'}"
            for r in self.roles)


def _do_run(args, s0, staged, busy, stop_flag, drip, set_page, driver) -> str:
    """The Run button, factored out only to keep the control loop's branches one line each."""
    out = staged["out"]
    if out is None:
        return "nothing staged — press Predict first"
    if s0.num_clients == 0:
        return (f"the low level is NOT connected to :{args.s0_ws_port} — start "
                f"{FAMILIES[driver.family][1]} (and it must be a {driver.wire.low_level_mode} "
                "family preset, in ABSOLUTE mode)")
    if not busy.acquire(blocking=False):
        return "busy"
    try:
        stop_flag.clear()
        n = drip(out)
        staged["out"] = None
        return f"executed {n} step(s); re-committed on the last one"
    finally:
        busy.release()


# ---------------------------------------------------------------------------
# the two family drivers
# ---------------------------------------------------------------------------
class _OfflineWholeBodyWire(hl_wholebody.WholeBodyWire):
    """``WholeBodyWire`` whose ``publish`` also carries the page's telemetry keys.

    ONE overridden method, and a subclass rather than an ``extra=`` parameter on ``DenseWire``:
    the deploy planners' ``publish`` is reviewed as the last gate before the socket, and adding a
    keyword there would put a UI concern inside it. Everything safety-relevant — the width and
    finiteness gate, the by-name remap, the clutch, the idle frame — is inherited untouched.
    """

    def publish(self, s0, frame: tuple, f_cmd, extra=None) -> tuple[bool, str]:  # type: ignore[override]
        ok, reason = hl_dense.is_safe_dense_command(frame[0], frame[1])
        if not ok:
            return False, reason
        s0.send_dense(frame[0], frame[1], self.joint_names, f_cmd=f_cmd, extra=extra)
        return True, ""


class _DenseDriver:
    """``wholebody`` -> 29 joint targets. The ghost is the chunk; the wire is the clutched chunk."""

    family = "dense"

    def __init__(self, args, session, layout, anchor, algo: str) -> None:
        del algo
        self.args = args
        self.session = session
        self.layout = layout
        self.ready = hl_dense.ready_pose_joint_targets(
            anchor.dof, G1_29DOF_SDK_DOF_NAMES, WB_JOINT_NAMES)
        post = hl_dense.PostAnchorConfig(mode=args.post_anchor)
        if post.carrying and args.wb_clutch != "joint":
            raise SystemExit("--post-anchor command needs --wb-clutch joint")
        self.clutch = hl_dense.JointClutch(self.ready, mode=args.wb_clutch, post=post)
        # What the checkpoint's joint columns MEAN. Read from the checkpoint, never a flag.
        self.joint_target = str(getattr(session.source, "wb_joint_target", "absolute"))
        try:
            self.wire = _OfflineWholeBodyWire(self.clutch, self.ready, legs=args.legs,
                                              joint_target=self.joint_target)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        self.horizon = _horizon_of(session)
        self._joint_slot = layout.joint_state_slot
        if self._joint_slot is None:
            raise SystemExit(
                f"layout {layout.mode!r} carries no 29-joint STATE slot, so there is no sample "
                "posture to hold at. Pass --hold ready, or serve a `wholebody` checkpoint. (The "
                "wholebody_geom_* modes replace the joint state with a geometry on purpose — see "
                "docs/RUNBOOK_wb_geom_ablation.md; they are not served anywhere yet.)")
        # ⚠️ `wb_joint_target` DOES NOT BRANCH ANYTHING BELOW, and that is worth stating because
        # the wire's own refusal (--wb-clutch none + delta) reads as though it should.
        # `build_wholebody_result` already adds the observation's own absolute joints back for a
        # delta checkpoint, so `result.predicted_joints` is a POSTURE either way and the wire, the
        # ghost and the leg readout all consume it unbranched.

    def stage(self, result) -> dict:
        joints = np.asarray(result.predicted_joints, dtype=np.float64)
        plan = self.wire.plan(_WBChunk(joints))
        rad, name = hl_wholebody.leg_excursion_rad(joints)
        note = (f"legs {np.degrees(rad):.1f}° ({name})"
                + (" HELD" if self.wire.legs == "hold" else "")
                + f" · joint_deg {result.metrics.get('joint_deg', float('nan')):.2f} vs target")
        return {"plan": plan, "joints": joints, "result": result, "note": note}

    def ghost(self, out, step: int) -> dict | None:
        if out is None:
            return None
        published = out["plan"]["published"]
        i = int(step) % int(published.shape[0])
        # The PUBLISHED joints, not the model's raw absolute pose: with the joint clutch armed
        # step 0 then sits exactly on the hold pose, so "the ghost did something wrong" and "the
        # robot was told something wrong" are the same statement.
        return hl_offline.ghost_extras(published[i], WB_JOINT_NAMES)

    def fcmd(self, out, i: int):
        rows = [out["result"].predicted_wrench.get(r) for r in self.layout.force_roles]
        return _gate_rows([None if w is None else w[i] for w in rows], self.args.force_gate_n)

    def grip_widths(self, out, i: int) -> dict:
        """``{role: width_m}`` for step ``i`` — the wholebody result's per-role gripper mapping."""
        table = getattr(out["result"], "predicted_gripper_m", None) or {}
        got = {}
        for role in self.layout.gripper_roles:
            w = table.get(role)
            if w is not None and int(np.shape(w)[0]) > i:
                got[role] = float(np.asarray(w, dtype=np.float64).reshape(-1)[i])
        return got

    # --- the hold pose ---------------------------------------------------
    def hold_target(self, cursor):
        """The absolute 29-joint posture this sample was OBSERVED in, or ``None`` for ready.

        Read off the sample's STATE, not off ``action_chunk[0]``, and the difference only shows
        on a ``delta`` checkpoint: the state is absolute in BOTH parameterizations (a delta state
        relative to itself would be identically zero), while row 0 of a delta chunk is ~0 rad —
        i.e. the 29-joints-zero pose, which is a standing robot told to straighten every leg.
        On an ``absolute`` checkpoint the two agree to 0.000e+00 by the exporter's own check.
        """
        if self.args.hold != "sample":
            return None
        _ref, sample = self.session.source.get(cursor.index)
        state = sample["state"].detach().cpu().numpy().astype(np.float64).reshape(
            -1, self.layout.state_dim)
        q = state[-1, self._joint_slot]
        if q.size != len(WB_JOINT_NAMES) or not np.all(np.isfinite(q)):
            return None
        return q

    def hold_gap_note(self, target) -> str:
        gap = np.abs(np.asarray(target, np.float64) - self.ready)
        worst = int(np.argmax(gap))
        return f"worst {gap[worst]:.3f} rad at {WB_JOINT_NAMES[worst]} from the ready crouch"

    def ramp(self, current, target, max_step: float):
        # VELOCITY ZERO for the whole ramp, the same rule `DenseWire.measured` states: the robot
        # is being repositioned, not tracking a trajectory, and a reference velocity derived from
        # the heartbeat's own step size would tell the low level otherwise.
        q, done = hl_offline.ramp_step(current[0], target, max_step)
        return (q, np.zeros(q.size)), done

    def rebase(self, frame) -> None:
        self.clutch.reset(np.asarray(frame[0], dtype=np.float64))

    def reset(self, target) -> None:
        self.clutch.reset(self.ready if target is None
                          else np.asarray(target, dtype=np.float64))


class _WBChunk:
    """The one field ``WholeBodyWire.plan`` reads off a ``WholeBodyChunk``.

    A shim rather than the frozen dataclass with three empty dicts: the wire only ever touches
    ``.joints``, and an offline result has no gated wrench to put in the other fields — its
    force reaches the wire through :meth:`_DenseDriver.fcmd` instead.
    """

    __slots__ = ("joints",)

    def __init__(self, joints: np.ndarray) -> None:
        self.joints = joints


class _ThreePointDriver:
    """``both_hands_head[_dual]`` -> 3 keypoint poses, decoded against the READY POSE."""

    family = "3pt"

    def __init__(self, args, session, layout, anchor, algo: str) -> None:
        del algo
        self.args = args
        self.layout = layout
        self.roles = tuple(layout.roles)
        missing = [r for r in ("left", "right", "head") if r not in self.roles]
        if missing:
            raise SystemExit(
                f"a 3pt wire needs all three predicted roles; this layout is missing {missing}. "
                "A partial keypoint checkpoint belongs on scripts/hl_serving.py, which HOLDS the "
                "points it does not predict at the ready pose.")
        # THE DECODE ANCHOR. The same one hl_serving_mc.py uses, and for the same reason: the
        # chunk is anchor-relative, so what it is decoded against IS the physical claim. The
        # ready pose's own keypoints already sit on the TCP levers the *-unified-anneal presets
        # want (resolve_deploy_anchor cross-checks them against holosoma's baked
        # home_pos_pelvis_3pt_tcp to 0.1 mm), so no further lever is applied here.
        self.anchor_pos = np.stack([anchor.pos[r] for r in self.roles])
        self.anchor_rot6d = np.stack([rotmat_to_rot6d(anchor.rotmat[r]) for r in self.roles])
        order = [self.roles.index(r) for r in ("left", "right", "head")]
        self._order = np.asarray(order, dtype=int)
        ready_pos = np.concatenate([anchor.pos[r] for r in ("left", "right", "head")])
        ready_orn = np.concatenate([rotmat_to_quat_xyzw(anchor.rotmat[r])
                                    for r in ("left", "right", "head")])
        self.wire = hl_offline.ThreePointWire(
            ready_pos, ready_orn, post=hl_dense.PostAnchorConfig(mode=args.post_anchor))
        self.horizon = _horizon_of(session)

    def stage(self, result) -> dict:
        chunk = decode_chunk(np.asarray(result.predicted_actions, dtype=np.float64), self.layout,
                             self.anchor_pos, self.anchor_rot6d,
                             force_enabled=bool(self.args.enable_force_cmd),
                             gate_n=float(self.args.force_gate_n))
        by_role = {rc.role: rc for rc in chunk.roles}
        pos = np.concatenate([by_role[r].position for r in ("left", "right", "head")], axis=1)
        orn = np.concatenate([
            np.stack([rotmat_to_quat_xyzw(m) for m in rot6d_to_rotmat(by_role[r].rot6d)])
            for r in ("left", "right", "head")], axis=1)
        plan = self.wire.plan(pos, orn)
        ok, gate = self.wire.gate_report(self.wire.frame(plan, 0))
        travel = {r: float(np.linalg.norm(by_role[r].position - by_role[r].position[0], axis=1).max())
                  for r in ("left", "right", "head")}
        note = (gate + " · travel " +
                ", ".join(f"{r[0].upper()} {travel[r] * 1e3:.0f} mm" for r in travel))
        return {"plan": plan, "chunk": chunk, "result": result, "note": note, "gate_ok": ok}

    def ghost(self, out, step: int) -> dict | None:
        # No body for a keypoint layout — see this module's docstring. The page's own commanded
        # triads are what moves, and they move when the chunk RUNS.
        del out, step
        return None

    def fcmd(self, out, i: int):
        rows = [rc.wrench_cmd for rc in out["chunk"].roles if rc.wrench_cmd is not None]
        if not rows:
            return None
        return np.concatenate([r[i][0:3] for r in rows])

    def grip_widths(self, out, i: int) -> dict:
        """``{role: width_m}`` for step ``i``, straight off the decoded chunk.

        ``decode_chunk`` has already un-normalized these (``* GRIPPER_WIDTH_REF_M``) and filled
        them ONLY for ``layout.gripper_roles``, so a sensorless hand is absent here rather than
        present as a zero — which is the distinction ``_Jaws.command`` needs to skip it.
        """
        got = {}
        for rc in out["chunk"].roles:
            if rc.gripper_m is not None and int(rc.gripper_m.shape[0]) > i:
                got[rc.role] = float(rc.gripper_m[i])
        return got

    def hold_target(self, cursor):
        # The 3pt chunk is decoded against the ready pose, so its own step 0 IS the ready pose:
        # 'sample' and 'ready' coincide and there is nothing to ramp to.
        del cursor
        return None

    def hold_gap_note(self, target) -> str:
        del target
        return "n/a"

    def ramp(self, current, target, max_step: float):
        del target, max_step
        return current, True

    def rebase(self, frame) -> None:
        del frame

    def reset(self, target) -> None:
        del target
        self.wire.clutch.reset()


def _horizon_of(session) -> int:
    """The chunk length this checkpoint predicts, off its ACTIVE algorithm block.

    The session already resolved which of the three blocks is live (that is what its ``config``
    is), so reading the field off it is what keeps this from re-deriving the algo->block map that
    ``hl_online._algo_config_key`` has a documented collapse bug about.
    """
    from fuji.s1_policy.data.layout import ACTION_HORIZON

    h = getattr(session.config, "action_horizon", None)
    return int(h) if isinstance(h, int) and not isinstance(h, bool) and h > 0 else int(ACTION_HORIZON)


def _gate_rows(rows, gate_n: float):
    """Per-hand predicted wrench rows -> the ``(3*n,)`` sensor-frame F^cmd, or None.

    Same two-level contract and the same sign flip in the same one place as the deploy planners:
    the model predicts what the Bota READ, the low level closes ``e = F^cmd + F_meas -> 0``, so
    the command reproducing the demonstrated press is ``-F_meas``. Gated on MAGNITUDE, which the
    flip preserves. BOTH hands or NEITHER, because a (6,) f_cmd is read as ``[L, R]``.
    """
    from fuji.utils.force_frame import cmd_from_meas_force

    out, any_live = [], False
    for row in rows:
        if row is None or not np.all(np.isfinite(np.asarray(row)[0:3])):
            out.append(np.zeros(3))
            continue
        f3 = np.asarray(row[0:3], dtype=np.float64)
        if float(np.linalg.norm(f3)) <= gate_n:
            out.append(np.zeros(3))
        else:
            out.append(cmd_from_meas_force(f3))
            any_live = True
    return np.concatenate(out) if any_live else None


if __name__ == "__main__":
    raise SystemExit(main())
