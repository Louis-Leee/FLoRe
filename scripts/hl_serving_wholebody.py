#!/usr/bin/env python3
"""ONLINE WHOLE-BODY PLANNER: ``diffusion_umi_wholebody`` straight onto the ``dense`` low level.

The deploy half of the 29-joint preset. Same predict -> preview -> execute UX as
``scripts/hl_serving_mc.py``, the same six sensor bridges, the same force gate, the same joint
clutch and the same ``:8770`` wire — with **one stage removed**::

  hl_serving_mc.py   2 D405 + 2 Bota + 2 jaws -> S1 (both_hands_head_dual, 41-D)
                       |  3 anchor-relative EE chunks
                     decode against the G1 READY POSE
                       |  3 absolute pose series
                     Motion_Completion   (a SECOND model, a SECOND interpreter, ~80 ms/chunk)
                       |  qpos(29)
                     JointClutch -> DenseWire -> :8770  {"mode":"dense"}

  THIS SCRIPT       2 D405 + 2 Bota + 2 jaws -> S1 (wholebody, 43-D)
                       |  qpos(29) + 2 grippers + 2 wrenches, ABSOLUTE
                     JointClutch -> DenseWire -> :8770  {"mode":"dense"}

The completion did not disappear — it ran ONCE, offline, over the training zarr
(``motion_completion/mc_to_hl_wholebody.py``), so at deploy there is no ``g1_mosyn`` conda env to
locate, no backend subprocess, no 30-60 s JIT warm-up and no ``--mc-foot-ik`` trade-off. The low
level is the SAME one either way (``MODE=dense-unified-anneal``, W&B ``7vxmnwo6``).

THE THREE THINGS THAT ARE GENUINELY DIFFERENT
=============================================
1. **The chunk is ABSOLUTE and there is no anchor.** ``layout.pose_dims()`` is empty: nothing was
   relativized at training time, so nothing is un-relativized here. The planner never seeds or
   commits a pose anchor — :class:`hl_wholebody.WholeBodyPlanner` carries a JOINT ring instead.

2. **The absolute joints live in the DATASET's posture space, not the robot's.**
   ``attrs.wb_cond_mode="home"`` re-based every take onto the G1's 29-joints-zero pose; the robot
   holds a 38 deg crouch after **A**. Measured over 30 episode starts of
   ``session_2026.08.30_1411_v3_dual_clean32_wb``, the two are **0.514 rad apart at the knee**
   (hip pitch 0.35-0.39, shoulder roll 0.32). :class:`hl_dense.JointClutch` is the whole answer
   and it is the DEFAULT: ``base + (q[i] - q[0])`` makes step 0 a bit-exact no-op, so the robot
   keeps its own crouch and follows the predicted joint MOTION. ``--wb-clutch none`` publishes the
   absolute pose into a 0.20 rad pre-arm gate and says so, loudly.

   The PROPRIOCEPTION has the same gap and a clutch cannot fix an input, so the state is the last
   COMMANDED target in the model's own space (``--state-source command``, the repo-wide
   ``obs_mode=last_target`` rule). The one frame with nothing commanded yet is seeded from the
   training data's own episode starts — see ``hl_wholebody.state_seed_from_zarr``.

3. **The LEGS are on the wire.** This is the preset, not a defect: a keypoint policy commands 3
   points and lets the low level pick a posture, this one commands the hips, knees and ankles too.
   Every Predict prints the chunk's measured leg excursion, and ``--legs hold`` pins the 12 leg
   joints to the clutch base for a first bring-up where the arms are the thing under test.

WHAT IS DELIBERATELY NOT HERE (first bring-up)
==============================================
The take recorder, recorded-episode replay, reach compensation, temporal ensembling, the
executed-trace QC sheet, and the 5pt wire family. A wholebody policy has no keypoint front-end at
all, so there is no ``FivePointWire`` to publish for: the chunk IS joints, and turning it into 5
keypoints would need an FK plus a decision about the two feet nobody predicted. Absent rather
than half-ported.

BRING-UP, in order — every one is checked at startup and named if missing::

    T1  bash 2_ft_bridge.sh --pipeline              both Botas, 6-DoF, :8771
    T2  camera L :8773   and   camera R :8777       both D405s (this model reads TWO views)
    T3  LEFT=1 RIGHT=1 KEYBOARD=0 bash 5_dual_gripper_keyboard.sh    LEFT :8784, RIGHT :8785
    T4  MODE=dense-unified-anneal HOLOSOMA_VIVE_RELATIVE=0 RECORD=1 \
          bash 3_run_policy_vive_force.sh           the low level (ABSOLUTE, RecordTap on)
    T5  ./scripts/run_hl_serving_wholebody.sh --checkpoint ckpt_0831_wb

T4's ``MODE`` **must** be a ``dense``/``wbt``-family preset and nothing cross-checks it: the
RecordTap frame carries no preset name. A mismatch is not dangerous — the low level's width guard
drops every frame of the wrong shape and the robot holds its ready pose — but it means "nothing
moves" reads as "T4 is not a dense preset" before it reads as anything else.

⚠️ SAFETY. This commands a physical G1 whole body and the legs are on the wire. First runs
gantry-suspended, operator on the hardware E-stop, force wire OFF, and ``--legs hold`` until the
arms are trusted. Per-joint motion is bounded by the low level's own
``HOLOSOMA_VIVE_MAX_JOINT_STEP`` (0.05 rad/tick); with the default joint clutch the chunk starts
from the pose the robot is already holding rather than jumping to the model's.
"""

from __future__ import annotations

import argparse
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

# The sensor plumbing, the gripper calibration and the freshness rules come from the left_only
# planner rather than being copied: they are model-agnostic and each has its own review history.
# Importing the module is side-effect-free (it binds nothing and parses no args until main()).
import hl_serving as HL  # noqa: E402

from fuji.client.s0_driver import S0Driver  # noqa: E402
from fuji.s1_policy.data.layout import ACTION_DT, ACTION_HZ, WB_JOINT_NAMES  # noqa: E402
from fuji.s1_policy.inference import ready_pose as RP  # noqa: E402
from fuji.s1_policy.serving import hl_dense, hl_wholebody  # noqa: E402
from fuji.s1_policy.serving.hl_online import (  # noqa: E402
    ACTION_HORIZON,
    OBS_MAX_AGE_S,
    exec_step_choices,
    load_hl_model,
    snap_exec_steps,
)
from fuji.s1_policy.serving.live_robot import (  # noqa: E402
    G1_29DOF_SDK_DOF_NAMES,
    JAW_JOINT_OPEN_M,
    build_sdk_to_urdf_index,
    jaw_joint_from_width_m,
    parse_record_frame,
    remap_dof_to_urdf,
)
from fuji.utils.force_frame import cmd_from_meas_force  # noqa: E402

#: Translucent violet — the same ghost colour ``hl_serving`` and ``hl_serving_mc`` use, so
#: "violet = what is about to happen" means one thing across all three planners.
_GHOST_RGBA = HL._GHOST_RGBA

#: How often the idle heartbeat re-publishes the held wire. The low level's stale gate is 0.25 s
#: and ``S0Driver`` stops re-broadcasting after 0.5 s of silence, so this only has to beat 0.25 s.
IDLE_HB_HZ = 60.0

#: Execute-N-of-S choices offered in the page, DERIVED from this process's horizon rather than
#: hardcoded — see :func:`hl_online.exec_step_choices`. ``ACTION_HORIZON`` is resolved from
#: ``FUJI_ACTION_HORIZON`` at import time and ``run_hl_serving_wholebody.sh`` exports it from the
#: checkpoint's own ``config.json``, so an H=100 checkpoint offers 100 and an H=200 one offers
#: 100 and 200 with no flag. The pin is what makes deriving it here sound: a checkpoint whose
#: horizon disagrees with the constant is refused outright by ``validate_model_layout``, so this
#: tuple can never advertise a window the model will not fill.
#: MODULE scope for the reason ``hl_serving._EXEC_CHOICES`` documents: main() (the snap) and
#: _run_viser (the dropdown) both read it, and as a main() local the dropdown raised NameError
#: inside _run_viser's broad except, surfacing only as a page that never loaded.
_EXEC_CHOICES = exec_step_choices(ACTION_HORIZON)


def _log(msg: str) -> None:
    print(f"[hl-wb] {msg}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="a wholebody HL-v2 checkpoint dir (hl_data.mode == 'wholebody')")
    p.add_argument("--algo", default="diffusion", choices=["diffusion", "flow", "act"],
                   help="which algorithm block the checkpoint carries (default: %(default)s). "
                        "'wholebody' is a LAYOUT, not a fourth algorithm — it is checked "
                        "separately, by name. run_hl_serving_wholebody.sh derives this from the "
                        "checkpoint, so it normally never has to be typed.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-steps", type=int, default=None,
                   help="sampler steps (default: the checkpoint's own)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8091)

    g = p.add_argument_group("sensor bridges (the dual rig)")
    g.add_argument("--force-ws", default="ws://127.0.0.1:8771",
                   help="Bota bridge, BOTH hands, run with --pipeline (default: %(default)s)")
    g.add_argument("--camera-ws", default="ws://127.0.0.1:8773", help="camera L (camera0_rgb)")
    g.add_argument("--camera-right-ws", default="ws://127.0.0.1:8777",
                   help="camera R (camera1_rgb). REQUIRED by a 2-camera checkpoint; pass an "
                        "empty string to declare a 1-camera rig, which then refuses a 2-camera "
                        "model by name instead of feeding it a grey view.")
    g.add_argument("--gripper-ws", default="ws://127.0.0.1:8784", help="LEFT jaw bridge")
    g.add_argument("--gripper-right-ws", default="ws://127.0.0.1:8785", help="RIGHT jaw bridge")
    g.add_argument("--record-ws", default="ws://127.0.0.1:8779",
                   help="low-level RecordTap (needs RECORD=1). Drives the live robot mesh and the "
                        "MEASURED joint clutch; without it the clutch runs open-loop from the "
                        "ready pose and --state-source measured has no input at all.")
    g.add_argument("--s0-ws-host", default="127.0.0.1")
    g.add_argument("--s0-ws-port", type=int, default=8770)
    g.add_argument("--allow-3dof-force", action="store_true",
                   help="opt into a 3-DoF F/T bring-up (else a missing --pipeline is fatal)")

    c = p.add_argument_group("cadence")
    c.add_argument("--obs-hz", type=float, default=ACTION_HZ)
    c.add_argument("--step-hz", type=float, default=ACTION_HZ,
                   help="chunk playback rate; MUST be the training grid's 30 Hz")
    c.add_argument("--drip-hz", type=float, default=IDLE_HB_HZ)
    c.add_argument("--exec-steps", type=int, default=16,
                   help=f"how many of the predicted steps to execute per chunk (live in the "
                        f"page). SNAPPED to this checkpoint's offered choices "
                        f"{list(_EXEC_CHOICES)}, which follow the horizon "
                        f"({ACTION_HORIZON} steps = {ACTION_HORIZON * ACTION_DT:.2f} s)")
    c.add_argument("--obs-max-age", type=float, default=0.0,
                   help="staleness limit (s) for the Predict refusal; 0 keeps the default and "
                        "disables only the refusal")
    c.add_argument("--auto-max-seconds", type=float, default=20.0,
                   help="wall-clock ceiling for the page's ▶▶ sequential run (live on the page). "
                        "A policy has no end of its own, so this is the bound that stops the body "
                        "even if the operator looks away.")

    d = p.add_argument_group("the wire")
    d.add_argument("--wb-clutch", default="joint", choices=list(hl_dense.CLUTCH_MODES),
                   help="'joint' (default) publishes the chunk's CHANGE from the robot's own "
                        "joints, so step 0 is a no-op and the pre-arm gate closes. 'none' "
                        "publishes the model's ABSOLUTE angles, which sit in the dataset's "
                        "cond_mode=home posture space — measured 0.514 rad (knee) from the ready "
                        "crouch, against a 0.20 rad gate.")
    d.add_argument("--legs", default="predict", choices=list(hl_wholebody.LEG_MODES),
                   help="what reaches the 12 leg joints. 'predict' (default) publishes all 29 — "
                        "the faithful wire. 'hold' pins the legs to the clutch base so only the "
                        "17 waist+arm joints follow the chunk; RECOMMENDED for a first "
                        "gantry-suspended run, and it needs --wb-clutch joint.")
    d.add_argument("--post-anchor", default="off", choices=list(hl_dense.POST_ANCHOR_MODES),
                   help="where the NEXT chunk's WIRE continues from. 'off' (default) re-bases the "
                        "clutch on measured joints every Predict. 'command' carries the LAST "
                        "EXECUTED step forward (PLAI), leak-bounded against those same measured "
                        "joints, so the accumulated command is not thrown away at the boundary.")
    d.add_argument("--post-anchor-eps-rad", type=float, default=0.15, metavar="RAD",
                   help="[--post-anchor command] per-joint leak bound (default 0.15, ceiling "
                        f"{hl_dense.MAX_POST_EPS_RAD:g}): how far the carried command may lead the "
                        "measured joints before it is pulled back. 0 collapses the mode to 'off'.")
    d.add_argument("--post-anchor-contact-n", type=float, default=1.5, metavar="N",
                   help="[--post-anchor command] measured |F| at or above which the carried "
                        "anchor DECAYS toward the robot instead of advancing")
    d.add_argument("--no-zero-arms", dest="zero_arms", action="store_false",
                   help="build the idle wire from the raw crouch default_qpos instead of the "
                        "arms-zeroed ready pose. Only correct against HOLOSOMA_INIT_READY_POSE=0.")
    d.set_defaults(zero_arms=True)
    d.add_argument("--yam-urdf", default=os.path.join(
        _FUJI_ROOT, "third_party", "UMI_collection", "robot_assets", "g1_yam",
        "g1_29dof_yam.urdf"), help="URDF for the drawn robot + the violet ghost")

    s = p.add_argument_group("proprioception (the 29-joint state)")
    s.add_argument("--state-source", default="command",
                   choices=list(hl_wholebody.STATE_SOURCES),
                   help="'command' (default) feeds the last COMMANDED joints in the MODEL's own "
                        "posture space — in-distribution by construction, because the exporter "
                        "proves action[:,0] == state exactly. 'measured' feeds RecordTap's "
                        "encoders, which is 0.514 rad (knee) OUTSIDE that space forever.")
    s.add_argument("--state-seed-zarr", default=None,
                   help="where to read the FIRST Predict's proprio seed from (default: the "
                        "checkpoint's own training zarr, relocated by basename). The seed is the "
                        "mean of that dataset's episode-start wb_state_joints; absent, the "
                        "cond_mode=home anchor's 29 zeros are used and the log says so.")
    s.add_argument("--no-state-seed", dest="state_seed", action="store_false",
                   help="skip the zarr read and seed at the cond_mode=home anchor (29 zeros)")
    s.set_defaults(state_seed=True)

    f = p.add_argument_group("force + gripper")
    # Both default ON. The page's switches are still the live control — this only says what they
    # READ at startup. Force still cannot reach the robot unless the low level was given
    # FORCE_CMD_WIRE=1, and it is hard-zeroed below --force-gate-n either way.
    f.add_argument("--enable-force-cmd", dest="enable_force_cmd", action="store_true",
                   default=True,
                   help="start with the page's force switch ON. THIS IS THE DEFAULT. The low level "
                        "ALSO needs FORCE_CMD_WIRE=1 — neither side alone commands force.")
    f.add_argument("--no-force-cmd", dest="enable_force_cmd", action="store_false",
                   help="start with the force switch OFF (pure position until the page turns it on)")
    f.add_argument("--force-gate-n", type=float, default=1.5,
                   help="below this |F| the command is hard-zeroed (still pure position)")
    f.add_argument("--command-gripper", dest="command_gripper", action="store_true", default=True,
                   help="autonomously drive BOTH jaws from the predicted widths. THE DEFAULT.")
    f.add_argument("--no-command-gripper", dest="command_gripper", action="store_false",
                   help="leave both jaws READ-ONLY — they start open and never close")
    f.add_argument("--grip-overshoot-mm", type=float, default=10.0)
    f.add_argument("--grip-close-below-mm", type=float, default=70.0)
    f.add_argument("--grip-confirm-steps", type=int, default=3)
    return p


# ---------------------------------------------------------------------------
# bring-up
# ---------------------------------------------------------------------------
def _camera_urls(args) -> list[str]:
    urls = [args.camera_ws]
    if str(args.camera_right_ws).strip():
        urls.append(args.camera_right_ws)
    return urls


def _self_check(subs: dict, cams: list[str], loaded, *, require_6dof: bool,
                consumes_force: bool) -> dict:
    """Probe every bridge this deploy reads, and name the missing one.

    Deliberately NOT ``hl_serving._self_check``: that one knows about one camera and one jaw, and
    a dual rig that forgot camera R would pass it and then feed the second encoder group a grey
    frame for the whole session. Same probe as ``hl_serving_mc._self_check``, which it is the
    wholebody twin of.
    """
    def _wait(sub, need_key=None, tries=40):
        for _ in range(tries):
            msg, _wall, _n = sub.latest()
            if msg is not None and (need_key is None or msg.get(need_key) is not None):
                return msg
            time.sleep(0.1)
        return None

    report: dict = {}
    fmsg = _wait(subs["force"])
    if fmsg is None:
        if consumes_force:
            raise SystemExit("self-check: no F/T frame on the Bota bridge "
                             "(is `bash 2_ft_bridge.sh --pipeline` running?)")
        report["force"] = "ABSENT — this checkpoint reads no wrench, so it is advisory"
    else:
        pipes = [r for r in loaded.layout.force_roles if fmsg.get(f"{r}_pipe") is not None]
        if consumes_force and require_6dof and len(pipes) != len(loaded.layout.force_roles):
            raise SystemExit(
                f"self-check: the F/T frame carries 6-DoF pipes for {pipes or 'no'} hand(s) but "
                f"this checkpoint reads {list(loaded.layout.force_roles)}. Its force_history was "
                "trained on 6-DoF gravity-only wrenches — start the bridge with --pipeline, or "
                "pass --allow-3dof-force for a bring-up (the model then sees OOD force).")
        report["force"] = f"ok ({len(pipes)}/{len(loaded.layout.force_roles)} hands, 6-DoF)"
    for i, url in enumerate(cams):
        if _wait(subs[f"camera{i}"], need_key="jpeg_b64") is None:
            raise SystemExit(
                f"self-check: no camera frame for {loaded.image_keys[i]} on {url}. This "
                f"checkpoint reads {loaded.num_cameras} view(s); a missing one is fed to its "
                "encoder group as a grey image and nothing downstream would say so.")
        report[f"camera{i} ({loaded.image_keys[i]})"] = f"ok ({url})"
    for role in loaded.layout.gripper_roles:
        gmsg = _wait(subs[f"grip_{role}"], need_key="actual")
        report[f"gripper {role}"] = (
            "ok" if gmsg is not None else
            "NO FRAME — the jaw width reads as UNMEASURED (state_valid=False), which is what the "
            "training converter does for a dead encoder; the grasp channel is degraded")
    # PARSED, not merely received: `parse_record_frame` is what the clutch and the live mesh
    # consume, and it rejects a frame whose `type` is not "record" or whose `dof_pos` is the wrong
    # length. A weaker "did any JSON arrive" check would report ok for a tap the clutch cannot use.
    rmsg = _wait(subs["record"], need_key="dof_pos", tries=20)
    rframe = None
    try:
        rframe = parse_record_frame(rmsg)
    except Exception:  # noqa: BLE001 - a malformed tap is a degraded mode, not a failure
        rframe = None
    if rframe is not None:
        report["RecordTap"] = "ok — live robot mesh + MEASURED joint clutch"
    elif rmsg is not None:
        report["RecordTap"] = (
            "PRESENT but UNUSABLE (not a 29-DOF `record` frame) — treated as absent: the mesh "
            "falls back to the ready pose and the joint clutch runs OPEN-LOOP")
    else:
        report["RecordTap"] = (
            "ABSENT — relaunch the low level with RECORD=1. The mesh holds the ready pose and the "
            "joint clutch runs OPEN-LOOP from it, so a tracking shortfall accumulates across "
            "chunks.")
    return report


def main(argv=None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    cams = _camera_urls(args)
    HL._limit_cpu_thread_pools(args.device)

    # --- 1. the model ---------------------------------------------------------------
    _log(f"loading {args.algo} checkpoint {args.checkpoint} on {args.device} …")
    loaded = load_hl_model(args.checkpoint, args.algo, device=args.device,
                           num_steps=args.num_steps, max_cameras=len(cams))
    if not loaded.layout.is_wholebody:
        raise SystemExit(
            f"checkpoint layout is {loaded.layout.mode!r}, not a wholebody one. This planner "
            "publishes the policy's OWN 29 joint angles; a keypoint checkpoint predicts poses. "
            "Serve it with scripts/hl_serving.py (3 points) or scripts/hl_serving_mc.py "
            "(3 trackers -> motion completion -> a whole-body low level).")
    if len(cams) != loaded.num_cameras:
        # load_hl_model already refuses "too few"; this catches the other direction, where a
        # bridge is running that the model was not trained on and would silently go unread.
        raise SystemExit(
            f"{len(cams)} camera bridge(s) configured but the checkpoint reads "
            f"{loaded.num_cameras} ({list(loaded.image_keys)}). Pass --camera-right-ws '' for a "
            "1-camera model, or the second URL for a 2-camera one.")
    consumes_force = bool(loaded.layout.force_roles)
    _log(f"ready: algo={loaded.algorithm} mode={loaded.layout.mode} "
         f"action_dim={loaded.layout.action_dim} state_dim={loaded.layout.state_dim} "
         f"To={loaded.obs_history_len} FH={loaded.force_hist_len} "
         f"cameras={list(loaded.image_keys)} force_roles={loaded.layout.force_roles} "
         f"gripper_roles={loaded.layout.gripper_roles}")
    _log(f"force command: {'ENABLED' if args.enable_force_cmd else 'off'} at launch "
         f"(gate {args.force_gate_n:g} N — below that |F^cmd| is hard-zeroed and the motion is "
         f"pure position). The low level must ALSO have FORCE_CMD_WIRE=1. Toggle live on the "
         f"page; --no-force-cmd starts it off.")
    _log(f"both jaws: {'COMMANDED' if args.command_gripper else 'READ-ONLY'} at launch "
         f"(they come up OPEN either way)")
    if args.enable_force_cmd and not consumes_force:
        raise SystemExit(
            "force command is enabled but this checkpoint predicts no force for any role, so "
            "there would be nothing to put on the wire. Pass --no-force-cmd, or serve a "
            "checkpoint whose active algorithm block declares force.")
    if abs(args.step_hz - 1.0 / ACTION_DT) > 0.5:
        raise SystemExit(
            f"--step-hz {args.step_hz} is not the training grid's {1.0 / ACTION_DT:.0f} Hz. The "
            "joints are published one per step, so any other rate plays the whole body back at "
            "the wrong speed.")
    # SNAP --exec-steps onto what the page can actually show. viser renders an off-list
    # `initial_value` as a BLANK selector whose `.value` is still that number, so an unsnapped
    # flag gives a dropdown showing nothing while the drip executes it — see
    # `hl_online.snap_exec_steps`. Done here in main() so the log line lands before the page.
    _snapped = snap_exec_steps(args.exec_steps, _EXEC_CHOICES)
    if _snapped != int(args.exec_steps):
        _log(f"--exec-steps={int(args.exec_steps)} is not one of {list(_EXEC_CHOICES)}; "
             f"snapped to {_snapped}.")
    args.exec_steps = _snapped
    _log(f"chunk: {ACTION_HORIZON} steps @ {1.0 / ACTION_DT:.0f} Hz "
         f"= {ACTION_HORIZON * ACTION_DT:.2f} s; executing {args.exec_steps} of them per chunk "
         f"({args.exec_steps * ACTION_DT:.2f} s), page choices {list(_EXEC_CHOICES)}")

    # --- 2. the ready pose: the clutch base + the idle wire -------------------------
    # NOT a decode anchor — there is nothing to decode. This is the pose the low level HOLDS after
    # `A`, and both the pre-arm gate and the clutch are defined against it.
    anchor, why = RP.resolve_deploy_anchor(("left", "right", "head"),
                                           zero_arms=args.zero_arms)
    if anchor is None or anchor.dof is None:
        raise SystemExit(
            f"cannot resolve the G1 ready pose, which is the clutch base and the idle wire here: "
            f"{why}")
    _log(f"ready pose: {anchor.describe()}")
    for note in anchor.notes:
        _log(f"  ⚠️  {note}")
    if not args.zero_arms:
        _log("⚠️  --no-zero-arms: only correct against HOLOSOMA_INIT_READY_POSE=0.")

    # --- 3. the proprio seed (the ONE frame with nothing commanded yet) -------------
    seed_zarr = None
    if args.state_seed:
        seed_zarr = args.state_seed_zarr or HL._checkpoint_zarr(args.checkpoint)
    seed_joints, seed_note = hl_wholebody.state_seed_from_zarr(seed_zarr)
    _log(f"proprio: --state-source {args.state_source} · {seed_note}")
    gap = float(np.abs(seed_joints - np.asarray(anchor.dof, dtype=np.float64)).max())
    worst = int(np.argmax(np.abs(seed_joints - np.asarray(anchor.dof, dtype=np.float64))))
    _log(f"  the model's posture space vs the ROBOT's ready pose: worst {gap:.3f} rad at "
         f"{WB_JOINT_NAMES[worst]}. The joint clutch publishes base + (q[i] - q[0]), so this gap "
         f"never reaches the wire — but it IS why --state-source measured is not the default.")
    if args.state_source == "measured":
        _log("⚠️  --state-source measured: RecordTap's encoders are fed to the model as the "
             f"29-joint state. They sit {gap:.3f} rad from the posture space the state quantiles "
             "were fitted on, so every frame of every chunk is an out-of-distribution "
             "observation. Use it to measure that, not to run a task.")

    # --- 4. the wire ----------------------------------------------------------------
    post = hl_dense.PostAnchorConfig(
        mode=args.post_anchor,
        eps_joint_rad=float(args.post_anchor_eps_rad),
        contact_n=float(args.post_anchor_contact_n),
    )
    _log(f"post-completion anchor: {post.describe()}")
    ready_joints = hl_dense.ready_pose_joint_targets(
        anchor.dof, G1_29DOF_SDK_DOF_NAMES, WB_JOINT_NAMES)
    clutch = hl_dense.JointClutch(ready_joints, mode=args.wb_clutch, post=post)
    if post.carrying and args.wb_clutch != "joint":
        raise SystemExit(
            "--post-anchor command needs --wb-clutch joint: the carried command IS the clutch "
            "base, and with --wb-clutch none nothing is published relative to it, so the mode "
            "would silently do nothing.")
    # What the checkpoint's 29 joint columns MEAN. Read from the checkpoint, never a flag: an
    # absolute chunk and a delta one have the same width and both look plausible.
    joint_target = str(loaded.wb_joint_target)
    if joint_target == "delta":
        _log("wb_joint_target='delta': this checkpoint predicts joint MOTION (q[k] - q[0], row 0 "
             "is ~0 rad), not a posture. The joint clutch publishes base + (q[i] - q[0]), which "
             "is offset-invariant, so THE WIRE IS IDENTICAL to an absolute checkpoint's — what "
             "changes is the proprio latch, which integrates instead of replacing, and that "
             "--wb-clutch none is refused.")
    else:
        _log(f"wb_joint_target={joint_target!r}: the 29 columns are absolute radians in the "
             "export's own posture space.")
    try:
        wire = hl_wholebody.WholeBodyWire(clutch, ready_joints, legs=args.legs,
                                          joint_target=joint_target)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.wb_clutch == "none":
        _log(f"⚠️  --wb-clutch none: the model's ABSOLUTE joints go on the wire. Measured "
             f"{gap:.3f} rad from the ready pose at {WB_JOINT_NAMES[worst]} against a 0.20 rad "
             "pre-arm gate — raise HOLOSOMA_VIVE_ARM_TOL_RAD or expect a refusal, and expect the "
             "body to take that offset up on the first executed step.")
    if args.legs == "hold":
        _log("--legs hold: the 12 leg joints are pinned to the clutch base for the whole chunk. "
             "The 17 waist + arm joints follow the prediction. This is NOT what the policy asked "
             "for — a task needing a crouch will not get one.")
    _log(f"wire: mode={wire.mode!r} · the low level MUST be a {wire.low_level_mode}-family "
         f"MODE · its pre-arm gate is {wire.gate_note}")

    # --- 5. the planner --------------------------------------------------------------
    planner = hl_wholebody.WholeBodyPlanner(
        loaded, obs_hz=args.obs_hz, state_source=args.state_source, seed_joints=seed_joints,
        joint_target=joint_target)
    planner.obs_max_age_s = (OBS_MAX_AGE_S if not args.obs_max_age or float(args.obs_max_age) <= 0
                             else float(args.obs_max_age))
    planner.grip_max_age_s = max(planner.obs_max_age_s, HL.GRIP_MAX_AGE_S)

    # --- 6. the bridges --------------------------------------------------------------
    subs = {
        "force": HL._WSSubscriber(args.force_ws, "force"),
        "record": HL._WSSubscriber(args.record_ws, "record"),
    }
    for i, url in enumerate(cams):
        subs[f"camera{i}"] = HL._WSSubscriber(url, f"camera{i}")
    grip_urls = {"left": args.gripper_ws, "right": args.gripper_right_ws}
    for role in loaded.layout.gripper_roles:
        subs[f"grip_{role}"] = HL._WSSubscriber(grip_urls[role], f"grip_{role}")
    for sub in subs.values():
        sub.start()
    report = _self_check(subs, cams, loaded, require_6dof=not args.allow_3dof_force,
                         consumes_force=consumes_force)
    for key in sorted(report):
        _log(f"  {key}: {report[key]}")

    s0 = S0Driver(host=args.s0_ws_host, port=args.s0_ws_port)
    s0.start()
    try:
        return _run_viser(args, loaded, planner, wire, subs, s0, anchor, consumes_force, report,
                          seed_note=seed_note)
    finally:
        s0.stop()
        for sub in subs.values():
            sub.stop()


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------
def _run_viser(args, loaded, planner, wire, subs, s0, anchor, consumes_force, report,
               *, seed_note: str = "") -> int:
    import viser
    from viser.extras import ViserUrdf

    grip_roles = tuple(loaded.layout.gripper_roles)
    force_roles = tuple(loaded.layout.force_roles)

    stop = threading.Event()
    dripping = threading.Event()
    busy = threading.Lock()
    grip_enabled = threading.Event()
    if args.command_gripper:
        grip_enabled.set()

    wire_lock = threading.Lock()
    # The published wire, seeded at the READY POSE so the very first heartbeat gives the low
    # level's joint-space pre-arm gate something to close against.
    last_wire = {"frame": wire.idle(), "fcmd": None}
    force_lock = threading.Lock()
    force_ctl = {"enabled": bool(args.enable_force_cmd), "gate_n": float(args.force_gate_n)}
    # Snapped again (main() already did) so THIS function owns the invariant "the dropdown's
    # initial_value is one of its options" no matter who calls it.
    exec_ctl = {"n": snap_exec_steps(args.exec_steps, _EXEC_CHOICES)}
    # SEQUENTIAL ROLLOUT state ("▶▶ Run to completion"), the same shape as the other two
    # planners': `gen` is bumped by Stop so an in-flight loop retires itself instead of being
    # killed mid-chunk, and `max_s` bounds it in WALL-CLOCK seconds because a policy — unlike a
    # recorded episode — has no end of its own to reach.
    auto_lock = threading.Lock()
    auto_ctl = {"running": False, "gen": 0, "max_s": float(args.auto_max_seconds)}
    auto_running = threading.Event()
    staged: dict = {"out": None}
    seed_ctl = {"n": 0}
    stats = {"n": 0, "ms": 0.0, "legs": 0.0, "leg_name": "", "err": ""}
    calib = HL.GripperCalib()
    grip_lock = threading.Lock()
    # ONE state machine PER HAND, built ONCE, over ITS OWN control dict. `build_grip_command`'s
    # state lives in the dict it is handed, not the closure, so a shared one would let the LEFT
    # jaw's confirm counter engage the RIGHT jaw's overshoot. `HL.new_grip_ctl` is the constructor
    # next to the consumer — the keys/units are its business, not this call site's (a hand-written
    # dict shipped `overshoot_mm` in mm against a reader wanting `overshoot_m` in metres, and every
    # Run died with KeyError on the first jaw command).
    grip_ctl = {role: HL.new_grip_ctl(overshoot_mm=args.grip_overshoot_mm,
                                      close_below_mm=args.grip_close_below_mm,
                                      confirm_steps=args.grip_confirm_steps)
                for role in grip_roles}
    grip_cmd = {role: HL.build_grip_command(calib, grip_ctl[role], grip_lock)
                for role in grip_roles}
    jaw_ever_commanded = {"v": False}

    # ---------------- readbacks the wire consumes ----------------
    # ⚠️ DEFINED BEFORE THE HEARTBEAT THREAD STARTS, not merely before its first use: the
    # heartbeat calls `_idle_sync_wire` -> `_record_frame` on its very first tick, and a `def`
    # further down the body has not run yet when that happens (observed in hl_serving_mc as
    # `NameError: cannot access free variable '_record_frame'`).
    def _record_frame():
        msg, _wall, _n = subs["record"].latest()
        try:
            return parse_record_frame(msg)
        except Exception:  # noqa: BLE001 - a malformed frame is viz/clutch input only
            return None

    def _measured_force_n() -> float | None:
        """Worst |F| across the hands the rig reads, or None if unknown.

        **None is not zero** — the carried anchor treats an unknown reading as "do not disable the
        guard", the same contract ``hl_anchor.ChunkAnchor.resolve`` states, so a dead Bota bridge
        cannot quietly turn a press into an advancing command.
        """
        if not force_roles:
            return None
        fmsg, _wall, _n = subs["force"].latest()
        bins = HL._pick_force_channels(fmsg, force_roles, not args.allow_3dof_force)
        if bins is None:
            return None
        arr = np.asarray(bins, dtype=np.float64).reshape(-1, 6)
        if not np.all(np.isfinite(arr)):
            return None
        return float(np.linalg.norm(arr[:, :3], axis=1).max())

    def _gate_force(wrench_rows) -> np.ndarray | None:
        """Per-hand predicted wrenches -> the ``(3*n_hands,)`` sensor-frame F^cmd, or None.

        Same two-level contract and the same sign flip in the same one place as the other two
        planners: the model predicts ``F_meas`` (what the Bota READ during the demo) and the low
        level closes ``e = F^cmd + F_meas -> 0``, so the command reproducing the demonstrated
        press is ``-F_meas``. Gated on the MAGNITUDE, which the flip preserves.

        BOTH hands or NEITHER: ``ViveLiveSource`` reads a (6,) f_cmd as ``[L, R]``, so a partial
        publish would leave the RIGHT command at whatever the low level last had. A below-gate
        hand is published as an explicit ZERO, which is what "pure position for that hand" means.
        """
        with force_lock:
            enabled, gate_n = force_ctl["enabled"], force_ctl["gate_n"]
        if not enabled or not force_roles:
            return None
        out, any_live = [], False
        for row in wrench_rows:
            if row is None or not np.all(np.isfinite(row[0:3])):
                out.append(np.zeros(3))
                continue
            f3 = np.asarray(row[0:3], dtype=np.float64)
            if float(np.linalg.norm(f3)) <= gate_n:
                out.append(np.zeros(3))
            else:
                out.append(cmd_from_meas_force(f3))
                any_live = True
        return np.concatenate(out) if any_live else None

    def _set_wire(frame, fcmd) -> None:
        with wire_lock:
            last_wire["frame"] = tuple(np.asarray(a, dtype=np.float64).copy() for a in frame)
            last_wire["fcmd"] = None if fcmd is None else np.asarray(fcmd, dtype=np.float64).copy()

    def _send(frame, fcmd) -> bool:
        """Publish ONE dense frame. False (and nothing sent) if unsafe.

        The gate lives in the wire object and is applied here and only here, at the last moment
        before the socket: the low level validates only WIDTH and ``json.dumps`` serializes NaN
        unchecked, so a bad joint target would otherwise reach the robot. On a refusal we send
        NOTHING and let the freshness gate (S0Driver 0.5 s -> the low level's 0.25 s) hold.
        """
        ok, reason = wire.publish(s0, frame, fcmd)
        if not ok:
            _log(f"REFUSED a {wire.mode} frame: {reason} — sending nothing; the stale gate holds.")
        return ok

    def _idle_sync_wire() -> None:
        """While UN-ARMED, track the robot's MEASURED joints; once ARMED, freeze.

        Both halves are load-bearing, and both were live-deploy bugs on the left_only planner:

        * TRACK WHILE UN-ARMED. The joint-space pre-arm gate compares the broadcast wire against
          the joints the robot is holding, so a wire pinned to a FROZEN ready pose refuses to arm
          the moment the two differ — and raising the tolerance instead is worse, because arming
          then FLIES the body to that stale target.
        * FREEZE ONCE ARMED. An armed low level applies the broadcast wire as its target every
          tick, so continuing to chase the measured pose erases the servo's own holding error —
          the arm DROOPS, one heartbeat at a time.

        A missing / unparseable RecordTap holds the last wire, which is the honest degradation:
        without a readback there is nothing to track toward.
        """
        rframe = _record_frame()
        if rframe is None or rframe.vive_armed:
            return
        frame = wire.measured(rframe)
        if frame is not None:
            _set_wire(frame, None)

    def _heartbeat() -> None:
        period = 1.0 / max(args.drip_hz, 1.0)
        clutch_period = 1.0 / 10.0     # the remap is cheap, but 60 Hz of it is pointless
        next_clutch = 0.0
        while not stop.is_set():
            if not dripping.is_set():
                try:
                    now = time.monotonic()
                    # Never chase measured while a plan is STAGED: the staged chunk was clutched
                    # to a base captured at Predict, and moving the published wire underneath it
                    # would make the first executed step a jump of exactly that difference.
                    if now >= next_clutch and staged["out"] is None:
                        next_clutch = now + clutch_period
                        _idle_sync_wire()
                    with wire_lock:
                        frame = tuple(a.copy() for a in last_wire["frame"])
                        fc = None if last_wire["fcmd"] is None else last_wire["fcmd"].copy()
                    # The switch is authoritative on EVERY tick, so turning it off (or Stop) makes
                    # the wire pure position within one heartbeat.
                    with force_lock:
                        on = force_ctl["enabled"]
                    _send(frame, fc if on else None)
                except Exception as exc:  # noqa: BLE001 - the heartbeat must never die
                    _log(f"heartbeat error: {exc!r}")
            time.sleep(period)

    threading.Thread(target=_heartbeat, daemon=True, name="hb").start()

    # ---------------- observation ingest ----------------
    def _ingest_loop() -> None:
        period = 1.0 / max(args.obs_hz, 1.0)
        require_6dof = not args.allow_3dof_force
        while not stop.is_set():
            t0 = time.monotonic()
            try:
                for i in range(loaded.num_cameras):
                    msg, _wall, _n = subs[f"camera{i}"].latest()
                    if msg is None:
                        continue
                    frame = HL._decode_jpeg_b64(msg.get("jpeg_b64"))
                    if frame is None:
                        continue
                    # Natural channel order — `_decode_jpeg_b64` already returns RGB and the training
                    # zarr is RGB; the flip that stood here until 2026-09-20 fed the model BGR (found and
                    # measured on the 5pt page — see hl_serving_5pt.py's ingest loop).
                    planner.push_image(HL._resize_center_crop_224(frame), camera=i)
                if force_roles:
                    fmsg, _wall, _n = subs["force"].latest()
                    bins = HL._pick_force_channels(fmsg, force_roles, require_6dof)
                    if bins is not None:
                        planner.push_force(bins)
                for role in grip_roles:
                    gmsg, _wall, _n = subs[f"grip_{role}"].latest()
                    if gmsg is None or gmsg.get("actual") is None:
                        continue
                    width = calib.to_width_m(float(gmsg["actual"]),
                                             float(gmsg.get("open", 0.0)),
                                             float(gmsg.get("closed", 0.0)))
                    planner.push_gripper(role, width)
                # `--state-source measured` only. A no-op in `command` mode by construction (the
                # planner ignores it), so this loop never has to know which source is armed —
                # keeping that decision in exactly one place.
                rframe = _record_frame()
                if rframe is not None and rframe.dof_pos is not None:
                    planner.push_measured_joints(rframe.dof_pos)
                planner.push_proprio_joints()
            except Exception as exc:  # noqa: BLE001 - ingest must survive a bad frame
                _log(f"ingest error: {exc!r}")
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    threading.Thread(target=_ingest_loop, daemon=True, name="ingest").start()

    # ---------------- the scene ----------------
    server = viser.ViserServer(host=args.host, port=args.port,
                              label="FUJI WHOLE-BODY planner (29 joints -> dense)")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", plane="xy", shadow_opacity=0.0, infinite_grid=True,
                          cell_size=0.25)
    # ONE WORLD, floor at z=0, pelvis at `anchor.pelvis_z`.
    #
    # BOTH bodies are pinned at that ONE height and neither carries a root, and that is the honest
    # render rather than a simplification: a wholebody chunk has NO root channel (the wire is 29
    # positions + 29 finite-differenced velocities), and the pelvis height is the low level's
    # balance controller's business, not the policy's. So a predicted crouch shows as the body
    # sinking, which is what the joints say, and the ghost registers with the live mesh by
    # construction — at step 0 they coincide exactly, because the joint clutch makes step 0 the
    # measured pose.
    robot_root = server.scene.add_frame("/robot", show_axes=False,
                                        position=(0.0, 0.0, float(anchor.pelvis_z)))
    urdf = ViserUrdf(server, urdf_or_path=Path(args.yam_urdf), root_node_name="/robot")
    urdf_joints = list(urdf.get_actuated_joint_names())
    dof_remap = build_sdk_to_urdf_index(urdf_joints)
    if dof_remap.missing_sdf:
        _log(f"⚠️  URDF is missing SDK joints {dof_remap.missing_sdf} — the live mesh is disabled "
             "rather than drawn with a limb in the wrong place")
    urdf.update_cfg(remap_dof_to_urdf(anchor.dof, dof_remap))
    del robot_root
    ghost_root = server.scene.add_frame("/ghost", show_axes=False, visible=True,
                                       position=(0.0, 0.0, float(anchor.pelvis_z)))
    ghost = ViserUrdf(server, urdf_or_path=Path(args.yam_urdf), root_node_name="/ghost",
                      mesh_color_override=_GHOST_RGBA)

    def _reset_ghost() -> None:
        """Put the violet body back at the ready pose.

        Joints ONLY, unlike ``hl_serving_mc``'s ``_reset_ghost``: that one also restores a
        floating ROOT, because a motion completion answers with a root and its ghost can take a
        step the live mesh cannot. A wholebody chunk has no root at all, so ``/ghost`` never
        moves from ``anchor.pelvis_z`` and there is nothing else to restore. Called at both the
        construction site and the Reset branch so the two cannot drift.
        """
        ghost.update_cfg(remap_dof_to_urdf(anchor.dof, dof_remap))

    _reset_ghost()

    # ---------------- predict ----------------
    def _predict() -> str:
        """Sample one chunk, clutch it into wire frames, stage it."""
        reasons, _ = HL._obs_freshness(planner, max_age_s=planner.obs_max_age_s,
                                       grip_max_age_s=planner.grip_max_age_s,
                                       consumes_force=consumes_force)
        if reasons and args.obs_max_age is not None and float(args.obs_max_age) != 0:
            return "REFUSED: " + "; ".join(reasons)
        if reasons:
            _log("⚠️  stale observation: " + "; ".join(reasons))
        seed_ctl["n"] += 1
        with force_lock:
            enabled, gate_n = force_ctl["enabled"], force_ctl["gate_n"]
        t0 = time.monotonic()
        try:
            chunk = planner.infer_chunk(seed=seed_ctl["n"], force_enabled=enabled, gate_n=gate_n)
        except Exception as exc:  # noqa: BLE001 - a failed sample must not publish
            stats["err"] = f"{type(exc).__name__}: {exc}"
            return f"SAMPLE FAILED: {stats['err']} — nothing staged."
        t_sample = (time.monotonic() - t0) * 1e3
        # WHERE THE NEXT CHUNK'S WIRE CONTINUES FROM (--post-anchor, owned by the clutch):
        #   off      re-base on the robot's MEASURED joints when a readback exists, so the chunk
        #            starts from where the body IS.
        #   command  the last EXECUTED step is carried forward instead, leak-bounded against those
        #            same measured joints and decayed under contact.
        try:
            wire_src = wire.sync(_record_frame(), _measured_force_n())
        except Exception as exc:  # noqa: BLE001 - a bad readback must not lose the chunk
            _log(f"wire re-base skipped: {exc!r}")
            wire_src = "last published"
        plan = wire.plan(chunk)
        leg_rad, leg_name = hl_wholebody.leg_excursion_rad(chunk.joints)
        stats["n"] += 1
        stats["ms"] = t_sample
        stats["legs"], stats["leg_name"] = leg_rad, leg_name
        staged["out"] = {"chunk": chunk, "plan": plan, "wire_src": wire_src}
        # The LEG excursion is reported on every Predict whichever --legs mode is armed, because
        # it is a property of the CHUNK: "how much leg did the policy ask for" is the number that
        # decides whether --legs hold is wanted, and it must not be invisible when it is applied.
        leg_note = (f" · legs {np.degrees(leg_rad):.1f}° ({leg_name})"
                    + (" HELD" if wire.legs == "hold" else ""))
        return (f"staged: sample {t_sample:.0f} ms · {plan['n']} step(s) · base {wire_src}"
                f"{leg_note}")

    # ---------------- execute ----------------
    def _drip(out) -> int:
        """Publish the first ``exec_ctl['n']`` steps, one per 30 Hz step."""
        chunk, plan = out["chunk"], out["plan"]
        n_steps = max(1, min(int(exec_ctl["n"]), int(plan["n"])))
        step_dt = 1.0 / max(args.step_hz, 1.0)
        drip_dt = 1.0 / max(args.drip_hz, 1.0)
        last = 0
        dripping.set()
        try:
            for i in range(n_steps):
                if stop.is_set():
                    break
                rows = [chunk.wrench_pred.get(r) for r in force_roles]
                rows = [None if w is None else w[i] for w in rows]
                fcmd = _gate_force(rows)
                frame = wire.frame(plan, i)
                _set_wire(frame, fcmd)
                last = i
                if grip_enabled.is_set() and s0.num_clients > 0:
                    for role in grip_roles:
                        widths = chunk.gripper_m.get(role)
                        if widths is None or widths.shape[0] <= i or not np.isfinite(widths[i]):
                            continue
                        frac, _sq = grip_cmd[role][0](float(widths[i]))
                        if np.isfinite(frac):
                            subs[f"grip_{role}"].send({"target_frac": float(frac)})
                            jaw_ever_commanded["v"] = True
                t_step = time.time()
                while True:
                    # The gate is re-evaluated per re-send, so flipping the switch off acts as an
                    # interrupt on a chunk that is already executing.
                    _send(frame, _gate_force(rows))
                    if stop.is_set():
                        break
                    remaining = step_dt - (time.time() - t_step)
                    if remaining <= 0:
                        break
                    time.sleep(min(drip_dt, remaining))
        finally:
            dripping.clear()
        # Commit BOTH sides to the LAST EXECUTED step, so the next chunk branches from where the
        # body actually stopped rather than from the un-executed chunk end:
        #   * the PROPRIO latch, in the MODEL's posture space — the model's own output, NOT the
        #     clutched wire value. That distinction is the whole point: the wire is in the robot's
        #     space and the observation must be in the model's, or the proprio drifts by the
        #     posture-anchor gap one chunk at a time. `commit_chunk_row` (not `set_joint_target`)
        #     because a wb_joint_target='delta' row is MOTION and has to be integrated onto the
        #     latch rather than replace it; for 'absolute' the two are the same call.
        #   * the WIRE base, in the robot's space. Under --post-anchor command this is the
        #     INTEGRATOR; with 'off' it is the within-chunk advance, overwritten at the next
        #     Predict's `wire.sync`.
        planner.commit_chunk_row(chunk.joints[last])
        wire.commit(wire.frame(plan, last))
        return last + 1

    def _release_jaw_latches() -> None:
        """Drop every hand's squeeze latch. Called on Stop / Reset: an abandoned grasp must not
        keep the next chunk's target pinned at the width the aborted one had reached."""
        for role in grip_roles:
            grip_cmd[role][1]()

    def _open_jaws(why: str) -> None:
        """Command BOTH jaws fully OPEN. Reset and shutdown, never mid-chunk.

        The startup state is the gripper bridge's job (``keyboard_gripper --initial-frac 1.0``
        brings both hands up open). This is the other end of the same contract: once we have
        squeezed, an abandoned grasp must not be left clamped. Only what we actually commanded is
        released, so with the switch off the jaws are left exactly where the bridge has them.
        """
        if not (jaw_ever_commanded["v"] and grip_roles):
            return
        _release_jaw_latches()
        for role in grip_roles:
            try:
                subs[f"grip_{role}"].send({"target_frac": 1.0})
            except Exception as exc:  # noqa: BLE001 - a jaw release must not mask the caller
                _log(f"{why}: could not open the {role} jaw ({exc!r})")
        _log(f"{why}: commanded BOTH jaws open")

    # ---------------- GUI ----------------
    with server.gui.add_folder("Status"):
        server.gui.add_text("layout", initial_value=f"{loaded.layout.mode} "
                            f"({loaded.layout.action_dim}-D action) · joints "
                            f"{wire.joint_target}",
                            disabled=True)
        server.gui.add_text("wire family", disabled=True,
                            initial_value=f"{wire.mode} — MODE={wire.low_level_mode}-family")
        server.gui.add_text("clutch", disabled=True,
                            initial_value=f"{args.wb_clutch} · legs {wire.legs} · "
                                          f"{wire.clutch.post.describe()[:60]}")
        server.gui.add_text("proprio", disabled=True,
                            initial_value=f"{args.state_source} · seed {seed_note[:80]}")
        g_obs = server.gui.add_text("obs age", initial_value="…", disabled=True)
        g_ll = server.gui.add_text("low level", initial_value="…", disabled=True)
        g_chunk = server.gui.add_text("last chunk", initial_value="no chunk yet", disabled=True)
        g_wire = server.gui.add_text("wire", initial_value="ready pose (idle)", disabled=True)
        g_state = server.gui.add_text("state joints", initial_value="…", disabled=True)

    with server.gui.add_folder("Control"):
        b_predict = server.gui.add_button("Predict")
        b_run = server.gui.add_button("Run staged chunk")
        # ONE-CLICK SEQUENTIAL ROLLOUT — the cadence the whole path is meant to run at:
        # predict -> execute, repeatedly, with the operator arming ONCE. Manual Predict/Run stay
        # as the single-step form for bring-up and for watching one chunk.
        b_auto = server.gui.add_button("▶▶ Run to completion (sequential)")
        b_auto_stop = server.gui.add_button("⏹ Stop sequential run", disabled=True)
        b_stop = server.gui.add_button("STOP")
        b_reset = server.gui.add_button("Reset to ready pose")
        # DERIVED from the horizon, never hardcoded: an H=200 checkpoint offers 200 here and an
        # H=50 one does not. Options stay BARE INTEGERS because the handler below does
        # `int(d_exec.value)` — the seconds belong in the hint, not in the option text.
        d_exec = server.gui.add_dropdown(
            "Execute how many steps", tuple(str(n) for n in _EXEC_CHOICES),
            initial_value=str(exec_ctl["n"]),
            hint=(f"of the {ACTION_HORIZON}-step chunk, at {1.0 / ACTION_DT:.0f} Hz: "
                  + ", ".join(f"{n}={n * ACTION_DT:.2f}s" for n in _EXEC_CHOICES)
                  + ". The chunk is always predicted in full; this is how much of it executes "
                    "before the next Predict."))
        s_auto_secs = server.gui.add_slider("Sequential run seconds", min=1.0, max=120.0, step=1.0,
                                            initial_value=float(auto_ctl["max_s"]))
        c_ghost = server.gui.add_checkbox("Show the prediction (violet)", initial_value=True)
        g_msg = server.gui.add_text("last action", initial_value="idle", disabled=True)

    with server.gui.add_folder("Force command"):
        c_force = server.gui.add_checkbox("Command force (needs FORCE_CMD_WIRE=1)",
                                         initial_value=bool(args.enable_force_cmd))
        s_gate = server.gui.add_slider("gate (N)", min=0.0, max=20.0, step=0.1,
                                       initial_value=float(args.force_gate_n))
        c_grip = server.gui.add_checkbox("Command both jaws", initial_value=args.command_gripper)

    with server.gui.add_folder("Bring-up report"):
        for key in sorted(report):
            server.gui.add_text(key[:24], initial_value=str(report[key])[:110], disabled=True)

    flags: dict = {"predict": False, "run": False, "reset": False, "ghost_step": 0}

    # ---------------- the sequential rollout ----------------
    def _run_to_completion() -> None:
        """predict → execute, repeatedly, until Stop / the ceiling. ONE arm gesture.

        WHAT IT DOES NOT CHANGE — the safety envelope. Every iteration goes through exactly the
        same ``_predict`` → ``_drip`` pair the buttons call: the same obs-freshness refusal, the
        same ``wire.sync`` re-base, the same per-step force gate re-read, the same
        ``wire.commit`` + proprio commit to the LAST EXECUTED step, the same ``busy`` mutex. It
        removes the click and nothing else.

        BOUNDED FOUR WAYS: ``⏹`` (retires the generation), ``STOP`` (also interrupts the chunk in
        flight, within one step), the wall-clock ceiling ``--auto-max-seconds`` (live on the
        slider), and the low level dropping off :8770.
        """
        if auto_running.is_set():
            g_msg.value = "ignored — a sequential run is already active; press ⏹"
            return
        if dripping.is_set():
            g_msg.value = "ignored — a chunk is executing"
            return
        if s0.num_clients == 0:
            g_msg.value = ("the low level is NOT connected to :8770 — start it in ABSOLUTE mode "
                           f"with a {wire.low_level_mode}-family MODE")
            return
        with auto_lock:
            auto_ctl["running"] = True
            my_gen = int(auto_ctl["gen"])
            max_s = float(auto_ctl["max_s"])
        auto_running.set()
        b_auto_stop.disabled = False
        # THE ONLY PLACE THIS ROLLOUT CLEARS `stop`. A stop latched by an earlier abort must not
        # make the first iteration a silent no-op; every LATER `stop` is the operator's STOP and
        # has to both interrupt the drip and end the loop, so clearing it per iteration would make
        # STOP unpressable here.
        stop.clear()
        staged["out"] = None
        n_chunks = n_steps = 0
        openloop_warn = {"t": 0.0}
        t0 = time.time()
        _log(f"sequential rollout: START (ceiling {max_s:g} s, {exec_ctl['n']} step(s)/chunk)")
        try:
            while not stop.is_set():
                with auto_lock:
                    if auto_ctl["gen"] != my_gen or not auto_ctl["running"]:
                        g_msg.value = (f"sequential run STOPPED — {n_chunks} chunk(s), "
                                       f"{n_steps} step(s)")
                        break
                elapsed = time.time() - t0
                if elapsed >= max_s:
                    g_msg.value = (f"sequential run finished — {max_s:g} s ceiling "
                                   f"({n_chunks} chunk(s), {n_steps} step(s))")
                    break
                # The low level going away mid-rollout is not "nothing moves": the heartbeat keeps
                # publishing into a dead socket while this loop keeps planning off it.
                if s0.num_clients == 0:
                    g_msg.value = (f"sequential run STOPPED — the low level dropped off :8770 "
                                   f"after {n_steps} step(s)")
                    _log("SEQUENTIAL: the low level disconnected from :8770 — rollout ended.")
                    break
                if not busy.acquire(blocking=False):
                    time.sleep(0.02)   # a manual Predict/Reset holds the mutex; wait it out
                    continue
                try:
                    # Rate-limited, because it is true for the WHOLE run when it is true at all.
                    # Worse here than for one manual chunk: with no readback every chunk re-bases
                    # on the last COMMANDED frame, so a tracking shortfall compounds with nothing
                    # measuring it.
                    if _record_frame() is None:
                        now = time.monotonic()
                        if now - openloop_warn["t"] > 5.0:
                            openloop_warn["t"] = now
                            _log("SEQUENTIAL: no usable RecordTap (RECORD=1?) — chunks re-base on "
                                 "the last COMMANDED frame, so the shortfall compounds.")
                    note = _predict()
                    out = staged["out"]
                    if out is None:
                        # A refusal or a failed sample. Ending the rollout is the only honest
                        # response: the alternative is a loop that keeps re-planning off a frozen
                        # scene until the ceiling, publishing the last frame the whole time.
                        g_msg.value = f"sequential run STOPPED — {note}"
                        _log(f"SEQUENTIAL: stopped, nothing staged — {note}")
                        break
                    flags["ghost_step"] = 0
                    n_done = _drip(out)
                    staged["out"] = None
                    n_chunks += 1
                    n_steps += int(n_done)
                    g_msg.value = (f"sequential: {n_chunks} chunk(s), {n_steps} step(s), "
                                   f"{elapsed:.1f}/{max_s:g} s · {note}")
                    if n_done == 0:   # stop fired inside the drip — do not spin
                        break
                finally:
                    busy.release()
        except Exception as exc:  # noqa: BLE001 - a rollout error must not kill the page
            import traceback

            _log("sequential rollout error:\n" + traceback.format_exc())
            g_msg.value = f"sequential run error: {exc!r}"
        finally:
            with auto_lock:
                if auto_ctl["gen"] == my_gen:
                    auto_ctl["running"] = False
            auto_running.clear()
            b_auto_stop.disabled = True
            _log(f"sequential rollout: END — {n_chunks} chunk(s), {n_steps} step(s), "
                 f"{time.time() - t0:.1f} s")

    @b_predict.on_click
    def _(_e) -> None:
        flags["predict"] = True

    @b_run.on_click
    def _(_e) -> None:
        flags["run"] = True

    @b_auto.on_click
    def _(_e) -> None:
        # Its OWN thread, not the page's callback thread and not the render loop: the rollout
        # blocks for the whole take and both of those must stay responsive — STOP is on one.
        threading.Thread(target=_run_to_completion, daemon=True, name="seq").start()

    @b_auto_stop.on_click
    def _(_e) -> None:
        # Retire the loop by bumping the generation: the chunk in flight finishes (bounded by
        # exec-steps) and the loop then exits without re-planning. Use STOP to interrupt the
        # motion ITSELF within one step.
        with auto_lock:
            auto_ctl["running"] = False
            auto_ctl["gen"] += 1

    @b_stop.on_click
    def _(_e) -> None:
        # Flag only; the loop thread owns the state. `stop` is read by the drip loop on every step
        # and by its inner re-send, so it interrupts an executing chunk within one step.
        stop.set()
        # STOP also RETIRES a sequential rollout, closing the window where the loop is between
        # iterations: it can then never re-plan after a STOP, only exit.
        with auto_lock:
            auto_ctl["running"] = False
            auto_ctl["gen"] += 1
        _release_jaw_latches()

    @b_reset.on_click
    def _(_e) -> None:
        flags["reset"] = True

    @d_exec.on_update
    def _(_e) -> None:
        # Never let a malformed selector value raise inside viser's callback: an exception here
        # leaves the control silently dead (the page keeps showing the new number while the drip
        # loop executes the old one). Snapped as well as parsed, so `exec_ctl["n"]` is by
        # construction a member of the offered set.
        try:
            n = int(d_exec.value)
        except (TypeError, ValueError):
            d_exec.value = str(exec_ctl["n"])
            return
        exec_ctl["n"] = snap_exec_steps(n, _EXEC_CHOICES)

    @s_auto_secs.on_update
    def _(_e) -> None:
        # Re-read every iteration, so shortening it mid-rollout takes effect at the next chunk
        # boundary. It can only ever stop the body SOONER.
        with auto_lock:
            auto_ctl["max_s"] = float(s_auto_secs.value)

    @c_force.on_update
    def _(_e) -> None:
        with force_lock:
            force_ctl["enabled"] = bool(c_force.value)

    @s_gate.on_update
    def _(_e) -> None:
        with force_lock:
            force_ctl["gate_n"] = float(s_gate.value)

    @c_grip.on_update
    def _(_e) -> None:
        (grip_enabled.set if c_grip.value else grip_enabled.clear)()

    @c_ghost.on_update
    def _(_e) -> None:
        ghost_root.visible = bool(c_ghost.value)

    # ---------------- the loop ----------------
    _log(f"page: http://{args.host}:{args.port}")
    _log("sequence: press A on the joystick (ready pose) → Predict → arm (start) → then either "
         "'Run staged chunk' for ONE chunk, or '▶▶ Run to completion' to keep going until ⏹ / "
         f"the {args.auto_max_seconds:g} s ceiling.")
    last_status = 0.0
    while True:
        try:
            # A sequential rollout owns Predict/Run/Reset for its duration. It holds `busy` only
            # WITHIN a chunk, so without these three refusals a manual press could land in the gap
            # between two iterations — a Predict would overwrite the chunk the rollout is about to
            # drip, and a Reset would clear the operator's latched STOP underneath a live take.
            if auto_running.is_set() and (flags["predict"] or flags["run"] or flags["reset"]):
                flags["predict"] = flags["run"] = flags["reset"] = False
                g_msg.value = ("a sequential run is active — press ⏹ to end it (or STOP to "
                               "interrupt the motion now)")
            if flags["reset"]:
                flags["reset"] = False
                stop.clear()
                staged["out"] = None
                wire.clutch.reset()
                planner.reset_joints()
                _open_jaws("reset")   # drops the latches too
                _reset_ghost()
                flags["ghost_step"] = 0
                # The PROPRIO state resets unconditionally — it is pure in-process bookkeeping.
                # The WIRE only resets while UN-ARMED: re-publishing the ready pose to an armed
                # low level would drive the whole body back to it from wherever it is, at an
                # unbounded distance. Armed, the wire holds the last executed frame and the idle
                # clutch stays frozen; press Reset before arming (or disarm first) to re-home.
                rframe = _record_frame()
                armed = rframe is not None and rframe.vive_armed
                if not armed:
                    _set_wire(wire.idle(), None)
                g_msg.value = ("reset: proprio + clutch + violet ghost back at the ready pose; "
                               "the WIRE is held (ARMED — disarm to re-home the body)" if armed
                               else "reset: proprio, clutch, ghost and wire back at the ready pose")
            if flags["predict"]:
                flags["predict"] = False
                if busy.acquire(blocking=False):
                    try:
                        g_msg.value = "predicting …"
                        g_msg.value = _predict()
                        if staged["out"] is not None:
                            flags["ghost_step"] = 0
                    finally:
                        busy.release()
                else:
                    g_msg.value = "busy"
            if flags["run"]:
                flags["run"] = False
                out = staged["out"]
                if out is None:
                    g_msg.value = "nothing staged — press Predict first"
                elif s0.num_clients == 0:
                    g_msg.value = ("the low level is NOT connected to :8770 — start it in "
                                   f"ABSOLUTE mode with a {wire.low_level_mode}-family MODE")
                elif busy.acquire(blocking=False):
                    try:
                        stop.clear()
                        n = _drip(out)
                        staged["out"] = None
                        g_msg.value = f"executed {n} step(s); re-committed on the last one"
                    finally:
                        busy.release()
                else:
                    g_msg.value = "busy"

            # The violet ghost plays the STAGED chunk's PUBLISHED joints on a loop — the numbers
            # on the wire, not the model's raw absolute pose. So with the default joint clutch its
            # step 0 sits exactly on the robot's own pose rather than 0.5 rad away from it, and
            # "the ghost did something wrong" and "the robot was told something wrong" are the same
            # statement. No IK: the joints ARE the command.
            out = staged["out"]
            if out is not None and ghost_root.visible:
                plan, chunk = out["plan"], out["chunk"]
                i = flags["ghost_step"] % int(plan["n"])
                flags["ghost_step"] = i + 1
                # An UNKNOWN jaw renders OPEN, never 0.0 — 0.0 is the CLOSED end of this joint's
                # travel, so the naive default draws a hand that is HOLDING something. It also
                # agrees with the hardware (both hands come up open by declaration) and with the
                # live mesh, whose remap_dof_to_urdf defaults are already OPEN.
                jaws = {}
                for role in grip_roles:
                    widths = chunk.gripper_m.get(role)
                    jaws[role] = (jaw_joint_from_width_m(float(widths[i]))
                                  if widths is not None and i < widths.shape[0]
                                  else JAW_JOINT_OPEN_M)
                ghost.update_cfg(remap_dof_to_urdf(
                    plan["published"][i], dof_remap,
                    left_jaw_m=jaws.get("left", JAW_JOINT_OPEN_M),
                    right_jaw_m=jaws.get("right", JAW_JOINT_OPEN_M)))

            now = time.monotonic()
            if now - last_status > 0.25:
                last_status = now
                _, readout = HL._obs_freshness(
                    planner, max_age_s=planner.obs_max_age_s,
                    grip_max_age_s=planner.grip_max_age_s, consumes_force=consumes_force)
                g_obs.value = readout
                rframe = _record_frame()
                if rframe is None:
                    g_ll.value = (f"{s0.num_clients} client(s) on :8770 · RecordTap ABSENT "
                                  "(no RECORD=1) — clutch is OPEN-LOOP")
                else:
                    g_ll.value = (f"{s0.num_clients} client(s) · running={rframe.running} "
                                  f"armed={rframe.vive_armed} servo={rframe.servo_on}")
                    if rframe.dof_pos is not None:
                        urdf.update_cfg(remap_dof_to_urdf(rframe.dof_pos, dof_remap))
                if stats["n"]:
                    g_chunk.value = (
                        f"{stats['n']} chunk(s) · sample {stats['ms']:.0f} ms · legs "
                        f"{np.degrees(stats['legs']):.1f}° ({stats['leg_name']})"
                        + (" HELD" if wire.legs == "hold" else "")
                        + (f" · last error {stats['err']}" if stats["err"] else ""))
                q_state, age = planner.joint_state()
                g_state.value = (
                    f"worst |q - ready| {np.abs(q_state - np.asarray(anchor.dof)).max():.3f} rad · "
                    + ("never written (SEED, flagged UNMEASURED)" if age is None
                       else f"{age * 1e3:.0f} ms old"))
                with wire_lock:
                    fc = last_wire["fcmd"]
                    frame = tuple(a.copy() for a in last_wire["frame"])
                g_wire.value = (f"{wire.status(frame)} · "
                                f"F^cmd {'OFF' if fc is None else f'{np.abs(fc).max():.1f} N'}")
            time.sleep(0.02)
        except KeyboardInterrupt:
            _log("interrupted")
            # Release before main()'s finally stops the subscribers — a Ctrl-C mid-task must not
            # leave a hand clamped with nothing left running to open it. Blocks briefly so the
            # frame actually gets on the wire.
            _open_jaws("shutdown")
            for _ in range(200):
                pending = False
                for role in grip_roles:
                    sub = subs[f"grip_{role}"]
                    with sub._lock:
                        pending = pending or bool(sub._send_q)
                if not pending:
                    break
                time.sleep(0.01)
            return 0
        except Exception as exc:  # noqa: BLE001 - the page must outlive one bad tick
            _log(f"loop error: {exc!r}")
            time.sleep(0.2)


if __name__ == "__main__":
    raise SystemExit(main())
