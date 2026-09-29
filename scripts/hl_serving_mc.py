#!/usr/bin/env python3
"""ONLINE 3-TRACKER PLANNER with MOTION COMPLETION -> a WHOLE-BODY low-level controller.

The deploy half of ``diffusion_umi_3tracker``. Same predict -> preview -> execute UX as
``scripts/hl_serving.py``, same sensor bring-up, same force gate, same freshness rules — but
the chunk it publishes has been through motion completion, and the wire it publishes on is
the whole body rather than 3 keypoints. ``--low-level`` picks which whole-body family::

    2 D405 + 2 Bota + 2 jaws --> S1 (both_hands_head_dual, H=50 @ 30 Hz)
                                       |  3 anchor-relative EE chunks
                    decode against the G1's READY POSE (the pose held after `A`)
                                       |  3 absolute pose series
                            Motion_Completion  (2 windows, ~80 ms)
                                       |  qpos(29) + its own FK
                    ,--------------------------------------------.
  --low-level 5pt   |                                            |  --low-level dense
  :8770 {"mode":"5pt"}                                :8770 {"mode":"dense"}
  5 keypoints, PELVIS frame                           29 joints, by NAME
  g1-…-5pt-hybrid-unified-anneal-wbt                  g1-…-dense-hybrid-unified-anneal-wbt
  (W&B tqnoooo6, MODE=5pt-unified-anneal)             (W&B 7vxmnwo6, MODE=dense-unified-anneal)

Everything upstream of the split is byte-identical: the same checkpoint, the same six sensor
bridges, the same ready-pose decode, the same completion. What differs is only how the
completed body is read out, and the low level's pre-arm gate along with it — keypoint-space in
metres for 5pt (on 3 of the 5 points), joint-space in radians for dense.

WHY A SIBLING SCRIPT AND NOT A FLAG ON hl_serving.py
====================================================
``hl_serving.py`` is LEFT-CANONICAL by construction and says so in a load-bearing refusal:
it builds the ``vr_3point`` LEFT slot from the model, holds RIGHT+HEAD at their measured
pose, applies the URDF<->model seam at exactly two LEFT boundaries, compensates reach on a
single TCP, and previews with a LEFT arm IK. **None of that machinery is used here.** A
dense low level has no keypoint front-end at all, so there is no ``WireMapper``, no
wrist->contact lever, no per-role hold, and no arm IK — the completion IS the preview.

What the two share is the part that took ten review rounds to get right, and that part
already lives in importable modules rather than in the script: :mod:`hl_online` (the
observation rings, the decode, the sampler seam), :mod:`hl_dense` (this pipeline's core),
:mod:`live_robot` (the RecordTap parse + the by-name SDK->URDF remap), :mod:`ready_pose` (the
deploy anchor). The sensor plumbing and the gripper calibration are imported FROM
``hl_serving`` directly, so a fix to either lands in both. Forking those would have been the
real duplication; forking a wire that does not exist in the other mode is not.

THE THREE THINGS THAT ARE GENUINELY DIFFERENT
=============================================
1. **The anchor is the ready pose, for all three roles.** An HL-v2 chunk is anchor-relative,
   and motion completion reads the chest as its ROOT channel, so the pose you decode against
   decides the POSTURE, not just the position. ``ready_pose.resolve_deploy_anchor`` FKs
   ``home_anchor.json::default_qpos`` with the 14 arm joints zeroed — literally the
   configuration ``HOLOSOMA_INIT_READY_POSE=1`` pins the low level's hold to after **A** — and
   its floor is z=0, which is why the completion's lift is PINNED to 0 rather than derived.

2. **The command is the completion's CHANGE, not its absolute pose** — on the DENSE wire
   (``--dense-clutch joint``, the default). A 7-DoF arm is redundant: measured on
   ``ckpt_0831_final`` seeded exactly at the ready pose, the completion puts the gripper
   within **0.84 mm** of what it was asked for while placing the ARM JOINTS **0.40 rad
   (23 deg)** elsewhere in the null space, and straightening the knee **0.54 rad** out of the
   ready crouch. Published absolutely, that offset either fails the 0.20 rad joint-space
   pre-arm gate or snaps the arm 23 deg on the first executed step. See
   :class:`hl_dense.JointClutch`.

   ``--low-level 5pt`` needs no clutch, and that is a consequence rather than an omission: a
   keypoint wire carries no null space, only the points the completion was TOLD to hit plus
   the two feet it inferred. Read out against the anchor's own pelvis, step 0 lands on
   ``_compute_vr_fk(ready pose)`` — precisely the low level's gate reference — so the gate
   closes by construction. See :class:`hl_dense.FivePointWire`.

3. **The preview IS the robot.** No IK: the violet ghost plays the completion's own 29 joints,
   which are the numbers on the wire. So "the ghost did something wrong" and "the robot was
   told something wrong" are the same statement, which is not true of an IK preview.

WHAT IS DELIBERATELY NOT HERE (first bring-up)
==============================================
The take recorder, recorded-episode replay, reach compensation, temporal ensembling, the
executed-trace QC sheet and the command-side anchor. Each is a real feature of the left_only
planner and each needs a dense-specific answer (a recorded dense take is 29 joints, not a
wire pose; reach compensation is defined on a TCP). They are absent rather than half-ported.

BRING-UP, in order — every one of these is checked at startup and named if missing::

    T1  bash 2_ft_bridge.sh --pipeline              both Botas, 6-DoF, :8771
    T2  camera L :8773   and   camera R :8777       both D405s (this model reads TWO views)
    T3  LEFT=1 RIGHT=1 KEYBOARD=0 bash 5_dual_gripper_keyboard.sh    LEFT :8784, RIGHT :8785
    T4  MODE=5pt-unified-anneal HOLOSOMA_VIVE_RELATIVE=0 RECORD=1 \
          bash 3_run_policy_vive_force.sh           the low level (ABSOLUTE, RecordTap on)
    T5  ./scripts/run_hl_serving_mc.sh --low-level 5pt --checkpoint ckpt_0831_final

``--low-level`` and T4's ``MODE`` are ONE decision spelled in two places and nothing checks
them against each other: the RecordTap frame carries no preset name, so a mismatch is not
detectable from here. It is also not dangerous — the low level's width guard drops every frame
of the wrong shape and the robot holds its ready pose — but it means **"nothing moves" reads as
"the two MODEs disagree"** before it reads as anything else.

⚠️ SAFETY. This commands a physical G1 whole body — the legs are on the wire too, and the
completion's own posture is not the ready pose's. First runs gantry-suspended, operator on the
hardware E-stop, force wire OFF. Expect the legs to take up the completion's standing posture
over the first chunk (bounded by the low level's ``HOLOSOMA_VIVE_MAX_JOINT_STEP``, 0.05
rad/tick per joint); with ``--dense-clutch joint`` they start from the crouch the robot is
already holding rather than jumping to it.
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

# The sensor plumbing, the gripper calibration and the frame seam come from the left_only
# planner rather than being copied: they are model-agnostic and each has its own review
# history. Importing the module is side-effect-free (it binds nothing and parses no args until
# main()); it also installs the UMI viz/replay/humi_ik paths this process wants anyway.
import hl_serving as HL  # noqa: E402

from fuji.client.s0_driver import S0Driver  # noqa: E402
from fuji.s1_policy.data.layout import ACTION_DT, ACTION_HZ  # noqa: E402
from fuji.s1_policy.inference import ready_pose as RP  # noqa: E402
from fuji.s1_policy.inference.display import (  # noqa: E402
    read_stored_extrinsic,
    wrist_pose_from_stored,
)
from fuji.s1_policy.serving import hl_anchor, hl_dense  # noqa: E402
from fuji.s1_policy.serving.hl_online import (  # noqa: E402
    ACTION_HORIZON,
    OBS_MAX_AGE_S,
    LivePlanner,
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
from fuji.utils.rot6d import (  # noqa: E402
    quat_xyzw_to_wxyz,
    rot6d_to_rotmat,
    rotmat_to_quat_xyzw,
    rotmat_to_rot6d,
)

#: The three roles a completion needs: both hands AND the chest (its root channel). Refused by
#: name rather than fed a fabricated root — the same requirement the offline panel states.
REQUIRED_ROLES = ("left", "right", "head")

#: ``--algo`` spellings that mean "the diffusion block, on the 3-TRACKER preset". The
#: checkpoint's own ``config.json`` names one of three ALGORITHM blocks (diffusion / flow /
#: act) and the layout is a separate fact, so ``diffusion_3tracker`` is an ALIAS rather than a
#: fourth algorithm: it resolves to ``diffusion`` and is checked against the layout, which this
#: planner already refuses by name if it is not ``both_hands_head*``. Accepted because the
#: preset is what an operator thinks in — ``run_hl_serving_mc.sh`` derives ``--algo`` from the
#: checkpoint anyway, so neither spelling normally has to be typed at all.
ALGO_ALIASES = {"diffusion_3tracker": "diffusion", "diffusion-3tracker": "diffusion"}

#: Translucent violet, the same ghost colour ``hl_serving`` uses for a played-back chunk, so
#: "violet = what is about to happen" means one thing across both planners.
_GHOST_RGBA = HL._GHOST_RGBA

#: HOW EACH PREDICTED PATH IS DRAWN, per role — and the head is deliberately not styled like a
#: hand. The chest keypoint of the ready pose is ``(-0.004, 0, 1.151)``, i.e. INSIDE the torso
#: mesh, and it translates ~2-3 cm over a chunk where a hand travels tens of cm. Line-only, it
#: is depth-occluded by both the live robot and the violet ghost and reads as "the head was
#: never predicted". Its triad is therefore long enough to protrude from a torso roughly 0.15 m
#: half-deep, while a hand — already in open air — gets a short one so a reach is not buried in
#: arrows. The chest is motion completion's ROOT conditioning, so this is the path whose
#: ORIENTATION decides the posture the whole body answers with.
_PLAN_STYLE = {
    "left":  {"rgb": (60, 160, 255),  "axes": 0.07, "stride": 10},
    "right": {"rgb": (255, 120, 200), "axes": 0.07, "stride": 10},
    "head":  {"rgb": (255, 200, 40),  "axes": 0.26, "stride": 12},
}
#: Grey, mid-length, for a role a future layout adds: drawn rather than silently skipped.
_PLAN_STYLE_FALLBACK = {"rgb": (200, 200, 200), "axes": 0.10, "stride": 10}


def _triad_indices(n: int, stride: int) -> np.ndarray:
    """Which chunk steps get an orientation triad: every ``stride``-th, plus ALWAYS the last.

    The final step is where the chunk ENDS, and that is what the next chunk's wire continues
    from (``wire.commit`` carries the last EXECUTED step forward), so it is never the sample to
    drop to a stride boundary.
    """
    n = max(int(n), 1)
    stride = max(int(stride), 1)
    return np.unique(np.append(np.arange(0, n, stride), n - 1))


#: How often the idle heartbeat re-publishes the held wire. The low level's stale gate is
#: 0.25 s and ``S0Driver`` stops re-broadcasting after 0.5 s of silence, so this only has to
#: beat 0.25 s comfortably.
IDLE_HB_HZ = 60.0

#: Execute-N-of-S choices offered in the page, DERIVED from this process's horizon rather than
#: hardcoded — see :func:`hl_online.exec_step_choices`. ``ACTION_HORIZON`` is resolved from
#: ``FUJI_ACTION_HORIZON`` at import time and ``run_hl_serving_mc.sh`` exports it from the
#: checkpoint's own ``config.json``, so an H=100 checkpoint offers 100 and an H=200 one offers
#: 100 and 200 with no flag. The pin is what makes deriving it here sound: a checkpoint whose
#: horizon disagrees with the constant is refused outright by ``validate_model_layout``, so this
#: tuple can never advertise a window the model will not fill. Note the COMPLETION is sized off
#: the same horizon (``ChunkCompleter`` batches the whole chunk), so a longer chunk costs
#: proportionally more completion windows, not a different cadence.
#: MODULE scope for the reason ``hl_serving._EXEC_CHOICES`` documents: main() (the snap) and
#: _run_viser (the dropdown) both read it, and as a main() local the dropdown raised NameError
#: inside _run_viser's broad except, surfacing only as a page that never loaded.
_EXEC_CHOICES = exec_step_choices(ACTION_HORIZON)


def _log(msg: str) -> None:
    print(f"[hl-mc] {msg}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="a 3-tracker HL-v2 checkpoint dir (layout both_hands_head[_dual])")
    p.add_argument("--algo", default="diffusion",
                   choices=["diffusion", "flow", "act", *ALGO_ALIASES],
                   help="which algorithm block the checkpoint carries (default: %(default)s). "
                        "'diffusion_3tracker' is an alias for 'diffusion' — the preset name, "
                        "accepted because that is what the run is called; the 3-tracker part is "
                        "a LAYOUT fact and is checked separately, by name. muse is not "
                        "supported here: its backend runs in another interpreter and the "
                        "completion needs a third, so that combination is untested.")
    p.add_argument("--low-level", default="dense", choices=list(hl_dense.WIRE_MODES),
                   help="which whole-body low-level FAMILY to publish for (default: "
                        "%(default)s). 'dense' = 29 joint targets by name (MODE="
                        "dense-unified-anneal, W&B 7vxmnwo6); '5pt' = five keypoints in the "
                        "ready pose's pelvis frame (MODE=5pt-unified-anneal, W&B tqnoooo6). "
                        "This MUST match the MODE the low level was launched with — nothing "
                        "cross-checks it, and a mismatch reads as a robot that never moves.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-steps", type=int, default=None,
                   help="sampler steps (default: the checkpoint's own)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)

    g = p.add_argument_group("sensor bridges (the dual rig)")
    g.add_argument("--force-ws", default="ws://127.0.0.1:8771",
                   help="Bota bridge, BOTH hands, run with --pipeline (default: %(default)s)")
    g.add_argument("--camera-ws", default="ws://127.0.0.1:8773", help="camera L (camera0_rgb)")
    g.add_argument("--camera-right-ws", default="ws://127.0.0.1:8777",
                   help="camera R (camera1_rgb). REQUIRED by a 2-camera checkpoint; pass an "
                        "empty string to declare a 1-camera rig, which then refuses a "
                        "2-camera model by name instead of feeding it a grey view.")
    g.add_argument("--gripper-ws", default="ws://127.0.0.1:8784", help="LEFT jaw bridge")
    g.add_argument("--gripper-right-ws", default="ws://127.0.0.1:8785", help="RIGHT jaw bridge")
    g.add_argument("--record-ws", default="ws://127.0.0.1:8779",
                   help="low-level RecordTap (needs RECORD=1). Drives the live robot mesh, the "
                        "measured joint clutch and the low-level-lost watchdog; without it the "
                        "clutch is open-loop from the ready pose and says so.")
    g.add_argument("--s0-ws-host", default="127.0.0.1")
    g.add_argument("--s0-ws-port", type=int, default=8770)
    g.add_argument("--allow-3dof-force", action="store_true",
                   help="opt into a 3-DoF F/T bring-up (else a missing --pipeline is fatal)")

    c = p.add_argument_group("cadence")
    c.add_argument("--obs-hz", type=float, default=ACTION_HZ)
    c.add_argument("--step-hz", type=float, default=ACTION_HZ,
                   help="chunk playback rate; MUST be the completion's 30 Hz grid")
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
                        "A policy has no end of its own, so this is the bound that stops the "
                        "body even if the operator looks away.")

    d = p.add_argument_group("the completion + the wire")
    d.add_argument("--dense-clutch", default="joint", choices=list(hl_dense.CLUTCH_MODES),
                   help="DENSE ONLY (inert with --low-level 5pt, which has no null space to "
                        "clutch). 'joint' (default) publishes the completion's CHANGE from the "
                        "robot's own joints, so step 0 is a no-op and the pre-arm gate closes; "
                        "'none' publishes its absolute angles, which is 0.40 rad off the ready "
                        "pose on the arms (measured) and needs HOLOSOMA_VIVE_ARM_TOL_RAD raised")
    d.add_argument("--upper-source", default="completion", choices=list(hl_dense.UPPER_SOURCES),
                   help="5pt ONLY. Where the three GATED keypoints come from. 'completion' "
                        "(default) reads all five out of the completion's FK. 'policy' "
                        "publishes the three poses the POLICY predicted and takes only the feet "
                        "from the completion — the two agree to 0.84 mm (the completion's hand "
                        "tracking error), so this is not an accuracy choice: it decides whether "
                        "the completion sits between the policy and the points the low level "
                        "gates on.")
    d.add_argument("--foot-source", default="completion", choices=list(hl_dense.FOOT_SOURCES),
                   help="5pt ONLY. Where the two UNGATED foot keypoints come from. 'completion' "
                        "(default) keeps them consistent with the posture the predicted hands "
                        "imply — all five points then describe ONE physically possible body, "
                        "which is the completion's real value on this wire. 'hold' pins them to "
                        "the ready pose's ankles: the completed feet move only 1.9-4.4 mm per "
                        "30-frame window and the low level warns rather than gates on them, but "
                        "a held foot will not follow the operator's yaw (48.5 deg at p90) and "
                        "the 5pt policy is a whole-body BALANCE controller reading all five — "
                        "that interaction is unmeasured on hardware, so gantry only.")
    d.add_argument("--post-anchor", default="off", choices=list(hl_dense.POST_ANCHOR_MODES),
                   help="where the NEXT chunk's WIRE continues from, decided AFTER the "
                        "completion. 'off' (default) is the previous behaviour: dense re-bases "
                        "on measured joints every Predict, 5pt publishes the readout absolutely. "
                        "'command' carries the LAST EXECUTED step forward (PLAI) so the "
                        "accumulated command is not thrown away at the chunk boundary — in JOINT "
                        "space for dense, per KEYPOINT for 5pt. It cannot be applied to the "
                        "decode anchor here: that anchor is also the completion's ROOT "
                        "conditioning, so moving it changes the posture.")
    d.add_argument("--post-anchor-eps-rad", type=float, default=0.15, metavar="RAD",
                   help="[--post-anchor command, DENSE] per-joint leak bound (default 0.15, "
                        f"ceiling {hl_dense.MAX_POST_EPS_RAD:g}): how far the carried command may "
                        "lead the measured joints before it is pulled back. Unclamped command "
                        "integration diverges — 0 collapses the mode back to 'off'.")
    d.add_argument("--post-anchor-eps-m", type=float, default=0.03, metavar="M",
                   help="[--post-anchor command, 5pt] per-keypoint position leak bound "
                        f"(default 0.03, ceiling {hl_anchor.MAX_EPS_M:g}). Same bound, same "
                        "reason, as hl_serving's --anchor-eps.")
    d.add_argument("--post-anchor-eps-deg", type=float, default=10.0, metavar="DEG",
                   help="[--post-anchor command, 5pt] the same bound on each keypoint's carried "
                        f"ORIENTATION lead (ceiling {np.degrees(hl_anchor.MAX_EPS_RAD):.0f})")
    d.add_argument("--post-anchor-contact-n", type=float, default=1.5, metavar="N",
                   help="[--post-anchor command] measured |F| at or above which the carried "
                        "anchor DECAYS back toward the robot instead of advancing: a command "
                        "that keeps advancing into a blocked hand is a press, not a reach")
    d.add_argument("--mc-no-wrist-ik", action="store_true",
                   help="skip the per-window wrist IK that lands the commanded hand to <1 mm "
                        "(measured: 41 ms/chunk for 11.40 mm -> 0.79 mm)")
    d.add_argument("--mc-foot-ik", default=hl_dense.DEFAULT_FOOT_FIX,
                   choices=("none", "inertialize"),
                   help="the causal leg-IK anti-slide pass, and what the page's 'Foot IK / "
                        "anti-slide' checkbox STARTS at (that checkbox is the live control, on "
                        "both wire families, applied at the next chunk boundary). Default off, "
                        "and measured why: 1376 ms per chunk at its CHEAPEST setting against a "
                        "481 ms re-plan budget, and on the dense wire it rewrites the LEGS, "
                        "which the clutch then publishes")
    d.add_argument("--mc-foot-ik-iters", type=int, default=None,
                   help="leg-IK Adam steps when --mc-foot-ik is on (default: MC's 150). It is a "
                        "convergence threshold, not a dial: 40 costs 1376 ms and removes 8%% of "
                        "the ankle slide, 150 costs 4887 ms and removes 95%%")
    d.add_argument("--mc-umi-dir", default=None,
                   help="UMI_collection checkout holding motion_completion/ (default: this repo's)")
    d.add_argument("--no-zero-arms", dest="zero_arms", action="store_false",
                   help="anchor at the raw crouch default_qpos instead of the arms-zeroed ready "
                        "pose. Only correct against HOLOSOMA_INIT_READY_POSE=0.")
    d.set_defaults(zero_arms=True)
    d.add_argument("--yam-urdf", default=os.path.join(
        _FUJI_ROOT, "third_party", "UMI_collection", "robot_assets", "g1_yam",
        "g1_29dof_yam.urdf"), help="URDF for the drawn robot + the FK cross-check")

    f = p.add_argument_group("force + gripper")
    # Both of these default ON. The page's switches are still the live control — this only says
    # what they READ at startup, so "enabled at launch, toggled from the panel" is one state and
    # not two. Force still cannot reach the robot unless the low level was given
    # FORCE_CMD_WIRE=1, and it is hard-zeroed below --force-gate-n either way, so the default
    # arms the planner's half of a two-sided switch rather than commanding force outright.
    f.add_argument("--enable-force-cmd", dest="enable_force_cmd", action="store_true",
                   default=True,
                   help="start with the page's force switch ON. THIS IS THE DEFAULT; the flag is "
                        "kept so existing commands and docs stay valid. The low level ALSO needs "
                        "FORCE_CMD_WIRE=1 — neither side alone commands force.")
    f.add_argument("--no-force-cmd", dest="enable_force_cmd", action="store_false",
                   help="start with the force switch OFF (pure position until the page's switch "
                        "is turned on)")
    f.add_argument("--force-gate-n", type=float, default=1.5,
                   help="below this |F| the command is hard-zeroed (still pure position)")
    f.add_argument("--command-gripper", dest="command_gripper", action="store_true",
                   default=True,
                   help="autonomously drive BOTH jaws from the predicted widths. THIS IS THE "
                        "DEFAULT; the flag is kept so existing commands stay valid.")
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


def _resolve_extrinsic(checkpoint: str):
    """``(eef_t_wrist, eef_R_wrist)`` — the dataset's stored-TCP -> wrist_yaw_link leg.

    Read from the checkpoint's OWN training zarr, never assumed: it is the same
    ``T_tracker_tcp`` the frame seam comes from, and getting it wrong is worth 33 cm (see
    ``hl_serving._install_frame_seam``, which refuses the pre-2026-08-06 convention outright —
    that refusal runs too, on the same file).
    """
    import zarr

    path = HL._checkpoint_zarr(checkpoint)
    if path is None:
        raise SystemExit(
            "the checkpoint declares no training zarr, so the stored-TCP -> wrist leg cannot "
            "be derived and the motion completion would be conditioned on a guessed frame.")
    attrs = dict(zarr.open(str(path), "r").attrs)
    extrinsic, note = read_stored_extrinsic(attrs)
    if extrinsic is None:
        raise SystemExit(f"{path}: no usable T_tracker_tcp ({note})")
    t_wrist, R_wrist = wrist_pose_from_stored(extrinsic)
    _log(f"stored-TCP -> wrist leg from {os.path.basename(str(path))}: "
         f"t={np.round(t_wrist, 4).tolist()} ({note or 'ok'})")
    return t_wrist, R_wrist


def _measured_five_point(args, anchor, readout):
    """``rframe -> (pos, orn) | None``: the ROBOT's own five keypoints, from RecordTap.

    Only built when a carried command anchor needs a leak reference (see
    :class:`hl_dense.KeypointClutch`). The FK is ``ready_pose.KeypointFK`` — the SAME one the
    ready pose itself is built from, lifted by the SAME ``anchor.pelvis_z`` and relativised by
    the SAME readout, so feeding it the anchor's own joints reproduces the ready frame
    bit-for-bit instead of approximately. A second FK here would be the one place a lever could
    disagree and turn the leak bound into a constant offset.

    Returns ``None`` (⇒ the clutch FREEZES rather than integrating blind) when there is no
    readback, when the FK is unavailable, or on any bad frame.
    """
    if args.post_anchor == "off":
        return None
    fk, why = RP.KeypointFK.load(urdf_path=args.yam_urdf)
    if fk is None:
        raise SystemExit(
            f"--post-anchor command on the 5pt wire needs the keypoint FK, which is the only "
            f"reference the carried command can be bounded against: {why}")
    lift = float(anchor.pelvis_z)

    def measured(rframe):
        if rframe is None or rframe.dof_pos is None:
            return None
        return readout.from_bodies(*fk.bodies(rframe.dof_pos, lift_z=lift))

    # Prove the seam on the way up, not on the rig: the anchor's own joints must FK back onto
    # the ready frame the wire idles at. A disagreement here is a lever/frame error that would
    # otherwise present as a permanent bias in the leak bound.
    check_pos, _ = readout.from_bodies(*fk.bodies(anchor.dof, lift_z=lift))
    gap = float(hl_dense.keypoint_gaps_m(check_pos, hl_dense.ready_pose_five_point(
        anchor, readout)[0]).max())
    if gap > 1e-6:
        raise SystemExit(
            f"the measured-keypoint FK disagrees with the ready-pose frame by {gap * 1e3:.3f} mm "
            "on the anchor's OWN joints. They must be the same FK; refusing to carry a command "
            "anchor whose leak reference is biased.")
    _log(f"5pt leak reference: RecordTap joints -> KeypointFK (ready-pose round trip "
         f"{gap * 1e6:.3f} um)")
    return measured


def _build_wire(args, anchor, completer):
    """The ``:8770`` wire for ``--low-level``, plus a log line explaining what it publishes.

    Both branches are seeded from the SAME ready pose — the pose the low level holds after
    **A** and the pose the chunk is decoded against — because both families' pre-arm gates
    compare the broadcast wire against that pose, one in radians and one in metres.
    """
    post = hl_dense.PostAnchorConfig(
        mode=args.post_anchor,
        eps_joint_rad=float(args.post_anchor_eps_rad),
        eps_m=float(args.post_anchor_eps_m),
        eps_rad=float(np.radians(args.post_anchor_eps_deg)),
        contact_n=float(args.post_anchor_contact_n),
    )
    _log(f"post-completion anchor: {post.describe()}")
    if args.low_level == "dense":
        ready_mc = hl_dense.ready_pose_joint_targets(
            anchor.dof, G1_29DOF_SDK_DOF_NAMES, completer.joint_names)
        clutch = hl_dense.JointClutch(ready_mc, mode=args.dense_clutch, post=post)
        wire = hl_dense.DenseWire(completer.joint_names, G1_29DOF_SDK_DOF_NAMES,
                                  clutch, ready_mc)
        if post.carrying and args.dense_clutch != "joint":
            raise SystemExit(
                "--post-anchor command needs --dense-clutch joint: the carried command IS the "
                "clutch base, and with --dense-clutch none nothing is published relative to it, "
                "so the mode would silently do nothing.")
        if args.dense_clutch == "none":
            _log("⚠️  --dense-clutch none: the completion's ABSOLUTE joints go on the wire. "
                 "Measured 0.40 rad (23 deg) of arm null-space offset from the ready pose at "
                 "step 0 against a 0.20 rad pre-arm gate — raise HOLOSOMA_VIVE_ARM_TOL_RAD or "
                 "expect a refusal, and expect the arm to take that offset up on the first "
                 "executed step.")
        return wire

    # 5pt. The readout frame is the ANCHOR's pelvis, never mc_frames.PELVIS_STAND_Z (0.79186):
    # that constant belongs to the completion's `cond_mode="home"` placement, and borrowing it
    # against an `absolute` placement would put every keypoint 35 mm low. See KeypointReadout.
    readout = hl_dense.KeypointReadout(
        pelvis_pos=(0.0, 0.0, float(anchor.pelvis_z)), pelvis_rotmat=None)
    ready_pos, ready_orn = hl_dense.ready_pose_five_point(anchor, readout)
    clutch = hl_dense.KeypointClutch(ready_pos, ready_orn, post=post)
    wire = hl_dense.FivePointWire(readout, ready_pos, ready_orn, clutch=clutch,
                                  measured_frame=_measured_five_point(args, anchor, readout),
                                  upper_source=args.upper_source,
                                  foot_source=args.foot_source)
    if args.dense_clutch != "joint":
        _log(f"note: --dense-clutch {args.dense_clutch} is inert with --low-level 5pt.")
    _log(f"5pt readout: {readout.describe()}")
    _log(f"5pt sources: upper 3 points from the {args.upper_source.upper()}, feet from "
         f"{args.foot_source.upper()}")
    if args.upper_source == "policy":
        _log("  the completion still runs (the feet exist nowhere else) but no longer sits "
             "between the policy and the three GATED points; the two agree to 0.84 mm.")
    if args.foot_source == "hold":
        _log("  ⚠️  feet HELD at the ready pose for every step. The low level does not gate "
             "them (300 mm warn only) and its arm clutch re-bases them to the stance the "
             "robot already holds — but it will NOT follow the operator's yaw (10.7 deg "
             "median / 48.5 p90 on 0902_merged3), and the balance interaction is unmeasured "
             "on hardware. Gantry only.")
    for i, label in enumerate(hl_dense.FIVE_POINT_LABELS):
        _log(f"  ready-pose {label:<11} {np.round(ready_pos[3 * i:3 * i + 3], 4).tolist()}"
             + ("  [gated]" if i < hl_dense.FIVE_POINT_GATED else "  [derived — warned only]"))
    return wire


def main(argv=None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    cams = _camera_urls(args)
    # The preset spelling resolves to its algorithm block here, once, so everything downstream
    # (the loader, the log lines, the refusals) sees one canonical name.
    if args.algo in ALGO_ALIASES:
        _log(f"--algo {args.algo} -> algorithm block {ALGO_ALIASES[args.algo]!r} "
             "(the 3-tracker part is a layout fact and is checked below, by name)")
        args.algo = ALGO_ALIASES[args.algo]
    HL._limit_cpu_thread_pools(args.device)

    # --- 1. the model -----------------------------------------------------------------
    _log(f"loading {args.algo} checkpoint {args.checkpoint} on {args.device} …")
    loaded = load_hl_model(args.checkpoint, args.algo, device=args.device,
                           num_steps=args.num_steps, max_cameras=len(cams))
    if tuple(loaded.layout.roles) != REQUIRED_ROLES:
        raise SystemExit(
            f"checkpoint layout {loaded.layout.mode!r} has roles {loaded.layout.roles}; motion "
            f"completion conditions on {REQUIRED_ROLES} — both hands AND the chest, which is "
            "its root channel. A left_only / both_hands checkpoint has no chest to condition "
            "the root on; serve it with scripts/hl_serving.py instead.")
    if len(cams) != loaded.num_cameras:
        # load_hl_model already refuses "too few"; this catches the other direction, where a
        # bridge is running that the model was not trained on and would silently go unread.
        raise SystemExit(
            f"{len(cams)} camera bridge(s) configured but the checkpoint reads "
            f"{loaded.num_cameras} ({list(loaded.image_keys)}). Pass --camera-right-ws '' for a "
            "1-camera model, or the second URL for a 2-camera one.")
    consumes_force = bool(loaded.layout.force_roles)
    _log(f"ready: algo={loaded.algorithm} mode={loaded.layout.mode} roles={loaded.layout.roles} "
         f"To={loaded.obs_history_len} FH={loaded.force_hist_len} "
         f"cameras={list(loaded.image_keys)} force_roles={loaded.layout.force_roles} "
         f"gripper_roles={loaded.layout.gripper_roles}")
    # Printed, not only shown on the page: "is force actually on?" is the first thing an
    # operator asks and the page's checkbox is three clicks away from the terminal they launched
    # in. Both halves are named because neither alone commands force.
    _log(f"force command: {'ENABLED' if args.enable_force_cmd else 'off'} at launch "
         f"(gate {args.force_gate_n:g} N — below that |F^cmd| is hard-zeroed and the motion is "
         f"pure position). The low level must ALSO have FORCE_CMD_WIRE=1. "
         f"Toggle live on the page; --no-force-cmd starts it off.")
    _log(f"both jaws: {'COMMANDED' if args.command_gripper else 'READ-ONLY'} at launch "
         f"(--no-command-gripper starts them read-only; they come up OPEN either way)")
    if args.enable_force_cmd and not loaded.layout.force_roles:
        raise SystemExit(
            "force command is enabled but this checkpoint predicts no force for any role, so "
            "there would be nothing to put on the wire. Pass --no-force-cmd, or serve a "
            "checkpoint whose active algorithm block declares force.")
    if abs(args.step_hz - 1.0 / ACTION_DT) > 0.5:
        raise SystemExit(
            f"--step-hz {args.step_hz} is not the completion's {1.0 / ACTION_DT:.0f} Hz grid. "
            "The completion is a 30 Hz checkpoint and its joints are published one per step, "
            "so any other rate plays the whole body back at the wrong speed.")
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

    # --- 2. the frame seam + the stored-TCP leg (both from the checkpoint's own zarr) ---
    # _install_frame_seam is the REFUSAL for a pre-2026-08-06 dataset; it is run for that
    # refusal and for its log line. The seam itself is a LEFT-only URDF<->model conversion in
    # the other planner and is not applied here — the completion consumes stored-TCP poses
    # directly, through the very extrinsic the seam is derived from.
    HL._install_frame_seam(argparse.Namespace(replay_zarr=None, checkpoint=args.checkpoint),
                          False)
    eef_t_wrist, eef_R_wrist = _resolve_extrinsic(args.checkpoint)

    # --- 3. the deploy anchor: where the chunk is decoded, i.e. the pose held after `A` ---
    anchor, why = RP.resolve_deploy_anchor(
        loaded.layout.roles, zero_arms=args.zero_arms)
    if anchor is None:
        raise SystemExit(
            f"cannot resolve the G1 ready pose, which IS the decode anchor here: {why}")
    _log(f"decode anchor: {anchor.describe()}")
    for note in anchor.notes:
        _log(f"  ⚠️  {note}")
    if not args.zero_arms:
        _log("⚠️  --no-zero-arms: only correct against HOLOSOMA_INIT_READY_POSE=0.")

    # --- 4. motion completion -------------------------------------------------------
    if args.mc_foot_ik != "none":
        # Not refused — an operator may want it for a static-posture check, and the page can turn
        # it on later anyway — but it cannot be quiet. Note what it does NOT do: Predict and the
        # drip are serialised by one mutex, so the drip never runs off the end of a chunk. The
        # cost lands BETWEEN chunks, as a pause with the wire heartbeating the last frame.
        print("⚠️  --mc-foot-ik is ON (the page's 'Foot IK / anti-slide' checkbox is the live "
              "control from here). Measured on this checkpoint: 1376 ms/chunk at 40 iters, "
              "4887 ms at 150, against a 481 ms re-plan budget — so every Predict takes that "
              "long and the body PAUSES between chunks instead of flowing through them. "
              + ("On this DENSE wire it also rewrites the LEGS, and the clutch publishes them: "
                 "worst per-step leg dq 0.0061 -> 0.0493 rad = 59% of the low level's slew "
                 "budget, on the legs holding the robot up."
                 if args.low_level == "dense" else
                 "On this 5pt wire it moves only the two FOOT keypoints, which pass through "
                 "absolutely and are not gated (warned at 300 mm)."), flush=True)
    completer = hl_dense.ChunkCompleter(
        umi_dir=args.mc_umi_dir, wrist_ik=not args.mc_no_wrist_ik,
        foot_fix=args.mc_foot_ik, foot_ik_iters=args.mc_foot_ik_iters,
        log=lambda m: print(f"[hl-mc][mc] {m}", flush=True))
    if completer.disabled_reason:
        raise SystemExit(
            f"motion completion is unavailable: {completer.disabled_reason}\n"
            "This planner cannot run without it — a dense low level fed no wire holds its ready "
            "pose forever, which looks exactly like a hung robot. Fix the prerequisite (see "
            "third_party/UMI_collection/motion_completion/README.md) or serve a 3pt low level "
            "with scripts/hl_serving.py.")
    _log("starting the motion-completion backend (~30-60 s: the ckpt + the JAX wrist-IK JIT) …")
    if not completer.start():
        raise SystemExit(f"motion completion failed to start: {completer.disabled_reason}")

    # --- 4b. the wire (the ONE thing --low-level decides) ---------------------------
    wire = _build_wire(args, anchor, completer)
    _log(f"wire: mode={wire.mode!r} · the low level MUST be MODE={wire.low_level_mode} · "
         f"its pre-arm gate is {wire.gate_note}")

    # --- 5. the planner + its anchor -------------------------------------------------
    planner = LivePlanner(loaded, obs_hz=args.obs_hz)
    planner.obs_max_age_s = (OBS_MAX_AGE_S if not args.obs_max_age or float(args.obs_max_age) <= 0
                             else float(args.obs_max_age))
    planner.grip_max_age_s = max(planner.obs_max_age_s, HL.GRIP_MAX_AGE_S)
    planner.seed_anchor(
        np.stack([anchor.pos[r] for r in loaded.layout.roles]),
        np.stack([rotmat_to_rot6d(anchor.rotmat[r]) for r in loaded.layout.roles]))

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
        return _run_viser(args, loaded, planner, completer, wire, subs, s0, anchor,
                          eef_t_wrist, eef_R_wrist, consumes_force, report)
    finally:
        completer.stop()
        s0.stop()
        for sub in subs.values():
            sub.stop()


def _self_check(subs: dict, cams: list[str], loaded, *, require_6dof: bool,
                consumes_force: bool) -> dict:
    """Probe every bridge this deploy actually reads, and name the missing one.

    Deliberately NOT ``hl_serving._self_check``: that one knows about one camera and one jaw,
    and its failure messages quote ``:8773``/``:8784`` by port. A dual rig that forgot camera R
    would pass it and then feed the second encoder group a grey frame for the whole session.
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
        cmsg = _wait(subs[f"camera{i}"], need_key="jpeg_b64")
        if cmsg is None:
            raise SystemExit(
                f"self-check: no camera frame for {loaded.image_keys[i]} on {url}. This "
                f"checkpoint reads {loaded.num_cameras} view(s); a missing one is fed to its "
                "encoder group as a grey image and nothing downstream would say so.")
        report[f"camera{i} ({loaded.image_keys[i]})"] = f"ok ({url})"
    for role in loaded.layout.gripper_roles:
        gmsg = _wait(subs[f"grip_{role}"], need_key="actual")
        report[f"gripper {role}"] = (
            "ok" if gmsg is not None else
            "NO FRAME — the jaw width reads as UNMEASURED (state_valid=False), which is what "
            "the training converter does for a dead encoder; the grasp channel is degraded")
    # PARSED, not merely received. `parse_record_frame` is what the clutch and the live mesh
    # actually consume, and it rejects a frame whose `type` is not "record" or whose `dof_pos`
    # is the wrong length — so a weaker "did any JSON arrive" check here would report `ok` for a
    # tap the clutch cannot use, and the operator would only find out from the live status line
    # contradicting the bring-up report. (Observed while driving the page against a mock tap.)
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
            "falls back to the completion and the joint clutch runs OPEN-LOOP")
    else:
        report["RecordTap"] = (
            "ABSENT — relaunch the low level with RECORD=1. The mesh falls back to the "
            "completion and the joint clutch runs OPEN-LOOP from the ready pose, so a tracking "
            "shortfall accumulates across chunks.")
    return report


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------
def _run_viser(args, loaded, planner, completer, wire, subs, s0, anchor,
               eef_t_wrist, eef_R_wrist, consumes_force, report) -> int:
    import viser
    from viser.extras import ViserUrdf

    import mc_frames as F  # on the path via ChunkCompleter's module load

    roles = tuple(loaded.layout.roles)
    grip_roles = tuple(loaded.layout.gripper_roles)
    force_roles = tuple(loaded.layout.force_roles)

    stop = threading.Event()
    dripping = threading.Event()
    busy = threading.Lock()
    grip_enabled = threading.Event()
    if args.command_gripper:
        grip_enabled.set()

    wire_lock = threading.Lock()
    # The published wire, always a full frame of whatever family `wire` is. Seeded at the READY
    # POSE so the very first heartbeat gives the low level's pre-arm gate something to close
    # against — joint-space for dense, keypoint-space for 5pt, the same pose either way.
    last_wire = {"frame": wire.idle(), "fcmd": None}
    force_lock = threading.Lock()
    force_ctl = {"enabled": bool(args.enable_force_cmd), "gate_n": float(args.force_gate_n)}
    # Snapped again (main() already did) so THIS function owns the invariant "the dropdown's
    # initial_value is one of its options" no matter who calls it.
    exec_ctl = {"n": snap_exec_steps(args.exec_steps, _EXEC_CHOICES)}
    # FOOT IK, live from the page — the causal leg-IK anti-slide pass inside the completion.
    #
    # It is a COMPLETION post-process, i.e. upstream of the wire split, so one switch serves both
    # families — but what it MEANS differs and the panel says so: on `dense` the rewritten legs
    # become published joint targets (measured worst per-step leg |dq| 0.0061 -> 0.0493 rad, 59%
    # of the low level's 0.0833 budget, on the legs holding the robot up); on `5pt` only the two
    # FOOT keypoints move, and those pass through absolutely and are not gated (warned at 300 mm).
    #
    # `applied_*` is what the completer is CURRENTLY configured with. The gap between the two is
    # closed at the top of `_predict` and NOWHERE ELSE: `ChunkCompleter.set_post_process` is not
    # lock-protected, while `complete()` holds the completer's lock for the whole chunk (1.4-4.9 s
    # with this on), so applying it from the viser callback would either race the settings or
    # freeze the page for seconds. The predicting thread is the single writer, serialised by
    # `busy`, and each chunk is its own backend episode — so a flip always lands on a chunk
    # BOUNDARY and can never put two different bodies in one motion.
    foot_lock = threading.Lock()
    foot_ctl = {"fix": str(args.mc_foot_ik), "iters": args.mc_foot_ik_iters,
                "applied_fix": str(args.mc_foot_ik), "applied_iters": args.mc_foot_ik_iters}
    # SEQUENTIAL ROLLOUT state ("▶▶ Run to completion"), the same shape as the left_only
    # planner's: `gen` is bumped by Stop so an in-flight loop retires itself instead of being
    # killed mid-chunk, and `max_s` bounds the rollout in WALL-CLOCK seconds because a policy —
    # unlike a recorded episode — has no end of its own to reach.
    auto_lock = threading.Lock()
    auto_ctl = {"running": False, "gen": 0, "max_s": float(args.auto_max_seconds)}
    auto_running = threading.Event()   # a sequential rollout is executing right now
    staged: dict = {"out": None}
    stats = hl_dense.DenseStats()
    seed_ctl = {"n": 0}
    calib = HL.GripperCalib()
    grip_lock = threading.Lock()
    # ONE state machine PER HAND, built ONCE, over ITS OWN control dict. Both halves of that are
    # load-bearing and the second was a live bug: `build_grip_command`'s state does NOT live in
    # the closure, it lives in the dict it is handed (`engaged`, `run`, `run_closing`,
    # `held_frac`), so a single shared dict would let the LEFT jaw's confirm counter engage the
    # RIGHT jaw's overshoot and let either hand's tightest-width latch pin the other's target.
    # `HL.new_grip_ctl` is the constructor next to the consumer — the keys/units are its
    # business, not this call site's (a hand-written dict here shipped `overshoot_mm` in mm
    # against a reader that wants `overshoot_m` in metres, and every Run died with
    # `KeyError('overshoot_m')` on the first jaw command).
    grip_ctl = {role: HL.new_grip_ctl(overshoot_mm=args.grip_overshoot_mm,
                                      close_below_mm=args.grip_close_below_mm,
                                      confirm_steps=args.grip_confirm_steps)
                for role in grip_roles}
    grip_cmd = {role: HL.build_grip_command(calib, grip_ctl[role], grip_lock)
                for role in grip_roles}

    # ---------------- readbacks the wire consumes ----------------
    # ⚠️ DEFINED BEFORE THE HEARTBEAT THREAD STARTS, not merely before its first use. The
    # heartbeat is spawned inside this function and calls `_idle_sync_wire` -> `_record_frame` on
    # its very first tick; a `def` further down the body has not run yet when that happens, so
    # the closure cell is still unbound and the tick died with
    # `NameError: cannot access free variable '_record_frame'`. Recoverable (the heartbeat
    # catches and retries, and the def lands microseconds later) but it means the first wire
    # frames were published from an un-synced base, and it is the kind of ordering that looks
    # fine on every read.
    def _record_frame():
        msg, _wall, _n = subs["record"].latest()
        try:
            return parse_record_frame(msg)
        except Exception:  # noqa: BLE001 - a malformed frame is viz/clutch input only
            return None

    def _measured_force_n() -> float | None:
        """Worst |F| across the hands the rig actually reads, or None if unknown.

        The post-completion anchor's contact cut-out consumes this. **None is not zero** — the
        clutch treats an unknown reading as "do not disable the guard", the same contract
        ``hl_anchor.ChunkAnchor.resolve`` states, so a dead Bota bridge cannot quietly turn a
        press into an advancing command.
        """
        if not force_roles:
            return None
        fmsg, _wall, _n = subs["force"].latest()
        # Always (n_roles, 6) — `_pick_force_channels` zero-pads a 3-DoF reading to six rather
        # than returning a narrower row, so the force block is columns [0:3] in every case.
        bins = HL._pick_force_channels(fmsg, force_roles, not args.allow_3dof_force)
        if bins is None:
            return None
        arr = np.asarray(bins, dtype=np.float64).reshape(-1, 6)
        if not np.all(np.isfinite(arr)):
            return None
        return float(np.linalg.norm(arr[:, :3], axis=1).max())

    def _gate_force(wrench_rows) -> np.ndarray | None:
        """Per-hand predicted wrenches -> the (3*n_hands,) sensor-frame F^cmd, or None.

        Same two-level contract as the left_only planner and the same sign flip in the same
        one place: the model predicts ``F_meas`` (what the Bota READ during the demo) and the
        low level closes ``e = F^cmd + F_meas -> 0``, so the command that reproduces the
        demonstrated press is ``-F_meas``. Gated on the MAGNITUDE, which the flip preserves.

        BOTH hands or NEITHER: ``ViveLiveSource`` reads a (6,) f_cmd as ``[L, R]``, so a
        partial publish would put the LEFT command on the wire and silently leave the RIGHT at
        whatever the low level last had. The gate is therefore evaluated per hand and a
        below-gate hand is published as an explicit ZERO, which is what "pure position for
        that hand" means on this wire.
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
        """Publish ONE frame of this wire's family. False (and nothing sent) if unsafe.

        The gate lives in the wire object and is applied here and only here, at the last moment
        before the socket: the low level validates only WIDTH and ``json.dumps`` serializes NaN
        unchecked, so a bad joint target or a de-normalised quaternion would reach the robot. On
        a refusal we send NOTHING and let the freshness gate (S0Driver 0.5 s -> the low level's
        0.25 s) hold the robot, which is the same policy ``fuji.client.is_safe_to_send``
        implements for the 3-point wire.
        """
        ok, reason = wire.publish(s0, frame, fcmd)
        if not ok:
            _log(f"REFUSED a {wire.mode} frame: {reason} — sending nothing; the stale gate holds.")
        return ok

    def _idle_sync_wire() -> None:
        """DENSE ONLY: while UN-ARMED, track the robot's MEASURED joints; once ARMED, freeze.

        Both halves are load-bearing, and both are ported from the left_only planner where each
        was a live-deploy bug:

        * TRACK WHILE UN-ARMED. The joint-space pre-arm gate compares the broadcast wire against
          the joints the robot is actually holding, so a wire pinned to a FROZEN ready pose
          refuses to arm the moment the two differ — and raising the tolerance instead is worse,
          because arming then FLIES the body to that stale target. (The left_only twin of this
          was a whole rig session lost to `per-point gap [0.31,0.31,0.0] m -> REFUSE`.)
        * FREEZE ONCE ARMED. An armed low level applies the broadcast wire as its target every
          tick, so continuing to chase the measured pose would erase the servo's own holding
          error — the arm DROOPS, one heartbeat at a time.

        A missing / unparseable RecordTap holds the last wire, which is the honest degradation:
        without a readback there is nothing to track toward. ``FivePointWire.measured`` returns
        None unconditionally, so on that family this is a no-op by construction rather than by a
        branch here — its gate reference is the frozen reference clip, not the encoders, so
        chasing measured would move the wire AWAY from what the gate wants.
        """
        rframe = _record_frame()
        if rframe is None or rframe.vive_armed:
            return
        frame = wire.measured(rframe)
        if frame is not None:
            _set_wire(frame, None)

    def _heartbeat() -> None:
        period = 1.0 / max(args.drip_hz, 1.0)
        clutch_period = 1.0 / 10.0     # the FK/remap is cheap, but 60 Hz of it is pointless
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
                        fc = (None if last_wire["fcmd"] is None
                              else last_wire["fcmd"].copy())
                    # The switch is authoritative on EVERY tick, so turning it off (or Stop)
                    # makes the wire pure position within one heartbeat — the heartbeat can
                    # never re-publish a stale force command.
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
                planner.push_proprio_from_anchor()
            except Exception as exc:  # noqa: BLE001 - ingest must survive a bad frame
                _log(f"ingest error: {exc!r}")
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    threading.Thread(target=_ingest_loop, daemon=True, name="ingest").start()

    # ---------------- the scene ----------------
    server = viser.ViserServer(host=args.host, port=args.port,
                              label="FUJI 3-tracker PLANNER (motion completion -> dense)")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", plane="xy", shadow_opacity=0.0, infinite_grid=True,
                          cell_size=0.25)
    # ONE WORLD, floor at z=0. That is the frame ``resolve_deploy_anchor`` expresses its poses
    # in (origin under the pelvis, ``pelvis_z`` above the floor) and the frame the completion
    # returns its root in, so the grid, the decoded paths, the live robot and the violet ghost
    # register by construction with no per-scene offset to maintain.
    #
    # The URDF's root LINK is ``pelvis``, so the mesh's own frame has to carry that height:
    # /robot sits at (0, 0, pelvis_z) and the ghost is re-positioned per frame from the
    # completion's own root, which is what lets the ghost take a step the live mesh cannot.
    robot_root = server.scene.add_frame(
        "/robot", show_axes=False, position=(0.0, 0.0, float(anchor.pelvis_z)))
    urdf = ViserUrdf(server, urdf_or_path=Path(args.yam_urdf), root_node_name="/robot")
    urdf_joints = list(urdf.get_actuated_joint_names())
    dof_remap = build_sdk_to_urdf_index(urdf_joints)
    if dof_remap.missing_sdf:
        _log(f"⚠️  URDF is missing SDK joints {dof_remap.missing_sdf} — the live mesh is "
             "disabled rather than drawn with a limb in the wrong place")
    urdf.update_cfg(remap_dof_to_urdf(anchor.dof, dof_remap))
    del robot_root
    ghost_root = server.scene.add_frame("/ghost", show_axes=False, visible=True,
                                        position=(0.0, 0.0, float(anchor.pelvis_z)))
    ghost = ViserUrdf(server, urdf_or_path=Path(args.yam_urdf), root_node_name="/ghost",
                      mesh_color_override=_GHOST_RGBA)

    def _reset_ghost() -> None:
        """Put the violet body back at the ready pose — **root included**.

        ⚠️ A completion's pose is 29 JOINTS *plus a floating ROOT*, and playback writes all
        three (``update_cfg``, ``ghost_root.position``, ``ghost_root.wxyz``) because the ghost
        has to be able to take a step the live mesh cannot. Restoring only the joints — which is
        what Reset used to do — leaves the violet body standing wherever the last take's
        completion ended, translated and YAWED, so the scene never looks neutral again and the
        operator cannot tell the ready pose from the leftover of the previous experiment. The
        root is display-only (it cannot reach the wire), which is exactly why nothing else
        catches it.

        Called at construction too, so "where the ghost starts" and "where Reset puts it" are
        one piece of code rather than two that can drift.
        """
        ghost.update_cfg(remap_dof_to_urdf(anchor.dof, dof_remap))
        ghost_root.position = (0.0, 0.0, float(anchor.pelvis_z))
        ghost_root.wxyz = (1.0, 0.0, 0.0, 0.0)

    _reset_ghost()

    paths: dict = {}
    #: Read by ``_draw_paths`` at draw time. A plain dict rather than a closure over the GUI
    #: handle, because the checkbox is created ~150 lines BELOW this and the page loop is not
    #: the only caller (see the ``_record_frame`` ordering bug: a `def` above its own free
    #: variable is fine until a thread reaches it first).
    view_ctl: dict = {"triads": True}

    def _draw_paths(out) -> None:
        """The three decoded EE paths, at the points the completion was conditioned on.

        ⚠️ **THE HEAD IS NOT A THIRD HAND, and a polyline cannot show it.** The chest keypoint
        sits at ``(-0.004, 0, 1.151)`` in this scene — INSIDE the torso mesh — so its line is
        depth-occluded by both the live robot and the violet ghost, and it translates ~2-3 cm
        over a chunk where the hands travel tens of cm. Drawn as a line only, it reads as "the
        head was not predicted at all". Yet it is the one path most worth reading: the chest is
        motion completion's ROOT conditioning, so its pose decides the POSTURE the whole body
        answers with, and its step-0 gap is the term that eats half the 5pt gate margin.

        So every role gets three things, sized per role by :data:`_PLAN_STYLE`:

        * the polyline (as before),
        * a point per predicted step, so a near-stationary role reads as a CLUSTER rather than
          as nothing, and
        * an orientation triad every ``stride`` steps — the rotation IS the head channel's
          content, and its axes are long enough to PROTRUDE from the mesh that surrounds the
          point.
        """
        want_triads = bool(view_ctl["triads"])
        for role in roles:
            st = _PLAN_STYLE.get(role, _PLAN_STYLE_FALLBACK)
            rgb = np.array(st["rgb"], np.uint8)
            pts = np.asarray(out["decoded"][role]["position"], dtype=np.float32)
            # REMOVE BEFORE ADDING, always. viser addresses nodes by NAME, so adding
            # `/plan/left/line` while the previous handle still holds that name replaces the
            # node — and the later `.remove()` on the stale handle would then delete the one
            # just drawn, leaving a role that vanishes on its SECOND chunk.
            for kind in ("line", "dots", "triads"):
                old = paths.pop(f"{role}:{kind}", None)
                if old is not None:
                    try:
                        old.remove()
                    except Exception:  # noqa: BLE001
                        pass
            seg = np.stack([pts[:-1], pts[1:]], axis=1)
            colour = np.broadcast_to(rgb, (seg.shape[0], 2, 3)).copy()
            paths[f"{role}:line"] = server.scene.add_line_segments(
                f"/plan/{role}/line", points=seg, colors=colour,
                line_width=4.0 if role == "head" else 3.0)
            paths[f"{role}:dots"] = server.scene.add_point_cloud(
                f"/plan/{role}/dots", points=pts,
                colors=np.broadcast_to(rgb, pts.shape).copy(),
                # float32, not the default float16: at z≈1.15 m a half float quantises to about
                # a millimetre, which is a visible stagger across a 26 mm chest excursion.
                point_size=0.006, point_shape="circle", precision="float32")
            idx = _triad_indices(len(pts), int(st["stride"]))
            rotmat = rot6d_to_rotmat(np.asarray(out["decoded"][role]["rot6d"]))[idx]
            paths[f"{role}:triads"] = server.scene.add_batched_axes(
                f"/plan/{role}/triads",
                batched_wxyzs=quat_xyzw_to_wxyz(rotmat_to_quat_xyzw(rotmat)).astype(np.float32),
                batched_positions=pts[idx],
                axes_length=float(st["axes"]), axes_radius=0.004, visible=want_triads)

    def _set_triads_visible(show: bool) -> None:
        view_ctl["triads"] = bool(show)
        for key, handle in paths.items():
            if key.endswith(":triads"):
                try:
                    handle.visible = bool(show)
                except Exception:  # noqa: BLE001 - a stale handle must not kill the callback
                    pass

    def _clear_paths() -> None:
        for key in list(paths):
            handle = paths.pop(key)
            try:
                handle.remove()
            except Exception:  # noqa: BLE001
                pass

    def _apply_foot_ik() -> None:
        """Push a pending foot-IK change into the completer, if the page changed it.

        Called from ``_predict`` only. Idempotent and cheap when nothing changed, which is the
        common case — the comparison is against ``applied_*`` rather than against the completer's
        privates so this file never reaches into ``ChunkCompleter``.
        """
        with foot_lock:
            fix, iters = str(foot_ctl["fix"]), foot_ctl["iters"]
            if fix == foot_ctl["applied_fix"] and iters == foot_ctl["applied_iters"]:
                return
            foot_ctl["applied_fix"], foot_ctl["applied_iters"] = fix, iters
        # Both are passed EVERY time: `set_post_process` treats a missing `foot_ik_iters` as
        # "clear the override", so naming only `foot_fix` would silently reset the iteration
        # count to MC's 150 — a 3.5x cost change nobody asked for.
        completer.set_post_process(foot_fix=fix, foot_ik_iters=iters)
        if fix == "none":
            _log("foot IK OFF — the completion's legs are its own output again (~80 ms/chunk).")
        else:
            _log(f"⚠️  foot IK ON ({iters if iters is not None else 150} leg-IK iters). Measured "
                 "on this checkpoint: 1376 ms/chunk at 40 iters, 4887 ms at 150, against a "
                 "481 ms re-plan budget — every Predict now takes that long, so the body PAUSES "
                 "between chunks (the wire heartbeats the last frame; it does not go limp). It "
                 + ("also REWRITES THE LEGS, and on this dense wire a leg rewrite is a leg "
                    "COMMAND: worst per-step leg |dq| 0.0061 -> 0.0493 rad, 59% of the low "
                    "level's 0.0833 budget." if wire.mode == "dense" else
                    "moves the two FOOT keypoints, which pass through absolutely and are not "
                    "gated (warned at 300 mm) — the 3 gated points are unaffected."))

    # ---------------- predict ----------------
    def _predict() -> str:
        """Sample a chunk, decode it against the running anchor, complete it, stage it."""
        reasons, _ = HL._obs_freshness(planner, max_age_s=planner.obs_max_age_s,
                                       grip_max_age_s=planner.grip_max_age_s,
                                       consumes_force=consumes_force)
        if reasons and args.obs_max_age is not None and float(args.obs_max_age) != 0:
            return "REFUSED: " + "; ".join(reasons)
        if reasons:
            _log("⚠️  stale observation: " + "; ".join(reasons))
        # THE ONE PLACE the page's foot-IK switch reaches the completer (see `foot_ctl`): on the
        # predicting thread, before this chunk's episode is opened, so a flip lands on a chunk
        # BOUNDARY and this chunk is completed by exactly one configuration end to end.
        _apply_foot_ik()
        seed_ctl["n"] += 1
        with force_lock:
            enabled, gate_n = force_ctl["enabled"], force_ctl["gate_n"]
        t0 = time.monotonic()
        chunk = planner.infer_chunk(seed=seed_ctl["n"], force_enabled=enabled, gate_n=gate_n)
        t_sample = (time.monotonic() - t0) * 1e3
        decoded = {rc.role: {"position": rc.position, "rot6d": rc.rot6d,
                             "gripper_m": rc.gripper_m, "wrench": rc.wrench_pred}
                   for rc in chunk.roles}
        cond = completer.build_cond(
            left_pos=decoded["left"]["position"],
            left_rotmat=rot6d_to_rotmat(decoded["left"]["rot6d"]),
            right_pos=decoded["right"]["position"],
            right_rotmat=rot6d_to_rotmat(decoded["right"]["rot6d"]),
            head_pos=decoded["head"]["position"],
            head_rotmat=rot6d_to_rotmat(decoded["head"]["rot6d"]),
            eef_t_wrist=eef_t_wrist, eef_R_wrist=eef_R_wrist, dt=ACTION_DT,
            grip_width={r: decoded[r]["gripper_m"] for r in grip_roles
                        if decoded[r]["gripper_m"] is not None},
            src={"fuji_checkpoint": os.path.basename(args.checkpoint),
                 "fuji_seed": seed_ctl["n"], "fuji_anchor": anchor.source})
        try:
            completed = completer.complete(cond)
        except Exception as exc:  # noqa: BLE001 - a failed completion must not publish
            stats.note_error(exc)
            return f"COMPLETION FAILED: {type(exc).__name__}: {exc} — nothing staged."
        stats.note(completed)
        # WHERE THE NEXT CHUNK'S WIRE CONTINUES FROM (--post-anchor, owned by the clutch):
        #   off      dense re-bases on the robot's MEASURED joints when a readback exists, so
        #            the chunk starts from where the body IS — the joint-space twin of the
        #            left_only planner's measured re-anchor; 5pt is a no-op that says so.
        #   command  the last EXECUTED step is carried forward instead, leak-bounded against
        #            those same measured values and decayed under contact.
        try:
            wire_src = wire.sync(_record_frame(), _measured_force_n())
        except Exception as exc:  # noqa: BLE001 - a bad readback must not lose the chunk
            _log(f"wire re-base skipped: {exc!r}")
            wire_src = "last published"
        plan = wire.plan(completed, decoded)
        out = {"decoded": decoded, "completed": completed, "plan": plan, "wire_src": wire_src}
        staged["out"] = out
        note = ""
        if hasattr(wire, "gate_report"):
            # The 5pt family can PREDICT the low level's own keypoint gate from step 0, against
            # the same reference, so the operator sees "this would arm" before pressing start.
            _ok, note = wire.gate_report(wire.frame(plan, 0))
            note = " · " + note
        # Foot IK's cost is reported from the backend's OWN measurement, not from the switch, so
        # "I turned it on" and "it ran, and cost this" stay separate claims. Shown only when the
        # pass is actually configured on: the no-op path still bills ~3 ms of bookkeeping, and a
        # "foot IK 3 ms" line next to an unchecked box is worse than no line at all.
        with foot_lock:
            foot_on = foot_ctl["applied_fix"] != "none"
        t_foot = float(completed.stage_ms().get("foot", 0.0))
        foot_note = f" · foot IK {t_foot:.0f} ms" if foot_on else ""
        return (f"staged: sample {t_sample:.0f} ms + completion {completed.ms:.0f} ms"
                f"{foot_note} · wrist err {completed.wrist_error_m() * 1e3:.2f} mm · "
                f"base {wire_src}{note}")

    # ---------------- execute ----------------
    def _drip(out) -> int:
        """Publish the first ``exec_ctl['n']`` completed steps, one per 30 Hz step."""
        plan = out["plan"]
        wrench = out["decoded"]
        n_steps = max(1, min(int(exec_ctl["n"]), int(plan["n"])))
        step_dt = 1.0 / max(args.step_hz, 1.0)
        drip_dt = 1.0 / max(args.drip_hz, 1.0)
        last = 0
        dripping.set()
        try:
            for i in range(n_steps):
                if stop.is_set():
                    break
                rows = [None if wrench[r]["wrench"] is None else wrench[r]["wrench"][i]
                        for r in force_roles]
                fcmd = _gate_force(rows)
                frame = wire.frame(plan, i)
                _set_wire(frame, fcmd)
                last = i
                if grip_enabled.is_set() and s0.num_clients > 0:
                    for role in grip_roles:
                        widths = wrench[role]["gripper_m"]
                        if widths is None or widths.shape[0] <= i or not np.isfinite(widths[i]):
                            continue
                        frac, _sq = grip_cmd[role][0](float(widths[i]))
                        if np.isfinite(frac):
                            subs[f"grip_{role}"].send({"target_frac": float(frac)})
                            jaw_ever_commanded["v"] = True
                t_step = time.time()
                while True:
                    # The gate is re-evaluated per re-send, so flipping the switch off acts as
                    # an interrupt on a chunk that is already executing.
                    _send(frame, _gate_force(rows))
                    if stop.is_set():
                        break
                    remaining = step_dt - (time.time() - t_step)
                    if remaining <= 0:
                        break
                    time.sleep(min(drip_dt, remaining))
        finally:
            dripping.clear()
        # Commit BOTH anchors to the LAST EXECUTED step, so the next chunk branches from where
        # the body actually stopped rather than from the un-executed chunk end: the per-role
        # pose anchor the model decodes against, and the wire base the frame was built from.
        # This second commit is the INTEGRATOR under --post-anchor command; with 'off' it is the
        # dense clutch's within-chunk advance (overwritten at the next Predict) and, on 5pt,
        # nothing at all — a frozen frame has nothing to accumulate.
        for role in roles:
            planner.commit_role(role, out["decoded"][role]["position"][last],
                                out["decoded"][role]["rot6d"][last])
        wire.commit(wire.frame(plan, last))
        return last + 1

    def _release_jaw_latches() -> None:
        """Drop every hand's squeeze latch. Called on Stop / Reset: an abandoned grasp must not
        keep the next chunk's target pinned at the width the aborted one had reached."""
        for role in grip_roles:
            grip_cmd[role][1]()

    jaw_ever_commanded = {"v": False}

    def _open_jaws(why: str) -> None:
        """Command BOTH jaws fully OPEN. Reset and shutdown, never mid-chunk.

        The startup state is the gripper bridge's job (``keyboard_gripper --initial-frac 1.0``
        brings both hands up open on every low level, so this planner never has to open them to
        begin with). This is the other end of the same contract: once we have squeezed, an
        abandoned grasp must not be left clamped — on the robot OR in the drawn ghost, whose jaw
        default is also open. Only what we actually commanded is released, mirroring
        ``hl_serving``'s shutdown path: with the switch off the jaws were never driven and are
        left exactly where the bridge has them.
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
        g_layout = server.gui.add_text("layout", initial_value=loaded.layout.mode, disabled=True)
        server.gui.add_text("wire family", disabled=True,
                            initial_value=f"{wire.mode} — MODE={wire.low_level_mode}")
        g_anchor = server.gui.add_text("anchor", initial_value=anchor.describe()[:110],
                                       disabled=True)
        server.gui.add_text("post-anchor", disabled=True,
                            initial_value=wire.clutch.post.describe()[:110])
        g_obs = server.gui.add_text("obs age", initial_value="…", disabled=True)
        g_ll = server.gui.add_text("low level", initial_value="…", disabled=True)
        g_mc = server.gui.add_text("completion", initial_value="no chunk yet", disabled=True)
        g_wire = server.gui.add_text("wire", initial_value="ready pose (idle)", disabled=True)
    del g_layout

    with server.gui.add_folder("Control"):
        b_predict = server.gui.add_button("Predict")
        b_run = server.gui.add_button("Run staged chunk")
        # ONE-CLICK SEQUENTIAL ROLLOUT — the same button the left_only planner has, and the
        # cadence the whole path is meant to run at: predict → complete → execute, repeatedly,
        # with the operator arming ONCE. Manual Predict/Run stay as the single-step form for
        # bring-up and for watching one chunk.
        b_auto = server.gui.add_button("▶▶ Run to completion (sequential)")
        b_auto_stop = server.gui.add_button("⏹ Stop sequential run", disabled=True)
        b_stop = server.gui.add_button("STOP")
        b_reset = server.gui.add_button("Reset to ready pose")
        d_exec = server.gui.add_dropdown(
            # DERIVED from the horizon, never hardcoded: an H=200 checkpoint offers 200 here and
            # an H=50 one does not. Options stay BARE INTEGERS because the handler below does
            # `int(d_exec.value)` — the seconds belong in the hint, not in the option text.
            "Execute how many steps", tuple(str(n) for n in _EXEC_CHOICES),
            initial_value=str(exec_ctl["n"]),
            hint=(f"of the {ACTION_HORIZON}-step chunk, at {1.0 / ACTION_DT:.0f} Hz: "
                  + ", ".join(f"{n}={n * ACTION_DT:.2f}s" for n in _EXEC_CHOICES)
                  + ". The chunk is always predicted (and completed) in full; this is how much "
                    "of it executes before the next Predict."))
        s_auto_secs = server.gui.add_slider(
            "Sequential run seconds", min=1.0, max=120.0, step=1.0,
            initial_value=float(auto_ctl["max_s"]))
        c_ghost = server.gui.add_checkbox("Show the completion (violet)", initial_value=True)
        # The HEAD/chest triad is the reason this exists: the chest keypoint is buried in the
        # torso mesh, so it is only legible as a protruding triad — and that triad is what the
        # completion is root-conditioned on. Hands get one too (their rotation decides the
        # grasp); switch it off when the arrows clutter a long reach.
        c_triads = server.gui.add_checkbox(
            "Show predicted pose triads (L/R/head)", initial_value=bool(view_ctl["triads"]))
        g_msg = server.gui.add_text("last action", initial_value="idle", disabled=True)

    with server.gui.add_folder("Force command"):
        c_force = server.gui.add_checkbox("Command force (needs FORCE_CMD_WIRE=1)",
                                         initial_value=bool(args.enable_force_cmd))
        s_gate = server.gui.add_slider("gate (N)", min=0.0, max=20.0, step=0.1,
                                       initial_value=float(args.force_gate_n))
        c_grip = server.gui.add_checkbox("Command both jaws", initial_value=args.command_gripper)

    # ---------------- motion completion post-process ----------------
    # ONE switch, BOTH families: foot IK runs inside the completion, upstream of the wire split,
    # so there is nothing per-family to wire — but its CONSEQUENCE is per-family and the label
    # says which, because "anti-slide" sounds cosmetic and on the dense wire it is a leg command.
    with server.gui.add_folder("Motion completion"):
        c_foot = server.gui.add_checkbox(
            "Foot IK / anti-slide  ⚠️ SLOW",
            initial_value=(str(args.mc_foot_ik) != "none"))
        server.gui.add_text(
            "foot IK cost", disabled=True,
            initial_value=(f"{args.mc_foot_ik_iters or 150} iters ≈ "
                           f"{1376 if (args.mc_foot_ik_iters or 150) <= 40 else 4887} ms/chunk "
                           f"vs 481 ms budget"))
        server.gui.add_text(
            "foot IK effect", disabled=True,
            initial_value=("dense: REWRITES THE LEGS -> published joint targets (leg |dq| "
                           "0.0061→0.0493 rad/step)" if wire.mode == "dense" else
                           "5pt: moves the 2 FOOT keypoints only (absolute, ungated); the 3 "
                           "gated points are unaffected"))
        g_foot = server.gui.add_text("foot IK state", disabled=True,
                                     initial_value=str(args.mc_foot_ik))

    with server.gui.add_folder("Bring-up report"):
        for key in sorted(report):
            server.gui.add_text(key[:24], initial_value=str(report[key])[:110], disabled=True)

    flags: dict = {"predict": False, "run": False, "reset": False, "ghost_step": 0}

    # ---------------- the sequential rollout ----------------
    def _run_to_completion() -> None:
        """predict → complete → execute, repeatedly, until Stop / the ceiling. ONE arm gesture.

        This is the cadence the whole path exists to run at — a chunk is 50 steps of 30 Hz
        (1.67 s) and ``--exec-steps`` of them actually execute, so clicking Predict/Run per chunk
        makes the robot pause every ~0.5 s at a chunk boundary. The manual buttons stay for
        bring-up (watch ONE completion, read its gate report, then decide); this is for running a
        task.

        WHAT IT DOES NOT CHANGE — the safety envelope. Every iteration goes through exactly the
        same ``_predict`` → ``_drip`` pair the buttons call: the same obs-freshness refusal, the
        same completion-failure abort, the same ``wire.sync`` re-base (measured under
        ``--post-anchor off``, leak-bounded carry under ``command``), the same per-step force
        gate re-read, the same ``wire.commit`` to the LAST EXECUTED step, the same ``busy``
        mutex. It removes the click and nothing else.

        BOUNDED FOUR WAYS, because a policy has no end of its own the way a recorded episode
        does: ``⏹`` (retires the generation), ``STOP`` (also interrupts the chunk in flight,
        within one step), the wall-clock ceiling ``--auto-max-seconds`` (live on the slider), and
        the low level dropping off :8770.
        """
        if auto_running.is_set():
            g_msg.value = "ignored — a sequential run is already active; press ⏹"
            return
        if dripping.is_set():
            g_msg.value = "ignored — a chunk is executing"
            return
        if s0.num_clients == 0:
            g_msg.value = ("the low level is NOT connected to :8770 — start it in ABSOLUTE mode "
                           f"(MODE={wire.low_level_mode})")
            return
        with auto_lock:
            auto_ctl["running"] = True
            my_gen = int(auto_ctl["gen"])
            max_s = float(auto_ctl["max_s"])
        auto_running.set()
        b_auto_stop.disabled = False
        # THE ONLY PLACE THIS ROLLOUT CLEARS `stop`. A stop latched by an earlier abort must not
        # make the first iteration a silent no-op; every LATER `stop` is the operator's STOP and
        # has to both interrupt the drip and end the loop, so clearing it per iteration (which is
        # what the manual Run branch does, correctly, for a single chunk) would make STOP
        # unpressable here.
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
                # The low level going away mid-rollout is not "nothing moves": the heartbeat
                # keeps publishing into a dead socket while this loop keeps planning off it.
                if s0.num_clients == 0:
                    g_msg.value = (f"sequential run STOPPED — the low level dropped off :8770 "
                                   f"after {n_steps} step(s)")
                    _log("SEQUENTIAL: the low level disconnected from :8770 — rollout ended.")
                    break
                if not busy.acquire(blocking=False):
                    time.sleep(0.02)   # a manual Predict/Reset holds the mutex; wait it out
                    continue
                try:
                    # Rate-limited, because it is true for the WHOLE run when it is true at all
                    # and this loop would otherwise print it once per chunk. Worse here than for
                    # one manual chunk: with no readback every chunk re-bases on the last
                    # COMMANDED frame, so a tracking shortfall compounds with nothing measuring
                    # it — and under `--post-anchor command` the leak bound has no reference, so
                    # `KeypointClutch` freezes instead of integrating blind.
                    if _record_frame() is None:
                        _now = time.monotonic()
                        if _now - openloop_warn["t"] > 5.0:
                            openloop_warn["t"] = _now
                            _log("SEQUENTIAL: no usable RecordTap (RECORD=1?) — chunks re-base on "
                                 "the last COMMANDED frame, so the shortfall compounds.")
                    note = _predict()
                    out = staged["out"]
                    if out is None:
                        # A refusal or a failed completion. Ending the rollout is the only honest
                        # response: the alternative is a loop that keeps re-planning off a frozen
                        # scene until the ceiling, publishing the last frame the whole time.
                        g_msg.value = f"sequential run STOPPED — {note}"
                        _log(f"SEQUENTIAL: stopped, nothing staged — {note}")
                        break
                    _draw_paths(out)
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
        # blocks for the whole take (up to --auto-max-seconds) and both of those must stay
        # responsive — STOP is on one of them.
        threading.Thread(target=_run_to_completion, daemon=True, name="seq").start()

    @b_auto_stop.on_click
    def _(_e) -> None:
        # Retire the loop by bumping the generation: the chunk in flight finishes (bounded by
        # exec-steps, ~0.5 s at 16) and the loop then exits without re-planning. Use STOP to
        # interrupt the motion ITSELF within one step.
        with auto_lock:
            auto_ctl["running"] = False
            auto_ctl["gen"] += 1

    @b_stop.on_click
    def _(_e) -> None:
        # Flag only; the loop thread owns the state (repo rule). `stop` is read by the drip
        # loop on every step and by its inner re-send, so it interrupts an executing chunk
        # within one step rather than at the chunk boundary.
        stop.set()
        # STOP also RETIRES a sequential rollout. `stop` alone would already end it (the loop
        # only clears it at its own entry), but bumping the generation makes that explicit and
        # closes the window where the loop is between iterations: it can then never re-plan
        # after a STOP, only exit.
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
        # boundary. It can only ever stop the body SOONER — a running rollout cannot be extended
        # past a Stop by dragging this.
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

    @c_triads.on_update
    def _(_e) -> None:
        # Toggles the ALREADY DRAWN triads too, not just the next chunk's: an operator flips
        # this while looking at a staged chunk, which is the only moment it matters.
        _set_triads_visible(bool(c_triads.value))

    @c_foot.on_update
    def _(_e) -> None:
        # A REQUEST, not the change. The completer is reconfigured by the predicting thread at
        # the top of `_predict` (see `_apply_foot_ik`), so this lands on a chunk boundary and can
        # never swap a post-process underneath a chunk that is being completed or executed. The
        # staged chunk keeps whatever it was completed with, which is why the state text below
        # says "next Predict" rather than pretending it took effect now.
        fix = "inertialize" if bool(c_foot.value) else "none"
        with foot_lock:
            foot_ctl["fix"] = fix
            pending = fix != foot_ctl["applied_fix"]
        iters = args.mc_foot_ik_iters or 150
        g_foot.value = f"{fix} (from the next Predict)" if pending else fix
        if fix == "none":
            g_msg.value = "foot IK OFF from the next Predict — completion back to ~80 ms/chunk"
        else:
            g_msg.value = (f"⚠️ foot IK ON from the next Predict — {iters} iters, ~"
                           f"{1376 if iters <= 40 else 4887} ms/chunk vs a 481 ms budget, so the "
                           "body will PAUSE between chunks"
                           + (" · it rewrites the LEGS, which this dense wire publishes"
                              if wire.mode == "dense" else " · it moves the 2 ungated feet"))

    # ---------------- the loop ----------------
    _log(f"page: http://{args.host}:{args.port}")
    _log("sequence: press A on the joystick (ready pose) → Predict → arm (start) → then either "
         "'Run staged chunk' for ONE chunk, or '▶▶ Run to completion' to keep going until ⏹ / "
         f"the {args.auto_max_seconds:g} s ceiling.")
    last_status = 0.0
    while True:
        try:
            # A sequential rollout owns Predict/Run/Reset for its duration. It holds `busy` only
            # WITHIN a chunk, so without these three refusals a manual press could land in the
            # gap between two iterations — a Predict would overwrite the chunk the rollout is
            # about to drip, and a Reset would clear the operator's latched STOP and re-home the
            # decode anchor underneath a live take. ⏹ is the way out, and it is right here.
            _seq = auto_running.is_set()
            if _seq and (flags["predict"] or flags["run"] or flags["reset"]):
                flags["predict"] = flags["run"] = flags["reset"] = False
                g_msg.value = ("a sequential run is active — press ⏹ to end it (or STOP to "
                               "interrupt the motion now)")
            if flags["reset"]:
                flags["reset"] = False
                stop.clear()
                staged["out"] = None
                if hasattr(wire, "clutch"):
                    wire.clutch.reset()
                _open_jaws("reset")   # drops the latches too
                planner.seed_anchor(
                    np.stack([anchor.pos[r] for r in roles]),
                    np.stack([rotmat_to_rot6d(anchor.rotmat[r]) for r in roles]))
                _clear_paths()
                # Joints AND root, via the same call the ghost was BUILT with — see
                # `_reset_ghost`. Restoring only the joints left the violet body standing where
                # the last take ended.
                _reset_ghost()
                # So a later Predict's playback starts its loop at step 0 of the NEW chunk
                # instead of resuming the retired one's phase.
                flags["ghost_step"] = 0
                # The DECODE state resets unconditionally — it is pure in-process pose maths and
                # decides what the NEXT chunk branches from. The WIRE only resets while UN-ARMED:
                # re-publishing the ready pose to an armed low level would drive the whole body
                # back to it from wherever it is, at an unbounded distance. Armed, the wire holds
                # the last executed frame and the idle clutch stays frozen; press Reset before
                # arming (or disarm first) to actually re-home.
                rframe = _record_frame()
                armed = rframe is not None and rframe.vive_armed
                if not armed:
                    _set_wire(wire.idle(), None)
                g_msg.value = ("reset: decode anchor + clutch + violet ghost back at the ready "
                               "pose; the WIRE is held (ARMED — disarm to re-home the body)"
                               if armed else
                               "reset: anchor, clutch, ghost and wire back at the ready pose")
            if flags["predict"]:
                flags["predict"] = False
                if busy.acquire(blocking=False):
                    try:
                        g_msg.value = "predicting …"
                        g_msg.value = _predict()
                        out = staged["out"]
                        if out is not None:
                            _draw_paths(out)
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
                                   f"ABSOLUTE mode (MODE={wire.low_level_mode})")
                elif busy.acquire(blocking=False):
                    try:
                        stop.clear()
                        n = _drip(out)
                        staged["out"] = None
                        g_msg.value = f"executed {n} step(s); re-anchored on the last one"
                    finally:
                        busy.release()
                else:
                    g_msg.value = "busy"

            # The violet ghost plays the STAGED completion on a loop; it is the numbers on the
            # wire, so it needs no IK and cannot disagree with what is commanded.
            out = staged["out"]
            if out is not None and ghost_root.visible:
                completed = out["completed"]
                i = flags["ghost_step"] % completed.n
                flags["ghost_step"] = i + 1
                # An UNKNOWN jaw renders OPEN, never 0.0 — 0.0 is the CLOSED end of this joint's
                # travel, so the old default drew a hand with no predicted width as a hand that
                # is HOLDING something. It also has to agree with the hardware, which now comes
                # up open on both hands by declaration (keyboard_gripper --initial-frac 1.0), and
                # with the live mesh, whose remap_dof_to_urdf defaults are already OPEN.
                # `jaw_joint_from_width_m` returns JAW_JOINT_OPEN_M for a non-finite width for
                # the same reason, so the two branches agree.
                jaws = {}
                for role in grip_roles:
                    widths = out["decoded"][role]["gripper_m"]
                    jaws[role] = (jaw_joint_from_width_m(float(widths[i]))
                                  if widths is not None else JAW_JOINT_OPEN_M)
                # On the DENSE wire these are the PUBLISHED joints, not the completion's raw
                # ones: the ghost must be what the robot is TOLD, so with --dense-clutch joint
                # it shows the clutched command and step 0 sits on the robot's own pose rather
                # than 23 deg away from it. On the 5pt wire the command is five keypoints, so
                # the ghost is the completion's own body — the thing those five were picked off,
                # and therefore still "what the robot was told" as closely as a 29-joint mesh
                # can say it.
                ghost_q = (out["plan"]["published"][i] if wire.mode == "dense"
                           else completed.joint_pos[i])
                ghost.update_cfg(F.urdf_cfg_from_qpos(
                    ghost_q, list(completer.joint_names), urdf_joints,
                    jaw_left=jaws.get("left", JAW_JOINT_OPEN_M),
                    jaw_right=jaws.get("right", JAW_JOINT_OPEN_M)))
                # The completion's own root, in the same floor-z=0 world.
                ghost_root.position = tuple(np.asarray(completed.root_pos[i], dtype=float))
                ghost_root.wxyz = tuple(np.asarray(completed.root_quat[i], dtype=float))

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
                g_mc.value = stats.summary()
                # What the COMPLETER is actually configured with, not what the switch asked for —
                # the two differ from a flip until the next Predict opens a new episode, and that
                # gap is the whole safety story of a live post-process switch.
                with foot_lock:
                    _ff, _fa = str(foot_ctl["fix"]), str(foot_ctl["applied_fix"])
                g_foot.value = _fa if _ff == _fa else f"{_fa} → {_ff} at the next Predict"
                with wire_lock:
                    fc = last_wire["fcmd"]
                    frame = tuple(a.copy() for a in last_wire["frame"])
                g_wire.value = (f"{wire.status(frame)} · "
                                f"F^cmd {'OFF' if fc is None else f'{np.abs(fc).max():.1f} N'}")
                g_anchor.value = anchor.describe()[:110]
            time.sleep(0.02)
        except KeyboardInterrupt:
            _log("interrupted")
            # Release before main()'s finally stops the subscribers — a Ctrl-C mid-task must not
            # leave a hand clamped on the object with nothing left running to open it. Blocks
            # briefly so the frame actually gets on the wire.
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
