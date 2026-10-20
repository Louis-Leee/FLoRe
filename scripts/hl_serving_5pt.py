#!/usr/bin/env python3
"""ONLINE FIVE-POINT PLANNER: ``diffusion_umi_5pt`` straight onto the ``5pt`` low level.

The deploy half of the five-keypoint preset. Same predict -> preview -> execute UX as
``scripts/hl_serving_mc.py`` and ``scripts/hl_serving_wholebody.py``, the same six sensor bridges,
the same force gate, the same ``:8770`` wire — with the completion removed and NOTHING put in its
place, because the policy predicts every slot the wire carries::

  hl_serving_mc.py   2 D405 + 2 Bota + 2 jaws -> S1 (both_hands_head_dual, 41-D)
                       |  3 anchor-relative EE chunks
                     decode against the G1 READY POSE
                       |  3 absolute pose series
                     Motion_Completion   (a SECOND model, a SECOND interpreter, ~80 ms/chunk)
                       |  qpos(29) -> its own FK
                     KeypointReadout -> FivePointWire -> :8770  {"mode":"5pt"}

  THIS SCRIPT       2 D405 + 2 Bota + 2 jaws -> S1 (fivepoint_geom_*, 59-D)
                       |  5 anchor-relative keypoint chunks + 2 grippers + 2 wrenches
                     decode against the G1 READY POSE
                     KeypointReadout -> FivePointDirectWire -> :8770  {"mode":"5pt"}

Same low level either way: ``MODE=5pt-unified-anneal``, W&B ``tqnoooo6``. No ``g1_mosyn`` conda
env, no backend subprocess, no 30-60 s JIT, no ``--mc-foot-ik`` trade-off.

THE OPERATOR'S SEQUENCE, AND WHERE EACH STEP LIVES
=================================================
    press A          the low level ramps to its NEUTRAL pose and HOLDS it. That pose is
                     ``home_anchor.json::default_qpos`` with the 14 arm joints zeroed
                     (``HOLOSOMA_INIT_READY_POSE=1``), and it is what this planner FKs.
    Predict          the chunk is DECODED against exactly that pose, so a chunk step means
                     "where the five points go, starting from where the robot is".
    on that basis    the KEYPOINT CLUTCH publishes ``p_base + (p[i] - p[0])`` with ``base`` = the
                     ready frame, so step 0 is a bit-exact no-op on the low level's own pre-arm
                     gate reference and every later step is the chunk's own displacement.
    -> pelvis frame  ``KeypointReadout`` expresses all five in the ANCHOR's pelvis frame:
                     15 positions + 20 xyzw quats, which IS ``_compute_vr_fk``'s
                     ``vr_pos_local``/``vr_orn_local``.

WHY THE CLUTCH IS **ON** BY DEFAULT HERE (it is off on the completion path)
==========================================================================
Measured on ``diffusion_umi_5pt_0902_merged3_h200_umi_ddp4/best_overall``, 6 samples spread across
the dataset — the gap between the wire's step 0 and the gate's reference::

    post-anchor        LEFT hand   RIGHT hand   torso     LEFT foot   RIGHT foot
    off (absolute)      59.4 mm     62.8 mm     9.3 mm     17.6 mm      9.2 mm
    command  (default)   0.0         0.0        0.0         0.0         0.0
    the gate            <= 70 mm    <= 70 mm   <= 70 mm    warn 300    warn 300

The DATA's own row 0 is its anchor to **0.000 mm / 0.000 deg** for all five roles, so those
59-63 mm are the MODEL's step-0 error, not a frame mistake — inside the 70 mm tolerance with 7 mm
to spare, i.e. arming works today and stops working on a slightly worse sample, silently, as a
refusal to arm. On the completion path the same number is 0.7-3.1 mm because the completion is
*told* to hit the decoded hand pose, which is why ``off`` is sane there and not here.

⚠️ ALL FIVE POINTS ARE CARRIED (``--carry all``), unlike the completion path's three. There the
feet were the completion's own stance, 69-91 mm from the ready crouch by construction, so
re-basing them shifted the foot trajectory 74.7 mm — a leg command nobody asked for. Here they are
decoded against the ready pose's OWN ankles and start 9-18 mm away, so carrying costs that as a
rigid offset and buys them the same measured leak bound the hands get. ``--carry upper`` restores
the reviewed behaviour.

A ONE-HAND CHECKPOINT ON THE SAME WIRE (``diffusion_umi``, layout ``left_only``)
==================================================================================
The plain ``diffusion_umi`` preset predicts the LEFT hand alone — 16-D action (pose + jaw + wrench),
a 7-D ``wrt_start`` state, ONE camera (``camera0_rgb`` = the LEFT D405), ONE Bota, ONE jaw — and
this page serves it onto the SAME ``5pt`` low level with the four points it does not predict
**HELD at the ready pose**, exactly as ``hl_serving.py``'s ``WireMapper`` held RIGHT + HEAD on the
3-point wire. Nothing is branched on in the page: ``hl_5pt.check_layout`` names the family,
``hl_5pt.build_planner`` picks the planner (the base ``LivePlanner`` observation, which is what such
a checkpoint was trained on), and ``FivePointDirectWire(held=...)`` fills the missing slots from the
SAME ``resolve_deploy_anchor`` the five-point path decodes against — so a held slot is a constant
row, the clutch publishes its base for it (the gate reference on chunk 1), and a Predict never
moves it. What follows from the layout, read off ``loaded.layout`` rather than declared:

* cameras: the checkpoint's own count; a configured bridge it does not read is simply unread
  (logged), a missing one is still refused by name (``load_hl_model``);
* force: only the predicted hand's Bota is binned into ``force_history``; the wire's ``f_cmd`` is
  ALWAYS ``[L, R]`` with a zero for the held hand (``hl_5pt.force_command``);
* jaws: only the predicted hand's is subscribed and commanded; the other stays as it is.

The same rule serves ``both_hands`` and ``both_hands_head[_dual]`` (fewer held slots), which is
what "adapt the 5pt path to 1pt" means here: a subset of the wire, not a second wire. A
``fivepoint_geom_*`` checkpoint is byte-identical to before (nothing is held, ``f_cmd`` was
already ``[L, R]``).

REPLAYING A RECORDED EPISODE ON THIS WIRE (``--replay-zarr``, 2026-09-19)
========================================================================
``hl_serving.py``'s recorded-episode replay (``flore/s1_policy/serving/hl_replay.py``), on the 5pt
wire: pick an episode of the checkpoint's own training zarr in the page and ▶▶ plays it — the
RECORDED hand pose(s) as the chunk, the RECORDED jaw as the jaw command, and the RECORDED Bota
wrench as ``F^cmd`` (through the same ``hl_5pt.force_command``: ``-F_meas``, gated on magnitude,
``[L, R]`` with a zero for a hand the layout does not carry). It is ONE ``if`` at the top of
``_predict``: the chunk comes from ``RecordedEpisodeSource.chunk_at`` (anchor-RELATIVE, the model's
own output form) and is decoded by the SAME ``decode_chunk`` against the SAME running anchor, so
everything downstream — held slots, clutch, gate, drip, jaw state machine, manual-jaw override,
STOP / ⏹ / the ceiling, the belief leash, and the ⏺ run recorder with its F/T and camera streams —
is the reviewed code, byte-identical. Two modes: ``sequential`` re-anchors every chunk to the last
executed step (the receding clutch, like a policy run); ``open-loop`` decodes the whole episode ONCE
against the arm's pose and drips it straight (a true open-loop tracking number). The page shows the
episode's FIRST recorded frame (every camera the checkpoint reads) and the cursor's frame. What the
converter marked unmeasured is never commanded: an unmeasured wrench step is zeroed, a dead-encoder
jaw width is held at the last measured one (``hl_replay.apply_step_validity``). ⚠️ The demo replays
RE-ORIENTED by the angle between the human's hand at the reference frame and the robot's decode
anchor (the page prints it as ``transport``) — physical, not a frame bug; see
``hl_replay._body_vs_world_transport``. Without ``--replay-zarr`` none of this exists on the page.

WHAT IS DELIBERATELY NOT HERE (first bring-up)
==============================================
The take recorder (the ⏺ run recorder is its replacement here), temporal ensembling, the
executed-trace QC sheet, and the ``dense`` wire. A five-point policy has no joint channel, so
there is no ``DenseWire`` to publish for; and the two ``hl_serving_mc.py`` flags that let a
3-tracker policy's points reach this wire (``--upper-source`` / ``--foot-source``) have no meaning
when the policy predicts all five.

BRING-UP, in order — every one is checked at startup and named if missing::

    T1  bash 2_ft_bridge.sh --pipeline              both Botas, 6-DoF, :8771
    T2  camera L :8773   and   camera R :8777       both D405s (this model reads TWO views)
    T3  LEFT=1 RIGHT=1 KEYBOARD=0 bash 5_dual_gripper_keyboard.sh    LEFT :8784, RIGHT :8785
    T4  MODE=5pt-unified-anneal HOLOSOMA_VIVE_RELATIVE=0 RECORD=1 \
          bash 3_run_policy_vive_force.sh           the low level (ABSOLUTE, RecordTap on)
    T5  ./scripts/run_hl_serving_5pt.sh --checkpoint <5pt ckpt>

T4's ``MODE`` **must** be ``5pt-unified-anneal`` and nothing cross-checks it: the RecordTap frame
carries no preset name. A mismatch is not dangerous — the low level's width guard drops every
frame of the wrong shape and the robot holds its ready pose — but "nothing moves" reads as "T4 is
the wrong family" before it reads as anything else.

⚠️ SAFETY. This commands a physical G1 whole body: the low level is a whole-body BALANCE
controller reading all five points, and two of them are feet. First runs gantry-suspended,
operator on the hardware E-stop, force wire OFF. Per-point motion is bounded by the low level's
own slew limit, and with the default clutch the chunk starts exactly where the robot is rather
than at the model's absolute claim.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_FLORE_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _FLORE_ROOT not in sys.path:
    sys.path.insert(0, _FLORE_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# The sensor plumbing, the gripper calibration and the freshness rules come from the left_only
# planner rather than being copied: they are model-agnostic and each has its own review history.
# Importing the module is side-effect-free (it binds nothing and parses no args until main()).
import hl_serving as HL  # noqa: E402

from flore.client.s0_driver import S0Driver  # noqa: E402
from flore.s1_policy.data.layout import ACTION_DT, ACTION_HZ  # noqa: E402
from flore.s1_policy.data.transforms import delta_ee_to_absolute  # noqa: E402
from flore.s1_policy.inference import ready_pose as RP  # noqa: E402
from flore.s1_policy.inference.keypoint_ik import KeypointIK  # noqa: E402
from flore.s1_policy.serving import (  # noqa: E402
    hl_5pt,
    hl_dense,
    hl_replay,
    hl_stream,
    rollout_record,
    tty_keys,
    walk_skill,
)
from flore.s1_policy.serving.hl_reach import MAX_CLAMP_M, ReachConfig  # noqa: E402
from flore.s1_policy.serving.gil_yield import YieldGate  # noqa: E402
from flore.s1_policy.serving.hl_online import (  # noqa: E402
    ACTION_HORIZON,
    OBS_MAX_AGE_S,
    decode_chunk,
    exec_step_choices,
    load_hl_model,
    snap_exec_steps,
)
from flore.s1_policy.serving.live_robot import (  # noqa: E402
    G1_29DOF_SDK_DOF_NAMES,
    JAW_JOINT_OPEN_M,
    build_sdk_to_urdf_index,
    jaw_joint_from_width_m,
    parse_record_frame,
    remap_dof_to_urdf,
)
from flore.utils.rot6d import (  # noqa: E402
    quat_xyzw_to_wxyz,
    rot6d_to_rotmat,
    rotmat_to_quat_xyzw,
    rotmat_to_rot6d,
)

#: Translucent violet — the same ghost colour every other planner and the offline viewer use, so
#: "violet = what is about to happen" means one thing across the whole repo.
_GHOST_RGBA = HL._GHOST_RGBA

#: Per-role plan styling. Same table and the same reasoning as ``hl_serving_mc._PLAN_STYLE``: the
#: CHEST keypoint is inside the torso mesh and translates centimetres where a hand travels tens,
#: so it needs long axes to protrude and a wide stride not to clutter. The two FEET get the same
#: treatment for the same reason — they sit at z ≈ -0.72 in the readout frame, under the body, and
#: their per-chunk motion is millimetres.
_PLAN_STYLE = {
    "left": {"rgb": (60, 160, 255), "axes": 0.07, "stride": 10},
    "right": {"rgb": (255, 120, 200), "axes": 0.07, "stride": 10},
    "head": {"rgb": (255, 200, 40), "axes": 0.26, "stride": 12},
    "left_foot": {"rgb": (120, 230, 140), "axes": 0.12, "stride": 12},
    "right_foot": {"rgb": (230, 160, 90), "axes": 0.12, "stride": 12},
}
_PLAN_STYLE_FALLBACK = {"rgb": (200, 200, 200), "axes": 0.10, "stride": 10}

#: How often the idle heartbeat re-publishes the held wire. The low level's stale gate is 0.25 s
#: and ``S0Driver`` stops re-broadcasting after 0.5 s of silence, so this only has to beat 0.25 s.
IDLE_HB_HZ = 60.0

#: Default ``--ood-stop``: how far outside the checkpoint's own ``[q01, q99]`` the observation may
#: sit, in band widths, before a ▶▶ run ENDS. ⚠️ It is not a model-quality threshold — it is a
#: DIVERGENCE detector, and 0.25 was read off the two rig runs that folded the robot. While the low
#: level is armed, ``obs_mode=last_target`` makes the state a pure function of the accumulated
#: COMMAND (``commit_role`` writes the executed step into the anchor and ``snapshot_obs`` derives the
#: whole observation from it — no measured quantity enters it), while the robot is rate-limited to
#: 8.33 mm/step over one L2 pool. So a command the arm cannot reach becomes an observation the robot
#: is not in, and the next chunk is sampled from it. Both logged runs are that curve, monotone and
#: self-amplifying: 0.16 -> 0.28 -> 0.33 -> 0.47 -> 0.65 band widths over five chunks in one, and
#: 0.03 -> 1.15 over sixteen in the other (``lh_dx`` 0.605 -> 0.876 m, past the arm's own reach).
#: 0.25 is therefore past the first excursion of a run that recovers and inside the second of one
#: that does not; every measured healthy start of every shipped checkpoint reads 0.00 (all 12
#: geometry columns inside the band, 24-84% of the way across each). Raise it to be more permissive,
#: 0 to go back to log-only.
OOD_STOP_BANDS_DEFAULT = 0.25

#: Default ``--obs-leak-m`` / ``--obs-leak-deg``: how far the COMMANDED anchor — which under
#: ``obs_mode=last_target`` is the whole observation — may run ahead of the MEASURED keypoints
#: before it is pulled back. Deliberately the SAME numbers ``PostAnchorConfig`` gives the published
#: base (``--post-anchor-eps-m`` 0.05 / ``--post-anchor-eps-deg`` 10), because the bug was that the
#: wire had this leash and the belief did not: ``KeypointClutch.sync`` bounded what gets published
#: to 5 cm of measured at every Predict while ``commit_role`` let the belief integrate the model's
#: own claim without bound. Not aliased to the clutch's flags on purpose — the two answer different
#: questions ("what may I publish" vs "what may I believe") and a run may want the belief on a
#: shorter leash than the wire — but the defaults are kept equal so there is one number to reason
#: about until someone deliberately splits them.
OBS_LEAK_M_DEFAULT = 0.05
OBS_LEAK_DEG_DEFAULT = 10.0

#: ---- the CONTINUOUS (HATO) cadence's own constants -------------------------------------------
#: How often a streamed run re-applies the clutch's leak bound (``wire.sync``). A chunked run does
#: it once per Predict, which at the default 16 exec steps is every 0.53 s — matched here so the
#: bound behaves identically rather than becoming a property of which cadence is running. Each call
#: is one RecordTap parse plus one ``KeypointFK`` (0.66 ms), so it is affordable at this rate and
#: wasteful at 30 Hz.
STREAM_SYNC_EVERY = 16
#: How long the wire may go without a live chunk before the run ENDS. A frozen wire is safe (the
#: last frame keeps being published and the low level holds) but it is not what the operator asked
#: for, and a sampler that cannot cover its own horizon will not recover on its own.
STREAM_STARVE_S = 1.0
#: Floor on how often the newest chunk's five paths are redrawn. Streaming produces a chunk every
#: ~50-200 ms and each redraw replaces 15 viser scene nodes, so the page is throttled rather than
#: flooded. Display only — the wire is unaffected.
STREAM_DRAW_MIN_S = 0.5
#: What ``Reset to ready pose`` answers while the low level is ARMED. Reset re-homes the DECODE
#: anchor and the clutch base, and while armed the wire is deliberately NOT re-published to the ready
#: frame (that would drive the whole body back to it from wherever it is) — so the robot stays where
#: it stopped while the NEXT chunk's step 0 is pinned on the ready frame: ``KeypointClutch.sync``
#: answers "READY frame (nothing executed yet)" with nothing carried and never looks at measured.
#: Reproduced: a hand 316 mm from home was commanded straight back to it, slew-capped at 5 mm/tick,
#: and the leak bound then yanked it to within 5 cm of measured 16 streamed steps later — the take
#: that "wanders at the start" with a drawn plan leaving from the ready pose. Refused whole, naming
#: the protocol that does work: the low level's ``start`` TOGGLES arm/disarm, disarming returns the
#: hands to its frozen reference (= the ready pose), and the pre-arm gate then passes on the re-homed
#: wire. Continuing from the stopped pose needs no Reset at all — STOP leaves anchor and clutch there.
RESET_REFUSED_ARMED = (
    "Reset REFUSED — the low level is ARMED, so the robot is NOT at the ready pose: a Reset now would "
    "pin the next chunk's step 0 on the ready frame and drag the arm back to it. To RE-HOME: joystick "
    "`start` (DISARM — the hands return to the ready pose) → Reset → `start` (re-arm) → Predict / ▶▶. "
    "To CONTINUE FROM HERE: press ▶▶ (or Predict → Run) directly — STOP left the anchor and the "
    "clutch where the body stopped, no Reset needed.")

#: Execute-N-of-S choices offered in the page, DERIVED from this process's horizon rather than
#: hardcoded — see :func:`hl_online.exec_step_choices`. ``ACTION_HORIZON`` is resolved from
#: ``FLORE_ACTION_HORIZON`` at import time and ``run_hl_serving_5pt.sh`` exports it from the
#: checkpoint's own ``config.json``, so an H=100 checkpoint offers 100 and an H=200 one offers
#: 100 and 200 with no flag. The pin is what makes deriving it here sound: a checkpoint whose
#: horizon disagrees with the constant is refused outright by ``validate_model_layout``, so this
#: tuple can never advertise a window the model will not fill.
#: MODULE scope for the reason ``hl_serving._EXEC_CHOICES`` documents: main() (the snap) and
#: _run_viser (the dropdown) both read it, and as a main() local the dropdown raised NameError
#: inside _run_viser's broad except, surfacing only as a page that never loaded.
_EXEC_CHOICES = exec_step_choices(ACTION_HORIZON)


def _log(msg: str) -> None:
    print(f"[hl-5pt] {msg}", flush=True)


def _triad_indices(n: int, stride: int) -> np.ndarray:
    """Which chunk steps get an orientation triad: every ``stride``-th, plus ALWAYS the last.

    The final step is what the next chunk's wire continues from (``wire.commit`` carries the last
    EXECUTED step forward), so it is never the sample to drop to a stride boundary. Same helper,
    same reason, as ``hl_serving_mc._triad_indices``.
    """
    n = max(int(n), 1)
    stride = max(int(stride), 1)
    return np.unique(np.append(np.arange(0, n, stride), n - 1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="an HL-v2 checkpoint dir: a five-point one (hl_data.mode == "
                        "'fivepoint_geom_*'), or any keypoint layout whose roles are a subset of "
                        "the wire's five — left_only (the plain diffusion_umi preset), both_hands, "
                        "both_hands_head[_dual]. The slots it does not predict are HELD at the "
                        "ready pose. Whole-body layouts are refused by name.")
    p.add_argument("--algo", default="diffusion", choices=["diffusion", "flow", "act"],
                   help="which algorithm block the checkpoint carries (default: %(default)s). The "
                        "five-point family is a LAYOUT, not a fourth algorithm — it is checked "
                        "separately, by name. run_hl_serving_5pt.sh derives this from the "
                        "checkpoint, so it normally never has to be typed.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-steps", type=int, default=None,
                   help="sampler steps (default: the checkpoint's own)")
    p.add_argument("--seed", type=int, default=None,
                   help="PIN the sampler seed, so every sample from one observation is the same "
                        "draw. Unset (the default) it increments per sample, which is why two "
                        "Predicts from an identical observation give different chunks — measured "
                        "on a five-point checkpoint, 7-19 mm of worst-step spread on a ~70-85 mm "
                        "reach, from the seed ALONE. Reset restores the counter either way, so "
                        "'back to the start' is bit-exact and pressing Predict again is a fresh "
                        "draw. Pin it to A/B anything else (a flag, a checkpoint, the scene); "
                        "leave it unset on a task run, where re-sampling a bad chunk is the point.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8092)

    g = p.add_argument_group("sensor bridges (the dual rig)")
    g.add_argument("--force-ws", default="ws://127.0.0.1:8771",
                   help="Bota bridge, BOTH hands, run with --pipeline (default: %(default)s)")
    g.add_argument("--camera-ws", default="ws://127.0.0.1:8773", help="camera L (camera0_rgb)")
    g.add_argument("--camera-right-ws", default="ws://127.0.0.1:8777",
                   help="camera R (camera1_rgb). REQUIRED by a 2-camera checkpoint; pass an empty "
                        "string to declare a 1-camera rig, which then refuses a 2-camera model by "
                        "name instead of feeding it a grey view. A 1-camera checkpoint (left_only) "
                        "simply never reads this bridge, configured or not.")
    g.add_argument("--live-camera-hz", type=float, default=4.0, metavar="HZ",
                   help="the page's '📷 Live cameras' folder: the CURRENT frame of EVERY configured camera "
                        "bridge (read by this checkpoint or not — camera R under a left_only model is shown, "
                        "display only) plus, per view the model reads, the exact 224x224 crop it is fed, "
                        "re-drawn this often. 0 turns the folder off. (default: %(default)s)")
    g.add_argument("--live-camera-width", type=int, default=320, metavar="PX",
                   help="width of the live camera thumbnails on the page; the bridge's full frame is "
                        "downscaled to it, aspect kept (default: %(default)s)")
    g.add_argument("--gripper-ws", default="ws://127.0.0.1:8784", help="LEFT jaw bridge")
    g.add_argument("--gripper-right-ws", default="ws://127.0.0.1:8785", help="RIGHT jaw bridge")
    g.add_argument("--record-ws", default="ws://127.0.0.1:8779",
                   help="low-level RecordTap (needs RECORD=1). Drives the live robot mesh and is "
                        "the MEASURED reference the carried keypoint command is leak-bounded "
                        "against; without it the clutch FREEZES rather than integrating blind.")
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
                   help="wall-clock ceiling for EITHER ▶▶ run, sequential or continuous (live on "
                        "the page). A policy has no end of its own, so this is the bound that stops "
                        "the body even if the operator looks away.")
    c.add_argument("--ood-stop", type=float, default=0.0, metavar="BANDS",
                   help="OPTIONAL BACKSTOP (default 0 = OFF, log only): end a ▶▶ run when the "
                        "observation the next chunk would be sampled FROM sits more than BANDS band "
                        "widths outside the checkpoint's own [q01, q99]. It refuses BEFORE the "
                        "sample, so nothing new is published and the wire holds the LAST EXECUTED "
                        "step — the same place a STOP leaves it, so ▶▶ with no Reset continues from "
                        "there. Jaw columns are excluded from the decision (still reported). ⚠️ It "
                        "is off by default because --obs-leak-m PREVENTS the divergence it detects: "
                        f"ending the run was the wrong response to a symptom. "
                        f"{OOD_STOP_BANDS_DEFAULT:g} is the measured threshold (past the first "
                        "excursion of a run that recovers, inside the second of one that does not); "
                        "1.0 is a loose backstop that only fires if the leash above failed.")
    c.add_argument("--no-ood-stop", dest="ood_stop", action="store_const", const=0.0,
                   help="never end a run on an out-of-band observation — log it and keep sampling "
                        "(the pre-2026-09-13 behaviour; same as --ood-stop 0)")
    c.add_argument("--obs-leak-m", type=float, default=OBS_LEAK_M_DEFAULT, metavar="M",
                   help="THE FIX for a diverging belief, and the reason --ood-stop is off by "
                        "default: bound the PLANNER's anchor — which under obs_mode=last_target IS "
                        "the whole observation — to within this many metres of the MEASURED "
                        f"keypoints at every Predict (default {OBS_LEAK_M_DEFAULT:g} m, the same "
                        "number KeypointClutch.sync already applies to the published base; 0 = off, "
                        "the pre-2026-09-13 behaviour). Inside the bound it is a NO-OP, so a healthy "
                        "run is byte-identical and action[:,0] == state still holds; it engages only "
                        "once the command has run further from the robot than the WIRE is allowed "
                        "to. No RecordTap = no correction (the anchor is held, never guessed).")
    c.add_argument("--obs-leak-deg", type=float, default=OBS_LEAK_DEG_DEFAULT, metavar="DEG",
                   help="the rotation half of --obs-leak-m: bound each keypoint's believed "
                        f"orientation to this geodesic deviation from measured (default "
                        f"{OBS_LEAK_DEG_DEFAULT:g}°, the clutch's own --post-anchor-eps-deg)")

    n = p.add_argument_group("the CONTINUOUS cadence (HATO-style asynchronous streaming)")
    n.add_argument("--cadence", default="sequential", choices=["sequential", "stream"],
                   help="which ▶▶ run the page STARTS on; the page's '▶▶ cadence' dropdown is the "
                        "live switch. 'sequential' (THE DEFAULT, and the reviewed path) is "
                        "predict-then-drip: the wire is FROZEN for the whole sample, which is "
                        "16-46%% of the time depending on how long a sample takes. 'stream' runs the "
                        "sampler on its own thread and publishes an ENSEMBLE of every chunk still "
                        "covering the current step, so the wire never freezes — HATO's §IV-B "
                        "deployment algorithm (third_party/hato/inference_node.py).")
    n.add_argument("--ensemble", default=hl_stream.DEFAULT_ENSEMBLE_MODE,
                   choices=list(hl_stream.ENSEMBLE_MODES),
                   help="[--cadence stream] how the overlapping predictions for one step are "
                        "reduced to one command; HATO's own four names. 'avg' (default, and HATO's) "
                        "measured best on jerk, 'act' best on tracking, 'new'/'old' are single-"
                        "prediction A/Bs. Live on the page.")
    n.add_argument("--ensemble-tau", type=float, default=hl_stream.DEFAULT_ACT_TAU,
                   help="[--ensemble act] the fold factor z = z*tau + c*(1-tau); 0.5 is HATO's. "
                        "Larger weights the OLDER survivors more.")
    n.add_argument("--replan-min-steps", type=int, default=0, metavar="N",
                   help="[--cadence stream] leave at least N chunk steps between two samples "
                        "(0 = re-plan as fast as the sampler allows, which is HATO's cadence and "
                        "gives the thickest overlap). Raise it to cut GPU duty; the blend then has "
                        "fewer members and is measured to stop helping below ~5.")
    n.add_argument("--stream-ik-ghost", dest="stream_ik_ghost", action="store_true",
                   help="[--cadence stream] keep solving the violet IK body DURING a streamed run. "
                        "Off by default: the solver is pure Python and would contend with the 30 Hz "
                        "publisher for the one interpreter, and the publisher's cadence is the "
                        "whole point of this mode while the body is display only. Live on the page.")
    n.set_defaults(stream_ik_ghost=False)

    d = p.add_argument_group("the wire")
    d.add_argument("--post-anchor", default="command", choices=list(hl_dense.POST_ANCHOR_MODES),
                   help="the KEYPOINT CLUTCH. 'command' (THE DEFAULT HERE, unlike the completion "
                        "path) publishes p_base + (p[i] - p[0]) so step 0 lands on the gate's own "
                        "reference and the base carries the last EXECUTED step forward, "
                        "leak-bounded against RecordTap's measured keypoints. 'off' publishes the "
                        "model's ABSOLUTE readout, whose step 0 is measured 59-63 mm from the "
                        "reference against a 70 mm tolerance.")
    d.add_argument("--carry", default="all", choices=list(hl_5pt.CARRY_MODES),
                   help="[--post-anchor command] which points the clutch carries. 'all' (default) "
                        "is all five: the feet are decoded against the ready pose's own ankles, so "
                        "carrying them costs a measured 9-18 mm rigid offset and gives them the "
                        "same leak bound the hands get. 'upper' carries the 3 GATED points and "
                        "passes the feet through absolutely — the completion path's behaviour, "
                        "whose reason (a completion standing 69-91 mm straighter) does not apply "
                        "here.")
    d.add_argument("--post-anchor-eps-m", type=float, default=0.05, metavar="M",
                   help="[--post-anchor command] per-keypoint leak bound (default 0.05): how far "
                        "the carried command may lead the robot's measured keypoints before it is "
                        "pulled back. 0 collapses the mode to 'off'.")
    d.add_argument("--post-anchor-eps-deg", type=float, default=10.0, metavar="DEG",
                   help="[--post-anchor command] the same bound on orientation")
    d.add_argument("--post-anchor-contact-n", type=float, default=1.5, metavar="N",
                   help="[--post-anchor command] measured |F| at or above which the carried anchor "
                        "DECAYS toward the robot instead of advancing")
    d.add_argument("--no-zero-arms", dest="zero_arms", action="store_false",
                   help="build the decode anchor and idle wire from the raw crouch default_qpos "
                        "instead of the arms-zeroed ready pose. Only correct against "
                        "HOLOSOMA_INIT_READY_POSE=0.")
    d.set_defaults(zero_arms=True)
    d.add_argument("--yam-urdf", default=os.path.join(
        _FLORE_ROOT, "third_party", "UMI_collection", "robot_assets", "g1_yam",
        "g1_29dof_yam.urdf"), help="URDF for the drawn robot, the violet IK body and the FK")

    v = p.add_argument_group("the violet IK body")
    v.add_argument("--no-ik-ghost", dest="ik_ghost", action="store_false",
                   help="skip the violet body entirely (the plan still draws as five paths). The "
                        "IK is DISPLAY ONLY — nothing solved here reaches the wire.")
    v.set_defaults(ik_ghost=True)

    f = p.add_argument_group("force + gripper")
    # Both default ON. The page's switches are still the live control — this only says what they
    # READ at startup. Force still cannot reach the robot unless the low level was given
    # FORCE_CMD_WIRE=1, and it is hard-zeroed below --force-gate-n either way.
    f.add_argument("--enable-force-cmd", dest="enable_force_cmd", action="store_true", default=True,
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

    r = p.add_argument_group(
        "reach compensation (HANDS only; default OFF)",
        description="run_hl_serving.sh's --reach-ki on this wire (flore/s1_policy/serving/hl_5pt.py "
                    "HandReach over hl_reach.ReachCompensator): accumulate the part of the commanded "
                    "HAND motion the arm did not execute and add it to the PUBLISHED target, only while "
                    "the hand performed less than --reach-stall-frac of what was asked. Hands only: the "
                    "feet are a stance the balance controller reads, the torso and any HELD slot pass "
                    "through byte-identically. The clutch base and the observation stay UNCOMPENSATED "
                    "(the residual is published, never committed), it is the IDENTITY while the low "
                    "level is un-armed, and the residual is dropped on STOP / Reset / disarm / a walk "
                    "start / the page toggle. Same flag names and defaults as run_hl_serving.sh.")
    r.add_argument("--reach-ki", type=float, default=0.0, metavar="KI",
                   help="ENABLE with this integral gain per published step (default 0 = OFF, "
                        "byte-identical wire). START AT 0.05; 0.1 and above spends much longer pinned "
                        "at the clamp. Live-toggleable in the page's 'Reach compensation' folder.")
    r.add_argument("--reach-clamp", type=float, default=0.03, metavar="M",
                   help="hard bound (m) on the accumulated residual per hand: the published hand "
                        f"target never exceeds the model's by more than this. Ceiling {MAX_CLAMP_M} m.")
    r.add_argument("--reach-stall-frac", type=float, default=0.5, metavar="F",
                   help="THE GATE: accumulate ONLY while the hand performed less than this fraction of "
                        "the motion the wire asked for over --reach-window steps (measured as "
                        "(dMeasured.dCommanded)/|dCommanded|^2). Commanded 6 cm, moved 3 -> leave it; "
                        "commanded 5 cm, did not move -> accumulate.")
    r.add_argument("--reach-window", type=int, default=9, metavar="N",
                   help="published steps the realized fraction is measured over (9 = 0.3 s at 30 Hz)")
    r.add_argument("--reach-gate", type=float, default=0.0, metavar="M",
                   help="optional secondary distance ceiling (m, 0 = off): stop growing the residual "
                        "once the shortfall exceeds this")
    r.add_argument("--reach-decay", type=float, default=0.995, metavar="D",
                   help="per-step leak on the stored residual (<= 1)")
    r.add_argument("--reach-contact-n", type=float, default=1.5, metavar="N",
                   help="CONTACT CUT-OUT per hand: at or above this measured |F| the residual stops "
                        "growing and decays — a hand pressing on the cart shows the same small "
                        "shortfall as one that fell short, and integrating THAT commands a harder press "
                        "toward the 40 N stop. A Bota this rig reads that did not answer counts as "
                        "contact. 0 disables it; do not on hardware.")

    w = p.add_argument_group(
        "walk skill (EXTENSION — always on the page; the bundled walk-with-box feet)",
        description="The walk-with-box clip's two FEET on the wire's two foot slots, under the wire's own "
                    "(unchanged) upper three points — flore/s1_policy/serving/walk_skill.py. One toggle "
                    "button on the page: press to walk, press again to ramp the feet back and stop. A second "
                    "toggle (🦵 Legs ← clip, hotkey l) puts the SAME feet + legs under the RUNNING policy "
                    "instead — the planner keeps predicting the upper three points, the clip drives the feet, "
                    "press again to hand them back (the feet pursue the planner's own target). The "
                    "feet ship with the package (flore/s1_policy/serving/assets/walk_with_box_feet.npz), "
                    "so nothing here needs a flag.")
    w.add_argument("--walk-clip", default=str(walk_skill.BUNDLED_CLIP), metavar="PATH",
                   help="the clip to play: an extracted feet NPZ (scripts/extract_walk_feet.py) or a raw "
                        "bones_seed_g1_holosoma_f32 motion NPZ. Default: the bundled walk-with-box feet.")
    w.add_argument("--walk-speed", type=float, default=1.0,
                   help="playback speed multiplier at start (live on the page, "
                        f"{walk_skill.SPEED_MIN:g}-{walk_skill.SPEED_MAX:g}). The feet's per-tick "
                        "demand scales with it, so 0.5 halves the printed slew ratio.")
    w.add_argument("--no-walk-loop", dest="walk_loop", action="store_false",
                   help="play the clip ONCE and ramp the feet back; the default loops it (with a "
                        "rate-bounded blend across the clip's wrap gap) until the button is pressed "
                        "again")
    w.set_defaults(walk_loop=True)
    w.add_argument("--walk-hz", type=float, default=walk_skill.LOW_LEVEL_HZ,
                   help="publish rate of the walk (default: the low level's %(default)s Hz tick, so "
                        "every tick gets a fresh foot target; the clip is time-resampled to it)")
    w.add_argument("--walk-ramp-speed", type=float, default=0.25, metavar="M_S",
                   help="rate bound (m/s) on the lead-in / wrap / lead-out ramps between the wire's "
                        "feet and the clip's (default 0.25 = 5 mm per 50 Hz tick, i.e. exactly the low "
                        "level's pinned position cap, so a ramp is never clamped)")
    w.add_argument("--walk-ramp-min-s", type=float, default=0.5, metavar="S",
                   help="floor on every ramp's duration (default %(default)s s)")
    w.add_argument("--no-walk-legs", dest="walk_legs", action="store_false",
                   help="do NOT send the clip's 29 reference joints (the wire's optional `ref_joints`) "
                        "beside its feet. Default ON: the low level's command_lower_body obs then "
                        "describes the same gait as the two foot keypoints instead of its frozen idle "
                        "clip's standing legs — the configuration that stepped IN PLACE on the robot. "
                        "Off = that pre-fix configuration, for an A/B; also a checkbox on the page.")
    w.set_defaults(walk_legs=True)
    k = p.add_argument_group("terminal hotkeys (the terminal this page was launched from)")
    k.add_argument("--no-keys", dest="keys", action="store_false",
                   help="do not read hotkeys off this terminal's stdin. Default ON when stdin is a foreground "
                        "TTY: p = ▶▶ run / ⏹ pause (ONE key: starts when idle, retires at the chunk boundary "
                        "when running), k = 🚶 walk toggle, l = 🦵 legs ← clip toggle (the clip's feet + legs "
                        "under the RUNNING policy; again = hand back), SPACE = STOP, h = this table. Each key runs the "
                        "SAME handler its page button does and nothing else; holosoma's own keyboard keys are "
                        "refused as bindings. Unavailable by itself (reason logged) under a pipe, nohup or a "
                        "background job — the page runs on without it.")
    k.set_defaults(keys=True)
    rc = p.add_argument_group("run recorder (the ⏺ button: one recording per episode — from the first Predict / "
                              "▶▶ to STOP / Reset / the low level disarming — to a time-named folder)")
    rc.add_argument("--record-dir", default=os.path.join(_FLORE_ROOT, "data", "rollouts"), metavar="DIR",
                    help="root of the recordings: DIR/<task>/<YYYY-MM-DD_HH-MM-SS>/, one folder per recording "
                         "(chunks / steps / published / measured / force / events .jsonl + meta.json). "
                         "Plot one with scripts/plot_rollout_record.py <folder>. Default data/rollouts.")
    rc.add_argument("--record-task", default=None, metavar="NAME",
                    help="the <task> folder name (push_gantry, push_cart, rotate_box, wipe_board …). The "
                         "launcher forwards TASK= here; default = read off the --checkpoint path.")
    rc.add_argument("--record-every-run", action="store_true",
                    help="start with 'Record EVERY run' ticked: the first Predict / ▶▶ after each end opens "
                         "the next folder instead of asking for a click.")
    rc.add_argument("--no-record-autoplot", dest="record_autoplot", action="store_false", default=True,
                    help="do not spawn scripts/plot_rollout_record.py in the background (nice -n 10) when a "
                         "recording is saved. Default: plot automatically; the page never waits on it.")
    rc.add_argument("--record-camera-every", type=int, default=1, metavar="N",
                    help="record every N-th frame of EVERY camera this checkpoint reads (one or two views, "
                         "read off the page's own subscriptions) into <run>/camera/<image_key>/NNNNNN.jpg + "
                         "camera/frames.jsonl — the bridge's own full-size JPEG, not the 224 crop. 1 (default) "
                         "= every frame (~30 fps, ~2 MB/s per view); 0 = no camera frames.")
    rp = p.add_argument_group("recorded-episode replay (drive the wire with a demonstration instead of the policy)")
    rp.add_argument("--replay-zarr", default=None, metavar="PATH|checkpoint",
                    help="enable RECORDED-EPISODE REPLAY from a converted .zarr.zip. 'checkpoint' resolves the "
                         "zarr the checkpoint was TRAINED on (hl_data.zarr_paths[0]), so the replayed demos are "
                         "exactly the policy's own training distribution. The page gains a 'Replay recorded "
                         "episode' folder: Chunk SOURCE (recorded replay | model, live), an episode picker (with the "
                         "episode's first recorded frame and the cursor's frame shown), a start frame, and a replay "
                         "mode. ▶▶ then plays the RECORDED poses / jaw / wrench (as F^cmd) through the SAME wire, "
                         "clutch, gate, drip, STOP and ⏺ recorder as a policy run. The model is still loaded.")
    rp.add_argument("--replay-episode", type=int, default=0, metavar="N",
                    help="the episode the replay panel starts on (default 0); selectable live in the page.")
    rp.add_argument("--replay-mode", default="sequential", choices=("sequential", "openloop"),
                    help="how ▶▶ plays a recorded episode. sequential (default): one ACTION_HORIZON chunk per "
                         "Predict, re-anchored to the LAST EXECUTED step every --exec-steps steps, exactly like a "
                         "policy run (the receding clutch; the servo's lag is folded forward). openloop: the WHOLE "
                         "episode decoded ONCE against the arm's pose and dripped straight through — one placement, "
                         "no re-anchor, a true open-loop tracking number. Live on the page.")
    return p


# ---------------------------------------------------------------------------
# bring-up
# ---------------------------------------------------------------------------
def _camera_urls(args) -> list[str]:
    urls = [args.camera_ws]
    if str(args.camera_right_ws).strip():
        urls.append(args.camera_right_ws)
    return urls


def _live_thumb(frame_rgb: np.ndarray, width: int) -> np.ndarray:
    """The page's display copy of a camera frame: downscaled to ``width`` px (aspect kept, INTER_AREA),
    channel order UNCHANGED — shown exactly as ``_decode_jpeg_b64`` hands it over (natural RGB), which
    is also what the planner is fed. A frame no wider than ``width`` (or ``width <= 0``) is returned as
    is, never upscaled."""
    import cv2

    h, w = frame_rgb.shape[:2]
    if width <= 0 or w <= width:
        return frame_rgb
    return cv2.resize(frame_rgb, (int(width), max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)


def _measured_five_point(args, anchor, readout):
    """``rframe -> (pos, orn) | None``: the ROBOT's measured keypoints, FK'd from RecordTap.

    The reference the carried command is leak-bounded against, and it has to be a MEASUREMENT:
    unclamped integration diverges to 224 mm over a take, while clamping to the ready pose would
    cap the task itself (a real reach moves a hand ~48 cm). On this wire "where the robot is" only
    exists as 29 encoder readings, so it is FK'd through the SAME ``KeypointFK`` the anchor and the
    training-time geometry use, cross-checked against the anchor at startup — the identical
    function ``hl_serving_mc._measured_five_point`` performs, kept here rather than imported
    because that module is a CLI entry point that parses args at import.
    """
    fk, why = RP.KeypointFK.load(urdf_path=args.yam_urdf)
    if fk is None:
        _log(f"⚠️  measured keypoints unavailable ({why}); a CARRIED command will FREEZE at the "
             "chunk boundary rather than integrate with nothing to bound it against.")
        return None
    check_pos, _ = readout.from_bodies(*fk.bodies(anchor.dof, lift_z=anchor.pelvis_z))
    ready_pos, _ = hl_dense.ready_pose_five_point(anchor, readout)
    gap = float(hl_dense.keypoint_gaps_m(check_pos, ready_pos).max())
    _log(f"measured-keypoint FK cross-check against the anchor: {gap * 1e6:.1f} µm "
         f"({'OK' if gap < 1e-4 else 'HIGH — the leak bound would compare two frames'})")

    def measured(rframe):
        if rframe is None or rframe.dof_pos is None:
            return None
        return readout.from_bodies(*fk.bodies(rframe.dof_pos, lift_z=anchor.pelvis_z))

    return measured


def _self_check(subs: dict, cams: list[str], loaded, *, require_6dof: bool,
                consumes_force: bool) -> dict:
    """Probe every bridge this deploy reads, and name the missing one.

    Deliberately NOT ``hl_serving._self_check``: that one knows about one camera and one jaw, and a
    dual rig that forgot camera R would pass it and then feed the second encoder group a grey frame
    for the whole session. Same probe as ``hl_serving_mc._self_check``/``hl_serving_wholebody``'s,
    which it is the five-point twin of.
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
                f"self-check: no camera frame for {loaded.image_keys[i]} on {url}. This checkpoint "
                f"reads {loaded.num_cameras} view(s); a missing one is fed to its encoder group as "
                "a grey image and nothing downstream would say so.")
        report[f"camera{i} ({loaded.image_keys[i]})"] = f"ok ({url})"
    for role in loaded.layout.gripper_roles:
        gmsg = _wait(subs[f"grip_{role}"], need_key="actual")
        report[f"gripper {role}"] = (
            "ok" if gmsg is not None else
            "NO FRAME — the jaw width reads as UNMEASURED (state_valid=False), which is what the "
            "training converter does for a dead encoder; the grasp channel is degraded")
    # PARSED, not merely received: `parse_record_frame` is what the leak bound and the live mesh
    # consume, and it rejects a frame whose `type` is not "record" or whose `dof_pos` is the wrong
    # length. A weaker "did any JSON arrive" check would report ok for a tap neither can use.
    rmsg = _wait(subs["record"], need_key="dof_pos", tries=20)
    try:
        rframe = parse_record_frame(rmsg)
    except Exception:  # noqa: BLE001 - a malformed tap is a degraded mode, not a failure
        rframe = None
    if rframe is not None:
        report["RecordTap"] = "ok — live robot mesh + the carried command's MEASURED leak bound"
    elif rmsg is not None:
        report["RecordTap"] = (
            "PRESENT but UNUSABLE (not a 29-DOF `record` frame) — treated as absent: the mesh "
            "falls back to the ready pose and a CARRIED command FREEZES at each boundary")
    else:
        report["RecordTap"] = (
            "ABSENT — relaunch the low level with RECORD=1. The mesh holds the ready pose and a "
            "CARRIED command FREEZES rather than integrating blind, which is the safe degradation "
            "but means the wire stops advancing between chunks.")
    return report


def main(argv=None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    cams = _camera_urls(args)
    HL._limit_cpu_thread_pools(args.device)

    # --- 1. the model ---------------------------------------------------------------
    _log(f"loading {args.algo} checkpoint {args.checkpoint} on {args.device} …")
    loaded = load_hl_model(args.checkpoint, args.algo, device=args.device,
                           num_steps=args.num_steps, max_cameras=len(cams))
    family = hl_5pt.validate_layout(loaded.layout)
    held = hl_5pt.held_roles(loaded.layout)
    if len(cams) > loaded.num_cameras:
        # load_hl_model already refuses "too few" (a missing view is SILENT: the encoder group eats
        # a grey frame). The other direction — a bridge running that this checkpoint was not
        # trained on — is the NORMAL state of a left_only checkpoint on the dual rig (camera R is
        # up for the 5pt checkpoints and simply unread here), so it is logged and trimmed, never
        # refused. The first URL is always camera0_rgb, the LEFT D405, which is the one such a
        # checkpoint reads.
        _log(f"NOTE: {len(cams)} camera bridge(s) configured but this checkpoint reads "
             f"{loaded.num_cameras} ({list(loaded.image_keys)}) — subscribing to the first "
             f"{loaded.num_cameras} only: {cams[:loaded.num_cameras]}. The other bridge is unread.")
        cams = cams[:loaded.num_cameras]
    consumes_force = bool(loaded.layout.force_roles)
    _log(f"ready: algo={loaded.algorithm} mode={loaded.layout.mode} family={family} "
         f"action_dim={loaded.layout.action_dim} state_dim={loaded.layout.state_dim} "
         f"state={list(loaded.layout.state_geom) or 'wrt_start_rot6d'} To={loaded.obs_history_len} "
         f"FH={loaded.force_hist_len} cameras={list(loaded.image_keys)} "
         f"force_roles={loaded.layout.force_roles} gripper_roles={loaded.layout.gripper_roles}")
    if held:
        _log(f"roles: the policy predicts {list(loaded.layout.roles)}; the wire's other "
             f"{len(held)} slot(s) {list(held)} are HELD at the READY POSE for the whole run — the "
             f"clutch publishes their base (= the low level's own pre-arm gate reference on the "
             f"first chunk), a Predict never moves them, and their f_cmd is a hard zero. This is "
             f"hl_serving.py's WireMapper rule (RIGHT + HEAD held) on the five-slot wire.")
    else:
        _log("roles: the policy predicts all five wire slots; nothing is held.")
    _log(f"force command: {'ENABLED' if args.enable_force_cmd else 'off'} at launch "
         f"(gate {args.force_gate_n:g} N — below that |F^cmd| is hard-zeroed and the motion is "
         f"pure position). The low level must ALSO have FORCE_CMD_WIRE=1.")
    _log(f"jaw(s) {list(loaded.layout.gripper_roles)}: "
         f"{'COMMANDED' if args.command_gripper else 'READ-ONLY'} at launch (they come up OPEN "
         f"either way)"
         + ("" if len(loaded.layout.gripper_roles) == 2 else
            " — the other hand's jaw is not read or commanded by this checkpoint")
         + ". A 'Manual jaws' press OVERRIDES the policy's jaw for that hand until Reset.")
    if args.enable_force_cmd and not consumes_force:
        raise SystemExit(
            "force command is enabled but this checkpoint predicts no force for any role, so there "
            "would be nothing to put on the wire. Pass --no-force-cmd, or serve a checkpoint whose "
            "active algorithm block declares force.")
    if abs(args.step_hz - 1.0 / ACTION_DT) > 0.5:
        raise SystemExit(
            f"--step-hz {args.step_hz} is not the training grid's {1.0 / ACTION_DT:.0f} Hz. The "
            "keypoints are published one frame per step, so any other rate plays the whole body "
            "back at the wrong speed.")
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
    if args.seed is None:
        _log("sampler seed: increments per sample (Reset returns it to the start, so the first "
             "Predict after a Reset repeats the first Predict after startup bit-exactly). Two "
             "Predicts WITHOUT a Reset are different draws from one conditional — measured on a "
             "five-point checkpoint, 7-19 mm of worst-step spread. --seed N pins it.")
    else:
        _log(f"sampler seed: PINNED at {int(args.seed)} — every sample from one observation is the "
             "same chunk. Intended for an A/B; on a task run a bad chunk cannot be re-sampled.")

    # --- 2. the decode anchor: the pose held after `A`, for ALL FIVE wire slots ---------
    # Resolved over the WIRE's roles and not the layout's, whatever the checkpoint predicts: the
    # idle frame, the gate reference and the HELD slots need every one of the five, and the planner
    # below is seeded with the layout's own subset of the same anchor.
    roles = tuple(loaded.layout.roles)
    anchor, why = RP.resolve_deploy_anchor(hl_5pt.WIRE_ROLES, zero_arms=args.zero_arms)
    if anchor is None or anchor.dof is None:
        raise SystemExit(
            f"cannot resolve the G1 ready pose, which IS the decode anchor, the clutch base and "
            f"the idle wire here: {why}")
    _log(f"decode anchor: {anchor.describe()}")
    for note in anchor.notes:
        _log(f"  ⚠️  {note}")
    if not args.zero_arms:
        _log("⚠️  --no-zero-arms: only correct against HOLOSOMA_INIT_READY_POSE=0.")

    # --- 3. the readout + the ready frame -------------------------------------------
    # The readout pelvis is the ANCHOR's — (0, 0, pelvis_z) with identity rotation — and NEVER
    # mc_frames.PELVIS_STAND_Z (0.79186), which belongs to the teleop path's cond_mode="home"
    # placement and would put every keypoint 35 mm low. See hl_dense.KeypointReadout.
    readout = hl_dense.KeypointReadout(np.array([0.0, 0.0, float(anchor.pelvis_z)]))
    ready_pos, ready_orn = hl_dense.ready_pose_five_point(anchor, readout)
    _log(f"readout: {readout.describe()}")
    _log("ready-pose wire (pelvis frame): " + " · ".join(
        f"{lab} {np.round(ready_pos.reshape(-1, 3)[i], 4).tolist()}"
        for i, lab in enumerate(hl_dense.FIVE_POINT_LABELS)))

    # --- 4. the wire ----------------------------------------------------------------
    post = hl_dense.PostAnchorConfig(
        mode=args.post_anchor,
        eps_m=float(args.post_anchor_eps_m),
        eps_rad=float(np.deg2rad(args.post_anchor_eps_deg)),
        contact_n=float(args.post_anchor_contact_n),
    )
    _log(f"post anchor: {post.describe()}")
    clutch = hl_5pt.build_clutch(ready_pos, ready_orn, post=post, carry=args.carry)
    measured_frame = _measured_five_point(args, anchor, readout)
    # The wire slots the policy does not predict, at the ready pose — the SAME anchor the chunk is
    # decoded against, so a held slot and a predicted one share one frame. Empty for a five-point
    # checkpoint.
    held_poses = hl_5pt.held_anchor_poses(anchor, loaded.layout)
    wire = hl_5pt.FivePointDirectWire(readout, ready_pos, ready_orn, clutch=clutch,
                                      measured_frame=measured_frame, held=held_poses)
    if not post.carrying:
        _log("⚠️  --post-anchor off: the model's ABSOLUTE readout goes on the wire. Measured on "
             "the shipped checkpoint, step 0 lands 59-63 mm from the low level's own pre-arm gate "
             "reference against a 70 mm tolerance — inside it, with 7 mm to spare, so expect a "
             "refusal to arm on a worse sample. The default 'command' makes step 0 exact.")
    else:
        _log(f"clutch: carrying {clutch.n_carried}/{len(hl_dense.FIVE_POINT_BODIES)} points, so "
             f"step 0 lands ON the gate reference"
             + ("" if args.carry == "all" else
                " — the two FEET pass through ABSOLUTELY and drift with the decode anchor"))
    _log(f"wire: mode={wire.mode!r} · the low level MUST be MODE={wire.low_level_mode} · "
         f"its pre-arm gate is {wire.gate_note}")

    # --- 5. the planner + its anchor -------------------------------------------------
    # Picked by the layout family (hl_5pt.build_planner): the geometric FivePointPlanner for a
    # fivepoint_geom_* checkpoint, the base LivePlanner's wrt_start observation (as
    # SubsetKeypointPlanner) for left_only / both_hands*. Both offer push_proprio() and
    # state_readout(), which is all the page asks of them.
    planner = hl_5pt.build_planner(loaded, obs_hz=args.obs_hz)
    planner.obs_max_age_s = (OBS_MAX_AGE_S if not args.obs_max_age or float(args.obs_max_age) <= 0
                             else float(args.obs_max_age))
    planner.grip_max_age_s = max(planner.obs_max_age_s, HL.GRIP_MAX_AGE_S)
    planner.seed_anchor(
        np.stack([anchor.pos[r] for r in roles]),
        np.stack([rotmat_to_rot6d(anchor.rotmat[r]) for r in roles]))
    # THE FRAME THE BELIEF LEASH COMPARES IN, proved on the ready pose before any Predict can be
    # leashed. The anchor above is the WORLD frame (floor z = 0); `wire.measured_keypoints` is the
    # wire's PELVIS frame; `_belief_bound` converts the readback with hl_5pt.wire_frame_to_anchor.
    # The FK cross-check above compares two READOUTS (both pelvis-relative) and so cannot see a slip
    # between anchor and wire — this one converts the ready pose's own wire frame back and demands
    # it land ON the anchor, per predicted role. Raw, the gap is pelvis_z (757 mm, 2026-09-18).
    _back_pos, _ = hl_5pt.wire_frame_to_anchor((ready_pos, ready_orn), readout)
    _frame_gap = max(
        float(np.linalg.norm(_back_pos.reshape(-1, 3)[hl_5pt.WIRE_ROLES.index(r)] - anchor.pos[r]))
        for r in roles)
    _log(f"belief-leash frame check: the ready pose's wire readout, converted back to the anchor "
         f"frame, lands {_frame_gap * 1e6:.1f} µm from the decode anchor on {list(roles)} "
         f"({'OK' if _frame_gap < 1e-4 else 'HIGH — the leash would compare two frames'})")
    if _frame_gap > 1e-3:
        raise SystemExit(
            f"the belief leash would compare the anchor against a readback in another frame "
            f"({_frame_gap * 1e3:.1f} mm apart on the ready pose); refusing to serve.")
    names0, state0, _written = planner.state_readout()
    _log(f"planner: {type(planner).__name__} · observation at the ready pose: " + " · ".join(
        f"{name} {value:+.4f}" for name, value in zip(names0, state0)))
    # IS THAT FIRST OBSERVATION SOMETHING THE POLICY HAS SEEN? Scored against the checkpoint's OWN
    # norm_stats['state'] band rather than against a per-task table: the ready pose is the same for
    # every checkpoint, but where the TRAINING episodes started is a dataset placement decision
    # (attrs.wb_cond_mode — `home` for push_cart / push_gantry, `absolute --dz-anchor hand` for
    # 0902 / rotate_box), and a state outside the band gives a plausible chunk from an extrapolated
    # conditional with nothing else on the page saying so. Re-read live on the page (`state range`)
    # and logged again on every Predict that samples outside it. See hl_5pt.state_range_report.
    state_stats = loaded.norm_stats.get("state") if loaded.norm_stats else None
    # The jaw columns are left out HERE: no bridge has reported yet, so they read 0.0 (closed) and
    # would flag every checkpoint whose training jaws were open. The page's live readout checks them.
    range0 = hl_5pt.state_range_report(names0, state0, state_stats,
                                       ignore=[n for n in names0 if n.endswith("_grip")])
    if range0.ok:
        _log(f"observation range: {range0.describe()} (jaw columns judged live, once their bridges "
             f"report) — the first Predict conditions on a state this checkpoint trained on.")
    else:
        _log(f"⚠️  observation range at the ready pose: {range0.describe()}. The first chunk "
             f"is an EXTRAPOLATION on those columns (the dataset's episodes did not start where "
             f"this robot does); read it with that in mind, or measure the dataset with "
             f"scripts/measure_5pt_state_gap.py.")
    # THE BELIEF'S LEASH, and why it is the primary answer to a diverging observation rather than a
    # gate that ends the run. See `_belief_bound` / FivePointPlanner.leak_anchor_toward.
    if float(args.obs_leak_m) > 0 and hasattr(planner, "leak_anchor_toward"):
        _log(f"belief leash: the commanded anchor — which IS the observation here "
             f"(obs_mode=last_target) — is pulled back to within {float(args.obs_leak_m) * 1e3:.0f} "
             f"mm / {float(args.obs_leak_deg):g}° of the MEASURED keypoints at every Predict "
             f"(--obs-leak-m/-deg), the same leash KeypointClutch.sync already gives the published "
             f"base. Inside it nothing moves (a healthy run is byte-identical); it engages only "
             f"once the command has run further from the robot than the WIRE may. No RecordTap = "
             f"no correction, the anchor is HELD. Bounded role(s): {list(roles)} — a wrt_start "
             f"state reads only their ROTATION, and that is the channel that ran on a left_only "
             f"run (67° commanded vs 47° measured, 2026-09-14) before this leash covered it.")
    elif hasattr(planner, "leak_anchor_toward"):
        _log("⚠️  belief leash: OFF (--obs-leak-m 0) — the commanded anchor may run arbitrarily far "
             "from the robot while the clutch keeps the published base within "
             f"{args.post_anchor_eps_m * 1e3:.0f} mm of it. That asymmetry is what folded the body "
             "twice: every later chunk conditions on a hand position the arm never reached.")
    if float(args.ood_stop) > 0:
        _log(f"observation gate: a ▶▶ run also ENDS when a body column sits more than "
             f"{float(args.ood_stop):g} band widths outside that range (--ood-stop) — a BACKSTOP "
             f"behind the leash above. It refuses BEFORE the sample, so the wire, the decode anchor "
             f"and the clutch base all stay on the LAST EXECUTED step and ▶▶ continues from there "
             f"with no Reset. Jaw columns are reported but never end a run.")
    else:
        _log("observation gate: OFF (--ood-stop 0, the default) — an out-of-band observation is "
             "LOGGED and sampled from anyway, because the leash above is what keeps it from getting "
             "there. --ood-stop 0.25 re-arms it (1.0 = a loose backstop).")

    # --- 5a. RECORDED-EPISODE REPLAY (extension; only with --replay-zarr) ----------------
    # Resolved HERE, before a socket is opened, so a wrong path / a non-replayable file / a layout the
    # source cannot read is one line at startup. `replay` stays None without the flag and the page
    # below is then the page as reviewed.
    replay = None
    if args.replay_zarr:
        replay_path = str(args.replay_zarr)
        if replay_path == "checkpoint":
            # The checkpoint records the zarr it was TRAINED on, so 'checkpoint' guarantees the replayed
            # demos are the distribution the policy learned — the apples-to-apples A/B.
            import json as _json

            with open(os.path.join(args.checkpoint, "config.json")) as fh:
                trained_paths = _json.load(fh).get("hl_data", {}).get("zarr_paths") or []
            if not trained_paths:
                raise SystemExit("--replay-zarr checkpoint: the checkpoint declares no hl_data.zarr_paths; "
                                 "pass an explicit .zarr.zip path.")
            replay_path = str(trained_paths[0])
            if not os.path.isabs(replay_path):
                replay_path = os.path.join(_FLORE_ROOT, replay_path)
            _log(f"replay: resolved the checkpoint's training zarr -> {replay_path}")
        if family != "keypoint_subset":
            raise SystemExit(
                f"--replay-zarr: recorded replay on this page serves the keypoint_subset family (left_only, "
                f"both_hands, both_hands_head[_dual]); this checkpoint's layout {loaded.layout.mode!r} is "
                f"{family!r}. Its two FEET are FK'd wb_* arrays with no camera-grid pose, which the replay "
                f"source does not read yet.")
        try:
            replay = hl_replay.RecordedEpisodeSource(replay_path, loaded.layout)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            raise SystemExit(f"--replay-zarr {replay_path}: {exc}") from exc
        _lens = replay.episode_lengths()
        _firsts = replay.first_usable_frames()
        _n_shift = sum(1 for f in _firsts if f not in (0, None))
        _n_dead = sum(1 for f in _firsts if f is None)
        _log(f"replay: {os.path.basename(replay.path)} — {replay.n_episodes} episodes, {replay.n_frames} frames, "
             f"lengths {min(_lens)}..{max(_lens)} (median {int(np.median(_lens))} = "
             f"{np.median(_lens) * ACTION_DT:.1f} s); {_n_shift} start on a masked frame (auto-seeked), "
             f"{_n_dead} have no playable frame; camera views {list(replay.image_keys)}")
        if not 0 <= int(args.replay_episode) < replay.n_episodes:
            raise SystemExit(f"--replay-episode {args.replay_episode} outside [0, {replay.n_episodes - 1}]")
        if float(args.auto_max_seconds) < float(np.median(_lens)) * ACTION_DT:
            _log(f"⚠️  replay: the ▶▶ ceiling --auto-max-seconds {args.auto_max_seconds:g} s is shorter than the "
                 f"MEDIAN episode ({np.median(_lens) * ACTION_DT:.1f} s) — a replay that reaches it stops there. "
                 f"Raise the page's '▶▶ run seconds' slider for a whole episode.")
        _speed = hl_replay.speed_report(replay, int(args.replay_episode), float(args.step_hz), _EXEC_CHOICES)
        if _speed:
            _log(f"replay: {_speed}. Speed only — the executed PATH is the recorded one; every chunk re-commands "
                 f"its step 0 (the pose the arm holds), so a k-step chunk advances the demo by k-1.")
        _log(f"replay: mode {args.replay_mode} · the RECORDED wrench goes on the wire as F^cmd through the same "
             f"switch and {args.force_gate_n:g} N gate as the policy's (-F_meas, [L, R]); an unmeasured (pre-tare) "
             f"step is zeroed, a dead-encoder jaw step holds the last measured width. ⚠️  The demo replays "
             f"RE-ORIENTED by the angle between the human's hand and the decode anchor (the page's 'transport' "
             f"line) — physical, see hl_replay._body_vs_world_transport.")
        t_ext = replay.attrs.get("T_tracker_tcp")
        if t_ext is not None:
            _log(f"replay: dataset puck→TCP extrinsic = {np.round(np.asarray(t_ext, dtype=np.float64)[:3, 3], 4).tolist()}")

    # --- 5b. the WALK SKILL (extension; always on) --------------------------------------
    # Resolved HERE, before a socket is opened, so a clip that is neither an extracted feet NPZ nor a
    # holosoma motion NPZ is refused by name at startup rather than on the first press of the button.
    if not (walk_skill.SPEED_MIN <= float(args.walk_speed) <= walk_skill.SPEED_MAX):
        raise SystemExit(f"--walk-speed {args.walk_speed} is outside "
                         f"[{walk_skill.SPEED_MIN:g}, {walk_skill.SPEED_MAX:g}]")
    if float(args.walk_hz) <= 0.0:
        raise SystemExit(f"--walk-hz {args.walk_hz} must be positive")
    args.walk_clip = os.path.expanduser(str(args.walk_clip))
    try:
        walk_clip = walk_skill.load_walk_clip(args.walk_clip)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--walk-clip {args.walk_clip}: {exc}") from exc
    st0 = walk_clip.stats()
    ready_feet, _ = walk_skill.feet_from_frame(wire.idle())
    gaps = st0.lead_in_gap_m(ready_feet)
    _log(f"walk skill: {walk_clip.describe()} (from {args.walk_clip})")
    _log("walk skill: " + st0.describe(
        speed=float(args.walk_speed), budget_m_per_tick=hl_5pt.LOW_LEVEL_SLEW_M_PER_TICK,
        budget_source="pre-flight estimate; the page re-scores against the MEASURED cap",
        publish_hz=float(args.walk_hz), feet_slewed=None))
    if walk_clip.has_joints:
        _log(f"walk skill: LEGS {'ON' if args.walk_legs else 'OFF (--no-walk-legs)'} — the clip's "
             f"{walk_skill.N_JOINTS} joints ride every walk frame as the wire's `ref_joints` (by name), so "
             f"the low level's command_lower_body is the clip's gait, not the frozen idle stance; the low "
             f"level confirms per tick on RecordTap (`ref_joints_live`, needs REF_JOINTS=1, its default).")
    else:
        _log(f"walk skill: LEGS unavailable — {walk_clip.legs_note}; the walk is feet-only (the "
             f"configuration that stepped in place on 2026-09-07).")
    _log(f"walk skill: lead-in from the ready feet to the clip's frame 0 is "
         f"{gaps[0] * 1e3:.0f} / {gaps[1] * 1e3:.0f} mm (L / R), ramped at "
         f"{float(args.walk_ramp_speed):g} m/s; the feet go on the wire ABSOLUTELY (the clip's own "
         f"pelvis-frame stance), the upper three slots hold the wire's last command, and the "
         f"clutch base follows every published frame. The low level must exempt the feet from its "
         f"slew caps (3_run_policy_vive_force.sh: SLEW_FEET=0, its default) — the page reads "
         f"whether it did off RecordTap (vr_cmd_slewed) and says so in 'clip demand'.")
    if abs(st0.yaw_range_deg) > 60.0:
        _log(f"⚠️  walk skill: this clip's heading spans {st0.yaw_range_deg:.0f}° — the low level "
             "pins the robot's heading to its frozen reference clip (motion_ref_ori_b), so a "
             "turning clip is fought by that term.")

    # --- 5b. reach compensation on the HANDS (default OFF; see hl_5pt.HandReach) ---------
    try:
        reach_cfg = ReachConfig(ki=float(args.reach_ki), clamp_m=float(args.reach_clamp),
                                stall_frac=float(args.reach_stall_frac), window=int(args.reach_window),
                                gate_m=float(args.reach_gate), decay=float(args.reach_decay),
                                contact_n=float(args.reach_contact_n))
    except ValueError as exc:
        raise SystemExit(f"--reach-*: {exc}") from exc
    reach = hl_5pt.HandReach(reach_cfg, roles, force_roles=loaded.layout.force_roles)
    if reach.enabled:
        _log(f"reach compensation ON for the HANDS {','.join(reach.roles)}: ki {reach_cfg.ki:g} · clamp "
             f"{reach_cfg.clamp_m * 100:.0f} cm · accumulate only while realized < "
             f"{reach_cfg.stall_frac:g} over {int(reach_cfg.window)} steps · decay {reach_cfg.decay:g} · "
             f"contact cut-out {reach_cfg.contact_n:g} N. The PUBLISHED hand target may exceed the "
             f"model's by up to {reach_cfg.clamp_m * 100:.0f} cm; the clutch base, the decode anchor "
             "and the observation stay uncompensated; identity while un-armed; the residual is "
             "dropped on STOP / Reset / disarm / a walk start / the page toggle. Feet, torso and "
             "held slots are untouched.")
        if reach.unprotected:
            _log(f"⚠️  reach compensation: {','.join(reach.unprotected)} has NO F/T channel on this "
                 "layout, so its contact cut-out can never engage — prefer a smaller --reach-clamp.")
    elif float(args.reach_ki) > 0.0:
        _log("reach compensation: --reach-ki given but this layout predicts no HAND — nothing to "
             "compensate; the wire is byte-identical to --reach-ki 0.")
    else:
        _log("reach compensation: OFF (--reach-ki 0) — the published wire is the clutched model "
             "target, byte-identical to before the feature. Start at --reach-ki 0.05 if the hands "
             "stop systematically short.")

    # --- 6. the bridges --------------------------------------------------------------
    # The run recorder's two bridge taps ride the subscribers' per-frame hook (`on_msg`, every frame
    # on the socket thread — `latest()` would decimate the ~250 Hz F/T stream to the 30 Hz poll).
    # `taps.recorder` is None until the page's RECORD block installs one, so each hook is one
    # `is None` check here.
    taps = rollout_record.RolloutTaps()
    subs = {
        "force": HL._WSSubscriber(args.force_ws, "force", on_msg=taps.on_force),
        "record": HL._WSSubscriber(args.record_ws, "record", on_msg=taps.on_record),
    }
    # The camera subscribers carry the recorder's per-VIEW hook the same way: `cams` is already the
    # checkpoint's own view count (trimmed above), so a one-camera checkpoint installs one hook and a
    # two-camera one two — the recording follows the subscriptions, nothing is configured twice.
    for i, url in enumerate(cams):
        subs[f"camera{i}"] = HL._WSSubscriber(url, f"camera{i}", on_msg=taps.camera_hook(i))
    # DISPLAY-ONLY subscriptions to the camera bridge(s) this checkpoint does NOT read (camera R under a
    # left_only model), for the page's "📷 Live cameras" folder alone — the operator lines the scene up
    # against the recorded first frame with BOTH eyes. Keyed `view{i}`, never `camera{i}`: the ingest
    # (`range(num_cameras)`), the self-check (`cams`) and the recorder (`camera{i}` keys) all stay on the
    # model's own views, so a dead camera R still refuses nothing and records nothing, exactly as before.
    if args.live_camera_hz > 0:
        for i, url in enumerate(_camera_urls(args)[len(cams):], start=len(cams)):
            subs[f"view{i}"] = HL._WSSubscriber(url, f"view{i}")
    grip_urls = {"left": args.gripper_ws, "right": args.gripper_right_ws}
    # BOTH jaws, not only the checkpoint's: the page's "Manual jaws" folder lets the operator open /
    # close either hand by hand, and a left_only run still has a RIGHT jaw. The self-check and the
    # policy's per-step command still read `layout.gripper_roles` only; an absent bridge here is a
    # subscriber that never connects (its readout says so), never a refusal.
    for role in ("left", "right"):
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
                          walk_clip=walk_clip, reach=reach, taps=taps, replay=replay)
    finally:
        s0.stop()
        for sub in subs.values():
            sub.stop()
        if replay is not None:
            try:
                replay.close()
            except Exception as exc:  # noqa: BLE001 - a zip handle on the way out
                _log(f"replay close error: {exc!r}")


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------
def _run_viser(args, loaded, planner, wire, subs, s0, anchor, consumes_force, report, *,
               walk_clip=None, reach=None, taps=None, replay=None) -> int:
    import viser
    from viser.extras import ViserUrdf

    roles = tuple(loaded.layout.roles)
    grip_roles = tuple(loaded.layout.gripper_roles)
    force_roles = tuple(loaded.layout.force_roles)
    # The checkpoint's own state band, for the `state range` readout and the per-Predict warning
    # (see main(); None when the checkpoint bundled no state statistics, which the report says).
    state_stats = loaded.norm_stats.get("state") if loaded.norm_stats else None
    if reach is None:   # a caller that did not build one (tests) gets the identity
        reach = hl_5pt.HandReach(ReachConfig(), roles, force_roles=force_roles)
    #: The operator's STOP: interrupts a drip / stream / walk within one step, and every run clears
    #: it at its own entry. ⚠️ NOT the process's shutdown flag — that is `shutdown`, below. The two
    #: were ONE Event, and pressing STOP then ended the heartbeat, the observation ingest and the IK
    #: worker for the rest of the process: every Predict after a STOP sampled from the image / force
    #: / jaw frozen at the moment of the press (the stream's sampler never checks freshness, so it
    #: was silent; the sequential Predict only logged "stale"), the wire stopped being re-published
    #: between chunks, and Reset's `stop.clear()` could not revive threads that had already returned.
    #: Found on the rig 2026-09-07 as "the first run is fine, everything after a STOP is wrong".
    stop = threading.Event()
    #: Set once, on the way out (Ctrl-C). The three daemon loops exit on THIS and only this.
    shutdown = threading.Event()
    dripping = threading.Event()
    busy = threading.Lock()
    grip_enabled = threading.Event()
    if args.command_gripper:
        grip_enabled.set()

    wire_lock = threading.Lock()
    # The published wire, seeded at the READY FRAME so the very first heartbeat gives the low
    # level's keypoint pre-arm gate exactly the reference it compares against.
    last_wire = {"frame": wire.idle(), "fcmd": None, "ref_joints": None}
    #: The ONE hook a wire-level extension may install on the publish funnel (`_send`, below): run on
    #: every frame — the drip's, the stream's, the heartbeat's — just before the gate, and given the
    #: frame + its `ref_joints`, it returns the pair to publish. `None` (the default, and the page as
    #: reviewed) is byte-identical. The walk skill's LEG OVERRIDE (its own block, far below) installs
    #: its substitution here while the clip has the two foot slots and removes it once the legs are
    #: handed back; nothing else writes here, and nothing above that block reads a name it defines.
    wire_hooks: dict = {"legs": None}
    #: The RUN RECORDER's taps (rollout_record.RolloutTaps): called at the publish funnel, the predict,
    #: the two execute loops and the ▶▶ start — each one `is None` check until the RECORD block (its
    #: own block, far below) installs one, and a few small copies + one non-blocking put after.
    #: Nothing here reads that object back; the block owns the button and the folder.
    if taps is None:
        taps = rollout_record.RolloutTaps()
    force_lock = threading.Lock()
    force_ctl = {"enabled": bool(args.enable_force_cmd), "gate_n": float(args.force_gate_n)}
    # Snapped again (main() already did) so THIS function owns the invariant "the dropdown's
    # initial_value is one of its options" no matter who calls it.
    exec_ctl = {"n": snap_exec_steps(args.exec_steps, _EXEC_CHOICES)}
    auto_lock = threading.Lock()
    auto_ctl = {"running": False, "gen": 0, "max_s": float(args.auto_max_seconds)}
    auto_running = threading.Event()
    # The CONTINUOUS cadence's live settings. Deliberately shares `auto_lock`/`auto_ctl`/
    # `auto_running` with the sequential rollout: ⏹, STOP, the seconds ceiling and the
    # "manual Predict/Run/Reset are refused while a run is active" rule are then ONE
    # implementation for both cadences rather than two that can drift apart.
    stream_ctl: dict = {"mode": str(args.ensemble), "tau": float(args.ensemble_tau),
                        "replan_min": max(0, int(args.replan_min_steps)),
                        "ghost": bool(args.stream_ik_ghost), "note": "idle",
                        "stats": None, "stream": None}
    #: The one integer that makes a chunk's rows addressable in absolute time — see
    #: :class:`hl_stream.StepClock`. Built once so a second streamed run cannot start on a stale one.
    step_clock = hl_stream.StepClock()
    #: RECORDED-EPISODE REPLAY state (extension; `replay` is None without --replay-zarr, and then
    #: `source` stays "model" and every hook below is one inert dict read). `source` decides what
    #: `_predict` stages; `episode` / `frame` are the cursor; `mode` the execution model.
    replay_lock = threading.Lock()
    replay_ctl: dict = {
        "source": "replay" if replay is not None else "model",
        "episode": int(getattr(args, "replay_episode", 0) or 0) if replay is not None else 0,
        "frame": 0, "mode": str(getattr(args, "replay_mode", "sequential") or "sequential"),
        "n_played": 0}
    if replay is not None:
        replay_ctl["frame"] = int(replay.first_usable_frame(replay_ctl["episode"]) or 0)
    replay_ui: dict = {}       # the panel's handles, filled when the panel is built (far below)
    replay_paths: dict = {}    # the green recorded-episode line(s)
    staged: dict = {"out": None}
    # THE SAMPLER SEED, and it is the only input to a Predict that Reset used not to restore. Kept
    # in an importable class (hl_5pt.SeedCounter) rather than a dict + two closures for the reason
    # its docstring gives: this planner samples from TWO places — `_predict` and the streaming
    # cadence's `_sample_once` — and a site that forgot the pin would look like the pin working
    # while that path's chunks drifted. It also makes the behaviour unit-testable without a GPU.
    seed_ctl = hl_5pt.SeedCounter(args.seed)
    stats = {"n": 0, "ms": 0.0, "span": "", "err": ""}
    #: How far the COMMANDED anchor is running ahead of the measured keypoints, and how often that
    #: lead had to be pulled back (`_belief_bound`). Reported on the page beside the clutch's own
    #: lead, because the two are the same quantity on two different leashes and the whole bug was
    #: that only one of them had one.
    belief = {"lead_m": 0.0, "n_clamp": 0, "t_log": 0.0}
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
    #: The two wire hands, for the MANUAL jaw controls — independent of `grip_roles` (what the policy
    #: predicts). `manual_jaw_touched` is the set the shutdown release adds to the policy's roles.
    MANUAL_JAW_ROLES = ("left", "right")
    manual_jaw_touched: set = set()
    #: THE PRESS WINS (2026-09-18). `_manual_jaw` latches a hand here; `_drip` and the stream ask
    #: `intercept` before publishing the policy's width and DROP it while latched; the Reset branch
    #: is the ONLY release. The rule lives in hl_5pt so the tests drive the shipped object.
    jaw_override = hl_5pt.ManualJawOverride(grip_roles)

    # ---------------- readbacks the wire consumes ----------------
    # ⚠️ DEFINED BEFORE THE HEARTBEAT THREAD STARTS, not merely before its first use: the heartbeat
    # calls `_idle_sync_wire` -> `_record_frame` on its very first tick, and a `def` further down
    # the body has not run yet when that happens (observed in hl_serving_mc as
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

    def _measured_force_by_hand() -> dict:
        """``role -> |F| (N)`` for every hand the rig reads, or ``{}`` when the bridge did not
        answer. Per hand rather than the worst-of (``_measured_force_n``), because the reach
        compensator's contact cut-out is per hand: the LEFT pressing on the cart must not stop the
        RIGHT from insisting, and a free RIGHT must not let the LEFT press harder. An empty dict is
        "unknown", which ``HandReach`` treats as CONTACT for a hand this rig has a Bota for."""
        if not force_roles:
            return {}
        fmsg, _wall, _n = subs["force"].latest()
        bins = HL._pick_force_channels(fmsg, force_roles, not args.allow_3dof_force)
        if bins is None:
            return {}
        arr = np.asarray(bins, dtype=np.float64).reshape(-1, 6)
        if not np.all(np.isfinite(arr)):
            return {}
        return {role: float(np.linalg.norm(arr[i, :3])) for i, role in enumerate(force_roles)}

    def _reach_apply(frame):
        """The frame to PUBLISH for one step: the hands' reach residual over the clutched model
        target (``hl_5pt.HandReach``). Returns the SAME object when the feature is off or the page
        un-ticked it, so that path is byte-identical to before. Called ONCE per published step at
        the two publish seams (the drip and the stream), never for the heartbeat's re-sends and
        never for what ``wire.commit`` receives — the residual is published, not integrated into
        the base. The measurement is the clutch's own (``wire.measured_keypoints``), refused when
        the RecordTap frame is older than ``REACH_MEASURED_MAX_AGE_S`` (hold, not grow); un-armed
        drops the residual and publishes the target as is."""
        if not reach.live:
            return frame
        msg, wall, _n = subs["record"].latest()
        fresh = (msg is not None and bool(wall)
                 and (time.time() - float(wall)) <= hl_5pt.REACH_MEASURED_MAX_AGE_S)
        rframe = _record_frame() if fresh else None
        armed = None if rframe is None else bool(rframe.vive_armed)
        measured = wire.measured_keypoints(rframe) if rframe is not None else None
        return reach.apply(frame, measured=measured, armed=armed,
                           force_by_hand=_measured_force_by_hand())

    def _reach_reset(reason: str) -> None:
        """Drop the hands' residual — the INTENT was abandoned (STOP, Reset, a walk taking the hands
        over, the page toggle). A residual that survived would be applied to the next, unrelated
        motion, or (after a walk, whose held hands ARE the compensated frame and become the clutch
        base) counted twice."""
        if not reach.enabled:
            return
        had = any(v > 1e-3 for v in reach.residual_mm().values())
        reach.reset()
        if had:
            _log(f"reach: residual dropped — {reason}.")

    def _gate_force(wrench_rows: dict) -> np.ndarray | None:
        """Per-hand predicted wrenches (``role -> row | None``) -> the ``(6,)`` ``[L, R]``
        sensor-frame F^cmd, or None.

        The arithmetic is ``hl_5pt.force_command`` — the same two-level contract and the same sign
        flip as the other planners (the model predicts ``F_meas``, the low level closes
        ``e = F^cmd + F_meas -> 0``, so the command is ``-F_meas``; gated on the MAGNITUDE). ALWAYS
        both hands wide: ``ViveLiveSource`` reads a (6,) f_cmd as ``[L, R]`` and a hand this
        checkpoint does not predict (``left_only``) must read as ZERO force, not as whatever the low
        level last had. The switch is read here so it stays the one live control.
        """
        with force_lock:
            enabled, gate_n = force_ctl["enabled"], force_ctl["gate_n"]
        if not enabled or not force_roles:
            return None
        return hl_5pt.force_command(wrench_rows, gate_n)

    def _set_wire(frame, fcmd, ref_joints=None) -> None:
        """Record what the heartbeat re-sends. ``ref_joints`` (the walk skill's reference legs)
        is part of that record: a frame stored WITHOUT it drops the legs from the wire, which is what
        every chunk step, Reset and the end of a walk do; a STOP mid-walk keeps them, so the frozen
        wire stays one consistent body (feet AND legs mid-stride) until the next command."""
        with wire_lock:
            last_wire["frame"] = tuple(np.asarray(a, dtype=np.float64).copy() for a in frame)
            last_wire["fcmd"] = None if fcmd is None else np.asarray(fcmd, dtype=np.float64).copy()
            last_wire["ref_joints"] = None if ref_joints is None else (
                tuple(ref_joints[0]), np.asarray(ref_joints[1], dtype=np.float64).copy(),
                np.asarray(ref_joints[2], dtype=np.float64).copy())

    def _send(frame, fcmd, ref_joints=None) -> bool:
        """Publish ONE 5pt frame. False (and nothing sent) if unsafe.

        The gate lives in the wire object and is applied here and only here, at the last moment
        before the socket: the low level validates only WIDTH and ``json.dumps`` serializes NaN
        unchecked, so a bad keypoint would otherwise reach the robot. On a refusal we send NOTHING
        and let the freshness gate (S0Driver 0.5 s -> the low level's 0.25 s) hold.
        ``ref_joints`` (None on every frame but the walk skill's) rides the same frame. With the
        LEG OVERRIDE engaged (`wire_hooks["legs"]`) the frame's two foot slots and its legs are
        replaced here, on the way to the gate, whoever the caller is.
        """
        hook = wire_hooks["legs"]
        if hook is not None:
            frame, ref_joints = hook.apply(frame, ref_joints)
        ok, reason = wire.publish(s0, frame, fcmd, ref_joints=ref_joints)
        if not ok:
            _log(f"REFUSED a {wire.mode} frame: {reason} — sending nothing; the stale gate holds.")
        # The run record gets the frame AS PUBLISHED (hook applied), after the gate's verdict.
        taps.publish(frame, fcmd, ok, override=hook is not None, ref_legs=ref_joints is not None)
        return ok

    def _heartbeat() -> None:
        """Re-publish the held wire. NO measured tracking on a keypoint wire, ever.

        ``FivePointWire.measured`` returns ``None`` by design and the dense planner's
        ``_idle_sync_wire`` has no counterpart here: the keypoint pre-arm gate's reference is the
        low level's FROZEN reference clip, not the robot's encoders, so chasing measured would move
        the wire AWAY from what the gate wants. The dense wire tracks because its own idle hold was
        the subject of two live bugs; the difference is stated rather than smoothed over.
        """
        period = 1.0 / max(args.drip_hz, 1.0)
        while not shutdown.is_set():
            if not dripping.is_set():
                try:
                    with wire_lock:
                        frame = tuple(a.copy() for a in last_wire["frame"])
                        fc = None if last_wire["fcmd"] is None else last_wire["fcmd"].copy()
                        rj = last_wire["ref_joints"]
                    # The switch is authoritative on EVERY tick, so turning it off (or Stop) makes
                    # the wire pure position within one heartbeat.
                    with force_lock:
                        on = force_ctl["enabled"]
                    _send(frame, fc if on else None, ref_joints=rj)
                except Exception as exc:  # noqa: BLE001 - the heartbeat must never die
                    _log(f"heartbeat error: {exc!r}")
            time.sleep(period)

    threading.Thread(target=_heartbeat, daemon=True, name="hb").start()

    # ---------------- observation ingest ----------------
    def _ingest_loop() -> None:
        period = 1.0 / max(args.obs_hz, 1.0)
        require_6dof = not args.allow_3dof_force
        while not shutdown.is_set():
            t0 = time.monotonic()
            try:
                for i in range(loaded.num_cameras):
                    msg, _wall, _n = subs[f"camera{i}"].latest()
                    if msg is None:
                        continue
                    frame = HL._decode_jpeg_b64(msg.get("jpeg_b64"))
                    if frame is None:
                        continue
                    # NATURAL channel order, no flip: `_decode_jpeg_b64` already converts cv2's BGR to
                    # RGB, and the training zarr's `camera*_rgb` IS RGB (the converter's `jpeg_decode`
                    # does the same BGR->RGB). Until 2026-09-20 this line flipped the frame a SECOND
                    # time and the model saw the rig in BGR: in a live wipe_board_new frame the red
                    # apple was 0.83% of the pixels and became 0.83% BLUE, a colour the 181 training
                    # episodes never contain (0.21% red, 0.00% blue; runbook §6). The mock rig's grey
                    # frame could not show it; the page's "📷 Live cameras" folder now draws this exact
                    # crop so a swapped channel order is on the page. hl_serving.py never flipped.
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
                # Family-agnostic: the geometric planner pushes its position ring as well as the
                # rotation + gripper rings (`push_proprio_geom` — pushing only the poses would leave
                # both gripper columns at their __init__ zeros, the CLOSED end of a legal range),
                # the wrt_start planner pushes the base class's two rings.
                planner.push_proprio()
            except Exception as exc:  # noqa: BLE001 - ingest must survive a bad frame
                _log(f"ingest error: {exc!r}")
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    threading.Thread(target=_ingest_loop, daemon=True, name="ingest").start()

    # ---------------- the scene ----------------
    server = viser.ViserServer(host=args.host, port=args.port,
                              label="FLoRe FIVE-POINT planner (5 keypoints -> 5pt)")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", plane="xy", shadow_opacity=0.0, infinite_grid=True,
                          cell_size=0.25)
    # ONE WORLD, floor at z=0. That is the frame `resolve_deploy_anchor` expresses its poses in, so
    # the drawn plan and the live mesh share it with no second transform. The LIVE mesh's pelvis is
    # pinned at `anchor.pelvis_z` (it has no root channel — RecordTap publishes joints); the violet
    # IK body carries its own floating root, because the CHEST is its root and a predicted chest
    # translation has to go somewhere.
    server.scene.add_frame("/robot", show_axes=False,
                           position=(0.0, 0.0, float(anchor.pelvis_z)))
    urdf = ViserUrdf(server, urdf_or_path=Path(args.yam_urdf), root_node_name="/robot")
    dof_remap = build_sdk_to_urdf_index(list(urdf.get_actuated_joint_names()))
    if dof_remap.missing_sdf:
        _log(f"⚠️  URDF is missing SDK joints {dof_remap.missing_sdf} — the live mesh is disabled "
             "rather than drawn with a limb in the wrong place")
    urdf.update_cfg(remap_dof_to_urdf(anchor.dof, dof_remap))

    ghost_root = ghost = ik = None
    #: Owns the "which solved body may be on screen" invariant — see :class:`hl_5pt.PreviewGate`
    #: for why a generation token rather than just a slot, and for the live bug it fixes.
    preview = hl_5pt.PreviewGate()
    ik_q: queue.Queue = queue.Queue(maxsize=1)
    #: Newest streamed chunk awaiting a redraw. Depth 1, and the PAGE LOOP is its only consumer:
    #: `paths` is written by `_draw_paths` and read by the triads checkbox on viser's callback
    #: thread, so keeping one writer means a streamed run adds no new race to a dict the rest of
    #: this file already treats as page-loop-owned.
    draw_q: queue.Queue = queue.Queue(maxsize=1)

    def _retire_ik() -> None:
        """Retire the staged body and drain a QUEUED solve, so neither can land after a Reset."""
        preview.retire()
        while True:
            try:
                ik_q.get_nowait()
            except queue.Empty:
                break

    # GIL PRIORITY FOR THE SAMPLER, and this is not an optimisation. The IK worker is PURE PYTHON
    # (yourdfpy FK + a 6x6 solve, thousands of times per chunk) while `sample_chunk` is a Python
    # loop issuing CUDA launches — so the two fight for one interpreter and the GPU goes idle
    # waiting for a thread that cannot get it. MEASURED ON THIS PAGE: a first Predict with the
    # worker idle costs 380 ms; three sequential Predicts with it solving cost 3758 ms — a 9.9x
    # slowdown, the same failure and the same magnitude as `hl_serving`'s single-arm ghost
    # (46 -> 3543 ms). The gate makes the worker stand back for the duration of a sample; it is
    # checked PER STEP inside `solve_chunk`, so a Predict never waits on more than one step's solve.
    infer_gate = YieldGate()
    if args.ik_ghost:
        ik, why = KeypointIK.load(urdf_path=args.yam_urdf)
        if ik is None:
            _log(f"⚠️  the violet IK body is unavailable ({why}); the plan still draws as five "
                 "paths.")
        else:
            mm, deg = ik.check_round_trip(anchor.dof)
            _log(f"IK round-trip on the decode anchor: {mm:.4f} mm / {deg:.4f} deg "
                 f"({'OK' if mm < 1.0 else 'HIGH — the violet body would sit off the plan'})")
            ghost_root = server.scene.add_frame("/ghost", show_axes=False, visible=True)
            ghost = ViserUrdf(server, urdf_or_path=Path(args.yam_urdf), root_node_name="/ghost",
                              mesh_color_override=_GHOST_RGBA)

    def _reset_ghost() -> None:
        """Put the violet body back at the ready pose — **root included**.

        ⚠️ An IK solution is 29 joints PLUS a floating root (the chest is the root), so restoring
        only the joints leaves the violet body standing wherever the last chunk ended, translated
        and yawed. Same bug ``hl_serving_mc._reset_ghost`` documents, same fix, and for the same
        reason no gate or wire check would catch it: the root is display-only. Called at the
        construction site AND in the Reset branch so the two cannot drift.

        Retires the IK generation FIRST (see :func:`_retire_ik`): parking the body while a solve is
        still allowed to publish just means the solve moves it straight back.
        """
        _retire_ik()
        if ghost is None or ghost_root is None:
            return
        ghost.update_cfg(remap_dof_to_urdf(anchor.dof, dof_remap))
        ghost_root.position = (0.0, 0.0, float(anchor.pelvis_z))
        ghost_root.wxyz = (1.0, 0.0, 0.0, 0.0)

    _reset_ghost()

    paths: dict = {}
    #: Read by ``_draw_paths`` at draw time. A plain dict rather than a closure over the GUI
    #: handle, because the checkbox is created below this and the page loop is not the only caller.
    view_ctl: dict = {"triads": True}

    def _draw_paths(chunk) -> None:
        """All FIVE decoded keypoint paths, in the anchor's world frame.

        Every role gets a polyline, a point per predicted step and an orientation triad every
        ``stride`` steps, sized per role by :data:`_PLAN_STYLE`. The reason is the same one
        ``hl_serving_mc._draw_paths`` documents and it applies to THREE of the five here: the chest
        sits inside the torso mesh and both feet sit under the body, all three translate
        centimetres or millimetres where a hand travels tens of centimetres, so a bare line reads
        as "never predicted". For the feet the triad is also the channel's real content — the
        measured foot signal is the operator's YAW (10.7 deg median per episode), not translation.
        """
        want_triads = bool(view_ctl["triads"])
        for rc in chunk.roles:
            role = rc.role
            st = _PLAN_STYLE.get(role, _PLAN_STYLE_FALLBACK)
            rgb = np.array(st["rgb"], np.uint8)
            pts = np.asarray(rc.position, dtype=np.float32)
            # REMOVE BEFORE ADDING, always. viser addresses nodes by NAME, so adding
            # `/plan/left/line` while the previous handle still holds that name replaces the node —
            # and the later `.remove()` on the stale handle would then delete the one just drawn,
            # leaving a role that vanishes on its SECOND chunk.
            for kind in ("line", "dots", "triads"):
                old = paths.pop(f"{role}:{kind}", None)
                if old is not None:
                    try:
                        old.remove()
                    except Exception:  # noqa: BLE001, S110 - a stale handle must not kill the draw
                        pass
            seg = np.stack([pts[:-1], pts[1:]], axis=1)
            paths[f"{role}:line"] = server.scene.add_line_segments(
                f"/plan/{role}/line", points=seg,
                colors=np.broadcast_to(rgb, (seg.shape[0], 2, 3)).copy(),
                line_width=4.0 if role == "head" else 3.0)
            paths[f"{role}:dots"] = server.scene.add_point_cloud(
                f"/plan/{role}/dots", points=pts,
                colors=np.broadcast_to(rgb, pts.shape).copy(),
                # float32, not the default float16: at z ≈ 1.15 m a half float quantises to about
                # a millimetre, which is a visible stagger across a 26 mm chest excursion.
                point_size=0.006, point_shape="circle", precision="float32")
            idx = _triad_indices(len(pts), int(st["stride"]))
            rotmat = rot6d_to_rotmat(np.asarray(rc.rot6d))[idx]
            paths[f"{role}:triads"] = server.scene.add_batched_axes(
                f"/plan/{role}/triads",
                batched_wxyzs=quat_xyzw_to_wxyz(rotmat_to_quat_xyzw(rotmat)).astype(np.float32),
                batched_positions=pts[idx],
                axes_length=float(st["axes"]), axes_radius=0.004, visible=want_triads)

    def _set_triads_visible(show: bool) -> None:
        view_ctl["triads"] = bool(show)
        # list(), not the live dict: this runs on viser's callback thread while `_draw_paths` may
        # be replacing entries on the page loop, and iterating a dict being resized raises.
        for key, handle in list(paths.items()):
            if key.endswith(":triads"):
                try:
                    handle.visible = bool(show)
                except Exception:  # noqa: BLE001, S110 - a stale handle must not kill the callback
                    pass

    def _clear_paths() -> None:
        for key in list(paths):
            handle = paths.pop(key)
            try:
                handle.remove()
            except Exception:  # noqa: BLE001, S110
                pass

    # ---------------- the violet IK body ----------------
    def _ik_worker() -> None:
        """Solve the staged chunk's five poses into a body. Its OWN KeypointIK, on its own thread.

        A ``yourdfpy.URDF`` carries the configuration it was last set to and this mutates it
        thousands of times per chunk, so sharing the startup instance would race the round-trip
        check and the render thread. ~0.3-1.0 s per 50-step chunk, which is why it is not on the
        drip path: the IK is DISPLAY ONLY and must never delay a published step.
        """
        solver, why = KeypointIK.load(urdf_path=args.yam_urdf)
        if solver is None:  # pragma: no cover - the startup load already succeeded
            _log(f"IK worker could not load its own solver ({why}); the violet body is disabled.")
            return
        while not shutdown.is_set():
            try:
                payload = ik_q.get(timeout=0.2)
            except queue.Empty:
                continue
            gen = int(payload["gen"])
            try:
                out = solver.solve_chunk(payload["poses"], seed_dof=anchor.dof,
                                         jaw_m=payload["jaw"], yield_gate=infer_gate)
            except Exception as exc:  # noqa: BLE001 - a failed solve hides the body, never kills
                _log(f"IK solve failed: {exc!r}")
                preview.give_up(gen)
                continue
            # Refused outright if a Reset (or a newer Predict) retired this generation while it was
            # solving. Compare-then-publish as two separate steps is the whole bug.
            preview.publish(gen, out, out.describe())

    if ik is not None:
        threading.Thread(target=_ik_worker, daemon=True, name="ik").start()

    def _queue_ik(chunk) -> None:
        # Bumping the generation here retires any solve still running on the PREVIOUS chunk, so
        # only one body can ever land. The previous body deliberately stays ON SCREEN until the new
        # solve arrives (a 200-step solve outlasts the drip, so blanking it would leave the page
        # bodyless for most of a sequential run) — `pending` is what stops that from reading as
        # current, and the readout says which chunk is being shown.
        if ik is None:
            return
        gen = preview.submit()
        # ALL FIVE roles — the HELD slots filled from the wire's own ready-pose rows, so a
        # left_only chunk still draws a whole body (the four held points sit exactly where the
        # live mesh holds them). `wire.all_rows` is the same fill `wire.plan` publishes from.
        try:
            rows = wire.all_rows(chunk)
        except ValueError as exc:
            _log(f"IK body skipped: {exc}")
            preview.give_up(gen)
            return
        poses = {role: (np.asarray(row["position"], dtype=np.float64),
                        rot6d_to_rotmat(np.asarray(row["rot6d"], dtype=np.float64)))
                 for role, row in rows.items()}
        jaw = {rc.role: rc.gripper_m for rc in chunk.roles if rc.gripper_m is not None}
        try:
            ik_q.get_nowait()          # depth-1: a superseded chunk is dropped, not queued
        except queue.Empty:
            pass
        try:
            ik_q.put_nowait({"gen": gen, "poses": poses, "jaw": jaw})
        except queue.Full:             # the newest chunk wins on the next Predict
            pass

    # ---------------- the belief's own leash ----------------
    def _belief_bound() -> None:
        """Pull the PLANNER's anchor back to within ``--obs-leak-m`` of the measured keypoints.

        THE ASYMMETRY THIS REMOVES. ``KeypointClutch.sync`` already bounds the clutch BASE — what is
        published — to within ``--post-anchor-eps-m`` of measured at every Predict. The planner's
        anchor, which under ``obs_mode=last_target`` IS the whole observation, had no such bound: it
        advances by the model's own executed step (``commit_role``) and nothing pulled it back. Two
        rig runs are the consequence — the wire stayed near the robot while the belief walked away
        from both, so every later chunk was sampled from a hand position the arm had never reached,
        and the body folded around it. Same leash, same numbers, applied to the belief:
        :meth:`hl_5pt.FivePointPlanner.leak_anchor_toward`.

        Called immediately BEFORE the range check and the sample, so what gets scored and what gets
        conditioned on are the corrected belief. It is a no-op inside the bound (a healthy run is
        byte-identical) and a no-op with no RecordTap (the anchor is held, never corrected toward a
        guess — ``sync``'s own rule). It covers BOTH families: ``FivePointPlanner`` and, since
        2026-09-14, ``SubsetKeypointPlanner`` (:meth:`hl_5pt.SubsetKeypointPlanner.leak_anchor_toward`)
        — this docstring used to call a ``wrt_start`` state immune, and a ``left_only`` run then
        walked its ROTATION 20° past the robot (``left_r6d0`` 1.65 band widths out) with the wire
        itself bounded. ⚠️ Under ``--post-anchor command`` it cannot change what a given chunk publishes (the
        clutch's ``p[i] - p[0]`` is invariant to a shift of the decode anchor) — it changes which
        chunk is sampled next, which is where the runaway lived.
        """
        if float(args.obs_leak_m) <= 0 or not hasattr(planner, "leak_anchor_toward"):
            return
        try:
            # ONE FRAME. `measured_keypoints` is the WIRE (pelvis) frame the clutch bounds against;
            # the planner's anchor is the WORLD frame it was seeded in (floor z = 0). Compared raw,
            # every role read pelvis_z = 757 mm of lead at the very FIRST Predict with the robot on
            # the ready pose (mock rig, 2026-09-18), and the leash dragged the whole belief 0.7 m
            # down. The startup check below proves the conversion on the ready pose itself.
            measured = hl_5pt.wire_frame_to_anchor(wire.measured_keypoints(_record_frame()),
                                                    wire.readout)
            lead, clamped = planner.leak_anchor_toward(
                measured, eps_m=float(args.obs_leak_m),
                eps_rad=float(np.radians(args.obs_leak_deg)))
        except Exception as exc:  # noqa: BLE001 - a bad readback must never lose the chunk
            _log(f"belief bound skipped: {exc!r}")
            return
        belief["lead_m"] = float(lead)
        if not clamped:
            return
        belief["n_clamp"] += 1
        now = time.monotonic()
        if now - belief["t_log"] > 2.0:      # rate-limited: true for a whole stretch of a run
            belief["t_log"] = now
            _log(f"belief re-based: the commanded anchor had run {lead * 1e3:.0f} mm ahead of the "
                 f"measured keypoints — pulled back to {float(args.obs_leak_m) * 1e3:.0f} mm "
                 f"(--obs-leak-m), so the next chunk conditions on where the robot IS. "
                 f"{belief['n_clamp']} time(s) this run.")

    # ---------------- the observation's own provenance ----------------
    def _obs_range_gate() -> str | None:
        """Score the observation the NEXT sample would condition on. ``None`` = go ahead.

        ALWAYS LOGGED. Historically that was all it did — "an out-of-band observation is a warning
        about the chunk's provenance, and the operator has the E-stop" — and two rig runs showed why
        that is not enough on its own. While the low level is ARMED the observation is a pure
        function of the accumulated COMMAND (``obs_mode=last_target``: ``commit_role`` writes the
        EXECUTED step into ``_anchor_pos`` and ``snapshot_obs`` derives every state column from it,
        so no measured quantity enters it), while the robot is bounded to 8.33 mm/step over one L2
        pool of all covered points. A command the arm cannot reach therefore becomes an observation
        the robot is not IN, and the next chunk is sampled from that — monotone and self-amplifying,
        which is what both logs are. The operator's E-stop is not the right bound for a loop that
        diverges in ~5 chunks (1.7 s of wall clock at the default 16 exec steps).

        So ``--ood-stop BANDS`` (0 = the old behaviour) refuses BEFORE the sample. Refusing here is
        what leaves the robot where it got to: nothing new is staged, the sequential rollout takes
        its existing "nothing staged -> end the run" branch, the idle heartbeat keeps re-publishing
        the LAST EXECUTED frame, and the decode anchor + clutch base are still on that same step —
        exactly the state a STOP leaves, so ▶▶ with no Reset continues from there.

        ⚠️ It is a bound on the OBSERVATION, not on the wire. It cannot catch a chunk that diverges
        without the state noticing (the residual ambiguity of a translation-invariant state: a
        ``lh_dx`` this far out cannot say whether the hand went forward or the chest went back), and
        it is not a substitute for the hardware E-stop.
        """
        try:
            names_s, row_s, _w = planner.state_readout()
            rng_s = hl_5pt.state_range_report(names_s, row_s, state_stats)
        except ValueError as exc:
            _log(f"observation range check skipped: {exc}")
            return None
        if rng_s.ok or not rng_s.checked:
            return None
        col, bands = rng_s.worst_body_excursion()
        limit = float(args.ood_stop)
        if limit <= 0 or bands <= limit:
            _log(f"⚠️  sampling from an observation {rng_s.describe()}")
            return None
        _log(f"⛔ REFUSED to sample: the observation is {rng_s.describe()}")
        _log(f"   {col} is {bands:.2f} band widths past its training range, over the "
             f"--ood-stop limit of {limit:g}. Holding at the LAST EXECUTED step — the wire, the "
             f"decode anchor and the clutch base are all still there, so ▶▶ continues from here "
             f"(no Reset). --ood-stop 0 turns this off.")
        return (f"REFUSED: observation {bands:.2f} band widths OUTSIDE the training range "
                f"({col}) — over --ood-stop {limit:g}; holding at the last executed step")

    # ---------------- predict ----------------
    # ---------------- RECORDED-EPISODE REPLAY: the chunk source (extension) ----------------
    # Everything here is inert without --replay-zarr (`replay is None` ⇒ `_replay_active()` is False
    # and `_predict` below is the page as reviewed). With it, `_predict_replay` builds the SAME
    # `staged["out"]` shape `_predict` does — a decoded ChunkResult + `wire.plan` — so `_drip`, the
    # commits, the taps, the paths and the IK body are one implementation for both sources.
    _REPLAY_GREEN = np.array((90, 220, 120), np.uint8)   # replay_zarr_viz.py's GT line colour

    def _replay_active() -> bool:
        with replay_lock:
            return replay is not None and replay_ctl["source"] == "replay"

    def _replay_len(episode: int) -> int:
        a, b = replay.episode_bounds(episode)
        return int(b - a)

    def _clear_replay_paths() -> None:
        with replay_lock:
            items = list(replay_paths.items())
            replay_paths.clear()
        for _key, handle in items:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001, S110 - a stale handle must not kill the draw
                pass

    def _draw_replay_paths(episode: int, ref_frame: int) -> None:
        """The WHOLE recorded episode, per predicted role, transported by the CURRENT anchor with
        the same `delta_ee_to_absolute` the chunk is decoded with — so the green line is where
        the hand will actually go, reorientation included, and a fat dot marks the cursor."""
        if replay is None:
            return
        _clear_replay_paths()
        if not bool(replay_ui.get("show_path", True)):
            return
        try:
            anchor_pos, anchor_rot6d = planner.snapshot_anchor()
            for ri, role in enumerate(roles):
                rel_pos, rel_rot6d = replay.episode_path_relative(episode, ref_frame, role)
                abs_pos, _ = delta_ee_to_absolute(rel_pos, rel_rot6d, anchor_pos[ri], anchor_rot6d[ri])
                pts = np.asarray(abs_pos, dtype=np.float32).reshape(-1, 3)
                if pts.shape[0] < 2:
                    continue
                seg = np.stack([pts[:-1], pts[1:]], axis=1)
                line = server.scene.add_line_segments(
                    f"/replay/{role}/line", points=seg,
                    colors=np.broadcast_to(_REPLAY_GREEN, (seg.shape[0], 2, 3)).copy(), line_width=2.5)
                dot = server.scene.add_point_cloud(
                    f"/replay/{role}/cursor", points=pts[int(ref_frame):int(ref_frame) + 1],
                    colors=_REPLAY_GREEN.reshape(1, 3).copy(), point_size=0.02, point_shape="circle",
                    precision="float32")
                with replay_lock:
                    replay_paths[f"{role}:line"] = line
                    replay_paths[f"{role}:cursor"] = dot
        except Exception as exc:  # noqa: BLE001 - display only
            _log(f"replay path draw skipped: {exc!r}")

    def _replay_refresh_panel(episode: int, frame: int, note: str | None = None) -> None:
        """Status line, the cursor's recorded frame, the episode's FIRST frame(s) (on an episode
        change) and the transport angle. Display only, fully guarded."""
        if replay is None or not replay_ui:
            return
        try:
            n = _replay_len(episode)
            if note is None:
                note = f"ep {episode} · frame {frame}/{n - 1} (t={replay.episode_times(episode)[frame]:.2f} s)"
            replay_ui["status"].value = note[:200]
            key0 = replay_ui["cams"][0]
            replay_ui["cursor_img"].image = replay.image_at(episode, frame, key0)
            if replay_ui.get("ep_shown") != episode:
                replay_ui["ep_shown"] = episode
                for key, handle in replay_ui["first_imgs"].items():
                    handle.image = replay.image_at(episode, 0, key)
            anchor_pos, anchor_rot6d = planner.snapshot_anchor()
            parts = []
            for ri, role in enumerate(roles):
                _p, rec_rot6d = replay.recorded_pose(episode, frame, role)
                parts.append(f"{role} {hl_replay.transport_rotation_deg(rec_rot6d, anchor_rot6d[ri]):.0f}°")
            replay_ui["transport"].value = ("re-oriented by " + " · ".join(parts)
                                            + " (human hand vs decode anchor; physical, not corrected)")
        except Exception as exc:  # noqa: BLE001 - display only
            _log(f"replay panel refresh skipped: {exc!r}")

    def _replay_seek(episode: int, frame: int, why: str = "") -> None:
        """Move the cursor (episode, frame), drop a staged chunk from the old position, redraw."""
        with replay_lock:
            replay_ctl["episode"] = int(episode)
            replay_ctl["frame"] = int(frame)
        staged["out"] = None
        _replay_refresh_panel(int(episode), int(frame),
                              note=(f"ep {episode} · frame {frame}/{_replay_len(episode) - 1}"
                                    + (f" — {why}" if why else "")))
        _draw_replay_paths(int(episode), int(frame))

    def _predict_replay() -> str:
        """The RECORDED chunk at the cursor, decoded and clutched exactly as a model chunk is.

        Skips the converter's masked frames (logged), ends the run at the episode's last frame
        (the cursor rewinds to the first playable frame so ▶▶ again replays it from the pose the
        arm is at NOW — the clutch semantic; Reset returns everything to the ready pose). No
        observation is consumed, so neither the freshness refusal nor the observation gate
        applies; the belief leash still runs, because the decode anchor is the same one."""
        with replay_lock:
            episode = int(replay_ctl["episode"])
            frame = int(replay_ctl["frame"])
            mode = str(replay_ctl["mode"])
        n_ep = _replay_len(episode)
        usable = replay.next_usable_frame(episode, frame) if frame < n_ep - 1 else None
        if frame >= n_ep - 1 or usable is None:
            first = int(replay.first_usable_frame(episode) or 0)
            with replay_lock:
                replay_ctl["frame"] = first
                replay_ctl["n_played"] += 1
                n_played = int(replay_ctl["n_played"])
            taps.event("replay_end", episode=int(episode), frame=int(frame), n_frames=int(n_ep),
                       reason="episode end" if frame >= n_ep - 1 else "no playable frame left")
            _replay_refresh_panel(episode, first, note=f"ep {episode} FINISHED ({n_ep} frames) — cursor rewound "
                                                        f"to {first}; ▶▶ replays it from where the arm is now")
            _draw_replay_paths(episode, first)
            return (f"REPLAY: episode {episode} finished ({n_ep} frames; play #{n_played}) — cursor rewound to "
                    f"frame {first}. ▶▶ again replays it from the CURRENT pose; Reset (un-armed) re-homes first.")
        if usable != frame:
            _log(f"replay: ep {episode} frames {frame}..{usable - 1} are masked in the recording — skipped.")
            frame = int(usable)
            with replay_lock:
                replay_ctl["frame"] = frame
        # The SAME leash the model path applies first: the decode anchor is pulled back toward the
        # measured keypoints, so the demo is re-based on where the robot IS, not where it was told.
        _belief_bound()
        t0 = time.monotonic()
        try:
            if mode == "openloop":
                plan_rec = replay.episode_plan(episode, frame)
                actions, n_valid = plan_rec.actions, int(plan_rec.n_frames)
                time_s, truncated = float(plan_rec.times_s[0]), bool(plan_rec.truncated)
                fsv, gsv = plan_rec.force_step_valid, plan_rec.gripper_step_valid
                g_valid, f_valid = plan_rec.gripper_valid, plan_rec.force_valid
            else:
                rc = replay.chunk_at(episode, frame)
                actions, n_valid, time_s, truncated = rc.actions, int(rc.n_valid), float(rc.time_s), False
                fsv, gsv = rc.force_step_valid, rc.gripper_step_valid
                g_valid, f_valid = rc.gripper_valid, rc.force_valid
            anchor_pos, anchor_rot6d = planner.snapshot_anchor()
            # Ungated here (force_enabled=True, gate 0): the LIVE gate is applied per step in _drip,
            # exactly as it is for a model chunk (`wrench_pred` rows -> `_gate_force`).
            chunk = decode_chunk(actions, loaded.layout, anchor_pos, anchor_rot6d, force_enabled=True, gate_n=0.0)
            chunk, blank = hl_replay.apply_step_validity(chunk, fsv, gsv)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not publish
            stats["err"] = f"{type(exc).__name__}: {exc}"
            return f"REPLAY FAILED at ep {episode} frame {frame}: {stats['err']} — nothing staged."
        t_read = (time.monotonic() - t0) * 1e3
        try:
            wire_src = wire.sync(_record_frame(), _measured_force_n())
        except Exception as exc:  # noqa: BLE001 - a bad readback must not lose the chunk
            _log(f"wire re-base skipped: {exc!r}")
            wire_src = "last published"
        plan = wire.plan(chunk)
        span = hl_5pt.chunk_span(plan["pos"], hl_dense.FIVE_POINT_LABELS, budget=wire.slew.budget,
                                 covered=wire.slew.slew_mask)
        _ok, gate = wire.gate_report(wire.frame(plan, 0))
        stats["n"] += 1
        stats["ms"] = t_read
        stats["span"] = span.describe()
        rp_info = {"episode": episode, "frame": frame, "mode": mode, "n_valid": n_valid, "n_ep": n_ep,
                   "time_s": time_s, "truncated": truncated, "gripper_valid": bool(g_valid),
                   "force_valid": bool(f_valid), "zarr": os.path.basename(replay.path),
                   "force_blanked": {k: int(v) for k, v in blank["force"].items()},
                   "gripper_filled": {k: int(v) for k, v in blank["gripper"].items()},
                   "gripper_dropped": list(blank["gripper_dropped"])}
        staged["out"] = {"chunk": chunk, "plan": plan, "wire_src": wire_src, "seq": stats["n"],
                         "replay": rp_info, "n_exec": (int(plan["n"]) if mode == "openloop" else None)}
        taps.chunk(seq=stats["n"], cadence="replay", decoded_pos=wire.decoded_positions(chunk),
                   pos=plan["pos"], orn=plan["orn"],
                   wrench={r: chunk.role(r).wrench_pred for r in force_roles},
                   grip={r: chunk.role(r).gripper_m for r in grip_roles},
                   seed=None, sample_ms=t_read, wire_src=wire_src,
                   n_exec=(int(plan["n"]) if mode == "openloop" else exec_ctl["n"]),
                   source="replay", replay={k: v for k, v in rp_info.items()
                                            if isinstance(v, (int, float, str, bool))})
        _draw_paths(chunk)
        _draw_replay_paths(episode, frame)
        # The violet body of a WHOLE episode would be minutes of background IK for one preview;
        # the per-chunk body is drawn as for a model chunk.
        if mode != "openloop" or int(plan["n"]) <= 4 * ACTION_HORIZON:
            _queue_ik(chunk)
        flags_txt = []
        if blank["force"]:
            flags_txt.append("wrench unmeasured on " + ", ".join(f"{k} {v} step(s)" for k, v in blank["force"].items())
                             + " → zero F^cmd there")
        if blank["gripper"]:
            flags_txt.append("jaw unmeasured on " + ", ".join(f"{k} {v} step(s)" for k, v in blank["gripper"].items())
                             + " → held")
        if blank["gripper_dropped"]:
            flags_txt.append("jaw unmeasured for the whole chunk on " + ",".join(blank["gripper_dropped"]) + " → not commanded")
        if truncated:
            flags_txt.append("TRUNCATED at a glitch hole")
        _replay_refresh_panel(episode, frame,
                              note=f"ep {episode} · frame {frame}/{n_ep - 1} (t={time_s:.2f} s) · {n_valid} step(s) staged"
                                   + (f" [{'; '.join(flags_txt)}]" if flags_txt else ""))
        return (f"REPLAY staged: ep {episode} frame {frame}/{n_ep - 1} (t={time_s:.2f} s) · {mode} · "
                f"{plan['n']} step(s) · read {t_read:.0f} ms · base {wire_src} · {span.describe()} · {gate}"
                + (f" · {'; '.join(flags_txt)}" if flags_txt else ""))

    def _predict() -> str:
        """Sample one chunk, decode all five roles, clutch them into wire frames, stage it.

        With the replay source selected (--replay-zarr + the page's Chunk SOURCE) the chunk is the
        RECORDED one instead (`_predict_replay`); everything after the decode is shared."""
        replaying = _replay_active()
        # The RUN RECORDER opens its folder at the FIRST Predict — a manual one or the one ▶▶ makes —
        # once the ⏺ button is armed (or on every_run), and stays open across chunks until STOP /
        # Reset / the low level disarming. A no-op while one is open; one `is None` check without it.
        taps.begin("replay" if replaying else "sequential")
        if replaying:
            return _predict_replay()
        reasons, _ = HL._obs_freshness(planner, max_age_s=planner.obs_max_age_s,
                                       grip_max_age_s=planner.grip_max_age_s,
                                       consumes_force=consumes_force)
        if reasons and args.obs_max_age is not None and float(args.obs_max_age) != 0:
            return "REFUSED: " + "; ".join(reasons)
        if reasons:
            _log("⚠️  stale observation: " + "; ".join(reasons))
        seed = seed_ctl.next()
        with force_lock:
            enabled, gate_n = force_ctl["enabled"], force_ctl["gate_n"]
        # Correct the belief FIRST, then score what will actually be conditioned on.
        _belief_bound()
        refuse = _obs_range_gate()
        if refuse is not None:
            return refuse
        t0 = time.monotonic()
        try:
            # Hold the interpreter against the IK worker for the whole sample — see `infer_gate`.
            with infer_gate.priority():
                chunk = planner.infer_chunk(seed=seed, force_enabled=enabled,
                                            gate_n=gate_n)
        except Exception as exc:  # noqa: BLE001 - a failed sample must not publish
            stats["err"] = f"{type(exc).__name__}: {exc}"
            return f"SAMPLE FAILED: {stats['err']} — nothing staged."
        t_sample = (time.monotonic() - t0) * 1e3
        # WHERE THE NEXT CHUNK'S WIRE CONTINUES FROM (--post-anchor, owned by the clutch):
        #   off      a keypoint wire is read out against the ready pose and not re-based; the
        #            clutch says so and `apply` is the identity.
        #   command  the last EXECUTED step is carried forward, leak-bounded against RecordTap's
        #            measured keypoints and decayed under contact. With NO measured frame it
        #            FREEZES rather than integrating blind.
        try:
            wire_src = wire.sync(_record_frame(), _measured_force_n())
        except Exception as exc:  # noqa: BLE001 - a bad readback must not lose the chunk
            _log(f"wire re-base skipped: {exc!r}")
            wire_src = "last published"
        plan = wire.plan(chunk)
        # `wire.slew.budget` is the cap MEASURED off RecordTap once the low level has been armed
        # long enough to show one, and the conservative pre-flight estimate until then — so the
        # printed ratio tracks a launcher edit with nothing to remember and nothing to re-tune.
        span = hl_5pt.chunk_span(plan["pos"], hl_dense.FIVE_POINT_LABELS, budget=wire.slew.budget,
                                 covered=wire.slew.slew_mask)
        _ok, gate = wire.gate_report(wire.frame(plan, 0))
        stats["n"] += 1
        stats["ms"] = t_sample
        stats["span"] = span.describe()
        staged["out"] = {"chunk": chunk, "plan": plan, "wire_src": wire_src, "seq": stats["n"]}
        taps.chunk(seq=stats["n"], cadence="sequential", decoded_pos=wire.decoded_positions(chunk),
                   pos=plan["pos"], orn=plan["orn"],
                   wrench={r: chunk.role(r).wrench_pred for r in force_roles},
                   grip={r: chunk.role(r).gripper_m for r in grip_roles},
                   seed=seed, sample_ms=t_sample, wire_src=wire_src, n_exec=exec_ctl["n"])
        _draw_paths(chunk)
        _queue_ik(chunk)
        # The seed is in the message, not only in the status text: this string is what the operator
        # copies into a log or a report, and a chunk without its seed is not reproducible.
        return (f"staged: seed {seed} · sample {t_sample:.0f} ms · {plan['n']} step(s) · "
                f"base {wire_src} · {span.describe()} · {gate}")

    # ---------------- execute ----------------
    def _drip(out) -> int:
        """Publish the first ``exec_ctl['n']`` steps, one per 30 Hz step."""
        chunk, plan = out["chunk"], out["plan"]
        # `n_exec` is set only by an OPEN-LOOP replay (the whole episode is one chunk and executes
        # in full); every other chunk executes the page's execute-N choice, as before.
        n_want = out.get("n_exec") or exec_ctl["n"]
        n_steps = max(1, min(int(n_want), int(plan["n"])))
        step_dt = 1.0 / max(args.step_hz, 1.0)
        drip_dt = 1.0 / max(args.drip_hz, 1.0)
        last = 0
        dripping.set()
        try:
            for i in range(n_steps):
                if stop.is_set():
                    break
                rows = {r: (None if chunk.role(r).wrench_pred is None
                            else chunk.role(r).wrench_pred[i]) for r in force_roles}
                fcmd = _gate_force(rows)
                frame = wire.frame(plan, i)
                taps.step(out.get("seq", -1), i, frame, rows)
                # What goes on the WIRE: the hands' reach residual over the clutched target
                # (--reach-ki; the identity, same object, when off). `frame` itself is what the
                # base commits to below — the residual is published, never integrated.
                pub = _reach_apply(frame)
                _set_wire(pub, fcmd)
                last = i
                if grip_enabled.is_set() and s0.num_clients > 0:
                    for role in grip_roles:
                        widths = chunk.role(role).gripper_m
                        if widths is None or widths.shape[0] <= i or not np.isfinite(widths[i]):
                            continue
                        with jaw_override.lock:
                            # A Manual-jaws press holds this hand until Reset: the policy's width
                            # is DROPPED, the pose still executes. Under the override's lock so a
                            # press can never land between this check and the send below.
                            if jaw_override.intercept(role):
                                continue
                            frac, _sq = grip_cmd[role][0](float(widths[i]))
                            if np.isfinite(frac):
                                subs[f"grip_{role}"].send({"target_frac": float(frac)})
                                jaw_ever_commanded["v"] = True
                t_step = time.time()
                while True:
                    # The gate is re-evaluated per re-send, so flipping the switch off acts as an
                    # interrupt on a chunk that is already executing.
                    _send(pub, _gate_force(rows))
                    if stop.is_set():
                        break
                    remaining = step_dt - (time.time() - t_step)
                    if remaining <= 0:
                        break
                    time.sleep(min(drip_dt, remaining))
        finally:
            dripping.clear()
        # Commit BOTH anchors to the LAST EXECUTED step, so the next chunk branches from where the
        # body actually stopped rather than from the un-executed chunk end: the per-role POSE
        # anchor the model decodes against (which is also this layout's observation — see
        # FivePointPlanner), and the WIRE base the frame was built from. The second is the
        # INTEGRATOR under --post-anchor command; with 'off' it is a no-op on a frozen frame.
        for rc in chunk.roles:
            planner.commit_role(rc.role, rc.position[last], rc.rot6d[last])
        # The UNCOMPENSATED frame, deliberately: committing the reach-compensated one would fold
        # the residual into the base every chunk and turn "at most clamp_m of extra reach" into
        # "clamp_m MORE per chunk".
        wire.commit(wire.frame(plan, last))
        # RECORDED REPLAY: walk the episode cursor forward by what was ACTUALLY executed, by TIME
        # (RecordedEpisodeSource.advance_frame) — and only if the cursor was not moved underneath
        # this chunk (an episode change while dripping), so a stale advance can never yank another
        # episode's cursor. Open-loop: the plan's rows ARE camera frames, so the cursor lands on
        # the last commanded one (the episode end when it ran through, the STOP frame otherwise).
        rp = out.get("replay")
        if rp is not None and replay is not None:
            if rp.get("mode") == "openloop":
                nxt = min(int(rp["frame"]) + max(last, 0), int(rp["n_ep"]) - 1)
            else:
                nxt = replay.advance_frame(rp["episode"], rp["frame"], last + 1)
                usable = replay.next_usable_frame(rp["episode"], nxt)
                if usable is not None:
                    nxt = usable
            with replay_lock:
                moved = not (replay_ctl["episode"] == rp["episode"] and replay_ctl["frame"] == rp["frame"])
                if not moved:
                    replay_ctl["frame"] = int(nxt)
            if not moved:
                _replay_refresh_panel(int(rp["episode"]), int(nxt))
        return last + 1

    def _release_jaw_latches() -> None:
        """Drop every hand's squeeze latch. Called on Stop / Reset: an abandoned grasp must not
        keep the next chunk's target pinned at the width the aborted one had reached."""
        for role in grip_roles:
            grip_cmd[role][1]()

    def _open_jaws(why: str, *, include_manual: bool = False) -> None:
        """Command the POLICY's jaw(s) fully OPEN — Reset and shutdown, never mid-chunk.

        ``include_manual`` (shutdown only) adds every jaw the operator drove from the "Manual jaws"
        folder: a Ctrl-C must not leave a hand clamped with nothing left running to open it. Reset
        deliberately does NOT: it re-homes the policy's jaws as before, and a jaw the operator closed
        by hand (a tool held, a snag being freed) stays where they put it.
        """
        to_open = list(grip_roles) if jaw_ever_commanded["v"] else []
        if include_manual:
            to_open += [r for r in MANUAL_JAW_ROLES if r in manual_jaw_touched and r not in to_open]
        if not to_open:
            return
        _release_jaw_latches()
        for role in to_open:
            try:
                subs[f"grip_{role}"].send({"target_frac": 1.0})
            except Exception as exc:  # noqa: BLE001 - a jaw release must not mask the caller
                _log(f"{why}: could not open the {role} jaw ({exc!r})")
        _log(f"{why}: commanded jaw(s) {to_open} open")

    # ---------------- GUI ----------------
    with server.gui.add_folder("Status"):
        server.gui.add_text("layout", disabled=True,
                            initial_value=f"{loaded.layout.mode} ({loaded.layout.action_dim}-D "
                                          f"action, {loaded.layout.state_dim}-D state)"
                                          + (f" · HELD: {','.join(wire.held_roles)}"
                                             if wire.held_roles else ""))
        server.gui.add_text("wire family", disabled=True,
                            initial_value=f"{wire.mode} — MODE={wire.low_level_mode}")
        server.gui.add_text("clutch", disabled=True,
                            initial_value=f"{args.post_anchor} · carry {args.carry} "
                                          f"({wire.clutch.n_carried}/5) · "
                                          f"{wire.clutch.post.describe()[:50]}")
        server.gui.add_text("anchor", disabled=True, initial_value=anchor.describe()[:110])
        g_obs = server.gui.add_text("obs age", initial_value="…", disabled=True)
        g_ll = server.gui.add_text("low level", initial_value="…", disabled=True)
        g_chunk = server.gui.add_text("last chunk", initial_value="no chunk yet", disabled=True)
        # VISIBLE, because it is the one input to a chunk that the operator cannot otherwise see.
        # Two Predicts from the same observation differ ONLY by this, and without it on the page
        # that reads as the model being unstable rather than as a different draw.
        g_seed = server.gui.add_text("sampler seed", initial_value=seed_ctl.describe(), disabled=True,
                                     hint="the sampler's noise seed. Unpinned it increments per "
                                          "sample, so pressing Predict again is a fresh draw from "
                                          "the same conditional; Reset returns it to the start, so "
                                          "'back to the beginning' is bit-exact. --seed N pins it.")
        g_wire = server.gui.add_text("wire", initial_value="ready frame (idle)", disabled=True)
        g_slew = server.gui.add_text("slew cap", initial_value=wire.slew.describe(), disabled=True)
        g_state = server.gui.add_text(
            "state (geometry)" if loaded.layout.state_is_geometric else "state (wrt_start)",
            initial_value="…", disabled=True)
        g_range = server.gui.add_text(
            "state range", initial_value="…", disabled=True,
            hint="the state row above against THIS checkpoint's own norm_stats['state'] [q01, q99] "
                 "band (what training mapped onto [-1, 1]). A column outside it means the next "
                 "Predict conditions on a state the policy never saw; nothing else on the page "
                 "would say so. Also flags an Euler column within 30° of the ±180° wrap.")
        g_ik = server.gui.add_text("IK body", initial_value="…", disabled=True)

    with server.gui.add_folder("Control") as f_control:
        b_predict = server.gui.add_button("Predict")
        b_run = server.gui.add_button("Run staged chunk")
        # THE CADENCE SWITCH. Both rollouts remove the click and nothing else about the safety
        # envelope; they differ in WHEN the sampler runs. This is the one control that chooses, so
        # `▶▶` below reads it — see `_run_to_completion` / `_run_streaming`. Manual Predict/Run stay
        # as the single-step form for bring-up and for watching one chunk.
        d_cadence = server.gui.add_dropdown(
            "▶▶ cadence", ("sequential (chunk by chunk)", "stream (HATO continuous)"),
            initial_value=("stream (HATO continuous)" if args.cadence == "stream"
                           else "sequential (chunk by chunk)"),
            hint="sequential: predict, then drip N steps — the wire is FROZEN for the whole "
                 "sample, measured 17-46% of the time, which is the stop-go. stream: the sampler "
                 "runs on its own thread and every step publishes a blend of every chunk still "
                 "covering it (HATO §IV-B), so the wire never freezes. Same clutch, same gate, "
                 "same wire.")
        b_auto = server.gui.add_button("▶▶ Run to completion")
        b_auto_stop = server.gui.add_button("⏹ Stop the run", disabled=True)
        b_stop = server.gui.add_button("STOP")
        b_reset = server.gui.add_button("Reset to ready pose")
        g_keys = server.gui.add_text(
            "hotkeys", initial_value="starting …", disabled=True,
            hint="the same buttons on the keyboard of the TERMINAL running this page (not the "
                 "browser): p = ▶▶ / ⏹, k = 🚶, l = 🦵 legs, c = ⬛ close BOTH jaws, u = ⬜ open BOTH "
                 "jaws (the Manual-jaws buttons, override until Reset), SPACE = STOP. --no-keys turns "
                 "it off.")
        # DERIVED from the horizon, never hardcoded: an H=200 checkpoint offers 200 here and an
        # H=50 one does not. Options stay BARE INTEGERS because the handler below does
        # `int(d_exec.value)` — the seconds belong in the hint, not in the option text.
        d_exec = server.gui.add_dropdown(
            "Execute how many steps", tuple(str(n) for n in _EXEC_CHOICES),
            initial_value=str(exec_ctl["n"]),
            hint=(f"of the {ACTION_HORIZON}-step chunk, at {1.0 / ACTION_DT:.0f} Hz: "
                  + ", ".join(f"{n}={n * ACTION_DT:.2f}s" for n in _EXEC_CHOICES)
                  + ". The chunk is always predicted in full; this is how much of it executes "
                    "before the next Predict. IGNORED by the continuous cadence, which has no "
                    "execute window — every step publishes."))
        s_auto_secs = server.gui.add_slider("▶▶ run seconds", min=1.0, max=120.0, step=1.0,
                                            initial_value=float(auto_ctl["max_s"]),
                                            hint="wall-clock ceiling for EITHER cadence. Re-read as "
                                                 "the run advances — every step when streaming, at "
                                                 "the next chunk boundary when sequential — so "
                                                 "shortening it can only stop the body sooner.")
        c_ghost = server.gui.add_checkbox("Show the violet IK body", initial_value=ik is not None,
                                          disabled=ik is None)
        c_triads = server.gui.add_checkbox("Plan orientation triads", initial_value=True)
        g_msg = server.gui.add_text("last action", initial_value="idle", disabled=True)

    # ---------------- LIVE CAMERAS: what the D405s see NOW (display only; 2026-09-20) ----------------
    # Every configured camera bridge — the view(s) the model reads and a configured-but-unread one
    # (camera R under a left_only checkpoint; `view{i}` in main()) — re-drawn at --live-camera-hz from
    # the bridge's own JPEG in its natural colours, downscaled to --live-camera-width; plus, per view the
    # model reads, the EXACT 224x224 crop it is fed (`snapshot_display_images`: the newest history frame,
    # denormalized). Next to the replay panel's recorded frames that is the alignment check the operator
    # asked for, and it is also how a wrong channel order, a swapped pair of bridges or a stale camera
    # becomes visible on the page. Nothing here reaches the planner, the wire or the ⏺ run recording; a
    # failure is logged once and skipped.
    live_cams: list[dict] = []
    live_cam_ui: dict = {}
    if args.live_camera_hz > 0:
        _lc_side = {0: "L", 1: "R"}
        with server.gui.add_folder("📷 Live cameras (what the D405s see NOW)", order=f_control.order + 0.3):
            g_livecam = server.gui.add_text(
                "cameras", initial_value="waiting for the first frame …", disabled=True,
                hint="per view: frame size · age of the newest frame · the bridge's rate. STALE = no new "
                     "frame for > 1 s (the bridge is down, or the D405 dropped off the hub).")
            for i, url in enumerate(_camera_urls(args)):
                read = f"camera{i}" in subs
                sub = subs.get(f"camera{i}") if read else subs.get(f"view{i}")
                if sub is None:
                    continue
                key = str(loaded.image_keys[i]) if read else f"camera{i}_rgb"
                handle = server.gui.add_image(
                    np.zeros((16, 16, 3), dtype=np.uint8), format="jpeg", jpeg_quality=75,
                    label=(f"{key} · {_lc_side.get(i, i)} :{url.rsplit(':', 1)[-1]} · "
                           + ("READ by the model" if read else "display only — this checkpoint does not read it")))
                live_cams.append({"key": key, "read": read, "sub": sub, "img": handle, "n": -1, "shape": None,
                                  "rate_n": None, "rate_t": 0.0, "hz": None, "stale": False, "drawn": False})
            model_imgs = [
                server.gui.add_image(np.zeros((16, 16, 3), dtype=np.uint8), format="jpeg",
                                     label=f"model input · {loaded.image_keys[i]} — the 224×224 crop it is fed "
                                           "(newest frame, natural colours)")
                for i in range(int(loaded.num_cameras))]
        live_cam_ui = {"status": g_livecam, "model_imgs": model_imgs, "failed": False}
        _log(f"LIVE CAMERAS: {len(live_cams)} view(s) on the page at {args.live_camera_hz:g} Hz — "
             + ", ".join(f"{lc['key']} ({'read by the model' if lc['read'] else 'display only'}, {lc['sub'].url})"
                         for lc in live_cams)
             + f" — plus the model's own {len(model_imgs)} crop(s). --live-camera-hz 0 removes the folder.")

    def _live_cameras_tick(now: float) -> None:
        """Re-draw every live view whose bridge sent a new frame, and the model's crop(s). Display only."""
        parts = []
        for lc in live_cams:
            msg, wall, n = lc["sub"].latest()
            if msg is None:
                parts.append(f"{lc['key']}: NO FRAME yet ({lc['sub'].url})")
                continue
            age = time.time() - wall
            if lc["rate_n"] is None or now - lc["rate_t"] >= 2.0:
                if lc["rate_n"] is not None and now > lc["rate_t"]:
                    lc["hz"] = (n - lc["rate_n"]) / (now - lc["rate_t"])
                lc["rate_n"], lc["rate_t"] = n, now
            if n != lc["n"]:
                lc["n"] = n
                frame = HL._decode_jpeg_b64(msg.get("jpeg_b64"))
                if frame is not None:
                    lc["img"].image = _live_thumb(frame, int(args.live_camera_width))
                    lc["shape"] = frame.shape
                    if not lc["drawn"]:
                        lc["drawn"] = True
                        _log(f"LIVE CAMERAS: {lc['key']} first frame {frame.shape[1]}×{frame.shape[0]} "
                             f"from {lc['sub'].url}")
            stale = age > 1.0
            if stale != lc["stale"]:
                lc["stale"] = stale
                _log(f"LIVE CAMERAS: {lc['key']} "
                     + (f"STALE — no new frame for {age:.1f} s ({lc['sub'].url})" if stale else "streaming again"))
            sh = lc["shape"]
            parts.append(f"{lc['key']}: " + ("" if sh is None else f"{sh[1]}×{sh[0]} · ")
                         + f"{age:.2f} s old" + ("" if lc["hz"] is None else f" · {lc['hz']:.0f} fps")
                         + (" ⚠ STALE" if stale else ""))
        if live_cam_ui.get("model_imgs"):
            # camera-major, oldest-first per camera: camera c's NEWEST history frame is c*To + To-1.
            disp = planner.snapshot_display_images()
            to = int(loaded.obs_history_len)
            for c, h in enumerate(live_cam_ui["model_imgs"]):
                idx = c * to + to - 1
                if 0 <= idx < len(disp):
                    h.image = disp[idx]
        live_cam_ui["status"].value = " · ".join(parts)[:240] or "no camera bridge configured"

    # ---------------- RECORDED-EPISODE REPLAY: the panel (extension) ----------------
    # Only with --replay-zarr. Selects WHICH chunk ▶▶ / Predict stage (the source), which episode and
    # frame, and how (the mode); shows the episode's FIRST recorded frame per camera the checkpoint
    # reads, and the cursor's frame. Changes are refused while a run is active (the cursor is the
    # run's), and every change drops a staged chunk from the old position.
    if replay is not None:
        _rp_lens = replay.episode_lengths()
        _rp_first = replay.first_usable_frames()
        _rp_options = hl_replay.episode_options(replay, _rp_first)
        _rp_cams = [k for k in loaded.image_keys if k in replay.image_keys] or [hl_replay.DEFAULT_IMAGE_KEY]
        _rp_modes = {"sequential": "sequential (re-anchor per chunk)", "openloop": "open-loop (whole episode, one placement)"}
        with replay_lock:
            _rp_ep0, _rp_fr0 = int(replay_ctl["episode"]), int(replay_ctl["frame"])
            _rp_mode0 = str(replay_ctl["mode"])
        with server.gui.add_folder("Replay recorded episode", order=f_control.order + 0.25):
            server.gui.add_text("file", initial_value=os.path.basename(replay.path)[:80], disabled=True)
            server.gui.add_text(
                "episodes", disabled=True,
                initial_value=(f"{replay.n_episodes} · {min(_rp_lens)}..{max(_rp_lens)} frames · "
                               f"{sum(1 for f in _rp_first if f not in (0, None))} start masked · "
                               f"{sum(1 for f in _rp_first if f is None)} unplayable"))
            d_rp_source = server.gui.add_dropdown(
                "Chunk SOURCE", ("recorded replay", "model"),
                initial_value="recorded replay" if replay_ctl["source"] == "replay" else "model",
                hint="what ▶▶ / Predict stage: the RECORDED episode at the cursor, or the policy's sample. "
                     "Live — the model stays loaded. Refused while a run is active (⏹ first).")
            d_rp_episode = server.gui.add_dropdown(
                "Episode", tuple(_rp_options), initial_value=_rp_options[_rp_ep0],
                hint="type to search (ep 0012 …). ⚠ starts at N = the converter masked the first frames; "
                     "the cursor auto-seeks there.")
            n_rp_episode = server.gui.add_number(
                "Episode #", initial_value=_rp_ep0, min=0, max=max(replay.n_episodes - 1, 0), step=1,
                hint="the same choice as a number (the dropdown follows).")
            s_rp_frame = server.gui.add_slider(
                "Start frame", min=0, max=max(_rp_lens[_rp_ep0] - 1, 1), step=1, initial_value=_rp_fr0,
                hint="where the next chunk starts; a masked frame snaps FORWARD to the next playable one.")
            d_rp_mode = server.gui.add_dropdown(
                "Replay mode", tuple(_rp_modes.values()), initial_value=_rp_modes[_rp_mode0],
                hint="sequential: one chunk per Predict, re-anchored to the LAST EXECUTED step every "
                     "'Execute how many steps' steps — the receding clutch a policy run has. open-loop: the "
                     "WHOLE episode decoded once against the arm's pose and dripped straight through (ignores "
                     "'Execute how many steps'; a true open-loop tracking number).")
            b_rp_rewind = server.gui.add_button("⏮ Rewind to the first playable frame")
            c_rp_path = server.gui.add_checkbox("Show the recorded episode path (green)", initial_value=True,
                                                hint="the whole episode at the predicted hand(s), transported by "
                                                     "the CURRENT decode anchor — where the arm will actually go.")
            g_rp_status = server.gui.add_text("replay", initial_value="…", disabled=True)
            g_rp_transport = server.gui.add_text(
                "transport", initial_value="…", disabled=True,
                hint="the demo replays RE-ORIENTED by this angle: the recorded hand orientation at the cursor "
                     "vs the decode anchor (R_anchor · R_recorded^T). Physical — the human held the tool one "
                     "way, the robot holds it another — and deliberately not corrected; see "
                     "hl_replay._body_vs_world_transport for the measured reasons.")
            _rp_first_imgs = {}
            for _key in _rp_cams:
                _rp_first_imgs[_key] = server.gui.add_image(
                    np.zeros((16, 16, 3), dtype=np.uint8), format="jpeg",
                    label=f"episode frame 0 · {_key} (the FIRST recorded frame)")
            g_rp_cursor_img = server.gui.add_image(
                np.zeros((16, 16, 3), dtype=np.uint8), format="jpeg",
                label=f"cursor · {_rp_cams[0]} (the frame the next chunk starts at)")
        replay_ui.update({"status": g_rp_status, "transport": g_rp_transport, "cams": list(_rp_cams),
                          "first_imgs": _rp_first_imgs, "cursor_img": g_rp_cursor_img, "show_path": True,
                          "syncing": False, "options": list(_rp_options), "first": list(_rp_first)})

        def _rp_busy() -> str | None:
            """Why a panel change is refused right now, or None."""
            if auto_running.is_set():
                return "a ▶▶ run is active — press ⏹ first"
            if dripping.is_set():
                return "a chunk is executing — wait for it"
            return None

        def _rp_set_widgets(episode: int, frame: int) -> None:
            """Put the three pickers on (episode, frame) WITHOUT re-entering their callbacks."""
            replay_ui["syncing"] = True
            try:
                if d_rp_episode.value != _rp_options[episode]:
                    d_rp_episode.value = _rp_options[episode]
                if int(n_rp_episode.value) != int(episode):
                    n_rp_episode.value = int(episode)
                s_rp_frame.max = max(_rp_lens[episode] - 1, 1)
                if int(s_rp_frame.value) != int(frame):
                    s_rp_frame.value = int(frame)
            finally:
                replay_ui["syncing"] = False

        def _rp_select_episode(episode: int) -> None:
            first = _rp_first[episode]
            n = _rp_lens[episode]
            _rp_set_widgets(episode, int(first or 0))
            if first is None:
                _replay_seek(episode, 0, "NO playable frame (every step 0 is masked) — pick another episode")
            elif first != 0:
                _replay_seek(episode, int(first), f"frames 0..{first - 1} are masked, cursor starts at {first}; "
                                                  f"{n} frames = {n * ACTION_DT:.1f} s")
            else:
                _replay_seek(episode, 0, f"{n} frames = {n * ACTION_DT:.1f} s — ▶▶ or Predict to stage")

        @d_rp_source.on_update
        def _(_e) -> None:
            if replay_ui["syncing"]:
                return
            want = "replay" if str(d_rp_source.value).startswith("recorded") else "model"
            why = _rp_busy()
            if why is not None:
                with replay_lock:
                    cur = replay_ctl["source"]
                replay_ui["syncing"] = True
                try:
                    d_rp_source.value = "recorded replay" if cur == "replay" else "model"
                finally:
                    replay_ui["syncing"] = False
                g_msg.value = f"chunk source unchanged — {why}"
                return
            with replay_lock:
                replay_ctl["source"] = want
            staged["out"] = None
            g_msg.value = (f"chunk source = {want} — Predict / ▶▶ now stage "
                           f"{'the RECORDED episode' if want == 'replay' else 'the POLICY'}")
            _log(f"replay: chunk source -> {want}")

        @d_rp_episode.on_update
        def _(_e) -> None:
            if replay_ui["syncing"]:
                return
            epi = _rp_options.index(str(d_rp_episode.value))
            why = _rp_busy()
            if why is not None:
                with replay_lock:
                    cur_ep, cur_fr = int(replay_ctl["episode"]), int(replay_ctl["frame"])
                _rp_set_widgets(cur_ep, cur_fr)
                g_msg.value = f"episode unchanged — {why}"
                return
            _rp_select_episode(int(epi))

        @n_rp_episode.on_update
        def _(_e) -> None:
            if replay_ui["syncing"]:
                return
            epi = int(np.clip(int(n_rp_episode.value), 0, replay.n_episodes - 1))
            if _rp_options[epi] != str(d_rp_episode.value):
                d_rp_episode.value = _rp_options[epi]   # the dropdown's callback does the seek

        @s_rp_frame.on_update
        def _(_e) -> None:
            if replay_ui["syncing"]:
                return
            with replay_lock:
                epi, cur_fr = int(replay_ctl["episode"]), int(replay_ctl["frame"])
            why = _rp_busy()
            if why is not None:
                _rp_set_widgets(epi, cur_fr)
                g_msg.value = f"start frame unchanged — {why}"
                return
            fr = int(np.clip(int(s_rp_frame.value), 0, _rp_lens[epi] - 1))
            usable = replay.next_usable_frame(epi, fr)
            note = ""
            if usable is not None and usable != fr:
                note = f"frame {fr} is masked — snapped to {usable}"
                fr = int(usable)
                _rp_set_widgets(epi, fr)
            _replay_seek(epi, fr, note)

        @d_rp_mode.on_update
        def _(_e) -> None:
            if replay_ui["syncing"]:
                return
            want = next((k for k, v in _rp_modes.items() if v == str(d_rp_mode.value)), "sequential")
            why = _rp_busy()
            if why is not None:
                with replay_lock:
                    cur = str(replay_ctl["mode"])
                replay_ui["syncing"] = True
                try:
                    d_rp_mode.value = _rp_modes.get(cur, _rp_modes["sequential"])
                finally:
                    replay_ui["syncing"] = False
                g_msg.value = f"replay mode unchanged — {why}"
                return
            with replay_lock:
                replay_ctl["mode"] = want
            staged["out"] = None
            g_msg.value = f"replay mode = {want} (Predict again)"

        @b_rp_rewind.on_click
        def _(_e) -> None:
            why = _rp_busy()
            if why is not None:
                g_msg.value = f"rewind refused — {why}"
                return
            with replay_lock:
                epi = int(replay_ctl["episode"])
            first = int(_rp_first[epi] or 0)
            _rp_set_widgets(epi, first)
            _replay_seek(epi, first, "rewound")
            g_msg.value = f"replay: ep {epi} rewound to frame {first}"

        @c_rp_path.on_update
        def _(_e) -> None:
            replay_ui["show_path"] = bool(c_rp_path.value)
            with replay_lock:
                epi, fr = int(replay_ctl["episode"]), int(replay_ctl["frame"])
            _draw_replay_paths(epi, fr)

        _rp_select_episode(_rp_ep0)
        _log(f"replay: panel ready — source {replay_ctl['source']}, episode {_rp_ep0} from frame "
             f"{replay_ctl['frame']}, mode {_rp_mode0}; first-frame view(s) {list(_rp_cams)}")

    with server.gui.add_folder("Continuous cadence (HATO)"):
        # Every control here is inert under the sequential cadence, and each is live DURING a
        # streamed run: the run pushes `stream_ctl` into its own ChunkStream on every step, because
        # the stream object is per-run and a callback must not reach into it.
        d_ens = server.gui.add_dropdown(
            "Ensemble", tuple(hl_stream.ENSEMBLE_MODES), initial_value=stream_ctl["mode"],
            hint="how the overlapping predictions for one step become one command. avg = plain "
                 "mean (HATO's default, measured best on jerk); act = z*tau + c*(1-tau) folded "
                 "oldest-to-newest (best measured tracking); new / old = a single prediction, for "
                 "an A/B. Blending happens in ABSOLUTE space, after each chunk is un-relativized "
                 "against its own anchor.")
        s_tau = server.gui.add_slider("act tau", min=0.0, max=1.0, step=0.05,
                                      initial_value=float(stream_ctl["tau"]),
                                      hint="only used by 'act'. Larger weights the OLDER survivors "
                                           "more; 0 collapses it to 'new'.")
        s_replan = server.gui.add_slider("Min steps between samples", min=0, max=50, step=1,
                                         initial_value=int(stream_ctl["replan_min"]),
                                         hint="0 = re-plan as fast as the sampler allows, which is "
                                              "HATO's cadence and the thickest overlap. Raising it "
                                              "cuts GPU duty and thins the blend; below ~5 live "
                                              "chunks the blend is measured to stop helping.")
        c_sghost = server.gui.add_checkbox("Solve the violet body while streaming",
                                           initial_value=bool(stream_ctl["ghost"]),
                                           disabled=ik is None,
                                           hint="OFF by default: the IK is pure Python and would "
                                                "contend with the 30 Hz publisher for the one "
                                                "interpreter. The body is display only; the "
                                                "publisher's cadence is the point of this mode.")
        g_stream = server.gui.add_text("stream", initial_value="idle", disabled=True)

    with server.gui.add_folder("Force command"):
        c_force = server.gui.add_checkbox("Command force (needs FORCE_CMD_WIRE=1)",
                                          initial_value=bool(args.enable_force_cmd))
        s_gate = server.gui.add_slider("gate (N)", min=0.0, max=20.0, step=0.1,
                                       initial_value=float(args.force_gate_n))
        c_grip = server.gui.add_checkbox(
            "Command both jaws" if len(grip_roles) == 2 else f"Command the {'/'.join(grip_roles)} jaw",
            initial_value=args.command_gripper)

    # ---------------- MANUAL jaws (extension, 2026-09-14) ----------------
    # Both hands, whatever the checkpoint predicts: a left_only run still has a RIGHT jaw the operator
    # may need to open or close by hand, and the policy's own jaw command exists only per step of a
    # Run. The buttons send the SAME `{"target_frac": f}` frame `_drip` sends, to the SAME subscriber,
    # so the bridge sees one client and one protocol. THE PRESS WINS (2026-09-18; it used to be
    # REFUSED while the policy drove that jaw, because the policy's per-step command would overwrite
    # it within 33 ms and the operator read that as a dead button): a press latches `jaw_override`
    # for that hand, both publish paths then DROP the policy's width for it while the pose keeps
    # executing, and nothing but Reset releases it — not ⏹, not STOP, not a new ▶▶, not the jaw
    # checkbox. Reset opens the policy's jaws as before and hands them back; a manual jaw the policy
    # does not predict is never blocked and stays where the operator put it; shutdown opens every
    # jaw the page ever commanded (see `_open_jaws`). The readout beside each hand is the
    # bridge's own frame, `motor_ok` included — the flag the bridge latches False when its control
    # thread dies (two bridges on one CAN channel, 2026-09-14), which nothing on this page showed
    # before: the jaw then "felt like it would not move" while `seq` kept counting.
    def _jaw_readout(role: str) -> str:
        sub = subs.get(f"grip_{role}")
        if sub is None:
            return "not subscribed"
        msg, wall, _n = sub.latest()
        if msg is None or msg.get("actual") is None:
            return (f"connected to {sub.url}, no frame yet" if sub.connected
                    else f"NO BRIDGE at {sub.url} (retrying)")
        try:
            o, c = float(msg["open"]), float(msg["closed"])
            span = o - c
            f_act = (float(msg["actual"]) - c) / span if abs(span) > 1e-9 else float("nan")
            f_cmd = (float(msg["cmd"]) - c) / span if abs(span) > 1e-9 else float("nan")
        except (KeyError, TypeError, ValueError):
            return f"frame without open/closed/cmd/actual: {sorted(msg)}"
        ok = msg.get("motor_ok")
        motor = "motor OK" if ok else ("⚠️ MOTOR FAULT (bridge de-energized it; relaunch T3)"
                                      if ok is False else "motor ?")
        driver = jaw_override.describe(role) or (
            "policy" if role in grip_roles and grip_enabled.is_set()
            and (dripping.is_set() or auto_running.is_set()) else "manual/idle")
        return (f"actual {f_act * 100:.0f}% open · cmd {f_cmd * 100:.0f}% · {motor} · "
                f"{time.time() - wall:.1f} s old · {driver}")

    def _manual_jaw(role: str, frac: float) -> str:
        sub = subs.get(f"grip_{role}")
        if sub is None:
            return f"{role} jaw: not subscribed"
        if not sub.connected:
            return (f"refused — no {role} jaw bridge at {sub.url}; a command queued now would fire "
                    f"whenever it connects")
        frac = float(min(max(frac, 0.0), 1.0))
        with jaw_override.lock:
            # THE PRESS WINS (2026-09-18): latch the hand first, then send, under the same lock the
            # two publish paths take — so the policy's next width for this hand is dropped and no
            # in-flight policy send can overwrite the press. Held until Reset, and only Reset.
            overriding = jaw_override.take(role, frac)
            if role in grip_roles:
                grip_cmd[role][1]()   # a manual move abandons any squeeze latch on this hand
            sub.send({"target_frac": frac})
        jaw_ever_commanded["v"] = True
        manual_jaw_touched.add(role)
        if overriding:
            _log(f"manual jaw: {role} → target_frac {frac:.2f} — OVERRIDE: the policy's {role} jaw "
                 f"commands are ignored until Reset")
            return f"{role} jaw → {frac * 100:.0f}% open — MANUAL OVERRIDE until Reset"
        _log(f"manual jaw: {role} → target_frac {frac:.2f}")
        return f"{role} jaw → {frac * 100:.0f}% open (manual)"

    g_jaw: dict = {}
    s_jaw: dict = {}
    with server.gui.add_folder("Manual jaws (both hands)"):
        for _role in MANUAL_JAW_ROLES:
            g_jaw[_role] = server.gui.add_text(f"{_role} jaw", initial_value="…", disabled=True)
            s_jaw[_role] = server.gui.add_slider(f"{_role} target (1 = open)", min=0.0, max=1.0,
                                                 step=0.05, initial_value=1.0)
            _b_open = server.gui.add_button(f"⬜ open {_role}")
            _b_close = server.gui.add_button(f"⬛ close {_role}")
            _b_set = server.gui.add_button(f"→ {_role} to slider")

            @_b_open.on_click
            def _(_e, role=_role) -> None:
                s_jaw[role].value = 1.0
                g_msg.value = _manual_jaw(role, 1.0)

            @_b_close.on_click
            def _(_e, role=_role) -> None:
                s_jaw[role].value = 0.0
                g_msg.value = _manual_jaw(role, 0.0)

            @_b_set.on_click
            def _(_e, role=_role) -> None:
                g_msg.value = _manual_jaw(role, float(s_jaw[role].value))

    with server.gui.add_folder("Reach compensation (hands)"):
        c_reach = server.gui.add_checkbox(
            "↗ Insist on un-executed HAND motion", initial_value=bool(reach.enabled),
            disabled=not reach.enabled,
            hint="adds the part of the commanded HAND motion the arm did not execute back onto the "
                 "PUBLISHED target, only while the hand performed < stall_frac of what was asked. "
                 "Hands only; the clutch base and the observation are untouched. Un-ticking drops "
                 "the residual. Relaunch with --reach-ki to change the gain.")
        server.gui.add_text(
            "what", disabled=True,
            initial_value=(
                "OFF — relaunch with --reach-ki 0.05 to enable" if not reach.enabled else
                f"hands {','.join(reach.roles)} · ki {reach.config.ki:g} · clamp "
                f"{reach.config.clamp_m * 100:.0f} cm · gate realized < {reach.config.stall_frac:g} "
                f"over {int(reach.config.window)} steps · decay {reach.config.decay:g} · contact "
                f"cut-out {reach.config.contact_n:g} N · identity while un-armed"))
        g_reach = server.gui.add_text("residual", initial_value=reach.describe(), disabled=True,
                                      hint="per hand: the extra reach currently on the wire, the "
                                           "realized fraction the gate keys on, and SATURATED when "
                                           "the hand is asking for all it is allowed and still not "
                                           "arriving — look for a blocked hand, a tare offset or a "
                                           "target out of reach, not a bigger gain.")

    # ---------------- the WALK SKILL (extension) ----------------
    # Always on the page, driving the ONE bundled clip (walk-with-box; `--walk-clip` overrides the
    # file). The walk is a WIRE-level skill (flore/s1_policy/serving/walk_skill.py): it publishes
    # frames the way `_drip` does (`_set_wire` -> `_send` -> `wire.commit`), holds `busy` AND
    # `dripping` for its whole duration so Predict/Run, both ▶▶ cadences and the heartbeat cannot
    # interleave a frame, and touches neither the planner's decode anchor nor its observation — the
    # state reads the three TRACKED points, which the walk holds constant, and never a foot. The two
    # upper-body things it also holds: the last F^cmd (gated by the switch on every step, exactly as
    # the heartbeat does) and the jaws (untouched). Everything above this comment is the page as
    # reviewed before the skill existed; nothing below is read by anything above it.
    walk_running = threading.Event()
    walk_ctl: dict = {"stop_req": False, "player": None, "speed": float(args.walk_speed),
                      "loop": bool(args.walk_loop),
                      "legs": bool(args.walk_legs) and walk_clip is not None and walk_clip.has_joints}
    walk_paths: dict = {}
    if walk_clip is None:   # a caller that did not resolve one (tests) still gets the same page
        walk_clip = walk_skill.load_walk_clip(os.path.expanduser(str(args.walk_clip)))
    if True:   # (kept as a block so the skill's whole surface stays one indented unit)
        _WALK_STYLE = {0: (120, 230, 140), 1: (230, 160, 90)}   # the plan's own foot colours

        def _walk_clip():
            return walk_clip

        def _walk_demand(speed: float) -> str:
            """The clip's per-tick demand, scored against what the low level has SAID: the cap
            `SlewMonitor` measured off RecordTap and whether the caps cover the feet at all
            (`vr_cmd_slewed`). Until it has said anything, the pre-flight estimate and 'if'."""
            b = wire.slew.budget
            return walk_clip.stats().describe(
                speed=speed, budget_m_per_tick=b.m_per_tick, budget_source=b.source,
                publish_hz=float(args.walk_hz), feet_slewed=wire.slew.feet_slewed)

        with server.gui.add_folder("Walk skill (walk-with-box feet, upper body held)"):
            s_wspeed = server.gui.add_slider(
                "Speed", min=walk_skill.SPEED_MIN, max=walk_skill.SPEED_MAX, step=0.05,
                initial_value=float(args.walk_speed),
                hint="playback speed of the clip. Live during a walk. 1.0 = the source's own 50 fps.")
            c_wloop = server.gui.add_checkbox(
                "Loop the clip", initial_value=bool(args.walk_loop),
                hint="ON: the clip repeats (a rate-bounded blend across its wrap gap) until the button "
                     "is pressed again. OFF: one pass, then the feet ramp back to the stance they "
                     "started from.")
            b_walk = server.gui.add_button("🚶 Walk  (press again to stop)")
            g_walk = server.gui.add_text("walk", initial_value="idle", disabled=True)
            g_wdemand = server.gui.add_text(
                "clip demand", initial_value=_walk_demand(walk_ctl["speed"]), disabled=True,
                hint="what the clip asks of the two foot slots per low-level tick (46 mm mean / 95 max "
                     "at 1x). The low level must EXEMPT the feet from its 5 mm/tick slew cap "
                     "(3_run_policy_vive_force.sh SLEW_FEET=0, its default) or the walk becomes a "
                     "crawl; this line says whether it did, read off RecordTap (vr_cmd_slewed) once "
                     "the low level is armed.")
            c_wlegs = server.gui.add_checkbox(
                "Send the clip's LEGS (ref_joints)", initial_value=bool(walk_ctl["legs"]),
                disabled=not walk_clip.has_joints,
                hint="ON: every walk frame also carries the clip's 29 reference joints by name, and the "
                     "low level's command_lower_body obs (the reference legs, 24 of its 184 dims) is "
                     "the clip's gait instead of its frozen idle STANCE — with the stance there the "
                     "robot stepped in place (2026-09-07). OFF: feet only, the pre-fix A/B. Needs the "
                     "low level launched with REF_JOINTS=1 (its default); the line below says whether "
                     "it is applying them, read off RecordTap.")
            g_wlegs = server.gui.add_text(
                "legs", disabled=True,
                initial_value=("clip legs ready — sent while a walk plays" if walk_ctl["legs"] else
                               ("OFF (--no-walk-legs / unchecked): feet only" if walk_clip.has_joints else
                                f"unavailable: {walk_clip.legs_note}")))

        def _refresh_walk_demand() -> None:
            try:
                g_wdemand.value = _walk_demand(walk_ctl["speed"])
            except Exception as exc:  # noqa: BLE001 - a bad clip must not kill the callback
                g_wdemand.value = f"cannot describe the clip: {exc!r}"

        def _clear_walk_paths() -> None:
            for key in list(walk_paths):
                handle = walk_paths.pop(key)
                try:
                    handle.remove()
                except Exception:  # noqa: BLE001, S110
                    pass

        def _draw_walk_paths(clip) -> None:
            """Both feet's full-clip paths in the anchor's world frame (pelvis frame + pelvis_z)."""
            _clear_walk_paths()
            origin = np.asarray(wire.readout.pelvis_pos, dtype=np.float64).reshape(1, 3)
            for k, label in enumerate(walk_skill.FOOT_LABELS):
                pts = (clip.foot_pos[:, k] + origin).astype(np.float32)
                rgb = np.array(_WALK_STYLE[k], np.uint8)
                seg = np.stack([pts[:-1], pts[1:]], axis=1)
                key = label.replace(" ", "_").lower()
                walk_paths[f"{key}:line"] = server.scene.add_line_segments(
                    f"/walk/{key}/line", points=seg,
                    colors=np.broadcast_to(rgb, (seg.shape[0], 2, 3)).copy(), line_width=2.0)

        def _draw_walk_marker(foot_pos) -> None:
            """The two foot targets being published RIGHT NOW, as two fat dots (remove-then-add:
            viser addresses nodes by name, so a stale handle's later remove() would delete the
            fresh node — the same rule `_draw_paths` follows)."""
            old = walk_paths.pop("cmd", None)
            if old is not None:
                try:
                    old.remove()
                except Exception:  # noqa: BLE001, S110
                    pass
            origin = np.asarray(wire.readout.pelvis_pos, dtype=np.float64).reshape(1, 3)
            pts = (np.asarray(foot_pos, dtype=np.float64).reshape(-1, 3) + origin).astype(np.float32)
            walk_paths["cmd"] = server.scene.add_point_cloud(
                "/walk/cmd", points=pts,
                colors=np.array([_WALK_STYLE[0], _WALK_STYLE[1]], np.uint8),
                point_size=0.03, point_shape="circle", precision="float32")

        def _run_walk() -> None:
            """Publish the clip's feet under the wire's current upper body until told to stop."""
            if auto_running.is_set():
                g_walk.value = "refused — a ▶▶ run is active; press ⏹ first"
                return
            if dripping.is_set():
                g_walk.value = "refused — a chunk is executing"
                return
            if s0.num_clients == 0:
                g_walk.value = ("refused — the low level is NOT connected to :8770 (start it in "
                                f"ABSOLUTE mode with MODE={wire.low_level_mode})")
                return
            if not busy.acquire(blocking=False):
                g_walk.value = "refused — busy (a Predict or Run holds the page)"
                return
            note = "walk ended"
            n_pub = n_refused = 0
            t_start = time.time()
            try:
                walk_running.set()
                dripping.set()
                # Same rule as Run: a STOP latched by an earlier abort must not make this a no-op,
                # and every LATER `stop` is the operator's STOP and has to end it within one step.
                stop.clear()
                walk_ctl["stop_req"] = False
                clip = _walk_clip()
                with wire_lock:
                    base = tuple(a.copy() for a in last_wire["frame"])
                    fcmd = None if last_wire["fcmd"] is None else last_wire["fcmd"].copy()
                # `base` is the last PUBLISHED frame, reach residual included: the held hands keep
                # it and `wire.commit` makes it the clutch base, so the residual is consumed here
                # and must not be applied a second time by the chunk that follows the walk.
                _reach_reset("walk start")
                start_pos, start_quat = walk_skill.feet_from_frame(base)
                step_dt = 1.0 / max(float(args.walk_hz), 1.0)
                # The legs start from the pose the low level HOLDS (the ready pose = its frozen
                # clip, `anchor.dof`, SDK order) so the lead-in ramp's first reference is the one the
                # robot is already being told, and the lead-out lands back on it — at which point
                # the field is dropped and the low level's own frozen clip takes over seamlessly.
                legs_on = bool(walk_ctl["legs"]) and clip.has_joints
                player = walk_skill.WalkPlayer(
                    clip, start_pos=start_pos, start_quat=start_quat, step_dt=step_dt,
                    speed=walk_ctl["speed"], loop=walk_ctl["loop"],
                    ramp_speed_m_s=float(args.walk_ramp_speed),
                    ramp_min_s=float(args.walk_ramp_min_s),
                    start_joints=(anchor.dof if legs_on else None),
                    joint_names=(G1_29DOF_SDK_DOF_NAMES if legs_on else None))
                walk_ctl["player"] = player
                st = clip.stats()
                gaps = st.lead_in_gap_m(start_pos)
                rframe0 = _record_frame()
                _log(f"WALK: START {clip.describe()} · speed {player.speed:g}x · loop "
                     f"{'on' if player.loop else 'off'} · publish {1.0 / step_dt:.0f} Hz · lead-in "
                     f"{gaps[0] * 1e3:.0f}/{gaps[1] * 1e3:.0f} mm over {player.lead_in_steps} step(s) · "
                     f"armed={None if rframe0 is None else rframe0.vive_armed}")
                _log("WALK: " + _walk_demand(player.speed))
                if wire.slew.feet_slewed is True:
                    _log("⚠️  WALK: this low level reports that its slew caps COVER the feet "
                         "(RecordTap vr_cmd_slewed) — the walk will be clamped to a crawl. Relaunch "
                         "3_run_policy_vive_force.sh with SLEW_FEET=0 (its default).")
                elif wire.slew.feet_slewed is None:
                    _log("WALK: the low level has not reported its slew coverage yet (no armed "
                         "RecordTap frame) — assuming SLEW_FEET=0; 'clip demand' updates once it has.")
                _log("WALK: the upper three slots are HELD at the wire's last command "
                     f"(L {np.round(base[0][0:3], 3).tolist()}, R {np.round(base[0][3:6], 3).tolist()}, "
                     f"torso {np.round(base[0][6:9], 3).tolist()}); F^cmd "
                     f"{'held' if fcmd is not None else 'none'}, jaws untouched.")
                if player.legs_enabled:
                    _log(f"WALK: LEGS ON — the clip's {walk_skill.N_JOINTS} reference joints ride every frame "
                         f"as `ref_joints`; lead-in from the ready legs spans {player.lead_in_joint_gap_rad:.3f} "
                         f"rad (worst joint) over the same {player.lead_in_steps} step(s). The low level "
                         f"applies them to command_lower_body while ARMED (5pt, REF_JOINTS=1) and says so on "
                         f"RecordTap (`ref_joints_live`) — watch the `legs` line.")
                else:
                    _log(f"WALK: LEGS OFF — feet only ({player.legs_note or 'unchecked on the page'}); "
                         "the low level's command_lower_body stays its frozen idle stance for the whole walk.")
                _draw_walk_paths(clip)
                t_next = time.monotonic()
                last_ui = 0.0
                while True:
                    if stop.is_set():
                        note = (f"walk STOPPED by STOP after {n_pub} step(s) — the wire is FROZEN on "
                                f"the last published feet ({player.progress()})")
                        break
                    if s0.num_clients == 0:
                        note = f"walk ended — the low level dropped off :8770 after {n_pub} step(s)"
                        _log("WALK: the low level disconnected from :8770 — ended.")
                        break
                    if walk_ctl["stop_req"]:
                        walk_ctl["stop_req"] = False
                        if not player.stopping:
                            player.request_stop()
                            _log(f"WALK: stop requested at {player.progress()} — ramping the feet "
                                 "back to where they started")
                    player.set_speed(walk_ctl["speed"])
                    player.set_loop(walk_ctl["loop"])
                    foot_pos, foot_quat, phase = player.step()
                    frame = walk_skill.compose_frame(base, foot_pos, foot_quat)
                    # The page's checkbox is live: unticking mid-walk drops the legs from the next
                    # frame (the low level reverts to its frozen stance — the operator asked for it).
                    legs = player.legs if walk_ctl["legs"] else None
                    with force_lock:
                        on = force_ctl["enabled"]
                    if not _send(frame, fcmd if on else None, ref_joints=legs):
                        # `_send` already logged the refusal and sent nothing. One refused frame is
                        # a decode/frame error, not noise: end rather than keep publishing past it.
                        n_refused += 1
                        note = (f"walk STOPPED — the wire REFUSED a frame at step {n_pub} "
                                f"({player.progress()}); the stale gate holds the robot")
                        break
                    _set_wire(frame, fcmd, ref_joints=legs)
                    # The clutch base follows the published wire, so a chunk predicted afterwards
                    # continues from where the walk left the feet (no-op under --post-anchor off).
                    wire.commit(frame)
                    n_pub += 1
                    # The cap is best MEASURED while the wire is saturated, and a walk saturates it:
                    # observe every published step (read-only; a parse + one norm).
                    try:
                        wire.slew.observe(_record_frame())
                    except Exception:  # noqa: BLE001, S110 - a bad tap frame is not a walk error
                        pass
                    now = time.monotonic()
                    if now - last_ui > 0.2:
                        last_ui = now
                        g_walk.value = (f"{clip.name[:40]} · {player.progress()} · {n_pub} step(s) · "
                                        f"{time.time() - t_start:.1f} s · speed {player.speed:g}x")
                        try:
                            _draw_walk_marker(foot_pos)
                        except Exception:  # noqa: BLE001, S110 - display only
                            pass
                    if player.done:
                        note = (f"walk finished — {n_pub} step(s) in {time.time() - t_start:.1f} s, the "
                                f"feet are back where they started ({player.laps} full lap(s))")
                        # Landed on the ready legs, which is what the low level's frozen clip holds:
                        # drop the field so the heartbeat's frames are plain again (byte-identical).
                        with wire_lock:
                            last_wire["ref_joints"] = None
                        break
                    t_next += step_dt
                    delay = t_next - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        t_next = time.monotonic()   # fell behind: do not burst, re-anchor the clock
            except Exception as exc:  # noqa: BLE001 - a walk error must not kill the page
                import traceback

                _log("WALK: error:\n" + traceback.format_exc())
                note = f"walk error: {exc!r}"
            finally:
                walk_ctl["player"] = None
                dripping.clear()
                walk_running.clear()
                busy.release()
                _clear_walk_paths()
                g_walk.value = note
                _log(f"WALK: END — {note} · {wire.slew.describe()}")
                _refresh_walk_demand()

        def _walk_toggle() -> str:
            """The 🚶 button's body — and the `k` hotkey's (tty_keys), so the two cannot drift."""
            # TOGGLE. Running -> request a graceful stop (the thread ramps the feet back and exits);
            # idle -> start on its own thread, never on viser's callback thread (a walk is seconds).
            if walk_running.is_set():
                walk_ctl["stop_req"] = True
                g_walk.value = "stopping — ramping the feet back to where they started"
                return g_walk.value
            if wire_hooks["legs"] is not None:
                # The LEG OVERRIDE (below) already has the two foot slots, under a planner that may be
                # running; this walk would hold the page and fight it for the same slots.
                g_walk.value = "refused — the LEG OVERRIDE has the feet; press l (or its button) to hand them back, then k"
                return g_walk.value
            g_walk.value = f"starting {walk_clip.name[:40]} …"
            threading.Thread(target=_run_walk, daemon=True, name="walk").start()
            return g_walk.value

        @b_walk.on_click
        def _(_e) -> None:
            _walk_toggle()

        @s_wspeed.on_update
        def _(_e) -> None:
            try:
                walk_ctl["speed"] = float(np.clip(float(s_wspeed.value), walk_skill.SPEED_MIN,
                                                  walk_skill.SPEED_MAX))
            except (TypeError, ValueError):
                return
            _refresh_walk_demand()

        @c_wlegs.on_update
        def _(_e) -> None:
            walk_ctl["legs"] = bool(c_wlegs.value) and walk_clip.has_joints
            _log(f"WALK: legs {'ON' if walk_ctl['legs'] else 'OFF'} (page checkbox)"
                 + ("" if not walk_running.is_set() else " — applies from the next published frame"))

        def _legs_status() -> str:
            """What the LOW LEVEL says about the legs (RecordTap `ref_joints_live`), next to what the
            page is sending — the two can disagree (REF_JOINTS=0, an older build, un-armed)."""
            with wire_lock:
                sending = last_wire["ref_joints"] is not None
            hook = wire_hooks["legs"]   # the LEG OVERRIDE adds the legs at the seam, not in last_wire
            sending = sending or (hook is not None and hook.sending_legs)
            rframe = _record_frame()
            if rframe is None:
                robot = "low level: no RecordTap"
            elif rframe.ref_joints_live is None:
                robot = "low level: does not report ref_joints (older build?)"
            elif rframe.ref_joints_live:
                robot = "low level: APPLYING the wire's legs to command_lower_body"
            else:
                robot = ("low level: frozen-clip legs" + (" ⚠️ although the page is sending legs — "
                         "REF_JOINTS=0 / un-armed?" if sending else ""))
            if not walk_clip.has_joints:
                return f"unavailable: {walk_clip.legs_note} · {robot}"
            if not walk_ctl["legs"]:
                return f"OFF (page/--no-walk-legs): feet only · {robot}"
            return (f"{'SENDING the clip legs (ref_joints)' if sending else 'ON — sent while a walk plays or the legs are overridden'}"
                    f" · {robot}")

        @c_wloop.on_update
        def _(_e) -> None:
            walk_ctl["loop"] = bool(c_wloop.value)

    # ---------------- the LEG OVERRIDE (extension) ----------------
    # The walk skill's OTHER composition (walk_skill.LegOverride, 2026-09-12): the clip's feet + legs
    # replace the planner's two FOOT slots on EVERY frame the page publishes while the planner keeps
    # running — Predict, Run, both ▶▶ cadences and the heartbeat are untouched and keep publishing;
    # only their frames' feet change on the way to the socket, at the one seam above (`_send` runs
    # `wire_hooks["legs"]`). This block is a CLOCK, not a publisher: it advances the same WalkPlayer
    # the 🚶 walk uses at the clip's rate and the publishers read the current tick off it. Three
    # things it tells the reviewed machinery: the clutch that slots 3-4 are externally driven
    # (`wire.externally_driven`, so its leak bound stops chasing the gait), the player where to hand
    # back (`home_provider` = the planner's LIVE foot target in `last_wire`, pursued at ≤ 5 mm/tick and
    # landed on exactly), and the 🚶 walk that the feet are taken (each refuses while the other is
    # active). STOP freezes the gait in place and the override stays engaged (▶▶ / Run resume it);
    # `l` / the button hands the legs back. Reads the walk folder's Speed / Loop / Legs controls.
    legs_engaged = threading.Event()
    legs_ctl: dict = {"override": None, "release_req": False}
    _FOOT_SLOTS = frozenset(range(hl_dense.FIVE_POINT_GATED, len(hl_dense.FIVE_POINT_BODIES)))

    def _policy_feet():
        """The planner's LIVE foot target — the feet of the last frame a planner path recorded
        (`_set_wire`: every drip / stream step; the heartbeat replays it). The hand-back's home."""
        with wire_lock:
            return walk_skill.feet_from_frame(last_wire["frame"])

    def _armed_now():
        """RecordTap's `vive_armed` off a FRESH frame, else None (unknown)."""
        msg, wall, _n = subs["record"].latest()
        if msg is None or not wall or (time.time() - float(wall)) > hl_5pt.REACH_MEASURED_MAX_AGE_S:
            return None
        rframe = _record_frame()
        return None if rframe is None else bool(rframe.vive_armed)

    with server.gui.add_folder("Leg override (the clip's feet + legs UNDER the running policy)"):
        b_legs = server.gui.add_button("🦵 Legs ← clip  (press again to hand back)")
        g_legs = server.gui.add_text(
            "legs override", initial_value="off — the planner's own feet", disabled=True,
            hint="ON: the walk-with-box clip's two feet (and its legs, per the checkbox above) replace "
                 "the planner's two foot slots on every published frame while the planner keeps "
                 "running — lead-in from the current feet, then the clip (Speed / Loop above are "
                 "live). Press again: the feet pursue the planner's own foot target at ≤ 5 mm/tick and "
                 "hand over on landing. STOP freezes the gait in place (▶▶ / Run resume it). Hotkey l.")

    def _run_legs() -> None:
        """The clock thread: install the substitution, advance the clip, hand back on request."""
        if walk_running.is_set():
            g_legs.value = "refused — the 🚶 walk has the feet (press k to end it first)"
            return
        if s0.num_clients == 0:
            g_legs.value = ("refused — the low level is NOT connected to :8770 (start it in ABSOLUTE "
                            f"mode with MODE={wire.low_level_mode})")
            return
        if _armed_now() is False:
            # holosoma's absolute arm clutch snapshots `held - wire` PER POINT at arm: arming into a
            # mid-stride wire would give each foot a constant offset for the rest of the session.
            g_legs.value = ("refused — the low level is NOT ARMED (RecordTap vive_armed False): arm it "
                            "(joystick start) first, so it never arms into a mid-stride foot wire")
            return
        if legs_engaged.is_set():
            return
        legs_engaged.set()
        note = "legs handed back"
        t_start = time.time()
        clip = walk_clip
        try:
            legs_ctl["release_req"] = False
            with wire_lock:
                base = tuple(a.copy() for a in last_wire["frame"])
            start_pos, start_quat = walk_skill.feet_from_frame(base)
            step_dt = 1.0 / max(float(args.walk_hz), 1.0)
            legs_on = bool(walk_ctl["legs"]) and clip.has_joints
            player = walk_skill.WalkPlayer(
                clip, start_pos=start_pos, start_quat=start_quat, step_dt=step_dt,
                speed=walk_ctl["speed"], loop=walk_ctl["loop"],
                ramp_speed_m_s=float(args.walk_ramp_speed),
                ramp_min_s=float(args.walk_ramp_min_s),
                start_joints=(anchor.dof if legs_on else None),
                joint_names=(G1_29DOF_SDK_DOF_NAMES if legs_on else None),
                home_provider=_policy_feet)
            override = walk_skill.LegOverride(player, legs_enabled=lambda: bool(walk_ctl["legs"]))
            legs_ctl["override"] = override
            gaps = clip.stats().lead_in_gap_m(start_pos)
            # Tell the clutch first, then install the hook: from the next published frame on, the
            # feet on the wire are the clip's and the planner's foot base is no longer leak-bounded.
            wire.externally_driven = _FOOT_SLOTS
            wire_hooks["legs"] = override
            _log(f"LEGS: ENGAGED {clip.describe()} under the running planner · speed {player.speed:g}x · "
                 f"loop {'on' if player.loop else 'off'} · clock {1.0 / step_dt:.0f} Hz · lead-in "
                 f"{gaps[0] * 1e3:.0f}/{gaps[1] * 1e3:.0f} mm over {player.lead_in_steps} step(s) · "
                 f"legs {'ON (ref_joints)' if player.legs_enabled else 'OFF'} · armed={_armed_now()} · "
                 f"planner {'RUNNING' if auto_running.is_set() else 'idle'}")
            _log("LEGS: " + _walk_demand(player.speed))
            _log("LEGS: the planner's three upper slots keep publishing as before; its own foot target keeps "
                 "being committed (not leak-bounded while overridden) and is what the hand-back pursues.")
            if wire.slew.feet_slewed is True:
                _log("⚠️  LEGS: this low level reports that its slew caps COVER the feet (RecordTap "
                     "vr_cmd_slewed) — the gait will be clamped to a crawl. Relaunch with SLEW_FEET=0.")
            _draw_walk_paths(clip)
            t_next = time.monotonic()
            last_ui = 0.0
            unarmed_since: float | None = None
            n_tick = 0
            while True:
                if legs_ctl["release_req"] and not override.stopping:
                    legs_ctl["release_req"] = False
                    override.request_release()
                    _log(f"LEGS: hand-back requested at {override.progress()} — pursuing the planner's feet")
                n_tick += 1
                if n_tick % 5 == 0 and not override.stopping:
                    # A DISARM mid-gait: hand back on our own, so the next arm never snapshots a
                    # mid-stride foot offset (see the refusal above). 0.5 s of agreement, not one frame.
                    armed = _armed_now()
                    now = time.monotonic()
                    if armed is False:
                        unarmed_since = now if unarmed_since is None else unarmed_since
                        if now - unarmed_since >= 0.5:
                            override.request_release()
                            _log("LEGS: the low level DISARMED — handing the legs back so it never arms "
                                 "into a mid-stride foot wire")
                    else:
                        unarmed_since = None
                player.set_speed(walk_ctl["speed"])
                player.set_loop(walk_ctl["loop"])
                # STOP (latched) freezes the gait where it is; a requested hand-back still proceeds.
                override.tick(hold=stop.is_set())
                if override.done:
                    # WAIT FOR ONE MORE SUBSTITUTED FRAME before letting go: the landing tick put the
                    # feet exactly on the planner's target, but the publishers (60 Hz) may not have
                    # sampled it yet, and uninstalling first makes the seam the tick BEFORE the landing
                    # (measured 0.3-1.3 mm — inside the ramp bound, and avoidable for nothing).
                    # Bounded, because a wire with no publisher at all must not hold the hook: the
                    # heartbeat alone is 60 Hz, so 0.25 s is ~15 chances.
                    n0, t_land = override.n_frames, time.monotonic()
                    while override.n_frames == n0 and (time.monotonic() - t_land) < 0.25:
                        time.sleep(0.005)
                    note = (f"legs handed back — landed on the planner's own feet after "
                            f"{override.n_ticks} tick(s) / {time.time() - t_start:.1f} s "
                            f"({player.laps} full lap(s), {override.n_frames} frame(s) substituted"
                            f"{'' if override.n_frames > n0 else ', ⚠️ nothing published the landed frame'})")
                    break
                now = time.monotonic()
                if now - last_ui > 0.2:
                    last_ui = now
                    g_legs.value = ("ENGAGED · " + override.describe()
                                    + (" · STOP latched: frozen" if stop.is_set() and not override.stopping else ""))
                    try:
                        _draw_walk_marker(override.current()[0])
                    except Exception:  # noqa: BLE001, S110 - display only
                        pass
                t_next += step_dt
                delay = t_next - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    t_next = time.monotonic()   # fell behind: do not burst, re-anchor the clock
        except Exception as exc:  # noqa: BLE001 - an override error must not kill the page
            import traceback

            _log("LEGS: error:\n" + traceback.format_exc())
            note = f"leg override error: {exc!r}"
        finally:
            # Uninstall FIRST: from here the planner's frames go out whole (their feet == the landed
            # feet, so nothing jumps), then the clutch resumes bounding the two slots.
            wire_hooks["legs"] = None
            wire.externally_driven = frozenset()
            legs_ctl["override"] = None
            legs_engaged.clear()
            _clear_walk_paths()
            g_legs.value = note
            _log(f"LEGS: OFF — {note}")

    def _legs_toggle() -> str:
        """The 🦵 button's body — and the `l` hotkey's (tty_keys), so the two cannot drift."""
        if legs_engaged.is_set():
            legs_ctl["release_req"] = True
            g_legs.value = "handing back — the feet pursue the planner's own foot target (≤ 5 mm/tick), then it has them"
            return g_legs.value
        if walk_running.is_set():
            g_legs.value = "refused — the 🚶 walk has the feet (press k to end it first)"
            return g_legs.value
        g_legs.value = f"engaging {walk_clip.name[:40]} under the planner …"
        threading.Thread(target=_run_legs, daemon=True, name="legs").start()
        return g_legs.value

    @b_legs.on_click
    def _(_e) -> None:
        _legs_toggle()

    # ---------------- the RUN RECORDER (extension) ----------------
    # The ⏺ button: record ONE ▶▶ run — every chunk the planner drew, every executed step, every
    # frame that reached :8770 (reach residual and leg override APPLIED, i.e. as published), every
    # RecordTap frame with the robot's five keypoints FK'd from its joints, every F/T frame — into
    # <--record-dir>/<task>/<YYYY-MM-DD_HH-MM-SS>/ for scripts/plot_rollout_record.py (the xyz and
    # force comparison curves). It is a TAP and nothing else (flore/s1_policy/serving/rollout_record.py):
    # the reviewed page above holds `taps` and its one-line calls at the publish funnel, the predict,
    # the two execute loops and the ▶▶ start; this block owns the button, the folder and the FK. Blue =
    # will not record, green = will / is recording. The recording is the operator's EPISODE (2026-09-20):
    # it opens at the first Predict (manual or ▶▶'s) once armed and ends ONLY on STOP, Reset, the low
    # level DISARMING (joystick start — RecordTap's vive_armed edge, seen by the recorder), the button
    # or Ctrl-C; a ▶▶ ending on its own (its chunks, the ceiling, ⏹, the low level dropping off) is an
    # event inside it, and manual Predict → Run presses are recorded like ▶▶'s. Reset returns the
    # button to blue/idle and the next armed Predict / ▶▶ opens a new folder. It never publishes,
    # commits, plans or reads the wire back, so a
    # rollout with the button green is the same rollout — and a leg override during it is recorded
    # as such (every published frame carries a `legs` flag; the edges become events).
    _rec_fk, _rec_fk_why = RP.KeypointFK.load(urdf_path=args.yam_urdf)
    if _rec_fk is None:
        _log(f"RECORD: measured keypoints unavailable ({_rec_fk_why}); measured.jsonl keeps the joints only")

    def _rec_measure(msg):
        """RecordTap dict -> the robot's measured ``(pos15, orn20)`` wire frame, on the recorder's own
        thread. A PRIVATE ``KeypointFK`` (yourdfpy mutates its scene per call, and the clutch's instance
        belongs to the drip thread) through the wire's own readout, which is pure — the identical
        arithmetic ``wire.measured_keypoints`` performs for the leak bound."""
        if _rec_fk is None:
            return None
        rframe = parse_record_frame(msg)
        if rframe is None or rframe.dof_pos is None:
            return None
        return wire.readout.from_bodies(*_rec_fk.bodies(rframe.dof_pos, lift_z=anchor.pelvis_z))

    def _rec_info() -> dict:
        """The run's LIVE settings, read once when its folder opens (the static ones are in `meta.static`)."""
        with force_lock:
            f_on, f_gate = force_ctl["enabled"], force_ctl["gate_n"]
        with replay_lock:
            rp_info = None if replay is None else {
                "zarr": os.path.basename(replay.path), "episode": int(replay_ctl["episode"]),
                "frame": int(replay_ctl["frame"]), "mode": str(replay_ctl["mode"])}
            source = "replay" if (replay is not None and replay_ctl["source"] == "replay") else "model"
        return {"exec_steps": int(exec_ctl["n"]), "auto_max_s": float(auto_ctl["max_s"]),
                "source": source, "replay": rp_info,
                "seed": seed_ctl.describe(), "force_cmd": bool(f_on), "force_gate_n": float(f_gate),
                "grip_cmd": bool(grip_enabled.is_set()), "reach": reach.describe(),
                "ensemble": str(stream_ctl["mode"]), "ensemble_tau": float(stream_ctl["tau"]),
                "replan_min": int(stream_ctl["replan_min"]),
                "walk_speed": float(walk_ctl["speed"]), "walk_loop": bool(walk_ctl["loop"]),
                "walk_legs": bool(walk_ctl["legs"]), "legs_override_at_start": bool(legs_engaged.is_set()),
                "clutch": wire.clutch.post.describe(), "slew": wire.slew.describe()}

    def _rec_git_sha() -> str:
        try:
            out = subprocess.run(["git", "-C", _FLORE_ROOT, "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=2.0)
            return out.stdout.strip() if out.returncode == 0 else "unknown"
        except Exception:  # noqa: BLE001 - provenance, never a stop
            return "unknown"

    _rec_static = {
        "checkpoint": str(args.checkpoint), "layout": str(loaded.layout.mode),
        "action_dim": int(loaded.layout.action_dim), "state_dim": int(loaded.layout.state_dim),
        "roles": list(roles), "force_roles": list(force_roles), "grip_roles": list(grip_roles),
        "held_roles": list(wire.held_roles), "wire_roles": list(hl_5pt.WIRE_ROLES),
        "wire_labels": list(hl_dense.FIVE_POINT_LABELS), "wire_bodies": list(hl_dense.FIVE_POINT_BODIES),
        "ready_pos": np.asarray(wire.idle()[0], dtype=np.float64).round(6).tolist(),
        "action_dt": float(ACTION_DT), "action_horizon": int(ACTION_HORIZON),
        "pelvis_z": float(anchor.pelvis_z), "urdf": str(args.yam_urdf),
        "post_anchor": str(args.post_anchor), "carry": str(args.carry), "reach_ki": float(args.reach_ki),
        "low_level_mode": str(wire.low_level_mode), "cadence_at_launch": str(args.cadence),
        "replay_zarr": (None if replay is None else os.path.abspath(replay.path)),
        "args": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                 for k, v in sorted(vars(args).items())},
        "git": _rec_git_sha(), "host": os.uname().nodename, "pid": os.getpid(),
        "python": sys.version.split()[0], "measured_fk": _rec_fk is not None,
    }
    _rec_task = (str(args.record_task).strip() if args.record_task
                 else rollout_record.task_from_checkpoint(args.checkpoint))
    # The camera VIEWS to record = the page's own camera subscriptions, one per image key the checkpoint
    # reads (a left_only checkpoint: camera0_rgb alone; a dual one: both) — so one or two D405s need no
    # flag, and a configured-but-unread bridge (camera R under a left_only model) is not recorded, exactly
    # as it is not observed. Every frame the bridge publishes while recording lands as the bridge's own
    # JPEG under camera/<key>/; --record-camera-every N thins it, 0 turns it off.
    _rec_cameras = [{"cam": i, "key": str(loaded.image_keys[i]), "url": subs[f"camera{i}"].url}
                    for i in range(int(loaded.num_cameras)) if f"camera{i}" in subs]
    recorder = rollout_record.RunRecorder(
        args.record_dir, _rec_task, static_meta=_rec_static, measure=_rec_measure, info_provider=_rec_info,
        log=_log, autoplot=bool(args.record_autoplot), plot_cmd=rollout_record.plot_argv,
        every_run=bool(args.record_every_run), cameras=_rec_cameras,
        camera_every=int(args.record_camera_every))
    taps.recorder = recorder
    _log(f"RECORD: ⏺ folders under {recorder.task_dir} — arm the button (blue → green), then Predict or ▶▶; "
         f"the recording runs across chunks and ▶▶ runs until STOP / Reset / the low level disarming (joystick "
         f"start) saves it. Plots "
         f"{'run in the background when a recording is saved' if args.record_autoplot else 'by hand (--no-record-autoplot)'}: "
         f"{recorder.plot_hint(recorder.task_dir / '<stamp>')}")
    if recorder.cameras:
        _log("RECORD: camera frames → <run>/camera/{" + ",".join(c["key"] for c in recorder.cameras) + "}/NNNNNN.jpg "
             f"+ camera/frames.jsonl — {len(recorder.cameras)} view(s), the bridge's own JPEG"
             + (f", every {recorder.camera_every}-th frame" if recorder.camera_every > 1 else ", every frame"))
    else:
        _log("RECORD: camera frames OFF (--record-camera-every 0)" if not int(args.record_camera_every)
             else "RECORD: no camera subscribed — no camera frames in the recordings")

    # Placed right under Control (viser orders siblings by `order`), where ▶▶ / ⏹ / STOP / Reset are.
    with server.gui.add_folder("⏺ Record a run", order=f_control.order + 0.5):
        _rec_label0, _rec_color0 = recorder.ui()
        b_rec = server.gui.add_button(
            _rec_label0, color=_rec_color0,
            hint="BLUE = not recording; click → GREEN = the next Predict / ▶▶ opens a recording. It stays "
                 "open across chunks — Predict → Run one chunk at a time, ▶▶ runs, ⏹, the seconds ceiling "
                 "all continue it — and ENDS on STOP, Reset, the low level DISARMING (joystick start) or a "
                 "click on this button → blue again with the folder's name. Reset → blue/idle; every "
                 f"recording gets its own time-named folder under {recorder.task_dir}. Recorded: predicted chunks (xyz of "
                 "all five points + the predicted wrench per hand), executed steps, every published "
                 "frame, RecordTap (joints → measured keypoints), the F/T stream, and every frame of each "
                 "camera the model reads (camera/<image_key>/NNNNNN.jpg, full-size JPEG). Plot with "
                 "scripts/plot_rollout_record.py <folder>.")
        c_rec_every = server.gui.add_checkbox(
            "Record EVERY run (stay armed)", initial_value=bool(args.record_every_run),
            hint="ON: no click needed — the first Predict / ▶▶ after each end opens the next folder. The "
                 "button still shows recording / saved.")
        g_rec = server.gui.add_text("recording", initial_value=recorder.describe()[:200], disabled=True)
    rec_ui = {"label": _rec_label0, "color": _rec_color0, "text": recorder.describe()[:200]}

    def _rec_refresh() -> None:
        """Button label + colour and the status line FROM the recorder's state (the page loop, 4 Hz,
        and every click). An update is sent only when the text changed, so an idle button costs
        nothing and a live count redraws at most 4 Hz."""
        label, color = recorder.ui()
        if label != rec_ui["label"]:
            rec_ui["label"] = label
            b_rec.label = label
        if color != rec_ui["color"]:
            rec_ui["color"] = color
            b_rec.color = color
        text = recorder.describe()[:200]
        if text != rec_ui["text"]:
            rec_ui["text"] = text
            g_rec.value = text

    def _rec_toggle() -> str:
        """The ⏺ button's body: idle/done → armed, armed → idle, recording → end this recording now."""
        note = recorder.toggle()
        _rec_refresh()
        _log(f"RECORD: {note}")
        return note

    def _rec_reset() -> None:
        """Reset's hook: a live recording is ENDED (Reset is an end gesture), then back to blue/idle (a
        different label than 'saved')."""
        note = recorder.reset()
        _rec_refresh()
        _log(f"RECORD: {note}")

    def _rec_shutdown() -> None:
        """Ctrl-C with a recording open: close it so the folder is complete (meta.json written)."""
        if recorder.state == "recording":
            _log("RECORD: " + recorder.run_end("page shutdown (Ctrl-C)"))

    @b_rec.on_click
    def _(_e) -> None:
        _rec_toggle()

    @c_rec_every.on_update
    def _(_e) -> None:
        recorder.every_run = bool(c_rec_every.value)
        _rec_refresh()

    with server.gui.add_folder("Bring-up report"):
        for key in sorted(report):
            server.gui.add_text(key[:24], initial_value=str(report[key])[:110], disabled=True)

    flags: dict = {"predict": False, "run": False, "reset": False, "ghost_step": 0}

    # ---------------- the sequential rollout ----------------
    def _run_to_completion() -> None:
        """predict → execute, repeatedly, until Stop / the ceiling. ONE arm gesture.

        WHAT IT DOES NOT CHANGE — the safety envelope. Every iteration goes through exactly the
        same ``_predict`` → ``_drip`` pair the buttons call: the same obs-freshness refusal, the
        same ``wire.sync`` re-base, the same per-step force gate re-read, the same ``wire.commit``
        + per-role anchor commit to the LAST EXECUTED step, the same ``busy`` mutex. It removes the
        click and nothing else.

        BOUNDED FOUR WAYS: ``⏹`` (retires the generation), ``STOP`` (also interrupts the chunk in
        flight, within one step), the wall-clock ceiling ``--auto-max-seconds`` (live on the
        slider), and the low level dropping off :8770 — plus the OPT-IN ``--ood-stop`` backstop
        (:func:`_obs_range_gate`, default off). What keeps the FEEDBACK LOOP honest is not a bound on
        the run at all: :func:`_belief_bound` corrects the observation toward the measured keypoints
        every iteration, so the loop stays in the training range instead of being stopped once it
        leaves it.
        """
        if auto_running.is_set():
            g_msg.value = "ignored — a ▶▶ run is already active; press ⏹"
            return
        if dripping.is_set():
            g_msg.value = "ignored — a chunk is executing"
            return
        if s0.num_clients == 0:
            g_msg.value = ("the low level is NOT connected to :8770 — start it in ABSOLUTE mode "
                           f"with MODE={wire.low_level_mode}")
            return
        with auto_lock:
            auto_ctl["running"] = True
            my_gen = int(auto_ctl["gen"])
            max_s = float(auto_ctl["max_s"])
        auto_running.set()
        b_auto_stop.disabled = False
        # THE ONLY PLACE THIS ROLLOUT CLEARS `stop`. A stop latched by an earlier abort must not
        # make the first iteration a silent no-op; every LATER `stop` is the operator's STOP and has
        # to both interrupt the drip and end the loop, so clearing it per iteration would make STOP
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
                    # Worse here than for one manual chunk: with a CARRIED command and no readback
                    # the clutch FREEZES at every boundary, so the wire simply stops advancing.
                    if wire.clutch.post.carrying and _record_frame() is None:
                        now = time.monotonic()
                        if now - openloop_warn["t"] > 5.0:
                            openloop_warn["t"] = now
                            _log("SEQUENTIAL: no usable RecordTap (RECORD=1?) — the carried "
                                 "keypoint command FREEZES at each boundary instead of advancing.")
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

    # ---------------- the CONTINUOUS (HATO) rollout ----------------
    def _run_streaming() -> None:
        """Sample on one thread, publish on another, and blend every chunk that covers this step.

        HATO's §IV-B deployment algorithm (``third_party/hato/inference_node.py``), on this wire.
        The sequential rollout above alternates "sample (wire FROZEN)" and "drip", so the robot
        holds for ``L / (L + E)`` of every cycle — measured 16.7 / 31.8 / 46.4% for a sampler
        costing 2 / 6 / 12 chunk steps against the default 16-step window. Here the two run in
        parallel and every step publishes, because a chunk is stamped with the step its row 0
        belongs to and several chunks therefore hold an opinion about the SAME future step.

        WHAT IS THE SAME AS THE SEQUENTIAL PATH — the entire safety envelope, deliberately:

        * the frame reaching the socket is built by ``wire.plan`` -> ``wire.frame`` and gated by
          ``wire.publish``'s ``is_safe_keypoint_command``, the same three calls;
        * the CLUTCH is the same object doing the same arithmetic. :func:`hl_stream.mini_chunk`
          hands ``KeypointClutch.apply`` exactly two rows (previous, current) and publishes row 1,
          which is ``p_base + (p_cur - p_prev)`` — the incremental form of the one constant offset
          a chunked run gets, so ``--post-anchor``/``--carry`` mean exactly what they meant;
        * ``wire.sync``'s measured leak bound still runs, at the same wall rate
          (:data:`STREAM_SYNC_EVERY`);
        * the per-step force gate is re-read on every re-send, so the switch still interrupts;
        * ⏹ / STOP / the seconds ceiling / the low level dropping off :8770 all bound it through
          the same ``auto_ctl`` generation the sequential rollout uses.

        WHAT IS DIFFERENT, and both are named on the page rather than left to be inferred:

        * ⚠️ THE EXECUTED PATH IS NOT ANY ONE CHUNK. It is a weighted mean of several, so the
          observation fed back (``obs_mode=last_target``) is a pose the model never predicted as
          such. That is the same assumption ACT's ``--temporal_agg`` and HATO both make, and both
          report it improves smoothness; it is an assumption all the same.
        * ⚠️ THE BLEND NEEDS OVERLAP. Measured, streaming cuts worst jerk 2.3x when 20-39 chunks
          are live, is a wash at 7-13, and is 1.4x WORSE at 3-6 — with few members, one arriving
          chunk moves the mean by a third of however much the members disagree. The overlap is
          ``H / L``, it is on the status line, and ``--replan-min-steps`` only ever lowers it.
        """
        if auto_running.is_set():
            g_msg.value = "ignored — a run is already active; press ⏹"
            return
        if dripping.is_set():
            g_msg.value = "ignored — a chunk is executing"
            return
        if s0.num_clients == 0:
            g_msg.value = ("the low level is NOT connected to :8770 — start it in ABSOLUTE mode "
                           f"with MODE={wire.low_level_mode}")
            return
        with auto_lock:
            auto_ctl["running"] = True
            my_gen = int(auto_ctl["gen"])
            max_s = float(auto_ctl["max_s"])
        auto_running.set()
        b_auto_stop.disabled = False
        stop.clear()
        staged["out"] = None
        step_clock.reset(0)
        stream = hl_stream.ChunkStream(ACTION_HORIZON, roles, mode=stream_ctl["mode"],
                                       tau=stream_ctl["tau"])
        sstats = hl_stream.StreamStats()
        # Handed to the page for its 4 Hz readout rather than formatted here on every step: the
        # publisher's job is the 33 ms deadline, and a status string is the page's.
        stream_ctl["stats"] = sstats
        stream_ctl["stream"] = stream
        step_dt = 1.0 / max(args.step_hz, 1.0)
        drip_dt = 1.0 / max(args.drip_hz, 1.0)
        sampler_live = threading.Event()
        sampler_live.set()

        def _still_running() -> bool:
            if stop.is_set():
                return False
            with auto_lock:
                return auto_ctl["gen"] == my_gen and bool(auto_ctl["running"])

        def _sample_once() -> tuple[object, int, str]:
            """One stamped sample. Returns ``(chunk|None, t_obs, note)``; never raises.

            ⚠️ It draws its own seed, unlike the pre-`--seed` code where the caller bumped the
            counter first. A pinned seed here does NOT make the stream's members identical — the
            observation advances between samples, which is the whole point of the cadence — but it
            does correlate them more than HATO's blend assumes, so pin it to A/B the scheduler and
            not to run a task.
            """
            # THE SAME GATE AS `_predict`, on the same observation, for the same reason: the
            # feedback loop that diverges is `obs_mode=last_target`, which belongs to the PLANNER
            # and not to whichever cadence is draining it. Unlike a failed sample (below, which the
            # blend rides out on the chunks it already has) this one ENDS the run — a stream that
            # keeps feeding chunks sampled from a runaway state into the ensemble is the failure
            # mode, not a gap in it.
            _belief_bound()
            refuse = _obs_range_gate()
            if refuse is not None:
                return None, 0, refuse
            seed = seed_ctl.next()
            with force_lock:
                enabled, gate_n = force_ctl["enabled"], force_ctl["gate_n"]
            t0 = time.monotonic()
            try:
                # Hold the interpreter against the IK worker for the sample, exactly as Predict
                # does. It does NOT hold it against the publisher — see hl_stream.sample_stamped,
                # which is why the stamp's critical section excludes the sample itself.
                with infer_gate.priority():
                    chunk, t_obs = hl_stream.sample_stamped(
                        planner, step_clock, seed=seed, force_enabled=enabled,
                        gate_n=gate_n, num_steps=args.num_steps)
            except Exception as exc:  # noqa: BLE001 - a failed sample must not end the run
                return None, 0, f"{type(exc).__name__}: {exc}"
            sstats.note_sample((time.monotonic() - t0) * 1e3)
            return chunk, t_obs, ""

        # WARM UP SYNCHRONOUSLY. Nothing may be published before a plan exists, and the first
        # frame has to land on the clutch base (== the low level's own pre-arm gate reference)
        # rather than wherever an empty stream would leave the wire.
        chunk0, t_obs0, why0 = _sample_once()
        if chunk0 is None:
            g_msg.value = f"continuous run NOT started — first sample failed: {why0}"
            _log(f"STREAM: refused to start, first sample failed — {why0}")
            with auto_lock:
                if auto_ctl["gen"] == my_gen:
                    auto_ctl["running"] = False
            auto_running.clear()
            b_auto_stop.disabled = True
            return
        try:
            wire_src = wire.sync(_record_frame(), _measured_force_n())
        except Exception as exc:  # noqa: BLE001 - a bad readback must not lose the run
            _log(f"wire re-base skipped: {exc!r}")
            wire_src = "last published"
        def _note_chunk(chunk, ms: float) -> None:
            """Feed the page's own chunk counters from a streamed sample.

            The span is computed on the DECODED positions rather than on ``wire.plan``'s output:
            the clutch and the readout differ from those by a rigid per-keypoint offset, which
            cancels in both an excursion and a per-step difference, so this is the same number for
            a fraction of the work and without touching the clutch off the publisher's thread.
            """
            stats["n"] += 1
            stats["ms"] = float(ms)
            try:
                pos = wire.decoded_positions(chunk)
                stats["span"] = hl_5pt.chunk_span(pos, hl_dense.FIVE_POINT_LABELS,
                                                  budget=wire.slew.budget,
                                                  covered=wire.slew.slew_mask).describe()
            except Exception as exc:  # noqa: BLE001 - a readout must not end a run
                _log(f"STREAM: chunk span unavailable: {exc!r}")

        stream.add(chunk0, t_obs0)
        _draw_paths(chunk0)
        _note_chunk(chunk0, sstats.sample_ms[-1] if sstats.sample_ms else 0.0)
        taps.chunk(seq=stats["n"], cadence="stream", decoded_pos=wire.decoded_positions(chunk0),
                   wrench={r: chunk0.role(r).wrench_pred for r in force_roles},
                   grip={r: chunk0.role(r).gripper_m for r in grip_roles}, t_obs=t_obs0,
                   sample_ms=stats["ms"], wire_src=wire_src)
        _log(f"STREAM: START — base {wire_src}, ensemble {stream.mode}"
             f"{f' tau {stream.tau:g}' if stream.mode == 'act' else ''}, ceiling {max_s:g} s, "
             f"H={ACTION_HORIZON}")

        def _sampler() -> None:
            """Keep the stream full. Its own thread; the publisher never waits on it."""
            last_obs = -(10 ** 9)
            last_draw = time.monotonic()
            try:
                while _still_running():
                    gap = stream_ctl["replan_min"]
                    if gap > 0 and step_clock.value - last_obs < gap:
                        time.sleep(step_dt)
                        continue
                    chunk, t_obs, why = _sample_once()
                    if chunk is None and why.startswith("REFUSED:"):
                        # Not a gap to ride out — the observation itself has left the training
                        # range. Retire this generation so the PUBLISHER's loop exits through its
                        # own `_still_running()` check, which holds the last frame on the way out.
                        stats["err"] = why
                        ood["why"] = why
                        with auto_lock:
                            if auto_ctl["gen"] == my_gen:
                                auto_ctl["running"] = False
                        break
                    if chunk is None:
                        stats["err"] = why
                        _log(f"STREAM: sample failed ({why}) — the blend keeps running on the "
                             "chunks it already has.")
                        time.sleep(0.1)
                        continue
                    last_obs = t_obs
                    try:
                        stream.add(chunk, t_obs)
                    except (ValueError, FloatingPointError) as exc:
                        sstats.refused += 1
                        _log(f"STREAM: REFUSED a chunk before it entered the blend ({exc}) — a bad "
                             "sample would otherwise poison every step it covers.")
                        continue
                    if t_obs + ACTION_HORIZON <= step_clock.value:
                        stream.note_late()
                    _note_chunk(chunk, sstats.sample_ms[-1] if sstats.sample_ms else 0.0)
                    taps.chunk(seq=stats["n"], cadence="stream", decoded_pos=wire.decoded_positions(chunk),
                               wrench={r: chunk.role(r).wrench_pred for r in force_roles},
                               grip={r: chunk.role(r).gripper_m for r in grip_roles}, t_obs=t_obs,
                               sample_ms=stats["ms"])
                    now = time.monotonic()
                    if now - last_draw >= STREAM_DRAW_MIN_S:
                        last_draw = now
                        # Handed to the page loop rather than drawn here: `paths` is also read by
                        # the triads checkbox on viser's callback thread, and the page loop is the
                        # one writer the rest of this file already assumes.
                        try:
                            draw_q.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            draw_q.put_nowait(chunk)
                        except queue.Full:
                            pass
                    if stream_ctl["ghost"]:
                        _queue_ik(chunk)
            except Exception:  # noqa: BLE001 - the publisher must not be left without a sampler
                import traceback

                _log("STREAM: sampler thread died:\n" + traceback.format_exc())
            finally:
                sampler_live.clear()

        threading.Thread(target=_sampler, daemon=True, name="stream-sample").start()

        t = 0
        prev = stream.at(0)
        starve_t0: float | None = None
        t0_wall = time.time()
        last_pub = None
        end_note: str | None = None
        # Written by the sampler THREAD, read by the publisher on the way out: the sampler is the
        # one that sees the observation, the publisher is the one that owns the status line.
        ood: dict[str, str | None] = {"why": None}
        dripping.set()
        try:
            while _still_running():
                # The page's dropdowns write `stream_ctl`; the live stream object is local to this
                # run, so the setting is pushed in here rather than reached for from a callback.
                # Both setters validate, and a mid-run change takes effect on THIS step.
                if stream.mode != stream_ctl["mode"]:
                    stream.mode = stream_ctl["mode"]
                if stream.tau != stream_ctl["tau"]:
                    stream.tau = stream_ctl["tau"]
                # RE-READ, not the value captured at the start: there is no chunk boundary here, so
                # the slider takes effect on the next step and can only ever stop the body sooner.
                with auto_lock:
                    max_s = float(auto_ctl["max_s"])
                elapsed = time.time() - t0_wall
                if elapsed >= max_s:
                    end_note = (f"continuous run finished — {max_s:g} s ceiling "
                                f"({sstats.steps} step(s), {sstats.samples} sample(s))")
                    break
                if s0.num_clients == 0:
                    end_note = (f"continuous run STOPPED — the low level dropped off :8770 "
                                f"after {sstats.steps} step(s)")
                    _log("STREAM: the low level disconnected from :8770 — run ended.")
                    break
                if not sampler_live.is_set() and stream.live_at(t) == 0:
                    end_note = ("continuous run STOPPED — the sampler thread is gone and the "
                                "stream is empty")
                    break
                now = time.monotonic()
                cur = stream.at(t)
                if cur is None:
                    # Nothing covers this step: HOLD. The wire stays valid and the low level keeps
                    # its last target, but the motion has stopped, so this is bounded rather than
                    # ridden out — a sampler that cannot cover its own horizon will not recover.
                    sstats.frozen += 1
                    starve_t0 = now if starve_t0 is None else starve_t0
                    with wire_lock:
                        hold = tuple(a.copy() for a in last_wire["frame"])
                        hfc = None if last_wire["fcmd"] is None else last_wire["fcmd"].copy()
                    with force_lock:
                        on = force_ctl["enabled"]
                    _send(hold, hfc if on else None)
                    if now - starve_t0 > STREAM_STARVE_S:
                        end_note = (f"continuous run STOPPED — no chunk covered step {t} for "
                                    f"{STREAM_STARVE_S:g} s; the sampler cannot keep up with "
                                    f"H={ACTION_HORIZON}")
                        _log(f"STREAM: starved at step {t} — {sstats.describe(step_dt)}")
                        break
                    time.sleep(min(drip_dt, step_dt))
                    continue
                starve_t0 = None
                if prev is None:
                    prev = cur          # first published frame: zero delta, so it IS the base
                plan = wire.plan(hl_stream.mini_chunk(prev, cur, roles))
                frame = wire.frame(plan, 1)
                # Published with the hands' reach residual (--reach-ki; identity when off); `frame`
                # stays what the base commits to — the same split as `_drip`.
                pub = _reach_apply(frame)
                rows = {r: cur.wrench.get(r) for r in force_roles}
                fcmd = _gate_force(rows)
                _set_wire(pub, fcmd)
                taps.step(-1, t, frame, rows, n_blended=cur.n_blended)
                if grip_enabled.is_set() and s0.num_clients > 0:
                    for role in grip_roles:
                        width = cur.gripper_m.get(role)
                        if width is None or not np.isfinite(width):
                            continue
                        with jaw_override.lock:
                            # Same rule as _drip: a manually held jaw drops the policy's width.
                            if jaw_override.intercept(role):
                                continue
                            frac, _sq = grip_cmd[role][0](float(width))
                            if np.isfinite(frac):
                                subs[f"grip_{role}"].send({"target_frac": float(frac)})
                                jaw_ever_commanded["v"] = True
                t_step = time.time()
                first_ok = True
                while True:
                    # Re-gated per re-send, exactly as the drip does, so flipping the force switch
                    # off acts as an interrupt within one step rather than at the next chunk.
                    sent = _send(pub, _gate_force(rows))
                    first_ok = first_ok and sent
                    if stop.is_set():
                        break
                    remaining = step_dt - (time.time() - t_step)
                    if remaining <= 0:
                        break
                    time.sleep(min(drip_dt, remaining))
                if not first_ok:
                    # A CHUNKED drip bounds a refusal to one chunk and re-plans; a stream advances
                    # its clutch base off every published step, so committing a frame the low level
                    # never received would integrate the run off a value nothing accepted. End it
                    # and let the stale gate hold — the wire's gate refusing at all means the
                    # stream has left the pelvis-frame sanity bound, not that one frame was noisy.
                    sstats.refused_frames += 1
                    end_note = (f"continuous run STOPPED — the wire REFUSED the frame at step {t}; "
                                "the stale gate is holding the robot")
                    _log(f"STREAM: refused frame at step {t} — run ended rather than integrating "
                         f"off a value the low level never received.")
                    break
                # COMMIT, and mark the clock in the SAME critical section: the stamp a concurrent
                # sample reads has to mean "the anchor is the target of step t". Both the decode
                # anchor (this layout's own observation) and the wire base advance to the step just
                # published, which is what makes the next step a delta from where the body IS.
                with step_clock.lock:
                    for role in roles:
                        planner.commit_role(role, cur.position[role], cur.rot6d[role])
                    step_clock.mark(t)
                wire.commit(frame)   # UNCOMPENSATED — the reach residual never enters the base
                prev = cur
                if t and t % STREAM_SYNC_EVERY == 0:
                    try:
                        wire.sync(_record_frame(), _measured_force_n())
                    except Exception as exc:  # noqa: BLE001 - a bad readback FREEZES the carry
                        _log(f"STREAM: leak-bound re-sync skipped: {exc!r}")
                sstats.note_step(None if last_pub is None else (now - last_pub) * 1e3,
                                 cur.n_blended, cur.newest_age)
                last_pub = now
                t += 1
            if end_note is None and ood["why"]:
                end_note = (f"continuous run STOPPED — {ood['why']} "
                            f"({sstats.steps} step(s), {sstats.samples} sample(s))")
            if end_note is None:      # ⏹ / STOP retired the generation, or STOP latched
                end_note = (f"continuous run STOPPED — {sstats.steps} step(s), "
                            f"{sstats.samples} sample(s)")
            g_msg.value = end_note
        except Exception as exc:  # noqa: BLE001 - a run error must not kill the page
            import traceback

            _log("STREAM: publisher error:\n" + traceback.format_exc())
            g_msg.value = f"continuous run error: {exc!r}"
        finally:
            dripping.clear()
            with auto_lock:
                if auto_ctl["gen"] == my_gen:
                    auto_ctl["running"] = False
            auto_running.clear()
            b_auto_stop.disabled = True
            _release_jaw_latches()
            # Wait for the sampler so a retired run can never `stream.add` into the next one's GPU
            # slot. Bounded and reported: a thread still inside `sample_chunk` holds CUDA state.
            for _ in range(100):
                if not sampler_live.is_set():
                    break
                time.sleep(0.05)
            if sampler_live.is_set():
                _log("STREAM: the sampler thread did not exit within 5 s — it is inside a sample; "
                     "the next run will queue behind it.")
            stream_ctl["note"] = sstats.describe(step_dt)
            stream_ctl["stats"] = None
            stream_ctl["stream"] = None
            _log(f"STREAM: END — {stream_ctl['note']} · {stream.stats()}")

    @b_predict.on_click
    def _(_e) -> None:
        flags["predict"] = True

    @b_run.on_click
    def _(_e) -> None:
        flags["run"] = True

    def _auto_start() -> str:
        """The ▶▶ button's body — and the `p` hotkey's while no run is active (tty_keys)."""
        # Its OWN thread, not the page's callback thread and not the render loop: either rollout
        # blocks for the whole take and both of those must stay responsive — STOP is on one.
        streaming = str(d_cadence.value).startswith("stream")
        replaying = _replay_active()
        if replaying and streaming:
            # A recorded episode has a cursor and one chunk per position; the HATO blend of several
            # overlapping recorded chunks is not a replay of anything. The chunked path plays it.
            streaming = False
            _log("replay: the ▶▶ cadence is 'stream' but the chunk source is the RECORDED episode — playing "
                 "it on the SEQUENTIAL cadence (a recorded chunk has a cursor, not a sampler).")
        target = _run_streaming if streaming else _run_to_completion
        # The RUN RECORDER brackets the run as an auto-run INSIDE the recording (its folder opens if
        # the ⏺ button is armed and none is open; the run's end is an event, NOT the recording's end —
        # that is STOP / Reset / disarm); with no recorder installed this is the body unchanged.
        cadence = "replay" if replaying else ("stream" if streaming else "sequential")
        target = taps.wrap_run(target, cadence, end_note=lambda: str(g_msg.value))
        # Echoed so a cadence chosen a while ago and forgotten is named at the moment it starts.
        if replaying:
            with replay_lock:
                _ep, _fr, _mode = int(replay_ctl["episode"]), int(replay_ctl["frame"]), str(replay_ctl["mode"])
            _left_s = (_replay_len(_ep) - _fr) * ACTION_DT
            g_msg.value = (f"starting the RECORDED REPLAY of ep {_ep} from frame {_fr} ({_mode}, "
                           f"~{_left_s:.1f} s of demo) …")
            if _left_s > float(auto_ctl["max_s"]) + 1e-9:
                _log(f"⚠️  replay: ep {_ep} has ~{_left_s:.1f} s left but the ▶▶ ceiling is "
                     f"{auto_ctl['max_s']:g} s — the run will stop at the ceiling; raise '▶▶ run seconds'.")
        else:
            g_msg.value = ("starting the CONTINUOUS (HATO) run …" if streaming
                           else "starting the SEQUENTIAL run …")
        threading.Thread(target=target, daemon=True,
                         name="replay" if replaying else ("stream" if streaming else "seq")).start()
        return g_msg.value

    def _auto_retire() -> str:
        """The ⏹ button's body — and the `p` hotkey's while a run is active (tty_keys)."""
        # Retire the loop by bumping the generation: the chunk in flight finishes (bounded by
        # exec-steps) and the loop then exits without re-planning. Use STOP to interrupt the motion
        # ITSELF within one step.
        with auto_lock:
            auto_ctl["running"] = False
            auto_ctl["gen"] += 1
        return "⏹ — the run retires at the chunk boundary (the chunk in flight finishes); p again resumes"

    def _stop_now() -> str:
        """The STOP button's body — and the SPACE hotkey's (tty_keys)."""
        # Flag only; the loop thread owns the state. `stop` is read by the drip loop on every step
        # and by its inner re-send, so it interrupts an executing chunk within one step.
        stop.set()
        # STOP also RETIRES a sequential rollout, closing the window where the loop is between
        # iterations: it can then never re-plan after a STOP, only exit.
        with auto_lock:
            auto_ctl["running"] = False
            auto_ctl["gen"] += 1
        _release_jaw_latches()
        # The heartbeat OUTLIVES a STOP (it exits on `shutdown`, not `stop`), so what it re-sends
        # from here on is the frozen wire — and that must be PURE POSITION. Dropping F^cmd makes the
        # next heartbeat the last force the low level is asked for; keeping it would go on pressing
        # the last predicted wrench against a target that no longer moves.
        with wire_lock:
            last_wire["fcmd"] = None
        # The INTENT is abandoned: a reach residual that survived would be applied to whatever the
        # next chunk asks for. The frozen wire keeps the residual it was published with.
        _reach_reset("stop")
        # STOP is one of the operator's END gestures for the RUN RECORDER (the recording is the
        # episode: a ▶▶ ending on its own, ⏹ or a manual Run does not end it). Closed on a helper
        # thread — STOP never waits on the disk. The next armed Predict / ▶▶ opens a NEW folder.
        _rec_note = taps.end("STOP")
        if _rec_note:
            _log(f"RECORD: {_rec_note}")
        return "STOP — the wire is frozen within one step and F^cmd dropped; p (▶▶) resumes from here, no Reset"

    @b_auto.on_click
    def _(_e) -> None:
        _auto_start()

    @b_auto_stop.on_click
    def _(_e) -> None:
        _auto_retire()

    @b_stop.on_click
    def _(_e) -> None:
        _stop_now()

    @b_reset.on_click
    def _(_e) -> None:
        flags["reset"] = True

    # ---------------- terminal HOTKEYS (extension; flore/s1_policy/serving/tty_keys.py) ----------------
    # The page's most-pressed buttons on the keyboard of the terminal that launched it. Each key calls
    # the SAME closure its button does (`_auto_start` / `_auto_retire` / `_walk_toggle` / `_legs_toggle`
    # / `_stop_now`, and `_jaws_both` = the Manual-jaws ⬛/⬜ bodies for both hands at once); the only
    # logic of its own is the ▶▶/⏹ pairing on ONE key and two orderings the buttons' own threads would
    # refuse anyway (a walk during a run, a run during a walk), answered here without spawning them so
    # the hotkey log names the remedy. Reads stdin only — holosoma's keyboard is ITS terminal's stdin
    # (sshkeyboard), so nothing typed here reaches the low level, and the bindings sit outside its map.
    def _jaws_both(frac: float) -> str:
        """The Manual-jaws ⬛ / ⬜ buttons' bodies for BOTH hands in one press — the `c` (close) and `u`
        (open) hotkeys (tty_keys, 2026-09-21). Per hand it is exactly the button's two statements — the
        slider, then `_manual_jaw`, which latches the OVERRIDE for that hand until Reset — so one key and
        two clicks put the same frames on the same subscribers. A hand whose bridge is down is refused by
        name inside `_manual_jaw` and the other hand still moves; the note names both."""
        notes = []
        for role in MANUAL_JAW_ROLES:
            s_jaw[role].value = frac
            notes.append(_manual_jaw(role, frac))
        note = " · ".join(notes)
        g_msg.value = note
        return note

    def _key_jaws_close() -> str:
        return _jaws_both(0.0)

    def _key_jaws_open() -> str:
        return _jaws_both(1.0)

    def _key_run() -> str:
        if auto_running.is_set():
            return _auto_retire()
        if walk_running.is_set():
            return "a WALK is playing — press k to stop it (the feet ramp back), then p"
        return _auto_start()

    def _key_walk() -> str:
        if auto_running.is_set():
            return "a ▶▶ run is active — press p to pause it at the chunk boundary, then k"
        return _walk_toggle()

    keys = tty_keys.TtyKeys({"run": _key_run, "walk": _key_walk, "legs": _legs_toggle, "stop": _stop_now,
                             "jaws_close": _key_jaws_close, "jaws_open": _key_jaws_open},
                            log=_log)
    if not args.keys:
        g_keys.value = "off (--no-keys)"
        _log("hotkeys: OFF (--no-keys)")
    elif keys.start():
        g_keys.value = tty_keys.describe_bindings(keys.bindings) + "  — in the TERMINAL running this page"
        _log(keys.help())
    else:
        g_keys.value = f"unavailable — {keys.reason}"
        _log(f"hotkeys: unavailable — {keys.reason}")

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

    @c_reach.on_update
    def _(_e) -> None:
        reach.active = bool(c_reach.value)
        if not reach.active:
            _reach_reset("compensation toggled off")

    @c_ghost.on_update
    def _(_e) -> None:
        if ghost_root is not None:
            ghost_root.visible = bool(c_ghost.value)

    @c_triads.on_update
    def _(_e) -> None:
        _set_triads_visible(bool(c_triads.value))

    @d_ens.on_update
    def _(_e) -> None:
        # Written to `stream_ctl` only; the running publisher picks it up on its next step. An
        # unknown value cannot get through (the dropdown's options ARE ENSEMBLE_MODES) but it is
        # validated on the way into the stream anyway.
        if str(d_ens.value) in hl_stream.ENSEMBLE_MODES:
            stream_ctl["mode"] = str(d_ens.value)

    @s_tau.on_update
    def _(_e) -> None:
        stream_ctl["tau"] = float(s_tau.value)

    @s_replan.on_update
    def _(_e) -> None:
        stream_ctl["replan_min"] = max(0, int(s_replan.value))

    @c_sghost.on_update
    def _(_e) -> None:
        stream_ctl["ghost"] = bool(c_sghost.value)

    # ---------------- the loop ----------------
    _log(f"page: http://{args.host}:{args.port}")
    _log("sequence: press A on the joystick (ready pose) → Predict → arm (start) → then either "
         "'Run staged chunk' for ONE chunk, or '▶▶ Run to completion' to keep going until ⏹ / "
         f"the {args.auto_max_seconds:g} s ceiling.")
    _log(f"▶▶ cadence starts on {args.cadence.upper()}"
         + ("" if args.cadence != "stream" else
            f" — asynchronous sampling, ensemble {args.ensemble}"
            f"{f' tau {args.ensemble_tau:g}' if args.ensemble == 'act' else ''}"
            f"{f', >= {args.replan_min_steps} step(s) between samples' if args.replan_min_steps else ''}"
            f", violet body {'ON' if args.stream_ik_ghost else 'off'} while streaming")
         + ". The page's '▶▶ cadence' dropdown is the live switch; SEQUENTIAL is the reviewed "
           "path and the default.")
    last_status = 0.0
    last_cam = 0.0
    while True:
        try:
            # A sequential rollout owns Predict/Run/Reset for its duration. It holds `busy` only
            # WITHIN a chunk, so without these three refusals a manual press could land in the gap
            # between two iterations — a Predict would overwrite the chunk the rollout is about to
            # drip, and a Reset would clear the operator's latched STOP and re-home the decode
            # anchor underneath a live take.
            if auto_running.is_set() and (flags["predict"] or flags["run"] or flags["reset"]):
                flags["predict"] = flags["run"] = flags["reset"] = False
                g_msg.value = ("a ▶▶ run is active — press ⏹ to end it (or STOP to interrupt the "
                               "motion now)")
            # The WALK skill owns the wire for its duration for the same reason: a Reset here would
            # re-home the clutch base under a live foot command. (`walk_running` is set only while
            # the walk thread runs, so this line is inert whenever the button has not been pressed.)
            if walk_running.is_set() and (flags["predict"] or flags["run"] or flags["reset"]):
                flags["predict"] = flags["run"] = flags["reset"] = False
                g_msg.value = ("the WALK skill is active — press its button again to stop it (the "
                               "feet ramp back), or STOP to freeze the wire now")
            # The LEG OVERRIDE leaves Predict/Run alone (that is its point) and refuses only a Reset:
            # re-homing the clutch base and the decode anchor under a live foot substitution would
            # leave the hand-back pursuing a stance the robot is not at.
            if legs_engaged.is_set() and flags["reset"]:
                flags["reset"] = False
                g_msg.value = ("the LEG OVERRIDE is engaged — press l (or its button) to hand the legs "
                               "back, then Reset")
            # The newest STREAMED chunk's five paths, redrawn here so `paths` keeps exactly one
            # writer. Throttled on the producer side (STREAM_DRAW_MIN_S), depth 1, newest wins.
            try:
                _draw_paths(draw_q.get_nowait())
            except queue.Empty:
                pass
            if flags["reset"]:
                flags["reset"] = False
                # ARMED ⇒ REFUSED, whole (RESET_REFUSED_ARMED names the failure this prevents). The
                # old half-reset — decode anchor + clutch to the ready pose, wire held — read as
                # careful and WAS the bug: holding the wire protects the instant of the press, and
                # the very next Predict publishes the ready frame as step 0 anyway.
                rframe = _record_frame()
                armed = rframe is not None and rframe.vive_armed
                if armed:
                    g_msg.value = RESET_REFUSED_ARMED
                    _log(RESET_REFUSED_ARMED)
                else:
                    stop.clear()
                    staged["out"] = None
                    wire.clutch.reset()
                    _reach_reset("reset to ready pose")
                    planner.seed_anchor(
                        np.stack([anchor.pos[r] for r in roles]),
                        np.stack([rotmat_to_rot6d(anchor.rotmat[r]) for r in roles]))
                    _open_jaws("reset")   # drops the latches too
                    # ... and hands every manually held jaw BACK to the policy — the ONLY release
                    # (⏹, STOP, a new ▶▶ and the jaw checkbox all leave the override in place).
                    released = jaw_override.release_all()
                    if released:
                        _log(f"reset: manual jaw override released for {list(released)} — the "
                             f"policy drives them again (just commanded open with the others)")
                    _clear_paths()
                    _reset_ghost()
                    flags["ghost_step"] = 0
                    # THE SEED, which every other line here already had a counterpart for. Without
                    # it a Reset restored the observation exactly (`seed_anchor` re-seeds the pose
                    # rings and clears `_geom_written`) and the next Predict still answered
                    # differently, because the sampler had moved on — the same gesture on a visibly
                    # different path, with nothing on the page to say why. No-op when --seed pinned it.
                    seed_ctl.reset()
                    # A recorded replay goes back to its episode's first playable frame with the
                    # anchor: "back to the start" means the demo starts over FROM the ready pose.
                    if replay is not None:
                        with replay_lock:
                            _ep = int(replay_ctl["episode"])
                        _first = int(replay.first_usable_frame(_ep) or 0)
                        if replay_ui:
                            _rp_set_widgets(_ep, _first)
                        _replay_seek(_ep, _first, "reset: cursor at the first playable frame, anchor at the ready pose")
                    # UN-ARMED, so the wire re-homes too. The low level ignores the wire until the
                    # next `start`, whose pre-arm gate then compares this ready frame against its own
                    # reference (0.1 mm apart) and passes; disarming already returned the hands to
                    # that reference, so anchor, clutch, wire and robot agree again — the startup
                    # state, which is the one every first run has shown to work.
                    _set_wire(wire.idle(), None)
                    _rec_reset()   # the ⏺ button back to blue/idle; the next armed ▶▶ opens a NEW folder
                    g_msg.value = ("reset: anchor, clutch, body and wire back at the ready pose"
                                   + ("" if rframe is not None else
                                      " ⚠️ RecordTap ABSENT — armed state UNKNOWN, assumed un-armed; "
                                      "run the low level with RECORD=1"))
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
                    g_msg.value = ("the low level is NOT connected to :8770 — start it in ABSOLUTE "
                                   f"mode with MODE={wire.low_level_mode}")
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

            # The violet body plays the solved chunk on a loop. It is the IK of the DECODED poses
            # (what the policy asked for), not of the clutched wire — and with the default clutch
            # those differ only by the constant per-keypoint offset step 0 removes, so the body
            # still starts on the robot. DISPLAY ONLY: nothing here reaches :8770.
            solved = preview.solved
            if solved is not None and ghost is not None and ghost_root is not None \
                    and ghost_root.visible and solved.n:
                i = flags["ghost_step"] % int(solved.n)
                flags["ghost_step"] = i + 1
                ghost_root.position = tuple(float(v) for v in solved.root_pos[i])
                ghost_root.wxyz = tuple(float(v) for v in solved.root_wxyz[i])
                out = staged["out"]
                jaws = {}
                for role in grip_roles:
                    widths = None if out is None else out["chunk"].role(role).gripper_m
                    # An UNKNOWN jaw renders OPEN, never 0.0 — 0.0 is the CLOSED end of this
                    # joint's travel, so the naive default draws a hand that is HOLDING something.
                    jaws[role] = (jaw_joint_from_width_m(float(widths[i]))
                                  if widths is not None and i < widths.shape[0]
                                  else JAW_JOINT_OPEN_M)
                ghost.update_cfg(remap_dof_to_urdf(
                    solved.dof[i], dof_remap,
                    left_jaw_m=jaws.get("left", JAW_JOINT_OPEN_M),
                    right_jaw_m=jaws.get("right", JAW_JOINT_OPEN_M)))

            now = time.monotonic()
            # LIVE CAMERAS (display only): the newest frame of every configured bridge + the model's crop.
            if live_cams and now - last_cam >= 1.0 / max(float(args.live_camera_hz), 1e-3):
                last_cam = now
                try:
                    _live_cameras_tick(now)
                except Exception as exc:  # noqa: BLE001 - display only, never the loop
                    if not live_cam_ui.get("failed"):
                        live_cam_ui["failed"] = True
                        _log(f"LIVE CAMERAS: panel refresh failed ({exc!r}) — later failures are not logged")
            if now - last_status > 0.25:
                last_status = now
                _, readout_txt = HL._obs_freshness(
                    planner, max_age_s=planner.obs_max_age_s,
                    grip_max_age_s=planner.grip_max_age_s, consumes_force=consumes_force)
                g_obs.value = readout_txt
                rframe = _record_frame()
                if rframe is None:
                    g_ll.value = (f"{s0.num_clients} client(s) on :8770 · RecordTap ABSENT "
                                  "(no RECORD=1) — a carried command FREEZES at each boundary")
                else:
                    g_ll.value = (f"{s0.num_clients} client(s) · running={rframe.running} "
                                  f"armed={rframe.vive_armed} servo={rframe.servo_on}")
                    if rframe.dof_pos is not None:
                        urdf.update_cfg(remap_dof_to_urdf(rframe.dof_pos, dof_remap))
                if stats["n"]:
                    g_chunk.value = (f"{stats['n']} chunk(s) · sample {stats['ms']:.0f} ms · "
                                     f"{stats['span']}"
                                     + (f" · last error {stats['err']}" if stats["err"] else ""))
                g_seed.value = seed_ctl.describe()
                names, state_row, written = planner.state_readout()
                g_state.value = (" ".join(f"{n}={v:+.3f}" for n, v in zip(names, state_row))
                                 + ("" if written else "  (SEED — flagged UNMEASURED)"))
                try:
                    rng = hl_5pt.state_range_report(names, state_row, state_stats)
                    # The LEASH beside the range, because they are the same fact twice: the belief
                    # leaving the band IS the belief leaving the robot, and the operator needs to
                    # see the correction working rather than infer it from the warning's absence.
                    leash = ""
                    if float(args.obs_leak_m) > 0 and hasattr(planner, "leak_anchor_toward"):
                        n_rebased = int(belief["n_clamp"])
                        rebased = f", re-based {n_rebased}x" if n_rebased else ""
                        leash = (f"  ·  belief {belief['lead_m'] * 1e3:.0f} mm ahead of measured "
                                 f"(leash {float(args.obs_leak_m) * 1e3:.0f} mm{rebased})")
                    g_range.value = rng.describe() + leash
                except ValueError as exc:
                    g_range.value = f"range check unavailable: {exc}"
                g_ik.value = preview.status() if ik is not None else "disabled"
                sst = stream_ctl["stats"]
                if sst is not None:
                    g_stream.value = sst.describe(1.0 / max(args.step_hz, 1.0))
                elif auto_running.is_set():
                    g_stream.value = "sequential run active (this cadence is idle)"
                else:
                    g_stream.value = f"idle · last: {stream_ctl['note']}"
                with wire_lock:
                    fc = last_wire["fcmd"]
                    frame = tuple(a.copy() for a in last_wire["frame"])
                g_wire.value = (f"{wire.status(frame)} · "
                                f"F^cmd {'OFF' if fc is None else f'{np.abs(fc).max():.1f} N'}")
                # The cap's provenance, always visible — not only when a chunk happens to exceed
                # it. An assumed budget and a measured one produce identical-looking ratios, so the
                # operator has to be able to see which one they are reading.
                g_slew.value = wire.slew.describe()
                g_reach.value = reach.describe()
                g_wlegs.value = _legs_status()
                for _role in MANUAL_JAW_ROLES:
                    g_jaw[_role].value = _jaw_readout(_role)
                _rec_refresh()
            time.sleep(0.02)
        except KeyboardInterrupt:
            _log("interrupted")
            shutdown.set()   # the heartbeat, the ingest and the IK worker exit on this, never on STOP
            keys.stop()      # restore the terminal (cbreak off) before anything else prints
            _rec_shutdown()  # a recording still open is closed and its folder kept
            # Release before main()'s finally stops the subscribers — a Ctrl-C mid-task must not
            # leave a hand clamped with nothing left running to open it.
            _open_jaws("shutdown", include_manual=True)
            for _ in range(200):
                pending = False
                for role in MANUAL_JAW_ROLES:
                    sub = subs.get(f"grip_{role}")
                    if sub is None:
                        continue
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
