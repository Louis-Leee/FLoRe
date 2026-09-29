#!/usr/bin/env python3
"""pysurvive -> WebSocket bridge feeding LIVE 2pt VR hand targets to the browser.

This is the live-teleop analogue of ``thirdparty/vive_umi/src/vive_umi/bridge.py``,
but it drives the **trained** ``holosoma_2pt_wbt`` policy in the mjswan browser
viz instead of HuMI's mink IK. It reads two VIVE trackers directly from
libsurvive/pysurvive on Linux, calibrates them onto the robot's DEFAULT-POSE
hand target, and serves the two obs terms the browser policy consumes live:

  * ``vr_2point_pos``  (6) — [L_xyz, R_xyz], robot PELVIS frame
  * ``vr_2point_orn``  (12) — [L_rot6d, R_rot6d], rot6d = first two COLUMNS of
                              the rotation matrix, C-order [r00,r01,r10,r11,r20,r21]

byte-for-byte the convention ``bake_motions._fk_into`` /
``HolosomaWBTPolicy._compute_vr_fk`` produce. Everything else in the 564-D obs
(command_lower_body, motion_ref_ori_b, projected_gravity, base_ang_vel,
joint_pos, joint_vel, actions) keeps coming from the injected ``idle_default``
clip + live sim state — this bridge ONLY supplies the live hands.

Why a WebSocket (not the ZMQ RPC vive_umi uses): the consumer is the browser
WASM runtime. The matching TS obs classes (``WbtVr2PointPosLive`` /
``WbtVr2PointOrnLive``) open a WS to this server and read the latest frame each
control step. WebSockets are exempt from the page's COOP/COEP isolation, so a
``ws://localhost`` connection from the cross-origin-isolated viz works directly.

Three design choices that differ from vive_umi (and fix its "rotation axes feel
slightly off" issue, diagnosed in the Opus design review):

  1. **No puck->hand axis permutation.** vive_umi applies HuMI's
     ``_TX_TRACKER_HAND`` (a non-Pika rig's cyclic axis permutation + wrist
     offset) before anchoring. That permutation is a guess for a bare puck and
     is the main suspect for the off-axis feel. Here the puck pose maps DIRECTLY
     to the pelvis-frame VR point (position) and wrist orientation; the only
     world transform is a pure yaw about Z (gravity is identity in both frames).

  2. **World-frame (left-multiply) relative orientation delta.** Output wrist
     orientation = ``delta_world * home_q`` where
     ``delta_world = cur_q_yawaligned * ref_q_yawaligned_inv`` is the puck's
     rotation since calibration expressed in the (yaw-aligned == pelvis-aligned)
     world frame. So rotating the puck about real vertical rotates the robot
     wrist about pelvis-Z, etc. — intuitive. vive_umi used a body-frame
     (right-multiply) delta ``home_q * (ref_inv * cur)`` whose rotation axes are
     the puck's own (mounting-dependent) axes, which is what felt "off".

  3. **Anchor = the trained default-pose pelvis-frame hand target**, NOT HuMI's
     ``stand`` hand site. The default pose's hands hang BELOW the pelvis
     (z=-0.155, see vive/home_anchor.json), matching ``idle_default``; calibrate
     the operator's hands to this so the robot holds still at the calibration
     pose. Per-hand position offset => the calibration pose maps EXACTLY to the
     home target (robot perfectly still), then live displacement is mirrored.

  4. **Rotated-lever offset FK.** The policy's vr_2point point is the CONTACT
     point ``wrist_origin + R(wrist)@offset`` (offset = VR_2POINT_OFFSETS, a
     ~0.18 m lever). ``_hand_command`` tracks that lever: the robot wrist origin
     follows the puck-origin displacement, and the lever ``R(wrist_q)@offset``
     swings with the live wrist orientation, so pos stays consistent with orn
     when you twist the hand in place (a pure twist sweeps the contact point on
     an arc). At calibration the lever cancels exactly (hold-still stays exact);
     with ``--orientation off`` the wrist is held at home so the lever is
     constant. Operator puck mounting: the per-hand calibration absorbs the
     constant lever, so you do NOT need to tape the puck at any particular point
     for POSITION tracking — but for the lever-arc to match reality the puck's
     orientation axes should roughly track your hand's, which the relative
     orientation anchor (#2) already handles.

Translation-first staging (per the user's request): with ``--orientation off``
(the default) the bridge streams live ``vr_2point_pos`` but holds
``vr_2point_orn`` constant at the default-pose home rot6d, so you can validate
position tracking before enabling live wrist rotation. Toggle live without
restarting: ``kill -USR2 <pid>`` or ``touch /tmp/vive_teleop.orientation``.
Re-calibrate the hand anchor: ``kill -USR1 <pid>`` or
``touch /tmp/vive_teleop.recalibrate`` while holding the calibration pose.

Deliberately depends only on numpy + websockets + pysurvive (no holosoma/mujoco
at runtime); the HOME anchor constants are loaded from vive/home_anchor.json and
all quaternion/rot6d math is inlined to match holosoma byte-for-byte.

Run (the launcher script sets LD_LIBRARY_PATH/PYTHONPATH for ~/libsurvive)::

    # Serials MUST match the authoritative launcher (scripts/_vive_teleop_bridge.sh:
    # left=LHR-75106789, right=LHR-60CDD21F; left replaced 2026-08-08 after the
    # old LHR-4AE9DA27 unit died).
    # Reversing L/R rotates the whole world→pelvis yaw ~180° (push forward → robot
    # reaches backward), because the yaw auto-cal derives the frame from the L−R
    # hand vector — confirm with `--list-serials` if unsure.
    wbt-vive-teleop-bridge --left-serial LHR-75106789 --right-serial LHR-60CDD21F \
        --calib-delay 3                 # hold the calib pose ~3 s at startup
    wbt-vive-teleop-bridge --orientation on   # enable live wrist rotation too

``--mode 5pt-direct``: FIVE REAL TRACKERS, NO MOTION COMPLETION
==============================================================
The rig gained a 4th and 5th puck (paired 2026-09-09), so the five points
``exp:g1-wbt-5pt-hybrid-unified-anneal`` tracks can be MEASURED rather than inferred. This
mode reads all five -- both hands, the CHEST, and one puck per ANKLE -- and publishes the
**same** ``{"mode":"5pt", pos[15], orn[20]}`` frame ``--mode 5pt`` produces, so the low level
(``MODE=5pt-unified-anneal``), the browser preset and ``4_pose_match_viz.sh`` are all unchanged.

What it is NOT is a variant of ``--mode 5pt``: that mode reads THREE pucks and derives the
other two points from Motion_Completion. Deleting the completion deletes its whole cost --
the ``g1_mosyn`` backend and its ~10 s load, the ~1 s before the first command, and the
**667 ms lag floor** (arithmetic, not tuning: a 30-frame dense conditioning window with a
20-frame stride) -- and it also deletes the completion's null space, which is what made the
two FEET a channel nobody could aim:

===============  ==============================  ==============================
                 ``--mode 5pt`` (completion)     ``--mode 5pt-direct``
===============  ==============================  ==============================
pucks            3                               5
command lag      ~667 ms (structural)            one tracker frame
feet             inferred; 69-91 mm from the     MEASURED; exactly ON it at
                 pre-arm reference by            calibration
                 construction (the ready pose
                 is a 38 deg crouch)
backend          g1_mosyn child process          none
===============  ==============================  ==============================

Four things to know before running it:

* **The anchor is a DIFFERENT ROBOT POSE from 2pt/3pt's.** 5pt anchors on the IDLE_DEFAULT /
  READY pose (``home_pos_pelvis_5pt_tcp``: hands at ``(0.4418, +-0.1486, 0.0952)``), which is
  what the low level HOLDS under ``HOLOSOMA_INIT_READY_POSE=1`` and therefore what its pre-arm
  gate compares against; 2pt/3pt anchor on ``default_qpos``, **0.354 m per hand** away. Per-role
  anchoring
  then puts every commanded point exactly on that reference at calibration, so an
  ABSOLUTE-mode (``HOLOSOMA_VIVE_RELATIVE=0``) ``start`` arms by construction.
* **The CHEST decides which way is forward** (``--world-yaw-from``, default ``chest``). The
  whole path is ``home + Rz(world_yaw) . (puck - puck_at_calib)``, with no per-role position
  transform anywhere -- so ``world_yaw`` is the single frame decision, and getting it wrong
  turns EVERY point at once while leaving the start pose exact and every gate green. Its
  symptom is "I step forward and the robot goes sideways" (2026-09-10, on the rig). 2pt/3pt
  solve it from the L-R HAND vector; this mode does not, because here the operator is standing
  and walking and their hands may be parked, stacked or full -- and a parked pair still yields
  a confident, wrong frame. Instead the relabelled CHEST heading gives it (measured **+0.20
  deg** in the correct frame over 979 real frames, vs **+90.09** for the raw puck -- see
  ``BODY_RELABEL_M``), and the FEET cross-check it: a disagreement past
  ``--world-yaw-tol-deg`` REFUSES to calibrate, which is also what finally makes two ankle
  pucks on the wrong ankles (a clean 180 deg) something other than silent.
  ⚠️ Note the relabel is needed for the WITNESS and not for the command: a commanded pose is a
  calibration-relative delta ``(cur . ref^-1) . home`` in which a constant mount rotation
  cancels exactly. Reading an absolute HEADING is the one place it cannot cancel, which is the
  same reason the completion path needs ``mc_frames.chest_body_from_puck``.
* **The feet are STANCE, not GAIT**, for two independent reasons. (a) ``command_lower_body``
  still comes from the FROZEN idle clip, because this bridge sends no ``ref_joints`` -- the
  exact channel whose absence made a walking clip step in place on the rig (see
  ``walk_skill.py``). (b) More fundamentally, the wire is PELVIS-FRAME, so an operator WALKING
  is not expressible: a real step leaves every keypoint roughly where it was relative to the
  walker's own pelvis, whereas this mode feeds each puck's ABSOLUTE displacement, so walking a
  metre forward asks the robot to put its hands, chest AND feet a metre in front of its own
  pelvis. Feeding a stride instead would mean referencing every puck to the OPERATOR's root
  rather than to the room -- a different mode, not a flag on this one. Take single steps in
  place, and treat a whole-body translation as out of scope until then.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import shutil
import signal
import sys
import threading
import time

import numpy as np

# pysurvive is injected via PYTHONPATH by the launcher (it is NOT a pip dep, same
# as in vive_umi). websockets is a normal dependency in the hssim env.
import pysurvive  # noqa: E402
import websockets  # noqa: E402

from wbt_realtime_eval.vive import load_home_anchor

# Localhost-only endpoints (security: never bind a routable interface).
WS_HOST = "127.0.0.1"
WS_PORT = 8770
RECAL_FLAG = "/tmp/vive_teleop.recalibrate"
ORIENT_FLAG = "/tmp/vive_teleop.orientation"

#: This module's internal role names -> the role names UMI_collection's recorder and
#: converter use (`recording_format.TRACKER_FILE_BY_ROLE`). Only the RAW-tracker publisher
#: (`--umi-tracker-port`) renames anything: the teleop wire on :8770 keeps its own names.
#: `head` is UMI's spelling for the CHEST puck — a body reference, not a neck (the G1 has no
#: neck joint), which is why the file it lands in is `tracker_head.npy` and the zarr key is
#: `head_pos`.
UMI_TRACKER_ROLE = {
    "left_hand": "left",
    "right_hand": "right",
    "head": "head",
    # The two 5pt FOOT pucks (`--mode 5pt-direct` only). Collection's recorder writes them to
    # `tracker_left_foot.npy` / `tracker_right_foot.npy` under exactly these names
    # (`recording_format.TRACKER_FILE_BY_ROLE`), which are also FUJI's own five-keypoint role
    # names — so a teleop-recorded 5-puck session lands in the same files a collection-recorded
    # one does. Present unconditionally: the map is keyed on roles that only exist when the
    # mode activates them, and a missing entry here would drop a puck from the recording with
    # no error anywhere.
    "left_foot": "left_foot",
    "right_foot": "right_foot",
}

#: The FIVE roles `--mode 5pt-direct` reads, in the POLICY's own keypoint order
#: (`robot_config.VR_5POINT_BODIES` = left/right `wrist_yaw_link`, `torso_link`, then both
#: `ankle_roll_link`). The order IS the wire layout — slots [0:3] of a 5pt `pos` are
#: byte-identically the 3pt triple — so it is spelled once here and asserted against
#: robot_config by `tests/test_vive_5pt_direct.py`.
DIRECT_5PT_ROLES = ("left_hand", "right_hand", "head", "left_foot", "right_foot")

#: Roles whose keypoint is on the GROUND. `--z-offset` is an operator-height trim for the
#: HANDS (and, in 3pt, the head); adding it to a foot is not a trim, it is a command to leave
#: the floor, which on a balance controller is a hop rather than a calibration nudge. So the
#: two feet are excluded from it -- and only from it (`--reach-scale` still applies, since
#: scaling a stride is the same statement as scaling a reach).
DIRECT_5PT_GROUND_ROLES = ("left_foot", "right_foot")

#: The CHEST/FOOT puck -> BODY frame relabel, right-multiplied: ``R_body = R_puck @ H``.
#:
#: A body-worn puck is bolted onto its harness ROTATED, so its raw libsurvive axes are NOT
#: the body frame the keypoint means. ``H = Ry(90) @ Rx(90)``, i.e. ``x_body = -z_puck``,
#: ``y_body = +x_puck``, ``z_body = -y_puck``. Equal BY VALUE to `mc_frames.CHEST_RELABEL_R`,
#: to `collection_viz._HEAD_RELABEL_M` and to the rotation block of
#: ``robot_assets/calibration/head_extrinsic.json`` (what the converter applies, so it is what
#: every recorded ``head_rot_axis_angle`` carries) -- pinned to all three by
#: `tests/test_vive_5pt_direct.py`, and restated here for the same reason those three restate
#: each other: this module must stay import-cheap (no viser, no zarr, no UMI checkout on the
#: path). The two FEET share it, which is a DECLARATION about the ankle brackets ("mount them
#: like the chest one") rather than a measurement -- see `collection_viz._BODY_RELABEL_M`.
#:
#: ⚠️ It is NOT needed to BUILD a command, and that is not a contradiction. Every commanded
#: pose on this bridge is a calibration-relative delta ``(cur . ref^-1) . home``, in which a
#: constant body-local mount rotation cancels exactly (`_point_command_5pt`, pinned). What it
#: IS needed for is reading an ABSOLUTE HEADING off a body-worn puck -- the one number the
#: whole frame hangs on. Measured on ``session_2026.08.27_12.08.03/take_000`` (979 frames):
#: the RELABELLED chest heading sits at **+0.20 deg** in this bridge's yaw-aligned world
#: (std 0.41, i.e. the operator faces robot +x, which is what the frame is built to achieve)
#: while the RAW puck heading reads **+90.09 deg**. Skipping the relabel therefore turns the
#: frame by -90 deg, whose symptom is "operator FORWARD comes out as the robot's RIGHT" --
#: and it is invisible at frame 0, because the home and the readout share the same wrong
#: heading, so the start pose is still exact and the pre-arm gate still closes.
#: ⚠️ ORDER 3 (``H @ H @ H == I``): applying it twice does not cancel, it lands 120 deg away.
#: NOT the hands' symmetric ``-y,-x,-z`` (`mc_frames.PUCK_R_WRIST`) -- different puck,
#: different bracket.
BODY_RELABEL_M = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])

#: `BODY_RELABEL_M`'s first COLUMN = the body's own +x expressed in PUCK axes
#: (``x_body = -z_puck``). Rotating it by the puck's raw quaternion gives the body's FORWARD
#: direction in the world with no matrix build, because ``R_body @ x_hat == R_puck @ H @ x_hat``.
BODY_FWD_IN_PUCK = BODY_RELABEL_M[:, 0].copy()

#: Where `--mode 5pt-direct` may solve the world yaw from, i.e. which puck decides what
#: "the operator's FORWARD" means. Every other mode is hard-wired to `hands` and stays so.
#:
#:   * ``chest``  (DEFAULT) -- the relabelled CHEST heading. It is the operator's TORSO, it is
#:     strapped on rather than held, and its relabel is the one that was MEASURED (see
#:     `BODY_RELABEL_M`). This is the only witness that cannot be fooled by what the operator
#:     happens to be doing with their hands.
#:   * ``feet``   -- the LEFT-minus-RIGHT foot vector onto +y. Geometric (it reads puck
#:     ORIGINS, so no bracket rotation enters) and therefore the natural cross-check on the
#:     chest; it also flips by 180 deg if the two ankle pucks are swapped.
#:   * ``hands``  -- the historical L-R hand vector. What 2pt/3pt/5pt/dense use. Correct only
#:     while the two hand pucks ARE a lateral pair: park them, hold them stacked front-to-back,
#:     or hold them 2 cm apart and this reads a frame that is turned or noise.
#:   * ``fixed``  -- take ``--yaw-deg`` verbatim, no witness. The collection viz's own
#:     ``WORLD_RPY`` convention is this (a fixed Rz), which is why it needs no calibration pose.
DIRECT_5PT_YAW_SOURCES = ("chest", "feet", "hands", "fixed")

#: Fewest metres of horizontal separation a LATERAL witness needs before its direction means
#: anything. The historical hand autocal guards at ``1e-4`` on the SQUARED norm = **1 cm**,
#: which is a tracker-noise-dominated yaw rather than a frame -- deliberately left untouched
#: on the 2pt/3pt path (it is what those reviewed runs used) and replaced here, where two
#: further witnesses exist to disagree with it.
LATERAL_MIN_SEP_M = 0.10

#: Default RAW-tracker publish rate + staleness timeout, both quoted from
#: `vive_tracker_bridge`'s own defaults so a teleop-recorded `tracker.npy` and a
#: collection-recorded one have the same sampling and the same stale labelling.
UMI_TRACKER_RATE_HZ = 90.0
UMI_TRACKER_STALE_S = 0.25

#: Fewest SOLVED base stations we will calibrate against. Two is the real floor, not a
#: preference: a single Gen2 lighthouse constrains a sparse tracker only up to a rotation
#: about its own optical axis, so libsurvive can hold a "lock" whose yaw wanders — and yaw
#: is exactly what the world-yaw autocal reads off the L-R hand vector. One station
#: therefore produces a plausible-looking but slowly-drifting frame, which is far worse on
#: a robot than refusing to start.
MIN_STATIONS = 2

#: Where libsurvive caches the solved lighthouse geometry. Read-only here.
LIBSURVIVE_CONFIG = os.path.expanduser("~/.config/libsurvive/config.json")

#: How to get an UNSOLVED lighthouse solved. Stated once, because BOTH of the obvious
#: instructions are wrong, and they are wrong in opposite directions.
#:
#: The pose solve is fed by ``GlobalSceneSolver`` (a loaded driver), which captures a "scene"
#: only when ALL FOUR of these hold (``driver_global_scene_solver.c:224-229``)::
#:
#:     activations_changed = last_light_change != last_capture_time[i]
#:     spreadout           = (last_event_time - last_capture_time[i]) > timebase_hz * 3
#:     light_static        = (last_event_time - last_change) > timebase_hz * .1
#:     not_moving          = standstill_time > SurviveSensorActivations_default_tolerance * 16
#:     if (activations_changed && spreadout && light_static && not_moving) add_scenes(...)
#:
#: ``default_tolerance`` is ``48e6 * 16.7/1000 + 5000``, so ``not_moving`` is only **0.27 s**
#: of stillness — but ``activations_changed`` means the pose must have CHANGED since the last
#: capture, and ``spreadout`` puts ≥3 s between captures. So:
#:
#:   * Continuous waving never satisfies ``not_moving`` -> no scenes captured at all.
#:   * A puck left in ONE place yields exactly ONE scene, then ``activations_changed`` never
#:     fires again -> no parallax -> the bundle adjustment never converges.
#:   * What works is a SEQUENCE OF DISTINCT STILL POSES: move, pause, move, pause.
#:
#: (An earlier version of this text said "put them down and do not touch them", reading only
#: ``poser_mpfit.c:172``'s 1 s ``isStationary`` gate and missing that the global solver needs
#: the pose to keep changing. UMI_collection's runbook says "wave slowly", which works only
#: because slow waving happens to contain the pauses. The gate above is what both were
#: approximating.)
#:
#: OOTX is a separate, earlier stage with its own requirement: the frame arrives one bit per
#: sweep and any corrupted bit throws the whole accumulation away (``ootx_decoder.c:132-139``
#: -> ``ootx_reset_buffer``), so it needs UNBROKEN line of sight to that station, and nothing
#: produces a pose until every attached decoder finishes. That is the ``Bad sync bit`` loop,
#: and the fix for a station that will not decode from where the pucks can be held is to keep
#: its cached factory blob instead (``FRESH_OOTX=0`` / ``--no-fresh-ootx``) -- the blob is
#: burned into the station hardware and does not change when the station moves.
SOLVE_PROCEDURE = (
    "[bridge]     HOW (neither 'wave it' nor 'hold it still' on its own works -- the solver\n"
    "[bridge]     captures a scene only while the puck is STILL, but needs the pose to have\n"
    "[bridge]     CHANGED since the last capture, >=3 s earlier):\n"
    "[bridge]       1. Power on EVERY station you intend to use.\n"
    "[bridge]       2. Hold a puck where several stations see its sensor face. Then MOVE,\n"
    "[bridge]          PAUSE ~1 s, MOVE somewhere different, PAUSE -- a sequence of distinct\n"
    "[bridge]          STILL poses, spread around the workspace in position AND rotation.\n"
    "[bridge]          Continuous waving captures NOTHING (never still); one fixed spot\n"
    "[bridge]          captures ONE scene and then stalls (no parallax).\n"
    "[bridge]       3. Budget 30-60 s. Only ~1 scene lands every 3 s, and the OOTX handshake\n"
    "[bridge]          before it is ~25-35 s PER STATION on its own.\n"
    "[bridge]       4. Expect a 'base station LHn LIVE (solved)' line per station.\n"
    "[bridge]     STUCK on 'Bad sync bit' for minutes? That is OOTX, not the pose solve, and\n"
    "[bridge]     more waving will not fix it -- re-run with FRESH_OOTX=0 to keep the cached\n"
    "[bridge]     factory blobs and re-solve only the positions.\n"
    "[bridge]     Leaner than this bridge, if you only want the geometry back:\n"
    "[bridge]       ~/libsurvive/bin/survive-cli        # no MC backend, no policy, no WS"
)


def read_libsurvive_lighthouses(path: str = LIBSURVIVE_CONFIG) -> list[dict]:
    """Parse the cached lighthouse blocks out of libsurvive's config.

    Returns one dict per configured station: ``{"index", "id", "mode", "ootx", "solved"}``
    where ``solved`` is libsurvive's ``PositionSet`` — i.e. "this station's pose has been
    solved and can be used". Missing/unreadable file → ``[]`` (the caller then falls back
    to what it observes on the radio).

    Hand-rolled instead of ``json.load`` because **the file is not valid JSON**: it is a
    sequence of top-level entries with no enclosing braces and inconsistent trailing
    commas (``"floor-offset":"-1.014" `` has none). ``json.load`` fails at char 3. We only
    need four scalars per block, so scan for them rather than trying to repair the syntax
    — a repair would have to guess, and guessing wrong about ``PositionSet`` is the one
    field that must never be wrong.
    """
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return []
    out: list[dict] = []
    # Blocks look like:  "lighthouse0":{ ... }
    for m in re.finditer(r'"lighthouse(\d+)"\s*:\s*\{(.*?)\n\}', text, re.S):
        idx, body = int(m.group(1)), m.group(2)

        def _field(name: str, _body: str = body) -> str | None:
            fm = re.search(rf'"{name}"\s*:\s*"([^"]*)"', _body)
            return fm.group(1) if fm else None

        out.append(
            {
                "index": idx,
                "id": _field("id"),
                "mode": _field("mode"),
                "ootx": _field("OOTXSet") == "1",
                "solved": _field("PositionSet") == "1",
            }
        )
    return sorted(out, key=lambda d: d["index"])


def report_cached_stations() -> list[dict]:
    """Print the cached lighthouse geometry and return it. MUST be called BEFORE
    ``pysurvive.SimpleContext``, which rewrites the file the moment it starts."""
    lhs = read_libsurvive_lighthouses(LIBSURVIVE_CONFIG)
    if not lhs:
        print(
            f"[bridge] base stations: NO cached geometry in {LIBSURVIVE_CONFIG} -- libsurvive must "
            "solve every station from scratch.",
            flush=True,
        )
        print(
            "[bridge]     That takes an OOTX decode per station plus a position solve, so it needs "
            "the LIVE bridge -- it\n"
            "[bridge]     cannot finish inside a --list-serials scan window.",
            flush=True,
        )
        print(SOLVE_PROCEDURE, flush=True)
        return lhs
    solved = [lh for lh in lhs if lh["solved"]]
    print(f"[bridge] base stations (cached): {len(lhs)} configured, {len(solved)} solved", flush=True)
    for lh in lhs:
        print(
            f"[bridge]   LH{lh['index']}  id={lh['id']}  ch/mode={lh['mode']}  "
            f"ootx={'y' if lh['ootx'] else 'n'}  solved={'y' if lh['solved'] else 'N (needs --fresh)'}",
            flush=True,
        )
    return lhs


def backup_before_fresh() -> str | None:
    """Copy the config aside and say what ``--fresh`` is about to clear. Returns the backup
    path, or None when there was nothing to back up.

    Module-level rather than a method because BOTH entry points take ``--fresh`` and only the
    live-bridge one used to back anything up: `--list-serials --fresh` forwarded the flag
    straight to libsurvive, which cleared all three stations on this rig with no backup and no
    warning. The backup is the whole reason a wiped config is recoverable, so it cannot live on
    one of the two paths.
    """
    lhs = read_libsurvive_lighthouses(LIBSURVIVE_CONFIG)
    if not lhs:
        return None  # nothing cached to lose
    backup: str | None = None
    try:
        backup = f"{LIBSURVIVE_CONFIG}.bak_prefresh_{time.strftime('%m%d_%H%M%S')}"
        shutil.copy2(LIBSURVIVE_CONFIG, backup)
        print(f"[bridge] --fresh: backed the cached geometry up to {backup}", flush=True)
    except OSError as exc:
        backup = None
        print(f"[bridge] --fresh: WARNING could not back up {LIBSURVIVE_CONFIG}: {exc!r}", flush=True)
    print(
        f"[bridge] --fresh CLEARS THE POSE OF ALL {len(lhs)} CONFIGURED STATIONS, not just "
        "the ones you can see.\n"
        "[bridge]     Any station that is powered OFF right now will be left UNSOLVED and will "
        "NOT come back\n"
        "[bridge]     on its own -- you would have to --fresh again with it on. Power on every "
        "station you\n"
        "[bridge]     intend to use BEFORE this point.",
        flush=True,
    )
    print(SOLVE_PROCEDURE, flush=True)
    return backup


# -----------------------------------------------------------------------------
# Pure-numpy quaternion (wxyz) + rot6d helpers — match holosoma byte-for-byte.
# -----------------------------------------------------------------------------
def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions (apply b then a in world: a*b)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of a UNIT wxyz quaternion = bare conjugate [w,-x,-y,-z].

    Byte-identical to holosoma's ``quat_inverse`` (no normalization). Every call
    site here feeds an already-normalized quat, so the conjugate IS the inverse.
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector v by unit wxyz quaternion q (forward, == R(q) @ v).

    Matches holosoma's ``quat_apply``; used to project a wrist-local offset
    (the VR contact lever) into the world/pelvis frame: ``R(wrist) @ offset``.
    """
    w, x, y, z = q
    # q * (0,v) * q^-1, vector part.
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    # t = 2 * cross(q_xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array([vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx)])


def quat_z(theta: float) -> np.ndarray:
    """Yaw quaternion (rotation about +Z by theta), wxyz."""
    return np.array([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)])


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """wxyz quaternion for the intrinsic Z(yaw)·Y(pitch)·X(roll) rotation (radians).

    Quoted, composition order and all, from UMI_collection's
    ``bridges/tracker/vive_tracker_bridge.quat_from_rpy``, because this is the world
    alignment ``R_world`` that COLLECTION bakes into ``tracker.npy`` and the whole point of
    the raw-tracker publisher below is that a teleop-recorded session and a collection-
    recorded session come out in the SAME world frame. Two spellings of this rotation is how
    one of the two datasets ends up rotated by ``WORLD_RPY`` and nothing says so — the joint
    labels survive it (they are pelvis-relative) but every ``robot0_eef_*`` column does not.
    """
    qz = quat_z(yaw)
    qy = np.array([math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0])
    qx = np.array([math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0])
    return quat_normalize(quat_mul(qz, quat_mul(qy, qx)))


def rot_z_vec(theta: float, v: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector about +Z by theta."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Reorder a wxyz quaternion to xyzw — the 3pt ``vr_3point_orn`` rep.

    The 3pt policy's ``vr_3point_orn`` obs term is a RAW xyzw quaternion (3 bodies
    × 4), matching ``HolosomaWBTPolicy._compute_vr_fk``'s non-2pt branch
    (``wxyz_to_xyzw(local_quat_wxyz)``). The 2pt path keeps rot6d — this helper is
    only used by the 3pt snapshot.
    """
    q = quat_normalize(q)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _canon_xyzw(q: np.ndarray) -> np.ndarray:
    """Pick the ``w >= 0`` representative of an xyzw quaternion.

    ``q`` and ``-q`` are the SAME rotation but two different obs vectors, and every
    quaternion-orn preset (``vr_3point_orn`` / ``vr_5point_orn``) trained exclusively on
    the ``+w`` branch — na475b60's own normalizer puts ``vr_5point_orn``'s w at mean
    +0.73 (hands) / +0.98 (head) / +0.96 (feet) with std <= 0.17, i.e. unimodal and
    nowhere near zero. Feeding ``-q`` is therefore a ~20-sigma input, not a harmless
    alias, and it shows up as the arms snapping to a nonsense pose.

    Idempotent, and a bit-for-bit no-op on data that is already ``+w``.
    """
    return -q if q[3] < 0.0 else q


def quat_wxyz_to_rot6d(q: np.ndarray) -> np.ndarray:
    """First two COLUMNS of R(q), flattened C-order [r00,r01,r10,r11,r20,r21].

    Byte-for-byte ``matrix_from_quat(q_wxyz)[..., :2].reshape(-1)`` (holosoma)
    and the browser's ``wbtQuatToRot6d`` — the SAME 6D rep the bake stores. The
    quaternion MUST be wxyz; feeding xyzw silently produces a wrong matrix.
    """
    w, x, y, z = quat_normalize(q)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    return np.array([r00, r01, r10, r11, r20, r21], dtype=np.float64)


# -----------------------------------------------------------------------------
# WORLD-YAW WITNESSES (which way is the operator FACING?)
#
# Every commanded point on this bridge is `home + Rz(world_yaw) . (puck - puck_at_calib)`.
# `world_yaw` is therefore the ONE frame decision in the whole path: there is no per-role
# position transform anywhere (a foot and a hand are turned by the SAME Rz), so a wrong
# `world_yaw` cannot show up as "one keypoint is off" -- it shows up as EVERY commanded
# displacement being rotated, i.e. "I step forward and the robot goes sideways", with the
# start pose still exact and every gate still green.
#
# Two ways to read it off a standing operator, and they fail differently:
#
#   LATERAL  -- a LEFT-minus-RIGHT vector between a pair of pucks, rotated onto +y. Reads
#               puck ORIGINS only, so no mount rotation enters; needs the pair to actually BE
#               a lateral pair, and degrades to noise as the separation shrinks.
#   HEADING  -- a single body-worn puck's own forward axis, rotated onto +x. Needs the puck's
#               BODY RELABEL (`BODY_RELABEL_M`); works with one puck, at any separation.
#
# Having BOTH is the point: they are independent, so they can be made to disagree out loud
# instead of the operator discovering the frame from the robot's behaviour.
# -----------------------------------------------------------------------------
def wrap_angle(a: float) -> float:
    """Wrap a radian angle into (-pi, pi]."""
    return float((float(a) + math.pi) % (2.0 * math.pi) - math.pi)


def ang_diff_deg(a: float, b: float) -> float:
    """Signed smallest difference a-b, in DEGREES. Circular, so 359 vs 1 is +2 not +358."""
    return float(math.degrees(wrap_angle(float(a) - float(b))))


def circ_mean(angles) -> float:
    """Circular mean of radian angles (vector mean, not the arithmetic one).

    Averaging two headings that straddle +-pi arithmetically gives 0 -- exactly backwards --
    and the operator's facing is unconstrained in the libsurvive frame, so it straddles as
    often as not.
    """
    s = sum(math.sin(float(a)) for a in angles)
    c = sum(math.cos(float(a)) for a in angles)
    return float(math.atan2(s, c))


def body_heading_from_puck(q_raw_wxyz) -> float:
    """World heading (rad) of a BODY-worn puck (chest / foot), `BODY_RELABEL_M` applied.

    ``R_body = R_puck @ H``, so the body's forward is ``R_puck @ (H @ x_hat)`` =
    ``R_puck @ BODY_FWD_IN_PUCK``; project onto XY and atan2, which is `mc_frames.yaw_of_R`'s
    own definition (projecting is what keeps a leaning operator's PITCH out of the heading --
    the chest puck pitches 1.4-30.5 deg on the reference session).
    """
    fwd = quat_rotate(quat_normalize(np.asarray(q_raw_wxyz, dtype=np.float64)), BODY_FWD_IN_PUCK)
    return float(math.atan2(float(fwd[1]), float(fwd[0])))


def world_yaw_from_heading(heading_rad: float) -> float:
    """The ``Rz(world_yaw)`` that turns a measured world HEADING onto the robot's +x."""
    return wrap_angle(-float(heading_rad))


def world_yaw_from_lateral(v_left_minus_right, min_sep_m: float = LATERAL_MIN_SEP_M):
    """The ``Rz(world_yaw)`` that turns a LEFT-minus-RIGHT vector onto +y, or None.

    ``Rz(theta) @ [vx,vy]`` lands on +Y for ``theta = atan2(vx, vy)`` (and lands on +Y, not
    -Y, for either sign of v -- the y component comes out ``+|v|``). Returns **None** when the
    pair is closer than ``min_sep_m`` horizontally, because at that point the direction is
    tracker noise and a plausible-looking frame is worse than no frame.
    """
    v = np.asarray(v_left_minus_right, dtype=np.float64)
    if float(v[0] ** 2 + v[1] ** 2) < float(min_sep_m) ** 2:
        return None
    return wrap_angle(math.atan2(float(v[0]), float(v[1])))


# -----------------------------------------------------------------------------
# Bridge
# -----------------------------------------------------------------------------
class ViveTeleopBridge:
    """Holds the pysurvive singleton, calibrates, and computes the live 2pt
    pelvis-frame VR command (pos[6] + rot6d[12]) the browser policy consumes."""

    def __init__(self, args):
        self._args = args
        self._lock = threading.Lock()
        # 2pt (2 hand trackers) vs 3pt (+ a HEAD tracker at torso_link). Default 2pt
        # so the existing launch is byte-identical. In 3pt the snapshot emits pos[9]
        # + orn[12]=3×quaternion(xyzw); in 2pt it emits pos[6] + orn[12]=2×rot6d.
        self._mode = getattr(args, "mode", "2pt")
        self._is_3pt = self._mode == "3pt"
        # MOTION-COMPLETION modes (user 2026-08-27). `5pt` and `dense` drive
        # exp:g1-wbt-5pt-hybrid-unified-anneal (184D) and
        # exp:g1-wbt-hybrid-unified-anneal (183D), which want FIVE tracked points or all
        # 29 joint targets. The rig has THREE trackers, so the same 3-tracker stream goes
        # through Motion_Completion first and the extra points come out of its FK. See
        # mc_teleop_source -- including the 667 ms latency floor, which is a property of
        # that checkpoint's dense 30-frame conditioning and not a tuning knob.
        self._is_mc = self._mode in ("5pt", "dense")
        # DIRECT five-tracker teleop (user 2026-09-10). The rig gained a 4th and 5th puck
        # (paired 2026-09-09), so the FIVE points exp:g1-wbt-5pt-hybrid-unified-anneal tracks
        # can be MEASURED instead of inferred: this mode reads all five pucks and publishes the
        # SAME `{"mode":"5pt", pos[15], orn[20]}` wire `--mode 5pt` produces, with NO motion
        # completion in the path. So the low level, the browser preset and Terminal 3's
        # `MODE=5pt-unified-anneal` are all unchanged, and what goes away is the completion's
        # whole cost: the g1_mosyn backend, its ~10 s load, the 667 ms command-lag floor, the
        # ~1 s of nothing at the start, and the two FEET being an inference nobody can aim.
        #
        # ⚠️ It is a SEPARATE mode rather than an auto-detect on "five pucks are on the radio",
        # because which points the operator is PLACING is a safety-relevant claim: the low
        # level's pre-arm gate only checks 3 of 5 (`_VR_5POINT_OPERATOR_PLACED`, sized for the
        # MC path's inferred feet), so a mode that quietly switched would change what "armed"
        # means without saying so.
        self._is_5pt_direct = self._mode == "5pt-direct"
        # Everything that means "there is a HEAD/chest tracker on the radio" keys off
        # this, not off _is_3pt: the MC modes need the chest puck too (it is the model's
        # only root channel -- pelvis XY + heading).
        self._uses_head = self._is_3pt or self._is_mc or self._is_5pt_direct
        self._mc: object | None = None  # the MotionCompletionSource, built in run_pysurvive
        left = args.left_serial or os.environ.get("VIVE_UMI_LEFT_SERIAL", "")
        right = args.right_serial or os.environ.get("VIVE_UMI_RIGHT_SERIAL", "")
        head = getattr(args, "head_serial", None) or os.environ.get("VIVE_UMI_HEAD_SERIAL", "")
        lfoot = getattr(args, "left_foot_serial", None) or os.environ.get("VIVE_UMI_LEFT_FOOT_SERIAL", "")
        rfoot = getattr(args, "right_foot_serial", None) or os.environ.get("VIVE_UMI_RIGHT_FOOT_SERIAL", "")
        if left and right and left == right:
            print(
                f"[bridge] ERROR: left and right serials are identical ({left}); "
                f"left_hand would never map — pass distinct --left-serial/--right-serial.",
                flush=True,
            )
            sys.exit(2)
        if self._uses_head and not head:
            print(
                f"[bridge] ERROR: --mode {self._mode} requires a head/chest tracker — pass --head-serial "
                "or VIVE_UMI_HEAD_SERIAL (run --list-serials to find the unmapped tracker).",
                flush=True,
            )
            sys.exit(2)
        if self._uses_head and head in (left, right):
            print(
                f"[bridge] ERROR: head serial ({head}) duplicates a hand serial; the three trackers must be distinct.",
                flush=True,
            )
            sys.exit(2)
        if self._is_5pt_direct and not (lfoot and rfoot):
            _missing = " and ".join(
                n for n, v in (("--left-foot-serial", lfoot), ("--right-foot-serial", rfoot)) if not v
            )
            print(
                f"[bridge] ERROR: --mode 5pt-direct requires {_missing} (or "
                "VIVE_UMI_LEFT_FOOT_SERIAL / VIVE_UMI_RIGHT_FOOT_SERIAL). A GUESSED foot serial "
                "does not fail -- it records/commands a MIRRORED stance with every gate green, "
                "which is why there is no fallback. Run --list-serials to see which pucks are on "
                "the radio.",
                flush=True,
            )
            sys.exit(2)
        self._serial_to_role = {left: "left_hand", right: "right_hand"}
        if self._uses_head:
            self._serial_to_role[head] = "head"
        if self._is_5pt_direct:
            # Distinctness is checked over ALL FIVE, not just the pair: two roles sharing a
            # serial silently leaves the later one unfed forever (the dict collapses), and the
            # snapshot then holds HOME with no hint which puck is "missing".
            for _role, _ser in (("left_foot", lfoot), ("right_foot", rfoot)):
                if _ser in self._serial_to_role:
                    print(
                        f"[bridge] ERROR: serial {_ser} is mapped to both "
                        f"'{self._serial_to_role[_ser]}' and '{_role}'; the five trackers must be "
                        "distinct.",
                        flush=True,
                    )
                    sys.exit(2)
                self._serial_to_role[_ser] = _role
        # role -> (raw_pos[3], raw_quat_wxyz[4]) in libsurvive Z-up world.
        self._raw_hand: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # role -> wall-clock time of the last pose update (staleness detection).
        self._raw_hand_t: dict[str, float] = {}
        # role -> count of pose updates. Only the RAW-TRACKER publisher reads it (see
        # `umi_tracker_snapshot`): UMI's own vive_tracker_bridge stamps a per-role `seq` and
        # its recorder dedups on it, so a stream that reused one counter for three pucks
        # would look like two of them never moved.
        self._raw_hand_seq: dict[str, int] = {}
        # World alignment for the RAW-tracker feed ONLY (`umi_tracker_snapshot`). It is a
        # DIFFERENT rotation from the teleop path's `world_yaw`, and deliberately so: the
        # teleop autocal solves a yaw from the operator's own L-R hand vector at calibration
        # time, while collection bakes a FIXED `WORLD_RPY` (this rig: 0 0 180) into
        # tracker.npy so every recorded session shares one world frame. A recorded dataset
        # cannot use the autocal value — it changes with every recalibration.
        _urpy = getattr(args, "umi_world_rpy_deg", None) or (0.0, 0.0, 0.0)
        self._q_umi_world = quat_from_rpy(*(math.radians(float(a)) for a in _urpy))
        self._umi_stale_timeout = float(getattr(args, "umi_stale_timeout", UMI_TRACKER_STALE_S))
        self._both_seen_at: float | None = None
        self._calibrated = False
        self._mc_needs_reset = False

        # --- base stations, 2 or 3, decided by what is actually live ----------
        # `_stations_live` holds the LH names that have delivered a SOLVED pose; that set is
        # the gate, so nothing here has to be told how many stations the room has. The rig
        # runs 2 or 3 interchangeably.
        self._stations_live: set[str] = set()
        self._stations_configured: list[str] = []
        self._stations_ok_logged = False
        self._stations_wait_logged = False
        self._min_stations = max(1, int(getattr(self._args, "min_stations", MIN_STATIONS)))
        if self._min_stations < MIN_STATIONS:
            print(
                f"[bridge] ⚠️  --min-stations {self._min_stations} is BELOW the {MIN_STATIONS}-station "
                "floor. One Gen2 lighthouse leaves a sparse tracker's yaw under-constrained, and yaw "
                "is exactly what the world-yaw autocal reads off the L-R hand vector — expect a frame "
                "that looks locked and slowly drifts.",
                flush=True,
            )

        # --- HOME anchor (the default-pose 2pt VR target, pelvis frame) -------
        anchor = load_home_anchor()
        # [L,R] x 3 — the VR point (wrist + offset) at the default pose.
        self._home_pos = np.asarray(anchor["home_pos_pelvis"], dtype=np.float64)  # (2,3)
        # [L,R] x 6 — default-pose rot6d (the translation-first / pre-msg orn).
        self._home_orn6d = np.asarray(anchor["home_orn_rot6d"], dtype=np.float64)  # (2,6)
        # [L,R] x 4 — default-pose wrist quat (wxyz); the orientation HOME the
        # live relative delta is composed onto.
        self._home_wrist_q = np.asarray(anchor["home_wrist_quat_wxyz"], dtype=np.float64)  # (2,4)
        # [L,R] x 3 — VR contact-point offset in the wrist (puck) LOCAL frame.
        # The policy's vr_2point point is wrist_origin + R(wrist)@offset; we apply
        # this rigid lever so pos stays consistent with orn when the operator
        # twists the hand (the offset sweeps an arc). At the calibration pose the
        # constant lever is absorbed by the per-hand position anchor (hold-still
        # is exact); only its CHANGE under rotation matters.
        self._vr_offsets = np.asarray(anchor["vr_2point_offsets"], dtype=np.float64)  # (2,3)
        self._home_sep = float(anchor["hand_separation_m"])
        self._roles = ["left_hand", "right_hand"]  # index 0=L, 1=R (matches offsets)

        # --- 3pt HEAD anchor (only when --mode 3pt) --------------------------------
        # The 3rd point is torso_link + offset (0,0,0.35) in the pelvis frame. Its
        # home pos/orn come from the SAME bake FK as the hands (home_pos_pelvis_3pt
        # index 2, home_body_quat_wxyz_3pt index 2). Unlike the hands' rot6d, the 3pt
        # orn rep is a RAW xyzw quaternion (the policy's vr_3point_orn term). The head
        # tracker's raw puck pose is calibrated the same way (position anchor + a
        # relative-orientation delta onto the home head quat). We keep the FULL 3pt
        # home arrays so the head slot is filled even before the head tracker locks.
        if self._uses_head:
            self._roles = ["left_hand", "right_hand", "head"]  # index 0=L,1=R,2=head
            self._home_pos_3pt = np.asarray(anchor["home_pos_pelvis_3pt"], dtype=np.float64)  # (3,3)
            # (3,4) wxyz per-body home quat (hands + head); the orn HOME the live
            # relative delta composes onto, re-expressed as xyzw for the obs.
            self._home_body_q_3pt = np.asarray(anchor["home_body_quat_wxyz_3pt"], dtype=np.float64)  # (3,4)
            self._vr_offsets_3pt = np.asarray(anchor["vr_3point_offsets"], dtype=np.float64)  # (3,3)

        # --- HAND tracking point: 0.18 m VR lever vs 0.242 m fingertip TCP ---------
        # WHICH point on the hand this bridge steers. It must match the point the
        # LOADED POLICY was supervised on, because the two are 6.7 cm apart per hand
        # and NO dimension check can tell them apart (the TCP retrain kept the obs at
        # 170D/680D — only the term NAME `vr_3point_pos_tcp` differs). See
        # `_run_policy_vive_force.sh`'s pre-flight and robot_config.VR_3POINT_OFFSETS_TCP.
        #
        # WHY it matters even in RELATIVE/clutch mode (the default deploy path), where
        # the policy commands `FK(ref) + (live - live0)` and the absolute home anchor
        # therefore CANCELS: the surviving term is the rotated lever
        # `R(wrist_q)@offset - R(home_q)@offset`, whose radius IS this offset. Steering a
        # 0.18 m lever into a policy tracking 0.242 m makes a pure wrist twist sweep an
        # arc 6.2 cm too small, i.e. an inconsistent (pos, orn) pair the policy never saw.
        # In ABSOLUTE mode the home anchor does not cancel, so the offset matters twice.
        #
        # HEAD is UNCHANGED either way (torso_link +0.35 z is the same third point in
        # both geometries), which is exactly why only the two hand rows are swapped.
        self._hand_point = getattr(args, "hand_point", "lever")
        if self._is_5pt_direct and self._hand_point != "tcp":
            # The 5pt family is TCP-ONLY: no 0.18 m-lever 5pt checkpoint was ever trained
            # (`robot_config.VR_5POINT_OFFSETS_TCP` has no lever twin), so there is no geometry
            # to steer here and no home array to steer it to. Refuse by name rather than
            # silently promoting it -- the two are 6.7 cm apart per hand at an identical wire
            # width, which is exactly the mismatch --hand-point exists to make impossible.
            print(
                f"[bridge] ERROR: --mode 5pt-direct requires --hand-point tcp (got "
                f"{self._hand_point!r}). The 5pt family was only ever trained on the 0.242 m "
                "fingertip TCP; there is no 0.18 m-lever 5pt checkpoint.",
                flush=True,
            )
            sys.exit(2)
        if self._hand_point == "tcp":
            if not (self._is_3pt or self._is_mc or self._is_5pt_direct):
                print(
                    "[bridge] ERROR: --hand-point tcp requires --mode 3pt. The TCP-unified "
                    "geometry was only ever retrained for exp:g1-wbt-3pt-hybrid-unified"
                    "[-anneal]; no 2pt checkpoint tracks the 0.242 m TCP, so steering it "
                    "here would be a 6.7 cm/hand train-deploy mismatch.",
                    flush=True,
                )
                sys.exit(2)
            # Swap ONLY the two hand rows (offset + position home). The orientation home
            # (`_home_wrist_q` / `_home_orn6d`) is untouched: the TCP shares the wrist's
            # orientation, so the rot6d/quat home is identical in both geometries.
            self._vr_offsets_3pt = np.asarray(anchor["vr_3point_offsets_tcp"], dtype=np.float64)
            self._home_pos_3pt = np.asarray(anchor["home_pos_pelvis_3pt_tcp"], dtype=np.float64)
            self._vr_offsets = self._vr_offsets_3pt[:2].copy()   # (2,3) hands
            self._home_pos = self._home_pos_3pt[:2].copy()       # (2,3) hands
            self._home_sep = float(anchor["hand_separation_tcp_m"])

        # --- DIRECT 5-point anchor (only when --mode 5pt-direct) -------------------
        # Its OWN arrays, deliberately not an override of the 2pt/3pt ones: the two families
        # anchor on DIFFERENT ROBOT POSES and mixing them is a 0.354 m PER-HAND error that
        # nothing downstream can see. (home_anchor.json quotes 0.289 m for the same fact; that
        # is the RMS over the 3pt triple, where the chest contributes exactly 0 because
        # torso_link does not move when the arms are zeroed.)
        #
        #   * 2pt/3pt anchor on `default_qpos` (`home_pos_pelvis*`), the pose those older
        #     presets' reference clip holds -- hands at (0.266, +-0.252, -0.194).
        #   * 5pt anchors on the IDLE_DEFAULT / READY pose (`home_pos_pelvis_5pt_tcp`) --
        #     `default_dof_angles` with the 14 arm joints ZEROED, hands at
        #     (0.4418, +-0.1486, 0.0952). That is what the low level actually HOLDS under
        #     `HOLOSOMA_INIT_READY_POSE=1` (the launcher's default) and therefore what its
        #     pre-arm gate compares the wire against -- holosoma's own documented gate
        #     reference, to 0.1 mm.
        #
        # So per-role anchoring here means the calibration pose maps EXACTLY onto the gate's
        # reference for ALL FIVE points, i.e. the gate closes by construction. That is strictly
        # better than the MC path, where the two FEET land 69-91 mm out because the ready pose
        # is a 38 deg crouch and the completion stands straighter -- the measurement that forced
        # `_VR_5POINT_OPERATOR_PLACED = 3` in the first place. The low level still gates only 3
        # of 5 (untouched on purpose; that constant is load-bearing for the MC path), so the two
        # feet are a free check here rather than a gated one.
        if self._is_5pt_direct:
            self._roles = list(DIRECT_5PT_ROLES)
            # (5,3) pelvis-frame home keypoints and (5,4) wxyz home body quats.
            self._home_pos_5pt = np.asarray(anchor["home_pos_pelvis_5pt_tcp"], dtype=np.float64)
            _hq_xyzw = np.asarray(anchor["home_orn_quat_xyzw_5pt"], dtype=np.float64)  # (5,4)
            self._home_body_q_5pt = np.stack(
                [quat_normalize(np.array([q[3], q[0], q[1], q[2]])) for q in _hq_xyzw], axis=0
            )
            # The levers, built as the 3pt-TCP triple PLUS TWO ZERO ROWS rather than re-typed.
            # That spelling makes the prefix property structural (it mirrors robot_config's own
            # `VR_5POINT_OFFSETS_TCP = [*VR_3POINT_OFFSETS_TCP, feet, feet]`), and a foot
            # keypoint IS the bare `ankle_roll_link` origin: a hand lever models a physical
            # mount and the head lever lifts torso_link to HMD height, but a foot has neither,
            # so a nonzero sole offset would price a point nothing else in the stack knows.
            self._vr_offsets_5pt = np.concatenate(
                [np.asarray(anchor["vr_3point_offsets_tcp"], dtype=np.float64), np.zeros((2, 3))], axis=0
            )
            self._home_sep = float(np.linalg.norm(self._home_pos_5pt[0] - self._home_pos_5pt[1]))
            self._home_foot_sep = float(np.linalg.norm(self._home_pos_5pt[3] - self._home_pos_5pt[4]))

        # --- WHICH PUCK DECIDES "FORWARD" (--mode 5pt-direct only) ----------------
        # The 2pt/3pt/5pt/dense frame is solved from the L-R HAND vector and stays so: those
        # are reviewed paths, the operator's hands ARE the instrument there, and a rig that
        # works must not change because a fifth mode was added. 5pt-direct is the mode where
        # that assumption breaks -- the operator is STANDING and WALKING, the hands may be
        # parked or holding a box, and the two channels being steered (chest, feet) are worn
        # rather than held. So it defaults to the CHEST heading and cross-checks. See
        # `DIRECT_5PT_YAW_SOURCES`.
        _yf = getattr(args, "world_yaw_from", None)
        _trim = float(getattr(args, "yaw_trim_deg", 0.0) or 0.0)
        if not self._is_5pt_direct and (_yf is not None or _trim != 0.0):
            print(
                "[bridge] ERROR: --world-yaw-from / --yaw-trim-deg are --mode 5pt-direct only. "
                f"This is --mode {self._mode}, whose frame comes from the L-R hand vector "
                "(--yaw-deg / --no-yaw-autocal) and is deliberately left byte-identical: "
                "3pt-unified-anneal teleop is a reviewed, working path and a frame change there "
                "is not a side effect this flag gets to have.",
                flush=True,
            )
            sys.exit(2)
        if getattr(args, "no_yaw_autocal", False) and _yf not in (None, "fixed"):
            print(
                f"[bridge] ERROR: --no-yaw-autocal says 'solve nothing' and --world-yaw-from "
                f"{_yf} says 'solve it from the {_yf}'. Pick one: drop --no-yaw-autocal, or use "
                "--world-yaw-from fixed --yaw-deg <deg>.",
                flush=True,
            )
            sys.exit(2)
        self._yaw_from = (
            "fixed" if getattr(args, "no_yaw_autocal", False) else (_yf or "chest")
        ) if self._is_5pt_direct else "hands"
        self._yaw_trim = math.radians(_trim)
        self._yaw_tol = math.radians(abs(float(getattr(args, "world_yaw_tol_deg", 40.0))))
        #: The witnesses as of the last calibration, and which one was USED. Reported on the
        #: status/heartbeat line because the frame is the one thing an operator cannot see:
        #: they see the robot move, and a turned frame reads as the policy misbehaving.
        self._yaw_witnesses: dict[str, float | None] = {}
        self._yaw_used = self._yaw_from
        self._yaw_refuse_logged_at = 0.0

        # --- calibration state -------------------------------------------------
        self._world_yaw = math.radians(args.yaw_deg)  # libsurvive Z-up -> pelvis
        # per-hand position anchor: yaw-aligned puck-origin pos at calibration.
        self._calib_pos_yaw = {r: np.zeros(3) for r in self._roles}
        # per-hand yaw-aligned puck quat at calibration (and its inverse). Used
        # for the relative-orientation delta AND the rotated-lever offset FK.
        self._calib_q_yaw = {r: np.array([1.0, 0.0, 0.0, 0.0]) for r in self._roles}
        self._ref_q_yaw_inv = {r: np.array([1.0, 0.0, 0.0, 0.0]) for r in self._roles}

        # --- live tunables -----------------------------------------------------
        self._reach_scale = float(args.reach_scale)
        self._z_offset = float(args.z_offset)
        self._orientation_on = args.orientation == "on"
        self._stale_timeout = float(args.stale_timeout)
        self._stale_logged = False  # one-shot stale/fresh transition logging

        # Pending toggles set by signal handlers / file flags and applied inside
        # snapshot() UNDER the lock. The signal handlers MUST NOT touch the lock:
        # signals run on the main thread (where the asyncio loop also runs), and
        # a handler firing while snapshot() holds the non-reentrant lock would
        # self-deadlock. So handlers only set these booleans.
        self._pending_recal = False
        self._pending_orient_toggle = False
        # Explicit orientation set (None = no pending set) from the WS control
        # channel; distinct from the toggle so the browser can command an
        # absolute state. Applied under the lock in _drain_pending_locked.
        self._pending_orient_set: bool | None = None

        self._seq = 0
        if args.no_align:
            self._calibrated = True  # debug: skip auto-cal (NOT anchored)

        signal.signal(signal.SIGUSR1, self._on_sigusr1)
        signal.signal(signal.SIGUSR2, self._on_sigusr2)

    # --- signal / flag handlers ----------------------------------------------
    # IMPORTANT: these run on the main thread between bytecodes and MUST be
    # async-signal-safe wrt our lock — they only set flags (no lock, no print of
    # shared state), which snapshot() drains under the lock. Acquiring the
    # non-reentrant lock here would deadlock if a signal lands mid-snapshot().
    def _on_sigusr1(self, *_):
        self._pending_recal = True

    def _on_sigusr2(self, *_):
        self._pending_orient_toggle = True

    # Public control entry points (called from the WS handler / event-loop
    # thread). Like the signal handlers, they only set flags — the actual state
    # change happens under the lock in _drain_pending_locked, so they are safe to
    # call from any thread without touching shared calibration state directly.
    def request_recalibrate(self) -> None:
        self._pending_recal = True

    def set_orientation(self, on: bool) -> None:
        self._pending_orient_set = on

    def _drain_pending_locked(self):
        """Apply pending signal/flag toggles. MUST be called holding self._lock."""
        if self._pending_recal:
            self._pending_recal = False
            self._calibrated = False
            self._both_seen_at = None
            print("[bridge] recal requested -> re-calibrating on next dual frame", flush=True)
        if self._pending_orient_toggle:
            self._pending_orient_toggle = False
            self._orientation_on = not self._orientation_on
            print(f"[bridge] orientation -> {'ON' if self._orientation_on else 'OFF (translation-first)'}", flush=True)
        if self._pending_orient_set is not None:
            self._orientation_on = self._pending_orient_set
            self._pending_orient_set = None
            print(
                f"[bridge] orientation set -> {'ON' if self._orientation_on else 'OFF (translation-first)'}", flush=True
            )

    # --- the world frame (--mode 5pt-direct) -----------------------------------
    def _heading_witnesses_locked(self) -> dict[str, float | None]:
        """The world-yaw estimates the five pucks support, as ``{name: yaw_rad or None}``.

        All four answer the SAME question (what ``Rz`` puts the operator's forward on robot
        +x) in incomparable ways, which is the point -- see the WORLD-YAW WITNESSES banner.

          ``chest``     relabelled CHEST heading -> +x.
          ``feet_dir``  circular mean of the two relabelled FOOT headings -> +x. Depends on the
                        ankle brackets matching the chest's DECLARED mount, so it is evidence
                        about the brackets, not a default.
          ``feet_sep``  LEFT-minus-RIGHT foot vector -> +y. Geometric; flips 180 deg if the two
                        ankle pucks are swapped.
          ``hands``     LEFT-minus-RIGHT hand vector -> +y. What every other mode uses.

        ``None`` means "this witness has nothing to say" (puck not locked, or a lateral pair
        closer than `LATERAL_MIN_SEP_M`) -- never a fabricated 0.0. Caller holds the lock.
        """
        raw = self._raw_hand
        out: dict[str, float | None] = {"chest": None, "feet_dir": None, "feet_sep": None, "hands": None}
        if "head" in raw:
            out["chest"] = world_yaw_from_heading(body_heading_from_puck(raw["head"][1]))
        if "left_foot" in raw and "right_foot" in raw:
            out["feet_dir"] = world_yaw_from_heading(
                circ_mean([body_heading_from_puck(raw["left_foot"][1]), body_heading_from_puck(raw["right_foot"][1])])
            )
            out["feet_sep"] = world_yaw_from_lateral(raw["left_foot"][0] - raw["right_foot"][0])
        if "left_hand" in raw and "right_hand" in raw:
            out["hands"] = world_yaw_from_lateral(raw["left_hand"][0] - raw["right_hand"][0])
        return out

    #: `--world-yaw-from` -> which `_heading_witnesses_locked` key it selects.
    _YAW_SOURCE_KEY = {"chest": "chest", "feet": "feet_sep", "hands": "hands"}

    def _resolve_world_yaw_5pt_locked(self) -> bool:
        """Set `self._world_yaw` for 5pt-direct, or return False to refuse calibration.

        Prints EVERY witness with its disagreement from the chosen one, on every calibration,
        because the frame is the one quantity the operator cannot perceive: a turned frame
        leaves the start pose exact, every gate green and the wire well-formed, and surfaces
        only as the robot moving the wrong way (see `BODY_RELABEL_M`). Two refusals:

          * the CHOSEN witness has nothing to say (its puck is not locked / a lateral pair is
            degenerate). Better to wait than to fall back silently to another puck.
          * ``chest`` and ``feet_sep`` disagree by more than ``--world-yaw-tol-deg``. Those two
            are the body-WORN pair -- one a measured heading, one pure geometry -- so a
            disagreement is a real mount fault, and the 180 deg case is specifically the two
            ankle pucks being on the wrong ankles, which is otherwise SILENT (the wire is
            valid, the gate closes, and the robot simply crosses its legs).

        The HANDS are informational here, never a refusal: parking the hand pucks or holding
        them stacked is normal in a walking test, and it is exactly the case where the
        historical autocal reads a frame that is turned.
        """
        w = self._heading_witnesses_locked()
        self._yaw_witnesses = dict(w)
        src = self._yaw_from
        if src == "fixed":
            base: float | None = wrap_angle(math.radians(float(self._args.yaw_deg)))
        else:
            base = w.get(self._YAW_SOURCE_KEY[src])
        now = time.time()

        def _refuse(msg: str) -> bool:
            # Throttled: _maybe_calibrate runs at the publish rate, and the whole point of the
            # message is that the operator can read it.
            if now - self._yaw_refuse_logged_at > 5.0:
                self._yaw_refuse_logged_at = now
                print(f"[bridge] NOT CALIBRATING -- {msg}", flush=True)
                print(f"[bridge]     witnesses: {self._fmt_witnesses(w)}", flush=True)
            return False

        if base is None:
            return _refuse(
                f"--world-yaw-from {src} has no reading yet "
                + (
                    "(chest puck not locked)"
                    if src == "chest"
                    else f"(the two feet are closer than {LATERAL_MIN_SEP_M * 100:.0f} cm horizontally, "
                    "or one is not locked -- STAND with your feet apart)"
                    if src == "feet"
                    else f"(the two hand pucks are closer than {LATERAL_MIN_SEP_M * 100:.0f} cm "
                    "horizontally, or one is not locked)"
                )
            )
        chest, feet = w["chest"], w["feet_sep"]
        if chest is not None and feet is not None and abs(wrap_angle(chest - feet)) > self._yaw_tol:
            return _refuse(
                f"the CHEST and the FEET disagree about which way you are facing by "
                f"{ang_diff_deg(chest, feet):+.1f} deg (tol {math.degrees(self._yaw_tol):.0f}). "
                "Near 180 deg = the two ankle pucks are on the WRONG ANKLES (swap "
                "LEFT_FOOT_SERIAL / RIGHT_FOOT_SERIAL); near 90 = a bracket is mounted turned. "
                "Stand square with your feet apart, or raise --world-yaw-tol-deg if you know "
                "which witness to believe."
            )

        self._world_yaw = wrap_angle(base + self._yaw_trim)
        self._yaw_used = src
        self._yaw_refuse_logged_at = 0.0
        print(
            f"[bridge] world yaw = {math.degrees(self._world_yaw):+.1f} deg from {src.upper()}"
            + (f" (incl. --yaw-trim-deg {math.degrees(self._yaw_trim):+.1f})" if self._yaw_trim else "")
            + f" -- witnesses: {self._fmt_witnesses(w)}",
            flush=True,
        )
        hands = w["hands"]
        if src != "hands" and hands is not None and abs(wrap_angle(hands - self._world_yaw)) > self._yaw_tol:
            # Not an error: it is the OLD default reporting that it would have picked a
            # different frame. Say what it means, since "the hands disagree" sounds like a fault.
            print(
                f"[bridge]     NOTE: the L-R HAND vector would have given "
                f"{math.degrees(hands):+.1f} deg ({ang_diff_deg(hands, self._world_yaw):+.1f} deg "
                "away). That is what 2pt/3pt/5pt/dense use, so if you are used to those: your "
                "hand pucks are not a lateral pair right now (parked, stacked front-to-back, or "
                "held close together) and this mode is deliberately not reading them.",
                flush=True,
            )
        fdir = w["feet_dir"]
        if fdir is not None and chest is not None and abs(wrap_angle(fdir - chest)) > self._yaw_tol:
            print(
                f"[bridge]     NOTE: the FOOT pucks' own heading is {ang_diff_deg(fdir, chest):+.1f} deg "
                "from the chest's, so the ankle brackets do NOT hold their pucks like the chest "
                "one. Nothing on the WIRE depends on that (a constant mount rotation cancels in "
                "every commanded delta), but collection's foot relabel does -- see "
                "collection_viz._BODY_RELABEL_M.",
                flush=True,
            )
        return True

    @staticmethod
    def _fmt_witnesses(w: dict[str, float | None]) -> str:
        """`{name: deg}` for the log, with `--` for a witness that has nothing to say."""
        return ", ".join(
            f"{k}=" + ("--" if v is None else f"{math.degrees(v):+.1f}") for k, v in sorted(w.items())
        )

    # --- calibration ----------------------------------------------------------
    def _maybe_calibrate(self):
        """While the operator holds the calibration pose (hands in front, clearly
        separated left/right at ~the robot's default-pose hand height):
          (1) DERIVE the world yaw from the raw L-R hand vector so the operator's
              lateral axis maps onto robot +Y (fixes the Z-up vs pelvis frame);
          (2) capture each hand's yaw-aligned position so live displacement maps
              onto the default-pose home target (robot holds STILL at calib);
          (3) capture each hand's yaw-aligned orientation as the relative-anchor
              reference, so the puck mounting rotation cancels.
        Fires `calib_delay` s after both hands are first seen."""
        if self._calibrated:
            return
        # Require ALL tracked roles (both hands, plus the head in 3pt mode) before
        # calibrating — the per-role loop below reads _raw_hand[role] for every role.
        if any(r not in self._raw_hand for r in self._roles):
            self._both_seen_at = None
            return
        # BASE-STATION GATE. Calibration derives the world yaw from ONE instantaneous L-R
        # hand vector, so it bakes whatever frame libsurvive has at that moment into every
        # later command. On too few solved stations that frame is under-constrained, and the
        # error is a constant yaw offset on both arms — indistinguishable from "the operator
        # was standing crooked", which is exactly how it gets misdiagnosed. Refuse until
        # enough are up, and restart the hold countdown so the operator does not lose the
        # pose while waiting.
        # _locked variant: we are already inside snapshot()'s `with self._lock:`, and
        # self._lock is NOT reentrant (see _stations_gate's docstring).
        ok, why = self._stations_gate_locked()
        if not ok:
            self._both_seen_at = None
            if not self._stations_wait_logged:
                self._stations_wait_logged = True
                print(f"[bridge] WAITING on base stations: {why}", flush=True)
            return
        if self._stations_wait_logged:
            self._stations_wait_logged = False
            print(f"[bridge] base stations ready ({why}) -- resuming calibration", flush=True)
        if self._both_seen_at is None:
            self._both_seen_at = time.time()
            if self._is_5pt_direct:
                _what = "all 5 trackers (hands + chest + both feet)"
                # The FRAME comes from the CHEST here (`--world-yaw-from`), not from the hands,
                # so the pose that matters is STANDING SQUARE -- and the hands only have to be
                # where you want the robot's hands to start. Telling the operator to hold the
                # old hand-pair pose would be telling them to satisfy a witness this mode does
                # not read.
                _pose = (
                    "STAND square, feet apart and flat, chest level and facing the direction you "
                    "want to be the robot's FORWARD; hands wherever you want them to start"
                )
            elif self._uses_head:
                _what = "all 3 trackers (hands + head)"
                _pose = "hands in front, separated L/R; head level"
            else:
                _what = "both hands"
                _pose = "hands in front, separated L/R"
            print(
                f"[bridge] {_what} seen -- HOLD the calibration pose ({_pose}); "
                f"calibrating in {self._args.calib_delay:.1f}s",
                flush=True,
            )
            return
        if time.time() - self._both_seen_at < self._args.calib_delay:
            return

        Lraw = self._raw_hand["left_hand"][0]
        Rraw = self._raw_hand["right_hand"][0]
        # (1) world yaw from the raw libsurvive horizontal L-R vector:
        # Rz(theta) maps [vx,vy] onto +Y  =>  theta = atan2(vx, vy).
        v = Lraw - Rraw
        if self._is_5pt_direct:
            # Five pucks means three INDEPENDENT witnesses of the operator's facing, so this
            # mode resolves + cross-checks instead of trusting the hands. It can REFUSE, which
            # is why it is a separate method and why we return before touching any anchor: a
            # frame we do not believe must leave the bridge un-calibrated (nothing on the wire,
            # the robot holds home) rather than publish a turned one.
            if not self._resolve_world_yaw_5pt_locked():
                return
        elif not self._args.no_yaw_autocal and (v[0] ** 2 + v[1] ** 2) > 1e-4:
            self._world_yaw = float(math.atan2(v[0], v[1]))
            self._yaw_used = "hands"
            print(
                f"[bridge] world yaw auto-cal: {math.degrees(self._world_yaw):+.1f} deg "
                f"(raw L-R horiz=[{v[0]:+.3f},{v[1]:+.3f}])",
                flush=True,
            )
        else:
            # No autocal, or the hands too close together to mean a direction: --yaw-deg stands.
            # Labelled `fixed` on the heartbeat so the line never claims a witness it did not read.
            self._yaw_used = "fixed"
            print(f"[bridge] world yaw fixed at {math.degrees(self._world_yaw):+.1f} deg", flush=True)

        # (2)+(3) per-hand position + orientation reference, yaw-aligned. Store
        # the yaw-aligned calib quat too (for the rotated-lever offset FK below).
        for role in self._roles:
            raw_p, raw_q = self._raw_hand[role]
            self._calib_pos_yaw[role] = rot_z_vec(self._world_yaw, raw_p)
            q_yaw = quat_mul(quat_z(self._world_yaw), quat_normalize(raw_q))
            self._calib_q_yaw[role] = q_yaw
            self._ref_q_yaw_inv[role] = quat_inv(q_yaw)

        op_sep = float(np.linalg.norm(rot_z_vec(self._world_yaw, Lraw) - rot_z_vec(self._world_yaw, Rraw)))
        # DIRECT 5pt: report the FOOT separation the same way the hands' is reported, and say
        # out loud that per-role anchoring has just put every commanded point ON the low level's
        # pre-arm reference. Both numbers are the operator's, not the robot's: per-role anchoring
        # SUBSTITUTES the robot's geometry for the operator's (each point tracks its own puck's
        # displacement from its own anchor), so a wide stance does not widen the robot's -- it
        # only means the operator's own feet and the robot's move 1:1 from different starts.
        _extra = ""
        if self._is_5pt_direct:
            _lf = rot_z_vec(self._world_yaw, self._raw_hand["left_foot"][0])
            _rf = rot_z_vec(self._world_yaw, self._raw_hand["right_foot"][0])
            _extra = (
                f", op_foot_sep={float(np.linalg.norm(_lf - _rf)):.3f} m, "
                f"robot_foot_sep={self._home_foot_sep:.3f} m"
            )
        self._calibrated = True
        # The MC episode's home anchor IS its frame 0, so a new calibration must start a
        # new episode -- otherwise the completion keeps re-basing onto the operator pose
        # from the LAST calibration and jumps by however far they have since drifted. Set
        # a flag rather than calling reset() here: this runs under the bridge lock and
        # reset() does a backend round trip.
        self._mc_needs_reset = self._is_mc
        print(
            f"[bridge] CALIBRATED: per-{'point' if self._is_5pt_direct else 'hand'} anchor set "
            f"(op_sep={op_sep:.3f} m, robot_sep={self._home_sep:.3f} m{_extra}, "
            f"hand_point={self._hand_point}, reach_scale={self._reach_scale:.2f}, "
            f"orientation={'ON' if self._orientation_on else 'OFF'})",
            flush=True,
        )
        if self._is_5pt_direct:
            print(
                "[bridge]     every commanded point now sits EXACTLY on the low level's pre-arm "
                "reference (the READY pose's own FK), so an ABSOLUTE-mode `start` arms by "
                "construction -- including the two FEET, which the motion-completion path could "
                "not close (69-91 mm) and which holosoma therefore does not gate.",
                flush=True,
            )
            if abs(self._z_offset) > 0.0:
                # The claim above is only true at z_offset 0, and the difference is exactly the
                # trim: say so rather than printing a promise the wire does not keep.
                print(
                    f"[bridge]     ...EXCEPT for --z-offset {self._z_offset:+.3f} m, which lifts "
                    "the hands and chest off that reference by exactly that much (the two FEET "
                    "are excluded from it). Expect the pre-arm gate to see that gap.",
                    flush=True,
                )

    def _hand_command(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (vr_pos[3], vr_rot6d[6], wrist_q_wxyz[4]) for hand k (0=L,1=R),
        pelvis frame. The third return is the wrist quaternion the rot6d was built
        from — used ONLY by the 3pt snapshot (which needs the quaternion rep); the
        2pt snapshot ignores it, so its payload stays byte-identical.

        Orientation: world-frame relative delta composed onto the default-pose
        home wrist quat, then -> rot6d. In translation-first mode (orientation
        OFF) the orn is the constant home rot6d so the wrists stay at the default
        pose while position tracks.

        Position: the policy's vr_2point point is the CONTACT POINT
        ``wrist_origin + R(wrist)@offset``, a rigid lever on the wrist. We mirror
        the displacement of THAT point from the calibration pose onto the
        default-pose home target. Tracking the lever (not just the puck origin)
        keeps pos consistent with orn when the operator twists the hand in place
        (the contact point sweeps an arc); without it a pure twist would move the
        wrist orientation while pos stayed put — an OOD (pos,orn) pair. At the
        calibration pose the lever cancels exactly (cur==calib), so hold-still is
        still exact. With orientation OFF the wrist is held at the home quat, so
        the live lever uses that constant orientation (still arc-consistent)."""
        role = self._roles[k]
        raw_p, raw_q = self._raw_hand[role]
        offset = self._vr_offsets[k]

        # Yaw-aligned current puck pose.
        p_yaw = rot_z_vec(self._world_yaw, raw_p)
        cur_q_yaw = quat_mul(quat_z(self._world_yaw), quat_normalize(raw_q))

        # --- orientation: world-frame delta since calib, onto the home wrist ---
        if self._orientation_on:
            delta_world = quat_mul(cur_q_yaw, self._ref_q_yaw_inv[role])
            wrist_q = quat_normalize(quat_mul(delta_world, self._home_wrist_q[k]))
            orn6d = quat_wxyz_to_rot6d(wrist_q)
        else:
            # Translation-first: wrist held at the default-pose home orientation.
            wrist_q = self._home_wrist_q[k]
            orn6d = self._home_orn6d[k].copy()

        # --- position of the CONTACT POINT (wrist_origin + R(wrist)@offset) ---
        # Model: the ROBOT wrist origin tracks the puck-origin displacement; the
        # contact point is wrist_origin + R(wrist_q)@offset where the lever is in
        # the ROBOT wrist frame (wrist_q). home_pos already bakes in
        # R(home_q)@offset, so subtract that to avoid double-counting:
        #   pos = home_pos + s*(p_yaw - calib_pos) + (R(wrist_q) - R(home_q))@offset
        # At calibration p_yaw==calib_pos AND wrist_q==home_q (delta=identity), so
        # both correction terms vanish -> pos=home (hold-still EXACT). In
        # orientation-OFF mode wrist_q==home_q always, so the lever term is 0 and
        # this reduces to the plain puck-displacement mapping. Under a live twist
        # (ON) the lever-arc R(wrist_q)@offset moves with the wrist so pos stays
        # consistent with orn.
        puck_disp = p_yaw - self._calib_pos_yaw[role]
        lever = quat_rotate(wrist_q, offset) - quat_rotate(self._home_wrist_q[k], offset)
        pos = self._home_pos[k] + self._reach_scale * puck_disp + lever
        pos[2] += self._z_offset
        return pos, orn6d, np.asarray(wrist_q, dtype=np.float64)

    def _head_command(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (vr_pos[3], vr_quat_xyzw[4]) for the HEAD point (3pt only), pelvis
        frame — the SAME mapping as ``_hand_command`` but with the 3pt head anchor
        (index 2 of the 3pt home arrays) and a QUATERNION orn output (the 3pt
        ``vr_3point_orn`` rep) instead of rot6d.

        Position uses the torso→head lever offset (0,0,0.35) exactly like the hands'
        contact-point lever, so a head twist keeps pos consistent with orn and
        hold-still is exact (lever cancels at calibration). Orientation: world-frame
        relative delta since calib composed onto the home head quat, then → xyzw. In
        translation-first mode the head orn is held at the home quat.
        """
        raw_p, raw_q = self._raw_hand["head"]
        offset = self._vr_offsets_3pt[2]  # (0,0,0.35) torso→head lever
        home_q = self._home_body_q_3pt[2]  # wxyz home head quat (identity at default)
        p_yaw = rot_z_vec(self._world_yaw, raw_p)

        if self._orientation_on:
            cur_q_yaw = quat_mul(quat_z(self._world_yaw), quat_normalize(raw_q))
            delta_world = quat_mul(cur_q_yaw, self._ref_q_yaw_inv["head"])
            head_q = quat_normalize(quat_mul(delta_world, home_q))
        else:
            head_q = home_q
        orn_xyzw = quat_wxyz_to_xyzw(head_q)

        puck_disp = p_yaw - self._calib_pos_yaw["head"]
        lever = quat_rotate(head_q, offset) - quat_rotate(home_q, offset)
        pos = self._home_pos_3pt[2] + self._reach_scale * puck_disp + lever
        pos[2] += self._z_offset
        return pos, orn_xyzw

    def _point_command_5pt(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(vr_pos[3], vr_quat_xyzw[4]) for keypoint ``k`` of the DIRECT five-tracker wire.

        The SAME mapping ``_head_command`` implements -- per-role position anchor + a
        world-frame relative orientation delta onto that role's home + the rotated-lever
        correction -- generalised from one derived point to all five MEASURED ones, against the
        5pt (READY-pose) home arrays. Nothing new is invented here; what is new is that every
        slot has a real puck behind it.

        Three properties worth stating, because each one is a question this mode invites:

        * **Hold-still is exact, per point.** At the calibration pose ``p_yaw == calib`` and
          ``delta == identity``, so both correction terms vanish and the output IS the home --
          which for this family is the low level's own pre-arm gate reference.
        * **A constant puck MOUNT rotation cancels, so no relabel is owed IN THE COMMAND.** The
          orientation is ``(cur . ref^-1) . home``; a body-local mount ``R_m`` appears in both
          ``cur`` and ``ref`` and drops out, and the puck ORIGIN is unmoved by it. Pinned by
          ``test_a_constant_mount_rotation_cancels_so_no_relabel_is_owed``.
          ⚠️ That is a statement about this function and NOT about the mode: ``self._world_yaw``
          is solved from an ABSOLUTE puck HEADING (`_resolve_world_yaw_5pt_locked`), where the
          mount cannot cancel and `BODY_RELABEL_M` is mandatory -- the same reason the
          completion path needs ``mc_frames.chest_body_from_puck``. Every term below is
          mount-invariant; the ``Rz`` they are all expressed in is not.
        * **The FEET carry no lever**, so a foot puck TWIST moves the commanded foot's
          orientation and NOT its position -- correct, because the tracked point is the
          ``ankle_roll_link`` ORIGIN. The hands' 0.242 m TCP lever and the chest's 0.35 m lever
          do sweep an arc, exactly as in 2pt/3pt.
        """
        role = self._roles[k]
        raw_p, raw_q = self._raw_hand[role]
        offset = self._vr_offsets_5pt[k]
        home_q = self._home_body_q_5pt[k]  # wxyz
        p_yaw = rot_z_vec(self._world_yaw, raw_p)

        if self._orientation_on:
            cur_q_yaw = quat_mul(quat_z(self._world_yaw), quat_normalize(raw_q))
            delta_world = quat_mul(cur_q_yaw, self._ref_q_yaw_inv[role])
            body_q = quat_normalize(quat_mul(delta_world, home_q))
        else:
            # Translation-first: this role holds the home orientation while position tracks.
            body_q = home_q

        puck_disp = p_yaw - self._calib_pos_yaw[role]
        lever = quat_rotate(body_q, offset) - quat_rotate(home_q, offset)
        pos = self._home_pos_5pt[k] + self._reach_scale * puck_disp + lever
        if role not in DIRECT_5PT_GROUND_ROLES:
            pos[2] += self._z_offset
        # Canonicalize the double cover to w >= 0 for the same reason every other quaternion
        # producer in this stack does: the 5pt policy trained ONLY on the +w branch (its
        # normalizer puts w at mean +0.73 hands / +0.98 head / +0.96 feet), so `-q` is a ~20
        # sigma input rather than a harmless alias. A no-op on data that is already +w.
        return pos, _canon_xyzw(quat_wxyz_to_xyzw(body_q))

    def snapshot(self) -> dict | None:
        """Build the {pos[6], orn[12], seq, ...} payload, or None when not ready
        (both hands not seen / not calibrated / a tracker is STALE). Returning
        None makes the browser hold the HOME anchor rather than a frozen pose."""
        # File-flag toggles set the same pending flags the signals do (applied
        # under the lock in _drain_pending_locked — never call handlers that take
        # the lock from here).
        if os.path.exists(RECAL_FLAG):
            try:
                os.remove(RECAL_FLAG)
            except OSError:
                pass
            self._pending_recal = True
        if os.path.exists(ORIENT_FLAG):
            try:
                os.remove(ORIENT_FLAG)
            except OSError:
                pass
            self._pending_orient_toggle = True

        now = time.time()
        with self._lock:
            self._drain_pending_locked()
            self._maybe_calibrate()
            # Not-ready guard: every tracked role must have been seen (both hands,
            # plus head in 3pt) — else hold HOME (None). _head_command below reads
            # _raw_hand["head"], so a missing head must gate the whole frame.
            if any(r not in self._raw_hand for r in self._roles):
                return None
            if not self._calibrated:
                return None
            # Staleness: if a tracker stopped updating (occlusion / dropout), do
            # NOT keep streaming its last pose — return None so the browser falls
            # back to HOME instead of holding a dead arm command.
            if self._stale_timeout > 0:
                for role in self._roles:
                    age = now - self._raw_hand_t.get(role, 0.0)
                    if age > self._stale_timeout:
                        if not self._stale_logged:
                            print(
                                f"[bridge] tracker {role} STALE ({age * 1000:.0f} ms > "
                                f"{self._stale_timeout * 1000:.0f} ms) -- holding HOME",
                                flush=True,
                            )
                            self._stale_logged = True
                        return None
            if self._stale_logged:
                print("[bridge] trackers fresh again -- resuming live command", flush=True)
                self._stale_logged = False
            mc_raw = None
            if self._is_mc:
                # MOTION-COMPLETION modes: the wire is NOT built from the per-hand home
                # anchor. The three RAW puck poses (plus the calibrated world yaw) go into
                # the completion, which re-bases them onto the G1's own 29-joints-zero
                # pose and returns a whole body; the 5 keypoints / 29 joint targets are
                # read out of ITS FK, in ITS pelvis frame. Nothing here needs
                # _hand_command / _head_command, and home_anchor.json has no 5pt arrays.
                #
                # Only COPY OUT here. The completion work happens after the lock is
                # released: it is numpy plus a queue append, and holding the bridge lock
                # across it would make the pysurvive reader wait on it every frame.
                mc_raw = (
                    float(self._world_yaw),
                    (self._raw_hand["left_hand"][0].copy(), self._raw_hand["left_hand"][1].copy()),
                    (self._raw_hand["right_hand"][0].copy(), self._raw_hand["right_hand"][1].copy()),
                    (self._raw_hand["head"][0].copy(), self._raw_hand["head"][1].copy()),
                )
            elif self._is_5pt_direct:
                # FIVE measured points. Computed under the lock like every other mode's command
                # (it reads `_raw_hand` / the calibration state), and cheap enough to be: five
                # quaternion products, no inference and no queue.
                direct_pts = [self._point_command_5pt(k) for k in range(len(self._roles))]
            else:
                Lp, Lo, Lq_wxyz = self._hand_command(0)
                Rp, Ro, Rq_wxyz = self._hand_command(1)
                if self._is_3pt:
                    Hp, Ho = self._head_command()  # pos[3], quat_xyzw[4]
            orientation_on = self._orientation_on
        if mc_raw is not None:
            return self._mc_snapshot(now, mc_raw, orientation_on)
        self._seq += 1
        if self._is_5pt_direct:
            # BYTE-COMPATIBLE with `--mode 5pt`'s payload minus the two `mc_*` provenance keys
            # (there is no completion frame and no completion lag to report, and reporting a
            # zero would claim a stage that does not exist). `ViveLiveSource` validates the
            # (15, 20) width PAIR against this `mode` tag and passes the vectors straight into
            # the obs, so the low level, the browser preset and the pose-match viewer all take
            # this frame unchanged -- the viewer reads `mc_lag_s`/`mc_qpos` with `.get` and
            # already says so when they are absent.
            return {
                "seq": self._seq,
                "t": now,
                "mode": "5pt",
                "pos": [v for p, _ in direct_pts for v in p.tolist()],  # 15
                "orn": [v for _, q in direct_pts for v in q.tolist()],  # 20 (5x quat xyzw)
                "orientation": orientation_on,
            }
        if self._is_3pt:
            # 3pt payload: pos[9] = [L,R,head]xyz; orn[12] = [L,R,head]xyzw
            # QUATERNION (NOT rot6d — the 3pt policy's vr_3point_orn rep). The hand
            # wrist quats (already computed inside _hand_command) are reused directly,
            # so there is no lossy rot6d→quat re-extraction.
            # Canonicalize the double cover to w >= 0 on all three, for the same reason the
            # bake and the rig FK do (see bake_motions._fk_into): the quaternion-orn presets
            # trained only on the +w branch, and `q` vs `-q` is one rotation but two very
            # different obs vectors. Defensive here rather than corrective — a calibrated
            # operator standing near the robot's ready pose is already +w, so this is a
            # measured no-op on the validated 3pt teleop path; it closes the branch for a
            # pose that wanders near the 180 deg seam.
            Lq = _canon_xyzw(quat_wxyz_to_xyzw(Lq_wxyz))
            Rq = _canon_xyzw(quat_wxyz_to_xyzw(Rq_wxyz))
            Ho = _canon_xyzw(np.asarray(Ho, dtype=np.float64))
            return {
                "seq": self._seq,
                "t": now,
                "mode": "3pt",
                "pos": [*Lp.tolist(), *Rp.tolist(), *Hp.tolist()],  # 9
                "orn": [*Lq.tolist(), *Rq.tolist(), *Ho.tolist()],  # 12 (3x quat xyzw)
                "orientation": orientation_on,
            }
        return {
            "seq": self._seq,
            "t": now,
            "mode": "2pt",
            "pos": [*Lp.tolist(), *Rp.tolist()],  # 6
            "orn": [*Lo.tolist(), *Ro.tolist()],  # 12 (2x rot6d)
            "orientation": orientation_on,
        }

    # --- motion-completion modes (--mode 5pt / dense) -------------------------
    # -- RAW tracker feed, in UMI_collection's recorder format --------------------------
    #
    # WHY THIS LIVES HERE. libsurvive's USB dongle is a single-claim singleton and this
    # process holds it, so nothing else on the machine can read the pucks while teleop is
    # running. Collection's own `bridges/tracker/vive_tracker_bridge.py` therefore cannot be
    # started alongside a teleop session — and its output, `tracker{,_right,_head}.npy`, is
    # the ONE input `processing/convert_to_umi.py` has no substitute for. Publishing the raw
    # poses from here is what makes a teleop session convertible by the SAME converter that
    # made every existing dataset, instead of needing a second one.
    #
    # It is a SEPARATE PORT from :8770 on purpose, not an extra key on the teleop frame.
    # :8770 carries the pelvis-frame COMMAND (calibration-relative, hand-point-levered,
    # world-yaw-autocalibrated); this carries the un-processed puck pose in the aligned world
    # frame. They are different quantities with different consumers, and a recorder that had
    # to pick a key out of a command frame would be one refactor away from recording the
    # command as if it were a measurement.
    def umi_tracker_snapshot(self, now: float | None = None) -> dict:
        """The role-keyed RAW tracker frame UMI's recorder ingests, or an empty-trackers dict.

        Byte-compatible with ``vive_tracker_bridge``'s ``{"mode","t","trackers":{role:{...}}}``:
        same key names, same wxyz quaternion convention (converted to xyzw ONLY in the
        converter), same per-role ``seq``/``stale``, same 90 Hz default broadcast. The role
        names are UMI's (``left``/``right``/``head``), not this module's internal
        ``left_hand``/``right_hand``.

        Published UNCONDITIONALLY once the flag is on — before calibration, before the MC
        backend, and whether or not the policy is armed. The recorder's take boundaries are
        the operator's Start/Stop, not this bridge's state machine, and a tracker stream that
        went quiet whenever teleop was not yet armed would silently truncate the head of every
        take. Staleness is reported per role instead (the recorder stores the flag).
        """
        now = time.time() if now is None else now
        trackers: dict[str, dict] = {}
        with self._lock:
            items = [
                (role, self._raw_hand.get(role), self._raw_hand_t.get(role, 0.0),
                 self._raw_hand_seq.get(role, 0))
                for role in self._roles
            ]
            n_lh = len(self._stations_live)
            lh_names = sorted(self._stations_live)
        for role, pq, t_upd, seq in items:
            if pq is None:
                continue
            raw_p, raw_q = pq
            p = quat_rotate(self._q_umi_world, np.asarray(raw_p, dtype=float))
            q = quat_normalize(quat_mul(self._q_umi_world, quat_normalize(np.asarray(raw_q, dtype=float))))
            trackers[UMI_TRACKER_ROLE[role]] = {
                "seq": int(seq),
                # The pose's OWN update time, not the broadcast time: the converter resamples
                # onto the camera grid and a broadcast stamp would quantise every puck to the
                # publisher's tick and hide a stalled one behind a fresh timestamp.
                "t": float(t_upd),
                "pos": [float(p[0]), float(p[1]), float(p[2])],
                "quat": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
                "stale": bool(self._umi_stale_timeout > 0 and (now - t_upd) > self._umi_stale_timeout),
            }
        return {
            # The tracking family this stream CAME FROM: 5pt when all five pucks are being
            # read (`--mode 5pt-direct`), else 3pt whenever a head puck is configured -- which
            # every MC mode is, since the chest is the completion's only root channel. The
            # recorder gates its own per-role files on ITS --mode, so this is provenance, not a
            # switch; but a 5-puck session labelled `3pt` would be a file that lies about which
            # pucks were on the radio.
            "mode": "5pt" if self._is_5pt_direct else ("3pt" if self._uses_head else "2pt"),
            "t": float(now),
            "trackers": trackers,
            "lighthouses": {"count": n_lh, "names": lh_names, "expected": 0},
        }

    def start_mc(self) -> bool:
        """Build + start the MotionCompletionSource. Only for --mode 5pt / dense.

        Imported lazily and only in these modes, so the 2pt/3pt paths keep their exact
        dependency set: this reaches into the UMI_collection checkout for `mc_frames` /
        `mc_backend_client` and spawns the g1_mosyn interpreter, none of which a 2pt run
        should be able to fail on.
        """
        if not self._is_mc:
            return True
        from wbt_realtime_eval.vive.mc_teleop_source import MotionCompletionSource

        self._mc = MotionCompletionSource(
            keypoint_mode=self._mode,
            hand_point=self._hand_point,
            lag_frames=int(getattr(self._args, "mc_lag_frames", 20)),
            wrist_ik=not getattr(self._args, "mc_no_wrist_ik", False),
            foot_fix="inertialize" if getattr(self._args, "mc_foot_ik", False) else "none",
            # None => the source's own precedence (MC_REF_FRAME env, else "frozen").
            ref_frame=getattr(self._args, "mc_ref_frame", None),
            # Same None => env => default precedence, for the same reason: WHERE the take is
            # placed is a per-session decision (a collection run must match the placement its
            # labels are prepared in), not a property of this call site.
            cond_mode=getattr(self._args, "mc_cond_mode", None),
            dz_anchor=getattr(self._args, "mc_dz_anchor", "hand"),
            dz_wrist=getattr(self._args, "mc_dz_wrist", None),
            # Ditto: whether the puck->wrist leg is applied is a per-session decision (a
            # recorded session wants it; a plain teleop run keeps the shipped behaviour), so
            # None hands it to the source's argument > MC_HAND_LEVER > default precedence.
            hand_lever=getattr(self._args, "mc_hand_lever", None),
            # `_mc_snapshot` hands over the RAW libsurvive chest puck, so the source owes it
            # the chest body relabel. Stated explicitly even though it is the default: the
            # offline drivers pass "body", and which one a call site needs is decided by
            # where its third pose came from, not by a preference.
            head_frame="puck",
            umi_dir=getattr(self._args, "mc_umi_dir", None),
            log=lambda m: print(m, flush=True),
        )
        if getattr(self._args, "mc_foot_ik", False):
            print(
                "[bridge] ⚠️  --mc-foot-ik is an OFFLINE setting: the causal leg IK costs "
                "1937 ms/window p50 = 291% duty, which takes the command lag from a constant 700 ms "
                "to 6100 ms p50 / 8867 ms max (measured). There is no "
                "real-time-capable value (below ~80 Adam steps the IK never reaches its "
                "targets and buys nothing). Use it only to inspect foot slide.",
                flush=True,
            )
        return self._mc.start()

    def stop_mc(self) -> None:
        if self._mc is not None:
            self._mc.stop()
            self._mc = None

    def _mc_snapshot(self, now: float, mc_raw, orientation_on: bool) -> dict | None:
        """Feed the completion and read back the commanded 5pt / dense payload."""
        if self._mc is None:
            return None
        world_yaw, left, right, head = mc_raw
        if self._mc_needs_reset:
            self._mc_needs_reset = False
            self._mc.reset()
        self._mc.push(world_yaw, left, right, head, now=now)
        got = self._mc.latest(now=now)
        if got is None:
            # Still filling the first window (30 frames = 1.0 s) -> hold HOME, exactly as
            # a stale tracker does. Never a partial or extrapolated command.
            return None
        self._seq += 1
        # ONE definition of the wire format, shared with the trackerless replay driver
        # (`vive/mc_replay_bridge.py`) — a sim fed a different payload shape than the rig
        # produces would prove nothing about the rig. `--publish-mc-qpos` is VIEWER-ONLY
        # (vive/mc_teleop_gate_viz.py) and additive; see `wire_payload`.
        from wbt_realtime_eval.vive.mc_teleop_source import wire_payload

        return wire_payload(
            self._mode,
            got,
            seq=self._seq,
            t=now,
            orientation=orientation_on,
            publish_qpos=bool(getattr(self._args, "publish_mc_qpos", False)),
        )

    # --- base stations (2 or 3, adaptively) -----------------------------------
    def _preflight_stations(self) -> None:
        """Report the CACHED lighthouse geometry before libsurvive starts, and guard the
        one way `--fresh` can quietly destroy it.

        `--fresh` maps to libsurvive's ``--force-calibrate --force-ootx``, which clears
        ``PositionSet``/``OOTXSet`` on **every** station in ``[0, activeLighthouses)`` — and
        ``activeLighthouses`` is initialised from this config, not from what is powered on.
        So running `--fresh` with a station switched off wipes that station's pose and never
        re-solves it, leaving a config that looks calibrated for the ones you had on and is
        silently unusable for the one you did not. That failure has already happened here
        once (the `config.json.broken_both_unsolved_*` backup next to the live file), so:
        back the file up first and say plainly what is about to be cleared.
        """
        report_cached_stations()
        if self._args.fresh:
            backup_before_fresh()

    def _note_configured_stations(self, ctx) -> None:
        """Record the lighthouse objects libsurvive created from its config."""
        names: list[str] = []
        for obj in ctx.Objects():
            try:
                nm = str(obj.Name(), "utf-8")
            except Exception:  # noqa: BLE001 — a malformed name must not abort startup
                continue
            if nm.startswith("LH"):
                names.append(nm)
        with self._lock:
            self._stations_configured = sorted(names)
        print(f"[bridge] base stations configured in libsurvive: {sorted(names) or 'none yet'}", flush=True)

    def _note_live_station(self, name: str) -> None:
        """First sighting of a SOLVED station on the radio: log it, and once enough are up,
        say so once. `_stations_live` is what the calibration gate reads."""
        with self._lock:
            if name in self._stations_live:
                return
            self._stations_live.add(name)
            n = len(self._stations_live)
            live = sorted(self._stations_live)
            enough = n >= self._min_stations
            announce = enough and not self._stations_ok_logged
            if announce:
                self._stations_ok_logged = True
        print(f"[bridge] base station {name} LIVE (solved) -- {n} up: {live}", flush=True)
        if announce:
            want = self._args.expect_stations
            extra = ""
            if want and n != want:
                extra = f" (you asked for STATIONS={want}; running on {n})"
            print(
                f"[bridge] base stations OK: {n} >= {self._min_stations} required{extra}"
                + ("  -- 2 stations means thinner coverage: expect more dropouts when you "
                   "turn your back on one." if n == 2 else ""),
                flush=True,
            )

    def _stations_gate(self) -> tuple[bool, str]:
        """(ok, reason), taking the lock. For callers that do NOT already hold it.

        ⚠️ `_maybe_calibrate` runs INSIDE `snapshot()`'s `with self._lock:`, so it must call
        `_stations_gate_locked` instead. Calling this one from there deadlocks the asyncio
        event loop on its first broadcast tick — `self._lock` is a plain `threading.Lock`,
        which is not reentrant. That is not hypothetical: it shipped in eed1b3e and hung the
        rig (2026-08-28). The bridge then published nothing, never calibrated, and stopped
        accepting WebSocket connections (21 queued on :8770), which surfaced at the policy as
        "no live frame" and then as "your hand pose is too far" — three symptoms, none of them
        naming a lock. The `_locked` suffix is this file's existing convention for exactly
        this (`_drain_pending_locked`); follow it rather than reaching for an RLock, which
        would let the same layering mistake pass silently next time.
        """
        with self._lock:
            return self._stations_gate_locked()

    def _stations_gate_locked(self) -> tuple[bool, str]:
        """(ok, reason) for "are there enough solved base stations to calibrate against".
        CALLER MUST HOLD ``self._lock``.

        Adaptive by construction: it counts what is actually LIVE rather than trusting the
        config or a flag, so 2 and 3 stations both just work and a station that dies between
        sessions is caught here instead of showing up as bad tracking. ``--expect-stations``
        turns the count into an assertion for an operator who knows what should be up.
        """
        n = len(self._stations_live)
        live = sorted(self._stations_live)
        if n < self._min_stations:
            return False, (
                f"only {n} solved base station(s) live {live}, need >= {self._min_stations}. "
                "Power them on / clear line of sight; if one was moved, restart with --fresh."
            )
        want = self._args.expect_stations
        if want and n < want:
            return False, (
                f"STATIONS={want} was requested but only {n} are live {live}. "
                "One is off, blocked, or unsolved (--fresh after moving it)."
            )
        return True, f"{n} station(s) live {live}"

    # --- pysurvive read loop (its own thread; singleton) ----------------------
    def run_pysurvive(self, stop: threading.Event):
        survive_args = sys.argv[:1] + ["--lighthouse-update-velocity", "0.5"]
        if self._args.fresh:
            # TWO independent resets, historically bundled. Splitting them is what makes a
            # station whose OOTX will not decode recoverable — see FRESH_OOTX in
            # 1_vive_bridge.sh, and UMI_collection's vive_tracker_bridge.py, which had this
            # split first and whose reasoning this mirrors deliberately.
            #   --force-calibrate 1  clears the base-station POSES -> re-solve the room.
            #                        This is what you want after moving or adding a station.
            #   --force-ootx 1       throws away each station's decoded FACTORY blob (id +
            #                        lens/motor constants) and re-reads it off the sync bits.
            #                        SLOW (one bit per sweep, ~25-35 s/station) and it needs
            #                        clean uninterrupted line of sight; a station the pucks
            #                        cannot see cleanly logs "Bad sync bit" and retries
            #                        forever while NOTHING produces a pose, so the launch
            #                        looks hung. The blob is burned into the station hardware
            #                        and does NOT change when you move it, so keeping it is
            #                        safe for a move.
            # Default stays ON (re-read) because it is the only thing that re-reads a
            # station's ID: with a cached blob libsurvive never attaches a decoder, so a
            # station SWAPPED onto a channel another one used would be silently used with the
            # wrong calibration.
            survive_args += ["--force-calibrate", "1"]
            if getattr(self._args, "fresh_ootx", True):
                survive_args += ["--force-ootx", "1"]
            else:
                print(
                    "[bridge] --no-fresh-ootx: keeping the CACHED factory blobs; only the "
                    "base-station POSITIONS are re-solved.\n"
                    "[bridge]     This skips the slow OOTX handshake -- use it when a fresh "
                    "launch hangs on 'Bad sync bit'.\n"
                    "[bridge]     ⚠️  A station SWAPPED onto an existing channel would NOT be "
                    "noticed this launch.",
                    flush=True,
                )
        print(f"[bridge] starting pysurvive {survive_args[1:]}", flush=True)
        print(f"[bridge] serial->role: {self._serial_to_role}", flush=True)
        self._preflight_stations()
        ctx = pysurvive.SimpleContext(survive_args)
        # `SimpleContext.__init__` snapshots the object list ONCE, before it starts the
        # tracking thread, and `create_lighthouse` runs during init off the cached config —
        # so the LH objects here are the CONFIGURED stations, not necessarily the powered-on
        # ones. Which are actually live is decided by what shows up in NextUpdated() below.
        self._note_configured_stations(ctx)
        name_to_serial: dict[str, str] = {}
        last_hb = 0.0
        while ctx.Running() and not stop.is_set():
            updated = ctx.NextUpdated()
            if not updated:
                continue
            name = str(updated.Name(), "utf-8")
            if name.startswith("LH"):
                # A lighthouse only reaches this stream once its pose is SOLVED
                # (`create_lighthouse` seeds has_update from PositionSet, and thereafter it
                # fires on pose updates), so "seen here" == "solved and contributing". This
                # used to be a bare `continue`, which is why a base station that was off,
                # unsolved, or newly added was completely invisible: the operator saw normal
                # hand tracking and no hint that it was running on one station. Record it,
                # then carry on ignoring it for pose purposes.
                self._note_live_station(name)
                continue
            if name not in name_to_serial:
                try:
                    s = pysurvive.survive_simple_serial_number(updated.ptr)
                    name_to_serial[name] = s.decode() if isinstance(s, bytes) else str(s)
                except Exception:
                    name_to_serial[name] = "?"
                ser = name_to_serial[name]
                role = self._serial_to_role.get(ser)
                print(f"[bridge] tracker {name} serial={ser} -> {role or 'UNMAPPED (ignored)'}", flush=True)
            ser = name_to_serial[name]
            role = self._serial_to_role.get(ser)
            # Accept every role in the active mapping (2 hands, + head in 3pt); any
            # other tracker is ignored. self._serial_to_role only contains "head"
            # when --mode 3pt, so 2pt behavior is unchanged.
            if role not in self._serial_to_role.values():
                continue
            pose, _ts = updated.Pose()
            pos = np.array([pose.Pos[0], pose.Pos[1], pose.Pos[2]], dtype=float)
            wxyz = np.array([pose.Rot[0], pose.Rot[1], pose.Rot[2], pose.Rot[3]], dtype=float)
            if not np.any(pos) and abs(wxyz[0]) >= 1.0 - 1e-9:
                continue  # zero/identity sentinel before a real lock
            now = time.time()
            with self._lock:
                self._raw_hand[role] = (pos, wxyz)
                self._raw_hand_t[role] = now  # for staleness detection
                self._raw_hand_seq[role] = self._raw_hand_seq.get(role, 0) + 1
            if now - last_hb > 1.0:
                with self._lock:
                    seen = sorted(self._raw_hand.keys())
                    cal = self._calibrated
                    n_lh = len(self._stations_live)
                # Station count belongs in the heartbeat, not only at startup: a base
                # station that browns out mid-session is otherwise invisible, and the
                # symptom (slightly worse tracking) reads as the policy's fault.
                #
                # "streaming" was a lie before calibration. `snapshot()` returns None while
                # `not _calibrated`, so the WS carries NOTHING — and the consumer then reports
                # "no live frame", which on the policy side used to surface as "your hand pose
                # is too far" (2026-08-28). Say which of the two states this is, and what the
                # operator has to do about it, on every heartbeat.
                _missing = [r for r in self._roles if r not in seen]
                if cal:
                    _lbl = "pucks" if self._is_5pt_direct else "hands"
                    # The FRAME belongs on the heartbeat for the same reason the station count
                    # does: it is invisible to the operator, it is the one thing that turns
                    # every commanded displacement at once, and its symptom ("the robot goes
                    # sideways") reads as the policy's fault. See BODY_RELABEL_M.
                    _yaw = f" yaw={math.degrees(self._world_yaw):+.1f}deg/{self._yaw_used}"
                    print(f"[bridge] streaming {_lbl}={seen} stations={n_lh}{_yaw} calibrated=True", flush=True)
                elif _missing:
                    print(
                        f"[bridge] NOT PUBLISHING (nothing on the wire) -- waiting for puck(s): "
                        f"{_missing}. seen={seen} stations={n_lh}",
                        flush=True,
                    )
                else:
                    _pose_hint = (
                        "STAND square with your feet apart and your chest facing forward"
                        if self._is_5pt_direct
                        else "hands in front, separated L/R; head level facing +x"
                    )
                    print(
                        f"[bridge] NOT PUBLISHING (nothing on the wire) -- all {len(seen)} puck(s) "
                        f"locked, stations={n_lh}, but NOT CALIBRATED yet: HOLD THE CALIBRATION POSE "
                        f"({_pose_hint}) and keep still ~3 s.",
                        flush=True,
                    )
                last_hb = now
        print("[bridge] pysurvive context stopped.", flush=True)


# -----------------------------------------------------------------------------
# WebSocket server
# -----------------------------------------------------------------------------
async def _serve_umi_tracker(bridge: ViveTeleopBridge, host: str, port: int,
                             rate_hz: float, stop: threading.Event):
    """Broadcast the RAW puck poses in UMI_collection's tracker-bridge format.

    A second, WRITE-ONLY server on its own port, so `recording/recorder.py --tracker-url`
    can be pointed at a live teleop session and write the same `tracker*.npy` files
    `processing/convert_to_umi.py` already knows how to read. Nothing sent by a client is
    read: the recorder never talks back on this socket, and accepting a control message here
    would put a second way to recalibrate next to a stream whose whole job is to be the
    unedited measurement.
    """
    clients: set = set()

    async def handler(ws):
        clients.add(ws)
        peer = getattr(ws, "remote_address", "?")
        print(f"[bridge] UMI-tracker client connected: {peer} (total {len(clients)})", flush=True)
        try:
            async for _ in ws:
                pass  # drain and ignore: write-only stream (see the docstring)
        except Exception:
            pass
        finally:
            clients.discard(ws)
            print(f"[bridge] UMI-tracker client gone: {peer} (total {len(clients)})", flush=True)

    async with websockets.serve(handler, host, port):
        # R_world is EVIDENCE, printed the way collection prints it: it is baked into every
        # recorded pose and is unrecoverable from the file afterwards, so "which world frame
        # is this session in" has to be answerable from the launch log.
        print(
            f"[bridge] RAW tracker (UMI recorder format) serving at ws://{host}:{port} "
            f"(broadcast {rate_hz:.0f} Hz) R_world(wxyz)="
            f"{np.round(bridge._q_umi_world, 4).tolist()}",
            flush=True,
        )
        period = 1.0 / rate_hz
        while not stop.is_set():
            if clients:
                websockets.broadcast(clients, json.dumps(bridge.umi_tracker_snapshot()))
            await asyncio.sleep(period)


async def _serve_all(bridge: ViveTeleopBridge, args, stop: threading.Event):
    """The teleop wire, plus the raw-tracker wire when `--umi-tracker-port` is given.

    One event loop for both: the snapshot builders are cheap, pure-Python reads under the
    same lock the pysurvive thread already takes, and a second loop would only add a second
    place for the GIL to be held while the first one is mid-broadcast.
    """
    tasks = [_serve(bridge, args.host, args.port, args.rate_hz, stop)]
    if int(getattr(args, "umi_tracker_port", 0) or 0) > 0:
        tasks.append(_serve_umi_tracker(bridge, args.host, int(args.umi_tracker_port),
                                        float(args.umi_tracker_rate_hz), stop))
    await asyncio.gather(*tasks)


async def _serve(bridge: ViveTeleopBridge, host: str, port: int, rate_hz: float, stop: threading.Event):
    clients: set = set()

    async def handler(ws):
        clients.add(ws)
        peer = getattr(ws, "remote_address", "?")
        print(f"[bridge] WS client connected: {peer} (total {len(clients)})", flush=True)
        try:
            # The connection is normally write-only (we broadcast frames), but
            # the browser's Recalibrate button sends a control message back:
            #   {"cmd":"recalibrate"} -> queue a recal (applied under the lock in
            #   snapshot()/_drain_pending_locked, same path as SIGUSR1).
            #   {"cmd":"orientation","on":bool} -> set orientation mode.
            async for raw in ws:
                try:
                    m = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                cmd = m.get("cmd") if isinstance(m, dict) else None
                if cmd == "recalibrate":
                    bridge.request_recalibrate()
                    print(f"[bridge] WS recalibrate request from {peer}", flush=True)
                elif cmd == "orientation":
                    bridge.set_orientation(bool(m.get("on", True)))
                    print(f"[bridge] WS orientation={'ON' if m.get('on', True) else 'OFF'} from {peer}", flush=True)
        except Exception:
            pass  # connection closed / errored — fall through to cleanup
        finally:
            clients.discard(ws)
            print(f"[bridge] WS client gone: {peer} (total {len(clients)})", flush=True)

    async with websockets.serve(handler, host, port):
        print(f"[bridge] WebSocket serving at ws://{host}:{port} (broadcast {rate_hz:.0f} Hz)", flush=True)
        period = 1.0 / rate_hz
        last_logged_state = None
        while not stop.is_set():
            snap = bridge.snapshot()
            if snap is not None and clients:
                # Non-blocking fan-out: websockets.broadcast queues per-connection
                # without awaiting, so a slow/backpressured client cannot
                # head-of-line-block the loop or the other clients.
                websockets.broadcast(clients, json.dumps(snap))
            # one-line state transition log (not per-frame spam)
            state = None if snap is None else (snap["orientation"], len(clients))
            if state != last_logged_state and snap is not None:
                print(
                    f"[bridge] streaming to {len(clients)} client(s); "
                    f"orientation={'ON' if snap['orientation'] else 'OFF (trans-first)'}",
                    flush=True,
                )
                last_logged_state = state
            await asyncio.sleep(period)


def _scan_radio(ctx, radio: dict) -> None:
    """Record every puck the libsurvive driver has added, whether or not it is being
    TRACKED, into ``radio`` (serial -> device name). Mutates in place; cheap enough to
    call in the scan loop.

    Two traps make this less obvious than ``ctx.Objects()``:

    * ``SimpleContext.objects`` is a snapshot taken in ``__init__``, BEFORE
      ``simple_start_thread``. Devices appear later (the ``Adding tracked object WMn``
      lines), so that list is empty for exactly the pucks we care about. Re-walk the
      live list with ``simple_get_{first,next}_object`` instead.
    * ``survive_simple_serial_number`` returns an EMPTY string for a device that was
      just added — the serial arrives asynchronously. So keep re-reading until it
      populates rather than caching the first (empty) answer.

    Without this, a puck that is powered and paired but cannot see a lighthouse is
    indistinguishable from one that is switched off, and the operator is told to check
    power when the real fix is to aim it at a base station.
    """
    try:
        curr = pysurvive.simple_get_first_object(ctx.ptr)
        while curr:
            obj = pysurvive.SimpleObject(curr)
            name = str(obj.Name(), "utf-8")
            if not name.startswith("LH"):  # skip base stations
                try:
                    s = pysurvive.survive_simple_serial_number(obj.ptr)
                    ser = (s.decode() if isinstance(s, bytes) else str(s)).strip()
                except Exception:
                    ser = ""
                if ser and ser not in radio:
                    radio[ser] = name
            curr = pysurvive.simple_get_next_object(ctx.ptr, curr)
    except Exception:
        # Discovery is a diagnostic; a binding change here must never crash the scan.
        # The caller degrades to the old "tracked vs not seen" reporting.
        pass


def _list_serials(args) -> int:
    """Discovery mode: scan pysurvive for connected trackers, print each one's
    LHR serial (and its current hand mapping if serials are configured), then
    return. No calibration, no WebSocket, no bridge — just answers "which
    serials are connected right now?" after you swap a tracker.

    Reuses the exact serial-extraction path as ``run_pysurvive`` (skip ``LH*``
    base stations; ``survive_simple_serial_number`` per device) so what it
    reports is what the live bridge would see. Honors ``--fresh`` (re-solve a
    moved lighthouse) and stops at two trackers or ``--list-timeout`` s."""
    left = args.left_serial or os.environ.get("VIVE_UMI_LEFT_SERIAL", "")
    right = args.right_serial or os.environ.get("VIVE_UMI_RIGHT_SERIAL", "")
    head = getattr(args, "head_serial", None) or os.environ.get("VIVE_UMI_HEAD_SERIAL", "")
    lfoot = getattr(args, "left_foot_serial", None) or os.environ.get("VIVE_UMI_LEFT_FOOT_SERIAL", "")
    rfoot = getattr(args, "right_foot_serial", None) or os.environ.get("VIVE_UMI_RIGHT_FOOT_SERIAL", "")
    # Scan for up to 3 trackers ONLY in 3pt mode, and up to 5 in `5pt-direct`. Keep the head
    # serial in the role map (so a discovered head is LABELLED even in a 2pt session) but do NOT
    # bump the early-stop count off the env alone — the wrapper always exports
    # VIVE_UMI_HEAD_SERIAL, so a plain 2pt `--list-serials` must still stop at 2
    # (byte-identical to before the 3pt work). The FEET follow the same rule for the same
    # reason: the launcher exports their serials whenever it knows them, and a 3pt scan that
    # waited for five pucks would run its whole timeout on a rig that has three.
    _m = getattr(args, "mode", "2pt")
    want = 5 if _m == "5pt-direct" else (3 if _m == "3pt" else 2)
    _all = ((left, "left"), (right, "right"), (head, "head"), (lfoot, "left_foot"), (rfoot, "right_foot"))
    _allowed = {"left", "right"} | ({"head"} if want >= 3 else set()) | (
        {"left_foot", "right_foot"} if want >= 5 else set()
    )
    serial_to_role = {s: r for s, r in _all if s and r in _allowed}

    # --fresh is REFUSED here, not merely discouraged. See `main`, which rejects the
    # combination before we ever get called; this stays as a belt-and-braces assert so a
    # future caller of _list_serials cannot reintroduce it.
    if args.fresh:
        print(
            "[bridge] --list-serials --fresh is refused (see main). Discovery reads the cached "
            "geometry; it cannot re-solve it.",
            flush=True,
        )
        return 2
    survive_args = sys.argv[:1] + ["--lighthouse-update-velocity", "0.5"]
    print(
        f"[bridge] --list-serials: scanning for up to {want} trackers (timeout "
        f"{args.list_timeout:.0f}s, cached calibration -- read-only)...",
        flush=True,
    )
    # The cached station geometry, before pysurvive can touch it. Discovery is where an
    # operator looks after ANY hardware change, so it should answer "how many base stations
    # do I have, and are they solved?" as well as "which pucks are attached" — those two
    # questions get asked together and were previously only half-answered here.
    cached = report_cached_stations()
    ctx = pysurvive.SimpleContext(survive_args)
    name_to_serial: dict[str, str] = {}
    stations_live: set[str] = set()  # LH names that delivered a SOLVED pose during the scan
    found: dict[str, str] = {}  # serial -> device name, in discovery order
    # Pucks that are ON THE RADIO (dongle paired + powered) but have produced NO pose
    # yet. This is a DIFFERENT failure from "not on the radio", and the operator fix is
    # different too: aim it at a base station / wave it, vs. power it on or re-pair it.
    # Distinguishing them needs the device LIST, not the pose stream — see _radio_pucks.
    radio: dict[str, str] = {}  # serial -> device name, every puck the driver has added
    deadline = time.time() + args.list_timeout
    while ctx.Running() and time.time() < deadline and len(found) < want:
        _scan_radio(ctx, radio)
        updated = ctx.NextUpdated()
        if not updated:
            continue
        name = str(updated.Name(), "utf-8")
        if name.startswith("LH"):
            stations_live.add(name)  # solved + contributing (see run_pysurvive)
            continue  # base station, not a hand tracker
        if name in name_to_serial:
            continue
        try:
            s = pysurvive.survive_simple_serial_number(updated.ptr)
            ser = s.decode() if isinstance(s, bytes) else str(s)
        except Exception:
            ser = "?"
        name_to_serial[name] = ser
        if ser not in found:
            found[ser] = name
            role = serial_to_role.get(ser)
            if role == "head":
                tag = " -> currently maps to HEAD"
            elif role:
                tag = f" -> currently maps to {role.upper()} hand"
            else:
                tag = " -> UNMAPPED"
            print(f"[bridge]   tracker {name}: serial={ser}{tag}", flush=True)
    _scan_radio(ctx, radio)
    # Pucks on the radio that never produced a pose. Reported separately below so
    # "powered but blind" never reads as "powered off".
    #
    # Only call them NOT TRACKED when the scan actually ran out the clock. The loop
    # early-stops the moment it has `want` trackers, so a 2pt scan of a 3-puck rig
    # returns with the chest puck simply never having been waited for — labelling that
    # "blind" would be the same wrong diagnosis this reporting exists to prevent.
    early_stop = len(found) >= want
    # Station verdict. Note the early-stop caveat applies here too: the loop can exit as
    # soon as it has `want` trackers, so a fast lock may simply not have waited long enough
    # for every station to report. Say which case it is rather than implying a fault.
    _ns = len(stations_live)
    if _ns >= MIN_STATIONS:
        print(f"[bridge] base stations LIVE during the scan: {_ns} {sorted(stations_live)}", flush=True)
    elif early_stop:
        print(
            f"[bridge] base stations LIVE during the scan: {_ns} {sorted(stations_live)} — but the scan "
            f"stopped early (it had its {want} tracker(s)), so this is a FLOOR, not a fault. "
            "Re-run with --list-timeout 20 to let every station report.",
            flush=True,
        )
    else:
        print(
            f"[bridge] ⚠ only {_ns} base station(s) reported in the whole scan window "
            f"{sorted(stations_live)}; the live bridge needs >= {MIN_STATIONS} and will WAIT.",
            flush=True,
        )
        # WHICH advice depends on the CACHED state, and the two fixes are opposite. Telling an
        # operator with an unsolved config to "check line of sight" sends them hunting a
        # hardware fault that isn't there; telling one with good cached geometry to --fresh
        # destroys the geometry that was fine. So branch on what the config actually says.
        _unsolved = [lh for lh in cached if not lh["solved"]]
        if not cached or len(_unsolved) == len(cached):
            print(
                "[bridge]     The CACHED geometry is unsolved/absent, so this is NOT a line-of-sight "
                "problem -- there is no\n"
                "[bridge]     station pose to track against yet. Re-running discovery cannot fix it; "
                "the solve needs the LIVE bridge.",
                flush=True,
            )
            print(SOLVE_PROCEDURE, flush=True)
        else:
            print(
                "[bridge]     The cached geometry HAS solved stations, so the pucks should be able to "
                "see them: check power\n"
                "[bridge]     and line of sight first. Only --fresh (on the LIVE bridge, every station "
                "powered on) if one\n"
                "[bridge]     was physically MOVED -- it clears every station's pose, including the "
                "ones that are fine.",
                flush=True,
            )
    no_pose = {s: n for s, n in radio.items() if s not in found}
    untracked = {} if early_stop else no_pose
    for ser, name in sorted(no_pose.items()):
        role = serial_to_role.get(ser)
        why = ("not scanned (the scan already had the "
               f"{want} tracker(s) this mode needs and stopped)" if early_stop
               else "⚠ ON THE RADIO but NOT TRACKED (no pose in the whole scan window)")
        print(
            f"[bridge]   tracker {name}: serial={ser} "
            f"{('-> ' + role.upper()) if role else '-> UNMAPPED'} {why}",
            flush=True,
        )

    if not found:
        print("[bridge] --list-serials: NO tracked trackers.", flush=True)
        if untracked:
            # The single most useful line in this function: the pucks ARE powered and
            # paired, so it is NOT a power/pairing problem. But which of the two remaining
            # causes it is depends on the config: with no solved station pose, "aim it at a
            # lighthouse" is a wild goose chase — the pucks may be seeing the stations
            # perfectly and still have nothing to solve against.
            solved_cached = [lh for lh in cached if lh["solved"]]
            print(
                f"[bridge]   {len(untracked)} puck(s) ARE on the radio (powered + paired) but "
                "produced NO pose -- do NOT re-pair, the pairing is fine.",
                flush=True,
            )
            if solved_cached:
                print(
                    "[bridge]   With solved station geometry cached, this is an aim/occlusion "
                    "problem: point the sensor face at a lighthouse and WAVE.",
                    flush=True,
                )
            else:
                print(
                    "[bridge]   And with NO solved station geometry, waving here changes nothing "
                    "-- re-solve on the live bridge first (see above).",
                    flush=True,
                )
            return 1
        print(
            "[bridge]   1. Another bridge is ALREADY running and holds the USB dongle "
            "(pysurvive is a single-claim singleton). If you saw 'LIBUSB_ERROR_BUSY' "
            "above, stop the other bridge first (Ctrl-C it, or kill its PID), then re-run.",
            flush=True,
        )
        print(
            "[bridge]   2. Trackers powered off / not in view of a lighthouse — power them "
            "on and wave them; try a longer --list-timeout.",
            flush=True,
        )
        return 1
    serials = list(found)
    print(f"[bridge] --list-serials: {len(serials)} tracker(s) found: {serials}", flush=True)

    # Report against the CONFIGURED roles first. `found` is in DISCOVERY order, which
    # varies run to run, so suggesting "--left-serial {serials[0]}" on a rig whose
    # serials are already pinned prints a different (and wrong) assignment every other
    # run — and getting L/R backwards is a ~180° yaw flip, the exact failure this flag
    # exists to prevent. Only fall back to a positional suggestion when nothing is
    # mapped yet, i.e. when there is genuinely no assignment to report.
    missing = [(r, s) for s, r in serial_to_role.items() if s not in found]
    mapped = [(r, s) for s, r in serial_to_role.items() if s in found]
    if mapped:
        print("[bridge] CONFIGURED roles seen on the radio:", flush=True)
        for role, ser in sorted(mapped):
            print(f"[bridge]   {role.upper():5s} = {ser}  ✓", flush=True)
        for role, ser in sorted(missing):
            if ser in untracked:
                # Powered + paired, just blind. Naming the actual fix matters: this is
                # the state a puck sits in when it is face-down on the bench.
                print(
                    f"[bridge]   {role.upper():5s} = {ser}  ⚠ ON THE RADIO but NOT TRACKED — "
                    "powered + paired, but it cannot see a base station. Aim its sensor face "
                    "at a lighthouse and WAVE it; do NOT re-pair.",
                    flush=True,
                )
            else:
                print(
                    f"[bridge]   {role.upper():5s} = {ser}  ✗ NOT ON THE RADIO — powered off, "
                    f"dongle unplugged/unpaired, or a different puck than configured",
                    flush=True,
                )
        unmapped = [s for s in serials if s not in serial_to_role]
        for ser in unmapped:
            print(f"[bridge]   (unmapped tracker on the radio: {ser})", flush=True)
        if not missing and not unmapped:
            print("[bridge] all configured trackers present — nothing to change.", flush=True)
        return 0 if not missing else 1

    print("[bridge] No serials configured. To use these (left hand on YOUR left):", flush=True)
    if len(serials) >= 2:
        print(
            f"[bridge]   wbt-vive-teleop-bridge --left-serial {serials[0]} "
            f"--right-serial {serials[1]}   (swap the two if L/R ends up mirrored)",
            flush=True,
        )
        if len(serials) >= 3:
            print(
                f"[bridge]   3pt: add --mode 3pt --head-serial {serials[2]}   "
                "(the tracker that is NOT a hand — mount it on the head)",
                flush=True,
            )
        if len(serials) >= 5:
            print(
                f"[bridge]   5pt-direct: add --mode 5pt-direct --left-foot-serial {serials[3]} "
                f"--right-foot-serial {serials[4]}   (WHICH foot is which is decided by "
                "STRAPPING, not by discovery order — swap them if the stance comes out mirrored)",
                flush=True,
            )
    else:
        print("[bridge]   (only one tracker seen — power on the other and re-run --list-serials)", flush=True)
    return 0


def main():
    anchor_default = load_home_anchor()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=WS_HOST, help="WS bind host (localhost only)")
    ap.add_argument("--port", type=int, default=WS_PORT, help="WS bind port")
    ap.add_argument("--rate-hz", type=float, default=90.0, help="WS broadcast rate")
    ap.add_argument("--left-serial", default=None, help="LHR serial of the LEFT tracker")
    ap.add_argument("--right-serial", default=None, help="LHR serial of the RIGHT tracker")
    ap.add_argument(
        "--head-serial",
        default=None,
        help="LHR serial of the HEAD tracker (REQUIRED for --mode 3pt; else "
        "VIVE_UMI_HEAD_SERIAL). Ignored in 2pt mode.",
    )
    ap.add_argument(
        "--left-foot-serial",
        default=None,
        help="LHR serial of the LEFT-FOOT tracker (REQUIRED for --mode 5pt-direct; else "
        "VIVE_UMI_LEFT_FOOT_SERIAL). Ignored in every other mode. There is no default and no "
        "first-seen latch: a guessed foot serial does not fail, it commands a MIRRORED stance "
        "with every gate green.",
    )
    ap.add_argument(
        "--right-foot-serial",
        default=None,
        help="LHR serial of the RIGHT-FOOT tracker (REQUIRED for --mode 5pt-direct; else "
        "VIVE_UMI_RIGHT_FOOT_SERIAL). Ignored in every other mode.",
    )
    ap.add_argument(
        "--mode",
        choices=["2pt", "3pt", "5pt", "dense", "5pt-direct"],
        default="2pt",
        help="tracking points: 2pt = two hand trackers (default; byte-identical "
        "to before) → pos[6]/orn[12] rot6d. 3pt = + a HEAD tracker (torso_link) → "
        "pos[9]/orn[12] as 3× xyzw quaternion, for exp:g1-wbt-3pt-hybrid. "
        "5pt / dense = the SAME three trackers put through Motion_Completion first, "
        "for exp:g1-wbt-5pt-hybrid-unified-anneal (pos[15]/orn[20], + both feet) and "
        "exp:g1-wbt-hybrid-unified-anneal (joint_pos[29]/joint_vel[29]). Both carry a "
        "667 ms command lag floor that belongs to the MC checkpoint — see "
        "vive/mc_teleop_source.py. 5pt-direct = the SAME pos[15]/orn[20] wire as 5pt but read "
        "from FIVE REAL TRACKERS (both hands, the chest, and one puck per ANKLE), with NO "
        "motion completion: no backend, no ~10 s load, no 667 ms lag, and both feet MEASURED "
        "instead of inferred. Requires --hand-point tcp and both foot serials.",
    )
    # --- motion-completion knobs (--mode 5pt / dense only) ---------------------
    ap.add_argument(
        "--mc-lag-frames",
        type=int,
        default=20,
        help="how far behind the operator the command runs, in 30 Hz frames (default 20 "
        "= 667 ms). This is NOT free tuning: the MC checkpoint conditions densely over a "
        "30-frame window with a 20-frame stride, so anything below 19 makes the command "
        "STALL between windows instead of running late. Raise it (e.g. 30) for extra "
        "margin on a loaded host.",
    )
    ap.add_argument(
        "--mc-no-wrist-ik",
        action="store_true",
        help="skip the per-window wrist IK (~20 ms). It is what pulls the commanded hand "
        "to 0.3 mm, so this is a diagnostic, not a speed-up worth taking.",
    )
    ap.add_argument(
        "--mc-foot-ik",
        action="store_true",
        help="enable the causal foot IK. OFFLINE ONLY: 1937 ms/window p50 = 291%% duty, so "
        "the command falls behind. There is no real-time-capable setting (below ~80 Adam "
        "steps the leg IK never reaches its targets), which is why the default is off.",
    )
    ap.add_argument(
        "--mc-ref-frame",
        default=None,
        choices=["frozen", "live"],
        help="--mode 5pt only: which pelvis frame the keypoints are read out in. `frozen` "
        "(default) is the FIXED reference pelvis captured at calibration — the same frame the "
        "home wrist was placed in, so frame 0's hands land on the pre-arm gate's own reference "
        "to 0.1 mm and the operator's body motion reaches the robot. `live` is the "
        "completion's own moving root, which is what shipped before 2026-08-29 and CANCELS "
        "that body motion (measured: a 300 mm walk moves the commanded points 7.3 mm) — kept "
        "for an A/B only. Also settable per process as MC_REF_FRAME. Inert for --mode dense: "
        "joint angles carry no pelvis. Full argument in mc_teleop_source's "
        "_readout_frame_locked.",
    )
    ap.add_argument(
        "--mc-cond-mode",
        default=None,
        choices=["home", "absolute"],
        help="BOTH MC modes: WHERE the operator's take is placed in the robot's world — the "
        "same decision, with the same two names, that mc_episode/mc_to_hl_wholebody make "
        "offline. `home` (default) re-bases each wrist onto the G1's 29-joints-ZERO pose and "
        "transfers only the operator's CHANGE, substituting the robot's 0.297 m hand "
        "separation for the operator's. `absolute` keeps the operator's own geometry and "
        "places the take with ONE rigid z lift (--mc-dz-anchor), and REQUIRES "
        "--mc-hand-lever tcp (it is the mode in which the puck->wrist relabel does not "
        "cancel; un-levered it commands every hand orientation 180 deg wrong). A RECORDED "
        "session must be driven in the placement its labels are prepared in — driving in one "
        "and labelling in the other puts every image and the joint target beside it in two "
        "different postures, and nothing downstream compares them; for teleop that is `home` "
        "on both sides (run_mc_to_hl_wholebody.sh's own default, and no dz to copy). Note "
        "`home` is also the ONLY placement whose frame 0 lands on the pre-arm gate's own "
        "reference, so arming moves nothing. Also settable per process as MC_COND_MODE.",
    )
    ap.add_argument(
        "--mc-dz-anchor",
        default="hand",
        choices=["hand", "chest"],
        help="--mc-cond-mode absolute only: WHICH measurement fixes the rigid z lift. `hand` "
        "(default) puts the calibration frame's mounted gripper z at RIG_WRIST_STAND_Z 0.95; "
        "`chest` puts the chest puck at RIG_CHEST_STAND_Z 1.30. Commanded hand HEIGHT is a "
        "conditioned channel of the completion, so this decides the POSTURE (a low placement "
        "crouches the pelvis to 0.45 m against a 0.77-0.84 standing band). mc_episode's third "
        "value `train` is deliberately not offered here — it reproduces MC's own training "
        "distribution, which under-lifts this rig.",
    )
    ap.add_argument(
        "--mc-dz-wrist",
        type=float,
        default=None,
        help="--mc-cond-mode absolute only: PIN the rigid z lift in metres instead of deriving "
        "it from the calibration pose. This is the one number that makes the online and the "
        "OFFLINE completion place the take identically: offline `auto_dz_wrist` averages the "
        "whole episode, online there is one frame to average, so left to auto on both sides "
        "they are close but not equal. Run once, read the `[mc] absolute placement` line, then "
        "pass the same value here and to `run_mc_to_hl_wholebody.sh --dz-wrist`.",
    )
    ap.add_argument(
        "--mc-hand-lever",
        default=None,
        choices=["none", "tcp"],
        help="BOTH MC modes: whether to apply the puck->wrist leg before the placement. "
        "`none` (default, and what shipped) feeds the yaw-aligned PUCK where the wrist "
        "belongs — the bridge's long-standing model, exact in ROTATION for --mc-cond-mode "
        "home because a constant body-local leg cancels in a calibration-relative delta. "
        "`tcp` applies mc_frames.wrist_from_puck (52.8 mm of rotating lever), which makes the "
        "commanded gripper pose bit-identical to what mc_to_hl_wholebody.py builds from the "
        "same motion, and makes an operator's wrist TWIST stop swinging the robot's hand ~5 cm. "
        "Measured online-vs-offline on 802 real frames with `none`: home 0.0 deg / 14-26 mm, "
        "absolute 180.0 deg / 79-86 mm. Use `tcp` for a RECORDED session; it is off by default "
        "because it moves a validated teleop path's commanded hand by up to 26 mm. Also "
        "settable per process as MC_HAND_LEVER.",
    )
    ap.add_argument(
        "--mc-umi-dir",
        default=None,
        help="the UMI_collection checkout holding motion_completion/ (default "
        "$MC_UMI_DIR or ~/FUJI/third_party/UMI_collection).",
    )
    # --- RAW tracker feed for UMI_collection's recorder (OFF unless a port is given) ---
    ap.add_argument(
        "--umi-tracker-port",
        type=int,
        default=0,
        help="ALSO serve the RAW puck poses on this localhost port, in the exact frame format "
        "UMI_collection's bridges/tracker/vive_tracker_bridge.py emits, so "
        "recording/recorder.py --tracker-url can record a TELEOP session into the raw layout "
        "processing/convert_to_umi.py converts. 0 (default) = off, and the process is then "
        "byte-identical to before. Needed because the libsurvive dongle is single-claim: "
        "collection's own tracker bridge cannot run while teleop holds it. This is a "
        "MEASUREMENT stream, not the command on :8770 — see umi_tracker_snapshot.",
    )
    ap.add_argument(
        "--umi-tracker-rate-hz",
        type=float,
        default=UMI_TRACKER_RATE_HZ,
        help=f"--umi-tracker-port broadcast rate (default {UMI_TRACKER_RATE_HZ:g}, quoted from "
        "collection's tracker bridge so a teleop-recorded tracker.npy is sampled like a "
        "collection-recorded one).",
    )
    ap.add_argument(
        "--umi-world-rpy-deg",
        type=float,
        nargs=3,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="--umi-tracker-port only: the FIXED world alignment baked into the recorded poses, "
        "degrees, intrinsic Z(yaw)*Y(pitch)*X(roll) — the same rotation collection passes as "
        "WORLD_RPY (this rig: 0 0 180). Default identity. It is NOT the teleop path's "
        "world-yaw autocal and must not be: the autocal is re-solved from the operator's hand "
        "vector at every calibration, so a dataset stamped with it would sit in a different "
        "world frame after every recalibration.",
    )
    ap.add_argument(
        "--umi-stale-timeout",
        type=float,
        default=UMI_TRACKER_STALE_S,
        help=f"--umi-tracker-port only: seconds without a pose update before that role is "
        f"flagged `stale` on the raw feed (default {UMI_TRACKER_STALE_S:g}, collection's own "
        "value). The recorder stores the flag; it does not drop the sample.",
    )
    ap.add_argument(
        "--publish-mc-qpos",
        action="store_true",
        help="BOTH MC modes: also put the whole completed body on the wire — the 29 joint "
        "angles, their names, and the completion's own root pose in the readout frame — as "
        "`mc_qpos`/`mc_joint_names`/`mc_root_pos`/`mc_root_quat`. Two consumers draw it: "
        "vive/mc_teleop_gate_viz.py (viser, whole body instead of five spheres) and the "
        "BROWSER SIM's light-purple reference ghost, which is how you tell 'the completion "
        "wanted a step' apart from 'the policy did not take it'. Purely additive and OFF by "
        "default, so the payload stays byte-identical for every existing consumer. Costs ~36 "
        "floats plus the names per frame on loopback. Turn it ON whenever you are comparing "
        "the sim against the completion — without it the ghost draws nothing and says so.",
    )
    ap.add_argument(
        "--hand-point",
        choices=["lever", "tcp"],
        default="lever",
        help="WHICH point on each hand this bridge steers, and it MUST match the "
        "loaded policy. lever (default; byte-identical to before) = the historical "
        "0.18 m VR-tracker lever, for exp:g1-wbt-2pt-* and every pre-2026-08-03 3pt "
        "checkpoint. tcp = the 0.242 m fingertip TCP of the TCP-UNIFIED retrain "
        "(exp:g1-wbt-3pt-hybrid-unified[-anneal], run wda3h4ll onward), where position "
        "and force share ONE point. Requires --mode 3pt. The two are 6.7 cm apart per "
        "hand and the obs width is identical, so nothing downstream can catch a "
        "mismatch — see robot_config.VR_3POINT_OFFSETS_TCP.",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="discard cached lighthouse geometry and re-solve it (move/pause/move ~30-60 s -- "
        "see the startup banner; FRESH_CALIB=1 in 1_vive_bridge.sh)",
    )
    ap.add_argument(
        "--no-fresh-ootx",
        dest="fresh_ootx",
        action="store_false",
        help="with --fresh, re-solve only the base-station POSITIONS and KEEP each station's "
        "cached factory blob (skip libsurvive's --force-ootx). The escape hatch for a rig "
        "whose stations are unchanged and whose OOTX will not decode from where the pucks can "
        "be held -- that is the 'Bad sync bit' retry loop. FRESH_OOTX=0 in 1_vive_bridge.sh.",
    )
    ap.add_argument(
        "--min-stations",
        type=int,
        default=MIN_STATIONS,
        help=f"refuse to calibrate until this many SOLVED base stations are live (default {MIN_STATIONS}). "
        "The rig runs 2 or 3 interchangeably and this adapts to whatever is on; the floor exists "
        "because ONE lighthouse leaves a sparse tracker's yaw under-constrained, which corrupts the "
        "world-yaw autocal into a constant offset on both arms.",
    )
    ap.add_argument(
        "--expect-stations",
        type=int,
        default=0,
        help="0 (default) = adaptive, use however many are live. Set 2 or 3 to ASSERT that count: "
        "the bridge then refuses to calibrate on fewer, so a station that died since the last "
        "session is caught at startup instead of as mysteriously worse tracking.",
    )
    ap.add_argument(
        "--list-serials",
        action="store_true",
        help="discovery mode: print every tracker's LHR serial (and which "
        "hand it currently maps to, if --left-serial/--right-serial / the "
        "env are set), then EXIT. Use this when you swap a tracker to find "
        "its new serial — no calibration, no WebSocket server is started. "
        "Honors --fresh (re-solve a moved lighthouse) and --list-timeout.",
    )
    ap.add_argument(
        "--list-timeout",
        type=float,
        default=20.0,
        help="--list-serials: stop scanning after this many seconds even if "
        "fewer than two trackers were seen (default 20).",
    )
    ap.add_argument(
        "--calib-delay", type=float, default=3.0, help="seconds to hold the calibration pose before auto-calibration"
    )
    ap.add_argument(
        "--orientation",
        choices=["on", "off"],
        default="off",
        help="live wrist rotation. Default OFF = translation-first staging "
        "(orn held at the default-pose home). Toggle live: kill -USR2 <pid>.",
    )
    ap.add_argument("--reach-scale", type=float, default=1.0, help="multiply operator hand displacement (1.0 = 1:1)")
    ap.add_argument(
        "--z-offset", type=float, default=0.0, help="global vertical trim (m) added to both hand targets (pelvis frame)"
    )
    ap.add_argument(
        "--yaw-deg", type=float, default=0.0, help="initial world yaw (deg) libsurvive->pelvis; overridden by autocal"
    )
    ap.add_argument(
        "--no-yaw-autocal", action="store_true", help="do NOT derive world yaw from the L-R hand vector at calibration"
    )
    # --- which puck decides "forward" (--mode 5pt-direct only) -----------------
    ap.add_argument(
        "--world-yaw-from",
        choices=list(DIRECT_5PT_YAW_SOURCES),
        default=None,
        help="[--mode 5pt-direct ONLY] which puck the world frame is solved from (default "
        "chest). chest = the relabelled CHEST heading onto +x -- worn, not held, and the one "
        "relabel that was MEASURED. feet = the L-R FOOT vector onto +y (geometric). hands = "
        "the historical L-R HAND vector, which is what every other mode uses and is wrong "
        "whenever the hand pucks are parked or stacked front-to-back. fixed = --yaw-deg "
        "verbatim (collection's own WORLD_RPY convention). REFUSED on any other mode: their "
        "frame is a reviewed working path.",
    )
    ap.add_argument(
        "--yaw-trim-deg",
        type=float,
        default=0.0,
        help="[--mode 5pt-direct ONLY] add this to whatever the witness solved, e.g. to test a "
        "suspected 90 deg. Printed with the resolved yaw so a trimmed frame is never silent.",
    )
    ap.add_argument(
        "--world-yaw-tol-deg",
        type=float,
        default=40.0,
        help="[--mode 5pt-direct ONLY] refuse to calibrate when the CHEST and the FEET "
        "disagree about the operator's facing by more than this (default 40). 180 deg means "
        "the two ankle pucks are on the wrong ankles -- silent otherwise. Also the threshold "
        "for the informational hand-vector / foot-bracket notes.",
    )
    ap.add_argument(
        "--stale-timeout",
        type=float,
        default=0.25,
        help="if a tracker stops updating for >this many seconds, hold HOME "
        "instead of streaming the frozen pose (0 disables; default 0.25)",
    )
    ap.add_argument(
        "--no-align",
        action="store_true",
        help="debug: skip auto-calibration; emits home_pos + yaw-rotated raw "
        "puck meters (rig-origin relative — expect large offsets, NOT a "
        "true passthrough and NOT anchored to the operator)",
    )
    args = ap.parse_args()

    # Discovery mode short-circuits everything else: just scan + print serials.
    # Runs before the serial-requirement warning since its whole purpose is to
    # FIND the serials you don't know yet (e.g. after swapping a tracker).
    if args.list_serials:
        # `--list-serials --fresh` is REFUSED, because it is destructive AND cannot succeed.
        # It forwards --force-calibrate --force-ootx, clearing the pose and the OOTX of every
        # configured station; rebuilding those needs an OOTX decode per station plus a position
        # solve driven by a moving tracker, which takes far longer than the scan window
        # (default 20 s) and needs the waving the live bridge asks for. So the combination
        # reliably ends in the worst state: geometry wiped, nothing re-solved, every puck
        # reported "ON THE RADIO but NOT TRACKED" — which reads as a hardware fault. That is
        # exactly what it did on this rig (three solved stations -> zero, 2026-08-28).
        # Discovery is meant to be READ-ONLY; a re-solve belongs to the live bridge.
        if args.fresh:
            print(
                "[bridge] REFUSING --list-serials --fresh: discovery is read-only, and --fresh "
                "would clear the pose AND\n"
                "[bridge]     OOTX of every configured station without being able to re-solve "
                "them inside the scan window.\n"
                "[bridge]     Result: geometry wiped, every puck reported NOT TRACKED.\n"
                "[bridge]     To LOOK at the geometry:      (drop --fresh) -- it prints the "
                "cached stations either way.\n"
                "[bridge]     To RE-SOLVE a moved station:  run the live bridge with --fresh, "
                "every station powered on.",
                flush=True,
            )
            print(SOLVE_PROCEDURE, flush=True)
            sys.exit(2)
        sys.exit(_list_serials(args))

    if args.host not in ("127.0.0.1", "localhost"):
        print(f"[bridge] REFUSING non-localhost host {args.host!r}", flush=True)
        sys.exit(2)
    if not (args.left_serial or os.environ.get("VIVE_UMI_LEFT_SERIAL")) or not (
        args.right_serial or os.environ.get("VIVE_UMI_RIGHT_SERIAL")
    ):
        print(
            "[bridge] WARNING: left/right serials not both set — pass --left-serial/"
            "--right-serial or VIVE_UMI_LEFT_SERIAL/RIGHT_SERIAL, else trackers are "
            "UNMAPPED and no command is produced.",
            flush=True,
        )

    # Report the anchor for the geometry actually selected — printing the 0.18 m lever
    # numbers on a --hand-point tcp run is exactly the kind of quiet mismatch this flag
    # exists to prevent.
    if args.mode == "5pt-direct":
        # A DIFFERENT ROBOT POSE from the 3pt arrays (the READY pose, arms zeroed), so printing
        # the 3pt home here would show numbers 0.354 m per hand from the ones actually in use --
        # and that gap is the whole hazard this mode's anchor block documents.
        _hp_pos = anchor_default["home_pos_pelvis_5pt_tcp"]
        _hp_sep = float(
            np.linalg.norm(
                np.asarray(_hp_pos[0], dtype=float) - np.asarray(_hp_pos[1], dtype=float)
            )
        )
        _hp_tag = "TCP-unified 0.242 m fingertip, on the READY-pose 5pt anchor"
    elif args.hand_point == "tcp":
        _hp_pos = anchor_default["home_pos_pelvis_3pt_tcp"]
        _hp_sep = anchor_default["hand_separation_tcp_m"]
        _hp_tag = "TCP-unified 0.242 m fingertip"
    else:
        _hp_pos = anchor_default["home_pos_pelvis"]
        _hp_sep = anchor_default["hand_separation_m"]
        _hp_tag = "0.18 m VR lever"
    print(
        f"[bridge] hand point: {_hp_tag}  (--hand-point {args.hand_point})",
        flush=True,
    )
    print(
        f"[bridge] HOME anchor: L pos={_hp_pos[0]} R pos={_hp_pos[1]} "
        f"(pelvis frame, sep={_hp_sep:.3f} m)",
        flush=True,
    )
    if args.mode == "5pt-direct" and getattr(args, "publish_mc_qpos", False):
        print(
            "[bridge] NOTE --publish-mc-qpos does nothing in --mode 5pt-direct: there is no "
            "completion, so there are no 29 joints and no root pose to publish. The five "
            "keypoints are on the wire as usual; `4_pose_match_viz.sh` will draw them plus the "
            "green reference and say that the live BODY is not available.",
            flush=True,
        )
    if args.mode == "5pt-direct":
        print(
            f"[bridge] HOME anchor (5pt): chest={_hp_pos[2]} Lfoot={_hp_pos[3]} Rfoot={_hp_pos[4]}"
            "  -- this IS the low level's pre-arm gate reference (FK of the READY pose), so the"
            " calibration pose arms by construction.",
            flush=True,
        )
        print(
            "[bridge] ⚠️  5pt-direct steers the FEET, and the deploy launcher exempts them from"
            " the per-tick slew cap (SLEW_FEET=0), so an operator STEP reaches the robot at full"
            " rate. It is still NOT locomotion, twice over: `command_lower_body` keeps coming"
            " from the FROZEN idle clip (this bridge sends no `ref_joints`, the channel whose"
            " absence made a gait clip step IN PLACE), and the wire is PELVIS-FRAME, so WALKING"
            " is not expressible -- a real step leaves every keypoint about where it was"
            " relative to the walker's own pelvis, while this mode sends each puck's ABSOLUTE"
            " displacement. Single steps in place; treat the feet as stance/posture.",
            flush=True,
        )
        print(
            f"[bridge] WORLD FRAME from {(getattr(args, 'world_yaw_from', None) or 'chest').upper()}"
            f" (--world-yaw-from; trim {float(getattr(args, 'yaw_trim_deg', 0.0)):+.1f} deg, "
            f"chest-vs-feet tol {float(getattr(args, 'world_yaw_tol_deg', 40.0)):.0f} deg)."
            " This is the ONE frame decision on the path: it turns EVERY commanded point at once,"
            " and a wrong one still leaves the start pose exact and every gate green -- the rig"
            " symptom is 'foot FORWARD, robot steps RIGHT'. Every witness is printed at"
            " calibration; the resolved frame rides on the heartbeat.",
            flush=True,
        )
    print(
        f"[bridge] PID {os.getpid()} -- recal: kill -USR1 {os.getpid()} (or touch "
        f"{RECAL_FLAG}); orientation toggle: kill -USR2 {os.getpid()} (or touch "
        f"{ORIENT_FLAG})",
        flush=True,
    )

    bridge = ViveTeleopBridge(args)
    # In --mode 5pt / dense, bring the completion up BEFORE claiming the pysurvive
    # singleton: the backend load is ~10 s (torch + JAX + the checkpoint) and a failure
    # there should abort while the radio is still free, not after the operator has been
    # asked to hold a calibration pose.
    if not bridge.start_mc():
        print("[bridge] ERROR: motion completion failed to start -- aborting", flush=True)
        return 2
    stop = threading.Event()
    pys_thread = threading.Thread(target=bridge.run_pysurvive, args=(stop,), daemon=True)
    pys_thread.start()
    try:
        asyncio.run(_serve_all(bridge, args, stop))
    except KeyboardInterrupt:
        print("\n[bridge] interrupted.", flush=True)
    finally:
        stop.set()
        pys_thread.join(timeout=1.0)
        bridge.stop_mc()


if __name__ == "__main__":
    main()
