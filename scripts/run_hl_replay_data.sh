#!/usr/bin/env bash
# Replay RECORDED data to the low-level controller — NO policy, NO GPU, NO inference.
#
# This is the sibling of run_hl_serving.sh for the case where you are NOT evaluating a
# model: you want the rig to execute one of the collected human demonstrations (e.g. one of
# the 94 clean94 episodes) so you can watch the arm reproduce the take. The action chunk is
# read straight out of the .zarr.zip, so a checkpoint would be dead weight — loading one
# would pin a GPU and several GB of VRAM to run a data playback.
#
# It is the SAME program (scripts/hl_serving.py) and therefore the SAME safety envelope as
# the policy planner: identical :8770 wire, identical execute-N-of-S drip (16/30/50), identical
# force gate (default OFF = pure position), identical clutch re-anchor to the measured EE.
# Only the CHUNK SOURCE differs. Run run_hl_serving.sh instead when you want the policy, or
# when you want to A/B the demo against the policy live in one page.
#
# WHAT YOU GET IN THE PAGE (http://localhost:$PORT):
#   * Episode           clickable dropdown over every episode (frame counts shown). Episodes
#                       whose first frames were MASKED by the converter are labelled
#                       "⚠ starts at N" and auto-seek there (on clean94, 8 of 94 start masked).
#   * Start frame       scrub anywhere; a masked frame snaps forward to the next playable one
#   * Load 1 recorded chunk -> Run this chunk   one chunk at a time (preview, then execute)
#   * Execute how many steps  16 | 30 | 50 — how many of the chunk's steps reach the robot
#                       (0.53 s / 1.0 s / 1.67 s of demo at the 30 Hz action grid)
#   * ⏩ Play episode    SEQUENTIAL: repeats load->execute to the episode end (arm ONCE),
#                       ⏹ Stop interrupts, ⏮ Rewind returns to the first playable frame
#
# WHY PLAY USED TO STUTTER (fixed 2026-08-08) — "一顿一顿", moves then stops then moves. It was
# NOT the low level, NOT the slew limiter and NOT the action grid: between chunks the planner ran
# the mink preview IK ACTION_HORIZON times, TWICE per chunk (preview + executed). At 6.14 ms per
# solve that is **599 ms measured of arm-idle time** against 533 ms of motion at exec-steps=16 —
# the arm moved for half a second, then froze for six tenths. And the mesh it solved was DISCARDED
# in both modes anyone runs: with RECORD=1 the live joints override it every render tick, and
# OFFLINE it is deliberately frozen at the ready pose. It is now solved only when it is really
# rendered (a low level running WITHOUT RECORD=1) and memoized across the preview/executed pair:
#     exec=16  duty 47% -> 99%      exec=30  62% -> 99%      exec=50  74% -> 99%
# What REMAINS is inherent, not a bug: every chunk re-commands its step 0 (the pose the arm already
# holds), so there is one 33 ms dwell per chunk boundary — 6.2% at exec=16, 2.0% at exec=50. If you
# still want it smoother, raise --exec-steps (fewer boundaries: 30 chunks at 16, but 10 at 50).
#   * green path        the WHOLE recorded episode drawn at the LEFT TCP — same green as
#                       replay_zarr_viz.py, transported by the SAME anchor the chunk decodes
#                       against, so the drawn demo and the commanded motion cannot disagree
#   * RED path          where the arm ACTUALLY went: the MEASURED LEFT TCP (RecordTap FK),
#                       recorded while a chunk is dripping, drawn against the green commanded
#                       path. Needs RECORD=1 (no live joints ⇒ nothing to measure). It samples
#                       the SAME TCP point the green line is drawn at, so red-vs-green in the
#                       scene is servo/IK error and nothing else.
#   * 3 axis triads     tcp / tracker / wrist_yaw_link, at the SAME wrist-relative offsets as
#                       the collection viz + zarr replay, FOLLOWING the live robot wrist. The
#                       green path starts exactly at the "tcp" triad — that is the datum.
#
# "EXECUTED VS RECORDED" PANEL — judge a replay instead of eyeballing it:
#   * 💾 Save comparison plot (PNG)   writes a QC sheet in the plot_take_qc.py house style:
#         row 0  x/y/z (mm)            SOLID = recorded data, DASHED = executed
#         row 1  roll/pitch/yaw (deg)  same convention, per-axis colours (xyz Euler, unwrapped)
#         row 2  |position error| (CM) + rotation geodesic (deg)
#         row 3  jaw width (mm)        recorded vs measured
#     Units: absolute POSES in mm (the frame contract is written in mm) and the jaw in mm (its
#     43-94 mm is the physical rig number), but every DISCREPANCY in cm — the error panel and
#     the title's RMS/max, so they read in the same unit as the path lengths beside them.
#     Title line carries pos RMS/max, rot mean/max, and EXECUTED-vs-RECORDED PATH LENGTH — the
#     last one is what catches a replay that tracked the shape but only travelled part of it
#     (the advance_frame off-by-one did exactly that: 70% of the arc with a per-sample error
#     that looked fine). NOTHING is written unless you press the button; the resolved absolute
#     path is then shown in the page (destination: --trace-dir, default <repo>/data/traces).
#   * 🧹 Clear executed path          re-arm the trace at the current cursor (refused mid-motion)
#   The comparison is against the TRANSPORTED (commanded) demo, not the raw recorded world
#   pose — the ~55° ready-pose reorientation below is physical, and differencing against the
#   raw recording would report it as a huge tracking error while hiding the real one.
#
# ↺ Reset start pose ALSO OPENS THE GRIPPER (2026-08-08). A replay that ended mid-grasp used to
# leave the jaw clamped while "reset" restored only the ARM's datum — so the next play began
# from a gripper state the demonstration never started in. It now sends the same fully-open
# target_frac=1.0 the teardown path uses. Only with --command-gripper (without it this planner
# is read-only on the jaw by contract, and the status line says so rather than silently
# opening a hand that may be holding something).
#
# OFFLINE=1 — LOOK AT THE FRAME WITH NOTHING RUNNING (start here):
#
#   OFFLINE=1 PORT=8088 ./scripts/run_hl_replay_data.sh
#
# No low level, no bridges, no GPU, no robot — just the viser page. It opens NO sockets and
# publishes NOTHING, so it is safe to run on a workstation with the rig powered down. Execute is
# hard-disabled (refused at the entry of the drip path, not merely greyed out). What you get:
#   * the yam URDF posed at the low level's REAL ready pose (home_anchor.default_qpos)
#   * the 3 triads on its LEFT wrist: tcp / tracker / wrist_yaw_link. "tracker" now renders the
#     PHYSICAL PUCK's axes (the measured -y,-x,-z relabel) instead of inheriting the wrist's, so
#     it is visibly 180 deg from the other two — as it must be, since the TCP is DERIVED from the
#     raw puck through that relabel.
#   * "Load 1 recorded chunk" previews the demo against the ready-pose anchor, and the green
#     recorded path is drawn from the tcp triad.
#   * a "frame seam" readout: on a current-convention dataset it says "identity (no seam)".
# This is the mode for answering "is the frame right?" without touching hardware.
#
# THE READY POSE IS "ARMS ZEROED" — AND THIS PLANNER NOW SEEDS THERE (2026-08-07).
# 3_run_policy_vive_force.sh launches with HOLOSOMA_INIT_READY_POSE=1, so after you press A the low
# level pins BOTH its INIT target and its frozen VIVE hold to `default_dof_angles with all 14 ARM
# joints = 0` — forearms HORIZONTAL, both hands (holosoma_wbt::_maybe_apply_init_ready_pose, whose
# own log line says exactly that). The PRE-ARM POSE GATE then compares the broadcast wire against
# that same held reference (_motion_command_t[0,:29] -> _compute_vr_fk).
#
# This planner used to seed its render, its decode anchor and its idle wire from the RAW crouch
# `home_anchor.default_qpos` instead, which is a DIFFERENT pose:
#     per-hand vr_3point contact gap  =  310.2 mm      pre-arm gate tolerance  =  70 mm
# i.e. the first broadcast wire was outside the gate. On a rig with RECORD=1 the idle heartbeat
# re-syncs the wire to the MEASURED pose before you arm, which is what had been masking it. Now the
# seed IS the arms-zeroed pose, so the page matches the robot the moment it opens, the forearms
# render level instead of drooping 44 deg, and the recorded trajectory starts at the drawn tcp triad
# in the pose the arm is actually in — which is what makes "press start, then replay" continuous.
# Use RAW_CROUCH=1 only for a low level started with HOLOSOMA_INIT_READY_POSE=0.
#
# PREREQUISITES for EXECUTING (not needed for OFFLINE=1):
#   * REQUIRED: the LOW-LEVEL controller in ABSOLUTE mode with RECORD=1 (companion command
#     printed below). RECORD=1 is what lets the planner see the real joints, so the triads +
#     mesh track the robot and the pre-arm pose gate passes with ~0 gap.
#   * OPTIONAL: the F/T bridge (:8771) and camera bridge (:8773). Replay never feeds them to
#     a model — they are DISPLAY only — so this launcher does not require them and the
#     self-check downgrades their absence to a note instead of refusing to start. (In
#     particular a bridge without --pipeline, i.e. no 6-DoF 'left_pipe', is fine here.)
#   * OPTIONAL: the gripper bridge (:8784), needed only to also replay the JAW
#     (--command-gripper) and to show live width.
#
# Usage:
#   OFFLINE=1 PORT=8088 ./scripts/run_hl_replay_data.sh             # VIZ ONLY, no hardware
#   PORT=8088 ./scripts/run_hl_replay_data.sh                       # default dataset below
#   PORT=8088 ./scripts/run_hl_replay_data.sh --replay-episode 7
#   PORT=8088 ./scripts/run_hl_replay_data.sh --exec-steps 8
#   ZARR=path/to/other.zarr.zip ./scripts/run_hl_replay_data.sh
#
# WHY THIS IS FRAME-SAFE: the recorded chunk is handed over in the model's OWN anchor-RELATIVE
# form, so the recording's lighthouse world frame CANCELS exactly and the demo replays from
# wherever the arm actually is. Nothing aligns lighthouse-world to pelvis, and there is no
# extra Rx(180) flip (the recorded poses are already puck-frame, like the planner's anchor).
# See flore/s1_policy/serving/hl_replay.py.
#
# ⚠ EXPECT THE DEMO TO LOOK REORIENTED — but only by ~55°, not ~107°. Because the chunk is
# anchor-relative, the stroke is rigidly rotated by R_wrist_robot @ R_wrist_handheld^T: measured
# **55.3° mean (46.2–78.4)** on clean94. That is the real angle between how you HELD the tool and
# how the G1 holds it at its ready pose (gripper forward, 44° NOSE-DOWN, 47° off level), and it is
# irreducible while step 0 stays on the measured pose — the only way to remove it is a
# demo-compatible READY POSE. So the green path will NOT overlay the trajectory you saw in
# collect_data.sh / run_replay_viz.sh / run_replay_zarr_viz.sh: it is CONGRUENT to it (the tool's
# angle to its own path is preserved to 0.000°) but REORIENTED, and the arm executes what is drawn.
#
# ⚠ THIS USED TO SAY 106.9°, AND ~52° OF THAT WAS A BUG (fixed 2026-08-07). The URDF<->model frame
# seam was a hardcoded Rx(180) = diag(1,-1,-1). A 180° flip was genuinely needed for the OLD
# datasets, but about the WRONG AXIS: the true puck->wrist relabel is 180° about (1,-1,0)/√2, whose
# axis is 45° from +x, so the two composed to a spurious **90° about the wrist z**. It cost 47.96°
# mean tool attack-angle error (69.5% of samples beyond 30° — for a wiping task that geometry IS
# the task) and fed the model a wrt_start wrong by 2.58° mean / 26° max. The seam is now DERIVED
# from the dataset's own T_tracker_tcp, and on the current convention it is EXACTLY IDENTITY: the
# low level is fed the raw-tracker action through ONE extrinsic and nothing else.
#
# A world-oriented replay was separately measured and rejected: it destroys the tool-relative
# geometry, makes the command depend on lighthouse calibration (13 cm), and drives episodes into
# the torso. Full measurements: flore/s1_policy/serving/hl_replay.py::_body_vs_world_transport.
#
# Env knobs:
#   FLORE_PY   python (default: <repo>/.venv/bin/python)
#   PORT      viser port (default: 8088)
#   OFFLINE   1 = viz only, no low level / no bridges / no publish (see above). Default 0.
#   RAW_CROUCH  1 = seed at the RAW home_anchor default_qpos instead of the arms-zeroed ready pose.
#             Default 0. Only for a low level started with HOLOSOMA_INIT_READY_POSE=0 — see the
#             READY POSE note below; with the launcher's default (=1) this puts the broadcast wire
#             310 mm/hand from the low level's pre-arm gate reference against a 70 mm tolerance.
#   ZARR      dataset to replay (default: the newest CURRENT-CONVENTION session)
#   LAYOUT    action layout. Only 'left_only' is accepted today — this planner applies the
#             URDF<->model frame flip and builds the vr_3point LEFT slot for the LEFT hand
#             ONLY, so anything else is rejected by --layout-mode rather than silently
#             driving the right arm 180 degrees off. Present so the constraint is visible.
# Extra args pass through to hl_serving.py, e.g.:
#   --replay-episode N       initial episode (default 0; selectable live in the page)
#   --exec-steps 16          steps executed per chunk (default 16; 16/30/50 live in the page)
#   --command-gripper        also replay the recorded JAW width (default OFF = read-only), and
#                            let '↺ Reset start pose' re-open the jaw
#   --trace-dir DIR          where 'Save comparison plot' writes (default <repo>/data/traces)
#   --no-selfcheck           skip the bridge probe entirely
set -eo pipefail

FLORE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLORE_PY="${FLORE_PY:-$FLORE_ROOT/.venv/bin/python}"
UMI="$FLORE_ROOT/third_party/UMI_collection"
IK_SRC="$UMI/third_party/humi_ik"
VIZ_SRC="$UMI/viz"
REPLAY_SRC="$UMI/replay"
PORT="${PORT:-8088}"
LAYOUT="${LAYOUT:-left_only}"
OFFLINE="${OFFLINE:-0}"
RAW_CROUCH="${RAW_CROUCH:-0}"
case "$OFFLINE" in
  0|1) ;;
  *) echo "[run] OFFLINE must be 0 or 1 (got '$OFFLINE')" >&2; exit 2 ;;
esac
case "$RAW_CROUCH" in
  0|1) ;;
  *) echo "[run] RAW_CROUCH must be 0 or 1 (got '$RAW_CROUCH')" >&2; exit 2 ;;
esac
# The newest session on the CURRENT (post-2026-08-06) puck->TCP convention, i.e. the one where the
# stored pose IS the wrist-oriented fingertip TCP and the frame seam resolves to identity.
#
# ⚠ NOT clean94. That session (and every dataset converted before 2026-08-06) has an
# IDENTITY-rotation extrinsic, which means a WRIST-frame lever was applied in the PUCK frame: its
# robot0_eef_pos is ~33 cm off in position and 180 deg off in rotation. hl_serving.py REFUSES such a
# file outright rather than serving a trajectory anchored to the wrong physical point. Re-convert a
# session with:  processing/convert_to_umi.py <session_dir> --hands left --hl-windows -o <out>
# ⚠ ALSO NOT a v2-grid file. Since 2026-08-07 the HL contract is 1 force bin at t0 (no
# history) -> 50 steps at 30 Hz, and hl_dataset._check_file REFUSES any zarr whose
# hl_action_horizon / hl_action_hz / hl_force_hist_len attrs disagree. Re-convert with the
# current converter (which defaults to the v3 grid); the v2 grid is still reachable via
# --hl-force-hist-len 8 --hl-action-horizon 16 --hl-action-hz 16 --hl-force-hist-span-s 1.0.
ZARR="${ZARR:-$UMI/data/datasets/session_2026.08.07_1556_v3.zarr.zip}"

if [ ! -x "$FLORE_PY" ]; then
  echo "[run] FLoRe venv python not found at $FLORE_PY (create .venv, then: uv pip install -e \".[viz]\")" >&2
  exit 2
fi
if [ ! -e "$ZARR" ]; then
  echo "[run] dataset not found: $ZARR" >&2
  echo "[run] pass one with ZARR=/path/to/session.zarr.zip $0" >&2
  echo "[run] (it must be a CURRENT-convention conversion — see the ZARR note in this script)" >&2
  exit 2
fi

EXTRA_ARGS=()
[ "$RAW_CROUCH" = "1" ] && EXTRA_ARGS+=(--no-zero-arms)
if [ "$OFFLINE" = "1" ]; then
  EXTRA_ARGS+=(--offline)
  echo "[run] OFFLINE FRAME VIZ on http://localhost:$PORT  (Ctrl-C to stop)"
  echo "[run]   python: $FLORE_PY  —  NO model, NO GPU, NO low level, NO bridges, NOTHING published"
  echo "[run]   dataset: $ZARR"
  echo "[run]   The page shows the yam URDF at the low level's REAL ready pose with the three"
  echo "[run]   triads (tcp / tracker / wrist_yaw_link) on its LEFT wrist. 'tracker' renders the"
  echo "[run]   PHYSICAL PUCK axes (the -y,-x,-z relabel), so it is 180 deg from tcp/wrist — as it"
  echo "[run]   must be, since the TCP is DERIVED from the raw puck through that relabel."
  echo "[run]   'Load 1 recorded chunk' previews the demo + green path from the tcp triad."
  echo "[run]   EXECUTE IS DISABLED in this mode. Drop OFFLINE=1 to drive the rig."
else
  echo "[run] RECORDED-DATA REPLAY on http://localhost:$PORT  (Ctrl-C to stop)"
  echo "[run]   python: $FLORE_PY  —  NO model, NO GPU (this is a data playback)"
  echo "[run]   dataset: $ZARR"
  echo "[run]   TIP: OFFLINE=1 $0  — inspect the frame first, with no hardware at all."
  echo "[run]   REQUIRED: the LOW-LEVEL in ABSOLUTE mode with RECORD=1 (pure position — force wire OFF):"
  echo "[run]     cd $FLORE_ROOT/third_party/holosoma_extensions/force_deploy && \\"
  echo "[run]       MODE=3pt-unified-anneal HOLOSOMA_VIVE_RELATIVE=0 RECORD=1 bash 3_run_policy_vive_force.sh"
  echo "[run]   OPTIONAL: F/T (:8771) + camera (:8773) bridges — DISPLAY only; replay does not read them."
  echo "[run]   ⚠ ARM BEFORE EXECUTE: the low level IGNORES this wire until you ARM it. On the joystick:"
  echo "[run]       (1) Y = ramp to ready pose  →  (2) A = START the policy  →  (3) 'start' button = ARM."
  echo "[run]     Only after 'start' (StateCommand.STAND_TOGGLE) does the arm track this replay. A alone"
  echo "[run]     just stands the policy — Execute in the page will do NOTHING until you press 'start'."
  echo "[run]   SAFETY: '⏩ Play episode' moves the arm CONTINUOUSLY to the episode end after one press."
  echo "[run]           Keep the hardware E-stop in hand; ⏹ Stop retires the loop after the current chunk."
fi
PYTHONPATH="$IK_SRC:$VIZ_SRC:$REPLAY_SRC:${PYTHONPATH:-}" "$FLORE_PY" \
  "$FLORE_ROOT/scripts/hl_serving.py" \
  --replay-zarr "$ZARR" \
  --layout-mode "$LAYOUT" \
  --port "$PORT" \
  "${EXTRA_ARGS[@]}" \
  "$@"
