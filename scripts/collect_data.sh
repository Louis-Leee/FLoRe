#!/usr/bin/env bash
# Collect UMI training data: thin wrapper around UMI_collection's
# run_collection_viz.sh with a selectable tracking MODE.
#
#   MODE=1pt : LEFT hand only    — tracker + camera + F/T + gripper
#   MODE=2pt : + RIGHT hand       — its OWN tracker + camera + F/T + gripper
#   MODE=3pt : + CHEST            — tracker ONLY (a body reference)     (default)
#   MODE=5pt : + BOTH FEET        — two more tracker-ONLY body references
#
# ─── VIVE TRACKERS: the reference recorder flow (umi_collection) ───
#
# ONE launch, ONE libsurvive process, every puck the MODE needs POWERED (solid green) and in
# view of the base stations BEFORE you press enter. That is the whole ritual. Verbatim from
# `umi_collection/sensors/vive_tracker.py` + `scripts/record_session.py`:
#
#   * libsurvive args `--force-calibrate 1 --force-ootx 1 --lighthouse-update-velocity 0.5`
#     on EVERY launch (the cache file is reset first): the stations are re-read (OOTX) and
#     RE-SOLVED at every process start. Each launch is therefore its own world frame; poses
#     are comparable within a launch, and WORLD_RPY stays valid because libsurvive anchors
#     the frame on the station with the lowest hardware BaseStationID, not on discovery order.
#   * The launch then BLOCKS until every configured tracker role has produced a pose
#     (`wait_until_tracking`, the reference recorder's --vive-warmup-timeout) — TRACKER_STARTUP_WAIT, default
#     120 s — and only then starts the cameras. Nothing to do while it waits. A reference Orin
#     with 2 stations and 5 pucks takes ~20-25 s; more stations = more blobs to decode.
#   * A puck that never streams is a FAILED launch, named per puck on the status board
#     (the bridge's table + VERDICT: off the radio / sees no station / lost its pose /
#     serial mismatch, versus the shared room solve). Fix THAT puck and relaunch.
#
# BASE STATIONS. Two is the floor (MIN_STATIONS=2, blocking: a one-station solve yields
# poses but no parallax). The set libsurvive uses is whatever the pucks can SEE from where
# they are when the launch starts, and it only grows during a run — so power exactly the
# stations you mean to use. To compare 2/3/4/6 stations, set STATIONS=N (the count you
# powered): the board then reports `N solved of M seen` and the time to READY, and warns
# when the seen count differs from N. Every station seen must decode its OOTX blob before
# libsurvive attempts ANY room solve, which is why fewer stations lock faster.
#
# There is no calibration pass, no cache reuse and no hold-still protocol: TRACKER_CALIB
# (fresh|ootx-only|reuse) is kept as an explicit escape hatch in the launcher and defaults
# to fresh. `ootx-ignore-sync-error` is written into libsurvive's config (default on): a
# tolerated sync bit cannot admit a corrupt blob (every frame is CRC32-checked) and it
# stopped the frame restarts that were stretching OOTX here.
#
# The RIG's 5pt layout, which is what MODE selects:
#
#     LEFT hand  : VIVE tracker + D405 camera + Bota F/T + DM4310 gripper
#     RIGHT hand : VIVE tracker + D405 camera + Bota F/T + DM4310 gripper
#     CHEST      : VIVE tracker ONLY  — no camera, no F/T, no gripper
#     LEFT/RIGHT FOOT (5pt) : VIVE tracker ONLY — the same statement as the CHEST
#
# 5pt is 3pt PLUS TWO PUCKS and nothing else: the feet ride the chest's code paths (same
# wire, same tracker_*.npy layout, same rotation-only puck->body relabel, bare axis triad,
# no camera/F-T/gripper). Role names are `left_foot` / `right_foot` (FLoRe's own
# `flore/s1_policy/data/layout.py::KEYPOINT_ORDER`). 5pt is gripper-only display
# (DISPLAY_MODE=humanoid is refused: the collection IK has no foot task), and the converter
# does not emit a foot keypoint yet (FLoRe derives `wb_*_foot_*` by FK; a recorded foot
# replacing that is a separate schema decision).
#
# EVERY RIGHT-HAND DEVICE IS OPTIONAL AND ITS ABSENCE IS A FACT, NOT AN ERROR: a 3pt take
# without the right camera/Bota/gripper carries tracker_right.npy and no camera_right/, and
# the converter emits no camera1_rgb / robot1_ft_wrench. Nothing is zero-filled. The
# pre-launch STATUS BOARD reports each missing device as ⚠️ degraded.
#
# PINNED DEVICE IDS (both handheld units, confirmed on the bus 2026-08-25) — resolved by a
# STABLE identifier, never by enumeration order (with two of each, "the first one" is a coin
# flip that mislabels a whole session):
#                    LEFT (primary)              RIGHT
#   tracker          LHR-75106789                LHR-60CDD21F
#   chest / feet     head LHR-3DC276BA           left_foot LHR-C68495D2  right_foot LHR-FF1D3DE2
#   D405 camera      323622272456                353322270760
#   Bota F/T         3875355D3532                347237663335
#   DM4310 CAN       can-umi                     resolved by CANable serial
#                    (0014002733354B0432333837)  (001D005433354B0332333837)
# These are the HANDHELD units; holosoma's robot-mounted pair (cameras 323622270532 /
# 353322271384, Botas 3870355D3532 / 386B355D3532) are DIFFERENT devices. Override any
# tracker with LEFT_SERIAL / RIGHT_SERIAL / HEAD_SERIAL / LEFT_FOOT_SERIAL / RIGHT_FOOT_SERIAL;
# list what is on the radio (read-only) with
#     third_party/UMI_collection/.venv-vive/bin/python \
#         third_party/UMI_collection/bridges/tracker/vive_tracker_bridge.py --list-serials
# LEFT-canonical: LHR-75106789 is the LEFT (primary) hand in EVERY mode — never remap it to
# 'right' for 1pt (that reversed mapping is the ~180° yaw flip the convention exists to kill).
#
# RIGHT-hand knobs (2pt/3pt/5pt; all pass straight through): RIGHT_CAMERA=0 / RIGHT_FT=0 /
# RIGHT_GRIPPER=0 skip a device; CAMERA_SERIAL= / RIGHT_CAMERA_SERIAL=, FT_LEFT_SERIAL= /
# FT_RIGHT_SERIAL= (MANDATORY to pin with two Botas: by-id names sort 3472… before 3875…),
# GRIPPER_RIGHT_CAN_SERIAL= / GRIPPER_RIGHT_CHANNEL=canN. Gripper keys, hold to move: f = LEFT
# close, g = LEFT open, j = RIGHT close, k = RIGHT open — off viser's camera keys w/a/s/d/q/e.
# Pressing `s` (or Start) AUTO-TARES every attached Bota: hold both grippers clear of contact.
#
# Pre-launch self-check + go/no-go gate: device preflight, VIVE trackers, gripper self-test,
# F/T wrench, stream health, optional axis check → one STATUS BOARD (✅/⚠️/❌/⏭️), then a
# confirm. Chatty bridges go to data/logs/{tracker,camera,gripper}.log (LOG_BRIDGES=0 keeps
# them inline); libsurvive's own OOTX / solve lines are echoed during the tracker wait.
# Escape hatches: PREFLIGHT=0 SELFTEST=0 SELFCHECK=0 --no-ft AXISCHECK=1 FORCE_START=1.
#
# What the viz DRAWS (display only): DISPLAY_MODE=auto (default) = gripper-only in every
# mode (yam meshes + frustum per hand, a bare triad per body puck, no IK); humanoid = the
# full G1 + mink IK (1pt/2pt/3pt only; needs RAW_FRAME=0); gripper = same as auto.
# RAW_FRAME=1 (DEFAULT since 2026-09-11; gripper-only display): draw the cluster at the puck's RAW libsurvive pose with
#   NO home anchor — the "tracker" triad IS the raw puck, the TCP is the one fixed puck->TCP
#   transform, wrist_yaw at TCP-0.242 m. A faithful twin of the recorded tracker.npy frame:
#   use it after moving/re-choosing base stations to confirm WORLD_RPY (Z up, +X = your
#   forward) and to eyeball the puck->TCP extrinsic. Display only; recording is unchanged.
#
# CAMERAS: CAMERA_WS_PORT=8873 / CAMERA_RIGHT_WS_PORT=8877 by default — NOT 8773/8777, which
# the deploy stack's robot-mounted bridges bind on the same box (a clash killed a whole
# session's images once; the launcher's port gate also checks). CAMERA_FPS=30 capture,
# CAMERA_PUBLISH_HZ=30 publish: the publish rate is the dataset's master timeline and must
# divide CAMERA_FPS (uniform decimation); 15 still converts, with a coarser action grid.
#
# Prereqs (one-time): the submodule's bootstrap.sh (three uv venvs + source-built libsurvive).
#
# Usage:
#   bash scripts/collect_data.sh                     # 3pt viz only (no recording)
#   RECORD=1 bash scripts/collect_data.sh            # 3pt + record takes to data/recordings
#   MODE=5pt RECORD=1 bash scripts/collect_data.sh   # all five pucks on, then record
#   MODE=5pt STATIONS=2 GRIPPER=0 KEYBOARD=0 bash scripts/collect_data.sh   # station-count test
#   MODE=1pt RECORD=1 bash scripts/collect_data.sh   # LEFT hand only
#   MODE=2   RECORD=1 bash scripts/collect_data.sh   # short form of MODE=2pt
#   bash scripts/collect_data.sh --mock              # synthetic, no hardware
#
# After recording, QC the session before converting:
#   third_party/UMI_collection/.venv/bin/python \
#       third_party/UMI_collection/processing/audit_session.py <session_dir>
#   bash third_party/UMI_collection/run_take_qc.sh <session_dir>   # per-take PNG sheets
# The no-jump requirement is CONJUNCTIVE over the MODE's keypoints (2pt: both hands; 3pt: all
# three). A declared keypoint that jumps or recorded nothing is an ERR; the converter masks
# only the PRIMARY's glitches, so `--exclude` is the remedy for a right/head/foot glitch.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: MODE=1pt|2pt|3pt|5pt bash scripts/collect_data.sh [--mock] [UMI options...]

Runs UMI_collection with the given tracking MODE (default 3pt):
  1pt : LEFT hand  — tracker + camera + F/T + gripper
  2pt : + RIGHT hand — its OWN tracker + camera + F/T + gripper
  3pt : + CHEST      — tracker ONLY (body reference; no camera/F-T/gripper)
  5pt : + BOTH FEET  — two more tracker-ONLY body references, handled exactly like
        the CHEST. Defaults LEFT LHR-C68495D2 / RIGHT LHR-FF1D3DE2 (override with
        LEFT_FOOT_SERIAL= / RIGHT_FOOT_SERIAL=). Gripper-only display.
MODE may also be given as 1, 2, 3 or 5. RECORD=1 records takes; --mock needs no hardware.

VIVE trackers (the reference recorder flow): power EVERY puck the MODE
needs (solid green) in view of the base stations, then launch. libsurvive runs
--force-calibrate 1 --force-ootx 1 --lighthouse-update-velocity 0.5 on a reset cache
(stations re-read + re-solved every launch) and the launch BLOCKS until every role has
produced a pose — TRACKER_STARTUP_WAIT, default 120 s — before the cameras start.
Nothing to do while it waits. A puck that never streams FAILS the launch and is named.

Base stations: MIN_STATIONS (default 2) is the floor — fewer solved stations is a
blocking ❌. STATIONS=N declares how many you powered for a 2/3/4/6 comparison: the
board prints `solved of seen` + time to READY and warns when seen != N. Every station
seen must decode its OOTX blob before ANY room solve, so fewer stations lock faster.

Right-hand devices are OPTIONAL (reported ⚠️ degraded, recorded as absent). Skip one
with RIGHT_CAMERA=0 / RIGHT_FT=0 / RIGHT_GRIPPER=0. Gripper keys (hold to move):
  f = LEFT close   g = LEFT open      j = RIGHT close   k = RIGHT open

Once the base stations are solved the launcher STOPS at a gate, prints each station's live
acc err (|up_z−1| against its tilt sensor, ~0 is good; refreshed every 2 s) and waits for
ENTER before the gripper self-test / cameras / F/T (STATION_GATE=0 skips; TTY only).
Before recording the launcher prints a self-check STATUS BOARD and asks go/no-go;
a blocking failure defaults to abort. Skip knobs: PREFLIGHT=0 SELFTEST=0 SELFCHECK=0
--no-ft AXISCHECK=1 FORCE_START=1. DISPLAY_MODE=auto|humanoid|gripper picks what the
viz draws (display only; humanoid is refused for 5pt and needs RAW_FRAME=0).
RAW_FRAME=1 (DEFAULT) draws the gripper cluster at the puck's RAW libsurvive pose, no home
anchor — the twin of the recorded tracker.npy frame (what replay_raw_viz shows afterwards),
so the axes you see are the axes you record. RAW_FRAME=0 = the old home-anchored view.
Cameras: CAMERA_WS_PORT=8873 CAMERA_RIGHT_WS_PORT=8877 (8773/8777 are the deploy stack's),
CAMERA_FPS=30, CAMERA_PUBLISH_HZ=30 (the dataset's master grid; must divide CAMERA_FPS).
EOF
}

for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    usage
    exit 0
  fi
done

# Normalize MODE (default 3pt; accept the short forms 1/2/3/5) and validate BEFORE
# launching, so a typo fails here with the usage text instead of deep in the stack.
MODE="${MODE:-3pt}"
case "$MODE" in
  1|1pt) MODE=1pt ;;
  2|2pt) MODE=2pt ;;
  3|3pt) MODE=3pt ;;
  5|5pt) MODE=5pt ;;
  *)
    echo "[collect_data] ERROR: bad MODE='$MODE' (want 1pt|2pt|3pt|5pt, or 1|2|3|5)" >&2
    usage >&2
    exit 2
    ;;
esac

# Station knobs are validated here too, so a typo fails before any device is touched.
for _k in STATIONS MIN_STATIONS TRACKER_STARTUP_WAIT; do
  _v="${!_k:-}"
  if [ -n "$_v" ] && ! [[ "$_v" =~ ^[0-9]+$ ]]; then
    echo "[collect_data] ERROR: $_k='$_v' must be a whole number" >&2
    exit 2
  fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UMI="$REPO_ROOT/third_party/UMI_collection"

if [ ! -f "$UMI/run_collection_viz.sh" ]; then
  echo "[collect_data] ERROR: submodule not initialized at $UMI" >&2
  echo "[collect_data]   run: git submodule update --init third_party/UMI_collection" >&2
  exit 1
fi

case "$MODE" in
  1pt) _pucks="left" ;;
  2pt) _pucks="left, right" ;;
  3pt) _pucks="left, right, head" ;;
  *)   _pucks="left, right, head, left_foot, right_foot" ;;
esac
echo "[collect_data] launching UMI_collection MODE=$MODE (viz on http://localhost:8080)"
if [ "$MODE" = "1pt" ]; then
  echo "[collect_data]   LEFT hand only: tracker + camera + F/T + gripper"
else
  echo "[collect_data]   LEFT + RIGHT hands: tracker + camera + F/T + gripper EACH$({ [ "$MODE" = 3pt ] || [ "$MODE" = 5pt ]; } && echo "; CHEST: tracker only")$([ "$MODE" = 5pt ] && echo "; BOTH FEET: tracker only")"
  echo "[collect_data]   right-hand devices are optional — a missing one is ⚠️ on the status board and"
  echo "[collect_data]   records no frames. RIGHT_CAMERA/RIGHT_FT/RIGHT_GRIPPER=0 to skip one deliberately."
fi
echo "[collect_data]   VIVE: power the $_pucks puck(s) (solid green) in view of the base stations;"
echo "[collect_data]   libsurvive re-reads + re-solves the stations (${TRACKER_STARTUP_WAIT:-120}s budget)," \
     "MIN_STATIONS=${MIN_STATIONS:-2}${STATIONS:+, STATIONS=$STATIONS powered}."
cd "$UMI"
MODE="$MODE" exec bash run_collection_viz.sh "$@"
