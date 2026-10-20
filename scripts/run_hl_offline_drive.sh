#!/usr/bin/env bash
# Launch the OFFLINE-SAMPLE DRIVER: a dataset anchor -> one chunk -> a REAL low level, and every
# control lives in the browser page.
#
# The third corner of a triangle the repo already had two of:
#
#   ./scripts/run_infer_s1_wholebody.sh    sample -> chunk -> a viser DRAWING     (no controller)
#   ./scripts/run_hl_serving_wholebody.sh  sensors -> chunk -> :8770              (no ground truth)
#   THIS SCRIPT                            sample -> chunk -> :8770, whose other end is the
#                                          mjswan MuJoCo/WASM low level in a browser tab.
#
# So it answers what neither of those can — what does the whole-body controller actually DO with
# a chunk we already know is good? — with NO RIG AT ALL: no trackers, no Botas, no D405s, no
# gantry, no G1. Two terminals and a browser.
#
# ⚠️ OPEN LOOP, BY CONSTRUCTION. The observation is the DATASET's, so the chunk cannot react to
# what the controller did with the last one. `▶▶ auto` advances the cursor by the executed steps
# and re-infers, which is an open-loop REHEARSAL of a recorded observation sequence: the right
# test for "does the low level track this", the wrong one for "does the policy recover".
#
# TWO TERMINALS
# =============
#   T1  this script (binds :8770 and loads the policy):
#         ./scripts/run_hl_offline_drive.sh --checkpoint <ckpt>
#
#   T2  the low level, in the browser. WHICH ONE depends on the checkpoint's layout, and T1
#       prints the exact line for you at startup:
#         wholebody             bash third_party/holosoma_extensions/scripts/_vive_teleop5_build.sh --mode dense
#         both_hands_head_dual  bash third_party/holosoma_extensions/scripts/_vive_teleop3_build.sh
#       Then open the URL it serves. It connects to :8770 by itself, and the FLoRe panel appears
#       in the page the moment it does.
#
#   ⚠️ Do NOT also start 1_vive_bridge.sh — this script IS the :8770 producer, and two writers on
#      one port is a wire that alternates between two intents.
#
#   ⚠️ The browser build needs the COOP/COEP cross-origin-isolation headers for SharedArrayBuffer
#      (`python -m wbt_realtime_eval.serve_bundle`, which those scripts use). A plain
#      `http.server` loads and then hangs in the WASM init.
#
# ON THE PAGE: pick an anchor (or an episode) -> Predict -> Run.
#   * Predict STAGES and publishes NOTHING. For a wholebody checkpoint the LIGHT-PURPLE
#     translucent G1 then plays the staged chunk on a loop, on a wire channel separate from the
#     command — so previewing is not publishing. A 3pt checkpoint has no body to draw (a keypoint
#     policy predicts three poses and no posture); its panel reports the chunk numerically and the
#     page's own commanded triads move when it RUNS.
#   * Run publishes the first N steps, one per 30 Hz tick.
#   * Reset returns to the hold pose; switching sample or episode resets automatically.
#   * Stop interrupts within one step and is never dimmed.
#
# THE HOLD POSE (--hold / HOLD): 'sample' (default) RAMPS the wire to the sample's own step-0
# posture before predicting, so the robot stands where the demonstration stood. That is a real
# move here — measured 0.86 rad at the shoulder on session_2026.09.02_merged3 — so it is ramped at
# RAMP rad/s and the clutch tracks the wire the whole way. 'ready' keeps the G1 ready pose and
# clutches the chunk onto it, which is exactly what the rig deploy does; use it when the question
# is "would this be safe on the robot".
#
# NO BROWSER? `./scripts/smoke_test_hl_offline.py --spawn --checkpoint <ckpt>` drives the same
# protocol from a 20-line WebSocket client and asserts on the numbers. That is the verification
# path — it needs no display, no WASM and no ONNX.
#
# Usage:
#   ./scripts/run_hl_offline_drive.sh --checkpoint checkpoints/.../step_00100000
#   HOLD=ready ./scripts/run_hl_offline_drive.sh --checkpoint <ckpt> --episode 12
#   PORT=8771 ./scripts/run_hl_offline_drive.sh --checkpoint <ckpt> --exec-steps 30
#
# Env knobs:
#   FLORE_PY  python (default: <repo>/.venv/bin/python)
#   PORT     the :8770 wire port (default 8770 — the browser build's own default)
#   DEVICE   torch device (default cuda)
#   HOLD     sample|ready — shorthand for --hold
#   RAMP     rad/s for the hold ramp (default: the script's 0.5)
#   ANCHOR   command|off — where the NEXT chunk continues from. Same spelling and the same
#            decision as run_hl_serving.sh's ANCHOR; maps onto --post-anchor. 'measured' is
#            accepted as a spelling of 'off' (there is no RecordTap behind a browser sim, so
#            there is nothing measured to anchor ON).
#   GRIPPER  1|0 — shorthand for --command-gripper. OFF by default, and NOT on the :8770 wire:
#            the jaws are reached over the gripper bridge's own socket (:8784 / :8785), so this
#            drives REAL CAN hardware even while the low level is a browser sim.
#   FLORE_ACTION_HORIZON  DO NOT SET IT — derived from the checkpoint, as every other launcher
#            does, because flore/s1_policy/data/layout.py resolves it at IMPORT time.
set -eo pipefail

FLORE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLORE_PY="${FLORE_PY:-$FLORE_ROOT/.venv/bin/python}"
PORT="${PORT:-8770}"
DEVICE="${DEVICE:-cuda}"

if [ ! -x "$FLORE_PY" ]; then
  echo "[run] FLoRe venv python not found at $FLORE_PY (create .venv, then: uv sync --locked)" >&2
  exit 2
fi

CKPT=""
_want=""
for _arg in "$@"; do
  case "$_want" in
    ckpt) CKPT="$_arg"; _want=""; continue ;;
  esac
  case "$_arg" in
    --checkpoint)   _want=ckpt ;;
    --checkpoint=*) CKPT="${_arg#--checkpoint=}" ;;
  esac
done
if [ -z "$CKPT" ]; then
  echo "[run] FATAL: --checkpoint is required (an HL-v2 checkpoint dir)." >&2
  exit 2
fi
if [ ! -f "$CKPT/config.json" ]; then
  echo "[run] FATAL: --checkpoint '$CKPT' has no config.json" >&2
  exit 2
fi

# LAYOUT -> FAMILY, HERE, BEFORE A GPU IS TOUCHED. The driver derives and refuses the same thing
# by name, but that costs a model load and several GB of VRAM first; the likely mistake is
# reaching for this launcher with a checkpoint whose family has no browser build.
read -r CKPT_MODE CKPT_ALGO <<EOF
$("$FLORE_PY" -c "import json,sys; c=json.load(open(sys.argv[1])); print((c.get('hl_data') or {}).get('mode',''), c.get('algo',''))" "$CKPT/config.json")
EOF
case "$CKPT_MODE" in
  wholebody)                        FAMILY=dense; BROWSER="scripts/_vive_teleop5_build.sh --mode dense" ;;
  both_hands_head|both_hands_head_dual)
                                    FAMILY=3pt;   BROWSER="scripts/_vive_teleop3_build.sh" ;;
  *)
    echo "[run] FATAL: $CKPT has hl_data.mode='$CKPT_MODE', which this driver has no low-level" >&2
    echo "[run]   family for. It drives two:" >&2
    echo "[run]     wholebody                    -> the dense browser low level" >&2
    echo "[run]     both_hands_head[_dual]       -> the 3pt browser low level" >&2
    echo "[run]   left_only / both_hands predict a SUBSET of the three points and belong on" >&2
    echo "[run]   ./scripts/run_hl_serving.sh, which HOLDS the ones they do not predict." >&2
    echo "[run]   wholebody_geom_* are not served anywhere yet (docs/RUNBOOK_wb_geom_ablation.md)." >&2
    exit 2 ;;
esac

# ACTION-CHUNK PIN, DERIVED FROM THE CHECKPOINT — same contract and the same helper shape as
# run_hl_serving{,_mc,_wholebody}.sh: layout.py resolves ACTION_HORIZON from FLORE_ACTION_HORIZON at
# IMPORT time, so a shell launcher is the only place that can still set it. It must come from the
# ACTIVE algorithm's block (the trainer leaves the other two at their dataclass defaults).
if ! CKPT_H="$("$FLORE_PY" - "$CKPT/config.json" <<'PY'
import json, sys
# Imported, not duplicated: a second copy of the algo->block map here would drift silently.
from flore.s1_policy.training.config import ALGO_CONFIG_FIELD

raw = json.load(open(sys.argv[1]))
algo = raw.get("algo")
field = ALGO_CONFIG_FIELD.get(algo)
if field is None:
    sys.exit(f"config.json algo={algo!r} is not one of {sorted(ALGO_CONFIG_FIELD)}")
block = raw.get(field)
if not isinstance(block, dict):
    sys.exit(f"config.json has no {field!r} block for algo={algo!r}")
horizon = block.get("action_horizon")
if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
    sys.exit(f"config.json {field}.action_horizon={horizon!r} is not a positive integer")
print(horizon)
PY
)"; then
  echo "[run] FATAL: could not read the action horizon from $CKPT/config.json (above)" >&2
  exit 2
fi
if [ -n "${FLORE_ACTION_HORIZON:-}" ] && [ "$FLORE_ACTION_HORIZON" != "$CKPT_H" ]; then
  echo "[run] FATAL: FLORE_ACTION_HORIZON=$FLORE_ACTION_HORIZON in the environment, but" >&2
  echo "[run]        $CKPT declares action_horizon=$CKPT_H. Refusing to guess which is meant." >&2
  exit 2
fi
export FLORE_ACTION_HORIZON="$CKPT_H"

EXTRA=()
case " $* " in *" --algo "*) ;; *) [ -n "$CKPT_ALGO" ] && EXTRA=(--algo "$CKPT_ALGO") ;; esac
# HOLD / RAMP are shorthands, not second sources of truth: an explicit flag wins.
case " $* " in
  *" --hold "*|*" --hold="*) ;;
  *) [ -n "${HOLD:-}" ] && EXTRA+=(--hold "$HOLD") ;;
esac
case " $* " in
  *" --ramp-rad-s "*|*" --ramp-rad-s="*) ;;
  *) [ -n "${RAMP:-}" ] && EXTRA+=(--ramp-rad-s "$RAMP") ;;
esac
# ANCHOR is spelled the way run_hl_serving.sh spells it, because it is the same DECISION: does the
# next chunk continue from the last EXECUTED COMMAND, or re-read the hold pose? It maps onto
# --post-anchor (the flag hl_serving_mc.py / hl_serving_wholebody.py use), and the two vocabularies
# line up: measured|off -> off, command -> command. There is no "measured" branch on this path at
# all (no RecordTap behind a browser sim), which is why `measured` is accepted as a spelling of
# `off` rather than silently doing something else.
case " $* " in
  *" --post-anchor "*|*" --post-anchor="*) ;;
  *)
    case "${ANCHOR:-}" in
      "")               ;;
      command)          EXTRA+=(--post-anchor command) ;;
      off|measured)     EXTRA+=(--post-anchor off) ;;
      *) echo "[run] FATAL: ANCHOR must be 'command' or 'off' (a.k.a. 'measured'), got '$ANCHOR'." >&2
         exit 2 ;;
    esac ;;
esac
# GRIPPER=1 turns the jaws on. It is OFF by default and it is NOT the :8770 wire -- see _Jaws in
# scripts/hl_offline_drive.py: the wire carries no gripper field in either family, so this reaches
# real CAN hardware even while the low level is a browser sim.
case " $* " in
  *" --command-gripper "*) ;;
  *) case "${GRIPPER:-}" in
       1|on|yes|true) EXTRA+=(--command-gripper) ;;
       ""|0|off|no|false) ;;
       *) echo "[run] FATAL: GRIPPER must be 1/on or 0/off, got '$GRIPPER'." >&2; exit 2 ;;
     esac ;;
esac

echo "[run] OFFLINE-SAMPLE DRIVER — a dataset anchor, through the policy, into a REAL low level."
echo "[run]   python: $FLORE_PY (device=$DEVICE)"
echo "[run]   checkpoint: $CKPT  algo=${CKPT_ALGO:-?}  layout=$CKPT_MODE  ->  $FAMILY wire"
echo "[run]              chunk=$FLORE_ACTION_HORIZON steps @ 30 Hz"
echo "[run]   wire: ws://127.0.0.1:$PORT  (this process is the SERVER; the browser connects to it)"
echo "[run]   hold pose: ${HOLD:-sample}  (sample = ramp to the demo's own step-0 posture first)"
echo "[run]   post-anchor: ${ANCHOR:-off}   jaws: ${GRIPPER:-0}  (1 = --command-gripper, real CAN)"
echo "[run]"
if [ "${ANCHOR:-}" = "command" ]; then
  echo "[run]   ⚠️  ANCHOR=command: each chunk's deltas are added to the LAST EXECUTED COMMANDED"
  echo "[run]      step, never to a measured joint angle. On the $FAMILY wire that is"
  case "$FAMILY" in
    3pt)   echo "[run]      KeypointClutch: p_pub = p_base + (p[i]-p[0]), R_pub = (R_base R[0]^T) R[i]" ;
           echo "[run]      -- one CONSTANT pelvis-frame offset per keypoint, all three carried." ;;
    dense) echo "[run]      JointClutch: q_pub = base + (q[i]-q[0]); needs --wb-clutch joint." ;;
  esac
  echo "[run]      Step 0 is a bit-exact no-op, and the base advances only on EXECUTED steps."
fi
if [ "${GRIPPER:-0}" != "0" ] && [ -n "${GRIPPER:-}" ]; then
  echo "[run]   ⚠️  GRIPPER=1: the jaws are driven over :8784 / :8785, NOT over :8770. Start the"
  echo "[run]      bridge first, or the target_frac frames are dropped silently:"
  echo "[run]        bash third_party/holosoma_extensions/force_deploy/5_dual_gripper_keyboard.sh"
fi
echo "[run]   T2 — THE LOW LEVEL, IN A BROWSER:"
echo "[run]     bash $FLORE_ROOT/third_party/holosoma_extensions/$BROWSER"
echo "[run]   then open the URL it prints. The FLoRe panel appears in the page on connect."
echo "[run]   ⚠️  Do NOT also start 1_vive_bridge.sh — this process is the :8770 producer."
echo "[run]   ⚠️  OPEN LOOP: the observation is the DATASET's, not the robot's. '▶▶ auto' walks"
echo "[run]     the cursor forward and re-infers; it is a rehearsal, not a closed loop."
echo "[run]   No browser? ./scripts/smoke_test_hl_offline.py --spawn --checkpoint $CKPT"
echo "[run]"

exec "$FLORE_PY" "$FLORE_ROOT/scripts/hl_offline_drive.py" \
  --device "$DEVICE" \
  --s0-ws-port "$PORT" \
  "${EXTRA[@]}" \
  "$@"
