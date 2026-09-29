#!/usr/bin/env bash
# Launch the ONLINE WHOLE-BODY PLANNER: diffusion_umi_wholebody -> the `dense` low level, direct.
#
# The deploy sibling of ./scripts/run_infer_s1_wholebody.sh (the offline page) and the
# completion-free twin of ./scripts/run_hl_serving_mc.sh:
#
#   run_hl_serving_mc.sh    3 EE poses --decode--> Motion_Completion --> qpos(29) --> :8770 dense
#   THIS SCRIPT             qpos(29) straight off the policy -------------------> :8770 dense
#
# The completion still ran — ONCE, offline, over the training zarr
# (third_party/UMI_collection/run_mc_to_hl_wholebody.sh). So there is no g1_mosyn conda env to
# find here, no backend child process, no ~30-60 s JIT warm-up, and no --mc-foot-ik trade-off.
# The LOW LEVEL is the same one either way (MODE=dense-unified-anneal, W&B 7vxmnwo6).
#
# ⚠️ T4's MODE must be a dense/wbt-FAMILY preset and nothing cross-checks it — the RecordTap
# frame carries no preset name. A mismatch is not dangerous (the low level's width guard drops
# every frame of the wrong shape and the robot holds its ready pose) but it IS silent, so
# "nothing moves" reads as "T4 is not a dense preset" before it reads as anything else.
#
# WHY A SEPARATE LAUNCHER (and a separate planner). This checkpoint's action is 43-D with an
# empty pose block: there is no keypoint to un-relativize, no anchor to decode against, no
# WireMapper, no TCP lever, no arm-IK preview. What IS shared lives in importable modules and is
# imported, not copied (hl_dense's JointClutch/DenseWire, hl_online's rings, live_robot's
# RecordTap parse, hl_serving's sensor plumbing) — see scripts/hl_serving_wholebody.py's header.
#
# PREREQUISITES — five terminals, in this order. The planner probes every one at startup and
# names the missing one rather than starting degraded.
#
#   T1  BOTH Botas, 6-DoF, :8771       cd .../force_deploy && bash 2_ft_bridge.sh --pipeline
#   T2  camera L :8773  AND  R :8777   this checkpoint reads TWO views; a missing one is fed to
#                                      its encoder group as a GREY frame and nothing downstream
#                                      would say so, which is why the planner refuses instead.
#   T3  both jaws                      LEFT=1 RIGHT=1 KEYBOARD=0 bash 5_dual_gripper_keyboard.sh
#                                      (LEFT can0 -> :8784, RIGHT can1 -> :8785)
#   T4  the LOW LEVEL, ABSOLUTE + RecordTap:
#         cd <repo>/third_party/holosoma_extensions/force_deploy && \
#           MODE=dense-unified-anneal HOLOSOMA_VIVE_RELATIVE=0 RECORD=1 FORCE_CMD_WIRE=1 \
#           bash 3_run_policy_vive_force.sh
#       * Pull the ONNX if it is not on disk:
#           ~/.holosoma_deps/miniconda3/envs/hssim/bin/python \
#             third_party/holosoma_extensions/scripts/_dl_wandb_onnx.py 7vxmnwo6
#       * HOLOSOMA_VIVE_RELATIVE=0 (ABSOLUTE) is REQUIRED: the planner publishes absolute joint
#         targets. RELATIVE would add them as a delta on top of the held pose.
#       * RECORD=1 is strongly recommended, not cosmetic: it is what lets the joint clutch
#         re-base on the robot's MEASURED joints each Predict. Without it the clutch runs
#         open-loop from the ready pose and a tracking shortfall accumulates across chunks.
#       * ⚠️ Terminal 1 of the VIVE bridge is NOT started here. The planner IS the producer on
#         :8770; running the tracker bridge as well means two writers on one port.
#   T5  this script.
#
# OPERATOR SEQUENCE, once all five are up: joystick **A** (the low level holds the READY POSE) ->
# Predict in the page -> joystick **start** to ARM -> Run. Same three presses as the other two.
#
# ⚠️ SAFETY — this commands the WHOLE BODY and the LEGS ARE ON THE WIRE (that is the preset, not
# a defect: a keypoint policy commands 3 points and lets the low level pick a posture, this one
# commands the hips, knees and ankles too). First runs: gantry-suspended, hand on the hardware
# E-stop, force switch OFF (--no-force-cmd), and LEGS=hold until the arms are trusted. The
# default joint clutch makes the first executed step a no-op on every joint, so the body starts
# from where it IS instead of jumping to the model's absolute pose — which sits in the dataset's
# cond_mode=home posture space, measured 0.514 rad (knee) from the ready crouch. Above 40 N of
# measured force the low level stops itself (FORCE_ESTOP_N, the joystick-B path).
#
# Usage:
#   ./scripts/run_hl_serving_wholebody.sh --checkpoint checkpoints/ckpt_0831_wb
#   LEGS=hold ./scripts/run_hl_serving_wholebody.sh --checkpoint <ckpt> --no-force-cmd
#   PORT=8091 ./scripts/run_hl_serving_wholebody.sh --checkpoint <ckpt> --post-anchor command
#
# Env knobs:
#   FUJI_PY   python (default: <repo>/.venv/bin/python)
#   PORT      viser port (default: 8091 — distinct from 8088/8090 so all three planners coexist)
#   DEVICE    torch device (default: cuda)
#   LEGS      predict|hold — a shorthand for --legs, so a first-run setting can be exported once
#   FUJI_ACTION_HORIZON  DO NOT SET IT — derived from the checkpoint, exactly as
#             run_hl_serving{,_mc}.sh do, because layout.py resolves it at IMPORT time.
set -eo pipefail

FUJI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FUJI_PY="${FUJI_PY:-$FUJI_ROOT/.venv/bin/python}"
PORT="${PORT:-8091}"
DEVICE="${DEVICE:-cuda}"

# ⚠️ Knobs that belong to the SIBLING launchers. Refused with a pointer rather than ignored: each
# is load-bearing over there, and an operator who sets one is entitled to assume it did something.
for _v in ANCHOR ANCHOR_EPS; do
  if [ -n "${!_v:-}" ]; then
    echo "[run] FATAL: $_v=${!_v} is a run_hl_serving.sh (left-only) knob and does nothing here." >&2
    echo "[run]   This path has NO decode anchor at all — the chunk is ABSOLUTE joint space." >&2
    echo "[run]   The equivalent is a FLAG on the published quantity:" >&2
    echo "[run]       --post-anchor command  [--post-anchor-eps-rad 0.15]" >&2
    exit 2
  fi
done
for _v in MOTION_COMPLETION_DIR MOTION_COMPLETION_PY MOTION_COMPLETION_CKPT; do
  if [ -n "${!_v:-}" ]; then
    echo "[run] NOTE: $_v is set but this planner runs NO motion completion." >&2
    echo "[run]   The completion ran once, offline, when the dataset was built" >&2
    echo "[run]   (third_party/UMI_collection/run_mc_to_hl_wholebody.sh). If you meant the" >&2
    echo "[run]   3-tracker deploy path, that is ./scripts/run_hl_serving_mc.sh." >&2
  fi
done

if [ ! -x "$FUJI_PY" ]; then
  echo "[run] FUJI venv python not found at $FUJI_PY (create .venv, then: uv pip install -e \".[viz]\")" >&2
  exit 2
fi

CKPT=""
# A MIRROR of the planner's force default, not a second source of truth: force on the wire needs
# BOTH sides (the planner's switch AND the low level's FORCE_CMD_WIRE=1) and neither alone does
# anything, so the T4 command echoed below has to carry the companion half — or the operator arms
# a switch that silently cannot fire. The planner starts that switch ON, so this defaults to ON
# too and --no-force-cmd is what clears it: the two sides are one decision, defaulted the same way.
LL_FORCE_ENV="FORCE_CMD_WIRE=1 "
_want=""
for _arg in "$@"; do
  case "$_want" in
    ckpt) CKPT="$_arg"; _want=""; continue ;;
  esac
  case "$_arg" in
    --checkpoint)   _want=ckpt ;;
    --checkpoint=*) CKPT="${_arg#--checkpoint=}" ;;
    --no-force-cmd) LL_FORCE_ENV="" ;;
  esac
done
if [ -z "$CKPT" ]; then
  echo "[run] FATAL: --checkpoint is required (a wholebody HL-v2 checkpoint dir)." >&2
  exit 2
fi
if [ ! -f "$CKPT/config.json" ]; then
  echo "[run] FATAL: --checkpoint '$CKPT' has no config.json" >&2
  exit 2
fi

# LAYOUT CHECK, HERE, BEFORE A GPU IS TOUCHED. The planner refuses a non-wholebody checkpoint too
# (by name, in main()), but that costs a model load and several GB of VRAM first — and the most
# likely mistake is reaching for this launcher with a 3-tracker checkpoint, which has its own
# working launcher two files over. Say so in one line instead.
CKPT_MODE="$("$FUJI_PY" -c \
  "import json,sys; print(json.load(open(sys.argv[1])).get('hl_data',{}).get('mode',''))" \
  "$CKPT/config.json")"
if [ "$CKPT_MODE" != "wholebody" ]; then
  echo "[run] FATAL: $CKPT has hl_data.mode='$CKPT_MODE', not 'wholebody'." >&2
  echo "[run]   This planner publishes the policy's OWN 29 joint angles. A keypoint checkpoint" >&2
  echo "[run]   predicts poses instead:" >&2
  echo "[run]     left_only / both_hands        -> ./scripts/run_hl_serving.sh" >&2
  echo "[run]     both_hands_head[_dual]        -> ./scripts/run_hl_serving_mc.sh (+ completion)" >&2
  exit 2
fi

# ACTION-CHUNK PIN, DERIVED FROM THE CHECKPOINT — same contract, same reasoning and the same
# helper shape as run_hl_serving{,_mc}.sh: fuji/s1_policy/data/layout.py resolves ACTION_HORIZON
# from FUJI_ACTION_HORIZON at IMPORT time, so a shell launcher is the only place that can still
# set it. It must come from the ACTIVE algorithm's block (the trainer leaves the other two at
# their dataclass defaults), which is why config.json's own `algo` picks the block.
if ! CKPT_H="$("$FUJI_PY" - "$CKPT/config.json" <<'PY'
import json, sys
# Imported, not duplicated: a second copy of the algo->block map here would drift silently.
from fuji.s1_policy.training.config import ALGO_CONFIG_FIELD

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
if [ -n "${FUJI_ACTION_HORIZON:-}" ] && [ "$FUJI_ACTION_HORIZON" != "$CKPT_H" ]; then
  echo "[run] FATAL: FUJI_ACTION_HORIZON=$FUJI_ACTION_HORIZON in the environment, but" >&2
  echo "[run]        $CKPT declares action_horizon=$CKPT_H. Refusing to guess which is meant." >&2
  exit 2
fi
export FUJI_ACTION_HORIZON="$CKPT_H"

# The ALGO defaults to diffusion in the planner, but the checkpoint states its own — pass it
# through so a flow/act wholebody checkpoint does not need the flag spelled twice.
CKPT_ALGO="$("$FUJI_PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('algo',''))" \
  "$CKPT/config.json")"
EXTRA=()
case " $* " in *" --algo "*) ;; *) [ -n "$CKPT_ALGO" ] && EXTRA=(--algo "$CKPT_ALGO") ;; esac
# LEGS is a shorthand, not a second source of truth: an explicit --legs on the command line wins.
case " $* " in
  *" --legs "*|*" --legs="*) ;;
  *) [ -n "${LEGS:-}" ] && EXTRA+=(--legs "$LEGS") ;;
esac

echo "[run] WHOLE-BODY PLANNER (29 joints -> dense, NO completion) on http://localhost:$PORT"
echo "[run]   python: $FUJI_PY (device=$DEVICE)"
echo "[run]   checkpoint: $CKPT  algo=${CKPT_ALGO:-?}  layout=$CKPT_MODE"
echo "[run]              chunk=$FUJI_ACTION_HORIZON steps @ 30 Hz  ->  29 joints + 2 grips + 2 wrenches"
echo "[run]   legs: ${LEGS:-predict}   (LEGS=hold pins the 12 leg joints to the clutch base)"
echo "[run]   COMPANION LOW LEVEL — dense, ABSOLUTE, RecordTap on (W&B run 7vxmnwo6):"
echo "[run]     cd $FUJI_ROOT/third_party/holosoma_extensions/force_deploy && \\"
echo "[run]       MODE=dense-unified-anneal HOLOSOMA_VIVE_RELATIVE=0 RECORD=1 ${LL_FORCE_ENV}\\"
echo "[run]       bash 3_run_policy_vive_force.sh"
if [ -z "$LL_FORCE_ENV" ]; then
  echo "[run]     (--no-force-cmd given: pure position. Drop it, and add FORCE_CMD_WIRE=1 there,"
  echo "[run]      to command force at all.)"
fi
echo "[run]   ⚠️  T4's MODE must be a dense/wbt-FAMILY preset and nothing cross-checks it: a"
echo "[run]     mismatch makes the low level drop every frame, so 'nothing moves' means 'these"
echo "[run]     two disagree' before it means anything else."
echo "[run]   ⚠️  Do NOT also start 1_vive_bridge.sh: this planner is the :8770 producer."
echo "[run]   ⚠️  BOTH cameras are required (L :8773, R :8777) — the checkpoint reads two views."
echo "[run]   PROPRIO is the last COMMANDED target in the MODEL's posture space; the FIRST"
echo "[run]     Predict is seeded from the training zarr's own episode starts (the log names the"
echo "[run]     file and the spread). --state-source measured is available and 0.514 rad OOD."
if [ -n "$LL_FORCE_ENV" ]; then
  echo "[run]   FORCE + BOTH JAWS START ENABLED. The page's switches are the live control; this"
  echo "[run]     only says what they read at launch. Force is hard-zeroed below 1.5 N either way"
  echo "[run]     and still needs FORCE_CMD_WIRE=1 there — neither side alone commands force."
else
  echo "[run]   FORCE starts OFF (--no-force-cmd); BOTH JAWS start ENABLED. The page's switches"
  echo "[run]     are the live control, so force can still be turned on there — but only if the"
  echo "[run]     low level also got FORCE_CMD_WIRE=1."
fi
echo "[run]   The low level STOPS ITSELF above FORCE_ESTOP_N=40 N of measured force."
echo "[run]   Arm ONCE, then '▶▶ Run to completion' keeps predicting and executing until ⏹ /"
echo "[run]     STOP / the --auto-max-seconds ceiling (20 s) / the low level leaving :8770."
echo "[run]   ⚠️  Gantry-suspended for first runs: the LEGS are on this wire."

exec "$FUJI_PY" "$FUJI_ROOT/scripts/hl_serving_wholebody.py" \
  --device "$DEVICE" \
  --port "$PORT" \
  "${EXTRA[@]}" \
  "$@"
