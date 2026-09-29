#!/usr/bin/env bash
# Launch the ONLINE 3-TRACKER PLANNER: diffusion_umi_3tracker -> MOTION COMPLETION -> a
# WHOLE-BODY low level. The deploy sibling of ./scripts/run_infer_s1_mc.sh (the offline viewer)
# and of ./scripts/run_hl_serving.sh (the left_only / 3-point planner).
#
#   3 predicted EE poses --decode against the READY POSE--> Motion_Completion --> qpos(29)+FK
#        --low-level 5pt   --> :8770 {"mode":"5pt"}   --> g1-…-5pt-hybrid-unified-anneal-wbt
#        --low-level dense --> :8770 {"mode":"dense"} --> g1-…-dense-hybrid-unified-anneal-wbt
#
# ⚠️ --low-level MUST match the MODE the low level (T4) was launched with. Nothing checks the
# two against each other — the RecordTap frame carries no preset name — and a mismatch is not
# dangerous but IS silent: the low level's width guard drops every frame and the robot holds
# its ready pose, so "nothing moves" reads as "the two MODEs disagree" first.
#
# WHY A SEPARATE LAUNCHER (and a separate planner). A dense low level has NO keypoint front-end,
# so none of run_hl_serving.sh's wire applies: no vr_3point slot, no wrist->contact lever, no
# per-role hold, no arm IK preview. What IS shared lives in importable modules and is imported,
# not copied — see scripts/hl_serving_mc.py's header. Serving a left_only checkpoint to a 3pt
# low level is still run_hl_serving.sh's job and is untouched by this file.
#
# PREREQUISITES — five terminals, in this order. The planner probes every one of them at
# startup and names the missing one rather than starting degraded.
#
#   T1  BOTH Botas, 6-DoF, :8771       cd .../force_deploy && bash 2_ft_bridge.sh --pipeline
#   T2  camera L :8773  AND  R :8777   this checkpoint reads TWO views; a missing one is fed to
#                                      its encoder group as a GREY frame and nothing downstream
#                                      would say so, which is why the planner refuses instead.
#   T3  both jaws                      LEFT=1 RIGHT=1 KEYBOARD=0 bash 5_dual_gripper_keyboard.sh
#                                      (LEFT can0 -> :8784, RIGHT can1 -> :8785)
#   T4  the LOW LEVEL, ABSOLUTE + RecordTap, in the family you chose:
#         cd <repo>/third_party/holosoma_extensions/force_deploy && \
#           MODE=5pt-unified-anneal HOLOSOMA_VIVE_RELATIVE=0 RECORD=1 \
#           bash 3_run_policy_vive_force.sh          # or MODE=dense-unified-anneal
#       * MODE=5pt-unified-anneal picks W&B run pq9shxs7 (the current 5pt fine-tune, ft60000-r2; steps
#         from 23000); MODE=dense-unified-anneal picks 7vxmnwo6. Pull whichever is not on disk:
#           ~/.holosoma_deps/miniconda3/envs/hssim/bin/python \
#             third_party/holosoma_extensions/scripts/_dl_wandb_onnx.py pq9shxs7
#       * HOLOSOMA_VIVE_RELATIVE=0 (ABSOLUTE) is REQUIRED: the planner publishes absolute joint
#         targets. RELATIVE mode would add them as a delta on top of the held pose.
#       * RECORD=1 is strongly recommended, not cosmetic: it is what lets the joint clutch
#         re-base on the robot's MEASURED joints each Predict. Without it the clutch runs
#         open-loop from the ready pose and a tracking shortfall accumulates across chunks.
#       * ⚠️  Terminal 1 of the VIVE bridge is NOT started here. The planner IS the producer on
#         :8770; running the tracker bridge as well means two writers on one port.
#   T5  this script.
#
# OPERATOR SEQUENCE, once all five are up: joystick **A** (the low level holds the READY POSE —
# the pose this planner decodes against) -> Predict in the page -> joystick **start** to ARM ->
# Run. Same three presses as the 3pt planner.
#
# ⚠️ SAFETY — this commands the WHOLE BODY, legs included, and the completion's own posture is
# not the ready pose's crouch. First runs: gantry-suspended, hand on the hardware E-stop, force
# switch OFF (the default). The default --dense-clutch joint makes the first executed step a
# no-op on every joint, so the body starts from where it is instead of jumping to the
# completion's absolute pose (measured: 0.40 rad of arm null-space offset, 0.54 rad of knee).
# Above 40 N of measured force the low level stops itself (FORCE_ESTOP_N, the joystick-B path).
#
# Usage:
#   PORT=8090 ./scripts/run_hl_serving_mc.sh --low-level 5pt   --checkpoint ckpt_0831_final
#   PORT=8090 ./scripts/run_hl_serving_mc.sh --low-level dense --checkpoint ckpt_0831_final
#   PORT=8090 ./scripts/run_hl_serving_mc.sh --checkpoint ckpt_0831_final --dense-clutch none
#
# Env knobs:
#   FUJI_PY   python (default: <repo>/.venv/bin/python)
#   PORT      viser port (default: 8090 — NOT 8088, so this and the 3pt planner can coexist)
#   DEVICE    torch device (default: cuda)
#   FUJI_ACTION_HORIZON  DO NOT SET IT — derived from the checkpoint, exactly as
#             run_hl_serving.sh does, because layout.py resolves it at IMPORT time.
#   MOTION_COMPLETION_{DIR,PY,CKPT,STEP}  override the completion's checkout / conda
#             interpreter / checkpoint (see motion_completion/mc_backend_client.py). All four
#             are reported at startup; a missing one is a REFUSAL here rather than a disabled
#             panel, because a dense low level with no wire holds its ready pose forever and
#             that looks exactly like a hung robot.
set -eo pipefail

FUJI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FUJI_PY="${FUJI_PY:-$FUJI_ROOT/.venv/bin/python}"
UMI="$FUJI_ROOT/third_party/UMI_collection"
PORT="${PORT:-8090}"
DEVICE="${DEVICE:-cuda}"

# ⚠️ ANCHOR belongs to run_hl_serving.sh (the LEFT-ONLY twin). The same IDEA exists here, but it
# is a different flag applied at a different place, so the env var is REFUSED with a pointer
# rather than ignored: it is load-bearing over there, and an operator who sets it is entitled to
# assume it did something.
#
# WHY IT MOVED. The decode anchor here is also motion completion's ROOT conditioning, so moving
# it changes the POSTURE the completion answers with, not just where the hands go — and the low
# level's pre-arm gate references FK of that one fixed pose. The integrator therefore sits AFTER
# the completion, on the quantity actually published: `--post-anchor command` carries the last
# EXECUTED step forward in JOINT space (dense) or per KEYPOINT (5pt).
for _v in ANCHOR ANCHOR_EPS; do
  if [ -n "${!_v:-}" ]; then
    echo "[run] FATAL: $_v=${!_v} is a run_hl_serving.sh (left-only) knob and does nothing here." >&2
    echo "[run]   The equivalent on this path is a FLAG, applied AFTER motion completion:" >&2
    echo "[run]       --post-anchor command   [--post-anchor-eps-rad 0.15 | --post-anchor-eps-m 0.03]" >&2
    echo "[run]   It cannot be applied to the DECODE anchor here: that anchor is also the" >&2
    echo "[run]   completion's root conditioning, so moving it changes the POSTURE, and the low" >&2
    echo "[run]   level gates against FK of exactly that one fixed pose." >&2
    echo "[run]   Unset $_v and pass --post-anchor command instead." >&2
    exit 2
  fi
done

if [ ! -x "$FUJI_PY" ]; then
  echo "[run] FUJI venv python not found at $FUJI_PY (create .venv, then: uv pip install -e \".[viz]\")" >&2
  exit 2
fi

# ACTION-CHUNK PIN, DERIVED FROM THE CHECKPOINT — same contract, same reasoning and the same
# shared helper as run_hl_serving.sh: fuji/s1_policy/data/layout.py resolves ACTION_HORIZON from
# FUJI_ACTION_HORIZON at IMPORT time, so a shell launcher is the only place that can still set
# it. It must come from the ACTIVE algorithm's block (the trainer leaves the other two at their
# dataclass defaults), which is why config.json's own `algo` picks the block.
CKPT=""
# Scanned for ONE reason: so the companion-low-level command this script echoes names the MODE
# that actually matches. The planner owns the flag and its default; this is a mirror, not a
# second source of truth, which is why an unrecognised value is passed through untouched and
# left for argparse to reject.
LOW_LEVEL="dense"
# Same mirror reasoning for the force switch: force on the wire needs BOTH sides (the planner's
# switch AND the low level's FORCE_CMD_WIRE=1) and neither alone does anything, so the echoed T4
# command has to carry the companion half or the operator arms a switch that silently cannot fire.
# The planner now starts with that switch ON, so this defaults to ON too and --no-force-cmd is
# what clears it — the two sides are one decision and they default the same way.
LL_FORCE_ENV="FORCE_CMD_WIRE=1 "
_want=""
for _arg in "$@"; do
  case "$_want" in
    ckpt) CKPT="$_arg"; _want=""; continue ;;
    low)  LOW_LEVEL="$_arg"; _want=""; continue ;;
  esac
  case "$_arg" in
    --checkpoint)      _want=ckpt ;;
    --checkpoint=*)    CKPT="${_arg#--checkpoint=}" ;;
    --low-level)       _want=low ;;
    --low-level=*)     LOW_LEVEL="${_arg#--low-level=}" ;;
    --no-force-cmd)    LL_FORCE_ENV="" ;;
  esac
done
case "$LOW_LEVEL" in
  5pt)   LL_MODE="5pt-unified-anneal";   LL_RUN="pq9shxs7"; LL_WIRE='{"mode":"5pt"} 5 keypoints' ;;
  dense) LL_MODE="dense-unified-anneal"; LL_RUN="7vxmnwo6"; LL_WIRE='{"mode":"dense"} 29 joints' ;;
  *)     LL_MODE="<see --help>";         LL_RUN="?";        LL_WIRE="?" ;;
esac
if [ -z "$CKPT" ]; then
  echo "[run] FATAL: --checkpoint is required (a 3-tracker HL-v2 checkpoint dir)." >&2
  exit 2
fi
if [ ! -f "$CKPT/config.json" ]; then
  echo "[run] FATAL: --checkpoint '$CKPT' has no config.json" >&2
  exit 2
fi
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
# through so a flow/act 3-tracker checkpoint does not need the flag spelled twice.
CKPT_ALGO="$("$FUJI_PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('algo',''))" \
  "$CKPT/config.json")"
EXTRA=()
case " $* " in *" --algo "*) ;; *) [ -n "$CKPT_ALGO" ] && EXTRA=(--algo "$CKPT_ALGO") ;; esac

echo "[run] 3-TRACKER PLANNER (motion completion -> ${LOW_LEVEL^^}) on http://localhost:$PORT"
echo "[run]   python: $FUJI_PY (device=$DEVICE)"
echo "[run]   checkpoint: $CKPT  algo=${CKPT_ALGO:-?}  chunk=$FUJI_ACTION_HORIZON steps @ 30 Hz"
echo "[run]   wire: $LL_WIRE   (--low-level $LOW_LEVEL)"
echo "[run]   the completion backend runs in its OWN conda interpreter (g1_mosyn) as a child"
echo "[run]     process and takes ~30-60 s to become ready (the ckpt + the JAX wrist-IK JIT)."
echo "[run]   COMPANION LOW LEVEL — $LOW_LEVEL, ABSOLUTE, RecordTap on (W&B run $LL_RUN):"
echo "[run]     cd $FUJI_ROOT/third_party/holosoma_extensions/force_deploy && \\"
echo "[run]       MODE=$LL_MODE HOLOSOMA_VIVE_RELATIVE=0 RECORD=1 ${LL_FORCE_ENV}\\"
echo "[run]       bash 3_run_policy_vive_force.sh"
if [ -z "$LL_FORCE_ENV" ]; then
  echo "[run]     (--no-force-cmd given: pure position. Drop it, and add FORCE_CMD_WIRE=1 there,"
  echo "[run]      to command force at all.)"
fi
echo "[run]   ⚠️  --low-level and that MODE are ONE decision in two places and nothing checks"
echo "[run]     them against each other: a mismatch makes the low level drop every frame, so"
echo "[run]     'nothing moves' means 'these two disagree' before it means anything else."
echo "[run]   ⚠️  Do NOT also start 1_vive_bridge.sh: this planner is the :8770 producer."
echo "[run]   ⚠️  BOTH cameras are required (L :8773, R :8777) — the checkpoint reads two views."
echo "[run]   FORCE + BOTH JAWS START ENABLED. The page's switches are still the live control;"
echo "[run]     this only says what they read at launch (--no-force-cmd / --no-command-gripper"
echo "[run]     start them off). Force is hard-zeroed below --force-gate-n 1.5 N either way, and"
echo "[run]     still needs FORCE_CMD_WIRE=1 there — neither side alone commands force."
echo "[run]   The low level STOPS ITSELF above FORCE_ESTOP_N=40 N of measured force."
echo "[run]   Arm ONCE, then '▶▶ Run to completion' keeps predicting and executing until ⏹ /"
echo "[run]     STOP / the --auto-max-seconds ceiling (20 s) / the low level leaving :8770."
echo "[run]     'Run staged chunk' is still there for one chunk at a time."
echo "[run]   ⚠️  Gantry-suspended for first runs: the LEGS are on this wire too."

PYTHONPATH="$UMI/third_party/humi_ik:$UMI/viz:$UMI/replay:${PYTHONPATH:-}" "$FUJI_PY" \
  "$FUJI_ROOT/scripts/hl_serving_mc.py" \
  --device "$DEVICE" \
  --port "$PORT" \
  "${EXTRA[@]}" \
  "$@"
