#!/usr/bin/env bash
# OFFLINE inference viewer + the G1 WHOLE BODY, completed by Motion_Completion.
#
# The same deterministic offline rollout `scripts/infer_s1_{diffusion,flow,act}.py` has always
# drawn — blue masked target, orange prediction, green recorded tail, per-keypoint yam gripper
# clusters, force arrows, charts, Metrics — PLUS a 29-DoF G1 humanoid produced by the
# Motion_Completion model from the SAME three predicted keypoints. That humanoid is NOT an arm
# IK: its legs, torso and pelvis are what a low-level controller would be handed, so the page
# answers "what does the ROBOT do with this chunk", not just "where does the hand go".
#
#   ./scripts/run_infer_s1_mc.sh ckpt_0831_best
#   ./scripts/run_infer_s1_mc.sh ckpt_0831_best --sample-index 21000 --mc-series target
#   ALGO=act PORT=8090 ./scripts/run_infer_s1_mc.sh checkpoints/ckpt_act_20k
#
# It needs a 3-TRACKER checkpoint (both hands AND the chest: the chest drives the model's root
# channel), so `both_hands_head` or `both_hands_head_dual` — e.g. diffusion_umi_3tracker.
# A left-only checkpoint disables the panel by name instead of drawing a fabricated root.
#
# WITHOUT this launcher nothing changes: `infer_s1_diffusion.py` with no --motion-completion is
# byte-for-byte the viewer it has always been (no humanoid, no G1 URDF load, no model
# subprocess, no panel). This script is only the flag plus the paths the flag needs.
#
# PREREQUISITES, all of them the SAME ones the zarr replay's --motion-completion uses (nothing
# extra is installed for FLoRe — the model runs in ITS env, this process never imports torch 2.8):
#   1. the Motion_Completion peer checkout + its g1_mosyn conda env + its checkpoint:
#        cd third_party/Motion_Completion
#        bash scripts/setup_env.sh
#        bash scripts/download_ckpt.sh
#   2. this repo's viz extra (viser), for any of the offline viewers:
#        uv pip install --python .venv/bin/python -e ".[viz]"
# A missing piece DISABLES the panel with a printed reason; the rollout viewer still comes up.
# Overrides, all reported when they fail: MOTION_COMPLETION_{DIR,CKPT,PY,STEP}.
#
# TWO PROCESSES, same split as `--algo muse`: the model lives only in g1_mosyn (torch 2.8 +
# pytorch3d + JAX, no zarr) and this viewer only in FLoRe's .venv, so the overlay spawns
# `motion_completion/mc_infer_server.py` under the conda interpreter and speaks a base64-npz
# line protocol over its pipe. NEVER install g1_mosyn's deps into .venv.
#
# WHERE THE CHUNK IS PLACED — the one thing this launcher does besides pass the flag.
# --motion-completion implies `--display-frame g1-ready-pose`: the chunk is anchor-relative, so
# the pose it decodes against decides where the hands are, and on the rig that pose is the one the
# low level holds after the operator presses **A** (home_anchor default_qpos with the 14 arm
# joints zeroed, FK'd out of the yam URDF — the exact seed scripts/hl_serving.py uses). The
# viewer's own DISPLAY_HOME_POS are three hand-picked uncalibrated spots that are NOT a robot
# configuration: they put the chest 30 cm in front of where the G1's chest actually is, leaving
# the hands 5 cm ahead of the root instead of 45 cm, and the completion answered the only way it
# could — arms folded back, elbows behind the torso (measured: left elbow 174 mm behind the
# shoulder, vs 9 mm at the ready pose). Pass `--display-frame display-home` to see the old
# placement; it is still the default for a run WITHOUT --motion-completion.
#
# THE PLAY HEAD IS THE "Action step" SLIDER. The chunk is 50 steps on the model's own 30 Hz
# grid, so no resampling happens: window_starts(50, 30, 10) = [0, 20], i.e. two windows per
# rollout, run in order as you drag. ⚠️ At the default foot-IK setting a window costs ~2 s, so
# the humanoid TRAILS the slider — drag once and let it catch up, or pass --mc-no-foot-ik.
# (That cost is a convergence threshold, not a dial; see the panel's own note.)
#
# Env knobs:
#   ALGO      diffusion (default) | flow | act   — picks which infer_s1_*.py runs
#   PORT      viser port (default: the chosen script's own)
#   DEVICE    torch device for the POLICY (default: cuda). The completion picks its own.
#   FLORE_PY   python (default: <repo>/.venv/bin/python)
# Everything after the checkpoint is forwarded, so every --mc-* flag and --sample-index /
# --seed / --num-steps work unchanged (`--help` on the chosen script lists them).
set -eo pipefail

FLORE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLORE_PY="${FLORE_PY:-$FLORE_ROOT/.venv/bin/python}"
ALGO="${ALGO:-diffusion}"
DEVICE="${DEVICE:-cuda}"

case "$ALGO" in
  diffusion|flow|act) ;;
  *) echo "[run] FATAL: ALGO must be diffusion | flow | act, got '$ALGO'." >&2; exit 2 ;;
esac

if [ "$#" -lt 1 ]; then
  echo "usage: ALGO=diffusion ./scripts/run_infer_s1_mc.sh <checkpoint-dir> [extra args]" >&2
  echo "   e.g. ./scripts/run_infer_s1_mc.sh ckpt_0831_best --sample-index 21000" >&2
  exit 2
fi
if [ ! -x "$FLORE_PY" ]; then
  echo "[run] FLoRe venv python not found at $FLORE_PY" >&2
  echo "      (create .venv, then: uv pip install --python .venv/bin/python -e \".[viz]\")" >&2
  exit 2
fi

CKPT="$1"; shift
if [ ! -f "$CKPT/config.json" ]; then
  echo "[run] FATAL: '$CKPT' has no config.json (a checkpoint DIRECTORY is expected)." >&2
  exit 2
fi

# ACTION-CHUNK PIN, DERIVED FROM THE CHECKPOINT — identical reasoning (and identical helper
# import) to run_hl_serving.sh: flore/s1_policy/data/layout.py resolves ACTION_HORIZON at IMPORT
# time from this env var, and inference/common.validate_model_layout pins it unconditionally, so
# an H=30 checkpoint on an unpinned build REFUSES TO LOAD. The checkpoint states its own
# horizon, so derive the pin from the very file about to be read and the mismatch becomes
# impossible rather than merely detected. The block must be the ACTIVE algo's: the trainer
# leaves the other two at their dataclass defaults, so an H=30 diffusion checkpoint really does
# read {model: 50, diffusion: 30, act: 50}.
if ! CKPT_H="$("$FLORE_PY" - "$CKPT/config.json" <<'PY'
import json, sys
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

SCRIPT="$FLORE_ROOT/scripts/infer_s1_$ALGO.py"
PORT_ARGS=()
if [ -n "${PORT:-}" ]; then PORT_ARGS=(--port "$PORT"); fi

# The MC overlay lives in UMI Collection's replay/ and puts motion_completion/ + viz/ on the
# path itself (mc_humanoid.py locates replay/ by walking up), so no PYTHONPATH is needed here —
# unlike run_hl_serving.sh, which imports the mink IK. Kept explicit so nobody adds one back:
# inserting those dirs at the FRONT of sys.path would shadow generically-named modules
# (collection_viz, replay_common, test_*) for this whole process.
echo "[run] offline inference + G1 motion completion: algo=$ALGO  device=$DEVICE"
echo "[run]   checkpoint: $CKPT  (action chunk $FLORE_ACTION_HORIZON steps, from its config.json)"
echo "[run]   needs a 3-tracker layout; a left-only checkpoint disables the humanoid by name."
echo "[run]   placement: the G1's DEPLOY READY POSE (--motion-completion implies"
echo "[run]     --display-frame g1-ready-pose). Pass --display-frame display-home for the old,"
echo "[run]     uncalibrated display homes — which fold the arms behind the torso."
echo "[run]   ⚠️ the completion is a SECOND model, downstream of the policy: it changes no"
echo "[run]     Metrics number and no drawn line. --mc-series target completes the dataset's"
echo "[run]     own chunk instead, as the control."
exec "$FLORE_PY" "$SCRIPT" "$CKPT" \
  --device "$DEVICE" \
  "${PORT_ARGS[@]}" \
  --motion-completion \
  "$@"
