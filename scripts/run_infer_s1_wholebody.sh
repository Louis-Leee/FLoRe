#!/usr/bin/env bash
# OFFLINE inference viewer for diffusion_umi_wholebody: the predicted 29 G1 joints, as a robot.
#
# The wholebody sibling of ./scripts/run_infer_s1_mc.sh, and it needs NO motion completion at
# all — the policy already outputs the joints, so there is nothing to complete:
#
#   run_infer_s1_mc.sh   3 predicted EE poses --> Motion_Completion (g1_mosyn, a 2nd process)
#                                             --> a 29-DoF G1 drawn beside the trajectory lines
#   THIS SCRIPT          the 29 predicted joints ARE the robot. Two bodies, no second model:
#                          🟪 violet = the PREDICTION      ⬜ pale = the dataset's TARGET
#
# So there are no MOTION_COMPLETION_* prerequisites, no conda env, no subprocess, and no ~2 s
# per-window lag on the play head. The only extra this page needs beyond a checkpoint is the viz
# extra (viser + yourdfpy) and the UMI submodule's yam URDF; without the URDF the charts and
# metrics still come up and the page says why the bodies are missing.
#
#   ./scripts/run_infer_s1_wholebody.sh checkpoints/ckpt_0831_wb
#   ./scripts/run_infer_s1_wholebody.sh <ckpt> --sample-index 4200 --seed 3
#   PORT=8092 DEVICE=cpu ./scripts/run_infer_s1_wholebody.sh <ckpt>
#   ./scripts/run_infer_s1_wholebody.sh <ckpt> --self-test       # headless JSON, no viser
#
# It needs a WHOLEBODY checkpoint (hl_data.mode == 'wholebody'). A keypoint checkpoint is
# refused here, by name, with a pointer at its own launcher — its action has no joint block, and
# this page's whole content is joints. One script serves all three algorithm blocks
# (diffusion / flow / act) because 'wholebody' is a LAYOUT; the algo is read from config.json.
#
# ⚠️ NOT a deploy preview. The bodies show ABSOLUTE joints in the dataset's cond_mode=home
# posture space. Deploy publishes base + (q[i] - q[0]) through a joint clutch and never this
# absolute pose (measured 0.514 rad apart at the knee from the ready-pose crouch). The deploy
# path is ./scripts/run_hl_serving_wholebody.sh.
#
# Env knobs:
#   PORT      viser port (default: 8088, the script's own)
#   DEVICE    torch device (default: cuda)
#   FLORE_PY   python (default: <repo>/.venv/bin/python)
# Everything after the checkpoint is forwarded (--sample-index / --seed / --num-steps /
# --zarr-path / --yam-urdf / --self-test; `--help` on the script lists them).
set -eo pipefail

FLORE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLORE_PY="${FLORE_PY:-$FLORE_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

if [ "$#" -lt 1 ]; then
  echo "usage: ./scripts/run_infer_s1_wholebody.sh <checkpoint-dir> [extra args]" >&2
  echo "   e.g. ./scripts/run_infer_s1_wholebody.sh checkpoints/ckpt_0831_wb --sample-index 4200" >&2
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

# LAYOUT CHECK BEFORE A GPU IS TOUCHED. The session refuses a keypoint checkpoint too (by name,
# in its constructor), but that costs a dataset rebuild and a model load first — and the likely
# mistake is reaching for this launcher with a 3-tracker checkpoint, which has its own working
# launcher one file over.
CKPT_MODE="$("$FLORE_PY" -c \
  "import json,sys; print(json.load(open(sys.argv[1])).get('hl_data',{}).get('mode',''))" \
  "$CKPT/config.json")"
if [ "$CKPT_MODE" != "wholebody" ]; then
  echo "[run] FATAL: $CKPT has hl_data.mode='$CKPT_MODE', not 'wholebody'." >&2
  echo "[run]   This page draws the predicted 29 JOINT angles. A keypoint checkpoint predicts" >&2
  echo "[run]   poses, so use its own viewer:" >&2
  echo "[run]     any keypoint layout          -> ./scripts/infer_s1_{diffusion,flow,act}.py" >&2
  echo "[run]     both_hands_head[_dual] + G1  -> ./scripts/run_infer_s1_mc.sh" >&2
  exit 2
fi

# ACTION-CHUNK PIN, DERIVED FROM THE CHECKPOINT — identical reasoning (and identical helper
# import) to run_infer_s1_mc.sh: flore/s1_policy/data/layout.py resolves ACTION_HORIZON at IMPORT
# time from this env var, and inference/common.validate_model_layout pins it unconditionally, so
# an H=30 checkpoint on an unpinned build REFUSES TO LOAD. The block must be the ACTIVE algo's:
# the trainer leaves the other two at their dataclass defaults.
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

PORT_ARGS=()
if [ -n "${PORT:-}" ]; then PORT_ARGS=(--port "$PORT"); fi

echo "[run] offline WHOLEBODY inference: device=$DEVICE"
echo "[run]   checkpoint: $CKPT  layout=$CKPT_MODE  (action chunk $FLORE_ACTION_HORIZON steps @ 30 Hz)"
echo "[run]   🟪 violet body = the PREDICTED 29 joints   ⬜ pale body = the dataset's TARGET"
echo "[run]   no motion completion, no second process: the policy already outputs joints."
echo "[run]   the floor is FK'd from the target's step 0 and HELD, so a predicted crouch shows"
echo "[run]     as the body sinking and a taller prediction visibly floats. There is no root"
echo "[run]     channel to place it with — that is complete, not missing (the deploy wire is 29"
echo "[run]     positions + 29 finite-differenced velocities)."
echo "[run]   ⚠️ ABSOLUTE joints in the dataset's cond_mode=home posture space. Deploy publishes"
echo "[run]     base + (q[i] - q[0]) through a joint clutch — see run_hl_serving_wholebody.sh."
exec "$FLORE_PY" "$FLORE_ROOT/scripts/infer_s1_wholebody.py" "$CKPT" \
  --device "$DEVICE" \
  "${PORT_ARGS[@]}" \
  "$@"
