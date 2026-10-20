#!/usr/bin/env bash
# Launch OFFLINE inference + viser visualization for a MuSe-on-UVA force checkpoint —
# the fourth sibling of scripts/infer_s1_{flow,diffusion,act}.py, in the SAME viewer:
# predicted chunk vs the recorded target on the 30 Hz grid, force included.
#
# TWO PROCESSES, because the UVA model and viser live in mutually exclusive environments:
#   * FRONT-END (viser + the HL-v2 dataset) runs in FLoRe's own .venv;
#   * BACK-END  (the UVA model) runs in the `uvaforce` conda env, spawned as a child and
#     spoken to over a pipe (scripts/muse_hl_infer_server.py).
# It touches nothing in run_muse_force_rollout_viz.sh / run_muse_force_serving.sh, and
# nothing in the flow/diffusion/act path.
#
# Usage:
#   ./scripts/run_infer_s1_muse.sh <checkpoint.slim.pt> --zarr-path <session_..._v3.zarr.zip>
#   PORT=8090 ./scripts/run_infer_s1_muse.sh <ckpt> --zarr-path <zarr> --seed 3
#   ./scripts/run_infer_s1_muse.sh <ckpt> --zarr-path <zarr> --self-test   # headless JSON
#
# A random-init WIRING CHECK (no checkpoint, no GPU needed, meaningless predictions):
#   ./scripts/run_infer_s1_muse.sh --zarr-path <zarr> --device cpu --self-test
#
# WHY --zarr-path IS REQUIRED: uva_hl_force co-trains over five sessions, so the checkpoint
# records a MIX rather than one file. Both loaders (the UVA window builder in the backend and
# FLoRe's HLDataset for the target) read exactly the file named here, and every inference
# cross-checks that the anchor pose they each resolved for the same frame agrees — that check
# is what proves the two are decoding in the same absolute frame.
#
# Env knobs:
#   FLORE_PY     front-end python  (default: <repo>/.venv/bin/python)
#   BACKEND_PY  uvaforce python   (default: ~/miniconda3/envs/uvaforce/bin/python)
#   PORT        viser port        (default: 8090 — clear of 8087/8088/8089)
#   DEVICE      backend torch device (default: cuda)
#   FLORE_ACTION_HORIZON  DO NOT SET IT. This script DERIVES the chunk length from the
#               checkpoint's own pickled hydra cfg (model.policy.num_actions) and exports it,
#               because flore/s1_policy/data/layout.py resolves the constant at IMPORT time and
#               only a launcher can still reach it. An explicit value that disagrees with the
#               checkpoint is a hard error rather than a silent override — same contract as
#               run_hl_serving.sh.
set -eo pipefail

FLORE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLORE_PY="${FLORE_PY:-$FLORE_ROOT/.venv/bin/python}"
BACKEND_PY="${BACKEND_PY:-$HOME/miniconda3/envs/uvaforce/bin/python}"
PORT="${PORT:-8090}"
DEVICE="${DEVICE:-cuda}"

if [ ! -x "$FLORE_PY" ]; then
  echo "[run] FLoRe venv python not found at $FLORE_PY (create .venv, then: uv pip install -e \".[viz]\")" >&2
  exit 2
fi
if [ ! -x "$BACKEND_PY" ]; then
  echo "[run] uvaforce python not found at $BACKEND_PY" >&2
  echo "      set BACKEND_PY=/path/to/uvaforce/bin/python (the UVA model env; see docs/RUNBOOK_muse_8gpu.md)" >&2
  exit 2
fi

# The checkpoint is the FIRST positional argument (may be absent for a random-init check).
CKPT=""
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
  CKPT="$1"
  shift
fi

# ACTION-CHUNK PIN, DERIVED FROM THE CHECKPOINT (see the env note above). The probe builds no
# model and touches no GPU; it only unpickles the cfg, which needs the uvaforce interpreter.
source "$FLORE_ROOT/scripts/lib/muse_pin.sh"
muse_pin_action_horizon "$BACKEND_PY" "$FLORE_ROOT" "$CKPT"

echo "[run] muse OFFLINE inference on http://localhost:$PORT  (Ctrl-C to stop)"
echo "[run]   front-end: $FLORE_PY"
echo "[run]   back-end : $BACKEND_PY (device=$DEVICE)"
echo "[run]   action chunk: FLORE_ACTION_HORIZON=$FLORE_ACTION_HORIZON steps"
CKPT_ARG=()
if [ -n "$CKPT" ]; then CKPT_ARG=("$CKPT"); fi
exec "$FLORE_PY" "$FLORE_ROOT/scripts/infer_s1_muse.py" \
  "${CKPT_ARG[@]}" \
  --backend-python "$BACKEND_PY" \
  --device "$DEVICE" \
  --port "$PORT" \
  "$@"
