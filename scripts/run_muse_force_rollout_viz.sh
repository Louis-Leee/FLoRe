#!/usr/bin/env bash
# Launch the MuSe-on-UVA force INFERENCE visualization in viser (OFFLINE — no
# hardware). Shows the recorded zarr trajectory (blue gradient) vs the UVA force
# policy's rollout (orange, LEFT hand only) in one G1 scene, with the PREDICTED
# future force/torque overlaid on the recorded force curves.
#
# This is a TWO-PROCESS tool because the UVA model and viser live in mutually
# exclusive environments (see scripts/muse_force_rollout_viz.py's docstring):
#   * the FRONT-END (viser + mink IK) runs in FLoRe's own .venv;
#   * the BACK-END (the UVA model) runs in the `uvaforce` conda env, spawned as a
#     child process and spoken to over a pipe.
# This launcher wires the two Pythons together. It does NOT modify ~/UMI_collection
# code or either training env: the mink IK stack + yam URDF are referenced
# READ-ONLY on PYTHONPATH.
#
# Usage:
#   ./scripts/run_muse_force_rollout_viz.sh                                   # random-init backend (pipeline check)
#   ./scripts/run_muse_force_rollout_viz.sh --checkpoint <uva_workspace.ckpt> # real model
#   PORT=8088 ./scripts/run_muse_force_rollout_viz.sh --episode 3 --device cpu
#
# Env knobs (no code edits):
#   FLORE_PY        front-end python           (default: <repo>/.venv/bin/python)
#   BACKEND_PY     uvaforce python            (default: ~/miniconda3/envs/uvaforce/bin/python)
#   DATASET_ROOT   dir with the muse_force zarr(default: third_party/UMI_collection/data/datasets)
#   PORT           viser port                 (default: 8087)
#   FORCE_FUSION   early|late|both            (default: the checkpoint's / config's)
#
# Any extra args after the knobs are forwarded to muse_force_rollout_viz.py
# (--checkpoint, --episode, --device, --model-size, --tcp-extrinsic, ...).
set -eo pipefail

FLORE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLORE_PY="${FLORE_PY:-$FLORE_ROOT/.venv/bin/python}"
BACKEND_PY="${BACKEND_PY:-$HOME/miniconda3/envs/uvaforce/bin/python}"
UMI="$FLORE_ROOT/third_party/UMI_collection"
IK_SRC="$UMI/third_party/humi_ik"
VIZ_SRC="$UMI/viz"
REPLAY_SRC="$UMI/replay"
PORT="${PORT:-8087}"

# DATASET_ROOT must be absolute (the UVA hydra config rejects a relative/empty
# dataset_root_dir). Resolve the default (or a relative override) to an abspath.
DATASET_ROOT_DEFAULT="$UMI/data/datasets"
DATASET_ROOT="${DATASET_ROOT:-$DATASET_ROOT_DEFAULT}"
if [ -d "$DATASET_ROOT" ]; then
  DATASET_ROOT="$(cd "$DATASET_ROOT" && pwd)"
fi

if [ ! -x "$FLORE_PY" ]; then
  echo "[run] FLoRe venv python not found at $FLORE_PY (create .venv, then: uv pip install -e \".[viz]\")" >&2
  exit 2
fi
if [ ! -x "$BACKEND_PY" ]; then
  echo "[run] uvaforce python not found at $BACKEND_PY" >&2
  echo "      set BACKEND_PY=/path/to/uvaforce/bin/python (the UVA model env; see docs/RUNBOOK_muse_8gpu.md)" >&2
  exit 2
fi

FORCE_FUSION_ARG=()
if [ -n "${FORCE_FUSION:-}" ]; then
  FORCE_FUSION_ARG=(--force-fusion "$FORCE_FUSION")
fi

echo "[run] muse-force rollout viz on http://localhost:$PORT  (Ctrl-C to stop)"
echo "[run]   front-end: $FLORE_PY"
echo "[run]   back-end : $BACKEND_PY (dataset-root $DATASET_ROOT)"
PYTHONPATH="$IK_SRC:$VIZ_SRC:$REPLAY_SRC:${PYTHONPATH:-}" "$FLORE_PY" \
  "$FLORE_ROOT/scripts/muse_force_rollout_viz.py" \
  --backend-python "$BACKEND_PY" \
  --dataset-root "$DATASET_ROOT" \
  --port "$PORT" \
  "${FORCE_FUSION_ARG[@]}" \
  "$@"
