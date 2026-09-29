#!/usr/bin/env bash
# Launch the FUJI flow-matching INFERENCE visualization in viser (OFFLINE — no
# hardware, reads only an already-converted .zarr.zip + an optional checkpoint).
# Shows the recorded zarr trajectory (blue gradient) vs. the DP-Flow rollout (orange)
# in one G1 scene, driven by the same mink IK as UMI_collection's run_replay_viz.sh,
# with gripper width + F/T curves. Pick a Start frame, click Rollout.
#
# This does NOT modify ~/UMI_collection code or the training env: the mink IK stack +
# yam URDF are referenced READ-ONLY on PYTHONPATH, and the tool runs in FUJI's own
# .venv (viz deps are the optional `viz` extra:  uv pip install -e ".[viz]").
#
# Usage:
#   ./scripts/run_rollout_viz.sh /tmp/session_3pt_ft.zarr.zip                     # random-init (pipeline check)
#   ./scripts/run_rollout_viz.sh /tmp/session_3pt_ft.zarr.zip --checkpoint checkpoints/dp_flow_3pt/step_00100000
#   PORT=8087 ./scripts/run_rollout_viz.sh <zarr> --episode 1 --device cpu
#
# Any extra args after the zarr path are forwarded to rollout_viz.py
# (--checkpoint, --episode, --device, --port, --num-denoising-steps, --reach-scale, …).
set -eo pipefail

FUJI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$FUJI_ROOT/.venv"
PY="${ROLLOUT_PY:-$VENV/bin/python}"
UMI="$FUJI_ROOT/third_party/UMI_collection"
# mink IK import (humi_data_collection.ikumi) + the IK wrappers + the reusable viser
# Timeline/wrench/chart helpers — all vendored read-only in UMI_collection.
IK_SRC="$UMI/third_party/humi_ik"
VIZ_SRC="$UMI/viz"
REPLAY_SRC="$UMI/replay"
PORT="${PORT:-8086}"

if [ "$#" -lt 1 ]; then
  echo "usage: ./scripts/run_rollout_viz.sh <converted.zarr.zip> [--checkpoint DIR] [extra args]" >&2
  echo "   e.g. ./scripts/run_rollout_viz.sh /tmp/session_3pt_ft.zarr.zip" >&2
  exit 2
fi

if [ ! -x "$PY" ]; then
  echo "[run] FUJI venv python not found at $PY (create .venv, then: uv pip install -e \".[viz]\")" >&2
  exit 2
fi

echo "[run] rollout viz on http://localhost:$PORT  (Ctrl-C to stop)"
PYTHONPATH="$IK_SRC:$VIZ_SRC:$REPLAY_SRC:${PYTHONPATH:-}" "$PY" \
  "$FUJI_ROOT/scripts/rollout_viz.py" --port "$PORT" "$@"
