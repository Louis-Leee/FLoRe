#!/usr/bin/env bash
# Launch the ONLINE MuSe-on-UVA force PLANNER (S1) — the high-level half of the
# S0↔S1 force deploy. This is the LIVE sibling of run_muse_force_rollout_viz.sh:
# instead of scrubbing a recorded episode, it assembles a live obs from the running
# rig, runs the force policy every step, and STREAMS the predicted LEFT-hand chunk
# to the low-level controller (over the VIVE-bridge WS :8770) while visualizing the
# rollout + predicted force + gripper + low-level online/offline status in viser.
#
# TWO PROCESSES (mutually exclusive envs, same split as the rollout viz):
#   * FRONT-END (viser + mink IK) runs in FUJI's .venv;
#   * BACK-END (the UVA force model + live obs-builder + S0Driver) runs in the
#     `uvaforce` conda env, spawned as a child and spoken to over a pipe.
# It modifies NOTHING in the other sim2real scripts or run_muse_force_rollout_viz.sh.
#
# PREREQUISITES (bring these up FIRST, in their own terminals):
#   * T1 VIVE bridge is NOT needed — this planner REPLACES the VIVE teleop.
#   * T2 F/T bridge WITH --pipeline (6-DoF gravity-only force for the planner):
#        cd .../force_deploy && bash 2_ft_bridge.sh --pipeline
#     (the WS fans out to all clients, so the low-level controller reads it too).
#   * Camera bridge on :8773 (D405) and gripper bridge on :8784 (LEFT) — the same
#     bridges collection/6_record_teleoperation.sh bring up. The planner READS the
#     gripper (proprioception) but does NOT command it (manual-only bring-up).
#   * The LOW-LEVEL controller, launched in ABSOLUTE mode (see the companion
#     command printed at the end / docs) so the planner's absolute targets apply
#     directly. Press joystick Y → ready pose, then A → start; the planner's viser
#     status turns GREEN when the low-level connects to :8770.
#
# Usage:
#   PORT=8087 ./scripts/run_muse_force_serving.sh \
#       --checkpoint third_party/unified_video_action/artifacts/best-ckpt-slim/epoch=0193-val_action_l2_distances=0.112.slim.pt
#
# Env knobs (no code edits):
#   FUJI_PY     front-end python  (default: <repo>/.venv/bin/python)
#   BACKEND_PY  uvaforce python   (default: ~/miniconda3/envs/uvaforce/bin/python)
#   PORT        viser port        (default: 8087)
#   DEVICE      backend torch device (default: cuda)
#   FORCE_FUSION early|late|both  (default: the checkpoint's)
# Extra args after the knobs pass through to the frontend, and unknown ones on to
# the backend. Cadence/consistency knobs (defaults match the training grid — change
# only with a reason):
#   --step-hz 10      chunk playback rate (≈ the ~10.3 Hz training action grid). Do NOT
#                     raise to the wire rate — that replays each chunk too fast.
#   --drip-hz 30      wire REFRESH rate to :8770 (freshness gate only; not playback speed).
#   --img-stride 4    push 1 image per 4 ingest ticks (stride-4 like training; ~3.2 s hist).
#   --allow-3dof-force  opt into a 3-DoF F/T bring-up (else the planner HARD-FAILS without
#                       the 6-DoF --pipeline stream — the model was trained on 6-DoF force).
set -eo pipefail

FUJI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FUJI_PY="${FUJI_PY:-$FUJI_ROOT/.venv/bin/python}"
BACKEND_PY="${BACKEND_PY:-$HOME/miniconda3/envs/uvaforce/bin/python}"
UMI="$FUJI_ROOT/third_party/UMI_collection"
IK_SRC="$UMI/third_party/humi_ik"
VIZ_SRC="$UMI/viz"
REPLAY_SRC="$UMI/replay"
PORT="${PORT:-8087}"
DEVICE="${DEVICE:-cuda}"

if [ ! -x "$FUJI_PY" ]; then
  echo "[run] FUJI venv python not found at $FUJI_PY (create .venv, then: uv pip install -e \".[viz]\")" >&2
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

# Autonomous gripper: the HL policy DRIVES the LEFT gripper from its predicted
# per-step width (position command on :8784), in lockstep with the pose chunk.
# ON by default (this deploy runs the gripper autonomously — no manual open/close).
# Set COMMAND_GRIPPER=0 to fall back to read-only proprioception (manual-only).
GRIPPER_ARG=()
if [ "${COMMAND_GRIPPER:-1}" != "0" ]; then
  GRIPPER_ARG=(--command-gripper)
fi

echo "[run] MuSe-force PLANNER on http://localhost:$PORT  (Ctrl-C to stop)"
echo "[run]   front-end: $FUJI_PY"
echo "[run]   back-end : $BACKEND_PY (device=$DEVICE)"
echo "[run]   REMINDER: run T2 as 'bash 2_ft_bridge.sh --pipeline' (6-DoF force for the planner),"
echo "[run]            and launch the LOW-LEVEL in ABSOLUTE mode:"
echo "[run]     cd $FUJI_ROOT/third_party/holosoma_extensions/force_deploy && \\"
echo "[run]       MODE=3pt-unified-anneal HOLOSOMA_VIVE_RELATIVE=0 RECORD=1 \\"
echo "[run]       ONNX=logs/WBT-Holosoma/iss0qs71/model_15500.onnx bash 3_run_policy_vive_force.sh"
PYTHONPATH="$IK_SRC:$VIZ_SRC:$REPLAY_SRC:${PYTHONPATH:-}" "$FUJI_PY" \
  "$FUJI_ROOT/scripts/muse_force_serving_viz.py" \
  --backend-python "$BACKEND_PY" \
  --device "$DEVICE" \
  --port "$PORT" \
  "${FORCE_FUSION_ARG[@]}" \
  "${GRIPPER_ARG[@]}" \
  "$@"
