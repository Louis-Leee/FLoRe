#!/usr/bin/env bash
# Launch the S0 low-level controller: holosoma's 3pt-hybrid whole-body policy on
# the Unitree G1. This is a thin wrapper around holosoma_extensions'
# `wbt_wrappers_inference.run_policy` (branch hybrid_control-sim2real-inference).
#
# S0 consumes 3pt targets from the FLoRe client over WS :8770 (holosoma's
# ViveLiveSource connects as a client). With NO F/T bridge running, the force
# obs terms read zeros → the 168-D 3pt-hybrid policy degrades to pure 3pt
# position tracking (the FLoRe deploy mode; force integration is a later TODO).
#
# Prereqs (one-time, see docs/deployment.md):
#   - holosoma conda env `hsinference` provisioned (setup_inference.sh + bootstrap-holosoma.sh)
#   - the 3pt-hybrid ONNX at logs/WBT-Holosoma/n4exfyvp/model_06500.onnx
#   - a ref-motion NPZ (any standing clip; the policy reads its ready pose)
#
# Usage:
#   bash scripts/run_s0.sh                        # live on G1 (interface enp132s0)
#   IFACE=eth0 ONNX=/path/to.onnx bash scripts/run_s0.sh
#
# Launch ORDER (see docs/deployment.md): run S0 first (it comes up in ready
# pose, waits for `A`), then the S1 server, then the FLoRe client. The client
# gates until S0 connects on :8770 AND the first S1 chunk lands.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_s0.sh

Environment overrides:
  PY=/path/to/python  PRESET=name  ONNX=/path/to/model.onnx
  REF_MOTION=/path/to/reference.npz  IFACE=network-interface  SERVO=0|1

This command controls a physical G1. No robot-free mode is implemented yet.
EOF
}

for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    usage
    exit 0
  fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOLO="$REPO_ROOT/third_party/holosoma_extensions"

# holosoma runs under its own conda env (NOT the FLoRe uv venv). Override PY to
# point at a different interpreter.
PY="${PY:-$HOME/.holosoma_deps/miniconda3/envs/hsinference/bin/python}"
PRESET="${PRESET:-g1-29dof-holosoma-3pt-hybrid-wbt}"
ONNX="${ONNX:-$HOLO/logs/WBT-Holosoma/n4exfyvp/model_06500.onnx}"
REF_MOTION="${REF_MOTION:-$HOME/data/sonic_manipulation_test/medium_heavy_two_hands_put_down_front_medium_R_001__A508.npz}"
IFACE="${IFACE:-enp132s0}"

if [ ! -x "$PY" ]; then
  echo "[run_s0] ERROR: python interpreter not found at $PY" >&2
  echo "[run_s0]   Provision the holosoma env first (scripts/setup_inference.sh in the submodule)," >&2
  echo "[run_s0]   or set PY=/path/to/python. See docs/deployment.md." >&2
  exit 1
fi

echo "[run_s0] preset=$PRESET onnx=$ONNX iface=$IFACE"
echo "[run_s0] The FLoRe client must serve WS :8770 (run scripts/run_client.py); holosoma connects to it."
echo "[run_s0] No F/T bridge => force obs = zeros => pure 3pt position tracking (deploy mode)."

# The FLoRe client is the VIVE bridge impersonator on :8770. HOLOSOMA_VIVE_RELATIVE
# and slew caps match the force_deploy defaults; the client sends ABSOLUTE
# pelvis-frame targets so we run in the same mode the real bridge does.
cd "$HOLO/src/extensions/wbt_wrappers"
HOLOSOMA_VIVE_WS_URL="ws://localhost:8770" \
HOLOSOMA_VIVE_RELATIVE=1 \
HOLOSOMA_VIVE_ORIENTATION=1 \
HOLOSOMA_INIT_READY_POSE=1 \
HOLOSOMA_ACTION_EMA_ALPHA=0.85 \
HOLOSOMA_VIVE_MAX_POS_STEP=0.005 \
HOLOSOMA_VIVE_MAX_ORN_STEP=0.02 \
HOLOSOMA_HYBRID_SERVO="${SERVO:-0}" \
  "$PY" -m wbt_wrappers_inference.run_policy \
    "inference:$PRESET" \
    --task.model-path "$ONNX" \
    --task.ref-motion-path "$REF_MOTION" \
    --task.interface "$IFACE" \
    --task.fk-odometry true "$@"
