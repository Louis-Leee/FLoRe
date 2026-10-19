#!/usr/bin/env bash
# workstation -> G1 Jetson: mirror the rig code byte-for-byte (this checkout's holosoma_extensions,
# uncommitted edits included) plus the low-level ONNX, to the container HOME (~/flore_rig/home =
# /home/rig), and link ~/flore_rig/rig.sh to its rig.sh. Re-run after every code change. Venvs are
# per-architecture and never copied (and --delete never touches the Jetson's own: they are excluded
# on both sides).
set -euo pipefail
G1="${G1:-unitree@192.168.123.164}"
DST="${DST:-flore_rig/home}"
HX="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # third_party/holosoma_extensions
RHX=FLoRe/third_party/holosoma_extensions                   # the same tree in the container HOME
# The launcher's default checkpoint for MODE=5pt-unified-anneal (3_run_policy_vive_force.sh): the
# released low level. The reference clip ships inside the tree (thirdparty/holosoma).
ONNX_RELS=(logs/WBT-Holosoma/pq9shxs7/model_62999.onnx)
for r in "${ONNX_RELS[@]}"; do
  [ -f "$HX/$r" ] || { echo "[sync] missing $HX/$r — download the released low level first (see the README)"; exit 1; }
done

# Never while the rig runs: the copy shares the robot NIC with the low level's DDS and rewrites code
# under live processes. Measured with the low level ARMED: a 784 MB ONNX copy took its inference
# 10 -> 12.7 ms and RL FPS to 49.6. SYNC_LIVE=1 overrides.
live=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$G1" \
         'timeout 10 docker ps --filter name=flore-rig- --format "{{.Names}}  {{.Command}}"' </dev/null) \
  || { echo "[sync] cannot list the robot's containers — refusing"; exit 1; }
if [ -n "$live" ] && [ "${SYNC_LIVE:-0}" != 1 ]; then
  echo "[sync] REFUSING: rig processes are running on the robot (stop them first, or SYNC_LIVE=1):"
  echo "$live" | sed 's/^/[sync]   /'
  exit 1
fi

RS=(rsync -a --delete --info=stats1 --exclude=.git --exclude=__pycache__ --exclude=.ruff_cache
    --exclude=.pytest_cache --exclude=node_modules --exclude='.venv*/')
ssh "$G1" "mkdir -p $(for r in "${ONNX_RELS[@]}"; do printf '%s ' "$DST/$RHX/$(dirname "$r")"; done)"

echo "== holosoma_extensions (no logs/, dist/, retargeting: nothing on the robot imports them)"
"${RS[@]}" --exclude=/logs/ --exclude=/dist/ --exclude=/thirdparty/mjswan/ \
    --exclude=/thirdparty/holosoma/src/holosoma_retargeting/ \
    "$HX/" "$G1:$DST/$RHX/"
echo "== low-level ONNX"
for r in "${ONNX_RELS[@]}"; do
  rsync -aL --info=stats1 "$HX/$r" "$G1:$DST/$RHX/$(dirname "$r")/"   # -L: a symlinked download lands as the file
done

# rig.sh lives in the synced tree; ~/flore_rig/rig.sh is a link to it.
ssh "$G1" "ln -sfn \$HOME/$DST/$RHX/force_deploy/onboard/rig.sh \$HOME/flore_rig/rig.sh"
ssh "$G1" "cd $DST/$RHX && sha256sum ${ONNX_RELS[*]} | cut -c1-16,65-"
echo "== synced to $G1:$DST"
