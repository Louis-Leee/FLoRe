#!/usr/bin/env bash
# Shard eval_checkpoints_physical.py over the GPUs of one box.
#
# Every checkpoint of a run is scored independently, so the work is embarrassingly
# parallel; the only shared state is the read-only zarr. Each shard writes its own
# JSONL so a crashed shard cannot corrupt another's records.
#
# Usage: eval_shard.sh <run_dir> <split> <out_dir> <n_gpus> [extra eval args...]
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_DIR=${1:?run dir}
SPLIT=${2:?split (val|train)}
OUT_DIR=${3:?out dir}
NGPU=${4:?n gpus}
shift 4

[[ -d "$RUN_DIR" ]] || { echo "FATAL: no such run dir: $RUN_DIR" >&2; exit 1; }

TAG=$(basename "$RUN_DIR")
mkdir -p "$OUT_DIR" logs

# Same ordering the script's own --run discovery uses, so shard membership is
# reproducible across invocations.
mapfile -t CKPTS < <(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d \
    -exec test -f '{}/model.safetensors' \; -print | sort)
[[ ${#CKPTS[@]} -gt 0 ]] || { echo "FATAL: no checkpoints under $RUN_DIR" >&2; exit 1; }
echo "$TAG/$SPLIT: ${#CKPTS[@]} checkpoints over $NGPU GPUs"

PIDS=()
for ((g = 0; g < NGPU; g++)); do
    SHARD=()
    for ((i = g; i < ${#CKPTS[@]}; i += NGPU)); do SHARD+=("${CKPTS[i]}"); done
    [[ ${#SHARD[@]} -gt 0 ]] || continue
    JSON="$OUT_DIR/${TAG}_${SPLIT}_gpu${g}.jsonl"
    LOG="logs/eval_${TAG}_${SPLIT}_gpu${g}.log"
    : >"$JSON"
    CUDA_VISIBLE_DEVICES="$g" .venv/bin/python scripts/eval_checkpoints_physical.py \
        "${SHARD[@]}" --split "$SPLIT" --json "$JSON" "$@" >"$LOG" 2>&1 &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
[[ $FAIL -eq 0 ]] || { echo "FATAL: at least one $TAG/$SPLIT shard failed; see logs/eval_${TAG}_${SPLIT}_gpu*.log" >&2; exit 1; }
echo "$TAG/$SPLIT done: $(cat "$OUT_DIR/${TAG}_${SPLIT}"_gpu*.jsonl | wc -l) records"
