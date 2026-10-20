#!/usr/bin/env bash
# Multi-GPU DP-Diffusion training via torchrun + DDP -- the diffusion sibling of
# scripts/run_train_act_ddp.sh, and the multi-GPU counterpart of
# scripts/run_train_diffusion_recipe.sh (which stays the single-GPU launcher).
#
# Usage: [SESSION=0823_merged] [RECIPE=umi] [BATCH=32] [NOFORCEIN=1] run_train_diffusion_ddp.sh <n_gpus> [gpu_group] [extra args...]
#   n_gpus     1..8
#   gpu_group  'low' = GPUs 0-3, 'high' = 4-7, 'all' = 0-7.
#              Defaults to 'low' for n_gpus<=4 and 'all' above that.
#   NOFORCEIN=1  force-OUTPUT-ONLY ablation: no wrench in the observation, force columns
#                still supervised. Shares the force-ful run's norm stats on purpose.
#
# Examples:
#   SESSION=0823_merged scripts/run_train_diffusion_ddp.sh 4              # GPUs 0,1,2,3
#   SESSION=0823_merged scripts/run_train_diffusion_ddp.sh 8              # all 8, crosses NUMA
#   SESSION=0823_merged scripts/run_train_diffusion_ddp.sh 2 high         # GPUs 4,5
#   SESSION=0823_merged BATCH=64 scripts/run_train_diffusion_ddp.sh 8     # fastest measured
#
# UNLIKE ACT this is capped at 8, not 4, because 8-way is where the throughput actually is
# here -- but read the block it prints before choosing N and BATCH. This model all-reduces
# 617 MiB of fp32 gradient per step (161.7 M parameters: 85.8 M ViT-B + 75.9 M UNet, 1.8x
# ACT's payload) and one batch-32 step is only ~108 ms of compute, so the run is
# COMMUNICATION-bound and the useful knob is a bigger per-rank batch, not more ranks at a
# smaller one. `nvidia-smi topo -m` says NODE inside GPUs 0-3 and 4-7, SYS across.
#
# The trainer permits DDP for act and diffusion only (`trainer._DDP_ALGOS`); flow is still
# refused, because its per-sample RTC delay draw makes the per-rank valid-element count vary
# by construction and nothing has measured what that does to the reduced gradient. The
# diffusion path is covered by tests/test_ddp_diffusion.py: gradient parity against a
# single-process step, EMA-shadow parity (diffusion's deploy weights), disjoint shards,
# exact validation re-aggregation, and a convergence check.
set -euo pipefail

cd "$(dirname "$0")/.."

N_GPUS=${1:?number of GPUs (1-8)}
shift || true
GPU_GROUP=${1:-}
# The second positional is OPTIONAL, so only consume it when it really is a group name --
# otherwise `run_train_diffusion_ddp.sh 4 --max-steps 50000` would eat the flag.
case "$GPU_GROUP" in
    low|high|all) shift ;;
    *) GPU_GROUP="" ;;
esac

if ! [[ "$N_GPUS" =~ ^[1-8]$ ]]; then
    echo "FATAL: n_gpus must be 1-8, got '$N_GPUS'" >&2; exit 1
fi
if [[ -z "$GPU_GROUP" ]]; then
    if (( N_GPUS > 4 )); then GPU_GROUP=all; else GPU_GROUP=low; fi
fi
case "$GPU_GROUP" in
    low)  ALL_GPUS=(0 1 2 3) ;;
    high) ALL_GPUS=(4 5 6 7) ;;
    all)  ALL_GPUS=(0 1 2 3 4 5 6 7) ;;
    *) echo "FATAL: gpu_group must be low|high|all, got '$GPU_GROUP'" >&2; exit 1 ;;
esac
if (( N_GPUS > ${#ALL_GPUS[@]} )); then
    echo "FATAL: gpu_group '$GPU_GROUP' has ${#ALL_GPUS[@]} GPUs but n_gpus=$N_GPUS." >&2
    echo "       Use 'all' to span both NUMA groups (slower per step -- see the header)." >&2
    exit 1
fi
GPUS=$(IFS=,; echo "${ALL_GPUS[*]:0:$N_GPUS}")

# Dataset generation, the SAME table as scripts/run_train_diffusion_recipe.sh /
# run_train_act_ddp.sh / run_train_s1.sh -- FOUR copies now, change them together. The full
# provenance of every entry (why the shards, why VAL_EPS, why the _h30 twin) lives in
# run_train_diffusion_recipe.sh; kept short here on purpose rather than re-pasted, so the two
# cannot drift in their PROSE while agreeing on their values.
#
# FH (force look-back) and H (action-chunk length) are EXPORTED, not taken from the caller:
# torchrun passes the environment to every rank, and both are resolved at IMPORT by
# flore.s1_policy.data.layout, so every rank must agree on them before the first collective.
VAL_EPS=4
H=50
SESSION=${SESSION:-0823_merged}
case "$SESSION" in
    0810_1535)       DS=(third_party/UMI_collection/data/datasets/session_2026.08.10_1535_v3.zarr.zip);       FH=1 ;;
    0808_2228)       DS=(third_party/UMI_collection/data/datasets/session_2026.08.08_2228_v3.zarr.zip);       FH=1 ;;
    0811_2110)       DS=(third_party/UMI_collection/data/datasets/session_2026.08.11_2110_v3.zarr.zip);       FH=1 ;;
    0811_2110_fh6)   DS=(third_party/UMI_collection/data/datasets/session_2026.08.11_2110_v3_fh6.zarr.zip);   FH=6 ;;
    0812_merged)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.12_merged_v3.zarr.zip);     FH=1 ;;
    0812_merged_fh6) DS=(third_party/UMI_collection/data/datasets/session_2026.08.12_merged_v3_fh6.zarr.zip); FH=6 ;;
    0814_merged)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.14_merged_v3.zarr.zip);     FH=1 ;;
    0814_merged_fh6) DS=(third_party/UMI_collection/data/datasets/session_2026.08.14_merged_v3_fh6.zarr.zip); FH=6 ;;
    # Sharded: one merged file would be 2.49 GiB, past GitHub's 2 GiB Git-LFS object limit.
    # Array ORDER is the concatenation order, and the val split is a TAIL of it.
    0815_merged)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.14_merged_v3.zarr.zip
                         third_party/UMI_collection/data/datasets/session_2026.08.15_2213_v3.zarr.zip); FH=1 ;;
    # 404 ep / 74428 frames in 3 chronological shards; VAL_EPS=12 (the 4-episode tail is
    # grasp-poor). The _h30 twin is the SAME files: H is a training pin, not file metadata.
    0817_merged)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.17_2033a_v3.zarr.zip
                         third_party/UMI_collection/data/datasets/session_2026.08.17_2033b_v3.zarr.zip
                         third_party/UMI_collection/data/datasets/session_2026.08.17_2200_v3.zarr.zip); FH=1; VAL_EPS=12 ;;
    0817_merged_h30) DS=(third_party/UMI_collection/data/datasets/session_2026.08.17_2033a_v3.zarr.zip
                         third_party/UMI_collection/data/datasets/session_2026.08.17_2033b_v3.zarr.zip
                         third_party/UMI_collection/data/datasets/session_2026.08.17_2200_v3.zarr.zip); FH=1; H=30; VAL_EPS=12 ;;
    # 118 ep / 34614 frames, one shard. VAL_EPS=2, not 4: the tail is frame-heavy, and 4
    # episodes would hold out 26% of the dataset.
    0822_1457)       DS=(third_party/UMI_collection/data/datasets/session_2026.08.22_1457_v3.zarr.zip); FH=1; VAL_EPS=2 ;;
    0822_1457_h30)   DS=(third_party/UMI_collection/data/datasets/session_2026.08.22_1457_v3.zarr.zip); FH=1; H=30; VAL_EPS=2 ;;
    # 219 ep / 100281 frames, and a strict superset of 0822_1457 (its first 118 episodes are
    # bit-identical) -- train on one or the other, never both. 3.74 GiB, so it is NOT in Git
    # LFS (hard 2 GiB/object) and NOT sharded to squeeze under it; it lives on S3:
    #   scripts/fetch_dataset.sh 0823_merged
    0823_merged)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.23_merged_v3.zarr.zip); FH=1; VAL_EPS=4 ;;
    0823_merged_h30) DS=(third_party/UMI_collection/data/datasets/session_2026.08.23_merged_v3.zarr.zip); FH=1; H=30; VAL_EPS=4 ;;
    # wipe_board = a NEW TASK (wiping a board), ONE single-tracker session (2026-09-06 22:01): LEFT
    # hand only -- 1 puck /
    # 1 camera / 1 Bota / 1 DM4310 -- so it trains left_only exactly like 0823, with no
    # motion completion and no foot/head arrays. 302 takes -> 291 ep / 80382 frames, ONE
    # zarr on S3 (scripts/fetch_dataset.sh wipe_board). 11 excluded: take_049 (aborted,
    # 1 frame) + 10 rotation glitches (8 RAW there-and-back excursions of 11-27 deg at
    # 738-2148 deg/s, 2 label-only there-and-backs of 8-10 deg/frame) -- the same rule
    # 0822/0823 applied. Zero position glitches. VAL_EPS=4 = the repo default: the tail
    # (take_298..301) is the session's four LONGEST takes (20-29 s, sustained contact, no
    # grasp), 2654 frames = 3.3% of the anchors.
    wipe_board)       DS=(third_party/UMI_collection/data/datasets/session_2026.09.06_wipe_board_v3_left_clean291.zarr.zip); FH=1; VAL_EPS=4 ;;
    # wipe_board_new = the board-wiping task RE-RECORDED a week later (2026-09-13 22:55, ONE sitting,
    # 181 takes) and kept as its OWN dataset -- NEVER merged with wipe_board (the user's call: a new
    # task key, not a supplement; norm stats, SESSION and S3 folder are all separate). Same rig shape
    # as wipe_board: MODE=1pt, LEFT hand only, 1 puck / 1 camera / 1 Bota / 1 DM4310 -> converted
    # `--hands left`, trains left_only with the umi recipe, no completion, no head/foot arrays.
    # 181 takes -> 181 ep / 34388 frames / 19.1 min, ONE 0.70 GiB zarr on S3
    # (scripts/fetch_dataset.sh wipe_board_new). ZERO takes excluded: audit_session.py 0 [ERR],
    # the converter dropped 0, and the left puck has 0 position / 0 rotation excursions and 0 glitch
    # frames (the converter now masks rotation excursions itself, so wipe_board's hand-made
    # rotation --exclude list has no counterpart here). F/T tared + compensated + sign-verified on
    # all 181; t0-bin |F| median 0.52 N / q99 20.0 / max 22.8 (Fx/Fz spans 16.3 / 23.2 N, a little
    # wider than wipe_board's 14.9 / 18.2). VAL_EPS=4 = the repo default: the tail (take_177..180) is
    # 13.5-18.8 s of sustained wiping, 1817 anchors = 5.3% of the 34293 usable.
    #
    # The _h30 twin is the SAME file (a 30-step chunk is a strict PREFIX of the stored 50 on the
    # t0-inclusive grid) with its OWN norm stats, fitted with FLORE_ACTION_HORIZON=30 exported
    # (norm_stats_hl_left_wipe_board_new_h30.json) -- exactly the wipe_board_h30 / 0823_merged_h30 rule.
    wipe_board_new)     DS=(third_party/UMI_collection/data/datasets/session_2026.09.13_wipe_board_new_v3_left_clean181.zarr.zip); FH=1; VAL_EPS=4 ;;
    wipe_board_new_h30) DS=(third_party/UMI_collection/data/datasets/session_2026.09.13_wipe_board_new_v3_left_clean181.zarr.zip); FH=1; H=30; VAL_EPS=4 ;;
    *) echo "FATAL: SESSION must be one of 0810_1535 0808_2228 0811_2110 0811_2110_fh6" \
            "0812_merged 0812_merged_fh6 0814_merged 0814_merged_fh6 0815_merged" \
            "0817_merged 0817_merged_h30 0822_1457 0822_1457_h30" \
            "0823_merged 0823_merged_h30 wipe_board wipe_board_new wipe_board_new_h30," \
            "got '$SESSION'" >&2; exit 1 ;;
esac
export FLORE_FORCE_HIST_LEN="$FH"
export FLORE_ACTION_HORIZON="$H"

# Same two recipes as the single-GPU launcher; see its header for what each bundles and why
# `umi` bundles three changes at once.
RECIPE=${RECIPE:-umi}
case "$RECIPE" in
    baseline)
        RECIPE_ARGS=(--lr 1e-4 --betas 0.9 0.95 --weight-decay 1e-4
                     --diffusion.no-image-aug --diffusion.encoder-lr-scale 1.0) ;;
    umi)
        RECIPE_ARGS=(--lr 3e-4 --betas 0.95 0.999 --weight-decay 1e-6
                     --diffusion.image-aug
                     --diffusion.image-aug-crop-ratio 0.95
                     --diffusion.image-aug-brightness 0.3
                     --diffusion.image-aug-contrast 0.4
                     --diffusion.image-aug-saturation 0.5
                     --diffusion.image-aug-hue 0.08
                     --diffusion.encoder-lr-scale 0.1) ;;
    *) echo "FATAL: RECIPE must be 'baseline' or 'umi', got '$RECIPE'" >&2; exit 1 ;;
esac

NS=checkpoints/norm_stats_hl_left_${SESSION}.json
# `_ddp${N_GPUS}` is in the name so a multi-GPU run can never overwrite the single-GPU
# `diffusion_${SESSION}_${RECIPE}` dir, and so the two are distinguishable on the W&B
# dashboard -- they are different optimization problems at the same step count.
RUN="diffusion_${SESSION}_${RECIPE}_ddp${N_GPUS}"
WB_NAME="diffusion-${SESSION}-${RECIPE}-ddp${N_GPUS}"

# NOFORCEIN=1 -- the FORCE-OUTPUT-ONLY ablation. The measured wrench leaves the OBSERVATION
# (no force encoder, no force token, so there is no path from the F/T to global_cond) while the
# six force ACTION columns stay supervised exactly as before. It asks: is force useful as an
# input, or is it already predictable from vision + state?
#
# Distinct from ACT's NOFORCE=1 (FLORE_DROP_FORCE), which also stops supervising those columns.
# The table in layout._resolve_drop_force_input spells out all three switches.
#
# EXPORTED so every torchrun rank resolves the same widths: layout.py reads this at IMPORT,
# and the ranks must agree on global_cond's width before the first collective.
#
# It deliberately does NOT get its own norm stats, unlike NOFORCE. Nothing about the DATA
# changes -- the dataset still emits the real 6-channel force window and the anchor set is
# bit-identical -- so this run shares the force-ful run's stats file, which is what makes the
# two directly comparable metric for metric, including val/force.
if [[ "${NOFORCEIN:-0}" == "1" ]]; then
    export FLORE_DROP_FORCE_INPUT=1
    RUN="${RUN}_noforcein"
    WB_NAME="${WB_NAME}-noforcein"
fi
# CKPT_ROOT is overridable because /data exists only on the 8-GPU workstation. Keep it OFF
# `/`: a diffusion step dir is ~3.1 GB (weights + optimizer moments + EMA shadow).
CKPT_ROOT=${CKPT_ROOT:-/data/flore/checkpoints}
OUT="$CKPT_ROOT/$RUN"
[[ -d "$CKPT_ROOT" ]] || mkdir -p "$CKPT_ROOT" 2>/dev/null || {
    echo "FATAL: checkpoint root '$CKPT_ROOT' is missing and not creatable." >&2
    echo "       Pass CKPT_ROOT=<dir> (e.g. CKPT_ROOT=checkpoints)." >&2; exit 1; }

# W&B: your own `wandb login` is used (export WANDB_BASE_URL for a self-hosted server).
# export WANDB_ENTITY=<your-entity>   # optional: no --wandb-entity flag exists, the entity is env-only

for _shard in "${DS[@]}"; do
    [[ -f "$_shard" ]] || { echo "FATAL: dataset missing: $_shard" >&2; exit 1; }
done
# Diffusion defaults to clip_sample=True, which REQUIRES normalized actions or DDIM's
# per-step x0 clamp crushes every out-of-range pose/force. config.py refuses the combination,
# but that failure arrives after N models are built -- check it here.
[[ -f "$NS" ]] || { echo "FATAL: norm stats missing: $NS -- run scripts/fit_hl_norm_stats.py" >&2; exit 1; }
if [[ -d "$OUT" ]] && [[ "$*" != *--resume-from* ]]; then
    echo "FATAL: $OUT exists and no --resume-from given; it would overwrite step dirs." >&2
    echo "       Pass --resume-from $OUT/step_XXXXXXXX, or move/remove the dir." >&2
    exit 1
fi

# A silent CPU fallback would burn days, and under DDP it would also HANG rather than fail:
# gloo replaces nccl, every rank runs on the same cores, and nothing says why.
CUDA_VISIBLE_DEVICES="$GPUS" .venv/bin/python - "$N_GPUS" <<'PY'
import sys, torch
want = int(sys.argv[1])
if not torch.cuda.is_available():
    sys.exit("FATAL: CUDA unavailable -- refusing to train on CPU")
have = torch.cuda.device_count()
if have < want:
    sys.exit(f"FATAL: asked for {want} GPUs but only {have} visible")
for i in range(have):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
PY

BATCH=${BATCH:-32}
# PER RANK. 8 ranks x 8 = 64 dataloader processes, each decoding JPEG-XL and holding its own
# zarr handles (HLDataset reopens per PID), so this is the knob to turn down first if the box
# runs out of RAM or file descriptors.
WORKERS=${WORKERS:-8}
# Train-window count for the epoch arithmetic printed below (display only -- the real number
# comes from the loader).
case "$SESSION" in
    0810_1535)       TRAIN_WINDOWS=58524 ;;
    0808_2228)       TRAIN_WINDOWS=31369 ;;
    0811_2110)       TRAIN_WINDOWS=28870 ;;
    0811_2110_fh6)   TRAIN_WINDOWS=28755 ;;
    0812_merged)     TRAIN_WINDOWS=31432 ;;
    0812_merged_fh6) TRAIN_WINDOWS=31317 ;;
    0814_merged)     TRAIN_WINDOWS=60850 ;;
    0814_merged_fh6) TRAIN_WINDOWS=60615 ;;
    0815_merged)     TRAIN_WINDOWS=89675 ;;
    0817_merged)     TRAIN_WINDOWS=72801 ;;
    0817_merged_h30) TRAIN_WINDOWS=72801 ;;
    0822_1457)       TRAIN_WINDOWS=30297 ;;
    0822_1457_h30)   TRAIN_WINDOWS=30297 ;;
    0823_merged)     TRAIN_WINDOWS=96925 ;;   # 100195 usable - 3270 val (VAL_EPS=4)
    # 20 MORE than H=50, not equal: this session has pose glitches and the
    # glitch-window exclusion reach scales with H, so the shorter horizon keeps windows
    # the longer one drops. Measured twice -- HLDataset.split_indices(4), and the
    # train_anchor_count of each horizon's own norm-stats fit.
    0823_merged_h30) TRAIN_WINDOWS=96945 ;;
    # 80027 usable - 2654 val (VAL_EPS=4); zero pose glitches, so the glitch filter drops
    # nothing and the 355 frames lost are the short-episode edge alone.
    wipe_board)       TRAIN_WINDOWS=77373 ;;
    # 34293 usable - 1817 val (VAL_EPS=4). Zero pose glitches; the 95 frames lost are the
    # short-episode edge alone. The H=30 twin keeps 20 MORE (34313 usable): the edge is the one
    # H-dependent term on a glitch-free file (an anchor needs a full chunk inside its own episode).
    wipe_board_new)     TRAIN_WINDOWS=32476 ;;
    wipe_board_new_h30) TRAIN_WINDOWS=32496 ;;
esac
EFFECTIVE=$((BATCH * N_GPUS))
cat <<EOF

  DDP diffusion: $N_GPUS process(es) on GPUs $GPUS  ->  $OUT
  SESSION=$SESSION  RECIPE=$RECIPE  H=$H  FH=$FH  force-input=$( [[ "${NOFORCEIN:-0}" == 1 ]] && echo MASKED-OUT || echo on )
  ${RECIPE_ARGS[*]}

  --batch-size is PER GPU (the torchrun convention):
      per-GPU batch     $BATCH
      EFFECTIVE batch   $EFFECTIVE
      updates/epoch     ~$((TRAIN_WINDOWS / EFFECTIVE))   (single-GPU: ~$((TRAIN_WINDOWS / BATCH)))
      100k steps        ~$((100000 * EFFECTIVE / TRAIN_WINDOWS)) epochs (single-GPU: ~$((100000 * BATCH / TRAIN_WINDOWS)))
      dataloader procs  $((WORKERS * N_GPUS)) total ($WORKERS per rank)

  LR is NOT auto-scaled, and for diffusion that matters more than it does for ACT: the umi
  recipe's 3e-4 was tuned at batch 32. At an effective $EFFECTIVE this is a DIFFERENT
  optimization problem from the single-GPU runs -- pass --lr explicitly if you want it changed.

  DO NOT shrink BATCH to hold the effective batch at 32. That is ACT's trick and it INVERTS
  here, measured on this box: the gradient all-reduce costs a FIXED 168/178/323 ms per step at
  2/4/8 ranks (617 MiB fp32, no NVLink) regardless of batch size, while one batch-32 step is
  only ~108 ms of compute. So BATCH=8 on 4 GPUs measured 124 samples/s aggregate -- 2.4x
  SLOWER than a single GPU at batch 32 (297). A BIGGER per-rank batch is what amortizes the
  fixed comm cost: batch 64 on 8 GPUs measured 1081 samples/s (3.64x). Full table in
  docs/RUNBOOK_S1_WORKSTATION.md section 8c.

  EMA is ON (the TrainConfig default for diffusion). Every rank keeps its own shadow, they
  stay bit-identical because the reduced gradient is, and only rank 0 writes -- so the
  deployed model.safetensors is the EMA of the averaged trajectory, as single-GPU.

  Watch for OVERFITTING, not undertraining: the finished 0822/0823 diffusion runs peaked at
  epoch 5-6 and then degraded 3-4x by 100k steps. A bigger effective batch means FEWER epochs
  per step budget, which helps -- but select on best_overall, not on the final step dir.
EOF
if [[ "$GPU_GROUP" == "all" ]] && (( N_GPUS > 4 )); then
    cat <<'EOF'
  ⚠ This job spans BOTH NUMA groups. `nvidia-smi topo -m` reports NODE within GPUs 0-3 and
    within 4-7 but SYS (PCIe + UPI) across, and there is no NVLink on this box, so roughly
    half of the 617 MiB per-step gradient all-reduce crosses the inter-socket link. Expect
    materially sublinear scaling; 4 GPUs inside one group can be the better trade.

EOF
fi

mkdir -p logs
set -o pipefail
# --standalone: single-node rendezvous, no external etcd. Each rank gets RANK/LOCAL_RANK/
# WORLD_SIZE from torchrun, which is exactly what flore...training.distributed reads.
CUDA_VISIBLE_DEVICES="$GPUS" .venv/bin/torchrun \
    --standalone --nnodes 1 --nproc-per-node "$N_GPUS" \
    scripts/train_s1.py \
    --algo diffusion --pipeline hl \
    --hl-data.zarr-paths "${DS[@]}" \
    --hl-data.mode left_only \
    --hl-data.val-last-episodes "$VAL_EPS" \
    --hl-data.norm-stats-path "$NS" \
    --diffusion.vit vit_b \
    --diffusion.diffusion-net unet \
    --diffusion.prediction-type epsilon \
    --diffusion.num-train-timesteps 50 \
    --diffusion.num-inference-steps 16 \
    --diffusion.input-pertub 0.1 \
    --diffusion.num-draws 1 \
    "${RECIPE_ARGS[@]}" \
    --batch-size "$BATCH" --num-workers "$WORKERS" \
    --grad-clip 1.0 \
    --max-steps 100000 --warmup-steps 1000 \
    --save-every 10000 --log-every 50 \
    --eval-seeds 0 1 2 3 \
    --sample-every-epochs 5 \
    --sample-diagnostic-batches 1 --sample-diagnostic-samples 4 \
    --artifact-every-epochs 50 \
    --epoch-artifact-every-epochs 20 \
    --wandb --wandb-project MUSE --wandb-run-name "$WB_NAME" \
    --output-dir "$OUT" \
    "$@" \
    2>&1 | tee -a "logs/$RUN.log"
