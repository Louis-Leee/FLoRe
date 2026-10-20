#!/usr/bin/env bash
# diffusion_umi_3tracker -- the 3-TRACKER extension of the "diffusion umi" preset.
#
# Same model, same UMI training recipe, same 30 Hz / 50-step v3 grid as
# scripts/run_train_diffusion_{recipe,ddp}.sh. What changes is the rig it describes: the
# left-only single-camera contract becomes two hands + the chest, TWO wrist cameras, TWO
# Bota F/T sensors and TWO grippers.
#
#                          diffusion umi          diffusion_umi_3tracker
#   layout mode            left_only              both_hands_head_dual
#   --hands (converter)    left                   both_head_dual
#   cameras (obs)          1  camera0_rgb         2  camera0_rgb + camera1_rgb
#   state (obs)            7                      20   L/R [rot6d 6 + grip 1] + head rot6d
#   force in (obs)         (1, 6)                 (1, 12)  two sensors, ONE ENCODER EACH
#   action out             50 x 16                50 x 41
#                          [pos3 rot6d6 grip1 force6] per hand + [pos3 rot6d6] head
#
# It does NOT touch the left-only preset: `both_hands_head_dual` is a NEW layout mode and
# `both_head_dual` a NEW --hands value, so every shipped checkpoint keeps its column
# meaning and every existing dataset converts byte-identically. See
# flore/s1_policy/data/layout.py for why the action COLUMNS did not have to move (they were
# reserved all along) and why `state` is the one thing that is NOT cross-loadable.
#
# Usage: [SESSION=0830_1411] [RECIPE=umi] [BATCH=24] [CKPT_ROOT=...] \
#            run_train_diffusion_umi_3tracker.sh [n_gpus] [gpu_group] [extra args...]
#
#   n_gpus     1 (default, plain python) .. 8 (torchrun + DDP)
#   gpu_group  'low' = GPUs 0-3, 'high' = 4-7, 'all' = 0-7.
#              Defaults to 'low' for n_gpus<=4 and 'all' above that.
#
# Examples:
#   scripts/run_train_diffusion_umi_3tracker.sh                       # 1 GPU
#   scripts/run_train_diffusion_umi_3tracker.sh 4                     # GPUs 0-3, DDP
#   SESSION=0830_merged BATCH=16 scripts/run_train_diffusion_umi_3tracker.sh 8
#
# COST, and why BATCH defaults BELOW the left-only preset's 32: the ViT forward is the
# dominant term and there are now TWO images per sample, so activation memory per sample is
# ~2x. 24 is a starting point, not a measurement -- raise it until it OOMs and say so.
set -euo pipefail

cd "$(dirname "$0")/.."

N_GPUS=${1:-1}
shift || true
GPU_GROUP=${1:-}
# The group is OPTIONAL, so only consume it when it really is a group name -- otherwise
# `... 4 --max-steps 50000` would eat the flag.
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
esac
if (( N_GPUS > ${#ALL_GPUS[@]} )); then
    echo "FATAL: gpu_group '$GPU_GROUP' has ${#ALL_GPUS[@]} GPUs but n_gpus=$N_GPUS." >&2
    exit 1
fi
GPUS=$(IFS=,; echo "${ALL_GPUS[*]:0:$N_GPUS}")

# Dataset table. UNLIKE the left-only launchers' table this one is short and will stay
# short: a session qualifies only if BOTH hands' camera + Bota + DM4310 were attached AND
# it was converted with `--hands both_head_dual`. The rig gained that hardware on
# 2026-08-25 and 0830 is the first session recorded with all six devices, so nothing
# earlier can appear here -- an older zarr has no hl_right_action_force at all and
# HLDataset refuses it by name.
#
# FH / H are EXPORTED, not taken from the caller: torchrun passes the environment to every
# rank and both are resolved at IMPORT by flore.s1_policy.data.layout, so the ranks must
# agree before the first collective.
#
# ⚠ EVERY entry is an --exclude'd conversion, and that is NOT optional here. The 0830 sessions
# have an intermittent lighthouse ORIENTATION glitch on the CHEST puck (out-jumps 640-3800
# deg/s that rejoin), and the converter's deglitch is POSITION-only -- so it lands in
# `head_rot_axis_angle` / `hl_head_action_pose`, there is no `head_pose_glitch` array, and
# `drop_glitch_windows` reads only `robot0_pose_glitch`. The head's rot6d is 6 of this preset's
# 41 SUPERVISED columns, so an unexcluded take trains those columns on fabricated rotation at
# full loss weight with nothing downstream able to mask it. `processing/audit_session.py` names
# the takes; the counts below are its verdict.
VAL_EPS=4
H=50
FH=1
DSDIR=third_party/UMI_collection/data/datasets
SESSION=${SESSION:-0830_1411}
case "$SESSION" in
    # 54 takes recorded -> 32 jump-free on ALL THREE keypoints (21 head-glitch takes + 1
    # right-hand excursion excluded). The session the preset was built against.
    0830_1411)   DS=("$DSDIR/session_2026.08.30_1411_v3_dual_clean32.zarr.zip"); VAL_EPS=3 ;;
    # 74 takes, the same rig later the same day; audit reports 39 jump-free (head=28, left=3,
    # right=7). Convert with that audit's own --exclude list.
    0830_2249)   DS=("$DSDIR/session_2026.08.30_2249_v3_dual_clean.zarr.zip") ;;
    # Both sittings of ONE task as a single merged recording (processing/merge_sessions.py;
    # take_000..053 + take_100..173), minus the UNION of both audits' exclusions -> ~71
    # episodes, which is the only training-SCALE option here. Train on the merge or on a
    # part, never on both.
    0830_merged) DS=("$DSDIR/session_2026.08.30_merged_v3_dual_clean.zarr.zip") ;;
    # Both 2026-08-31 sittings of ONE task merged (take_000..073 + take_100..230 = 205
    # takes), minus the 89 takes that are not jump-free on ALL THREE keypoints -> 116
    # episodes, the largest dual dataset so far. The exclusion is the audit's OWN
    # `all_roles_clean` verdict, not just its [ERR ] lines: 65 of the 205 takes glitch on
    # the PRIMARY (left) puck's ROTATION, which the audit prints as [warn] because the
    # primary does get a robot0_pose_glitch mask -- but that mask is POSITION-only, so a
    # left rotation glitch still lands in supervised columns [3:9] with nothing able to
    # drop it. Reproduce the list with processing/audit_session.py and keep every take
    # whose `dirty_roles` is empty.
    0831_merged) DS=("$DSDIR/session_2026.08.31_merged_v3_dual_clean116.zarr.zip") ;;
    *) echo "FATAL: SESSION must be 0830_1411 | 0830_2249 | 0830_merged | 0831_merged, got '$SESSION'" >&2
       echo "       Convert one first (audit -> exclude -> convert):" >&2
       echo "         cd third_party/UMI_collection" >&2
       echo "         .venv/bin/python processing/audit_session.py data/recordings/session_2026.08.30_14.11.49" >&2
       echo "         # collect every '[ERR ] take_NNN' name, then:" >&2
       echo "         .venv/bin/python processing/convert_to_umi.py \\" >&2
       echo "             data/recordings/session_2026.08.30_14.11.49 \\" >&2
       echo "             -o data/datasets/session_2026.08.30_1411_v3_dual_clean32.zarr.zip \\" >&2
       echo "             --hands both_head_dual --exclude take_000 take_001 ..." >&2
       exit 1 ;;
esac
export FLORE_FORCE_HIST_LEN="$FH"
export FLORE_ACTION_HORIZON="$H"

MODE=both_hands_head_dual
CAMS=(camera0_rgb camera1_rgb)

# The same two recipes as the left-only launchers; see run_train_diffusion_recipe.sh's
# header for what each bundles and why `umi` bundles three changes at once.
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

# `_dual` in the norm-stats name because the stats are NOT interchangeable with the
# left-only ones: 41 action columns instead of 16, 20 state, and a 12-channel force window.
# The trainer asserts every width at load (`_check_norm_stats`), so a mixed-up file is a
# refusal rather than a silently mis-scaled run -- but the distinct name is what stops
# someone reaching for it in the first place.
NS=checkpoints/norm_stats_hl_dual_${SESSION}.json
RUN="diffusion_umi_3tracker_${SESSION}_${RECIPE}"
WB_NAME="diffusion-umi-3tracker-${SESSION}-${RECIPE}"
if (( N_GPUS > 1 )); then
    RUN="${RUN}_ddp${N_GPUS}"; WB_NAME="${WB_NAME}-ddp${N_GPUS}"
fi

CKPT_ROOT=${CKPT_ROOT:-/data/flore/checkpoints}
OUT="$CKPT_ROOT/$RUN"
[[ -d "$CKPT_ROOT" ]] || mkdir -p "$CKPT_ROOT" 2>/dev/null || {
    echo "FATAL: checkpoint root '$CKPT_ROOT' is missing and not creatable." >&2
    echo "       Pass CKPT_ROOT=<dir> (e.g. CKPT_ROOT=checkpoints)." >&2; exit 1; }

# W&B: your own `wandb login` is used (export WANDB_BASE_URL for a self-hosted server).
# export WANDB_ENTITY=<your-entity>   # optional: no --wandb-entity flag exists, the entity is env-only

for _shard in "${DS[@]}"; do
    [[ -f "$_shard" ]] || { echo "FATAL: dataset missing: $_shard" >&2
        echo "       Convert it with --hands both_head_dual (see the SESSION table above)." >&2
        exit 1; }
done
# Diffusion defaults to clip_sample=True, which REQUIRES normalized actions or DDIM's
# per-step x0 clamp crushes every out-of-range pose/force. config.py refuses the combination,
# but that failure arrives after N models are built -- check it here.
if [[ ! -f "$NS" ]]; then
    echo "FATAL: norm stats missing: $NS" >&2
    echo "       Fit them over the SAME mode and the SAME cameras -- HLDataset records the" >&2
    echo "       camera list in its split provenance and the trainer compares it exactly:" >&2
    echo "         .venv/bin/python scripts/fit_hl_norm_stats.py \\" >&2
    echo "             --zarr-paths ${DS[*]} --mode $MODE \\" >&2
    echo "             --image-keys ${CAMS[*]} --val-last-episodes $VAL_EPS --out $NS" >&2
    exit 1
fi
if [[ -d "$OUT" ]] && [[ "$*" != *--resume-from* ]]; then
    echo "FATAL: $OUT exists and no --resume-from given; it would overwrite step dirs." >&2
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

BATCH=${BATCH:-24}
WORKERS=${WORKERS:-8}
cat <<EOF

  diffusion_umi_3tracker: $N_GPUS process(es) on GPUs $GPUS  ->  $OUT
  SESSION=$SESSION  RECIPE=$RECIPE  MODE=$MODE  H=$H  FH=$FH
  cameras: ${CAMS[*]}   norm stats: $NS
  ${RECIPE_ARGS[*]}

  Widths this preset trains (all DERIVED from the layout, none settable):
      action     50 x 41   L[pos3 rot6d6 grip1 force6] R[same] head[pos3 rot6d6]
      state      20        L/R wrt_start_rot6d + gripper, head wrt_start_rot6d
      force in   1 x 12    two 6-DoF wrenches, one encoder group each
      vision     2 x 768   two wrist D405s, one CLS token each

  Every one of the 41 columns is supervised here. Under the left-only modes the right
  hand's gripper + 6 force columns were reserved-but-masked, so a both_hands_head run
  optimized 34 of 41 -- that difference, not the architecture, is what this preset is.
  (No backticks in this heredoc: <<EOF performs command substitution, so a quoted
  identifier would silently run as a command and print nothing.)

EOF
if (( N_GPUS > 1 )); then
    cat <<'EOF'
  DDP: --batch-size is PER GPU. LR is NOT auto-scaled, and the umi recipe's 3e-4 was tuned
  at batch 32 on the SINGLE-camera model -- an effective batch of N*BATCH here is a
  different optimization problem. Pass --lr explicitly if you want it changed.
  The trainer permits DDP for act and diffusion only (trainer._DDP_ALGOS).

EOF
fi

mkdir -p logs
set -o pipefail
TRAIN_ARGS=(
    scripts/train_s1.py
    --algo diffusion --pipeline hl
    --hl-data.zarr-paths "${DS[@]}"
    --hl-data.mode "$MODE"
    --hl-data.image-keys "${CAMS[@]}"
    --hl-data.val-last-episodes "$VAL_EPS"
    --hl-data.norm-stats-path "$NS"
    --diffusion.vit vit_b
    --diffusion.diffusion-net unet
    --diffusion.prediction-type epsilon
    --diffusion.num-train-timesteps 50
    --diffusion.num-inference-steps 16
    --diffusion.input-pertub 0.1
    --diffusion.num-draws 1
    "${RECIPE_ARGS[@]}"
    --batch-size "$BATCH" --num-workers "$WORKERS"
    --grad-clip 1.0
    --max-steps 100000 --warmup-steps 1000
    --save-every 10000 --log-every 50
    --eval-seeds 0 1 2 3
    --sample-every-epochs 5
    --sample-diagnostic-batches 1 --sample-diagnostic-samples 4
    --artifact-every-epochs 50
    --epoch-artifact-every-epochs 20
    --wandb --wandb-project MUSE --wandb-run-name "$WB_NAME"
    --output-dir "$OUT"
    "$@"
)
if (( N_GPUS > 1 )); then
    CUDA_VISIBLE_DEVICES="$GPUS" .venv/bin/torchrun \
        --standalone --nnodes 1 --nproc-per-node "$N_GPUS" \
        "${TRAIN_ARGS[@]}" 2>&1 | tee -a "logs/$RUN.log"
else
    CUDA_VISIBLE_DEVICES="$GPUS" .venv/bin/python \
        "${TRAIN_ARGS[@]}" 2>&1 | tee -a "logs/$RUN.log"
fi
