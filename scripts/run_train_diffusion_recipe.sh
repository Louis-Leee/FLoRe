#!/usr/bin/env bash
# Single-GPU DP-Diffusion training with a selectable TRAINING RECIPE.
#
# SINGLE GPU by design. For MULTI-GPU diffusion use scripts/run_train_diffusion_ddp.sh
# (torchrun + DDP, 1-8 GPUs) -- do not add torchrun here. The two are kept apart because they
# are different optimization problems at the same step count (--batch-size is PER RANK under
# torchrun), and because the run/output names must differ or a DDP run would overwrite a
# single-GPU one. Placing several single-GPU runs side by side with CUDA_VISIBLE_DEVICES is
# still the right move for an A/B: N independent runs answer N questions, where one N-GPU run
# answers one faster.
#
# RECIPE is the variable under test:
#
#   baseline  FLoRe's historical settings: lr 1e-4, betas (0.9, 0.95), wd 1e-4, no image
#             augmentation, one LR for the whole model.
#   umi       Upstream UMI's settings, found by diffing against the pristine clone at
#             third_party/universal_manipulation_interface (real-stanford @ d095ba9):
#               * lr 3e-4, betas (0.95, 0.999), wd 1e-6
#                 (config/train_diffusion_unet_timm_umi_workspace.yaml, `optimizer:`)
#               * RandomCrop 0.95 + Resize + ColorJitter(0.3, 0.4, 0.5, 0.08)
#                 (`obs_encoder.transforms`) -- FLoRe had NO image augmentation at all
#               * the pretrained ViT at 0.1x the base LR
#                 (workspace/train_diffusion_unet_image_workspace.py:62)
#
# Everything else is held fixed, so `baseline` vs `umi` on the same SESSION and the same
# --max-steps is a clean A/B on the recipe. It BUNDLES THREE CHANGES: a win does not say
# which one earned it. Split them (--diffusion.no-image-aug, or
# --diffusion.encoder-lr-scale 1.0) before spending anything on the answer.
#
# What is deliberately NOT copied from UMI: rot6d stays quantile-normalized (our own ACT
# evidence favors it), the observation stays 1 frame, and the augmentation runs in TRAIN MODE
# ONLY -- upstream also augments at validation and at deploy, which would move a val curve
# for two reasons at once. See flore/s1_policy/model/umi_parity.py.
#
# Usage: [SESSION=0812_merged] [RECIPE=umi] [BATCH=32] run_train_diffusion_recipe.sh <gpu> [extra args...]
#
# Examples:
#   RECIPE=baseline scripts/run_train_diffusion_recipe.sh 0 --max-steps 100000
#   RECIPE=umi      scripts/run_train_diffusion_recipe.sh 4 --max-steps 100000
set -euo pipefail

cd "$(dirname "$0")/.."

GPU=${1:?GPU index (0-7)}
shift || true
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "FATAL: gpu must be 0-7, got '$GPU'" >&2; exit 1; }

# Dataset generation + the force look-back baked into its norm stats. Same table as
# run_train_act_ddp.sh: FH is EXPORTED because every consumer reads FORCE_HIST_LEN as a
# def-time default, and hl_dataset._check_file refuses a zarr whose hl_force_hist_len attr
# disagrees -- pairing them here makes a mismatch impossible rather than merely detected.
#
# H is the ACTION-CHUNK length, exported as FLORE_ACTION_HORIZON the same way FH is exported as
# FLORE_FORCE_HIST_LEN. UNLIKE FH it is not a property of the file: every shipped dataset stores
# 50 steps at 30 Hz, and a shorter H trains on a strict PREFIX of that same chunk (t0-inclusive
# grid, unchanged dt), so an H=30 vs H=50 A/B needs no re-convert. It still belongs in the
# SESSION key rather than in the caller's environment, because NS/RUN/OUT are all derived from
# SESSION -- H changes glitch_post_s and therefore the anchor set and the action statistics, so
# the two variants need separate norm stats and the trainer's split-provenance check refuses
# them crossed. Hence the `_h30` suffix, exactly like `_fh6`.#
# VAL_EPS is the size of the TAIL held out for validation, per SESSION. The only split
# strategy is `last_global_flattened_episodes`, so the tail's CONTENT is whatever was recorded
# last -- and on 0814/0815 that tail was transport-only, leaving val/objective (what checkpoint
# selection reads) blind to grasping. It defaults to 4 so every pre-0817 session keeps the exact
# split its norm stats were fit against; only a session that needs a different tail overrides it.
VAL_EPS=4
H=50
SESSION=${SESSION:-0812_merged}
case "$SESSION" in
    0810_1535)       DS=(third_party/UMI_collection/data/datasets/session_2026.08.10_1535_v3.zarr.zip);       FH=1 ;;
    0808_2228)       DS=(third_party/UMI_collection/data/datasets/session_2026.08.08_2228_v3.zarr.zip);       FH=1 ;;
    0811_2110)       DS=(third_party/UMI_collection/data/datasets/session_2026.08.11_2110_v3.zarr.zip);       FH=1 ;;
    0811_2110_fh6)   DS=(third_party/UMI_collection/data/datasets/session_2026.08.11_2110_v3_fh6.zarr.zip);   FH=6 ;;
    0812_merged)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.12_merged_v3.zarr.zip);     FH=1 ;;
    0812_merged_fh6) DS=(third_party/UMI_collection/data/datasets/session_2026.08.12_merged_v3_fh6.zarr.zip); FH=6 ;;
    # 0814_merged = the same task, THREE sittings, 283 ep -- a strict superset of
    # 0812_merged with the 0812 take numbers (and so its 5 exclusions) preserved.
    # 0812_merged stays put: ckpt_diff_baseline and ckpt_diff_umi were trained on it.
    0814_merged)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.14_merged_v3.zarr.zip);     FH=1 ;;
    0814_merged_fh6) DS=(third_party/UMI_collection/data/datasets/session_2026.08.14_merged_v3_fh6.zarr.zip); FH=6 ;;
    # 0815_merged = FOUR sittings, 541 ep / 90684 frames, superset of 0814_merged. SHARDED
    # (shard 1 IS the 0814 file) because the single merged file is 2.49 GiB > GitHub's 2 GiB
    # Git-LFS object limit; DS is an array and the order is the concatenation order, so the
    # val TAIL depends on it. 48% of the dataset is transport-only and the last 205 episodes
    # all are, so --val-last-episodes 4 sees no grasp -- see the UMI_collection .gitignore.
    0815_merged)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.14_merged_v3.zarr.zip
                         third_party/UMI_collection/data/datasets/session_2026.08.15_2213_v3.zarr.zip); FH=1 ;;
    # ONE task, TWO sittings on 0817, 404 ep / 74428 frames. A NEW dataset, not a superset of
    # 0815_merged: different task setup, and the cleanest audit yet (405/405 takes usable, zero
    # position glitches, zero rotation label glitches; the single exclusion take_043 is a 4.06 s
    # tracker blackout). THREE shards: 74428 frames at the measured 37.3 KiB/frame is 2.58 GiB,
    # and even the 20.33.36 sitting alone came to 2.33 GiB as one file (rejected at push), so
    # that sitting is split at take_116 into two ~1.17 GiB halves. Order is chronological:
    # 2033a (takes 000-116) -> 2033b (takes 117-347) -> 2200 (the second sitting).
    #
    # The _h30 twin is the SAME three files: H is a training pin, not a property of the data.
    0817_merged)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.17_2033a_v3.zarr.zip
                         third_party/UMI_collection/data/datasets/session_2026.08.17_2033b_v3.zarr.zip
                         third_party/UMI_collection/data/datasets/session_2026.08.17_2200_v3.zarr.zip); FH=1; VAL_EPS=12 ;;
    0817_merged_h30) DS=(third_party/UMI_collection/data/datasets/session_2026.08.17_2033a_v3.zarr.zip
                         third_party/UMI_collection/data/datasets/session_2026.08.17_2033b_v3.zarr.zip
                         third_party/UMI_collection/data/datasets/session_2026.08.17_2200_v3.zarr.zip); FH=1; H=30; VAL_EPS=12 ;;
    # 0822_1457 = a NEW task again (driver into the socket on the disc, not the 0817 setup),
    # ONE sitting, 118 ep / 34614 frames in ONE 1.30 GiB shard -- 40.2 KiB/frame, so it fits
    # under the 2 GiB LFS limit with no a/b split. 123 takes were recorded; the 5 excluded
    # (take_032 043 059 075 096) are rotation LABEL glitches, which the position deglitch and
    # the loader's window-drop both ignore, so they have to be excluded by hand. The other
    # session recorded the same afternoon (14.44.52, a single 490 s take) is deliberately NOT
    # part of this dataset: it is 8 minutes of the rig parked while the scene was set up.
    #
    # VAL_EPS=2, not the repo default 4, because the tail is frame-heavy: takes 102-122 run
    # 13-126 s where 000-101 are ~5 s, so 4 episodes would hold out 9065 anchors = 26% of the
    # dataset. 2 gives 30297 train / 4302 val (12.4%) = the last two takes (39 s + 113 s).
    # Fit the norm stats at --val-last-episodes 2 or split_provenance refuses the file.
    0822_1457)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.22_1457_v3.zarr.zip); FH=1; VAL_EPS=2 ;;
    0822_1457_h30) DS=(third_party/UMI_collection/data/datasets/session_2026.08.22_1457_v3.zarr.zip); FH=1; H=30; VAL_EPS=2 ;;
    # 0823_merged = the SAME task as 0822 (driver into the socket on the disc), 0822 + three
    # 0823 sittings merged: 226 takes -> 219 ep / 100281 frames = 55.7 min, and it STRICTLY
    # SUPERSEDES 0822_1457 -- its first 118 episodes are BIT-IDENTICAL to that file (verified
    # on all 18 arrays + images), so this is 0822 plus 65667 new frames, not a re-conversion.
    # Seven exclusions, all rotation LABEL glitches (there-and-back tracker excursions of
    # 10-30 deg that the position-only deglitch and the loader's window-drop both ignore):
    # take_032 043 059 075 096 from the 0822 block, take_366 371 from 0823 20.39.12.
    #
    # 3.74 GiB, so it is NOT in Git LFS (hard 2 GiB/object limit) -- it lives on S3:
    #   scripts/fetch_dataset.sh 0823_merged
    #
    # VAL_EPS=4, the repo default, and unlike 0822 that needs no special pin: 219 episodes
    # make the tail cheap (last 4 = 3270 of 100195 usable anchors = 3.3%). Fit the norm stats
    # at --val-last-episodes 4 or split_provenance refuses the file.
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
RUN="diffusion_${SESSION}_${RECIPE}"
# CKPT_ROOT is overridable because /data exists only on the 8-GPU workstation; on a box
# without it every guard below passes and the run dies at the first save. Keep it OFF `/`:
# a diffusion step dir is ~3.1 GB (weights + optimizer moments + EMA shadow), 3x ACT's.
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
# per-step x0 clamp crushes every out-of-range pose/force. config.py refuses the
# combination, but that failure arrives after the model is built -- check it here.
[[ -f "$NS" ]] || { echo "FATAL: norm stats missing: $NS -- run scripts/fit_hl_norm_stats.py" >&2; exit 1; }
if [[ -d "$OUT" ]] && [[ "$*" != *--resume-from* ]]; then
    echo "FATAL: $OUT exists and no --resume-from given; it would overwrite step dirs." >&2
    echo "       Pass --resume-from $OUT/step_XXXXXXXX, or move/remove the dir." >&2
    exit 1
fi

# A silent CPU fallback exits 0 and would burn days.
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("FATAL: CUDA unavailable -- refusing to train on CPU")
print("  GPU:", torch.cuda.get_device_name(0))
PY

BATCH=${BATCH:-32}
# Train-window count for the epoch arithmetic printed below (display only -- the real
# number comes from the loader).
case "$SESSION" in
    0810_1535)       TRAIN_WINDOWS=58524 ;;
    0808_2228)       TRAIN_WINDOWS=31369 ;;
    0811_2110)       TRAIN_WINDOWS=28870 ;;
    0811_2110_fh6)   TRAIN_WINDOWS=28755 ;;
    0812_merged)     TRAIN_WINDOWS=31432 ;;
    0812_merged_fh6) TRAIN_WINDOWS=31317 ;;  # 115 fewer: the 200 ms glitch pre-window is wider
    0814_merged)     TRAIN_WINDOWS=60850 ;;  # 1.9x 0812_merged
    0814_merged_fh6) TRAIN_WINDOWS=60615 ;;  # 235 fewer, same reason
    0815_merged)     TRAIN_WINDOWS=89675 ;;  # 1.47x 0814_merged (both shards, val-last 4)
    # 0817 is a NEW dataset, and its arms use VAL_EPS=12 (grasp-poor tail), so this is
    # 74325 usable - 1524 val. IDENTICAL for the _h30 twin: this session has zero glitches,
    # so the one filter whose reach scales with H (the glitch window) drops nothing either
    # way -- both runs see the same 72801 anchors and differ only in chunk length.
    0817_merged)     TRAIN_WINDOWS=72801 ;;
    0817_merged_h30) TRAIN_WINDOWS=72801 ;;
    # 34599 usable - 4302 val (VAL_EPS=2). Same for the _h30 twin: the 15 windows the glitch
    # filter drops are dropped at both horizons, so the two runs see the same anchors.
    0822_1457)       TRAIN_WINDOWS=30297 ;;
    0822_1457_h30)   TRAIN_WINDOWS=30297 ;;
    # 100195 usable - 3270 val (VAL_EPS=4). The glitch filter drops nothing here
    # (robot0_pose_glitch is all-zero on this file), so 86 of the 100281 frames are lost
    # only to the short-episode edge and both horizons see the same anchors.
    0823_merged)     TRAIN_WINDOWS=96925 ;;
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
cat <<EOF

  Diffusion (single GPU $GPU)  ->  $OUT
  SESSION=$SESSION  RECIPE=$RECIPE
  ${RECIPE_ARGS[*]}

      batch           $BATCH
      updates/epoch   ~$((TRAIN_WINDOWS / BATCH))
      100k steps      ~$((100000 * BATCH / TRAIN_WINDOWS)) epochs

  EMA is ON (the TrainConfig default for diffusion -- UMI relies on it), so every step dir
  carries the optimizer moments AND the EMA shadow: ~3.1 GB each.

EOF

mkdir -p logs
set -o pipefail
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/train_s1.py \
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
    --batch-size "$BATCH" --num-workers 8 \
    --grad-clip 1.0 \
    --max-steps 100000 --warmup-steps 1000 \
    --save-every 10000 --log-every 50 \
    --eval-seeds 0 1 2 3 \
    --sample-every-epochs 5 \
    --sample-diagnostic-batches 1 --sample-diagnostic-samples 4 \
    --artifact-every-epochs 50 \
    --epoch-artifact-every-epochs 20 \
    --wandb --wandb-project MUSE --wandb-run-name "diffusion-${SESSION}-${RECIPE}" \
    --output-dir "$OUT" \
    "$@" \
    2>&1 | tee -a "logs/$RUN.log"
