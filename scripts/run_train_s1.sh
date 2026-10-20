#!/usr/bin/env bash
# Launch ONE of the three S1 presets, SINGLE-GPU, on a chosen dataset generation.
# Defaults to session_2026.08.10_1535_v3 (140 ep); SESSION=0808_2228 selects the 106-ep file.
# See docs/RUNBOOK_S1_WORKSTATION.md -- this is that runbook's step 8, made
# re-runnable and guarded.
#
# Usage: [SESSION=0810_1535] run_train_s1.sh <flow|diffusion|act> <gpu_index> [extra args...]
#
# For MULTI-GPU ACT use scripts/run_train_act_ddp.sh instead; flow and diffusion are
# single-GPU only (DDP is verified for ACT alone and refuses the other two).
#
# Everything the run needs is set HERE, not inherited: `tmux new` starts a fresh
# login shell and propagates only tmux's update-environment list, so DS=/NS=/WANDB_*
# exported outside arrive EMPTY -- and an empty --hl-data.zarr-paths does NOT trip
# the "No data" guard (tyro turns "" into the truthy tuple ('',)), so the run builds
# a ViT, opens a W&B run, then dies minutes later inside the loader.
set -euo pipefail

cd "$(dirname "$0")/.."

ALGO=${1:?algo (flow|diffusion|act)}
GPU=${2:?gpu index}
shift 2

case "$ALGO" in
    flow|diffusion|act) ;;
    *) echo "FATAL: algo must be flow|diffusion|act, got '$ALGO'" >&2; exit 1 ;;
esac

# Dataset generation, same SESSION switch as scripts/run_train_act_ddp.sh. 0810_1535
# (140 ep) is the default: it is 1.83x the trainable windows of 0808_2228 AND the first
# dataset whose FORCE channel carries real task signal (sustained contact plateaus, median
# 1.55 s above 2 N, vs 0808 never exceeding 1.17 N).
#
# The norm-stats FILENAME is derived from SESSION, never hardcoded: 0810's force span is
# ~4.7x wider, so reusing the 0808 stats would compress every force label toward zero. The
# trainer's split-provenance check refuses a mismatch, but this makes the right file default.
# FH is the force-history bin count baked into that file. It is derived from SESSION and
# EXPORTED, never taken from the caller: the zarr's hl_force_hist_len attr and FLoRe's
# resolved FORCE_HIST_LEN must agree or hl_dataset._check_file refuses the file, and
# pairing them here makes the mismatch impossible instead of merely detected.
#
# DS is an ARRAY because a dataset can need more than one file: GitHub's Git-LFS limit is a
# hard 2 GiB per object, so anything past ~90k frames ships SHARDED (0815_merged is 2.49 GiB
# as one file and was rejected with `[422] Size must be less than or equal to 2147483648`).
# The loader takes a list, and episode order is the CONCATENATION order -- the val split is a
# tail of it -- so the order inside a case is load-bearing, not cosmetic.
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
SESSION=${SESSION:-0810_1535}
case "$SESSION" in
    0810_1535)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.10_1535_v3.zarr.zip);     FH=1 ;;
    0808_2228)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.08_2228_v3.zarr.zip);     FH=1 ;;
    0811_2110)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.11_2110_v3.zarr.zip);     FH=1 ;;
    0811_2110_fh6) DS=(third_party/UMI_collection/data/datasets/session_2026.08.11_2110_v3_fh6.zarr.zip); FH=6 ;;
    # ONE task recorded in TWO sittings on 0812 and merged (115 ep / 32869 frames): the
    # strongest force channel of any session (Fz span 18.6 N vs 0811's 14.0).
    0812_merged)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.12_merged_v3.zarr.zip);     FH=1 ;;
    0812_merged_fh6) DS=(third_party/UMI_collection/data/datasets/session_2026.08.12_merged_v3_fh6.zarr.zip); FH=6 ;;
    # The same task, THREE sittings (+ 0814_16.41.40), 283 ep: a strict superset of
    # 0812_merged that keeps the 0812 take numbers, hence the same 5 exclusions.
    0814_merged)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.14_merged_v3.zarr.zip);     FH=1 ;;
    0814_merged_fh6) DS=(third_party/UMI_collection/data/datasets/session_2026.08.14_merged_v3_fh6.zarr.zip); FH=6 ;;
    # FOUR sittings (+ 0815_22.13.46), 541 ep / 90684 frames -- the current dataset, and the
    # first one that is SHARDED: shard 1 is byte-identical to 0814_merged above, shard 2 is
    # the 258 new takes. 48% of it is now transport-only and the last 205 episodes are ALL
    # transport-only, so --val-last-episodes 4 validates on no grasp at all; see the
    # .gitignore block in UMI_collection for the two ways out.
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
NS=checkpoints/norm_stats_hl_left_${SESSION}.json
RUN="${ALGO}_${SESSION}"
# Checkpoints go to /data (6.4 TB) -- the repo disk is 92% full and three runs at
# --save-every 10000 want ~80 GB.
# CKPT_ROOT is overridable because /data exists only on the 8-GPU workstation; on a box
# without it every guard below passes and the run dies at the first save.
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
[[ -f "$NS" ]] || { echo "FATAL: norm stats missing: $NS -- run scripts/fit_hl_norm_stats.py" >&2; exit 1; }
# Resuming is the ONLY legitimate reason to point at an existing output dir: without
# --resume-from the trainer silently overwrites same-numbered step_XXXXXXXX/ dirs.
if [[ -d "$OUT" ]] && [[ "$*" != *--resume-from* ]]; then
    echo "FATAL: $OUT exists and no --resume-from given; it would overwrite step dirs." >&2
    echo "       Pass --resume-from $OUT/step_XXXXXXXX, or move/remove the dir." >&2
    exit 1
fi

# A silent CPU fallback would burn days: the trainer downgrades to CPU when CUDA is
# unavailable and still exits 0. Fail loudly instead.
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python - "$GPU" <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit(f"FATAL: CUDA unavailable on GPU {sys.argv[1]} -- refusing to train on CPU")
print(f"GPU {sys.argv[1]}:", torch.cuda.get_device_name(0))
PY

# Per-preset knobs. Each preset reads ONLY its own prefix, so a flag from another
# preset cannot silently take effect -- an inapplicable one is refused loudly.
EXTRA=()
case "$ALGO" in
    flow)      EXTRA=(--rtc-delay-range 0 5) ;;                    # RTC is flow-only
    diffusion) EXTRA=(--diffusion.diffusion-net unet) ;;           # EMA is default-ON
    act)       EXTRA=(--act.lr-backbone 1e-5 --act.kl-weight 10) ;;
esac

mkdir -p logs

set -o pipefail
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/train_s1.py \
    --algo "$ALGO" \
    --hl-data.zarr-paths "${DS[@]}" \
    --hl-data.val-last-episodes "$VAL_EPS" \
    --hl-data.norm-stats-path "$NS" \
    --batch-size 32 --num-workers 8 \
    --max-steps 100000 --warmup-steps 1000 --lr 1e-4 \
    --save-every 10000 --log-every 50 \
    --wandb --wandb-project MUSE --wandb-run-name "${ALGO}-${SESSION}" \
    "${EXTRA[@]}" \
    --output-dir "$OUT" \
    "$@" \
    2>&1 | tee -a "logs/$RUN.log"
