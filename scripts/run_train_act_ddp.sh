#!/usr/bin/env bash
# Multi-GPU ACT training via torchrun + DDP. Defaults to session_2026.08.10_1535_v3
# (140 episodes, 60,024 trainable windows). SESSION picks the dataset generation, and with
# it the force look-back the model is trained on:
#
#   0810_1535      140 ep, L=1  (default)      0808_2228      106 ep, L=1 (free-space force)
#   0811_2110      105 ep, L=1                 0811_2110_fh6  105 ep, L=6 -- same episodes
#
# Usage: [SESSION=0810_1535] [BATCH=32] run_train_act_ddp.sh <n_gpus> [gpu_group] [extra args...]
#   n_gpus     1..4  (see the NUMA note below)
#   gpu_group  'low' = GPUs 0-3 (default), 'high' = GPUs 4-7
#
# Examples:
#   scripts/run_train_act_ddp.sh 4                 # GPUs 0,1,2,3 (SESSION=0810_1535 default)
#   scripts/run_train_act_ddp.sh 2 high            # GPUs 4,5
#   scripts/run_train_act_ddp.sh 4 low --max-steps 25000
#
# The single-GPU sibling is scripts/run_train_s1.sh; this file deliberately mirrors
# its guards (dataset/stats presence, CUDA check, no-clobber) so the two behave alike.
set -euo pipefail

cd "$(dirname "$0")/.."

N_GPUS=${1:?number of GPUs (1-4)}
GPU_GROUP=${2:-low}
shift || true
[[ $# -gt 0 ]] && shift || true

case "$GPU_GROUP" in
    low)  ALL_GPUS=(0 1 2 3) ;;
    high) ALL_GPUS=(4 5 6 7) ;;
    *) echo "FATAL: gpu_group must be 'low' (GPUs 0-3) or 'high' (GPUs 4-7), got '$GPU_GROUP'" >&2
       exit 1 ;;
esac

if ! [[ "$N_GPUS" =~ ^[1-4]$ ]]; then
    # Capped at 4 ON PURPOSE, not arbitrarily: `nvidia-smi topo -m` reports NODE within
    # GPUs 0-3 and within 4-7, but SYS (PCIe + QPI/UPI across NUMA nodes) between the
    # groups. There is no NVLink on this box, so an 8-way job would route half its
    # 336 MB/step gradient all-reduce over the inter-socket link and can end up SLOWER
    # than 4-way. Measure before lifting this.
    echo "FATAL: n_gpus must be 1-4 (one NUMA group; crossing 0-3/4-7 traverses QPI), got '$N_GPUS'" >&2
    exit 1
fi

GPUS=$(IFS=,; echo "${ALL_GPUS[*]:0:$N_GPUS}")

# Dataset generation. 0810_1535 (140 ep) is the current default and the first dataset whose
# FORCE channel carries real task signal -- every episode has a sustained contact plateau
# (median 1.55 s above 2 N), where 0808_2228 never exceeded 1.17 N and was effectively
# free-space. Override with SESSION=0808_2228 to reproduce the older run.
#
# Norm stats are NOT interchangeable between the two: 0810's force span is 4.7x wider, so
# reusing the 0808 file would compress every force label toward zero. The name is derived
# from SESSION for exactly that reason, and the trainer's split-provenance check refuses a
# mismatch anyway.
#
# FH is the force-history bin count baked into that file, derived from SESSION and EXPORTED
# (torchrun passes the environment to every rank). The zarr's hl_force_hist_len attr and
# FUJI's resolved FORCE_HIST_LEN must agree or hl_dataset._check_file refuses the file;
# pairing them here makes a mismatch impossible rather than merely detected. 0811_2110 vs
# 0811_2110_fh6 are the SAME 105 episodes converted at L=1 (33 ms) and L=6 (200 ms) — that
# pair is the force-look-back A/B, so run them with everything else held fixed.
#
# H is the ACTION-CHUNK length, exported as FUJI_ACTION_HORIZON the same way FH is exported as
# FUJI_FORCE_HIST_LEN. UNLIKE FH it is not a property of the file: every shipped dataset stores
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
    # 0812_merged is ONE task recorded in TWO sittings (15.38.36 + 16.21.52) and merged:
    # 115 ep / 32869 frames, and the strongest force channel of any session so far
    # (Fz span 18.6 N vs 0811's 14.0). Same L=1 / L=6 pairing as 0811.
    0812_merged)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.12_merged_v3.zarr.zip);     FH=1 ;;
    0812_merged_fh6) DS=(third_party/UMI_collection/data/datasets/session_2026.08.12_merged_v3_fh6.zarr.zip); FH=6 ;;
    # 0814_merged SUPERSEDES 0812_merged: the same task and rig, now THREE sittings
    # (15.38.36 + 16.21.52 + 08.14_16.41.40) = 283 ep. It is a strict superset, and the
    # 0812 take numbers were preserved by the merge, so the 5 rotation-glitch exclusions
    # are the same 5 names. 0812_merged is kept because ckpt_0812_*, ckpt_diff_baseline and
    # ckpt_diff_umi were trained on it -- do not re-point it.
    #
    # NOTE the 0814 sitting is a mix of recording styles: 113 takes are a full
    # open->grasp->transport cycle and 55 are transport-only (block already held, gripper
    # pinned at 43 mm). Both are correctly labelled, but 55/283 episodes contain NO grasp
    # event, so a model's grasp behaviour is learned from a slightly thinner slice than the
    # episode count suggests.
    0814_merged)     DS=(third_party/UMI_collection/data/datasets/session_2026.08.14_merged_v3.zarr.zip);     FH=1 ;;
    0814_merged_fh6) DS=(third_party/UMI_collection/data/datasets/session_2026.08.14_merged_v3_fh6.zarr.zip); FH=6 ;;
    # 0815_merged SUPERSEDES 0814_merged the same way: FOUR sittings (+ 08.15_22.13.46) =
    # 541 ep / 90684 frames, still a strict superset with the same 5 exclusions. It is the
    # first SHARDED entry -- shard 1 IS the 0814 file, shard 2 is the 258 new takes -- because
    # the single merged file is 2.49 GiB and GitHub's Git-LFS object limit is 2 GiB.
    #
    # The style mix noted above got much more lopsided: only 53 of the 258 new takes are a
    # full cycle, 205 are transport-only, so 261/541 = 48% of the dataset has NO grasp event.
    # Worse, those 205 are the LAST episodes, so --val-last-episodes 4 (below) validates on
    # transport only. Pass --hl-data.val-last-episodes 0 and select on step count, or shard
    # the 0815 sitting by style and put the full-cycle half last.
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
    # t0-inclusive grid) with its OWN norm stats, fitted with FUJI_ACTION_HORIZON=30 exported
    # (norm_stats_hl_left_wipe_board_new_h30.json) -- exactly the wipe_board_h30 / 0823_merged_h30 rule.
    wipe_board_new)     DS=(third_party/UMI_collection/data/datasets/session_2026.09.13_wipe_board_new_v3_left_clean181.zarr.zip); FH=1; VAL_EPS=4 ;;
    wipe_board_new_h30) DS=(third_party/UMI_collection/data/datasets/session_2026.09.13_wipe_board_new_v3_left_clean181.zarr.zip); FH=1; H=30; VAL_EPS=4 ;;
    *) echo "FATAL: SESSION must be one of 0810_1535 0808_2228 0811_2110 0811_2110_fh6" \
            "0812_merged 0812_merged_fh6 0814_merged 0814_merged_fh6 0815_merged" \
            "0817_merged 0817_merged_h30 0822_1457 0822_1457_h30" \
            "0823_merged 0823_merged_h30 wipe_board wipe_board_new wipe_board_new_h30," \
            "got '$SESSION'" >&2; exit 1 ;;
esac
export FUJI_FORCE_HIST_LEN="$FH"
export FUJI_ACTION_HORIZON="$H"
NS=checkpoints/norm_stats_hl_left_${SESSION}.json
RUN="act_${SESSION}_ddp${N_GPUS}"
# Derived here rather than inlined at the torchrun call so any suffix below reaches BOTH the
# output dir and the W&B run name; otherwise two runs share one name and the A/B is unreadable
# on the dashboard even though the checkpoint dirs are distinct.
WB_NAME="act-${SESSION}-ddp${N_GPUS}"

# NOFORCE=1 -- the PURE-POSITION ablation. Drops the F/T sensor from the layout, so the
# force encoder and force token are not built and the force action columns are reserved but
# UNSUPERVISED. Everything else (dataset, batch, LR, horizon, obs frames, backbone) is
# whatever SESSION already selected, which is the point: it isolates force as the single
# variable against the matching force-ful run.
#
# EXPORTED so every torchrun rank resolves the same layout -- the ranks must agree on
# action/state widths and on how many collectives the step performs, and layout.py resolves
# this at IMPORT, before any rank could be told otherwise.
#
# Its own NORM STATS, because the bundle must describe the layout: with no sensor the
# dataset emits a 0-channel force window, and normalizing that against a 6-wide
# force_history entry is a broadcast error. fit_hl_norm_stats omits the key when no anchor
# has a measured window, which is exactly this case.
if [[ "${NOFORCE:-0}" == "1" ]]; then
    export FUJI_DROP_FORCE=1
    RUN="${RUN}_noforce"
    WB_NAME="${WB_NAME}-noforce"
    NS="${NS%.json}_noforce.json"
fi
# CKPT_ROOT is overridable because /data exists only on the 8-GPU workstation; on a box
# without it every guard below passes and the run dies at the first save.
CKPT_ROOT=${CKPT_ROOT:-/data/fuji/checkpoints}
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
if [[ -d "$OUT" ]] && [[ "$*" != *--resume-from* ]]; then
    echo "FATAL: $OUT exists and no --resume-from given; it would overwrite step dirs." >&2
    echo "       Pass --resume-from $OUT/step_XXXXXXXX, or move/remove the dir." >&2
    exit 1
fi

# A silent CPU fallback would burn days, and under DDP it would also hang rather than fail.
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
# Train-window count for the epoch arithmetic printed below (display only -- the real
# number comes from the loader). 0810_1535: 60,024 usable - 1,500 val = 58,524.
case "$SESSION" in
    0810_1535)     TRAIN_WINDOWS=58524 ;;
    0808_2228)     TRAIN_WINDOWS=31369 ;;
    0811_2110)     TRAIN_WINDOWS=28870 ;;
    0811_2110_fh6) TRAIN_WINDOWS=28755 ;;   # 115 fewer: the 200 ms glitch pre-window is wider
    0812_merged)     TRAIN_WINDOWS=31432 ;;
    0812_merged_fh6) TRAIN_WINDOWS=31317 ;;  # 115 fewer, same reason as the 0811 pair
    0814_merged)     TRAIN_WINDOWS=60850 ;;  # 1.9x 0812_merged
    0814_merged_fh6) TRAIN_WINDOWS=60615 ;;  # 235 fewer, same reason as the pairs above
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

  DDP ACT: $N_GPUS process(es) on GPUs $GPUS  ->  $OUT

  --batch-size is PER GPU (the torchrun convention):
      per-GPU batch     $BATCH
      EFFECTIVE batch   $((BATCH * N_GPUS))
      updates/epoch     ~$((TRAIN_WINDOWS / (BATCH * N_GPUS)))   (single-GPU: ~$((TRAIN_WINDOWS / BATCH)))

  LR is NOT auto-scaled. At an effective batch of $((BATCH * N_GPUS)) this is a DIFFERENT
  optimization problem from the finished single-GPU run in9pye4q. ACT's two-group split
  (trunk 1e-5 / head 1e-4) makes linear scaling a judgment call, so it is left to you:
  pass --lr explicitly if you want it changed.

EOF

mkdir -p logs
set -o pipefail
# --standalone: single-node rendezvous, no external etcd. Each rank gets RANK/LOCAL_RANK/
# WORLD_SIZE from torchrun, which is exactly what fuji...training.distributed reads.
CUDA_VISIBLE_DEVICES="$GPUS" .venv/bin/torchrun \
    --standalone --nnodes 1 --nproc-per-node "$N_GPUS" \
    scripts/train_s1.py \
    --algo act \
    --hl-data.zarr-paths "${DS[@]}" \
    --hl-data.val-last-episodes "$VAL_EPS" \
    --hl-data.norm-stats-path "$NS" \
    --batch-size "$BATCH" --num-workers 8 \
    --max-steps 100000 --warmup-steps 1000 --lr 1e-4 \
    --save-every 10000 --log-every 50 \
    --act.lr-backbone 1e-5 --act.kl-weight 10 \
    --wandb --wandb-project MUSE --wandb-run-name "$WB_NAME" \
    --output-dir "$OUT" \
    "$@" \
    2>&1 | tee -a "logs/$RUN.log"
