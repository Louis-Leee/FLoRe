#!/usr/bin/env bash
# The diffusion_umi_5pt HORIZON ablation, as a QUEUE.
#
# Each cell is one full `run_train_diffusion_umi_5pt.sh` invocation on ALL the GPUs you give it,
# run one after another. Three reasons this exists rather than three typed commands, each of
# which has cost a wasted queue slot when done by hand:
#
#   1. EVERY CELL NEEDS ITS OWN NORM STATS. `mode` is in the exact-equality dataset_knobs and
#      the action quantiles depend on the HORIZON -- a 200-step chunk reaches further from its
#      anchor than a 50-step one, so its per-column q01/q99 are wider at an IDENTICAL width.
#      Crossed, the file loads, `_check_norm_stats` passes, and diffusion's `clip_sample=True`
#      then clamps every genuinely out-of-range column on every DDIM step. This fits any missing
#      one FIRST, with FLORE_ACTION_HORIZON exported before the fit (the pin is resolved at
#      IMPORT, so exporting it afterwards silently fits the wrong horizon).
#   2. A CELL THAT ALREADY HAS AN OUTPUT DIR IS SKIPPED, not overwritten and not fatal, so
#      re-running after an interruption resumes the QUEUE instead of dying on cell 1.
#   3. THE PLAN IS PRINTABLE. `--dry-run` shows every cell's mode, stats file and output dir
#      without touching a GPU -- the only cheap way to notice that SESSION does not store the
#      horizon you asked for.
#
# THE DEFAULT GRID -- the ablation that was asked for: one observation, three horizons.
#
#   cell             H    chunk         asks
#   xy3_rpy:50       50   1.67 s        the baseline length every other preset trains at
#   xy3_rpy:100     100   3.33 s        does a longer chunk help this 5-point action?
#   xy3_rpy:200     200   6.67 s        ...and does it keep helping?
#
# ⚠️ A LONG HORIZON COSTS SUPERVISION AND THE MASK HIDES IT. The median episode of
# 0902_merged3 is 177 frames = 5.9 s, so only ~55.5% of anchors have a full 100-step future and
# ~23.5% a full 200-step one. The rest are masked, which is correct and silent: the loss simply
# averages over fewer columns. READ THE PER-GROUP val/* -- not the scalar val loss -- before
# concluding that a long chunk did or did not help, because the three cells are not scoring the
# same number of steps. (This is the trap the earlier H=30-vs-H=50 A/B fell into: 41% of its
# apparent gap was an artifact of scoring different horizons.)
#
# The state ABLATION is a separate axis and is NOT in the default grid, because the horizon
# question was the one asked. Add cells to run it: CELLS="xy3_rpy:100 none:100 xy3:100 rpy:100"
# gives the 4-way "which half of the observation carries the run" at one horizon, the same shape
# run_train_wb_geom_grid.sh uses.
#
# COST. The wholebody grid measured 160 samples/s = 6.7 steps/s at batch 24 on ONE RTX PRO 5000,
# i.e. ~4 h per 100k steps, and DDP does not change steps/s (each rank runs its own batch). This
# preset's action is 59 wide rather than 43, which is a wider UNet output but the same two ViT
# forwards, so budget the same order and TIME THE FIRST CELL before trusting it. H=200's UNet
# time axis is 4x H=50's. For a screening pass use MAX_STEPS=40000 -- the val curves separate
# long before 100k and a cell can be extended later with --resume-from.
#
# Usage
#   scripts/run_train_5pt_grid.sh --dry-run                        # print the plan
#   scripts/run_train_5pt_grid.sh 8                                # the 3-cell grid, 8 GPUs
#   CELLS="xy3_rpy:50" scripts/run_train_5pt_grid.sh 4             # one cell
#   MAX_STEPS=40000 scripts/run_train_5pt_grid.sh 8                # screening pass
#
# Env (all forwarded to the per-cell launcher unless noted):
#   SESSION      dataset key. Default 0902_merged3_h200 -- the only registered entry that
#                stores 200 steps, so the default grid needs no other value.
#   CELLS        space-separated "<WB_STATE>:<H>" pairs. Default: the 3 above.
#   MAX_STEPS    per cell. Default 100000 (the launcher's own default).
#   BATCH        per GPU. Default 24 (the launcher's).
#   NOISE_XY_M / NOISE_RPY_DEG   state jitter. Defaults 0.010 m / 2.0 deg, and the launcher
#                zeroes whichever the cell's mode cannot carry. See the launcher's header for
#                why the habitual 2.5 cm / 10 deg is 1-4x too large for THIS state.
#   CKPT_ROOT / DSDIR   as the launcher.
#   GRID_LOG     queue log. Default logs/5pt_grid.log
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; shift; fi
N_GPUS=${1:-8}
shift || true

SESSION=${SESSION:-0902_merged3_h200}
CELLS=${CELLS:-"xy3_rpy:50 xy3_rpy:100 xy3_rpy:200"}
MAX_STEPS=${MAX_STEPS:-100000}
BATCH=${BATCH:-24}
NOISE_XY_M=${NOISE_XY_M:-0.010}
NOISE_RPY_DEG=${NOISE_RPY_DEG:-2.0}
CKPT_ROOT=${CKPT_ROOT:-/data/flore/checkpoints}
DSDIR=${DSDIR:-third_party/UMI_collection/data/datasets}
GRID_LOG=${GRID_LOG:-logs/5pt_grid.log}
VAL_EPS=4
CAMS=(camera0_rgb camera1_rgb)
mkdir -p logs

# The dataset path and the stats/run naming are DERIVED BY THE LAUNCHER, so asking it rather
# than re-deriving them here is what stops the two drifting apart.
plan_for() {  # $1=WB_STATE $2=H  -> prints "MODE<TAB>DS<TAB>NS<TAB>OUT"
    SESSION="$SESSION" WB_STATE="$1" H="$2" \
    NOISE_XY_M="$NOISE_XY_M" NOISE_RPY_DEG="$NOISE_RPY_DEG" \
    CKPT_ROOT="$CKPT_ROOT" DSDIR="$DSDIR" \
        bash scripts/run_train_diffusion_umi_5pt.sh --print-plan "$N_GPUS"
}
# ⚠ $N_GPUS is passed because the launcher appends `_ddp<N>` to BOTH the run name and the output
# dir whenever it is >1. Resolved at the default of 1, every path above would be the SINGLE-GPU
# one while the run below writes the ddp one -- so the printed plan would name a directory
# nothing creates, and reason (2) in this header would silently stop working: the
# already-exists SKIP would test the wrong path, and an interrupted DDP queue would re-fail
# every finished cell on the launcher's own "exists and no --resume-from" guard instead of
# skipping it.

declare -a PLAN_STATE PLAN_H PLAN_MODE PLAN_DS PLAN_NS PLAN_OUT
for cell in $CELLS; do
    state="${cell%%:*}"; horizon="${cell##*:}"
    if [[ "$state" == "$cell" || -z "$horizon" ]]; then
        echo "FATAL: cell '$cell' is not <WB_STATE>:<H> (e.g. xy3_rpy:100)" >&2; exit 1
    fi
    if ! line=$(plan_for "$state" "$horizon"); then
        echo "FATAL: cell '$cell' is not runnable:" >&2
        plan_for "$state" "$horizon" >&2 || true
        exit 1
    fi
    IFS=$'\t' read -r mode ds ns out <<<"$line"
    PLAN_STATE+=("$state"); PLAN_H+=("$horizon"); PLAN_MODE+=("$mode")
    PLAN_DS+=("$ds"); PLAN_NS+=("$ns"); PLAN_OUT+=("$out")
done

echo
echo "  diffusion_umi_5pt grid: ${#PLAN_STATE[@]} cell(s) on $N_GPUS GPU(s), SESSION=$SESSION,"
echo "                          MAX_STEPS=$MAX_STEPS, BATCH=$BATCH/GPU"
echo
printf "  %-9s %-5s %-24s %-9s %-48s %s\n" CELL H MODE STATUS "NORM STATS" "OUTPUT"
for i in "${!PLAN_STATE[@]}"; do
    status="run"
    [[ -d "${PLAN_OUT[$i]}" ]] && status="SKIP"
    [[ ! -f "${PLAN_NS[$i]}" ]] && status="$status+fit"
    printf "  %-9s %-5s %-24s %-9s %-48s %s\n" \
        "${PLAN_STATE[$i]}" "${PLAN_H[$i]}" "${PLAN_MODE[$i]}" "$status" \
        "$(basename "${PLAN_NS[$i]}")" "${PLAN_OUT[$i]}"
done
echo
echo "  dataset: ${PLAN_DS[0]}"
if [[ -f "${PLAN_DS[0]}" ]]; then
    echo "           $(du -h "${PLAN_DS[0]}" 2>/dev/null | cut -f1) on disk"
else
    echo "  ⚠ MISSING. Either fetch it (scripts/fetch_dataset.sh) or build it -- it is TWO"
    echo "    post-processes over a --hands both_head_dual conversion, in this order:"
    echo "      (UMI)  ./run_mc_to_hl_wholebody.sh <dual>.zarr.zip -o <dual>_wbabs.zarr.zip \\"
    echo "                 --cond-mode absolute --dz-anchor hand"
    echo "      (FLoRe) uv run scripts/add_foot_targets.py <dual>_wbabs.zarr.zip \\"
    echo "                 -o <dual>_wbabs_5pt.zarr.zip"
    echo "    See docs/RUNBOOK_diffusion_umi_5pt.md step 1."
fi
echo
# DISK, checked before the first cell rather than discovered by cell 3 at 04:00. Nothing prunes
# old step dirs, and `training_state.pt` (optimizer + EMA shadow + RNG) is ~80% of a
# checkpoint's bytes -- so a step dir is ~3.5 GB against the ~0.7 GB deploy payload.
PREV=""
STEP_GB=4
SAVE_EVERY=10000     # the LAUNCHER's default (--save-every 10000), not a guess
for a in "$@"; do [[ "$PREV" == "--save-every" ]] && SAVE_EVERY="$a"; PREV="$a"; done
(( SAVE_EVERY > 0 )) || SAVE_EVERY=10000
PER_CELL=$(( (MAX_STEPS / SAVE_EVERY + 1) * STEP_GB + 2 ))
NEED=$(( PER_CELL * ${#PLAN_STATE[@]} ))
HAVE=$(df -BG --output=avail "$(dirname "$CKPT_ROOT")" 2>/dev/null | tail -1 | tr -dc '0-9')
HAVE=${HAVE:-0}
echo "  disk: ~${PER_CELL} GB/cell x ${#PLAN_STATE[@]} cells = ~${NEED} GB needed under $CKPT_ROOT; ${HAVE} GB free"
if (( HAVE > 0 && HAVE < NEED )); then
    echo "  ⚠ NOT ENOUGH for the whole queue. Options, cheapest first:"
    echo "      --save-every 50000                 fewer resume points (~$(( (MAX_STEPS/50000+1)*STEP_GB+2 )) GB/cell)"
    echo "      CELLS=\"...\"                        run the grid in two passes"
    echo "      CKPT_ROOT=<bigger volume>"
fi
echo
if (( DRY )); then echo "  --dry-run: nothing launched."; exit 0; fi
if [[ ! -f "${PLAN_DS[0]}" ]]; then
    echo "FATAL: the dataset above is missing; refusing to start a queue that cannot run." >&2
    exit 1
fi

started=$(date -Is)
{ echo; echo "=== 5pt grid started $started  SESSION=$SESSION cells='$CELLS' n_gpus=$N_GPUS"; } >>"$GRID_LOG"
declare -a RESULT
for i in "${!PLAN_STATE[@]}"; do
    state="${PLAN_STATE[$i]}"; horizon="${PLAN_H[$i]}"
    cell="$state:$horizon"
    if [[ -d "${PLAN_OUT[$i]}" ]]; then
        echo "[grid] SKIP $cell — ${PLAN_OUT[$i]} already exists (resume it by hand with --resume-from)"
        RESULT+=("$cell skipped")
        continue
    fi
    # Norm stats FIRST, with the horizon pin exported INTO the fit: the quantiles depend on H and
    # flore.s1_policy.data.layout resolves FLORE_ACTION_HORIZON at IMPORT, so a pin set after the
    # process starts cannot reach the fitter's def-time defaults.
    #
    # No --wb-joint-target here (unlike the wholebody grid): that knob parameterizes the 29 JOINT
    # columns, and this preset has none -- HLDataset refuses a non-default value for a layout
    # whose action is not a joint configuration.
    if [[ ! -f "${PLAN_NS[$i]}" ]]; then
        echo "[grid] fitting norm stats for $cell -> ${PLAN_NS[$i]}"
        if ! FLORE_ACTION_HORIZON="$horizon" FLORE_FORCE_HIST_LEN=1 \
                .venv/bin/python scripts/fit_hl_norm_stats.py \
                    --zarr-paths "${PLAN_DS[$i]}" \
                    --mode "${PLAN_MODE[$i]}" \
                    --image-keys "${CAMS[@]}" \
                    --val-last-episodes "$VAL_EPS" \
                    --out "${PLAN_NS[$i]}" \
                    >>"$GRID_LOG" 2>&1; then
            echo "[grid] FAILED to fit norm stats for $cell — see $GRID_LOG; continuing"
            RESULT+=("$cell FAILED(norm-stats)")
            continue
        fi
    fi
    echo "[grid] === $cell (${PLAN_MODE[$i]}, H=$horizon) -> ${PLAN_OUT[$i]}"
    if SESSION="$SESSION" WB_STATE="$state" H="$horizon" \
       NOISE_XY_M="$NOISE_XY_M" NOISE_RPY_DEG="$NOISE_RPY_DEG" \
       BATCH="$BATCH" CKPT_ROOT="$CKPT_ROOT" DSDIR="$DSDIR" \
           bash scripts/run_train_diffusion_umi_5pt.sh "$N_GPUS" all \
               --max-steps "$MAX_STEPS" "$@"; then
        RESULT+=("$cell ok")
    else
        # Never abort the queue on one cell: a single OOM or a bad flag must not cost the others
        # their overnight slot. The summary below is what says which need a re-run.
        echo "[grid] FAILED $cell — continuing to the next cell"
        RESULT+=("$cell FAILED")
    fi
done

echo
echo "  diffusion_umi_5pt grid finished (started $started):"
for line in "${RESULT[@]}"; do echo "    $line"; done
{ echo "=== 5pt grid finished $(date -Is)"; printf '    %s\n' "${RESULT[@]}"; } >>"$GRID_LOG"
[[ " ${RESULT[*]} " == *FAILED* ]] && exit 1
exit 0
