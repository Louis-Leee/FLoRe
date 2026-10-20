#!/usr/bin/env bash
# The diffusion_umi_wholebody GEOMETRIC-STATE x HORIZON ablation, as a QUEUE.
#
# Each cell is one full `run_train_diffusion_umi_wholebody.sh` invocation on ALL the GPUs you
# give it, run one after another. This script exists for exactly three reasons, each of which
# has cost a wasted queue slot when done by hand:
#
#   1. EVERY CELL NEEDS ITS OWN NORM STATS. `mode` is in the exact-equality dataset_knobs and
#      the action quantiles depend on the horizon, so a cell cannot borrow another's file --
#      and the trainer's refusal arrives AFTER the queue has moved on. This fits any missing
#      one first, with FLORE_ACTION_HORIZON exported before the fit (the pin is resolved at
#      import, so exporting it afterwards silently fits the wrong horizon).
#   2. A CELL THAT ALREADY HAS AN OUTPUT DIR IS SKIPPED, not overwritten and not fatal. So
#      re-running after an interruption resumes the QUEUE instead of dying on cell 1 -- the
#      launcher itself (correctly) refuses to write into an existing dir without --resume-from.
#   3. THE PLAN IS PRINTABLE. `--dry-run` shows every cell's mode, horizon, sigma, stats file
#      and output dir without touching a GPU, which is the only cheap way to notice that e.g.
#      SESSION does not store the horizon you asked for.
#
# THE GRID (the default CELLS, in the order they run):
#
#   cell            state                          state_dim   asks
#   xy_rpy:100      [rel_xy 4][rpy 6][Lg][Rg]         12        the candidate
#   none:100        [Lg][Rg]                           2        the CONTROL: does joint-space
#                                                              proprioception help at all?
#   xy:100          [rel_xy 4][Lg][Rg]                 6        is POSITION the useful half?
#   rpy:100         [rpy 6][Lg][Rg]                    8        is ATTITUDE the useful half?
#   ...then the same four at H=200.
#
# H=100 runs first as a block on purpose: the four states at ONE horizon already answer "which
# observation", and stopping after cell 4 leaves a complete 4-way comparison rather than half of
# two. Split the grid across two boxes with CELLS if you want both horizons at once.
#
# ⚠️ WB_STATE=joints (the shipped 29-joint state) is NOT in the default grid, because it is the
# run you already have at H=50 and it is the mode the deploy path still accepts. Add it
# explicitly (CELLS="joints:100 ...") if you want it re-run at the new horizons.
#
# COST. Measured here at 160 samples/s = 6.7 steps/s at batch 24 on ONE RTX PRO 5000, i.e.
# ~4 h per 100k steps. DDP does not change steps/s (each rank runs its own batch), so budget
# ~4 h per H=100 cell and somewhat more at H=200 (a 4x longer UNet time axis behind the same
# two ViT forwards), and ~1.5-2 days for all eight. TIME THE FIRST CELL before trusting that.
# For a screening pass use MAX_STEPS=40000 -- the val curves separate long before 100k, and a
# cell can be extended later with --resume-from.
#
# Usage
#   scripts/run_train_wb_geom_grid.sh --dry-run                      # print the plan
#   scripts/run_train_wb_geom_grid.sh 8                              # the 8-cell grid, 8 GPUs
#   CELLS="xy_rpy:100 none:100" scripts/run_train_wb_geom_grid.sh 8  # just two cells
#   MAX_STEPS=40000 scripts/run_train_wb_geom_grid.sh 8              # screening pass
#
# Env (all forwarded to the per-cell launcher unless noted):
#   SESSION      dataset key. Default 0902_merged3_h200 -- the only registered entry that
#                stores 200 steps, so the default grid needs no other value.
#   CELLS        space-separated "<WB_STATE>:<H>" pairs. Default: the 8-cell grid above.
#   MAX_STEPS    per cell. Default 100000 (the launcher's own default).
#   BATCH        per GPU. Default 24 (the launcher's).
#   NOISE_XY_M / NOISE_RPY_DEG   state jitter. Defaults 0.010 m / 2.0 deg, and the launcher
#                zeroes whichever the cell's mode cannot carry.
#   WB_TARGET    absolute (default) | delta. Applies to every cell.
#   CKPT_ROOT / DSDIR   as the launcher.
#   GRID_LOG     queue log. Default logs/wb_geom_grid.log
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; shift; fi
N_GPUS=${1:-8}
shift || true

SESSION=${SESSION:-0902_merged3_h200}
CELLS=${CELLS:-"xy_rpy:100 none:100 xy:100 rpy:100 xy_rpy:200 none:200 xy:200 rpy:200"}
MAX_STEPS=${MAX_STEPS:-100000}
BATCH=${BATCH:-24}
WB_TARGET=${WB_TARGET:-absolute}
NOISE_XY_M=${NOISE_XY_M:-0.010}
NOISE_RPY_DEG=${NOISE_RPY_DEG:-2.0}
CKPT_ROOT=${CKPT_ROOT:-/data/flore/checkpoints}
DSDIR=${DSDIR:-third_party/UMI_collection/data/datasets}
GRID_LOG=${GRID_LOG:-logs/wb_geom_grid.log}
VAL_EPS=4
CAMS=(camera0_rgb camera1_rgb)
mkdir -p logs

# The dataset path and the stats/run naming are DERIVED BY THE LAUNCHER, so asking it rather
# than re-deriving them here is what stops the two drifting apart -- the trap the repo's
# `dataset_knobs` discipline exists for. `--print-plan` makes the launcher resolve its own
# variables and stop before touching a GPU.
plan_for() {  # $1=WB_STATE $2=H  -> prints "MODE<TAB>DS<TAB>NS<TAB>OUT"
    SESSION="$SESSION" WB_TARGET="$WB_TARGET" WB_STATE="$1" H="$2" \
    NOISE_XY_M="$NOISE_XY_M" NOISE_RPY_DEG="$NOISE_RPY_DEG" \
    CKPT_ROOT="$CKPT_ROOT" DSDIR="$DSDIR" \
        bash scripts/run_train_diffusion_umi_wholebody.sh --print-plan "$N_GPUS"
}
# ⚠ $N_GPUS is passed because the launcher appends `_ddp<N>` to BOTH the run name and the output
# dir whenever it is >1. Resolved at the default of 1, every path above would be the SINGLE-GPU
# one while the run below writes the ddp one -- so the printed plan would name a directory
# nothing creates, and the already-exists SKIP would test the wrong path, making an interrupted
# DDP queue re-fail every finished cell on the launcher's own "exists and no --resume-from"
# guard instead of skipping it.

declare -a PLAN_STATE PLAN_H PLAN_MODE PLAN_DS PLAN_NS PLAN_OUT
for cell in $CELLS; do
    state="${cell%%:*}"; horizon="${cell##*:}"
    if [[ "$state" == "$cell" || -z "$horizon" ]]; then
        echo "FATAL: cell '$cell' is not <WB_STATE>:<H> (e.g. xy_rpy:100)" >&2; exit 1
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
echo "  wb_geom grid: ${#PLAN_STATE[@]} cell(s) on $N_GPUS GPU(s), SESSION=$SESSION,"
echo "                WB_TARGET=$WB_TARGET, MAX_STEPS=$MAX_STEPS, BATCH=$BATCH/GPU"
echo
printf "  %-8s %-5s %-24s %-6s %-46s %s\n" CELL H MODE STATUS "NORM STATS" "OUTPUT"
for i in "${!PLAN_STATE[@]}"; do
    status="run"
    [[ -d "${PLAN_OUT[$i]}" ]] && status="SKIP"
    [[ ! -f "${PLAN_NS[$i]}" ]] && status="$status+fit"
    printf "  %-8s %-5s %-24s %-6s %-46s %s\n" \
        "${PLAN_STATE[$i]}" "${PLAN_H[$i]}" "${PLAN_MODE[$i]}" "$status" \
        "$(basename "${PLAN_NS[$i]}")" "${PLAN_OUT[$i]}"
done
echo
echo "  dataset: ${PLAN_DS[0]}"
[[ -f "${PLAN_DS[0]}" ]] || echo "  ⚠ MISSING — fetch it first (scripts/fetch_dataset.sh, or build it: docs/RUNBOOK_wb_geom_ablation.md step 1)"
echo
# DISK, checked before the first cell rather than discovered by cell 6 at 04:00. Nothing prunes
# old step dirs, and `training_state.pt` (optimizer + EMA shadow + RNG) is ~80% of a
# checkpoint's bytes -- so a step dir is ~3.5 GB against the 706 MB deploy payload, and the
# default cadence writes max_steps/save_every of them plus best_overall + best_force.
PREV=""
STEP_GB=4
# The LAUNCHER's default, not a guess: it passes --save-every 10000, so the honest
# estimate is 10 step dirs per 100k-step cell. An explicit --save-every on this
# script's command line overrides it (it is forwarded to the launcher).
SAVE_EVERY=10000
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
    echo "    (deploy weights are only 0.7 GB each; the rest is training_state.pt, which only"
    echo "     --resume-from needs. scripts/mirror_final_to_nfs.sh moves finished runs off.)"
fi
echo
if (( DRY )); then echo "  --dry-run: nothing launched."; exit 0; fi
if [[ ! -f "${PLAN_DS[0]}" ]]; then
    echo "FATAL: the dataset above is missing; refusing to start a queue that cannot run." >&2
    exit 1
fi

started=$(date -Is)
{ echo; echo "=== wb_geom grid started $started  SESSION=$SESSION cells='$CELLS' n_gpus=$N_GPUS"; } >>"$GRID_LOG"
declare -a RESULT
for i in "${!PLAN_STATE[@]}"; do
    state="${PLAN_STATE[$i]}"; horizon="${PLAN_H[$i]}"
    cell="$state:$horizon"
    if [[ -d "${PLAN_OUT[$i]}" ]]; then
        echo "[grid] SKIP $cell — ${PLAN_OUT[$i]} already exists (resume it by hand with --resume-from)"
        RESULT+=("$cell skipped")
        continue
    fi
    # Norm stats FIRST, and with the horizon pin exported into the fit: the quantiles depend on
    # H, and flore.s1_policy.data.layout resolves FLORE_ACTION_HORIZON at IMPORT, so a pin set
    # after the process starts cannot reach the fitter's def-time defaults.
    if [[ ! -f "${PLAN_NS[$i]}" ]]; then
        echo "[grid] fitting norm stats for $cell -> ${PLAN_NS[$i]}"
        if ! FLORE_ACTION_HORIZON="$horizon" FLORE_FORCE_HIST_LEN=1 \
                .venv/bin/python scripts/fit_hl_norm_stats.py \
                    --zarr-paths "${PLAN_DS[$i]}" \
                    --mode "${PLAN_MODE[$i]}" \
                    --image-keys "${CAMS[@]}" \
                    --val-last-episodes "$VAL_EPS" \
                    --wb-joint-target "$WB_TARGET" \
                    --out "${PLAN_NS[$i]}" \
                    >>"$GRID_LOG" 2>&1; then
            echo "[grid] FAILED to fit norm stats for $cell — see $GRID_LOG; continuing to the next cell"
            RESULT+=("$cell FAILED(norm-stats)")
            continue
        fi
    fi
    echo "[grid] === $cell (${PLAN_MODE[$i]}, H=$horizon) -> ${PLAN_OUT[$i]}"
    if SESSION="$SESSION" WB_TARGET="$WB_TARGET" WB_STATE="$state" H="$horizon" \
       NOISE_XY_M="$NOISE_XY_M" NOISE_RPY_DEG="$NOISE_RPY_DEG" \
       BATCH="$BATCH" CKPT_ROOT="$CKPT_ROOT" DSDIR="$DSDIR" \
           bash scripts/run_train_diffusion_umi_wholebody.sh "$N_GPUS" all \
               --max-steps "$MAX_STEPS" "$@"; then
        RESULT+=("$cell ok")
    else
        # Never abort the queue on one cell: a single OOM or a bad flag must not cost the other
        # seven their overnight slot. The summary below is what says which ones need a re-run.
        echo "[grid] FAILED $cell — continuing to the next cell"
        RESULT+=("$cell FAILED")
    fi
done

echo
echo "  wb_geom grid finished (started $started):"
for line in "${RESULT[@]}"; do echo "    $line"; done
{ echo "=== wb_geom grid finished $(date -Is)"; printf '    %s\n' "${RESULT[@]}"; } >>"$GRID_LOG"
# Non-zero if ANY cell failed, so a wrapper (cron, tmux exit trap) can notice.
[[ " ${RESULT[*]} " == *FAILED* ]] && exit 1
exit 0
