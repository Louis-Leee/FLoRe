#!/usr/bin/env bash
# Shared plumbing for the per-preset realtime mjswan viz launchers
# (run_force_ext_realtime_viz.sh, run_hybrid_servo_realtime_viz.sh,
# run_hybrid_layerA_realtime_viz.sh). Modeled on run_upperbody_realtime_viz.sh
# but factored so the three presets share one motion set + one ONNX-selection +
# build + serve pipeline, differing only in a few env vars the caller sets before
# calling `wbt_realtime_viz_main`:
#
#   PRESET             (required) wbt-realtime-eval --preset value
#   DEFAULT_WANDB_RUN  (optional) entity/project/run_id — this launcher's CANONICAL run,
#                                 used when neither MODEL nor WANDB_RUN is set. Leave it
#                                 EMPTY for a preset with no valid checkpoint yet, so a
#                                 bare invocation asks for one instead of grabbing an
#                                 arbitrary local ONNX.
#   OUT                (optional) bundle output dir
#   PORT               (optional) serve port (default 8000)
#   EXTRA_BUILD_ARGS   (optional bash array) extra flags passed to wbt-realtime-eval
#
# ONNX source resolution (in priority order):
#   1. MODEL=/abs/path.onnx                     — pinned local file.
#   2. WANDB_RUN=entity/project/run_id          — download newest (or WANDB_MODEL)
#                                                 model_*.onnx from the W&B run.
#   3. DEFAULT_WANDB_RUN (from the wrapper)     — same download path, the launcher's
#                                                 canonical run. (Before 2026-08-04 this
#                                                 was documented but NOT read, so a bare
#                                                 invocation silently used 4.)
#   4. RUN_DIR=/abs/run (local)                 — newest model_*.onnx, snapshot-copied.
#                                                 SHARED across presets, so this is a
#                                                 last resort, not a per-preset default.
#
# NOT executed directly — sourced by the per-preset wrappers.

# REPO = training_runs -> wbt_training -> wbt_wrappers -> extensions -> src -> REPO
REPO="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Prefer the known-good hssim conda bin when present (a base-conda `wbt-realtime-eval`
# shim without mujoco/onnx would crash mid-build). Override with HSSIM_BIN=/abs/env/bin.
_HSSIM_BIN="${HSSIM_BIN:-$HOME/.holosoma_deps/miniconda3/envs/${ENV_NAME:-hssim}/bin}"
if [ -x "$_HSSIM_BIN/wbt-realtime-eval" ]; then
    export PATH="$_HSSIM_BIN:$PATH"
fi

# Clips are staged FROM here. When all 20 curated clips below are present (the held-out split
# of the training motion set) exactly those are staged; otherwise EVERY *.npz in SRC is, so any
# directory of holosoma-format clips works (SRC=/abs/dir).
SRC="${SRC:-/data/dataset/sonic_manipulation_holosoma_test}"
# Local run dir. Whether the CALLER set it explicitly is load-bearing for the ONNX
# resolution order: an EXPLICIT RUN_DIR=/abs/run is a deliberate "use this local run"
# request and must outrank the wrapper's DEFAULT_WANDB_RUN (it is a documented recipe in
# every launcher header), whereas the implicit default below is only a last resort.
# Captured BEFORE defaulting, since after the `:-` expansion the two are indistinguishable.
if [ -n "${RUN_DIR:-}" ]; then _RUN_DIR_EXPLICIT=1; else _RUN_DIR_EXPLICIT=0; fi
RUN_DIR="${RUN_DIR:-$REPO/logs/WBT-Holosoma}"

# ── 20 curated clips: 8 LOCOMOTION (walk/turn/walk-through-door) + 12 static
# MANIPULATION, all biased to grasps-against-load so the F^cmd/F^ext/wall/servo viz
# reads clearly. The walking set exercises the policy's gait + base tracking while
# still under a commanded/contact force (crate carry, single-hand-loaded gait, open
# door + walk through). idle_default is auto-injected by the builder (ideal for a
# clean F^ext / wall step-response), so it is not staged. All 20 verified present in
# the _test split (2026-07-16; walk travel 4.6–10.2s, net_xy up to 5.5 m).
_VIZ_CLIPS=(
  # ── 8 LOCOMOTION (需要走动的) ────────────────────────────────────
  lift_crate_walk_ff_loop_360_R_001__A199                        # 2H crate carry, full loop — walk WITH heavy load (10.2s)
  big_heavy_one_hand_walk_ff_start_270_R_001__A508              # 1H heavy, idle->walk start (single-hand anchor + gait)
  big_light_one_hand_walk_turn_270_R_003__A508                 # 1H walk with 270° turn (large heading change, 5.5m)
  medium_heavy_one_hand_turn_walk_360_R_001__A504              # turn-in-place then walk (idle->locomotion under load)
  grab_walk_ff_180_001__A548                                   # grab then walk forward (manip -> locomotion transition)
  medium_big_heavy_one_hand_walk_ff_stop_180_R_001__A504       # walk then STOP (deceleration under load)
  inside_door_knob_left_side_open_walk_R_001__A512            # open knob + walk through (directional force + gait)
  outside_door_handle_left_side_open_walk_switch_hand_close_R_001__A512  # open, walk, switch hand, close (rich)
  # ── 12 static MANIPULATION (strongest force/contact readers) ─────
  big_heavy_two_hands_hold_R_001__A508                          # 2H heavy static HOLD (canonical steady wall)
  big_heavy_two_hands_pick_up_front_medium_R_001__A508         # 2H heavy pick-up
  medium_big_heavy_two_hands_put_down_front_medium_R_001__A508  # 2H put-down
  medium_heavy_one_hand_pick_up_front_medium_R_001__A508       # 1H front pick-up (single-hand anchor)
  big_heavy_one_hand_pick_up_front_high_R_001__A524           # 1H high pick-up
  big_heavy_one_hand_behind_high_to_behind_low_R_001__A524    # 1H behind high->low (vertical F^ext)
  neutral_item_pick_up_floor_R_002__A542                      # floor pick-up (deep reach)
  low_button_push_002__A484                                   # literal push-against-resistance (on-theme wall)
  inside_door_knob_left_side_pull_aggresively_R_001__A512     # aggressive knob pull (dynamic load)
  item_pick_up_standing_R_001__A410                           # standing grasp (clean short single-hand)
  cellpone_put_away_stand_R_001__A423                         # small-object precision (TCP)
  Neutral_throw_ball_001__A057                                # throw (dynamic release; contrast clip)
)

_viz_fail() { echo "[realtime-viz] FAIL: $1" >&2; exit 1; }

# Resolve the python interpreter that sits next to the wbt-realtime-eval console
# script (same env), for the serve module + the W&B download helper.
_viz_resolve_py() {
    local wre; wre="$(command -v wbt-realtime-eval 2>/dev/null || true)"
    if [ -n "$wre" ]; then
        local d; d="$(dirname "$wre")"
        [ -x "$d/python" ] && { echo "$d/python"; return; }
        [ -x "$d/python3" ] && { echo "$d/python3"; return; }
    fi
    command -v python3 || command -v python || true
}

# Download the newest (or WANDB_MODEL) model_*.onnx from a W&B run into a local
# dir and echo the path. Uses wandb.Api (WANDB_BASE_URL honored by the env).
_viz_download_wandb_onnx() {
    local run_path="$1" py="$2" dest="$3"
    mkdir -p "$dest"
    "$py" - "$run_path" "$dest" "${WANDB_MODEL:-}" <<'PY'
import sys, glob, os
import wandb
run_path, dest, want = sys.argv[1], sys.argv[2], (sys.argv[3] or None)
api = wandb.Api()
run = api.run(run_path)
files = [f for f in run.files() if f.name.endswith(".onnx") and os.path.basename(f.name).startswith("model_")]
if not files:
    sys.exit(f"[realtime-viz] no model_*.onnx files on W&B run {run_path}")
if want:
    files = [f for f in files if os.path.basename(f.name) == want]
    if not files:
        sys.exit(f"[realtime-viz] {want} not found on W&B run {run_path}")
    pick = files[0]
else:
    # Newest by the trailing step number in model_<step>.onnx.
    def step(f):
        b = os.path.basename(f.name)
        try:
            return int(b.replace("model_", "").replace(".onnx", ""))
        except ValueError:
            return -1
    pick = max(files, key=step)
pick.download(root=dest, replace=True)
print(os.path.join(dest, pick.name))
PY
}

# Stage the 20 curated clips into a fresh per-run motions dir; echo the dir.
_viz_stage_motions() {
    local motions="$1"
    case "$motions" in
        "" | "/" | /*/) _viz_fail "refusing to rm -rf MOTIONS=$motions" ;;
    esac
    [ "${motions#/}" != "$motions" ] || _viz_fail "MOTIONS must be absolute, got: $motions"
    rm -rf "$motions"; mkdir -p "$motions"
    local miss=0 c
    for c in "${_VIZ_CLIPS[@]}"; do
        [ -f "$SRC/$c.npz" ] || miss=$((miss + 1))
    done
    if [ "$miss" -eq 0 ]; then
        for c in "${_VIZ_CLIPS[@]}"; do cp -f "$SRC/$c.npz" "$motions/$c.npz"; done
        local staged; staged="$(find "$motions" -maxdepth 1 -name '*.npz' | wc -l)"
        [ "$staged" -eq "${#_VIZ_CLIPS[@]}" ] || _viz_fail "staged $staged != ${#_VIZ_CLIPS[@]} clips."
    else
        echo "  $miss of the ${#_VIZ_CLIPS[@]} curated clips are not in SRC=$SRC; staging every *.npz there." >&2
        local f n=0
        for f in "$SRC"/*.npz; do
            [ -f "$f" ] || continue
            cp -f "$f" "$motions/"; n=$((n + 1))
        done
        [ "$n" -gt 0 ] || _viz_fail "no *.npz clips in SRC=$SRC (set SRC=/abs/dir of holosoma-format clips)."
    fi
    echo "$motions"
}

wbt_realtime_viz_main() {
    [ -n "${PRESET:-}" ] || _viz_fail "PRESET not set by the wrapper."
    OUT="${OUT:-$REPO/dist/wbt_realtime_eval_$(echo "$PRESET" | tr -c 'a-zA-Z0-9' '_')}"
    PORT="${PORT:-8000}"

    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        command -v wbt-realtime-eval >/dev/null 2>&1 \
            || _viz_fail "wbt-realtime-eval not found. conda activate hssim (or set HSSIM_BIN)."
        local wre_py; wre_py="$(_viz_resolve_py)"
        [ -n "$wre_py" ] && "$wre_py" -c "import mujoco, onnx, onnxruntime, mjswan, wbt_realtime_eval" \
            || _viz_fail "the wbt-realtime-eval env cannot import mujoco/onnx/mjswan (activate hssim or set HSSIM_BIN)."
        [ -d "$SRC" ] || _viz_fail "SRC motion dir not found: $SRC"
        if command -v wbt-realtime-eval-patch-mjswan >/dev/null 2>&1; then
            echo "[realtime-viz] ensuring mjswan runtime patches are applied (idempotent)..."
            wbt-realtime-eval-patch-mjswan || _viz_fail "mjswan patch step failed — see above."
        fi
        echo "[realtime-viz] preflight OK."
    fi
    local py; py="$(_viz_resolve_py)"

    # ── Resolve the ONNX ──────────────────────────────────────────────────────
    #   MODEL > WANDB_RUN > explicit RUN_DIR > DEFAULT_WANDB_RUN > implicit RUN_DIR
    # All three CALLER-supplied sources (MODEL / WANDB_RUN / an explicit RUN_DIR) outrank
    # the wrapper's DEFAULT_WANDB_RUN — each is a documented recipe in every launcher
    # header, so honoring the canonical run must not make "RUN_DIR=/abs/run" unreachable.
    #
    # DEFAULT_WANDB_RUN is the wrapper's canonical run, and until 2026-08-04 it was
    # documented-but-never-read: a bare `bash run_<x>_realtime_viz.sh` skipped it and fell
    # through to "newest model_*.onnx anywhere under logs/WBT-Holosoma", which is shared by
    # every preset — so it could silently pick a checkpoint from a DIFFERENT experiment and
    # be caught only (if at all) by the normalizer dim check. Honoring it makes a bare
    # invocation reproducible per launcher.
    local model=""
    local _wandb_run=""
    # An explicit RUN_DIR suppresses the DEFAULT_WANDB_RUN fallback (but never WANDB_RUN,
    # which the caller passed just as deliberately and which stays higher priority).
    if [ -n "${WANDB_RUN:-}" ]; then
        _wandb_run="$WANDB_RUN"
    elif [ "$_RUN_DIR_EXPLICIT" != "1" ]; then
        _wandb_run="${DEFAULT_WANDB_RUN:-}"
    fi
    if [ -n "${MODEL:-}" ]; then
        [ -f "$MODEL" ] || _viz_fail "MODEL not found: $MODEL"
        model="$MODEL"
        echo "=== using pinned MODEL=$model ==="
    elif [ -n "$_wandb_run" ]; then
        if [ -z "${WANDB_RUN:-}" ]; then
            echo "=== no MODEL/WANDB_RUN set; using this launcher's canonical DEFAULT_WANDB_RUN ==="
        fi
        echo "=== pulling ONNX from W&B run $_wandb_run (${WANDB_MODEL:-newest model_*.onnx}) ==="
        [ -n "$py" ] || _viz_fail "no python interpreter found for the W&B download."
        local _dl_dir="$REPO/logs/wandb_downloads/$(echo "$_wandb_run" | tr '/' '_')"
        model="$(_viz_download_wandb_onnx "$_wandb_run" "$py" "$_dl_dir" | tail -n 1)"
        [ -f "$model" ] || _viz_fail "W&B download did not yield a model file (got: $model)"
        echo "    downloaded: $model"
    else
        # Either the caller set RUN_DIR explicitly (a deliberate local-run request), or
        # nothing was set AND this launcher pins no canonical DEFAULT_WANDB_RUN (deliberate
        # for a preset with no valid checkpoint yet — e.g. the R2 TCP-unified twin before
        # its retrain).
        if [ "$_RUN_DIR_EXPLICIT" = "1" ]; then
            echo "=== using explicit RUN_DIR=$RUN_DIR (newest model_*.onnx) ==="
        else
            # The IMPLICIT default dir is SHARED across presets, so anything found here may
            # belong to a different experiment. setup_builder's obs-dim + hand-geometry
            # checks are what stop a wrong one from actually being served.
            echo "=== no MODEL / WANDB_RUN / DEFAULT_WANDB_RUN; falling back to newest local ckpt under $RUN_DIR ==="
            echo "    NOTE: that dir is shared across presets — pass WANDB_RUN=... or MODEL=... to be explicit."
        fi
        [ -d "$RUN_DIR" ] \
            || _viz_fail "RUN_DIR does not exist: $RUN_DIR (and no MODEL / WANDB_RUN set$([ "$_RUN_DIR_EXPLICIT" = 1 ] || echo "; this launcher pins no DEFAULT_WANDB_RUN")). Pass WANDB_RUN=entity/project/run_id or MODEL=/abs/path.onnx."
        local latest=""
        while IFS= read -r f; do latest="$f"; done < <(
            find "$RUN_DIR" -maxdepth 2 -name 'model_*.onnx' -printf '%T@ %p\n' 2>/dev/null \
                | sort -n | awk '{ $1=""; sub(/^ /,""); print }')
        [ -n "$latest" ] || _viz_fail "no model_*.onnx under $RUN_DIR (set MODEL=/abs/path or WANDB_RUN=...)."
        local step snap; step="$(basename "$latest" .onnx)"
        snap="$(dirname "$latest")/realtime_${PRESET}_${step#model_}"
        mkdir -p "$snap"; model="$snap/$step.onnx"
        echo "=== newest local ckpt: $latest ==="
        cp -f "$latest" "$model.tmp"; mv -f "$model.tmp" "$model"
    fi

    # Per-PRESET motions dir so the three launchers can build concurrently without
    # racing on a shared dir (each does rm -rf + re-stage of the same 20 clips).
    local _motions_slug; _motions_slug="$(echo "$PRESET" | tr -c 'a-zA-Z0-9' '_')"
    local motions; motions="$(_viz_stage_motions "$REPO/logs/realtime_motions_viz_20_$_motions_slug")"
    echo "=== [$(date +%H:%M:%S)] building '$PRESET' bundle (strip ONNX + bake $(find "$motions" -maxdepth 1 -name '*.npz' | wc -l) + idle_default + WASM) ==="
    wbt-realtime-eval \
        --preset "$PRESET" \
        --model-path "$model" \
        --motion-dir "$motions" \
        --output-dir "$OUT" \
        "${EXTRA_BUILD_ARGS[@]}" \
        --no-launch
    echo "=== bundle built: $OUT ==="

    if [ "${NO_LAUNCH:-0}" = "1" ]; then
        echo "Build-only (NO_LAUNCH=1). Re-open later with the COOP/COEP server:"
        echo "    \"$py\" -m wbt_realtime_eval.serve_bundle \"$OUT\" --port $PORT"
        echo "Then, from your laptop:  ssh -L $PORT:localhost:$PORT <this-box>"
        return 0
    fi
    echo "=== [$(date +%H:%M:%S)] serving on http://localhost:$PORT  (Ctrl-C to stop) ==="
    echo "    ssh -L $PORT:localhost:$PORT <this-box>  then open http://localhost:$PORT"
    exec "$py" -m wbt_realtime_eval.serve_bundle "$OUT" --port "$PORT"
}
