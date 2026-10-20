#!/usr/bin/env bash
# Launch the ONLINE FIVE-POINT PLANNER: diffusion_umi_5pt -> the `5pt` low level, direct.
#
# The deploy sibling of ./scripts/run_infer_s1_5pt.sh (the offline page) and the completion-free
# twin of `./scripts/run_hl_serving_mc.sh --low-level 5pt`:
#
#   run_hl_serving_mc.sh --low-level 5pt   3 EE poses --decode--> Motion_Completion
#                                            --> qpos(29) --FK--> 5 keypoints --> :8770 5pt
#   THIS SCRIPT                            5 keypoints straight off the policy --> :8770 5pt
#
# There is no offline completion pass to substitute either (unlike run_hl_serving_wholebody.sh,
# whose joints came from one): a 5pt dataset is a wholebody dataset PLUS eight FK'd foot arrays
# (scripts/add_foot_targets.py), so the policy learned every slot the wire carries. The LOW LEVEL
# is the same one either way: MODE=5pt-unified-anneal, W&B pq9shxs7 (ft60000-r2, model_62999;
# superseded tqnoooo6 on 2026-09-12).
#
# ⚠️ T4's MODE must be EXACTLY 5pt-unified-anneal and nothing cross-checks it — the RecordTap frame
# carries no preset name. A mismatch is not dangerous (the low level's width guard drops every
# frame of the wrong shape and the robot holds its ready pose) but it IS silent, so "nothing moves"
# reads as "T4 is the wrong family" before it reads as anything else.
#
# WHY A SEPARATE LAUNCHER (and a separate planner). This checkpoint's state is a pelvis-frame
# GEOMETRY, not a wrt_start rotation, so there is nothing for hl_serving.py's WireMapper or
# hl_serving_wholebody.py's joint latch to do; and it predicts five keypoints, so there is no
# completion to run and no --mc-foot-ik decision. What IS shared lives in importable modules and is
# imported, not copied (hl_dense's KeypointReadout/KeypointClutch/FivePointWire, hl_online's rings,
# live_robot's RecordTap parse, hl_serving's sensor plumbing, ready_pose's anchor) — see
# scripts/hl_serving_5pt.py's header.
#
# RECORDED-EPISODE REPLAY (2026-09-19): `REPLAY=1 TASK=wipe_board_all ./scripts/run_hl_serving_5pt.sh`
# adds hl_serving.py's replay to THIS page — pick one of the checkpoint's own training episodes in the
# "Replay recorded episode" folder (the episode's first recorded frame is shown per camera) and ▶▶
# plays the RECORDED hand pose(s), the RECORDED jaw and the RECORDED Bota wrench (as F^cmd, -F_meas
# through the same switch and gate) on the same wire, clutch, STOP, hotkeys and ⏺ run recorder (F/T +
# camera streams included). REPLAY_ZARR=<path> replays another converted session; REPLAY_EPISODE=N,
# REPLAY_MODE=sequential|openloop. The model stays loaded and "Chunk SOURCE" switches back to it live.
#
# LIVE CAMERAS (2026-09-20): the page's "📷 Live cameras" folder shows the CURRENT frame of EVERY
# configured D405 (L :8773 and R :8777 — camera R is display-only under a left_only checkpoint, never
# observed / self-checked / recorded) plus the exact 224x224 crop the model is fed per view it reads,
# in natural colours, 4 Hz. Flags: --live-camera-hz (0 = off), --live-camera-width. The same day the
# ingest stopped flipping the frame to BGR (the model had seen a red apple as blue; runbook §6).
#
# PREREQUISITES — five terminals, in this order. The planner probes every one at startup and names
# the missing one rather than starting degraded.
#
#   T1  BOTH Botas, 6-DoF, :8771       cd .../force_deploy && bash 2_ft_bridge.sh --pipeline
#   T2  camera L :8773  AND  R :8777   this checkpoint reads TWO views; a missing one is fed to its
#                                      encoder group as a GREY frame and nothing downstream would
#                                      say so, which is why the planner refuses instead.
#   T3  both jaws                      LEFT=1 RIGHT=1 KEYBOARD=0 bash 5_dual_gripper_keyboard.sh
#                                      (LEFT can0 -> :8784, RIGHT can1 -> :8785)
#   T4  the LOW LEVEL, ABSOLUTE + RecordTap:
#         cd <repo>/third_party/holosoma_extensions/force_deploy && \
#           MODE=5pt-unified-anneal HOLOSOMA_VIVE_RELATIVE=0 RECORD=1 FORCE_CMD_WIRE=1 \
#           bash 3_run_policy_vive_force.sh
#       * The low-level ONNX lives under third_party/holosoma_extensions/logs/WBT-Holosoma/<run>/
#         (train one with wbt_training/training_runs/train_hybrid_unified_5pt_dense_8gpu.sh).
#       * HOLOSOMA_VIVE_RELATIVE=0 (ABSOLUTE) is REQUIRED: the planner publishes absolute
#         pelvis-frame keypoints. RELATIVE would add them as a delta on top of the held pose.
#       * RECORD=1 is strongly recommended, not cosmetic: it is the MEASURED reference the carried
#         keypoint command is leak-bounded against. Without it the clutch FREEZES at every chunk
#         boundary rather than integrating blind — safe, but the wire stops advancing.
#       * ⚠️ Terminal 1 of the VIVE bridge is NOT started here. The planner IS the producer on
#         :8770; running the tracker bridge as well means two writers on one port.
#   T5  this script.
#
# OPERATOR SEQUENCE, once all five are up: joystick **A** (the low level ramps to and HOLDS its
# neutral/READY pose) -> Predict in the page -> joystick **start** to ARM -> Run. That "A first"
# order is the whole design: the chunk is DECODED against that pose and the clutch makes step 0 a
# bit-exact no-op on it, so the policy's output is a displacement from where the robot actually is.
#
# ⚠️ SAFETY — the 5pt low level is a whole-body BALANCE controller reading all five points, and two
# of them are FEET. First runs: gantry-suspended, hand on the hardware E-stop, force switch OFF
# (--no-force-cmd). The default keypoint clutch (--post-anchor command) makes the first executed
# step a no-op on every point, so the body starts from where it IS rather than at the model's
# absolute claim — measured 59-63 mm away on the hands, against a 70 mm pre-arm tolerance. Above
# 40 N of measured force the low level stops itself (FORCE_ESTOP_N, the joystick-B path).
#
# Usage:
#   ./scripts/run_hl_serving_5pt.sh --checkpoint checkpoints/umi_5pt_rotate_box/h50
#   ./scripts/run_hl_serving_5pt.sh --checkpoint <ckpt> --no-force-cmd          # first run
#   CARRY=upper ./scripts/run_hl_serving_5pt.sh --checkpoint <ckpt>             # feet absolute
#   PORT=8092 ./scripts/run_hl_serving_5pt.sh --checkpoint <ckpt> --post-anchor off
#   CADENCE=stream ./scripts/run_hl_serving_5pt.sh --checkpoint <ckpt>   # HATO async, page-switchable
#   ./scripts/run_hl_serving_5pt.sh --checkpoint <ckpt> --seed 7   # one observation, one chunk
#   TASK=wipe_board H=30 ./scripts/run_hl_serving_5pt.sh --no-force-cmd
#                                   # a LEFT-ONLY (plain diffusion_umi) checkpoint — see below
#   TASK=rotate_box      ./scripts/run_hl_serving_5pt.sh --no-force-cmd
#   TASK=push_cart H=100 ./scripts/run_hl_serving_5pt.sh --no-force-cmd
#   TASK=push_gantry     ./scripts/run_hl_serving_5pt.sh --no-force-cmd
#   TASK=push_gantry_m2  ./scripts/run_hl_serving_5pt.sh --no-force-cmd
#   TASK=push_gantry_new ./scripts/run_hl_serving_5pt.sh --no-force-cmd   # the 2026-09-13 sitting, its own task
#   TASK=wipe_board_new H=30 ./scripts/run_hl_serving_5pt.sh --no-force-cmd   # wipe_board re-recorded 2026-09-13, its own task (H=30|50)
#   TASK=wipe_board_all   ./scripts/run_hl_serving_5pt.sh   # 2026-09-17 DDP8, every board-wiping sitting (1364 ep); H=30 = the 2026-09-19 DDP4 run
#   TASK=wipe_board_all_act [H=30] ./scripts/run_hl_serving_5pt.sh   # the SAME 1364 ep trained as ACT (live weights, 2026-09-19); --algo act is read off config.json
#   TASK=rotate_box_all [H=30] ./scripts/run_hl_serving_5pt.sh   # 2026-09-17 DDP8 diffusion_umi_3TRACKER (both_hands_head_dual, 395 ep; H=30 = the 2026-09-19 DDP4 run):
#                                   # family keypoint_subset — L/R/torso from the policy, both FEET HELD
#                                   # a TASK shorthand for --checkpoint: the registry's own local
#                                   # dir, checkpoints/umi_5pt_<TASK>/h<H> (scripts/fetch_checkpoint.sh
#                                   # <TASK>_5pt_h<H> pulls it). Nothing else is task-specific here —
#                                   # the same planner serves every fivepoint_geom_* checkpoint, and
#                                   # the startup log + the page's `state range` line say whether the
#                                   # ready pose is inside THAT checkpoint's training range.
#                                   # ⚠️ A plain task name is the task's CURRENT dataset, and the
#                                   # banner's `trained on:` line names the zarr it came from. Since
#                                   # 2026-09-12: push_gantry = the 389-ep merged3 pair, rotate_box =
#                                   # merged6 (249 ep), push_cart = merged13 (276 ep), wipe_board =
#                                   # the 291-ep left_only pair (H=30|50, no 100). The superseded
#                                   # runs keep their own names for an A/B — push_gantry_m2 (276 ep),
#                                   # rotate_box_base (183 ep), push_cart_base (232 ep) — each a
#                                   # registry row <name>_5pt_h<H>.
#
#   # The WALK SKILL panel is always on the page: the bundled walk-with-box clip's two FEET on the
#   # wire under the unchanged upper body, one toggle button (docs/RUNBOOK_hl_serving_5pt.md §4,
#   # "The walk skill"). ⚠️ The low level must exempt the feet from its 5 mm/tick slew cap —
#   # 3_run_policy_vive_force.sh does (SLEW_FEET=0, its default); the page reads whether it did
#   # off RecordTap and says so in 'clip demand'. --walk-clip PATH swaps the clip file.
#
# A ONE-HAND CHECKPOINT ON THIS WIRE. The plain `diffusion_umi` preset (layout left_only: the LEFT
# hand's pose + jaw + wrench, ONE camera = the LEFT D405 on :8773, ONE Bota, ONE jaw) is served by
# the SAME planner: the four wire slots it does not predict (RIGHT hand, torso, both feet) are HELD
# at the READY POSE — the pose the low level holds after A, which is also the pre-arm gate's own
# reference — for the whole run, exactly as run_hl_serving.sh's WireMapper held RIGHT + HEAD on the
# 3-point wire. The clutch publishes their base, a Predict never moves them, their f_cmd is a hard
# zero, and only the LEFT jaw/Bota/camera are read (camera R and the RIGHT jaw bridge may keep
# running; they are simply unread). Nothing is a flag: hl_5pt.check_layout names the family from
# hl_data.mode, and both_hands / both_hands_head[_dual] follow the same rule with fewer held slots.
# A fivepoint_geom_* checkpoint is served byte-identically to before.
#
# ⚠️ A chunk is a DRAW, not a function of the observation. Two Predicts from a bit-identical
# observation differ by the sampler seed alone by a measured 6.7-18.6 mm at the worst step of a
# 65-88 mm reach (push_cart H=100) — an order of magnitude more than the camera (0.1-0.3 mm) or the
# force window at 1% of range (0.6-1.1 mm) contribute. `Reset to ready pose` returns the seed along
# with the anchor, so "back to the start" is bit-exact; pressing Predict again advances it, so
# re-sampling a chunk you do not like is still one press. `--seed N` pins it for an A/B. The value
# is on the page as `sampler seed` and in every `staged:` message.
#
# Env knobs:
#   FLORE_PY   python (default: <repo>/.venv/bin/python)
#   PORT      viser port (default: 8092 — distinct from 8088/8090/8091 so all four coexist)
#   DEVICE    torch device (default: cuda)
#   TASK, H   a shorthand for --checkpoint: checkpoints/umi_5pt_${TASK}/h${H} (H defaults to 50),
#             the local dir scripts/fetch_checkpoint.sh's registry uses for the per-task runs
#             (rotate_box, push_cart, push_gantry, push_gantry_new, wipe_board, wipe_board_new = each
#             task's CURRENT dataset; rotate_box_base, push_cart_base, push_gantry_m2 = the superseded
#             runs). H is 50 or 100 (wipe_board / wipe_board_new: 30 or 50). Refused with the exact fetch command if the dir is not
#             on disk; an explicit --checkpoint wins.
#   WALK_CLIP a shorthand for --walk-clip (an extracted feet NPZ or a raw holosoma motion NPZ)
#   CARRY     all|upper — a shorthand for --carry, so a first-run setting can be exported once
#   OBS_LEAK  metres (default 0.05) / OBS_LEAK_DEG (default 10) — shorthands for --obs-leak-m
#             and --obs-leak-deg: how far the COMMANDED anchor, which IS this family's whole
#             observation (obs_mode=last_target), may run ahead of the MEASURED keypoints before it
#             is pulled back. The same leash KeypointClutch.sync gives the PUBLISHED base, which
#             the observation never had — the asymmetry that folded the body twice. 0 = off.
#   OOD_STOP  band widths (default 0, OFF) — a shorthand for --ood-stop, an optional BACKSTOP that
#             ENDS a ▶▶ run when the observation it would sample from sits that far outside the
#             checkpoint's own [q01, q99]. 0.25 is the measured threshold; 1.0 a loose backstop.
#             The wire holds the LAST EXECUTED step, so ▶▶ continues from there with no Reset.
#   CADENCE   sequential|stream — which cadence ▶▶ STARTS on. The page's '▶▶ cadence' dropdown is
#             the live switch either way; `sequential` (the default) is the reviewed chunked path,
#             `stream` is HATO's asynchronous one. See docs/RUNBOOK_hl_serving_5pt.md §4.
#   ENSEMBLE  avg|act|new|old — [stream] how the overlapping chunks become one command. `avg` is
#             both HATO's default and the measured best on jerk; live on the page.
#   FLORE_ACTION_HORIZON  DO NOT SET IT — derived from the checkpoint, exactly as the sibling
#             launchers do, because layout.py resolves it at IMPORT time.
set -eo pipefail

FLORE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLORE_PY="${FLORE_PY:-$FLORE_ROOT/.venv/bin/python}"
PORT="${PORT:-8092}"
DEVICE="${DEVICE:-cuda}"

# ⚠️ Knobs that belong to the SIBLING launchers. Refused with a pointer rather than ignored: each
# is load-bearing over there, and an operator who sets one is entitled to assume it did something.
for _v in ANCHOR ANCHOR_EPS; do
  if [ -n "${!_v:-}" ]; then
    echo "[run] FATAL: $_v=${!_v} is a run_hl_serving.sh (left-only) knob and does nothing here." >&2
    echo "[run]   There it moves the DECODE anchor. Here the decode anchor is the pose the robot" >&2
    echo "[run]   holds after A and is also the pre-arm gate's reference, so it is fixed. The" >&2
    echo "[run]   equivalent sits one stage later, on the quantity actually published:" >&2
    echo "[run]       --post-anchor command  [--post-anchor-eps-m 0.05]   (already the DEFAULT)" >&2
    exit 2
  fi
done
if [ -n "${LEGS:-}" ]; then
  echo "[run] FATAL: LEGS=$LEGS is a run_hl_serving_wholebody.sh knob and does nothing here." >&2
  echo "[run]   That wire publishes 29 JOINT targets, so it can hold the 12 leg ones. This wire" >&2
  echo "[run]   publishes 5 KEYPOINTS and the low level solves the joints itself — which is the" >&2
  echo "[run]   whole reason this preset exists. The nearest thing is --carry upper, which lets" >&2
  echo "[run]   the two FOOT keypoints pass through absolutely instead of being re-based." >&2
  exit 2
fi
for _v in MOTION_COMPLETION_DIR MOTION_COMPLETION_PY MOTION_COMPLETION_CKPT; do
  if [ -n "${!_v:-}" ]; then
    echo "[run] NOTE: $_v is set but this planner runs NO motion completion." >&2
    echo "[run]   A five-point policy predicts every slot the wire carries. If you meant the" >&2
    echo "[run]   3-tracker deploy path, that is:" >&2
    echo "[run]       ./scripts/run_hl_serving_mc.sh --low-level 5pt" >&2
  fi
done

if [ ! -x "$FLORE_PY" ]; then
  echo "[run] FLoRe venv python not found at $FLORE_PY (create .venv, then: uv pip install -e \".[viz]\")" >&2
  exit 2
fi

CKPT=""
# A MIRROR of the planner's force default, not a second source of truth: force on the wire needs
# BOTH sides (the planner's switch AND the low level's FORCE_CMD_WIRE=1) and neither alone does
# anything, so the T4 command echoed below has to carry the companion half — or the operator arms a
# switch that silently cannot fire.
LL_FORCE_ENV="FORCE_CMD_WIRE=1 "
_want=""
for _arg in "$@"; do
  case "$_want" in
    ckpt) CKPT="$_arg"; _want=""; continue ;;
  esac
  case "$_arg" in
    --checkpoint)   _want=ckpt ;;
    --checkpoint=*) CKPT="${_arg#--checkpoint=}" ;;
    --no-force-cmd) LL_FORCE_ENV="" ;;
  esac
done
TASK_EXTRA=()
if [ -z "$CKPT" ] && [ -n "${TASK:-}" ]; then
  # TASK= is a NAME for a checkpoint dir, nothing more: the registry's local dir for that task's
  # 5pt run at horizon H (scripts/fetch_checkpoint.sh, rows `<TASK>_5pt_h<H>`). It is resolved to
  # a plain --checkpoint so the planner never learns a task name — every fivepoint_geom_* checkpoint
  # is served by the same code, and what differs between tasks (where the training episodes
  # started, the force channel, the horizon) is read off the checkpoint itself at startup.
  case "${H:-50}" in
    30|50|100|200) ;;
    *) echo "[run] FATAL: H=${H} — the per-task checkpoints exist at H=50 / 100 (wipe_board: 30 / 50; 200 is a training cell only)." >&2; exit 2 ;;
  esac
  CKPT="$FLORE_ROOT/checkpoints/umi_5pt_${TASK}/h${H:-50}"
  if [ ! -f "$CKPT/config.json" ]; then
    echo "[run] FATAL: TASK=$TASK H=${H:-50} names $CKPT, which is not on disk. Pull it with" >&2
    echo "[run]     scripts/fetch_checkpoint.sh ${TASK}_5pt_h${H:-50}" >&2
    echo "[run]   Known keys: scripts/fetch_checkpoint.sh --list" >&2
    exit 2
  fi
  TASK_EXTRA=(--checkpoint "$CKPT")
  # The ONE thing the page does learn from TASK= is a FOLDER LABEL: the ⏺ run recorder writes each
  # recording under data/rollouts/<TASK>/<YYYY-MM-DD_HH-MM-SS>/ (RECORD_DIR= moves the root).
  # It changes nothing about how the checkpoint is served.
  TASK_EXTRA+=(--record-task "$TASK")
  echo "[run] TASK=$TASK H=${H:-50} -> --checkpoint $CKPT  (recordings: --record-task $TASK)"
elif [ -n "${TASK:-}" ]; then
  # An explicit --checkpoint wins for the MODEL; TASK= still names the recorder's folder.
  TASK_EXTRA=(--record-task "$TASK")
  echo "[run] NOTE: TASK=$TASK does not pick the checkpoint (an explicit --checkpoint was given); it only" >&2
  echo "[run]       labels the ⏺ run recorder's folder (--record-task $TASK)." >&2
fi
if [ -z "$CKPT" ]; then
  echo "[run] FATAL: --checkpoint is required (a five-point HL-v2 checkpoint dir), or TASK=<task> [H=30|50|100]." >&2
  exit 2
fi
if [ ! -f "$CKPT/config.json" ]; then
  echo "[run] FATAL: --checkpoint '$CKPT' has no config.json" >&2
  exit 2
fi

# LAYOUT CHECK, HERE, BEFORE A GPU IS TOUCHED. The planner refuses a non-fivepoint checkpoint too
# (by name, in hl_5pt.validate_layout), but that costs a model load and several GB of VRAM first —
# and the most likely mistake is reaching for this launcher with a 3-tracker or wholebody
# checkpoint, each of which has its own working launcher.
CKPT_MODE="$("$FLORE_PY" -c \
  "import json,sys; print(json.load(open(sys.argv[1])).get('hl_data',{}).get('mode',''))" \
  "$CKPT/config.json")"
FAMILY=fivepoint
case "$CKPT_MODE" in
  fivepoint_geom_*) ;;
  left_only|right_only|both_hands|both_hands_head|both_hands_head_dual)
    # A KEYPOINT SUBSET of the wire's five: the planner HOLDS the slots this checkpoint does not
    # predict at the ready pose (hl_5pt.held_anchor_poses / FivePointDirectWire.held). The same
    # checkpoint is also servable on the 3-point wire (run_hl_serving.sh, left_only only) or through
    # motion completion (run_hl_serving_mc.sh, both_hands_head[_dual]) — this one is for the 5pt
    # low level, MODE=5pt-unified-anneal.
    FAMILY=keypoint_subset ;;
  *)
    echo "[run] FATAL: $CKPT has hl_data.mode='$CKPT_MODE', which this wire cannot carry." >&2
    echo "[run]   This planner publishes FIVE keypoint poses — the ones the policy predicted, the" >&2
    echo "[run]   rest HELD at the ready pose. Servable here: fivepoint_geom_*, left_only," >&2
    echo "[run]   both_hands, both_hands_head[_dual]. The others:" >&2
    echo "[run]     wholebody                     -> ./scripts/run_hl_serving_wholebody.sh" >&2
    echo "[run]     wholebody_geom_*              -> not servable yet (no deploy builds that obs)" >&2
    exit 2 ;;
esac
# How many camera streams the checkpoint reads (hl_data.image_keys, or the single image_key),
# which wire slots are HELD, and WHAT IT WAS TRAINED ON (the split's zarr basenames + episode count,
# from _provenance — the one line that tells merged3 from merged2 when the local dir is named for
# the TASK, not the dataset). Read here only to be ECHOED; the planner derives all of it again.
read -r CKPT_NCAM CKPT_HELD CKPT_DATA <<<"$("$FLORE_PY" - "$CKPT/config.json" <<'PY2'
import json, os, sys
raw = json.load(open(sys.argv[1]))
hl = raw.get("hl_data", {})
keys = hl.get("image_keys") or [hl.get("image_key", "camera0_rgb")]
from flore.s1_policy.data.layout import get_layout
from flore.s1_policy.serving.hl_5pt import held_roles
held = held_roles(get_layout(str(hl.get("mode", "left_only"))))
split = raw.get("_provenance", {}).get("split", {})
zarrs = split.get("dataset_knobs", {}).get("zarr_paths") or hl.get("zarr_paths") or []
data = " + ".join(os.path.basename(str(z)) for z in zarrs) or "(no zarr_paths recorded)"
n_ep = split.get("total_episodes")
# The dataset is the LAST field on purpose: `read` hands the remainder of the line to its last
# variable, so the spaces in it survive the split that the two fields before it rely on.
print(len(keys), ",".join(held) if held else "-", f"{data} ({n_ep} episodes)" if n_ep else data)
PY2
)"

# ACTION-CHUNK PIN, DERIVED FROM THE CHECKPOINT — same contract, same reasoning and the same helper
# shape as the sibling launchers: flore/s1_policy/data/layout.py resolves ACTION_HORIZON from
# FLORE_ACTION_HORIZON at IMPORT time, so a shell launcher is the only place that can still set it.
# It must come from the ACTIVE algorithm's block (the trainer leaves the other two at their
# dataclass defaults), which is why config.json's own `algo` picks the block.
if ! CKPT_H="$("$FLORE_PY" - "$CKPT/config.json" <<'PY'
import json, sys
# Imported, not duplicated: a second copy of the algo->block map here would drift silently.
from flore.s1_policy.training.config import ALGO_CONFIG_FIELD

raw = json.load(open(sys.argv[1]))
algo = raw.get("algo")
field = ALGO_CONFIG_FIELD.get(algo)
if field is None:
    sys.exit(f"config.json algo={algo!r} is not one of {sorted(ALGO_CONFIG_FIELD)}")
block = raw.get(field)
if not isinstance(block, dict):
    sys.exit(f"config.json has no {field!r} block for algo={algo!r}")
horizon = block.get("action_horizon")
if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
    sys.exit(f"config.json {field}.action_horizon={horizon!r} is not a positive integer")
print(horizon)
PY
)"; then
  echo "[run] FATAL: could not read the action horizon from $CKPT/config.json (above)" >&2
  exit 2
fi
if [ -n "${FLORE_ACTION_HORIZON:-}" ] && [ "$FLORE_ACTION_HORIZON" != "$CKPT_H" ]; then
  echo "[run] FATAL: FLORE_ACTION_HORIZON=$FLORE_ACTION_HORIZON in the environment, but" >&2
  echo "[run]        $CKPT declares action_horizon=$CKPT_H. Refusing to guess which is meant." >&2
  exit 2
fi
export FLORE_ACTION_HORIZON="$CKPT_H"

# The ALGO defaults to diffusion in the planner, but the checkpoint states its own — pass it
# through so a flow/act five-point checkpoint does not need the flag spelled twice.
CKPT_ALGO="$("$FLORE_PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('algo',''))" \
  "$CKPT/config.json")"
EXTRA=()
case " $* " in *" --algo "*) ;; *) [ -n "$CKPT_ALGO" ] && EXTRA=(--algo "$CKPT_ALGO") ;; esac
# CARRY is a shorthand, not a second source of truth: an explicit --carry on the line wins.
case " $* " in
  *" --carry "*|*" --carry="*) ;;
  *) [ -n "${CARRY:-}" ] && EXTRA+=(--carry "$CARRY") ;;
esac
# WALK_CLIP likewise. Checked for existence HERE (the planner would too, but after a model load):
# a walk clip that is not on disk should fail before several GB of VRAM are allocated.
case " $* " in
  *" --walk-clip "*|*" --walk-clip="*) ;;
  *)
    if [ -n "${WALK_CLIP:-}" ]; then
      if [ ! -f "$WALK_CLIP" ]; then
        echo "[run] FATAL: WALK_CLIP=$WALK_CLIP is not a file (an extracted feet NPZ from" >&2
        echo "[run]   scripts/extract_walk_feet.py, or a raw bones_seed_g1_holosoma_f32 motion NPZ)." >&2
        exit 2
      fi
      EXTRA+=(--walk-clip "$WALK_CLIP")
    fi ;;
esac
# Same rule for the CADENCE pair. Validated here rather than passed through blind: argparse would
# refuse a typo too, but only after a model load and several GB of VRAM.
case "${CADENCE:-}" in
  ""|sequential|stream) ;;
  *) echo "[run] FATAL: CADENCE='$CADENCE' — want sequential|stream." >&2; exit 2 ;;
esac
case " $* " in
  *" --cadence "*|*" --cadence="*) ;;
  *) [ -n "${CADENCE:-}" ] && EXTRA+=(--cadence "$CADENCE") ;;
esac
case "${ENSEMBLE:-}" in
  ""|avg|act|new|old) ;;
  *) echo "[run] FATAL: ENSEMBLE='$ENSEMBLE' — want avg|act|new|old (HATO's four)." >&2; exit 2 ;;
esac
case " $* " in
  *" --ensemble "*|*" --ensemble="*) ;;
  *) [ -n "${ENSEMBLE:-}" ] && EXTRA+=(--ensemble "$ENSEMBLE") ;;
esac
# The ⏺ RUN RECORDER's two knobs (the page's --record-dir / --record-every-run; explicit flags win).
# RECORD_DIR= moves the root of data/rollouts/<task>/<stamp>/; RECORD_EVERY=1 starts with "Record
# EVERY run" ticked, so a session where every take matters needs no click before each one.
case " $* " in
  *" --record-dir "*|*" --record-dir="*) ;;
  *) [ -n "${RECORD_DIR:-}" ] && EXTRA+=(--record-dir "$RECORD_DIR") ;;
esac
# OOD_STOP: a shorthand for --ood-stop, so `OOD_STOP=0` restores the pre-2026-09-13 log-only
# behaviour (and any other threshold) without editing the command line. An explicit --ood-stop /
# --no-ood-stop on the line wins, exactly as CARRY and CADENCE work.
#
# OOD_EFF is what the BANNER prints, and it is resolved from the same three inputs the page will
# see rather than from OOD_STOP alone: a banner that reads one of them is how the --post-anchor
# line came to claim `command (DEFAULT)` on a run launched with `off`.
OOD_EFF="${OOD_STOP:-0}"
case " $* " in
  *" --no-ood-stop "*) OOD_EFF=0 ;;
  *" --ood-stop "*) OOD_EFF="$(printf '%s\n' "$@" | grep -A1 -x -- '--ood-stop' | tail -1)" ;;
  *" --ood-stop="*) OOD_EFF="$(printf '%s\n' "$@" | sed -n 's/^--ood-stop=//p' | tail -1)" ;;
esac
if [ -n "${OOD_STOP:-}" ]; then
  case " $* " in
    *" --ood-stop "*|*" --ood-stop="*|*" --no-ood-stop "*) ;;
    *) EXTRA+=(--ood-stop "$OOD_STOP") ;;
  esac
fi
if [ -n "${OBS_LEAK:-}" ]; then
  case " $* " in
    *" --obs-leak-m "*|*" --obs-leak-m="*) ;;
    *) EXTRA+=(--obs-leak-m "$OBS_LEAK") ;;
  esac
fi
if [ -n "${OBS_LEAK_DEG:-}" ]; then
  case " $* " in
    *" --obs-leak-deg "*|*" --obs-leak-deg="*) ;;
    *) EXTRA+=(--obs-leak-deg "$OBS_LEAK_DEG") ;;
  esac
fi
case "${RECORD_EVERY:-}" in
  1 | true | TRUE | yes | on)
    case " $* " in *" --record-every-run "*) ;; *) EXTRA+=(--record-every-run) ;; esac ;;
esac
# RECORD_CAMERA_EVERY=N thins the recorder's camera stream (every N-th frame of each view the model
# reads; 0 = no camera frames). Default 1 = every frame, ~2 MB/s per view. An explicit flag wins.
if [ -n "${RECORD_CAMERA_EVERY:-}" ]; then
  case "$RECORD_CAMERA_EVERY" in
    ''|*[!0-9]*) echo "[run] FATAL: RECORD_CAMERA_EVERY='$RECORD_CAMERA_EVERY' — want a non-negative integer." >&2; exit 2 ;;
  esac
  case " $* " in
    *" --record-camera-every "*|*" --record-camera-every="*) ;;
    *) EXTRA+=(--record-camera-every "$RECORD_CAMERA_EVERY") ;;
  esac
fi
# RECORDED-EPISODE REPLAY (2026-09-19): REPLAY=1 replays the checkpoint's OWN training zarr
# (--replay-zarr checkpoint); REPLAY_ZARR=<path> a different converted .zarr.zip. The page then
# gains the "Replay recorded episode" folder (episode picker with the episode's FIRST recorded frame,
# start frame, sequential | open-loop mode, Chunk SOURCE recorded replay | model) and ▶▶ plays the
# RECORDED poses / jaw / wrench (as F^cmd) through the SAME wire, clutch, gate, STOP and ⏺ recorder as
# a policy run. REPLAY_EPISODE=N picks the starting episode, REPLAY_MODE=sequential|openloop the mode.
# Explicit flags win. Without any of these the page is the page as reviewed.
REPLAY_ZARR_EFF="${REPLAY_ZARR:-}"
case "${REPLAY:-}" in
  1 | true | TRUE | yes | on) REPLAY_ZARR_EFF="${REPLAY_ZARR_EFF:-checkpoint}" ;;
esac
if [ -n "$REPLAY_ZARR_EFF" ]; then
  case " $* " in
    *" --replay-zarr "*|*" --replay-zarr="*) ;;
    *) EXTRA+=(--replay-zarr "$REPLAY_ZARR_EFF") ;;
  esac
fi
if [ -n "${REPLAY_EPISODE:-}" ]; then
  case "$REPLAY_EPISODE" in
    ''|*[!0-9]*) echo "[run] FATAL: REPLAY_EPISODE='$REPLAY_EPISODE' — want a non-negative integer." >&2; exit 2 ;;
  esac
  case " $* " in
    *" --replay-episode "*|*" --replay-episode="*) ;;
    *) EXTRA+=(--replay-episode "$REPLAY_EPISODE") ;;
  esac
fi
if [ -n "${REPLAY_MODE:-}" ]; then
  case "$REPLAY_MODE" in
    sequential|openloop) ;;
    *) echo "[run] FATAL: REPLAY_MODE='$REPLAY_MODE' — want sequential or openloop." >&2; exit 2 ;;
  esac
  case " $* " in
    *" --replay-mode "*|*" --replay-mode="*) ;;
    *) EXTRA+=(--replay-mode "$REPLAY_MODE") ;;
  esac
fi

echo "[run] FIVE-POINT PLANNER (5 keypoints -> 5pt, NO completion) on http://localhost:$PORT"
echo "[run]   python: $FLORE_PY (device=$DEVICE)"
echo "[run]   checkpoint: $CKPT  algo=${CKPT_ALGO:-?}  layout=$CKPT_MODE  family=$FAMILY"
echo "[run]              trained on: $CKPT_DATA"
if [ "$FAMILY" = fivepoint ]; then
  echo "[run]              chunk=$FLORE_ACTION_HORIZON steps @ 30 Hz  ->  5 poses + 2 grips + 2 wrenches"
else
  echo "[run]              chunk=$FLORE_ACTION_HORIZON steps @ 30 Hz  ->  the predicted role(s) only;"
  echo "[run]              wire slots HELD at the READY POSE for the whole run: ${CKPT_HELD}"
  echo "[run]              (the clutch publishes their base = the gate's own reference; a Predict"
  echo "[run]               never moves them; their f_cmd is a hard zero; only the predicted hand's"
  echo "[run]               jaw / Bota are read — run_hl_serving.sh's WireMapper rule, on five slots)"
fi
echo "[run]   clutch: --post-anchor command (DEFAULT), carry ${CARRY:-all} — step 0 lands ON the"
echo "[run]     low level's own pre-arm gate reference. The model's ABSOLUTE step 0 sits 59-63 mm"
echo "[run]     from it on the push_cart / rotate_box checkpoints (measured; push_gantry's own"
echo "[run]     episode starts sit 18-20 mm mean / 56 mm max (hands) and 44 / 64 mm (chest) from"
echo "[run]     it, merged3 measured), against a 70 mm tolerance — so --post-anchor"
echo "[run]     off would arm on a good sample and refuse on a slightly worse one. The page"
echo "[run]     measures the actual gap of every staged chunk (the 'gate' field)."
echo "[run]   observation range: the startup log and the page's 'state range' line score the"
echo "[run]     ready-pose observation against THIS checkpoint's own norm_stats band — the check"
echo "[run]     that says whether the first Predict is a prediction or an extrapolation, per task."
if [ "${OBS_LEAK:-0.05}" = 0 ] || [ "${OBS_LEAK:-0.05}" = 0.0 ]; then
  echo "[run]   ⚠️  BELIEF LEASH OFF (OBS_LEAK=0): the commanded anchor — which under"
  echo "[run]     obs_mode=last_target IS the whole observation — may run arbitrarily far from the"
  echo "[run]     robot, while the clutch keeps the PUBLISHED base within 50 mm of measured. That"
  echo "[run]     asymmetry is what folded the body twice (lh_dx 0.605 -> 0.876 m over 16 chunks,"
  echo "[run]     past the arm's own reach): every later chunk conditions on a hand position the arm"
  echo "[run]     never reached. Nothing goes limp — a balance controller asked for an unreachable"
  echo "[run]     hand folds the torso and knees with kp still full."
else
  echo "[run]   belief leash: the commanded anchor is pulled back to within ${OBS_LEAK:-0.05} m /"
  echo "[run]     ${OBS_LEAK_DEG:-10}° of the MEASURED keypoints at every Predict (--obs-leak-m/-deg) —"
  echo "[run]     the SAME leash KeypointClutch.sync already gives the published base, which the"
  echo "[run]     observation never had. Inside it nothing moves, so a healthy run is byte-identical"
  echo "[run]     and action[:,0] == state still holds; it engages only once the command has run"
  echo "[run]     further from the robot than the WIRE may. No RecordTap = no correction (HELD)."
  echo "[run]     The page shows the live lead beside 'state range'."
fi
if [ "$OOD_EFF" = 0 ] || [ "$OOD_EFF" = 0.0 ]; then
  echo "[run]   observation gate: OFF (--ood-stop 0, the default) — an out-of-band observation is"
  echo "[run]     LOGGED and sampled from anyway. Ending a run was the wrong answer to a SYMPTOM;"
  echo "[run]     the leash above is what stops it becoming one. OOD_STOP=0.25 re-arms the gate"
  echo "[run]     (the measured threshold), OOD_STOP=1.0 as a loose backstop behind the leash."
else
  echo "[run]   observation gate: a ▶▶ run ALSO ENDS when a body state column sits more than"
  echo "[run]     $OOD_EFF band widths outside this checkpoint's own [q01, q99] (--ood-stop) — a"
  echo "[run]     BACKSTOP behind the leash. It refuses BEFORE the sample, so nothing new is"
  echo "[run]     published and the wire, the decode anchor and the clutch base all stay on the LAST"
  echo "[run]     EXECUTED step — where the robot already is. ▶▶ continues with NO Reset."
fi
echo "[run]   COMPANION LOW LEVEL — 5pt, ABSOLUTE, RecordTap on (W&B run pq9shxs7):"
echo "[run]     cd $FLORE_ROOT/third_party/holosoma_extensions/force_deploy && \\"
echo "[run]       MODE=5pt-unified-anneal HOLOSOMA_VIVE_RELATIVE=0 RECORD=1 ${LL_FORCE_ENV}\\"
echo "[run]       bash 3_run_policy_vive_force.sh"
if [ -z "$LL_FORCE_ENV" ]; then
  echo "[run]     (--no-force-cmd given: pure position. Drop it, and add FORCE_CMD_WIRE=1 there,"
  echo "[run]      to command force at all.)"
fi
echo "[run]   ⚠️  T4's MODE must be EXACTLY 5pt-unified-anneal and nothing cross-checks it: a"
echo "[run]     mismatch makes the low level drop every frame, so 'nothing moves' means 'these"
echo "[run]     two disagree' before it means anything else."
echo "[run]   ⚠️  Do NOT also start 1_vive_bridge.sh: this planner is the :8770 producer."
if [ "${CKPT_NCAM:-2}" -ge 2 ]; then
  echo "[run]   ⚠️  BOTH cameras are required (L :8773, R :8777) — the checkpoint reads two views."
else
  echo "[run]   ONE camera is read (L :8773 = camera0_rgb). Camera R (:8777) may be running; it is"
  echo "[run]     simply not subscribed to — the planner trims to the checkpoint's own count."
fi
if [ "$FAMILY" = fivepoint ]; then
  echo "[run]   OBSERVATION is the last COMMANDED target's own pelvis-frame geometry (rel_xy3 + rpy"
  echo "[run]     + 2 jaws), which is exactly what obs_mode=last_target means here. Before the first"
  echo "[run]     Predict it is the READY POSE, flagged UNMEASURED so the model's no-data embedding"
  echo "[run]     handles it."
else
  echo "[run]   OBSERVATION is the last COMMANDED target's rotation relative to the run's START"
  echo "[run]     (wrt_start_rot6d per predicted role) + its jaw — the base LivePlanner's own state,"
  echo "[run]     the one run_hl_serving.sh serves this layout with. It reads the IDENTITY before"
  echo "[run]     the first Predict, which is every training episode's first frame."
fi
echo "[run]   The page draws the violet body by per-limb IK of the predicted five points. It is"
echo "[run]     DISPLAY ONLY — the low level solves its own whole-body problem from the wire."
if [ -n "$LL_FORCE_ENV" ]; then
  echo "[run]   FORCE + BOTH JAWS START ENABLED. The page's switches are the live control; this"
  echo "[run]     only says what they read at launch. Force is hard-zeroed below 1.5 N either way"
  echo "[run]     and still needs FORCE_CMD_WIRE=1 there — neither side alone commands force."
else
  echo "[run]   FORCE starts OFF (--no-force-cmd); BOTH JAWS start ENABLED. The page's switches"
  echo "[run]     are the live control, so force can still be turned on there — but only if the"
  echo "[run]     low level also got FORCE_CMD_WIRE=1."
fi
echo "[run]   The low level STOPS ITSELF above FORCE_ESTOP_N=40 N of measured force."
echo "[run]   Arm ONCE, then '▶▶ Run to completion' keeps predicting and executing until ⏹ /"
echo "[run]     STOP / the --auto-max-seconds ceiling (20 s) / the low level leaving :8770 /"
echo "[run]     the observation gate below."
echo "[run]   ▶▶ cadence starts on ${CADENCE:-sequential} — the page's '▶▶ cadence' dropdown is the"
echo "[run]     live switch. SEQUENTIAL is the reviewed chunked path and holds the wire still for"
echo "[run]     ~24% of a run (measured); STREAM samples asynchronously and publishes the ENSEMBLE"
echo "[run]     (${ENSEMBLE:-avg}) of every chunk covering each step, ~1-2% still. RUNBOOK §4."
echo "[run]   ⚠️  Gantry-suspended for first runs: this low level balances on all five points."
echo "[run]   ⏺ RECORD a run: the blue button under Control → green = the next Predict / ▶▶ opens a"
echo "[run]     recording (predicted xyz + wrench, executed steps, every published frame, RecordTap →"
echo "[run]     measured keypoints, the F/T stream, every frame of each camera the model reads → camera/<key>/*.jpg)"
echo "[run]     into ${RECORD_DIR:-$FLORE_ROOT/data/rollouts}/${TASK:-<task>}/<time>/. It runs across chunks,"
echo "[run]     manual Predict/Run presses and ▶▶ runs (⏹ and the ceiling do NOT end it) until STOP /"
echo "[run]     Reset / the low level disarming (joystick start). Plots: scripts/plot_rollout_record.py <folder>"
echo "[run]     (spawned automatically when a recording is saved). RECORD_EVERY=1 = no click per recording;"
echo "[run]     RECORD_CAMERA_EVERY=${RECORD_CAMERA_EVERY:-1} (N = every N-th camera frame, 0 = none)."
if [ -n "$REPLAY_ZARR_EFF" ]; then
  echo "[run]   ⏩ RECORDED-EPISODE REPLAY: --replay-zarr $REPLAY_ZARR_EFF (episode ${REPLAY_EPISODE:-0},"
  echo "[run]     mode ${REPLAY_MODE:-sequential}). The page's 'Replay recorded episode' folder picks the episode"
  echo "[run]     (its FIRST recorded frame is shown) and ▶▶ / p plays the RECORDED poses, jaw and wrench"
  echo "[run]     (F^cmd = -F_meas through the same switch + gate) on the SAME wire / clutch / STOP / ⏺ recorder."
  echo "[run]     Chunk SOURCE switches back to the model live. The demo replays RE-ORIENTED by the hand-vs-"
  echo "[run]     anchor angle (the page's 'transport' line) — physical, not a bug. RUNBOOK §4c."
else
  echo "[run]   ⏩ REPLAY=1 (or REPLAY_ZARR=<zarr>) adds recorded-episode replay to the page — ▶▶ then plays a"
  echo "[run]     RECORDED demonstration (poses, jaw, wrench as F^cmd) instead of the policy. RUNBOOK §4c."
fi

exec "$FLORE_PY" "$FLORE_ROOT/scripts/hl_serving_5pt.py" \
  --device "$DEVICE" \
  --port "$PORT" \
  ${TASK_EXTRA[@]+"${TASK_EXTRA[@]}"} \
  "${EXTRA[@]}" \
  "$@"
