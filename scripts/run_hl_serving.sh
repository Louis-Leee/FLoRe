#!/usr/bin/env bash
# Launch the ONLINE HL-v2 (flow | diffusion) PLANNER — a SINGLE-process viser control
# panel that assembles a live obs from the running rig, runs the HL-v2 policy on
# Predict/Run, and STREAMS the LEFT-hand chunk (pose + gripper + optional gated F^cmd)
# to the low-level controller over the VIVE-bridge WS :8770, visualizing the rollout +
# predicted force + gripper + low-level online/offline status in viser.
#
# ONE PROCESS for flow/diffusion/act: those models load in the SAME FUJI .venv that holds
# viser + mink + mujoco, so there is no backend subprocess and no base64-npz pipe.
# scripts/hl_serving.py is the whole thing.
#
# --algo muse is the exception, and only in WHERE THE MODEL RUNS. The MuSe-on-UVA policy needs
# the uvaforce env (torch 2.7.1 + hydra + the MAR/VAE trees), which is mutually exclusive with
# this one, so it comes up as a child process and is spoken to over a pipe
# (scripts/muse_hl_infer_server.py + fuji/s1_policy/serving/hl_muse.py). EVERYTHING ELSE is the
# same code path: the same wire mapper, force gate, drip cadence, recorder, reach compensation,
# ghost robot and joystick stop. Unlike run_muse_force_serving.sh, which put the whole planner
# in the backend and therefore carries its own (older) copy of the deploy contract.
#
# PREREQUISITES (bring these up FIRST, in their own terminals — same bridges as the muse
# planner; the low-level MUST run in ABSOLUTE mode so the planner's absolute targets apply):
#   * T2 F/T bridge WITH --pipeline (6-DoF gravity-only force):  bash 2_ft_bridge.sh --pipeline
#   * Camera bridge on :8773 (D405) and gripper bridge on :8784 (LEFT).
#   * The LOW-LEVEL controller in ABSOLUTE mode (companion command printed below), launched
#     with RECORD=1 so it broadcasts its live joint state + pelvis-frame F_meas on the
#     RecordTap :8779 — the planner then renders the REAL robot mesh and shows the exact
#     F_meas the servo consumes. (Without RECORD=1 the planner still works: the mesh falls
#     back to the predicted-IK render and the F/T panel shows the raw :8771 sensor wire.)
#     For PURE POSITION (default), leave the force wire OFF. To let the planner command
#     force, launch the low-level with FORCE_CMD_WIRE=1 AND toggle the force switch in the
#     viser page; below the gate it is still pure position.
#
# READY POSE = "ARMS ZEROED" (2026-08-07). 3_run_policy_vive_force.sh runs with
# HOLOSOMA_INIT_READY_POSE=1, so after 'A' the low level holds `default_dof_angles with all 14 ARM
# joints = 0` (forearms horizontal) and gates arming against THAT pose. hl_serving.py now seeds its
# render, decode anchor and idle wire there too (--zero-arms, default ON) — previously it seeded at
# the raw crouch `home_anchor.default_qpos`, which is 310 mm/hand from the gate reference against a
# 70 mm tolerance, and only the RECORD=1 idle re-sync was saving the arm gate. Pass --no-zero-arms
# only if you started the low level with HOLOSOMA_INIT_READY_POSE=0.
#
# ISOLATION: this planner only READS the low level (S0 pose wire :8770 + read-only sensor
# bridges + the read-only RecordTap :8779). It does NOT change 3_run_policy_vive_force.sh's
# standalone/VIVE behavior; RECORD=1 is an existing knob of that script (the opt-in teleop
# recorder tap), so running the standalone low-level test unchanged is byte-identical.
#
# Usage (matching scripts/infer_s1_{flow,diffusion,act}.py):
#   PORT=8088 ./scripts/run_hl_serving.sh --algo flow      --checkpoint checkpoints/ckpt_flow_final
#   PORT=8088 ./scripts/run_hl_serving.sh --algo diffusion --checkpoint checkpoints/ckpt_diffusion_final
#   PORT=8088 ./scripts/run_hl_serving.sh --algo act       --checkpoint checkpoints/ckpt_act_20k
#   PORT=8088 ./scripts/run_hl_serving.sh --algo muse      --checkpoint <uva_hl_force.slim.pt>
# (muse also needs BACKEND_PY, or the default ~/miniconda3/envs/uvaforce/bin/python; the
#  matching OFFLINE tool is ./scripts/run_infer_s1_muse.sh)
#
# REPLAY A RECORDED EPISODE instead of the policy (--replay-zarr): drive the rig with one of
# the collected demonstrations through the IDENTICAL predict->preview->execute pipeline, so
# "what the human demo did" and "what the policy predicts" are comparable on the same rig.
#   PORT=8088 ./scripts/run_hl_serving.sh --algo flow --checkpoint checkpoints/ckpt_flow_final \
#       --replay-zarr checkpoint
# 'checkpoint' resolves the zarr the checkpoint was TRAINED on (clean94), so the replayed
# demos are exactly the distribution the policy learned; pass an explicit .zarr.zip to
# replay a different session. The viser page then gains a "Replay recorded episode" panel:
#   * Chunk SOURCE      recorded replay | model  (switch live, no relaunch — the model stays loaded)
#   * Episode           clickable dropdown over all 94 episodes (frame counts shown). Episodes
#                       whose first frames were MASKED by the converter are labelled
#                       "⚠ starts at N" and auto-seek there (on clean94, 8 of the 94 start
#                       masked); an episode with no playable frame at all is labelled too.
#   * Start frame       scrub to any frame; a masked frame snaps forward to the next playable one
#   * Play episode      SEQUENTIAL mode: repeats predict->execute at the Execute-how-many-steps
#                       cadence straight through to the episode end (arm ONCE), Stop interrupts
#   * green path        the whole recorded episode drawn at the LEFT TCP, same green as
#                       replay_zarr_viz.py, transported by the SAME anchor as the executed chunk
# The existing "Execute how many steps" dropdown (16/30/50) governs replay too, and the force
# gate / gripper switch / clutch re-anchor behave exactly as in model mode.
#
# WHY THIS IS FRAME-SAFE: the recorded chunk is fed in the model's OWN anchor-RELATIVE form,
# so the recording's lighthouse world frame CANCELS and the demo replays from wherever the arm
# actually is. Nothing aligns lighthouse-world to pelvis. See fuji/s1_policy/serving/hl_replay.py.
#
# The older ``ckpt_*_stamped`` symlink pair was trained on the 2026.07.30 clean99
# session, whose puck->TCP extrinsic was the pre-fix (0.21, 0, 0) rather than the
# corrected (0.225, 0, -0.05). That extrinsic moves the physical point the pose
# MEANS, so the same hand motion yields a different EE-local delta -> the stamped
# model is out of distribution on today's rig. (The session's world_align differs
# too, but that one is harmless: the action is anchor-relative, so a global world
# rotation cancels exactly -- verified to machine precision. The extrinsic is the
# real driver.) Always pass the ``_final`` pair for the live rig.
#
# Env knobs:
#   FUJI_PY   python (default: <repo>/.venv/bin/python)
#   PORT      viser port (default: 8088)
#   DEVICE    torch device (default: cuda)
#   TASK, H   a shorthand for --checkpoint (the 5pt launcher's, 2026-09-20): checkpoints/umi_5pt_${TASK}/h${H}
#             (H defaults to 50; scripts/fetch_checkpoint.sh <TASK>_5pt_h<H> pulls it). --algo is read
#             off the checkpoint's config.json when omitted. TASK= ALSO labels the ⏺ run recorder's
#             folder (--rollout-task): data/rollouts/<TASK>/<YYYY-MM-DD_HH-MM-SS>/. With an explicit
#             --checkpoint, TASK= is the label only.
#             Only LEFT-canonical checkpoints fit this wire (left_only — wipe_board*, the 0823 runs);
#             a five-point or 3-tracker checkpoint belongs on run_hl_serving_5pt.sh.
#   RECORD_EVERY=1          the ⏺ run recorder starts with "Record EVERY run" ticked (--rollout-every-run)
#   RECORD_CAMERA_EVERY=N   record every N-th camera frame into each run folder (--rollout-camera-every; 0 = none)
#   ROLLOUT_DIR=DIR         the run recorder's root (default data/rollouts; --rollout-dir). NOT the deploy
#                           take's --record-dir — see "THE ⏺ RUN RECORDER" below.
#   FUJI_ACTION_HORIZON  DO NOT SET IT — this script DERIVES the action-chunk length from
#             --checkpoint's own config.json and exports it (an H=30 checkpoint pins 30, an
#             H=50 one pins 50), because the constant is resolved at IMPORT time and only a
#             launcher can still reach it. Set it explicitly and a disagreement with the
#             checkpoint is a hard error rather than a silent override.
# Extra args pass through to hl_serving.py, e.g.:
#   --force-gate-n 6         initial force gate (N); adjustable live in the page
#   --enable-force-cmd       start with the force switch ON (default OFF = pure position)
#   --command-gripper        autonomously drive the LEFT jaw (default OFF = read-only)
#   --exec-steps 16          how many of the 50 predicted steps to execute (default 16; 16/30/50 live)
#   --record-ws ws://…:8779  low-level RecordTap (live mesh + F_meas); default already :8779
#   --num-steps N            override sampler steps (flow Euler / diffusion DDIM; ACT is 1-pass)
#
# ACT + TEMPORAL ENSEMBLING (upstream's --temporal_agg). An ACT checkpoint carries
# temporal_ensemble=True in its own config, so the planner turns it ON automatically and says so
# at startup. Run then RE-QUERIES the policy EVERY step from the CURRENT observation and executes
# the exp(-0.01*i)-weighted blend of every still-live prediction for that instant, instead of
# dripping N steps of one chunk. Measured 6.14 ms per re-query+blend against the 33.3 ms budget at
# 30 Hz (5.4x headroom). There is a live checkbox in the page's Control folder
# ("Temporal ensembling") so the same checkpoint can be A/B'd against the plain chunk cadence
# without a relaunch; it applies on the NEXT Run. flow/diffusion have no such entry point, so for
# them the checkbox is present but disabled and the cadence is unchanged.
#   --allow-3dof-force       opt into a 3-DoF F/T bring-up (else HARD-FAIL without --pipeline)
#   --replay-zarr PATH|checkpoint   enable recorded-episode replay (see the REPLAY block above)
#   --replay-episode N       initial replay episode (default 0; selectable live in the page)
#   --record                 RECORD every execution as a re-runnable take (see below)
#   --record-dir DIR         where takes go (default data/deploy_takes/<timestamp>)
#   --reach-ki 0.05          OPT-IN reach compensation (default 0 = OFF; see below)
#
# WHEN A FINE ADJUSTMENT WON'T LAND (--reach-ki). Measured on the recorded takes: the low level
# is a VELOCITY follower — it reproduces how the wire MOVES (displacement gain 0.80/0.94/0.80)
# but a STANDING position error produces no corrective motion (corr(lead, next tick) =
# -0.18/-0.04/-0.01), and steady-state gain is ~1 so this is NOT a droop. The hand therefore
# parks 1.5-2.9 cm short, and because the planner re-anchors the decode at the MEASURED pose on
# every Predict, that shortfall is FORGIVEN each chunk — nothing keeps score, so the last
# centimetres before a grasp never arrive. --reach-ki adds the un-executed part back onto the
# PUBLISHED WIRE only (never the obs / decode anchor / chunk), clamped (--reach-clamp, default
# 3 cm, ceiling 6 cm) and leaky (--reach-decay). THE GATE IS "did the arm move": it accumulates
# only while the arm performed < --reach-stall-frac (0.5) of the commanded motion over
# --reach-window (9) steps -- i.e. commanded 6 cm / moved 3 cm => leave it alone; commanded 5 cm /
# didn't move => accumulate. Measured on 12 real takes that gate is 100%/0% (catches every stalled
# step, never fires while tracking); an earlier DISTANCE-based gate managed only 18%/24%, because
# the standing lead is ~2.5 cm whether the arm is tracking or frozen. CONTACT CUT-OUT (--reach-contact-n, default 1.5 N): a hand pressing on
# something immovable shows the SAME small shortfall as one that fell short, so above that force
# the residual DECAYS instead of growing — otherwise it would command a harder press toward the
# 40 N stop (5-45% of accumulating steps per recorded take had >1.5 N of contact).
# Reset on Stop / Reset / new rollout / low-level-lost / un-tick. Live
# toggle + residual readout in the page's "Reach compensation" folder; ⚠ SATURATED there means
# the arm still is not arriving and something else is wrong. TUNE IT OFFLINE FIRST against your
# own takes (no robot, no bridges):
#   .venv/bin/python scripts/tune_reach_comp.py data/deploy_takes/<session> --ki 0.02 0.05 0.1
# ⚠ This commands reach the policy did not ask for: bring it up gantry + hardware E-stop, force
# wire OFF, starting at 0.05. Default OFF is byte-identical to before.
#
# IF A PREDICT FEELS SLOW / THE PAGE STUTTERS / THE GPU LOOKS UNUSED (fixed 2026-08-15). The
# ghost-robot arm IK and a diffusion sample are BOTH GIL-bound (the sample is ~6100 tiny CUDA
# launches issued from Python: cpu/wall = 1.00, it never blocks on the GPU), so the IK's own thread
# did not overlap the sampler, it interleaved with it: one Predict measured 46 ms with the ghost
# idle and 3543 ms with it solving (77x), at ~1.3% GPU utilization — the device was starved because
# the thread that feeds it could not get the interpreter. Fixed by a cooperative yield gate
# (fuji/s1_policy/serving/gil_yield.py: the IK stands back per step while an inference is in
# flight), an analytic Jacobian for that IK (646 -> 150 ms/chunk), and not solving at all while the
# ghost is hidden. All three are VIZ-ONLY — the wire, the drip cadence and the freshness gates are
# untouched. If a preview ever gets expensive again, UNTICK "Show ghost robot" in the page: that is
# the zero-code way to confirm the viz is what is slowing the policy.
#
# RECORDING A DEPLOY RUN (--record). Turns a real-robot run into a dataset: each EXECUTION
# (Run / Play / sequential) writes one take_NNN holding the executed plan's OWN observation
# (conditioning image, state, force window) + its raw chunk + decode anchor + seed, the measured
# 29-DOF joint readback and FK'd LEFT TCP at ~30 Hz, the wire published per step, the raw Bota
# stream, and the camera JPEGs. A Predict that is never Run is NOT recorded (retries are counted);
# a ⏺ toggle in the page arms/disarms it live, so nothing is written until you want it. Needs
# RECORD=1 on the low level for the measured streams. Replay + re-infer a take offline (no robot,
# no bridges) with:
#   .venv/bin/python scripts/reinfer_hl_take.py data/deploy_takes/<session> \
#       --checkpoint <ckpt> --algo diffusion --port 8089
# The stored observation re-encodes to the byte-identical model input, so re-inferring a recorded
# prediction with the same checkpoint + seed reproduces the deploy chunk exactly — that is the
# regression check; a different checkpoint answers "what would the new model have done here".
# See docs/RUNBOOK_hl_serving.md §"Recording a deploy run".
#
# THE ⏺ RUN RECORDER (2026-09-20, ported from run_hl_serving_5pt.sh). A SECOND recorder beside the
# deploy take, for a different question: the take is a re-runnable dataset entry (observation + raw
# chunk), this is the run's TIME SERIES for the figures — the policy's predicted wrench at the wall
# time each step executed, every F/T frame, every RecordTap frame with the robot's three wire points
# FK'd, every published frame and every camera frame — one folder per RECORDING under
# data/rollouts/<TASK>/<YYYY-MM-DD_HH-MM-SS>/, the SAME layout the 5pt page writes. The blue button
# right under Control arms it (→ green); the first Predict / ▶▶ opens the folder and it stays open
# across chunks, manual Predict/Run presses and ▶▶ runs (⏹ / the ceiling / the low level lost do NOT
# end it) until ↺ Reset, the low level disarming (joystick start) or the button. Then, as after a 5pt run:
#   uv run scripts/plot_rollout_record.py data/rollouts/<TASK>/<stamp>          # xyz + force plots
#   .venv/bin/python overleaf/tools/gen_fig_rollout_force_pred.py --latest data/rollouts/<TASK>
#   RECORD_EVERY=1 TASK=wipe_board_new ./scripts/run_hl_serving.sh --enable-force-cmd --command-gripper --reach-ki 0.1
# Needs RECORD=1 on the low level for the measured streams (the same knob the arm gate needs).
set -eo pipefail

FUJI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FUJI_PY="${FUJI_PY:-$FUJI_ROOT/.venv/bin/python}"
UMI="$FUJI_ROOT/third_party/UMI_collection"
IK_SRC="$UMI/third_party/humi_ik"
VIZ_SRC="$UMI/viz"
REPLAY_SRC="$UMI/replay"
PORT="${PORT:-8088}"
DEVICE="${DEVICE:-cuda}"
# ANCHOR=command turns on command-side anchoring (PLAI) for WHICHEVER algorithm is being served —
# act, diffusion, flow and muse all decode through the one re-anchor, so this is algo-agnostic.
# Default "measured" is the previous behaviour, byte for byte. ANCHOR_EPS overrides the leak bound
# (m); everything else stays on hl_serving.py's defaults and can still be set with the raw flags.
ANCHOR="${ANCHOR:-measured}"
case "$ANCHOR" in
  measured|command) ;;
  *) echo "[run] FATAL: ANCHOR must be 'measured' or 'command', got '$ANCHOR'." >&2; exit 2 ;;
esac
ANCHOR_ARGS=(--anchor-mode "$ANCHOR")
if [ -n "${ANCHOR_EPS:-}" ]; then ANCHOR_ARGS+=(--anchor-eps "$ANCHOR_EPS"); fi

if [ ! -x "$FUJI_PY" ]; then
  echo "[run] FUJI venv python not found at $FUJI_PY (create .venv, then: uv pip install -e \".[viz]\")" >&2
  exit 2
fi

# ACTION-CHUNK PIN, DERIVED FROM THE CHECKPOINT. -----------------------------------------
# fuji/s1_policy/data/layout.py resolves ACTION_HORIZON from FUJI_ACTION_HORIZON at IMPORT
# time, and hl_serving.py sizes its execute-N-of-S choices, its clamps and its status text
# from that module constant — so a shell launcher is the only place that can still set it.
#
# Getting it wrong is loud, not silent: inference/common.validate_model_layout pins
# action_horizon UNCONDITIONALLY (unlike force_hist_len, which the live planner takes from
# the checkpoint), so an H=30 checkpoint on an unpinned build refuses to load. But "loud at
# the robot" is still a wasted bring-up, and the checkpoint already STATES its horizon — so
# derive the pin from the very file about to be loaded and the mismatch becomes impossible
# rather than merely detected. Same reasoning as run_train_s1.sh deriving FH/H from SESSION.
#
# It must come from the ACTIVE algorithm's block, not from any action_horizon in the file:
# the trainer pushes the layout onto the algo block it is training and leaves the other two
# (plus `data`) at their dataclass defaults, so an H=30 diffusion checkpoint really does
# read {model: 50, diffusion: 30, act: 50, data: 50}. read_checkpoint_config hard-checks
# config.algo against --algo, so the file's own `algo` picks the right block.
#
# --algo muse is the one exception: a UVA checkpoint is a .pt FILE with its config pickled
# inside (only the uvaforce interpreter can unpickle it), so the pin is derived by a probe
# subprocess instead — scripts/lib/muse_pin.sh, shared with run_infer_s1_muse.sh.
CKPT=""
ALGO=""
_want=""
for _arg in "$@"; do
  if [ -n "$_want" ]; then
    case "$_want" in ckpt) CKPT="$_arg" ;; algo) ALGO="$_arg" ;; esac
    _want=""
    continue
  fi
  case "$_arg" in
    --checkpoint)   _want=ckpt ;;
    --checkpoint=*) CKPT="${_arg#--checkpoint=}" ;;
    --algo)         _want=algo ;;
    --algo=*)       ALGO="${_arg#--algo=}" ;;
  esac
done

# TASK= / H= — the five-point launcher's shorthand, so the same checkpoint is named the same way on
# both pages: `TASK=wipe_board_new` here and on run_hl_serving_5pt.sh serve checkpoints/umi_5pt_wipe_board_new/h50.
TASK_EXTRA=()
if [ -z "$CKPT" ] && [ -n "${TASK:-}" ]; then
  CKPT="$FUJI_ROOT/checkpoints/umi_5pt_${TASK}/h${H:-50}"
  if [ ! -f "$CKPT/config.json" ]; then
    echo "[run] FATAL: TASK=$TASK H=${H:-50} names $CKPT, which is not on disk. Pull it with" >&2
    echo "[run]     scripts/fetch_checkpoint.sh ${TASK}_5pt_h${H:-50}" >&2
    exit 2
  fi
  TASK_EXTRA=(--checkpoint "$CKPT")
  # The ONE other thing the page learns from TASK= is a FOLDER LABEL: the ⏺ run recorder writes each
  # recording under data/rollouts/<TASK>/<YYYY-MM-DD_HH-MM-SS>/ (ROLLOUT_DIR= moves the root).
  TASK_EXTRA+=(--rollout-task "$TASK")
  echo "[run] TASK=$TASK H=${H:-50} -> --checkpoint $CKPT  (recordings: --rollout-task $TASK)"
elif [ -n "${TASK:-}" ]; then
  # An explicit --checkpoint wins for the MODEL; TASK= still names the recorder's folder.
  TASK_EXTRA=(--rollout-task "$TASK")
  echo "[run] NOTE: TASK=$TASK does not pick the checkpoint (an explicit --checkpoint was given); it only" >&2
  echo "[run]       labels the ⏺ run recorder's folder (--rollout-task $TASK)." >&2
fi

if [ "$ALGO" = "muse" ]; then
  BACKEND_PY="${BACKEND_PY:-$HOME/miniconda3/envs/uvaforce/bin/python}"
  if [ ! -x "$BACKEND_PY" ]; then
    echo "[run] FATAL: --algo muse needs the uvaforce python (the env that can import" >&2
    echo "[run]        unified_video_action); not executable at $BACKEND_PY." >&2
    echo "[run]        Set BACKEND_PY=/path/to/uvaforce/bin/python — see docs/RUNBOOK_muse_8gpu.md." >&2
    exit 2
  fi
  # shellcheck source=scripts/lib/muse_pin.sh
  source "$FUJI_ROOT/scripts/lib/muse_pin.sh"
  muse_pin_action_horizon "$BACKEND_PY" "$FUJI_ROOT" "$CKPT" || exit 2
  set -- "$@" --backend-python "$BACKEND_PY"
elif [ -n "$CKPT" ]; then
  # Omitting --checkpoint is legal (model-free replay), and then there is no checkpoint to
  # derive from: the build keeps layout.py's default and replay pins the rate itself.
  if [ ! -f "$CKPT/config.json" ]; then
    echo "[run] FATAL: --checkpoint '$CKPT' has no config.json" >&2
    exit 2
  fi
  if ! _CKPT_PIN="$("$FUJI_PY" - "$CKPT/config.json" <<'PY'
import json, sys
# Imported, not duplicated: a second copy of the algo->block map here would drift silently.
# This is a THROWAWAY process — its own ACTION_HORIZON resolution cannot affect the run,
# which is launched afterwards with the env already set.
from fuji.s1_policy.training.config import ALGO_CONFIG_FIELD

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
print(horizon, algo)
PY
  )"; then
    echo "[run] FATAL: could not read the action horizon from $CKPT/config.json (above)" >&2
    exit 2
  fi
  read -r CKPT_H CKPT_ALGO <<<"$_CKPT_PIN"
  if [ -z "$ALGO" ]; then
    # --algo omitted with a checkpoint (the TASK= shorthand, typically): the checkpoint's own
    # config.json says which family it is, and hl_serving.py hard-checks the two agree anyway.
    ALGO="$CKPT_ALGO"
    TASK_EXTRA+=(--algo "$ALGO")
    echo "[run]   --algo $ALGO (read off $CKPT/config.json)"
  fi
  if [ -n "${FUJI_ACTION_HORIZON:-}" ] && [ "$FUJI_ACTION_HORIZON" != "$CKPT_H" ]; then
    echo "[run] FATAL: FUJI_ACTION_HORIZON=$FUJI_ACTION_HORIZON in the environment, but" >&2
    echo "[run]        $CKPT declares action_horizon=$CKPT_H. Refusing to guess which is meant." >&2
    echo "[run]        Unset FUJI_ACTION_HORIZON and let the checkpoint decide." >&2
    exit 2
  fi
  export FUJI_ACTION_HORIZON="$CKPT_H"
fi

# The ⏺ RUN RECORDER's knobs (the page's --rollout-every-run / --rollout-camera-every / --rollout-dir;
# an explicit flag wins). Same env names as run_hl_serving_5pt.sh, except the root: RECORD_DIR there
# is ROLLOUT_DIR here, because --record-dir on THIS page is the deploy take's session directory.
EXTRA=()
case "${RECORD_EVERY:-}" in
  1 | true | TRUE | yes | on)
    case " $* " in *" --rollout-every-run "*) ;; *) EXTRA+=(--rollout-every-run) ;; esac ;;
esac
if [ -n "${RECORD_CAMERA_EVERY:-}" ]; then
  case "$RECORD_CAMERA_EVERY" in
    ''|*[!0-9]*) echo "[run] FATAL: RECORD_CAMERA_EVERY='$RECORD_CAMERA_EVERY' — want a non-negative integer." >&2; exit 2 ;;
  esac
  case " $* " in
    *" --rollout-camera-every "*|*" --rollout-camera-every="*) ;;
    *) EXTRA+=(--rollout-camera-every "$RECORD_CAMERA_EVERY") ;;
  esac
fi
case " $* " in
  *" --rollout-dir "*|*" --rollout-dir="*) ;;
  *) [ -n "${ROLLOUT_DIR:-}" ] && EXTRA+=(--rollout-dir "$ROLLOUT_DIR") ;;
esac
if [ -n "${RECORD_DIR:-}" ]; then
  echo "[run] NOTE: RECORD_DIR is the 5pt launcher's knob; here the ⏺ run recorder's root is ROLLOUT_DIR=" >&2
  echo "[run]       (--record-dir on this page is the deploy TAKE directory). RECORD_DIR ignored." >&2
fi

echo "[run] HL-v2 PLANNER on http://localhost:$PORT  (Ctrl-C to stop)"
echo "[run]   python: $FUJI_PY (device=$DEVICE)"
if [ "$ALGO" = "muse" ]; then
  echo "[run]   muse BACKEND: $BACKEND_PY (uvaforce env; the model runs there, not here)"
  echo "[run]   action chunk: FUJI_ACTION_HORIZON=$FUJI_ACTION_HORIZON steps (from the checkpoint's num_actions)"
  echo "[run]   NOTE: the first Predict waits on the backend's cold start (~1 min: hydra + MAR + the VAE)."
elif [ -n "$CKPT" ]; then
  echo "[run]   action chunk: FUJI_ACTION_HORIZON=$FUJI_ACTION_HORIZON steps (from $CKPT/config.json)"
fi
echo "[run]   REMINDER: run T2 as 'bash 2_ft_bridge.sh --pipeline' (6-DoF force for the planner),"
echo "[run]            and launch the LOW-LEVEL in ABSOLUTE mode (pure position — force wire OFF):"
echo "[run]     cd $FUJI_ROOT/third_party/holosoma_extensions/force_deploy && \\"
echo "[run]       MODE=3pt-unified-anneal HOLOSOMA_VIVE_RELATIVE=0 RECORD=1 \\"
echo "[run]       ONNX=logs/WBT-Holosoma/<force-qualified-run>/model_XXXXX.onnx bash 3_run_policy_vive_force.sh"
echo "[run]   PHASE 0 (above) = PURE POSITION: no FORCE_CMD_WIRE, so the low level IGNORES the wire's"
echo "[run]     f_cmd entirely and only translation+rotation reach the robot. This is the default."
echo "[run]   PHASE 1 (force) needs BOTH ends to opt in — add FORCE_CMD_WIRE=1 to that low-level launch"
echo "[run]     AND toggle the force switch in the viser page. Below the 1.5 N gate it is still pure"
echo "[run]     position: the command is hard-zeroed and (GATE_CMD=1) F_meas is kept out of the obs +"
echo "[run]     servo too, so no force loop runs however hard the Bota is pressed."
echo "[run]   The low level STOPS ITSELF (joystick-B path) above FORCE_ESTOP_N=40 N of measured force."
_ROLL_TASK="${TASK:-<read off the checkpoint path>}"
echo "[run]   ⏺ run recorder: data/rollouts/${_ROLL_TASK}/<YYYY-MM-DD_HH-MM-SS>/ per recording — from the"
echo "[run]     first Predict / ▶▶ until ↺ Reset / the low level disarming (joystick start) / the button;"
echo "[run]     manual Predict → Run chunks, ▶▶ runs and ⏹ all stay inside ONE recording."
case "${RECORD_EVERY:-}" in
  1 | true | TRUE | yes | on) echo "[run]     RECORD_EVERY=1: every recording opens by itself (no click needed)" ;;
  *) echo "[run]     arm it in the page (blue → green) before Predict / ▶▶, or launch with RECORD_EVERY=1" ;;
esac
echo "[run]     then: .venv/bin/python overleaf/tools/gen_fig_rollout_force_pred.py --latest data/rollouts/${_ROLL_TASK}"
if [ "$ANCHOR" = "command" ]; then
  echo "[run]   ⚠️  ANCHOR=command: each chunk decodes from the LAST EXECUTED COMMANDED step, not"
  echo "[run]      the measured pose. Measured on the recorded takes, the measured-pose re-anchor"
  echo "[run]      WIPES 84% of the accumulated shortfall and puts a 16.0 mm BACKWARD step in the"
  echo "[run]      published target at every chunk boundary; this removes both. The carried anchor"
  echo "[run]      is leaked onto a ${ANCHOR_EPS:-0.03} m ball around the arm and decays back while"
  echo "[run]      in contact (>=1.5 N) — UNCLAMPED it diverges ~22 cm, so do not raise the leak."
  echo "[run]      Applies to act / diffusion / flow / muse alike. ANCHOR=measured to go back."
fi
PYTHONPATH="$IK_SRC:$VIZ_SRC:$REPLAY_SRC:${PYTHONPATH:-}" "$FUJI_PY" \
  "$FUJI_ROOT/scripts/hl_serving.py" \
  --device "$DEVICE" \
  --port "$PORT" \
  "${ANCHOR_ARGS[@]}" \
  "${TASK_EXTRA[@]}" \
  "${EXTRA[@]}" \
  "$@"
