#!/usr/bin/env bash
# OFFLINE inference viewer for diffusion_umi_5pt: five predicted keypoints, and the G1 body they
# imply — drawn by per-limb IK, with NO motion completion.
#
#   run_infer_s1_mc.sh          3 predicted poses -> Motion_Completion (a SECOND model, its own
#                                 conda env, its own checkpoint) -> a 29-DoF humanoid
#   run_infer_s1_wholebody.sh   29 predicted joints -> two G1 bodies (no trajectory to draw)
#   THIS SCRIPT                 5 predicted keypoints -> two G1 bodies AND five trajectory pairs
#
# It is the SAME page every keypoint checkpoint gets (scripts/infer_s1_diffusion.py +
# fuji/s1_policy/inference/viewer.py) — the blue/orange predicted-vs-target polylines, the gripper
# clusters, the TCP/tracker/wrist triads, the F^cmd arrows, the charts, RGB playback — plus one
# opt-in overlay: --keypoint-ik draws the whole body, violet for the PREDICTION and pale blue for
# the dataset TARGET. Five points leave nothing to infer, so the body is a kinematic consequence
# of the chunk and an IK is the whole of it: no conda env, no subprocess, no second checkpoint.
#
# ⚠️ --keypoint-ik IMPLIES --display-frame g1-ready-pose, and that is a PREREQUISITE, not a
# preference. The five roles' anchors are not in one frame in the dataset: the three TRACKED ones
# carry the operator's recorded world pose while the two FEET are KeypointFK(wb_action_joints) in
# the completion's PELVIS frame, and nothing stored relates them. Hung on their own display homes
# the feet land metres from the body — measured, 543 mm of IK residual. Re-based onto the pose the
# robot holds after **A**, all five share one coherent frame and the IK closes on the TARGET to
# 0.2-0.4 mm. Pass --display-frame display-home to see the raw trajectories instead; the body then
# refuses, with that reason on the page.
#
# ⚠️ The scene's yaw-only turn to the operator's forward is applied to the THREE TRACKED roles only.
# A foot's displacement is already body-frame, so turning it by the operator's heading (-169 deg on
# this rig) would move a ~73 mm foot excursion up to 145 mm the wrong way. One matrix per SOURCE
# FRAME, which is what display.world_rotmat_by_role exists to say.
#
# PREREQUISITES
#   * the viz extra (yourdfpy + viser):  uv pip install --python .venv/bin/python -e ".[viz]"
#   * both submodules, for the URDF and holosoma's home_anchor.json:
#       git submodule update --init third_party/UMI_collection            # NOT --recursive
#       git submodule update --init --recursive third_party/holosoma_extensions
#   * the checkpoint's own training zarr on disk (the page replays REAL dataset anchors).
#     scripts/fetch_dataset.sh 0902_merged3_h200_5pt
#
# A KEYPOINT SUBSET checkpoint works too — left_only (the plain diffusion_umi preset, e.g. the
# wipe_board runs), both_hands, both_hands_head[_dual]: the predicted role(s) draw as before and
# the roles the checkpoint does not predict are HELD at the ready pose for BOTH bodies, which is
# what scripts/hl_serving_5pt.py publishes for the same checkpoint. The panel names the held roles.
# Without a chest role the scene's yaw-only turn to the operator's forward is read off the predicted
# HAND's grip yaw at the anchor instead (ready_pose.grip_yaw_alignment; the page names the source) —
# measured to track the chest heading to 2-8 deg at episode start on this rig's 3-tracker datasets.
# Un-turned, a left_only chunk was drawn in the recorded world's axes (wipe_board's operator faces
# -118.8 deg) next to a hand re-based to face +x, i.e. MIRRORED.
#
# Usage:
#   ./scripts/run_infer_s1_5pt.sh checkpoints/umi_5pt_rotate_box/h50
#   ./scripts/run_infer_s1_5pt.sh checkpoints/umi_5pt_wipe_board/h50              # left_only
#   ./scripts/run_infer_s1_5pt.sh <ckpt> --sample-index 12000 --seed 3
#   ./scripts/run_infer_s1_5pt.sh <ckpt> --self-test          # headless JSON, no viser, no IK
#   NO_IK=1 ./scripts/run_infer_s1_5pt.sh <ckpt>              # trajectories only
#
# Env knobs:
#   FUJI_PY  python (default: <repo>/.venv/bin/python)
#   PORT     viser port (default: 8087, the offline viewer's own)
#   DEVICE   torch device (default: cuda)
#   NO_IK    set to 1 to skip --keypoint-ik (and therefore the implied ready-pose frame)
set -eo pipefail

FUJI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FUJI_PY="${FUJI_PY:-$FUJI_ROOT/.venv/bin/python}"
PORT="${PORT:-8087}"
DEVICE="${DEVICE:-cuda}"

if [ ! -x "$FUJI_PY" ]; then
  echo "[run] FUJI venv python not found at $FUJI_PY (create .venv, then: uv pip install -e \".[viz]\")" >&2
  exit 2
fi
if [ "$#" -lt 1 ]; then
  echo "usage: $0 <five-point checkpoint dir> [extra infer_s1_* flags]" >&2
  exit 2
fi
CKPT="$1"; shift
if [ ! -f "$CKPT/config.json" ]; then
  echo "[run] FATAL: '$CKPT' has no config.json" >&2
  exit 2
fi

# LAYOUT CHECK BEFORE A GPU IS TOUCHED. The viewer itself handles every layout, so it would open a
# perfectly good page for a 3-tracker checkpoint and merely disable the body — with the reason a
# few lines into stderr. Say it here instead, and point at the right launcher.
CKPT_MODE="$("$FUJI_PY" -c \
  "import json,sys; print(json.load(open(sys.argv[1])).get('hl_data',{}).get('mode',''))" \
  "$CKPT/config.json")"
HELD_NOTE=""
case "$CKPT_MODE" in
  fivepoint_geom_*) ;;
  left_only|right_only|both_hands|both_hands_head|both_hands_head_dual)
    # A KEYPOINT SUBSET: the roles this checkpoint does not predict are HELD at the ready pose for
    # both bodies (fuji/s1_policy/inference/keypoint_humanoid.py, `held_poses`) — the offline twin
    # of what scripts/hl_serving_5pt.py publishes for the same checkpoint. The trajectory pairs
    # draw for the predicted roles only; the body is whole.
    HELD_NOTE="the roles '$CKPT_MODE' does not predict are HELD at the ready pose (both bodies)"
    ;;
  wholebody|wholebody_geom_*)
    echo "[run] FATAL: $CKPT has hl_data.mode='$CKPT_MODE' — it predicts 29 JOINT angles, which" >&2
    echo "[run]   have no trajectory to draw. Use ./scripts/run_infer_s1_wholebody.sh." >&2
    exit 2 ;;
  "")
    echo "[run] FATAL: $CKPT/config.json has no hl_data.mode" >&2
    exit 2 ;;
  *)
    echo "[run] FATAL: $CKPT has hl_data.mode='$CKPT_MODE', which this page cannot draw a body" >&2
    echo "[run]   from: it is neither 'fivepoint_geom_*' nor a keypoint subset of the five" >&2
    echo "[run]   (left_only / both_hands / both_hands_head[_dual])." >&2
    exit 2 ;;
esac

# ACTION-CHUNK PIN, DERIVED FROM THE CHECKPOINT. layout.py resolves ACTION_HORIZON from
# FUJI_ACTION_HORIZON at IMPORT time, so a shell launcher is the only place that can still set it —
# and an H=100/200 five-point checkpoint is one of the three this preset ships. Same helper shape as
# every other launcher; the ACTIVE algorithm's block is what carries the real value.
if ! CKPT_H="$("$FUJI_PY" - "$CKPT/config.json" <<'PY'
import json, sys
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
print(horizon)
PY
)"; then
  echo "[run] FATAL: could not read the action horizon from $CKPT/config.json (above)" >&2
  exit 2
fi
if [ -n "${FUJI_ACTION_HORIZON:-}" ] && [ "$FUJI_ACTION_HORIZON" != "$CKPT_H" ]; then
  echo "[run] FATAL: FUJI_ACTION_HORIZON=$FUJI_ACTION_HORIZON in the environment, but" >&2
  echo "[run]        $CKPT declares action_horizon=$CKPT_H. Refusing to guess which is meant." >&2
  exit 2
fi
export FUJI_ACTION_HORIZON="$CKPT_H"

CKPT_ALGO="$("$FUJI_PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('algo','diffusion'))" \
  "$CKPT/config.json")"
case "$CKPT_ALGO" in
  diffusion) ENTRY=infer_s1_diffusion.py ;;
  flow)      ENTRY=infer_s1_flow.py ;;
  act)       ENTRY=infer_s1_act.py ;;
  *) echo "[run] FATAL: unknown algo '$CKPT_ALGO' in $CKPT/config.json" >&2; exit 2 ;;
esac

EXTRA=()
# --keypoint-ik unless NO_IK, and unless the caller already said something about it.
case " $* " in
  *" --keypoint-ik "*|*" --display-frame "*) ;;
  *) [ "${NO_IK:-0}" = "1" ] || EXTRA+=(--keypoint-ik) ;;
esac

echo "[run] OFFLINE FIVE-POINT VIEWER on http://localhost:$PORT"
echo "[run]   python: $FUJI_PY (device=$DEVICE)"
echo "[run]   checkpoint: $CKPT  algo=$CKPT_ALGO  layout=$CKPT_MODE  entry=$ENTRY"
if [ -n "$HELD_NOTE" ]; then
  echo "[run]              chunk=$FUJI_ACTION_HORIZON steps @ 30 Hz -> the predicted role(s) only;"
  echo "[run]              $HELD_NOTE"
else
  echo "[run]              chunk=$FUJI_ACTION_HORIZON steps @ 30 Hz -> 5 poses + 2 grips + 2 wrenches"
fi
if [ "${NO_IK:-0}" = "1" ]; then
  echo "[run]   NO_IK=1: trajectories only. The five blue/orange pairs still draw; there is no body."
else
  echo "[run]   --keypoint-ik: VIOLET = the prediction's body, PALE BLUE = the dataset target's."
  echo "[run]     Per-limb damped least squares in the SAME URDF the page renders — no second"
  echo "[run]     model, no conda env. The CHEST keypoint is the root (so its residual is 0 by"
  echo "[run]     construction) and the waist is held, which means the drawn PELVIS is a display"
  echo "[run]     artefact and not a physical claim; the panel reports how far it travelled."
  echo "[run]   It IMPLIES --display-frame g1-ready-pose. The three tracked roles and the two FK'd"
  echo "[run]     feet are in DIFFERENT source frames, so the display homes are not one coherent"
  echo "[run]     frame for five roles (measured: 543 mm of IK residual). On the ready pose the IK"
  echo "[run]     closes on the TARGET to 0.2-0.4 mm."
fi
echo "[run]   The page's own panel: 'Five-point IK body (G1)' — hide either body, pick which"
echo "[run]     series the residual reports, and read the per-step / per-chunk residual."
echo "[run]   READ THE FEET WITH CARE: the foot target is a STANCE, not a footstep (pelvis frame,"
echo "[run]     no root), its per-chunk motion is 1.9-4.4 mm against 2.3-3.2 mm of placement"
echo "[run]     noise, and its real content is the operator's YAW. See"
echo "[run]     docs/RUNBOOK_diffusion_umi_5pt.md."

exec "$FUJI_PY" "$FUJI_ROOT/scripts/$ENTRY" "$CKPT" \
  --device "$DEVICE" \
  --port "$PORT" \
  "${EXTRA[@]}" \
  "$@"
