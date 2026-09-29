#!/bin/bash
# 8-GPU training for the 5-POINT and DENSE twins of the LOCAL R1 anneal preset:
#
#   EXP=exp:g1-wbt-5pt-hybrid-unified-anneal   184D actor (736D w/ history 4)  [DEFAULT]
#   EXP=exp:g1-wbt-hybrid-unified-anneal       183D actor (732D w/ history 4)
#
# Both are `exp:g1-wbt-3pt-hybrid-unified-anneal` with the OBSERVATION swapped and NOTHING
# else: the injector cfg (R1 anneal — live Jᵀ servo, authority annealed 1→0 performance-
# gated by ServoAuthorityCurriculum), the 17-term unified reward INCLUDING the two TCP
# precision terms (tcp_position w=4.0 σ=0.10 m, tcp_orientation w=2.0 σ=0.25 rad), the
# force-alpha ramp from 0, the force-runaway estop, the yam URDF, the algo and the
# WBT_HAND_MASS_DR phase gating are all the SAME OBJECTS as the 3pt twin's. So "force
# handling identical to the 3pt anneal preset" is structural, not a promise.
#
# WHAT THE TRACKING FRONT-END IS:
#   5pt  — the 3pt keypoint set (left/right hand @ the 0.242 m fingertip TCP, head @
#          torso_link +0.35 z) plus BOTH FEET (`*_ankle_roll_link`, bare origin — no sole
#          lever). That is exactly `exp:g1-wbt-maskedmimic`'s `five_point` hover mode after
#          its alias table (torso→torso_link, left/right_hand→*_wrist_yaw_link).
#          vr_5point_pos_tcp(15) + vr_5point_orn(20, quaternion) replace the 9+12 of 3pt.
#   dense — no keypoints at all: `exp:g1-wbt`'s motion_command (58D = all 29 joint targets
#          + velocities), which also subsumes the 24D command_lower_body slice. The two
#          hands' TCP is still explicitly priced by the inherited tcp_precision rewards.
#
# WHY ITS OWN LAUNCHER: train_hybrid_3pt_unified_8gpu.sh's preflight asserts the actor obs
# slot 01 is literally `obs01_vr_3point_pos_tcp` (its stale-checkout guard). Neither preset
# here has that slot, so reusing that script would either fail spuriously or require
# loosening a guard that protects the live pfl96opy lineage. Left untouched on purpose.
#
# ⚠️ TRAIN FROM SCRATCH. Every unified checkpoint is 170D (185D for the global-vel family);
# 184D/183D match neither, so `--training.checkpoint` from m3s0pw9i / 4o9zbs7i / iss0qs71 /
# pfl96opy CANNOT be used — it will fail on a shape mismatch, which is the desired outcome.
#
# ⚠️ DEPLOY IS NOT WIRED. `wbt_wrappers_inference` has no 184D/183D preset and the mjswan
# viz scripts are keyed to the 170D ones, so these produce training checkpoints only.
#
# Inherited caveats (identical to exp:g1-wbt-3pt-hybrid-unified-anneal, by construction):
#   - Orientation is a QUATERNION, so the known double-cover OOD at deploy applies (the 6D
#     rep lives only in the `-global-vel` family).
#   - Root-invariant (no motion_ref_pos_b / _vel_b / base_lin_vel in the actor) — tracks in
#     place, does not walk.
#
# Data: /data/dataset/sonic_manipulation_holosoma (native holosoma format).
#
# Usage:
#   conda activate hssim
#   # 5-point (default):
#   bash training_runs/train_hybrid_unified_5pt_dense_8gpu.sh
#
#   # dense 29-DOF tracking:
#   EXP=exp:g1-wbt-hybrid-unified-anneal bash training_runs/train_hybrid_unified_5pt_dense_8gpu.sh
#
#   # fewer GPUs / dataset elsewhere:
#   NGPUS=4 NUM_ENVS=16384 bash training_runs/train_hybrid_unified_5pt_dense_8gpu.sh
#   MOTION_DIR=/your/path bash training_runs/train_hybrid_unified_5pt_dense_8gpu.sh
#
#   # ACTOR joint-velocity dropout (DEFAULT OFF, inherited from the 3pt-hybrid lineage):
#   WBT_JV_DROPOUT_ENABLED=1 WBT_JV_DROPOUT_PROB=0.5 \
#       bash training_runs/train_hybrid_unified_5pt_dense_8gpu.sh
#
#   # phase 2 — gripper-mass DR + the m_distal sync (one knob drives both):
#   WBT_HAND_MASS_DR=1 bash training_runs/train_hybrid_unified_5pt_dense_8gpu.sh
#
# Notes:
#   - 33-DOF g1_29dof_yam.urdf (29 actuated + 4 gripper jaws): the "29 != 33 actuators"
#     IsaacLab warning is expected/benign.
#   - W&B watch-list, same as the 3pt anneal twin: force_track_pos_eq2 and
#     force_error_direct climbing, tcp_position_error_exp / tcp_orientation_error_exp
#     climbing, servo_alpha annealing off 1.0, integrator saturation LOW, force_runaway
#     trigger rate ~0. For 5pt additionally watch the two ankle bodies in the 14-body
#     position term — feet are now BOTH a tracked obs point and a reward body.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export OMNI_KIT_ACCEPT_EULA=YES
# W&B: your own `wandb login` is used (export WANDB_BASE_URL for a self-hosted server). No
# account? Append `--logger.mode offline` (tyro is last-wins, extra args are forwarded).

# ACTOR joint-velocity dropout knobs (read at import time by hybrid_3pt.py, whose term both
# new layouts reuse). Exported at their DEFAULT-DISABLED values so they land in the launched
# env + the W&B config; overridable from the caller.
export WBT_JV_DROPOUT_ENABLED="${WBT_JV_DROPOUT_ENABLED:-0}"
export WBT_JV_DROPOUT_PROB="${WBT_JV_DROPOUT_PROB:-0.5}"
export WBT_JV_DROPOUT_MODE="${WBT_JV_DROPOUT_MODE:-per_episode}"
export WBT_JV_DROPOUT_NOISE="${WBT_JV_DROPOUT_NOISE:-0.0}"

# Phase-2 gripper-mass DR (read at import time by tcp_precision.py). DEFAULT OFF: phase 1
# measures the precision ceiling, phase 2 then measures what robustness costs.
export WBT_HAND_MASS_DR="${WBT_HAND_MASS_DR:-0}"

NGPUS="${NGPUS:-8}"
NUM_ENVS="${NUM_ENVS:-32768}"
MOTION_DIR="${MOTION_DIR:-/data/dataset/sonic_manipulation_holosoma}"
WANDB_ENTITY="${WANDB_ENTITY:-}"  # empty = your default W&B entity
EXP="${EXP:-exp:g1-wbt-5pt-hybrid-unified-anneal}"

# Default RUN_NAME off EXP so the two twins cannot silently overwrite each other's run name.
case "$EXP" in
    exp:g1-wbt-5pt-hybrid-unified-anneal) _DEFAULT_RUN_NAME="WBT-Hybrid5ptUni-anneal-yam-8gpu" ;;
    exp:g1-wbt-hybrid-unified-anneal)     _DEFAULT_RUN_NAME="WBT-HybridDenseUni-anneal-yam-8gpu" ;;
    *)
        echo "[launcher] FAIL: this script only drives exp:g1-wbt-5pt-hybrid-unified-anneal and" >&2
        echo "[launcher]       exp:g1-wbt-hybrid-unified-anneal. Got EXP=$EXP." >&2
        echo "[launcher]       For the 3pt twins use training_runs/train_hybrid_3pt_unified_8gpu.sh." >&2
        exit 1
        ;;
esac
RUN_NAME="${RUN_NAME:-$_DEFAULT_RUN_NAME}"

REPO_ROOT="$(cd "$PROJECT_DIR/../../../.." && pwd)"

# ---------------------------------------------------------------------------
# Preflight (URDF, editable imports, preset, obs layout, TCP geometry + reward, dataset).
# Skip with SKIP_PREFLIGHT=1.
# ---------------------------------------------------------------------------
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    _fail() { echo "[preflight] FAIL: $1" >&2; exit 1; }

    _URDF="$REPO_ROOT/thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof_yam.urdf"
    [ -f "$_URDF" ] || _fail "yam URDF missing ($_URDF). Run: git submodule update --init --recursive"

    python -c "import holosoma, wbt_training" 2>/dev/null \
        || _fail "cannot import holosoma / wbt_training. Run: pip install -e thirdparty/holosoma/src/holosoma && pip install -e src/extensions/wbt_wrappers/wbt_training"
    python -c "import wbt_training.config_values.hybrid_unified_5pt_dense" 2>/dev/null \
        || _fail "hybrid_unified_5pt_dense module not importable — this checkout predates the 5pt/dense presets (2026-08-27). git pull on this box and re-run."

    # The preset must resolve, carry the expected tracking front-end, and still be the R1
    # anneal twin with the TCP precision reward. Same class of stale-checkout / wrong-branch
    # guard as the 3pt launcher's, keyed to what is load-bearing for THESE presets.
    python - "$EXP" <<'PY' || _fail "preset/geometry preflight failed (see message above)"
import sys

from wbt_training.config_values.experiment import DEFAULTS
from wbt_training.config_values.hybrid import TCP_OFFSET_X
from wbt_training.config_values.hybrid_unified_5pt_dense import VR_5POINT_BODIES, VR_5POINT_OFFSETS_TCP
from wbt_training.config_values.tcp_precision import HAND_MASS_DR_ENABLED

exp_name = sys.argv[1]
key = exp_name.removeprefix("exp:").replace("-", "_")
if key not in DEFAULTS:
    print(f"[preflight] unknown preset {exp_name!r} (key {key!r})", file=sys.stderr)
    raise SystemExit(1)
exp = DEFAULTS[key]
# The 3pt anneal twin is the reference for everything except `observation`. Comparing against
# ITS objects (rather than importing _UNIFIED_ANNEAL_CFG directly) keeps this phase-agnostic:
# WBT_HAND_MASS_DR=1 legitimately swaps the injector for the sync_distal_mass_from_sim twin.
ref = DEFAULTS["g1_wbt_3pt_hybrid_unified_anneal"]
ref_inj = ref.command.setup_terms["hybrid_contact_injector"].params["hybrid_config"]

terms = list(exp.observation.groups["actor_obs"].terms)
EXPECTED_SLOT0 = {
    "g1_wbt_5pt_hybrid_unified_anneal": "obs01_vr_5point_pos_tcp",
    "g1_wbt_hybrid_unified_anneal": "obs00_motion_command",
}
want = EXPECTED_SLOT0[key]
if want not in terms:
    print(f"[preflight] {exp_name} actor obs is {terms}, expected a {want!r} slot", file=sys.stderr)
    raise SystemExit(1)

if key == "g1_wbt_5pt_hybrid_unified_anneal":
    # Hands must still be ON the force TCP, and the feet must carry no lever. Getting either
    # wrong re-splits position from force, which is the whole point of the unified family.
    hands = [list(o) for o in VR_5POINT_OFFSETS_TCP[:2]]
    if hands != [[TCP_OFFSET_X, 0.0, 0.0]] * 2:
        print(f"[preflight] 5pt hand offsets {hands} != force TCP {[TCP_OFFSET_X, 0.0, 0.0]}", file=sys.stderr)
        raise SystemExit(1)
    if [list(o) for o in VR_5POINT_OFFSETS_TCP[3:]] != [[0.0, 0.0, 0.0]] * 2:
        print(f"[preflight] 5pt foot offsets are {VR_5POINT_OFFSETS_TCP[3:]}, expected bare origins", file=sys.stderr)
        raise SystemExit(1)
    tracked = list(exp.command.setup_terms["motion_command"].params["motion_config"].body_names_to_track)
    missing = [b for b in VR_5POINT_BODIES if b not in tracked]
    if missing:
        print(f"[preflight] 5pt bodies {missing} are not in body_names_to_track", file=sys.stderr)
        raise SystemExit(1)
else:
    stray = [t for t in terms if "vr_" in t or "command_lower_body" in t]
    if stray:
        print(f"[preflight] dense preset carries keypoint slots {stray} — it should be joint-target only", file=sys.stderr)
        raise SystemExit(1)

# R1 anneal, not R2 observe-only: these were asked for as `-anneal`.
inj = exp.command.setup_terms["hybrid_contact_injector"].params["hybrid_config"]
if inj is not ref_inj or inj.servo_observe_only or not inj.servo_enabled:
    print(
        f"[preflight] {exp_name} is not on the R1 anneal injector cfg "
        f"(servo_enabled={inj.servo_enabled} servo_observe_only={inj.servo_observe_only}, "
        f"same object as the 3pt anneal twin: {inj is ref_inj})",
        file=sys.stderr,
    )
    raise SystemExit(1)
if "servo_authority" not in exp.curriculum.setup_terms:
    print(f"[preflight] {exp_name} has no servo_authority curriculum — the anneal would never happen", file=sys.stderr)
    raise SystemExit(1)

rw = exp.reward.terms
missing = [t for t in ("tcp_position_error_exp", "tcp_orientation_error_exp") if t not in rw]
if missing:
    print(f"[preflight] {exp_name} reward is MISSING {missing} — this checkout predates the TCP precision change.", file=sys.stderr)
    raise SystemExit(1)

# Phase gating: DR and the m_distal sync must be ON together or OFF together (one knob).
has_dr = "randomize_mass_startup" in exp.randomization.setup_terms
if (has_dr, inj.sync_distal_mass_from_sim) != (HAND_MASS_DR_ENABLED, HAND_MASS_DR_ENABLED):
    print(
        f"[preflight] phase mismatch: WBT_HAND_MASS_DR={int(HAND_MASS_DR_ENABLED)} but "
        f"mass_dr={has_dr} sync_distal={inj.sync_distal_mass_from_sim}. Mass DR without the "
        "m_distal sync desyncs F_meas by up to 2.04 N (> theta_on=2.0 N).",
        file=sys.stderr,
    )
    raise SystemExit(1)

# Everything except `observation` must still be the 3pt anneal twin's OBJECTS. This is the
# guard for the actual design claim; a value-equal copy here would be a silent fork.
drifted = [
    f
    for f in ("command", "reward", "curriculum", "termination", "randomization", "robot", "algo")
    if getattr(exp, f) is not getattr(ref, f)
]
if drifted:
    print(f"[preflight] {exp_name} has forked from exp:g1-wbt-3pt-hybrid-unified-anneal on {drifted}", file=sys.stderr)
    raise SystemExit(1)

phase = "2 (TCP precision + gripper-mass DR + m_distal sync)" if HAND_MASS_DR_ENABLED else "1 (TCP precision only)"
front = "5 points: 2 hands @ TCP + head + 2 feet" if key.startswith("g1_wbt_5pt") else "DENSE 29-DOF joint targets"
print(f"[preflight] OK: {exp_name} | {len(terms)} obs terms | {front}")
print(f"[preflight] force stack == exp:g1-wbt-3pt-hybrid-unified-anneal (same objects), R1 anneal, servo live")
print(
    f"[preflight] phase {phase} | {len(rw)} reward terms | "
    f"tcp_position w={rw['tcp_position_error_exp'].weight} sigma={rw['tcp_position_error_exp'].params['sigma']}, "
    f"tcp_orientation w={rw['tcp_orientation_error_exp'].weight} sigma={rw['tcp_orientation_error_exp'].params['sigma']}"
)
PY

    # These presets have no compatible checkpoint (see the header) — refuse a resume rather
    # than let torchrun die deep inside the load with a raw shape mismatch.
    for _a in "$@"; do
        case "$_a" in
            --training.checkpoint|--training-checkpoint|--training.checkpoint=*|--training-checkpoint=*)
                _fail "these presets are 184D/183D and every existing unified checkpoint is 170D — there is nothing to resume from. Train from scratch (drop --training.checkpoint)."
                ;;
        esac
    done

    [ -d "$MOTION_DIR" ] || _fail "motion dir not found: $MOTION_DIR. Set MOTION_DIR=/your/path."
    _NNPZ=$(find "$MOTION_DIR" -maxdepth 1 -name '*.npz' 2>/dev/null | head -1)
    [ -n "$_NNPZ" ] || _fail "no .npz files under MOTION_DIR=$MOTION_DIR"

    echo "[preflight] OK: URDF, imports, dataset ($MOTION_DIR) present."
    echo "[preflight] EXP=$EXP RUN_NAME=$RUN_NAME NGPUS=$NGPUS NUM_ENVS=$NUM_ENVS"
    echo "[preflight] JV dropout: enabled=$WBT_JV_DROPOUT_ENABLED prob=$WBT_JV_DROPOUT_PROB mode=$WBT_JV_DROPOUT_MODE noise=$WBT_JV_DROPOUT_NOISE (default DISABLED)"
    echo "[preflight] hand-mass DR: WBT_HAND_MASS_DR=$WBT_HAND_MASS_DR (default 0 = phase 1, precision only)"
fi

cd "$REPO_ROOT"  # repo root (holosoma_extensions)

exec torchrun \
    --standalone --nnodes=1 --nproc_per_node="$NGPUS" \
    -m wbt_training.train_agent \
    "$EXP" \
    simulator:isaacsim \
    --robot.asset.urdf-file g1/g1_29dof_yam.urdf \
    --training.headless True \
    --training.num-envs "$NUM_ENVS" \
    --algo.config.num-mini-batches 32 \
    --logger.video.enabled False \
    --logger.entity "$WANDB_ENTITY" \
    --logger.project "${WANDB_PROJECT:-FLoRe}" \
    --logger.name "$RUN_NAME" \
    --command.setup-terms.motion-command.params.motion-config.motion-dir "$MOTION_DIR" \
    "$@"
