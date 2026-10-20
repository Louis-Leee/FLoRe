#!/usr/bin/env bash
# diffusion_umi_wholebody -- the WHOLE-BODY sibling of diffusion_umi_3tracker.
#
# Same model, same UMI recipe, same rig, same 30 Hz / 50-step v3 grid. What changes is WHAT
# the policy outputs: the G1's 29 JOINT ANGLES, i.e. the configuration the `dense` low level
# consumes directly, instead of three keypoint POSES a deploy-time motion completion still has
# to turn into one. The completion moves to DATA-PREPARATION time.
#
#                          diffusion_umi_3tracker      diffusion_umi_wholebody
#   layout mode            both_hands_head_dual        wholebody
#   --hands (converter)    both_head_dual              both_head_dual  (the SAME zarr, plus
#                                                      wb_* arrays -- see PREREQUISITE)
#   cameras (obs)          2  camera0_rgb + camera1_rgb        same
#   force in (obs)         (1, 12)  two sensors                same
#   state (obs)            20   L/R rot6d+grip, head rot6d
#                                                      31   29 JOINT ANGLES + L/R grip
#   action out             50 x 41                     50 x 43
#                          L[pos3 rot6d6 grip1 force6] R[same] head[pos3 rot6d6]
#                                                      [joints 29][L grip1 force6][R grip1 force6]
#   frame                  anchor-relative per keypoint        ABSOLUTE joint space
#
# It does NOT touch the 3-tracker preset. `wholebody` is a NEW layout mode over the SAME
# --hands value, and a wholebody dataset is a strict SUPERSET of a dual one -- so one zarr
# trains either preset and every shipped checkpoint keeps its column meaning. The
# no-regression direction is asserted (tests/test_diffusion_umi_wholebody.py::
# test_the_dual_preset_is_unchanged_by_augmentation).
#
# WB_TARGET -- the one thing about the joint columns that IS settable, and the only knob here
# that changes what the network is asked to learn:
#
#   WB_TARGET=absolute  (default)  the export's own radians. Row 0 == the state, to 0.000e+00.
#   WB_TARGET=delta                q[k] - q[0], i.e. anchor-relative joint MOTION; row 0 is
#                                  exactly zero and the posture offset is gone from the target.
#
# The STATE stays absolute in both (proprioception must be: a delta state relative to itself is
# identically zero), and the deploy-side JointClutch publishes base + (q[i] - q[0]), which is
# invariant to a constant offset -- so THE PUBLISHED WIRE IS THE SAME FUNCTION OF THE OUTPUT in
# both modes. What differs is what the network represents: `delta` asks for the motion alone,
# `absolute` asks for the motion plus a posture offset the clutch then subtracts back out. That
# is why this is a per-run A/B and not a second layout: same widths, same wire, same low level.
#
# ⚠️ It lands in the EXACT-equality dataset_knobs, so norm stats are NOT shared -- this script
# points at a `_delta`-suffixed file and the trainer refuses a crossed one by name. It is also
# baked into the checkpoint, so serving and the offline viewer follow it automatically
# (--wb-clutch none is REFUSED for a delta checkpoint: near-zero radians published absolutely is
# the 29-joints-zero pose).
#
# Three things that are NOT settable here, because they are the layout:
#   * nothing is ANCHOR-RELATIVE IN POSE SPACE. pose_dims() is empty either way, so serving has
#     no SE(3) un-relativization to do -- WB_TARGET picks the JOINT-space parameterization only.
#   * there is NO ROOT CHANNEL. DenseWire sends (29 joint pos, 29 joint vel) and derives the
#     velocities from the positions by finite difference, so 29 columns on this grid is the
#     complete command. A root the policy predicted would command nothing.
#   * the CHEST puck still gates every sample even though no keypoint block owns it -- it is
#     the completion's root conditioning, so a chest glitch corrupts all 29 joint columns.
#
# PREREQUISITE, and the ONE new step versus the 3-tracker preset. The dataset must carry the
# wb_* joint arrays, which motion completion writes ONCE, offline:
#
#   cd third_party/UMI_collection
#   ./run_mc_to_hl_wholebody.sh data/datasets/<dual>.zarr.zip -o data/datasets/<dual>_wb.zarr.zip
#   ./run_mc_to_hl_wholebody.sh data/datasets/<dual>_wb.zarr.zip --check     # verify
#
# That needs the Motion_Completion checkout + its g1_mosyn conda env + a downloaded checkpoint
# (MOTION_COMPLETION_{DIR,PY,CKPT}); see third_party/UMI_collection/motion_completion/README.md.
# ⚠️ --cond-mode is stamped as attrs.wb_cond_mode and every shard in a run must agree (the
# joint values are absolute, so two anchors are two posture spaces); HLDataset refuses a mix.
#
# WB_STATE -- WHAT THE OBSERVATION CARRIES. The second settable axis, and the one added because
# the first choice here turned out to be the deploy problem. `joints` feeds the 29 completed
# joint angles, which is honest proprioception and the channel MEASURED to be far out of
# distribution at the robot: over the 159 valid episode starts of session_2026.09.02_merged3 the
# export's joints sit 0.2750 rad (mean) / 0.6643 rad (worst) from the pose the G1 holds after
# **A**, with 16 of 29 past the low level's 0.20 rad pre-arm gate. The other three values feed a
# pelvis-frame GEOMETRIC reduction of those same joints instead:
#
#   WB_STATE      mode                   state                              state_dim
#   joints        wholebody              [29 joints][Lg][Rg]                   31   (default)
#   xy_rpy        wholebody_geom_xy_rpy  [rel_xy 4][rpy 6][Lg][Rg]             12
#   xy            wholebody_geom_xy      [rel_xy 4][Lg][Rg]                     6
#   rpy           wholebody_geom_rpy     [rpy 6][Lg][Rg]                        8
#   none          wholebody_geom_none    [Lg][Rg]                               2
#
#   rel_xy = [(p_L - p_R).x, (p_L - p_R).y, p_L.x, p_L.y]   metres, PELVIS frame
#   rpy    = extrinsic-XYZ Euler of R_L then R_R            radians, PELVIS frame  (no pelvis
#            rpy: in the pelvis frame the pelvis is identity)
#
# In those features the SAME 159 episode starts land 0.28-0.58 sigma from the ready pose on
# every x/rotation channel. The residue is the two `y` channels at 2.3-2.5 sigma (~9 cm): the
# completion places the operator's hands wider apart than the ready pose does, which is a
# PLACEMENT property of the export and not something an observation transform can remove.
# The ACTION is byte-identical across all five (43 columns, same joints, same `dense` low
# level), so the 2x2 grid over {rel_xy} x {rpy} is a controlled experiment on the observation.
# ⚠️ The geom modes FK the joints at load time through the shared KeypointFK, so the training
# box needs `yourdfpy` plus the holosoma_extensions + UMI_collection submodules. Refused by name
# if absent -- never zero-filled, because a zero geometry is a plausible posture.
#
# NOISE_XY_M / NOISE_RPY_DEG -- zero-mean Gaussian jitter on the geometric state columns of every
# TRAINING sample (validation and the norm-stats fit stay clean). Defaults 0.010 m / 2.0 deg,
# chosen to COVER the measured systematic deploy offsets (lr_dx 11 mm, lp_x 15 mm, left
# roll/pitch/yaw 1.3/4.2/2.2 deg) so a deployed value lands ~1 sigma inside the training
# distribution. ⚠️ These features are TIGHT -- over 7153 frames lr_dx/lr_dy std is 2.6/2.2 cm and
# left roll/yaw std is 2.9/4.4 deg -- so the 2.5 cm / 10-15 deg a proprioception-noise habit
# reaches for is 1.0x and 3-4x the SIGNAL'S OWN std and would erase the channel while leaving the
# loss curve looking healthy. Treat anything above ~0.015 m / ~3 deg as an "is this feature
# useful at all" upper bound. Ignored (and refused if nonzero) for WB_STATE=joints|none.
#
# H -- the predicted chunk length, in steps on the fixed 30 Hz grid. 50 (default) = 1.67 s, the
# shipped contract; 100 = 3.33 s; 200 = 6.67 s. A SHORTER horizon is a strict prefix of a longer
# one at the same dt, so ONE 200-step conversion serves all three -- but a 50-step dataset cannot
# serve H=100, and the SESSION table below declares each entry's stored horizon so that is
# refused here with the re-convert command rather than deep inside the loader.
# ⚠️ Long chunks are not free of data consequences: on 0902_merged3 (median episode 177 frames =
# 5.9 s) only 55.5% of anchors have a full 100-step future inside their episode and 23.5% a full
# 200-step one. The rest are tail-padded and masked -- correct, but it means the late steps of a
# 200-step chunk are supervised by a quarter of the anchors. Read the per-group val metrics
# before concluding a long chunk did not help.
#
# Usage: [SESSION=0831_merged] [WB_TARGET=absolute] [WB_STATE=joints] [H=50] \
#            [NOISE_XY_M=0.010] [NOISE_RPY_DEG=2.0] [RECIPE=umi] [BATCH=24] [CKPT_ROOT=...] \
#            [DSDIR=...] run_train_diffusion_umi_wholebody.sh [n_gpus] [gpu_group] [extra args...]
#
#   n_gpus     1 (default, plain python) .. 8 (torchrun + DDP)
#   gpu_group  'low' = GPUs 0-3, 'high' = 4-7, 'all' = 0-7.
#              Defaults to 'low' for n_gpus<=4 and 'all' above that.
#
# Examples:
#   scripts/run_train_diffusion_umi_wholebody.sh                       # 1 GPU
#   scripts/run_train_diffusion_umi_wholebody.sh 4                     # GPUs 0-3, DDP
#   SESSION=0831_merged BATCH=16 scripts/run_train_diffusion_umi_wholebody.sh 8
#
#   # The absolute-vs-delta A/B, both at once on one 8-GPU box (separate output dirs and W&B
#   # run names are DERIVED from WB_TARGET, so the two cannot overwrite each other):
#   SESSION=0902_merged3 WB_TARGET=absolute ... 4 low   &
#   SESSION=0902_merged3 WB_TARGET=delta    ... 4 high  &
#
#   # One cell of the geometric-state x horizon grid, 8 GPUs. Every axis is in the output dir
#   # and the W&B name, so cells cannot overwrite each other:
#   SESSION=0902_merged3_h200 WB_STATE=xy_rpy H=100 \
#       scripts/run_train_diffusion_umi_wholebody.sh 8
#   # ...and the whole grid, queued: scripts/run_train_wb_geom_grid.sh
#   #    (docs/RUNBOOK_wb_geom_ablation.md is the full bring-up, including the data prep)
#
# COST is the 3-tracker preset's: two ViT forwards per sample dominate, so BATCH defaults to
# the same 24. The wider action head (43 vs 41) and the wider state (31 vs 20) are noise
# against that.
set -euo pipefail

cd "$(dirname "$0")/.."

# --print-plan: resolve THIS run's mode / dataset / norm-stats path / output dir, print them as
# one TAB-separated line, and stop before any GPU, model or file check. It exists so
# run_train_wb_geom_grid.sh can ASK for those four strings instead of re-deriving the naming
# rules -- two copies of "which suffix goes in the stats filename" is exactly the drift the
# repo's exact-equality dataset_knobs discipline is there to make loud, and a queue that guesses
# wrong wastes a slot per cell. Consumed before N_GPUS so it can be the only argument.
PRINT_PLAN=0
if [[ "${1:-}" == "--print-plan" ]]; then PRINT_PLAN=1; shift; fi

N_GPUS=${1:-1}
shift || true
GPU_GROUP=${1:-}
# The group is OPTIONAL, so only consume it when it really is a group name -- otherwise
# `... 4 --max-steps 50000` would eat the flag.
case "$GPU_GROUP" in
    low|high|all) shift ;;
    *) GPU_GROUP="" ;;
esac
if ! [[ "$N_GPUS" =~ ^[1-8]$ ]]; then
    echo "FATAL: n_gpus must be 1-8, got '$N_GPUS'" >&2; exit 1
fi
if [[ -z "$GPU_GROUP" ]]; then
    if (( N_GPUS > 4 )); then GPU_GROUP=all; else GPU_GROUP=low; fi
fi
case "$GPU_GROUP" in
    low)  ALL_GPUS=(0 1 2 3) ;;
    high) ALL_GPUS=(4 5 6 7) ;;
    all)  ALL_GPUS=(0 1 2 3 4 5 6 7) ;;
esac
if (( N_GPUS > ${#ALL_GPUS[@]} )); then
    echo "FATAL: gpu_group '$GPU_GROUP' has ${#ALL_GPUS[@]} GPUs but n_gpus=$N_GPUS." >&2
    exit 1
fi
GPUS=$(IFS=,; echo "${ALL_GPUS[*]:0:$N_GPUS}")

# Dataset table. Same qualification as the 3-tracker preset (both hands' camera + Bota +
# DM4310 attached AND converted with --hands both_head_dual), PLUS the wb_* augmentation --
# so every entry here is a `_wb.zarr.zip`, and the un-augmented sibling is refused by name.
#
# FH / H are EXPORTED, not taken from the caller: torchrun passes the environment to every
# rank and both are resolved at IMPORT by flore.s1_policy.data.layout, so the ranks must agree
# before the first collective.
#
# The exclusions are the SOURCE conversion's -- see run_train_diffusion_umi_3tracker.sh for
# why the 0830/0831 head-rotation glitches must be excluded at CONVERT time. That matters more
# here, not less: the chest puck is the completion's root channel, so a head glitch corrupts
# all 29 joint columns rather than 6 of 41.
VAL_EPS=4
# The chunk length THIS RUN trains, and the length the dataset STORES. They are separate because
# a shorter horizon is a strict prefix of a longer one at the same dt: one 200-step conversion
# serves H=50/100/200, and STORED_H (set per SESSION entry below) is what refuses the other
# direction here instead of inside HLDataset._check_file.
H=${H:-50}
if ! [[ "$H" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: H must be a positive integer number of 30 Hz steps, got '$H'" >&2; exit 1
fi
STORED_H=50
FH=1
DSDIR=${DSDIR:-third_party/UMI_collection/data/datasets}
SESSION=${SESSION:-0831_merged}
WB_TARGET=${WB_TARGET:-absolute}
case "$WB_TARGET" in
    absolute|delta) ;;
    *) echo "FATAL: WB_TARGET must be 'absolute' or 'delta', got '$WB_TARGET'" >&2; exit 1 ;;
esac
# WB_STATE -> layout mode. A named MODE rather than a flag or an env pin because state_dim moves
# with the composition and `layout_mode` is what validate_model_layout pins a checkpoint against;
# see the header and flore/s1_policy/data/layout.py. `joints` is the default so an existing
# command trains exactly the same bytes it did before this axis existed.
WB_STATE=${WB_STATE:-joints}
case "$WB_STATE" in
    joints) MODE=wholebody ;;
    xy_rpy) MODE=wholebody_geom_xy_rpy ;;
    xy)     MODE=wholebody_geom_xy ;;
    rpy)    MODE=wholebody_geom_rpy ;;
    none)   MODE=wholebody_geom_none ;;
    *) echo "FATAL: WB_STATE must be joints | xy_rpy | xy | rpy | none, got '$WB_STATE'" >&2
       exit 1 ;;
esac
# Jitter on the geometric state columns, TRAIN samples only. Defaults cover the measured deploy
# offsets (see the header); zeroed for the two modes with no geometric block, because the trainer
# REFUSES a nonzero sigma there rather than silently ignoring it -- a no-op augmentation is
# invisible in the loss, in the widths and in W&B.
NOISE_XY_M=${NOISE_XY_M:-0.010}
NOISE_RPY_DEG=${NOISE_RPY_DEG:-2.0}
case "$WB_STATE" in
    joints|none) NOISE_XY_M=0; NOISE_RPY_DEG=0 ;;
    xy)          NOISE_RPY_DEG=0 ;;
    rpy)         NOISE_XY_M=0 ;;
esac
case "$SESSION" in
    # The H=200 re-export of 0902_merged3: the SAME 167 episodes (same session, same 18-take
    # --exclude list, same `--cond-mode absolute --dz-anchor hand` completion), converted at
    # `--hl-action-horizon 200`. It is a SEPARATE entry rather than a replacement because the
    # window arrays are 4x wider on disk and every existing checkpoint's norm stats were fitted
    # against the 50-step file. Serves H=50, 100 and 200 (prefix relation).
    # Built by docs/RUNBOOK_wb_geom_ablation.md step 1.
    0902_merged3_h200)
        DS=("$DSDIR/session_2026.09.02_merged3_v3_dual_clean167_h200_wbabs.zarr.zip")
        STORED_H=200 ;;
    # 167 episodes / 38663 frames, THREE 2026-09-02 sittings of one task merged, and the first
    # wholebody dataset exported at `--cond-mode absolute --dz-anchor hand` as its PRIMARY (not
    # as an A/B). It supersedes 0902_merged: same task, same rig, +49 clean episodes.
    # ⚠ `attrs.wb_cond_mode='absolute'`, so it is NOT mixable in one run with any `_wb` file
    # above -- and read the 0902_merged_abs block below before choosing it, because the measured
    # comparison on the 118-episode subset preferred `home`. It is the default for THIS session
    # because that is what was asked for and shipped; the `home` twin of these 167 episodes does
    # not exist.
    0902_merged3) DS=("$DSDIR/session_2026.09.02_merged3_v3_dual_clean167_wbabs.zarr.zip") ;;
    # 118 episodes / 27351 frames, two 2026-09-02 sittings of one task merged. The NEWEST
    # wholebody dataset and the first one built as a wholebody dataset from the start.
    # ⚠ 6 of its 118 episodes ran at 15.00 Hz (every other camera frame dropped) and are
    # written wb valid=0 by mc_to_hl_wholebody -- deliberately, since the completion would
    # otherwise fabricate dynamics at half its training rate. So the supervised fraction is
    # 85.98% of steps rather than ~100%, and that is the file, not a fitting mistake.
    # NOT the default: the default stays 0831_merged so an existing command keeps training
    # the same bytes. Pass SESSION=0902_merged explicitly.
    0902_merged) DS=("$DSDIR/session_2026.09.02_merged_v3_dual_clean118_wb.zarr.zip") ;;
    # The SAME 118 episodes exported at `--cond-mode absolute --dz-anchor hand` instead of
    # `home`. Kept as a registered A/B rather than a note, because `attrs.wb_cond_mode` is in
    # the exact-equality knobs so the two can never be mixed in one run -- switching is a
    # re-export + a re-fit + a re-train, and this entry is what makes that one command.
    # ⚠ MEASURED, it is WORSE on every deploy-relevant number, so it is not the recommendation.
    # Against the FK'd ready pose over all 112 valid episode starts:
    #     mean |gap| over 29 joints        home 0.2016 rad   absolute 0.2756
    #     joints past the 0.20 rad gate    home 13/29        absolute 17/29
    #     worst single joint gap           home 0.5385 rad   absolute 1.2449
    #     left_knee gap                    home -0.5166      absolute -0.5257   <- unchanged
    #     spread of the episode STARTS     home 0.2838 rad   absolute 1.7353
    # The last row is the decisive one: `home` re-bases every take onto ONE canonical anchor,
    # so the start postures are 0.28 rad apart and `state_seed_from_zarr`'s mean IS a pose;
    # `absolute` keeps each take's own world pose, so the operator's stance and arm-orientation
    # variation lands directly in the joint targets (right_wrist_roll +0.63, shoulder_roll
    # +-0.53) and the seed is meaningless. And the knee row shows what the placement CANNOT
    # fix: the completion straightens out of the ready pose's 38 deg crouch whatever it is
    # handed, so the 0.52 rad crouch gap is the completion MODEL's property, not the anchor's.
    # The offline viewer's per-window numbers (wrist IK 0.1 mm, foot slide, planted-foot z) are
    # WITHIN-episode completion quality and are good in BOTH modes -- the frame chain is sub-mm
    # in every placement by design -- so they do not discriminate here. Do not pick on them.
    0902_merged_abs) DS=("$DSDIR/session_2026.09.02_merged_v3_dual_clean118_wbabs.zarr.zip") ;;
    # 116 episodes, the largest dual dataset. The preset's default.
    0831_merged) DS=("$DSDIR/session_2026.08.31_merged_v3_dual_clean116_wb.zarr.zip") ;;
    # ~71 episodes, both 2026-08-30 sittings of one task merged.
    0830_merged) DS=("$DSDIR/session_2026.08.30_merged_v3_dual_clean_wb.zarr.zip") ;;
    # 32 jump-free takes; the smallest thing worth a real run. VAL_EPS=3 like the 3pt preset.
    0830_1411)   DS=("$DSDIR/session_2026.08.30_1411_v3_dual_clean32_wb.zarr.zip"); VAL_EPS=3 ;;
    *) echo "FATAL: SESSION must be 0902_merged3_h200 | 0902_merged3 | 0902_merged |" >&2
       echo "       0902_merged_abs | 0831_merged | 0830_merged | 0830_1411, got '$SESSION'" >&2
       echo "       Augment a dual conversion first (convert -> complete -> check):" >&2
       echo "         cd third_party/UMI_collection" >&2
       echo "         ./run_mc_to_hl_wholebody.sh \\" >&2
       echo "             data/datasets/session_2026.08.31_merged_v3_dual_clean116.zarr.zip \\" >&2
       echo "             -o data/datasets/session_2026.08.31_merged_v3_dual_clean116_wb.zarr.zip" >&2
       echo "         ./run_mc_to_hl_wholebody.sh <out>.zarr.zip --check" >&2
       exit 1 ;;
esac
if (( H > STORED_H )); then
    echo "FATAL: H=$H but SESSION=$SESSION stores only $STORED_H action steps." >&2
    echo "       The horizon pin can only take a PREFIX of what a dataset holds. Re-convert" >&2
    echo "       the session at the longer horizon and re-run the completion:" >&2
    echo "         cd third_party/UMI_collection" >&2
    echo "         .venv/bin/python processing/convert_to_umi.py <session> \\" >&2
    echo "             -o <out>.zarr.zip --hands both_head_dual --hl-action-horizon $H \\" >&2
    echo "             --exclude <the audit's list>" >&2
    echo "         ./run_mc_to_hl_wholebody.sh <out>.zarr.zip -o <out>_wbabs.zarr.zip \\" >&2
    echo "             --cond-mode absolute --dz-anchor hand" >&2
    echo "       ...or use SESSION=0902_merged3_h200 (see docs/RUNBOOK_wb_geom_ablation.md)." >&2
    exit 1
fi
export FLORE_FORCE_HIST_LEN="$FH"
export FLORE_ACTION_HORIZON="$H"

CAMS=(camera0_rgb camera1_rgb)

# The same two recipes as every other launcher; see run_train_diffusion_recipe.sh's header for
# what each bundles and why `umi` bundles three changes at once.
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

# `_wb` in the norm-stats name because they are NOT interchangeable with the dual preset's:
# 43 action columns instead of 41, 31 state, and -- the part no width check would catch -- the
# state and the first 29 action columns are RADIANS rather than rot6d/metres. The trainer
# asserts every width at load (`_check_norm_stats`) and compares `dataset_knobs` for exact
# equality (which now carries wb_cond_mode), so a mixed-up file is a refusal; the distinct
# name is what stops someone reaching for it in the first place.
#
# WB_TARGET adds a `_delta` SUFFIX rather than always spelling the target: `absolute` keeps the
# historical name, so every norm-stats file already fitted (and every checkpoint that recorded
# one) stays valid. The two targets differ by a per-sample offset q[0], which is a completely
# different per-column scale at an identical width -- the trainer catches a crossed file through
# dataset_knobs, and the distinct name is what stops the reach in the first place.
#
# WB_STATE and H add SUFFIXES the same way WB_TARGET does, and for the same two reasons. Both
# land in the exact-equality `dataset_knobs` the trainer compares (`mode` always; `action_horizon`
# whenever it is not the contract's 50), so a crossed file is a refusal by name -- and both
# genuinely change the statistics: WB_STATE changes the state WIDTH AND UNITS (31 radians vs 12
# metres+radians), and H changes the action QUANTILES at an identical width, because a 200-step
# chunk reaches further from its anchor than a 50-step one. `joints`/`50` keep the historical
# name, so every stats file already fitted (and every checkpoint that recorded one) stays valid.
NS_SUFFIX=""
[[ "$WB_TARGET" != "absolute" ]] && NS_SUFFIX="${NS_SUFFIX}_${WB_TARGET}"
[[ "$WB_STATE" != "joints" ]] && NS_SUFFIX="${NS_SUFFIX}_${WB_STATE}"
(( H != 50 )) && NS_SUFFIX="${NS_SUFFIX}_h${H}"
NS=checkpoints/norm_stats_hl_wb_${SESSION}${NS_SUFFIX}.json
RUN="diffusion_umi_wholebody_${SESSION}_${RECIPE}${NS_SUFFIX}"
WB_NAME="diffusion-umi-wholebody-${SESSION}-${RECIPE}${NS_SUFFIX//_/-}"
if (( N_GPUS > 1 )); then
    RUN="${RUN}_ddp${N_GPUS}"; WB_NAME="${WB_NAME}-ddp${N_GPUS}"
fi

CKPT_ROOT=${CKPT_ROOT:-/data/flore/checkpoints}
OUT="$CKPT_ROOT/$RUN"
# The plan, BEFORE anything is created or checked: the caller (the grid queue) needs these four
# strings precisely so it can decide whether to fit stats and whether to skip.
if (( PRINT_PLAN )); then
    printf '%s\t%s\t%s\t%s\n' "$MODE" "${DS[0]}" "$NS" "$OUT"
    exit 0
fi
[[ -d "$CKPT_ROOT" ]] || mkdir -p "$CKPT_ROOT" 2>/dev/null || {
    echo "FATAL: checkpoint root '$CKPT_ROOT' is missing and not creatable." >&2
    echo "       Pass CKPT_ROOT=<dir> (e.g. CKPT_ROOT=checkpoints)." >&2; exit 1; }

# W&B: your own `wandb login` is used (export WANDB_BASE_URL for a self-hosted server).
# export WANDB_ENTITY=<your-entity>   # optional: no --wandb-entity flag exists, the entity is env-only

for _shard in "${DS[@]}"; do
    [[ -f "$_shard" ]] || { echo "FATAL: dataset missing: $_shard" >&2
        echo "       Augment the dual conversion with UMI_collection's" >&2
        echo "       ./run_mc_to_hl_wholebody.sh (see the SESSION table above)." >&2
        exit 1; }
done
# Diffusion defaults to clip_sample=True, which REQUIRES normalized actions or DDIM's per-step
# x0 clamp crushes every out-of-range joint/force. config.py refuses the combination, but that
# failure arrives after N models are built -- check it here.
if [[ ! -f "$NS" ]]; then
    echo "FATAL: norm stats missing: $NS" >&2
    echo "       Fit them over the SAME mode and the SAME cameras -- HLDataset records the" >&2
    echo "       camera list AND the completion anchor in its split provenance, and the" >&2
    echo "       trainer compares that dict exactly:" >&2
    echo "         .venv/bin/python scripts/fit_hl_norm_stats.py \\" >&2
    echo "             --zarr-paths ${DS[*]} --mode $MODE \\" >&2
    echo "             --image-keys ${CAMS[*]} --val-last-episodes $VAL_EPS \\" >&2
    echo "             --wb-joint-target $WB_TARGET --out $NS" >&2
    echo "       ...with FLORE_ACTION_HORIZON=$H exported FIRST (the fitted quantiles depend on" >&2
    echo "       the horizon, and the pin is resolved at import)." >&2
    exit 1
fi
if [[ -d "$OUT" ]] && [[ "$*" != *--resume-from* ]]; then
    echo "FATAL: $OUT exists and no --resume-from given; it would overwrite step dirs." >&2
    exit 1
fi

# A silent CPU fallback would burn days, and under DDP it would also HANG rather than fail:
# gloo replaces nccl, every rank runs on the same cores, and nothing says why.
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

BATCH=${BATCH:-24}
WORKERS=${WORKERS:-8}
# Spelled per mode rather than derived, so the summary the operator reads names the UNITS -- the
# one thing a width cannot say, and the reason `wholebody` and `wholebody_geom_*` are refused as
# cross-loads even though their 43 action columns are identical.
case "$WB_STATE" in
    joints) STATE_DESC="31       29 JOINT ANGLES at t0 (rad) + L/R gripper opening"
            STATE_NOTE="WB_STATE=joints feeds the completion's raw joints -- the channel measured 0.2750 rad
  (mean) / 0.6643 rad (worst) from the robot's own ready pose, 16 of 29 past the 0.20 rad
  pre-arm gate. WB_STATE=xy_rpy trades it for a pelvis-frame geometry within 0.6 sigma." ;;
    xy_rpy) STATE_DESC="12       [rel_xy 4 (m)][rpy 6 (rad)] + L/R gripper opening"
            STATE_NOTE="rel_xy = (p_L - p_R).xy then p_L.xy, PELVIS frame (p_L IS p_L - p_pelvis: the pelvis is
  the frame origin). rpy = extrinsic-XYZ Euler of R_L then R_R, PELVIS frame. Both FK'd from
  the SAME wb_state_joints through the shared KeypointFK the deploy anchor uses." ;;
    xy)     STATE_DESC="6        [rel_xy 4 (m)] + L/R gripper opening"
            STATE_NOTE="rel_xy = (p_L - p_R).xy then p_L.xy, PELVIS frame. No wrist attitude at all: this cell
  asks whether hand-vs-hand and hand-vs-body POSITION alone is the useful part." ;;
    rpy)    STATE_DESC="8        [rpy 6 (rad)] + L/R gripper opening"
            STATE_NOTE="rpy = extrinsic-XYZ Euler of R_L then R_R, PELVIS frame. No positions: this cell asks
  whether wrist ATTITUDE alone is the useful part." ;;
    none)   STATE_DESC="2        L/R gripper opening only"
            STATE_NOTE="The CONTROL cell: no joint-space proprioception at all, so the 2x2 grid can say whether
  either geometric block helped or the task is vision-plus-force only." ;;
esac
cat <<EOF

  diffusion_umi_wholebody: $N_GPUS process(es) on GPUs $GPUS  ->  $OUT
  SESSION=$SESSION  RECIPE=$RECIPE  MODE=$MODE  H=$H (of $STORED_H stored)  FH=$FH
  WB_TARGET=$WB_TARGET  WB_STATE=$WB_STATE  state jitter: ${NOISE_XY_M} m / ${NOISE_RPY_DEG} deg
  cameras: ${CAMS[*]}   norm stats: $NS

  Widths this run trains (all DERIVED from the layout; only the STATE moves, via WB_STATE):
      action    $H x 43   [joints 29][L grip1 force6][R grip1 force6]
      state     $STATE_DESC
      force in   1 x 12   two 6-DoF wrenches, one encoder group each
      vision     2 x 768  two wrist D405s, one CLS token each

  The 29 joint columns are radians in holosoma dof_names order -- the exact vector the dense low
  level publishes. There is no root channel. WB_TARGET=$WB_TARGET decides what they MEAN:
  'absolute' = the export's own posture space, row 0 == state; 'delta' = q[k] - q[0], joint
  MOTION with row 0 exactly zero. Either way the deploy-side JointClutch publishes
  base + (q[i] - q[0]), so the wire is the same -- and --wb-clutch none is refused for 'delta'.

  state goes through a LEARNED Linear (--diffusion.state-projection), not the zero-pad slot:
  raw radians/metres concatenated beside 768-dim ViT features is a scale mismatch a projection
  absorbs. That is third_party/Psi0's shape too (pi0.py: one nn.Linear over the state).
  Pass --diffusion.no-state-projection for the zero-pad ablation.

  $STATE_NOTE
  (No backticks in this heredoc: <<EOF performs command substitution.)

EOF
if (( N_GPUS > 1 )); then
    cat <<'EOF'
  DDP: --batch-size is PER GPU. LR is NOT auto-scaled, and the umi recipe's 3e-4 was tuned at
  batch 32 on the SINGLE-camera model -- an effective batch of N*BATCH here is a different
  optimization problem. Pass --lr explicitly if you want it changed.
  The trainer permits DDP for act and diffusion only (trainer._DDP_ALGOS).

EOF
fi

mkdir -p logs
set -o pipefail
TRAIN_ARGS=(
    scripts/train_s1.py
    --algo diffusion --pipeline hl
    --hl-data.zarr-paths "${DS[@]}"
    --hl-data.mode "$MODE"
    --hl-data.image-keys "${CAMS[@]}"
    --hl-data.val-last-episodes "$VAL_EPS"
    --hl-data.norm-stats-path "$NS"
    --hl-data.wb-joint-target "$WB_TARGET"
    --hl-data.state-noise-xy-m "$NOISE_XY_M"
    --hl-data.state-noise-rpy-deg "$NOISE_RPY_DEG"
    --diffusion.vit vit_b
    --diffusion.diffusion-net unet
    --diffusion.prediction-type epsilon
    --diffusion.num-train-timesteps 50
    --diffusion.num-inference-steps 16
    --diffusion.input-pertub 0.1
    --diffusion.num-draws 1
    --diffusion.state-projection
    "${RECIPE_ARGS[@]}"
    --batch-size "$BATCH" --num-workers "$WORKERS"
    --grad-clip 1.0
    --max-steps 100000 --warmup-steps 1000
    --save-every 10000 --log-every 50
    --eval-seeds 0 1 2 3
    --sample-every-epochs 5
    --sample-diagnostic-batches 1 --sample-diagnostic-samples 4
    --artifact-every-epochs 50
    --epoch-artifact-every-epochs 20
    --wandb --wandb-project MUSE --wandb-run-name "$WB_NAME"
    --output-dir "$OUT"
    "$@"
)
if (( N_GPUS > 1 )); then
    CUDA_VISIBLE_DEVICES="$GPUS" .venv/bin/torchrun \
        --standalone --nnodes 1 --nproc-per-node "$N_GPUS" \
        "${TRAIN_ARGS[@]}" 2>&1 | tee -a "logs/$RUN.log"
else
    CUDA_VISIBLE_DEVICES="$GPUS" .venv/bin/python \
        "${TRAIN_ARGS[@]}" 2>&1 | tee -a "logs/$RUN.log"
fi
