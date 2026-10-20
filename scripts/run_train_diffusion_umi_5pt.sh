#!/usr/bin/env bash
# diffusion_umi_5pt -- FIVE keypoint poses out, a pelvis-frame GEOMETRY in.
#
# The third member of the diffusion_umi family. Same model, same UMI recipe, same rig, same
# 30 Hz grid, the SAME zarr as the other two. What changes is both ends at once:
#
#                     diffusion_umi_3tracker   diffusion_umi_wholebody   diffusion_umi_5pt
#   layout mode       both_hands_head_dual     wholebody[_geom_*]        fivepoint_geom_*
#   --hands           both_head_dual           both_head_dual            both_head_dual
#   action out        H x 41                   H x 43                    H x 59
#                     L R head keypoints       29 joint angles           L R head + BOTH FEET
#   action frame      anchor-relative          ABSOLUTE joint space      anchor-relative
#   state in          20  rot6d + grips        31  29 joints + grips     14  geometry + grips
#   low level         5pt (needs completion)   dense                     5pt (DIRECT)
#
# WHY. `wholebody` commands the 29 joints, so its output has to land in a posture space the
# robot is actually in -- and measured over the 159 valid episode starts of 0902_merged3 it does
# not: 0.2750 rad mean from the pose the G1 holds after the operator presses A, worst joint
# 0.6643, and 16 OF 29 JOINTS PAST the low level's 0.20 rad pre-arm gate. Two mechanisms absorb
# that (JointClutch on the wire, --state-source command in the observation) and neither closes
# it. A 5-point wire is gated on 3 OF 5 KEYPOINTS AT 70 mm instead -- the two feet are the
# completion's inference, which holosoma warns about rather than refusing on -- and a chunk
# decoded against the FK'd ready pose lands 0.7-3.1 mm (hands) / 28.9-36.3 mm (chest) from that
# reference. Same task, same rig, same data: the gate closes by construction on one family and
# structurally cannot on the other. The redundant arm's null space (0.40 rad = 23 deg) and the
# knee's 0.54 rad stand-up simply stop being COMMANDED, because the low level solves for them.
#
# WHAT THE STATE IS, and why it is not the 29 joints. WB_STATE picks a pelvis-frame GEOMETRIC
# reduction instead of raw proprioception, for the same reason the wholebody_geom_* grid exists:
#
#   rel_xy3 (6)  the THREE pairwise xy differences among the three TRACKED points --
#                (L-R).xy, (L-chest).xy, (R-chest).xy. Every column is a DIFFERENCE, so the
#                block is TRANSLATION-INVARIANT: it does not depend on where the pelvis is,
#                only on how the frame is ORIENTED. That is deliberately stricter than
#                wholebody_geom's rel_xy, which carries the left hand's absolute pelvis-frame
#                xy (`lp_*`) -- a channel the two export placements move by 61-140 mm.
#   rpy (6)      extrinsic-XYZ Euler of R_L then R_R, PELVIS frame. Both hands' TCP attitude.
#
# Both are FK'd from the SAME `wb_state_joints` through the shared KeypointFK the deploy anchor,
# hl_dense's command anchor and wb_geom all use -- ONE FK, so a lever fix propagates with no
# re-export. Measured on 0902_merged3: worst |RPY| 41.5 deg over 7153 frames, 0.00% past
# 160 deg, so the Euler wrap that rot6d exists to avoid is not reached (geom_series warns per
# dataset if a future task gets there).
#
# WHERE THE FEET COME FROM. Nothing on this rig sees a foot: three pucks, no foot hardware. The
# two foot columns are `KeypointFK(wb_action_joints[t, k])`'s ankle poses -- a pure function of
# the joint target `diffusion_umi_wholebody` already trains on -- written by
# scripts/add_foot_targets.py. So a 5pt dataset is a wholebody dataset plus eight arrays, ONE
# session feeds all three presets, and no second completion ever runs. Three properties of that
# channel are measured and named rather than hoped about:
#
#   * PELVIS frame with NO root pose, so it is a STANCE ("where the foot sits relative to the
#     body"), not a footstep. A take where the operator walks produces no walking foot target.
#     Safe for the ACTION because absolute_to_delta_ee is covariant under a rigid transform
#     applied to both a pose and its anchor, and ref_frame=per_keypoint gives each keypoint its
#     own -- so a pelvis-frame foot composes correctly beside a recorded-world-frame hand.
#   * The trade is small: the chest's net XY excursion is 37.9 mm median / 111.9 mm p90 over the
#     167 episodes, and 86% of episodes never leave a 100 mm ball.
#   * The per-chunk foot SIGNAL is small too -- completed feet move 1.9-4.4 mm per 30-frame
#     window while the two export placements disagree by 2.3-3.2 mm on that same quantity. The
#     channel is worth predicting for the operator's YAW (10.7 deg median, 48.5 p90, which does
#     re-place the feet), not for translation. READ THE PER-GROUP val/* before concluding the
#     feet learned anything.
#
# IT DOES NOT TOUCH THE OTHER TWO PRESETS. `fivepoint_geom_*` are NEW layout modes over the SAME
# --hands value, the foot arrays are additive, and `attrs.wb_foot_targets` is the discriminator
# -- so run_train_diffusion_umi_{3tracker,wholebody}.sh train exactly the bytes they did before.
# Asserted by tests/test_diffusion_umi_5pt.py.
#
# PREREQUISITE -- two post-processes over one conversion, in this order:
#   1  UMI_collection: ./run_mc_to_hl_wholebody.sh <dual>.zarr.zip -o <dual>_wbabs.zarr.zip \
#          --cond-mode absolute --dz-anchor hand
#   2  FLoRe:  uv run scripts/add_foot_targets.py <dual>_wbabs.zarr.zip -o <...>_5pt.zarr.zip
#             uv run scripts/add_foot_targets.py <...>_5pt.zarr.zip --check
# Step 2 needs `yourdfpy` plus the two submodules the FK reads (see the runbook's prerequisites);
# it is CPU-only, no GPU and no conda env.
#
# SERVING IS NOT WIRED. No deploy path builds this observation yet, so every fivepoint_* mode is
# refused by name at the serving launchers, exactly as the wholebody_geom_* family is. What the
# deploy side WOULD need is small and is written down in docs/RUNBOOK_diffusion_umi_5pt.md
# ("What serving still needs"), which is also why wb_geom.geom_from_bodies takes FK OUTPUT
# rather than a joint vector: the 5pt wire's own keypoints are already those bodies.
#
# Usage:
#   ./scripts/run_train_diffusion_umi_5pt.sh [n_gpus] [low|high|all] [extra train_s1.py args]
#   H=200 WB_STATE=xy3_rpy ./scripts/run_train_diffusion_umi_5pt.sh 4 low
#
#   INIT_FROM=/data/flore/checkpoints/<base>/step_00100000 SESSION=rotate_box_merged6_h200 \
#       ./scripts/run_train_diffusion_umi_5pt.sh 4 --max-steps 2000 --warmup-steps 100 \
#           --lr 3e-5 --save-every 500          # FINETUNE onto a supplementary session
#   ⚠️ No `--` before the extra args: tyro treats everything after one as POSITIONAL and dies
#   with "Unrecognized options: --max-steps, 2000, ..." naming flags its own help lists.
#
# Env knobs:  SESSION  H  WB_STATE  NOISE_XY_M  NOISE_RPY_DEG  RECIPE  BATCH  WORKERS
#             DSDIR  CKPT_ROOT  INIT_FROM
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --print-plan: resolve THIS run's mode / dataset / norm-stats path / output dir, print them as
# one TAB-separated line, and stop before any GPU, model or file check. Same contract as the
# wholebody launcher's, and for the same reason: run_train_5pt_grid.sh ASKS for those four
# strings instead of re-deriving the naming rules.
PRINT_PLAN=0
if [[ "${1:-}" == "--print-plan" ]]; then PRINT_PLAN=1; shift; fi

N_GPUS=${1:-1}
shift || true
GPU_GROUP=${1:-}
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

VAL_EPS=4
# The chunk length THIS RUN trains vs the length the dataset STORES. Separate because a shorter
# horizon is a strict PREFIX of a longer one at the same dt: one 200-step conversion serves
# H=50/100/200, and STORED_H is what refuses the other direction here rather than inside
# HLDataset._check_file. ⚠ Re-INDEXING a stored 50-step window into a 200-step one is NOT the
# same thing and is measured wrong (0.52 rad on 23% of joint steps, 100% of wrench steps) --
# that is why the H=200 dataset is a separate re-convert, not a reshape.
H=${H:-50}
if ! [[ "$H" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: H must be a positive integer number of 30 Hz steps, got '$H'" >&2; exit 1
fi
STORED_H=50
FH=1
DSDIR=${DSDIR:-third_party/UMI_collection/data/datasets}
SESSION=${SESSION:-0902_merged3_h200}

# WB_STATE -> layout mode. A named MODE rather than a flag, because state_dim moves with the
# composition and `layout_mode` is what validate_model_layout pins a checkpoint against.
WB_STATE=${WB_STATE:-xy3_rpy}
case "$WB_STATE" in
    xy3_rpy) MODE=fivepoint_geom_xy3_rpy ;;
    xy3)     MODE=fivepoint_geom_xy3 ;;
    rpy)     MODE=fivepoint_geom_rpy ;;
    none)    MODE=fivepoint_geom_none ;;
    *) echo "FATAL: WB_STATE must be xy3_rpy | xy3 | rpy | none, got '$WB_STATE'" >&2
       echo "       (there is no 'joints' cell here -- dropping the 29 raw joint angles from" >&2
       echo "        the observation is the whole point of this preset; train" >&2
       echo "        run_train_diffusion_umi_wholebody.sh for that.)" >&2
       exit 1 ;;
esac
# Jitter on the geometric state columns, TRAIN samples only (StateNoise wraps the train subset,
# never the val subset and never the norm-stats fit). Applied PRE-collate and therefore
# PRE-normalization, so these are RAW units -- metres and degrees, physical claims about how well
# the robot knows where its hands are.
#
# ⚠ THE HABITUAL MAGNITUDES ARE WRONG FOR THIS STATE BY 1-4x. Over 7153 frames of 0902_merged3
# the lr_dx/lr_dy std is 2.6/2.2 cm and the left roll/yaw std is 2.9/4.4 deg, so a "2.5 cm /
# 10-15 deg" reflex erases the channel while the loss curve stays perfectly healthy. The
# defaults below are sized to COVER the measured 11-15 mm / 1.3-2.2 deg deploy offsets and no
# more. A sigma the mode's state cannot carry is REFUSED by the trainer rather than ignored --
# a no-op augmentation is invisible in the loss, the widths and W&B.
NOISE_XY_M=${NOISE_XY_M:-0.010}
NOISE_RPY_DEG=${NOISE_RPY_DEG:-2.0}
case "$WB_STATE" in
    none) NOISE_XY_M=0; NOISE_RPY_DEG=0 ;;
    xy3)  NOISE_RPY_DEG=0 ;;
    rpy)  NOISE_XY_M=0 ;;
esac

# Dataset table. Every entry is a `_5pt.zarr.zip`: the wb_* augmentation AND the eight foot
# arrays. The un-footed sibling is refused by name (HLDataset._check_foot_targets), and so is a
# file whose foot frame attr is not `pelvis_stance_no_root`.
case "$SESSION" in
    # The H=200 re-export of 0902_merged3 with foot targets added. SAME 167 episodes, same
    # 18-take --exclude list, same `--cond-mode absolute --dz-anchor hand` completion. Serves
    # H=50, 100 and 200 by the prefix relation -- it is the dataset the 3-cell horizon ablation
    # in docs/RUNBOOK_diffusion_umi_5pt.md runs on.
    0902_merged3_h200)
        DS=("$DSDIR/session_2026.09.02_merged3_v3_dual_clean167_h200_wbabs_5pt.zarr.zip")
        STORED_H=200 ;;
    # The 50-step twin of the same session, for a cheap H=50-only sanity run: 2.6 GB rather
    # than 3.4, and its window arrays are 4x narrower on disk.
    0902_merged3)
        DS=("$DSDIR/session_2026.09.02_merged3_v3_dual_clean167_wbabs_5pt.zarr.zip") ;;
    # push_cart, 2026-09-05: a NEW task and the first 5pt dataset completed in the `home`
    # placement. 11 sittings merged (312 takes) minus an 80-take --exclude list, so 232
    # episodes -- the largest 5pt file. `--cond-mode home` re-bases every episode onto the
    # G1's 29-joints-zero pose instead of keeping the operator's world poses and guessing an
    # origin height, which is why it needs no --dz-anchor at all.
    #
    # It is NOT mixable with the 0902 entries above and the trainer says so by name:
    # `wb_cond_mode` is an exact-equality dataset knob, so a norm-stats file fitted on one
    # placement is a refusal against the other rather than a silent rescale.
    push_cart_h200)
        DS=("$DSDIR/session_2026.09.05_merged_v3_dual_clean232_h200_wbhome_5pt.zarr.zip")
        STORED_H=200 ;;
    # rotate_box, 2026-09-06: the 0902_merged3 task EXTENDED. Five sittings -- 0902's own three
    # plus two new 2026-09-06 ones -- re-merged from the RAW sittings (merge_sessions.py refuses
    # a merged dir as input, so this is not 0902_merged3 + 2), 250 takes minus a 67-take
    # --exclude list => 183 episodes / 39683 frames. Take offsets 0/100/200 are byte-identical
    # to 0902_merged3's, which is why its 18-take exclusion list is reused VERBATIM here and the
    # first 167 episodes are the same recordings.
    #
    # Both placements are shipped, as the SAME 183 episodes completed twice, so the placement
    # A/B is one flag rather than a re-export:
    #   rotate_box_h200       --cond-mode home                      (no dz, no guessed origin)
    #   rotate_box_h200_abs   --cond-mode absolute --dz-anchor hand  (dz +0.78..+0.89 m/episode)
    #
    # ⚠️ They can NEVER be trained together and neither can share norm stats with the other:
    # `wb_cond_mode` is an exact-equality dataset knob, so a crossed stats file is a refusal by
    # name rather than a silent rescale. $SESSION is in the stats filename for exactly that
    # reason, so the two fit into separate files automatically.
    rotate_box_h200)
        DS=("$DSDIR/session_2026.09.06_merged5_v3_dual_clean183_h200_wbhome_5pt.zarr.zip")
        STORED_H=200 ;;
    rotate_box_h200_abs)
        DS=("$DSDIR/session_2026.09.06_merged5_v3_dual_clean183_h200_wbabs_5pt.zarr.zip")
        STORED_H=200 ;;
    # The two 2026-09-06 SUPPLEMENTS, recorded after the first rig test of each task. They exist
    # to be FINETUNED onto that task's own step_00100000 (INIT_FROM=...), not trained from
    # scratch: 81 and 44 episodes is a fraction of the 183/232 the base runs saw.
    #
    # Same task, same rig, same `--cond-mode home` completion and the same `--hl-action-horizon
    # 200`, so they serve H=50/100/200 by the prefix relation exactly as the base files do and
    # `validate_init_norm_stats` accepts the pair. They are SEPARATE SESSIONs rather than extra
    # shards of the base ones because a finetune wants the new data ALONE -- add them to the
    # base DS array instead if what you want is a from-scratch run over the union, and re-fit.
    #
    # ⚠️ rotate_box_sup's first 15 usable takes come from sitting 01.14.08, which is ALREADY one
    # of rotate_box_h200's five merged sittings -- so ~19% of this file is data the base weights
    # have seen. The other 66 takes (sitting 14.26.12) are new. Harmless for a finetune (and it
    # anchors it slightly), but it is not 81 takes of new information.
    # ⚠️ The PLACEMENT is per TASK. `rotate_box` supports both (`home` and
    # `absolute --dz-anchor hand`) and so ships both, mirroring its base file; `push_cart`
    # supports `home` ONLY. Every checkpoint published so far is `home`, so `_abs` has no weights
    # to be finetuned FROM -- it is here for the placement A/B.
    rotate_box_sup_h200)
        DS=("$DSDIR/session_2026.09.06_rotate_box_sup_v3_dual_clean81_h200_wbhome_5pt.zarr.zip")
        STORED_H=200 ;;
    rotate_box_sup_h200_abs)
        DS=("$DSDIR/session_2026.09.06_rotate_box_sup_v3_dual_clean81_h200_wbabs_5pt.zarr.zip")
        STORED_H=200 ;;
    push_cart_sup_h200)
        DS=("$DSDIR/session_2026.09.06_push_cart_sup_v3_dual_clean44_h200_wbhome_5pt.zarr.zip")
        STORED_H=200 ;;
    # THE SUPPLEMENT RE-MERGED INTO ITS BASE AT THE RECORDING LEVEL -- and for a FINETUNE this
    # is the entry to reach for. ONE file, re-merged from all the RAW sittings and re-converted,
    # so it SUPERSEDES the base file rather than sitting beside it:
    #   rotate_box   6 sittings, 326 takes - 77 =  249 episodes   (183 + 66)
    #   push_cart   13 sittings, 370 takes - 94 =  276 episodes   (232 + 44)
    # Offsets 0/100/... reproduce the previous merges byte-identically, which is why the audit
    # re-derives their exclusion lists verbatim (67 + 10 and 76 + 14, plus push_cart's four
    # stale-head takes the converter refuses itself) and why the first 183 / 232 episodes ARE
    # the recordings the base checkpoints were trained on.
    #
    # ⚠️ A RE-MERGE, not two shards, because `01.14.08` is BOTH one of rotate_box's original five
    # sittings and one of its supplement's two: listing the two files together would train 15
    # episodes twice. Re-merging from the raw sittings de-duplicates by construction -- 249, not
    # 183+81=264.
    #
    # WHY THE WHOLE FILE and not just the new episodes: the NORMALIZATION, not the gradient. A
    # supplement fitted ALONE has ranges 20-72% NARROWER than the weights were trained with
    # (push_cart's actions median -72.2%, its force window -65.9%, one action column -99.5%),
    # because 44 or 81 episodes of one sitting cover a tighter slice than 232 or 183 do. Fitting
    # on that slice maps it to the full [-1, 1], and `--diffusion.clip-sample` then CLAMPS
    # everything outside it on every DDIM step -- a 2000-step finetune would hand back a policy
    # whose reachable action range is a fraction of the task's, with a healthy loss curve. Over
    # the merged file the quantiles stay the base run's to 1.3-2.8% (measured), so the loaded
    # weights start where they already are and nothing is forgotten.
    #
    # ⚠️ WHERE the new episodes land differs by task and it decides what val/* MEANS. The
    # converter discovers takes with sorted() on the NAME, and merged take numbers are not padded
    # to one width -- once a sitting exceeds 100 takes the later offsets go to four digits and
    # `take_1200` string-sorts BEFORE `take_200`. Measured:
    #   rotate_box_merged6   new sitting at kept idx 167..248 (the TAIL)   -> val = the NEW eps
    #   push_cart_merged13   new sittings at kept idx 193..238 (MIDDLE)    -> val = take_902..905,
    #                        the SAME 4 episodes the base run scored
    # Neither is leakage (each base run held its own val out too), but only push_cart's val/* is
    # comparable to its base run's final number. Re-merging in a different order would fix the
    # ordering and re-point every exclusion (they are keyed on take numbers).
    rotate_box_merged6_h200)
        DS=("$DSDIR/session_2026.09.06_rotate_box_merged6_v3_dual_clean249_h200_wbhome_5pt.zarr.zip")
        STORED_H=200 ;;
    rotate_box_merged6_h200_abs)
        DS=("$DSDIR/session_2026.09.06_rotate_box_merged6_v3_dual_clean249_h200_wbabs_5pt.zarr.zip")
        STORED_H=200 ;;
    push_cart_merged13_h200)
        DS=("$DSDIR/session_2026.09.06_push_cart_merged13_v3_dual_clean276_h200_wbhome_5pt.zarr.zip")
        STORED_H=200 ;;
    # push_gantry, 2026-09-11: the first FIVE-PUCK recording (MODE=5pt, both feet tracked).
    # Two sittings merged (285 takes: 16:38 = 184 on the FAR-pi VIVE stack, 20:00 = 101 on the
    # 09-10 stack with the foot pucks swapped and corrected in the raw data), converted with
    # `--feet recorded` so the RECORDED feet ride along as {left,right}_foot_* / hl_*_foot_*
    # keys, and with the per-role glitch masks instead of an --exclude list: 276 episodes kept,
    # 9 takes dropped by the converter (primary-hand teleports of 33-246 cm), 0.3-2.8% of
    # frames flagged per role. F/T: the 16:38 sitting is ZEROED (its right Bota had no
    # compensated stream; the operator zeroed it), the 20:00 sitting carries its real tared
    # compensated wrenches (both hands, 20-40 N while pushing) — 184 zero-force episodes +
    # 92 live ones in one file.
    # ⚠️ The FOOT TARGETS this preset trains on are still the completion's FK
    # (add_foot_targets.py, pelvis_stance_no_root) — the recorded feet are in the file for
    # a recorded-feet layout that does not exist yet.
    push_gantry_merged2_h200)
        DS=("$DSDIR/session_2026.09.11_push_gantry_merged2_v3_dual_feet_clean276_h200_wbhome_5pt.zarr.zip")
        STORED_H=200 ;;
    # push_gantry + the 2026-09-12 THIRD sitting (113 takes), re-merged at the RECORDING level
    # into one zarr exactly as merged2 was: 398 takes -> 389 episodes, the SAME 9 converter-
    # dropped takes (all 20:00 primary-hand teleports) and still no --exclude list. This is the
    # entry to train; merged2 stays in the table because the 2026-09-12 H=50/H=100 checkpoints
    # name its zarr in `_provenance.split` and `validate_split_provenance` compares it exactly.
    #
    # Three things the third sitting adds, each measured rather than assumed:
    #   * A THIRD FORCE REGIME. 16:38 is zeroed, 20:00 pushes at 20-40 N, and 09-12 is tared but
    #     LIGHT: per-take peak |F| median 5.6 N (left) / 8.7 N (right) against 32 / 40 N, over a
    #     0.106 m median net hand travel against 0.824 m. Read the force `val/*` knowing the file
    #     now holds 184 zero-force + 92 heavy + 113 light episodes.
    #   * A WORLD FRAME ROTATED ~90 deg IN YAW. libsurvive re-solves the room every launch and
    #     anchors it on one station's pose (today's log: "Using LH 2 (ad3c6894) as reference"), so
    #     the same physical push reads -124 deg in world XY on 09-11 and -31 deg on 09-12 while
    #     the hand camera shows the identical scene and facing. It is ABSORBED here and only
    #     here: `--cond-mode home` re-bases every episode onto the robot's own zero pose, so the
    #     whole cond set is a rigid transform of the scene, the backend canonicalises each
    #     episode by its own yaw (`episode canon: yaw=...`), and the action is anchor-relative
    #     per keypoint. What would NOT be absorbed is a world ROLL/PITCH; the inter-foot vector's
    #     elevation reads -3.8 / +0.6 / -1.3 deg across the three sittings (within-sitting std
    #     3-4.5), i.e. one shared vertical. ⚠️ Anything that consumes an ABSOLUTE world direction
    #     (WORLD_RPY, the teleop bridge's yaw autocal) is NOT protected by any of that.
    #   * MORE ROTATION GLITCHES, all masked. 83 of 113 takes carry at least one there-and-back
    #     orientation excursion (peak 105 deg) and 3 carry a 103-150 mm unbridged position jump,
    #     all below the drop gate; the left hand's flagged-frame share goes 2.0% -> 3.0% for the
    #     whole file. Root cause is visible in the cached calib: the third station (ch 2,
    #     id 3872623286) solved with rotational variance 1.26 / 0.90 against ~0.005-0.03 for the
    #     other two. Worth re-solving before the next sitting rather than excluding takes.
    push_gantry_merged3_h200)
        DS=("$DSDIR/session_2026.09.12_push_gantry_merged3_v3_dual_feet_clean389_h200_wbhome_5pt.zarr.zip")
        STORED_H=200 ;;
    # push_gantry_new, 2026-09-13: a NEW TASK KEY on purpose, not a fourth push_gantry sitting.
    # ONE sitting (`session_2026.09.13_01.14.08`, 208 takes, 2 lighthouses, MODE=5pt), the same
    # chain as push_gantry (`--hands both_head_dual --feet recorded --hl-action-horizon 200`,
    # `--cond-mode home`, add_foot_targets.py) plus one new converter knob, `--feet-glitch-scale 2`:
    # the two FOOT streams are scanned with their glitch out-jump gates doubled (6 m/s / 800 deg/s,
    # label gate 2 m/s), because every gate was measured on HANDS and a real foot swing here
    # reaches 1.8 m/s on the 30 Hz grid -- the hands and chest keep the strict gates, and a foot
    # never drops a take either way. ⚠️ 66 of the 208 takes have NO right-camera frames (the
    # D405 went away at take_144 = 01:37:37 and never came back; take_041 and take_128 are
    # isolated dropouts) and a 2-camera preset cannot use them, so they are `--exclude`d and the
    # file is 142 episodes. NOT mergeable with merged3 by `DS=(a b)`: a different task key, and
    # its right-foot puck teleports (47 unbridged excursions over ~34 takes) are masked, not used.
    push_gantry_new_h200)
        DS=("$DSDIR/session_2026.09.13_push_gantry_new_v3_dual_feet_clean142_h200_wbhome_5pt.zarr.zip")
        STORED_H=200 ;;
    *)
       echo "FATAL: unknown SESSION '$SESSION'." >&2
       echo "       Known: rotate_box_h200, rotate_box_h200_abs (H<=200)," >&2
       echo "              push_cart_h200 (H<=200), 0902_merged3_h200 (H<=200)," >&2
       echo "              rotate_box_merged6_h200[_abs], push_cart_merged13_h200 (H<=200," >&2
       echo "              push_gantry_merged3_h200 (H<=200, the 5-puck recording, 3 sittings" >&2
       echo "                -- the one to train), push_gantry_merged2_h200 (its 2-sitting" >&2
       echo "                predecessor, kept for the 09-12 checkpoints' split provenance)," >&2
       echo "              push_gantry_new_h200 (H<=200, the 2026-09-13 sitting as its OWN task," >&2
       echo "                142 ep, --feet-glitch-scale 2)," >&2
       echo "                the FINETUNE files: base + supplement, re-merged into one)," >&2
       echo "              rotate_box_sup_h200, push_cart_sup_h200 (H<=200, the supplement ALONE" >&2
       echo "                -- see the table's note on why the union is usually right)," >&2
       echo "              0902_merged3 (H<=50)." >&2
       echo "       To add one, run BOTH post-processes over a --hands both_head_dual zarr:" >&2
       echo "         (UMI) ./run_mc_to_hl_wholebody.sh <dual>.zarr.zip -o <dual>_wbabs.zarr.zip \\" >&2
       echo "                   --cond-mode absolute --dz-anchor hand" >&2
       echo "         (FLoRe) uv run scripts/add_foot_targets.py <dual>_wbabs.zarr.zip \\" >&2
       echo "                   -o <dual>_wbabs_5pt.zarr.zip" >&2
       exit 1 ;;
esac
if (( H > STORED_H )); then
    echo "FATAL: H=$H but SESSION=$SESSION stores only $STORED_H steps per chunk." >&2
    echo "       A LONGER horizon is a re-CONVERT, never a re-index: the targets are resampled" >&2
    echo "       from the raw ~77 Hz tracker / ~215 Hz wrench onto the camera grid, so" >&2
    echo "       re-indexing a stored 50-step window along the frame axis is wrong by up to" >&2
    echo "       0.52 rad on 23% of joint steps and 100% of wrench steps (measured)." >&2
    echo "       Use SESSION=0902_merged3_h200, or re-convert at --hl-action-horizon $H." >&2
    exit 1
fi

# FH / H are EXPORTED, not taken from the caller: torchrun passes the environment to every rank
# and both are resolved at IMPORT by flore.s1_policy.data.layout, so the ranks must agree before
# the first collective.
export FLORE_FORCE_HIST_LEN="$FH"
export FLORE_ACTION_HORIZON="$H"

CAMS=(camera0_rgb camera1_rgb)

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

# `_5pt` in the norm-stats name because these are interchangeable with NOTHING: 59 action columns
# instead of 41 or 43, and a state that is metres+radians of pelvis-frame geometry rather than
# rot6d or joint radians. The trainer asserts every width at load (`_check_norm_stats`) and
# compares `dataset_knobs` for exact equality (which carries `mode`, `wb_cond_mode` and, whenever
# it is not 50, `action_horizon`), so a crossed file is a refusal -- the distinct name is what
# stops someone reaching for it in the first place.
#
# WB_STATE and H add SUFFIXES: both genuinely change the statistics (WB_STATE the state WIDTH and
# UNITS; H the action QUANTILES at an identical width, because a 200-step chunk reaches further
# from its anchor than a 50-step one). `xy3_rpy`/`50` keep the bare name.
NS_SUFFIX=""
[[ "$WB_STATE" != "xy3_rpy" ]] && NS_SUFFIX="${NS_SUFFIX}_${WB_STATE}"
(( H != 50 )) && NS_SUFFIX="${NS_SUFFIX}_h${H}"
NS=checkpoints/norm_stats_hl_5pt_${SESSION}${NS_SUFFIX}.json
RUN="diffusion_umi_5pt_${SESSION}_${RECIPE}${NS_SUFFIX}"
WB_NAME="diffusion-umi-5pt-${SESSION}-${RECIPE}${NS_SUFFIX//_/-}"
# FINETUNE: start from another checkpoint's WEIGHTS (fresh optimizer, step 0, this SESSION's own
# data and its own norm stats). `_ft` is in the run name because the output dir must not collide
# with a from-scratch run of the same SESSION+H, and because a W&B run that silently began at
# somebody else's weights is unreadable a week later.
if [[ -n "${INIT_FROM:-}" ]]; then
    [[ -f "$INIT_FROM/config.json" && -f "$INIT_FROM/model.safetensors" ]] || {
        echo "FATAL: INIT_FROM='$INIT_FROM' is not a deploy checkpoint dir." >&2
        echo "       Wanted <dir>/{model.safetensors,config.json,norm_stats.json} -- a" >&2
        echo "       step_NNNNNNNN/ or best_overall/ directory is one. Pull it with:" >&2
        echo "         ./scripts/fetch_checkpoint.sh <key>" >&2
        exit 1; }
    RUN="${RUN}_ft"; WB_NAME="${WB_NAME}-ft"
    EXTRA_ARGS=(--init-from "$INIT_FROM")
else
    EXTRA_ARGS=()
fi
if (( N_GPUS > 1 )); then
    RUN="${RUN}_ddp${N_GPUS}"; WB_NAME="${WB_NAME}-ddp${N_GPUS}"
fi

CKPT_ROOT=${CKPT_ROOT:-/data/flore/checkpoints}
OUT="$CKPT_ROOT/$RUN"
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
        echo "       Run BOTH post-processes over the dual conversion (see the header's" >&2
        echo "       PREREQUISITE): run_mc_to_hl_wholebody.sh, then add_foot_targets.py." >&2
        exit 1; }
done
if [[ ! -f "$NS" ]]; then
    echo "FATAL: norm stats missing: $NS" >&2
    echo "       Fit them over the SAME mode, the SAME cameras and the SAME horizon --" >&2
    echo "       HLDataset records all three in its split provenance and the trainer compares" >&2
    echo "       that dict exactly:" >&2
    echo "         FLORE_ACTION_HORIZON=$H .venv/bin/python scripts/fit_hl_norm_stats.py \\" >&2
    echo "             --zarr-paths ${DS[*]} --mode $MODE \\" >&2
    echo "             --image-keys ${CAMS[*]} --val-last-episodes $VAL_EPS --out $NS" >&2
    echo "       The pin must be exported BEFORE the fit: it is resolved at import." >&2
    exit 1
fi
if [[ -d "$OUT" ]] && [[ "$*" != *--resume-from* ]]; then
    echo "FATAL: $OUT exists and no --resume-from given; it would overwrite step dirs." >&2
    exit 1
fi

# A silent CPU fallback would burn days, and under DDP it would also HANG rather than fail.
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
case "$WB_STATE" in
    xy3_rpy) STATE_DESC="14       [rel_xy3 6 (m)][rpy 6 (rad)] + L/R gripper opening"
             STATE_NOTE="rel_xy3 = (L-R).xy, (L-chest).xy, (R-chest).xy -- three DIFFERENCES, so the block is
  translation-invariant and carries no absolute placement at all. rpy = extrinsic-XYZ Euler of
  R_L then R_R. Both PELVIS frame, both FK'd from wb_state_joints through the shared
  KeypointFK the deploy anchor uses." ;;
    xy3)     STATE_DESC="8        [rel_xy3 6 (m)] + L/R gripper opening"
             STATE_NOTE="Positions only: does hand-vs-hand and hand-vs-chest GEOMETRY alone carry the run?" ;;
    rpy)     STATE_DESC="8        [rpy 6 (rad)] + L/R gripper opening"
             STATE_NOTE="Attitude only: does wrist ORIENTATION alone carry the run?" ;;
    none)    STATE_DESC="2        L/R gripper opening only"
             STATE_NOTE="The CONTROL cell: no geometric proprioception at all, so the grid can say whether
  either block helped or the task is vision-plus-force only." ;;
esac
cat <<EOF

  diffusion_umi_5pt: $N_GPUS process(es) on GPUs $GPUS  ->  $OUT
  SESSION=$SESSION  RECIPE=$RECIPE  MODE=$MODE  H=$H (of $STORED_H stored)  FH=$FH
  WB_STATE=$WB_STATE  state jitter: ${NOISE_XY_M} m / ${NOISE_RPY_DEG} deg  (TRAIN subset only)
  cameras: ${CAMS[*]}   norm stats: $NS

  Widths this run trains (all DERIVED from the layout; only the STATE moves, via WB_STATE):
      action    $H x 59   L[pos3 rot6d6 grip1 force6] R[same] chest[pos3 rot6d6]
                          + Lfoot[pos3 rot6d6] Rfoot[pos3 rot6d6]
      state     $STATE_DESC
      force in   1 x 12   two 6-DoF wrenches, one encoder group each
      vision     2 x 768  two wrist D405s, one CLS token each

  Slots [0:41] of the action are BYTE-IDENTICAL in meaning to diffusion_umi_3tracker's, which is
  the same prefix promise holosoma's VR_5POINT_BODIES makes on the wire. The two are still NOT
  cross-loadable -- layout_mode is the pin -- and it is the STATE that makes them incomparable,
  not the action: nothing about 14 dims of pelvis-frame geometry resembles 20 dims of
  episode-start-relative rot6d.

  The two FOOT blocks are the completion's inference, FK'd from this file's own
  wb_action_joints (scripts/add_foot_targets.py). PELVIS frame, no root pose -- a STANCE, not a
  footstep. Their per-chunk signal is 1.9-4.4 mm against 2.3-3.2 mm of placement noise, so read
  the per-group val/* before concluding they learned anything.

  state goes through a LEARNED Linear (--diffusion.state-projection), not the zero-pad slot: raw
  metres/radians concatenated beside 768-dim ViT features is a scale mismatch a projection
  absorbs. Pass --diffusion.no-state-projection for the zero-pad ablation.

  $STATE_NOTE
  (No backticks in this heredoc: <<EOF performs command substitution.)

EOF
if [[ -n "${INIT_FROM:-}" ]]; then
cat <<EOF
  FINETUNE: weights start at $INIT_FROM
    Fresh optimizer, fresh scheduler (so --warmup-steps warms up AGAIN), fresh EMA seeded from
    those weights, global_step back to 0. Nothing else is carried -- this is not --resume-from.
    The trainer refuses a checkpoint whose architecture or column MEANING differs, and prints
    the normalization shift between its stats and $NS. A few percent is the finetune absorbing a
    different distribution; a large number means these are not the same task.
    Defaults you almost certainly want to override, because they are sized for 100k steps:
      --max-steps 2000 --warmup-steps 100 --lr 3e-5 --save-every 500

EOF
fi
if (( N_GPUS > 1 )); then
    cat <<'EOF'
  DDP: --batch-size is PER GPU. LR is NOT auto-scaled, and the umi recipe's 3e-4 was tuned at
  batch 32 on the SINGLE-camera model -- an effective batch of N*BATCH here is a different
  optimization problem. Pass --lr explicitly if you want it changed.

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
    "${EXTRA_ARGS[@]}"
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
