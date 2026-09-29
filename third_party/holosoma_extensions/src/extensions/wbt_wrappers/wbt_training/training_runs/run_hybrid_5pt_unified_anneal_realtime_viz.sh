#!/usr/bin/env bash
# Build + launch the in-browser mjswan visualization for
# exp:g1-wbt-5pt-hybrid-unified-anneal — the 5-POINT LOCAL C1 UNIFIED anneal twin (R1).
#
# ── Relationship to the 3pt twin (run_hybrid_3pt_unified_anneal_realtime_viz.sh) ──
# SAME force stack, SAME unified reward (including the two TCP-precision terms), SAME
# browser physics (causal spring wall + F/T sensor + contact-gated integrator), SAME R1
# anneal route. The ONLY difference is the TRACKING FRONT-END: 5 points instead of 3, so
# 170D -> 184D. On the training side both presets are literally
# `replace(g1_wbt_3pt_hybrid_unified_anneal, observation=...)`, i.e. every non-observation
# config field is the SAME OBJECT — see config_values/hybrid_unified_5pt_dense.py.
#
# ── WHERE THE FIVE POINTS ARE ──────────────────────────────────────────────────
#   1. left hand   @ left_wrist_yaw_link  + (0.242, 0, 0)   the fingertip TCP
#   2. right hand  @ right_wrist_yaw_link + (0.242, 0, 0)   the fingertip TCP
#   3. head        @ torso_link           + (0, 0, 0.35)    ≈ the HMD point
#   4. left foot   @ left_ankle_roll_link  (bare origin, no sole lever)
#   5. right foot  @ right_ankle_roll_link (bare origin, no sole lever)
# That is exactly exp:g1-wbt-maskedmimic's `five_point` hover mode after its alias table
# (torso→torso_link, left/right_hand→*_wrist_yaw_link). The feet are APPENDED, so slots
# [0:9] of the position vector are byte-identically the 3pt twin's.
#
# ── Why this viz forces observe-only (load-bearing, inherited verbatim) ────────
# An R1 checkpoint bakes `servo_observe_only=False`; the terminal α lives ONLY in
# curriculum runtime state and is NEVER serialized (verified on dlcxc59d's own config:
# servo_enabled=True / servo_observe_only=False with Env/servo_alpha still 1.0). If the
# browser trusted the baked flag it would run the FULL Jᵀ ctrl injection. main.py keys off
# the "unified" preset NAME and forces `servo_observe_only=True`, which is what upholds
# "no unified twin injects a servo at deploy". NOT the -layerA pattern: `servo_enabled`
# stays True so the runtime keeps publishing the WOUND I/Δx the actor obs consume.
#
# Caveat while watching an EARLY checkpoint: if α was still ≈1 at export, the policy was
# trained WITH servo help it does not get here, so the arms may under-reach in contact.
# That gap closing as α anneals to 0 IS the R1 experiment.
#
# Live in the browser:
#   * purple F^cmd / orange F^ext / cyan F^meas arrows, all TCP-anchored
#   * the translucent virtual spring wall, reddening as the TCP penetrates it
#   * a top-center badge: UNIFIED POLICY (free space / L+R in contact)
#   * panel rows: F meas (N), e=Fc+Fm (N), contact φ, I (N), setpoint shift Δx (cm)
#   * FIVE tracking spheres on the live robot (solid) + the reference ghost (translucent):
#     cyan L hand, magenta R hand, yellow head, bright-green L foot, dark-green R foot.
#     The two hand spheres are CO-LOCATED with the WHITE force sphere (TCP-unified
#     geometry); the two green foot spheres are what is NEW vs the 3pt bundle — watch them
#     against the ghost's during walking clips.
# --preset g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt (184D obs -> ONNX [1, 736]).
#
# ── Getting the ONNX (same as the other realtime viz launchers) ───────────────
#   Local newest:  RUN_DIR=/abs/run  bash .../run_hybrid_5pt_unified_anneal_realtime_viz.sh
#   Pinned:        MODEL=/abs/model_XXXXX.onnx  bash .../run_hybrid_5pt_unified_anneal_realtime_viz.sh
#   From W&B:      WANDB_RUN=<entity>/<project>/<run_id> \
#                  [WANDB_MODEL=model_XXXXX.onnx]  bash .../run_hybrid_5pt_unified_anneal_realtime_viz.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_realtime_viz_common.sh"

PRESET="${PRESET:-g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt}"
# Canonical run: dlcxc59d (WBT-Hybrid5ptUniAnneal-bones-32gpu-gloo) — verified from its
# W&B config to be exp:g1-wbt-5pt-hybrid-unified-anneal: actor obs00..obs13 with
# obs01 = hybrid_unified_5pt_dense:vr_5point_local_target_tcp, ONNX obs [1, 736], R1 anneal
# (servo_enabled=True, servo_observe_only=False, servo_authority curriculum present), phase 1
# (sync_distal_mass_from_sim=False), TCP rewards w=4.0/σ=0.10 m and w=2.0/σ=0.25 rad.
#
# 736D is UNIQUE in the preset registry, so unlike the 3pt twin there is no
# same-width-different-geometry checkpoint that could be pinned here by mistake — the obs-dim
# check alone would reject any other family.
# The release pins no canonical W&B run: pass MODEL=/abs/model_XXXXX.onnx (or WANDB_RUN= /
# RUN_DIR=), or leave all three unset to use the NEWEST local export under logs/WBT-Holosoma.
DEFAULT_WANDB_RUN=""
OUT="${OUT:-$REPO/dist/wbt_realtime_eval_hybrid_5pt_unified_anneal}"
# 8012: the next free port (8000-8004 and 8006-8011 are taken by the other launchers), so
# this can serve side by side with the 3pt twin on 8009 for an A/B. serve_bundle cannot bind
# over an active LISTEN socket; override with PORT=... if 8012 is busy on your box.
PORT="${PORT:-8012}"
# --show-control-point: for a hybrid preset main.py supersedes the VR sphere with a WHITE
# TCP marker (the force contact point), so this is safe/consistent with the 3pt twin.
EXTRA_BUILD_ARGS=(--show-control-point)

wbt_realtime_viz_main
