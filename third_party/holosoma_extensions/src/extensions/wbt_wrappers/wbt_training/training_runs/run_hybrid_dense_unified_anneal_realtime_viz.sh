#!/usr/bin/env bash
# Build + launch the in-browser mjswan visualization for
# exp:g1-wbt-hybrid-unified-anneal — the DENSE-tracking LOCAL C1 UNIFIED anneal twin (R1).
#
# ── Relationship to the 3pt / 5pt twins ────────────────────────────────────────
# SAME force stack, SAME unified reward (including the two TCP-precision terms), SAME
# browser physics (causal spring wall + F/T sensor + contact-gated integrator), SAME R1
# anneal route. The ONLY difference is the TRACKING FRONT-END: this preset carries NO
# keypoints at all — `motion_command` (58D = all 29 joint targets + velocities, which
# already subsumes the 24D command_lower_body slice) replaces the keypoint block, so
# 170D -> 183D. On the training side it is literally
# `replace(g1_wbt_3pt_hybrid_unified_anneal, observation=...)` — every non-observation
# config field is the SAME OBJECT. See config_values/hybrid_unified_5pt_dense.py.
#
# The two hands' TCP tracking is STILL explicitly priced here: tcp_position_error_exp
# (w=4.0, σ=0.10 m, servo_shifted) and tcp_orientation_error_exp (w=2.0, σ=0.25 rad) read
# the motion command and the robot data views directly, never the obs layout — so the dense
# policy is paid for putting each 0.242 m fingertip TCP where the reference wants it even
# though it observes joint targets rather than keypoints.
#
# ⚠️ 183D/732D IS ALSO exp:g1-wbt-3pt-hybrid-global-vel's WIDTH. That is an arithmetic
# coincidence, not a shared layout: the term names differ completely and not one slot lines
# up. The obs-dim check therefore CANNOT tell the two apart, so main.py carries
# `_assert_checkpoint_tracking_family`, which compares the checkpoint's own obs term names
# (dense `motion_command` vs a `vr_*` keypoint slot) and refuses the build on a mismatch.
# Pointing this launcher at a global-vel checkpoint (or that launcher at this one) fails
# loudly instead of building a bundle that feeds the policy a different vector and falls.
#
# ── Why this viz forces observe-only (load-bearing, inherited verbatim) ────────
# An R1 checkpoint bakes `servo_observe_only=False`; the terminal α lives ONLY in curriculum
# runtime state and is NEVER serialized (verified on 7r54008a's own config: servo_enabled=True
# / servo_observe_only=False with Env/servo_alpha still 1.0). main.py keys off the "unified"
# preset NAME and forces `servo_observe_only=True`, upholding "no unified twin injects a servo
# at deploy". NOT the -layerA pattern: `servo_enabled` stays True so the runtime keeps
# publishing the WOUND I/Δx the actor obs consume.
#
# Live in the browser:
#   * purple F^cmd / orange F^ext / cyan F^meas arrows, all TCP-anchored
#   * the translucent virtual spring wall, reddening as the TCP penetrates it
#   * a top-center badge: UNIFIED POLICY (free space / L+R in contact)
#   * panel rows: F meas (N), e=Fc+Fm (N), contact φ, I (N), setpoint shift Δx (cm)
#   * the WHITE TCP force sphere per hand — and NO tracking-point spheres, deliberately:
#     this policy tracks all 29 joints, so there is no keypoint to mark. Judge tracking
#     against the reference GHOST's whole pose instead of against marker pairs. (That
#     absence is the quickest visual check that you loaded the dense bundle and not a
#     keypoint one.)
# --preset g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt (183D obs -> ONNX [1, 732]).
#
# ── Getting the ONNX (same as the other realtime viz launchers) ───────────────
#   Local newest:  RUN_DIR=/abs/run  bash .../run_hybrid_dense_unified_anneal_realtime_viz.sh
#   Pinned:        MODEL=/abs/model_XXXXX.onnx  bash .../run_hybrid_dense_unified_anneal_realtime_viz.sh
#   From W&B:      WANDB_RUN=<entity>/<project>/<run_id> \
#                  [WANDB_MODEL=model_XXXXX.onnx]  bash .../run_hybrid_dense_unified_anneal_realtime_viz.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_realtime_viz_common.sh"

PRESET="${PRESET:-g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt}"
# Canonical run: 7r54008a (WBT-HybridDenseUniAnneal-bones-32gpu-gloo) — verified from its
# W&B config to be exp:g1-wbt-hybrid-unified-anneal: actor obs00..obs11 with
# obs00 = holosoma...wbt:motion_command and NO vr_*/command_lower_body slot, ONNX obs
# [1, 732], R1 anneal (servo_enabled=True, servo_observe_only=False, servo_authority
# curriculum present), phase 1 (sync_distal_mass_from_sim=False).
#
# Pinning the RIGHT run matters more here than for the 5pt twin: a global-vel checkpoint has
# the identical 732D width. The tracking-family guard in main.py is the backstop.
# The release pins no canonical W&B run: pass MODEL=/abs/model_XXXXX.onnx (or WANDB_RUN= /
# RUN_DIR=), or leave all three unset to use the NEWEST local export under logs/WBT-Holosoma.
DEFAULT_WANDB_RUN=""
OUT="${OUT:-$REPO/dist/wbt_realtime_eval_hybrid_dense_unified_anneal}"
# 8013: the next free port after the 5pt twin's 8012, so all three unified anneal bundles
# (3pt 8009, 5pt 8012, dense 8013) can serve side by side. serve_bundle cannot bind over an
# active LISTEN socket; override with PORT=... if 8013 is busy on your box.
PORT="${PORT:-8013}"
# --show-control-point: for a hybrid preset main.py supersedes the VR sphere with a WHITE
# TCP marker (the force contact point). Kept for parity with the other hybrid launchers —
# the flag's own vr_2point branch is a no-op here (no 2pt term in this preset).
EXTRA_BUILD_ARGS=(--show-control-point)

wbt_realtime_viz_main
