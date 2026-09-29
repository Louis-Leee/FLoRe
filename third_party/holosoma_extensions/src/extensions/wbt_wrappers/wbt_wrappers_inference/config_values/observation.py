"""Observation configurations for TML and BM WBT wrapper policies."""

from __future__ import annotations

from holosoma_inference.config.config_types.observation import ObservationConfig

# =============================================================================
# TML WBT Observation Configuration
# =============================================================================

tml_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "tml_command",
            "motion_anchor_ori_b",
            "projected_gravity",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ]
    },
    obs_dims={
        "tml_command": 58,
        "motion_anchor_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "tml_command": 1.0,
        "motion_anchor_ori_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 1},
)

# =============================================================================
# BM WBT Observation Configuration
# =============================================================================

bm_wbt = ObservationConfig(
    obs_dict={"actor_obs": ["command", "motion_ref_ori_b", "base_ang_vel", "joint_pos", "joint_vel", "actions"]},
    obs_dims={"command": 58, "motion_ref_ori_b": 6, "base_ang_vel": 3, "joint_pos": 29, "joint_vel": 29, "actions": 29},
    obs_scales={
        "command": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_ang_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 1},
)

# =============================================================================
# Holosoma WBT Observation Configuration (matches wbt_training)
# 157D per frame x 4 history = 628D total
# Term order must match training alphabetical sort: command, motion_ref_ori_b,
# projected_gravity, base_ang_vel, joint_pos, joint_vel, actions
# =============================================================================

holosoma_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "command",
            "motion_ref_ori_b",
            "projected_gravity",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    },
    obs_dims={
        "command": 58,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "command": 1.0,
        "motion_ref_ori_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 4},
)

# =============================================================================
# Holosoma 3pt WBT Observation Configuration
# 144D per frame x 4 history = 576D total
# command_lower_body(24) + vr_3point_pos(9) + vr_3point_orn(12) +
# motion_ref_ori_b(6) + projected_gravity(3) + base_ang_vel(3) +
# joint_pos(29) + joint_vel(29) + actions(29) = 144
# =============================================================================

holosoma_3pt_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "command_lower_body",
            "vr_3point_pos",
            "vr_3point_orn",
            "motion_ref_ori_b",
            "projected_gravity",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    },
    obs_dims={
        "command_lower_body": 24,
        "vr_3point_pos": 9,
        "vr_3point_orn": 12,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "command_lower_body": 1.0,
        "vr_3point_pos": 1.0,
        "vr_3point_orn": 1.0,
        "motion_ref_ori_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 4},
)

# =============================================================================
# Holosoma 3pt Global WBT Observation Configuration
# 147D per frame x 4 history = 588D total
# command_lower_body(24) + vr_3point_pos(9) + vr_3point_orn(12) +
# motion_ref_pos_b(3) + motion_ref_ori_b(6) + projected_gravity(3) +
# base_ang_vel(3) + joint_pos(29) + joint_vel(29) + actions(29) = 147
# =============================================================================

holosoma_3pt_global_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "command_lower_body",
            "vr_3point_pos",
            "vr_3point_orn",
            "motion_ref_pos_b",
            "motion_ref_ori_b",
            "projected_gravity",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    },
    obs_dims={
        "command_lower_body": 24,
        "vr_3point_pos": 9,
        "vr_3point_orn": 12,
        "motion_ref_pos_b": 3,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "command_lower_body": 1.0,
        "vr_3point_pos": 1.0,
        "vr_3point_orn": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_ori_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 4},
)

# =============================================================================
# Holosoma 2pt WBT Observation Configuration (matches wbt_training exp:g1-wbt-2pt)
# 141D per frame x 4 history = 564D total
# command_lower_body(24) + vr_2point_pos(6) + vr_2point_orn(12, 6D rot rep) +
# motion_ref_ori_b(6) + projected_gravity(3) + base_ang_vel(3) +
# joint_pos(29) + joint_vel(29) + actions(29) = 141
# =============================================================================

holosoma_2pt_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "command_lower_body",
            "vr_2point_pos",
            "vr_2point_orn",
            "motion_ref_ori_b",
            "projected_gravity",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    },
    obs_dims={
        "command_lower_body": 24,
        "vr_2point_pos": 6,
        "vr_2point_orn": 12,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "command_lower_body": 1.0,
        "vr_2point_pos": 1.0,
        "vr_2point_orn": 1.0,
        "motion_ref_ori_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 4},
)

# =============================================================================
# Holosoma 2pt + UniFP-style force-tracking Observation Configuration
# (matches wbt_training exp:g1-wbt-2pt-force)
# 147D per frame x 4 history = 588D total
# command_lower_body(24) + vr_2point_pos(6) + vr_2point_orn(12, 6D rot rep) +
# force_cmd_local(6) + motion_ref_ori_b(6) + projected_gravity(3) +
# base_ang_vel(3) + joint_pos(29) + joint_vel(29) + actions(29) = 147
#
# Term order matches training alphabetical sort (obs0..obs9). At eval time the
# policy fills `force_cmd_local` with zeros (no F^cmd injected) — UniFP Eq.2
# target then collapses to the raw `ref_w`, so this exercises pure 2pt position
# tracking on a force-trained checkpoint.
#
# FRAME (--F_on_hand_frame parity): `force_cmd_local` is fed in the PELVIS frame
# by default. A checkpoint trained with `--F_on_hand_frame` instead expects this
# term in each hand's (wrist_yaw) frame — set HOLOSOMA_FORCE_CMD_HAND_FRAME=1 at
# deploy so the policy rotates the pelvis-frame F^cmd override into the hand frame
# per step (see holosoma_wbt.py:_force_cmd_local_obs). The 6D layout is identical
# in both frames, and F^cmd=0 maps to 0 in either frame, so the default
# (zero-force) eval is frame-invariant and unaffected by the flag.
# =============================================================================

holosoma_2pt_force_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "command_lower_body",
            "vr_2point_pos",
            "vr_2point_orn",
            "force_cmd_local",
            "motion_ref_ori_b",
            "projected_gravity",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    },
    obs_dims={
        "command_lower_body": 24,
        "vr_2point_pos": 6,
        "vr_2point_orn": 12,
        "force_cmd_local": 6,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "command_lower_body": 1.0,
        "vr_2point_pos": 1.0,
        "vr_2point_orn": 1.0,
        # Training uses scale=0.04 on force_cmd_local; obs values are zeros at
        # eval time so the scale is moot, but kept here for parity if F^cmd is
        # ever piped in for non-zero eval (e.g. teleop force commands).
        "force_cmd_local": 0.04,
        "motion_ref_ori_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 4},
)

# =============================================================================
# Holosoma 2pt + UniFP force-tracking with F^ext IN THE ACTOR
# (matches wbt_training exp:g1-wbt-2pt-force-ext, wandb run fmotwon3)
# 153D per frame x 4 history = 612D total
# command_lower_body(24) + vr_2point_pos(6) + vr_2point_orn(12, 6D rot rep) +
# force_cmd_local(6) + force_ext_local(6) + motion_ref_ori_b(6) +
# projected_gravity(3) + base_ang_vel(3) + joint_pos(29) + joint_vel(29) +
# actions(29) = 153
#
# Delta vs holosoma_2pt_force_wbt (147D): force_ext_local ADDED after force_cmd_local
# (training actor layout obs3_force_cmd_local then obs4_force_ext_local). Reverts the
# UniFP actor/critic asymmetry — the actor observes F^ext directly. Both force terms
# scale 0.04 (pelvis/root frame). At headless eval both fill with zeros
# (holosoma_wbt.py) — the Eq.2 target collapses to the raw ref (pure 2pt tracking).
#
# CRITICAL ORDERING (2026-07-12 fix): the actor_obs LIST below MUST be in the
# alphabetically-SORTED key order the trained ONNX expects, because both the
# training ObservationManager (managers/observation/manager.py:139 sorts term keys)
# AND this inference path (holosoma_wbt.py:1429 iterates self._obs_terms in LIST
# order) concatenate that way. The training keys are un-zero-padded obs0..obs10
# (force_tracking.py:1447), so "obs10_actions" string-sorts to slot 2 (right after
# obs0_command_lower_body). Verified against fmotwon3 model_27000.onnx: `actions`
# is at obs slot [24]. A sequential list (actions last) misaligns every term past
# slot 0 → the policy flails and falls. obs_dims/obs_scales are dicts (keyed by
# name) so their order is irrelevant; ONLY the actor_obs list order matters.
# =============================================================================

holosoma_2pt_force_ext_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "command_lower_body",  # obs0
            "actions",  # obs10 — string sort places it at slot 2
            "vr_2point_pos",  # obs1
            "vr_2point_orn",  # obs2
            "force_cmd_local",  # obs3
            "force_ext_local",  # obs4
            "motion_ref_ori_b",  # obs5
            "projected_gravity",  # obs6
            "base_ang_vel",  # obs7
            "joint_pos",  # obs8
            "joint_vel",  # obs9
        ]
    },
    obs_dims={
        "command_lower_body": 24,
        "vr_2point_pos": 6,
        "vr_2point_orn": 12,
        "force_cmd_local": 6,
        "force_ext_local": 6,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "command_lower_body": 1.0,
        "vr_2point_pos": 1.0,
        "vr_2point_orn": 1.0,
        # Both force channels use the training 0.04 scale (= 1/f_{cmd,ext}_max=25 N).
        "force_cmd_local": 0.04,
        "force_ext_local": 0.04,
        "motion_ref_ori_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 4},
)

# =============================================================================
# Holosoma 2pt hybrid position-force controller Observation Configuration
# (matches wbt_training exp:g1-wbt-2pt-hybrid — BOTH the servo-ON run 31wei7by
# AND the Layer-A/servo-OFF run 6y8314vq, which share the SAME 165D actor obs)
# 165D per frame x 4 history = 660D total
# command_lower_body(24) + vr_2point_pos(6) + vr_2point_orn(12, 6D rot rep) +
# force_cmd_local(6) + force_meas_local(6) + integrator_state(6) + x_tgt_shift(6)
# + motion_ref_ori_b(6) + projected_gravity(3) + base_ang_vel(3) + joint_pos(29)
# + joint_vel(29) + actions(29) = 165
#
# Term order is hybrid.py's SORTED actor keys obs00..obs12 (hybrid.py:1030-1050).
# F^ext is critic-only in hybrid (NOT in this actor); the actor's causal force
# channel is force_meas_local (the simulated F/T reading). integrator_state and
# x_tgt_shift are the servo-internal states — allocated at fixed 165D even when
# servo is off (they read zeros). Scales: force_cmd_local + force_meas_local 0.04
# (=1/25 N); integrator_state + x_tgt_shift 1.0. In headless eval all four force
# terms fill with zeros (holosoma_wbt.py), so this runs pure 2pt tracking on a
# hybrid checkpoint; the browser fills them from the spring-wall + servo sim.
# =============================================================================

holosoma_2pt_hybrid_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "command_lower_body",
            "vr_2point_pos",
            "vr_2point_orn",
            "force_cmd_local",
            "force_meas_local",
            "integrator_state",
            "x_tgt_shift",
            "motion_ref_ori_b",
            "projected_gravity",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    },
    obs_dims={
        "command_lower_body": 24,
        "vr_2point_pos": 6,
        "vr_2point_orn": 12,
        "force_cmd_local": 6,
        "force_meas_local": 6,
        "integrator_state": 6,
        "x_tgt_shift": 6,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "command_lower_body": 1.0,
        "vr_2point_pos": 1.0,
        "vr_2point_orn": 1.0,
        "force_cmd_local": 0.04,
        "force_meas_local": 0.04,
        "integrator_state": 1.0,
        "x_tgt_shift": 1.0,
        "motion_ref_ori_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 4},
)

# =============================================================================
# Holosoma 3pt hybrid position-force controller Observation Configuration
# (matches wbt_training exp:g1-wbt-3pt-hybrid — BOTH the servo-ON and the
# Layer-A/servo-OFF variants, which share the SAME 168D actor obs)
# 168D per frame x 4 history = 672D total
# command_lower_body(24) + vr_3point_pos(9) + vr_3point_orn(12, quat) +
# force_cmd_local(6) + force_meas_local(6) + integrator_state(6) + x_tgt_shift(6)
# + motion_ref_ori_b(6) + projected_gravity(3) + base_ang_vel(3) + joint_pos(29)
# + joint_vel(29) + actions(29) = 168
#
# Term order is hybrid_3pt.py's SORTED actor keys obs00..obs12 — the 2pt-hybrid
# layout with vr_2point_{pos,orn} (6+12) swapped for vr_3point_{pos,orn} (9+12),
# so 165D → 168D. F^ext is critic-only in hybrid (NOT in this actor); the actor's
# causal force channel is force_meas_local (the simulated F/T reading).
# integrator_state and x_tgt_shift are servo-internal states — present at fixed
# 168D even when servo is off (they read zeros). Scales: force_cmd_local +
# force_meas_local 0.04 (=1/25 N); integrator_state + x_tgt_shift 1.0. In headless
# eval all four force terms fill with zeros (holosoma_wbt.py), so this runs pure
# 3pt tracking on a hybrid checkpoint; the browser fills them from the spring-wall
# + servo sim. The ACTOR joint-velocity dropout is a TRAIN-TIME obs-noise knob;
# at deploy dof_vel is fed clean (or zeroed via WBT_JV_EVAL_ZERO on the training
# side) — either way the eval obs LAYOUT (168D, joint_vel at this slot) is identical.
#
# 3pt-vs-2pt DELTA: vr_3point_orn is 12D RAW QUATERNION (3 bodies × 4, xyzw) —
# NOT the 2pt rot6d (2 wrists × 6). HolosomaWBTPolicy._compute_vr_fk already emits
# the per-body quaternion in its non-2pt branch, so the NPZ-replay/headless 3pt
# path needs no policy change; only the LIVE VIVE teleop override is 3pt-aware.
# =============================================================================

holosoma_3pt_hybrid_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "command_lower_body",
            "vr_3point_pos",
            "vr_3point_orn",
            "force_cmd_local",
            "force_meas_local",
            "integrator_state",
            "x_tgt_shift",
            "motion_ref_ori_b",
            "projected_gravity",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    },
    obs_dims={
        "command_lower_body": 24,
        "vr_3point_pos": 9,
        "vr_3point_orn": 12,
        "force_cmd_local": 6,
        "force_meas_local": 6,
        "integrator_state": 6,
        "x_tgt_shift": 6,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "command_lower_body": 1.0,
        "vr_3point_pos": 1.0,
        "vr_3point_orn": 1.0,
        "force_cmd_local": 0.04,
        "force_meas_local": 0.04,
        "integrator_state": 1.0,
        "x_tgt_shift": 1.0,
        "motion_ref_ori_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 4},
)

# =============================================================================
# Holosoma 3pt hybrid + global root pos/vel + 6D orientation Observation Config
# (matches wbt_training exp:g1-wbt-3pt-hybrid-global-vel — BOTH the servo-ON and
# the Layer-A/servo-OFF variants, which share the SAME 183D actor obs)
# 183D per frame x 4 history = 732D total
# command_lower_body(24) + vr_3point_pos(9) + vr_3point_orn_6d(18, 6D rot rep) +
# motion_ref_pos_b(3) + motion_ref_vel_b(3) + force_cmd_local(6) +
# force_meas_local(6) + integrator_state(6) + x_tgt_shift(6) + motion_ref_ori_b(6)
# + projected_gravity(3) + base_ang_vel(3) + base_lin_vel(3) + joint_pos(29) +
# joint_vel(29) + actions(29) = 183
#
# Term order is hybrid_3pt_global_vel.py's SORTED actor keys obs00..obs15 (also
# baked into the bqlzkwok ONNX experiment_config — the runtime term-ORDER
# validator cross-checks this list against the checkpoint at load). Two deltas
# vs holosoma_3pt_hybrid_wbt (168D):
#   (1) the tracked-body orientation is the 6D rotation rep `vr_3point_orn_6d`
#       (3 bodies × 6 = 18D, first two rotation-matrix columns) instead of the
#       raw xyzw quaternion `vr_3point_orn` (3 × 4 = 12D). The runtime FK emits
#       6D because the term name carries "vr_3point_orn_6d" (holosoma_wbt.py
#       `_vr_orn_6d`) — no quaternion double-cover OOD at deploy. The live VIVE
#       bridge still streams 3×quat (wire format unchanged); the policy converts
#       quat→6D at ingest (`_quat12_to_rot6d18`).
#   (2) the exp:g1-wbt-global-vel root-state trio motion_ref_pos_b +
#       motion_ref_vel_b + base_lin_vel is added, so the base can TRANSLATE
#       through the world (plain 3pt-hybrid is root-invariant and cannot walk).
#       detect_obs_mode → "3pt_global" (motion_ref_pos_b present) → first-tick
#       motion alignment fires; base_lin_vel needs a root-velocity source at
#       deploy (--task.fk-odometry true, or mocap).
# The four force/servo terms behave exactly as in holosoma_3pt_hybrid_wbt
# (zeros in headless eval → pure 3pt tracking; live Bota F/T + Jᵀ servo fill
# them on hardware). Scales: force_cmd_local + force_meas_local 0.04 (=1/25 N).
# =============================================================================

holosoma_3pt_hybrid_global_vel_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "command_lower_body",
            "vr_3point_pos",
            "vr_3point_orn_6d",
            "motion_ref_pos_b",
            "motion_ref_vel_b",
            "force_cmd_local",
            "force_meas_local",
            "integrator_state",
            "x_tgt_shift",
            "motion_ref_ori_b",
            "projected_gravity",
            "base_ang_vel",
            "base_lin_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    },
    obs_dims={
        "command_lower_body": 24,
        "vr_3point_pos": 9,
        "vr_3point_orn_6d": 18,
        "motion_ref_pos_b": 3,
        "motion_ref_vel_b": 3,
        "force_cmd_local": 6,
        "force_meas_local": 6,
        "integrator_state": 6,
        "x_tgt_shift": 6,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "base_lin_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "command_lower_body": 1.0,
        "vr_3point_pos": 1.0,
        "vr_3point_orn_6d": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_vel_b": 1.0,
        "force_cmd_local": 0.04,
        "force_meas_local": 0.04,
        "integrator_state": 1.0,
        "x_tgt_shift": 1.0,
        "motion_ref_ori_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "base_lin_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 4},
)

# =============================================================================
# Holosoma 3pt Hybrid Global-Vel UNIFIED WBT Observation Configuration
# (C1 unified single-policy, exp:g1-wbt-3pt-hybrid-global-vel-unified[-anneal])
# 185D per frame x 4 history = 740D total
#
# = the 183D holosoma_3pt_hybrid_global_vel_wbt layout + contact_phase(2) inserted
# right after force_meas_local (training obs07 slot). contact_phase is the SOFT
# deployable contact label ∈[0,1] per hand (hysteresis 2/1.5 N + 50 ms LPF on
# ‖F_meas‖). Headless eval fills it with zeros (free space — matches training's
# zeroed buffer at reset); the browser mjswan runtime computes it live from the
# spring-wall F_meas and publishes wbt_cphase__{l,r}. One config serves BOTH
# training twins (R2 observe-only / R1 anneal) — identical obs, and at deploy
# neither injects a servo (observe-only by construction / α annealed to 0).
# =============================================================================

holosoma_3pt_hybrid_global_vel_unified_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "command_lower_body",
            "vr_3point_pos",
            "vr_3point_orn_6d",
            "motion_ref_pos_b",
            "motion_ref_vel_b",
            "force_cmd_local",
            "force_meas_local",
            "contact_phase",
            "integrator_state",
            "x_tgt_shift",
            "motion_ref_ori_b",
            "projected_gravity",
            "base_ang_vel",
            "base_lin_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    },
    obs_dims={
        "command_lower_body": 24,
        "vr_3point_pos": 9,
        "vr_3point_orn_6d": 18,
        "motion_ref_pos_b": 3,
        "motion_ref_vel_b": 3,
        "force_cmd_local": 6,
        "force_meas_local": 6,
        "contact_phase": 2,
        "integrator_state": 6,
        "x_tgt_shift": 6,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "base_lin_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "command_lower_body": 1.0,
        "vr_3point_pos": 1.0,
        "vr_3point_orn_6d": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_vel_b": 1.0,
        "force_cmd_local": 0.04,
        "force_meas_local": 0.04,
        "contact_phase": 1.0,
        "integrator_state": 1.0,
        "x_tgt_shift": 1.0,
        "motion_ref_ori_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "base_lin_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 4},
)

# =============================================================================
# Holosoma 3pt hybrid UNIFIED (pelvis-LOCAL) Observation Configuration
# (matches wbt_training exp:g1-wbt-3pt-hybrid-unified AND its R1 twin
#  exp:g1-wbt-3pt-hybrid-unified-anneal — both share the SAME 170D actor obs;
#  the R2/R1 difference is a training-time curriculum, never an obs difference)
# 170D per frame x 4 history = 680D total
# command_lower_body(24) + vr_3point_pos_tcp(9) + vr_3point_orn(12, quat) +
# force_cmd_local(6) + force_meas_local(6) + contact_phase(2) +
# integrator_state(6) + x_tgt_shift(6) + motion_ref_ori_b(6) +
# projected_gravity(3) + base_ang_vel(3) + joint_pos(29) + joint_vel(29) +
# actions(29) = 170
#
# TCP-UNIFIED HAND POINT (user 2026-08-03): the position slot is `vr_3point_pos_tcp`,
# NOT `vr_3point_pos`. Same 9D width and the same three bodies, but the two HAND offsets
# are (0.242, 0, 0) — the fingertip TCP the force stack already acts on — instead of the
# 0.18 m VR-tracker lever, so position and force supervise ONE point (6.7 cm per hand
# apart before this change). The distinct term name is what selects the 0.242 FK in
# holosoma_wbt.py (`_vr_pos_tcp`), and it is also why the pre-2026-08-03 unified
# checkpoints (iss0qs71 / m3s0pw9i, trained at 0.18) can no longer be loaded against this
# preset: obs WIDTH is unchanged, so only the per-term name check catches the mismatch —
# `_validate_obs_term_order` raises on slot 1 rather than silently feeding the policy a
# target 6.7 cm from where it trained. Current matching run: wda3h4ll (retrained with
# tcp_position_error_exp w=4.0 / tcp_orientation_error_exp w=2.0 pricing the TCP).
#
# Term order is hybrid_unified.py's SORTED actor keys obs00..obs13 — the 168D
# holosoma_3pt_hybrid_wbt layout with contact_phase(2) INSERTED right after
# force_meas_local (so every slot from integrator_state on shifts by +1). This is
# the pelvis-LOCAL unified family: NO root pos/vel trio and the tracked-body
# orientation stays the raw xyzw QUATERNION vr_3point_orn (12D), so
# detect_obs_mode → "3pt" (root-invariant; this policy does NOT walk — that is
# the -global-vel-unified twin, whose inference preset is not on this branch yet).
#
# contact_phase is the SOFT, DEPLOYABLE per-hand contact label ∈[0,1]:
# hysteresis on ‖F_meas‖ (ON above 2.0 N, OFF below 1.5 N, holds in the dead
# band) smoothed by a 50 ms first-order LPF. Training computes it from the
# SIMULATED F/T reading (hybrid.py `contact_phase`), never from the privileged
# depth gate (`contact_gate` stays critic-only) — precisely so the real Bota
# reproduces the same signal at deploy. The runtime therefore computes it LIVE
# from the pelvis-frame Bota force (holosoma_wbt.py `_update_contact_phase`);
# with no live F/T source it reads zeros (free space — matching training's
# zeroed buffer at reset), so a headless/NPZ replay runs pure 3pt tracking.
# (Known gap: training's F_meas carries a simulated inertial term that the
# deploy bridge's RNEA compensation removes, so the deploy label fires on
# contact and essentially never on free-space motion — see the README.)
#
# UNIFIED ≠ SERVO-ON: these checkpoints were trained with the Jᵀ servo demoted
# to a pure REFERENCE GENERATOR (`servo_observe_only=True`, or annealed to α=0).
# The integrator KEEPS winding — integrator_state (I) and x_tgt_shift (Δx) are
# live, load-bearing actor inputs — but Δq_arm is NEVER injected into the PD
# path; the policy is the sole executor. holosoma_wbt.py keys observe-only off
# the PRESENCE of this contact_phase term, so it holds for both twins regardless
# of the flag baked into the checkpoint (an R1 ONNX carries
# servo_observe_only=False, and its annealed terminal α is never serialized).
# Scales: force_cmd_local + force_meas_local 0.04 (=1/25 N); contact_phase,
# integrator_state and x_tgt_shift 1.0.
# =============================================================================

holosoma_3pt_hybrid_unified_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "command_lower_body",
            "vr_3point_pos_tcp",
            "vr_3point_orn",
            "force_cmd_local",
            "force_meas_local",
            "contact_phase",
            "integrator_state",
            "x_tgt_shift",
            "motion_ref_ori_b",
            "projected_gravity",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    },
    obs_dims={
        "command_lower_body": 24,
        "vr_3point_pos_tcp": 9,
        "vr_3point_orn": 12,
        "force_cmd_local": 6,
        "force_meas_local": 6,
        "contact_phase": 2,
        "integrator_state": 6,
        "x_tgt_shift": 6,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "command_lower_body": 1.0,
        "vr_3point_pos_tcp": 1.0,
        "vr_3point_orn": 1.0,
        "force_cmd_local": 0.04,
        "force_meas_local": 0.04,
        "contact_phase": 1.0,
        "integrator_state": 1.0,
        "x_tgt_shift": 1.0,
        "motion_ref_ori_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 4},
)

# =============================================================================
# Holosoma 5pt Hybrid UNIFIED (LOCAL) WBT Observation Configuration
# (exp:g1-wbt-5pt-hybrid-unified-anneal, W&B run dlcxc59d)
# 184D per frame x 4 history = 736D total
#
# = holosoma_3pt_hybrid_unified_wbt (170D) with the TRACKING FRONT-END widened from 3 to
# 5 points: vr_3point_pos_tcp(9) + vr_3point_orn(12) become vr_5point_pos_tcp(15) +
# vr_5point_orn(20). +14D. EVERY other slot — the force block (F^cmd, F^meas,
# contact_phase, I, Δx), the scales, the history — is the 170D config's, unchanged.
#
# THE FIVE POINTS = the 3pt set (left hand @ 0.242 m TCP, right hand @ 0.242 m TCP,
# head @ torso_link +0.35 z) PLUS BOTH FEET (`*_ankle_roll_link`, bare origin). That is
# exp:g1-wbt-maskedmimic's `five_point` hover mode after its alias resolution. The feet
# are APPENDED, not interleaved, so slots [0:9]/[0:12] of the two vectors stay
# byte-identical IN MEANING to the 3pt layout (see robot_config.VR_5POINT_BODIES).
#
# DEPLOY is wired (user 2026-08-27): `detect_obs_mode` returns "5pt",
# `holosoma_wbt._vr_offsets()` returns the 3pt-TCP triple + two ZERO foot offsets, and
# `_compute_vr_fk` loops over 5 frame IDs — the whole clutch / slew / arm-gate path was
# already written against the array WIDTH, so it needed no per-point-count widening.
# Registered as `g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt`.
# This config remains the single source of truth for the obs ORDER + WIDTHS, which the
# browser-viz preset (wbt_realtime_eval.presets, same key) mirrors and which
# test_presets.py::test_preset_dims_match_observation_config pins.
# =============================================================================

holosoma_5pt_hybrid_unified_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "command_lower_body",
            "vr_5point_pos_tcp",
            "vr_5point_orn",
            "force_cmd_local",
            "force_meas_local",
            "contact_phase",
            "integrator_state",
            "x_tgt_shift",
            "motion_ref_ori_b",
            "projected_gravity",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    },
    obs_dims={
        "command_lower_body": 24,
        "vr_5point_pos_tcp": 15,
        "vr_5point_orn": 20,
        "force_cmd_local": 6,
        "force_meas_local": 6,
        "contact_phase": 2,
        "integrator_state": 6,
        "x_tgt_shift": 6,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "command_lower_body": 1.0,
        "vr_5point_pos_tcp": 1.0,
        "vr_5point_orn": 1.0,
        "force_cmd_local": 0.04,
        "force_meas_local": 0.04,
        "contact_phase": 1.0,
        "integrator_state": 1.0,
        "x_tgt_shift": 1.0,
        "motion_ref_ori_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 4},
)

# =============================================================================
# Holosoma DENSE Hybrid UNIFIED (LOCAL) WBT Observation Configuration
# (exp:g1-wbt-hybrid-unified-anneal, W&B run 7r54008a)
# 183D per frame x 4 history = 732D total
#
# = holosoma_wbt's DENSE tracking front-end (`command` = 58D motion_command = all 29
# joint targets + velocities — NO keypoints, and it already subsumes the 24D
# command_lower_body slice) + the SAME force block the keypoint unified configs carry
# (force_cmd_local 6 + force_meas_local 6 + contact_phase 2 + integrator_state 6 +
# x_tgt_shift 6 = 26D), inserted right after the tracking term.
# 183 = 58 + 26 + 6 + 3 + 3 + 29 + 29 + 29.
#
# ⚠️ 183D/732D COLLIDES with holosoma_3pt_hybrid_global_vel_wbt (also 183D/732D). That is
# an arithmetic coincidence, not a shared layout — the term names differ completely — but
# it means the obs-DIM check alone cannot tell the two apart. main.py's
# `_assert_checkpoint_tracking_family` is the guard that does (dense joint targets vs VR
# keypoints), keyed on the checkpoint's own obs term names.
#
# Registered as `g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt` (user 2026-08-27). At
# deploy the hybrid force terms read zeros without a live F/T source, exactly as they do
# for the other hybrid configs. The live teleop path overrides the 58D `command` term from
# the completion's joint stream (`holosoma_wbt._live_dense_command`, remapped onto
# `dof_names` BY NAME); with no live bridge this replays the NPZ clip's motion_command,
# i.e. the legacy dense behaviour. See the 5pt note above.
# =============================================================================

holosoma_dense_hybrid_unified_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "command",
            "force_cmd_local",
            "force_meas_local",
            "contact_phase",
            "integrator_state",
            "x_tgt_shift",
            "motion_ref_ori_b",
            "projected_gravity",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    },
    obs_dims={
        "command": 58,
        "force_cmd_local": 6,
        "force_meas_local": 6,
        "contact_phase": 2,
        "integrator_state": 6,
        "x_tgt_shift": 6,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "command": 1.0,
        "force_cmd_local": 0.04,
        "force_meas_local": 0.04,
        "contact_phase": 1.0,
        "integrator_state": 1.0,
        "x_tgt_shift": 1.0,
        "motion_ref_ori_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 4},
)

# =============================================================================
# Holosoma 2pt Global WBT Observation Configuration
# 144D per frame x 4 history = 576D total
# command_lower_body(24) + vr_2point_pos(6) + vr_2point_orn(12, 6D rot rep) +
# motion_ref_pos_b(3) + motion_ref_ori_b(6) + projected_gravity(3) +
# base_ang_vel(3) + joint_pos(29) + joint_vel(29) + actions(29) = 144
# =============================================================================

holosoma_2pt_global_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "command_lower_body",
            "vr_2point_pos",
            "vr_2point_orn",
            "motion_ref_pos_b",
            "motion_ref_ori_b",
            "projected_gravity",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    },
    obs_dims={
        "command_lower_body": 24,
        "vr_2point_pos": 6,
        "vr_2point_orn": 12,
        "motion_ref_pos_b": 3,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "command_lower_body": 1.0,
        "vr_2point_pos": 1.0,
        "vr_2point_orn": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_ori_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 4},
)

# =============================================================================
# Holosoma Global WBT Observation Configuration
# 160D per frame x 4 history = 640D total
# command(58) + motion_ref_ori_b(6) + motion_ref_pos_b(3) +
# projected_gravity(3) + base_ang_vel(3) +
# joint_pos(29) + joint_vel(29) + actions(29) = 160
# =============================================================================

holosoma_global_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "command",
            "motion_ref_ori_b",
            "motion_ref_pos_b",
            "projected_gravity",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    },
    obs_dims={
        "command": 58,
        "motion_ref_ori_b": 6,
        "motion_ref_pos_b": 3,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "command": 1.0,
        "motion_ref_ori_b": 1.0,
        "motion_ref_pos_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 4},
)

# =============================================================================
# Holosoma Global+Vel WBT Observation Configuration
# 166D per frame x 4 history = 664D total
# command(58) + motion_ref_ori_b(6) + motion_ref_pos_b(3) +
# motion_ref_vel_b(3) + projected_gravity(3) + base_ang_vel(3) +
# base_lin_vel(3) + joint_pos(29) + joint_vel(29) + actions(29) = 166
# =============================================================================

holosoma_global_vel_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "command",
            "motion_ref_ori_b",
            "motion_ref_pos_b",
            "motion_ref_vel_b",
            "projected_gravity",
            "base_ang_vel",
            "base_lin_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    },
    obs_dims={
        "command": 58,
        "motion_ref_ori_b": 6,
        "motion_ref_pos_b": 3,
        "motion_ref_vel_b": 3,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "base_lin_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "command": 1.0,
        "motion_ref_ori_b": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_vel_b": 1.0,
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "base_lin_vel": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 4},
)

# =============================================================================
# Holosoma Terrain WBT Observation Configuration
# 449D per frame x 1 history = 449D total
# Insertion order matches wbt_training.config_values.terrain:wbt_terrain_actor_obs:
#   motion_command(58) + motion_ref_pos_b(3) + motion_ref_ori_b(6) +
#   base_lin_vel(3) + base_ang_vel(3) + dof_pos(29) + dof_vel(29) +
#   actions(29) + height_scan(289) = 449
# =============================================================================

holosoma_terrain_wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "motion_command",
            "motion_ref_pos_b",
            "motion_ref_ori_b",
            "base_lin_vel",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "height_scan",
        ]
    },
    obs_dims={
        "motion_command": 58,
        "motion_ref_pos_b": 3,
        "motion_ref_ori_b": 6,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "height_scan": 289,
    },
    obs_scales={
        "motion_command": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
        "height_scan": 1.0,
    },
    history_length_dict={"actor_obs": 1},
)

# =============================================================================
# Holosoma Terrain no-heightscan WBT Observation Configuration
# 160D per frame (no height_scan rays). Matches the insertion order used by
# wbt_training.config_values.terrain:g1_29dof_wbt_terrain_observation_no_heightscan:
#   motion_command(58) + motion_ref_pos_b(3) + motion_ref_ori_b(6) +
#   base_lin_vel(3) + base_ang_vel(3) + dof_pos(29) + dof_vel(29) + actions(29)
# = 160. Order matters — it's the order the exported ONNX expects.
# =============================================================================

holosoma_terrain_wbt_no_heightscan = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "motion_command",
            "motion_ref_pos_b",
            "motion_ref_ori_b",
            "base_lin_vel",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ]
    },
    obs_dims={
        "motion_command": 58,
        "motion_ref_pos_b": 3,
        "motion_ref_ori_b": 6,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "motion_command": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={"actor_obs": 1},
)

# TEMPORARY: TMLCHIPPolicy bypasses this dict; it exists for downstream tooling.
_tml_chip_dims = {
    "lower_command": 24,
    "vr_3point_pos": 9,
    "vr_3point_ori": 12,
    "motion_anchor_ori_b": 6,
    "projected_gravity": 3,
    "omega": 3,
    "q": 29,
    "dq": 29,
    "last_action": 29,
    "compliance": 3,
}

tml_chip = ObservationConfig(
    obs_dict={"actor_obs": list(_tml_chip_dims.keys())},
    obs_dims=dict(_tml_chip_dims),
    obs_scales={k: 1.0 for k in _tml_chip_dims},
    history_length_dict={"actor_obs": 1},
)


# Training exports obs terms in alphabetical order, so the obs0N_ prefixes
# force the layout the ONNX input expects.
def _holosoma_chip_obs(dims: dict[str, int]) -> ObservationConfig:
    return ObservationConfig(
        obs_dict={"actor_obs": list(dims.keys())},
        obs_dims=dict(dims),
        obs_scales={k: 1.0 for k in dims},
        history_length_dict={"actor_obs": 10},
    )


holosoma_chip_local_wbt = _holosoma_chip_obs(
    {
        "obs00_command_lower_body": 24,
        "obs01_vr_3point_pos": 9,
        "obs02_vr_3point_orn": 12,
        "obs03_motion_ref_ori_b": 6,
        "obs04_projected_gravity": 3,
        "obs05_base_ang_vel": 3,
        "obs06_dof_pos": 29,
        "obs07_dof_vel": 29,
        "obs08_actions": 29,
        "obs09_compliance": 3,
    }
)

holosoma_chip_global_wbt = _holosoma_chip_obs(
    {
        "obs00_command_lower_body": 24,
        "obs01_vr_3point_pos": 9,
        "obs02_vr_3point_orn": 12,
        "obs03_motion_ref_pos_b": 3,
        "obs04_motion_ref_ori_b": 6,
        "obs05_projected_gravity": 3,
        "obs06_base_ang_vel": 3,
        "obs07_dof_pos": 29,
        "obs08_dof_vel": 29,
        "obs09_actions": 29,
        "obs10_compliance": 3,
    }
)

# =============================================================================
# Holosoma MaskedMimic Observation Configuration
# 2409D total, no history stacking.
# reduced_coords(67) | target_poses(1260) | target_masks(180) |
# joint_targets(870) | joint_mode(1) | prev_actions(29) | ee_masses(2)
# =============================================================================

holosoma_maskedmimic = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "reduced_coords",
            "target_poses",
            "target_masks",
            "joint_targets",
            "joint_mode",
            "prev_actions",
            "ee_masses",
        ]
    },
    obs_dims={
        "reduced_coords": 67,
        "target_poses": 1260,
        "target_masks": 180,
        "joint_targets": 870,
        "joint_mode": 1,
        "prev_actions": 29,
        "ee_masses": 2,
    },
    obs_scales={
        "reduced_coords": 1.0,
        "target_poses": 1.0,
        "target_masks": 1.0,
        "joint_targets": 1.0,
        "joint_mode": 1.0,
        "prev_actions": 1.0,
        "ee_masses": 1.0,
    },
    history_length_dict={"actor_obs": 1},
)
