"""Preset definitions for the seven holosoma-*-wbt obs configurations.

Each preset is a thin record mapping the inference preset name (the same one
``batch_headless_eval.sh --preset <name>`` uses) to:

* the obs term keys in policy order (the order the trained ONNX expects)
* the per-term TS class to wire into the mjswan policy components
* the per-term base width (single-frame dim, before history stacking)
* the VR FK family (``2pt`` / ``3pt`` / ``5pt`` / ``none``) that ``bake_motions``
  should pre-compute for this preset's motions

The component sizes here MUST match
``wbt_wrappers_inference.config_values.observation`` — that module is the
single source of truth for what a given training preset's ONNX consumes;
this file just mirrors the relevant subset for browser playback.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _Term:
    """A single obs term: ``key`` is the policy_obs name (matches Python side);
    ``ts_name`` is the TS class registered in :mod:`obs_registry`; ``base_dim``
    is the single-frame width (history is applied by ``WbtHistoryBase``)."""

    key: str
    ts_name: str
    base_dim: int


# fmt: off
# Per-term width table (single source of truth across all presets).
_TERMS = {
    "command":             _Term("command",            "WbtCommand",          58),
    "command_lower_body":  _Term("command_lower_body", "WbtCommandLowerBody", 24),
    "vr_2point_pos":       _Term("vr_2point_pos",      "WbtVr2PointPos",       6),
    "vr_2point_orn":       _Term("vr_2point_orn",      "WbtVr2PointOrn",      12),
    "vr_3point_pos":       _Term("vr_3point_pos",      "WbtVr3PointPos",       9),
    # TCP-unified 3pt hand point (exp:g1-wbt-3pt-hybrid-unified[-anneal], 2026-08-03):
    # same 9D, hands baked at the 0.242 m force TCP instead of the 0.18 m VR lever.
    "vr_3point_pos_tcp":   _Term("vr_3point_pos_tcp",  "WbtVr3PointPosTcp",    9),
    "vr_3point_orn":       _Term("vr_3point_orn",      "WbtVr3PointOrn",      12),
    # 5-POINT tracking (exp:g1-wbt-5pt-hybrid-unified-anneal, 2026-08-27): the 3pt-TCP
    # set + BOTH FEET (*_ankle_roll_link, bare origin). 15D = 5 bodies x 3; 20D = 5 x 4
    # (quat xyzw, same rep as vr_3point_orn). Slots [0:9]/[0:12] are the 3pt vectors
    # unchanged — the bake builds these by CONCATENATION onto the 3pt tables.
    "vr_5point_pos_tcp":   _Term("vr_5point_pos_tcp",  "WbtVr5PointPosTcp",   15),
    "vr_5point_orn":       _Term("vr_5point_orn",      "WbtVr5PointOrn",      20),
    # 6D rot rep of the 3pt orientation (3 bodies x 6 = 18D) — exp:g1-wbt-3pt-hybrid-
    # global-vel. Distinct from the 12D quat vr_3point_orn so 3pt-hybrid is untouched.
    "vr_3point_orn_6d":    _Term("vr_3point_orn_6d",   "WbtVr3PointOrn6d",    18),
    "motion_ref_pos_b":    _Term("motion_ref_pos_b",   "WbtMotionRefPosB",     3),
    "motion_ref_ori_b":    _Term("motion_ref_ori_b",   "WbtMotionRefOriB",     6),
    "motion_ref_vel_b":    _Term("motion_ref_vel_b",   "WbtMotionRefVelB",     3),
    "projected_gravity":   _Term("projected_gravity",  "WbtProjectedGravity",  3),
    "base_ang_vel":        _Term("base_ang_vel",       "WbtBaseAngVel",        3),
    "base_lin_vel":        _Term("base_lin_vel",       "WbtBaseLinVel",        3),
    "joint_pos":           _Term("joint_pos",          "WbtJointPos",         29),
    "joint_vel":           _Term("joint_vel",          "WbtJointVel",         29),
    "actions":             _Term("actions",            "WbtPrevActions",      29),
    # UniFP-style 2pt force-tracking preset (exp:g1-wbt-2pt-force). 6D =
    # 2 hands x [Fx, Fy, Fz] in robot root frame. Training-time obs scale
    # is 0.04 (i.e. divide by f_cmd_max=25 N to land in [-1, 1]); the TS
    # class applies this scale internally so live sliders can be in raw N.
    "force_cmd_local":     _Term("force_cmd_local",    "WbtForceCmdLocal",     6),
    # UniFP F^ext-in-actor term (exp:g1-wbt-2pt-force-ext). Same 6D = 2 hands x
    # [Fx, Fy, Fz] in robot root frame, scale 0.04. The shipped 2pt-force keeps
    # F^ext critic-only; this term is only wired into the -force-ext preset.
    "force_ext_local":     _Term("force_ext_local",    "WbtForceExtLocal",     6),
    # Hybrid actor terms (exp:g1-wbt-2pt-hybrid). Each 6D = 2 hands x [x,y,z] in
    # robot root frame. force_meas scale 0.04; integrator/x_tgt scale 1.0. Fed
    # from the mjswan-fork runtime's spring-wall + Jᵀ servo via the extras bus.
    "force_meas_local":    _Term("force_meas_local",   "WbtForceMeasLocal",    6),
    "integrator_state":    _Term("integrator_state",   "WbtIntegratorState",   6),
    "x_tgt_shift":         _Term("x_tgt_shift",         "WbtXTgtShift",         6),
    # C1 unified soft contact-phase label (exp:g1-wbt-3pt-hybrid-global-vel-unified).
    # 2D = 1 per hand, ∈[0,1] (hysteresis 2/1.5 N + 50 ms LPF on ‖F_meas‖), scale 1.0.
    # Fed from the mjswan runtime's spring-wall sim via extras wbt_cphase__{l,r}.
    "contact_phase":       _Term("contact_phase",      "WbtContactPhase",      2),
    # Live VIVE-tracker variants of the 2pt VR terms (same widths as their
    # baked counterparts so the obs vector stays 564-D). Fed over a WebSocket
    # from the vive bridge instead of the baked motion table.
    "vr_2point_pos_live":  _Term("vr_2point_pos_live", "WbtVr2PointPosLive",   6),
    "vr_2point_orn_live":  _Term("vr_2point_orn_live", "WbtVr2PointOrnLive",  12),
    # Live VIVE-tracker variants of the 3pt VR terms. Same widths as the baked
    # 3pt terms so the obs stays 680-D, but note the position term is the
    # TCP-unified one (`_tcp`): its 9D is the 0.242 m FINGERTIP TCP, not the
    # 0.18 m VR lever, and the bridge must be run with --hand-point tcp to match
    # (the widths are identical, so nothing downstream can catch a mismatch).
    # The orn is 3x xyzw QUATERNION (not rot6d) — the plain-3pt rep.
    "vr_3point_pos_tcp_live": _Term("vr_3point_pos_tcp_live", "WbtVr3PointPosTcpLive",  9),
    "vr_3point_orn_live":     _Term("vr_3point_orn_live",     "WbtVr3PointOrnLive",    12),
    # MOTION-COMPLETION live terms (bridge --mode 5pt / dense; user 2026-08-27). Same
    # widths as their baked twins, so the obs stays 736-D / 732-D. The 5pt position term
    # is the TCP-unified one for the same reason the 3pt one is; the two feet carry NO
    # lever (the keypoint IS the ankle_roll_link origin).
    "vr_5point_pos_tcp_live": _Term("vr_5point_pos_tcp_live", "WbtVr5PointPosTcpLive", 15),
    "vr_5point_orn_live":     _Term("vr_5point_orn_live",     "WbtVr5PointOrnLive",    20),
    # 58D = 29 joint targets ++ 29 joint velocities, from the completion instead of the
    # baked clip. NOT interchangeable with `command`: it needs policy_joint_names to
    # remap the wire's joint order, which the baked term never has to do.
    "command_live":     _Term("command_live",     "WbtCommandLive",   58),
}
# fmt: on


@dataclass(frozen=True)
class Preset:
    name: str
    """Inference preset name (matches ``--preset`` in ``batch_headless_eval.sh``)."""
    obs_keys: tuple[str, ...]
    """Obs term keys in policy order (the order the ONNX expects)."""
    history_length: int
    """Per-term history; for the holosoma-*-wbt family this is always 4."""
    vr_family: str
    """One of ``2pt`` / ``3pt`` / ``5pt`` / ``none`` — declares what ``bake_motions``
    needs to compute for each NPZ. Informational: ``bake_motion`` currently computes
    EVERY table unconditionally (one Pinocchio FK pass already visits all the frames),
    so this is documentation + a coverage signal, not a switch."""

    @property
    def per_frame_dim(self) -> int:
        return sum(_TERMS[k].base_dim for k in self.obs_keys)

    @property
    def total_dim(self) -> int:
        return self.per_frame_dim * self.history_length

    def ts_components(self) -> list[dict]:
        """Return the policy_config components list (no params yet — main.py
        injects per-component params like ``default_joint_angles``)."""
        return [{"name": _TERMS[k].ts_name} for k in self.obs_keys]


_HOLOSOMA_WBT_TERMS = (
    "command",
    "motion_ref_ori_b",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
)

_HOLOSOMA_3PT_WBT_TERMS = (
    "command_lower_body",
    "vr_3point_pos",
    "vr_3point_orn",
    "motion_ref_ori_b",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
)

_HOLOSOMA_2PT_WBT_TERMS = (
    "command_lower_body",
    "vr_2point_pos",
    "vr_2point_orn",
    "motion_ref_ori_b",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
)

# Live VIVE-teleop variant of the 2pt preset: IDENTICAL term order/widths/history
# (so the obs is still 564-D and the same normalizer applies), but vr_2point_pos
# /orn are sourced LIVE from the VIVE bridge over a WebSocket instead of the
# baked motion table. Every other term (command_lower_body, motion_ref_ori_b,
# projected_gravity, base_ang_vel, joint_pos, joint_vel, actions) is unchanged —
# command_lower_body / motion_ref_ori_b still come from the injected idle_default
# standing clip; the rest from live sim state.
_HOLOSOMA_2PT_WBT_LIVE_TERMS = (
    "command_lower_body",
    "vr_2point_pos_live",
    "vr_2point_orn_live",
    "motion_ref_ori_b",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
)

# 2pt + UniFP force tracking. Training-side actor obs0..obs9 uses
# alphabetical sort with explicit prefixes (obs3_force_cmd_local). The
# inference-side ObservationConfig (holosoma_2pt_force_wbt) preserves that
# same order; this tuple mirrors it.
_HOLOSOMA_2PT_FORCE_WBT_TERMS = (
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
)

# 2pt + UniFP force tracking with F^ext ADDED to the actor (exp:g1-wbt-2pt-force-ext,
# wandb run fmotwon3). 153D = 147 + 6.
#
# CRITICAL ORDERING (2026-07-12 fix): the holosoma core ObservationManager
# concatenates a group's terms in ALPHABETICALLY-SORTED key order, NOT insertion
# order (managers/observation/manager.py:139 — `sorted_keys = sorted(...)`). The
# force-ext TRAINING config keys are UN-zero-padded obs0..obs10
# (force_tracking.py:1447-1463), so under string sort "obs10_actions" sorts right
# after "obs0_command_lower_body" and BEFORE "obs1_vr_2point_pos". The trained
# fmotwon3 ONNX therefore expects `actions` at slot [24] — verified against
# model_27000.onnx. This tuple MUST reproduce that sorted layout (actions 2nd), or
# every term past slot 0 is misaligned and the robot flails/falls on load. The
# shipped 2pt-force preset is immune (max key obs9, single-digit → sorts
# sequentially); the hybrid presets dodge it by zero-padding obs00..obs12. Order:
#   command_lower_body, actions, vr_2point_pos, vr_2point_orn, force_cmd_local,
#   force_ext_local, motion_ref_ori_b, projected_gravity, base_ang_vel,
#   joint_pos, joint_vel  (= sorted(obs0_.., obs10_actions)).
_HOLOSOMA_2PT_FORCE_EXT_WBT_TERMS = (
    "command_lower_body",  # obs0
    "actions",  # obs10 — sorts to slot 2 (string sort: "obs10" < "obs1")
    "vr_2point_pos",  # obs1
    "vr_2point_orn",  # obs2
    "force_cmd_local",  # obs3
    "force_ext_local",  # obs4
    "motion_ref_ori_b",  # obs5
    "projected_gravity",  # obs6
    "base_ang_vel",  # obs7
    "joint_pos",  # obs8 (dof_pos)
    "joint_vel",  # obs9 (dof_vel)
)

# Live VIVE-teleop variant of the 2pt FORCE preset: the force term tuple with
# ONLY vr_2point_pos/orn swapped for their _live counterparts (fed over the
# WebSocket from the vive bridge instead of the baked motion table). Derived by
# substitution — NOT hand-retyped — so force_cmd_local provably stays in slot 4
# (index 3) and the layout cannot drift from the trained ONNX. The _live terms
# have IDENTICAL widths to the baked ones (6 / 12, see _TERMS), so per-frame
# stays 147 and total stays 147*4 = 588 — byte-for-byte the same obs contract as
# the replay force preset, just a different SOURCE for the two VR terms. Every
# non-VR obs term (command_lower_body, motion_ref_ori_b, force_cmd_local, …)
# still comes from the injected idle_default clip + live sim state + sliders,
# exactly as in the replay force preset.
_HOLOSOMA_2PT_FORCE_WBT_LIVE_TERMS = tuple(
    {"vr_2point_pos": "vr_2point_pos_live", "vr_2point_orn": "vr_2point_orn_live"}.get(k, k)
    for k in _HOLOSOMA_2PT_FORCE_WBT_TERMS
)

# 2pt hybrid position-force controller (exp:g1-wbt-2pt-hybrid; servo-ON run
# 31wei7by AND Layer-A/servo-OFF run 6y8314vq — BOTH share this 165D obs). Order
# is hybrid.py's sorted actor keys obs00..obs12 (zero-padded so the obs manager's
# alphabetical concat reproduces the on-disk TML layout). 165D = 141 + 6 (F^cmd)
# + 6 (F^meas) + 6 (I) + 6 (Δx). F^ext is critic-only in hybrid (NOT in the actor).
_HOLOSOMA_2PT_HYBRID_WBT_TERMS = (
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
)

# 3pt hybrid position-force controller (exp:g1-wbt-3pt-hybrid; servo-ON AND
# Layer-A/servo-OFF share this 168D obs). = the 2pt-hybrid layout with
# vr_2point_{pos,orn} (6+12) swapped for vr_3point_{pos,orn} (9+12), so 165D → 168D.
# Order is hybrid_3pt.py's sorted actor keys obs00..obs12 (zero-padded so the obs
# manager's alphabetical concat reproduces the on-disk TML layout). 168D = 24 + 9 + 12
# + 6 (F^cmd) + 6 (F^meas) + 6 (I) + 6 (Δx) + 6 + 3 + 3 + 29 + 29 + 29. F^ext is
# critic-only in hybrid (NOT in the actor).
_HOLOSOMA_3PT_HYBRID_WBT_TERMS = (
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
)

# 3pt hybrid + global root pos/vel + 6D orientation (exp:g1-wbt-3pt-hybrid-global-vel;
# servo-ON AND Layer-A/servo-OFF share this 183D obs). = the 3pt-hybrid layout with two
# deltas: vr_3point_orn (12D quat) → vr_3point_orn_6d (18D 6D rot rep), and the
# exp:g1-wbt-global-vel root-state trio motion_ref_pos_b + motion_ref_vel_b + base_lin_vel
# added. Order is hybrid_3pt_global_vel.py's sorted actor keys obs00..obs15. 183D = 24 + 9
# + 18 + 3 + 3 + 6 (F^cmd) + 6 (F^meas) + 6 (I) + 6 (Δx) + 6 + 3 + 3 + 3 + 29 + 29 + 29.
_HOLOSOMA_3PT_HYBRID_GLOBAL_VEL_WBT_TERMS = (
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
)

# C1 unified single-policy (exp:g1-wbt-3pt-hybrid-global-vel-unified[-anneal]; the R2
# observe-only and R1 anneal twins share this 185D obs). = the 183D hybrid-global-vel
# layout + contact_phase (2D soft label) right after force_meas_local (training obs07).
# 185D = 183 + 2. At deploy NEITHER twin injects a servo (observe-only by construction /
# α annealed to 0) — the runtime keeps the integrator as a reference generator only.
_HOLOSOMA_3PT_HYBRID_GLOBAL_VEL_UNIFIED_WBT_TERMS = (
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
)

# LOCAL C1 unified single-policy (exp:g1-wbt-3pt-hybrid-unified[-anneal]; the R2
# observe-only and R1 anneal twins share this 170D obs). = the pelvis-local 168D
# 3pt-hybrid layout + contact_phase (2D soft label) right after force_meas_local
# (training obs05). NO root pos/vel trio, QUATERNION vr_3point_orn — root-invariant
# local family. 170D = 168 + 2. At deploy NEITHER twin injects a servo (observe-only
# by construction / α annealed to 0) — the runtime keeps the integrator as a
# reference generator only (main.py's "unified" name override).
#
# TCP-UNIFIED HAND POINT (user 2026-08-03): the position slot is `vr_3point_pos_tcp` —
# both HANDS tracked at wrist_yaw +x 0.242 m (the fingertip TCP the force stack acts on)
# instead of the 0.18 m VR lever, so position and force share ONE point. Same 9D width;
# the head point is unchanged. The distinct term name keeps the pre-2026-08-03 unified
# checkpoints (trained at 0.18) from being baked against the new geometry.
_HOLOSOMA_3PT_HYBRID_UNIFIED_WBT_TERMS = (
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
)

# 5-POINT LOCAL C1 unified single-policy (exp:g1-wbt-5pt-hybrid-unified-anneal, W&B run
# dlcxc59d). 184D = the 170D TCP-unified layout above with the tracking front-end widened
# from 3 to 5 points: vr_3point_pos_tcp(9)+vr_3point_orn(12) -> vr_5point_pos_tcp(15)+
# vr_5point_orn(20), i.e. +14D. Derived from the 170D tuple BY SUBSTITUTION, not
# hand-retyped, so the force block can never drift out of position relative to its twin
# (force_cmd_local must stay slot 3 — that is what the trained ONNX expects).
#
# THE FIVE POINTS: left hand @ the 0.242 m fingertip TCP, right hand @ the same TCP, head
# @ torso_link +0.35 z, then BOTH FEET at the bare `*_ankle_roll_link` origin. Exactly
# exp:g1-wbt-maskedmimic's `five_point` hover mode after its alias table. Feet APPENDED
# (not interleaved) so the first three slots keep their 3pt meaning.
_HOLOSOMA_5PT_HYBRID_UNIFIED_WBT_TERMS = tuple(
    {"vr_3point_pos_tcp": "vr_5point_pos_tcp", "vr_3point_orn": "vr_5point_orn"}.get(k, k)
    for k in _HOLOSOMA_3PT_HYBRID_UNIFIED_WBT_TERMS
)

# DENSE LOCAL C1 unified single-policy (exp:g1-wbt-hybrid-unified-anneal, W&B run
# 7r54008a). 183D = NO keypoints at all: `command` (58D = motion_command = all 29 joint
# targets + velocities, which already subsumes the 24D command_lower_body slice) + the
# SAME 26D force block the keypoint unified layouts carry, in the same relative position
# (right after the tracking term, as hybrid.py places it). Order is the training config's
# sorted actor keys obs00..obs11.
#
# ⚠️ 183D/732D is ALSO g1-29dof-holosoma-3pt-hybrid-global-vel-wbt's width. Arithmetic
# coincidence, not a shared layout: the term names differ completely, so nothing can load
# across them by accident — but the obs-DIM check cannot tell them apart, which is why
# main.py carries `_assert_checkpoint_tracking_family` (dense vs keypoints, keyed on the
# checkpoint's own obs term names).
_HOLOSOMA_DENSE_HYBRID_UNIFIED_WBT_TERMS = (
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
)

# LIVE VIVE-teleop variant of the 3pt hybrid UNIFIED preset
# (exp:g1-wbt-3pt-hybrid-unified-anneal, run wda3h4ll — the checkpoint
# `MODE=3pt-unified-anneal` drives on the real G1). 170D/frame x 4 = 680D: the
# _HOLOSOMA_3PT_HYBRID_UNIFIED_WBT_TERMS tuple above with ONLY the two VR slots
# swapped for their live twins, BY SUBSTITUTION rather than retyped, so the force
# block provably keeps its slot indices (the trained ONNX depends on that).
#
# There is deliberately no REPLAY twin of this preset: `bake_motions` only stores
# the 0.18 m-lever `vr3_pos_local` table, so a baked 3pt-unified preset would feed
# the policy a target 6.7 cm from the TCP it trained on at the same 9D width.
# Live-only until the bake grows a TCP table; the order is pinned against the
# upstream ObservationConfig by
# tests/test_presets.py::test_live_3pt_unified_preset_matches_upstream_actor_obs.
#
# Only the two VR terms are live; everything else is identical to any other
# hybrid bundle — command_lower_body / motion_ref_ori_b from the injected
# idle_default standing clip, force_cmd_local from the sliders, force_meas_local
# / integrator_state / x_tgt_shift from the mjswan-fork runtime's spring wall +
# Jᵀ servo bus, contact_phase recomputed from that same F^meas.
_HOLOSOMA_3PT_HYBRID_UNIFIED_WBT_LIVE_TERMS = tuple(
    {"vr_3point_pos_tcp": "vr_3point_pos_tcp_live", "vr_3point_orn": "vr_3point_orn_live"}.get(k, k)
    for k in _HOLOSOMA_3PT_HYBRID_UNIFIED_WBT_TERMS
)

# LIVE motion-completion variants of the 5pt (184D) and DENSE (183D) unified anneal
# presets (user 2026-08-27). The rig has THREE trackers; the bridge puts them through
# Motion_Completion and publishes either the five keypoints or the 29 joint targets, so
# these are the SAME layouts with ONLY the tracking slot(s) swapped for their live twins
# — again BY SUBSTITUTION off the baked tuples, so the 26D force block provably keeps its
# slot indices (the trained ONNX depends on that and nothing else would catch a shift).
#
# ⚠️ Both carry a ~667 ms COMMAND LAG. It is a property of the MC checkpoint (dense
# conditioning over a 30-frame window, 20-frame stride), not of the browser or the host:
# a window cannot run until its last cond frame exists, and gap-free playback needs at
# least stride-1 = 19 frames of lag. Compute is ~45 ms/window (~7% duty), measured.
_HOLOSOMA_5PT_HYBRID_UNIFIED_WBT_LIVE_TERMS = tuple(
    {"vr_5point_pos_tcp": "vr_5point_pos_tcp_live", "vr_5point_orn": "vr_5point_orn_live"}.get(k, k)
    for k in _HOLOSOMA_5PT_HYBRID_UNIFIED_WBT_TERMS
)

_HOLOSOMA_DENSE_HYBRID_UNIFIED_WBT_LIVE_TERMS = tuple(
    {"command": "command_live"}.get(k, k)
    for k in _HOLOSOMA_DENSE_HYBRID_UNIFIED_WBT_TERMS
)

# LIVE motion-completion variant of the BASE whole-body tracker (exp:g1-wbt, W&B run
# ruo0o6fr). 157D x 4 = 628D — the SAME width as its replay twin, because this is the
# one-term substitution `command` -> `command_live` and nothing else.
#
# WHY THE BASE TRACKER IS TELEOPERABLE AT ALL: its whole tracking front-end is that 58D
# `command` term (29 joint targets ++ 29 velocities), which is exactly what the bridge's
# `--mode dense` wire carries. So the completion drives it through the same code path as
# the dense-unified policy — the bridge does not even need a new mode.
#
# ⚠️ `g1-29dof-holosoma-wbt` AND `…-dense-hybrid-unified-anneal-wbt` ARE THE SAME TRACKING
# FAMILY, and the relationship is exact, not approximate: the dense tuple is this tuple
# with the 26D force block (force_cmd_local + force_meas_local + contact_phase +
# integrator_state + x_tgt_shift) INSERTED AFTER `command`, and nothing else — 157 + 26 =
# 183. Set-wise dense ⊃ wbt, and wbt's own term ORDER survives inside dense unchanged.
# Both checkpoints even spell the front-end identically in their own metadata
# (`obs0_motion_command` / `obs00_motion_command`), and both train on the same
# `motion_dir=/data/dataset/bones_seed_g1_holosoma_f32`. So do NOT read anything below as
# a wbt-specific handicap: the ONLY thing dense adds is force.
#
# WHAT THE OPERATOR DOES *NOT* GET, structural and SHARED WITH 5pt/dense: none of the
# three motion-completion presets carries `motion_ref_pos_b`, so none of them is ever told
# where the reference ROOT is. Heading and world position are pinned to the injected clip;
# the trackers steer POSTURE only. (`x_tgt_shift` in the dense/5pt tuples is NOT a root
# channel — it is the force servo's per-hand Δx from the mjswan spring wall.) The replay
# numbers say the same thing directly: 3.5 cm torso-frame MPJPE against 12 cm of root
# error. It is the policy family's limit, not the teleop's.
#
# The one REAL wbt-vs-dense difference at deploy: this 628D obs has no force slot at all,
# so every force env var is inert here, while dense/5pt genuinely consume F^cmd.
_HOLOSOMA_WBT_LIVE_TERMS = tuple(
    {"command": "command_live"}.get(k, k)
    for k in _HOLOSOMA_WBT_TERMS
)

_HOLOSOMA_2PT_GLOBAL_WBT_TERMS = (
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
)

_HOLOSOMA_3PT_GLOBAL_WBT_TERMS = (
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
)

_HOLOSOMA_GLOBAL_WBT_TERMS = (
    "command",
    "motion_ref_ori_b",
    "motion_ref_pos_b",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
)

_HOLOSOMA_GLOBAL_VEL_WBT_TERMS = (
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
)


PRESETS: dict[str, Preset] = {
    "g1-29dof-holosoma-wbt": Preset(
        name="g1-29dof-holosoma-wbt", obs_keys=_HOLOSOMA_WBT_TERMS, history_length=4, vr_family="none"
    ),
    "g1-29dof-holosoma-2pt-wbt": Preset(
        name="g1-29dof-holosoma-2pt-wbt", obs_keys=_HOLOSOMA_2PT_WBT_TERMS, history_length=4, vr_family="2pt"
    ),
    "g1-29dof-holosoma-2pt-wbt-live": Preset(
        name="g1-29dof-holosoma-2pt-wbt-live", obs_keys=_HOLOSOMA_2PT_WBT_LIVE_TERMS, history_length=4, vr_family="2pt"
    ),
    "g1-29dof-holosoma-2pt-force-wbt": Preset(
        name="g1-29dof-holosoma-2pt-force-wbt",
        obs_keys=_HOLOSOMA_2PT_FORCE_WBT_TERMS,
        history_length=4,
        vr_family="2pt",
    ),
    # UniFP force-tracking with F^ext in the actor (exp:g1-wbt-2pt-force-ext,
    # run fmotwon3). 153D/frame x 4 = 612D. Same critic as 2pt-force; only the
    # actor gains force_ext_local. vr_family "2pt" (baked 2pt FK + force_hand_ref).
    "g1-29dof-holosoma-2pt-force-ext-wbt": Preset(
        name="g1-29dof-holosoma-2pt-force-ext-wbt",
        obs_keys=_HOLOSOMA_2PT_FORCE_EXT_WBT_TERMS,
        history_length=4,
        vr_family="2pt",
    ),
    # Hybrid position-force controller (exp:g1-wbt-2pt-hybrid). 165D/frame x 4 =
    # 660D. servo-ON (run 31wei7by) and Layer-A/servo-OFF (run 6y8314vq) share the
    # SAME obs; the mjswan-fork runtime's servo_enabled (from experiment_config,
    # overridden false for the -layerA preset via main.py) decides whether the Jᵀ
    # servo actually injects Δq. vr_family "2pt" (baked 2pt FK + force_hand_ref).
    "g1-29dof-holosoma-2pt-hybrid-wbt": Preset(
        name="g1-29dof-holosoma-2pt-hybrid-wbt",
        obs_keys=_HOLOSOMA_2PT_HYBRID_WBT_TERMS,
        history_length=4,
        vr_family="2pt",
    ),
    "g1-29dof-holosoma-2pt-hybrid-layerA-wbt": Preset(
        name="g1-29dof-holosoma-2pt-hybrid-layerA-wbt",
        obs_keys=_HOLOSOMA_2PT_HYBRID_WBT_TERMS,
        history_length=4,
        vr_family="2pt",
    ),
    # 3pt hybrid position-force controller (exp:g1-wbt-3pt-hybrid). 168D/frame x 4 =
    # 672D. servo-ON and Layer-A/servo-OFF share the SAME obs; main.py's hybrid dispatch
    # keys off force_meas_local in obs_keys (present here) and forces the servo OFF when
    # "layera" is in the preset name. vr_family "3pt" (baked 3pt FK + force_hand_ref).
    "g1-29dof-holosoma-3pt-hybrid-wbt": Preset(
        name="g1-29dof-holosoma-3pt-hybrid-wbt",
        obs_keys=_HOLOSOMA_3PT_HYBRID_WBT_TERMS,
        history_length=4,
        vr_family="3pt",
    ),
    "g1-29dof-holosoma-3pt-hybrid-layerA-wbt": Preset(
        name="g1-29dof-holosoma-3pt-hybrid-layerA-wbt",
        obs_keys=_HOLOSOMA_3PT_HYBRID_WBT_TERMS,
        history_length=4,
        vr_family="3pt",
    ),
    # 3pt hybrid + global root pos/vel + 6D orientation (exp:g1-wbt-3pt-hybrid-global-vel).
    # 183D/frame x 4 = 732D. servo-ON and Layer-A/servo-OFF share the SAME obs; main.py's
    # hybrid dispatch keys off force_meas_local in obs_keys (present) and forces the servo
    # OFF when "layera" is in the preset name. vr_family "3pt" (baked 3pt FK, incl. the new
    # vr3_orn6d_local 18D table + force_hand_ref).
    "g1-29dof-holosoma-3pt-hybrid-global-vel-wbt": Preset(
        name="g1-29dof-holosoma-3pt-hybrid-global-vel-wbt",
        obs_keys=_HOLOSOMA_3PT_HYBRID_GLOBAL_VEL_WBT_TERMS,
        history_length=4,
        vr_family="3pt",
    ),
    "g1-29dof-holosoma-3pt-hybrid-global-vel-layerA-wbt": Preset(
        name="g1-29dof-holosoma-3pt-hybrid-global-vel-layerA-wbt",
        obs_keys=_HOLOSOMA_3PT_HYBRID_GLOBAL_VEL_WBT_TERMS,
        history_length=4,
        vr_family="3pt",
    ),
    # C1 unified single-policy (exp:g1-wbt-3pt-hybrid-global-vel-unified[-anneal]).
    # 185D/frame x 4 = 740D. Both training twins share the SAME obs; at deploy neither
    # injects a servo — the runtime's integrator runs observe-only (reference generator).
    "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt": Preset(
        name="g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt",
        obs_keys=_HOLOSOMA_3PT_HYBRID_GLOBAL_VEL_UNIFIED_WBT_TERMS,
        history_length=4,
        vr_family="3pt",
    ),
    "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-anneal-wbt": Preset(
        name="g1-29dof-holosoma-3pt-hybrid-global-vel-unified-anneal-wbt",
        obs_keys=_HOLOSOMA_3PT_HYBRID_GLOBAL_VEL_UNIFIED_WBT_TERMS,
        history_length=4,
        vr_family="3pt",
    ),
    # LOCAL C1 unified single-policy (exp:g1-wbt-3pt-hybrid-unified[-anneal]).
    # 170D/frame x 4 = 680D. Both training twins share the SAME obs; at deploy neither
    # injects a servo — the runtime's integrator runs observe-only (reference generator).
    "g1-29dof-holosoma-3pt-hybrid-unified-wbt": Preset(
        name="g1-29dof-holosoma-3pt-hybrid-unified-wbt",
        obs_keys=_HOLOSOMA_3PT_HYBRID_UNIFIED_WBT_TERMS,
        history_length=4,
        vr_family="3pt",
    ),
    "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt": Preset(
        name="g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt",
        obs_keys=_HOLOSOMA_3PT_HYBRID_UNIFIED_WBT_TERMS,
        history_length=4,
        vr_family="3pt",
    ),
    # 5-POINT LOCAL C1 unified (exp:g1-wbt-5pt-hybrid-unified-anneal, run dlcxc59d).
    # 184D/frame x 4 = 736D. Same force stack + observe-only deploy contract as its 3pt
    # twin ("unified" in the name is what triggers main.py's observe-only override);
    # only the tracking front-end differs (5 points instead of 3).
    #
    # vr_family "5pt": the bake needs the two EXTRA foot FK columns (vr5_pos_tcp_local /
    # vr5_orn_local) on top of everything the "3pt" family produces.
    "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt": Preset(
        name="g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt",
        obs_keys=_HOLOSOMA_5PT_HYBRID_UNIFIED_WBT_TERMS,
        history_length=4,
        vr_family="5pt",
    ),
    # DENSE LOCAL C1 unified (exp:g1-wbt-hybrid-unified-anneal, run 7r54008a). 183D/frame
    # x 4 = 732D. No keypoints — the policy tracks all 29 joint targets directly — so it
    # draws NO tracking-point spheres (there is no keypoint to mark); the white TCP force
    # marker + the 3-arrow force viz still apply, since those come from the hybrid gate.
    #
    # vr_family "3pt" (NOT "none"): the browser still needs the baked 3pt/2pt tables for
    # force_hand_ref_local (the live per-hand force-error metric) and the reference ghost,
    # and "none" reads as "this preset needs no FK", which is false. The dense ACTOR obs
    # itself carries no vr_* term — that is enforced by the term tuple, not by this field.
    "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt": Preset(
        name="g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt",
        obs_keys=_HOLOSOMA_DENSE_HYBRID_UNIFIED_WBT_TERMS,
        history_length=4,
        vr_family="3pt",
    ),
    # Live VIVE-teleop variant of the force preset (parallel to the plain
    # 2pt-wbt-live). vr_family MUST stay "2pt": the baked 2pt FK tables still
    # feed the non-VR obs terms (command_lower_body, motion_ref_ori_b) and the
    # idle force_hand_ref metric; only vr_2point_pos/orn are sourced live.
    "g1-29dof-holosoma-2pt-force-wbt-live": Preset(
        name="g1-29dof-holosoma-2pt-force-wbt-live",
        obs_keys=_HOLOSOMA_2PT_FORCE_WBT_LIVE_TERMS,
        history_length=4,
        vr_family="2pt",
    ),
    "g1-29dof-holosoma-3pt-wbt": Preset(
        name="g1-29dof-holosoma-3pt-wbt", obs_keys=_HOLOSOMA_3PT_WBT_TERMS, history_length=4, vr_family="3pt"
    ),
    # LIVE VIVE-teleop for the 3pt hybrid UNIFIED anneal policy (run wda3h4ll) —
    # the sim twin of the rig's `MODE=3pt-unified-anneal`. vr_family MUST stay
    # "3pt": the baked 3pt FK tables still feed the non-VR obs terms
    # (command_lower_body / motion_ref_ori_b) and the idle force_hand_ref metric;
    # only the two VR terms are sourced live. The name deliberately carries
    # `-anneal`: the R1 and R2 twins share this obs byte-for-byte, but the R1
    # anneal run is the only one that exists on this rig, so a name that says so
    # keeps the launcher and the deploy MODE spelled the same way.
    "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt-live": Preset(
        name="g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt-live",
        obs_keys=_HOLOSOMA_3PT_HYBRID_UNIFIED_WBT_LIVE_TERMS,
        history_length=4,
        vr_family="3pt",
    ),
    # MOTION-COMPLETION live teleop (bridge --mode 5pt / dense). vr_family stays "3pt":
    # it declares what the RIG streams (three trackers), and it is what bake_motions keys
    # off. The extra points are produced by the completion, not by another tracker.
    "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt-live": Preset(
        name="g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt-live",
        obs_keys=_HOLOSOMA_5PT_HYBRID_UNIFIED_WBT_LIVE_TERMS,
        history_length=4,
        vr_family="3pt",
    ),
    "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt-live": Preset(
        name="g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt-live",
        obs_keys=_HOLOSOMA_DENSE_HYBRID_UNIFIED_WBT_LIVE_TERMS,
        history_length=4,
        vr_family="3pt",
    ),
    # LIVE teleop of the BASE tracker (exp:g1-wbt, run ruo0o6fr) — the sim twin of the
    # rig's `MODE=wbt`. vr_family stays "none", matching its replay twin: this preset has
    # no VR terms at all, so bake_motions has no FK tables to build. The three trackers
    # reach it only through the completion's 29 joint targets on the dense wire.
    "g1-29dof-holosoma-wbt-live": Preset(
        name="g1-29dof-holosoma-wbt-live",
        obs_keys=_HOLOSOMA_WBT_LIVE_TERMS,
        history_length=4,
        vr_family="none",
    ),
    "g1-29dof-holosoma-2pt-global-wbt": Preset(
        name="g1-29dof-holosoma-2pt-global-wbt",
        obs_keys=_HOLOSOMA_2PT_GLOBAL_WBT_TERMS,
        history_length=4,
        vr_family="2pt",
    ),
    "g1-29dof-holosoma-3pt-global-wbt": Preset(
        name="g1-29dof-holosoma-3pt-global-wbt",
        obs_keys=_HOLOSOMA_3PT_GLOBAL_WBT_TERMS,
        history_length=4,
        vr_family="3pt",
    ),
    "g1-29dof-holosoma-global-wbt": Preset(
        name="g1-29dof-holosoma-global-wbt", obs_keys=_HOLOSOMA_GLOBAL_WBT_TERMS, history_length=4, vr_family="none"
    ),
    "g1-29dof-holosoma-global-vel-wbt": Preset(
        name="g1-29dof-holosoma-global-vel-wbt",
        obs_keys=_HOLOSOMA_GLOBAL_VEL_WBT_TERMS,
        history_length=4,
        vr_family="none",
    ),
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        supported = ", ".join(sorted(PRESETS))
        raise ValueError(
            f"Unsupported preset {name!r}. wbt-realtime-eval supports the "
            f"holosoma-*-wbt family: {supported}. Other inference presets "
            f"(terrain, chip, maskedmimic, …) need additional browser-side "
            f"inputs (height-scan rays, compliance vectors, masked-mimic "
            f"targets) that this package does not bake yet."
        )
    return PRESETS[name]


def term_ts_name(key: str) -> str:
    """Return the TS class name for a given obs term key (used by main.py)."""
    return _TERMS[key].ts_name


def term_base_dim(key: str) -> int:
    return _TERMS[key].base_dim
