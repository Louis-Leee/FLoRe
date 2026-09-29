"""TCP precision rewards + gripper-mass DR for `exp:g1-wbt-3pt-hybrid-unified[-anneal]`.

Plan: docs/2026-08-05-tcp-precision-translation-orientation-plan.md.

WHY THIS MODULE EXISTS — two structural pricing gaps, neither fixable by retuning a
weight (plan §1, all numbers read from pfl96opy's ONNX `experiment_config` metadata):

  Gap A (translation). NONE of the 15 reward terms takes the 0.242 m TCP as input.
    * `force_track_pos_eq2` → `hybrid:force_track_eq2_servo` builds BOTH sides of its
      residual with `_force_ref_pos_w` / `_force_cur_pos_w`, which rotate the FK offset
      `d` by the SAME live `body_quat_w`. So `R·d` cancels algebraically:
          cur_w − target_w = (wrist_pos_w − wrist_pos_ref_w) − Δx
      Eq.2 constrains the wrist_yaw ORIGIN; `d` is mathematically absent (that
      cancellation is deliberate — 2026-05-28 frame alignment, so yaw drift cannot
      pollute force tracking — hence we ADD a term rather than "fix" Eq.2).
    * `motion_relative_body_position_error_exp_no_wrist` explicitly excludes both
      `*_wrist_yaw_link`.
    So the TCP sits in the gap between the two. `obs01_vr_3point_pos_tcp` IS 0.242 m
    geometry, but that is OBSERVATION: the policy sees the TCP target and is not paid
    for hitting it. The 2026-08-03 TCP-unification moved the obs and not the reward.

  Gap B (orientation). TCP pose == wrist_yaw pose (rigid), and a wrist attitude error θ
    levers into a TCP POSITION error of AT MOST 2·d·sin(θ/2) — 4.8 cm at θ=0.2 rad.
    "At most" is load-bearing: the true chord scales by sin∠(rotation axis, d), so the
    bound is attained only for axes PERPENDICULAR to the 0.242 m offset (pitch, yaw), and
    a pure ROLL about the offset axis moves the TCP by EXACTLY ZERO. Roll error is
    therefore invisible to `tcp_position_error_exp` and is priced solely by
    `tcp_orientation_error_exp` — which is a reason to keep both terms, and a caveat when
    reading T1: if roll dominates, the position term can look healthy while the hand is
    still mis-oriented.
    That lever error reaches neither Eq.2 (cancelled) nor the 14-body position term
    (wrist excluded), so only the orientation term can price it — and it is diluted twice:
    `mean` over 14 bodies, at σ=0.4. A wrist 0.2 rad off costs 0.035 total reward (vs
    feet_slip −2.0, limits_dof_pos −10.0).

THE FIX, in one sentence: `tcp_position_error_exp` rotates the ref-side offset by the
REF quaternion and the cur-side offset by the LIVE quaternion, so `R·d` does NOT cancel
and the orientation lever is priced as the position error it physically is;
`tcp_orientation_error_exp` re-prices wrist attitude undiluted.

Both are ADDITIVE — Eq.2, the 14-body position term, and the 14-body orientation term
are all left byte-identical (that last one still governs torso/shoulder/elbow/leg pose;
wrist attitude being priced twice is intentional double supervision, matching how
upperbody.py layers a tight TCP term over the loose 14-body one).

OBS IS UNTOUCHED (170D). Nothing here reads or writes the observation, so pfl96opy
warm-starts into these presets with no network surgery.

ALSO HERE (plan §5, phase 2): the gripper-mass domain randomization config. It lives in
this module because it is inseparable from `hybrid.HybridContactInjector`'s distal-mass
sync — randomizing `wrist_yaw` mass without updating m_distal desynchronizes the
simulated F/T sensor from the dynamics by up to 2.04 N, the same order as
`contact_phase_theta_on = 2.0 N`. See `g1_29dof_wbt_randomization_hand_mass`.

PURE READERS: every reward function here only reads `motion_command` state, the robot
data views, and the `_wbt_*` buffers `HybridContactInjector` owns. Nothing advances the
force pulse or mutates env state (single-writer rule).
"""

from __future__ import annotations

import os
from dataclasses import replace as _replace
from typing import Any

import torch

from holosoma.config_types.randomization import RandomizationTermCfg
from holosoma.config_types.reward import RewardTermCfg
from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization as _base_randomization

from wbt_training.config_values.falcon_force import _xyzw_to_wxyz
from wbt_training.config_values.force_tracking import (
    _ensure_force_isaac_indices,
    _ensure_force_motion_indices,
    _quat_apply_wxyz,
)
from wbt_training.config_values.hybrid import TCP_OFFSET_X
from wbt_training.config_values.hybrid_unified import (
    _UNIFIED_ANNEAL_CFG,
    _UNIFIED_CFG,
    _unified_command,
    _unified_reward,
    _unified_termination,
    g1_29dof_wbt_unified_anneal_curriculum,
    g1_29dof_wbt_unified_anneal_reward,
    g1_29dof_wbt_unified_reward,
)

# The two tracked hands. Deliberately a module constant rather than a cfg field: these
# rewards are meaningless for any other body (the 0.242 m lever is gripper geometry), and
# keeping them out of the params dict keeps the terms' W&B config diff-clean.
_HAND_BODY_NAMES: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")

# Default reward kernel widths. Chosen from the σ-sensitivity tables in plan §2.1/§2.2,
# not tuned: σ_pos=0.10 m keeps real gradient across 1–5 cm (5→2 cm buys Δr=0.182, ×w=4.0
# = 0.73 total reward — 20× what wrist attitude is worth today) while not saturating to 0
# by 20 cm the way σ=0.06 does; σ_orn=0.25 rad gives 0.852 at 5.7° (≈2.4 cm of TCP error)
# so there is headroom to earn, where σ=0.15 would flatten early-training arm swings.
_TCP_POS_SIGMA: float = 0.10  # m
_TCP_ORN_SIGMA: float = 0.25  # rad


def _tcp_offsets_b(env: Any, tcp_offset_x: float) -> torch.Tensor:
    """Cache the per-hand TCP offset tensor `(h, 3)` = `(tcp_offset_x, 0, 0)` on env.

    Cache key includes the offset so a run that overrides `tcp_offset_x` cannot be served
    a stale tensor (the same class of bug `observation._resolve_offsets` guards against:
    two terms sharing body names but differing in offsets).
    """
    cached = getattr(env, "_wbt_tcp_precision_offsets", None)
    if isinstance(cached, tuple) and len(cached) == 2:
        key, off = cached
        if isinstance(off, torch.Tensor) and key == float(tcp_offset_x):
            return off
    off = torch.tensor([(float(tcp_offset_x), 0.0, 0.0)] * len(_HAND_BODY_NAMES), dtype=torch.float, device=env.device)
    env._wbt_tcp_precision_offsets = (float(tcp_offset_x), off)
    return off


def _tcp_cur_pos_w(env: Any, off: torch.Tensor) -> torch.Tensor:
    """LIVE world TCP: `wrist_pos_w + R(wrist_quat_w) @ d`, shape `(n, h, 3)`.

    `_robot.data.body_quat_w` is WXYZ (IsaacSim native) — same source and convention as
    `force_tracking._force_cur_pos_w` and `HybridContactInjector._current_tcp_w`, so this
    is the SAME physical point the spring wall / F^ext / F^meas / Jᵀ servo act on.
    """
    body_ids = _ensure_force_isaac_indices(env, _HAND_BODY_NAMES)
    wrist_pos_w = env.simulator._robot.data.body_pos_w[:, body_ids, :]
    wrist_quat_wxyz = env.simulator._robot.data.body_quat_w[:, body_ids, :]
    off_b = off.unsqueeze(0).expand(env.num_envs, len(_HAND_BODY_NAMES), -1)
    return wrist_pos_w + _quat_apply_wxyz(wrist_quat_wxyz, off_b)


def _tcp_ref_pos_w(env: Any, off: torch.Tensor) -> torch.Tensor:
    """REFERENCE world TCP: `body_pos_relative_w + R(body_quat_relative_w) @ d`, `(n, h, 3)`.

    ⚠️ THE WHOLE POINT OF THIS MODULE IS THE QUATERNION ON THIS LINE. Eq.2 rotates the
    ref-side offset by the LIVE wrist quat, which makes `R·d` cancel in `cur − ref`; here
    the ref side uses the MOTION-REFERENCE attitude, so it does not cancel and a wrist
    orientation error surfaces as the TCP position error it physically is (plan §1.3:
    1.2 cm at 0.05 rad → 9.6 cm at 0.4 rad). Same asymmetry `upperbody._tcp_ref_pos_w_relative`
    uses, and the same reason.

    Anchored on `body_pos_relative_w` (yaw-relocated motion ref) — identical anchor to
    `motion_relative_body_position_error_exp` and Eq.2, so no third reference frame is
    introduced and free base-yaw cannot inject phantom error. Motion-command quats are
    XYZW, so convert before `_quat_apply_wxyz` (WXYZ).
    """
    mc = env.command_manager.get_state("motion_command")
    fm_idx = _ensure_force_motion_indices(env, _HAND_BODY_NAMES)
    wrist_pos_ref_w = mc.body_pos_relative_w[:, fm_idx, :]
    wrist_quat_ref_wxyz = _xyzw_to_wxyz(mc.body_quat_relative_w[:, fm_idx, :])
    off_b = off.unsqueeze(0).expand(env.num_envs, len(_HAND_BODY_NAMES), -1)
    return wrist_pos_ref_w + _quat_apply_wxyz(wrist_quat_ref_wxyz, off_b)


def tcp_position_error_exp(
    env: Any, *, sigma: float = _TCP_POS_SIGMA, tcp_offset_x: float = TCP_OFFSET_X, servo_shifted: bool = True
) -> torch.Tensor:
    """0.242 m TCP position-tracking reward — the term Eq.2's cancellation leaves out.

        target[n,h] = ref_tcp[n,h] + Δx[n,h]                      [m]
        err[n,h]    = Σ_xyz (cur_tcp[n,h] − target[n,h])²         [m²]
        r[n]        = exp(−mean_h err[n,h] / σ²)

    Shape `(num_envs,)`, range (0, 1]. `servo_shifted=False` drops Δx (pure kinematic
    target) — used by the unit tests and available for ablation.

    HOW THIS AVOIDS FIGHTING Eq.2, WITHOUT A CONTACT GATE. In contact the integrator
    deliberately pushes the position target INTO the wall — that overshoot IS the UniFP
    force mechanism — so a TCP sitting away from the KINEMATIC ref is then correct, and a
    term chasing pure `ref` would pull against it. So we chase the SAME shifted setpoint
    Eq.2 chases, `ref + Δx` (`_wbt_delta_x`) — the two terms then agree on where the hand
    belongs in EVERY regime, and this one adds only the 0.242 m lever + reference attitude
    that Eq.2 structurally cannot see. Identical in form to `upperbody.py`'s
    `tcp_track_pos_exp_relative` (same Eq.2-shifted TCP target, same reference-quat
    asymmetry); the only difference is the servo Δx form of the setpoint that the unified
    presets use.

    ⚠️ Δx IS NOT ZERO IN FREE SPACE — an earlier revision of this docstring said it was,
    and that was wrong (review 2026-08-05). `integrator_observe_step` gates only the
    INTEGRAL: `new_integ = gate·clip(…)`, but `task_wrench = S⊙(F_cmd + I)` and
    `Δx = task_wrench/K`, so with the hard `_wbt_contact_gate` at 0 the integral is 0 while
    Δx is still `F_cmd/K` — a genuine free-space PRE-LOAD:

        k_env=100 → |F_cmd| cap  8.0 N → |Δx| =  4.00 cm
        k_env=400 → |F_cmd| cap 25.0 N → |Δx| = 12.50 cm

    and F_cmd is on for ~84% of an episode (ramp+hold+ramp vs idle). This is INTENTIONAL
    UniFP behaviour, not a bug — Eq.2 pre-extends the arm toward the wall by F/K so the
    contact is met with the right force — and because we read the same buffer, the two
    terms stay aligned through it. But it means the honest statement is "both terms carry
    the same pre-load", NOT "the pre-load is absent in free space". Consequence for T1: the
    TCP is deliberately commanded up to 12.5 cm PAST the kinematic reference while
    approaching, so `tcp_position_error_exp` raw is the error against that shifted target —
    do not read it as raw distance-to-clip. `servo_shifted=False` is the ablation that
    measures distance-to-clip instead.

    ⚠️ WHY NOT A `1 − contact_phase` GATE (the first version of this term, 2026-08-05):
    gating the per-hand error by the free-space fraction makes the reward
    `exp(−Σ(g·err) / (Σg·σ²))`, which as BOTH hands approach contact drives the numerator
    to 0 while the denominator hits its clamp — so r → exp(0) = 1.0 and the term pays its
    FULL weight no matter how badly the TCP is tracking. That is a live reward hack worth
    up to +2.5 (at 10 cm error) and +3.9 (at 20 cm) per step against a `feet_slip` of
    −2.0, and it is strongest exactly in early training when errors are largest: press the
    wall, stop tracking, collect full marks. The Δx-shifted target removes the tug-of-war
    the gate existed to solve without introducing any contact_phase dependence — so there
    is neither a hack nor contact shyness, and no reset-warmup special case is needed
    (a teleport transient is now priced as the tracking error it actually is).

    PURE READER.
    """
    off = _tcp_offsets_b(env, tcp_offset_x)
    target_w = _tcp_ref_pos_w(env, off)
    if servo_shifted:
        # Servo/integrator setpoint shift (m), world frame, (n,h,3) — the SAME buffer
        # `force_track_eq2_servo` adds, so the two terms share one notion of "where the
        # hand should be". NOT zero in free space: it carries the F_cmd/K pre-load (up to
        # 12.5 cm) whenever a force is commanded — see the docstring. The isinstance guard
        # keeps MagicMock/no-injector unit-test envs on the plain path.
        dx = getattr(env, "_wbt_delta_x", None)
        if isinstance(dx, torch.Tensor):
            target_w = target_w + dx
    err = (_tcp_cur_pos_w(env, off) - target_w).square().sum(dim=-1)  # (n,h) m²
    return torch.exp(-err.mean(dim=-1) / sigma**2)


def tcp_orientation_error_exp(env: Any, *, sigma: float = _TCP_ORN_SIGMA) -> torch.Tensor:
    """Dedicated wrist/TCP attitude reward (TCP pose == wrist_yaw pose, rigid offset).

        err[n,h] = quat_error_magnitude(body_quat_relative_w[n,h], robot_body_quat_w[n,h])²
        r[n]     = exp(−mean_{h ∈ {L,R}} err[n,h] / σ²)

    Mathematically the same form as upstream `motion_relative_body_orientation_error_exp`
    with exactly two deltas, which together are the entire point (plan §1.3):
      1. mean over 2 hands, not 14 bodies  → dilution 1/14 → 1/2, i.e. a 7× de-dilution
      2. σ = 0.25 rad, not 0.4             → tightens by (0.4/0.25)² = 2.56×
    ⇒ per-wrist attitude gradient **17.9×** the status quo (7 × 2.56).

    The per-body gradient of `exp(−mean_h err_h/σ²)` scales as `w/(N·σ²)`, so this is
    `2.0/(2·0.25²) = 16.0` against the old `2.0/(14·0.4²) = 0.893`. (An earlier revision of
    this docstring said "~36×", which is `14 × 2.56` — that would only hold if this term
    meaned over ONE body. It means over TWO, so the correct factor is 7 × 2.56, exactly 2×
    smaller. Verified by finite difference. Corrected 2026-08-05 after review.)

    The 14-body term is KEPT (it governs torso/shoulder/elbow/leg attitude; dropping it
    would collapse whole-body pose). Wrist attitude therefore appears in two terms — the
    dedicated one supplies gradient, the whole-body one keeps global consistency. Same
    loose-global + tight-TCP layering `upperbody.py` already ships.

    This term is the ONLY thing pricing a ROLL error about the 0.242 m offset axis: such a
    rotation moves the TCP position by exactly zero, so `tcp_position_error_exp` cannot see
    it at all (see Gap B in the module docstring). Another reason not to drop either term.

    Both quats are XYZW (`quat_error_magnitude` default `w_last=True`). PURE READER.
    """
    from holosoma.utils.rotations import quat_error_magnitude

    mc = env.command_manager.get_state("motion_command")
    fm_idx = _ensure_force_motion_indices(env, _HAND_BODY_NAMES)
    err = quat_error_magnitude(mc.body_quat_relative_w[:, fm_idx, :], mc.robot_body_quat_w[:, fm_idx, :]) ** 2
    return torch.exp(-err.mean(dim=-1) / sigma**2)


# ---------------------------------------------------------------------------
# Reward cfgs: the unified R2/R1 rewards + the two TCP precision terms.
#
# `w=4.0` on position (above Eq.2's 3.0 — TCP position accuracy is the objective of this
# change) and `w=2.0` on orientation (level with the 14-body attitude term). Every
# inherited term keeps its weight, so this is purely additive: 15 terms → 17.
# ---------------------------------------------------------------------------
_TCP_PRECISION_TERMS = {
    "tcp_position_error_exp": RewardTermCfg(
        func="wbt_training.config_values.tcp_precision:tcp_position_error_exp",
        weight=4.0,
        params={"sigma": _TCP_POS_SIGMA, "tcp_offset_x": TCP_OFFSET_X, "servo_shifted": True},
    ),
    "tcp_orientation_error_exp": RewardTermCfg(
        func="wbt_training.config_values.tcp_precision:tcp_orientation_error_exp",
        weight=2.0,
        params={"sigma": _TCP_ORN_SIGMA},
    ),
}

g1_29dof_wbt_unified_tcp_reward = _replace(
    g1_29dof_wbt_unified_reward, terms={**g1_29dof_wbt_unified_reward.terms, **_TCP_PRECISION_TERMS}
)
g1_29dof_wbt_unified_anneal_tcp_reward = _replace(
    g1_29dof_wbt_unified_anneal_reward, terms={**g1_29dof_wbt_unified_anneal_reward.terms, **_TCP_PRECISION_TERMS}
)


# ---------------------------------------------------------------------------
# Phase 2: gripper-mass domain randomization (plan §5).
#
# THERE IS NO "gripper" BODY TO TARGET. `g1_29dof.asset.collapse_fixed_joints=True` →
# IsaacSim `merge_fixed_joints=True`, so the two FIXED joints of the hand chain
# (`*_ft_sensor_joint`, `*_hand_palm_joint`) are absorbed at URDF import: neither
# `*_ft_sensor_link` nor `*_rubber_hand` is an addressable PhysX body, and
# `SceneEntityCfg(body_names=[...])` raises for them. The two finger joints are PRISMATIC
# and survive as separate bodies (and are absent from the 32-entry
# `robot_config.body_names`, so they are not targetable either).
#
# That is fine, because the merged body IS the gripper (masses from g1_29dof_yam.urdf):
#     merged left_wrist_yaw_link      = 1.442568 kg
#       ├─ bare wrist_yaw_link          0.084576 kg   ( 5.9%)
#       └─ ft_sensor 0.25 + hand 1.107992 = 1.357992 kg  (94.1%)  ← the gripper
#     fingers 2 × 0.0710042 = 0.142008 kg  → separate prismatic bodies, NOT in this body
# So "randomize the gripper mass" and "scale `*_wrist_yaw_link`" are the same operation;
# the former is the intent, the latter the only available handle.
#
# WHY MASS AND NOT GRAVITY: there is no gravity randomization anywhere in this codebase
# (none of the 12 `randomize_*` terms in randomization/terms/locomotion.py touches it),
# and changing g is a WHOLE-ROBOT effect — leg balance, landing impulse, and the
# `projected_gravity` obs all move with it, injecting variance unrelated to "the hand
# weighs something different". Mass DR delivers exactly the intended physics, and all of
# it at once: static torque m·g·L, the m·a term inside F_meas, and the inertia tensor
# (isaaclab `randomize_rigid_body_mass` has `recompute_inertia=True` by default, scaling
# `default_inertia` by the mass ratio).
# ---------------------------------------------------------------------------

# Gripper-chain masses from `g1_29dof_yam.urdf`, per hand. Mirrored (not imported) because
# the URDF is only readable at runtime and these must be available at config-import time;
# `tests/test_tcp_precision.py::test_gripper_mass_constants_match_the_urdf` parses the URDF
# and pins all three, so a gripper respec fails loudly instead of silently desyncing the
# simulated F/T sensor. Also mirrored into `hybrid.py` — the same test pins that too.
MERGED_WRIST_DEFAULT_KG: float = 1.44256847  # bare wrist_yaw + ft_sensor + rubber_hand
GRIPPER_SCALED_PART_KG: float = 1.357992  # ft_sensor + rubber_hand (inside the merged body)
GRIPPER_FINGERS_KG: float = 0.1420084  # 2 × finger — PRISMATIC, NOT inside the merged body

# Scale range for the merged wrist_yaw body. SCALE semantics (`randomize_mass_startup`:
# link_mass_range scales, added_mass_range adds), sampled per (env, body) so the two hands
# differ independently — which is what the real robot looks like.
#
# Physical meaning (total gripper incl. fingers, nominal 1.500 kg):
#     s=0.8 → 1.228 kg (−0.272)   s=1.0 → 1.500 kg   s=1.5 → 2.179 kg (+0.679)
# Deliberately asymmetric: real deviation is mostly HEAVIER (a payload, an added fixture),
# and the lower bound only needs to cover URDF mass-modelling error — which is real and
# was measurably 0.705 kg/hand in the browser MJCF (see wbt_realtime_eval/urdf_inertia.py).
HAND_MASS_SCALE_RANGE: tuple[float, float] = (0.8, 1.5)

# Only the two merged wrist bodies. NOT `experiment._ee_mass_body_names`, which drags in
# pelvis/hip/knee (from the robot default) + ankles + wrist_roll/pitch: at scale [0.7,1.4]
# that buys 1.010–2.020 kg of swing on wrist_yaw but only 0.399 kg across roll+pitch, and
# the leg entries are locomotion DR — pure variance against this objective.
HAND_MASS_BODY_NAMES: tuple[str, ...] = _HAND_BODY_NAMES

_randomize_hand_mass_term = RandomizationTermCfg(
    func="holosoma.managers.randomization.terms.locomotion:randomize_mass_startup",
    params={
        "enable_link_mass": True,
        "link_mass_range": list(HAND_MASS_SCALE_RANGE),  # SCALE (not ADD)
        "enable_base_mass": False,  # torso stays nominal — this is about the HAND
    },
)

# The base WBT randomization (pushes / friction / base COM / dof_pos bias) plus mass DR.
# NOTE: `g1_29dof_wbt_randomization_ee_mass` is NOT usable for this — pinned holosoma's
# `wbt/g1/randomization.py` exports only `g1_29dof_wbt_randomization` and `_w_object`, so
# experiment.py's `try/except ImportError` is sitting in its fallback branch and that name
# is currently an alias to the base config. Which means the existing
# `g1_wbt_maskedmimic_ee_mass` / `_aggressive_dr` presets have NO mass DR at all today (a
# real silent degradation, tracked separately — plan §5.4). Hence we build the cfg here.
g1_29dof_wbt_randomization_hand_mass = _replace(
    _base_randomization,
    setup_terms={**_base_randomization.setup_terms, "randomize_mass_startup": _randomize_hand_mass_term},
)


# ---------------------------------------------------------------------------
# Phase gating: the DR is a SEPARATE 8-GPU run from the precision fix, but the user asked
# for both to land on `exp:g1-wbt-3pt-hybrid-unified[-anneal]` rather than on new preset
# names. Reconciled with an env knob, exactly the mechanism hybrid_3pt.py already uses for
# WBT_JV_DROPOUT_* (read at import time, surfaced in the launched env + the W&B config):
#
#     WBT_HAND_MASS_DR=0   (DEFAULT) → phase 1: TCP precision rewards only, no mass DR.
#     WBT_HAND_MASS_DR=1             → phase 2: + gripper-mass DR + m_distal sync.
#
# Why an env knob and not a CLI override: the injector config lives inside
# CommandTermCfg.params (a dict[str, Any]), which tyro treats as opaque — so
# `sync_distal_mass_from_sim` is unreachable from the command line (same constraint that
# forced `-servo-off` to be its own preset). The two must move TOGETHER (DR without the
# m_distal sync is the 2.04 N desync bug), so one knob flips both.
#
# Keeping the default OFF is what preserves the A/B: phase 1 measures the precision
# ceiling, phase 2 then measures what robustness costs. DR raises robustness and LOWERS
# peak accuracy, so co-launching them would leave "precision didn't improve" unattributable.
HAND_MASS_DR_ENABLED: bool = os.environ.get("WBT_HAND_MASS_DR", "0").strip().lower() in ("1", "true", "yes")

# Injector cfgs with the distal-mass sync switched on, and the command/termination carriers
# that reference them. Built unconditionally (cheap, frozen dataclasses) so both branches
# are import-time reachable and testable; only selected when the knob is set.
_UNIFIED_CFG_MASSDR = _replace(_UNIFIED_CFG, sync_distal_mass_from_sim=True)
_UNIFIED_ANNEAL_CFG_MASSDR = _replace(_UNIFIED_ANNEAL_CFG, sync_distal_mass_from_sim=True)

g1_29dof_wbt_unified_massdr_command = _unified_command(_UNIFIED_CFG_MASSDR)
g1_29dof_wbt_unified_anneal_massdr_command = _unified_command(_UNIFIED_ANNEAL_CFG_MASSDR)
g1_29dof_wbt_unified_massdr_termination = _unified_termination(_UNIFIED_CFG_MASSDR)
g1_29dof_wbt_unified_anneal_massdr_termination = _unified_termination(_UNIFIED_ANNEAL_CFG_MASSDR)

# R1's servo-authority curriculum also carries a HybridContactConfig. Today
# `ServoAuthorityCurriculum` only reads the threshold/step/beta/hold keys and never touches
# `self.hc_cfg`, so this is currently cosmetic — but leaving a stale cfg object in a params
# dict is exactly how a later edit silently reads `sync_distal_mass_from_sim=False` while
# the injector has it True. Re-point it so ALL FOUR carriers agree on one cfg object.
# `force_alpha` needs no twin: its param is a HandForceConfig, which has no such field.
_servo_authority_massdr_term = _replace(
    g1_29dof_wbt_unified_anneal_curriculum.setup_terms["servo_authority"],
    params={
        **g1_29dof_wbt_unified_anneal_curriculum.setup_terms["servo_authority"].params,
        "hybrid_config": _UNIFIED_ANNEAL_CFG_MASSDR,
    },
)
g1_29dof_wbt_unified_anneal_massdr_curriculum = _replace(
    g1_29dof_wbt_unified_anneal_curriculum,
    setup_terms={**g1_29dof_wbt_unified_anneal_curriculum.setup_terms, "servo_authority": _servo_authority_massdr_term},
    reset_terms={**g1_29dof_wbt_unified_anneal_curriculum.reset_terms, "servo_authority": _servo_authority_massdr_term},
)

# Reward twins carrying the massdr injector cfg. The `force_error_direct` /
# `contact_init_bonus` / `approach_guidance` / `force_rate_penalty` terms take a
# HybridContactConfig in their params, so they must point at the SAME cfg object the
# injector uses — otherwise `sync_distal_mass_from_sim` would read True in one place and
# False in another. Weights and term keys are identical either way (checkpoint lineage).
g1_29dof_wbt_unified_massdr_tcp_reward = _replace(
    _unified_reward(_UNIFIED_CFG_MASSDR), terms={**_unified_reward(_UNIFIED_CFG_MASSDR).terms, **_TCP_PRECISION_TERMS}
)
g1_29dof_wbt_unified_anneal_massdr_tcp_reward = _replace(
    _unified_reward(_UNIFIED_ANNEAL_CFG_MASSDR),
    terms={**_unified_reward(_UNIFIED_ANNEAL_CFG_MASSDR).terms, **_TCP_PRECISION_TERMS},
)


def distal_mass_from_body_mass(m_body: torch.Tensor) -> torch.Tensor:
    """Map a randomized merged-wrist body mass → the F/T sensor's distal mass, exactly.

        s          = m_body / MERGED_WRIST_DEFAULT_KG
        m_distal(s) = s · GRIPPER_SCALED_PART_KG + GRIPPER_FINGERS_KG

    The fingers are separate PRISMATIC bodies, so `wrist_yaw`'s scale does NOT touch them
    — attributing the whole body delta to the distal mass would be wrong. (The error of
    that approximation is only 0.017–0.042 kg ⇒ 0.05–0.13 N at a=3 m/s², i.e. under the
    0.5 N noise floor. The exact expression is no harder, so there is no reason to
    approximate.)

    Why this must be synced at all: `simulated_measured_force` adds `m_distal · a_wrist`
    to F_meas, and real hardware necessarily measures its ACTUAL distal mass. Leaving
    m_distal at the nominal 1.5 kg while the body is scaled to 1.5 desyncs the simulated
    sensor from the dynamics by 0.679 kg → 2.04 N at a=3 m/s², against
    `contact_phase_theta_on = 2.0 N` and `sensor_noise_std = 0.5 N`. Same order as the
    contact threshold, so it would corrupt the hysteresis label, `force_error_direct`, and
    Eq.2's force target.

    Elementwise; any shape. `s=1` returns exactly the nominal 1.500092 kg.
    """
    return (m_body / MERGED_WRIST_DEFAULT_KG) * GRIPPER_SCALED_PART_KG + GRIPPER_FINGERS_KG


def unified_tcp_preset_overrides(*, anneal: bool) -> dict[str, Any]:
    """The `replace(...)` kwargs experiment.py applies to a local unified preset.

    ONE place decides what phase 1 vs phase 2 means, so the R2 and R1 twins cannot drift
    apart and experiment.py stays declarative.

    Phase 1 (default): `reward` only — the two TCP precision terms added on top of the
    unified reward. Command / termination / randomization / robot are untouched, so obs
    stays 170D and pfl96opy warm-starts.

    Phase 2 (`WBT_HAND_MASS_DR=1`): additionally swaps `randomization` for the
    mass-DR config, points `robot.randomize_link_body_names` at just the two wrist bodies,
    and swaps command / reward / termination onto the cfg carriers that have
    `sync_distal_mass_from_sim=True`. Obs is STILL 170D — DR changes physics, not the obs
    layout — so a phase-1 checkpoint warm-starts into phase 2 as well.
    """
    if not HAND_MASS_DR_ENABLED:
        return {"reward": g1_29dof_wbt_unified_anneal_tcp_reward if anneal else g1_29dof_wbt_unified_tcp_reward}
    if anneal:
        return {
            "reward": g1_29dof_wbt_unified_anneal_massdr_tcp_reward,
            "command": g1_29dof_wbt_unified_anneal_massdr_command,
            "termination": g1_29dof_wbt_unified_anneal_massdr_termination,
            "curriculum": g1_29dof_wbt_unified_anneal_massdr_curriculum,
            "randomization": g1_29dof_wbt_randomization_hand_mass,
        }
    return {
        "reward": g1_29dof_wbt_unified_massdr_tcp_reward,
        "command": g1_29dof_wbt_unified_massdr_command,
        "termination": g1_29dof_wbt_unified_massdr_termination,
        "randomization": g1_29dof_wbt_randomization_hand_mass,
    }


__all__ = [
    "tcp_position_error_exp",
    "tcp_orientation_error_exp",
    "g1_29dof_wbt_unified_tcp_reward",
    "g1_29dof_wbt_unified_anneal_tcp_reward",
    "g1_29dof_wbt_randomization_hand_mass",
    "unified_tcp_preset_overrides",
    "distal_mass_from_body_mass",
    "HAND_MASS_DR_ENABLED",
    "MERGED_WRIST_DEFAULT_KG",
    "GRIPPER_SCALED_PART_KG",
    "GRIPPER_FINGERS_KG",
    "HAND_MASS_SCALE_RANGE",
    "HAND_MASS_BODY_NAMES",
]
