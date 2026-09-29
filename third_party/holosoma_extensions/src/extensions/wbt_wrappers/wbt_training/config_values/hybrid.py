"""Hybrid position-force controller v1 for the `exp:g1-wbt-2pt-hybrid` preset.

Forks the plain 2-point tracker (`g1_wbt_2pt`), references the UniFP force machinery
from `force_tracking.py`, and replaces the SCRIPTED F^ext disturbance pulse with a
CAUSAL contact spring: F^ext = K_env · penetration(x_tcp) · n against a per-env
virtual plane (D1). F^cmd stays a commanded pulse. A simulated F/T sensor reads
F^meas = F^ext + inertial(m·a) + bias + noise, transformed sensor->TCP via a
constant 6×6 wrench adjoint (xyz force active, 3 moment channels dormant per D8, D4).

TWO LAYERS (one module, one preset):
  Layer A (architecture A): F^meas in the ACTOR + critic obs (real sensor in the loop,
    NOT critic-only — we are not sensorless); rebalanced torso-stable reward; causal
    contact. Trainable with servo_enabled=False.
  Layer B (architecture B, behind servo_enabled): Jᵀ contact-gated force servo writes
    Δq_arm into the arm PD target so steady-state force reaches F^cmd independent of K
    (D2); integrator I + x_tgt in obs; servo-inclusive Eq.2 + pre-integrator force
    reward; force-runaway estop.
Deferred to v2: D6 high-rate {F_latest,F_mean,dF/dt} summary + sensor latency, 200 Hz
multi-rate servo, 6D moment loop, deploy Bota ingest.

Design: docs/2026-07-08-hybrid-position-force-controller-design.md (D1/D2/D4/D7/D8, §8).
Single-writer rule: only HybridContactInjector recomputes F^ext/F^meas (substep 0);
all obs/reward functions are pure readers of the env-attached buffers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from loguru import logger

from wbt_training.config_values.force_tracking import HandForceConfig, _rot_matrix_from_rpy

# --- Hardware geometry (verified vs g1_29dof_yam.urdf; MUST match the inference
# playground hybrid_pf_playground.py, asserted by test not imported). -----------
FT_OFFSET_X: float = 0.082  # FT sensor fixed child of wrist_yaw, +x (URDF joint origin, unchanged)
TCP_OFFSET_X: float = 0.242  # grasp/TCP fingertip midpoint, +x of wrist_yaw (2026-07-10: palm standoff
#                              0.0295→0.0565 moved TCP 0.215→0.242; FT sensor stayed at 0.082)
FT_TCP_LEVER_X: float = TCP_OFFSET_X - FT_OFFSET_X  # 0.16 m rigid lever (was 0.133)

# R_sensor: {side}_ft_sensor_joint rpy=(0,-pi/2,pi). Symmetric involution:
# sensor +x -> wrist +z, +y -> wrist -y, +z -> wrist +x.
R_SENSOR: torch.Tensor = _rot_matrix_from_rpy((0.0, -1.5707963267948966, 3.141592653589793))


def sensor_to_tcp_wrench(w_sensor: torch.Tensor) -> torch.Tensor:
    """Constant sensor->TCP 6-D wrench adjoint (…, 6) -> (…, 6).

    f_T = R_sensor · f_S
    m_T = R_sensor · m_S − lever × f_T,  lever = (FT_TCP_LEVER_X, 0, 0)

    For a pure 3-D force (m_S = 0) the force channel is rotation-only and the moment
    channel is the lever-induced term. Batched over any leading dims. The moment
    channels are DORMANT in M1 (D8) — computed but sliced off by force consumers.
    """
    r = R_SENSOR.to(w_sensor)
    f_s = w_sensor[..., :3]
    m_s = w_sensor[..., 3:]
    f_t = torch.einsum("ij,...j->...i", r, f_s)  # R_sensor @ f_S
    m_rot = torch.einsum("ij,...j->...i", r, m_s)  # R_sensor @ m_S
    lever = torch.tensor([FT_TCP_LEVER_X, 0.0, 0.0], device=w_sensor.device, dtype=w_sensor.dtype)
    lever = lever.expand_as(f_t)
    m_t = m_rot - torch.cross(lever, f_t, dim=-1)
    return torch.cat([f_t, m_t], dim=-1)


@dataclass(frozen=True)
class HybridContactConfig:
    """Knobs for the M1 hybrid contact + simulated-sensor variant.

    Wraps a `HandForceConfig` for the (unchanged) F^cmd command pulse and adds the
    causal-contact + sensor-model knobs. Defaults per design doc §8/D4/D8.
    """

    # F^cmd command pulse (delegated to HandForceConfig semantics).
    f_cmd_max: float = 25.0  # N
    stiffness_K: float = 200.0  # N/m, Eq.2 target shift

    # Reward shaping (Eq.2 kernel), inherited form.
    sigma: float = 0.3  # m

    # F^cmd trapezoidal pulse phases, env steps @ 50 Hz (match HandForceConfig).
    ramp_steps: tuple[int, int] = (50, 150)
    hold_steps: tuple[int, int] = (175, 450)
    idle_steps: tuple[int, int] = (50, 150)

    # --- Causal contact (D1) ---
    # Per-env K_env log-uniform range (N/m): soft (constant-force) to hard (rigid).
    # Bounded so the contact regime OVERLAPS the 25 N command / ~23 N wrist-saturation
    # regime (§5.1), not 1000s of N (Opus review). NOTE the FORCE-REACHABILITY invariant
    # (2026-07-10 review fix): the softest wall can only deliver k_env_min·max_penetration_m
    # N before the penetration clamp binds. With k_env_min=100, max_penetration_m=0.08 that
    # is 8 N < f_cmd_max=25 N — so a 25 N command on a soft wall is PHYSICALLY unreachable.
    # We therefore CLAMP the per-episode |F_cmd| to the sampled wall's reachable force in
    # reset() (see `f_cmd_reachable_clamp`), rather than raise k_env_min so high the soft
    # regime disappears. This keeps a rich soft→hard K_env range AND a well-posed force target.
    k_env_min: float = 100.0
    k_env_max: float = 800.0
    # Virtual-plane sampling: plane passes through a point offset from the hand's
    # reset TCP along a random inward normal; distance so the reachable push regime
    # produces contact. plane_dist_max < F_cmd/K (0.125 m) so the max-reward point
    # requires penetration during the hold phase (degenerate-avoidance, Opus review).
    plane_dist_min: float = 0.02  # m: min |signed distance| of plane from reset TCP
    plane_dist_max: float = 0.09  # m (< 0.125 = f_cmd_max/stiffness_K)
    # Max penetration clamp (m) — guards a teleport spike. Raised 0.05→0.08 (2026-07-10)
    # so the servo has force headroom on stiffer walls; k_env_max·max_penetration=64 N (pre-clamp).
    max_penetration_m: float = 0.08
    # Hard clamp on |F_ext| (N) after the spring law, so no substep ever injects a
    # wrist-shattering force even under a transient deep penetration. Above f_cmd_max (25)
    # so a legitimate command is never clipped, below the estop cutoff (40 N).
    max_force_n: float = 35.0
    # --- 6D-ready contact model (§9.1 seam #3, DORMANT in v1 per D8). Written now so 6D =
    # config flip + retrain, NOT a rewrite of the contact model (2026-07-10, decision 4). ---
    # K_env is a 6-vec [K_lin(3), K_rot(3)]: the linear entries use k_env_min/max above
    # (isotropic, sampled per episode); the ROTATIONAL entries use k_env_rot_* and are 0 in
    # v1 (moment dormant) — flip these >0 (+ activate the moment channels in servo_selection,
    # + torque-limit-aware F_cmd) to turn on 6D. The contact geometry carries a full SURFACE
    # FRAME (normal + 2 tangents), and penetration produces BOTH linear depth AND an
    # orientation error, so the moment path exists (multiplied by K_rot=0 today).
    k_env_rot_min: float = 0.0  # N·m/rad — 0 = moment dormant (v1). >0 activates 6D moment contact.
    k_env_rot_max: float = 0.0
    max_orientation_error_rad: float = 0.20  # clamp on the surface-attitude error (dormant guard)

    # F_cmd-along-normal (2026-07-10 well-posedness fix): with all-3-axis force control
    # (servo_selection=(1,1,1)) a compression-only plane can only react along +normal, so a
    # tangential/outward F_cmd component can NEVER be nulled (integrator winds to I_max, the
    # force reward has a ceiling <1). Fix: sample F_cmd's DIRECTION along the (episode-fixed)
    # −plane_normal so the whole command lies on the one axis the wall can react to; only the
    # trapezoidal MAGNITUDE pulses. Set False to restore the legacy random-direction F_cmd
    # (only sound with a full 6-DOF reactive contact, i.e. v2). See reset(). Design §8.6.
    f_cmd_along_normal: bool = True
    # Clamp the per-episode |F_cmd| to the sampled wall's reachable force
    # min(f_cmd_max, k_env·max_penetration_m) so the force target is always physically
    # attainable (else soft-wall force tracking is unwinnable regardless of the servo).
    f_cmd_reachable_clamp: bool = True

    # --- Simulated F/T sensor (D4) ---
    distal_mass_kg: float = 1.5  # rubber_hand 1.108 + 2×0.071 + sensor 0.25 (D4)
    # Read the ACTUAL post-randomization wrist-body mass out of PhysX at setup and derive
    # m_distal per (env, hand) from it, instead of using the nominal scalar above. Required
    # whenever hand-mass DR is enabled: F_meas carries m_distal·a_wrist, and real hardware
    # necessarily measures its true distal mass, so a scaled body with a nominal m_distal
    # desyncs the simulated sensor from the dynamics by up to 2.04 N at a=3 m/s² (scale 1.5)
    # — the same order as contact_phase_theta_on=2.0 N and 4× sensor_noise_std. DEFAULT
    # False so every pre-existing preset keeps the exact scalar arithmetic (bit-identical:
    # `distal_mass=None` takes literally the old code path in simulated_measured_force).
    # See tcp_precision.g1_29dof_wbt_randomization_hand_mass and distal_mass_from_body_mass.
    sync_distal_mass_from_sim: bool = False
    sensor_noise_std: float = 0.5  # N, per-substep white noise (domain-randomized in v2)
    sensor_bias_std: float = 0.3  # N, per-episode constant bias
    include_inertial: bool = True  # synthesize F_inertial = m·a (set False for clean-sim ablation)

    # --- Jᵀ force servo (Layer B, D2). ---
    # DATACLASS default is False (safe library default) so the Tasks 1–6 Layer-A unit tests
    # stay self-contained (they use a bare HybridContactConfig() and must NOT depend on the
    # Task 8–10 servo code). The SHIPPED PRESET, however, sets servo_enabled=True (Task 11's
    # _DEFAULT_HYBRID_CFG) — so LAUNCHING `exp:g1-wbt-2pt-hybrid` DOES force tracking (user
    # decision 2026-07-10: "默认开伺服"). A/B ablation = launch with a servo_enabled=False
    # override. This "safe default, opinionated preset" split delivers force-tracking-on-launch
    # while keeping the plan buildable task-by-task. Obs width is fixed 165D either way.
    servo_enabled: bool = False  # dataclass default; the PRESET overrides to True (Task 11).
    # --- C1 observe-only integrator (unified single-policy design, 2026-07-22 spec). ---
    # True (only meaningful with servo_enabled=True): the contact-gated integrator keeps
    # computing I/Δx every substep — feeding the SAME obs terms (integrator_state,
    # x_tgt_shift) and the SAME servo-inclusive Eq.2 reward target — but Δq_arm is NEVER
    # injected into the PD path. The integrator degrades from controller to REFERENCE
    # GENERATOR; the policy is the sole executor. No Jacobian needed in this mode.
    servo_observe_only: bool = False
    # --- Soft deployable contact-phase label (C1, spec §3; user 2026-07-24: θ_on=2 N). ---
    # Binary hysteresis on ‖F_meas‖ (ON above θ_on, OFF below θ_off, holds in the dead
    # band) smoothed by a first-order LPF (τ seconds). ∈[0,1], per hand. Computed from the
    # MEASURED force (deployable — real Bota gives the same signal), NOT the privileged
    # depth>0 gate (critic-only). Consumed by the actor obs `contact_phase` and the
    # unified reward gates (hybrid_unified.py). Buffers updated every substep regardless
    # of these values, so the cost is identical for existing presets.
    contact_phase_theta_on: float = 2.0  # N — above this = contact (user 2026-07-24: 3→2; ≈4σ of sensor_noise_std=0.5)
    contact_phase_theta_off: float = 1.5  # N — below this = no contact (spurious touch); hysteresis lower edge
    contact_phase_tau_s: float = 0.05  # s, LPF time constant (≈2.5 env steps @ 50 Hz)
    # Post-reset warmup (env steps) during which the hysteresis is FORCED OFF. The reset
    # teleport spikes body_lin_acc_w, and F_meas carries m·a (1.5 kg × 1.33 m/s² = 2 N = θ_on),
    # so without this the label slams to 1 in free space at ~every episode start — the same
    # inertial-transient failure mode that made the old F_meas estop kill episodes (see
    # force_runaway). Real robots don't teleport, so suppressing the first few sim steps
    # keeps the label deployable (a step counter exists at deploy too). 2026-07-24 review.
    contact_phase_warmup_steps: int = 5
    # Jᵀ residual gain k in Δq_arm = k·Jᵀ·(K·Δx). UNITS: rad/(N·m) (task_wrench[N]·J[m/rad]→[N·m],
    # ×k→[rad]). ANCHOR (2026-07-10 fix, was 0.0002 ≈ 351× too weak → servo near-inert): the
    # "unity-authority" gain is k = 1/K_p ≈ 1/14.25 ≈ 0.07 (arm PD stiffness, robot_config.py),
    # at which the injected PD torque K_p·Δq reproduces the intended joint torque Jᵀ·wrench 1:1.
    # Start slightly below unity for stability margin; raise toward 0.07 per the W&B watch-list.
    servo_gain_k: float = 0.05  # rad/(N·m) — near unity-authority (1/K_p≈0.07), NOT the old 0.0002 stub.
    integrator_gain_ki: float = 0.5  # K_I on the contact-gated integral term (continuous-time units, 1/s per §9.3).
    # I_max clip (N — the integrator is in FORCE units, see servo_residual_step), anti-windup.
    # DESIGN CHANGE (2026-07-10, user decision "力单位积分"): the integrator accumulates the FORCE
    # error e=F_cmd+F_meas directly into a force offset I_force (N), and task_wrench = F_cmd + I_force.
    # This makes the integral-path loop gain = k·K_eff·J²·K_I — INDEPENDENT of the guessed
    # stiffness_K (the middle form K·Δx=F_cmd+K·I had loop gain ∝ stiffness_K, so sweeping the K
    # guess swept stability and diverged at the high end; verified by scalar sim). Steady state
    # e→0 ⟹ F_applied→F_cmd exactly, K-independent. I_force is bounded to ±integrator_max N; 40 N
    # is generous headroom above f_cmd_max=25 (task_wrench then feeds k·Jᵀ, clamped by servo_dq_max).
    integrator_max: float = 40.0  # N (force-unit integral); one-step ΔI = K_I·f_cmd_max·dt ≪ this.
    servo_dq_max: float = 0.15  # rad, hard clamp on |Δq_arm| per joint (don't let the servo dominate π).
    # Which task axes the servo closes force on (selection s, length-6, xyz active per D8).
    # (1,1,1,0,0,0) = close all 3 force axes, moment axes off (dormant). NOTE (2026-07-10 frame
    # review): all-force/no-moment is a BLOCK selection that commutes with rotation, so applying
    # it in the world frame is frame-invariant (correct). A PARTIAL within-block split (e.g.
    # normal-force + tangential-compliance, design §8.6) would NOT be frame-invariant in world —
    # it would require closing in a contact-aligned orthonormal frame. v1 is all-3-axis force
    # (user 2026-07-10), so world-frame closure is correct; partial hybrid selection is v2.
    servo_selection: tuple[float, ...] = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    # Force-runaway estop cutoff (N): ‖F_ext‖ (the CONTACT force, NOT F_meas) above this →
    # terminate (design §7.9). Above max_force_n so the sim contact-force clamp keeps it DORMANT
    # in sim (deploy-safety valve on the unclamped real-Bota reading); below the ~46 N (2×23)
    # two-wrist regime. NOTE (2026-07-10): thresholding on F_ext not F_meas — F_meas carries the
    # inertial term m·a, which spikes past 40 N in free space at reset and killed ~every episode.
    force_runaway_cutoff: float = 40.0

    # --- Geometry (mirror module constants; overridable for a different gripper) ---
    tcp_offset_x: float = TCP_OFFSET_X
    ft_offset_x: float = FT_OFFSET_X
    sensor_rpy: tuple[float, float, float] = (0.0, -1.5707963267948966, 3.141592653589793)

    # Force application bodies (parent articulation links PhysX targets).
    body_names: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")

    seed: int = 0

    def hand_force_config(self) -> HandForceConfig:
        """Build the F^cmd-pulse HandForceConfig (F^ext pulse is UNUSED — F^ext is
        causal here, so f_ext_max=0 disables the scripted ext pulse entirely)."""
        return HandForceConfig(
            f_cmd_max=self.f_cmd_max,
            f_ext_max=0.0,  # F^ext is causal (K_env spring), NOT a scripted pulse
            stiffness_K=self.stiffness_K,
            sigma=self.sigma,
            ramp_steps=self.ramp_steps,
            hold_steps=self.hold_steps,
            idle_steps=self.idle_steps,
            body_names=self.body_names,
            sensor_rpy=self.sensor_rpy,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Causal contact model (D1): virtual plane, penetration, spring reaction.
# Pure functions (no env, no sim) — fully MagicMock/tensor unit-testable.
# ---------------------------------------------------------------------------


def penetration_depth(x_tcp: torch.Tensor, normal: torch.Tensor, point: torch.Tensor, max_pen: float) -> torch.Tensor:
    """Compression-only penetration depth of the TCP behind a plane.

    Plane has OUTWARD unit `normal` (pointing into free space, toward the robot) and
    passes through `point`. A point on the free side has n·(x_tcp − point) > 0;
    penetration (behind the surface, inside the wall) is n·(point − x_tcp) > 0:

        d = clamp(max(0, n·(point − x_tcp)), 0, max_pen)

    Broadcasts over leading dims; returns shape = x_tcp.shape[:-1].
    """
    signed = ((point - x_tcp) * normal).sum(dim=-1)  # n · (point − x_tcp)
    return signed.clamp(min=0.0).clamp(max=max_pen)


def contact_force_world(
    k_env: torch.Tensor, depth: torch.Tensor, normal: torch.Tensor, max_force_n: float = 30.0
) -> torch.Tensor:
    """Causal contact reaction on the robot: F_ext = K_env · depth · n (along +normal),
    magnitude-clamped to `max_force_n`.

    `k_env` and `depth` broadcast to `normal`'s leading dims. Compression-only
    (depth ≥ 0 by construction), so the force always pushes the robot OUT of the wall.
    The clamp keeps a transient deep penetration from injecting a wrist-shattering
    force (the raw spring can exceed the 25 N command / ~23 N wrist-saturation regime).
    Algebraically identical to the design's `F_ext = −K_env·(x_tcp − x_wall)` for the
    penetrating case (verified: n·(point−x_tcp) = −(x_tcp−x_wall)·n along the normal).
    """
    mag = (k_env * depth).clamp(max=max_force_n)  # scalar magnitude ≤ max_force_n
    return mag.unsqueeze(-1) * normal


def sample_plane(
    num_envs: int, num_hands: int, cfg: "HybridContactConfig", gen, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a per-(env, hand) virtual plane: a random inward unit normal and a
    point at a random signed distance in front of the origin.

    NOTE: the returned `point` is expressed RELATIVE to the hand — the injector adds
    the hand's reset TCP so the plane sits `plane_dist ∈ [min, max]` in front of the
    reset hand along `-normal` (so a forward push reaches it). Normals are drawn on
    the unit sphere then normalized. Uses the module `torch.Generator` for
    reproducibility (parity with force_tracking._resample_pulse).
    """
    raw = torch.randn((num_envs, num_hands, 3), generator=gen)
    normal = raw / raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dist = (
        torch.rand((num_envs, num_hands), generator=gen) * (cfg.plane_dist_max - cfg.plane_dist_min)
        + cfg.plane_dist_min
    )
    # Point sits `dist` in front of the hand along -normal (the injector adds reset TCP).
    point = -dist.unsqueeze(-1) * normal
    return normal.to(device), point.to(device)


def surface_tangents(normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a per-(…) orthonormal SURFACE FRAME (t1, t2) spanning the plane, given the
    unit `normal` (…,3). Returns (t1, t2) each (…,3) with {t1, t2, normal} right-handed
    orthonormal. 6D-ready seam (§9.1 #3): the contact geometry carries a FULL surface frame,
    not just a normal, so the dormant orientation-error / moment path has axes to live on.
    Uses the standard "pick the least-aligned world axis, cross" construction (branch-free)."""
    # Reference axis = whichever world axis is least parallel to `normal` (avoids degeneracy).
    ax = torch.zeros_like(normal)
    least = normal.abs().argmin(dim=-1, keepdim=True)  # (…,1)
    ax.scatter_(-1, least, 1.0)
    t1 = torch.cross(normal, ax, dim=-1)
    t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    t2 = torch.cross(normal, t1, dim=-1)  # already unit (normal ⟂ t1, both unit)
    return t1, t2


def contact_moment_world(k_env_rot: torch.Tensor, orient_err: torch.Tensor, max_orient_err: float) -> torch.Tensor:
    """DORMANT 6D moment channel (§9.1 seam #3, D8). Contact moment from the TCP-vs-surface
    orientation error: M = K_env_rot ⊙ clamp(orient_err) (…,3), in world axes aligned with the
    surface frame. `k_env_rot` (…,3) is 0 in v1 → returns 0 (moment dormant); flip
    cfg.k_env_rot_* > 0 to activate 6D. Written+shaped now so 6D = config flip, not a rewrite.
    `orient_err` (…,3) is the axis-angle attitude error of the TCP frame vs the surface frame."""
    err = orient_err.clamp(min=-max_orient_err, max=max_orient_err)
    return k_env_rot * err


# ---------------------------------------------------------------------------
# Simulated F/T sensor (D4): F_meas = F_ext + inertial(m·a) + bias + noise.
# ---------------------------------------------------------------------------


def simulated_measured_force(
    f_ext_w: torch.Tensor,
    wrist_acc_w: torch.Tensor,
    bias_w: torch.Tensor,
    cfg: "HybridContactConfig",
    noise: torch.Tensor,
    distal_mass: torch.Tensor | None = None,
) -> torch.Tensor:
    """Synthesize the world-frame F/T sensor reading from the causal contact force.

        F_meas = F_ext + [m_distal · a_wrist if include_inertial] + bias + noise

    All args broadcast over leading dims (…, 3). `noise` is pre-sampled by the caller
    (the injector owns the RNG) so this stays a deterministic, unit-testable function.
    Hardware does static gravity comp (D4), so no gravity term — only the dynamic
    inertial pseudo-force. The moment channels are dormant in M1 (D8): this returns
    3-D force only; the injector zero-pads to the 6-D buffer.

    `distal_mass`: optional per-(env, hand) distal mass in kg, broadcastable against
    `wrist_acc_w` (i.e. shaped (…, 1) for a (…, 3) acceleration). Populated by the
    injector from the ACTUAL post-randomization body mass when hand-mass DR is on (see
    `tcp_precision.distal_mass_from_body_mass`); `None` falls back to the scalar
    `cfg.distal_mass_kg`.

    ⚠️ PRECISELY WHAT "bit-identical" MEANS HERE. The injector allocates
    `_wbt_distal_mass` UNCONDITIONALLY (at the nominal value), so in production the
    TENSOR branch is what runs even with DR off — the `None` scalar branch is only taken
    by unit tests and non-IsaacSim backends. The result is still bit-identical because
    `torch.full((n,h,1), v) * acc == v * acc` exactly in float32 for these values (pinned
    by `test_tcp_precision.py::test_scalar_distal_mass_path_is_bit_identical`), NOT
    because the old code path is still live. Do not "simplify" on the assumption that the
    scalar path runs in training.
    """
    out = f_ext_w.clone()
    if cfg.include_inertial:
        m = cfg.distal_mass_kg if distal_mass is None else distal_mass
        out = out + m * wrist_acc_w
    return out + bias_w + noise


# ---------------------------------------------------------------------------
# HybridContactInjector: sole owner of the hybrid force buffers + causal-contact
# substep hook. Replaces HandForceInjector's SCRIPTED F^ext pulse with a CAUSAL
# K_env contact spring (D1) + simulated F/T sensor (D4). 6D-ready, xyz active (D8).
# ---------------------------------------------------------------------------
from holosoma.managers.command.base import CommandTermBase  # noqa: E402

from wbt_training.config_values.force_tracking import (  # noqa: E402
    _allocate_pulse_state,
    _ensure_force_isaac_indices,
    _quat_apply_inverse_wxyz,
    _quat_apply_wxyz,
    _step_pulse,
    restart_force_pulse_for_envs,
)


class HybridContactInjector(CommandTermBase):
    """Sole owner of the hybrid force buffers + the causal-contact substep hook.

    Replaces force_tracking.HandForceInjector's SCRIPTED F^ext pulse with a CAUSAL
    K_env contact spring against a per-env virtual plane (D1), and adds a simulated
    F/T sensor (D4). F^cmd keeps the trapezoidal pulse (via HandForceConfig with
    f_ext_max=0). 6D-ready buffers, xyz active (D8).

    setup(): allocate buffers + pulse state; monkey-patch the substep hook.
    reset(env_ids): restart the F^cmd pulse, resample K_env + plane + sensor bias.
    Substep hook: advance F^cmd (substep 0) -> compute causal F^ext + F^meas from
    live TCP -> (servo write when enabled) -> inject F^ext into PhysX -> original PD.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.hc_cfg: HybridContactConfig = cfg.params["hybrid_config"]
        self.hf_cfg: HandForceConfig = self.hc_cfg.hand_force_config()
        self.body_names = list(self.hc_cfg.body_names)
        self._isaac_body_ids: torch.Tensor | None = None
        self._tcp_offsets_b: torch.Tensor | None = None
        self._original_apply_force = None
        # Layer-B (servo) state cached across substeps; only touched when servo_enabled.
        self._policy_actions: torch.Tensor | None = None  # committed _actions_after_delay @ substep 0
        self._joint_term = None  # the JointPositionActionTerm the servo injects into
        self._jac_wrist_body_ids: list[int] | None = None
        self._jac_arm_cols: list[list[int]] | None = None
        self._arm_pd_slice: list[list[int]] | None = None

    def setup(self) -> None:
        env, cfg = self.env, self.hc_cfg
        n, h = env.num_envs, len(self.body_names)
        dev = env.device
        # F^cmd buffers (root + world), reused by the pulse machinery.
        env._wbt_force_cmd_b = torch.zeros(n, h, 3, device=dev)
        env._wbt_force_ext_b = torch.zeros(n, h, 3, device=dev)  # unused (ext is causal), kept for advance API
        env._wbt_force_cmd_w = torch.zeros(n, h, 3, device=dev)
        env._wbt_force_ext_w = torch.zeros(n, h, 6, device=dev)  # 6D-ready: causal contact
        env._wbt_force_meas_w = torch.zeros(n, h, 6, device=dev)  # 6D-ready: simulated sensor
        env._wbt_contact_gate = torch.zeros(n, h, device=dev)
        env._wbt_penetration = torch.zeros(n, h, device=dev)  # depth (m), critic-only privileged
        env._wbt_k_env = torch.zeros(n, h, device=dev)
        env._wbt_k_env_rot = torch.zeros(n, h, 3, device=dev)  # 6D-ready rotational stiffness (0=dormant)
        env._wbt_plane_normal = torch.zeros(n, h, 3, device=dev)
        env._wbt_surface_t1 = torch.zeros(n, h, 3, device=dev)  # 6D-ready surface frame tangent 1
        env._wbt_surface_t2 = torch.zeros(n, h, 3, device=dev)  # 6D-ready surface frame tangent 2
        env._wbt_plane_point = torch.zeros(n, h, 3, device=dev)
        env._wbt_plane_seeded = torch.zeros(n, dtype=torch.bool, device=dev)  # plane anchored yet?
        env._wbt_sensor_bias_w = torch.zeros(n, h, 3, device=dev)
        # Soft contact-phase label (C1 unified, 2026-07-22): hysteresis+LPF state, ∈[0,1] per
        # hand. Allocated UNCONDITIONALLY (same fixed-width rule as the servo buffers) — the
        # 185D unified obs reads _wbt_contact_phase; existing presets simply never read it.
        env._wbt_contact_phase = torch.zeros(n, h, device=dev)  # soft deployable gate ∈[0,1]
        env._wbt_contact_hyst = torch.zeros(n, h, device=dev)  # binary hysteresis state
        env._wbt_force_ext_prev_w = torch.zeros(n, h, 3, device=dev)  # last env-step clean F_ext
        # EPISODE-FIXED F^cmd direction (root frame, unit) + per-episode reachable-force CAP (N),
        # seeded in reset(). The reused _step_pulse auto-resamples f_target mid-episode to a fresh
        # UNCLAMPED, RANDOM-direction vector (force_tracking._resample_pulse) — which would silently
        # defeat BOTH well-posedness invariants (F_cmd-along-normal + reachable clamp) after the
        # first pulse cycle (Opus review 2026-07-10). _advance_fcmd_pulse re-conforms every step to
        # these buffers, so only the trapezoidal MAGNITUDE pulses; the direction + cap stay
        # episode-fixed (matching the plane normal, which is also only set at reset).
        env._wbt_fcmd_dir_b = torch.zeros(n, h, 3, device=dev)  # unit push direction, root frame (legacy path)
        # WORLD-fixed F_cmd direction (unit) — aligned with −plane_normal, episode-fixed. The plane
        # normal is WORLD-fixed at reset; storing F_cmd's direction in WORLD (not root) keeps the
        # command aligned with the (world-fixed) wall for the WHOLE episode even as the PELVIS
        # reorients tracking the clip (Opus frames review 2026-07-10). A physical wall is world-fixed,
        # so the command must be too — else the root-live-rotated F_cmd drifts off the normal and the
        # compression-only wall can't null the tangential component (integrator winds).
        env._wbt_fcmd_dir_w = torch.zeros(n, h, 3, device=dev)  # unit push direction, WORLD frame
        env._wbt_fcmd_reach = torch.full((n, h), cfg.f_cmd_max, device=dev)  # reachable |F_cmd| cap (N)
        # Servo state (Layer B). Allocated UNCONDITIONALLY (even servo_enabled=False) so the
        # obs is a FIXED 165D width — the integrator_state/x_tgt_shift obs terms always read a
        # real (zeroed-when-off) buffer, one network + checkpoint lineage for both modes.
        env._wbt_integ_state = torch.zeros(n, h, 3, device=dev)  # I (task-space integral, xyz)
        env._wbt_delta_x = torch.zeros(n, h, 3, device=dev)  # Δx (servo setpoint shift, xyz)
        env._wbt_servo_dq = torch.zeros(n, len(env.simulator.dof_names), device=dev)  # PD-target residual
        # Pulse state (F^cmd only; f_ext_max=0 makes the ext pulse a no-op).
        env._wbt_force_state = _allocate_pulse_state(env, self.hf_cfg)
        env._wbt_hybrid_cfg = cfg
        env._wbt_force_cfg = self.hf_cfg  # so obs helpers / restart find it
        # TCP offset (the physically correct contact anchor, 0.242 m — 2026-07-10, was 0.215).
        self._tcp_offsets_b = torch.tensor([(cfg.tcp_offset_x, 0.0, 0.0)] * h, dtype=torch.float, device=dev)
        env._wbt_force_offsets_b = self._tcp_offsets_b  # consumed by Eq.2 reward FK
        # Resolve isaac body ids.
        self._isaac_body_ids = _ensure_force_isaac_indices(env, self.body_names)
        # Per-(env, hand) distal mass for the simulated F/T sensor's m·a term, shaped
        # (n, h, 1) to broadcast against the (n, h, 3) acceleration. Allocated
        # UNCONDITIONALLY at the nominal value; only populated from the sim when
        # sync_distal_mass_from_sim is set (hand-mass DR), so the default path stays the
        # exact scalar arithmetic. See _sync_distal_mass.
        env._wbt_distal_mass = torch.full((n, h, 1), cfg.distal_mass_kg, device=dev)
        if cfg.sync_distal_mass_from_sim:
            self._sync_distal_mass()
        # Layer-B servo Jacobian index resolution (only when servo actuates — the C1
        # observe-only mode runs the integrator without a Jacobian or action term).
        if cfg.servo_enabled and not cfg.servo_observe_only:
            self._resolve_jacobian_indices()
            # Cache the joint-position action term: the servo injects its residual into that term's
            # `_actions_after_delay` (the ONLY tensor the real PD torque loop reads —
            # joint_control.py:159,188: torque = Kp·(actions·action_scales + default − dof_pos) − …).
            # Writing env.simulator.dof_pos_target does NOTHING (it exists only in recording.py).
            self._joint_term = env.action_manager.get_term("joint_control")
        # Monkey-patch the substep hook.
        self._original_apply_force = env._apply_force_in_physics_step

        def _patched():
            # ORDER MATTERS: when the Layer-B servo is on it must write the joint action term's
            # `_actions_after_delay` BEFORE the underlying PD runs, so `_original_apply_force()`
            # (= action_manager.apply_actions → the PD torque compute) is called LAST. Layer-A
            # (servo off) is behaviorally identical to the old order (nothing writes actions).
            self._advance_fcmd_pulse()  # substep 0: advance F^cmd (hybrid-owned)
            self._compute_contact_and_sensor()  # every substep: causal F^ext + F^meas
            if self.hc_cfg.servo_enabled:
                if self.hc_cfg.servo_observe_only:
                    self._observe_step()  # C1: I/Δx for obs+reward; NO action injection
                else:
                    # Cache the POLICY's committed `_actions_after_delay` at SUBSTEP 0 ONLY (before
                    # the servo first overwrites it), so every substep reconstructs actions = policy
                    # + servo shift rather than accumulating the servo's own prior writes. The joint
                    # term's `_substep_idx` is reset to 0 in process_actions() (once per env-step)
                    # and incremented in apply_actions(); it is 0 on the first substep here.
                    if self._joint_term._substep_idx == 0:
                        self._policy_actions = self._joint_term._actions_after_delay.clone()
                    self._servo_step()  # injects servo_dq into _actions_after_delay (reconstruct)
            self._inject_force_ext()  # every substep: write F^ext to PhysX
            self._original_apply_force()  # PD consumes the (servo-augmented) actions LAST

        env._apply_force_in_physics_step = _patched
        # NOTE: plane/K_env/bias are seeded on the FIRST real reset() (post first
        # physics step), NOT here — body_pos_w is not yet valid at setup time.

    def reset(self, env_ids) -> None:
        if env_ids is None:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.env.device)
        if env_ids.numel() == 0:
            return
        env, cfg = self.env, self.hc_cfg
        gen = env._wbt_force_state["gen"]
        h = len(self.body_names)
        m = env_ids.numel()
        # F^cmd pulse restart + F^cmd buffer zeroing for these envs (SAFE on the 6D
        # ext buffer: restart does a scalar `= 0.0` broadcast assign).
        restart_force_pulse_for_envs(env, env_ids, cfg=self.hf_cfg)
        # K_env log-uniform per (env, hand).
        log_lo, log_hi = torch.log(torch.tensor(cfg.k_env_min)), torch.log(torch.tensor(cfg.k_env_max))
        k = torch.exp(torch.rand((m, h), generator=gen) * (log_hi - log_lo) + log_lo)
        env._wbt_k_env[env_ids] = k.to(env.device)  # (m,h)

        # --- Well-posedness fix (2026-07-10): reachable-force clamp + F_cmd-along-normal. ---
        # (i) Clamp the per-episode F_cmd PEAK magnitude to what the sampled wall can react
        #     (k_env·max_penetration), so the force target is always physically attainable.
        fcmd_state = env._wbt_force_state["cmd"]
        f_target_b = fcmd_state["f_target"][env_ids]  # (m,h,3) root-frame peak command
        # Per-episode reachable-force cap (N): min(f_cmd_max, k_env·max_penetration). Stored so
        # _advance_fcmd_pulse can re-apply it after every _step_pulse (which auto-resamples).
        reachable = (
            (k.to(env.device) * cfg.max_penetration_m).clamp(max=cfg.f_cmd_max)
            if cfg.f_cmd_reachable_clamp
            else torch.full((m, h), cfg.f_cmd_max, device=env.device)
        )  # (m,h) N
        env._wbt_fcmd_reach[env_ids] = reachable
        if cfg.f_cmd_reachable_clamp:
            tnorm = f_target_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (m,h,1)
            scale = (reachable.unsqueeze(-1) / tnorm).clamp(max=1.0)  # never AMPLIFY, only cap
            f_target_b = f_target_b * scale
        # (ii) F_cmd-along-normal: orient the VIRTUAL PLANE so its outward normal = −push
        #      direction, i.e. the wall sits on the F_cmd side and reacts exactly along the
        #      commanded axis. Rotate F_cmd (root) → world via the reset root quat, then set
        #      normal = −unit(F_cmd_world). With all-3-axis force control this makes
        #      e=F_cmd+F_meas nullable.
        tcp0 = self._current_tcp_w()[env_ids]  # (m, h, 3) reset TCP
        rand_normal, _ = sample_plane(m, h, cfg, gen, env.device)  # random normal + (unused) dist
        dist = (torch.rand((m, h), generator=gen) * (cfg.plane_dist_max - cfg.plane_dist_min) + cfg.plane_dist_min).to(
            env.device
        )
        if cfg.f_cmd_along_normal:
            root_quat = env.simulator._robot.data.root_quat_w[env_ids].unsqueeze(1).expand(m, h, 4)  # wxyz
            f_target_w = _quat_apply_wxyz(root_quat, f_target_b.to(env.device))  # (m,h,3) world push dir
            fw_norm = f_target_w.norm(dim=-1, keepdim=True)
            # Fall back to the random normal for near-zero commands (direction undefined).
            push_dir = torch.where(fw_norm > 1e-4, f_target_w / fw_norm.clamp(min=1e-6), rand_normal)
            normal = -push_dir  # outward normal opposes the push; robot presses in −normal
        else:
            normal = rand_normal  # legacy random plane (only sound with 6-DOF reactive contact, v2)
        # Write the (possibly clamped) F_cmd target back, anchor the plane, mark seeded.
        fcmd_state["f_target"][env_ids] = f_target_b
        # EPISODE-FIXED F_cmd unit direction (root frame). This is the direction the plane normal
        # was anti-aligned to; _advance_fcmd_pulse re-projects every resampled f_target onto it so
        # the command never drifts off the (fixed) wall normal mid-episode. Near-zero commands get
        # a zero direction → _advance_fcmd_pulse leaves the (tiny) resampled vector untouched.
        f_dir_norm = f_target_b.norm(dim=-1, keepdim=True)
        env._wbt_fcmd_dir_b[env_ids] = torch.where(
            f_dir_norm > 1e-6, f_target_b / f_dir_norm.clamp(min=1e-6), torch.zeros_like(f_target_b)
        )
        # WORLD-fixed F_cmd direction = −plane_normal (the wall reacts along +normal, the robot
        # pushes along −normal). Episode-fixed in WORLD so it stays aligned with the world-fixed
        # wall even as the pelvis reorients (frames-review fix). For the along-normal path this is
        # exactly the (unit) push_dir; for the legacy random-plane path we still store −normal so
        # _advance_fcmd_pulse has a consistent world command axis.
        env._wbt_fcmd_dir_w[env_ids] = -normal
        env._wbt_plane_normal[env_ids] = normal
        env._wbt_plane_point[env_ids] = tcp0 - dist.unsqueeze(-1) * normal  # `dist` in front along −normal
        env._wbt_plane_seeded[env_ids] = True
        # 6D-ready DORMANT (§9.1 #3): build the full surface frame (normal + 2 tangents) and
        # sample the ROTATIONAL K_env. k_env_rot_min/max=0 in v1 → K_rot=0 → moment stays 0.
        t1, t2 = surface_tangents(normal)
        env._wbt_surface_t1[env_ids] = t1
        env._wbt_surface_t2[env_ids] = t2
        if cfg.k_env_rot_max > 0.0:  # 6D active (v2): log-uniform rotational stiffness
            rlog_lo, rlog_hi = torch.log(torch.tensor(cfg.k_env_rot_min)), torch.log(torch.tensor(cfg.k_env_rot_max))
            k_rot = torch.exp(torch.rand((m, h, 3), generator=gen) * (rlog_hi - rlog_lo) + rlog_lo)
            env._wbt_k_env_rot[env_ids] = k_rot.to(env.device)
        else:  # v1: dormant (0 → zero moment, path still exercised)
            env._wbt_k_env_rot[env_ids] = 0.0
        bias = torch.randn((m, h, 3), generator=gen) * cfg.sensor_bias_std
        env._wbt_sensor_bias_w[env_ids] = bias.to(env.device)
        # Zero the causal + servo buffers for reset envs.
        env._wbt_force_ext_w[env_ids] = 0.0
        env._wbt_force_meas_w[env_ids] = 0.0
        env._wbt_contact_gate[env_ids] = 0.0
        env._wbt_integ_state[env_ids] = 0.0
        env._wbt_delta_x[env_ids] = 0.0
        env._wbt_contact_phase[env_ids] = 0.0
        env._wbt_contact_hyst[env_ids] = 0.0
        env._wbt_force_ext_prev_w[env_ids] = 0.0

    def step(self) -> None:
        return  # pulse + contact advance on the substep hook, not here.

    def _sync_distal_mass(self) -> None:
        """Populate `env._wbt_distal_mass` (n,h,1) from the ACTUAL sim wrist-body masses.

        Called from setup() when `sync_distal_mass_from_sim` is set (hand-mass DR). The
        ORDERING that makes this work without any hook: base_task.py:164-170 runs
        `randomization_manager.setup()` (:166) BEFORE `command_manager.setup()` (:170), and
        `randomize_mass_startup` is a SETUP term — so by the time this injector's setup()
        runs, PhysX already holds the randomized masses and we can simply read them back.

        INDEXING — the one real footgun here, because two index spaces are in play:
          * CONFIG order  = `robot_config.body_names` (32 entries, IsaacGym/holosoma order)
          * PHYSX order   = IsaacSim's own DFS order; `sim.body_ids[cfg_i] = physx_i`
        `body_ids` is NOT the identity (see the worked example in isaacsim.py's
        `find_rigid_body_indice` docstring: config `[0,1,4,8,12,...]` ≠ PhysX
        `[0,1,2,3,4,...]`), so mixing the two silently reads a DIFFERENT body's mass.
        `self._isaac_body_ids` is in PHYSX space — `find_rigid_body_indice` returns a
        CONFIG index and `_ensure_force_isaac_indices` maps it back with `sim.body_ids[i]`
        — which is the same space `_robot.data.body_pos_w` and `root_physx_view` use. So
        these indices apply to `get_masses()` directly, and `test_tcp_precision.py::
        test_distal_mass_readback_uses_the_physx_index_space` pins that.

        (Do NOT copy `maskedmimic_observations._end_effector_mass_obs` here: it indexes
        `get_masses()` with CONFIG-space indices AND its targets `*_rubber_hand` are not in
        the 32-body config list at all, so its ids are [-1, -1] and that obs is identically
        zero. It is doubly broken and is not a precedent for anything — filed separately,
        out of scope for this change.)

        `get_masses()` returns a CPU tensor, hence the explicit `.to(env.device)`.

        If the view is unavailable (unit tests, non-IsaacSim backends) the nominal value
        allocated in setup() is left in place, which is the pre-DR behavior.
        """
        env = self.env
        robot = getattr(env.simulator, "_robot", None)
        view = getattr(robot, "root_physx_view", None) if robot is not None else None
        masses = getattr(view, "get_masses", None)
        if not callable(masses):
            # RAISE, do not warn. `sync_distal_mass_from_sim` is only ever set deliberately
            # (WBT_HAND_MASS_DR=1), and silently keeping the nominal m_distal under active
            # mass DR reintroduces exactly the ≤2.04 N sensor desync this method exists to
            # prevent — above contact_phase_theta_on=2.0 N, so it corrupts the contact label
            # rather than merely adding noise. In an 8-GPU run one logger.warning scrolls past
            # unseen and the whole run is quietly invalid; failing at setup costs seconds.
            raise RuntimeError(
                "hybrid: sync_distal_mass_from_sim is set (hand-mass DR) but "
                "simulator._robot.root_physx_view.get_masses() is unavailable, so m_distal "
                f"cannot be read back and would stay at the nominal {self.hc_cfg.distal_mass_kg} kg "
                "— up to 2.04 N of F_meas error at a=3 m/s², above contact_phase_theta_on. "
                "Use the IsaacSim backend, or unset WBT_HAND_MASS_DR to train without mass DR."
            )
        from wbt_training.config_values.tcp_precision import distal_mass_from_body_mass

        m_all = masses().to(env.device)  # (n_envs, n_bodies) PhysX order, CPU→device
        m_body = m_all[:, self._isaac_body_ids]  # (n, h)
        env._wbt_distal_mass = distal_mass_from_body_mass(m_body).unsqueeze(-1)  # (n,h,1)
        logger.info(
            f"hybrid: m_distal synced from sim — body mass "
            f"[{float(m_body.min()):.4f}, {float(m_body.max()):.4f}] kg → m_distal "
            f"[{float(env._wbt_distal_mass.min()):.4f}, {float(env._wbt_distal_mass.max()):.4f}] kg "
            f"(nominal {self.hc_cfg.distal_mass_kg})."
        )

    def _advance_fcmd_pulse(self) -> None:
        """Advance ONLY the F^cmd trapezoidal pulse once per env-step (substep 0),
        rotating cmd root->world. Hybrid-owned (does NOT call
        force_tracking._advance_force_pulses, which would write the 3D ext buffer
        into our 6D _wbt_force_ext_w and crash). Idempotent via last_step."""
        env, cfg = self.env, self.hf_cfg
        state = env._wbt_force_state
        cur_step = env.episode_length_buf
        if not bool((cur_step != state["last_step"]).any()):
            return
        # Capture last step's final clean F_ext for the force-rate reward (pure readers).
        # Runs exactly once per env step (post idempotence check), while the buffer still
        # holds the previous step's last-substep value.
        env._wbt_force_ext_prev_w[:] = env._wbt_force_ext_w[..., :3]
        alpha = float(env._wbt_force_alpha)
        cmd_b_raw = _step_pulse(state["cmd"], cfg, gen=state["gen"]) * alpha  # (n,h,3) root frame
        n, h = env.num_envs, len(self.body_names)
        root_quat = env.simulator._robot.data.root_quat_w  # (n,4) wxyz
        root_quat_h = root_quat.unsqueeze(1).expand(n, h, 4)
        # RE-CONFORM to the episode-fixed direction + reachable cap (Opus review 2026-07-10 fix):
        # _step_pulse auto-resamples f_target mid-episode to a fresh UNCLAMPED, RANDOM-direction
        # vector, which would drift F_cmd off the (fixed) wall normal + past the reachable force.
        # We keep only the trapezoidal MAGNITUDE and re-impose the episode-fixed direction + cap.
        #
        # FRAMES FIX (2026-07-10, Opus frames review): the wall/plane is WORLD-fixed, so the command
        # direction is stored+applied in WORLD (_wbt_fcmd_dir_w = −plane_normal), NOT re-rotated from
        # a root-frame direction by the LIVE root quat (which drifts off the world-fixed normal when
        # the pelvis reorients tracking the clip → unnullable tangential error). We build the WORLD
        # command from the world-fixed direction × the (magnitude) schedule, then derive the
        # pelvis-frame command (for the force_cmd_local obs, which reads _wbt_force_cmd_b) by rotating
        # world→root with the live root quat. Envs with a zero fixed-dir (near-zero reset command)
        # fall back to the raw root pulse (direction undefined — harmless, tiny magnitude).
        dir_w = env._wbt_fcmd_dir_w  # (n,h,3) unit, WORLD, episode-fixed (= −plane_normal)
        has_dir = dir_w.norm(dim=-1, keepdim=True) > 1e-6  # (n,h,1)
        mag = cmd_b_raw.norm(dim=-1, keepdim=True).clamp(max=env._wbt_fcmd_reach.unsqueeze(-1))  # capped |F|
        cmd_w_conformed = mag * dir_w  # world command along the fixed wall normal
        cmd_w_fallback = _quat_apply_wxyz(root_quat_h, cmd_b_raw)  # legacy: rotate raw root pulse → world
        env._wbt_force_cmd_w[:] = torch.where(has_dir, cmd_w_conformed, cmd_w_fallback)
        # Pelvis-frame command for the obs (force_cmd_local reads _wbt_force_cmd_b).
        env._wbt_force_cmd_b[:] = _quat_apply_inverse_wxyz(root_quat_h, env._wbt_force_cmd_w)
        state["last_step"] = cur_step.clone()

    def _substep_dt(self) -> float:
        """Physics substep dt (s) for the servo integrator + contact-phase LPF.

        Reads the REAL simulator attribute `sim_dt` (= 1/fps, set by every holosoma
        simulator backend) first; `physics_dt` kept for forward-compat; falls back to
        the historical hardcoded 0.005 (fps=200). isinstance-guarded so MagicMock
        child attributes don't masquerade as a dt (2026-07-24 review F2/F3 — the old
        getattr chain never matched a real attribute and float(MagicMock) == 1.0)."""
        sim = self.env.simulator
        for attr in ("sim_dt", "physics_dt"):
            v = getattr(sim, attr, None)
            if isinstance(v, (int, float)):
                return float(v)
        return 0.005

    def _current_tcp_w(self) -> torch.Tensor:
        """Live world-frame TCP via FK on wrist_yaw (fresh per substep, wxyz)."""
        env = self.env
        bpos = env.simulator._robot.data.body_pos_w[:, self._isaac_body_ids, :]
        bquat = env.simulator._robot.data.body_quat_w[:, self._isaac_body_ids, :]
        h = len(self.body_names)
        off = self._tcp_offsets_b.unsqueeze(0).expand(env.num_envs, h, -1)
        return bpos + _quat_apply_wxyz(bquat, off)

    def _compute_contact_and_sensor(self) -> None:
        env, cfg = self.env, self.hc_cfg
        x_tcp = self._current_tcp_w()
        depth = penetration_depth(x_tcp, env._wbt_plane_normal, env._wbt_plane_point, cfg.max_penetration_m)
        # Envs whose plane is not yet seeded (before their first reset) contribute no
        # contact — the plane buffers are still zero, and depth would be spurious.
        depth = depth * env._wbt_plane_seeded.unsqueeze(-1).float()
        f_ext = contact_force_world(env._wbt_k_env, depth, env._wbt_plane_normal, cfg.max_force_n)  # (n,h,3)
        env._wbt_force_ext_w[..., :3] = f_ext
        env._wbt_penetration[:] = depth  # store for the critic-only privileged obs
        env._wbt_contact_gate[:] = (depth > 0.0).float()
        # 6D-ready DORMANT moment path (§9.1 #3, D8): TCP-vs-surface orientation error on the
        # surface frame → contact moment M = K_env_rot ⊙ orient_err. With k_env_rot=0 (v1) this
        # is identically 0 — but the path is EXERCISED, so 6D = flip k_env_rot_*>0, not a rewrite.
        bquat = env.simulator._robot.data.body_quat_w[:, self._isaac_body_ids, :]
        tcp_axis_w = _quat_apply_wxyz(bquat, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand_as(f_ext))
        # attitude error components along the two surface tangents (0 when TCP axis ∥ normal):
        e_t1 = (tcp_axis_w * env._wbt_surface_t1).sum(-1, keepdim=True)
        e_t2 = (tcp_axis_w * env._wbt_surface_t2).sum(-1, keepdim=True)
        orient_err = (e_t1 * env._wbt_surface_t1 + e_t2 * env._wbt_surface_t2) * (depth > 0.0).unsqueeze(-1).float()
        moment = contact_moment_world(env._wbt_k_env_rot, orient_err, cfg.max_orientation_error_rad)  # 0 in v1
        env._wbt_force_ext_w[..., 3:] = moment  # dormant (K_rot=0) but written — the 6D seam is live
        # Simulated sensor: F_meas = F_ext + inertial + bias + noise.
        acc = env.simulator._robot.data.body_lin_acc_w[:, self._isaac_body_ids, :]
        gen = env._wbt_force_state["gen"]
        noise = (
            torch.randn(f_ext.shape, generator=gen).to(env.device) * cfg.sensor_noise_std
            if cfg.sensor_noise_std > 0.0
            else torch.zeros_like(f_ext)
        )
        # m_distal: the (n,h,1) buffer when it is a real tensor (nominal, or synced from the
        # randomized sim masses), else None → the scalar cfg.distal_mass_kg path. The
        # isinstance guard keeps MagicMock-backed unit tests on the scalar path.
        m_distal = env._wbt_distal_mass if isinstance(getattr(env, "_wbt_distal_mass", None), torch.Tensor) else None
        env._wbt_force_meas_w[..., :3] = simulated_measured_force(
            f_ext, acc, env._wbt_sensor_bias_w, cfg, noise, distal_mass=m_distal
        )
        # Soft contact phase (C1 unified): hysteresis on ‖F_meas‖ + first-order LPF (substep dt).
        # Deployable — driven by the MEASURED force (real Bota gives the same signal), never the
        # privileged depth gate. Updated every substep for every preset (buffers always exist;
        # existing presets just don't read them).
        dt = self._substep_dt()
        fmag = env._wbt_force_meas_w[..., :3].norm(dim=-1)
        hyst = env._wbt_contact_hyst
        hyst[:] = torch.where(
            fmag > cfg.contact_phase_theta_on,
            torch.ones_like(hyst),
            torch.where(fmag < cfg.contact_phase_theta_off, torch.zeros_like(hyst), hyst),
        )
        # Post-reset warmup: the reset teleport's inertial spike in F_meas (m·a ≫ θ_on)
        # must not flip the label ON in free space — force hysteresis OFF for the first
        # contact_phase_warmup_steps env steps of each episode (2026-07-24 review, F1).
        warm = (env.episode_length_buf >= cfg.contact_phase_warmup_steps).float().unsqueeze(-1)
        hyst *= warm
        beta = 1.0 - math.exp(-dt / max(cfg.contact_phase_tau_s, 1e-6))
        env._wbt_contact_phase += beta * (hyst - env._wbt_contact_phase)

    def _inject_force_ext(self) -> None:
        env = self.env
        robot = env.simulator._robot
        if not hasattr(robot, "set_external_force_and_torque"):
            return
        # World -> body-local (IsaacLab is_global=False), like HandForceInjector.
        bquat = robot.data.body_quat_w[:, self._isaac_body_ids, :]
        force_body = _quat_apply_inverse_wxyz(bquat, env._wbt_force_ext_w[..., :3])
        torques = torch.zeros_like(force_body)  # moment dormant (D8)
        positions = self._tcp_offsets_b.unsqueeze(0).expand(env.num_envs, -1, -1)
        env_ids = torch.arange(env.num_envs, device=env.device)
        robot.set_external_force_and_torque(
            forces=force_body, torques=torques, positions=positions, env_ids=env_ids, body_ids=self._isaac_body_ids
        )

    # --- Layer B: TCP Jacobian accessor (get_jacobians slice + remap + offset, Task 9). ---
    def _resolve_jacobian_indices(self):
        """Precompute (once at setup) the PhysX body idx + arm Jacobian columns per hand,
        and where the arm residual maps into the 29-DOF PD target. Floating base → +6.

        Grounded vs IsaacLab 2.3.0: get_jacobians() is world-aligned, link-origin, PhysX DFS
        order (≠ holosoma config order). Actuated joint j is at Jacobian COLUMN j+6 (floating
        base). Resolve the wrist body + 7 arm joints via find_bodies/find_joints (NOT the
        config idx). The PD/action vector IS in config (dof_names) order."""
        robot = self.env.simulator._robot
        # Per-hand wrist body idx (PhysX/DFS order, from find_bodies — NOT the config idx).
        self._jac_wrist_body_ids = [robot.find_bodies(n)[0][0] for n in self.body_names]
        # Per-hand 7 arm joint names (yam URDF): shoulder p/r/y, elbow, wrist r/p/y.
        arm_joint_names = {
            "left_wrist_yaw_link": [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ],
            "right_wrist_yaw_link": [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
        }
        self._jac_arm_cols = []  # per hand: PhysX joint id + 6 (floating base)
        self._arm_pd_slice = []  # per hand: index into the 29-DOF PD/action vector
        sim = self.env.simulator
        dof_names = list(sim.dof_names)
        for bn in self.body_names:
            names = arm_joint_names[bn]
            physx_ids = [robot.find_joints(nm, preserve_order=True)[0][0] for nm in names]
            self._jac_arm_cols.append([j + 6 for j in physx_ids])  # +6: floating base
            # PD/action order == config dof order; map joint name → its index in dof_names.
            self._arm_pd_slice.append([dof_names.index(nm) for nm in names])

    def _tcp_jacobian_world(self):
        """(n, h, 6, n_arm) world-aligned TCP Jacobian: get_jacobians slice + TCP offset shift.

        get_jacobians() → (n, n_bodies, 6, n_dof) world-aligned at the wrist link origin. Slice
        the wrist body row + the 7 arm columns, then shift the linear rows to the TCP:
        J_v += −[r]× J_ω with r = R_wrist_w @ (tcp_offset,0,0) in WORLD coords (rotate the
        constant body offset by the live wrist quat — the one place IsaacLab base-frame code
        must NOT be copied verbatim). Rotational rows are unchanged (offset_rot = I)."""
        robot = self.env.simulator._robot
        jac_full = robot.root_physx_view.get_jacobians()  # (n, n_bodies, 6, n_dof)
        n = self.env.num_envs
        h = len(self.body_names)
        na = len(self._jac_arm_cols[0])
        out = jac_full.new_zeros(n, h, 6, na)
        for hi, (bidx, cols) in enumerate(zip(self._jac_wrist_body_ids, self._jac_arm_cols)):
            j_wrist = jac_full[:, bidx][:, :, cols]  # (n, 6, na) at wrist_yaw link origin, world-aligned
            # Shift to TCP: J_v += -[r]× J_ω, r = R_wrist_w @ (tcp_offset,0,0) in WORLD frame.
            wrist_quat = robot.data.body_quat_w[:, self._isaac_body_ids[hi]]  # (n,4) wxyz
            r_local = torch.tensor([self.hc_cfg.tcp_offset_x, 0.0, 0.0], device=j_wrist.device).expand(n, 3)
            r_w = _quat_apply_wxyz(wrist_quat, r_local)  # (n,3) world
            j_ang = j_wrist[:, 3:, :]  # (n,3,na)
            # -[r]× J_ω : cross(-r, each angular column)
            j_lin_shift = torch.cross(-r_w.unsqueeze(-1).expand(n, 3, na), j_ang, dim=1)  # (n,3,na)
            out[:, hi, :3, :] = j_wrist[:, :3, :] + j_lin_shift
            out[:, hi, 3:, :] = j_ang  # rotational rows unchanged (offset_rot = I)
        return out

    # --- Layer B: servo substep step (fetch J → servo_residual_step → inject arm action shift). ---
    def _servo_step(self) -> None:
        """Fetch the TCP Jacobian, run the contact-gated force servo, and INJECT the arm residual
        into the joint action term's `_actions_after_delay` (the tensor the real PD torque loop
        reads), RECONSTRUCTED off the cached policy actions (assignment, NOT +=). Called from the
        substep hook AFTER _compute_contact_and_sensor and BEFORE _original_apply_force (so the PD
        consumes the servo-augmented actions). Only runs when servo_enabled."""
        env = self.env
        # dt: the servo runs on the SUBSTEP hook (control_decimation×/env-step), so the integrator
        # MUST use the substep dt (sim_dt≈5 ms), NOT dt_env_step (20 ms) — else I integrates 4×
        # too fast and desyncs from the Colgate K_I≤2b/T budget (design §9.3 "per-station dt").
        dt = self._substep_dt()
        j_tcp = self._tcp_jacobian_world()  # (n, h, 6, n_arm)
        dq_arm, new_integ, delta_x = servo_residual_step(
            j_tcp,
            env._wbt_force_cmd_w,
            env._wbt_force_meas_w[..., :3],
            env._wbt_contact_gate,
            env._wbt_integ_state,
            self.hc_cfg,
            dt=dt,
        )
        env._wbt_integ_state[:] = new_integ
        env._wbt_delta_x[:] = delta_x
        # Scatter the per-hand arm residual (rad of joint-target shift) into the 29-DOF servo-dq buffer.
        env._wbt_servo_dq.zero_()
        h, na = len(self.body_names), len(self._arm_pd_slice[0])
        dq_ph = dq_arm.reshape(env.num_envs, h, na)
        for hi, cols in enumerate(self._arm_pd_slice):
            env._wbt_servo_dq[:, cols] += dq_ph[:, hi, :]
        # INJECT into the joint-position action term's `_actions_after_delay` — the ONLY tensor the
        # real PD torque loop reads (joint_control.py:159,188: torque = Kp·(actions·action_scales +
        # default − dof_pos) − Kd·dof_vel). The residual servo_dq is a joint-target shift in RAD;
        # actions are scaled by `action_scales` before entering the PD, so to shift the joint target
        # by servo_dq[rad] we add servo_dq / action_scales to `_actions_after_delay`. Then the extra
        # PD torque is exactly Kp·servo_dq = Kp·k·Jᵀ·wrench, and with the unity-authority gain k≈1/Kp
        # that reproduces the intended joint torque Jᵀ·wrench (the servo_gain_k anchor holds).
        #
        # RECONSTRUCT (assignment off the cached policy actions), NOT `+=`: `_policy_actions` is the
        # committed `_actions_after_delay` cached at SUBSTEP 0 (before the servo first writes it), so
        # each substep re-derives actions = policy + servo_shift. A `+=` would ACCUMULATE the servo's
        # own prior writes across the 4 substeps → free-space drift/runaway, which desired-behavior
        # (1) forbids. NOTE: the action OBS/reward read `action_manager.action` (the raw policy
        # action), NOT `_actions_after_delay`, so the servo residual stays invisible to the policy's
        # action channel — it only enters the physical PD, exactly as intended.
        action_scales = env.action_scales  # (n_dof,) set by JointPositionActionTerm.setup (env attr)
        servo_action_shift = env._wbt_servo_dq / action_scales.clamp(min=1e-6)  # rad → action units
        # R1 servo-authority anneal (hybrid_unified.ServoAuthorityCurriculum owns
        # _wbt_servo_alpha; absent buffer = full authority — existing presets unchanged).
        alpha = getattr(env, "_wbt_servo_alpha", None)
        if isinstance(alpha, torch.Tensor):
            servo_action_shift = servo_action_shift * alpha
        self._joint_term._actions_after_delay[:] = self._policy_actions + servo_action_shift

    def _observe_step(self) -> None:
        """C1 observe-only path: run the contact-gated integrator (same dt discipline as
        the servo — substep physics_dt) and write I/Δx to the env buffers the obs terms
        + the servo-inclusive Eq.2 reward read. NEVER touches the action term or the
        Jacobian view (neither is resolved in this mode)."""
        env = self.env
        dt = self._substep_dt()
        new_integ, delta_x, _ = integrator_observe_step(
            env._wbt_force_cmd_w,
            env._wbt_force_meas_w[..., :3],
            env._wbt_contact_gate,
            env._wbt_integ_state,
            self.hc_cfg,
            dt=dt,
        )
        env._wbt_integ_state[:] = new_integ
        env._wbt_delta_x[:] = delta_x


# ---------------------------------------------------------------------------
# Observation readers (PURE readers of the injector's env-attached buffers).
# ---------------------------------------------------------------------------


def force_meas_local(env, *, cfg: "HybridContactConfig | None" = None) -> torch.Tensor:
    """Simulated F/T reading in the ROOT frame, force channels only (xyz active per
    D8), shape (num_envs, num_hands*3). Wired into BOTH the actor and the critic:
    the F/T sensor is present in sim (this simulated sensor) AND on the real robot
    (Bota), so F_meas is a legitimate policy input — the sensor-in-the-loop design
    (doc §0 pt 2 / §8.4), NOT a privileged-critic-only quantity. The noise+bias+
    inertial model lives in the injector so the actor sees a realistically noisy
    reading (sim2real). Rotates world→root via the live root quat (same convention
    as force_cmd_local, so the actor can implicitly form e=F_cmd+F_meas in one frame).
    PURE READER. The 3 moment channels ([...,3:]) are sliced off (dormant, D8).
    """
    f_meas_w = env._wbt_force_meas_w[..., :3]  # (n, h, 3)
    n, h = f_meas_w.shape[0], f_meas_w.shape[1]
    root_quat = env.simulator._robot.data.root_quat_w  # (n,4) wxyz
    root_quat_h = root_quat.unsqueeze(1).expand(n, h, 4)
    f_meas_b = _quat_apply_inverse_wxyz(root_quat_h, f_meas_w)  # world -> root
    return f_meas_b.reshape(n, -1)


def contact_gate(env, *, cfg: "HybridContactConfig | None" = None) -> torch.Tensor:
    """Critic-only privileged obs: per-hand contact flag (1 if penetrating), shape
    (num_envs, num_hands). PURE READER."""
    return env._wbt_contact_gate


def contact_phase(env, *, cfg: "HybridContactConfig | None" = None) -> torch.Tensor:
    """ACTOR-facing soft contact-phase label ∈[0,1], shape (num_envs, num_hands).
    Hysteresis(‖F_meas‖; θ_on/θ_off) → LPF(τ). DEPLOYABLE (measured force only —
    contrast contact_gate, the privileged critic-only depth>0 flag). PURE READER."""
    return env._wbt_contact_phase


def integrator_state(env, *, cfg: "HybridContactConfig | None" = None) -> torch.Tensor:
    """Servo integrator state I in the ROOT frame, shape (num_envs, num_hands*3).
    ACTOR + critic obs. Present at FIXED 165D width even when servo_enabled=False
    (then it reads the zeroed _wbt_integ_state buffer) — so obs width never keys off
    the servo flag (checkpoint/ONNX stability). Markov: the servo-inclusive reward
    target depends on this hidden integrator state, so the actor must observe it
    (decision #7d). PURE READER. Rotated world→root like force_meas_local."""
    integ_w = env._wbt_integ_state  # (n, h, 3) world
    n, h = integ_w.shape[0], integ_w.shape[1]
    root_quat_h = env.simulator._robot.data.root_quat_w.unsqueeze(1).expand(n, h, 4)
    return _quat_apply_inverse_wxyz(root_quat_h, integ_w).reshape(n, -1)


def x_tgt_shift(env, *, cfg: "HybridContactConfig | None" = None) -> torch.Tensor:
    """Servo task-space setpoint shift Δx in the ROOT frame, shape (num_envs, num_hands*3).
    ACTOR + critic obs, fixed width (zeroed when servo off). PURE READER."""
    dx_w = env._wbt_delta_x  # (n, h, 3) world
    n, h = dx_w.shape[0], dx_w.shape[1]
    root_quat_h = env.simulator._robot.data.root_quat_w.unsqueeze(1).expand(n, h, 4)
    return _quat_apply_inverse_wxyz(root_quat_h, dx_w).reshape(n, -1)


# --- CRITIC-ONLY privileged sim-only ground truth (design §8.4; added 2026-07-10 review). ---
# These read env parameters/state that do NOT exist on the real robot, so they are NEVER in the
# actor obs (would break sim2real). They sharpen the value function across the K_env sweep.
def force_ext_local(env, *, cfg: "HybridContactConfig | None" = None) -> torch.Tensor:
    """CLEAN causal F_ext (no D4 noise/bias/inertial) in the ROOT frame, force channels
    [:3], shape (num_envs, num_hands*3). Critic-only privileged. PURE READER. Contrast with
    force_meas_local, which is the NOISY reading the actor sees; the critic gets ground truth."""
    f_ext_w = env._wbt_force_ext_w[..., :3]  # (n, h, 3) world, clean
    n, h = f_ext_w.shape[0], f_ext_w.shape[1]
    root_quat_h = env.simulator._robot.data.root_quat_w.unsqueeze(1).expand(n, h, 4)
    return _quat_apply_inverse_wxyz(root_quat_h, f_ext_w).reshape(n, -1)


def k_env_privileged(env, *, cfg: "HybridContactConfig | None" = None) -> torch.Tensor:
    """Per-(env,hand) contact stiffness K_env (N/m), shape (num_envs, num_hands). The hidden
    environment parameter the actor cannot observe — lets the critic explain force outcomes
    across the log-uniform K_env randomization. Critic-only privileged. PURE READER."""
    return env._wbt_k_env


def penetration_privileged(env, *, cfg: "HybridContactConfig | None" = None) -> torch.Tensor:
    """Per-(env,hand) penetration depth (m), shape (num_envs, num_hands). Contact-strength
    ground truth (0 in free space). Critic-only privileged. PURE READER. Requires the injector
    to store the depth it computes each substep in `_wbt_penetration` (Task 4)."""
    return env._wbt_penetration


# ---------------------------------------------------------------------------
# Layer B: pure servo math — contact-gated integrator + Jᵀ residual (D2, §4.0).
# ---------------------------------------------------------------------------


def integrator_observe_step(f_cmd_w, f_meas_w, contact_gate, integ_state, cfg, dt):
    """Contact-gated force integrator, Jacobian-free (C1 observe-only core). PURE.

    The first half of servo_residual_step (§4.0 steps e→I→task_wrench→Δx), extracted so
    the observe-only injector path can run it without a Jacobian view. Same force-unit
    integral semantics (see servo_residual_step's docstring for the design rationale).
    Returns (new_integ (n,h,3) [N], delta_x (n,h,3) [m], task_wrench (n,h,3) [N])."""
    s = torch.tensor(cfg.servo_selection[:3], device=f_cmd_w.device, dtype=f_cmd_w.dtype)  # (3,)
    e = (f_cmd_w + f_meas_w) * s  # (n,h,3), force-error [N], moment axes already 0
    gate = contact_gate.unsqueeze(-1)  # (n,h,1)
    # FORCE-unit integral I_force [N] (gated + anti-windup), NOT a task-space displacement.
    new_integ = gate * torch.clamp(
        integ_state + cfg.integrator_gain_ki * e * dt, -cfg.integrator_max, cfg.integrator_max
    )
    # Desired task wrench = feedforward F_cmd + integral force offset (both N), +F_cmd sign.
    task_wrench = s * (f_cmd_w + new_integ)  # (n,h,3) N — the force we want the PD spring to make
    # Task-space setpoint shift Δx [m] for the obs/reward layer (servo-inclusive Eq.2 target).
    delta_x = task_wrench / cfg.stiffness_K  # (n,h,3) m
    return new_integ, delta_x, task_wrench


def servo_residual_step(j_tcp_w, f_cmd_w, f_meas_w, contact_gate, integ_state, cfg, dt):
    """Contact-gated force-servo step (design §4.0). PURE (tensors only).

    j_tcp_w: (n,h,6,n_arm) world-aligned TCP Jacobian (rows [0:3]=linear, [3:6]=angular).
    Returns (dq_arm (n, h*n_arm), new_integ (n,h,3) in FORCE units [N], delta_x (n,h,3) [m]).
    Force channels only (xyz, D8); moment axes masked by cfg.servo_selection[3:]==0.

    Math (world frame, force channels only) — FORCE-UNIT integral (2026-07-10 design change):
        e          = S ⊙ (F_cmd + F_meas)                       # (n,h,3) N, steady-state → 0
        I_force    = gate · clip(I + K_I·e·dt, −I_max, +I_max)   # (n,h,3) N, gated anti-windup
        task_wrench = S ⊙ (F_cmd + I_force)                      # (n,h,3) N, the force the PD should make
        Δq_arm     = clip(k · Jᵀ[:3]·task_wrench, ±dq_max)       # (n,h,n_arm) rad
        Δx         = task_wrench / K                             # (n,h,3) m — obs/reward setpoint shift only

    WHY force units (vs the old middle form task_wrench=K·Δx, Δx=F_cmd/K+I): accumulating the
    integral as a FORCE offset removes the guessed `stiffness_K` from the integral-path loop gain
    (loop gain = k·K_eff·J²·K_I, independent of the K guess), so steady-state force reaches F_cmd
    K-independently AND stability decouples from the K guess. The old form's loop gain ∝ stiffness_K
    made "sweep the K guess" a stability sweep that diverged at the high end (verified by scalar sim).
    `stiffness_K` now only maps task_wrench → the task-space setpoint shift Δx for the obs/reward
    (units: I_force [N] / K [N/m] = m, dimensionally sound).

    Sign (the §4.0 divergence footgun): task_wrench along +F_cmd (feedforward + integral SAME sign)
    so pressing harder extends the setpoint into the wall → negative feedback. gate=0 freezes I.

    Integrator/Δx math delegated to integrator_observe_step (C1 extraction, 2026-07-22) —
    the two paths are behavior-identical by construction.
    """
    new_integ, delta_x, task_wrench = integrator_observe_step(f_cmd_w, f_meas_w, contact_gate, integ_state, cfg, dt)
    # Map the desired wrench to joints via Jᵀ (linear rows only).
    jp = j_tcp_w[..., :3, :]  # (n,h,3,n_arm) linear rows
    dq_per_hand = cfg.servo_gain_k * torch.einsum("nhca,nhc->nha", jp, task_wrench)  # Jᵀ·wrench
    dq_per_hand = torch.clamp(dq_per_hand, -cfg.servo_dq_max, cfg.servo_dq_max)  # (n,h,n_arm)
    n = f_cmd_w.shape[0]
    dq_arm = dq_per_hand.reshape(n, -1)  # (n, h*n_arm)
    return dq_arm, new_integ, delta_x


# ---------------------------------------------------------------------------
# Eq.2 tracking reward for the hybrid preset (slices the 6D causal F_ext to [:3]).
# ---------------------------------------------------------------------------
from wbt_training.config_values.force_tracking import _force_cur_pos_w, _force_ref_pos_w  # noqa: E402


def force_track_pos_eq2_hybrid(env, *, cfg: "HandForceConfig | None" = None) -> torch.Tensor:
    """Eq.2 tracking reward for the hybrid preset. Identical to
    force_tracking.force_track_pos_eq2 but slices the 6-D causal F_ext buffer to its
    force channels [:3] before the target shift (moment dormant, D8). PURE READER."""
    cfg = cfg or HandForceConfig()
    ref_w = _force_ref_pos_w(env, cfg)
    f_total_w = env._wbt_force_cmd_w + env._wbt_force_ext_w[..., :3]  # 3D force only
    target_w = ref_w + f_total_w / cfg.stiffness_K
    cur_w = _force_cur_pos_w(env, cfg)
    err_l1 = (cur_w - target_w).abs().sum(dim=(-2, -1))
    return torch.exp(-err_l1 * 2.0 / cfg.sigma)


def force_track_eq2_servo(env, *, cfg: "HandForceConfig | None" = None) -> torch.Tensor:
    """Servo-inclusive Eq.2 tracking reward (Layer B, Task 11). When the servo is ON, the
    task-space setpoint is shifted by the servo's Δx (which folds in the contact-gated integral),
    so the target is ref + Δx (NOT ref + (F_cmd+F_ext)/K). When OFF, delegates to the Layer-A
    force_track_pos_eq2_hybrid (which slices the 6-D F_ext to [:3] — the dormant-moment footgun).
    PURE READER. Branch keyed off env._wbt_hybrid_cfg.servo_enabled."""
    cfg = cfg or HandForceConfig()
    hc = getattr(env, "_wbt_hybrid_cfg", None)
    if hc is None or not hc.servo_enabled:
        return force_track_pos_eq2_hybrid(env, cfg=cfg)
    ref_w = _force_ref_pos_w(env, cfg)
    target_w = ref_w + env._wbt_delta_x  # servo-inclusive setpoint shift (m), already xyz
    cur_w = _force_cur_pos_w(env, cfg)
    err_l1 = (cur_w - target_w).abs().sum(dim=(-2, -1))
    return torch.exp(-err_l1 * 2.0 / cfg.sigma)


def force_error_reward_preintegrator(env, *, cfg: "HybridContactConfig | None" = None) -> torch.Tensor:
    """Direct force-tracking reward on the policy's PRE-INTEGRATOR contribution (anti-free-ride,
    design §4.A). The contact-gated integrator drives e=F_cmd+F_meas→0 regardless of policy
    quality, so a naive exp(−‖e‖/σ) saturates and the policy free-rides. We ADD BACK the
    integrator's force contribution so the reward reflects only what the POLICY realized:

        e_pre = (F_cmd_w + F_meas_w[:3]) + I_force          # I_force is the force-unit integral (N)
        r     = exp(−‖e_pre‖·2 / σ_F)

    At the servo fixed point F_meas shifts by −I_force, so e_pre is INVARIANT to the integrator
    (the reward credits the policy, not the patch). σ_F reuses f_cmd_max as the natural force
    scale. Slices _wbt_force_meas_w[..., :3] (moment dormant, D8). PURE READER."""
    cfg = cfg or HybridContactConfig()
    e = env._wbt_force_cmd_w + env._wbt_force_meas_w[..., :3]  # (n,h,3) N, post-contact error
    e_pre = e + env._wbt_integ_state  # add back the integrator's force offset (N) → policy-only error
    err_l1 = e_pre.abs().sum(dim=(-2, -1))  # (n,)
    sigma_f = cfg.f_cmd_max  # N, natural force scale
    return torch.exp(-err_l1 * 2.0 / sigma_f)


def force_runaway(env, *, cfg: "HybridContactConfig | None" = None) -> torch.Tensor:
    """Force-runaway estop termination (design §7.9): ‖F_ext‖ (the CONTACT force) above the
    cutoff on ANY hand → terminate that env. Returns a (num_envs,) bool done flag. PURE READER.

    Keys off F_ext, NOT F_meas — 2026-07-10 fix. F_meas = F_ext + m_distal·a_wrist + bias +
    noise, so a free-space wrist-acceleration transient (especially the reset teleport, which
    spikes body_lin_acc_w) drives ‖F_meas‖ well past the 40 N cutoff even with ZERO contact
    (40 N / 1.5 kg = 26.7 m/s², trivially exceeded). The old ‖F_meas‖-based estop therefore
    fired at ~every reset, pinning ep_len at ~1.2 (A/B-proven 12.5× vs estop-off on the first
    8-GPU run). The estop's intent is "contact force ran away", so it belongs on F_ext. In sim
    F_ext is hard-clamped to max_force_n < cutoff, so this is dormant in sim (the clamp is the
    real sim guard) and acts purely as a deploy-safety valve on the unclamped real-Bota reading."""
    cfg = cfg or HybridContactConfig()
    fext_norm = env._wbt_force_ext_w[..., :3].norm(dim=-1)  # (n, h)
    return (fext_norm > cfg.force_runaway_cutoff).any(dim=-1)


# ---------------------------------------------------------------------------
# Wired-up cfg objects: command, observation, reward, curriculum.
# ---------------------------------------------------------------------------
from dataclasses import replace as _replace  # noqa: E402

from holosoma.config_types.command import CommandTermCfg  # noqa: E402
from holosoma.config_types.curriculum import CurriculumTermCfg  # noqa: E402
from holosoma.config_types.observation import ObsGroupCfg, ObservationManagerCfg, ObsTermCfg  # noqa: E402
from holosoma.config_types.reward import RewardTermCfg  # noqa: E402
from holosoma.config_values.wbt.g1.curriculum import g1_29dof_wbt_curriculum  # noqa: E402
from holosoma.config_values.wbt.g1.observation import critic_obs_shared_terms  # noqa: E402
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward as _core_reward  # noqa: E402

from wbt_training.config_values.command import g1_29dof_wbt_command as _core_command  # noqa: E402
from wbt_training.config_values.force_tracking import (  # noqa: E402, F401
    ForceAlphaCurriculum,
    force_cmd_local,  # referenced by dotted path
)
from wbt_training.config_values.observation import _actor_obs_2pt_terms  # noqa: E402

# NOTE (2026-07-10): this ships servo-OFF through Tasks 6–7 so the Layer-A milestone is
# launchable BEFORE the Task 8–10 servo code exists. Task 11's FINAL step flips this to
# `HybridContactConfig(servo_enabled=True)` — the one line that makes the SHIPPED preset do
# force tracking on launch (user decision). Obs is fixed 165D either way.
_DEFAULT_HYBRID_CFG = HybridContactConfig(servo_enabled=True)  # SHIPPED servo-ON (Task 11, user 2026-07-10)
# HandForceConfig for the force_cmd_local obs / Eq.2 reward / ForceAlphaCurriculum
# (must share the SAME body_names + stiffness as the injector).
_DEFAULT_HAND_FORCE_CFG = _DEFAULT_HYBRID_CFG.hand_force_config()


# --- Command: motion_command (inherited) + hybrid_contact_injector (new) ---
_hybrid_injector_term = CommandTermCfg(
    func="wbt_training.config_values.hybrid:HybridContactInjector", params={"hybrid_config": _DEFAULT_HYBRID_CFG}
)
g1_29dof_wbt_hybrid_command = _replace(
    _core_command,
    setup_terms={
        "motion_command": _core_command.setup_terms["motion_command"],
        "hybrid_contact_injector": _hybrid_injector_term,
    },
    reset_terms={**_core_command.reset_terms, "hybrid_contact_injector": _hybrid_injector_term},
)


# --- Observation: actor sees F^cmd, F^meas (sensor in the loop), AND the servo state
# integrator_state (I) + x_tgt_shift (Δx); critic adds the same + privileged contact_gate.
#
# ⚠️ FIXED WIDTH — ALWAYS 165D: the servo obs terms (I, Δx) are present in EVERY run, ZEROED
# when servo_enabled=False (buffers allocated unconditionally in the injector setup, Task 4).
# obs width does NOT key off servo_enabled — a config-bool-dependent width would break
# checkpoint/ONNX loading and orphan a Layer-A baseline. Markov note: I/Δx MUST be in obs
# because the servo-inclusive reward target depends on the hidden integrator state.
#
# F^meas obs noise=0.0 (the D4 sensor noise/bias/inertial is already in the buffer); scale 0.04
# matches force_cmd_local. F^meas right after F^cmd, then I/Δx.
#
# ⚠️ obs prefixes are ZERO-PADDED two-digit (obs00..obs12). The obs manager concatenates in
# `sorted(keys())` ALPHABETICAL order (manager.py:141), NOT insertion order; a 1-digit scheme
# string-sorts "obs10" between "obs1" and "obs2" and scrambles the vector.
_actor_obs_2pt_hybrid_terms = {
    "obs00_command_lower_body": _actor_obs_2pt_terms["obs0_command_lower_body"],
    "obs01_vr_2point_pos": _actor_obs_2pt_terms["obs1_vr_2point_pos"],
    "obs02_vr_2point_orn": _actor_obs_2pt_terms["obs2_vr_2point_orn"],
    "obs03_force_cmd_local": ObsTermCfg(
        func="wbt_training.config_values.force_tracking:force_cmd_local", scale=0.04, noise=0.0
    ),
    "obs04_force_meas_local": ObsTermCfg(
        func="wbt_training.config_values.hybrid:force_meas_local", scale=0.04, noise=0.0
    ),
    "obs05_integrator_state": ObsTermCfg(
        func="wbt_training.config_values.hybrid:integrator_state", scale=1.0, noise=0.0
    ),
    "obs06_x_tgt_shift": ObsTermCfg(func="wbt_training.config_values.hybrid:x_tgt_shift", scale=1.0, noise=0.0),
    "obs07_motion_ref_ori_b": _actor_obs_2pt_terms["obs3_motion_ref_ori_b"],
    "obs08_projected_gravity": _actor_obs_2pt_terms["obs4_projected_gravity"],
    "obs09_base_ang_vel": _actor_obs_2pt_terms["obs5_base_ang_vel"],
    "obs10_dof_pos": _actor_obs_2pt_terms["obs6_dof_pos"],
    "obs11_dof_vel": _actor_obs_2pt_terms["obs7_dof_vel"],
    "obs12_actions": _actor_obs_2pt_terms["obs8_actions"],
}  # 141 + 6 (F^cmd) + 6 (F^meas) + 6 (I) + 6 (Δx) = 165D actor (660D w/ history_length=4)

_critic_obs_2pt_hybrid_terms = {
    **critic_obs_shared_terms,
    "force_cmd_local": ObsTermCfg(
        func="wbt_training.config_values.force_tracking:force_cmd_local", scale=0.04, noise=0.0
    ),
    "integrator_state": ObsTermCfg(func="wbt_training.config_values.hybrid:integrator_state", scale=1.0, noise=0.0),
    "x_tgt_shift": ObsTermCfg(func="wbt_training.config_values.hybrid:x_tgt_shift", scale=1.0, noise=0.0),
    "force_meas_local": ObsTermCfg(func="wbt_training.config_values.hybrid:force_meas_local", scale=0.04, noise=0.0),
    "contact_gate": ObsTermCfg(func="wbt_training.config_values.hybrid:contact_gate", scale=1.0, noise=0.0),
    # Privileged sim-only ground truth (design §8.4): critic-only (never in the actor; they do
    # not exist at deploy): clean F_ext, hidden K_env, penetration depth.
    "force_ext_local": ObsTermCfg(func="wbt_training.config_values.hybrid:force_ext_local", scale=0.04, noise=0.0),
    "k_env_privileged": ObsTermCfg(func="wbt_training.config_values.hybrid:k_env_privileged", scale=0.001, noise=0.0),
    "penetration_privileged": ObsTermCfg(
        func="wbt_training.config_values.hybrid:penetration_privileged", scale=10.0, noise=0.0
    ),
}
g1_29dof_wbt_observation_2pt_hybrid = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_2pt_hybrid_terms
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=10, terms=_critic_obs_2pt_hybrid_terms
        ),
    }
)


# --- Reward: core 2pt reward + BALANCED force (torso-stable rebalance, decision #2) ---
# Per-term transforms applied to the inherited core terms:
#   - drop  motion_relative_body_position_error_exp   (Eq.2 owns the wrist position)
#   - bump  motion_relative_body_orientation_error_exp  weight 1.0 -> 2.0 (torso posture)
#   - tighten motion_global_body_ang_vel  sigma 3.14 -> 2.0 (torso ang-vel damping)
_HYBRID_REWARD_TRANSFORMS = {
    "motion_relative_body_orientation_error_exp": lambda v: _replace(v, weight=2.0),
    "motion_global_body_ang_vel": lambda v: _replace(v, params={**v.params, "sigma": 2.0}),
}
g1_29dof_wbt_2pt_hybrid_reward = _replace(
    _core_reward,
    terms={
        **{
            k: _HYBRID_REWARD_TRANSFORMS.get(k, lambda v: v)(v)
            for k, v in _core_reward.terms.items()
            if k != "motion_relative_body_position_error_exp"
        },
        # Wrist position owned by Eq.2 (no-wrist swap, same as 2pt-force).
        "motion_relative_body_position_error_exp_no_wrist": RewardTermCfg(
            func="wbt_training.config_values.force_tracking:motion_relative_body_position_error_exp_no_wrist",
            params={"sigma": 0.3, "exclude_body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link")},
            weight=1.0,
        ),
        # Eq.2 force tracking — weight 3.0 (HALF of 2pt-force's 6.0). Servo-inclusive target
        # (ref + Δx when servo on, else ref + (F_cmd+F_ext)/K) via force_track_eq2_servo; both
        # paths slice _wbt_force_ext_w[:3]. Dict key kept "force_track_pos_eq2" (Task-6 test).
        "force_track_pos_eq2": RewardTermCfg(
            func="wbt_training.config_values.hybrid:force_track_eq2_servo",
            weight=3.0,
            params={"cfg": _DEFAULT_HAND_FORCE_CFG},
        ),
        # Pre-integrator DIRECT force reward (anti-free-ride §4.A): credits the policy's own
        # feedforward force, invariant to the integrator's patch. Small positive weight.
        "force_error_preintegrator": RewardTermCfg(
            func="wbt_training.config_values.hybrid:force_error_reward_preintegrator",
            weight=1.0,
            params={"cfg": _DEFAULT_HYBRID_CFG},
        ),
        "feet_slip": RewardTermCfg(
            func="wbt_training.config_values.force_tracking:feet_slip",
            weight=-2.0,
            params={"body_names": ("left_ankle_roll_link", "right_ankle_roll_link")},
        ),
    },
)


# --- Curriculum: core + ForceAlphaCurriculum (F^cmd alpha; disabled by default) ---
_force_alpha_term = CurriculumTermCfg(
    func="wbt_training.config_values.force_tracking:ForceAlphaCurriculum",
    params={"hand_force_config": _DEFAULT_HAND_FORCE_CFG},
)
g1_29dof_wbt_hybrid_curriculum = _replace(
    g1_29dof_wbt_curriculum,
    setup_terms={**g1_29dof_wbt_curriculum.setup_terms, "force_alpha": _force_alpha_term},
    reset_terms={**g1_29dof_wbt_curriculum.reset_terms, "force_alpha": _force_alpha_term},
    step_terms={**g1_29dof_wbt_curriculum.step_terms, "force_alpha": _force_alpha_term},
)


# --- Termination: core (timeout + bad_tracking) + force_runaway estop (Layer B, §7.9) ---
from holosoma.config_types.termination import TerminationTermCfg  # noqa: E402
from holosoma.config_values.wbt.g1.termination import g1_29dof_wbt_termination as _core_termination  # noqa: E402

g1_29dof_wbt_hybrid_termination = _replace(
    _core_termination,
    terms={
        **_core_termination.terms,
        # ‖F_ext‖ > cutoff on any hand → terminate (not a timeout; a real bad-state done).
        "force_runaway": TerminationTermCfg(
            func="wbt_training.config_values.hybrid:force_runaway", params={"cfg": _DEFAULT_HYBRID_CFG}
        ),
    },
)


# ---------------------------------------------------------------------------
# PHASE-1 servo-OFF variant for the two-phase curriculum (user 2026-07-10):
# "先训练好 position tracking，再加上 force tracking (servo)". Phase 1 trains the
# stand + position tracking with servo_enabled=False (F_cmd/F_meas still in obs so
# the network sees the sensor channels, and the obs stays FIXED 165D); Phase 2
# resumes the SAME checkpoint under the servo-ON preset (`--training.checkpoint …`),
# where the ONLY change is servo_enabled → True. Because obs width is identical in
# both, the actor/critic weights transfer with no surgery.
#
# WHY a separate preset (not a CLI flag): the injector config lives inside
# CommandTermCfg.params (a dict[str, Any]) — tyro treats that value as opaque, so
# `--command…params.cfg.servo-enabled False` CANNOT reach it. A registered preset
# is the only way to flip it.
# ---------------------------------------------------------------------------
_HYBRID_CFG_SERVO_OFF = _replace(_DEFAULT_HYBRID_CFG, servo_enabled=False)
_HAND_FORCE_CFG_SERVO_OFF = _HYBRID_CFG_SERVO_OFF.hand_force_config()

_hybrid_injector_term_servo_off = CommandTermCfg(
    func="wbt_training.config_values.hybrid:HybridContactInjector", params={"hybrid_config": _HYBRID_CFG_SERVO_OFF}
)
g1_29dof_wbt_hybrid_command_servo_off = _replace(
    _core_command,
    setup_terms={
        "motion_command": _core_command.setup_terms["motion_command"],
        "hybrid_contact_injector": _hybrid_injector_term_servo_off,
    },
    reset_terms={**_core_command.reset_terms, "hybrid_contact_injector": _hybrid_injector_term_servo_off},
)

# Same reward terms/weights; only the two cfg-carriers point at the servo-off config so the
# preintegrator reward + Eq.2 servo-branch see servo_enabled=False (Eq.2 then delegates to the
# Layer-A position path). Reward KEYS stay identical → checkpoint/critic lineage intact.
g1_29dof_wbt_2pt_hybrid_reward_servo_off = _replace(
    g1_29dof_wbt_2pt_hybrid_reward,
    terms={
        **g1_29dof_wbt_2pt_hybrid_reward.terms,
        "force_track_pos_eq2": _replace(
            g1_29dof_wbt_2pt_hybrid_reward.terms["force_track_pos_eq2"], params={"cfg": _HAND_FORCE_CFG_SERVO_OFF}
        ),
        "force_error_preintegrator": _replace(
            g1_29dof_wbt_2pt_hybrid_reward.terms["force_error_preintegrator"], params={"cfg": _HYBRID_CFG_SERVO_OFF}
        ),
    },
)

# Estop still wired (dormant in sim — F_ext ≤ max_force_n < cutoff — but keeps the termination
# manager identical across phases so nothing else shifts).
g1_29dof_wbt_hybrid_termination_servo_off = _replace(
    _core_termination,
    terms={
        **_core_termination.terms,
        "force_runaway": TerminationTermCfg(
            func="wbt_training.config_values.hybrid:force_runaway", params={"cfg": _HYBRID_CFG_SERVO_OFF}
        ),
    },
)
