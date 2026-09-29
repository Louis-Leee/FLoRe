"""C1 unified single-policy preset pieces for `exp:g1-wbt-3pt-hybrid-global-vel-unified`.

Design: docs/superpowers/specs/2026-07-22-unified-hybrid-policy-design.md. The Layer-B
integrator becomes a REFERENCE GENERATOR (hybrid.py servo_observe_only=True): it keeps
computing I/Δx into the obs + the servo-inclusive Eq.2 target, but injects nothing —
the policy is the executor. This module adds ONLY the genuinely-new pieces: the four
unified reward terms, the 185D actor obs (183D global-vel + 2D contact_phase), the
ServoAuthorityCurriculum (R1 anneal twin), and the wired cfg objects for BOTH presets:

  * R2 mainline `exp:g1-wbt-3pt-hybrid-global-vel-unified` — observe-only integrator +
    force-alpha ramp from 0 ("position-first" happens inside ONE run);
  * R1 anneal twin `...-anneal` — full Jᵀ servo whose ACTUATION authority anneals 1→0
    performance-gated (JSRL/RAJS); terminal config ≡ R2's, so the runs A/B directly.

LOCAL twins `exp:g1-wbt-3pt-hybrid-unified[-anneal]` (user 2026-07-24): the SAME R2/R1
recipe on the ORIGINAL pelvis-local 168D 3pt-hybrid base — no root pos/vel trio,
quaternion vr_3point_orn. Because the whole unified command/reward/curriculum/
termination stack is tracking-point- AND obs-agnostic (it reads the `_wbt_*` buffers,
never the obs layout), the local twins reuse those carrier objects VERBATIM; the only
new piece is the 170D actor obs `g1_29dof_wbt_observation_3pt_unified_hybrid` (168D +
contact_phase). Mirrors how `exp:g1-wbt-3pt-hybrid` itself reuses hybrid.py's carriers.

hybrid.py owns the buffers/single-writer injector; every function here is a PURE READER.
"""

from __future__ import annotations

from dataclasses import replace as _replace

import torch

from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.curriculum import CurriculumTermCfg
from holosoma.config_types.observation import ObsGroupCfg, ObservationManagerCfg, ObsTermCfg
from holosoma.config_types.reward import RewardTermCfg
from holosoma.config_types.termination import TerminationTermCfg
from holosoma.config_values.wbt.g1.termination import g1_29dof_wbt_termination as _core_termination
from holosoma.managers.curriculum.base import CurriculumTermBase

from wbt_training.config_values.command import g1_29dof_wbt_command as _core_command
from wbt_training.config_values.force_tracking import _force_cur_pos_w
from wbt_training.config_values.hybrid import (
    _DEFAULT_HYBRID_CFG,
    HybridContactConfig,
    g1_29dof_wbt_2pt_hybrid_reward,
    g1_29dof_wbt_hybrid_curriculum,
    g1_29dof_wbt_observation_2pt_hybrid,
)
from wbt_training.config_values.hybrid_3pt import _actor_obs_3pt_hybrid_terms
from wbt_training.config_values.hybrid_3pt_global_vel import _actor_obs_3pt_hybrid_global_vel_terms


# ---------------------------------------------------------------------------
# Unified reward terms (spec §4). All gated by the SOFT contact_phase c∈[0,1]
# (deployable, hysteresis+LPF) — never a hard step at the contact boundary.
# Reward reads CLEAN F_ext (noise asymmetry: noisy obs, clean reward — FORGE).
# ---------------------------------------------------------------------------
def force_error_direct(env, *, cfg: "HybridContactConfig | None" = None) -> torch.Tensor:
    """Contact- AND command-gated direct force-tracking reward:
    c · I(‖F_cmd‖>0) · exp(−‖F_cmd+F_ext‖₁·2/σ_F) per hand, mean over hands. Replaces
    force_error_preintegrator (observe-only mode has no servo to free-ride on). Clean
    F_ext, not noisy F_meas (reward asymmetry). The cmd_on gate (2026-07-24 review:
    idle-phase farming) keeps F_cmd=0 phases pure position tracking — without it, a
    light 2 N brush during the pulse's idle window is NET-POSITIVE for near walls
    (income ≈1.0/step vs Eq.2 cost 0.375 @ 2 cm). PURE READER."""
    cfg = cfg or HybridContactConfig()
    e = env._wbt_force_cmd_w + env._wbt_force_ext_w[..., :3]  # (n,h,3) N, clean
    r = torch.exp(-e.abs().sum(dim=-1) * 2.0 / cfg.f_cmd_max)  # (n,h)
    cmd_on = (env._wbt_force_cmd_w.norm(dim=-1) > 1e-3).float()  # (n,h)
    return (env._wbt_contact_phase * cmd_on * r).mean(dim=-1)


def contact_init_bonus(env, *, cfg: "HybridContactConfig | None" = None) -> torch.Tensor:
    """Anti-contact-shyness: pay soft COMMANDED contact below the force ceiling
    (max_force_n): c · I(‖F_cmd‖>0) · I(‖F_meas‖ < ceiling), mean over hands. Pays
    CONTINUOUSLY while in contact (per spec §4 — the c-weighting is the load-bearing
    anti-bounce mechanism: holding keeps c≈1 while touch-release averages c<1; do NOT
    convert to an initiation impulse). The cmd_on gate (2026-07-24 review) stops the
    idle-phase (F_cmd=0) farming loop — anti-shyness is only needed when a force task
    exists, mirroring approach_guidance. Ceiling on MEASURED force (conservative:
    trips below the 40 N F_ext estop). PURE READER."""
    cfg = cfg or HybridContactConfig()
    fmag = env._wbt_force_meas_w[..., :3].norm(dim=-1)  # (n,h)
    ok = (fmag < cfg.max_force_n).float()
    cmd_on = (env._wbt_force_cmd_w.norm(dim=-1) > 1e-3).float()  # (n,h)
    return (env._wbt_contact_phase * cmd_on * ok).mean(dim=-1)


def approach_guidance(env, *, cfg: "HybridContactConfig | None" = None, d0: float = 0.05) -> torch.Tensor:
    """Pre-contact wall-approach shaping: (1−c) · I(‖F_cmd‖>0) · exp(−d_wall/d0), mean
    over hands. Active only while a force command exists but contact hasn't started —
    the other half of the anti-shyness pair. d_wall = free-side TCP distance to the
    per-episode virtual plane (sim-only geometry is fine in a REWARD). PURE READER."""
    cfg = cfg or HybridContactConfig()
    cur_w = _force_cur_pos_w(env, cfg.hand_force_config())  # (n,h,3) live TCP
    d = ((cur_w - env._wbt_plane_point) * env._wbt_plane_normal).sum(dim=-1).clamp(min=0.0)
    seeded = env._wbt_plane_seeded.unsqueeze(-1).float()
    cmd_on = (env._wbt_force_cmd_w.norm(dim=-1) > 1e-3).float()
    r = torch.exp(-d / d0)
    return ((1.0 - env._wbt_contact_phase) * cmd_on * seeded * r).mean(dim=-1)


def force_rate_penalty(env, *, cfg: "HybridContactConfig | None" = None) -> torch.Tensor:
    """Contact-transient smoothness: ‖ΔF_ext‖₁ per env step, summed over hands, /f_cmd_max.
    Applied with NEGATIVE weight. Clean F_ext vs its previous env-step snapshot
    (_wbt_force_ext_prev_w) so the sensor noise floor doesn't pollute it. PURE READER."""
    cfg = cfg or HybridContactConfig()
    df = (env._wbt_force_ext_w[..., :3] - env._wbt_force_ext_prev_w).abs().sum(dim=(-2, -1))
    return df / cfg.f_cmd_max


# ---------------------------------------------------------------------------
# Cfg carriers. R2 (mainline): observe-only. R1 (anneal twin): full servo, authority
# annealed 1→0 by ServoAuthorityCurriculum — terminal config ≡ R2's.
# ---------------------------------------------------------------------------
_UNIFIED_CFG = _replace(_DEFAULT_HYBRID_CFG, servo_observe_only=True)  # servo_enabled stays True
_UNIFIED_ANNEAL_CFG = _DEFAULT_HYBRID_CFG  # full servo; authority scaled by env._wbt_servo_alpha
_UNIFIED_HAND_CFG = _UNIFIED_CFG.hand_force_config()


def _unified_reward(cfg: HybridContactConfig):
    """Fork the hybrid reward: drop the preintegrator term (nothing to free-ride on in
    observe-only mode; in the anneal twin the direct term measures the same quantity),
    add the four unified terms. force_track_pos_eq2 (weight 3.0, servo-inclusive
    ref+Δx target) and the whole position family are inherited unchanged."""
    return _replace(
        g1_29dof_wbt_2pt_hybrid_reward,
        terms={
            **{k: v for k, v in g1_29dof_wbt_2pt_hybrid_reward.terms.items() if k != "force_error_preintegrator"},
            "force_error_direct": RewardTermCfg(
                func="wbt_training.config_values.hybrid_unified:force_error_direct", weight=2.0, params={"cfg": cfg}
            ),
            "contact_init_bonus": RewardTermCfg(
                func="wbt_training.config_values.hybrid_unified:contact_init_bonus", weight=0.3, params={"cfg": cfg}
            ),
            "approach_guidance": RewardTermCfg(
                func="wbt_training.config_values.hybrid_unified:approach_guidance",
                weight=0.2,
                params={"cfg": cfg, "d0": 0.05},
            ),
            "force_rate_penalty": RewardTermCfg(
                func="wbt_training.config_values.hybrid_unified:force_rate_penalty", weight=-0.05, params={"cfg": cfg}
            ),
        },
    )


g1_29dof_wbt_unified_reward = _unified_reward(_UNIFIED_CFG)
g1_29dof_wbt_unified_anneal_reward = _unified_reward(_UNIFIED_ANNEAL_CFG)


# ---------------------------------------------------------------------------
# 185D actor obs = the 183D 3pt-hybrid-global-vel layout + contact_phase (2D)
# inserted after force_meas (slots ≥07 shift +1). Every reused term is the SAME
# ObsTermCfg object. Zero-padded sorted keys obs00..obs16 (sorted-concat contract).
# 185D × history 4 = 740D. Critic = hybrid critic VERBATIM (privileged terms incl.
# the hard contact_gate stay critic-side; the soft contact_phase is the actor's).
# ---------------------------------------------------------------------------
_B = _actor_obs_3pt_hybrid_global_vel_terms
_actor_obs_unified_terms = {
    "obs00_command_lower_body": _B["obs00_command_lower_body"],
    "obs01_vr_3point_pos": _B["obs01_vr_3point_pos"],
    "obs02_vr_3point_orn_6d": _B["obs02_vr_3point_orn_6d"],
    "obs03_motion_ref_pos_b": _B["obs03_motion_ref_pos_b"],
    "obs04_motion_ref_vel_b": _B["obs04_motion_ref_vel_b"],
    "obs05_force_cmd_local": _B["obs05_force_cmd_local"],
    "obs06_force_meas_local": _B["obs06_force_meas_local"],
    "obs07_contact_phase": ObsTermCfg(func="wbt_training.config_values.hybrid:contact_phase", scale=1.0, noise=0.0),
    "obs08_integrator_state": _B["obs07_integrator_state"],
    "obs09_x_tgt_shift": _B["obs08_x_tgt_shift"],
    "obs10_motion_ref_ori_b": _B["obs09_motion_ref_ori_b"],
    "obs11_projected_gravity": _B["obs10_projected_gravity"],
    "obs12_base_ang_vel": _B["obs11_base_ang_vel"],
    "obs13_base_lin_vel": _B["obs12_base_lin_vel"],
    "obs14_dof_pos": _B["obs13_dof_pos"],
    "obs15_dof_vel": _B["obs14_dof_vel"],
    "obs16_actions": _B["obs15_actions"],
}  # 24+9+18+3+3+6+6+2+6+6+6+3+3+3+29+29+29 = 185D actor (740D w/ history 4)

g1_29dof_wbt_observation_3pt_unified_hybrid_global_vel = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_unified_terms),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=10,
            terms=g1_29dof_wbt_observation_2pt_hybrid.groups["critic_obs"].terms,
        ),
    }
)


# ---------------------------------------------------------------------------
# 170D LOCAL actor obs = the 168D 3pt-hybrid layout + contact_phase (2D) inserted
# after force_meas (slots ≥05 shift +1) — for `exp:g1-wbt-3pt-hybrid-unified[-anneal]`,
# the pelvis-local twins of the global-vel unified presets (same R2/R1 recipe, but
# NO root pos/vel trio and quaternion vr_3point_orn — root-invariant like the base
# 3pt-hybrid). Every reused term is the SAME ObsTermCfg object (incl. the obs12
# JointVelDropout wrapper, default OFF). Zero-padded sorted keys obs00..obs13.
# 170D × history 4 = 680D. Critic = hybrid critic VERBATIM.
# ---------------------------------------------------------------------------
_L = _actor_obs_3pt_hybrid_terms

# TCP-UNIFIED HAND POINT (user 2026-08-03): the position-tracking point for the two
# hands moves from the 0.18 m VR-tracker lever onto the 0.242 m fingertip TCP — the
# SAME point the force stack (spring wall / F^ext / F^meas / Jᵀ servo / Eq.2 FK)
# already acts on, so position and force are supervised at one place instead of two
# anchors 6.2 cm apart. Head point unchanged. Offsets: robot_config.VR_3POINT_OFFSETS_TCP.
#
# The term is RENAMED `vr_3point_pos_tcp` (not a re-parameterised `vr_3point_pos`)
# so a checkpoint's own config records which geometry it trained on — the pre-
# 2026-08-03 unified runs (iss0qs71 / m3s0pw9i) carry `vr_3point_pos` and will not be
# silently deployed against the new FK. Width is UNCHANGED (9D), so the 170D actor obs
# and every downstream dim assertion stay exactly as they were.
#
# Scope: ONLY these two local unified presets. The base g1_wbt_3pt_hybrid, the
# global-vel unified twins, and every other 3pt/2pt preset keep `vr_3point_pos`
# (0.18) byte-identically.
_VR_3POINT_POS_TCP_TERM = ObsTermCfg(
    func="wbt_training.config_values.observation:vr_3point_local_target_tcp", scale=1.0, noise=0.0
)

_actor_obs_local_unified_terms = {
    "obs00_command_lower_body": _L["obs00_command_lower_body"],
    "obs01_vr_3point_pos_tcp": _VR_3POINT_POS_TCP_TERM,
    "obs02_vr_3point_orn": _L["obs02_vr_3point_orn"],
    "obs03_force_cmd_local": _L["obs03_force_cmd_local"],
    "obs04_force_meas_local": _L["obs04_force_meas_local"],
    # SAME ObsTermCfg object as the 185D layout's obs07_contact_phase (review 2026-07-24:
    # a value-equal duplicate could silently diverge if the term ever gains params).
    "obs05_contact_phase": _actor_obs_unified_terms["obs07_contact_phase"],
    "obs06_integrator_state": _L["obs05_integrator_state"],
    "obs07_x_tgt_shift": _L["obs06_x_tgt_shift"],
    "obs08_motion_ref_ori_b": _L["obs07_motion_ref_ori_b"],
    "obs09_projected_gravity": _L["obs08_projected_gravity"],
    "obs10_base_ang_vel": _L["obs09_base_ang_vel"],
    "obs11_dof_pos": _L["obs10_dof_pos"],
    "obs12_dof_vel": _L["obs11_dof_vel"],
    "obs13_actions": _L["obs12_actions"],
}  # 24+9+12+6+6+2+6+6+6+3+3+29+29+29 = 170D actor (680D w/ history 4)

g1_29dof_wbt_observation_3pt_unified_hybrid = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_local_unified_terms
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=10,
            terms=g1_29dof_wbt_observation_2pt_hybrid.groups["critic_obs"].terms,
        ),
    }
)


# ---------------------------------------------------------------------------
# R1 scaffold-retirement curriculum.
# ---------------------------------------------------------------------------
class ServoAuthorityCurriculum(CurriculumTermBase):
    """R1 scaffold-retirement curriculum: anneal the servo's ACTUATION authority
    env._wbt_servo_alpha 1→0, performance-gated (JSRL/RAJS style — held until the
    force-alpha ramp completes, then drop a step only
    while the contact force-tracking error EMA is below threshold). Monotonic; only
    the injected Δq_arm is scaled (hybrid.py _servo_step reads the buffer) — the
    integrator, I/Δx obs, and the Eq.2 reward target keep FULL authority so the
    reference the policy chases never changes. Terminal α=0 ≡ the observe-only (R2)
    configuration actuation-wise."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.hc_cfg: HybridContactConfig = cfg.params["hybrid_config"]
        self.err_threshold_n: float = float(cfg.params.get("force_err_threshold_n", 2.0))
        self.alpha_step: float = float(cfg.params.get("alpha_step", 0.02))
        self.ema_beta: float = float(cfg.params.get("ema_beta", 0.01))
        # Spec §5 R1: "mean‖F_cmd+F_meas‖ < threshold 持续 N iters 才降档" — the EMA must
        # stay below threshold for `hold_resets` CONSECUTIVE reset evaluations before each
        # −alpha_step drop, and the counter restarts after every drop (or violation). At
        # ~10 env resets per training step (8192 envs), an ungated drop-per-reset would
        # anneal 1→0 in seconds once the servo tracks well — no time for the policy to
        # take over at each level (2026-07-24 review). 200 holds/drop × 50 drops spreads
        # the retirement over ~10k below-threshold resets.
        self.hold_resets: int = int(cfg.params.get("hold_resets", 200))
        self._err_ema: float | None = None
        self._below_count: int = 0

    def setup(self) -> None:
        # getattr+isinstance (not hasattr) so MagicMock test doubles don't shadow the
        # allocation — same pattern as ForceAlphaCurriculum.setup.
        existing = getattr(self.env, "_wbt_servo_alpha", None)
        if isinstance(existing, torch.Tensor):
            existing.fill_(1.0)
        else:
            self.env._wbt_servo_alpha = torch.tensor(1.0, device=self.env.device, dtype=torch.float)

    def reset(self, env_ids) -> None:
        env = self.env
        # Force-alpha readiness gate (2026-07-24 review F1): while the force curriculum
        # is still ramping (α_g < 1), F_cmd is scaled down and ‖F_cmd+F_ext‖ is small
        # because the FORCES are small — not because tracking is good. An absolute 2 N
        # threshold would open prematurely and permanently burn the monotonic α. Hold
        # all annealing until the force task is at full strength. (getattr+isinstance:
        # absent/mock buffer → no gate, unit tests unaffected.)
        fa = getattr(env, "_wbt_force_alpha", None)
        if isinstance(fa, torch.Tensor) and float(fa) < 0.999:
            return
        gate = env._wbt_contact_phase  # (n,h) soft, deployable
        # Sample only COMMAND-CARRYING contacts: with F_cmd≈0 (force-alpha ramp start,
        # or incidental brushes) ‖F_cmd+F_ext‖ = ‖F_ext‖ is not a tracking error, and in
        # the hysteresis decay band it can sit below the threshold — which would erode
        # the monotonic α before the force task even begins (2026-07-24 review).
        cmd_on = env._wbt_force_cmd_w.norm(dim=-1) > 1e-3  # (n,h)
        mask = (gate > 0.5) & cmd_on
        if not bool(mask.any()):
            return  # no commanded in-contact sample this reset → hold α
        e = (env._wbt_force_cmd_w + env._wbt_force_ext_w[..., :3]).norm(dim=-1)  # (n,h) N clean
        sample = float(e[mask].mean())
        self._err_ema = (
            sample if self._err_ema is None else (1 - self.ema_beta) * self._err_ema + self.ema_beta * sample
        )
        if self._err_ema < self.err_threshold_n:
            self._below_count += 1
            if self._below_count >= self.hold_resets:
                new_alpha = max(0.0, float(env._wbt_servo_alpha) - self.alpha_step)
                env._wbt_servo_alpha.fill_(new_alpha)
                self._below_count = 0  # re-earn the hold at the new authority level
        else:
            self._below_count = 0
        log_dict = getattr(env, "log_dict", None)
        if isinstance(log_dict, dict):
            log_dict["servo_alpha"] = env._wbt_servo_alpha.detach().clone()

    def step(self) -> None:
        return


# ---------------------------------------------------------------------------
# Command: motion_command + injector carrying the unified cfg (R2 observe-only /
# R1 full-servo). Identical wiring to hybrid.py's g1_29dof_wbt_hybrid_command.
# ---------------------------------------------------------------------------
def _unified_command(cfg: HybridContactConfig):
    term = CommandTermCfg(func="wbt_training.config_values.hybrid:HybridContactInjector", params={"hybrid_config": cfg})
    return _replace(
        _core_command,
        setup_terms={"motion_command": _core_command.setup_terms["motion_command"], "hybrid_contact_injector": term},
        reset_terms={**_core_command.reset_terms, "hybrid_contact_injector": term},
    )


g1_29dof_wbt_unified_command = _unified_command(_UNIFIED_CFG)
g1_29dof_wbt_unified_anneal_command = _unified_command(_UNIFIED_ANNEAL_CFG)

# ---------------------------------------------------------------------------
# Curriculum: force-alpha ramp ENABLED (R2's "position-first-for-free": α_g starts
# at 0 ≡ pure free-space tracking, ramps on episode-length health — the literature-
# backed alternative to a hard two-phase resume). R1 adds the servo-authority anneal.
# ---------------------------------------------------------------------------
_UNIFIED_HAND_CFG_RAMP = _replace(_UNIFIED_HAND_CFG, force_alpha_init=0.0, alpha_step=0.01)
_force_alpha_ramp_term = CurriculumTermCfg(
    func="wbt_training.config_values.force_tracking:ForceAlphaCurriculum",
    params={"hand_force_config": _UNIFIED_HAND_CFG_RAMP},
)
_servo_authority_term = CurriculumTermCfg(
    func="wbt_training.config_values.hybrid_unified:ServoAuthorityCurriculum",
    params={
        "hybrid_config": _UNIFIED_ANNEAL_CFG,
        "force_err_threshold_n": 2.0,
        "alpha_step": 0.02,
        "ema_beta": 0.01,
        "hold_resets": 200,
    },
)
g1_29dof_wbt_unified_curriculum = _replace(
    g1_29dof_wbt_hybrid_curriculum,
    setup_terms={**g1_29dof_wbt_hybrid_curriculum.setup_terms, "force_alpha": _force_alpha_ramp_term},
    reset_terms={**g1_29dof_wbt_hybrid_curriculum.reset_terms, "force_alpha": _force_alpha_ramp_term},
    step_terms={**g1_29dof_wbt_hybrid_curriculum.step_terms, "force_alpha": _force_alpha_ramp_term},
)
# NOTE: CurriculumManager instantiates ONE object per class-based term name and calls
# setup/reset/step on ALL class instances regardless of which term dict they appear in
# (dict membership only routes STATELESS functions) — so listing servo_authority in
# setup+reset (not step) is documentation of intent, not routing; its step() is a no-op.
g1_29dof_wbt_unified_anneal_curriculum = _replace(
    g1_29dof_wbt_unified_curriculum,
    setup_terms={**g1_29dof_wbt_unified_curriculum.setup_terms, "servo_authority": _servo_authority_term},
    reset_terms={**g1_29dof_wbt_unified_curriculum.reset_terms, "servo_authority": _servo_authority_term},
)


# ---------------------------------------------------------------------------
# Termination: core + force_runaway estop with the matching cfg carrier.
# ---------------------------------------------------------------------------
def _unified_termination(cfg: HybridContactConfig):
    return _replace(
        _core_termination,
        terms={
            **_core_termination.terms,
            "force_runaway": TerminationTermCfg(
                func="wbt_training.config_values.hybrid:force_runaway", params={"cfg": cfg}
            ),
        },
    )


g1_29dof_wbt_unified_termination = _unified_termination(_UNIFIED_CFG)
g1_29dof_wbt_unified_anneal_termination = _unified_termination(_UNIFIED_ANNEAL_CFG)

__all__ = [
    "force_error_direct",
    "contact_init_bonus",
    "approach_guidance",
    "force_rate_penalty",
    "ServoAuthorityCurriculum",
    "g1_29dof_wbt_unified_reward",
    "g1_29dof_wbt_unified_anneal_reward",
    "g1_29dof_wbt_observation_3pt_unified_hybrid_global_vel",
    "g1_29dof_wbt_observation_3pt_unified_hybrid",
    "g1_29dof_wbt_unified_command",
    "g1_29dof_wbt_unified_anneal_command",
    "g1_29dof_wbt_unified_curriculum",
    "g1_29dof_wbt_unified_anneal_curriculum",
    "g1_29dof_wbt_unified_termination",
    "g1_29dof_wbt_unified_anneal_termination",
]
