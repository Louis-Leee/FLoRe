"""MaskedMimic + hybrid position-force controller (`exp:g1-maskmimic-hybrid`).

Ports the `exp:g1-wbt-2pt-hybrid` HYBRID controller (causal K_env contact spring +
simulated wrist F/T sensor with F_meas in the actor + Jᵀ contact-gated force servo +
force-runaway estop) onto the `exp:g1-wbt-maskedmimic` recipe (multi-motion loader,
4 conditioning modes {2pt, 3pt, 5pt, 29dof-dense}, ABSOLUTE-world-frame tracking,
low-LR adaptive-KL PPO). Design: docs/superpowers/plans/2026-07-15-g1-maskmimic-hybrid.md.

WHAT IS REUSED VERBATIM (the whole hybrid physics stack is frame-agnostic):
  - `HybridContactInjector` (hybrid.py): sole owner of the `_wbt_force_*` buffers, the
    causal K_env contact spring, the simulated F/T sensor, and (servo_enabled) the Jᵀ
    servo that injects Δq into `joint_control._actions_after_delay`. It reads ONLY
    world-frame body FK / root_quat / force buffers — never a motion-ref frame — so it
    transplants onto the maskmimic `WholeBodyTrackingManager` env with zero change.
  - Obs readers `force_meas_local`, `contact_gate`, `integrator_state`, `x_tgt_shift`,
    `force_ext_local`, `k_env_privileged`, `penetration_privileged` (hybrid.py) — all
    rotate world→root (pelvis), same convention as `force_cmd_local`.
  - `force_error_reward_preintegrator`, `force_runaway` (hybrid.py) — frame-free
    (read only force buffers), reused verbatim.

WHAT IS NEW HERE:
  1. GLOBAL-frame servo Eq.2 rewards (`force_track_pos_eq2_hybrid_global` /
     `force_track_eq2_servo_global`): byte-identical to hybrid.py's relative twins
     with `_force_ref_pos_w` → `_force_ref_pos_w_global` (maskmimic tracks bodies in the
     absolute world frame, so the shifted target must anchor on `mc.body_pos_w`). They
     slice the 6-D `_wbt_force_ext_w[..., :3]` — reusing force_tracking's own
     `_force_eq2_target_w_global` would broadcast-FAIL, since it adds the unsliced 6-D
     ext buffer to the 3-D cmd buffer.
  2. ACTOR joint-velocity dropout (`_MaskedMimicHybridObservationProvider` +
     `JointVelDropoutConfig`): drops/zeroes the ACTOR's proprioceptive `dof_vel`
     channel (inside `noisy_reduced_coords_obs`), leaving the CRITIC's clean `dof_vel`
     term untouched. Motivation: a corrupted real Bota reading (wrist-acceleration
     transient) destabilises a velocity-reliant policy; the user wants quiet quasi-static
     motion, so the actor can lean off its own `dof_vel` and eventually zero it at deploy.

NON-INVASION: this module only READS hybrid.py / force_tracking.py / maskedmimic_*.py.
It never mutates their module-level objects (`_reroute_mm_terms` returns a fresh dict;
the servo-off config is a `replace()` on a frozen dataclass). `exp:g1-wbt-2pt-hybrid`,
`exp:g1-wbt-maskedmimic`, `exp:g1-wbt-maskedmimic-force`, and the mjswan viz are untouched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import replace as _replace
from typing import Any

import torch

from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.curriculum import CurriculumTermCfg
from holosoma.config_types.observation import ObsGroupCfg, ObservationManagerCfg, ObsTermCfg
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_types.termination import TerminationTermCfg

from wbt_training.config_values.command import g1_29dof_wbt_maskedmimic_command
from wbt_training.config_values.force_tracking import (
    HandForceConfig,
    _force_cur_pos_w,
    _force_ref_pos_w_global,
    g1_29dof_wbt_force_curriculum,
)
from wbt_training.config_values.maskedmimic_commands import MaskedMimicMultiMotionCommand
from wbt_training.config_values.hybrid import (
    HybridContactConfig,
    HybridContactInjector,  # noqa: F401  (referenced by dotted-path in CommandTermCfg)
)
from wbt_training.config_values.maskedmimic_force import MASKEDMIMIC_FORCE_HOVER_MODES, MASKEDMIMIC_FORCE_OBS_CONFIG
from wbt_training.config_values.maskedmimic_observations import (
    _MaskedMimicObservationProvider,
    _uniform_noise_like,
    g1_29dof_wbt_maskedmimic_reward,
    g1_29dof_wbt_maskedmimic_termination,
    g1_29dof_wbt_observation_maskedmimic,
    MaskedMimicObservationField,
)

# ===========================================================================
# 1. GLOBAL-frame servo Eq.2 rewards (the only new physics-adjacent code).
#
# hybrid.py's force_track_pos_eq2_hybrid / force_track_eq2_servo anchor the shifted
# target on the *relative* motion ref (_force_ref_pos_w, body_pos_relative_w).
# MaskedMimic tracks bodies in the ABSOLUTE world frame, so these twins anchor on
# _force_ref_pos_w_global (== mc.body_pos_w, incl. env_origins — the same field
# maskmimic's motion_global_body_position_error_exp reads). Everything else is
# byte-identical to the relative variants: slice the 6-D causal F_ext to [:3]
# (moment dormant, D8), same kernel exp(−err_l1·2/σ), same F=0⇒pure-tracking invariant.
# ===========================================================================


def force_track_pos_eq2_hybrid_global(env: Any, *, cfg: "HandForceConfig | None" = None) -> torch.Tensor:
    """GLOBAL-frame Eq.2 tracking reward (Layer A / servo-OFF path). Anchors the
    (F_cmd+F_ext)/K-shifted target on the ABSOLUTE global motion ref. Slices the 6-D
    causal F_ext buffer to its force channels [:3] (moment dormant, D8). PURE READER.

    Identical to hybrid.force_track_pos_eq2_hybrid except _force_ref_pos_w →
    _force_ref_pos_w_global — so it matches maskmimic's own global body reward frame."""
    cfg = cfg or HandForceConfig()
    ref_w = _force_ref_pos_w_global(env, cfg)
    f_total_w = env._wbt_force_cmd_w + env._wbt_force_ext_w[..., :3]  # 3D force only (6-D ext sliced)
    target_w = ref_w + f_total_w / cfg.stiffness_K
    cur_w = _force_cur_pos_w(env, cfg)
    err_l1 = (cur_w - target_w).abs().sum(dim=(-2, -1))
    return torch.exp(-err_l1 * 2.0 / cfg.sigma)


def force_track_eq2_servo_global(env: Any, *, cfg: "HandForceConfig | None" = None) -> torch.Tensor:
    """Servo-inclusive GLOBAL-frame Eq.2 tracking reward (Layer B). When the servo is ON,
    the task-space setpoint is shifted by the servo's Δx (which folds in the contact-gated
    integral), so the target is ref_global + Δx (both world-frame → frame-consistent). When
    OFF, delegates to the Layer-A force_track_pos_eq2_hybrid_global. Branch keyed off
    env._wbt_hybrid_cfg.servo_enabled. PURE READER.

    GLOBAL-frame analogue of hybrid.force_track_eq2_servo (only the ref anchor differs)."""
    cfg = cfg or HandForceConfig()
    hc = getattr(env, "_wbt_hybrid_cfg", None)
    if hc is None or not hc.servo_enabled:
        return force_track_pos_eq2_hybrid_global(env, cfg=cfg)
    ref_w = _force_ref_pos_w_global(env, cfg)
    target_w = ref_w + env._wbt_delta_x  # servo-inclusive setpoint shift (m, world), already xyz
    cur_w = _force_cur_pos_w(env, cfg)
    err_l1 = (cur_w - target_w).abs().sum(dim=(-2, -1))
    return torch.exp(-err_l1 * 2.0 / cfg.sigma)


# ===========================================================================
# 2. ACTOR joint-velocity dropout on the maskmimic proprioception.
#
# In the maskmimic obs the ACTOR's dof_vel lives INSIDE obs00_noisy_reduced_coords_obs
# (built by the provider's _reduced_coords_obs(apply_noise=True)), while the CRITIC's
# dof_vel is a separate clean dotted-path term (obs09_joint_vel). So a dropout hook in the
# provider's apply_noise=True branch drops the actor's dof_vel ONLY — the critic stays
# privileged-clean (textbook asymmetric actor-critic; the policy gradient stays unbiased
# because the extra state the critic sees is action-independent).
# ===========================================================================


@dataclass(frozen=True)
class JointVelDropoutConfig:
    """ACTOR joint-velocity dropout / zeroing knobs.

    enabled       : master switch.
    dropout_prob  : fraction of envs (per-episode) or steps (per-step) whose ACTOR dof_vel
                    is zeroed. 0.0 ⇒ never drop; 1.0 ⇒ always drop (== eval_zero for training).
    per_episode   : True (default) ⇒ a (num_envs,) mask sampled at reset, held for the whole
                    episode (so a fraction of envs run ENTIRE episodes with dof_vel≡0, making
                    "zero dof_vel at inference" in-distribution). False ⇒ i.i.d. per-step
                    Bernoulli (VERIFIED not frozen by the get_field per-step cache — _build_outputs
                    re-runs once per env-step on the cache miss).
    noise_scale   : extra uniform noise added to the SURVIVING (non-dropped) actor dof_vel,
                    on top of the base 0.5 uniform noise (additive-on-additive, intended).
    eval_zero     : force ALL actor dof_vel to 0 regardless of the mask (inference/eval parity).
    """

    enabled: bool = True
    dropout_prob: float = 0.5
    per_episode: bool = True
    noise_scale: float = 0.0
    eval_zero: bool = False


# Module-level default, overridable from the launcher via env vars (like HOVER_MASK_MODES).
#   WBT_JV_DROPOUT_ENABLED = 0|1
#   WBT_JV_DROPOUT_PROB    = float in [0,1]
#   WBT_JV_DROPOUT_MODE    = per_episode | per_step
#   WBT_JV_DROPOUT_NOISE   = float >= 0
#   WBT_JV_EVAL_ZERO       = 0|1   (inference/eval: zero ALL actor dof_vel)
def _jv_dropout_config_from_env() -> JointVelDropoutConfig:
    """Build the JointVelDropoutConfig from env vars, falling back to the dataclass defaults.

    Read ONCE at import time (module-level), matching how the maskmimic launcher passes
    HOVER_MASK_MODES. Robust to unset / malformed values (fall back to the default field)."""
    d = JointVelDropoutConfig()

    def _bool(name: str, default: bool) -> bool:
        v = os.environ.get(name)
        if v is None or v == "":
            return default
        return v.strip().lower() in ("1", "true", "yes", "on")

    def _float(name: str, default: float) -> float:
        v = os.environ.get(name)
        if v is None or v == "":
            return default
        try:
            return float(v)
        except ValueError:
            return default

    mode = os.environ.get("WBT_JV_DROPOUT_MODE", "").strip().lower()
    per_episode = d.per_episode if mode not in ("per_episode", "per_step") else (mode == "per_episode")
    return JointVelDropoutConfig(
        enabled=_bool("WBT_JV_DROPOUT_ENABLED", d.enabled),
        dropout_prob=_float("WBT_JV_DROPOUT_PROB", d.dropout_prob),
        per_episode=per_episode,
        noise_scale=_float("WBT_JV_DROPOUT_NOISE", d.noise_scale),
        eval_zero=_bool("WBT_JV_EVAL_ZERO", d.eval_zero),
    )


_DEFAULT_JV_DROPOUT_CFG = _jv_dropout_config_from_env()


class _MaskedMimicHybridObservationProvider(_MaskedMimicObservationProvider):
    """MaskedMimic obs provider with (a) the restricted 4-mode hover set and (b) ACTOR
    joint-velocity dropout. Only `_reduced_coords_obs` (actor branch) and `reset` differ
    from the base; everything else (target poses, masking, history) is inherited verbatim."""

    def __init__(self, env, jv_cfg: JointVelDropoutConfig = _DEFAULT_JV_DROPOUT_CFG, **kwargs) -> None:
        # Restrict the hover-mask mode set to the 4 trained modes (reuse the maskmimic-force
        # obs config so the keypoint/joint regex patterns are byte-identical to the base preset).
        super().__init__(env, obs_cfg=MASKEDMIMIC_FORCE_OBS_CONFIG, **kwargs)
        self.jv_cfg = jv_cfg
        # (num_envs,) bool: True ⇒ this env's ACTOR dof_vel is zeroed this episode.
        # Lazily allocated by SAMPLING (not zero-fill) so episode-0 envs get a real draw even if
        # the field's reset() never fires before the first __call__ (reviewer fix 2026-07-15).
        self._jv_dropout_mask: torch.Tensor | None = None

    def _ensure_jv_mask(self) -> torch.Tensor:
        if self._jv_dropout_mask is None:
            if self.jv_cfg.per_episode:
                self._jv_dropout_mask = torch.rand(self.num_envs, device=self.device) < self.jv_cfg.dropout_prob
            else:
                self._jv_dropout_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._jv_dropout_mask

    def _apply_jv_dropout(self, joint_vel_rel: torch.Tensor) -> torch.Tensor:
        """Drop / zero / noise the ACTOR proprioceptive joint velocity. joint_vel_rel: (n, n_dof).
        Called ONLY from the actor (apply_noise=True) branch of _reduced_coords_obs."""
        cfg = self.jv_cfg
        if not cfg.enabled:
            return joint_vel_rel
        if cfg.eval_zero:
            return torch.zeros_like(joint_vel_rel)
        if cfg.per_episode:
            mask = self._ensure_jv_mask()  # (n,) bool
        else:
            mask = torch.rand(self.num_envs, device=self.device) < cfg.dropout_prob  # i.i.d. per step
        out = joint_vel_rel * (~mask).unsqueeze(-1).to(joint_vel_rel.dtype)  # zero the dropped envs
        if cfg.noise_scale > 0.0:  # extra noise on the SURVIVING envs only
            out = out + _uniform_noise_like(torch.zeros_like(out), cfg.noise_scale) * (~mask).unsqueeze(-1).to(
                out.dtype
            )
        return out

    def _reduced_coords_obs(self, *, apply_noise: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Verbatim copy of the base method with ONE insertion: in the actor (apply_noise=True)
        branch, apply the joint-velocity dropout right after the base joint_vel noise. The critic
        (apply_noise=False) path is byte-identical to the base → critic dof_vel is never dropped.

        Kept as a full copy (not a super() call + patch) because the base returns a concatenated
        tensor with joint_vel already baked in; a test asserts the non-dof_vel components still
        match the base output so this copy cannot silently drift."""
        joint_pos_rel = self._joint_pos_rel()
        joint_vel_rel = self._joint_vel_rel()
        projected_gravity = self._projected_gravity_obs()
        base_ang_vel = self._base_ang_vel_obs()

        if apply_noise:
            joint_pos_rel = _uniform_noise_like(joint_pos_rel, 0.01)
            joint_vel_rel = _uniform_noise_like(joint_vel_rel, 0.5)
            joint_vel_rel = self._apply_jv_dropout(joint_vel_rel)  # <-- ACTOR-only dropout (NEW)
            projected_gravity = _uniform_noise_like(projected_gravity, 0.01)
            base_ang_vel = _uniform_noise_like(base_ang_vel, 0.2)

        base_lin_vel = None
        if self.obs_cfg.include_base_lin_vel:
            base_lin_vel = self._base_lin_vel_obs()
            if apply_noise:
                base_lin_vel = _uniform_noise_like(base_lin_vel, 0.1)

        if apply_noise and self._s2r_enabled:
            projected_gravity, base_lin_vel, base_ang_vel = self._apply_fk_reduced_noise(
                projected_gravity, base_lin_vel, base_ang_vel
            )

        components = [joint_pos_rel, joint_vel_rel, projected_gravity]
        if base_lin_vel is not None:
            components.append(base_lin_vel)
        components.append(base_ang_vel)

        reduced = torch.cat(components, dim=-1)
        return reduced, projected_gravity

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # Base reset expands env_ids=None → arange; mirror that here BEFORE resampling the mask so
        # a full-env reset (env_ids=None) resamples all masks rather than index-erroring.
        if env_ids is None:
            resolved = torch.arange(self.num_envs, device=self.device)
        else:
            resolved = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super().reset(env_ids)  # base handles its own None→arange + numel==0 short-circuit
        if resolved.numel() == 0:
            return
        if self.jv_cfg.per_episode:
            mask = self._ensure_jv_mask()
            m = resolved.numel()
            mask[resolved] = torch.rand(m, device=self.device) < self.jv_cfg.dropout_prob


def _get_hybrid_provider(env: Any) -> _MaskedMimicHybridObservationProvider:
    provider = getattr(env, "_wbt_maskedmimic_hybrid_obs_provider", None)
    if provider is None:
        provider = _MaskedMimicHybridObservationProvider(env)
        setattr(env, "_wbt_maskedmimic_hybrid_obs_provider", provider)
    return provider


class MaskedMimicHybridObservationField(MaskedMimicObservationField):
    """MaskedMimic obs term whose provider is the hybrid provider (restricted hover modes +
    ACTOR joint-vel dropout). Identical to MaskedMimicObservationField except it resolves the
    hybrid-specific provider under a distinct env attr (never collides with the base / force
    providers), exactly like maskedmimic_force.MaskedMimicForceObservationField."""

    def _provider(self) -> _MaskedMimicObservationProvider:
        if self.provider is None:
            self.provider = _get_hybrid_provider(self.env)
        return self.provider


_MM_HYBRID_FIELD = "wbt_training.config_values.maskedmimic_hybrid:MaskedMimicHybridObservationField"


def _reroute_mm_terms(terms: dict[str, ObsTermCfg]) -> dict[str, ObsTermCfg]:
    """Return a COPY with every MaskedMimicObservationField term re-pointed at the hybrid
    provider field func; all other terms preserved byte-for-byte. Never mutates the input dict
    (non-invasion: the base maskmimic term dicts must stay untouched)."""
    out: dict[str, ObsTermCfg] = {}
    for name, term in terms.items():
        if term.func.endswith(":MaskedMimicObservationField"):
            out[name] = _replace(term, func=_MM_HYBRID_FIELD)
        else:
            out[name] = term
    return out


# ===========================================================================
# 2b. Multi-motion command that re-seeds the HYBRID causal state at mid-episode
# motion-boundary teleports.
#
# THE BUG THIS FIXES (found in code review 2026-07-15): MaskedMimicMultiMotionCommand.step()
# teleports envs whose clip has ended to a NEW reference pose MID-episode (maskedmimic_commands.py
# :212-236) — NOT a full env reset, so the CommandManager never dispatches HybridContactInjector.
# reset() for them. The base step() calls restart_force_pulse_for_envs(), which re-seeds only the
# SCRIPTED F^cmd pulse buffers — it has no knowledge of the hybrid causal state. So after a
# teleport the TCP jumps to a new world position while the VIRTUAL CONTACT PLANE (_wbt_plane_*),
# K_env, sensor bias, F^cmd-along-normal direction, reachable cap, and the servo integrator/Δx
# all stay anchored at the PREVIOUS clip's reset — injecting a spurious contact force onto the
# freshly teleported pose + carrying stale integrator wind-up across the boundary. 2pt-hybrid never
# hit this because it forks the single-clip MotionCommand, which never resamples mid-episode.
#
# FIX: after the parent teleport (+ its refresh_sim_tensors(), so body_pos_w is valid), call the
# injector's reset() for exactly the teleported envs — which re-seeds the plane / K_env / bias /
# F^cmd direction+cap / integrator / Δx (and re-restarts the pulse, idempotently) at the NEW pose.
# The injector instance is reachable via command_manager.get_state (it is registered under the same
# name in setup+reset, so ONE instance owns the buffers — manager.py:218-222). No shared-file edit:
# this is a NEW subclass wired ONLY into the maskmimic-hybrid preset.
# ===========================================================================


class MaskedMimicHybridMultiMotionCommand(MaskedMimicMultiMotionCommand):
    """MaskedMimic multi-motion command that also re-seeds the hybrid injector's causal-contact +
    servo state for envs teleported at a mid-episode motion boundary (see the section header)."""

    def step(self) -> None:
        # Recompute which envs are about to be teleported, using the SAME predicate as the parent
        # (time_steps incremented at the TOP of the parent step()). We recompute here rather than
        # capture the parent's local because step() does not expose ended_env_ids.
        per_motion_end = self.motion.motion_end_idx[self.motion_ids.to(self.motion.motion_end_idx.device)].to(
            self.device
        )
        # Parent increments self.time_steps += 1 first, then thresholds `>=`; mirror that by
        # thresholding the POST-increment value (time_steps + 1).
        ended_env_ids = torch.where((self.time_steps + 1) >= per_motion_end)[0]

        super().step()  # teleport + refresh_sim_tensors + restart_force_pulse_for_envs (base)

        if ended_env_ids.numel() > 0:
            injector = self._env.command_manager.get_state("hybrid_contact_injector")
            if injector is not None:
                # Re-seed the hybrid causal state (plane/K_env/bias/F^cmd-dir/reachable/integrator/Δx)
                # for the teleported envs at their NEW pose. body_pos_w is valid: the parent already
                # called refresh_sim_tensors() before restart_force_pulse_for_envs.
                injector.reset(ended_env_ids)


# ===========================================================================
# 3. Config objects (default = SHIPPED servo-ON, per user 2026-07-15).
# ===========================================================================
# Reuse the SAME shipped hybrid config as exp:g1-wbt-2pt-hybrid (servo ON, all the
# de-risked contact/sensor/servo/estop knobs). It is frame-agnostic — nothing in it
# assumes the 2pt relative obs.
_DEFAULT_HYBRID_CFG = HybridContactConfig(servo_enabled=True)
_DEFAULT_HAND_FORCE_CFG = _DEFAULT_HYBRID_CFG.hand_force_config()

_HYBRID_CFG_SERVO_OFF = _replace(_DEFAULT_HYBRID_CFG, servo_enabled=False)
_HAND_FORCE_CFG_SERVO_OFF = _HYBRID_CFG_SERVO_OFF.hand_force_config()


# ---------------------------------------------------------------------------
# 3.1 Observation manager (fixed width, lexicographic-safe).
#
# Actor  = maskmimic actor terms (rerouted → hybrid provider) + obs90..93 force/servo.
# Critic = maskmimic critic terms (rerouted)                 + obs90..97 (+privileged).
# history_length=1 for BOTH (maskmimic convention, NOT the 2pt-hybrid 4/10).
#
# ⚠️ obs90.. prefixes sort STRICTLY AFTER every base maskmimic key (obs00..obs17) under the
# manager's lexicographic sort (manager.py:141) — so the maskmimic block stays a strict prefix
# of the hybrid layout. Never rename these below obs90. The four shared force/servo terms are
# present at FIXED width regardless of servo_enabled (I/Δx buffers allocated unconditionally in
# the injector; zeroed when servo off) → one network + checkpoint lineage for on/off + two-phase.
# ---------------------------------------------------------------------------
_BASE_ACTOR_TERMS = g1_29dof_wbt_observation_maskedmimic.groups["actor_obs"].terms
_BASE_CRITIC_TERMS = g1_29dof_wbt_observation_maskedmimic.groups["critic_obs"].terms

_FORCE_CMD_TERM = ObsTermCfg(func="wbt_training.config_values.force_tracking:force_cmd_local", scale=0.04, noise=0.0)
_FORCE_MEAS_TERM = ObsTermCfg(func="wbt_training.config_values.hybrid:force_meas_local", scale=0.04, noise=0.0)
_INTEG_TERM = ObsTermCfg(func="wbt_training.config_values.hybrid:integrator_state", scale=1.0, noise=0.0)
_XTGT_TERM = ObsTermCfg(func="wbt_training.config_values.hybrid:x_tgt_shift", scale=1.0, noise=0.0)
_CONTACT_GATE_TERM = ObsTermCfg(func="wbt_training.config_values.hybrid:contact_gate", scale=1.0, noise=0.0)
_FORCE_EXT_TERM = ObsTermCfg(func="wbt_training.config_values.hybrid:force_ext_local", scale=0.04, noise=0.0)
_K_ENV_TERM = ObsTermCfg(func="wbt_training.config_values.hybrid:k_env_privileged", scale=0.001, noise=0.0)
_PENETRATION_TERM = ObsTermCfg(func="wbt_training.config_values.hybrid:penetration_privileged", scale=10.0, noise=0.0)

# Actor: maskmimic terms (restricted modes + dropout) + F^cmd, F^meas, I, Δx (sensor in the loop).
_actor_obs_maskedmimic_hybrid_terms = {
    **_reroute_mm_terms(_BASE_ACTOR_TERMS),
    "obs90_force_cmd_local": _FORCE_CMD_TERM,
    "obs91_force_meas_local": _FORCE_MEAS_TERM,
    "obs92_integrator_state": _INTEG_TERM,
    "obs93_x_tgt_shift": _XTGT_TERM,
}

# Critic: maskmimic critic terms + the same four + privileged sim-only ground truth
# (contact_gate, clean F_ext, hidden K_env, penetration depth — never in the actor).
_critic_obs_maskedmimic_hybrid_terms = {
    **_reroute_mm_terms(_BASE_CRITIC_TERMS),
    "obs90_force_cmd_local": _FORCE_CMD_TERM,
    "obs91_force_meas_local": _FORCE_MEAS_TERM,
    "obs92_integrator_state": _INTEG_TERM,
    "obs93_x_tgt_shift": _XTGT_TERM,
    "obs94_contact_gate": _CONTACT_GATE_TERM,
    "obs95_force_ext_local": _FORCE_EXT_TERM,
    "obs96_k_env_privileged": _K_ENV_TERM,
    "obs97_penetration_privileged": _PENETRATION_TERM,
}

g1_29dof_wbt_observation_maskedmimic_hybrid = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=1, terms=_actor_obs_maskedmimic_hybrid_terms
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=_critic_obs_maskedmimic_hybrid_terms
        ),
    }
)


# ---------------------------------------------------------------------------
# 3.2 Reward: known-good maskmimic reward + the force-tracking swaps proven on
# maskmimic-force + the hybrid servo/preintegrator additions.
#
#   - swap motion_global_body_pos → motion_global_body_pos_no_wrist (Eq.2 owns the wrist)
#   - drop motion_global_wrist_pos (Eq.2 owns it); KEEP wrist ori/lin_vel/ang_vel (orthogonal)
#   - add force_track_pos_eq2 → force_track_eq2_servo_global, w=6.0 (matches maskmimic-force;
#     the servo already accounts for its effect by moving the target to ref+Δx, so weight
#     stays at the known-good value — NOT halved)
#   - add force_error_preintegrator (w=1.0, anti-free-ride; force-error, frame-free)
#   - add feet_slip (w=-2.0)
# EVERYTHING else in the maskmimic reward is inherited UNCHANGED (qbdjrrxt is known-good).
# ---------------------------------------------------------------------------
_MM_GLOBAL_BODY_POS = "motion_global_body_pos"
_MM_WRIST_POS = "motion_global_wrist_pos"


def _build_hybrid_reward(hand_force_cfg: HandForceConfig, hybrid_cfg: HybridContactConfig) -> RewardManagerCfg:
    """Build the maskmimic-hybrid reward. Factored so the servo-on and servo-off variants share
    identical KEYS + weights (only the cfg-carriers differ → checkpoint/critic lineage intact)."""
    return RewardManagerCfg(
        terms={
            # Inherit every maskmimic term EXCEPT the full-body global position term (swapped for
            # no-wrist below) and the dedicated wrist-position term (dropped — Eq.2 owns the wrist).
            **{
                k: v
                for k, v in g1_29dof_wbt_maskedmimic_reward.terms.items()
                if k not in (_MM_GLOBAL_BODY_POS, _MM_WRIST_POS)
            },
            # No-wrist GLOBAL body position reward (replaces motion_global_body_pos). ONE fixed key
            # across servo-on/off so the reward KEY set is identical (checkpoint lineage).
            "motion_global_body_pos_no_wrist": RewardTermCfg(
                func="wbt_training.config_values.force_tracking:motion_global_body_position_error_exp_no_wrist",
                params={"sigma": 0.3, "exclude_body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link")},
                weight=1.0,  # matches the motion_global_body_pos baseline weight
            ),
            # Servo-inclusive GLOBAL Eq.2 tracking (ref_global + Δx when servo on, else
            # ref_global + (F_cmd+F_ext[:3])/K). w=6.0 matches maskmimic-force (NOT halved).
            "force_track_pos_eq2": RewardTermCfg(
                func="wbt_training.config_values.maskedmimic_hybrid:force_track_eq2_servo_global",
                weight=6.0,
                params={"cfg": hand_force_cfg},
            ),
            # Pre-integrator DIRECT force reward (anti-free-ride §4.A): credits the policy's own
            # feedforward force, invariant to the servo's integral patch. Frame-free (force buffers).
            "force_error_preintegrator": RewardTermCfg(
                func="wbt_training.config_values.hybrid:force_error_reward_preintegrator",
                weight=1.0,
                params={"cfg": hybrid_cfg},
            ),
            # Feet-slip penalty: F at the hand pulls the CoM sideways; without this the policy
            # reward-hacks via foot slip (fine in sim, fails on bounded real friction).
            "feet_slip": RewardTermCfg(
                func="wbt_training.config_values.force_tracking:feet_slip",
                weight=-2.0,
                params={"body_names": ("left_ankle_roll_link", "right_ankle_roll_link")},
            ),
        }
    )


g1_29dof_wbt_maskedmimic_hybrid_reward = _build_hybrid_reward(_DEFAULT_HAND_FORCE_CFG, _DEFAULT_HYBRID_CFG)
g1_29dof_wbt_maskedmimic_hybrid_reward_servo_off = _build_hybrid_reward(
    _HAND_FORCE_CFG_SERVO_OFF, _HYBRID_CFG_SERVO_OFF
)


# ---------------------------------------------------------------------------
# 3.3 Command: maskmimic multi-motion command (teleport-aware hybrid subclass) +
# hybrid_contact_injector.
#
# Two changes vs the base maskmimic command:
#   (a) motion_command func is repointed to MaskedMimicHybridMultiMotionCommand (re-seeds the
#       hybrid causal state at mid-episode motion-boundary teleports — see §2b). Repointed in
#       ALL of setup/reset/step_terms so the SAME subclass instance owns every lifecycle hook.
#   (b) hybrid_contact_injector added to BOTH setup_terms (buffer alloc + substep monkey-patch)
#       AND reset_terms (per-episode pulse restart + K_env/plane/bias resample + force zeroing).
#       Registering only in setup_terms would leave reset() dead → the pulse + F^ext leak across
#       episodes (the maskmimic-force / force_tracking footgun). NOT in step_terms (the injector
#       advances on the substep-0 physics hook, not in step()).
# ---------------------------------------------------------------------------
_MM_HYBRID_CMD_FUNC = "wbt_training.config_values.maskedmimic_hybrid:MaskedMimicHybridMultiMotionCommand"
_MM_BASE_CMD_SUFFIX = ":MaskedMimicMultiMotionCommand"


def _repoint_motion_command(terms: dict) -> dict:
    """Return a COPY of a term dict with the motion_command func repointed from the base
    MaskedMimicMultiMotionCommand to the teleport-aware hybrid subclass. Params preserved."""
    out = dict(terms)
    mc = out.get("motion_command")
    if mc is not None and mc.func.endswith(_MM_BASE_CMD_SUFFIX):
        out["motion_command"] = _replace(mc, func=_MM_HYBRID_CMD_FUNC)
    return out


def _build_hybrid_command(hybrid_cfg: HybridContactConfig):
    injector = CommandTermCfg(
        func="wbt_training.config_values.hybrid:HybridContactInjector", params={"hybrid_config": hybrid_cfg}
    )
    base = g1_29dof_wbt_maskedmimic_command
    return _replace(
        base,
        setup_terms={**_repoint_motion_command(base.setup_terms), "hybrid_contact_injector": injector},
        reset_terms={**_repoint_motion_command(base.reset_terms), "hybrid_contact_injector": injector},
        step_terms=_repoint_motion_command(base.step_terms),
    )


g1_29dof_wbt_maskedmimic_hybrid_command = _build_hybrid_command(_DEFAULT_HYBRID_CFG)
g1_29dof_wbt_maskedmimic_hybrid_command_servo_off = _build_hybrid_command(_HYBRID_CFG_SERVO_OFF)


# ---------------------------------------------------------------------------
# 3.4 Curriculum: maskmimic-force curriculum (force_alpha term, DISABLED by default →
# full force from step 0). Identical to the maskmimic-force preset.
# ---------------------------------------------------------------------------
g1_29dof_wbt_maskedmimic_hybrid_curriculum = g1_29dof_wbt_force_curriculum

# Re-export a curriculum-term cfg for symmetry / direct use (unused in the preset itself).
_force_alpha_term = CurriculumTermCfg(
    func="wbt_training.config_values.force_tracking:ForceAlphaCurriculum",
    params={"hand_force_config": _DEFAULT_HAND_FORCE_CFG},
)


# ---------------------------------------------------------------------------
# 3.5 Termination: maskmimic termination (timeout + bad_tracking) + force_runaway estop
# (keys off F_ext, dormant in sim — F_ext ≤ max_force_n < cutoff — deploy-safety valve).
# ---------------------------------------------------------------------------
def _build_hybrid_termination(hybrid_cfg: HybridContactConfig):
    return _replace(
        g1_29dof_wbt_maskedmimic_termination,
        terms={
            **g1_29dof_wbt_maskedmimic_termination.terms,
            "force_runaway": TerminationTermCfg(
                func="wbt_training.config_values.hybrid:force_runaway", params={"cfg": hybrid_cfg}
            ),
        },
    )


g1_29dof_wbt_maskedmimic_hybrid_termination = _build_hybrid_termination(_DEFAULT_HYBRID_CFG)
g1_29dof_wbt_maskedmimic_hybrid_termination_servo_off = _build_hybrid_termination(_HYBRID_CFG_SERVO_OFF)


__all__ = [
    "JointVelDropoutConfig",
    "MaskedMimicHybridObservationField",
    "force_track_pos_eq2_hybrid_global",
    "force_track_eq2_servo_global",
    "g1_29dof_wbt_observation_maskedmimic_hybrid",
    "g1_29dof_wbt_maskedmimic_hybrid_reward",
    "g1_29dof_wbt_maskedmimic_hybrid_reward_servo_off",
    "g1_29dof_wbt_maskedmimic_hybrid_command",
    "g1_29dof_wbt_maskedmimic_hybrid_command_servo_off",
    "g1_29dof_wbt_maskedmimic_hybrid_curriculum",
    "g1_29dof_wbt_maskedmimic_hybrid_termination",
    "g1_29dof_wbt_maskedmimic_hybrid_termination_servo_off",
    "MASKEDMIMIC_FORCE_HOVER_MODES",
]
