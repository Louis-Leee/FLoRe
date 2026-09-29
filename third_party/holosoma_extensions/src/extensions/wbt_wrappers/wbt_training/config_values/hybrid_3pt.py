"""3-point hybrid position-force controller (`exp:g1-wbt-3pt-hybrid`).

Ports the `exp:g1-wbt-2pt-hybrid` HYBRID controller (causal K_env contact spring +
simulated wrist F/T sensor with F_meas in the actor + Jᵀ contact-gated force servo +
force-runaway estop, Layer A + B) onto the **3-point tracking** base (`g1_wbt_3pt`).
Design: docs/superpowers/specs/2026-07-16-g1-wbt-3pt-hybrid-design.md.

WHY THIS PORT IS TINY: `g1_wbt_2pt_hybrid` and `g1_wbt_3pt` both derive from the SAME
base `g1_wbt` with the SAME reward / command / curriculum / termination — they differ
ONLY in the actor observation. The whole hybrid physics/reward/command/termination stack
in `hybrid.py` is tracking-point-agnostic (it reads world-frame wrist FK + the
`_wbt_force_*` buffers + root_quat, never anything 2pt-specific), and the hybrid CRITIC
obs is identical for 2pt and 3pt. So the ONLY genuinely new code here is:

  1. `g1_29dof_wbt_observation_3pt_hybrid`: a 168D actor obs = the 2pt-hybrid layout with
     vr_2point_{pos,orn} (6+12) swapped for vr_3point_{pos,orn} (9+12). Critic is reused
     from hybrid.py VERBATIM. Zero-padded sorted keys obs00..obs12 (the obs manager
     concatenates in sorted(keys()) alphabetical order — a 1-digit scheme would string-sort
     "obs10" between "obs1" and "obs2" and scramble the vector).
  2. `JointVelDropout`: an ACTOR-only joint-velocity dropout obs term (referenced from
     exp:g1-maskmimic-hybrid's JointVelDropoutConfig + WBT_JV_DROPOUT_* env knobs), so a
     corrupted real Bota reading can't destabilise a velocity-reliant policy. It wraps the
     core `dof_vel` term at the actor's obs11 slot ONLY; the critic's clean dof_vel term is
     untouched (textbook asymmetric actor-critic). DEFAULT DISABLED here (user 2026-07-16) —
     the shipped preset first reproduces 2pt-hybrid behavior; flip WBT_JV_DROPOUT_ENABLED=1
     to turn it on. Obs width is a FIXED 168D whether dropout is on or off.

NON-INVASION: this module only READS hybrid.py / force_tracking.py / observation.py /
maskedmimic_{hybrid,observations}.py. The command / reward / curriculum / termination (and
their servo-off twins) are the exact `hybrid.py` objects, reused by experiment.py directly
(frame-agnostic) — NOT re-wrapped here. exp:g1-wbt-2pt-hybrid, exp:g1-maskmimic-hybrid, and
the shipped mjswan viz are untouched.
"""

from __future__ import annotations

import os

import torch

from holosoma.config_types.observation import ObsGroupCfg, ObservationManagerCfg, ObsTermCfg
from holosoma.managers.observation.base import ObservationTermBase
from holosoma.managers.observation.terms.wbt import dof_vel as _core_dof_vel

from wbt_training.config_values.hybrid import g1_29dof_wbt_observation_2pt_hybrid
from wbt_training.config_values.maskedmimic_hybrid import JointVelDropoutConfig
from wbt_training.config_values.maskedmimic_observations import _uniform_noise_like
from wbt_training.config_values.observation import _actor_obs_3pt_terms

# Base uniform-noise magnitude the core actor dof_vel term applies (observation.py
# obs7_dof_vel: noise=0.5). JointVelDropout applies this noise INTERNALLY (before the
# dropout, so a dropped env is EXACTLY 0 — in-distribution with the deploy-zero) and sets
# noise=0.0 on its ObsTermCfg so the obs manager does not add a second noise pass.
_DOF_VEL_BASE_NOISE: float = 0.5


# ---------------------------------------------------------------------------
# ACTOR joint-velocity dropout obs term (default DISABLED; env-var overridable).
#
# 3pt's dof_vel is a STANDALONE obs term (unlike maskmimic, where it is fused inside a
# provider). So the clean port is a stateful ObservationTermBase that wraps the core
# dof_vel term and is wired ONLY into the ACTOR's obs11 slot; the critic keeps its own
# clean dof_vel (critic_obs_shared_terms) → the dropout is actor-only.
#
# Reuses exp:g1-maskmimic-hybrid's JointVelDropoutConfig dataclass + the SAME env knobs
#   WBT_JV_DROPOUT_ENABLED = 0|1   (DEFAULT 0 here — user 2026-07-16; maskmimic defaults 1)
#   WBT_JV_DROPOUT_PROB    = float in [0,1]
#   WBT_JV_DROPOUT_MODE    = per_episode | per_step
#   WBT_JV_DROPOUT_NOISE   = float >= 0   (extra noise on the SURVIVING envs)
#   WBT_JV_EVAL_ZERO       = 0|1   (inference/eval: zero ALL actor dof_vel)
# ---------------------------------------------------------------------------
def _jv_dropout_config_from_env() -> JointVelDropoutConfig:
    """Build the JointVelDropoutConfig from env vars, DEFAULTING TO DISABLED.

    Read ONCE at import time (module-level), matching how the maskmimic launcher passes its
    knobs. Robust to unset / malformed values (fall back to the disabled-by-default field).
    Deliberately does NOT call maskedmimic_hybrid._jv_dropout_config_from_env, whose default
    is enabled=True — the 3pt-hybrid preset ships with the dropout OFF (user 2026-07-16)."""
    default = JointVelDropoutConfig(enabled=False)

    def _bool(name: str, fallback: bool) -> bool:
        v = os.environ.get(name)
        if v is None or v == "":
            return fallback
        return v.strip().lower() in ("1", "true", "yes", "on")

    def _float(name: str, fallback: float) -> float:
        v = os.environ.get(name)
        if v is None or v == "":
            return fallback
        try:
            return float(v)
        except ValueError:
            return fallback

    mode = os.environ.get("WBT_JV_DROPOUT_MODE", "").strip().lower()
    per_episode = default.per_episode if mode not in ("per_episode", "per_step") else (mode == "per_episode")
    return JointVelDropoutConfig(
        enabled=_bool("WBT_JV_DROPOUT_ENABLED", default.enabled),
        dropout_prob=_float("WBT_JV_DROPOUT_PROB", default.dropout_prob),
        per_episode=per_episode,
        noise_scale=_float("WBT_JV_DROPOUT_NOISE", default.noise_scale),
        eval_zero=_bool("WBT_JV_EVAL_ZERO", default.eval_zero),
    )


_DEFAULT_JV_DROPOUT_CFG = _jv_dropout_config_from_env()


class JointVelDropout(ObservationTermBase):
    """ACTOR-only joint-velocity dropout / zeroing, as a stateful observation term.

    Wraps the core `dof_vel` obs term. Per compute: raw dof_vel → base 0.5 uniform noise
    → dropout (zero the masked envs; optional extra noise on survivors). The base noise is
    applied INTERNALLY (before dropout) so a dropped env reads EXACTLY 0 — in-distribution
    with the deploy-zero (WBT_JV_EVAL_ZERO=1). The ObsTermCfg carries noise=0.0 so the obs
    manager does not double-apply noise.

    Config is the module-level `_DEFAULT_JV_DROPOUT_CFG` (env-var driven at import time),
    matching how maskmimic reads its dropout config. per_episode masks are sampled at
    reset() and held for the whole episode (a fraction of envs run ENTIRE episodes with
    dof_vel≡0, making "zero dof_vel at inference" in-distribution); per_step re-samples
    each compute. PURE w.r.t. the env except its own (num_envs,) episode mask."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.jv_cfg: JointVelDropoutConfig = cfg.params.get("jv_cfg", _DEFAULT_JV_DROPOUT_CFG)
        self.base_noise: float = float(cfg.params.get("base_noise", _DOF_VEL_BASE_NOISE))
        self.num_envs: int = env.num_envs
        self.device = env.device
        # (num_envs,) bool: True ⇒ this env's ACTOR dof_vel is zeroed this episode. Lazily
        # SAMPLED (not zero-filled) so episode-0 envs get a real per-episode draw even if
        # reset() never fires before the first __call__ (mirrors the maskmimic reviewer fix).
        self._jv_dropout_mask: torch.Tensor | None = None

    def _ensure_jv_mask(self) -> torch.Tensor:
        if self._jv_dropout_mask is None:
            if self.jv_cfg.per_episode:
                self._jv_dropout_mask = torch.rand(self.num_envs, device=self.device) < self.jv_cfg.dropout_prob
            else:
                self._jv_dropout_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._jv_dropout_mask

    def _apply_jv_dropout(self, joint_vel: torch.Tensor) -> torch.Tensor:
        """Drop / zero / noise the ACTOR joint velocity. joint_vel: (n, n_dof), already
        carrying the base uniform noise. Mirrors maskedmimic_hybrid._apply_jv_dropout."""
        cfg = self.jv_cfg
        if not cfg.enabled:
            return joint_vel
        if cfg.eval_zero:
            return torch.zeros_like(joint_vel)
        if cfg.per_episode:
            mask = self._ensure_jv_mask()  # (n,) bool
        else:
            mask = torch.rand(self.num_envs, device=self.device) < cfg.dropout_prob  # i.i.d. per step
        keep = (~mask).unsqueeze(-1).to(joint_vel.dtype)
        out = joint_vel * keep  # zero the dropped envs
        if cfg.noise_scale > 0.0:  # extra noise on the SURVIVING envs only
            out = out + _uniform_noise_like(torch.zeros_like(out), cfg.noise_scale) * keep
        return out

    def __call__(self, env, **kwargs) -> torch.Tensor:
        joint_vel = _core_dof_vel(env)
        joint_vel = _uniform_noise_like(joint_vel, self.base_noise)  # base 0.5 actor noise (internal)
        return self._apply_jv_dropout(joint_vel)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if not self.jv_cfg.per_episode:
            return
        mask = self._ensure_jv_mask()
        if env_ids is None:
            resolved = torch.arange(self.num_envs, device=self.device)
        else:
            resolved = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if resolved.numel() == 0:
            return
        mask[resolved] = torch.rand(resolved.numel(), device=self.device) < self.jv_cfg.dropout_prob


# ---------------------------------------------------------------------------
# 168D 3pt-hybrid actor observation + reused hybrid critic.
#
# Actor = 2pt-hybrid layout with vr_2point_{pos,orn} (6+12) → vr_3point_{pos,orn} (9+12),
# so 165D → 168D. Force/servo terms (F^cmd, F^meas, I, Δx) inserted at obs03..obs06 exactly
# as in hybrid.py. dof_vel at obs11 is the JointVelDropout term (actor-only). Zero-padded
# keys obs00..obs12 so the obs manager's sorted-key concat reproduces this on-disk layout.
#
# ⚠️ FIXED WIDTH — ALWAYS 168D regardless of servo_enabled: the servo obs terms (I, Δx) are
# present in EVERY run, zeroed when servo off (buffers allocated unconditionally in the
# injector). obs width does NOT key off servo_enabled → one network + checkpoint lineage for
# servo on/off + the two-phase resume. Likewise dropout on/off keeps obs11 present at 29D.
#
# CRITIC is reused from hybrid.py VERBATIM (critic_obs_shared_terms tracks 3pt via the motion
# command; the force / servo / privileged critic terms are tracking-agnostic). It is NEVER
# subject to the dropout — the critic keeps its clean privileged dof_vel.
# ---------------------------------------------------------------------------
_actor_obs_3pt_hybrid_terms = {
    "obs00_command_lower_body": _actor_obs_3pt_terms["obs0_command_lower_body"],
    "obs01_vr_3point_pos": _actor_obs_3pt_terms["obs1_vr_3point_pos"],
    "obs02_vr_3point_orn": _actor_obs_3pt_terms["obs2_vr_3point_orn"],
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
    "obs07_motion_ref_ori_b": _actor_obs_3pt_terms["obs3_motion_ref_ori_b"],
    "obs08_projected_gravity": _actor_obs_3pt_terms["obs4_projected_gravity"],
    "obs09_base_ang_vel": _actor_obs_3pt_terms["obs5_base_ang_vel"],
    "obs10_dof_pos": _actor_obs_3pt_terms["obs6_dof_pos"],
    # ACTOR-only joint-velocity dropout (base 0.5 noise applied INSIDE the term → noise=0.0
    # here so the manager does not add a second pass). Stateful ObservationTermBase.
    "obs11_dof_vel": ObsTermCfg(func="wbt_training.config_values.hybrid_3pt:JointVelDropout", scale=1.0, noise=0.0),
    "obs12_actions": _actor_obs_3pt_terms["obs8_actions"],
}  # 24 + 9 + 12 + 6 + 6 + 6 + 6 + 6 + 3 + 3 + 29 + 29 + 29 = 168D actor (672D w/ history_length=4)

# Critic reused from the 2pt-hybrid preset VERBATIM (tracking-agnostic).
_critic_obs_3pt_hybrid_terms = g1_29dof_wbt_observation_2pt_hybrid.groups["critic_obs"].terms

g1_29dof_wbt_observation_3pt_hybrid = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_3pt_hybrid_terms
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=10, terms=_critic_obs_3pt_hybrid_terms
        ),
    }
)

__all__ = ["JointVelDropout", "JointVelDropoutConfig", "g1_29dof_wbt_observation_3pt_hybrid"]
