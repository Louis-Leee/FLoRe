"""MaskedMimic + UniFP-style per-hand force-tracking variant.

This module ports the `exp:g1-wbt-2pt-force` force mechanism (UniFP Eq.2
position-shifted hand tracking, F^ext trapezoidal disturbance, actor/critic
F^cmd/F^ext asymmetry) onto the `exp:g1-wbt-maskedmimic` recipe. The *force math*
is byte-identical to `force_tracking.py` (same HandForceInjector, same pulse
schedule, same F^cmd/F^ext physics, same Eq.2 reward kernel `exp(−err_l1·2/σ)`);
only two things change for the maskedmimic context:

1. **Frame**: maskedmimic tracks bodies in the ABSOLUTE world frame
   (`motion_global_body_position_error_exp` reads `mc.body_pos_w`), so the force
   reward uses the GLOBAL Eq.2 variant `force_track_pos_eq2_global` (anchored on
   `mc.body_pos_w`) rather than the yaw-relocated `force_track_pos_eq2`. Likewise
   the baseline body-position reward is swapped for the no-wrist GLOBAL variant so
   Eq.2 owns the wrist position alone.

2. **Hover modes**: the user only trains {2pt, 3pt, 5pt, 29dof-dense}, so the
   hover-mask mode set is restricted to exactly those four
   (`both_hands_only`, `three_point`, `five_point`, `full_joints_pelvis`).

The actor sees the maskedmimic obs PLUS F^cmd (6D, hand-frame-able via
`--F_on_hand_frame`); the critic additionally sees F^ext (UniFP privilege).

OBS CONCAT ORDER: holosoma's ObservationManager concatenates a group's terms in
`sorted(term_keys)` — LEXICOGRAPHIC key order, NOT dict insertion order (see
`managers/observation/manager.py` `compute_group`). The base maskedmimic terms use
zero-padded numeric prefixes `obs00_`..`obs17_`; we name the force terms
`obs98_force_cmd_local` / `obs99_force_ext_local` so they sort STRICTLY AFTER every
base term (`"obs17..." < "obs98..." < "obs99..."`), making the base maskedmimic
obs layout an exact prefix of the force layout. This invariant rests entirely on
the `obsNN_` naming convention: a non-prefixed key (e.g. `"force_cmd_local"`) would
sort BEFORE `"obs00..."` and silently shift the whole maskedmimic block — so never
rename these terms without a high numeric prefix.

Single source of truth for force state stays the env-attached `_wbt_force_*`
buffers owned by `HandForceInjector` / `ForceAlphaCurriculum` (imported, unchanged).
"""

from __future__ import annotations

from dataclasses import replace as _replace
from typing import Any

from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.curriculum import CurriculumTermCfg
from holosoma.config_types.observation import ObsGroupCfg, ObsTermCfg, ObservationManagerCfg
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

from wbt_training.config_values.command import g1_29dof_wbt_maskedmimic_command
from wbt_training.config_values.force_tracking import (
    HandForceConfig,
    HandForceInjector,  # noqa: F401  (referenced by dotted-path in CommandTermCfg)
    g1_29dof_wbt_force_curriculum,
)
from wbt_training.config_values.maskedmimic_observations import (
    DEFAULT_HOVER_MASK_MODES,
    DEFAULT_MASKEDMIMIC_OBS_CONFIG,
    MaskedMimicObservationField,
    _MaskedMimicObservationProvider,
    g1_29dof_wbt_maskedmimic_reward,
    g1_29dof_wbt_observation_maskedmimic,
)

_DEFAULT_HAND_FORCE_CFG = HandForceConfig()


# ---------------------------------------------------------------------------
# 1. Hover-mode restriction: {2pt, 3pt, 5pt, 29dof-dense}.
#
# The user trains only these four conditioning modes. Reusing the four canonical
# entries from DEFAULT_HOVER_MASK_MODES keeps their keypoint/joint regex patterns
# byte-identical to the base maskedmimic preset (so a checkpoint trained on the
# base preset's full mode set still recognises these). Per the design decision,
# "29dof dense" maps to `full_joints_pelvis` (all 29 joints conditioned + pelvis
# keypoint only) — pure joint-space dense tracking.
# ---------------------------------------------------------------------------
MASKEDMIMIC_FORCE_HOVER_MODE_KEYS = (
    "both_hands_only",  # 2pt  : left_hand + right_hand keypoints
    "three_point",  # 3pt  : torso + left_hand + right_hand
    "five_point",  # 5pt  : torso + ankles + hands
    "full_joints_pelvis",  # 29dof: all joints + pelvis keypoint
)

MASKEDMIMIC_FORCE_HOVER_MODES = {k: DEFAULT_HOVER_MASK_MODES[k] for k in MASKEDMIMIC_FORCE_HOVER_MODE_KEYS}

# Obs config identical to the base maskedmimic obs config EXCEPT the hover-mask
# mode set is restricted to the four trained modes. Everything else (future
# steps, masking style, noise) is inherited verbatim so the obs layout/widths are
# unchanged.
MASKEDMIMIC_FORCE_OBS_CONFIG = _replace(
    DEFAULT_MASKEDMIMIC_OBS_CONFIG,
    hover_mask_modes=MASKEDMIMIC_FORCE_HOVER_MODES,
)


# ---------------------------------------------------------------------------
# 2. Force-aware maskedmimic observation provider + term.
#
# The base MaskedMimicObservationField always builds its provider with
# DEFAULT_MASKEDMIMIC_OBS_CONFIG (the full hover-mode set). To restrict the modes
# without touching the base class, we subclass with a force-specific provider
# stored under a DISTINCT env attribute (`_wbt_maskedmimic_force_obs_provider`) so
# the two never collide if both presets are ever imported in one process. The
# provider class itself is reused unchanged — only its `obs_cfg` differs.
# ---------------------------------------------------------------------------
def _get_force_provider(env: Any) -> _MaskedMimicObservationProvider:
    provider = getattr(env, "_wbt_maskedmimic_force_obs_provider", None)
    if provider is None:
        provider = _MaskedMimicObservationProvider(env, obs_cfg=MASKEDMIMIC_FORCE_OBS_CONFIG)
        setattr(env, "_wbt_maskedmimic_force_obs_provider", provider)
    return provider


class MaskedMimicForceObservationField(MaskedMimicObservationField):
    """MaskedMimic obs term whose provider uses the restricted 4-mode hover set.

    Identical to `MaskedMimicObservationField` except it resolves the
    force-specific provider (restricted hover modes) rather than the default one.
    """

    def _provider(self) -> _MaskedMimicObservationProvider:
        if self.provider is None:
            self.provider = _get_force_provider(self.env)
        return self.provider


_MM_FORCE_FIELD = "wbt_training.config_values.maskedmimic_force:MaskedMimicForceObservationField"


def _mm_force_field_term(field: str) -> ObsTermCfg:
    return ObsTermCfg(func=_MM_FORCE_FIELD, params={"field": field}, scale=1.0, noise=0.0)


# Rebuild the maskedmimic actor/critic terms so every MaskedMimicObservationField
# entry routes through the FORCE provider (restricted modes), then append the
# force obs at the end. We copy the base term dicts and re-point only the
# maskedmimic-field terms (leaving the holosoma `motion_command` / proprio terms
# on their original dotted paths, since those are stateless functions).
_BASE_ACTOR_TERMS = g1_29dof_wbt_observation_maskedmimic.groups["actor_obs"].terms
_BASE_CRITIC_TERMS = g1_29dof_wbt_observation_maskedmimic.groups["critic_obs"].terms


def _reroute_mm_terms(terms: dict[str, ObsTermCfg]) -> dict[str, ObsTermCfg]:
    """Return a copy with every MaskedMimicObservationField term re-pointed at the
    force provider field func; all other terms preserved byte-for-byte."""
    out: dict[str, ObsTermCfg] = {}
    for name, term in terms.items():
        if term.func.endswith(":MaskedMimicObservationField"):
            out[name] = _replace(term, func=_MM_FORCE_FIELD)
        else:
            out[name] = term
    return out


# force_cmd_local: scale 0.04, no noise — identical knobs to the 2pt-force preset.
_FORCE_CMD_TERM = ObsTermCfg(
    func="wbt_training.config_values.force_tracking:force_cmd_local", scale=0.04, noise=0.0
)
_FORCE_EXT_TERM = ObsTermCfg(
    func="wbt_training.config_values.force_tracking:force_ext_local", scale=0.04, noise=0.0
)

# Actor: maskedmimic terms (restricted modes) + F^cmd (UniFP: actor sees its own
# compliance target only). The `obs98_` numeric prefix sorts AFTER the base
# `obs00_`..`obs07_` terms under the manager's LEXICOGRAPHIC key sort (NOT insertion
# order), so the maskedmimic obs layout is preserved as a strict prefix.
_actor_obs_maskedmimic_force_terms = {
    **_reroute_mm_terms(_BASE_ACTOR_TERMS),
    "obs98_force_cmd_local": _FORCE_CMD_TERM,
}

# Critic: maskedmimic critic terms (restricted modes) + F^cmd + F^ext (UniFP
# privilege — the value function gets the external disturbance the actor must infer).
_critic_obs_maskedmimic_force_terms = {
    **_reroute_mm_terms(_BASE_CRITIC_TERMS),
    "obs98_force_cmd_local": _FORCE_CMD_TERM,
    "obs99_force_ext_local": _FORCE_EXT_TERM,
}

g1_29dof_wbt_observation_maskedmimic_force = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=1, terms=_actor_obs_maskedmimic_force_terms
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=_critic_obs_maskedmimic_force_terms
        ),
    }
)


# --- --F_on_hand_frame ablation (actor-only hand-frame F^cmd obs) ---
#
# Same contract as force_tracking.with_force_cmd_hand_frame: flip ONLY the actor's
# force_cmd_local term to hand_frame=True, leaving the critic (privileged, stable
# pelvis frame) untouched. Defined here (rather than reusing the 2pt helper) only
# so the no-op message can name the right preset; the math is identical.
def with_force_cmd_hand_frame_mm(observation_cfg: ObservationManagerCfg) -> ObservationManagerCfg:
    """Flip the ACTOR's force_cmd_local obs term to hand frame; critic untouched.

    Mirrors `force_tracking.with_force_cmd_hand_frame` so the shared
    `train_agent._apply_f_on_hand_frame` path works unchanged on this preset
    (that path calls `force_tracking.with_force_cmd_hand_frame`, which keys off
    the `:force_cmd_local` func suffix — the SAME func this preset uses — so the
    flag already works without this helper; it is kept for symmetry / direct use)."""
    groups = dict(observation_cfg.groups)
    actor = groups.get("actor_obs")
    if actor is None:
        return observation_cfg
    new_terms: dict[str, ObsTermCfg] = {}
    changed = False
    for name, term in actor.terms.items():
        if term.func.endswith(":force_cmd_local"):
            new_terms[name] = _replace(term, params={**term.params, "hand_frame": True})
            changed = True
        else:
            new_terms[name] = term
    if not changed:
        return observation_cfg
    groups["actor_obs"] = _replace(actor, terms=new_terms)
    return _replace(observation_cfg, groups=groups)


g1_29dof_wbt_observation_maskedmimic_force_handframe = with_force_cmd_hand_frame_mm(
    g1_29dof_wbt_observation_maskedmimic_force
)


# ---------------------------------------------------------------------------
# 3. Reward: maskedmimic global rewards + global Eq.2 force + feet_slip.
#
# Swaps vs g1_29dof_wbt_maskedmimic_reward:
#   - `motion_global_body_pos` (full-body global position reward) → the no-wrist
#     GLOBAL variant, so Eq.2 owns the wrist position alone (no tug-of-war on the
#     rubber_hand rigid sub-chain — same rationale as the 2pt-force preset, but in
#     the global frame to match maskedmimic).
#   - DROP the dedicated `motion_global_wrist_pos` reward (it would re-impose a raw
#     wrist-position target competing with Eq.2's shifted target). The wrist
#     ORIENTATION / lin-vel / ang-vel rewards are KEPT (orthogonal to Eq.2 — the FK
#     offset cancels so Eq.2 rewards wrist-ORIGIN translation, not rotation; the
#     hand contact pose stays supervised, exactly as in the 2pt-force preset).
#   - ADD `force_track_pos_eq2_global` (w=6.0) + `feet_slip` (w=-2.0), identical
#     weights to the 2pt-force preset.
# All other maskedmimic terms (anchor pos/ori, body ori/vel, action_rate, joint
# limit, undesired_contacts) are inherited UNCHANGED.
# ---------------------------------------------------------------------------
_MM_GLOBAL_BODY_POS = "motion_global_body_pos"
_MM_WRIST_POS = "motion_global_wrist_pos"

g1_29dof_wbt_maskedmimic_force_reward = RewardManagerCfg(
    terms={
        # Inherit every maskedmimic term EXCEPT the full-body global position term
        # (swapped for no-wrist below) and the dedicated wrist-position term (dropped
        # — Eq.2 owns the wrist position).
        **{
            k: v
            for k, v in g1_29dof_wbt_maskedmimic_reward.terms.items()
            if k not in (_MM_GLOBAL_BODY_POS, _MM_WRIST_POS)
        },
        # No-wrist GLOBAL body position reward (replaces motion_global_body_pos).
        "motion_global_body_pos_no_wrist": RewardTermCfg(
            func="wbt_training.config_values.force_tracking:motion_global_body_position_error_exp_no_wrist",
            params={"sigma": 0.3, "exclude_body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link")},
            weight=1.0,  # matches motion_global_body_pos baseline weight
        ),
        # UniFP Eq.2 position-shifted hand tracking — GLOBAL-frame variant (anchored
        # on mc.body_pos_w to match maskedmimic's global rewards). w=6.0 mirrors the
        # 2pt-force preset.
        "force_track_pos_eq2": RewardTermCfg(
            func="wbt_training.config_values.force_tracking:force_track_pos_eq2_global",
            weight=6.0,
            params={"cfg": _DEFAULT_HAND_FORCE_CFG},
        ),
        # Feet-slip penalty (GH parity): func returns POSITIVE magnitude, cfg weight
        # carries the minus. F^ext at the hand pulls the CoM sideways; without this the
        # policy reward-hacks via foot slip (fine in sim, fails on bounded real friction).
        "feet_slip": RewardTermCfg(
            func="wbt_training.config_values.force_tracking:feet_slip",
            weight=-2.0,
            params={"body_names": ("left_ankle_roll_link", "right_ankle_roll_link")},
        ),
    }
)


# ---------------------------------------------------------------------------
# 4. Command: maskedmimic multi-motion command + hand_force_injector.
#
# The maskedmimic command (MaskedMimicMultiMotionCommand) is registered in
# setup/reset/step_terms. We ADD hand_force_injector to setup_terms (buffer alloc +
# substep monkey-patch) AND reset_terms (per-episode pulse restart + force zeroing),
# mirroring g1_29dof_wbt_force_command. step_terms keeps ONLY the maskedmimic command
# (the injector's pulse advances in the substep-0 physics hook, not in step()).
#
# CRITICAL: registering the injector ONLY in setup_terms would leave its reset() as
# dead code (CommandManager skips reset for terms not in reset_terms), leaking the
# trapezoidal pulse + F^ext across episodes → start-of-episode force shock. See the
# force_tracking.py comment on g1_29dof_wbt_force_command.
# ---------------------------------------------------------------------------
_hand_force_injector_term = CommandTermCfg(
    func="wbt_training.config_values.force_tracking:HandForceInjector",
    params={"hand_force_config": _DEFAULT_HAND_FORCE_CFG},
)

g1_29dof_wbt_maskedmimic_force_command = _replace(
    g1_29dof_wbt_maskedmimic_command,
    setup_terms={
        **g1_29dof_wbt_maskedmimic_command.setup_terms,
        "hand_force_injector": _hand_force_injector_term,
    },
    reset_terms={
        **g1_29dof_wbt_maskedmimic_command.reset_terms,
        "hand_force_injector": _hand_force_injector_term,
    },
)


# ---------------------------------------------------------------------------
# 5. Curriculum: identical to the 2pt-force curriculum (force_alpha term added on
# top of the WBT base average-episode tracker). The curriculum is DISABLED by
# default (alpha_step=0.0, force_alpha_init=1.0 → full force from step 0); re-enable
# via HandForceConfig as documented in force_tracking.py.
# ---------------------------------------------------------------------------
g1_29dof_wbt_maskedmimic_force_curriculum = g1_29dof_wbt_force_curriculum

# Re-export the curriculum-term cfg for symmetry with force_tracking (unused here
# but keeps the module self-describing).
_force_alpha_term = CurriculumTermCfg(
    func="wbt_training.config_values.force_tracking:ForceAlphaCurriculum",
    params={"hand_force_config": _DEFAULT_HAND_FORCE_CFG},
)


__all__ = [
    "MASKEDMIMIC_FORCE_HOVER_MODE_KEYS",
    "MASKEDMIMIC_FORCE_HOVER_MODES",
    "MASKEDMIMIC_FORCE_OBS_CONFIG",
    "MaskedMimicForceObservationField",
    "g1_29dof_wbt_observation_maskedmimic_force",
    "g1_29dof_wbt_observation_maskedmimic_force_handframe",
    "with_force_cmd_hand_frame_mm",
    "g1_29dof_wbt_maskedmimic_force_reward",
    "g1_29dof_wbt_maskedmimic_force_command",
    "g1_29dof_wbt_maskedmimic_force_curriculum",
]
