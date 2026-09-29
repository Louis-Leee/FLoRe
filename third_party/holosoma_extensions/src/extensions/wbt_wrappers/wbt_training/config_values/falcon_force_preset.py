"""Assembled config for ``exp:g1-wbt-2pt-force-falcon`` — the FALCON-style dual-agent
UniFP force-tracking recipe. Purely additive: built by ``dataclasses.replace`` off the
``g1_wbt_maskedmimic_force`` pieces, adding the four recipe-specific deltas:

  1. **command**  — the maskedmimic multi-motion command + HandForceInjector (both
     reused verbatim) PLUS a ``FalconVelocityCommand`` term (name ``locomotion_command``)
     that drives the lower agent and publishes ``env.stance_mask``.
  2. **observation** — the maskedmimic-force obs with the actor F^cmd term flipped to the
     SENSOR frame (req 1), a velocity-command term for the lower agent, and TCP obs
     deferred (the TCP signal lives in the reward, not the obs, for the lean pass).
  3. **reward** — every maskedmimic-force term TAGGED lower/upper for the
     ``DualGroupRewardManager``, plus the lean velocity-command locomotion terms
     (lower) and the TCP precision terms (upper). Eq.2 + all UniFP force math UNCHANGED.
  4. **algo**  — the dual-agent ``PPODecoupled`` (two actors + two critics).

The env class is ``FalconDecoupledManager`` (swaps in DualGroupRewardManager + action
slice maps + stance_mask). Nothing here mutates ``g1_wbt_maskedmimic_force`` or any
existing preset — all objects are new (frozen-dataclass ``replace`` returns copies).
"""

from __future__ import annotations

from dataclasses import replace as _replace

from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.observation import ObservationManagerCfg, ObsTermCfg
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

from wbt_training.config_values.falcon_force import UPPER_BODY_TAG, LOWER_BODY_TAG
from wbt_training.config_values.force_tracking import HandForceConfig
from wbt_training.config_values.maskedmimic_force import (
    g1_29dof_wbt_maskedmimic_force_command,
    g1_29dof_wbt_maskedmimic_force_reward,
    g1_29dof_wbt_observation_maskedmimic_force,
)

_DEFAULT_HAND_FORCE_CFG = HandForceConfig()

# FALCON velocity-command ranges + stand probability (from the FALCON G1 config).
_VELOCITY_COMMAND_RANGES = {
    "lin_vel_x": (-1.0, 1.0),
    "lin_vel_y": (-1.0, 1.0),
    "ang_vel_yaw": (-1.0, 1.0),
}
_STAND_PROB = 0.4
_LOCO_RESAMPLE_S = 10.0

# ---------------------------------------------------------------------------
# 1. Command: maskedmimic-force command (motion + HandForceInjector) + velocity command.
#    FalconVelocityCommand is registered under `locomotion_command` so
#    command_manager.commands resolves for the reused velocity-tracking rewards.
# ---------------------------------------------------------------------------
_velocity_command_term = CommandTermCfg(
    func="wbt_training.config_values.falcon_locomotion:FalconVelocityCommand",
    params={
        "command_ranges": _VELOCITY_COMMAND_RANGES,
        "stand_prob": _STAND_PROB,
        "command_dim": 3,
    },
)

g1_29dof_wbt_falcon_force_command = _replace(
    g1_29dof_wbt_maskedmimic_force_command,
    # `locomotion_command_resampling_time` MUST live at the MANAGER level (not the
    # term params): LocomotionCommand.step() reads it from CommandManager.command_cfg
    # (built from CommandManagerCfg.params), NOT from the term's own params. Placing it
    # in the term params makes the mid-episode resample branch dead (returns early on
    # resample_time=None), so each env holds one walk/stand command for the whole
    # episode. Mirrors the canonical quadruped config's manager-level placement.
    params={
        **g1_29dof_wbt_maskedmimic_force_command.params,
        "locomotion_command_resampling_time": _LOCO_RESAMPLE_S,
    },
    setup_terms={
        **g1_29dof_wbt_maskedmimic_force_command.setup_terms,
        "locomotion_command": _velocity_command_term,
    },
    reset_terms={
        **g1_29dof_wbt_maskedmimic_force_command.reset_terms,
        "locomotion_command": _velocity_command_term,
    },
    step_terms={
        **g1_29dof_wbt_maskedmimic_force_command.step_terms,
        "locomotion_command": _velocity_command_term,
    },
)

# ---------------------------------------------------------------------------
# 2. Observation: sensor-frame actor F^cmd + velocity-command term for the lower agent.
#    Start from the maskedmimic-force obs, flip the actor force_cmd_local to sensor
#    frame, and append the velocity command (obs99_ so it sorts last). Critic keeps
#    pelvis-frame F^cmd + F^ext (UniFP privilege) untouched.
# ---------------------------------------------------------------------------
from wbt_training.config_values.force_tracking import with_force_cmd_sensor_frame as _with_sensor  # noqa: E402

_obs_sensorframe = _with_sensor(g1_29dof_wbt_observation_maskedmimic_force)

_VELOCITY_CMD_OBS = ObsTermCfg(
    func="wbt_training.config_values.falcon_locomotion:velocity_command", scale=1.0, noise=0.0
)


def _append_velocity_command(obs_cfg: ObservationManagerCfg) -> ObservationManagerCfg:
    """Append the 3-D velocity command to BOTH actor and critic groups (obs97_ prefix,
    sorts before the obs98/99 force terms but after the maskedmimic obs00.. terms)."""
    groups = dict(obs_cfg.groups)
    for grp_name in ("actor_obs", "critic_obs"):
        grp = groups[grp_name]
        groups[grp_name] = _replace(grp, terms={"obs97_velocity_command": _VELOCITY_CMD_OBS, **grp.terms})
    return _replace(obs_cfg, groups=groups)


g1_29dof_wbt_observation_falcon_force = _append_velocity_command(_obs_sensorframe)

# ---------------------------------------------------------------------------
# 3. Reward: tag every maskedmimic-force term lower/upper for DualGroupRewardManager,
#    add the lean velocity-command locomotion terms (lower) + TCP precision (upper).
# ---------------------------------------------------------------------------
# Routing of the inherited maskedmimic-force terms: manipulation / hand / whole-body
# tracking → UPPER; locomotion-stability penalties → LOWER. (Whole-body tracking terms
# that touch both halves are routed UPPER so the manipulation agent owns pose fidelity;
# the lower agent is shaped by the velocity-command terms below. This split is a tuning
# choice, easily revisited.)
_UPPER_INHERITED = {
    "motion_global_anchor_pos",
    "motion_global_anchor_ori",
    "motion_global_body_pos_no_wrist",
    "motion_global_body_ori",
    "motion_global_body_lin_vel",
    "motion_global_body_ang_vel",
    "motion_global_wrist_ori",
    "motion_global_wrist_lin_vel",
    "motion_global_wrist_ang_vel",
    "force_track_pos_eq2",
}
_LOWER_INHERITED = {
    "feet_slip",
    "action_rate_l2",
    "joint_limit",
    "undesired_contacts",
}


def _tag(term: RewardTermCfg, group: str) -> RewardTermCfg:
    tag = UPPER_BODY_TAG if group == "upper" else LOWER_BODY_TAG
    tags = list(term.tags) if term.tags else []
    if tag not in tags:
        tags = tags + [tag]
    return _replace(term, tags=tags)


_tagged_inherited: dict[str, RewardTermCfg] = {}
for _name, _term in g1_29dof_wbt_maskedmimic_force_reward.terms.items():
    _grp = "upper" if _name in _UPPER_INHERITED else "lower" if _name in _LOWER_INHERITED else "upper"
    _tagged_inherited[_name] = _tag(_term, _grp)

# New LOWER-body velocity-command locomotion terms (lean set).
_LOCO_TERMS = {
    "loco_tracking_lin_vel": _tag(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            params={"tracking_sigma": 0.25},
            weight=1.5,
        ),
        "lower",
    ),
    "loco_tracking_ang_vel": _tag(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            params={"tracking_sigma": 0.25},
            weight=0.75,
        ),
        "lower",
    ),
    "loco_ang_vel_xy": _tag(
        RewardTermCfg(func="wbt_training.config_values.falcon_locomotion:base_ang_vel_xy_penalty", weight=-0.05),
        "lower",
    ),
    "loco_orientation": _tag(
        RewardTermCfg(func="wbt_training.config_values.falcon_locomotion:upright_orientation", weight=-1.0),
        "lower",
    ),
    "loco_alive": _tag(
        RewardTermCfg(func="wbt_training.config_values.falcon_locomotion:alive", weight=0.15),
        "lower",
    ),
}

# New UPPER-body TCP precision terms.
_TCP_TERMS = {
    "tcp_track_pos": _tag(
        RewardTermCfg(
            func="wbt_training.config_values.falcon_force:tcp_track_pos_exp",
            params={"cfg": _DEFAULT_HAND_FORCE_CFG},
            weight=2.0,
        ),
        "upper",
    ),
    "tcp_track_orn": _tag(
        RewardTermCfg(
            func="wbt_training.config_values.falcon_force:tcp_track_orn_exp",
            params={"cfg": _DEFAULT_HAND_FORCE_CFG},
            weight=1.5,
        ),
        "upper",
    ),
}

g1_29dof_wbt_falcon_force_reward = RewardManagerCfg(
    terms={**_tagged_inherited, **_LOCO_TERMS, **_TCP_TERMS},
    only_positive_rewards=g1_29dof_wbt_maskedmimic_force_reward.only_positive_rewards,
)


__all__ = [
    "g1_29dof_wbt_falcon_force_command",
    "g1_29dof_wbt_observation_falcon_force",
    "g1_29dof_wbt_falcon_force_reward",
]
