"""Relative-frame TCP (fingertip) precision reward for the single-actor
`exp:g1-wbt-2pt-force-upperbody` preset, plus the assembled reward + termination cfgs.

WHY RELATIVE (not the global falcon_force.tcp_track_*): this preset forks the
relative-frame `g1_wbt_2pt_force` and does NOT pin absolute base yaw. The global TCP
funcs compare ABSOLUTE motion quats (`mc.body_quat_w`) against the live robot — a free
yaw offset then injects phantom error. These variants use the YAW-RELOCATED motion ref
(`mc.body_pos_relative_w` / `mc.body_quat_relative_w`), matching `force_track_pos_eq2`
(force_tracking.py) and the core relative orientation reward (wbt.py). See the
2026-07-06 design spec §3.1 and the Opus review disposition (F3/F4).

TCP position is the manipulation signal: the ref TCP rotates the 0.215 m fingertip
offset by the (yaw-relocated) MOTION wrist quat while cur rotates it by the LIVE wrist
quat, so a wrist-orientation error surfaces as a fingertip-position error (the offset
does NOT cancel — the opposite of Eq.2, which cancels it). The origin term still agrees
with Eq.2 because both anchor on `body_pos_relative_w` and both apply the same
`(F^cmd+F^ext)/K` shift. F=0 ⇒ target == raw relative fingertip ref (F=0 invariant).

Quaternion conventions: live `_robot.data.body_quat_w` is WXYZ; motion-command quats
(`body_quat_relative_w`, `robot_body_quat_w`) are XYZW. XYZW→WXYZ before `_quat_apply_wxyz`;
`quat_error_magnitude(..., w_last=True)` for XYZW orientation error.

PURE READERS — never advance the force pulse (single-writer rule; only HandForceInjector).
"""

from __future__ import annotations

from dataclasses import replace as _replace
from typing import Any

import torch

from holosoma.config_types.reward import RewardTermCfg
from holosoma.config_values.wbt.g1.termination import g1_29dof_wbt_termination
from wbt_training.config_values.falcon_force import _ensure_tcp_offsets_b, _TCP_OFFSETS_B, _xyzw_to_wxyz
from wbt_training.config_values.force_tracking import (
    HandForceConfig,
    _ensure_force_isaac_indices,
    _ensure_force_motion_indices,
    _quat_apply_wxyz,
    g1_29dof_wbt_2pt_force_reward,
)

# Shared default force config (frozen). Threaded into the TCP reward terms for parity
# with the inherited Eq.2 term (force_tracking.py wires the same _DEFAULT_HAND_FORCE_CFG),
# so a run that overrides HandForceConfig would flow the same cfg through all force terms.
_DEFAULT_HAND_FORCE_CFG = HandForceConfig()


def _tcp_cur_pos_w(env: Any, cfg: HandForceConfig, tcp_off_b: torch.Tensor) -> torch.Tensor:
    """Live world-frame TCP position via FK on wrist_yaw: wrist_pos_w + R(q_live_wxyz) @ off."""
    body_ids = _ensure_force_isaac_indices(env, cfg.body_names)
    wrist_pos_w = env.simulator._robot.data.body_pos_w[:, body_ids, :]
    wrist_quat_wxyz = env.simulator._robot.data.body_quat_w[:, body_ids, :]  # WXYZ
    n = len(cfg.body_names)
    off = tcp_off_b.unsqueeze(0).expand(env.num_envs, n, -1)
    return wrist_pos_w + _quat_apply_wxyz(wrist_quat_wxyz, off)


def _tcp_ref_pos_w_relative(env: Any, cfg: HandForceConfig, tcp_off_b: torch.Tensor) -> torch.Tensor:
    """Reference world-frame TCP via FK on the YAW-RELOCATED motion wrist pose:
    body_pos_relative_w + R(q_relative_motion) @ off. Uses the RELATIVE motion quat
    (XYZW→WXYZ) so the offset does NOT cancel (the orientation-lever signal) yet the
    lever points along the robot's yaw-relocated frame (no free-yaw cross-talk)."""
    mc = env.command_manager.get_state("motion_command")
    fm_idx = _ensure_force_motion_indices(env, cfg.body_names)
    wrist_pos_ref_w = mc.body_pos_relative_w[:, fm_idx, :]  # yaw-relocated ref
    wrist_quat_rel_xyzw = mc.body_quat_relative_w[:, fm_idx, :]  # XYZW
    wrist_quat_rel_wxyz = _xyzw_to_wxyz(wrist_quat_rel_xyzw)
    n = len(cfg.body_names)
    off = tcp_off_b.unsqueeze(0).expand(env.num_envs, n, -1)
    return wrist_pos_ref_w + _quat_apply_wxyz(wrist_quat_rel_wxyz, off)


def tcp_track_pos_exp_relative(
    env: Any,
    *,
    cfg: HandForceConfig | None = None,
    tcp_offsets_b: tuple[tuple[float, float, float], ...] = _TCP_OFFSETS_B,
    sigma: float = 0.15,
) -> torch.Tensor:
    """TCP (fingertip) position-tracking reward, Eq.2-shifted, RELATIVE (yaw-relocated) frame.

    target_tcp_w = ref_tcp_w + (F^cmd_w + F^ext_w) / K
    err_l1       = Σ_{h∈L,R} Σ_xyz | cur_tcp_h^w − target_tcp_h^w |
    r            = exp(−err_l1 · 2 / σ)          (single scalar per env)
    """
    cfg = cfg or HandForceConfig()
    tcp_off = _ensure_tcp_offsets_b(env, tcp_offsets_b)  # cached (2,3) tensor, no per-step realloc
    ref_tcp_w = _tcp_ref_pos_w_relative(env, cfg, tcp_off)
    f_total_w = env._wbt_force_cmd_w + env._wbt_force_ext_w  # world frame
    target_tcp_w = ref_tcp_w + f_total_w / cfg.stiffness_K
    cur_tcp_w = _tcp_cur_pos_w(env, cfg, tcp_off)
    err_l1 = (cur_tcp_w - target_tcp_w).abs().sum(dim=(-2, -1))  # (n,)
    return torch.exp(-err_l1 * 2.0 / sigma)


def tcp_track_orn_exp_relative(env: Any, *, cfg: HandForceConfig | None = None, sigma: float = 0.25) -> torch.Tensor:
    """TCP orientation-tracking reward, RELATIVE (yaw-relocated) frame.

    err[n, h] = quat_error_magnitude(body_quat_relative_w[n,h], robot_body_quat_w[n,h]) ** 2
    r[n]      = exp(−mean_h err[n, h] / σ²)

    Compares the YAW-RELOCATED motion wrist quat against the live robot wrist quat (both
    XYZW), matching the core relative orientation reward — NOT the absolute body_quat_w
    the global falcon_force.tcp_track_orn_exp uses.
    """
    cfg = cfg or HandForceConfig()
    from holosoma.utils.rotations import quat_error_magnitude

    mc = env.command_manager.get_state("motion_command")
    fm_idx = _ensure_force_motion_indices(env, cfg.body_names)
    q_ref = mc.body_quat_relative_w[:, fm_idx, :]  # XYZW, yaw-relocated
    q_live = mc.robot_body_quat_w[:, fm_idx, :]  # XYZW, live robot
    error = quat_error_magnitude(q_ref, q_live) ** 2  # (n, h)
    return torch.exp(-error.mean(-1) / sigma**2)


# ---------------------------------------------------------------------------
# Assembled reward: inherited 2pt-force reward + relative-frame TCP precision.
# The 2pt-force reward already provides the stand mechanism (leg + pelvis/torso
# clip-tracking) and the upper-body 2pt + Eq.2 + wrist-orientation tracking; we
# ADD only the two TCP terms. Eq.2 weight (6.0), curriculum (off), and every other
# term are inherited UNCHANGED (match 2pt-force per the design decision).
# ---------------------------------------------------------------------------
g1_29dof_wbt_2pt_force_upperbody_reward = _replace(
    g1_29dof_wbt_2pt_force_reward,
    terms={
        **g1_29dof_wbt_2pt_force_reward.terms,
        # Fingertip translation precision (Eq.2-shifted, relative frame). σ=0.15 m
        # is tighter than the 12-body relative pos term (σ=0.3) and the wrist-owning
        # Eq.2 (σ=0.3), so this refines the fingertip landing once the arm is close.
        "tcp_track_pos": RewardTermCfg(
            func="wbt_training.config_values.upperbody:tcp_track_pos_exp_relative",
            weight=2.0,
            params={"sigma": 0.15, "cfg": _DEFAULT_HAND_FORCE_CFG},
        ),
        # Fingertip/grasp orientation precision (relative frame). σ=0.25 rad is
        # tighter than the inherited full-14 relative orientation term (σ=0.4); both
        # are yaw-relocated relative-frame so they agree in direction (intentional
        # loose-global + tight-TCP double supervision, not a conflict).
        "tcp_track_orn": RewardTermCfg(
            func="wbt_training.config_values.upperbody:tcp_track_orn_exp_relative",
            weight=1.5,
            params={"sigma": 0.25, "cfg": _DEFAULT_HAND_FORCE_CFG},
        ),
    },
)


# ---------------------------------------------------------------------------
# Termination: drop the two ankles from the 0.25 m body-position gate (a small
# standing-vs-clip ankle deviation must not early-terminate) — keep the wrists.
# NESTED replace: bad_motion_body_pos_body_names lives in terms["bad_tracking"].params,
# not as a top-level TerminationManagerCfg field. bad_ref_pos (z-only height-collapse
# detector on torso_link) + bad_ref_ori (gravity-tilt) are the fall detectors, kept.
# ---------------------------------------------------------------------------
_upperbody_bad_tracking = g1_29dof_wbt_termination.terms["bad_tracking"]
g1_29dof_wbt_upperbody_termination = _replace(
    g1_29dof_wbt_termination,
    terms={
        **g1_29dof_wbt_termination.terms,
        "bad_tracking": _replace(
            _upperbody_bad_tracking,
            params={
                **_upperbody_bad_tracking.params,
                "bad_motion_body_pos_body_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
            },
        ),
    },
)


__all__ = [
    "tcp_track_pos_exp_relative",
    "tcp_track_orn_exp_relative",
    "g1_29dof_wbt_2pt_force_upperbody_reward",
    "g1_29dof_wbt_upperbody_termination",
]
