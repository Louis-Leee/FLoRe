"""FALCON-style decoupled UniFP force-tracking recipe — additive building blocks.

This module hosts the pieces that make ``exp:g1-wbt-2pt-force-falcon`` different
from the ``exp:g1-wbt-maskedmimic-force`` base it is built on, WITHOUT touching any
existing preset (hard additive constraint). It currently owns:

* **TCP-frame precision reward** (requirement #3): a fingertip-midpoint (TCP)
  translation + orientation tracking reward for the upper body, on top of the
  unchanged UniFP Eq.2 hand-compliance reward.

Later steps add the per-group reward manager, the decoupled env class, and the
FALCON lower-body locomotion command/reward here or in sibling modules.

── Why a TCP reward is NOT redundant with Eq.2 ──
The UniFP Eq.2 reward (``force_track_pos_eq2_global``) deliberately CANCELS the
wrist→hand offset: it rotates the offset by the SAME live wrist quaternion on both
the ``cur`` and ``ref`` sides, so the residual ``cur_w − target_w`` is a pure
wrist-ORIGIN translation error (reach the arm out by ``F/K``), orthogonal to wrist
orientation. For manipulation we ALSO want the fingertip (TCP) to land precisely,
which depends on wrist orientation through the ~0.215 m lever arm. So the TCP
translation reward does the OPPOSITE of Eq.2's cancel-trick:

    cur_tcp_w = wrist_pos_w        + R(q_wrist_LIVE)   @ tcp_offset_b
    ref_tcp_w = wrist_pos_ref_w    + R(q_wrist_MOTION) @ tcp_offset_b     # motion quat!

so a wrist-orientation error shows up as a fingertip-position error. The target is
Eq.2-SHIFTED by the same ``(F^cmd+F^ext)/K`` so the origin component AGREES with
Eq.2 (no tug-of-war) and the ONLY new signal is the orientation-lever term:

    cur_tcp − (ref_tcp + F/K)
      = [cur_wrist − (ref_wrist + F/K)]         # == Eq.2 origin residual (reinforcing)
      + (R_live − R_motion) @ tcp_offset_b      # NEW: orientation-lever precision

F^cmd=F^ext=0 ⇒ target = raw motion TCP ⇒ pure fingertip tracking (F=0 invariant).

Frame: GLOBAL (absolute ``mc.body_pos_w`` / ``mc.body_quat_w``), matching the
maskedmimic-force base's ``motion_global_*`` rewards and ``force_track_pos_eq2_global``.

Quaternion conventions (the live trap): the simulator's ``body_quat_w`` is WXYZ
(IsaacSim native); the motion command's ``body_quat_w`` / ``robot_body_quat_w`` are
XYZW (converted at load, wbt.py:138). We keep the module omni-free (MagicMock-testable)
by reusing ``force_tracking._quat_apply_wxyz`` for the live side and converting the
motion XYZW quat to WXYZ before applying; orientation error uses holosoma's
``quat_error_magnitude`` (w_last=True, XYZW).
"""

from __future__ import annotations

from typing import Any

import torch

from wbt_training.config_values.force_tracking import (
    HandForceConfig,
    _ensure_force_isaac_indices,
    _ensure_force_motion_indices,
    _quat_apply_wxyz,
)

# Reward-group routing tag: any RewardTermCfg whose `tags` contains this string is
# routed to the UPPER-body reward stream; every other term routes to LOWER. Chosen
# as a plain tag so no new config type is needed (RewardTermCfg.tags already exists).
UPPER_BODY_TAG = "upper_body"
LOWER_BODY_TAG = "lower_body"  # optional/explicit; absence of UPPER_BODY_TAG ⇒ lower

# TCP (fingertip grasp midpoint) offset from each wrist_yaw_link local frame, per
# hand. ~0.215 m forward along +x per the committed g1_29dof_yam.urdf comment
# ("grasp/TCP fingertips reach ~0.215 m forward of {side}_wrist_yaw_link along +x");
# the rubber_hand chain is axis-aligned with wrist_yaw (R(rubber_hand)==R(wrist_yaw)),
# so the grasp midpoint sits on the wrist +x axis (y,z ≈ 0). Derived from the LIVE
# URDF, NOT the stale body_offsets_b=0.0815 (which is the gripper BASE, not the TCP).
# Shape: (num_hands, 3) in wrist_yaw_link local frame.
_TCP_OFFSETS_B: tuple[tuple[float, float, float], ...] = ((0.215, 0.0, 0.0), (0.215, 0.0, 0.0))

# Default precision sigmas: tighter than the base maskedmimic wrist reward
# (pos σ=0.3, ori σ=0.4) so the TCP term demands manipulation-grade accuracy.
_TCP_POS_SIGMA = 0.15  # m
_TCP_ORN_SIGMA = 0.25  # rad


def _ensure_tcp_offsets_b(env: Any, tcp_offsets_b: tuple[tuple[float, float, float], ...]) -> torch.Tensor:
    """Cache the per-hand TCP offset tensor on env (parallels _ensure_force_offsets_b)."""
    off = getattr(env, "_wbt_tcp_offsets_b", None)
    if off is not None and isinstance(off, torch.Tensor):
        return off
    off = torch.tensor(tcp_offsets_b, dtype=torch.float, device=env.device)
    env._wbt_tcp_offsets_b = off
    return off


def _xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Convert an (..., 4) xyzw quaternion to wxyz (so _quat_apply_wxyz can use it)."""
    return q[..., [3, 0, 1, 2]]


def _tcp_cur_pos_w(env: Any, cfg: HandForceConfig, tcp_off_b: torch.Tensor) -> torch.Tensor:
    """Live world-frame TCP (fingertip) position via FK on wrist_yaw.

    cur_tcp_w = wrist_pos_w + R(q_wrist_LIVE) @ tcp_offset_b.
    Live wrist quat is WXYZ (IsaacSim native). Shape (num_envs, num_hands, 3).
    """
    body_ids = _ensure_force_isaac_indices(env, cfg.body_names)
    wrist_pos_w = env.simulator._robot.data.body_pos_w[:, body_ids, :]
    wrist_quat_wxyz = env.simulator._robot.data.body_quat_w[:, body_ids, :]
    n = len(cfg.body_names)
    off = tcp_off_b.unsqueeze(0).expand(env.num_envs, n, -1)
    return wrist_pos_w + _quat_apply_wxyz(wrist_quat_wxyz, off)


def _tcp_ref_pos_w_global(env: Any, cfg: HandForceConfig, tcp_off_b: torch.Tensor) -> torch.Tensor:
    """Reference world-frame TCP position via FK on the MOTION wrist pose (global frame).

    ref_tcp_w = wrist_pos_ref_w + R(q_wrist_MOTION) @ tcp_offset_b.

    Uses the MOTION wrist quaternion (NOT the live one) so the offset does NOT cancel
    against `_tcp_cur_pos_w` — that is the whole point: a wrist-orientation error then
    surfaces as a fingertip-position error (the manipulation-precision signal), unlike
    Eq.2 which uses the live quat on both sides to cancel the offset. Motion quat is
    XYZW; converted to WXYZ for `_quat_apply_wxyz`. Anchor is the absolute global motion
    ref `mc.body_pos_w` (matches the maskedmimic-force base's global-frame rewards).
    Shape (num_envs, num_hands, 3).
    """
    mc = env.command_manager.get_state("motion_command")
    fm_idx = _ensure_force_motion_indices(env, cfg.body_names)
    wrist_pos_ref_w = mc.body_pos_w[:, fm_idx, :]  # (n, h, 3) absolute global motion ref
    wrist_quat_motion_xyzw = mc.body_quat_w[:, fm_idx, :]  # (n, h, 4) XYZW motion ref
    wrist_quat_motion_wxyz = _xyzw_to_wxyz(wrist_quat_motion_xyzw)
    n = len(cfg.body_names)
    off = tcp_off_b.unsqueeze(0).expand(env.num_envs, n, -1)
    return wrist_pos_ref_w + _quat_apply_wxyz(wrist_quat_motion_wxyz, off)


def tcp_track_pos_exp(
    env: Any,
    *,
    cfg: HandForceConfig | None = None,
    tcp_offsets_b: tuple[tuple[float, float, float], ...] = _TCP_OFFSETS_B,
    sigma: float = _TCP_POS_SIGMA,
) -> torch.Tensor:
    """TCP (fingertip) position-tracking reward, Eq.2-shifted, GLOBAL frame.

        target_tcp_w = ref_tcp_w + (F^cmd_w + F^ext_w) / K
        err_l1       = Σ_{h∈L,R} Σ_xyz | cur_tcp_h^w − target_tcp_h^w |
        r            = exp(−err_l1 · 2 / σ)          (single scalar per env)

    Same reduction shape as `force_track_pos_eq2_global` (L1 over hands×xyz, scale
    2/σ). The Eq.2 shift `(F^cmd+F^ext)/K` reuses the world-frame force buffers
    UNCHANGED (requirement #7 — the force math is not modified), so the origin
    component agrees with Eq.2 and only the orientation-lever term is new. F=0 ⇒
    target = raw motion TCP ⇒ pure fingertip tracking. PURE READER (does not advance
    the pulse — single-writer rule, Math Foundation §8).
    """
    cfg = cfg or HandForceConfig()
    tcp_off = _ensure_tcp_offsets_b(env, tcp_offsets_b)
    ref_tcp_w = _tcp_ref_pos_w_global(env, cfg, tcp_off)
    f_total_w = env._wbt_force_cmd_w + env._wbt_force_ext_w  # (n, h, 3) world frame
    target_tcp_w = ref_tcp_w + f_total_w / cfg.stiffness_K
    cur_tcp_w = _tcp_cur_pos_w(env, cfg, tcp_off)
    err_l1 = (cur_tcp_w - target_tcp_w).abs().sum(dim=(-2, -1))  # sum over (hands, xyz) → (n,)
    return torch.exp(-err_l1 * 2.0 / sigma)


def tcp_track_orn_exp(
    env: Any,
    *,
    cfg: HandForceConfig | None = None,
    sigma: float = _TCP_ORN_SIGMA,
) -> torch.Tensor:
    """TCP (grasp-axis) orientation-tracking reward, GLOBAL frame.

        err[n, h] = quat_error_magnitude(q_wrist_motion[n, h], q_wrist_live[n, h]) ** 2
        r[n]      = exp(−mean_h err[n, h] / σ²)

    The TCP frame is rigidly axis-aligned with wrist_yaw on the yam gripper
    (R(rubber_hand)==R(wrist_yaw)), so the grasp-axis orientation IS the wrist
    orientation; this is a tighter-σ (manipulation-grade) orientation term than the
    base maskedmimic wrist-ori reward, kept as a dedicated TCP term so the recipe is
    self-documenting and future rotated grasp frames can extend it. Compares the
    MOTION wrist quat (`mc.body_quat_w`, XYZW) to the LIVE robot wrist quat
    (`mc.robot_body_quat_w`, XYZW) via holosoma's `quat_error_magnitude` (w_last=True).
    PURE READER.
    """
    cfg = cfg or HandForceConfig()
    from holosoma.utils.rotations import quat_error_magnitude

    mc = env.command_manager.get_state("motion_command")
    fm_idx = _ensure_force_motion_indices(env, cfg.body_names)
    q_motion = mc.body_quat_w[:, fm_idx, :]  # XYZW
    q_live = mc.robot_body_quat_w[:, fm_idx, :]  # XYZW
    error = quat_error_magnitude(q_motion, q_live) ** 2  # (n, h)
    return torch.exp(-error.mean(-1) / sigma**2)


# ===========================================================================
# DualGroupRewardManager — per-group (lower_body / upper_body) reward streams for
# the FALCON-style dual-agent recipe, WITHOUT touching core RewardManager.
#
# The core RewardManager sums every term into ONE scalar `_reward_buf` (returned
# into env.rew_buf, base_task:501). For dual-agent PPO we ALSO need two summed
# streams — one per agent — so each agent's critic learns its own value / GAE:
#   * lower_body agent  → locomotion reward group (velocity/gait/feet/orientation…)
#   * upper_body agent  → manipulation group (arm tracking + Eq.2 force + TCP)
#
# This subclass is purely additive:
#   * It builds a term-name → group map from each term's `RewardTermCfg.tags`
#     (a pre-existing field): a tag containing UPPER_BODY_TAG routes UPPER, else LOWER.
#   * It overrides compute() to accumulate each term's scaled reward into TWO extra
#     buffers (`_group_reward_buf["lower"/"upper"]`) IN THE SAME LOOP, while still
#     returning the unchanged scalar sum (so the single-scalar env.rew_buf path and
#     every existing consumer are byte-identical).
#   * The dual-agent algo reads `env.reward_manager.group_reward_buf` per step.
#
# Routing is by term NAME, resolved once against `self._term_names` (which already
# EXCLUDES weight==0 terms — _initialize_terms skips them). only_positive_rewards
# is applied to BOTH group buffers independently (mirroring the scalar clip) so a
# per-group reward never goes negative when the global cfg requests non-negativity.
# ===========================================================================


def _resolve_reward_group(term_cfg: Any) -> str:
    """Map a RewardTermCfg to 'upper' or 'lower' via its `tags` (default lower)."""
    tags = getattr(term_cfg, "tags", None) or []
    return "upper" if UPPER_BODY_TAG in tags else "lower"


def make_dual_group_reward_manager_class() -> type:
    """Build the DualGroupRewardManager class, importing the core base lazily.

    Deferred import keeps this module omni-free / MagicMock-testable at import time
    (holosoma.managers.reward.manager transitively pulls sim deps in some builds).
    Returns a subclass of the core RewardManager.
    """
    from holosoma.managers.reward.manager import RewardManager as _RewardManager

    class DualGroupRewardManager(_RewardManager):
        """RewardManager that also produces per-group (lower/upper) summed rewards."""

        def __init__(self, cfg: Any, env: Any, device: str):
            super().__init__(cfg, env, device)
            # term-name → 'lower'/'upper', built from tags on the ACTIVE terms only.
            self._reward_group_of: dict[str, str] = {
                name: _resolve_reward_group(tc) for name, tc in zip(self._term_names, self._term_cfgs)
            }
            self._group_reward_buf: dict[str, torch.Tensor] = {
                "lower": torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device),
                "upper": torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device),
            }

        @property
        def group_reward_buf(self) -> dict[str, torch.Tensor]:
            """Per-group summed reward from the LAST compute() call (lower/upper)."""
            return self._group_reward_buf

        @property
        def reward_group_of(self) -> dict[str, str]:
            """term-name → 'lower'/'upper' routing map (for introspection/tests)."""
            return dict(self._reward_group_of)

        def compute(self, dt: float) -> torch.Tensor:
            # Recompute the scalar AND the two group streams in one pass. We reproduce
            # the core loop (rather than call super().compute()) so per-term rew_scaled
            # is available for group routing; the scalar result is identical.
            self._reward_buf[:] = 0.0
            self._group_reward_buf["lower"][:] = 0.0
            self._group_reward_buf["upper"][:] = 0.0

            for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
                if term_name in self._term_instances:
                    rew_raw = self._term_instances[term_name](self.env, **term_cfg.params)
                else:
                    rew_raw = self._term_funcs[term_name](self.env, **term_cfg.params)

                if rew_raw.shape[0] != self.env.num_envs:
                    raise ValueError(
                        f"Reward term '{term_name}' returned wrong shape. "
                        f"Expected [{self.env.num_envs}], got {rew_raw.shape}"
                    )

                rew_scaled = rew_raw * term_cfg.weight * dt
                self._reward_buf += rew_scaled
                self._group_reward_buf[self._reward_group_of[term_name]] += rew_scaled

                self._episode_sums[term_name] += rew_scaled
                self._episode_sums_raw[term_name] += rew_raw

            if self.cfg.only_positive_rewards:
                # Clip the scalar AND each group independently so no agent sees a
                # negative-total reward when the cfg forbids it (the scalar clip is
                # NOT decomposable per group, so we clip per group directly).
                self._reward_buf[:] = torch.clip(self._reward_buf, min=0.0)
                self._group_reward_buf["lower"][:] = torch.clip(self._group_reward_buf["lower"], min=0.0)
                self._group_reward_buf["upper"][:] = torch.clip(self._group_reward_buf["upper"], min=0.0)

            return self._reward_buf

    return DualGroupRewardManager


__all__ = [
    "tcp_track_pos_exp",
    "tcp_track_orn_exp",
    "UPPER_BODY_TAG",
    "LOWER_BODY_TAG",
    "make_dual_group_reward_manager_class",
    "_resolve_reward_group",
]
