"""FALCON-style lower-body velocity-command locomotion — LEAN version.

Additive building blocks for the lower agent of ``exp:g1-wbt-2pt-force-falcon``:
the lower body follows a randomly-sampled velocity command (walk / stand) instead
of tracking the sonic-manipulation clip's legs. This is the LEAN scope (confirmed):
reuse holosoma's self-contained velocity-tracking reward terms + a velocity-command
sampler that also writes ``env.stance_mask``; FALCON's ~27 fine-grained gait terms
(stance_*, feet_heading, hip_pos, contact_no_vel, …) are deferred to a later pass.

Pieces:
  * ``FalconVelocityCommand`` — a ``LocomotionCommand`` subclass registered under the
    name ``locomotion_command`` (so ``command_manager.commands`` resolves) that samples
    [lin_vel_x, lin_vel_y, ang_vel_yaw] per episode with a stand probability, and writes
    the per-env stance flag to ``env.stance_mask``.
  * ``velocity_command`` obs term — the 3-D command the lower actor sees.
  * ``upright_orientation`` / ``base_ang_vel_xy_penalty`` / ``alive`` — thin, WBT-safe
    reward terms depending only on ``robot_root_states`` + ``base_quat`` (no terrain
    manager, gait phase, or feet buffers), so they work in the WBT env unchanged.
  * The velocity-tracking + dof-limit + action-rate terms are REUSED directly from
    ``holosoma.managers.reward.terms.locomotion`` / ``…:wbt`` via config (they read the
    same ``command_manager.commands`` and root states), so no new code for those.
"""

from __future__ import annotations

from typing import Any

import torch

from holosoma.managers.command.terms.locomotion import LocomotionCommand
from holosoma.utils.rotations import quat_rotate_inverse


class FalconVelocityCommand(LocomotionCommand):
    """Velocity command (walk/stand) that also publishes ``env.stance_mask``.

    Subclasses the stock ``LocomotionCommand`` (3-D command, stand_prob sampling) and,
    after each resample, sets a boolean ``env.stance_mask`` (True = commanded to stand,
    i.e. |lin_vel_xy| and |ang_vel_yaw| ≈ 0) that lower-body reward terms can gate on.
    Registered under the term name ``locomotion_command`` so ``command_manager.commands``
    exposes its buffer to the reused velocity-tracking rewards.
    """

    def setup(self) -> None:
        super().setup()
        # Allocate the stance flag on the env if the env class hasn't already.
        if not hasattr(self.env, "stance_mask") or self.env.stance_mask is None:
            self.env.stance_mask = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        self._refresh_stance_mask(None)

    def reset(self, env_ids) -> None:
        super().reset(env_ids)
        self._refresh_stance_mask(env_ids)

    def step(self) -> None:
        super().step()
        # Command may have been resampled mid-episode by the base step(); refresh all.
        self._refresh_stance_mask(None)

    def _refresh_stance_mask(self, env_ids) -> None:
        cmds = self.commands
        if cmds is None:
            return
        # Stand = near-zero commanded velocity (matches the base sampler's stand zeroing).
        stance = torch.norm(cmds[:, :3], dim=1) < 1e-6
        if not hasattr(self.env, "stance_mask") or self.env.stance_mask is None:
            self.env.stance_mask = stance
        elif env_ids is None:
            self.env.stance_mask[:] = stance
        else:
            idx = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.env.device)
            self.env.stance_mask[idx.long()] = stance[idx.long()]


# ---------------------------------------------------------------------------
# Observation term: the 3-D velocity command the LOWER actor tracks.
# ---------------------------------------------------------------------------


def velocity_command(env: Any) -> torch.Tensor:
    """The [lin_vel_x, lin_vel_y, ang_vel_yaw] command, shape (num_envs, 3).

    Deploy note: at inference this is fed by the joystick / high-level planner
    (the lower agent's runtime input), analogous to how F^cmd is supplied. PURE READER."""
    return env.command_manager.commands


# ---------------------------------------------------------------------------
# WBT-safe lower-body reward terms (root-state only; no terrain/gait/feet deps).
# The velocity-TRACKING terms themselves are reused directly from
# holosoma.managers.reward.terms.locomotion (tracking_lin_vel / tracking_ang_vel /
# penalty_ang_vel_xy) + …:wbt (penalty_action_rate / limits_dof_pos), since those
# read only command_manager.commands + root states. These three are extra thin ones.
# ---------------------------------------------------------------------------


def _base_ang_vel_local(env: Any) -> torch.Tensor:
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(env.simulator.base_quat, ang_vel_world, w_last=True)


def _projected_gravity(env: Any) -> torch.Tensor:
    """Gravity in the base frame (0,0,-1 rotated into base). Level ⇒ (0,0,-1)."""
    grav_world = torch.zeros(env.num_envs, 3, device=env.device)
    grav_world[:, 2] = -1.0
    return quat_rotate_inverse(env.simulator.base_quat, grav_world, w_last=True)


def upright_orientation(env: Any) -> torch.Tensor:
    """Penalty (positive) on non-upright base: ‖projected_gravity_xy‖². 0 when level."""
    pg = _projected_gravity(env)
    return torch.sum(pg[:, :2].square(), dim=1)


def base_ang_vel_xy_penalty(env: Any) -> torch.Tensor:
    """Penalty (positive) on roll/pitch angular velocity: ‖ang_vel_xy‖²."""
    return torch.sum(_base_ang_vel_local(env)[:, :2].square(), dim=1)


def alive(env: Any) -> torch.Tensor:
    """Constant +1 survival bonus (encourages the lower body to keep the robot up)."""
    return torch.ones(env.num_envs, device=env.device)


__all__ = [
    "FalconVelocityCommand",
    "velocity_command",
    "upright_orientation",
    "base_ang_vel_xy_penalty",
    "alive",
]
