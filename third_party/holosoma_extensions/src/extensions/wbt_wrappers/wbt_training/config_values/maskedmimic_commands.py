"""
MaskedMimic-specific command terms.
"""

from __future__ import annotations

import math
from dataclasses import field

import torch
from pydantic.dataclasses import dataclass

from holosoma.config_types.command import MotionConfig
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import get_euler_xyz, quat_apply, quat_from_euler_xyz, quat_inverse, quat_mul, yaw_quat


def _sample_uniform(
    low: float | torch.Tensor, high: float | torch.Tensor, shape: tuple[int, ...], device: str | torch.device
) -> torch.Tensor:
    low_t = torch.as_tensor(low, dtype=torch.float32, device=device)
    high_t = torch.as_tensor(high, dtype=torch.float32, device=device)
    return low_t + (high_t - low_t) * torch.rand(shape, device=device, dtype=torch.float32)


def _time_steps_to_bin_indices(time_steps: torch.Tensor, bin_count: int, time_step_total: int) -> torch.Tensor:
    return torch.clamp((time_steps * bin_count) // max(time_step_total, 1), 0, bin_count - 1)


def _smooth_bin_values(values: torch.Tensor, kernel: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return values
    values = torch.nn.functional.pad(values.unsqueeze(0).unsqueeze(0), (0, kernel_size - 1), mode="replicate")
    return torch.nn.functional.conv1d(values, kernel.view(1, 1, -1)).view(-1)


def _compute_sampling_probabilities(
    *,
    bin_failed_count: torch.Tensor,
    bin_visit_count: torch.Tensor,
    kernel: torch.Tensor,
    adaptive_kernel_size: int,
    adaptive_failure_rate_cap_beta: float,
    adaptive_blending_alpha: float,
) -> torch.Tensor:
    num_bins = int(bin_failed_count.numel())
    uniform = torch.full_like(bin_failed_count, 1.0 / float(num_bins))

    if float(bin_visit_count.sum().item()) <= 0.0:
        return uniform

    failure_rates = torch.zeros_like(bin_failed_count)
    visited_mask = bin_visit_count > 0
    failure_rates[visited_mask] = bin_failed_count[visited_mask] / bin_visit_count[visited_mask]

    mean_failure_rate = float(failure_rates.mean().item())
    if mean_failure_rate <= 0.0:
        return uniform

    capped_failure_rates = torch.clamp(failure_rates, max=float(adaptive_failure_rate_cap_beta) * mean_failure_rate)
    capped_failure_rates = _smooth_bin_values(capped_failure_rates, kernel, adaptive_kernel_size).clamp_min_(0.0)
    capped_sum = float(capped_failure_rates.sum().item())
    if capped_sum <= 0.0:
        return uniform

    preliminary_probabilities = capped_failure_rates / capped_sum
    return adaptive_blending_alpha * preliminary_probabilities + (1.0 - adaptive_blending_alpha) * uniform


class TMLAdaptiveTimestepsSampler:
    """Adaptive sampler matching ``mm_sonic_sampling`` Stage-2 logic."""

    def __init__(
        self,
        motion_time_step_total: int,
        *,
        device: str,
        env_fps: int,
        adaptive_kernel_size: int = 1,
        adaptive_lambda: float = 0.8,
        adaptive_blending_alpha: float = 0.1,
        adaptive_failure_rate_cap_beta: float = 200.0,
    ) -> None:
        self.device = device
        self.motion_time_step_total = motion_time_step_total
        self.env_fps = env_fps
        self.adaptive_kernel_size = adaptive_kernel_size
        self.adaptive_lambda = adaptive_lambda
        self.adaptive_blending_alpha = adaptive_blending_alpha
        self.adaptive_failure_rate_cap_beta = adaptive_failure_rate_cap_beta

        # Match the IsaacLab implementation's ~1 second binning.
        self.num_bins = int(self.motion_time_step_total // max(self.env_fps, 1)) + 1
        self.kernel = torch.tensor(
            [self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)], dtype=torch.float32, device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()
        self.metrics: dict[str, torch.Tensor] = {}
        self.init_buffers()

    def init_buffers(self) -> None:
        self.bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float32, device=self.device)
        self.bin_visit_count = torch.zeros(self.num_bins, dtype=torch.float32, device=self.device)

    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = _compute_sampling_probabilities(
            bin_failed_count=self.bin_failed_count,
            bin_visit_count=self.bin_visit_count,
            kernel=self.kernel,
            adaptive_kernel_size=self.adaptive_kernel_size,
            adaptive_failure_rate_cap_beta=self.adaptive_failure_rate_cap_beta,
            adaptive_blending_alpha=self.adaptive_blending_alpha,
        )
        sampled_bins = torch.multinomial(probabilities, num_samples, replacement=True)
        phases = (sampled_bins + _sample_uniform(0.0, 1.0, (num_samples,), self.device)) / float(self.num_bins)
        return sampled_bins, phases

    def update_counts(self, sampled_bins: torch.Tensor, episode_failed: torch.Tensor) -> None:
        if sampled_bins.numel() == 0:
            return
        self.bin_visit_count += torch.bincount(sampled_bins, minlength=self.num_bins).to(dtype=torch.float32)
        if torch.any(episode_failed):
            failed_bins = sampled_bins[episode_failed]
            self.bin_failed_count += torch.bincount(failed_bins, minlength=self.num_bins).to(dtype=torch.float32)

    def get_stats(self) -> None:
        probabilities = _compute_sampling_probabilities(
            bin_failed_count=self.bin_failed_count,
            bin_visit_count=self.bin_visit_count,
            kernel=self.kernel,
            adaptive_kernel_size=self.adaptive_kernel_size,
            adaptive_failure_rate_cap_beta=self.adaptive_failure_rate_cap_beta,
            adaptive_blending_alpha=self.adaptive_blending_alpha,
        )
        entropy = -(probabilities * (probabilities + 1e-12).log()).sum()
        self.metrics["sampling_entropy"] = entropy / math.log(self.num_bins)
        top_prob, top_bin = probabilities.max(dim=0)
        self.metrics["sampling_top1_prob"] = top_prob
        self.metrics["sampling_top1_bin"] = top_bin.float() / self.num_bins


@dataclass(frozen=True)
class TMLMaskedMimicMotionConfig(MotionConfig):
    """MotionConfig with Stage-2 multi-motion sampling/reset parameters."""

    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_blending_alpha: float = 0.1
    adaptive_failure_rate_cap_beta: float = 200.0
    start_at_timestep_zero_prob: float = 0.0
    freeze_at_timestep_zero_prob: float = 0.0
    enable_default_pose_prepend: bool = False
    enable_default_pose_append: bool = False
    # Transition mode for the default-pose blend that `setup()`/`_resample_command`
    # read (`"interpolation"` ⇒ smooth blend from the standing default pose to the
    # motion's first frame; any other value ⇒ rely on enable_default_pose_prepend).
    # MUST be a declared field: `command.py` passes it as a kwarg, and pydantic
    # frozen dataclasses SILENTLY DROP unknown kwargs — without this field the
    # value never lands and `setup()` raises AttributeError reading it (the base
    # g1-wbt-maskedmimic preset crashed here, surfaced by the launch smoke test).
    default_pose_transition_strategy: str = "interpolation"


class MaskedMimicMultiMotionCommand(MotionCommand):
    """Motion command with Stage-2 TML adaptive multi-motion sampling."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if isinstance(cfg.params["motion_config"], TMLMaskedMimicMotionConfig):
            self.motion_cfg = cfg.params["motion_config"]
        else:
            self.motion_cfg = TMLMaskedMimicMotionConfig(**cfg.params["motion_config"])
        self.init_pose_cfg = self.motion_cfg.noise_to_initial_pose

    def setup(self) -> None:
        super().setup()
        # Cache default pose roll/pitch from config (constant, avoids per-reset tensor creation)
        strategy = self.motion_cfg.default_pose_transition_strategy
        if strategy == "interpolation" or self.motion_cfg.enable_default_pose_prepend:
            init_state = self._env.robot_config.init_state
            init_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
            self._default_init_roll, self._default_init_pitch, _ = get_euler_xyz(init_quat, w_last=True)
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler = TMLAdaptiveTimestepsSampler(
                self.motion.time_step_total,
                device=self.device,
                env_fps=int(1 / self._env.dt),
                adaptive_kernel_size=self.motion_cfg.adaptive_kernel_size,
                adaptive_lambda=self.motion_cfg.adaptive_lambda,
                adaptive_blending_alpha=self.motion_cfg.adaptive_blending_alpha,
                adaptive_failure_rate_cap_beta=self.motion_cfg.adaptive_failure_rate_cap_beta,
            )
            self.init_buffers()

    def init_buffers(self):
        super().init_buffers()
        self.sampled_bin_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        update_counts_mask = None
        pending_lengths = getattr(self._env, "_pending_episode_lengths", None)
        if pending_lengths is not None:
            update_counts_mask = pending_lengths[env_ids] > 0
        self._resample_command(env_ids, update_counts_mask=update_counts_mask)

    def step(self) -> None:
        self.time_steps += 1

        per_motion_end = self.motion.motion_end_idx[self.motion_ids.to(self.motion.motion_end_idx.device)].to(
            self.device
        )
        ended_env_ids = torch.where(self.time_steps >= per_motion_end)[0]
        if ended_env_ids.numel() > 0:
            self._resample_command(ended_env_ids)
            sim = self._env.simulator
            sim.set_actor_root_state_tensor_robots(ended_env_ids, sim.robot_root_states)
            sim.set_dof_state_tensor_robots(ended_env_ids, sim.dof_state)  # type: ignore[attr-defined]
            sim.refresh_sim_tensors()
            # Mid-episode motion-boundary teleport is NOT a full env reset, so the
            # command manager does not dispatch HandForceInjector.reset() for these
            # envs. Without restarting the force pulse here, an env whose motion ends
            # mid-hold-plateau would carry a held ~25 N F^ext onto the freshly
            # teleported reference pose — a force shock the plain MotionCommand never
            # produces (it never resamples mid-episode). No-op on non-force presets
            # (the helper early-returns when the force buffers are not allocated).
            # Lazy import to avoid a command.py↔force_tracking.py↔maskedmimic_commands.py
            # import cycle (force_tracking imports command at module bottom).
            from wbt_training.config_values.force_tracking import restart_force_pulse_for_envs

            restart_force_pulse_for_envs(self._env, ended_env_ids)

        self._update_relative_targets()

    def _adaptive_sampling(self, env_ids: torch.Tensor, *, update_counts_mask: torch.Tensor | None = None) -> None:
        if env_ids.numel() == 0:
            return

        if update_counts_mask is None:
            update_counts_mask = torch.ones(env_ids.numel(), dtype=torch.bool, device=self.device)
        if torch.any(update_counts_mask):
            episode_failed = self._env.termination_manager.terminated[env_ids][update_counts_mask]
            completed_bins = self.sampled_bin_indices[env_ids][update_counts_mask]
            self.adaptive_timesteps_sampler.update_counts(completed_bins, episode_failed)

        sampled_bins, phases = self.adaptive_timesteps_sampler.sample(env_ids.numel())
        self.sampled_bin_indices[env_ids] = sampled_bins

        global_steps = (phases * float(self.motion.time_step_total - 1)).long()
        self.time_steps[env_ids] = global_steps

        motion_ids = torch.bucketize(self.time_steps[env_ids], self.motion.motion_end_idx.to(self.device), right=True)
        motion_ids.clamp_(max=self.motion.num_motions - 1)
        self.motion_ids[env_ids] = motion_ids

    def _resample_command(self, env_ids: torch.Tensor, *, update_counts_mask: torch.Tensor | None = None) -> None:
        if env_ids.numel() == 0:
            return

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self._adaptive_sampling(env_ids, update_counts_mask=update_counts_mask)
        else:
            sampled_bins = torch.randint(0, int(self.motion.time_step_total), (env_ids.numel(),), device=self.device)
            self.time_steps[env_ids] = sampled_bins
            self.sampled_bin_indices[env_ids] = _time_steps_to_bin_indices(
                sampled_bins,
                int(self.motion.time_step_total // max(int(1 / self._env.dt), 1)) + 1,
                self.motion.time_step_total,
            )
            motion_ids = torch.bucketize(
                self.time_steps[env_ids], self.motion.motion_end_idx.to(self.device), right=True
            )
            motion_ids.clamp_(max=self.motion.num_motions - 1)
            self.motion_ids[env_ids] = motion_ids

        # Force some envs to start at motion frame 0 (= prepend's default standing pose).
        # Prepend frames are part of the motion data, so motion_start_idx points to
        # prepend frame 0. Robot is initialized at default pose and tracks the smooth
        # interpolation to the motion's first frame.
        strategy = self.motion_cfg.default_pose_transition_strategy
        _use_start_at_zero = strategy == "interpolation" or self.motion_cfg.enable_default_pose_prepend
        _at_start_mask = None
        if _use_start_at_zero:
            motion_start_idx = self.motion.motion_start_idx[self.motion_ids[env_ids]].to(self.device)
            prob = self.motion_cfg.start_at_timestep_zero_prob
            if prob >= 1.0:
                force_start = torch.ones(env_ids.numel(), dtype=torch.bool, device=self.device)
            elif prob > 0.0:
                force_start = torch.rand(env_ids.numel(), device=self.device) < prob
            else:
                force_start = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
            if torch.any(force_start):
                self.time_steps[env_ids[force_start]] = motion_start_idx[force_start]
            _at_start_mask = self.time_steps[env_ids] == motion_start_idx

        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_ori = self.body_quat_w[env_ids, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids].clone()

        # For envs starting at frame 0, init robot in default standing pose.
        # With "interpolation" strategy this is redundant (reference already is default pose)
        # but ensures the robot state exactly matches the prepend's first frame.
        if _at_start_mask is not None and torch.any(_at_start_mask):
            n_start = int(_at_start_mask.sum().item())
            init_state = self._env.robot_config.init_state
            env_origins = self._env.simulator.scene.env_origins[env_ids]
            # Default root: keep motion x/y, use config z/roll/pitch
            root_pos[_at_start_mask, 2] = env_origins[_at_start_mask, 2] + init_state.pos[2]
            root_lin_vel[_at_start_mask] = 0.0
            root_ang_vel[_at_start_mask] = 0.0
            # Root rotation: keep yaw from motion, use config roll/pitch
            init_roll, init_pitch = self._default_init_roll, self._default_init_pitch
            _, _, motion_yaw = get_euler_xyz(root_ori[_at_start_mask], w_last=True)
            default_rot = quat_from_euler_xyz(init_roll.expand(n_start), init_pitch.expand(n_start), motion_yaw)
            root_ori[_at_start_mask] = default_rot
            # Default joint angles and zero velocities
            joint_pos[_at_start_mask] = self._env.default_dof_pos_base.expand(n_start, -1)
            joint_vel[_at_start_mask] = 0.0

        pose_ranges = torch.tensor(
            [self.motion_cfg.pose_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
            dtype=torch.float32,
            device=self.device,
        )
        pose_noise = _sample_uniform(pose_ranges[:, 0], pose_ranges[:, 1], (env_ids.numel(), 6), self.device)
        root_pos += pose_noise[:, :3]
        root_ori = quat_mul(
            quat_from_euler_xyz(pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5]), root_ori, w_last=True
        )

        velocity_ranges = torch.tensor(
            [self.motion_cfg.velocity_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
            dtype=torch.float32,
            device=self.device,
        )
        velocity_noise = _sample_uniform(
            velocity_ranges[:, 0], velocity_ranges[:, 1], (env_ids.numel(), 6), self.device
        )
        root_lin_vel += velocity_noise[:, :3]
        root_ang_vel += velocity_noise[:, 3:]

        joint_pos += _sample_uniform(
            self.motion_cfg.joint_position_range[0],
            self.motion_cfg.joint_position_range[1],
            tuple(joint_pos.shape),
            self.device,
        )

        soft_joint_pos_limits = self._env.simulator.dof_pos_limits  # type: ignore[attr-defined]
        joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, 0], soft_joint_pos_limits[:, 1])

        self._env.simulator.dof_pos[env_ids] = joint_pos
        self._env.simulator.dof_vel[env_ids] = joint_vel
        self._env.simulator.robot_root_states[env_ids, :3] = root_pos
        self._env.simulator.robot_root_states[env_ids, 3:7] = root_ori
        self._env.simulator.robot_root_states[env_ids, 7:10] = root_lin_vel
        self._env.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel

        if self.motion.has_object:
            object_states = torch.cat(
                [
                    self.object_pos_w[env_ids],
                    self.object_quat_w[env_ids],
                    self.object_lin_vel_w[env_ids],
                    torch.zeros_like(self.object_lin_vel_w[env_ids]),
                ],
                dim=-1,
            )
            self._env.simulator.set_actor_states([self.object_name], env_ids, object_states)

    def _update_relative_targets(self) -> None:
        use_root = (self._env.episode_length_buf == 0).unsqueeze(1).float()

        ref_pos_w = self.root_pos_w * use_root + self.ref_pos_w * (1 - use_root)
        ref_quat_w = self.root_quat_w * use_root + self.ref_quat_w * (1 - use_root)
        robot_ref_pos_w = self.robot_root_pos_w * use_root + self.robot_ref_pos_w * (1 - use_root)
        robot_ref_quat_w = self.robot_root_quat_w * use_root + self.robot_ref_quat_w * (1 - use_root)

        ref_pos_w_repeat = ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)
        ref_quat_w_repeat = ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)
        robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)
        robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)

        delta_quat_w = yaw_quat(
            quat_mul(robot_ref_quat_w_repeat, quat_inverse(ref_quat_w_repeat, w_last=True), w_last=True), w_last=True
        )
        self.body_quat_relative_w = quat_mul(delta_quat_w, self.body_quat_w, w_last=True)

        delta_pos_w_height = ref_pos_w_repeat - robot_ref_pos_w_repeat
        delta_pos_w_height[..., :2] = 0.0

        self.body_pos_relative_w = (
            robot_ref_pos_w_repeat
            + delta_pos_w_height
            + quat_apply(delta_quat_w, self.body_pos_w - ref_pos_w_repeat, w_last=True)
        )


__all__ = ["MaskedMimicMultiMotionCommand", "TMLMaskedMimicMotionConfig"]
