"""MaskedMimic observation builder aligned with mm_sonic_sampling."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from numbers import Number
from typing import TYPE_CHECKING, Any

import torch

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.base import ObservationTermBase
from holosoma.managers.observation.terms import wbt as holosoma_wbt_obs
from holosoma.managers.termination.terms.wbt import BadTrackingZOnly
from holosoma.utils.rotations import quat_apply, quat_error_magnitude, quat_inverse, quat_mul
from wbt_training.config_values.command import G1_29DOF_WBT_BODY_NAMES

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


DEFAULT_CONDITIONABLE_BODIES = (
    "pelvis",
    "torso",
    "right_ankle_roll_link",
    "left_ankle_roll_link",
    "left_hand",
    "right_hand",
)

END_EFFECTOR_MASS_BODY_NAMES = ("left_rubber_hand", "right_rubber_hand")
END_EFFECTOR_MASS_NOISE_KG = 0.01

DEFAULT_HOVER_MASK_MODES = {
    "full_6pt": {"keypoints": (".*",), "joints": (".*",)},
    "full_joints_pelvis": {"keypoints": ("pelvis",), "joints": (".*",)},
    "pelvis_only": {"keypoints": ("pelvis",), "joints": ()},
    "three_point": {"keypoints": ("torso", "left_hand", "right_hand"), "joints": ()},
    "five_point": {
        "keypoints": ("torso", "left_hand", "right_hand", "left_ankle_roll_link", "right_ankle_roll_link"),
        "joints": (),
    },
    "upper_body_keypoints_lower_body_joints": {
        "keypoints": ("torso", "left_hand", "right_hand"),
        "joints": ("waist_.*", ".*hip.*", ".*knee.*", ".*ankle.*"),
    },
    "hands_keypoints_lower_body_joints": {
        "keypoints": ("left_hand", "right_hand"),
        "joints": ("waist_.*", ".*hip.*", ".*knee.*", ".*ankle.*"),
    },
    "left_hand_keypoints_lower_body_joints": {
        "keypoints": ("left_hand",),
        "joints": ("waist_.*", ".*hip.*", ".*knee.*", ".*ankle.*"),
    },
    "right_hand_keypoints_lower_body_joints": {
        "keypoints": ("right_hand",),
        "joints": ("waist_.*", ".*hip.*", ".*knee.*", ".*ankle.*"),
    },
    "left_hand_only": {"keypoints": ("left_hand",), "joints": ()},
    "right_hand_only": {"keypoints": ("right_hand",), "joints": ()},
    "both_hands_only": {"keypoints": ("left_hand", "right_hand"), "joints": ()},
    "pelvis_both_hands": {"keypoints": ("pelvis", "left_hand", "right_hand"), "joints": ()},
    "both_hands_both_feet_only": {
        "keypoints": ("left_hand", "right_hand", "left_ankle_roll_link", "right_ankle_roll_link"),
        "joints": (),
    },
    "upper_body": {
        "keypoints": ("pelvis", "torso", "left_hand", "right_hand"),
        "joints": ("waist_.*", ".*shoulder.*", ".*elbow.*", ".*wrist.*"),
    },
    "lower_body": {
        "keypoints": ("pelvis", "torso", "left_ankle_roll_link", "right_ankle_roll_link"),
        "joints": ("waist_.*", ".*hip.*", ".*knee.*", ".*ankle.*"),
    },
    "hands_only": {
        "keypoints": ("pelvis", "torso", "left_hand", "right_hand"),
        "joints": ("waist_.*", ".*shoulder.*", ".*elbow.*", ".*wrist.*"),
    },
    "feet_only": {
        "keypoints": ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link"),
        "joints": ("waist_.*", ".*hip.*", ".*knee.*", ".*ankle.*"),
    },
    "pelvis_upper_body_joints": {
        "keypoints": ("pelvis",),
        "joints": ("waist_.*", ".*shoulder.*", ".*elbow.*", ".*wrist.*"),
    },
    "pelvis_torso_arm_joints": {"keypoints": ("pelvis", "torso"), "joints": (".*shoulder.*", ".*elbow.*", ".*wrist.*")},
}


@dataclass(frozen=True)
class Sim2RealNoiseConfig:
    """Mocap measurement noise + head→pelvis FK simulation. Overridable via WBT_* env vars."""

    enabled: bool = False
    mocap_prob: float = 0.5
    mocap_pos_noise_std: float = 0.005
    mocap_quat_noise_std: float = 0.01
    fk_prob: float = 0.05
    fk_lever_arm: float = 0.53
    fk_pos_bias_std: float = 0.015
    fk_quat_bias_std: float = 0.02
    fk_vel_jitter_std: float = 0.05
    fk_lever_jitter_std: float = 0.003


def _load_sim2real_noise_cfg(cfg: "Sim2RealNoiseConfig") -> "Sim2RealNoiseConfig":
    """Override Sim2RealNoiseConfig fields from environment variables."""
    env_enabled = os.environ.get("WBT_SIM2REAL_NOISE", "").strip()
    if not (cfg.enabled or env_enabled == "1"):
        return cfg
    return Sim2RealNoiseConfig(
        enabled=True,
        mocap_prob=float(os.environ.get("WBT_MOCAP_PROB", str(cfg.mocap_prob))),
        mocap_pos_noise_std=float(os.environ.get("WBT_MOCAP_POS_STD", str(cfg.mocap_pos_noise_std))),
        mocap_quat_noise_std=float(os.environ.get("WBT_MOCAP_QUAT_STD", str(cfg.mocap_quat_noise_std))),
        fk_prob=float(os.environ.get("WBT_FK_PROB", str(cfg.fk_prob))),
        fk_lever_arm=float(os.environ.get("WBT_FK_LEVER", str(cfg.fk_lever_arm))),
        fk_pos_bias_std=float(os.environ.get("WBT_FK_POS_STD", str(cfg.fk_pos_bias_std))),
        fk_quat_bias_std=float(os.environ.get("WBT_FK_QUAT_STD", str(cfg.fk_quat_bias_std))),
        fk_vel_jitter_std=float(os.environ.get("WBT_FK_VEL_STD", str(cfg.fk_vel_jitter_std))),
        fk_lever_jitter_std=float(os.environ.get("WBT_FK_LEVER_JITTER", str(cfg.fk_lever_jitter_std))),
    )


@dataclass(frozen=True)
class MaskedMimicObsConfig:
    num_future_steps: int
    num_historical_steps: int
    visible_target_pose_prob: float
    repeat_mask_probability: float
    force_max_conditioned_bodies_prob: float
    force_small_num_conditioned_bodies_prob: float
    reduced_obs_noise: float
    target_obs_noise: float
    future_pose_noise: float
    time_alpha: float
    time_beta: float
    realtime_future_steps: bool
    include_next_frame_target: bool
    include_keypoint_linear_velocity: bool
    joint_space_mode_prob: float = 0.1
    include_base_lin_vel: bool = True
    masking_style: str = "hover"
    hover_mask_modes: dict[str, dict[str, tuple[str, ...]]] | None = None
    hover_mask_mode_filter: tuple[str, ...] | None = None
    hover_mask_sparsity_randomization_enabled: bool = False
    sim2real_noise: Sim2RealNoiseConfig = Sim2RealNoiseConfig()


DEFAULT_MASKEDMIMIC_OBS_CONFIG = MaskedMimicObsConfig(
    num_future_steps=30,
    num_historical_steps=0,
    visible_target_pose_prob=0.9,
    repeat_mask_probability=0.8,
    force_max_conditioned_bodies_prob=0.1,
    force_small_num_conditioned_bodies_prob=0.1,
    reduced_obs_noise=0.01,
    target_obs_noise=0.01,
    future_pose_noise=0.0,
    time_alpha=2.0,
    time_beta=5.0,
    realtime_future_steps=False,
    include_next_frame_target=False,
    include_keypoint_linear_velocity=False,
    joint_space_mode_prob=0.1,
    include_base_lin_vel=True,
    masking_style="hover",
    hover_mask_modes=DEFAULT_HOVER_MASK_MODES,
    hover_mask_sparsity_randomization_enabled=False,
)


def _uniform_noise_like(x: torch.Tensor, magnitude: float) -> torch.Tensor:
    if magnitude <= 0.0:
        return x
    return x + (torch.rand_like(x) * 2.0 - 1.0) * magnitude


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_inverse(quat, w_last=True), vec, w_last=True)


def _canonicalize_quat_w_positive(quat: torch.Tensor) -> torch.Tensor:
    sign = torch.where(quat[..., 3:4] < 0.0, -1.0, 1.0)
    return quat * sign


def _motion_slice(motion: object, field: str, indices: torch.Tensor) -> torch.Tensor:
    data = getattr(motion, field)
    return data[indices.to(data.device)].to(indices.device)


def _get_matching_indices(patterns: tuple[str, ...], names: list[str]) -> list[int]:
    indices: list[int] = []
    for pattern in patterns:
        regex = re.compile(pattern)
        for idx, name in enumerate(names):
            if regex.search(name):
                indices.append(idx)
    return sorted(set(indices))


def _get_motion_command(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


class _MaskedMimicObservationProvider:
    """Shared per-env builder for masked-mimic observations."""

    def __init__(
        self,
        env: WholeBodyTrackingManager,
        *,
        obs_cfg: MaskedMimicObsConfig = DEFAULT_MASKEDMIMIC_OBS_CONFIG,
        conditionable_body_names: tuple[str, ...] = DEFAULT_CONDITIONABLE_BODIES,
        fixed_conditionable_body_names: tuple[str, ...] | None = (),
    ) -> None:
        self.env = env
        self.obs_cfg = obs_cfg
        self.device = env.device
        self.command = _get_motion_command(env)
        self.num_envs = env.num_envs
        self.action_dim = int(env.action_manager.action.shape[-1])
        self.masking_style = str(getattr(self.obs_cfg, "masking_style", "hover")).strip().lower()
        if self.masking_style not in {"legacy", "hover"}:
            raise ValueError(f"Unsupported masking_style={self.masking_style!r}. Expected 'legacy' or 'hover'.")

        command_body_name_to_idx = {name: i for i, name in enumerate(self.command.motion_cfg.body_names_to_track)}
        robot_body_names = list(getattr(env.simulator, "body_names", []))
        robot_body_name_to_idx = {name: i for i, name in enumerate(robot_body_names)}
        joint_names = list(getattr(env, "dof_names", []))
        if not joint_names:
            joint_names = list(getattr(env.simulator, "dof_names", []))
        self.joint_names = joint_names

        alias_from_conditionable_to_robot_candidates = {
            "torso": ["torso_link"],
            "head": ["head_link"],
            "left_hand": ["left_rubber_hand", "left_wrist_yaw_link"],
            "right_hand": ["right_rubber_hand", "right_wrist_yaw_link"],
            "left hand": ["left_rubber_hand", "left_wrist_yaw_link"],
            "right hand": ["right_rubber_hand", "right_wrist_yaw_link"],
            "left_wrist": ["left_wrist_yaw_link"],
            "right_wrist": ["right_wrist_yaw_link"],
        }
        alias_from_conditionable_to_robot = {}
        for alias, candidates in alias_from_conditionable_to_robot_candidates.items():
            for candidate in candidates:
                if candidate in robot_body_name_to_idx:
                    alias_from_conditionable_to_robot[alias] = candidate
                    break
            if alias not in alias_from_conditionable_to_robot and candidates:
                alias_from_conditionable_to_robot[alias] = candidates[0]
        alias_from_conditionable_to_command = {
            "torso": "torso_link",
            "head": "torso_link",
            "head_link": "torso_link",
            "left_hand": "left_wrist_yaw_link",
            "right_hand": "right_wrist_yaw_link",
            "left hand": "left_wrist_yaw_link",
            "right hand": "right_wrist_yaw_link",
            "left_wrist": "left_wrist_yaw_link",
            "right_wrist": "right_wrist_yaw_link",
            "left_rubber_hand": "left_wrist_yaw_link",
            "right_rubber_hand": "right_wrist_yaw_link",
        }

        self._end_effector_mass_body_ids = [
            robot_body_name_to_idx.get(body_name, -1) for body_name in END_EFFECTOR_MASS_BODY_NAMES
        ]

        resolved_command_body_names: list[str] = []
        resolved_robot_body_names: list[str] = []
        missing_from_command: list[str] = []
        missing_from_robot: list[str] = []
        missing_alias_sources_from_robot: list[str] = []

        for body_name in conditionable_body_names:
            robot_target_name = alias_from_conditionable_to_robot.get(body_name, body_name)
            source_name = robot_target_name
            if source_name not in command_body_name_to_idx:
                source_name = alias_from_conditionable_to_command.get(body_name, source_name)
            if source_name not in command_body_name_to_idx:
                source_name = alias_from_conditionable_to_command.get(robot_target_name, source_name)
            if source_name not in command_body_name_to_idx:
                missing_from_command.append(body_name)
                continue
            if robot_target_name not in robot_body_name_to_idx:
                missing_from_robot.append(body_name)
                continue
            if source_name not in robot_body_name_to_idx:
                missing_alias_sources_from_robot.append(source_name)
                continue
            resolved_robot_body_names.append(robot_target_name)
            resolved_command_body_names.append(source_name)

        if missing_from_command:
            raise KeyError(
                "Conditionable body names are missing from command body_names (and alias fallback). "
                f"Missing: {missing_from_command}. "
                f"Available command names: {self.command.motion_cfg.body_names_to_track}"
            )
        if missing_from_robot:
            raise KeyError(
                "Conditionable body names are missing from robot body_names. "
                f"Missing: {missing_from_robot}. "
                f"Available robot names: {robot_body_names}"
            )
        if missing_alias_sources_from_robot:
            raise KeyError(
                "Resolved source body names are missing from robot body_names. "
                f"Missing: {missing_alias_sources_from_robot}. "
                f"Available robot names: {robot_body_names}"
            )

        self.conditionable_body_names = list(conditionable_body_names)
        # Index into full simulator body list (used for motion.body_pos_w which has all sim bodies)
        self.conditionable_body_ids = torch.tensor(
            [robot_body_name_to_idx[name] for name in resolved_command_body_names], dtype=torch.long, device=self.device
        )
        # Tracked-body-relative indices for use with MotionCommand public properties
        # (robot_body_pos_w, robot_body_quat_w, etc.) instead of simulator internals.
        # Source bodies are always in the tracked list. Target bodies fall back to
        # the source body's tracked index when not tracked (e.g. left_rubber_hand).
        tracked_name_to_tracked_idx = {name: i for i, name in enumerate(self.command.motion_cfg.body_names_to_track)}
        self.conditionable_tracked_source_ids = torch.tensor(
            [tracked_name_to_tracked_idx[name] for name in resolved_command_body_names],
            dtype=torch.long,
            device=self.device,
        )
        self.conditionable_tracked_target_ids = torch.tensor(
            [
                tracked_name_to_tracked_idx.get(name, tracked_name_to_tracked_idx[src])
                for name, src in zip(resolved_robot_body_names, resolved_command_body_names)
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.num_conditionable_bodies = len(self.conditionable_body_names)
        if self.num_conditionable_bodies == 0:
            raise ValueError("conditionable_body_names must be non-empty.")

        conditionable_name_to_local = {name: i for i, name in enumerate(self.conditionable_body_names)}
        pelvis_local_idx = conditionable_name_to_local.get("pelvis")
        if pelvis_local_idx is None:
            anchor_body_name = str(self.command.motion_cfg.body_name_ref[0]).strip()
            pelvis_local_idx = conditionable_name_to_local.get(anchor_body_name)
        self.pelvis_conditionable_local_idx = pelvis_local_idx

        if fixed_conditionable_body_names:
            fixed_missing = [name for name in fixed_conditionable_body_names if name not in conditionable_name_to_local]
            if fixed_missing:
                raise ValueError(
                    "fixed_conditionable_body_names must be a subset of conditionable_body_names. "
                    f"Missing subset names: {fixed_missing}. "
                    f"Conditionable names: {self.conditionable_body_names}"
                )
            fixed_local_ids = [conditionable_name_to_local[name] for name in fixed_conditionable_body_names]
            self.fixed_conditionable_body_local_ids: torch.Tensor | None = torch.tensor(
                fixed_local_ids, dtype=torch.long, device=self.device
            )
        else:
            self.fixed_conditionable_body_local_ids = None

        motion_fps_raw = getattr(self.command.motion, "fps", None)
        if isinstance(motion_fps_raw, torch.Tensor):
            motion_fps = float(motion_fps_raw.detach().view(-1)[0].item()) if motion_fps_raw.numel() > 0 else 0.0
        elif isinstance(motion_fps_raw, Number):
            motion_fps = float(motion_fps_raw)
        else:
            try:
                motion_fps = float(motion_fps_raw[0])
            except Exception:
                motion_fps = 0.0
        if not (motion_fps > 0.0):
            motion_fps = 1.0 / float(env.dt) if getattr(env, "dt", 0.0) > 0.0 else 50.0
        self.motion_fps = motion_fps

        self._s2r_cfg = _load_sim2real_noise_cfg(self.obs_cfg.sim2real_noise)
        self._s2r_enabled = self._s2r_cfg.enabled
        if self._s2r_enabled:
            self._mocap_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._fk_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._fk_pos_bias = torch.zeros(self.num_envs, 3, device=self.device)
            self._fk_quat_bias = torch.zeros(self.num_envs, 3, device=self.device)

        self.prev_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.hist_reduced: torch.Tensor | None = None
        self.hist_clean_reduced: torch.Tensor | None = None
        self.last_reduced: torch.Tensor | None = None
        self.last_clean_reduced: torch.Tensor | None = None
        self.last_noisy_projected_gravity: torch.Tensor | None = None
        self.last_clean_projected_gravity: torch.Tensor | None = None

        self._total_future_steps = self.obs_cfg.num_future_steps
        self._rolling_future_steps = self._total_future_steps
        if self._rolling_future_steps < 0:
            raise ValueError(f"Invalid future-step config: num_future_steps={self._total_future_steps}")

        self._rolling_target_masks = torch.zeros(
            self.num_envs,
            self._rolling_future_steps,
            self.num_conditionable_bodies,
            2,
            dtype=torch.float32,
            device=self.device,
        )
        self._rolling_target_pose_masks = torch.zeros(
            self.num_envs, self._rolling_future_steps, dtype=torch.float32, device=self.device
        )
        self._rolling_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._visible_future_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._hover_body_masks = torch.zeros(
            self.num_envs, self.num_conditionable_bodies, dtype=torch.bool, device=self.device
        )
        self._hover_joint_masks = torch.zeros(
            self.num_envs, len(self.joint_names), dtype=torch.bool, device=self.device
        )
        self._hover_masks_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hover_reset_mask = True
        self._hover_mode_names: list[str] = []
        self._hover_mode_to_body_ids: dict[str, torch.Tensor] = {}
        self._hover_mode_to_joint_ids: dict[str, torch.Tensor] = {}
        self._hover_mode_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if self.masking_style == "hover":
            self._initialize_hover_masking()

        self._cached_outputs: dict[str, torch.Tensor] | None = None
        self._last_state_signature: dict[str, torch.Tensor] | None = None
        self._last_step_count: int = -1

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        self.prev_actions[env_ids] = 0.0
        if self.hist_reduced is not None:
            self.hist_reduced[env_ids] = 0.0
        if self.hist_clean_reduced is not None:
            self.hist_clean_reduced[env_ids] = 0.0
        if self.last_reduced is not None:
            self.last_reduced[env_ids] = 0.0
        if self.last_clean_reduced is not None:
            self.last_clean_reduced[env_ids] = 0.0
        if self.last_noisy_projected_gravity is not None:
            self.last_noisy_projected_gravity[env_ids] = 0.0
        if self.last_clean_projected_gravity is not None:
            self.last_clean_projected_gravity[env_ids] = 0.0

        if self._s2r_enabled:
            n = env_ids.numel()
            self._mocap_env_mask[env_ids] = torch.rand(n, device=self.device) < self._s2r_cfg.mocap_prob
            self._fk_env_mask[env_ids] = torch.rand(n, device=self.device) < self._s2r_cfg.fk_prob
            self._fk_pos_bias[env_ids] = torch.randn(n, 3, device=self.device) * self._s2r_cfg.fk_pos_bias_std
            self._fk_quat_bias[env_ids] = torch.randn(n, 3, device=self.device) * self._s2r_cfg.fk_quat_bias_std

        self._rolling_initialized[env_ids] = False
        self._visible_future_steps[env_ids] = 0
        if self._rolling_future_steps > 0:
            self._rolling_target_masks[env_ids] = 0.0
            self._rolling_target_pose_masks[env_ids] = 0.0
        self._hover_body_masks[env_ids] = False
        self._hover_joint_masks[env_ids] = False
        self._hover_masks_initialized[env_ids] = False

        self._cached_outputs = None

    def _initialize_hover_masking(self) -> None:
        configured_modes = self.obs_cfg.hover_mask_modes or DEFAULT_HOVER_MASK_MODES

        mode_filter = self.obs_cfg.hover_mask_mode_filter
        if mode_filter is None:
            env_filter = os.environ.get("HOVER_MASK_MODES", "")
            if env_filter:
                mode_filter = tuple(m.strip() for m in env_filter.split(",") if m.strip())
        if mode_filter:
            filtered = {k: v for k, v in configured_modes.items() if k in mode_filter}
            missing = set(mode_filter) - set(filtered)
            if missing:
                print(f"[HOVER-MODES] WARNING: requested modes not found: {missing}", file=sys.stderr)
            print(f"[HOVER-MODES] Restricted to {len(filtered)} modes: {list(filtered.keys())}", file=sys.stderr)
            configured_modes = filtered

        if not configured_modes:
            raise ValueError("hover masking requires non-empty hover_mask_modes.")
        if not self.joint_names:
            raise ValueError("hover masking requires joint names to be available.")

        self._hover_mode_names = list(configured_modes.keys())
        for mode_name, mode_cfg in configured_modes.items():
            body_patterns = tuple(mode_cfg.get("keypoints", ()))
            joint_patterns = tuple(mode_cfg.get("joints", ()))
            body_ids = torch.tensor(
                _get_matching_indices(body_patterns, self.conditionable_body_names),
                dtype=torch.long,
                device=self.device,
            )
            joint_ids = torch.tensor(
                _get_matching_indices(joint_patterns, self.joint_names), dtype=torch.long, device=self.device
            )
            if body_ids.numel() == 0 and joint_ids.numel() == 0:
                raise ValueError(
                    f"hover mask mode {mode_name!r} matched no keypoints and no joints. "
                    f"Body patterns={body_patterns}, joint patterns={joint_patterns}."
                )
            self._hover_mode_to_body_ids[mode_name] = body_ids
            self._hover_mode_to_joint_ids[mode_name] = joint_ids

        if len(configured_modes) == 1:
            mode_regions = list(configured_modes.values())[0]
            if len(mode_regions) == 1:
                self._hover_reset_mask = False

    def _current_state_signature(self) -> dict[str, torch.Tensor]:
        motion_command = _get_motion_command(self.env)
        return {
            "time_steps": motion_command.time_steps.detach().clone(),
            "motion_ids": motion_command.motion_ids.detach().clone(),
            "episode_length_buf": self.env.episode_length_buf.detach().clone(),
            "actions": self.env.action_manager.action.detach().clone(),
        }

    def _current_step_count(self) -> int:
        return int(self.env.episode_length_buf.sum().item())

    def _is_same_state(self, signature: dict[str, torch.Tensor]) -> bool:
        if self._last_state_signature is None:
            return False
        return all(torch.equal(self._last_state_signature[key], signature[key]) for key in signature)

    def _advance_state(self, signature: dict[str, torch.Tensor]) -> None:
        if self._last_state_signature is None:
            self.prev_actions = signature["actions"].clone()
            return

        if self.obs_cfg.num_historical_steps > 0 and self.last_reduced is not None:
            history_entry = torch.cat([self.last_reduced, self.prev_actions], dim=-1)
            assert self.hist_reduced is not None
            self.hist_reduced[:, 1:] = self.hist_reduced[:, :-1].clone()
            self.hist_reduced[:, 0] = history_entry

        if self.obs_cfg.num_historical_steps > 0 and self.last_clean_reduced is not None:
            critic_history_entry = torch.cat([self.last_clean_reduced, self.prev_actions], dim=-1)
            assert self.hist_clean_reduced is not None
            self.hist_clean_reduced[:, 1:] = self.hist_clean_reduced[:, :-1].clone()
            self.hist_clean_reduced[:, 0] = critic_history_entry

        self.prev_actions = signature["actions"].clone()

        prev_episode_lengths = self._last_state_signature["episode_length_buf"]
        done_mask = signature["episode_length_buf"] <= prev_episode_lengths
        if done_mask.any():
            self.prev_actions[done_mask] = 0.0
            if self.hist_reduced is not None:
                self.hist_reduced[done_mask] = 0.0
            if self.hist_clean_reduced is not None:
                self.hist_clean_reduced[done_mask] = 0.0
            self._rolling_initialized[done_mask] = False
            self._visible_future_steps[done_mask] = 0
            if self._rolling_future_steps > 0:
                self._rolling_target_masks[done_mask] = 0.0
                self._rolling_target_pose_masks[done_mask] = 0.0
            if self.masking_style == "hover" and self._hover_reset_mask:
                self._hover_masks_initialized[done_mask] = False
                self._hover_body_masks[done_mask] = False
                self._hover_joint_masks[done_mask] = False

    def _resample_hover_masks(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return

        mode_indices = torch.randint(0, len(self._hover_mode_names), (env_ids.numel(),), device=self.device)
        self._hover_mode_indices[env_ids] = mode_indices
        self._hover_body_masks[env_ids] = False
        self._hover_joint_masks[env_ids] = False

        for mode_idx, mode_name in enumerate(self._hover_mode_names):
            selected_envs = env_ids[(mode_indices == mode_idx).nonzero(as_tuple=True)[0]]
            if selected_envs.numel() == 0:
                continue
            body_ids = self._hover_mode_to_body_ids[mode_name]
            joint_ids = self._hover_mode_to_joint_ids[mode_name]
            if body_ids.numel() > 0:
                self._hover_body_masks[selected_envs.unsqueeze(1), body_ids.unsqueeze(0)] = True
            if joint_ids.numel() > 0:
                self._hover_joint_masks[selected_envs.unsqueeze(1), joint_ids.unsqueeze(0)] = True

        if self.obs_cfg.hover_mask_sparsity_randomization_enabled:
            self._hover_body_masks[env_ids] = torch.bernoulli(self._hover_body_masks[env_ids].float() * 0.5).bool()
            self._hover_joint_masks[env_ids] = torch.bernoulli(self._hover_joint_masks[env_ids].float() * 0.5).bool()

        self._hover_masks_initialized[env_ids] = True

    def _hover_target_masks_for_envs(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hover_body_masks = self._hover_body_masks[env_ids].to(torch.float32)
        target_pose_masks = hover_body_masks.any(dim=1).to(torch.float32)
        return hover_body_masks, target_pose_masks

    def _joint_pos_rel(self) -> torch.Tensor:
        return self.env.simulator.dof_pos - self.env.default_dof_pos

    def _joint_vel_rel(self) -> torch.Tensor:
        return self.env.simulator.dof_vel

    def _projected_gravity_obs(self) -> torch.Tensor:
        return holosoma_wbt_obs.projected_gravity(self.env)

    def _base_lin_vel_obs(self) -> torch.Tensor:
        return holosoma_wbt_obs.base_lin_vel(self.env)

    def _base_ang_vel_obs(self) -> torch.Tensor:
        return holosoma_wbt_obs.base_ang_vel(self.env)

    def _apply_fk_reduced_noise(
        self, projected_gravity: torch.Tensor, base_lin_vel: torch.Tensor | None, base_ang_vel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        mask = self._fk_env_mask
        if not mask.any():
            return projected_gravity, base_lin_vel, base_ang_vel
        cfg = self._s2r_cfg

        projected_gravity = projected_gravity.clone()
        projected_gravity[mask] = projected_gravity[mask] + self._fk_quat_bias[mask]

        if base_lin_vel is not None:
            vel_jitter = torch.randn_like(self._fk_pos_bias) * cfg.fk_vel_jitter_std
            lever_jitter = (
                torch.randn_like(self._fk_pos_bias) * cfg.fk_lever_jitter_std * cfg.fk_lever_arm * self.motion_fps
            )
            base_lin_vel = base_lin_vel.clone()
            base_lin_vel[mask] = (
                base_lin_vel[mask] + self._fk_pos_bias[mask] * self.motion_fps + vel_jitter[mask] + lever_jitter[mask]
            )

        base_ang_vel = base_ang_vel.clone()
        base_ang_vel[mask] = base_ang_vel[mask] + self._fk_quat_bias[mask] * 0.5

        return projected_gravity, base_lin_vel, base_ang_vel

    def _apply_sim2real_target_noise(self, target_pose: torch.Tensor) -> torch.Tensor:
        if not self._s2r_enabled:
            return target_pose
        cfg = self._s2r_cfg
        target_pose = target_pose.clone()

        mocap_mask = self._mocap_env_mask
        if mocap_mask.any():
            pos_noise = (
                torch.randn(self.num_envs, target_pose.shape[1], target_pose.shape[2], 3, device=self.device)
                * cfg.mocap_pos_noise_std
            )
            target_pose[:, :, :, :3] = target_pose[:, :, :, :3] + pos_noise * mocap_mask[:, None, None, None]

        fk_mask = self._fk_env_mask
        if fk_mask.any() and self.pelvis_conditionable_local_idx is not None:
            pidx = self.pelvis_conditionable_local_idx
            # Broadcast fk_pos_bias [num_envs, 3] → [num_envs, num_slots, 3] via unsqueeze
            target_pose[:, :, pidx, :3] = (
                target_pose[:, :, pidx, :3] + self._fk_pos_bias[:, None, :] * fk_mask[:, None, None]
            )
        return target_pose

    def _reduced_coords_obs(self, *, apply_noise: bool) -> tuple[torch.Tensor, torch.Tensor]:
        joint_pos_rel = self._joint_pos_rel()
        joint_vel_rel = self._joint_vel_rel()
        projected_gravity = self._projected_gravity_obs()
        base_ang_vel = self._base_ang_vel_obs()

        if apply_noise:
            joint_pos_rel = _uniform_noise_like(joint_pos_rel, 0.01)
            joint_vel_rel = _uniform_noise_like(joint_vel_rel, 0.5)
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

    def _sample_new_body_masks(self, num_samples: int) -> torch.Tensor:
        if self.masking_style == "hover":
            raise RuntimeError("_sample_new_body_masks should not be used in hover masking mode.")
        b = self.num_conditionable_bodies
        if num_samples <= 0:
            return torch.zeros(0, b, 2, dtype=torch.float32, device=self.device)

        if self.fixed_conditionable_body_local_ids is not None:
            fixed_masks = torch.zeros(num_samples, b, 2, dtype=torch.float32, device=self.device)
            fixed_masks[:, self.fixed_conditionable_body_local_ids, :] = 1.0
            return fixed_masks

        use_all_bodies = (
            torch.rand(num_samples, device=self.device) >= self.obs_cfg.force_small_num_conditioned_bodies_prob
        )
        max_num_small = min(3, b)
        max_num_bodies = torch.where(
            use_all_bodies,
            torch.full((num_samples,), b, dtype=torch.long, device=self.device),
            torch.full((num_samples,), max_num_small, dtype=torch.long, device=self.device),
        )
        num_active_bodies = (
            torch.floor(torch.rand(num_samples, device=self.device) * max_num_bodies.to(torch.float32)).to(torch.long)
            + 1
        )
        force_max_bodies = torch.rand(num_samples, device=self.device) <= self.obs_cfg.force_max_conditioned_bodies_prob
        num_active_bodies[force_max_bodies] = b

        rand_values = torch.rand(num_samples, b, device=self.device)
        ranks = torch.argsort(torch.argsort(rand_values, dim=1), dim=1)
        active_body_ids = ranks < num_active_bodies.unsqueeze(1)
        rotation_enabled = torch.randint(0, 2, (num_samples, b), device=self.device, dtype=torch.bool)

        new_mask = torch.zeros(num_samples, b, 2, dtype=torch.float32, device=self.device)
        new_mask[:, :, 0] = active_body_ids.to(torch.float32)
        new_mask[:, :, 1] = (rotation_enabled & active_body_ids).to(torch.float32)
        return new_mask

    def _ensure_pose_visible(self, masks: torch.Tensor) -> None:
        if masks.numel() == 0:
            return
        is_visible = masks.reshape(masks.shape[0], -1).any(dim=1)
        if is_visible.all():
            return
        fallback_local_idx = (
            int(self.fixed_conditionable_body_local_ids[0].item())
            if self.fixed_conditionable_body_local_ids is not None
            else 0
        )
        masks[~is_visible, fallback_local_idx, :] = 1.0

    def _sample_fresh_body_masks(
        self, num_samples: int, previous_masks: torch.Tensor | None, *, force_pose_visible: bool
    ) -> torch.Tensor:
        b = self.num_conditionable_bodies
        if num_samples <= 0:
            return torch.zeros(0, b, 2, dtype=torch.float32, device=self.device)

        sampled = self._sample_new_body_masks(num_samples)
        if previous_masks is not None and previous_masks.numel() > 0 and self.obs_cfg.repeat_mask_probability > 0.0:
            repeat_mask = torch.rand(num_samples, device=self.device) < self.obs_cfg.repeat_mask_probability
            previous_visible = previous_masks.reshape(num_samples, -1).any(dim=1)
            repeat_mask = repeat_mask & previous_visible
            if repeat_mask.any():
                sampled[repeat_mask] = previous_masks[repeat_mask]

        if not force_pose_visible:
            hide_prob = max(0.0, 1.0 - self.obs_cfg.visible_target_pose_prob)
            if hide_prob > 0.0:
                fully_hidden = torch.rand(num_samples, device=self.device) < hide_prob
                if fully_hidden.any():
                    sampled[fully_hidden] = 0.0
        else:
            self._ensure_pose_visible(sampled)
        return sampled

    def _visible_future_slot_mask(self) -> torch.Tensor:
        if self._total_future_steps <= 0:
            return torch.zeros(self.num_envs, 0, dtype=torch.float32, device=self.device)
        slot_ids = torch.arange(self._total_future_steps, device=self.device).unsqueeze(0)
        return (slot_ids < self._visible_future_steps.unsqueeze(1)).to(torch.float32)

    def _resample_visible_future_steps(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0 or self._total_future_steps <= 0:
            return
        self._visible_future_steps[env_ids] = torch.randint(
            1, self._total_future_steps + 1, (env_ids.numel(),), device=self.device
        )

    def _initialize_rolling_window(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if self._rolling_future_steps == 0:
            self._rolling_initialized[env_ids] = True
            return

        self._resample_visible_future_steps(env_ids)
        if self.masking_style != "hover":
            previous_masks = None
            for slot in range(self._rolling_future_steps):
                sampled_masks = self._sample_fresh_body_masks(
                    int(env_ids.numel()), previous_masks, force_pose_visible=True
                )
                self._rolling_target_masks[env_ids, slot] = sampled_masks
                self._rolling_target_pose_masks[env_ids, slot] = (
                    sampled_masks.reshape(sampled_masks.shape[0], -1).any(dim=1).to(torch.float32)
                )
                previous_masks = sampled_masks
        self._rolling_initialized[env_ids] = True

    def _future_schedule(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        motion = self.command.motion
        current_idx = self.command.time_steps.long()
        motion_ids = self.command.motion_ids.long()
        _dev = motion_ids.device
        motion_start_idx = motion.motion_start_idx[motion_ids.to(motion.motion_start_idx.device)].to(_dev).long()
        motion_end_idx = motion.motion_end_idx[motion_ids.to(motion.motion_end_idx.device)].to(_dev).long()
        motion_last_idx = torch.maximum(motion_start_idx, motion_end_idx - 1)

        if self._rolling_future_steps > 0:
            init_env_ids = torch.where(~self._rolling_initialized)[0]
            if init_env_ids.numel() > 0:
                self._initialize_rolling_window(init_env_ids)

        if self.masking_style == "hover":
            init_mask_env_ids = torch.where(~self._hover_masks_initialized)[0]
            if init_mask_env_ids.numel() > 0:
                self._resample_hover_masks(init_mask_env_ids)

        if self._rolling_future_steps == 0:
            empty_indices = torch.zeros(self.num_envs, 0, dtype=torch.long, device=self.device)
            empty_masks = torch.zeros(
                self.num_envs, 0, self.num_conditionable_bodies, 2, dtype=torch.float32, device=self.device
            )
            empty_pose_masks = torch.zeros(self.num_envs, 0, dtype=torch.float32, device=self.device)
            return empty_indices, empty_masks, empty_pose_masks

        clamped_current_idx = torch.minimum(torch.maximum(current_idx, motion_start_idx), motion_last_idx)
        slot_offsets = torch.arange(1, self._rolling_future_steps + 1, dtype=torch.long, device=self.device).unsqueeze(
            0
        )
        future_indices = torch.minimum(clamped_current_idx.unsqueeze(1) + slot_offsets, motion_last_idx.unsqueeze(1))
        visible_slot_mask = self._visible_future_slot_mask()
        if self.masking_style == "hover":
            expanded_hover_masks, expanded_hover_pose_masks = self._hover_target_masks_for_envs(
                torch.arange(self.num_envs, device=self.device)
            )
            target_masks = expanded_hover_masks.unsqueeze(1).expand(-1, self._rolling_future_steps, -1)
            target_masks = target_masks * visible_slot_mask.unsqueeze(-1)
            target_pose_masks = expanded_hover_pose_masks.unsqueeze(1).expand(-1, self._rolling_future_steps)
            target_pose_masks = target_pose_masks * visible_slot_mask
            return future_indices, target_masks, target_pose_masks

        target_masks = self._rolling_target_masks * visible_slot_mask.unsqueeze(-1).unsqueeze(-1)
        target_pose_masks = self._rolling_target_pose_masks * visible_slot_mask
        return future_indices, target_masks, target_pose_masks

    def _future_target_pose(self, future_indices: torch.Tensor) -> torch.Tensor:
        # Use MotionCommand public properties (simulator-agnostic) with tracked-body indices
        robot_body_pos_w = self.command.robot_body_pos_w[:, self.conditionable_tracked_target_ids]
        robot_body_lin_vel_w = self.command.robot_body_lin_vel_w[:, self.conditionable_tracked_target_ids]
        robot_body_quat_w = self.command.robot_body_quat_w[:, self.conditionable_tracked_target_ids]
        source_body_pos_w = self.command.robot_body_pos_w[:, self.conditionable_tracked_source_ids]
        source_body_lin_vel_w = self.command.robot_body_lin_vel_w[:, self.conditionable_tracked_source_ids]
        source_body_quat_w = self.command.robot_body_quat_w[:, self.conditionable_tracked_source_ids]
        motion = self.command.motion
        env_origins = self.env.simulator.scene.env_origins[:, None, :]

        if future_indices.shape[1] == 0:
            d = 10 if self.obs_cfg.include_keypoint_linear_velocity else 7
            return torch.zeros(
                self.num_envs, 0, len(self.conditionable_body_ids), d, dtype=torch.float32, device=self.device
            )

        # Vectorized: gather all slots at once [num_envs, num_slots, num_bodies, D]
        _bp = getattr(motion, "body_pos_w")
        _bv = getattr(motion, "body_lin_vel_w")
        _bq = getattr(motion, "body_quat_w")
        _fi = future_indices.to(_bp.device)
        all_tgt_source_pos_w = _bp[_fi][:, :, self.conditionable_body_ids].to(self.device) + env_origins.unsqueeze(1)
        all_tgt_source_lin_vel_w = _bv[_fi][:, :, self.conditionable_body_ids].to(self.device)
        all_tgt_source_quat_w = _bq[_fi][:, :, self.conditionable_body_ids].to(self.device)

        # Source-to-target local transforms (constant across slots, computed once)
        s2t_local_pos = _quat_apply_inverse(source_body_quat_w, robot_body_pos_w - source_body_pos_w)
        s2t_local_quat = quat_mul(quat_inverse(source_body_quat_w, w_last=True), robot_body_quat_w, w_last=True)
        s2t_local_lin_vel = _quat_apply_inverse(source_body_quat_w, robot_body_lin_vel_w - source_body_lin_vel_w)

        # Expand [num_envs, num_bodies, D] -> [num_envs, num_slots, num_bodies, D]
        # Must use .expand() not .unsqueeze() because quat_apply/quat_mul flatten internally
        _ns = future_indices.shape[1]
        s2t_pos_e = s2t_local_pos.unsqueeze(1).expand(-1, _ns, -1, -1)
        s2t_quat_e = s2t_local_quat.unsqueeze(1).expand(-1, _ns, -1, -1)
        s2t_vel_e = s2t_local_lin_vel.unsqueeze(1).expand(-1, _ns, -1, -1)
        robot_bpos_e = robot_body_pos_w.unsqueeze(1).expand(-1, _ns, -1, -1)
        robot_bquat_e = robot_body_quat_w.unsqueeze(1).expand(-1, _ns, -1, -1)
        robot_bvel_e = robot_body_lin_vel_w.unsqueeze(1).expand(-1, _ns, -1, -1)

        # Vectorized target computation across all slots at once
        tgt_pos_w = all_tgt_source_pos_w + quat_apply(all_tgt_source_quat_w, s2t_pos_e, w_last=True)
        tgt_lin_vel_w = all_tgt_source_lin_vel_w + quat_apply(all_tgt_source_quat_w, s2t_vel_e, w_last=True)
        tgt_quat_w = quat_mul(all_tgt_source_quat_w, s2t_quat_e, w_last=True)

        delta_pos = _quat_apply_inverse(robot_bquat_e, tgt_pos_w - robot_bpos_e)
        delta_quat = quat_mul(quat_inverse(robot_bquat_e, w_last=True), tgt_quat_w, w_last=True)
        delta_quat = _canonicalize_quat_w_positive(delta_quat)
        if self.obs_cfg.include_keypoint_linear_velocity:
            delta_lin_vel = _quat_apply_inverse(robot_bquat_e, tgt_lin_vel_w - robot_bvel_e)
            return torch.cat([delta_pos, delta_lin_vel, delta_quat], dim=-1)
        return torch.cat([delta_pos, delta_quat], dim=-1)

    def _mask_target_pose(self, target_pose: torch.Tensor, target_masks_for_obs: torch.Tensor) -> torch.Tensor:
        if target_masks_for_obs.ndim == 3:
            target_pose_masked = target_pose * target_masks_for_obs.unsqueeze(-1)
        elif self.obs_cfg.include_keypoint_linear_velocity:
            trans_mask = target_masks_for_obs[:, :, :, 0:1]
            rot_mask = target_masks_for_obs[:, :, :, 1:2]
            target_pose_masked = torch.cat([target_pose[..., :6] * trans_mask, target_pose[..., 6:] * rot_mask], dim=-1)
        else:
            trans_mask = target_masks_for_obs[:, :, :, 0:1]
            rot_mask = target_masks_for_obs[:, :, :, 1:2]
            target_pose_masked = torch.cat([target_pose[..., :3] * trans_mask, target_pose[..., 3:] * rot_mask], dim=-1)
        return target_pose_masked.reshape(self.num_envs, -1)

    def _future_joint_targets(self, future_indices: torch.Tensor) -> torch.Tensor:
        motion = self.command.motion
        if future_indices.shape[1] == 0:
            num_joints = getattr(motion, "joint_pos").shape[-1]
            return torch.zeros(self.num_envs, 0, num_joints, dtype=torch.float32, device=self.device)
        _jp = getattr(motion, "joint_pos")
        return _jp[future_indices.to(_jp.device)].to(self.device)

    def _apply_joint_mode_target_masks(self, target_masks: torch.Tensor, joint_mode: torch.Tensor) -> torch.Tensor:
        target_masks_for_obs = target_masks.clone()
        joint_mode_env_ids = torch.where(joint_mode)[0]
        if joint_mode_env_ids.numel() == 0:
            return target_masks_for_obs
        target_masks_for_obs[joint_mode_env_ids] = 0.0
        if self.pelvis_conditionable_local_idx is not None:
            target_masks_for_obs[joint_mode_env_ids, :, self.pelvis_conditionable_local_idx, 0] = 1.0
        return target_masks_for_obs

    def _end_effector_mass_obs(self, apply_noise: bool) -> torch.Tensor:
        masses = torch.zeros(
            self.num_envs, len(self._end_effector_mass_body_ids), dtype=torch.float32, device=self.device
        )
        robot = getattr(self.env.simulator, "_robot", None)
        if robot is not None and hasattr(robot, "root_physx_view"):
            current_masses = robot.root_physx_view.get_masses().to(self.device)
            for output_idx, body_id in enumerate(self._end_effector_mass_body_ids):
                if body_id >= 0:
                    masses[:, output_idx] = current_masses[:, body_id]
        if apply_noise:
            masses = _uniform_noise_like(masses, END_EFFECTOR_MASS_NOISE_KG)
        return masses.clamp_min_(0.0)

    def _build_outputs(self) -> dict[str, torch.Tensor]:
        reduced, noisy_projected_gravity = self._reduced_coords_obs(apply_noise=True)
        critic_reduced, clean_projected_gravity = self._reduced_coords_obs(apply_noise=False)
        noisy_end_effector_masses = self._end_effector_mass_obs(apply_noise=True)
        future_indices, target_masks, target_poses_masks = self._future_schedule()
        target_pose_clean = self._future_target_pose(future_indices)
        target_pose = _uniform_noise_like(target_pose_clean, self.obs_cfg.future_pose_noise)
        target_pose = self._apply_sim2real_target_noise(target_pose)
        joint_targets = self._future_joint_targets(future_indices)
        visible_slot_mask = self._visible_future_slot_mask().unsqueeze(-1)

        if self.masking_style == "hover":
            target_masks_for_obs = target_masks
            target_poses_masks = target_masks_for_obs.any(dim=-1)
            hover_joint_mask = self._hover_joint_masks.to(joint_targets.dtype).unsqueeze(1)
            joint_targets_masked = (joint_targets * hover_joint_mask * visible_slot_mask).reshape(self.num_envs, -1)
            joint_space_mode_mask = self._hover_joint_masks.any(dim=1).to(torch.float32).unsqueeze(-1)
        else:
            joint_mode = torch.rand(self.num_envs, device=self.device) < self.obs_cfg.joint_space_mode_prob
            keypoint_mode = ~joint_mode
            target_masks_for_obs = self._apply_joint_mode_target_masks(target_masks, joint_mode)
            target_poses_masks = target_masks_for_obs.reshape(self.num_envs, self.obs_cfg.num_future_steps, -1).any(
                dim=-1
            )
            joint_targets_masked = joint_targets * visible_slot_mask
            joint_targets_masked[keypoint_mode] = 0.0
            joint_targets_masked = joint_targets_masked.reshape(self.num_envs, -1)
            joint_space_mode_mask = joint_mode.to(torch.float32).unsqueeze(-1)

        target_pose_masked = self._mask_target_pose(target_pose, target_masks_for_obs)
        critic_target_pose_masked = self._mask_target_pose(target_pose_clean, target_masks_for_obs)
        target_masks_flat = target_masks_for_obs.reshape(self.num_envs, -1)

        if self.obs_cfg.num_historical_steps > 0:
            history_entry_dim = reduced.shape[-1] + self.action_dim
            critic_history_entry_dim = critic_reduced.shape[-1] + self.action_dim
            if self.hist_reduced is None:
                self.hist_reduced = torch.zeros(
                    self.num_envs, self.obs_cfg.num_historical_steps, history_entry_dim, device=self.device
                )
            if self.hist_clean_reduced is None:
                self.hist_clean_reduced = torch.zeros(
                    self.num_envs, self.obs_cfg.num_historical_steps, critic_history_entry_dim, device=self.device
                )
            historical_pose_obs = self.hist_reduced.reshape(self.num_envs, -1)
            critic_historical_pose_obs = self.hist_clean_reduced.reshape(self.num_envs, -1)
        else:
            historical_pose_obs = reduced.new_zeros(self.num_envs, 0)
            critic_historical_pose_obs = critic_reduced.new_zeros(self.num_envs, 0)

        self.last_reduced = reduced
        self.last_clean_reduced = critic_reduced
        self.last_noisy_projected_gravity = noisy_projected_gravity
        self.last_clean_projected_gravity = clean_projected_gravity

        return {
            "noisy_reduced_coords_obs": reduced,
            "masked_mimic_target_poses": target_pose_masked,
            "critic_masked_mimic_target_poses": critic_target_pose_masked,
            "masked_mimic_target_masks": target_masks_flat,
            "joint_space_target_poses": joint_targets_masked,
            "joint_space_mode_mask": joint_space_mode_mask,
            "historical_pose_obs": historical_pose_obs,
            "critic_historical_pose_obs": critic_historical_pose_obs,
            "previous_actions": self.prev_actions,
            "end_effector_masses": noisy_end_effector_masses,
            "critic_end_effector_masses": self._end_effector_mass_obs(apply_noise=False),
            "masked_mimic_target_poses_masks": target_poses_masks.to(dtype=reduced.dtype),
            "critic_projected_gravity": clean_projected_gravity,
            "noisy_projected_gravity": noisy_projected_gravity,
        }

    def get_field(self, field: str) -> torch.Tensor:
        # Fast path: use cheap scalar check to detect same-step cache hits
        # (multiple obs terms call get_field() within the same env step)
        step_count = self._current_step_count()
        if self._cached_outputs is not None and step_count == self._last_step_count:
            return self._cached_outputs[field]

        # State changed — need full signature for advance_state (uses cloned tensors)
        signature = self._current_state_signature()
        if self._last_state_signature is not None:
            self._advance_state(signature)

        self._cached_outputs = self._build_outputs()
        self._last_state_signature = signature
        self._last_step_count = step_count
        return self._cached_outputs[field]


def _get_provider(env: WholeBodyTrackingManager) -> _MaskedMimicObservationProvider:
    provider = getattr(env, "_wbt_maskedmimic_obs_provider", None)
    if provider is None:
        provider = _MaskedMimicObservationProvider(env)
        setattr(env, "_wbt_maskedmimic_obs_provider", provider)
    return provider


class MaskedMimicObservationField(ObservationTermBase):
    """Stateful observation term that serves cached masked-mimic fields."""

    def __init__(self, cfg, env: Any):
        super().__init__(cfg, env)
        self.provider: _MaskedMimicObservationProvider | None = None

    def _provider(self) -> _MaskedMimicObservationProvider:
        if self.provider is None:
            self.provider = _get_provider(self.env)
        return self.provider

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self.provider is not None:
            self.provider.reset(env_ids)

    def __call__(self, env: Any, *, field: str) -> torch.Tensor:
        return self._provider().get_field(field)


# ---------------------------------------------------------------------------
# MaskedMimic reward functions, termination class, and config presets
# (merged from maskedmimic.py)
# ---------------------------------------------------------------------------


def robot_body_lin_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Tracked-body linear velocities expressed in the robot anchor frame."""
    motion_command = _get_motion_command(env)
    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    anchor_quat_inv = quat_inverse(motion_command.robot_ref_quat_w, w_last=True)[:, None, :].expand(-1, num_bodies, -1)
    body_lin_vel_b = quat_apply(
        anchor_quat_inv.reshape(-1, 4), motion_command.robot_body_lin_vel_w.reshape(-1, 3), w_last=True
    ).view(env.num_envs, -1)
    return body_lin_vel_b


def robot_body_ang_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Tracked-body angular velocities expressed in the robot anchor frame."""
    motion_command = _get_motion_command(env)
    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    anchor_quat_inv = quat_inverse(motion_command.robot_ref_quat_w, w_last=True)[:, None, :].expand(-1, num_bodies, -1)
    body_ang_vel_b = quat_apply(
        anchor_quat_inv.reshape(-1, 4), motion_command.robot_body_ang_vel_w.reshape(-1, 3), w_last=True
    ).view(env.num_envs, -1)
    return body_ang_vel_b


def _resolve_body_indices(motion_command: MotionCommand, body_names: list[str] | None) -> list[int]:
    if body_names is None:
        return list(range(len(motion_command.motion_cfg.body_names_to_track)))
    tracked = list(motion_command.motion_cfg.body_names_to_track)
    return [tracked.index(name) for name in body_names if name in tracked]


def motion_global_body_position_error_exp(
    env: WholeBodyTrackingManager, sigma: float, body_names: list[str] | None = None
) -> torch.Tensor:
    motion_command = _get_motion_command(env)
    body_indices = _resolve_body_indices(motion_command, body_names)
    if len(body_indices) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    error = torch.sum(
        torch.square(motion_command.body_pos_w[:, body_indices] - motion_command.robot_body_pos_w[:, body_indices]),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_orientation_error_exp(
    env: WholeBodyTrackingManager, sigma: float, body_names: list[str] | None = None
) -> torch.Tensor:
    motion_command = _get_motion_command(env)
    body_indices = _resolve_body_indices(motion_command, body_names)
    if len(body_indices) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    error = (
        quat_error_magnitude(
            motion_command.body_quat_w[:, body_indices], motion_command.robot_body_quat_w[:, body_indices]
        )
        ** 2
    )
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_linear_velocity_error_exp(
    env: WholeBodyTrackingManager, sigma: float, body_names: list[str] | None = None
) -> torch.Tensor:
    motion_command = _get_motion_command(env)
    body_indices = _resolve_body_indices(motion_command, body_names)
    if len(body_indices) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    error = torch.sum(
        torch.square(
            motion_command.body_lin_vel_w[:, body_indices] - motion_command.robot_body_lin_vel_w[:, body_indices]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_angular_velocity_error_exp(
    env: WholeBodyTrackingManager, sigma: float, body_names: list[str] | None = None
) -> torch.Tensor:
    motion_command = _get_motion_command(env)
    body_indices = _resolve_body_indices(motion_command, body_names)
    if len(body_indices) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    error = torch.sum(
        torch.square(
            motion_command.body_ang_vel_w[:, body_indices] - motion_command.robot_body_ang_vel_w[:, body_indices]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / sigma**2)


class MaskedMimicBadTrackingZOnly(BadTrackingZOnly):
    """MaskedMimic bad-tracking logic."""

    def __call__(self, env: Any, **kwargs):
        motion_command = self.env.command_manager.get_state("motion_command")
        assert motion_command.motion_cfg.body_names_to_track == self.body_names_to_track, (
            "body_names_to_track in motion_command and termination.params are not the same"
            f"motion_command.motion_cfg.body_names_to_track: {motion_command.motion_cfg.body_names_to_track}"
            f"termination.params['body_names_to_track']: {self.body_names_to_track}"
        )

        bad_tracking = self.bad_ref_pos(motion_command) | self.bad_motion_body_pos(motion_command)

        if motion_command.motion.has_object:
            bad_tracking |= self.bad_object_pos(motion_command) | self.bad_object_ori(motion_command)

        return bad_tracking


_actor_obs_maskedmimic_terms = {
    "obs00_noisy_reduced_coords_obs": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:MaskedMimicObservationField",
        params={"field": "noisy_reduced_coords_obs"},
        scale=1.0,
        noise=0.0,
    ),
    "obs01_masked_mimic_target_poses": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:MaskedMimicObservationField",
        params={"field": "masked_mimic_target_poses"},
        scale=1.0,
        noise=0.0,
    ),
    "obs02_masked_mimic_target_masks": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:MaskedMimicObservationField",
        params={"field": "masked_mimic_target_masks"},
        scale=1.0,
        noise=0.0,
    ),
    "obs03_joint_space_target_poses": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:MaskedMimicObservationField",
        params={"field": "joint_space_target_poses"},
        scale=1.0,
        noise=0.0,
    ),
    "obs04_joint_space_mode_mask": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:MaskedMimicObservationField",
        params={"field": "joint_space_mode_mask"},
        scale=1.0,
        noise=0.0,
    ),
    "obs05_historical_pose_obs": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:MaskedMimicObservationField",
        params={"field": "historical_pose_obs"},
        scale=1.0,
        noise=0.0,
    ),
    "obs06_previous_actions": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:MaskedMimicObservationField",
        params={"field": "previous_actions"},
        scale=1.0,
        noise=0.0,
    ),
    "obs07_end_effector_masses": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:MaskedMimicObservationField",
        params={"field": "end_effector_masses"},
        scale=1.0,
        noise=0.0,
    ),
}

_critic_obs_maskedmimic_terms = {
    "obs00_command": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:motion_command", scale=1.0, noise=0.0),
    "obs01_motion_anchor_pos_b": ObsTermCfg(
        func="wbt_training.config_values.observation:motion_anchor_pos_b", scale=1.0, noise=0.0
    ),
    "obs02_motion_anchor_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b", scale=1.0, noise=0.0
    ),
    "obs03_body_pos": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:robot_body_pos_b", scale=1.0, noise=0.0),
    "obs04_body_ori": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:robot_body_ori_b", scale=1.0, noise=0.0),
    "obs05_body_lin_vel": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:robot_body_lin_vel_b", scale=1.0, noise=0.0
    ),
    "obs06_body_ang_vel": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:robot_body_ang_vel_b", scale=1.0, noise=0.0
    ),
    "obs07_base_ang_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_ang_vel", scale=1.0, noise=0.0),
    "obs08_joint_pos": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos", scale=1.0, noise=0.0),
    "obs09_joint_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel", scale=1.0, noise=0.0),
    "obs10_actions": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:actions", scale=1.0, noise=0.0),
    "obs11_critic_masked_mimic_target_poses": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:MaskedMimicObservationField",
        params={"field": "critic_masked_mimic_target_poses"},
        scale=1.0,
        noise=0.0,
    ),
    "obs12_masked_mimic_target_masks": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:MaskedMimicObservationField",
        params={"field": "masked_mimic_target_masks"},
        scale=1.0,
        noise=0.0,
    ),
    "obs13_joint_space_target_poses": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:MaskedMimicObservationField",
        params={"field": "joint_space_target_poses"},
        scale=1.0,
        noise=0.0,
    ),
    "obs14_joint_space_mode_mask": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:MaskedMimicObservationField",
        params={"field": "joint_space_mode_mask"},
        scale=1.0,
        noise=0.0,
    ),
    "obs15_critic_historical_pose_obs": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:MaskedMimicObservationField",
        params={"field": "critic_historical_pose_obs"},
        scale=1.0,
        noise=0.0,
    ),
    "obs16_critic_projected_gravity": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:MaskedMimicObservationField",
        params={"field": "critic_projected_gravity"},
        scale=1.0,
        noise=0.0,
    ),
    "obs17_critic_end_effector_masses": ObsTermCfg(
        func="wbt_training.config_values.maskedmimic_observations:MaskedMimicObservationField",
        params={"field": "critic_end_effector_masses"},
        scale=1.0,
        noise=0.0,
    ),
}

g1_29dof_wbt_observation_maskedmimic = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=1, terms=_actor_obs_maskedmimic_terms
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=_critic_obs_maskedmimic_terms
        ),
    }
)

g1_29dof_wbt_maskedmimic_reward = RewardManagerCfg(
    terms={
        "motion_global_anchor_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_global_anchor_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        "motion_global_body_pos": RewardTermCfg(
            func="wbt_training.config_values.maskedmimic_observations:motion_global_body_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_global_body_ori": RewardTermCfg(
            func="wbt_training.config_values.maskedmimic_observations:motion_global_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        "motion_global_body_lin_vel": RewardTermCfg(
            func="wbt_training.config_values.maskedmimic_observations:motion_global_body_linear_velocity_error_exp",
            params={"sigma": 1.0},
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="wbt_training.config_values.maskedmimic_observations:motion_global_body_angular_velocity_error_exp",
            params={"sigma": 3.14},
            weight=1.0,
        ),
        "motion_global_wrist_pos": RewardTermCfg(
            func="wbt_training.config_values.maskedmimic_observations:motion_global_body_position_error_exp",
            params={"sigma": 0.3, "body_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"]},
            weight=1.2,
        ),
        "motion_global_wrist_ori": RewardTermCfg(
            func="wbt_training.config_values.maskedmimic_observations:motion_global_body_orientation_error_exp",
            params={"sigma": 0.4, "body_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"]},
            weight=1.2,
        ),
        "motion_global_wrist_lin_vel": RewardTermCfg(
            func="wbt_training.config_values.maskedmimic_observations:motion_global_body_linear_velocity_error_exp",
            params={"sigma": 1.0, "body_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"]},
            weight=1.2,
        ),
        "motion_global_wrist_ang_vel": RewardTermCfg(
            func="wbt_training.config_values.maskedmimic_observations:motion_global_body_angular_velocity_error_exp",
            params={"sigma": 3.14, "body_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"]},
            weight=1.2,
        ),
        "action_rate_l2": RewardTermCfg(func="holosoma.managers.reward.terms.wbt:penalty_action_rate", weight=-0.1),
        "joint_limit": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos", params={"soft_dof_pos_limit": 0.9}, weight=-10.0
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)"
                    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
                ),
            },
            weight=-0.1,
        ),
    }
)

g1_29dof_wbt_maskedmimic_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded", is_timeout=True
        ),
        "bad_tracking": TerminationTermCfg(
            func="wbt_training.config_values.maskedmimic_observations:MaskedMimicBadTrackingZOnly",
            params={
                "bad_ref_pos_threshold": 0.25,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.25,
                "body_names_to_track": G1_29DOF_WBT_BODY_NAMES,
                "bad_motion_body_pos_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                "bad_object_pos_threshold": 0.25,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)
