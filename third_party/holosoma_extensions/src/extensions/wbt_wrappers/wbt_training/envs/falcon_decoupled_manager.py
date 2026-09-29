"""FALCON-style dual-agent env for the decoupled UniFP force-tracking recipe.

`FalconDecoupledManager` is the `env_class` for `exp:g1-wbt-2pt-force-falcon`. It is
a thin, ADDITIVE subclass of the stock `WholeBodyTrackingManager` that adds exactly
the env-side hooks the dual-agent PPO + FALCON locomotion need, WITHOUT touching core:

1. **Per-group reward streams.** After `super().__init__()`, it swaps the base
   `RewardManager` for a `DualGroupRewardManager` (built from the SAME reward cfg),
   so `env.reward_manager.group_reward_buf["lower"/"upper"]` is available each step.
   Safe because the core `RewardManager` has no `setup()` and is only used via
   `.reset()`/`.compute()` (both after `__init__`); `prepare_manager_fields` only
   scans MuJoCo `@required_field` decorators, which the subclass shares identically.

2. **Action-slice index maps.** `lower_action_indices` / `upper_action_indices` split
   the 29-D action vector by JOINT NAME (robust to the IsaacLab↔MuJoCo permutation —
   we match names in `self.dof_names`, the actual action order, rather than hardcode
   scattered indices). Lower = 12 leg + 3 waist; upper = 14 arms. The dual-agent algo
   scatters its two actor outputs into the 29-D env action via these maps.

3. **stance_mask hook.** A per-env {0,1} stance flag the FALCON lower-body locomotion
   reward reads (populated by the velocity-command term in a later step; defaults to
   all-standing here so the env is valid before that lands).

Everything else — physics, motion command, HandForceInjector, observation — is
inherited unchanged. The existing `WholeBodyTrackingManager` and its presets are
untouched (this is a new class selected only by the new preset's `env_class`).
"""

from __future__ import annotations

import torch

from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager

# Canonical joint-name partition for the G1 29-DOF dual-agent split.
# Lower agent (15 DOF) = 12 leg + 3 waist; upper agent (14 DOF) = both arms.
# Matched BY NAME against self.dof_names so the split is correct regardless of the
# sim's DOF ordering (the action vector is in self.dof_names order).
_LEG_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
_WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
_ARM_JOINTS = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
# Waist → LOWER agent (FALCON convention, confirmed with the user).
_LOWER_JOINTS = _LEG_JOINTS + _WAIST_JOINTS
_UPPER_JOINTS = _ARM_JOINTS


class FalconDecoupledManager(WholeBodyTrackingManager):
    def __init__(self, tyro_config, *, device):
        super().__init__(tyro_config, device=device)
        self._install_dual_group_reward_manager(tyro_config)
        self._build_action_slice_maps()
        self._init_stance_mask()

    # ------------------------------------------------------------------ reward
    def _install_dual_group_reward_manager(self, tyro_config) -> None:
        """Replace the base RewardManager with the per-group variant (additive)."""
        from wbt_training.config_values.falcon_force import make_dual_group_reward_manager_class

        dual_cls = make_dual_group_reward_manager_class()
        # Rebuild from the SAME reward cfg the base manager used; term funcs/params/
        # weights are identical, so episode-sum logging and the scalar rew_buf path
        # are byte-identical — only the extra per-group buffers are new.
        self.reward_manager = dual_cls(tyro_config.reward, self, self.device)

    # ------------------------------------------------------------- action split
    def _build_action_slice_maps(self) -> None:
        """Index maps splitting the 29-D action (in self.dof_names order) into the
        lower (15) and upper (14) agent slices, matched by joint name."""
        name_to_idx = {n: i for i, n in enumerate(self.dof_names)}
        missing_lower = [n for n in _LOWER_JOINTS if n not in name_to_idx]
        missing_upper = [n for n in _UPPER_JOINTS if n not in name_to_idx]
        if missing_lower or missing_upper:
            raise ValueError(
                f"FalconDecoupledManager: dof_names missing expected joints "
                f"(lower missing={missing_lower}, upper missing={missing_upper}). "
                f"Got dof_names={list(self.dof_names)}"
            )
        lower = [name_to_idx[n] for n in _LOWER_JOINTS]
        upper = [name_to_idx[n] for n in _UPPER_JOINTS]
        # Sanity: the two slices must partition exactly the 29 action dims.
        assert sorted(lower + upper) == list(range(self.num_dofs)), (
            f"lower+upper action slices must partition all {self.num_dofs} DOFs; "
            f"got {len(lower)}+{len(upper)}={len(lower) + len(upper)}"
        )
        self.lower_action_indices = torch.tensor(lower, dtype=torch.long, device=self.device)
        self.upper_action_indices = torch.tensor(upper, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------- stance mask
    def _init_stance_mask(self) -> None:
        """Per-env stance flag (True = stand, False = walk) for the FALCON lower-body
        locomotion reward. BOOL so reward terms can use it directly in torch.where
        (e.g. locomotion.base_height's stance gate). Defaults to all-standing; the
        FalconVelocityCommand term overwrites it each reset/resample."""
        self.stance_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
