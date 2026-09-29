"""TML Whole-Body Tracking Policy.

Integrates the TML (TML-Humand-Deploy) policy inference pipeline into the
Holosoma framework. The TML policy uses an external reference motion (.npz)
file and builds observations with Isaac joint ordering, as opposed to the
Holosoma WBT policy which embeds reference motions inside the ONNX model.

Key differences from WholeBodyTrackingPolicy:
- ONNX model takes only 'obs' input (no 'time_step')
- Reference motion loaded from external .npz file
- Observations built in explicit TML order (not alphabetically sorted)
- Joint positions/velocities reordered between Isaac and Mujoco (real robot) conventions
- Per-joint action scaling from ONNX metadata (comma-separated strings)
- IMU-based heading correction instead of Pinocchio FK
"""

from collections import deque

import numpy as np
import onnxruntime
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy
from holosoma_inference.utils.math.quat import (
    matrix_from_quat,
    quat_apply,
    quat_inverse,
    quat_mul,
    quat_rotate_inverse,
    quat_to_rpy,
    rpy_to_quat,
    subtract_frame_transforms,
)
from wbt_wrappers_inference.config_values.robot_config import (
    LOWER_BODY_INDICES as _LOWER_BODY_INDICES,
    VR_3POINT_BODIES as _VR_3POINT_BODIES,
    VR_3POINT_OFFSETS as _VR_3POINT_OFFSETS_LIST,
)

# fmt: off
# Isaac <-> Mujoco (real robot) joint ordering permutations for G1 29-DOF.
# Isaac groups joints by type (pitch/roll/yaw across legs, then torso, then arms),
# while Mujoco groups by body part (left leg, right leg, waist, left arm, right arm).
ISAAC_TO_MUJOCO = np.array([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
    2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
])

MUJOCO_TO_ISAAC = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
    5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
])
# fmt: on


class TMLWholeBodyTrackingPolicy(WholeBodyTrackingPolicy):
    """Whole-body tracking policy using the TML inference pipeline.

    Extends WholeBodyTrackingPolicy to reuse stiff hold mode, motion clip
    controls (keyboard 's' / joystick 'start'), and clock synchronization,
    while overriding the observation building and inference to match the
    TML pipeline.
    """

    def __init__(self, config: InferenceConfig):
        # TML-specific mutable state (set before super().__init__ which calls setup_policy)
        self.tml_last_action = np.zeros(29)
        self.tml_motion_ticker = 0
        self.tml_heading_initialized = False
        self.tml_init_heading_inv_wxyz = None

        super().__init__(config)

    # ========================================================================
    # Policy Setup
    # ========================================================================

    def setup_policy(self, model_path):
        """Load TML ONNX model and reference motion.

        Unlike the Holosoma WBT policy, this does NOT use Pinocchio or a
        time_step input. Instead it loads default_joint_pos and action_scale
        from ONNX custom metadata (comma-separated strings) and the reference
        motion from an external .npz file.
        """
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        # Extract TML metadata (comma-separated strings, NOT JSON)
        meta = self.onnx_policy_session.get_modelmeta().custom_metadata_map
        tml_default_pos_isaac = np.array([float(x) for x in meta["default_joint_pos"].split(",")])
        tml_action_scale_raw = np.array([float(x) for x in meta["action_scale"].split(",")])
        if len(tml_action_scale_raw) != 29:
            tml_action_scale_raw = np.ones(29) * tml_action_scale_raw

        # Store in both orderings
        self.tml_default_pos_isaac = tml_default_pos_isaac
        self.tml_default_pos_mujoco = tml_default_pos_isaac[ISAAC_TO_MUJOCO]
        self.tml_action_scale_mujoco = tml_action_scale_raw[ISAAC_TO_MUJOCO]

        # Load reference motion
        ref_motion_path = self.config.task.ref_motion_path
        if not ref_motion_path:
            raise ValueError("task.ref_motion_path must be provided for TML WBT policy")
        self.ref_motion = np.load(ref_motion_path)
        self.motion_length = self.ref_motion["joint_pos"].shape[0]
        self.ref_q_pos = self.ref_motion["joint_pos"].copy()  # [T, 29] Isaac order
        self.ref_q_vel = self.ref_motion["joint_vel"].copy()  # [T, 29] Isaac order
        self.ref_anchor_poses = self.ref_motion["body_pos_w"][:, 0].copy()  # [T, 3]
        self.ref_anchor_orns_wxyz = self.ref_motion["body_quat_w"][:, 0].copy()  # [T, 4] wxyz

        logger.info(f"TML reference motion loaded: {self.motion_length} frames from {ref_motion_path}")

        # Compute initial reference heading (for relative anchor orientation)
        self.ref_init_root_pos = self.ref_anchor_poses[0].copy()
        self.ref_init_root_pos[2] = 0.0
        first_root_quat_wxyz = self.ref_anchor_orns_wxyz[0]
        _, _, ref_yaw = quat_to_rpy(first_root_quat_wxyz)
        ref_heading_quat_wxyz = rpy_to_quat((0, 0, ref_yaw)).reshape(1, 4)
        self.ref_init_heading_inv_wxyz = quat_inverse(ref_heading_quat_wxyz)

        # First frame joint positions for init target (Mujoco order)
        self.tml_init_q_mujoco = self.ref_q_pos[0, ISAAC_TO_MUJOCO]

        # Dummy values so WholeBodyTrackingPolicy._capture_policy_state doesn't crash
        self.motion_command_0 = np.zeros((1, 58))
        self.ref_quat_xyzw_0 = np.zeros((1, 4))
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()

        # No KP/KD in TML ONNX metadata (comes from robot config instead)
        self.onnx_kp = None
        self.onnx_kd = None

        # Policy callable (for BasePolicy compatibility; not used directly)
        def policy_act(input_feed):
            return self.onnx_policy_session.run(None, input_feed)[0]

        self.policy = policy_act

    # ========================================================================
    # Inference
    # ========================================================================

    def rl_inference(self, robot_state_data):
        """Build TML observations, run ONNX model, and convert actions.

        Observations are built in the exact TML order (not alphabetically
        sorted) to match the ONNX model's expected input layout:
            command(58) | anchor_ori(6) | gravity(3) | omega(3) | q(29) | dq(29) | last_action(29)
        """
        if not self.motion_clip_progressing:
            self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
            self.curr_motion_timestep = self.timestep_util.timestep
            self.tml_motion_ticker = self.curr_motion_timestep

        obs = self._build_tml_observation(robot_state_data)

        # Run ONNX inference
        ort_inputs = {"obs": obs.astype(np.float32)}
        ort_outs = self.onnx_policy_session.run(None, ort_inputs)
        action_isaac = ort_outs[0].flatten()  # (29,) Isaac order

        # Store raw action for next step's observation (Isaac order)
        self.tml_last_action = action_isaac.copy()

        # Convert to real robot order, apply per-joint scaling, add TML default pose
        q_target = action_isaac[ISAAC_TO_MUJOCO] * self.tml_action_scale_mujoco + self.tml_default_pos_mujoco

        # BasePolicy.policy_action() computes: q_target = scaled_policy_action + default_dof_angles
        # So we provide the offset such that adding default_dof_angles gives our q_target.
        self.scaled_policy_action = (q_target - self.default_dof_angles).reshape(1, -1)
        self.last_policy_action = action_isaac.reshape(1, -1)

        # Advance motion ticker
        if self.motion_clip_progressing:
            self._set_motion_timestep()
            self.tml_motion_ticker = self.curr_motion_timestep

        return self.scaled_policy_action

    def _build_tml_observation(self, robot_state_data):
        """Build the observation vector in TML order.

        The observation consists of (in this exact order):
        1. command (58D): ref_joint_pos + ref_joint_vel from .npz at current ticker
        2. motion_anchor_ori_b (6D): relative rotation matrix first 2 columns, flattened
        3. projected_gravity (3D): gravity vector in body frame
        4. omega (3D): angular velocity from IMU
        5. q (29D): joint positions in Isaac order, minus TML default
        6. dq (29D): joint velocities in Isaac order
        7. last_action (29D): previous raw ONNX output (Isaac order)
        """
        # Extract robot state
        base_quat_wxyz = robot_state_data[0, 3:7]
        dof_pos = robot_state_data[0, 7 : 7 + self.num_dofs]
        ang_vel = robot_state_data[0, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]
        dof_vel = robot_state_data[0, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs]

        # Capture IMU heading on first inference call (lazy initialization)
        if not self.tml_heading_initialized:
            _, _, yaw = quat_to_rpy(base_quat_wxyz)
            heading_quat = rpy_to_quat((0, 0, yaw)).reshape(1, 4)
            self.tml_init_heading_inv_wxyz = quat_inverse(heading_quat)
            self.tml_heading_initialized = True

        # Heading-corrected root orientation
        root_orn_wxyz = quat_mul(self.tml_init_heading_inv_wxyz, base_quat_wxyz.reshape(1, 4))

        # Reference motion at current ticker
        mid = min(self.tml_motion_ticker, self.motion_length - 1)
        ref_joint_pos = self.ref_q_pos[mid]  # (29,) Isaac order
        ref_joint_vel = self.ref_q_vel[mid]  # (29,) Isaac order
        ref_anchor_orn_wxyz = self.ref_anchor_orns_wxyz[mid]  # (4,) wxyz

        # Relative anchor orientation (remove reference motion's initial heading)
        rel_anchor_orn_wxyz = quat_mul(self.ref_init_heading_inv_wxyz, ref_anchor_orn_wxyz.reshape(1, 4))

        # motion_anchor_ori_b: relative rotation from robot body to motion anchor
        relative_quat_wxyz = subtract_frame_transforms(root_orn_wxyz, rel_anchor_orn_wxyz)
        rot_matrix = matrix_from_quat(relative_quat_wxyz)  # (1, 3, 3)
        motion_anchor_ori_b = rot_matrix[..., :2].reshape(-1)  # (6,)

        # Projected gravity in body frame
        projected_gravity = quat_rotate_inverse(root_orn_wxyz, np.array([[0, 0, -1]])).flatten()

        # Assemble in TML order
        return np.concatenate(
            [
                np.concatenate([ref_joint_pos, ref_joint_vel]),  # command (58)
                motion_anchor_ori_b,  # motion_anchor_ori_b (6)
                projected_gravity,  # projected_gravity (3)
                ang_vel,  # omega (3)
                dof_pos[MUJOCO_TO_ISAAC] - self.tml_default_pos_isaac,  # q (29)
                dof_vel[MUJOCO_TO_ISAAC],  # dq (29)
                self.tml_last_action,  # last_action (29)
            ]
        ).reshape(1, -1)

    # ========================================================================
    # Initialization Target
    # ========================================================================

    def get_init_target(self, robot_state_data):
        """Interpolate toward first frame of the reference motion."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            target = self.tml_init_q_mujoco.reshape(1, -1)
            q_target = dof_pos + (target - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    # ========================================================================
    # Observation Buffer (BasePolicy compatibility)
    # ========================================================================

    def get_current_obs_buffer_dict(self, robot_state_data):
        """Provide observation buffer for BasePolicy compatibility.

        Not used during normal TML inference (rl_inference builds obs directly),
        but needed if prepare_obs_for_rl is ever called outside rl_inference.
        """
        current_obs_buffer_dict = {}
        current_obs_buffer_dict["tml_command"] = np.zeros((1, 58))
        current_obs_buffer_dict["motion_anchor_ori_b"] = np.zeros((1, 6))
        current_obs_buffer_dict["projected_gravity"] = np.zeros((1, 3))
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles
        current_obs_buffer_dict["dof_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
        ]
        current_obs_buffer_dict["actions"] = self.last_policy_action
        return current_obs_buffer_dict

    # ========================================================================
    # Control Handlers
    # ========================================================================

    def _handle_start_policy(self):
        """Start policy without Pinocchio yaw capture.

        Skips WholeBodyTrackingPolicy._handle_start_policy (which uses
        Pinocchio FK) and instead resets the heading flag so it's captured
        lazily from IMU on the first inference call.
        """
        BasePolicy._handle_start_policy(self)
        self._stiff_hold_active = False
        self.tml_heading_initialized = False

    def _handle_stop_policy(self):
        """Stop policy and return to stiff hold."""
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.tml_motion_ticker = 0
        self.tml_last_action = np.zeros(29)
        self.tml_heading_initialized = False
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0
        self.logger.info(colored("TML policy stopped, entering stiff hold", "yellow"))

    def _handle_start_motion_clip(self):
        """Start motion clip and reset TML ticker."""
        super()._handle_start_motion_clip()
        self.tml_motion_ticker = 0

    # ========================================================================
    # Policy State (multi-slot support)
    # ========================================================================

    def _capture_policy_state(self):
        """Capture TML-specific state for multi-policy slot switching."""
        state = BasePolicy._capture_policy_state(self)
        state.update(
            {
                "tml_default_pos_isaac": self.tml_default_pos_isaac.copy(),
                "tml_default_pos_mujoco": self.tml_default_pos_mujoco.copy(),
                "tml_action_scale_mujoco": self.tml_action_scale_mujoco.copy(),
                "ref_motion": self.ref_motion,
                "motion_length": self.motion_length,
                "ref_q_pos": self.ref_q_pos,
                "ref_q_vel": self.ref_q_vel,
                "ref_anchor_poses": self.ref_anchor_poses,
                "ref_anchor_orns_wxyz": self.ref_anchor_orns_wxyz,
                "ref_init_root_pos": self.ref_init_root_pos.copy(),
                "ref_init_heading_inv_wxyz": self.ref_init_heading_inv_wxyz.copy(),
                "tml_init_q_mujoco": self.tml_init_q_mujoco.copy(),
                # Dummy values for parent compatibility
                "motion_command_0": np.zeros((1, 58)),
                "ref_quat_xyzw_0": np.zeros((1, 4)),
            }
        )
        return state

    def _restore_policy_state(self, state):
        """Restore TML-specific state after policy slot switch."""
        BasePolicy._restore_policy_state(self, state)
        self.tml_default_pos_isaac = state["tml_default_pos_isaac"].copy()
        self.tml_default_pos_mujoco = state["tml_default_pos_mujoco"].copy()
        self.tml_action_scale_mujoco = state["tml_action_scale_mujoco"].copy()
        self.ref_motion = state["ref_motion"]
        self.motion_length = state["motion_length"]
        self.ref_q_pos = state["ref_q_pos"]
        self.ref_q_vel = state["ref_q_vel"]
        self.ref_anchor_poses = state["ref_anchor_poses"]
        self.ref_anchor_orns_wxyz = state["ref_anchor_orns_wxyz"]
        self.ref_init_root_pos = state["ref_init_root_pos"].copy()
        self.ref_init_heading_inv_wxyz = state["ref_init_heading_inv_wxyz"].copy()
        self.tml_init_q_mujoco = state["tml_init_q_mujoco"].copy()
        # Reset mutable state
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.tml_motion_ticker = 0
        self.tml_last_action = np.zeros(29)
        self.tml_heading_initialized = False
        # Dummy values for parent
        self.motion_command_0 = np.zeros((1, 58))
        self.ref_quat_xyzw_0 = np.zeros((1, 4))
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()

    def _on_policy_switched(self, model_path):
        """Reset TML state after policy slot switch."""
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.tml_motion_ticker = 0
        self._stiff_hold_active = True
        self.tml_last_action = np.zeros(29)
        self.tml_heading_initialized = False
        self.robot_yaw_offset = 0.0


# DEPRECATED: TMLCHIPPolicy is a temporary shim so run_policy.py can load
# legacy TML CHIP checkpoints. Remove once CHIP models are re-exported in
# holosoma format. Mirrors TML-Humanoid-Deploy RLCHIPPolicy (rl_policy.py).
_TML_VR_3POINT_OFFSETS = np.asarray(_VR_3POINT_OFFSETS_LIST, dtype=np.float64)
_TML_LOWER_ISAAC = np.asarray(_LOWER_BODY_INDICES)


class TMLCHIPPolicy(TMLWholeBodyTrackingPolicy):
    """DEPRECATED: see module-level note. TML-trained CHIP (3-point + lower-body command + compliance)."""

    _CHIP_HIST_LENGTH = 10
    _CHIP_HIST_TERM_DIMS = (("projected_gravity", 3), ("omega", 3), ("q", 29), ("dq", 29), ("last_action", 29))

    def __init__(self, config: InferenceConfig):
        self._chip_hist: dict[str, deque] = {}
        super().__init__(config)

    def _reset_chip_history(self):
        for name, dim in self._CHIP_HIST_TERM_DIMS:
            self._chip_hist[name] = deque(
                [np.zeros(dim, dtype=np.float32) for _ in range(self._CHIP_HIST_LENGTH)], maxlen=self._CHIP_HIST_LENGTH
            )

    def setup_policy(self, model_path):
        super().setup_policy(model_path)

        bn = self.ref_motion["body_names"].tolist() if "body_names" in self.ref_motion.files else None
        if bn is None:
            raise ValueError("TML CHIP ref motion must have 'body_names' for VR 3-point lookup")
        vr_idx = [bn.index(n) for n in _VR_3POINT_BODIES]
        ref_vr_pos_w = self.ref_motion["body_pos_w"][:, vr_idx].astype(np.float64)
        ref_vr_quat_wxyz = self.ref_motion["body_quat_w"][:, vr_idx].astype(np.float64)

        T = ref_vr_pos_w.shape[0]
        flat_q = ref_vr_quat_wxyz.reshape(-1, 4)
        flat_off = np.broadcast_to(_TML_VR_3POINT_OFFSETS[None, :, :], (T, 3, 3)).reshape(-1, 3)
        offsets_world = quat_apply(flat_q, flat_off).reshape(T, 3, 3)
        self._chip_ref_vr_pos_w = ref_vr_pos_w + offsets_world
        self._chip_ref_vr_quat_wxyz = ref_vr_quat_wxyz

        # Pre-bake per-frame quantities that depend only on the fixed ref motion.
        self._chip_ref_anchor_orn_inv_wxyz = quat_inverse(self.ref_anchor_orns_wxyz)
        # quat_mul has a batched bug (see _build_chip_observation); compute per-frame here.
        self._chip_rel_anchor_orn_wxyz = np.stack(
            [
                quat_mul(self.ref_init_heading_inv_wxyz, self.ref_anchor_orns_wxyz[t : t + 1]).reshape(4)
                for t in range(T)
            ]
        )
        self._chip_lower_cmd = np.concatenate(
            [self.ref_q_pos[:, _TML_LOWER_ISAAC], self.ref_q_vel[:, _TML_LOWER_ISAAC]], axis=1
        ).astype(np.float32)
        self._chip_compliance_vec = np.asarray(self.config.task.compliance, dtype=np.float32)

        self._reset_chip_history()

    def rl_inference(self, robot_state_data):
        if not self.motion_clip_progressing:
            self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
            self.curr_motion_timestep = self.timestep_util.timestep
            self.tml_motion_ticker = self.curr_motion_timestep

        obs = self._build_chip_observation(robot_state_data)

        ort_outs = self.onnx_policy_session.run(None, {"obs": obs.astype(np.float32, copy=False)})
        action_isaac = ort_outs[0].flatten()

        self.tml_last_action = action_isaac.copy()

        q_target = action_isaac[ISAAC_TO_MUJOCO] * self.tml_action_scale_mujoco + self.tml_default_pos_mujoco
        self.scaled_policy_action = (q_target - self.default_dof_angles).reshape(1, -1)
        self.last_policy_action = action_isaac.reshape(1, -1)

        if self.motion_clip_progressing:
            self._set_motion_timestep()
            self.tml_motion_ticker = self.curr_motion_timestep

        return self.scaled_policy_action

    def _build_chip_observation(self, robot_state_data):
        base_quat_wxyz = robot_state_data[0, 3:7]
        dof_pos = robot_state_data[0, 7 : 7 + self.num_dofs]
        ang_vel = robot_state_data[0, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]
        dof_vel = robot_state_data[0, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs]

        mid = min(self.tml_motion_ticker, self.motion_length - 1)

        ref_anchor_pos_w = self.ref_anchor_poses[mid]
        ref_anchor_orn_inv = self._chip_ref_anchor_orn_inv_wxyz[mid : mid + 1]

        delta_w = self._chip_ref_vr_pos_w[mid] - ref_anchor_pos_w
        anchor_inv_bcast = np.broadcast_to(ref_anchor_orn_inv, (3, 4))
        vr_pos_local = quat_apply(anchor_inv_bcast, delta_w).astype(np.float32)
        # quat_mul has a batched-reshape bug for N>1 inputs (holosoma_inference
        # utils/math/quat.py:90 stacks then reshapes without transposing), so
        # call per-body here to produce the correct product.
        vr_orn_local_wxyz = np.stack(
            [quat_mul(ref_anchor_orn_inv, self._chip_ref_vr_quat_wxyz[mid, b : b + 1]).reshape(4) for b in range(3)]
        ).astype(np.float32)

        if not self.tml_heading_initialized:
            _, _, yaw = quat_to_rpy(base_quat_wxyz)
            heading_quat = rpy_to_quat((0, 0, yaw)).reshape(1, 4)
            self.tml_init_heading_inv_wxyz = quat_inverse(heading_quat)
            self.tml_heading_initialized = True
        root_orn_wxyz = quat_mul(self.tml_init_heading_inv_wxyz, base_quat_wxyz.reshape(1, 4))
        rel_quat_wxyz = subtract_frame_transforms(root_orn_wxyz, self._chip_rel_anchor_orn_wxyz[mid : mid + 1])
        motion_anchor_ori_b = matrix_from_quat(rel_quat_wxyz)[..., :2].reshape(-1).astype(np.float32)

        projected_gravity = quat_rotate_inverse(root_orn_wxyz, np.array([[0, 0, -1]])).flatten().astype(np.float32)
        q_isaac = (dof_pos[MUJOCO_TO_ISAAC] - self.tml_default_pos_isaac).astype(np.float32)
        dq_isaac = dof_vel[MUJOCO_TO_ISAAC].astype(np.float32)

        self._chip_hist["projected_gravity"].append(projected_gravity)
        self._chip_hist["omega"].append(np.asarray(ang_vel, dtype=np.float32))
        self._chip_hist["q"].append(q_isaac)
        self._chip_hist["dq"].append(dq_isaac)
        self._chip_hist["last_action"].append(self.tml_last_action)

        return np.concatenate(
            [
                self._chip_lower_cmd[mid],
                vr_pos_local.reshape(-1),
                vr_orn_local_wxyz.reshape(-1),
                motion_anchor_ori_b,
                np.concatenate(self._chip_hist["projected_gravity"]),
                np.concatenate(self._chip_hist["omega"]),
                np.concatenate(self._chip_hist["q"]),
                np.concatenate(self._chip_hist["dq"]),
                np.concatenate(self._chip_hist["last_action"]),
                self._chip_compliance_vec,
            ]
        ).reshape(1, -1)

    def _handle_start_policy(self):
        super()._handle_start_policy()
        self._reset_chip_history()

    def _handle_stop_policy(self):
        super()._handle_stop_policy()
        self._reset_chip_history()
