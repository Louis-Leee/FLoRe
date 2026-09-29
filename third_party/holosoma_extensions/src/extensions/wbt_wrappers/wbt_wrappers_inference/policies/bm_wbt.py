"""BM Whole-Body Tracking Policy.

BM WBT uses ONNX models where motion
reference tensors are embedded in the model and selected with `time_step`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.policies.wbt import PinocchioRobot, WholeBodyTrackingPolicy
from holosoma_inference.utils.math.misc import get_index_of_a_in_b
from holosoma_inference.utils.math.quat import (
    matrix_from_quat,
    quat_inverse,
    quat_mul,
    quat_to_rpy,
    rpy_to_quat,
    subtract_frame_transforms,
)

# fmt: off
# Isaac <-> Mujoco (real robot) joint ordering permutations for G1 29-DOF.
ISAAC_TO_MUJOCO = np.array([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
    2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
])
MUJOCO_TO_ISAAC = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
    5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
])
# fmt: on


class BMWholeBodyTrackingPolicy(WholeBodyTrackingPolicy):
    """Whole-body tracking policy for ONNX-embedded motion models."""

    _TERM_DIMS = {
        "command": 58,
        "motion_ref_ori_b": 6,
        "base_ang_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    }

    def __init__(self, config: InferenceConfig):
        self.bm_last_action = np.zeros(29, dtype=np.float32)
        self.bm_motion_command_t = np.zeros((1, 58), dtype=np.float32)
        self.bm_motion_command_0 = np.zeros((1, 58), dtype=np.float32)
        self.bm_anchor_pos_w_t = np.zeros((1, 3), dtype=np.float32)
        self.bm_anchor_pos_w_0 = np.zeros((1, 3), dtype=np.float32)
        self.bm_anchor_orn_wxyz_t = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        self.bm_anchor_orn_wxyz_0 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        self.bm_heading_initialized = False
        self.bm_init_heading_inv_wxyz = None
        self.bm_ref_init_heading_inv_wxyz = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        self.bm_init_q_mujoco = np.zeros(29, dtype=np.float32)
        self.bm_obs_terms: list[str] = []
        self.bm_obs_dim: int | None = None
        self.bm_ref_body_index = 7
        self.bm_anchor_body_name: str | None = None
        self.policy2robot_index = ISAAC_TO_MUJOCO.copy()
        self.robot2policy_index = MUJOCO_TO_ISAAC.copy()
        self.bm_use_pinocchio_ref = False
        self._warned_ref_body_fallback = False
        super().__init__(config)

    @staticmethod
    def _parse_csv_float_array(raw_value: str) -> np.ndarray:
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, (int, float)):
            return np.asarray([parsed], dtype=np.float32)
        if isinstance(parsed, list):
            values = np.asarray(parsed, dtype=np.float32).reshape(-1)
            if values.size > 0:
                return values

        values = np.fromstring(raw_value, sep=",", dtype=np.float32)
        if values.size == 0:
            raise ValueError(f"Unable to parse metadata value: {raw_value!r}")
        return values

    @staticmethod
    def _parse_csv_tokens(raw_value: str) -> list[str]:
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, list):
            return [str(token).strip() for token in parsed if str(token).strip()]
        return [token.strip() for token in raw_value.split(",") if token.strip()]

    @staticmethod
    def _decode_metadata_string(raw_value: str | None) -> str:
        if raw_value is None:
            return ""
        value = str(raw_value).strip()
        if not value:
            return ""

        try:
            parsed = json.loads(value)
            if isinstance(parsed, str):
                value = parsed
        except json.JSONDecodeError:
            pass

        # Handle escaped XML payloads stored as plain strings in metadata.
        if "\\n" in value or '\\"' in value:
            try:
                value = bytes(value, "utf-8").decode("unicode_escape")
            except UnicodeDecodeError:
                pass
        return value.strip()

    @staticmethod
    def _load_raw_metadata(model_path: str) -> dict[str, str]:
        metadata: dict[str, str] = {}
        onnx_model = onnx.load(model_path)
        for prop in onnx_model.metadata_props:
            metadata[prop.key] = prop.value
        return metadata

    @staticmethod
    def _infer_obs_dim(session: onnxruntime.InferenceSession) -> int | None:
        obs_input = None
        for inp in session.get_inputs():
            if inp.name == "obs":
                obs_input = inp
                break
        if obs_input is None and session.get_inputs():
            obs_input = session.get_inputs()[0]
        if obs_input is None:
            return None

        shape = obs_input.shape
        if len(shape) < 2:
            return None
        dim = shape[-1]
        return int(dim) if isinstance(dim, int) else None

    def _term_dim(self, term: str) -> int:
        if term not in self._TERM_DIMS:
            raise ValueError(f"Unsupported BM observation term '{term}'. Supported terms: {sorted(self._TERM_DIMS)}")
        return self._TERM_DIMS[term]

    def _setup_joint_mappings(self, metadata_csv: dict[str, str]):
        policy_joint_names = self._parse_csv_tokens(metadata_csv.get("joint_names", ""))
        robot_joint_names = list(self.config.robot.dof_names)

        if len(policy_joint_names) == self.num_dofs and set(policy_joint_names) == set(robot_joint_names):
            self.policy2robot_index = np.array(
                get_index_of_a_in_b(robot_joint_names, policy_joint_names), dtype=np.int64
            )
            self.robot2policy_index = np.array(
                get_index_of_a_in_b(policy_joint_names, robot_joint_names), dtype=np.int64
            )
            logger.info("BM joint mapping resolved from ONNX joint_names metadata")
        else:
            self.policy2robot_index = ISAAC_TO_MUJOCO.copy()
            self.robot2policy_index = MUJOCO_TO_ISAAC.copy()
            logger.warning("BM joint_names metadata missing/incompatible, using ISAAC<->MUJOCO fallback mapping")

    def _setup_reference_body_index(self, metadata: dict[str, str]):
        anchor_body_name = self._decode_metadata_string(metadata.get("anchor_body_name", ""))
        body_names = self._parse_csv_tokens(metadata.get("body_names", ""))
        if not anchor_body_name or not body_names:
            return

        if anchor_body_name not in body_names:
            logger.warning(
                "BM anchor_body_name '{}' not found in body_names metadata. Falling back to index {}.",
                anchor_body_name,
                self.bm_ref_body_index,
            )
            return

        self.bm_anchor_body_name = anchor_body_name
        self.bm_ref_body_index = body_names.index(anchor_body_name)
        logger.info(
            "BM reference body resolved from metadata: '{}' (index {})", anchor_body_name, self.bm_ref_body_index
        )

    def _setup_pinocchio_reference(self, model_path: str, metadata: dict[str, str]):
        robot_urdf = self._decode_metadata_string(metadata.get("robot_urdf"))

        if not isinstance(robot_urdf, str) or not robot_urdf.strip():
            urdf_path = getattr(self.config.task, "robot_urdf_path", "")
            if urdf_path and Path(urdf_path).is_file():
                robot_urdf = Path(urdf_path).read_text()
                logger.info(f"BM loaded robot URDF from config: {urdf_path}")
            else:
                raise ValueError(
                    f"BM policy requires a robot URDF for Pinocchio FK but none was found. "
                    f"ONNX model has no 'robot_urdf' metadata and task.robot_urdf_path={urdf_path!r} is not a valid file."
                )

        self.pinocchio_robot = PinocchioRobot(self.config.robot, robot_urdf)
        if self.bm_anchor_body_name:
            anchor_frame_id = self.pinocchio_robot.robot_model.getFrameId(self.bm_anchor_body_name)
            if anchor_frame_id < self.pinocchio_robot.robot_model.nframes:
                self.pinocchio_robot.ref_body_frame_id = anchor_frame_id
            else:
                logger.warning(
                    "BM anchor body '{}' not found in Pinocchio model. Using robot config ref body.",
                    self.bm_anchor_body_name,
                )
        self.bm_use_pinocchio_ref = True

    @staticmethod
    def _quat_yaw(quat_wxyz: np.ndarray) -> float:
        quat_flat = quat_wxyz.reshape(-1, 4)[0]
        _, _, yaw = quat_to_rpy(quat_flat)
        return float(yaw)

    def _remove_yaw_offset(self, quat_wxyz: np.ndarray, yaw_offset: float) -> np.ndarray:
        if abs(yaw_offset) < 1e-6:
            return quat_wxyz
        yaw_quat = rpy_to_quat((0.0, 0.0, -yaw_offset)).reshape(1, 4)
        yaw_quat = np.broadcast_to(yaw_quat, quat_wxyz.shape)
        return quat_mul(yaw_quat, quat_wxyz)

    def _capture_robot_yaw_offset(self):
        robot_state_data = self.interface.get_low_state()
        if robot_state_data is None:
            self.robot_yaw_offset = 0.0
            return
        if self.bm_use_pinocchio_ref:
            robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)
        else:
            robot_ref_ori = robot_state_data[:, 3:7]
        self.robot_yaw_offset = self._quat_yaw(robot_ref_ori)

    def _capture_motion_yaw_offset(self):
        self.motion_yaw_offset = self._quat_yaw(self.bm_anchor_orn_wxyz_0)

    def setup_policy(self, model_path):
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        metadata = dict(self.onnx_policy_session.get_modelmeta().custom_metadata_map)
        if "default_joint_pos" not in metadata or "action_scale" not in metadata:
            raise ValueError(
                "BM ONNX metadata must contain 'default_joint_pos' and 'action_scale'. "
                f"Available keys: {sorted(metadata.keys())}"
            )

        bm_default_pos_isaac = self._parse_csv_float_array(metadata["default_joint_pos"])
        bm_action_scale_raw = self._parse_csv_float_array(metadata["action_scale"])
        if bm_default_pos_isaac.size != 29:
            raise ValueError(f"default_joint_pos must contain 29 values, got {bm_default_pos_isaac.size}")
        if bm_action_scale_raw.size == 1:
            bm_action_scale_raw = np.full(29, bm_action_scale_raw.item(), dtype=np.float32)
        elif bm_action_scale_raw.size != 29:
            raise ValueError(f"action_scale must contain 1 or 29 values, got {bm_action_scale_raw.size}")

        self.bm_default_pos_isaac = bm_default_pos_isaac
        self._setup_joint_mappings(metadata)
        self.bm_default_pos_mujoco = bm_default_pos_isaac[self.policy2robot_index]
        self.bm_action_scale_mujoco = bm_action_scale_raw[self.policy2robot_index]
        self.bm_ref_body_index = 7
        self._setup_reference_body_index(metadata)
        self._setup_pinocchio_reference(model_path, metadata)

        self.bm_obs_terms = list(self.config.observation.obs_dict.get("actor_obs", []))
        if not self.bm_obs_terms:
            raise ValueError("BM observation term list is empty.")

        obs_dim_from_terms = sum(self._term_dim(term) for term in self.bm_obs_terms)
        self.bm_obs_dim = self._infer_obs_dim(self.onnx_policy_session) or obs_dim_from_terms
        if self.bm_obs_dim != obs_dim_from_terms:
            raise ValueError(
                f"BM observation dimension mismatch. model expects {self.bm_obs_dim}, "
                f"config terms sum to {obs_dim_from_terms}, terms={self.bm_obs_terms}"
            )

        output_0 = self._run_policy(
            np.zeros((1, 1), dtype=np.float32), np.zeros((1, self.bm_obs_dim), dtype=np.float32)
        )
        self.bm_motion_command_t = np.concatenate([output_0["joint_pos"], output_0["joint_vel"]], axis=1)
        self.bm_motion_command_0 = self.bm_motion_command_t.copy()
        self.bm_anchor_pos_w_t = output_0["anchor_pos_w"].copy()
        self.bm_anchor_pos_w_0 = output_0["anchor_pos_w"].copy()
        self.bm_anchor_orn_wxyz_t = output_0["anchor_orn_wxyz"].copy()
        self.bm_anchor_orn_wxyz_0 = output_0["anchor_orn_wxyz"].copy()
        self.bm_init_q_mujoco = output_0["joint_pos"].reshape(-1)[self.policy2robot_index]

        _, _, ref_yaw = quat_to_rpy(self.bm_anchor_orn_wxyz_0.reshape(-1))
        ref_heading_quat_wxyz = rpy_to_quat((0.0, 0.0, ref_yaw)).reshape(1, 4)
        self.bm_ref_init_heading_inv_wxyz = quat_inverse(ref_heading_quat_wxyz)

        # Compatibility fields for WholeBodyTrackingPolicy slot state.
        self.motion_command_t = self.bm_motion_command_t.copy()
        self.motion_command_0 = self.bm_motion_command_0.copy()
        self.ref_quat_xyzw_t = np.zeros((1, 4), dtype=np.float32)
        self.ref_quat_xyzw_0 = np.zeros((1, 4), dtype=np.float32)

        self.onnx_kp = None
        self.onnx_kd = None

        def policy_act(_input_feed):
            raise RuntimeError("BM policy uses custom rl_inference path.")

        self.policy = policy_act

    def _build_onnx_input_feed(self, time_step: np.ndarray, obs: np.ndarray) -> dict[str, np.ndarray]:
        obs_input_name = "obs" if "obs" in self.onnx_input_names else self.onnx_input_names[0]
        input_feed = {obs_input_name: obs.astype(np.float32)}
        if "time_step" in self.onnx_input_names:
            input_feed["time_step"] = time_step.astype(np.float32)
        return input_feed

    def _select_anchor_body_output(self, tensor: np.ndarray, output_name: str) -> np.ndarray:
        ref_body_index = self.bm_ref_body_index

        if tensor.ndim == 3:
            if ref_body_index >= tensor.shape[1]:
                ref_body_index = 0
                if not self._warned_ref_body_fallback:
                    logger.warning(
                        "BM ref body index {} out of range for '{}' (num bodies={}). Using index 0.",
                        self.bm_ref_body_index,
                        output_name,
                        tensor.shape[1],
                    )
                    self._warned_ref_body_fallback = True
            return tensor[:, ref_body_index, :]

        if tensor.ndim == 2:
            if ref_body_index >= tensor.shape[0]:
                ref_body_index = 0
                if not self._warned_ref_body_fallback:
                    logger.warning(
                        "BM ref body index {} out of range for '{}' (num bodies={}). Using index 0.",
                        self.bm_ref_body_index,
                        output_name,
                        tensor.shape[0],
                    )
                    self._warned_ref_body_fallback = True
            return tensor[ref_body_index : ref_body_index + 1, :]

        raise ValueError(f"Unexpected {output_name} shape: {tensor.shape}")

    def _run_policy(self, time_step: np.ndarray, obs: np.ndarray) -> dict[str, np.ndarray]:
        outputs = self.onnx_policy_session.run(None, self._build_onnx_input_feed(time_step, obs))
        output_map = {name: value for name, value in zip(self.onnx_output_names, outputs)}

        required = ("actions", "joint_pos", "joint_vel", "body_quat_w")
        missing = [name for name in required if name not in output_map]
        if missing:
            raise ValueError(f"BM ONNX missing outputs {missing}. Available outputs: {self.onnx_output_names}")

        actions = output_map["actions"].reshape(-1).astype(np.float32, copy=False)
        joint_pos = output_map["joint_pos"].reshape(1, -1).astype(np.float32, copy=False)
        joint_vel = output_map["joint_vel"].reshape(1, -1).astype(np.float32, copy=False)

        body_quat_w = output_map["body_quat_w"]
        anchor_orn_wxyz = self._select_anchor_body_output(body_quat_w, "body_quat_w")

        anchor_pos_w = np.zeros((1, 3), dtype=np.float32)
        if "body_pos_w" in output_map:
            body_pos_w = output_map["body_pos_w"]
            if body_pos_w.ndim in (2, 3):
                anchor_pos_w = self._select_anchor_body_output(body_pos_w, "body_pos_w")
            else:
                raise ValueError(f"Unexpected body_pos_w shape: {body_pos_w.shape}")

        return {
            "actions": actions,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "anchor_pos_w": anchor_pos_w.astype(np.float32, copy=False),
            "anchor_orn_wxyz": anchor_orn_wxyz.astype(np.float32, copy=False),
        }

    def _compute_obs_terms(self, robot_state_data) -> dict[str, np.ndarray]:
        base_quat_wxyz = robot_state_data[0, 3:7]
        dof_pos = robot_state_data[0, 7 : 7 + self.num_dofs]
        base_ang_vel = robot_state_data[0, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]
        dof_vel = robot_state_data[0, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs]

        if not self.bm_heading_initialized:
            _, _, yaw = quat_to_rpy(base_quat_wxyz)
            heading_quat = rpy_to_quat((0.0, 0.0, yaw)).reshape(1, 4)
            self.bm_init_heading_inv_wxyz = quat_inverse(heading_quat)
            self.bm_heading_initialized = True

        root_orn_wxyz = quat_mul(self.bm_init_heading_inv_wxyz, base_quat_wxyz.reshape(1, 4))
        robot_ref_ori_wxyz = (
            self._get_ref_body_orientation_in_world(robot_state_data) if self.bm_use_pinocchio_ref else root_orn_wxyz
        )
        motion_ref_ori_wxyz = self._remove_yaw_offset(self.bm_anchor_orn_wxyz_t.copy(), self.motion_yaw_offset)
        robot_ref_ori_wxyz = self._remove_yaw_offset(robot_ref_ori_wxyz, self.robot_yaw_offset)

        relative_quat_wxyz = subtract_frame_transforms(robot_ref_ori_wxyz, motion_ref_ori_wxyz)
        motion_ref_ori_b = matrix_from_quat(relative_quat_wxyz)[..., :2].reshape(-1).astype(np.float32)

        command = self.bm_motion_command_t.reshape(-1).astype(np.float32, copy=False)
        dof_pos_policy = (dof_pos[self.robot2policy_index] - self.bm_default_pos_isaac).astype(np.float32, copy=False)
        dof_vel_policy = dof_vel[self.robot2policy_index].astype(np.float32, copy=False)

        return {
            "command": command,
            "motion_ref_ori_b": motion_ref_ori_b,
            "base_ang_vel": base_ang_vel.astype(np.float32, copy=False),
            "joint_pos": dof_pos_policy,
            "joint_vel": dof_vel_policy,
            "actions": self.bm_last_action.astype(np.float32, copy=False),
        }

    def _build_bm_observation(self, robot_state_data) -> np.ndarray:
        term_values = self._compute_obs_terms(robot_state_data)
        obs = np.concatenate([term_values[term] for term in self.bm_obs_terms], axis=0).reshape(1, -1)
        if self.bm_obs_dim is not None and obs.shape[1] != self.bm_obs_dim:
            raise ValueError(
                f"BM observation dimension mismatch. got={obs.shape[1]} expected={self.bm_obs_dim} "
                f"terms={self.bm_obs_terms}"
            )
        return obs.astype(np.float32, copy=False)

    def rl_inference(self, robot_state_data):
        if not self.motion_clip_progressing:
            self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
            self.curr_motion_timestep = self.timestep_util.timestep
            self.bm_motion_command_t = self.bm_motion_command_0.copy()
            self.bm_anchor_pos_w_t = self.bm_anchor_pos_w_0.copy()
            self.bm_anchor_orn_wxyz_t = self.bm_anchor_orn_wxyz_0.copy()

        obs = self._build_bm_observation(robot_state_data)
        output = self._run_policy(np.array([[self.curr_motion_timestep]], dtype=np.float32), obs)
        action_policy = output["actions"]
        action_robot = action_policy[self.policy2robot_index]

        self.bm_last_action = action_policy.copy()
        self.last_policy_action = action_robot.reshape(1, -1)
        joint_pos_robot = output["joint_pos"][:, self.policy2robot_index]
        joint_vel_robot = output["joint_vel"][:, self.policy2robot_index]
        self.bm_motion_command_t = np.concatenate([output["joint_pos"], output["joint_vel"]], axis=1)
        self.motion_command_t = np.concatenate([joint_pos_robot, joint_vel_robot], axis=1)
        self.bm_anchor_pos_w_t = output["anchor_pos_w"]
        self.bm_anchor_orn_wxyz_t = output["anchor_orn_wxyz"]

        q_target = action_robot * self.bm_action_scale_mujoco + self.bm_default_pos_mujoco
        self.scaled_policy_action = (q_target - self.default_dof_angles).reshape(1, -1)

        if self.motion_clip_progressing:
            self._set_motion_timestep()

        return self.scaled_policy_action

    def get_init_target(self, robot_state_data):
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            target = self.bm_init_q_mujoco.reshape(1, -1)
            q_target = dof_pos + (target - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def get_current_obs_buffer_dict(self, robot_state_data):
        # This policy builds observations directly in rl_inference; return placeholders
        # for compatibility with BasePolicy helpers.
        term_values = self._compute_obs_terms(robot_state_data)
        return {term: value.reshape(1, -1) for term, value in term_values.items() if term in self.obs_dims}

    def _handle_start_policy(self):
        BasePolicy._handle_start_policy(self)
        self._stiff_hold_active = False
        self.bm_heading_initialized = False
        self._capture_robot_yaw_offset()
        self._capture_motion_yaw_offset()

    def _handle_stop_policy(self):
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.bm_last_action = np.zeros(29, dtype=np.float32)
        self.bm_heading_initialized = False
        self.bm_motion_command_t = self.bm_motion_command_0.copy()
        self.bm_anchor_pos_w_t = self.bm_anchor_pos_w_0.copy()
        self.bm_anchor_orn_wxyz_t = self.bm_anchor_orn_wxyz_0.copy()
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0
        self.logger.info(colored("BM policy stopped, entering stiff hold", "yellow"))

    def _handle_start_motion_clip(self):
        super()._handle_start_motion_clip()
        self.bm_motion_command_t = self.bm_motion_command_0.copy()
        self.bm_anchor_pos_w_t = self.bm_anchor_pos_w_0.copy()
        self.bm_anchor_orn_wxyz_t = self.bm_anchor_orn_wxyz_0.copy()

    def _capture_policy_state(self):
        state = BasePolicy._capture_policy_state(self)
        state.update(
            {
                "bm_default_pos_isaac": self.bm_default_pos_isaac.copy(),
                "bm_default_pos_mujoco": self.bm_default_pos_mujoco.copy(),
                "bm_action_scale_mujoco": self.bm_action_scale_mujoco.copy(),
                "bm_motion_command_0": self.bm_motion_command_0.copy(),
                "bm_anchor_pos_w_0": self.bm_anchor_pos_w_0.copy(),
                "bm_anchor_orn_wxyz_0": self.bm_anchor_orn_wxyz_0.copy(),
                "bm_ref_init_heading_inv_wxyz": self.bm_ref_init_heading_inv_wxyz.copy(),
                "bm_init_q_mujoco": self.bm_init_q_mujoco.copy(),
                "bm_obs_terms": list(self.bm_obs_terms),
                "bm_obs_dim": self.bm_obs_dim,
                "bm_ref_body_index": self.bm_ref_body_index,
                "policy2robot_index": self.policy2robot_index.copy(),
                "robot2policy_index": self.robot2policy_index.copy(),
                "bm_use_pinocchio_ref": self.bm_use_pinocchio_ref,
                "pinocchio_robot": self.pinocchio_robot if self.bm_use_pinocchio_ref else None,
                "motion_command_0": np.zeros((1, 58), dtype=np.float32),
                "ref_quat_xyzw_0": np.zeros((1, 4), dtype=np.float32),
            }
        )
        return state

    def _restore_policy_state(self, state):
        BasePolicy._restore_policy_state(self, state)
        self.bm_default_pos_isaac = state["bm_default_pos_isaac"].copy()
        self.bm_default_pos_mujoco = state["bm_default_pos_mujoco"].copy()
        self.bm_action_scale_mujoco = state["bm_action_scale_mujoco"].copy()
        self.bm_motion_command_0 = state["bm_motion_command_0"].copy()
        self.bm_anchor_pos_w_0 = state["bm_anchor_pos_w_0"].copy()
        self.bm_anchor_orn_wxyz_0 = state["bm_anchor_orn_wxyz_0"].copy()
        self.bm_ref_init_heading_inv_wxyz = state["bm_ref_init_heading_inv_wxyz"].copy()
        self.bm_init_q_mujoco = state["bm_init_q_mujoco"].copy()
        self.bm_obs_terms = list(state["bm_obs_terms"])
        self.bm_obs_dim = state["bm_obs_dim"]
        self.bm_ref_body_index = state["bm_ref_body_index"]
        self.policy2robot_index = state["policy2robot_index"].copy()
        self.robot2policy_index = state["robot2policy_index"].copy()
        self.bm_use_pinocchio_ref = state["bm_use_pinocchio_ref"]
        if self.bm_use_pinocchio_ref and state.get("pinocchio_robot") is not None:
            self.pinocchio_robot = state["pinocchio_robot"]

        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.bm_last_action = np.zeros(29, dtype=np.float32)
        self.bm_heading_initialized = False
        self.bm_motion_command_t = self.bm_motion_command_0.copy()
        self.bm_anchor_pos_w_t = self.bm_anchor_pos_w_0.copy()
        self.bm_anchor_orn_wxyz_t = self.bm_anchor_orn_wxyz_0.copy()

        self.motion_command_0 = np.zeros((1, 58), dtype=np.float32)
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_0 = np.zeros((1, 4), dtype=np.float32)
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()

    def _on_policy_switched(self, model_path):
        _ = model_path
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self._stiff_hold_active = True
        self.bm_last_action = np.zeros(29, dtype=np.float32)
        self.bm_heading_initialized = False
        self.bm_motion_command_t = self.bm_motion_command_0.copy()
        self.bm_anchor_pos_w_t = self.bm_anchor_pos_w_0.copy()
        self.bm_anchor_orn_wxyz_t = self.bm_anchor_orn_wxyz_0.copy()
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
