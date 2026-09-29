"""Headless MuJoCo simulation factory for sim2sim evaluation.

Provides:
- MuJoCoInterface: in-process MuJoCo interface matching UnitreeInterface contract
- load_onnx_metadata: load ONNX metadata once (shared across callers)
- build_config_from_onnx: build InferenceConfig from ONNX metadata
- _setup_headless_sim: create MuJoCo simulation for headless eval (run_policy.py path)
"""

from __future__ import annotations

import dataclasses
import glob as _glob
import json
import os
import shutil
import tempfile

import mujoco
import numpy as np
import onnxruntime as ort
from loguru import logger

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_types.observation import ObservationConfig

from wbt_wrappers_inference.config_values.robot_config import ISAACLAB_TO_MUJOCO_DOF as _ISAAC_TO_MUJOCO
from wbt_wrappers_inference.config_values.inference import _g1_29dof_wbt_wrapper_robot
from wbt_wrappers_inference.config_values.task import WBTTaskConfig, MaskedMimicTaskConfig
from wbt_wrappers_inference.utils.obs_utils import detect_obs_mode
from wbt_wrappers_inference.utils.terrain_replay import add_onpath_obstacle_replay, combine_terrains


def _prepare_headless_replay_assets(ref_motion_path: str, terrain_path: str) -> tuple[str, str]:
    """Replay-pair raw terrain NPY with its motion for single-env headless eval.

    Training terrain runs do not consume the raw `*_motion.npz` and raw
    `*_terrain.npy` independently. They first pass through the obstacle replay
    helper, which samples a deterministic single-env layout (seed=42) and
    offsets the motion onto that layout. Mirror that here so headless sim2sim
    evaluates the same paired artifacts instead of a mismatched raw pair.

    All generated files are written to a per-process temp dir — the source
    motion dataset is never modified.

    Returns:
        `(paired_motion_path, paired_terrain_path)` where the terrain path is an
        OBJ mesh ready for MuJoCo loading.
    """
    if not terrain_path.endswith(".npy"):
        return ref_motion_path, terrain_path

    tmp_dir = tempfile.mkdtemp(prefix="headless_replay_")
    # add_onpath_obstacle_replay writes `<motion_stem>_42.npz` next to the
    # input motion. Copy the source into the temp dir first so the output
    # lands in tmp, not the source dataset directory.
    tmp_motion = os.path.join(tmp_dir, os.path.basename(ref_motion_path))
    shutil.copyfile(ref_motion_path, tmp_motion)
    combined_terrain = combine_terrains([terrain_path])
    paired_motion_path, terrain_mesh = add_onpath_obstacle_replay(tmp_motion, str(combined_terrain))

    paired_terrain_path = os.path.join(tmp_dir, "terrain.obj")
    terrain_mesh.export(paired_terrain_path)
    logger.info(
        f"Prepared replay-paired headless terrain assets: motion={paired_motion_path} terrain={paired_terrain_path}"
    )
    return paired_motion_path, paired_terrain_path


# ---------------------------------------------------------------------------
# MuJoCo Interface
# ---------------------------------------------------------------------------
class MuJoCoInterface:
    """MuJoCo-backed robot interface for headless sim2sim evaluation.

    State layout matches UnitreeInterface.get_low_state():
        [base_pos(3) | quat_wxyz(4) | joint_pos(N) | base_lin_vel(3) | base_ang_vel(3) | joint_vel(N)]
    For G1-29dof: N=29, total=71 elements, shape (1, 71).
    """

    def __init__(
        self,
        mj_model,
        mj_data,
        robot_config,
        kp,
        kd,
        torque_limit,
        num_joints: int = 29,
        steps_per_control: int = 1,
        no_mocap: bool = False,
        mocap_yaw_offset: float = 0.0,
        mocap_use_orientation: bool = False,
        force_ext_pelvis: np.ndarray | None = None,
    ):
        self.model = mj_model
        self.data = mj_data
        self.robot_config = robot_config
        self.sdk_type = "mujoco"
        self._num_joints = num_joints
        self._steps_per_control = steps_per_control
        self._no_mocap = no_mocap
        self._mocap_yaw_offset = mocap_yaw_offset
        self._mocap_use_orientation = mocap_use_orientation
        # --- Eval-3 external-force injection (UniFP F_ext, g1-wbt-2pt-force) ---
        # force_ext_pelvis: (2,3) PELVIS-frame Newtons [left, right], or None to
        # disable. Injected EXACTLY like training (force_tracking.py:457-487):
        # a PURE FORCE (torque=0) applied AT THE rubber_hand POINT on each
        # wrist_yaw body, EVERY physics substep, with the pelvis-frame vector
        # rotated to world by the LIVE pelvis quat each step.
        self._force_ext_pelvis = None
        self._force_ext_hand_ids = None
        if force_ext_pelvis is not None and np.any(np.asarray(force_ext_pelvis) != 0.0):
            self._force_ext_pelvis = np.asarray(force_ext_pelvis, dtype=np.float64).reshape(2, 3)
            self._force_ext_hand_ids = [
                mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
                for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")
            ]
            # rubber_hand FK offset in wrist_yaw local frame (left, right).
            # YAM gripper: net wrist_yaw→rubber_hand offset 0.0815 (= ft_sensor
            # 0.072 + palm 0.0095). Must match HandForceConfig.body_offsets_b.
            self._force_ext_offsets_b = np.array([[0.0815, +0.003, 0.0], [0.0815, -0.003, 0.0]], dtype=np.float64)
            self._force_ext_qfrc = np.zeros(mj_model.nv, dtype=np.float64)
        # Precompute yaw rotation for mocap frame offset.
        # When mocap_use_orientation=False: position/velocity in "mocap frame",
        #   orientation in "IMU frame" (= MuJoCo frame) → frame mismatch.
        # When mocap_use_orientation=True: position/velocity AND orientation all
        #   in "mocap frame" → no mismatch.
        c, s = np.cos(mocap_yaw_offset), np.sin(mocap_yaw_offset)
        self._mocap_rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        # Precompute quaternion for yaw rotation: q_yaw = [cos(θ/2), 0, 0, sin(θ/2)] (wxyz)
        self._mocap_quat = np.array([np.cos(mocap_yaw_offset / 2), 0.0, 0.0, np.sin(mocap_yaw_offset / 2)])
        self._state_dim = 3 + 4 + num_joints + 3 + 3 + num_joints
        self._buf = np.zeros((1, self._state_dim), dtype=np.float64)
        self._kp_base = np.asarray(kp, dtype=np.float64)
        self._kd_base = np.asarray(kd, dtype=np.float64)
        self.kp = self._kp_base.copy()
        self.kd = self._kd_base.copy()
        self.torque_limit = np.asarray(torque_limit, dtype=np.float64)
        self._gain_level = 1.0

    @staticmethod
    def _quat_mul_wxyz(q1, q2):
        """Multiply two wxyz quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    def get_low_state(self) -> np.ndarray:
        n = self._num_joints
        buf = self._buf
        has_offset = self._mocap_yaw_offset != 0.0
        if self._no_mocap:
            buf[0, 0:3] = 0.0
        elif has_offset:
            # Position in "mocap frame" (rotated from MuJoCo frame)
            xy = self._mocap_rot @ self.data.qpos[0:2]
            buf[0, 0:2] = xy
            buf[0, 2] = self.data.qpos[2]
        else:
            buf[0, 0:3] = self.data.qpos[0:3]
        # Orientation
        if has_offset and self._mocap_use_orientation:
            # Both pos and ori in "mocap frame" — no mismatch
            buf[0, 3:7] = self._quat_mul_wxyz(self._mocap_quat, self.data.qpos[3:7])
        else:
            # Ori from "IMU" (= MuJoCo frame, no rotation)
            buf[0, 3:7] = self.data.qpos[3:7]
        buf[0, 7 : 7 + n] = self.data.qpos[7 : 7 + n]
        if self._no_mocap:
            buf[0, 7 + n : 7 + n + 3] = 0.0
        elif has_offset:
            # Velocity in "mocap frame"
            xy_vel = self._mocap_rot @ self.data.qvel[0:2]
            buf[0, 7 + n] = xy_vel[0]
            buf[0, 7 + n + 1] = xy_vel[1]
            buf[0, 7 + n + 2] = self.data.qvel[2]
        else:
            buf[0, 7 + n : 7 + n + 3] = self.data.qvel[0:3]
        buf[0, 7 + n + 3 : 7 + n + 6] = self.data.qvel[3:6]
        buf[0, 7 + n + 6 : 7 + 2 * n + 6] = self.data.qvel[6 : 6 + n]
        return buf

    def _apply_force_ext(self):
        """Inject the Eval-3 external force into data.qfrc_applied for one substep.

        Matches training's _inject_force_ext (force_tracking.py:457-487): a PURE
        force (torque=0) applied AT THE rubber_hand POINT (wrist_yaw origin +
        R(wrist_quat)@offset), with the pelvis-frame command rotated to the world
        frame by the LIVE pelvis quat. mj_applyFT maps the world wrench at that
        world point onto generalized forces, so the 4.15 cm offset's moment arm
        is included — applying at the wrist COM (xfrc_applied) would drop it.
        mj_step consumes qfrc_applied; we zero it first so substeps don't stack.
        """
        if self._force_ext_pelvis is None:
            return
        self._force_ext_qfrc.fill(0.0)
        d = self.data
        # Live pelvis (root) world rotation from the free joint quat (wxyz).
        w, x, y, z = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
        R_root = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        zero_torque = np.zeros(3)
        for hi, body_id in enumerate(self._force_ext_hand_ids):
            if body_id < 0:
                continue
            f_pelvis = self._force_ext_pelvis[hi]
            if not np.any(f_pelvis):
                continue
            f_world = R_root @ f_pelvis
            hand_pt = d.xpos[body_id] + d.xmat[body_id].reshape(3, 3) @ self._force_ext_offsets_b[hi]
            mujoco.mj_applyFT(self.model, d, f_world, zero_torque, hand_pt, body_id, self._force_ext_qfrc)
        d.qfrc_applied[:] = self._force_ext_qfrc

    def step(self, target_q):
        n = self._num_joints
        tau = (target_q - self.data.qpos[7 : 7 + n]) * self.kp - self.data.qvel[6 : 6 + n] * self.kd
        self.data.ctrl[:] = np.clip(tau, -self.torque_limit, self.torque_limit)
        self._apply_force_ext()
        mujoco.mj_step(self.model, self.data)

    def send_low_command(
        self, cmd_q, cmd_dq=None, cmd_tau=None, dof_pos_latest=None, kp_override=None, kd_override=None
    ):
        """Match BaseInterface.send_low_command — PD control + decimated physics steps."""
        target_q = np.asarray(cmd_q).flatten()
        saved_kp, saved_kd = self.kp, self.kd
        if kp_override is not None:
            self.kp = np.asarray(kp_override, dtype=np.float64)
        if kd_override is not None:
            self.kd = np.asarray(kd_override, dtype=np.float64)
        for _ in range(self._steps_per_control):
            self.step(target_q)
        self.kp, self.kd = saved_kp, saved_kd

    def update_config(self, robot_config):
        self.robot_config = robot_config
        # Propagate KP/KD from robot_config to base gains — matches real hardware
        # where update_config pushes ONNX gains to motor controller.
        if hasattr(robot_config, "motor_kp") and robot_config.motor_kp is not None:
            self._kp_base = np.asarray(robot_config.motor_kp, dtype=np.float64)
            self.kp = self._kp_base * self._gain_level
        if hasattr(robot_config, "motor_kd") and robot_config.motor_kd is not None:
            self._kd_base = np.asarray(robot_config.motor_kd, dtype=np.float64)
            self.kd = self._kd_base * self._gain_level

    def get_joystick_msg(self):
        return None

    def get_joystick_key(self, wc_msg=None):
        return None

    @property
    def kp_level(self):
        return self._gain_level

    @kp_level.setter
    def kp_level(self, value):
        self._gain_level = value
        self.kp = self._kp_base * value
        self.kd = self._kd_base * value

    def shutdown(self):
        pass


# ---------------------------------------------------------------------------
# ONNX metadata loading (single session, shared across callers)
# ---------------------------------------------------------------------------
def _find_latest_onnx(onnx_path_or_dir: str) -> str:
    """Resolve to a concrete ONNX file.

    If given an existing ``.onnx`` FILE, return it verbatim — this honors an
    explicit ``--task.model-path`` checkpoint. Only when given a DIRECTORY do we
    glob and pick the latest. CRITICAL for benchmarking: the training run may
    still be writing new checkpoints into the same dir, so globbing-the-latest
    would silently evaluate a moving (and sometimes mid-write/corrupt) target
    instead of the requested checkpoint.
    """
    abspath = os.path.abspath(onnx_path_or_dir)
    if os.path.isfile(abspath) and abspath.endswith(".onnx"):
        return abspath
    onnx_files = sorted(_glob.glob(os.path.join(abspath, "*.onnx")))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX files in {abspath}")
    return onnx_files[-1]


def load_onnx_metadata(onnx_path_or_dir: str) -> tuple[str, dict | None, dict]:
    """Load ONNX metadata once.  Returns (onnx_path, exp_config, meta_map).

    Accepts either an exact ``.onnx`` file (preferred — honors the requested
    checkpoint) or a directory (globs the latest). *exp_config* is None for
    TML-format ONNX files (flat metadata, no experiment_config).
    """
    onnx_path = _find_latest_onnx(onnx_path_or_dir)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta_map = session.get_modelmeta().custom_metadata_map
    exp_config = json.loads(meta_map["experiment_config"]) if "experiment_config" in meta_map else None
    del session
    return onnx_path, exp_config, meta_map


# ---------------------------------------------------------------------------
# Config + policy factory
# ---------------------------------------------------------------------------
def build_config_from_onnx(
    onnx_dir: str,
    ref_motion_path: str,
    mm_tracking_mode: str = "",
    metadata: tuple[str, dict | None, dict] | None = None,
    action_ema_alpha: float = 1.0,
    action_clip_sim2real: float = 0.0,
    action_ramp_steps: int = 0,
) -> tuple[InferenceConfig, dict]:
    """Build InferenceConfig and sim params from ONNX checkpoint metadata.

    Handles both holosoma format (experiment_config JSON) and TML format
    (flat comma-separated metadata strings).

    If *metadata* is provided (from load_onnx_metadata), reuses it to avoid
    opening a second ONNX session.
    """
    if metadata is not None:
        onnx_path, exp_config, meta = metadata
    else:
        onnx_path, exp_config, meta = load_onnx_metadata(onnx_dir)

    if exp_config is None:
        return _build_tml_config(onnx_path, meta, ref_motion_path)
    return _build_holosoma_config(
        onnx_path,
        exp_config,
        meta,
        ref_motion_path,
        mm_tracking_mode,
        action_ema_alpha=action_ema_alpha,
        action_clip_sim2real=action_clip_sim2real,
        action_ramp_steps=action_ramp_steps,
    )


# TML observation_names → policy-internal names and dimensions
_TML_NAME_MAP = {
    "command": "tml_command",
    "motion_anchor_ori_b": "motion_anchor_ori_b",
    "gravity_dir": "projected_gravity",
    "base_ang_vel": "base_ang_vel",
    "joint_pos": "dof_pos",
    "joint_vel": "dof_vel",
    "actions": "actions",
}
_TML_DIM_MAP = {
    "tml_command": 58,
    "motion_anchor_ori_b": 6,
    "projected_gravity": 3,
    "base_ang_vel": 3,
    "dof_pos": 29,
    "dof_vel": 29,
    "actions": 29,
}


def _build_tml_config(onnx_path, meta, ref_motion_path):
    """Build config from TML-format ONNX (flat metadata, no experiment_config)."""
    raw_names = [n.strip() for n in meta.get("observation_names", "").split(",") if n.strip()]
    mapped_names = [_TML_NAME_MAP.get(n, n) for n in raw_names]
    obs_dims = {n: _TML_DIM_MAP.get(n, 1) for n in mapped_names}

    observation = ObservationConfig(
        obs_dict={"actor_obs": mapped_names},
        obs_dims=obs_dims,
        obs_scales={n: 1.0 for n in mapped_names},
        history_length_dict={"actor_obs": 1},
    )

    task = WBTTaskConfig(
        model_path=onnx_path,
        policy_type="tml_wbt",
        rl_rate=50,
        policy_action_scale=1.0,
        use_phase=False,
        state_input="interface",
        ref_motion_path=os.path.abspath(ref_motion_path),
    )

    config = InferenceConfig(robot=_g1_29dof_wbt_wrapper_robot, observation=observation, task=task)

    # KP/KD from TML metadata (Isaac order → MuJoCo order)
    kp_isaac = np.array([float(x) for x in meta["joint_stiffness"].split(",")])
    kd_isaac = np.array([float(x) for x in meta["joint_damping"].split(",")])
    sim_params = {
        "kp": kp_isaac[_ISAAC_TO_MUJOCO],
        "kd": kd_isaac[_ISAAC_TO_MUJOCO],
        "torque_limit": np.full(29, 200.0),  # generous default for G1
        "sim_dt": 0.005,  # 200 Hz physics
        "control_dt": 0.02,  # 50 Hz control (decimation=4)
        "obs_mode": "tml",
        "robot_urdf": meta.get("robot_urdf", ""),
    }
    return config, sim_params


_WBT_FUNC_DIMS = {
    "motion_ref_ori_b": 6,
    "projected_gravity": 3,
    "base_ang_vel": 3,
    "dof_pos": 29,
    "dof_vel": 29,
    "actions": 29,
}


def _build_holosoma_config(
    onnx_path,
    exp_config,
    meta,
    ref_motion_path,
    mm_tracking_mode,
    action_ema_alpha=1.0,
    action_clip_sim2real=0.0,
    action_ramp_steps=0,
):
    """Build config from holosoma-format ONNX (experiment_config JSON)."""
    obs_cfg = exp_config["observation"]["groups"]["actor_obs"]
    terms = obs_cfg.get("terms", {})
    term_names = sorted(terms.keys())
    history_length = obs_cfg.get("history_length", 1)

    # Get total ONNX obs dim for back-computing unknown term dims. The obs input
    # is named "obs" in the deploy export and "actor_obs" in the browser/mjswan
    # export — accept either (else onnx_obs_dim stays None and the force/servo
    # terms, which aren't in _WBT_FUNC_DIMS, can't be back-computed).
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_obs_dim = None
    for inp in session.get_inputs():
        if inp.name in ("obs", "actor_obs") and len(inp.shape) >= 2 and isinstance(inp.shape[-1], int):
            onnx_obs_dim = inp.shape[-1]
    del session

    obs_dims = {}
    unknown_terms = []
    for name, term_cfg in terms.items():
        dim = None
        if isinstance(term_cfg, dict):
            dim = term_cfg.get("dim")
            if dim is None:
                func = term_cfg.get("func", "")
                short_name = func.split(":")[-1] if ":" in func else ""
                dim = _WBT_FUNC_DIMS.get(short_name)
        else:
            dim = int(term_cfg)
        if dim is not None:
            obs_dims[name] = dim
        else:
            unknown_terms.append(name)

    if unknown_terms and onnx_obs_dim is not None:
        per_frame = onnx_obs_dim // max(history_length, 1)
        known_total = sum(obs_dims.values())
        remaining = per_frame - known_total
        if len(unknown_terms) == 1:
            obs_dims[unknown_terms[0]] = remaining
        else:
            per_unknown = remaining // len(unknown_terms)
            for t in unknown_terms:
                obs_dims[t] = per_unknown

    if not obs_dims:
        logger.warning("Could not determine obs dims from ONNX metadata")

    obs_scales = {name: 1.0 for name in term_names}
    observation = ObservationConfig(
        obs_dict={"actor_obs": term_names},
        obs_dims=obs_dims,
        obs_scales=obs_scales,
        history_length_dict={"actor_obs": history_length},
    )

    obs_mode = detect_obs_mode(term_names)
    is_maskedmimic = obs_mode == "maskedmimic"

    if is_maskedmimic:
        task = MaskedMimicTaskConfig(
            model_path=onnx_path,
            policy_type="holosoma_maskedmimic",
            rl_rate=50,
            policy_action_scale=1.0,
            use_phase=False,
            state_input="interface",
            ref_motion_path=os.path.abspath(ref_motion_path),
            maskedmimic_mode=mm_tracking_mode or "three_point",
            action_ema_alpha=action_ema_alpha,
            action_clip_sim2real=action_clip_sim2real,
            action_ramp_steps=action_ramp_steps,
        )
    else:
        task = WBTTaskConfig(
            model_path=onnx_path,
            policy_type="holosoma_wbt",
            rl_rate=50,
            policy_action_scale=1.0,
            action_scales_by_effort_limit_over_p_gain=True,
            use_phase=False,
            state_input="interface",
            ref_motion_path=os.path.abspath(ref_motion_path),
        )

    config = InferenceConfig(robot=_g1_29dof_wbt_wrapper_robot, observation=observation, task=task)

    sim_fps = exp_config["simulator"]["config"]["sim"]["fps"]
    decimation = exp_config["simulator"]["config"]["sim"]["control_decimation"]
    # Extract robot URDF from metadata (for FK odometry)
    robot_urdf = meta.get("robot_urdf", "")
    if robot_urdf:
        try:
            robot_urdf = json.loads(robot_urdf)
        except (json.JSONDecodeError, TypeError):
            pass

    sim_params = {
        "kp": np.array([float(x) for x in meta["kp"].strip("[]").split(",")]),
        "kd": np.array([float(x) for x in meta["kd"].strip("[]").split(",")]),
        "torque_limit": np.array(exp_config["robot"]["dof_effort_limit_list"], dtype=np.float64),
        "sim_dt": 1.0 / sim_fps,
        "control_dt": decimation / sim_fps,
        "obs_mode": obs_mode,
        "robot_urdf": robot_urdf,
    }

    return config, sim_params


class _NullVelCmd:
    def start(self):
        pass

    def poll_velocity(self):
        return None

    def zero(self):
        pass


class _NullStateCmd:
    def start(self):
        pass

    def poll_commands(self):
        return []


class _HardwareSource:
    """Minimal stub satisfying BasePolicy._shared_hardware_source contract."""

    def __init__(self, interface, rl_rate=50):
        self.interface = interface
        self.sdk_type = "mujoco"
        self.logger = logger
        self.rate = None
        self.rl_rate = rl_rate
        self.use_joystick = False
        self.use_keyboard = False
        self._velocity_input = _NullVelCmd()
        self._command_provider = _NullStateCmd()


def _setup_headless_sim(config: InferenceConfig):
    """Create MuJoCo simulation for headless eval. Returns setup dict."""
    from wbt_wrappers_inference.utils.eval_utils import resolve_xml, _load_model

    task = config.task
    if not task.headless_output:
        raise ValueError("--task.headless-output is required for headless eval")
    if not task.ref_motion_path:
        raise ValueError("--task.ref-motion-path is required for headless eval")

    # Resolve the EXACT checkpoint: if model_path is a concrete .onnx file use it
    # verbatim (do NOT glob-latest from its dir — training may still be writing
    # newer/mid-write checkpoints there). Fall back to the dir only when a dir or
    # non-existent file is given.
    _abs_model = os.path.abspath(task.model_path)
    onnx_ref = (
        _abs_model if (os.path.isfile(_abs_model) and _abs_model.endswith(".onnx")) else os.path.dirname(_abs_model)
    )
    ref_motion_path = os.path.abspath(task.ref_motion_path)
    terrain_obj = os.path.abspath(task.headless_terrain) if task.headless_terrain else None

    if terrain_obj is not None:
        ref_motion_path, terrain_obj = _prepare_headless_replay_assets(ref_motion_path, terrain_obj)
        # Task config is frozen; rebuild it with the replayed motion path so
        # policy-side motion loading matches the simulator's paired assets.
        config = dataclasses.replace(config, task=dataclasses.replace(config.task, ref_motion_path=ref_motion_path))
        task = config.task

    # Get sim_params from ONNX metadata (physics dt, KP/KD, torque limits)
    metadata = load_onnx_metadata(onnx_ref)
    _, sim_params = build_config_from_onnx(
        onnx_ref,
        ref_motion_path,
        mm_tracking_mode=getattr(task, "maskedmimic_mode", ""),
        metadata=metadata,
        action_ema_alpha=getattr(task, "action_ema_alpha", 1.0),
        action_clip_sim2real=getattr(task, "action_clip_sim2real", 0.0),
        action_ramp_steps=getattr(task, "action_ramp_steps", 0),
    )

    # Create MuJoCo model + data with initial pose from reference. Auto-derive the
    # render/sim scene from the ONNX robot_urdf_path stem (e.g. g1_29dof_yam → the
    # yam-gripper scene) when no explicit --task.headless-xml is given, so a yam
    # checkpoint renders the yam gripper. metadata = (onnx_path, exp_config, meta_map).
    _meta_map = metadata[2] if metadata is not None and len(metadata) >= 3 else None
    xml_path = resolve_xml(task.headless_xml or None, meta=_meta_map)
    mj_model = _load_model(xml_path, terrain_obj=terrain_obj)
    if terrain_obj:
        logger.info(f"Terrain mesh loaded: {terrain_obj}")
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = sim_params["sim_dt"]

    ref = np.load(ref_motion_path)
    ncols = ref["joint_pos"].shape[1]
    if ncols == 36:
        mj_data.qpos[7:36] = ref["joint_pos"][0, 7:]
        # For holosoma-format terrain replay, body_pos_w/body_quat_w are the
        # source of truth after obstacle replay offsets are baked in. joint_pos
        # root DOFs are not updated by the replay helper.
        if "body_pos_w" in ref and "body_quat_w" in ref:
            mj_data.qpos[0:3] = ref["body_pos_w"][0, 0]
            mj_data.qpos[3:7] = ref["body_quat_w"][0, 0]
        else:
            mj_data.qpos[0:3] = ref["joint_pos"][0, :3]
            mj_data.qpos[3:7] = ref["joint_pos"][0, 3:7]
    else:
        mj_data.qpos[7:36] = ref["joint_pos"][0][_ISAAC_TO_MUJOCO]
        mj_data.qpos[0:3] = ref["body_pos_w"][0, 0]
        mj_data.qpos[3:7] = ref["body_quat_w"][0, 0]

    if task.headless_random_start:
        rng = np.random.default_rng()
        dx, dy = rng.uniform(-2.0, 2.0, size=2)
        dyaw = rng.uniform(-np.pi, np.pi)
        mj_data.qpos[0] += dx
        mj_data.qpos[1] += dy
        cy, sy = np.cos(dyaw / 2), np.sin(dyaw / 2)
        q_yaw = np.array([cy, 0.0, 0.0, sy])
        w0, x0, y0, z0 = mj_data.qpos[3:7]
        mj_data.qpos[3] = q_yaw[0] * w0 - q_yaw[3] * z0
        mj_data.qpos[4] = q_yaw[0] * x0 - q_yaw[3] * y0
        mj_data.qpos[5] = q_yaw[0] * y0 + q_yaw[3] * x0
        mj_data.qpos[6] = q_yaw[0] * z0 + q_yaw[3] * w0
        logger.info(f"Random start: dx={dx:.2f}m dy={dy:.2f}m dyaw={np.degrees(dyaw):.1f}°")

    mujoco.mj_forward(mj_model, mj_data)

    mocap_yaw_rad = np.radians(task.headless_mocap_yaw_offset)
    if task.headless_random_mocap_yaw:
        mocap_yaw_rad = float(np.random.default_rng().uniform(-np.pi, np.pi))
        logger.info(f"Random mocap-IMU yaw offset: {np.degrees(mocap_yaw_rad):.1f}°")

    return {
        "config": config,
        "model": mj_model,
        "data": mj_data,
        "sim_params": sim_params,
        "metadata": metadata,
        "xml_path": xml_path,
        "ref_motion_path": ref_motion_path,
        "terrain_obj": terrain_obj,  # replayed OBJ (if terrain NPY was given), else None
        "ref_npz": ref,
        "mocap_yaw_rad": mocap_yaw_rad,
    }
