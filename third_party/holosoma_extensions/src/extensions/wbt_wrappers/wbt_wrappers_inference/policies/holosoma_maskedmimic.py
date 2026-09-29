"""Holosoma MaskedMimic Whole-Body Tracking Policy.

Deploys MaskedMimic ONNX models (2409D obs) on the G1 robot with flexible
tracking modes:

- ``three_point``: torso + left_hand + right_hand (VR headset + controllers)
- ``three_point_head``: same as three_point but auto-converts head mocap → torso
  via fixed offset (for head-mounted mocap marker + wrist controllers)
- ``three_point_pelvis``: pelvis + torso + left_hand + right_hand
- ``three_point_pelvis_only``: pelvis + left_hand + right_hand (mocap pelvis + controllers)
- ``full_body``: all 6 conditionable bodies

When ``mocap_body_name`` targets the head (e.g. a VR headset), use
``three_point_head`` mode which automatically applies the fixed head→torso
offset so the policy receives the correct torso position. Alternatively,
set ``mocap_head_to_torso`` manually with ``three_point`` mode.

Obs layout: reduced_coords(67) | target_poses(1260) | target_masks(180) |
joint_targets(870) | joint_mode(1) | prev_actions(29) | ee_masses(2) = 2409D.

For ``exp:g1-wbt-maskedmimic-force`` checkpoints the ONNX obs is 2415D — the
same 2409D layout plus a trailing 6D ``force_cmd_local`` term ([L_xyz, R_xyz] in
the pelvis/root frame, ×0.04-scaled). This layout is auto-detected from the ONNX
``obs`` input width (2409 → no force, 2415 → force), so the standard maskedmimic
path is byte-for-byte unchanged. The F^cmd term is zero by default (UniFP Eq.2
with F=0 ⇒ pure tracking); set ``HOLOSOMA_FORCE_CMD_LOCAL`` (and optionally
``HOLOSOMA_FORCE_CMD_HAND_FRAME=1`` for a ``--F_on_hand_frame`` checkpoint) to
inject a commanded contact force, exactly like ``holosoma_wbt.py``.

The normalizer is baked into the ONNX but masked-zero entries can produce
extreme normalized values. We pre-clip observations using normalizer stats
from the companion .pt checkpoint.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
from collections import defaultdict
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np
import onnxruntime
import pinocchio as pin
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.policies.wbt import PinocchioRobot, WholeBodyTrackingPolicy
from holosoma_inference.utils.math.quat import (
    matrix_from_quat,
    quat_apply,
    quat_inverse,
    quat_mul,
    quat_rotate_inverse,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from wbt_wrappers_inference.utils.normalizer import (
    NORM_CLIP as _NORM_CLIP,
    clip_normalized_obs as _shared_clip_normalized_obs,
    clip_normalized_obs_inplace as _shared_clip_inplace,
    load_normalizer_from_path,
)
from wbt_wrappers_inference.utils.obs_utils import compute_motion_alignment as _shared_compute_motion_alignment
from wbt_wrappers_inference.config_values.robot_config import (
    CONDITIONABLE_BODIES as _CONDITIONABLE_BODIES,
    G1_30BODY_COND_INDICES as _G1_30BODY_COND_INDICES,
    G1_32BODY_COND_INDICES as _G1_32BODY_COND_INDICES,
    VR_2POINT_BODIES as _VR_2POINT_BODIES,
    VR_2POINT_OFFSETS as _VR_2POINT_OFFSETS_LIST,
)

# Wrist contact-point lever in the wrist-LOCAL frame ([L, R], ~0.18 m), the SAME
# offsets vive_teleop_bridge applies. The bridge streams the CONTACT point
# (wrist_origin + R(wrist)@offset); we subtract the lever to recover the
# wrist-ORIGIN displacement so the clutch delta matches the MM wrist body pose
# (the conditionable body IS the wrist_yaw_link origin, not the contact point).
_VR_2POINT_OFFSETS = np.array(_VR_2POINT_OFFSETS_LIST, dtype=np.float32)

_REDUCED_COORDS_DIM = 29 * 2 + 9  # joint_pos + joint_vel + proj_grav + lin_vel + ang_vel = 67
_OBS_DIM = 2409
_FORCE_CMD_DIM = 6  # per-hand F^cmd [L_xyz, R_xyz] appended for exp:g1-wbt-maskedmimic-force
_FORCE_CMD_OBS_SCALE = 0.04  # = 1 / HandForceConfig.f_cmd_max (25 N); matches training ObsTermCfg.scale
_NUM_FUTURE_STEPS = 30
_EVAL_VISIBLE_STEPS = 15  # training uses U[1,30]; median=15
_CLIP_OBS = 100.0  # must match training clip_observations — clips raw obs before normalization
_INIT_INTERP_STEPS = 1000  # ~5s at 200Hz control loop
_DEFAULT_POSE_PREPEND_S = 2.0  # seconds of default-pose transition prepended to motion

_NUM_COND_BODIES = len(_CONDITIONABLE_BODIES)


def _rot_matrix_from_rpy(rpy: tuple[float, float, float]) -> np.ndarray:
    """3×3 rotation for a URDF `rpy` (extrinsic X-Y-Z: R = Rz·Ry·Rx). Numpy twin of
    force_tracking._rot_matrix_from_rpy — used for the --F_on_sensor_frame deploy gate
    (constant wrist→ft_sensor rotation). Kept local so this package has no training-package dep."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return (rz @ ry @ rx).astype(np.float32)


_TARGET_POSES_DIM = _NUM_FUTURE_STEPS * _NUM_COND_BODIES * 7  # 1260
_TARGET_MASKS_DIM = _NUM_FUTURE_STEPS * _NUM_COND_BODIES  # 180
_JOINT_TARGETS_DIM = _NUM_FUTURE_STEPS * 29  # 870

_TRACKING_MODES = {
    "three_point": {
        "body_indices": [1, 4, 5],  # torso, left_hand, right_hand
        "joint_mode": False,
    },
    "three_point_head": {
        "body_indices": [1, 4, 5],  # same as three_point; URDF head→torso offset via config
        "joint_mode": False,
    },
    "three_point_real_head": {
        "body_indices": [1, 4, 5],  # same as three_point; real head→torso offset via config
        "joint_mode": False,
    },
    "three_point_pelvis": {
        "body_indices": [0, 1, 4, 5],  # pelvis + torso + hands
        "joint_mode": False,
    },
    "three_point_pelvis_only": {
        "body_indices": [0, 4, 5],  # pelvis + hands (no torso)
        "joint_mode": False,
    },
    "two_point": {
        "body_indices": [4, 5],  # left_wrist_yaw_link, right_wrist_yaw_link (requires pelvis mocap)
        "joint_mode": False,
    },
    "five_point": {
        "body_indices": [1, 2, 3, 4, 5],  # torso + ankles + wrists (matches training hover mask)
        "joint_mode": False,
    },
    "full_body": {"body_indices": [0, 1, 2, 3, 4, 5], "joint_mode": True},
}


def _canonicalize_quat_xyzw(q):
    if q[3] < 0:
        return -q
    return q


def _clip_normalized_obs_inplace(obs, scratch, norm_mean, norm_std_eps):
    """Pre-clip in-place so the baked-in ONNX normalizer produces clipped values."""
    _shared_clip_inplace(obs, scratch, norm_mean, norm_std_eps)


class HolosomaMaskedMimicPolicy(WholeBodyTrackingPolicy):
    """MaskedMimic WBT policy for real-robot deployment with flexible tracking modes."""

    def __init__(self, config: InferenceConfig):
        _n_joints = config.robot.num_joints
        self._last_action = np.zeros(_n_joints, dtype=np.float32)
        self._action_scale = np.ones(_n_joints, dtype=np.float32)
        self._norm_mean: np.ndarray | None = None
        self._norm_std_eps: np.ndarray | None = None
        self._tracking_mode = getattr(config.task, "maskedmimic_mode", "three_point")
        if self._tracking_mode not in _TRACKING_MODES:
            raise ValueError(
                f"Unknown maskedmimic_mode={self._tracking_mode!r}. Supported: {list(_TRACKING_MODES.keys())}"
            )
        mode_cfg = _TRACKING_MODES[self._tracking_mode]
        self._visible_body_indices: list[int] = list(mode_cfg["body_indices"])  # type: ignore[call-overload]
        self._joint_mode: bool = bool(mode_cfg["joint_mode"])
        self._pinocchio_frame_ids: list[int] = []
        self._ref_quat_xyzw_0 = np.array([0, 0, 0, 1], dtype=np.float32)
        self._ref_pos_xyz_0 = np.zeros(3, dtype=np.float32)
        self._ref_quat_xyzw_t = np.array([0, 0, 0, 1], dtype=np.float32)
        self._ref_pos_xyz_t = np.zeros(3, dtype=np.float32)
        self._motion_command_0 = np.zeros(58, dtype=np.float32)
        self._motion_command_t = np.zeros(58, dtype=np.float32)

        # External motion (loaded from ref_motion_path NPZ)
        self._ext_motion = None  # dict with joint_pos, joint_vel, body_pos_w, body_quat_w
        self._ext_motion_length = 0
        self._ext_frame_idx = 0
        self._num_prepend_frames = 0  # number of prepend frames (bypass ONNX during these)
        self._ext_body_npz_indices: list[int] | None = None  # NPZ body indices for conditionable bodies
        self._prepend_stiff_active = False  # True = drive prepend from _get_manual_command with stiff KP

        # Init-interpolation state (keeps stiff KP/KD during init → motion pose transition)
        self._init_interpolating = False
        self._init_start_q = None
        self._init_interp_count = 0

        # Motion-to-mocap alignment (computed at first inference step)
        # Transforms reference trajectory from motion-data frame to mocap world frame
        self._align_rot_mat: np.ndarray | None = None  # 3x3 rotation (yaw only)
        self._align_pos_offset: np.ndarray | None = None  # 3D translation
        self._align_yaw_quat_wxyz: np.ndarray | None = None  # yaw-only quaternion for rotating orientations

        # Heading correction: robot heading vs motion frame-0 heading.
        # Captured at policy start, applied to reference FK root when no mocap alignment.
        # Matches TML/dense heading removal — makes target_poses heading-invariant.
        self._heading_correction_quat_xyzw: np.ndarray | None = None  # xyzw quat for FK root correction

        # Pre-allocated hot-path buffers
        self._obs_buf = np.zeros((1, _OBS_DIM), dtype=np.float32)
        self._clip_scratch = np.zeros((1, _OBS_DIM), dtype=np.float32)
        self._reduced_coords_buf = np.zeros(_REDUCED_COORDS_DIM, dtype=np.float32)
        self._target_poses_buf = np.zeros(_TARGET_POSES_DIM, dtype=np.float32)
        self._target_masks_buf = np.zeros(_TARGET_MASKS_DIM, dtype=np.float32)
        self._joint_targets_buf = np.zeros(_JOINT_TARGETS_DIM, dtype=np.float32)
        self._joint_mode_mask = np.array([1.0 if self._joint_mode else 0.0], dtype=np.float32)
        self._ee_masses = np.zeros(2, dtype=np.float32)  # always zero — no payload tracking on G1

        # --- exp:g1-wbt-maskedmimic-force support ---
        # Whether the loaded ONNX expects the trailing 6D force_cmd_local term
        # (auto-detected from the ONNX input width in setup_policy: 2409 → no
        # force, 2415 → force preset). Mirrors holosoma_wbt.py's force handling.
        self._force_obs_enabled = False
        # Optional eval-time F^cmd override (pelvis/root frame, RAW Newtons):
        # HOLOSOMA_FORCE_CMD_LOCAL="L_fx,L_fy,L_fz,R_fx,R_fy,R_fz". Unset → zeros
        # (pure position tracking; UniFP Eq.2 with F=0 collapses to plain tracking).
        # Training applies ObsTermCfg.scale=0.04 and the runtime does NOT, so we
        # bake 0.04 in. F^cmd is obs-only (never written to physics).
        self._force_cmd_pelvis_raw: np.ndarray | None = None  # (2,3) unscaled Newtons
        self._force_cmd_local_scaled: np.ndarray | None = None  # (6,) pelvis-frame pre-scaled
        # --F_on_hand_frame parity: a checkpoint trained with --F_on_hand_frame
        # expects force_cmd_local in each HAND (wrist_yaw) frame. The world root
        # orientation cancels (pelvis IS the articulation root), so only the live
        # wrist-relative-to-pelvis rotation matters — computed by FK on the actual
        # joints with an identity freeflyer root. F^cmd=0 ⇒ 0 (frame-invariant).
        self._force_cmd_hand_frame: bool = os.environ.get("HOLOSOMA_FORCE_CMD_HAND_FRAME", "") in ("1", "true", "True")
        # --F_on_sensor_frame parity: re-express F^cmd in each {side}_ft_sensor_link
        # frame = wrist frame post-rotated by the CONSTANT R_sensor (ft_sensor_joint
        # rpy="0 -pi/2 pi", == HandForceConfig.sensor_rpy):
        # F_sensor = R_sensor^T @ (R_rel^-1 @ F_pelvis). Mutually exclusive with the
        # hand-frame gate. F^cmd=0 ⇒ 0 (frame-invariant).
        self._force_cmd_sensor_frame: bool = os.environ.get("HOLOSOMA_FORCE_CMD_SENSOR_FRAME", "") in (
            "1",
            "true",
            "True",
        )
        if self._force_cmd_hand_frame and self._force_cmd_sensor_frame:
            raise ValueError(
                "HOLOSOMA_FORCE_CMD_HAND_FRAME and HOLOSOMA_FORCE_CMD_SENSOR_FRAME are mutually "
                "exclusive; set at most one to match the checkpoint's --F_on_{hand,sensor}_frame."
            )
        self._force_sensor_rpy: tuple[float, float, float] = (0.0, -1.5707963267948966, 3.141592653589793)
        self._force_wrist_frame_ids: list[int] | None = None
        _fc_raw = os.environ.get("HOLOSOMA_FORCE_CMD_LOCAL", "")
        if _fc_raw:
            _vals = [float(x) for x in _fc_raw.split(",")]
            if len(_vals) != 6:
                raise ValueError(
                    f"HOLOSOMA_FORCE_CMD_LOCAL must be 6 comma-separated Newtons "
                    f"[L_fx,L_fy,L_fz,R_fx,R_fy,R_fz], got {len(_vals)}: {_fc_raw!r}"
                )
            self._force_cmd_pelvis_raw = np.array(_vals, dtype=np.float32).reshape(2, 3)
            self._force_cmd_local_scaled = np.array(_vals, dtype=np.float32) * _FORCE_CMD_OBS_SCALE
        self._delta_poses = np.zeros((len(self._visible_body_indices), 3), dtype=np.float32)
        self._delta_quats = np.zeros((len(self._visible_body_indices), 4), dtype=np.float32)
        # Mocap-aligned reference body pose buffers (reused by _transform_ref_body_to_mocap)
        self._mocap_ref_pos = np.zeros((_NUM_COND_BODIES, 3), dtype=np.float32)
        self._mocap_ref_quat_xyzw = np.zeros((_NUM_COND_BODIES, 4), dtype=np.float32)
        # FK pre-allocated buffers (3 pos + 4 quat + max 29 joints = 36)
        self._fk_q_buf = np.zeros(3 + 4 + _n_joints, dtype=np.float64)
        self._fk_body_pos = np.zeros((_NUM_COND_BODIES, 3), dtype=np.float32)
        self._fk_body_quat = np.zeros((_NUM_COND_BODIES, 4), dtype=np.float32)
        # rl_inference pre-allocated buffers
        self._timestep_buf = np.zeros((1, 1), dtype=np.float32)
        self._scaled_action_buf = np.zeros((1, _n_joints), dtype=np.float32)
        self._gravity_world = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        # Diagnostic counter for first-step logging
        self._inference_step = 0
        # Action smoothing for sim2real (configured via MaskedMimicTaskConfig)
        self._action_ema_alpha = getattr(config.task, "action_ema_alpha", 1.0)
        self._action_clip_sim2real = getattr(config.task, "action_clip_sim2real", 0.0)
        self._action_ramp_steps = getattr(config.task, "action_ramp_steps", 0)
        self._action_step_count = 0
        self._smoothed_action = np.zeros(_n_joints, dtype=np.float32)

        # ------------------------------------------------------------------
        # Optional LIVE VIVE 2pt teleop (offboard run_policy → real robot, or
        # MuJoCo sim2sim validation). Parallel to holosoma_wbt's live-VIVE path
        # ([[vive-live-runpolicy-realrobot]]) but a DIFFERENT injection: MM has no
        # vr_2point obs term — the two wrist targets are WORLD-frame body poses
        # (CONDITIONABLE_BODIES idx 4=left_wrist_yaw, 5=right_wrist_yaw) that
        # _build_target_poses_and_masks turns into per-step local deltas. So we
        # inject a SLEWED RELATIVE-CLUTCH delta onto those world targets:
        #   tgt_pos = held_ref + R_root @ (tracker wrist-origin displacement)
        #   tgt_R   = [R_root · (R_live·R_live0ᵀ) · R_rootᵀ] · R_held_ref   (orn-on)
        # The delta is 0 / identity at arm ⇒ ZERO-JUMP, and the whole override is
        # gated on a live source + the two_point mode + armed, so when
        # HOLOSOMA_VIVE_WS_URL is unset the policy is byte-identical to the NPZ
        # replay path. Only meaningful for the two_point tracking mode (the only
        # mode whose visible bodies are exactly the two wrists).
        self._is_2pt_mm = self._tracking_mode == "two_point"
        # Optional teleop READY POSE (HOLOSOMA_TELEOP_READY_POSE): override the
        # frozen-clip FRAME-0 ARM joints so the held / INIT-ramp stance has the
        # forearms horizontal-and-forward (the g1_29dof_yam URDF default = arms at
        # zero), making horizontal VIVE-tracker motion map onto horizontal hand
        # motion. Parsed identically to holosoma_wbt's same-named flag; applied in
        # setup_policy after the NPZ load, gated on the two_point mode. Unset /
        # 0/false/off ⇒ byte-identical to the plain NPZ replay path.
        self._teleop_ready_pose_raw: str = os.environ.get("HOLOSOMA_TELEOP_READY_POSE", "").strip()
        self._vive_live = None
        self._vive_armed: bool = False
        # World-frame wrist TARGETS the live override last commanded (cond_idx →
        # world pos[3]) and the matching MEASURED wrist world pos, cached for the
        # interactive-sim trajectory viz. Empty when not armed.
        self._vive_last_target_w: dict[int, np.ndarray] = {}
        self._vive_last_measured_w: dict[int, np.ndarray] = {}
        # World-frame slew anchors: the last delta we actually fed (pos[2,3] world,
        # rot6d[12]=[L,R] world). Seeded 0 / identity on (re)arm so the first fed
        # frame is exactly the held reference (zero-jump), then ramps.
        self._vr_last_dpos: np.ndarray | None = None  # (6,) [Lxyz, Rxyz] world
        self._vr_last_drot6d: np.ndarray | None = None  # (12,) [L_rot6d, R_rot6d] world
        # Clutch origin snapshot (bridge pelvis-frame rep): tracker contact pos[6]
        # and wrist rot6d[12] at the instant of arm / recenter. ``None`` ⇒ not
        # anchored (no override). Doubles as the armed-anchor sentinel.
        self._vr_clutch_live0: np.ndarray | None = None  # (6,) tracker contact pos at anchor
        self._vr_clutch_orn0: np.ndarray | None = None  # (12,) tracker wrist rot6d at anchor
        # The exact live sample the override consumed THIS tick (None on stale /
        # disarmed ticks) — kept for parity with the WBT path's diagnostics.
        self._last_live_pos: np.ndarray | None = None
        self._last_live_orn: np.ndarray | None = None
        # Per-hand, per-tick slew caps (50 Hz: 0.02 m/tick ≈ 1.0 m/s). Tunable for
        # hardware vs sim — START small on hardware (see deploy memory).
        self._vive_max_pos_step = float(os.environ.get("HOLOSOMA_VIVE_MAX_POS_STEP", "0.02"))
        self._vive_max_orn_step = float(os.environ.get("HOLOSOMA_VIVE_MAX_ORN_STEP", "0.05"))
        # Orientation clutch: OFF by default (position-only teleop, wrist orn held
        # at the frozen reference). Set HOLOSOMA_VIVE_ORIENTATION=1 AND launch the
        # bridge with --orientation on to also track wrist rotation.
        self._vive_orientation = os.environ.get("HOLOSOMA_VIVE_ORIENTATION", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        _vive_ws = os.environ.get("HOLOSOMA_VIVE_WS_URL", "").strip()
        if _vive_ws:
            if not self._is_2pt_mm:
                logger.warning(
                    colored(
                        f"HOLOSOMA_VIVE_WS_URL set but maskedmimic_mode={self._tracking_mode!r} is not "
                        "'two_point' — live VIVE teleop is 2pt-only; the override will be inert.",
                        "yellow",
                    )
                )
            from wbt_wrappers_inference.utils.vive_live_source import ViveLiveSource

            _stale = float(os.environ.get("HOLOSOMA_VIVE_STALE_TIMEOUT", "0.25"))
            self._vive_live = ViveLiveSource(_vive_ws, stale_timeout=_stale)
            _orn = "POS+ORN (orn relative)" if self._vive_orientation else "POSITION-only (orn=frozen ref)"
            logger.info(
                colored(
                    f"LIVE VIVE MaskedMimic 2pt teleop ENABLED from {_vive_ws} [{_orn}] (relative/clutch; "
                    f"slew cap pos={self._vive_max_pos_step} m/tick, orn={self._vive_max_orn_step}/tick)",
                    "cyan",
                )
            )
            if self._vive_orientation:
                logger.info(
                    colored(
                        "  ⚠️ orientation clutch ON — launch the bridge with --orientation on, else orn stays frozen.",
                        "yellow",
                    )
                )

        # Debug observation logging for sim2real comparison.
        # Activated by HOLOSOMA_DEBUG_LOG env var (set to output NPZ path).
        self._debug_log: dict | None = None
        self._debug_log_path: str = ""

        def _emergency_save_debug(signum=None, frame=None):
            self._save_debug_log()
            if signum is not None:
                # Restore default handler and re-raise so the base policy class
                # can run its motor cleanup (zero commands, re-engage safety).
                # sys.exit(1) skips this cleanup → robot left in last motor state → crash.
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        signal.signal(signal.SIGINT, _emergency_save_debug)
        signal.signal(signal.SIGTERM, _emergency_save_debug)
        atexit.register(self._save_debug_log)

        super().__init__(config)

    def _save_debug_log(self):
        if self._debug_log is None or not self._debug_log_path:
            return
        save_dict = {}
        for k, v in self._debug_log.items():
            if v:
                save_dict[k] = np.array(v)
        if save_dict:
            np.savez(self._debug_log_path, **save_dict)
            logger.info(
                colored(
                    f"Debug log saved: {self._debug_log_path} ({len(save_dict.get('obs_raw', []))} frames)", "magenta"
                )
            )
        self._debug_log = None

    def _teardown_motion(self):
        """Stop motion playback and reset timestep."""
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep

    def _reset_motion_state(self):
        """Reset motion reference and tracking state to initial values."""
        self._last_action[:] = 0.0
        self._motion_command_t[:] = self._motion_command_0
        self._ref_quat_xyzw_t[:] = self._ref_quat_xyzw_0
        self._ref_pos_xyz_t[:] = self._ref_pos_xyz_0
        self._ext_frame_idx = 0
        self._inference_step = 0
        self._align_rot_mat = None
        self._align_pos_offset = None
        self._align_yaw_quat_wxyz = None
        self._heading_initialized = False
        self._heading_correction_quat_xyzw = None
        self._smoothed_action[:] = 0.0
        self._action_step_count = 0

    def _compute_motion_alignment(self, robot_state_data: np.ndarray):
        """Compute transform from motion-data frame to mocap world frame.

        Uses shared :func:`compute_motion_alignment` (same code as holosoma_wbt).
        Additionally stores yaw-only quaternion for rotating body orientations.
        """
        robot_pos = robot_state_data[0, 0:3].copy()
        robot_quat_wxyz = robot_state_data[0, 3:7]
        ref_pos_0 = self._ext_motion["body_pos_w"][0, 0]
        ref_quat_wxyz_0 = self._ext_motion["body_quat_w"][0, 0]

        self._align_rot_mat, self._align_pos_offset, yaw_offset = _shared_compute_motion_alignment(
            robot_pos, robot_quat_wxyz, ref_pos_0, ref_quat_wxyz_0
        )

        self._align_yaw_quat_wxyz = np.array(
            [np.cos(yaw_offset / 2), 0, 0, np.sin(yaw_offset / 2)], dtype=np.float32
        ).reshape(1, 4)

        logger.info(f"Motion alignment: yaw_offset={np.degrees(yaw_offset):.1f}° pos_offset={self._align_pos_offset}")

    def _transform_ref_body_to_mocap(self, frame_idx: int):
        """Get reference body positions and quaternions in mocap world frame.

        Returns (body_pos_mocap[N_cond,3], body_quat_xyzw_mocap[N_cond,4])
        for the conditionable bodies at the given NPZ frame index.
        """
        idx = min(frame_idx, self._ext_motion_length - 1)
        all_pos = self._ext_motion["body_pos_w"][idx]
        all_quat_wxyz = self._ext_motion["body_quat_w"][idx]

        buf_pos = self._mocap_ref_pos
        buf_quat = self._mocap_ref_quat_xyzw

        for i, npz_bi in enumerate(self._ext_body_npz_indices):
            pos_rotated = self._align_rot_mat @ all_pos[npz_bi]
            buf_pos[i] = (pos_rotated + self._align_pos_offset).astype(np.float32)

            q_wxyz = all_quat_wxyz[npz_bi].reshape(1, 4).astype(np.float32)
            q_rotated_wxyz = quat_mul(self._align_yaw_quat_wxyz, q_wxyz).flatten()
            buf_quat[i] = wxyz_to_xyzw(q_rotated_wxyz.reshape(1, 4)).flatten()

        return buf_pos, buf_quat

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------

    def setup_policy(self, model_path):
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]
        self._onnx_out_idx = {name: i for i, name in enumerate(self.onnx_output_names)}

        # Auto-detect the force-tracking obs layout from the ONNX "obs" input
        # width: 2409 → standard maskedmimic, 2415 → maskedmimic-force (trailing
        # 6D force_cmd_local). Resize the obs scratch buffers accordingly so the
        # standard policy is byte-for-byte unchanged.
        self._detect_force_obs_layout()

        metadata = dict(self.onnx_policy_session.get_modelmeta().custom_metadata_map)

        action_scale_raw = json.loads(metadata["action_scale"])
        self._action_scale = np.array(action_scale_raw, dtype=np.float32)
        if self._action_scale.size == 1:
            self._action_scale = np.full(self.num_dofs, self._action_scale.item(), dtype=np.float32)

        robot_urdf = metadata.get("robot_urdf", "")
        if robot_urdf:
            try:
                robot_urdf = json.loads(robot_urdf)
            except json.JSONDecodeError:
                pass
        if not robot_urdf:
            urdf_path = getattr(self.config.task, "robot_urdf_path", "")
            if urdf_path and Path(urdf_path).is_file():
                robot_urdf = Path(urdf_path).read_text()
        if robot_urdf:
            self.pinocchio_robot = PinocchioRobot(self.config.robot, robot_urdf)
            self._setup_pinocchio_frame_ids()

        self._load_normalizer(model_path)

        # Read default_dof_angles from ONNX experiment_config (matches training standing pose).
        # Must happen before _prepend_default_pose_transition which uses default_dof_angles.
        exp_config = json.loads(metadata.get("experiment_config", "{}"))
        default_angles_dict = exp_config.get("robot", {}).get("init_state", {}).get("default_joint_angles", {})
        dof_names = json.loads(metadata.get("dof_names", "[]"))
        if default_angles_dict and dof_names:
            self.default_dof_angles = np.array(
                [default_angles_dict.get(name, 0.0) for name in dof_names], dtype=np.float64
            )
            logger.info(f"default_dof_angles from ONNX (non-zero: {np.count_nonzero(self.default_dof_angles)})")
        else:
            self.default_dof_angles = np.array(self.robot_config.stiff_startup_pos, dtype=np.float64)
            logger.warning("No default_joint_angles in ONNX metadata, using stiff_startup_pos")

        # Load external reference motion (required)
        ref_motion_path = getattr(self.config.task, "ref_motion_path", "")
        if not ref_motion_path:
            raise ValueError("ref_motion_path is required — external NPZ motion must be provided")
        self._load_external_motion(ref_motion_path)

        # Optional teleop READY POSE: override the frozen-clip frame-0 ARM joints
        # (+ the matching conditionable wrist world poses) so the held / INIT
        # stance is forearm-horizontal. 2pt only; no-op when the flag is unset.
        if self._teleop_ready_pose_raw and self._is_2pt_mm:
            self._apply_teleop_ready_pose()
        elif self._teleop_ready_pose_raw:
            logger.warning(
                colored(
                    f"HOLOSOMA_TELEOP_READY_POSE set but maskedmimic_mode={self._tracking_mode!r} is not "
                    "'two_point' — ready-pose override is 2pt-only; ignoring.",
                    "yellow",
                )
            )

        # Optionally prepend default-pose transition (for models trained with
        # default_pose_transition_strategy='interpolation').
        if hasattr(self, "pinocchio_robot") and getattr(self.config.task, "prepend_default_pose_transition", False):
            duration = getattr(self.config.task, "prepend_duration_s", _DEFAULT_POSE_PREPEND_S)
            self._prepend_default_pose_transition(duration_s=duration)

        self.onnx_kp = np.array(json.loads(metadata["kp"]), dtype=np.float32) if "kp" in metadata else None
        self.onnx_kd = np.array(json.loads(metadata["kd"]), dtype=np.float32) if "kd" in metadata else None
        if self.onnx_kp is not None and self.onnx_kd is not None:
            self.robot_config = _replace(
                self.robot_config, motor_kp=tuple(self.onnx_kp.tolist()), motor_kd=tuple(self.onnx_kd.tolist())
            )
            self.interface.update_config(self.robot_config)
            logger.info(colored("MaskedMimic: using ONNX KP/KD (training gains) instead of config", "green"))
        else:
            # Fallback: use config motor_kp/kd for KP ramp
            self.onnx_kp = np.array(self.robot_config.motor_kp, dtype=np.float32)
            self.onnx_kd = np.array(self.robot_config.motor_kd, dtype=np.float32)
            logger.warning("Using KP/KD from config (no ONNX metadata)")

        # Initial motion reference from external NPZ
        joint_pos_0, joint_vel_0, ref_pos_0, ref_quat_xyzw_0 = self._get_ext_motion_at_frame(0)
        self._motion_command_0[: self.num_dofs] = joint_pos_0
        self._motion_command_0[self.num_dofs :] = joint_vel_0
        self._ref_quat_xyzw_0[:] = ref_quat_xyzw_0
        self._ref_pos_xyz_0[:] = ref_pos_0

        self._motion_command_t[:] = self._motion_command_0
        # Stable reshaped views (not re-created per step)
        self.motion_command_t = self._motion_command_t.reshape(1, -1)
        self.motion_command_0 = self._motion_command_0.reshape(1, -1)

        self._ref_quat_xyzw_t[:] = self._ref_quat_xyzw_0
        self.ref_quat_xyzw_t = self._ref_quat_xyzw_t.reshape(1, 4)
        self.ref_quat_xyzw_0 = self._ref_quat_xyzw_0.reshape(1, 4)

        self._ref_pos_xyz_t[:] = self._ref_pos_xyz_0

        self._init_q_target = joint_pos_0.copy()
        self.per_joint_policy_action_scale = self._action_scale.reshape(1, -1)

        def policy_act(_input_feed):
            raise RuntimeError("HolosomaMaskedMimic uses custom rl_inference.")

        self.policy = policy_act

        if self._norm_mean is None:
            logger.error(
                "NORMALIZER NOT LOADED — MaskedMimic REQUIRES the .pt file alongside "
                "the .onnx for pre-clipping. Without it, the policy WILL produce garbage "
                "actions and the robot will fall immediately. Place the .pt file at: "
                f"{os.path.splitext(model_path)[0]}.pt"
            )

        logger.info(
            f"HolosomaMaskedMimic: mode={self._tracking_mode}, "
            f"visible_bodies={[_CONDITIONABLE_BODIES[i] for i in self._visible_body_indices]}, "
            f"normalizer={'loaded' if self._norm_mean is not None else 'MISSING'}, "
            f"action_scale_range=[{self._action_scale.min():.4f}, {self._action_scale.max():.4f}], "
            f"ext_motion_frames={self._ext_motion_length}, "
            f"future_frames={'yes' if self._ext_body_npz_indices is not None else 'no'}, "
            f"action_ema_alpha={self._action_ema_alpha}, "
            f"action_clip_sim2real={self._action_clip_sim2real}, "
            f"action_ramp_steps={self._action_ramp_steps}"
        )

    def _setup_pinocchio_frame_ids(self):
        model = self.pinocchio_robot.robot_model
        for body_name in _CONDITIONABLE_BODIES:
            fid = model.getFrameId(body_name)
            if fid >= model.nframes:
                logger.warning(f"Body '{body_name}' not found in URDF")
            self._pinocchio_frame_ids.append(fid)

    def _load_normalizer(self, onnx_path: str):
        """Load normalizer mean/std — uses shared :func:`load_normalizer_from_path`."""
        self._norm_mean, self._norm_std_eps = load_normalizer_from_path(onnx_path)
        if self._norm_mean is not None:
            logger.info(f"Loaded normalizer stats (clip={_NORM_CLIP})")
        else:
            logger.warning(
                "MaskedMimic REQUIRES normalizer stats. Without it, masked-zero entries produce garbage actions."
            )

    def _load_external_motion(self, npz_path: str):
        """Load an external NPZ reference motion (same format as dense/TML policies).

        NPZ fields: joint_pos [T,36], joint_vel [T,35],
                     body_pos_w [T,N_bodies,3], body_quat_w [T,N_bodies,4] (wxyz),
                     body_names [N_bodies] (optional).
        """
        data = np.load(npz_path)
        self._ext_motion = {
            "joint_pos": data["joint_pos"].astype(np.float32),
            "joint_vel": data["joint_vel"].astype(np.float32),
            "body_pos_w": data["body_pos_w"].astype(np.float32),
            "body_quat_w": data["body_quat_w"].astype(np.float32),
        }
        self._ext_motion_length = self._ext_motion["joint_pos"].shape[0]
        self._ext_frame_idx = 0

        # Map conditionable body names → NPZ body indices for future-frame lookups.
        # Explicit body_names key takes priority; otherwise infer from body count.
        self._ext_body_npz_indices = None
        n_bodies = self._ext_motion["body_pos_w"].shape[1] if "body_pos_w" in data else 0
        if "body_names" in data:
            bnames = [b.decode() if isinstance(b, bytes) else str(b) for b in data["body_names"].tolist()]
            try:
                self._ext_body_npz_indices = [bnames.index(bn) for bn in _CONDITIONABLE_BODIES]
                logger.info(
                    f"NPZ body index mapping (explicit): {dict(zip(_CONDITIONABLE_BODIES, self._ext_body_npz_indices))}"
                )
            except ValueError as e:
                logger.warning(f"Could not map all conditionable bodies in NPZ: {e}")
                self._ext_body_npz_indices = None
        elif n_bodies == 32:
            self._ext_body_npz_indices = list(_G1_32BODY_COND_INDICES)
            logger.info(
                f"NPZ body index mapping (default 32-body): {dict(zip(_CONDITIONABLE_BODIES, self._ext_body_npz_indices))}"
            )
        elif n_bodies == 30:
            self._ext_body_npz_indices = list(_G1_30BODY_COND_INDICES)
            logger.info(
                f"NPZ body index mapping (default 30-body, r_wrist proxied): {dict(zip(_CONDITIONABLE_BODIES, self._ext_body_npz_indices))}"
            )
        elif n_bodies > 0:
            logger.warning(
                f"NPZ has {n_bodies} bodies but no body_names — cannot infer mapping, FK fallback will be used"
            )

        logger.info(f"External motion loaded: {self._ext_motion_length} frames from {npz_path}")

    def _prepend_default_pose_transition(self, duration_s: float = 2.0):
        raise NotImplementedError("deprecated — use run_policy.py for headless sim2sim eval")
        """Prepend transition frames from default standing pose to motion start.

        Creates interpolated frames where joints go from default_dof_angles to the
        motion's initial joint positions. Body poses are computed via Pinocchio FK
        for conditionable bodies so that target_poses deltas are correct during the
        transition. Non-conditionable bodies copy frame 0's values.

        This ensures the reference motion starts at the robot's actual standing pose,
        so target_poses deltas are near-zero at frame 0 and ramp up smoothly.
        """
        rl_rate = getattr(self.config.task, "rl_rate", 50)
        num_frames = int(duration_s * rl_rate)
        if num_frames <= 0:
            return

        motion = self._ext_motion
        jp0 = motion["joint_pos"][0]  # [36] = root_pos(3) + root_quat(4) + joints(29)
        jv0 = motion["joint_vel"][0]  # [35] = root_lin(3) + root_ang(3) + joints(29)
        bp0 = motion["body_pos_w"][0]  # [N_bodies, 3]
        bq0 = motion["body_quat_w"][0]  # [N_bodies, 4] wxyz

        # Default joint positions for the prepend start
        has_root_prefix = jp0.shape[0] > self.num_dofs
        default_jp = np.zeros_like(jp0)
        if has_root_prefix:
            default_jp[7:] = self.default_dof_angles.astype(np.float32)
            default_jp[:3] = bp0[0, :3]  # root pos from frame 0 pelvis
            default_jp[3:7] = bq0[0]  # root quat from frame 0 pelvis (wxyz)
        else:
            default_jp[:] = self.default_dof_angles.astype(np.float32)

        motion_joints_0 = jp0[7:] if has_root_prefix else jp0

        alphas = np.linspace(0.0, 1.0, num_frames, endpoint=False, dtype=np.float32)[:, np.newaxis]
        prepend_jp = (1 - alphas) * default_jp + alphas * jp0
        prepend_jv = np.zeros((num_frames, jv0.shape[0]), dtype=np.float32)
        # Tile frame-0 body poses; conditionable bodies get FK-overwritten below
        prepend_bp = np.tile(bp0[np.newaxis], (num_frames, 1, 1))
        prepend_bq = np.tile(bq0[np.newaxis], (num_frames, 1, 1))

        if self._ext_body_npz_indices is not None:
            root_pos = bp0[0, :3].astype(np.float64)
            root_quat_wxyz = bq0[0]
            root_quat_xyzw = wxyz_to_xyzw(root_quat_wxyz.reshape(1, 4)).flatten().astype(np.float64)

            for fi in range(num_frames):
                alpha = alphas[fi, 0]
                interp_joints = ((1 - alpha) * self.default_dof_angles + alpha * motion_joints_0).astype(np.float32)
                fk_pos, fk_quat_xyzw = self._run_fk(interp_joints, root_pos, root_quat_xyzw)
                fk_pos = fk_pos.copy()
                fk_quat_xyzw = fk_quat_xyzw.copy()

                for ci, npz_bi in enumerate(self._ext_body_npz_indices):
                    prepend_bp[fi, npz_bi] = fk_pos[ci]
                    # FK returns xyzw, convert to wxyz for NPZ storage
                    fk_wxyz = xyzw_to_wxyz(fk_quat_xyzw[ci].reshape(1, 4)).flatten()
                    prepend_bq[fi, npz_bi] = fk_wxyz

        motion["joint_pos"] = np.concatenate([prepend_jp, motion["joint_pos"]], axis=0)
        motion["joint_vel"] = np.concatenate([prepend_jv, motion["joint_vel"]], axis=0)
        motion["body_pos_w"] = np.concatenate([prepend_bp, motion["body_pos_w"]], axis=0)
        motion["body_quat_w"] = np.concatenate([prepend_bq, motion["body_quat_w"]], axis=0)

        self._num_prepend_frames = num_frames
        self._ext_motion_length = motion["joint_pos"].shape[0]
        logger.info(
            colored(
                f"Prepended {num_frames} default-pose transition frames "
                f"({duration_s}s at {rl_rate}Hz) to MaskedMimic motion",
                "cyan",
            )
        )

    def _apply_teleop_ready_pose(self) -> None:
        """Override the ext-motion FRAME-0 ARM joints with a teleop ready pose.

        Mirrors ``HolosomaWBTPolicy._apply_teleop_ready_pose`` but for the MM obs:
        the held / INIT-ramp stance comes from frame 0 of the NPZ, frozen during
        teleop (``motion_clip_progressing=False``). For MM the held WRIST TARGET is
        the conditionable body's WORLD pose — read from ``body_pos_w[0]`` /
        ``body_quat_w[0]`` on the aligned-NPZ path and from frame-0 joint FK on the
        fallback path — so after rewriting the frame-0 arm joints we ALSO recompute
        the conditionable bodies' frame-0 world poses via FK, keeping both paths
        consistent with the new stance. Frame 0 ONLY (frames 1+ untouched so a
        later clip-play tracks the real choreography). 2pt-only.

        Accepted values (case-insensitive), identical to the WBT flag:
          1|true|zero|both   → BOTH arms to the URDF default (all arm DOFs = 0)
          right | left       → that side only
          0|false|no|off     → explicit no-op (disabled; clean, no warning)
          <14 floats>        → explicit [L_arm x7, R_arm x7] radians, SDK arm order

        The URDF (g1_29dof_yam) default arm pose (zeros) is forearm-horizontal /
        forward with the upper arm hanging down (the G1 elbow link bakes in ~90°).
        """
        raw = self._teleop_ready_pose_raw.lower()
        if raw in ("0", "false", "no", "off"):
            logger.info(colored(f"Teleop READY POSE disabled (HOLOSOMA_TELEOP_READY_POSE={raw!r}).", "cyan"))
            return

        dof_names = list(getattr(self.config.robot, "dof_names", []))
        if len(dof_names) < 29:
            logger.warning(
                colored(
                    "HOLOSOMA_TELEOP_READY_POSE set but robot dof_names < 29 — ignoring ready-pose override.", "red"
                )
            )
            return
        left_arm = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]
        right_arm = [n.replace("left_", "right_") for n in left_arm]
        try:
            left_idx = [dof_names.index(n) for n in left_arm]
            right_idx = [dof_names.index(n) for n in right_arm]
        except ValueError as exc:
            logger.warning(
                colored(f"HOLOSOMA_TELEOP_READY_POSE: arm joint not found in dof_names ({exc}); skip.", "red")
            )
            return

        left_vals: np.ndarray | None = None
        right_vals: np.ndarray | None = None
        explicit = False
        if raw in ("1", "true", "zero", "both"):
            left_vals = np.zeros(7, dtype=np.float32)
            right_vals = np.zeros(7, dtype=np.float32)
            label = "both arms → URDF default (forearms horizontal)"
        elif raw == "right":
            right_vals = np.zeros(7, dtype=np.float32)
            label = "right arm → URDF default (forearm horizontal)"
        elif raw == "left":
            left_vals = np.zeros(7, dtype=np.float32)
            label = "left arm → URDF default (forearm horizontal)"
        else:
            try:
                floats = [float(x) for x in self._teleop_ready_pose_raw.split(",")]
            except ValueError:
                logger.warning(
                    colored(f"HOLOSOMA_TELEOP_READY_POSE={self._teleop_ready_pose_raw!r} unparseable; skip.", "red")
                )
                return
            if len(floats) != 14:
                logger.warning(
                    colored(
                        f"HOLOSOMA_TELEOP_READY_POSE expects 'zero'/'right'/'left' or 14 floats "
                        f"[L_arm x7, R_arm x7], got {len(floats)}; skip.",
                        "red",
                    )
                )
                return
            left_vals = np.array(floats[:7], dtype=np.float32)
            right_vals = np.array(floats[7:], dtype=np.float32)
            label = "explicit arm angles"
            explicit = True

        model = getattr(getattr(self, "pinocchio_robot", None), "robot_model", None)
        if explicit and model is None:
            logger.warning(
                colored(
                    "HOLOSOMA_TELEOP_READY_POSE explicit angles requested but no Pinocchio model is loaded — "
                    "CANNOT clamp to joint limits; applying as-is. Verify your angles are in-range.",
                    "red",
                )
            )

        def _clamp(names: list[str], vals: np.ndarray) -> np.ndarray:
            if model is None:
                return vals
            out = vals.copy()
            for i, n in enumerate(names):
                if model.existJointName(n):
                    qi = model.idx_qs[model.getJointId(n)]
                    lo, hi = float(model.lowerPositionLimit[qi]), float(model.upperPositionLimit[qi])
                    out[i] = float(np.clip(out[i], lo, hi))
            return out

        assert self._ext_motion is not None  # loaded in setup_policy before this runs
        jp = self._ext_motion["joint_pos"]  # [T, 36] (root+29) or [T, 29]
        root_off = 7 if jp.shape[1] > 29 else 0
        if left_vals is not None:
            left_vals = _clamp(left_arm, left_vals)
            jp[0, [root_off + i for i in left_idx]] = left_vals
        if right_vals is not None:
            right_vals = _clamp(right_arm, right_vals)
            jp[0, [root_off + i for i in right_idx]] = right_vals
        jv = self._ext_motion["joint_vel"]
        vroot_off = 6 if jv.shape[1] > 29 else 0
        zero_idx = (left_idx if left_vals is not None else []) + (right_idx if right_vals is not None else [])
        if zero_idx:
            jv[0, [vroot_off + i for i in zero_idx]] = 0.0

        # Recompute the conditionable bodies' FRAME-0 world poses so the aligned-NPZ
        # held wrist target (read from body_pos_w[0]/body_quat_w[0]) matches the new
        # arm joints. Without this the sim's held wrist target would still reflect
        # the ORIGINAL clip arms even though the robot ramps to the ready-pose arms.
        if self._ext_body_npz_indices is not None and hasattr(self, "pinocchio_robot"):
            new_joints = jp[0, root_off:] if root_off else jp[0]
            new_joints = new_joints.astype(np.float32)
            root_pos = self._ext_motion["body_pos_w"][0, 0].astype(np.float64)
            root_quat_wxyz = self._ext_motion["body_quat_w"][0, 0]
            root_quat_xyzw = wxyz_to_xyzw(root_quat_wxyz.reshape(1, 4)).flatten().astype(np.float64)
            fk_pos, fk_quat_xyzw = self._run_fk(new_joints, root_pos, root_quat_xyzw)
            fk_pos = fk_pos.copy()
            fk_quat_xyzw = fk_quat_xyzw.copy()
            for ci, npz_bi in enumerate(self._ext_body_npz_indices):
                self._ext_motion["body_pos_w"][0, npz_bi] = fk_pos[ci].astype(np.float32)
                self._ext_motion["body_quat_w"][0, npz_bi] = (
                    xyzw_to_wxyz(fk_quat_xyzw[ci].reshape(1, 4)).flatten().astype(np.float32)
                )

        logger.info(
            colored(f"Teleop READY POSE: {label} (HOLOSOMA_TELEOP_READY_POSE={self._teleop_ready_pose_raw!r})", "cyan")
        )
        if explicit:
            resolved = []
            if left_vals is not None:
                resolved += [f"{n}={v:+.3f}" for n, v in zip(left_arm, left_vals)]
            if right_vals is not None:
                resolved += [f"{n}={v:+.3f}" for n, v in zip(right_arm, right_vals)]
            logger.info(colored("  resolved (post-clamp, rad): " + ", ".join(resolved), "cyan"))

    def _get_ext_motion_at_frame(self, frame_idx: int):
        """Get motion reference data from external NPZ at given frame."""
        idx = min(frame_idx, self._ext_motion_length - 1)
        raw_pos = self._ext_motion["joint_pos"][idx]
        raw_vel = self._ext_motion["joint_vel"][idx]
        # NPZ joint_pos is [root_xyz(3) + root_quat(4) + joints(N)] = 36 for 29-DOF
        # NPZ joint_vel is [root_lin(3) + root_ang(3) + joints(N)] = 35 for 29-DOF
        joint_pos = raw_pos[7:] if raw_pos.shape[0] > self.num_dofs else raw_pos
        joint_vel = raw_vel[6:] if raw_vel.shape[0] > self.num_dofs else raw_vel
        root_pos = self._ext_motion["body_pos_w"][idx, 0]  # pelvis = body 0
        root_quat_wxyz = self._ext_motion["body_quat_w"][idx, 0]  # wxyz
        root_quat_xyzw = wxyz_to_xyzw(root_quat_wxyz.reshape(1, 4)).flatten()
        return joint_pos, joint_vel, root_pos, root_quat_xyzw

    # -----------------------------------------------------------------
    # Pinocchio FK
    # -----------------------------------------------------------------

    def _run_fk(
        self, joint_pos_sdk: np.ndarray, root_pos: np.ndarray | None = None, root_quat_xyzw: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run FK, return [6,3] world positions and [6,4] xyzw quaternions.

        Args:
            joint_pos_sdk: Joint positions in SDK ordering.
            root_pos: Root position (3,). Defaults to self._ref_pos_xyz_t.
            root_quat_xyzw: Root quaternion in xyzw (4,). Defaults to self._ref_quat_xyzw_t.
        """
        model = self.pinocchio_robot.robot_model
        data = self.pinocchio_robot.robot_data

        joint_pos_pin = joint_pos_sdk[self.pinocchio_robot.real2pinocchio_index]
        q = self._fk_q_buf
        q[:3] = root_pos if root_pos is not None else self._ref_pos_xyz_t
        q[3:7] = root_quat_xyzw if root_quat_xyzw is not None else self._ref_quat_xyzw_t
        q[7:] = joint_pos_pin

        pin.framesForwardKinematics(model, data, q)

        body_pos_w = self._fk_body_pos
        body_quat_xyzw = self._fk_body_quat
        for i, fid in enumerate(self._pinocchio_frame_ids):
            frame_pose = data.oMf[fid]
            body_pos_w[i] = frame_pose.translation
            body_quat_xyzw[i] = pin.Quaternion(frame_pose.rotation).coeffs()
        return body_pos_w, body_quat_xyzw

    # -----------------------------------------------------------------
    # Live VIVE clutch helpers (shared math with holosoma_wbt, MM-local copies)
    # -----------------------------------------------------------------

    @staticmethod
    def _slew_vec(prev: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
        """Move ``prev`` toward ``target`` by at most ``max_step`` (L2) this tick.

        ``max_step <= 0`` disables the cap. Returns a fresh float32 array.
        """
        delta = target - prev
        dist = float(np.linalg.norm(delta))
        if max_step <= 0.0 or dist <= max_step or dist == 0.0:
            return target.astype(np.float32).copy()
        return (prev + delta * (max_step / dist)).astype(np.float32)

    @staticmethod
    def _rot6d_to_R(r6: np.ndarray) -> np.ndarray:
        """Gram-Schmidt rebuild of a 3x3 rotation from a 6D rep (first two columns)."""
        m = r6.reshape(3, 2).astype(np.float64)
        b0 = m[:, 0]
        n0 = np.linalg.norm(b0)
        c0 = b0 / n0 if n0 > 1e-8 else np.array([1.0, 0.0, 0.0])
        b1 = m[:, 1]
        c1 = b1 - (c0 @ b1) * c0
        n1 = np.linalg.norm(c1)
        c1 = c1 / n1 if n1 > 1e-8 else np.array([0.0, 1.0, 0.0])
        c2 = np.cross(c0, c1)
        return np.stack([c0, c1, c2], axis=1)

    @staticmethod
    def _R_to_rot6d(R: np.ndarray) -> np.ndarray:
        """First two columns of R, flattened — the 6D rep (inverse of _rot6d_to_R)."""
        return R[:, :2].reshape(-1).astype(np.float32)

    def _wrist_cond_to_hand(self, cond_idx: int) -> int:
        """Map a conditionable-body index (4=left_wrist, 5=right_wrist) to the
        bridge hand index (0=L, 1=R). Built from VR_2POINT_BODIES so it stays in
        sync with the bridge's [L, R] ordering."""
        body_name = _CONDITIONABLE_BODIES[cond_idx]
        return _VR_2POINT_BODIES.index(body_name)

    def _compute_vive_world_delta(self, robot_root_quat_wxyz: np.ndarray):
        """Per-hand world-frame clutch delta from the live trackers.

        Recovers each wrist's ORIGIN displacement in the pelvis frame (the bridge
        streams the CONTACT point ``wrist_origin + R(wrist)@offset``; we subtract
        the lever so the delta lands on the wrist_yaw_link origin = the MM
        conditionable body), then rotates it into the WORLD frame via the live
        robot root. Returns ``(dpos[2,3] world, dR[list of 2 (3,3)] world)`` or
        ``None`` when there is no live frame / no clutch anchor. Caches the live
        sample consumed for diagnostics. delta is 0 / identity at the anchor
        (live==live0) ⇒ zero-jump arm.

        The lever subtraction uses the bridge's streamed wrist rot6d, which is
        ALWAYS present regardless of whether the bridge runs ``--orientation
        on/off`` — so the recovered ORIGIN displacement is correct even if the
        bridge swings the contact lever while we track POSITION-only (the
        ``_vive_orientation`` flag governs only whether we ALSO rotate the wrist
        TARGET, not the position math). When the bridge holds orn constant the
        lever term is identical at now/anchor and cancels, so this reduces to the
        plain contact-point displacement — no behavior change in that common case.
        """
        assert self._vive_live is not None  # caller guards; narrows for mypy
        live_pos, live_orn = self._vive_live.latest()
        self._last_live_pos = live_pos
        self._last_live_orn = live_orn
        if live_pos is None or self._vr_clutch_live0 is None:
            return None
        R_root = matrix_from_quat(robot_root_quat_wxyz.reshape(1, 4))[0].astype(np.float64)  # body→world
        dpos = np.zeros((2, 3), dtype=np.float32)
        dR = [np.eye(3, dtype=np.float64), np.eye(3, dtype=np.float64)]
        clutch_orn0 = self._vr_clutch_orn0
        clutch_live0 = self._vr_clutch_live0
        # Whether we can subtract the rotating lever to recover the true wrist
        # ORIGIN: needs both the live wrist rot6d AND the anchor rot6d. The anchor
        # rot6d is snapshotted whenever the bridge sent orn (see
        # _snapshot_clutch_anchor), independent of the orn-CLUTCH flag.
        lever_recover = live_orn is not None and clutch_orn0 is not None
        # Whether we rotate the wrist TARGET with the trackers (orn clutch).
        orn_active = bool(self._vive_orientation and lever_recover)
        for h in range(2):
            pos_now = live_pos[h * 3 : h * 3 + 3].astype(np.float64)
            pos_0 = clutch_live0[h * 3 : h * 3 + 3].astype(np.float64)
            offset = _VR_2POINT_OFFSETS[h].astype(np.float64)
            if lever_recover:
                assert live_orn is not None and clutch_orn0 is not None
                R_now = self._rot6d_to_R(live_orn[h * 6 : h * 6 + 6])
                R_0 = self._rot6d_to_R(clutch_orn0[h * 6 : h * 6 + 6])
                # wrist ORIGIN = contact - R(wrist)@lever; displacement since anchor.
                # Correct whether or not the bridge swings the lever (orn on/off).
                disp_pelvis = (pos_now - R_now @ offset) - (pos_0 - R_0 @ offset)
                if orn_active:
                    # rotation delta since anchor, conjugated into the world frame
                    dR[h] = R_root @ (R_now @ R_0.T) @ R_root.T
            else:
                # No orn stream at all ⇒ the bridge cannot be swinging the lever
                # either, so the lever cancels and the plain contact displacement
                # is the origin displacement. Orientation delta stays identity.
                disp_pelvis = pos_now - pos_0
            dpos[h] = (R_root @ disp_pelvis).astype(np.float32)
        return dpos, dR

    @staticmethod
    def _identity_drot6d() -> np.ndarray:
        """Two stacked identity-rotation rot6d reps ([L, R]) = the zero orn delta."""
        eye6 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # cols 0,1 of I3
        return np.concatenate([eye6, eye6])

    def _vive_wrist_targets(self, robot_state_data: np.ndarray, held_bpos_w, held_bquat_xyzw):
        """Build the live-VIVE override world targets for the visible wrists.

        Returns ``(override_pos {cond_idx: pos[3]}, override_quat_xyzw {cond_idx:
        quat[4]})`` for the two wrist bodies = the FROZEN-frame held pose
        (``held_*`` args, STATIC across the future lookahead) plus a slewed
        world-frame clutch delta, or ``({}, {})`` when not armed (clip tracks
        normally → byte-identical when off).

        When ARMED the wrists are ALWAYS overridden — even on a stale frame we hold
        the last fed delta (never let the clip's moving future lookahead drift the
        hands). The delta is 0 / identity at (re)arm, and SLEW-limited, so
        first-connect / reconnect / stale→fresh RAMP instead of stepping.
        """
        empty: tuple[dict, dict] = ({}, {})
        # getattr-guarded so the obs builder is robust to minimal/partial
        # construction (e.g. the obs-parity test harness) — the override is opt-in
        # and only fires with a live source + two_point mode + armed.
        if not (
            getattr(self, "_vive_live", None) is not None
            and getattr(self, "_is_2pt_mm", False)
            and getattr(self, "_vive_armed", False)
        ):
            return empty
        robot_root_quat_wxyz = robot_state_data[0, 3:7]
        delta = self._compute_vive_world_delta(robot_root_quat_wxyz)
        if delta is not None:
            dpos, dR = delta  # dpos (2,3) world, dR list of 2 (3,3) world
            target_dpos = dpos.reshape(6).astype(np.float32)
            target_drot6d = np.concatenate([self._R_to_rot6d(dR[0]), self._R_to_rot6d(dR[1])]).astype(np.float32)
            if self._vr_last_dpos is None or self._vr_last_drot6d is None:
                # First fed frame after (re)arm: ramp FROM zero delta / identity orn
                # so the hands glide onto the operator's offset, never jump.
                self._vr_last_dpos = np.zeros(6, dtype=np.float32)
                self._vr_last_drot6d = self._identity_drot6d()
            prev_dpos: np.ndarray = self._vr_last_dpos
            prev_drot6d: np.ndarray = self._vr_last_drot6d
            dpos_slew = self._slew_vec(prev_dpos, target_dpos, self._vive_max_pos_step)
            drot6d_slew = self._slew_vec(prev_drot6d, target_drot6d, self._vive_max_orn_step)
            # The rot6d slew linearly interpolates off the SO(3) manifold; re-project
            # each hand before use (only matters when orn actually moves).
            if self._vive_orientation and self._vr_clutch_orn0 is not None:
                for h in range(2):
                    s = slice(h * 6, h * 6 + 6)
                    drot6d_slew[s] = self._R_to_rot6d(self._rot6d_to_R(drot6d_slew[s]))
            self._vr_last_dpos = dpos_slew.copy()
            self._vr_last_drot6d = drot6d_slew.copy()
        else:
            # Stale / no anchor while ARMED: HOLD the last fed delta (do NOT revert
            # to the clip future — that would drift the hands). Nothing fed yet ⇒
            # zero delta = the held frozen pose (a static hand hold).
            dpos_slew = self._vr_last_dpos if self._vr_last_dpos is not None else np.zeros(6, dtype=np.float32)
            drot6d_slew = self._vr_last_drot6d if self._vr_last_drot6d is not None else self._identity_drot6d()

        override_pos: dict[int, np.ndarray] = {}
        override_quat_xyzw: dict[int, np.ndarray] = {}
        for bi in self._visible_body_indices:
            if _CONDITIONABLE_BODIES[bi] not in _VR_2POINT_BODIES:
                continue  # only the two wrists are tracker-driven
            h = self._wrist_cond_to_hand(bi)
            held_pos = held_bpos_w[bi].astype(np.float64)
            held_R = matrix_from_quat(xyzw_to_wxyz(held_bquat_xyzw[bi].reshape(1, 4)))[0].astype(np.float64)
            tgt_pos = held_pos + dpos_slew[h * 3 : h * 3 + 3].astype(np.float64)
            if self._vive_orientation and self._vr_clutch_orn0 is not None:
                tgt_R = self._rot6d_to_R(drot6d_slew[h * 6 : h * 6 + 6]) @ held_R
            else:
                tgt_R = held_R
            override_pos[bi] = tgt_pos.astype(np.float32)
            override_quat_xyzw[bi] = pin.Quaternion(tgt_R).coeffs().astype(np.float32)
        return override_pos, override_quat_xyzw

    # -----------------------------------------------------------------
    # Observation building
    # -----------------------------------------------------------------

    def _build_reduced_coords(self, robot_state_data: np.ndarray) -> np.ndarray:
        buf = self._reduced_coords_buf
        dof_pos = robot_state_data[0, 7 : 7 + self.num_dofs]
        dof_vel = robot_state_data[0, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs]
        base_ang_vel = robot_state_data[0, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]

        root_quat_wxyz = robot_state_data[0, 3:7].reshape(1, 4)
        proj_grav = quat_rotate_inverse(root_quat_wxyz, self._gravity_world).flatten()

        base_lin_vel_w = robot_state_data[0, 7 + self.num_dofs : 7 + self.num_dofs + 3].reshape(1, 3)
        base_lin_vel_b = quat_rotate_inverse(root_quat_wxyz, base_lin_vel_w).flatten()

        o = 0
        buf[o : o + self.num_dofs] = dof_pos - self.default_dof_angles
        o += self.num_dofs
        buf[o : o + self.num_dofs] = dof_vel
        o += self.num_dofs
        buf[o : o + 3] = proj_grav
        o += 3
        buf[o : o + 3] = base_lin_vel_b
        o += 3
        buf[o : o + 3] = base_ang_vel
        # Per-term clip matching training clip_observations (applied before normalization)
        np.clip(buf, -_CLIP_OBS, _CLIP_OBS, out=buf)
        return buf

    def _build_target_poses_and_masks(self, robot_state_data: np.ndarray) -> None:
        """Write target_poses, target_masks, joint_targets into pre-allocated buffers.

        When external NPZ motion is loaded with body data AND motion alignment is
        available, uses the NPZ body positions/quats directly (transformed to mocap
        frame) instead of FK. This gives the policy correct world-frame tracking
        signals including pelvis position error.

        Without alignment (no ext_motion or no body_names), falls back to FK-based
        deltas which miss root-position tracking.
        """
        dof_pos = robot_state_data[0, 7 : 7 + self.num_dofs]

        # Robot FK using the robot's actual root from robot_state_data
        robot_root_pos = robot_state_data[0, 0:3].astype(np.float64)
        robot_root_quat_wxyz = robot_state_data[0, 3:7]
        robot_root_quat_xyzw = wxyz_to_xyzw(robot_root_quat_wxyz.reshape(1, 4)).flatten().astype(np.float64)
        robot_bpos_w, robot_bquat_xyzw = self._run_fk(dof_pos, robot_root_pos, robot_root_quat_xyzw)

        # Save robot FK results before reference FK overwrites the shared buffer.
        robot_bpos_w = robot_bpos_w.copy()
        robot_bquat_xyzw = robot_bquat_xyzw.copy()

        # Pre-compute robot body inverse rotations (wxyz) for delta computation
        robot_q_inv_wxyz_list = []
        for bi in self._visible_body_indices:
            robot_q_wxyz = xyzw_to_wxyz(robot_bquat_xyzw[bi].reshape(1, 4))
            robot_q_inv_wxyz_list.append(quat_inverse(robot_q_wxyz))

        # Zero out buffers
        self._target_poses_buf[:] = 0.0
        self._target_masks_buf[:] = 0.0
        self._joint_targets_buf[:] = 0.0

        # Use direct NPZ body positions (aligned to mocap frame) when available.
        # This gives correct world-frame deltas including pelvis position tracking.
        use_aligned_npz = self._ext_body_npz_indices is not None and self._align_rot_mat is not None

        # NOTE: Training s2t (source-to-target) offset is zero for G1 without body aliases:
        # source = robot_body_pos_w, target = robot_body_pos_w (same), so s2t = 0.
        # The delta is simply body_local(motion_future_body - robot_current_body).

        # LIVE VIVE override (two_point + armed only): replace the two wrist world
        # targets with the operator's tracker-driven pose, held STATIC across the
        # whole future lookahead so the frozen clip's future frames can't drift the
        # hands. ``held_*`` = the reference wrist world pose at the FROZEN frame
        # (``_ext_frame_idx``); the clutch delta is added on top. Empty dicts when
        # not armed ⇒ the loop below is byte-identical to the NPZ replay path.
        if use_aligned_npz:
            held_bpos_w, held_bquat_xyzw = self._transform_ref_body_to_mocap(self._ext_frame_idx)
        else:
            assert self._ext_motion is not None
            raw_held = self._ext_motion["joint_pos"][min(self._ext_frame_idx, self._ext_motion_length - 1)]
            held_joints = raw_held[7:] if raw_held.shape[0] > self.num_dofs else raw_held
            held_bpos_w, held_bquat_xyzw = self._run_fk(held_joints, robot_root_pos, robot_root_quat_xyzw)
            held_bpos_w = held_bpos_w.copy()
            held_bquat_xyzw = held_bquat_xyzw.copy()
        vive_pos, vive_quat_xyzw = self._vive_wrist_targets(robot_state_data, held_bpos_w, held_bquat_xyzw)

        # Trajectory-viz tap: cache the commanded wrist WORLD target + the measured
        # wrist WORLD pos for the visible wrists, so the interactive viewer can draw
        # tracker-vs-executed polylines + report the error. Pure read-out — empty
        # when not armed; never affects the obs. (Stored as cond_idx → world xyz.)
        if vive_pos:
            self._vive_last_target_w = {bi: vive_pos[bi].copy() for bi in vive_pos}
            self._vive_last_measured_w = {bi: robot_bpos_w[bi].astype(np.float32).copy() for bi in vive_pos}
        else:
            self._vive_last_target_w = {}
            self._vive_last_measured_w = {}

        for step in range(_EVAL_VISIBLE_STEPS):
            step_offset_poses = step * _NUM_COND_BODIES * 7
            step_offset_masks = step * _NUM_COND_BODIES

            ft = min(self._ext_frame_idx + step + 1, self._ext_motion_length - 1)
            raw_pos = self._ext_motion["joint_pos"][ft]
            future_joints = raw_pos[7:] if raw_pos.shape[0] > self.num_dofs else raw_pos

            if use_aligned_npz:
                ref_bpos_w, ref_bquat_xyzw = self._transform_ref_body_to_mocap(ft)
            else:
                ref_bpos_w, ref_bquat_xyzw = self._run_fk(future_joints, robot_root_pos, robot_root_quat_xyzw)

            for li, bi in enumerate(self._visible_body_indices):
                robot_q_inv_wxyz = robot_q_inv_wxyz_list[li]

                if bi in vive_pos:
                    # Tracker-driven wrist: STATIC held pose + clutch delta (same
                    # for every future step, so the policy sees a steady target).
                    tgt_pos = vive_pos[bi]
                    tgt_q_wxyz = xyzw_to_wxyz(vive_quat_xyzw[bi].reshape(1, 4))
                else:
                    tgt_pos = ref_bpos_w[bi]
                    tgt_q_wxyz = xyzw_to_wxyz(ref_bquat_xyzw[bi].reshape(1, 4))

                dp = quat_apply(robot_q_inv_wxyz, (tgt_pos - robot_bpos_w[bi]).reshape(1, 3))
                dq_wxyz = quat_mul(robot_q_inv_wxyz, tgt_q_wxyz).flatten()
                dq_xyzw = wxyz_to_xyzw(dq_wxyz.reshape(1, 4)).flatten().astype(np.float32)

                idx = step_offset_poses + bi * 7
                self._target_poses_buf[idx : idx + 3] = dp.flatten().astype(np.float32)
                self._target_poses_buf[idx + 3 : idx + 7] = _canonicalize_quat_xyzw(dq_xyzw)
                self._target_masks_buf[step_offset_masks + bi] = 1.0

            # In training hover mode, joint_targets are masked by hover_joint_mask.
            # For keypoint-only modes (three_point, etc.) hover_joint_mask is all zeros,
            # so training sees all-zero joint_targets with joint_mode_mask=0.
            # Only fill when _joint_mode=True (full_body with joint conditioning).
            if self._joint_mode:
                jt_offset = step * self.num_dofs
                self._joint_targets_buf[jt_offset : jt_offset + self.num_dofs] = future_joints

        # Per-term clip matching training clip_observations (applied before normalization)
        np.clip(self._target_poses_buf, -_CLIP_OBS, _CLIP_OBS, out=self._target_poses_buf)
        np.clip(self._joint_targets_buf, -_CLIP_OBS, _CLIP_OBS, out=self._joint_targets_buf)

    def _detect_force_obs_layout(self) -> None:
        """Enable the trailing 6D force_cmd_local term iff the ONNX obs input is
        2415-wide (= 2409 base + 6 force). 2409 → standard maskedmimic (no force)."""
        obs_dim = None
        for inp in self.onnx_policy_session.get_inputs():
            if inp.name == "obs" and inp.shape and isinstance(inp.shape[-1], int):
                obs_dim = int(inp.shape[-1])
                break
        if obs_dim == _OBS_DIM + _FORCE_CMD_DIM:
            self._force_obs_enabled = True
            new_dim = _OBS_DIM + _FORCE_CMD_DIM
            self._obs_buf = np.zeros((1, new_dim), dtype=np.float32)
            self._clip_scratch = np.zeros((1, new_dim), dtype=np.float32)
            frame = (
                "SENSOR (ft_sensor_link)"
                if self._force_cmd_sensor_frame
                else "HAND (wrist_yaw)"
                if self._force_cmd_hand_frame
                else "pelvis (root)"
            )
            override = (
                self._force_cmd_pelvis_raw.reshape(-1).tolist()
                if self._force_cmd_pelvis_raw is not None
                else "zeros (pure tracking)"
            )
            logger.info(
                colored(
                    f"MaskedMimic-FORCE obs layout: {new_dim}D (base {_OBS_DIM} + force {_FORCE_CMD_DIM}); "
                    f"F^cmd frame={frame}, override={override}",
                    "magenta",
                )
            )
        else:
            self._force_obs_enabled = False
            if self._force_cmd_pelvis_raw is not None:
                logger.warning(
                    f"HOLOSOMA_FORCE_CMD_LOCAL set but ONNX obs width is {obs_dim} (not "
                    f"{_OBS_DIM + _FORCE_CMD_DIM}) — this is NOT a force checkpoint; F^cmd override ignored."
                )

    def _force_cmd_local_obs(self, robot_state_data: np.ndarray) -> np.ndarray:
        """Build the 6D force_cmd_local obs term (already ×0.04-scaled).

        Zero when no HOLOSOMA_FORCE_CMD_LOCAL override is set. Otherwise returns
        the commanded force in the frame the trained policy expects: pelvis/root by
        default, each hand's wrist_yaw frame under HOLOSOMA_FORCE_CMD_HAND_FRAME=1, or
        each {side}_ft_sensor_link frame under HOLOSOMA_FORCE_CMD_SENSOR_FRAME=1
        (= wrist frame post-rotated by the constant R_sensor). Mirrors
        holosoma_wbt.HolosomaWBTPolicy._force_cmd_local_obs exactly.
        """
        if self._force_cmd_pelvis_raw is None:
            return np.zeros(_FORCE_CMD_DIM, dtype=np.float32)
        if not (self._force_cmd_hand_frame or self._force_cmd_sensor_frame):
            return self._force_cmd_local_scaled.copy()
        _env = "HOLOSOMA_FORCE_CMD_SENSOR_FRAME" if self._force_cmd_sensor_frame else "HOLOSOMA_FORCE_CMD_HAND_FRAME"
        if not hasattr(self, "pinocchio_robot"):
            logger.warning(f"{_env} set but no Pinocchio model; using pelvis frame.")
            return self._force_cmd_local_scaled.copy()

        model = self.pinocchio_robot.robot_model
        data = self.pinocchio_robot.robot_data
        if self._force_wrist_frame_ids is None:
            self._force_wrist_frame_ids = []
            for n in _VR_2POINT_BODIES:
                fid = model.getFrameId(n)
                if fid >= model.nframes:
                    logger.warning(f"{_env} wrist body '{n}' not found in URDF; R_rel will be inaccurate")
                self._force_wrist_frame_ids.append(fid)
        # Identity freeflyer root → oMf[wrist].rotation is the wrist orientation
        # IN the pelvis frame (R_rel). F_hand = R_rel^-1 @ F_pelvis.
        dof_pos = robot_state_data[0, 7 : 7 + self.num_dofs]
        joint_pos_pin = dof_pos[self.pinocchio_robot.real2pinocchio_index]
        q = np.concatenate([np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), joint_pos_pin])
        pin.framesForwardKinematics(model, data, q)

        r_sensor = _rot_matrix_from_rpy(self._force_sensor_rpy) if self._force_cmd_sensor_frame else None
        out = np.empty((2, 3), dtype=np.float32)
        for hi, fid in enumerate(self._force_wrist_frame_ids):
            R_rel = np.asarray(data.oMf[fid].rotation)
            f_hand = R_rel.T @ self._force_cmd_pelvis_raw[hi]  # world→hand
            out[hi] = (r_sensor.T @ f_hand) if r_sensor is not None else f_hand  # world→sensor
        return (out.reshape(-1) * _FORCE_CMD_OBS_SCALE).astype(np.float32)

    def _build_observation(self, robot_state_data: np.ndarray) -> np.ndarray:
        reduced_coords = self._build_reduced_coords(robot_state_data)
        self._build_target_poses_and_masks(robot_state_data)

        components = [
            reduced_coords,
            self._target_poses_buf,
            self._target_masks_buf,
            self._joint_targets_buf,
            self._joint_mode_mask,
            self._last_action,
            self._ee_masses,
        ]
        # Append the trailing 6D F^cmd term ONLY for force checkpoints (2415D),
        # matching the training actor layout (force_cmd_local is the last term).
        if self._force_obs_enabled:
            components.append(self._force_cmd_local_obs(robot_state_data))

        obs = np.concatenate(components).reshape(1, -1).astype(np.float32)

        if self._norm_mean is not None:
            obs = _shared_clip_normalized_obs(obs, self._norm_mean, self._norm_std_eps)
        return obs

    # -----------------------------------------------------------------
    # Inference loop
    # -----------------------------------------------------------------

    def rl_inference(self, robot_state_data):
        if not self.motion_clip_progressing:
            self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
            self.curr_motion_timestep = self.timestep_util.timestep

        # Compute motion-to-mocap alignment on first step — ONLY when mocap is
        # providing real position data.  Without mocap, root_pos is always [0,0,0]
        # and alignment would create growing world-frame deltas that the robot
        # can't correct.  The FK fallback path is correct for local tracking
        # because it computes both robot and reference body poses at the same
        # root, producing only joint-configuration deltas.
        if self._align_rot_mat is None and self._ext_body_npz_indices is not None:
            root_pos = robot_state_data[0, 0:3]
            has_mocap = not (np.all(root_pos == 0))
            if has_mocap:
                self._compute_motion_alignment(robot_state_data)
            elif self._inference_step == 0:
                logger.info(
                    colored(
                        "No mocap data (root pos=0) — using FK-based local tracking. "
                        "Pelvis world-frame position will NOT be tracked.",
                        "yellow",
                    )
                )

        # Lazy heading capture (same pattern as TML/dense): on the very first
        # rl_inference call after _reset_motion_state, compute heading correction.
        # For MM the FK fallback already uses robot root (heading-invariant),
        # but we capture the correction quat for logging and potential future use.
        if not self._heading_initialized:
            robot_quat_wxyz = robot_state_data[0, 3:7]
            robot_yaw = float(
                np.arctan2(
                    2 * (robot_quat_wxyz[0] * robot_quat_wxyz[3] + robot_quat_wxyz[1] * robot_quat_wxyz[2]),
                    1 - 2 * (robot_quat_wxyz[2] ** 2 + robot_quat_wxyz[3] ** 2),
                )
            )
            motion_quat_wxyz = xyzw_to_wxyz(self._ref_quat_xyzw_t.reshape(1, 4)).flatten()
            motion_yaw = float(
                np.arctan2(
                    2 * (motion_quat_wxyz[0] * motion_quat_wxyz[3] + motion_quat_wxyz[1] * motion_quat_wxyz[2]),
                    1 - 2 * (motion_quat_wxyz[2] ** 2 + motion_quat_wxyz[3] ** 2),
                )
            )
            yaw_diff = robot_yaw - motion_yaw
            self._heading_correction_quat_xyzw = np.array(
                [0, 0, np.sin(yaw_diff / 2), np.cos(yaw_diff / 2)], dtype=np.float64
            )
            self._heading_initialized = True
            logger.info(
                f"Heading correction (lazy): robot_yaw={np.degrees(robot_yaw):.1f}° "
                f"motion_yaw={np.degrees(motion_yaw):.1f}° diff={np.degrees(yaw_diff):.1f}°"
            )

        obs = self._build_observation(robot_state_data)
        if self._debug_log is not None:
            obs_raw = obs.flatten().copy()

        self._timestep_buf[0, 0] = self.curr_motion_timestep
        input_feed = {"obs": obs, "time_step": self._timestep_buf}
        outputs = self.onnx_policy_session.run(None, input_feed)

        raw_action = outputs[self._onnx_out_idx["actions"]].flatten()
        # Sim2real action clip (tighter than training's 100.0)
        clip_val = self._action_clip_sim2real if self._action_clip_sim2real > 0 else _CLIP_OBS
        action = np.clip(raw_action, -clip_val, clip_val)
        # EMA smoothing: blend with previous smoothed action
        alpha = self._action_ema_alpha
        if alpha < 1.0:
            self._smoothed_action[:] = alpha * action + (1.0 - alpha) * self._smoothed_action
            action = self._smoothed_action.copy()
        # Ramp-up: scale down actions during the first N steps
        if self._action_ramp_steps > 0 and self._action_step_count < self._action_ramp_steps:
            ramp = float(self._action_step_count) / float(self._action_ramp_steps)
            action = action * ramp
        self._action_step_count += 1

        if self._debug_log is not None:
            dof_pos = robot_state_data[0, 7 : 7 + self.num_dofs]
            self._debug_log["obs_raw"].append(obs_raw)
            self._debug_log["obs_clipped"].append(obs.flatten().copy())
            self._debug_log["action"].append(action.copy())
            self._debug_log["dof_pos"].append(dof_pos.copy())
            self._debug_log["timestep"].append(self.curr_motion_timestep)
            self._debug_log["robot_state"].append(robot_state_data[0].copy())
            self._debug_log["term_reduced_coords"].append(self._reduced_coords_buf.copy())
            self._debug_log["term_target_poses"].append(self._target_poses_buf.copy())
            self._debug_log["term_target_masks"].append(self._target_masks_buf.copy())
            self._debug_log["term_joint_targets"].append(self._joint_targets_buf.copy())
            self._debug_log["frame_idx"].append(float(self._ext_frame_idx))

        self._last_action[:] = action
        self.last_policy_action = self._last_action.reshape(1, -1)

        # Diagnostic logging for the first few inference steps
        if self._inference_step < 3:
            obs_flat = obs.flatten()
            reduced = obs_flat[:_REDUCED_COORDS_DIM]
            target_poses = obs_flat[_REDUCED_COORDS_DIM : _REDUCED_COORDS_DIM + _TARGET_POSES_DIM]
            masks = obs_flat[
                _REDUCED_COORDS_DIM + _TARGET_POSES_DIM : _REDUCED_COORDS_DIM + _TARGET_POSES_DIM + _TARGET_MASKS_DIM
            ]
            scaled_act = action * self._action_scale
            q_target = scaled_act + self.default_dof_angles
            raw_absmax = np.abs(raw_action).max()
            logger.info(
                f"[DIAG step={self._inference_step}] "
                f"obs: min={obs_flat.min():.3f} max={obs_flat.max():.3f} mean={obs_flat.mean():.3f} | "
                f"reduced_coords: min={reduced.min():.3f} max={reduced.max():.3f} | "
                f"target_poses nonzero={np.count_nonzero(target_poses)}/{_TARGET_POSES_DIM} | "
                f"masks sum={masks.sum():.0f}/{_TARGET_MASKS_DIM} | "
                f"raw_onnx_action: absmax={raw_absmax:.3f} | "
                f"smoothed_action: min={action.min():.3f} max={action.max():.3f} | "
                f"scaled_action: min={scaled_act.min():.3f} max={scaled_act.max():.3f} | "
                f"q_target: min={q_target.min():.3f} max={q_target.max():.3f} | "
                f"ema_alpha={self._action_ema_alpha} clip={clip_val:.1f} action_ramp={self._action_step_count}/{self._action_ramp_steps} | "
                f"timestep={self.curr_motion_timestep} frame_idx={self._ext_frame_idx} | "
                f"normalizer={'YES' if self._norm_mean is not None else 'NO'}"
            )
            self._inference_step += 1

        # Update motion reference from external NPZ
        jp, jv, rp, rq = self._get_ext_motion_at_frame(self._ext_frame_idx)
        self._motion_command_t[: self.num_dofs] = jp
        self._motion_command_t[self.num_dofs :] = jv
        self._ref_quat_xyzw_t[:] = rq
        self._ref_pos_xyz_t[:] = rp

        np.multiply(self._last_action, self._action_scale, out=self._scaled_action_buf[0])
        self.scaled_policy_action = self._scaled_action_buf

        if self.motion_clip_progressing:
            self._set_motion_timestep()
            self._ext_frame_idx = min(self._ext_frame_idx + 1, self._ext_motion_length - 1)

        return self.scaled_policy_action

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def get_init_target(self, robot_state_data):
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            target = self._init_q_target.reshape(1, -1)
            q_target = dof_pos + (target - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def _handle_init_state(self):
        """Interpolate to motion init pose while keeping stiff KP/KD.

        Without this, pressing Y instantly drops from stiff KP (350) to policy
        KP (~40), causing the robot to collapse under gravity before the policy
        can respond.  The interpolation runs inside _get_manual_command which
        keeps stiff KP/KD the whole time.
        """
        # INIT can be pressed at any time (incl. while VIVE is armed). Always drop
        # to a known DISARMED state with cleared clutch anchors so a later START →
        # arm ramps from zero rather than off a stale pre-INIT anchor.
        self._vive_armed = False
        self._reset_vive_anchors()
        self._init_interpolating = True
        self._init_start_q = self._stiff_hold_q.copy()
        self._init_interp_count = 0
        self.logger.info(
            colored("Init: interpolating to motion init pose (stiff KP/KD, ~5s). Wait for 'Init pose reached'.", "cyan")
        )

    def _get_manual_command(self, robot_state_data):
        if not self._stiff_hold_active:
            return None

        q_target = self._stiff_hold_q.copy()

        if self._init_interpolating:
            alpha = min(self._init_interp_count / float(_INIT_INTERP_STEPS), 1.0)
            init_target = self._init_q_target.reshape(1, -1)
            q_target = self._init_start_q + (init_target - self._init_start_q) * alpha
            self._init_interp_count += 1
            if alpha >= 1.0:
                self._init_interpolating = False
                self._stiff_hold_q = q_target.copy()
                self.logger.info(colored("Init pose reached", "green"))
            return {"q": q_target, "kp": self._stiff_hold_kp, "kd": self._stiff_hold_kd}

        if self._prepend_stiff_active:
            # Drive prepend transition with stiff KP — direct position tracking
            # (no ONNX inference), same requirements as init interpolation.
            raw_jp = self._ext_motion["joint_pos"][self._ext_frame_idx]
            ref_joints = raw_jp[7:] if raw_jp.shape[0] > self.num_dofs else raw_jp
            q_target = ref_joints.reshape(1, -1)
            # Update motion reference bookkeeping
            jp, jv, rp, rq = self._get_ext_motion_at_frame(self._ext_frame_idx)
            self._motion_command_t[: self.num_dofs] = jp
            self._motion_command_t[self.num_dofs :] = jv
            self._ref_quat_xyzw_t[:] = rq
            self._ref_pos_xyz_t[:] = rp
            # Advance frame
            if self.motion_clip_progressing:
                self._set_motion_timestep()
                self._ext_frame_idx = min(self._ext_frame_idx + 1, self._ext_motion_length - 1)
            if self._ext_frame_idx % 25 == 0:
                self.logger.info(
                    f"[Prepend {self._ext_frame_idx}/{self._num_prepend_frames}] "
                    f"stiff KP=[{self._stiff_hold_kp.min():.0f},{self._stiff_hold_kp.max():.0f}]"
                )
            # Prepend done → re-capture heading and enable policy with ONNX KP
            if self._ext_frame_idx >= self._num_prepend_frames:
                self._prepend_stiff_active = False
                # Re-capture heading AFTER prepend — robot orientation changed during
                # joint tracking. Stale heading = aggressive rotation at ONNX start.
                robot_state = self.interface.get_low_state()
                if robot_state is not None:
                    rq_wxyz = robot_state[0, 3:7]
                    robot_yaw = float(
                        np.arctan2(
                            2 * (rq_wxyz[0] * rq_wxyz[3] + rq_wxyz[1] * rq_wxyz[2]),
                            1 - 2 * (rq_wxyz[2] ** 2 + rq_wxyz[3] ** 2),
                        )
                    )
                    mq_wxyz = xyzw_to_wxyz(self._ref_quat_xyzw_t.reshape(1, 4)).flatten()
                    motion_yaw = float(
                        np.arctan2(
                            2 * (mq_wxyz[0] * mq_wxyz[3] + mq_wxyz[1] * mq_wxyz[2]),
                            1 - 2 * (mq_wxyz[2] ** 2 + mq_wxyz[3] ** 2),
                        )
                    )
                    yaw_diff = robot_yaw - motion_yaw
                    self._heading_correction_quat_xyzw = np.array(
                        [0, 0, np.sin(yaw_diff / 2), np.cos(yaw_diff / 2)], dtype=np.float64
                    )
                    logger.info(
                        f"Heading re-captured after prepend: robot={np.degrees(robot_yaw):.1f}° "
                        f"motion={np.degrees(motion_yaw):.1f}° diff={np.degrees(yaw_diff):.1f}°"
                    )
                BasePolicy._handle_start_policy(self)
                self._stiff_hold_active = False
                self.logger.info(colored("Prepend complete — policy running with ONNX KP", "green"))
            return {"q": q_target, "kp": self._stiff_hold_kp, "kd": self._stiff_hold_kd}

        return {"q": q_target, "kp": self._stiff_hold_kp, "kd": self._stiff_hold_kd}

    def _handle_start_policy(self):
        """Start policy with correct ordering to prevent race conditions.

        Phase 1: Set up all state rl_inference reads (while use_policy_action
                 is still False so the policy loop cannot fire yet).
        Phase 2: Enable policy immediately, disable stiff hold.
        """
        self._init_interpolating = False

        # Phase 1: Reset state before enabling policy
        self._debug_log_path = os.environ.get("HOLOSOMA_DEBUG_LOG", "")
        if self._debug_log_path:
            self._debug_log = defaultdict(list)
            logger.info(colored(f"Debug logging ON → {self._debug_log_path}", "magenta"))

        # NOTE: _handle_start_motion_clip → _reset_motion_state sets
        # _heading_initialized=False and _heading_correction_quat_xyzw=None.
        # Heading is lazily captured on the first rl_inference call
        # (same pattern as TML/dense).
        self._handle_start_motion_clip()

        if self._num_prepend_frames > 0:
            raise NotImplementedError("prepend mode not yet supported")
            self._prepend_stiff_active = True
            self.logger.info(
                colored(
                    f"Prepend active ({self._num_prepend_frames} frames, "
                    f"~{self._num_prepend_frames / 50.0:.1f}s) — stiff KP held during transition.",
                    "cyan",
                )
            )
        elif self._vive_live is not None and self._is_2pt_mm:
            # Live-VIVE 2pt teleop: START brings the policy up holding the FROZEN
            # frame-0 standing pose, trackers DISARMED. The operator then arms with
            # the 'start' button (_toggle_vive_teleop) once stable. Mirrors the WBT
            # live path — distinct from CLIP play (which advances the NPZ).
            BasePolicy._handle_start_policy(self)
            self._stiff_hold_active = False
            self.motion_clip_progressing = False  # FREEZE — stand, don't auto-play
            self._vive_armed = False  # trackers ignored until 'start' arms them
            self.logger.info(
                colored(
                    "START — policy holding the standing pose. Press 'start' (keyboard '=') to arm "
                    "VIVE tracking; 'x'/'X' to recenter the clutch.",
                    "green",
                )
            )
            self.logger.warning(
                colored(
                    "⚠️ NO SOFTWARE ABORT on this MaskedMimic path — STOP only re-stiffens at the current "
                    "pose (it does NOT limp), and there is no soft-release. The HARDWARE E-STOP is the only "
                    "true abort. Gantry + a dedicated E-stop holder are mandatory on hardware.",
                    "red",
                )
            )
        else:
            # No prepend — enable policy immediately (NPZ clip auto-plays).
            BasePolicy._handle_start_policy(self)
            self._stiff_hold_active = False
            self.logger.info(colored("Motion clip auto-started", "green"))

    def _handle_stop_policy(self):
        self._save_debug_log()
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self._prepend_stiff_active = False
        self._init_interpolating = False
        self._vive_armed = False  # STOP disarms live VIVE teleop
        self._reset_vive_anchors()
        self._teardown_motion()
        self._reset_motion_state()
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0
        self.logger.info(colored("HolosomaMaskedMimic stopped, entering stiff hold", "yellow"))

    # -----------------------------------------------------------------
    # Live VIVE arming (relative/clutch) — parallel to holosoma_wbt
    # -----------------------------------------------------------------

    def _reset_vive_anchors(self) -> None:
        """Drop the slew + clutch anchors so a later (re)arm ramps cleanly from
        the held reference (zero delta) rather than a stale held delta."""
        self._vr_last_dpos = None
        self._vr_last_drot6d = None
        self._vr_clutch_live0 = None
        self._vr_clutch_orn0 = None

    def _snapshot_clutch_anchor(self) -> bool:
        """Snapshot the clutch origin = the trackers' current contact pos[6] (+ wrist
        rot6d[12] when orn clutch on). After this the commanded wrist target is the
        held frozen pose + the tracker displacement SINCE the anchor, so arming is
        zero-jump no matter where the trackers physically are. Returns False (logs)
        when there is no live frame."""
        assert self._vive_live is not None  # callers guard; narrows for mypy
        live_pos, live_orn = self._vive_live.latest()
        if live_pos is None:
            self.logger.warning(colored("VIVE clutch: no live frame (stale/disconnected).", "red"))
            return False
        self._vr_clutch_live0 = live_pos.astype(np.float32).copy()
        # Snapshot the wrist-rot6d origin WHENEVER the bridge sends orn — not only
        # when the orn CLUTCH is enabled — so _compute_vive_world_delta can always
        # subtract the (possibly swinging) contact lever and recover the true wrist
        # ORIGIN displacement. The orn-clutch flag separately controls whether we
        # also rotate the wrist target.
        if live_orn is not None:
            self._vr_clutch_orn0 = live_orn.astype(np.float32).copy()
        else:
            self._vr_clutch_orn0 = None
        # Reset the slew anchor so the first commanded frame after (re)anchoring is
        # exactly the held reference (delta 0), then ramps from there.
        self._vr_last_dpos = None
        self._vr_last_drot6d = None
        return True

    def _toggle_vive_teleop(self) -> None:
        """Arm/disarm live VIVE 2pt teleop (the 'start' button).

        Requires the policy running (A=START first) + a VIVE source + two_point
        mode. Arming snapshots the clutch anchor at the trackers' CURRENT pose and
        freezes the clip, so the hands start exactly at the held reference and
        follow only relative motion (zero jump, trackers can sit anywhere).
        Disarming returns the hands to the frozen reference."""
        if self._vive_live is None or not self._is_2pt_mm:
            self.logger.warning(colored("VIVE arm: no live 2pt source — nothing to toggle.", "yellow"))
            return
        if not self.use_policy_action:
            self.logger.warning(colored("VIVE arm: press START first so the policy is running.", "yellow"))
            return
        if self._vive_armed:
            self._vive_armed = False
            self._reset_vive_anchors()
            self.logger.info(colored("VIVE teleop DISARMED — hands hold the standing reference.", "yellow"))
            return
        if not self._snapshot_clutch_anchor():
            return
        self.motion_clip_progressing = False  # keep the clip frozen; only hands track
        self._vive_armed = True
        self.logger.info(
            colored(
                "VIVE teleop ARMED (relative/clutch) — hands start at the reference and follow your "
                "trackers' motion. Press 'x'/'X' to recenter (slide hands back + re-anchor).",
                "green",
            )
        )

    def _vive_recenter(self) -> None:
        """Recenter the clutch: re-snap the anchor at the trackers' current pose so
        the hands GLIDE back to the frozen reference and the tracker origin resets,
        without disarming. No-op (with a log) when not armed."""
        if not self._vive_armed:
            self.logger.warning(colored("VIVE recenter: not armed — press 'start' to arm first.", "yellow"))
            return
        if self._snapshot_clutch_anchor():
            self.logger.warning(
                colored(
                    "VIVE RECENTERED — hands are GLIDING back to the reference pose (slew-capped, a real "
                    "motion); tracker origin reset.",
                    "cyan",
                )
            )

    def _dispatch_command(self, cmd):
        """Add VIVE arm/recenter commands on top of the base dispatch.

        * StateCommand.STAND_TOGGLE (keyboard '=', joystick 'start') → arm/disarm
          live VIVE teleop. Lets the operator stand under the policy first (START),
          then hand the hands to the trackers — distinct from CLIP play.
        * StateCommand.SWITCH_MODE (keyboard 'x', joystick 'X') → recenter the
          clutch. base treats SWITCH_MODE as a no-op, so repurposing it is safe.
        """
        from holosoma_inference.inputs.api.commands import StateCommand  # type: ignore[import-not-found]

        if cmd == StateCommand.STAND_TOGGLE:
            self._toggle_vive_teleop()
        elif cmd == StateCommand.SWITCH_MODE:
            self._vive_recenter()
        else:
            super()._dispatch_command(cmd)

    def _handle_start_motion_clip(self):
        super()._handle_start_motion_clip()
        self._reset_motion_state()

    def _capture_policy_state(self):
        state = BasePolicy._capture_policy_state(self)
        state.update(
            {
                "motion_command_0": self._motion_command_0.copy(),
                "ref_quat_xyzw_0": self._ref_quat_xyzw_0.copy(),
                "ref_pos_xyz_0": self._ref_pos_xyz_0.copy(),
                "per_joint_policy_action_scale": (
                    self.per_joint_policy_action_scale.copy()
                    if self.per_joint_policy_action_scale is not None
                    else None
                ),
            }
        )
        return state

    def _restore_policy_state(self, state):
        BasePolicy._restore_policy_state(self, state)
        self._motion_command_0[:] = state["motion_command_0"]
        self._ref_quat_xyzw_0[:] = state["ref_quat_xyzw_0"]
        self._ref_pos_xyz_0[:] = state.get("ref_pos_xyz_0", np.zeros(3, dtype=np.float32))
        saved = state["per_joint_policy_action_scale"]
        self.per_joint_policy_action_scale = saved.copy() if saved is not None else None
        self._init_interpolating = False
        self._vive_armed = False
        self._reset_vive_anchors()
        self._teardown_motion()
        self._reset_motion_state()

    def _on_policy_switched(self, model_path):
        self._stiff_hold_active = True
        self._init_interpolating = False
        self._vive_armed = False
        self._reset_vive_anchors()
        self._teardown_motion()
        self._reset_motion_state()

    def get_current_obs_buffer_dict(self, robot_state_data):
        return {
            "motion_command": self._motion_command_t.reshape(1, -1),
            "base_ang_vel": robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6],
            "dof_pos": robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles,
            "dof_vel": robot_state_data[:, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs],
            "actions": self.last_policy_action,
        }
