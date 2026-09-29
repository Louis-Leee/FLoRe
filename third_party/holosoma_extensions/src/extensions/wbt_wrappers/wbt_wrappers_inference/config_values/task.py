"""Task configurations for TML and BM WBT wrapper policies."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dataclasses import replace

from pydantic.dataclasses import dataclass

from holosoma_inference.config.config_types.task import TaskConfig

_BUNDLED_G1_URDF = str(Path(__file__).parent.parent / "urdf" / "g1_29dof.urdf")


@dataclass(frozen=True)
class WBTTaskConfig(TaskConfig):
    """Extended task config for WBT wrapper policies."""

    policy_type: str = ""
    """Policy type identifier for explicit selection (e.g. 'tml_wbt', 'bm_wbt')."""

    force_unit_action_scale: bool = False
    """If True, override the ONNX-metadata action_scale to 1.0 on load. Use for
    implicit-PD terrain policies where training applied no scaling
    (``q_target = default + action``); the metadata value is a safety cap only
    and would otherwise be applied twice on the inference side."""

    action_ema_alpha: float = 1.0
    """EMA low-pass smoothing factor for the policy action (1.0 = no smoothing /
    byte-identical pass-through). HolosomaWBTPolicy is trained WITHOUT any action
    filter, so the unfiltered full-KP command can chatter at the pelvis/hips on
    hardware (sim is clean). alpha<1 low-passes the commanded action:
    ``a_t = alpha*a_raw + (1-alpha)*a_{t-1}``; the smoothed action is also fed back
    as the next-step ``actions`` obs term. Because the policy never saw this filter
    in training, a SMALL alpha adds phase lag and can destabilize balance — the
    CAUTIOUS hardware band is 0.7-0.9 (start at 0.85, step down a notch between
    runs); alpha<0.5 is aggressive/experimental (gantry only) and alpha<0.2 is
    rejected unless ``HOLOSOMA_ACTION_EMA_ALLOW_AGGRESSIVE=1``. Overridable live via
    the ``HOLOSOMA_ACTION_EMA_ALPHA`` env var (env wins)."""

    ref_motion_path: str = ""
    """Path to external reference motion .npz file. Supported by TML and MaskedMimic policies."""

    prepend_default_pose_transition: bool = False
    """If True, auto-prepend a smooth transition from default standing pose to the motion's
    first frame when loading an external reference motion. This allows the policy to drive
    the robot from the stiff startup pose to the motion start, rather than using the
    high-gain stiff interpolation. Requires a model trained with
    default_pose_transition_strategy='interpolation'."""

    prepend_duration_s: float = 2.0
    """Duration in seconds of the prepended default-to-motion transition.
    Should match default_pose_prepend_duration_s used during training."""

    robot_urdf_path: str = _BUNDLED_G1_URDF
    """Path to robot URDF for Pinocchio FK. Falls back to ONNX metadata if empty."""

    fk_odometry: bool = False
    """If True, use Pinocchio FK-based odometry to estimate root position from
    joint kinematics + foot contact. For global tracking without external mocap."""

    mocap_body_name: str = ""
    """OptiTrack rigid body name for mocap global tracking. Empty = no mocap."""

    sim_mocap: bool = False
    """If True, receive root position from sim via UDP (interactive sim2sim).
    Matches sim2real mocap path — sim bridge publishes ground-truth root state,
    SimMocapReceiver reads it, MocapInterfaceWrapper injects into get_low_state."""

    mocap_use_orientation: bool = False
    """If True, override IMU orientation with mocap. Usually False (IMU is lower latency)."""

    mocap_head_to_torso: bool = False
    """If True, apply URDF head_link→torso offset to mocap position.
    Use when mocap marker is at the neck (URDF head_link position).
    G1 head_link→torso_link offset: [-0.004, 0, +0.044] in body frame."""

    mocap_real_head_to_torso: bool = False
    """If True, apply real head→torso offset (motive marker on top of head, ~10cm above
    URDF head_link/neck). Combined offset: [-0.004, 0, -0.056] in body frame.
    Use with three_point_real_head mode for head-mounted mocap + wrist controllers."""

    # --- Headless sim eval settings ---
    headless_sim_eval: bool = False
    """Enable headless MuJoCo sim2sim evaluation mode."""

    headless_output: str = ""
    """Output video path (required when headless_sim_eval=True)."""

    headless_max_frames: int = 0
    """Max frames to simulate (0 = full motion length)."""

    headless_metrics_output: str = ""
    """Metrics JSON path. Default: <output>_metrics.json."""

    headless_debug_log: str = ""
    """Debug NPZ log path. Empty = disabled."""

    headless_xml: str = ""
    """Custom MuJoCo XML path. Empty = default G1 XML."""

    headless_no_mocap: bool = False
    """Zero out base position/velocity (simulate no external mocap)."""

    headless_fk_mode: Literal["pinocchio", "mujoco", "none"] = "pinocchio"
    """FK odometry mode when headless_no_mocap: pinocchio, mujoco, none."""

    headless_fixed_camera: bool = False
    """Use fixed camera instead of tracking camera."""

    headless_cam_distance: float = 0.0
    """Camera distance (0 = default)."""

    headless_random_start: bool = False
    """Random xy offset and yaw for robot placement."""

    headless_mocap_yaw_offset: float = 0.0
    """Simulated mocap-IMU yaw offset (degrees)."""

    headless_random_mocap_yaw: bool = False
    """Randomize mocap-IMU yaw offset."""

    headless_terrain: str = ""
    """Path to terrain OBJ mesh or NPY (auto-pairs with the motion NPZ) for
    terrain-aware headless eval. Replaces the flat floor."""

    headless_no_video: bool = False
    """Skip all rendering (reference + policy frames + video compose) and emit
    metrics JSON only. ~4x faster — use for large force-benchmark sweeps; keep
    video for spot-check clips."""


DEFAULT_COMPLIANCE: tuple[float, float, float] = (0.5, 0.5, 0.0)
"""Neutral CHIP compliance default — matches TML-Humanoid-Deploy
RLCHIPPolicy (rl_policy.py:18). Training samples U(0,1) for the first
two axes and fixes axis 2 to 0."""


tml_wbt = WBTTaskConfig(
    model_path="",
    policy_type="tml_wbt",
    rl_rate=50,
    policy_action_scale=1.0,
    use_phase=False,
    ref_motion_path="",
    state_input="interface",
)

bm_wbt = WBTTaskConfig(
    model_path="", policy_type="bm_wbt", rl_rate=50, policy_action_scale=1.0, use_phase=False, state_input="interface"
)

holosoma_wbt = WBTTaskConfig(
    model_path="",
    policy_type="holosoma_wbt",
    rl_rate=50,
    policy_action_scale=1.0,
    action_scales_by_effort_limit_over_p_gain=True,
    use_phase=False,
    state_input="interface",
)

holosoma_3pt_wbt = holosoma_wbt


@dataclass(frozen=True)
class CHIPTaskConfig(WBTTaskConfig):
    compliance: tuple[float, float, float] = DEFAULT_COMPLIANCE


# TEMPORARY: TML-trained CHIP uses its own policy class (different joint order,
# different action_scale parsing). Remove once CHIP is re-exported as holosoma.
tml_chip = CHIPTaskConfig(
    model_path="", policy_type="tml_chip", rl_rate=50, policy_action_scale=1.0, use_phase=False, state_input="interface"
)

holosoma_chip_local_wbt = CHIPTaskConfig(
    model_path="",
    policy_type="holosoma_wbt",
    rl_rate=50,
    policy_action_scale=1.0,
    action_scales_by_effort_limit_over_p_gain=True,
    use_phase=False,
    state_input="interface",
)
holosoma_chip_global_wbt = holosoma_chip_local_wbt


@dataclass(frozen=True)
class MaskedMimicTaskConfig(WBTTaskConfig):
    """Task config for MaskedMimic policies with tracking mode selection."""

    maskedmimic_mode: str = "three_point"
    """Tracking mode: 'two_point', 'three_point', 'three_point_pelvis', 'three_point_pelvis_only', 'five_point', or 'full_body'."""

    action_ema_alpha: float = 1.0
    """EMA smoothing factor for actions (1.0=no smoothing, 0.3=strong smoothing).
    Sim2real: set to 0.3-0.5 to prevent actuator instability from large per-step actions."""

    action_clip_sim2real: float = 0.0
    """If >0, clip raw ONNX action to [-clip, clip] before scaling. Tighter than training's 100.0.
    Sim2real: 2.0-3.0 is typical. 0 disables (uses default 100.0 clip)."""

    action_ramp_steps: int = 0
    """Number of RL steps to linearly ramp action magnitude from 0 to full after policy start.
    Sim2real: 10-25 steps (0.2-0.5s at 50Hz) prevents the first-step spike."""


holosoma_maskedmimic_3pt = MaskedMimicTaskConfig(
    model_path="",
    policy_type="holosoma_maskedmimic",
    rl_rate=50,
    policy_action_scale=1.0,
    use_phase=False,
    maskedmimic_mode="three_point",
    state_input="interface",
)

holosoma_maskedmimic_3pt_head = replace(
    holosoma_maskedmimic_3pt, maskedmimic_mode="three_point_head", mocap_head_to_torso=True
)

holosoma_maskedmimic_3pt_real_head = replace(
    holosoma_maskedmimic_3pt, maskedmimic_mode="three_point_real_head", mocap_real_head_to_torso=True
)

holosoma_maskedmimic_pelvis_wrists = replace(holosoma_maskedmimic_3pt, maskedmimic_mode="three_point_pelvis_only")

holosoma_maskedmimic_4pt = replace(holosoma_maskedmimic_3pt, maskedmimic_mode="three_point_pelvis")

holosoma_maskedmimic_5pt = replace(holosoma_maskedmimic_3pt, maskedmimic_mode="five_point")

holosoma_maskedmimic_full = replace(holosoma_maskedmimic_3pt, maskedmimic_mode="full_body")

holosoma_maskedmimic_2pt = replace(holosoma_maskedmimic_3pt, maskedmimic_mode="two_point")
