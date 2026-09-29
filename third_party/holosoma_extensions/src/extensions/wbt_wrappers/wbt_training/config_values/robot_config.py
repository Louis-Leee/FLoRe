"""G1 29-DOF WBT robot configuration.

Extends the core holosoma RobotConfig with WBT-specific body and DOF
mappings (VR tracking, Isaac<->MuJoCo DOF reordering, conditionable bodies)
and provides ``build_g1_implicit_actuators`` — the per-motor calibration
registered on ``robot.control.implicit_actuator_builder`` when a preset
uses ``control_mode="implicit_position_target"``.
"""

from __future__ import annotations

from dataclasses import fields as _dataclass_fields

import numpy as np
from pydantic.dataclasses import dataclass

from holosoma.config_types.robot import RobotConfig
from holosoma.config_values.robot import g1_29dof as _base_g1_29dof


@dataclass(frozen=True)
class WBTRobotConfig(RobotConfig):
    dof_names_23dof: list[str] | None = None
    dof23_to_dof29_indices: list[int] | None = None
    wbt_body_names: list[str] | None = None
    isaaclab_to_mujoco_dof: tuple[int, ...] | None = None
    vr_3point_bodies: list[str] | None = None
    vr_3point_offsets: list[list[float]] | None = None
    vr_2point_bodies: list[str] | None = None
    vr_2point_offsets: list[list[float]] | None = None
    vr_1point_bodies: list[str] | None = None
    vr_1point_offsets: list[list[float]] | None = None
    conditionable_bodies: list[str] | None = None
    body_name_to_mj_idx: dict[str, int] | None = None
    cond_indices_30body: list[int] | None = None


# fmt: off
_base = {f.name: getattr(_base_g1_29dof, f.name) for f in _dataclass_fields(_base_g1_29dof)}

g1_29dof_wbt = WBTRobotConfig(
    **_base,
    dof_names_23dof=[
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
    ],
    dof23_to_dof29_indices=[
        0, 1, 2, 3, 4, 5,      # left leg (6)
        6, 7, 8, 9, 10, 11,    # right leg (6)
        12, 13, 14,             # waist (3)
        15, 16, 17, 18,         # left arm (4, no wrist)
        22, 23, 24, 25,         # right arm (4, no wrist)
    ],
    wbt_body_names=[
        "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
        "torso_link", "left_shoulder_roll_link", "left_elbow_link",
        "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link",
        "right_wrist_yaw_link",
    ],
    isaaclab_to_mujoco_dof=(
        0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
        2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
    ),
    vr_3point_bodies=["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"],
    vr_3point_offsets=[[0.18, -0.025, 0.0], [0.18, +0.025, 0.0], [0.0, 0.0, 0.35]],
    vr_2point_bodies=["left_wrist_yaw_link", "right_wrist_yaw_link"],
    vr_2point_offsets=[[0.18, -0.025, 0.0], [0.18, +0.025, 0.0]],
    vr_1point_bodies=["right_wrist_yaw_link"],
    vr_1point_offsets=[[0.18, +0.025, 0.0]],
    conditionable_bodies=[
        "pelvis", "torso_link", "right_ankle_roll_link",
        "left_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
    ],
    body_name_to_mj_idx={
        "pelvis": 1,
        "left_hip_pitch_link": 2, "left_hip_roll_link": 3, "left_hip_yaw_link": 4,
        "left_knee_link": 5, "left_ankle_pitch_link": 6, "left_ankle_roll_link": 7,
        "left_foot_contact_point": 7,
        "right_hip_pitch_link": 8, "right_hip_roll_link": 9, "right_hip_yaw_link": 10,
        "right_knee_link": 11, "right_ankle_pitch_link": 12, "right_ankle_roll_link": 13,
        "right_foot_contact_point": 13,
        "waist_yaw_link": 14, "waist_roll_link": 15, "torso_link": 16,
        "left_shoulder_pitch_link": 17, "left_shoulder_roll_link": 18,
        "left_shoulder_yaw_link": 19, "left_elbow_link": 20,
        "left_wrist_roll_link": 21, "left_wrist_pitch_link": 22, "left_wrist_yaw_link": 23,
        "right_shoulder_pitch_link": 24, "right_shoulder_roll_link": 25,
        "right_shoulder_yaw_link": 26, "right_elbow_link": 27,
        "right_wrist_roll_link": 28, "right_wrist_pitch_link": 29, "right_wrist_yaw_link": 30,
    },
    cond_indices_30body=[0, 17, 13, 6, 24, 29],
)
# fmt: on

# ---------------------------------------------------------------------------
# Module-level aliases and derived constants
# ---------------------------------------------------------------------------
G1_32BODY_NAMES = g1_29dof_wbt.body_names
G1_29DOF_JOINT_NAMES = g1_29dof_wbt.dof_names
G1_23DOF_JOINT_NAMES = g1_29dof_wbt.dof_names_23dof
DOF23_TO_DOF29_INDICES = g1_29dof_wbt.dof23_to_dof29_indices
G1_29DOF_WBT_BODY_NAMES = g1_29dof_wbt.wbt_body_names

ISAACLAB_TO_MUJOCO_DOF = np.array(g1_29dof_wbt.isaaclab_to_mujoco_dof)
MUJOCO_TO_ISAACLAB_DOF = np.argsort(ISAACLAB_TO_MUJOCO_DOF)

_LOWER_JOINT_INDICES_MUJOCO = list(range(12))
LOWER_BODY_INDICES = [ISAACLAB_TO_MUJOCO_DOF[i] for i in _LOWER_JOINT_INDICES_MUJOCO]

VR_3POINT_BODIES = g1_29dof_wbt.vr_3point_bodies
VR_3POINT_OFFSETS = g1_29dof_wbt.vr_3point_offsets

# --- TCP-unified 3-point offsets (user 2026-08-03) --------------------------
# The DEFAULT VR offsets put each hand's POSITION-tracking point 0.18 m along
# wrist_yaw +x (a VR-tracker mounting lever) while the hybrid FORCE point is the
# 0.242 m fingertip TCP (hybrid.py TCP_OFFSET_X) — a deliberate 6.2 cm split, plus
# a per-hand lateral term whose SIGN even flips between the two conventions
# (VR left −0.025 / force left +0.003). For `exp:g1-wbt-3pt-hybrid-unified[-anneal]`
# the user wants ONE point: position and force both at the TCP.
#
# So the hands move to EXACTLY the force anchor `(tcp_offset_x, 0, 0)` — the same
# vector hybrid.py:450 writes into `env._wbt_force_offsets_b` — with the ±0.025
# lateral DROPPED (keeping it would leave the two points 2.5 cm apart, i.e. not
# unified). The head point (torso_link +0.35 z) is unrelated to the hands and is
# carried over unchanged.
#
# WHY A SEPARATE CONSTANT rather than editing vr_3point_offsets in place: that
# literal is shared by EVERY 3pt preset (g1_wbt_3pt, …-hybrid, …-global-vel, and
# the 2pt/1pt families via their own literals). Mutating it would silently move the
# tracked point of every already-trained checkpoint 6.2 cm at deploy. This constant
# is consumed ONLY by the `vr_3point_pos_tcp` obs term (hybrid_unified.py), which
# the two unified presets opt into by name.
#
# ⚠️ MIRRORED, NOT IMPORTED: `from …hybrid import TCP_OFFSET_X` here would close the
# cycle robot_config → hybrid → force_tracking → observation → robot_config. Same
# convention hybrid.py:10-11 uses for the inference playground constants — the value
# is pinned by `test_hybrid_3pt_unified.py::test_tcp_unified_offset_matches_hybrid`,
# which fails loudly if hybrid.TCP_OFFSET_X ever moves without this following.
TCP_UNIFIED_HAND_OFFSET_X = 0.242  # == wbt_training.config_values.hybrid.TCP_OFFSET_X
VR_3POINT_OFFSETS_TCP = [
    [TCP_UNIFIED_HAND_OFFSET_X, 0.0, 0.0],  # left  hand: the force TCP itself
    [TCP_UNIFIED_HAND_OFFSET_X, 0.0, 0.0],  # right hand: the force TCP itself
    # Head (torso_link +0.35 z) is unrelated to the hands and carried over UNCHANGED.
    # Spelled as a literal rather than `vr_3point_offsets[2]` because that field is
    # Optional (mypy cannot index it); equality with the default is pinned by
    # test_hybrid_3pt_unified.py::test_tcp_unified_head_point_unchanged.
    [0.0, 0.0, 0.35],
]
VR_2POINT_BODIES = g1_29dof_wbt.vr_2point_bodies
VR_2POINT_OFFSETS = g1_29dof_wbt.vr_2point_offsets
VR_1POINT_BODIES = g1_29dof_wbt.vr_1point_bodies
VR_1POINT_OFFSETS = g1_29dof_wbt.vr_1point_offsets

CONDITIONABLE_BODIES = g1_29dof_wbt.conditionable_bodies
BODY_NAME_TO_MJ_IDX = g1_29dof_wbt.body_name_to_mj_idx

G1_32BODY_COND_INDICES = [G1_32BODY_NAMES.index(bn) for bn in CONDITIONABLE_BODIES]
G1_30BODY_COND_INDICES = g1_29dof_wbt.cond_indices_30body


# ---------------------------------------------------------------------------
# G1 implicit-actuator motor calibration
# Registered on ``robot.control.implicit_actuator_builder`` when a preset sets
# ``control_mode="implicit_position_target"``. Tuned so IsaacSim's built-in PD
# drive applies torque with the correct stiffness/damping/armature.
# ---------------------------------------------------------------------------
_NATURAL_FREQ = 10 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425
_STIFFNESS_5020 = _ARMATURE_5020 * _NATURAL_FREQ**2
_STIFFNESS_7520_14 = _ARMATURE_7520_14 * _NATURAL_FREQ**2
_STIFFNESS_7520_22 = _ARMATURE_7520_22 * _NATURAL_FREQ**2
_STIFFNESS_4010 = _ARMATURE_4010 * _NATURAL_FREQ**2
_DAMPING_5020 = 2.0 * _DAMPING_RATIO * _ARMATURE_5020 * _NATURAL_FREQ
_DAMPING_7520_14 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_14 * _NATURAL_FREQ
_DAMPING_7520_22 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_22 * _NATURAL_FREQ
_DAMPING_4010 = 2.0 * _DAMPING_RATIO * _ARMATURE_4010 * _NATURAL_FREQ


def build_g1_implicit_actuators():
    # Imported lazily so this module is importable without IsaacLab.
    from isaaclab.actuators import ImplicitActuatorCfg

    return {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": _STIFFNESS_7520_14,
                ".*_hip_roll_joint": _STIFFNESS_7520_22,
                ".*_hip_yaw_joint": _STIFFNESS_7520_14,
                ".*_knee_joint": _STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": _DAMPING_7520_14,
                ".*_hip_roll_joint": _DAMPING_7520_22,
                ".*_hip_yaw_joint": _DAMPING_7520_14,
                ".*_knee_joint": _DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": _ARMATURE_7520_14,
                ".*_hip_roll_joint": _ARMATURE_7520_22,
                ".*_hip_yaw_joint": _ARMATURE_7520_14,
                ".*_knee_joint": _ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * _STIFFNESS_5020,
            damping=2.0 * _DAMPING_5020,
            armature=2.0 * _ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * _STIFFNESS_5020,
            damping=2.0 * _DAMPING_5020,
            armature=2.0 * _ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=_STIFFNESS_7520_14,
            damping=_DAMPING_7520_14,
            armature=_ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": _STIFFNESS_5020,
                ".*_shoulder_roll_joint": _STIFFNESS_5020,
                ".*_shoulder_yaw_joint": _STIFFNESS_5020,
                ".*_elbow_joint": _STIFFNESS_5020,
                ".*_wrist_roll_joint": _STIFFNESS_5020,
                ".*_wrist_pitch_joint": _STIFFNESS_4010,
                ".*_wrist_yaw_joint": _STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": _DAMPING_5020,
                ".*_shoulder_roll_joint": _DAMPING_5020,
                ".*_shoulder_yaw_joint": _DAMPING_5020,
                ".*_elbow_joint": _DAMPING_5020,
                ".*_wrist_roll_joint": _DAMPING_5020,
                ".*_wrist_pitch_joint": _DAMPING_4010,
                ".*_wrist_yaw_joint": _DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": _ARMATURE_5020,
                ".*_shoulder_roll_joint": _ARMATURE_5020,
                ".*_shoulder_yaw_joint": _ARMATURE_5020,
                ".*_elbow_joint": _ARMATURE_5020,
                ".*_wrist_roll_joint": _ARMATURE_5020,
                ".*_wrist_pitch_joint": _ARMATURE_4010,
                ".*_wrist_yaw_joint": _ARMATURE_4010,
            },
        ),
    }
