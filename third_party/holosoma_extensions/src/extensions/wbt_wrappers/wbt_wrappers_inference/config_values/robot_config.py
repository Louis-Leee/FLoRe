"""G1 29-DOF WBT robot configuration for inference.

Extends the core holosoma_inference RobotConfig with WBT-specific body and DOF
mappings (VR tracking, Isaac<->MuJoCo DOF reordering, conditionable bodies).
"""

from __future__ import annotations

from dataclasses import fields as _dataclass_fields
from typing import TypeVar

import numpy as np
from pydantic.dataclasses import dataclass

from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.config.config_values import robot

_T = TypeVar("_T")


def _req(value: _T | None) -> _T:
    """Narrow an Optional config field to its value, asserting it is populated.

    The WBT fields are declared ``... | None`` (base-class contract) but the
    ``g1_29dof_wbt`` singleton populates all of them, so unwrapping here is
    both type-correct and a cheap runtime guard against a mis-built config.
    """
    if value is None:
        raise ValueError("required WBT robot-config field is None (mis-built config?)")
    return value


@dataclass(frozen=True)
class WBTRobotConfig(RobotConfig):
    body_names: tuple[str, ...] | None = None
    dof_names_23dof: tuple[str, ...] | None = None
    dof23_to_dof29_indices: tuple[int, ...] | None = None
    wbt_body_names: tuple[str, ...] | None = None
    isaaclab_to_mujoco_dof: tuple[int, ...] | None = None
    vr_3point_bodies: tuple[str, ...] | None = None
    vr_3point_offsets: tuple[tuple[float, ...], ...] | None = None
    vr_2point_bodies: tuple[str, ...] | None = None
    vr_2point_offsets: tuple[tuple[float, ...], ...] | None = None
    vr_1point_bodies: tuple[str, ...] | None = None
    vr_1point_offsets: tuple[tuple[float, ...], ...] | None = None
    conditionable_bodies: tuple[str, ...] | None = None
    body_name_to_mj_idx: dict[str, int] | None = None
    cond_indices_30body: tuple[int, ...] | None = None


# fmt: off
g1_29dof_wbt = WBTRobotConfig(
    **{f.name: getattr(robot.g1_29dof, f.name) for f in _dataclass_fields(robot.g1_29dof)},
    body_names=(
        "pelvis",
        "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
        "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
        "left_foot_contact_point",
        "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
        "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
        "right_foot_contact_point",
        "waist_yaw_link", "waist_roll_link", "torso_link",
        "left_shoulder_pitch_link", "left_shoulder_roll_link",
        "left_shoulder_yaw_link", "left_elbow_link",
        "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
        "right_shoulder_pitch_link", "right_shoulder_roll_link",
        "right_shoulder_yaw_link", "right_elbow_link",
        "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
    ),
    dof_names_23dof=(
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
    ),
    dof23_to_dof29_indices=(
        0, 1, 2, 3, 4, 5,      # left leg (6)
        6, 7, 8, 9, 10, 11,    # right leg (6)
        12, 13, 14,             # waist (3)
        15, 16, 17, 18,         # left arm (4, no wrist)
        22, 23, 24, 25,         # right arm (4, no wrist)
    ),
    wbt_body_names=(
        "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
        "torso_link", "left_shoulder_roll_link", "left_elbow_link",
        "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link",
        "right_wrist_yaw_link",
    ),
    isaaclab_to_mujoco_dof=(
        0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
        2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
    ),
    vr_3point_bodies=("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"),
    vr_3point_offsets=((0.18, -0.025, 0.0), (0.18, +0.025, 0.0), (0.0, 0.0, 0.35)),
    vr_2point_bodies=("left_wrist_yaw_link", "right_wrist_yaw_link"),
    vr_2point_offsets=((0.18, -0.025, 0.0), (0.18, +0.025, 0.0)),
    vr_1point_bodies=("right_wrist_yaw_link",),
    vr_1point_offsets=((0.18, +0.025, 0.0),),
    conditionable_bodies=(
        "pelvis", "torso_link", "right_ankle_roll_link",
        "left_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
    ),
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
    cond_indices_30body=(0, 17, 13, 6, 24, 29),
)
# fmt: on

# ---------------------------------------------------------------------------
# Module-level aliases and derived constants (same names as the old constants.py)
# ---------------------------------------------------------------------------
G1_32BODY_NAMES = list(_req(g1_29dof_wbt.body_names))
G1_29DOF_JOINT_NAMES = list(g1_29dof_wbt.dof_names)
G1_23DOF_JOINT_NAMES = list(_req(g1_29dof_wbt.dof_names_23dof))
DOF23_TO_DOF29_INDICES = list(_req(g1_29dof_wbt.dof23_to_dof29_indices))
G1_29DOF_WBT_BODY_NAMES = list(_req(g1_29dof_wbt.wbt_body_names))

ISAACLAB_TO_MUJOCO_DOF = np.array(g1_29dof_wbt.isaaclab_to_mujoco_dof)
MUJOCO_TO_ISAACLAB_DOF = np.argsort(ISAACLAB_TO_MUJOCO_DOF)

_LOWER_JOINT_INDICES_MUJOCO = list(range(12))
LOWER_BODY_INDICES = [ISAACLAB_TO_MUJOCO_DOF[i] for i in _LOWER_JOINT_INDICES_MUJOCO]

VR_3POINT_BODIES = list(_req(g1_29dof_wbt.vr_3point_bodies))
VR_3POINT_OFFSETS = [list(o) for o in _req(g1_29dof_wbt.vr_3point_offsets)]

# TCP-unified 3-point offsets (user 2026-08-03) — the deploy-side mirror of
# wbt_training.config_values.robot_config.VR_3POINT_OFFSETS_TCP on the hybrid_control
# branch. Both HAND points sit on the 0.242 m fingertip TCP (== force_deploy_math.py
# TCP_OFFSET_X, the anchor the whole force stack already acts on) instead of the default
# 0.18 m VR-tracker lever, so position and force supervise ONE point; the head point is
# carried over unchanged. The ±0.025 lateral term is DROPPED on purpose — keeping it
# would leave the two conventions 2.5 cm apart, i.e. not unified.
#
# Consumed ONLY when the loaded checkpoint's actor obs carries the `vr_3point_pos_tcp`
# term (exp:g1-wbt-3pt-hybrid-unified[-anneal] retrained 2026-08-03 onward, e.g. run
# wda3h4ll); every other preset keeps VR_3POINT_OFFSETS. Keying on the TERM NAME rather
# than the preset flag is what makes a pre-2026-08-03 unified checkpoint (iss0qs71 /
# m3s0pw9i, trained at 0.18) impossible to cross-deploy against the wrong FK.
#
# Sourced from force_deploy_math.TCP_OFFSET_X rather than re-typing 0.242, so the
# position anchor and the force anchor CANNOT drift apart — that identity is the entire
# point of the TCP unification. force_deploy_math imports only numpy/dataclasses, so
# there is no import cycle here (unlike the training tree, where robot_config →
# hybrid → force_tracking → observation → robot_config forced a mirrored literal).
from wbt_wrappers_inference.utils.force_deploy_math import TCP_OFFSET_X as _TCP_OFFSET_X  # noqa: E402

TCP_UNIFIED_HAND_OFFSET_X = _TCP_OFFSET_X
VR_3POINT_OFFSETS_TCP = [
    [TCP_UNIFIED_HAND_OFFSET_X, 0.0, 0.0],
    [TCP_UNIFIED_HAND_OFFSET_X, 0.0, 0.0],
    [0.0, 0.0, 0.35],  # head: torso_link +0.35 z, unchanged (mirrors the training literal)
]

# 5-POINT tracking (user 2026-08-27) — the deploy/viz-side mirror of
# wbt_training.config_values.hybrid_unified_5pt_dense.VR_5POINT_{BODIES,OFFSETS_TCP},
# for exp:g1-wbt-5pt-hybrid-unified-anneal (W&B run dlcxc59d). The five points are
# exp:g1-wbt-maskedmimic's `five_point` hover mode after its alias table
# (torso→torso_link, left/right_hand→*_wrist_yaw_link): the 3pt set PLUS both FEET.
#
# Spelled as `[*VR_3POINT_BODIES, ...]` / `[*VR_3POINT_OFFSETS_TCP, ...]` rather than
# re-typed so the 3pt PREFIX property is structural — slots [0:3] of the 5pt position
# vector are byte-identically the 3pt-TCP vector, which is what the training layout
# guarantees and what lets the browser build the 15D table by concatenation.
#
# HANDS STAY ON THE 0.242 m TCP (this is the *unified* family: position and force are
# supervised at ONE point). FEET CARRY NO OFFSET: a hand offset is a physical mounting
# lever and the head offset lifts torso_link to HMD height, but there is no such lever
# for a foot — maskedmimic tracks the raw `*_ankle_roll_link`, and BODY_NAME_TO_MJ_IDX
# above even maps `left_foot_contact_point` onto the same MuJoCo index as
# `left_ankle_roll_link`. A nonzero sole offset would price a point nothing else knows.
VR_5POINT_FEET_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
VR_5POINT_BODIES = [*VR_3POINT_BODIES, *VR_5POINT_FEET_BODIES]
VR_5POINT_FOOT_OFFSET = [0.0, 0.0, 0.0]
VR_5POINT_OFFSETS_TCP = [
    *[list(o) for o in VR_3POINT_OFFSETS_TCP],  # L hand @ TCP, R hand @ TCP, head @ +0.35 z
    list(VR_5POINT_FOOT_OFFSET),
    list(VR_5POINT_FOOT_OFFSET),
]

VR_2POINT_BODIES = list(_req(g1_29dof_wbt.vr_2point_bodies))
VR_2POINT_OFFSETS = [list(o) for o in _req(g1_29dof_wbt.vr_2point_offsets)]
VR_1POINT_BODIES = list(_req(g1_29dof_wbt.vr_1point_bodies))
VR_1POINT_OFFSETS = [list(o) for o in _req(g1_29dof_wbt.vr_1point_offsets)]

CONDITIONABLE_BODIES = list(_req(g1_29dof_wbt.conditionable_bodies))
BODY_NAME_TO_MJ_IDX = g1_29dof_wbt.body_name_to_mj_idx

G1_32BODY_COND_INDICES = [G1_32BODY_NAMES.index(bn) for bn in CONDITIONABLE_BODIES]
G1_30BODY_COND_INDICES = list(_req(g1_29dof_wbt.cond_indices_30body))
