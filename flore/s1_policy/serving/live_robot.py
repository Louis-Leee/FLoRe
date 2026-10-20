"""Live low-level readback helpers for the online HL-v2 planner.

The online planner (``scripts/hl_serving.py``) renders the REAL G1 + yam gripper
mesh from the low-level controller's live joint readback, and displays the exact
force the low level servos on. Both come from the low level's ``RecordTap``
broadcast on ``ws://127.0.0.1:8779`` (opt-in, ``RECORD=1``): one JSON frame per
50 Hz control tick carrying ``dof_pos`` (29 absolute joint angles, rad), the
pelvis-frame ``f_meas`` / ``f_meas_ungated`` buffers the servo consumes, etc.

This module is the *pure* half of that plumbing — the SDK→URDF joint remap and
the RecordTap frame parser — so the load-bearing joint-ordering math is unit
testable WITHOUT viser, a robot, or a live websocket. The viser wiring itself
(subscribe / update_cfg / gui) stays in ``scripts/hl_serving.py``.

Joint-order contract (the trap this module exists to get right):
  * ``dof_pos`` is in the holosoma **SDK ``dof_names`` order** (legs pitch/roll/yaw,
    then waist, then arms) — the same order ``default_dof_angles`` is written in and
    the obs subtracts. See :data:`G1_29DOF_SDK_DOF_NAMES` (mirrored from
    ``holosoma .../config/config_values/robot.py``). The ``weak_motor_joint_index``
    table in that file is a DIFFERENT enumeration (yaw/roll/pitch) for a separate
    purpose and MUST NOT be used to lay out the state vector.
  * The yam URDF (``g1_29dof_yam.urdf``) exposes 31 actuated joints in yourdfpy
    document order — the 29 G1 joints (all present) PLUS the two gripper
    ``*_finger1`` prismatic joints. We remap by NAME, so a subtle order change on
    either side can never silently mis-pose a limb.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The holosoma G1 29-DOF SDK joint order (``robot.py`` ``g1_29dof.dof_names``).
# ``dof_pos`` on the RecordTap wire is in EXACTLY this order. Mirrored here (not
# imported) because the holosoma config package is not importable from FLoRe's
# .venv; kept in lockstep by :func:`build_sdk_to_urdf_index`, which fails loudly
# if a name is missing from the URDF.
G1_29DOF_SDK_DOF_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
N_SDK_DOF = len(G1_29DOF_SDK_DOF_NAMES)  # 29

# The yam URDF gripper prismatic joints (jaw). Their upper limit is 0.0475 m.
LEFT_FINGER_JOINT = "left_gripper_finger1_joint"
RIGHT_FINGER_JOINT = "right_gripper_finger1_joint"
#: The finger joint's upper travel limit (m) — and the end that renders the jaw WIDE OPEN,
#: because the joint OPENS the jaw as it grows (see :func:`jaw_joint_from_width_m`).
JAW_OPEN_M = 0.0475

# ---------------------------------------------------------------------------
# JAW APERTURE -> FINGER JOINT. The joint is an OPENING displacement at HALF the aperture's
# scale: q = 0 is the two fingers TOUCHING (shut), q = JAW_OPEN_M is the jaw wide open. It is
# therefore not one-to-one with the aperture — the fact that is invisible until the jaw is
# animated from a measured width, at which point a naive map renders SATURATED.
#
# Derived from URDF geometry, not assumed (`tests/test_live_robot.py` re-derives it from the
# shipped URDF, so a gripper change cannot leave these constants silently stale):
#
#   * `left_gripper_finger2_joint` MIMICS finger1 (multiplier +1.0) about a MIRRORED parent, so
#     the two fingers travel in OPPOSITE directions along the jaw axis — each finger's origin
#     moves 47.5 mm over the travel, so every metre of q adds TWO metres of aperture;
#   * at q = 0 the two finger MESHES touch (0.05 mm minimum separation) and their body centroids
#     stand 7.6 mm apart along that axis; at q = JAW_OPEN_M they stand 102.6 mm apart. So q OPENS
#     the jaw, and the full travel spans 2 x 0.0475 = 0.095 m — exactly the physical jaw's
#     0..95 mm range (:data:`~flore.s1_policy.data.transforms.GRIPPER_WIDTH_REF_M`).
#
#   aperture(q) = JAW_APERTURE_PER_M * q      so      q = width / JAW_APERTURE_PER_M
#
# ⚠ DO NOT RE-DERIVE THIS FROM THE FINGER LINK ORIGINS — that is exactly how it was got wrong.
# Their y separation runs 90.1 mm -> -4.9 mm across the travel, which fits a *closing* jaw
# (`aperture = 0.0901238 - 2q`, zero residual) and is what commit fc624a6 shipped; its test
# re-derived the same quantity, so it passed. But each finger's body extends ~68 mm PAST its own
# origin on the far side, so the ORIGINS cross while the FINGERS move apart. The resulting
# inverted map drew a shut jaw on a physically open gripper in `run_hl_serving` — on the solid
# live mesh AND on the ghost. Measure the finger BODIES (meshes), never the origins.
JAW_APERTURE_PER_M = 2.0
#: The finger-joint value that renders the jaw FULLY OPEN — the TOP of the travel, because the
#: joint opens. Spelled out as its own name because it is what every seed pose and every "no
#: reading" fallback wants, and the other end (0.0) silently draws a shut hand.
JAW_JOINT_OPEN_M = JAW_OPEN_M
#: ...and the value that renders it fully CLOSED, i.e. the two fingers touching.
JAW_JOINT_CLOSED_M = 0.0


def jaw_joint_from_width_m(width_m: float) -> float:
    """A gripper APERTURE width (m) → the yam URDF finger-joint value (m), clamped to travel.

    ``width_m`` is the full aperture as the gripper bridge reports it and as a take stores it
    (``jaw_m`` / ``grip_cmd_m``), i.e. the same quantity
    :data:`~flore.s1_policy.data.transforms.GRIPPER_WIDTH_REF_M` normalizes — NOT a per-finger
    displacement. Pass it here instead of into ``left_jaw_m``/``right_jaw_m`` directly.

    A non-finite width returns :data:`JAW_JOINT_OPEN_M` (rendered fully OPEN), which is the
    honest default for "this tick has no jaw reading": it keeps the mesh out of a fake grasp, and
    an absent stream then reads as a jaw that never closes rather than one clamped shut.
    """
    w = float(width_m)
    if not np.isfinite(w):
        return JAW_JOINT_OPEN_M
    return float(np.clip(w / JAW_APERTURE_PER_M, JAW_JOINT_CLOSED_M, JAW_OPEN_M))


@dataclass(frozen=True)
class JointRemap:
    """A precomputed name-based SDK(29)→URDF(actuated) remap.

    ``src[j]`` is the index into the 29-element ``dof_pos`` that fills URDF
    actuated slot ``j`` (or -1 if that slot is a gripper finger / unmatched).
    ``left_finger`` / ``right_finger`` are the URDF slots for the jaw prismatic
    joints (or -1 if the URDF has none). ``unmatched_urdf`` lists URDF joints that
    are neither an SDK joint nor a known finger (filled with 0.0) — expected empty
    for the real g1_29dof_yam URDF; surfaced so the caller can warn if not.
    """

    src: np.ndarray  # (n_urdf,) int, -1 where not sourced from dof_pos
    n_urdf: int
    left_finger: int
    right_finger: int
    unmatched_urdf: tuple[str, ...]
    missing_sdf: tuple[str, ...]  # SDK names absent from the URDF (should be empty)


def build_sdk_to_urdf_index(
    urdf_actuated_names: list[str] | tuple[str, ...],
    *,
    sdk_dof_names: tuple[str, ...] = G1_29DOF_SDK_DOF_NAMES,
    left_finger_joint: str = LEFT_FINGER_JOINT,
    right_finger_joint: str = RIGHT_FINGER_JOINT,
) -> JointRemap:
    """Build a name-based remap from the 29-element SDK ``dof_pos`` into the URDF's
    actuated-joint config vector (as returned by ``ViserUrdf.get_actuated_joint_names()``).

    Pure + deterministic. Matching is by exact joint name, so neither side's
    positional order matters — the whole point is to be immune to the pitch/roll/yaw
    ordering trap. Every SDK joint SHOULD map to exactly one URDF slot; any that do
    not are reported in ``missing_sdf`` (a hard inconsistency the caller should flag).
    """
    names = list(urdf_actuated_names)
    n_urdf = len(names)
    sdk_index = {name: i for i, name in enumerate(sdk_dof_names)}
    src = np.full(n_urdf, -1, dtype=np.int64)
    left_finger = right_finger = -1
    unmatched: list[str] = []
    for j, name in enumerate(names):
        if name in sdk_index:
            src[j] = sdk_index[name]
        elif name == left_finger_joint:
            left_finger = j
        elif name == right_finger_joint:
            right_finger = j
        else:
            unmatched.append(name)
    matched_sdk = {names[j] for j in range(n_urdf) if src[j] >= 0}
    missing = tuple(n for n in sdk_dof_names if n not in matched_sdk)
    return JointRemap(
        src=src,
        n_urdf=n_urdf,
        left_finger=left_finger,
        right_finger=right_finger,
        unmatched_urdf=tuple(unmatched),
        missing_sdf=missing,
    )


def remap_dof_to_urdf(
    dof_pos_29: np.ndarray,
    remap: JointRemap,
    *,
    left_jaw_m: float = JAW_JOINT_OPEN_M,
    right_jaw_m: float = JAW_JOINT_OPEN_M,
) -> np.ndarray:
    """Lay a 29-element SDK ``dof_pos`` (rad) into the URDF actuated config vector
    using ``remap``, splicing the gripper JOINT value (metres, clamped to
    [JAW_JOINT_CLOSED_M, JAW_OPEN_M]) into the finger slots. Unmatched URDF slots are left 0.0.

    ⚠ Both jaws take a FINGER-JOINT value, not an aperture width — put a measured/predicted width
    through :func:`jaw_joint_from_width_m` first.

    BOTH default to :data:`JAW_JOINT_OPEN_M` (open), not to 0.0. Neither hand has a jaw stream in
    most callers (this rig only reads the LEFT bridge on ``:8784``), and an unspecified jaw must
    render as the safe, honest "not grasping" pose: a shut hand reads as "it is holding the
    object". 0.0 is the CLOSED end of this joint's travel, so the old defaults drew both hands
    permanently clamped — the RIGHT one on every frame of every session.

    Returns ``(remap.n_urdf,) float64`` ready for ``ViserUrdf.update_cfg``.
    """
    dof = np.asarray(dof_pos_29, dtype=np.float64).reshape(-1)
    if dof.shape[0] != N_SDK_DOF:
        raise ValueError(f"dof_pos must have {N_SDK_DOF} entries, got {dof.shape[0]}")
    out = np.zeros(remap.n_urdf, dtype=np.float64)
    sourced = remap.src >= 0
    out[sourced] = dof[remap.src[sourced]]
    if remap.left_finger >= 0:
        out[remap.left_finger] = float(np.clip(left_jaw_m, JAW_JOINT_CLOSED_M, JAW_OPEN_M))
    if remap.right_finger >= 0:
        out[remap.right_finger] = float(np.clip(right_jaw_m, JAW_JOINT_CLOSED_M, JAW_OPEN_M))
    return out


@dataclass(frozen=True)
class RecordFrame:
    """A parsed low-level RecordTap frame (the fields the planner viz uses).

    ``f_meas`` / ``f_meas_ungated`` are the pelvis-frame per-hand force the servo
    consumes — ``(2,3)`` = [[Lx,Ly,Lz],[Rx,Ly,Lz]] Newtons (post/pre the gate-on-cmd
    zeroing respectively). ``dof_pos`` is ``(29,)`` rad in the SDK order. Missing
    optional fields are ``None`` (the wire only carries ``f_cmd``/``vr_cmd_pos``/
    ``contact_phase`` in some presets).

    ``vr_cmd_pos`` is ``(n,3)`` pelvis-frame metres — the keypoint TARGET the low level's
    policy actually saw this tick, ``n`` = 2/3/5 per preset. It is **post-slew**: the value
    after ``HOLOSOMA_VIVE_MAX_POS_STEP`` was applied, which is what lets
    :class:`~flore.s1_policy.serving.hl_5pt.SlewMonitor` MEASURE the low level's rate cap
    instead of quoting a launcher constant this process cannot read.

    ``vr_cmd_slewed`` is the per-ROW companion: ``(n,)`` bools saying which of those points the
    cap actually COVERED this tick (``HOLOSOMA_VIVE_SLEW_FEET=0`` exempts the 5pt feet, so it
    reads ``[T,T,T,F,F]``). ``None`` when the low level predates the key — treat as all covered,
    the historical behaviour.

    ``ref_joints_live`` (5pt only): did the wire frame's optional ``ref_joints`` — the walk
    skill's gait legs, by name — drive the low level's ``command_lower_body`` obs this tick?
    ``False`` = the frozen idle clip's legs (the historical path); ``None`` = not a 5pt preset or
    a low level that predates the key."""

    seq: int
    running: bool
    servo_on: bool
    vive_armed: bool  # low-level VIVE teleop ARMED (start pressed) — the wire is being applied
    dof_pos: np.ndarray  # (29,)
    root_quat_wxyz: np.ndarray | None  # (4,)
    f_meas: np.ndarray | None  # (2,3) pelvis N — post-gate (servo input)
    f_meas_ungated: np.ndarray | None  # (2,3) pelvis N — pre-gate (true reading)
    f_cmd: np.ndarray | None  # (2,3) pelvis N applied command, or None
    vr_cmd_pos: np.ndarray | None = None  # (n,3) pelvis m, POST-slew applied target, or None
    vr_cmd_slewed: np.ndarray | None = None  # (n,) bool: which rows the slew caps covered, or None
    ref_joints_live: bool | None = None  # 5pt: the wire's ref_joints drove command_lower_body?


def _as_nx3(value) -> np.ndarray | None:
    """``[[x,y,z], ...]`` -> ``(n,3)``, or ``None``. Unlike :func:`_as_2x3` the row count is
    NOT pinned: ``vr_cmd_pos`` carries 2/3/5 points depending on the low level's preset, and a
    consumer that cares checks ``.shape[0]`` against its own wire width."""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)  # raises on ragged / non-numeric JSON
    except (ValueError, TypeError):
        return None
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _as_bool_n(value, n: int) -> np.ndarray | None:
    """``[true, ...]`` -> ``(n,)`` bool, or ``None`` if absent, ragged or not ``n`` long — the mask
    is only meaningful row-for-row against ``vr_cmd_pos``."""
    if value is None or n <= 0:
        return None
    try:
        arr = np.asarray(value)
    except (ValueError, TypeError):
        return None
    if arr.ndim != 1 or arr.shape[0] != n:
        return None
    try:
        return arr.astype(bool)
    except (ValueError, TypeError):
        return None


def _as_2x3(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)  # raises on ragged / non-numeric JSON
    except (ValueError, TypeError):
        return None
    if arr.shape != (2, 3) or not np.all(np.isfinite(arr)):
        return None
    return arr


def parse_record_frame(msg: dict | None) -> RecordFrame | None:
    """Parse one RecordTap JSON dict into a :class:`RecordFrame`, or ``None`` if it
    is not a usable ``record`` frame (wrong type, or ``dof_pos`` missing / wrong
    length / non-finite). Optional force/quat fields degrade to ``None`` rather than
    rejecting the whole frame, so the mesh still renders when a preset omits them.
    """
    if not isinstance(msg, dict) or msg.get("type") != "record":
        return None
    raw_dof = msg.get("dof_pos")
    if raw_dof is None:
        return None
    try:
        dof = np.asarray(raw_dof, dtype=np.float64).reshape(-1)  # raises on ragged/non-numeric
    except (ValueError, TypeError):
        return None
    if dof.shape[0] != N_SDK_DOF or not np.all(np.isfinite(dof)):
        return None

    quat = msg.get("root_quat_wxyz")
    quat_arr: np.ndarray | None = None
    if quat is not None:
        try:
            q = np.asarray(quat, dtype=np.float64).reshape(-1)
        except (ValueError, TypeError):
            q = np.zeros(0)
        if q.shape[0] == 4 and np.all(np.isfinite(q)) and float(np.linalg.norm(q)) > 1e-6:
            quat_arr = q

    vr_cmd_pos = _as_nx3(msg.get("vr_cmd_pos"))
    return RecordFrame(
        seq=int(msg.get("seq", -1)),
        running=bool(msg.get("running", False)),
        servo_on=bool(msg.get("servo_on", False)),
        vive_armed=bool(msg.get("vive_armed", False)),
        dof_pos=dof,
        root_quat_wxyz=quat_arr,
        f_meas=_as_2x3(msg.get("f_meas")),
        f_meas_ungated=_as_2x3(msg.get("f_meas_ungated")),
        f_cmd=_as_2x3(msg.get("f_cmd")),
        vr_cmd_pos=vr_cmd_pos,
        vr_cmd_slewed=(_as_bool_n(msg.get("vr_cmd_slewed"), vr_cmd_pos.shape[0])
                       if vr_cmd_pos is not None else None),
        ref_joints_live=(None if not isinstance(msg.get("ref_joints_live"), bool)
                         else bool(msg.get("ref_joints_live"))),
    )


__all__ = [
    "G1_29DOF_SDK_DOF_NAMES",
    "N_SDK_DOF",
    "LEFT_FINGER_JOINT",
    "RIGHT_FINGER_JOINT",
    "JAW_OPEN_M",
    "JAW_APERTURE_PER_M",
    "JAW_JOINT_OPEN_M",
    "JAW_JOINT_CLOSED_M",
    "jaw_joint_from_width_m",
    "JointRemap",
    "build_sdk_to_urdf_index",
    "remap_dof_to_urdf",
    "RecordFrame",
    "parse_record_frame",
]
