"""The G1's DEPLOY READY POSE, as per-role display homes for the offline viewer.

WHY THIS EXISTS
---------------
An HL-v2 action chunk is **anchor-relative**: the dataset stores
``p_rel = R_ref^T (p_abs - p_ref)`` against the sample's own observation anchor, and the
only thing that turns it back into a place in the world is the pose you decode it against.
On the rig that pose is not a choice — it is where the robot IS when the operator presses
**A**. ``scripts/hl_serving.py`` spells it out (``_display_seed_dof`` +
"RE-SEED THE DECODE ANCHOR"): ``home_anchor.json::default_qpos`` with the 14 arm joints
ZEROED, which is what ``HOLOSOMA_INIT_READY_POSE=1`` pins the low level's INIT target and
frozen hold to.

The offline viewer instead decodes against :data:`~fuji.s1_policy.inference.display.DISPLAY_HOME_POS`
— three hand-picked, uncalibrated spots that exist to make the puck cluster render +Z up.
That is fine for looking at a trajectory, and it was silently wrong the moment a WHOLE-BODY
model started reading those poses, because the display homes are not a robot configuration:

    stored-point x, fore-aft        hands        chest        hands in FRONT of the chest
    display homes                   +0.350       +0.300       +0.050 m
    G1 deploy ready pose            +0.442       -0.004       +0.446 m

A 40 cm error in the hand-versus-torso offset, almost all of it the CHEST being placed 30 cm
too far forward. Motion completion reads the chest as its root channel, so it was told "your
torso is here, your hands are 5 cm in front of it" and answered the only way it could: fold
the arms back, elbows behind the body. Nothing about the completion was wrong — the
conditioning was not a pose a G1 can stand in.

WHAT THIS MODULE DOES AND DOES NOT CHANGE
-----------------------------------------
It supplies the per-role home POSITION (and rotation, derived rather than assumed — it comes
out identity on this rig) that :func:`~fuji.s1_policy.inference.display.resolve_role_geometry`
hangs the scene on. It deliberately does **not** change the rebase MATH: positions stay
TRANSLATED and orientations keep the right-multiplied world delta, because rotating positions
by ``R_ref^T`` — which is exactly what a "fully deploy-faithful" rigid decode would do here,
``R_ready`` being identity — is the wrong-coordinate-system bug ``display.py``'s module
docstring documents and removes. So this fixes the PLACEMENT (what the completion reads for
posture) and leaves the path orientation in the collection-viz convention the operator
already trusts. The trade-off is stated rather than hidden: on the rig, deploy re-expresses
the operator's displacement in the robot's wrist frame and therefore reorients the path (the
measured ~107 deg of ``docs/`` "rig replay reoriented"), and this view does not.

EVERY NUMBER IS DERIVED
-----------------------
``default_qpos`` and the head lever come from holosoma's own ``home_anchor.json``; the arm
joints are selected BY NAME the same way ``hl_serving._ARM_SDK_IDX`` selects them; the
per-body poses are FK'd out of this rig's ``g1_29dof_yam.urdf``; the floor comes from the
lowest ``ankle_roll_link`` plus :data:`SOLE_BELOW_ANKLE_ROLL`. Nothing is typed in. The
result is cross-checked against ``home_anchor.json``'s own baked ``home_pos_pelvis_3pt_tcp``
(holosoma's Pinocchio bake of the SAME configuration), so a URDF change or a re-baked anchor
that disagrees is reported instead of silently re-placing the robot.

Both inputs live in submodules, so every failure returns a REASON and the caller falls back
to the display homes — the same "disable with an explanation, never abort the launch"
contract the motion-completion panel uses.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from fuji.s1_policy.inference.display import TCP_ALONG_WRIST_X_M

#: Repo root (``fuji/s1_policy/inference/ready_pose.py`` -> 4 levels up).
_FUJI_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

#: holosoma's baked VIVE home anchor. Same path ``scripts/hl_serving.py::_HOME_ANCHOR_PATH``
#: reads, spelled once per repo would be better — but importing ``hl_serving`` pulls in viser
#: and a 6k-line rig script, so the path is repeated and pinned equal by a test.
DEFAULT_HOME_ANCHOR_PATH = os.path.join(
    _FUJI_ROOT, "third_party", "holosoma_extensions", "src", "extensions", "wbt_wrappers",
    "wbt_realtime_eval", "wbt_realtime_eval", "vive", "home_anchor.json")

#: This rig's render/FK URDF, the same one every other viewer loads.
DEFAULT_YAM_URDF_PATH = os.path.join(
    _FUJI_ROOT, "third_party", "UMI_collection", "robot_assets", "g1_yam", "g1_29dof_yam.urdf")

#: Substrings that select the 14 ARM joints out of the 29 SDK names. Mirrors
#: ``hl_serving._ARM_SDK_IDX``, and by NAME rather than index for the same reason: an SDK
#: re-ordering must not silently zero the wrong joints.
_ARM_JOINT_KEYS = ("shoulder", "elbow", "wrist")

#: How far the planted SOLE sits below ``ankle_roll_link``. Same measurement as
#: ``mc_frames.SOLE_BELOW_ANKLE_ROLL`` (foot collision cylinders at origin z=-0.025, radius
#: 0.010); duplicated because this module must import with no ``third_party`` on the path, and
#: pinned equal to the submodule's value by a test.
SOLE_BELOW_ANKLE_ROLL = 0.035

#: Which URDF body carries each layout role, and the lever from that body's origin to the
#: point the DATASET stores for it (body frame, metres).
#:
#: * hands: ``wrist_yaw_link`` + the 0.242 m fingertip lever — the SAME
#:   :data:`~fuji.s1_policy.inference.display.TCP_ALONG_WRIST_X_M` the drawn TCP triad uses,
#:   and the same point holosoma's TCP-unified 3-point geometry acts on
#:   (``home_anchor.json::vr_3point_offsets_tcp`` = ``[0.242, 0, 0]``, cross-checked below).
#: * head: ``torso_link`` + the chest keypoint lever, READ from ``home_anchor.json`` rather
#:   than written here, because it is holosoma's conditioning geometry and not ours.
#: * feet: the bare ``*_ankle_roll_link`` ORIGIN, no lever — the ``fivepoint_*`` family's two
#:   extra roles. They are here so :func:`resolve_deploy_anchor` can answer for a FIVE-role
#:   layout, whose chunk is decoded against all five: without them it refuses by name (which is
#:   right for a 3-role planner asking for a foot, and wrong for a 5pt policy that predicts
#:   one). The lever is zero for the same reason :meth:`KeypointFK.load` gives the ankles zero —
#:   holosoma's ``VR_5POINT_OFFSETS_TCP`` is zero there and ``mc_frames.KEYPOINT_OFFSETS``
#:   agrees — so ``pos[role]`` and :attr:`DeployAnchor.foot_pos` are the SAME numbers, reached
#:   by role name and by side respectively. Both spellings are kept: the 5pt WIRE indexes by
#:   side (it predates any foot role) and a LAYOUT indexes by role.
ROLE_BODY: dict[str, str] = {
    "left": "left_wrist_yaw_link",
    "right": "right_wrist_yaw_link",
    "head": "torso_link",
    "left_foot": "left_ankle_roll_link",
    "right_foot": "right_ankle_roll_link",
}
#: Private alias kept because this module already spells ``_ROLE_BODY`` in a dozen places; the
#: public name is what a sibling module (``keypoint_ik``) reads, so "which body carries this
#: role" is answerable without reaching into a private.
_ROLE_BODY = ROLE_BODY
_HAND_LEVER = np.array([TCP_ALONG_WRIST_X_M, 0.0, 0.0], dtype=np.float64)
#: Fallback chest lever, used only if ``home_anchor.json`` has no ``vr_3point_offsets_tcp``.
_CHEST_LEVER_FALLBACK = np.array([0.0, 0.0, 0.35], dtype=np.float64)

#: Bodies whose lowest point defines the floor.
_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")

#: Tolerance (m) for the cross-check against ``home_anchor.json``'s own baked crouch anchor.
#: 1 mm: holosoma bakes with Pinocchio and we FK with yourdfpy off the same URDF, so agreement
#: is a machine-precision question, not a modelling one — a real disagreement is centimetres.
_BAKE_TOL_M = 1e-3


@dataclass(frozen=True)
class DeployAnchor:
    """The pose the robot holds when the operator presses **A**, as display homes.

    ``pos`` / ``rotmat`` are keyed by layout role and give the pose of the point the DATASET
    stores for that role (fingertip TCP for a hand, the chest keypoint for the head), in a
    world frame whose origin is under the pelvis and whose **z=0 is the floor the robot stands
    on** — which is why motion completion can be handed these with ``dz`` pinned to 0 instead
    of deriving a lift from a puck.
    """

    pos: dict[str, np.ndarray]
    rotmat: dict[str, np.ndarray]
    pelvis_z: float
    source: str
    notes: tuple[str, ...] = field(default=())
    #: The 29 SDK joint angles the poses above were FK'd FROM — i.e. the configuration the low
    #: level pins its INIT target and frozen hold to after **A**. Carried because a DENSE
    #: deploy commands joints, not keypoints, so it needs the pose itself and not only where
    #: the pose puts the hands (``fuji/s1_policy/serving/hl_dense.py``). ``None`` only if a
    #: future caller builds this dataclass without one.
    dof: np.ndarray | None = None
    #: Both ``*_ankle_roll_link`` ORIGINS (no lever), keyed ``left``/``right``, in the same
    #: world frame as :attr:`pos`. Carried for the **5pt** low-level family, whose keypoint
    #: vector is ``[L hand, R hand, torso, L foot, R foot]`` — the two feet are not roles the
    #: planner predicts, so its idle wire cannot be built from :attr:`pos` alone. They are
    #: already FK'd here to find the floor, so this is a return rather than extra work. Empty
    #: only if a caller built this dataclass without them.
    foot_pos: dict[str, np.ndarray] = field(default_factory=dict)
    foot_rotmat: dict[str, np.ndarray] = field(default_factory=dict)

    def describe(self) -> str:
        left, head = self.pos.get("left"), self.pos.get("head")
        reach = "" if left is None or head is None else f"  hands {left[0] - head[0]:+.3f} m in front of the chest"
        return (f"G1 deploy ready pose (pelvis z {self.pelvis_z:.4f}, floor z=0)"
                f"{reach}  [{self.source}]")


class KeypointFK:
    """FK 29 SDK joint angles into the five bodies a whole-body keypoint wire carries.

    Exposed as a class because a POST-COMPLETION command anchor needs the same FK the ready
    pose is built from, applied to the robot's **measured** joints: the carried anchor has to
    be leak-bounded against where the robot actually is, and on the 5pt wire "where the robot
    is" only exists as 29 encoder readings until something FKs them. Reusing this instead of
    writing a second FK is the point — two FKs off the same URDF that disagree by a lever is
    precisely the failure ``_cross_check_bake`` exists to catch.

    :meth:`bodies` returns poses keyed by URDF body name, which is exactly what
    ``hl_dense.KeypointReadout.from_bodies`` consumes, so the measured frame goes through the
    identical relativisation as the ready-pose frame with no parallel maths.
    """

    def __init__(self, model, remap, levers: dict[str, np.ndarray]) -> None:
        self._model = model
        self._remap = remap
        self._levers = {body: np.asarray(v, dtype=np.float64).reshape(3)
                        for body, v in levers.items()}
        self._bodies = tuple(sorted(self._levers))

    @property
    def body_names(self) -> tuple[str, ...]:
        return self._bodies

    def lever(self, body: str) -> np.ndarray:
        """The body-frame lever this FK applies to ``body``, or zeros if it applies none.

        Public because an INVERSE kinematics has to target the same point this FK reports
        (:mod:`fuji.s1_policy.inference.keypoint_ik`): a hand's stored point is the fingertip
        TCP while the URDF body is the wrist, so the IK must subtract exactly the lever this
        added. Reading it back rather than re-spelling 0.242 m is the whole reason a second
        copy of that number does not exist.
        """
        return self._levers.get(str(body), np.zeros(3, dtype=np.float64)).copy()

    @classmethod
    def load(cls, *, urdf_path: str | None = None,
             home_anchor_path: str | None = None) -> tuple[KeypointFK | None, str]:
        """``(fk, reason)``. Never raises — a fresh clone has neither submodule."""
        urdf = urdf_path or DEFAULT_YAM_URDF_PATH
        anchor_path = home_anchor_path or DEFAULT_HOME_ANCHOR_PATH
        if not os.path.exists(urdf):
            return None, f"the yam URDF is not checked out ({urdf})"
        try:
            import yourdfpy  # noqa: PLC0415 - optional, viz extra
        except ImportError as exc:
            return None, (f"yourdfpy is unavailable ({exc}); install the viz extra: "
                          'uv pip install --python .venv/bin/python -e ".[viz]"')
        try:
            from fuji.s1_policy.serving.live_robot import (  # noqa: PLC0415
                build_sdk_to_urdf_index,
            )
            chest_lever = _CHEST_LEVER_FALLBACK
            if os.path.exists(anchor_path):
                with open(anchor_path) as handle:
                    levers = json.load(handle).get("vr_3point_offsets_tcp")
                if levers is not None:
                    levers = np.asarray(levers, dtype=np.float64)
                    if levers.shape == (3, 3):
                        chest_lever = levers[2].copy()
            model = yourdfpy.URDF.load(urdf, load_meshes=False, build_scene_graph=True)
            remap = build_sdk_to_urdf_index(model.actuated_joint_names)
        except Exception as exc:  # noqa: BLE001 - any input problem disables the feature
            return None, f"could not load the keypoint FK ({exc!r})"
        # The ankles get NO lever: the 5pt keypoint IS the body origin (holosoma's
        # VR_5POINT_OFFSETS_TCP is zero there, and mc_frames.KEYPOINT_OFFSETS agrees).
        levers = {_ROLE_BODY["left"]: _HAND_LEVER, _ROLE_BODY["right"]: _HAND_LEVER,
                  _ROLE_BODY["head"]: chest_lever}
        levers.update({body: np.zeros(3) for body in _FOOT_BODIES})
        return cls(model, remap, levers), ""

    def bodies(self, dof, *, lift_z: float = 0.0) -> tuple[dict, dict]:
        """``(pos_by_body, rotmat_by_body)`` with each body's lever already applied.

        Poses are in the URDF's **pelvis** frame — holosoma's ``_compute_vr_fk`` frame, which
        has no floor in it — plus ``lift_z`` on z. Pass ``lift_z=anchor.pelvis_z`` to land in
        the same world frame :func:`resolve_deploy_anchor` reports, so a ``KeypointReadout``
        built on that anchor cancels the lift exactly and the ready pose is reproduced
        bit-for-bit rather than approximately (pinned by a test).
        """
        from fuji.s1_policy.serving.live_robot import remap_dof_to_urdf  # noqa: PLC0415

        q = np.asarray(dof, dtype=np.float64).reshape(-1)
        if q.size != 29 or not np.all(np.isfinite(q)):
            raise ValueError(f"dof must be 29 finite SDK joint angles, got shape {q.shape}")
        self._model.update_cfg(remap_dof_to_urdf(q, self._remap))
        lift = np.array([0.0, 0.0, float(lift_z)], dtype=np.float64)
        pos, rot = {}, {}
        for body, lever in self._levers.items():
            matrix = np.asarray(self._model.get_transform(body, "pelvis"), dtype=np.float64)
            rot[body] = matrix[:3, :3].copy()
            pos[body] = matrix[:3, 3] + matrix[:3, :3] @ lever + lift
        return pos, rot

    def pelvis_z(self, dof) -> float:
        """The floor-to-pelvis height of ``dof``: the planted sole under the lowest ankle."""
        pos, _ = self.bodies(dof)
        return SOLE_BELOW_ANKLE_ROLL - min(float(pos[body][2]) for body in _FOOT_BODIES)


def _load_default_qpos(path: str) -> tuple[np.ndarray, dict]:
    with open(path) as handle:
        payload = json.load(handle)
    qpos = np.asarray(payload.get("default_qpos", ()), dtype=np.float64)
    if qpos.ndim != 1 or qpos.size != 29 or not np.all(np.isfinite(qpos)):
        raise ValueError(
            f"{path}: default_qpos must be 29 finite floats, got shape {qpos.shape}")
    return qpos, payload


def _zero_arm_joints(qpos: np.ndarray, sdk_names: list[str] | tuple[str, ...]) -> np.ndarray:
    """``default_qpos`` with the 14 ARM joints set to 0 — the low level's real ready pose."""
    index = [i for i, name in enumerate(sdk_names) if any(k in name for k in _ARM_JOINT_KEYS)]
    if len(index) != 14:
        raise ValueError(
            f"expected 14 arm joints among the 29 SDK names, matched {len(index)}: "
            f"{[sdk_names[i] for i in index]}. The ready pose is defined as "
            "default_dof_angles with those 14 at zero, so a different count means the SDK "
            "name list changed and this selection must be revisited.")
    out = np.asarray(qpos, dtype=np.float64).copy()
    out[index] = 0.0
    return out


def _cross_check_bake(payload: dict, crouch_pos: dict[str, np.ndarray],
                      pelvis_z: float) -> list[str]:
    """Compare our CROUCH FK against holosoma's own baked ``home_pos_pelvis_3pt_tcp``.

    The ready pose zeroes the arms, so it has no baked counterpart to check — but the crouch
    pose it is derived FROM does, and it exercises the identical path (same JSON, same URDF,
    same levers, same FK). Reported, never raised: a stale bake must not stop the viewer, but
    it must not pass unmentioned either.
    """
    baked = payload.get("home_pos_pelvis_3pt_tcp")
    if baked is None:
        return ["home_anchor.json has no home_pos_pelvis_3pt_tcp; the FK could not be "
                "cross-checked against holosoma's own bake"]
    baked = np.asarray(baked, dtype=np.float64)
    if baked.shape != (3, 3):
        return [f"home_pos_pelvis_3pt_tcp has shape {baked.shape}, expected (3, 3); skipped "
                "the FK cross-check"]
    notes: list[str] = []
    for row, role in enumerate(("left", "right", "head")):
        ours = crouch_pos[role] - np.array([0.0, 0.0, pelvis_z])   # back to pelvis-relative
        gap = float(np.linalg.norm(ours - baked[row]))
        if gap > _BAKE_TOL_M:
            notes.append(
                f"ready-pose FK disagrees with home_anchor.json's baked "
                f"home_pos_pelvis_3pt_tcp for {role!r} by {gap * 1e3:.1f} mm "
                f"(ours {np.round(ours, 4).tolist()} vs baked {np.round(baked[row], 4).tolist()}). "
                "The URDF and the bake describe different robots; the FK is used anyway.")
    return notes


def resolve_deploy_anchor(
    roles,
    *,
    home_anchor_path: str | None = None,
    urdf_path: str | None = None,
    zero_arms: bool = True,
) -> tuple[DeployAnchor | None, str]:
    """``(anchor, reason)`` — the ready pose for ``roles``, or ``(None, why not)``.

    ``zero_arms=False`` keeps the raw crouch ``default_qpos`` instead (44 deg nose-down
    forearms), mirroring ``hl_serving --no-zero-arms``: useful only against a low level
    started with ``HOLOSOMA_INIT_READY_POSE=0``.

    Never raises for a missing submodule, a missing file or a missing ``yourdfpy`` — those are
    the normal states of a fresh clone, and the caller's fallback (the display homes) still
    draws a correct trajectory.
    """
    wanted = [str(r) for r in roles]
    missing = [r for r in wanted if r not in _ROLE_BODY]
    if missing:
        return None, (f"no ready-pose body is defined for role(s) {missing}; known roles are "
                      f"{sorted(_ROLE_BODY)}")
    anchor_path = home_anchor_path or DEFAULT_HOME_ANCHOR_PATH
    urdf = urdf_path or DEFAULT_YAM_URDF_PATH
    if not os.path.exists(anchor_path):
        return None, (f"holosoma's home_anchor.json is not checked out ({anchor_path}); "
                      "git submodule update --init --recursive third_party/holosoma_extensions")
    if not os.path.exists(urdf):
        return None, f"the yam URDF is not checked out ({urdf})"
    fk, why = KeypointFK.load(urdf_path=urdf, home_anchor_path=anchor_path)
    if fk is None:
        return None, why
    try:
        from fuji.s1_policy.serving.live_robot import G1_29DOF_SDK_DOF_NAMES  # noqa: PLC0415

        qpos, payload = _load_default_qpos(anchor_path)

        def _by_role(dof: np.ndarray) -> tuple[dict, dict, float, dict, dict]:
            """The one FK, re-keyed from body names to layout roles / foot sides."""
            pelvis_z = fk.pelvis_z(dof)
            pos_body, rot_body = fk.bodies(dof, lift_z=pelvis_z)
            pos = {role: pos_body[body].copy() for role, body in _ROLE_BODY.items()}
            rot = {role: rot_body[body].copy() for role, body in _ROLE_BODY.items()}
            foot_pos = {side: pos_body[body].copy()
                        for side, body in zip(("left", "right"), _FOOT_BODIES)}
            foot_rot = {side: rot_body[body].copy()
                        for side, body in zip(("left", "right"), _FOOT_BODIES)}
            return pos, rot, pelvis_z, foot_pos, foot_rot

        crouch_pos, _, crouch_pelvis_z, _, _ = _by_role(qpos)
        notes = _cross_check_bake(payload, crouch_pos, crouch_pelvis_z)
        dof = _zero_arm_joints(qpos, G1_29DOF_SDK_DOF_NAMES) if zero_arms else qpos
        pos, rot, pelvis_z, foot_pos, foot_rot = _by_role(dof)
    except Exception as exc:  # noqa: BLE001 - any input problem degrades to the display homes
        return None, f"could not derive the ready pose ({exc!r})"

    if not zero_arms:
        notes.append("--no-zero-arms: the raw crouch default_qpos, whose forearms droop 44 deg "
                     "nose-down. Only correct against HOLOSOMA_INIT_READY_POSE=0.")
    return DeployAnchor(
        pos={role: pos[role] for role in wanted},
        rotmat={role: rot[role] for role in wanted},
        pelvis_z=float(pelvis_z),
        source=("home_anchor.default_qpos"
                + (" + 14 arm joints zeroed" if zero_arms else " (raw crouch)")
                + f" -> {os.path.basename(urdf)} FK"),
        notes=tuple(notes),
        dof=np.asarray(dof, dtype=np.float64).copy(),
        foot_pos=foot_pos,
        foot_rotmat=foot_rot,
    ), ""


def world_yaw_alignment(head_rotmat) -> tuple[np.ndarray, float]:
    """``(W, heading_deg)`` — the yaw-only world rotation that turns the take to face +x.

    THE SECOND HALF OF THE PLACEMENT, and the one that decides LEFT from RIGHT. The display
    re-bases every role's ORIENTATION onto its home, so the drawn body always faces world +x —
    but positions are only translated, so they stay in the RECORDED world's axes. Those two
    agree only if the recording happened to be aligned with +x, and on this rig it is not: the
    operator's chest reads **~-170 deg** in the stored frame. So a hand moving to the
    operator's LEFT was drawn moving to the robot's RIGHT — mirrored, next to a robot facing
    the other way, with no readout saying so.

    ``W = Rz(-heading)`` fixes it, taken from the CHEST because that is the body reference (and
    the channel motion completion reads as its root). A layout with NO chest (``left_only``,
    ``both_hands``) falls back to :func:`grip_yaw_alignment` — the hands' grip yaw — which is a
    MEASURED proxy, not a guess: see that function. Yaw-only on purpose: it turns the scene about
    the world's own vertical, so unlike a full ``R_ref^T`` it cannot render a lift as a sideways
    move.

    Returns identity + 0.0 for a non-finite or degenerate input, so a bad anchor degrades to
    the previous behaviour instead of rotating the scene by a garbage angle.
    """
    matrix = np.asarray(head_rotmat, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return np.eye(3), 0.0
    # Heading = the world-frame direction the body's +x points, projected onto the floor.
    # Same projection mc_frames.yaw_of_R uses for the root conditioning, so the scene and the
    # completion's root channel are turned by one consistent number.
    heading = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    if not np.isfinite(heading):
        return np.eye(3), 0.0
    cos, sin = np.cos(-heading), np.sin(-heading)
    return np.array([[cos, -sin, 0.0],
                     [sin, cos, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64), float(np.degrees(heading))


#: A hand whose TCP x-axis is within 20 deg of vertical has no readable yaw (its floor projection
#: is shorter than sin 20 deg): :func:`grip_yaw_alignment` skips it rather than turn the scene by
#: the noise of a near-degenerate atan2.
GRIP_YAW_MIN_FLOOR_PROJECTION = float(np.sin(np.radians(20.0)))


def _floor_yaw(rotmat) -> float | None:
    """Yaw (rad) of a frame's +x axis projected onto the floor, or None if it has no yaw."""
    matrix = np.asarray(rotmat, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None
    if float(np.hypot(matrix[0, 0], matrix[1, 0])) < GRIP_YAW_MIN_FLOOR_PROJECTION:
        return None
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def grip_yaw_alignment(
    hand_rotmats: Mapping[str, np.ndarray],
    home_rotmats: Mapping[str, np.ndarray | None] | None = None,
) -> tuple[np.ndarray | None, float, tuple[str, ...]]:
    """``(W, heading_deg, used_roles)`` — the yaw-only world turn read off the HANDS' grip yaw.

    The fallback for a layout with NO chest role (``left_only`` — the plain ``diffusion_umi``
    preset, e.g. ``wipe_board`` — and ``both_hands``). :func:`world_yaw_alignment` turns the scene
    by the chest's heading; without a chest, the drawn hand's ORIENTATION is still re-based onto
    its home (so it faces the ready pose's way) while its DISPLACEMENT stays in the recorded
    world's axes, and those two disagree by the operator's whole heading. Measured on
    ``wipe_board`` (291 episode starts): the LEFT TCP's x-axis reads **-118.8 deg** (std 7.0),
    so a hand moving along its own forward was drawn going backwards next to a body facing +x —
    the same mirror the chest heading fixes for the 3-tracker layouts.

    Why the hand's yaw is an acceptable stand-in for the operator's heading is also MEASURED,
    not assumed. On the two 3-tracker datasets, at episode start, ``yaw(hand) - yaw(chest)``:

    ==================  ==========  ============  =====  =========
    dataset             hand        median (deg)  std    > 45 deg
    ==================  ==========  ============  =====  =========
    rotate_box 249 ep   LEFT        +1.8          6.6    0 %
    rotate_box 249 ep   RIGHT       +10.0         6.9    0 %
    push_cart 276 ep    LEFT        -8.3          8.0    0 %
    push_cart 276 ep    RIGHT       +8.8          7.7    0 %
    ==================  ==========  ============  =====  =========

    The operator holds the gripper pointing forward when a take starts. It is still the TOOL's
    yaw, so the caller names the source on the page ("LEFT grip yaw (no chest anchor)") and the
    chest is preferred whenever the layout has one.

    Each hand's heading is taken RELATIVE TO ITS HOME orientation (``yaw(R_ref) - yaw(R_home)``,
    ``None`` home = identity; the arms-zeroed ready pose's hands face +x exactly, so today the two
    agree) — that is what makes the turned displacement consistent with the re-based orientation
    for ANY home, not just a level one. Several hands are combined by a CIRCULAR mean. A hand
    whose x-axis is within 20 deg of vertical (:data:`GRIP_YAW_MIN_FLOOR_PROJECTION`) or whose
    home is, is skipped; with no usable hand the result is ``(None, 0.0, ())`` and the caller
    leaves the scene unturned and says so.
    """
    homes = {} if home_rotmats is None else home_rotmats
    used: list[str] = []
    unit = []
    for role in hand_rotmats:
        yaw = _floor_yaw(hand_rotmats[role])
        if yaw is None:
            continue
        home = homes.get(role)
        if home is not None:
            home_yaw = _floor_yaw(home)
            if home_yaw is None:
                continue
            yaw -= home_yaw
        used.append(str(role))
        unit.append((np.cos(yaw), np.sin(yaw)))
    if not unit:
        return None, 0.0, ()
    mean = np.mean(np.asarray(unit, dtype=np.float64), axis=0)
    if float(np.hypot(*mean)) < 1e-9:   # hands pointing opposite ways: no mean heading
        return None, 0.0, ()
    heading = float(np.arctan2(mean[1], mean[0]))
    cos, sin = np.cos(-heading), np.sin(-heading)
    world = np.array([[cos, -sin, 0.0],
                      [sin, cos, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    return world, float(np.degrees(heading)), tuple(used)


# ---------------------------------------------------------------------------
# CLI, shared by the three infer_s1_* entry points
# ---------------------------------------------------------------------------
DISPLAY_FRAMES = ("display-home", "g1-ready-pose")


def add_cli_args(parser) -> None:
    """Add ``--display-frame`` (and its two escape hatches) exactly once, for all viewers."""
    group = parser.add_argument_group("display frame")
    group.add_argument(
        "--display-frame", choices=DISPLAY_FRAMES, default=None,
        help="where the home-anchored scene is HUNG. 'display-home' is the hand-picked upright "
             "home the collection viz and zarr replay use, and is the default UNLESS "
             "--motion-completion is on. 'g1-ready-pose' hangs it on the pose the robot "
             "actually holds after the operator presses A (home_anchor default_qpos, arms "
             "zeroed, FK'd), so the chunk is placed where deploy would place it — required for "
             "a meaningful whole-body completion, hence the implied default there. Pass it "
             "explicitly to override either way.")
    group.add_argument(
        "--ready-pose-no-zero-arms", action="store_true",
        help="with --display-frame g1-ready-pose, use the raw crouch default_qpos instead of "
             "zeroing the 14 arm joints (only for HOLOSOMA_INIT_READY_POSE=0).")
    group.add_argument("--home-anchor-json", default=None,
                       help="override holosoma's home_anchor.json path")


def options_from_args(args) -> dict:
    """``run_viewer(display_frame=...)`` kwargs from a parsed namespace.

    ``--display-frame`` defaults to ``None`` so an UNSET flag can be told apart from an
    explicit one: unset means "display-home", except with ``--motion-completion``, where the
    display homes are not a robot configuration at all and the completion would be conditioned
    on a torso 30 cm from where the robot's torso is. Implying it here rather than in the three
    entry points keeps the two flags' interaction in one place.
    """
    chosen = getattr(args, "display_frame", None)
    if chosen is None:
        # ``--keypoint-ik`` implies it for a DIFFERENT reason than ``--motion-completion``, and
        # both are stated where they are decided. The completion needs it because the display
        # homes put the chest 30 cm from where a G1's chest is, so the POSTURE it answers with is
        # wrong. The five-point IK needs it because the display homes are not one coherent FRAME
        # for five roles at all: the three tracked ones and the two FK'd feet have no stored
        # relation, so hung on their own homes the feet land metres from the body (measured
        # 543 mm of IK residual). Neither is a preference.
        chosen = ("g1-ready-pose"
                  if (getattr(args, "motion_completion", False)
                      or getattr(args, "keypoint_ik", False))
                  else "display-home")
    return {
        "display_frame": str(chosen),
        "zero_arms": not bool(getattr(args, "ready_pose_no_zero_arms", False)),
        "home_anchor_path": getattr(args, "home_anchor_json", None),
    }


__all__ = [
    "DEFAULT_HOME_ANCHOR_PATH",
    "DEFAULT_YAM_URDF_PATH",
    "DISPLAY_FRAMES",
    "ROLE_BODY",
    "DeployAnchor",
    "KeypointFK",
    "SOLE_BELOW_ANKLE_ROLL",
    "add_cli_args",
    "options_from_args",
    "resolve_deploy_anchor",
    "world_yaw_alignment",
    "GRIP_YAW_MIN_FLOOR_PROJECTION",
    "grip_yaw_alignment",
]
