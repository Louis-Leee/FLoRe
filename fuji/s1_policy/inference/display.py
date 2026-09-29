"""Display-frame geometry for the offline S1 inference viewer.

This module owns the numbers that decide WHERE the yam gripper cluster and the three
axis triads sit around a decoded trajectory, and how a stored absolute pose is re-based
so the scene reads +Z up. The viewer owns the viser scene graph; this owns the math.

Deliberately pure numpy — no viser, no yourdfpy, no ``third_party`` import — so every
number here is unit-testable headless, and importing it can never drag the submodule or
a GUI toolkit into a training/serving process.

Two ideas carry the whole module:

**1. The cluster is DATASET-DERIVED, never hardcoded.** The converter composes
``world_T_stored = world_T_puck @ T_tracker_tcp`` and stamps that extrinsic into the
zarr's ``T_tracker_tcp`` attr, so the stored pose means different physical points on
different datasets: with the corrected ``(0.225, 0, -0.05)`` puck->TCP extrinsic the
stored point IS the fingertip TCP, while the older ``(0.21, 0, 0)`` datasets stored a
point ~5 cm high and ~1.5 cm short of it. :func:`resolve_role_geometry` reads whatever
the dataset was built with and places the cluster accordingly, so the TCP triad lands
exactly on the drawn trajectory for a corrected dataset and honestly sits off it for a
pre-fix one (surfaced as a readout rather than hidden).

**2. Home-anchoring matches the collection viz / zarr replay EXACTLY.** The recorded pose is
the physical VIVE puck held in the hand, whose +z points at the floor, so a faithful render
draws the gripper hanging upside down. :func:`home_anchor_positions` /
:func:`home_anchor_rotmats` re-base a pose onto an upright home using the SAME math as
``gripper_only_view.home_anchor`` (the collection viz and ``replay_zarr_viz.py`` both call
it), so the offline-inference ground-truth trajectory is drawn IDENTICALLY to the zarr replay
the operator trusts — that parity is the whole requirement here::

    p_disp  = p_home + (p_cur - p_ref)         (pure TRANSLATION; world directions kept)
    R_disp  = R_home @ (R_cur @ R_ref^T)       (R_home = I)

At the reference this yields exactly ``(p_home, R_home)`` — the anchor frame renders
axis-aligned with +Z up.

The load-bearing property is that positions are TRANSLATED, not rotated: a +z move stays a
+z move, so the grid keeps meaning "the floor" and the whole coordinate system reads the way
it did during collection. An earlier build rotated positions by ``R_ref^T`` (a "rigid" whole-
scene transform); because the puck's ``R_ref`` points +z at the floor, that ``R_ref^T`` tilt
rotated the entire path and every world-vertical motion into the puck frame — the
wrong-coordinate-system bug this convention now avoids. The orientation delta is therefore
the RIGHT-multiplied world delta ``R_cur @ R_ref^T`` (matching the untilted positions), NOT
the left-multiplied ``R_ref^T @ R_cur``.

Translation is rigid on DIFFERENCES and the relative rotation is a conjugation, so both error
metrics the Metrics panel reports — position distance and rotation geodesic — are preserved
exactly; only the path-vs-gripper-axis angle is not (the collection viz and zarr replay do
not preserve it either). What is deliberately NOT preserved is ABSOLUTE world position and
orientation: the scene is moved to a fixed home and the anchor rotated to identity. Turn the
rebase off for a strict raw-frame check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Display triad offsets, expressed in the yam WRIST-YAW link frame (metres).
# ---------------------------------------------------------------------------
# These mirror UMI Collection ``viz/collection_viz.py``'s display constants
# (``_TCP_ALONG_WRIST_X_M`` / ``_TRACKER_TRUE_XYZ_M`` / ``_WRIST_YAW_XYZ_M``) so this
# viewer draws the SAME three triads, at the SAME places, as the MODE=1pt collection viz
# and the zarr replay. They are copied rather than imported because ``collection_viz``
# imports viser at module scope and the submodule may not be checked out at all — this
# module must stay pure. The copy is pinned against the submodule's source by
# ``tests/test_inference_display.py::test_triad_offsets_match_collection_viz``.
#
# DISPLAY ONLY. The converter reads its own puck->TCP extrinsic from
# ``robot_assets/calibration/tcp_extrinsic.json`` and this viewer reads it back off the
# dataset (see :func:`read_stored_extrinsic`); neither consults the constants below. In
# particular ``TCP_ALONG_WRIST_X_M`` is the wrist->TCP lever used to POSITION the drawn
# TCP triad relative to the gripper mesh, and is a different leg from the converter's
# puck->TCP extrinsic — the two must not be conflated or set equal.
#: Wrist→TCP lever (m). 0.242 is the MEASURED wrist_yaw_link→fingertip-midpoint distance and
#: is confirmed by the yam URDF itself: FK puts the fingertip meshes at x ∈ [0.1624, 0.2562]
#: in the wrist frame (closed pinch band from x ≈ 0.2386), so 0.242 lands on the pad while the
#: previous 0.21 sat ~35 mm behind it, inside the finger shanks. Keeping 0.21 made the triad
#: labelled "tcp" miss the drawn fingertips and forced a bogus −3.2 cm "measurement gap" onto
#: the wrist_yaw_link triad; with 0.242 that gap closes to exactly zero by construction (see
#: below), i.e. the triad named wrist_yaw_link finally sits ON the URDF wrist origin.
TCP_ALONG_WRIST_X_M = 0.242

#: Triad name -> its fixed offset in the wrist-yaw frame. Insertion order is the draw
#: order. All three are DERIVED from the two measured legs, so they cannot drift apart:
#:   "tcp"            = wrist + 0.242·x̂                 (the fingertip-midpoint grasp point)
#:   "tracker"        = TCP + (−0.225, 0, +0.05) = wrist + (+0.017, 0, +0.05)
#:                      (the physical VIVE puck origin — the measured puck→TCP extrinsic in
#:                       robot_assets/calibration/tcp_extrinsic.json, read back off the
#:                       dataset by :func:`read_stored_extrinsic`)
#:   "wrist_yaw_link" = TCP − 0.242·x̂ = (0, 0, 0)        (the wrist origin itself)
TRIAD_OFFSETS_WRIST: dict[str, tuple[float, float, float]] = {
    "tcp": (TCP_ALONG_WRIST_X_M, 0.0, 0.0),
    "tracker": (TCP_ALONG_WRIST_X_M - 0.225, 0.0, 0.05),
    "wrist_yaw_link": (TCP_ALONG_WRIST_X_M - 0.242, 0.0, 0.0),
}

#: The measured puck->wrist body-axis relabel (new x = old -y, new y = old -x, new z = old -z).
#: A proper rotation (det=+1) and its own inverse. The physical VIVE puck is bolted onto the tool
#: rotated, so the puck frame and the wrist frame do NOT share an orientation — the mistake that
#: cost a 33 cm / 180 deg error in every stored eef before 2026-08-06. Mirrors UMI Collection
#: ``viz/collection_viz.py::_RAW_RELABEL_M`` and the rotation block of
#: ``robot_assets/calibration/tcp_extrinsic.json``.
PUCK_TO_WRIST_RELABEL = np.array([[0.0, -1.0, 0.0],
                                  [-1.0, 0.0, 0.0],
                                  [0.0, 0.0, -1.0]], dtype=np.float64)

#: Axis-triad draw sizes, so every viewer that reproduces the collection-viz cluster uses
#: ONE set of numbers. Mirrors ``collection_viz`` / ``replay_zarr_viz``: the TCP triad is
#: drawn slightly larger than the other two, which is how an operator tells them apart at a
#: glance. ``(axes_length, axes_radius, origin_radius)`` in metres.
TRIAD_DRAW_SIZES: dict[str, tuple[float, float, float]] = {
    "tcp": (0.08, 0.004, 0.010),
    "tracker": (0.07, 0.0035, 0.009),
    "wrist_yaw_link": (0.07, 0.0035, 0.009),
}


def triad_offsets_from_tcp() -> dict[str, np.ndarray]:
    """Triad name -> offset from the **TCP** (not the wrist), in the wrist/TCP frame.

    :data:`TRIAD_OFFSETS_WRIST` is expressed relative to the WRIST-YAW origin, which is what
    a viewer needs when it already has a wrist frame to parent under (the offline viewer, the
    collection viz, ``replay_zarr_viz``). The ONLINE planner instead drives a frame at the
    **TCP** — that is the point its decoded trajectory and its recorded green path are both
    expressed in — so it needs the same three triads measured from there.

    Rebasing here rather than in the caller keeps ONE source for the geometry: the returned
    ``"tcp"`` entry is exactly zero, which is the property that makes the drawn TCP triad
    coincide with the drawn trajectory (and is asserted by the tests). Note the offsets are
    identical in the wrist and TCP frames up to this translation because the two frames share
    an orientation — the TCP is a pure +x translation along the wrist reach axis.
    """
    tcp = np.asarray(TRIAD_OFFSETS_WRIST["tcp"], dtype=np.float64)
    return {
        name: np.asarray(off, dtype=np.float64) - tcp
        for name, off in TRIAD_OFFSETS_WRIST.items()
    }


#: Per-role upright display home position (metres). Left/right mirror UMI Collection
#: ``gripper_only_view.HOME_WRIST_POSE`` (pinned by a test); ``head`` has no entry there,
#: so it gets its own deterministic spot in front and above the hands rather than being
#: silently stacked on the left hand's home. Like the submodule's, these are not
#: calibrated anything — just comfortable, reproducible places to hang the cluster. The
#: home ROTATION is always identity, which is what makes the anchor frame render +Z up.
#: ``left_foot``/``right_foot`` are the ``fivepoint_*`` family's two FK'd roles. They get a
#: deterministic spot UNDER the hands rather than the left hand's (the no-entry fallback), so a
#: display-home page draws five separated clusters instead of three plus a pile. Like the rest of
#: this table they are not calibrated anything — and unlike the rest they are not even in the
#: same source frame as the hands (see :func:`world_rotmat_by_role`), which is why a five-point
#: BODY is only drawn in the ``g1-ready-pose`` frame.
DISPLAY_HOME_POS: dict[str, tuple[float, float, float]] = {
    "left": (0.35, 0.18, 0.95),
    "right": (0.35, -0.18, 0.95),
    "head": (0.30, 0.0, 1.30),
    "left_foot": (0.30, 0.12, 0.10),
    "right_foot": (0.30, -0.12, 0.10),
}

#: Roles that carry physical gripper GEOMETRY in the yam URDF, and therefore the full mesh
#: + triad cluster. Any other role (the head puck) gets a single triad at its stored point —
#: drawing a "TCP" for a body reference with no gripper would be a fabricated frame.
#:
#: Deliberately NOT ``layout.gripper_roles``: that declares which hands carry this rig's ONE
#: physical gripper+F/T (left only, even in bimanual modes), whereas this is about whether a
#: hand HAS mesh geometry to draw. A ``both_hands`` layout should still render a right-hand
#: gripper — the right hand exists in the URDF and the trajectory is real; only its gripper
#: WIDTH channel is unmeasured, and that already renders as a closed jaw.
GRIPPER_ROLES = ("left", "right")

#: The corrected puck->TCP calibration, quoted in operator-facing notes so a stale dataset's
#: message names the value to re-convert WITH. Mirrors UMI Collection
#: ``robot_assets/calibration/tcp_extrinsic.json``; display/reporting only — the actual
#: geometry always comes from the dataset's own attr.
_CORRECTED_PUCK_TCP_M = (0.225, 0.0, -0.05)

#: How close (mm) the TCP triad must sit to the trajectory before it reads as "on the TCP".
#: 0.05 mm is below any real calibration resolution AND above an f32 round-trip of the
#: corrected extrinsic (~6e-6 mm), so the provenance note and the GUI datum badge that both
#: derive from it can never disagree — a single source, not two hand-picked thresholds.
TCP_RESIDUAL_TOL_MM = 0.05

# ---------------------------------------------------------------------------
# Force arrows
# ---------------------------------------------------------------------------
#: Arrow length per newton (m/N). Matches UMI Collection ``replay_common.WrenchArrows``'s
#: default so the offline viewer, the zarr replay and the raw replay all render a given force
#: at the same visual size — the same parity argument as the triad offsets above. At this rate
#: the 0812 session's peak 18.5 N draws ~18 cm, which is the order of an action chunk's own
#: travel, so it reads as a comparable quantity rather than a decoration.
FORCE_ARROW_SCALE_M_PER_N = 0.01
#: Hard cap on drawn length (m), so one spike cannot fill the scene and hide the trajectory.
FORCE_ARROW_MAX_LEN_M = 0.25
#: Below this the arrow is hidden entirely. 0.2 N is under the Bota's own noise floor on this
#: rig (free-space |F| sits at 0.42 N median), so a hidden arrow means "no contact" rather
#: than "too small to see", and a jittering sub-newton stub never competes with the triads.
FORCE_ARROW_MIN_N = 0.2


def force_arrow_local_segment(
    force_sensor: np.ndarray,
    *,
    scale_m_per_n: float = FORCE_ARROW_SCALE_M_PER_N,
    max_len_m: float = FORCE_ARROW_MAX_LEN_M,
    min_n: float = FORCE_ARROW_MIN_N,
    as_command: bool = True,
) -> np.ndarray | None:
    """One force arrow as a ``(2, 3)`` ``[tail, tip]`` segment in **TCP-LOCAL** coordinates.

    ``force_sensor`` is the raw ``(3,)`` Bota triple as stored/predicted, i.e. a MEASURED
    wrench. TWO conversions are applied here, and only here:

    1. the sensor->TCP relabel (:data:`fuji.utils.force_frame.TCP_FROM_SENSOR`), so the segment
       comes back in the TCP's own axes and a viewer can hand it straight to a viser node
       PARENTED under the frame it already drove to the trajectory point. That is the
       load-bearing choice — composing the world rotation here instead would let the arrow and
       the rendered gripper drift apart the moment the display frame is rebased, which is
       exactly how the wrench arrows once ended up drawn against an unrelabelled frame.
    2. ``as_command`` (DEFAULT ON): the measured->commanded sign flip
       (:data:`fuji.utils.force_frame.CMD_FROM_MEAS_SIGN`), so the arrow shows ``F^cmd`` — the
       direction the arm will PUSH — rather than the reaction the sensor read. Pass
       ``as_command=False`` to draw the raw measurement instead.

    Both live inside this one function for the same reason: a caller that applied either itself
    would double-apply it, and both are involutions, so the second application is INVISIBLE
    (two half turns / two sign flips are the identity — silently back to the original bug).

    Returns ``None`` when the magnitude is below ``min_n`` (caller hides the node) or when the
    input is non-finite — a NaN wrench must not become an arrow pointing at infinity. Neither
    conversion changes the magnitude, so the threshold means the same thing either way.
    """
    from fuji.utils.force_frame import cmd_from_meas_force, tcp_from_sensor_force

    force = np.asarray(force_sensor, dtype=np.float64).reshape(-1)
    if force.shape != (3,):
        raise ValueError(f"force_sensor must be a 3-vector, got shape {np.shape(force_sensor)}")
    if not np.isfinite(force).all():
        return None
    magnitude = float(np.linalg.norm(force))
    if magnitude < float(min_n):
        return None
    length = min(magnitude * float(scale_m_per_n), float(max_len_m))
    drawn = cmd_from_meas_force(force) if as_command else force
    tip = tcp_from_sensor_force(drawn) / magnitude * length
    return np.stack((np.zeros(3), tip), axis=0)


@dataclass(frozen=True)
class RoleGeometry:
    """Where one role's display cluster hangs off its decoded trajectory point.

    ``wrist_offset`` is the wrist-yaw origin expressed RELATIVE TO THE STORED/DECODED
    pose, so a viewer parents a wrist frame at this local offset under a frame driven to
    the trajectory point, then hangs the triads and gripper meshes off the wrist at their
    fixed :data:`TRIAD_OFFSETS_WRIST` poses — reproducing the collection-viz cluster
    around the exact drawn trajectory.

    ``tcp_triad_offset`` is where the TCP triad lands relative to that same trajectory
    point (``wrist_offset`` + the TCP triad's wrist-local offset). It is ZERO exactly when
    the dataset's extrinsic makes the stored point the true fingertip TCP, so its norm is
    the honest "is my trajectory datum actually the TCP?" residual.
    """

    role: str
    wrist_offset: np.ndarray          # (3,)
    tcp_triad_offset: np.ndarray      # (3,)
    home_pos: np.ndarray              # (3,)
    is_gripper_role: bool
    #: Rotation the rebased anchor renders AT — ``R_home`` in
    #: :func:`home_anchor_rotmats`. ``None`` (the default) means identity, which is the
    #: collection-viz convention: the anchor frame comes out axis-aligned with +Z up. It is a
    #: real matrix only for a home taken from a robot configuration
    #: (:mod:`fuji.s1_policy.inference.ready_pose`), where the home ORIENTATION is FK'd rather
    #: than chosen — on this rig that FK returns identity to 2e-4 rad, so passing it through
    #: changes nothing today and cannot silently ignore a future non-level ready pose.
    home_rotmat: np.ndarray | None = None
    #: Wrist-relative ROTATION for the "tracker" (puck) triad — a (3, 3) matrix, ``R_ext^T``
    #: from the dataset extrinsic. The triad is a CHILD of the wrist frame, so without this it
    #: inherits the WRIST/TCP axes instead of showing the physical puck's own orientation.
    #: Identity (the default) for an identity-rotation extrinsic. See
    #: :func:`tracker_triad_local_rotmat`.
    tracker_local_rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))


def read_stored_extrinsic(attrs: Mapping[str, Any] | None) -> tuple[np.ndarray, str | None]:
    """Read a dataset's ``T_tracker_tcp`` puck->stored-point extrinsic.

    Returns ``(T[4, 4], note)``. ``note`` is ``None`` when the attr was a clean
    pure-translation matrix, else a human-readable explanation the caller should surface:
    silently substituting identity for a missing/garbled extrinsic would mis-place the
    whole cluster by centimetres with no visible symptom.
    """
    raw = None if attrs is None else attrs.get("T_tracker_tcp")
    if raw is None:
        return np.eye(4), (
            "dataset has no T_tracker_tcp attr; assuming an IDENTITY puck->TCP extrinsic, "
            "so the stored pose is drawn as the puck itself and the TCP triad will sit "
            "~23 cm off the trajectory"
        )
    try:
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return np.eye(4), f"dataset T_tracker_tcp is not numeric ({raw!r}); assuming identity"
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return np.eye(4), (
            f"dataset T_tracker_tcp is not a finite 4x4 matrix (shape {matrix.shape}); "
            "assuming identity"
        )
    if np.allclose(matrix[:3, :3], np.eye(3), atol=1e-6):
        # An IDENTITY rotation is now the ANOMALY: the corrected extrinsic carries the measured
        # (-y,-x,-z) relabel, so identity means a PRE-2026-08-06 dataset whose stored eef is
        # ~33 cm / 180 deg wrong (a wrist-frame lever applied in the puck frame). The cluster math
        # below is exact for either, but the underlying TRAJECTORY is not — hence the note.
        return matrix, (
            f"dataset T_tracker_tcp rotation is IDENTITY (translation "
            f"{np.round(matrix[:3, 3], 4).tolist()}) — this is a PRE-2026-08-06 conversion whose "
            f"robot0_eef_pos is ~33 cm off in position and 180 deg off in rotation. The cluster is "
            f"drawn correctly for what is STORED, but the stored trajectory is wrong: re-convert."
        )
    return matrix, None


def wrist_pose_from_stored(extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The wrist-yaw frame expressed relative to the STORED/decoded pose: ``(offset, rotmat)``.

    The converter stored ``world_T_stored = world_T_puck @ T_tracker_tcp``, so recovering the
    wrist is the full 4x4 inverse of that composition — NOT a subtraction of translations::

        puck_T_wrist = [ M | -(M @ tracker_offset) ]        (M = PUCK_TO_WRIST_RELABEL)
        stored_T_wrist = inv(T_tracker_tcp) @ puck_T_wrist

    ``M`` is the measured (-y, -x, -z) puck->wrist body-axis relabel: the puck is bolted on
    rotated, so the puck frame and the wrist frame do NOT share an orientation.

    WHY THE FULL INVERSE (the bug this replaces). The old version returned
    ``-(tracker_offset + t)``, adding a PUCK-frame translation to a WRIST-frame vector and
    ignoring the rotation entirely. That is only valid when the two frames' axes coincide,
    which they never do. Measured error: 102.8 mm on a pre-2026-08-06 dataset and 333.5 mm on a
    current-convention one, plus a 180 deg orientation error on the former. The identical
    conflation lived in UMI Collection's ``replay/replay_{raw,zarr}_viz.py`` and was fixed there
    the same way (``wrist_pose_from_stored``); see that submodule's
    ``docs/REPLAY_AND_FIDELITY.md`` section "The authoritative pose chain".

    Returns the rotation too, because for a pre-fix dataset the stored frame is the raw puck and
    the wrist axes are 180 deg away from it — a caller that only applies the offset draws the
    whole cluster with the wrong orientation.
    """
    matrix = np.asarray(extrinsic, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"extrinsic must be 4x4, got shape {matrix.shape}")
    rot, trans = matrix[:3, :3], matrix[:3, 3]
    tracker = np.asarray(TRIAD_OFFSETS_WRIST["tracker"], dtype=np.float64)
    puck_rot_wrist = PUCK_TO_WRIST_RELABEL
    puck_trans_wrist = -(PUCK_TO_WRIST_RELABEL @ tracker)
    return rot.T @ (puck_trans_wrist - trans), rot.T @ puck_rot_wrist


def wrist_offset_from_stored(extrinsic: np.ndarray) -> np.ndarray:
    """Translation-only view of :func:`wrist_pose_from_stored` (the wrist ORIGIN).

    Kept for callers that only place a frame and read its rotation separately. Prefer
    :func:`wrist_pose_from_stored` — dropping the rotation silently mis-orients the cluster on
    any dataset whose stored frame is not the wrist frame.
    """
    return wrist_pose_from_stored(extrinsic)[0]


def tcp_triad_offset_from_stored(extrinsic: np.ndarray) -> np.ndarray:
    """Where the TCP triad lands relative to the stored/decoded pose point.

    Zero exactly when the dataset's extrinsic makes the stored point the true fingertip
    TCP (the corrected ``(0.225, 0, -0.05)`` puck->TCP calibration), so
    ``norm(...)`` is the residual a viewer should report rather than paper over.
    """
    return wrist_offset_from_stored(extrinsic) + np.asarray(TRIAD_OFFSETS_WRIST["tcp"], dtype=np.float64)


def tracker_triad_local_rotmat(extrinsic: np.ndarray) -> np.ndarray:
    """Wrist-relative ROTATION for the "tracker" (puck) triad — a (3, 3) matrix.

    The tracker triad is drawn as a CHILD of the wrist frame, so with no local rotation it
    inherits the WRIST axes. But the physical VIVE puck is bolted on rotated: the converter
    stored ``world_T_stored = world_T_puck @ T_tracker_tcp``, so ``R_puck = R_stored · R_ext^T``
    and the local rotation that renders the triad back at the true puck orientation is
    ``R_ext^T`` (the wrist frame here is at the stored/TCP orientation, which is exactly the
    frame the triad is parented under). IDENTITY for an identity-rotation extrinsic — a
    pre-2026-08-06 dataset or the head puck — which correctly leaves the triad at the stored
    orientation rather than inventing a relabel.

    Mirrors UMI Collection ``replay_zarr_viz.puck_triad_local_quat`` (both are ``R_ext.T``);
    pinned against it by ``tests/test_inference_display.py``. Without it the tracker triad's
    axes render ~180 deg off (the ``(-y, -x, -z)`` relabel) while TCP and the trajectory —
    which do not depend on the extrinsic's rotation — stay correct.
    """
    matrix = np.asarray(extrinsic, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"extrinsic must be 4x4, got shape {matrix.shape}")
    return matrix[:3, :3].T.copy()


def resolve_role_geometry(
    layout,
    file_attrs: Sequence[Mapping[str, Any]] | None,
    home_pose: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[dict[str, RoleGeometry], list[str]]:
    """Resolve every layout role's display cluster geometry from the dataset attrs.

    ``file_attrs`` is one root-attrs mapping per source zarr, in the dataset's file order.
    All files in one session share a rig, so they must agree on the extrinsic; a
    disagreement is reported and the FIRST file's value is used (picking silently would
    mis-place the cluster for every frame drawn from the other files).

    ``home_pose`` overrides where the home-anchored scene is HUNG: ``role -> (pos, rotmat)``,
    normally :func:`fuji.s1_policy.inference.ready_pose.resolve_deploy_anchor`'s G1 ready pose.
    ``None`` (the default) keeps :data:`DISPLAY_HOME_POS` with an identity home rotation, i.e.
    the collection-viz convention, byte for byte. A role the override does not mention keeps
    its display home and says so, rather than being placed by a half-applied frame.

    Returns ``(role -> RoleGeometry, notes)``. ``notes`` is the ordered list of
    human-readable provenance/anomaly lines the caller should print or surface in the GUI.
    """
    attrs_list = [dict(attrs) for attrs in (file_attrs or []) if attrs is not None]
    notes: list[str] = []
    if not attrs_list:
        extrinsic, note = read_stored_extrinsic(None)
        if note:
            notes.append(note)
    else:
        extrinsic, note = read_stored_extrinsic(attrs_list[0])
        if note:
            notes.append(note)
        for index, other in enumerate(attrs_list[1:], start=1):
            other_extrinsic, _ = read_stored_extrinsic(other)
            if not np.allclose(other_extrinsic, extrinsic, atol=1e-9):
                notes.append(
                    f"source file {index} has a DIFFERENT T_tracker_tcp than file 0 "
                    f"({other_extrinsic[:3, 3].tolist()} vs {extrinsic[:3, 3].tolist()}); "
                    "drawing every role with file 0's extrinsic"
                )

    wrist_offset = wrist_offset_from_stored(extrinsic)
    tcp_offset = tcp_triad_offset_from_stored(extrinsic)
    tracker_local = tracker_triad_local_rotmat(extrinsic)
    residual_mm = float(np.linalg.norm(tcp_offset)) * 1000.0
    if residual_mm < TCP_RESIDUAL_TOL_MM:
        notes.append(
            f"stored pose IS the fingertip TCP for this dataset's extrinsic "
            f"{extrinsic[:3, 3].tolist()}; the TCP triad coincides with both trajectories"
        )
    else:
        notes.append(
            f"stored pose is NOT the fingertip TCP for this dataset's extrinsic "
            f"{extrinsic[:3, 3].tolist()}: the TCP triad sits {residual_mm:.1f} mm off the "
            f"drawn trajectory (offset {np.round(tcp_offset, 4).tolist()} m). The corrected "
            f"puck->TCP calibration is {_CORRECTED_PUCK_TCP_M}; re-convert with it to put the "
            f"trajectory on the TCP. (The offset above is the exact discrepancy — do not "
            f"assume a particular cause; several older extrinsics exist.)"
        )

    geometry: dict[str, RoleGeometry] = {}
    for role in layout.roles:
        is_gripper_role = role in GRIPPER_ROLES
        home_rotmat = None
        override = None if home_pose is None else home_pose.get(role)
        if override is not None:
            home = np.asarray(override[0], dtype=np.float64).reshape(3)
            home_rotmat = np.asarray(override[1], dtype=np.float64).reshape(3, 3)
        else:
            if home_pose is not None:
                notes.append(
                    f"role {role!r} is not in the home-pose override; it keeps its display "
                    f"home {DISPLAY_HOME_POS.get(role)} while the other roles move, so the "
                    "inter-role geometry on this page is MIXED")
            home = DISPLAY_HOME_POS.get(role)
            if home is None:
                # An unknown future role still gets a deterministic, non-overlapping home
                # rather than being silently stacked on the left hand's.
                home = DISPLAY_HOME_POS["left"]
                notes.append(f"role {role!r} has no display home; reusing the left hand's {home}")
            home = np.asarray(home, dtype=np.float64)
        geometry[role] = RoleGeometry(
            role=role,
            wrist_offset=wrist_offset.copy(),
            tcp_triad_offset=tcp_offset.copy(),
            home_pos=home,
            is_gripper_role=is_gripper_role,
            tracker_local_rotmat=tracker_local.copy(),
            home_rotmat=home_rotmat,
        )
    return geometry, notes


def check_yaw_only(world_rotmat: np.ndarray) -> np.ndarray:
    """Validate a shared world rotation as YAW-ONLY, and return it as ``(3, 3) float64``.

    The whole safety argument for :func:`home_anchor_positions`' ``world_rotmat`` is that it
    turns the scene about the world's own vertical: a rotation that leaves ``z`` alone cannot
    move a lift into a sideways motion, so the grid keeps meaning the floor and every
    world-vertical quantity survives. A general rotation would not, and it is the exact bug
    this module removed once already — so the constraint is CHECKED here rather than trusted.
    """
    matrix = np.asarray(world_rotmat, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"world_rotmat must be (3, 3), got {matrix.shape}")
    z_fixed = (abs(matrix[2, 2] - 1.0) < 1e-6
               and abs(matrix[2, 0]) < 1e-6 and abs(matrix[2, 1]) < 1e-6
               and abs(matrix[0, 2]) < 1e-6 and abs(matrix[1, 2]) < 1e-6)
    if not z_fixed:
        raise ValueError(
            "world_rotmat must be YAW-ONLY (a rotation about world +z), got\n"
            f"{np.round(matrix, 4)}\nA general rotation tilts world-vertical into the puck "
            "frame, which renders a physical lift as a sideways move — see this module's "
            "docstring on the rebase that was removed for exactly that.")
    if not np.isclose(float(np.linalg.det(matrix)), 1.0, atol=1e-6):
        raise ValueError(f"world_rotmat must be a proper rotation, det={np.linalg.det(matrix)}")
    return matrix


def home_anchor_positions(
    positions: np.ndarray,
    ref_pos: np.ndarray,
    home_pos: np.ndarray,
    ref_rotmat: np.ndarray | None = None,
    home_rotmat: np.ndarray | None = None,
    world_rotmat: np.ndarray | None = None,
) -> np.ndarray:
    """Re-base positions onto an upright home by pure TRANSLATION: ``p_home + (p - p_ref)``.

    This is the collection-viz / zarr-replay convention (``gripper_only_view.home_anchor``):
    the world displacement since the reference is carried onto a fixed home, and world
    DIRECTIONS are preserved — a +z move stays a +z move, so the grid keeps meaning "the
    floor" and the whole coordinate system reads the way it did during collection. That is
    the property an operator relies on and the reason positions are NOT rotated here.

    ``ref_rotmat`` / ``home_rotmat`` are accepted but IGNORED for positions (kept in the
    signature so the position and orientation calls stay symmetric and a caller can pass the
    same reference to both). The translation is rigid on DIFFERENCES, so it exactly preserves
    the prediction-vs-target position error the Metrics panel reports; applied with ONE shared
    reference to the target chunk, the prediction and the recorded tail, it keeps all three
    registered (the tail's first row still coincides with the target's step 0). What it does
    NOT preserve is the path-vs-gripper-axis angle — deliberately, because the collection viz
    and zarr replay this matches do not either, and matching them is the point.

    ``world_rotmat`` (default ``None`` = identity, i.e. the behaviour above unchanged) is a
    **shared world-frame rotation applied to the displacement**: ``p_home + W (p - p_ref)``.
    It exists for one reason — the display re-bases every role's ORIENTATION onto its home
    (so the drawn body faces +x) while leaving positions in the recorded world's axes, and
    those two are only consistent if the recorded world happens to be aligned with the home.
    It is not: this rig's operator faces ~170 deg away from stored +x, so a leftward hand
    motion rendered as a RIGHTWARD one next to a body pointing +x. Passing the yaw-only
    ``W = Rz(-heading)`` (see :func:`fuji.s1_policy.inference.ready_pose.world_yaw_alignment`)
    turns the whole scene to the body's own forward and makes left mean left.

    ⚠️ It must be YAW-ONLY and it must be the SAME matrix for every role. A full
    ``R_ref^T`` would tilt world-vertical into the puck frame — the wrong-coordinate-system
    bug this module's docstring documents and removed — while a per-role matrix would rotate
    each role's displacement differently and silently destroy the inter-role geometry a
    whole-body model reads. Both are rejected by construction here rather than by convention:
    the argument is a single matrix, and its z axis is checked.
    """
    del ref_rotmat, home_rotmat  # positions are translated, never rotated (see docstring)
    positions = np.asarray(positions, dtype=np.float64)
    reference = np.asarray(ref_pos, dtype=np.float64)
    home = np.asarray(home_pos, dtype=np.float64)
    if reference.shape != (3,) or home.shape != (3,):
        raise ValueError(
            f"ref_pos/home_pos must be 3-vectors, got {reference.shape} and {home.shape}"
        )
    delta = positions - reference
    if world_rotmat is not None:
        delta = np.einsum("ij,...j->...i", check_yaw_only(world_rotmat), delta)
    return delta + home


def home_anchor_rotmats(
    rotmats: np.ndarray, ref_rotmat: np.ndarray, home_rotmat: np.ndarray | None = None,
    world_rotmat: np.ndarray | None = None
) -> np.ndarray:
    """Re-base rotations onto an upright display home: ``R_home @ (R_cur @ R_ref^T)``.

    The collection-viz / zarr-replay convention (``gripper_only_view.home_anchor``): the
    per-sample world-frame delta since the reference, ``R_cur @ R_ref^T``, is applied to a
    fixed home orientation (identity by default). At ``R_cur == R_ref`` this is exactly
    ``R_home`` — the anchor frame renders axis-aligned with +Z up instead of the puck's
    physically-correct-but-unreadable +z-into-the-floor.

    The reference-inverse is on the RIGHT (``R_cur @ R_ref^T``), NOT the left — this is the
    load-bearing side. The left-multiplied variant ``R_ref^T @ R_cur`` also lands identity at
    the reference and also preserves the geodesic, so those properties cannot catch a swap;
    but it corresponds to ROTATING positions by ``R_ref^T`` (a whole-scene tilt into the
    puck's floor-ward frame), which is exactly the wrong-coordinate-system bug this convention
    exists to avoid. Positions here are only translated (see :func:`home_anchor_positions`),
    so the matching orientation delta must be the right-multiplied world delta.

    The pred-vs-target relative rotation ``(R_home R_p R_ref^T)(R_home R_t R_ref^T)^T`` is a
    CONJUGATION of ``R_p R_t^T`` by ``R_home R_ref^T``, so its trace — the geodesic ANGLE the
    Metrics panel reports — is unchanged; only the axis it spins about is shown in the display
    frame. Pinned numerically against ``gripper_only_view.home_anchor`` by a test.
    """
    rotmats = np.asarray(rotmats, dtype=np.float64)
    if rotmats.shape[-2:] != (3, 3):
        raise ValueError(f"rotmats must have trailing shape (3, 3), got {rotmats.shape}")
    reference = np.asarray(ref_rotmat, dtype=np.float64)
    if reference.shape != (3, 3):
        raise ValueError(f"ref_rotmat must be (3, 3), got {reference.shape}")
    delta = np.einsum("...ij,kj->...ik", rotmats, reference)  # R_cur @ R_ref^T
    if world_rotmat is not None:
        # CONJUGATE by the same shared world rotation the positions were turned by:
        # W (R R_ref^T) W^T. Conjugation, not a left-multiply — a left-multiply would move the
        # anchor off its home (the drawn body would no longer face +x at step 0) and would no
        # longer be the SAME transform the positions got. As a conjugation it is still exactly
        # the identity at the reference and still preserves the pred-vs-target geodesic the
        # Metrics panel reports, so turning the scene costs no number.
        world = check_yaw_only(world_rotmat)
        delta = np.einsum("ij,...jk,lk->...il", world, delta, world)
    if home_rotmat is None:
        return delta
    home = np.asarray(home_rotmat, dtype=np.float64)
    if home.shape != (3, 3):
        raise ValueError(f"home_rotmat must be (3, 3), got {home.shape}")
    return np.einsum("ij,...jk->...ik", home, delta)


def world_rotmat_by_role(
    shared: np.ndarray | None, roles: Sequence[str], recorded_roles: Sequence[str]
) -> dict[str, np.ndarray | None]:
    """``role -> the world rotation ITS SOURCE FRAME needs``. One matrix per frame, not per role.

    :func:`home_anchor_positions`' warning stands unchanged — the shared yaw-only matrix must
    not become a free per-role choice, because rotating two roles' displacements differently
    destroys the inter-role geometry a whole-body reader depends on. This function exists for
    the ONE case where that argument does not apply: a layout whose roles were not all recorded
    in the same frame.

    ``fivepoint_*`` is that case, and there is no way around it. Its three tracked roles carry
    the operator's RECORDED WORLD displacement (this rig's chest reads -169 deg, which is
    exactly what ``shared`` exists to undo), while its two FEET are
    ``KeypointFK(wb_action_joints)`` — already in the completion's PELVIS frame, where the body
    faces +x by construction. Nothing in the dataset relates the two frames (``wb_root_pose`` is
    written as provenance and never read), so no single matrix can put all five in one frame:
    ``shared`` is right for the three and identity is right for the two, and applying ``shared``
    to a foot would turn its displacement by the operator's heading for no reason. On this rig
    that is not a rounding error — a foot travels up to ~73 mm within a chunk, and 169 deg of
    that is ~145 mm of drawn motion in the wrong direction, plus the same rotation on the yaw
    channel the foot target actually carries (10.7 deg median per episode).

    So the rule is **one matrix per SOURCE FRAME**, and the invariant is preserved in the form
    it was written for: every role sharing a source frame shares a matrix. ``recorded_roles``
    should be :func:`~fuji.s1_policy.data.layout.tracker_roles_required`'s answer, which is the
    dataset's own statement of which roles a tracker recorded — for every other layout it equals
    the full role list, so this returns the shared matrix for all of them and changes nothing.

    ``shared=None`` (raw frame, or the caller opted out) returns ``None`` for every role.
    """
    if shared is None:
        return {str(role): None for role in roles}
    checked = check_yaw_only(shared)
    recorded = {str(role) for role in recorded_roles}
    return {str(role): (checked if str(role) in recorded else None) for role in roles}


__all__ = [
    "DISPLAY_HOME_POS",
    "FORCE_ARROW_MAX_LEN_M",
    "FORCE_ARROW_MIN_N",
    "FORCE_ARROW_SCALE_M_PER_N",
    "GRIPPER_ROLES",
    "RoleGeometry",
    "TCP_ALONG_WRIST_X_M",
    "TCP_RESIDUAL_TOL_MM",
    "TRIAD_DRAW_SIZES",
    "TRIAD_OFFSETS_WRIST",
    "check_yaw_only",
    "force_arrow_local_segment",
    "triad_offsets_from_tcp",
    "home_anchor_positions",
    "home_anchor_rotmats",
    "read_stored_extrinsic",
    "resolve_role_geometry",
    "tcp_triad_offset_from_stored",
    "world_rotmat_by_role",
    "tracker_triad_local_rotmat",
    "wrist_offset_from_stored",
]
