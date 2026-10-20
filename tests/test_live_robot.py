"""Pure tests for the live low-level readback helpers (SDK→URDF joint remap +
RecordTap frame parse) used by the online HL-v2 planner. No viser/robot/websocket."""

from __future__ import annotations

import os

import numpy as np
import pytest

from flore.s1_policy.serving.live_robot import (
    G1_29DOF_SDK_DOF_NAMES,
    JAW_APERTURE_PER_M,
    JAW_JOINT_CLOSED_M,
    JAW_JOINT_OPEN_M,
    JAW_OPEN_M,
    LEFT_FINGER_JOINT,
    N_SDK_DOF,
    RIGHT_FINGER_JOINT,
    build_sdk_to_urdf_index,
    jaw_joint_from_width_m,
    parse_record_frame,
    remap_dof_to_urdf,
)

_UMI = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "UMI_collection",
)
_YAM_URDF = os.path.join(_UMI, "robot_assets", "g1_yam", "g1_29dof_yam.urdf")


# ---------------------------------------------------------------------------
# Name-based remap: the load-bearing anti-mis-pose contract.
# ---------------------------------------------------------------------------
def test_remap_matches_by_name_not_position():
    # A URDF order that is a REVERSED, finger-augmented permutation of the SDK order.
    urdf_names = list(reversed(G1_29DOF_SDK_DOF_NAMES)) + [LEFT_FINGER_JOINT, RIGHT_FINGER_JOINT]
    remap = build_sdk_to_urdf_index(urdf_names)
    assert remap.n_urdf == N_SDK_DOF + 2
    assert remap.missing_sdf == ()  # every SDK joint found
    assert remap.unmatched_urdf == ()  # only the two fingers are non-SDK
    assert remap.left_finger == N_SDK_DOF  # appended first
    assert remap.right_finger == N_SDK_DOF + 1

    # A dof_pos where entry i == i lets us prove each URDF slot pulled the RIGHT joint.
    dof = np.arange(N_SDK_DOF, dtype=np.float64)
    out = remap_dof_to_urdf(dof, remap, left_jaw_m=0.02, right_jaw_m=0.01)
    for j, name in enumerate(urdf_names):
        if name in (LEFT_FINGER_JOINT, RIGHT_FINGER_JOINT):
            continue
        sdk_i = G1_29DOF_SDK_DOF_NAMES.index(name)
        assert out[j] == sdk_i, f"slot {j} ({name}) pulled wrong joint"
    assert out[remap.left_finger] == 0.02
    assert out[remap.right_finger] == 0.01


def test_remap_clamps_jaw_and_zeros_unmatched():
    urdf_names = list(G1_29DOF_SDK_DOF_NAMES) + [LEFT_FINGER_JOINT, "some_extra_passive_joint"]
    remap = build_sdk_to_urdf_index(urdf_names)
    assert remap.unmatched_urdf == ("some_extra_passive_joint",)
    assert remap.right_finger == -1  # no right finger in this URDF
    dof = np.ones(N_SDK_DOF, dtype=np.float64)
    out = remap_dof_to_urdf(dof, remap, left_jaw_m=99.0)  # over the limit
    assert out[remap.left_finger] == JAW_OPEN_M  # clamped
    # The unmatched passive joint slot stays 0.
    assert out[urdf_names.index("some_extra_passive_joint")] == 0.0


def test_remap_rejects_wrong_dof_length():
    remap = build_sdk_to_urdf_index(list(G1_29DOF_SDK_DOF_NAMES))
    with pytest.raises(ValueError, match="29 entries"):
        remap_dof_to_urdf(np.zeros(28), remap)


def test_both_jaws_default_to_open():
    """An UNSPECIFIED jaw must render OPEN, on BOTH hands.

    The rig has one gripper stream (LEFT, :8784), so the RIGHT finger joint is whatever the
    default puts there for an entire session. It used to be 0.0 — the CLOSED end of this joint's
    travel — so the right hand was drawn permanently clamped shut on every frame, and the left one
    was too until its first fresh width arrived.
    """
    urdf_names = list(G1_29DOF_SDK_DOF_NAMES) + [LEFT_FINGER_JOINT, RIGHT_FINGER_JOINT]
    remap = build_sdk_to_urdf_index(urdf_names)
    out = remap_dof_to_urdf(np.zeros(N_SDK_DOF), remap)
    assert out[remap.left_finger] == JAW_JOINT_OPEN_M
    assert out[remap.right_finger] == JAW_JOINT_OPEN_M
    assert JAW_JOINT_OPEN_M != JAW_JOINT_CLOSED_M, "the two travel ends must not be the same value"


def test_real_yam_urdf_has_every_sdk_joint():
    """Contract against the ACTUAL deploy URDF: all 29 SDK joints resolve, and both
    gripper fingers are found (so the live mesh can never silently drop a limb)."""
    yourdfpy = pytest.importorskip("yourdfpy")
    if not os.path.isfile(_YAM_URDF):
        pytest.skip("g1_29dof_yam.urdf not present")
    urdf = yourdfpy.URDF.load(_YAM_URDF, load_meshes=False, build_scene_graph=False)
    actuated = list(urdf.actuated_joint_names)
    remap = build_sdk_to_urdf_index(actuated)
    assert remap.missing_sdf == (), f"URDF is missing SDK joints: {remap.missing_sdf}"
    assert remap.left_finger >= 0 and remap.right_finger >= 0
    # Every SDK joint maps to a distinct URDF slot.
    sourced = remap.src[remap.src >= 0]
    assert sorted(sourced.tolist()) == list(range(N_SDK_DOF))


# ---------------------------------------------------------------------------
# JAW APERTURE -> FINGER JOINT. The bug this pins: the finger joint is an OPENING
# displacement at HALF the aperture's scale. Getting the direction wrong renders the jaw
# BACKWARDS, which is what `run_hl_serving` shipped between fc624a6 and this fix — a shut
# hand drawn over a physically open gripper, on the solid mesh AND on the violet ghost.
#
# ⚠ THE TRAP THAT LET THAT SHIP, because it lived in THIS FILE: the previous version of
# `test_jaw_constants_match_the_real_urdf` re-derived the "aperture" as the y separation of the
# two finger LINK ORIGINS. That quantity runs 90.1 mm -> -4.9 mm across the travel and fits a
# closing jaw with zero residual — so the test agreed with the inverted map and passed. Each
# finger's BODY extends ~68 mm past its own origin on the far side, so the origins cross while
# the fingers move apart. Any re-derivation here must measure the finger MESHES.
# ---------------------------------------------------------------------------
def test_jaw_joint_opens_with_travel_and_is_half_the_aperture():
    """The direction, which is the whole bug: a WIDER aperture is a LARGER joint value."""
    wide = jaw_joint_from_width_m(0.094)
    narrow = jaw_joint_from_width_m(0.044)
    assert wide > narrow, "a wider jaw must map to MORE finger travel"
    assert wide == pytest.approx(0.047, abs=1e-3), "94 mm is (nearly) fully open -> top of travel"
    assert narrow == pytest.approx(0.022, abs=1e-3), "a 44 mm grasp sits mid-travel"
    # HALF the aperture, because finger2 mimics finger1 about a mirrored parent: feeding the
    # width straight in would run 94 mm past the 47.5 mm limit and saturate the whole range.
    assert jaw_joint_from_width_m(0.06) == pytest.approx(0.06 / JAW_APERTURE_PER_M, abs=1e-9)
    assert jaw_joint_from_width_m(0.2) == JAW_OPEN_M, "a width past full open must clamp, not wrap"


def test_jaw_joint_is_monotone_and_spans_visible_travel():
    """A real take's 43.8-94.3 mm range must sweep a visible fraction of the travel."""
    widths = np.linspace(0.0438, 0.0943, 25)
    q = np.array([jaw_joint_from_width_m(w) for w in widths])
    assert np.all(np.diff(q) >= -1e-12), "must be non-decreasing in width"
    assert q.max() - q.min() > 0.015, f"sweep only {q.max() - q.min():.4f} m of travel"


def test_jaw_joint_clamps_to_travel_and_survives_nonfinite():
    for w in (-1.0, 0.0, 0.2, 5.0):
        assert JAW_JOINT_CLOSED_M <= jaw_joint_from_width_m(w) <= JAW_OPEN_M
    assert jaw_joint_from_width_m(0.0) == JAW_JOINT_CLOSED_M, "zero width IS the shut jaw"
    # A missing reading renders OPEN rather than faking a grasp.
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert jaw_joint_from_width_m(bad) == JAW_JOINT_OPEN_M


def _left_finger_body_verts(urdf, link_name):
    """World-frame-independent visual mesh vertices of a finger link, in its OWN link frame."""
    trimesh = pytest.importorskip("trimesh")
    out = []
    for vis in urdf.link_map[link_name].visuals:
        mesh = getattr(vis.geometry, "mesh", None)
        fn = getattr(mesh, "filename", None) if mesh is not None else None
        if fn is None:
            continue
        tm = trimesh.load(urdf._filename_handler(fn), force="mesh")
        T = vis.origin if vis.origin is not None else np.eye(4)
        out.append((T[:3, :3] @ np.asarray(tm.vertices).T).T + T[:3, 3])
    if not out:
        pytest.skip(f"{link_name} has no loadable visual mesh")
    return np.vstack(out)


def test_jaw_constants_match_the_real_urdf():
    """RE-DERIVE the aperture map from the shipped URDF's finger BODIES, so a gripper change
    cannot leave JAW_APERTURE_PER_M / JAW_JOINT_OPEN_M silently stale.

    Measured on the finger MESH CENTROIDS along the jaw's motion axis — not the link origins,
    which cross and therefore fit the map backwards (see the block comment above). Also pins the
    structural fact the halving rests on: finger2 MIMICS finger1 with multiplier +1, about a
    mirrored parent, so the two fingers travel in OPPOSITE directions.
    """
    yourdfpy = pytest.importorskip("yourdfpy")
    pytest.importorskip("trimesh")
    if not os.path.isfile(_YAM_URDF):
        pytest.skip("g1_29dof_yam.urdf not present")
    urdf = yourdfpy.URDF.load(_YAM_URDF, load_meshes=True, build_scene_graph=True)
    mimic = urdf.joint_map["left_gripper_finger2_joint"].mimic
    assert mimic is not None and mimic.joint == LEFT_FINGER_JOINT
    assert mimic.multiplier == pytest.approx(1.0)

    names = list(urdf.actuated_joint_names)
    i = names.index(LEFT_FINGER_JOINT)
    assert "left_gripper_finger2_joint" not in names, "finger2 must be mimic-driven, not actuated"
    hand = "left_rubber_hand"
    V1 = _left_finger_body_verts(urdf, "left_gripper_finger1_link")
    V2 = _left_finger_body_verts(urdf, "left_gripper_finger2_link")

    def centroids(q):
        cfg = np.zeros(len(names))
        cfg[i] = q
        urdf.update_cfg(cfg)
        T1 = urdf.get_transform("left_gripper_finger1_link", hand)
        T2 = urdf.get_transform("left_gripper_finger2_link", hand)
        return T1[:3, :3] @ V1.mean(0) + T1[:3, 3], T2[:3, :3] @ V2.mean(0) + T2[:3, 3]

    # The jaw's motion axis, taken from how the fingers actually move (no hardcoded axis).
    a0, b0 = centroids(0.0)
    a1, _ = centroids(JAW_OPEN_M)
    axis = (a1 - a0) / np.linalg.norm(a1 - a0)

    qs = np.linspace(0.0, JAW_OPEN_M, 11)
    sep = np.array([float((a - b) @ axis) for a, b in (centroids(q) for q in qs)])
    slope, intercept = np.polyfit(qs, sep, 1)
    assert np.max(np.abs(sep - (intercept + slope * qs))) < 1e-9, "aperture is not affine in q"
    # THE POLARITY: the finger bodies move APART as the joint grows.
    assert slope > 0, "the finger joint must OPEN the jaw, not close it"
    assert slope == pytest.approx(JAW_APERTURE_PER_M, abs=1e-6)
    # ...and q = 0 is the SHUT jaw: the bodies are near-coincident there, two orders of
    # magnitude below the 95 mm they reach at full travel.
    assert abs(intercept) < 0.010, f"q=0 should be a shut jaw, got {intercept * 1000:.1f} mm apart"
    assert sep[-1] - sep[0] == pytest.approx(0.095, abs=1e-6), (
        "full travel must span the physical jaw's 0..95 mm range (GRIPPER_WIDTH_REF_M)")
    # ...and the round trip: a width put through the map renders at that width.
    for w in (0.05, 0.07, 0.085):
        rendered = float(np.subtract(*centroids(jaw_joint_from_width_m(w))) @ axis) - intercept
        assert rendered == pytest.approx(w, abs=1e-6)


# ---------------------------------------------------------------------------
# RecordTap frame parse.
# ---------------------------------------------------------------------------
def _good_msg(**over):
    msg = {
        "type": "record",
        "seq": 7,
        "running": True,
        "servo_on": True,
        "vive_armed": True,
        "dof_pos": list(np.linspace(-0.5, 0.5, N_SDK_DOF)),
        "dof_vel": [0.0] * N_SDK_DOF,
        "root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "f_meas": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "f_meas_ungated": [[1.5, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "f_cmd": [[0.0, 0.0, 8.0], [0.0, 0.0, 0.0]],
    }
    msg.update(over)
    return msg


def test_parse_good_frame():
    fr = parse_record_frame(_good_msg())
    assert fr is not None
    assert fr.seq == 7 and fr.running and fr.servo_on and fr.vive_armed
    assert parse_record_frame(_good_msg(vive_armed=False)).vive_armed is False  # default-safe
    assert fr.dof_pos.shape == (N_SDK_DOF,)
    np.testing.assert_allclose(fr.root_quat_wxyz, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(fr.f_meas, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(fr.f_meas_ungated[0, 0], 1.5)
    np.testing.assert_allclose(fr.f_cmd[0, 2], 8.0)


def test_parse_rejects_non_record_and_bad_dof():
    assert parse_record_frame(None) is None
    assert parse_record_frame({"type": "ft", "dof_pos": [0.0] * N_SDK_DOF}) is None
    assert parse_record_frame(_good_msg(dof_pos=[0.0] * (N_SDK_DOF - 1))) is None  # short
    bad = _good_msg()
    bad["dof_pos"][3] = float("nan")
    assert parse_record_frame(bad) is None


def test_record_frame_carries_the_slew_coverage_mask_row_for_row():
    """`vr_cmd_slewed` says which `vr_cmd_pos` rows the low level's caps covered this tick; it is
    only meaningful against those rows, so a wrong-width or absent mask parses to None (= all
    covered, the pre-key behaviour) rather than a guess."""
    msg = _good_msg()
    msg["vr_cmd_pos"] = [[0.0, 0.0, 0.0]] * 5
    msg["vr_cmd_slewed"] = [True, True, True, False, False]
    fr = parse_record_frame(msg)
    assert fr.vr_cmd_slewed is not None and fr.vr_cmd_slewed.dtype == bool
    assert fr.vr_cmd_slewed.tolist() == [True, True, True, False, False]
    msg["vr_cmd_slewed"] = [True, False]
    assert parse_record_frame(msg).vr_cmd_slewed is None
    del msg["vr_cmd_slewed"]
    assert parse_record_frame(msg).vr_cmd_slewed is None
    msg["vr_cmd_slewed"] = [1, 1, 1, 0, 0]          # JSON ints are fine
    assert parse_record_frame(msg).vr_cmd_slewed.tolist() == [True, True, True, False, False]
    msg.pop("vr_cmd_pos")
    msg["vr_cmd_slewed"] = [True] * 5
    fr = parse_record_frame(msg)
    assert fr.vr_cmd_pos is None and fr.vr_cmd_slewed is None   # no rows to be coverage OF


def test_parse_is_exception_safe_on_ragged_and_nonnumeric():
    """The parser is documented to REJECT malformed frames by returning None, never by
    raising — a corrupt/foreign 'record' frame must not crash the render/status threads."""
    # Ragged / non-numeric dof_pos -> None (not a raised ValueError).
    assert parse_record_frame(_good_msg(dof_pos="garbage")) is None
    assert parse_record_frame(_good_msg(dof_pos=[[1, 2], [3]])) is None
    # Ragged / non-numeric force fields degrade to None but keep the frame (dof still good).
    fr = parse_record_frame(_good_msg(f_meas=[[1, 2, 3], [4, 5]], f_cmd="nope"))
    assert fr is not None and fr.f_meas is None and fr.f_cmd is None
    # Non-numeric quat degrades to None, frame still parses.
    fr2 = parse_record_frame(_good_msg(root_quat_wxyz=["a", "b", "c", "d"]))
    assert fr2 is not None and fr2.root_quat_wxyz is None


def test_parse_degrades_optional_fields_to_none():
    # A frame with dof_pos but no force / a malformed quat still parses (mesh renders).
    fr = parse_record_frame(_good_msg(f_meas=None, f_cmd=None, root_quat_wxyz=[0, 0, 0, 0]))
    assert fr is not None
    assert fr.f_meas is None and fr.f_cmd is None
    assert fr.root_quat_wxyz is None  # zero-norm quat rejected
    # A wrong-shape force is dropped, not raised.
    assert parse_record_frame(_good_msg(f_meas=[1.0, 2.0, 3.0]))  # (3,) not (2,3)
    assert parse_record_frame(_good_msg(f_meas=[1.0, 2.0, 3.0])).f_meas is None


def test_record_frame_carries_ref_joints_live_as_a_tri_state():
    """5pt only: did the wire's optional `ref_joints` (the walk skill's legs) drive the low level's
    command_lower_body this tick? True / False, and None for a preset or build that never says."""
    msg = _good_msg()
    assert parse_record_frame(msg).ref_joints_live is None
    msg["ref_joints_live"] = True
    assert parse_record_frame(msg).ref_joints_live is True
    msg["ref_joints_live"] = False
    assert parse_record_frame(msg).ref_joints_live is False
    msg["ref_joints_live"] = None
    assert parse_record_frame(msg).ref_joints_live is None
    msg["ref_joints_live"] = "yes"          # not a bool: unknown, never a guess
    assert parse_record_frame(msg).ref_joints_live is None
