"""Contract tests for the G1 DEPLOY READY POSE as the offline viewer's display home.

An HL-v2 chunk is anchor-relative, so the pose it is decoded against is not a cosmetic
choice — it decides where the hands ARE, and therefore what a whole-body completion is
conditioned on. What is pinned here:

* **the number that was wrong.** The display homes leave the hands 5 cm in front of the chest;
  the robot's ready pose leaves them 45 cm in front. That 40 cm is what folded the completion's
  arms behind the torso, so it gets an explicit assertion rather than living only in a comment.
* **it is DERIVED, not typed.** Every value comes from holosoma's ``home_anchor.json`` and this
  rig's URDF, and it is cross-checked against holosoma's own baked anchor — so a re-baked
  anchor or a changed URDF is reported instead of silently re-placing the robot.
* **the same 14 arm joints as the rig.** ``hl_serving`` selects them by NAME; a drift here
  would place the viewer at a pose the low level does not hold.
* **it degrades, never aborts.** Both inputs live in submodules.
* **default OFF.** The plain viewers must be byte-for-byte unchanged, which means the display
  homes stay the default and ``RoleGeometry.home_rotmat`` stays ``None``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flore.s1_policy.inference import ready_pose as RP
from flore.s1_policy.inference.display import (
    DISPLAY_HOME_POS, RoleGeometry, resolve_role_geometry,
)

_REPO = Path(__file__).resolve().parents[1]
_ROLES = ("left", "right", "head")


class _Layout:
    def __init__(self, roles=_ROLES) -> None:
        self.roles = tuple(roles)


def _anchor_or_skip(**kw):
    anchor, reason = RP.resolve_deploy_anchor(_ROLES, **kw)
    if anchor is None:
        pytest.skip(f"deploy ready pose unavailable: {reason}")
    return anchor


# --------------------------------------------------------------------------- the number
def test_the_ready_pose_puts_the_hands_well_in_front_of_the_chest():
    """THE BUG THIS FIXES, as a number.

    Motion completion reads the chest as its root channel, so what matters is not where the
    hands are in absolute terms but where they are RELATIVE to the torso. The display homes
    say +0.05 m; a G1 holding its ready pose says +0.45 m. Conditioned on the first, the model
    can only fold the arms back — measured 174 mm of elbow behind the shoulder, vs 9 mm here.
    """
    anchor = _anchor_or_skip()
    ready_reach = float(anchor.pos["left"][0] - anchor.pos["head"][0])
    display_reach = float(DISPLAY_HOME_POS["left"][0] - DISPLAY_HOME_POS["head"][0])
    assert ready_reach > 0.40, ready_reach
    assert display_reach < 0.10, display_reach
    # The chest is the term that moved: the hands shift ~9 cm, the chest ~30 cm.
    assert abs(anchor.pos["head"][0] - DISPLAY_HOME_POS["head"][0]) > 0.25
    assert abs(anchor.pos["left"][0] - DISPLAY_HOME_POS["left"][0]) < 0.15


def test_the_ready_pose_stands_on_the_floor_so_dz_needs_no_guessing():
    """z=0 is the floor by construction — which is why MC's lift is PINNED to 0 there.

    The alternative (``auto_dz_chest``) targets ``RIG_CHEST_STAND_Z`` = 1.30 while this pose's
    own chest keypoint sits ~15 cm lower, so deriving a lift here would raise the whole take
    off the robot. The pelvis landing in the G1's standing band is the check that the floor
    reference is real and not an arbitrary offset.
    """
    anchor = _anchor_or_skip()
    assert 0.70 < anchor.pelvis_z < 0.85, anchor.pelvis_z
    # Hands below the chest, chest below the head-height a standing G1's display home assumes.
    assert anchor.pos["left"][2] < anchor.pos["head"][2]
    assert 1.0 < float(anchor.pos["head"][2]) < 1.25, anchor.pos["head"]


def test_the_two_hands_are_symmetric_and_the_home_rotation_is_level():
    """Arms zeroed means forearms LEVEL and mirrored — the property that names this pose.

    The rotation coming out identity is what lets the display frame keep its existing
    orientation math untouched; it is asserted rather than assumed so a future non-level ready
    pose shows up here instead of as a silently wrong render.
    """
    anchor = _anchor_or_skip()
    left, right = anchor.pos["left"], anchor.pos["right"]
    np.testing.assert_allclose(left[[0, 2]], right[[0, 2]], atol=1e-9)
    assert left[1] > 0 > right[1]
    assert abs(abs(left[1]) - abs(right[1])) < 1e-3
    for role in _ROLES:
        np.testing.assert_allclose(anchor.rotmat[role], np.eye(3), atol=1e-3)


def test_zeroing_the_arms_actually_changes_the_pose():
    """``--ready-pose-no-zero-arms`` must not be a synonym.

    The crouch ``default_qpos`` droops the forearms 44 deg nose-down; the low level pins the
    arms-zeroed pose under ``HOLOSOMA_INIT_READY_POSE=1``. Using the wrong one is a real
    placement error (the rig measures 310 mm/hand between them at the VR contact point), so
    the two must be distinguishable and the crouch one must say what it is.
    """
    zeroed = _anchor_or_skip()
    crouch = _anchor_or_skip(zero_arms=False)
    assert float(np.linalg.norm(zeroed.pos["left"] - crouch.pos["left"])) > 0.15
    assert any("no-zero-arms" in note for note in crouch.notes), crouch.notes
    assert not any("no-zero-arms" in note for note in zeroed.notes)


# --------------------------------------------------------------------------- derived, not typed
def test_the_fk_agrees_with_holosomas_own_baked_anchor():
    """The crouch pose is FK'd here and BAKED there; silence means they agree to 1 mm.

    ``home_anchor.json::home_pos_pelvis_3pt_tcp`` is holosoma's Pinocchio bake of the same
    configuration through the same levers, so this is the one check that says the viewer and
    the rig are talking about the same robot. Any disagreement is surfaced as a note, and this
    asserts there are none — which is also what pins the 0.242 m hand lever and the chest's
    (0, 0, 0.35) against holosoma's ``vr_3point_offsets_tcp``.
    """
    anchor = _anchor_or_skip()
    assert anchor.notes == (), anchor.notes


def test_the_arm_joint_selection_matches_the_rig_by_name():
    """14 joints, chosen by name — the same set ``hl_serving._ARM_SDK_IDX`` zeroes.

    Indices would be a silent trap across an SDK re-order, so both sides match substrings; this
    pins that the substrings still select exactly the 14 arm joints and no leg or waist joint.
    """
    from flore.s1_policy.serving.live_robot import G1_29DOF_SDK_DOF_NAMES

    picked = [n for n in G1_29DOF_SDK_DOF_NAMES if any(k in n for k in RP._ARM_JOINT_KEYS)]
    assert len(picked) == 14, picked
    assert not any(k in n for n in picked for k in ("hip", "knee", "ankle", "waist"))
    # And the selector itself refuses a name list it cannot make 14 out of.
    with pytest.raises(ValueError, match="expected 14 arm joints"):
        RP._zero_arm_joints(np.zeros(29), ["hip"] * 29)


def test_the_sole_offset_matches_the_motion_completion_module():
    """Duplicated constant, pinned — the floor must mean the same thing on both sides.

    ``ready_pose`` cannot import ``mc_frames`` (it must work with no ``third_party`` on the
    path), so the value is copied. If the two drift, the robot's feet stop touching the grid
    the chunk is drawn on and nothing else says so.
    """
    mc_frames = _REPO / "third_party" / "UMI_collection" / "motion_completion" / "mc_frames.py"
    if not mc_frames.exists():
        pytest.skip("UMI_collection submodule is not checked out")
    import re

    text = mc_frames.read_text()
    match = re.search(r"^SOLE_BELOW_ANKLE_ROLL\s*=\s*([0-9.]+)", text, re.MULTILINE)
    assert match is not None, "mc_frames no longer defines SOLE_BELOW_ANKLE_ROLL"
    assert float(match.group(1)) == RP.SOLE_BELOW_ANKLE_ROLL


def test_the_home_anchor_path_matches_hl_servings():
    """One file, two readers. A drift here silently places the viewer off the rig."""
    hl_serving = (_REPO / "scripts" / "hl_serving.py").read_text()
    tail = RP.DEFAULT_HOME_ANCHOR_PATH.split("third_party/")[1]
    for part in tail.split("/"):
        assert f'"{part}"' in hl_serving, (part, "hl_serving._HOME_ANCHOR_PATH changed shape")


# --------------------------------------------------------------------------- degrades
@pytest.mark.parametrize("kwargs, needle", [
    ({"home_anchor_path": "/definitely/not/here.json"}, "home_anchor.json is not checked out"),
    ({"urdf_path": "/definitely/not/here.urdf"}, "yam URDF is not checked out"),
])
def test_a_missing_input_returns_a_reason_instead_of_raising(kwargs, needle):
    """A fresh clone must still launch the viewer; the frame falls back and says why."""
    anchor, reason = RP.resolve_deploy_anchor(_ROLES, **kwargs)
    assert anchor is None
    assert needle in reason


def test_an_unknown_role_is_refused_by_name():
    """A role with no URDF body would be placed by accident; naming it is the fix."""
    anchor, reason = RP.resolve_deploy_anchor(("left", "elbow_cam"))
    assert anchor is None
    assert "elbow_cam" in reason and "known roles" in reason


def test_a_garbled_default_qpos_is_reported_not_raised(tmp_path):
    """A truncated / re-shaped bake must degrade, not crash the launch."""
    if not (_REPO / "third_party/UMI_collection/robot_assets/g1_yam/g1_29dof_yam.urdf").is_file():
        pytest.skip("UMI_collection submodule is not checked out (the URDF is checked before the bake)")
    bad = tmp_path / "home_anchor.json"
    bad.write_text(json.dumps({"default_qpos": [0.0] * 12}))
    anchor, reason = RP.resolve_deploy_anchor(_ROLES, home_anchor_path=str(bad))
    assert anchor is None
    assert "default_qpos" in reason


# --------------------------------------------------------------------------- default OFF
def test_resolve_role_geometry_is_unchanged_without_an_override():
    """The plain viewers keep the display homes AND a None home rotation, exactly as before."""
    geometry, _notes = resolve_role_geometry(_Layout(), None)
    for role in _ROLES:
        np.testing.assert_allclose(geometry[role].home_pos, DISPLAY_HOME_POS[role], atol=0.0)
        assert geometry[role].home_rotmat is None
    assert RoleGeometry(role="left", wrist_offset=np.zeros(3), tcp_triad_offset=np.zeros(3),
                        home_pos=np.zeros(3), is_gripper_role=True).home_rotmat is None


def test_resolve_role_geometry_takes_the_override_and_carries_its_rotation():
    anchor = _anchor_or_skip()
    home_pose = {r: (anchor.pos[r], anchor.rotmat[r]) for r in _ROLES}
    geometry, _notes = resolve_role_geometry(_Layout(), None, home_pose)
    for role in _ROLES:
        np.testing.assert_allclose(geometry[role].home_pos, anchor.pos[role], atol=0.0)
        np.testing.assert_allclose(geometry[role].home_rotmat, anchor.rotmat[role], atol=0.0)


def test_a_partial_override_says_the_page_is_mixed():
    """Half a frame is the dangerous state: the lines still draw, in two placements at once."""
    anchor = _anchor_or_skip()
    geometry, notes = resolve_role_geometry(
        _Layout(), None, {"left": (anchor.pos["left"], anchor.rotmat["left"])})
    np.testing.assert_allclose(geometry["head"].home_pos, DISPLAY_HOME_POS["head"], atol=0.0)
    assert any("MIXED" in note and "head" in note for note in notes), notes


# --------------------------------------------------------------------------- heading
def _rz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_the_heading_alignment_undoes_the_bodys_yaw():
    """``W = Rz(-heading)`` — and it is EXACTLY yaw, whatever pitch/roll the puck carries.

    The chest puck is worn on a harness, so it is never level; reading a full rotation would
    tilt the scene. Composing a 30 deg pitch onto a 40 deg heading must leave the answer at
    Rz(-40).
    """
    world, heading = RP.world_yaw_alignment(_rz(40.0))
    assert heading == pytest.approx(40.0)
    np.testing.assert_allclose(world, _rz(-40.0), atol=1e-12)
    np.testing.assert_allclose(world @ _rz(40.0), np.eye(3), atol=1e-12)
    pitch = np.array([[np.cos(0.5), 0.0, np.sin(0.5)], [0.0, 1.0, 0.0],
                      [-np.sin(0.5), 0.0, np.cos(0.5)]])
    _, tilted_heading = RP.world_yaw_alignment(_rz(40.0) @ pitch)
    assert tilted_heading == pytest.approx(40.0, abs=1e-9)


@pytest.mark.parametrize("bad", [np.full((3, 3), np.nan), np.eye(4), np.zeros((3, 3))])
def test_a_degenerate_head_rotation_leaves_the_scene_unturned(bad):
    """A bad anchor must degrade to the previous behaviour, not rotate by a garbage angle."""
    world, heading = RP.world_yaw_alignment(bad)
    np.testing.assert_allclose(world, np.eye(3), atol=0.0)
    assert heading == 0.0


def _ry(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def test_grip_yaw_alignment_reads_the_hands_heading_relative_to_its_home():
    """The no-chest fallback: ``W = Rz(-(yaw(hand) - yaw(home)))``, exactly yaw, home-relative.

    wipe_board's LEFT hand reads -118.8 deg at episode start (std 7.0 over 291 starts); with the
    arms-zeroed ready pose's hand facing +x (yaw 0) that is the whole turn. A home that itself
    faces 30 deg must shift the answer by 30 — that is what keeps the turned displacement
    consistent with the re-based orientation for ANY home. Pitch on the hand must not leak.
    """
    world, heading, used = RP.grip_yaw_alignment({"left": _rz(-118.8)})
    assert used == ("left",)
    assert heading == pytest.approx(-118.8)
    np.testing.assert_allclose(world, _rz(118.8), atol=1e-12)
    world, heading, used = RP.grip_yaw_alignment({"left": _rz(-118.8)}, {"left": _rz(30.0)})
    assert heading == pytest.approx(-148.8)
    np.testing.assert_allclose(world, _rz(148.8), atol=1e-12)
    _, tilted, _ = RP.grip_yaw_alignment({"left": _rz(-118.8) @ _ry(-17.2)}, {"left": None})
    assert tilted == pytest.approx(-118.8, abs=1e-9)
    # and the same matrix the chest path would produce for the same heading
    np.testing.assert_allclose(RP.grip_yaw_alignment({"left": _rz(40.0)})[0],
                               RP.world_yaw_alignment(_rz(40.0))[0], atol=1e-12)


def test_grip_yaw_alignment_is_a_circular_mean_over_both_hands():
    """Two hands at 170 and -170 deg mean 180, not 0 — an arithmetic mean would face the wrong way."""
    world, heading, used = RP.grip_yaw_alignment({"left": _rz(170.0), "right": _rz(-170.0)})
    assert used == ("left", "right")
    assert abs(abs(heading) - 180.0) < 1e-9
    np.testing.assert_allclose(world, _rz(180.0), atol=1e-12)
    # opposite hands have no mean heading: refuse rather than pick one
    world, heading, used = RP.grip_yaw_alignment({"left": _rz(90.0), "right": _rz(-90.0)})
    assert world is None and heading == 0.0 and used == ()


def test_grip_yaw_alignment_skips_a_hand_with_no_readable_yaw():
    """A hand pointing within 20 deg of vertical has no yaw; so does a garbage or vertical HOME."""
    down = _rz(-118.8) @ _ry(-90.0)                     # x-axis straight down
    assert RP.grip_yaw_alignment({"left": down}) == (None, 0.0, ())
    assert RP.grip_yaw_alignment({"left": _rz(-118.8) @ _ry(-75.0)})[0] is None   # 15 deg off vertical
    assert RP.grip_yaw_alignment({"left": _rz(-118.8) @ _ry(-65.0)})[0] is not None  # 25 deg: fine
    assert RP.grip_yaw_alignment({"left": np.full((3, 3), np.nan)}) == (None, 0.0, ())
    assert RP.grip_yaw_alignment({"left": _rz(-118.8)}, {"left": down}) == (None, 0.0, ())
    # the unusable hand is dropped, the other one still turns the scene
    world, heading, used = RP.grip_yaw_alignment({"left": down, "right": _rz(-118.8)})
    assert used == ("right",) and heading == pytest.approx(-118.8)
    assert RP.GRIP_YAW_MIN_FLOOR_PROJECTION == pytest.approx(np.sin(np.radians(20.0)))


def test_this_rigs_operator_really_does_face_backwards():
    """THE BUG: the recorded chest heading is ~-170 deg, so left and right were swapped.

    Positions are only TRANSLATED onto the home while orientations are re-based onto it, so the
    drawn body always faces +x — and those two are consistent only if the recording was aligned
    with +x. This pins that it is not, on real data, because the whole heading correction is
    dead weight if the number is ever ~0 and nobody would notice it going stale.
    """
    from flore.s1_policy.inference.common import HLSampleSource  # noqa: F401  (import check)

    heading = -174.8  # measured, session 0831 sample 20940 (mean -169 over 38 samples)
    world, _ = RP.world_yaw_alignment(_rz(heading))
    travel_world = np.array([-0.057, 0.479, -0.003])   # 48 cm, recorded
    turned = world @ travel_world
    assert travel_world[1] > 0 and turned[1] < 0, (travel_world, turned)
    # ...and the turn is rigid: it moves no length and no height.
    assert np.linalg.norm(turned) == pytest.approx(np.linalg.norm(travel_world), abs=1e-12)
    assert turned[2] == pytest.approx(travel_world[2], abs=1e-12)


def test_a_non_yaw_world_rotation_is_refused():
    """The guard that stops the removed 'rigid tilt' bug from coming back through this door."""
    from flore.s1_policy.inference.display import check_yaw_only, home_anchor_positions

    np.testing.assert_allclose(check_yaw_only(_rz(30.0)), _rz(30.0), atol=0.0)
    tilt = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])   # Rx(90)
    for call in (lambda m: check_yaw_only(m),
                 lambda m: home_anchor_positions(np.zeros((2, 3)), np.zeros(3), np.zeros(3),
                                                 world_rotmat=m)):
        with pytest.raises(ValueError, match="YAW-ONLY"):
            call(tilt)
    with pytest.raises(ValueError, match="proper rotation"):
        check_yaw_only(np.diag([1.0, -1.0, 1.0]))   # a MIRROR keeps z but flips handedness


def test_the_world_turn_costs_no_metric():
    """A shared rigid world rotation must leave every reported error exactly where it was.

    Position error is a norm of differences and the rotation is orthogonal; the orientation is
    CONJUGATED, so its geodesic is a trace that the conjugation cannot change. Asserting it
    numerically is what lets the frame be turned without re-validating the Metrics panel.
    """
    from flore.s1_policy.inference.display import home_anchor_positions, home_anchor_rotmats

    rng = np.random.default_rng(0)
    pred, targ = rng.normal(size=(7, 3)), rng.normal(size=(7, 3))
    ref, home, W = rng.normal(size=3), rng.normal(size=3), _rz(-174.8)
    plain = [home_anchor_positions(p, ref, home) for p in (pred, targ)]
    turned = [home_anchor_positions(p, ref, home, world_rotmat=W) for p in (pred, targ)]
    np.testing.assert_allclose(np.linalg.norm(plain[0] - plain[1], axis=-1),
                               np.linalg.norm(turned[0] - turned[1], axis=-1), atol=1e-12)

    # Compared as the TRACE, not as arccos(trace): the geodesic is a monotone function of it,
    # and arccos amplifies f64 noise to ~1e-8 rad right at the 0 the self-pair sits on, so an
    # arccos comparison cannot express "exactly preserved" at all.
    def _trace(a, b):
        return np.einsum("...ij,...ij->...", a, b)   # trace(A B^T)

    rots = [np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(4)]
    rots = np.stack([r * np.sign(np.linalg.det(r)) for r in rots])
    ref_rot = rots[0]
    plain_r = home_anchor_rotmats(rots, ref_rot)
    turned_r = home_anchor_rotmats(rots, ref_rot, world_rotmat=W)
    np.testing.assert_allclose(_trace(plain_r, plain_r[0]),
                               _trace(turned_r, turned_r[0]), atol=1e-12)
    # And the anchor still lands on its home rotation, which is what keeps the body facing +x.
    np.testing.assert_allclose(turned_r[0], np.eye(3), atol=1e-12)


# --------------------------------------------------------------------------- CLI
class _Args:
    def __init__(self, **kw) -> None:
        base = dict(display_frame=None, ready_pose_no_zero_arms=False,
                    home_anchor_json=None, motion_completion=False)
        base.update(kw)
        self.__dict__.update(base)


def test_motion_completion_implies_the_ready_pose_frame():
    """The two flags interact in ONE place, because the completion is meaningless without it."""
    assert RP.options_from_args(_Args())["display_frame"] == "display-home"
    assert RP.options_from_args(
        _Args(motion_completion=True))["display_frame"] == "g1-ready-pose"
    # An EXPLICIT choice wins in both directions — that is what --display-frame is for.
    assert RP.options_from_args(_Args(motion_completion=True, display_frame="display-home")
                               )["display_frame"] == "display-home"
    assert RP.options_from_args(_Args(display_frame="g1-ready-pose")
                               )["display_frame"] == "g1-ready-pose"


def test_the_zero_arms_flag_inverts():
    assert RP.options_from_args(_Args())["zero_arms"] is True
    assert RP.options_from_args(_Args(ready_pose_no_zero_arms=True))["zero_arms"] is False


def test_the_keypoint_fk_reproduces_the_ready_pose_frame_bit_for_bit():
    """The measured-keypoint FK and the ready-pose FK must be the SAME FK.

    A carried command anchor on the 5pt wire is leak-bounded against the robot's own keypoints,
    FK'd from RecordTap. If that FK disagreed with the one the ready frame is built from — a
    different lever, a different lift, a different body — the bound would carry a constant bias
    and the wire would sit permanently offset. Fed the anchor's OWN joints, the two must land on
    top of each other, so the test is an equality rather than a tolerance.
    """
    from flore.s1_policy.serving import hl_dense

    anchor, why = RP.resolve_deploy_anchor(("left", "right", "head"))
    if anchor is None:
        pytest.skip(f"the ready pose is unavailable here: {why}")
    fk, why = RP.KeypointFK.load()
    assert fk is not None, why
    assert set(fk.body_names) == set(hl_dense.FIVE_POINT_BODIES)

    readout = hl_dense.KeypointReadout(pelvis_pos=(0.0, 0.0, float(anchor.pelvis_z)))
    want_pos, want_orn = hl_dense.ready_pose_five_point(anchor, readout)
    got_pos, got_orn = readout.from_bodies(*fk.bodies(anchor.dof, lift_z=anchor.pelvis_z))
    np.testing.assert_allclose(got_pos, want_pos, atol=1e-12)
    np.testing.assert_allclose(got_orn, want_orn, atol=1e-12)

    # And the FK's own floor agrees with the anchor's, which is what makes lift_z the one number
    # that has to be passed rather than two conventions that happen to match.
    assert fk.pelvis_z(anchor.dof) == pytest.approx(anchor.pelvis_z, abs=1e-12)


def test_the_keypoint_fk_refuses_a_bad_joint_vector():
    """29 finite angles or nothing: a short or NaN readback must not silently FK into a pose
    that would then bound a published command."""
    fk, why = RP.KeypointFK.load()
    if fk is None:
        pytest.skip(why)
    with pytest.raises(ValueError, match="29 finite"):
        fk.bodies(np.zeros(28))
    bad = np.zeros(29)
    bad[3] = np.nan
    with pytest.raises(ValueError, match="29 finite"):
        fk.bodies(bad)
