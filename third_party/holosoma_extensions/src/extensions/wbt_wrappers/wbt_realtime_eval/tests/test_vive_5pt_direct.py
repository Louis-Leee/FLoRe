"""``--mode 5pt-direct``: FIVE real trackers onto the 5pt wire, with NO motion completion.

The rig gained a 4th and 5th puck (paired 2026-09-09), so the five points
``exp:g1-wbt-5pt-hybrid-unified-anneal`` tracks can be MEASURED rather than inferred by
Motion_Completion. The wire is byte-shaped exactly like ``--mode 5pt``'s, which is what keeps
the low level, the browser preset and the pose-match viewer unchanged — and is also why almost
nothing downstream can catch a mistake here. Hence these tests, which pin the four claims the
mode rests on:

1. **the geometry**: five roles, in ``robot_config.VR_5POINT_BODIES`` order, with that config's
   own levers (0.242 m TCP on both hands, 0.35 m on the chest, ZERO on both feet);
2. **the anchor**: the READY pose (arms zeroed), which is what the low level HOLDS and gates
   against — *not* the ``default_qpos`` pose the 2pt/3pt arrays use, 0.354 m per hand away.
   Per-point
   anchoring then lands every commanded point exactly on that reference at the calibration
   pose, so an ABSOLUTE-mode ``start`` arms by construction;
3. **no relabel is owed**: a constant puck-mount rotation cancels in a calibration-relative
   delta, which is the whole reason the completion path needs
   ``mc_frames.chest_body_from_puck`` (it consumes an ABSOLUTE chest heading) and this does not;
4. **non-invasion**: 2pt / 3pt / 5pt(MC) / dense payloads, roles and refusals are untouched.

pysurvive is injected via PYTHONPATH by ``scripts/_vive_teleop_bridge.sh``, not pip, so it is
stubbed — the bridge only touches it inside ``run_pysurvive`` / ``_list_serials``, which these
tests never enter.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

if "pysurvive" not in sys.modules:
    _m = types.ModuleType("pysurvive")
    _m.SimpleContext = object
    _m.survive_simple_serial_number = lambda *_a, **_k: b"?"
    sys.modules["pysurvive"] = _m

from wbt_realtime_eval.vive import load_home_anchor  # noqa: E402
from wbt_realtime_eval.vive import vive_teleop_bridge as vtb  # noqa: E402

_REPO = Path(__file__).resolve().parents[5]
_LAUNCHER = _REPO / "force_deploy/1_vive_bridge.sh"
_INNER = _REPO / "scripts/_vive_teleop_bridge.sh"
_needs_launchers = pytest.mark.skipif(
    not (_LAUNCHER.is_file() and _INNER.is_file()),
    reason="the VIVE bridge launchers are not part of this checkout",
)

#: The two foot pucks this rig paired on 2026-09-09, spelled here so a silent edit of either
#: launcher's default fails a test. WHICH foot is which is a strapping fact.
LEFT_FOOT = "LHR-C68495D2"
RIGHT_FOOT = "LHR-FF1D3DE2"

#: holosoma's own documented 5pt pre-arm gate reference for the two hands (pelvis frame, m) —
#: `_compute_vr_fk(READY POSE)`. Quoted rather than derived so that "the calibration pose lands
#: on the gate reference" is checked against a number from the OTHER side of the boundary.
GATE_REF_HAND_X = 0.4418
GATE_REF_HAND_Y = 0.1486
GATE_REF_HAND_Z = 0.0952

TCP_OFFSET_X = 0.242  # the only 5pt hand geometry ever trained
HEAD_OFFSET_Z = 0.35  # torso_link -> the head/HMD point


def _args(**over):
    base = dict(
        mode="5pt-direct",
        left_serial="LHR-L",
        right_serial="LHR-R",
        head_serial="LHR-H",
        left_foot_serial="LHR-LF",
        right_foot_serial="LHR-RF",
        hand_point="tcp",
        orientation="on",
        reach_scale=1.0,
        z_offset=0.0,
        yaw_deg=0.0,
        # False (the real default) so the fixture exercises the CHEST resolver rather than
        # short-circuiting to `fixed`. The synthetic operator below faces libsurvive +x, so the
        # resolved yaw is EXACTLY 0 and every "lands on the gate reference" assertion stays exact.
        no_yaw_autocal=False,
        world_yaw_from=None,
        yaw_trim_deg=0.0,
        world_yaw_tol_deg=40.0,
        stale_timeout=0.0,
        calib_delay=0.0,
        fresh=False,
        no_align=False,
        host="127.0.0.1",
        port=8770,
        rate_hz=50.0,
        list_serials=False,
        list_timeout=5.0,
        min_stations=2,
        expect_stations=0,
        mc_lag_frames=20,
        mc_no_wrist_ik=False,
        mc_foot_ik=False,
        mc_umi_dir=None,
        mc_ref_frame=None,
        umi_tracker_port=0,
        umi_tracker_rate_hz=90.0,
        umi_world_rpy_deg=None,
        umi_stale_timeout=0.25,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _quat_from_axis_angle(axis, deg: float) -> np.ndarray:
    """wxyz quaternion, inlined so a bridge-side helper change cannot silently agree."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    h = math.radians(deg) / 2.0
    return np.array([math.cos(h), *(math.sin(h) * a)])


def _rot(q_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in q_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


#: `BODY_RELABEL_M` as a quaternion, built from the factorization the calibration file states
#: (``H = Ry(90) @ Rx(90)``) instead of read out of the bridge — so the matrix is pinned to its
#: own stated meaning by `test_the_body_relabel_is_the_committed_chest_extrinsic`.
_Q_H = vtb.quat_mul(_quat_from_axis_angle([0, 1, 0], 90.0), _quat_from_axis_angle([1, 0, 0], 90.0))


def _body_puck_quat(q_body_wxyz) -> np.ndarray:
    """The RAW quat a BODY-worn puck must carry for its relabelled BODY frame to read `q_body`.

    ``R_body = R_puck @ H``  =>  ``R_puck = R_body @ H^T``  =>  ``q_puck = q_body (x) conj(q_H)``.
    Every chest/foot pose in these tests goes through this, because the world frame is now solved
    from an ABSOLUTE body heading: a puck rotation picked at random is an operator facing a random
    direction, and three of those disagree by ~180° and are (rightly) refused.
    """
    return vtb.quat_mul(np.asarray(q_body_wxyz, dtype=float), vtb.quat_inv(_Q_H))


#: Where each puck sits on a standing operator, in the OPERATOR's own frame (x forward, y left,
#: z up, metres). At heading 0 these are the numbers this file used before the frame became a
#: measured quantity, so every exact assertion below is unchanged.
_BODY_PLACE = {
    "left_hand": np.array([0.35, 0.30, 1.05]),
    "right_hand": np.array([0.36, -0.28, 1.04]),
    "head": np.array([0.02, 0.01, 1.42]),
    "left_foot": np.array([0.01, 0.14, 0.06]),
    "right_foot": np.array([0.00, -0.15, 0.06]),
}

#: A per-role body-frame TILT, so the five poses stay DISTINCT (a test that passes must not be
#: passing because every role reads the same puck) while every heading stays EXACTLY the
#: operator's: a pitch or a roll leaves ``atan2`` of the projected forward alone, a YAW would not.
_BODY_TILT = {
    "head": _quat_from_axis_angle([0, 1, 0], 4.0),
    "left_foot": _quat_from_axis_angle([1, 0, 0], 2.0),
    "right_foot": _quat_from_axis_angle([1, 0, 0], -3.0),
}
#: The two hands carry no relabel and no heading witness, so their rotations are free.
_HAND_TILT = {
    "left_hand": _quat_from_axis_angle([0, 0, 1], 12.0),
    "right_hand": _quat_from_axis_angle([0, 0, 1], -9.0),
}


def _operator(heading_deg: float = 0.0, place=None) -> dict:
    """Five RAW puck poses for one operator STANDING at `heading_deg` in the libsurvive frame.

    Coherent by construction: one ``Rz(psi)`` turns both the placements and the body rotations,
    so all four world-yaw witnesses (chest heading, foot heading, foot separation, hand
    separation) agree on `heading_deg` — which is what a real operator produces and what the
    cross-check demands. `place` overrides individual body-frame placements (that is how the
    parked-hands and swapped-feet cases are built).
    """
    q_heading = _quat_from_axis_angle([0, 0, 1], heading_deg)
    Rz = _rot(q_heading)
    out = {}
    for role, p_body in {**_BODY_PLACE, **(place or {})}.items():
        p = Rz @ np.asarray(p_body, dtype=float)
        if role in _BODY_TILT:
            q = _body_puck_quat(vtb.quat_mul(q_heading, _BODY_TILT[role]))
        else:
            q = vtb.quat_mul(q_heading, _HAND_TILT[role])
        out[role] = (p, np.asarray(q, dtype=float))
    return out


#: The default calibration pose: an operator standing square, facing libsurvive +x.
_CALIB = _operator(0.0)


def _fresh(poses=None, **over):
    """A 5pt-direct bridge with every puck locked and two SOLVED base stations seeded.

    The stations are seeded rather than stubbed away because the calibration gate is real: it
    refuses below two solved stations (one lighthouse leaves a sparse puck's yaw
    under-constrained, and yaw is exactly what the world-yaw autocal reads). Seeding them lets
    these tests drive the ACTUAL `_maybe_calibrate`, so the anchor they check is the one an
    operator gets — not one poked in by setting `_calibrated = True`.
    """
    b = vtb.ViveTeleopBridge(_args(**over))
    b._stations_live = {"LH0", "LH1"}  # noqa: SLF001
    for role, pose in (poses or _CALIB).items():
        b._raw_hand[role] = (pose[0].copy(), pose[1].copy())  # noqa: SLF001
        b._raw_hand_t[role] = 1e18  # noqa: SLF001
        b._raw_hand_seq[role] = 1  # noqa: SLF001
    return b


def _calibrated(poses=None, **over):
    """`_fresh`, taken through the real calibration path (calib_delay 0 => two calls)."""
    b = _fresh(poses=poses, **over)
    b._maybe_calibrate()  # noqa: SLF001 - first call records "all seen", the second calibrates
    b._maybe_calibrate()  # noqa: SLF001
    assert b._calibrated, "the fixture must be calibrated for the anchor tests to mean anything"  # noqa: SLF001
    return b


def _move(b, role: str, dp=(0.0, 0.0, 0.0), q=None) -> None:
    """Displace / re-orient ONE puck from its calibration pose."""
    p0, q0 = _CALIB[role]
    b._raw_hand[role] = (p0 + np.asarray(dp, dtype=float), np.asarray(q if q is not None else q0))  # noqa: SLF001
    b._raw_hand_t[role] = 1e18  # noqa: SLF001


def _home_q_xyzw() -> np.ndarray:
    """The (5,4) xyzw home rotations, NORMALIZED — which is what the bridge publishes.

    The committed array is unit only to ~5e-9 (it was read off a bake, not hand-typed), and the
    bridge normalizes on the way through. Comparing against the normalized value keeps these
    assertions EXACT instead of hiding a real 5 mm-class error under a loose tolerance.
    """
    q = np.asarray(load_home_anchor()["home_orn_quat_xyzw_5pt"], dtype=float)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _pts(snap) -> tuple[np.ndarray, np.ndarray]:
    """(pos (5,3), orn xyzw (5,4)) out of a wire frame."""
    return (
        np.asarray(snap["pos"], dtype=float).reshape(5, 3),
        np.asarray(snap["orn"], dtype=float).reshape(5, 4),
    )


# ---------------------------------------------------------------------------
# 1. the geometry: five roles, holosoma's own bodies and levers
# ---------------------------------------------------------------------------


def test_the_five_roles_are_in_holosomas_vr_5point_body_order() -> None:
    """The role ORDER is the wire layout, and the wire is 15/20 wide whichever order it is in —
    so this mapping is the only thing standing between a reorder and a policy fed a hand where
    it expects an ankle. Read from robot_config, not re-typed."""
    from wbt_wrappers_inference.config_values.robot_config import VR_5POINT_BODIES

    want = {
        "left_hand": "left_wrist_yaw_link",
        "right_hand": "right_wrist_yaw_link",
        "head": "torso_link",
        "left_foot": "left_ankle_roll_link",
        "right_foot": "right_ankle_roll_link",
    }
    assert [want[r] for r in vtb.DIRECT_5PT_ROLES] == list(VR_5POINT_BODIES)


def test_the_levers_are_robot_configs_own_5pt_offsets() -> None:
    """0.242 m TCP on both hands, 0.35 m on the chest, and ZERO on both feet.

    A foot keypoint IS the bare `ankle_roll_link` origin: a hand lever models a physical mount
    and the head lever lifts torso_link to HMD height, but a foot has neither, so a nonzero sole
    offset would price a point nothing else in the stack knows about.
    """
    from wbt_wrappers_inference.config_values.robot_config import VR_5POINT_OFFSETS_TCP

    got = vtb.ViveTeleopBridge(_args())._vr_offsets_5pt  # noqa: SLF001
    np.testing.assert_allclose(got, np.asarray(VR_5POINT_OFFSETS_TCP, dtype=float), atol=0.0)
    # ...and spelled out, so a config change that moves BOTH sides together still gets read.
    assert got[0][0] == got[1][0] == TCP_OFFSET_X
    assert got[2][2] == HEAD_OFFSET_Z
    np.testing.assert_allclose(got[3:], 0.0, atol=0.0)


def test_the_first_three_slots_are_the_3pt_triple() -> None:
    """That prefix property is what makes 5pt a SUPERSET of 3pt — the same statement the
    training layout and the browser's 15D concatenation both rely on."""
    anchor = load_home_anchor()
    b = vtb.ViveTeleopBridge(_args())
    np.testing.assert_allclose(
        b._vr_offsets_5pt[:3], np.asarray(anchor["vr_3point_offsets_tcp"], dtype=float), atol=0.0  # noqa: SLF001
    )


# ---------------------------------------------------------------------------
# 2. the anchor: the READY pose, and the gate closing by construction
# ---------------------------------------------------------------------------


def test_the_anchor_is_the_ready_pose_not_the_default_pose() -> None:
    """The 0.354 m-per-hand trap, named. 2pt/3pt anchor on `default_qpos`; the low level under
    HOLOSOMA_INIT_READY_POSE=1 holds the READY pose (arms zeroed) and gates the wire against
    ITS FK. Anchoring a 5pt wire on the 3pt arrays would put every hand ~0.29 m from the gate's
    reference — refused in absolute mode, and invisible in relative mode until it is not."""
    anchor = load_home_anchor()
    b = vtb.ViveTeleopBridge(_args())
    np.testing.assert_allclose(
        b._home_pos_5pt, np.asarray(anchor["home_pos_pelvis_5pt_tcp"], dtype=float), atol=0.0  # noqa: SLF001
    )
    five = np.asarray(anchor["home_pos_pelvis_5pt_tcp"], dtype=float)[:3]
    three = np.asarray(anchor["home_pos_pelvis_3pt_tcp"], dtype=float)
    per_point = np.linalg.norm(five - three, axis=1)
    # 0.354 m PER HAND. (home_anchor.json quotes 0.289 m for the same fact; that is the RMS over
    # the 3pt triple, in which the chest contributes exactly 0 because torso_link does not move
    # when the arms are zeroed. The per-hand number is the one an operator would feel.)
    np.testing.assert_allclose(per_point, [0.3537, 0.3537, 0.0], atol=1e-4)


def test_the_hands_home_is_holosomas_documented_gate_reference() -> None:
    """Checked against a number quoted from the OTHER side of the boundary (holosoma's own
    `_compute_vr_fk(READY POSE)`), not against anything this repo derives."""
    home = np.asarray(load_home_anchor()["home_pos_pelvis_5pt_tcp"], dtype=float)
    for k, sign in ((0, +1.0), (1, -1.0)):
        np.testing.assert_allclose(
            home[k], [GATE_REF_HAND_X, sign * GATE_REF_HAND_Y, GATE_REF_HAND_Z], atol=1e-4
        )


def test_the_calibration_pose_lands_every_point_on_the_gate_reference() -> None:
    """Hold-still is EXACT, per point — including the two FEET.

    This is the mode's whole reason to exist next to `--mode 5pt`: there the feet are the
    completion's inference and land 69–91 mm from this same reference by construction (the ready
    pose is a 38° crouch, the completion stands straighter), which is why holosoma gates only 3
    of 5. Measured pucks make all five exact, so an ABSOLUTE-mode `start` arms by construction.
    """
    anchor = load_home_anchor()
    snap = _calibrated().snapshot()
    assert snap is not None
    pos, orn = _pts(snap)
    np.testing.assert_allclose(pos, np.asarray(anchor["home_pos_pelvis_5pt_tcp"], dtype=float), atol=1e-12)
    np.testing.assert_allclose(orn, _home_q_xyzw(), atol=1e-12)


def test_a_translation_of_one_puck_moves_only_that_point_one_to_one() -> None:
    """Per-role anchoring, stated as a property: the mapping is 1:1 and does not couple the
    points. In particular a FOOT puck steers its own ankle keypoint and nothing else — the
    channel that on the completion path was an inference off the three upper pucks."""
    anchor = np.asarray(load_home_anchor()["home_pos_pelvis_5pt_tcp"], dtype=float)
    for k, role in enumerate(vtb.DIRECT_5PT_ROLES):
        b = _calibrated()
        _move(b, role, dp=(0.11, -0.07, 0.05))
        pos, _ = _pts(b.snapshot())
        np.testing.assert_allclose(pos[k], anchor[k] + [0.11, -0.07, 0.05], atol=1e-12)
        others = [i for i in range(5) if i != k]
        np.testing.assert_allclose(pos[others], anchor[others], atol=1e-12)


def test_reach_scale_multiplies_every_point_including_the_feet() -> None:
    """A stride and a reach are the same statement, so `--reach-scale` covers both."""
    anchor = np.asarray(load_home_anchor()["home_pos_pelvis_5pt_tcp"], dtype=float)
    b = _calibrated(reach_scale=2.0)
    for role in vtb.DIRECT_5PT_ROLES:
        _move(b, role, dp=(0.05, 0.0, 0.0))
    pos, _ = _pts(b.snapshot())
    np.testing.assert_allclose(pos, anchor + [0.10, 0.0, 0.0], atol=1e-12)


def test_the_z_offset_trim_skips_the_two_feet() -> None:
    """`--z-offset` is an operator-height trim for the HANDS (and the chest). On a foot it is
    not a trim, it is a command to leave the floor — a hop, on a balance controller. So the two
    ground points are excluded from it, and only from it."""
    anchor = np.asarray(load_home_anchor()["home_pos_pelvis_5pt_tcp"], dtype=float)
    pos, _ = _pts(_calibrated(z_offset=0.05).snapshot())
    np.testing.assert_allclose(pos[:3, 2], anchor[:3, 2] + 0.05, atol=1e-12)
    np.testing.assert_allclose(pos[3:, 2], anchor[3:, 2], atol=1e-12)


# ---------------------------------------------------------------------------
# 3. rotation: no relabel is owed, and only the levered points sweep an arc
# ---------------------------------------------------------------------------


def test_a_constant_mount_rotation_cancels_so_no_relabel_is_owed() -> None:
    """WHY `_point_command_5pt` needs no `chest_body_from_puck`: the COMMAND cannot see a mount.

    Every rotation in the command is a calibration-RELATIVE delta ``(cur · ref⁻¹) · home``, so a
    constant body-local mount rotation appears in both `cur` and `ref` and drops out; the puck
    ORIGIN is unmoved by it either way. Checked on ALL FIVE pucks at once, with a real motion on
    top so it cannot pass by both sides being the home.

    ⚠️ The frame is PINNED here (`--world-yaw-from fixed`), and that is the point rather than a
    convenience: `world_yaw` is solved from an ABSOLUTE body heading, where a mount rotation
    does NOT cancel — so this test is a statement about the command and the next one is the
    statement about the frame. Together they are why the relabel is mandatory in exactly one
    place. Before the frame became a measured quantity this test read as "no relabel is owed
    anywhere", which was true of the code it tested and not of the mode.
    """
    mount = _quat_from_axis_angle([0.3, -0.5, 0.81], 137.0)
    plain = _calibrated(world_yaw_from="fixed")
    remounted = _fresh(world_yaw_from="fixed")
    for role, (p0, q0) in _CALIB.items():
        remounted._raw_hand[role] = (p0.copy(), vtb.quat_mul(q0, mount))  # noqa: SLF001
    remounted._maybe_calibrate()  # noqa: SLF001
    remounted._maybe_calibrate()  # noqa: SLF001
    assert remounted._calibrated  # noqa: SLF001

    twist = _quat_from_axis_angle([0, 0, 1], 25.0)
    for role in vtb.DIRECT_5PT_ROLES:
        q0 = _CALIB[role][1]
        _move(plain, role, dp=(0.03, 0.02, -0.01), q=vtb.quat_mul(twist, q0))
        remounted._raw_hand[role] = (  # noqa: SLF001
            _CALIB[role][0] + np.array([0.03, 0.02, -0.01]),
            vtb.quat_mul(twist, vtb.quat_mul(q0, mount)),
        )
        remounted._raw_hand_t[role] = 1e18  # noqa: SLF001

    a, b = plain.snapshot(), remounted.snapshot()
    assert a is not None and b is not None
    np.testing.assert_allclose(a["pos"], b["pos"], atol=1e-12)
    np.testing.assert_allclose(a["orn"], b["orn"], atol=1e-12)


def test_a_foot_twist_rotates_the_foot_without_translating_it() -> None:
    """The feet carry NO lever, so the tracked point is the ankle ORIGIN: an ankle roll must
    change the commanded orientation and leave the commanded position alone."""
    anchor = np.asarray(load_home_anchor()["home_pos_pelvis_5pt_tcp"], dtype=float)
    b = _calibrated()
    before = _pts(b.snapshot())[1][3].copy()
    _move(b, "left_foot", q=vtb.quat_mul(_quat_from_axis_angle([1, 0, 0], 20.0), _CALIB["left_foot"][1]))
    pos, orn = _pts(b.snapshot())
    np.testing.assert_allclose(pos[3], anchor[3], atol=1e-12)
    assert float(np.linalg.norm(orn[3] - before)) > 1e-3


def test_a_hand_twist_sweeps_the_tcp_lever_arc() -> None:
    """The hands DO carry a lever (0.242 m), so a pure twist in place moves the commanded TCP
    along an arc — which is what keeps (pos, orn) a pair the policy has seen. The magnitude is
    checked against the lever radius rather than a recorded constant."""
    anchor = np.asarray(load_home_anchor()["home_pos_pelvis_5pt_tcp"], dtype=float)
    deg = 30.0
    b = _calibrated()
    _move(b, "left_hand", q=vtb.quat_mul(_quat_from_axis_angle([0, 0, 1], deg), _CALIB["left_hand"][1]))
    pos, _ = _pts(b.snapshot())
    moved = float(np.linalg.norm(pos[0] - anchor[0]))
    want = 2.0 * TCP_OFFSET_X * math.sin(math.radians(deg) / 2.0)  # chord of the lever arc
    assert abs(moved - want) < 1e-9, f"lever arc {moved:.6f} m vs chord {want:.6f} m"


def test_orientation_off_holds_every_point_at_its_home_rotation() -> None:
    """Translation-first staging, unchanged in meaning: the wrists/chest/ankles hold the home
    rotation while position tracks, so the lever term is constant and the mapping reduces to the
    plain puck displacement."""
    anchor = np.asarray(load_home_anchor()["home_pos_pelvis_5pt_tcp"], dtype=float)
    b = _calibrated(orientation="off")
    for role in vtb.DIRECT_5PT_ROLES:
        _move(b, role, dp=(0.0, 0.0, 0.09), q=_quat_from_axis_angle([1, 1, 0], 40.0))
    pos, orn = _pts(b.snapshot())
    np.testing.assert_allclose(pos, anchor + [0.0, 0.0, 0.09], atol=1e-12)
    np.testing.assert_allclose(orn, _home_q_xyzw(), atol=1e-12)


def test_every_quaternion_on_the_wire_is_unit_norm_and_plus_w() -> None:
    """The 5pt policy trained on the +w branch ONLY (its normalizer puts w at mean +0.73 hands /
    +0.98 head / +0.96 feet), so `-q` is a ~20σ input rather than a harmless alias — and the low
    level COMPOSES these with its arm-clutch anchor, which scales rather than fails."""
    b = _calibrated()
    # A pose near the 180° seam is exactly where the branch used to flip for every point at once.
    for role in vtb.DIRECT_5PT_ROLES:
        _move(b, role, q=vtb.quat_mul(_quat_from_axis_angle([0, 0, 1], 179.0), _CALIB[role][1]))
    _, orn = _pts(b.snapshot())
    for k in range(5):
        np.testing.assert_allclose(float(np.linalg.norm(orn[k])), 1.0, atol=1e-9)
        assert orn[k][3] >= 0.0, f"point {k} is on the -w branch: {orn[k]}"


def test_the_reported_rotation_is_the_world_frame_delta_onto_the_home() -> None:
    """Spelled out once, in matrix form, so "which side the delta multiplies on" is pinned: a
    puck rotation about real vertical must rotate the commanded body about pelvis-Z (the
    world-frame, LEFT-multiplied convention), not about the puck's own mounting axes."""
    b = _calibrated()
    dq = _quat_from_axis_angle([0, 0, 1], 33.0)
    _move(b, "head", q=vtb.quat_mul(dq, _CALIB["head"][1]))
    _, orn = _pts(b.snapshot())
    home_q = np.asarray(load_home_anchor()["home_orn_quat_xyzw_5pt"][2], dtype=float)
    home_wxyz = np.array([home_q[3], home_q[0], home_q[1], home_q[2]])
    want = _rot(dq) @ _rot(home_wxyz)
    got = _rot(np.array([orn[2][3], orn[2][0], orn[2][1], orn[2][2]]))
    np.testing.assert_allclose(got, want, atol=1e-9)


# ---------------------------------------------------------------------------
# 3b. the WORLD FRAME — the one decision that turns every point at once
#
# `world_yaw` is the whole frame: the command is
# `home + Rz(world_yaw) · (puck − puck_at_calib)` with NO per-role position transform, so a
# wrong `world_yaw` cannot show up as "one keypoint is off". It shows up as every commanded
# displacement being rotated, with the start pose still exact and every gate still green —
# which is how it reached the rig on 2026-09-10 as "I move my foot FORWARD and the robot steps
# to the RIGHT".
# ---------------------------------------------------------------------------


def test_the_body_relabel_is_the_committed_chest_extrinsic() -> None:
    """`BODY_RELABEL_M` == its own stated factorization == the converter's own calibration file.

    Three independent spellings of one physical fact (the chest bracket's rotation), and the
    bridge now DEPENDS on it: it is what turns a raw puck heading into the operator's facing.
    Order 3, so a second application does not cancel — it lands 120° away.
    """
    H = np.asarray(vtb.BODY_RELABEL_M, dtype=float)
    np.testing.assert_allclose(H, _rot(_Q_H), atol=1e-12)  # H == Ry(90) @ Rx(90)
    np.testing.assert_allclose(H @ H @ H, np.eye(3), atol=1e-12)
    assert not np.allclose(H @ H, np.eye(3)), "order 3, not an involution — H^-1 != H"
    np.testing.assert_allclose(np.linalg.det(H), 1.0, atol=1e-12)
    # BODY_FWD_IN_PUCK is the first COLUMN, i.e. x_body = -z_puck, as the file's axis_map says.
    np.testing.assert_allclose(vtb.BODY_FWD_IN_PUCK, np.array([0.0, 0.0, -1.0]), atol=0.0)

    import json

    cal = Path.home() / "FUJI/third_party/UMI_collection/robot_assets/calibration/head_extrinsic.json"
    if not cal.is_file():
        pytest.skip(f"UMI_collection not checked out at {cal}")
    T = np.asarray(json.loads(cal.read_text())["T_tracker_head"], dtype=float)
    np.testing.assert_allclose(T[:3, :3], H, atol=0.0)
    np.testing.assert_allclose(T[:3, 3], np.zeros(3), atol=0.0), "rotation-ONLY: the chest has no lever"


@pytest.mark.parametrize("heading", [0.0, 37.0, -118.8, 169.0])
def test_the_frame_is_solved_from_the_chest_at_any_operator_heading(heading) -> None:
    """The operator may stand facing anywhere; the CHEST says which way that is.

    `world_yaw` must come out as exactly −heading (the Rz that puts the operator's forward on
    robot +x). −118.8° and +169° are this rig's two real measured operator headings (the
    wipe_board and 3-tracker sessions), i.e. the frame is never near identity in practice.
    """
    b = _calibrated(poses=_operator(heading))
    assert b._yaw_used == "chest"  # noqa: SLF001
    assert abs(vtb.ang_diff_deg(b._world_yaw, math.radians(-heading))) < 1e-9  # noqa: SLF001
    # ...and all four witnesses agree, because a real operator is ONE rigid body.
    for name, val in b._yaw_witnesses.items():  # noqa: SLF001
        assert val is not None, name
        assert abs(vtb.ang_diff_deg(val, math.radians(-heading))) < 3.0, (name, math.degrees(val))


@pytest.mark.parametrize("heading", [0.0, 37.0, -118.8])
@pytest.mark.parametrize("role,slot", [("left_foot", 3), ("right_foot", 4), ("head", 2), ("left_hand", 0)])
def test_operator_forward_is_robot_forward_for_every_point(heading, role, slot) -> None:
    """THE regression test for the rig report: move a puck along the OPERATOR's own forward and
    the commanded point must move along robot +x, at any operator heading.

    Before the frame was measured this held only when the operator happened to be standing the
    way the L-R hand vector implied. 0.20 m is a stride-sized step; the two off-axis components
    must be ZERO, not merely small, because the whole failure mode is a rotation of this vector.
    """
    poses = _operator(heading)
    b = _calibrated(poses=poses)
    base = np.asarray(b.snapshot()["pos"], dtype=float).reshape(5, 3)
    psi = math.radians(heading)
    fwd_world = np.array([math.cos(psi), math.sin(psi), 0.0])  # the operator's OWN forward
    p0, q0 = poses[role]
    b._raw_hand[role] = (p0 + 0.20 * fwd_world, q0)  # noqa: SLF001
    b._raw_hand_t[role] = 1e18  # noqa: SLF001
    moved = np.asarray(b.snapshot()["pos"], dtype=float).reshape(5, 3)
    np.testing.assert_allclose(moved[slot] - base[slot], np.array([0.20, 0.0, 0.0]), atol=1e-12)


def test_parked_hand_pucks_turn_the_frame_and_the_chest_does_not() -> None:
    """The measured rig failure, reproduced: the HANDS witness is not safe in this mode.

    A standing, walking operator's hand pucks are parked or full as often as not. Park them in a
    line pointing AWAY from the operator (a table, a bench, a belt clip) and the L-R hand vector
    — which is what 2pt/3pt solve their frame from, and what this mode used to — reads the
    operator's FORWARD as their lateral axis. The frame comes out 90° turned, and an operator
    step forward is published as motion along the robot's −y: "the robot steps to the RIGHT".
    Nothing fails; the wire is well formed and the pre-arm gate still closes.

    `--world-yaw-from chest` is unaffected by the same parked hands, which is the whole reason
    it is the default here.
    """
    # Hands parked front-to-back along the operator's forward, 0.4 m apart: LEFT behind RIGHT.
    parked = {"left_hand": np.array([0.20, 0.0, 0.80]), "right_hand": np.array([0.60, 0.0, 0.80])}
    poses = _operator(0.0, place=parked)

    # At the DEFAULT tolerance, deliberately: the parked-hands frame is not refused by anything.
    # The chest and the feet still agree with each other, so no cross-check fires -- the hands
    # source simply answers the wrong number, confidently. That is the failure being pinned.
    hands = _calibrated(poses=poses, world_yaw_from="hands")
    assert abs(vtb.ang_diff_deg(hands._world_yaw, math.radians(-90.0))) < 1e-9  # noqa: SLF001
    base = np.asarray(hands.snapshot()["pos"], dtype=float).reshape(5, 3)
    hands._raw_hand["left_foot"] = (poses["left_foot"][0] + np.array([0.20, 0.0, 0.0]), poses["left_foot"][1])  # noqa: SLF001
    hands._raw_hand_t["left_foot"] = 1e18  # noqa: SLF001
    got = np.asarray(hands.snapshot()["pos"], dtype=float).reshape(5, 3)[3] - base[3]
    # A forward step published as the robot's RIGHT. This is the bug, asserted.
    np.testing.assert_allclose(got, np.array([0.0, -0.20, 0.0]), atol=1e-12)

    chest = _calibrated(poses=poses)  # same parked hands, default source
    assert abs(vtb.ang_diff_deg(chest._world_yaw, 0.0)) < 1e-9  # noqa: SLF001
    base2 = np.asarray(chest.snapshot()["pos"], dtype=float).reshape(5, 3)
    chest._raw_hand["left_foot"] = (poses["left_foot"][0] + np.array([0.20, 0.0, 0.0]), poses["left_foot"][1])  # noqa: SLF001
    chest._raw_hand_t["left_foot"] = 1e18  # noqa: SLF001
    got2 = np.asarray(chest.snapshot()["pos"], dtype=float).reshape(5, 3)[3] - base2[3]
    np.testing.assert_allclose(got2, np.array([0.20, 0.0, 0.0]), atol=1e-12)


def test_skipping_the_relabel_would_turn_the_frame_90_degrees() -> None:
    """The chest witness is only a witness BECAUSE of the relabel — the completion path's own
    documented failure (`mc_frames.CHEST_RELABEL_R`: raw heading +90.09°, relabelled +0.20°).

    So a mount rotation is exactly what the frame CAN see, which is the complement of
    `test_a_constant_mount_rotation_cancels_so_no_relabel_is_owed`: the command cannot see one,
    the frame can, and that asymmetry is the reason the relabel lives in one place only.
    """
    q_chest = _operator(0.0)["head"][1]
    relabelled = vtb.body_heading_from_puck(q_chest)
    fwd_raw = vtb.quat_rotate(q_chest, np.array([1.0, 0.0, 0.0]))
    raw = math.atan2(float(fwd_raw[1]), float(fwd_raw[0]))
    assert abs(math.degrees(relabelled)) < 1e-9, "the relabelled chest heading IS the operator's"
    # Exactly 90 deg, and structurally so rather than by luck: H's first ROW is (0,1,0), so the
    # RAW puck's own +x is the operator's LEFT. That is the +90.09 deg vs +0.20 deg that
    # `mc_frames.CHEST_RELABEL_R` measured on 979 real frames.
    assert abs(vtb.ang_diff_deg(raw, relabelled) - 90.0) < 1e-9, math.degrees(raw)

    # And a re-mounted chest bracket moves the FRAME, not just the drawn axes: it is the one
    # thing here that a mount rotation CAN change.
    turned = dict(_operator(0.0))
    turned["head"] = (turned["head"][0], _body_puck_quat(_quat_from_axis_angle([0, 0, 1], 30.0)))
    b = _calibrated(poses=turned, world_yaw_tol_deg=180.0)
    assert abs(vtb.ang_diff_deg(b._world_yaw, math.radians(-30.0))) < 1e-9  # noqa: SLF001


def test_two_foot_pucks_on_the_wrong_ankles_refuse_to_calibrate() -> None:
    """A swap used to be SILENT: the wire is valid, the widths match, the pre-arm gate closes,
    and the robot simply crosses its legs. The FEET witness flips 180° while the chest does not,
    so the cross-check names it — the one check that makes a strapping mistake fail loudly."""
    swapped = _operator(0.0, place={"left_foot": _BODY_PLACE["right_foot"], "right_foot": _BODY_PLACE["left_foot"]})
    b = _fresh(poses=swapped)
    b._maybe_calibrate()  # noqa: SLF001
    b._maybe_calibrate()  # noqa: SLF001
    assert not b._calibrated, "a 180 deg chest-vs-feet disagreement must not calibrate"  # noqa: SLF001
    assert b.snapshot() is None, "and nothing may reach the wire"  # noqa: SLF001
    # Raising the tolerance is the documented override, and it must actually work (an operator
    # who KNOWS the chest is the one to believe has to be able to run).
    ok = _calibrated(poses=swapped, world_yaw_tol_deg=180.0)
    assert abs(vtb.ang_diff_deg(ok._world_yaw, 0.0)) < 1e-9  # noqa: SLF001


def test_a_degenerate_witness_refuses_rather_than_falling_back() -> None:
    """Feet together => the foot SEPARATION direction is tracker noise. `--world-yaw-from feet`
    must refuse, not quietly borrow the chest's answer: a silent fallback is how an operator ends
    up believing they tested a witness they did not."""
    together = _operator(0.0, place={"left_foot": np.array([0.01, 0.01, 0.06]), "right_foot": np.array([0.0, -0.01, 0.06])})
    b = _fresh(poses=together, world_yaw_from="feet")
    b._maybe_calibrate()  # noqa: SLF001
    b._maybe_calibrate()  # noqa: SLF001
    assert not b._calibrated  # noqa: SLF001
    assert b._yaw_witnesses["feet_sep"] is None  # noqa: SLF001
    assert b._yaw_witnesses["chest"] is not None, "the chest still has plenty to say"  # noqa: SLF001
    # The same stance is fine on the chest, because a heading needs one puck and no separation.
    assert _calibrated(poses=together)._yaw_used == "chest"  # noqa: SLF001


def test_the_yaw_trim_rotates_the_whole_frame_by_exactly_that() -> None:
    """The bring-up knob: if the frame is off by a suspected 90°, an operator must be able to
    test that without a code change. It is added AFTER the witness and reported with it."""
    poses = _operator(25.0)
    plain = _calibrated(poses=poses)
    trimmed = _calibrated(poses=poses, yaw_trim_deg=90.0)
    assert abs(vtb.ang_diff_deg(trimmed._world_yaw, plain._world_yaw + math.radians(90.0))) < 1e-9  # noqa: SLF001
    # A trim turns the COMMANDED displacement and nothing else: the calibration pose still lands
    # on the gate reference (both terms are anchored), which is what keeps `start` arming.
    home = np.asarray(load_home_anchor()["home_pos_pelvis_5pt_tcp"], dtype=float)
    np.testing.assert_allclose(np.asarray(trimmed.snapshot()["pos"], dtype=float).reshape(5, 3), home, atol=1e-12)


def test_the_frame_flags_are_refused_on_the_older_modes() -> None:
    """2pt/3pt/5pt/dense teleop is a reviewed, working path (3pt-unified-anneal is in use), so
    the new flags may not reach it even by accident. Refused BY NAME at construction."""
    for mode in ("2pt", "3pt", "5pt", "dense"):
        kw = dict(mode=mode, hand_point="lever" if mode == "2pt" else "tcp")
        with pytest.raises(SystemExit):
            vtb.ViveTeleopBridge(_args(world_yaw_from="chest", **kw))
        with pytest.raises(SystemExit):
            vtb.ViveTeleopBridge(_args(yaw_trim_deg=15.0, **kw))
        # ...and their own frame still comes from the L-R hand vector, untouched.
        assert vtb.ViveTeleopBridge(_args(**kw))._yaw_from == "hands"  # noqa: SLF001


def test_no_yaw_autocal_and_an_explicit_source_are_refused_together() -> None:
    """Two flags saying opposite things about the same number. Pick one, out loud."""
    with pytest.raises(SystemExit):
        vtb.ViveTeleopBridge(_args(no_yaw_autocal=True, world_yaw_from="chest"))
    b = vtb.ViveTeleopBridge(_args(no_yaw_autocal=True))
    assert b._yaw_from == "fixed", "no_yaw_autocal alone means 'take --yaw-deg verbatim'"  # noqa: SLF001


# ---------------------------------------------------------------------------
# 4. the wire, and the gates on it
# ---------------------------------------------------------------------------


def test_the_wire_is_pos15_orn20_tagged_5pt_with_no_mc_keys() -> None:
    """Byte-compatible with `--mode 5pt` minus the two `mc_*` provenance keys — there is no
    completion frame and no completion lag, and reporting a zero would claim a stage that does
    not exist. `ViveLiveSource` validates the (15, 20) width PAIR against this tag."""
    snap = _calibrated().snapshot()
    assert snap is not None
    assert snap["mode"] == "5pt"
    assert len(snap["pos"]) == 15
    assert len(snap["orn"]) == 20
    assert not any(k.startswith("mc_") for k in snap), snap.keys()
    assert snap["orientation"] is True


def test_the_wire_widths_are_the_pair_the_consumer_validates() -> None:
    """Read off `ViveLiveSource`'s own table rather than re-typed: the PAIR is what identifies
    the family (2pt's 2×rot6d and 3pt's 3×quat are both 12 wide)."""
    from wbt_wrappers_inference.utils.vive_live_source import _MODE_WIDTHS

    snap = _calibrated().snapshot()
    assert (len(snap["pos"]), len(snap["orn"])) == _MODE_WIDTHS["5pt"]


def test_every_one_of_the_five_pucks_gates_the_frame() -> None:
    """A missing or STALE puck must hold HOME (a None snapshot), never publish a partial or
    frozen command. All five, because the two feet are new roles and the not-ready guard is what
    keeps a dropped ankle from reaching a balance controller as a stale target."""
    for role in vtb.DIRECT_5PT_ROLES:
        b = _calibrated(stale_timeout=0.25)
        del b._raw_hand[role]  # noqa: SLF001
        assert b.snapshot() is None, f"a missing {role} must hold HOME"
        b = _calibrated(stale_timeout=0.25)
        b._raw_hand_t[role] = 0.0  # noqa: SLF001 - epoch = maximally stale
        assert b.snapshot() is None, f"a stale {role} must hold HOME"


def test_nothing_is_published_before_calibration() -> None:
    """`snapshot()` returning None is what makes the consumer hold HOME. It was also the
    "streaming" lie that surfaced on the rig as \"your hand pose is too far\"."""
    b = _fresh()
    b.snapshot()  # calib_delay 0: the first call records "all seen"
    assert b.snapshot() is not None
    b2 = _fresh(calib_delay=1e6)
    assert b2.snapshot() is None, "an un-calibrated bridge must publish NOTHING"
    assert b2.snapshot() is None


def test_too_few_solved_base_stations_refuses_to_calibrate() -> None:
    """The gate is real for five pucks too: one lighthouse leaves a sparse puck's yaw
    under-constrained, and yaw is exactly what the world-yaw autocal reads off the L-R hand
    vector — so the frame looks locked and drifts."""
    b = _fresh()
    b._stations_live = {"LH0"}  # noqa: SLF001
    b.snapshot()
    b.snapshot()
    assert b._calibrated is False  # noqa: SLF001
    assert b.snapshot() is None


def test_the_five_roles_are_active_and_the_serial_map_covers_them() -> None:
    b = vtb.ViveTeleopBridge(_args())
    assert b._roles == list(vtb.DIRECT_5PT_ROLES)  # noqa: SLF001
    assert b._uses_head is True  # noqa: SLF001
    assert b._is_mc is False, "there is no completion in this path"  # noqa: SLF001
    assert sorted(b._serial_to_role.values()) == sorted(vtb.DIRECT_5PT_ROLES)  # noqa: SLF001


# ---------------------------------------------------------------------------
# 5. the refusals — each one a mistake that does NOT fail on its own
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["left_foot_serial", "right_foot_serial"])
def test_a_missing_foot_serial_is_refused(missing) -> None:
    """A guessed foot serial does not fail — it commands a MIRRORED stance with every gate
    green — so there is no default and no first-seen latch at this layer."""
    with pytest.raises(SystemExit):
        vtb.ViveTeleopBridge(_args(**{missing: ""}))


def test_a_duplicated_serial_is_refused_across_all_five_roles() -> None:
    """Two roles sharing a serial silently collapses the dict, leaving one puck unfed forever —
    which surfaces as HOME being held with no hint which role is 'missing'."""
    for over in (
        {"left_foot_serial": "LHR-L"},  # a foot duplicating a hand
        {"right_foot_serial": "LHR-H"},  # a foot duplicating the chest
        {"right_foot_serial": "LHR-LF"},  # the two feet duplicating each other
    ):
        with pytest.raises(SystemExit):
            vtb.ViveTeleopBridge(_args(**over))


def test_hand_point_lever_is_refused_for_the_direct_mode() -> None:
    """5pt is TCP-only: no 0.18 m-lever 5pt checkpoint was ever trained. The two geometries are
    6.7 cm apart per hand at an IDENTICAL wire width, so nothing downstream can catch it."""
    with pytest.raises(SystemExit):
        vtb.ViveTeleopBridge(_args(hand_point="lever"))


# ---------------------------------------------------------------------------
# 6. the RAW tracker feed (for recording a teleop session)
# ---------------------------------------------------------------------------


def test_the_raw_umi_feed_carries_all_five_pucks_tagged_5pt() -> None:
    """A 5-puck session labelled `3pt` would be a file that lies about which pucks were on the
    radio, and a role missing from `UMI_TRACKER_ROLE` would drop a puck from the recording with
    no error anywhere. The role NAMES are collection's own (`left_foot` / `right_foot`), which
    is what puts a teleop-recorded foot in the same file a collection-recorded one lands in."""
    frame = _calibrated().umi_tracker_snapshot(now=1e18)
    assert frame["mode"] == "5pt"
    assert sorted(frame["trackers"]) == ["head", "left", "left_foot", "right", "right_foot"]
    for role, t in frame["trackers"].items():
        assert len(t["pos"]) == 3 and len(t["quat"]) == 4, role
        assert t["stale"] is False, role


def test_the_raw_feed_is_the_MEASUREMENT_not_the_command() -> None:
    """The raw feed must carry the un-anchored puck pose, not the pelvis-frame command: a
    recorder that could pick a key out of a command frame is one refactor from recording the
    command as data."""
    frame = _calibrated().umi_tracker_snapshot(now=1e18)
    np.testing.assert_allclose(frame["trackers"]["left_foot"]["pos"], _CALIB["left_foot"][0], atol=1e-12)


# ---------------------------------------------------------------------------
# 7. non-invasion: the pre-existing modes
# ---------------------------------------------------------------------------


def test_the_pre_existing_wires_are_untouched() -> None:
    """By VALUE, on the payload: 2pt and 3pt keep their exact tags, widths and rotation reps,
    and neither grows a foot role or an mc_* key."""
    for mode, want_pos, want_orn, want_roles in (
        ("2pt", 6, 12, ["left_hand", "right_hand"]),
        ("3pt", 9, 12, ["left_hand", "right_hand", "head"]),
    ):
        b = vtb.ViveTeleopBridge(
            _args(mode=mode, hand_point="tcp" if mode == "3pt" else "lever", left_foot_serial="", right_foot_serial="")
        )
        assert b._roles == want_roles, mode  # noqa: SLF001
        assert b._is_5pt_direct is False, mode  # noqa: SLF001
        pose = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        for role in b._roles:  # noqa: SLF001
            b._raw_hand[role] = pose  # noqa: SLF001
            b._raw_hand_t[role] = 1e18  # noqa: SLF001
        b._calibrated = True  # noqa: SLF001
        snap = b.snapshot()
        assert snap is not None and snap["mode"] == mode
        assert len(snap["pos"]) == want_pos and len(snap["orn"]) == want_orn
        assert not any(k.startswith("mc_") for k in snap), snap.keys()


def test_the_completion_modes_still_read_three_pucks_and_need_no_foot_serial() -> None:
    """`--mode 5pt` / `dense` are the COMPLETION path and are unchanged: three roles, no foot
    serial required, and they still refuse without a chest puck."""
    for mode in ("5pt", "dense"):
        b = vtb.ViveTeleopBridge(_args(mode=mode, left_foot_serial="", right_foot_serial=""))
        assert b._roles == ["left_hand", "right_hand", "head"], mode  # noqa: SLF001
        assert b._is_mc is True and b._is_5pt_direct is False, mode  # noqa: SLF001
        with pytest.raises(SystemExit):
            vtb.ViveTeleopBridge(_args(mode=mode, head_serial=""))


def test_the_2pt_and_3pt_raw_feeds_keep_their_own_tags() -> None:
    for mode, tag, n in (("2pt", "2pt", 2), ("3pt", "3pt", 3)):
        b = vtb.ViveTeleopBridge(
            _args(mode=mode, hand_point="tcp" if mode == "3pt" else "lever", left_foot_serial="", right_foot_serial="")
        )
        for role in b._roles:  # noqa: SLF001
            b._raw_hand[role] = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))  # noqa: SLF001
            b._raw_hand_t[role] = 1e18  # noqa: SLF001
        frame = b.umi_tracker_snapshot(now=1e18)
        assert frame["mode"] == tag
        assert len(frame["trackers"]) == n


# ---------------------------------------------------------------------------
# 8. the launchers — read as TEXT, because they `exec` into the bridge
# ---------------------------------------------------------------------------


@_needs_launchers
def test_the_operator_launcher_offers_the_mode_without_stealing_the_old_spellings() -> None:
    """`MODE=5` / `MODE=5pt` must keep meaning the COMPLETION wire: silently re-pointing them
    would change what a reviewed command line does."""
    src = _LAUNCHER.read_text()
    assert "5pt-direct|5direct|5trk) MODE=5pt-direct ;;" in src
    assert "5|5pt) MODE=5pt ;;" in src, "MODE=5/5pt must still select the completion path"
    assert "--mode 5pt-direct" in src


@_needs_launchers
def test_the_operator_launcher_refuses_the_completion_knobs() -> None:
    """Each of MC_COND_MODE / MC_HAND_LEVER / … names a stage this mode does not have. Accepting
    one silently is how an operator concludes, from an unchanged robot, that a placement does not
    matter."""
    src = _LAUNCHER.read_text()
    for var in ("MC_COND_MODE", "MC_DZ_ANCHOR", "MC_DZ_WRIST", "MC_HAND_LEVER", "MC_LAG_FRAMES", "MC_REF_FRAME", "FOOT_IK"):
        assert var in src.split("for _v in ")[1].split(";")[0], var


@_needs_launchers
def test_both_launchers_default_the_feet_to_this_rigs_paired_pucks() -> None:
    """The SAME pair `run_collection_viz.sh` defaults to, so collection and teleop cannot
    disagree about which ankle is which."""
    assert LEFT_FOOT in _LAUNCHER.read_text() and RIGHT_FOOT in _LAUNCHER.read_text()
    inner = _INNER.read_text()
    assert f'VIVE_UMI_LEFT_FOOT_SERIAL:={LEFT_FOOT}' in inner
    assert f'VIVE_UMI_RIGHT_FOOT_SERIAL:={RIGHT_FOOT}' in inner


def test_the_collection_launcher_agrees_on_the_foot_assignment() -> None:
    """Cross-repo, by reading UMI_collection's own launcher: one room, one pair of pucks, one
    assignment. Skipped by name when that checkout is absent."""
    umi = Path.home() / "FUJI/third_party/UMI_collection/run_collection_viz.sh"
    if not umi.is_file():
        pytest.skip(f"UMI_collection not checked out at {umi}")
    src = umi.read_text()
    assert f'LEFT_FOOT_SERIAL:-{LEFT_FOOT}' in src
    assert f'RIGHT_FOOT_SERIAL:-{RIGHT_FOOT}' in src


@_needs_launchers
def test_the_inner_launcher_forwards_five_serials_and_pins_the_tcp() -> None:
    """The 5-tracker branch must pass both foot serials AND `--hand-point tcp` — the one thing
    no width check downstream can tell you."""
    inner = _INNER.read_text()
    branch = inner.split('if [ "$_MODE_5TRK" = "1" ]; then')[1].split("if [ \"$_MODE_3TRK\"")[0]
    for flag in ("--left-serial", "--right-serial", "--head-serial", "--left-foot-serial", "--right-foot-serial"):
        assert flag in branch, flag
    assert "--hand-point tcp" in branch
    assert "--mc-" not in branch, "there is no completion in this path"


@_needs_launchers
def test_the_three_tracker_detector_does_not_match_the_five_tracker_mode() -> None:
    """`5pt-direct` is its own token, which is what keeps the 3pt/5pt/dense branch (and its
    completion flags) byte-identical."""
    inner = _INNER.read_text()
    assert "3pt|5pt|dense) _MODE_3TRK=1;;" in inner
    assert "5pt-direct) _MODE_5TRK=1;;" in inner


@_needs_launchers
def test_the_inner_launcher_passes_the_frame_knobs_only_on_the_five_tracker_branch() -> None:
    """The frame flags must reach the 5-tracker branch and must NOT appear in the 3-tracker one.

    That split is the "do not touch 3pt-unified-anneal" promise, spelled where it can be
    checked: the bridge refuses the flags on those modes anyway, so a leak here would turn a
    working teleop launch into an immediate exit(2).
    """
    inner = _INNER.read_text()
    five = inner.split('if [ "$_MODE_5TRK" = "1" ]; then')[1].split('if [ "$_MODE_3TRK"')[0]
    three = inner.split('if [ "$_MODE_3TRK" = "1" ]; then')[1]
    for flag in ("--world-yaw-from", "--yaw-trim-deg", "--world-yaw-tol-deg"):
        assert flag in five, flag
        assert flag not in three, f"{flag} must not reach the 3pt/5pt/dense branch"
    assert 'YAW_FROM:-chest' in five, "the CHEST is the default source for this mode"
    assert 'want chest|feet|hands|fixed' in five, "a typo must fail, not silently pick a witness"


@_needs_launchers
def test_the_operator_launcher_documents_and_validates_the_frame_knobs() -> None:
    """The operator meets this mode through `1_vive_bridge.sh`, so the knob and the failure it
    prevents have to be legible there — and a bad YAW_FROM must die before the radio is claimed
    (libsurvive is single-claim: a late exit costs a re-plug)."""
    src_l = _LAUNCHER.read_text()
    assert "YAW_FROM=chest|feet|hands|fixed" in src_l
    assert "YAW_TRIM_DEG" in src_l and "YAW_TOL_DEG" in src_l
    assert "want chest|feet|hands|fixed" in src_l
    # The measured symptom, named where an operator will look for it.
    assert "steps to the RIGHT" in src_l


@_needs_launchers
def test_discovery_forwards_the_mode_so_it_waits_for_five_pucks() -> None:
    """`--list-serials` early-stops as soon as it has `want` trackers, so a 5-puck rig scanned
    as 3pt reports three and EXITS before either foot is ever seen — silently answering the one
    question discovery is being asked."""
    src = _LAUNCHER.read_text()
    head = src.split('if [ "$_LIST" = "1" ]; then')[1].split("fi")[0]
    assert "--mode 5pt-direct" in head
