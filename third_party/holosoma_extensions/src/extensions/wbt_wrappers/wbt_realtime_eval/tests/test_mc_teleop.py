"""`--mode 5pt` / `--mode dense`: 3 VIVE trackers -> motion completion -> 5pt / 29 DoF.

Three things these guard, none of which any downstream width check could catch:

  1. THE KEYPOINT CONTRACT. `mc_frames.select_keypoints("5pt")` must return exactly
     holosoma's `VR_5POINT_BODIES` in order, with exactly `VR_5POINT_OFFSETS_TCP`. The 5pt
     obs is 15D/20D either way, so a reordered or differently-levered set would feed the
     policy a plausible-looking vector forever. Both sides are re-read from their real
     source files here, so this fails if EITHER moves.
  2. THE LATENCY FLOOR. 19 frames is derived from (window, overlap), not chosen. If the
     checkpoint's schedule changes, the derivation must move with it.
  3. NON-INVASION. The 2pt and 3pt wire payloads must be byte-identical, and the two new
     modes must refuse to run without the chest tracker (it is the model's only root
     channel).

pysurvive is injected via PYTHONPATH by the launcher, not pip, so it is stubbed — the
bridge only touches it inside `run_pysurvive`, which these tests never enter. The
Motion_Completion backend is likewise never spawned: every test either exercises pure
logic or injects a fake source. The end-to-end acceptance (bit-exact frame chain, real-time
duty, measured lag) is `vive/mc_teleop_selftest.py`, which needs the model and a take.
"""

from __future__ import annotations

import pathlib
import sys
import types

import numpy as np
import pytest

# Stub the PYTHONPATH-injected USB singleton before the bridge module is imported.
if "pysurvive" not in sys.modules:
    _m = types.ModuleType("pysurvive")
    _m.SimpleContext = object
    _m.survive_simple_serial_number = lambda *_a, **_k: b"?"
    sys.modules["pysurvive"] = _m

from wbt_realtime_eval.vive import mc_teleop_source as mts  # noqa: E402
from wbt_realtime_eval.vive import vive_teleop_bridge as vtb  # noqa: E402

pytest.importorskip("numpy")


def _mc_frames():
    """`mc_frames` out of the UMI_collection checkout, or skip by name."""
    try:
        F, _ = mts._load_mc_modules(mts._DEFAULT_UMI_DIR)  # noqa: SLF001
    except (FileNotFoundError, ImportError) as exc:
        pytest.skip(f"UMI_collection/motion_completion not available: {exc}")
    return F


# ---------------------------------------------------------------------------
# 1. the keypoint contract, both sides re-read from their real sources
# ---------------------------------------------------------------------------

def test_5pt_keypoints_are_holosomas_vr_5point_bodies_in_order() -> None:
    """The completion's `5pt` selection == VR_5POINT_BODIES, same order, same levers.

    Both sides come from their own source of truth: the bodies/offsets from
    `wbt_wrappers_inference.config_values.robot_config` (which the 184D
    ObservationConfig is built against) and the selection from `mc_frames`. The 5pt obs
    is 15D/20D whichever order the points are in, so this equality is the only thing
    standing between a reorder and a silently wrong policy input.
    """
    F = _mc_frames()
    from wbt_wrappers_inference.config_values.robot_config import (
        VR_5POINT_BODIES,
        VR_5POINT_OFFSETS_TCP,
    )

    assert list(F.KEYPOINT_MODES["5pt"]) == list(VR_5POINT_BODIES)
    offsets = F.KEYPOINT_OFFSETS["tcp"]
    for name, want in zip(VR_5POINT_BODIES, VR_5POINT_OFFSETS_TCP, strict=True):
        got = np.asarray(offsets.get(name, np.zeros(3)), dtype=float)
        np.testing.assert_allclose(got, np.asarray(want, dtype=float), atol=0.0)


def test_5pt_is_the_3pt_set_plus_both_feet() -> None:
    """The first three slots keep their 3pt MEANING — that is what makes 5pt a superset.

    Asserted structurally (prefix equality) rather than by re-typing the names, so it
    holds through a gripper respec that moves both lists together.
    """
    F = _mc_frames()
    three, five = list(F.KEYPOINT_MODES["3pt"]), list(F.KEYPOINT_MODES["5pt"])
    assert five[: len(three)] == three
    assert five[len(three) :] == ["left_ankle_roll_link", "right_ankle_roll_link"]


def test_feet_carry_no_lever_in_either_hand_convention() -> None:
    """A foot keypoint IS the body origin. A hand lever models a physical mount and a
    foot has none; a nonzero sole offset would price a point nothing else in the stack
    knows about."""
    F = _mc_frames()
    for conv in ("tcp", "lever"):
        for foot in ("left_ankle_roll_link", "right_ankle_roll_link"):
            assert foot not in F.KEYPOINT_OFFSETS[conv]


# ---------------------------------------------------------------------------
# 2. the latency floor is DERIVED, not chosen
# ---------------------------------------------------------------------------

def test_min_smooth_lag_is_derived_from_the_window_schedule() -> None:
    """19 frames = stride - 1, and the reason is the worst-case inter-window gap.

    Window w finalises at wall frame `20w + 29`; the next lands at `20w + 49`. Commanding
    `f - D` therefore needs `f - D <= 20w + 29` at the worst case `f = 20w + 48`, i.e.
    `D >= 19`. Recomputed here from (window, overlap) so a checkpoint whose schedule
    changes moves the constant instead of quietly invalidating it.
    """
    stride = mts.MC_WINDOW - mts.MC_OVERLAP
    assert stride == mts.MC_STRIDE
    worst_f = mts.MC_STRIDE + mts.MC_WINDOW - 2      # 20w + 48, relative to 20w
    newest = mts.MC_WINDOW - 1                        # 20w + 29, relative to 20w
    assert mts.MC_MIN_SMOOTH_LAG_FRAMES == worst_f - newest
    assert mts.MC_MIN_SMOOTH_LAG_FRAMES == stride - 1


def test_the_schedule_matches_the_checkpoints_own_window_starts() -> None:
    """MC_WINDOW / MC_OVERLAP must be the numbers `mc_frames.window_starts` strides by —
    a mismatch would queue windows at the wrong frames and stitch at the seams."""
    F = _mc_frames()
    starts = F.window_starts(200, mts.MC_WINDOW, mts.MC_OVERLAP)
    assert starts[:4] == [0, mts.MC_STRIDE, 2 * mts.MC_STRIDE, 3 * mts.MC_STRIDE]
    assert F.deployment_lag_frames(mts.MC_WINDOW) == mts.MC_WINDOW
    # The conservative published bound (one full window) must not be SMALLER than the
    # continuity floor, else the docs would contradict the code.
    assert F.deployment_lag_frames(mts.MC_WINDOW) >= mts.MC_MIN_SMOOTH_LAG_FRAMES


def test_below_the_floor_warns_by_name(monkeypatch, capsys) -> None:
    """A too-small lag must SAY it will stall. Silently stuttering a whole-body command
    is worse than running late, so this is a loud path, not a clamp."""
    monkeypatch.setattr(mts, "_load_mc_modules", lambda _d: (object(), _FakeClientModule()))
    msgs: list[str] = []
    mts.MotionCompletionSource(lag_frames=5, log=msgs.append)
    assert any("STALL" in m for m in msgs), msgs
    mts.MotionCompletionSource(lag_frames=mts.MC_STRIDE, log=lambda m: msgs.append("BAD:" + m))
    assert not any(m.startswith("BAD:") and "STALL" in m for m in msgs)


class _FakeClientModule:
    """Stand-in for `mc_backend_client` — never spawns the g1_mosyn interpreter."""

    @staticmethod
    def resolve():
        return {}, "stubbed"

    class MCBackend:
        def __init__(self, *, log=None):
            self.log = log

        def start(self):
            return False

        def stop(self):
            pass


# ---------------------------------------------------------------------------
# 3. the quaternion helpers (the wire's orn is xyzw; MC's is wxyz)
# ---------------------------------------------------------------------------

def test_quat_helpers_round_trip_and_match_the_wire_order() -> None:
    q = np.array([0.5, 0.5, 0.5, 0.5])  # wxyz, unit
    assert mts.quat_wxyz_to_xyzw(q).tolist() == [0.5, 0.5, 0.5, 0.5]
    q2 = np.array([np.cos(0.3), np.sin(0.3), 0.0, 0.0])
    np.testing.assert_allclose(mts.quat_wxyz_to_xyzw(q2), [np.sin(0.3), 0.0, 0.0, np.cos(0.3)])
    # q (x) q^-1 == identity, and the product is Hamilton (holosoma's convention).
    np.testing.assert_allclose(
        mts.quat_mul_wxyz(q2, mts.quat_inv_wxyz(q2)), [1.0, 0.0, 0.0, 0.0], atol=1e-15
    )
    a = np.array([0.9238795, 0.3826834, 0.0, 0.0])   # Rx(45)
    b = np.array([0.9238795, 0.0, 0.3826834, 0.0])   # Ry(45)
    # Hamilton a(x)b != b(x)a — pins that this is not accidentally commutative/flipped.
    assert not np.allclose(mts.quat_mul_wxyz(a, b), mts.quat_mul_wxyz(b, a))


# ---------------------------------------------------------------------------
# 3b. THE READOUT FRAME — the single decision that determines whether the
#     operator's body motion reaches the robot at all
# ---------------------------------------------------------------------------
# Measured on session_2026.08.27_1208_v3 ep 0 with the real model (see
# `_readout_frame_locked`): reading out in the completion's OWN moving root turned a
# 300 mm operator walk into 3.2 mm of commanded motion, because the root conditioning IS
# the chest puck and so the readout frame moves with the operator. These tests reproduce
# that arithmetic with NO model: the completion's output is synthesised, which is the only
# way to state "a rigid translation of the whole body must survive" as an exact equality.

def _fake_source(F, ref_frame, *, n=4):
    """A `MotionCompletionSource` with its output ring filled in by hand.

    `object.__new__` because the real `__init__` spawns the g1_mosyn backend. Everything
    `_five_point_locked` touches is set, and nothing else is — a readout test has no
    business owning a model process.
    """
    src = object.__new__(mts.MotionCompletionSource)
    src._F = F
    src._kp_mode = "5pt"
    src._hand_point = "tcp"
    src._ref_frame = ref_frame
    src._n = n
    # A 30-long stand-in for `load_mc_names`' FK order. Only the SELECTED bodies need real
    # names — `select_keypoints` reaches them by `body_names.index(nm)` — so the rest are
    # filler. That the real order is the model's is asserted by the keypoint-contract tests
    # above and by `mc_frames`' own suite; what is under test here is the readout frame.
    src._body_names = list(F.KEYPOINT_MODES["5pt"])
    src._body_names += [f"_filler{i}" for i in range(30 - len(src._body_names))]
    assert len(src._body_names) == 30
    src._fk_pos = np.zeros((n, 30, 3))
    src._fk_quat = np.zeros((n, 30, 4))
    src._fk_quat[..., 0] = 1.0                      # identity rotations everywhere
    src._joint_ang = np.zeros((n, 29))
    src._root_pos = np.zeros((n, 3))
    src._root_quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    src._valid = np.ones(n, dtype=bool)
    src._joint_names = [f"j{i}" for i in range(29)]
    return src


def _place_body(src, F, base_xyz, *, t):
    """Put the 5 selected bodies at `base_xyz` + the zero-config layout, at frame `t`."""
    for nm, off in (
        ("left_wrist_yaw_link", F.ZERO_CFG_WRIST_REL_PELVIS["left"]),
        ("right_wrist_yaw_link", F.ZERO_CFG_WRIST_REL_PELVIS["right"]),
        ("torso_link", np.array([0.0, 0.0, 0.0])),
        ("left_ankle_roll_link", np.array([0.0, +0.119, F.ZERO_CFG_ANKLE_REL_PELVIS_Z])),
        ("right_ankle_roll_link", np.array([0.0, -0.119, F.ZERO_CFG_ANKLE_REL_PELVIS_Z])),
    ):
        src._fk_pos[t, src._body_names.index(nm)] = np.asarray(base_xyz) + off


_WALK = np.array([0.30, -0.07, 0.02])   # the operator takes a step


def _walk_readouts(F, ref_frame):
    """Frame 0 and frame 1 of the SAME synthetic walk, read out in `ref_frame`.

    Both tests below share this so they differ in NOTHING but the mode — including that the
    completion's root translates with the body, which it really does (the root conditioning
    IS the chest puck; measured 0.2948 m of a commanded 0.30 m). Setting the root in only
    one of the two would make the frozen test pass for the wrong reason: with `_root_pos`
    left at zeros the LIVE frame is accidentally fixed too, and a reverted fix still passes.
    """
    src = _fake_source(F, ref_frame)
    p_ref, R_ref = F.pelvis_ref_pose(np.array([1.3, -2.1, 0.9]), np.eye(3))
    src._anchor = {"pelvis": (p_ref, R_ref)}
    _place_body(src, F, p_ref, t=0)
    _place_body(src, F, p_ref + _WALK, t=1)
    src._root_pos[0] = p_ref
    src._root_pos[1] = p_ref + _WALK
    return (np.asarray(src._five_point_locked(0)["pos"]).reshape(5, 3),
            np.asarray(src._five_point_locked(1)["pos"]).reshape(5, 3))


def test_frozen_readout_preserves_a_rigid_body_translation() -> None:
    """THE regression. The operator's whole body moving by d must move every commanded
    keypoint by exactly d — that is what "I walk forward and the robot reaches forward"
    means, and it is what the 3pt path has always done (`home + puck displacement`)."""
    F = _mc_frames()
    a, b = _walk_readouts(F, "frozen")
    np.testing.assert_allclose(b - a, np.tile(_WALK, (5, 1)), atol=1e-12)


def test_live_readout_cancels_that_translation_which_is_the_bug() -> None:
    """The other half of the same statement, from the same inputs, so the fix cannot be
    silently reverted: with the completion's own root as the frame, a rigid translation
    reads as NO motion at all. Not a hypothetical — it is what the rig did, and what the
    operator saw as "I walk forward and the point does not move"."""
    F = _mc_frames()
    a, b = _walk_readouts(F, "live")
    np.testing.assert_allclose(b, a, atol=1e-12)


def _walk_ghost_roots(F, ref_frame):
    """The REFERENCE GHOST's root, frame 0 vs frame 1 of the same synthetic walk.

    Built exactly like `_walk_readouts` and from the same `_WALK`, because the whole claim
    being tested is that the ghost and the commanded keypoints are ONE picture -- read out in
    the same frame, moving together. Diverging fixtures could not catch them drifting apart.
    """
    src = _fake_source(F, ref_frame)
    p_ref, R_ref = F.pelvis_ref_pose(np.array([1.3, -2.1, 0.9]), np.eye(3))
    src._anchor = {"pelvis": (p_ref, R_ref)}
    _place_body(src, F, p_ref, t=0)
    _place_body(src, F, p_ref + _WALK, t=1)
    src._root_pos[0] = p_ref
    src._root_pos[1] = p_ref + _WALK
    return (np.asarray(src._root_in_readout_frame_locked(0)[0]),
            np.asarray(src._root_in_readout_frame_locked(1)[0]))


def test_the_ghost_root_carries_exactly_the_motion_the_keypoints_carry() -> None:
    """Without a root the ghost can only flex its joints in place -- so a completion taking a
    STEP would draw as one standing still, which is precisely the reading the ghost exists to
    settle ("did the completion ask, or did the controller refuse?"). Same walk, same frame,
    same displacement as `test_frozen_readout_preserves_a_rigid_body_translation`."""
    F = _mc_frames()
    a, b = _walk_ghost_roots(F, "frozen")
    np.testing.assert_allclose(b - a, _WALK, atol=1e-12)
    # A root sitting exactly ON the frozen pelvis reads out as the origin, which is what lets
    # the browser compose this onto the reference clip's root at all. NOTE this fixture puts
    # it there (`_root_pos[0] = p_ref`) to isolate the frame math; the REAL completion's frame-0
    # root does not sit there -- it stands low and tilted (measured 12.7 mm, 3.2 deg), so the
    # drawn ghost starts a couple of centimetres off the pale one. That offset is expected and
    # is ~20-50x smaller than a step; it is the HANDS that match to sub-mm, not the pelvis.
    np.testing.assert_allclose(a, np.zeros(3), atol=1e-12)


def test_a_live_ghost_stands_still_because_live_cancels_the_body_motion() -> None:
    """Not a bug in the ghost: the measured content of `--ref-frame live`. Pinned so nobody
    "fixes" the standing-still ghost by reading its root out in a DIFFERENT frame from the
    keypoints -- that would make the ghost and the command disagree, i.e. make the picture a
    lie, while looking like an improvement."""
    F = _mc_frames()
    a, b = _walk_ghost_roots(F, "live")
    np.testing.assert_allclose(b, a, atol=1e-12)
    np.testing.assert_allclose(a, np.zeros(3), atol=1e-12)


def test_frozen_frame_zero_lands_on_the_pre_arm_gates_own_reference() -> None:
    """Why `frozen` uses `pelvis_ref_pose` and not the model's output root.

    The gate's held reference under HOLOSOMA_INIT_READY_POSE is `default_dof_angles` with
    the 14 arm joints zeroed, so its HAND points are the zero config's wrist + the 0.242 m
    TCP lever. The home the completion starts from is that same zero-config offset in that
    same frame — so reading out in it makes frame 0 land ON the gate reference, and an
    ABSOLUTE-mode operator standing at calibration arms by construction rather than by
    guessing a pose. Measured with the real model: 0.09 / 0.33 mm.
    """
    F = _mc_frames()
    src = _fake_source(F, "frozen")
    p_ref, R_ref = F.pelvis_ref_pose(np.array([0.48, -0.21, 0.79]), np.eye(3))
    src._anchor = {"pelvis": (p_ref, R_ref)}
    _place_body(src, F, p_ref, t=0)
    got = np.asarray(src._five_point_locked(0)["pos"]).reshape(5, 3)
    tcp = np.array([0.242, 0.0, 0.0])     # KEYPOINT_OFFSETS["tcp"] on a wrist body
    np.testing.assert_allclose(got[0], F.ZERO_CFG_WRIST_REL_PELVIS["left"] + tcp, atol=1e-9)
    np.testing.assert_allclose(got[1], F.ZERO_CFG_WRIST_REL_PELVIS["right"] + tcp, atol=1e-9)


def test_frozen_frame_is_yaw_only_so_the_completions_lean_cannot_tilt_it() -> None:
    """`pelvis_ref_pose` must drop pitch/roll. The model's output root carries up to 11 deg
    of tilt mid-episode; at the 0.45 m hand radius that is ~9 cm of commanded motion the
    operator never asked for, which is the second reason the live root is the wrong frame."""
    F = _mc_frames()
    yaw = 0.7
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0],
                   [0.0, 0.0, 1.0]])
    pitch = np.array([[np.cos(0.2), 0.0, np.sin(0.2)], [0.0, 1.0, 0.0],
                      [-np.sin(0.2), 0.0, np.cos(0.2)]])
    p_a, R_a = F.pelvis_ref_pose(np.array([1.0, 2.0, 0.3]), Rz)
    p_b, R_b = F.pelvis_ref_pose(np.array([1.0, 2.0, 5.5]), Rz @ pitch)
    np.testing.assert_allclose(R_a, R_b, atol=1e-12)          # the lean is dropped
    np.testing.assert_allclose(p_a, p_b, atol=1e-12)          # and so is the root's z
    assert p_a[2] == pytest.approx(F.PELVIS_STAND_Z)


def test_the_home_wrist_and_the_readout_frame_are_the_SAME_frame() -> None:
    """They must be, or the readout does not start where the home put it — and the two are
    computed in different files. `home_wrist_pose` calls `pelvis_ref_pose` for exactly this
    reason; this asserts the composition rather than trusting the call."""
    F = _mc_frames()
    root_p0 = np.array([-0.4, 1.9, 0.83])
    yaw = -2.4
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0],
                   [0.0, 0.0, 1.0]])
    p_pelvis, R_pelvis = F.pelvis_ref_pose(root_p0, Rz)
    for side in ("left", "right"):
        p_home, R_home = F.home_wrist_pose(root_p0, Rz, side)
        np.testing.assert_allclose(R_home, R_pelvis, atol=1e-12)
        # The home expressed IN the readout frame is the zero config, exactly.
        local = R_pelvis.T @ (p_home - p_pelvis)
        np.testing.assert_allclose(local, F.ZERO_CFG_WRIST_REL_PELVIS[side], atol=1e-12)


def test_the_dense_COMMAND_is_untouched_by_the_frame_choice() -> None:
    """Joint angles carry no pelvis, so `--mode dense`'s COMMAND must be bit-identical either
    way. Stated as a test because the flag is documented as inert there.

    Scoped to the command keys on purpose. The same dict now also carries `root_pos`/
    `root_quat` for the reference ghost, and those ARE frame-dependent — they have to be, or
    the ghost would be drawn in a different frame from the keypoints it is meant to explain.
    Asserting the whole dict equal would therefore force the ghost to be wrong.
    """
    F = _mc_frames()
    out = []
    for rf in ("frozen", "live"):
        src = _fake_source(F, rf)
        src._kp_mode = "dense"
        src._anchor = {"pelvis": F.pelvis_ref_pose(np.array([1.3, -2.1, 0.9]), np.eye(3))}
        src._joint_ang[1] = np.linspace(-0.5, 0.5, 29)
        src._root_pos[1] = [9.0, 9.0, 9.0]      # nonsense root: must not reach the command
        out.append(src._dense_locked(1))
    for k in ("joint_pos", "joint_vel", "joint_names"):
        assert out[0][k] == out[1][k], f"the dense command's {k} moved with the readout frame"
    # And the viewer-only half really is the frame-dependent one, so this test cannot pass by
    # the root having been silently dropped from the payload.
    assert set(out[0]) - {"joint_pos", "joint_vel", "joint_names"} == {"root_pos", "root_quat"}
    assert out[0]["root_pos"] != out[1]["root_pos"]


def test_an_unknown_readout_frame_raises_instead_of_falling_back(monkeypatch) -> None:
    """Reading out in the wrong frame is invisible until the operator notices the robot
    ignoring them, so a typo must fail at construction — including via the env pin."""
    F = _mc_frames()
    src = _fake_source(F, "frozen")
    assert mts.MC_DEFAULT_REF_FRAME == "frozen"
    assert set(mts.MC_REF_FRAMES) == {"frozen", "live"}
    with pytest.raises(ValueError, match="ref_frame must be one of"):
        mts.MotionCompletionSource.__init__(
            src, keypoint_mode="5pt", ref_frame="pelvis", umi_dir=mts._DEFAULT_UMI_DIR)
    monkeypatch.setenv("MC_REF_FRAME", "nonsense")
    with pytest.raises(ValueError, match="MC_REF_FRAME"):
        mts.MotionCompletionSource.__init__(
            src, keypoint_mode="5pt", umi_dir=mts._DEFAULT_UMI_DIR)


# ---------------------------------------------------------------------------
# 3c. THE CHEST BODY FRAME — the OTHER half of "the robot does not go where I go"
# ---------------------------------------------------------------------------
# `root_cond_from_chest` reads the chest pose's HEADING, and that heading is the ONLY
# absolute puck rotation in the chain: every other rotation is a calibration-relative delta,
# which cancels a constant mount. So the chest bracket's rotation does NOT cancel, and the
# live bridge — which has no converter in front of it — must apply the relabel itself.
# Measured on session_2026.08.27_12.08.03/take_000 (979 frames): the raw puck reads +90.09 deg
# of heading (std 0.41) where the relabelled chest reads +0.20, i.e. exactly the operator's
# facing after the bridge's L-R world-yaw autocal. `mc_frames`' own suite pins the matrix and
# the 90 deg; these tests pin what this module does with it, end to end.

def _cond_source(F, head_frame, *, ref_frame="frozen", cond_mode="home", hand_lever="none"):
    """A source with just enough state for `_cond_frame_locked`. No backend, no model."""
    src = object.__new__(mts.MotionCompletionSource)
    src._F = F
    src._kp_mode = "5pt"
    src._hand_point = "tcp"
    src._ref_frame = ref_frame
    src._head_frame = head_frame
    src._anchor = None
    src._raw0 = {}
    # The PLACEMENT. `home` is both the default and what these chest-relabel tests are about:
    # the relabel is the one absolute puck rotation in the chain either way, and `dz` is
    # exactly 0.0 in home mode, so keeping it here holds these tests to the pre-`absolute`
    # behaviour rather than to whatever the default happens to be later. The absolute branch
    # has its own file (test_teleop_umi_collect.py).
    src._cond_mode = cond_mode
    src._dz_anchor = "hand"
    src._dz_pin = None
    src._dz = 0.0
    # The puck->wrist leg, defaulted OFF here for the same reason: `none` is what shipped, and
    # these tests are about the CHEST relabel, which the hand leg does not touch.
    src._hand_lever = hand_lever
    src._log = lambda _m: None
    return src


def _commanded_tcp(F, src, grip, side_idx):
    """A PERFECT completion's commanded TCP for one hand, in the frozen readout frame.

    The real model reproduces its gripper_base condition to 0.30 mm, so inverting the mount
    off the conditioning pose and applying the 0.242 m lever is a faithful stand-in — and it
    isolates the frame chain, which is what is under test. Uses the source's OWN anchor, so
    the home and the readout frame are the same ones the live path uses.
    """
    pg, qg = grip[side_idx * 2], grip[side_idx * 2 + 1]
    side = ("left", "right")[side_idx]
    pw, Rw = F.wrist_from_mc_gripper(np.asarray(pg)[None, :],
                                    F.quat_wxyz_to_R(np.asarray(qg)[None, :]), side)
    tcp = pw[0] + Rw[0] @ F.KEYPOINT_OFFSETS["tcp"][f"{side}_wrist_yaw_link"]
    p_pelvis, R_pelvis = src._anchor["pelvis"]
    return R_pelvis.T @ (tcp - p_pelvis)


def _raw_pucks(F, hand_step=np.zeros(3)):
    """Two tracker samples: an operator facing robot +x who steps their hands by `hand_step`.

    The chest puck is the RAW one, i.e. `R_puck = H^-1 = H @ H` for a chest body facing +x —
    the thing the live bridge actually receives.
    """
    H = F.CHEST_RELABEL_R
    q_chest = F.R_to_quat_wxyz((H @ H)[None, ...])[0]
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    L0, R0, C0 = np.array([0.4, 0.2, 1.0]), np.array([0.4, -0.2, 1.0]), np.array([0.1, 0.0, 1.3])
    return (
        ((L0, ident), (R0, ident), (C0, q_chest)),
        ((L0 + hand_step, ident), (R0 + hand_step, ident), (C0, q_chest)),
    )


def test_the_live_path_relabels_the_chest_puck_into_the_body_frame() -> None:
    """`head_frame="puck"` must recover the chest BODY heading from a raw puck, and
    `head_frame="body"` must leave it alone. Same input both ways, so the two statements
    cannot both be satisfied by an accident."""
    F = _mc_frames()
    s0, _ = _raw_pucks(F)
    got = {}
    for hf in ("puck", "body"):
        src = _cond_source(F, hf)
        src._cond_frame_locked(0.0, *s0)
        got[hf] = float(np.rad2deg(F.yaw_of_R(src._anchor["pelvis"][1])))
    assert got["puck"] == pytest.approx(0.0, abs=1e-6), (
        f"the relabel was NOT applied: the pelvis heading came out {got['puck']:+.2f} deg for "
        f"an operator facing robot +x")
    assert abs(abs(got["body"]) - 90.0) < 1e-6, (
        f"expected the raw puck to read 90 deg off (that is the bug being fixed), "
        f"got {got['body']:+.2f}")


def test_the_relabel_turns_the_heading_and_touches_nothing_else() -> None:
    """It is rotation-only: the chest is a body reference with no lever, so `root_pos` — the
    pelvis XY the whole home is placed at — must be bit-identical either way. If this ever
    fails, the relabel picked up a translation and the robot moved to a different place."""
    F = _mc_frames()
    s0, _ = _raw_pucks(F)
    roots = []
    for hf in ("puck", "body"):
        src = _cond_source(F, hf)
        grip = src._cond_frame_locked(0.0, *s0)
        roots.append(np.asarray(grip[4]))
        # and the hands still start on the zero config, whichever heading the pelvis has
        for k, side in enumerate(("left", "right")):
            np.testing.assert_allclose(
                _commanded_tcp(F, src, grip, k),
                F.ZERO_CFG_WRIST_REL_PELVIS[side] + [0.242, 0.0, 0.0], atol=1e-9)
    np.testing.assert_array_equal(roots[0], roots[1])


def test_a_raw_chest_puck_sends_the_operators_FORWARD_step_to_the_robots_RIGHT() -> None:
    """THE regression, stated as the operator's own experience.

    Note the frame bug is INVISIBLE at frame 0 (asserted just above): the home and the
    readout frame share whatever heading was fed, so the start pose is exact either way and
    only motion reveals it. That is why it survived a whole rig session.
    """
    F = _mc_frames()
    step = np.array([0.25, 0.0, 0.0])            # the operator pushes both hands FORWARD
    got = {}
    for hf in ("puck", "body"):
        src = _cond_source(F, hf)
        s0, s1 = _raw_pucks(F, hand_step=step)
        a = _commanded_tcp(F, src, src._cond_frame_locked(0.0, *s0), 0)
        b = _commanded_tcp(F, src, src._cond_frame_locked(0.0, *s1), 0)
        got[hf] = b - a
    np.testing.assert_allclose(got["puck"], step, atol=1e-9)      # forward -> forward
    np.testing.assert_allclose(got["body"], [0.0, -step[0], 0.0], atol=1e-9)  # -> its RIGHT


def test_an_unknown_head_frame_raises_instead_of_falling_back() -> None:
    """Same reason the readout frame raises: a wrong value here is a silent 90 deg, and there
    is no env override on purpose — which of the two a call site needs is a fact about where
    its third pose came from, not a preference."""
    F = _mc_frames()
    src = _cond_source(F, "puck")
    assert mts.MC_DEFAULT_HEAD_FRAME == "puck"
    assert set(mts.MC_HEAD_FRAMES) == {"puck", "body"}
    with pytest.raises(ValueError, match="head_frame must be one of"):
        mts.MotionCompletionSource.__init__(
            src, keypoint_mode="5pt", head_frame="chest", umi_dir=mts._DEFAULT_UMI_DIR)


def test_the_offline_drivers_declare_body_because_the_converter_already_relabelled() -> None:
    """The relabel has ORDER 3, so a doubled application does not cancel — it turns the take
    ~120 deg and reads as a bad model. Both zarr-fed drivers must therefore opt out, and the
    live bridge must not. Checked at the call site because the failure is silent at runtime."""
    root = pathlib.Path(mts.__file__).parent
    for name in ("mc_replay_bridge.py", "mc_teleop_selftest.py"):
        src = (root / name).read_text()
        assert 'head_frame="body"' in src, (
            f"{name} feeds CondEpisode's root, which the converter already relabelled -- it "
            f"must pass head_frame=\"body\" or the chest relabel is applied twice")
    bridge = (root / "vive_teleop_bridge.py").read_text()
    assert 'head_frame="puck"' in bridge, (
        "the live bridge hands over the RAW chest puck and must ask for the relabel")


# ---------------------------------------------------------------------------
# 4. the bridge: mode plumbing + non-invasion
# ---------------------------------------------------------------------------

def _args(**over):
    base = dict(
        mode="2pt", left_serial="LHR-L", right_serial="LHR-R", head_serial="LHR-H",
        hand_point="lever", orientation="off", reach_scale=1.0, z_offset=0.0,
        yaw_deg=0.0, no_yaw_autocal=True, stale_timeout=0.0, calib_delay=0.0,
        fresh=False, no_align=True, host="127.0.0.1", port=8770, rate_hz=50.0,
        list_serials=False, list_timeout=5.0,
        mc_lag_frames=20, mc_no_wrist_ik=False, mc_foot_ik=False, mc_umi_dir=None,
        mc_ref_frame=None,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


class _FakeSource:
    """A MotionCompletionSource that returns a canned command, no model involved."""

    def __init__(self, payload):
        self.payload = payload
        self.pushes: list[tuple] = []
        self.resets = 0

    def push(self, world_yaw, left, right, head, now=None):
        self.pushes.append((world_yaw, left, right, head))
        return len(self.pushes) - 1

    def latest(self, now=None):
        return self.payload

    def reset(self):
        self.resets += 1

    def stop(self):
        pass


def _armed(mode, payload):
    """A calibrated bridge in `mode` with all three pucks fresh and a fake source."""
    b = vtb.ViveTeleopBridge(_args(mode=mode, hand_point="tcp"))
    pose = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    for role in ("left_hand", "right_hand", "head"):
        b._raw_hand[role] = pose  # noqa: SLF001
        b._raw_hand_t[role] = 1e18  # noqa: SLF001
    b._calibrated = True  # noqa: SLF001
    b._mc = _FakeSource(payload)  # noqa: SLF001
    return b


def test_5pt_snapshot_is_pos15_orn20_tagged_5pt() -> None:
    payload = {"pos": list(range(15)), "orn": [0.0] * 20, "frame": 7, "lag_s": 0.667,
               "names": ["a", "b", "c", "d", "e"]}
    snap = _armed("5pt", payload).snapshot()
    assert snap is not None
    assert snap["mode"] == "5pt"
    assert len(snap["pos"]) == 15
    assert len(snap["orn"]) == 20
    # The lag is reported ON THE WIRE, so a consumer can see it without reading logs.
    assert snap["mc_lag_s"] == pytest.approx(0.667)
    assert snap["mc_frame"] == 7


def test_dense_snapshot_ships_joint_names_so_nothing_index_aligns() -> None:
    """MC's joint order and the policy's `policy_joint_names` are both real orders and
    they are NOT the same, so the wire carries the names and the remap is by name."""
    payload = {"joint_pos": [0.1] * 29, "joint_vel": [0.0] * 29,
               "joint_names": [f"j{i}" for i in range(29)], "frame": 3, "lag_s": 0.7}
    snap = _armed("dense", payload).snapshot()
    assert snap is not None
    assert snap["mode"] == "dense"
    assert len(snap["joint_pos"]) == 29
    assert len(snap["joint_vel"]) == 29
    assert len(snap["joint_names"]) == 29
    assert "pos" not in snap and "orn" not in snap


def test_mc_modes_hold_home_until_the_first_window_lands() -> None:
    """`latest()` returning None must become a None SNAPSHOT — the browser then holds the
    HOME anchor, exactly as it does for a stale tracker. Never a partial command."""
    b = _armed("5pt", None)
    assert b.snapshot() is None


def test_mc_modes_require_the_chest_tracker() -> None:
    """The chest puck is the model's ONLY root channel (pelvis XY + heading), so 5pt and
    dense must refuse without it rather than run rootless."""
    for mode in ("5pt", "dense"):
        with pytest.raises(SystemExit):
            vtb.ViveTeleopBridge(_args(mode=mode, head_serial=""))


def test_mc_modes_use_three_roles_like_3pt() -> None:
    for mode in ("3pt", "5pt", "dense"):
        b = vtb.ViveTeleopBridge(_args(mode=mode, hand_point="tcp"))
        assert b._roles == ["left_hand", "right_hand", "head"], mode  # noqa: SLF001
        assert b._uses_head is True, mode  # noqa: SLF001
    b2 = vtb.ViveTeleopBridge(_args(mode="2pt"))
    assert b2._roles == ["left_hand", "right_hand"]  # noqa: SLF001
    assert b2._uses_head is False  # noqa: SLF001
    assert b2._is_mc is False  # noqa: SLF001


def test_recalibration_restarts_the_mc_episode() -> None:
    """The completion's home anchor IS its frame 0. Without a restart it would keep
    re-basing onto the operator pose from the LAST calibration, so the commanded body
    jumps by however far they have drifted since."""
    b = _armed("5pt", {"pos": [0.0] * 15, "orn": [0.0] * 20, "frame": 0, "lag_s": 0.0,
                       "names": list("abcde")})
    b._mc_needs_reset = True  # noqa: SLF001 - what _maybe_calibrate sets
    b.snapshot()
    assert b._mc.resets == 1  # noqa: SLF001
    b.snapshot()
    assert b._mc.resets == 1, "reset must fire once per calibration, not every frame"  # noqa: SLF001


def test_2pt_and_3pt_payloads_are_untouched_by_the_new_modes() -> None:
    """Non-invasion, by VALUE: the pre-existing wires keep their exact tags and widths,
    and neither grows an mc_* field."""
    for mode, want_pos, want_orn in (("2pt", 6, 12), ("3pt", 9, 12)):
        b = vtb.ViveTeleopBridge(_args(mode=mode, hand_point="tcp" if mode == "3pt" else "lever"))
        pose = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        for role in b._roles:  # noqa: SLF001
            b._raw_hand[role] = pose  # noqa: SLF001
            b._raw_hand_t[role] = 1e18  # noqa: SLF001
        b._calibrated = True  # noqa: SLF001
        snap = b.snapshot()
        assert snap is not None and snap["mode"] == mode
        assert len(snap["pos"]) == want_pos
        assert len(snap["orn"]) == want_orn
        assert not any(k.startswith("mc_") for k in snap), snap.keys()


# ---------------------------------------------------------------------------
# 5. the browser presets + their HOME fallbacks
# ---------------------------------------------------------------------------

_LIVE_5PT = "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt-live"
_LIVE_DENSE = "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt-live"
#: The BASE tracker's live twin (exp:g1-wbt). Rides the same dense wire as _LIVE_DENSE — the
#: 58D `command` term is its whole tracking front-end — but 628D with no force block.
_LIVE_WBT = "g1-29dof-holosoma-wbt-live"


def test_live_mc_presets_are_their_baked_twins_with_only_the_tracking_slot_swapped() -> None:
    """The whole safety argument: the 26D force block keeps its slot INDICES.

    Both live tuples are built by substituting the tracking slot(s) in the baked tuple, so
    this equality is what makes "the ONNX still sees force_cmd_local at the same offset"
    structural rather than a promise. Widths must land back on 736 / 732.
    """
    from wbt_realtime_eval.presets import get_preset

    for live, baked, dim, swap in (
        (_LIVE_5PT, "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt", 736,
         {"vr_5point_pos_tcp": "vr_5point_pos_tcp_live", "vr_5point_orn": "vr_5point_orn_live"}),
        (_LIVE_DENSE, "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt", 732,
         {"command": "command_live"}),
        # The BASE tracker has no force block to protect, but the SAME argument matters for a
        # different reason: 628D is its replay twin's width too, and that twin is the verified
        # path. Equality here is what proves live and replay are the same policy input modulo
        # one term's source — so a live-teleop regression cannot be blamed on the layout.
        (_LIVE_WBT, "g1-29dof-holosoma-wbt", 628, {"command": "command_live"}),
    ):
        lp, bp = get_preset(live), get_preset(baked)
        assert lp.total_dim == dim == bp.total_dim, live
        assert tuple(swap.get(k, k) for k in bp.obs_keys) == lp.obs_keys, live
        # Every non-tracking slot is the SAME KEY at the SAME INDEX.
        for i, (a, b) in enumerate(zip(bp.obs_keys, lp.obs_keys, strict=True)):
            if a not in swap:
                assert a == b, f"{live} slot {i}: {a!r} became {b!r}"


def test_mc_home_fallbacks_exist_and_are_the_right_width() -> None:
    """A short HOME is silently zero-filled by the TS side (`wbtVrLiveHome` only accepts an
    exact-length array), and a zero 5pt position commands all five points to the pelvis
    origin. So the length is asserted against the preset's own width table, and the build
    entry raises on a mismatch rather than shipping it."""
    from wbt_realtime_eval.main_vive5 import _MODES, _flatten
    from wbt_realtime_eval.presets import term_base_dim
    from wbt_realtime_eval.vive import load_home_anchor

    anchor = load_home_anchor()
    for _mode, (_preset, fields) in _MODES.items():
        for key, field in fields.items():
            assert field in anchor, f"home_anchor.json is missing {field!r} for {key!r}"
            assert len(_flatten(anchor[field])) == term_base_dim(key), (key, field)


def test_mc_home_fallbacks_are_the_baked_idle_default_pose() -> None:
    """The committed 5pt/dense HOMEs must BE the idle_default reference pose, re-derived.

    They are DELIBERATELY not the default_joint_angles pose the 2pt/3pt arrays use (0.289 m
    apart): in the MC modes the bridge anchors on Motion_Completion's own 29-joints-zero
    pose and never reads this file, so the right 'hold still' target is the pose the
    browser's reference ghost holds. Re-derived here from a real bake so the two cannot
    drift; skipped by name when no baked idle_default is on disk.
    """
    import glob

    from wbt_realtime_eval.vive import load_home_anchor

    cands = sorted(glob.glob(
        str(pathlib.Path(__file__).resolve().parents[5] / "dist" / "*_baked" / "idle_default.npz")
    ))
    if not cands:
        pytest.skip("no baked idle_default.npz on disk (build any 5pt bundle first)")
    z = np.load(cands[-1])
    if "vr5_pos_tcp_local" not in z.files:
        pytest.skip(f"{cands[-1]} predates the vr5 bake tables")
    anchor = load_home_anchor()
    np.testing.assert_allclose(
        np.asarray(anchor["home_pos_pelvis_5pt_tcp"], float).reshape(-1),
        z["vr5_pos_tcp_local"][0], atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(anchor["home_orn_quat_xyzw_5pt"], float).reshape(-1),
        z["vr5_orn_local"][0], atol=1e-6,
    )
    dense = np.asarray(anchor["home_command_dense"], float)
    np.testing.assert_allclose(dense[:29], z["joint_pos"][0], atol=1e-6)
    # Velocities are ZERO by construction: a static hold has none, and the live wire
    # supplies finite-differenced ones.
    np.testing.assert_array_equal(dense[29:], np.zeros(29))
    # And the 5pt prefix IS the 3pt-TCP vector of that same bake — the property that makes
    # 5pt a superset rather than a re-derivation.
    np.testing.assert_array_equal(z["vr5_pos_tcp_local"][:, :9], z["vr3_pos_tcp_local"])
    np.testing.assert_array_equal(z["vr5_orn_local"][:, :12], z["vr3_orn_local"])


def test_the_mc_live_presets_do_not_perturb_any_other_preset() -> None:
    """Non-invasion by VALUE. The live tuples are derived by substitution off the baked
    ones, so a typo in a mapping key could leave the SOURCE tuple mutated."""
    from wbt_realtime_eval.presets import PRESETS, get_preset

    for name, dim in (
        ("g1-29dof-holosoma-2pt-wbt", 564),
        ("g1-29dof-holosoma-3pt-hybrid-wbt", 672),
        ("g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt", 680),
        ("g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt-live", 680),
        ("g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt", 736),
        ("g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt", 732),
        ("g1-29dof-holosoma-wbt", 628),
    ):
        assert get_preset(name).total_dim == dim, name
    live_only = {"vr_5point_pos_tcp_live", "vr_5point_orn_live", "command_live"}
    for name, preset in PRESETS.items():
        if name in (_LIVE_5PT, _LIVE_DENSE, _LIVE_WBT):
            continue
        assert not (live_only & set(preset.obs_keys)), name


def test_the_dense_live_key_canonicalizes_onto_the_command_slot() -> None:
    """`command_live` -> strip `_live` -> `command`, with no rename table involved.

    Every dense checkpoint bakes the slot as `command` (or `motion_command`), so the
    actor-obs guard rejects the whole family unless the live key reduces to it. An earlier
    spelling — `command_dense_live` — reduced to `command_dense` and needed an extra canon
    entry to work; the name was changed instead, so the guard has one less special case.
    """
    from wbt_realtime_eval.main_vive3 import _canon_obs_term

    assert _canon_obs_term("command_live".removesuffix("_live")) == "command"
    assert _canon_obs_term("obs00_motion_command") == "command"
    assert _canon_obs_term("obs00_tml_command") == "command"
    # And the keypoint live spellings must NOT collapse onto anything else.
    for k in ("vr_5point_pos_tcp", "vr_5point_orn", "vr_3point_pos_tcp"):
        assert _canon_obs_term(k) == k


# ---------------------------------------------------------------------------
# 6. the REPLAY bridge's episode control (the browser's episode picker)
#
# A recorded session is 100+ takes and the bridge spawns the completion backend at startup,
# so picking a take used to mean a restart -- and a backend respawn -- per take. The bridge
# now READS its socket, and the page's panel drives it. Two things carry all the risk and
# neither is visible at runtime: the RANGE CHECK (an out-of-range index reaching the feed task
# takes the driver down mid-run, from a click in a browser) and the QUEUE (switching from the
# socket task would mutate the completion source between a push and its latest()).
# ---------------------------------------------------------------------------

from wbt_realtime_eval.vive import mc_replay_bridge as mrb  # noqa: E402


def _replay_state(episode: int = 3, n_episodes: int = 128) -> dict:
    return {"episode": episode, "n_episodes": n_episodes, "want_episode": None}


def test_a_picked_episode_is_only_QUEUED_never_applied_by_the_socket() -> None:
    """The socket task records the index and nothing else.

    Loading an episode resets the completion source and re-anchors its home. Doing that from
    the socket task could land between a `push()` and its `latest()`, scoring the new take's
    cond frames against the old window schedule -- a seam that reads as a completion artifact,
    not as a bug here. So `want_episode` is the ONLY thing that may change.
    """
    st = _replay_state()
    line = mrb.apply_control(st, {"cmd": "episode", "index": 27})
    assert st["want_episode"] == 27
    assert st["episode"] == 3, "the socket must not claim the switch already happened"
    assert "27" in line


def test_an_out_of_range_episode_is_refused_and_says_the_range() -> None:
    """The one input that comes straight from a browser. `index` is whatever the page sent, so
    the bound is checked HERE; letting it through would raise inside the feed task, which owns
    the only path that keeps publishing commands."""
    st = _replay_state(episode=3, n_episodes=54)
    for bad in (54, 999, -1):
        line = mrb.apply_control(st, {"cmd": "episode", "index": bad})
        assert st["want_episode"] is None, f"episode {bad} was queued"
        assert "54" in line and "0..53" in line, line


def test_a_malformed_index_is_refused_rather_than_coerced() -> None:
    st = _replay_state()
    for bad in ({"cmd": "episode"}, {"cmd": "episode", "index": "seven"},
                {"cmd": "episode", "index": None}):
        assert "malformed" in mrb.apply_control(st, bad)
        assert st["want_episode"] is None


def test_recalibrate_from_the_page_restarts_the_CURRENT_take() -> None:
    """The page's Recalibrate button predates this bridge and used to be swallowed silently.

    A replay has no operator pose to re-capture, so the honest equivalent is restarting the
    take -- which IS a fresh home anchor, the same thing `--loop` does at the end of a lap.
    Note it must queue the current episode even though that equals `episode`: the feed applies
    any request it finds, so "same index" is a restart and not a no-op.
    """
    st = _replay_state(episode=11)
    assert "11" in mrb.apply_control(st, {"cmd": "recalibrate"})
    assert st["want_episode"] == 11


def test_unknown_and_non_object_control_messages_change_nothing() -> None:
    st = _replay_state()
    for msg in ({"cmd": "shutdown"}, {}, [1, 2, 3], "recalibrate", None, 7):
        assert mrb.apply_control(st, msg), f"{msg!r} was accepted without a word"
        assert st["want_episode"] is None


def test_the_completion_buffers_are_sized_for_the_LONGEST_take() -> None:
    """`MotionCompletionSource` fixes its buffers at construction and `reset()` reallocates at
    the SAME size, so a per-episode size truncates any longer take the page switches to --
    silently, as a motion that stops partway through. Source-level because the alternative is
    spawning the backend."""
    src = pathlib.Path(mrb.__file__).read_text()
    assert "episode_frames=max_frames + MC_WINDOW" in src, \
        "the source is sized per-episode again; a switch to a longer take would truncate"
    assert "(z.ends - z.starts).max()" in src, "max_frames no longer spans the whole dataset"


def test_the_episode_wire_keys_are_the_ONES_THE_PAGE_READS() -> None:
    """The picker only exists if both sides spell the same four keys.

    They are attached OUTSIDE `wire_payload` on purpose -- that is the one wire definition the
    replay shares with the live bridge, and a live rig has no episodes -- which also means no
    width or shape check anywhere would catch a rename. The browser gate is `n_episodes > 0`:
    that is what keeps the panel off the rig's page, so it is asserted from the TS too.
    """
    src = pathlib.Path(mrb.__file__).read_text()
    obs = (pathlib.Path(mrb.__file__).parents[1] / "template" / "WbtObservations.ts").read_text()
    for key in ("episode", "n_episodes", "episode_frames", "episode_frame"):
        assert f'frame["{key}"]' in src, f"the bridge no longer publishes {key}"
        assert f"msg.{key}" in obs, f"the page no longer reads {key}"
    assert "cmd: 'episode', index: idx" in obs, "the page no longer sends the episode command"
    assert "nEpisodes <= 0) return" in obs, \
        "the panel no longer gates on n_episodes, so a LIVE rig page would grow the picker"


def test_a_reset_cannot_land_between_a_cond_SEND_and_its_bookkeeping() -> None:
    """The race that made every mid-stream switch poison a whole episode.

    `_drain_cond` ships its batch with the state lock RELEASED (a ~37 ms round trip under it
    stalls the tracker loop), so a `reset()` used to be able to slip in between the send and
    the `_sent` update. `start_episode` then emptied the backend's cond buffer while the
    previous episode's frames were still landing in it, and the counts diverged by exactly the
    batch: measured live as `window 0 needs cond frames up to 30, only 29 pushed`, after which
    every later window failed "streaming is in order" and the episode published nothing at all.

    Reachable from the rig, not just from the replay picker: a Recalibrate mid-motion is the
    same mid-stream reset. So the ordering is asserted here rather than avoided by the caller.
    """
    import threading
    import time

    F = _mc_frames()
    order: list[str] = []

    class _BlockingBackend:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.gate = threading.Event()

        def push(self, **_kw) -> None:
            order.append("push:enter")
            self.entered.set()
            assert self.gate.wait(5.0), "test gate never opened"
            order.append("push:exit")

        def start_episode(self, _n, **_kw) -> None:
            order.append("start_episode")

    src = object.__new__(mts.MotionCompletionSource)
    src._F = F                      # noqa: SLF001
    src._log = lambda *_a, **_k: None  # noqa: SLF001
    src._n = 60                     # noqa: SLF001
    src._wrist_ik = True            # noqa: SLF001
    src._foot_fix = "none"          # noqa: SLF001
    src._lock = threading.Lock()    # noqa: SLF001
    src._io_lock = threading.Lock()  # noqa: SLF001
    backend = _BlockingBackend()
    src._backend = backend          # noqa: SLF001
    src._sent = 0                   # noqa: SLF001
    grip = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    src._cond_q = __import__("collections").deque([(1, grip)])  # noqa: SLF001

    sender = threading.Thread(target=src._drain_cond, daemon=True)  # noqa: SLF001
    sender.start()
    assert backend.entered.wait(5.0), "the drain never reached the backend"
    resetter = threading.Thread(target=src.reset, daemon=True)
    resetter.start()
    time.sleep(0.2)
    assert "start_episode" not in order, \
        "reset() started a new episode while the previous episode's cond frames were IN FLIGHT"
    backend.gate.set()
    sender.join(5.0)
    resetter.join(5.0)
    assert order == ["push:enter", "push:exit", "start_episode"], order
    # And the count the whole schedule keys on must reflect the NEW episode, not the batch that
    # was in flight when the reset was asked for.
    assert src._sent == 0, src._sent  # noqa: SLF001
