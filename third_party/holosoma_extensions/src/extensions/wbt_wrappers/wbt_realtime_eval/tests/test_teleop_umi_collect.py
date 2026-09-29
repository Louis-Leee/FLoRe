"""Recording a TELEOP session as UMI training data: the placement, and the raw tracker feed.

The three claims that make `8_collect_teleop_umi.sh` produce a file `convert_to_umi.py` +
`run_mc_to_hl_wholebody.sh` accept unchanged, none of which a shape check downstream could
catch:

  1. THE PLACEMENT AGREES WITH THE OFFLINE ONE. `--mc-cond-mode absolute --mc-dz-anchor hand`
     must put the commanded gripper poses exactly where `mc_episode.CondEpisode` puts them
     with `cond_mode="absolute", dz_anchor="hand"`. Both branches produce a valid-looking
     (n,3)/(n,4) cond frame at the same width, so a divergence is invisible until a trained
     policy is 20 cm off — the online path is what the robot did, the offline path is what
     the labels say it did, and only these tests compare them.
  2. `home` IS UNTOUCHED. Every teleop session so far ran in `home`, and the absolute branch
     was added inside the per-frame hot path. `dz` must be EXACTLY 0.0 there and the output
     must still be the home-anchor formula.
  3. THE RAW TRACKER FEED IS UMI'S FRAME, NOT A LOOKALIKE. Same key names, same wxyz
     convention, same per-role `seq`, UMI's role names, and the same `R_world` construction
     collection bakes into `tracker.npy` — because a session recorded in a different world
     frame is silently rotated against every existing dataset.

pysurvive is stubbed (the launcher injects it via PYTHONPATH) and the Motion_Completion
backend is never spawned: `MCBackend.__init__` only resolves paths.
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

if "pysurvive" not in sys.modules:
    _m = types.ModuleType("pysurvive")
    _m.SimpleContext = object
    _m.survive_simple_serial_number = lambda *_a, **_k: b"?"
    sys.modules["pysurvive"] = _m

from wbt_realtime_eval.vive import mc_teleop_source as mts  # noqa: E402
from wbt_realtime_eval.vive import vive_teleop_bridge as vtb  # noqa: E402


def _mc_frames():
    """`mc_frames` out of the UMI_collection checkout, or skip by name."""
    try:
        F, _ = mts._load_mc_modules(mts._DEFAULT_UMI_DIR)  # noqa: SLF001
    except (FileNotFoundError, ImportError) as exc:
        pytest.skip(f"UMI_collection/motion_completion not available: {exc}")
    return F


def _source(**kw):
    """A source with the module loaded but nothing started (no backend process)."""
    _mc_frames()  # skip early with the real reason if the checkout is absent
    try:
        return mts.MotionCompletionSource(keypoint_mode="dense", log=lambda _m: None, **kw)
    except (FileNotFoundError, ImportError) as exc:  # pragma: no cover - covered by the skip
        pytest.skip(f"motion completion unavailable: {exc}")


def _quat(axis: str, deg: float) -> np.ndarray:
    """wxyz quaternion about a principal axis."""
    h = math.radians(deg) / 2.0
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([math.cos(h), *(math.sin(h) * c for c in v)], dtype=float)


# A plausible operator pose in the libsurvive world: hands out in front, apart, chest above.
_LEFT = (np.array([0.55, 0.18, 0.62]), _quat("z", 12.0))
_RIGHT = (np.array([0.57, -0.20, 0.60]), _quat("z", -9.0))
_HEAD = (np.array([0.10, 0.02, 1.02]), _quat("z", 3.0))


# ---------------------------------------------------------------------------
# 1. the placement matches the offline one, per-frame and in the lift
# ---------------------------------------------------------------------------

def test_absolute_placement_matches_mc_episode_frame_zero() -> None:
    """Online `absolute`+`hand` == `CondEpisode.from_traj(cond_mode="absolute")` at frame 0.

    Frame 0 is the only frame the two CAN be compared at without a recorded take, and it is
    also the only frame where the offline `dz` is unambiguous (its `auto_dz_wrist` averages
    the whole episode; a one-frame episode makes that the same number). If this drifts, every
    recorded image sits beside a joint target from a different posture.
    """
    F = _mc_frames()
    ep_mod = _cond_episode_module()
    src = _source(cond_mode="absolute", dz_anchor="hand", hand_lever="tcp")

    # The ONLINE frame chain: what the bridge pushes to the backend for cond frame 0.
    with src._lock:  # noqa: SLF001 - the method is documented as caller-locks
        Lp, Lq, Rp, Rq, root_p, root_q = src._cond_frame_locked(0.0, _LEFT, _RIGHT, _HEAD)  # noqa: SLF001

    # The OFFLINE frame chain over a 1-frame "episode" holding the same poses.
    #   * the chest must arrive already in the BODY frame (`from_poses` documents that the
    #     caller owns the relabel; the online path does it internally, `head_frame="puck"`);
    #   * the eef->wrist leg is the PUCK->WRIST one, and the "stored eef" we hand over is the
    #     puck pose — because `wrist_from_eef(puck, PUCK_T_WRIST, PUCK_R_WRIST)` IS
    #     `wrist_from_puck`, i.e. step 2 of the online chain under `hand_lever="tcp"`. (Before
    #     the lever existed this argument was identity, which matched the old "treat the puck as
    #     the wrist" step — and made the two chains agree on a pose that was 180 deg off the
    #     wrist. Passing the real leg is what makes this comparison mean something.)
    hp, hR = F.chest_body_from_puck(_HEAD[0][None, :], F.quat_wxyz_to_R(_HEAD[1][None, :]))
    cond = ep_mod.CondEpisode.from_poses(
        left_eef_pos=_LEFT[0][None, :], left_eef_rotmat=F.quat_wxyz_to_R(_LEFT[1][None, :]),
        right_eef_pos=_RIGHT[0][None, :], right_eef_rotmat=F.quat_wxyz_to_R(_RIGHT[1][None, :]),
        head_pos=hp, head_rotmat=hR,
        dt=1.0 / 30.0, cond_mode="absolute", dz_anchor="hand",
        eef_t_wrist=F.PUCK_T_WRIST, eef_R_wrist=F.PUCK_R_WRIST,
    )

    assert src.stats()["dz_wrist"] == pytest.approx(float(cond.dz_wrist), abs=1e-9)
    np.testing.assert_allclose(Lp, cond.left_pos[0], atol=1e-9)
    np.testing.assert_allclose(Rp, cond.right_pos[0], atol=1e-9)
    np.testing.assert_allclose(root_p, cond.root_pos[0], atol=1e-9)
    # Rotations are unaffected by a z lift, so they must agree exactly, up to the double cover.
    for got, want in ((Lq, cond.left_quat[0]), (Rq, cond.right_quat[0]),
                      (root_q, cond.root_quat[0])):
        assert min(np.abs(got - want).max(), np.abs(got + want).max()) < 1e-9


def test_absolute_hand_anchor_puts_the_gripper_at_the_standing_height() -> None:
    """`dz_anchor="hand"` is defined as "the frame-0 mounted gripper z lands on 0.95 m".

    Stated as its own test because it is the POSTURE decision: commanded hand height is a
    conditioned channel of the completion, so a lift that lands somewhere else is a robot
    crouching or reaching for no reason the operator asked for.
    """
    F = _mc_frames()
    src = _source(cond_mode="absolute", dz_anchor="hand", hand_lever="tcp")
    with src._lock:  # noqa: SLF001
        Lp, _, Rp, _, _, _ = src._cond_frame_locked(0.0, _LEFT, _RIGHT, _HEAD)  # noqa: SLF001
    assert float(np.mean([Lp[2], Rp[2]])) == pytest.approx(F.RIG_WRIST_STAND_Z, abs=1e-9)


def test_pinned_dz_wins_over_the_anchor() -> None:
    """`--mc-dz-wrist` must override, byte-exactly: it is how online and offline are made equal."""
    src = _source(cond_mode="absolute", dz_anchor="hand", dz_wrist=0.1234, hand_lever="tcp")
    with src._lock:  # noqa: SLF001
        src._cond_frame_locked(0.0, _LEFT, _RIGHT, _HEAD)  # noqa: SLF001
    st = src.stats()
    assert st["dz_wrist"] == pytest.approx(0.1234)
    assert st["dz_pinned"] is True


def test_absolute_preserves_the_operators_hand_separation() -> None:
    """The reason `absolute` exists for collection: `home` substitutes the robot's separation.

    `home` re-bases each wrist onto the G1's own zero-config wrist poses, so the two hands come
    out 0.297 m apart no matter how the operator held theirs. `absolute` must not.
    """
    F = _mc_frames()
    want = float(np.linalg.norm(_LEFT[0][:2] - _RIGHT[0][:2]))
    out = {}
    for mode in ("home", "absolute"):
        src = _source(cond_mode=mode, dz_anchor="hand", hand_lever="tcp")
        with src._lock:  # noqa: SLF001
            Lp, _, Rp, _, _, _ = src._cond_frame_locked(0.0, _LEFT, _RIGHT, _HEAD)  # noqa: SLF001
        out[mode] = float(np.linalg.norm(np.asarray(Lp)[:2] - np.asarray(Rp)[:2]))
    # The mount is a lever, so the gripper separation is not the puck separation; allow for it
    # rather than pretending otherwise. What must hold is that absolute TRACKS the operator and
    # home does not.
    assert abs(out["absolute"] - want) < 0.10
    assert abs(out["home"] - want) > 0.05, (
        f"home should have replaced the operator's {want:.3f} m separation, got {out['home']:.3f}"
    )
    del F


# ---------------------------------------------------------------------------
# 2. non-invasion: home is exactly what it was
# ---------------------------------------------------------------------------

def test_home_is_the_default_and_lifts_nothing() -> None:
    """No env, no argument -> `home`, and `dz` stays EXACTLY 0.0 (not merely small)."""
    src = _source()
    assert src.stats()["cond_mode"] == "home"
    with src._lock:  # noqa: SLF001
        src._cond_frame_locked(0.0, _LEFT, _RIGHT, _HEAD)  # noqa: SLF001
    assert src.stats()["dz_wrist"] == 0.0


def test_home_output_is_the_home_anchor_formula() -> None:
    """The home branch still equals `home_anchor_traj` + mount, computed independently here."""
    F = _mc_frames()
    src = _source(cond_mode="home")
    with src._lock:  # noqa: SLF001
        Lp, _, _, _, _, _ = src._cond_frame_locked(0.0, _LEFT, _RIGHT, _HEAD)  # noqa: SLF001
    hp, hR = F.chest_body_from_puck(_HEAD[0][None, :], F.quat_wxyz_to_R(_HEAD[1][None, :]))
    _, root_R = F.root_cond_from_chest(hp, hR)
    p_home, R_home = F.home_wrist_pose(hp[0], root_R[0], "left")
    # Frame 0 == the home pose itself (zero operator displacement so far), then mounted.
    pg, _ = F.mc_gripper_from_wrist(p_home[None, :], R_home[None, :], "left")
    np.testing.assert_allclose(Lp, pg[0], atol=1e-9)


def test_a_recalibration_re_derives_the_lift() -> None:
    """`reset()` must clear `dz`: the operator may recalibrate at a different hand height, and
    a stale lift would keep placing the take by a pose that is no longer frame 0."""
    src = _source(cond_mode="absolute", dz_anchor="hand", hand_lever="tcp")
    with src._lock:  # noqa: SLF001
        src._cond_frame_locked(0.0, _LEFT, _RIGHT, _HEAD)  # noqa: SLF001
    first = src.stats()["dz_wrist"]
    assert first != 0.0
    # reset() would take the io lock and talk to the backend; clear the episode state the way
    # it does, which is what the assertion is about.
    with src._lock:  # noqa: SLF001
        src._anchor = None  # noqa: SLF001
        src._raw0 = {}  # noqa: SLF001
        src._dz = 0.0  # noqa: SLF001
        higher = (_LEFT[0] + np.array([0.0, 0.0, 0.25]), _LEFT[1])
        src._cond_frame_locked(0.0, higher, _RIGHT, _HEAD)  # noqa: SLF001
    assert src.stats()["dz_wrist"] != pytest.approx(first)


# ---------------------------------------------------------------------------
# 3. the knobs refuse what they cannot do
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["train", "floor", ""])
def test_unknown_dz_anchor_raises(bad: str) -> None:
    """`train` especially: offline it is a diagnostic, here it would crouch a real robot."""
    _mc_frames()
    if bad == "":
        pytest.skip("empty falls back to the default by design (`dz_anchor or DEFAULT`)")
    with pytest.raises(ValueError, match="dz_anchor"):
        mts.MotionCompletionSource(keypoint_mode="dense", dz_anchor=bad, log=lambda _m: None)


def test_unknown_cond_mode_raises() -> None:
    _mc_frames()
    with pytest.raises(ValueError, match="cond_mode"):
        mts.MotionCompletionSource(keypoint_mode="dense", cond_mode="rebase", log=lambda _m: None)


def test_cond_mode_is_settable_per_process(monkeypatch) -> None:
    """MC_COND_MODE, same precedence as MC_REF_FRAME: an operator flips ONE launch."""
    _mc_frames()
    monkeypatch.setenv("MC_COND_MODE", "absolute")
    monkeypatch.setenv("MC_HAND_LEVER", "tcp")  # absolute is refused without it — see below
    st = _source().stats()
    assert st["cond_mode"] == "absolute"
    assert st["hand_lever"] == "tcp"


@pytest.mark.parametrize("bad", ["wrist", "puck", "0.242"])
def test_unknown_hand_lever_raises(bad: str) -> None:
    _mc_frames()
    with pytest.raises(ValueError, match="hand_lever"):
        mts.MotionCompletionSource(keypoint_mode="dense", hand_lever=bad, log=lambda _m: None)


def test_absolute_without_the_lever_is_REFUSED() -> None:
    """The bug this refusal exists for, and the one the rig actually hit.

    `absolute` feeds the operator's ABSOLUTE hand pose, so the constant puck->wrist relabel
    stops cancelling the way it does in `home`'s calibration-relative delta — and that relabel
    is a 180-deg rotation (`mc_frames.PUCK_R_WRIST`, the measured -y,-x,-z axis map). Measured
    on 802 frames of session_2026.09.02_22.38.14/take_000, online vs the offline chain fed the
    same motion: absolute 180.0 deg + 79-86 mm, home 0.0 deg + 14-26 mm. On the robot that is
    the arms thrashing the instant teleop is armed, which reads as a MODEL problem — so it
    cannot be a warning, and the default `hand_lever` cannot be `tcp` either (that would
    silently move a validated teleop path's commanded hand by up to 26 mm).
    """
    _mc_frames()
    with pytest.raises(ValueError, match="requires hand_lever"):
        mts.MotionCompletionSource(keypoint_mode="dense", cond_mode="absolute",
                                   log=lambda _m: None)
    with pytest.raises(ValueError, match="requires hand_lever"):
        mts.MotionCompletionSource(keypoint_mode="dense", cond_mode="absolute",
                                   hand_lever="none", log=lambda _m: None)


def test_the_lever_is_the_converters_own_two_legs() -> None:
    """`hand_lever="tcp"` must apply exactly `mc_frames.wrist_from_puck` to the HANDS ONLY.

    Checked at the level that matters — the commanded gripper pose — against the offline chain
    handed the SAME puck poses with the puck->wrist leg as its `eef_*` argument. That
    composition IS `wrist_from_puck`, so agreement here is the statement "the online commanded
    pose is what mc_to_hl_wholebody.py would build from this motion".

    The chest is excluded on purpose: it is a body reference with no wrist and no TCP, and its
    own relabel is a different matrix (order 3, so a mix-up does not cancel).
    """
    F = _mc_frames()
    src = _source(cond_mode="absolute", dz_anchor="hand", hand_lever="tcp")
    with src._lock:  # noqa: SLF001
        Lp, Lq, Rp, Rq, root_p, _ = src._cond_frame_locked(0.0, _LEFT, _RIGHT, _HEAD)  # noqa: SLF001
    dz = src.stats()["dz_wrist"]
    for side, (p_puck, q_puck), got_p, got_q in (("left", _LEFT, Lp, Lq),
                                                 ("right", _RIGHT, Rp, Rq)):
        p_wr, R_wr = F.wrist_from_puck(p_puck[None, :], F.quat_wxyz_to_R(q_puck[None, :]))
        pg, Rg = F.mc_gripper_from_wrist(p_wr, R_wr, side)
        np.testing.assert_allclose(got_p, F.lift_z(pg, dz)[0], atol=1e-9)
        want_q = F.R_to_quat_wxyz(Rg)[0]
        assert min(np.abs(got_q - want_q).max(), np.abs(got_q + want_q).max()) < 1e-9
    # the chest/root must be the RAW puck position (relabel is rotation-only, lift is z)
    np.testing.assert_allclose(root_p[:2], _HEAD[0][:2], atol=1e-9)


def test_the_lever_leaves_home_rotation_exact_and_only_moves_position() -> None:
    """WHY `home` survived the missing lever for a whole rig bring-up, and what it still costs.

    `home` takes a calibration-relative delta, `(R R0^T) R_home`, in which a constant body-local
    leg cancels identically — so turning the lever on must not move the commanded ROTATION at
    all. What it does move is position, by the ROTATING part of the 52.8 mm lever: measured
    14-26 mm on a real take, which is the residual disagreement with the offline labels.
    """
    F = _mc_frames()
    got = {}
    for lever in ("none", "tcp"):
        src = _source(cond_mode="home", hand_lever=lever)
        with src._lock:  # noqa: SLF001
            src._cond_frame_locked(0.0, _LEFT, _RIGHT, _HEAD)  # noqa: SLF001
            # frame 0 IS the home, so move the hand to expose the rotating lever
            moved = (_LEFT[0] + np.array([0.12, -0.03, 0.05]), _quat("y", 40.0))
            got[lever] = src._cond_frame_locked(0.0, moved, _RIGHT, _HEAD)  # noqa: SLF001
    q_none, q_tcp = np.asarray(got["none"][1]), np.asarray(got["tcp"][1])
    assert min(np.abs(q_none - q_tcp).max(), np.abs(q_none + q_tcp).max()) < 1e-12, (
        "the lever is body-local and MUST cancel in home's delta rotation")
    dp = float(np.linalg.norm(np.asarray(got["none"][0]) - np.asarray(got["tcp"][0])))
    assert 1e-4 < dp <= 2.0 * float(np.linalg.norm(F.PUCK_T_WRIST)) + 1e-9, (
        f"expected a bounded rotating-lever shift, got {dp * 1e3:.1f} mm")


# ---------------------------------------------------------------------------
# 4. the RAW tracker feed is UMI's frame
# ---------------------------------------------------------------------------

class _FakeBridge:
    """Just enough `ViveTeleopBridge` state for `umi_tracker_snapshot` to be exercised."""

    def __init__(self, rpy=(0.0, 0.0, 180.0), uses_head=True):
        import threading
        self._lock = threading.Lock()
        self._roles = ["left_hand", "right_hand"] + (["head"] if uses_head else [])
        self._uses_head = uses_head
        # This fake represents the 2pt/3pt bridge, so the five-tracker mode is off. The raw
        # feed's `mode` tag reads it (a 5-puck session must not be labelled `3pt`), and the
        # 5-role case is covered against a REAL bridge in tests/test_vive_5pt_direct.py rather
        # than by widening this stand-in.
        self._is_5pt_direct = False
        self._raw_hand = {"left_hand": _LEFT, "right_hand": _RIGHT, "head": _HEAD}
        self._raw_hand_t = dict.fromkeys(self._roles, 1000.0)
        self._raw_hand_seq = {"left_hand": 7, "right_hand": 8, "head": 9}
        self._stations_live = {"LH0", "LH1"}
        self._q_umi_world = vtb.quat_from_rpy(*(math.radians(a) for a in rpy))
        self._umi_stale_timeout = 0.25

    snapshot = None


def _snap(bridge, now=1000.0):
    return vtb.ViveTeleopBridge.umi_tracker_snapshot(bridge, now)


def test_raw_feed_has_umi_role_names_and_schema() -> None:
    """The recorder dispatches on these exact keys; `head` is UMI's name for the CHEST puck."""
    snap = _snap(_FakeBridge())
    assert snap["mode"] == "3pt"
    assert set(snap["trackers"]) == {"left", "right", "head"}
    for role, payload in snap["trackers"].items():
        assert set(payload) == {"seq", "t", "pos", "quat", "stale"}, role
        assert len(payload["pos"]) == 3 and len(payload["quat"]) == 4, role
    assert snap["trackers"]["left"]["seq"] == 7
    assert snap["trackers"]["head"]["seq"] == 9


def test_raw_feed_applies_collections_world_alignment() -> None:
    """`WORLD_RPY="0 0 180"` must land on the poses, the same way collection lands it.

    Computed here from UMI's OWN bridge helpers rather than restated, so this fails if either
    side's composition order changes. A silent mismatch rotates a whole dataset by 180 deg —
    which the joint labels survive (they are pelvis-relative) but no `robot0_eef_*` column
    does.
    """
    umi = pytest.importorskip("importlib").import_module  # noqa: F841
    ub = _umi_tracker_bridge_module()
    snap = _snap(_FakeBridge(rpy=(0.0, 0.0, 180.0)))
    q_world = ub.quat_from_rpy(0.0, 0.0, math.radians(180.0))
    for role, src in (("left", _LEFT), ("right", _RIGHT), ("head", _HEAD)):
        want_p = ub.rot_by_quat(q_world, src[0])
        want_q = ub.quat_normalize(ub.quat_mul(q_world, ub.quat_normalize(src[1])))
        np.testing.assert_allclose(snap["trackers"][role]["pos"], want_p, atol=1e-12)
        np.testing.assert_allclose(snap["trackers"][role]["quat"], want_q, atol=1e-12)


def test_identity_world_align_is_the_raw_pose() -> None:
    """The default is identity, so an unset UMI_WORLD_RPY records the libsurvive frame as-is."""
    snap = _snap(_FakeBridge(rpy=(0.0, 0.0, 0.0)))
    np.testing.assert_allclose(snap["trackers"]["left"]["pos"], _LEFT[0], atol=1e-12)
    np.testing.assert_allclose(snap["trackers"]["left"]["quat"], _LEFT[1], atol=1e-12)


def test_raw_feed_reports_the_poses_own_update_time_and_staleness() -> None:
    """`t` is when the PUCK moved, not when the frame was broadcast.

    The converter resamples onto the camera grid; a broadcast stamp would quantise every puck
    to the publisher's tick and hide a stalled one behind a fresh timestamp.
    """
    b = _FakeBridge()
    b._raw_hand_t["right_hand"] = 999.0  # 1 s old
    snap = _snap(b, now=1000.0)
    assert snap["trackers"]["left"]["t"] == 1000.0
    assert snap["trackers"]["right"]["t"] == 999.0
    assert snap["trackers"]["right"]["stale"] is True
    assert snap["trackers"]["left"]["stale"] is False


def test_a_puck_that_never_locked_is_absent_not_zero_filled() -> None:
    """Absence is a FACT: UMI's whole raw layout refuses to zero-fill a missing device."""
    b = _FakeBridge()
    del b._raw_hand["head"]
    snap = _snap(b)
    assert "head" not in snap["trackers"]
    assert set(snap["trackers"]) == {"left", "right"}


@pytest.mark.parametrize(
    ("port", "want"),
    [(None, ["teleop"]), (0, ["teleop"]), (8772, ["teleop", "umi"])],
)
def test_only_a_nonzero_port_starts_the_second_server(monkeypatch, port, want) -> None:
    """Non-invasion: a teleop run that is not being recorded opens exactly one socket.

    `port=None` is every existing caller (an args object that has never heard of the flag —
    a 2pt run, mc_teleop_gate_viz), `0` is how the launcher spells "off", and `8772` is a
    recording session. Tested through `_serve_all` rather than by re-reading the gate
    expression, because the failure being guarded against is a second WS server bound during
    an ordinary deploy run.
    """
    import asyncio
    started: list[str] = []

    async def _fake_teleop(*_a, **_k):
        started.append("teleop")

    async def _fake_umi(*_a, **_k):
        started.append("umi")

    monkeypatch.setattr(vtb, "_serve", _fake_teleop)
    monkeypatch.setattr(vtb, "_serve_umi_tracker", _fake_umi)

    class _Args:
        host, port_, rate_hz = "127.0.0.1", 8770, 90.0
        umi_tracker_rate_hz = 90.0

    args = _Args()
    args.port = 8770
    if port is not None:
        args.umi_tracker_port = port
    asyncio.run(vtb._serve_all(object(), args, _DummyStop()))  # noqa: SLF001
    assert sorted(started) == sorted(want)


class _DummyStop:
    """A `threading.Event` stand-in: the fake servers return immediately, so it is never read."""

    def is_set(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# 5. the console's RE-ANCHOR gate (the thing that makes a take's labels valid)
# ---------------------------------------------------------------------------
# `cond_mode=home` re-bases the hands onto a REFERENCE POSE: online that is the last
# calibration, offline `mc_to_hl_wholebody.py` uses each EPISODE's own frame 0. So a take must
# BEGIN at a calibration or its labels describe a posture the robot was never in — by the
# operator's drift since the last calibration, in position AND rotation, with nothing
# downstream able to see it. `teleop_umi_ctl.s` therefore touches the bridge's recal flag and
# waits for the command wire to go quiet (un-calibrated) and come back (calibrating done, the
# completion producing) before it starts the take. These pin that handshake, including both
# ways it must REFUSE rather than record.


def _ctl_module():
    """`teleop_umi_ctl` out of the force_deploy sandbox (a script dir, not a package)."""
    import importlib.util
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    # tests -> wbt_realtime_eval -> wbt_wrappers -> extensions -> src -> <holosoma_extensions>
    root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
    path = os.path.join(root, "force_deploy", "teleop_umi_ctl.py")
    if not os.path.isfile(path):
        pytest.skip(f"teleop_umi_ctl.py not found at {path}")
    if "teleop_umi_ctl" in sys.modules:
        return sys.modules["teleop_umi_ctl"]
    spec = importlib.util.spec_from_file_location("teleop_umi_ctl", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["teleop_umi_ctl"] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeWire:
    """A stand-in for the bridge's :8770 broadcast: publishes while `live`, silent otherwise.

    Silence is the real signal — `vive_teleop_bridge._serve` sends NOTHING while `snapshot()`
    returns None, which is exactly its un-calibrated state.
    """

    def __init__(self, live: bool = True) -> None:
        self.live = live
        self.port = None
        self._server = None
        self._clients: set = set()

    async def start(self):
        import asyncio

        import websockets
        self._server = await websockets.serve(self._handler, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        self._task = asyncio.create_task(self._pump())
        return self

    async def _handler(self, ws):
        self._clients.add(ws)
        try:
            async for _ in ws:
                pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._clients.discard(ws)

    async def _pump(self):
        import asyncio
        import json
        while True:
            if self.live and self._clients:
                import websockets
                websockets.broadcast(self._clients, json.dumps({"mode": "dense", "seq": 1}))
            await asyncio.sleep(0.05)

    async def stop(self):
        self._task.cancel()
        self._server.close()
        await self._server.wait_closed()


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_reanchor_touches_the_flag_and_waits_for_the_command_to_come_back(tmp_path) -> None:
    ctl = _ctl_module()
    flag = tmp_path / "recal"

    async def go():
        import asyncio
        wire = await _FakeWire(live=True).start()
        try:
            # The fake bridge behaves like the real one: it stops commanding shortly after the
            # flag appears, then resumes once "calibration" has landed.
            async def bridge():
                while not flag.exists():
                    await asyncio.sleep(0.02)
                wire.live = False
                await asyncio.sleep(1.2)
                wire.live = True
            task = asyncio.create_task(bridge())
            ok, detail = await ctl._reanchor(f"ws://127.0.0.1:{wire.port}", str(flag))  # noqa: SLF001
            await task
            return ok, detail
        finally:
            await wire.stop()

    ok, detail = _run(go())
    assert ok, detail
    assert flag.exists(), "the recal flag must be the mechanism — no new control path"


def test_reanchor_refuses_when_the_bridge_ignores_the_flag(tmp_path) -> None:
    """A bridge that keeps commanding never took the flag (wrong flag path, or another bridge
    owns it). Recording then would give the take a stale anchor, silently."""
    ctl = _ctl_module()
    flag = tmp_path / "recal"

    async def go():
        wire = await _FakeWire(live=True).start()  # never goes quiet
        try:
            return await ctl._reanchor(f"ws://127.0.0.1:{wire.port}", str(flag))  # noqa: SLF001
        finally:
            await wire.stop()

    ok, detail = _run(go())
    assert not ok
    assert "never stopped commanding" in detail


def test_reanchor_refuses_when_the_bridge_never_comes_back(tmp_path, monkeypatch) -> None:
    """Calibration did not land (a puck untracked, too few base stations). The operator gets
    told, and NOTHING is recorded — a take with no anchor is worse than no take."""
    ctl = _ctl_module()
    monkeypatch.setattr(ctl, "_RESUME_TIMEOUT_S", 1.5)
    flag = tmp_path / "recal"

    async def go():
        wire = await _FakeWire(live=False).start()  # silent from the start, stays silent
        try:
            return await ctl._reanchor(f"ws://127.0.0.1:{wire.port}", str(flag))  # noqa: SLF001
        finally:
            await wire.stop()

    ok, detail = _run(go())
    assert not ok
    assert "did not resume" in detail


def test_reanchor_refuses_an_unreachable_bridge(tmp_path) -> None:
    ctl = _ctl_module()
    ok, detail = _run(ctl._reanchor("ws://127.0.0.1:1", str(tmp_path / "recal")))  # noqa: SLF001
    assert not ok
    assert "cannot reach the teleop bridge" in detail


def test_once_and_no_recal_do_not_gate(monkeypatch) -> None:
    """`--once` is for scripts (the caller owns the sequence) and `--no-recal` is the operator's
    explicit opt-out; both must leave the recorder's protocol exactly as it was."""
    ctl = _ctl_module()
    seen: dict = {}
    # `main` does `asyncio.run(_run(...))`; replace both so nothing connects.
    monkeypatch.setattr(ctl, "_run", lambda url, task, once, recal: seen.update(recal=recal))
    monkeypatch.setattr(ctl.asyncio, "run", lambda _whatever: 0)
    for argv, want_gate in ((["--once", "start"], False),
                            (["--no-recal"], False),
                            ([], True)):
        seen.clear()
        assert ctl.main(argv) == 0
        assert (seen["recal"] is not None) is want_gate, (argv, seen)
    # and when it IS on, it carries the two addresses the bridge actually uses
    seen.clear()
    ctl.main(["--wire-url", "ws://h:1", "--recal-flag", "/tmp/f"])
    assert seen["recal"] == {"wire_url": "ws://h:1", "flag": "/tmp/f"}


# ---------------------------------------------------------------------------
# helpers that reach into the peer checkouts
# ---------------------------------------------------------------------------

def _load_peer(name: str, *parts: str):
    """Import a module out of the UMI_collection checkout by path, or skip by name.

    Registered in ``sys.modules`` under its OWN name before ``exec_module``, not a synthetic
    one: ``mc_episode`` is a dataclass module, and ``dataclasses`` resolves a string annotation
    by looking the defining class's ``__module__`` up in ``sys.modules`` — under an unregistered
    name that lookup returns None and the import dies inside ``dataclasses``.
    """
    import importlib.util
    import os
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(mts._DEFAULT_UMI_DIR, *parts)  # noqa: SLF001
    if not os.path.isfile(path):
        pytest.skip(f"{name} not found at {path}")
    _mc_frames()  # puts motion_completion/ on sys.path for the peer's own `import mc_frames`
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cond_episode_module():
    """`mc_episode` — the OFFLINE frame chain this module's absolute branch must match."""
    return _load_peer("mc_episode", "motion_completion", "mc_episode.py")


def _umi_tracker_bridge_module():
    """`vive_tracker_bridge` — imported for its quaternion helpers, to avoid restating them."""
    return _load_peer("vive_tracker_bridge", "bridges", "tracker", "vive_tracker_bridge.py")
