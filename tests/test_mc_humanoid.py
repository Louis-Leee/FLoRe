"""Contract tests for the offline viewer's WHOLE-BODY motion-completion extension.

The model itself needs a GPU, a conda env and a 104 MB checkpoint, so the end-to-end run is
``scripts/smoke_test_infer_mc.py``. What is pinned HERE is everything that can fail without
any of that, and every one of these is a silent failure mode:

* **inert by default.** The extension must be a same-level addition, so a viewer built
  without the flag has to be byte-for-byte the tool it was — including that every method is
  callable and does nothing, since the render loop calls them unconditionally.
* **the frame it conditions on.** The humanoid is only meaningful if it is executing the
  chunk the page DRAWS. Feeding it a re-derivation instead of the viewer's own display poses
  would put it in a parallel frame that still looks plausible.
* **the placement default.** ``absolute`` is right for this viewer *because* its display home
  is the rig's standing height, and ``home`` is right for the zarr replay. Getting it
  backwards is not a display bug: the commanded hand height is a conditioned channel, so the
  wrong placement makes the model crouch.
* **the refusal.** A layout without all three keypoints has no root channel, and a fabricated
  one would render a confidently wrong robot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from flore.s1_policy.inference.display import DISPLAY_HOME_POS
from flore.s1_policy.inference.mc_humanoid import (
    REQUIRED_ROLES, SERIES, MCHumanoid, _find_umi_replay_dir, options_from_args,
)

_REPO = Path(__file__).resolve().parents[1]
_UMI_REPLAY = _REPO / "third_party" / "UMI_collection" / "replay"
_UMI_MC = _REPO / "third_party" / "UMI_collection" / "motion_completion"


class _Args:
    """Stand-in for the argparse namespace ``options_from_args`` reads."""

    def __init__(self, **kw) -> None:
        defaults = dict(motion_completion=False, mc_urdf=None, mc_series="prediction",
                        mc_cond_mode="absolute", mc_no_wrist_ik=False, mc_no_foot_ik=False,
                        mc_foot_ik_iters=None)
        defaults.update(kw)
        self.__dict__.update(defaults)


class _Poses:
    """Minimal stand-in for viewer._DisplayPoses (only the four fields MC reads)."""

    def __init__(self, position, wxyz) -> None:
        self.predicted_position = np.asarray(position, dtype=np.float64)
        self.predicted_wxyz = tuple(tuple(q) for q in np.asarray(wxyz, dtype=np.float64))
        self.target_position = self.predicted_position + 0.01
        self.target_wxyz = self.predicted_wxyz


def _display_poses(roles=REQUIRED_ROLES, n=50):
    """One drawable chunk per role, at that role's own display home plus a small sweep."""
    out = {}
    for index, role in enumerate(roles):
        home = np.asarray(DISPLAY_HOME_POS[role], dtype=np.float64)
        sweep = np.linspace(0.0, 0.05 * (index + 1), n)
        position = home[None, :] + np.stack([sweep, 0.5 * sweep, -0.2 * sweep], axis=-1)
        wxyz = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
        out[role] = _Poses(position, wxyz)
    return out


# ---------------------------------------------------------------------------
# 1. Inert without the flag
# ---------------------------------------------------------------------------
def test_options_from_args_is_none_without_the_flag():
    """``run_viewer(motion_completion=None)`` is the "page unchanged" path, so the absent
    flag must produce exactly None rather than a dict with enabled=False — the viewer prints
    a DISABLED reason for any dict it is handed, and that would appear on every plain run."""
    assert options_from_args(_Args()) is None
    assert options_from_args(_Args(motion_completion=True)) is not None


def test_options_from_args_maps_every_flag_to_its_constructor_kwarg():
    """The two --mc-no-* flags INVERT, which is the pair a typo would silently swap: the
    result still runs, just with a post-process the operator thinks is off (or on)."""
    options = options_from_args(_Args(
        motion_completion=True, mc_series="target", mc_cond_mode="home",
        mc_urdf="/tmp/x.urdf", mc_no_wrist_ik=True, mc_foot_ik_iters=80))
    assert options == {
        "enabled": True, "urdf_path": "/tmp/x.urdf", "series": "target",
        "cond_mode": "home", "wrist_ik": False, "foot_ik": True, "foot_ik_iters": 80,
    }
    assert options_from_args(_Args(motion_completion=True))["foot_ik"] is True
    assert options_from_args(_Args(motion_completion=True, mc_no_foot_ik=True))["foot_ik"] is False


def test_a_disabled_instance_is_callable_and_does_nothing():
    """The render loop calls poll/render_at every tick and _rebuild_scene calls on_rollout,
    all WITHOUT branching, so a disabled instance must absorb them. It also must not touch
    the server — ``None`` here would raise on any attribute access."""
    mc = MCHumanoid(None, enabled=False)
    assert not mc.available
    assert "not requested" in mc.disabled_reason
    mc.build_gui(None)                       # would raise on a real gui call
    mc.on_rollout(None, {}, (), True)        # would raise if it read `result`
    assert mc.poll(0) is False
    mc.render_at(0)
    mc.clear()
    mc.stop()


def test_the_series_choice_is_validated_at_construction():
    """A typo'd series must fail at launch, not silently fall back to the prediction: the
    'target' run is the CONTROL for a posture that looks wrong, so a silent substitution
    would make the two indistinguishable."""
    with pytest.raises(ValueError, match="series must be one of"):
        MCHumanoid(None, enabled=False, series="predicted")
    for series in SERIES:
        MCHumanoid(None, enabled=False, series=series)


# ---------------------------------------------------------------------------
# 2. The frame it conditions on
# ---------------------------------------------------------------------------
@pytest.fixture()
def mc_modules():
    """``mc_episode`` + ``mc_frames`` — numpy-only, so importable with no torch and no GPU."""
    if not _UMI_MC.is_dir():
        pytest.skip("UMI_collection submodule not checked out")
    for path in (str(_UMI_MC), str(_REPO / "third_party" / "UMI_collection" / "viz")):
        if path not in sys.path:
            sys.path.append(path)
    try:
        import mc_episode
        import mc_frames
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"motion_completion not importable: {exc!r}")
    return mc_episode, mc_frames


def _cond_from_stub(mc_modules, *, series="prediction", home_anchored=True, roles=REQUIRED_ROLES,
                    cond_mode="absolute", n=50, dz_wrist=None, panel_dz=None,
                    placement_note=""):
    """Drive ``MCHumanoid._build_cond`` with no backend and no viser.

    ``dz_wrist`` is the CONSTRUCTOR pin (what the ready-pose frame passes); ``panel_dz`` is
    what the overlay's GUI would hand in. Both are here because their precedence is the
    contract: an operator sweeping dz on the panel must still win over a caller's default.
    """
    from flore.s1_policy.inference.display import wrist_pose_from_stored

    mc = MCHumanoid(None, enabled=False, series=series, dz_wrist=dz_wrist,
                    placement_note=placement_note)
    eef_t_wrist, eef_R_wrist = wrist_pose_from_stored(np.eye(4))
    mc._pending = {
        "display_poses": _display_poses(roles, n),
        "eef_t_wrist": eef_t_wrist,
        "eef_R_wrist": eef_R_wrist,
        "grip": {"prediction": {}, "target": {}},
        "home_anchored": home_anchored,
        "identity": "unit test",
    }
    return mc, mc._build_cond(cond_mode=cond_mode, dz_wrist=panel_dz, dz_anchor="chest",
                              window=30, overlap=10)


def test_the_conditioning_is_the_displayed_chunk_not_a_rederivation(mc_modules):
    """Un-lift the conditioning, walk the dataset's own wrist+mount lever back off it, and
    the result must be EXACTLY the position the viewer drew. Anything else means the
    humanoid is executing a parallel copy of the chunk in some other frame — which still
    renders, and still looks like a robot doing the task."""
    _, F = mc_modules
    mc, cond = _cond_from_stub(mc_modules)
    poses = mc._pending["display_poses"]
    for side, key in (("left", "left_pos"), ("right", "right_pos")):
        drawn = poses[side].predicted_position
        rot = F.quat_wxyz_to_R(np.asarray(poses[side].predicted_wxyz, dtype=np.float64))
        p_wrist, R_wrist = F.wrist_from_eef(drawn, rot, mc._pending["eef_t_wrist"],
                                           mc._pending["eef_R_wrist"])
        p_expected, _ = F.mc_gripper_from_wrist(p_wrist, R_wrist, side)
        np.testing.assert_allclose(cond.unlift(getattr(cond, key)), p_expected, atol=0.0)


def test_the_series_dropdown_selects_which_drawn_chunk_is_completed(mc_modules):
    """prediction and target are DIFFERENT chunks, so the choice has to reach the cond."""
    _, cond_pred = _cond_from_stub(mc_modules, series="prediction")
    _, cond_target = _cond_from_stub(mc_modules, series="target")
    assert not np.allclose(cond_pred.left_pos, cond_target.left_pos)
    # ...and the offset is the stub's own target-vs-prediction shift in XY, with Z exactly
    # cancelled — because the chest anchor NORMALISES height: shifting the whole take up by
    # 1 cm makes it lift 1 cm less. That is the property the absolute default relies on, so
    # it is worth asserting here rather than only in the numeric dz check below.
    np.testing.assert_allclose(cond_target.left_pos - cond_pred.left_pos,
                               np.broadcast_to([0.01, 0.01, 0.0], cond_pred.left_pos.shape),
                               atol=1e-12)


def test_absolute_placement_needs_no_lift_in_this_viewers_display_frame(mc_modules):
    """The whole reason this viewer defaults to ``absolute`` where the zarr replay defaults
    to ``home``: its display home ALREADY is the rig's standing height, so the chest anchor
    derives dz ~= 0 and the robot stands on the grid with its hands on the drawn chunk. If
    DISPLAY_HOME_POS and mc_frames.RIG_CHEST_STAND_Z ever drift apart this is what says so —
    the consequence would be a silently mis-placed take, i.e. a crouch."""
    _, F = mc_modules
    assert DISPLAY_HOME_POS["head"][2] == F.RIG_CHEST_STAND_Z
    assert DISPLAY_HOME_POS["left"][2] == F.RIG_WRIST_STAND_Z
    _, cond = _cond_from_stub(mc_modules, cond_mode="absolute")
    assert cond.cond_mode == "absolute" and cond.dz_anchor == "chest"
    assert abs(cond.dz_wrist) < 0.05, f"chest anchor wanted a real lift: {cond.dz_wrist:+.4f}"


def test_the_ready_pose_frame_pins_the_lift_and_the_panel_still_overrides_it(mc_modules):
    """``--display-frame g1-ready-pose`` FK'd the floor to z=0, so dz is pinned, not derived.

    Deriving it there is actively wrong: ``auto_dz_chest`` aims at ``RIG_CHEST_STAND_Z`` = 1.30
    while the robot's own chest keypoint at that pose sits ~15 cm lower, so the whole take
    would be lifted off the robot and the commanded hands pushed out of the checkpoint's
    training band. The panel's own pin must still win, or an operator sweeping dz would find
    the control dead.
    """
    _, pinned = _cond_from_stub(mc_modules, dz_wrist=0.0)
    assert pinned.dz_wrist == 0.0 and pinned.dz_anchor == "pinned"
    _, panel = _cond_from_stub(mc_modules, dz_wrist=0.0, panel_dz=0.07)
    assert panel.dz_wrist == 0.07, "the panel's pin must beat the caller's default"
    _, derived = _cond_from_stub(mc_modules)
    assert derived.dz_anchor == "chest", "no pin => the anchor still decides"


def test_the_uncalibrated_display_home_placement_warns_by_name(mc_modules):
    """The original bug was SILENT: a confident robot with its arms folded behind its back.

    So the placement that produces it now says so, with the number — otherwise the only way to
    notice is to recognise the posture, which is exactly what took a round trip.
    """
    _, cond = _cond_from_stub(mc_modules)
    joined = " ".join(cond.warnings())
    assert "UNCALIBRATED display homes" in joined
    assert "g1-ready-pose" in joined and "elbows behind the torso" in joined
    # ...and it goes quiet once the caller HAS placed the take.
    _, placed = _cond_from_stub(mc_modules, dz_wrist=0.0, placement_note="G1 deploy ready pose")
    assert not any("UNCALIBRATED" in w for w in placed.warnings()), placed.warnings()
    assert placed.src["flore_placement"] == "G1 deploy ready pose"


def test_the_raw_frame_is_allowed_but_warns_that_the_robot_leaves_the_grid(mc_modules):
    """With the home-anchor toggle OFF the raw absolute poses are completed, which is honest
    but no longer registered with the drawn lines. That must be SAID, not silently different:
    it is the same picture minus one property, so nothing about it looks wrong."""
    _, cond = _cond_from_stub(mc_modules, home_anchored=False)
    assert any("home-anchor is OFF" in w for w in cond.warnings()), cond.warnings()
    _, anchored = _cond_from_stub(mc_modules, home_anchored=True)
    assert not any("home-anchor is OFF" in w for w in anchored.warnings())


def test_the_grid_is_the_action_grid_so_no_resampling_happens(mc_modules):
    """MC is a 30 Hz checkpoint and the v3 action chunk is a 30 Hz grid, so feeding it the
    chunk directly must produce no rate warning and MC's own two-window schedule."""
    _, cond = _cond_from_stub(mc_modules, n=50)
    assert not any("Hz" in w for w in cond.warnings()), cond.warnings()
    assert cond.n == 50 and cond.starts == [0, 20]


# ---------------------------------------------------------------------------
# 3. The refusal
# ---------------------------------------------------------------------------
def test_a_layout_without_all_three_keypoints_is_refused_by_name():
    """The model's root channel (pelvis XY + heading) comes from the CHEST, so a left-only or
    both-hands layout cannot drive it. Refusing beats fabricating a root: a wrong root yaw
    drives the hand error 0.3 mm -> 229 mm while still drawing a confident robot."""

    class _Overlay:
        available = True
        disabled_reason = ""

        def clear(self):
            raise AssertionError("must refuse before touching the scene")

        def on_cond(self, _build):
            raise AssertionError("must refuse before conditioning")

    class _Result:
        layout_mode = "both_hands"
        roles = ()

    mc = MCHumanoid(None, enabled=False)
    mc.enabled = True
    mc.overlay = _Overlay()
    mc.disabled_reason = ""      # clear the "not requested" reason the off-construction set
    assert mc.available
    mc.on_rollout(_Result(), _display_poses(("left", "right")), (), True)
    assert not mc.available
    assert "no head role" in mc.disabled_reason
    assert "both_hands_head" in mc.disabled_reason      # names the fix, not just the fault


def test_the_replay_dir_lookup_is_honest_about_a_missing_submodule():
    """It must return None (-> a printed DISABLED reason) rather than raise, so a checkout
    without the submodule still gets the plain viewer."""
    found = _find_umi_replay_dir()
    if _UMI_REPLAY.is_dir():
        assert found is not None and Path(found) == _UMI_REPLAY
    else:
        assert found is None
