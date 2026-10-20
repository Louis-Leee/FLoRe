"""Live-observation FRESHNESS: the planner rings, and the launcher's use of them.

WHY THIS FILE EXISTS. Every ingest source in the deploy stack degrades by REPEATING its last
value, never by going absent:

* a dead camera bridge leaves ``_WSSubscriber.latest()`` returning the same frame, which the
  ingest thread then re-pushes at 30 Hz;
* a dead Bota reader thread leaves the F/T *publisher* broadcasting a frozen payload with a
  fresh ``seq``/``t``, so even a consumer that checked those would see "current";
* an exception in the ingest thread froze every ring at once, permanently, while Predict kept
  returning confident chunks.

None of that was detectable: the image ring held bare arrays with no timestamps, the gripper a
bare float, and ``state_valid`` was unconditionally all-True at deploy so a never-written
gripper (0.0 m, about -2.6 normalized against a checkpoint's own quantiles) was fed to the
model as a real measurement of a closed jaw. These tests pin the age primitive, the gripper
validity flag it feeds, and the launcher wiring that refuses to Predict on a frozen scene.

Measured stakes, on ``ckpt_diff_umi``: swapping the conditioning image moves the predicted
path by ~38% of its own travel and zeroing the force window ~28%, so a frozen observation is
not a cosmetic problem — it is the model planning confidently against a scene that is gone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from flore.s1_policy.data.layout import FORCE_DIM, get_layout
from flore.s1_policy.serving.hl_online import (
    GRIP_MAX_AGE_S,
    OBS_MAX_AGE_S,
    LivePlanner,
    replay_only_model,
)

_REPO = Path(__file__).resolve().parents[1]


def _planner(To: int = 1) -> LivePlanner:
    """A planner with no policy: the rings + layout half is all these tests touch."""
    return LivePlanner(replay_only_model("left_only", obs_history_len=To), obs_hz=30.0)


def _grip_col() -> int:
    slots = get_layout("left_only").state_slots()
    return slots["left"]["gripper"]


# ---------------------------------------------------------------------------
# The age primitive
# ---------------------------------------------------------------------------
def test_obs_ages_are_none_until_each_channel_is_written():
    """``None`` means NEVER WRITTEN, which is a different and worse state than stale: a
    never-written gripper reads a legal 0.0 m, so its absence cannot be inferred from the
    value. Callers have to be able to tell the two apart."""
    lp = _planner()
    ages = lp.obs_ages()
    assert ages == {"image": None, "force": None, "gripper": None}


def test_obs_ages_measure_wall_clock_not_ring_relative_time():
    """The force ring evicts against its NEWEST SAMPLE (correct: the window must be
    span-wide at any push rate). That is exactly why it cannot detect a freeze — a frozen ring
    re-bins to the same window forever. ``obs_ages`` is the wall-clock companion."""
    lp = _planner()
    t0 = 1000.0
    lp.push_force(np.arange(FORCE_DIM, dtype=np.float64), t=t0)
    lp.push_image(np.zeros((224, 224, 3), dtype=np.uint8), t=t0)
    lp.push_gripper("left", 0.05, t=t0)

    ages = lp.obs_ages(t_now=t0 + 5.0)
    assert ages["force"] == pytest.approx(5.0)
    assert ages["image"] == pytest.approx(5.0)
    assert ages["gripper"] == pytest.approx(5.0)


def test_a_frozen_force_ring_still_bins_VALID_which_is_the_whole_point():
    """Regression witness for the hole this file closes. A single sample makes the window
    read valid, and re-binning later never changes that — no wall-clock check exists inside
    the binner, by design. So validity alone can never tell you the sensor died; only the age
    can."""
    lp = _planner()
    lp.push_force(np.full(FORCE_DIM, 7.0), t=500.0)

    batch, _, _ = lp.build_batch()
    assert bool(batch["force_history_valid"].item()) is True
    # ... and still "valid" an hour later, with the same numbers.
    again, _, _ = lp.build_batch()
    assert bool(again["force_history_valid"].item()) is True
    assert np.allclose(again["force_history"].numpy(), batch["force_history"].numpy())
    # The age is the only thing that moved.
    assert lp.obs_ages(t_now=500.0 + 3600.0)["force"] == pytest.approx(3600.0)


def test_gripper_age_reports_the_OLDEST_role_so_one_dead_jaw_is_not_masked():
    """NOTE: no layout in this repo has two gripper roles today — even ``both_hands`` exposes
    only ``('left',)`` — so the multi-role aggregation is exercised by seeding the latch dict
    directly. It is written defensively because the arithmetic that is right for one jaw (take
    the only stamp) silently becomes wrong for two (a live jaw would mask a dead one), and that
    is not the kind of bug to discover on the first bimanual checkpoint."""
    lp = _planner()
    assert lp.layout.gripper_roles == ("left",)

    lp._t_grip = {"left": 100.0, "right": None}
    # One role never written -> the aggregate is "never written", NOT the live role's age.
    assert lp.obs_ages(t_now=100.0)["gripper"] is None

    lp._t_grip = {"left": 100.0, "right": 90.0}
    assert lp.obs_ages(t_now=100.0)["gripper"] == pytest.approx(10.0)  # the older one wins


# ---------------------------------------------------------------------------
# state_valid: the deploy side of the training converter's unmeasured-gripper flag
# ---------------------------------------------------------------------------
def test_gripper_state_is_INVALID_until_a_width_is_measured():
    """A missing :8784 bridge used to feed 0.0 m through as a real measurement. The model
    zeroes invalid entries after normalization (``state * state_valid``), which is what
    training did for a dead encoder — so False here means 'unmeasured', not 'closed'."""
    lp = _planner()
    obs, _, _ = lp.snapshot_obs()
    gcol = _grip_col()
    assert not obs.state_valid[:, gcol].any()
    # The rotation channels are untouched by the gripper verdict.
    assert obs.state_valid[:, 0:6].all()


def test_gripper_state_is_VALID_right_after_a_push():
    lp = _planner()
    lp.push_gripper("left", 0.05)
    lp.push_proprio_from_anchor()
    obs, _, _ = lp.snapshot_obs()
    gcol = _grip_col()
    assert obs.state_valid[:, gcol].all()
    assert obs.state[-1, gcol] == pytest.approx(0.05 / 0.095)


def test_gripper_state_goes_INVALID_once_the_latch_is_stale():
    """The bridge dying mid-run pinned the last width forever. Age, not value, catches it."""
    import time as _time

    lp = _planner()
    lp.push_gripper("left", 0.05, t=_time.monotonic() - (GRIP_MAX_AGE_S + 5.0))
    lp.push_proprio_from_anchor()
    obs, _, _ = lp.snapshot_obs()
    gcol = _grip_col()
    assert not obs.state_valid[:, gcol].any()
    # The VALUE is still the last measurement — only its credibility changed.
    assert obs.state[-1, gcol] == pytest.approx(0.05 / 0.095)


def test_the_staleness_thresholds_are_tunable_per_planner():
    """``hl_serving`` drives all three consumers (page readout, Predict refusal, this flag)
    from one ``--obs-max-age``, so they can never disagree about what current means."""
    import time as _time

    lp = _planner()
    lp.grip_max_age_s = 1e9
    lp.push_gripper("left", 0.05, t=_time.monotonic() - 3600.0)
    lp.push_proprio_from_anchor()
    obs, _, _ = lp.snapshot_obs()
    assert obs.state_valid[:, _grip_col()].all()


# ---------------------------------------------------------------------------
# The launcher's verdict helper
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hlsrv():
    """Import scripts/hl_serving.py as a module (its heavy viser/IK imports are all inside
    ``_run_viser``, so this is ~1 s and touches no hardware)."""
    spec = importlib.util.spec_from_file_location("_hlsrv_under_test", _REPO / "scripts" / "hl_serving.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hlsrv_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakePlanner:
    def __init__(self, ages):
        self._ages = ages
        self.obs_max_age_s = OBS_MAX_AGE_S
        self.grip_max_age_s = GRIP_MAX_AGE_S

    def obs_ages(self, t_now=None):
        return self._ages


def _reasons(hlsrv, ages):
    return hlsrv._obs_freshness(_FakePlanner(ages), max_age_s=OBS_MAX_AGE_S,
                                grip_max_age_s=GRIP_MAX_AGE_S)


def test_fresh_observation_blocks_nothing(hlsrv):
    reasons, readout = _reasons(hlsrv, {"image": 0.02, "force": 0.005, "gripper": 0.03})
    assert reasons == []
    assert "!" not in readout


def test_a_stale_camera_or_ft_blocks_and_a_stale_gripper_does_not(hlsrv):
    """The asymmetry is deliberate: refuse when the model would be guessing (image ~38% of
    predicted-path sensitivity, force ~28%), degrade when it can absorb it (gripper 0.4%, and
    ``snapshot_obs`` already flags it unmeasured)."""
    reasons, _ = _reasons(hlsrv, {"image": 9.0, "force": 0.005, "gripper": 0.03})
    assert len(reasons) == 1 and "camera" in reasons[0]

    reasons, _ = _reasons(hlsrv, {"image": 0.02, "force": 9.0, "gripper": 0.03})
    assert len(reasons) == 1 and "F/T" in reasons[0]

    reasons, readout = _reasons(hlsrv, {"image": 0.02, "force": 0.005, "gripper": 99.0})
    assert reasons == [], "a stale gripper must not refuse a Predict"
    assert "grip" in readout and "!" in readout


def test_a_channel_that_never_arrived_blocks_too(hlsrv):
    reasons, readout = _reasons(hlsrv, {"image": None, "force": None, "gripper": None})
    assert len(reasons) == 2
    assert all("no frame ever arrived" in r for r in reasons)
    assert "none" in readout


# ---------------------------------------------------------------------------
# PURE-POSITION (FLORE_DROP_FORCE): the F/T must leave the BLOCKING set
# ---------------------------------------------------------------------------
def test_a_pure_position_checkpoint_is_not_blocked_by_an_unwritten_force_ring(hlsrv):
    """A force-free checkpoint never writes the force ring AT ALL, so gating on its age
    refused every Predict for the whole session.

    Not a hypothetical and not merely vacuous: with ``force_roles == ()``,
    ``_pick_force_channels`` returns None for every frame (it concatenates an empty list) and
    ``require_6dof`` is False, so neither ``push_force`` branch in ``_push_force_from_msg``
    runs. ``obs_ages()['force']`` therefore stays None forever. Observed live on
    ``ckpt_noforce``: ``self-check: {'force': 'ok (6dof)', ...}`` on a healthy bridge, then
    ``REFUSED — stale observation: F/T :8771: no frame ever arrived`` on every press."""
    never_written = {"image": 0.02, "force": None, "gripper": 0.03}

    reasons, readout = hlsrv._obs_freshness(
        _FakePlanner(never_written), max_age_s=OBS_MAX_AGE_S,
        grip_max_age_s=GRIP_MAX_AGE_S, consumes_force=False)
    assert reasons == [], f"a pure-position checkpoint must still Predict, got {reasons}"
    assert "F/T n/a" in readout, "a not-consumed sensor must not read like a dead one"

    # ... and the force-ful default is unchanged, so this cannot silently disarm the real gate.
    reasons, readout = hlsrv._obs_freshness(
        _FakePlanner(never_written), max_age_s=OBS_MAX_AGE_S, grip_max_age_s=GRIP_MAX_AGE_S)
    assert len(reasons) == 1 and "F/T" in reasons[0]
    assert "F/T none" in readout


def test_the_camera_blocks_a_pure_position_checkpoint_too(hlsrv):
    """Dropping the F/T must not drop the CAMERA: a force-free model still consumes images,
    which is exactly the line ``_self_check``'s ``force_advisory`` draws at startup."""
    for ages in ({"image": None, "force": None, "gripper": 0.03},
                 {"image": 9.0, "force": None, "gripper": 0.03}):
        reasons, _ = hlsrv._obs_freshness(
            _FakePlanner(ages), max_age_s=OBS_MAX_AGE_S,
            grip_max_age_s=GRIP_MAX_AGE_S, consumes_force=False)
        assert len(reasons) == 1 and "camera" in reasons[0]


def test_an_empty_force_role_tuple_yields_no_channels(hlsrv):
    """The upstream cause, pinned directly: this is why the ring is never written."""
    msg = {"left": [1.0, 2.0, 3.0], "left_pipe": [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]}
    assert hlsrv._pick_force_channels(msg, (), False) is None
    assert hlsrv._pick_force_channels(msg, (), True) is None
    # The LEFT fallback the F/T PANEL uses does produce a reading off the same message, so a
    # pure-position run still shows the operator the sensor the 40 N low-level stop reads.
    shown = hlsrv._pick_force_channels(msg, ("left",), False)
    assert shown is not None and shown.shape == (6,)


def test_the_ft_panel_does_not_go_dark_on_a_pure_position_checkpoint(src: str):
    """The panel must read a display-only role tuple, not the model's (possibly empty) one."""
    assert '_display_force_roles = force_roles or ("left",)' in src
    assert "_pick_force_channels(fmsg, _display_force_roles, require_6dof)" in src, (
        "the F/T panel is back on the model's force_roles; it goes dark when they are empty")


def test_every_freshness_gate_passes_the_consumes_force_verdict(src: str):
    """All THREE call sites, or the one that was missed refuses a pure-position Predict."""
    assert src.count("consumes_force=_consumes_force") == 3, (
        "a _obs_freshness call site is not passing consumes_force")
    assert "_consumes_force = bool(force_roles)" in src


# ---------------------------------------------------------------------------
# Launcher wiring — source-level, matching tests/test_act_temporal_ensemble.py's approach
# for this script (its callbacks are closures inside _run_viser and cannot be reached without
# building a viser page).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def src() -> str:
    return (_REPO / "scripts" / "hl_serving.py").read_text()


def test_the_ingest_loop_cannot_die_on_one_bad_frame(src: str):
    """It is the ONLY writer of the image and proprio rings, and it had no exception guard at
    all — one `float(None)` from a bridge publishing ``"open": null`` killed it permanently
    while Predict kept working off the frozen ring."""
    body = src.split("def _ingest_loop():", 1)[1].split("threading.Thread(target=_ingest_loop", 1)[0]
    assert "try:" in body and "except Exception" in body, (
        "_ingest_loop lost its per-tick exception guard — an escaping error freezes the whole "
        "observation permanently and silently")
    assert "_ingest_once(" in body


def test_predict_refuses_a_stale_observation(src: str):
    body = src.split("def _predict():", 1)[1].split("def _run_staged():", 1)[0]
    assert "_obs_freshness(" in body, "the Predict path no longer checks observation freshness"
    assert "REFUSED" in body
    # ... and the refusal must not apply to the paths that read no observation.
    assert "args.offline" in body and "replay_only" in body and "_replay_source_active()" in body


def test_the_sequential_loop_checks_freshness_and_reports_open_loop(src: str):
    body = src.split("def _run_to_completion():", 1)[1].split("\n    def ", 1)[0]
    assert "_obs_freshness(" in body, (
        "sequential rollout would keep planning off a frozen scene with nobody watching")
    assert "if not _resync_anchor_from_robot():" in body, (
        "the sequential loop must not DISCARD the resync verdict the way it used to — "
        "_predict logs OPEN-LOOP loudly and this path was silent")


def test_the_wrt_start_reference_is_relatched_with_the_arms_zeroed_anchor(src: str):
    """``commit_role`` deliberately never touches ``_start_rot6d``, so without this call the
    state channel stayed referenced to home_anchor's CROUCH wrist rotation — 47.08 deg from the
    pose the robot actually holds — for the whole run unless the operator pressed the button."""
    assert src.count("planner.reset_start_pose()") >= 2, (
        "the startup anchor re-seed must re-latch the wrt_start reference, not rely on the "
        "operator pressing ↺ Reset start pose")


def test_the_page_shows_observation_age_regardless_of_the_ft_view_toggle(src: str):
    """The old page exposed the :8771 age only when 'Show low-level F_meas' was UNTICKED, and
    it defaults TICKED — so by default nothing could reveal a frozen sensor or camera."""
    assert 'obs_fresh = gui.add_text("obs age"' in src
    assert "obs_fresh.value" in src
