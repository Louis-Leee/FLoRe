"""Contract tests for :mod:`fuji.s1_policy.serving.hl_trace` — the executed-path recorder
and the executed-vs-recorded comparison sheet.

What is pinned here is what decides whether the red line and the saved PNG can be TRUSTED as
evidence about a replay:

  * the trace records only while armed, drops non-finite poses, and rings (never grows
    unbounded) while REPORTING what it dropped;
  * the reference is put on the trace's own clock by NEAREST-neighbour rotation (a
    component-wise rot6d lerp is not a rotation) and CLAMPED, never extrapolated;
  * a rotation error is the representation-free geodesic, and the plotted Euler series is
    unwrapped so a +-180 deg crossing cannot read as a full-scale spike;
  * the path-length statistic exists, because the failure mode that actually shipped once
    (the ``advance_frame`` off-by-one) tracked the SHAPE correctly while travelling only 70%
    of the demonstration — invisible to a per-sample error metric.
"""

from __future__ import annotations

import numpy as np
import pytest

from fuji.s1_policy.serving.hl_trace import (
    ExecutionTrace,
    ReferencePath,
    resample_reference_to_trace,
    rot6d_to_rpy_deg,
    rotation_geodesic_deg,
    trace_stats,
)
from fuji.utils.rot6d import rotmat_to_rot6d, rotvec_to_rotmat

_IDENT6 = rotmat_to_rot6d(np.eye(3))


def _rot6d_about_z(deg: float) -> np.ndarray:
    return rotmat_to_rot6d(rotvec_to_rotmat(np.array([0.0, 0.0, np.radians(deg)])))


# ---------------------------------------------------------------------------
# ExecutionTrace: the recorder
# ---------------------------------------------------------------------------
def test_an_unarmed_trace_records_nothing():
    """The render loop calls ``add`` every tick; without this gate a trace would accumulate
    the minutes of IDLE HOLD before a play, whose flat line dominates every axis range in the
    sheet and buries the motion the operator wants to see."""
    tr = ExecutionTrace()
    for i in range(10):
        tr.add(float(i), np.zeros(3), _IDENT6)
    assert len(tr.snapshot()) == 0
    assert tr.recording is False


def test_arming_clears_the_previous_play_and_rebases_the_clock():
    """Each play is compared against its OWN reference, so concatenating two plays would draw
    a jump between the end of one and the start of the next — motion the arm never made. And
    t=0 must be the arm-start moment in both series, so the clock re-bases on arm."""
    tr = ExecutionTrace()
    tr.arm(100.0)
    tr.add(100.5, np.array([1.0, 0.0, 0.0]), _IDENT6)
    assert len(tr.snapshot()) == 1

    tr.arm(500.0, episode=7, label="play ep7")
    tr.add(500.25, np.array([2.0, 0.0, 0.0]), _IDENT6)
    snap = tr.snapshot()
    assert len(snap) == 1, "arming must discard the previous play"
    assert snap.t[0] == pytest.approx(0.25), "the clock is relative to THIS arm call"
    assert snap.pos[0, 0] == pytest.approx(2.0)
    assert tr.episode == 7 and tr.label == "play ep7"


def test_disarm_keeps_the_samples_so_the_sheet_can_be_saved_afterwards():
    """The operator presses save AFTER the episode finishes, so ending a play must stop
    recording without discarding the evidence."""
    tr = ExecutionTrace()
    tr.arm(0.0)
    tr.add(0.1, np.zeros(3), _IDENT6)
    tr.disarm()
    tr.add(0.2, np.ones(3), _IDENT6)  # ignored — no longer recording
    assert tr.recording is False
    assert len(tr.snapshot()) == 1


def test_a_non_finite_pose_is_skipped_rather_than_logged():
    """A bad FK/parse yields NaN. Logging it would put a NaN in the plotted series (which
    matplotlib draws as a GAP) and in the RMS statistic (which becomes NaN), i.e. one bad tick
    would destroy the whole sheet's numbers."""
    tr = ExecutionTrace()
    tr.arm(0.0)
    tr.add(0.1, np.array([np.nan, 0.0, 0.0]), _IDENT6)
    tr.add(0.2, np.zeros(3), np.full(6, np.inf))
    tr.add(0.3, np.zeros(2), _IDENT6)          # wrong shape
    tr.add(0.4, np.zeros(3), _IDENT6)          # the only good one
    assert len(tr.snapshot()) == 1


def test_the_trace_rings_at_capacity_and_reports_what_it_dropped():
    """The trace lives for the whole process lifetime, so it MUST be bounded — but silently
    truncating would make the sheet look like a complete record of a long run. Oldest-first,
    because the interesting part of an overlong run is what just happened."""
    tr = ExecutionTrace(capacity=5)
    tr.arm(0.0)
    for i in range(9):
        tr.add(float(i), np.array([float(i), 0.0, 0.0]), _IDENT6)
    snap = tr.snapshot()
    assert len(snap) == 5
    assert snap.pos[0, 0] == pytest.approx(4.0), "the OLDEST samples are the ones dropped"
    assert snap.pos[-1, 0] == pytest.approx(8.0)
    assert tr.dropped == 4


def test_len_and_positions_agree_with_snapshot_without_copying_everything():
    """The render loop polls ``len`` every tick and ``positions`` at the redraw cadence; both
    are shortcuts around ``snapshot`` and must not drift from it."""
    tr = ExecutionTrace()
    tr.arm(0.0)
    for i in range(4):
        tr.add(float(i), np.array([float(i), 1.0, 2.0]), _IDENT6)
    snap = tr.snapshot()
    assert len(tr) == len(snap) == 4
    assert np.allclose(tr.positions(), snap.pos)
    assert tr.positions().shape == (4, 3)


def test_an_empty_trace_snapshots_to_correctly_shaped_empty_arrays():
    """The plot path and the readout both touch a snapshot before anything is recorded; a
    ragged empty would raise inside the render loop instead of reading as "nothing yet"."""
    snap = ExecutionTrace().snapshot()
    assert snap.t.shape == (0,) and snap.pos.shape == (0, 3)
    assert snap.rot6d.shape == (0, 6) and snap.grip_m.shape == (0,)
    assert len(snap) == 0


# ---------------------------------------------------------------------------
# Resampling the reference onto the trace clock
# ---------------------------------------------------------------------------
def test_positions_interpolate_and_t_offset_places_trace_zero_at_the_start_frame():
    """A play can begin MID-episode, so the reference's clock must be shifted by the start
    frame's own time — otherwise the whole comparison is offset by however far into the
    demonstration the operator started, and every error number is meaningless."""
    ref = ReferencePath(
        t=np.array([0.0, 1.0, 2.0, 3.0]),
        pos=np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]),
        rot6d=np.repeat(_IDENT6[None], 4, axis=0),
        t_offset=1.0,  # the trace armed at reference time 1.0 s
    )
    pos, _ = resample_reference_to_trace(ref, np.array([0.0, 0.5, 1.0]))
    assert pos[:, 0] == pytest.approx([1.0, 1.5, 2.0])


def test_rotations_are_nearest_neighbour_so_no_interpolated_pose_is_invented():
    """A component-wise lerp of rot6d is NOT a rotation (it leaves the manifold, and
    Gram-Schmidt then silently renormalizes it into some third orientation). At the v3 grid the
    reference is already 33 ms-spaced, so nearest is within half a frame."""
    ref = ReferencePath(
        t=np.array([0.0, 1.0]),
        pos=np.zeros((2, 3)),
        rot6d=np.stack([_rot6d_about_z(0.0), _rot6d_about_z(90.0)]),
    )
    _, rot = resample_reference_to_trace(ref, np.array([0.1, 0.4, 0.6, 0.9]))
    # Every output must be one of the two INPUT rotations, never a blend of them.
    ang = rotation_geodesic_deg(rot, np.repeat(_rot6d_about_z(0.0)[None], 4, axis=0))
    assert ang == pytest.approx([0.0, 0.0, 90.0, 90.0], abs=1e-6)


def test_times_past_the_reference_are_clamped_not_extrapolated():
    """The trace outlasts the demonstration (the arm keeps holding after the last chunk). An
    extrapolated reference would draw commanded motion that never existed and inflate the
    error at exactly the moment the arm is correctly stationary."""
    ref = ReferencePath(
        t=np.array([0.0, 1.0]),
        pos=np.array([[0.0, 0, 0], [1.0, 0, 0]]),
        rot6d=np.repeat(_IDENT6[None], 2, axis=0),
    )
    pos, _ = resample_reference_to_trace(ref, np.array([-5.0, 0.5, 99.0]))
    assert pos[:, 0] == pytest.approx([0.0, 0.5, 1.0]), "clamped to the reference's own ends"


def test_a_single_sample_reference_is_held_rather_than_raising():
    """A one-frame episode (or a start frame at the very end) must degrade to "hold", because
    the save button is reachable in that state and must not raise inside a GUI callback."""
    ref = ReferencePath(t=np.array([2.0]), pos=np.array([[7.0, 0, 0]]), rot6d=_IDENT6[None])
    pos, rot = resample_reference_to_trace(ref, np.array([0.0, 1.0]))
    assert pos[:, 0] == pytest.approx([7.0, 7.0])
    assert rot.shape == (2, 6)


# ---------------------------------------------------------------------------
# Rotation reporting
# ---------------------------------------------------------------------------
def test_the_plotted_euler_series_is_unwrapped_across_the_180_degree_seam():
    """A wiping stroke routinely crosses +-180 deg. A raw Euler series then draws a ~360 deg
    vertical jump where the tool barely moved — the same "never diff a representation at its
    wrap point" trap that produced a phantom jitter finding on this data before."""
    yaws = np.array([170.0, 175.0, 179.0, -179.0, -175.0])
    rpy = rot6d_to_rpy_deg(np.stack([_rot6d_about_z(y) for y in yaws]))
    steps = np.abs(np.diff(rpy[:, 2]))
    assert steps.max() < 10.0, f"unwrapped yaw must be continuous, got steps {steps}"
    assert rpy[-1, 2] > rpy[0, 2], "the series must keep increasing through the seam"


def test_the_rotation_error_is_the_geodesic_angle():
    """Reported as ONE representation-free number (the angle of Ra^T Rb), matching what the
    offline viewer's Metrics panel reports — so the two tools cannot disagree about how far
    off a rotation was."""
    assert rotation_geodesic_deg(_IDENT6[None], _rot6d_about_z(30.0)[None])[0] == pytest.approx(30.0)
    assert rotation_geodesic_deg(_IDENT6[None], _IDENT6[None])[0] == pytest.approx(0.0)
    # Symmetric and sign-free: it is a distance, not a signed per-axis difference.
    a, b = _rot6d_about_z(10.0)[None], _rot6d_about_z(-50.0)[None]
    assert rotation_geodesic_deg(a, b)[0] == pytest.approx(rotation_geodesic_deg(b, a)[0])
    assert rotation_geodesic_deg(a, b)[0] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_a_perfect_replay_reports_zero_error_and_equal_path_lengths():
    tr = ExecutionTrace()
    tr.arm(0.0)
    pts = np.stack([np.linspace(0, 0.3, 10), np.zeros(10), np.zeros(10)], axis=1)
    for i, p in enumerate(pts):
        tr.add(i / 30.0, p, _IDENT6)
    snap = tr.snapshot()
    st = trace_stats(snap, pts, np.repeat(_IDENT6[None], 10, axis=0))
    assert st.n == 10
    assert st.pos_rms_cm == pytest.approx(0.0)
    assert st.rot_mean_deg == pytest.approx(0.0)
    assert st.path_exec_m == pytest.approx(st.path_ref_m)
    assert st.path_exec_m == pytest.approx(0.3)


def test_path_length_catches_a_replay_that_tracked_the_shape_but_travelled_less():
    """THE failure this statistic exists for. The ``advance_frame`` off-by-one made the arm
    trace only ~70% of the demonstration's path length at 4 steps/chunk while every per-sample
    comparison looked plausible — a cursor-vs-wall-clock check could not see it, and neither
    can an error RMS taken against a resampled reference."""
    n = 20
    ref = np.stack([np.linspace(0, 1.0, n), np.zeros(n), np.zeros(n)], axis=1)
    exe = np.stack([np.linspace(0, 0.7, n), np.zeros(n), np.zeros(n)], axis=1)  # same shape, 70%
    tr = ExecutionTrace()
    tr.arm(0.0)
    for i, p in enumerate(exe):
        tr.add(i / 30.0, p, _IDENT6)
    st = trace_stats(tr.snapshot(), ref, np.repeat(_IDENT6[None], n, axis=0))
    assert st.path_ref_m == pytest.approx(1.0)
    assert st.path_exec_m == pytest.approx(0.7)
    assert st.path_exec_m / st.path_ref_m == pytest.approx(0.7, abs=1e-6)


def test_stats_truncate_to_the_shorter_series_instead_of_broadcasting():
    """The reference is resampled onto the trace's clock, so the two are normally equal
    length — but a caller passing a mismatched pair must get a truncated comparison rather
    than a numpy broadcast error inside a GUI callback."""
    tr = ExecutionTrace()
    tr.arm(0.0)
    for i in range(5):
        tr.add(i / 30.0, np.zeros(3), _IDENT6)
    st = trace_stats(tr.snapshot(), np.zeros((3, 3)), np.repeat(_IDENT6[None], 3, axis=0))
    assert st.n == 3


def test_empty_stats_are_zeros_not_nan():
    """``one_line`` is formatted into the sheet title; NaNs there would read as a broken
    plot rather than an empty one."""
    st = trace_stats(ExecutionTrace().snapshot(), np.zeros((0, 3)), np.zeros((0, 6)))
    assert st.n == 0 and st.pos_rms_cm == 0.0 and st.path_ref_m == 0.0
    assert "nan" not in st.one_line().lower()


# ---------------------------------------------------------------------------
# The sheet itself
# ---------------------------------------------------------------------------
def test_saving_with_an_empty_trace_raises_instead_of_writing_a_blank_png(tmp_path):
    """An operator who pressed save with nothing recorded must be TOLD, not handed a file
    that looks like a result."""
    from fuji.s1_policy.serving.hl_trace import plot_trace_vs_reference

    pytest.importorskip("matplotlib")
    ref = ReferencePath(t=np.array([0.0, 1.0]), pos=np.zeros((2, 3)),
                        rot6d=np.repeat(_IDENT6[None], 2, axis=0))
    out = tmp_path / "empty.png"
    with pytest.raises(ValueError, match="fewer than 2 samples"):
        plot_trace_vs_reference(ExecutionTrace().snapshot(), ref, out)
    assert not out.exists()


def test_the_sheet_is_written_with_the_stats_it_reports(tmp_path):
    """End-to-end: a real PNG lands at the returned path (which the page shows the operator),
    and the returned stats describe the data that was plotted."""
    pytest.importorskip("matplotlib")
    from fuji.s1_policy.serving.hl_trace import plot_trace_vs_reference

    n = 40
    t = np.arange(n) / 30.0
    ref_pos = np.stack([np.linspace(0, 0.2, n), np.zeros(n), np.full(n, 0.9)], axis=1)
    ref = ReferencePath(
        t=t, pos=ref_pos,
        rot6d=np.stack([_rot6d_about_z(float(d)) for d in np.linspace(0, 45, n)]),
        grip_m=np.linspace(0.04, 0.09, n),
    )
    tr = ExecutionTrace()
    tr.arm(0.0)
    for i in range(n):  # a 5 mm constant lag in +y, so the error panel has something to show
        tr.add(t[i], ref_pos[i] + np.array([0.0, 0.005, 0.0]),
               _rot6d_about_z(float(np.linspace(0, 45, n)[i])), grip_m=0.05)
    path, st = plot_trace_vs_reference(tr.snapshot(), ref, tmp_path / "sheet.png", title="unit")
    assert path.endswith("sheet.png")
    assert (tmp_path / "sheet.png").stat().st_size > 5_000, "a real figure, not a stub"
    assert st.n == n
    assert st.pos_rms_cm == pytest.approx(0.5, abs=0.02)  # the injected 5 mm lag, IN CM
    assert st.rot_mean_deg == pytest.approx(0.0, abs=1e-6)


def test_the_sheet_creates_its_output_directory(tmp_path):
    """--trace-dir defaults to a path that does not exist yet (``<repo>/data/traces``), and the
    first save must not fail on that."""
    pytest.importorskip("matplotlib")
    from fuji.s1_policy.serving.hl_trace import plot_trace_vs_reference

    tr = ExecutionTrace()
    tr.arm(0.0)
    for i in range(4):
        tr.add(i / 30.0, np.zeros(3), _IDENT6)
    ref = ReferencePath(t=np.array([0.0, 1.0]), pos=np.zeros((2, 3)),
                        rot6d=np.repeat(_IDENT6[None], 2, axis=0))
    out = tmp_path / "deep" / "deeper" / "sheet.png"
    plot_trace_vs_reference(tr.snapshot(), ref, out)
    assert out.exists()
