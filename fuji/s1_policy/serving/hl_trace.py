"""Execution TRACE for the online HL planner — "what did the arm actually do?".

The planner already knows what it COMMANDED (the recorded chunk, drawn as the green
episode path) and what the robot REPORTS (the RecordTap :8779 joint broadcast, FK'd to the
measured LEFT TCP). This module is the thin recorder + comparator between the two, so a
replay can be judged instead of eyeballed:

  * :class:`ExecutionTrace` — a bounded, thread-safe append log of measured TCP samples,
    written from the render loop (the one thread that already FKs the live wrist every
    tick, so recording costs a list append and no extra FK).
  * :func:`resample_reference_to_trace` — puts the RECORDED demonstration on the trace's
    own clock so the two can be plotted and differenced per sample.
  * :func:`plot_trace_vs_reference` — the QC sheet (xyz + rpy panels, solid = recorded,
    dashed = executed), written to a PNG only when the operator asks for one.

WHY THE COMPARISON IS ON THE *COMMANDED* PATH, NOT THE RECORDING'S WORLD POSE. The
replayed chunk is anchor-RELATIVE (see :mod:`fuji.s1_policy.serving.hl_replay`), so the
demonstration is transported into the robot's world by
``R_transport = R_wrist_robot @ R_wrist_handheld^T`` — a ~55° rigid reorientation on this
rig, which is physical (how the operator held the tool vs how the G1 holds it) and NOT a
frame bug. Plotting raw lighthouse-world recorded xyz against measured pelvis-frame xyz
would show that 55° as a huge "tracking error" and hide the real one. So the reference this
module compares against is the TRANSPORTED path the planner actually commanded — the same
curve the green line draws. Position error then means servo/IK error, which is the
question the plot exists to answer.

Time base: the trace stamps a MONOTONIC clock relative to its own start, and the reference
carries the demonstration's own relative camera clock. Both start at the first executed
sample, so t=0 is "the arm started moving" in both series.

Robot-free and viser-free: everything here takes plain arrays, so the resampling and the
error statistics are unit-testable headless. matplotlib is imported lazily INSIDE the plot
function — a serving process that never presses "save" must not pay for the import.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fuji.utils.rot6d import rot6d_to_rotmat

# How many measured samples to keep. The render loop runs at 30 Hz, so this is ~11 minutes
# — longer than any single episode (the longest clean94 take is 577 frames = 19 s) with room
# for a whole session of back-to-back plays. Bounded on purpose: the trace lives for the
# lifetime of the process and an unbounded list would grow without limit on a rig left
# running overnight. When it fills, the OLDEST samples are dropped (a ring), because the
# interesting part of a long run is what just happened.
TRACE_CAPACITY = 20_000


@dataclass(frozen=True)
class TraceSamples:
    """An immutable snapshot of a trace: ``t`` seconds since the trace started, ``pos``
    (N,3) measured LEFT TCP in the render world, ``rot6d`` (N,6) measured wrist rotation,
    ``grip_m`` (N,) measured jaw width in metres (NaN where the bridge was silent)."""

    t: np.ndarray
    pos: np.ndarray
    rot6d: np.ndarray
    grip_m: np.ndarray

    def __len__(self) -> int:
        return int(self.t.shape[0])


class ExecutionTrace:
    """Bounded, thread-safe log of where the arm ACTUALLY went.

    Appended from the render thread (`add`), read from GUI callback threads (`snapshot`).
    The lock is held only around list mutation / copy — no FK, no I/O — so it can never
    stall the 30 Hz render loop.

    ``arm``/``disarm`` gate recording so the trace covers the EXECUTION and not the minutes
    of idle hold before it: a trace full of a stationary arm would dominate every axis range
    in the plot and bury the motion. The planner arms it when a drip starts and disarms when
    the episode finishes.
    """

    def __init__(self, capacity: int = TRACE_CAPACITY):
        self._cap = int(capacity)
        self._lock = threading.Lock()
        self._t: list[float] = []
        self._pos: list[np.ndarray] = []
        self._rot: list[np.ndarray] = []
        self._grip: list[float] = []
        self._recording = False
        self._t0: float | None = None
        self._dropped = 0
        # Bookkeeping for the label on the saved sheet — which demonstration this trace is of.
        self.episode: int | None = None
        self.label: str = ""

    # -- lifecycle --------------------------------------------------------
    def arm(self, now: float, *, episode: int | None = None, label: str = "") -> None:
        """Start (or RESTART) recording, clearing anything already logged.

        Clearing is deliberate: each play is compared against its own reference, and
        concatenating two plays would draw a discontinuity the arm never travelled. ``now``
        is passed in (a monotonic reading) rather than sampled here so tests are
        deterministic and the caller's clock is the single source of truth."""
        with self._lock:
            for buf in (self._t, self._pos, self._rot, self._grip):
                buf.clear()
            self._dropped = 0
            self._t0 = float(now)
            self._recording = True
            self.episode = None if episode is None else int(episode)
            self.label = str(label)

    def disarm(self) -> None:
        """Stop recording but KEEP the samples, so the operator can still save the sheet
        after the episode ended."""
        with self._lock:
            self._recording = False

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._recording

    @property
    def dropped(self) -> int:
        """How many samples the ring discarded (0 unless a run exceeded the capacity)."""
        with self._lock:
            return self._dropped

    # -- write ------------------------------------------------------------
    def add(self, now: float, pos: np.ndarray, rot6d: np.ndarray, grip_m: float = float("nan")) -> None:
        """Append one measured sample. A no-op when not armed, and silently skips a
        non-finite pose (a bad FK/parse must not poison the plot with a NaN spike)."""
        p = np.asarray(pos, dtype=np.float64).reshape(-1)
        r = np.asarray(rot6d, dtype=np.float64).reshape(-1)
        if p.shape[0] != 3 or r.shape[0] != 6:
            return
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(r))):
            return
        with self._lock:
            if not self._recording or self._t0 is None:
                return
            self._t.append(float(now) - self._t0)
            self._pos.append(p.copy())
            self._rot.append(r.copy())
            self._grip.append(float(grip_m))
            if len(self._t) > self._cap:
                for buf in (self._t, self._pos, self._rot, self._grip):
                    del buf[0]
                self._dropped += 1

    # -- read -------------------------------------------------------------
    def __len__(self) -> int:
        """Sample count, WITHOUT copying the arrays.

        The render loop polls this every tick to decide whether the red line needs a rebuild;
        going through :meth:`snapshot` for that would stack the whole log (up to
        ``TRACE_CAPACITY`` x 10 floats) several times a second to learn one integer."""
        with self._lock:
            return len(self._t)

    def positions(self) -> np.ndarray:
        """``(N,3)`` measured positions only — the red scene line needs nothing else."""
        with self._lock:
            if not self._pos:
                return np.zeros((0, 3))
            return np.stack(self._pos).astype(np.float64)

    def snapshot(self) -> TraceSamples:
        with self._lock:
            n = len(self._t)
            if n == 0:
                return TraceSamples(np.zeros(0), np.zeros((0, 3)), np.zeros((0, 6)), np.zeros(0))
            return TraceSamples(
                np.asarray(self._t, dtype=np.float64),
                np.stack(self._pos).astype(np.float64),
                np.stack(self._rot).astype(np.float64),
                np.asarray(self._grip, dtype=np.float64),
            )


@dataclass(frozen=True)
class ReferencePath:
    """The RECORDED demonstration as the planner COMMANDED it: absolute render-world
    positions + rotations on the demonstration's own relative clock.

    Built by the planner from ``episode_path_relative`` + the live anchor — i.e. the exact
    curve the green line draws — so a position difference against a trace is servo error and
    not a frame difference. ``grip_m`` is the recorded jaw width (metres) or None."""

    t: np.ndarray        # (N,) seconds, the episode's relative camera clock
    pos: np.ndarray      # (N,3) render-world absolute
    rot6d: np.ndarray    # (N,6)
    grip_m: np.ndarray | None = None
    t_offset: float = 0.0  # the reference time the trace's t=0 corresponds to


def rot6d_to_rpy_deg(rot6d: np.ndarray) -> np.ndarray:
    """(N,6) rot6d -> (N,3) intrinsic-xyz Euler angles in DEGREES, UNWRAPPED.

    xyz Euler to match ``plot_take_qc.py``'s orientation panel, so the QC sheet an operator
    already reads and this one describe rotation the same way. Unwrapped along the sample
    axis because a wiping stroke routinely crosses the +-180 deg seam, and a raw Euler series
    then draws a full-scale vertical jump where the tool barely moved (the same trap noted in
    the zarr-replay jitter work: never diff a representation at its wrap point)."""
    r = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    if r.shape[0] == 0:
        return np.zeros((0, 3))
    from scipy.spatial.transform import Rotation

    rpy = Rotation.from_matrix(rot6d_to_rotmat(r)).as_euler("xyz", degrees=True)
    return np.unwrap(np.asarray(rpy, dtype=np.float64).reshape(-1, 3), period=360.0, axis=0)


def rotation_geodesic_deg(a_rot6d: np.ndarray, b_rot6d: np.ndarray) -> np.ndarray:
    """Per-sample rotation error in degrees: the geodesic angle of ``Ra^T Rb``.

    The honest scalar for orientation error — unlike a per-axis Euler difference it is
    representation-free (no gimbal/wrap artefacts) and is what the offline viewer's Metrics
    panel reports, so the two tools agree on "how far off was the rotation"."""
    Ra = rot6d_to_rotmat(np.asarray(a_rot6d, dtype=np.float64).reshape(-1, 6))
    Rb = rot6d_to_rotmat(np.asarray(b_rot6d, dtype=np.float64).reshape(-1, 6))
    if Ra.shape[0] == 0 or Rb.shape[0] == 0:
        return np.zeros(0)
    tr = np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def resample_reference_to_trace(ref: ReferencePath, t_trace: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The reference path sampled at the trace's timestamps -> ``(pos[N,3], rot6d[N,6])``.

    Positions are linearly interpolated; ROTATIONS ARE NEAREST-NEIGHBOUR, deliberately. A
    component-wise lerp of rot6d (or of Euler angles) is not a rotation and would bend the
    orientation curve between samples — at 30 Hz the reference grid is already the same
    33 ms as the action grid (the v3 contract's obs/action coincidence), so nearest is
    within half a frame and cannot invent a rotation the demonstration never held.

    Times outside the reference are CLAMPED to its ends rather than extrapolated: the trace
    can outlast the demonstration (the arm keeps holding after the last chunk), and an
    extrapolated reference would draw motion that was never commanded."""
    t = np.asarray(t_trace, dtype=np.float64).reshape(-1)
    rt = np.asarray(ref.t, dtype=np.float64).reshape(-1) - float(ref.t_offset)
    pos = np.asarray(ref.pos, dtype=np.float64).reshape(-1, 3)
    rot = np.asarray(ref.rot6d, dtype=np.float64).reshape(-1, 6)
    if t.size == 0 or rt.size == 0:
        return np.zeros((t.size, 3)), np.zeros((t.size, 6))
    if rt.size == 1:
        return np.repeat(pos[:1], t.size, axis=0), np.repeat(rot[:1], t.size, axis=0)
    tc = np.clip(t, rt[0], rt[-1])
    out_pos = np.stack([np.interp(tc, rt, pos[:, k]) for k in range(3)], axis=1)
    j = np.clip(np.searchsorted(rt, tc, side="left"), 0, rt.size - 1)
    j_lo = np.clip(j - 1, 0, rt.size - 1)
    pick_lo = np.abs(rt[j_lo] - tc) <= np.abs(rt[j] - tc)
    idx = np.where(pick_lo, j_lo, j)
    return out_pos, rot[idx]


@dataclass(frozen=True)
class TraceStats:
    """Summary of executed-vs-commanded agreement, printed in the sheet's title block.

    Position errors are in CENTIMETRES — the same unit as the path lengths beside them and as
    ``plot_take_qc.py``'s dev-from-median panels, so the whole sheet reads in one length unit.
    The field NAMES carry the unit (``_cm``) because a name saying ``_mm`` while holding cm is
    the exact silent-unit trap that has already cost a review round on this data (the gripper
    column read as a fraction when it was metres). Two decimals, since a good replay is well
    under 1 cm and ``0.4`` alone would throw away the interesting digit."""

    n: int
    duration_s: float
    pos_rms_cm: float
    pos_max_cm: float
    rot_mean_deg: float
    rot_max_deg: float
    path_exec_m: float
    path_ref_m: float

    def one_line(self) -> str:
        return (f"n={self.n}  {self.duration_s:.1f}s   pos RMS {self.pos_rms_cm:.2f} cm "
                f"(max {self.pos_max_cm:.2f})   rot mean {self.rot_mean_deg:.1f}deg "
                f"(max {self.rot_max_deg:.1f})   path {self.path_exec_m * 100:.1f} vs "
                f"{self.path_ref_m * 100:.1f} cm")


def trace_stats(tr: TraceSamples, ref_pos: np.ndarray, ref_rot6d: np.ndarray) -> TraceStats:
    """Executed-vs-commanded error statistics on the trace's own clock.

    ``path_exec`` / ``path_ref`` are summed polyline lengths — the check that catches a
    replay that tracked the SHAPE but only travelled part of it (the failure mode the
    ``advance_frame`` off-by-one produced: 70% of the demonstration's path length with a
    per-sample error that looked fine)."""
    p = np.asarray(tr.pos, dtype=np.float64).reshape(-1, 3)
    rp = np.asarray(ref_pos, dtype=np.float64).reshape(-1, 3)
    n = int(min(p.shape[0], rp.shape[0]))
    if n == 0:
        return TraceStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    d = np.linalg.norm(p[:n] - rp[:n], axis=1)
    rot = rotation_geodesic_deg(tr.rot6d[:n], np.asarray(ref_rot6d).reshape(-1, 6)[:n])
    return TraceStats(
        n=n,
        duration_s=float(tr.t[n - 1] - tr.t[0]) if n >= 2 else 0.0,
        pos_rms_cm=float(np.sqrt(np.mean(d ** 2)) * 1e2),
        pos_max_cm=float(np.max(d) * 1e2),
        rot_mean_deg=float(np.mean(rot)) if rot.size else 0.0,
        rot_max_deg=float(np.max(rot)) if rot.size else 0.0,
        path_exec_m=float(np.sum(np.linalg.norm(np.diff(p[:n], axis=0), axis=1))) if n >= 2 else 0.0,
        path_ref_m=float(np.sum(np.linalg.norm(np.diff(rp[:n], axis=0), axis=1))) if n >= 2 else 0.0,
    )


def plot_trace_vs_reference(
    tr: TraceSamples,
    ref: ReferencePath,
    out_path: str | Path,
    *,
    title: str = "",
    dpi: int = 130,
) -> tuple[str, TraceStats]:
    """Write the comparison sheet to ``out_path`` -> ``(absolute path, stats)``.

    Layout (one column of stacked panels, sharing the time axis, in the
    ``plot_take_qc.py`` house style):

        row 0   x, y, z  (mm)          SOLID = recorded/commanded, DASHED = executed
        row 1   roll, pitch, yaw (deg) same convention, per-axis colours
        row 2   |position error| (CM) + rotation geodesic (deg, twin axis)
        row 3   gripper width (mm)     SOLID recorded, DASHED measured

    Units: absolute POSES in mm (the frame contract they are checked against is written in mm)
    and the jaw in mm (its 43-94 mm is the physical rig number), but every DISCREPANCY — the
    error panel and the title's RMS/max — in cm, alongside the path lengths.

    Solid-vs-dashed (not colour) separates data from execution, so the three
    x/y/z colours stay free to mean the AXIS in both series — the same reading as the QC
    sheets. Raises ``ValueError`` on an empty trace instead of writing a blank PNG: an
    operator who pressed save with nothing recorded needs to be told, not handed a file."""
    import matplotlib

    matplotlib.use("Agg")  # headless: this runs inside a serving process with no display
    import matplotlib.pyplot as plt

    if len(tr) < 2:
        raise ValueError("nothing to plot: the execution trace has fewer than 2 samples "
                         "(press ⏩ Play or ▶ Run first — the trace records while the arm moves)")
    ref_pos, ref_rot = resample_reference_to_trace(ref, tr.t)
    st = trace_stats(tr, ref_pos, ref_rot)
    t = tr.t
    exec_rpy = rot6d_to_rpy_deg(tr.rot6d)
    ref_rpy = rot6d_to_rpy_deg(ref_rot)
    # Both series are unwrapped independently, so they can settle on branches a whole turn
    # apart (e.g. +178 vs -182) and read as a 360 deg error that does not exist. Re-base the
    # executed branch onto the reference's using the median offset, rounded to whole turns:
    # a pure branch difference is removed exactly, while a real error is untouched.
    if exec_rpy.shape[0] and ref_rpy.shape[0]:
        exec_rpy = exec_rpy - 360.0 * np.round(np.median(exec_rpy - ref_rpy, axis=0) / 360.0)[None, :]
    axis_colors = ("#dc3232", "#28b446", "#3c6ef0")  # replay_common.AXIS_COLORS (x,y,z)
    # Legends live OUTSIDE the axes. Inside, they sit exactly where the interesting data is: a
    # fully-open jaw is a flat line at the TOP of its panel and disappeared under an
    # "upper right" box, as did the x-position peak. Nothing on a QC sheet may be hidden by its
    # own key, and `bbox_inches="tight"` just widens the PNG to fit them.
    _leg = dict(fontsize=7, framealpha=0.9, loc="upper left", borderaxespad=0.0)

    fig, axes = plt.subplots(4, 1, figsize=(11.0, 11.0), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 2, 1.6], "hspace": 0.12})

    ax = axes[0]
    for k, (name, col) in enumerate(zip("xyz", axis_colors, strict=True)):
        ax.plot(t, ref_pos[:, k] * 1e3, color=col, lw=1.6, label=f"{name} data")
        ax.plot(t, tr.pos[:, k] * 1e3, color=col, lw=1.3, ls="--", label=f"{name} exec")
    ax.set_ylabel("position (mm)")
    ax.legend(ncol=1, bbox_to_anchor=(1.02, 1.0), **_leg)
    ax.grid(alpha=0.25)
    ax.set_title(title or "executed vs recorded", fontsize=11, loc="left")

    ax = axes[1]
    for k, (name, col) in enumerate(zip(("roll", "pitch", "yaw"), axis_colors, strict=True)):
        ax.plot(t, ref_rpy[:, k], color=col, lw=1.6, label=f"{name} data")
        ax.plot(t, exec_rpy[:, k], color=col, lw=1.3, ls="--", label=f"{name} exec")
    ax.set_ylabel("orientation xyz (deg)")
    ax.legend(ncol=1, bbox_to_anchor=(1.02, 1.0), **_leg)
    ax.grid(alpha=0.25)

    ax = axes[2]
    # ERRORS ARE IN CENTIMETRES (operator's call), matching the path lengths in the title line
    # and plot_take_qc.py's dev-from-median panels. The ABSOLUTE rows above stay in mm on
    # purpose — they are poses, and the frame contract those are checked against is written in
    # mm (wrist->TCP 0.242 m, puck->TCP offsets) — as does the jaw row, whose 43-94 mm is the
    # physical number read off the rig. So: poses in mm, discrepancies in cm.
    err_cm = np.linalg.norm(tr.pos - ref_pos, axis=1) * 1e2
    rot_err = rotation_geodesic_deg(tr.rot6d, ref_rot)
    ax.plot(t, err_cm, color="#333333", lw=1.3, label="|pos err|")
    ax.set_ylabel("pos err (cm)")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(t, rot_err, color="#b05ccc", lw=1.1, label="rot err")
    ax2.set_ylabel("rot err (deg)", color="#b05ccc")
    ax2.tick_params(axis="y", labelcolor="#b05ccc")
    # BOTH error axes are pinned to 0. These are MAGNITUDES, and matplotlib's autoscale
    # otherwise fits the band the data happens to occupy: a steady 0.4 cm lag renders as a
    # full-height line against a "1e-13 + 0.4" offset axis, which reads as violent oscillation
    # instead of "flat and small". Against zero, a good replay looks good. `top` is left
    # autoscaled so a real excursion still fills the panel.
    for a_, series in ((ax, err_cm), (ax2, rot_err)):
        top = float(np.max(series)) if series.size else 1.0
        a_.set_ylim(0.0, max(top * 1.15, 1e-9))
    ax2.legend(handles=[*ax.get_lines(), *ax2.get_lines()], ncol=1,
               bbox_to_anchor=(1.10, 1.0), **_leg)

    ax = axes[3]
    drew = False
    if ref.grip_m is not None:
        g_ref = np.asarray(ref.grip_m, dtype=np.float64).reshape(-1)
        if g_ref.size == ref.t.size and g_ref.size >= 2:
            rt = np.asarray(ref.t, dtype=np.float64) - float(ref.t_offset)
            ax.plot(t, np.interp(np.clip(t, rt[0], rt[-1]), rt, g_ref) * 1e3,
                    color="#e08b1f", lw=1.6, label="gripper data")
            drew = True
    if np.any(np.isfinite(tr.grip_m)):
        ax.plot(t, tr.grip_m * 1e3, color="#e08b1f", lw=1.3, ls="--", label="gripper exec")
        drew = True
    ax.set_ylabel("jaw (mm)")
    ax.set_xlabel("time since execution started (s)")
    ax.grid(alpha=0.25)
    if drew:
        ax.legend(ncol=1, bbox_to_anchor=(1.02, 1.0), **_leg)
    else:
        ax.text(0.5, 0.5, "no gripper data recorded", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#888888")

    fig.suptitle(st.one_line(), fontsize=9, y=0.995, x=0.01, ha="left", color="#444444")
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out), st


__all__ = [
    "TRACE_CAPACITY",
    "ExecutionTrace",
    "ReferencePath",
    "TraceSamples",
    "TraceStats",
    "plot_trace_vs_reference",
    "resample_reference_to_trace",
    "rot6d_to_rpy_deg",
    "rotation_geodesic_deg",
    "trace_stats",
]
