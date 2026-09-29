"""Record a RUN of the page — from the first Predict / ▶▶ after the ⏺ button was armed until the
operator ENDS it (STOP, Reset, the joystick ``start`` disarming the low level, the button itself,
Ctrl-C) — what the POLICY asked for, what the WIRE published, what the ROBOT did, what the F/T
SENSORS read and what the CAMERAS saw, into one time-named folder per recording, so a run on the
real robot leaves its numbers behind and ``scripts/plot_rollout_record.py`` can draw them.

WHAT IS RECORDED, AND FROM WHERE. Six JSONL streams + ``meta.json``, every line stamped with the
PAGE's wall clock at the moment the page saw the datum (one clock for all six, so they align)::

    chunks.jsonl     every sample the planner drew: the clutched wire targets of ALL its rows
                     (sequential) or its decoded anchor-frame positions (stream — the wire frame is
                     built per step there), the predicted wrench + jaw per hand per row, the seed,
                     the sample time, the base it continued from.        <- _predict / the sampler
    steps.jsonl      every EXECUTED step: which chunk + row (stream: the absolute step clock),
                     the clutched target BEFORE reach compensation and the leg override, and the
                     wrench rows the force gate saw for it.              <- _drip / the stream loop
    published.jsonl  every frame that reached :8770 — the drip's, the stream's, the heartbeat's
                     re-sends — AS PUBLISHED (reach residual + leg override applied), with F^cmd,
                     whether reference legs rode along and whether the override held the feet.
                                                                          <- _send
    measured.jsonl   every RecordTap frame (50 Hz) verbatim, plus the robot's five keypoints FK'd
                     from its 29 joints on the recorder's own thread (the SAME ``KeypointFK`` +
                     ``KeypointReadout`` the clutch's leak bound uses, a PRIVATE instance because
                     yourdfpy's scene is mutated per call and two threads must not share it).
                                                                          <- :8779
    force.jsonl      every F/T bridge frame (~250 Hz) verbatim: ``*_pipe`` is the 6-DoF tared
                     SENSOR-frame wrench the model consumes; ``left``/``right`` the low level's
                     3-DoF stream.                                        <- :8771
    events.jsonl     recording begin / end (with what ENDED it), every ▶▶ run's begin / end INSIDE
                     it (with the page's own end note), armed / disarmed edges, the leg override's
                     on/off edges (derived from the published frames' flag), writer errors.
    camera/          every distinct frame of every camera the checkpoint READS — one or two views,
                     read off the page's own subscriptions — as the bridge's OWN JPEG (the full
                     640x480 capture, never the 224 crop, nothing re-encoded): one folder per view,
                     ``camera/<image_key>/000123.jpg``, plus the index ``camera/frames.jsonl`` (one
                     line per frame: the page clock ``t``, the bridge's ``seq`` / ``t_bridge``, the
                     file). A view that sends nothing keeps its empty folder and a zero count in
                     ``meta.camera``, so a dead bridge is visible rather than silent.
                                                                          <- :8773 / :8777 (on_msg)

WHY A TAP AT THE EXISTING SEAMS, AND NOT A SECOND SUBSCRIBER OR A SECOND PUBLISHER. The page has ONE
publish funnel (``_send``), one predict, two execute loops and two bridge subscribers that already
see every frame (``_WSSubscriber(on_msg=)``). A tap at each is the only way the record can say what
the wire ACTUALLY carried — reach residual, leg override, gate refusals included — rather than what
some second reader of the same data believes it did.

WHY IT CANNOT SLOW THE ROLLOUT — the property the page depends on. Every tap is one attribute read
(``recording``) and, while recording, a few small copies plus one ``SimpleQueue.put`` (never blocks,
reentrant); JSON encoding, the FK and the disk all happen on the recorder's OWN thread. A tap never
raises to its caller: the publish funnel, the drip and the socket threads are exactly the places an
exception must not come from. Nothing here reads the wire, the clutch or the planner back — a
recorder that touched ``wire.commit`` or ``last_wire`` would be a second controller.

THE BUTTON is a four-state machine, and its colour IS the state::

    idle (blue)  --click-->  armed (green)  --the next Predict / ▶▶-->  recording (green, live counts)
       ^                        ^                                            |
       |   Reset                |  nothing predicted (a refused start)       |  STOP / Reset / the low level
       +---------- done (blue, the folder) <---------------------------------+  DISARMED (joystick start) /
                     |   click                                                  the button itself / Ctrl-C
                     +----> armed

A RECORDING IS THE OPERATOR'S EPISODE, NOT ONE ▶▶ (2026-09-20). It opens at the first Predict — a manual
one or the one ▶▶ makes — after the button was armed, and it stays open across chunks: a ▶▶ run ending on
its own (all its chunks done, the seconds ceiling, ⏹ retiring it at a chunk boundary, the low level
dropping off :8770) does NOT end it, and neither does the gap between two manual Predict → Run presses.
The operator may run one episode as ▶▶, as Predict / Run one chunk at a time, or as any mix, and it is
ONE folder until THEY end it: STOP, Reset, the joystick ``start`` disarming the low level (RecordTap's
``vive_armed`` falling edge, seen on the socket thread), the button, or Ctrl-C. ▶▶ runs inside it are
brackets — ``auto_run_begin`` / ``auto_run_end`` events carrying the page's own end note, listed in
``meta.auto_runs`` — and the recording's ``end_note`` says what ended IT. Reset returns to ``idle`` with
a different label than ``done`` (the operator asked for the two to read differently), and every
recording gets a NEW folder ``<root>/<task>/<YYYY-MM-DD_HH-MM-SS>/`` — the task from ``--record-task``
(the launcher's ``TASK=``: push_gantry, push_cart, rotate_box, wipe_board …). ``every_run``
(``RECORD_EVERY=1``) needs no click: the next Predict / ▶▶ after an end opens the next folder.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

#: The six streams, in the order ``meta.json`` reports their counts.
STREAMS: tuple[str, ...] = ("chunks", "steps", "published", "measured", "force", "events")
#: The camera frames: ``<run>/camera/<image_key>/NNNNNN.jpg`` + the index ``<run>/camera/frames.jsonl``
#: (``camera`` in ``counts`` / ``meta.streams``). Deliberately NOT in STREAMS: the folder exists only
#: when the recorder was handed cameras, and its index lives inside the folder, not beside the six.
CAMERA_STREAM = "camera"
CAMERA_DIR = "camera"
CAMERA_INDEX = "frames.jsonl"
#: The bridge's own fields kept beside each frame (its ``t`` is written as ``t_bridge`` — ``t`` is
#: the page clock, as on every other line of the record).
CAMERA_MSG_KEYS: tuple[str, ...] = ("seq", "capture_seq", "t", "t_host", "w", "h", "stream")
_JPEG_SOI = b"\xff\xd8"
STATES: tuple[str, ...] = ("idle", "armed", "recording", "done")
#: The button colour per state — viser's Mantine colour names. Blue = "not recording", green =
#: "will / is recording"; that is the operator's whole reading of it.
STATE_COLOR: dict[str, str] = {"idle": "blue", "armed": "green", "recording": "green", "done": "blue"}
#: Folder-name timestamp; sorts lexically = chronologically, no characters a shell dislikes.
STAMP_FMT = "%Y-%m-%d_%H-%M-%S"
#: How long a full file flush may wait; also the writer's idle wake-up.
FLUSH_S = 0.5
#: The writer thread's bounded join at the end of a run. It only ever has the tail of one queue
#: left to encode; longer than this means the disk is gone, and the page must not hang on it.
WRITER_JOIN_S = 5.0
#: Rounding, per quantity, so a line is a few hundred bytes rather than a few thousand: positions
#: and joints to 1 µm / 1 µrad, quaternions to 1e-6, forces to 0.1 mN.
ROUND_M = 6
ROUND_Q = 6
ROUND_N = 4

_SENTINEL = object()


def run_stamp(t: float | None = None) -> str:
    """``2026-09-12_17-03-21`` for wall time ``t`` (now if None), in LOCAL time — the operator's."""
    return time.strftime(STAMP_FMT, time.localtime(time.time() if t is None else float(t)))


def task_from_checkpoint(checkpoint: str | os.PathLike | None) -> str:
    """A folder label read off a checkpoint path when ``--record-task`` was not given.

    ``checkpoints/umi_5pt_push_gantry/h50`` -> ``push_gantry`` (the registry's per-task dir shape the
    launcher's ``TASK=`` resolves to); ``checkpoints/wipe_board/<run>/step_00100000`` -> ``<run>``
    (a ``step_*`` dir names its run); anything else -> its last path component; nothing -> ``run``.
    The launcher forwards ``TASK=`` as ``--record-task``, so this is only the fallback.
    """
    if checkpoint is None or not str(checkpoint).strip():
        return "run"
    parts = [p for p in Path(str(checkpoint)).expanduser().parts if p not in ("/", "")]
    while parts and (parts[-1].startswith("step_") or parts[-1] in (".", "..")):
        parts.pop()
    if not parts:
        return "run"
    leaf = parts[-1]
    if len(parts) >= 2 and len(leaf) <= 5 and leaf[:1] == "h" and leaf[1:].isdigit():
        leaf = parts[-2]                       # .../umi_5pt_<task>/h50
    if leaf.startswith("umi_5pt_"):
        leaf = leaf[len("umi_5pt_"):]
    if leaf.startswith("umi_") and len(leaf) > 4:
        leaf = leaf[4:]
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in leaf).strip("._")
    return safe or "run"


def _round(arr, nd: int):
    """A JSON-able rounded list from an array-like, or None."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float64)
    return np.round(a, nd).tolist()


def _safe_name(raw, fallback: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(raw)).strip("._")
    return safe or fallback


def camera_specs(cameras) -> list[dict]:
    """Normalise what the page knows about its cameras into ``[{"cam", "key", "url"}, ...]``.

    Accepts ``(index, key)`` / ``(index, key, url)`` tuples, dicts with those fields, or bare key
    strings (indexed in order) — whatever the caller has, for ONE or TWO views (or more). Keys become
    folder names (sanitised, made unique), so ``camera0_rgb`` / ``camera1_rgb`` are the checkpoint's
    own image keys and a run folder reads like the training data does. None / empty -> no cameras.
    """
    specs: list[dict] = []
    seen: set[str] = set()
    for i, item in enumerate(cameras or ()):
        if isinstance(item, dict):
            cam, key, url = item.get("cam", i), item.get("key"), item.get("url")
        elif isinstance(item, (tuple, list)):
            cam = item[0] if len(item) > 0 else i
            key = item[1] if len(item) > 1 else None
            url = item[2] if len(item) > 2 else None
        else:
            cam, key, url = i, item, None
        try:
            cam = int(cam)
        except (TypeError, ValueError):
            cam = i
        name = _safe_name(key if key not in (None, "") else f"camera{cam}", f"camera{cam}")
        base, n = name, 2
        while name in seen:                    # two views with one key: never share a folder
            name = f"{base}_{n}"
            n += 1
        seen.add(name)
        specs.append({"cam": cam, "key": name, "url": None if url is None else str(url)})
    return specs


def _cam_state(spec: dict) -> dict:
    """One view's live counters for a run."""
    return {**spec, "n": 0, "bytes": 0, "seen": 0, "dup": 0, "skipped": 0, "bad": 0,
            "last_seq": None, "t_first": None, "t_last": None}


class _Run:
    """One recording's mutable state: its folder, files, queue, writer thread and counters."""

    def __init__(self, run_dir: Path, t0: float, info: dict) -> None:
        self.dir = run_dir
        self.t0 = float(t0)
        self.info = dict(info)
        self.q: queue.SimpleQueue = queue.SimpleQueue()
        self.files: dict[str, object] = {}
        self.counts: dict[str, int] = {s: 0 for s in (*STREAMS, CAMERA_STREAM)}
        #: per camera index: the view's spec + live counters (see ``_cam_state``)
        self.cams: dict[int, dict] = {}
        self.errors = 0
        self.legs_last: bool | None = None
        self.armed_last: bool | None = None
        #: the last ``vive_armed`` the SOCKET thread saw (the writer keeps its own ``armed_last`` for
        #: the events) — a True → False edge while recording is the low level being DISARMED.
        self.armed_seen: bool | None = None
        #: the ▶▶ runs inside this recording: ``{"cadence", "t_begin", "t_end", "note"}`` each
        self.segments: list[dict] = []
        self.seg_open: dict | None = None
        self.thread: threading.Thread | None = None
        self.t_end: float | None = None
        self.end_note = ""


class RunRecorder:
    """The ⏺ button's state machine + the six-stream writer. One per page.

    ``measure`` is ``msg -> (pos15, orn20) | None``: the RecordTap dict to the robot's measured five
    keypoints, run on the WRITER thread (build it on a private ``KeypointFK``). ``info_provider`` is
    ``() -> dict`` of the run's live settings (cadence, exec steps, seed …), read once at
    :meth:`run_begin`. ``plot_cmd`` is ``run_dir -> argv``; when set and ``autoplot`` is on, it is
    spawned DETACHED (low priority, output to ``plot.log`` in the folder) after a run is saved —
    the page never waits on it.

    ``end_on_disarm`` (default True) ends the recording when RecordTap's ``vive_armed`` goes True →
    False while recording — the joystick ``start`` — seen on the socket thread and closed on a helper
    thread (:meth:`end_async`); the page's STOP / Reset / Ctrl-C call :meth:`end_async` /
    :meth:`run_end` themselves. ``ends_on`` is the button's own wording of those gestures.

    ``cameras`` names the views to record (see :func:`camera_specs`: the page passes its camera
    subscriptions, i.e. exactly the ``image_keys`` the checkpoint reads — one or two). Each frame
    the bridge publishes while recording lands as ``camera/<key>/NNNNNN.jpg`` with a line in
    ``camera/frames.jsonl``; ``camera_every`` keeps every N-th distinct frame per view (1 = all,
    0 = record no camera frames at all). A frame from a camera index that was NOT declared is
    recorded anyway under ``camera<i>``: the recorder follows the page's subscriptions, not a list
    that could go stale.
    """

    def __init__(self, root: str | os.PathLike, task: str, *, static_meta: dict | None = None,
                 measure: Callable[[dict], tuple | None] | None = None,
                 info_provider: Callable[[], dict] | None = None,
                 log: Callable[[str], None] | None = None,
                 autoplot: bool = False, plot_cmd: Callable[[Path], list[str]] | None = None,
                 every_run: bool = False, cameras=None, camera_every: int = 1,
                 end_on_disarm: bool = True, ends_on: str = "STOP / Reset / disarm") -> None:
        self.root = Path(root).expanduser()
        raw = str(task).strip()
        self.task = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in raw).strip("._") or "run"
        self.static_meta = dict(static_meta or {})
        self.measure = measure
        self.info_provider = info_provider
        self._log = log if log is not None else (lambda s: print(s, flush=True))
        self.autoplot = bool(autoplot)
        self.plot_cmd = plot_cmd
        self.every_run = bool(every_run)
        self.end_on_disarm = bool(end_on_disarm)
        self.ends_on = str(ends_on)
        self.camera_every = max(0, int(camera_every))
        self.cameras: list[dict] = [] if self.camera_every == 0 else camera_specs(cameras)
        self._lock = threading.Lock()
        self._state = "idle"
        #: Read WITHOUT the lock by every tap — a plain bool, flipped only under the lock.
        self.recording = False
        self._run: _Run | None = None
        self.last_dir: Path | None = None
        self.last_summary = ""
        self.n_runs = 0
        self._after_reset = False
        #: a pending :meth:`end_async` — joined by every state change and by :meth:`run_end` itself
        self._ender: threading.Thread | None = None
        self._t_wall_minus_mono = time.time() - time.monotonic()

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> str:
        return self._state

    @property
    def run_dir(self) -> Path | None:
        run = self._run
        return None if run is None else run.dir

    @property
    def task_dir(self) -> Path:
        return self.root / self.task

    def counts(self) -> dict[str, int]:
        run = self._run
        return dict(run.counts) if run is not None else {}

    def frames_written(self) -> int:
        run = self._run
        return 0 if run is None else int(run.counts["published"])

    def _settle(self) -> None:
        """Join a pending :meth:`end_async`, so the caller reads a SETTLED state. Never from the
        ender itself (:meth:`run_end` calls this too)."""
        th = self._ender
        if th is not None and th is not threading.current_thread() and th.is_alive():
            th.join(timeout=WRITER_JOIN_S + 1.0)

    def arm(self) -> str:
        self._settle()
        with self._lock:
            if self._state == "recording":
                return "already recording"
            self._state = "armed"
        return f"ARMED — recording starts at the next Predict / ▶▶ and ends on {self.ends_on}"

    def disarm(self) -> str:
        self._settle()
        with self._lock:
            if self._state != "armed":
                return f"not armed ({self._state})"
            self._state = "idle"
        return "disarmed — nothing is recorded until the button is armed again"

    def toggle(self) -> str:
        """The button's body: idle/done -> armed, armed -> idle, recording -> end this recording now."""
        self._settle()
        state = self._state
        if state == "armed":
            return self.disarm()
        if state == "recording":
            return self.run_end("ended by the ⏺ button (the robot goes on)")
        return self.arm()

    def reset(self) -> str:
        """Reset -> ``idle``: a live recording is ENDED first (Reset is one of the operator's end
        gestures), then the button reads idle with the reset label."""
        self._settle()
        note = ""
        if self._state == "recording":
            note = self.run_end("reset") + " · "
        with self._lock:
            self._state = "idle"
            self._after_reset = True
        return note + "recorder idle — the next Predict / ▶▶ opens a new folder once armed"

    # -------------------------------------------------------------------- UI
    def ui(self) -> tuple[str, str]:
        """``(label, colour)`` for the button, from the state alone."""
        state = self._state
        color = STATE_COLOR[state]
        every = " · EVERY run records" if self.every_run else ""
        if state == "armed":
            return f"⏺ ARMED — recording starts at the next Predict / ▶▶  (click to disarm){every}", color
        if state == "recording":
            run = self._run
            if run is None:                    # an end is on its way: a helper thread is closing the folder
                return "⏺ saving the recording …", color
            c = run.counts
            return (f"⏺ RECORDING {run.dir.name} · {c['chunks']} chunks · {c['published']} frames — "
                    f"ends on {self.ends_on}  (click to end now)"), color
        if state == "done":
            return f"✓ SAVED {self.last_summary}  (click to arm again){every}", color
        if self._after_reset and self.last_dir is not None:
            return f"⏺ Record the next run?  (reset · last saved {self.task}/{self.last_dir.name}){every}", color
        return f"⏺ Record the next run?  (click = arm){every}", color

    def describe(self) -> str:
        run = self._run
        if run is not None and self._state == "recording":
            c = run.counts
            return (f"recording {run.dir} · {time.time() - run.t0:.1f} s · chunks {c['chunks']} · "
                    f"steps {c['steps']} · ▶▶ runs {len(run.segments)} · published {c['published']} · "
                    f"measured {c['measured']} · force {c['force']}"
                    + (f" · camera {c['camera']}" if run.cams else "")
                    + (f" · ⚠️ {run.errors} writer error(s)" if run.errors else ""))
        if self._state == "done" and self.last_dir is not None:
            return f"saved {self.last_dir} — plot: {self.plot_hint(self.last_dir)}"
        return f"{self._state} · folder root {self.task_dir}"

    @staticmethod
    def plot_hint(run_dir: Path) -> str:
        return f"uv run scripts/plot_rollout_record.py {run_dir}"

    # ------------------------------------------------------------- run edges
    def run_begin(self, cadence: str = "", info: dict | None = None) -> bool:
        """A RECORDING starts here — the first Predict (manual, or the one a ▶▶ makes) after the
        button was armed, or on ``every_run``: open a folder. True when a folder was opened; False
        when one is already open (the seams call this on EVERY Predict — a no-op mid-recording), when
        not armed, or when the folder could not be opened."""
        if self.recording:                      # the fast path — one attribute read per Predict
            return False
        self._settle()                          # an end on its way (STOP / disarm): wait for its folder
        with self._lock:
            if self._state == "recording":
                return False
            if self._state != "armed" and not self.every_run:
                return False
            t0 = time.time()
            run_dir = self.task_dir / run_stamp(t0)
            n = 1
            while run_dir.exists():            # two runs inside one second: never overwrite
                n += 1
                run_dir = self.task_dir / f"{run_stamp(t0)}_{n}"
            files: dict[str, object] = {}
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                for s in STREAMS:
                    files[s] = open(run_dir / f"{s}.jsonl", "w", buffering=1 << 16)  # noqa: SIM115
                if self.cameras:
                    # One folder per view up front — an empty one afterwards SAYS the bridge sent nothing.
                    for spec in self.cameras:
                        (run_dir / CAMERA_DIR / spec["key"]).mkdir(parents=True, exist_ok=True)
                    files[CAMERA_STREAM] = open(run_dir / CAMERA_DIR / CAMERA_INDEX, "w", buffering=1 << 16)  # noqa: SIM115
            except OSError as exc:
                self._log(f"RECORD: cannot open {run_dir} ({exc!r}) — NOT recording this run")
                for f in files.values():
                    try:
                        f.close()
                    except OSError:
                        pass
                return False
            merged = {"cadence": str(cadence)}
            if self.info_provider is not None:
                try:
                    merged.update(dict(self.info_provider()))
                except Exception as exc:  # noqa: BLE001 - a bad readout must not stop the record
                    merged["info_error"] = repr(exc)
            if info:
                merged.update(dict(info))
            run = _Run(run_dir, t0, merged)
            run.files = files
            run.cams = {spec["cam"]: _cam_state(spec) for spec in self.cameras}
            run.thread = threading.Thread(target=self._writer, args=(run,), daemon=True,
                                          name="rollout-rec")
            self._run = run
            self._state = "recording"
            self._after_reset = False
            self.recording = True
            run.thread.start()
        self.event("run_begin", **{k: v for k, v in merged.items() if isinstance(v, (int, float, str, bool))})
        self._log(f"RECORD: ⏺ {run_dir} — recording from this {cadence or 'Predict'} on, until "
                  f"{self.ends_on} (a ▶▶ ending on its own does not end it)")
        return True

    def run_end(self, note: str = "") -> str:
        """The recording ENDS (STOP / Reset / the low level disarmed / the button / Ctrl-C): close the
        folder. Safe from any thread and idempotent; a pending :meth:`end_async` is joined first.

        A recording in which NOTHING WAS PREDICTED — no chunk, no step; the heartbeat's re-sends
        alone do not count — is discarded and the arm kept, so a refused start never costs the
        operator a click.
        """
        self._settle()
        with self._lock:
            run = self._detach_locked(note)
        if run is None:
            return "no recording to end"
        return self._finish(run)

    def _detach_locked(self, note: str) -> _Run | None:
        """Under the lock: take the live run off the taps (``recording`` is False the instant this
        returns, so the very next Predict can open a new folder) and stamp its end. None = nothing
        was recording."""
        run = self._run
        if run is None or self._state != "recording":
            return None
        self.recording = False
        self._run = None
        run.t_end = time.time()
        run.end_note = str(note)
        seg = run.seg_open
        if seg is not None:                    # a ▶▶ still running: its bracket closes with the recording
            seg["t_end"] = run.t_end
            seg["note"] = seg["note"] or f"(the recording ended: {run.end_note})"
            run.seg_open = None
        return run

    def _finish(self, run: _Run) -> str:
        """Drain the writer, close the files, write ``meta.json`` (or discard an empty folder), move
        the button on. The slow half of an end — on the caller's thread for :meth:`run_end`, on a
        helper thread for :meth:`end_async`."""
        note = run.end_note
        run.q.put(("events", run.t_end, "run_end", {"note": run.end_note,
                                                    "duration_s": round(run.t_end - run.t0, 3),
                                                    "n_auto_runs": len(run.segments)}))
        run.q.put(_SENTINEL)
        if run.thread is not None:
            run.thread.join(timeout=WRITER_JOIN_S)
            if run.thread.is_alive():
                run.errors += 1
                self._log(f"RECORD: ⚠️ the writer did not finish within {WRITER_JOIN_S:g} s — "
                          f"{run.dir} may be incomplete")
        for f in run.files.values():
            try:
                f.close()
            except OSError:
                pass
        c = run.counts
        if c["chunks"] == 0 and c["steps"] == 0:
            shutil.rmtree(run.dir, ignore_errors=True)
            with self._lock:
                if self._state == "recording":     # unless a Reset already moved it on
                    self._state = "armed"
            msg = (f"nothing predicted ({note or 'no Predict happened'}) — nothing recorded, folder discarded, "
                   f"still ARMED")
            self._log(f"RECORD: {msg}")
            return msg
        meta = self._meta(run)
        try:
            with open(run.dir / "meta.json", "w") as fh:
                json.dump(meta, fh, indent=1, default=str)
        except OSError as exc:
            self._log(f"RECORD: ⚠️ could not write meta.json ({exc!r})")
        dur = run.t_end - run.t0
        with self._lock:
            if self._state == "recording":         # a Reset that raced this end has already set idle
                self._state = "done"
            self.last_dir = run.dir
            self.n_runs += 1
            self.last_summary = (f"{self.task}/{run.dir.name} · {dur:.0f} s · {c['chunks']} chunks · "
                                 f"{c['published']} frames")
        summary = (f"saved {run.dir} — {dur:.1f} s · {note or 'ended'} · chunks {c['chunks']} · steps {c['steps']} · "
                   f"▶▶ runs {len(run.segments)} · published {c['published']} · measured {c['measured']} · "
                   f"force {c['force']}"
                   + (f" · camera {self._camera_brief(run)}" if run.cams else "")
                   + (f" · ⚠️ {run.errors} writer error(s)" if run.errors else ""))
        self._log(f"RECORD: ✓ {summary}")
        self._log(f"RECORD: plot with  {self.plot_hint(run.dir)}")
        if self.autoplot and self.plot_cmd is not None:
            self._spawn_plot(run.dir)
        return summary

    def end_async(self, note: str = "") -> bool:
        """End the recording WITHOUT waiting — for the socket thread (the low level disarming) and for
        STOP, neither of which may block on the disk. The run is taken off the taps HERE (so a Predict
        that follows at once opens a new folder, after joining the helper); the folder is closed on a
        helper thread that every later state change (and :meth:`run_end`) joins. True when there was
        a recording to end."""
        with self._lock:
            run = self._detach_locked(note)
            if run is None:
                return False
            th = threading.Thread(target=self._finish_worker, args=(run,), daemon=True,
                                  name="rollout-rec-end")
            self._ender = th
        th.start()
        return True

    def _finish_worker(self, run: _Run) -> None:
        try:
            self._finish(run)
        except Exception as exc:  # noqa: BLE001 - the end must never take the page down
            self._log(f"RECORD: ⚠️ closing the recording {run.dir} failed ({exc!r})")
        finally:
            with self._lock:
                if self._ender is threading.current_thread():
                    self._ender = None

    # -------------------------------------------------------- the ▶▶ brackets
    def auto_run_begin(self, cadence: str) -> None:
        """A ▶▶ run starts INSIDE the recording (``RolloutTaps.wrap_run``): an event + an entry in
        ``meta.auto_runs``. NOT a recording edge — the recording spans as many ▶▶ runs and manual
        Predict / Run presses as the operator makes before ending it."""
        run = self._run
        if run is None or not self.recording:
            return
        seg = {"cadence": str(cadence), "t_begin": time.time(), "t_end": None, "note": ""}
        run.seg_open = seg
        run.segments.append(seg)
        self.event("auto_run_begin", cadence=str(cadence), n=len(run.segments))

    def auto_run_end(self, note: str = "") -> None:
        """The ▶▶ run inside the recording ended (on its own, ⏹, the ceiling, an error, the low level
        dropping off): close its bracket with the page's own note. The recording goes on."""
        run = self._run
        if run is None or not self.recording:
            return
        seg = run.seg_open
        if seg is None:
            return
        seg["t_end"] = time.time()
        seg["note"] = str(note)
        run.seg_open = None
        self.event("auto_run_end", cadence=seg["cadence"], note=seg["note"],
                   duration_s=round(seg["t_end"] - seg["t_begin"], 3), n=len(run.segments))

    def _camera_brief(self, run: _Run) -> str:
        """``742 camera0_rgb + 741 camera1_rgb`` — the per-view frame counts, for the log line."""
        return " + ".join(f"{st['n']} {st['key']}" for _cam, st in sorted(run.cams.items())) or "0"

    def _camera_meta(self, run: _Run) -> dict | None:
        """``meta.camera``: where the frames are and, per view, how many came (+ what was dropped and
        why), so a run whose camera bridge died reads as such. None when no camera was recorded."""
        if not run.cams:
            return None
        views = []
        for cam in sorted(run.cams):
            st = run.cams[cam]
            span = (st["t_last"] - st["t_first"]) if (st["t_first"] is not None and st["n"] > 1) else 0.0
            views.append({"cam": cam, "key": st["key"], "url": st.get("url"), "dir": f"{CAMERA_DIR}/{st['key']}",
                          "frames": int(st["n"]), "bytes": int(st["bytes"]), "seen": int(st["seen"]),
                          "duplicates": int(st["dup"]), "skipped": int(st["skipped"]), "bad": int(st["bad"]),
                          "fps": round((st["n"] - 1) / span, 2) if span > 0 else None,
                          "t_first": st["t_first"], "t_last": st["t_last"]})
        return {"dir": CAMERA_DIR, "index": f"{CAMERA_DIR}/{CAMERA_INDEX}", "every": int(self.camera_every),
                "file_pattern": f"{CAMERA_DIR}/<key>/NNNNNN.jpg",
                "encoding": "the camera bridge's own JPEG as received (full capture, not the model's 224 crop)",
                "views": views}

    def _meta(self, run: _Run) -> dict:
        streams = {s: f"{s}.jsonl" for s in STREAMS}
        if run.cams:
            streams[CAMERA_STREAM] = f"{CAMERA_DIR}/{CAMERA_INDEX}"
        return {
            "task": self.task,
            "stamp": run.dir.name,
            "t_begin": run.t0,
            "t_end": run.t_end,
            "duration_s": None if run.t_end is None else round(run.t_end - run.t0, 3),
            "end_note": run.end_note,
            "auto_runs": [dict(seg) for seg in run.segments],
            "n_auto_runs": len(run.segments),
            "run": run.info,
            "counts": dict(run.counts),
            "writer_errors": run.errors,
            "streams": streams,
            "time_base": "every 't' is the PAGE's wall clock (time.time()) when it saw the datum; "
                         "bridge frames also keep their own 't' inside 'msg' (camera: 't_bridge')",
            "camera": self._camera_meta(run),
            "static": self.static_meta,
        }

    def _spawn_plot(self, run_dir: Path) -> None:
        try:
            argv = list(self.plot_cmd(run_dir))
            if shutil.which("nice"):
                argv = ["nice", "-n", "10", *argv]
            log = open(run_dir / "plot.log", "w")  # noqa: SIM115 - handed to the child
            subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True, cwd=str(run_dir))
            log.close()
            self._log(f"RECORD: plotting in the background → {run_dir / 'plot.log'}")
        except Exception as exc:  # noqa: BLE001 - a failed plot must not touch the page
            self._log(f"RECORD: could not start the plot ({exc!r}); run it by hand: "
                      f"{self.plot_hint(run_dir)}")

    # ------------------------------------------------------------------ taps
    # Every tap: ONE attribute read when idle; a few small copies + one non-blocking put when
    # recording; never raises. Encoding happens on the writer thread.
    def _put(self, item: tuple) -> None:
        run = self._run
        if run is not None:
            run.q.put(item)

    def chunk(self, *, seq: int, cadence: str, decoded_pos, pos=None, orn=None, wrench: dict | None = None,
              grip: dict | None = None, t_obs: int | None = None, seed=None, sample_ms: float | None = None,
              wire_src: str | None = None, n_exec: int | None = None, source: str | None = None,
              replay: dict | None = None) -> None:
        """One staged chunk. ``source`` names where it came from (``"model"`` when absent, ``"replay"``
        for a recorded episode) and ``replay`` carries that episode's identity (scalars only:
        episode, frame, mode, n_valid, time_s, zarr …); both are omitted by the model path and land
        as ``null`` in ``chunks.jsonl``, so every existing reader sees the same fields it did."""
        if not self.recording:
            return
        try:
            self._put(("chunks", time.time(), {
                "seq": int(seq), "cadence": str(cadence),
                "source": None if source is None else str(source),
                "replay": None if replay is None else {
                    str(k): v for k, v in dict(replay).items() if isinstance(v, (int, float, str, bool)) or v is None},
                "decoded_pos": None if decoded_pos is None else np.asarray(decoded_pos, dtype=np.float64).copy(),
                "pos": None if pos is None else np.asarray(pos, dtype=np.float64).copy(),
                "orn": None if orn is None else np.asarray(orn, dtype=np.float64).copy(),
                "wrench": {k: (None if v is None else np.asarray(v, dtype=np.float64).copy())
                           for k, v in (wrench or {}).items()},
                "grip": {k: (None if v is None else np.asarray(v, dtype=np.float64).copy())
                         for k, v in (grip or {}).items()},
                "t_obs": None if t_obs is None else int(t_obs),
                "seed": None if seed is None else int(seed),
                "sample_ms": None if sample_ms is None else float(sample_ms),
                "wire_src": None if wire_src is None else str(wire_src),
                "n_exec": None if n_exec is None else int(n_exec)}))
        except Exception:  # noqa: BLE001, S110 - a tap never raises to the predict path
            pass

    def step(self, seq: int, row: int, frame: tuple, wrench_rows: dict | None = None, *,
             n_blended: int | None = None) -> None:
        if not self.recording:
            return
        try:
            self._put(("steps", time.time(), {
                "chunk": int(seq), "row": int(row),
                "target_pos": np.asarray(frame[0], dtype=np.float64).copy(),
                "target_orn": np.asarray(frame[1], dtype=np.float64).copy(),
                "wrench": {k: (None if v is None else np.asarray(v, dtype=np.float64).copy())
                           for k, v in (wrench_rows or {}).items()},
                "n_blended": None if n_blended is None else int(n_blended)}))
        except Exception:  # noqa: BLE001, S110 - a tap never raises to the drip
            pass

    def publish(self, frame: tuple, fcmd, ok: bool, *, override: bool = False, ref_legs: bool = False) -> None:
        """One published frame. ``override`` = the leg override held the two foot slots; ``ref_legs`` =
        reference legs (``ref_joints``) rode the frame. Recorded as ``legs`` / ``ref_joints``."""
        if not self.recording:
            return
        try:
            self._put(("published", time.time(), {
                "pos": np.asarray(frame[0], dtype=np.float64).copy(),
                "orn": np.asarray(frame[1], dtype=np.float64).copy(),
                "fcmd": None if fcmd is None else np.asarray(fcmd, dtype=np.float64).copy(),
                "ok": bool(ok), "legs": bool(override), "ref_joints": bool(ref_legs)}))
        except Exception:  # noqa: BLE001, S110 - a tap never raises to the publish funnel
            pass

    def measured(self, msg, t_mono: float | None = None) -> None:
        """A RecordTap frame, on the socket thread. The FK happens on the writer thread. This is also
        where the low level DISARMING is seen: ``vive_armed`` True → False while recording (the
        joystick ``start``; joystick B stops the stream instead and is no edge) ends the recording on
        a helper thread — the socket never waits on the disk."""
        if not self.recording:
            return
        try:
            t = time.time() if t_mono is None else float(t_mono) + self._t_wall_minus_mono
            self._put(("measured", t, msg))
            if self.end_on_disarm and isinstance(msg, dict) and "vive_armed" in msg:
                armed = bool(msg.get("vive_armed"))
                run = self._run
                if run is not None:
                    prev, run.armed_seen = run.armed_seen, armed
                    if prev and not armed:
                        self.end_async("the low level DISARMED (joystick start) — the recording ends with the arm")
        except Exception:  # noqa: BLE001, S110 - a tap never raises to the socket
            pass

    def force(self, msg, t_mono: float | None = None) -> None:
        """An F/T bridge frame, on the socket thread."""
        if not self.recording:
            return
        try:
            t = time.time() if t_mono is None else float(t_mono) + self._t_wall_minus_mono
            self._put(("force", t, msg))
        except Exception:  # noqa: BLE001, S110 - a tap never raises to the socket
            pass

    def camera(self, cam: int, msg, t_mono: float | None = None) -> None:
        """A camera bridge frame, on ITS socket thread. The JPEG travels as the bridge's own base64
        string (a reference, no copy); the decode, the JPEG check and the file write happen on the
        writer thread. Nothing here is an image operation."""
        if not self.recording or not self.camera_every:
            return
        try:
            if not isinstance(msg, dict):
                return
            b64 = msg.get("jpeg_b64")
            if not b64:
                return
            t = time.time() if t_mono is None else float(t_mono) + self._t_wall_minus_mono
            self._put((CAMERA_STREAM, t, int(cam), b64, {k: msg.get(k) for k in CAMERA_MSG_KEYS if k in msg}))
        except Exception:  # noqa: BLE001, S110 - a tap never raises to the socket
            pass

    def event(self, kind: str, **fields) -> None:
        if not self.recording:
            return
        try:
            self._put(("events", time.time(), str(kind), dict(fields)))
        except Exception:  # noqa: BLE001, S110
            pass

    # ---------------------------------------------------------------- writer
    def _writer(self, run: _Run) -> None:
        last_flush = time.monotonic()
        warned = False
        while True:
            try:
                item = run.q.get(timeout=FLUSH_S)
            except queue.Empty:
                item = None
            if item is _SENTINEL:
                break
            if item is not None:
                try:
                    self._write(run, item)
                except Exception as exc:  # noqa: BLE001 - one bad line must not end the record
                    run.errors += 1
                    if not warned:
                        warned = True
                        self._log(f"RECORD: writer error {exc!r} (further ones counted, not logged)")
                    try:
                        line = json.dumps({"t": time.time(), "kind": "writer_error", "error": repr(exc),
                                           "stream": item[0] if isinstance(item, tuple) and item else "?"})
                        run.files["events"].write(line + "\n")
                    except Exception:  # noqa: BLE001, S110
                        pass
            now = time.monotonic()
            if now - last_flush >= FLUSH_S:
                last_flush = now
                for f in run.files.values():
                    f.flush()
        for f in run.files.values():
            f.flush()

    def _write_camera(self, run: _Run, t: float, cam: int, b64, cmeta: dict) -> bool:
        """One frame -> ``camera/<key>/NNNNNN.jpg`` + one index line. False = nothing written: a repeat
        of the bridge's last ``seq`` (a bridge re-sending its latest capture), a frame ``camera_every``
        skips, or bytes that are not a JPEG (no SOI marker) — each counted per view in ``meta.camera``."""
        st = run.cams.get(cam)
        if st is None:
            # A camera the page subscribed to but nobody declared: recorded anyway, named by its index.
            st = run.cams[cam] = _cam_state({"cam": cam, "key": f"camera{cam}", "url": None})
            (run.dir / CAMERA_DIR / st["key"]).mkdir(parents=True, exist_ok=True)
            if CAMERA_STREAM not in run.files:
                run.files[CAMERA_STREAM] = open(run.dir / CAMERA_DIR / CAMERA_INDEX, "a", buffering=1 << 16)  # noqa: SIM115
        seq = cmeta.get("seq")
        if seq is not None and seq == st["last_seq"]:
            st["dup"] += 1
            return False
        st["last_seq"] = seq
        st["seen"] += 1
        if self.camera_every > 1 and (st["seen"] - 1) % self.camera_every:
            st["skipped"] += 1
            return False
        data = base64.b64decode(b64)
        if len(data) < 4 or data[:2] != _JPEG_SOI:
            st["bad"] += 1
            return False
        name = f"{st['n']:06d}.jpg"
        rel = f"{CAMERA_DIR}/{st['key']}/{name}"
        with open(run.dir / rel, "wb") as fh:
            fh.write(data)
        st["n"] += 1
        st["bytes"] += len(data)
        if st["t_first"] is None:
            st["t_first"] = t
        st["t_last"] = t
        rec = {"t": t, "cam": cam, "key": st["key"], "file": rel, "bytes": len(data), "seq": seq,
               "capture_seq": cmeta.get("capture_seq"), "t_bridge": cmeta.get("t"), "t_host": cmeta.get("t_host"),
               "w": cmeta.get("w"), "h": cmeta.get("h"), "stream": cmeta.get("stream")}
        run.files[CAMERA_STREAM].write(json.dumps(rec, separators=(",", ":")) + "\n")
        return True

    def _write(self, run: _Run, item: tuple) -> None:
        stream, t = item[0], float(item[1])
        if stream == CAMERA_STREAM:
            if self._write_camera(run, t, int(item[2]), item[3], item[4]):
                run.counts[CAMERA_STREAM] += 1
            return
        if stream == "events":
            kind, fields = item[2], item[3]
            rec = {"t": t, "kind": kind, **fields}
        elif stream == "measured":
            msg = item[2]
            kp = None
            if self.measure is not None:
                try:
                    kp = self.measure(msg)
                except Exception as exc:  # noqa: BLE001 - a failed FK is a null, never a stop
                    kp = None
                    run.errors += 1
                    if run.errors == 1:
                        self._log(f"RECORD: measured-keypoint FK failed ({exc!r}); recording joints only")
            armed = bool(msg.get("vive_armed", False)) if isinstance(msg, dict) else None
            if armed is not None and armed != run.armed_last:
                run.armed_last = armed
                self._write(run, ("events", t, "armed" if armed else "disarmed", {}))
            rec = {"t": t, "msg": msg,
                   "kp_pos": None if kp is None else _round(kp[0], ROUND_M),
                   "kp_orn": None if kp is None else _round(kp[1], ROUND_Q)}
        elif stream == "force":
            rec = {"t": t, "msg": item[2]}
        elif stream == "published":
            d = item[2]
            legs = bool(d["legs"])
            if legs != run.legs_last:
                if run.legs_last is not None or legs:
                    self._write(run, ("events", t, "legs_on" if legs else "legs_off", {}))
                run.legs_last = legs
            rec = {"t": t, "pos": _round(d["pos"], ROUND_M), "orn": _round(d["orn"], ROUND_Q),
                   "fcmd": _round(d["fcmd"], ROUND_N), "ok": d["ok"], "legs": legs,
                   "ref_joints": bool(d["ref_joints"])}
        elif stream == "steps":
            d = item[2]
            rec = {"t": t, "chunk": d["chunk"], "row": d["row"],
                   "target_pos": _round(d["target_pos"], ROUND_M),
                   "target_orn": _round(d["target_orn"], ROUND_Q),
                   "wrench": {k: _round(v, ROUND_N) for k, v in d["wrench"].items()},
                   "n_blended": d["n_blended"]}
        elif stream == "chunks":
            d = item[2]
            rec = {"t": t, "seq": d["seq"], "cadence": d["cadence"], "t_obs": d["t_obs"],
                   "seed": d["seed"], "sample_ms": d["sample_ms"], "wire_src": d["wire_src"],
                   "n_exec": d["n_exec"], "source": d.get("source"), "replay": d.get("replay"),
                   "decoded_pos": _round(d["decoded_pos"], ROUND_M),
                   "pos": _round(d["pos"], ROUND_M), "orn": _round(d["orn"], ROUND_Q),
                   "wrench": {k: _round(v, ROUND_N) for k, v in d["wrench"].items()},
                   "grip": {k: _round(v, ROUND_M) for k, v in d["grip"].items()}}
        else:
            raise ValueError(f"unknown stream {stream!r}")
        run.files[stream].write(json.dumps(rec, separators=(",", ":"), allow_nan=True) + "\n")
        run.counts[stream] += 1


class RolloutTaps:
    """What the REVIEWED page holds: one object with a ``recorder`` slot, whose methods are the taps
    the publish funnel, the predict, the two execute loops and the two bridge subscribers call.

    With ``recorder`` None (the page before the extension block installs one, and every test that
    builds the page without it) each call is one ``is None`` check — the reviewed paths are
    otherwise byte-identical. The block sets ``recorder`` once, at construction.
    """

    def __init__(self) -> None:
        self.recorder: RunRecorder | None = None

    # the two subscriber hooks (``_WSSubscriber(on_msg=)``): (msg, t_monotonic) on the socket thread
    def on_record(self, msg, t_mono: float) -> None:
        r = self.recorder
        if r is not None:
            r.measured(msg, t_mono)

    def on_force(self, msg, t_mono: float) -> None:
        r = self.recorder
        if r is not None:
            r.force(msg, t_mono)

    def camera_hook(self, cam: int) -> Callable[[object, float], None]:
        """The ``on_msg`` for camera ``cam``'s subscriber — one per view the page subscribes to, so a
        one-camera checkpoint installs one and a two-camera checkpoint two, with nothing to configure."""
        cam = int(cam)

        def on_camera(msg, t_mono: float) -> None:
            r = self.recorder
            if r is not None:
                r.camera(cam, msg, t_mono)

        on_camera.__name__ = f"on_camera{cam}"
        return on_camera

    # the page's seams
    def publish(self, frame: tuple, fcmd, ok: bool, *, override: bool = False, ref_legs: bool = False) -> None:
        r = self.recorder
        if r is not None:
            r.publish(frame, fcmd, ok, override=override, ref_legs=ref_legs)

    def step(self, seq: int, row: int, frame: tuple, wrench_rows: dict | None = None, *,
             n_blended: int | None = None) -> None:
        r = self.recorder
        if r is not None:
            r.step(seq, row, frame, wrench_rows, n_blended=n_blended)

    def chunk(self, **kw) -> None:
        r = self.recorder
        if r is not None:
            r.chunk(**kw)

    def event(self, kind: str, **fields) -> None:
        r = self.recorder
        if r is not None:
            r.event(kind, **fields)

    def begin(self, cadence: str = "sequential") -> bool:
        """The Predict seam: open a recording if the button is armed (or ``every_run``) and none is
        open — so a manual Predict → Run session is recorded exactly as a ▶▶ is. A no-op while one is
        open; one ``is None`` check with no recorder installed."""
        r = self.recorder
        return r is not None and r.run_begin(cadence)

    def end(self, note: str, *, wait: bool = False) -> str | None:
        """An operator END gesture — STOP, Reset, Ctrl-C (the low level disarming is seen by the
        recorder itself, on ``on_record``): close the recording. None when nothing was recording.
        ``wait=False`` returns at once and closes it on a helper thread (STOP must not wait on the
        disk); ``wait=True`` (shutdown) blocks until the folder is complete."""
        r = self.recorder
        if r is None or not r.recording:
            return None
        if wait:
            return r.run_end(note)
        return f"closing the recording ({note}) …" if r.end_async(note) else None

    def wrap_run(self, fn: Callable[[], None], cadence: str, *,
                 end_note: Callable[[], str] | None = None) -> Callable[[], None]:
        """``fn`` (a ▶▶ run's body) INSIDE the recording: opens one if the button is armed and none is
        open (``run_begin``), brackets the body as an auto-run (``auto_run_begin`` / ``auto_run_end``
        with the page's own end note) — and does NOT end the recording when the body returns or
        raises: that is the operator's (STOP / Reset / disarm / the button / Ctrl-C). With no
        recorder installed at run time the body runs as is."""

        def run() -> None:
            r = self.recorder
            if r is not None:
                r.run_begin(cadence)
                r.auto_run_begin(cadence)
            try:
                fn()
            finally:
                if r is not None:
                    note = ""
                    if end_note is not None:
                        try:
                            note = str(end_note())
                        except Exception:  # noqa: BLE001 - the note is decoration
                            note = ""
                    r.auto_run_end(note)

        run.__name__ = getattr(fn, "__name__", "run")
        return run


def plot_argv(run_dir: Path, python: str | None = None) -> list[str]:
    """The background plot command: ``<python> scripts/plot_rollout_record.py <run_dir>``, the
    script located relative to THIS file (installed or in the source tree)."""
    here = Path(__file__).resolve()
    candidates = [here.parents[3] / "scripts" / "plot_rollout_record.py",     # the source tree
                  Path.cwd() / "scripts" / "plot_rollout_record.py"]
    script = next((c for c in candidates if c.is_file()), candidates[0])
    return [python or sys.executable, str(script), str(run_dir)]


__all__ = [
    "STREAMS",
    "CAMERA_STREAM",
    "CAMERA_DIR",
    "CAMERA_INDEX",
    "CAMERA_MSG_KEYS",
    "STATES",
    "STATE_COLOR",
    "FLUSH_S",
    "run_stamp",
    "task_from_checkpoint",
    "camera_specs",
    "RunRecorder",
    "RolloutTaps",
    "plot_argv",
]
