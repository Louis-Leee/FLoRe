"""Deploy-time RECORDER for the online HL-v2 planner — "what did the policy see, plan, and
actually do?", written to disk as a re-runnable take.

The planner already computes everything worth keeping; it just throws it away. This module
is the buffer + atomic writer that keeps it, so a real-robot run becomes a dataset you can
replay, re-infer, and QC offline — the deploy-side sibling of the collection converter.

WHAT ONE TAKE HOLDS
-------------------
A take spans one EXECUTION, and is keyed on execution rather than on Predict because that is
the operator's own unit: press Predict as many times as you like, and the take that gets
written is the one whose plan was actually RUN.

  * ``predictions`` — the plan(s) that were executed: for each, the model's own observation
    (the 224x224 conditioning images it saw, the state, the force window), its raw
    ``(H, action_dim)`` chunk, the decode anchor, the seed, and how many steps were executed.
  * ``executed`` — the measured robot at ~30 Hz while the wire was moving: the full 29-DOF
    ``dof_pos`` readback (the operator's "真实run的轨迹（也就是机器人的各个joint）"), the FK'd
    LEFT TCP pose, the measured jaw, and the measured force.
  * ``commanded`` — the wire actually published per step, so a tracking error can be
    attributed to the servo rather than to the plan.
  * ``camera`` — the raw JPEG stream during execution, undecorated, at the bridge's own rate.

WHY THE OBSERVATION IS STORED UN-NORMALIZED. Quantile normalization is a property of the
CHECKPOINT (its ``norm_stats``), not of the rig. Storing the raw physical observation means a
recorded take can be re-inferred against a DIFFERENT checkpoint later — which is most of the
value of recording deploys at all — instead of freezing one model's scaling into the data.
The conditioning images are stored as the uint8 crop the normalizer consumed (recovered
through :func:`~flore.s1_policy.serving.hl_online.denormalize_image_hwc`, which round-trips
exactly) in a LOSSLESS format, so re-encoding them reproduces the fed tensor bit-for-bit —
see :meth:`TakeRecorder._write_predictions` for the 4.4 mm error a lossy one caused.

WHY A DIRECTORY OF ``.npy`` + JSON, and not a zarr. A deploy take is small (seconds, not the
whole session), is written from a live control process that must never block on compression,
and is read back by one tool. ``np.save`` on a background thread + an atomic
``.tmp``→``rename`` is the same shape ``record_teleop.py`` already uses for the teleop
recorder in the holosoma submodule, so an operator has one mental model for both. The
converter's zarr remains the format for TRAINING data.

THREADING. ``on_*`` methods only append to plain lists under one lock and are called from the
planner's control/drip/render threads; the flush (``np.save`` of a few MB + image writes) runs
on a worker thread so it never sits on the arm's critical path. A take is committed by
renaming a fully-written ``.tmp`` directory, so a process killed mid-flush can lose the newest
take but can never leave a half-written one that a reader would mistake for complete.

Robot-free and viser-free: everything here takes plain arrays and paths, so the buffering,
the schema, and the writer are unit-testable headless.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from flore.s1_policy.data.layout import ACTION_DT, ACTION_HORIZON, ACTION_HZ
from flore.s1_policy.serving.hl_online import ObsSnapshot, denormalize_image_hwc

#: Schema tag written into every ``take_meta.json``. Bump when a stored array's meaning
#: changes; the reader refuses an unknown value rather than silently misreading columns.
SCHEMA = "hl_deploy_v1"

#: Column layout of ``executed.npy`` — the measured robot while the wire was moving. One row
#: per render tick (~30 Hz). Written as ONE flat float64 array (not a dict of arrays) so it
#: loads as a plain 2-D matrix, with the names carried in ``take_meta.json`` so a reader never
#: has to guess an offset. Fixed width: a row that cannot be filled is not appended at all.
EXEC_COLS: tuple[str, ...] = (
    "t",  # seconds since the take armed (monotonic)
    "chunk_index",  # which executed prediction was dripping (-1 = none/ensembled)
    "step_index",  # which action step of that chunk (-1 = unknown/ensembled)
    # measured LEFT UMI-TCP in the PELVIS frame (FK of the live joints), the same point the
    # action chunk is authored about
    "tcp_x", "tcp_y", "tcp_z",
    "tcp_r00", "tcp_r01", "tcp_r02", "tcp_r10", "tcp_r11", "tcp_r12",  # rot6d
    "jaw_m",  # measured jaw width, metres (NaN when the gripper bridge was silent)
    "vive_armed",  # 1.0 while the low level was applying the wire
    # measured force the SERVO consumed (pelvis frame, post-gate), per hand L then R
    "fmeas_lx", "fmeas_ly", "fmeas_lz", "fmeas_rx", "fmeas_ry", "fmeas_rz",
)
N_EXEC_COLS = len(EXEC_COLS)

#: Column layout of ``commanded.npy`` — the wire published per executed step.
CMD_COLS: tuple[str, ...] = (
    "t",  # seconds since the take armed (monotonic)
    "chunk_index",
    "step_index",
    # the vr_3point wire: 3 contact points (L, R, HEAD) then 3 quats xyzw
    "pos_lx", "pos_ly", "pos_lz", "pos_rx", "pos_ry", "pos_rz", "pos_hx", "pos_hy", "pos_hz",
    "orn_lx", "orn_ly", "orn_lz", "orn_lw",
    "orn_rx", "orn_ry", "orn_rz", "orn_rw",
    "orn_hx", "orn_hy", "orn_hz", "orn_hw",
    "grip_cmd_m",  # commanded jaw width in metres (NaN when the jaw is read-only)
    "fcmd_x", "fcmd_y", "fcmd_z",  # published F^cmd (sensor frame) or NaN when pose-only
)
N_CMD_COLS = len(CMD_COLS)

#: 29-DOF joint readback, kept in its own array because it is the widest stream and the one a
#: reader most often wants alone. Row: ``[t, *dof_pos(29)]``.
N_DOF_COLS = 1 + 29

#: ``force_meas.npy`` — the RAW sensor-frame F/T stream at the sensor's own rate (~200 Hz),
#: logged during execution. Row: ``[t, *channels]``; the channel count is the layout's
#: ``force_hist_channels`` (6 per force hand) and is recorded in ``take_meta.json``.

#: How many EXECUTED predictions one take keeps in full (observation included).
#:
#: This is a MEMORY bound, and it exists because of the ACT temporal-ensembling cadence: that
#: path re-queries the policy EVERY step, so a 20 s sequential rollout at 30 Hz is ~600 genuine
#: inferences, each carrying its own ``(To,3,224,224)`` float32 observation — measured **602 KiB
#: of RAM per prediction and ~154 KiB on disk**, i.e. ~360 MB resident and ~95 MB written for one
#: take. A live control process must not grow like that unbounded.
#:
#: 400 keeps ~13 s of ensembled motion (and any realistic number of chunk-cadence plans, which
#: produce ONE prediction per execute). Past the cap the observation is DROPPED but the
#: prediction is still recorded — its chunk, anchor, seed and timing are small and are what the
#: replay needs — and the take's metadata says how many lost their observation, so a truncated
#: take can never be mistaken for a complete one. Nothing is silently discarded.
MAX_FULL_PREDICTIONS = 400


@dataclass
class RecordedPrediction:
    """One EXECUTED plan: the policy's input, its raw output, and what was run from it.

    ``obs`` is the un-normalized observation snapshot the model was fed (see the module
    docstring). ``prediction`` is the raw ``(H, action_dim)`` chunk in physical units, i.e.
    before any URDF/wire conversion — so a reader can re-decode it against any anchor.
    ``n_exec`` is how many steps were actually dripped, which is what makes the take
    self-describing about the execute-N-of-S choice.
    """

    # None when there was no model inference behind this plan (a recorded-episode REPLAY) or
    # when the memory cap dropped it (see MAX_FULL_PREDICTIONS). The reader distinguishes the
    # two cases from the take's `source` field + `n_obs_dropped`.
    obs: ObsSnapshot | None
    anchor_pos: np.ndarray  # (n_roles,3) decode anchor, pelvis frame
    anchor_rot6d: np.ndarray  # (n_roles,6) decode anchor, MODEL frame
    prediction: np.ndarray  # (H, action_dim) raw physical
    seed: int
    num_steps: int
    t_predict: float  # seconds since the take armed, when the sampler ran
    t_exec_start: float  # when its first step was published (NaN if never executed)
    n_exec: int
    source: str  # "model" | "replay" — a replayed chunk has no real observation
    replay_episode: int = -1
    replay_frame: int = -1
    n_predicts_before: int = 0  # how many Predicts were discarded before this one ran


@dataclass
class TakeBuffer:
    """In-RAM accumulation of ONE take. Appended under :class:`TakeRecorder`'s lock."""

    label: str = ""
    start_wall: float = 0.0
    start_mono: float = 0.0
    predictions: list[RecordedPrediction] = field(default_factory=list)
    executed: list[list[float]] = field(default_factory=list)
    dof: list[list[float]] = field(default_factory=list)
    commanded: list[list[float]] = field(default_factory=list)
    force_meas: list[list[float]] = field(default_factory=list)
    jpegs: list[bytes] = field(default_factory=list)
    jpeg_index: list[list[float]] = field(default_factory=list)  # [t, seq]
    _last_cam_seq: int = -1
    # Predict presses that were superseded by another Predict before any Run. Counted, not
    # stored: the operator explicitly asked to record "按run的时候为标准", and keeping every
    # discarded plan would multiply the take size by the number of retries for data that,
    # by construction, never moved the robot.
    n_predicts_discarded: int = 0
    # Predictions past MAX_FULL_PREDICTIONS whose OBSERVATION was dropped to bound memory.
    # Surfaced in take_meta.json (never silent): a reader must be able to tell "this plan had
    # no observation because it was a replay" from "…because the cap was hit".
    n_obs_dropped: int = 0

    def n_frames(self) -> dict[str, int]:
        return {
            "predictions": len(self.predictions),
            "executed": len(self.executed),
            "dof": len(self.dof),
            "commanded": len(self.commanded),
            "force_meas": len(self.force_meas),
            "camera": len(self.jpegs),
        }

    def total_rows(self) -> int:
        n = self.n_frames()
        return sum(n.values())


class TakeRecorder:
    """Buffer + atomic writer for deploy takes. One open take at a time.

    Lifecycle, driven by the planner:

        ``arm(label)``            a Run/Play is starting -> open a take
        ``on_prediction(...)``    an executed plan's inputs + raw output
        ``on_executed(...)``      one measured render tick (while dripping)
        ``on_commanded(...)``     one published wire step
        ``on_force(...)``         one raw F/T sample
        ``on_camera(...)``        one JPEG from the bridge
        ``close()``              flush to ``<session>/take_NNN/`` on a worker thread

    Every ``on_*`` is a no-op when no take is open, so the planner can call them
    unconditionally from its hot loops without an ``if recording`` at each site. All of them
    are cheap appends under one lock; the lock is never held across I/O.
    """

    def __init__(
        self,
        session_dir: str | Path,
        meta: dict | None = None,
        *,
        layout_mode: str = "left_only",
        image_mean: float = 0.5,
        image_std: float = 0.5,
    ):
        self.session_dir = Path(session_dir)
        self._lock = threading.Lock()
        self._buf: TakeBuffer | None = None
        # RESUME the numbering from what is already on disk rather than starting at 0. A second
        # planner run pointed at the same --record-dir (an operator re-launching after a crash,
        # or two sessions sharing a directory) would otherwise write take_000 again and DELETE
        # the first run's take — the writer's atomic rename removes the destination first, so
        # this is data loss, not a name collision.
        self._take_idx = self._next_take_index()
        self._flush_threads: list[threading.Thread] = []
        self._meta = dict(meta or {})
        self._session_written = False
        self.last_saved: str | None = None
        self.last_error: str | None = None
        self.layout_mode = str(layout_mode)
        # The image affine used to turn the stored NORMALIZED obs back into uint8. Taken from
        # the CHECKPOINT's own values, not defaulted: a mismatch here stores a correctly-shaped
        # but wrongly-exposed image, which would then re-normalize to a different tensor than
        # the model was fed — the one thing this recorder exists to keep exact.
        self._image_mean = float(image_mean)
        self._image_std = float(image_std)

    def _next_take_index(self) -> int:
        """One past the highest ``take_NNN`` already in the session directory (0 if empty).

        Counts ``.tmp`` leftovers too: a killed flush may still be renamed into place by a
        concurrent process, and reusing its number would collide."""
        highest = -1
        try:
            entries = list(self.session_dir.glob("take_*"))
        except OSError:
            return 0
        for entry in entries:
            digits = entry.name[len("take_"):].split(".")[0]
            if digits.isdigit():
                highest = max(highest, int(digits))
        return highest + 1

    # -- lifecycle --------------------------------------------------------
    @property
    def recording(self) -> bool:
        with self._lock:
            return self._buf is not None

    @property
    def take_index(self) -> int:
        """Index the NEXT take will be written as."""
        with self._lock:
            return self._take_idx

    def elapsed(self, now: float | None = None) -> float:
        """Seconds since the open take armed (0.0 when idle).

        Every stored ``t`` is take-relative, so callers stamp their rows through this rather
        than each keeping its own zero — two clocks would put the measured and commanded
        streams on different time axes, which is precisely what the comparison depends on."""
        mono = time.monotonic() if now is None else float(now)
        with self._lock:
            return 0.0 if self._buf is None else mono - self._buf.start_mono

    def counts(self) -> dict[str, int]:
        """Live row counts of the open take (all zeros when idle) — for a status readout."""
        with self._lock:
            if self._buf is None:
                return {k: 0 for k in
                        ("predictions", "executed", "dof", "commanded", "force_meas", "camera")}
            return self._buf.n_frames()

    def arm(self, label: str, *, now: float | None = None) -> None:
        """Open a fresh take, discarding anything already buffered.

        Restart-clears rather than concatenating: each take is one execution compared against
        its own plan, and splicing two would draw a path the arm never travelled (the same
        reason :meth:`flore.s1_policy.serving.hl_trace.ExecutionTrace.arm` clears)."""
        mono = time.monotonic() if now is None else float(now)
        with self._lock:
            self._buf = TakeBuffer(label=str(label), start_wall=time.time(), start_mono=mono)

    def note_discarded_predict(self) -> None:
        """A Predict was superseded before it ran. Counted into the open take's metadata."""
        with self._lock:
            if self._buf is not None:
                self._buf.n_predicts_discarded += 1

    def close(self, *, block: bool = False) -> str | None:
        """Close the open take and flush it. Returns the destination path, or None if there
        was nothing to write.

        ``block`` waits for the write (used on shutdown, where a daemon thread would be
        killed mid-``np.save`` and lose a just-finished take)."""
        with self._lock:
            buf = self._buf
            self._buf = None
            if buf is None:
                return None
            # Discard a take that recorded NOTHING (armed then immediately stopped, or an
            # offline page where no wire was ever published). A take with a prediction but no
            # motion is still kept: "the policy planned this and the arm did not move" is a
            # real observation worth having.
            if buf.total_rows() == 0:
                return None
            take_dir = self.session_dir / f"take_{self._take_idx:03d}"
            self._take_idx += 1
            self._flush_threads = [t for t in self._flush_threads if t.is_alive()]
        thread = threading.Thread(
            target=self._flush, args=(take_dir, buf), name=f"hl-record-flush-{take_dir.name}",
        )
        thread.start()
        with self._lock:
            self._flush_threads.append(thread)
        if block:
            thread.join()
        return str(take_dir)

    def close_pending(self, timeout: float = 30.0) -> None:
        """Shutdown: flush the open take synchronously, then join outstanding writes."""
        self.close(block=True)
        with self._lock:
            threads = list(self._flush_threads)
        deadline = time.monotonic() + float(timeout)
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))

    # -- ingest -----------------------------------------------------------
    def on_prediction(self, pred: RecordedPrediction) -> None:
        """Register one EXECUTED plan.

        Past :data:`MAX_FULL_PREDICTIONS` the OBSERVATION is dropped and the rest of the
        prediction is kept (see that constant for the ensembled-cadence memory arithmetic). The
        drop is counted so the take reports it rather than looking complete."""
        with self._lock:
            if self._buf is None:
                return
            pred.n_predicts_before = self._buf.n_predicts_discarded
            if len(self._buf.predictions) >= MAX_FULL_PREDICTIONS:
                if pred.obs is not None:
                    pred.obs = None
                    self._buf.n_obs_dropped += 1
            self._buf.predictions.append(pred)

    def on_executed(
        self,
        t: float,
        *,
        tcp_pos: np.ndarray,
        tcp_rot6d: np.ndarray,
        dof_pos: np.ndarray | None,
        jaw_m: float | None,
        vive_armed: bool,
        f_meas: np.ndarray | None,
        chunk_index: int = -1,
        step_index: int = -1,
    ) -> None:
        """One measured tick. Silently drops a non-finite pose (a bad FK must not poison the
        arrays), matching :meth:`ExecutionTrace.add`'s contract."""
        p = np.asarray(tcp_pos, dtype=np.float64).reshape(-1)
        r = np.asarray(tcp_rot6d, dtype=np.float64).reshape(-1)
        if p.shape[0] != 3 or r.shape[0] != 6:
            return
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(r))):
            return
        fm = np.full(6, np.nan)
        if f_meas is not None:
            arr = np.asarray(f_meas, dtype=np.float64)
            if arr.shape == (2, 3):
                fm = arr.reshape(-1)
        row = [float(t), float(chunk_index), float(step_index), *p.tolist(), *r.tolist(),
               float("nan") if jaw_m is None else float(jaw_m),
               1.0 if vive_armed else 0.0, *fm.tolist()]
        dof_row = None
        if dof_pos is not None:
            d = np.asarray(dof_pos, dtype=np.float64).reshape(-1)
            if d.shape[0] == 29 and np.all(np.isfinite(d)):
                dof_row = [float(t), *d.tolist()]
        with self._lock:
            if self._buf is None:
                return
            self._buf.executed.append(row)
            if dof_row is not None:
                self._buf.dof.append(dof_row)

    def on_commanded(
        self,
        t: float,
        *,
        wire_pos9: np.ndarray,
        wire_orn12: np.ndarray,
        grip_cmd_m: float | None,
        f_cmd: np.ndarray | None,
        chunk_index: int = -1,
        step_index: int = -1,
    ) -> None:
        """One published wire step (what was COMMANDED, before the servo)."""
        wp = np.asarray(wire_pos9, dtype=np.float64).reshape(-1)
        wo = np.asarray(wire_orn12, dtype=np.float64).reshape(-1)
        if wp.shape[0] != 9 or wo.shape[0] != 12:
            return
        fc = np.full(3, np.nan)
        if f_cmd is not None:
            arr = np.asarray(f_cmd, dtype=np.float64).reshape(-1)
            if arr.shape[0] >= 3:
                fc = arr[:3]
        row = [float(t), float(chunk_index), float(step_index), *wp.tolist(), *wo.tolist(),
               float("nan") if grip_cmd_m is None else float(grip_cmd_m), *fc.tolist()]
        with self._lock:
            if self._buf is not None:
                self._buf.commanded.append(row)

    def on_force(self, t: float, wrench: np.ndarray) -> None:
        """One RAW sensor-frame F/T sample, at the sensor's own rate."""
        w = np.asarray(wrench, dtype=np.float64).reshape(-1)
        if w.size == 0 or not np.all(np.isfinite(w)):
            return
        row = [float(t), *w.tolist()]
        with self._lock:
            if self._buf is not None:
                self._buf.force_meas.append(row)

    def on_camera(self, t: float, jpeg: bytes, seq: int) -> None:
        """One camera frame, deduped by the bridge's own ``seq`` (it bumps only on a genuine
        new capture, so polling faster than the camera cannot inflate the stream)."""
        if not jpeg:
            return
        with self._lock:
            if self._buf is None:
                return
            if int(seq) >= 0 and int(seq) == self._buf._last_cam_seq:
                return
            self._buf._last_cam_seq = int(seq)
            self._buf.jpegs.append(bytes(jpeg))
            self._buf.jpeg_index.append([float(t), float(seq)])

    # -- write ------------------------------------------------------------
    def _write_session_meta(self) -> None:
        if self._session_written:
            return
        self.session_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(self._meta)
        payload.setdefault("schema", SCHEMA)
        payload.setdefault("created_wall", time.time())
        (self.session_dir / "session_meta.json").write_text(json.dumps(payload, indent=2))
        self._session_written = True

    def _flush(self, take_dir: Path, buf: TakeBuffer) -> None:
        try:
            self._write_session_meta()
            self._write_take(take_dir, buf)
            self.last_saved = str(take_dir)
            self.last_error = None
            n = buf.n_frames()
            print(f"[hl-record] saved {take_dir.name}: {n['predictions']} prediction(s), "
                  f"{n['executed']} measured, {n['commanded']} commanded, {n['camera']} frames "
                  f"-> {take_dir}", flush=True)
            if buf.n_obs_dropped:
                # NO SILENT CAPS: say exactly what this take cannot answer.
                print(f"[hl-record] NOTE {take_dir.name}: {buf.n_obs_dropped} prediction(s) past "
                      f"the {MAX_FULL_PREDICTIONS}-plan cap were stored WITHOUT their "
                      f"observation (chunk/anchor/seed kept). Those cannot be re-inferred "
                      f"exactly; re-infer them by FRAME instead.", flush=True)
        except Exception as exc:  # noqa: BLE001 — a write failure must not kill the planner
            import traceback

            self.last_error = repr(exc)
            print(f"[hl-record] FAILED to write {take_dir}:\n{traceback.format_exc()}", flush=True)

    def _write_take(self, take_dir: Path, buf: TakeBuffer) -> None:
        tmp = Path(str(take_dir) + ".tmp")
        if tmp.is_dir():
            shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "camera").mkdir(parents=True, exist_ok=True)

        def _save(name: str, rows: list, width: int) -> None:
            arr = (np.asarray(rows, dtype=np.float64) if rows
                   else np.zeros((0, width), dtype=np.float64))
            np.save(tmp / name, arr)

        _save("executed.npy", buf.executed, N_EXEC_COLS)
        _save("dof_pos.npy", buf.dof, N_DOF_COLS)
        _save("commanded.npy", buf.commanded, N_CMD_COLS)
        force_ch = max((len(r) - 1 for r in buf.force_meas), default=0)
        _save("force_meas.npy", buf.force_meas, force_ch + 1)

        for i, jpeg in enumerate(buf.jpegs):
            (tmp / "camera" / f"frame_{i:06d}.jpg").write_bytes(jpeg)
        _save("camera/frame_index.npy", buf.jpeg_index, 2)

        pred_meta = self._write_predictions(tmp, buf)

        take_meta = {
            "schema": SCHEMA,
            "label": buf.label,
            # The layout the stored chunks are sliced by. Needed to READ them at all (the
            # per-role slot offsets come from it), so it lives in the take rather than being
            # assumed by the reader.
            "layout_mode": self.layout_mode,
            "image_mean": self._image_mean,
            "image_std": self._image_std,
            "start_wall": buf.start_wall,
            "stop_wall": time.time(),
            "duration_s": (buf.executed[-1][0] - buf.executed[0][0]) if len(buf.executed) >= 2 else 0.0,
            "n_frames": buf.n_frames(),
            "n_predicts_discarded": buf.n_predicts_discarded,
            "n_obs_dropped": buf.n_obs_dropped,
            "max_full_predictions": MAX_FULL_PREDICTIONS,
            "columns": {
                "executed": list(EXEC_COLS),
                "commanded": list(CMD_COLS),
                "dof_pos": ["t", *[f"dof_{i:02d}" for i in range(29)]],
                "force_meas": ["t", *[f"ch_{i}" for i in range(force_ch)]],
                "camera_index": ["t", "seq"],
            },
            "grid": {
                "action_horizon": ACTION_HORIZON,
                "action_hz": ACTION_HZ,
                "action_dt_s": ACTION_DT,
            },
            "predictions": pred_meta,
        }
        (tmp / "take_meta.json").write_text(json.dumps(take_meta, indent=2))

        if take_dir.is_dir():
            shutil.rmtree(take_dir, ignore_errors=True)
        os.rename(tmp, take_dir)

    def _write_predictions(self, tmp: Path, buf: TakeBuffer) -> list[dict]:
        """Write per-prediction arrays + return their JSON metadata.

        One subdirectory per executed prediction (``pred_000/`` …) holding the chunk, the
        anchor, the observation, and the conditioning images.

        THE CONDITIONING IMAGES ARE PNG (LOSSLESS), NOT JPEG. Storing them as JPEG looked free
        — the bridge already sent JPEG, so "re-encoding costs nothing that was not already
        lost" — and it is wrong: the model was fed the tensor built from the DECODED pixels, so
        a second lossy encode changes the input. Measured on the 0808 diffusion checkpoint, a
        JPEG-95 round trip moved the re-inferred chunk by **4.4 mm** mean position — the same
        order as a real model difference, which would have made every re-inference
        unfalsifiable. PNG round-trips bit-exactly, so a same-checkpoint same-seed re-run
        reproduces the deploy chunk EXACTLY (pinned by ``tests/test_hl_record.py``). The raw
        CAMERA stream stays JPEG: those are the bridge's own bytes, written verbatim, never
        re-encoded.
        """
        import cv2

        out: list[dict] = []
        for i, pred in enumerate(buf.predictions):
            pdir = tmp / f"pred_{i:03d}"
            pdir.mkdir(parents=True, exist_ok=True)
            np.save(pdir / "chunk.npy", np.asarray(pred.prediction, dtype=np.float64))
            np.save(pdir / "anchor_pos.npy", np.asarray(pred.anchor_pos, dtype=np.float64))
            np.save(pdir / "anchor_rot6d.npy", np.asarray(pred.anchor_rot6d, dtype=np.float64))
            n_images = 0
            obs = pred.obs
            if obs is not None:
                np.save(pdir / "state.npy", np.asarray(obs.state, dtype=np.float64))
                np.save(pdir / "state_valid.npy", np.asarray(obs.state_valid, dtype=bool))
                np.save(pdir / "force_history.npy",
                        np.asarray(obs.force_history, dtype=np.float64))
                images = np.asarray(obs.images, dtype=np.float32)
                for t in range(images.shape[0]):
                    rgb = denormalize_image_hwc(images[t], self._image_mean, self._image_std)
                    ok, enc = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    if ok:
                        (pdir / f"obs_{t:02d}.png").write_bytes(enc.tobytes())
                        n_images += 1
            out.append({
                "index": i,
                "dir": pdir.name,
                "source": pred.source,
                "seed": int(pred.seed),
                "num_steps": int(pred.num_steps),
                "t_predict_s": float(pred.t_predict),
                "t_exec_start_s": float(pred.t_exec_start),
                "n_exec": int(pred.n_exec),
                "chunk_shape": list(np.asarray(pred.prediction).shape),
                "n_obs_images": n_images,
                "force_history_valid": (None if obs is None else bool(obs.force_history_valid)),
                "replay_episode": int(pred.replay_episode),
                "replay_frame": int(pred.replay_frame),
                "n_predicts_before": int(pred.n_predicts_before),
            })
        return out


__all__ = [
    "CMD_COLS",
    "EXEC_COLS",
    "N_CMD_COLS",
    "N_DOF_COLS",
    "N_EXEC_COLS",
    "SCHEMA",
    "RecordedPrediction",
    "TakeBuffer",
    "TakeRecorder",
]
