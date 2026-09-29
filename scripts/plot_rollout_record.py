#!/usr/bin/env python
"""Plot ONE recorded ▶▶ run of the five-point page (the ⏺ button in ``scripts/hl_serving_5pt.py``) or of
the 3-point page (the same button in ``scripts/hl_serving.py``, since 2026-09-20): the wire's keypoints'
x / y / z — five (LEFT, RIGHT, head, two feet) or three (LEFT, RIGHT, head), the count read off
``meta.static.wire_labels`` — every predicted horizon, the executed target, the published wire, the
target the low level actually saw (post-slew) and the ROBOT (its joints FK'd) — and the force per hand
per axis — the policy's predicted F_meas, the F^cmd that went on the wire and the Bota's measured
wrench — plus a summary of the tracking and force errors.

    uv run scripts/plot_rollout_record.py data/rollouts/push_gantry/2026-09-12_17-03-21
    uv run scripts/plot_rollout_record.py data/rollouts/push_gantry --all          # every run under it
    uv run scripts/plot_rollout_record.py data/rollouts --all --force              # redo them all
    uv run scripts/plot_rollout_record.py <run> --summary-only                     # numbers, no PNG

Writes INTO the run folder: ``keypoints_xyz.png``, ``force_xyz.png``, ``force_lowlevel.png`` (when
RecordTap carried the low level's pelvis-frame force), ``camera_strip.png`` (when the run recorded
camera frames: one row per view, frames evenly spaced over the run, captioned with their time),
``summary.json`` and ``summary.txt``. Reads only; never touches the JSONL or the JPEGs. The page spawns this itself when a recording is saved (``plot.log`` in
the folder says how it went); ``--no-record-autoplot`` there leaves it to you.

WHAT EACH LINE IS (the recorder's docstring, ``fuji/s1_policy/serving/rollout_record.py``, is the
contract): ``target`` is the clutched model target per executed step, BEFORE reach compensation and
the leg override; ``published`` is what reached :8770 — reach residual + leg override applied, every
frame including the heartbeat's re-sends; ``low level saw`` is RecordTap's ``vr_cmd_pos``, the target
AFTER the low level's slew caps, so published-vs-saw is the clamp and saw-vs-robot is the controller's
own tracking error; ``robot`` is the FK of the 29 measured joints through the same ``KeypointFK`` the
clutch's leak bound uses. Every predicted horizon (grey) is drawn at the wall time its row 0 was
executed (sequential: the drip's step; stream: the step-clock stamp), so the fan of horizons over a
point shows how much consecutive chunks disagreed. Un-armed stretches are shaded grey, the leg
override orange.

FORCE: the model predicts ``F_meas`` (what the Bota read during the demo, SENSOR frame) and the wire
carries ``F^cmd = -F_meas`` gated at |F| <= gate — so ``predicted`` and ``measured`` are directly
comparable and ``F^cmd`` is their mirror image. The low level's own view (``f_meas_ungated`` /
``f_cmd`` on RecordTap) is in the PELVIS frame and gets its own figure rather than a shared axis.

Needs matplotlib (the ``viz`` extra). ``load_run`` / ``summarize`` need only numpy, so the numbers are
available where the plots are not.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

AXES = ("x", "y", "z")
FAXES = ("Fx", "Fy", "Fz")
DEFAULT_LABELS = ("LEFT hand", "RIGHT hand", "head/torso", "LEFT foot", "RIGHT foot")
HANDS = ("left", "right")
#: What a finished run folder contains once this ran; ``--all`` skips folders that have them.
#: ``camera_strip.png`` is deliberately NOT here: a run without camera frames never produces one, and
#: listing it would make ``--all`` redo every such folder forever.
OUTPUTS = ("keypoints_xyz.png", "force_xyz.png", "summary.json", "summary.txt")
#: The point count of a folder whose ``meta.static`` names no ``wire_labels`` (the five-point page's
#: early recordings); every later folder carries its labels and the count is read off them.
N_POINTS = 5
#: The recorder's camera folder + index (``rollout_record.CAMERA_DIR`` / ``CAMERA_INDEX``, spelled here
#: so this script stays importable with numpy alone).
CAMERA_DIR = "camera"
CAMERA_INDEX = "frames.jsonl"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict]:
    """Every parseable line. A torn LAST line (the page killed mid-write) is skipped, not fatal."""
    if not path.is_file():
        return []
    out: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _col(rows: list[dict], key: str, width: int, sub: str | None = None) -> np.ndarray:
    """``(N, width)`` float array of ``row[key]`` (or ``row[key][sub]``); a missing / None / wrong-width
    value is a NaN row, never an exception."""
    out = np.full((len(rows), width), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        v = r.get(key)
        if sub is not None and isinstance(v, dict):
            v = v.get(sub)
        if v is None:
            continue
        try:
            a = np.asarray(v, dtype=np.float64).reshape(-1)
        except (ValueError, TypeError):
            continue
        if a.size == width:
            out[i] = a
    return out


def _times(rows: list[dict], t0: float) -> np.ndarray:
    return np.asarray([float(r.get("t", np.nan)) for r in rows], dtype=np.float64) - t0


@dataclass
class Run:
    """One run folder, as arrays. Every ``t`` is seconds since the run began (``meta.t_begin``)."""

    dir: Path
    meta: dict
    t0: float
    labels: tuple[str, ...]
    hands: tuple[str, ...]
    dt: float
    horizon: int
    chunks: list[dict] = field(default_factory=list)
    steps: dict = field(default_factory=dict)
    published: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    force: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    #: ``camera/frames.jsonl`` as arrays (``t``, ``cam``, ``key``, ``file``, ``seq``); empty when none
    camera: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.meta.get('task', self.dir.parent.name)}/{self.meta.get('stamp', self.dir.name)}"

    @property
    def n_points(self) -> int:
        """How many keypoints the wire carried: 5 on the five-point page, 3 on the 3-point page."""
        return len(self.labels)

    @property
    def duration_s(self) -> float:
        d = self.meta.get("duration_s")
        if d is not None:
            return float(d)
        ts = [a["t"][-1] for a in (self.steps, self.published, self.measured, self.force) if a and a["t"].size]
        return float(max(ts)) if ts else 0.0

    def cadence(self) -> str:
        c = str((self.meta.get("run") or {}).get("cadence") or "")
        if not c and self.chunks:
            c = str(self.chunks[0].get("cadence", ""))
        return c or "?"


def load_run(run_dir: str | Path) -> Run:
    """Read a run folder (``meta.json`` + the six JSONL streams) into arrays."""
    d = Path(run_dir)
    meta_path = d / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{d} has no meta.json — not a finished run folder")
    with open(meta_path) as fh:
        meta = json.load(fh)
    static = meta.get("static") or {}
    t0 = float(meta.get("t_begin", 0.0))
    labels = tuple(static.get("wire_labels") or DEFAULT_LABELS)
    n_pts = len(labels)   # 5 (five-point page) or 3 (3-point page): every width below follows it
    hands = tuple(static.get("force_roles") or ())
    run = Run(dir=d, meta=meta, t0=t0, labels=labels, hands=hands,
              dt=float(static.get("action_dt") or (1.0 / 30.0)),
              horizon=int(static.get("action_horizon") or 50))

    # chunks — kept as dicts (ragged: pos may be None on the stream), arrays inside
    for r in _read_jsonl(d / "chunks.jsonl"):
        c = dict(r)
        c["t"] = float(r.get("t", t0)) - t0
        for k in ("decoded_pos", "pos", "orn"):
            c[k] = None if r.get(k) is None else np.asarray(r[k], dtype=np.float64)
        c["wrench"] = {h: (None if v is None else np.asarray(v, dtype=np.float64))
                       for h, v in (r.get("wrench") or {}).items()}
        c["grip"] = {h: (None if v is None else np.asarray(v, dtype=np.float64))
                     for h, v in (r.get("grip") or {}).items()}
        run.chunks.append(c)

    rows = _read_jsonl(d / "steps.jsonl")
    run.steps = {"t": _times(rows, t0),
                 "chunk": np.asarray([int(r.get("chunk", -1)) for r in rows], dtype=np.int64),
                 "row": np.asarray([int(r.get("row", -1)) for r in rows], dtype=np.int64),
                 "target_pos": _col(rows, "target_pos", 3 * n_pts),
                 "n_blended": np.asarray([r.get("n_blended") if r.get("n_blended") is not None else np.nan
                                         for r in rows], dtype=np.float64)}
    for h in HANDS:
        run.steps[f"wrench_{h}"] = _col(rows, "wrench", 6, sub=h)
    if not hands:   # a folder written before the layout named its hands: infer from the data
        run.hands = tuple(h for h in HANDS if np.isfinite(run.steps[f"wrench_{h}"]).any())

    rows = _read_jsonl(d / "published.jsonl")
    run.published = {"t": _times(rows, t0), "pos": _col(rows, "pos", 3 * n_pts),
                     "orn": _col(rows, "orn", 4 * n_pts), "fcmd": _col(rows, "fcmd", 6),
                     "legs": np.asarray([bool(r.get("legs", False)) for r in rows], dtype=bool),
                     "ok": np.asarray([bool(r.get("ok", True)) for r in rows], dtype=bool),
                     "ref_joints": np.asarray([bool(r.get("ref_joints", False)) for r in rows], dtype=bool)}

    rows = _read_jsonl(d / "measured.jsonl")
    msgs = [r.get("msg") if isinstance(r.get("msg"), dict) else {} for r in rows]
    vr = np.full((len(rows), 3 * n_pts), np.nan)
    for i, m in enumerate(msgs):
        v = m.get("vr_cmd_pos")
        if v is None:
            continue
        try:
            a = np.asarray(v, dtype=np.float64)
        except (ValueError, TypeError):
            continue
        if a.ndim == 2 and a.shape[1] == 3 and a.shape[0] == n_pts:
            vr[i] = a.reshape(-1)
    run.measured = {"t": _times(rows, t0), "kp_pos": _col(rows, "kp_pos", 3 * n_pts),
                    "kp_orn": _col(rows, "kp_orn", 4 * n_pts), "vr_cmd_pos": vr,
                    "armed": np.asarray([bool(m.get("vive_armed", False)) for m in msgs], dtype=bool),
                    "running": np.asarray([bool(m.get("running", False)) for m in msgs], dtype=bool),
                    "seq": np.asarray([int(m.get("seq", -1)) for m in msgs], dtype=np.int64),
                    "f_meas": _col(msgs, "f_meas", 6), "f_meas_ungated": _col(msgs, "f_meas_ungated", 6),
                    "f_cmd": _col(msgs, "f_cmd", 6), "dof_pos": _col(msgs, "dof_pos", 29),
                    "ref_joints_live": np.asarray([m.get("ref_joints_live") is True for m in msgs], dtype=bool)}

    rows = _read_jsonl(d / "force.jsonl")
    msgs = [r.get("msg") if isinstance(r.get("msg"), dict) else {} for r in rows]
    run.force = {"t": _times(rows, t0),
                 "t_bridge": np.asarray([float(m.get("t", np.nan)) for m in msgs], dtype=np.float64) - t0}
    for h in HANDS:
        run.force[f"{h}_pipe"] = _col(msgs, f"{h}_pipe", 6)
        run.force[h] = _col(msgs, h, 3)
    run.events = [dict(r, t=float(r.get("t", t0)) - t0) for r in _read_jsonl(d / "events.jsonl")]

    rows = [r for r in _read_jsonl(d / CAMERA_DIR / CAMERA_INDEX) if r.get("file")]
    if rows:
        run.camera = {"t": _times(rows, t0),
                      "cam": np.asarray([int(r.get("cam", -1)) for r in rows], dtype=np.int64),
                      "key": np.asarray([str(r.get("key", f"camera{r.get('cam', '?')}")) for r in rows], dtype=object),
                      "file": [str(r["file"]) for r in rows],
                      "seq": np.asarray([int(r["seq"]) if r.get("seq") is not None else -1 for r in rows], dtype=np.int64),
                      "bytes": np.asarray([int(r.get("bytes", 0) or 0) for r in rows], dtype=np.int64)}
    return run


# ---------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------
def hold_at(t_src: np.ndarray, v_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Zero-order hold of a sampled signal at ``t_query`` (the latest sample at or before each query;
    NaN before the first). Commands ARE held between publishes, so this is the right resampler for a
    command, and a fine one for a 50 Hz measurement against a 60 Hz command."""
    t_src = np.asarray(t_src, dtype=np.float64)
    v_src = np.asarray(v_src, dtype=np.float64)
    t_query = np.asarray(t_query, dtype=np.float64)
    out = np.full((t_query.size,) + v_src.shape[1:], np.nan)
    if t_src.size == 0 or t_query.size == 0:
        return out
    order = np.argsort(t_src, kind="stable")
    ts, vs = t_src[order], v_src[order]
    idx = np.searchsorted(ts, t_query, side="right") - 1
    ok = idx >= 0
    out[ok] = vs[idx[ok]]
    return out


def intervals(t: np.ndarray, flag: np.ndarray, t_end: float | None = None) -> list[tuple[float, float]]:
    """``[(t_on, t_off), …]`` where ``flag`` is True, closed at ``t_end`` (or the last sample)."""
    t = np.asarray(t, dtype=np.float64)
    flag = np.asarray(flag, dtype=bool)
    if t.size == 0:
        return []
    out: list[tuple[float, float]] = []
    on: float | None = None
    for ti, fi in zip(t, flag):
        if fi and on is None:
            on = float(ti)
        elif not fi and on is not None:
            out.append((on, float(ti)))
            on = None
    if on is not None:
        out.append((on, float(t_end if t_end is not None else t[-1])))
    return out


def _mask_in(t: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(t.shape, dtype=bool)
    for a, b in spans:
        m |= (t >= a) & (t <= b)
    return m


def _err_stats(err: np.ndarray) -> dict:
    """``{rms, max_abs, mean, n}`` over the finite entries of a 1-D error, or NaNs when there are none."""
    e = np.asarray(err, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"rms": float("nan"), "max_abs": float("nan"), "mean": float("nan"), "n": 0}
    return {"rms": float(np.sqrt(np.mean(e * e))), "max_abs": float(np.abs(e).max()),
            "mean": float(e.mean()), "n": int(e.size)}


def chunk_horizons(run: Run) -> list[dict]:
    """Every predicted horizon placed in time and in the WIRE frame: ``{"seq", "t": (H,), "pos": (H,15),
    "wrench": {hand: (H,6)}, "n_exec", "cadence"}``.

    Sequential: row 0 executed at the drip step ``(chunk == seq, row == 0)``; its ``pos`` is the
    clutched plan itself. Stream: row k belongs to absolute step ``t_obs + k``; the plan is per step
    there, so the chunk's decoded shape is translated onto the executed target at step ``t_obs``
    (``obs_mode=last_target`` makes that target the chunk's own anchor, up to the blend). A chunk whose
    row 0 was never executed (the run ended first) is dropped.
    """
    st = run.steps
    if not st or st["t"].size == 0:
        return []
    out: list[dict] = []
    for c in run.chunks:
        seq = int(c.get("seq", -1))
        cadence = str(c.get("cadence", ""))
        base = None
        if cadence == "stream" and c.get("t_obs") is not None:
            hit = np.where((st["chunk"] == -1) & (st["row"] == int(c["t_obs"])))[0]
            if hit.size == 0:
                hit = np.where((st["chunk"] == -1) & (st["row"] == int(c["t_obs"]) + 1))[0]
            if hit.size == 0:
                continue
            t_row0 = float(st["t"][hit[0]])
            dec = c.get("decoded_pos")
            if dec is None:
                continue
            base = dec - dec[0][None, :] + st["target_pos"][hit[0]][None, :]
        else:
            hit = np.where((st["chunk"] == seq) & (st["row"] == 0))[0]
            if hit.size == 0:
                continue
            t_row0 = float(st["t"][hit[0]])
            base = c.get("pos")
            if base is None and c.get("decoded_pos") is not None:
                dec = c["decoded_pos"]
                base = dec - dec[0][None, :] + st["target_pos"][hit[0]][None, :]
        if base is None:
            continue
        h = int(base.shape[0])
        out.append({"seq": seq, "cadence": cadence, "t": t_row0 + run.dt * np.arange(h), "pos": base,
                    "wrench": {k: v for k, v in c["wrench"].items() if v is not None},
                    "n_exec": c.get("n_exec")})
    return out


def summarize(run: Run) -> dict:
    """The numbers: per point per axis (robot − published, robot − low level saw, published − saw), per
    hand per axis (measured − predicted force), durations and counts. NaN where a stream is absent."""
    pub, meas, st, force = run.published, run.measured, run.steps, run.force
    t_end = run.duration_s
    legs = intervals(pub["t"], pub["legs"], t_end) if pub and pub["t"].size else []
    armed = intervals(meas["t"], meas["armed"], t_end) if meas and meas["t"].size else []
    have_meas = bool(meas and meas["t"].size and np.isfinite(meas["kp_pos"]).any())
    have_saw = bool(meas and meas["t"].size and np.isfinite(meas["vr_cmd_pos"]).any())
    # Score over ARMED time when the low level said so (an un-armed low level ignores the wire, so
    # its "error" is not a tracking error); everywhere when RecordTap never told us.
    if have_meas or have_saw:
        sel = _mask_in(meas["t"], armed) if armed else np.ones(meas["t"].shape, dtype=bool)
        pub_at = hold_at(pub["t"], pub["pos"], meas["t"]) if pub["t"].size else np.full_like(meas["kp_pos"], np.nan)
    points = {}
    for i, lab in enumerate(run.labels):
        row = {}
        for a, ax in enumerate(AXES):
            j = 3 * i + a
            cell = {}
            if have_meas:
                cell["robot_minus_published"] = _err_stats((meas["kp_pos"][:, j] - pub_at[:, j])[sel])
            if have_meas and have_saw:
                cell["robot_minus_lowlevel_target"] = _err_stats(
                    (meas["kp_pos"][:, j] - meas["vr_cmd_pos"][:, j])[sel])
            if have_saw:
                cell["published_minus_lowlevel_target"] = _err_stats(
                    (pub_at[:, j] - meas["vr_cmd_pos"][:, j])[sel])
            if pub["t"].size:
                p = pub["pos"][:, j]
                p = p[np.isfinite(p)]
                cell["published_span_m"] = float(p.max() - p.min()) if p.size else float("nan")
            row[ax] = cell
        points[lab] = row
    hands = {}
    for h in run.hands:
        pipe = force.get(f"{h}_pipe")
        pred = st.get(f"wrench_{h}")
        hrow = {}
        have_pipe = pipe is not None and pipe.size and np.isfinite(pipe).any()
        have_pred = pred is not None and pred.size and np.isfinite(pred).any()
        if have_pipe and have_pred:
            # Compare over the executed span only: before the first step nothing was predicted.
            t_lo, t_hi = float(st["t"].min()), float(st["t"].max()) + run.dt
            win = (force["t"] >= t_lo) & (force["t"] <= t_hi)
            pred_at = hold_at(st["t"], pred, force["t"])
            for a, ax in enumerate(FAXES):
                hrow[ax] = {"measured_minus_predicted": _err_stats((pipe[:, a] - pred_at[:, a])[win]),
                            "measured_abs_max": float(np.nanmax(np.abs(pipe[win, a]))) if win.any() else float("nan"),
                            "predicted_abs_max": float(np.nanmax(np.abs(pred[:, a])))}
            mag_m = np.linalg.norm(pipe[win][:, :3], axis=1) if win.any() else np.zeros(0)
            mag_p = np.linalg.norm(pred[:, :3], axis=1)
            hrow["|F|"] = {"measured_max": float(np.nanmax(mag_m)) if mag_m.size else float("nan"),
                           "measured_mean": float(np.nanmean(mag_m)) if mag_m.size else float("nan"),
                           "predicted_max": float(np.nanmax(mag_p)), "predicted_mean": float(np.nanmean(mag_p))}
        elif have_pred:
            hrow["note"] = "no F/T stream recorded (bridge :8771 absent?) — prediction only"
        elif have_pipe:
            hrow["note"] = "no wrench predicted for this hand (layout) — measurement only"
        else:
            hrow["note"] = "no force data"
        if pub["t"].size and np.isfinite(pub["fcmd"]).any():
            k = 3 * HANDS.index(h) if h in HANDS else 0
            fc = pub["fcmd"][:, k:k + 3]
            fin = np.isfinite(fc).all(axis=1)
            hrow["fcmd_on_wire"] = {"frames_with_fcmd": int(fin.sum()),
                                    "nonzero_frames": int((np.linalg.norm(fc[fin], axis=1) > 1e-9).sum()),
                                    "max_abs": float(np.abs(fc[fin]).max()) if fin.any() else 0.0}
        hands[h] = hrow
    step_dt = np.diff(st["t"]) if st and st["t"].size > 1 else np.zeros(0)
    pub_dt = np.diff(pub["t"]) if pub and pub["t"].size > 1 else np.zeros(0)
    return {
        "run": run.name, "dir": str(run.dir), "cadence": run.cadence(), "duration_s": round(t_end, 3),
        "camera": camera_summary(run),
        "counts": run.meta.get("counts", {}),
        "chunks": len(run.chunks),
        "steps": int(st["t"].size) if st else 0,
        "step_period_ms": {"mean": float(step_dt.mean() * 1e3) if step_dt.size else float("nan"),
                           "max": float(step_dt.max() * 1e3) if step_dt.size else float("nan")},
        "publish_period_ms": {"mean": float(pub_dt.mean() * 1e3) if pub_dt.size else float("nan"),
                              "max": float(pub_dt.max() * 1e3) if pub_dt.size else float("nan")},
        "refused_frames": int((~pub["ok"]).sum()) if pub and pub["t"].size else 0,
        "armed_intervals_s": [[round(a, 3), round(b, 3)] for a, b in armed],
        "armed_fraction": (sum(b - a for a, b in armed) / t_end) if (armed and t_end > 0) else None,
        "legs_override_intervals_s": [[round(a, 3), round(b, 3)] for a, b in legs],
        "frames_with_ref_joints": int(pub["ref_joints"].sum()) if pub and pub["t"].size else 0,
        "measured_keypoints": have_meas, "lowlevel_target_seen": have_saw,
        "points_m": points, "hands_N": hands,
        "end_note": run.meta.get("end_note", ""),
        "settings": run.meta.get("run", {}),
        "checkpoint": (run.meta.get("static") or {}).get("checkpoint"),
        "source": chunk_source(run),
    }


def chunk_source(run: Run) -> dict:
    """Where the chunks came from: ``{"source": "model" | "replay" | "mixed", "replay": {...} | None}``.
    A RECORDED-EPISODE REPLAY (``hl_serving_5pt.py --replay-zarr``) stamps every chunk with
    ``source="replay"`` and the episode's identity; a model run stamps nothing (read as ``model``)."""
    sources = {str(c.get("source") or "model") for c in run.chunks}
    src = "model" if not sources else (sources.pop() if len(sources) == 1 else "mixed")
    rp = None
    for c in run.chunks:
        if c.get("replay"):
            r = dict(c["replay"])
            if rp is None:
                rp = {"zarr": r.get("zarr"), "mode": r.get("mode"), "episodes": [], "first_frame": r.get("frame"),
                      "last_frame": r.get("frame"), "n_ep": r.get("n_ep")}
            if r.get("episode") not in rp["episodes"]:
                rp["episodes"].append(r.get("episode"))
            rp["last_frame"] = r.get("frame")
    return {"source": src, "replay": rp}


def _view_keys(cam: dict) -> list[str]:
    """The views in CAMERA-INDEX order (camera0_rgb before camera1_rgb), whatever order their first frames
    arrived in."""
    first: dict[str, int] = {}
    for c, k in zip(cam["cam"].tolist(), cam["key"].tolist()):
        first[str(k)] = min(int(c), first.get(str(k), int(c)))
    return sorted(first, key=lambda k: (first[k], k))


def camera_summary(run: Run) -> dict:
    """Per view: frames on disk, the rate they came at, their size — from the index (the truth on disk),
    with the recorder's own per-view drop counters (``meta.camera.views``) beside them when present."""
    views: dict[str, dict] = {}
    cam = run.camera
    if cam and cam["t"].size:
        for key in _view_keys(cam):
            sel = cam["key"] == key
            t = cam["t"][sel]
            span = float(t[-1] - t[0]) if t.size > 1 else 0.0
            views[str(key)] = {"frames": int(sel.sum()), "fps": round((t.size - 1) / span, 2) if span > 0 else None,
                               "mb": round(float(cam["bytes"][sel].sum()) / 1e6, 2),
                               "t_first_s": round(float(t[0]), 3), "t_last_s": round(float(t[-1]), 3),
                               "dir": f"{CAMERA_DIR}/{key}"}
    meta_cam = run.meta.get("camera") or {}
    for v in meta_cam.get("views") or []:
        key = str(v.get("key"))
        row = views.setdefault(key, {"frames": int(v.get("frames") or 0), "fps": v.get("fps"), "mb": None,
                                     "t_first_s": None, "t_last_s": None, "dir": v.get("dir")})
        row["recorder"] = {k: v.get(k) for k in ("seen", "duplicates", "skipped", "bad", "url")}
    return {"views": views, "n_views": len(views), "frames": int(sum(v["frames"] for v in views.values())),
            "every": meta_cam.get("every"), "index": f"{CAMERA_DIR}/{CAMERA_INDEX}" if views else None}


def summary_text(s: dict) -> str:
    """The summary as a readable table (also ``summary.txt``)."""
    lines = [f"run {s['run']}  ·  {s['cadence']}  ·  {s['duration_s']:.1f} s  ·  {s['chunks']} chunk(s) / "
             f"{s['steps']} step(s)  ·  step {s['step_period_ms']['mean']:.1f} ms mean / "
             f"{s['step_period_ms']['max']:.0f} max  ·  publish {s['publish_period_ms']['mean']:.1f} ms mean",
             f"end: {s['end_note']}",
             f"armed {s['armed_fraction'] * 100:.0f}% of the run" if s["armed_fraction"] is not None
             else "armed: unknown (no RecordTap)",
             f"leg override: {len(s['legs_override_intervals_s'])} interval(s) "
             f"{s['legs_override_intervals_s']}" if s["legs_override_intervals_s"] else "leg override: none"]
    src = s.get("source") or {}
    if src.get("source", "model") != "model":
        rp = src.get("replay") or {}
        n_ep = rp.get("n_ep")
        of_n = "" if n_ep is None else f" of {n_ep}"
        lines.append(f"source: {src['source']} — RECORDED EPISODE(S) {rp.get('episodes')} of {rp.get('zarr')} · "
                     f"{rp.get('mode')} · frames {rp.get('first_frame')}..{rp.get('last_frame')}{of_n} "
                     "(the chunks are the recording, not a prediction)")
    cam = s.get("camera") or {}
    if cam.get("views"):
        parts = []
        for key, v in cam["views"].items():
            fps = f"{v['fps']:.1f} fps" if v.get("fps") is not None else "rate n/a"
            mb = f" · {v['mb']:.1f} MB" if v.get("mb") is not None else ""
            drops = v.get("recorder") or {}
            bad = sum(int(drops.get(k) or 0) for k in ("bad",))
            parts.append(f"{key} {v['frames']} frames · {fps}{mb}" + (f" · ⚠️ {bad} not-JPEG" if bad else ""))
        every = cam.get("every")
        lines.append("camera: " + "; ".join(parts) + (f"  (every {every}-th frame)" if every and every > 1 else "")
                     + f"  → {cam['index']}")
    else:
        lines.append("camera: no frames recorded")
    lines.append("")
    if s["measured_keypoints"]:
        lines.append("tracking, ROBOT − PUBLISHED (mm, over armed time): rms / max|·| / mean")
        for lab, row in s["points_m"].items():
            cells = []
            for ax in AXES:
                e = row[ax].get("robot_minus_published")
                cells.append(f"{ax}: {e['rms'] * 1e3:6.1f} / {e['max_abs'] * 1e3:6.1f} / {e['mean'] * 1e3:+6.1f}"
                             if e and e["n"] else f"{ax}:   n/a")
            lines.append(f"  {lab:<11} " + "   ".join(cells))
        if s["lowlevel_target_seen"]:
            lines.append("clamp, PUBLISHED − LOW LEVEL SAW (mm): rms / max|·|   — the slew caps' bite")
            for lab, row in s["points_m"].items():
                cells = []
                for ax in AXES:
                    e = row[ax].get("published_minus_lowlevel_target")
                    cells.append(f"{ax}: {e['rms'] * 1e3:6.1f} / {e['max_abs'] * 1e3:6.1f}" if e and e["n"]
                                 else f"{ax}:   n/a")
                lines.append(f"  {lab:<11} " + "   ".join(cells))
    else:
        lines.append("tracking: no measured keypoints (RecordTap absent, or the FK unavailable)")
    lines.append("")
    for h, row in s["hands_N"].items():
        if "note" in row:
            lines.append(f"force {h}: {row['note']}")
            continue
        cells = []
        for ax in FAXES:
            e = row[ax]["measured_minus_predicted"]
            cells.append(f"{ax}: {e['rms']:5.2f} / {e['max_abs']:5.2f} / {e['mean']:+5.2f}")
        m = row["|F|"]
        lines.append(f"force {h}, MEASURED − PREDICTED (N): rms / max|·| / mean   " + "   ".join(cells))
        lines.append(f"      |F| measured max {m['measured_max']:.2f} mean {m['measured_mean']:.2f} · "
                     f"predicted max {m['predicted_max']:.2f} mean {m['predicted_mean']:.2f}"
                     + (f" · F^cmd on the wire: {row['fcmd_on_wire']['nonzero_frames']}/"
                        f"{row['fcmd_on_wire']['frames_with_fcmd']} frames non-zero, max "
                        f"{row['fcmd_on_wire']['max_abs']:.2f} N" if "fcmd_on_wire" in row else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _shade(ax, spans, color, label):
    first = True
    for a, b in spans:
        ax.axvspan(a, b, color=color, alpha=0.12, lw=0, label=label if first else None)
        first = False


def plot_keypoints(run: Run, out: Path, *, dpi: int = 110) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pub, meas, st = run.published, run.measured, run.steps
    horizons = chunk_horizons(run)
    t_end = run.duration_s
    legs = intervals(pub["t"], pub["legs"], t_end) if pub and pub["t"].size else []
    unarmed = intervals(meas["t"], ~meas["armed"], t_end) if meas and meas["t"].size else []
    ready = (run.meta.get("static") or {}).get("ready_pos")
    n_pts = run.n_points
    fig, axes = plt.subplots(n_pts, 3, figsize=(17, 3 * n_pts), sharex=True, squeeze=False)
    for i in range(n_pts):
        for a in range(3):
            ax = axes[i, a]
            j = 3 * i + a
            for k, h in enumerate(horizons):
                ax.plot(h["t"], h["pos"][:, j], color="0.55", lw=0.7, alpha=0.45,
                        label="predicted horizons (each chunk)" if (i == 0 and a == 0 and k == 0) else None)
            if st and st["t"].size:
                ax.step(st["t"], st["target_pos"][:, j], where="post", color="C0", lw=1.2,
                        label="executed target (clutched, pre-reach/legs)" if (i == 0 and a == 0) else None)
            if pub and pub["t"].size:
                ax.plot(pub["t"], pub["pos"][:, j], color="C1", lw=1.0, alpha=0.9,
                        label="published on :8770 (as sent)" if (i == 0 and a == 0) else None)
            if meas and meas["t"].size and np.isfinite(meas["vr_cmd_pos"][:, j]).any():
                ax.plot(meas["t"], meas["vr_cmd_pos"][:, j], color="C2", lw=1.0, ls="--",
                        label="low level saw (post-slew, RecordTap)" if (i == 0 and a == 0) else None)
            if meas and meas["t"].size and np.isfinite(meas["kp_pos"][:, j]).any():
                ax.plot(meas["t"], meas["kp_pos"][:, j], color="k", lw=1.6,
                        label="ROBOT (FK of measured joints)" if (i == 0 and a == 0) else None)
            if ready is not None and len(ready) == 3 * n_pts:
                ax.axhline(float(ready[j]), color="0.3", lw=0.6, ls=":",
                           label="ready pose (gate reference)" if (i == 0 and a == 0) else None)
            _shade(ax, legs, "tab:orange", "leg override engaged" if (i == 0 and a == 0) else None)
            _shade(ax, unarmed, "0.2", "low level UN-ARMED" if (i == 0 and a == 0) else None)
            ax.set_title(f"{run.labels[i]} · {AXES[a]}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if a == 0:
                ax.set_ylabel("m (pelvis frame)")
            if i == n_pts - 1:
                ax.set_xlabel("s since ▶▶")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    settings = run.meta.get("run") or {}
    fig.suptitle(f"{run.name} — {run.cadence()} · exec {settings.get('exec_steps', '?')} step(s)/chunk · "
                 f"{len(run.chunks)} chunk(s) / {st['t'].size if st else 0} step(s) · {run.duration_s:.1f} s · "
                 f"{Path(str((run.meta.get('static') or {}).get('checkpoint', ''))).name}\n"
                 f"end: {run.meta.get('end_note', '')}", fontsize=11)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_force(run: Run, out: Path, *, dpi: int = 110) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pub, st, force = run.published, run.steps, run.force
    hands = tuple(run.hands) or HANDS
    horizons = chunk_horizons(run)
    t_end = run.duration_s
    legs = intervals(pub["t"], pub["legs"], t_end) if pub and pub["t"].size else []
    fig, axes = plt.subplots(len(hands), 3, figsize=(17, 4.2 * len(hands)), sharex=True, squeeze=False)
    for r, h in enumerate(hands):
        pipe = force.get(f"{h}_pipe")
        pred = st.get(f"wrench_{h}")
        k = 3 * HANDS.index(h) if h in HANDS else 0
        for a in range(3):
            ax = axes[r, a]
            first = r == 0 and a == 0
            for n, hz in enumerate(horizons):
                w = hz["wrench"].get(h)
                if w is not None and w.shape[0] == hz["t"].size:
                    ax.plot(hz["t"], w[:, a], color="0.55", lw=0.6, alpha=0.4,
                            label="predicted horizons (each chunk)" if (first and n == 0) else None)
            if pipe is not None and pipe.size and np.isfinite(pipe[:, a]).any():
                ax.plot(force["t"], pipe[:, a], color="C3", lw=0.8, alpha=0.9,
                        label="MEASURED F_meas (Bota *_pipe, sensor frame)" if first else None)
            if pred is not None and pred.size and np.isfinite(pred[:, a]).any():
                ax.step(st["t"], pred[:, a], where="post", color="C0", lw=1.4,
                        label="policy's predicted F_meas (executed step)" if first else None)
            if pub and pub["t"].size and np.isfinite(pub["fcmd"][:, k + a]).any():
                ax.plot(pub["t"], pub["fcmd"][:, k + a], color="C1", lw=1.0, ls="--",
                        label="F^cmd on the wire (= −F_meas, gated)" if first else None)
            _shade(ax, legs, "tab:orange", "leg override engaged" if first else None)
            ax.axhline(0.0, color="0.3", lw=0.5)
            ax.set_title(f"{h.upper()} hand · {FAXES[a]}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if a == 0:
                ax.set_ylabel("N (sensor frame)")
            if r == len(hands) - 1:
                ax.set_xlabel("s since ▶▶")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, frameon=False)
    settings = run.meta.get("run") or {}
    fig.suptitle(f"{run.name} — force · F^cmd {'ON' if settings.get('force_cmd') else 'off'} · gate "
                 f"{settings.get('force_gate_n', '?')} N · {run.cadence()}", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_force_lowlevel(run: Run, out: Path, *, dpi: int = 110) -> Path | None:
    """The LOW LEVEL's own force view off RecordTap (pelvis frame): what it measured, what it applied."""
    meas = run.measured
    if not meas or not meas["t"].size or not np.isfinite(meas["f_meas_ungated"]).any():
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True)
    for r, h in enumerate(HANDS):
        for a in range(3):
            ax = axes[r, a]
            j = 3 * r + a
            first = r == 0 and a == 0
            ax.plot(meas["t"], meas["f_meas_ungated"][:, j], color="C3", lw=0.9,
                    label="f_meas_ungated (low level, pelvis frame)" if first else None)
            if np.isfinite(meas["f_meas"][:, j]).any():
                ax.plot(meas["t"], meas["f_meas"][:, j], color="C4", lw=0.8, alpha=0.8,
                        label="f_meas after the gate (what the servo used)" if first else None)
            if np.isfinite(meas["f_cmd"][:, j]).any():
                ax.plot(meas["t"], meas["f_cmd"][:, j], color="C1", lw=1.0, ls="--",
                        label="f_cmd applied (slewed)" if first else None)
            ax.axhline(0.0, color="0.3", lw=0.5)
            ax.set_title(f"{h.upper()} hand · {FAXES[a]} (pelvis)", fontsize=10)
            ax.grid(True, alpha=0.3)
            if a == 0:
                ax.set_ylabel("N (pelvis frame)")
            if r == 1:
                ax.set_xlabel("s since ▶▶")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"{run.name} — the low level's force view (RecordTap)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def plot_camera_strip(run: Run, out: Path, *, n_cols: int = 8, thumb_w: int = 240) -> Path | None:
    """A contact sheet of the recorded camera frames: one ROW per view (``camera0_rgb`` above
    ``camera1_rgb``), ``n_cols`` frames evenly spaced over the run, each captioned with its time since the
    run began — a glance at what the model saw, for one or two views alike. PIL only (matplotlib's own
    dependency), so it needs nothing beyond the ``viz`` extra. None when the run has no frames; a frame
    that fails to decode is drawn as a grey tile rather than failing the sheet."""
    cam = run.camera
    if not cam or not cam["t"].size:
        return None
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    thumb_h = int(round(thumb_w * 0.75))
    cap_h, pad, label_w = 16, 4, 110
    rows: list[tuple[str, list]] = []
    for key in _view_keys(cam):
        idx = np.where(cam["key"] == key)[0]
        pick = idx[np.linspace(0, idx.size - 1, min(n_cols, idx.size)).round().astype(int)]
        tiles = []
        for i in pick:
            path = run.dir / cam["file"][i]
            try:
                im = Image.open(path).convert("RGB")
                im.thumbnail((thumb_w, thumb_h))
            except Exception:  # noqa: BLE001 - one unreadable frame must not fail the sheet
                im = Image.new("RGB", (thumb_w, thumb_h), (70, 70, 70))
            tiles.append((im, f"+{cam['t'][i]:.2f} s  #{int(cam['seq'][i])}" if cam["seq"][i] >= 0
                          else f"+{cam['t'][i]:.2f} s"))
        rows.append((str(key), tiles))
    n_cols_eff = max(len(t) for _k, t in rows)
    W = label_w + n_cols_eff * (thumb_w + pad) + pad
    H = pad + len(rows) * (thumb_h + cap_h + pad) + 22
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 4), f"{run.name} · camera frames, {run.duration_s:.1f} s", fill=(0, 0, 0))
    y = 22 + pad
    for key, tiles in rows:
        draw.text((pad, y + thumb_h // 2 - 6), key, fill=(0, 0, 0))
        x = label_w
        for im, caption in tiles:
            sheet.paste(im, (x + (thumb_w - im.width) // 2, y + (thumb_h - im.height) // 2))
            draw.text((x + 2, y + thumb_h + 1), caption, fill=(60, 60, 60))
            x += thumb_w + pad
        y += thumb_h + cap_h + pad
    sheet.save(out)
    return out


def process_run(run_dir: Path, *, dpi: int = 110, summary_only: bool = False, log=print) -> dict:
    """Load, summarise, plot (unless ``summary_only``), write ``summary.json`` / ``summary.txt``."""
    run = load_run(run_dir)
    s = summarize(run)
    written = []
    if not summary_only:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            log("plot_rollout_record: matplotlib is not installed — writing the summary only. Install the viz "
                'extra: uv pip install --python .venv/bin/python -e ".[viz]"')
            summary_only = True
    if not summary_only:
        written.append(plot_keypoints(run, run_dir / "keypoints_xyz.png", dpi=dpi))
        written.append(plot_force(run, run_dir / "force_xyz.png", dpi=dpi))
        p = plot_force_lowlevel(run, run_dir / "force_lowlevel.png", dpi=dpi)
        if p is not None:
            written.append(p)
        try:
            p = plot_camera_strip(run, run_dir / "camera_strip.png")
        except Exception as exc:  # noqa: BLE001 - the strip is decoration; the numbers must still land
            log(f"plot_rollout_record: camera strip failed ({exc!r}) — skipped")
            p = None
        if p is not None:
            written.append(p)
    s["plots"] = [p.name for p in written]
    s["plotted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(run_dir / "summary.json", "w") as fh:
        json.dump(s, fh, indent=1, default=_json_default)
    text = summary_text(s)
    with open(run_dir / "summary.txt", "w") as fh:
        fh.write(text + "\n")
    log(text)
    log(f"wrote {', '.join(p.name for p in written) + ', ' if written else ''}summary.json, summary.txt → {run_dir}")
    return s


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return str(o)


def find_runs(root: Path, *, force: bool) -> list[Path]:
    """Every run folder (has ``meta.json``) under ``root``, itself included; without ``--force`` only the
    ones missing an output."""
    if (root / "meta.json").is_file():
        cands = [root]
    else:
        cands = sorted(p.parent for p in root.glob("**/meta.json"))
    if force:
        return cands
    return [c for c in cands if not all((c / o).is_file() for o in OUTPUTS)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="run folder(s) (…/<task>/<stamp>), or with --all any folder above them")
    p.add_argument("--all", action="store_true", help="process every run folder found under each path")
    p.add_argument("--force", action="store_true", help="re-plot runs that already have their outputs")
    p.add_argument("--summary-only", action="store_true", help="numbers only (no matplotlib needed)")
    p.add_argument("--dpi", type=int, default=110)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    targets: list[Path] = []
    for raw in args.paths:
        root = Path(raw).expanduser()
        if not root.exists():
            print(f"plot_rollout_record: {root} does not exist", file=sys.stderr)
            return 2
        if args.all:
            targets.extend(find_runs(root, force=args.force))
        elif (root / "meta.json").is_file():
            targets.append(root)
        else:
            runs = find_runs(root, force=True)
            if not runs:
                print(f"plot_rollout_record: {root} is not a run folder (no meta.json) and holds none", file=sys.stderr)
                return 2
            print(f"plot_rollout_record: {root} is not a run folder; it holds {len(runs)} — pass --all to plot "
                  f"them, or name one: {runs[-1]}", file=sys.stderr)
            return 2
    if not targets:
        print("plot_rollout_record: nothing to do (every run already has its outputs; --force redoes them)")
        return 0
    failed = 0
    for d in targets:
        try:
            process_run(d, dpi=args.dpi, summary_only=args.summary_only)
        except Exception as exc:  # noqa: BLE001 - one bad folder must not stop --all
            failed += 1
            print(f"plot_rollout_record: {d}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
