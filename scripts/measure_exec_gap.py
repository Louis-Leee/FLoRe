#!/usr/bin/env python3
"""Decompose the HL→LL execution gap on RECORDED deploy takes. Robot-free, CPU-only.

WHY THIS EXISTS
---------------
``hl_reach.py`` reports a *time-averaged* standing lead and fits a first-order plant to it.
Both statistics are blind to the one structural question a correction law has to answer first:

    is the shortfall a STANDING OFFSET (→ an integrator is the right tool, and every
    published corrector — HERO, TRACT, Leaky PLAI, APEX — is an integrator or a P term on a
    maintained command), or is it a WITHIN-CHUNK RAMP that the next re-anchor wipes (→ an
    integrator is targeting a quantity that does not survive a chunk boundary, and the right
    object is a stateless step-indexed feedforward)?

An average over a chunk cannot separate those, because step 0 IS the anchor: the lead is ~0
there by construction and grows as the chunk plays out. So this script never averages over
step index — it profiles against it, and measures the drop at the chunk boundary directly.

WHAT IT MEASURES
----------------
M0  frame sanity — is ``anchor_pos`` the same physical quantity as ``measured_tcp``?
M1  ``lead_axis = a·v_axis + b`` with a BLOCK bootstrap over takes. The INTERCEPT ``b`` is the
    lag-vs-offset discriminator; a velocity-lead plant (ExtremControl's model) requires
    ``b ≈ 0`` and ``a = k_d/k_p > 0``.
M1b the near-zero-velocity subset (both commanded and measured speed < 1 cm/s) — a direct read
    of the standing offset with no model at all.
M2  lead vs STEP-WITHIN-CHUNK, split by Koren (1980) into along-commanded-path (lag) and
    normal (geometric) components.
M3  the chunk-boundary drop — what fraction of the accumulated shortfall the re-anchor wipes.
M4  contact vs free space, and ``corr(‖e‖, ‖F_meas‖)`` during contact (the RL2AC hypothesis
    that a learned controller uses tracking error as an implicit force channel).
M5  the realized fraction as a FUNCTION OF TIMESCALE, ``g(T)``, plus the **closure identity**
    ``(1-g_DC) x mean commanded step per STEP INDEX == the independently measured ramp``.
    The single-number estimators of "the" realized fraction disagree by 2x, and the reason is
    not that some are buggy: ``g`` genuinely depends on the window it is measured over, so the
    bare statement "the low level realizes g of each commanded increment" is not well posed.
    Fitting ``g(T) = g_DC·[1 − (tau/T)(1 − e^(−T/tau))]`` separates a DC deficit from a lag.
    Plus the per-axis DC gain, i.e. whether the attenuation is ANISOTROPIC (it is, strongly).
M6  lag vs gain-deficit as competing single-parameter models on identical rows.
M7  the chunk-boundary DISCONTINUITY of the published target, and the divergence bound of
    deleting the re-anchor (pure command integration) at ``eps`` in {inf, 5 cm, 3 cm} — split
    into the anchor lead (which ``eps`` bounds by construction) and the commanded-vs-measured
    gap (which it does not), because the archived file conflated the two.
M8  the force channel by step-within-chunk: the model's wrench PREDICTION error on every take,
    and on the 12 takes that did publish a force, real commanded-vs-measured force TRACKING.

M5/M6/M7 supersede the throwaway scripts whose outputs were archived as ``mechanism.txt`` and
``boundary_and_divergence.txt``. Three of those files' conclusions do not survive being redone
here — see that directory's README.

TWO SERIES, BECAUSE THE PUBLISHED TARGET DIFFERS BY CADENCE
-----------------------------------------------------------
Per prediction (``--series a``): commanded is the ``TemporalEnsembler`` blend when the take ran
the 1-step-per-query cadence (ACT), else ``chunk[1]``; measured is that prediction's own
``anchor_pos``. This is the signal ``hl_reach``'s published lead was measured on.

Per measured tick (``--series b``): commanded is ``decoded_chunk_positions(pred)[step_index]``
using the recorder's own per-tick stamp, measured is ``measured_tcp()[tick]``. **Series A is
meaningless on a chunk-cadence take** — ``chunk[1]`` sits ~0.7 mm from the anchor by
construction, the same trap that made an early version of ``tune_reach_comp`` report 0.07 cm
for takes that really had 2.66 cm. Series B is the only valid one there.

Both series are in TCP terms end to end, which is why M0 is run first: the wire's LEFT contact
point and the measured UMI-TCP are different points on the same rigid body and must never be
differenced (see ``DeployTake.commanded_reference``).

USAGE
-----
    .venv/bin/python scripts/measure_exec_gap.py data/deploy_takes
    .venv/bin/python scripts/measure_exec_gap.py data/deploy_takes --algo diffusion
    .venv/bin/python scripts/measure_exec_gap.py data/deploy_takes --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fuji.s1_policy.serving.hl_take import DeployTake, find_takes  # noqa: E402

#: "Standing" means both the command and the arm are essentially still. At a 36 ms prediction
#: period this is 0.36 mm/tick, tight enough that a velocity lead would have collapsed.
NEAR_ZERO_MPS = 0.01
#: The deploy contact gate (``hl_serving --force-gate-n`` and the low level's own
#: ``FORCE_CMD_WIRE_GATE_N``), so "in contact" here means the same thing it means on the rig.
CONTACT_N = 1.5
#: Step bins. Coarse late bins because a chunk executes ~30 steps, so 30-49 is sparse.
BINS = ((0, 0), (1, 2), (3, 5), (6, 9), (10, 14), (15, 19), (20, 29), (30, 49))
AXES = ("x", "y", "z")


# --------------------------------------------------------------------------- series builders
def series_per_prediction(take: DeployTake, role: str = "left") -> dict | None:
    """Commanded/measured per PREDICTION, reproducing ``tune_reach_comp``'s construction."""
    chunks, meas, t_pred = [], [], []
    for p in take.predictions:
        if p.anchor_pos is None or p.chunk is None:
            continue
        try:
            c = take.decoded_chunk_positions(p, role)
        except Exception:  # noqa: BLE001 — a replay-sourced prediction has no decodable chunk
            continue
        if c.shape[0] < 2:
            continue
        chunks.append(c)
        meas.append(np.asarray(p.anchor_pos[0], dtype=np.float64))
        t_pred.append(float(p.t_predict_s))
    if len(chunks) < 5:
        return None
    n_exec = [int(p.n_exec) for p in take.predictions if p.n_exec is not None]
    ensembled = bool(n_exec) and all(n == 1 for n in n_exec) and len(chunks) > 2
    if ensembled:
        from fuji.s1_policy.model.act.temporal_ensemble import TemporalEnsembler

        horizon = chunks[0].shape[0]
        # Orientation is irrelevant to a POSITION audit, so feed a constant valid rot6d rather
        # than reconstructing per-step rotations the compensator would never read.
        rot = np.tile(np.array([1.0, 0, 0, 0, 1, 0]), (horizon, 1))
        ens = TemporalEnsembler(horizon, k=0.01)
        cmd = np.asarray([ens.add(c, rot)["position"] for c in chunks])
    else:
        cmd = np.asarray([c[1] for c in chunks])
    meas_a = np.asarray(meas)
    t = np.asarray(t_pred)
    v_meas = _central(meas_a, t)
    v_cmd = _central(cmd, t)
    ok = np.isfinite(v_meas).all(1) & np.isfinite(v_cmd).all(1)
    if ok.sum() < 10:
        return None
    return _pack(cmd[ok], meas_a[ok], v_cmd[ok], v_meas[ok],
                 _force_by_fraction(take, len(cmd), role)[ok],
                 step=None, chunk=None,
                 dt=float(np.median(np.diff(t))) if t.size > 1 else np.nan,
                 ensembled=ensembled)


def series_per_tick(take: DeployTake, role: str = "left") -> dict | None:
    """Commanded/measured per measured TICK, carrying the step index and chunk id."""
    pos, _ = take.measured_tcp()
    if pos.shape[0] < 8 or not take.predictions:
        return None
    t = take.times
    by_index = {p.index: p for p in take.predictions}
    decoded: dict[int, np.ndarray] = {}
    anchors: dict[int, np.ndarray] = {}
    cmd = np.full_like(pos, np.nan)
    step = np.full(pos.shape[0], -1, dtype=int)
    chunk = np.full(pos.shape[0], -1, dtype=int)
    w_pred = np.full((pos.shape[0], 6), np.nan)
    for j in range(pos.shape[0]):
        ci, si = take.executed_step_at(j)
        if ci < 0 or si < 0 or ci not in by_index:
            continue
        if ci not in decoded:
            try:
                decoded[ci] = take.decoded_chunk_positions(by_index[ci], role)
            except Exception:  # noqa: BLE001
                decoded[ci] = np.zeros((0, 3))
            anchors[ci] = _anchor_of(by_index[ci], take, role)
        d = decoded[ci]
        if si < d.shape[0]:
            cmd[j] = d[si]
            step[j] = si
            chunk[j] = ci
            pw = take.predicted_wrench(by_index[ci], role)
            if pw is not None and si < pw.shape[0]:
                w_pred[j] = pw[si]
    _, fm = take.servo_force_pelvis(role)
    fn = (np.linalg.norm(fm, axis=1) if fm.shape[0] == pos.shape[0]
          else np.zeros(pos.shape[0]))
    fn = np.where(np.isfinite(fn), fn, 0.0)
    v_cmd = _central(cmd, t)
    ok = np.isfinite(cmd).all(1) & (step >= 0) & np.isfinite(v_cmd).all(1)
    if ok.sum() < 20:
        return None
    out = _pack(cmd[ok], pos[ok], v_cmd[ok], _central(pos, t)[ok], fn[ok],
                step=step[ok], chunk=chunk[ok],
                dt=float(np.median(np.diff(t))), ensembled=False,
                raw=(cmd, pos, step, chunk, ok, fn))
    # Additive keys for M5-M8. Nothing above reads them, so the M0-M4 numbers are unchanged.
    out["anchors"] = anchors
    out["decoded"] = decoded
    out["w_pred"] = w_pred[ok]
    out["w_meas"] = _wrench_at_ticks(take, t)[ok]
    out["path"], out["step_disp"] = _chunk_step_path(decoded, chunk[ok], step[ok])
    # Each row labelled with how many VALID rows its own chunk has, so a report can hold the
    # chunk population fixed while it sweeps the pair window (see gain_vs_window).
    u, c = np.unique(chunk[ok], return_counts=True)
    n_of = dict(zip(u.tolist(), c.tolist()))
    out["chunk_nrows"] = np.array([n_of.get(int(x), 0) for x in chunk])
    out["f_wire"] = _wire_force_at_ticks(take, t)[ok]
    out["wire_force_nonzero"] = bool(np.isfinite(out["f_wire"]).any()
                                     and np.nanmax(np.abs(out["f_wire"])) > 1e-9)
    return out


def _chunk_step_path(decoded: dict[int, np.ndarray], ch: np.ndarray,
                     st: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(path[n], step_disp[m])`` — commanded arc length to each row's STEP, and the per-step
    commanded displacements over the executed range.

    Measured along the DECODED CHUNK at full step resolution, not along the sampled ticks. This
    matters twice over. The measured stream here runs at ~54 ms while a chunk step is 33 ms, so
    consecutive ticks skip ~1.6 step indices: a polyline through the sampled ticks short-cuts the
    commanded arc (biasing the path DOWN and the deficit UP), and a "displacement per step"
    averaged over tick pairs is really per 1.6 steps (biasing the closure identity's lever arm UP
    by the same factor). The ramp M2 measures is per STEP INDEX, so the closure has to be too.
    """
    cum: dict[int, np.ndarray] = {}
    for ci, d in decoded.items():
        if d.shape[0] < 2:
            cum[ci] = np.zeros(max(d.shape[0], 1))
            continue
        cum[ci] = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(d, axis=0), axis=1))]
    path = np.zeros(ch.shape[0])
    for i in range(ch.shape[0]):
        c = cum.get(int(ch[i]))
        if c is not None and 0 <= st[i] < c.shape[0]:
            path[i] = c[st[i]]
    disp: list[np.ndarray] = []
    for ci, d in decoded.items():
        m = ch == ci
        if not m.any() or d.shape[0] < 2:
            continue
        # Only the steps that were actually executed: an interrupted chunk's tail was never asked
        # for, so including it would average in displacements the arm never saw.
        top = int(st[m].max())
        if top >= 1:
            disp.append(np.linalg.norm(np.diff(d[:top + 1], axis=0), axis=1))
    return path, (np.concatenate(disp) if disp else np.zeros(0))


def _wire_force_at_ticks(take: DeployTake, t: np.ndarray) -> np.ndarray:
    """``(N,3)`` the force PUBLISHED on ``:8770``, resampled to the tick clock and re-expressed
    in the MEASURED sign so it may be differenced against ``measured_wrench_sensor``.

    The stored wire value is already negated (``F_cmd = -F_meas``, SENSOR frame), so undoing that
    flip is what puts the two streams in one convention. Comparing them without it would report
    twice the force as an error.
    """
    from fuji.utils.force_frame import CMD_FROM_MEAS_SIGN

    tc, fc = take.commanded_force_sensor()
    if fc.shape[0] < 2 or t.size == 0:
        return np.full((t.size, 3), np.nan)
    idx = np.clip(np.searchsorted(tc, t, side="right") - 1, 0, tc.shape[0] - 1)
    return fc[idx] / CMD_FROM_MEAS_SIGN


def _anchor_of(pred, take: DeployTake, role: str) -> np.ndarray:
    """The chunk's own stored anchor position for ``role`` — step 0 of its decoded path."""
    try:
        ri = take._layout().roles.index(role)  # noqa: SLF001 — read-only, same package contract
        return np.asarray(pred.anchor_pos[ri], dtype=np.float64)
    except Exception:  # noqa: BLE001
        return np.full(3, np.nan)


def _wrench_at_ticks(take: DeployTake, t: np.ndarray) -> np.ndarray:
    """``(N,6)`` raw Bota wrench resampled nearest-at-or-before onto the measured tick clock.

    SENSOR frame, MEASURED sign — the same frame and sign :meth:`DeployTake.predicted_wrench`
    returns, which is the only pairing that may be differenced. ``servo_force_pelvis`` is
    pelvis-frame and must never be compared against a sensor-frame prediction.
    """
    tw, wr = take.measured_wrench_sensor()
    if wr.shape[0] < 2 or t.size == 0:
        return np.full((t.size, 6), np.nan)
    idx = np.clip(np.searchsorted(tw, t, side="right") - 1, 0, tw.shape[0] - 1)
    return wr[idx]


def _central(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    v = np.full_like(x, np.nan)
    if x.shape[0] < 3:
        return v
    dt = t[2:] - t[:-2]
    good = dt > 1e-6
    v[1:-1][good] = (x[2:][good] - x[:-2][good]) / dt[good, None]
    return v


def _force_by_fraction(take: DeployTake, n: int, role: str) -> np.ndarray:
    """Contact magnitude resampled onto the prediction count by fraction-of-take.

    The measured rows and the predictions share a clock but not a rate; both are ordered, so
    align by position rather than inventing a join (same choice ``tune_reach_comp`` makes).
    """
    _, fm = take.servo_force_pelvis(role)
    if fm.shape[0] < 2 or n == 0:
        return np.zeros(n)
    mag = np.linalg.norm(fm, axis=1)
    mag = np.where(np.isfinite(mag), mag, 0.0)
    idx = np.clip((np.linspace(0.0, 1.0, n) * (len(mag) - 1)).astype(int), 0, len(mag) - 1)
    return mag[idx]


def _pack(cmd, meas, v_cmd, v_meas, fn, *, step, chunk, dt, ensembled, raw=None) -> dict:
    lead = cmd - meas
    tang = v_cmd / np.maximum(np.linalg.norm(v_cmd, axis=1, keepdims=True), 1e-9)
    e_par = np.einsum("ij,ij->i", lead, tang)
    return dict(lead=lead, v_meas=v_meas, v_cmd=v_cmd, fn=fn, step=step, chunk=chunk,
                e_par=e_par, e_perp=np.linalg.norm(lead - e_par[:, None] * tang, axis=1),
                nrm=np.linalg.norm(lead, axis=1),
                sp=np.linalg.norm(v_meas, axis=1), sp_cmd=np.linalg.norm(v_cmd, axis=1),
                dt=dt, ensembled=ensembled, raw=raw)


# ------------------------------------------------------------------------------- statistics
def block_bootstrap(fn, groups, n_boot: int, rng) -> tuple:
    """Resample TAKES, never samples: ticks inside a take are serially correlated, so a
    sample-level CI on 20k ticks would be a fiction two orders of magnitude too tight."""
    base = fn(groups)
    if base is None:
        return None, None, None
    k, out = len(groups), []
    for _ in range(n_boot):
        v = fn([groups[i] for i in rng.integers(0, k, k)])
        if v is not None:
            out.append(v)
    if not out:
        return base, None, None
    arr = np.asarray(out)
    return base, np.nanpercentile(arr, 2.5, axis=0), np.nanpercentile(arr, 97.5, axis=0)


def fit_lead_vs_velocity(groups, axis: int):
    y = np.concatenate([g["lead"][:, axis] for g in groups])
    x = np.concatenate([g["v_meas"][:, axis] for g in groups])
    if y.size < 10 or np.ptp(x) < 1e-6:
        return None
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_res = float(np.sum((y - A @ coef) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return np.array([coef[0], coef[1], 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan])


def standing_lead(groups):
    parts, n = [], 0
    for g in groups:
        m = (g["sp"] < NEAR_ZERO_MPS) & (g["sp_cmd"] < NEAR_ZERO_MPS)
        if m.any():
            parts.append(g["lead"][m])
            n += int(m.sum())
    if not parts:
        return None
    a = np.concatenate(parts)
    return np.array([*a.mean(axis=0), np.linalg.norm(a, axis=1).mean(), n])


def step_profile(groups, key: str, regime: str | None = None):
    out = []
    for lo, hi in BINS:
        vals = []
        for g in groups:
            if g["step"] is None:
                continue
            m = (g["step"] >= lo) & (g["step"] <= hi)
            if regime == "contact":
                m &= g["fn"] >= CONTACT_N
            elif regime == "free":
                m &= g["fn"] < CONTACT_N
            if m.any():
                vals.append(np.abs(g[key][m]) if key == "e_par" else g[key][m])
        out.append(np.concatenate(vals).mean() if vals else np.nan)
    return np.asarray(out)


def axis_step_profile(groups, axis: int):
    out = []
    for lo, hi in BINS:
        vals = [g["lead"][(g["step"] >= lo) & (g["step"] <= hi), axis]
                for g in groups if g["step"] is not None
                and ((g["step"] >= lo) & (g["step"] <= hi)).any()]
        out.append(np.concatenate(vals).mean() if vals else np.nan)
    return np.asarray(out)


def boundary_drop(groups):
    """(before, after, fraction_wiped, n) across chunk transitions.

    Only transitions where the OLD chunk had reached step >= 3 and the NEW one is at step <= 1
    count, so "before" is a genuinely accumulated shortfall and "after" is a fresh anchor.
    """
    before, after = [], []
    for g in groups:
        if g["raw"] is None:
            continue
        cmd, pos, step, chunk, ok, _ = g["raw"]
        idx = np.where(ok)[0]
        if idx.size < 2:
            continue
        nrm = np.linalg.norm(cmd - pos, axis=1)
        for a, b in zip(idx[:-1], idx[1:]):
            if chunk[b] != chunk[a] and step[a] >= 3 and step[b] <= 1:
                before.append(nrm[a])
                after.append(nrm[b])
    if len(before) < 20:
        return None
    bef, aft = np.asarray(before), np.asarray(after)
    return np.array([bef.mean(), aft.mean(), 1.0 - aft.mean() / bef.mean(), len(bef)])


def contact_corr(groups):
    e, f = [], []
    for g in groups:
        m = g["fn"] >= CONTACT_N
        if m.sum() > 5:
            e.append(g["nrm"][m])
            f.append(g["fn"][m])
    if not e:
        return None
    e, f = np.concatenate(e), np.concatenate(f)
    if e.size < 30 or e.std() < 1e-9 or f.std() < 1e-9:
        return None
    return np.array([float(np.corrcoef(e, f)[0, 1]), e.size, e.mean(), f.mean()])


# ------------------------------------------- M5-M8: the mechanism, in the INCREMENT domain
#: Pair separations (in measured ticks) for the within-chunk increment estimator. A pure GAIN
#: gives the same answer at every ``w``; a LAG's apparent gain RISES with ``w``, because a fixed
#: time lag is a shrinking fraction of a longer window. So this sweep is itself the lag-vs-gain
#: discriminator and needs no model comparison to read — which is how it turned out that BOTH
#: are present and the archived "a gain deficit, not a lag" was a false dichotomy.
PAIR_WINDOWS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 25, 30)
#: Minimum valid rows for a chunk to enter the FIXED-population sweep. Without this the sweep is
#: confounded: only long chunks can supply a long pair, and long chunks are the ones that were
#: never interrupted, which is itself correlated with how hard the command was pushing. Pooling
#: over all chunks reads g(54 ms) = 0.32; on a fixed population the same window reads 0.44.
LONG_CHUNK_ROWS = 31
#: A window past the knee of g(T), i.e. where the lag has washed out and what is left is the DC
#: gain. This is the timescale a within-chunk pre-distortion actually acts on.
CHUNK_SCALE_W = 16
#: Displacement floor for ratio-style estimators (0.5 mm). Below it the denominator is
#: comparable to the measurement noise and the ratio reports rectification, not the plant.
MIN_DISP_M = 0.0005


def within_chunk_pairs(g: dict, w: int,
                       min_rows: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(dcmd[n,3], dmeas[n,3], f[n], k_hi[n])`` for row pairs ``(i, i+w)`` inside ONE chunk.

    Differencing INSIDE a chunk is what makes this estimator defensible. Every row of a chunk is
    decoded against that chunk's single stored anchor, so the anchor's own measurement noise is a
    common offset that cancels EXACTLY in ``dcmd``. What is left as the regressor is the
    published command, which is noise-free — so ordinary least squares is unbiased here, and the
    errors-in-variables attenuation that drags a noisy-regressor slope toward zero does not
    apply. Nor does the rectification bias that inflates a magnitude ratio, because the fit is
    on signed vectors rather than on norms.
    """
    cmd, pos, step, chunk, ok, fn = g["raw"]
    n = cmd.shape[0]
    empty = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int))
    if n <= w:
        return empty
    lo = np.arange(0, n - w)
    hi = lo + w
    m = ok[lo] & ok[hi] & (chunk[lo] >= 0) & (chunk[lo] == chunk[hi])
    if min_rows > 0:
        nr = g.get("chunk_nrows")
        if nr is None:
            return empty
        m &= nr[lo] >= min_rows
    if not m.any():
        return empty
    lo, hi = lo[m], hi[m]
    return cmd[hi] - cmd[lo], pos[hi] - pos[lo], 0.5 * (fn[lo] + fn[hi]), step[hi]


def _stack_pairs(groups, w: int, min_rows: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dc, dm, ff = [], [], []
    for g in groups:
        if g.get("raw") is None:
            continue
        a, b, f, _ = within_chunk_pairs(g, w, min_rows)
        if a.shape[0]:
            dc.append(a)
            dm.append(b)
            ff.append(f)
    if not dc:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(dc), np.concatenate(dm), np.concatenate(ff)


def gain_increment(groups, w: int = 1, axis: int | None = None, min_rows: int = 0):
    """``[g, R², n]`` — least-squares increment gain through the origin, within chunks.

    ``axis=None`` is the isotropic fit over all three axes at once,
    ``g = <dcmd, dmeas> / <dcmd, dcmd>``. Signed, increment-domain, noise-free regressor — but
    the answer DEPENDS ON ``w``, so a bare "the realized fraction is g" is not a well-posed
    statement. See :func:`gain_vs_window`.
    """
    a, b, _ = _stack_pairs(groups, w, min_rows)
    if a.shape[0] == 0:
        return None
    if axis is None:
        keep = np.linalg.norm(a, axis=1) >= MIN_DISP_M
        x, y = a[keep].ravel(), b[keep].ravel()
    else:
        keep = np.abs(a[:, axis]) >= MIN_DISP_M
        x, y = a[keep, axis], b[keep, axis]
    if x.size < 30 or float(x @ x) <= 0:
        return None
    gg = float(x @ y) / float(x @ x)
    res = y - gg * x
    ss = float(np.sum((y - y.mean()) ** 2))
    return np.array([gg, 1.0 - float(res @ res) / ss if ss > 0 else np.nan, x.size])


def gain_vs_window(groups, min_rows: int = LONG_CHUNK_ROWS):
    """``[g(w) for w in PAIR_WINDOWS]`` on a FIXED chunk population — the gain-vs-timescale curve.

    Holding the population fixed is what makes the sweep interpretable. Pooling over all chunks
    conflates two effects that push the same way: longer windows wash out the lag (real dynamics),
    AND only long chunks can supply a long pair while long chunks are the uninterrupted ones
    (selection). The pooled curve therefore overstates the rise.
    """
    out = []
    for w in PAIR_WINDOWS:
        r = gain_increment(groups, w, None, min_rows)
        out.append(np.nan if r is None else r[0])
    return np.asarray(out)


def _lag_shape(t: np.ndarray, tau: float) -> np.ndarray:
    """Fraction of a RAMP input a first-order lag realizes over a window of length ``t``."""
    tau = max(float(tau), 1e-9)
    return 1.0 - (tau / t) * (1.0 - np.exp(-t / tau))


def fit_gain_lag(groups, dt: float, min_rows: int = LONG_CHUNK_ROWS):
    """``[g_dc, tau_s, rms, rms_flat]`` — fit ``g(T) = g_dc · [1 − (tau/T)(1 − e^(−T/tau))]``.

    Two parameters against the whole gain-vs-timescale curve, which is the smallest model that
    can hold a DC deficit and a lag at once. ``tau`` is scanned on a log grid and ``g_dc`` solved
    in closed form at each ``tau`` (it enters linearly), so this cannot fail to converge and does
    not need an optimiser. ``rms_flat`` is the same curve's residual under a single constant gain,
    which is the model this measurement replaces.
    """
    g = gain_vs_window(groups, min_rows)
    t = np.asarray(PAIR_WINDOWS, dtype=np.float64) * float(dt)
    m = np.isfinite(g)
    if m.sum() < 4:
        return None
    g, t = g[m], t[m]
    best = None
    for tau in np.geomspace(1e-3, 2.0, 400):
        h = _lag_shape(t, tau)
        denom = float(h @ h)
        if denom <= 0:
            continue
        gdc = float(h @ g) / denom
        sse = float(np.sum((g - gdc * h) ** 2))
        if best is None or sse < best[0]:
            best = (sse, gdc, tau)
    if best is None:
        return None
    sse, gdc, tau = best
    return np.array([gdc, tau, np.sqrt(sse / g.size), float(np.std(g))])


def gain_pertick_speed(groups):
    """``[G, n]`` — estimator **E4**: ‖v_meas‖ on ‖v_cmd‖ through the origin, per tick.

    Kept only so the estimator table can show what it does. It is a ratio of RECTIFIED
    magnitudes, so it is biased upward by ``E|eps| = sigma*sqrt(2/pi)`` wherever the true
    increment is small, and its regressor carries the measured stream's noise, which attenuates
    the slope. The two biases fight; the net is not predictable a priori.
    """
    x = np.concatenate([g["sp_cmd"] for g in groups])
    y = np.concatenate([g["sp"] for g in groups])
    m = np.isfinite(x) & np.isfinite(y) & (x > 0.005)
    if m.sum() < 30:
        return None
    x, y = x[m], y[m]
    return np.array([float(x @ y) / float(x @ x), x.size])


def gain_path_ratio(groups):
    """``[median(lead/path), n]`` — estimator **E1**, the raw magnitude ratio, path > 5 mm."""
    r = []
    for g in groups:
        m = g["path"] > 0.005
        if m.any():
            r.append(g["nrm"][m] / g["path"][m])
        continue
    if not r:
        return None
    a = np.concatenate(r)
    return np.array([float(np.median(a)), a.size])


def ramp_slope(groups):
    """``[mm/step]`` growth of ‖lead‖ over step bins with centre > 0 — the independent ramp the
    closure identity has to reproduce."""
    v = step_profile(groups, "nrm")
    centres = np.array([(a + b) / 2 for a, b in BINS])
    ok = np.isfinite(v) & (centres > 0)
    if ok.sum() < 3:
        return None
    return np.array([float(np.polyfit(centres[ok], v[ok], 1)[0])])


def mean_step_disp(groups):
    """``[m]`` mean commanded displacement per CHUNK STEP — the closure identity's lever arm.

    Per step index, measured on the decoded chunk, because that is the axis M2's ramp is
    expressed in. Averaging over tick pairs instead would inflate this by the ~1.6 step indices
    a single 54 ms tick advances, and the closure would then "fail" for a unit-mismatch reason.
    """
    d = [g["step_disp"] for g in groups if g.get("step_disp") is not None and g["step_disp"].size]
    if not d:
        return None
    return np.array([float(np.concatenate(d).mean())])


def model_compare(groups):
    """``[R²_lag, R²_gain, R²_both, tau_joint, deficit_joint, deficit_gain, n]``.

    LAG   ``‖lead‖ ~ tau·‖v_cmd‖``            — a velocity lead (ExtremControl's plant).
    GAIN  ``‖lead‖ ~ (1-g)·path_since_anchor`` — an unrealized fraction of what was asked.
    Both through the origin, R² about the mean, on identical rows.
    """
    y = np.concatenate([g["nrm"] for g in groups])
    xg = np.concatenate([g["path"] for g in groups])
    xl = np.concatenate([g["sp_cmd"] for g in groups])
    m = np.isfinite(y) & np.isfinite(xg) & np.isfinite(xl)
    if m.sum() < 100:
        return None
    y, xg, xl = y[m], xg[m], xl[m]
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return None

    def _r2(A):
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return 1.0 - float(np.sum((y - A @ coef) ** 2)) / ss_tot, coef

    r2_lag, _ = _r2(xl[:, None])
    r2_gain, c_gain = _r2(xg[:, None])
    r2_both, c_both = _r2(np.column_stack([xl, xg]))
    return np.array([r2_lag, r2_gain, r2_both, c_both[0], c_both[1], c_gain[0], y.size])


def boundary_jump(groups):
    """``[mean_jump, p95_jump, mean_in_chunk_step, spike_mps, n]`` for the PUBLISHED target.

    Same transition filter as M3 (old chunk at step >= 3, new one at step <= 1) so the count is
    comparable. The jump is a discontinuity in what the wire carries, independent of how the arm
    responds to it.
    """
    jumps, in_chunk, dts = [], [], []
    for g in groups:
        if g.get("raw") is None:
            continue
        cmd, _, step, chunk, ok, _ = g["raw"]
        idx = np.where(ok)[0]
        if idx.size < 2:
            continue
        dts.append(g["dt"])
        for a, b in zip(idx[:-1], idx[1:]):
            d = float(np.linalg.norm(cmd[b] - cmd[a]))
            if chunk[b] != chunk[a]:
                if step[a] >= 3 and step[b] <= 1:
                    jumps.append(d)
            else:
                in_chunk.append(d)
    if len(jumps) < 20:
        return None
    j = np.asarray(jumps)
    dt = float(np.nanmedian(dts)) if dts else np.nan
    return np.array([j.mean(), float(np.percentile(j, 95)),
                     float(np.mean(in_chunk)) if in_chunk else np.nan,
                     j.mean() / dt if dt and np.isfinite(dt) else np.nan, j.size])


def divergence_replay(groups, eps: float | None):
    """``[end_gap, mean_peak_gap, p95_peak_gap, mean_peak_lead, n_takes]``, metres.

    Replays PURE COMMAND INTEGRATION: each chunk is anchored on the previous chunk's own final
    published target instead of on the measured pose — i.e. the re-anchor amnesty is deleted —
    with the carried anchor clamped to within ``eps`` of the measured pose exactly as
    ``AnchorCarry.resolve`` does it (clamp on ``‖anchor − measured‖``, written back into the
    carried state, ``None`` = no clamp).

    TWO quantities, because they differ and the archived ``boundary_and_divergence.txt`` conflated
    them. The **anchor lead** is sampled AT THE RE-ANCHOR, immediately after the clamp, because
    that is the only instant ``resolve`` is called and therefore the only instant ``eps`` binds —
    it saturates at ``eps`` by construction, so reporting it as a finding is circular. Between two
    re-anchors the arm keeps moving while the carried anchor is held, so the same difference
    measured mid-chunk legitimately exceeds ``eps`` and is NOT what the clamp promises. The
    **commanded-vs-measured gap** is that lead plus the chunk's own excursion, so it exceeds
    ``eps`` by design and is the quantity an operator actually cares about.

    This bounds how much FURTHER the arm would have been ASKED to go. It says nothing about how
    much closer it would have GOT: the arm's response to a larger ask is not in the recording.
    """
    ends, peak_gap, peak_lead = [], [], []
    for g in groups:
        if g.get("raw") is None or not g.get("anchors"):
            continue
        cmd, pos, _, chunk, ok, _ = g["raw"]
        rows = np.where(ok)[0]
        if rows.size < 5:
            continue
        anch = g["anchors"]
        first = int(chunk[rows[0]])
        if first not in anch or not np.isfinite(anch[first]).all():
            continue
        virt = np.asarray(anch[first], dtype=np.float64).copy()
        gaps, leads, prev_row, prev_target = [], [], -1, None
        for r in rows:
            ci = int(chunk[r])
            if ci not in anch or not np.isfinite(anch[ci]).all():
                continue
            if prev_row >= 0 and ci != int(chunk[prev_row]):
                virt = prev_target.copy()
                if eps is not None:
                    d = virt - pos[r]
                    n = float(np.linalg.norm(d))
                    if n > eps > 0.0:
                        virt = pos[r] + d * (eps / n)
                # Sampled HERE and only here: this is the re-anchor instant, the one moment the
                # real AnchorCarry.resolve runs and the clamp is in force.
                leads.append(float(np.linalg.norm(virt - pos[r])))
            target = virt + (cmd[r] - anch[ci])
            gaps.append(float(np.linalg.norm(target - pos[r])))
            prev_row, prev_target = r, target
        if len(gaps) > 5 and leads:
            ends.append(gaps[-1])
            peak_gap.append(max(gaps))
            peak_lead.append(max(leads))
    if len(ends) < 5:
        return None
    p = np.asarray(peak_gap)
    return np.array([float(np.mean(ends)), float(p.mean()), float(np.percentile(p, 95)),
                     float(np.mean(peak_lead)), p.size])


def wire_force_profile(groups, norm_of: str = "err"):
    """Mean ‖F_wire − F_meas‖ (N) by step bin over the takes that actually published a force.

    This IS a tracking comparison — commanded on the wire against what the sensor then read —
    and it is the force-side twin of M2. Both streams are put in the MEASURED sign by
    :func:`_wire_force_at_ticks`; differencing them raw would double the apparent error.
    """
    sub = [g for g in groups if g.get("wire_force_nonzero")]
    if not sub:
        return None
    out = []
    for lo, hi in BINS:
        vals = []
        for g in sub:
            fw, fm = g["f_wire"], g["w_meas"][:, 0:3]
            m = ((g["step"] >= lo) & (g["step"] <= hi) & np.isfinite(fw).all(1)
                 & np.isfinite(fm).all(1) & (np.linalg.norm(fw, axis=1) > 1e-9))
            if m.any():
                src = {"err": fw[m] - fm[m], "cmd": fw[m], "meas": fm[m]}[norm_of]
                vals.append(np.linalg.norm(src, axis=1))
        out.append(np.concatenate(vals).mean() if vals else np.nan)
    return np.asarray(out)


def wire_force_gain(groups):
    """``[g_F, R², n]`` — realized fraction of the COMMANDED force, ‖F_meas‖ on ‖F_wire‖.

    The force-side analogue of M5's position gain, on the gated ticks only. Reported as a
    magnitude regression because the wire carries only the 3-DoF linear force, so there is no
    torque channel to close the vector fit with.
    """
    sub = [g for g in groups if g.get("wire_force_nonzero")]
    if not sub:
        return None
    xs, ys = [], []
    for g in sub:
        fw, fm = g["f_wire"], g["w_meas"][:, 0:3]
        m = np.isfinite(fw).all(1) & np.isfinite(fm).all(1) & (np.linalg.norm(fw, axis=1) > CONTACT_N)
        if m.any():
            xs.append(fw[m].ravel())
            ys.append(fm[m].ravel())
    if not xs:
        return None
    x, y = np.concatenate(xs), np.concatenate(ys)
    if x.size < 30 or float(x @ x) <= 0:
        return None
    gg = float(x @ y) / float(x @ x)
    res = y - gg * x
    ss = float(np.sum((y - y.mean()) ** 2))
    return np.array([gg, 1.0 - float(res @ res) / ss if ss > 0 else np.nan, x.size])


def wrench_err_profile(groups, comp: slice, norm_of: str = "err"):
    """Mean ‖predicted − measured‖ wrench by step bin, over ``comp`` (force or torque rows).

    NOTE ON SEMANTICS. This is a PREDICTION error, not a tracking error. ``predicted_wrench`` is
    the model's guess at what the sensor would read, and the wire never carried a force command
    in this corpus, so there is no commanded-vs-measured force comparison to be had. What this
    profile answers is narrower and still useful: does the FORCE channel's error ramp with step
    index the way the POSITION channel's does? If it does not, the position ramp is specifically
    an execution phenomenon rather than the model losing coherence late in a chunk.
    """
    out = []
    for lo, hi in BINS:
        vals = []
        for g in groups:
            if g.get("w_pred") is None or g["step"] is None:
                continue
            wp, wm = g["w_pred"][:, comp], g["w_meas"][:, comp]
            m = (g["step"] >= lo) & (g["step"] <= hi) & np.isfinite(wp).all(1) & np.isfinite(wm).all(1)
            if m.any():
                src = {"err": wp[m] - wm[m], "pred": wp[m], "meas": wm[m]}[norm_of]
                vals.append(np.linalg.norm(src, axis=1))
        out.append(np.concatenate(vals).mean() if vals else np.nan)
    return np.asarray(out)


# ------------------------------------------------------------------------------------ report
def _ci(base, lo, hi, i, scale=1000.0, w=6, p=2):
    if base is None or not np.isfinite(base[i]):
        return "--"
    if lo is None:
        return f"{scale*base[i]:{w}.{p}f}"
    return f"{scale*base[i]:{w}.{p}f} [{scale*lo[i]:.{p}f},{scale*hi[i]:.{p}f}]"


def _closure(groups, est):
    """Predicted ramp ``(1-g) x mean commanded step`` for one estimator, in metres/step.

    Bootstrapped as ONE statistic so the gain and the lever arm are resampled together — they
    come from the same takes and are correlated, so multiplying two independent CIs would be
    wrong in both directions.
    """
    e, d = est(groups), mean_step_disp(groups)
    if e is None or d is None:
        return None
    return np.array([(1.0 - e[0]) * d[0]])


def report_mechanism(groups: list[dict], *, n_boot: int, rng) -> dict:
    """M5-M8: which estimator of the realized fraction survives its own closure identity, the
    lag-vs-gain comparison, the boundary discontinuity + divergence bound, and the force channel.
    """
    out: dict = {}
    ramp = block_bootstrap(ramp_slope, groups, n_boot, rng)
    disp = block_bootstrap(mean_step_disp, groups, n_boot, rng)

    dt = float(np.nanmedian([g["dt"] for g in groups]))
    print("\nM5  THE REALIZED FRACTION IS FREQUENCY-DEPENDENT — one number is not well posed")
    print("    Closure identity, and it needs no model: the ramp M2 measures per STEP INDEX must")
    print("    equal the unrealized part of the commanded displacement per step.")
    print(f"      ramp {_ci(*ramp, 0, 1000.0, 5, 3)} mm/step  /  step "
          f"{_ci(*disp, 0, 1000.0, 5, 3)} mm")
    cl = block_bootstrap(lambda gg: (lambda r, d: None if r is None or d is None
                                     else np.array([r[0] / d[0]]))(ramp_slope(gg), mean_step_disp(gg)),
                         groups, n_boot, rng)
    if cl[0] is not None:
        print(f"      =>  DC deficit (1-g) = {_ci(*cl, 0, 1.0, 6, 4)}"
              f"   =>  g_DC = {1.0 - cl[0][0]:.4f}")
        out["m5_closure"] = dict(deficit=cl[0][0], g_dc=1.0 - cl[0][0],
                                 ci=[cl[1][0], cl[2][0]] if cl[1] is not None else None)

    print("\n    g(T) from within-chunk increments over a window T, on a FIXED population of")
    print(f"    chunks with >= {LONG_CHUNK_ROWS} valid rows, so every T sees the SAME chunks:")
    curve = block_bootstrap(lambda gg: gain_vs_window(gg, LONG_CHUNK_ROWS), groups, n_boot, rng)
    pooled = block_bootstrap(lambda gg: gain_vs_window(gg, 0), groups, n_boot, rng)
    print(f"      {'w':>4}{'T (ms)':>9}{'g  fixed population':>28}{'g pooled':>12}")
    out["m5_curve"] = []
    for i, w in enumerate(PAIR_WINDOWS):
        if curve[0] is None or not np.isfinite(curve[0][i]):
            continue
        print(f"      {w:>4}{1000 * w * dt:>9.0f}{_ci(*curve, i, 1.0, 6, 4):>28}"
              f"{pooled[0][i]:>12.4f}")
        out["m5_curve"].append(dict(w=w, t_ms=1000 * w * dt, g_fixed=curve[0][i],
                                    g_pooled=pooled[0][i]))
    print("    The pooled column is CONFOUNDED and must not be read as the same curve: only long")
    print("    chunks can supply a long pair, and a chunk is long precisely when nothing")
    print("    interrupted it. Short chunks realize a LOWER fraction, so pooling exaggerates the")
    print("    rise. Both effects are real; they are not the same effect.")

    fit = block_bootstrap(lambda gg: fit_gain_lag(gg, dt, LONG_CHUNK_ROWS), groups, n_boot, rng)
    if fit[0] is not None:
        print("\n    fit  g(T) = g_DC · [1 − (tau/T)(1 − exp(−T/tau))]")
        print(f"      g_DC = {_ci(*fit, 0, 1.0, 6, 4)}     tau = {_ci(*fit, 1, 1000.0, 5, 1)} ms")
        print(f"      RMS residual {fit[0][2]:.4f}  vs  {fit[0][3]:.4f} for a single constant gain")
        print("    => the mechanism is a DC GAIN DEFICIT *AND* A LAG. Measuring at one timescale")
        print("       forces all of it onto whichever of the two you happened to test, which is")
        print("       how 'a gain deficit, not a lag' and 'tau straddles zero' were both reached.")
        gdc = fit[0][0]
        per_tick = gain_increment(groups, 1, None, 0)
        print("\n    CONSEQUENCE FOR A CORRECTOR — invert the gain at the timescale you act on:")
        print(f"      within-chunk displacement (~1 s)   1/{gdc:.3f} = {1 / gdc:.3f}x   <- defensible")
        if per_tick is not None:
            pt = per_tick[0]
            print(f"      per published tick ({1000 * dt:.0f} ms)        1/{pt:.3f} = {1 / pt:.3f}x"
                  f"   <- over-drives by {(1 / pt) / (1 / gdc):.2f}x")
            print("       and it injects exactly the high-frequency content this curve shows the")
            print("       plant rejects, so the extra ask converts to chatter, not to tracking.")
        out["m5_fit"] = dict(g_dc=gdc, tau_s=fit[0][1], rms=fit[0][2], rms_flat=fit[0][3],
                             correction_chunk_scale=1.0 / gdc,
                             correction_per_tick=1.0 / per_tick[0] if per_tick is not None else None)

    print("\n    For the record, what the single-number estimators report on the same takes —")
    print("    a 2x spread that is now EXPLAINED by the curve above rather than adjudicated:")
    print(f"    {'estimator':<44}{'g':>20}{'predicted ramp (mm/step)':>30}{'closes?':>9}")
    def _e1(g):
        r = gain_path_ratio(g)
        return None if r is None else np.array([1.0 - r[0], r[1]])

    def _e3(g):
        r = model_compare(g)
        return None if r is None else np.array([1.0 - r[5], r[6]])

    ests: list[tuple[str, object]] = [
        ("E1  median(‖lead‖ / path), path>5mm", _e1),
        ("E3  ‖lead‖ ~ (1-g)·path, regression", _e3),
        ("E4  ‖v_meas‖ on ‖v_cmd‖, per tick", gain_pertick_speed),
    ]
    for w in (1, CHUNK_SCALE_W):
        ests.append((f"E5  increment, within-chunk, w={w:<2d}",
                     lambda g, ww=w: gain_increment(g, ww, None, LONG_CHUNK_ROWS)))
    out["m5"] = []
    for name, est in ests:
        base, lo, hi = block_bootstrap(est, groups, n_boot, rng)
        cb, cl, ch = block_bootstrap(lambda g, e=est: _closure(g, e), groups, n_boot, rng)
        if base is None or cb is None:
            continue
        closes = "--"
        if ramp[0] is not None and cl is not None:
            closes = "YES" if cl[0] <= ramp[0][0] <= ch[0] else "no"
        print(f"    {name:<44}{_ci(base, lo, hi, 0, 1.0, 6, 3):>20}"
              f"{_ci(cb, cl, ch, 0, 1000.0, 6, 3):>30}{closes:>9}")
        out["m5"].append(dict(estimator=name.strip(), g=base[0],
                              predicted_ramp_mm=1000 * cb[0], closes=closes))

    print(f"\n    per-axis DC gain — ANISOTROPY. At the CHUNK timescale (w={CHUNK_SCALE_W},"
          f" {1000 * CHUNK_SCALE_W * dt:.0f} ms) on the")
    print("    fixed population, because a per-tick per-axis gain would mix in the lag.")
    ax = {}
    for i, nm in enumerate(AXES):
        ax[nm] = block_bootstrap(lambda g, a=i: gain_increment(g, CHUNK_SCALE_W, a, LONG_CHUNK_ROWS),
                                 groups, n_boot, rng)
        if ax[nm][0] is not None:
            print(f"      g_{nm} = {_ci(*ax[nm], 0, 1.0, 6, 4)}   n = {int(ax[nm][0][2])}")
    out["m5_per_axis"] = {k: (v[0][0] if v[0] is not None else None) for k, v in ax.items()}
    print("    An ANISOTROPIC gain bends the realized path off the commanded one even when the")
    print("    command is straight, so part of M2's |e_normal| is predicted by this alone —")
    print("    something a scalar 1/g provably cannot correct. Tested in exec_gap_corrector.py.")

    base, lo, hi = block_bootstrap(model_compare, groups, n_boot, rng)
    if base is not None:
        print(f"\nM6  LAG vs GAIN on identical rows (n = {int(base[6])})")
        print(f"      LAG   ‖lead‖ ~ tau·‖v_cmd‖             R² = {_ci(base, lo, hi, 0, 1.0, 5, 3)}")
        print(f"      GAIN  ‖lead‖ ~ (1-g)·path_since_anchor R² = {_ci(base, lo, hi, 1, 1.0, 5, 3)}")
        print(f"      BOTH                                   R² = {_ci(base, lo, hi, 2, 1.0, 5, 3)}")
        print(f"      joint fit: tau = {_ci(base, lo, hi, 3, 1.0, 6, 4)} s"
              f"   (1-g) = {_ci(base, lo, hi, 4, 1.0, 6, 4)}")
        print(f"      gain-only: (1-g) = {_ci(base, lo, hi, 5, 1.0, 6, 4)}"
              f"  ->  g = {1.0 - base[5]:.3f}")
        out["m6"] = dict(r2_lag=base[0], r2_gain=base[1], r2_both=base[2],
                         tau_joint_s=base[3], deficit_joint=base[4], deficit_gain=base[5],
                         n=int(base[6]))

    base, lo, hi = block_bootstrap(boundary_jump, groups, n_boot, rng)
    if base is not None:
        print(f"\nM7  CHUNK-BOUNDARY DISCONTINUITY of the published target (n = {int(base[4])})")
        print(f"      ‖cmd_new − cmd_old‖  {_ci(base, lo, hi, 0)} mm   p95 {1000 * base[1]:.2f} mm")
        print(f"      a normal in-chunk step is {_ci(base, lo, hi, 2)} mm"
              f"  =>  the jump is {base[0] / base[2]:.1f}x one step")
        print(f"      as a commanded velocity spike: {_ci(base, lo, hi, 3, 1.0, 5, 3)} m/s")
        out["m7_jump"] = dict(jump_mm=1000 * base[0], p95_mm=1000 * base[1],
                              in_chunk_mm=1000 * base[2], spike_mps=base[3], n=int(base[4]))

    print("\n    divergence bound if you DELETE the re-anchor (pure command integration).")
    print("    Bounds the ASK, never the closeness — the arm's response to a bigger ask is not")
    print("    in the recording.")
    print(f"      {'clamp':<12}{'end gap':>12}{'mean peak gap':>26}{'p95 peak':>12}"
          f"{'peak anchor lead':>20}")
    print("    Two columns because eps bounds the ANCHOR LEAD, not the gap: the gap is the lead")
    print("    plus the chunk's own excursion, so it exceeds eps by construction.")
    out["m7_divergence"] = []
    for label, eps in (("none", None), ("eps = 5 cm", 0.05), ("eps = 3 cm", 0.03)):
        b, lo2, hi2 = block_bootstrap(lambda g, e=eps: divergence_replay(g, e), groups, n_boot, rng)
        if b is None:
            continue
        print(f"      {label:<12}{_ci(b, None, None, 0, 1000.0, 8, 1):>12}"
              f"{_ci(b, lo2, hi2, 1, 1000.0, 6, 1):>26}{1000 * b[2]:12.1f}"
              f"{1000 * b[3]:20.1f}")
        out["m7_divergence"].append(dict(clamp=label, end_mm=1000 * b[0], peak_gap_mm=1000 * b[1],
                                         p95_peak_gap_mm=1000 * b[2],
                                         peak_anchor_lead_mm=1000 * b[3], n_takes=int(b[4])))

    print("\nM8  FORCE CHANNEL vs step-within-chunk")
    n_cmd = sum(1 for g in groups if g.get("wire_force_nonzero"))
    print(f"      takes whose :8770 wire ever carried a NONZERO force command: {n_cmd}/{len(groups)}.")
    out["m8_wire_force_takes"] = n_cmd
    if n_cmd == 0:
        print("      => NO commanded-vs-measured force TRACKING is available here; the force")
        print("         switch was off for every take, so only the PREDICTION error exists.")
    else:
        wf = block_bootstrap(lambda g: wire_force_profile(g, "err"), groups, n_boot, rng)
        wc = block_bootstrap(lambda g: wire_force_profile(g, "cmd"), groups, n_boot, rng)
        wg = block_bootstrap(wire_force_gain, groups, n_boot, rng)
        print("      force TRACKING on those takes (wire command vs sensor, both measured-sign):")
        print(f"      {'step':<8}{'‖F_wire−F_meas‖ (N)':>26}{'‖F_wire‖ (N)':>18}")
        out["m8_wire"] = []
        for i, (a, b) in enumerate(BINS):
            if wf[0] is None:
                break
            print(f"      {f'{a}-{b}':<8}{_ci(*wf, i, 1.0, 7, 3):>26}{_ci(*wc, i, 1.0, 6, 3):>18}")
            out["m8_wire"].append(dict(step=[a, b], err_n=wf[0][i], cmd_n=wc[0][i]))
        if wg[0] is not None:
            print(f"      realized force fraction g_F = {_ci(*wg, 0, 1.0, 6, 3)}"
                  f"   R² {wg[0][1]:.3f}   n = {int(wg[0][2])}")
            print("      Compare against M5's position g. A force channel that realizes a very")
            print("      different fraction than the position channel cannot be corrected by one")
            print("      shared gain, which is what a force-conditioned corrector assumes.")
            out["m8_wire_gain"] = dict(g_force=wg[0][0], r2=wg[0][1], n=int(wg[0][2]))
    fp = block_bootstrap(lambda g: wrench_err_profile(g, slice(0, 3)), groups, n_boot, rng)
    tp = block_bootstrap(lambda g: wrench_err_profile(g, slice(3, 6)), groups, n_boot, rng)
    mp = block_bootstrap(lambda g: wrench_err_profile(g, slice(0, 3), "meas"), groups, n_boot, rng)
    pos_p = block_bootstrap(lambda g: step_profile(g, "nrm"), groups, n_boot, rng)
    print(f"      {'step':<8}{'‖F_pred−F_meas‖ (N)':>26}{'‖tau err‖ (Nm)':>20}"
          f"{'‖F_meas‖ (N)':>16}{'‖lead‖ (mm)':>14}")
    out["m8"] = []
    for i, (a, b) in enumerate(BINS):
        print(f"      {f'{a}-{b}':<8}{_ci(*fp, i, 1.0, 7, 3):>26}{_ci(*tp, i, 1.0, 6, 3):>20}"
              f"{_ci(*mp, i, 1.0, 6, 3):>16}{_ci(*pos_p, i, 1000.0, 6, 2):>14}")
        out["m8"].append(dict(step=[a, b], f_err_n=fp[0][i], tau_err_nm=tp[0][i],
                              f_meas_n=mp[0][i], lead_mm=1000 * pos_p[0][i]))
    centres = np.array([(a + b) / 2 for a, b in BINS])
    print("\n      normalised ramp (slope / value at step 0) — is the force error chunk-phased too?")
    for nm, prof in (("‖F_pred−F_meas‖", fp[0]), ("‖F_meas‖", mp[0]), ("‖lead‖", pos_p[0])):
        okm = np.isfinite(prof) & (centres > 0)
        if okm.sum() >= 3 and np.isfinite(prof[0]) and abs(prof[0]) > 1e-12:
            sl = float(np.polyfit(centres[okm], prof[okm], 1)[0])
            print(f"        {nm:<18}{sl / prof[0]:+.4f} / step   (raw slope {sl:+.5f} per step)")
            out.setdefault("m8_ramp", {})[nm] = dict(slope=sl, normalised=sl / prof[0])
    return out


def report(label: str, groups: list[dict], *, per_tick: bool, n_boot: int, rng) -> dict:
    out: dict = {"label": label, "takes": len(groups),
                 "rows": int(sum(g["lead"].shape[0] for g in groups))}
    print(f"\n{'=' * 98}\n{label}  |  {out['takes']} takes, {out['rows']} rows"
          f"  |  median dt {1000 * np.nanmedian([g['dt'] for g in groups]):.1f} ms\n{'=' * 98}")

    print("M1  lead_axis = a·v_axis + b   [block bootstrap over takes, 95% CI]")
    print(f"    {'axis':<5}{'a  (s)':>26}{'b = INTERCEPT (mm)':>30}{'R²':>8}")
    out["m1"] = {}
    for ax, name in enumerate(AXES):
        base, lo, hi = block_bootstrap(lambda g, a=ax: fit_lead_vs_velocity(g, a),
                                      groups, n_boot, rng)
        if base is None:
            continue
        print(f"    {name:<5}{_ci(base, lo, hi, 0, 1.0, 8, 4):>26}"
              f"{_ci(base, lo, hi, 1):>30}{base[2]:8.3f}")
        out["m1"][name] = dict(a=base[0], b_mm=1000 * base[1], r2=base[2],
                              b_ci_mm=[1000 * lo[1], 1000 * hi[1]] if lo is not None else None)

    base, lo, hi = block_bootstrap(standing_lead, groups, n_boot, rng)
    print(f"\nM1b standing lead — both speeds < {100 * NEAR_ZERO_MPS:.0f} cm/s")
    if base is None:
        print("    no samples")
    else:
        for ax, name in enumerate(AXES):
            print(f"    lead_{name}  {_ci(base, lo, hi, ax, w=7)} mm")
        print(f"    ‖lead‖   {_ci(base, lo, hi, 3, w=7)} mm   n = {int(base[3 + 1])}")
        out["m1b"] = dict(mean_mm=[1000 * base[i] for i in range(3)], norm_mm=1000 * base[3],
                          norm_ci_mm=[1000 * lo[3], 1000 * hi[3]] if lo is not None else None,
                          n=int(base[4]))

    if per_tick:
        print("\nM2  lead vs STEP-WITHIN-CHUNK, split by Koren (mm)  — step 0 IS the anchor")
        prof = {}
        for key, name in (("nrm", "‖lead‖"), ("e_par", "|e_along| lag"),
                          ("e_perp", "|e_normal| geom")):
            prof[name] = block_bootstrap(lambda g, k=key: step_profile(g, k),
                                        groups, n_boot, rng)
        print(f"    {'step':<8}{'‖lead‖':>24}{'|e_along| lag':>24}{'|e_normal| geom':>24}{'norm frac':>10}")
        out["m2"] = []
        for i, (a, b) in enumerate(BINS):
            cells = [_ci(*prof[n], i) for n in ("‖lead‖", "|e_along| lag", "|e_normal| geom")]
            tot, nor = prof["‖lead‖"][0][i], prof["|e_normal| geom"][0][i]
            frac = f"{nor / tot:10.2f}" if np.isfinite(tot) and tot > 0 else "        --"
            print(f"    {f'{a}-{b}':<8}{cells[0]:>24}{cells[1]:>24}{cells[2]:>24}{frac}")
            out["m2"].append(dict(step=[a, b], norm_mm=1000 * tot,
                                  along_mm=1000 * prof["|e_along| lag"][0][i],
                                  normal_mm=1000 * nor))
        print("\n    growth rate over steps ≥1 (least squares on bin means):")
        centres = np.array([(a + b) / 2 for a, b in BINS])
        for name in ("‖lead‖", "|e_along| lag", "|e_normal| geom"):
            v = prof[name][0]
            ok = np.isfinite(v) & (centres > 0)
            if ok.sum() >= 3:
                slope = float(np.polyfit(centres[ok], 1000 * v[ok], 1)[0])
                print(f"      {name:<18}{slope:+.3f} mm/step  (={slope * 30:+.1f} mm over a "
                      f"30-step chunk, {slope * 50:+.1f} over 50)")

        print("\n    per-axis mean lead vs step (mm) — sign: negative = arm is PAST the command")
        print(f"    {'step':<8}{'x':>9}{'y':>9}{'z':>9}")
        ap = {n: block_bootstrap(lambda g, a=i: axis_step_profile(g, a), groups, 0, rng)[0]
              for i, n in enumerate(AXES)}
        for i, (a, b) in enumerate(BINS):
            print(f"    {f'{a}-{b}':<8}" + "".join(f"{1000 * ap[n][i]:9.2f}" for n in AXES))

        base, lo, hi = block_bootstrap(boundary_drop, groups, n_boot, rng)
        if base is not None:
            print(f"\nM3  CHUNK BOUNDARY — the re-anchor's amnesty  (n = {int(base[3])} transitions)")
            print(f"    ‖lead‖ before {1000 * base[0]:6.2f} mm  →  after {1000 * base[1]:6.2f} mm"
                  f"   =  {100 * base[2]:.0f}%"
                  + (f" [{100 * lo[2]:.0f}%, {100 * hi[2]:.0f}%]" if lo is not None else "")
                  + " WIPED")
            out["m3"] = dict(before_mm=1000 * base[0], after_mm=1000 * base[1],
                             frac_wiped=base[2],
                             ci=[lo[2], hi[2]] if lo is not None else None, n=int(base[3]))

        print(f"\nM4  contact (‖F‖ ≥ {CONTACT_N} N) vs free space, ‖lead‖ by step (mm)")
        bc = block_bootstrap(lambda g: step_profile(g, "nrm", "contact"), groups, n_boot, rng)
        bf = block_bootstrap(lambda g: step_profile(g, "nrm", "free"), groups, n_boot, rng)
        print(f"    {'step':<8}{'contact':>26}{'free':>26}{'ratio':>8}")
        for i, (a, b) in enumerate(BINS):
            r = (f"{bc[0][i] / bf[0][i]:8.2f}"
                 if np.isfinite(bc[0][i]) and np.isfinite(bf[0][i]) and bf[0][i] > 0 else "      --")
            print(f"    {f'{a}-{b}':<8}{_ci(*bc, i):>26}{_ci(*bf, i):>26}{r}")

    base, lo, hi = block_bootstrap(contact_corr, groups, n_boot, rng)
    if base is not None:
        print(f"\nM4b RL2AC — corr(‖e‖, ‖F_meas‖) in contact = {base[0]:+.3f}"
              + (f" [{lo[0]:+.3f}, {hi[0]:+.3f}]" if lo is not None else "")
              + f"   n = {int(base[1])}")
        print(f"    mean ‖e‖ {1000 * base[2]:.1f} mm, mean ‖F‖ {base[3]:.2f} N."
              " A CI straddling 0 means the 'tracking error IS the force channel' story")
        print("    is NOT supported on this data; a NEGATIVE corr contradicts it outright.")
        out["m4b"] = dict(corr=base[0], ci=[lo[0], hi[0]] if lo is not None else None,
                          n=int(base[1]))
    if per_tick:
        out.update(report_mechanism(groups, n_boot=n_boot, rng=rng))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="data/deploy_takes (scans every session) or one session dir")
    ap.add_argument("--algo", default=None, help="only sessions with this session_meta algo")
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write the numbers here")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    root = Path(args.root)
    sessions = ([root] if (root / "take_meta.json").is_file() or find_takes(root)
                and not any(p.is_dir() and (p / "session_meta.json").is_file()
                            for p in root.iterdir())
                else sorted(p for p in root.iterdir() if p.is_dir()))

    per_pred: dict[str, list] = defaultdict(list)
    per_tick: dict[str, list] = defaultdict(list)
    frame_err: list[float] = []
    counts: list[tuple] = []
    for sd in sessions:
        mp = sd / "session_meta.json"
        algo = str(json.loads(mp.read_text()).get("algo", "?")) if mp.is_file() else "?"
        if args.algo and algo != args.algo:
            continue
        na = nb = 0
        for td in find_takes(sd):
            try:
                take = DeployTake(td)
            except Exception as exc:  # noqa: BLE001 — a malformed take must not stop the sweep
                print(f"  skip {td.name}: {exc}", file=sys.stderr)
                continue
            pos, _ = take.measured_tcp()
            if len(take) and take.predictions:
                for p in take.predictions[:40]:
                    j = int(np.argmin(np.abs(take.times - p.t_predict_s)))
                    a = np.asarray(p.anchor_pos)
                    if a.ndim == 2 and a.shape[0]:
                        frame_err.append(float(np.linalg.norm(a[0] - pos[j])))
            # The ALL bucket is only meaningful when more than one algo is in scope. Under --algo
            # it would be a byte-identical duplicate of the single per-algo report, doubling both
            # the runtime (M5-M8 bootstrap dominates) and the archived output.
            sa = series_per_prediction(take)
            if sa:
                per_pred[algo].append(sa)
                if not args.algo:
                    per_pred["ALL"].append(sa)
                na += 1
            sb = series_per_tick(take)
            if sb:
                per_tick[algo].append(sb)
                if not args.algo:
                    per_tick["ALL"].append(sb)
                nb += 1
        if na or nb:
            counts.append((sd.name, algo, na, nb))

    if frame_err:
        fe = np.asarray(frame_err)
        print(f"M0  frame sanity  ‖anchor_pos − measured_tcp‖ over {fe.size} predictions: "
              f"median {1000 * np.median(fe):.2f} mm, p95 {1000 * np.percentile(fe, 95):.2f} mm")
        print("    ~0 ⇒ the two are the SAME quantity, so every number below is one TCP frame.")

    results = {"m0_median_mm": 1000 * float(np.median(frame_err)) if frame_err else None,
               "sessions": [dict(session=s, algo=a, n_series_a=x, n_series_b=y)
                            for s, a, x, y in counts], "reports": []}
    for algo, groups in sorted(per_pred.items()):
        if groups:
            results["reports"].append(
                report(f"SERIES A (per prediction)  algo={algo}", groups,
                       per_tick=False, n_boot=args.n_boot, rng=rng))
    for algo, groups in sorted(per_tick.items()):
        if groups:
            results["reports"].append(
                report(f"SERIES B (per tick, step-resolved)  algo={algo}", groups,
                       per_tick=True, n_boot=args.n_boot, rng=rng))

    print("\nsessions scanned:")
    for s, a, x, y in counts:
        print(f"  {s}  algo={a:<10} series-A takes={x:<3} series-B takes={y}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=float))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
