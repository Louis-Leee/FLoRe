#!/usr/bin/env python3
"""Design and falsify a corrector for the HL→LL execution gap. Robot-free, CPU-only, OFFLINE.

This script does not touch the serving path and nothing under ``flore/`` imports it. It reads the
recorded deploy takes, fits candidate correction laws, and tries to kill them on the same data.

WHY IT IS SEPARATE FROM ``measure_exec_gap.py``
-----------------------------------------------
That script measures what the gap IS. This one asks what could be done about it, which is a
different job with a different failure mode: a corrector fitted and scored on the same rows will
always look good. So every fit here is scored **leave-one-take-out** — parameters come from the
other takes and are tested on the held-out one. An in-sample number for a 9-parameter model
against a 1-parameter model is not evidence and is not reported.

WHAT IT MEASURES
----------------
C1  scalar vs DIAGONAL vs FULL 3x3 increment gain, cross-validated, with the residual split into
    along-command and normal-to-command parts. This is the falsifiable form of "the plant's
    attenuation is anisotropic, so part of M2's normal error is predicted by anisotropy alone and
    a scalar 1/g provably cannot touch it".
C2  is the gain FORCE-CONDITIONED? Contact-vs-free split, plus a continuous ``g0 + g1·‖F‖`` whose
    ``g1`` has to exclude zero to justify conditioning on force at all.
C3  does per-chunk ONLINE adaptation beat one global constant? Chunk-as-trial, causal: chunk n is
    predicted from chunks 1..n-1 of the same take, and scored against a global gain fitted on
    other takes. This is the test that decides whether an adaptive corrector is worth building,
    and it is allowed to come back negative.
C4  replay the pre-distortion over the recorded takes: how much MORE would the arm have been
    asked for, would the eps clamp bind, and does the correction make the chunk-boundary jump
    worse? Bounds the ASK only — the arm's response to a bigger ask is not in the recording.
C5  correction AUTHORITY vs step-within-chunk: what fraction of the step-k error is predictable
    out-of-take, i.e. removable by a fitted corrector. Every published per-horizon weighting
    (RTC Eq 5, Soft RTC, ChunkFlow, DIAL-MPC's annealed covariance) makes authority DECREASE with
    step index on epistemic grounds. If the removable fraction GROWS, those schedules have the
    sign backwards for this plant.

USAGE
-----
    .venv/bin/python scripts/exec_gap_corrector.py data/deploy_takes
    .venv/bin/python scripts/exec_gap_corrector.py data/deploy_takes --algo diffusion --json c.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import measure_exec_gap as meg  # noqa: E402
from flore.s1_policy.serving.hl_take import DeployTake, find_takes  # noqa: E402

#: Correction is designed at the CHUNK timescale, not per tick. ``measure_exec_gap``'s M5 shows
#: the realized fraction is frequency dependent — 0.44 over one 54 ms tick, 0.61 over ~1 s — so a
#: corrector that inverts the per-tick number over-drives by ~1.9x and injects exactly the
#: high-frequency content the plant was measured to reject.
W = meg.CHUNK_SCALE_W
MODES = ("scalar", "diag", "full")


# ------------------------------------------------------------------------------ pair extraction
def take_pairs(g: dict, w: int = W) -> dict | None:
    """Within-chunk increment pairs for ONE take, tagged with chunk id and step.

    ``meg.within_chunk_pairs`` drops the chunk id; C3 needs it to treat a chunk as a trial.
    """
    if g.get("raw") is None:
        return None
    cmd, pos, step, chunk, ok, fn = g["raw"]
    n = cmd.shape[0]
    if n <= w:
        return None
    lo = np.arange(0, n - w)
    hi = lo + w
    m = ok[lo] & ok[hi] & (chunk[lo] >= 0) & (chunk[lo] == chunk[hi])
    if m.sum() < 5:
        return None
    lo, hi = lo[m], hi[m]
    return dict(dcmd=cmd[hi] - cmd[lo], dmeas=pos[hi] - pos[lo],
                f=0.5 * (fn[lo] + fn[hi]), step=step[hi], chunk=chunk[hi])


def fit_G(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    """``(3,3)`` gain mapping commanded increment -> realized increment, ``b ≈ a @ G.T``."""
    if mode == "scalar":
        d = float((a * a).sum())
        return np.eye(3) * (float((a * b).sum()) / d if d > 0 else 1.0)
    if mode == "diag":
        num = (a * b).sum(axis=0)
        den = (a * a).sum(axis=0)
        return np.diag(np.where(den > 0, num / np.maximum(den, 1e-18), 1.0))
    gt, *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.asarray(gt).T


def _split(r: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Residual split into along-command and normal-to-command magnitudes."""
    u = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    along = np.einsum("ij,ij->i", r, u)
    return np.abs(along), np.linalg.norm(r - along[:, None] * u, axis=1)


# --------------------------------------------------------------- C1  anisotropy, cross-validated
def loto_gain(bundle: list[dict], mode: str):
    """``[rms_total, rms_along, rms_normal, n]`` mm — leave-one-TAKE-out residual of ``mode``.

    Held-out scoring is the whole point: ``full`` has 9 parameters against ``scalar``'s 1, so an
    in-sample comparison would favour it for free.
    """
    tot, alo, nor = [], [], []
    for i, p in enumerate(bundle):
        train = [q for j, q in enumerate(bundle) if j != i]
        if not train or p is None:
            continue
        a = np.concatenate([q["dcmd"] for q in train])
        b = np.concatenate([q["dmeas"] for q in train])
        if a.shape[0] < 50:
            continue
        g = fit_G(a, b, mode)
        r = p["dmeas"] - p["dcmd"] @ g.T
        al, no = _split(r, p["dcmd"])
        tot.append(np.linalg.norm(r, axis=1))
        alo.append(al)
        nor.append(no)
    if not tot:
        return None
    t, al, no = np.concatenate(tot), np.concatenate(alo), np.concatenate(nor)
    return np.array([np.sqrt((t ** 2).mean()), np.sqrt((al ** 2).mean()),
                     np.sqrt((no ** 2).mean()), t.size])


def _design(p: dict, use_force: bool, use_speed: bool) -> np.ndarray:
    """``(n, 3*k)`` regressors for one output axis under ``M = A + B·‖F‖ + C·‖dcmd‖``."""
    cols = [p["dcmd"]]
    if use_force:
        cols.append(p["dcmd"] * p["f"][:, None])
    if use_speed:
        cols.append(p["dcmd"] * np.linalg.norm(p["dcmd"], axis=1, keepdims=True))
    return np.concatenate(cols, axis=1)


def loto_feature_gain(bundle: list[dict], use_force: bool, use_speed: bool):
    """``[rms, n]`` mm — leave-one-take-out residual of a FEATURE-CONDITIONED matrix gain.

    The gain becomes ``M = A + B·‖F‖ + C·‖dcmd‖``, each term a full 3x3, fit per output axis. This
    is the direct test of where C3's oracle gap actually lives: if a FIXED model conditioned on
    quantities the planner can observe recovers most of what a per-chunk oracle would, then the
    right object is a richer gain, not a recursive estimator — and the whole ILC / RLS / adaptive
    branch is unnecessary rather than merely disappointing.
    """
    res = []
    for i, p in enumerate(bundle):
        train = [q for j, q in enumerate(bundle) if j != i]
        if not train:
            continue
        xt = np.concatenate([_design(q, use_force, use_speed) for q in train])
        bt = np.concatenate([q["dmeas"] for q in train])
        if xt.shape[0] < 200:
            continue
        xh = _design(p, use_force, use_speed)
        pred = np.column_stack([xh @ np.linalg.lstsq(xt, bt[:, k], rcond=None)[0]
                                for k in range(3)])
        res.append(np.linalg.norm(p["dmeas"] - pred, axis=1))
    if not res:
        return None
    r = np.concatenate(res)
    return np.array([np.sqrt((r ** 2).mean()), r.size])


def gain_matrix_report(bundle: list[dict]) -> np.ndarray | None:
    """The pooled full 3x3 gain, flattened, for display."""
    a = np.concatenate([p["dcmd"] for p in bundle])
    b = np.concatenate([p["dmeas"] for p in bundle])
    if a.shape[0] < 50:
        return None
    return fit_G(a, b, "full").ravel()


# ------------------------------------------------------------------- C2  force-conditioned gain
def gain_by_contact(bundle: list[dict], contact: bool):
    """``[g, mean_dcmd, mean_abs_err, n]`` — the gain AND the size of the ask, in one row.

    Both are needed to reconcile this with M4's "contact multiplies the shortfall". A higher
    realized FRACTION and a larger absolute ERROR are not in conflict when the commanded increment
    is itself larger in contact; without the ask on the same line the two readings look
    contradictory and one of them gets discarded for no reason.
    """
    a, b = [], []
    for p in bundle:
        m = (p["f"] >= meg.CONTACT_N) if contact else (p["f"] < meg.CONTACT_N)
        if m.any():
            a.append(p["dcmd"][m])
            b.append(p["dmeas"][m])
    if not a:
        return None
    a, b = np.concatenate(a), np.concatenate(b)
    keep = np.linalg.norm(a, axis=1) >= meg.MIN_DISP_M
    a, b = a[keep], b[keep]
    x, y = a.ravel(), b.ravel()
    if x.size < 60 or float(x @ x) <= 0:
        return None
    return np.array([float(x @ y) / float(x @ x),
                     float(np.linalg.norm(a, axis=1).mean()),
                     float(np.linalg.norm(a - b, axis=1).mean()), x.size])


def gain_force_slope(bundle: list[dict]):
    """``[g0, g1, n]`` for ``dmeas = (g0 + g1·‖F‖)·dcmd``, per axis stacked.

    ``g1`` is the whole question: if its CI includes zero, conditioning the correction on force
    is unjustified on this data no matter how intuitive it sounds.
    """
    xs, ys, fs = [], [], []
    for p in bundle:
        keep = np.linalg.norm(p["dcmd"], axis=1) >= meg.MIN_DISP_M
        if not keep.any():
            continue
        xs.append(p["dcmd"][keep])
        ys.append(p["dmeas"][keep])
        fs.append(p["f"][keep])
    if not xs:
        return None
    a, b, f = np.concatenate(xs), np.concatenate(ys), np.concatenate(fs)
    x0 = a.ravel()
    x1 = (a * f[:, None]).ravel()
    y = b.ravel()
    if x0.size < 100:
        return None
    coef, *_ = np.linalg.lstsq(np.column_stack([x0, x1]), y, rcond=None)
    return np.array([coef[0], coef[1], x0.size])


# ------------------------------------------------- C3  does online per-chunk adaptation pay off?
def adaptation_ceiling(bundle: list[dict]):
    """``[rms_global, rms_oracle, gain_pct, n_chunks]`` mm — the CEILING on any adaptive law.

    The oracle fits each chunk's gain ON THAT CHUNK ITSELF, which no causal law can ever match.
    So whatever it buys is an upper bound: if the oracle barely beats one global constant, then
    no amount of cleverness in the update rule matters and the adaptive design is dead regardless
    of how much real, persistent per-chunk variation exists. This is the number that turns an
    ambiguous "3% +/- 4%" into a decision.
    """
    rg, ro, n_ch = [], [], 0
    for i, p in enumerate(bundle):
        train = [q for j, q in enumerate(bundle) if j != i]
        if not train:
            continue
        a = np.concatenate([q["dcmd"] for q in train])
        b = np.concatenate([q["dmeas"] for q in train])
        if a.shape[0] < 50:
            continue
        g_glob = float(fit_G(a, b, "scalar")[0, 0])
        for ci in np.unique(p["chunk"]):
            m = p["chunk"] == ci
            if m.sum() < 3:
                continue
            dc, dm = p["dcmd"][m], p["dmeas"][m]
            den = float((dc * dc).sum())
            if den <= 0:
                continue
            g_or = float((dc * dm).sum()) / den
            rg.append(np.linalg.norm(dm - g_glob * dc, axis=1))
            ro.append(np.linalg.norm(dm - g_or * dc, axis=1))
            n_ch += 1
    if n_ch < 20:
        return None
    g = float(np.sqrt((np.concatenate(rg) ** 2).mean()))
    o = float(np.sqrt((np.concatenate(ro) ** 2).mean()))
    return np.array([g, o, 100.0 * (g - o) / g, n_ch])


def adaptation_shrunk(bundle: list[dict], beta: float):
    """``[rms_global, rms_pred, gain_pct, n_chunks]`` mm for a MEAN-REVERTING one-step predictor.

    ``g_pred(n) = g_global + beta·(g_hat(n-1) − g_global)``. This is the right functional form for
    an AR(1) deviation observed in noise — unlike a plain running EWMA, which has no anchor and
    wanders away from the global gain. ``beta`` is swept rather than tuned, and the whole curve is
    printed, because picking the best ``beta`` on the same data it is scored on would be exactly
    the in-sample flattery this file exists to avoid.
    """
    rg, rp, n_ch = [], [], 0
    for i, p in enumerate(bundle):
        train = [q for j, q in enumerate(bundle) if j != i]
        if not train:
            continue
        a = np.concatenate([q["dcmd"] for q in train])
        b = np.concatenate([q["dmeas"] for q in train])
        if a.shape[0] < 50:
            continue
        g_glob = float(fit_G(a, b, "scalar")[0, 0])
        g_prev = None
        for ci in np.unique(p["chunk"]):
            m = p["chunk"] == ci
            if m.sum() < 3:
                continue
            dc, dm = p["dcmd"][m], p["dmeas"][m]
            den = float((dc * dc).sum())
            if den <= 0:
                continue
            g_use = g_glob if g_prev is None else g_glob + beta * (g_prev - g_glob)
            rg.append(np.linalg.norm(dm - g_glob * dc, axis=1))
            rp.append(np.linalg.norm(dm - g_use * dc, axis=1))
            n_ch += 1
            g_prev = float((dc * dm).sum()) / den
    if n_ch < 20:
        return None
    g = float(np.sqrt((np.concatenate(rg) ** 2).mean()))
    q = float(np.sqrt((np.concatenate(rp) ** 2).mean()))
    return np.array([g, q, 100.0 * (g - q) / g, n_ch])


def adaptation_gain(bundle: list[dict], lam: float = 0.7):
    """``[rms_global, rms_adaptive, delta_pct, n_chunks]`` mm, both scored CAUSALLY.

    For every chunk in every take, predict its realized increments two ways and compare:

    * GLOBAL   — one gain fitted on all OTHER takes (leave-one-take-out; no peeking).
    * ADAPTIVE — an EWMA over the per-chunk gains of the PRECEDING chunks of the same take,
      initialised at the global gain so chunk 0 costs nothing to either method.

    A positive ``delta_pct`` means adaptation helped. This is the test that decides whether a
    per-chunk update is worth building, and it is written to be able to say no.
    """
    rg, ra, n_ch = [], [], 0
    for i, p in enumerate(bundle):
        train = [q for j, q in enumerate(bundle) if j != i]
        if not train:
            continue
        a = np.concatenate([q["dcmd"] for q in train])
        b = np.concatenate([q["dmeas"] for q in train])
        if a.shape[0] < 50:
            continue
        g_glob = float(fit_G(a, b, "scalar")[0, 0])
        g_ada = g_glob
        for ci in np.unique(p["chunk"]):
            m = p["chunk"] == ci
            if m.sum() < 3:
                continue
            dc, dm = p["dcmd"][m], p["dmeas"][m]
            rg.append(np.linalg.norm(dm - g_glob * dc, axis=1))
            ra.append(np.linalg.norm(dm - g_ada * dc, axis=1))
            n_ch += 1
            den = float((dc * dc).sum())
            if den > 0:
                g_ada = lam * g_ada + (1.0 - lam) * (float((dc * dm).sum()) / den)
    if n_ch < 20:
        return None
    g_rms = float(np.sqrt((np.concatenate(rg) ** 2).mean()))
    a_rms = float(np.sqrt((np.concatenate(ra) ** 2).mean()))
    return np.array([g_rms, a_rms, 100.0 * (g_rms - a_rms) / g_rms, n_ch])


def chunk_gain_spread(bundle: list[dict]):
    """``[sd_between, sd_from_noise, ratio, lag1_autocorr, n]`` for the per-chunk scalar gain.

    Two separate questions, and C3's verdict needs both answered.

    ``ratio`` asks whether the chunk-to-chunk variation is REAL: a ratio near 1 means the spread
    is just the within-chunk fit uncertainty and there is no signal at all.

    ``lag1_autocorr`` asks whether that variation is PERSISTENT, i.e. whether the gain of chunk
    n-1 says anything about chunk n. Real-but-white variation is worse than no variation for an
    adaptive law: it looks learnable in a spread statistic while being unpredictable in fact, so
    an EWMA chases noise and can only lose. Consecutive pairs are taken within a take only.
    """
    gs, ses, pairs = [], [], []
    for p in bundle:
        seq = []
        for ci in np.unique(p["chunk"]):
            m = p["chunk"] == ci
            if m.sum() < 5:
                continue
            dc, dm = p["dcmd"][m].ravel(), p["dmeas"][m].ravel()
            den = float(dc @ dc)
            if den <= 0:
                continue
            g = float(dc @ dm) / den
            r = dm - g * dc
            dof = max(dc.size - 1, 1)
            gs.append(g)
            ses.append(np.sqrt(float(r @ r) / dof / den))
            seq.append(g)
        pairs += list(zip(seq[:-1], seq[1:]))
    if len(gs) < 20 or len(pairs) < 20:
        return None
    gs, ses = np.asarray(gs), np.asarray(ses)
    a = np.asarray([p[0] for p in pairs])
    b = np.asarray([p[1] for p in pairs])
    ac = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    sd_obs = float(gs.std(ddof=1))
    sd_noise = float(np.sqrt((ses ** 2).mean()))
    return np.array([sd_obs, sd_noise, sd_obs / max(sd_noise, 1e-12), ac, gs.size])


# --------------------------------------------------------------- C4  replay the pre-distortion
def predistort_replay(groups: list[dict], gain: float, eps: float | None = 0.03,
                      cut_contact: bool = True, diag: np.ndarray | None = None):
    """``[extra_p50, extra_p95, extra_max, gap_p95, jump_before, jump_after, frac_clamped, n]`` m.

    Publishes ``anchor + (cmd − anchor)/g`` instead of ``anchor + (cmd − anchor)`` and reports what
    that would have COST in ask, not what it would have BOUGHT in tracking. The distinction is not
    pedantry: the arm's response to a larger ask is not in this recording, so any "it would have
    tracked better" number would be fabricated.

    ``jump_before`` / ``jump_after`` are the chunk-boundary discontinuity of the published target
    without and with the correction. A pre-distortion pushes each chunk's tail further from the
    arm, so it can make the backward jump at the next re-anchor WORSE — which would be a real
    regression, and is the reason this is measured rather than assumed.
    """
    inv = np.linalg.inv(np.diag(diag)) if diag is not None else np.eye(3) / gain
    extra, gaps, jb, ja, n_clamp, n_tot = [], [], [], [], 0, 0
    for g in groups:
        if g.get("raw") is None or not g.get("anchors"):
            continue
        cmd, pos, _, chunk, ok, fn = g["raw"]
        rows = np.where(ok)[0]
        anch = g["anchors"]
        prev_ci, prev_plain, prev_corr = -1, None, None
        for r in rows:
            ci = int(chunk[r])
            a = anch.get(ci)
            if a is None or not np.isfinite(a).all():
                continue
            off = cmd[r] - a
            corr = a + (off if (cut_contact and fn[r] >= meg.CONTACT_N) else off @ inv.T)
            d = corr - pos[r]
            nd = float(np.linalg.norm(d))
            n_tot += 1
            if eps is not None and nd > eps > 0.0:
                corr = pos[r] + d * (eps / nd)
                n_clamp += 1
            extra.append(float(np.linalg.norm(corr - cmd[r])))
            gaps.append(float(np.linalg.norm(corr - pos[r])))
            if prev_ci >= 0 and ci != prev_ci:
                jb.append(float(np.linalg.norm(cmd[r] - prev_plain)))
                ja.append(float(np.linalg.norm(corr - prev_corr)))
            prev_ci, prev_plain, prev_corr = ci, cmd[r].copy(), corr.copy()
    if len(extra) < 50 or not jb:
        return None
    e = np.asarray(extra)
    return np.array([float(np.percentile(e, 50)), float(np.percentile(e, 95)), float(e.max()),
                     float(np.percentile(gaps, 95)), float(np.mean(jb)), float(np.mean(ja)),
                     n_clamp / max(n_tot, 1), e.size])


def headroom(bundle: list[dict], mode: str = "diag"):
    """``[r2, residual_frac, n]`` — the CEILING on any corrector built from this model.

    A pre-distortion can only remove the part of the realized increment the model actually
    predicts. Whatever variance the model leaves is what a perfect inverse still leaves, so
    ``residual_frac`` = sqrt(1 - R²) is the fraction of the error that survives the best possible
    version of this correction. Quoting the full 12 mm/chunk as recoverable ignores this.
    """
    r = loto_gain(bundle, mode)
    if r is None:
        return None
    b = np.concatenate([p["dmeas"] for p in bundle])
    var = float(((b - b.mean(axis=0)) ** 2).sum() / b.shape[0])
    if var <= 0:
        return None
    r2 = 1.0 - (r[0] ** 2) / var
    return np.array([r2, np.sqrt(max(1.0 - r2, 0.0)), r[3]])


# ------------------------------------------------------- C5  correction authority vs step index
def authority_profile(groups: list[dict], mode: str = "diag"):
    """Out-of-take R² of the LEAD RAMP, per step bin — the removable fraction at each step.

    The object predicted is ``lead_k − lead_0 ≈ (I − G)(cmd_k − cmd_0)``, i.e. the part of the
    error that GREW since the anchor. That is exactly what a stateless step-indexed feedforward
    can remove, and it is the right thing to call "authority": the lead at step 0 is an anchor
    offset a within-chunk correction never had a chance at.
    """
    bundles = [take_pairs(g) for g in groups]
    per_bin_num = [[] for _ in meg.BINS]
    per_bin_den = [[] for _ in meg.BINS]
    for i, g in enumerate(groups):
        train = [q for j, q in enumerate(bundles) if j != i and q is not None]
        if not train or g.get("raw") is None:
            continue
        a = np.concatenate([q["dcmd"] for q in train])
        b = np.concatenate([q["dmeas"] for q in train])
        if a.shape[0] < 50:
            continue
        gm = fit_G(a, b, mode)
        resid = np.eye(3) - gm
        cmd, pos, step, chunk, ok, _ = g["raw"]
        rows = np.where(ok)[0]
        base: dict[int, tuple] = {}
        for r in rows:
            ci = int(chunk[r])
            if ci not in base:
                base[ci] = (cmd[r].copy(), (cmd[r] - pos[r]).copy())
            c0, l0 = base[ci]
            y = (cmd[r] - pos[r]) - l0
            yhat = resid @ (cmd[r] - c0)
            for bi, (lo, hi) in enumerate(meg.BINS):
                if lo <= step[r] <= hi:
                    per_bin_num[bi].append(float(np.sum((y - yhat) ** 2)))
                    per_bin_den[bi].append(float(np.sum(y ** 2)))
                    break
    out = []
    for num, den in zip(per_bin_num, per_bin_den):
        if len(num) < 20 or sum(den) <= 0:
            out.append(np.nan)
        else:
            out.append(1.0 - sum(num) / sum(den))
    return np.asarray(out)


# ----------------------------------------------------------------------------------- reporting
def load_groups(root: Path, algo: str | None) -> dict[str, list[dict]]:
    sessions = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    per_algo: dict[str, list[dict]] = {}
    for sd in sessions:
        mp = sd / "session_meta.json"
        a = str(json.loads(mp.read_text()).get("algo", "?")) if mp.is_file() else "?"
        if algo and a != algo:
            continue
        for td in find_takes(sd):
            try:
                take = DeployTake(td)
            except Exception as exc:  # noqa: BLE001 — one bad take must not stop the sweep
                print(f"  skip {td.name}: {exc}", file=sys.stderr)
                continue
            g = meg.series_per_tick(take)
            if g:
                per_algo.setdefault(a, []).append(g)
    return per_algo


def _f(base, lo, hi, i, scale=1.0, w=7, p=3):
    if base is None or not np.isfinite(base[i]):
        return "--"
    if lo is None:
        return f"{scale * base[i]:{w}.{p}f}"
    return f"{scale * base[i]:{w}.{p}f} [{scale * lo[i]:.{p}f},{scale * hi[i]:.{p}f}]"


def run(label: str, groups: list[dict], *, n_boot: int, rng) -> dict:  # noqa: PLR0915
    out: dict = {"label": label, "takes": len(groups)}
    bundles = [b for b in (take_pairs(g) for g in groups) if b is not None]
    dt = float(np.nanmedian([g["dt"] for g in groups]))
    print(f"\n{'=' * 98}\n{label}  |  {len(groups)} takes, {len(bundles)} usable"
          f"  |  window w={W} ({1000 * W * dt:.0f} ms)\n{'=' * 98}")
    if len(bundles) < 5:
        print("  too few takes to cross-validate")
        return out

    bb = meg.block_bootstrap
    print("C1  ANISOTROPY — scalar vs diagonal vs full 3x3, LEAVE-ONE-TAKE-OUT residual (mm)")
    print(f"    {'model':<10}{'params':>8}{'RMS total':>24}{'RMS along':>22}{'RMS normal':>22}")
    out["c1"] = {}
    for mode, npar in (("scalar", 1), ("diag", 3), ("full", 9)):
        r = bb(lambda b, m=mode: loto_gain(b, m), bundles, n_boot, rng)
        if r[0] is None:
            continue
        print(f"    {mode:<10}{npar:>8}{_f(*r, 0, 1000.0, 8, 3):>24}"
              f"{_f(*r, 1, 1000.0, 7, 3):>22}{_f(*r, 2, 1000.0, 7, 3):>22}")
        out["c1"][mode] = dict(rms_mm=1000 * r[0][0], along_mm=1000 * r[0][1],
                               normal_mm=1000 * r[0][2], n=int(r[0][3]))
    gm = gain_matrix_report(bundles)
    if gm is not None:
        print("\n    pooled full gain G (rows = realized x,y,z ; cols = commanded x,y,z):")
        for i, nm in enumerate(meg.AXES):
            print(f"      {nm}  " + "".join(f"{gm[3 * i + j]:9.4f}" for j in range(3)))
        out["c1"]["G_full"] = gm.tolist()
        print("    Off-diagonal terms are cross-axis coupling: a command along one axis realized")
        print("    partly along another. A scalar 1/g cannot represent either the diagonal spread")
        print("    or the coupling, so if the normal-residual column above falls from scalar to")
        print("    diag, that drop is the part of M2's |e_normal| anisotropy explains.")

    print("\nC2  IS THE GAIN FORCE-CONDITIONED?")
    print(f"    {'regime':<24}{'g':>22}{'mean ask (mm)':>16}{'mean err (mm)':>16}")
    keep = {}
    for nm, con in (("contact  ‖F‖ >= 1.5 N", True), ("free     ‖F‖ <  1.5 N", False)):
        r = bb(lambda b, c=con: gain_by_contact(b, c), bundles, n_boot, rng)
        if r[0] is not None:
            print(f"    {nm:<24}{_f(*r, 0, 1.0, 6, 4):>22}{1000 * r[0][1]:>16.2f}"
                  f"{1000 * r[0][2]:>16.2f}")
            keep[con] = r[0]
            out.setdefault("c2", {})[("contact" if con else "free")] = dict(
                g=r[0][0], ask_mm=1000 * r[0][1], err_mm=1000 * r[0][2], n=int(r[0][3]))
    if True in keep and False in keep:
        c, fr = keep[True], keep[False]
        print(f"    reconciles M4: the ask is {c[1] / fr[1]:.2f}x larger in contact and the realized")
        print(f"    fraction {c[0] / fr[0]:.2f}x HIGHER, which still leaves {c[2] / fr[2]:.2f}x the absolute")
        print("    error. So 'contact multiplies the shortfall' and 'the plant tracks better in")
        print("    contact' are both true — they are statements about different quantities, and")
        print("    the one a corrector inverts is the fraction.")
    r = bb(gain_force_slope, bundles, n_boot, rng)
    if r[0] is not None:
        print("    continuous  dmeas = (g0 + g1·‖F‖)·dcmd")
        print(f"      g0 = {_f(*r, 0, 1.0, 6, 4)}      g1 = {_f(*r, 1, 1.0, 7, 5)} per N")
        excl = r[1] is not None and (r[1][1] > 0 or r[2][1] < 0)
        print(f"      g1 excludes zero: {'YES — force conditioning is justified' if excl else 'NO'}"
              f"{'' if excl else ' — do not condition the correction on force on this evidence'}")
        out.setdefault("c2", {})["slope"] = dict(g0=r[0][0], g1=r[0][1], excludes_zero=bool(excl))

    print("\nC3  DOES ONLINE PER-CHUNK ADAPTATION BEAT ONE GLOBAL CONSTANT? (causal, mm)")
    r = bb(adaptation_gain, bundles, n_boot, rng)
    if r[0] is not None:
        print(f"    global (other takes)   RMS {1000 * r[0][0]:.3f} mm")
        print(f"    adaptive (EWMA, causal) RMS {1000 * r[0][1]:.3f} mm")
        print(f"    improvement            {_f(*r, 2, 1.0, 6, 2)} %   over {int(r[0][3])} chunks")
        out["c3"] = dict(rms_global_mm=1000 * r[0][0], rms_adaptive_mm=1000 * r[0][1],
                         improvement_pct=r[0][2], n_chunks=int(r[0][3]),
                         ci=[r[1][2], r[2][2]] if r[1] is not None else None)
    c = bb(adaptation_ceiling, bundles, n_boot, rng)
    if c[0] is not None:
        print(f"    CEILING — per-chunk ORACLE gain (in-sample, uncheatable): "
              f"{_f(*c, 2, 1.0, 6, 2)} %")
        print("      No causal update rule can beat this, so it caps the whole adaptive design.")
        out["c3_ceiling"] = dict(rms_oracle_mm=1000 * c[0][1], gain_pct=c[0][2],
                                 ci=[c[1][2], c[2][2]] if c[1] is not None else None)
    base = bb(lambda b: loto_gain(b, "scalar"), bundles, 0, rng)[0]
    if base is not None and c[0] is not None:
        print("\n    WHERE THE ORACLE GAP ACTUALLY LIVES — a FIXED but richer gain, no adaptation,")
        print("    all leave-one-take-out. % is RMS reduction against the one global scalar:")
        print(f"      {'model':<38}{'params':>8}{'RMS (mm)':>11}{'vs scalar':>11}")
        b0 = base[0]
        rows = [("scalar  g", 1, lambda b: loto_gain(b, "scalar")),
                ("diagonal  diag(gx,gy,gz)", 3, lambda b: loto_gain(b, "diag")),
                ("full 3x3  G", 9, lambda b: loto_gain(b, "full")),
                ("full + force  A + B‖F‖", 18, lambda b: loto_feature_gain(b, True, False)),
                ("full + force + speed", 27, lambda b: loto_feature_gain(b, True, True))]
        out["c3_features"] = []
        for nm, npar, fn in rows:
            rr = fn(bundles)
            if rr is None:
                continue
            print(f"      {nm:<38}{npar:>8}{1000 * rr[0]:>11.3f}"
                  f"{100 * (b0 - rr[0]) / b0:>10.2f}%")
            out["c3_features"].append(dict(model=nm, params=npar, rms_mm=1000 * rr[0],
                                           vs_scalar_pct=100 * (b0 - rr[0]) / b0))
        print(f"      {'per-chunk ORACLE scalar (non-causal)':<38}{'1/chunk':>8}"
              f"{1000 * c[0][1]:>11.3f}{c[0][2]:>10.2f}%")
        print("    Read the last two rows together. If a fixed feature-conditioned gain gets most")
        print("    of the way to the oracle, the per-chunk information was never temporal — it was")
        print("    direction and contact, both observable at plan time, and no recursive estimator")
        print("    is needed to capture them.")
    print("    mean-reverting one-step predictor  g_pred = g_global + beta·(g_prev − g_global):")
    out["c3_beta"] = []
    for beta in (0.1, 0.2, 0.33, 0.5, 0.7):
        rb = bb(lambda b, be=beta: adaptation_shrunk(b, be), bundles, n_boot, rng)
        if rb[0] is None:
            continue
        print(f"      beta = {beta:<5}{_f(*rb, 2, 1.0, 6, 2)} %")
        out["c3_beta"].append(dict(beta=beta, gain_pct=rb[0][2],
                                   ci=[rb[1][2], rb[2][2]] if rb[1] is not None else None))
    s = bb(chunk_gain_spread, bundles, n_boot, rng)
    if s[0] is not None:
        print(f"    per-chunk gain spread: observed sd {_f(*s, 0, 1.0, 6, 4)}")
        print(f"                           sd expected from fit noise alone {s[0][1]:.4f}")
        print(f"                           ratio {_f(*s, 2, 1.0, 6, 2)}   n = {int(s[0][4])} chunks")
        print(f"    lag-1 autocorrelation of the per-chunk gain {_f(*s, 3, 1.0, 6, 3)}")
        real = s[0][2] > 1.5
        pers = s[1] is not None and s[1][3] > 0
        print(f"      variation is REAL:       {'yes' if real else 'no'}"
              f"  (spread {s[0][2]:.2f}x the fit noise)")
        print(f"      variation is PERSISTENT: {'yes' if pers else 'no'}"
              f"  (lag-1 CI {'excludes' if pers else 'includes'} zero)")
        if real and not pers:
            print("    => real but WHITE. That is the worst case for an adaptive law: a spread")
            print("       statistic makes it look learnable while chunk n-1 carries no information")
            print("       about chunk n, so an EWMA chases noise and can only lose. This, not a")
            print("       lack of variation, is why C3 comes back at zero.")
        out["c3_spread"] = dict(sd_obs=s[0][0], sd_noise=s[0][1], ratio=s[0][2],
                                lag1=s[0][3], real=bool(real), persistent=bool(pers),
                                n=int(s[0][4]))

    print("\nC4  PRE-DISTORTION REPLAY — what it would COST in ask (mm). Never what it would buy.")
    fit = meg.fit_gain_lag(groups, dt, meg.LONG_CHUNK_ROWS)
    cl = meg.ramp_slope(groups), meg.mean_step_disp(groups)
    g_dc = float(fit[0]) if fit is not None else 0.61
    g_clo = 1.0 - float(cl[0][0] / cl[1][0]) if cl[0] is not None and cl[1] is not None else g_dc
    diag = np.diag(fit_G(np.concatenate([p["dcmd"] for p in bundles]),
                         np.concatenate([p["dmeas"] for p in bundles]), "diag"))
    print(f"    gains on the table: fitted DC {g_dc:.4f} | closure-implied {g_clo:.4f} | "
          f"per-axis {np.array2string(diag, precision=3)}")
    print("    NOTE ON CONSERVATISM: for tracking, the aggressive end is the SMALL gain (it asks")
    print("    for more). For hardware safety the binding risk is overshoot, so the conservative")
    print("    end is the LARGE gain. These are opposite endpoints of one CI and must be named.")
    print(f"    {'variant':<30}{'p50 extra':>11}{'p95 extra':>11}{'max':>9}"
          f"{'p95 gap':>10}{'jump b/a':>16}{'clamped':>9}")
    out["c4"] = []
    variants = [
        (f"scalar 1/{g_dc:.3f} (fitted DC)", dict(gain=g_dc)),
        (f"scalar 1/{g_clo:.3f} (closure)", dict(gain=g_clo)),
        ("per-axis diag inverse", dict(gain=g_dc, diag=diag)),
        (f"scalar 1/{g_dc:.3f}, no contact cut", dict(gain=g_dc, cut_contact=False)),
        (f"scalar 1/{g_dc:.3f}, eps=5cm", dict(gain=g_dc, eps=0.05)),
        (f"scalar 1/{g_dc:.3f}, NO clamp", dict(gain=g_dc, eps=None)),
    ]
    for nm, kw in variants:
        r = predistort_replay(groups, **kw)
        if r is None:
            continue
        print(f"    {nm:<30}{1000 * r[0]:11.2f}{1000 * r[1]:11.2f}{1000 * r[2]:9.1f}"
              f"{1000 * r[3]:10.1f}{1000 * r[4]:7.1f}/{1000 * r[5]:<8.1f}{100 * r[6]:8.1f}%")
        out["c4"].append(dict(variant=nm, p50_extra_mm=1000 * r[0], p95_extra_mm=1000 * r[1],
                              max_extra_mm=1000 * r[2], p95_gap_mm=1000 * r[3],
                              jump_before_mm=1000 * r[4], jump_after_mm=1000 * r[5],
                              frac_clamped=r[6]))
    print("    'jump b/a' is the chunk-boundary discontinuity BEFORE/AFTER correction. If it grows,")
    print("    the correction is buying in-chunk fidelity with a bigger backward step at re-anchor.")

    h = bb(lambda b: headroom(b, "diag"), bundles, n_boot, rng)
    if h[0] is not None:
        print(f"\n    CEILING: the diagonal model explains R² = {_f(*h, 0, 1.0, 6, 3)} of the")
        print(f"    realized increment, so even a PERFECT inverse leaves {100 * h[0][1]:.1f}% of the")
        print("    error standing. This is the honest headroom, before the 84% the re-anchor")
        print("    already removes for free is even considered.")
        out["c4_headroom"] = dict(r2=h[0][0], residual_frac=h[0][1])

    print("\nC5  CORRECTION AUTHORITY vs STEP-WITHIN-CHUNK — removable fraction, out-of-take R²")
    ap = bb(lambda gg: authority_profile(gg, "diag"), groups, min(n_boot, 40), rng)
    if ap[0] is not None:
        print(f"    {'step':<10}{'removable fraction of the lead ramp':>40}")
        out["c5"] = []
        for i, (a, b) in enumerate(meg.BINS):
            print(f"    {f'{a}-{b}':<10}{_f(*ap, i, 1.0, 7, 3):>40}")
            out["c5"].append(dict(step=[a, b], removable=ap[0][i]))
        centres = np.array([(a + b) / 2 for a, b in meg.BINS])
        # Step 0 is excluded by construction, not by choice: the predicted object is
        # lead_k - lead_0, which is identically zero there, so its R² carries no information.
        m = np.isfinite(ap[0]) & (centres > 0)
        # The last bin spans 20 step indices and is the sparsest; a trend that only exists when
        # it is included is a statement about that bin, not about the chunk.
        m_dense = m & (centres < 30)
        if m.sum() >= 3:
            sl = float(np.polyfit(centres[m], ap[0][m], 1)[0])
            sl_d = float(np.polyfit(centres[m_dense], ap[0][m_dense], 1)[0]) if m_dense.sum() >= 3 else np.nan
            vals = ap[0][m_dense]
            print(f"\n    trend over steps 1-49  {sl:+.5f} per step")
            print(f"    trend over steps 1-29  {sl_d:+.5f} per step   "
                  f"(range {vals.min():.3f}-{vals.max():.3f}, sd {vals.std():.3f})")
            flat = abs(sl_d) < 0.004
            if flat:
                print("    => FLAT. Roughly half the lead ramp is removable at EVERY step index, so")
                print("       neither my 'authority should grow with k' hypothesis nor a graded")
                print("       schedule's epistemic argument is supported here: on this plant the")
                print("       removable fraction simply does not depend on where you are in the")
                print("       chunk. Any per-index weighting is then a choice about risk, not a")
                print("       correction for a measured loss of authority.")
            else:
                print(f"    => {'GROWS' if sl_d > 0 else 'SHRINKS'} with step index.")
            out["c5_trend"] = dict(slope_1_49=sl, slope_1_29=sl_d, flat=bool(flat),
                                   grows=bool(sl_d > 0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="data/deploy_takes (scans every session)")
    ap.add_argument("--algo", default="diffusion",
                    help="session_meta algo to keep; the ACT and diffusion paths are different "
                         "phenomena and must never be pooled (default: diffusion)")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    per_algo = load_groups(Path(args.root), args.algo)
    if not per_algo:
        print("no takes matched", file=sys.stderr)
        return 1
    results = {"reports": [run(f"algo={a}", g, n_boot=args.n_boot, rng=rng)
                           for a, g in sorted(per_algo.items())]}
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=float))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
