#!/usr/bin/env python3
"""Profile the FORCE distribution: training support vs deploy reality. Robot-free, CPU-only.

WHY THIS EXISTS
---------------
The position channel is robust and the force channel is brittle, and "the policy meets a force
it has not seen" was a hunch with no number attached. You cannot target a coverage gap you have
not measured, and you cannot tell an *augmentation* problem from a *parameterization* problem by
looking at loss curves: both show up as a force head that is quietly wrong at deploy.

This script measures the gap on data already on disk. It needs NO zarr — the training force
support is read out of each checkpoint's ``norm_stats.json`` (quantile stats are a property of
the dataset, so two algos trained on one session give bit-identical force stats), and the deploy
side comes from ``data/deploy_takes/``. Each deploy session is matched to its training support
through ``session_meta.json``'s ``frame_seam``, which names the dataset the checkpoint was fit on.

WHAT IT MEASURES
----------------
F0  the training force support per dataset, and the pairwise 3-D box IoU between sessions —
    i.e. does a new session even overlap the last one's force coverage?
F1  deploy-vs-training OOD rate, JOINT (outside the [q01,q99] box) against MARGINAL (|z| > 3).
    If they are equal the excursion is a MAGNITUDE problem; if joint >> marginal it is a
    DIRECTION/combination problem, and those want different augmentations.
F2  octant occupancy of the contact force direction, per session and pooled. A session that
    lives in one octant cannot teach any other octant.
F3  **the load-bearing one.** Split the deploy stream ``F = b + d`` into a causal 1 s baseline
    ``b`` (the tare gauge — what an anchor-relative force parameterization would subtract) and
    a transient ``d`` (the contact event it would keep), then ask what share of today's OOD is
    in ``b`` alone. That share is removable by re-parameterization, for free, with no new data.
F4  wrench self-consistency: fit ``T = r x F`` per window and report the residual as a fraction
    of ``|T|``. This is BOTH the proposed validity filter and the admissibility gate for the
    rotation/scaling augmentations — a wrench that is not a single-point contact must not be
    rotated or rescaled as if it were.
F5  the planner's force GAIN: regress the predicted chunk's force on the observed force bin, per
    horizon step. Under DECISION-T0 step 0 lands ON t0 and its force target IS ~the measured
    force, so a tracking head must show slope ~1 at k=0. Anything less is shrinkage, and
    shrinkage is a target-distribution problem, not an extrapolation problem.
F6  tare quality: where each take's pre-contact baseline sits. A take tared under load carries a
    wrong gauge, which is an unfixable label error under an absolute parameterization.

TWO TRAPS THIS SCRIPT AVOIDS, both of which produced wrong numbers on the first pass:

* ``executed.npy``'s ``fmeas_*`` columns are **post-gate** — the deploy force gate zeroes them
  below its threshold, so a distribution built from them reads a spurious spike at exactly 0 and
  a p50 of 0.00. All distribution work here uses ``force_meas.npy``, the raw sensor stream.
* a "quietest window" noise-floor search finds a GATED (all-zero) window and reports a noise
  floor of exactly 0. Windows with a zero-variance axis are rejected.

Complements ``measure_exec_gap.py``'s M8, which measures the *plant's* force gain (commanded ->
measured, ``g_F = 1.26``). F5 here measures the *planner's* (observed -> commanded). They compose:
the loop under-responds even though the plant over-delivers.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import numpy as np

LAB = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
#: Left-hand HL-v2 block is ``[pos 3][rot6d 6][grip 1][force 6]``; force lands at 10:16.
ACTION_FORCE = slice(10, 16)
#: ``force_meas.npy`` row is ``[t, *channels]``.
FM_FORCE = slice(1, 7)
#: The deploy force gate, N. Rows below it are not acted on, so direction stats are gated on it.
GATE_N = 1.5


# --------------------------------------------------------------------------- loading

def load_training_support(ckpt_glob: str) -> dict[str, dict[str, Any]]:
    """dataset basename -> force support, read from every checkpoint's ``norm_stats.json``."""
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(glob.glob(ckpt_glob)):
        try:
            ns = json.load(open(p))
        except (OSError, json.JSONDecodeError):
            continue
        fh = ns.get("force_history")
        if not fh:
            continue
        meta = ns.get("_meta", {})
        zp = meta.get("zarr_paths") or meta.get("dataset_knobs", {}).get("zarr_paths")
        if not zp:
            continue
        ds = os.path.basename(str(zp[0]))
        rec = {k: np.asarray(fh[k], float)[:6] for k in ("q01", "q99", "mean", "std")}
        rec["eps"] = meta.get("total_episodes")
        rec["anchors"] = meta.get("train_anchor_count")
        out[ds] = rec
    return out


def load_deploy_sessions(root: str) -> list[dict[str, Any]]:
    """One entry per deploy session: the raw force stream plus the dataset it was trained on."""
    out = []
    for sd in sorted(glob.glob(os.path.join(root, "*"))):
        smp = os.path.join(sd, "session_meta.json")
        if not os.path.exists(smp):
            continue
        try:
            sm = json.load(open(smp))
        except (OSError, json.JSONDecodeError):
            continue
        F, takes = [], []
        for td in sorted(glob.glob(os.path.join(sd, "take_*"))):
            fp = os.path.join(td, "force_meas.npy")
            if not os.path.exists(fp):
                continue
            a = np.load(fp)
            if a.ndim == 2 and a.shape[1] >= 7 and a.shape[0] > 0:
                F.append(a[:, FM_FORCE])
                takes.append(td)
        if not F:
            continue
        out.append(dict(sess=os.path.basename(sd),
                        ds=str(sm.get("frame_seam") or "").split("::")[0],
                        algo=sm.get("algo"), takes=takes, F=np.concatenate(F)))
    return out


def causal_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Causal running mean over ``n`` samples, expanding at the head (no look-ahead)."""
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    lo = np.maximum(np.arange(x.shape[0]) + 1 - n, 0)
    hi = np.arange(x.shape[0]) + 1
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


# --------------------------------------------------------------------------- F0

def f0_training_support(train: dict[str, dict[str, Any]]) -> None:
    print("=" * 100)
    print("F0  TRAINING FORCE SUPPORT, and how much consecutive sessions OVERLAP")
    print("=" * 100)
    print(f"{'dataset':42s} {'eps':>4s} " + "".join(f"{lab + '[q01,q99]':>17s}" for lab in LAB[:3]))
    for d in sorted(train):
        t = train[d]
        print(f"{d[:42]:42s} {str(t['eps']):>4s} "
              + "".join(f"  [{t['q01'][i]:6.2f},{t['q99'][i]:6.2f}]" for i in range(3)))

    uniq, seen = [], set()
    for d in sorted(train):
        key = tuple(np.round(train[d]["q01"], 3)) + tuple(np.round(train[d]["q99"], 3))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)

    def iou(a: dict, b: dict) -> float:
        lo = np.maximum(a["q01"][:3], b["q01"][:3])
        hi = np.minimum(a["q99"][:3], b["q99"][:3])
        inter = float(np.prod(np.maximum(hi - lo, 0.0)))
        va = float(np.prod(a["q99"][:3] - a["q01"][:3]))
        vb = float(np.prod(b["q99"][:3] - b["q01"][:3]))
        den = va + vb - inter
        return inter / den if den > 0 else 0.0

    print("\n3-D force-box IoU between distinct training sessions (1.0 = identical support):")
    print(f"{'':32s}" + "".join(f"{d.split('_')[1][5:][:8]:>9s}" for d in uniq))
    offs = []
    for a in uniq:
        row = [iou(train[a], train[b]) for b in uniq]
        offs += [v for b, v in zip(uniq, row) if b != a]
        print(f"{a.replace('session_', '')[:32]:32s}" + "".join(f"{v:9.2f}" for v in row))
    if offs:
        print(f"\noff-diagonal IoU: median={np.median(offs):.2f} min={min(offs):.2f} max={max(offs):.2f}")
        print("  -> a low median means each session teaches a DIFFERENT force region, so pooling")
        print("     sessions widens the support far more than adding takes to one session does.")


# --------------------------------------------------------------------------- F1

def f1_ood(sessions: list[dict], train: dict) -> None:
    print()
    print("=" * 100)
    print("F1  DEPLOY OOD RATE — is the excursion a MAGNITUDE or a COMBINATION problem?")
    print("=" * 100)
    print("   outside-box = outside the training [q01,q99] on >=1 axis  (JOINT)")
    print("   any|z|>3    = >3 training sigma on >=1 axis               (MARGINAL / magnitude)")
    print("   ratio ~ 1 => magnitude.  ratio >> 1 => the axis COMBINATION is what is novel.")
    print()
    print(f"{'deploy session':17s} {'rows':>7s} {'outside-box':>12s} {'any|z|>3':>10s} "
          f"{'any|z|>5':>9s} {'ratio':>7s}")
    print("-" * 100)
    tb = tm = tn = 0.0
    for s in sessions:
        t = train.get(s["ds"])
        if t is None:
            continue
        F = s["F"]
        z = np.abs((F - t["mean"]) / np.maximum(t["std"], 1e-9))
        box = float(((F < t["q01"]) | (F > t["q99"])).any(1).mean())
        m3 = float((z > 3).any(1).mean())
        m5 = float((z > 5).any(1).mean())
        tb += box * F.shape[0]
        tm += m3 * F.shape[0]
        tn += F.shape[0]
        print(f"{s['sess']:17s} {F.shape[0]:7d} {100 * box:11.1f}% {100 * m3:9.1f}% "
              f"{100 * m5:8.1f}% {box / max(m3, 1e-9):7.1f}x")
    if tn:
        print("-" * 100)
        print(f"{'ALL':17s} {int(tn):7d} {100 * tb / tn:11.1f}% {100 * tm / tn:9.1f}%")


# --------------------------------------------------------------------------- F2

def f2_direction(sessions: list[dict]) -> None:
    print()
    print("=" * 100)
    print(f"F2  DIRECTION COVERAGE — octant of the force direction, rows with |F| > {GATE_N} N")
    print("=" * 100)
    names = [f"{'+' if i & 4 else '-'}{'+' if i & 2 else '-'}{'+' if i & 1 else '-'}"
             for i in range(8)]
    print(f"{'deploy session':17s} {'n':>7s} " + "".join(f"{o:>7s}" for o in names))
    print("-" * 100)
    pool = np.zeros(8)
    for s in sessions:
        f = s["F"][:, :3]
        f = f[np.linalg.norm(f, axis=1) > GATE_N]
        if f.shape[0] < 50:
            continue
        idx = (((f[:, 0] > 0).astype(int) << 2) | ((f[:, 1] > 0).astype(int) << 1)
               | (f[:, 2] > 0).astype(int))
        h = np.bincount(idx, minlength=8)
        pool += h
        top = 100 * h.max() / f.shape[0]
        print(f"{s['sess']:17s} {f.shape[0]:7d} " + "".join(f"{100 * x / f.shape[0]:6.1f}%" for x in h)
              + (f"   <- {top:.0f}% in one octant" if top > 60 else ""))
    if pool.sum():
        print("-" * 100)
        print(f"{'ALL (pooled)':17s} {int(pool.sum()):7d} "
              + "".join(f"{100 * x / pool.sum():6.1f}%" for x in pool))
        print("\n  Each session concentrates in one or two octants while the POOL is spread: the")
        print("  union of sessions covers the sphere, but no single training set does.")


# --------------------------------------------------------------------------- F3

def f3_baseline_vs_transient(sessions: list[dict], train: dict, fs_hz: float) -> None:
    print()
    print("=" * 100)
    print("F3  WOULD RE-ANCHORING THE FORCE CHANNEL REMOVE THE OOD?")
    print("=" * 100)
    print("Split F = b + d:  b = causal 1 s mean (the tare GAUGE, what an anchor-relative force")
    print("parameterization subtracts), d = F - b (the contact TRANSIENT, what it keeps).")
    print()
    print(f"{'deploy session':17s} {'rows':>7s} {'raw-OOD':>9s} {'base-OOD':>9s} "
          f"{'trans-OOD':>10s} {'removable':>10s}")
    print("-" * 100)
    n_win = int(round(fs_hz))
    TR = TD = TN = 0
    for s in sessions:
        t = train.get(s["ds"])
        if t is None:
            continue
        raw = bas = tra = n = 0
        for td in s["takes"]:
            a = np.load(os.path.join(td, "force_meas.npy"))
            if a.ndim != 2 or a.shape[1] < 7 or a.shape[0] < n_win:
                continue
            F = a[:, FM_FORCE]
            b = causal_mean(F, n_win)
            sd = np.maximum(t["std"], 1e-9)
            raw += int((np.abs((F - t["mean"]) / sd) > 3).any(1).sum())
            bas += int((np.abs((b - t["mean"]) / sd) > 3).any(1).sum())
            tra += int((np.abs((F - b) / sd) > 3).any(1).sum())
            n += F.shape[0]
        if n < 500:
            continue
        TR += raw
        TD += tra
        TN += n
        print(f"{s['sess']:17s} {n:7d} {100 * raw / n:8.1f}% {100 * bas / n:8.1f}% "
              f"{100 * tra / n:9.1f}% {100 * (1 - tra / max(raw, 1)):9.0f}%")
    if TN:
        print("-" * 100)
        print(f"{'ALL':17s} {TN:7d} {100 * TR / TN:8.1f}% {'':9s} {100 * TD / TN:9.1f}% "
              f"{100 * (1 - TD / max(TR, 1)):9.0f}%")
        print("\n  'removable' = the share of today's OOD rows a baseline-relative force target")
        print("  would move back INSIDE the training support, with no new data and no augmentation.")


# --------------------------------------------------------------------------- F4

def _fit_moment_arm(F: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares ``r`` in ``T = r x F``; residual as a fraction of ``|T|``."""
    n = F.shape[0]
    A = np.zeros((3 * n, 3))
    A[0::3, 1] = F[:, 2]
    A[0::3, 2] = -F[:, 1]
    A[1::3, 0] = -F[:, 2]
    A[1::3, 2] = F[:, 0]
    A[2::3, 0] = F[:, 1]
    A[2::3, 1] = -F[:, 0]
    b = T.reshape(-1)
    r, *_ = np.linalg.lstsq(A, b, rcond=None)
    return r, float(np.linalg.norm(A @ r - b) / max(np.linalg.norm(b), 1e-9))


def f4_wrench_consistency(sessions: list[dict], fs_hz: float) -> None:
    print()
    print("=" * 100)
    print("F4  WRENCH SELF-CONSISTENCY — is T explained by r x F for one stable contact point?")
    print("=" * 100)
    print("Per 1 s window with real contact. Small residual => a clean single-point contact, which")
    print("is (a) a validity FILTER and (b) the admissibility gate for rotating/rescaling a wrench.")
    print()
    W = int(round(fs_hz))
    res_all: list[float] = []
    r_all: list[float] = []
    for s in sessions:
        F = s["F"]
        rs, rr = [], []
        for st in range(0, F.shape[0] - W, W):
            w = F[st:st + W]
            if np.linalg.norm(w[:, :3], axis=1).mean() < 1.0:
                continue
            if np.linalg.norm(w[:, 3:], axis=1).mean() < 1e-3:
                continue
            r, res = _fit_moment_arm(w[:, :3], w[:, 3:])
            rs.append(res)
            rr.append(float(np.linalg.norm(r)))
        if len(rs) < 3:
            continue
        res_all += rs
        r_all += rr
        print(f"{s['sess']:17s} n_win={len(rs):4d}  residual/|T| p50={np.percentile(rs, 50):5.2f} "
              f"p90={np.percentile(rs, 90):5.2f}   |r| p50={np.percentile(rr, 50) * 1000:6.1f} mm")
    if res_all:
        print("-" * 100)
        print(f"ALL n={len(res_all)}  residual/|T| p10={np.percentile(res_all, 10):.2f} "
              f"p50={np.percentile(res_all, 50):.2f} p90={np.percentile(res_all, 90):.2f}")
        print(f"  implied moment arm |r|: p50={np.percentile(r_all, 50) * 1000:.0f} mm "
              f"p90={np.percentile(r_all, 90) * 1000:.0f} mm")
        print("  A p10-to-p90 spread of several x is what makes the residual usable as a filter.")


# --------------------------------------------------------------------------- F5

def f5_planner_force_gain(root: str) -> None:
    print()
    print("=" * 100)
    print("F5  THE PLANNER'S FORCE GAIN — does the head reproduce the force it is shown?")
    print("=" * 100)
    print("DECISION-T0: chunk step 0 lands ON t0 and its force target IS ~the observed force bin,")
    print("so a tracking head must show slope ~1 at k=0. Less than that is SHRINKAGE.")
    print()
    X, Y = [], []
    skipped = 0
    for pd in sorted(glob.glob(os.path.join(root, "*", "take_*", "pred_*"))):
        cp, fp = os.path.join(pd, "chunk.npy"), os.path.join(pd, "force_history.npy")
        if not (os.path.exists(cp) and os.path.exists(fp)):
            continue
        C, Fh = np.load(cp), np.load(fp)
        if C.ndim != 2 or C.shape[1] < 16 or Fh.ndim != 2 or Fh.shape[1] < 6:
            continue
        if np.allclose(Fh, 0.0):           # force bridge silent / history invalid
            skipped += 1
            continue
        X.append(Fh[-1, :6])
        Y.append(C[:, ACTION_FORCE])
    if not X:
        print("no predictions with a force head found")
        return
    X = np.asarray(X)
    H = min(y.shape[0] for y in Y)
    Y = np.stack([y[:H] for y in Y])
    print(f"n={X.shape[0]} predictions (H={H}); {skipped} skipped with an all-zero force bin\n")
    print(f"{'k':>3s} {'t (s)':>6s} " + "".join(f"{a:>8s}" for a in LAB[:3]) + f" {'corr(Fz)':>9s}")
    print("-" * 50)
    ks = sorted(set(list(range(0, H, max(H // 10, 1))) + [H - 1]))
    for k in ks:
        sl = []
        for i in range(3):
            x, y = X[:, i], Y[:, k, i]
            A = np.vstack([x, np.ones_like(x)]).T
            s, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            sl.append(s)
        c = float(np.corrcoef(X[:, 2], Y[:, k, 2])[0, 1])
        print(f"{k:3d} {k / 30.0:6.2f} " + "".join(f"{v:8.3f}" for v in sl) + f" {c:9.3f}")
    flat = np.median(np.linalg.norm(Y[:, 0, :3] - Y[:, H - 1, :3], axis=1))
    print(f"\n  sd(predicted force xyz) / sd(observed force xyz) = "
          f"{np.linalg.norm(Y[:, 0, :3].std(0)) / max(np.linalg.norm(X[:, :3].std(0)), 1e-9):.2f}")
    print(f"  |chunk[0,F] - chunk[-1,F]| p50 = {flat:.3f} N  vs "
          f"sd(observed) = {np.linalg.norm(X[:, :3].std(0)):.3f} N")
    print("  A low slope at k=0 that decays smoothly, plus a nearly FLAT chunk, is a low-gain")
    print("  near-constant regressor: it does not extrapolate badly, it barely responds at all.")


# --------------------------------------------------------------------------- F6

def f6_tare(root: str, fs_hz: float) -> None:
    print()
    print("=" * 100)
    print("F6  TARE QUALITY — where does each take's pre-contact baseline sit?")
    print("=" * 100)
    n0 = max(int(round(0.25 * fs_hz)), 5)
    starts = []
    for td in sorted(glob.glob(os.path.join(root, "*", "take_*"))):
        fp = os.path.join(td, "force_meas.npy")
        if not os.path.exists(fp):
            continue
        a = np.load(fp)
        if a.ndim != 2 or a.shape[1] < 7 or a.shape[0] < 4 * n0:
            continue
        starts.append(a[:n0, FM_FORCE].mean(axis=0))
    if not starts:
        print("no takes with a usable force stream")
        return
    S = np.asarray(starts)
    print(f"n_takes={S.shape[0]}  (first {n0 / fs_hz:.2f} s of each take)")
    print("       " + "".join(f"{lab:>9s}" for lab in LAB))
    for nm, v in (("mean", S.mean(0)), ("std", S.std(0)), ("min", S.min(0)), ("max", S.max(0))):
        print(f"{nm:>6s} " + "".join(f"{x:9.3f}" for x in v))
    mag = np.linalg.norm(S[:, :3], axis=1)
    print(f"|F_start|: p50={np.percentile(mag, 50):.3f} p99={np.percentile(mag, 99):.3f} "
          f"max={mag.max():.3f} N")
    print("  A p50 near zero says the tare works; a large max says at least one take was tared")
    print("  UNDER LOAD, which is an unfixable label error while the force target is absolute.")


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("takes_root", nargs="?", default="data/deploy_takes",
                    help="directory of deploy sessions (default: data/deploy_takes)")
    ap.add_argument("--checkpoints", default="checkpoints/*/norm_stats.json",
                    help="glob for checkpoint norm_stats.json (default: checkpoints/*/norm_stats.json)")
    ap.add_argument("--force-hz", type=float, default=200.0,
                    help="sample rate of force_meas.npy, Hz (default: 200)")
    ap.add_argument("--only", default="", help="comma-separated subset of F0..F6 to run")
    args = ap.parse_args()

    train = load_training_support(args.checkpoints)
    sessions = load_deploy_sessions(args.takes_root)
    print(f"{len(train)} training datasets with force stats; {len(sessions)} deploy sessions "
          f"({sum(s['F'].shape[0] for s in sessions)} raw force rows)")
    matched = sum(1 for s in sessions if s["ds"] in train)
    print(f"{matched}/{len(sessions)} deploy sessions matched to a training support via frame_seam")

    want = {w.strip().upper() for w in args.only.split(",") if w.strip()}
    def run(tag: str) -> bool:
        return not want or tag in want

    if run("F0"):
        f0_training_support(train)
    if run("F1"):
        f1_ood(sessions, train)
    if run("F2"):
        f2_direction(sessions)
    if run("F3"):
        f3_baseline_vs_transient(sessions, train, args.force_hz)
    if run("F4"):
        f4_wrench_consistency(sessions, args.force_hz)
    if run("F5"):
        f5_planner_force_gain(args.takes_root)
    if run("F6"):
        f6_tare(args.takes_root, args.force_hz)


if __name__ == "__main__":
    main()
