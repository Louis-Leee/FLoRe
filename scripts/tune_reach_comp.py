#!/usr/bin/env python3
"""Tune / audit reach compensation against RECORDED deploy takes — no robot, no bridges.

WHY THIS EXISTS
---------------
``--reach-ki`` adds un-asked-for reach to a physical arm, so the gain and the clamp should be
chosen from data rather than dialled into a live rig. Every recorded take already holds both
series the compensator needs — the published wire and the measured TCP (each prediction's
``anchor_pos`` IS the measured TCP at that instant, in the model frame, so the wire-vs-TCP lever
offset cancels) — so a take can answer two questions offline:

1. **Is the shortfall real on this take, and how big?** (the diagnosis)
2. **What would the compensator have published, and does it stay inside the clamp?** (the audit)

WHAT IT CANNOT TELL YOU
-----------------------
It replays the compensator against the shortfall the robot ACTUALLY had. It cannot predict how the
arm would have responded to the corrected target — that needs the robot, because the low level's
response to a larger lead is exactly the unknown here. So read the ``extra reach`` column as "how
much more the arm would have been asked for", NOT as "how much closer it would have got". The
clamp column is the safety-relevant one: it bounds the ask no matter what the arm does.

USAGE
-----
    .venv/bin/python scripts/tune_reach_comp.py data/deploy_takes/<session>
    .venv/bin/python scripts/tune_reach_comp.py data/deploy_takes/<session>/take_000 --ki 0.1 0.15 0.3

Prints a per-take diagnosis then a gain sweep. ``--json`` emits the same numbers for a report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flore.s1_policy.serving.hl_reach import ReachCompensator, ReachConfig  # noqa: E402
from flore.s1_policy.serving.hl_take import DeployTake, find_takes  # noqa: E402


def _series(take: DeployTake, role: str = "left", *, ensembled: bool | None = None):
    """(commanded, measured) Nx3 in the MODEL frame, one row per recorded prediction.

    ``measured`` is that prediction's own anchor — where the arm actually was.

    ``commanded`` must be **what the rig would really publish**, and that differs by cadence:

    * ACT temporal ensembling (``n_exec == 1`` on every prediction — the rig re-queries every
      step and publishes ``_wire_from_step(blend...)``): the published target is the
      ``exp(-k*i)``-weighted BLEND of every still-live prediction, which sits ~2.7 cm ahead of
      the arm on this data. Reconstructed here by replaying :class:`TemporalEnsembler` over the
      stored chunks — the same class, same ``k``, so this is the actual published series.
    * chunk cadence: step ``i`` of the newest chunk, whose step 1 is only ~0.7 mm from the
      anchor by construction (step 0 IS the anchor).

    Getting this wrong makes the whole audit meaningless — measuring against ``chunk[1]`` on an
    ensembled take reports a 0.07 cm shortfall for a rig that really had 2.66 cm, i.e. it would
    recommend a dead zone that gates the compensator off exactly when it is needed.
    """
    idx = 0
    chunks, meas = [], []
    for p in take.predictions:
        if p.anchor_pos is None or p.chunk is None:
            continue
        try:
            chunk = take.decoded_chunk_positions(p, role)
        except Exception:  # noqa: BLE001 — a replay-sourced prediction has no decodable chunk
            continue
        if chunk.shape[0] < 2:
            continue
        chunks.append(chunk)
        meas.append(np.asarray(p.anchor_pos[idx], dtype=np.float64))
    if not chunks:
        return np.zeros((0, 3)), np.zeros((0, 3)), False
    if ensembled is None:
        # The recorder writes n_exec per prediction; the ensembled cadence is 1-step-per-query.
        n_exec = [int(p.n_exec) for p in take.predictions if p.n_exec is not None]
        ensembled = bool(n_exec) and all(n == 1 for n in n_exec) and len(chunks) > 2
    if ensembled:
        from flore.s1_policy.model.act.temporal_ensemble import TemporalEnsembler

        H = chunks[0].shape[0]
        # Orientation is irrelevant to a POSITION audit and the compensator never touches it, so
        # feed a constant valid rot6d rather than reconstructing per-step rotations.
        rot = np.tile(np.array([1.0, 0, 0, 0, 1, 0]), (H, 1))
        ens = TemporalEnsembler(H, k=0.01)
        cmd = np.asarray([ens.add(c, rot)["position"] for c in chunks])
    else:
        cmd = np.asarray([c[1] for c in chunks])
    return cmd, np.asarray(meas), bool(ensembled)


def diagnose(cmd: np.ndarray, meas: np.ndarray) -> dict:
    """The shortfall the arm actually had on this take."""
    if len(cmd) == 0:
        return {}
    short = cmd - meas
    n = np.linalg.norm(short, axis=1)
    mean_vec = short.mean(axis=0)
    return {
        "n": int(len(cmd)),
        "shortfall_p50_cm": float(np.percentile(n, 50) * 100),
        "shortfall_p95_cm": float(np.percentile(n, 95) * 100),
        # A large PERSISTENT component is the signature the compensator targets: a bias the arm
        # never closes, as opposed to zero-mean tracking jitter.
        "persistent_cm": float(np.linalg.norm(mean_vec) * 100),
        "persistent_xyz_cm": [float(v * 100) for v in mean_vec],
        "fluctuating_rms_cm": float(np.sqrt(((short - mean_vec) ** 2).sum(1).mean()) * 100),
    }


def audit(cmd: np.ndarray, meas: np.ndarray, cfg: ReachConfig,
          force_n: np.ndarray | None = None) -> dict:
    """Replay the compensator over the take and report what it would have published.

    ``force_n`` (per-step measured contact magnitude) must be supplied when the take has it, or
    the audit would over-report accumulation: the contact cut-out is what stops a blocked hand's
    shortfall being integrated into a harder press, and on real takes it fires on a large
    fraction of the otherwise-accumulating steps.
    """
    comp = ReachCompensator(cfg)
    extra = np.empty(len(cmd))
    for i in range(len(cmd)):
        f = None if force_n is None else float(force_n[i])
        extra[i] = float(np.linalg.norm(comp.compensate(cmd[i], meas[i], f) - cmd[i]))
    st = comp.stats()
    return {
        "ki": cfg.ki,
        "clamp_cm": cfg.clamp_m * 100,
        "gate_cm": cfg.gate_m * 100,
        "extra_p50_cm": float(np.percentile(extra, 50) * 100),
        "extra_p95_cm": float(np.percentile(extra, 95) * 100),
        "extra_max_cm": float(extra.max() * 100),
        "within_clamp": bool(extra.max() <= cfg.clamp_m + 1e-9),
        "frac_saturated": float((extra >= cfg.clamp_m - 1e-9).mean()),
        "n_accumulated": st["n_applied"],
        "n_tracking_fine": st["n_gated"],
        "n_contact_blocked": st["n_contact_blocked"],
    }


def _force_series(take: DeployTake, n: int) -> np.ndarray | None:
    """Per-prediction LEFT contact magnitude (N), resampled onto the prediction count.

    The measured rows and the predictions run on the same take-relative clock but at different
    rates, so align by fraction-of-take (both are ordered) rather than inventing a join.
    """
    from flore.s1_policy.serving.hl_record import EXEC_COLS

    ec = {name: i for i, name in enumerate(EXEC_COLS)}
    ex = take.executed
    if ex.size == 0 or n == 0:
        return None
    try:
        f = ex[:, [ec["fmeas_lx"], ec["fmeas_ly"], ec["fmeas_lz"]]]
    except (KeyError, IndexError):
        return None
    mag = np.linalg.norm(f, axis=1)
    if not np.isfinite(mag).any():
        return None
    mag = np.where(np.isfinite(mag), mag, 0.0)
    idx = np.clip((np.linspace(0.0, 1.0, n) * (len(mag) - 1)).astype(int), 0, len(mag) - 1)
    return mag[idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a session dir (audits every take) or a single take dir")
    # Defaults MIRROR scripts/hl_serving.py's --reach-* defaults on purpose: the point of this
    # tool is to predict what that launcher would do, so a different default here would audit a
    # configuration nobody runs.
    ap.add_argument("--ki", type=float, nargs="+", default=[0.02, 0.05, 0.1],
                    help="integral gains to sweep (default 0.02 0.05 0.1)")
    ap.add_argument("--clamp", type=float, default=0.03, help="clamp (m), default 0.03")
    ap.add_argument("--stall-frac", type=float, default=0.5,
                    help="THE GATE: accumulate only while the arm did less than this fraction of "
                         "the commanded motion over --window steps (default 0.5)")
    ap.add_argument("--window", type=int, default=9,
                    help="steps the realized fraction is measured over (default 9 = 0.3 s)")
    ap.add_argument("--gate", type=float, default=0.0,
                    help="OPTIONAL secondary distance ceiling (m), default 0 = off")
    ap.add_argument("--decay", type=float, default=0.995, help="per-step leak, default 0.995")
    ap.add_argument("--contact-n", type=float, default=1.5,
                    help="contact cut-out (N), default 1.5 — above this the residual decays")
    ap.add_argument("--role", default="left")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.path)
    takes = ([root] if os.path.isfile(os.path.join(root, "take_meta.json"))
             else [str(p) for p in find_takes(root)])
    if not takes:
        print(f"no takes found under {root}", file=sys.stderr)
        return 2

    report = []
    for td in takes:
        try:
            take = DeployTake(td)
        except Exception as e:  # noqa: BLE001 — a partial take should not abort the sweep
            print(f"{os.path.basename(td)}: unreadable ({e})", file=sys.stderr)
            continue
        cmd, meas, ensembled = _series(take, args.role)
        if len(cmd) == 0:
            print(f"{os.path.basename(td)}: no model-sourced predictions with a decodable chunk",
                  file=sys.stderr)
            continue
        d = diagnose(cmd, meas)
        entry = {"take": os.path.basename(td), "label": take.meta.get("label"),
                 "cadence": "ensembled" if ensembled else "chunk",
                 "diagnosis": d, "sweep": []}
        fser = _force_series(take, len(cmd))
        entry["has_force"] = fser is not None
        for ki in args.ki:
            entry["sweep"].append(audit(cmd, meas, ReachConfig(
                ki=ki, clamp_m=args.clamp, stall_frac=args.stall_frac,
                window=args.window, gate_m=args.gate, decay=args.decay,
                contact_n=args.contact_n), fser))
        report.append(entry)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"\n=== SHORTFALL DIAGNOSIS ({len(report)} take(s)) ===")
    print("  the gap between what the wire asked for and where the arm was, per prediction.")
    print("  'persistent' is the part that never closes — what compensation targets.\n")
    print(f"{'take':10s} {'cadence':>10s} {'n':>5s} {'p50':>8s} {'p95':>8s} "
          f"{'persist':>9s} {'fluct':>8s}")
    for e in report:
        d = e["diagnosis"]
        print(f"{e['take']:10s} {e['cadence']:>10s} {d['n']:5d} "
              f"{d['shortfall_p50_cm']:7.2f}cm {d['shortfall_p95_cm']:7.2f}cm "
              f"{d['persistent_cm']:8.2f}cm {d['fluctuating_rms_cm']:7.2f}cm")
    # NOTE the gate is NOT a distance: it keys on whether the arm MOVED (realized fraction over
    # --window steps). The shortfall columns above are diagnosis only — on this data the standing
    # lead is ~2.5 cm whether the arm is tracking or frozen (corr with realized fraction ~ +0.08),
    # which is exactly why an earlier distance-based gate was wrong.

    print(f"\n=== GAIN SWEEP (clamp {args.clamp * 100:g} cm, accumulate only while the arm did "
          f"<{args.stall_frac:g} of the ask over {args.window} steps, decay {args.decay:g}, "
          f"contact cut-out {args.contact_n:g} N) ===")
    print("  'extra reach' = how much MORE the arm would have been asked for. It cannot say how")
    print("  much closer the arm would have got — that needs the robot.\n")
    print(f"{'take':10s} {'ki':>5s} {'extra p50':>10s} {'p95':>8s} {'max':>8s} "
          f"{'clamp ok':>9s} {'saturated':>10s} {'accum/tracking/contact':>23s}")
    for e in report:
        for s in e["sweep"]:
            print(f"{e['take']:10s} {s['ki']:5.2f} {s['extra_p50_cm']:9.2f}cm "
                  f"{s['extra_p95_cm']:7.2f}cm {s['extra_max_cm']:7.2f}cm "
                  f"{'yes' if s['within_clamp'] else 'NO':>9s} "
                  f"{s['frac_saturated'] * 100:9.1f}% "
                  f"{s['n_accumulated']:6d}/{s['n_tracking_fine']:5d}/"
                  f"{s['n_contact_blocked']:<6d}")
    if any(not e.get("has_force") for e in report):
        print("\n  ⚠ some takes carry no usable F_meas — their audit assumes NO contact, so it")
        print("    OVER-reports accumulation (the contact cut-out could not fire).")
    print("\nPick the smallest ki whose extra reach covers the 'persist' column above; the clamp")
    print("is the hard bound on what the arm can be asked for beyond the model's own target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
