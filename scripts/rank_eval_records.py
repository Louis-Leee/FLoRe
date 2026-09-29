#!/usr/bin/env python3
"""Aggregate eval_checkpoints_physical.py JSONL shards into a ranking table.

The eval is sharded across GPUs, so records for one run arrive in several files and
in shard order rather than step order. This merges them, sorts by step, and reports
the composite ranking plus the train/val generalization gap when both splits exist.

The composite is a z-score over the five physical metrics rather than a raw sum:
position_cm (~0.7) and torque_nm (~0.03) differ by more than an order of magnitude,
so a raw sum is effectively "rank by position and rotation only". Standardizing per
metric across the compared checkpoints gives each metric equal weight, which is what
"best overall physical accuracy" is meant to mean here.

Usage: rank_eval_records.py <dir_with_jsonl> [--metrics a,b,c]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("position_cm", "rotation_geodesic_deg", "gripper_mm", "force_n", "torque_nm")
SHORT = {"position_cm": "pos_cm", "rotation_geodesic_deg": "rot_deg", "gripper_mm": "grip_mm",
         "force_n": "force_N", "torque_nm": "torq_Nm"}


def step_of(name: str) -> int:
    """Sort key: step_NNNNNNNN by number, best_*/other names last but stable."""
    m = re.search(r"step_(\d+)", name)
    return int(m.group(1)) if m else -1


def load(dirpath: Path) -> dict[tuple[str, str], dict[str, dict]]:
    """-> {(run, split): {ckpt_name: record}}. Later records win, so a re-run of one
    shard supersedes the stale value instead of being double-counted."""
    out: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for path in sorted(dirpath.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ckpt = Path(rec["checkpoint"])
            out[(ckpt.parent.name, rec.get("split", "val"))][ckpt.name] = rec
    return out


def composite(records: dict[str, dict], metrics: tuple[str, ...]) -> dict[str, float]:
    """Mean z-score across metrics; lower is better. Zero-variance metrics drop out."""
    stats = {}
    for m in metrics:
        vals = [r[m] for r in records.values()]
        sd = statistics.pstdev(vals)
        stats[m] = (statistics.fmean(vals), sd)
    scores = {}
    for name, rec in records.items():
        zs = [(rec[m] - mu) / sd for m, (mu, sd) in stats.items() if sd > 0]
        scores[name] = statistics.fmean(zs) if zs else 0.0
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--metrics", default=",".join(METRICS))
    args = ap.parse_args()
    metrics = tuple(m for m in args.metrics.split(",") if m.strip())

    data = load(args.dir)
    if not data:
        print(f"no .jsonl records under {args.dir}")
        return 1

    for (run, split) in sorted(data):
        recs = data[(run, split)]
        scores = composite(recs, metrics)
        order = sorted(recs, key=step_of)
        n = {r["_samples"] for r in recs.values()}
        print(f"\n=== {run}  split={split}  checkpoints={len(recs)}  anchors={n} ===")
        head = "".join(f"{SHORT[m]:>10}" for m in metrics)
        print(f"{'checkpoint':<20}{head}{'composite_z':>13}")
        for name in order:
            row = "".join(f"{recs[name][m]:>10.4f}" for m in metrics)
            print(f"{name:<20}{row}{scores[name]:>13.4f}")

        print("  best per metric:")
        for m in metrics:
            best = min(recs, key=lambda k: recs[k][m])
            print(f"    {m:<24} {recs[best][m]:>10.5f}  <- {best}")
        rank = sorted(scores, key=lambda k: scores[k])
        print(f"  composite ranking (best first): {', '.join(rank[:5])}")
        print(f"  WORST composite: {rank[-1]} ({scores[rank[-1]]:.3f})")

    # Generalization gap, per run, only where both splits were scored.
    for run in sorted({r for r, _ in data}):
        tr, va = data.get((run, "train")), data.get((run, "val"))
        if not tr or not va:
            continue
        shared = sorted(set(tr) & set(va), key=step_of)
        if not shared:
            continue
        print(f"\n=== {run}  train/val generalization gap ({len(shared)} checkpoints) ===")
        print(f"{'checkpoint':<20}{'train_pos':>11}{'val_pos':>10}{'ratio':>8}"
              f"{'train_force':>13}{'val_force':>11}{'ratio':>8}")
        for name in shared:
            t, v = tr[name], va[name]
            pr = v["position_cm"] / t["position_cm"] if t["position_cm"] else float("nan")
            fr = v["force_n"] / t["force_n"] if t["force_n"] else float("nan")
            print(f"{name:<20}{t['position_cm']:>11.4f}{v['position_cm']:>10.4f}{pr:>8.2f}"
                  f"{t['force_n']:>13.4f}{v['force_n']:>11.4f}{fr:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
