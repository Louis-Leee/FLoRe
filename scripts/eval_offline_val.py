#!/usr/bin/env python3
"""Score a checkpoint's OFFLINE inference over EVERY anchor of its validation split.

    .venv/bin/python scripts/eval_offline_val.py checkpoints/umi_5pt_wipe_board_new/h50 \
        --out data/eval_offline/wipe_board_new_h50.json
    .venv/bin/python scripts/eval_offline_val.py --table data/eval_offline/*.json

One inference per validation anchor (= every camera frame of the last ``hl_data.val_last_episodes``
episodes that HLDataset keeps as a usable anchor, exactly the set the trainer's ``val/*`` draws from),
through the SAME ``*InferenceSession`` the offline viewer and ``--self-test`` use: the checkpoint's own
dataset knobs (cameras, history, glitch filter), ``validate_model_layout``, the sha-checked weights, the
bundled norm stats, DDIM at the checkpoint's own step count, one fixed seed per anchor. So a number here
is the number the viewer shows for that anchor, summed over the split.

Errors are against the dataset's own target chunk, per role and per action step, in physical units:
``position_cm`` = |Δp| of the keypoint, ``rotation_deg`` = geodesic angle of the rot6d, ``gripper_mm`` =
|Δ opening|, ``force_n`` = |ΔF| (3-vector) and ``torque_nm`` = |Δτ| in the Bota sensor frame -- the
definitions of ``inference.common._physical_metrics`` / ``training.metrics.physical_trajectory_sums``,
and every anchor's aggregate is asserted equal to the viewer's ``result.metrics`` as it is accumulated.
Only supervised entries count (the action mask: an episode that ends inside the chunk masks the tail,
a hand a layout does not carry has no columns at all).

⚠️ Whole-chunk means are NOT comparable across action horizons: step 49 is further into the future than
step 29. The JSON therefore also carries ``first_k`` prefix means (k = 1, 16, 30, 50, 100 up to H), the
per-step curve, and per-role means; compare an H=30 against an H=50 run on ``first_30``, and read the
robot-facing number off ``first_16`` (the executed window). The .npz beside the JSON holds every
per-(anchor, step) error so any other statistic can be computed without re-sampling.

The action-horizon pin (``FUJI_ACTION_HORIZON``) is read off the checkpoint and exported BEFORE the
first fuji import, the launchers' contract; a conflicting value already in the environment is refused.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ALGO_BLOCK = {"diffusion": "diffusion", "flow": "model", "act": "act"}
METRICS = ("position_cm", "rotation_deg", "gripper_mm", "force_n", "torque_nm")
#: the viewer's key for each of ours (identical definitions; only the rotation key is spelled differently)
VIEWER_KEY = {"position_cm": "position_cm", "rotation_deg": "rotation_geodesic_deg", "gripper_mm": "gripper_mm",
              "force_n": "force_n", "torque_nm": "torque_nm"}
PREFIXES = (1, 16, 30, 50, 100)


def pin_horizon(raw: dict) -> int:
    algo = str(raw.get("algo") or "diffusion")
    block = raw.get(ALGO_BLOCK[algo]) or {}
    horizon = block.get("action_horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise SystemExit(f"config.json {ALGO_BLOCK[algo]}.action_horizon={horizon!r} is not a positive integer")
    have = os.environ.get("FUJI_ACTION_HORIZON")
    if have not in (None, "", str(horizon)):
        raise SystemExit(f"FUJI_ACTION_HORIZON={have} in the environment but the checkpoint declares {horizon}")
    os.environ["FUJI_ACTION_HORIZON"] = str(horizon)
    return horizon


def open_session(checkpoint: Path, algo: str, device: str, num_steps: int | None):
    if algo == "diffusion":
        from fuji.s1_policy.inference.diffusion import DiffusionInferenceSession as cls
    elif algo == "flow":
        from fuji.s1_policy.inference.flow import FlowInferenceSession as cls
    elif algo == "act":
        from fuji.s1_policy.inference.act import ACTInferenceSession as cls
    else:
        raise SystemExit(f"unknown algo {algo!r}")
    return cls(checkpoint, device=device, num_steps=num_steps)


def geodesic_deg(rot6d_a: np.ndarray, rot6d_b: np.ndarray) -> np.ndarray:
    from fuji.utils.rot6d import rot6d_to_rotmat

    ra, rb = rot6d_to_rotmat(rot6d_a), rot6d_to_rotmat(rot6d_b)
    rel = np.einsum("...ij,...kj->...ik", ra, rb)
    cosine = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def role_errors(rt) -> dict[str, np.ndarray]:
    """Per-step error of one role, NaN where the target is not supervised. Keys follow what the role carries."""
    out: dict[str, np.ndarray] = {}
    pos = np.linalg.norm(rt.predicted_position - rt.target_position, axis=-1) * 100.0
    out["position_cm"] = np.where(rt.position_valid.all(axis=-1), pos, np.nan)
    rot = geodesic_deg(rt.predicted_rot6d, rt.target_rot6d)
    out["rotation_deg"] = np.where(rt.rotation_valid.all(axis=-1), rot, np.nan)
    if rt.predicted_gripper_m is not None:
        grip = np.abs(rt.predicted_gripper_m - rt.target_gripper_m) * 1000.0
        out["gripper_mm"] = np.where(rt.gripper_valid, grip, np.nan)
    if rt.predicted_wrench is not None:
        d = rt.predicted_wrench - rt.target_wrench
        out["force_n"] = np.where(rt.wrench_valid[:, :3].all(axis=-1), np.linalg.norm(d[:, :3], axis=-1), np.nan)
        out["torque_nm"] = np.where(rt.wrench_valid[:, 3:].all(axis=-1), np.linalg.norm(d[:, 3:], axis=-1), np.nan)
    return out


def stats(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "rmse": float(np.sqrt(np.mean(v * v))), "p90": float(np.percentile(v, 90)), "max": float(v.max())}


def summarize(errors: dict[str, dict[str, np.ndarray]], horizon: int) -> dict:
    """errors[metric][role] is (N, H) with NaN where unsupervised."""
    out: dict = {}
    for metric, by_role in errors.items():
        if not by_role:
            continue
        allv = np.concatenate([a.reshape(-1) for a in by_role.values()])
        entry = {"all": stats(allv), "per_role": {r: stats(a) for r, a in by_role.items()}}
        stacked = np.concatenate(list(by_role.values()), axis=0)          # (roles*N, H)
        with np.errstate(invalid="ignore"):
            entry["per_step_mean"] = [float(x) if np.isfinite(x) else None for x in np.nanmean(stacked, axis=0)]
        entry["first_k"] = {}
        for k in PREFIXES:
            if k <= horizon:
                entry["first_k"][str(k)] = stats(stacked[:, :k])
        out[metric] = entry
    return out


def run(args) -> int:
    checkpoint = Path(args.checkpoint)
    raw = json.loads((checkpoint / "config.json").read_text())
    horizon = pin_horizon(raw)
    algo = str(raw.get("algo") or "diffusion")
    session = open_session(checkpoint, algo, args.device, args.num_steps)
    try:
        source = session.source
        refs = [r for r in source.refs if r.split == args.split]
        if args.limit:
            refs = refs[: args.limit]
        if not refs:
            raise SystemExit(f"no {args.split} anchors")
        episodes: dict[int, int] = {}
        for r in refs:
            episodes[r.global_episode_id] = episodes.get(r.global_episode_id, 0) + 1
        print(f"[eval] {checkpoint}  algo={algo} layout={source.layout.mode} H={horizon} "
              f"sampler_steps={session.num_steps} seeds={args.seeds}", flush=True)
        print(f"[eval] {args.split} anchors: {len(refs)} over episodes {episodes} "
              f"(dataset {source.dataset.episode_count} episodes, {len(source.refs)} anchors)", flush=True)

        errors: dict[str, dict[str, list[np.ndarray]]] = {m: {} for m in METRICS}
        anchors, failures, t0 = [], [], time.time()
        for n, ref in enumerate(refs):
            for seed in args.seeds:
                try:
                    res = session.infer(ref.dataset_index, seed=seed)
                except (FloatingPointError, ValueError) as exc:
                    failures.append({"dataset_index": ref.dataset_index, "seed": seed, "error": str(exc)})
                    continue
                check: dict[str, list[np.ndarray]] = {m: [] for m in METRICS}
                for rt in res.roles:
                    for metric, arr in role_errors(rt).items():
                        errors[metric].setdefault(rt.role, []).append(arr)
                        check[metric].append(arr)
                # The viewer's own per-sample aggregate, asserted equal: same masks, same units.
                for metric, arrs in check.items():
                    mine = np.concatenate(arrs) if arrs else np.array([])
                    mine = float(np.nanmean(mine)) if np.isfinite(mine).any() else math.nan
                    theirs = float(res.metrics[VIEWER_KEY[metric]])
                    if not ((math.isnan(mine) and math.isnan(theirs)) or math.isclose(mine, theirs, rel_tol=1e-6, abs_tol=1e-7)):
                        raise AssertionError(f"{metric} disagrees with the viewer at anchor {ref.dataset_index}: "
                                             f"{mine!r} vs {theirs!r}")
                anchors.append((ref.dataset_index, ref.global_episode_id, ref.episode_frame, seed))
            if n % args.log_every == 0 or n == len(refs) - 1:
                el = time.time() - t0
                print(f"[eval] {n + 1}/{len(refs)}  {el:.0f} s  ({el / (n + 1) * 1000:.0f} ms/anchor, "
                      f"eta {el / (n + 1) * (len(refs) - n - 1):.0f} s)", flush=True)
        wall = time.time() - t0
        stacked = {m: {r: np.stack(v) for r, v in by_role.items()} for m, by_role in errors.items() if by_role}
        summary = summarize(stacked, horizon)
        prov = raw.get("_provenance", {})
        meta = {
            "checkpoint": str(checkpoint), "model_sha256": prov.get("model_sha256"),
            "algo": algo, "layout": source.layout.mode, "roles": list(source.layout.roles),
            "force_roles": list(source.layout.force_roles), "gripper_roles": list(source.layout.gripper_roles),
            "action_horizon": horizon, "sampler_steps": session.num_steps, "seeds": args.seeds,
            "split": args.split, "val_last_episodes": source.val_last_episodes,
            "zarr": [str(p) for p in source.resolved_paths], "episodes": {str(k): v for k, v in episodes.items()},
            "n_anchors": len(refs), "n_inferences": len(anchors), "failures": failures,
            "wall_s": wall, "ms_per_inference": wall / max(1, len(anchors)) * 1000.0,
            "limit": args.limit, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        out = Path(args.out) if args.out else REPO / "data/eval_offline" / f"{checkpoint.parent.name}_{checkpoint.name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"meta": meta, "summary": summary}, indent=1))
        npz = {f"{m}__{r}": a for m, by_role in stacked.items() for r, a in by_role.items()}
        npz["anchors"] = np.asarray(anchors, dtype=np.int64)
        npz["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(out.with_suffix(".npz"), **npz)
        print(f"[eval] wrote {out} (+ .npz)  {len(anchors)} inferences in {wall:.0f} s, {len(failures)} failures")
        print_table([out])
        return 0
    finally:
        session.close()


def _fmt(s: dict, key: str = "mean", nd: int = 2) -> str:
    return "—" if not s or s.get("n", 0) == 0 else f"{s[key]:.{nd}f}"


def print_table(paths: list[Path]) -> None:
    rows = [json.loads(Path(p).read_text()) for p in paths]
    print("\n| checkpoint | layout | H | anchors | pos cm | rot deg | grip mm | force N | torque N·m |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for d in rows:
        m, s = d["meta"], d["summary"]
        name = f"{Path(m['checkpoint']).parent.name}/{Path(m['checkpoint']).name}"
        cells = [_fmt(s.get(k, {}).get("all"), nd=3 if k == "torque_nm" else 2) for k in METRICS]
        print(f"| {name} | {m['layout']} | {m['action_horizon']} | {m['n_anchors']} | " + " | ".join(cells) + " |")
    print("\nfirst 16 steps (the executed window) — mean:")
    print("| checkpoint | pos cm | rot deg | grip mm | force N | torque N·m |")
    print("|---|---:|---:|---:|---:|---:|")
    for d in rows:
        m, s = d["meta"], d["summary"]
        name = f"{Path(m['checkpoint']).parent.name}/{Path(m['checkpoint']).name}"
        cells = [_fmt(s.get(k, {}).get("first_k", {}).get("16"), nd=3 if k == "torque_nm" else 2) for k in METRICS]
        print(f"| {name} | " + " | ".join(cells) + " |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", help="checkpoint dir (config.json + model.safetensors + norm_stats.json)")
    ap.add_argument("--table", nargs="+", metavar="JSON", help="print the summary table of these result files and exit")
    ap.add_argument("--out", help="result JSON (default data/eval_offline/<parent>_<name>.json); .npz written beside it")
    ap.add_argument("--split", default="validation", choices=("validation", "train"))
    ap.add_argument("--seeds", default="0", help="comma-separated sampling seeds, each run on every anchor")
    ap.add_argument("--num-steps", type=int, default=None, help="sampler steps (default: the checkpoint's)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="first N anchors only (smoke)")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args(argv)
    if args.table:
        print_table([Path(p) for p in args.table])
        return 0
    if not args.checkpoint:
        ap.error("a checkpoint dir or --table is required")
    args.seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
