#!/usr/bin/env python3
"""Rank S1 checkpoints by PHYSICAL trajectory error over the COMPLETE validation split.

Why this exists: ``val/objective`` is the training surrogate (for diffusion, the
epsilon-prediction MSE), and it rises monotonically while the sampled trajectory
error keeps improving. Selecting on it picks a demonstrably worse model. This
script samples each checkpoint the way deploy does and scores it with the same
``physical_trajectory_sums`` the trainer uses, so the numbers are comparable to
W&B's ``val/sample_*`` -- but over all validation anchors instead of 4.

Usage:
  .venv/bin/python scripts/eval_checkpoints_physical.py <ckpt_dir> [<ckpt_dir> ...]
  .venv/bin/python scripts/eval_checkpoints_physical.py --run checkpoints/hl_left_clean99_flow_dit
Options:
  --split S     "val" (default) or "train"; train quantifies the generalization gap
  --batches N   limit to N batches (default 0 = the entire split)
  --first-n-steps N   score only the first N action steps -- REQUIRED to compare
                checkpoints trained at different action horizons (see below)
  --seeds a,b   sampling seeds to average over (default 0)
  --json PATH   append one JSON record per checkpoint

One invocation scores every checkpoint against ONE dataloader and ONE norm_stats (both
taken from the FIRST target), so all targets in a single run must share a horizon and a
norm-stats file. Compare across horizons by running each horizon separately (with
``FUJI_ACTION_HORIZON`` pinned to match) and passing the SAME ``--first-n-steps`` to both.

Selecting a checkpoint is only ever valid on ``--split val``. ``--split train``
exists to measure how far val error sits above train error on the SAME sampler and
metric; a checkpoint that wins on train is by definition the most memorized one.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from fuji.s1_policy.data.hl_dataset import HLDataset
from fuji.s1_policy.data.layout import ACTION_SPAN_S, FORCE_HIST_SPAN_S, get_layout
from fuji.s1_policy.data.norm_stats import (
    load_norm_stats,
    normalize_quantile,
    unnormalize_quantile,
)
from fuji.s1_policy.inference.common import dataclass_config
from fuji.s1_policy.training.metrics import physical_trajectory_sums
from fuji.s1_policy.training.trainer import _move_batch, collate_hl_batch, isolated_rng
from fuji.utils.dataset_paths import resolve_dataset_paths

METRICS = ("position_cm", "rotation_geodesic_deg", "gripper_mm", "force_n", "torque_nm")


def build_model(cfg: dict):
    from fuji.s1_policy.model.dp_act import DPACTConfig, DPACTModel
    from fuji.s1_policy.model.dp_diffusion import DPDiffusionModel
    from fuji.s1_policy.model.dp_flow import DPFlowModel

    algo = cfg["algo"]
    if algo == "act":
        # Built through dataclass_config, not SimpleNamespace: it rejects unknown/missing
        # fields instead of handing the model a silently incomplete config, and it forces
        # pretrained=False so loading a local checkpoint cannot reach for ImageNet weights.
        return DPACTModel(dataclass_config(DPACTConfig, cfg["act"]))
    sub = cfg["diffusion"] if algo == "diffusion" else cfg["model"]
    cls = DPDiffusionModel if algo == "diffusion" else DPFlowModel
    return cls(types.SimpleNamespace(**sub))


def evaluate(ckpt: Path, dl, layout, norm_stats, seeds, device, max_batches: int,
             first_n: int = 0) -> dict:
    cfg = json.loads((ckpt / "config.json").read_text())
    model = build_model(cfg)
    # strict=True: a silently-skipped key would score a partly-random model and
    # quietly corrupt the ranking this script exists to produce.
    model.load_state_dict(load_file(str(ckpt / "model.safetensors")), strict=True)
    model.to(device).eval()

    totals = {k: [0.0, 0.0] for k in METRICS}
    n_samples = 0
    with torch.no_grad():
        for seed in seeds:
            # Same isolation the trainer's diagnostics use, so a seed cannot leak
            # into the next checkpoint's sampling.
            with isolated_rng(seed):
                for i, batch in enumerate(dl):
                    if max_batches and i >= max_batches:
                        break
                    target = batch["actions"].clone()
                    mask = batch["action_mask"].clone()
                    normed = dict(batch)
                    for key in ("actions", "state", "force_history"):
                        if key in norm_stats and key in normed:
                            arr = normalize_quantile(normed[key].numpy(), norm_stats[key])
                            normed[key] = torch.from_numpy(arr).to(normed[key].dtype)
                    normed = _move_batch(normed, device)
                    model.reset()
                    pred = model.sample_actions(
                        normed, generator=torch.Generator().manual_seed(seed * 100003 + i))
                    physical = unnormalize_quantile(
                        pred.detach().float().cpu().numpy(), norm_stats["actions"])
                    if not np.isfinite(physical).all():
                        raise FloatingPointError(f"non-finite sample from {ckpt}")
                    if first_n:
                        if first_n > target.shape[1]:
                            raise ValueError(
                                f"--first-n-steps {first_n} exceeds this checkpoint's chunk "
                                f"length {target.shape[1]}")
                        physical = physical[:, :first_n]
                        target = target[:, :first_n]
                        mask = mask[:, :first_n]
                    sums = physical_trajectory_sums(physical, target.numpy(), mask, layout)
                    for name, (num, den) in sums.items():
                        totals[name][0] += num
                        totals[name][1] += den
                    n_samples += int(target.shape[0])
    out = {k: (n / d if d > 0 else math.nan) for k, (n, d) in totals.items()}
    out["_samples"] = n_samples
    out["_valid_counts"] = {k: int(v[1]) for k, v in totals.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*", type=Path)
    ap.add_argument("--run", type=Path, help="evaluate every step_*/ and best_*/ under this run dir")
    ap.add_argument("--split", choices=("val", "train"), default="val",
                    help="val = selection; train = generalization-gap reference only")
    ap.add_argument("--batches", type=int, default=0, help="0 = entire split")
    # Scoring an H=50 checkpoint over its own 50 steps against an H=30 one over its 30 is NOT a
    # comparison: the extra 20 steps sit further into the future and are inherently harder, which
    # on the 0817 pair read as "H=50 is 41% worse" when the first 30 steps were a wash. Because a
    # shorter horizon trains on a strict PREFIX of the same stored chunk (same anchors, same
    # targets, same dt), truncating the longer run to N makes the two directly comparable -- and
    # only the first --exec-steps of the chunk are ever dripped to the robot anyway.
    ap.add_argument("--first-n-steps", type=int, default=0, metavar="N",
                    help="score only the first N action steps (0 = the whole chunk). Use it to "
                         "compare checkpoints trained at DIFFERENT action horizons.")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    targets = list(args.checkpoints)
    if args.run:
        targets += sorted(p for p in args.run.iterdir()
                          if p.is_dir() and (p / "model.safetensors").exists())
    if not targets:
        return int(bool(sys.stderr.write("no checkpoints given\n")))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    ref = json.loads((targets[0] / "config.json").read_text())
    hl = ref["hl_data"]
    layout = get_layout(hl["mode"])
    norm_stats = load_norm_stats(str(targets[0] / "norm_stats.json"))
    # The recorded paths are the TRAINING machine's; relocate them here (see
    # fuji/utils/dataset_paths.py). File ORDER is preserved because it decides episode order.
    zarr_paths = list(resolve_dataset_paths(hl["zarr_paths"], log=print))
    ds = HLDataset(zarr_paths, mode=hl["mode"], drop_glitch_windows=True,
                   glitch_pre_s=FORCE_HIST_SPAN_S, glitch_post_s=ACTION_SPAN_S,
                   require_episode_ends=True)
    train_idx, val_idx = ds.split_indices(hl["val_last_episodes"])
    idx = train_idx if args.split == "train" else val_idx
    # shuffle=False keeps the anchor order identical for every checkpoint.
    dl = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_hl_batch)
    scope = f"FULL {args.split} split" if not args.batches else f"first {args.batches} batches"
    steps = (f"first {args.first_n_steps} steps" if args.first_n_steps
             else "the whole chunk each checkpoint predicts")
    print(f"{args.split} anchors: {len(idx)}  batches: {len(dl)}  scope: {scope}  seeds: {seeds}")
    print(f"scored over: {steps}")
    print(f"{'checkpoint':<24} {'pos_cm':>9} {'rot_deg':>9} {'grip_mm':>9} "
          f"{'force_N':>9} {'torq_Nm':>9} {'n':>7}")

    results = []
    for ckpt in targets:
        try:
            r = evaluate(ckpt, dl, layout, norm_stats, seeds, torch.device("cuda"), args.batches,
                         args.first_n_steps)
        except Exception as exc:
            print(f"{ckpt.name:<24} FAILED: {type(exc).__name__}: {exc}")
            continue
        results.append((ckpt, r))
        print(f"{ckpt.name:<24} {r['position_cm']:>9.4f} {r['rotation_geodesic_deg']:>9.4f} "
              f"{r['gripper_mm']:>9.4f} {r['force_n']:>9.4f} {r['torque_nm']:>9.5f} "
              f"{r['_samples']:>7}")
        if args.json:
            with args.json.open("a") as fh:
                fh.write(json.dumps({"checkpoint": str(ckpt), "split": args.split,
                                     "first_n_steps": args.first_n_steps, **r}) + "\n")

    if results:
        label = "best per metric" if args.split == "val" else "lowest TRAIN error per metric (NOT a selection)"
        print(f"\n{label}:")
        for name in METRICS:
            ck, r = min(results, key=lambda kv: kv[1][name])
            print(f"  {name:<24} {r[name]:.5f}  <- {ck.name}")
    ds.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
