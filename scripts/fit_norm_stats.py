#!/usr/bin/env python3
"""Fit quantile (q01/q99) normalization stats for DP-Flow S1 training.

The DP-Flow trainer (``scripts/train_s1.py``) can bake a ``norm_stats.json`` into
its checkpoints and normalize ``state`` + ``actions`` to ~[-1, 1] before the DiT
(``fuji/s1_policy/training/trainer.py::_apply_norm``). Under the default
``ref_frame=per_keypoint`` the ``state`` is the RAW absolute EE pose (metres, can be
several metres from the world origin), which is poorly scaled for AdaLN conditioning
— so fitting stats first is strongly recommended for any real (non-smoke) run.

This iterates the SAME :class:`UMIDataset` the trainer builds (same ``ref_frame`` /
``action_horizon`` / ``obs_history_len``), so the stats it writes match exactly what
the trainer will feed the model. It stacks each sample's ``state`` and per-row
``action_chunk`` and calls :func:`compute_quantile_stats`.

Usage
-----
    uv run python scripts/fit_norm_stats.py \
        --zarr-paths third_party/UMI_collection/data/datasets/muse_force_20260725_flat.zarr.zip \
        --action-horizon 16 --ref-frame per_keypoint \
        --out checkpoints/norm_stats_muse_force.json

Pass the SAME ``--action-horizon`` / ``--ref-frame`` / ``--obs-history-len`` you will
train with. ``--max-samples`` subsamples the frame index for speed (default: all).
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from fuji.s1_policy.data.norm_stats import compute_quantile_stats, save_norm_stats
from fuji.s1_policy.data.umi_dataset import UMIDataset

logger = logging.getLogger("fit_norm_stats")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr-paths", nargs="+", required=True, help="One or more .zarr.zip paths.")
    ap.add_argument("--out", required=True, help="Output norm_stats.json path.")
    ap.add_argument("--action-horizon", type=int, default=16, help="MUST match training.")
    ap.add_argument("--ref-frame", default="per_keypoint",
                    choices=["per_keypoint", "left", "right", "head", "world"],
                    help="MUST match training.")
    ap.add_argument("--obs-history-len", type=int, default=1, help="MUST match training.")
    ap.add_argument("--obs-history-stride", type=int, default=1, help="MUST match training.")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Subsample this many frames for the fit (0 = use all).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = UMIDataset(
        list(args.zarr_paths),
        action_horizon=args.action_horizon,
        ref_frame=args.ref_frame,
        obs_history_len=args.obs_history_len,
        obs_history_stride=args.obs_history_stride,
    )
    n_total = len(ds)
    if args.max_samples and args.max_samples < n_total:
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(n_total, size=args.max_samples, replace=False)
    else:
        idxs = np.arange(n_total)
    logger.info("Fitting norm stats over %d / %d samples (ref_frame=%s, H=%d, To=%d)",
                len(idxs), n_total, args.ref_frame, args.action_horizon, args.obs_history_len)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for i, idx in enumerate(idxs):
        s = ds[int(idx)]
        st = s["state"].numpy()          # (32,) for To==1, else (To, 32)
        states.append(st.reshape(-1, st.shape[-1]))
        actions.append(s["action_chunk"].numpy())  # (H, 32)
        if (i + 1) % 500 == 0:
            logger.info("  %d/%d", i + 1, len(idxs))
    ds.close()

    samples = {
        "state": np.concatenate(states, axis=0),      # (N*To, 32)
        "actions": np.concatenate(actions, axis=0),    # (N*H, 32)
    }
    stats = compute_quantile_stats(samples)
    save_norm_stats(stats, args.out)
    logger.info("Saved norm stats -> %s", args.out)
    for key, st in stats.items():
        rng = np.maximum(st.q99 - st.q01, 0.0)
        logger.info("  %-8s q01..q99 span: min=%.4f max=%.4f mean=%.4f",
                    key, float(rng.min()), float(rng.max()), float(rng.mean()))


if __name__ == "__main__":
    main()
