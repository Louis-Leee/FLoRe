#!/usr/bin/env python3
"""Tyro CLI entry point for S1 training (flow matching | diffusion | ACT).

Examples
--------
    # HL-v2 flow matching (default algo):
    uv run python scripts/train_s1.py \
        --hl-data.zarr-paths session.zarr.zip \
        --hl-data.val-last-episodes 4 \
        --hl-data.norm-stats-path checkpoints/norm_stats_hl_left_clean99.json \
        --batch-size 32 --max-steps 100000 --wandb

    # HL-v2 diffusion sibling (DDPM/DDIM, EMA default-on):
    uv run python scripts/train_s1.py --algo diffusion \
        --hl-data.zarr-paths session.zarr.zip \
        --hl-data.val-last-episodes 4 \
        --hl-data.norm-stats-path checkpoints/norm_stats_hl_left_clean99.json \
        --diffusion.diffusion-net unet --batch-size 32 --max-steps 100000

    # HL-v2 ACT sibling (single-pass CVAE; ResNet18 trunk trains at its own LR):
    uv run python scripts/train_s1.py --algo act \
        --hl-data.zarr-paths session.zarr.zip \
        --hl-data.val-last-episodes 4 \
        --hl-data.norm-stats-path checkpoints/norm_stats_hl_left_clean94.json \
        --act.kl-weight 10 --batch-size 32 --max-steps 100000

    # Legacy pre-v2 32-dim flow pipeline:
    uv run python scripts/train_s1.py --pipeline legacy \
        --data.zarr-paths sessionA.zarr.zip --model.action-horizon 50

Enable train-time RTC (flow only) with e.g. ``--rtc-delay-range 0 5``; the diffusion and
ACT samplers have no frozen-prefix path and refuse it.

Each algorithm's knobs live under its own prefix (``--model.*`` for flow,
``--diffusion.*``, ``--act.*``) and ONLY the active one is used, so a stale flag from
another algorithm cannot silently take effect. Its W&B loss namespace follows the same
split: ``train/flow_loss/*``, ``train/diffusion_loss/*``, ``train/act_loss/*``. ACT's is an
L1 (order-1) error while the other two are squared, so ACT's ``objective`` is comparable
across ACT runs but NOT against a flow or diffusion number.
"""

from __future__ import annotations

import logging

import tyro

from flore.s1_policy.training.config import TrainConfig
from flore.s1_policy.training.trainer import Trainer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = tyro.cli(TrainConfig)
    # Which data config carries the paths depends on the pipeline: 'hl' reads hl_data,
    # 'legacy' reads data. Check the ACTIVE one so an HL run isn't blocked by an empty
    # legacy `data.zarr_paths` (and vice versa).
    if config.pipeline == "hl":
        if not config.hl_data.zarr_paths:
            raise SystemExit("No data: pass --hl-data.zarr-paths <one or more .zarr.zip files>")
    else:
        if not config.data.zarr_paths:
            raise SystemExit("No data: pass --data.zarr-paths <one or more .zarr.zip files>")
    Trainer(config).train()


if __name__ == "__main__":
    main()
