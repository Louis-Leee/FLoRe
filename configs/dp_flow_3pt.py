"""Reference DP-Flow 3-point training config.

This is a plain-Python config (the FUJI config system is tyro dataclasses, not YAML):
``build()`` returns a fully-populated :class:`TrainConfig`. Import and pass it to
``Trainer``, or use it as a template for CLI overrides with ``scripts/train_s1.py``.

3-point action/state layout (32 dims, 29 real + 3 pad), single 224x224 camera. DiT
decoder (RTC-capable) with train-time RTC enabled (delay 0..5).
"""

from __future__ import annotations

from fuji.s1_policy.model.dp_flow import DPFlowConfig
from fuji.s1_policy.training.config import DataConfig, TrainConfig


def build(zarr_paths: tuple[str, ...] = (), norm_stats_path: str = "") -> TrainConfig:
    model = DPFlowConfig(
        vit="vit_b",
        pretrained=True,
        diffusion_net="dit",
        action_dim=32,
        action_horizon=50,
        state_dim=32,
        num_cameras=1,
        model_channels=512,
        num_blocks=12,
        n_heads=16,
        pos_emb_type="rope",
        max_seq_len=64,
        inference_steps=10,
        flow_beta_a=1.5,
        flow_beta_b=1.0,
        num_draws=8,
        filled_state_dim=64,
        # Train-time RTC frozen-prefix delays: 0..5 steps.
        rtc_delay_range=(0, 5),
    )
    data = DataConfig(
        zarr_paths=zarr_paths,
        action_horizon=50,
        image_key="camera0_rgb",
        image_mean=0.5,
        image_std=0.5,
        norm_stats_path=norm_stats_path,
    )
    return TrainConfig(
        model=model,
        data=data,
        batch_size=32,
        lr=1e-4,
        weight_decay=1e-4,
        grad_clip=1.0,
        max_steps=100_000,
        warmup_steps=1_000,
        bf16=True,
        output_dir="checkpoints/dp_flow_3pt",
        save_every=5_000,
        wandb=False,
    )


if __name__ == "__main__":
    cfg = build()
    print(cfg)
