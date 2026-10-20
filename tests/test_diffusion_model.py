"""DP-Diffusion model: forward-loss finiteness, DDIM sample shape, masking invariants,
the HL force-in/out path, and the plain-sampler RTC guard. Uses a tiny stub conv encoder
so no timm weights are downloaded and the tests run on CPU in well under a second.

Mirrors tests/test_model_forward.py (the flow sibling) so the two models are held to the
same behavioral contract — the point of building them as equal-level siblings."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from flore.s1_policy.model.dp_diffusion import DPDiffusionConfig, DPDiffusionModel

# HL-v2 left_only shape.
ACTION_DIM = 16
ACTION_HORIZON = 16
STATE_DIM = 7


class StubEncoder(nn.Module):
    """Minimal pluggable encoder: (N, 3, H, W) -> (N, num_features). No downloads."""

    def __init__(self, num_features: int = 64):
        super().__init__()
        self.num_features = num_features
        self.conv = nn.Conv2d(3, num_features, kernel_size=16, stride=16)

    def forward(self, x):
        return self.conv(x).mean(dim=(-1, -2))  # global-average-pool to (N, num_features)


def _tiny_model(**cfg_overrides) -> DPDiffusionModel:
    defaults = dict(
        diffusion_net="unet",
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        state_dim=STATE_DIM,
        obs_history_len=2,
        force_hist_len=8,
        force_hist_channels=6,
        layout_mode="left_only",
        num_draws=1,
        num_inference_steps=4,
        pretrained=False,
    )
    defaults.update(cfg_overrides)
    return DPDiffusionModel(DPDiffusionConfig(**defaults), encoder=StubEncoder())


def _batch(batch_size: int = 2, To: int = 2, with_masks: bool = True,
           force_valid: bool = True) -> dict:
    B = batch_size
    batch = {
        "images": [torch.randn(B, 3, 224, 224) for _ in range(To)],
        "img_masks": [torch.ones(B, dtype=torch.bool) for _ in range(To)],
        "state": torch.randn(B, To, STATE_DIM),
        "state_valid": torch.ones(B, To, STATE_DIM, dtype=torch.bool),
        "force_history": torch.randn(B, 8, 6),
        "force_history_valid": torch.full((B,), force_valid, dtype=torch.bool),
        "actions": torch.randn(B, ACTION_HORIZON, ACTION_DIM),
    }
    if with_masks:
        batch["action_mask"] = torch.ones(B, ACTION_HORIZON, ACTION_DIM, dtype=torch.bool)
    return batch


# ---------------------------------------------------------------------------
# Forward loss
# ---------------------------------------------------------------------------
def test_forward_loss_finite():
    torch.manual_seed(0)
    model = _tiny_model()
    out = model(_batch())
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() > 0
    assert "diffusion_loss" in out["metrics"]


def test_forward_loss_finite_multidraw():
    torch.manual_seed(0)
    model = _tiny_model(num_draws=4)
    out = model(_batch(batch_size=3))
    assert torch.isfinite(out["loss"])


def test_forward_loss_finite_no_masks():
    torch.manual_seed(0)
    model = _tiny_model()
    out = model(_batch(with_masks=False))
    assert torch.isfinite(out["loss"])


def test_forward_loss_finite_input_pertub():
    """input_pertub>0 (DDPM-IP) keeps the loss finite and positive."""
    torch.manual_seed(0)
    model = _tiny_model(input_pertub=0.1)
    out = model(_batch())
    assert torch.isfinite(out["loss"]) and out["loss"].item() > 0


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def test_sample_actions_shape():
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    out = model.sample_actions(_batch(), num_steps=4)
    assert out.shape == (2, ACTION_HORIZON, ACTION_DIM)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_sample_actions_default_num_steps():
    """num_steps=None uses config.num_inference_steps."""
    torch.manual_seed(0)
    model = _tiny_model(num_inference_steps=5)
    model.eval()
    out = model.sample_actions(_batch())
    assert out.shape == (2, ACTION_HORIZON, ACTION_DIM)


def test_dit_decoder_forward_and_sample():
    """The DiT decoder path works for forward + DDIM sampling."""
    torch.manual_seed(0)
    model = _tiny_model(diffusion_net="dit", model_channels=128, num_blocks=2, n_heads=4,
                        diffusion_step_embed_dim=64)
    out = model(_batch())
    assert torch.isfinite(out["loss"])
    model.eval()
    s = model.sample_actions(_batch(), num_steps=3)
    assert s.shape == (2, ACTION_HORIZON, ACTION_DIM)
    assert torch.isfinite(s).all()


def test_sample_is_reproducible_with_generator():
    """Same seed generator -> identical samples (the DDIM loop is deterministic given
    the initial noise; UMI's DDIM has no per-step stochasticity)."""
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    b = _batch()
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    s1 = model.sample_actions(b, num_steps=4, generator=g1)
    s2 = model.sample_actions(b, num_steps=4, generator=g2)
    np.testing.assert_allclose(s1.numpy(), s2.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Masking invariants (mirrors the flow sibling's tests)
# ---------------------------------------------------------------------------
def test_action_mask_excludes_masked_entries():
    """_compute_diffusion_loss must ignore masked-out (step, dim) entries: changing
    pred/target on the masked entries must not change the loss."""
    torch.manual_seed(0)
    model = _tiny_model()
    B, T = 2, ACTION_HORIZON
    pred = torch.randn(B, T, ACTION_DIM)
    target = torch.randn(B, T, ACTION_DIM)
    mask = torch.ones(B, T, ACTION_DIM, dtype=torch.bool)
    mask[:, :, 10:] = False  # force cols unmeasured for this rig

    loss_a = model._compute_diffusion_loss(pred, target, None, None, mask).item()
    pred2 = pred.clone()
    pred2[:, :, 10:] += 100.0  # perturb only masked dims
    loss_b = model._compute_diffusion_loss(pred2, target, None, None, mask).item()
    assert loss_a == pytest.approx(loss_b, abs=1e-6)


def test_action_dim_mask_excludes_pad_dims():
    """action_dim_mask (per-dim) drops padded dims from the loss."""
    torch.manual_seed(0)
    model = _tiny_model()
    B, T = 2, ACTION_HORIZON
    pred = torch.randn(B, T, ACTION_DIM)
    target = torch.randn(B, T, ACTION_DIM)
    dim_mask = torch.ones(B, ACTION_DIM, dtype=torch.bool)
    dim_mask[:, 10:] = False

    loss_a = model._compute_diffusion_loss(pred, target, None, dim_mask, None).item()
    pred2 = pred.clone()
    pred2[:, :, 10:] += 100.0
    loss_b = model._compute_diffusion_loss(pred2, target, None, dim_mask, None).item()
    assert loss_a == pytest.approx(loss_b, abs=1e-6)


def test_action_is_pad_excludes_timesteps():
    """action_is_pad (per-step) drops whole timesteps from the loss."""
    torch.manual_seed(0)
    model = _tiny_model()
    B, T = 2, ACTION_HORIZON
    pred = torch.randn(B, T, ACTION_DIM)
    target = torch.randn(B, T, ACTION_DIM)
    is_pad = torch.zeros(B, T, dtype=torch.bool)
    is_pad[:, T // 2:] = True

    loss_a = model._compute_diffusion_loss(pred, target, is_pad, None, None).item()
    pred2 = pred.clone()
    pred2[:, T // 2:] += 100.0
    loss_b = model._compute_diffusion_loss(pred2, target, is_pad, None, None).item()
    assert loss_a == pytest.approx(loss_b, abs=1e-6)


def test_masked_actions_zeroed_before_noising():
    """The forward pass must zero unsupervised action entries before add_noise: changing
    ONLY the masked entries of `actions` must leave the loss unchanged (the decoder input
    at those positions is scaled pure noise either way, and the loss drops them)."""
    torch.manual_seed(0)
    model = _tiny_model()
    b = _batch()
    b["action_mask"][:, :, 10:] = False  # force cols unmeasured
    # Re-seed identically and perturb only masked action entries.
    torch.manual_seed(42)
    out_a = model(b)["loss"].item()
    b2 = {k: (v.clone() if torch.is_tensor(v) else [t.clone() for t in v]) for k, v in b.items()}
    b2["actions"][:, :, 10:] += 100.0
    torch.manual_seed(42)
    out_b = model(b2)["loss"].item()
    assert out_a == pytest.approx(out_b, abs=1e-5)


# ---------------------------------------------------------------------------
# Force-history validity gating (the "no data" != "no contact" contract)
# ---------------------------------------------------------------------------
def test_invalid_force_history_uses_learned_embedding():
    """A force_history_valid=False sample routes through the encoder's learned invalid
    embedding, so the actual window values do not affect global_cond. Verified: changing
    only the (masked-invalid) window leaves the loss unchanged given a fixed seed."""
    torch.manual_seed(0)
    model = _tiny_model()
    b = _batch(force_valid=False)
    torch.manual_seed(7)
    out_a = model(b)["loss"].item()
    b2 = {k: (v.clone() if torch.is_tensor(v) else [t.clone() for t in v]) for k, v in b.items()}
    b2["force_history"] += 50.0
    torch.manual_seed(7)
    out_b = model(b2)["loss"].item()
    assert out_a == pytest.approx(out_b, abs=1e-5)


def test_missing_force_history_raises():
    """A force model with no force_history in the batch fails loudly (a zero-filled
    window would read as 'no contact')."""
    torch.manual_seed(0)
    model = _tiny_model()
    b = _batch()
    del b["force_history"]
    with pytest.raises(KeyError, match="force_history"):
        model(b)


# ---------------------------------------------------------------------------
# Observation history sizing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("To", [1, 2, 4])
def test_history_forward_and_sample(To):
    torch.manual_seed(0)
    model = _tiny_model(obs_history_len=To)
    out = model(_batch(To=To))
    assert torch.isfinite(out["loss"])
    model.eval()
    s = model.sample_actions(_batch(To=To), num_steps=3)
    assert s.shape == (2, ACTION_HORIZON, ACTION_DIM)
    assert torch.isfinite(s).all()


def test_history_grows_global_cond():
    """global_cond width scales with To (To pseudo-cameras + To state frames), leaving the
    force slot fixed (the model is language-free)."""
    m1 = _tiny_model(obs_history_len=1)
    m3 = _tiny_model(obs_history_len=3)
    # ConditionalUnet1D folds global_cond into cond_dim; read the first residual block's
    # cond_encoder input Linear (dsed + global_cond_dim). Compare the DELTA instead.
    enc = m1.encoder.num_features
    fsd = m1.config.filled_state_dim
    # global_cond = To*enc + To*fsd + force. Delta from To=1 to To=3 is 2*(enc+fsd).
    g1 = m1.decoder.down_modules[0][0].cond_encoder[1].in_features
    g3 = m3.decoder.down_modules[0][0].cond_encoder[1].in_features
    assert g3 - g1 == 2 * (enc + fsd)


# ---------------------------------------------------------------------------
# Per-group training metrics (diffusion_loss/<group>) — logging only
# ---------------------------------------------------------------------------
def test_forward_emits_per_group_metrics():
    """forward() surfaces diffusion_loss/<group> alongside diffusion_loss, all finite.
    left_only has left/pos/rot/grip/force groups (no right/head)."""
    torch.manual_seed(0)
    model = _tiny_model()
    out = model(_batch())
    m = out["metrics"]
    for g in ("left", "pos", "rot", "grip", "force"):
        assert f"diffusion_loss/{g}" in m, f"missing diffusion_loss/{g}"
        assert np.isfinite(m[f"diffusion_loss/{g}"])
        assert m[f"diffusion_loss/{g}"] >= 0.0
    # right / head are absent in left_only.
    assert "diffusion_loss/right" not in m
    assert "diffusion_loss/head" not in m


def test_per_group_metrics_skipped_without_layout():
    """With no layout_mode, per-group metrics are skipped (no legacy fallback)."""
    torch.manual_seed(0)
    model = _tiny_model(layout_mode="", force_hist_channels=0, force_hist_len=0)
    b = _batch()
    del b["force_history"]  # no force encoder now
    del b["force_history_valid"]
    out = model(b)
    assert set(out["metrics"].keys()) == {"diffusion_loss"}


def test_per_group_metric_matches_manual_masked_mse():
    """A per-group metric equals the masked-MSE computed by hand over that group's dims,
    with the SAME time-include + action_dim_mask + action_mask composition."""
    torch.manual_seed(0)
    model = _tiny_model()
    B, T = 2, ACTION_HORIZON
    pred = torch.randn(B, T, ACTION_DIM)
    target = torch.randn(B, T, ACTION_DIM)
    is_pad = torch.zeros(B, T, dtype=torch.bool)
    is_pad[:, T // 2:] = True
    mask = torch.ones(B, T, ACTION_DIM, dtype=torch.bool)

    metrics = model._extra_train_metrics(pred, target, is_pad, None, mask)

    from flore.s1_policy.data.layout import get_layout
    groups = get_layout("left_only").metric_group_dims()

    sq = (pred - target) ** 2
    include = (~is_pad).unsqueeze(-1).float()
    for g, dims in groups.items():
        dims = [d for d in dims if d < ACTION_DIM]
        w = (include * mask.float())[:, :, dims]
        num = (sq[:, :, dims] * w).sum()
        den = w.sum().clamp(min=1.0)
        assert metrics[f"diffusion_loss/{g}"] == pytest.approx((num / den).item(), abs=1e-5)


# ---------------------------------------------------------------------------
# Plain-sampler RTC guard
# ---------------------------------------------------------------------------
def test_sampler_rejects_rtc_delay():
    """The diffusion sampler is plain: a nonzero delay_estimate must fail loudly rather
    than silently emit un-chunked actions."""
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    with pytest.raises(NotImplementedError, match="Real-Time Chunking"):
        model.sample_actions(_batch(), num_steps=4, delay_estimate=3)


def test_sampler_allows_zero_delay():
    """delay_estimate=0 (the serving default when RTC is off) is fine."""
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    out = model.sample_actions(_batch(), num_steps=4, steps_executed=0, delay_estimate=0)
    assert out.shape == (2, ACTION_HORIZON, ACTION_DIM)


def test_reset_clears_cache():
    """reset() flips the RTC-cache validity flag (parity with the flow model)."""
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    model.sample_actions(_batch(), num_steps=3)
    assert model._last_actions_valid is True
    model.reset()
    assert model._last_actions_valid is False


def test_sample_actions_cpu_generator_ok():
    """A CPU generator against a CPU model samples fine; the device-relocation guard is a
    no-op when devices already match (the CUDA-mismatch path can't run on a CPU CI)."""
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    g = torch.Generator().manual_seed(7)
    out = model.sample_actions(_batch(), num_steps=3, generator=g)
    assert out.shape == (2, ACTION_HORIZON, ACTION_DIM)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# prediction_type branches (epsilon / sample / v_prediction)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pred_type", ["epsilon", "sample", "v_prediction"])
def test_prediction_type_forward_and_sample(pred_type):
    """Each prediction_type builds a matching scheduler, trains (finite loss) and samples
    the right shape — the 'sample'/'v_prediction' target branches were otherwise untested."""
    torch.manual_seed(0)
    model = _tiny_model(prediction_type=pred_type)
    out = model(_batch())
    assert torch.isfinite(out["loss"]) and out["loss"].item() > 0
    model.eval()
    s = model.sample_actions(_batch(), num_steps=3)
    assert s.shape == (2, ACTION_HORIZON, ACTION_DIM)
    assert torch.isfinite(s).all()


def test_unsupported_prediction_type_raises():
    """An unsupported prediction_type fails loudly in forward (not silently)."""
    torch.manual_seed(0)
    # The scheduler itself accepts arbitrary strings; the model's forward is what rejects.
    model = _tiny_model()
    model.noise_scheduler.config.prediction_type = "bogus"
    with pytest.raises(ValueError, match="prediction_type"):
        model(_batch())


# ---------------------------------------------------------------------------
# TrainConfig guards for the diffusion algo
# ---------------------------------------------------------------------------
def test_config_diffusion_clip_sample_requires_norm_stats():
    """algo='diffusion' with clip_sample=True but no norm_stats must raise (the sampler
    would crush out-of-[-1,1] pose/force at deploy)."""
    from flore.s1_policy.model.dp_diffusion import DPDiffusionConfig
    from flore.s1_policy.training.config import HLDataConfig, TrainConfig

    with pytest.raises(ValueError, match="clip_sample"):
        TrainConfig(algo="diffusion", pipeline="hl",
                    diffusion=DPDiffusionConfig(clip_sample=True),
                    hl_data=HLDataConfig(norm_stats_path=""))


def test_config_diffusion_clip_sample_ok_with_norm_stats():
    """The same combination is fine once a norm_stats path is provided."""
    from flore.s1_policy.model.dp_diffusion import DPDiffusionConfig
    from flore.s1_policy.training.config import HLDataConfig, TrainConfig

    cfg = TrainConfig(algo="diffusion", pipeline="hl",
                      diffusion=DPDiffusionConfig(clip_sample=True),
                      hl_data=HLDataConfig(norm_stats_path="stats.json"))
    assert cfg.diffusion.clip_sample is True


def test_config_diffusion_no_clip_needs_no_norm_stats():
    """clip_sample=False (train unnormalized) needs no norm_stats and pushes layout dims
    onto the diffusion config (not the flow one)."""
    from flore.s1_policy.model.dp_diffusion import DPDiffusionConfig
    from flore.s1_policy.training.config import TrainConfig

    cfg = TrainConfig(algo="diffusion", pipeline="hl",
                      diffusion=DPDiffusionConfig(clip_sample=False))
    assert cfg.active_model is cfg.diffusion
    assert cfg.diffusion.action_dim == 16 and cfg.diffusion.layout_mode == "left_only"
    assert cfg.ema is True  # default ON for diffusion
    assert cfg.model.action_horizon == 50  # flow config untouched


# ---------------------------------------------------------------------------
# The force-OUTPUT-ONLY ablation (FLORE_DROP_FORCE_INPUT)
# ---------------------------------------------------------------------------
def test_drop_force_input_builds_no_force_encoder_but_still_predicts_force(monkeypatch):
    """No wrench in, force still out — end to end, on a real model and a real batch.

    ``tests/test_layout.py`` pins the layout arithmetic; this pins that the arithmetic reaches
    the built model. Three things have to hold together or the ablation silently is not one:

    1. ``force_encoder`` is not built at all, so there is no path from the wrench to
       ``global_cond`` — not a zeroed input, which a model can still learn the mask of.
    2. The batch STILL CARRIES a real 6-channel ``force_history``, and the model must accept
       and ignore it. This is what lets the run reuse the force-ful run's norm-stats file:
       ``Trainer._apply_norm`` normalizes ``force_history`` against a 6-wide entry, and a
       0-channel array would fail its width check.
    3. The force action columns are still SUPERVISED, so ``metric_sums`` reports a ``force``
       group and the loss depends on the force predictions.
    """
    from flore.s1_policy.data import layout as layout_mod
    from flore.s1_policy.training.config import HLDataConfig, TrainConfig

    monkeypatch.setattr(layout_mod, "DROP_FORCE_INPUT", True)
    cfg = TrainConfig(algo="diffusion", pipeline="hl",
                      hl_data=HLDataConfig(zarr_paths=("x.zarr.zip",), mode="left_only"),
                      diffusion=DPDiffusionConfig(clip_sample=False, pretrained=False))
    assert cfg.diffusion.force_hist_channels == 0
    assert cfg.diffusion.force_hist_len == 0
    assert cfg.diffusion.action_dim == 16, "the force COLUMNS must stay in the action vector"

    torch.manual_seed(0)
    model = DPDiffusionModel(
        DPDiffusionConfig(
            diffusion_net="unet", action_dim=cfg.diffusion.action_dim,
            action_horizon=ACTION_HORIZON, state_dim=STATE_DIM, obs_history_len=2,
            force_hist_len=cfg.diffusion.force_hist_len,
            force_hist_channels=cfg.diffusion.force_hist_channels,
            layout_mode="left_only", num_inference_steps=4, pretrained=False),
        encoder=StubEncoder())
    assert model.force_encoder is None, "a force encoder was built despite the ablation"
    assert not any("force_encoder" in n for n, _ in model.named_parameters())

    # The batch still has the real wrench; the model must accept and ignore it.
    batch = _batch()
    assert batch["force_history"].shape[-1] == 6
    out = model(batch)
    assert torch.isfinite(out["loss"])
    assert "force" in out["metric_sums"], "the force group stopped being reported"
    num, den = out["metric_sums"]["force"]
    assert float(den) > 0, "the force columns are no longer supervised — that is DROP_FORCE"

    # And the loss really does depend on the force predictions: perturbing only the force
    # columns of the target must move it.
    torch.manual_seed(0)
    base = float(model(batch)["loss"].detach())
    moved = {**batch, "actions": batch["actions"].clone()}
    moved["actions"][..., 10:16] += 5.0
    torch.manual_seed(0)
    assert float(model(moved)["loss"].detach()) != pytest.approx(base), (
        "the loss ignores the force columns, so nothing supervises the force output")


# ---------------------------------------------------------------------------
# EMA save/restore in the trainer's save_checkpoint
# ---------------------------------------------------------------------------
def test_save_checkpoint_saves_ema_and_restores_live_weights(tmp_path):
    """save_checkpoint(..., ema) must (a) write the EMA-averaged weights and (b) restore
    the LIVE training weights afterward (try/finally), so the optimizer never continues
    from the smoothed weights. This is the deploy-critical EMA mechanism."""
    import safetensors.torch

    from flore.s1_policy.model.dp_diffusion import DPDiffusionConfig
    from flore.s1_policy.training.config import TrainConfig
    from flore.s1_policy.training.trainer import save_checkpoint

    torch.manual_seed(0)
    model = _tiny_model()
    from diffusers.training_utils import EMAModel

    ema = EMAModel(model.parameters(), decay=0.5, use_ema_warmup=True)
    # Drive the live weights away from the EMA shadow with a few fake optimizer steps.
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    b = _batch()
    for _ in range(3):
        opt.zero_grad()
        model(b)["loss"].backward()
        opt.step()
        ema.step(model.parameters())

    # Snapshot the live weights (what must be restored) and the EMA shadow (what must save).
    live = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    ema_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.restore(model.parameters())

    # clip_sample=False so the config's clip_sample<->norm_stats guard doesn't require a
    # stats file just to construct the config for this checkpoint-mechanics test.
    cfg = TrainConfig(algo="diffusion", pipeline="hl",
                      diffusion=DPDiffusionConfig(clip_sample=False))
    provenance = {"sentinel": {"value": 1}}
    provenance_before = json.loads(json.dumps(provenance))
    ckpt = save_checkpoint(model, cfg, 10, tmp_path, "", ema, provenance=provenance)
    assert provenance == provenance_before

    # (a) the SAVED tensors equal the EMA shadow (not the live weights).
    saved = safetensors.torch.load_file(str(ckpt / "model.safetensors"))
    checkpoint_config = json.loads((ckpt / "config.json").read_text())
    assert checkpoint_config["_provenance"]["deploy_weights"] == "ema"
    assert checkpoint_config["_provenance"]["model_sha256"] == hashlib.sha256(
        (ckpt / "model.safetensors").read_bytes()
    ).hexdigest()
    a_key = next(k for k in saved if "weight" in k and saved[k].numel() > 1)
    assert torch.allclose(saved[a_key], ema_weights[a_key], atol=1e-6)
    assert not torch.allclose(saved[a_key], live[a_key], atol=1e-6), \
        "saved weights equal live weights — EMA copy_to did not take effect"

    # (b) the LIVE model weights are restored (unchanged by the save).
    for k, v in model.state_dict().items():
        assert torch.allclose(v, live[k], atol=1e-6), f"live weight {k} not restored after EMA save"


def test_save_checkpoint_rejects_ema_config_mismatch(tmp_path):
    from flore.s1_policy.model.dp_diffusion import DPDiffusionConfig
    from flore.s1_policy.training.config import TrainConfig
    from flore.s1_policy.training.trainer import save_checkpoint

    model = _tiny_model()
    configured = TrainConfig(
        algo="diffusion",
        pipeline="hl",
        diffusion=DPDiffusionConfig(clip_sample=False),
        ema=True,
    )
    with pytest.raises(ValueError, match="disagrees with the checkpoint EMA object"):
        save_checkpoint(model, configured, 1, tmp_path / "missing_ema")

    disabled = TrainConfig(
        algo="diffusion",
        pipeline="hl",
        diffusion=DPDiffusionConfig(clip_sample=False),
        ema=False,
    )
    with pytest.raises(ValueError, match="disagrees with the checkpoint EMA object"):
        save_checkpoint(model, disabled, 1, tmp_path / "unexpected_ema", ema=object())
