"""DP-ACT model: CVAE loss finiteness, single-pass sampling, masking invariants, the HL
force-in/out path, upstream-parity checks, and the refusals that keep an ACT checkpoint
from being mistaken for an iterative sampler.

Mirrors tests/test_model_forward.py (flow) and tests/test_diffusion_model.py (diffusion)
so all three models are held to the same behavioral contract — the point of building them
as equal-level siblings. Uses a tiny stub backbone so no torchvision weights are
downloaded and the tests run on CPU in well under a second.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from fuji.s1_policy.data import layout
from fuji.s1_policy.model.dp_act import DPACTConfig, DPACTModel, _sinusoid_encoding_table

# HL-v2 left_only shape on the v3 grid. These size the STUB tensors below and are
# deliberately literal — the model must handle the shape it is handed, independent of what
# this process's layout is pinned to. Assertions ABOUT the layout use `layout.*`.
ACTION_DIM = 16
ACTION_HORIZON = 50
STATE_DIM = 7
HIDDEN = 128


class StubBackbone(nn.Module):
    """Minimal pluggable trunk: (N, 3, H, W) -> (features, pos). No downloads.

    Returns a 14x14 grid at 224x224 (stride 16) rather than ResNet's 7x7 — the model must
    not care about the grid size, only that features and pos agree.
    """

    def __init__(self, num_channels: int = 64, hidden_dim: int = HIDDEN):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(3, num_channels, kernel_size=16, stride=16)

    def forward(self, x):
        f = self.conv(x)
        pos = torch.zeros(1, self.hidden_dim, f.shape[2], f.shape[3], dtype=f.dtype, device=f.device)
        return f, pos


def _tiny_model(**overrides) -> DPACTModel:
    defaults = dict(
        hidden_dim=HIDDEN,
        dim_feedforward=256,
        nheads=4,
        enc_layers=1,
        dec_layers=2,
        vae_enc_layers=1,
        latent_dim=8,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        state_dim=STATE_DIM,
        obs_history_len=1,
        force_hist_len=1,
        force_hist_channels=6,
        force_enc_width=16,
        force_enc_out_dim=8,
        layout_mode="left_only",
        pretrained=False,
    )
    defaults.update(overrides)
    model = DPACTModel(DPACTConfig(**defaults), encoder=StubBackbone(hidden_dim=defaults["hidden_dim"]))
    # Dropout makes sampling stochastic in train() mode; every real caller evals first.
    return model.eval()


def _batch(batch_size: int = 2, To: int = 1, with_masks: bool = True, force_valid: bool = True,
           force_channels: int = 6) -> dict:
    B = batch_size
    batch = {
        "images": [torch.randn(B, 3, 224, 224) for _ in range(To)],
        "img_masks": [torch.ones(B, dtype=torch.bool) for _ in range(To)],
        "state": torch.randn(B, To, STATE_DIM),
        "state_valid": torch.ones(B, To, STATE_DIM),
        "actions": torch.randn(B, ACTION_HORIZON, ACTION_DIM),
    }
    if force_channels:
        batch["force_history"] = torch.randn(B, 1, force_channels)
        batch["force_history_valid"] = torch.full((B,), force_valid, dtype=torch.bool)
    if with_masks:
        batch["action_mask"] = torch.ones(B, ACTION_HORIZON, ACTION_DIM)
    return batch


# ---------------------------------------------------------------------------
# Upstream parity
# ---------------------------------------------------------------------------
def test_the_sinusoid_table_matches_upstream_bit_for_bit():
    """The CVAE encoder's position table is a PORT: upstream builds it with a per-position
    Python list comprehension, this builds it vectorized. A silent numerical difference
    here would shift the style encoder's notion of "which action step is this" — and the
    whole point of porting rather than reinventing is weight-for-weight equivalence."""
    def upstream(n_position: int, d_hid: int):
        def vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        table = np.array([vec(p) for p in range(n_position)])
        table[:, 0::2] = np.sin(table[:, 0::2])
        table[:, 1::2] = np.cos(table[:, 1::2])
        return torch.FloatTensor(table).unsqueeze(0)

    for n, d in ((52, 512), (3, 8), (100, 128)):
        assert torch.equal(upstream(n, d), _sinusoid_encoding_table(n, d)), f"drift at n={n} d={d}"


def test_the_backbone_renormalizes_to_imagenet_statistics():
    """LOAD-BEARING for the pretrained trunk. The ResNet is ImageNet-pretrained with FROZEN
    BatchNorm, so its baked-in statistics only transfer if the input distribution matches —
    and FUJI's shared dataset emits (x/255-0.5)/0.5, whose std is 2.23x too small. Measured
    without the fix: layer4 features at cosine 0.671 / relative L2 0.765 versus the correct
    path, with 69.4% of activations ReLU-dead instead of 59.6%.

    Asserts the folded affine reproduces upstream's `Normalize(ImageNet)` on a [0,1] image
    EXACTLY, which is the only way the frozen-BN argument holds."""
    from fuji.s1_policy.model.act.detr_backbone import IMAGENET_MEAN, IMAGENET_STD, ResNetBackbone

    torch.manual_seed(0)
    raw = torch.randint(0, 256, (2, 3, 64, 64)).float()
    fuji_in = (raw / 255.0 - 0.5) / 0.5          # exactly what HLDataset._load_image yields
    import torchvision.transforms as T
    upstream_in = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(raw / 255.0)   # policy.py:20-22

    bb = ResNetBackbone("resnet18", 64, pretrained=False, renormalize=True)
    recovered = fuji_in * bb.renorm_scale + bb.renorm_shift
    torch.testing.assert_close(recovered, upstream_in, rtol=0, atol=1e-6)

    # And it is OPT-OUTABLE, staying a pass-through when disabled.
    off = ResNetBackbone("resnet18", 64, pretrained=False, renormalize=False)
    assert not hasattr(off, "renorm_scale")


def test_the_dataset_image_affine_is_pushed_into_the_act_config():
    """The re-normalization must INVERT the dataset's own affine, so a drift between
    hl_data.image_mean/std and act.dataset_image_mean/std would feed the frozen-BN trunk
    out-of-distribution input — the exact failure it exists to prevent. Pushed, not set."""
    from fuji.s1_policy.training.config import HLDataConfig, TrainConfig

    cfg = TrainConfig(algo="act", pipeline="hl",
                      hl_data=HLDataConfig(image_mean=0.25, image_std=0.75,
                                           norm_stats_path="s.json"))
    assert cfg.act.dataset_image_mean == 0.25
    assert cfg.act.dataset_image_std == 0.75


def test_state_projection_cannot_be_disabled_for_act():
    """ACT has no zero-padded state slot (state is a transformer token), so clearing this
    would send validate_model_layout down the zero-pad branch and AttributeError on a
    `filled_state_dim` that does not exist. Refuse with the reason instead."""
    from fuji.s1_policy.training.config import HLDataConfig, TrainConfig

    with pytest.raises(ValueError, match="state_projection cannot be False"):
        TrainConfig(algo="act", act=DPACTConfig(state_projection=False),
                    hl_data=HLDataConfig(norm_stats_path="s.json"))


def test_the_action_head_reads_the_FIRST_decoder_layer():
    """Upstream takes `hs[0]` (detr_vae.py:131) with return_intermediate_dec=True — the
    SHALLOWEST decoder layer, not the deepest. That is surprising enough that someone will
    "fix" it to hs[-1], which is a DIFFERENT MODEL and would silently pass every other test.

    Pinned by construction: with 3 decoder layers, only layer 1 may receive gradient from
    the action head. (Consequence, documented in D6 of the design doc: layers 2..N are dead
    weight — 43.4% of params at dec_layers=7.)"""
    model = _tiny_model(dec_layers=3)
    out = model(_batch(batch_size=1))
    out["loss"].backward()
    layers = model.transformer.decoder.layers
    g0 = sum(float(p.grad.abs().sum()) for p in layers[0].parameters() if p.grad is not None)
    deep = sum(float(p.grad.abs().sum()) for layer in layers[1:]
               for p in layer.parameters() if p.grad is not None)
    assert g0 > 0.0, "the first decoder layer must be on the action head's gradient path"
    assert deep == 0.0, (
        "a deeper decoder layer received gradient — the readout is no longer hs[0], i.e. no "
        "longer the reference architecture")


def test_truncating_the_decoder_to_the_readout_depth_is_bit_identical():
    """`_decode` reads hs[0], so INFERENCE only RUNS one layer — and that must change
    nothing numerically.

    The stack is sequential, so entry 0 cannot depend on layers 2..N; running them and
    discarding them (which is what upstream does) is pure waste. Measured on the 100k-step
    checkpoint: 4.84 -> 3.01 ms per CUDA forward, 1.61x, max |Δaction| = 0.0. This pins the
    equality so the optimization can never start silently changing the policy — and so the
    truncation stays tied to `_READOUT_LAYER_INDEX` instead of being a stale hardcoded 1."""
    from fuji.s1_policy.model.dp_act import _READOUT_LAYER_INDEX

    model = _tiny_model(dec_layers=4).eval()
    batch = _batch(batch_size=2)
    with torch.no_grad():
        truncated = model.sample_actions(batch)
        # Force the full stack by asking for every layer, then read the same index.
        src, pos = model._encode_vision(batch["images"], batch.get("img_masks"))
        flat_state = model._flat_state(batch, 2, src.device)
        force_feat = model._force_feature(batch, src.device)
        z = torch.zeros((2, model.latent_dim), dtype=model.dtype, device=src.device)
        extra = [model.latent_out_proj(z), model.input_proj_robot_state(flat_state)]
        if model.force_token_proj is not None:
            extra.append(model.force_token_proj(force_feat).to(model.dtype))
        hs_full = model.transformer(
            src, model.query_embed.weight.to(model.dtype), pos,
            extra_tokens=[t.to(model.dtype) for t in extra],
            additional_pos_embed=model.additional_pos_embed.weight.to(model.dtype),
            decoder_layers=None)
        full = model.action_head(hs_full[_READOUT_LAYER_INDEX]).float()
    assert hs_full.shape[0] == 4, "decoder_layers=None must still run the whole stack"
    assert torch.equal(truncated, full), (
        "truncating the decoder to the readout depth changed the output — the readout index "
        "and the truncation depth have drifted apart, or a layer gained a side effect")


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_a_grad_enabled_forward_uses_every_decoder_layer_for_ddp(mode):
    """A grad-enabled forward must touch the WHOLE decoder stack even though only layer 1
    gets gradient — INCLUDING in eval() mode.

    Not a numerics requirement; it is DDP's. `Trainer` wraps the model in
    `DistributedDataParallel(find_unused_parameters=False)`, whose fast reducer requires every
    parameter to participate in the forward. A layer skipped for speed becomes UNUSED and step
    2 dies with "Expected to have finished reduction in the prior iteration".

    **The `eval` case is the load-bearing one.** Gating the truncation on `self.training` — the
    obvious choice — passes every single-process test and still breaks DDP, because
    `tests/test_ddp_training.py` deliberately trains ACT in `.eval()` mode (to remove dropout
    as a cross-rank divergence source). The gate is therefore `torch.is_grad_enabled()`. This
    test runs on one GPU, unlike the DDP suite it protects."""
    model = _tiny_model(dec_layers=4)
    model.train() if mode == "train" else model.eval()
    out = model(_batch(batch_size=2))
    out["loss"].backward()
    named = dict(model.named_parameters())
    for i in range(4):
        got = [n for n in named if n.startswith(f"transformer.decoder.layers.{i}.")]
        assert got, f"decoder layer {i} has no parameters?"
        assert all(named[n].grad is not None for n in got), (
            f"in {mode}() mode decoder layer {i} did not participate in the grad-enabled "
            f"forward — DDP with find_unused_parameters=False will refuse this model on step 2")


def test_a_wrong_chunk_length_is_refused_with_a_legible_message():
    """Upstream truncates (policy.py:24-25); we refuse, because a wrong length means the
    dataset and the model disagree about the horizon. Without the explicit check it surfaced
    as an opaque tensor-size error from inside the CVAE encoder's attention."""
    model = _tiny_model()
    batch = _batch(batch_size=1)
    batch["actions"] = torch.randn(1, ACTION_HORIZON - 8, ACTION_DIM)
    batch["action_mask"] = torch.ones(1, ACTION_HORIZON - 8, ACTION_DIM)
    with pytest.raises(ValueError, match="action_horizon"):
        model(batch)


def test_the_visual_grid_is_a_token_set_not_a_pooled_vector():
    """THE architectural difference from the DiT/UNet siblings: ACT cross-attends to a
    spatial feature grid. A ResNet18 at 224x224 gives 7x7 = 49 tokens per frame; if this
    ever collapsed to 1 (a pooled CLS-like vector) the preset would have quietly become a
    worse copy of DP-Flow."""
    model = _tiny_model()
    feats, pos = model._encode_vision([torch.randn(1, 3, 224, 224)], None)
    assert feats.shape[2] * feats.shape[3] > 1, "the visual conditioning must be a token GRID"
    assert feats.shape[1] == HIDDEN, "input_proj must map to hidden_dim"
    assert pos.shape[2:] == feats.shape[2:], "position embedding must match the feature grid"


# ---------------------------------------------------------------------------
# Training forward
# ---------------------------------------------------------------------------
def test_forward_loss_finite_and_reports_both_terms():
    model = _tiny_model()
    out = model(_batch())
    assert torch.isfinite(out["loss"])
    # ACT's objective is L1 + kl_weight * KL; both terms must be visible, because a run
    # where the KL has collapsed to 0 (posterior == prior, latent unused) is a DIFFERENT
    # model than one where it has not, and the total alone cannot tell you.
    assert {"act_loss", "l1", "kl"} <= set(out["metrics"])
    assert out["loss"].item() == pytest.approx(
        out["metrics"]["l1"] + model.config.kl_weight * out["metrics"]["kl"], rel=1e-5)


def test_forward_loss_finite_without_any_masks():
    model = _tiny_model()
    out = model(_batch(with_masks=False))
    assert torch.isfinite(out["loss"])


def test_the_l1_term_divides_by_valid_count_not_numel():
    """DELIBERATE DEVIATION from upstream (policy.py:31 divides by numel()).

    With half the chunk masked, upstream's L1 halves purely because the denominator still
    counts the masked entries — which silently reweights L1 against the KL term sample by
    sample. On this dataset only 83.7% of anchors have a full 50-step horizon, so that
    drift would be real and per-sample. FUJI's L1 is a true masked mean, so masking half
    the steps must leave it UNCHANGED when the surviving residuals are unchanged.
    """
    model = _tiny_model()
    torch.manual_seed(0)
    pred = torch.randn(2, ACTION_HORIZON, ACTION_DIM)
    target = torch.zeros(2, ACTION_HORIZON, ACTION_DIM)

    full = torch.ones(2, ACTION_HORIZON, ACTION_DIM)
    half = full.clone()
    half[:, ACTION_HORIZON // 2 :, :] = 0.0

    l1_full = model._masked_l1(pred, target, None, None, full)
    # Zero the residual where the mask is 0, so the surviving entries are identical.
    l1_half = model._masked_l1(pred * half, target, None, None, half)
    assert l1_half.item() == pytest.approx(
        (pred[:, : ACTION_HORIZON // 2].abs().mean()).item(), rel=1e-5)
    # And it is NOT the numel-divided value, which would be ~half.
    assert l1_half.item() > 0.5 * l1_full.item() * 1.5


def test_masked_action_entries_do_not_reach_the_style_encoder():
    """A masked step holds a stale/zero value, not a measurement. The CVAE encoder attends
    across the WHOLE chunk, so leaking it would put non-existent information into z — and
    at inference z is 0, i.e. the model would have been conditioned at train time on
    something it can never see at deploy. Changing ONLY masked entries must not move the
    loss."""
    model = _tiny_model()
    batch_a = _batch(batch_size=1)
    mask = torch.ones(1, ACTION_HORIZON, ACTION_DIM)
    mask[:, 20:, :] = 0.0
    batch_a["action_mask"] = mask

    batch_b = {k: (v.clone() if torch.is_tensor(v) else [t.clone() for t in v])
               for k, v in batch_a.items()}
    batch_b["actions"][:, 20:, :] += 1000.0  # garbage, entirely inside the masked region

    torch.manual_seed(0)
    loss_a = model(batch_a)["loss"].item()
    torch.manual_seed(0)
    loss_b = model(batch_b)["loss"].item()
    assert loss_a == pytest.approx(loss_b, rel=1e-6), "masked entries must not affect the loss"


def test_action_mask_excludes_masked_entries_from_the_reported_sums():
    model = _tiny_model()
    batch = _batch()
    mask = torch.ones(2, ACTION_HORIZON, ACTION_DIM)
    mask[:, :, 10:] = 0.0  # mask the whole force block
    batch["action_mask"] = mask
    out = model(batch)
    num, den = out["metric_sums"]["force"]
    assert float(den) == 0.0, "a fully-masked group must contribute no denominator"
    assert float(num) == 0.0


def test_metric_sums_are_absolute_not_squared_errors():
    """ACT optimizes L1, so its reported sums must be order-1. If these were squared, the
    logged ``objective`` would not be the quantity being minimized and best-checkpoint
    selection would be steering by a different metric than the optimizer."""
    model = _tiny_model()
    out = model(_batch(batch_size=1))
    pred = out["aux"]["pred"].detach()
    target = torch.zeros_like(pred)
    from fuji.s1_policy.model.loss_metrics import absolute_error_sums

    expected = absolute_error_sums(pred, target)["overall"]
    got = absolute_error_sums(pred, target, group_dims={})["overall"]
    assert float(got[0]) == pytest.approx(float(expected[0]))
    # And the model's own overall matches its l1 term exactly.
    num, den = out["metric_sums"]["overall"]
    assert (float(num) / float(den)) == pytest.approx(out["metrics"]["l1"], rel=1e-5)


def test_kl_rides_in_metric_sums_as_a_sum_and_count():
    """The trainer aggregates metric_sums by exact sum/count over a whole validation pass;
    a mean-of-batch-means would be wrong on a ragged final batch. So KL must be a
    (sum, count) pair, not a pre-averaged scalar."""
    model = _tiny_model()
    out = model(_batch(batch_size=2))
    num, den = out["metric_sums"]["kl"]
    assert float(den) == 2.0, "the count must be the batch size"
    assert (float(num) / float(den)) == pytest.approx(out["metrics"]["kl"], rel=1e-5)


def test_a_fully_padded_sample_does_not_produce_nan():
    """The CLS and state tokens are never masked in the style encoder (upstream's
    cls_joint_is_pad). Without that, a sample whose every action step is padding masks
    every key and MultiheadAttention softmaxes over nothing -> NaN, poisoning the batch."""
    model = _tiny_model()
    batch = _batch(batch_size=1)
    batch["action_mask"] = torch.zeros(1, ACTION_HORIZON, ACTION_DIM)
    out = model(batch)
    assert torch.isfinite(out["loss"]), "an all-padding sample must not NaN the loss"


def test_a_step_masked_only_for_force_is_not_treated_as_padding():
    """Padding means "this step has no measurement at all". A step supervised for pose but
    masked for force still carries a real pose measurement, and dropping it from the style
    encoder would discard data the loss does supervise."""
    model = _tiny_model()
    mask = torch.ones(2, ACTION_HORIZON, ACTION_DIM)
    mask[:, :, 10:] = 0.0  # force masked at EVERY step, pose valid everywhere
    pad = model._pad_from_masks(mask, None, (2, ACTION_HORIZON), torch.device("cpu"))
    assert pad is not None and not pad.any(), "no step is padding — every step has a valid pose"


def test_explicit_action_is_pad_wins_over_the_derived_mask():
    model = _tiny_model()
    explicit = torch.zeros(2, ACTION_HORIZON, dtype=torch.bool)
    explicit[:, -5:] = True
    pad = model._pad_from_masks(torch.ones(2, ACTION_HORIZON, ACTION_DIM), explicit,
                                (2, ACTION_HORIZON), torch.device("cpu"))
    assert torch.equal(pad, explicit)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def test_sample_actions_shape_and_single_pass():
    model = _tiny_model()
    out = model.sample_actions(_batch(batch_size=3))
    assert out.shape == (3, ACTION_HORIZON, ACTION_DIM)
    assert torch.isfinite(out).all()


def test_sampling_is_deterministic_in_eval_mode_without_a_seed():
    """ACT decodes from the prior MEAN (z = 0), so a rollout is bit-reproducible with no
    generator at all — unlike both siblings, which need a seeded generator. This is what
    makes an ACT deploy repeatable, and it is worth pinning because the mechanism (z = 0)
    is one line that a "let's sample the latent" change would silently remove."""
    model = _tiny_model()
    batch = _batch()
    a = model.sample_actions(batch)
    b = model.sample_actions(batch)
    assert torch.equal(a, b)


def test_dropout_makes_sampling_nondeterministic_in_train_mode():
    """The flip side, pinned so nobody "simplifies away" the .eval() calls in the loaders:
    the DETR transformer carries dropout=0.1, so train() mode is stochastic even at z=0.
    sample_actions deliberately does NOT flip the mode itself (that would corrupt a
    mid-epoch diagnostic), so the callers' .eval() is load-bearing."""
    model = _tiny_model().train()
    batch = _batch()
    a = model.sample_actions(batch)
    b = model.sample_actions(batch)
    assert not torch.equal(a, b), "dropout should be active in train() mode"


def test_sampling_the_latent_is_opt_in_and_then_honors_the_generator():
    model = _tiny_model(sample_latent_at_inference=True)
    batch = _batch()
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    g3 = torch.Generator().manual_seed(8)
    same = torch.equal(model.sample_actions(batch, generator=g1),
                       model.sample_actions(batch, generator=g2))
    assert same, "the same seed must reproduce the rollout"
    assert not torch.equal(model.sample_actions(batch, generator=torch.Generator().manual_seed(7)),
                           model.sample_actions(batch, generator=g3))


@pytest.mark.parametrize("num_steps", [0, 2, 10, 16])
def test_sample_actions_refuses_any_step_count_but_one(num_steps):
    """A CVAE has no sampler loop. Accepting num_steps=16 would let an operator believe
    they traded compute for quality when nothing changed — so it is refused, not ignored."""
    model = _tiny_model()
    with pytest.raises(ValueError, match="one forward pass"):
        model.sample_actions(_batch(), num_steps=num_steps)


def test_sample_actions_accepts_one_and_none():
    model = _tiny_model()
    batch = _batch()
    assert torch.equal(model.sample_actions(batch, num_steps=1), model.sample_actions(batch))


def test_sampler_rejects_rtc_arguments():
    """There is no iterative trajectory whose prefix could be frozen. A caller wiring RTC
    by mistake must fail loudly rather than silently get un-chunked actions."""
    model = _tiny_model()
    with pytest.raises(NotImplementedError, match="Real-Time Chunking"):
        model.sample_actions(_batch(), delay_estimate=3)
    with pytest.raises(NotImplementedError, match="steps_executed"):
        model.sample_actions(_batch(), steps_executed=4)
    # Zero/absent is the normal path.
    model.sample_actions(_batch(), delay_estimate=0, steps_executed=0)


def test_reset_clears_the_cached_chunk():
    model = _tiny_model()
    model.sample_actions(_batch())
    assert model._last_actions_valid is True
    model.reset()
    assert model._last_actions_valid is False


def test_the_chunk_tracks_the_OBSERVATION_and_is_never_a_stale_cache():
    """THE deploy-safety test. `_last_actions` exists only for serving's reset()/diagnostics
    parity, but a sampler that RETURNED it would freeze the policy at its first chunk while
    every structural assertion still passed — verified by mutation: returning the cache in
    the exact deploy configuration (eval mode, deterministic latent) left the whole suite
    green, and drove a LivePlanner chunk 0.9 m away from what the live observation warranted.

    Deliberately run in eval() with the default z=0, i.e. the configuration where the model
    IS otherwise deterministic; the other determinism tests hold the observation fixed, so
    only this one can distinguish "deterministic" from "frozen"."""
    model = _tiny_model()
    torch.manual_seed(0)
    a = _batch(batch_size=1)
    b = {k: (v.clone() if torch.is_tensor(v) else [t.clone() for t in v]) for k, v in a.items()}
    b["images"] = [torch.randn_like(a["images"][0])]  # a genuinely different scene

    first = model.sample_actions(a)
    second = model.sample_actions(b)
    assert not torch.equal(first, second), \
        "a new observation must produce a new chunk — the sampler is returning a stale cache"
    # Re-presenting the ORIGINAL observation must reproduce the original chunk, which rules
    # out the opposite failure (an internal state that drifts between calls).
    assert torch.equal(model.sample_actions(a), first)


def test_the_force_INPUT_actually_changes_the_prediction():
    """Force in/out is the reason this preset exists on FUJI rather than upstream ALOHA, and
    every other force test is satisfied by a token that is present but INERT: multiplying
    the force token by 0.0 (leaving the encoder, the projection and the token count intact)
    passed the entire suite. So assert the only thing that matters — that changing ONLY the
    wrench window changes the output."""
    model = _tiny_model()
    torch.manual_seed(0)
    a = _batch(batch_size=1)
    b = {k: (v.clone() if torch.is_tensor(v) else [t.clone() for t in v]) for k, v in a.items()}
    b["force_history"] = a["force_history"] + 5.0  # same shape, a different contact

    assert not torch.equal(model.sample_actions(a), model.sample_actions(b)), \
        "the force token is present but inert — force is not reaching the decoder"


def test_the_force_VALIDITY_flag_changes_the_prediction():
    """The companion to the above: an unmeasured window must be presented as "no data" (the
    learned embedding), not as a measurement. If the flag were dropped on the way to the
    encoder, a stale hold would read as a real zero-force reading."""
    model = _tiny_model()
    torch.manual_seed(0)
    with torch.no_grad():  # a non-zero invalid_embed, else zero-init hides the substitution
        model.force_encoder.invalid_embed.fill_(0.7)
    a = _batch(batch_size=1, force_valid=True)
    b = {k: (v.clone() if torch.is_tensor(v) else [t.clone() for t in v]) for k, v in a.items()}
    b["force_history_valid"] = torch.zeros(1, dtype=torch.bool)

    assert not torch.equal(model.sample_actions(a), model.sample_actions(b)), \
        "force_history_valid is not reaching the encoder"


def test_sample_actions_relocates_a_cpu_generator_for_a_cuda_model():
    """Guarded on CPU: the relocation helper must be a no-op when devices already match,
    which is the property that keeps the CUDA path from double-seeding."""
    model = _tiny_model(sample_latent_at_inference=True)
    g = torch.Generator().manual_seed(3)
    assert model._device_generator(g, torch.device("cpu")) is g
    assert model._device_generator(None, torch.device("cpu")) is None


# ---------------------------------------------------------------------------
# Force path
# ---------------------------------------------------------------------------
def test_force_adds_exactly_one_token_and_no_force_matches_upstream_token_count():
    """The force token is FUJI's one addition to ACT's sequence (ALOHA has no F/T sensor).
    It must be exactly one token appended AFTER z and state, because additional_pos_embed
    rows are positional: inserting anywhere else would re-map every later token."""
    with_force = _tiny_model()
    assert with_force._n_extra_tokens == 3
    assert with_force.additional_pos_embed.num_embeddings == 3

    without = _tiny_model(force_hist_len=0, force_hist_channels=0)
    assert without._n_extra_tokens == 2, "no force -> byte-identical to upstream ACT"
    assert without.force_encoder is None and without.force_token_proj is None
    out = without(_batch(force_channels=0))
    assert torch.isfinite(out["loss"])


def test_missing_force_history_raises_rather_than_reading_as_no_contact():
    """A zero-filled window is indistinguishable from a free-space-tared 'no contact'
    measurement, so a silent default would teach the model to trust fabricated data."""
    model = _tiny_model()
    batch = _batch()
    del batch["force_history"]
    with pytest.raises(KeyError, match="no contact"):
        model(batch)


def test_an_invalid_force_window_uses_the_learned_no_data_embedding():
    """Shares ForceHistoryEncoder with both siblings, so 'no data' means the same thing to
    all three models. The encoding must depend on the VALID flag, not just the values."""
    model = _tiny_model()
    torch.manual_seed(0)
    fh = torch.randn(2, 1, 6)
    with torch.no_grad():
        model.force_encoder.invalid_embed.fill_(0.5)
        valid = model.force_encoder(fh, torch.ones(2, dtype=torch.bool))
        invalid = model.force_encoder(fh, torch.zeros(2, dtype=torch.bool))
    assert not torch.allclose(valid, invalid)
    assert torch.allclose(invalid, torch.full_like(invalid, 0.5))


def test_the_style_encoder_sees_force_only_when_asked():
    """Upstream's style encoder deliberately sees proprioception + actions and never the
    image: observations belong on the DECODER side. Force is an observation, so it is
    off by default; the opt-in exists but must not be silently on."""
    default = _tiny_model()
    assert default.encoder_force_proj is None
    assert default._n_vae_prefix == 2  # [CLS] + state

    opted_in = _tiny_model(encoder_sees_force=True)
    assert opted_in.encoder_force_proj is not None
    assert opted_in._n_vae_prefix == 3
    assert torch.isfinite(opted_in(_batch())["loss"])


# ---------------------------------------------------------------------------
# Observation shape guards
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("To", [1, 2, 3])
def test_history_forward_and_sample(To):
    model = _tiny_model(obs_history_len=To)
    out = model(_batch(To=To))
    assert torch.isfinite(out["loss"])
    assert model.sample_actions(_batch(To=To)).shape == (2, ACTION_HORIZON, ACTION_DIM)


def test_a_wrong_number_of_image_entries_is_refused():
    """collate flattens cameras and history into ONE list, so a count mismatch would
    silently change the visual token set rather than raise.

    The IMAGE count is varied alone (state is left at the correct To=2 width), because
    the state-width guard runs first in ``_decode`` and would otherwise mask this one."""
    model = _tiny_model(obs_history_len=2)
    batch = _batch(To=2)
    batch["images"] = batch["images"][:1]
    batch["img_masks"] = batch["img_masks"][:1]
    with pytest.raises(ValueError, match="num_cameras \\* obs_history_len"):
        model(batch)


def test_a_wrong_state_width_is_refused():
    model = _tiny_model(obs_history_len=1)
    batch = _batch()
    batch["state"] = torch.randn(2, 1, STATE_DIM + 3)
    batch["state_valid"] = torch.ones(2, 1, STATE_DIM + 3)
    with pytest.raises(ValueError, match="obs_history_len \\* state_dim"):
        model(batch)


def test_state_valid_zeroes_fabricated_entries():
    """0.0 is a LEGAL gripper width, so an ungated invalid entry feeds the model a made-up
    measurement. Same convention as both siblings."""
    model = _tiny_model()
    batch = _batch(batch_size=1)
    batch["state"] = torch.full((1, 1, STATE_DIM), 5.0)
    batch["state_valid"] = torch.zeros(1, 1, STATE_DIM)
    flat = model._flat_state(batch, 1, torch.device("cpu"))
    assert torch.count_nonzero(flat) == 0


def test_a_masked_out_camera_contributes_a_zero_feature():
    """A missing camera must never inject garbage; it contributes zeros, matching the
    siblings' zero-feature convention."""
    model = _tiny_model()
    img = torch.randn(2, 3, 224, 224)
    masks = [torch.tensor([True, False])]
    feats, _ = model._encode_vision([img], masks)
    assert torch.count_nonzero(feats[1]) == 0
    assert torch.count_nonzero(feats[0]) > 0


# ---------------------------------------------------------------------------
# Per-group metrics and the two-LR split
# ---------------------------------------------------------------------------
def test_forward_emits_per_group_metrics_under_the_act_namespace():
    model = _tiny_model()
    metrics = model(_batch())["metrics"]
    for group in ("left", "pos", "rot", "grip", "force"):
        assert f"act_loss/{group}" in metrics
    assert "act_loss/right" not in metrics, "left_only must not log a right-hand group"


def test_per_group_metrics_are_skipped_without_a_layout():
    model = _tiny_model(layout_mode="")
    metrics = model(_batch())["metrics"]
    assert not any(k.startswith("act_loss/") for k in metrics)


def test_a_per_group_metric_matches_a_manual_masked_l1():
    model = _tiny_model()
    batch = _batch(batch_size=2)
    mask = torch.ones(2, ACTION_HORIZON, ACTION_DIM)
    mask[:, 30:, :] = 0.0
    batch["action_mask"] = mask
    out = model(batch)
    pred = out["aux"]["pred"].detach()
    target = batch["actions"] * mask
    pos_dims = [0, 1, 2]
    manual = (pred[:, :30, pos_dims] - target[:, :30, pos_dims]).abs().mean().item()
    assert out["metrics"]["act_loss/pos"] == pytest.approx(manual, rel=1e-4)


def test_param_groups_split_the_trunk_from_the_transformer():
    """ACT's published recipe trains the ImageNet trunk at its own (lower) LR; one shared
    LR either destroys the pretrained features or starves the transformer."""
    model = DPACTModel(
        DPACTConfig(hidden_dim=64, dim_feedforward=128, nheads=4, enc_layers=1, dec_layers=1,
                    vae_enc_layers=1, latent_dim=4, action_dim=ACTION_DIM,
                    action_horizon=8, state_dim=STATE_DIM, layout_mode="left_only",
                    pretrained=False, lr_backbone=1e-5),
    )
    groups = model.param_groups(1e-4)
    assert [g["lr"] for g in groups] == [1e-4, 1e-5]
    trunk_params = sum(p.numel() for p in groups[1]["params"])
    assert trunk_params > 0, "the ResNet trunk must land in the backbone group"
    # Every parameter is accounted for exactly once — a dropped group means silently
    # frozen weights.
    assigned = sum(p.numel() for g in groups for p in g["params"])
    assert assigned == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_a_model_without_a_trunk_group_still_returns_one_group():
    """The stub-backbone path has no parameter named 'backbone', which must yield a single
    group rather than an empty second one (AdamW rejects an empty param group)."""
    model = _tiny_model()
    groups = model.param_groups(1e-4)
    assert all(len(g["params"]) > 0 for g in groups)
    torch.optim.AdamW(groups, lr=1e-4)  # must not raise


def test_the_trainer_ACTUALLY_consumes_param_groups(tmp_path, monkeypatch):
    """`param_groups` being correct is useless if the Trainer ignores it. Verified by
    mutation: replacing the trainer's `model.param_groups(lr)` with plain
    `model.parameters()` — collapsing ACT to ONE learning rate and training the
    ImageNet trunk at the transformer's LR — passed every other test in the suite.

    Also pins that LambdaLR captures BOTH base LRs, which is what makes the 10x
    trunk/head ratio survive warmup + cosine for the whole run."""
    from tests.test_hl_dataset import _write_hl_zarr

    from fuji.s1_policy.model.dp_act import DPACTConfig
    from fuji.s1_policy.training.config import HLDataConfig, TrainConfig
    from fuji.s1_policy.training.trainer import Trainer

    pytest.importorskip("zarr")
    zarr_path = _write_hl_zarr(tmp_path / "pg.zarr.zip", n_frames=8, n_episodes=2, image_hw=8)
    cfg = TrainConfig(
        algo="act", pipeline="hl", device="cpu", lr=1e-4,
        act=DPACTConfig(hidden_dim=32, dim_feedforward=64, nheads=4, enc_layers=1,
                        dec_layers=1, vae_enc_layers=1, latent_dim=4, force_enc_width=8,
                        force_enc_out_dim=8, pretrained=False, lr_backbone=1e-5,
                        require_norm_stats=False),
        hl_data=HLDataConfig(zarr_paths=(zarr_path,), drop_glitch_windows=False,
                             val_last_episodes=1,
                             # ACT refuses to train unnormalized; this test is about
                             # the optimizer, not normalization, so opt out on the model.
                             ),
    )
    trainer = Trainer(cfg)
    assert len(trainer.optimizer.param_groups) == 2, \
        "the trainer is not using the model's param_groups — ACT is training at ONE lr"
    # base_lrs, not the live `lr`: LambdaLR has already applied the step-0 warmup factor
    # (1/warmup_steps) by construction. base_lrs is what the schedule multiplies, so this is
    # the assertion that pins the 10x trunk/head ratio for the whole run.
    assert trainer.scheduler.base_lrs == [1e-4, 1e-5], \
        "the scheduler must carry both base LRs, or the trunk/head ratio drifts"
    # The ratio survives the schedule: both groups scale by the same factor.
    live = [g["lr"] for g in trainer.optimizer.param_groups]
    assert live[0] / live[1] == pytest.approx(10.0), "the trunk/head LR ratio was not preserved"


def test_flow_still_gets_a_single_optimizer_group(tmp_path, monkeypatch):
    """The other half of the same seam: at the DEFAULT ``encoder_lr_scale=1.0`` the
    flow/diffusion optimizer must stay the single-group AdamW they were trained and
    deployed with.

    Flow and diffusion now DO define ``param_groups`` (they gained UMI's reduced encoder
    LR), so this pins the behavior rather than the absence of the method: at scale 1.0 the
    hook returns ``self.parameters()`` and the optimizer has exactly one group. Asserting
    ``not hasattr`` would have made adding the knob look like a regression."""
    from fuji.s1_policy.model.dp_act import DPACTModel
    from fuji.s1_policy.model.dp_diffusion import DPDiffusionConfig, DPDiffusionModel
    from fuji.s1_policy.model.dp_flow import DPFlowConfig, DPFlowModel
    from fuji.s1_policy.model.umi_parity import encoder_lr_param_groups

    assert hasattr(DPACTModel, "param_groups")
    for cls in (DPFlowModel, DPDiffusionModel):
        assert hasattr(cls, "param_groups")

    # Scale 1.0 -> the raw parameter iterator, so AdamW builds one group exactly as before.
    for config_cls in (DPFlowConfig, DPDiffusionConfig):
        assert config_cls().encoder_lr_scale == 1.0, "the default must preserve one LR"

    model = torch.nn.Sequential()
    model.add_module("encoder", torch.nn.Linear(2, 2))
    model.add_module("decoder", torch.nn.Linear(2, 2))
    groups = encoder_lr_param_groups(model, 1e-4, 1.0)
    assert not isinstance(groups, list), "scale 1.0 must return the plain parameter iterator"
    assert len(torch.optim.AdamW(groups, lr=1e-4).param_groups) == 1

    # Scale != 1.0 -> two groups, encoder scaled, everything else at the base lr.
    groups = encoder_lr_param_groups(model, 1e-4, 0.1)
    assert [g["lr"] for g in groups] == [1e-4, pytest.approx(1e-5)]
    assert [len(g["params"]) for g in groups] == [2, 2]  # weight+bias each side


# ---------------------------------------------------------------------------
# Config validation (TrainConfig-level guards)
# ---------------------------------------------------------------------------
def test_train_config_pushes_layout_dims_onto_the_act_block_only():
    """The trap this guards: ``active_model`` used to fall through to the FLOW config for
    any algo it did not name, so every layout-derived dimension landed on ``model`` while
    the ``act`` block shipped with never-configured defaults."""
    from fuji.s1_policy.training.config import HLDataConfig, TrainConfig

    cfg = TrainConfig(algo="act", pipeline="hl", hl_data=HLDataConfig(norm_stats_path="s.json"))
    assert cfg.active_model is cfg.act
    assert cfg.act.action_dim == ACTION_DIM
    # Against the LAYOUT, not this module's ACTION_HORIZON: that literal only sizes the stub
    # tensors below, while this assertion is about what _post_init_hl PUSHES, and the layout
    # value is re-pinnable per process (FUJI_ACTION_HORIZON, for an H=50 vs H=30 A/B).
    assert cfg.act.action_horizon == layout.ACTION_HORIZON
    assert cfg.act.state_dim == STATE_DIM
    assert cfg.act.layout_mode == "left_only"
    assert cfg.act.force_hist_channels == 6
    # The inactive flow block keeps its defaults, unmodified.
    assert cfg.model.layout_mode == "" and cfg.model.action_dim == 32


def test_act_defaults_to_no_ema():
    from fuji.s1_policy.training.config import HLDataConfig, TrainConfig

    hl = HLDataConfig(norm_stats_path="s.json")
    assert TrainConfig(algo="act", hl_data=hl).ema is False
    assert TrainConfig(algo="act", hl_data=hl, ema=True).ema is True


@pytest.mark.parametrize(("kwargs", "message"), [
    (dict(inference_steps=10), "ONE forward pass"),
    (dict(kl_weight=-1.0), "kl_weight"),
    (dict(latent_dim=0), "latent_dim"),
    (dict(hidden_dim=65), "even"),
    (dict(hidden_dim=64, nheads=5), "divisible"),
    (dict(lr_backbone=float("nan")), "lr_backbone"),
    (dict(backbone="resnet99"), "backbone"),
])
def test_act_config_validation(kwargs, message):
    from fuji.s1_policy.training.config import HLDataConfig, TrainConfig

    with pytest.raises(ValueError, match=message):
        TrainConfig(algo="act", act=DPACTConfig(**kwargs),
                    hl_data=HLDataConfig(norm_stats_path="s.json"))


def test_act_refuses_to_train_unnormalized():
    """ACT's L1 is taken over the RAW action vector, which mixes metres / rot6d / a [0,1]
    gripper fraction / N and N·m. Measured unnormalized block shares: pos 7.6%, rot 22.6%,
    grip 21.6%, force 43.3%, **torque 4.9%** — the torque columns end up effectively
    unsupervised while the loss still looks perfectly healthy. Same class of silent trap the
    diffusion sibling's clip_sample refusal exists for."""
    from fuji.s1_policy.training.config import HLDataConfig, TrainConfig

    with pytest.raises(ValueError, match="requires a norm_stats path"):
        TrainConfig(algo="act", pipeline="hl")
    # Explicit override, and the normal path, both work.
    TrainConfig(algo="act", act=DPACTConfig(require_norm_stats=False))
    TrainConfig(algo="act", hl_data=HLDataConfig(norm_stats_path="stats.json"))


def test_rtc_is_refused_for_act_instead_of_landing_on_the_flow_config():
    """The old guard named only diffusion, so ``--rtc-delay-range`` with algo='act' skipped
    the refusal and then wrote ``self.model.rtc_delay_range`` — mutating the INACTIVE flow
    config while the ACT run silently got no RTC."""
    from fuji.s1_policy.training.config import HLDataConfig, TrainConfig

    with pytest.raises(ValueError, match="Real-Time Chunking"):
        TrainConfig(algo="act", rtc_delay_range=(0, 5),
                    hl_data=HLDataConfig(norm_stats_path="s.json"))


def test_legacy_pipeline_is_refused_for_act():
    """``_post_init_legacy`` hardcodes ``self.model.*``, so this combination would have
    configured the FLOW config for an ACT run."""
    from fuji.s1_policy.training.config import HLDataConfig, TrainConfig

    with pytest.raises(ValueError, match="flow-only"):
        TrainConfig(algo="act", pipeline="legacy",
                    hl_data=HLDataConfig(norm_stats_path="s.json"))


def test_an_unknown_algo_is_refused_by_both_the_enum_and_active_model():
    from fuji.s1_policy.training.config import HLDataConfig, TrainConfig

    with pytest.raises(ValueError, match="algo must be one of"):
        TrainConfig(algo="bogus")
    cfg = TrainConfig(algo="act", hl_data=HLDataConfig(norm_stats_path="s.json"))
    object.__setattr__(cfg, "algo", "bogus")  # bypass __post_init__
    with pytest.raises(ValueError, match="algo must be one of"):
        _ = cfg.active_model


def test_the_algo_config_field_map_covers_every_algo_and_names_real_fields():
    """The map is the single source of truth for "which config.json block is real". Every
    entry must name an actual TrainConfig field, or a loader reads a missing key."""
    import dataclasses

    from fuji.s1_policy.training.config import ALGO_CONFIG_FIELD, ALGOS, TrainConfig

    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    assert set(ALGO_CONFIG_FIELD) == set(ALGOS)
    for algo, field_name in ALGO_CONFIG_FIELD.items():
        assert field_name in fields, f"{algo!r} maps to non-existent field {field_name!r}"
    # Distinct blocks: two algos sharing one block would make a checkpoint ambiguous.
    assert len(set(ALGO_CONFIG_FIELD.values())) == len(ALGO_CONFIG_FIELD)


def test_a_checkpoint_predating_a_new_algo_still_resumes():
    """REGRESSION GUARD. Resume compares the whole serialized TrainConfig by exact dict
    equality, so ADDING an algorithm adds a key that every already-written checkpoint
    lacks — which silently broke resume for ckpt_flow_final the moment `act` landed
    (verified: the only difference was the added key).

    An INACTIVE block missing from one side is elided; the ACTIVE block, `algo` itself, and
    any key present on BOTH sides are still compared in full."""
    from fuji.s1_policy.training.trainer import _align_inactive_algo_blocks

    # A flow checkpoint written before `act` existed.
    saved = {"algo": "flow", "model": {"vit": "vit_b"}, "diffusion": {"vit": "vit_b"}}
    current = {"algo": "flow", "model": {"vit": "vit_b"}, "diffusion": {"vit": "vit_b"},
               "act": {"backbone": "resnet18"}}
    a, b = _align_inactive_algo_blocks(saved, current)
    assert a == b, "a pre-existing flow resume must not be broken by the added act block"
    assert "act" not in a and "act" not in b
    # The caller's dicts are untouched.
    assert "act" in current

    # The ACTIVE block is never elided: an act run whose own block is missing must refuse.
    a, b = _align_inactive_algo_blocks(
        {"algo": "act", "model": {}}, {"algo": "act", "model": {}, "act": {"backbone": "resnet18"}})
    assert a != b, "the ACTIVE block must always be compared — it defines the architecture"

    # A real EDIT to an inactive block (present on both sides) still refuses.
    a, b = _align_inactive_algo_blocks(
        {"algo": "flow", "model": {}, "act": {"kl_weight": 10}},
        {"algo": "flow", "model": {}, "act": {"kl_weight": 1}})
    assert a != b

    # Different algorithms are never bridged.
    a, b = _align_inactive_algo_blocks({"algo": "flow", "model": {"x": 1}},
                                       {"algo": "act", "act": {"y": 2}})
    assert a != b


def test_the_trainer_algo_table_agrees_with_the_config_map():
    """Two tables (trainer model classes, config field names) that must not drift: a
    mismatch builds the right model from the wrong config block."""
    from fuji.s1_policy.training.config import ALGO_CONFIG_FIELD
    from fuji.s1_policy.training.trainer import _ALGO_MODELS

    assert set(_ALGO_MODELS) == set(ALGO_CONFIG_FIELD)
    for algo, (model_name, field_name, loss_key) in _ALGO_MODELS.items():
        assert field_name == ALGO_CONFIG_FIELD[algo]
        assert loss_key.endswith("_loss")
        # The class is held by NAME and resolved late (so tests can monkeypatch a stub in);
        # that only works if the name really is importable from the trainer's namespace.
        import fuji.s1_policy.training.trainer as trainer_module

        assert isinstance(getattr(trainer_module, model_name), type), \
            f"{algo!r} names {model_name!r}, which is not a class in the trainer's globals"
