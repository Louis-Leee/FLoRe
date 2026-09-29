"""Tests for the causal force-history encoder.

The load-bearing property is CAUSALITY: the encoder sees a window that ends at the
observation instant, and no output may depend on a bin later than itself. A leak here
would let the policy condition on the future during training and then face a strictly
poorer input at deploy — the classic silently-optimistic failure.
"""

from __future__ import annotations

import pytest
import torch

from fuji.s1_policy.data.layout import FORCE_DIM, FORCE_HIST_LEN
from fuji.s1_policy.model.force_encoder import ForceHistoryEncoder


def _enc(**kw):
    kw.setdefault("hist_len", FORCE_HIST_LEN)
    kw.setdefault("channels", FORCE_DIM)
    kw.setdefault("width", 32)
    kw.setdefault("out_dim", 16)
    return ForceHistoryEncoder(**kw)


def test_output_shape_and_dtype():
    enc = _enc()
    out = enc(torch.randn(4, FORCE_HIST_LEN, FORCE_DIM))
    assert out.shape == (4, 16)
    assert out.dtype == torch.float32


def test_conv_stack_is_causal():
    """Perturbing the NEWEST bin must not change any earlier conv output. (It must
    change the last one, or the newest measurement is being ignored.)"""
    torch.manual_seed(0)
    enc = _enc().eval()
    x = torch.randn(1, FORCE_HIST_LEN, FORCE_DIM)
    with torch.no_grad():
        h = enc.conv(x.transpose(1, 2))
        x2 = x.clone()
        x2[0, -1] = 99.0
        h2 = enc.conv(x2.transpose(1, 2))
    assert torch.allclose(h[:, :, :-1], h2[:, :, :-1], atol=1e-6), "future leaked backward"
    assert not torch.allclose(h[:, :, -1], h2[:, :, -1], atol=1e-6), "newest bin ignored"


def test_receptive_field_is_five_bins():
    """Two k=3 causal convs -> a 5-bin receptive field. Pinned because it is the documented
    reach, and a padding change would silently alter it.

    Exercised at hist_len=8 explicitly: the v3 contract feeds ONE bin (t0 only, no
    look-back), where "how far back does a perturbation reach" is not expressible. The
    encoder's reach is a property of its architecture, not of the live contract, so it is
    tested on a window long enough to show it — and test_single_bin_history_is_supported
    covers the shape the v3 grid actually uses."""
    L = 8
    torch.manual_seed(0)
    enc = _enc(hist_len=L).eval()
    x = torch.randn(1, L, FORCE_DIM)
    with torch.no_grad():
        h = enc.conv(x.transpose(1, 2))
        x3 = x.clone()
        x3[0, 0] = 99.0                       # perturb only the OLDEST bin
        h3 = enc.conv(x3.transpose(1, 2))
    changed = [not torch.allclose(h[:, :, i], h3[:, :, i], atol=1e-6)
               for i in range(L)]
    assert changed[:5] == [True] * 5, changed
    assert changed[5:] == [False] * (L - 5), changed


def test_single_bin_history_is_supported():
    """The v3 contract's actual shape: FORCE_HIST_LEN == 1 (t0 only).

    A 1-length sequence is the degenerate case for a causal conv stack (left-padding is
    the entire input), so pin that it still produces a finite encoding of the right width
    and that the encoding RESPONDS to the single bin's value — a collapse to a constant
    here would silently discard the force input altogether."""
    torch.manual_seed(0)
    enc = _enc(hist_len=1).eval()
    x = torch.randn(2, 1, FORCE_DIM)
    with torch.no_grad():
        y = enc(x)
        x2 = x.clone()
        x2[0, 0, 0] += 5.0
        y2 = enc(x2)
    assert y.shape == (2, 16) and torch.isfinite(y).all()
    assert not torch.allclose(y[0], y2[0], atol=1e-4), "encoding ignores the only bin"
    # The untouched batch row must be unaffected (no cross-sample leakage).
    assert torch.allclose(y[1], y2[1], atol=1e-6)


def test_readout_sees_both_the_newest_bin_and_the_window_mean():
    """The readout concatenates the final timestep with a mean over time, so an identical
    newest bin with a different history still yields a different encoding — otherwise the
    temporal profile (the whole point of a history) would be discarded."""
    L = 8   # a temporal profile needs >1 bin; the v3 contract's 1 bin cannot express one
    torch.manual_seed(0)
    enc = _enc(hist_len=L).eval()
    a = torch.zeros(1, L, FORCE_DIM)
    b = torch.zeros(1, L, FORCE_DIM)
    a[0, :, 0] = torch.linspace(0.0, 5.0, L)      # rising into contact
    b[0, :, 0] = torch.linspace(10.0, 5.0, L)     # releasing to the SAME value
    assert torch.allclose(a[0, -1], b[0, -1])                   # identical newest bin
    with torch.no_grad():
        assert not torch.allclose(enc(a), enc(b), atol=1e-4)


def test_invalid_window_uses_the_learned_embedding_not_the_data():
    """An unmeasured window must NOT be presented as a measurement of zero force: a
    free-space-tared wrench is ~0, so zero and no-data are otherwise indistinguishable."""
    enc = _enc().eval()
    x = torch.randn(3, FORCE_HIST_LEN, FORCE_DIM) * 10.0
    valid = torch.tensor([True, False, False])
    with torch.no_grad():
        out = enc(x, valid)
    # Both invalid rows collapse to the SAME learned embedding, regardless of their data.
    assert torch.allclose(out[1], out[2], atol=1e-6)
    assert torch.allclose(out[1], enc.invalid_embed, atol=1e-6)
    assert not torch.allclose(out[0], out[1], atol=1e-4)


def test_invalid_embed_is_zero_initialised():
    """Zero init: an untrained model treats an unmeasured window as a neutral
    contribution rather than injecting a random bias."""
    assert torch.count_nonzero(_enc().invalid_embed) == 0


def test_valid_none_treats_every_sample_as_measured():
    enc = _enc().eval()
    x = torch.randn(2, FORCE_HIST_LEN, FORCE_DIM)
    with torch.no_grad():
        assert torch.allclose(enc(x, None), enc(x, torch.tensor([True, True])), atol=1e-6)


def test_gradients_reach_every_parameter():
    torch.manual_seed(0)
    enc = _enc()
    enc(torch.randn(4, FORCE_HIST_LEN, FORCE_DIM)).sum().backward()
    for name, p in enc.named_parameters():
        if name == "invalid_embed":
            continue                      # only trains when a sample is invalid
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_invalid_embed_trains_only_when_a_sample_is_invalid():
    torch.manual_seed(0)
    enc = _enc()
    x = torch.randn(2, FORCE_HIST_LEN, FORCE_DIM)
    enc(x, torch.tensor([True, True])).sum().backward()
    assert enc.invalid_embed.grad.abs().sum() == 0
    enc.zero_grad()
    enc(x, torch.tensor([True, False])).sum().backward()
    assert enc.invalid_embed.grad.abs().sum() > 0


def test_bf16_forward():
    """The model runs under bf16 autocast; the encoder must not force a dtype mismatch
    in the global_cond concat."""
    enc = _enc().to(torch.bfloat16)
    out = enc(torch.randn(2, FORCE_HIST_LEN, FORCE_DIM), torch.tensor([True, False]))
    assert out.dtype == torch.bfloat16 and out.shape == (2, 16)


def test_multi_hand_channel_width():
    enc = _enc(channels=2 * FORCE_DIM)
    assert enc(torch.randn(2, FORCE_HIST_LEN, 2 * FORCE_DIM)).shape == (2, 16)


def test_shape_mismatch_raises_rather_than_broadcasting():
    """A window/channel mismatch mis-times or mis-routes every force input, so it must be
    loud — not silently reshaped."""
    enc = _enc()
    with pytest.raises(ValueError, match="built for"):
        enc(torch.randn(2, 4, FORCE_DIM))                 # wrong history length
    with pytest.raises(ValueError, match="built for"):
        enc(torch.randn(2, FORCE_HIST_LEN, 3))            # wrong channel count
    with pytest.raises(ValueError, match=r"\(B, L, C\)"):
        enc(torch.randn(2, FORCE_HIST_LEN))               # missing an axis


def test_degenerate_construction_raises():
    with pytest.raises(ValueError):
        _enc(hist_len=0)
    with pytest.raises(ValueError):
        _enc(channels=0)


def test_both_tensors_are_moved_together_by_the_model():
    """REGRESSION. dp_flow's _encode_global_cond explicitly does fh.to(device), which
    documents intent to accept a HOST-side batch (serving / eval / a notebook). Leaving
    the companion validity flag behind fails exactly there, and is invisible in training
    because Trainer._move_batch moves everything. Move both or neither."""
    import inspect

    from fuji.s1_policy.model import dp_flow

    src = inspect.getsource(dp_flow.DPFlowModel._encode_global_cond)
    assert "fh.to(device)" in src
    # The flag must be moved too (guarded for None).
    assert "fhv.to(device) if fhv is not None else None" in src


def test_encoder_accepts_mixed_placement_of_the_validity_flag():
    """The encoder itself must not care where the flag came from, as long as the caller
    moved it — this pins the arithmetic that broke (keep * out + (1-keep) * embed)."""
    enc = _enc().eval()
    x = torch.randn(3, FORCE_HIST_LEN, FORCE_DIM)
    with torch.no_grad():
        a = enc(x, torch.tensor([True, False, True]))
        b = enc(x, torch.tensor([1.0, 0.0, 1.0]))      # float flags work identically
    assert torch.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# action_mask must gate the decoder INPUT, not just the loss weight
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_draws", [1, 4])
@pytest.mark.parametrize("rtc", [None, (0, 5)])
def test_masked_targets_cannot_influence_the_loss(num_draws, rtc):
    """REGRESSION (was CRITICAL). action_mask gated only the loss WEIGHT, but `actions`
    also feeds x_t — the decoder's INPUT — and the DiT attends across the whole chunk. So
    a stale-held or zero-filled target at step 15 measurably steered the prediction at
    step 0 (observed max |dv| 2.5-3.3, loss shifted in the 3rd decimal).

    This test is the pin: two batches differing ONLY inside ~action_mask must give a
    bit-identical loss, every per-group metric, AND an identical v_t. Reverting the fix
    leaves the rest of the suite green, so nothing else catches it."""
    from fuji.s1_policy.model.dp_flow import DPFlowConfig, DPFlowModel

    torch.manual_seed(0)
    model = DPFlowModel(DPFlowConfig(
        vit="vit_b", pretrained=False, diffusion_net="dit", model_channels=64,
        num_blocks=1, n_heads=4, action_dim=16, action_horizon=16, state_dim=1,
        force_hist_len=FORCE_HIST_LEN, force_hist_channels=FORCE_DIM,
        layout_mode="left_only", num_draws=num_draws, rtc_delay_range=rtc)).eval()

    mask = torch.ones(2, 16, 16, dtype=torch.bool)
    mask[:, 12:, :] = False            # last 4 steps unsupervised
    mask[:, :, 10:] = False            # ...and the force block everywhere
    obs = {
        "images": [torch.randn(2, 3, 224, 224)],
        "img_masks": [torch.ones(2, dtype=torch.bool)],
        "state": torch.rand(2, 1),
        "force_history": torch.randn(2, FORCE_HIST_LEN, FORCE_DIM),
        "force_history_valid": torch.ones(2, dtype=torch.bool),
        "action_mask": mask,
    }
    a1 = torch.randn(2, 16, 16)
    a2 = a1.clone()
    a2[~mask] = 999.0                  # garbage ONLY where unsupervised
    assert torch.equal(a1[mask], a2[mask])

    torch.manual_seed(1)
    o1 = model({**obs, "actions": a1})
    torch.manual_seed(1)
    o2 = model({**obs, "actions": a2})

    assert o1["loss"].item() == o2["loss"].item(), (o1["loss"].item(), o2["loss"].item())
    for k in o1["metrics"]:
        assert o1["metrics"][k] == o2["metrics"][k], k
    # The decoder's own output must be untouched, not merely the reduced loss.
    assert torch.equal(o1["aux"]["v_t"], o2["aux"]["v_t"])


def test_legacy_mask_path_is_untouched_by_the_interpolant_gate():
    """The zeroing applies ONLY when action_mask is supplied. A legacy batch
    (action_is_pad + action_dim_mask, no action_mask) must be byte-identical to before, so
    the already-deployed 32-dim checkpoint stays reproducible."""
    import inspect

    from fuji.s1_policy.model import base

    src = inspect.getsource(base.FlowMatchingVLABase.forward)
    i = src.index("x_t = time_expanded")
    guard = src[max(0, i - 400):i]
    assert "if action_mask is not None:" in guard
    assert "actions = actions * action_mask" in guard
