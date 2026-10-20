"""DP-Flow model: forward-loss finiteness, sample shape, and the RTC hard-mask
invariant. Uses a tiny stub conv encoder so no timm weights are downloaded and the
tests run on CPU in well under a second."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from flore.s1_policy.model.dp_flow import DPFlowConfig, DPFlowModel

ACTION_DIM = 32
ACTION_HORIZON = 50


class StubEncoder(nn.Module):
    """Minimal pluggable encoder: (N, 3, H, W) -> (N, num_features). No downloads."""

    def __init__(self, num_features: int = 64):
        super().__init__()
        self.num_features = num_features
        self.conv = nn.Conv2d(3, num_features, kernel_size=16, stride=16)

    def forward(self, x):
        return self.conv(x).mean(dim=(-1, -2))  # global-average-pool to (N, num_features)


def _tiny_model(**cfg_overrides) -> DPFlowModel:
    defaults = dict(
        diffusion_net="dit",
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        num_cameras=1,
        model_channels=128,
        num_blocks=2,
        n_heads=4,
        diffusion_step_embed_dim=64,
        num_draws=1,
        pretrained=False,
    )
    defaults.update(cfg_overrides)
    return DPFlowModel(DPFlowConfig(**defaults), encoder=StubEncoder())


def _batch(batch_size: int = 2, with_masks: bool = True) -> dict:
    B = batch_size
    batch = {
        "images": [torch.randn(B, 3, 224, 224)],
        "img_masks": [torch.ones(B, dtype=torch.bool)],
        "state": torch.randn(B, ACTION_DIM),
        "actions": torch.randn(B, ACTION_HORIZON, ACTION_DIM),
    }
    if with_masks:
        batch["action_is_pad"] = torch.zeros(B, ACTION_HORIZON, dtype=torch.bool)
        dim_mask = torch.ones(B, ACTION_DIM, dtype=torch.bool)
        dim_mask[:, 29:] = False  # last 3 are zero-pad
        batch["action_dim_mask"] = dim_mask
    return batch


def test_forward_loss_finite():
    torch.manual_seed(0)
    model = _tiny_model()
    out = model(_batch())
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() > 0
    assert "flow_loss" in out["metrics"]


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


def test_action_dim_mask_excludes_pad_dims():
    """_compute_flow_loss must ignore masked-out dims: changing v_t/u_t on the
    masked (pad) dims must not change the loss."""
    torch.manual_seed(0)
    model = _tiny_model()
    B, T = 2, ACTION_HORIZON
    v_t = torch.randn(B, T, ACTION_DIM)
    u_t = torch.randn(B, T, ACTION_DIM)
    dim_mask = torch.ones(B, ACTION_DIM, dtype=torch.bool)
    dim_mask[:, 29:] = False

    loss_a = model._compute_flow_loss(v_t, u_t, None, None, dim_mask).item()
    v_t2 = v_t.clone()
    v_t2[:, :, 29:] += 100.0  # perturb only masked dims
    loss_b = model._compute_flow_loss(v_t2, u_t, None, None, dim_mask).item()
    assert loss_a == pytest.approx(loss_b, abs=1e-6)


def test_action_is_pad_and_prefix_mask_exclude_timesteps():
    """Temporal masks (action_is_pad, prefix_mask) drop whole timesteps from the loss."""
    torch.manual_seed(0)
    model = _tiny_model()
    B, T = 2, ACTION_HORIZON
    v_t = torch.randn(B, T, ACTION_DIM)
    u_t = torch.randn(B, T, ACTION_DIM)
    is_pad = torch.zeros(B, T, dtype=torch.bool)
    is_pad[:, T // 2 :] = True

    loss_a = model._compute_flow_loss(v_t, u_t, is_pad, None, None).item()
    v_t2 = v_t.clone()
    v_t2[:, T // 2 :] += 100.0  # perturb only padded timesteps
    loss_b = model._compute_flow_loss(v_t2, u_t, is_pad, None, None).item()
    assert loss_a == pytest.approx(loss_b, abs=1e-6)


def test_sample_actions_shape():
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    out = model.sample_actions(_batch(), num_steps=4)
    assert out.shape == (2, ACTION_HORIZON, ACTION_DIM)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_unet_decoder_forward_and_sample():
    """The UNet decoder path works for forward + non-RTC sampling."""
    torch.manual_seed(0)
    model = _tiny_model(diffusion_net="unet", num_draws=1)
    out = model(_batch())
    assert torch.isfinite(out["loss"])
    model.eval()
    s = model.sample_actions(_batch(), num_steps=3)
    assert s.shape == (2, ACTION_HORIZON, ACTION_DIM)


def test_rtc_frozen_prefix_hard_mask_invariant():
    """With a valid cache + delay_estimate>0, the first ``delay`` rows of the sampled
    chunk EXACTLY equal the shifted previous actions (RTC hard-mask invariant)."""
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    batch = _batch()

    # First call establishes the RTC cache (_last_actions).
    model.sample_actions(batch, num_steps=4)
    prev = model._last_actions.clone()  # (B, AH, AD)

    steps_executed = 4
    delay = 6
    sampled = model.sample_actions(batch, num_steps=4, steps_executed=steps_executed, delay_estimate=delay)

    B = sampled.shape[0]
    prefix_len = min(delay, ACTION_HORIZON - steps_executed)
    idx = (torch.arange(ACTION_HORIZON)[None, :, None] + steps_executed).clamp(0, ACTION_HORIZON - 1)
    idx = idx.expand(B, ACTION_HORIZON, ACTION_DIM)
    expected = torch.gather(prev.expand(B, -1, -1), 1, idx)

    np.testing.assert_allclose(
        sampled[:, :prefix_len].numpy(),
        expected[:, :prefix_len].numpy(),
        atol=1e-6,
    )


def test_rtc_no_cache_is_plain_sampling():
    """delay_estimate>0 but no valid cache (fresh reset) -> plain sampling, no crash."""
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    model.reset()
    out = model.sample_actions(_batch(), num_steps=4, steps_executed=2, delay_estimate=5)
    assert out.shape == (2, ACTION_HORIZON, ACTION_DIM)


def test_rtc_frozen_prefix_survives_smoothing():
    """policy_smooth_lam>0 must NOT move the frozen prefix rows (they are already
    executing on the robot). The smoother is applied to the suffix only; the
    prefix must stay bit-exact equal to the shifted previous chunk."""
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    batch = _batch()

    model.sample_actions(batch, num_steps=4)
    prev = model._last_actions.clone()

    steps_executed = 4
    delay = 6
    sampled = model.sample_actions(
        batch, num_steps=4, steps_executed=steps_executed, delay_estimate=delay,
        policy_smooth_lam=0.5,
    )

    B = sampled.shape[0]
    prefix_len = min(delay, ACTION_HORIZON - steps_executed)
    idx = (torch.arange(ACTION_HORIZON)[None, :, None] + steps_executed).clamp(0, ACTION_HORIZON - 1)
    idx = idx.expand(B, ACTION_HORIZON, ACTION_DIM)
    expected = torch.gather(prev.expand(B, -1, -1), 1, idx)

    np.testing.assert_allclose(
        sampled[:, :prefix_len].numpy(),
        expected[:, :prefix_len].numpy(),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Observation history (obs_history_len = To > 1)
# ---------------------------------------------------------------------------
def _history_batch(To: int, batch_size: int = 2) -> dict:
    """A batch with To pseudo-camera image entries and a (B, To, AD) state, as
    collate_umi_batch emits for obs_history_len == To."""
    B = batch_size
    dim_mask = torch.ones(B, ACTION_DIM, dtype=torch.bool)
    dim_mask[:, 29:] = False
    return {
        "images": [torch.randn(B, 3, 224, 224) for _ in range(To)],
        "img_masks": [torch.ones(B, dtype=torch.bool) for _ in range(To)],
        "state": torch.randn(B, To, ACTION_DIM),
        "actions": torch.randn(B, ACTION_HORIZON, ACTION_DIM),
        "action_is_pad": torch.zeros(B, ACTION_HORIZON, dtype=torch.bool),
        "action_dim_mask": dim_mask,
    }


@pytest.mark.parametrize("To", [2, 4])
def test_history_forward_and_sample(To):
    """obs_history_len=To: forward loss is finite and sample_actions has the right
    shape, with To pseudo-camera images and a (B, To, AD) state (zero-pad path)."""
    torch.manual_seed(0)
    model = _tiny_model(obs_history_len=To)
    out = model(_history_batch(To))
    assert torch.isfinite(out["loss"])
    model.eval()
    s = model.sample_actions(_history_batch(To), num_steps=3)
    assert s.shape == (2, ACTION_HORIZON, ACTION_DIM)
    assert torch.isfinite(s).all()


@pytest.mark.parametrize("To", [2, 3])
def test_history_forward_state_projection(To):
    """The learned state-projection path also handles a (B, To, AD) state (shared
    Linear applied per frame, then concatenated)."""
    torch.manual_seed(0)
    model = _tiny_model(obs_history_len=To, state_projection=True)
    out = model(_history_batch(To))
    assert torch.isfinite(out["loss"])
    model.eval()
    s = model.sample_actions(_history_batch(To), num_steps=3)
    assert s.shape == (2, ACTION_HORIZON, ACTION_DIM)


def test_history_grows_global_cond():
    """global_cond width scales with To (To pseudo-cameras + To state frames)."""
    m1 = _tiny_model(obs_history_len=1)
    m3 = _tiny_model(obs_history_len=3)
    # The decoder's global_cond_encoder input Linear reflects the conditioning width.
    w1 = m1.decoder.global_cond_encoder[1].in_features
    w3 = m3.decoder.global_cond_encoder[1].in_features
    # enc_out (num_features) + state (filled_state_dim) both ×To.
    enc = m1.encoder.num_features
    fsd = m1.config.filled_state_dim
    assert w1 == enc + fsd
    assert w3 == 3 * enc + 3 * fsd


# ---------------------------------------------------------------------------
# Per-group training metrics (flow_loss/<group>) — logging only, no gradient
# ---------------------------------------------------------------------------
def test_metric_groups_match_dataset_layout():
    """The model's per-group dim indices MUST mirror the dataset's 32-dim slot layout.
    If the dataset layout ever changes, this test fails so the metric groups get fixed."""
    from flore.s1_policy.data.umi_dataset import _HEAD, _LEFT, _RIGHT
    from flore.s1_policy.model.dp_flow import _METRIC_GROUP_DIMS

    def _kp_dims(spec):
        pos, rot, grip = spec
        dims = list(range(pos.start, pos.stop)) + list(range(rot.start, rot.stop))
        if grip is not None:
            dims.append(grip)
        return sorted(dims)

    assert sorted(_METRIC_GROUP_DIMS["left"]) == _kp_dims(_LEFT)
    assert sorted(_METRIC_GROUP_DIMS["right"]) == _kp_dims(_RIGHT)
    assert sorted(_METRIC_GROUP_DIMS["head"]) == _kp_dims(_HEAD)

    # Quantity groups must map to the ACTUAL pos/rot6d/gripper slots of every keypoint,
    # derived from the dataset slices (not just structurally disjoint).
    exp_pos, exp_rot, exp_grip = set(), set(), set()
    for pos, rot, grip in (_LEFT, _RIGHT, _HEAD):
        exp_pos |= set(range(pos.start, pos.stop))
        exp_rot |= set(range(rot.start, rot.stop))
        if grip is not None:
            exp_grip.add(grip)
    assert set(_METRIC_GROUP_DIMS["pos"]) == exp_pos
    assert set(_METRIC_GROUP_DIMS["rot"]) == exp_rot
    assert set(_METRIC_GROUP_DIMS["grip"]) == exp_grip

    # Quantity groups must exactly re-partition the union of the keypoint real dims.
    kp_union = set(_METRIC_GROUP_DIMS["left"]) | set(_METRIC_GROUP_DIMS["right"]) | set(_METRIC_GROUP_DIMS["head"])
    qty_union = set(_METRIC_GROUP_DIMS["pos"]) | set(_METRIC_GROUP_DIMS["rot"]) | set(_METRIC_GROUP_DIMS["grip"])
    assert kp_union == qty_union
    # pos/rot/grip are disjoint.
    assert not (set(_METRIC_GROUP_DIMS["pos"]) & set(_METRIC_GROUP_DIMS["rot"]))
    assert not (set(_METRIC_GROUP_DIMS["pos"]) & set(_METRIC_GROUP_DIMS["grip"]))
    assert not (set(_METRIC_GROUP_DIMS["rot"]) & set(_METRIC_GROUP_DIMS["grip"]))
    # pad dims [29:32] are in no group.
    assert not (qty_union & {29, 30, 31})


def test_forward_emits_per_group_metrics():
    """forward() surfaces flow_loss/<group> alongside flow_loss, all finite."""
    torch.manual_seed(0)
    model = _tiny_model()
    out = model(_batch())
    m = out["metrics"]
    for g in ("left", "right", "head", "pos", "rot", "grip"):
        assert f"flow_loss/{g}" in m, f"missing flow_loss/{g}"
        assert np.isfinite(m[f"flow_loss/{g}"])
        assert m[f"flow_loss/{g}"] >= 0.0


def test_per_group_metric_matches_manual_masked_mse():
    """A per-group metric equals the masked-MSE computed by hand over that group's
    dims, with the SAME time-include and action_dim_mask convention as flow_loss."""
    torch.manual_seed(0)
    model = _tiny_model()
    B, T = 2, ACTION_HORIZON
    v_t = torch.randn(B, T, ACTION_DIM)
    u_t = torch.randn(B, T, ACTION_DIM)
    is_pad = torch.zeros(B, T, dtype=torch.bool)
    is_pad[:, T // 2 :] = True
    dim_mask = torch.ones(B, ACTION_DIM, dtype=torch.bool)
    dim_mask[:, 29:] = False

    metrics = model._extra_train_metrics(v_t, u_t, is_pad, None, dim_mask)

    from flore.s1_policy.model.dp_flow import _METRIC_GROUP_DIMS

    sq = (v_t - u_t) ** 2
    include = (~is_pad).unsqueeze(-1).float()  # (B, T, 1)
    for g, dims in _METRIC_GROUP_DIMS.items():
        dims = [d for d in dims if d < ACTION_DIM]
        w = include * dim_mask.unsqueeze(1).float()
        w = w[:, :, dims]
        num = (sq[:, :, dims] * w).sum()
        den = w.sum().clamp(min=1.0)
        expected = (num / den).item()
        assert metrics[f"flow_loss/{g}"] == pytest.approx(expected, abs=1e-6)


def test_per_group_metrics_respect_dim_mask():
    """Perturbing only masked-out (pad) dims must not change any per-group metric —
    pad dims belong to no group and are masked anyway."""
    torch.manual_seed(0)
    model = _tiny_model()
    B, T = 2, ACTION_HORIZON
    v_t = torch.randn(B, T, ACTION_DIM)
    u_t = torch.randn(B, T, ACTION_DIM)
    dim_mask = torch.ones(B, ACTION_DIM, dtype=torch.bool)
    dim_mask[:, 29:] = False

    a = model._extra_train_metrics(v_t, u_t, None, None, dim_mask)
    v_t2 = v_t.clone()
    v_t2[:, :, 29:] += 100.0
    b = model._extra_train_metrics(v_t2, u_t, None, None, dim_mask)
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-6)


def test_absent_keypoint_group_logs_zero():
    """If a whole keypoint is masked out (absent in this sample), its group logs 0.0."""
    torch.manual_seed(0)
    model = _tiny_model()
    B, T = 2, ACTION_HORIZON
    v_t = torch.randn(B, T, ACTION_DIM)
    u_t = torch.randn(B, T, ACTION_DIM)
    dim_mask = torch.ones(B, ACTION_DIM, dtype=torch.bool)
    dim_mask[:, 10:20] = False  # right hand absent
    dim_mask[:, 29:] = False

    m = model._extra_train_metrics(v_t, u_t, None, None, dim_mask)
    assert m["flow_loss/right"] == pytest.approx(0.0, abs=1e-6)
    assert m["flow_loss/left"] > 0.0
