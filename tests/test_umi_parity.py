"""UMI training-recipe parity: train-time image augmentation and the reduced encoder LR.

Both knobs were added after diffing FUJI against the pristine upstream clone at
``third_party/universal_manipulation_interface``. They are OFF by default, so the first
thing every test here pins is that a default-configured model is untouched.

CPU-only, stub conv encoder, no downloads — same style as tests/test_diffusion_model.py.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn as nn

from fuji.s1_policy.model.dp_diffusion import DPDiffusionConfig, DPDiffusionModel
from fuji.s1_policy.model.dp_flow import DPFlowConfig, DPFlowModel
from fuji.s1_policy.model.umi_parity import (
    ImageAugment,
    build_image_augment,
    encoder_lr_param_groups,
)

# UMI's shipped strengths (train_diffusion_unet_timm_umi_workspace.yaml).
UMI_JITTER = dict(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)


class StubEncoder(nn.Module):
    """(N, 3, H, W) -> (N, num_features), deterministic, no downloads."""

    def __init__(self, num_features: int = 32):
        super().__init__()
        self.num_features = num_features
        self.conv = nn.Conv2d(3, num_features, kernel_size=16, stride=16)

    def forward(self, x):
        return self.conv(x).mean(dim=(-1, -2))


def _normalized_image(batch: int = 4, size: int = 224) -> torch.Tensor:
    """A batch in the dataset's own convention: ``(x/255 - 0.5)/0.5`` -> ``[-1, 1]``."""
    torch.manual_seed(0)
    return torch.rand(batch, 3, size, size) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# ImageAugment
# ---------------------------------------------------------------------------
def test_augment_is_a_bit_exact_noop_in_eval_mode():
    """The one difference from upstream that matters most: UMI's transforms are nn.Modules
    that ignore ``self.training``, so it augments during validation, during its sampling
    diagnostic, and at deploy. If ours did that, turning augmentation on would move the val
    curve for two reasons at once and no A/B on it could be read."""
    aug = ImageAugment(**UMI_JITTER)
    images = _normalized_image()

    aug.eval()
    assert torch.equal(aug(images), images), "eval mode must return the input unchanged"

    aug.train()
    assert not torch.equal(aug(images), images), "train mode must actually perturb"


def test_augment_preserves_shape_dtype_device_and_range():
    aug = ImageAugment(**UMI_JITTER).train()
    images = _normalized_image()
    out = aug(images)

    assert out.shape == images.shape and out.dtype == images.dtype
    assert out.device == images.device
    # ColorJitter clamps into [0,1]; our inverse affine maps that back into [-1,1]. A value
    # outside means the round trip is wrong (e.g. jitter applied to signed data directly).
    assert torch.isfinite(out).all()
    assert out.min() >= -1.0 - 1e-6 and out.max() <= 1.0 + 1e-6


def test_jitter_operates_on_unit_range_not_on_signed_data():
    """The correctness point of the round trip. ``brightness`` is a MULTIPLY: applied to
    ``[-1, 1]`` data a factor < 1 pulls dark pixels UP toward zero (brighter) while pulling
    bright pixels down — it is not a brightness change at all. Applied after the inverse
    affine it moves every pixel the same direction, which is what upstream means.

    Pinned with brightness only and a fixed factor, so the expected value is exact.
    """
    aug = ImageAugment(crop_ratio=1.0, brightness=0.3, contrast=0.0,
                       saturation=0.0, hue=0.0).train()
    images = _normalized_image(batch=2)
    torch.manual_seed(1234)
    out = aug(images)

    unit_in = images * 0.5 + 0.5
    unit_out = out * 0.5 + 0.5
    # One brightness factor is drawn per forward, so every pixel shares it: the ratio in
    # UNIT space is a single constant (up to the [0,1] clamp).
    unclamped = (unit_out > 1e-6) & (unit_out < 1.0 - 1e-6)
    ratios = (unit_out[unclamped] / unit_in[unclamped])
    assert ratios.std() < 1e-4, "brightness must be one multiplicative factor in [0,1] space"
    factor = float(ratios.mean())
    assert 0.7 - 1e-3 <= factor <= 1.3 + 1e-3, f"factor {factor} outside ColorJitter's 1±0.3"
    assert abs(factor - 1.0) > 1e-4, "the draw was a no-op; the test proves nothing"


def test_crop_ratio_matches_upstreams_derivation():
    """UMI: ``RandomCrop(size=int(image_shape[0] * ratio))`` then resize back
    (TimmObsEncoder.__init__). At 224 and 0.95 that is a 212 crop upsampled to 224."""
    aug = ImageAugment(crop_ratio=0.95, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0)
    crop = aug.geometry[0]
    assert tuple(crop.size) == (212, 212) == (int(224 * 0.95),) * 2
    # Resize must go back to an explicit (H, W); a bare int resizes the SHORT side only.
    assert tuple(aug.geometry[1].size) == (224, 224)
    assert aug.train()(_normalized_image()).shape[-2:] == (224, 224)


def test_augment_holds_no_parameters_or_buffers():
    """So it adds nothing to model.safetensors and nothing for DDP to reduce or broadcast —
    which is also why an older checkpoint may omit the config fields (see
    inference.common._OPTIONAL_MODEL_CONFIG_FIELDS)."""
    aug = ImageAugment(**UMI_JITTER)
    assert list(aug.parameters()) == []
    assert list(aug.buffers()) == []
    assert aug.state_dict() == {}


@pytest.mark.parametrize(("kwargs", "message"), [
    (dict(crop_ratio=0.0), "crop_ratio"),
    (dict(crop_ratio=1.5), "crop_ratio"),
    (dict(image_std=0.0), "image_std"),
    (dict(brightness=-0.1), "brightness"),
    (dict(hue=0.6), "hue"),
    (dict(image_size=1), "image_size"),
])
def test_augment_rejects_nonsense_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ImageAugment(**kwargs)


def test_augment_rejects_a_five_dim_batch_with_a_useful_message():
    """torchvision's Resize fails on 5-D with a 'same number of spatial dimensions' error
    that says nothing about cameras or history frames."""
    aug = ImageAugment(**UMI_JITTER).train()
    with pytest.raises(ValueError, match="flat \\(N, C, H, W\\)"):
        aug(torch.randn(2, 2, 3, 224, 224))


def test_augment_rejects_a_resolution_it_was_not_built_for():
    """The crop size is derived from image_size, so a different input silently crops a
    different FRACTION than configured."""
    aug = ImageAugment(image_size=224, **UMI_JITTER).train()
    with pytest.raises(ValueError, match="built for 224x224"):
        aug(_normalized_image(size=112))


def test_build_image_augment_is_none_unless_asked_and_refuses_an_all_noop_config():
    assert build_image_augment(DPDiffusionConfig()) is None
    assert build_image_augment(DPFlowConfig()) is None

    cfg = DPDiffusionConfig(image_aug=True, image_aug_crop_ratio=1.0, image_aug_brightness=0.0,
                            image_aug_contrast=0.0, image_aug_saturation=0.0, image_aug_hue=0.0)
    with pytest.raises(ValueError, match="every augmentation strength is a no-op"):
        build_image_augment(cfg)


def test_build_image_augment_reads_the_dataset_affine():
    """It must use the config's own affine, not a hardcoded 0.5/0.5 — the whole point of
    TrainConfig pushing dataset_image_mean/std onto the model."""
    aug = build_image_augment(DPDiffusionConfig(
        image_aug=True, dataset_image_mean=0.25, dataset_image_std=0.125))
    assert (aug.image_mean, aug.image_std) == (0.25, 0.125)


# ---------------------------------------------------------------------------
# encoder_lr_param_groups
# ---------------------------------------------------------------------------
def test_force_encoder_is_not_scaled_down():
    """``force_encoder`` is trained FROM SCRATCH, so it belongs with the decoder at the full
    rate. Its name does not start with ``"encoder."`` — a substring match would have caught
    it and silently starved the one module that most needs to learn."""
    model = nn.Module()
    model.encoder = nn.Linear(2, 2)
    model.force_encoder = nn.Linear(2, 2)
    model.decoder = nn.Linear(2, 2)

    groups = encoder_lr_param_groups(model, 1e-4, 0.1)
    assert len(groups) == 2
    full, scaled = groups
    assert full["lr"] == 1e-4 and scaled["lr"] == pytest.approx(1e-5)
    scaled_ids = {id(p) for p in scaled["params"]}
    assert {id(p) for p in model.encoder.parameters()} == scaled_ids
    assert not scaled_ids & {id(p) for p in model.force_encoder.parameters()}
    assert not scaled_ids & {id(p) for p in model.decoder.parameters()}


def test_encoder_lr_scale_refuses_to_be_a_silent_noop():
    """A scale that matches nothing means the run trains at one LR while its config and its
    W&B panel both claim otherwise."""
    model = nn.Module()
    model.decoder = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="no trainable parameter name starts with"):
        encoder_lr_param_groups(model, 1e-4, 0.1)


@pytest.mark.parametrize("scale", [0.0, -0.1])
def test_encoder_lr_scale_rejects_non_positive(scale):
    with pytest.raises(ValueError, match="encoder_lr_scale must be > 0"):
        encoder_lr_param_groups(nn.Linear(2, 2), 1e-4, scale)


# ---------------------------------------------------------------------------
# Model integration
# ---------------------------------------------------------------------------
def _tiny(model_cls, config_cls, **overrides):
    defaults = dict(
        diffusion_net="unet", action_dim=16, action_horizon=16, state_dim=7,
        obs_history_len=1, force_hist_len=1, force_hist_channels=6,
        layout_mode="left_only", pretrained=False,
    )
    defaults.update(overrides)
    return model_cls(config_cls(**defaults), encoder=StubEncoder())


@pytest.mark.parametrize(("model_cls", "config_cls"),
                         [(DPDiffusionModel, DPDiffusionConfig), (DPFlowModel, DPFlowConfig)])
def test_models_default_to_no_augmentation(model_cls, config_cls):
    """A run that does not ask for the UMI recipe must be byte-identical to before."""
    model = _tiny(model_cls, config_cls)
    assert model.image_aug is None
    assert model.config.encoder_lr_scale == 1.0
    # scale 1.0 -> the plain iterator, so AdamW builds ONE group exactly as before.
    assert len(torch.optim.AdamW(model.param_groups(1e-4), lr=1e-4).param_groups) == 1


@pytest.mark.parametrize(("model_cls", "config_cls"),
                         [(DPDiffusionModel, DPDiffusionConfig), (DPFlowModel, DPFlowConfig)])
def test_augmentation_reaches_the_encoder_only_in_train_mode(model_cls, config_cls):
    """End-to-end through the model's own ``_encode_vision``: the stub encoder is
    deterministic, so any feature difference between train() and eval() on the SAME input
    can only come from the augmentation."""
    model = _tiny(model_cls, config_cls, image_aug=True)
    assert model.image_aug is not None

    images = [_normalized_image(batch=2)]
    masks = [torch.ones(2, dtype=torch.bool)]

    model.eval()
    with torch.no_grad():
        a = model._encode_vision(images, masks)
        b = model._encode_vision(images, masks)
    assert torch.equal(a, b), "eval must be deterministic — no augmentation there"

    model.train()
    with torch.no_grad():
        torch.manual_seed(7)
        c = model._encode_vision(images, masks)
    assert not torch.allclose(a, c), "train mode must see augmented frames"


@pytest.mark.parametrize(("model_cls", "config_cls"),
                         [(DPDiffusionModel, DPDiffusionConfig), (DPFlowModel, DPFlowConfig)])
def test_augmentation_adds_no_checkpoint_keys(model_cls, config_cls):
    """An augmented run's weights must stay loadable by an un-augmented build and vice
    versa — both readers use strict=True."""
    plain = _tiny(model_cls, config_cls)
    augmented = _tiny(model_cls, config_cls, image_aug=True)
    assert set(plain.state_dict()) == set(augmented.state_dict())


# ---------------------------------------------------------------------------
# TrainConfig + checkpoint back-compat
# ---------------------------------------------------------------------------
def _hl_config(**overrides):
    from fuji.s1_policy.training.config import HLDataConfig, TrainConfig

    kwargs = dict(algo="diffusion", pipeline="hl",
                  hl_data=HLDataConfig(norm_stats_path="s.json"))
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


def test_train_config_pushes_the_dataset_image_affine_onto_flow_and_diffusion():
    """ImageAugment inverts this affine before ColorJitter; a drift makes every jitter
    strength mean something other than UMI's value. So it is PUSHED, like every other
    layout-derived field, never set independently."""
    from fuji.s1_policy.training.config import HLDataConfig, TrainConfig

    hl = HLDataConfig(norm_stats_path="s.json", image_mean=0.4, image_std=0.2)
    for algo, field in (("diffusion", "diffusion"), ("flow", "model"), ("act", "act")):
        cfg = TrainConfig(algo=algo, pipeline="hl", hl_data=hl)
        active = getattr(cfg, field)
        assert (active.dataset_image_mean, active.dataset_image_std) == (0.4, 0.2), algo


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf"), True])
def test_train_config_rejects_a_bad_encoder_lr_scale(scale):
    """Nothing downstream would catch it: 0 freezes the ViT silently and a negative ascends
    its gradient, both with a healthy-looking loss curve."""
    with pytest.raises(ValueError, match="encoder_lr_scale"):
        _hl_config(diffusion=DPDiffusionConfig(encoder_lr_scale=scale))


def test_trainer_builds_two_groups_at_scale_0_1(tmp_path, monkeypatch):
    """The seam ACT's lr_backbone already uses: ``Trainer`` calls ``model.param_groups`` and
    ``LambdaLR`` then multiplies each group by its OWN initial_lr, so warmup + cosine
    preserve the ratio for the whole run."""
    model = _tiny(DPDiffusionModel, DPDiffusionConfig, encoder_lr_scale=0.1)
    optimizer = torch.optim.AdamW(model.param_groups(3e-4), lr=3e-4)
    assert len(optimizer.param_groups) == 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 0.5)

    # base_lrs, not the live lr: LambdaLR has already applied the factor once at
    # construction. base_lrs is what the schedule multiplies, so this pins the 10x ratio
    # for the whole run.
    assert scheduler.base_lrs == [3e-4, pytest.approx(3e-5)]
    live = [g["lr"] for g in optimizer.param_groups]
    assert live == [pytest.approx(1.5e-4), pytest.approx(1.5e-5)]
    assert live[0] / live[1] == pytest.approx(10.0), "the encoder/decoder ratio must survive"


def test_older_checkpoints_stay_loadable_without_the_new_fields():
    """``dataclass_config`` refuses a checkpoint missing any field — deliberately, since a
    defaulted DIMENSION means serving weights that disagree with their config. These fields
    are exempt because they are train-time only. Everything else must still be refused."""
    from fuji.s1_policy.inference.common import dataclass_config

    full = dataclasses.asdict(DPDiffusionConfig(action_dim=16, layout_mode="left_only"))
    new_fields = {"image_aug", "image_aug_crop_ratio", "image_aug_brightness",
                  "image_aug_contrast", "image_aug_saturation", "image_aug_hue",
                  "encoder_lr_scale", "dataset_image_mean", "dataset_image_std"}
    assert new_fields <= set(full), "field names drifted from the exemption list"

    old = {k: v for k, v in full.items() if k not in new_fields}
    rebuilt = dataclass_config(DPDiffusionConfig, old)
    assert rebuilt.image_aug is False and rebuilt.encoder_lr_scale == 1.0
    assert rebuilt.action_dim == 16 and rebuilt.layout_mode == "left_only"

    # The strictness that remains.
    with pytest.raises(ValueError, match="missing"):
        dataclass_config(DPDiffusionConfig, {k: v for k, v in full.items() if k != "action_dim"})
    with pytest.raises(ValueError, match="unsupported"):
        dataclass_config(DPDiffusionConfig, {**full, "not_a_field": 1})


def test_act_still_requires_its_image_affine():
    """ACT reads dataset_image_* AT INFERENCE (ImageNet re-normalization before a
    frozen-BatchNorm trunk), so defaulting them would feed the trunk out-of-distribution
    input — the exact bug the re-normalization exists to prevent. The exemption is keyed on
    the absence of ``imagenet_renormalize``, not on a class name."""
    from fuji.s1_policy.inference.common import dataclass_config
    from fuji.s1_policy.model.dp_act import DPACTConfig

    full = dataclasses.asdict(DPACTConfig())
    for field in ("dataset_image_mean", "dataset_image_std"):
        with pytest.raises(ValueError, match=field):
            dataclass_config(DPACTConfig, {k: v for k, v in full.items() if k != field})
