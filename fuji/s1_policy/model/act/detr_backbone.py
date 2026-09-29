"""ResNet feature backbone + 2D sine position embedding for the ACT preset.

PORTED, not imported, from ``third_party/act/detr/models/{backbone,position_encoding}.py``
(itself DETR, Copyright (c) Facebook, Inc. — Apache 2.0). The math is kept
bit-for-bit; what changed is only plumbing that does not affect the numerics:

* ``NestedTensor``/``Joiner``/``is_main_process`` dropped. Upstream carries the DETR
  detection-mask plumbing whose mask path is already commented out (backbone.py:77-83),
  so ``Joiner`` reduced to "call backbone, call position embedding" — one line here.
* ``pretrained=is_main_process()`` -> an explicit ``pretrained`` flag. Upstream ties
  weight download to distributed rank; FUJI's loaders must be able to force
  ``pretrained=False`` so loading a local checkpoint never hits the network
  (:func:`fuji.s1_policy.inference.common.dataclass_config` sets exactly that).
* ``torchvision.models.resnet18(pretrained=...)`` -> the ``weights=`` enum. The bool
  kwarg was removed in torchvision 0.15; this repo runs 0.28.
* ``return_interm_layers`` (DETR's segmentation head) dropped — ACT never sets
  ``--masks``, so only ``layer4`` was ever returned.

:class:`FrozenBatchNorm2d` is kept EXACTLY as upstream, including the ``eps`` added
before the rsqrt (upstream's docstring records that non-resnet backbones produce NaNs
without it). It matters here for a second reason: ACT trains with batch sizes where
BatchNorm running stats would be noisy, and freezing them is what makes a 106-episode
run behave like the reference.
"""

from __future__ import annotations

import math

import torch
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter

#: Backbone short name -> (torchvision ctor name, weights enum name, channel width).
#: ACT's default and only tested backbone is resnet18; resnet34/50 are offered because
#: the upstream ``num_channels`` branch already covered them and a 106-episode dataset
#: may want more capacity. The channel width is DECLARED here rather than inferred, so a
#: wrong entry fails at construction instead of mis-sizing ``input_proj`` silently.
RESNET_SPECS: dict[str, tuple[str, str, int]] = {
    "resnet18": ("resnet18", "ResNet18_Weights", 512),
    "resnet34": ("resnet34", "ResNet34_Weights", 512),
    "resnet50": ("resnet50", "ResNet50_Weights", 2048),
}


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with fixed batch statistics AND fixed affine parameters.

    Copy-paste from ``torchvision.ops.misc`` with an ``eps`` added before the rsqrt,
    exactly as DETR/ACT carry it (without the eps, non-resnet backbones produce NaNs).
    """

    def __init__(self, n: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: Tensor) -> Tensor:
        # Reshapes moved to the front to stay fuser-friendly (upstream comment).
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class PositionEmbeddingSine2D(nn.Module):
    """2D sine/cosine position embedding over a conv feature map.

    Ported from ``position_encoding.py::PositionEmbeddingSine`` with ACT's own
    modification retained: it derives ``not_mask`` from the tensor itself
    (``torch.ones_like(x[0, [0]])``) instead of an input padding mask, because ACT
    dropped DETR's ``NestedTensor``. ``normalize=True`` matches ``build_position_encoding``
    for the ``sine`` choice, which is ACT's default.

    Returns ``(1, 2 * num_pos_feats, H, W)`` — batch-independent by construction, so the
    caller broadcasts/repeats it. ``num_pos_feats`` is ``hidden_dim // 2``, so the two
    halves (y then x) concatenate back to exactly ``hidden_dim``.
    """

    def __init__(self, num_pos_feats: int = 64, temperature: int = 10000,
                 normalize: bool = True, scale: float | None = None) -> None:
        super().__init__()
        self.num_pos_feats = int(num_pos_feats)
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, [0]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


#: ImageNet per-channel statistics, on a ``[0, 1]`` image. Upstream ACT applies exactly
#: these immediately before the ResNet (``policy.py:20-22``).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNetBackbone(nn.Module):
    """ResNet trunk (through ``layer4``) with frozen BatchNorm + a sine position map.

    ``forward`` returns ``(features, pos)``: ``(B, num_channels, H/32, W/32)`` and the
    matching ``(1, hidden_dim, H/32, W/32)`` embedding. At ACT's 224x224 input that is a
    ``7x7`` grid = **49 visual tokens per camera frame**, which is the whole point of the
    ACT preset next to FUJI's existing ViT-CLS encoders: the transformer cross-attends to
    a spatial grid instead of a single pooled vector.

    IMAGENET RE-NORMALIZATION (``renormalize=True``, the default)
    ------------------------------------------------------------
    The trunk is ImageNet-pretrained AND its BatchNorm is FROZEN, which means the
    normalization statistics baked into every BN layer are ImageNet's. Those transfer only
    if the input distribution matches. Upstream ACT guarantees that by applying
    ``Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`` to a ``[0,1]`` image
    right before the backbone (``policy.py:20-22``).

    FUJI's shared :class:`HLDataset` instead emits ``(x/255 - 0.5)/0.5`` — a ``[-1,1]``
    image, which is what the ViT presets want. Feeding that straight to a frozen-BN ResNet
    is measurably out of distribution: the input std is **2.23x too small**, so after
    ``conv1`` + frozen ``bn1`` the activation std is **0.635x** upstream's, and the
    ``layer4`` features land at **cosine similarity 0.64 / relative L2 0.79** versus the
    correctly-normalized path, with 68.7% of activations ReLU-dead instead of 59.1%.

    So this module UNDOES the dataset's affine and re-applies ImageNet's, giving the trunk
    exactly the input upstream gives it. Done here rather than by changing the dataset's
    ``image_mean``/``image_std`` because that pair is SHARED with the flow and diffusion
    presets (and is recorded in the norm-stats/split provenance): forking it per algorithm
    would mean a second data pipeline and non-interchangeable norm stats.

    ``dataset_mean``/``dataset_std`` must match the dataset's own values, and are stored on
    the config so a checkpoint records what it assumed. Set ``renormalize=False`` to feed
    the dataset tensor through untouched (what the pre-fix behavior was).
    """

    def __init__(self, name: str, hidden_dim: int, *, pretrained: bool, dilation: bool = False,
                 renormalize: bool = True, dataset_mean: float = 0.5, dataset_std: float = 0.5) -> None:
        super().__init__()
        try:
            ctor_name, weights_name, num_channels = RESNET_SPECS[name]
        except KeyError:
            raise ValueError(
                f"unknown ACT backbone {name!r}; expected one of {sorted(RESNET_SPECS)}") from None
        weights = None
        if pretrained:
            # torchvision >= 0.15 replaced the `pretrained` bool with a weights enum. The
            # download only happens when pretrained is True; every checkpoint loader in
            # this repo forces it False so a local restore never touches the network.
            # IMAGENET1K_V1 is pinned rather than .DEFAULT on purpose: for resnet50 the
            # default is V2, i.e. different weights than upstream ACT trained against.
            weights = getattr(torchvision.models, weights_name).IMAGENET1K_V1
        trunk = getattr(torchvision.models, ctor_name)(
            replace_stride_with_dilation=[False, False, dilation],
            weights=weights,
            norm_layer=FrozenBatchNorm2d,
        )
        self.body = IntermediateLayerGetter(trunk, return_layers={"layer4": "0"})
        self.num_channels = int(num_channels)
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be even (it is split y/x), got {hidden_dim}")
        self.position_embedding = PositionEmbeddingSine2D(hidden_dim // 2, normalize=True)

        self.renormalize = bool(renormalize)
        if self.renormalize:
            if float(dataset_std) == 0.0:
                raise ValueError("dataset_std must be non-zero to invert the dataset normalization")
            # Fold "undo (x/255 - m_d)/s_d, then apply (x - m_i)/s_i" into ONE affine:
            #     x_out = (x_in * s_d + m_d - m_i) / s_i
            # Buffers (not parameters): fixed constants, but they ride in the checkpoint so a
            # restored model cannot silently disagree with the dataset it was trained against.
            m_d, s_d = float(dataset_mean), float(dataset_std)
            m_i = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            s_i = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
            self.register_buffer("renorm_scale", s_d / s_i)
            self.register_buffer("renorm_shift", (m_d - m_i) / s_i)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        if self.renormalize:
            image = image * self.renorm_scale.to(image.dtype) + self.renorm_shift.to(image.dtype)
        features = self.body(image)["0"]
        pos = self.position_embedding(features).to(features.dtype)
        return features, pos
