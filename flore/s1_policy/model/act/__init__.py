"""ACT (Action Chunking Transformer) building blocks, ported from ``third_party/act``.

Kept in its own subpackage rather than beside the DiT/UNet decoders because these are a
FOREIGN architecture with its own provenance: a DETR transformer whose positional
embeddings are injected inside every attention layer, and a ResNet trunk with frozen
BatchNorm. Mixing them into ``model/dit/`` would invite someone to "unify" the two
transformer implementations, which would break weight compatibility with the reference.

The assembled model lives in :mod:`flore.s1_policy.model.dp_act`; these are its parts.
Imports are lazy so a flow-only or diffusion-only process never constructs torchvision
backbones or pays for the import.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ACTTransformer",
    "FrozenBatchNorm2d",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PositionEmbeddingSine2D",
    "RESNET_SPECS",
    "ResNetBackbone",
    "TemporalEnsembler",
    "TransformerEncoder",
    "TransformerEncoderLayer",
]


def __getattr__(name: str) -> Any:
    if name in {"FrozenBatchNorm2d", "IMAGENET_MEAN", "IMAGENET_STD",
                "PositionEmbeddingSine2D", "RESNET_SPECS", "ResNetBackbone"}:
        from flore.s1_policy.model.act import detr_backbone

        value = getattr(detr_backbone, name)
    elif name in {"ACTTransformer", "TransformerEncoder", "TransformerEncoderLayer"}:
        from flore.s1_policy.model.act import detr_transformer

        value = getattr(detr_transformer, name)
    elif name == "TemporalEnsembler":
        from flore.s1_policy.model.act.temporal_ensemble import TemporalEnsembler

        value = TemporalEnsembler
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
