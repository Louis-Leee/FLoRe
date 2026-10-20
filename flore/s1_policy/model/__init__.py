"""S1 VLA models: lazily exported Flow, Diffusion and ACT siblings.

Keeping these imports lazy matters for the equal-level model contract: importing any one
implementation must not initialize the others. For ACT that is not merely tidiness — its
module pulls in ``torchvision.models`` and can construct an ImageNet-pretrained ResNet, so
an eager import would make every flow-only serving process pay for a vision-model import
it never uses. The public ``from flore.s1_policy.model import ...`` API is unchanged.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DPACTConfig",
    "DPACTModel",
    "DPDiffusionConfig",
    "DPDiffusionModel",
    "DPFlowConfig",
    "DPFlowModel",
    "DiffusionVLABase",
    "FlowMatchingVLABase",
    "apply_rtc_time_masking",
]


def __getattr__(name: str) -> Any:
    if name in {"DPACTConfig", "DPACTModel"}:
        from flore.s1_policy.model.dp_act import DPACTConfig, DPACTModel

        value = {"DPACTConfig": DPACTConfig, "DPACTModel": DPACTModel}[name]
    elif name in {"FlowMatchingVLABase", "apply_rtc_time_masking"}:
        from flore.s1_policy.model.base import FlowMatchingVLABase, apply_rtc_time_masking

        value = {
            "FlowMatchingVLABase": FlowMatchingVLABase,
            "apply_rtc_time_masking": apply_rtc_time_masking,
        }[name]
    elif name == "DiffusionVLABase":
        from flore.s1_policy.model.diffusion_base import DiffusionVLABase

        value = DiffusionVLABase
    elif name in {"DPDiffusionConfig", "DPDiffusionModel"}:
        from flore.s1_policy.model.dp_diffusion import DPDiffusionConfig, DPDiffusionModel

        value = {
            "DPDiffusionConfig": DPDiffusionConfig,
            "DPDiffusionModel": DPDiffusionModel,
        }[name]
    elif name in {"DPFlowConfig", "DPFlowModel"}:
        from flore.s1_policy.model.dp_flow import DPFlowConfig, DPFlowModel

        value = {"DPFlowConfig": DPFlowConfig, "DPFlowModel": DPFlowModel}[name]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
