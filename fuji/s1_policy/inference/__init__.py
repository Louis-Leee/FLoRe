"""Offline inference for trained HL-v2 S1 policies.

The core package intentionally does not import the optional viser frontend. Import
``fuji.s1_policy.inference.viewer`` only when launching the browser UI.
"""

from fuji.s1_policy.inference.common import RolloutResult, SampleRef

__all__ = [
    "RolloutResult",
    "SampleRef",
]
