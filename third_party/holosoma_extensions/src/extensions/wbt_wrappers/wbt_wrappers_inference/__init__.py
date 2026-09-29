"""Holosoma Inference Extension - TML and BM Whole-Body Tracking Wrappers.

This package extends holosoma_inference with two additional WBT policy variants:

- TMLWholeBodyTrackingPolicy: Uses external .npz reference motions and Isaac joint ordering.
- BMWholeBodyTrackingPolicy: Uses ONNX-embedded motion references selected via time_step.

The configurations are auto-discovered via entry points when this package is installed.
"""

from __future__ import annotations

# Side-effect import: installs monkey-patches on core's
# WholeBodyTrackingPolicy for the "Press Enter to continue" stdin gate
# (HOLOSOMA_SKIP_STIFF_PROMPT=1) and the per-tick "Motion timestep"
# INFO spam (HOLOSOMA_LOG_MOTION_TIMESTEP=1 to restore). Must live here,
# not in holosoma_service, so `pi eval` stage 3
# (run_holosoma_reference_headless) picks it up — that path imports
# wbt_wrappers_inference but not holosoma_service.
from wbt_wrappers_inference import _core_log_patch as _core_log_patch  # noqa: F401

__version__ = "0.1.0"

__all__ = ["__version__"]
