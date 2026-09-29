"""Holosoma training extension for G1 29-DOF whole-body tracking.

Provides experiment presets that reuse holosoma core WBT configs for the G1
robot, plus ONNX export utilities that produce checkpoints compatible with
the ``wbt_wrappers_inference`` TML and BM policy wrappers.
"""

from __future__ import annotations

try:
    from wbt_training.config_values import command, experiment  # noqa: F401
except ImportError:
    # holosoma not available; registration deferred until install.
    pass

# FALCON dual-agent checkpoint-config reload shim: makes PPODecoupledConfig survive
# dict re-validation on the eval/deploy path (see utils/config_reload for the why).
# Idempotent + no-op for every non-FALCON preset; safe to run at import.
try:
    from wbt_training.utils.config_reload import install_falcon_config_reload_patch

    install_falcon_config_reload_patch()
except ImportError:
    pass
