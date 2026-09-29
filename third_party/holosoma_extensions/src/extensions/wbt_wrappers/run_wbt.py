#!/usr/bin/env python3
"""Launcher that sets up PYTHONPATH and delegates to wbt_wrappers_inference.run_policy.

Works inside Docker without manual PYTHONPATH/cd:
    python3 /workspace/holosoma-extension/src/extensions/wbt_wrappers/run_wbt.py \
        inference:g1-wbt \
        --task.model-path /workspace/holosoma-extension/wbt_models/model_29999.onnx \
        --task.ref-motion-path /workspace/holosoma-extension/wbt_models/walk_turn_left.npz
"""

import os
import sys

# Add the wbt_wrappers directory to sys.path so wbt_wrappers_inference is importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from wbt_wrappers_inference.run_policy import main  # noqa: E402

if __name__ == "__main__":
    main()
