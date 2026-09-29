"""Realtime browser-based eval for the holosoma_2pt_wbt preset.

This is a self-contained mjswan static-bundle builder. It does NOT modify the
existing run_policy / batch_headless_eval pipeline.
"""

from wbt_realtime_eval.main import main, setup_builder

__all__ = ["main", "setup_builder"]
