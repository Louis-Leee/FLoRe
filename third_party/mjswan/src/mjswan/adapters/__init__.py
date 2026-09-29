"""Adapters for converting external framework types to mjswan internals.

Currently supports mjlab as a soft dependency.
"""

from .mjlab_adapter import (
    adapt_actions,
    adapt_commands,
    adapt_observations,
    adapt_terminations,
    resolve_action_scales,
)
from .mjlab_compat import apply_mjlab_sim_options, ensure_mjlab_extensions

__all__ = [
    "adapt_observations",
    "adapt_actions",
    "adapt_commands",
    "adapt_terminations",
    "resolve_action_scales",
    "apply_mjlab_sim_options",
    "ensure_mjlab_extensions",
]
