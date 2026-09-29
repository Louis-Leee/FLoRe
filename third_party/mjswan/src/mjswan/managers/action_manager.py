"""Action manager configuration for mjswan.

Provides serialization utilities for action term configurations.
The actual action term classes live in ``mjswan.envs.mdp.actions``.

Example::

    from mjswan.envs.mdp.actions import JointPositionActionCfg

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
            use_default_offset=True,
        ),
    }
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..envs.mdp.actions.actions import ActionTermCfg


def serialize_actions(actions: Mapping[str, ActionTermCfg]) -> dict[str, Any]:
    """Serialize a dict of action term configs to JSON-compatible format.

    Returns a dict keyed by term name, each value being the term's
    ``to_dict()`` output.
    """
    return {name: term_cfg.to_dict() for name, term_cfg in actions.items()}


__all__ = ["serialize_actions"]
