"""Event manager configuration for mjswan.

Provides ``EventTermCfg`` for scene-level reset events such as spawn
position randomization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..envs.mdp.events import EventFunc


@dataclass
class EventTermCfg:
    """Configuration for a single event term.

    Mirrors ``mjlab.managers.event_manager.EventTermCfg``.
    Only ``mode="reset"`` events are supported in the browser runtime.
    """

    func: EventFunc
    """Event function sentinel that maps to a TS event class."""

    mode: str = "reset"
    """Event trigger mode. Only ``"reset"`` is handled by the browser runtime."""

    params: dict[str, Any] = field(default_factory=dict)
    """Parameters forwarded to the TS event constructor."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for the TS ``EventManager``."""
        entry: dict[str, Any] = {"name": self.func.ts_name}
        merged: dict[str, Any] = {**self.func.defaults, **self.params}
        if merged:
            entry["params"] = merged
        return entry


__all__ = ["EventTermCfg"]
