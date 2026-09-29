"""Built-in event function sentinels for mjswan.

Each object mirrors a function from ``mjlab.envs.mdp.events``.
These are sentinel objects that carry metadata mapping to the TypeScript
event class used by the browser-side ``EventManager``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EventFunc:
    """Sentinel representing a JS-side event implementation.

    Attributes:
        ts_name: The TypeScript event class name in the ``Events`` registry.
        defaults: Default parameters merged into the JSON config entry.
        ts_src: Absolute path to a ``.ts`` file that exports the class
            named ``ts_name``. When set, the file is injected into the
            browser bundle at build time. Leave ``None`` for built-in classes.
    """

    ts_name: str
    defaults: dict = field(default_factory=dict)
    ts_src: str | None = None


_custom_registry: dict[str, EventFunc] = {}


def register_event_func(mjlab_name: str, sentinel: EventFunc) -> None:
    """Register a custom ``EventFunc`` sentinel for an mjlab event function."""
    _custom_registry[mjlab_name] = sentinel


reset_root_state_uniform = EventFunc("ResetRootStateUniform")
"""Reset root state with uniform random pose sampling.

mjlab: ``mjlab.envs.mdp.events.reset_root_state_uniform``
"""

reset_root_state_from_flat_patches = EventFunc("ResetRootStateFromFlatPatches")
"""Reset root state by placing the robot on a random flat terrain patch.

mjlab: ``mjlab.envs.mdp.events.reset_root_state_from_flat_patches``
"""

__all__ = [
    "EventFunc",
    "register_event_func",
    "reset_root_state_uniform",
    "reset_root_state_from_flat_patches",
]
