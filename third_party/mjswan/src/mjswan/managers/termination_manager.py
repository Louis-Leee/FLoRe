"""Termination manager configuration for mjswan.

Provides ``TerminationTermCfg`` with an API compatible with
``mjlab.managers.termination_manager``.

Example (identical to mjlab)::

    from mjswan.managers.termination_manager import TerminationTermCfg
    from mjswan.envs.mdp import terminations as term_fns

    terminations = {
        "time_out": TerminationTermCfg(
            func=term_fns.time_out, time_out=True,
        ),
        "fallen": TerminationTermCfg(
            func=term_fns.bad_orientation,
            params={"limit_angle": 1.0},
        ),
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..envs.mdp.terminations import TermFunc


@dataclass
class TerminationTermCfg:
    """Configuration for a single termination term.

    Mirrors ``mjlab.managers.termination_manager.TerminationTermCfg``.
    """

    func: TermFunc
    """Termination function sentinel that maps to a TS termination class."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments forwarded to the TS termination constructor."""

    time_out: bool = False
    """Whether this term is a truncation (time-based) rather than a
    terminal failure.  Maps to the ``time_out`` flag in the JSON config."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for the TS ``TerminationManager``.

        Produces an entry of the form::

            {"name": "BadOrientation", "params": {"limit_angle": 1.0}, "time_out": true}
        """
        if self.func.unsupported_reason is not None:
            raise NotImplementedError(self.func.unsupported_reason)

        entry: dict[str, Any] = {"name": self.func.ts_name}

        # Merge function defaults with explicit params
        merged: dict[str, Any] = {**self.func.defaults, **self.params}
        if merged:
            entry["params"] = merged

        if self.time_out:
            entry["time_out"] = True

        return entry


__all__ = ["TerminationTermCfg"]
