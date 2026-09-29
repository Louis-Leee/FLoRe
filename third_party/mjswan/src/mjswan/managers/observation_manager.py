"""Observation manager configuration for mjswan.

Provides ``ObservationTermCfg`` and ``ObservationGroupCfg`` with an API
compatible with ``mjlab.managers.observation_manager``.  mjswan only
runs inference in the browser, so training-only fields (noise, delay,
nan_policy, ...) are accepted for API compatibility but silently ignored
at build time.

Example (identical to mjlab)::

    from mjswan.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjswan.envs.mdp import observations as obs_fns

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "base_lin_vel": ObservationTermCfg(func=obs_fns.base_lin_vel),
                "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
                "projected_gravity": ObservationTermCfg(
                    func=obs_fns.projected_gravity
                ),
                "joint_pos": ObservationTermCfg(
                    func=obs_fns.joint_pos_rel, scale=0.5
                ),
                "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                "last_action": ObservationTermCfg(func=obs_fns.last_action),
            },
            enable_corruption=True,
        ),
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..envs.mdp.observations import ObsFunc


@dataclass
class ObservationTermCfg:
    """Configuration for a single observation term.

    Mirrors ``mjlab.managers.observation_manager.ObservationTermCfg``.

    Processing pipeline in mjlab: compute -> noise -> clip -> scale -> delay -> history.
    In mjswan the TS runtime handles scale and history; noise and delay are
    training-only and therefore accepted but ignored.
    """

    func: ObsFunc
    """Observation function sentinel that maps to a TS observation class."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments forwarded to the TS observation constructor."""

    scale: tuple[float, ...] | float | None = None
    """Scaling factor(s) applied element-wise to the observation output."""

    clip: tuple[float, float] | None = None
    """(min, max) clipping range applied after scaling."""

    history_length: int = 0
    """Number of past frames to stack. 0 = current only (no history)."""

    flatten_history_dim: bool = True
    """Whether to flatten history into the feature dimension.
    Accepted for API compatibility; mjswan always flattens."""

    # --- mjlab training-only fields (accepted, ignored at build time) ---

    noise: Any = None
    """Noise config. Accepted for mjlab compatibility; ignored in mjswan."""

    delay_min_lag: int = 0
    delay_max_lag: int = 0
    delay_per_env: bool = True
    delay_hold_prob: float = 0.0
    delay_update_period: int = 0
    delay_per_env_phase: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for the TS ``PolicyRunner``.

        Produces an ``ObservationConfigEntry`` of the form::

            {"name": "BaseLinearVelocity", "scale": 0.5, "history_steps": 3, ...}
        """
        if self.func.unsupported_reason is not None:
            raise NotImplementedError(self.func.unsupported_reason)

        entry: dict[str, Any] = {"name": self.func.ts_name}

        # Merge function defaults with explicit params
        merged: dict[str, Any] = {**self.func.defaults, **self.params}

        if self.scale is not None:
            merged["scale"] = (
                list(self.scale) if isinstance(self.scale, tuple) else self.scale
            )
        if self.clip is not None:
            merged["clip"] = list(self.clip)
        if self.history_length > 0:
            merged["history_steps"] = self.history_length

        entry.update(merged)
        return entry


@dataclass
class ObservationGroupCfg:
    """Configuration for an observation group.

    Mirrors ``mjlab.managers.observation_manager.ObservationGroupCfg``.

    An observation group bundles multiple terms together.  The TS-side
    ``PolicyRunner`` concatenates term outputs in registration order.
    """

    terms: dict[str, ObservationTermCfg] = field(default_factory=dict)
    """Named observation terms, concatenated in registration order."""

    concatenate_terms: bool = True
    """Accepted for mjlab compatibility; mjswan always concatenates."""

    enable_corruption: bool = False
    """Accepted for mjlab compatibility; ignored (no training in browser)."""

    history_length: int | None = None
    """Group-level history override. If set, applies to all terms."""

    flatten_history_dim: bool = True
    """Accepted for mjlab compatibility; mjswan always flattens."""

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize the group's terms to a JSON-compatible list.

        If ``history_length`` is set at the group level, it overrides
        per-term settings (matching mjlab behaviour).
        """
        result = []
        for term_cfg in self.terms.values():
            d = term_cfg.to_dict()
            # Group-level history overrides term-level
            if self.history_length is not None and self.history_length > 0:
                d["history_steps"] = self.history_length
            result.append(d)
        return result


__all__ = ["ObservationTermCfg", "ObservationGroupCfg"]
