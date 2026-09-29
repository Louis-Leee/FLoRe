"""Adapter for converting mjlab types to mjswan internal representations.

mjlab is a **soft dependency** — this module never fails at import time.
When mjlab is not installed the ``adapt_*`` functions simply return their
inputs unchanged (they are assumed to already be mjswan types).

The adapter detects mjlab types by checking the module path of the class
(``type(obj).__module__``) rather than ``isinstance``, so mjlab does not
need to be importable for mjswan to function.

Mapping strategy
----------------
Because mjswan sentinels and classes **share the same names** as their
mjlab counterparts, the adapter resolves mappings dynamically via
``getattr`` on the mjswan modules — no hardcoded registries required.

* **Observation / termination functions**: ``func.__name__`` is looked
  up directly on ``mjswan.envs.mdp.observations`` /
  ``mjswan.envs.mdp.terminations``.
* **Action configs**: ``type(cfg).__name__`` is looked up on
  ``mjswan.envs.mdp.actions``, and dataclass fields are copied
  automatically.
"""

from __future__ import annotations

import dataclasses
import re
import warnings
from collections.abc import Mapping
from typing import Any

from ..command import CommandTermConfig as MjswanCommandTermConfig
from ..command import _custom_registry as _custom_command_registry
from ..envs.mdp import actions as _actions_module
from ..envs.mdp import events as _events_module
from ..envs.mdp import observations as _obs_module
from ..envs.mdp import terminations as _term_module
from ..envs.mdp.actions.actions import (
    ActionTermCfg as MjswanActionTermCfg,
)
from ..envs.mdp.events import EventFunc
from ..envs.mdp.events import _custom_registry as _custom_event_registry
from ..envs.mdp.observations import ObsFunc, _custom_registry
from ..envs.mdp.terminations import TermFunc
from ..envs.mdp.terminations import _custom_registry as _custom_term_registry
from ..managers.event_manager import EventTermCfg as MjswanEventTermCfg
from ..managers.observation_manager import (
    ObservationGroupCfg as MjswanObservationGroupCfg,
)
from ..managers.observation_manager import (
    ObservationTermCfg as MjswanObservationTermCfg,
)
from ..managers.termination_manager import (
    TerminationTermCfg as MjswanTerminationTermCfg,
)


def _is_from_mjlab(obj: Any) -> bool:
    """Check whether *obj*'s class originates from the ``mjlab`` package."""
    module = getattr(type(obj), "__module__", "") or ""
    return module.startswith("mjlab")


# ---------------------------------------------------------------------------
# Observation adaptation
# ---------------------------------------------------------------------------


def _adapt_obs_func(func: Any, term_name: str | None = None) -> ObsFunc:
    """Convert an mjlab observation callable to an mjswan ``ObsFunc`` sentinel.

    If *func* is already an mjswan ``ObsFunc`` it is returned as-is, so
    mjswan sentinels can be passed directly inside mjlab ``ObservationTermCfg``
    for functions that have no mjlab equivalent.

    Otherwise, looks up ``func.__name__`` directly on
    ``mjswan.envs.mdp.observations``.  When the function name resolves to an
    unsupported sentinel (e.g. ``builtin_sensor`` used for standard state
    observations in some mjlab tasks), ``term_name`` is tried as a fallback so
    that terms like ``base_lin_vel`` and ``base_ang_vel`` are resolved
    correctly even though their underlying mjlab function is ``builtin_sensor``.
    """
    if isinstance(func, ObsFunc):
        return func
    name = getattr(func, "__name__", None)
    sentinel = getattr(_obs_module, name, None) if name else None
    if (
        name
        and name in _custom_registry
        and (
            not isinstance(sentinel, ObsFunc) or sentinel.unsupported_reason is not None
        )
    ):
        return _custom_registry[name]
    if isinstance(sentinel, ObsFunc) and sentinel.unsupported_reason is None:
        return sentinel
    # Fall back to term name when the function name is missing or unsupported
    if term_name:
        if term_name in _custom_registry:
            return _custom_registry[term_name]
        fallback = getattr(_obs_module, term_name, None)
        if isinstance(fallback, ObsFunc):
            return fallback
    if isinstance(sentinel, ObsFunc):
        return sentinel
    # Fall back to user-registered custom sentinels
    if name and name in _custom_registry:
        return _custom_registry[name]
    if term_name and term_name in _custom_registry:
        return _custom_registry[term_name]
    raise ValueError(
        f"No mjswan mapping for mjlab observation function '{name}'. "
        f"Ensure a matching ObsFunc sentinel exists in "
        f"mjswan.envs.mdp.observations, or register one with "
        f"mjswan.envs.mdp.observations.register_obs_func()."
    )


def _sanitize_obs_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strip mjlab-specific params that are not JSON-serializable.

    ``asset_cfg`` (a ``SceneEntityCfg``) is removed.  When it carries
    entity-scoping information, it is promoted into JSON-friendly fields so
    browser-side observation classes can resolve the correct MuJoCo entities
    at runtime.
    """
    if "asset_cfg" not in params:
        return params
    result = {k: v for k, v in params.items() if k != "asset_cfg"}
    asset_cfg = params["asset_cfg"]
    if _is_from_mjlab(asset_cfg):
        entity_name = getattr(asset_cfg, "name", None)
        if entity_name:
            result["entity_name"] = entity_name
        joint_names = getattr(asset_cfg, "joint_names", None)
        if joint_names:
            names = (
                list(joint_names)
                if isinstance(joint_names, (list, tuple))
                else [joint_names]
            )
            result["joint_names"] = names
            if len(names) == 1:
                name = names[0]
                result["joint_name"] = f"{entity_name}/{name}" if entity_name else name
        site_names = getattr(asset_cfg, "site_names", None)
        if site_names:
            name = (
                site_names[0] if isinstance(site_names, (list, tuple)) else site_names
            )
            # mjlab namespaces entity sites as "{entity_name}/{site_name}"
            result["site_name"] = f"{entity_name}/{name}" if entity_name else name
    return result


def _adapt_obs_term(
    term: Any, term_name: str | None = None
) -> MjswanObservationTermCfg:
    """Convert a single mjlab ``ObservationTermCfg`` to mjswan."""
    raw_params = dict(getattr(term, "params", None) or {})
    return MjswanObservationTermCfg(
        func=_adapt_obs_func(term.func, term_name=term_name),
        params=_sanitize_obs_params(raw_params),
        scale=getattr(term, "scale", None),
        clip=getattr(term, "clip", None),
        history_length=getattr(term, "history_length", 0) or 0,
    )


def _adapt_obs_group(group: Any) -> MjswanObservationGroupCfg:
    """Convert a single mjlab ``ObservationGroupCfg`` to mjswan."""
    raw_terms = getattr(group, "terms", None) or {}
    terms = {
        name: _adapt_obs_term(cfg, term_name=name) for name, cfg in raw_terms.items()
    }
    return MjswanObservationGroupCfg(
        terms=terms,
        concatenate_terms=getattr(group, "concatenate_terms", True),
        enable_corruption=getattr(group, "enable_corruption", False),
        history_length=getattr(group, "history_length", None),
    )


def adapt_observations(
    observations: dict[str, Any] | None,
) -> dict[str, MjswanObservationGroupCfg] | None:
    """Adapt observation groups, converting mjlab types if detected.

    If the values are already ``mjswan.ObservationGroupCfg`` instances they
    are returned as-is.  mjlab ``ObservationGroupCfg`` instances are
    converted transparently.
    """
    if observations is None:
        return None
    return {
        key: group
        if isinstance(group, MjswanObservationGroupCfg)
        else _adapt_obs_group(group)
        if _is_from_mjlab(group)
        else group
        for key, group in observations.items()
    }


# ---------------------------------------------------------------------------
# Termination adaptation
# ---------------------------------------------------------------------------


def _adapt_term_func(func: Any) -> TermFunc:
    """Convert an mjlab termination callable to an mjswan ``TermFunc`` sentinel.

    If *func* is already an mjswan ``TermFunc`` it is returned as-is, so
    mjswan sentinels can be passed directly inside mjlab ``TerminationTermCfg``
    for functions that have no mjlab equivalent.

    Otherwise, looks up ``func.__name__`` directly on
    ``mjswan.envs.mdp.terminations``.
    """
    if isinstance(func, TermFunc):
        return func
    name = getattr(func, "__name__", None)
    sentinel = getattr(_term_module, name, None) if name else None
    if isinstance(sentinel, TermFunc):
        return sentinel
    if name and name in _custom_term_registry:
        return _custom_term_registry[name]
    raise ValueError(
        f"No mjswan mapping for mjlab termination function '{name}'. "
        f"Ensure a matching TermFunc sentinel exists in "
        f"mjswan.envs.mdp.terminations, or register one with "
        f"mjswan.envs.mdp.terminations.register_termination_func()."
    )


def _sanitize_termination_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strip mjlab-only termination params while keeping useful scope data."""
    if not params:
        return params

    result = {
        k: v for k, v in params.items() if k != "asset_cfg" and not _is_from_mjlab(v)
    }
    asset_cfg = params.get("asset_cfg")
    if not _is_from_mjlab(asset_cfg):
        return result

    entity_name = getattr(asset_cfg, "name", None)
    if entity_name:
        result["entity_name"] = entity_name

    body_names = getattr(asset_cfg, "body_names", None)
    if isinstance(body_names, (list, tuple)):
        result["body_names"] = [str(name) for name in body_names]
    elif isinstance(body_names, str):
        result["body_names"] = [body_names]

    return result


def _adapt_term_cfg(term: Any) -> MjswanTerminationTermCfg:
    """Convert a single mjlab ``TerminationTermCfg`` to mjswan."""
    raw_params = dict(getattr(term, "params", None) or {})
    return MjswanTerminationTermCfg(
        func=_adapt_term_func(term.func),
        params=_sanitize_termination_params(raw_params),
        time_out=getattr(term, "time_out", False),
    )


def adapt_terminations(
    terminations: dict[str, Any] | None,
) -> dict[str, MjswanTerminationTermCfg] | None:
    """Adapt termination configs, converting mjlab types if detected."""
    if terminations is None:
        return None
    return {
        key: term
        if isinstance(term, MjswanTerminationTermCfg)
        else _adapt_term_cfg(term)
        if _is_from_mjlab(term)
        else term
        for key, term in terminations.items()
    }


# ---------------------------------------------------------------------------
# Command adaptation
# ---------------------------------------------------------------------------


def _adapt_command_cfg(term: Any) -> MjswanCommandTermConfig:
    """Convert a single mjlab ``CommandTermCfg`` to mjswan."""

    if isinstance(term, MjswanCommandTermConfig):
        return term

    class_name = type(term).__name__
    spec = _custom_command_registry.get(class_name)
    if spec is None:
        raise ValueError(
            f"No mjswan mapping for mjlab command config '{class_name}'. "
            f"Register one with mjswan.register_command_term()."
        )

    serialized = dict(spec.serializer(term))
    return MjswanCommandTermConfig(term_name=spec.ts_name, params=serialized)


def adapt_commands(
    commands: Mapping[str, Any] | None,
) -> dict[str, MjswanCommandTermConfig] | None:
    """Adapt command configs, converting mjlab types if detected."""

    if commands is None:
        return None

    adapted: dict[str, MjswanCommandTermConfig] = {}
    for key, term in commands.items():
        if isinstance(term, MjswanCommandTermConfig):
            adapted[key] = term
            continue
        if _is_from_mjlab(term):
            try:
                adapted[key] = _adapt_command_cfg(term)
            except ValueError as exc:
                warnings.warn(
                    f"Skipping command term '{key}': {exc}",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
            continue
        adapted[key] = term
    return adapted


# ---------------------------------------------------------------------------
# Action adaptation
# ---------------------------------------------------------------------------


def _adapt_action_cfg(term: Any) -> MjswanActionTermCfg | None:
    """Convert a single mjlab ``ActionTermCfg`` to mjswan.

    Looks up ``type(term).__name__`` on ``mjswan.envs.mdp.actions`` to
    find the corresponding mjswan class, then copies all matching
    dataclass fields automatically.

    Returns ``None`` if no mjswan equivalent exists; the caller is
    responsible for dropping the entry.
    """
    class_name = type(term).__name__
    mjswan_cls = getattr(_actions_module, class_name, None)

    if mjswan_cls is None or not (
        isinstance(mjswan_cls, type) and issubclass(mjswan_cls, MjswanActionTermCfg)
    ):
        warnings.warn(
            f"mjlab action type '{class_name}' has no mjswan equivalent. "
            f"It will be skipped.",
            category=RuntimeWarning,
            stacklevel=3,
        )
        return None

    # Copy all matching dataclass fields from the mjlab instance
    kwargs: dict[str, Any] = {}
    entity_name = getattr(term, "entity_name", None)
    for f in dataclasses.fields(mjswan_cls):
        if f.name == "unsupported_reason":
            continue
        val = getattr(term, f.name, dataclasses.MISSING)
        if val is not dataclasses.MISSING:
            kwargs[f.name] = val

    # mjlab namespaces actuator names as "{entity_name}/{name}"; prefix them
    # so they match the fully-qualified policy_joint_names at runtime.
    if entity_name and "actuator_names" in kwargs:
        raw = kwargs["actuator_names"]
        if isinstance(raw, (list, tuple)):
            kwargs["actuator_names"] = tuple(f"{entity_name}/{n}" for n in raw)

    return mjswan_cls(**kwargs)


def adapt_actions(
    actions: Mapping[str, Any] | None,
) -> Mapping[str, MjswanActionTermCfg] | None:
    """Adapt action configs, converting mjlab types if detected."""
    if actions is None:
        return None
    result: dict[str, MjswanActionTermCfg] = {}
    for key, term in actions.items():
        if isinstance(term, MjswanActionTermCfg):
            result[key] = term
        elif _is_from_mjlab(term):
            adapted = _adapt_action_cfg(term)
            if adapted is not None:
                result[key] = adapted
        else:
            result[key] = term
    return result


def resolve_action_scales(
    actions: Mapping[str, MjswanActionTermCfg] | None,
    joint_names: list[str],
) -> None:
    """Resolve regex-pattern scale/offset dicts in action configs to literal joint names.

    mjlab stores per-joint scale as ``{".*_hip_joint": 0.37, ...}`` using regex
    patterns.  The browser runtime does exact string lookups, so patterns are
    expanded here against *joint_names* (the ordered list of joints the policy
    controls, prefixed with the entity name, e.g. ``"robot/left_hip_joint"``).

    Mutates the ``scale`` and ``offset`` fields of each action term in-place.
    """
    if not actions or not joint_names:
        return

    def _resolve(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        resolved: dict[str, float] = {}
        for pattern, val in value.items():
            try:
                regex = re.compile(pattern)
            except re.error:
                resolved[pattern] = val
                continue
            for joint_name in joint_names:
                bare = joint_name.split("/")[-1] if "/" in joint_name else joint_name
                if regex.fullmatch(bare) or regex.fullmatch(joint_name):
                    resolved[joint_name] = val
        return resolved if resolved else value

    for term in actions.values():
        scale = getattr(term, "scale", None)
        if isinstance(scale, dict):
            setattr(term, "scale", _resolve(scale))
        offset = getattr(term, "offset", None)
        if isinstance(offset, dict):
            setattr(term, "offset", _resolve(offset))


# ---------------------------------------------------------------------------
# Event adaptation
# ---------------------------------------------------------------------------


def _adapt_event_func(func: Any) -> EventFunc:
    """Convert an mjlab event callable to an mjswan ``EventFunc`` sentinel."""
    if isinstance(func, EventFunc):
        return func
    name = getattr(func, "__name__", None)
    sentinel = getattr(_events_module, name, None) if name else None
    if isinstance(sentinel, EventFunc):
        return sentinel
    if name and name in _custom_event_registry:
        return _custom_event_registry[name]
    raise ValueError(
        f"No mjswan mapping for mjlab event function '{name}'. "
        f"Register one with mjswan.envs.mdp.events.register_event_func()."
    )


def _sanitize_event_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strip mjlab-only event params while keeping joint scoping data."""
    if not params:
        return params

    result = {
        k: v for k, v in params.items() if k != "asset_cfg" and not _is_from_mjlab(v)
    }
    asset_cfg = params.get("asset_cfg")
    if not _is_from_mjlab(asset_cfg):
        return result

    entity_name = getattr(asset_cfg, "name", None)
    if entity_name:
        result["entity_name"] = entity_name

    joint_names = getattr(asset_cfg, "joint_names", None)
    if isinstance(joint_names, (list, tuple)):
        result["joint_names"] = [str(name) for name in joint_names]
    elif isinstance(joint_names, str):
        result["joint_names"] = [joint_names]

    joint_ids = getattr(asset_cfg, "joint_ids", None)
    if isinstance(joint_ids, (list, tuple)):
        result["joint_ids"] = [int(idx) for idx in joint_ids]

    return result


def _adapt_event_cfg(term: Any) -> MjswanEventTermCfg | None:
    """Convert a single mjlab ``EventTermCfg`` to mjswan.

    Only ``mode="reset"`` events are meaningful for the browser runtime.
    """
    mode = getattr(term, "mode", "reset")
    if mode != "reset":
        return None
    try:
        func = _adapt_event_func(term.func)
    except ValueError as exc:
        warnings.warn(str(exc), category=RuntimeWarning, stacklevel=3)
        return None
    raw_params = dict(getattr(term, "params", None) or {})
    params = _sanitize_event_params(raw_params)
    return MjswanEventTermCfg(func=func, mode=mode, params=params)


def adapt_events(
    events: Mapping[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Adapt event configs, converting mjlab types if detected.

    Returns a list of serialized event dicts ready for JSON output,
    filtering to ``mode="reset"`` only.
    """
    if not events:
        return None
    result: list[dict[str, Any]] = []
    for key, term in events.items():
        if isinstance(term, MjswanEventTermCfg):
            if term.mode == "reset":
                result.append(term.to_dict())
        elif _is_from_mjlab(term):
            adapted = _adapt_event_cfg(term)
            if adapted is not None:
                result.append(adapted.to_dict())
        # non-mjlab, non-mjswan entries are skipped
    return result or None


__all__ = [
    "adapt_events",
    "adapt_observations",
    "adapt_actions",
    "adapt_terminations",
    "resolve_action_scales",
]
