"""Action term configurations for mjswan.

Mirrors ``mjlab.envs.mdp.actions.actions`` class hierarchy.  In mjswan
these configuration objects are **not** built into runtime ``ActionTerm``
instances — instead they serialize to JSON config entries consumed by the
browser-side ``runtime.ts``.

Usage (identical to mjlab)::

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

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass(kw_only=True)
class ActionTermCfg(abc.ABC):
    """Base configuration for an action term.

    Mirrors ``mjlab.managers.action_manager.ActionTermCfg``.
    """

    entity_name: str = "robot"
    """Name of the entity in the scene.  Accepted for mjlab compatibility;
    mjswan targets the single policy entity."""

    clip: dict[str, tuple] | None = None
    """Optional per-actuator clipping bounds after scale/offset.
    Accepted for mjlab compatibility; not yet applied in TS runtime."""

    unsupported_reason: str | None = None
    """If set, raises ``NotImplementedError`` at build time."""

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for the TS runtime."""
        raise NotImplementedError


@dataclass(kw_only=True)
class BaseActionCfg(ActionTermCfg):
    """Configuration for actions that control actuator transmissions.

    Mirrors ``mjlab.envs.mdp.actions.actions.BaseActionCfg``.
    """

    actuator_names: tuple[str, ...] | list[str] = (".*",)
    """Actuator names (regex patterns) to control."""

    scale: float | list[float] | dict[str, float] = 1.0
    """Action scale applied to raw policy output."""

    offset: float | list[float] | dict[str, float] = 0.0
    """Action offset added after scaling."""

    preserve_order: bool = False
    """Accepted for mjlab compatibility; ignored in mjswan."""

    def to_dict(self) -> dict[str, Any]:
        if self.unsupported_reason is not None:
            raise NotImplementedError(self.unsupported_reason)

        entry: dict[str, Any] = {}
        if self.scale != 1.0:
            entry["scale"] = self.scale
        if self.offset != 0.0:
            entry["offset"] = self.offset
        entry["actuator_names"] = list(self.actuator_names)
        return entry


@dataclass(kw_only=True)
class JointPositionActionCfg(BaseActionCfg):
    """Configuration for joint position control.

    Mirrors ``mjlab.envs.mdp.actions.actions.JointPositionActionCfg``.

    ``stiffness`` and ``damping`` are mjswan-specific fields for PD control
    in the browser runtime.  In mjlab these are actuator model properties,
    but mjswan needs them in the policy config because the TS runtime
    computes PD externally for motor actuators (biastype=none).
    """

    use_default_offset: bool = True
    """When True, action=0 commands the default joint pose."""

    stiffness: float | list[float] | dict[str, float] | None = None
    """Position gain (kp) for PD control.  Scalar, per-joint list, or dict
    mapping joint names (must match ``policy_joint_names``) to values.
    Only used by the TS runtime for motor actuators with external PD."""

    damping: float | list[float] | dict[str, float] | None = None
    """Velocity gain (kd) for PD control.  Scalar, per-joint list, or dict
    mapping joint names (must match ``policy_joint_names``) to values.
    Only used by the TS runtime for motor actuators with external PD."""

    def to_dict(self) -> dict[str, Any]:
        if self.unsupported_reason is not None:
            raise NotImplementedError(self.unsupported_reason)

        entry: dict[str, Any] = {"type": "joint_position"}
        if self.scale != 1.0:
            entry["scale"] = self.scale
        if self.offset != 0.0:
            entry["offset"] = self.offset
        entry["actuator_names"] = list(self.actuator_names)
        entry["use_default_offset"] = self.use_default_offset
        if self.stiffness is not None:
            entry["stiffness"] = self.stiffness
        if self.damping is not None:
            entry["damping"] = self.damping
        return entry


@dataclass(kw_only=True)
class JointVelocityActionCfg(BaseActionCfg):
    """Configuration for joint velocity control.

    Mirrors ``mjlab.envs.mdp.actions.actions.JointVelocityActionCfg``.

    .. note::
        Not supported in mjswan. Accepted for API compatibility.
    """

    use_default_offset: bool = True

    unsupported_reason: str | None = field(
        default=(
            "JointVelocityAction is not supported in mjswan: the browser "
            "runtime only supports joint_position and torque control types."
        )
    )

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError(self.unsupported_reason)


@dataclass(kw_only=True)
class JointEffortActionCfg(BaseActionCfg):
    """Configuration for joint effort (torque) control.

    Mirrors ``mjlab.envs.mdp.actions.actions.JointEffortActionCfg``.
    """

    stiffness: float | list[float] | dict[str, float] | None = None
    """Position gain (kp).  mjswan-specific; see ``JointPositionActionCfg``."""

    damping: float | list[float] | dict[str, float] | None = None
    """Velocity gain (kd).  mjswan-specific; see ``JointPositionActionCfg``."""

    def to_dict(self) -> dict[str, Any]:
        if self.unsupported_reason is not None:
            raise NotImplementedError(self.unsupported_reason)

        entry: dict[str, Any] = {"type": "torque"}
        if self.scale != 1.0:
            entry["scale"] = self.scale
        if self.offset != 0.0:
            entry["offset"] = self.offset
        entry["actuator_names"] = list(self.actuator_names)
        if self.stiffness is not None:
            entry["stiffness"] = self.stiffness
        if self.damping is not None:
            entry["damping"] = self.damping
        return entry


# ---------------------------------------------------------------------------
# Tendon actions (stubs — not supported in browser runtime)
# ---------------------------------------------------------------------------

_TENDON_UNSUPPORTED = (
    "Tendon actions are not supported in mjswan: the browser runtime does "
    "not expose tendon control APIs."
)


@dataclass(kw_only=True)
class TendonLengthActionCfg(BaseActionCfg):
    """Stub for mjlab compatibility. Not supported in mjswan."""

    unsupported_reason: str | None = field(default=_TENDON_UNSUPPORTED)

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError(self.unsupported_reason)


@dataclass(kw_only=True)
class TendonVelocityActionCfg(BaseActionCfg):
    """Stub for mjlab compatibility. Not supported in mjswan."""

    unsupported_reason: str | None = field(default=_TENDON_UNSUPPORTED)

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError(self.unsupported_reason)


@dataclass(kw_only=True)
class TendonEffortActionCfg(BaseActionCfg):
    """Stub for mjlab compatibility. Not supported in mjswan."""

    unsupported_reason: str | None = field(default=_TENDON_UNSUPPORTED)

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError(self.unsupported_reason)


# ---------------------------------------------------------------------------
# Site actions (stub — not supported in browser runtime)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class SiteEffortActionCfg(BaseActionCfg):
    """Stub for mjlab compatibility. Not supported in mjswan."""

    unsupported_reason: str | None = field(
        default=(
            "SiteEffortAction is not supported in mjswan: the browser runtime "
            "does not expose site force/torque APIs."
        )
    )

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError(self.unsupported_reason)


__all__ = [
    "ActionTermCfg",
    "BaseActionCfg",
    "JointPositionActionCfg",
    "JointVelocityActionCfg",
    "JointEffortActionCfg",
    "TendonLengthActionCfg",
    "TendonVelocityActionCfg",
    "TendonEffortActionCfg",
    "SiteEffortActionCfg",
]
