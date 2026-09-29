"""Motion asset configuration and management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .policy import PolicyHandle


@dataclass
class MotionConfig:
    """Configuration for a bundled reference motion asset."""

    name: str
    """Display name shown in the viewer motion selector."""

    source: str | None = None
    """Local path to a bundled ``.npz`` motion file."""

    anchor_body_name: str = ""
    """Reference anchor body name for tracking observations."""

    body_names: tuple[str, ...] = ()
    """Ordered body names used by the tracking task."""

    dataset_joint_names: list[str] | None = None
    """Joint ordering present in the motion dataset."""

    fps: float = 50.0
    """Playback frame rate (Hz). Used as ``sampleHz`` in ``TrackingCommand``."""

    data: bytes | None = None
    """Optional in-memory ``.npz`` payload, used for downloaded W&B artifacts."""

    default: bool = False
    """Whether this motion should be selected by default for the policy."""

    loop: bool = True
    """Whether the motion restarts from the beginning when it reaches the last frame."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for future extensions."""

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize the motion for root ``config.json`` summaries."""
        data: dict[str, Any] = {"name": self.name}
        if self.default:
            data["default"] = True
        return data

    def to_dict(self, path: str) -> dict[str, Any]:
        """Serialize the motion for a policy JSON file."""
        data: dict[str, Any] = {
            "name": self.name,
            "path": path,
            "fps": self.fps,
            "anchor_body_name": self.anchor_body_name,
            "body_names": list(self.body_names),
        }
        if self.dataset_joint_names:
            data["dataset_joint_names"] = list(self.dataset_joint_names)
        if self.default:
            data["default"] = True
        if not self.loop:
            data["loop"] = False
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


class MotionHandle:
    """Handle for configuring a motion after it has been added to a policy."""

    def __init__(
        self, motion_config: MotionConfig, policy: PolicyHandle | None
    ) -> None:
        self._config = motion_config
        self._policy = policy

    @property
    def name(self) -> str:
        """Display name for the motion."""
        return self._config.name

    def set_metadata(self, key: str, value: Any) -> MotionHandle:
        """Set metadata for this motion."""
        self._config.metadata[key] = value
        return self


__all__ = ["MotionConfig", "MotionHandle"]
