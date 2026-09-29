"""Policy configuration and management.

This module defines the PolicyConfig dataclass and PolicyHandle class for
ONNX policy configuration and command management.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import onnx

from .command import CommandTermConfig, velocity_command
from .motion import MotionConfig, MotionHandle

if TYPE_CHECKING:
    from .envs.mdp.actions.actions import ActionTermCfg
    from .managers.observation_manager import ObservationGroupCfg
    from .managers.termination_manager import TerminationTermCfg
    from .scene import SceneHandle


@dataclass
class PolicyConfig:
    """Configuration for an ONNX policy."""

    name: str
    """Name of the policy."""

    model: onnx.ModelProto
    """ONNX model for the policy."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for the policy."""

    source_path: str | None = None
    """Optional source path for the policy ONNX file."""

    config_path: str | None = None
    """Optional source path for the policy config JSON file."""

    commands: dict[str, CommandTermConfig] = field(default_factory=dict)
    """Command terms keyed by their policy-visible names."""

    observations: dict[str, ObservationGroupCfg] | None = None
    """Observation group configurations (mjlab-compatible).

    Keys are group names (e.g. ``"policy"``, ``"critic"``).  Values are
    ``ObservationGroupCfg`` instances whose terms are serialized into
    ``observations`` in the policy JSON at build time.
    """

    actions: Mapping[str, ActionTermCfg] | None = None
    """Action term configurations (mjlab-compatible).

    Keys are term names (e.g. ``"joint_pos"``).  Values are
    ``ActionTermCfg`` subclass instances serialized into ``actions``
    in the policy JSON at build time.
    """

    terminations: dict[str, TerminationTermCfg] | None = None
    """Termination term configurations (mjlab-compatible).

    Keys are term names (e.g. ``"time_out"``, ``"fallen"``).  Values are
    ``TerminationTermCfg`` instances serialized into ``terminations``
    in the policy JSON at build time.
    """

    policy_joint_names: list[str] | None = None
    """Ordered list of joint names controlled by the policy.

    Required by the browser-side ``PolicyRunner`` to map policy outputs to
    the correct actuators in the MuJoCo model.  When set, serialized as
    ``policy_joint_names`` at the top level of the policy JSON config.
    """

    default_joint_pos: list[float] | None = None
    """Default joint positions corresponding to ``policy_joint_names``.

    Used by the browser runtime when ``use_default_offset=True``: action=0
    commands this pose.  Must be in the same order as ``policy_joint_names``.
    """

    encoder_bias: list[float] | None = None
    """Per-joint encoder bias corresponding to ``policy_joint_names``.

    Used by the browser runtime to mirror mjlab's joint-position action path:
    the final target written to actuators is ``processed_action - encoder_bias``.
    """

    initial_qpos: list[float] | None = None
    """Optional initial qpos samples or defaults for runtime reset logic."""

    initial_qvel: list[float] | None = None
    """Optional initial qvel samples or defaults for runtime reset logic."""

    extras: dict[str, Any] | None = None
    """Optional extra policy config payload serialized verbatim into JSON."""

    motions: list[MotionConfig] = field(default_factory=list)
    """Reference motions available for this policy."""

    default: bool = False
    """Whether this policy should be the initially selected one in the viewer.

    When multiple policies in a scene have ``default=True``, the first one wins.
    """


class PolicyHandle:
    """Handle for configuring a policy and its commands.

    Commands should be passed via the ``commands=`` parameter of
    :meth:`~mjswan.scene.SceneHandle.add_policy`.
    :meth:`add_velocity_command` is provided as a convenience shortcut for the
    common locomotion case.

    Example:
        policy = scene.add_policy(
            policy=onnx.load("locomotion.onnx"),
            name="Locomotion",
            config_path="locomotion.json",
            commands={"velocity": mjswan.velocity_command()},
        )
        # or using the shortcut:
        policy = scene.add_policy(...).add_velocity_command()
    """

    def __init__(self, policy_config: PolicyConfig, scene: SceneHandle) -> None:
        self._config = policy_config
        self._scene = scene

    @property
    def name(self) -> str:
        """Name of the policy."""
        return self._config.name

    @property
    def model(self) -> onnx.ModelProto:
        """ONNX model for the policy."""
        return self._config.model

    def add_velocity_command(
        self,
        lin_vel_x: tuple[float, float] = (-1.0, 1.0),
        lin_vel_y: tuple[float, float] = (-0.5, 0.5),
        ang_vel_z: tuple[float, float] = (-1.0, 1.0),
        default_lin_vel_x: float = 0.5,
        default_lin_vel_y: float = 0.0,
        default_ang_vel_z: float = 0.0,
        name: str = "velocity",
    ) -> PolicyHandle:
        """Add a standard velocity command group.

        This is a convenience method for adding the common velocity command
        pattern used in locomotion policies.

        Args:
            lin_vel_x: Range for forward velocity (min, max)
            lin_vel_y: Range for lateral velocity (min, max)
            ang_vel_z: Range for yaw rate (min, max)
            default_lin_vel_x: Default forward velocity
            default_lin_vel_y: Default lateral velocity
            default_ang_vel_z: Default yaw rate

        Returns:
            Self for method chaining.
        """
        cmd = velocity_command(
            lin_vel_x=lin_vel_x,
            lin_vel_y=lin_vel_y,
            ang_vel_z=ang_vel_z,
            default_lin_vel_x=default_lin_vel_x,
            default_lin_vel_y=default_lin_vel_y,
            default_ang_vel_z=default_ang_vel_z,
        )
        self._config.commands[name] = cmd
        return self

    def set_metadata(self, key: str, value: Any) -> PolicyHandle:
        """Set metadata for this policy.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            Self for method chaining.
        """
        self._config.metadata[key] = value
        return self

    def _append_motion(self, motion: MotionConfig) -> MotionHandle:
        if motion.default:
            for existing in self._config.motions:
                existing.default = False
        self._config.motions.append(motion)
        return MotionHandle(motion, self)

    def add_motion(
        self,
        *,
        name: str,
        source: str,
        fps: float = 50.0,
        anchor_body_name: str,
        body_names: tuple[str, ...] | list[str],
        dataset_joint_names: list[str] | None = None,
        default: bool = False,
        loop: bool = True,
    ) -> MotionHandle:
        """Add a bundled ``.npz`` reference motion to this policy."""
        motion = MotionConfig(
            name=name,
            source=source,
            fps=fps,
            anchor_body_name=anchor_body_name,
            body_names=tuple(body_names),
            dataset_joint_names=(
                list(dataset_joint_names)
                if dataset_joint_names is not None
                else (
                    list(self._config.policy_joint_names)
                    if self._config.policy_joint_names is not None
                    else None
                )
            ),
            default=default,
            loop=loop,
        )
        return self._append_motion(motion)

    def add_synthetic_default_pose_motion(
        self,
        *,
        name: str = "default",
        default_dof: list[float] | tuple[float, ...],
        anchor_body_name: str,
        body_names: tuple[str, ...] | list[str],
        dataset_joint_names: list[str] | None = None,
        root_pos: tuple[float, float, float] = (0.0, 0.0, 0.78),
        root_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        fps: float = 50.0,
        default: bool = True,
        loop: bool = False,
    ) -> MotionHandle:
        """Synthesize a single-frame "static stand" reference motion in memory.

        Returns a :class:`MotionHandle` for an in-memory NPZ that holds a
        single frame at ``default_dof`` with the floating-base anchor at
        ``root_pos`` / ``root_quat_wxyz``. The browser-side
        ``TrackingCommand`` parses this exactly like a fetched ``.npz`` and
        plays it as a one-frame loop, so the live robot is commanded back to
        the default pose with no need for a real motion file.

        Useful when:
          - ``--motion-dir`` is empty or supplied at user discretion: every
            policy bundle should have at least ONE motion to bind to so the
            picker isn't empty;
          - debugging a checkpoint without preparing a reference clip;
          - eval setups where the user wants a baseline "no motion command"
            row alongside their real clips.

        The single-frame design (T=1) is sufficient for the parent
        ``TrackingCommand`` (frames are clamped to the last index so a
        T=1 clip just holds steady at frame 0). Subclasses that bake extra
        per-frame tables (e.g. holosoma's ``WbtTrackingCommand``) should
        zero-fill those keys when missing — the synthetic NPZ deliberately
        skips them to stay schema-minimal.

        Parameters
        ----------
        name :
            Label shown in the viewer's motion picker (default ``"default"``).
        default_dof :
            29-DOF (or however many joints the robot has) sequence of joint
            angles. Order MUST match ``dataset_joint_names`` (or, when omitted,
            the policy's ``policy_joint_names``).
        anchor_body_name :
            Name of the floating-base anchor — typically ``"pelvis"`` for
            humanoids and ``"base_link"`` for quadrupeds. Forwarded verbatim
            to ``MotionConfig.anchor_body_name``.
        body_names :
            Ordered tuple of body names that ``TrackingCommand`` will reference
            when reading ``body_pos_w`` / ``body_quat_w``. The synthetic NPZ
            populates one row per body, all parked at the floating-base pose
            (good enough for ghost rendering; bodies appear pinned together
            but the live robot is what users watch).
        dataset_joint_names :
            Optional override for the dataset joint ordering. Defaults to the
            policy's ``policy_joint_names`` when set.
        root_pos :
            Floating-base position in world coordinates. Default
            ``(0, 0, 0.78)`` is the upright G1 pelvis height; pick a different
            z for shorter robots.
        root_quat_wxyz :
            Floating-base orientation as ``(w, x, y, z)``. Default identity.
        fps :
            Playback rate. Doesn't matter much for a one-frame clip but kept
            for API parity with :meth:`add_motion`.
        default :
            Whether this synthetic clip should be the initially selected one
            (default True — the typical use case for an auto-injected default).
        loop :
            Whether playback wraps. Default False — a one-frame static pose
            doesn't benefit from looping.

        Returns
        -------
        MotionHandle
            Same handle type :meth:`add_motion` returns. Chain ``.set_metadata``
            on it just like any other motion.
        """
        import io

        import numpy as np

        if dataset_joint_names is None and self._config.policy_joint_names is not None:
            dataset_joint_names = list(self._config.policy_joint_names)
        if dataset_joint_names is None:
            raise ValueError(
                "add_synthetic_default_pose_motion requires dataset_joint_names "
                "(or policy_joint_names on the policy)."
            )
        n_dof = len(dataset_joint_names)
        if len(default_dof) != n_dof:
            raise ValueError(
                f"default_dof has {len(default_dof)} entries but dataset_joint_names "
                f"has {n_dof}; lengths must match."
            )
        if len(body_names) == 0:
            raise ValueError("body_names must contain at least one entry")

        dof_arr = np.asarray(default_dof, dtype=np.float32)
        rp = np.asarray(root_pos, dtype=np.float32)
        rq = np.asarray(root_quat_wxyz, dtype=np.float32)

        # Single-frame (T=1) layout matching mjswan's parent TrackingCommand
        # parser at template/src/core/command/TrackingCommand.ts:467-480.
        # The parser treats `joint_pos` as a per-frame DOF-only array (no
        # root prefix) and writes `jointPos[i]` into `mjData.qpos` at
        # `datasetQposAdr[i]` — see the loop in `applyReferenceStateToSim`
        # (line ~590). Adding the floating-base 7-prefix here would shift
        # every DOF by 7 slots and scramble the static pose. Floating-base
        # pose is recovered from `body_pos_w` / `body_quat_w` instead.
        joint_pos = dof_arr.reshape(1, n_dof)
        joint_vel = np.zeros((1, n_dof), dtype=np.float32)

        n_bodies = len(body_names)
        # Park every body at the floating-base pose. Visually all bodies
        # collapse onto the pelvis frame for the ghost — fine because the
        # synthetic default is meant to be "no motion command, hold the live
        # pose"; users typically don't display the ghost on it.
        body_pos_w = np.tile(rp, (1, n_bodies, 1)).reshape(1, n_bodies, 3).astype(np.float32)
        body_quat_w = np.tile(rq, (1, n_bodies, 1)).reshape(1, n_bodies, 4).astype(np.float32)
        body_lin_vel_w = np.zeros((1, n_bodies, 3), dtype=np.float32)
        body_ang_vel_w = np.zeros((1, n_bodies, 3), dtype=np.float32)

        body_names_arr = np.asarray(list(body_names), dtype="S")
        joint_names_arr = np.asarray(list(dataset_joint_names), dtype="S")

        buf = io.BytesIO()
        np.savez(
            buf,
            fps=np.asarray(fps, dtype=np.float32),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            body_names=body_names_arr,
            joint_names=joint_names_arr,
        )
        payload = buf.getvalue()

        motion = MotionConfig(
            name=name,
            data=payload,
            fps=fps,
            anchor_body_name=anchor_body_name,
            body_names=tuple(body_names),
            dataset_joint_names=list(dataset_joint_names),
            default=default,
            loop=loop,
        )
        return self._append_motion(motion)

    def add_motion_from_wandb(
        self,
        *,
        name: str | None = None,
        wandb_run_path: str | None = None,
        run_id: str | None = None,
        entity: str | None = None,
        project: str | None = None,
        fps: float = 50.0,
        anchor_body_name: str,
        body_names: tuple[str, ...] | list[str],
        dataset_joint_names: list[str] | None = None,
        default: bool = False,
        loop: bool = True,
    ) -> MotionHandle:
        """Download a motion artifact from W&B and attach it to this policy."""
        from .wandb_utils import fetch_motion_npz_from_wandb_run, resolve_wandb_run_path

        resolved_run_path = resolve_wandb_run_path(
            wandb_run_path=wandb_run_path,
            run_id=run_id,
            entity=entity,
            project=project,
        )
        motion_name, payload = fetch_motion_npz_from_wandb_run(resolved_run_path)
        motion = MotionConfig(
            name=name or motion_name,
            data=payload,
            fps=fps,
            anchor_body_name=anchor_body_name,
            body_names=tuple(body_names),
            dataset_joint_names=(
                list(dataset_joint_names)
                if dataset_joint_names is not None
                else (
                    list(self._config.policy_joint_names)
                    if self._config.policy_joint_names is not None
                    else None
                )
            ),
            default=default,
            loop=loop,
        )
        if default:
            for existing in self._config.motions:
                existing.default = False
        self._config.motions.append(motion)
        return MotionHandle(motion, self)


__all__ = ["PolicyConfig", "PolicyHandle"]
