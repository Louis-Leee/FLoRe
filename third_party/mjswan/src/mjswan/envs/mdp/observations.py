"""Built-in observation function sentinels for mjswan.

Each object mirrors a function from ``mjlab.envs.mdp.observations``.
In mjswan these are **not** called at runtime — they are sentinel
objects that carry metadata mapping to the TypeScript observation class
used by the browser-side ``PolicyRunner``.

Usage (identical to mjlab)::

    from mjswan.envs.mdp import observations as obs_fns
    from mjswan.managers.observation_manager import ObservationTermCfg

    ObservationTermCfg(func=obs_fns.base_lin_vel)
    ObservationTermCfg(func=obs_fns.joint_pos_rel, scale=0.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObsFunc:
    """Sentinel representing a JS-side observation implementation.

    Attributes:
        ts_name: The TypeScript observation class name in the
            ``Observations`` registry (e.g. ``"BaseLinearVelocity"``).
        defaults: Default parameters merged into the JSON config entry.
            These map mjlab semantics to the existing TS class API.
        unsupported_reason: If set, this sentinel is accepted for API
            compatibility but raises ``NotImplementedError`` at build time
            with this message.
        ts_src: Absolute path to a ``.ts`` file that exports the class
            named ``ts_name``.  When set, the file is injected into the
            browser bundle at build time so the custom observation class is
            available to the ``PolicyRunner``.  Leave ``None`` for built-in
            classes already present in ``observations.ts``.
    """

    ts_name: str
    defaults: dict = field(default_factory=dict)
    unsupported_reason: str | None = None
    ts_src: str | None = None


# ---------------------------------------------------------------------------
# Custom observation registry
# ---------------------------------------------------------------------------

_custom_registry: dict[str, ObsFunc] = {}
"""Maps mjlab observation function names to user-supplied ``ObsFunc`` sentinels.

Populated via :func:`register_obs_func`.  The mjlab adapter checks this
registry as a fallback after the built-in sentinel lookup fails.
"""


def register_obs_func(mjlab_name: str, sentinel: ObsFunc) -> None:
    """Register a custom ``ObsFunc`` sentinel for an mjlab observation function.

    Call this before :meth:`~mjswan.Builder.build` so the adapter can
    resolve the function and the builder can inject any custom TypeScript
    source into the browser bundle.

    Args:
        mjlab_name: The mjlab observation function name
            (e.g. ``"ee_to_object_distance"``).
        sentinel: An :class:`ObsFunc` describing the browser-side
            implementation.  Set ``unsupported_reason`` to mark the
            observation as unsupported (silently skipped at build time).
            Set ``ts_src`` to the absolute path of a ``.ts`` file that
            exports the class named by ``ts_name``.

    Example — mark as unsupported::

        register_obs_func(
            "ee_to_object_distance",
            ObsFunc(ts_name="", unsupported_reason="not available in browser"),
        )

    Example — provide a custom TypeScript implementation::

        register_obs_func(
            "my_custom_obs",
            ObsFunc(ts_name="MyCustomObs", ts_src="/path/to/MyCustomObs.ts"),
        )
    """
    _custom_registry[mjlab_name] = sentinel


# ---------------------------------------------------------------------------
# Root state
# ---------------------------------------------------------------------------

base_lin_vel = ObsFunc("BaseLinearVelocity", {"world_frame": False})
"""Linear velocity of the robot base in the base frame.

mjlab: ``asset.data.root_link_lin_vel_b``
"""

base_ang_vel = ObsFunc("BaseAngularVelocity")
"""Angular velocity of the robot base in the base frame.

mjlab: ``asset.data.root_link_ang_vel_b``
"""

projected_gravity = ObsFunc("ProjectedGravityB")
"""Gravity vector projected into the base frame.

mjlab: ``asset.data.projected_gravity_b``
"""

# ---------------------------------------------------------------------------
# Joint state
# ---------------------------------------------------------------------------

joint_pos_rel = ObsFunc(
    "JointPos",
    {
        "subtract_default": True,
        "history_steps": 1,
        "entity_name": "robot",
        "joint_names": "all",
    },
)
"""Joint positions relative to the default pose.

mjlab: ``asset.data.joint_pos - asset.data.default_joint_pos``
"""

joint_vel_rel = ObsFunc(
    "JointVelocities",
    {
        "entity_name": "robot",
        "joint_names": "all",
    },
)
"""Joint velocities relative to the default velocities.

mjlab: ``asset.data.joint_vel - asset.data.default_joint_vel``
"""

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

last_action = ObsFunc("PrevActions", {"history_steps": 1})
"""The most recent action tensor.

mjlab: ``env.action_manager.action``
"""

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

generated_commands = ObsFunc("GeneratedCommands")
"""Current command tensor from a named command term.

Requires ``params={"command_name": "<name>"}``.

mjlab: ``env.command_manager.get_command(command_name)``
"""

# ---------------------------------------------------------------------------
# Modern Isaac-compatible implementations
# ---------------------------------------------------------------------------

projected_gravity_isaac = ObsFunc(
    "ProjectedGravity", {"joint_name": "floating_base_joint"}
)
"""Gravity vector projected into the base frame (Isaac-compatible implementation).

Uses ``floating_base_joint`` frame by default.
Pass ``gravity`` via ``params`` to override the gravity vector.
mjlab: ``asset.data.projected_gravity_b``
"""

joint_positions_isaac = ObsFunc(
    "JointPositions", {"joint_names": "isaac", "subtract_default": True}
)
"""Joint positions with Isaac joint ordering, relative to the default pose.

mjlab: ``asset.data.joint_pos - asset.data.default_joint_pos``
"""

previous_actions = ObsFunc("PreviousActions", {"history_steps": 1})
"""Most recent action tensor (Isaac-compatible implementation).

mjlab: ``env.action_manager.action``
"""

# ---------------------------------------------------------------------------
# Command observations
# ---------------------------------------------------------------------------

velocity_command_with_oscillators = ObsFunc("VelocityCommandWithOscillators")
"""Velocity command augmented with oscillator signals (16 dims)."""

impedance_command = ObsFunc("ImpedanceCommand")
"""Impedance control command as an observation term."""

joint_pos_cos_sin = ObsFunc("JointPosCosSin")
"""Cosine and sine of a single joint angle. Shape: [num_envs, 2].

Pass ``params={"joint_name": "<name>"}`` or ``params={"joint_index": <i>}``
to select the target joint.
"""

# ---------------------------------------------------------------------------
# Motion tracking
# ---------------------------------------------------------------------------

motion_anchor_pos_b = ObsFunc("MotionAnchorPosB")
"""Reference anchor position relative to the robot anchor frame."""

motion_anchor_ori_b = ObsFunc("MotionAnchorOriB")
"""Reference anchor orientation relative to the robot anchor frame."""

robot_body_pos_b = ObsFunc("RobotBodyPosB")
"""Robot body positions expressed in the robot anchor frame."""

robot_body_ori_b = ObsFunc("RobotBodyOriB")
"""Robot body orientations expressed in the robot anchor frame."""

# ---------------------------------------------------------------------------
# Sensors (not supported in browser)
# ---------------------------------------------------------------------------

builtin_sensor = ObsFunc(
    ts_name="BuiltinSensor",
)
"""Raw data from a named BuiltinSensor.
"""

height_scan = ObsFunc(
    ts_name="",
    unsupported_reason=(
        "height_scan is not supported in mjswan: RayCastSensor is not "
        "available in the browser runtime."
    ),
)
"""Height scan from a RayCastSensor.

.. note::
    Not supported in mjswan. Accepted for API compatibility so that mjlab
    configs can be imported without modification, but raises
    ``NotImplementedError`` at build time.
"""


__all__ = [
    "ObsFunc",
    "register_obs_func",
    "_custom_registry",
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "generated_commands",
    "projected_gravity_isaac",
    "joint_positions_isaac",
    "previous_actions",
    "velocity_command_with_oscillators",
    "impedance_command",
    "joint_pos_cos_sin",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "robot_body_pos_b",
    "robot_body_ori_b",
    "builtin_sensor",
    "height_scan",
]
