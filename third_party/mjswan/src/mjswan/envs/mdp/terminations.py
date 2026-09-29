"""Built-in termination function sentinels for mjswan.

Each object mirrors a function from ``mjlab.envs.mdp.terminations``.
In mjswan these are **not** called at runtime — they are sentinel
objects that carry metadata mapping to the TypeScript termination class
used by the browser-side ``TerminationManager``.

Usage (identical to mjlab)::

    from mjswan.envs.mdp import terminations as term_fns
    from mjswan.managers.termination_manager import TerminationTermCfg

    TerminationTermCfg(func=term_fns.time_out, time_out=True)
    TerminationTermCfg(func=term_fns.bad_orientation, params={"limit_angle": 1.0})
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TermFunc:
    """Sentinel representing a JS-side termination implementation.

    Attributes:
        ts_name: The TypeScript termination class name in the
            ``Terminations`` registry (e.g. ``"TimeOut"``).
        defaults: Default parameters merged into the JSON config entry.
        unsupported_reason: If set, this sentinel is accepted for API
            compatibility but raises ``NotImplementedError`` at build time.
        ts_src: Absolute path to a ``.ts`` file that exports the class
            named ``ts_name``. When set, the file is injected into the
            browser bundle at build time so the custom termination class is
            available to the browser-side ``TerminationManager``. Leave
            ``None`` for built-in classes already present in
            ``terminations.ts``.
    """

    ts_name: str
    defaults: dict = field(default_factory=dict)
    unsupported_reason: str | None = None
    ts_src: str | None = None


# ---------------------------------------------------------------------------
# Custom termination registry
# ---------------------------------------------------------------------------

_custom_registry: dict[str, TermFunc] = {}
"""Maps mjlab termination function names to user-supplied ``TermFunc`` sentinels.

Populated via :func:`register_termination_func`. The mjlab adapter checks this
registry as a fallback after the built-in sentinel lookup fails.
"""


def register_termination_func(mjlab_name: str, sentinel: TermFunc) -> None:
    """Register a custom ``TermFunc`` sentinel for an mjlab termination.

    Call this before :meth:`~mjswan.Builder.build` so the adapter can
    resolve the function and the builder can inject any custom TypeScript
    source into the browser bundle.

    Args:
        mjlab_name: The mjlab termination function name
            (e.g. ``"out_of_terrain_bounds"``).
        sentinel: A :class:`TermFunc` describing the browser-side
            implementation. Set ``unsupported_reason`` to mark the
            termination as unsupported. Set ``ts_src`` to the absolute path
            of a ``.ts`` file that exports the class named by ``ts_name``.
    """
    _custom_registry[mjlab_name] = sentinel


# ---------------------------------------------------------------------------
# Episode timeout
# ---------------------------------------------------------------------------

time_out = TermFunc("TimeOut")
"""Terminate when the episode length exceeds its maximum.

mjlab: ``env.episode_length_buf >= env.max_episode_length``
"""

# ---------------------------------------------------------------------------
# Orientation / height checks
# ---------------------------------------------------------------------------

bad_orientation = TermFunc("BadOrientation")
"""Terminate when the asset's orientation exceeds a limit angle.

Requires ``params={"limit_angle": <radians>}``.

mjlab: ``torch.acos(-projected_gravity[:, 2]).abs() > limit_angle``
"""

root_height_below_minimum = TermFunc("RootHeightBelowMinimum")
"""Terminate when the asset's root height is below a minimum.

Requires ``params={"minimum_height": <meters>}``.

mjlab: ``asset.data.root_link_pos_w[:, 2] < minimum_height``
"""

# ---------------------------------------------------------------------------
# Safety / diagnostics (not supported in browser)
# ---------------------------------------------------------------------------

illegal_contact = TermFunc(
    ts_name="",
    unsupported_reason=(
        "illegal_contact is not supported in mjswan: contact force checks on "
        "specific bodies are not available in the browser runtime. "
        "This termination is a training-time safety check and is not needed "
        "for browser-side policy inference."
    ),
)
"""Terminate when a non-foot body makes illegal contact.

.. note::
    Not supported in mjswan. Accepted for API compatibility so that mjlab
    configs can be imported without modification, but raises
    ``NotImplementedError`` at build time.
"""

nan_detection = TermFunc(
    ts_name="",
    unsupported_reason=(
        "nan_detection is not supported in mjswan: NaN/Inf detection "
        "across the full physics state is not available in the browser runtime. "
        "The browser simulation will simply diverge visually if NaN occurs."
    ),
)
"""Terminate when NaN/Inf values appear in physics state.

.. note::
    Not supported in mjswan. Accepted for API compatibility so that mjlab
    configs can be imported without modification, but raises
    ``NotImplementedError`` at build time.
"""


__all__ = [
    "TermFunc",
    "register_termination_func",
    "time_out",
    "bad_orientation",
    "root_height_below_minimum",
    "illegal_contact",
    "nan_detection",
]
