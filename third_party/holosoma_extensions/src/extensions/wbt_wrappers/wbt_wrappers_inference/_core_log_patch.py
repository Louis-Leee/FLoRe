"""Import-time monkey-patches for core ``WholeBodyTrackingPolicy``.

Two behaviors in ``holosoma_inference.policies.wbt.WholeBodyTrackingPolicy``
we override without editing pinned core:

1. **Stiff-hold stdin prompt.** ``__init__`` blocks on ``input()`` at
   startup when stdin is a TTY. ``pi eval`` / ``pi run wbt-teleop``
   export ``HOLOSOMA_SKIP_STIFF_PROMPT=1`` but core never reads it —
   the gate is an ``isatty()`` check only.
2. **Per-tick ``_set_motion_timestep`` INFO spam.** Core logs
   ``Motion timestep: N → N+1`` every tick. For a 254-frame ``pi eval``
   that's 254 INFO lines flooding the console.

The patch lives in ``wbt_wrappers_inference`` (NOT ``holosoma_service``)
so it installs on every WBT path — ``pi eval`` stage 3
(``run_holosoma_reference_headless``) imports
``wbt_wrappers_inference.policies.holosoma_wbt`` directly, without
touching ``holosoma_service``. Putting the patch in the service layer
misses that path.

Both patches are idempotent and run at import time from
``wbt_wrappers_inference/__init__.py``. Env knobs:

* ``HOLOSOMA_SKIP_STIFF_PROMPT=1`` — skip the stdin gate (default off).
* ``HOLOSOMA_LOG_MOTION_TIMESTEP=1`` — restore per-tick INFO log
  (default off — suppressed by this patch).

Unrecognized values fall back to the default; matches the unified
``holosoma_driver._env.env_bool`` semantics without taking the dep
(this module must stay importable from ``pi eval``'s hermetic
runfiles where ``holosoma_driver`` isn't in the graph).
"""

from __future__ import annotations

import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off", ""})


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    return default


def _install_stiff_prompt_patch() -> None:
    """Make the core ``input()`` call a no-op under the env gate.

    Core uses the unqualified builtin ``input()`` inside
    ``WholeBodyTrackingPolicy.__init__``. We inject a no-op ``input``
    into the wbt module's globals; Python name resolution prefers the
    module-local name over the builtin, so all calls inside wbt.py
    resolve to our shim. No other modules are affected.
    """
    if not _env_bool("HOLOSOMA_SKIP_STIFF_PROMPT", default=False):
        return

    from holosoma_inference.policies import wbt as _wbt_mod

    if _wbt_mod.__dict__.get("_wbt_wrappers_input_patched"):
        return

    def _noop_input(*_a, **_kw) -> str:
        return ""

    _wbt_mod.__dict__["input"] = _noop_input
    _wbt_mod.__dict__["_wbt_wrappers_input_patched"] = True


def _install_motion_timestep_log_patch() -> None:
    """Downgrade the per-tick ``_set_motion_timestep`` log to DEBUG.

    We reimplement the method body verbatim except the per-tick INFO
    becomes DEBUG. The "Reached end timestep" one-shot stays at INFO.
    """
    if _env_bool("HOLOSOMA_LOG_MOTION_TIMESTEP", default=False):
        return

    from termcolor import colored

    from holosoma_inference.policies import wbt as _wbt_mod

    cls = _wbt_mod.WholeBodyTrackingPolicy
    original = cls._set_motion_timestep
    if getattr(original, "_wbt_wrappers_patched", False):
        return

    def _patched_set_motion_timestep(self):
        if not self.motion_clip_progressing:
            return

        if self.use_sim_time:
            self.curr_motion_timestep = self.timestep_util.get_timestep(log=self.logger)
        else:
            self.curr_motion_timestep += 1

        if (end := self.config.task.motion_end_timestep) and self.curr_motion_timestep >= end:
            self.logger.info(colored(f"Reached end timestep {end}, stopping motion clip", "yellow"))
            self.motion_clip_progressing = False
            self.curr_motion_timestep = end

    _patched_set_motion_timestep._wbt_wrappers_patched = True  # type: ignore[attr-defined]
    cls._set_motion_timestep = _patched_set_motion_timestep


def install() -> None:
    """Apply both patches. Idempotent."""
    _install_stiff_prompt_patch()
    _install_motion_timestep_log_patch()


install()

__all__ = ["install"]
