"""Live keyboard control of the deploy F^cmd (commanded contact force).

The mjswan browser sim2sim exposes 12 F^cmd sliders (±x/±y/±z per hand). The
offboard ``run_policy`` deploy previously only took a STATIC F^cmd set once at
launch via ``HOLOSOMA_FORCE_CMD_LOCAL``. This module brings the slider idea to
the real robot: a small keyboard controller nudges the per-hand pelvis-frame
F^cmd live, which feeds BOTH the ``force_cmd_local`` obs term and (for the hybrid
checkpoint) the Jᵀ servo feed-forward — exactly the two places the static env
value fed.

Design (pure ↔ I/O split so the math is headless-testable):

* :func:`apply_force_key` — PURE. Given the current per-hand raw F^cmd ``(2,3)``
  N, the selected hand, and a key, returns the updated ``(raw, sel_hand, changed,
  msg)``. Clamped to ``±clamp_n`` per axis. No global state, no I/O.
* :class:`ForceCmdKeyController` — owns the selected-hand state and subscribes to
  the shared holosoma keyboard listener (the SAME singleton the keyboard command
  provider uses). ``poll(raw)`` drains the queue and folds every pending key
  through :func:`apply_force_key`. Starting the listener is idempotent, so this
  coexists with a keyboard command channel and ALSO works when the primary
  channel is the joystick (``state_input="interface"``) — which otherwise never
  starts the listener.

Keys (single selected hand; lowercased by sshkeyboard so Shift is irrelevant):

    u / j : +x / -x   (forward push / pull-back, pelvis frame)
    h / k : +y / -y   (+y is the robot's LEFT)
    y / n : +z / -z   (up / down)
    l     : toggle the selected hand  (LEFT <-> RIGHT)
    c     : zero the selected hand's F^cmd
    p     : zero BOTH hands' F^cmd

All keys are collision-free vs the holosoma keyboard command + locomotion maps
(``] o i v b f g r = z m x``, digits ``1-9``, ``w s a d q e``, tare ``t``).
"""

from __future__ import annotations

import numpy as np

# key -> (hand_selector, axis, sign). hand_selector None means "the selected hand".
_AXIS_KEYS: dict[str, tuple[int, float]] = {
    "u": (0, +1.0),  # +x
    "j": (0, -1.0),  # -x
    "h": (1, +1.0),  # +y
    "k": (1, -1.0),  # -y
    "y": (2, +1.0),  # +z
    "n": (2, -1.0),  # -z
}
_AXIS_NAMES = ("x", "y", "z")
_HAND_NAMES = ("LEFT", "RIGHT")

# All keys this controller consumes (for gating / help text).
FORCE_CMD_KEYS: tuple[str, ...] = (*_AXIS_KEYS.keys(), "l", "c", "p")

HELP = "LIVE F^cmd keys: u/j=±x  h/k=±y  y/n=±z (selected hand)  |  l=switch hand  c=zero hand  p=zero both"


def apply_force_key(
    raw: np.ndarray, sel_hand: int, key: str, *, step_n: float = 2.0, clamp_n: float = 25.0
) -> tuple[np.ndarray, int, bool, str | None]:
    """Fold one key into the per-hand F^cmd. PURE — returns new state.

    Args:
        raw: ``(2, 3)`` per-hand pelvis-frame F^cmd in Newtons ([L_xyz, R_xyz]).
        sel_hand: currently selected hand (0=left, 1=right).
        key: a lowercase keycode (unknown keys are ignored, ``changed=False``).
        step_n: Newtons added/removed per axis keypress.
        clamp_n: per-axis magnitude clamp (matches the browser slider ±range).

    Returns:
        ``(new_raw, new_sel_hand, changed, msg)``. ``new_raw`` is always a fresh
        array (never mutates the input). ``msg`` is a short log line when the
        state changed, else ``None``.
    """
    out = np.asarray(raw, dtype=np.float32).reshape(2, 3).copy()
    if key in _AXIS_KEYS:
        axis, sign = _AXIS_KEYS[key]
        out[sel_hand, axis] = float(np.clip(out[sel_hand, axis] + sign * step_n, -clamp_n, clamp_n))
        msg = (
            f"F^cmd[{_HAND_NAMES[sel_hand]}] {_AXIS_NAMES[axis]}{'+' if sign > 0 else '-'} → {out[sel_hand].tolist()} N"
        )
        return out, sel_hand, True, msg
    if key == "l":
        sel_hand = 1 - sel_hand
        return out, sel_hand, True, f"F^cmd selected hand → {_HAND_NAMES[sel_hand]}"
    if key == "c":
        out[sel_hand] = 0.0
        return out, sel_hand, True, f"F^cmd[{_HAND_NAMES[sel_hand]}] zeroed"
    if key == "p":
        out[:] = 0.0
        return out, sel_hand, True, "F^cmd zeroed (BOTH hands)"
    return out, sel_hand, False, None


class ForceCmdKeyController:
    """Subscribes to the shared keyboard listener and folds keys into F^cmd.

    ``poll(raw)`` returns ``(new_raw, changed, messages)`` — a fresh ``(2,3)``
    array, whether anything changed this tick, and any human-readable log lines.
    """

    def __init__(self, *, step_n: float = 2.0, clamp_n: float = 25.0) -> None:
        self.step_n = float(step_n)
        self.clamp_n = float(clamp_n)
        self.sel_hand = 0
        self._queue = None
        self._started = False

    def start(self) -> bool:
        """Subscribe to (and idempotently start) the shared keyboard listener.

        Returns True if a live keyboard is available (a TTY). Import is deferred
        so this module stays importable without the holosoma input stack (tests).
        """
        from holosoma_inference.inputs.impl.keyboard import get_keyboard_listener

        listener = get_keyboard_listener()
        self._queue = listener.subscribe()
        self._started = listener.start()
        return self._started

    def poll(self, raw: np.ndarray) -> tuple[np.ndarray, bool, list[str]]:
        """Drain pending keys and fold them into ``raw`` (a ``(2,3)`` N array)."""
        out = np.asarray(raw, dtype=np.float32).reshape(2, 3).copy()
        msgs: list[str] = []
        if self._queue is None:
            return out, False, msgs
        changed_any = False
        while True:
            try:
                key = self._queue.popleft()
            except IndexError:
                break
            out, self.sel_hand, changed, msg = apply_force_key(
                out, self.sel_hand, key, step_n=self.step_n, clamp_n=self.clamp_n
            )
            if changed:
                changed_any = True
                if msg:
                    msgs.append(msg)
        return out, changed_any, msgs
