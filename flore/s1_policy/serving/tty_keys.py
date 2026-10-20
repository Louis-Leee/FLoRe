"""Terminal HOTKEYS for the five-point page: its most-pressed buttons on the keyboard of the terminal
that launched it (T5), so an operator standing at the rig can pause a ▶▶ run, start the WALK skill,
swap the LEGS under a running policy, close or open BOTH jaws and resume without reaching for the
browser.

WHY THE TERMINAL, AND NOT THE JOYSTICK OR A GLOBAL HOOK. The joystick belongs to the LOW LEVEL (T4):
holosoma's provider reads it, and a button reaches this page only if the low level publishes it on
RecordTap — one low-level change per button, on a controller whose spare buttons are not spare (`X`
is RECENTER, a no-op in absolute mode only because the mode refuses it). A global keyboard hook
(evdev, as ``keyboard_gripper_dual_ws.py`` uses) sees every keystroke in every window, and the deploy
stack already has two keyboard readers whose maps collide (``o`` is holosoma's STOP and the gripper's
default OPEN). This reader owns exactly ONE terminal: the TTY on the page's own stdin, in cbreak mode.
holosoma's keyboard provider is ``sshkeyboard`` on ITS process's stdin, so a key typed into T5 never
reaches T4 and vice versa. The bindings are nevertheless chosen OUTSIDE holosoma's map
(:data:`HOLOSOMA_KEYBOARD_KEYS`) and :class:`TtyKeys` REFUSES a binding inside it, so a key typed into
the wrong terminal by mistake is a no-op there rather than a START.

WHAT A KEY DOES IS WHAT THE BUTTON DOES. The page hands this module named callables — the ▶▶/⏹
toggle, the 🚶 toggle, the 🦵 legs toggle, the Manual-jaws close / open for BOTH hands, STOP — that
ARE the button handlers' bodies (the ``on_click`` decorators call the same closures). Nothing here knows about the wire, the clutch or a chunk; every refusal (a chunk
executing, a walk during a ▶▶ run, the low level off :8770) is the page's own, and comes back as the
note the page's text field shows. An action that raises is reported, never re-raised: the reader
thread must outlive one bad press exactly as the render loop outlives one bad tick.

SAFETY OF THE READER ITSELF.
* It refuses (and says why) when stdin is not a TTY, or when the process is not the terminal's
  FOREGROUND job — reading a background job's TTY stops the whole process with SIGTTIN, i.e. the
  page would freeze the moment it was launched with ``&``.
* It restores the terminal attributes on :meth:`stop` AND at interpreter exit: a terminal left in
  cbreak mode swallows the operator's next shell line.
* It parses ESCAPE SEQUENCES by their grammar. An arrow key is ``ESC [ A`` and Alt-p is ``ESC p``;
  delivered byte by byte, ``[``/``A``/``p`` would be looked up as keys. A CSI/SS3 sequence is consumed
  to its final byte and an Alt-chord as two bytes — never "the rest of the read", which lost the key
  typed right after an arrow (measured on a PTY in the test). A sequence split across reads is
  carried for :data:`ESC_CARRY_S`; a lone ESC older than that was the Escape key and is dropped.
* It keeps ISIG (``tty.setcbreak``), so Ctrl-C still interrupts the page.

Bindings (:data:`DEFAULT_BINDINGS`)::

    p      ▶▶ Run to completion  /  ⏹ Stop the run     (one key: starts when idle, retires when running)
    k      🚶 Walk                (one key: starts when idle, ramps the feet back when playing)
    l      🦵 Legs ← clip         (one key: the clip's feet + legs replace the planner's two foot
                                  slots UNDER the running policy; again = hand them back)
    c      ⬛ close BOTH jaws      (the Manual-jaws ⬛ button of each hand, in one press: the press
                                  OVERRIDES the policy's jaw for that hand until Reset, as the
                                  buttons do; a hand with no jaw bridge is refused by name, the
                                  other still moves)
    u      ⬜ open BOTH jaws       (the Manual-jaws ⬜ button of each hand; same override rule)
    SPACE  STOP                   (the operator's interrupt: freezes the wire within one step)
    h  ?   print this table

The ▶▶/⏹ pair is one key because that is how a rig is operated: pause at a chunk boundary, walk,
resume. The retire is ⏹ (the chunk in flight finishes, up to exec-steps × 33 ms) and never STOP —
STOP has its own key, because an interrupt must not be one press away from "pause".
"""

from __future__ import annotations

import atexit
import os
import select
import sys
import threading
import time
from collections.abc import Callable, Mapping

Action = Callable[[], "str | None"]

ESC = "\x1b"
#: How long a read ending in a lone ESC waits for the rest of its sequence before it is taken as the
#: Escape KEY (and dropped). Terminals deliver a sequence within a millisecond; a human cannot follow
#: an Escape press with a key inside 100 ms, so the next key after a real Escape is never swallowed.
ESC_CARRY_S = 0.1

#: ``p`` ▶▶/⏹, ``k`` 🚶, ``l`` 🦵 legs, ``c`` ⬛ close both jaws, ``u`` ⬜ open both jaws, SPACE STOP,
#: ``h``/``?`` help. ``k`` rather than ``w`` because ``w`` is holosoma's forward-velocity key (see
#: :data:`HOLOSOMA_KEYBOARD_KEYS`); ``l`` rather than ``x`` (the user's first suggestion) because
#: ``x`` is holosoma's RECENTER — typed into the low level's terminal by mistake it would re-anchor
#: the teleop frame, exactly the no-op-in-the-wrong-terminal rule this map exists for; ``l`` is only
#: the F^cmd hand selector there. ``c``/``u`` (2026-09-21) rather than ``o`` for "open" because ``o``
#: is BOTH holosoma's STOP and the gripper evdev bridge's default OPEN — the very collision this map
#: exists to stay out of; ``c`` = close, ``u`` = un-grip, neither in holosoma's map.
DEFAULT_BINDINGS: Mapping[str, str] = {"p": "run", "k": "walk", "l": "legs", "c": "jaws_close",
                                       "u": "jaws_open", " ": "stop", "h": "help", "?": "help"}

#: holosoma's own keyboard map (``holosoma_inference/inputs/impl/keyboard.py`` — ``]`` START, ``o``
#: STOP, ``i`` INIT, ``v b f g r`` Kp, ``=`` arm toggle, ``z`` soft-stop, ``m`` clip, ``x`` recenter,
#: ``1-9`` policy switch, ``w s a d q e`` velocity) plus ``holosoma_wbt``'s ``t``/``T`` (F/T tare).
#: A binding on one of these is REFUSED: the two readers never share a terminal, but an operator
#: who types into the wrong one must get a no-op, not a START.
HOLOSOMA_KEYBOARD_KEYS = frozenset("]oivbfgr=zmxwsadqetT123456789")

#: The actions a binding may name. ``help`` is served here; the rest are the page's.
KNOWN_ACTIONS = ("run", "walk", "legs", "jaws_close", "jaws_open", "stop", "help")

_KEY_LABEL = {" ": "SPACE", "\t": "TAB", "\r": "ENTER", "\n": "ENTER"}
_ACTION_LABEL = {
    "run": "▶▶ run / ⏹ pause (one key)",
    "walk": "🚶 walk toggle",
    "legs": "🦵 legs ← clip toggle (under the running policy)",
    "jaws_close": "⬛ close BOTH jaws (manual override until Reset)",
    "jaws_open": "⬜ open BOTH jaws",
    "stop": "STOP",
    "help": "help",
}


def key_label(ch: str) -> str:
    return _KEY_LABEL.get(ch, ch)


def describe_bindings(bindings: Mapping[str, str] = DEFAULT_BINDINGS) -> str:
    """One line, action-major, e.g. ``p = ▶▶ run / ⏹ pause (one key) · k = 🚶 walk toggle · ...``."""
    by_action: dict[str, list[str]] = {}
    for ch, act in bindings.items():
        by_action.setdefault(act, []).append(key_label(ch))
    parts = []
    for act in KNOWN_ACTIONS:
        if act in by_action:
            parts.append(f"{' / '.join(by_action[act])} = {_ACTION_LABEL[act]}")
    return " · ".join(parts)


def validate_bindings(bindings: Mapping[str, str], actions: Mapping[str, Action]) -> None:
    """Refuse a binding on holosoma's keys, on an unknown action, or on an action the page did not hand over."""
    for ch, act in bindings.items():
        if not isinstance(ch, str) or len(ch) != 1:
            raise ValueError(f"hotkey {ch!r}: a binding is exactly one character")
        if ch in HOLOSOMA_KEYBOARD_KEYS:
            raise ValueError(
                f"hotkey {key_label(ch)!r} is in holosoma's own keyboard map — a key typed into the low "
                f"level's terminal by mistake must be a no-op there, not a command"
            )
        if ch == ESC:
            raise ValueError("hotkey ESC cannot be bound: it introduces every escape sequence")
        if act not in KNOWN_ACTIONS:
            raise ValueError(f"hotkey {key_label(ch)!r}: unknown action {act!r} (known: {KNOWN_ACTIONS})")
        if act != "help" and act not in actions:
            raise ValueError(f"hotkey {key_label(ch)!r}: the page handed no {act!r} action")


class TtyKeys:
    """Read single keys off ONE terminal (stdin, cbreak) on a daemon thread and run the page's actions.

    ``actions`` maps an action name to a callable returning a note (or None). ``dispatch``/``feed`` are
    pure (no terminal) and carry the whole key → action → note logic, so they are what the tests pin;
    ``start``/``stop`` own the terminal mode and the thread.
    """

    def __init__(
        self,
        actions: Mapping[str, Action],
        *,
        bindings: Mapping[str, str] = DEFAULT_BINDINGS,
        stdin=None,
        log: Callable[[str], None] | None = None,
        require_foreground: bool = True,
        poll_s: float = 0.2,
    ) -> None:
        validate_bindings(bindings, actions)
        self.actions = dict(actions)
        self.bindings = dict(bindings)
        self._stdin = sys.stdin if stdin is None else stdin
        self._log = log if log is not None else (lambda s: print(s, flush=True))
        self._require_fg = bool(require_foreground)
        self._poll_s = float(poll_s)
        self._fd: int | None = None
        self._saved_attrs = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._atexit_registered = False
        self._esc_carry = ""      # a sequence split across reads (or a lone ESC awaiting its verdict)
        self._esc_carry_t = 0.0
        self.n_keys = 0          # bytes that named a binding
        self.n_ignored = 0       # bytes that named nothing (escape sequences count once per read)
        self.reason: str | None = None  # why start() refused, when it did

    # ----------------------------------------------------------------- pure part
    def help(self) -> str:
        return "hotkeys (this terminal): " + describe_bindings(self.bindings)

    def dispatch(self, ch: str) -> str | None:
        """Map ONE character to its action and run it. Returns the note to show, or None for an unbound key."""
        act = self.bindings.get(ch)
        if act is None:
            self.n_ignored += 1
            return None
        self.n_keys += 1
        if act == "help":
            return self.help()
        fn = self.actions[act]
        try:
            note = fn()
        except Exception as exc:  # noqa: BLE001 — the reader outlives one bad press
            return f"hotkey {key_label(ch)} ({act}): error {exc!r}"
        return f"hotkey {key_label(ch)} ({act}): {note}" if note else f"hotkey {key_label(ch)} ({act})"

    def feed(self, data: bytes, *, now: float | None = None) -> list[str]:
        """Feed the bytes of one read; escape sequences are consumed by their GRAMMAR, never by position.

        ``ESC [`` (CSI: arrows, Home/End, F5+) and ``ESC O`` (SS3: F1-F4) run to their final byte in
        ``@..~``; any other ``ESC x`` is Alt-x and is two bytes. A sequence split across two reads is
        carried into the next one (for at most :data:`ESC_CARRY_S`, after which a lone ESC was the
        Escape key itself and the next key is a key). So an arrow followed by ``k`` in the same read
        still fires ``k`` — the first version dropped the rest of the read and lost it (measured on a
        PTY in the test).
        """
        t = time.monotonic() if now is None else float(now)
        text = data.decode("utf-8", errors="replace")
        if self._esc_carry:
            if t - self._esc_carry_t <= ESC_CARRY_S:
                text = self._esc_carry + text
            else:
                self.n_ignored += 1  # the old lone ESC was the Escape key: dropped, nothing fired
            self._esc_carry = ""
        notes: list[str] = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch != ESC:
                note = self.dispatch(ch)
                if note is not None:
                    notes.append(note)
                i += 1
                continue
            if i + 1 >= n:
                self._esc_carry, self._esc_carry_t = ESC, t  # split read or a bare Escape: decide next read
                break
            nxt = text[i + 1]
            if nxt in "[O":
                j = i + 2
                while j < n and not ("@" <= text[j] <= "~"):
                    j += 1  # parameter / intermediate bytes (0x20-0x3F)
                if j >= n:
                    self._esc_carry, self._esc_carry_t = text[i:], t  # the final byte is in the next read
                    break
                i = j + 1
            else:
                i += 2  # Alt-<key>
            self.n_ignored += 1
        return notes

    # ------------------------------------------------------------- terminal part
    def availability(self) -> str | None:
        """None when the reader can run here, else the reason it must not."""
        try:
            fd = self._stdin.fileno()
        except (AttributeError, ValueError, OSError) as exc:
            return f"stdin has no file descriptor ({exc!r})"
        if not os.isatty(fd):
            return "stdin is not a TTY (launched under a pipe, nohup or a service) — hotkeys need the terminal"
        if self._require_fg:
            try:
                fg = os.tcgetpgrp(fd)
            except OSError as exc:
                return f"cannot read the terminal's foreground process group ({exc!r})"
            if fg != os.getpgrp():
                return ("this process is not the terminal's FOREGROUND job (launched with '&'?) — "
                        "reading its keys would stop the page with SIGTTIN")
        return None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Put the terminal in cbreak mode and start the reader. False (with ``reason`` set) when refused."""
        import termios
        import tty

        with self._lock:
            if self.active:
                return True
            self.reason = self.availability()
            if self.reason is not None:
                return False
            fd = self._stdin.fileno()
            self._saved_attrs = termios.tcgetattr(fd)
            tty.setcbreak(fd)  # no ICANON, no ECHO; ISIG kept so Ctrl-C still works
            self._fd = fd
            if not self._atexit_registered:
                atexit.register(self.stop)
                self._atexit_registered = True
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="tty-keys")
            self._thread.start()
            return True

    def stop(self) -> None:
        """Stop the reader and RESTORE the terminal. Idempotent; also runs at interpreter exit."""
        self._stop.set()
        with self._lock:
            if self._fd is not None and self._saved_attrs is not None:
                import termios

                try:
                    termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)
                except (OSError, termios.error):
                    pass
            self._fd = None
            self._saved_attrs = None
        th = self._thread
        if th is not None and th is not threading.current_thread():
            th.join(timeout=max(self._poll_s * 3, 0.5))
        self._thread = None

    def _loop(self) -> None:
        fd = self._fd
        while not self._stop.is_set() and fd is not None:
            try:
                ready, _, _ = select.select([fd], [], [], self._poll_s)
            except (OSError, ValueError):
                break  # the fd went away under us (stop() or a closed terminal)
            if not ready:
                continue
            try:
                data = os.read(fd, 64)
            except OSError:
                break
            if not data:
                break  # EOF — the terminal closed
            for note in self.feed(data):
                try:
                    self._log(note)
                except Exception:  # noqa: BLE001 — logging must not kill the reader
                    pass
