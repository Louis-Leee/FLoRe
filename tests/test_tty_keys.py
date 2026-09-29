"""Terminal hotkeys for the 5pt page (``fuji/s1_policy/serving/tty_keys.py``) and their wiring.

The pure half (key → action → note, escape-sequence swallowing, the binding validator) is checked
directly; the terminal half against a real PTY pair from the stdlib (``pty.openpty``), which is a TTY
this process does NOT own as its controlling terminal — exactly the case the foreground check must
refuse and ``require_foreground=False`` must accept. The wiring pins are source-level over
``scripts/hl_serving_5pt.py`` for the same reason ``test_walk_skill.py``'s are: the page is one closure
over six sockets and a GPU, and what has to hold is that a key and its button call ONE closure.
"""

from __future__ import annotations

import os
import pty
import re
import termios
import threading
import time
from pathlib import Path

import pytest

from fuji.s1_policy.serving import tty_keys as K

_PAGE = Path(__file__).resolve().parents[1] / "scripts" / "hl_serving_5pt.py"


def _actions(log=None):
    calls: list[str] = []
    fired = threading.Event()

    def mk(name):
        def fn():
            calls.append(name)
            fired.set()
            return f"{name} ok"
        return fn

    return {"run": mk("run"), "walk": mk("walk"), "legs": mk("legs"), "stop": mk("stop"),
            "jaws_close": mk("jaws_close"), "jaws_open": mk("jaws_open")}, calls, fired


# ------------------------------------------------------------------ pure part
def test_default_bindings_map_the_four_buttons_and_help():
    acts, calls, _ = _actions()
    k = K.TtyKeys(acts, stdin=open(os.devnull, "rb"), require_foreground=False)
    assert k.dispatch("p") == "hotkey p (run): run ok"
    assert k.dispatch("k") == "hotkey k (walk): walk ok"
    assert k.dispatch("l") == "hotkey l (legs): legs ok"
    assert k.dispatch("c") == "hotkey c (jaws_close): jaws_close ok"
    assert k.dispatch("u") == "hotkey u (jaws_open): jaws_open ok"
    assert k.dispatch(" ") == "hotkey SPACE (stop): stop ok"
    assert calls == ["run", "walk", "legs", "jaws_close", "jaws_open", "stop"]
    for ch in ("h", "?"):
        note = k.dispatch(ch)
        assert note.startswith("hotkeys (this terminal): ") and "▶▶" in note and "SPACE = STOP" in note
        assert "l = 🦵 legs" in note and "c = ⬛ close BOTH jaws" in note and "u = ⬜ open BOTH jaws" in note
    assert calls == ["run", "walk", "legs", "jaws_close", "jaws_open", "stop"]  # help runs nothing
    assert k.n_keys == 8


def test_l_is_the_legs_key_and_x_is_refused_because_it_is_holosomas_recenter():
    """The user's first choice for the leg override was `x`; it is holosoma's RECENTER, so the map
    refuses it — typed into the low level's terminal by mistake it would re-anchor the teleop frame."""
    assert K.DEFAULT_BINDINGS["l"] == "legs" and "l" not in K.HOLOSOMA_KEYBOARD_KEYS
    assert "x" in K.HOLOSOMA_KEYBOARD_KEYS
    with pytest.raises(ValueError, match="holosoma's own keyboard map"):
        K.validate_bindings({"x": "legs"}, {"legs": lambda: None})
    assert "legs" in K.KNOWN_ACTIONS


def test_c_and_u_drive_both_jaws_and_o_is_refused_because_it_is_holosomas_stop_and_the_grippers_open():
    """One key = the Manual-jaws button of EACH hand (2026-09-21, the user's ask: close both jaws from the
    keyboard exactly as the panel buttons do). `o`, the natural 'open', is holosoma's STOP and the gripper
    evdev bridge's default OPEN — the collision this map exists to avoid — so it is `u`."""
    assert K.DEFAULT_BINDINGS["c"] == "jaws_close" and K.DEFAULT_BINDINGS["u"] == "jaws_open"
    assert not ({"c", "u"} & K.HOLOSOMA_KEYBOARD_KEYS)
    assert "o" in K.HOLOSOMA_KEYBOARD_KEYS
    with pytest.raises(ValueError, match="holosoma's own keyboard map"):
        K.validate_bindings({"o": "jaws_open"}, {"jaws_open": lambda: None})
    acts, _, _ = _actions()
    with pytest.raises(ValueError, match="handed no 'jaws_close' action"):
        K.TtyKeys({k: v for k, v in acts.items() if k != "jaws_close"},
                  stdin=open(os.devnull, "rb"), require_foreground=False)


def test_an_unbound_key_is_ignored_and_counted_not_an_error():
    acts, calls, _ = _actions()
    k = K.TtyKeys(acts, stdin=open(os.devnull, "rb"), require_foreground=False)
    for ch in "xzq1\r\n\t":
        assert k.dispatch(ch) is None
    assert calls == [] and k.n_ignored == 7 and k.n_keys == 0


def test_escape_sequences_are_consumed_by_grammar_so_alt_p_and_arrows_fire_nothing():
    acts, calls, _ = _actions()
    k = K.TtyKeys(acts, stdin=open(os.devnull, "rb"), require_foreground=False)
    assert k.feed(b"\x1b[A") == []        # arrow up (CSI): the '[' and 'A' must not be looked up
    assert k.feed(b"\x1b[1;5C") == []     # Ctrl-right: parameter bytes up to the final 'C'
    assert k.feed(b"\x1bOP") == []        # F1 (SS3)
    assert k.feed(b"\x1bp") == []         # Alt-p: the 'p' must NOT start a run
    assert k.feed(b"\x1bk\x1b ") == []    # Alt-k then Alt-space in one read
    assert calls == [] and k.n_ignored == 6
    assert k.feed(b"p") == ["hotkey p (run): run ok"]
    assert k.feed(b"kp") == ["hotkey k (walk): walk ok", "hotkey p (run): run ok"]  # two keys in one read
    assert calls == ["run", "walk", "run"]


def test_a_key_on_either_side_of_a_sequence_in_the_same_read_still_fires():
    """The first version dropped the rest of the read after an ESC and lost the `k` after an arrow."""
    acts, calls, _ = _actions()
    k = K.TtyKeys(acts, stdin=open(os.devnull, "rb"), require_foreground=False)
    assert k.feed(b"p\x1b[B") == ["hotkey p (run): run ok"]
    assert k.feed(b"\x1b[Ak") == ["hotkey k (walk): walk ok"]
    assert k.feed(b"\x1b[A k\x1bOQp") == ["hotkey SPACE (stop): stop ok", "hotkey k (walk): walk ok",
                                            "hotkey p (run): run ok"]   # arrow, SPACE, k, F2, p
    assert calls == ["run", "walk", "stop", "walk", "run"]


def test_a_sequence_split_across_reads_is_carried_and_a_lone_escape_key_is_not():
    acts, calls, _ = _actions()
    k = K.TtyKeys(acts, stdin=open(os.devnull, "rb"), require_foreground=False)
    # Split arrow: ESC in one read, "[A" 2 ms later — the 'A' must not fire even if it were bound.
    assert k.feed(b"\x1b", now=10.000) == []
    assert k.feed(b"[Ak", now=10.002) == ["hotkey k (walk): walk ok"]
    # A CSI whose final byte comes in the next read.
    assert k.feed(b"\x1b[1;", now=11.000) == []
    assert k.feed(b"5Cp", now=11.001) == ["hotkey p (run): run ok"]
    # The Escape KEY itself, then `p` half a second later: the p is a key, not Alt-p.
    assert k.feed(b"\x1b", now=20.0) == []
    assert k.feed(b"p", now=20.5) == ["hotkey p (run): run ok"]
    assert calls == ["walk", "run", "run"]


def test_an_action_that_raises_is_reported_not_re_raised():
    def boom():
        raise RuntimeError("no low level")

    k = K.TtyKeys({"run": boom, "walk": lambda: None, "legs": lambda: None, "stop": lambda: "frozen",
                   "jaws_close": lambda: None, "jaws_open": lambda: None},
                  stdin=open(os.devnull, "rb"), require_foreground=False)
    assert k.dispatch("p") == "hotkey p (run): error RuntimeError('no low level')"
    assert k.dispatch("k") == "hotkey k (walk)"          # a None note is a bare acknowledgement
    assert k.dispatch(" ") == "hotkey SPACE (stop): frozen"


@pytest.mark.parametrize("ch", sorted("wsadqeo]irxz=mtT123456789"))
def test_a_binding_on_one_of_holosomas_own_keys_is_refused(ch):
    acts, _, _ = _actions()
    with pytest.raises(ValueError, match="holosoma's own keyboard map"):
        K.TtyKeys(acts, bindings={ch: "run"}, stdin=open(os.devnull, "rb"), require_foreground=False)


def test_the_default_bindings_are_all_outside_holosomas_map():
    assert not (set(K.DEFAULT_BINDINGS) & K.HOLOSOMA_KEYBOARD_KEYS)
    assert set(K.DEFAULT_BINDINGS.values()) == set(K.KNOWN_ACTIONS)


def test_the_validator_refuses_unknown_missing_multichar_and_esc_bindings():
    acts, _, _ = _actions()
    with pytest.raises(ValueError, match="unknown action"):
        K.TtyKeys(acts, bindings={"p": "reset"}, stdin=open(os.devnull, "rb"), require_foreground=False)
    with pytest.raises(ValueError, match="handed no 'stop' action"):
        K.TtyKeys({"run": acts["run"], "walk": acts["walk"]}, bindings={" ": "stop"},
                  stdin=open(os.devnull, "rb"), require_foreground=False)
    with pytest.raises(ValueError, match="exactly one character"):
        K.TtyKeys(acts, bindings={"pp": "run"}, stdin=open(os.devnull, "rb"), require_foreground=False)
    with pytest.raises(ValueError, match="ESC cannot be bound"):
        K.TtyKeys(acts, bindings={"\x1b": "stop"}, stdin=open(os.devnull, "rb"), require_foreground=False)


def test_describe_bindings_is_action_major_and_names_the_space_key():
    d = K.describe_bindings()
    assert d.startswith("p = ▶▶ run / ⏹ pause (one key)")
    assert "k = 🚶 walk toggle" in d and "SPACE = STOP" in d and "h / ? = help" in d


# -------------------------------------------------------------- terminal part
def test_a_pipe_is_refused_by_name_and_the_page_runs_on_without_the_reader():
    r, w = os.pipe()
    try:
        acts, _, _ = _actions()
        k = K.TtyKeys(acts, stdin=os.fdopen(r, "rb", buffering=0), log=lambda s: None)
        assert k.start() is False
        assert k.reason is not None and "not a TTY" in k.reason
        assert not k.active
        k.stop()  # idempotent with nothing started
    finally:
        os.close(w)


def test_a_tty_this_process_does_not_own_fails_the_foreground_check():
    master, slave = pty.openpty()
    try:
        acts, _, _ = _actions()
        k = K.TtyKeys(acts, stdin=os.fdopen(slave, "rb", buffering=0, closefd=False))
        reason = k.availability()
        assert reason is not None and "foreground" in reason.lower()
        assert k.start() is False and not k.active
    finally:
        os.close(master)
        os.close(slave)


def test_on_a_real_pty_keys_fire_actions_and_stop_restores_the_terminal():
    master, slave = pty.openpty()
    notes: list[str] = []
    try:
        before = termios.tcgetattr(slave)
        assert before[3] & termios.ICANON and before[3] & termios.ECHO  # a fresh pty is cooked + echoing
        acts, calls, fired = _actions()
        k = K.TtyKeys(acts, stdin=os.fdopen(slave, "rb", buffering=0, closefd=False),
                      log=notes.append, require_foreground=False, poll_s=0.05)
        assert k.start() is True and k.active
        during = termios.tcgetattr(slave)
        assert not (during[3] & termios.ICANON) and not (during[3] & termios.ECHO)  # cbreak: raw keys, no echo
        assert during[3] & termios.ISIG                                            # Ctrl-C still interrupts

        os.write(master, b"p")
        assert fired.wait(2.0), "the p key did not reach the action"
        fired.clear()
        os.write(master, b"\x1b[A")  # an arrow key must fire nothing
        os.write(master, b"k")
        assert fired.wait(2.0)
        os.write(master, b"h")
        deadline = time.time() + 2.0
        while time.time() < deadline and not any(n.startswith("hotkeys (this terminal)") for n in notes):
            time.sleep(0.02)
        assert calls == ["run", "walk"]
        assert [n for n in notes if n.startswith("hotkey ")] == ["hotkey p (run): run ok", "hotkey k (walk): walk ok"]
        assert any(n.startswith("hotkeys (this terminal): p = ▶▶") for n in notes)

        k.stop()
        assert not k.active
        after = termios.tcgetattr(slave)
        assert after[3] == before[3] and after[:3] == before[:3]  # the terminal is exactly as it was
        k.stop()  # idempotent
        assert k.start() is True  # and restartable
        k.stop()
        assert termios.tcgetattr(slave)[3] == before[3]
    finally:
        os.close(master)
        os.close(slave)


# ------------------------------------------------------------------- wiring
def _src() -> str:
    return _PAGE.read_text()


def test_the_page_exposes_no_keys_and_hands_the_four_actions_to_one_reader():
    s = _src()
    assert '"--no-keys", dest="keys", action="store_false"' in s
    assert re.search(r'tty_keys\.TtyKeys\(\{"run": _key_run, "walk": _key_walk, "legs": _legs_toggle, "stop": _stop_now,'
                     r'\s*"jaws_close": _key_jaws_close, "jaws_open": _key_jaws_open\},', s)
    # One import statement hands the page every serving module (single-line or parenthesized — the
    # 2026-09-19 replay batch made it the latter); tty_keys must be among them, not a second import.
    m = re.search(r"from fuji\.s1_policy\.serving import (\([^)]*\)|[^\n]*)", s)
    assert m, "the page imports the serving package once"
    for name in ("hl_5pt", "hl_dense", "hl_stream", "rollout_record", "tty_keys", "walk_skill"):
        assert re.search(rf"\b{name}\b", m.group(1)), name
    assert s.count("import tty_keys") + s.count(" tty_keys,") + s.count(" tty_keys\n") <= 2


@pytest.mark.parametrize("button, closure", [
    ("b_auto", "_auto_start"),
    ("b_auto_stop", "_auto_retire"),
    ("b_stop", "_stop_now"),
    ("b_walk", "_walk_toggle"),
    ("b_legs", "_legs_toggle"),
])
def test_each_button_is_a_thin_call_to_the_closure_its_hotkey_uses(button, closure):
    s = _src()
    # `@b_x.on_click` / `def _(_e) -> None:` / `<closure>()` — nothing else in the handler body.
    pat = re.compile(rf"@{button}\.on_click\n\s+def _\(_e\) -> None:\n\s+{closure}\(\)\n\n", re.M)
    assert pat.search(s), f"{button} must call {closure}() and nothing else"
    assert f"def {closure}() -> str:" in s


def test_the_jaw_keys_are_the_manual_jaw_buttons_of_both_hands_and_nothing_else():
    """`c` / `u` must be the ⬛ / ⬜ button bodies applied to EVERY manual-jaw role: the slider, then
    `_manual_jaw` (which latches the override) — no refusal of their own, no extra frame."""
    s = _src()
    body = s[s.index("def _jaws_both(frac: float) -> str:"): s.index("def _key_jaws_close() -> str:")]
    assert "for role in MANUAL_JAW_ROLES:" in body
    assert "s_jaw[role].value = frac" in body and "notes.append(_manual_jaw(role, frac))" in body
    assert "g_msg.value = note" in body
    for name, frac in (("_key_jaws_close", "0.0"), ("_key_jaws_open", "1.0")):
        fn = s[s.index(f"def {name}() -> str:"):]
        assert fn.split("\n")[1].strip() == f"return _jaws_both({frac})"
    # the per-hand buttons are unchanged: the same two statements per press
    close_btn = s[s.index("@_b_close.on_click"): s.index("@_b_set.on_click")]
    assert "s_jaw[role].value = 0.0" in close_btn and "_manual_jaw(role, 0.0)" in close_btn
    open_btn = s[s.index("@_b_open.on_click"): s.index("@_b_close.on_click")]
    assert "s_jaw[role].value = 1.0" in open_btn and "_manual_jaw(role, 1.0)" in open_btn


def test_the_hotkey_pairs_run_and_pause_on_one_key_and_refuses_the_two_orderings_by_name():
    s = _src()
    run = s[s.index("def _key_run() -> str:"): s.index("def _key_walk() -> str:")]
    assert "if auto_running.is_set():\n            return _auto_retire()" in run
    assert "if walk_running.is_set():" in run and "press k to stop it" in run
    assert run.rstrip().endswith("return _auto_start()")
    walk = s[s.index("def _key_walk() -> str:"): s.index("keys = tty_keys.TtyKeys(")]
    assert "if auto_running.is_set():" in walk and "press p to pause it" in walk
    assert walk.rstrip().endswith("return _walk_toggle()")


def test_the_reader_is_stopped_on_ctrl_c_so_the_terminal_is_restored():
    s = _src()
    i = s.index("except KeyboardInterrupt:")
    assert "keys.stop()" in s[i: i + 400]


def test_the_retire_key_is_the_boundary_stop_and_never_the_interrupt():
    """`p` while running retires (⏹); STOP has its own key — an interrupt must not be one press from pause."""
    s = _src()
    run = s[s.index("def _key_run() -> str:"): s.index("def _key_walk() -> str:")]
    assert "_stop_now" not in run and "stop.set()" not in run
    assert K.DEFAULT_BINDINGS[" "] == "stop" and K.DEFAULT_BINDINGS["p"] == "run"
