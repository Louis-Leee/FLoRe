"""Tests for the GIL yield gate (flore.s1_policy.serving.gil_yield).

The property that matters is the one that was measured on the rig: a pure-Python viz thread must
not be able to stretch an inference. So these tests do not merely check the latch bits — the
central one runs a real GIL-bound spinner against a real GIL-bound "inference" and asserts the
inference stays fast, which is the failure the gate exists to prevent (46 ms -> 3543 ms measured
with the ghost-robot arm IK against a 16-step diffusion sample).
"""

from __future__ import annotations

import threading
import time

import pytest

from flore.s1_policy.serving.gil_yield import YieldGate


def _burn(n: int) -> float:
    """A GIL-bound workload (pure Python, no sleeping, no I/O) — a stand-in for the ghost arm IK
    and for the sampler's Python-side launch loop."""
    acc = 0.0
    for i in range(n):
        acc += (i % 7) * 1.000001
    return acc


def test_open_by_default():
    """Nothing may block before anyone asks for priority."""
    g = YieldGate()
    assert not g.busy
    assert g.wait() is True


def test_priority_closes_and_reopens():
    g = YieldGate()
    with g.priority():
        assert g.busy
    assert not g.busy
    assert g.wait() is True


def test_priority_is_reentrant():
    """A nested priority section must not open the gate when the INNER one exits."""
    g = YieldGate()
    with g.priority():
        with g.priority():
            assert g.busy
        assert g.busy, "inner exit must not release the gate"
    assert not g.busy


def test_exception_inside_priority_still_reopens():
    """A gate left closed by a crashed predict would freeze the ghost preview for the rest of the
    session — the exact kind of silent viz death this file is trying to avoid."""
    g = YieldGate()
    with pytest.raises(RuntimeError):
        with g.priority():
            raise RuntimeError("sampler blew up")
    assert not g.busy
    assert g.wait() is True


def test_waiter_blocks_until_release():
    g = YieldGate()
    released_at = {}
    started = threading.Event()

    def waiter():
        started.set()
        g.wait()
        released_at["t"] = time.perf_counter()

    with g.priority():
        th = threading.Thread(target=waiter, daemon=True)
        th.start()
        started.wait(2.0)
        time.sleep(0.15)
        assert "t" not in released_at, "waiter passed the gate while priority was held"
        t_release = time.perf_counter()
    th.join(timeout=2.0)
    assert "t" in released_at
    assert released_at["t"] - t_release < 0.5, "waiter did not resume promptly after release"


def test_wait_times_out_and_proceeds_rather_than_hanging():
    """A viz thread that runs anyway is a slow predict; one that never runs again is a dead
    preview. So a lost release must degrade, not hang."""
    g = YieldGate(wait_timeout_s=0.05)
    g._enter()  # deliberately never exited (simulates a lost release)
    t0 = time.perf_counter()
    assert g.wait() is False, "a timed-out wait must report False"
    dt = time.perf_counter() - t0
    assert 0.04 <= dt < 1.0, f"wait should return after ~the timeout, took {dt:.3f}s"


def test_rejects_bad_timeout():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="wait_timeout_s"):
            YieldGate(wait_timeout_s=bad)


def test_stats_count_priority_sections_and_yields():
    g = YieldGate()
    with g.priority():
        pass
    assert g.stats()["n_priority"] == 1
    assert g.stats()["depth"] == 0


def test_yielding_spinner_does_not_stretch_a_gil_bound_section():
    """THE MEASURED FAILURE, in miniature.

    A pure-Python spinner and a pure-Python "inference" contend for the GIL. Checking the gate at
    the spinner's own chunk boundaries must keep the inference near its uncontended cost, and the
    same spinner WITHOUT the gate must be measurably worse. Both regimes are timed in-process so
    the comparison holds on any machine (the absolute numbers are hardware-specific; the ordering
    is what is pinned).
    """
    g = YieldGate()
    ITER = 60_000        # one spinner "step"
    INFER = 400_000      # one "inference"

    def timed_infer() -> float:
        t0 = time.perf_counter()
        _burn(INFER)
        return time.perf_counter() - t0

    # 1) uncontended baseline
    base = min(timed_infer() for _ in range(3))

    # 2) contended, spinner does NOT yield
    stop = threading.Event()

    def spin_rude():
        while not stop.is_set():
            _burn(ITER)

    th = threading.Thread(target=spin_rude, daemon=True)
    th.start()
    time.sleep(0.05)
    rude = min(timed_infer() for _ in range(3))
    stop.set()
    th.join(timeout=5)

    # 3) contended, spinner yields through the gate
    stop2 = threading.Event()

    def spin_polite():
        while not stop2.is_set():
            g.wait()
            _burn(ITER)

    th2 = threading.Thread(target=spin_polite, daemon=True)
    th2.start()
    time.sleep(0.05)
    polite_times = []
    for _ in range(3):
        with g.priority():
            polite_times.append(timed_infer())
    polite = min(polite_times)
    stop2.set()
    g._open.set()
    th2.join(timeout=5)

    assert rude > base * 1.3, (
        f"the un-yielded spinner did not measurably contend (base {base * 1e3:.1f} ms, "
        f"rude {rude * 1e3:.1f} ms) — this test cannot then prove the gate helps")
    assert polite < rude, (
        f"yielding ({polite * 1e3:.1f} ms) must beat not yielding ({rude * 1e3:.1f} ms)")
    assert polite < base * 1.6, (
        f"a yielding spinner should leave the section near its uncontended cost: "
        f"base {base * 1e3:.1f} ms vs polite {polite * 1e3:.1f} ms")
