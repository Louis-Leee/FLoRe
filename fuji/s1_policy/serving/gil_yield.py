"""A cooperative "stand back while the policy runs" gate for pure-Python viz threads.

WHY THIS EXISTS
---------------
``scripts/hl_serving.py`` runs the policy on the GUI callback thread and the ghost-robot arm IK
on a *background* thread. Both are **GIL-bound**: the IK is pure Python + small NumPy calls, and
a diffusion ``sample_actions`` is ~6100 tiny CUDA launches driven from Python (measured
``cpu/wall = 1.00`` — the launch loop holds the GIL for the whole sample, it never blocks on the
GPU). "Background thread" therefore buys nothing: the two threads interleave at the interpreter's
switch interval and each one's work is stretched by the other.

Measured on this box (RTX PRO 5000, ``ckpt_0814_diff_umi``: ViT-B + UNet, 16 DDIM steps):

    diffusion predict, ghost thread idle        46 ms
    diffusion predict, ghost thread solving   3543 ms     <-- 77x
    diffusion predict, ghost thread yielding    59 ms

The 77x is not a GPU problem — GPU utilization during the fight is ~1.3%, which is exactly the
"感觉 GPU 没有在被使用" the operator reported. The GPU is idle because nothing can reach it: the
thread that issues its work is waiting for the GIL.

WHAT THIS DOES
--------------
:class:`YieldGate` is a latch a *low-priority* thread checks at its own chunk boundaries and a
*high-priority* section holds while it runs. The low-priority thread blocks (releasing the GIL in
``Event.wait``) instead of competing, so the sample runs at full speed; it resumes on its own the
moment the section exits.

Deliberately cooperative rather than a mutex around the model call:

* The IK checks the gate BETWEEN steps, so a predict never waits for a solve to finish — the
  worst case is one step's ~2 ms, not a whole 0.6 s chunk.
* It cannot deadlock the render/publish path. A gate left held by a crashed predict would freeze
  the ghost preview forever, so :meth:`priority` is a context manager (released on exception) and
  :meth:`wait` takes a timeout: a wait that times out PROCEEDS. A viz thread that runs anyway is
  a slow predict; a viz thread that never runs again is a dead preview.
* Nothing here touches the robot. The wire, the drip cadence and the freshness gates are
  untouched — this only decides which Python thread gets to run.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

#: A viz thread waiting this long assumes the priority holder died and proceeds anyway. Long
#: enough for the slowest single inference here (a 16-step diffusion sample is ~50 ms; ACT is
#: ~6 ms), short enough that a lost release costs one visibly-late ghost update, not a hang.
DEFAULT_WAIT_TIMEOUT_S = 5.0


class YieldGate:
    """Let one section run without pure-Python threads competing for the GIL.

    Usage — the expensive, latency-critical section::

        with gate.priority():
            chunk = planner.infer_chunk(...)

    and, in the viz thread's own loop, at a point where pausing is harmless::

        for step in chunk:
            gate.wait()          # blocks (GIL-free) while a predict is in flight
            solve(step)

    Re-entrant across *nested* priority sections and safe with several priority threads at once
    (a counter, so an inner ``with`` does not open the gate on exit). ``wait()`` is not a lock:
    many viz threads may pass simultaneously — this throttles them against the priority section,
    it does not serialize them against each other.
    """

    def __init__(self, *, wait_timeout_s: float = DEFAULT_WAIT_TIMEOUT_S) -> None:
        if not (wait_timeout_s > 0.0):
            raise ValueError(f"wait_timeout_s must be > 0, got {wait_timeout_s}")
        self._wait_timeout_s = float(wait_timeout_s)
        self._open = threading.Event()
        self._open.set()  # open = viz threads may run
        self._lock = threading.Lock()
        self._depth = 0
        self._n_priority = 0
        self._n_yields = 0
        self._n_timeouts = 0

    # -- the viz (low-priority) side ------------------------------------
    def wait(self) -> bool:
        """Block while a priority section is in flight. True if clear to proceed.

        Returns False only when the wait TIMED OUT, and the caller should proceed regardless —
        the boolean is for diagnostics, not permission. Cheap in the common case (an already-set
        ``Event``), so calling it per step of a 50-step solve is fine.
        """
        if self._open.is_set():
            return True
        with self._lock:
            self._n_yields += 1
        if self._open.wait(timeout=self._wait_timeout_s):
            return True
        with self._lock:
            self._n_timeouts += 1
        return False

    # -- the policy (high-priority) side --------------------------------
    @contextmanager
    def priority(self):
        """Hold the gate CLOSED for the duration of the block.

        A context manager so an exception inside the model call cannot leave the gate shut (that
        would silently kill the ghost preview for the rest of the session)."""
        self._enter()
        try:
            yield self
        finally:
            self._exit()

    def _enter(self) -> None:
        with self._lock:
            self._depth += 1
            self._n_priority += 1
            self._open.clear()

    def _exit(self) -> None:
        with self._lock:
            self._depth = max(0, self._depth - 1)
            if self._depth == 0:
                self._open.set()

    # -- introspection --------------------------------------------------
    @property
    def busy(self) -> bool:
        """True while at least one priority section is in flight."""
        return not self._open.is_set()

    def stats(self) -> dict:
        with self._lock:
            return {
                "n_priority": self._n_priority,
                "n_yields": self._n_yields,
                "n_timeouts": self._n_timeouts,
                "depth": self._depth,
            }
