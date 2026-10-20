"""Tests for RealTimeChunkingPolicy threading + action delivery semantics.

Uses a FAKE ChunkPolicy that emits deterministic, monotonically increasing
chunks with a controllable artificial latency (event- or sleep-gated), so the
four load-bearing RTC behaviors can be asserted deterministically on CPU:

  (a) no gaps  — get_action returns actions in order across a chunk boundary.
  (b) skip_count — after a replan that consumed k steps of latency, the new
      chunk resumes at index k (no repeated / backward action).
  (c) s_min throttle — inference count over N steps is <= N/s_min + 1.
  (d) reset() discards an in-flight inference's result (generation counter).

The RTC semantics here are a verbatim port of rfm_heavy's production RTC; these
tests are the FLoRe-local subset that pins the behaviors called out in the port
task. See the source repo's test_rtc_policy.py for the full timeline suite.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from flore.s1_policy.rtc import RealTimeChunkingPolicy

ACTION_DIM = 32


# ---------------------------------------------------------------------------
# Fake chunk policies.
# ---------------------------------------------------------------------------
class DeterministicChunkPolicy:
    """Fake ChunkPolicy: chunk k has actions valued (k*100 + i) on every dim.

    Optional per-call artificial latency via ``inference_delay`` (seconds).
    Records prev_actions / steps_executed for assertions.
    """

    def __init__(self, chunk_size: int = 10, action_dim: int = ACTION_DIM, inference_delay: float = 0.0):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.inference_delay = inference_delay
        self.call_count = 0
        self.steps_executed_history: list[tuple[int, int]] = []
        self.prev_actions_history: list[np.ndarray | None] = []
        # Policy-protocol passthrough attributes RTC reads.
        self.image_res = 224
        self.history_len = 0
        self.config: dict = {"image_res": 224, "history_len": 0}

    def predict_chunk(self, images, state, prompt=None, steps_executed=0, delay_estimate=0, prev_actions=None):
        chunk_idx = self.call_count
        self.call_count += 1
        self.steps_executed_history.append((steps_executed, delay_estimate))
        self.prev_actions_history.append(None if prev_actions is None else prev_actions.copy())
        if self.inference_delay > 0:
            time.sleep(self.inference_delay)
        chunk = np.empty((1, self.chunk_size, self.action_dim), dtype=np.float32)
        for i in range(self.chunk_size):
            chunk[0, i, :] = chunk_idx * 100 + i
        return chunk

    def reset(self) -> None:
        pass


class EventGatedChunkPolicy:
    """Fake ChunkPolicy whose inferences block on per-call events.

    Chunk k holds value (k*100 + i); when RTC hands a valid prefix (delay_estimate
    > 0 and prev_actions is not None), the first ``prefix_len`` rows are copied
    from the previous chunk exactly as the real RTC model would inpaint them, so
    the takeover boundary is continuous.
    """

    def __init__(self, chunk_size: int = 10, action_dim: int = ACTION_DIM):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.call_count = 0
        self.started_events: list[threading.Event] = []
        self.complete_events: list[threading.Event] = []
        self.steps_executed_history: list[tuple[int, int]] = []
        self.prev_actions_history: list[np.ndarray | None] = []
        self.image_res = 224
        self.history_len = 0
        self.config: dict = {"image_res": 224, "history_len": 0}

    def add_events(self, count: int) -> None:
        for _ in range(count):
            self.started_events.append(threading.Event())
            self.complete_events.append(threading.Event())

    def predict_chunk(self, images, state, prompt=None, steps_executed=0, delay_estimate=0, prev_actions=None):
        chunk_idx = self.call_count
        self.call_count += 1
        self.steps_executed_history.append((steps_executed, delay_estimate))
        self.prev_actions_history.append(None if prev_actions is None else prev_actions.copy())
        if chunk_idx < len(self.started_events):
            self.started_events[chunk_idx].set()
        if chunk_idx < len(self.complete_events):
            self.complete_events[chunk_idx].wait(timeout=5.0)

        chunk = np.empty((1, self.chunk_size, self.action_dim), dtype=np.float32)
        for i in range(self.chunk_size):
            chunk[0, i, :] = chunk_idx * 100 + i
        # RTC inpaint: freeze the shifted prefix so the takeover is continuous.
        if delay_estimate > 0 and prev_actions is not None and steps_executed < self.chunk_size:
            remaining = self.chunk_size - steps_executed
            prefix_len = min(delay_estimate, remaining)
            chunk[0, :prefix_len, :] = prev_actions[0, steps_executed : steps_executed + prefix_len, :]
        return chunk

    def reset(self) -> None:
        pass


def _obs(state_dim: int = ACTION_DIM):
    images = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
    state = np.random.rand(state_dim).astype(np.float32)
    return images, state


# ---------------------------------------------------------------------------
# (a) No gaps across a chunk boundary.
# ---------------------------------------------------------------------------
def test_actions_in_order_no_gaps_across_boundary():
    """With s_min=chunk_size and naive_async, RTC reproduces synchronous open-loop:
    actions are drip-fed strictly in order and every chunk boundary is seamless."""
    chunk_size = 5
    mock = DeterministicChunkPolicy(chunk_size=chunk_size)
    policy = RealTimeChunkingPolicy(mock, s_min=chunk_size, delay_estimate=0, naive_async=True)

    got = []
    for _ in range(chunk_size * 3):  # cross two boundaries
        images, state = _obs()
        action, _ = policy.get_action(images, state, timeout_secs=5.0)
        got.append(int(action[0]))
    policy.stop()

    # Synchronous reference: chunk k feeds values k*100 + [0..chunk_size).
    expected = []
    for k in range(3):
        expected.extend(k * 100 + i for i in range(chunk_size))
    assert got == expected


# ---------------------------------------------------------------------------
# (b) skip_count: new chunk resumes at the number of steps consumed during the
#     replan latency — no repeated or backward action at takeover.
# ---------------------------------------------------------------------------
def test_skip_count_resumes_without_repeat_or_backtrack():
    mock = EventGatedChunkPolicy(chunk_size=10)
    mock.add_events(5)
    policy = RealTimeChunkingPolicy(mock, s_min=1, delay_estimate=5)

    def get_one():
        images, state = _obs()
        action, obs_idx = policy.get_action(images, state, timeout_secs=2.0)
        return int(action[0]), obs_idx

    got, obs_ids = [], []

    # Chunk #0 completes; consume two of its steps (0, 1).
    mock.complete_events[0].set()
    for _ in range(2):
        a, o = get_one()
        got.append(a)
        obs_ids.append(o)

    # Inference #1 has started (s_min=1). Consume two more from chunk #0 (2, 3)
    # while #1 is in-flight, then let it complete.
    mock.started_events[1].wait(timeout=1.0)
    for _ in range(2):
        a, o = get_one()
        got.append(a)
        obs_ids.append(o)
    mock.complete_events[1].set()
    time.sleep(0.02)

    # Takeover: chunk #1 froze the shifted prefix, and skip_count advances the
    # read index past the steps consumed during the replan → continuous 4,5,6.
    for _ in range(3):
        a, o = get_one()
        got.append(a)
        obs_ids.append(o)
    policy.stop()

    assert got == [0, 1, 2, 3, 4, 5, 6]
    # obs_idx flips to the second chunk exactly at the takeover step.
    assert obs_ids == [1, 1, 1, 1, 2, 2, 2]
    # Inference #1 saw steps_executed > 0 (RTC replan), #0 started at 0.
    assert mock.steps_executed_history[0][0] == 0
    assert mock.steps_executed_history[1][0] > 0


def test_skip_count_naive_async_jumps_to_new_chunk_start():
    """Contrast: naive_async has skip_count=0, so takeover jumps to the new
    chunk's index 0 (a discontinuity), proving skip_count is what removes it."""
    mock = EventGatedChunkPolicy(chunk_size=10)
    mock.add_events(5)
    policy = RealTimeChunkingPolicy(mock, s_min=1, delay_estimate=0, naive_async=True)

    def get_one():
        images, state = _obs()
        action, _ = policy.get_action(images, state, timeout_secs=2.0)
        return int(action[0])

    got = []
    mock.complete_events[0].set()
    got.append(get_one())
    got.append(get_one())
    mock.started_events[1].wait(timeout=1.0)
    got.append(get_one())
    got.append(get_one())
    mock.complete_events[1].set()
    time.sleep(0.02)
    got.extend(get_one() for _ in range(3))
    policy.stop()

    # Chunk #1 starts at value 100 with no prefix continuity.
    assert got == [0, 1, 2, 3, 100, 101, 102]


# ---------------------------------------------------------------------------
# (c) s_min throttle: inferences over N steps <= N/s_min + 1.
# ---------------------------------------------------------------------------
def test_s_min_throttle_bounds_inference_count():
    s_min = 8
    n_steps = 40
    chunk_size = 50  # never exhausted, so replans are driven purely by s_min
    mock = DeterministicChunkPolicy(chunk_size=chunk_size, inference_delay=0.0)
    policy = RealTimeChunkingPolicy(mock, s_min=s_min, delay_estimate=0)

    for _ in range(n_steps):
        images, state = _obs()
        policy.get_action(images, state, timeout_secs=2.0)
        time.sleep(0.002)  # let the background thread keep pace
    policy.stop()

    max_allowed = n_steps // s_min + 1
    assert mock.call_count <= max_allowed, f"{mock.call_count} inferences > bound {max_allowed}"


# ---------------------------------------------------------------------------
# (d) reset() discards an in-flight inference's result (generation counter).
# ---------------------------------------------------------------------------
def test_reset_discards_in_flight_result():
    mock = EventGatedChunkPolicy(chunk_size=5)
    mock.add_events(3)
    policy = RealTimeChunkingPolicy(mock, s_min=1, delay_estimate=2)

    # Drain chunk #0 to force inference #1 to start.
    mock.complete_events[0].set()
    for _ in range(5):
        images, state = _obs()
        policy.get_action(images, state, timeout_secs=2.0)
    mock.started_events[1].wait(timeout=2.0)

    # reset() while inference #1 is mid-flight — bumps the generation counter.
    policy.reset()

    # Release #1; its result must be dropped by the generation check.
    mock.complete_events[1].set()
    time.sleep(0.05)

    assert policy._current_chunk is None
    assert policy._obs_counter == 0
    assert policy._step_counter == 0
    assert policy._inference_count == 0

    # The first inference AFTER reset must start fresh (prev_actions=None).
    mock.complete_events[2].set()
    for _ in range(2):
        images, state = _obs()
        policy.get_action(images, state, timeout_secs=2.0)
    policy.stop()

    assert mock.prev_actions_history[0] is None
    assert mock.prev_actions_history[2] is None


def test_reset_keeps_background_thread_alive():
    mock = EventGatedChunkPolicy(chunk_size=5)
    mock.add_events(3)
    policy = RealTimeChunkingPolicy(mock, s_min=1, delay_estimate=2)
    thread_before = policy._inference_thread

    mock.complete_events[0].set()
    for _ in range(5):
        images, state = _obs()
        policy.get_action(images, state, timeout_secs=2.0)
    mock.started_events[1].wait(timeout=2.0)
    mock.complete_events[1].set()

    policy.reset()
    assert policy._inference_thread is thread_before
    assert policy._inference_thread.is_alive()
    policy.stop()


# ---------------------------------------------------------------------------
# Supporting behaviors: exception propagation + first-call prev_actions=None.
# ---------------------------------------------------------------------------
def test_exception_propagates_to_caller():
    class FailingPolicy(DeterministicChunkPolicy):
        def predict_chunk(self, *a, **k):
            super().predict_chunk(*a, **k)
            raise RuntimeError("boom")

    mock = FailingPolicy(chunk_size=5)
    policy = RealTimeChunkingPolicy(mock, s_min=0, delay_estimate=0)
    with pytest.raises(RuntimeError, match="boom"):
        for _ in range(5):
            images, state = _obs()
            policy.get_action(images, state, timeout_secs=2.0)
    policy.stop()


def test_first_inference_receives_none_prev_actions():
    mock = DeterministicChunkPolicy(chunk_size=5)
    policy = RealTimeChunkingPolicy(mock, s_min=1, delay_estimate=2)
    for _ in range(6):
        images, state = _obs()
        policy.get_action(images, state, timeout_secs=2.0)
    policy.stop()
    assert mock.prev_actions_history[0] is None
