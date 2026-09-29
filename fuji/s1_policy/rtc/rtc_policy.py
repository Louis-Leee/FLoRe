"""RealTimeChunkingPolicy - runs predict_chunk() on a background thread.

Ported from FAR-lab42_vr rfm_heavy inference/vla/rtc_policy.py, which was
itself ported from FAR-pi rfmpi/inference/rtc_policy.py. The threading
semantics (Condition, s_min throttle, chunk takeover + skip_count, generation
counter for reset-during-inference) are production-proven and preserved
exactly — see the header note in the original.

Algorithm summary: a foreground caller pushes an observation every control
step via get_action() and drip-feeds actions out of the most recent chunk. A
background thread wakes when at least s_min steps have elapsed since the last
inference start, snapshots the current observation, and runs predict_chunk()
off-lock. When it finishes, it installs the new chunk and advances the read
index by skip_count = how many actions were consumed during the inference so
playback resumes without repeating or skipping steps. A generation counter lets
reset() discard an in-flight inference's result without killing the thread.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Any

import numpy as np
from typing_extensions import override

from fuji.s1_policy.rtc.policy import ChunkPolicy, Policy
from fuji.s1_policy.rtc.types import ActionChunk, Observation


class RealTimeChunkingPolicy(Policy):
    """Real-Time Chunking wrapper that runs predict_chunk() on a background thread."""

    def __init__(
        self,
        policy: ChunkPolicy,
        s_min: int = 10,
        delay_estimate: int = 0,
        naive_async: bool = False,
        chunk_wait_secs: float = 0.1,
        num_warmup_inferences: int = 5,
        timeout_secs: float = 3.0,
        logger=None,
    ):
        self._policy = policy
        self._s_min = s_min
        self._delay_estimate = delay_estimate
        self._naive_async = naive_async
        self._chunk_wait_secs = chunk_wait_secs
        self._num_warmup_inferences = num_warmup_inferences
        self._timeout_secs = timeout_secs
        self._logger = logger if logger is not None else logging.getLogger(__name__)

        self._cond = threading.Condition()
        self._observation: Observation | None = None
        self._obs_counter: int = 0
        self._step_counter: int = 0
        self._last_inference_start_step: int | None = None
        # Most recent chunk returned by predict_chunk. Drives get_action()
        # playback and also seeds the next predict_chunk call's RTC prefix
        # (via prev_actions = _current_chunk.actions[None]).
        self._current_chunk: ActionChunk | None = None
        self._action_idx: int = 0
        self._thread_exception: Exception | None = None
        self._inference_count: int = 0
        # Bumped by reset(); the inference loop captures it before releasing
        # the lock and drops its write-back if it changed during inference.
        # This lets reset() cancel an in-flight inference's effect without
        # killing the background thread (whose TLS holds cudagraph state).
        self._generation: int = 0

        self._running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="rtc_inference",
            daemon=True,
        )
        self._inference_thread.start()

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        self._inference_thread.join(timeout=5.0)

    def reset(self) -> None:
        """Reset all state while keeping the background thread alive.

        Preferred over :meth:`restart` after warmup: ``torch.compile`` with
        ``reduce-overhead`` captures CUDA graphs whose tree manager lives
        in thread-local storage on the thread that ran the first compiled
        call. Killing that thread invalidates the TLS entry and the next
        compiled call trips ``cudagraph_trees``' key-in-tls assert.

        Any in-flight inference is left running, but its result will be
        dropped by the loop's generation check before it can overwrite
        ``_current_chunk``.
        """
        with self._cond:
            self._generation += 1
            self._observation = None
            self._obs_counter = 0
            self._step_counter = 0
            self._last_inference_start_step = None
            self._current_chunk = None
            self._action_idx = 0
            self._thread_exception = None
            self._inference_count = 0
            self._cond.notify_all()
        self._policy.reset()

    def restart(self) -> None:
        """Stop the background thread, reset all state, and start a new thread.

        Used after warmup to ensure no in-flight inference pollutes real
        inference with dummy data. The caller is responsible for re-running
        warmup on the new thread — torch._inductor's CUDA graph tree
        manager lives in TLS, so the new thread starts without compiled
        graphs. Prefer :meth:`reset` when the background thread owns
        compiled CUDA graphs.
        """
        self.stop()
        self._policy.reset()
        self._observation = None
        self._obs_counter = 0
        self._step_counter = 0
        self._last_inference_start_step = None
        self._current_chunk = None
        self._action_idx = 0
        self._thread_exception = None
        self._inference_count = 0
        self._generation = 0
        self._running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="rtc_inference",
            daemon=True,
        )
        self._inference_thread.start()

    def wait_for_inference_count(self, target_count: int, timeout_secs: float = 600.0) -> None:
        """Block until the background thread has completed ``target_count`` inferences."""
        deadline = time.monotonic() + timeout_secs
        with self._cond:
            while self._inference_count < target_count:
                if self._thread_exception is not None:
                    raise self._thread_exception
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for RTC warmup inference count {self._inference_count}/{target_count}"
                    )
                self._cond.wait(timeout=remaining)

    def warmup(
        self,
        state: np.ndarray,
        images: np.ndarray,
        prompt: str = "warmup",
        min_inferences: int = 2,
        timeout_secs: float = 600.0,
    ) -> None:
        """Run enough ``act()`` calls to warm both initial and RTC replan graphs.

        The first RTC ``predict_chunk`` has no previous actions, so it compiles
        the non-RTC branch. The second one has a previous chunk available and
        compiles the true RTC branch. Drive enough foreground actions to trigger
        that second background inference, then wait for it to complete before
        callers reset the episode state.
        """
        target_count = max(1, min_inferences)
        required_actions = max(self._s_min + 1, target_count)
        for _ in range(required_actions):
            self.act(state, images, prompt=prompt)
        self.wait_for_inference_count(target_count, timeout_secs=timeout_secs)

    def _can_start_inference(self) -> bool:
        if self._observation is None:
            return False
        if self._last_inference_start_step is None:
            return True
        return self._step_counter - self._last_inference_start_step >= self._s_min

    def _inference_loop(self) -> None:
        with self._cond:
            while self._running:
                self._cond.wait_for(lambda: self._can_start_inference() or not self._running)
                if not self._running:
                    break

                obs = self._observation
                if obs is None:
                    continue
                self._observation = None
                self._last_inference_start_step = self._step_counter

                current_chunk_for_rtc = self._current_chunk
                action_idx_at_inference_start = self._action_idx
                delay_estimate = self._delay_estimate
                prev_actions = current_chunk_for_rtc.actions[None] if current_chunk_for_rtc is not None else None
                inference_generation = self._generation

                if self._naive_async:
                    current_chunk_for_rtc = None
                    delay_estimate = 0
                    prev_actions = None

                self._cond.release()
                inference_error: Exception | None = None
                inference_traceback: str | None = None
                chunk: ActionChunk | None = None
                _inf_start = time.perf_counter()
                try:
                    steps_executed = action_idx_at_inference_start if current_chunk_for_rtc is not None else 0
                    self._logger.info(
                        "RTC: starting predict_chunk (inference #%d, steps_executed=%d)",
                        self._inference_count + 1,
                        steps_executed,
                    )
                    action_chunk = self._policy.predict_chunk(
                        obs.images[None],
                        obs.state[None],
                        obs.prompt,
                        steps_executed=steps_executed,
                        delay_estimate=delay_estimate,
                        prev_actions=prev_actions,
                    )[0]
                    _inf_dur = (time.perf_counter() - _inf_start) * 1000
                    self._logger.info(
                        "RTC: predict_chunk done (inference #%d, %.1fms)",
                        self._inference_count + 1,
                        _inf_dur,
                    )
                    chunk = ActionChunk(actions=action_chunk, obs_idx=obs.obs_idx)
                except Exception as e:
                    # Capture the traceback here; sys.exc_info() is cleared after
                    # the try/except exits, so calling format_exc() below would
                    # return "NoneType: None" and the real cause would be lost.
                    inference_error = e
                    inference_traceback = traceback.format_exc()
                finally:
                    self._cond.acquire()

                stale = inference_generation != self._generation

                if inference_error is not None:
                    if stale:
                        # reset() was called during this inference; the
                        # caller has already discarded this episode's
                        # context, so the failure is no longer
                        # observable. Log and keep the thread alive.
                        self._logger.warning(
                            "RTC: dropping stale inference error (gen %d != %d): %r",
                            inference_generation,
                            self._generation,
                            inference_error,
                        )
                        continue
                    self._logger.error(
                        "RTC inference error: %r\n%s",
                        inference_error,
                        inference_traceback,
                    )
                    self._thread_exception = inference_error
                    self._cond.notify_all()
                    return

                if stale:
                    # Generation changed mid-inference; discard the result
                    # so it can't seed the next chunk's RTC prefix or be
                    # served from get_action().
                    #
                    # DEVIATION from rfm_heavy (whose model was stateless):
                    # our DPFlowModel caches `_last_actions` INTERNALLY at the
                    # tail of sample_actions — the stale inference has already
                    # re-marked that cache valid, overwriting the reset() the
                    # caller issued mid-flight. Re-clear the policy HERE (on
                    # the background thread, strictly after the stale write-
                    # back) so the previous episode's chunk can never seed the
                    # next episode's frozen prefix, regardless of interleaving.
                    self._policy.reset()
                    self._logger.info(
                        "RTC: dropping stale inference result (gen %d != %d)",
                        inference_generation,
                        self._generation,
                    )
                    continue

                if self._naive_async:
                    skip_count = 0
                elif action_idx_at_inference_start > 0:
                    actions_consumed = self._action_idx - action_idx_at_inference_start
                    skip_count = max(0, min(actions_consumed, len(chunk.actions)))
                else:
                    skip_count = 0

                self._current_chunk = chunk
                self._action_idx = skip_count
                self._inference_count += 1
                self._cond.notify_all()

    def get_action(
        self,
        images: np.ndarray,
        state: np.ndarray,
        prompt: str | None = None,
        timeout_secs: float = 3.0,
    ) -> tuple[np.ndarray, int]:
        with self._cond:
            # Fail fast if the inference thread has died
            if self._thread_exception is not None:
                raise self._thread_exception

            self._obs_counter += 1
            self._observation = Observation(
                images=images,
                state=state,
                prompt=prompt,
                obs_idx=self._obs_counter,
            )
            self._step_counter += 1
            self._cond.notify_all()

            if self._current_chunk is not None and self._action_idx >= len(self._current_chunk.actions):
                self._current_chunk = None

            if self._current_chunk is None:
                self._cond.wait(timeout=timeout_secs)
                if self._thread_exception is not None:
                    raise self._thread_exception
                if self._current_chunk is None:
                    raise TimeoutError(f"Timeout ({timeout_secs}s) waiting for action")

            action = self._current_chunk.actions[self._action_idx]
            obs_idx = self._current_chunk.obs_idx
            self._action_idx += 1
            return action, obs_idx

    @override
    def act(self, state, images, prompt=None, **kwargs):
        # Always use long timeout — torch.compile warmup can take minutes
        # and gRPC requests may arrive before warmup completes
        assert state.shape[0] == 1 and images.shape[0] == 1, (
            f"RTC act() expects batch size 1, got state={state.shape[0]}, images={images.shape[0]}"
        )
        timeout = max(self._timeout_secs, 600.0)
        action, _ = self.get_action(images[0], state[0], prompt, timeout_secs=timeout)
        return action[None]

    @override
    def reset_context(self) -> None:
        """Clear per-episode state so the next inference starts fresh.

        Delegates to :meth:`reset` so an in-flight inference cannot
        overwrite the cleared state when it completes.
        """
        self.reset()

    @property
    def config(self) -> dict[str, Any]:
        return self._policy.config

    @property
    def image_res(self) -> int:
        return self._policy.image_res

    @property
    def history_len(self) -> int:
        return self._policy.history_len
