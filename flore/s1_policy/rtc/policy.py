"""Policy protocol definitions for S1 action-policy inference.

Ported from FAR-lab42_vr rfm_heavy inference/vla/policy.py — the abstract
Protocol contract only, stripped of torch so the RTC layer stays numpy-only.

`ChunkPolicy` is the interface the DP-Flow model adapter must satisfy; `Policy`
is the interface `RealTimeChunkingPolicy` (and any local wrapper) exposes to
the gRPC server.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ChunkPolicy(Protocol):
    """Protocol for policies that generate action chunks from observations.

    A ChunkPolicy predicts a *chunk* of future actions at once and may
    maintain internal state (e.g., for Real-Time Chunking).
    """

    def predict_chunk(
        self,
        images: np.ndarray,
        state: np.ndarray,
        prompt: str | None = None,
        steps_executed: int = 0,
        delay_estimate: int = 0,
        prev_actions: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict an action chunk from an observation.

        Args:
            images: Camera images, shape (num_cameras, H, W, C) or batched.
            state: Robot state, shape (state_dim,) or batched.
            prompt: Optional language instruction.
            steps_executed: Steps executed from the previous chunk (for RTC).
            delay_estimate: Estimated inference delay in steps (for RTC).
            prev_actions: Previous chunk's actions (same shape/dtype as this
                method's return), or None on the first call of an episode.
                Used as the RTC frozen prefix.

        Returns:
            Action chunk, shape (chunk_size, action_dim) or batched.
        """
        ...

    def reset(self) -> None:
        """Reset internal state. Called at episode boundaries."""
        ...


@runtime_checkable
class Policy(Protocol):
    """Protocol defining the interface for policy inference wrappers.

    Required Attributes:
        config: Configuration dictionary for the policy.
        image_res: Expected image resolution (height/width in pixels).
        history_len: Number of historical observations used (0 if none).

    Required Methods:
        act: Generate an action from the current observation.
        reset_context: Reset internal state between episodes.
    """

    config: dict[str, Any]
    image_res: int
    history_len: int

    def act(
        self,
        state: np.ndarray,
        images: np.ndarray,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate an action from the current observation.

        Args:
            state: Robot state, shape (1, state_dim).
            images: Camera image(s), shape (1, num_cameras, H, W, C).
            prompt: Optional language instruction.
            **kwargs: Additional policy-specific arguments.

        Returns:
            Predicted action as a numpy array, dtype float32, shape (1, action_dim).
        """
        ...

    def reset_context(self) -> None:
        """Reset internal state between episodes."""
        ...
