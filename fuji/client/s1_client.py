"""gRPC client stub for the S1 DP-Flow policy server.

Connects to ``FujiPolicyService`` on ``host:port`` and exposes a simple
``predict(state, image_jpeg) -> action`` interface for the control loop.
"""

from __future__ import annotations

import logging
from typing import Any

import grpc
import numpy as np

logger = logging.getLogger(__name__)


class S1Client:
    """Synchronous gRPC client to the FUJI S1 policy server.

    Each :meth:`predict` call is a blocking Predict RPC that returns one
    drip-fed action step (the server-side RTC handles chunking and timing).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 50051, timeout: float = 600.0):
        self._address = f"{host}:{port}"
        self._timeout = timeout
        self._channel: grpc.Channel | None = None
        self._stub: Any = None
        self._pb2: Any = None
        self._pb2_grpc: Any = None
        self._connected = False

    def connect(self) -> dict:
        """Connect to the server and return policy info."""
        # Stubs are generated from fuji_policy.proto and checked into the package.
        from fuji.s1_policy.serving.proto import fuji_policy_pb2 as pb2
        from fuji.s1_policy.serving.proto import fuji_policy_pb2_grpc as pb2_grpc

        self._pb2 = pb2
        self._pb2_grpc = pb2_grpc
        self._channel = grpc.insecure_channel(
            self._address,
            options=[
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ],
        )
        self._stub = pb2_grpc.FujiPolicyServiceStub(self._channel)
        info = self._stub.GetPolicyInfo(pb2.PolicyInfoRequest(), timeout=self._timeout)
        self._connected = True
        logger.info(
            "S1Client connected to %s: action_dim=%d, horizon=%d, rtc=%s, s_min=%d",
            self._address,
            info.action_dim,
            info.action_horizon,
            info.rtc_enabled,
            info.s_min,
        )
        return {
            "action_dim": info.action_dim,
            "action_horizon": info.action_horizon,
            "checkpoint": info.checkpoint,
            "rtc_enabled": info.rtc_enabled,
            "s_min": info.s_min,
            "delay_estimate": info.delay_estimate,
        }

    def predict(self, state: np.ndarray, image_jpeg: bytes, prompt: str = "") -> np.ndarray:
        """Send one observation, receive one drip-fed action (32D).

        Blocks until the server's RTC scheduler has an action ready. On cold
        start this can take tens of seconds (model load + first inference).
        """
        assert self._connected, "call connect() first"
        req = self._pb2.PredictRequest(
            state=state.astype(np.float32).tolist(),
            image=image_jpeg,
            prompt=prompt,
        )
        resp = self._stub.Predict(req, timeout=self._timeout)
        return np.asarray(resp.action, dtype=np.float32)

    def predict_chunk(self, state: np.ndarray, image_jpeg: bytes, prompt: str = "") -> np.ndarray:
        """Request a full action chunk (debug/offline use)."""
        assert self._connected, "call connect() first"
        req = self._pb2.PredictRequest(
            state=state.astype(np.float32).tolist(),
            image=image_jpeg,
            prompt=prompt,
        )
        resp = self._stub.PredictChunk(req, timeout=self._timeout)
        return np.asarray(resp.actions, dtype=np.float32).reshape(resp.horizon, resp.action_dim)

    def reset(self) -> None:
        """Signal episode boundary to the server (clears RTC state)."""
        assert self._connected, "call connect() first"
        self._stub.Reset(self._pb2.ResetRequest(), timeout=10.0)
        logger.info("S1Client: episode reset sent")

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._connected = False
