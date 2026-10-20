#!/usr/bin/env python3
"""Launch the FLoRe S1 DP-Flow gRPC policy server.

Loads a DP-Flow checkpoint, wraps it in the RTC scheduler (default) or a
synchronous drip wrapper, and serves ``FlorePolicyService`` over gRPC.

Examples::

    uv run scripts/serve_s1.py --checkpoint checkpoints/dp_flow/step_00100000
    uv run scripts/serve_s1.py --checkpoint <ckpt> --no-rtc
    uv run scripts/serve_s1.py --checkpoint <ckpt> --host 0.0.0.0   # LAN (warns)
"""

from __future__ import annotations

import dataclasses
import logging

import tyro

from flore.s1_policy.serving.server import build_server, load_dpflow_chunk_policy, make_servicer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ServeArgs:
    """CLI args for the S1 policy server."""

    # Checkpoint directory (contains model.safetensors + config.json [+ norm_stats.json]).
    checkpoint: str
    # gRPC bind port.
    port: int = 50051
    # Bind host. Loopback by DEFAULT (security rule); set 0.0.0.0 for LAN deploys (logs a warning).
    host: str = "127.0.0.1"
    # Enable Real-Time Chunking (async background inference). --no-rtc for synchronous open-loop.
    rtc: bool = True
    # RTC: minimum control steps between the starts of two background inferences.
    rtc_s_min: int = 10
    # RTC: estimated inference latency in control steps (frozen-prefix length).
    rtc_delay_estimate: int = 4
    # RTC: naive async mode (disable the frozen-prefix skip logic).
    rtc_naive_async: bool = False
    # Number of flow-matching Euler denoising steps at inference.
    num_denoising_steps: int = 10
    # PyTorch device.
    device: str = "cuda"


def main(args: ServeArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    logger.info("Loading DP-Flow checkpoint from %s ...", args.checkpoint)
    chunk_policy, info = load_dpflow_chunk_policy(
        args.checkpoint,
        num_denoising_steps=args.num_denoising_steps,
        device=args.device,
    )

    servicer = make_servicer(
        chunk_policy,
        rtc=args.rtc,
        s_min=args.rtc_s_min,
        delay_estimate=args.rtc_delay_estimate,
        naive_async=args.rtc_naive_async,
        action_dim=info["action_dim"],
        action_horizon=info["action_horizon"],
        checkpoint=args.checkpoint,
    )
    server, bound_port = build_server(servicer, host=args.host, port=args.port)
    logger.info(
        "S1 server ready on %s:%d (rtc=%s, s_min=%d, delay_estimate=%d, action_dim=%d, horizon=%d).",
        args.host,
        bound_port,
        args.rtc,
        args.rtc_s_min,
        args.rtc_delay_estimate,
        info["action_dim"],
        info["action_horizon"],
    )
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down S1 server ...")
        server.stop(grace=5)


if __name__ == "__main__":
    main(tyro.cli(ServeArgs))
