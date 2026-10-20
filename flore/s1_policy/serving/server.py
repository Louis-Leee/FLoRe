"""gRPC server for the FLoRe S1 DP-Flow policy.

Simplified from the rfm_heavy ``server.py`` (1371 lines). Design rationale kept
from that server:

  * ``ThreadPoolExecutor(max_workers=1)`` — GPU inference is inherently
    sequential, so a single worker serializes all RPCs and no per-request lock
    is needed. The RTC background inference thread is the only other thread that
    touches the model, and its handoff is guarded by the RTC Condition.
  * Bind ``127.0.0.1`` by DEFAULT (repo security rule: never 0.0.0.0 by
    default). ``--host`` opts into a LAN bind and logs a warning.

What is DROPPED vs rfm_heavy: LoadPolicy (this server hosts exactly one
checkpoint chosen at startup), multi-backend dispatch, the N-env
``camera_<C>_env<E>`` batching, embodiment configs, the REST adapter, and
replay-episode debug mode.

The real model is wired via :func:`load_dpflow_chunk_policy`, which is guarded
so unit tests can inject a fake ``ChunkPolicy`` through :func:`make_servicer`
without importing torch or downloading a checkpoint.
"""

from __future__ import annotations

import logging
import time
from concurrent import futures
from typing import Any

import grpc
import numpy as np

from flore.s1_policy.rtc import ChunkPolicy, Observation, Policy, RealTimeChunkingPolicy
from flore.s1_policy.serving.proto import flore_policy_pb2 as pb2
from flore.s1_policy.serving.proto import flore_policy_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)

# 64 MB channel limits — a single 224x224x3 JPEG is a few tens of KB, so this is
# generous. Kept for parity with the FAR-pi client options.
_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", 64 * 1024 * 1024),
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
]


# ---------------------------------------------------------------------------
# JPEG image codec (single camera).
# ---------------------------------------------------------------------------
def decode_image_jpeg(data: bytes) -> np.ndarray:
    """Decode JPEG bytes to an ``(H, W, 3)`` uint8 RGB array."""
    import cv2

    buf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG image bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_image_jpeg(rgb: np.ndarray, quality: int = 95) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 RGB array to JPEG bytes. Used by clients/tests."""
    import cv2

    bgr = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to JPEG-encode image")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Non-RTC synchronous drip-feed wrapper (used when --no-rtc).
# ---------------------------------------------------------------------------
class SyncChunkDripPolicy(Policy):
    """Predict a chunk synchronously and hand out one step per ``get_action`` call.

    Open-loop: re-predicts only when the current chunk is exhausted. Provides the
    same ``get_action`` interface as :class:`RealTimeChunkingPolicy` so the
    servicer's Predict path is transport-uniform whether or not RTC is enabled.
    """

    def __init__(self, policy: ChunkPolicy) -> None:
        self._policy = policy
        self._chunk: np.ndarray | None = None
        self._idx: int = 0
        self._obs_counter: int = 0

    def get_action(
        self,
        images: np.ndarray,
        state: np.ndarray,
        prompt: str | None = None,
        timeout_secs: float = 3.0,
    ) -> tuple[np.ndarray, int]:
        self._obs_counter += 1
        if self._chunk is None or self._idx >= len(self._chunk):
            self._chunk = np.asarray(self._policy.predict_chunk(images[None], state[None], prompt)[0])
            self._idx = 0
        action = self._chunk[self._idx]
        self._idx += 1
        return action, self._obs_counter

    def act(self, state: np.ndarray, images: np.ndarray, prompt: str | None = None, **kwargs: Any) -> np.ndarray:
        action, _ = self.get_action(images[0], state[0], prompt)
        return action[None]

    def reset(self) -> None:
        self._chunk = None
        self._idx = 0
        self._obs_counter = 0
        self._policy.reset()

    def reset_context(self) -> None:
        self.reset()

    @property
    def config(self) -> dict[str, Any]:
        return getattr(self._policy, "config", {})

    @property
    def image_res(self) -> int:
        return getattr(self._policy, "image_res", 0)

    @property
    def history_len(self) -> int:
        return getattr(self._policy, "history_len", 0)


# ---------------------------------------------------------------------------
# gRPC servicer.
# ---------------------------------------------------------------------------
class FlorePolicyServicer(pb2_grpc.FlorePolicyServiceServicer):
    """Serves Predict / PredictChunk / Reset / GetPolicyInfo for one policy.

    Thread-safety relies on the server's ``max_workers=1`` executor serializing
    RPCs; the only other thread is RTC's background inference thread, whose
    handoff is internally synchronized.
    """

    def __init__(
        self,
        chunk_policy: ChunkPolicy,
        drip_policy: Policy,
        *,
        action_dim: int,
        action_horizon: int,
        checkpoint: str = "",
        rtc_enabled: bool = True,
        s_min: int = 10,
        delay_estimate: int = 4,
        act_timeout_secs: float = 600.0,
    ) -> None:
        self._chunk_policy = chunk_policy
        self._drip_policy = drip_policy
        self._action_dim = action_dim
        self._action_horizon = action_horizon
        self._checkpoint = checkpoint
        self._rtc_enabled = rtc_enabled
        self._s_min = s_min
        self._delay_estimate = delay_estimate
        self._act_timeout_secs = act_timeout_secs

    @staticmethod
    def _decode_obs(request: Any) -> tuple[np.ndarray, np.ndarray, str | None]:
        """Decode a PredictRequest into (state[state_dim], images[1,H,W,3], prompt)."""
        state = np.asarray(list(request.state), dtype=np.float32)
        rgb = decode_image_jpeg(request.image)  # (H, W, 3) uint8
        images = rgb[None]  # single camera -> (num_cameras=1, H, W, 3)
        prompt = request.prompt or None
        return state, images, prompt

    def Predict(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Drip-feed one action step from the (RTC or sync) scheduler."""
        # Validation aborts are done OUTSIDE the inference try/except: context.abort
        # raises to unwind the RPC, so catching it below would re-wrap it as INTERNAL.
        try:
            state, images, prompt = self._decode_obs(request)
        except Exception as e:  # noqa: BLE001 — a malformed request is the client's fault
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Malformed request: {e}")
        if state.size == 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "state must be non-empty")
        try:
            _t0 = time.perf_counter()
            action, _obs_idx = self._drip_policy.get_action(images, state, prompt, timeout_secs=self._act_timeout_secs)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            logger.debug("Predict: %.1fms", (time.perf_counter() - _t0) * 1000)
            return pb2.PredictResponse(action=action.tolist())
        except Exception as e:  # noqa: BLE001 — surface as INTERNAL to the client
            logger.exception("Predict failed")
            context.abort(grpc.StatusCode.INTERNAL, f"Predict error: {e}")

    def PredictChunk(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Return the whole predicted chunk in one call (stateless debug path)."""
        try:
            state, images, prompt = self._decode_obs(request)
        except Exception as e:  # noqa: BLE001 — malformed request is the client's fault
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Malformed request: {e}")
        if state.size == 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "state must be non-empty")
        try:
            _t0 = time.perf_counter()
            chunk = np.asarray(
                self._chunk_policy.predict_chunk(images[None], state[None], prompt)[0],
                dtype=np.float32,
            )
            if chunk.ndim != 2:
                raise ValueError(f"Expected chunk shape (H, D), got {chunk.shape}")
            horizon, action_dim = chunk.shape
            logger.debug("PredictChunk: %.1fms H=%d D=%d", (time.perf_counter() - _t0) * 1000, horizon, action_dim)
            return pb2.ChunkResponse(
                actions=chunk.reshape(-1).tolist(),
                horizon=horizon,
                action_dim=action_dim,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("PredictChunk failed")
            context.abort(grpc.StatusCode.INTERNAL, f"PredictChunk error: {e}")

    def Reset(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Clear per-episode RTC/drip state."""
        try:
            self._drip_policy.reset_context()
            return pb2.ResetResponse(ok=True)
        except Exception as e:  # noqa: BLE001
            logger.exception("Reset failed")
            context.abort(grpc.StatusCode.INTERNAL, f"Reset error: {e}")
            return pb2.ResetResponse(ok=False)

    def GetPolicyInfo(self, request: Any, context: grpc.ServicerContext) -> Any:
        return pb2.PolicyInfoResponse(
            action_dim=self._action_dim,
            action_horizon=self._action_horizon,
            checkpoint=self._checkpoint,
            rtc_enabled=self._rtc_enabled,
            s_min=self._s_min,
            delay_estimate=self._delay_estimate,
        )


# ---------------------------------------------------------------------------
# Servicer + server construction.
# ---------------------------------------------------------------------------
def make_servicer(
    chunk_policy: ChunkPolicy,
    *,
    rtc: bool = True,
    s_min: int = 10,
    delay_estimate: int = 4,
    naive_async: bool = False,
    action_dim: int = 32,
    action_horizon: int = 50,
    checkpoint: str = "",
    act_timeout_secs: float = 600.0,
) -> FlorePolicyServicer:
    """Build a servicer around a ``ChunkPolicy`` (real adapter or an injected fake).

    Wraps ``chunk_policy`` in :class:`RealTimeChunkingPolicy` when ``rtc`` is set,
    else in :class:`SyncChunkDripPolicy`. Tests pass a fake ``chunk_policy`` here
    so the server can be exercised without a model or GPU.
    """
    drip_policy: Policy
    if rtc:
        drip_policy = RealTimeChunkingPolicy(
            chunk_policy,
            s_min=s_min,
            delay_estimate=delay_estimate,
            naive_async=naive_async,
        )
    else:
        drip_policy = SyncChunkDripPolicy(chunk_policy)

    return FlorePolicyServicer(
        chunk_policy,
        drip_policy,
        action_dim=action_dim,
        action_horizon=action_horizon,
        checkpoint=checkpoint,
        rtc_enabled=rtc,
        s_min=s_min,
        delay_estimate=delay_estimate,
        act_timeout_secs=act_timeout_secs,
    )


def build_server(
    servicer: FlorePolicyServicer,
    *,
    host: str = "127.0.0.1",
    port: int = 50051,
    max_workers: int = 1,
) -> tuple[grpc.Server, int]:
    """Create and start a gRPC server for ``servicer``. Returns (server, bound_port).

    Pass ``port=0`` to let the OS pick a free port (tests use this). ``host``
    defaults to loopback; a non-loopback bind logs a warning (repo security rule).
    """
    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "Binding gRPC server to non-loopback host %r — this exposes the policy on the LAN. "
            "Ensure this is intended (default is 127.0.0.1).",
            host,
        )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=_GRPC_OPTIONS)
    pb2_grpc.add_FlorePolicyServiceServicer_to_server(servicer, server)
    bound_port = server.add_insecure_port(f"{host}:{port}")  # TODO: TLS for cross-network deploys
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind gRPC server to {host}:{port}")
    server.start()
    logger.info("FlorePolicyService listening on %s:%d (rtc=%s)", host, bound_port, servicer._rtc_enabled)
    return server, bound_port


# ---------------------------------------------------------------------------
# Real model adapter (guarded — not imported by tests).
# ---------------------------------------------------------------------------
class DPFlowChunkPolicy:
    """Adapts a FLoRe ``DPFlowModel`` to the :class:`ChunkPolicy` protocol.

    normalize state in / unnormalize actions out, image -> model tensor, and
    forward ``steps_executed`` / ``delay_estimate`` into ``sample_actions``.

    DEVIATION from the task brief: the brief described
    ``DPFlowModel.sample_actions(images, state, num_steps, prev_actions=...,
    steps_executed=..., delay_estimate=...)``. The model that actually landed
    (``flore/s1_policy/model/base.py::FlowMatchingVLABase.sample_actions``) takes
    a ``batch_dict`` and OWNS the RTC frozen prefix INTERNALLY via
    ``self._last_actions`` — it does NOT accept ``prev_actions``. This adapter
    therefore forwards ``steps_executed`` / ``delay_estimate`` and lets the model
    manage its own prefix; the ``prev_actions`` handed in by the RTC scheduler is
    accepted for protocol compatibility but not forwarded. See the report / TODO:
    confirm reset-during-inference cannot leave the model's ``_last_actions``
    polluted after RTC drops a stale chunk.
    """

    def __init__(
        self,
        model: Any,
        *,
        num_denoising_steps: int = 10,
        image_mean: float = 0.5,
        image_std: float = 0.5,
        norm_stats: dict[str, Any] | None = None,
        state_key: str = "state",
        action_key: str = "actions",
        device: str = "cuda",
        default_prompt: str = "",
        action_dim: int = 32,
        action_horizon: int = 50,
        image_res: int = 224,
    ) -> None:
        self._model = model
        self._num_denoising_steps = num_denoising_steps
        self._image_mean = image_mean
        self._image_std = image_std
        self._norm_stats = norm_stats or {}
        self._state_key = state_key
        self._action_key = action_key
        self._device = device
        self._default_prompt = default_prompt
        self._action_dim = action_dim
        self._action_horizon = action_horizon
        # Policy-protocol passthrough attributes (RTC delegates to these).
        self.image_res = image_res
        self.history_len = 0
        self.config = {
            "num_denoising_steps": num_denoising_steps,
            "action_dim": action_dim,
            "action_horizon": action_horizon,
        }

    def _images_to_tensor(self, images: np.ndarray) -> list:
        """(B, num_cameras, H, W, C) uint8/float -> list of (B, C, H, W) float tensors."""
        import torch

        arr = np.asarray(images)
        if arr.ndim != 5:
            raise ValueError(f"Expected images (B, num_cameras, H, W, C), got {arr.shape}")
        arr = arr.astype(np.float32) / 255.0
        arr = (arr - self._image_mean) / self._image_std
        # (B, num_cameras, H, W, C) -> (B, num_cameras, C, H, W)
        arr = np.transpose(arr, (0, 1, 4, 2, 3))
        num_cameras = arr.shape[1]
        return [torch.from_numpy(np.ascontiguousarray(arr[:, c])).to(self._device) for c in range(num_cameras)]

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        from flore.s1_policy.data.norm_stats import normalize_quantile

        if self._state_key in self._norm_stats:
            return normalize_quantile(state, self._norm_stats[self._state_key])
        return state

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        from flore.s1_policy.data.norm_stats import unnormalize_quantile

        if self._action_key in self._norm_stats:
            return unnormalize_quantile(actions, self._norm_stats[self._action_key])
        return actions

    def _chunk_to_absolute(self, chunk: np.ndarray, anchor_state: np.ndarray) -> np.ndarray:
        """Un-relativize an anchor-relative chunk to ABSOLUTE pelvis-frame poses.

        CRITICAL train/deploy contract: training (umi_dataset.__getitem__) makes
        EVERY row of the chunk relative to the SINGLE anchor pose (the obs frame),
        via absolute_to_delta_ee(chunk, ref=anchor). So the model's output is
        anchor-relative — NOT step-to-step incremental. The un-relativize MUST
        happen HERE, on the server, because only the server knows the chunk's
        true anchor (the obs `state` captured when this inference started). The
        drip-fed client can't see chunk boundaries and would compound the deltas
        (runaway motion) if it integrated them sequentially.

        anchor_state is the UNNORMALIZED 32-D obs pose (pos3+rot6d per keypoint).
        Grip dims and pad dims pass through unchanged.
        """
        from flore.s1_policy.data.transforms import delta_ee_to_absolute

        # (pos_slice, rot6d_slice) per keypoint — MUST match umi_dataset's layout.
        specs = [(slice(0, 3), slice(3, 9)), (slice(10, 13), slice(13, 19)), (slice(20, 23), slice(23, 29))]
        out = chunk.copy()  # (B, H, AD); grip + pad dims pass through
        for b in range(chunk.shape[0]):
            for pos_sl, rot_sl in specs:
                abs_pos, abs_rot6d = delta_ee_to_absolute(
                    rel_pos=chunk[b, :, pos_sl],          # (H, 3)
                    rel_rot6d=chunk[b, :, rot_sl],        # (H, 6)
                    ref_pos=anchor_state[pos_sl],         # (3,)
                    ref_rot6d=anchor_state[rot_sl],       # (6,)
                )
                out[b, :, pos_sl] = abs_pos
                out[b, :, rot_sl] = abs_rot6d
        return out

    def predict_chunk(
        self,
        images: np.ndarray,
        state: np.ndarray,
        prompt: str | None = None,
        steps_executed: int = 0,
        delay_estimate: int = 0,
        prev_actions: np.ndarray | None = None,
    ) -> np.ndarray:
        import torch

        state = np.asarray(state, dtype=np.float32)
        norm_state = self._normalize_state(state).astype(np.float32)
        batch_dict = {
            "images": self._images_to_tensor(images),
            "state": torch.from_numpy(norm_state).to(self._device),
        }
        with torch.no_grad():
            out = self._model.sample_actions(
                batch_dict,
                num_steps=self._num_denoising_steps,
                steps_executed=steps_executed,
                delay_estimate=delay_estimate,
            )
        actions = out.detach().float().cpu().numpy()  # (B, H, AD) normalized
        actions = self._unnormalize_actions(actions)  # anchor-relative delta-EE
        # Un-relativize to ABSOLUTE targets against THIS chunk's anchor (the obs
        # state), so the client can forward them statelessly (no compounding).
        # _chunk_to_absolute indexes the anchor per-keypoint (anchor[slice(3, 9)] etc.), so
        # it needs a 1-D (AD,) vector. Real callers pass state as (1, AD) via state[None]
        # (the model batch dim); flatten to (AD,) here so the per-keypoint slicing is
        # correct — passing (1, AD) would slice axis 0 and raise inside delta_ee_to_absolute.
        anchor = np.asarray(state, dtype=np.float32).reshape(-1)
        actions = self._chunk_to_absolute(actions, anchor)
        return actions.astype(np.float32)

    def reset(self) -> None:
        self._model.reset()


def load_dpflow_chunk_policy(
    checkpoint: str,
    *,
    num_denoising_steps: int | None = None,
    device: str = "cuda",
) -> tuple[DPFlowChunkPolicy, dict[str, Any]]:
    """Load a DP-Flow checkpoint into a :class:`DPFlowChunkPolicy`.

    Guarded (torch + model imports are deferred here) so tests that inject a fake
    ``ChunkPolicy`` never pay for them. Reads ``config.json`` (full ``TrainConfig``
    with a nested ``model`` dict), ``model.safetensors``, and optional
    ``norm_stats.json`` from the checkpoint dir.

    Returns (chunk_policy, info) where ``info`` carries action_dim / action_horizon
    for the servicer's GetPolicyInfo.
    """
    import dataclasses
    import json
    from pathlib import Path

    from flore.s1_policy.data.norm_stats import load_norm_stats
    from flore.s1_policy.model.dp_flow import DPFlowConfig, DPFlowModel

    ckpt_dir = Path(checkpoint)
    raw = json.loads((ckpt_dir / "config.json").read_text())

    # ALGO GUARD — must come FIRST. This loader builds a DPFlowModel and reads the model
    # config from raw["model"]. For an algo='diffusion' run TrainConfig serializes the
    # TRAINED config under raw["diffusion"] and leaves raw["model"] at flow DEFAULTS
    # (layout_mode='', force_hist_channels=0, obs_history_len=1, action_dim=32). So every
    # HL/history/force guard below reads those defaults and PASSES, and the loader would
    # then build a default 32-dim flow model and load_state_dict(strict=False) a 16-dim
    # diffusion checkpoint — silently returning a garbage model (mismatched keys logged as
    # warnings, GetPolicyInfo reporting the wrong action_dim) instead of refusing. Serving
    # implements only the flow DP model, so refuse a diffusion checkpoint outright.
    algo = str(raw.get("algo", "flow") or "flow")
    if algo != "flow":
        raise NotImplementedError(
            f"checkpoint was trained with algo={algo!r}, but serving implements only the "
            f"flow-matching model (DPFlowModel). Its real config lives under the "
            f"'{algo}' block of config.json, which this loader does not read. Serve a "
            f"flow checkpoint, or wire serving for the {algo} model first.")

    # config.json is the full TrainConfig; the model config is nested under "model".
    model_cfg_raw = raw.get("model", raw)
    data_cfg_raw = raw.get("data", {})

    # GUARDRAIL: serving currently supports ONLY the per_keypoint frame + single-frame
    # observation. A model trained with a shared reference frame (ref_frame in
    # left/right/head/world) or an observation history (obs_history_len > 1) would be
    # served WRONG here — _chunk_to_absolute un-relativizes every keypoint against its
    # OWN anchor (hardcoded per_keypoint), which is SILENTLY incorrect for a shared
    # frame, and predict_chunk builds a single-frame batch, which mis-sizes global_cond
    # for a history model. Fail loudly at load rather than emit corrupt robot targets.
    # TODO: wire serving to transforms.resolve_base (shared frame) + a To-frame obs
    # buffer, then relax this guard.
    ref_frame = str(data_cfg_raw.get("ref_frame", "per_keypoint"))
    if ref_frame != "per_keypoint":
        raise NotImplementedError(
            f"checkpoint was trained with data.ref_frame={ref_frame!r}, but serving only "
            f"supports 'per_keypoint' today (_chunk_to_absolute hardcodes per-keypoint "
            f"un-relativization). Deploying this model would SILENTLY produce wrong "
            f"absolute poses. Wire serving to transforms.resolve_base first.")
    # An HL-v2 checkpoint (non-empty model.layout_mode, or a force encoder) uses a
    # fundamentally different observation and action layout than _chunk_to_absolute /
    # predict_chunk implement (16-dim action blocks, wrt_start_rot6d state, force-history
    # conditioning, the t0-inclusive grid). Serving is NOT yet wired for it, so refuse at
    # load rather than deploy a model that would crash opaquely or emit wrong targets.
    layout_mode = str(model_cfg_raw.get("layout_mode", "") or "")
    if layout_mode or int(model_cfg_raw.get("force_hist_channels", 0) or 0) > 0:
        raise NotImplementedError(
            f"checkpoint is an HL-v2 model (layout_mode={layout_mode!r}, "
            f"force_hist_channels={model_cfg_raw.get('force_hist_channels', 0)}), but "
            f"serving still implements the pre-v2 32-dim layout (_chunk_to_absolute "
            f"hardcodes it, predict_chunk builds no force_history / wrt_start state). "
            f"Wire serving for the HL contract before deploying this checkpoint.")
    # obs_history_len is authoritative under MODEL: TrainConfig pushes it there for BOTH
    # pipelines (_post_init_hl / _post_init_legacy), while HLDataConfig has no such field
    # so config.json's `data` block lacks it — reading `data` first would resolve to the
    # default 1 and silently pass an HL history model. Model-first is correct everywhere.
    obs_history_len = int(model_cfg_raw.get("obs_history_len", data_cfg_raw.get("obs_history_len", 1)))
    if obs_history_len > 1:
        raise NotImplementedError(
            f"checkpoint was trained with obs_history_len={obs_history_len}, but serving "
            f"builds a single-frame observation (predict_chunk) and would mis-size "
            f"global_cond. Add a To-frame observation buffer to serving first.")

    field_types = {f.name: f.type for f in dataclasses.fields(DPFlowConfig)}
    kwargs: dict[str, Any] = {}
    for k, v in model_cfg_raw.items():
        if k not in field_types:
            continue
        # JSON turns tuples into lists; coerce tuple-typed fields back.
        if isinstance(v, list):
            v = tuple(v)
        kwargs[k] = v
    cfg = DPFlowConfig(**kwargs)

    model = DPFlowModel(cfg)
    model.load_checkpoint(str(ckpt_dir))
    model.to(device).eval()

    norm_stats = None
    ns_path = ckpt_dir / "norm_stats.json"
    if ns_path.exists():
        norm_stats = load_norm_stats(ns_path)

    # Pick the action-stats key present in the file (dataset naming isn't fixed).
    action_key = "actions"
    if norm_stats is not None:
        for candidate in ("actions", "action_chunk", "action"):
            if candidate in norm_stats:
                action_key = candidate
                break

    chunk_policy = DPFlowChunkPolicy(
        model,
        num_denoising_steps=num_denoising_steps or cfg.inference_steps,
        image_mean=float(data_cfg_raw.get("image_mean", 0.5)),
        image_std=float(data_cfg_raw.get("image_std", 0.5)),
        norm_stats=norm_stats,
        action_key=action_key,
        device=device,
        action_dim=cfg.action_dim,
        action_horizon=cfg.action_horizon,
    )
    info = {"action_dim": cfg.action_dim, "action_horizon": cfg.action_horizon}
    return chunk_policy, info


__all__ = [
    "FlorePolicyServicer",
    "SyncChunkDripPolicy",
    "DPFlowChunkPolicy",
    "make_servicer",
    "build_server",
    "load_dpflow_chunk_policy",
    "decode_image_jpeg",
    "encode_image_jpeg",
    "Observation",
]
