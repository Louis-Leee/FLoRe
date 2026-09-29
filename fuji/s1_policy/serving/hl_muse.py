"""``--algo muse``: the MuSe-on-UVA force policy behind the HL-v2 serving seams.

The fourth sibling of flow / diffusion / act, with ONE structural difference: the model
does not live in this interpreter. ``unified_video_action`` (MAR + a frozen KL-16 VAE +
hydra) only runs in the ``uvaforce`` env, and viser + mink + the rest of the HL-v2 stack
only run in FUJI's ``.venv``. So the model is a SUBPROCESS
(:mod:`scripts.muse_hl_infer_server`) reached over the numpy-only line protocol in
:mod:`fuji.utils.muse_wire`, and this module is the shim that makes it look
like a local policy to everything above it.

Why bother, when ``scripts/muse_force_serving_{server,viz}.py`` already deploys a UVA force
model: those put the WHOLE planner in the backend, so they carry their own second copy of the
wire mapper, the anchor bookkeeping and the viser page — including a stale ``wrist->TCP 0.21``
lever (today's unified deploy is 0.242) and none of the force gate / recorder / reach
compensation / ghost / joystick-stop work that landed in ``hl_serving.py`` since. Routing the
UVA model through the SAME planner instead means it inherits all of that, and a rig change is
made once. Those two scripts are untouched and keep working.

What crosses the pipe (and what does not)
-----------------------------------------
The frontend sends a FULLY RELATIVIZED observation — anchor-relative pose history, the
``wrt_start`` rotation channel, gripper metres, physical wrench — and gets back the chunk in
UVA's own ``(H, 10)`` relative form plus ``(H, 6)`` physical wrench. The backend therefore
knows nothing about pelvis frames, TCP levers or the ``:8770`` wire; and un-relativization
happens exactly once, in :func:`~fuji.s1_policy.serving.hl_online.decode_chunk`, the same
function every other algorithm (and recorded replay) goes through.

That is safe because the two relative conventions are the SAME operation:
``absolute_to_delta_ee`` computes ``R_rel = R_ref^T R_a`` / ``p_rel = R_ref^T (p_a - p_ref)``,
which is exactly UMI's ``convert_pose_mat_rep(..., 'relative')`` = ``T_ref^-1 T_a``; and both
spell rot6d as the first two ROWS of the rotation matrix (UMI's ``mat_to_pose10d`` slices
``mat[..., :2, :]``). ``tests/test_hl_muse.py`` pins that numerically rather than trusting
this paragraph.

Native-layout mapping (UVA ``(H,10)`` + ``(H,6)`` -> FUJI ``left_only`` ``(H,16)``)
    ``[0:3]`` pos_rel · ``[3:9]`` rot6d_rel · ``[9]`` gripper · ``[10:16]`` wrench
UVA's gripper channel is trained in METRES (``normalizer_type: none``) while the FUJI native
layout carries a FRACTION of :data:`~fuji.s1_policy.data.transforms.GRIPPER_WIDTH_REF_M`, so
it is divided here and multiplied back by ``decode_chunk`` — a round trip, not a rescale.

The action horizon must be PINNED
---------------------------------
UVA's chunk length is a checkpoint property (``model.policy.num_actions``, 32 for
``uva_hl_force``) and FUJI's is a module constant resolved at import time
(:data:`~fuji.s1_policy.data.layout.ACTION_HORIZON`, from ``FUJI_ACTION_HORIZON``). They must
agree or every horizon-derived thing downstream — the execute-N-of-S choices, the recorder's
grid, the offline target chunk — is sized for a different chunk than the model emits.
:func:`load_muse_model` refuses on a mismatch and names the env var; the launchers
(``run_hl_serving.sh``, ``run_infer_s1_muse.sh``) DERIVE the pin from the checkpoint, so
getting it wrong takes deliberate effort.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from fuji.s1_policy.data.layout import ACTION_DT, ACTION_HORIZON, FORCE_DIM, get_layout
from fuji.s1_policy.data.transforms import (
    GRIPPER_WIDTH_REF_M,
    absolute_to_delta_ee,
    rot6d_relative,
)
from fuji.s1_policy.serving.hl_online import (
    LivePlanner,
    LoadedHLModel,
    batch_from_obs,
    decode_chunk,
    sample_chunk,
)
from fuji.utils import muse_wire as wire

#: ``algo`` string this module serves. Not in the trainer's ``ALGO_CONFIG_FIELD`` on purpose:
#: nothing here is trained by ``scripts/train_s1.py``, so a muse entry there would advertise a
#: training path that does not exist.
MUSE_ALGO = "muse"

#: Extra batch keys :class:`MuseLivePlanner` adds on top of ``batch_from_obs``'s output.
#: The HL-v2 observation carries NO absolute position and expresses rotation relative to the
#: EPISODE START, while UVA wants both relative to the CURRENT anchor — so these two arrays
#: are the only things the standard batch cannot supply. Namespaced so a stray key can never
#: be mistaken for a model input by the local algorithms (which ignore unknown keys anyway).
MUSE_EEF_POS_REL_KEY = "muse_eef_pos_rel"
MUSE_EEF_ROT6D_REL_KEY = "muse_eef_rot6d_rel"

#: UVA action-vector width: pos3 + rot6d6 + gripper1 (LEFT hand only; the right hand and head
#: are not predicted, which is why this only ever serves the ``left_only`` layout).
UVA_ACTION_DIM = 10

#: Image normalization the FUJI ring is told to use for a muse checkpoint. 0.5/0.5 is not a
#: preference: UVA's ``process_data`` does ``x*255`` then ``x/127.5 - 1`` on a 0-1 tensor, which
#: is EXACTLY ``(x/255 - 0.5)/0.5`` — so with these values the ring already holds the model's
#: own image space and the wire conversion back to 0-1 is exact rather than lossy. They are
#: constants (not knobs) because both sides must agree and only one pair is correct.
MUSE_IMAGE_MEAN = 0.5
MUSE_IMAGE_STD = 0.5

#: How long to wait for the backend's ``ready`` handshake. The backend imports torch + hydra,
#: builds MAR, loads a ~750 MB slim checkpoint and restores the frozen VAE; measured ~40-70 s
#: cold on this workstation, so this is deliberately generous. A hang here is almost always a
#: missing pretrained tree or the wrong interpreter, both of which the backend says on stderr.
HANDSHAKE_TIMEOUT_S = 600.0

#: Per-inference timeout. A UVA sample is ~0.6-2 s; 60 s means "the backend is wedged or dead".
INFER_TIMEOUT_S = 60.0


def _log(msg: str) -> None:
    print(f"[hl-muse] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# The backend's self-description (the `ready` handshake)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MuseMeta:
    """Everything the frontend needs to know about the loaded UVA checkpoint.

    Read off the backend's ``ready`` message rather than re-derived here, because the
    authority is the checkpoint's own pickled hydra cfg and only the uvaforce interpreter can
    unpickle it. Every field below is a knob a retrained checkpoint can legitimately move.
    """

    num_actions: int  # chunk length the model emits (32 for uva_hl_force)
    action_dim: int  # 10
    force_dim: int  # 6
    history_len: int  # low-dim history-half length (== num_actions)
    img_indices: tuple[int, ...]  # positional slots inside the length-2H window
    n_img_cond: int  # conditioning frames the action head reads (first half)
    img_step_stride: int  # spacing of those frames, in action steps (4 => 133 ms)
    force_history_length: int  # steps of wrench the encoder keeps
    ft_scale: tuple[float, ...]  # (6,) dataset pre-scale; serving MUST match training
    obs_pose_repr: str
    action_pose_repr: str
    use_force: bool
    predict_force: bool
    n_action_steps: int
    down_sample_steps: int
    checkpoint: str
    random_init: bool
    #: The UVA action head's OWN sampler settings, carried for the page/CLI readout only.
    #: Fixed by the checkpoint (``autoregressive_model_params.num_sampling_steps`` /
    #: ``num_iter``) and deliberately NOT exposed as ``--num-steps``.
    num_sampling_steps: str | int | None = None
    num_iter: int | None = None
    epoch: int | None = None
    global_step: int | None = None
    zarr_path: str = ""
    episode_frame_nums: tuple[int, ...] = ()
    valid_anchor_min: int = 0
    valid_anchor_slack_end: int = 0
    #: Store-episode indices this checkpoint actually TRAINED on. UVA holds out by ratio
    #: (``used_episode_indices``), which is a different split from FUJI's last-N convention —
    #: carried so an offline tool can say whether a scrubbed episode was in-distribution
    #: instead of leaving a tighter-than-metric overlay to be mistaken for generalization.
    train_episodes: tuple[int, ...] = ()

    @classmethod
    def from_meta(cls, meta: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> "MuseMeta":
        def ints(name: str) -> tuple[int, ...]:
            if name not in arrays:
                return ()
            return tuple(int(v) for v in np.asarray(arrays[name]).reshape(-1))

        img_indices = ints("img_indices")
        n_cond = int(meta["n_img_cond"])
        cond = img_indices[:n_cond]
        # The conditioning frames must be EVENLY spaced, because the FUJI ring that supplies
        # them is a fixed-stride ingest (obs_history_dt_s), which cannot express an uneven
        # comb. uva_hl_force's [-12,-8,-4,0] is even; refuse anything else rather than serve
        # a history whose frames are silently at the wrong times.
        diffs = {cond[i + 1] - cond[i] for i in range(len(cond) - 1)}
        if len(cond) > 1 and len(diffs) != 1:
            raise ValueError(
                f"backend's conditioning image indices {cond} are not evenly spaced "
                f"(step diffs {sorted(diffs)}). The live ingest supplies a fixed-stride "
                "history, so an uneven comb cannot be reproduced at deploy time."
            )
        stride = diffs.pop() if diffs else 1
        if stride < 1:
            raise ValueError(f"conditioning image stride must be >= 1, got {stride}")
        return cls(
            num_actions=int(meta["num_actions"]),
            action_dim=int(meta["action_dim"]),
            force_dim=int(meta["force_dim"]),
            history_len=int(meta["history_len"]),
            img_indices=img_indices,
            n_img_cond=n_cond,
            img_step_stride=int(stride),
            force_history_length=int(meta["force_history_length"]),
            ft_scale=tuple(float(v) for v in np.asarray(arrays["ft_scale"]).reshape(-1)),
            obs_pose_repr=str(meta["obs_pose_repr"]),
            action_pose_repr=str(meta["action_pose_repr"]),
            use_force=bool(meta["use_force"]),
            predict_force=bool(meta["predict_force"]),
            n_action_steps=int(meta["n_action_steps"]),
            down_sample_steps=int(meta["down_sample_steps"]),
            checkpoint=str(meta.get("checkpoint", "")),
            random_init=bool(meta.get("random_init", False)),
            num_sampling_steps=meta.get("num_sampling_steps"),
            num_iter=meta.get("num_iter"),
            epoch=meta.get("epoch"),
            global_step=meta.get("global_step"),
            zarr_path=str(meta.get("zarr_path", "")),
            episode_frame_nums=ints("episode_frame_nums"),
            valid_anchor_min=int(meta.get("valid_anchor_min", 0)),
            valid_anchor_slack_end=int(meta.get("valid_anchor_slack_end", 0)),
            train_episodes=ints("train_episodes"),
        )

    @property
    def obs_history_dt_s(self) -> float:
        """Seconds between conditioning frames — ``img_step_stride`` action steps.

        DERIVED from the model's own comb rather than configured, so an ``hl_serving``
        launched against a checkpoint with a different image spacing gets the matching ingest
        stride automatically (that script multiplies this by ``--obs-hz``)."""
        return int(self.img_step_stride) * ACTION_DT


# ---------------------------------------------------------------------------
# The backend subprocess
# ---------------------------------------------------------------------------
class MuseBackend:
    """A live ``scripts/muse_hl_infer_server.py`` child, plus its handshake.

    Owns exactly one request at a time (guarded by a lock): the protocol is a strict
    request/reply over one pipe, and the planner only ever infers from one thread — but
    ``hl_serving``'s viser callbacks run on several, and an interleaved write would splice two
    requests into one unparseable line.

    Backend stderr is drained by a daemon thread and echoed with a prefix. That is not
    cosmetic: the pipe has a finite buffer, and hydra + MAR + the VAE print enough that a
    backend nobody reads from BLOCKS on write and looks exactly like a hung model.
    """

    def __init__(
        self,
        backend_python: str | Path,
        *,
        checkpoint: str | Path | None,
        device: str = "cuda",
        zarr_path: str | Path | None = None,
        dataset_root: str | Path | None = None,
        force_fusion: str | None = None,
        config_name: str = "uva_hl_force",
        server_script: str | Path | None = None,
        extra_args: Sequence[str] = (),
    ) -> None:
        py = os.path.abspath(os.path.expanduser(str(backend_python)))
        if not os.access(py, os.X_OK):
            raise FileNotFoundError(
                f"MuSe backend interpreter {py!r} is not executable. That python must be the "
                "uvaforce env (torch 2.7.1+cu128 + hydra + the MAR/VAE trees) — FUJI's own "
                ".venv cannot import unified_video_action. Set BACKEND_PY."
            )
        script = Path(server_script) if server_script else (
            Path(__file__).resolve().parents[3] / "scripts" / "muse_hl_infer_server.py")
        if not script.is_file():
            raise FileNotFoundError(f"MuSe backend script not found: {script}")

        argv = [py, str(script), "--device", str(device), "--config-name", str(config_name)]
        if checkpoint:
            argv += ["--checkpoint", os.path.abspath(os.path.expanduser(str(checkpoint)))]
        if zarr_path:
            argv += ["--zarr-path", os.path.abspath(os.path.expanduser(str(zarr_path)))]
        if dataset_root:
            argv += ["--dataset-root", os.path.abspath(os.path.expanduser(str(dataset_root)))]
        if force_fusion:
            argv += ["--force-fusion", str(force_fusion)]
        argv += list(extra_args)

        _log("spawning backend: " + " ".join(argv))
        self._proc = subprocess.Popen(  # noqa: S603 - argv is built here, not from a shell
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()
        self._closed = False
        self._stderr_tail: list[str] = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="muse-backend-stderr", daemon=True)
        self._stderr_thread.start()
        self.meta = self._handshake()

    # --- plumbing ---------------------------------------------------------
    def _drain_stderr(self) -> None:
        stream = self._proc.stderr
        if stream is None:
            return
        for line in stream:
            text = line.rstrip()
            # Keep a short tail so a death can be REPORTED. Without it the frontend can only
            # say "backend exited", and the actual cause (a missing kl16.ckpt, a CUDA OOM)
            # has already scrolled past in a terminal nobody was watching.
            self._stderr_tail.append(text)
            del self._stderr_tail[:-40]
            print(f"[muse-backend] {text}", file=sys.stderr, flush=True)

    def _died(self) -> str:
        tail = "\n  ".join(self._stderr_tail[-12:]) or "(no stderr captured)"
        return (f"MuSe backend exited with code {self._proc.poll()}. Last stderr:\n  {tail}")

    def _read_message(self, timeout_s: float) -> tuple[str, dict, dict[str, np.ndarray]]:
        """Block for the next PROTOCOL line, echoing non-protocol stdout as chatter."""
        stream = self._proc.stdout
        if stream is None:
            raise RuntimeError("MuSe backend has no stdout pipe")
        deadline = time.monotonic() + float(timeout_s)
        while True:
            if self._proc.poll() is not None:
                raise RuntimeError(self._died())
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"MuSe backend produced no reply within {timeout_s:g}s. {self._died()}"
                    if self._proc.poll() is not None else
                    f"MuSe backend produced no reply within {timeout_s:g}s (still running).")
            line = stream.readline()
            if not line:  # EOF
                raise RuntimeError(self._died())
            if not wire.is_message(line):
                text = line.rstrip()
                if text:
                    print(f"[muse-backend] {text}", file=sys.stderr, flush=True)
                continue
            return wire.unpack(line)

    def _handshake(self) -> MuseMeta:
        tag, meta, arrays = self._read_message(HANDSHAKE_TIMEOUT_S)
        if tag == wire.TAG_ERROR:
            raise RuntimeError(f"MuSe backend failed to start: {meta.get('message')}")
        if tag != wire.TAG_READY:
            raise RuntimeError(f"expected a {wire.TAG_READY!r} handshake, got {tag!r}")
        resolved = MuseMeta.from_meta(meta, arrays)
        _log(
            f"backend ready: H={resolved.num_actions} act_dim={resolved.action_dim} "
            f"img_cond={resolved.n_img_cond}@stride{resolved.img_step_stride} "
            f"force_hist={resolved.force_history_length} ft_scale={list(resolved.ft_scale)} "
            f"use_force={resolved.use_force} predict_force={resolved.predict_force} "
            f"ckpt={resolved.checkpoint or '<random-init>'}"
        )
        return resolved

    def request(
        self, meta: dict, *, timeout_s: float = INFER_TIMEOUT_S, **arrays: np.ndarray
    ) -> tuple[dict, dict[str, np.ndarray]]:
        """Send one request, block for its reply. Raises on a backend-reported error."""
        if self._closed:
            raise RuntimeError("MuSe backend is closed")
        line = wire.pack("", meta, **arrays)
        with self._lock:
            stdin = self._proc.stdin
            if stdin is None:
                raise RuntimeError("MuSe backend has no stdin pipe")
            try:
                stdin.write(line + "\n")
                stdin.flush()
            except (BrokenPipeError, ValueError) as exc:
                raise RuntimeError(self._died()) from exc
            tag, reply_meta, reply_arrays = self._read_message(timeout_s)
        if tag == wire.TAG_ERROR:
            raise RuntimeError(f"MuSe backend refused the request: {reply_meta.get('message')}")
        if tag != wire.TAG_CHUNK:
            raise RuntimeError(f"expected a {wire.TAG_CHUNK!r} reply, got {tag!r}")
        return reply_meta, reply_arrays

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass
        try:
            self._proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()

    def __enter__(self) -> "MuseBackend":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# The local model shim + its stand-in config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MuseConfig:
    """The subset of a model config the HL-v2 serving stack reads, for a remote model.

    ``LoadedHLModel.config`` is consumed by name — ``action_horizon`` in the layout checks,
    ``temporal_ensemble``/``temporal_ensemble_k`` by :meth:`LivePlanner._ensembler`,
    ``force_hist_len`` by the loader — so a UVA checkpoint needs an object that answers those.
    ``temporal_ensemble`` is False because UVA has no per-step re-query path here (a chunk
    costs ~0.6-2 s; blending at 30 Hz is not on the table), and the viser checkbox correctly
    shows it as unavailable.
    """

    action_horizon: int
    force_hist_len: int
    action_dim: int
    state_dim: int
    #: UVA's own diffusion-head step counts, carried for the page's readout only. They are
    #: fixed by the checkpoint (``num_sampling_steps`` / ``num_iter``) and NOT exposed as
    #: ``--num-steps``: a knob that silently did nothing would be worse than its absence
    #: (same call as the ACT CLI's missing ``--num-steps``).
    num_sampling_steps: str | int | None = None
    num_iter: int | None = None
    temporal_ensemble: bool = False
    temporal_ensemble_k: float = 0.01


class MuseRemoteModel:
    """A ``sample_actions``-shaped facade over :class:`MuseBackend`.

    Deliberately NOT an ``nn.Module``: there are no local parameters, and pretending
    otherwise invites a ``.to(device)`` that would silently do nothing. ``reset`` / ``to`` /
    ``eval`` exist because the shared serving seams call them, and they are honest no-ops.

    The point of implementing this interface rather than bypassing it is that
    :func:`~fuji.s1_policy.serving.hl_online.sample_chunk` then works unchanged, so the muse
    path inherits its seed handling, its (no-op, empty-stats) unnormalization and — the one
    that matters at a robot — its ``FloatingPointError`` on a non-finite chunk.
    """

    def __init__(self, backend: MuseBackend, meta: MuseMeta, layout) -> None:
        self.backend = backend
        self.meta = meta
        self.layout = layout

    # --- no-op nn.Module surface -----------------------------------------
    def reset(self) -> None:
        """No-op: the backend holds no per-episode state (each request is self-contained)."""

    def to(self, *_a, **_k) -> "MuseRemoteModel":
        return self

    def eval(self) -> "MuseRemoteModel":
        return self

    # --- the real entry point --------------------------------------------
    def sample_actions(
        self, batch: Mapping[str, Any], *, num_steps: int = 1, generator=None
    ) -> torch.Tensor:
        """One UVA rollout from a FUJI batch -> ``(1, H, action_dim)`` in the native layout.

        ``batch`` is ``batch_from_obs``'s output PLUS :data:`MUSE_EEF_POS_REL_KEY` and
        :data:`MUSE_EEF_ROT6D_REL_KEY` (see :class:`MuseLivePlanner`). Their absence is a
        wiring error, not a runtime condition — most visibly it is what a plain
        ``scripts/reinfer_hl_take.py`` run would hit, because a recorded take stores the HL-v2
        observation, which has no absolute position in it. Say so instead of guessing zeros:
        a zero-motion pose history is a PLAUSIBLE observation, so a fabricated one would
        produce a confident, subtly wrong chunk.
        """
        for key in (MUSE_EEF_POS_REL_KEY, MUSE_EEF_ROT6D_REL_KEY):
            if key not in batch:
                raise KeyError(
                    f"muse inference needs {key!r} in the batch. The HL-v2 observation carries "
                    "no absolute EE position and expresses rotation relative to the EPISODE "
                    "START, so the anchor-relative pose history cannot be reconstructed from "
                    "it — MuseLivePlanner adds both keys. A stored deploy take does not have "
                    "them, so re-inferring one with --algo muse is not supported yet."
                )
        if int(num_steps) != 1:
            raise ValueError(
                f"the UVA action head has no exposed sampler loop; num_steps must be 1, got "
                f"{num_steps}. Its own counts are fixed by the checkpoint "
                f"(num_sampling_steps={self.meta.num_sampling_steps}, "
                f"num_iter={self.meta.num_iter}).")

        seed = 0 if generator is None else int(generator.initial_seed())
        # Back to the 0-1 CHW tensor UVA's `process_data` expects. Exact (not a re-quantized
        # uint8 round trip) because the ring was told to normalize with MUSE_IMAGE_MEAN/STD.
        images = np.clip(
            _images_from_batch(batch, self.meta.n_img_cond) * MUSE_IMAGE_STD + MUSE_IMAGE_MEAN,
            0.0, 1.0)
        state = _first_row_stack(batch["state"])  # (To, state_dim)
        force = _first_row_stack(batch["force_history"])  # (FH, ch)
        force_valid = bool(np.asarray(_to_numpy(batch["force_history_valid"])).reshape(-1)[0])

        slots = self.layout.state_slots()["left"]
        wrt_start = np.asarray(state[:, slots["wrt_start_rot6d"]], dtype=np.float64)
        grip_col = slots.get("gripper")
        grip_m = (
            np.asarray(state[:, grip_col], dtype=np.float64) * GRIPPER_WIDTH_REF_M
            if grip_col is not None else np.zeros(state.shape[0], dtype=np.float64)
        )

        reply_meta, arrays = self.backend.request(
            {"kind": wire.KIND_LIVE, "seed": int(seed), "force_valid": force_valid},
            images=images.astype(np.float32),
            eef_pos_rel=np.asarray(batch[MUSE_EEF_POS_REL_KEY], dtype=np.float64),
            eef_rot6d_rel=np.asarray(batch[MUSE_EEF_ROT6D_REL_KEY], dtype=np.float64),
            eef_rot6d_wrt_start=wrt_start,
            grip_m=grip_m,
            force=np.asarray(force, dtype=np.float64),
        )
        native = native_chunk_from_uva(
            arrays["action_rel"], arrays.get("force_pred"), self.layout,
            predict_force=self.meta.predict_force,
        )
        _ = reply_meta
        return torch.from_numpy(native).unsqueeze(0)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_row_stack(value) -> np.ndarray:
    """``(1, N, C)`` batch tensor -> ``(N, C)`` numpy, tolerating an already-squeezed ``(N, C)``.

    ``batch_from_obs`` always emits the batch axis, but ``HLDataset`` squeezes ``state`` to
    ``(state_dim,)`` when ``To == 1``, and the muse path can legitimately be handed either
    shape by a caller assembling a batch itself.
    """
    arr = np.asarray(_to_numpy(value), dtype=np.float64)
    if arr.ndim == 3:
        return arr[0]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr[None, :]
    raise ValueError(f"expected a (1,N,C)/(N,C) array, got shape {arr.shape}")


def _images_from_batch(batch: Mapping[str, Any], n_expected: int) -> np.ndarray:
    """``batch['images']`` (a list of ``(1,3,224,224)`` tensors, oldest-first) -> ``(To,3,224,224)``.

    Still in the FUJI ``[-1,1]`` normalized image space; the backend converts to the 0-1 CHW
    tensor UVA's ``process_data`` expects (which re-derives exactly the same ``x/127.5 - 1``).
    """
    imgs = batch["images"]
    if isinstance(imgs, (list, tuple)):
        stack = np.stack([np.asarray(_to_numpy(im)).reshape(3, 224, 224) for im in imgs])
    else:
        arr = np.asarray(_to_numpy(imgs))
        stack = arr[0] if arr.ndim == 5 else arr
    if stack.shape[0] != int(n_expected):
        raise ValueError(
            f"the batch carries {stack.shape[0]} conditioning frames but this checkpoint "
            f"reads {n_expected}. The ingest history length is set from the backend's own "
            "img_indices, so a mismatch means the planner was built for a different model.")
    return stack


def native_chunk_from_uva(
    action_rel: np.ndarray, force_pred: np.ndarray | None, layout, *, predict_force: bool
) -> np.ndarray:
    """UVA ``(H,10)`` relative action (+ ``(H,6)`` physical wrench) -> FUJI native ``(H,16)``.

    Pure, so ``tests/test_hl_muse.py`` can pin the layout mapping (and the gripper
    metres->fraction round trip) without a backend. Both inputs stay in UVA's own units;
    the ONLY conversion here is the gripper, because the FUJI slot is a fraction of
    :data:`~fuji.s1_policy.data.transforms.GRIPPER_WIDTH_REF_M`.
    """
    act = np.asarray(action_rel, dtype=np.float64)
    if act.ndim != 2 or act.shape[1] != UVA_ACTION_DIM:
        raise ValueError(f"expected a (H,{UVA_ACTION_DIM}) UVA action chunk, got {act.shape}")
    horizon = act.shape[0]
    slots = layout.slots("left")
    native = np.zeros((horizon, layout.action_dim), dtype=np.float64)
    native[:, slots.pos] = act[:, 0:3]
    native[:, slots.rot6d] = act[:, 3:9]
    if slots.gripper is not None:
        native[:, slots.gripper] = act[:, 9] / GRIPPER_WIDTH_REF_M
    if slots.force is not None:
        if predict_force:
            wrench = np.asarray(force_pred if force_pred is not None else (), dtype=np.float64)
            if wrench.shape != (horizon, FORCE_DIM):
                raise ValueError(
                    f"predict_force=True but the backend returned a wrench of shape "
                    f"{wrench.shape}; expected ({horizon}, {FORCE_DIM}).")
            native[:, slots.force] = wrench
        # predict_force=False leaves the force columns at EXACT zero, which is what
        # gate_force_command sees: ||F|| = 0 never exceeds any gate >= 0, so a pure-position
        # UVA checkpoint publishes no force command no matter how the switch is set. Loudly
        # announced at load time (load_muse_model) rather than left to be discovered.
    return native


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_muse_model(
    checkpoint: str | Path | None,
    *,
    backend_python: str | Path,
    device: str = "cuda",
    layout_mode: str = "left_only",
    zarr_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    force_fusion: str | None = None,
    config_name: str = "uva_hl_force",
    backend: MuseBackend | None = None,
) -> tuple[LoadedHLModel, MuseBackend]:
    """Bring up the MuSe backend and wrap it as a :class:`LoadedHLModel`.

    Returns ``(loaded, backend)`` — the caller owns the backend's lifetime and must
    ``close()`` it (``hl_serving`` does so in its shutdown path). ``backend`` may be passed in
    to reuse an already-handshaked child (the offline session shares one between its
    dataset-anchor requests).

    Refuses, loudly, when:
      * the layout is not ``left_only`` — UVA-umi predicts ONE hand, so any layout with a
        right hand or head would leave those roles decoded from all-zero slots;
      * ``ACTION_HORIZON != num_actions`` — see the module docstring; the message names
        ``FUJI_ACTION_HORIZON`` because a launcher is the only place that can still set it;
      * ``down_sample_steps != 1`` — the checkpoint would be on a different time grid than
        this build's ``ACTION_DT``, so every step would land at the wrong instant.
    """
    layout = get_layout(str(layout_mode))
    if layout.legacy:
        raise ValueError("muse serving does not support the legacy32 layout")
    if layout.roles != ("left",):
        raise ValueError(
            f"muse serving needs the left_only layout; got {layout.mode!r} with roles "
            f"{layout.roles}. UVA-umi predicts a SINGLE hand (act_dim=10), so any other role "
            "would be decoded from all-zero action slots and driven to the anchor.")

    own_backend = backend is None
    be = backend or MuseBackend(
        backend_python,
        checkpoint=checkpoint,
        device=device,
        zarr_path=zarr_path,
        dataset_root=dataset_root,
        force_fusion=force_fusion,
        config_name=config_name,
    )
    try:
        meta = be.meta
        if meta.action_dim != UVA_ACTION_DIM:
            raise ValueError(
                f"this checkpoint's action width is {meta.action_dim}, not {UVA_ACTION_DIM} "
                "(pos3 + rot6d6 + gripper1). A multi-robot UVA checkpoint is not supported.")
        if meta.down_sample_steps != 1:
            raise ValueError(
                f"checkpoint was trained with down_sample_steps={meta.down_sample_steps}, so "
                f"one model step is {meta.down_sample_steps} recorded frames — but this build's "
                f"ACTION_DT is {ACTION_DT * 1e3:.2f} ms (one frame). Every predicted step would "
                "be published at the wrong time. Retrain with down_sample_steps=1.")
        if int(meta.num_actions) != int(ACTION_HORIZON):
            raise ValueError(
                f"checkpoint emits {meta.num_actions}-step chunks but this process resolved "
                f"ACTION_HORIZON={ACTION_HORIZON}. Relaunch with "
                f"FUJI_ACTION_HORIZON={meta.num_actions} (the launchers derive it from the "
                "checkpoint; a manual export that disagrees is refused rather than guessed).")
        if meta.force_dim != FORCE_DIM:
            raise ValueError(
                f"checkpoint force width {meta.force_dim} != this build's FORCE_DIM {FORCE_DIM}")
        if meta.random_init:
            _log("WARNING: backend is RANDOM-INITIALISED (no --checkpoint) — this is a wiring "
                 "check only. Every chunk it returns is meaningless; do NOT arm the robot.")
        if not meta.use_force:
            _log("WARNING: use_force=False in this checkpoint — the force window this planner "
                 "assembles is sent but IGNORED by the model.")
        if not meta.predict_force:
            _log("WARNING: predict_force=False in this checkpoint — the predicted wrench is "
                 "EXACTLY ZERO, so the force command stays gated whatever the viser switch "
                 "says, and the planner is pure position.")

        config = MuseConfig(
            action_horizon=int(meta.num_actions),
            force_hist_len=int(meta.force_history_length),
            action_dim=int(layout.action_dim),
            state_dim=int(layout.state_dim),
            num_sampling_steps=meta.num_sampling_steps,
            num_iter=meta.num_iter,
        )
        loaded = LoadedHLModel(
            algorithm=MUSE_ALGO,
            model=MuseRemoteModel(be, meta, layout),
            config=config,
            layout=layout,
            norm_stats={},  # UVA normalizes inside the backend (ft_scale + normalizer_type)
            device=torch.device("cpu"),  # nothing runs locally; the child owns the GPU
            num_steps=1,
            obs_history_len=int(meta.n_img_cond),
            obs_history_dt_s=float(meta.obs_history_dt_s),
            image_mean=MUSE_IMAGE_MEAN,
            image_std=MUSE_IMAGE_STD,
            image_key="camera0_rgb",
            force_hist_len=int(meta.force_history_length),
        )
    except Exception:
        if own_backend:
            be.close()
        raise
    return loaded, be


# ---------------------------------------------------------------------------
# The planner: LivePlanner + the one stream HL-v2's observation does not carry
# ---------------------------------------------------------------------------
class MuseLivePlanner(LivePlanner):
    """:class:`LivePlanner` plus an ABSOLUTE EE-position history ring.

    Everything UVA conditions on is already in the HL-v2 observation except one thing:
      * ``robot0_eef_rot_axis_angle_wrt_start`` == the ``state`` wrt_start channel, verbatim;
      * ``robot0_gripper_width`` == the ``state`` gripper fraction x ``GRIPPER_WIDTH_REF_M``;
      * ``robot0_ft_wrench`` == the binned ``force_history`` (this checkpoint's
        ``force_history_length`` bins, each one action step wide — which is exactly what the
        training converter wrote into the zarr at 30 Hz, so the filter matches);
      * the 4 conditioning images == the ``To`` image ring, because ``load_muse_model`` sets
        ``obs_history_len``/``obs_history_dt_s`` from the checkpoint's own image comb and
        ``hl_serving`` derives its ingest stride from those;
      * ``robot0_eef_pos`` / ``robot0_eef_rot_axis_angle`` are ANCHOR-relative, and the HL-v2
        state has no absolute position and rotates against the EPISODE START instead — so
        this subclass keeps the absolute pose history that supplies them.

    Kept as a subclass rather than a change to :class:`LivePlanner` so the flow / diffusion /
    act paths are byte-identical: they never construct this class.

    TWO DEPLOY-DISTRIBUTION GAPS, neither introduced here, both worth knowing before believing
    a rollout:

    1. In the ``last_target`` obs contract the pose ring logs the ANCHOR, and the anchor only
       moves on ``commit_anchor`` after a chunk executes. So between commits every history row
       holds the same pose and the anchor-relative position history is EXACTLY ZERO, whereas in
       training it is 400 ms of real motion. That is a property of the contract (the
       ``wrt_start`` rotation channel has it too, for every algorithm), not of this shim.
    2. ``hl_serving`` fills the image ring on a STRIDE (4 ticks at 30 Hz) so the To frames land
       on the model's own 133 ms comb, which also means the newest frame can be up to one
       stride (133 ms) old at Predict time, where training had it exactly at t0. Same behavior
       every To>1 flow/diffusion checkpoint gets; fixing it would need a separate
       newest-frame slot, and diverging from the shared ingest for one algorithm is the more
       expensive mistake.
    """

    #: Bound on the seqlock retry in :meth:`snapshot_muse_obs`. A retry costs one extra ring
    #: copy (~microseconds) against a 33 ms push period, so hitting this bound means the
    #: ingest thread is running far faster than configured, not that the loop is wrong.
    _SNAPSHOT_RETRIES = 8

    def __init__(self, loaded: LoadedHLModel, obs_hz: float = 30.0) -> None:
        super().__init__(loaded, obs_hz=obs_hz)
        if loaded.algorithm != MUSE_ALGO:
            raise ValueError(
                f"MuseLivePlanner is for algo={MUSE_ALGO!r}; got {loaded.algorithm!r}. The "
                "local algorithms must use LivePlanner (this subclass adds a ring they do not "
                "read and overrides the proprio push).")
        # Ring of ABSOLUTE anchor positions, oldest-first, one row per ring slot — the same
        # length and the same push instants as the base's rotation/gripper rings, so row i of
        # all three describes the same observation frame.
        self._m_pos: list[np.ndarray] = [self._anchor_pos[0].copy() for _ in range(self._To)]
        #: Bumped on every proprio push. Lets :meth:`snapshot_muse_obs` detect that a push
        #: landed between its two locked reads (the base's ring copy is inside
        #: ``snapshot_obs``, which takes the same non-reentrant lock, so one atomic read of
        #: both is not available without touching the base class).
        self._m_seq = 0

    # --- ingest -----------------------------------------------------------
    def push_proprio_from_anchor(self) -> None:
        """Append the current anchor pose + gripper to ALL THREE rings, atomically.

        Reimplements the base method instead of calling it, because the base takes the same
        non-reentrant lock: a ``super()`` call plus a second critical section for ``_m_pos``
        could interleave with a reader and leave row i of the position ring describing a
        different frame than row i of the rotation ring — a silent one-tick pose/rotation
        skew. Keep in lockstep with :meth:`LivePlanner.push_proprio_from_anchor`;
        ``tests/test_hl_muse.py`` asserts the three rings stay the same length and aligned.
        """
        with self._lock:
            self._buf_rot6d.append(self._anchor_rot6d.copy())
            self._buf_grip.append(self._grip_frac_vec())
            self._m_pos.append(self._anchor_pos[0].copy())
            for buf in (self._buf_rot6d, self._buf_grip, self._m_pos):
                if len(buf) > self._To:
                    buf.pop(0)
            self._m_seq += 1

    def seed_anchor(self, anchor_pos: np.ndarray, anchor_rot6d: np.ndarray) -> None:
        super().seed_anchor(anchor_pos, anchor_rot6d)
        self._reseed_pos_ring()

    def reset_start_pose(self) -> None:
        super().reset_start_pose()
        self._reseed_pos_ring()

    def _reseed_pos_ring(self) -> None:
        """Refill the position ring with the current anchor (fresh episode / re-seed).

        Mirrors what the base does to the rotation/gripper rings. Not folded into the same
        critical section as the base's re-seed, which is tolerable here and not in the push
        path: every slot is being set to the SAME value, so the worst an interleaved push can
        do is leave one row holding a real pose that is, at that instant, the anchor anyway.
        """
        with self._lock:
            self._m_pos = [self._anchor_pos[0].copy() for _ in range(self._To)]
            self._m_seq += 1

    # --- observation ------------------------------------------------------
    def snapshot_muse_obs(self):
        """``(obs, anchor_pos, anchor_rot6d, eef_pos_rel, eef_rot6d_rel)``.

        The standard HL-v2 observation plus the two anchor-relative pose-history arrays UVA
        needs, all describing the same ring state. The seqlock retry is what makes that
        "same": ``snapshot_obs`` and the position-ring read are separate acquisitions of one
        non-reentrant lock, so a proprio push between them would pair row i of the position
        history with row i of a rotation history one tick newer.

        Both relative arrays are computed against the SNAPSHOTTED ANCHOR — the very pose
        :func:`decode_chunk` un-relativizes the answer against — so "obs base == action base"
        holds structurally rather than by ingest timing.
        """
        for _ in range(self._SNAPSHOT_RETRIES):
            with self._lock:
                seq_before = self._m_seq
                pos_hist = np.stack(self._m_pos)  # (To,3) absolute
            obs, anchor_pos, anchor_rot6d = super().snapshot_obs()
            with self._lock:
                if self._m_seq == seq_before:
                    break
        else:
            _log("WARNING: proprio ring changed under every snapshot attempt; the pose history "
                 "may be one ingest tick out of step with the rotation history.")

        slots = self.layout.state_slots()["left"]
        rot6d_wrt_start = np.asarray(obs.state[:, slots["wrt_start_rot6d"]], dtype=np.float64)
        # Anchor-relative rotation, recovered from the wrt_start channel rather than from a
        # second rotation ring:  R_anchor^T R_t = (R_start^T R_anchor)^T (R_start^T R_t).
        anchor_wrt_start = rot6d_relative(anchor_rot6d[0], self._start_rot6d_snapshot())
        rot6d_rel = rot6d_relative(rot6d_wrt_start, anchor_wrt_start)
        # Anchor-relative position: the translation half of absolute_to_delta_ee. Its rotation
        # output is discarded on purpose — feeding it the (absolute) anchor rotation for every
        # row would give R_anchor^T R_anchor = I, not the per-row rotation above.
        pos_rel, _ = absolute_to_delta_ee(
            pos_hist, np.tile(anchor_rot6d[0], (pos_hist.shape[0], 1)),
            anchor_pos[0], anchor_rot6d[0],
        )
        return obs, anchor_pos, anchor_rot6d, pos_rel, rot6d_rel

    def _start_rot6d_snapshot(self) -> np.ndarray:
        with self._lock:
            return self._start_rot6d[0].copy()

    # --- one inference ----------------------------------------------------
    def infer_chunk(
        self, *, seed: int, force_enabled: bool, gate_n: float, num_steps: int | None = None,
        capture: dict | None = None,
    ):
        """The muse override of :meth:`LivePlanner.infer_chunk`.

        Identical in shape and contract to the base — snapshot, batch, sample, decode, and the
        same ``capture`` out-parameter the deploy recorder reads — differing only in the two
        extra batch keys the remote model needs. The anchor is NOT advanced here (that stays
        ``commit_anchor``'s job), so a preview is repeatable and never moves the robot.
        """
        if self.m.model is None:
            raise RuntimeError(
                "this MuseLivePlanner has no policy — only recorded-episode replay is "
                "available. Launch with --algo muse and a --checkpoint.")
        if num_steps is not None and int(num_steps) != 1:
            raise ValueError(
                f"the UVA action head decodes a chunk in one pass; num_steps must be 1 or "
                f"omitted, got {num_steps}.")
        obs, anchor_pos, anchor_rot6d, pos_rel, rot6d_rel = self.snapshot_muse_obs()
        batch = batch_from_obs(self.m, obs)
        batch[MUSE_EEF_POS_REL_KEY] = pos_rel
        batch[MUSE_EEF_ROT6D_REL_KEY] = rot6d_rel
        prediction = sample_chunk(self.m, batch, seed=seed, num_steps=None)
        if capture is not None:
            capture.update(
                obs=obs, anchor_pos=anchor_pos, anchor_rot6d=anchor_rot6d,
                prediction=prediction, seed=int(seed), num_steps=1,
            )
        return decode_chunk(
            prediction, self.layout, anchor_pos, anchor_rot6d,
            force_enabled=force_enabled, gate_n=gate_n,
        )


def resolve_backend_python(explicit: str | Path | None = None) -> str:
    """Locate the uvaforce interpreter: ``explicit`` -> ``$BACKEND_PY`` -> the conda default.

    Kept here (not in the launcher) so ``scripts/infer_s1_muse.py`` and ``scripts/hl_serving.py``
    resolve it identically, and so a bare ``python scripts/infer_s1_muse.py`` works without the
    shell wrapper.
    """
    for candidate in (explicit, os.environ.get("BACKEND_PY"),
                      os.path.expanduser("~/miniconda3/envs/uvaforce/bin/python")):
        if candidate and os.access(os.path.expanduser(str(candidate)), os.X_OK):
            return os.path.abspath(os.path.expanduser(str(candidate)))
    raise FileNotFoundError(
        "could not find the uvaforce python (the env that can import unified_video_action). "
        "Pass --backend-python or set BACKEND_PY=/path/to/uvaforce/bin/python. See "
        "docs/RUNBOOK_muse_8gpu.md for how that env is built."
    )


def probe_num_actions(checkpoint: str | Path, backend_python: str | Path) -> int:
    """``model.policy.num_actions`` of a UVA checkpoint, read via the backend interpreter.

    The launchers need this BEFORE FUJI's python starts, because ``ACTION_HORIZON`` is
    resolved at import time — the exact reason ``run_hl_serving.sh`` already derives the pin
    from a flow/diffusion/act ``config.json``. A UVA checkpoint keeps its config as a pickled
    OmegaConf inside the ``.pt``, which only the uvaforce interpreter can unpickle, so the
    probe is a subprocess rather than a local read.
    """
    script = Path(__file__).resolve().parents[3] / "scripts" / "muse_hl_infer_server.py"
    out = subprocess.run(  # noqa: S603 - argv built here
        [os.path.abspath(os.path.expanduser(str(backend_python))), str(script),
         "--checkpoint", os.path.abspath(os.path.expanduser(str(checkpoint))), "--print-meta"],
        capture_output=True, text=True, check=False,
    )
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return int(json.loads(line)["num_actions"])
    raise RuntimeError(
        f"could not read num_actions from {checkpoint}: the backend printed no meta JSON.\n"
        f"stdout: {out.stdout[-800:]}\nstderr: {out.stderr[-1500:]}")


__all__ = [
    "MUSE_ALGO",
    "MUSE_EEF_POS_REL_KEY",
    "MUSE_EEF_ROT6D_REL_KEY",
    "UVA_ACTION_DIM",
    "MuseBackend",
    "MuseConfig",
    "MuseLivePlanner",
    "MuseMeta",
    "MuseRemoteModel",
    "load_muse_model",
    "native_chunk_from_uva",
    "probe_num_actions",
    "resolve_backend_python",
]
