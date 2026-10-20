"""Offline HL-v2 inference session for a MuSe-on-UVA force checkpoint.

The fourth sibling of :mod:`~flore.s1_policy.inference.flow`,
:mod:`~flore.s1_policy.inference.diffusion` and :mod:`~flore.s1_policy.inference.act`, and the
one that is structurally different in two ways:

1. **The model runs in another process.** See :mod:`flore.s1_policy.serving.hl_muse` — UVA only
   loads in the ``uvaforce`` env. So there is no ``config.json`` / ``model.safetensors`` to
   read, no ``strict=True`` weight load and no bundled ``norm_stats.json``; the checkpoint is
   validated by the backend instead (EMA weights really populated, frozen VAE really restored,
   ``ft_scale`` consistent) and its grid arrives over the handshake.

2. **There are TWO datasets, and their agreement is the test.** The observation is built by
   the backend from the UVA training dataset (``prepare_data_predict_action`` at an explicit
   ``(episode, anchor)`` — the eval path step for step), while the TARGET chunk, the decode
   anchor and every display array come from FLoRe's own :class:`HLDataset` over the SAME zarr.
   Two independent loaders, two independent relativizations. So every ``infer()`` cross-checks
   that the anchor pose they each resolved for the same frame agrees to
   :data:`ANCHOR_TOLERANCE_M` / :data:`ANCHOR_TOLERANCE_ROT`, and refuses otherwise. That is
   the load-bearing check in this file: a mismatch means the two are decoding in different
   absolute frames (a non-identity ``world_align`` on one side, a different puck→TCP extrinsic,
   an episode-index off-by-one), all of which otherwise present as a plausible-looking rollout
   that is simply wrong.

Why not have the backend build the FLoRe-style observation too, so both paths share one
builder: because the honest question offline is "does the trained model reproduce the recorded
demonstration", and that is best answered by feeding it the exact window its own dataloader
would have. The LIVE path (:class:`~flore.s1_policy.serving.hl_muse.MuseLivePlanner`) has no
zarr and must assemble from sensor rings, so the two builders genuinely differ — the same split
``muse_force_infer_server.py`` / ``muse_force_serving_server.py`` already have. The anchor
cross-check is what keeps that split honest.

Anchor coverage: FLoRe's HL anchors and UVA's window bounds are not the same set. UVA needs
``num_actions - 1 - history_padding_length`` frames of history and ``num_actions`` frames of
future; FLoRe masks its chunk tail instead. :attr:`MuseInferenceSession.legal_indices` is the
intersection, and it is what the CLI seeds the viewer's sample slider from.

Glitch dropping defaults OFF here, unlike the three local sessions. ``HLDataset`` refuses a
``glitch_pre_s`` shorter than the observation look-back, and a muse checkpoint's look-back is
``(n_img_cond - 1) * img_step_stride`` steps = 400 ms against a one-step default — a knob the
``hl_data`` block cannot reach. Turning it off rather than raising is right on the merits too:
the UVA loader applies its OWN pose-glitch window exclusion to the window it builds, so FLoRe's
pass would only be filtering which anchors are OFFERED, not what the model is fed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flore.s1_policy.data.layout import ACTION_DT
from flore.s1_policy.inference.common import (
    HLSampleSource,
    RolloutResult,
    build_rollout_result,
)
from flore.s1_policy.serving.hl_muse import (
    MUSE_ALGO,
    MUSE_IMAGE_MEAN,
    MUSE_IMAGE_STD,
    MuseBackend,
    load_muse_model,
    native_chunk_from_uva,
)
from flore.utils import muse_wire as wire

#: Agreement required between the FLoRe dataset's anchor and the backend's raw-zarr anchor.
#: 1 mm / ~0.06 deg: both read the same float32 zarr column, so the only expected difference is
#: float32->float64 rounding (~1e-7). Anything at the millimetre scale is a real frame
#: disagreement, not noise — the two things that produce one (a ``world_align`` rotation, a
#: different TCP extrinsic) are centimetres and degrees.
ANCHOR_TOLERANCE_M = 1e-3
ANCHOR_TOLERANCE_ROT = 1e-3


class MuseInferenceSession:
    """A deterministic offline inference session for one UVA force checkpoint.

    ``infer()`` is reproducible: the backend seeds torch per request, so the same
    ``(sample_index, seed)`` gives the same chunk across clicks and process restarts — unlike
    a bare ``sample_tokens``, whose diffusion heads draw from the global RNG.
    """

    algorithm = MUSE_ALGO

    def __init__(
        self,
        zarr_path: str | Path,
        *,
        checkpoint: str | Path | None,
        backend_python: str | Path,
        device: str = "cuda",
        dataset_root: str | Path | None = None,
        val_last_episodes: int = 0,
        drop_glitch_windows: bool = False,
        force_fusion: str | None = None,
        config_name: str = "uva_hl_force",
    ) -> None:
        zarr = Path(zarr_path).expanduser().resolve()
        if not zarr.exists():
            raise FileNotFoundError(f"zarr not found: {zarr}")

        # The backend comes up FIRST: its handshake states the image comb and the force window,
        # and the FLoRe dataset must be built to match (obs_history_len / obs_history_dt_s), not
        # the other way round. A checkpoint is the authority on its own grid.
        self.backend = MuseBackend(
            backend_python,
            checkpoint=checkpoint,
            device=device,
            zarr_path=zarr,
            dataset_root=dataset_root,
            force_fusion=force_fusion,
            config_name=config_name,
        )
        try:
            self.loaded, _ = load_muse_model(
                checkpoint,
                backend_python=backend_python,
                device=device,
                zarr_path=zarr,
                backend=self.backend,
            )
            self.meta = self.backend.meta
            hl_data = {
                "zarr_paths": [str(zarr)],
                "mode": "left_only",
                "image_key": "camera0_rgb",
                "image_mean": MUSE_IMAGE_MEAN,
                "image_std": MUSE_IMAGE_STD,
                "obs_history_len": int(self.meta.n_img_cond),
                "obs_history_dt_s": float(self.meta.obs_history_dt_s),
                "drop_invalid_force_history": False,
                "drop_glitch_windows": bool(drop_glitch_windows),
                "norm_stats_path": "",
                "val_last_episodes": int(val_last_episodes),
            }
            self.source = HLSampleSource(hl_data)
        except Exception:
            self.backend.close()
            raise

        self.num_steps = 1
        #: 1, like ACT: the UVA action head decodes in one pass. Its OWN diffusion-head step
        #: counts are checkpoint properties (see :attr:`MuseMeta.num_sampling_steps`) and are
        #: deliberately not offered as a spinner that would silently do nothing.
        self.max_num_steps = 1
        self._closed = False
        self.legal_indices = self._resolve_legal_indices()

    # --- anchor coverage --------------------------------------------------
    def _anchor_bounds(self, episode_frames: int) -> tuple[int, int]:
        """``(lo, hi)`` episode-relative anchors whose UVA window is fully in-episode.

        Mirrors the backend's own bound (which is the authority and re-checks): the history half
        may be edge-clamped by at most ``history_padding_length`` frames, and the future half
        needs ``num_actions`` frames past the anchor.
        """
        lo = int(self.meta.valid_anchor_min)
        hi = int(episode_frames) - int(self.meta.valid_anchor_slack_end) - 1
        return lo, hi

    def _resolve_legal_indices(self) -> tuple[int, ...]:
        """FLoRe dataset indices whose anchor is also a legal UVA window.

        Reported rather than silently skipped: the viewer indexes ``source`` directly, so an
        illegal index still reaches :meth:`infer` and must fail with a clear message. This is
        what lets the CLI seed the slider somewhere that works.
        """
        frames = self.meta.episode_frame_nums
        out: list[int] = []
        for ref in self.source.refs:
            if ref.file_index != 0:
                continue
            if ref.file_episode_id >= len(frames):
                continue
            lo, hi = self._anchor_bounds(frames[ref.file_episode_id])
            if lo <= ref.episode_frame <= hi:
                out.append(ref.dataset_index)
        return tuple(out)

    # --- inference --------------------------------------------------------
    def infer(self, sample_index: int, *, seed: int = 0, num_steps: int | None = None) -> RolloutResult:
        if self._closed:
            raise RuntimeError("MuseInferenceSession is closed")
        if num_steps is not None and int(num_steps) != 1:
            raise ValueError(
                f"the UVA action head decodes a chunk in one pass; num_steps must be 1 or "
                f"omitted, got {num_steps}.")
        ref, sample = self.source.get(sample_index)
        if ref.file_index != 0:
            raise ValueError(
                f"sample {sample_index} comes from file index {ref.file_index}; this session "
                "serves a single zarr (the backend opens exactly one).")

        reply_meta, arrays = self.backend.request(
            {
                "kind": wire.KIND_ZARR,
                "episode": int(ref.file_episode_id),
                "anchor": int(ref.episode_frame),
                "seed": int(seed),
            }
        )
        self._check_anchor(ref, sample, arrays)
        prediction = native_chunk_from_uva(
            arrays["action_rel"], arrays.get("force_pred"), self.source.layout,
            predict_force=self.meta.predict_force,
        )
        _ = reply_meta
        return build_rollout_result(
            algorithm=self.algorithm,
            num_steps=1,
            seed=seed,
            ref=ref,
            sample=sample,
            predicted_actions=prediction,
            source=self.source,
        )

    def _check_anchor(self, ref, sample, arrays) -> None:
        """Refuse when the two loaders disagree about this frame's absolute anchor pose.

        See the module docstring: this is the end-to-end frame check, and it is cheap, so it
        runs on EVERY inference rather than once at startup — an episode-index mapping that is
        right for episode 0 and wrong for episode 7 would otherwise pass a startup check.
        """
        if "anchor_pos" not in arrays:
            raise RuntimeError("backend reply carries no anchor pose to cross-check")
        be_pos = np.asarray(arrays["anchor_pos"], dtype=np.float64).reshape(3)
        be_rot = np.asarray(arrays["anchor_rot6d"], dtype=np.float64).reshape(6)
        flore_pos = sample["anchor_pos"].detach().cpu().numpy().astype(np.float64).reshape(-1, 3)[0]
        flore_rot = sample["anchor_rot6d"].detach().cpu().numpy().astype(np.float64).reshape(-1, 6)[0]
        d_pos = float(np.max(np.abs(be_pos - flore_pos)))
        d_rot = float(np.max(np.abs(be_rot - flore_rot)))
        if d_pos > ANCHOR_TOLERANCE_M or d_rot > ANCHOR_TOLERANCE_ROT:
            raise ValueError(
                "FLoRe's dataset and the UVA backend disagree about the anchor pose for "
                f"episode {ref.file_episode_id} frame {ref.episode_frame} (dataset index "
                f"{ref.dataset_index}): |dpos|={d_pos * 1e3:.2f} mm, |drot6d|={d_rot:.4f}.\n"
                f"  FLoRe  pos={flore_pos.round(5).tolist()} rot6d={flore_rot.round(4).tolist()}\n"
                f"  UVA   pos={be_pos.round(5).tolist()} rot6d={be_rot.round(4).tolist()}\n"
                "The prediction would be un-relativized against a different frame than it was "
                "conditioned in. Usual causes: the zarr's root attrs carry a non-identity "
                "world_align (FLoRe applies it, the UVA loader does not), or the two loaders "
                "enumerate episodes differently."
            )

    # --- lifecycle --------------------------------------------------------
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.source.close()
        self.backend.close()

    def __enter__(self) -> "MuseInferenceSession":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    # --- reporting --------------------------------------------------------
    def episode_split(self, ref) -> str:
        """``"train"`` / ``"held-out"`` / ``"unknown"`` for a sample, by UVA's OWN split.

        NOT the ``split`` field on :class:`SampleRef`, which follows FLoRe's last-N-episodes
        convention. UVA holds out by ratio (``used_episode_indices``), so the two disagree, and
        the honest answer to "was this rollout in-distribution" is this one. Reported rather
        than folded into the viewer's label because the viewer's label is the dataset's, and
        silently redefining it would make two tools disagree about the same word.
        """
        if not self.meta.train_episodes:
            return "unknown"
        return "train" if int(ref.file_episode_id) in set(self.meta.train_episodes) else "held-out"

    def describe(self) -> str:
        """One line for the launcher log: the grid, the force contract, the anchor coverage."""
        total = len(self.source)
        legal = len(self.legal_indices)
        n_train = len(self.meta.train_episodes)
        held_out = (f", UVA held out {self.source.dataset.episode_count - n_train}/"
                    f"{self.source.dataset.episode_count} episodes" if n_train else "")
        return (
            f"muse offline: H={self.meta.num_actions} @ {1.0 / ACTION_DT:.4g} Hz, "
            f"img {self.meta.n_img_cond}x@stride{self.meta.img_step_stride}, "
            f"force_hist={self.meta.force_history_length} "
            f"(use_force={self.meta.use_force}, predict_force={self.meta.predict_force}), "
            f"anchors {legal}/{total} usable"
            + ("" if legal == total else
               f" (the rest are inside {self.meta.valid_anchor_min} frames of an episode start "
               f"or {self.meta.valid_anchor_slack_end} of its end, where UVA's window would run "
               "off the episode)")
            + held_out
        )


__all__ = ["MuseInferenceSession", "ANCHOR_TOLERANCE_M", "ANCHOR_TOLERANCE_ROT"]
