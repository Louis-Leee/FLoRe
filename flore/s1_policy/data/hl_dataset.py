"""Torch Dataset for the HL-v2 high-level policy contract.

This is a SIBLING of :mod:`flore.s1_policy.data.umi_dataset`, not a replacement. The
pre-v2 dataset stays untouched so the already-trained/deployed 32-dim checkpoint keeps
loading bit-for-bit; new training uses this module.

The contract (see :mod:`flore.s1_policy.data.layout` for the column layout, and the
HL-v2 section of ``UMI_collection/processing/umi_convert_core.py`` for the grids)::

    OBSERVATION ending at camera frame t0 (To = obs_history_len frames, oldest-first,
    time-anchored at t0 - k*obs_history_dt_s; a single camera at To == 1 drops the leading
    axis)
      image          (n_cams*To, 3, 224, 224)  camera-major, oldest-first within a camera
      state          (To, state_dim)    per keypoint [wrt_start_rot6d 6][gripper 1?]
      force_history  (8, 6*n_ft)        causal anti-aliased wrench bins over [t0-1s, t0]

    ACTION over the half-open interval [t0, t0+1s), 16 steps on an exact 16 Hz grid.
    T0-INCLUSIVE (DECISION-T0, UMI-on-Air parity): step 0 lands ON t0, so the chunk is a
    strict shift of the observation; step 0's POSE is the anchor identity (teaches
    nothing on its own) while its force/gripper carry the current signal.
      action_chunk   (16, action_dim)
        per hand:  [pos 3 | rot6d 6 | gripper 1 | force 6]
        head:      [pos 3 | rot6d 6]

Two things this dataset does NOT feed the model, deliberately:

* **No absolute EE POSITION.** ``state`` carries orientation-relative-to-episode-start
  (``wrt_start_rot6d``, see the Observation section) but never absolute position:
  position lives in the image + the anchor-relative action. Feeding the absolute pose
  is what made the pre-v2 deployed policy see conditioning far outside its training
  range (training state was absolute lighthouse-world metres, up to ~11 m; serving sent
  pelvis-frame values). ``wrt_start_rot6d`` is frame-INVARIANT to the world origin, so
  it does not reintroduce that gap. The absolute anchor is still RETURNED
  (``anchor_pos`` / ``anchor_rot6d``) because serving needs it to un-relativize, but it
  is not part of ``state``.
* **No force in ``state``.** The force history is a TIME SERIES with its own encoder;
  flattening it into ``state`` would both discard its temporal structure and (for two
  hands) overflow the model's ``filled_state_dim``, where it would be silently
  truncated by ``_pad_state``.

The WHOLE-BODY mode (``mode="wholebody"``)
------------------------------------------
One mode replaces the pose columns with the G1's 29 JOINT ANGLES — the configuration the
``dense`` low level consumes, produced at DATA-PREPARATION time by motion completion
(``UMI_collection/motion_completion/mc_to_hl_wholebody.py``) and stored as extra ``wb_*``
arrays beside the ``hl_*`` ones. Everything else about the sample is unchanged: the same two
camera frames, the same ``hl_<role>_action_{gripper,force}`` targets, the same
``hl_<role>_force_history`` input, the same masks AND-ed the same way. Four differences, all
consequences of the action no longer being a Cartesian pose:

* **The chunk is ABSOLUTE.** No anchor, no ``absolute_to_delta_ee``, no un-relativization at
  serving (``layout.pose_dims()`` is empty). ``anchor_pos``/``anchor_rot6d`` are still
  returned for the two hands, as the provenance a viewer needs to draw where the hands were —
  the model never sees them and no decode depends on them.
* **``state`` is joint space**: ``[joints 29][L grip 1][R grip 1]``, the anchor frame's own
  joint values, not ``wrt_start_rot6d``. So ``state_valid`` gates on whether the completion
  covered the frame (``wb_state_valid``) rather than on an episode-start orientation
  reference, and the episode-start glitch flag is irrelevant to it.

  The ``wholebody_geom_*`` modes replace that 29-wide block with a pelvis-frame GEOMETRIC
  reduction of the same joints — ``rel_xy`` (hand-to-hand and hand-to-pelvis xy) and/or ``rpy``
  (each wrist's Euler attitude) — because the raw joints are the channel measured to be far
  out of distribution at the robot (``layout.py``: 16 of 29 past the 0.20 rad pre-arm gate).
  The features are FK'd ONCE in ``__init__`` through the SAME
  :class:`~flore.s1_policy.inference.ready_pose.KeypointFK` the deploy anchor uses, gated by the
  same ``wb_state_valid``, and the ACTION is untouched — so the family is a controlled
  experiment on the observation alone. See :mod:`flore.s1_policy.data.wb_geom`.
* **The joint mask is the AND over EVERY tracker.** ``wb_action_valid`` says the completion
  covered the step and the step is inside the episode; a dropout on ANY of the three pucks
  (including the chest, which no keypoint role in this mode owns) means the completion was
  conditioned on interpolated input there, so the 29 columns are AND-ed with every role's
  ``hl_<role>_action_valid`` too. The gripper/force masks then inherit that term in place of
  the pose validity they use in the keypoint modes — same asymmetry, same reason: a target
  the body cannot reach is not a useful label.
* **All 43 columns are supervised**, so the reserved-but-masked bookkeeping has nothing to do
  here (``layout.unmeasured_dims()`` is empty).

⚠️ The joints' absolute values are only meaningful relative to the completion's ANCHOR, which
is recorded as ``attrs.wb_cond_mode``. Mixing two anchors in one training run would train one
model on two posture spaces with no shape to say so, so every file in a run must agree — this
loader refuses a disagreement by name.

Frames
------
Poses on disk are ABSOLUTE aligned-world. Each keypoint's chunk is made relative to
ITS OWN anchor pose (``R_ref^T (p - p_ref)``, ``R_ref^T R``), i.e. every hand ends up
in its own local frame — the pre-v2 ``ref_frame="per_keypoint"`` semantics, which is
also the only frame mode serving implements. Gripper and force are frame-invariant
and pass through un-relativized; the wrench stays in the Bota sensor frame, exactly as
recorded (``attrs["hl_force_frame"] == "bota_sensor"``).

Masks
-----
``action_mask (16, action_dim)`` is the per-step, per-dim supervision mask, built by
AND-ing the converter's validity arrays into the layout's column groups:

* pose columns of step k: ``hl_<role>_action_valid`` — the step lies inside the episode,
  the tracker covered it, and it is not inside a glitch or an implausible jump.
* gripper column of step k: additionally requires a live encoder stream
  (``hl_<role>_action_gripper_valid``). On the shipped session this AND is a behavioural
  NO-OP — the gripper mask is strictly LOOSER there (41 of 7,696 steps valid where the
  pose mask is not, and zero the other way) — so it is defence-in-depth, not a live
  correction. It is what protects a dead-encoder take, where the converter zero-fills the
  width and 0.0 is a legal "fully closed".
* force columns of step k: additionally require a measured, tare-clean wrench bin
  (``hl_<role>_action_force_valid``).

Note the asymmetry: a step past the episode end masks force too, even where the wrench
there was measured and clean. That discards a few trainable force targets at each episode
tail, and is deliberate — a target the pose cannot reach is not a useful force label.

A sample whose force HISTORY is invalid (``hl_<role>_force_history_valid``) has
``force_history_valid=False``; the force-history encoder is expected to gate on it
rather than treat a stale hold as a measurement. Such samples are still trainable for
pose/gripper, so they are kept by default (``drop_invalid_force_history=False``).

With TWO F/T sensors (``both_hands_head_dual``) that flag is **per sensor**, a
``(n_ft,)`` bool in ``layout.force_roles`` order — the same order ``force_history``'s
6-channel blocks are written in. The two Botas tare independently (one request per
sensor, each under its own lock) and drop out independently, so a single shared bool
would let one hand's gap replace the other hand's *measured* wrench with the encoder's
learned "no data" embedding. One sensor still yields a scalar, so the shipped
single-hand contract — and every serving/eval consumer built against it — is unchanged.

Glitch-window sample dropping (``drop_glitch_windows``, default ON)
------------------------------------------------------------------
Per-step masking is not enough for a tracker glitch. The load-bearing corruption is at
the anchor ``t0``: a glitch there poisons the anchor pose, so the anchor-relative
transform makes **every** one of the 16 targets wrong — even the steps whose own
absolute pose is clean — and no per-step mask covers the anchor.

So this dataset (matching the converter's documented-but-previously-unimplemented
"a training sampler excludes the windows that straddle a flagged frame") DROPS the
whole anchor from the index when its window overlaps a corrupted-pose time region. The
window is the SYMMETRIC ``[t0 - FORCE_HIST_SPAN_S, t0 + ACTION_SPAN_S]`` (1 s each side).
The post side is load-bearing — the 16 action targets live in ``[t0, t0 + ACTION_SPAN_S)``
(t0-inclusive, DECISION-T0). The pre side is ALSO load-bearing now that the observation
is a history (DECISION-OBS2): this anchor reads the past for its *image* frames (at
``t0 - k*obs_history_dt_s``) AND its *wrt_start* orientation, in addition to the *force*
history — a tracker glitch inside the ``(To-1)*obs_history_dt_s`` look-back corrupts the
orientation channel of an obs frame, and one inside the force window is handled by
``force_history_valid``. FORCE_HIST_SPAN_S (1 s) comfortably covers the default obs
look-back (``(2-1)*0.25 = 0.25 s``); it is kept at the full second as conservative
insurance (~1.5% extra anchors on the shipped session). Keep ``glitch_pre_s`` >= the obs
look-back if you shrink it; ``glitch_post_s`` must stay at ``ACTION_SPAN_S``. (Separately,
a glitch at the EPISODE-START frame corrupts the wrt_start REFERENCE for the whole
episode — those samples keep their anchor but get ``state_valid=False`` on the wrt_start
slot; see ``__getitem__``.)

The corrupted region is derived from the converter's OWN outputs — never re-thresholded
here, so the two can't drift:

* ``hl_<role>_action_valid`` steps that are invalid AND **not tail padding** (the step's
  own time is at/before the episode's last camera frame). It also catches mid-stream
  tracker dropouts (frozen/interpolated pose), which are equally untrustworthy targets.
* ``robot0_pose_glitch`` frames as point intervals.

  DATASETS BUILT BEFORE 2026-08-05 UNDER-REPORT BOTH OF THESE, and clean94 is one of
  them. Two converter-side bugs, now fixed, made a *bridged* lighthouse teleport
  invisible: ``robot0_pose_glitch`` was built from unbridged excursions only (empty by
  construction for a bridged one), and ``MAX_TARGET_SPEED_MPS`` was set from free-hand
  speed (4.0 m/s) rather than from the camera-grid step distribution it screens, so it
  flagged 0 of 634,928 clean94 steps. Net effect on the SHIPPED clean94: episodes 30 and
  31 carry fabricated 118 mm / 127 mm one-frame interpolation ramps (1.76 / 1.91 m/s, vs
  a 0.506 m/s max across the 89 clean episodes) with ``pose_glitch == 0`` on every frame
  and ``action_valid == 1`` on the straddling steps — i.e. supervised at full loss
  weight. Re-converting the session picks up both fixes (masked frames 10 -> 65 of
  39,683; invalid action steps 1.805% -> 1.867%); this loader needs no change because it
  already reads whatever the converter reports. See
  ``umi_convert_core.MAX_TARGET_SPEED_MPS`` and ``convert_to_umi._corrupt_spans``.

The scan is PER EPISODE (``episode_timestamps`` is relative and restarts each episode),
so a window never leaks a glitch across a demonstration boundary. On a clean episode
nothing is dropped, including its legitimate tail anchors. Set
``drop_glitch_windows=False`` to keep every frame (e.g. to test step-masking in
isolation).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from torch.utils.data import Dataset

from flore.s1_policy.data.layout import (
    ACTION_DT,
    ACTION_HZ,
    ACTION_HORIZON,
    ACTION_HORIZON_ENV,
    ACTION_SPAN_S,
    FORCE_DIM,
    FORCE_HIST_LEN,
    FORCE_HIST_SPAN_S,
    ACTION_HORIZON_DEFAULT,
    FOOT_ACTION_KEYS,
    JOINT_DIM,
    OBS_HISTORY_DT_S,
    OBS_HISTORY_LEN,
    WB_JOINT_NAMES,
    WB_JOINT_TARGETS,
    WB_STATE_BLOCK_DIM,
    get_layout,
    hands_flag,
    tracker_roles_required,
    zarr_keys_for,
)
from flore.s1_policy.data.wb_geom import FK_BLOCKS, geom_series, load_geom_fk
from flore.s1_policy.data.transforms import (
    GRIPPER_WIDTH_REF_M,
    absolute_to_delta_ee,
    gripper_width_to_norm,
    rot6d_relative,
    rotvec_to_rot6d,
)

logger = logging.getLogger(__name__)

_IMAGE_CODECS_REGISTERED = False

# Increment whenever HLDataset changes the tensors or validity masks produced from an
# otherwise identical source archive. Norm stats and checkpoints bind to this value.
HL_TRANSFORM_CONTRACT_VERSION = 1


def _register_image_codecs() -> None:
    """Register the JpegXl numcodecs codec (id ``imagecodecs_jpegxl``) the converter
    used for ``camera0_rgb``. Idempotent; must run before the first array-metadata
    load or the read raises ``codec not available``."""
    global _IMAGE_CODECS_REGISTERED
    if _IMAGE_CODECS_REGISTERED:
        return
    from imagecodecs.numcodecs import register_codecs

    register_codecs()
    _IMAGE_CODECS_REGISTERED = True


def zip_content_fingerprint(path: str) -> str:
    """Cryptographically bind provenance to the exact portable ZIP artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: str) -> tuple[int, int, int, int]:
    stat = os.stat(path)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class HLDataset(Dataset):
    """Map-style dataset over ``.zarr.zip`` files carrying the HL-v2 ``hl_*`` arrays.

    Parameters
    ----------
    zarr_paths : str | list[str]
        One or more ``.zarr.zip`` paths, each written by ``convert_to_umi.py`` with
        ``--hands``/``--hl-windows``.
    mode : str
        Layout mode (``left_only`` | ``right_only`` | ``both_hands`` |
        ``both_hands_head`` | ``both_hands_head_dual`` | ``wholebody``). Must be satisfiable
        by every file: a file lacking a required role's arrays is a hard error, never a
        silently smaller dataset. ``wholebody`` additionally requires the ``wb_*`` joint
        arrays (see "The WHOLE-BODY mode" in the module docstring); a plain
        ``--hands both_head_dual`` zarr is refused by name with the exporter command.
    image_key, image_mean, image_std
        Image handling: ``(x/255 - mean) / std``. Defaults map [0,255] -> [-1,1].
    image_keys : tuple[str, ...] | None
        MULTI-CAMERA: every camera to read, in the order the model's vision encoder will
        see them. ``None`` (default) means ``(image_key,)`` — the single-camera contract,
        byte-identical to before this parameter existed. The 3-tracker rig has two wrist
        D405s (``camera0_rgb`` LEFT / ``camera1_rgb`` RIGHT), paired onto camera0's grid
        by the converter's causal frame SELECTION, so both keys are already on the same
        anchor timeline and no extra resampling happens here.

        ``image`` is then ``(n_cams * To, C, H, W)``, **camera-major and oldest-first
        within a camera** — ``[cam0 t-Δ, cam0 t0, cam1 t-Δ, cam1 t0]``. ``collate_hl_batch``
        splits that leading axis into the model's ``images`` list, and the model builds
        ``enc_out_dim = num_cameras * To * num_features`` in the same order. With one
        camera this is exactly the old ``(To, C, H, W)``; with ``n_cams*To == 1`` it
        collapses to ``(C, H, W)``, also as before.
    drop_invalid_force_history : bool
        When True, anchors whose force-history window is unmeasured are excluded from
        the index entirely (rather than kept with ``force_history_valid=False``).
    drop_glitch_windows : bool
        When True (the default), an anchor is excluded from the index if its
        ``[t0 - FORCE_HIST_SPAN_S, t0 + ACTION_SPAN_S]`` window overlaps a
        corrupted-pose time region (a tracker glitch or dropout). See the module
        docstring; the region is derived from the converter's ``action_valid`` /
        ``robot0_pose_glitch``, never re-thresholded here.
    glitch_pre_s, glitch_post_s : float
        History / future half-widths of the sample window used for the overlap test.
        Default to the layout's ``FORCE_HIST_SPAN_S`` / ``ACTION_SPAN_S`` so the drop
        matches what ``__getitem__`` actually reads. Do not shrink them below the real
        spans — that would keep samples whose history or chunk still straddles a glitch.
    obs_history_len : int
        Number of observation frames (image + state) per sample, oldest-first, ending at
        ``t0`` (DECISION-OBS2, UMI-on-Air parity). ``2`` (default) gives a short-horizon
        velocity cue; ``1`` recovers the single-frame contract (``image`` is ``(C,H,W)``
        and ``state`` is ``(state_dim,)`` rather than carrying a leading ``To`` axis).
    obs_history_dt_s : float
        Wall-clock spacing between consecutive obs frames. TIME-anchored (nearest camera
        frame at/before ``t0 - k*dt``), so it is robust to a non-uniform camera grid; a
        fixed frame stride would be a variable time offset on jittery sessions. The
        look-back stays within the anchor's episode and clamps to its first frame.
    require_episode_ends : bool
        Require a strictly increasing ``meta/episode_ends`` partition ending at the frame
        count. Enable for every episode validation split; the legacy one-episode fallback
        is unsafe for splitting and is then a hard error.
    wb_joint_target : str
        ``wholebody`` only: what the 29 joint TARGET columns mean —
        :data:`~flore.s1_policy.data.layout.WB_JOINT_TARGETS`. ``absolute`` (default) is the
        export's own radians; ``delta`` subtracts the chunk's own row 0, giving anchor-relative
        joint motion. The STATE stays absolute in both. Refused on a keypoint mode.
    """

    def __init__(
        self,
        zarr_paths: str | list[str],
        mode: str = "left_only",
        image_key: str = "camera0_rgb",
        image_mean: float = 0.5,
        image_std: float = 0.5,
        image_keys: tuple[str, ...] | list[str] | None = None,
        drop_invalid_force_history: bool = False,
        drop_glitch_windows: bool = True,
        glitch_pre_s: float = FORCE_HIST_SPAN_S,
        glitch_post_s: float = ACTION_SPAN_S,
        obs_history_len: int = OBS_HISTORY_LEN,
        obs_history_dt_s: float = OBS_HISTORY_DT_S,
        require_episode_ends: bool = False,
        wb_joint_target: str = "absolute",
    ) -> None:
        self.layout = get_layout(mode)
        if self.layout.legacy:
            raise ValueError(
                "HLDataset cannot use the frozen 'legacy32' layout (it has no force "
                "columns and a padded, non-contiguous slot table). Use UMIDataset for "
                "legacy checkpoints."
            )
        self.mode = mode
        # Which TRACKERS a sample depends on — NOT `layout.roles`. They are equal for every
        # keypoint mode; for `wholebody` the roles are the two hand blocks while the joints
        # come from a completion conditioned on the chest too, so the glitch scan and the
        # required-array check must read this. See layout.tracker_roles_required.
        self.tracker_roles: tuple[str, ...] = tracker_roles_required(mode)
        # What the 29 joint TARGET columns mean (see layout.WB_JOINT_TARGETS). Refused rather
        # than ignored off `wholebody`: a keypoint layout has no joint block for it to describe,
        # so honouring it silently would let a launcher pass it to the wrong preset and believe
        # the target changed.
        self.wb_joint_target = str(wb_joint_target)
        if self.wb_joint_target not in WB_JOINT_TARGETS:
            raise ValueError(
                f"wb_joint_target must be one of {WB_JOINT_TARGETS}, got {wb_joint_target!r}")
        if self.wb_joint_target != "absolute" and not self.layout.is_wholebody:
            raise ValueError(
                f"wb_joint_target={self.wb_joint_target!r} needs a whole-body layout (29 joint "
                f"target columns); mode={mode!r} predicts keypoint poses, whose anchor-relative "
                "frame is chosen by `ref_frame` instead.")
        # `image_keys` is the full ordered camera list; `image_key` stays the FIRST one, so
        # every existing single-image consumer (`episode_image` for the offline viewer,
        # `n_frames`, the provenance knob) keeps its meaning without a branch.
        if image_keys is None:
            image_keys = (image_key,)
        self.image_keys: tuple[str, ...] = tuple(str(k) for k in image_keys)
        if not self.image_keys:
            raise ValueError("image_keys must name at least one camera")
        if len(set(self.image_keys)) != len(self.image_keys):
            # A duplicate would double-feed one camera into global_cond while the model
            # believed it had two views — invisible in every shape.
            raise ValueError(f"image_keys has duplicates: {self.image_keys}")
        self.image_key = self.image_keys[0]
        self.image_mean = float(image_mean)
        self.image_std = float(image_std)
        self.drop_invalid_force_history = bool(drop_invalid_force_history)
        self.drop_glitch_windows = bool(drop_glitch_windows)
        self.glitch_pre_s = float(glitch_pre_s)
        self.glitch_post_s = float(glitch_post_s)
        # Observation history (DECISION-OBS2): To frames of image + state, oldest-first,
        # time-anchored at t0 - k*obs_history_dt_s. To == 1 recovers the single-frame
        # contract. The history stays WITHIN the anchor's episode (episode_timestamps
        # restarts each episode) and clamps to the episode's first frame before start
        # (UMI's start-pad: the oldest frames repeat the first).
        self.obs_history_len = max(1, int(obs_history_len))
        self.obs_history_dt_s = float(obs_history_dt_s)
        self.require_episode_ends = bool(require_episode_ends)
        # The glitch pre-window must cover everything this anchor reads from the past.
        # Since DECISION-OBS2 the obs history reaches back (To-1)*obs_history_dt_s for its
        # image + wrt_start orientation frames; a glitch there corrupts them. Refuse a
        # pre-window shorter than that look-back so shrinking one knob can't silently
        # re-admit glitch-straddling observation frames. On the v3 grid the obs is the
        # single frame t0, so the look-back is EXACTLY 0 and the (now 1-bin, 0.033 s)
        # FORCE_HIST_SPAN_S default satisfies this trivially; the check earns its keep the
        # moment obs_history_len is raised above 1 without also raising glitch_pre_s.
        if self.drop_glitch_windows:
            obs_lookback = (self.obs_history_len - 1) * self.obs_history_dt_s
            if self.glitch_pre_s + 1e-9 < obs_lookback:
                raise ValueError(
                    f"glitch_pre_s={self.glitch_pre_s} is shorter than the observation "
                    f"look-back {obs_lookback} s ((obs_history_len-1)*obs_history_dt_s); "
                    f"an obs frame could straddle a glitch undetected. Raise glitch_pre_s "
                    f"to >= {obs_lookback} s."
                )

        if isinstance(zarr_paths, str):
            zarr_paths = [zarr_paths]
        self.zarr_paths = list(zarr_paths)

        # (file_idx, frame) for every usable anchor. Built by scanning metadata only,
        # then closing every store: a live ZipStore holds a file handle that does not
        # survive pickling to DataLoader workers, and a handle inherited across fork()
        # corrupts under concurrent reads. Per-process handles are opened lazily in
        # __getitem__ via _root().
        self._index: list[tuple[int, int]] = []
        # Per-file resolved zarr key names, so __getitem__ never re-derives them.
        self._keys: list[dict] = []
        # Per-file (relative timestamps, episode-start-frame-of-each-frame). Small int/
        # float arrays (pickle fine, unlike zarr handles) so __getitem__ can build the
        # time-anchored obs history and locate each keypoint's episode-start orientation
        # WITHOUT re-reading meta per sample. episode_timestamps is RELATIVE, so both are
        # scoped per episode.
        self._ts: list[np.ndarray] = []
        self._ep_start_frame: list[np.ndarray] = []
        # Per-file [n_frames] bool: is this frame's EPISODE-START frame corrupt? The
        # wrt_start orientation channel is measured relative to the episode's first
        # frame, so a glitch THERE biases every anchor in the episode; __getitem__ marks
        # the wrt_start state slot invalid for those samples. Uses robot0_pose_glitch (the
        # unbridged-excursion flag) — the only per-frame pose-corruption signal on the
        # camera grid (action_valid is per future step, not per anchor frame).
        self._ep_start_glitched: list[np.ndarray] = []
        # Public split metadata is aligned with sample indices while ``_index`` keeps its
        # historical (file_idx, frame) tuple API. Episode IDs are flattened in zarr-path
        # order, so they remain unambiguous across multiple files.
        self.sample_episode_ids: list[int] = []
        self.episode_identities: list[tuple[int, int]] = []
        self._episode_ends_by_file: list[list[int]] = []
        self._episode_ends_valid: list[bool] = []
        self._source_content_fingerprints: list[str] = []
        self._source_file_identities: list[tuple[int, int, int, int]] = []
        # Per-file root attrs, captured during this metadata scan because every store is
        # closed at the end of it. Small JSON-ish dicts (~1.4 KB/file), so they pickle to
        # DataLoader workers fine, unlike a zarr handle. Exposed read-only via
        # :attr:`file_attrs` for consumers that must be dataset-derived rather than
        # hardcoded — notably the offline viewer's display geometry, which reads each
        # dataset's own ``T_tracker_tcp`` puck->TCP extrinsic.
        self._file_attrs: list[dict] = []
        # `wholebody` only: the completion ANCHOR every file was exported against
        # (attrs.wb_cond_mode). Two anchors in one run are two posture spaces with nothing in
        # the tensor shapes to say so, so the first file's value becomes the run's contract and
        # _check_file refuses any disagreement. None for every keypoint mode.
        self._wb_cond_mode: str | None = None
        # `wholebody_geom_*` only: per-file (n_frames, geom_dim) float array, FK'd ONCE here
        # from `wb_state_joints` through the same KeypointFK the deploy anchor uses. A small
        # array like `_ts`, so it pickles to every DataLoader worker; a per-__getitem__ FK
        # would put 0.66 ms of single-threaded work behind every image decode. Empty list for
        # every other mode. See flore/s1_policy/data/wb_geom.py.
        self._geom_blocks: tuple[str, ...] = tuple(
            b for b in self.layout.state_geom if b in FK_BLOCKS)
        self._wb_geom: list[np.ndarray] = []
        # block -> its columns inside `_wb_geom` (which carries ONLY the FK blocks, in
        # `_geom_blocks` order). Kept separate from `layout.state_geom_slots()` — the slices
        # into the STATE frame — because the two coincide today only because no current mode
        # mixes `joints` with a geom block. Deriving each side from its own layout is what keeps
        # that a coincidence instead of an assumption.
        self._geom_src_slots: dict[str, slice] = {}
        _cur = 0
        for _block in self._geom_blocks:
            _width = WB_STATE_BLOCK_DIM[_block]
            self._geom_src_slots[_block] = slice(_cur, _cur + _width)
            _cur += _width
        geom_fk = None
        if self._geom_blocks:
            # BEFORE the file loop: a missing yourdfpy/submodule must not surface after 30 s
            # of sha256 per multi-GB file. Refused, never zero-filled — the geometry IS this
            # mode's proprioception, and a zero block is a plausible posture (hands together
            # at the pelvis) that no shape or metric would flag.
            geom_fk, why = load_geom_fk()
            if geom_fk is None:
                raise ValueError(
                    f"mode={mode!r} conditions on the FK'd geometric state blocks "
                    f"{list(self._geom_blocks)}, but the shared KeypointFK is unavailable: "
                    f"{why}. That FK is the same one the deploy ready pose and hl_dense's "
                    f"command anchor use, so it is deliberately not reimplemented here — "
                    f"install it, or train mode='wholebody' (29 raw joints) / "
                    f"'wholebody_geom_none' (grippers only), neither of which needs a URDF.")
        episode_offset = 0

        for fi, path in enumerate(self.zarr_paths):
            identity_before = _file_identity(path)
            fingerprint = zip_content_fingerprint(path)
            identity_after = _file_identity(path)
            if identity_before != identity_after:
                raise ValueError(f"{path} changed while HLDataset fingerprinted it")
            self._source_content_fingerprints.append(fingerprint)
            self._source_file_identities.append(identity_after)
            root, store = self._open(path)
            try:
                attrs = dict(root.attrs)
                data = root["data"]
                self._check_file(path, attrs, data)
                self._keys.append(self._resolve_keys(attrs, data))
                self._file_attrs.append(attrs)

                n_frames = int(data[self.image_key].shape[0])
                starts, ends, ends_valid = self._episode_bounds(
                    root, n_frames, strict=self.require_episode_ends, return_valid=True
                )
                self._episode_ends_by_file.append([int(x) for x in ends])
                self._episode_ends_valid.append(ends_valid)
                self.episode_identities.extend((fi, local_id) for local_id in range(len(ends)))
                self._ts.append(np.asarray(data["episode_timestamps"][:], dtype=np.float64).reshape(-1))
                esf = self._episode_start_frames(root, n_frames, starts=starts, ends=ends)
                self._ep_start_frame.append(esf)
                if "robot0_pose_glitch" in data:
                    pg = np.asarray(data["robot0_pose_glitch"][:], dtype=np.float32).reshape(-1) > 0.5
                    self._ep_start_glitched.append(pg[esf])  # start-frame flag, per frame
                else:
                    self._ep_start_glitched.append(np.zeros(n_frames, dtype=bool))
                if self._geom_blocks:
                    # Read while the store is still open (4.3 MB for a 37k-frame file), then FK
                    # every COVERED frame. `wb_state_valid` gates it: an uncovered frame holds
                    # the exporter's zero-fill and 0 rad is a legal posture, so FK'ing it would
                    # invent a pose for a frame that has none. Those rows stay zero here and
                    # `__getitem__` marks them state_valid=False.
                    self._wb_geom.append(geom_series(
                        geom_fk,
                        np.asarray(data["wb_state_joints"][:], dtype=np.float64),
                        np.asarray(data["wb_state_valid"][:], dtype=np.float32).reshape(-1) > 0.5,
                        self._geom_blocks,
                        label=f" [{Path(path).name}]",
                    ))
                keep = np.ones(n_frames, dtype=bool)
                if self.drop_invalid_force_history:
                    for role in self.layout.force_roles:
                        vkey = f"hl_{role}_force_history_valid"
                        keep &= np.asarray(data[vkey][:], dtype=np.float32).reshape(-1) > 0.5
                n_glitch = 0
                if self.drop_glitch_windows:
                    glitch_keep = self._glitch_window_keep(root, data, n_frames)
                    n_glitch = int((keep & ~glitch_keep).sum())
                    keep &= glitch_keep
                kept_frames = np.nonzero(keep)[0]
                self._index.extend((fi, int(f)) for f in kept_frames)
                local_episode_ids = np.searchsorted(ends, kept_frames, side="right")
                self.sample_episode_ids.extend(episode_offset + int(ep) for ep in local_episode_ids)
                logger.info(
                    "HLDataset: %s mode=%s frames=%d usable=%d (dropped %d glitch-window) "
                    "(hands_mode=%s, hl_windows=%s)",
                    path,
                    mode,
                    n_frames,
                    int(keep.sum()),
                    n_glitch,
                    attrs.get("hands_mode"),
                    attrs.get("hl_windows"),
                )
                episode_offset += len(ends)
            finally:
                store.close()

        if not self._index:
            raise ValueError(
                f"HLDataset: no usable anchors across {self.zarr_paths} "
                f"(mode={mode}, drop_invalid_force_history={self.drop_invalid_force_history})"
            )

        self._roots: dict[int, object] = {}
        self._stores: dict[int, object] = {}
        self._handle_pid: int | None = None

        self.episode_count = len(self.episode_identities)
        fingerprint_payload = json.dumps(self._episode_ends_by_file, separators=(",", ":"), sort_keys=False).encode(
            "utf-8"
        )
        self.episode_ends_fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()
        anchor_payload = json.dumps(self._index, separators=(",", ":"), sort_keys=False).encode("utf-8")
        self.anchor_indices_fingerprint = hashlib.sha256(anchor_payload).hexdigest()
        source_payload = json.dumps(
            self._source_content_fingerprints,
            separators=(",", ":"),
            sort_keys=False,
        ).encode("utf-8")
        self.source_content_fingerprint = hashlib.sha256(source_payload).hexdigest()

    # -- file validation ---------------------------------------------------
    def _check_file(self, path: str, attrs: dict, data) -> None:
        """Fail loudly on any mismatch between this file and the requested contract.

        Every check here is one that would otherwise surface as a silently wrong
        dataset: a grid mismatch mis-times every label, a missing role yields fewer
        keypoints than the model has columns for, and a frame-convention mismatch is
        invisible in the values.
        """
        if not attrs.get("hl_windows"):
            raise ValueError(
                f"{path}: no HL-v2 windows (attrs.hl_windows is falsy). Re-convert "
                f"with: convert_to_umi.py <session> -o <out> --hands {hands_flag(self.mode)}"
            )
        # Grid parameters must match the layout constants exactly — an 8-bin/1 s
        # history and a 16-step/1 s chunk are baked into the model's input widths.
        for attr, want, what in (
            ("hl_force_hist_len", FORCE_HIST_LEN, "force-history bins"),
        ):
            got = attrs.get(attr)
            if got is None:
                raise ValueError(f"{path}: attrs.{attr} is required to verify the HL grid contract")
            try:
                numeric = float(got)
            except (TypeError, ValueError):
                numeric = math.nan
            if not np.isfinite(numeric) or numeric != float(want):
                raise ValueError(
                    f"{path}: attrs.{attr}={got} but this build expects "
                    f"{want} {what}. Re-convert or re-pin the contract."
                )
        # The ACTION HORIZON is the one grid parameter checked as ``stored >= wanted``
        # rather than ``==``, because on a t0-inclusive grid at a FIXED dt a shorter chunk
        # is a strict PREFIX of a longer one: step k lands at t0 + k*dt either way, so the
        # first H of a 50-step file ARE the 50-step file's own first H steps, bit for bit.
        # That is what lets an H=50 dataset train an H=30 run (FLORE_ACTION_HORIZON=30)
        # with no re-convert — the two runs then differ ONLY in the horizon, which is what
        # makes the A/B controlled. ``__getitem__`` takes that prefix.
        #
        # The relation holds ONLY at an identical dt, so the rate is what gets pinned
        # exactly (below), and here we additionally verify the file's OWN span/horizon —
        # not just its advertised hz — resolves to this build's ACTION_DT. Checking the
        # count alone would accept a 50-step/2.5 s file (20 Hz) for a 30-step request and
        # silently train on labels stretched 1.5x.
        stored_h = attrs.get("hl_action_horizon")
        if stored_h is None:
            raise ValueError(f"{path}: attrs.hl_action_horizon is required to verify the HL grid contract")
        try:
            stored_h_num = float(stored_h)
        except (TypeError, ValueError):
            stored_h_num = math.nan
        if not np.isfinite(stored_h_num) or stored_h_num < float(ACTION_HORIZON):
            raise ValueError(
                f"{path}: attrs.hl_action_horizon={stored_h} but this build expects at "
                f"least {ACTION_HORIZON} action steps. A shorter stored chunk cannot be "
                f"extended (only truncated), and re-indexing the stored windows along the "
                f"frame axis is NOT equivalent — the targets are resampled from the raw "
                f"high-rate streams onto a jittery camera grid (measured on merged3: up to "
                f"0.52 rad of joint error on 23% of steps). Re-convert with "
                f"--hl-action-horizon {ACTION_HORIZON} and re-run the wb export, or lower "
                f"{ACTION_HORIZON_ENV} to at most {int(stored_h_num)}."
            )
        stored_span = attrs.get("hl_action_span_s")
        try:
            stored_dt = float(stored_span) / stored_h_num
        except (TypeError, ValueError, ZeroDivisionError):
            stored_dt = math.nan
        if not np.isfinite(stored_dt) or abs(stored_dt - ACTION_DT) > 1e-9:
            raise ValueError(
                f"{path}: attrs.hl_action_span_s={stored_span} over "
                f"attrs.hl_action_horizon={stored_h} is a step dt of {stored_dt}, but this "
                f"build's ACTION_DT is {ACTION_DT}. The prefix relation that allows a "
                f"shorter horizon holds only at an IDENTICAL dt — re-convert this session "
                f"with --hl-action-hz {ACTION_HZ:g}."
            )
        # Spans come from the LAYOUT, not a literal: a hardcoded 1.0 would keep passing
        # stale files (and start rejecting correct ones) the moment the contract is
        # re-pinned to a different span — i.e. the check would accept exactly the files
        # it exists to reject. ``hl_action_span_s`` is NOT here: it scales with the stored
        # horizon, so it is checked above as ``span/horizon == ACTION_DT`` instead. The
        # force-history span stays exact — that window is an INPUT width baked into the
        # encoder, with no prefix relation to fall back on.
        for attr, want in (
            ("hl_force_hist_span_s", FORCE_HIST_SPAN_S),
            ("hl_action_hz", ACTION_HZ),
        ):
            got = attrs.get(attr)
            if got is None:
                raise ValueError(f"{path}: attrs.{attr} is required to verify the HL grid contract")
            try:
                numeric = float(got)
            except (TypeError, ValueError):
                numeric = math.nan
            if not np.isfinite(numeric) or abs(numeric - float(want)) > 1e-9:
                raise ValueError(
                    f"{path}: attrs.{attr}={got} but this build expects {want}. Re-convert or re-pin the contract."
                )
        # The ACTION GRID PHASE (DECISION-T0). This build expects the t0-INCLUSIVE grid
        # (step 0 lands ON t0; converter stamps hl_action_first_step="t0"). The pre-v2
        # grid EXCLUDED t0 (started at t0+dt) and carries IDENTICAL horizon/span/hz attrs,
        # so the one-dt shift is INVISIBLE in every other check and in the array values
        # themselves — a stale file trains a t0-inclusive model on labels that lead the
        # observation by one dt, and mixing grids in one dataset is internally
        # inconsistent. So this attr is the single discriminator and MUST be checked, not
        # inferred (the converter's own docstring says so). A non-"t0" value is a hard
        # error; a NULL/absent value (pre-provenance file, which could be EITHER grid and
        # is undetectable from values) is refused too rather than silently assumed t0.
        first_step = attrs.get("hl_action_first_step")
        if first_step != "t0":
            raise ValueError(
                f"{path}: attrs.hl_action_first_step={first_step!r}, but this build "
                f"expects the t0-inclusive action grid ('t0'). A pre-v2 file EXCLUDED t0 "
                f"(started at t0+dt) with identical horizon/span attrs, so the one-dt "
                f"shift is invisible in the values — training on it would mis-time every "
                f"label. Re-convert with the current converter (which stamps "
                f"hl_action_first_step='t0')."
            )
        # The gripper column is normalized by dividing metres by GRIPPER_WIDTH_REF_M.
        # The converter records the width its own calibration used, precisely so this can
        # be ASSERTED rather than assumed — a per-session calibration/gripper_range.json
        # with a different width_open_m (e.g. 0.085 vs 0.095) silently rescales every
        # gripper label by 12% and moves UMI's 0.08 m grasp threshold from 0.94 to 0.84
        # in normalized space.
        rig_open = attrs.get("gripper_width_open_m")
        if rig_open is not None and (
            not np.isfinite(float(rig_open)) or abs(float(rig_open) - GRIPPER_WIDTH_REF_M) > 1e-9
        ):
            raise ValueError(
                f"{path}: converted against a full-open gripper width of {rig_open} m, "
                f"but this build normalizes by GRIPPER_WIDTH_REF_M="
                f"{GRIPPER_WIDTH_REF_M} m (flore/s1_policy/data/transforms.py). Every "
                f"gripper label would be scaled by {float(rig_open) / GRIPPER_WIDTH_REF_M:.4f}. "
                f"Re-convert with the matching calibration, or re-pin the constant."
            )
        if attrs.get("hl_pose_frame") not in (None, "absolute_aligned_world"):
            raise ValueError(
                f"{path}: attrs.hl_pose_frame={attrs['hl_pose_frame']!r}; this dataset relativizes ABSOLUTE poses."
            )
        # The force columns pass through UN-relativized on the strength of the wrench
        # being in the sensor frame. If a future converter rotates it into world or TCP
        # frame (a natural follow-up, since the pose targets already are aligned-world),
        # the masks stay valid and the values stay plausible — invisible in training,
        # surfacing only as a policy pushing the wrong way. So gate it like the pose.
        if attrs.get("hl_force_frame") not in (None, "bota_sensor"):
            raise ValueError(
                f"{path}: attrs.hl_force_frame={attrs['hl_force_frame']!r}, but this "
                f"dataset passes the force columns through un-relativized, which is only "
                f"correct for the raw sensor frame. Teach the dataset the new frame "
                f"before training on it."
            )
        # A file converted for MORE roles than the requested mode loads fine and silently
        # discards the extra roles' columns — a bimanual recording training a left-only
        # checkpoint throws away half the demonstration. The converter stamps hands_mode
        # precisely so a loader can assert; the reverse direction is already caught by the
        # missing-array check below, so this closes the one-directional gap.
        got_mode = attrs.get("hands_mode")
        if got_mode is not None and got_mode != hands_flag(self.mode):
            raise ValueError(
                f"{path}: converted with --hands {got_mode!r} but loading as "
                f"mode={self.mode!r} (which expects --hands "
                f"{hands_flag(self.mode)!r}). Re-convert for this mode, or train the "
                f"mode the data was converted for — silently dropping the extra roles' "
                f"columns would discard part of every demonstration."
            )
        # BACKSTOP for a null hands_mode. The attr must tolerate None (pre-provenance
        # files predate it), but a null used to make the gate above vacuous: a file
        # converted with `--hl-windows` and no `--hands` emits every tracker on disk and
        # stamped None, so EVERY mode was accepted and silently discarded the rest.
        # The converter now derives the attr, so this only fires on an older file — but
        # comparing the roles actually present is a stronger check either way.
        if got_mode is None:
            present = {k.split("_")[1] for k in data if k.startswith("hl_")}
            # `tracker_roles`, not `layout.roles`: a `wholebody` file legitimately carries the
            # chest's hl_ arrays (its joints were completed from them) while owning no chest
            # BLOCK, so comparing against the roles would refuse the very file the mode wants.
            extra = present - set(self.tracker_roles) - set(self.layout.roles)
            if extra:
                raise ValueError(
                    f"{path}: carries HL arrays for roles {sorted(present)} but "
                    f"mode={self.mode!r} only consumes "
                    f"{sorted(set(self.tracker_roles) | set(self.layout.roles))}; "
                    f"{sorted(extra)} would be silently discarded (the file has no "
                    f"hands_mode attr, so the mode could not be checked directly). "
                    f"Train a mode that covers them, or re-convert with --hands "
                    f"{hands_flag(self.mode)}."
                )
        if self.layout.state_is_geometric:
            # The SAME check for both families, because both read the completion's output: a
            # `fivepoint_*` run needs `wb_joint_names` for the FK that builds its geometry, the
            # one-anchor `wb_cond_mode` contract for the same reason `wholebody` does, and the
            # grid/horizon contract because its FOOT arrays were derived from
            # `wb_action_joints` at that horizon.
            self._check_wholebody(path, attrs, data)
        if any(role in FOOT_ACTION_KEYS for role in self.layout.roles):
            self._check_foot_targets(path, attrs, data)
        # Required arrays, per role. The POSE pair is keyed on `tracker_roles`, not
        # `layout.roles`: they are identical for every keypoint mode, and for `wholebody` the
        # chest's arrays are required even though no role owns a chest BLOCK, because
        # `_glitch_window_keep` and the joint mask both read `hl_head_action_valid`.
        missing = []
        primary = str(attrs.get("robot0_primary_role", "left"))
        for role in sorted(set(self.tracker_roles) | set(self.layout.roles)):
            if role in FOOT_ACTION_KEYS:
                # Checked by `_check_foot_targets` (shapes included), and its arrays are `wb_*`
                # — asking for `hl_left_foot_action_pose` here would name a key that by design
                # does not and will not exist.
                continue
            need = [f"hl_{role}_action_pose", f"hl_{role}_action_valid"]
            if role in self.layout.gripper_roles:
                # The gripper VALIDITY mask is required, not optional. Treating it as
                # optional-with-a-fallback silently reintroduces exactly the bug the
                # converter's "always emit + mask" fix removed: a dead-encoder take is
                # zero-filled, and falling back to pose validity would supervise those
                # 16 steps toward a fabricated "fully closed".
                need += [f"hl_{role}_action_gripper", f"hl_{role}_action_gripper_valid"]
                # The OBSERVED opening at t0 likewise — 0.0 is a legal value (closed),
                # so an absent key must not degrade to it.
                need.append(zarr_keys_for(primary)[role][2])
            if role in self.layout.force_roles:
                need += [
                    f"hl_{role}_force_history",
                    f"hl_{role}_force_history_valid",
                    f"hl_{role}_action_force",
                    f"hl_{role}_action_force_valid",
                ]
            missing += [k for k in need if k not in data]
        if missing:
            raise ValueError(
                f"{path}: mode={self.mode!r} needs arrays {sorted(set(missing))} which "
                f"are absent. The file's hands_mode is {attrs.get('hands_mode')!r} — "
                f"re-convert with a --hands mode that covers {self.layout.roles}."
            )
        absent_cams = [k for k in self.image_keys if k not in data]
        if absent_cams:
            raise ValueError(
                f"{path}: missing image key(s) {absent_cams} (asked for "
                f"{list(self.image_keys)}, file has "
                f"{sorted(k for k in data if k.endswith('_rgb'))}). A second camera needs "
                f"a session recorded with that hand's D405 attached AND converted by a "
                f"converter that emits camera1_rgb (attrs.n_cameras reports how many the "
                f"file carries).")

    def _check_wholebody(self, path: str, attrs: dict, data) -> None:
        """Gate the ``wb_*`` joint block: grid, width, joint ORDER, and one shared anchor.

        Every check here is the whole-body twin of one the ``hl_*`` block already gets, for the
        same reason: the joint arrays are numerically plausible under any of these mistakes.
        A wrong grid mis-times the labels, a wrong joint order poses the wrong limb at full
        loss weight, and two completion anchors in one run are two posture spaces.
        """
        if not attrs.get("wb_windows"):
            raise ValueError(
                f"{path}: no whole-body windows (attrs.wb_windows is falsy). This zarr is a "
                f"plain --hands {hands_flag(self.mode)} conversion; add the 29-DoF joint "
                f"arrays with UMI_collection's\n"
                f"  ./run_mc_to_hl_wholebody.sh {path} -o <out>.zarr.zip\n"
                f"(motion completion runs there, ONCE, not at serving time)."
            )
        got_joints = attrs.get("wb_joints")
        if int(got_joints or 0) != JOINT_DIM:
            raise ValueError(
                f"{path}: attrs.wb_joints={got_joints} but this build's layout has "
                f"{JOINT_DIM} joint columns.")
        # The joint ORDER is the one thing here with no numeric tell: any permutation of 29
        # angles is a valid pose, and a leg/arm swap trains happily and renders a robot doing
        # something else. The exporter stamps the names it wrote, so compare them.
        got_names = attrs.get("wb_joint_names")
        if got_names is not None and tuple(str(n) for n in got_names) != WB_JOINT_NAMES:
            got_t = tuple(str(n) for n in got_names)
            first = next((i for i, (a, b) in enumerate(zip(got_t, WB_JOINT_NAMES)) if a != b),
                         min(len(got_t), len(WB_JOINT_NAMES)))
            raise ValueError(
                f"{path}: attrs.wb_joint_names disagrees with this build's WB_JOINT_NAMES "
                f"from index {first} ({got_t[first:first + 3]} vs "
                f"{WB_JOINT_NAMES[first:first + 3]}). A permutation of joint angles is a "
                f"valid pose, so nothing downstream would notice — re-export against the "
                f"current order (flore/s1_policy/data/layout.py::WB_JOINT_NAMES).")
        # Same grid contract, and checked the same way: the horizon may be LONGER than this
        # build trains on (a prefix at an identical dt is bit-exact, see the hl_ block above),
        # but the rate and the grid PHASE must match exactly.
        stored_h = attrs.get("wb_action_horizon")
        try:
            stored_h_num = float(stored_h)
        except (TypeError, ValueError):
            stored_h_num = math.nan
        if not np.isfinite(stored_h_num) or stored_h_num < float(ACTION_HORIZON):
            raise ValueError(
                f"{path}: attrs.wb_action_horizon={stored_h} but this build expects at least "
                f"{ACTION_HORIZON} joint steps (a shorter stored chunk can only be "
                f"truncated, never extended). Re-run run_mc_to_hl_wholebody.sh over a source "
                f"converted with --hl-action-horizon {ACTION_HORIZON}: the exporter takes the "
                f"horizon from the source file's own hl_* attrs, so the two stay on one grid.")
        stored_span = attrs.get("wb_action_span_s")
        try:
            stored_dt = float(stored_span) / stored_h_num
        except (TypeError, ValueError, ZeroDivisionError):
            stored_dt = math.nan
        if not np.isfinite(stored_dt) or abs(stored_dt - ACTION_DT) > 1e-9:
            raise ValueError(
                f"{path}: attrs.wb_action_span_s={stored_span} over "
                f"attrs.wb_action_horizon={stored_h} is a step dt of {stored_dt}, but this "
                f"build's ACTION_DT is {ACTION_DT}.")
        got_hz = attrs.get("wb_action_hz")
        try:
            hz_num = float(got_hz)
        except (TypeError, ValueError):
            hz_num = math.nan
        if not np.isfinite(hz_num) or abs(hz_num - ACTION_HZ) > 1e-9:
            raise ValueError(
                f"{path}: attrs.wb_action_hz={got_hz} but this build's ACTION_HZ is "
                f"{ACTION_HZ}.")
        if attrs.get("wb_action_first_step") != "t0":
            raise ValueError(
                f"{path}: attrs.wb_action_first_step="
                f"{attrs.get('wb_action_first_step')!r}, but this build expects the "
                f"t0-inclusive grid ('t0') — the same one dt shift that is invisible in the "
                f"hl_ block is invisible here.")
        # ONE completion anchor per run (see the module docstring). The first file's value is
        # the contract; note the assignment happens only after the file has passed everything
        # above, so a rejected file cannot become the reference.
        cond = attrs.get("wb_cond_mode")
        cond = None if cond is None else str(cond)
        if self._wb_cond_mode is None:
            self._wb_cond_mode = cond
        elif cond != self._wb_cond_mode:
            raise ValueError(
                f"{path}: exported with attrs.wb_cond_mode={cond!r} but an earlier file in "
                f"this dataset used {self._wb_cond_mode!r}. The joint values are ABSOLUTE and "
                f"only mean anything relative to that anchor, so mixing two would train one "
                f"model on two posture spaces with nothing in the shapes to say so. Re-export "
                f"every shard with the same --cond-mode.")
        missing = [k for k in ("wb_action_joints", "wb_action_valid", "wb_state_joints",
                               "wb_state_valid") if k not in data]
        if missing:
            raise ValueError(
                f"{path}: attrs.wb_windows is set but arrays {missing} are absent — the "
                f"file was written by an older/partial exporter. Re-run "
                f"run_mc_to_hl_wholebody.sh.")
        for key, want_tail in (("wb_action_joints", (int(stored_h_num), JOINT_DIM)),
                               ("wb_action_valid", (int(stored_h_num),)),
                               ("wb_state_joints", (JOINT_DIM,)),
                               ("wb_state_valid", (1,))):
            got_shape = tuple(int(x) for x in data[key].shape[1:])
            if got_shape != want_tail:
                raise ValueError(
                    f"{path}: data/{key} has per-frame shape {got_shape}, expected "
                    f"{want_tail}.")

    def _check_foot_targets(self, path: str, attrs: dict, data) -> None:
        """The ``fivepoint_*`` family's eight FOOT arrays: present, shaped, and PROVENANCE.

        A foot is the one keypoint on this rig that no sensor ever saw, so its arrays cannot be
        validated against a recording — only against the derivation they claim. Hence the frame
        attr is checked by VALUE and not merely for presence: a foot pose written in the
        completion's world frame instead of the pelvis frame is the same shape, the same
        magnitude class, and trains to a plausible-looking loss, but it would be answering
        "where in the room" for a channel the wire reads as "where under the body".
        """
        if not attrs.get("wb_foot_targets"):
            raise ValueError(
                f"{path}: no foot targets (attrs.wb_foot_targets is falsy). Layout mode "
                f"{self.mode!r} predicts FIVE keypoints and the rig tracks three, so the two "
                f"ankles are FK'd from this file's own `wb_action_joints` by\n"
                f"  uv run scripts/add_foot_targets.py {path} -o <out>_5pt.zarr.zip\n"
                f"(no model, no GPU, no second completion — see the script's docstring)."
            )
        want_frame = "pelvis_stance_no_root"
        got_frame = attrs.get("wb_foot_frame")
        if got_frame != want_frame:
            raise ValueError(
                f"{path}: attrs.wb_foot_frame={got_frame!r}, expected {want_frame!r}. The foot "
                f"target must be `KeypointFK(joints)`'s ankle pose with NO root pose applied — "
                f"a STANCE relative to the pelvis. A world-frame foot has the same shape and "
                f"the same units, so nothing downstream would notice.")
        # The horizon rule is the hl_/wb_ one: stored >= this build's, prefix-truncated.
        try:
            stored_h = int(float(attrs.get("wb_action_horizon")))
        except (TypeError, ValueError):
            stored_h = -1
        missing = []
        for role in sorted(r for r in self.layout.roles if r in FOOT_ACTION_KEYS):
            pose_key, valid_key = FOOT_ACTION_KEYS[role]
            anchor_pos, anchor_rot, _ = zarr_keys_for("left")[role]
            for key, want_tail in ((pose_key, (stored_h, 6)), (valid_key, (stored_h,)),
                                   (anchor_pos, (3,)), (anchor_rot, (3,))):
                if key not in data:
                    missing.append(key)
                    continue
                got = tuple(int(x) for x in data[key].shape[1:])
                if got != want_tail:
                    raise ValueError(
                        f"{path}: data/{key} has per-frame shape {got}, expected {want_tail} "
                        f"(from attrs.wb_action_horizon={stored_h}). Re-run "
                        f"scripts/add_foot_targets.py against THIS file — the foot arrays were "
                        f"written from a different horizon than the joints they came from.")
        if missing:
            raise ValueError(
                f"{path}: attrs.wb_foot_targets is set but arrays {sorted(missing)} are absent. "
                f"The file was written by an older/partial run of "
                f"scripts/add_foot_targets.py; re-run it.")

    def _resolve_keys(self, attrs: dict, data) -> dict:
        """Per-role zarr array names + the observation gripper source key."""
        primary = str(attrs.get("robot0_primary_role", "left"))
        zarr_keys = zarr_keys_for(primary)
        out: dict = {"roles": {}}
        # Every TRACKER role, not just the layout's blocks: `wholebody` needs the chest's
        # `hl_head_action_valid` for its joint mask even though the chest owns no block. For a
        # keypoint mode the two sets are equal, so nothing here changes.
        for role in sorted(set(self.tracker_roles) | set(self.layout.roles)):
            if role in FOOT_ACTION_KEYS:
                # A foot's chunk is `wb_*`, not `hl_*`: nothing about it came out of the UMI
                # converter (see layout._FOOT_ZARR_KEYS on why the honest name is worth a
                # branch). Same (T,H,6) shape and same pos-then-rotvec order, so everything
                # downstream of this dict is identical to the chest's path.
                pose_key, valid_key = FOOT_ACTION_KEYS[role]
                entry = {"pose": pose_key, "valid": valid_key}
            else:
                entry = {
                    "pose": f"hl_{role}_action_pose",
                    "valid": f"hl_{role}_action_valid",
                }
            if role in self.layout.gripper_roles:
                # All three are REQUIRED by _check_file, so no optional fallback: a
                # missing validity mask or observation key would silently supervise a
                # fabricated gripper value (0.0 is legal — fully closed).
                entry["gripper"] = f"hl_{role}_action_gripper"
                entry["gripper_valid"] = f"hl_{role}_action_gripper_valid"
                # The OBSERVED gripper opening at t0 comes from the camera-grid array
                # (robot0_/robot1_gripper_width) — the state at the anchor, not the
                # future-target array.
                entry["gripper_obs"] = zarr_keys[role][2]
            if role in self.layout.force_roles:
                entry["force_hist"] = f"hl_{role}_force_history"
                entry["force_hist_valid"] = f"hl_{role}_force_history_valid"
                entry["force"] = f"hl_{role}_action_force"
                entry["force_valid"] = f"hl_{role}_action_force_valid"
            # Absolute anchor pose at t0 (camera grid), needed to relativize.
            entry["anchor_pos"] = zarr_keys[role][0]
            entry["anchor_rot"] = zarr_keys[role][1]
            out["roles"][role] = entry
        if self.layout.state_is_geometric:
            # Named here rather than inline in __getitem__ for the same reason the per-role
            # keys are: one table says which array feeds which model channel.
            #
            # Keyed on state_is_geometric, not is_wholebody: the `fivepoint_*` family reads the
            # SAME `wb_state_joints` / `wb_state_valid` pair for its proprioception (through the
            # FK reduction) while its ACTION is keypoint poses, so the two halves of this table
            # are gated separately. A 5-point layout that was handed `wb_action_joints` would
            # not crash — it would just never look at it, which is the kind of dead key that
            # later reads as "this mode trains on joints too".
            out["wb"] = {"state": "wb_state_joints", "state_valid": "wb_state_valid"}
            if self.layout.is_wholebody:
                out["wb"].update({"action": "wb_action_joints",
                                  "action_valid": "wb_action_valid"})
        return out

    # -- episode bounds ----------------------------------------------------
    def _episode_bounds(
        self,
        root,
        n_frames: int,
        warn: bool = False,
        strict: bool = False,
        return_valid: bool = False,
    ):
        """``(starts, ends)`` frame indices partitioning ``[0, n_frames)`` per episode.

        A file without a usable ``meta/episode_ends`` is treated as ONE episode. Shared
        by the glitch scan and the obs-history/episode-start mapping so they can never
        disagree about where a demonstration begins (``episode_timestamps`` is RELATIVE
        and restarts each episode, so a wrong boundary corrupts both). ``warn`` logs the
        fallback — the glitch scan sets it, since a collapsed multi-episode file makes
        its relative-time window math silently wrong."""
        valid = True
        try:
            ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64).reshape(-1)
        except Exception:  # noqa: BLE001 — treat a missing/broken meta as one episode
            valid = False
            ends = np.asarray([n_frames], dtype=np.int64)
        if ends.size == 0 or int(ends[-1]) != n_frames or int(ends[0]) <= 0 or np.any(np.diff(ends) <= 0):
            valid = False
        if not valid:
            if strict:
                raise ValueError(
                    "HLDataset: validation splitting requires meta/episode_ends to be a "
                    f"strictly increasing, non-empty partition ending at n_frames={n_frames}; "
                    f"got {ends.tolist() if ends.size else []}."
                )
            if warn:
                logger.warning(
                    "HLDataset: episode_ends missing or not cumulative to n_frames=%d "
                    "(got %s); per-episode scan falls back to ONE episode. If this file "
                    "really has multiple episodes, the result may be wrong — re-convert it.",
                    n_frames,
                    ends.tolist() if ends.size else "[]",
                )
            ends = np.asarray([n_frames], dtype=np.int64)
        starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
        if return_valid:
            return starts, ends, valid
        return starts, ends

    def _episode_start_frames(
        self,
        root,
        n_frames: int,
        starts: np.ndarray | None = None,
        ends: np.ndarray | None = None,
    ) -> np.ndarray:
        """``[n_frames]`` int: for each frame, the FIRST frame index of its episode.

        Used to read every keypoint's EPISODE-START orientation (the ``wrt_start``
        reference) and to clamp the obs-history look-back to the episode's own frames.
        Only ``robot0_demo_start_pose`` is on disk (primary hand), but each keypoint's
        start rotation is just its own rot array AT this index, so this integer map is
        all the dataset needs — no converter change, works for every mode."""
        if starts is None or ends is None:
            starts, ends = self._episode_bounds(root, n_frames)
        out = np.zeros(n_frames, dtype=np.int64)
        for a, b in zip(starts, ends):
            out[int(a) : int(b)] = int(a)
        return out

    def _history_frames(self, fi: int, frame: int) -> np.ndarray:
        """``[To]`` frame indices for the obs history at ``frame``, oldest-first.

        Time-anchored (DECISION-OBS2): frame k targets ``t0 - (To-1-k)*obs_history_dt_s``
        and resolves to the NEAREST camera frame at or before that time WITHIN the
        anchor's own episode. Before the episode's first frame it clamps to that first
        frame (UMI's start-pad: the oldest slots repeat frame 0). The last slot is always
        ``frame`` itself (t0). To == 1 returns ``[frame]``."""
        To = self.obs_history_len
        if To == 1:
            return np.asarray([frame], dtype=np.int64)
        ts = self._ts[fi]
        ep_start = int(self._ep_start_frame[fi][frame])
        t0 = float(ts[frame])
        # Search only within this episode's frames (relative timestamps restart each
        # episode, so a global searchsorted would cross boundaries).
        ep_ts = ts[ep_start : frame + 1]
        out = np.empty(To, dtype=np.int64)
        for k in range(To):
            back = (To - 1 - k) * self.obs_history_dt_s
            target = t0 - back
            # rightmost episode-local index whose time is <= target; clamp to episode start.
            j = int(np.searchsorted(ep_ts, target, side="right")) - 1
            out[k] = ep_start + max(0, j)
        return out

    # -- glitch-window drop ------------------------------------------------
    def _glitch_window_keep(self, root, data, n_frames: int) -> np.ndarray:
        """``[n_frames]`` bool: False for anchors whose sample window straddles a
        tracker glitch/dropout. See the module docstring.

        Derived entirely from the converter's own per-step ``hl_<role>_action_valid``
        and per-frame ``robot0_pose_glitch`` — nothing is re-thresholded here, so this
        filter can never disagree with the mask the loss already applies. The scan is
        per episode because ``episode_timestamps`` is RELATIVE (restarts each episode),
        and a window must never see a glitch from a neighbouring demonstration.
        """
        keep = np.ones(n_frames, dtype=bool)
        starts, ends = self._episode_bounds(root, n_frames, warn=True)

        ts = np.asarray(data["episode_timestamps"][:], dtype=np.float64).reshape(-1)
        # Per-step END times relative to the anchor. T0-INCLUSIVE grid (DECISION-T0):
        # step k lands at t0 + k*dt (k=0..H-1, step 0 ON the anchor), and step k's bin
        # spans (t_k - dt, t_k] — so its END time is t0 + k*dt, matching the converter's
        # future_step_times. The pre-v2 grid used (k+1)*dt (t0 excluded); using the stale
        # offset here would shift every glitch interval one dt late.
        step_end_off = ACTION_DT * np.arange(0, ACTION_HORIZON, dtype=np.float64)

        # Every role whose invalidity signals a corrupted POSE. force_roles are excluded:
        # a masked force step is a sensor-coverage/tare issue, not a pose glitch, and is
        # handled by force_history_valid / the per-step force mask, not by dropping the
        # whole trainable pose sample.
        # `tracker_roles`, not `layout.roles`. Identical for every keypoint mode; for
        # `wholebody` this is what keeps the CHEST puck in the scan — a chest excursion
        # corrupts the completion's root conditioning and therefore all 29 joint columns,
        # while no chest ROLE exists in that layout to be iterated.
        pose_valid_keys = [f"hl_{role}_action_valid" for role in self.tracker_roles]
        # ``[:, :ACTION_HORIZON]`` for the same reason __getitem__ truncates: the file may
        # store a LONGER chunk than this build trains on. It must be the same prefix, or
        # this filter would drop anchors for glitches in steps the loss never sees — and
        # the (Te, stored_H) validity would not even broadcast against the (Te, H) grid
        # built from step_end_off below.
        # The whole-body joint targets get the SAME treatment: a step the completion did not
        # cover (a trimmed episode, a window the backend never ran) is as untrustworthy a
        # target as a tracker dropout, and per-step masking cannot repair the anchor.
        # The whole-body joint targets AND the 5-point family's derived FEET both live or die
        # by `wb_action_valid`: the feet are FK'd from those very joints, so a step the
        # completion did not cover has no foot pose either. `state_is_geometric` is the common
        # condition (every mode that reads the completion at all), which keeps a future family
        # from having to remember to add itself here.
        if self.layout.state_is_geometric:
            pose_valid_keys.append("wb_action_valid")
        pose_valids = [np.asarray(data[k][:], dtype=np.float32)[:, :ACTION_HORIZON] > 0.5 for k in pose_valid_keys]
        # Corrupted-anchor frames. The converter writes `robot0_pose_glitch` for the PRIMARY
        # hand and, since 2026-09-11, a `<role>_pose_glitch` per emitted tracker role: a
        # right/chest/foot excursion corrupts THAT role's anchor pose exactly as a primary
        # one does, and used to be answered only by `--exclude`-ing the whole take. Take the
        # UNION over every tracker role this layout reads (an absent per-role mask simply
        # contributes nothing, so a pre-2026-09-11 file reads exactly as before).
        glitch_frame = None
        glitch_keys = ["robot0_pose_glitch"] + [f"{role}_pose_glitch" for role in self.tracker_roles]
        for key in glitch_keys:
            if key not in data:
                continue
            g = np.asarray(data[key][:], dtype=np.float32).reshape(-1) > 0.5
            glitch_frame = g if glitch_frame is None else (glitch_frame | g)

        for a, b in zip(starts, ends):
            a, b = int(a), int(b)
            if b <= a:
                continue
            rt = ts[a:b]  # relative anchor times, restart at ~0
            ep_end = float(rt[-1])
            # Corrupted time instants in this episode, as [lo, hi] intervals.
            lo_list: list[np.ndarray] = []
            hi_list: list[np.ndarray] = []
            step_end = rt[:, None] + step_end_off[None, :]  # (Te, H) absolute step ends
            step_lo = step_end - ACTION_DT
            not_tail = step_end <= ep_end + 1e-9  # a step past ep_end is padding
            for pv in pose_valids:
                bad = (~pv[a:b]) & not_tail  # invalid AND not tail padding
                if bad.any():
                    lo_list.append(step_lo[bad])
                    hi_list.append(step_end[bad])
            if glitch_frame is not None:
                pg = glitch_frame[a:b]
                if pg.any():  # point intervals at the frame
                    lo_list.append(rt[pg])
                    hi_list.append(rt[pg])
            if not lo_list:
                continue
            bad_lo = np.concatenate(lo_list)
            bad_hi = np.concatenate(hi_list)
            # Drop anchor i if its window [rt_i - pre, rt_i + post] overlaps any bad
            # interval. Overlap (not containment): the sample is only as clean as its
            # worst touched sample.
            win_lo = rt - self.glitch_pre_s
            win_hi = rt + self.glitch_post_s
            drop = ((bad_lo[None, :] <= win_hi[:, None]) & (bad_hi[None, :] >= win_lo[:, None])).any(axis=1)
            keep[a:b] &= ~drop
            # Observability: a whole demonstration vanishing from training is a much
            # louder event than the per-file total conveys (a systemically-glitched take
            # drops every anchor while the file still reports usable>0). Warn per episode.
            if drop.all() and (b - a) > 0:
                logger.warning(
                    "HLDataset: episode [%d:%d] lost ALL %d anchors to the glitch-window "
                    "filter — this demonstration contributes nothing to training. Inspect "
                    "the take (systemic tracker glitch?) or set drop_glitch_windows=False.",
                    a,
                    b,
                    b - a,
                )
        return keep

    # -- handles -----------------------------------------------------------
    @staticmethod
    def _open(path: str):
        import zarr

        _register_image_codecs()
        store = zarr.ZipStore(path, mode="r")
        return zarr.group(store=store), store

    def _root(self, fi: int):
        pid = os.getpid()
        if self._handle_pid != pid:
            # A fork happened (or first use): never reuse an inherited handle.
            self._roots = {}
            self._stores = {}
            self._handle_pid = pid
        root = self._roots.get(fi)
        if root is None:
            root, store = self._open(self.zarr_paths[fi])
            try:
                actual = _file_identity(self.zarr_paths[fi])
                expected = self._source_file_identities[fi]
                if actual != expected:
                    raise ValueError(
                        f"{self.zarr_paths[fi]} changed after HLDataset indexed it: "
                        f"expected file identity {expected}, got {actual}"
                    )
            except Exception:
                store.close()
                raise
            self._roots[fi] = root
            self._stores[fi] = store
        return root

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_roots"] = {}
        state["_stores"] = {}
        state["_handle_pid"] = None
        return state

    def __len__(self) -> int:
        return len(self._index)

    @property
    def file_attrs(self) -> tuple[MappingProxyType, ...]:
        """Read-only root attrs per source zarr, in ``zarr_paths`` order.

        Provenance for consumers that must derive rig geometry from the DATASET rather than
        hardcode it (the offline viewer reads each file's ``T_tracker_tcp`` puck->TCP
        extrinsic this way, since it differs between datasets). Wrapped read-only so a
        consumer cannot mutate the cached attrs a later reader would see."""
        return tuple(MappingProxyType(attrs) for attrs in self._file_attrs)

    def split_indices(self, val_last_episodes: int = 4) -> tuple[list[int], list[int]]:
        """Return sample indices for an episode-disjoint train/validation split.

        Episode IDs are global flattened IDs: all episodes from zarr file 0, then file 1,
        and so on. The final ``val_last_episodes`` IDs are validation. No frame from a
        validation episode can therefore appear in the train list.
        """
        n_val = int(val_last_episodes)
        if n_val < 0 or n_val >= self.episode_count:
            raise ValueError(f"val_last_episodes must be in [0, {self.episode_count - 1}], got {n_val}")
        if n_val and not all(self._episode_ends_valid):
            raise ValueError("validation splitting requires valid meta/episode_ends in every zarr file")
        first_val = self.episode_count - n_val
        train = [i for i, ep in enumerate(self.sample_episode_ids) if ep < first_val]
        val = [i for i, ep in enumerate(self.sample_episode_ids) if ep >= first_val]
        if n_val and (not train or not val):
            raise ValueError(f"episode split produced train={len(train)} and val={len(val)} usable anchors")
        return train, val

    def split_provenance(self, val_last_episodes: int = 4) -> dict:
        """JSON-serializable identity of the episode split and anchor-affecting knobs.

        ``dataset_knobs`` is compared for EXACT equality against a norm-stats file's own
        record (``trainer.validate_split_provenance``), so a key added here invalidates
        every stats file written before it. ``image_keys`` is therefore recorded ONLY when
        there is more than one camera: a single-camera run's provenance stays byte-identical
        to what every existing ``norm_stats_hl_*.json`` carries, and the multi-camera case —
        which has no pre-existing stats to be compatible with — records the full list.
        """
        n_val = int(val_last_episodes)
        train, val = self.split_indices(n_val)
        first_val = self.episode_count - n_val
        extra_knobs: dict = {}
        if len(self.image_keys) > 1:
            extra_knobs["image_keys"] = list(self.image_keys)
        # Recorded only for `wholebody` (None everywhere else), so every existing
        # norm_stats_hl_*.json keeps comparing equal. It belongs in the EXACT-equality block
        # because the anchor decides what the absolute joint values mean: stats fitted over a
        # `home`-anchored export describe a different posture space than an `absolute` one, and
        # the widths are identical.
        if self._wb_cond_mode is not None:
            extra_knobs["wb_cond_mode"] = self._wb_cond_mode
        # Recorded only when it is NOT the shipped default, the same compatibility rule
        # `image_keys` follows above: every norm_stats_hl_wb_*.json written before this knob
        # existed describes an `absolute` target and must keep comparing equal. It belongs in the
        # exact-equality block because the two targets differ by q[0] -- a per-sample offset --
        # so a delta run's per-column quantiles are a completely different scale at an identical
        # width, and clip_sample would crush the mismatched one silently.
        if self.wb_joint_target != "absolute":
            extra_knobs["wb_joint_target"] = self.wb_joint_target
        # The HORIZON, recorded only when it is NOT the contract default — the same
        # compatibility rule `image_keys` and `wb_joint_target` follow above, so every
        # norm_stats_hl_*.json written before this key existed keeps comparing equal.
        #
        # It belongs in the exact-equality block because the fitted QUANTILES depend on it and
        # nothing else here would notice: a 200-step chunk reaches further from its anchor than
        # a 50-step one, so its per-column q01/q99 are wider — at an IDENTICAL width. Crossed,
        # the values load, `_check_norm_stats` passes, and diffusion's `clip_sample=True` then
        # clamps every genuinely out-of-range joint/force to a stale range on every DDIM step.
        # (This closes the hole the earlier H=30-vs-H=50 A/B ran with. Those two runs' stats
        # were fitted separately and named separately, but nothing enforced it.)
        if ACTION_HORIZON != ACTION_HORIZON_DEFAULT:
            extra_knobs["action_horizon"] = int(ACTION_HORIZON)
        return {
            "transform_contract_version": HL_TRANSFORM_CONTRACT_VERSION,
            "strategy": "last_global_flattened_episodes",
            "val_last_episodes": n_val,
            "total_episodes": self.episode_count,
            "train_episode_ids": list(range(first_val)),
            "val_episode_ids": list(range(first_val, self.episode_count)),
            "episode_identities": [list(x) for x in self.episode_identities],
            "episode_ends_fingerprint": self.episode_ends_fingerprint,
            "anchor_indices_fingerprint": self.anchor_indices_fingerprint,
            "source_content_fingerprint": self.source_content_fingerprint,
            "train_anchor_count": len(train),
            "val_anchor_count": len(val),
            "dataset_knobs": {
                # Keep paths portable across workstations. The ordered episode metadata
                # fingerprint below identifies the split contents; absolute checkout paths
                # would make an otherwise identical W&B resume artifact unusable elsewhere.
                "zarr_paths": [Path(p).as_posix() for p in self.zarr_paths],
                "mode": self.mode,
                "image_key": self.image_key,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
                "drop_invalid_force_history": self.drop_invalid_force_history,
                "drop_glitch_windows": self.drop_glitch_windows,
                "glitch_pre_s": self.glitch_pre_s,
                "glitch_post_s": self.glitch_post_s,
                "obs_history_len": self.obs_history_len,
                "obs_history_dt_s": self.obs_history_dt_s,
                **extra_knobs,
            },
        }

    def _forward_tail_span(self, file_index: int, file_frame: int) -> tuple[int, int, int]:
        """``(fi, frame, episode_end)`` for the recorded tail starting at ``file_frame``.

        Shared bounds resolution for :meth:`episode_forward_positions`,
        :meth:`episode_forward_poses` and :meth:`episode_image` so they can never disagree
        about where the tail begins or which episode boundary stops it. ``file_index`` /
        ``file_frame`` are the raw ``(fi, frame)`` anchor identity from an
        :class:`HLSampleSource` ``SampleRef``, NOT a dataset sample index — the two diverge
        once glitch windows are dropped."""
        fi = int(file_index)
        if not 0 <= fi < len(self.zarr_paths):
            raise IndexError(f"file_index {fi} is outside [0, {len(self.zarr_paths) - 1}]")
        ends = np.asarray(self._episode_ends_by_file[fi], dtype=np.int64)
        n_frames = int(ends[-1])
        frame = int(file_frame)
        if not 0 <= frame < n_frames:
            raise IndexError(f"file_frame {frame} is outside [0, {n_frames - 1}] for file {fi}")
        # Frames of episode e are [starts[e], ends[e]); searchsorted(side="right") matches
        # how the index assigns sample_episode_ids, so the tail stops at this episode's end.
        episode_end = int(ends[int(np.searchsorted(ends, frame, side="right"))])
        return fi, frame, episode_end

    def episode_forward_positions(self, file_index: int, file_frame: int) -> dict[str, np.ndarray]:
        """Absolute aligned-world EE positions from ``file_frame`` to the END of its
        episode, one ``(N, 3)`` array per layout role (oldest-first, ``N >= 1``).

        This is the RECORDED remainder of the demonstration the anchor at ``file_frame``
        was sampled from, read straight off the camera-grid ``*_eef_pos`` / ``head_pos``
        arrays — the SAME absolute frame the decoded action chunk lives in (a chunk is
        ``R_ref^T (p - p_ref)`` about this frame's own anchor and step 0 IS that anchor,
        so ``episode_forward_positions[role][0]`` equals the chunk's step-0 target), which
        lets a viewer draw it continuous with the sample's target line. Position only: the
        z-down/z-up rot6d convention is orientation-only and does not touch position.

        The tail is the RAW recording — it is NOT glitch-masked. The sampled anchor's own
        ``[t0 - pre, t0 + post]`` window is clean (that is what ``drop_glitch_windows``
        guarantees), but a tracker excursion further downstream stays in the line, on
        purpose: this overlay is for reading the demonstration's overall shape, not for
        training. The scan uses ``meta/episode_ends`` so the tail never crosses a
        demonstration boundary.
        """
        fi, frame, episode_end = self._forward_tail_span(file_index, file_frame)
        data = self._root(fi)["data"]
        keys = self._keys[fi]
        out: dict[str, np.ndarray] = {}
        for role in self.layout.roles:
            pos_key = keys["roles"][role]["anchor_pos"]
            tail = np.asarray(data[pos_key][frame:episode_end], dtype=np.float64)
            out[role] = tail.reshape(tail.shape[0], 3)
        return out

    def episode_forward_poses(self, file_index: int, file_frame: int) -> tuple[np.ndarray, dict[str, dict]]:
        """Recorded absolute poses (+ gripper width) from ``file_frame`` to episode end.

        The pose-and-gripper superset of :meth:`episode_forward_positions`, for a viewer
        that PLAYS the recorded demonstration forward rather than just drawing its path.
        Returns ``(times_s, per_role)`` where:

        * ``times_s`` is ``(N,)`` seconds relative to ``file_frame``, read from
          ``episode_timestamps`` (RELATIVE, restarts each episode), so playback can honour
          the real camera grid (~15 Hz, mildly non-uniform) instead of assuming a fixed dt.
          ``times_s[0] == 0``.
        * ``per_role[role]`` is ``{"position": (N, 3), "rot6d": (N, 6), "gripper_m":
          (N,) | None, "wrench": (N, 6) | None, "wrench_valid": (N,) | None}`` in the SAME
          absolute aligned-world frame as :meth:`episode_forward_positions` and the decoded
          action chunk. Row 0 is the sample's own anchor (step-0 target), so a viewer can
          rebase the whole tail against that one reference exactly as it rebases the action
          chunk. ``gripper_m`` is the recorded ``*_gripper_width`` in metres for gripper
          roles, ``None`` otherwise — it drives the mesh jaw during playback (there is no
          prediction to drive it).

        ``wrench`` is each frame's OWN measured wrench, taken as step 0 of that frame's
        ``hl_<role>_action_force`` chunk rather than from the camera-grid
        ``robot0_ft_wrench``. Both describe the same instant, but the action array is the one
        already resampled onto the 30 Hz action grid and paired with a per-step validity mask,
        so a playback arrow shows exactly the quantity the model is scored against instead of
        a second, subtly different resampling of it. It stays in the **bota_sensor** frame like
        everything else on disk; rotating it into TCP axes for display is
        :mod:`flore.utils.force_frame`'s job, not the loader's. ``None`` for a role with no
        F/T sensor.

        Same RAW-recording caveat as :meth:`episode_forward_positions`: NOT glitch-masked
        beyond the anchor's own clean window, and the tail never crosses an episode
        boundary (``meta/episode_ends``)."""
        fi, frame, episode_end = self._forward_tail_span(file_index, file_frame)
        data = self._root(fi)["data"]
        keys = self._keys[fi]
        # Relative camera times of the tail, re-based to 0 at the play-start frame. self._ts
        # is per-file and RELATIVE (already restarts each episode), so slicing the tail and
        # subtracting its first entry cannot leak a neighbouring episode's clock.
        ts = self._ts[fi][frame:episode_end]
        times_s = (ts - ts[0]).astype(np.float64) if ts.size else np.zeros(0, dtype=np.float64)
        per_role: dict[str, dict] = {}
        for role in self.layout.roles:
            k = keys["roles"][role]
            pos = np.asarray(data[k["anchor_pos"]][frame:episode_end], dtype=np.float64).reshape(-1, 3)
            rotvec = np.asarray(data[k["anchor_rot"]][frame:episode_end], dtype=np.float64).reshape(-1, 3)
            entry: dict = {"position": pos, "rot6d": rotvec_to_rot6d(rotvec)}
            grip_key = k.get("gripper_obs")
            if grip_key is not None and role in self.layout.gripper_roles:
                entry["gripper_m"] = np.asarray(data[grip_key][frame:episode_end], dtype=np.float64).reshape(-1)
            else:
                entry["gripper_m"] = None
            # Step 0 of each frame's own action chunk == that frame's measured wrench, on the
            # action grid and with its own validity bit (see the docstring on why not
            # robot0_ft_wrench). Sliced as [tail, 0] so one read covers the whole tail.
            if "force" in k and role in self.layout.force_roles:
                wrench = np.asarray(data[k["force"]][frame:episode_end, 0], dtype=np.float64)
                valid = np.asarray(data[k["force_valid"]][frame:episode_end, 0], dtype=np.float64)
                entry["wrench"] = wrench.reshape(-1, FORCE_DIM)
                entry["wrench_valid"] = valid.reshape(-1) > 0.5
            else:
                entry["wrench"] = None
                entry["wrench_valid"] = None
            per_role[role] = entry
        return times_s, per_role

    def episode_image(self, file_index: int, file_frame: int) -> np.ndarray:
        """Raw ``(H, W, 3)`` uint8 camera frame at ``(file_index, file_frame)``.

        The undecorated recording (NOT the normalized model input), for a viewer to show
        the observation while it plays the demonstration forward. Bounds are the same
        ``(fi, frame)`` anchor identity the other tail readers use; ``file_frame`` here is
        an ABSOLUTE frame within the episode's ``[frame, episode_end)`` span, so a playback
        cursor can read any frame it has stepped to."""
        fi, frame, episode_end = self._forward_tail_span(file_index, file_frame)
        # _forward_tail_span already bounds frame to a real episode; episode_end is the
        # exclusive end, and frame < episode_end holds because frame is a valid in-episode
        # anchor, so this read is always in range. The array yields (H, W, 3) directly, so
        # the image dimensions are not hardcoded here.
        del episode_end
        data = self._root(fi)["data"]
        return np.asarray(data[self.image_key][frame], dtype=np.uint8)

    # -- sample ------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        fi, frame = self._index[idx]
        data = self._root(fi)["data"]
        keys = self._keys[fi]
        lay = self.layout
        H, AD = ACTION_HORIZON, lay.action_dim
        To, SD = self.obs_history_len, lay.state_dim
        hist_frames = self._history_frames(fi, frame)  # (To,) oldest-first, last == frame
        ep_start = int(self._ep_start_frame[fi][frame])
        # A glitch AT the episode-start frame corrupts the wrt_start REFERENCE, biasing
        # the orientation channel for every anchor in the episode. Mark those wrt_start
        # slots invalid so a consumer gates rather than trusts a poisoned reference.
        start_ref_bad = bool(self._ep_start_glitched[fi][frame])
        state_slots = lay.state_slots()

        action = np.zeros((H, AD), dtype=np.float64)
        mask = np.zeros((H, AD), dtype=bool)
        # State is a To-frame history (DECISION-OBS2): (To, state_dim), oldest-first. Per
        # keypoint it carries wrt_start_rot6d (+ gripper for gripper roles); see
        # layout.state_slots. state_valid mirrors it, False where a value is fabricated.
        state = np.zeros((To, SD), dtype=np.float64)
        state_valid = np.ones((To, SD), dtype=bool)
        force_hist = np.zeros((FORCE_HIST_LEN, lay.force_hist_channels), dtype=np.float64)
        # PER F/T SENSOR, one bool each, in ``layout.force_roles`` order — the same order
        # the channel blocks of ``force_hist`` are written in. A shared scalar would let one
        # hand's sensor gap discard the OTHER hand's measured wrench, and with the encoder
        # substituting a learned "no data" embedding for an invalid window that is a
        # silent, total loss of the good half. Kept as a length-1 vector for a one-sensor
        # rig too, then collapsed to a scalar at return so the shipped single-hand contract
        # (and every serving/eval consumer built against it) is untouched.
        hist_valid = np.ones(max(1, len(lay.force_roles)), dtype=bool)
        anchor_pos = np.zeros((len(lay.roles), 3), dtype=np.float64)
        anchor_rot6d = np.zeros((len(lay.roles), 6), dtype=np.float64)

        # --- WHOLE-BODY: the 29 joint columns + the validity every other column inherits ---
        # `wb_step_valid` stands in for the per-role pose validity of the keypoint modes, and
        # it is the AND over BOTH signals because the two failures are independent: the
        # completion may not cover a step (`wb_action_valid`) and a tracker may have dropped
        # out under it, in which case the completion ran on interpolated conditioning and its
        # joints are fabricated at full loss weight. Every TRACKER role is included — the
        # chest most of all, since it is the completion's root channel and owns no block here.
        wb_step_valid: np.ndarray | None = None
        if lay.is_wholebody:
            wbk = keys["wb"]
            wb_step_valid = np.asarray(data[wbk["action_valid"]][frame], dtype=np.float32)[:H] > 0.5
            for role in self.tracker_roles:
                rv = np.asarray(data[keys["roles"][role]["valid"]][frame], dtype=np.float32)[:H]
                wb_step_valid &= rv > 0.5
            # `[:H]` for the same prefix reason the pose targets truncate: the file may store a
            # LONGER chunk than this build trains on, and on the t0-inclusive grid at a fixed
            # dt the first H rows ARE this horizon's labels.
            wb_act = np.asarray(data[wbk["action"]][frame], dtype=np.float64)[:H]
            joint_valid = wb_step_valid
            if self.wb_joint_target == "delta":
                # ANCHOR-RELATIVE in joint space: q[k] - q[0], so row 0 is exactly zero and the
                # chunk is pure CHANGE (layout.WB_JOINT_TARGETS). The anchor is the chunk's OWN
                # first row rather than `wb_state_joints`, even though the exporter makes them
                # equal to 0.000e+00 -- it is the same quantity the deploy-side JointClutch
                # subtracts (`q - q[0]`), so training and serving cannot disagree about which
                # row the offset came from.
                #
                # ⚠️ Row 0 is the WHOLE chunk's reference, so an uncovered anchor step poisons
                # every row rather than one: `wb_action_joints` is zero-filled where the
                # completion never reached, and 0 rad is a legal posture, so the values cannot
                # say so. AND the per-step mask with row 0's own validity -- absolute mode is
                # unaffected because there each row stands alone. This is a SECOND line: with
                # `drop_glitch_windows` on (the default) such an anchor is dropped before it gets
                # here, so what this covers is every run that turns that knob off.
                wb_act = wb_act - wb_act[0]
                joint_valid = wb_step_valid & bool(wb_step_valid[0])
            action[:, lay.joints] = wb_act
            mask[:, lay.joints] = joint_valid[:, None]

        # --- GEOMETRIC state: the anchor frame's own configuration, per obs-history frame -----
        # ABSOLUTE (no wrt_start reference), so a glitch at the episode-start frame is
        # irrelevant to it. Either the 29 raw joints (`wholebody`), their pelvis-frame geometric
        # reduction (`wholebody_geom_*` / `fivepoint_*`, precomputed in __init__), or nothing but
        # the jaws (`*_geom_none`) — every variant derived from `wb_state_joints` and gated by
        # the SAME `wb_state_valid`, so the modes differ in width and units but never in which
        # frames they trust.
        #
        # OUTSIDE the `is_wholebody` branch above, because the `fivepoint_*` family has this
        # state with a KEYPOINT action. That split is the whole reason `state_is_geometric`
        # exists; while these two blocks were one, a geometric state implied a joint action.
        if lay.state_is_geometric:
            wbk = keys["wb"]
            jslot = lay.joint_state_slot
            gslots = lay.state_geom_slots()
            for hk, hf in enumerate(hist_frames):
                if jslot is not None:
                    state[hk, jslot] = np.asarray(data[wbk["state"]][int(hf)], dtype=np.float64)
                for block in self._geom_blocks:
                    state[hk, gslots[block]] = self._wb_geom[fi][
                        int(hf), self._geom_src_slots[block]]
                covered = bool(
                    np.asarray(data[wbk["state_valid"]][int(hf)], dtype=np.float32).reshape(-1)[0] > 0.5)
                if not covered:
                    # A frame the completion never reached holds whatever the exporter
                    # zero-filled, and 0 rad is a legal posture (the 29-joints-zero pose), so
                    # the value cannot say it is fabricated. Identical for the geometric blocks:
                    # zero there is "both hands at the pelvis", equally plausible and equally
                    # fabricated.
                    if jslot is not None:
                        state_valid[hk, jslot] = False
                    for block in self._geom_blocks:
                        state_valid[hk, gslots[block]] = False

        for ri, role in enumerate(lay.roles):
            k = keys["roles"][role]
            slots = lay.slots(role)
            sslot = state_slots[role]

            # --- anchor pose at t0 (single-frame reads only) ---------------
            # Per-frame indexing, NOT the whole-column `data[key][:]` the pre-v2
            # dataset used: a full read of an (T,16,6) f32 array would be ~13 MB per
            # sample, per key.
            a_pos = np.asarray(data[k["anchor_pos"]][frame], dtype=np.float64)
            a_rot6d = rotvec_to_rot6d(np.asarray(data[k["anchor_rot"]][frame], dtype=np.float64))
            anchor_pos[ri] = a_pos
            anchor_rot6d[ri] = a_rot6d

            # --- wrt_start orientation, per obs-history frame (state) ----------
            # TWO INDEPENDENT QUESTIONS, and they used to be one `if`. "Does this keypoint
            # write a wrt_start orientation into the state?" is answered by
            # `state_is_geometric`; "does it have pose TARGETS?" is answered by whether the
            # action is a joint configuration. Every mode but `fivepoint_*` answers them the
            # same way, which is exactly why collapsing them held for so long — a 5-point
            # layout has a geometric state AND pose targets, and under the old single branch it
            # would have written five rot6d blocks into a 14-wide vector.
            if not lay.state_is_geometric:
                # This keypoint's orientation at each history frame, relative to its
                # EPISODE-START orientation (UMI's _wrt_start; rotation only, frame-invariant
                # so OOD-safe). ref = the same rot array AT the episode's first frame.
                start_rot6d = rotvec_to_rot6d(np.asarray(data[k["anchor_rot"]][ep_start], dtype=np.float64))
                for hk, hf in enumerate(hist_frames):
                    cur_rot6d = rotvec_to_rot6d(np.asarray(data[k["anchor_rot"]][int(hf)], dtype=np.float64))
                    state[hk, sslot["wrt_start_rot6d"]] = rot6d_relative(cur_rot6d, start_rot6d)
                # A corrupted episode-start reference poisons this channel for the whole
                # episode — flag it (value kept for shape stability; consumer gates on valid).
                if start_ref_bad:
                    state_valid[:, sslot["wrt_start_rot6d"]] = False

            if wb_step_valid is not None:
                # WHOLE-BODY: no pose targets — the joints ARE the target and
                # `slots.pos`/`slots.rot6d` are deliberately 0-width. The per-step validity the
                # gripper/force columns AND against is the joint one, so a step the body cannot
                # reach is not supervised as a gripper or force label either — the same
                # asymmetry the keypoint modes apply with the pose mask.
                step_valid = wb_step_valid
            else:
                # --- pose targets -> relative to THIS keypoint's own anchor ----
                # ``[:H]`` is the PREFIX TRUNCATION that lets one zarr serve several horizons
                # (FLORE_ACTION_HORIZON): the file stores hl_action_horizon >= H steps on this
                # build's dt (asserted in _check_file), and on the t0-inclusive grid step k is
                # at t0 + k*dt regardless of how many steps follow it, so the first H rows ARE
                # this horizon's labels bit-for-bit. It is a no-op when stored == H.
                pose = np.asarray(data[k["pose"]][frame], dtype=np.float64)[:H]  # (H, 6)
                rel_pos, rel_rot6d = absolute_to_delta_ee(
                    pose[:, :3], rotvec_to_rot6d(pose[:, 3:]), a_pos, a_rot6d)
                action[:, slots.pos] = rel_pos
                action[:, slots.rot6d] = rel_rot6d

                step_valid = np.asarray(data[k["valid"]][frame], dtype=np.float32)[:H] > 0.5  # (H,)
                mask[:, slots.pos] = step_valid[:, None]
                mask[:, slots.rot6d] = step_valid[:, None]

            # --- gripper: ABSOLUTE in both state and targets ---------------
            # Same reserve-but-mask discipline as force below: a hand block always
            # owns a gripper column, but only ``layout.gripper_roles`` have a physical
            # gripper. A sensorless hand's column stays zero and masked FALSE rather
            # than training toward a constant.
            if slots.gripper is not None and role in lay.gripper_roles:
                gw = np.asarray(data[k["gripper"]][frame], dtype=np.float64)[:H]  # (H,) metres
                action[:, slots.gripper] = gripper_width_to_norm(gw)
                # A take whose encoder stream died is zero-filled AND flagged; without
                # the mask its constant 0 would train as a real "fully closed".
                g_valid = np.asarray(data[k["gripper_valid"]][frame], dtype=np.float32)[:H] > 0.5
                mask[:, slots.gripper] = step_valid & g_valid

                # The OBSERVED opening at each history frame, into the state's gripper
                # slot. robot0_gripper_width has no companion mask and the converter
                # ZERO-FILLS it when gripper_range.json is absent (a stderr warning only)
                # — and 0.0 is a legal width (fully closed), so a fabricated observation
                # is invisible in the values. Each history frame uses that frame's
                # t0-inclusive step-0 validity as the proxy for encoder coverage; real
                # recordings contain occasional validity transitions within To.
                gcol = sslot["gripper"]
                for hk, hf in enumerate(hist_frames):
                    history_valid = bool(
                        np.asarray(data[k["gripper_valid"]][int(hf)], dtype=np.float32).reshape(-1)[0] > 0.5
                    )
                    if history_valid:
                        obs_w = np.asarray(data[k["gripper_obs"]][int(hf)], dtype=np.float64).reshape(-1)[0]
                        state[hk, gcol] = gripper_width_to_norm(obs_w)
                    else:
                        state_valid[hk, gcol] = False

            # --- force: history (input) + targets (output) -----------------
            # Every HAND block RESERVES force columns so that adding a second F/T
            # sensor later moves no checkpoint column. But only the roles in
            # ``layout.force_roles`` actually carry a sensor on this rig, so a
            # reserved-but-sensorless hand's columns stay ZERO and masked FALSE —
            # never supervised toward a fabricated 0, which would teach the policy
            # that the other hand always feels nothing.
            if slots.force is not None and role in lay.force_roles:
                fi_role = lay.force_roles.index(role)
                ch = slice(fi_role * FORCE_DIM, (fi_role + 1) * FORCE_DIM)
                force_hist[:, ch] = np.asarray(data[k["force_hist"]][frame], dtype=np.float64)
                hist_valid[fi_role] = bool(
                    np.asarray(data[k["force_hist_valid"]][frame], dtype=np.float32).reshape(-1)[0] > 0.5
                )

                action[:, slots.force] = np.asarray(data[k["force"]][frame], dtype=np.float64)[:H]
                f_valid = np.asarray(data[k["force_valid"]][frame], dtype=np.float32)[:H] > 0.5
                # Force needs BOTH: inside the episode AND a measured, tare-clean bin.
                mask[:, slots.force] = (step_valid & f_valid)[:, None]

        # Images: every camera x every history frame, CAMERA-MAJOR and oldest-first within
        # a camera. One camera + To == 1 collapses to the single-frame (C, H, W) contract;
        # anything more yields (n_cams*To, C, H, W) — both handled by collate_hl_batch and
        # the model's pseudo-camera path, which sizes global_cond as num_cameras*To*features
        # and so depends on THIS order.
        images = np.stack([self._load_image(data, key, int(hf))
                           for key in self.image_keys for hf in hist_frames], axis=0)
        image = images[0] if images.shape[0] == 1 else images

        return {
            "image": torch.from_numpy(image).float(),
            "state": torch.from_numpy(state[0] if To == 1 else state).float(),
            "state_valid": torch.from_numpy(state_valid[0] if To == 1 else state_valid),
            "force_history": torch.from_numpy(force_hist).float(),
            # Scalar for a one-sensor layout (the historical contract), (n_ft,) for more.
            # Same "the shape follows the hardware" convention as `image`/`state` above.
            "force_history_valid": (torch.tensor(bool(hist_valid[0]))
                                    if hist_valid.size == 1
                                    else torch.from_numpy(hist_valid)),
            "action_chunk": torch.from_numpy(action).float(),
            "action_mask": torch.from_numpy(mask),
            # Serving needs the anchor to un-relativize; the model never sees it.
            "anchor_pos": torch.from_numpy(anchor_pos).float(),
            "anchor_rot6d": torch.from_numpy(anchor_rot6d).float(),
        }

    def _load_image(self, data, key: str, frame: int) -> np.ndarray:
        img = np.asarray(data[key][int(frame)], dtype=np.float32)  # (H, W, 3)
        img = (img / 255.0 - self.image_mean) / self.image_std
        return np.transpose(img, (2, 0, 1))  # (C, H, W)

    def close(self) -> None:
        for store in self._stores.values():
            try:
                store.close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
        self._roots.clear()
        self._stores.clear()


class StateNoise(Dataset):
    """Wrap a dataset and add zero-mean Gaussian jitter to its ``state``. TRAIN ONLY.

    Why a WRAPPER and not a knob on :class:`HLDataset`
    -------------------------------------------------
    One ``HLDataset`` instance backs both splits (the trainer builds two ``Subset``s over it),
    and ``scripts/fit_hl_norm_stats.py`` builds the very same object. A ``noise=`` flag on the
    dataset would therefore either jitter the VALIDATION objective — making a run's val loss
    incomparable to its own ablation siblings — or bake the augmentation into the quantiles the
    trainer then normalizes by. Wrapping only the train subset makes "clean val, clean stats"
    structural instead of a convention someone has to remember.

    WHERE in the pipeline: this is pre-collate and therefore pre-normalization
    (``Trainer._apply_norm`` runs on the batch), so ``sigma`` is in the state's RAW units —
    metres for ``rel_xy``, radians for ``rpy``. That is the point: "1 cm" and "2 degrees" are
    physical claims about how well the robot knows where its hands are, not numbers in a
    normalized space whose scale moves with the dataset.

    Sigma comes from :func:`flore.s1_policy.data.wb_geom.noise_sigma`, i.e. from the LAYOUT, so
    the gripper columns get exactly zero and a state re-composition cannot silently move the
    jitter onto them.

    Draws come from ``torch.randn``, which DataLoader seeds per worker from the loader's
    generator — so the noise is reproducible under the trainer's own RNG save/restore, and two
    workers never draw the same stream. ``state_valid`` is deliberately NOT touched: a
    fabricated column is already flagged, and jittering a flagged column changes nothing the
    consumer trusts.
    """

    def __init__(self, base: Dataset, sigma) -> None:
        sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
        if sigma.size == 0 or not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError(f"sigma must be finite and >= 0 per column, got {sigma!r}")
        if not np.any(sigma > 0):
            # A no-op wrapper is worse than no wrapper: it costs a randn per sample and hides
            # "the noise flags never reached the layout" behind a plausible-looking pipeline.
            raise ValueError(
                "StateNoise was built with an all-zero sigma — it would be a no-op. Either "
                "pass a nonzero --hl-data.state-noise-* or do not wrap the dataset.")
        self.base = base
        self.sigma = torch.from_numpy(sigma).float()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base[idx]
        state = sample["state"]
        if state.shape[-1] != self.sigma.shape[-1]:
            raise ValueError(
                f"StateNoise sigma has {self.sigma.shape[-1]} columns but the sample's state "
                f"has {state.shape[-1]}; the sigma was built for a different layout.")
        # Broadcasts over a leading obs-history axis when To > 1, and each frame gets its OWN
        # draw — the sensor noise the flag models is per-observation, not per-sample.
        sample["state"] = state + self.sigma * torch.randn_like(state)
        return sample
