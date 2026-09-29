"""Map-style torch Dataset over UMI Zarr ReplayBuffers (``.zarr.zip``).

Reads the LEFT-canonical UMI schema written by
``UMI_collection/processing/convert_to_umi.py``: robot0 = the PRIMARY hand, and the
file's ``robot0_primary_role`` attr ('left' for modern 1pt/2pt/3pt; absent/'right'
for legacy single-right-hand zarrs) routes each zarr keypoint into the fixed 32-dim
slot layout below. 2pt/3pt add robot1 (the OTHER hand) and head. Produces fixed
32-dim action/state vectors so a model trained on 3pt can also overfit-test on 1pt
data (absent keypoints are zero-filled and masked out via ``action_dim_mask``).

On-disk keys (under ``/data``)::

    camera0_rgb                [T, H, W, 3] uint8
    robot0_eef_pos             [T, 3]       PRIMARY-hand TCP xyz  (left, LEFT-canonical)
    robot0_eef_rot_axis_angle  [T, 3]       PRIMARY-hand TCP rotvec
    robot0_gripper_width       [T, 1]       PRIMARY-hand width, metres
    # 2pt/3pt extension (optional):
    robot1_eef_pos / robot1_eef_rot_axis_angle / robot1_gripper_width   the OTHER hand
    head_pos [T, 3] / head_rot_axis_angle [T, 3]                        head/torso
    episode_timestamps         [T]          (unused here)
  /meta/episode_ends           [n_ep]       cumulative; [-1] == T_total
  root.attrs["robot0_primary_role"]         'left' (modern) | 'right' (legacy/default)

Fixed 32-dim action / state layout (``keypoint_order = [left, right, head]``)::

    [ 0: 3]  left  pos          [ 3: 9]  left  rot6d      [ 9]      left  gripper
    [10:13]  right pos          [13:19]  right rot6d      [19]      right gripper
    [20:23]  head  pos          [23:29]  head  rot6d      [29:32]   zero pad

29 real dims + 3 zero-pad = 32. Under LEFT-canonical 1pt only the LEFT block
([0:10]) is real; under legacy 1pt only the RIGHT block ([10:20]) is real. The
absent blocks are zero and masked False in ``action_dim_mask``.

Sample (a plain dict):
    image           float32 (C, H, W), normalized to [-1, 1]
    state           float32 (32,)       anchor absolute EE pose (32 layout)
    action_chunk    float32 (H, 32)     poses relative to the anchor (GCL RELATIVE SE(3))
    action_is_pad   bool    (H,)         True where the chunk runs past episode end
    action_dim_mask bool    (32,)        True for real dims present in this sample
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from fuji.s1_policy.data.transforms import (
    REF_FRAME_CHOICES as _REF_FRAME_CHOICES,
    absolute_to_delta_ee,
    gripper_width_to_norm,
    resolve_base,
    rotvec_to_rot6d,
)

logger = logging.getLogger(__name__)

# Per-keypoint slot layout in the 32-dim vector.
#   name: (pos_slice, rot6d_slice, gripper_index_or_None)
_LEFT = (slice(0, 3), slice(3, 9), 9)
_RIGHT = (slice(10, 13), slice(13, 19), 19)
_HEAD = (slice(20, 23), slice(23, 29), None)

ACTION_DIM = 32

_IMAGE_CODECS_REGISTERED = False


def _register_image_codecs() -> None:
    """Register the JpegXl numcodecs codec (id ``imagecodecs_jpegxl``) that
    ``convert_to_umi`` used to write ``camera0_rgb``. Idempotent + best-effort:
    zarr only decodes those chunks lazily, so this must run before the first
    array-metadata load or the read raises ``codec not available``. Prefers the
    modern ``imagecodecs.numcodecs`` (bundled with imagecodecs); a build without
    the numcodecs shim leaves images unreadable, which surfaces loudly on first
    frame access rather than being silently swallowed here."""
    global _IMAGE_CODECS_REGISTERED
    if _IMAGE_CODECS_REGISTERED:
        return
    from imagecodecs.numcodecs import register_codecs

    register_codecs()
    _IMAGE_CODECS_REGISTERED = True


class UMIDataset(Dataset):
    """Map-style dataset over one or more ``.zarr.zip`` UMI ReplayBuffers."""

    def __init__(
        self,
        zarr_paths: str | list[str],
        action_horizon: int = 50,
        image_key: str = "camera0_rgb",
        image_mean: float = 0.5,
        image_std: float = 0.5,
        ref_frame: str = "per_keypoint",
        obs_history_len: int = 1,
        obs_history_stride: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        zarr_paths : str | list[str]
            One or more ``.zarr.zip`` paths.
        action_horizon : int
            H — number of future frames per action chunk (chunk covers frames
            ``[anchor, anchor + H)``; ``action[0]`` corresponds to the observation
            frame, so it is ~identity in relative space).
        image_key : str
            Zarr camera key (default ``camera0_rgb``).
        image_mean, image_std : float
            Image normalization: ``(x/255 - mean) / std``. Defaults map [0, 255]
            uint8 to [-1, 1], matching GCL's stated image convention.
        ref_frame : str
            Which frame the state + action_chunk are expressed in:

              * ``per_keypoint`` (default, byte-identical to the historical layout):
                each keypoint is relative to ITS OWN anchor pose; ``state`` holds each
                keypoint's ABSOLUTE anchor pose.
              * ``left`` / ``right`` / ``head``: ONE shared SE(3) frame = that
                keypoint's anchor pose (HoMMI's gripper-centric frame). Every
                keypoint's ``state`` and ``action_chunk`` is expressed in this single
                frame, so the non-reference keypoints encode their configuration
                relative to the reference hand. Absolute world position is dropped.
              * ``world``: absolute poses everywhere (identity base); ``action`` rows
                are absolute future poses (``action[0]`` is NOT ~identity).
        obs_history_len : int
            To — number of observation frames per sample (>=1). The window ENDS at the
            anchor (frames ``anchor - stride*(To-1) .. anchor``), edge-clamped to the
            episode start so it never crosses an episode boundary. ``To == 1`` (default)
            yields the historical single-frame ``image (C,H,W)`` / ``state (32,)``;
            ``To > 1`` yields ``image (To,C,H,W)`` / ``state (To,32)``.
        obs_history_stride : int
            Stride (in camera frames) between the To observation frames.
        """
        self.action_horizon = int(action_horizon)
        self.image_key = image_key
        self.image_mean = float(image_mean)
        self.image_std = float(image_std)
        if ref_frame not in _REF_FRAME_CHOICES:
            raise ValueError(f"ref_frame must be one of {_REF_FRAME_CHOICES}, got {ref_frame!r}")
        self.ref_frame = ref_frame
        self.obs_history_len = max(1, int(obs_history_len))
        self.obs_history_stride = max(1, int(obs_history_stride))

        if isinstance(zarr_paths, str):
            zarr_paths = [zarr_paths]
        self.zarr_paths = list(zarr_paths)

        # Build a flat index of (file_idx, ep_start, ep_end, anchor) for every frame.
        # We open each store ONLY to scan metadata here, then close it — a live
        # ZipStore holds a file handle that does not survive pickling to DataLoader
        # workers (the shared handle corrupts under concurrent reads). Per-process
        # handles are opened lazily in __getitem__ via _root().
        self._index: list[tuple[int, int, int, int]] = []
        self._has_left: list[bool] = []
        self._has_head: list[bool] = []
        # Per-file zarr-key -> 32-dim-slot spec list, resolved from the file's
        # ``robot0_primary_role`` attr (LEFT-canonical). Built once here so
        # __getitem__ never re-derives it.
        self._specs: list[list] = []

        for fi, path in enumerate(self.zarr_paths):
            root, store = self._open(path)
            data = root["data"]
            has_left = "robot1_eef_pos" in data
            has_head = "head_pos" in data
            self._has_left.append(has_left)
            self._has_head.append(has_head)

            # LEFT-canonical routing: a modern converter writes robot0 = the PRIMARY
            # hand and stamps ``robot0_primary_role`` ('left' for 1pt/2pt/3pt). Legacy
            # zarrs (no attr) predate LEFT-canonical and stored the single hand as the
            # RIGHT hand, so default to 'right' to keep them mapping into the RIGHT slot.
            primary_role = str(dict(root.attrs).get("robot0_primary_role", "right"))
            self._specs.append(
                self._build_specs(primary_role, has_left, has_head, set(data.keys())))

            episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
            prev = 0
            for end in episode_ends:
                end = int(end)
                for anchor in range(prev, end):
                    self._index.append((fi, prev, end, anchor))
                prev = end
            logger.info(
                "UMIDataset: %s frames=%d episodes=%d primary=%s left=%s head=%s",
                path,
                int(episode_ends[-1]) if len(episode_ends) else 0,
                len(episode_ends),
                primary_role,
                has_left,
                has_head,
            )
            store.close()

            # A shared reference frame reads its base pose out of that keypoint's slot,
            # so that keypoint MUST be present in this file — otherwise resolve_base
            # would read an all-zero (masked, absent) slot and hand a degenerate
            # zero-rotation base to every SE(3) transform. 'world' needs no keypoint.
            if self.ref_frame in ("left", "right", "head"):
                present = {name for name, _keys, _slot in self._specs[fi]}
                if self.ref_frame not in present:
                    raise ValueError(
                        f"ref_frame={self.ref_frame!r} but keypoint {self.ref_frame!r} "
                        f"is absent in {path} (present: {sorted(present)}). Cannot "
                        f"anchor a shared frame on a missing keypoint.")

        # Per-process lazily-opened handles. Keyed by (pid, file_idx) and never
        # pickled (excluded in __getstate__) — DataLoader uses fork() on Linux, so a
        # worker inherits the parent's open ZipStore fd; sharing that one fd across
        # processes corrupts reads. The pid guard forces each process (parent or
        # worker) to open its own handle on first access.
        self._roots: dict[int, object] = {}
        self._stores: dict[int, object] = {}
        self._handle_pid: int | None = None

    @staticmethod
    def _open(path: str):
        import zarr

        # camera0_rgb is stored with the vendored UMI JpegXl codec, whose stored
        # id is 'imagecodecs_jpegxl'. numcodecs only knows that id after
        # register_codecs() runs, so reading a frame otherwise raises
        # "codec not available: 'imagecodecs_jpegxl'". Register once per process
        # here (idempotent) — before any lazy array-metadata load.
        _register_image_codecs()
        store = zarr.ZipStore(path, mode="r")
        root = zarr.group(store=store)
        return root, store

    def _root(self, fi: int):
        """Lazily open (and cache, per-process) the zarr group for file index ``fi``."""
        pid = os.getpid()
        if self._handle_pid != pid:
            # A fork happened (or first use): discard any inherited handles.
            self._roots = {}
            self._stores = {}
            self._handle_pid = pid
        root = self._roots.get(fi)
        if root is None:
            root, store = self._open(self.zarr_paths[fi])
            self._roots[fi] = root
            self._stores[fi] = store
        return root

    def __getstate__(self) -> dict:
        # Drop live file handles so the dataset pickles cleanly to workers; each
        # worker reopens its own handles lazily on first access.
        state = self.__dict__.copy()
        state["_roots"] = {}
        state["_stores"] = {}
        state["_handle_pid"] = None
        return state

    def __len__(self) -> int:
        return len(self._index)

    def _load_keypoint(self, data, prefix: str, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return (pos[N,3], rot6d[N,6], gripper[N] or None) for a keypoint prefix.

        ``prefix`` is a (pos_key, rot_key, gripper_key_or_None) triple.
        """
        pos_key, rot_key, grip_key = prefix
        pos = np.asarray(data[pos_key][:], dtype=np.float64)[frames]  # (N, 3)
        rotvec = np.asarray(data[rot_key][:], dtype=np.float64)[frames]  # (N, 3)
        rot6d = rotvec_to_rot6d(rotvec)  # (N, 6)
        grip = None
        if grip_key is not None:
            gw = np.asarray(data[grip_key][:], dtype=np.float64)[frames].reshape(-1)  # (N,)
            grip = gripper_width_to_norm(gw)
        return pos, rot6d, grip

    @staticmethod
    def _build_specs(primary_role: str, has_left: bool, has_head: bool,
                     available_keys: "set | None" = None):
        """Zarr-key-triple -> 32-dim-slot spec list, routed by ``robot0_primary_role``.

        The 32-dim vector has fixed per-hand slots (_LEFT / _RIGHT / _HEAD); WHICH
        zarr keys feed which slot depends on which hand robot0 is:

          * LEFT-canonical (modern, primary_role='left'): robot0 = LEFT hand -> _LEFT
            slot (with its gripper), robot1 = RIGHT hand -> _RIGHT slot, head -> _HEAD.
          * legacy (primary_role='right', no attr): robot0 = RIGHT hand -> _RIGHT slot,
            robot1 (if any) = LEFT hand -> _LEFT, head -> _HEAD. Matches the historical
            single-right-hand layout so old zarrs are unchanged.

        A keypoint's gripper key is nulled when the zarr lacks it (``available_keys``):
        this rig has a SINGLE physical gripper on the primary hand, so robot1 (the
        other hand) has no gripper stream — nulling it leaves that gripper slot zero
        AND masked-False, so the loss never supervises the absent gripper toward a
        false constant 0. Absent keypoints (no robot1 / no head) are simply omitted;
        their slots stay zero + masked-False via action_dim_mask downstream."""
        avail = available_keys  # None => trust the triples as-is (legacy tests)

        def triple(pos, rot, grip):
            # Null the gripper key when the file doesn't carry it.
            if grip is not None and avail is not None and grip not in avail:
                grip = None
            return (pos, rot, grip)

        R0 = triple("robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width")
        R1 = triple("robot1_eef_pos", "robot1_eef_rot_axis_angle", "robot1_gripper_width")
        HEAD_KEYS = ("head_pos", "head_rot_axis_angle", None)
        specs = []
        if primary_role == "left":
            specs.append(("left", R0, _LEFT))          # robot0 = primary LEFT hand
            if has_left:  # robot1 present == the RIGHT hand under LEFT-canonical
                specs.append(("right", R1, _RIGHT))
        else:
            specs.append(("right", R0, _RIGHT))         # legacy: robot0 = RIGHT hand
            if has_left:  # robot1 present == the LEFT hand (legacy bimanual)
                specs.append(("left", R1, _LEFT))
        if has_head:
            specs.append(("head", HEAD_KEYS, _HEAD))
        return specs

    def __getitem__(self, idx: int) -> dict:
        fi, ep_start, ep_end, anchor = self._index[idx]
        root = self._root(fi)
        data = root["data"]
        H = self.action_horizon
        To = self.obs_history_len

        # Chunk frame indices, clamped to the episode end; pad flag marks clamped rows.
        raw_frames = anchor + np.arange(H)
        pad = raw_frames >= ep_end
        chunk_frames = np.clip(raw_frames, ep_start, ep_end - 1)

        # Observation-history frames: a causal window ENDING at the anchor
        # (obs_frames[-1] == anchor), edge-clamped to ep_start so it never crosses
        # into a previous episode. To == 1 => obs_frames == [anchor] (unchanged).
        obs_frames = np.clip(
            anchor - self.obs_history_stride * np.arange(To - 1, -1, -1),
            ep_start, ep_end - 1,
        )

        specs = self._specs[fi]
        # Load each keypoint once over [obs window | chunk]; split afterwards. The
        # anchor pose (reference for per-keypoint mode) is obs poses at index -1.
        all_frames = np.concatenate([obs_frames, chunk_frames])
        loaded = {}
        anchor_abs_state = np.zeros(ACTION_DIM, dtype=np.float64)
        dim_mask = np.zeros(ACTION_DIM, dtype=bool)
        for name, keys, (pos_sl, rot_sl, grip_i) in specs:
            pos, rot6d, grip = self._load_keypoint(data, keys, all_frames)
            obs_pos, chunk_pos = pos[:To], pos[To:]
            obs_rot6d, chunk_rot6d = rot6d[:To], rot6d[To:]
            obs_grip = grip[:To] if grip is not None else None
            chunk_grip = grip[To:] if grip is not None else None
            loaded[name] = (pos_sl, rot_sl, grip_i, obs_pos, obs_rot6d, obs_grip,
                            chunk_pos, chunk_rot6d, chunk_grip)
            # Absolute anchor pose (last obs frame) — used to resolve the shared base.
            anchor_abs_state[pos_sl] = obs_pos[-1]
            anchor_abs_state[rot_sl] = obs_rot6d[-1]
            dim_mask[pos_sl] = True
            dim_mask[rot_sl] = True
            if grip_i is not None and grip is not None:
                dim_mask[grip_i] = True

        # Resolve the ONE shared base for left/right/head/world; None for per_keypoint
        # (each keypoint stays relative to its own anchor).
        shared = self.ref_frame != "per_keypoint"
        base_pos, base_rot6d = (resolve_base(anchor_abs_state, self.ref_frame) if shared
                                else (None, None))

        state_hist = np.zeros((To, ACTION_DIM), dtype=np.float64)
        action_chunk = np.zeros((H, ACTION_DIM), dtype=np.float64)
        for name, (pos_sl, rot_sl, grip_i, obs_pos, obs_rot6d, obs_grip,
                   chunk_pos, chunk_rot6d, chunk_grip) in loaded.items():
            if shared:
                # Every keypoint (all To obs frames + all H chunk rows) expressed in
                # the single anchor-defined base B (HoMMI gripper-centric frame).
                ref_pos, ref_rot6d = base_pos, base_rot6d
                s_pos, s_rot6d = absolute_to_delta_ee(obs_pos, obs_rot6d, ref_pos, ref_rot6d)
            else:
                # per_keypoint: state holds each keypoint's ABSOLUTE pose per obs frame;
                # action is relative to that keypoint's OWN anchor (obs frame -1).
                ref_pos, ref_rot6d = obs_pos[-1], obs_rot6d[-1]
                s_pos, s_rot6d = obs_pos, obs_rot6d
            state_hist[:, pos_sl] = s_pos
            state_hist[:, rot_sl] = s_rot6d

            rel_pos, rel_rot6d = absolute_to_delta_ee(chunk_pos, chunk_rot6d, ref_pos, ref_rot6d)
            action_chunk[:, pos_sl] = rel_pos
            action_chunk[:, rot_sl] = rel_rot6d

            # Gripper is ABSOLUTE in both state and chunk, in every frame mode.
            if grip_i is not None and obs_grip is not None:
                state_hist[:, grip_i] = obs_grip
                action_chunk[:, grip_i] = chunk_grip

        images = self._load_images(data, obs_frames)  # (To, C, H, W)

        # Squeeze the To axis away for the historical single-frame contract.
        if To == 1:
            images = images[0]                # (C, H, W)
            state_out = state_hist[0]          # (32,)
        else:
            state_out = state_hist             # (To, 32)

        return {
            "image": torch.from_numpy(images).float(),
            "state": torch.from_numpy(state_out).float(),
            "action_chunk": torch.from_numpy(action_chunk).float(),
            "action_is_pad": torch.from_numpy(pad),
            "action_dim_mask": torch.from_numpy(dim_mask),
        }

    def _load_images(self, data, frames: np.ndarray) -> np.ndarray:
        """Load + normalize observation images at ``frames`` -> (len(frames), C, H, W).

        Loops per frame (frames may repeat near an episode start after clamping, which
        zarr fancy-indexing does not allow); the window is small (obs_history_len)."""
        arr = data[self.image_key]
        out = []
        for f in frames:
            img = np.asarray(arr[int(f)], dtype=np.float32)  # (H, W, 3) uint8-range
            img = img / 255.0
            img = (img - self.image_mean) / self.image_std
            out.append(np.transpose(img, (2, 0, 1)))  # (C, H, W)
        return np.stack(out, axis=0)  # (T, C, H, W)

    def close(self) -> None:
        for store in self._stores.values():
            try:
                store.close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
        self._roots.clear()
        self._stores.clear()
