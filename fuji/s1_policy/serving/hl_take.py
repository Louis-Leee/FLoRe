"""READER for deploy takes written by :mod:`fuji.s1_policy.serving.hl_record`.

The recorder's counterpart: it turns a ``take_NNN/`` directory back into arrays with named
columns, and — the point of the exercise — lets a checkpoint be re-run on the observation the
policy actually saw at deploy time.

Two questions a recorded take answers, and this module serves both:

  * **"replay 看看效果"** — what did the arm DO versus what was COMMANDED? That is
    :attr:`DeployTake.executed` against :attr:`DeployTake.commanded`, on one shared clock,
    reusing the same :mod:`fuji.s1_policy.serving.hl_trace` statistics + QC sheet the online
    planner writes, so a deploy take and a live run are judged by identical numbers.
  * **"用checkpoint在这个上面任意选一frame尝试prediction"** — what would the policy predict
    HERE, now? That is :meth:`DeployTake.obs_at`, which rebuilds the stored observation into
    the model batch through the SAME :func:`~fuji.s1_policy.serving.hl_online.batch_from_obs`
    the live planner uses. A frame is selectable two ways: an executed PREDICTION (whose
    observation was recorded verbatim) or any measured TICK (whose observation is rebuilt from
    the readback — see :meth:`obs_at` for exactly what that can and cannot reconstruct).

WHY RE-INFERENCE ON A RECORDED OBSERVATION IS EXACT (for a recorded prediction). The take
stores the un-normalized observation and the seed; normalization is re-applied from the
checkpoint being tested. So re-inferring a recorded prediction with the SAME checkpoint and
seed reproduces the deploy chunk to sampler determinism — which is what makes the tool usable
as a regression check ("does this checkpoint still do what it did on the rig?"), not merely a
viewer. Re-inferring with a DIFFERENT checkpoint answers "what would the new model have done
at this exact instant", which is the comparison a deploy dataset exists for.

Read-only and robot-free: no viser, no torch import at module scope, no writes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fuji.s1_policy.data.layout import FORCE_DIM
from fuji.s1_policy.serving.hl_online import ObsSnapshot
from fuji.s1_policy.serving.hl_record import CMD_COLS, EXEC_COLS, SCHEMA
from fuji.s1_policy.serving.hl_trace import ReferencePath, TraceSamples

#: Schema tags this reader understands. An unknown tag is refused rather than read with the
#: current column names, because a renamed/reordered column would silently become a different
#: physical quantity (the same trap the dataset's attr gates exist for).
SUPPORTED_SCHEMAS = frozenset({SCHEMA})


@dataclass(frozen=True)
class TakePrediction:
    """One executed plan read back off disk.

    ``obs_*`` is the un-normalized observation the model was fed. ``images`` is the uint8
    conditioning crop, oldest-first (the last is t0), stored LOSSLESSLY — feed it back through
    :func:`~fuji.s1_policy.serving.hl_online.normalize_image_chw` to recover the exact tensor
    (see :meth:`TakeRecorder._write_predictions` for why lossless matters here).
    """

    index: int
    source: str
    seed: int
    num_steps: int
    t_predict_s: float
    t_exec_start_s: float
    n_exec: int
    chunk: np.ndarray  # (H, action_dim) raw physical
    anchor_pos: np.ndarray  # (n_roles,3)
    anchor_rot6d: np.ndarray  # (n_roles,6)
    images: tuple[np.ndarray, ...]  # To x (224,224,3) uint8
    state: np.ndarray | None  # (To,state_dim)
    state_valid: np.ndarray | None  # (To,state_dim) bool
    force_history: np.ndarray | None  # (FH,ch)
    force_history_valid: bool
    replay_episode: int = -1
    replay_frame: int = -1


class DeployTake:
    """One recorded deploy take, loaded eagerly except for the camera JPEGs.

    Arrays are small (a 20 s take is a few hundred rows) so they load up front; the camera
    stream is left on disk and decoded per frame on request, exactly as the offline viewers
    treat a zarr's image column.
    """

    def __init__(self, take_dir: str | Path):
        self.path = Path(take_dir)
        meta_path = self.path / "take_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"{self.path} is not a deploy take (no take_meta.json). Point this at a "
                f"take_NNN directory written by scripts/hl_serving.py --record.")
        self.meta: dict = json.loads(meta_path.read_text())
        schema = self.meta.get("schema")
        if schema not in SUPPORTED_SCHEMAS:
            raise ValueError(
                f"{self.path}: take schema {schema!r} is not one of "
                f"{sorted(SUPPORTED_SCHEMAS)} — its columns may mean something else. "
                "Read it with the matching FUJI revision.")
        self.label = str(self.meta.get("label", ""))
        self.executed = self._load("executed.npy", len(EXEC_COLS))
        self.commanded = self._load("commanded.npy", len(CMD_COLS))
        self.dof = self._load("dof_pos.npy", 30)
        self.force_meas = self._load("force_meas.npy", None)
        self.camera_index = self._load("camera/frame_index.npy", 2)
        self._exec_col = {name: i for i, name in enumerate(EXEC_COLS)}
        self._cmd_col = {name: i for i, name in enumerate(CMD_COLS)}
        self.predictions = tuple(self._load_prediction(entry)
                                 for entry in self.meta.get("predictions", ()))
        #: Streams whose clock had to be rebased on read — see :meth:`_rebase_legacy_clocks`.
        #: Empty for any take written after the recorder fix (2026-08-17).
        self.legacy_clock_streams: tuple[str, ...] = self._rebase_legacy_clocks()

    def _rebase_legacy_clocks(self) -> tuple[str, ...]:
        """Put ``force_meas`` / ``camera_index`` back on the take-relative clock, if needed.

        Takes recorded before 2026-08-17 stamped those two streams with raw ``time.monotonic()``
        while ``executed``/``commanded`` used take-relative ``elapsed()``. The two epochs differ
        by the process's start time (~10^6 s), so *every* cross-stream lookup silently resolved
        to sample 0: the camera panel froze on the take's first frame and "what did the sensor
        feel here" answered with the first sample forever. Nothing about the DATA is wrong — only
        its zero — so the fix on read is exact up to one offset.

        DETECTION IS DISJOINTNESS, not a magnitude threshold: a legacy stream's whole range sits
        beyond the end of the executed stream, which cannot happen once both share a zero. That
        makes this a no-op for correctly-recorded takes rather than something that might fire on
        an unusually long one.

        The offset is recovered by aligning first samples. Both streams begin when the take arms,
        so the residual is the difference in their first arrivals — measured at 20-40 ms on the
        0817 session, i.e. under one 30 Hz step, and far below the 10 s chart window or the
        nearest-sample lookups this feeds. It is NOT claimed to be exact: `legacy_clock_streams`
        names what was touched so a caller can say so out loud.
        """
        if not len(self):
            return ()          # no executed stream to align against; leave the arrays alone
        t0 = float(self.times[0])
        end = float(self.times[-1])
        rebased: list[str] = []
        for name, arr in (("force_meas", self.force_meas), ("camera_index", self.camera_index)):
            if not arr.size:
                continue
            if float(arr[:, 0].min()) > end + 1.0:
                arr[:, 0] -= float(arr[0, 0]) - t0
                rebased.append(name)
        return tuple(rebased)

    def _load(self, name: str, width: int | None) -> np.ndarray:
        path = self.path / name
        if not path.is_file():
            return np.zeros((0, width if width is not None else 1), dtype=np.float64)
        arr = np.asarray(np.load(path), dtype=np.float64)
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[0] if arr.size else 0, -1)
        if width is not None and arr.size and arr.shape[1] != width:
            raise ValueError(
                f"{path}: expected {width} columns, found {arr.shape[1]} — the take was "
                "written by a different schema than its take_meta.json claims.")
        return arr

    def _load_prediction(self, entry: Mapping[str, Any]) -> TakePrediction:
        import cv2

        pdir = self.path / str(entry.get("dir", f"pred_{int(entry.get('index', 0)):03d}"))
        chunk = np.asarray(np.load(pdir / "chunk.npy"), dtype=np.float64)
        anchor_pos = np.asarray(np.load(pdir / "anchor_pos.npy"), dtype=np.float64)
        anchor_rot6d = np.asarray(np.load(pdir / "anchor_rot6d.npy"), dtype=np.float64)

        def _opt(name: str, dtype=np.float64):
            p = pdir / name
            return None if not p.is_file() else np.asarray(np.load(p), dtype=dtype)

        images = []
        for t in range(int(entry.get("n_obs_images", 0))):
            p = pdir / f"obs_{t:02d}.png"
            if not p.is_file():
                continue
            bgr = cv2.imdecode(np.frombuffer(p.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return TakePrediction(
            index=int(entry.get("index", 0)),
            source=str(entry.get("source", "model")),
            seed=int(entry.get("seed", 0)),
            num_steps=int(entry.get("num_steps", 0)),
            t_predict_s=float(entry.get("t_predict_s", float("nan"))),
            t_exec_start_s=float(entry.get("t_exec_start_s", float("nan"))),
            n_exec=int(entry.get("n_exec", 0)),
            chunk=chunk,
            anchor_pos=anchor_pos,
            anchor_rot6d=anchor_rot6d,
            images=tuple(images),
            state=_opt("state.npy"),
            state_valid=_opt("state_valid.npy", bool),
            force_history=_opt("force_history.npy"),
            force_history_valid=bool(entry.get("force_history_valid") or False),
            replay_episode=int(entry.get("replay_episode", -1)),
            replay_frame=int(entry.get("replay_frame", -1)),
        )

    # -- measured / commanded views ---------------------------------------
    def __len__(self) -> int:
        """Number of measured ticks — the take's natural frame count."""
        return int(self.executed.shape[0])

    @property
    def times(self) -> np.ndarray:
        """(N,) seconds since the take armed, for the measured stream."""
        return self.executed[:, self._exec_col["t"]] if len(self) else np.zeros(0)

    def measured_tcp(self) -> tuple[np.ndarray, np.ndarray]:
        """``(pos[N,3], rot6d[N,6])`` measured LEFT UMI-TCP in the pelvis frame."""
        if not len(self):
            return np.zeros((0, 3)), np.zeros((0, 6))
        pos = self.executed[:, [self._exec_col[k] for k in ("tcp_x", "tcp_y", "tcp_z")]]
        rot = self.executed[:, [self._exec_col[k] for k in (
            "tcp_r00", "tcp_r01", "tcp_r02", "tcp_r10", "tcp_r11", "tcp_r12")]]
        return pos, rot

    def measured_jaw(self) -> np.ndarray:
        return (self.executed[:, self._exec_col["jaw_m"]] if len(self) else np.zeros(0))

    def measured_dof(self) -> tuple[np.ndarray, np.ndarray]:
        """``(t[N], dof_pos[N,29])`` — the raw joint readback stream."""
        if not self.dof.size:
            return np.zeros(0), np.zeros((0, 29))
        return self.dof[:, 0], self.dof[:, 1:]

    # -- the force + jaw streams, each labelled with ITS OWN frame and sign ------------------
    # Three different force quantities live in a take and they are NOT interchangeable. Every
    # accessor below says which it returns, because the whole class of bug here is drawing one
    # while believing it is another (see `fuji/utils/force_frame.py`).

    def measured_wrench_sensor(self) -> tuple[np.ndarray, np.ndarray]:
        """``(t[N], wrench[N,6])`` — the raw Bota stream at the sensor's own rate (~200 Hz).

        **SENSOR frame, MEASURED sign**: byte-for-byte what the F/T bridge published, with no
        relabel and no negation. This is "what the sensor felt". To draw it as the command a
        deploy would publish, negate via :data:`fuji.utils.force_frame.CMD_FROM_MEAS_SIGN` —
        but a chart of a measurement should stay measured (the repo-wide split: arrows are
        re-expressed, charts are the honest stored signal).
        """
        if not self.force_meas.size or self.force_meas.shape[1] < 7:
            return np.zeros(0), np.zeros((0, FORCE_DIM))
        return self.force_meas[:, 0], self.force_meas[:, 1:1 + FORCE_DIM]

    def servo_force_pelvis(self, role: str = "left") -> tuple[np.ndarray, np.ndarray]:
        """``(t[N], force[N,3])`` — the low level's own post-EMA, post-gate ``F_meas``.

        **PELVIS frame** (already rotated by the low level using the joint readback this
        process does not have), MEASURED sign. Literally the vector fed to the servo, so it is
        the fidelity view of "what the controller acted on" — and being pelvis-frame it must
        NOT be run through the sensor→TCP relabel.
        """
        if role not in ("left", "right"):
            raise KeyError(f"servo_force_pelvis: role must be left|right, got {role!r}")
        if not len(self):
            return np.zeros(0), np.zeros((0, 3))
        cols = [self._exec_col[f"fmeas_{role[0]}{a}"] for a in ("x", "y", "z")]
        return self.times, self.executed[:, cols]

    def commanded_force_sensor(self) -> tuple[np.ndarray, np.ndarray]:
        """``(t[N], f_cmd[N,3])`` — the force actually PUBLISHED on the ``:8770`` wire.

        **SENSOR frame, COMMAND sign** — this one is already negated and already gated, because
        it is the post-``_gate_wire_force`` value. So it is the only force stream in a take that
        must NOT be negated again for display. All-zero means the planner's force switch was
        OFF (or every step sat under the gate), which is what a pure-position run looks like.
        """
        if not self.commanded.size:
            return np.zeros(0), np.zeros((0, 3))
        cols = [self._cmd_col[f"fcmd_{a}"] for a in ("x", "y", "z")]
        return self.commanded[:, self._cmd_col["t"]], self.commanded[:, cols]

    def commanded_jaw(self) -> tuple[np.ndarray, np.ndarray]:
        """``(t[N], width_m[N])`` — the jaw APERTURE commanded per published step.

        Same units as :meth:`measured_jaw` (a full aperture in metres), so the two plot on one
        axis: a persistent gap between them IS the grip force, because a position-only jaw
        converts a position error into a squeeze.
        """
        if not self.commanded.size:
            return np.zeros(0), np.zeros(0)
        return (self.commanded[:, self._cmd_col["t"]],
                self.commanded[:, self._cmd_col["grip_cmd_m"]])

    def predicted_wrench(self, pred: TakePrediction, role: str = "left") -> np.ndarray | None:
        """``(H,6)`` the model's PREDICTED wrench for every step of ``pred``'s chunk.

        **SENSOR frame, MEASURED sign** — the model was trained on ``F_meas``, so this is a
        prediction *of a measurement*. ``None`` when the layout has no force slot for ``role``.
        """
        slots = self._layout().slots(role)
        if slots.force is None or pred.chunk.size == 0:
            return None
        return np.asarray(pred.chunk[:, slots.force], dtype=np.float64)

    def predicted_jaw(self, pred: TakePrediction, role: str = "left") -> np.ndarray | None:
        """``(H,)`` the predicted jaw APERTURE (m) per chunk step, or ``None`` if none."""
        slots = self._layout().slots(role)
        if slots.gripper is None or pred.chunk.size == 0:
            return None
        from fuji.s1_policy.data.transforms import GRIPPER_WIDTH_REF_M

        return np.asarray(pred.chunk[:, slots.gripper], dtype=np.float64) * GRIPPER_WIDTH_REF_M

    def executed_step_at(self, frame: int) -> tuple[int, int]:
        """``(chunk_index, step_index)`` the recorder stamped on measured row ``frame``.

        Which step of which plan the arm was executing at that instant — so a viewer can show
        the prediction's OWN value for that step instead of interpolating one.  ``(-1, -1)``
        when the row carries no stamp (idle hold).
        """
        if not len(self):
            return -1, -1
        i = int(np.clip(frame, 0, len(self) - 1))
        return (int(self.executed[i, self._exec_col["chunk_index"]]),
                int(self.executed[i, self._exec_col["step_index"]]))

    def _layout(self):
        from fuji.s1_policy.data.layout import get_layout

        return get_layout(str(self.meta.get("layout_mode", "left_only")))

    def trace_samples(self) -> TraceSamples:
        """The measured stream as a :class:`TraceSamples`, so this take can be fed to the
        SAME ``hl_trace`` statistics + QC sheet the live planner uses. One implementation of
        "how well did it track" across live and recorded runs."""
        pos, rot = self.measured_tcp()
        return TraceSamples(t=self.times, pos=pos, rot6d=rot, grip_m=self.measured_jaw())

    def commanded_reference(self) -> ReferencePath | None:
        """The published wire as a :class:`ReferencePath` on the take's own clock.

        The wire carries the LEFT CONTACT point + the wrist quaternion, while the measured
        stream is the LEFT UMI-TCP — different points on the same rigid body. Converting one
        into the other needs the deploy levers, which the reader deliberately does not own,
        so this returns the wire's own LEFT contact positions and states that in the sheet's
        title rather than silently comparing two different physical points. For a true
        commanded-vs-measured comparison in TCP terms, re-decode a prediction's chunk against
        its stored anchor (:meth:`decoded_chunk_positions`), which IS in TCP terms.
        """
        if not self.commanded.size:
            return None
        t = self.commanded[:, self._cmd_col["t"]]
        pos = self.commanded[:, [self._cmd_col[k] for k in ("pos_lx", "pos_ly", "pos_lz")]]
        # Wire orientation is a quaternion (xyzw); ReferencePath wants rot6d.
        from fuji.utils.rot6d import quat_xyzw_to_rotmat, rotmat_to_rot6d

        quats = self.commanded[:, [self._cmd_col[k] for k in (
            "orn_lx", "orn_ly", "orn_lz", "orn_lw")]]
        rot6d = np.stack([rotmat_to_rot6d(quat_xyzw_to_rotmat(q)) for q in quats]) \
            if quats.shape[0] else np.zeros((0, 6))
        grip = self.commanded[:, self._cmd_col["grip_cmd_m"]]
        return ReferencePath(t=t, pos=pos, rot6d=rot6d, grip_m=grip, t_offset=0.0)

    def camera_frame(self, index: int) -> np.ndarray | None:
        """Decode camera JPEG ``index`` → ``(H,W,3)`` uint8 RGB, or None if absent."""
        import cv2

        path = self.path / "camera" / f"frame_{int(index):06d}.jpg"
        if not path.is_file():
            return None
        bgr = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def camera_index_for_time(self, t: float) -> int | None:
        """The camera frame nearest (and not after, when possible) time ``t`` seconds.

        Nearest-at-or-before, so the frame returned is one the policy could actually have
        conditioned on at ``t`` — an interpolated or future frame would be a small lie in
        exactly the direction that flatters a re-inference."""
        if not self.camera_index.size:
            return None
        ts = self.camera_index[:, 0]
        prior = np.flatnonzero(ts <= float(t))
        return int(prior[-1]) if prior.size else 0

    def prediction_for_time(self, t: float) -> TakePrediction | None:
        """The executed prediction that was dripping at time ``t`` (the last one whose
        execution had started), or None."""
        started = [p for p in self.predictions
                   if np.isfinite(p.t_exec_start_s) and p.t_exec_start_s <= float(t)]
        return started[-1] if started else None

    def decoded_chunk_positions(self, pred: TakePrediction, role: str = "left") -> np.ndarray:
        """Absolute pelvis-frame TCP positions of ``pred``'s chunk, decoded against its OWN
        stored anchor — i.e. the trajectory that WAS planned, in the measured stream's frame.

        This is the honest commanded curve to compare a measured path against: same point
        (UMI-TCP), same frame (pelvis), same decode the planner used.
        """
        from fuji.s1_policy.data.layout import get_layout
        from fuji.s1_policy.data.transforms import delta_ee_to_absolute

        layout = get_layout(str(self.meta.get("layout_mode", "left_only")))
        ri = layout.roles.index(role)
        slots = layout.slots(role)
        pos, _ = delta_ee_to_absolute(
            pred.chunk[:, slots.pos], pred.chunk[:, slots.rot6d],
            pred.anchor_pos[ri], pred.anchor_rot6d[ri],
        )
        return pos

    def obs_from_prediction(self, pred: TakePrediction, loaded) -> ObsSnapshot:
        """Rebuild ``pred``'s stored observation into a feedable
        :class:`~fuji.s1_policy.serving.hl_online.ObsSnapshot`.

        EXACT: every field was recorded verbatim, and the images re-normalize through the same
        affine that produced them, so feeding this to the SAME checkpoint with the SAME seed
        reproduces the deploy chunk. Raises if the take stored no observation for this
        prediction (a REPLAY-sourced chunk has none — there was no model inference).
        """
        from fuji.s1_policy.serving.hl_online import ObsSnapshot, normalize_image_chw

        if pred.state is None or not pred.images:
            raise ValueError(
                f"prediction {pred.index} carries no recorded observation (source="
                f"{pred.source!r}); only a MODEL-sourced prediction has one. Use obs_at() to "
                "rebuild an observation from the readback instead.")
        mean = float(self.meta.get("image_mean", 0.5))
        std = float(self.meta.get("image_std", 0.5))
        images = np.stack([normalize_image_chw(img, mean, std) for img in pred.images])
        To = int(getattr(loaded, "obs_history_len", images.shape[0]))
        if images.shape[0] != To:
            raise ValueError(
                f"take stored {images.shape[0]} conditioning frame(s) but this checkpoint "
                f"wants To={To} — it was trained with a different observation history.")
        fh = np.asarray(pred.force_history if pred.force_history is not None
                        else np.zeros((int(loaded.force_hist_len),
                                       int(loaded.layout.force_hist_channels))), dtype=np.float64)
        return ObsSnapshot(
            images=images,
            state=np.asarray(pred.state, dtype=np.float64),
            state_valid=(np.asarray(pred.state_valid, dtype=bool) if pred.state_valid is not None
                         else np.ones_like(pred.state, dtype=bool)),
            force_history=fh,
            force_history_valid=bool(pred.force_history_valid),
        )

    def obs_at(self, frame: int, loaded) -> tuple[ObsSnapshot, np.ndarray, np.ndarray]:
        """Rebuild an observation at measured TICK ``frame`` → ``(obs, anchor_pos, anchor_rot6d)``.

        This is the "任意选一frame" path: pick any instant of the recorded run, not just the
        instants where the policy happened to be queried, and ask what the checkpoint predicts
        there. Returns the anchor too, because a chunk decoded at this frame must branch from
        the pose the robot ACTUALLY held at it — the measured TCP — which is exactly the clutch
        semantic the live planner applies at every Predict.

        WHAT IS RECONSTRUCTED, AND WHAT IS APPROXIMATED — stated because a re-inference that
        quietly differs from the deploy one is worse than no tool:

        * **Image** — the camera frame at-or-before this tick, center-cropped by the SAME
          function the live ingest uses. Faithful up to the bridge's frame rate.
        * **Force window** — the raw F/T samples inside the trained span ending at this tick,
          binned by the same causal rule; ``valid`` is False when the span holds no sample.
        * **State** — ``wrt_start_rot6d`` needs the EPISODE-START rotation the live planner had
          latched. A take's own first measured tick is used as that reference, which is right
          for a take that began at the ready pose (the normal case: the recorder arms when a
          Run starts) and is an approximation for a take armed mid-motion. The gripper channel
          is the measured jaw at this tick, which is exact.

        For a bit-exact reproduction of a deploy inference, prefer
        :meth:`obs_from_prediction` on a recorded prediction.
        """
        from fuji.s1_policy.data.transforms import gripper_width_to_norm, rot6d_relative
        from fuji.s1_policy.serving.hl_online import (
            ObsSnapshot,
            normalize_image_chw,
            resize_center_crop_224,
        )

        if not len(self):
            raise ValueError(f"{self.path}: take has no measured ticks to select a frame from")
        idx = int(np.clip(frame, 0, len(self) - 1))
        pos, rot = self.measured_tcp()
        t = float(self.times[idx])
        layout = loaded.layout
        To = int(loaded.obs_history_len)

        # --- image: nearest at-or-before, through the live crop path ---
        mean, std = float(loaded.image_mean), float(loaded.image_std)
        cam_i = self.camera_index_for_time(t)
        rgb = None if cam_i is None else self.camera_frame(cam_i)
        if rgb is None:
            # No camera stream: mid-grey, the same cold-start fill LivePlanner._image_stack
            # uses. Loud enough to notice in the viewer (a flat grey conditioning image) and
            # honest — fabricating a frame would make the prediction look conditioned when it
            # was not.
            frame_chw = normalize_image_chw(
                np.full((224, 224, 3), 128, dtype=np.uint8), mean, std)
        else:
            frame_chw = normalize_image_chw(resize_center_crop_224(rgb), mean, std)
        images = np.repeat(frame_chw[None], To, axis=0)  # To=1 on the v3 contract

        # --- state: wrt_start rotation + measured gripper fraction ---
        state = np.zeros((To, layout.state_dim), dtype=np.float64)
        state_valid = np.ones((To, layout.state_dim), dtype=bool)
        start_rot6d = rot[0]
        jaw = float(self.measured_jaw()[idx]) if len(self) else float("nan")
        for role, entry in layout.state_slots().items():
            if role != "left":
                continue  # LEFT-canonical deploy: only the left wrist is measured here
            for ti in range(To):
                state[ti, entry["wrt_start_rot6d"]] = rot6d_relative(rot[idx], start_rot6d)
                if "gripper" in entry and np.isfinite(jaw):
                    state[ti, entry["gripper"]] = float(gripper_width_to_norm(np.asarray(jaw)))

        # --- force window: the trained span ending at this tick ---
        fh_len = int(loaded.force_hist_len)
        ch = int(layout.force_hist_channels)
        force_win = np.zeros((fh_len, ch), dtype=np.float64)
        fh_valid = False
        if ch and self.force_meas.size:
            span = float(loaded.force_hist_span_s)
            ts = self.force_meas[:, 0]
            vals = self.force_meas[:, 1:1 + ch]
            inside = np.flatnonzero((ts <= t) & (ts > t - span))
            if inside.size:
                dt = span / fh_len
                bins: list[list[np.ndarray]] = [[] for _ in range(fh_len)]
                for j in inside:
                    age = t - float(ts[j])
                    b = int(np.clip(fh_len - 1 - int(np.floor(age / dt)), 0, fh_len - 1))
                    bins[b].append(vals[j])
                fh_valid = all(len(b) > 0 for b in bins)
                last = None
                for b in range(fh_len):
                    if bins[b]:
                        force_win[b] = np.mean(np.stack(bins[b]), axis=0)
                        last = force_win[b]
                    elif last is not None:
                        force_win[b] = last
                first = next((force_win[b] for b in range(fh_len) if bins[b]), None)
                if first is not None:
                    for b in range(fh_len):
                        if bins[b]:
                            break
                        force_win[b] = first

        obs = ObsSnapshot(images=images, state=state, state_valid=state_valid,
                          force_history=force_win, force_history_valid=fh_valid)
        n_roles = len(layout.roles)
        anchor_pos = np.zeros((n_roles, 3), dtype=np.float64)
        anchor_rot6d = np.zeros((n_roles, 6), dtype=np.float64)
        for ri, role in enumerate(layout.roles):
            anchor_pos[ri] = pos[idx]
            anchor_rot6d[ri] = rot[idx]
        return obs, anchor_pos, anchor_rot6d

    def summary(self) -> dict:
        """A short JSON-friendly description, for a headless check."""
        n = self.meta.get("n_frames", {})
        return {
            "path": str(self.path),
            "label": self.label,
            "schema": self.meta.get("schema"),
            "duration_s": float(self.meta.get("duration_s", 0.0)),
            "n_frames": dict(n),
            "n_predictions": len(self.predictions),
            "n_predicts_discarded": int(self.meta.get("n_predicts_discarded", 0)),
            "legacy_clock_streams": list(self.legacy_clock_streams),
            "predictions": [
                {
                    "index": p.index,
                    "source": p.source,
                    "seed": p.seed,
                    "n_exec": p.n_exec,
                    "chunk_shape": list(p.chunk.shape),
                    "n_obs_images": len(p.images),
                    "has_state": p.state is not None,
                    "force_history_valid": p.force_history_valid,
                }
                for p in self.predictions
            ],
        }


def find_takes(session_dir: str | Path) -> list[Path]:
    """Every ``take_NNN`` directory in a recorded session, in order.

    Skips a ``.tmp`` leftover: that is a flush killed mid-write, and reading one would present
    a partial take as complete."""
    root = Path(session_dir)
    if (root / "take_meta.json").is_file():
        return [root]  # pointed straight at one take
    return sorted(p for p in root.glob("take_*")
                  if p.is_dir() and not p.name.endswith(".tmp"))


__all__ = ["SUPPORTED_SCHEMAS", "DeployTake", "TakePrediction", "find_takes"]
