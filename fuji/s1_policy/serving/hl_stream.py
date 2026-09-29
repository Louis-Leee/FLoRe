"""ASYNCHRONOUS CHUNK STREAMING: HATO's inference cadence, for a FUJI keypoint wire.

WHAT PROBLEM THIS SOLVES, AND WHAT IT DOES NOT
==============================================
Every FUJI deploy planner runs one cadence: ``Predict`` (the sampler blocks) then ``Run`` (drip
``exec_steps`` of the chunk at 30 Hz), repeatedly. While the sampler runs, **nothing new is
published** — the heartbeat re-sends the last frame, so the low level holds. The motion is
therefore a train of moving windows separated by holds, which is exactly what the operator sees as
"一段一段" / stop-go, and the hold fraction is ``L / (L + E)`` for a sampler that costs ``L`` chunk
steps and an execute window of ``E``.

``third_party/hato`` (Lin et al., *Learning Visuotactile Skills with Two Multifingered Hands*,
§IV-B "Deployment") removes the hold instead of shortening it. Two ideas, and they only work
together:

**1. The sampler and the wire run in PARALLEL.** ``hato/inference_node.py``'s
``ZMQInferenceServer.serve`` loops "take the newest observation, sample, ship the chunk" on its own
process, and ``ZMQInferenceClient.act`` — called once per control tick — never blocks on it: it
drains whatever has arrived and publishes one action. The control loop's period is then the
publish period, not the publish period plus an inference.

**2. A chunk is stamped with the OBSERVATION's timestep, so overlapping chunks are COMMENSURABLE.**
The chunk sampled from the observation at step ``t_obs`` means "row ``k`` belongs to step
``t_obs + k``". Several chunks, sampled at different times, therefore hold an opinion about the
SAME future step, and ``act`` reduces them to one::

    while pt < self.t and len(acts) > 0:   # latency skip: rows whose step has passed
        pt += 1; acts = acts[1:]           # are DROPPED, never executed late
    ...
    "avg" -> mean       "act" -> z = z*tau + c*(1-tau)      "new" -> newest      "old" -> oldest

The paper's own words: *"it computes the average of the predictions over multiple timesteps
(similar to the temporal ensemble in [ACT])… We have found that the inclusion of action aggregation
greatly improves motion smoothness."* The blend is what makes the handover smooth; the asynchrony
is what removes the hold. Doing only the first would replace one big jump per chunk with a small
jump per arrival — measured below, that is what happens when the overlap gets thin.

WHY THE BLEND IS LEGITIMATE HERE (and the one place it would not be)
===================================================================
HATO's action is the **absolute** desired joint position, so two chunks sampled a moment apart live
in one frame and averaging them means something. FUJI's chunk is **ANCHOR-RELATIVE**, so the raw
rows of two chunks are expressed in two different frames and blending them would mix frames — up to
82 cm of error at the oldest survivor of an H=50 window at 0.5 m/s
(:mod:`fuji.s1_policy.model.act.temporal_ensemble`, which records the same trap for the ACT-paper
variant of this). This module therefore consumes :class:`~fuji.s1_policy.serving.hl_online.
ChunkResult` — already un-relativized by :func:`~fuji.s1_policy.serving.hl_online.decode_chunk`
against the anchor that was live when that chunk was sampled — and blends **absolute** poses.

⚠️ THE STAMP MUST BE EXACT, which is why :class:`StepClock` exists. ``t_obs`` is not a wall clock:
it is "which published step's commanded target was the planner's anchor when the observation was
taken". Under ``obs_mode=last_target`` those are the same fact, so the publisher marks the clock in
the SAME critical section in which it commits the anchor, and :class:`StreamPlanner` reads the mark
in the same critical section in which it snapshots the observation. Off by one step is 33 ms of
skew on every chunk — small, but it is free to be right.

MEASURED, on ``checkpoints/umi_5pt/h100_step_00100000`` (H=100)
===============================================================
Open loop on REAL consecutive dataset anchors of ``0902_merged3`` episode 29 (a 548 mm reach): one
chunk sampled per recorded frame, then the same pool of chunks driven through both schedulers.
``kick`` is the per-step change in per-step travel (mm/step², i.e. what the arm feels as a jerk),
``track`` is how far the published DISPLACEMENT falls behind the recorded one::

    sampler L   scheduler            frozen   kick p95/max     track p50/p95
    2 (67 ms)   sequential E=16      16.7%    6.5 / 25.4       17.1 / 117.5 mm
                stream avg            0.0%    2.9 / 11.0        2.0 /   9.2 mm
    6 (200 ms)  sequential E=16      31.8%    5.0 / 21.5       33.4 / 231.9 mm
                stream avg            0.0%    5.0 / 22.7        2.9 /  14.2 mm
    12 (400 ms) sequential E=16      46.4%    4.9 / 25.0       51.1 / 430.4 mm
                stream avg            0.0%   11.6 / 35.2        3.2 /  25.5 mm

Read it in that order:

* **``frozen`` is the stutter and it goes to zero.** It is ``L/(L+E)`` and nothing about the model
  changes it — 17% at this box's measured 51 ms sample, 32% with the violet IK body competing for
  the interpreter (the bring-up's own 174 ms figure), 46% at 400 ms.
* **``track`` is the cost of the freeze**, and it is the largest number on the page: the sequential
  wire falls 17-51 mm (p50) behind the motion it is reproducing, because for a sixth to a half of
  every second it is not moving at all. Streaming holds 2.0-3.2 mm.
* **``kick`` improves only while the OVERLAP is thick.** 2.3x better at L=2 (20-39 live chunks), a
  wash at L=6 (7-13), and 1.4x WORSE at L=12 (3-6) — with few members, one arriving or expiring
  chunk moves the mean by a third of however much the members disagree, so the ensemble becomes its
  own step source. ⚠️ Streaming is not a smoother for a slow sampler; it is a smoother for a FAST
  one. ``H / L >= ~10`` is where the numbers above turn.

Mode ranking at L=2/6 (same run): ``avg`` best kick, ``act`` (tau 0.5) best p95 tracking and within
1.4x on kick, ``new`` mid, ``old`` clearly worst (it commits to a stale plan: track p95 42-46 mm).
:data:`DEFAULT_ENSEMBLE_MODE` is ``avg``, which is also ``hato/run_env.py``'s default.

⚠️ OPEN LOOP is the caveat on ``track`` specifically. Each chunk is conditioned on the RECORDED
observation for its frame, not on what this scheduler commanded, so ``track`` measures "can the
cadence keep up with the demonstrated timing" and not closed-loop task success. ``frozen`` and
``kick`` are properties of the scheduler alone and carry over unchanged.

WHAT THIS MODULE IS NOT
=======================
Not RTC. :class:`~fuji.s1_policy.serving.rtc.RealTimeChunkingPolicy` (the S1 gRPC stack) solves the
same wall-clock problem by INPAINTING the frozen prefix into the next sample, so the model itself
produces a continuous trajectory; that needs the DiT decoder's per-position time embedding and a
sampler modification. This is the cheap, model-agnostic half: no training change, no sampler change,
nothing new on the wire, and every published frame still goes through the reviewed
``FivePointWire.publish`` gate. The two are complementary and not alternatives.

Not the ACT ensembler. :class:`~fuji.s1_policy.model.act.TemporalEnsembler` implements the
ACT-paper weighting (``exp(-k i)``, oldest-heaviest) for a caller that re-queries the policy EVERY
step — a cadence that needs inference to finish inside one 33 ms step, which no checkpoint here
does. It also advances its own clock on every ``add``, so it cannot be read without being written.
This module is absolute-step-indexed and read-only at ``at()``, which is what asynchrony requires.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from fuji.s1_policy.serving.hl_online import (
    ChunkResult,
    RoleChunk,
    batch_from_obs,
    decode_chunk,
    sample_chunk,
)
from fuji.utils.rot6d import rot6d_to_rotmat, rotmat_to_rot6d

#: How the opinions about one future step are reduced to one command. HATO's own four
#: (``inference_node.py:134-144``), by their own names so a reader can find them upstream.
#:
#: ``avg`` — the plain mean. HATO's default (``run_env.py``) and the measured best on jerk.
#: ``act`` — ``z = z*tau + c*(1-tau)`` folded oldest-to-newest, i.e. geometrically decaying weights
#:           with the NEWEST heaviest. Named for ACT, though it is not ACT's ``exp(-k i)``
#:           (which is oldest-heaviest — see :class:`~fuji.s1_policy.model.act.TemporalEnsembler`).
#: ``new`` — the newest surviving prediction only. Most responsive, least smooth.
#: ``old`` — the oldest surviving prediction only. Smooth WITHIN one chunk and worst measured
#:           tracking, because it keeps executing a plan the observation has already outdated.
ENSEMBLE_MODES = ("avg", "act", "new", "old")
DEFAULT_ENSEMBLE_MODE = "avg"
#: ``act``'s fold factor. 0.5 is ``hato/run_env.py``'s ``temporal_ensemble_act_tau``. Larger tau
#: weights the OLDER survivors more; ``tau = 0`` collapses ``act`` to ``new``.
DEFAULT_ACT_TAU = 0.5


def ensemble_weights(n: int, mode: str = DEFAULT_ENSEMBLE_MODE,
                     tau: float = DEFAULT_ACT_TAU) -> np.ndarray:
    """Normalized blend weights for ``n`` surviving predictions, **oldest first**.

    Every mode is expressed as weights — including the two that are selections (``new`` / ``old``
    are one-hot) — so the blend has ONE code path. A selecting mode then reproduces its chosen
    prediction to float precision rather than approximately, which is what makes ``new`` a usable
    A/B against a blend rather than a different pipeline.

    ``act`` unrolls HATO's recursion ``z = c[0]; z = z*tau + c[i]*(1-tau)``: for ``n`` survivors the
    weights are ``[tau^(n-1), tau^(n-2)(1-tau), …, (1-tau)]``, which sums to 1 for any tau.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if mode not in ENSEMBLE_MODES:
        raise ValueError(f"mode must be one of {ENSEMBLE_MODES}, got {mode!r}")
    if mode == "new":
        w = np.zeros(n)
        w[-1] = 1.0
        return w
    if mode == "old":
        w = np.zeros(n)
        w[0] = 1.0
        return w
    if mode == "avg":
        return np.full(n, 1.0 / n)
    t = float(tau)
    if not (np.isfinite(t) and 0.0 <= t <= 1.0):
        raise ValueError(f"tau must be finite in [0, 1], got {tau!r}")
    if n == 1:
        return np.ones(1)
    w = np.empty(n)
    w[0] = t ** (n - 1)
    w[1:] = t ** np.arange(n - 2, -1, -1) * (1.0 - t)
    total = float(w.sum())
    # Exactly 1.0 analytically; renormalized because tau^(n-1) underflows for large n and the
    # published command must be a convex combination of poses, not a scaled one.
    return w / total if total > 0.0 else np.full(n, 1.0 / n)


def blend_rot6d(rot6d: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``(m, 6)`` rot6d rows -> one blended rot6d whose rows are ORTHONORMAL.

    A weighted mean of rotations is not a group operation. This is the standard first-order blend
    the repo already uses for the ACT ensembler: average the rotation MATRICES, then project back
    onto SO(3) with the Gram-Schmidt that ``rot6d_to_rotmat`` applies. It is valid because the
    survivors are predictions of the SAME instant and differ by small angles.

    The projection is done HERE rather than left to the consumer so that ``at()`` returns a
    well-formed rotation: a raw matrix mean has rows that are neither unit nor orthogonal, and the
    only thing downstream of this is a wire whose quaternion the low level composes without
    checking.
    """
    r = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    mats = rot6d_to_rotmat(r)                                  # (m, 3, 3), each orthonormal
    mean = np.einsum("i,ijk->jk", w, mats)
    # rotmat_to_rot6d takes the first two ROWS; rot6d_to_rotmat re-orthonormalizes them. The pair
    # is the projection, and running it twice is what leaves the OUTPUT orthonormal too.
    return rotmat_to_rot6d(rot6d_to_rotmat(rotmat_to_rot6d(mean[None]))[0][None])[0]


@dataclass(frozen=True)
class StreamStep:
    """One published step's worth of command, blended from every chunk that still covers it.

    ``position`` / ``rot6d`` are ABSOLUTE, in the decode anchor's world frame — the same frame
    :func:`~fuji.s1_policy.serving.hl_online.decode_chunk` produces and the wire's readout expects.
    ``wrench`` is the UNGATED prediction, so the caller applies the live force gate exactly as it
    does for a non-streamed chunk.
    """

    step: int
    position: dict[str, np.ndarray]
    rot6d: dict[str, np.ndarray]
    gripper_m: dict[str, float] = field(default_factory=dict)
    wrench: dict[str, np.ndarray] = field(default_factory=dict)
    #: How many chunks contributed. 1 means "this step is one chunk's row", i.e. no smoothing —
    #: the number the page must show, because a thin overlap is when streaming stops helping.
    n_blended: int = 0
    #: Age in steps of the oldest / newest contributor (``step - t_obs``). ``newest_age`` is the
    #: effective inference latency the command is carrying.
    oldest_age: int = 0
    newest_age: int = 0


class StepClock:
    """The step index whose commanded target the planner's anchor currently holds.

    Not a wall clock and not the publisher's loop counter: it is the ONE integer that makes a
    chunk's rows addressable in absolute time. The publisher :meth:`mark` s it inside the same
    critical section in which it commits the anchor; a sampler reads :attr:`value` inside the same
    critical section in which it snapshots the observation (see :class:`StreamPlanner`). Both use
    :attr:`lock`, so a stamp can never straddle a commit.

    Re-entrant because the publisher's critical section calls into planner methods that take their
    own lock, and because :meth:`mark` is convenient to call from inside a caller's ``with``.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._t = 0

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def value(self) -> int:
        with self._lock:
            return self._t

    def mark(self, t: int) -> None:
        with self._lock:
            self._t = int(t)

    def reset(self, t: int = 0) -> None:
        self.mark(t)


class ChunkStream:
    """Overlapping decoded chunks, addressed by ABSOLUTE step, blended on read.

    HATO's ``act_q`` with the protocol taken out: chunks are stored whole against their
    observation step instead of being scattered into per-step buckets on arrival, which makes the
    same three operations exact rather than incremental — the latency skip (a row whose step has
    passed is simply never asked for), eviction (a chunk is live iff ``t_obs <= t < t_obs + H``),
    and the blend.

    Written by the sampler thread (:meth:`add`), read by the publisher (:meth:`at`). The lock is
    held only to copy out the surviving ``(entry, age)`` list — the blend itself runs outside it,
    which it may because an entry's arrays are never mutated after :meth:`add`.
    """

    def __init__(self, horizon: int, roles, *, mode: str = DEFAULT_ENSEMBLE_MODE,
                 tau: float = DEFAULT_ACT_TAU, max_entries: int | None = None) -> None:
        h = int(horizon)
        if h < 1:
            raise ValueError(f"horizon must be >= 1, got {h}")
        if mode not in ENSEMBLE_MODES:
            raise ValueError(f"mode must be one of {ENSEMBLE_MODES}, got {mode!r}")
        self.horizon = h
        self.roles = tuple(str(r) for r in roles)
        #: Bound on stored chunks. A chunk is only useful for H steps, so more than ``2H`` of them
        #: means the sampler is producing faster than one per half-step; the oldest are dropped and
        #: counted rather than growing the store without limit.
        self.max_entries = int(max_entries) if max_entries is not None else 2 * h
        self._lock = threading.Lock()
        self._mode = str(mode)
        self._tau = float(tau)
        self._entries: list[tuple[int, dict]] = []      # (t_obs, {role: arrays}), t_obs ascending
        self._n_added = 0
        self._n_evicted = 0
        self._n_overflow = 0
        self._n_late = 0

    # --- settings the page may change mid-run ----------------------------
    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in ENSEMBLE_MODES:
            raise ValueError(f"mode must be one of {ENSEMBLE_MODES}, got {value!r}")
        with self._lock:
            self._mode = str(value)

    @property
    def tau(self) -> float:
        with self._lock:
            return self._tau

    @tau.setter
    def tau(self, value: float) -> None:
        v = float(value)
        if not (np.isfinite(v) and 0.0 <= v <= 1.0):
            raise ValueError(f"tau must be finite in [0, 1], got {value!r}")
        with self._lock:
            self._tau = v

    def reset(self) -> None:
        """Drop every stored chunk. Call when the INTENT is abandoned — a new run, Stop, Reset.

        NOT on an ordinary clutch re-base: the blend is in ABSOLUTE space, so a chunk sampled
        before the base moved is still expressed in the same world frame and stays valid. Clearing
        it on every re-base would leave one survivor and silently turn the feature off while the
        page still said it was on — the trap
        :meth:`~fuji.s1_policy.serving.hl_online.LivePlanner.reset_temporal_ensemble` records.
        """
        with self._lock:
            self._entries = []

    # --- write -----------------------------------------------------------
    def add(self, chunk: ChunkResult, t_obs: int) -> None:
        """Store a decoded chunk against the step its ROW 0 belongs to.

        ``t_obs`` is :attr:`StepClock.value` as read when this chunk's observation was snapshotted.
        Rows before the current publish step are not stripped here: they are simply never asked
        for, so the latency skip costs nothing and leaves the store addressable from any step.
        """
        t = int(t_obs)
        rows: dict[str, dict] = {}
        for rc in chunk.roles:
            if rc.role not in self.roles:
                continue
            pos = np.asarray(rc.position, dtype=np.float64)
            rot = np.asarray(rc.rot6d, dtype=np.float64)
            if pos.shape[0] < self.horizon or rot.shape[0] < self.horizon:
                raise ValueError(
                    f"role {rc.role!r} has {pos.shape[0]} steps but this stream was built for "
                    f"{self.horizon}; a short chunk would run the coverage out early")
            if not (np.isfinite(pos[:self.horizon]).all() and np.isfinite(rot[:self.horizon]).all()):
                raise FloatingPointError(
                    f"role {rc.role!r} carries a non-finite pose — refused here rather than at the "
                    "wire, so a bad sample never enters the blend and poisons later steps")
            rows[rc.role] = {
                "position": pos[:self.horizon].copy(),
                "rot6d": rot[:self.horizon].copy(),
                "gripper_m": (None if rc.gripper_m is None
                              else np.asarray(rc.gripper_m, dtype=np.float64)[:self.horizon].copy()),
                "wrench": (None if rc.wrench_pred is None
                           else np.asarray(rc.wrench_pred, dtype=np.float64)[:self.horizon].copy()),
            }
        missing = [r for r in self.roles if r not in rows]
        if missing:
            raise ValueError(f"chunk has no {missing} role(s); this stream publishes "
                             f"{list(self.roles)}")
        with self._lock:
            self._entries.append((t, rows))
            self._entries.sort(key=lambda e: e[0])
            self._n_added += 1
            while len(self._entries) > self.max_entries:
                self._entries.pop(0)
                self._n_overflow += 1

    # --- read ------------------------------------------------------------
    def live_at(self, step: int) -> int:
        """How many stored chunks cover ``step``. Cheap; does not blend or evict."""
        t = int(step)
        with self._lock:
            return sum(1 for (t0, _r) in self._entries if t0 <= t < t0 + self.horizon)

    def at(self, step: int) -> StreamStep | None:
        """The blended command for absolute ``step``, or ``None`` if nothing covers it.

        ``None`` is the honest answer to "the sampler has not caught up", and the caller must treat
        it as HOLD (re-publish the last frame) rather than as zero. Exhausted chunks are evicted as
        a side effect, so a long run does not accumulate them.
        """
        t = int(step)
        with self._lock:
            keep, live = [], []
            for t0, rows in self._entries:
                if t0 + self.horizon <= t:
                    self._n_evicted += 1          # can never contribute again
                    continue
                keep.append((t0, rows))
                if t0 <= t:
                    live.append((t0, rows))
            self._entries = keep
            mode, tau = self._mode, self._tau
        if not live:
            return None
        live.sort(key=lambda e: e[0])             # OLDEST FIRST — the weights assume it
        ages = [t - t0 for (t0, _r) in live]
        w = ensemble_weights(len(live), mode, tau)

        pos_out: dict[str, np.ndarray] = {}
        rot_out: dict[str, np.ndarray] = {}
        grip_out: dict[str, float] = {}
        wr_out: dict[str, np.ndarray] = {}
        for role in self.roles:
            stack_p = np.stack([rows[role]["position"][a] for (_t0, rows), a in zip(live, ages)])
            stack_r = np.stack([rows[role]["rot6d"][a] for (_t0, rows), a in zip(live, ages)])
            pos_out[role] = np.einsum("i,ij->j", w, stack_p)
            rot_out[role] = blend_rot6d(stack_r, w)
            # The gripper and the wrench are blended with the SAME weights but over their OWN
            # survivors: a role may have no gripper at all, and a masked step's force column is a
            # hard zero rather than a value, so a member that cannot contribute is dropped and the
            # weights renormalized instead of being counted as agreeing with zero.
            gn = gd = 0.0
            wa = np.zeros(6)
            wd = 0.0
            for ((_t0, rows), a, wi) in zip(live, ages, w):
                g = rows[role]["gripper_m"]
                if g is not None and np.isfinite(g[a]):
                    gn += wi * float(g[a])
                    gd += wi
                f = rows[role]["wrench"]
                if f is not None and np.isfinite(f[a]).all():
                    wa += wi * f[a]
                    wd += wi
            if gd > 0.0:
                grip_out[role] = gn / gd
            if wd > 0.0:
                wr_out[role] = wa / wd
        return StreamStep(step=t, position=pos_out, rot6d=rot_out, gripper_m=grip_out,
                          wrench=wr_out, n_blended=len(live), oldest_age=max(ages),
                          newest_age=min(ages))

    def note_late(self) -> None:
        """Count one arrival whose whole chunk was already in the past. Diagnostic only."""
        with self._lock:
            self._n_late += 1

    def stats(self) -> dict:
        with self._lock:
            return {"mode": self._mode, "tau": self._tau, "horizon": self.horizon,
                    "stored": len(self._entries), "added": self._n_added,
                    "evicted": self._n_evicted, "overflow_dropped": self._n_overflow,
                    "arrived_too_late": self._n_late}


def sample_stamped(planner, clock: StepClock, *, seed: int, force_enabled: bool, gate_n: float,
                   num_steps: int | None = None) -> tuple[ChunkResult, int]:
    """One decoded chunk **plus the absolute step its row 0 belongs to**.

    This is :meth:`~fuji.s1_policy.serving.hl_online.LivePlanner.infer_chunk` with exactly one
    thing added and exactly one thing moved:

    * ADDED — the observation snapshot happens inside :attr:`StepClock.lock`, so the returned
      ``t_obs`` and the anchor the chunk was decoded against describe the same instant. Under
      ``obs_mode=last_target`` "the anchor" IS "the last published step's command", so the stamp is
      that step's index and nothing has to be inferred from a wall clock.
    * MOVED — the SAMPLE runs outside the lock. Holding it across a 50-500 ms sampler would block
      the 30 Hz publisher for the whole inference, which is the exact stall this module exists to
      remove; the lock is held only for the microseconds of the snapshot.

    Kept generic over the planner (it touches only ``m`` / ``layout`` / ``snapshot_obs``) so the
    other whole-body families can stream through the same function rather than a copy of the
    decode.
    """
    with clock.lock:
        t_obs = clock.value
        obs, anchor_pos, anchor_rot6d = planner.snapshot_obs()
    batch = batch_from_obs(planner.m, obs)
    prediction = sample_chunk(planner.m, batch, seed=int(seed), num_steps=num_steps)
    chunk = decode_chunk(prediction, planner.layout, anchor_pos, anchor_rot6d,
                         force_enabled=force_enabled, gate_n=gate_n)
    return chunk, int(t_obs)


def mini_chunk(prev: StreamStep, cur: StreamStep, roles) -> ChunkResult:
    """``(prev, cur)`` -> a TWO-ROW :class:`ChunkResult`, for the wire's own ``plan``.

    THIS IS THE WHOLE INTEGRATOR, and it is deliberately not new arithmetic.
    :meth:`~fuji.s1_policy.serving.hl_dense.KeypointClutch.apply` publishes
    ``p_pub[i] = p_base + (p[i] - p[0])`` and ``R_pub[i] = (R_base R[0]^T) R[i]``. Hand it exactly
    two rows and take row 1::

        p_pub = p_base + (p_cur - p_prev)          R_pub = (R_base R_prev^T) R_cur

    which is the incremental form of "one constant offset, held for the run" — telescoping over the
    steps gives back a single fixed offset, because ``base`` is advanced to each published frame by
    ``wire.commit``. So a streamed run uses the SAME reviewed clutch code as a chunked one, with no
    second formula to keep in step, and every property that code has comes along: the first
    published frame lands bit-exactly on the base (``prev is cur`` at the start, so the delta is
    zero), only the carried points are re-based, the feet pass through absolutely, and
    ``--post-anchor off`` makes ``apply`` the identity so the published stream is the model's own
    absolute readout.

    ⚠️ Row 0 of the returned chunk is NOT for publishing. It exists only to give ``apply`` the
    ``p[0]`` it differences against; the caller publishes ``wire.frame(plan, 1)``.
    """
    order = tuple(str(r) for r in roles)
    out: list[RoleChunk] = []
    for role in order:
        pos = np.stack([prev.position[role], cur.position[role]])
        rot = np.stack([prev.rot6d[role], cur.rot6d[role]])
        grip = None
        if role in prev.gripper_m and role in cur.gripper_m:
            grip = np.array([prev.gripper_m[role], cur.gripper_m[role]], dtype=np.float64)
        wr = None
        if role in prev.wrench and role in cur.wrench:
            wr = np.stack([prev.wrench[role], cur.wrench[role]])
        out.append(RoleChunk(role=role, position=pos, rot6d=rot, gripper_m=grip,
                             wrench_pred=wr, wrench_cmd=None))
    return ChunkResult(roles=tuple(out), action_times_s=np.zeros(2),
                       end_anchor_pos=np.stack([cur.position[r] for r in order]),
                       end_anchor_rot6d=np.stack([cur.rot6d[r] for r in order]))


@dataclass
class StreamStats:
    """What a streamed run must be able to say about ITSELF, per step and per sample.

    Three of these are safety-relevant rather than cosmetic:

    * ``worst_period_ms`` — the publisher's own cadence. It shares one interpreter with the
      sampler, and a starved publisher republishes late, which is the stutter this feature exists
      to remove. A number that has drifted far from ``1000/30`` says the fix is not working.
    * ``frozen`` — steps for which no chunk was live. Published as a HOLD, so the wire is valid and
      the robot is safe, but the motion has stopped; a run that accumulates these is a sampler that
      cannot keep up with its own horizon.
    * ``n_blended`` — the overlap. Below ~5 the blend is measured to stop helping (see the module
      docstring), so it belongs on the page rather than in a log.
    """

    steps: int = 0
    frozen: int = 0
    samples: int = 0
    sample_ms: list[float] = field(default_factory=list)
    period_ms: list[float] = field(default_factory=list)
    worst_period_ms: float = 0.0
    n_blended_last: int = 0
    newest_age_last: int = 0
    #: Chunks the stream refused at :meth:`ChunkStream.add` (non-finite, short, a missing role).
    refused: int = 0
    #: Frames the WIRE's own gate refused. Non-zero should end the run: a chunked drip bounds a
    #: refusal to one chunk, but a stream integrates its base off every published step, so
    #: continuing past one means integrating off a frame the low level never received.
    refused_frames: int = 0

    def note_step(self, period_ms: float | None, blended: int, newest_age: int) -> None:
        self.steps += 1
        self.n_blended_last = int(blended)
        self.newest_age_last = int(newest_age)
        if period_ms is not None:
            self.period_ms.append(float(period_ms))
            self.worst_period_ms = max(self.worst_period_ms, float(period_ms))
            if len(self.period_ms) > 2048:
                del self.period_ms[:1024]

    def note_sample(self, ms: float) -> None:
        self.samples += 1
        self.sample_ms.append(float(ms))
        if len(self.sample_ms) > 256:
            del self.sample_ms[:128]

    def describe(self, step_dt_s: float) -> str:
        # SNAPSHOT both rings first. They are appended to (and trimmed) by the publisher and the
        # sampler while the page loop formats this, and `np.percentile` on a list being resized
        # would compute over a moving target.
        smp_ms = list(self.sample_ms)
        per_ms = list(self.period_ms)
        nom = step_dt_s * 1e3
        smp = (f"{float(np.percentile(smp_ms, 50)):.0f} ms" if smp_ms else "—")
        lat = (f"{float(np.percentile(smp_ms, 50)) / nom:.1f}" if smp_ms else "—")
        per = (f"p50 {float(np.percentile(per_ms, 50)):.1f} / worst "
               f"{self.worst_period_ms:.1f} ms" if per_ms else "—")
        bad = "".join(
            [f" · chunks REFUSED {self.refused}" if self.refused else "",
             f" · frames REFUSED {self.refused_frames}" if self.refused_frames else ""])
        return (f"{self.steps} step(s), {self.samples} sample(s) · sample {smp} "
                f"(= {lat} steps) · overlap {self.n_blended_last} · lag "
                f"{self.newest_age_last} step(s) · publish {per} (nominal {nom:.1f}) · "
                f"frozen {self.frozen}{bad}")


__all__ = [
    "DEFAULT_ACT_TAU",
    "DEFAULT_ENSEMBLE_MODE",
    "ENSEMBLE_MODES",
    "ChunkStream",
    "StepClock",
    "StreamStats",
    "StreamStep",
    "blend_rot6d",
    "ensemble_weights",
    "mini_chunk",
    "sample_stamped",
]
