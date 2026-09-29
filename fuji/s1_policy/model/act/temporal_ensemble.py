"""Temporal ensembling for ACT — upstream's ``--temporal_agg``, ported to FUJI's frames.

WHY IT EXISTS (measured on this dataset, not a theoretical concern)
------------------------------------------------------------------
ACT decodes from the prior mean (``z = 0``) with a deterministic decoder, so on an anchor
where the demonstrations went two ways it must pick one mode or average them. The future
wipe direction in ``session_2026.08.08_2228_v3`` **is** bimodal in the anchor frame
(x-direction histogram trough/peak = 0.054), and averaging a bimodal wipe means stalling —
exactly the "jerky or pauses in the middle of an episode" symptom ACT's own tuning note
describes. Upstream's answer is this: re-query EVERY step and blend the overlapping
predictions with exponentially-decaying weights, so a chunk boundary is a smooth handover
instead of a jump, and one badly-sampled chunk is outvoted by its neighbours.

THE ALGORITHM (``imitate_episodes.py:216-259``)
----------------------------------------------
At wall-step ``t`` the policy predicts a chunk covering ``[t, t + H)``. Every earlier step
``t' <= t`` whose chunk still reaches ``t`` also holds a prediction for ``t``. Blend them:

    w_i ∝ exp(-k * i),  i = 0 for the OLDEST surviving prediction

With ``k > 0`` **older predictions get MORE weight**. That is upstream's ordering and it is
deliberate: an older chunk was conditioned on a more distant observation, but it has been
corroborated by every step since, so weighting it up smooths the executed path instead of
chasing the newest (noisiest) chunk.

⚠ BLEND IN ABSOLUTE SPACE — THE ONE THING THAT DOES NOT PORT DIRECTLY
--------------------------------------------------------------------
Upstream's actions are ABSOLUTE joint-angle targets, so two chunks predicted one step apart
live in the same frame and averaging them is meaningful. **FUJI's action chunk is
ANCHOR-RELATIVE** (each step is ``R_ref^T (p - p_ref)`` about the observation-time EE pose).
Two chunks predicted at different wall-steps are therefore expressed in *different frames*,
and blending their raw/normalized values would mix frames. Measured magnitude of that
mistake, at the H=50 window's oldest survivor (49 steps old at 30 Hz):

    hand speed 0.05 m/s ->  8.2 cm of frame offset
               0.20 m/s -> 32.7 cm
               0.50 m/s -> 81.7 cm

So this ensembler consumes **ABSOLUTE, already-decoded** poses — each chunk decoded against
its OWN anchor (which :meth:`LivePlanner.infer_chunk` already does) — and blends those. That
is the frame-correct analogue of upstream, not a deviation from its intent.

Rotations are blended as a weighted mean of the rot6d rows followed by re-orthonormalization
(``rot6d_to_rotmat`` applies Gram-Schmidt, then ``rotmat_to_rot6d`` returns to the
representation). A weighted mean of rotations is not a group operation, so this is an
approximation — valid because the survivors of a blend are predictions of the SAME instant
and differ by small angles; it is the same first-order approximation upstream makes by
averaging joint angles. Positions, gripper and wrench are ordinary vector-space blends.

The other change from upstream: it detects "is this slot populated?" with
``torch.all(actions_for_curr_step != 0, axis=1)`` against a zero-initialized buffer, so an
exactly-all-zero prediction reads as ABSENT. On ALOHA's joint targets that is merely
unlikely; here "hold still" is genuinely all-zeros in a relative pose block and a masked
step's gripper/force columns are hard zeros, so occupancy is tracked with an explicit mask.
"""

from __future__ import annotations

import numpy as np

from fuji.utils.rot6d import rot6d_to_rotmat, rotmat_to_rot6d


class TemporalEnsembler:
    """Exponentially-weighted blend of overlapping ABSOLUTE chunk predictions.

    One instance per role per episode. Feed one chunk per wall-step with :meth:`add`, which
    returns the blended values for that step.

    Parameters
    ----------
    action_horizon : int
        ``H`` — steps per predicted chunk (50 on the v3 grid).
    k : float
        Exponential decay rate; ``0.01`` is upstream's value (``imitate_episodes.py:255``).
        ``k = 0`` is a plain mean over the survivors; larger ``k`` concentrates weight on the
        OLDEST survivor (upstream's ordering — see the module docstring).
    capacity : int, optional
        Ring length in wall-steps; defaults to ``action_horizon``. A chunk only ever reaches
        ``H`` steps ahead, so ``capacity >= action_horizon`` makes eviction exact rather than
        approximate — nothing that could still contribute is ever dropped.
    """

    def __init__(self, action_horizon: int, *, k: float = 0.01, capacity: int | None = None) -> None:
        H = int(action_horizon)
        if H < 1:
            raise ValueError(f"action_horizon must be >= 1, got {H}")
        if not (np.isfinite(k) and k >= 0.0):
            raise ValueError(f"k must be finite and >= 0, got {k!r}")
        cap = H if capacity is None else int(capacity)
        if cap < H:
            raise ValueError(
                f"capacity {cap} is below action_horizon {H}: a chunk reaches H steps ahead, "
                f"so a smaller ring would evict predictions that can still contribute")
        self.action_horizon = H
        self.k = float(k)
        self.capacity = cap
        self.reset()

    def reset(self) -> None:
        """Drop every stored prediction. Call when the INTENT is abandoned — a fresh episode,
        a Reset-start-pose, a Stop. NOT on an ordinary clutch re-anchor: the blend is in
        ABSOLUTE space, so a prediction made before the anchor moved is still valid, and
        resetting on every re-anchor would leave one survivor and silently disable the
        feature. See :meth:`LivePlanner.reset_temporal_ensemble`."""
        self._slots: list[dict | None] = [None] * self.capacity
        #: Wall-step index of the NEXT chunk to be added (monotonic; never wraps).
        self.step = 0

    def __len__(self) -> int:
        """How many stored chunks are currently live."""
        return sum(1 for s in self._slots if s is not None)

    def add(
        self,
        position: np.ndarray,
        rot6d: np.ndarray,
        gripper_m: np.ndarray | None = None,
        wrench: np.ndarray | None = None,
    ) -> dict:
        """Store the chunk predicted at the current wall-step; return that step's blend.

        Every argument must be **ABSOLUTE** — decoded against the anchor that was live when
        this chunk was predicted (see the module docstring on why relative values cannot be
        blended). Shapes: ``position`` ``(H, 3)``, ``rot6d`` ``(H, 6)``, ``gripper_m``
        ``(H,)``, ``wrench`` ``(H, 6)``.

        Returns ``{"position": (3,), "rot6d": (6,), "gripper_m": float|None,
        "wrench": (6,)|None, "n_blended": int}``.
        """
        H = self.action_horizon
        pos = np.asarray(position, dtype=np.float64)
        rot = np.asarray(rot6d, dtype=np.float64)
        if pos.shape != (H, 3):
            raise ValueError(f"position must be ({H}, 3), got {pos.shape}")
        if rot.shape != (H, 6):
            raise ValueError(f"rot6d must be ({H}, 6), got {rot.shape}")
        if not (np.isfinite(pos).all() and np.isfinite(rot).all()):
            raise FloatingPointError("chunk pose contains non-finite values")
        entry: dict = {"position": pos, "rot6d": rot}
        for name, value, shape in (("gripper_m", gripper_m, (H,)), ("wrench", wrench, (H, 6))):
            if value is None:
                entry[name] = None
                continue
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape != shape:
                raise ValueError(f"{name} must be {shape}, got {arr.shape}")
            entry[name] = arr

        t = self.step
        self._slots[t % self.capacity] = entry
        self.step = t + 1

        # Gather every live prediction FOR WALL-STEP t, oldest first: the chunk stored at
        # wall-step t-age holds step t at its own index `age`. Oldest-first ordering is what
        # makes exp(-k*i) match upstream's weighting.
        rows: list[tuple[dict, int]] = []
        for age in range(min(t, H - 1), -1, -1):
            slot = self._slots[(t - age) % self.capacity]
            if slot is not None:
                rows.append((slot, age))
        if not rows:  # unreachable (one was just stored), but never return garbage
            rows = [(entry, 0)]

        w = self.weights(len(rows))
        blended_pos = np.zeros(3)
        rot_acc = np.zeros((3, 3))
        grip_num = grip_den = 0.0
        wr_acc = np.zeros(6)
        wr_den = 0.0
        for (slot, age), wi in zip(rows, w):
            blended_pos += wi * slot["position"][age]
            # Weighted mean of ROTATION MATRICES, re-orthonormalized once at the end. Done in
            # matrix space rather than on the raw rot6d rows because a rot6d row pair is only
            # meaningful after Gram-Schmidt; averaging matrices then projecting is the
            # standard first-order rotation blend.
            rot_acc += wi * rot6d_to_rotmat(slot["rot6d"][age][None])[0]
            if slot["gripper_m"] is not None and np.isfinite(slot["gripper_m"][age]):
                grip_num += wi * float(slot["gripper_m"][age])
                grip_den += wi
            if slot["wrench"] is not None and np.isfinite(slot["wrench"][age]).all():
                wr_acc += wi * slot["wrench"][age]
                wr_den += wi

        blended_rot6d = rotmat_to_rot6d(rot_acc[None])[0]
        return {
            "position": blended_pos,
            "rot6d": blended_rot6d,
            "gripper_m": (grip_num / grip_den) if grip_den > 0.0 else None,
            "wrench": (wr_acc / wr_den) if wr_den > 0.0 else None,
            "n_blended": len(rows),
        }

    def weights(self, n: int) -> np.ndarray:
        """Normalized ``exp(-k*i)`` weights for ``n`` survivors, oldest first.

        Exposed so a caller (or a test) can inspect the blend without running the model.
        """
        if n < 1:
            return np.zeros(0, dtype=np.float64)
        w = np.exp(-self.k * np.arange(int(n), dtype=np.float64))
        return w / w.sum()
