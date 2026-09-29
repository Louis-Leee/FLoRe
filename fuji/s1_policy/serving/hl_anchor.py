"""Where the NEXT chunk's decode anchor comes from: the MEASURED pose, or the last COMMAND.

THE PROBLEM THIS EXISTS FOR
---------------------------
FUJI's chunks are anchor-relative, and every ``Predict`` re-anchors on the arm's measured
TCP (``_resync_anchor_from_robot``). That re-anchor is what stops the drip-fed client from
compounding deltas — but it has two costs, both measured on 69 recorded diffusion takes
(23 601 paired ticks, ``scripts/measure_exec_gap.py``, block bootstrap over takes):

1. **It throws away the accumulated command.** The shortfall grows with the commanded path
   travelled since the anchor — ``(1 − g) = 0.392 [0.338, 0.470]``, i.e. the low level
   realizes only ``g ≈ 0.61`` of each commanded increment — and a competing lag model loses
   6:1 in R² (0.076 vs 0.460), with the joint fit's time constant
   ``τ = 0.0145 s [−0.0029, +0.0293]`` straddling zero. So there is no lag to wait out: the
   error is proportional to path, it reaches **~12 mm over a 30-step chunk**, and the
   re-anchor **wipes 84 % [81, 86] of it** (16.52 → 2.68 mm, n = 1104 boundaries). The arm
   is therefore permanently in transient: it would arrive, but the planner re-issues from the
   arm's own lagging pose before it can.
2. **It puts a step in the published target.** At every chunk boundary the commanded pose
   jumps **16.02 mm [13.76, 18.27]** (p95 36.60 mm) *backward, toward the arm* — 13.6× a
   normal in-chunk step of 1.18 mm, a 0.067 m/s spike on a 30 Hz wire.

``mode="command"`` anchors the next chunk on the last EXECUTED commanded step instead, so the
plan continues from where it *asked* to be. Both costs go away by construction. This is
Policy-Level Action Integrator / PLAI (IndustReal, arXiv 2305.17110), whose Appendix C proves
the leaky form is equivalent to PI control, and whose non-converging ``Nominal`` baseline is
verbatim the re-anchor-on-measured scheme above.

WHY THE LEAK IS NOT OPTIONAL
----------------------------
Replaying pure command integration over the same takes — no clamp — the commanded pose
diverges from the measured one to **224 mm by the end of a take** (mean peak 229 mm, p95 peak
574 mm). With ``eps_m`` it saturates exactly where you put it: 5 cm → 49 mm, 3 cm → 30 mm.
So ``eps_m`` is the whole safety story, not a tuning knob, and it is clamped to
:data:`MAX_EPS_M` here so a typo cannot ask the arm for a quarter of a metre. This is PLAI's
``s^d ← s_t ⊕ min(s^d ⊖ s_t, ε)`` with IndustReal's ε of the same order
(``[0.05, 0.05, 0.03] m``, their Table XII). The clamped value is written BACK into the
carried anchor — a clamp that only masked the output would let the internal state run away
and would re-open the jump the mode exists to remove.

WHY CONTACT CUTS OUT
--------------------
Contact multiplies the shortfall **1.8×** early in a chunk (4.29 vs 2.35 mm at step 0), so a
command anchor that keeps advancing against a blocked hand is a press, not a reach — the same
hazard the time-domain-passivity literature documents for its drift compensators, where the
stored correction is dumped "when changing the direction of motion or releasing the wall"
(arXiv 2002.02296, which runs a correction gain of 0.025 and warns that "a gain K of 1 would
cause force peaks"). Above ``contact_n`` the carried anchor is therefore DECAYED back toward
the measured pose rather than advanced. Decayed, not snapped: snapping would re-introduce the
16 mm jump this mode removes.

``mode="measured"`` is the default and is byte-identical to the previous behaviour — it
returns the measured pose unchanged, having touched nothing.

Robot-free and import-light: plain arrays in, plain arrays out, no torch, no viser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fuji.utils.rot6d import rot6d_to_rotmat, rotmat_to_rot6d, rotmat_to_rotvec, rotvec_to_rotmat

#: Hard ceiling on how far the carried command anchor may lead the arm, whatever the operator
#: types. Mirrors ``hl_reach.MAX_CLAMP_M``: both bound an un-asked-for ADDITION to the target
#: of a physical arm, so they get the same ceiling for the same reason.
MAX_EPS_M = 0.06
#: Hard ceiling on the orientation lead (~20°). The wrist is the axis a wiping task loads, so a
#: large carried orientation error is a torque, not a reach.
MAX_EPS_RAD = 0.35

ANCHOR_MODES = ("measured", "command")


@dataclass(frozen=True)
class AnchorConfig:
    """``mode="measured"`` (default) is the previous behaviour, exactly.

    ``eps_m`` / ``eps_rad`` bound how far the carried command anchor may lead the measured
    pose; ``contact_n`` is the measured contact magnitude at or above which the carried anchor
    decays back toward the arm instead of advancing; ``decay`` is that pull-back factor per
    resolve (0.9 ⇒ the lead halves in ~7 ticks, so no step appears on the wire).
    """

    mode: str = "measured"
    eps_m: float = 0.03
    eps_rad: float = 0.175  # 10°
    contact_n: float = 1.5
    decay: float = 0.9


class ChunkAnchor:
    """Decides the next chunk's decode anchor. One instance per served LEFT role.

    Call :meth:`note_commanded` with the last EXECUTED commanded step after each chunk, and
    :meth:`resolve` with the freshly measured pose before each inference. In ``measured`` mode
    :meth:`resolve` is the identity on its input.
    """

    def __init__(self, cfg: AnchorConfig | None = None):
        cfg = cfg or AnchorConfig()
        if cfg.mode not in ANCHOR_MODES:
            raise ValueError(f"anchor mode must be one of {ANCHOR_MODES}, got {cfg.mode!r}")
        if not (0.0 <= cfg.eps_m <= MAX_EPS_M):
            raise ValueError(
                f"anchor eps_m must be in [0, {MAX_EPS_M}] m, got {cfg.eps_m}. This bounds an "
                "un-asked-for lead on a physical arm; raise MAX_EPS_M deliberately if you mean it.")
        if not (0.0 <= cfg.eps_rad <= MAX_EPS_RAD):
            raise ValueError(f"anchor eps_rad must be in [0, {MAX_EPS_RAD}] rad, got {cfg.eps_rad}")
        if not (0.0 < cfg.decay <= 1.0):
            raise ValueError(f"anchor decay must be in (0, 1], got {cfg.decay}")
        if cfg.contact_n < 0.0:
            raise ValueError(f"anchor contact_n must be >= 0, got {cfg.contact_n}")
        self.cfg = cfg
        self._pos: np.ndarray | None = None
        self._rot6d: np.ndarray | None = None
        self._n_resolved = 0
        self._n_bootstrap = 0
        self._n_clamp_pos = 0
        self._n_clamp_rot = 0
        self._n_contact = 0
        self._max_lead_m = 0.0

    # -- state ------------------------------------------------------------------------------
    def disabled(self) -> bool:
        """True when this instance cannot change any published value."""
        return self.cfg.mode == "measured"

    def reset(self) -> None:
        """Drop the carried command anchor (Reset / re-arm / clutch release).

        The next :meth:`resolve` bootstraps from the measured pose, so a re-arm can never
        branch from a command issued before the operator intervened.
        """
        self._pos = None
        self._rot6d = None

    def note_commanded(self, pos: np.ndarray, rot6d: np.ndarray) -> None:
        """Record the LAST EXECUTED commanded step as the pose the next chunk continues from.

        Must be the last *executed* step, not the chunk end: an interrupted chunk's un-executed
        tail was never asked for, and carrying it would hand the next chunk a target the arm was
        never driven toward (the same reason ``LivePlanner.commit_role`` lands on ``last_i``).
        Ignored in ``measured`` mode and on any non-finite input.
        """
        if self.disabled():
            return
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        r = np.asarray(rot6d, dtype=np.float64).reshape(6)
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(r))):
            return
        self._pos = p.copy()
        self._rot6d = r.copy()

    # -- the decision -----------------------------------------------------------------------
    def resolve(self, measured_pos: np.ndarray, measured_rot6d: np.ndarray,
                force_n: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """``(pos, rot6d)`` the next chunk should decode against.

        ``measured_pos`` / ``measured_rot6d`` are the arm's freshly measured LEFT TCP pose in
        the planner's own (model) frame. ``force_n`` is the measured LEFT contact magnitude in
        newtons, or None when unknown — **None does NOT cut out**, because a missing force
        reading must not silently disable a guard the caller believes is armed; pass 0.0 for a
        genuine free-space reading.
        """
        mp = np.asarray(measured_pos, dtype=np.float64).reshape(3)
        mr = np.asarray(measured_rot6d, dtype=np.float64).reshape(6)
        if self.disabled():
            return mp, mr
        self._n_resolved += 1
        if (self._pos is None or self._rot6d is None
                or not (np.all(np.isfinite(mp)) and np.all(np.isfinite(mr)))):
            # Bootstrap (first chunk after arm/reset) or an unusable measurement: the measured
            # pose IS the anchor, which is also the safe fallback.
            self._n_bootstrap += 1
            return mp, mr

        pos = self._pos.copy()
        rot6d = self._rot6d.copy()

        if force_n is not None and float(force_n) >= self.cfg.contact_n:
            # CONTACT: decay the carried lead toward the arm instead of advancing it. Checked
            # FIRST, before the clamps, so a hand pressing at the clamp limit still unwinds.
            self._n_contact += 1
            pos = mp + (pos - mp) * self.cfg.decay
            rot6d = _rot_toward(mr, rot6d, self.cfg.decay)

        # Position leak: PLAI's min(s^d ⊖ s_t, ε), written back so the state cannot run away.
        d = pos - mp
        lead = float(np.linalg.norm(d))
        self._max_lead_m = max(self._max_lead_m, lead)
        if lead > self.cfg.eps_m > 0.0:
            pos = mp + d * (self.cfg.eps_m / lead)
            self._n_clamp_pos += 1
        elif self.cfg.eps_m == 0.0:
            pos = mp.copy()
            self._n_clamp_pos += 1

        # Orientation leak: same bound, applied to the geodesic deviation.
        rot6d, clamped = _clamp_rot(mr, rot6d, self.cfg.eps_rad)
        if clamped:
            self._n_clamp_rot += 1

        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(rot6d))):
            # Never hand a non-finite anchor to the decode: every subsequent published pose
            # would inherit the NaN and holosoma serializes NaN unchecked.
            self._pos = None
            self._rot6d = None
            return mp, mr

        self._pos = pos.copy()
        self._rot6d = rot6d.copy()
        return pos, rot6d

    def stats(self) -> dict:
        """Counters for the rig log / a take's meta. ``max_lead_m`` is the largest command→arm
        gap that was ever *carried*, i.e. after the clamp, so it must never exceed ``eps_m``
        by more than the pre-clamp measurement of the tick that hit the limit."""
        return {
            "mode": self.cfg.mode,
            "eps_m": self.cfg.eps_m,
            "n_resolved": self._n_resolved,
            "n_bootstrap": self._n_bootstrap,
            "n_clamped_pos": self._n_clamp_pos,
            "n_clamped_rot": self._n_clamp_rot,
            "n_contact_pullback": self._n_contact,
            "max_lead_m": self._max_lead_m,
            "carrying": self._pos is not None,
        }


# ------------------------------------------------------------------------------- rotation math
def _rot_toward(base_rot6d: np.ndarray, rot6d: np.ndarray, keep: float) -> np.ndarray:
    """``rot6d`` moved a fraction ``1 - keep`` of the way back to ``base_rot6d`` (geodesic)."""
    r_base = rot6d_to_rotmat(np.asarray(base_rot6d, dtype=np.float64))
    r_cur = rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64))
    rv = rotmat_to_rotvec(r_base.T @ r_cur)
    return rotmat_to_rot6d(r_base @ rotvec_to_rotmat(rv * float(keep)))


def _clamp_rot(base_rot6d: np.ndarray, rot6d: np.ndarray,
               eps_rad: float) -> tuple[np.ndarray, bool]:
    """Bound the geodesic deviation of ``rot6d`` from ``base_rot6d`` to ``eps_rad``."""
    r_base = rot6d_to_rotmat(np.asarray(base_rot6d, dtype=np.float64))
    r_cur = rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64))
    rv = rotmat_to_rotvec(r_base.T @ r_cur)
    ang = float(np.linalg.norm(rv))
    if eps_rad <= 0.0:
        return rotmat_to_rot6d(r_base), ang > 0.0
    if ang <= eps_rad:
        return np.asarray(rot6d, dtype=np.float64).reshape(6), False
    return rotmat_to_rot6d(r_base @ rotvec_to_rotmat(rv * (eps_rad / ang))), True
