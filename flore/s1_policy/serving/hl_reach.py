"""Reach compensation — make the real hand arrive where the HL policy asked it to.

THE SYMPTOM
-----------
On the rig the predicted chunk looks right in viser, the arm moves in the right direction,
but a *fine* adjustment (3-4 cm — the last move before a grasp) does not land: the hand
stops a couple of centimetres short and stays there. Coarse motion looks fine.

WHAT THE RECORDED DEPLOY TAKES ACTUALLY SHOW
--------------------------------------------
Measured on the 12 takes in ``data/deploy_takes/20260814_235118`` (a real G1 run of
``ckpt_0812_L6_bestval``, ACT + temporal ensembling), comparing the published wire against
the RecordTap joint readback FK'd to the UMI TCP. Both series are taken in the model frame
(the chunk's own ``anchor_pos`` IS the measured TCP), so the wire-vs-TCP lever offset
cancels and the numbers below are not confounded by it:

* **The arm is a VELOCITY follower, not a position follower.**
  ``corr(wire step, measured step)`` = **+0.37 / +0.41 / +0.26** per axis with a
  displacement gain of **0.80 / 0.94 / 0.80** — the arm reproduces how the target *moves*.
  But ``corr(lead, next tick motion)`` = **-0.18 / -0.04 / -0.01**: a standing position
  error produces essentially NO corrective motion. Over a 0.5 s window the shape is
  realized at **0.93** of what was asked.
* **So a position offset never closes.** The lead (wire minus measured) holds a persistent
  mean of **1.9 / 1.9 / 2.3 cm** across three different takes and is still there 44 s
  later — a steady-state error, not a transient lag. Fitting a first-order plant
  ``m += a*(w - m)`` returns a **negative** ``a`` on all three axes: the proportional
  model does not fit, because the offset is simply not what drives the arm.
* **Steady-state gain is ~1, so this is NOT a droop.** On lag-free samples (command parked
  under 1 cm/s) the static map is slope **1.009 / 1.187 / 1.057** — the workspace is not
  compressed. An earlier "gain 0.64" reading was pure lag artifact.

WHY THE LOOP CANNOT SEE ITS OWN SHORTFALL
-----------------------------------------
FLoRe's chunks are anchor-relative and step 0 is (by construction) the anchor, so every
Predict re-anchors at the robot's MEASURED pose (``_resync_anchor_from_robot``). Each new
chunk therefore says "move X from wherever you actually are" — which **forgives the
residual of the previous chunk**. Combined with a low level that ignores a standing offset,
the two form a loop with no integral action anywhere: the shortfall is re-forgiven at every
re-anchor and nothing ever asks for the missing centimetres. The 30 Hz ensembled cadence
re-anchors every tick, so it forgives fastest.

That is the whole bug: **the plan is right, the execution is short, and no one is keeping
score.** Ensembling itself is nearly innocent — disabling it changes the demanded path
length by only 1.08x (measured on smoothed paths), so it is not the cause.

WHAT THIS MODULE DOES
---------------------
Supplies the missing integral term, in the ONE place that faces hardware: the published
wire. :class:`ReachCompensator` accumulates the part of the commanded motion the arm did
not execute and adds it to the target, so the wire keeps insisting instead of resetting:

    residual_k = clamp( decay * residual_{k-1} + ki * (commanded_k - measured_k) )
    published_k = commanded_k + residual_k

It deliberately does **not** touch the model, the observation, the decode anchor, or the
chunk. The anchor stays on the measured pose (that is what keeps the model in
distribution — its proprioception must describe where the arm really is), and only the
outgoing target is nudged. Rotation is left alone entirely: the reported failure is
positional, and a wrist that insists on an un-executed rotation is a much worse thing to
get wrong on a physical arm.

THE FOUR SAFETY PROPERTIES, AND WHY EACH EXISTS
-----------------------------------------------
1. **Stall gating** (``stall_frac``) — "only when the arm is not moving". The residual grows only
   while the *realized fraction* over the last ``window`` steps,
   ``(Δmeasured · Δcommanded)/‖Δcommanded‖²``, is below ``stall_frac``. That is the operator's own
   rule: *commanded 6 cm, moved 3 cm → fine, leave it; commanded 5 cm, didn't move → accumulate.*

   The gate keys on MOTION, not on distance, and that distinction is load-bearing. The first
   version of this class gated on the shortfall DISTANCE (accumulate while the lead is under
   1.5 cm) and it was simply wrong: measured on the 12 recorded takes it caught **18 %** of the
   genuinely stalled steps while falsely accumulating on **24 %** of the steps where the arm was
   tracking fine. The realized-fraction gate scores **100 % / 0 %** on that same data. The reason
   is that ``corr(shortfall, realized fraction) = +0.077`` — the standing lead is ~2.5 cm whether
   the arm is tracking beautifully or frozen solid, because the ensembled blend always runs ahead
   of the hand. Distance does not carry the information the gate needs; "did it move" does.
1b. **Contact cut-out** (``contact_n``). A hand pressing on something immovable is "not moving" by
   every measure, so the stall gate OPENS on it — and integrating there commands a harder press.
   Checked before the stall gate for exactly that reason.
2. **Hard clamp** (``clamp_m``, default 3 cm). The residual is a norm-clamped vector, so the
   published target can never exceed the model's own target by more than this. This is what
   makes the failure mode bounded: with the compensator saturated the arm is asking for
   ``clamp_m`` more reach than the policy did, and no more, ever.
3. **Decay** (``decay``). Nothing is remembered forever. If the arm is against a fixture and
   truly cannot move, the residual saturates at the clamp rather than winding up, and it
   bleeds away once the gate opens or :meth:`reset` is called on a disarm/stop.

DEFAULT OFF. Constructed with ``ki = 0`` (or via :meth:`disabled`) the class is an exact
identity — :meth:`compensate` returns its input unchanged and no state is kept — so the
existing rig behaviour is byte-identical until an operator opts in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Largest residual (m) the operator may configure. A compensator that can add more than a
#: hand's width of unasked-for reach is not a fine-adjustment aid any more, and this runs on a
#: physical arm whose only true abort is the hardware E-stop.
MAX_CLAMP_M = 0.06


@dataclass(frozen=True)
class ReachConfig:
    """Tuning for :class:`ReachCompensator`.

    Parameters
    ----------
    ki:
        Integral gain per published step. ``0`` disables the compensator entirely. At 30 Hz a
        value of ``0.15`` closes a standing 2 cm shortfall to the clamp over roughly 1 s,
        which is the same order as the 1.67 s chunk span — fast enough to matter within one
        grasp approach, slow enough that a transient does not become a lunge.
    clamp_m:
        Hard norm limit (m) on the accumulated residual. This IS the safety bound: the
        published target never exceeds the model's target by more than this.
    stall_frac:
        **THE GATE — "is the arm actually moving?"** Accumulate only while the *realized
        fraction* over the last ``window`` steps is below this:

            realized = (Δmeasured · Δcommanded) / ‖Δcommanded‖²

        i.e. how much of the motion the wire asked for the arm actually performed, projected on
        the commanded direction. ``0.5`` means "integrate only when the arm did less than half of
        what was asked". This is the operator's own rule: *commanded 6 cm, moved 3 cm → fine,
        don't accumulate; commanded 5 cm, didn't move → accumulate.*

        Measured on the 12 recorded takes, this is the criterion that works and the shortfall
        distance is the one that does NOT. Gating on realized fraction catches **100 %** of the
        genuinely stalled steps with **0 %** false positives on steps where the arm was tracking
        well; gating on shortfall distance (<1.5 cm) caught only **18 %** of stalls while falsely
        accumulating on **24 %** of healthy steps. The reason is direct:
        ``corr(shortfall, realized fraction) = +0.077`` — the standing lead is ~2.5 cm whether the
        arm is tracking beautifully or not moving at all, because the ensembled blend always runs
        ahead of the hand. Distance simply does not carry the information the gate needs.
    window:
        How many published steps the realized fraction is measured over (default 9 ≈ 0.3 s at
        30 Hz). Long enough that a single-tick readback jitter cannot read as a stall, short
        enough to react inside one grasp approach.
    min_cmd_m:
        Minimum commanded displacement over the window (m) for the realized fraction to be
        meaningful. Below this the wire is effectively parked, ``0/0`` is undefined, and there is
        no "shortfall" to speak of — so the residual holds rather than integrating noise.
    gate_m:
        Optional **secondary** distance ceiling (m). ``0`` (the default) disables it. When set,
        the residual also stops growing once the shortfall exceeds it, which bounds compensation
        to fine adjustments even if the stall gate opens during a large motion.
    decay:
        Per-step multiplier applied to the stored residual (a leak). ``1.0`` is a pure
        integrator; the default leaks slowly so a stale residual cannot outlive the situation
        that produced it.
    contact_n:
        **Contact cut-out** (N). Above this measured force the residual STOPS growing and
        decays. This is the guard against the one genuinely dangerous confusion this class can
        make: "the arm did not reach its target" and "the arm is pressing on something that will
        not move" produce the SAME small standing shortfall, and integrating the second commands
        the hand to press harder. Measured on the recorded takes, 5–45 % of the steps that would
        accumulate (per take) had >1.5 N of live contact, so this is the common case, not a
        corner. ``0`` disables the cut-out — do not do that on hardware.
    """

    ki: float = 0.0
    clamp_m: float = 0.03
    stall_frac: float = 0.5
    window: int = 9
    min_cmd_m: float = 0.005
    gate_m: float = 0.0
    decay: float = 0.995
    contact_n: float = 1.5

    def __post_init__(self) -> None:
        for name in ("ki", "clamp_m", "stall_frac", "min_cmd_m", "gate_m", "decay", "contact_n"):
            v = float(getattr(self, name))
            if not np.isfinite(v):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)!r}")
            if v < 0.0:
                raise ValueError(f"{name} must be >= 0, got {v}")
        if int(self.window) < 2:
            raise ValueError(
                f"window must be >= 2 steps to measure a displacement, got {self.window}")
        if self.stall_frac > 1.0:
            raise ValueError(
                f"stall_frac={self.stall_frac} > 1 would treat an arm that OVERSHOOTS its target "
                "as stalled and keep pushing it further — the gate must be a fraction of the "
                "commanded motion")
        if self.clamp_m > MAX_CLAMP_M:
            raise ValueError(
                f"clamp_m={self.clamp_m} m exceeds the {MAX_CLAMP_M} m ceiling: this adds "
                "un-asked-for reach on a physical arm, so the bound is not operator-tunable "
                "past a fine adjustment.")
        if self.decay > 1.0:
            raise ValueError(
                f"decay={self.decay} > 1 would AMPLIFY the stored residual every step "
                "(an exponential runaway, not an integrator)")
        if self.ki > 0.0 and self.clamp_m <= 0.0:
            raise ValueError("ki > 0 with clamp_m = 0 can never move the target; set clamp_m > 0")

    @property
    def enabled(self) -> bool:
        return self.ki > 0.0


class ReachCompensator:
    """Accumulates un-executed commanded motion and adds it back to the published target.

    One instance per role. Feed the commanded and measured positions of each published step
    to :meth:`compensate`; it returns the position to actually put on the wire.

    Thread-safety: not synchronized. The rig calls it from whichever executor owns the wire
    at the time (they are mutually exclusive under ``busy``), which is the same discipline
    the temporal ensembler already relies on.
    """

    def __init__(self, config: ReachConfig | None = None) -> None:
        self.config = config if config is not None else ReachConfig()
        self._residual = np.zeros(3, dtype=np.float64)
        self._n_applied = 0
        self._n_gated = 0
        self._n_saturated = 0
        self._n_contact_blocked = 0
        # Rolling history for the realized-fraction gate: the UNCOMPENSATED commanded target and
        # the measured pose, one entry per published step. Bounded to window+1 entries, so this is
        # O(window) memory and cannot grow during a long rollout.
        self._hist_cmd: list[np.ndarray] = []
        self._hist_meas: list[np.ndarray] = []
        self._last_realized: float | None = None

    @classmethod
    def disabled(cls) -> ReachCompensator:
        """An explicit identity compensator (``ki = 0``)."""
        return cls(ReachConfig(ki=0.0))

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def residual(self) -> np.ndarray:
        """Current accumulated residual (m), a copy. Zeros when disabled."""
        return self._residual.copy()

    @property
    def residual_norm(self) -> float:
        return float(np.linalg.norm(self._residual))

    @property
    def saturated(self) -> bool:
        """True when the residual is at the clamp — the arm is asking for all the extra reach
        it is ever allowed. On the rig this reads as "it still is not getting there"."""
        if not self.enabled or self.config.clamp_m <= 0.0:
            return False
        return self.residual_norm >= self.config.clamp_m - 1e-9

    @property
    def realized_fraction(self) -> float | None:
        """How much of the last window's commanded motion the arm actually performed, or None
        when the window is not full yet / the wire was effectively parked. This is the quantity
        the gate keys on, exposed so the page can show WHY it is or is not accumulating."""
        return self._last_realized

    def stats(self) -> dict:
        """Counters for the page / a take's metadata. ``n_gated`` counts steps where the arm was
        judged to be TRACKING FINE (realized fraction at or above ``stall_frac``), i.e. exactly
        the steps the operator asked not to compensate."""
        return {
            "enabled": self.enabled,
            "ki": self.config.ki,
            "clamp_m": self.config.clamp_m,
            "stall_frac": self.config.stall_frac,
            "window": int(self.config.window),
            "gate_m": self.config.gate_m,
            "decay": self.config.decay,
            "contact_n": self.config.contact_n,
            "residual_m": self.residual_norm,
            "realized": self._last_realized,
            "saturated": self.saturated,
            "n_applied": self._n_applied,
            "n_gated": self._n_gated,
            "n_saturated": self._n_saturated,
            "n_contact_blocked": self._n_contact_blocked,
        }

    def reset(self) -> None:
        """Drop the accumulated residual AND the motion history. Call whenever the INTENT is
        abandoned — Stop, a disarm, a Reset-start-pose, the low-level-lost watchdog, or a new
        take. A residual that survives a stop would be applied to the next, unrelated motion;
        a surviving history would judge the new motion by the old one's progress."""
        self._residual = np.zeros(3, dtype=np.float64)
        self._hist_cmd.clear()
        self._hist_meas.clear()
        self._last_realized = None

    def _update_history(self, cmd: np.ndarray, meas: np.ndarray) -> float | None:
        """Append this step and return the realized fraction over the window, or None when it is
        not yet measurable (history too short, or the wire barely moved so 0/0 is undefined)."""
        w = int(self.config.window)
        self._hist_cmd.append(cmd.copy())
        self._hist_meas.append(meas.copy())
        if len(self._hist_cmd) > w + 1:
            del self._hist_cmd[0]
            del self._hist_meas[0]
        if len(self._hist_cmd) <= w:
            return None
        d_cmd = self._hist_cmd[-1] - self._hist_cmd[0]
        d_meas = self._hist_meas[-1] - self._hist_meas[0]
        n2 = float(d_cmd @ d_cmd)
        if n2 < self.config.min_cmd_m ** 2:
            return None  # wire effectively parked over the window — nothing was asked for
        return float((d_meas @ d_cmd) / n2)

    def compensate(self, commanded: np.ndarray, measured: np.ndarray | None,
                   force_n: float | None = None) -> np.ndarray:
        """Return the position to publish for one step.

        Parameters
        ----------
        commanded:
            ``(3,)`` the target this step would have published with no compensation (absolute
            pelvis-frame UMI TCP, model frame).
        measured:
            ``(3,)`` the robot's actual TCP at this instant, in the SAME frame — or ``None``
            when there is no fresh readback (RECORD off / stale / offline). ``None`` is not an
            error: it means the shortfall is unknowable this step, so the stored residual is
            re-applied unchanged (held, not grown) and nothing new is integrated.
        force_n:
            Measured contact force magnitude (N) on this hand, or ``None`` when unknown. At or
            above ``config.contact_n`` the residual decays instead of growing: a hand pressing
            on something that will not move shows the same small standing shortfall as one that
            merely fell short, and integrating THAT commands it to press harder. ``None`` is
            treated as "no contact information", which still accumulates — so on a rig where the
            F/T is not being read, prefer a smaller ``clamp_m``.

        Returns the compensated position. Disabled ⇒ ``commanded`` unchanged (a copy).

        A non-finite input is treated as "unknowable": the commanded value is returned and the
        residual is left alone, because feeding NaN into an accumulator would poison every
        later step — and ``is_safe_to_send`` downstream must remain the thing that decides
        whether a non-finite target reaches the wire, not this class silently repairing it.
        """
        cmd = np.asarray(commanded, dtype=np.float64).reshape(-1)
        if cmd.shape != (3,):
            raise ValueError(f"commanded must be (3,), got {np.shape(commanded)}")
        if not self.enabled:
            return cmd.copy()
        if not np.isfinite(cmd).all():
            return cmd.copy()

        if measured is None:
            # No fresh readback: hold the residual (do not integrate a shortfall we cannot see).
            return cmd + self._residual

        meas = np.asarray(measured, dtype=np.float64).reshape(-1)
        if meas.shape != (3,):
            raise ValueError(f"measured must be (3,), got {np.shape(measured)}")
        if not np.isfinite(meas).all():
            return cmd + self._residual

        cfg = self.config
        # The shortfall is measured against the UNCOMPENSATED target — the model's actual
        # intent. Using the compensated target instead would feed the residual back into its own
        # error term (an integrator on its own output), which double-counts and winds up.
        shortfall = cmd - meas
        residual = cfg.decay * self._residual
        # THE GATE: is the arm actually moving? Measured over a window, because a single tick
        # carries only ~0.7 mm of intended motion and mostly readback noise.
        realized = self._update_history(cmd, meas)
        self._last_realized = realized
        # CONTACT CUT-OUT first: a hand pressing on something immovable is "not moving" by every
        # measure, so the stall gate would OPEN on it. Integrating there commands a harder press.
        blocked = (cfg.contact_n > 0.0 and force_n is not None and np.isfinite(force_n)
                   and float(force_n) >= cfg.contact_n)
        # Optional secondary distance ceiling (off by default): keeps compensation to fine
        # adjustments even if the stall gate opens during a large motion.
        too_far = cfg.gate_m > 0.0 and float(np.linalg.norm(shortfall)) >= cfg.gate_m
        if blocked:
            self._n_contact_blocked += 1
        elif realized is None or too_far:
            # Not measurable yet (window not full / wire parked), or outside the distance
            # ceiling: hold rather than guess.
            self._n_gated += 1
        elif realized < cfg.stall_frac:
            # The arm did less than stall_frac of what was asked ⇒ it is NOT keeping up, so the
            # un-executed part is real and worth insisting on.
            residual = residual + cfg.ki * shortfall
            self._n_applied += 1
        else:
            # The arm IS tracking (commanded 6 cm, moved 3+ cm): the operator's explicit
            # instruction is to leave this alone — the motion is happening, it is merely lagging,
            # and adding to the target here would overshoot.
            self._n_gated += 1
        n = float(np.linalg.norm(residual))
        if cfg.clamp_m > 0.0 and n > cfg.clamp_m:
            residual = residual * (cfg.clamp_m / n)
            self._n_saturated += 1
        self._residual = residual
        return cmd + residual
