"""Funnel-shaped force-recovery augmentation: 1001-Demos' trajectory augmentation, in
WRENCH space instead of SE(3).

The problem this exists for
---------------------------
At deploy the low level does not track the commanded wrench well (``measure_exec_gap``'s
M8: tracking error 3.3-5.8 N against a commanded 4.8-5.5 N), so the force the planner
MEASURES on contact jumps somewhere its training distribution never went. 22.7% of deploy
force rows already fall outside the training ``[q01,q99]`` box (``docs/force_ood_design.md``
§1.2). Once there, the planner has never been shown what to DO about it, so the next chunk
is worse than the last — the cascade any OOD-recovery method is trying to break.

The fix that does not need new data: take a SUCCESSFUL contact window, keep its demonstrated
wrench trajectory as the thing to come BACK to, and synthesize many trajectories that start
somewhere out-of-distribution and converge onto it. That is exactly the shape of 1001-Demos
(``docs/papers/1001demos.pdf``, Pan et al., CoRL 2025), whose free-space augmentation samples
a novel start pose in a cone/ball around the demo's own start and optimizes a trajectory back
onto the demo's pre-contact pose under a *funnel loss* whose weight grows as ``(k/T)^3``.

The mapping, term by term
-------------------------
=========================================  ==================================================
1001-Demos (SE(3) end-effector trajectory) this module (wrench trajectory ``w = (F, T)``)
=========================================  ==================================================
``x^k in SE(3)``, k = 1..T                 ``u_k in R^6``, k = 0..H-1, on the contract's grid
demo trajectory ``O_ee``                   ``d_k`` = the stored ``hl_<role>_action_force`` chunk
``x_goal``, ONE fixed pre-contact pose     ``d_0`` -- the wrench at t0, the nominal force for
that every trajectory converges to         this contact (:attr:`FunnelConfig.target_mode`)
``x_init`` in a ``theta`` cone, radius r   :func:`sample_start`: F in a BALL of radius r,
                                           T rotated within a CONE of half-angle theta
``L_funnel``, ``w_k = w_min +
(w_max-w_min)(k/T)^3``                     :func:`funnel_weights` -- same schedule, verbatim
``L_smooth`` (velocity jerkiness)          the 2nd-difference term, ``lambda_acc``
``L_collision`` (scene TSDF)               :func:`wrench_consistency` -- the ``T = r x F``
                                           single-contact residual (force_ood_design §2.5)
``L_render`` (stay near training views)    DROPPED ON PURPOSE: leaving the training support
                                           at k=0 is the entire point here
=========================================  ==================================================

Three deliberate departures, each with its reason
-------------------------------------------------
1. **We augment INSIDE contact; 1001-Demos refuses to.** Its contact-aware module detects the
   first contact frame and augments only BEFORE it, "preserving the contact dynamics in the
   original demo" -- because it must RE-RENDER the scene and cannot simulate what a perturbed
   contact would look like. We re-render nothing: the wrench is not legible in a 224x224
   fisheye frame the way a pose change is, so the action-view consistency problem that forces
   1001-Demos into 3D Gaussian Splatting does not bite this channel. Contact is where our OOD
   lives, so contact is where we augment.

2. **The funnel target is a WRENCH the data actually contains, never a fitted line.** The
   intuition that a contact ramp is "linear from 0 to 5 N", so a line could stand in for it,
   does not survive measurement: over the 1.667 s H=50 window,
   ``max|F - line(F_0 -> F_{H-1})|`` is **0.59x the total excursion at the median** (2898
   contact windows of ``session_2026.08.23_merged_v3``). A line is off by more than half the
   thing it is meant to represent. 1001-Demos uses linear interpolation only as the
   *initialization* of a nonconvex optimizer. :func:`linear_reference` builds the line anyway
   -- purely so a viewer can DRAW the naive alternative next to the real one.
   *Which* wrench is the target is :attr:`FunnelConfig.target_mode`, and it is the one real
   modelling choice here: ``d_0`` (the default; one fixed setpoint, the paper's own shape) or
   ``d_k`` step by step. They are 86 degrees apart at the median -- see that field.

3. **The optimum is closed-form, so there is no optimizer.** Funnel + smoothness are both
   quadratic and the endpoints are hard equality constraints, so the minimizer is one small
   linear solve per window (:func:`solve_funnel_trajectory`) -- exact, deterministic, and
   independent of any initialization. 1001-Demos needs an iterative solver because a TSDF
   collision term and the SE(3) manifold make its problem nonconvex; neither applies here.

Scope, stated so it cannot be mistaken
--------------------------------------
This augments the FORCE channel only: it produces ``(force_history, action_force)`` pairs and
leaves the image, the state and the pose action columns at the demo's own values. So it
teaches "whatever wrench you find yourself measuring, the pose plan is the demonstrated one
and the force plan funnels back onto it" -- the right lesson for a force-recovery skill under
position-primary control, and an assumption rather than a fact. It is the wrench analogue of
1001-Demos' own ``Aug Action Only`` baseline, and it is honest about that.

**Numpy only, on purpose.** No torch, no zarr, no ``fuji.s1_policy`` import -- so the
UMI_collection replay viewer (a different repo, a different venv) can import this exact code
instead of carrying a second, drifting copy of the funnel. Same reasoning as
:mod:`fuji.utils.muse_wire`. A drifting copy would be silent here in the worst way: the viewer
is the acceptance test for the augmentation, so a viewer that draws a *different* funnel than
training consumes is worse than no viewer.

Nothing in the training path imports this yet -- it is a generator plus its viewer, and wiring
it into ``HLDataset`` is a separate, deliberate step.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

#: Width of one wrench: ``[Fx Fy Fz Tx Ty Tz]``. Duplicated from
#: ``fuji.s1_policy.data.layout.FORCE_DIM`` rather than imported, because that module's package
#: ``__init__`` pulls in torch and this file must stay importable from the collection venv.
#: ``tests/test_force_funnel.py`` pins the two together, exactly as ``test_force_frame.py``
#: pins the duplicated ``SENSOR_TO_TCP`` in the replay tools.
WRENCH_DIM = 6

#: Column split inside one wrench. The wrench lives in the **bota sensor frame** end to end
#: (``attrs["hl_force_frame"] == "bota_sensor"``); nothing here rotates it, so a generated
#: chunk is directly comparable to the stored one. Drawing it in TCP axes is the viewer's job,
#: via :mod:`fuji.utils.force_frame`.
FORCE_SLICE = slice(0, 3)
TORQUE_SLICE = slice(3, 6)

#: Smallest horizon that has any free interior step once BOTH endpoints are pinned. H=3 leaves
#: exactly one, which is degenerate (no 2nd difference); 4 is the first horizon where the
#: funnel is a curve rather than a corner.
MIN_HORIZON = 4

#: Legal values of :attr:`FunnelConfig.target_mode`. See that field for what they mean and for
#: the 86-degree measurement that makes them genuinely different.
TARGET_MODES = ("anchor", "demo")

#: Per-mode ``lambda_acc``, each set where the GENERATED peak-slew distribution lands on the
#: demonstrations' own (see :attr:`FunnelConfig.lambda_acc` for why that is the criterion).
#: The two modes need different values because they ask for different amounts of motion: a
#: constant setpoint is reached and then held, while a moving reference has to be tracked all
#: the way. Measured over 315 contact-gated windows against a demo peak of 27.2 N/s (p50):
#: ``anchor`` at 0.5 -> 28.3 N/s, ``demo`` at 4.0 -> 29.3 N/s. One shared value would leave
#: ``anchor`` at 21.1 N/s, i.e. 22% SLOWER than any human demo -- the under-response that
#: force_ood_design §1.4 measured as the actual failure.
LAMBDA_ACC_BY_TARGET_MODE = {"anchor": 0.5, "demo": 4.0}


@dataclasses.dataclass(frozen=True)
class FunnelConfig:
    """Every knob, with defaults measured on ``session_2026.08.23_merged_v3`` (219 episodes,
    3566 usable H=50 windows sampled every 25th anchor) rather than guessed.

    The paper's own settings, for reference: rotation bound swept over
    ``{20,30,40,50,60} deg`` with success plateauing at **50 deg** (Fig. A1), position radius
    0.15 m in simulation and "the Euclidean distance between the initial demonstration start
    pose and the pre-contact pose" in the real-world runs -- i.e. **the radius is the demo's
    own excursion**, which is why :attr:`force_radius_scale` multiplies the measured ramp
    instead of naming an absolute newton value. 20 augmented episodes per demo in simulation,
    ~145 in the real-world runs; :attr:`n_aug` defaults lower because our unit is a 1.667 s
    window and a session yields tens of thousands of them.
    """

    #: Augmented trajectories per contact window.
    n_aug: int = 8

    # --- where the OOD start comes from (1001-Demos' cone-and-ball, in wrench space) -------
    #: Ball radius as a multiple of the demo's own force excursion ``|F_{H-1} - F_0|``. That
    #: excursion has p50 4.43 N here, so 2.0 puts the median sampling radius near 8.9 N --
    #: enough to produce a start like ``(-8, +2, +1)`` against a ``+5 N`` target, i.e. wrong
    #: sign and wrong magnitude, which is the failure this augmentation is aimed at.
    force_radius_scale: float = 2.0
    #: Floor, so a window whose force barely moves still gets a meaningful perturbation.
    force_radius_min_n: float = 2.0
    #: Ceiling. Kept well under ``FORCE_ESTOP_N`` (40 N) -- see :attr:`max_force_n`.
    force_radius_max_n: float = 20.0
    #: Sample in the ANNULUS ``[frac*r, r]``, not the whole ball. Uniform-in-volume already
    #: concentrates near the shell, but a bare ball still spends samples on starts that are
    #: barely out of distribution and teach nothing. Set to 0.0 for the full ball.
    force_radius_min_frac: float = 0.35

    #: Half-angle of the cone the demo's TORQUE direction is perturbed within -- the paper's
    #: ``theta`` quaternion cone, at the paper's own plateau value of 50 deg. Torque maps to
    #: the ROTATION half of the pose because a torque is what a rotation of the contact does;
    #: force maps to the translation half.
    torque_cone_deg: float = 50.0
    #: Torque magnitude is scaled by a LOG-uniform factor in ``[1/k, k]``, so scaling up and
    #: down are equally likely. A pure direction perturbation would miss the magnitude axis,
    #: and force_ood_design §1.2 measured deploy OOD to be a magnitude problem first.
    torque_mag_scale_max: float = 2.5
    #: Used instead of the cone when the demo torque is ~0 and has no direction to perturb:
    #: the start torque is then drawn from an annulus of this radius.
    torque_radius_min_nm: float = 0.15
    #: Largest fitted moment arm ``|r|`` that is still trusted to carry the force perturbation
    #: into a torque. Measured on this dataset: ``|r|`` p50 206 mm, p90 302 mm, max 409 mm
    #: against a tool 242 mm from the sensor -- so 0.5 m is a "this fit went wrong" guard, not
    #: a shaping term. Beyond it :func:`sample_start` falls back to perturbing the demo's own
    #: torque instead of the implied one.
    max_moment_arm_m: float = 0.5

    # --- the funnel (paper §3.2, verbatim schedule) ------------------------------------------
    #: ``w_k = w_min + (w_max - w_min) * (k/(H-1))**power``, then normalized to sum to 1 (the
    #: paper asks for both ``sum w_k = 1`` and ``0 <= w_min,w_max <= 1``, which only hold
    #: together after a normalization). ``w_min = 0`` leaves step 0 completely free -- the wide
    #: mouth of the funnel.
    w_min: float = 0.0
    w_max: float = 1.0
    #: The cubic is the paper's. Raising it merges later (a longer wide mouth), lowering it
    #: merges sooner. :func:`merge_step` reports where a given setting actually merges.
    funnel_power: float = 3.0

    #: WHAT the funnel converges onto — the one genuine modelling choice in this module.
    #:
    #: ``"anchor"`` (default): the target is the single wrench ``d[0]`` — the force at t0, the
    #: nominal wrench for *this* contact, and the vector the viewer's red ``F^cmd`` arrow draws.
    #: Every trajectory converges to the SAME point, which is 1001-Demos' own shape: its
    #: ``x_goal`` is one fixed pre-contact pose and the paper's words are "converge consistently
    #: to **the same** pre-contact pose of the original demonstration". The augmented sample
    #: then teaches a *recovery manoeuvre* — "you are off-nominal; here is a 1.67 s plan that
    #: puts the contact force back where it belongs" — and deliberately does NOT try to
    #: reproduce the stroke the demo went on to make, since that stroke is already covered by
    #: the unaugmented data.
    #:
    #: ``"demo"``: the reference is the demo chunk step by step, ``d[k]``, terminating on
    #: ``d[H-1]``. This is "re-plan the demonstrated stroke from a bad force".
    #:
    #: The two are far apart, and by a measured amount: over one 1.667 s chunk the demonstrated
    #: force turns a median **86 deg** (>45 deg in 87% of windows) and drifts 5.59 N while
    #: ``|F(t0)|`` is only 3.63 N — a wiping stroke reverses direction inside the horizon. So
    #: ``"anchor"`` holds a setpoint the demo itself leaves, and ``"demo"`` chases a stroke the
    #: off-nominal robot arguably should not be continuing. Both are defensible; they are a
    #: clean two-arm experiment on one axis, which is why this is a switch and not a rewrite.
    target_mode: str = "anchor"

    #: Weight on the 1st difference. Off by default: the 2nd-difference term alone already
    #: bounds the slew rate, and a velocity penalty biases the trajectory toward standing
    #: still, which fights the funnel.
    lambda_vel: float = 0.0
    #: Weight on the 2nd difference -- 1001-Demos' ``L_smooth`` ("penalize velocity
    #: jerkiness"). Both this and the funnel term are averaged over their own term counts, so
    #: the value means the same thing at H=30 and H=50.
    #:
    #: **Chosen by matching the demonstrations' own slew, not by taste.** The tempting
    #: criterion -- "make the funnel's mouth wide, merge late" -- is wrong here: a trajectory
    #: that dawdles on its way back is one that changes force more slowly than any human demo
    #: ever did, and teaching a policy to under-respond is the exact failure
    #: ``docs/force_ood_design.md`` §1.4 measured (a force head with a gain of 0.3). So
    #: lambda is set where the generated slew distribution lands on the demo's. Over 219
    #: contact windows of ``session_2026.08.23_merged_v3``, at 4.0:
    #:
    #: =====================  ==========  ==========
    #: peak ``|dF/dt|``       demo        generated
    #: =====================  ==========  ==========
    #: p50                    29.7 N/s    30.1 N/s
    #: p90                    54.3 N/s    44.7 N/s
    #: mean-over-window p50   9.2 N/s     9.7 N/s
    #: =====================  ==========  ==========
    #:
    #: The consequence is a merge at ~20% of the horizon (0.33 s), which is not the funnel
    #: failing to be wide -- it is how long a demo-speed force takes to cross a ~7 N gap.
    #:
    #: ``None`` (the default) means "the value measured for this ``target_mode``" --
    #: :data:`LAMBDA_ACC_BY_TARGET_MODE`. Read :attr:`resolved_lambda_acc`, never this field, or
    #: a ``None`` reaches the solver. Set a float to override.
    lambda_acc: float | None = None

    # --- admissibility (the analogue of the paper's collision constraint) --------------------
    #: Reject a trajectory that exceeds this. Deliberately ~2x the largest force in the whole
    #: dataset (13.2 N) -- going beyond the data IS the point -- while staying far below
    #: holosoma's ``FORCE_ESTOP_N`` of 40 N, so nothing this generates could teach the policy
    #: to ask for a wrench that would trip the e-stop.
    max_force_n: float = 25.0
    #: Same idea for torque; the dataset's own worst case is 1.63 N.m.
    max_torque_nm: float = 5.0
    #: Slew ceiling. The demo chunks themselves peak at 175 N/s (p50 28.7), so this is a loose
    #: safety net that only fires on a pathological lambda, not a shaping term.
    max_dforce_n_per_s: float = 200.0
    max_dtorque_nm_per_s: float = 40.0

    #: ``T = r x F`` admissibility, as force_ood_design §2.5 proposes: only accept a generated
    #: wrench whose single-contact residual is within ``slack`` of **the reference's own**
    #: residual. Relative, not absolute, so the augmentation is never held to a higher physical
    #: standard than the thing it converges onto -- an absolute threshold would silently reject
    #: every sample from a window that was already messy.
    #:
    #: "The reference's" and not "the window's", and the difference is large. A single wrench
    #: has a torque component parallel to its own force that **no lever can produce**, so under
    #: ``target_mode="anchor"``, where the reference is the single wrench ``d[0]``, its residual
    #: can far exceed the window average that a whole-chunk fit reports. Budgeting against the
    #: window instead rejected **40-44%** of draws and left slots unfilled — it was demanding
    #: that the augmentation be a cleaner point contact than its own target.
    consistency_slack: float = 1.5
    #: Absolute floor for that budget, so a freakishly clean source window (residual ~0) does
    #: not make its own augmentations impossible to satisfy.
    consistency_floor: float = 0.5
    #: Experimental arm: overwrite the generated torque with ``r_fit x F`` so the wrench is
    #: single-contact-consistent BY CONSTRUCTION instead of by rejection. Off by default
    #: because it breaks the terminal constraint on T by however much the fit is wrong
    #: (residual/|T| p50 = 0.305 on this dataset), which is a real cost, not a rounding error.
    project_torque: bool = False

    #: Resampling budget per requested trajectory before that slot is given up on. Exhausting
    #: it is REPORTED (:class:`RecoveryBatch.n_rejected`), never silently absorbed. At the
    #: defaults, 12.9% of draws are thrown away (190 ``inconsistent_wrench`` + 69
    #: ``torque_over_max`` out of 2011) and **every** window still delivers its full
    #: ``n_aug`` -- so 20 is slack, not a binding limit.
    max_tries_per_sample: int = 20

    @property
    def resolved_lambda_acc(self) -> float:
        """:attr:`lambda_acc`, or the value measured for this :attr:`target_mode` if it is None."""
        if self.lambda_acc is not None:
            return float(self.lambda_acc)
        return LAMBDA_ACC_BY_TARGET_MODE[self.target_mode]

    def __post_init__(self) -> None:
        if self.n_aug < 1:
            raise ValueError(f"n_aug must be >= 1, got {self.n_aug}")
        if not 0.0 <= self.w_min <= 1.0 or not 0.0 <= self.w_max <= 1.0:
            raise ValueError(f"w_min/w_max must lie in [0,1], got {self.w_min}/{self.w_max}")
        if self.w_max <= 0.0:
            raise ValueError("w_max must be > 0 or the funnel never pulls toward the demo")
        if self.funnel_power <= 0.0:
            raise ValueError(f"funnel_power must be > 0, got {self.funnel_power}")
        if self.target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {self.target_mode!r}")
        if self.lambda_vel < 0.0 or (self.lambda_acc is not None and self.lambda_acc < 0.0):
            raise ValueError("smoothness weights must be >= 0")
        if self.lambda_vel == 0.0 and self.resolved_lambda_acc == 0.0:
            # Not merely a bad setting -- it stops being a trajectory problem. With no
            # smoothness the objective is DIAGONAL, so every interior step minimizes its own
            # term independently and lands exactly on ``d_k``. The result is the demo with a
            # step discontinuity at k=0: no mouth, no funnel, and a jump no plant could track.
            # Refused here rather than caught downstream, where the symptom (a 1-step spike)
            # looks like a data bug instead of a configuration one.
            raise ValueError(
                "lambda_vel and lambda_acc cannot both be 0: without a smoothness term every "
                "interior step snaps to the demo independently, which is a step discontinuity "
                "at k=0, not a funnel"
            )
        if not 0.0 <= self.force_radius_min_frac < 1.0:
            raise ValueError(f"force_radius_min_frac must lie in [0,1), got {self.force_radius_min_frac}")
        if not 0.0 < self.torque_cone_deg <= 180.0:
            raise ValueError(f"torque_cone_deg must lie in (0,180], got {self.torque_cone_deg}")
        if self.torque_mag_scale_max < 1.0:
            raise ValueError(f"torque_mag_scale_max must be >= 1, got {self.torque_mag_scale_max}")
        if self.max_tries_per_sample < 1:
            raise ValueError("max_tries_per_sample must be >= 1")


@dataclasses.dataclass
class RecoveryBatch:
    """What :func:`generate_recovery_chunks` produced for ONE source window.

    ``n_rejected`` and ``reject_reasons`` exist so a caller can report what it dropped. A
    generator that silently returns fewer trajectories than asked reads as "the data supports
    this everywhere", which is the kind of quiet truncation that makes an augmentation look
    better than it is.
    """

    #: ``(n, H, 6)`` generated wrench trajectories, sensor frame, model grid.
    chunks: np.ndarray
    #: ``(n, 6)`` the OOD starts. Equal to ``chunks[:, 0]`` by construction (the start is a
    #: hard constraint) and kept separately because it is ALSO the force observation to pair
    #: with the chunk -- see :func:`observation_for`.
    starts: np.ndarray
    #: ``(H, 6)`` the source demo chunk, verbatim from the zarr. This is CONTEXT — what the
    #: demonstration actually did over the window — and under ``target_mode="anchor"`` it is
    #: **not** what the trajectories converge onto. Read :attr:`reference` for that.
    demo: np.ndarray
    #: ``(H, 6)`` what the funnel actually aimed at (:func:`funnel_reference` of ``demo``).
    #: Equal to ``demo`` in ``"demo"`` mode; a constant hold of ``demo[0]`` in ``"anchor"`` mode.
    #: Carried separately and explicitly because a viewer that drew ``demo`` and called it "the
    #: target" would be drawing the wrong curve in the default mode.
    reference: np.ndarray
    #: ``(n,)`` first step at which each trajectory has closed 90% of its initial gap.
    merge_steps: np.ndarray
    #: How many candidate draws were thrown away, and why.
    n_rejected: int = 0
    reject_reasons: dict[str, int] = dataclasses.field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.chunks.shape[0])


# ---------------------------------------------------------------------------
# The funnel weights and the closed-form solve
# ---------------------------------------------------------------------------
def funnel_weights(horizon: int, *, w_min: float = 0.0, w_max: float = 1.0,
                   power: float = 3.0) -> np.ndarray:
    """1001-Demos' funnel schedule ``w_k = w_min + (w_max - w_min) (k/T)^power``, normalized.

    Near-zero at k=0 (the trajectory is free to start anywhere -- the funnel's mouth) and
    largest at k=H-1 (it must agree with the demo -- the funnel's throat). The paper states
    both ``sum_k w_k = 1`` and ``0 <= w_{min,max} <= 1``; those are jointly satisfiable only
    by normalizing after forming the shape, which is what happens here. Normalizing also makes
    :attr:`FunnelConfig.lambda_acc` mean the same thing at every horizon.
    """
    if horizon < MIN_HORIZON:
        raise ValueError(f"horizon must be >= {MIN_HORIZON}, got {horizon}")
    k = np.arange(horizon, dtype=np.float64) / float(horizon - 1)
    raw = w_min + (w_max - w_min) * np.power(k, power)
    total = float(raw.sum())
    if total <= 0.0:
        raise ValueError("funnel weights summed to zero; w_max must be > 0")
    return raw / total


def _difference_matrix(horizon: int, order: int) -> np.ndarray:
    """``(H-order, H)`` finite-difference operator, so ``D @ u`` is the k-th difference."""
    d = np.eye(horizon, dtype=np.float64)
    for _ in range(order):
        d = np.diff(d, axis=0)
    return d


#: Cache of the factorizable part of the solve. The system matrix depends ONLY on
#: ``(horizon, weights, lambda_vel, lambda_acc)`` -- not on the demo, not on the start -- so
#: thousands of windows at one config reuse one build. Keyed on the weights' bytes because a
#: float tuple of length 50 is a clumsy key and the array is already contiguous float64.
_SYSTEM_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _funnel_system(horizon: int, weights: np.ndarray, lambda_vel: float,
                   lambda_acc: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(M, K_inv_ready)`` for the quadratic funnel problem.

    Minimizing, per axis and with ``u_0`` / ``u_{H-1}`` pinned::

        J(u) = sum_k a_k (u_k - d_k)^2
             + (lam_v/(H-1)) sum_k (u_{k+1} - u_k)^2
             + (lam_a/(H-2)) sum_k (u_{k+1} - 2 u_k + u_{k-1})^2

    is ``J = u' M u - 2 u' A d + const`` with ``M = A + G``. Writing ``u = S v + c`` for the
    free interior ``v`` and the boundary-carrying ``c``, the stationarity condition is
    ``(S' M S) v = S' (A d - M c)``, and ``S' M S`` is just ``M`` with the two boundary rows
    and columns struck out. Returns ``M`` and that interior block, both float64.

    Note what does NOT appear: any weighting between force and torque. The objective is
    diagonal in the axis index, so the six channels solve independently and the solution is
    homogeneous of degree 1 in ``(d, u_0)``. Newtons and newton-metres therefore need no
    common normalizer -- each axis funnels in its own units, identically shaped.
    """
    key = (int(horizon), weights.tobytes(), float(lambda_vel), float(lambda_acc))
    hit = _SYSTEM_CACHE.get(key)
    if hit is not None:
        return hit
    m = np.diag(weights).astype(np.float64)
    if lambda_vel > 0.0:
        d1 = _difference_matrix(horizon, 1)
        m += (lambda_vel / (horizon - 1)) * (d1.T @ d1)
    if lambda_acc > 0.0:
        d2 = _difference_matrix(horizon, 2)
        m += (lambda_acc / (horizon - 2)) * (d2.T @ d2)
    interior = m[1:horizon - 1, 1:horizon - 1].copy()
    # Numerical backstop, not the primary guard. With either lambda > 0 the interior block is
    # positive definite (the clamped biharmonic operator is), so ``FunnelConfig``'s refusal of
    # "both lambdas zero" is what actually keeps this reachable code unreachable through the
    # public API. It still fires for a direct call with a pathological weight vector -- e.g. a
    # ``funnel_power`` large enough to underflow the low-k weights to exactly 0 -- where
    # ``np.linalg.solve`` would otherwise raise a LinAlgError far from its cause.
    if np.linalg.cond(interior) > 1e12:
        raise ValueError(
            "the funnel system is singular: nothing constrains the interior steps. Raise "
            "lambda_acc (or lambda_vel, or w_min), or lower funnel_power."
        )
    out = (m, interior)
    _SYSTEM_CACHE[key] = out
    return out


def funnel_reference(demo: np.ndarray, target_mode: str = "anchor") -> np.ndarray:
    """The ``(H, 6)`` trajectory the funnel pulls toward, and whose last row it must END on.

    ``"anchor"`` holds ``d[0]`` — the wrench at t0 — for every step, so every generated
    trajectory converges to ONE point (1001-Demos' single ``x_goal``). ``"demo"`` returns the
    demo chunk unchanged, so the funnel tracks the demonstrated stroke step by step.

    Everything downstream reads the reference and never the raw demo, which is what keeps the
    two modes from being half-applied: the funnel weights, the terminal constraint and
    :func:`merge_step` all measure against the same array.
    """
    demo = np.asarray(demo, dtype=np.float64)
    if target_mode == "demo":
        return demo
    if target_mode == "anchor":
        return np.repeat(demo[:1], demo.shape[0], axis=0)
    raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {target_mode!r}")


def solve_funnel_trajectory(demo: np.ndarray, start: np.ndarray, *,
                            cfg: FunnelConfig | None = None) -> np.ndarray:
    """The exact minimizer of the funnel objective: ``(H, 6)``, starting at ``start`` and
    ending bit-exact on the reference's last row (see :func:`funnel_reference`).

    One dense solve of size ``(H-2, H-2)`` with a 6-column right-hand side -- 48x48 at the
    contract's H=50, so the cost is irrelevant next to reading the window off disk.
    """
    cfg = cfg or FunnelConfig()
    demo = np.asarray(demo, dtype=np.float64)
    if demo.ndim != 2 or demo.shape[1] != WRENCH_DIM:
        raise ValueError(f"demo must be (H,{WRENCH_DIM}), got {demo.shape}")
    horizon = int(demo.shape[0])
    if horizon < MIN_HORIZON:
        raise ValueError(f"horizon must be >= {MIN_HORIZON}, got {horizon}")
    start = np.asarray(start, dtype=np.float64).reshape(-1)
    if start.shape[0] != WRENCH_DIM:
        raise ValueError(f"start must be ({WRENCH_DIM},), got {start.shape}")

    reference = funnel_reference(demo, cfg.target_mode)
    weights = funnel_weights(horizon, w_min=cfg.w_min, w_max=cfg.w_max, power=cfg.funnel_power)
    m, interior = _funnel_system(horizon, weights, cfg.lambda_vel, cfg.resolved_lambda_acc)

    # c carries the two hard endpoint constraints; the interior is what we solve for.
    c = np.zeros((horizon, WRENCH_DIM), dtype=np.float64)
    c[0] = start
    c[-1] = reference[-1]
    rhs = (weights[:, None] * reference - m @ c)[1:horizon - 1]
    out = c.copy()
    out[1:horizon - 1] = np.linalg.solve(interior, rhs)
    return out


def linear_reference(demo: np.ndarray, start: np.ndarray,
                     target_mode: str = "anchor") -> np.ndarray:
    """The straight line from ``start`` to the funnel's target -- the naive recovery, for CONTRAST.

    This is 1001-Demos' trajectory *initialization*, not its objective, and it is not what
    this module generates. It exists so a viewer can draw the obvious alternative next to the
    funnel and make the difference visible: the line ignores the demo's shape entirely, and
    the demo's shape is not a line (``max|F - line|`` is 0.59x the excursion at the median on
    ``session_2026.08.23_merged_v3``).
    """
    demo = np.asarray(demo, dtype=np.float64)
    target = funnel_reference(demo, target_mode)[-1:]
    start = np.asarray(start, dtype=np.float64).reshape(1, -1)
    horizon = int(demo.shape[0])
    t = (np.arange(horizon, dtype=np.float64) / float(horizon - 1))[:, None]
    return start + (target - start) * t


def merge_step(chunk: np.ndarray, reference: np.ndarray, *, tol_frac: float = 0.1) -> int:
    """First step at which ``chunk`` has closed ``1 - tol_frac`` of its initial gap to ``reference``.

    Pass the **reference** (:func:`funnel_reference`), not the raw demo — under
    ``target_mode="anchor"`` the two differ, and measuring convergence against a trajectory the
    funnel was never aiming at would report a merge that did not happen.

    The one number that says whether a funnel setting is usable: a trajectory that only meets
    the demo on the very last step never shows the policy a converged state, since the chunk
    is re-planned long before it ends. Returns ``H-1`` if it never gets there earlier.
    """
    chunk = np.asarray(chunk, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    gap = np.linalg.norm(chunk - reference, axis=1)
    gap0 = float(gap[0])
    if gap0 <= 1e-12:
        return 0
    below = np.nonzero(gap <= tol_frac * gap0)[0]
    return int(below[0]) if below.size else int(chunk.shape[0] - 1)


# ---------------------------------------------------------------------------
# Sampling the out-of-distribution start (the cone and the ball)
# ---------------------------------------------------------------------------
def _unit_vectors(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """``(n,3)`` directions uniform on the sphere, via normalized isotropic Gaussians."""
    v = rng.standard_normal((n, 3))
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    # A Gaussian draw is zero with probability zero, but "probability zero" is not "never"
    # over millions of windows, and a zero here would produce a NaN direction.
    degenerate = norm[:, 0] < 1e-12
    if degenerate.any():
        v[degenerate] = np.array([1.0, 0.0, 0.0])
        norm[degenerate] = 1.0
    return v / norm


def _annulus_radius(rng: np.random.Generator, radius: float, min_frac: float) -> float:
    """Radius uniform in the VOLUME of the shell ``[min_frac*r, r]`` (so ``r^3`` interpolates)."""
    lo = float(min_frac) ** 3
    return float(radius) * float(lo + (1.0 - lo) * rng.random()) ** (1.0 / 3.0)


def _orthonormal_pair(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to ``axis``."""
    # Cross with whichever world axis is LEAST aligned with `axis`; crossing with the most
    # aligned one is what produces a near-zero vector and a garbage basis.
    helper = np.zeros(3)
    helper[int(np.argmin(np.abs(axis)))] = 1.0
    e1 = np.cross(axis, helper)
    e1 /= max(np.linalg.norm(e1), 1e-12)
    return e1, np.cross(axis, e1)


def sample_in_cone(axis: np.ndarray, half_angle_rad: float, rng: np.random.Generator) -> np.ndarray:
    """A unit vector uniform on the spherical cap of half-angle ``half_angle_rad`` about ``axis``.

    This is 1001-Demos' "``theta`` quaternion cone", applied to a vector: the set of directions
    reachable from ``axis`` by a rotation of at most ``theta``. Uniform in SOLID ANGLE (via
    ``cos psi``), not in the angle itself -- sampling the angle uniformly would pile draws up
    near the cone's axis and under-cover its rim, which is exactly the part we want.
    """
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return _unit_vectors(rng, 1)[0]
    axis = axis / norm
    cos_max = math.cos(float(half_angle_rad))
    cos_psi = 1.0 - rng.random() * (1.0 - cos_max)
    sin_psi = math.sqrt(max(0.0, 1.0 - cos_psi * cos_psi))
    phi = 2.0 * math.pi * rng.random()
    e1, e2 = _orthonormal_pair(axis)
    return cos_psi * axis + sin_psi * (math.cos(phi) * e1 + math.sin(phi) * e2)


def _radius_basis(demo: np.ndarray, target_mode: str) -> float:
    """How far the DEMO itself had to travel to reach the funnel's target, in newtons.

    1001-Demos sets its ball radius to "the Euclidean distance between the initial
    demonstration start pose and the pre-contact pose" — i.e. the demo's own displacement to
    the goal — which is why the radius is a MULTIPLE of this rather than an absolute value.
    The right displacement depends on what the goal is:

    * ``"demo"``: the goal is ``d[H-1]``, so the displacement is ``|d[H-1] - d[0]|``, the force's
      excursion over the chunk (p50 4.43 N on this dataset).
    * ``"anchor"``: the goal is ``d[0]`` itself, so the chunk's excursion is the wrong quantity
      — the reference does not move at all. The displacement that reached this setpoint is the
      ramp from no contact up to it, i.e. ``|d[0]|`` (p50 3.63 N on contact-gated anchors).
    """
    demo = np.asarray(demo, dtype=np.float64)
    if target_mode == "demo":
        return float(np.linalg.norm(demo[-1, FORCE_SLICE] - demo[0, FORCE_SLICE]))
    if target_mode == "anchor":
        return float(np.linalg.norm(demo[0, FORCE_SLICE]))
    raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {target_mode!r}")


def sample_start(demo: np.ndarray, rng: np.random.Generator, *,
                 cfg: FunnelConfig | None = None,
                 moment_arm: np.ndarray | None = None) -> np.ndarray:
    """One out-of-distribution wrench start ``(6,)`` for the window ``demo``.

    Force -- the TRANSLATION half of 1001-Demos' pose perturbation -- is displaced from the
    demo's own ``F_0`` by a direction uniform on the sphere and a radius drawn from the
    annulus ``[frac*r, r]``, with ``r`` set by the demo's own excursion
    (:attr:`FunnelConfig.force_radius_scale`), which is the paper's real-world rule for its
    ball radius. Centering on ``F_0`` rather than on zero is likewise the paper's choice: it
    samples "within a theta quaternion cone around EACH original viewpoint".

    Torque -- the ROTATION half -- is **not** perturbed independently, and that correction was
    forced by measurement. Drawing ``F`` and ``T`` independently produced wrenches that no
    contact can exert, and the ``T = r x F`` gate threw away **50%** of them (334 of 634 draws
    across 75 windows -- every single rejection was ``inconsistent_wrench``). A force of
    ``(-8, +2, +1)`` at the demo's contact point implies a specific torque; keeping the demo's
    torque next to it asserts the lever changed to exactly cancel the force change.

    So the start torque is built from the force: ``T = s * R_cone(theta) @ (r_fit x F_start)``,
    where ``r_fit`` is the contact point :func:`wrench_consistency` fitted to this window. The
    cone and the log-uniform scale then represent a lever that MOVED -- which is what a
    mis-executed contact actually does -- rather than a torque unrelated to its own force.

    Two properties this buys, both exact:

    * ``r x F`` is perpendicular to ``F``, and *any* torque perpendicular to a force is
      realizable by *some* moment arm. So the base start wrench is always physical, however
      badly conditioned ``r_fit`` happens to be (a demo whose force barely rotates only
      determines ``r`` up to a multiple of ``F``, and that ambiguity does not matter here).
    * :func:`solve_funnel_trajectory` is linear and solves the six axes with one shared
      matrix, so ``T_k = r x F_k`` at the inputs implies it at **every** output step. The
      funnel therefore preserves single-contact consistency exactly, and the only
      inconsistency in a generated chunk is the demo's own plus the bounded cone departure.

    Falls back to perturbing the demo's own ``T_0`` when there is no usable fit -- a degenerate
    or absurd ``|r_fit|`` (:attr:`FunnelConfig.max_moment_arm_m`), or a window with no
    measurable torque at all, where a cone about a zero vector is undefined.
    """
    cfg = cfg or FunnelConfig()
    demo = np.asarray(demo, dtype=np.float64)
    f0, t0 = demo[0, FORCE_SLICE], demo[0, TORQUE_SLICE]

    radius_f = float(np.clip(cfg.force_radius_scale * _radius_basis(demo, cfg.target_mode),
                             cfg.force_radius_min_n, cfg.force_radius_max_n))
    f_start = f0 + _annulus_radius(rng, radius_f, cfg.force_radius_min_frac) * _unit_vectors(rng)[0]

    if moment_arm is None:
        moment_arm = wrench_consistency(demo)[1]
    arm = np.asarray(moment_arm, dtype=np.float64).reshape(3)
    arm_len = float(np.linalg.norm(arm))
    log_k = math.log(cfg.torque_mag_scale_max)
    scale = math.exp(rng.uniform(-log_k, log_k))

    if 1e-6 < arm_len <= cfg.max_moment_arm_m:
        base = np.cross(arm, f_start)
        base_mag = float(np.linalg.norm(base))
        if base_mag > 1e-9:
            t_start = (base_mag * scale) * sample_in_cone(base, math.radians(cfg.torque_cone_deg), rng)
            return np.concatenate([f_start, t_start])

    t0_mag = float(np.linalg.norm(t0))
    if t0_mag < cfg.torque_radius_min_nm:
        # The chunk's torque EXCURSION regardless of target_mode: this branch is only reached
        # when |t0| is already below the floor, so |t0| carries no scale to multiply and the
        # excursion is the only magnitude the window offers.
        excursion_t = float(np.linalg.norm(demo[-1, TORQUE_SLICE] - t0))
        radius_t = max(cfg.force_radius_scale * excursion_t, cfg.torque_radius_min_nm)
        t_start = t0 + _annulus_radius(rng, radius_t, cfg.force_radius_min_frac) * _unit_vectors(rng)[0]
    else:
        t_start = (t0_mag * scale) * sample_in_cone(t0, math.radians(cfg.torque_cone_deg), rng)
    return np.concatenate([f_start, t_start])


# ---------------------------------------------------------------------------
# Admissibility: the analogue of 1001-Demos' collision constraint
# ---------------------------------------------------------------------------
def wrench_consistency(chunk: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit one contact point ``r`` to ``T = r x F`` over a window; return ``(residual_frac, r)``.

    ``residual_frac`` is ``|T_fit - T| / |T|`` over the whole window, so it is scale-free and
    directly comparable to the numbers in ``docs/force_ood_design.md`` §1.5 / §2.5. A small
    value means the window really is a single-point contact, which is the only case where
    rotating or rescaling a wrench corresponds to a physical situation -- hence its use as an
    admissibility gate rather than only as a data-quality filter. Measured on this dataset's
    contact windows: p50 **0.305**, p90 **0.467**, implied ``|r|`` p50 **206 mm** against a
    tool 242 mm from the sensor.

    Returns ``(0.0, zeros)`` for a window with no measurable torque -- nothing to explain, so
    nothing to reject.
    """
    chunk = np.asarray(chunk, dtype=np.float64)
    f, t = chunk[:, FORCE_SLICE], chunk[:, TORQUE_SLICE]
    # T = r x F is linear in r: stack the 3 rows of -[F]x per sample.
    n = f.shape[0]
    a = np.zeros((3 * n, 3), dtype=np.float64)
    a[0::3, 1] = f[:, 2]
    a[0::3, 2] = -f[:, 1]
    a[1::3, 0] = -f[:, 2]
    a[1::3, 2] = f[:, 0]
    a[2::3, 0] = f[:, 1]
    a[2::3, 1] = -f[:, 0]
    b = t.reshape(-1)
    denom = float(np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0, np.zeros(3)
    r, *_ = np.linalg.lstsq(a, b, rcond=None)
    return float(np.linalg.norm(a @ r - b) / denom), r


def moment_arm_for_wrench(wrench: np.ndarray) -> np.ndarray:
    """The minimum-norm contact point that reproduces ONE wrench: ``r = (F x T) / |F|^2``.

    ``r x F`` recovers the component of ``T`` perpendicular to ``F`` exactly; the parallel
    component is irreducible, because no lever can produce a torque along its own force.

    Needed because :func:`wrench_consistency` fits ``r`` over a whole window, which
    best-explains the *average* wrench and can be a poor fit to any single step. Under
    ``target_mode="anchor"`` the reference IS a single step (``d[0]``), so the arm used to build
    the start torque has to be that step's own arm — otherwise the reference does not lie on the
    ``T = r x F`` surface the funnel is being kept on, and the admissibility gate rejects the
    result. Measured: using the window fit instead pushed rejection from 12.9% to **18.1%** and
    left one slot unfilled.
    """
    wrench = np.asarray(wrench, dtype=np.float64).reshape(-1)
    f, t = wrench[FORCE_SLICE], wrench[TORQUE_SLICE]
    f2 = float(f @ f)
    if f2 < 1e-12:
        return np.zeros(3)
    return np.cross(f, t) / f2


def check_admissible(chunk: np.ndarray, *, dt: float, reference_residual: float,
                     cfg: FunnelConfig | None = None) -> str | None:
    """``None`` if ``chunk`` is physically admissible, else a short reason string.

    Reason strings are the keys of :attr:`RecoveryBatch.reject_reasons`, so they are terse and
    stable enough to aggregate over a whole dataset.
    """
    cfg = cfg or FunnelConfig()
    f = chunk[:, FORCE_SLICE]
    t = chunk[:, TORQUE_SLICE]
    if float(np.linalg.norm(f, axis=1).max()) > cfg.max_force_n:
        return "force_over_max"
    if float(np.linalg.norm(t, axis=1).max()) > cfg.max_torque_nm:
        return "torque_over_max"
    if chunk.shape[0] > 1 and dt > 0.0:
        if float(np.linalg.norm(np.diff(f, axis=0), axis=1).max()) / dt > cfg.max_dforce_n_per_s:
            return "dforce_over_max"
        if float(np.linalg.norm(np.diff(t, axis=0), axis=1).max()) / dt > cfg.max_dtorque_nm_per_s:
            return "dtorque_over_max"
    budget = max(cfg.consistency_slack * reference_residual, cfg.consistency_floor)
    if wrench_consistency(chunk)[0] > budget:
        return "inconsistent_wrench"
    return None


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------
def generate_recovery_chunks(demo: np.ndarray, *, dt: float, cfg: FunnelConfig | None = None,
                             rng: np.random.Generator | int | None = None) -> RecoveryBatch:
    """Generate :attr:`FunnelConfig.n_aug` funnel trajectories that recover onto ``demo``.

    ``demo`` is one stored ``hl_<role>_action_force`` chunk, ``(H, 6)``, in the sensor frame;
    ``dt`` is the grid step (``1/30 s`` under the v3 contract) and is used only by the slew
    checks. Each trajectory starts at its own sampled OOD wrench and ends bit-exact on
    ``demo[-1]``; rejected draws are resampled up to
    :attr:`FunnelConfig.max_tries_per_sample` times and then counted, never hidden.
    """
    cfg = cfg or FunnelConfig()
    demo = np.asarray(demo, dtype=np.float64)
    if demo.ndim != 2 or demo.shape[1] != WRENCH_DIM:
        raise ValueError(f"demo must be (H,{WRENCH_DIM}), got {demo.shape}")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    # Fit the contact point ONCE per window: it is a property of the demo, so refitting it per
    # draw would be the same lstsq 8-160 times over for an identical answer.
    demo_residual, window_arm = wrench_consistency(demo)
    reference = funnel_reference(demo, cfg.target_mode)
    # The gate's budget comes from what the funnel CONVERGES ONTO. Identical to demo_residual in
    # "demo" mode; in "anchor" mode it is d[0]'s own residual, which is the only fair standard
    # for a trajectory that spends most of its length holding d[0].
    reference_residual = (demo_residual if cfg.target_mode == "demo"
                          else wrench_consistency(reference)[0])
    # The arm the START torque is built from must be the arm of whatever the funnel converges
    # onto: the window fit for a whole-chunk reference, the target step's own arm for a
    # single-point one (see moment_arm_for_wrench).
    arm = window_arm if cfg.target_mode == "demo" else moment_arm_for_wrench(reference[-1])
    chunks: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    rejects: dict[str, int] = {}
    n_rejected = 0

    for _ in range(cfg.n_aug):
        for _try in range(cfg.max_tries_per_sample):
            start = sample_start(demo, generator, cfg=cfg, moment_arm=arm)
            chunk = solve_funnel_trajectory(demo, start, cfg=cfg)
            if cfg.project_torque:
                chunk = chunk.copy()
                chunk[:, TORQUE_SLICE] = np.cross(np.broadcast_to(arm, (chunk.shape[0], 3)),
                                                  chunk[:, FORCE_SLICE])
            reason = check_admissible(chunk, dt=dt, reference_residual=reference_residual, cfg=cfg)
            if reason is None:
                chunks.append(chunk)
                starts.append(chunk[0].copy())
                break
            rejects[reason] = rejects.get(reason, 0) + 1
            n_rejected += 1

    if chunks:
        stacked = np.stack(chunks)
        merges = np.array([merge_step(c, reference) for c in chunks], dtype=np.int64)
        start_arr = np.stack(starts)
    else:
        stacked = np.zeros((0, demo.shape[0], WRENCH_DIM))
        merges = np.zeros(0, dtype=np.int64)
        start_arr = np.zeros((0, WRENCH_DIM))
    return RecoveryBatch(chunks=stacked, starts=start_arr, demo=demo, reference=reference,
                         merge_steps=merges, n_rejected=n_rejected, reject_reasons=rejects)


def observation_for(chunk: np.ndarray, force_hist_len: int) -> np.ndarray:
    """The ``(L, 6)`` force observation to pair with a generated ``chunk``.

    Under the v3 contract ``L == 1`` and the observation bin IS the wrench at ``t0`` -- and
    that is not an approximation: on ``session_2026.08.23_merged_v3``,
    ``hl_left_force_history[t, 0]`` equals ``hl_left_action_force[t, 0]`` to **0.0 on every
    axis of every sampled anchor**. So the observation that belongs to a trajectory starting
    at ``chunk[0]`` is ``chunk[0]``, exactly as the user's own framing has it ("the force
    history is just the force at the moment the new trajectory starts").

    For ``L > 1`` the earlier bins are unknown -- they would be the history of a contact that
    never happened -- so they are filled by holding ``chunk[0]``. That is a **flat** history
    where the real one would have had slope, and a model with a force-history encoder can
    read that flatness as a cue. Stated here rather than buried: at ``L > 1`` this
    augmentation is weaker than at ``L == 1``, which is the contract's default.
    """
    if force_hist_len < 1:
        raise ValueError(f"force_hist_len must be >= 1, got {force_hist_len}")
    return np.repeat(np.asarray(chunk, dtype=np.float64)[:1], force_hist_len, axis=0)


# ---------------------------------------------------------------------------
# Window selection: WHERE in an episode this augmentation is well-posed
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class WindowGate:
    """Which anchors are worth augmenting. Yields on ``session_2026.08.23_merged_v3`` are in
    each field's note, measured over 3566 usable sampled windows.

    The contact requirement is the deliberate inversion of 1001-Demos' contact handling: it
    augments only the PRE-contact phase and preserves contact untouched; we do the opposite,
    because the wrench channel is only interesting where there is a wrench.
    """

    #: Peak ``|F|`` the window must reach. 1.5 N is the deploy force gate, so below it the
    #: low level would not act on the force at all. Keeps 81.5%.
    contact_min_n: float = 1.5
    #: ``|F|`` **at the anchor frame t0 itself** — "contact starts where the force exceeds this".
    #:
    #: This is the gate that was missing, and its absence was a real defect rather than a
    #: tuning choice. :attr:`contact_min_n` only asks that the window reach contact SOMEWHERE,
    #: so **7.6%** of accepted anchors had ``|F(t0)| < 1 N`` — the hand was still in free space
    #: at t0. Augmenting there generates a recovery from an out-of-distribution wrench to
    #: *approximately no force*, which is not a contact behaviour at all, and in the viewer it
    #: draws a degenerate bundle collapsing onto the gripper.
    anchor_min_force_n: float = 1.0
    #: ``|F|`` at the chunk's LAST step — the funnel's throat, i.e. the recovery target. Same
    #: argument from the other end: a target of ~0 N is not a contact force to recover to.
    #: Only 0.9% of anchors fail this alone; together with the anchor gate, 91.6% of what the
    #: old gate accepted survives, so the cost is the degenerate cases and nothing else.
    target_min_force_n: float = 1.0
    #: Minimum ``|F_{H-1} - F_0|``. Without it, a window whose force barely moves gives a
    #: funnel with nothing to converge ONTO that differs from where it started. Keeps 81.3%
    #: at 1.0 N together with the contact gate.
    ramp_min_n: float = 1.0
    #: Require every step of the chunk to be a supervised force target. 88.9% of anchors
    #: qualify; the rest are episode tails, where the pose target does not exist either.
    require_all_valid: bool = True
    #: Anchor stride. The chunk is 50 steps, so consecutive anchors are ~98% the same window;
    #: 15 (0.5 s) keeps the bank from being dominated by near-duplicates.
    stride: int = 15


def select_windows(action_force_valid: np.ndarray, force_hist_valid: np.ndarray,
                   peak_force_n: np.ndarray, ramp_n: np.ndarray, *,
                   anchor_force_n: np.ndarray | None = None,
                   target_force_n: np.ndarray | None = None,
                   gate: WindowGate | None = None) -> np.ndarray:
    """Indices (into the arrays given) of anchors that pass ``gate``.

    Takes already-reduced per-anchor quantities rather than the zarr itself, so this stays
    numpy-only and directly unit-testable; ``scripts/augment_force_recovery.py`` is what knows
    how to compute them from a dataset.

    ``anchor_force_n`` (``|F(t0)|``) and ``target_force_n`` (``|F(t0+H-1)|``) are optional only
    so an older caller keeps working; **pass them.** Without them the contact-at-the-endpoints
    gates cannot be applied and the degenerate windows described on
    :attr:`WindowGate.anchor_min_force_n` come back.
    """
    gate = gate or WindowGate()
    valid = np.asarray(action_force_valid)
    ok = (valid.all(axis=1) if gate.require_all_valid else valid.any(axis=1))
    ok &= np.asarray(force_hist_valid).reshape(-1) > 0.5
    ok &= np.asarray(peak_force_n).reshape(-1) >= gate.contact_min_n
    ok &= np.asarray(ramp_n).reshape(-1) >= gate.ramp_min_n
    if anchor_force_n is not None:
        ok &= np.asarray(anchor_force_n).reshape(-1) >= gate.anchor_min_force_n
    if target_force_n is not None:
        ok &= np.asarray(target_force_n).reshape(-1) >= gate.target_min_force_n
    idx = np.nonzero(ok)[0]
    return idx[::gate.stride] if gate.stride > 1 else idx


__all__ = [
    "FORCE_SLICE",
    "MIN_HORIZON",
    "TORQUE_SLICE",
    "WRENCH_DIM",
    "TARGET_MODES",
    "FunnelConfig",
    "RecoveryBatch",
    "WindowGate",
    "check_admissible",
    "funnel_reference",
    "funnel_weights",
    "generate_recovery_chunks",
    "linear_reference",
    "merge_step",
    "observation_for",
    "sample_in_cone",
    "sample_start",
    "select_windows",
    "solve_funnel_trajectory",
    "wrench_consistency",
]
