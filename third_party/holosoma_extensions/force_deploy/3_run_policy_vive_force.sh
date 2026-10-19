#!/usr/bin/env bash
# TERMINAL 3 — offboard run_policy on the REAL G1: VIVE (pos+orn) + live F/T + servo.
#
# The FULL combined mode: two VIVE trackers drive the hands, the live Bota fills
# force_meas_local, and (SERVO=1) the Jᵀ servo + the live F^cmd keys let you PUSH.
#
# ┌─ STAGING (do NOT skip on hardware) ───────────────────────────────────────┐
# │ Run 1:  SERVO=0            — validate VIVE tracking + F/T sign/tare.        │
# │ Run 2:  SERVO=1  (this)    — servo pushes; ramp F^cmd in slowly with keys.  │
# └────────────────────────────────────────────────────────────────────────────┘
#
# ⚠️ With SERVO=1, F^cmd is NOT contact-gated: pressing 'u' (F^cmd +x) injects
#    Δq = k·Jᵀ·F^cmd into the arms IMMEDIATELY, even in FREE SPACE — the arms
#    push forward whether or not they're touching anything. Start every hand at
#    0 N (they are), press keys ONE AT A TIME, watch the arm. Gantry + a
#    dedicated hardware-E-stop operator are mandatory. There is NO soft abort.
#
# GATE_CMD=1 (F_meas-gate-on-F^cmd): a hand with no F^cmd feeds ZERO F/T reading to
#    the policy (obs + servo), so it is a pure POSITION controller and cannot chase
#    sensor noise. The measured force re-enters ONLY on the commanded hand(s)
#    (u/j/h/k/y/n opens the gate; c zeros the selected hand and p zeros both; l only
#    switches selection). Direct-VIVE MODE=3pt-unified-anneal defaults ON for this
#    pure-S0 bring-up; absolute/S1 input and every other mode default OFF.
#
# ┌─ THE THREE FORCE THRESHOLDS (one number each, all in Newtons) ──────────────┐
# │ 1.5 N  FORCE_CMD_WIRE_GATE_N — and, with the wire on, GATE_CMD_N with it.   │
# │        BELOW it there is no force at all: no F^cmd reaches the buffer AND    │
# │        (GATE_CMD=1) no F_meas reaches the obs/servo, however hard the Bota   │
# │        is pressed. That is pure position control, and it is the default      │
# │        state. Mirrors the planner's own gate (hl_serving --force-gate-n).    │
# │  40 N  FORCE_ESTOP_N — HARD STOP. Runs the joystick-'B' handler: policy off, │
# │        stiff hold at the current pose, teleop/planner disarmed, F^cmd zeroed.│
# │        Needs a re-home (Y) + START (A) to come back. Checked on the          │
# │        COMPENSATED + tared per-hand ‖F‖ that T2 publishes (the tool's own    │
# │        weight is already removed, so ~0 N in free space) — NOT the raw Bota  │
# │        wrench — and BEFORE the deploy-side F_meas EMA / GATE_CMD, so neither │
# │        can hide it. Whichever hands FORCE_HANDS puts in the loop count.      │
# │        (Was 20 N to 2026-08-14: below the 25 N F^cmd clamp and only 1.5 N    │
# │        over the 18.5 N hardest press in the 0814 demos, so it false-tripped.)│
# │        ⚠️ It stops the POLICY, it does not de-energize the robot — the       │
# │        hardware E-STOP is still the only true abort.                         │
# │  55 N  HOLOSOMA_FORCE_RUNAWAY_N — warn + freeze the optional Jᵀ servo for    │
# │        the tick. Not an abort; observe-only presets only get the warning.    │
# │        DERIVED as FORCE_ESTOP_N + 15, never typed, so it stays ABOVE the     │
# │        stop, and the stop stays the SINGLE high-force behaviour (the valve is│
# │        reachable only with the stop off, FORCE_ESTOP_N=0 → back to 35).      │
# └────────────────────────────────────────────────────────────────────────────┘
#
# LIVE F^cmd KEYS (ENABLED; the unified default is NOT force-qualified — warns, no lock):
#   on the SELECTED hand (default LEFT),
#   u/j = +x/-x   h/k = +y/-y   y/n = +z/-z   |   l = switch hand   c = zero hand   p = zero both
# Each press = 2 N toward a target (clamp 25 N), slew-limited to 2 N/tick.
# 'l' toggles L/R (starts on LEFT); F/T tare = joystick R2 / keyboard T (IN FREE SPACE).
#
# ┌─ C1 UNIFIED modes (MODE=3pt-unified / 3pt-unified-anneal) ─────────────────┐
# │ These checkpoints are SINGLE-POLICY force controllers: the Jᵀ servo was    │
# │ demoted in TRAINING to a pure reference generator (observe-only, or its    │
# │ authority annealed 1→0), so the POLICY is the only executor.               │
# │  * SERVO is IGNORED — no Δq is ever injected in EITHER unified mode.       │
# │  * ⚠️ THAT DOES NOT MAKE IT SAFE. The POLICY itself pushes: x_tgt_shift =   │
# │    (F^cmd + I)/K is nonzero the instant F^cmd is nonzero — even in FREE    │
# │    SPACE (5 N ⇒ 2.5 cm of setpoint shift) — and the policy is trained to   │
# │    chase it. The current default wda3h4ll/model_19000 is NOT force-        │
# │    qualified either (servo_alpha=1 for its WHOLE history — the anneal      │
# │    never fired — so it trained WITH servo help it does not get here and    │
# │    may UNDER-reach). Its F^cmd keys are ENABLED with a loud warning; start │
# │    Run 1 at F^cmd=0, then ramp one 2 N nudge at a time.                    │
# │  * ⚠️ TCP-UNIFIED GEOMETRY (2026-08-03 retrain, run wda3h4ll). BOTH hand    │
# │    TRACKING points moved from the 0.18 m VR-tracker lever to the 0.242 m   │
# │    fingertip TCP — the same anchor the force stack always used — so        │
# │    position and force finally supervise ONE point (they were 6.7 cm apart  │
# │    per hand). The obs slot is `vr_3point_pos_tcp`; width is UNCHANGED at   │
# │    170D/680D, so NO dimension check can catch a wrong-geometry checkpoint. │
# │    Only the per-term NAME distinguishes them, which is why every pre-      │
# │    2026-08-03 unified export (iss0qs71/*, m3s0pw9i/*) is now REFUSED —     │
# │    by a pre-flight check here (before DDS init) and by the runtime.        │
# │  * ⚠️ wda3h4ll COLLAPSED AT STEP 19498. model_19000 is the LAST GOOD        │
# │    export and is the default; model_19500+ are REFUSED below. Episode      │
# │    length went 499.8 → 26 and every reward fell ~100x in ONE step, then    │
# │    the exports diverged to NaN (376 NaNs per normalizer + all-NaN actions, │
# │    measured on 19500/20000/22500) — the same failure as the retired        │
# │    iss0qs71 window. Do NOT "just take the newest checkpoint" here.         │
# │  * The integrator STILL winds: integrator_state (I) / x_tgt_shift (Δx) are │
# │    live, load-bearing actor inputs — so there is no "-layerA" (sensing-    │
# │    only) variant: that name would misdescribe a live integrator.           │
# │  * NEW obs channel: the 2D soft `contact_phase` φ∈[0,1] per hand, computed │
# │    live from the Bota ‖F_meas‖ (hysteresis 2.0/1.5 N + 50 ms LPF). Tunable │
# │    via HOLOSOMA_CONTACT_PHASE_{THETA_ON,THETA_OFF,TAU_S,WARMUP_STEPS}.     │
# │    φ is forced 0 for the first 5 ticks AND until the START auto-tare burst │
# │    finishes (TARE_S), so a pre-tare bias cannot latch a phantom "in        │
# │    contact". ⚠️ TARE_N=0 removes that second window — the runtime warns.    │
# │  * So F/T SIGN + TARE are now MORE load-bearing than before: a wrong sign  │
# │    or a contact-baked tare corrupts BOTH force_meas_local AND φ, and the   │
# │    unified policy's whole contact behaviour is φ-gated. Validate with      │
# │    4_ft_dynamic_compare.sh first, and keep the grippers in free space for  │
# │    the whole START window.                                                 │
# │  * GATE_CMD=1 deliberately makes an uncommanded hand blind to contact:      │
# │    force_meas_local is zero and φ decays to zero. This is the requested     │
# │    pure-position startup behavior for 3pt-unified-anneal, whose default is  │
# │    therefore ON below. The default ckpt's F^cmd keys are ENABLED (warn-only)│
# │    so keep its F^cmd at 0 for Run 1. Open the gate only in FREE SPACE;      │
# │    closing a 25 N command can take up to 260 ms while it slews to zero. Use │
# │    GATE_CMD=0 for full sensor/contact fidelity. Runaway detection still sees │
# │    raw force, but it only freezes the optional servo; it is not a soft abort.│
# └────────────────────────────────────────────────────────────────────────────┘
#
# AUTO-TARE ON START: pressing A (START) auto-fires a BURST of F/T re-tares
# (HOLOSOMA_FORCE_TARE_ON_START × over …_SECONDS, default 5×/2s) so the LAST tare
# zeroes the sensor AFTER the ready-pose swing settles — the policy starts from ~0
# force instead of crashing on a nonzero startup reading. ⚠️ Grippers MUST be in
# FREE SPACE at START (a contact force at tare bakes in as a constant bias).
#
# NOTE: the F/T feed is now the RNEA-compensated + 12 Hz anti-aliased force from
# 2_ft_bridge.sh (FT_ALGO=rnea) — the inertial phantom is removed at the source and
# the high-frequency residual is filtered before the 50 Hz sample, so the policy-side
# FEMA below is now a SECONDARY smoother (start at 1.0 = off; only lower it if you
# still see chatter). It does NOT anti-alias (it runs at 50 Hz, after the sample).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # holosoma_extensions
PY="${PY:-$HOME/.holosoma_deps/miniconda3/envs/hssim/bin/python}"

# Keep the validated preset/checkpoint authoritative. The fallback CLI parser takes
# the last duplicate, so a trailing override would otherwise bypass the checks below.
for _ARG in "$@"; do
  case "$_ARG" in
    inference:*|--task.model-path|--task.model-path=*|--task.model_path|--task.model_path=*)
      echo "[3_run_policy_vive_force] do not override preset/model via trailing args; use MODE=... and ONNX=..."
      exit 2
      ;;
  esac
done

# --- Flip these between runs -------------------------------------------------
MODE="${MODE:-2pt}"          # 2pt = exp:g1-wbt-2pt-hybrid (2 hand trackers; DEFAULT,
                             #   byte-identical to before). 3pt = exp:g1-wbt-3pt-hybrid
                             #   (+ a HEAD tracker; Terminal-1 bridge MUST be MODE=3).
                             #   3pt-global-vel = exp:g1-wbt-3pt-hybrid-global-vel
                             #   (walk-enabled 3pt hybrid, 183D obs: + root pos/vel
                             #   trio, 6D tracker orn; SAME 3-tracker bridge MODE=3;
                             #   needs fk-odometry for base_lin_vel — already on below).
                             #   3pt-unified / 3pt-unified-anneal = the C1 UNIFIED
                             #   single-policy twins exp:g1-wbt-3pt-hybrid-unified[-anneal]
                             #   (170D = pelvis-local 168D + the 2D soft contact_phase;
                             #   root-invariant, does NOT walk; SAME 3-tracker bridge
                             #   MODE=3). ⚠️ On these the Jᵀ servo is NEVER injected —
                             #   SERVO below is IGNORED (see the note above). ⚠️ Both now
                             #   consume the TCP-UNIFIED obs (hands tracked at 0.242 m,
                             #   term vr_3point_pos_tcp), so ONLY a checkpoint retrained
                             #   on that geometry loads: -anneal defaults to the wda3h4ll
                             #   retrain, and 3pt-unified has NO default (no R2 TCP run
                             #   exists yet — pass ONNX= explicitly).
                             #   5pt-unified-anneal = exp:g1-wbt-5pt-hybrid-unified-anneal
                             #   (736D: the 170D unified obs with the tracking front-end
                             #   widened to FIVE keypoints — hands + head + BOTH FEET).
                             #   dense-unified-anneal = exp:g1-wbt-hybrid-unified-anneal
                             #   (732D: NO keypoints, all 29 joint targets + velocities).
                             #   ⚠️ Both are MOTION-COMPLETION modes: the rig still has only
                             #   three trackers, so Terminal 1 MUST be the matching bridge
                             #   mode (MODE=5 / MODE=dense), which runs the completion and
                             #   streams the wider command. A 3-tracker bridge against these
                             #   presets is width-rejected → the robot holds the ready pose.
                             #   ⚠️ ~667 ms of command lag is STRUCTURAL here (window/stride
                             #   arithmetic, not a slow host) — see the banner at launch.
                             #   FOOT_IK=1 enables the completion's leg IK; it is OFF by
                             #   default because it measures 291% duty and 6.1 s of command
                             #   lag (vs 0.7 s) — not real-time, and not tunable.
                             #   wbt = exp:g1-wbt, the BASE whole-body tracker (628D, run
                             #   ruo0o6fr). ALSO a motion-completion mode and it rides the
                             #   SAME Terminal-1 bridge as dense (MODE=dense, or MODE=wbt
                             #   which the bridge accepts as that spelling): both steer the
                             #   58D `command` term, so the wire is identical.
                             #   ⚠️ NO FORCE BLOCK in this obs. Every force knob below —
                             #   SERVO, FEMA, TARE_*, F^cmd, GATE_CMD — is INERT: there is
                             #   no slot for any of it. The F/T bridge (Terminal 2) is
                             #   optional here and only feeds the 20 N auto-stop.
                             #   ⚠️ NO ROOT CHANNEL either: this preset has no
                             #   motion_ref_pos_b, so heading and world position stay pinned
                             #   to the injected clip and your trackers steer POSTURE only.
                             #   Measured on replay: 3.5 cm torso-frame MPJPE vs 12 cm of
                             #   root error — the policy's own limit, not the teleop's.
SERVO="${SERVO:-0}"          # 0 = Layer-A (F/T obs only, no push); 1 = Jᵀ servo ON.
                             #   ⚠️ IGNORED on MODE=3pt-unified[-anneal]: those policies were
                             #   trained with the servo demoted to a pure REFERENCE GENERATOR
                             #   (observe-only / α annealed to 0), so the runtime keeps the
                             #   integrator winding (I/Δx are live actor inputs) but NEVER
                             #   injects Δq. The policy itself is the force controller.
FEMA="${FEMA:-1.0}"          # policy-side F_meas low-pass alpha: 1.0 = OFF (raw, now
                             #   the default since the bridge anti-aliases). 0.5-0.85
                             #   = extra smoothing at the cost of lag. Secondary to the
                             #   bridge's 12 Hz stage; NOT an anti-alias (runs post-sample).
TARE_N="${TARE_N:-5}"        # auto-tare BURST count on START (0 = off; with a >0 window,
                             #   a count of 1 is floored to 2 so one tare lands AFTER settle)
TARE_S="${TARE_S:-2.0}"      # auto-tare burst window (seconds); last tare fires at its end
GATE_CMD_DEFAULT=0           # Resolved after MODE selection. 3pt-unified-anneal defaults ON for
                             #   direct-VIVE position-only S0 bring-up; absolute/S1 and all
                             #   established modes remain OFF.
                             # GATE_CMD_N's default is resolved below, AFTER FORCE_CMD_WIRE — it
                             #   differs by who authors F^cmd (keyboard nudges vs the HL planner).
                             # FORCE_HANDS's default is resolved below, after VIVE_RELATIVE — the
                             #   absolute/S1 planner is LEFT-only by construction, direct VIVE is not.
FORCE_ESTOP_N="${FORCE_ESTOP_N:-40}"  # HARD F/T STOP (N), 0 = off. A measured ‖F‖ above this on an
                             #   in-the-loop hand means an impact or a runaway, not the task — the
                             #   policy STOPS exactly as joystick B does (stiff hold at the current
                             #   pose, teleop/planner DISARMED, F^cmd zeroed), and needs a re-home +
                             #   START to come back. Distinct from HOLOSOMA_FORCE_RUNAWAY_N below,
                             #   which only warns and freezes the optional Jᵀ servo.
WINDUP_WARN_N="${WINDUP_WARN_N:-200}"  # ‖I‖ ALARM threshold (N) of the UNIFIED presets' observe-only
                             #   integrator (runtime default 20 = half of the 40 N/axis clip). It only
                             #   PRINTS: "servo integrator ‖I‖=20.0N > 20N … re-tare in FREE SPACE".
                             #   Under the HL planner a hand APPROACHING a surface with an F^cmd of a
                             #   few N winds I to 20 N within seconds — that is the unified policy
                             #   closing the last centimetres, not a drifted tare — so 2026-09-20 the
                             #   operator asked for 10x: 200 N cannot be reached (‖I‖ ≤ 40·√3 ≈ 69 N),
                             #   i.e. the alarm is OFF on the rig. 0 = off explicitly; WINDUP_WARN_N=20
                             #   restores the reviewed alarm. Nothing else moves: the clip, the runaway
                             #   valve and FORCE_ESTOP_N are untouched.
                             #   WHICH ‖F‖: the per-hand force T2 PUBLISHES — payload-compensated
                             #   (RNEA by default) + tared + 12 Hz anti-aliased, sensor frame. The
                             #   tool's own weight is already out, so free space reads ~0 N and 40 N
                             #   is 40 N of contact. It is NOT the raw Bota wrench (that ships under
                             #   the separate left_raw/right_raw keys, which the policy never reads),
                             #   and it is taken BEFORE the deploy-side F_meas EMA and GATE_CMD, so
                             #   neither of those can hide a spike. Suppressed for the START tare
                             #   window (TARE_S + 1.5 s settle) because a pre-tare bias is the same
                             #   order as the threshold. ⚠️ NOT a substitute for the hardware
                             #   E-STOP: it stops the POLICY, it does not de-energize the robot.
                             #   WHY 40 AND NOT 20 (raised 2026-08-15): 20 was set before any of
                             #   this ran on the rig, and it was under BOTH numbers that bound a
                             #   legitimate press. (a) The hardest press in the whole 283-episode
                             #   0814 demo set is 18.5 N (p99 = 15.3 N, over 986k samples of
                             #   robot0_ft_wrench_hires) — 20 left 1.5 N of headroom, so a wipe
                             #   only slightly firmer than the demos stops the robot; it did, at
                             #   25.3 N. (b) 20 also sat BELOW the 25 N keyboard F^cmd clamp, i.e.
                             #   a full-scale operator push tripped the stop with its own command.
                             #   40 N is 2.2x the demo maximum and above every commandable force,
                             #   while still far under the ~9.7 N·m of wrist torque that a 40 N tip
                             #   load puts on the 0.242 m TCP lever. Lower it back toward 25 for a
                             #   first contact run on a NEW task whose force scale is unknown.
RECORD="${RECORD:-0}"        # 0 = OFF (default, byte-identical): no recording, inference unchanged.
                             #   1 = ON: the policy opens a broadcast-only RECORD tap (ws :8779) that
                             #   ships F^cmd + pelvis F_meas/servo + 29-DOF pos/vel + a running(START)
                             #   flag each tick to the out-of-process recorder (6_record_teleoperation.sh).
                             #   A slow/absent recorder cannot stall inference (bounded, drop-oldest). To
                             #   record a session ALSO run T2 with FT_EMIT_RAW=1 + start 6_...sh (T5/T6).
CPU_THREADS="${CPU_THREADS:-1}"  # Thread cap for BOTH math pools inside the policy process:
                             #   onnxruntime's intra-op pool (the ONNX forward pass) and
                             #   OpenBLAS/OpenMP behind numpy+pinocchio (obs math, FK,
                             #   Jacobian). DEFAULT 1 — see below for why that is not a
                             #   downgrade.
                             #
                             #   ⚠️ There is NO "C++ vs Python inference" switch to flip: the
                             #   forward pass ALREADY runs in C++ (onnxruntime native kernels),
                             #   FK/Jacobian in C++ (pinocchio/Eigen) and DDS in C++
                             #   (unitree_interface pybind11 + CycloneDDS .so). Python is only
                             #   the 50 Hz glue. The CPU was going to BUSY-WAIT, not to Python:
                             #   both libraries default their pool size to the CORE COUNT and
                             #   then SPIN between calls, so on the 24-core deploy box each pool
                             #   burned ~22 cores waiting out a 20 ms tick that needs <3 ms of
                             #   work. MEASURED here (wda3h4ll/model_19000, 50 Hz):
                             #     onnxruntime  default → 22.7 cores, p99 7.6 ms
                             #                  intra=1 →  0.12 cores, p99 3.1 ms
                             #     numpy/BLAS   default → 21.7 cores, p99 1.94 ms
                             #                  =1      →  0.03 cores, p99 0.034 ms
                             #   So capping is faster on the TAIL as well as ~200x cheaper: the
                             #   spinning pools were contending with each other and with the
                             #   VIVE/F-T/record threads. ONNX actions are BITWISE IDENTICAL
                             #   across every thread/spin combination (verified on random obs),
                             #   so this cannot change what the robot does.
                             #
                             #   Set HERE rather than in the library default on purpose: a
                             #   tight batch loop WANTS the big spinning pool (3647 it/s vs
                             #   923 at intra=1), so headless eval and every other in-process
                             #   caller keep onnxruntime's own defaults. Only this launcher has
                             #   a 20 ms tick as its binding constraint.
                             #
                             #   CPU_THREADS=0 = UNCAPPED: restores the pre-2026-08-11 defaults
                             #   (pool = core count, spinning ON) for an A/B measurement only.
                             #   Raise to 2-4 only if a much larger future policy stops fitting
                             #   the 20 ms tick; 1 leaves 17 ms of headroom today.
if [[ ! "$CPU_THREADS" =~ ^[0-9]+$ ]]; then
  echo "[3_run_policy_vive_force] bad CPU_THREADS='$CPU_THREADS' (want a non-negative integer; 0 = uncapped)"
  exit 2
fi
if [ "$CPU_THREADS" = "0" ]; then
  # Uncapped: let each library size its own pool from the core count (what it did before
  # this knob existed) and re-enable onnxruntime's busy-wait so the A/B is honest.
  _ORT_INTRA=0
  _ORT_SPIN="${ORT_SPIN:-1}"
  _BLAS_THREADS="$(nproc)"
  echo "[3_run_policy_vive_force] ⚠️  CPU_THREADS=0 — math pools UNCAPPED + onnxruntime spinning ON."
  echo "[3_run_policy_vive_force]     This is the old default and it saturated the box (~22 cores per"
  echo "[3_run_policy_vive_force]     pool at 50 Hz) with a WORSE p99. For measurement only."
else
  _ORT_INTRA="$CPU_THREADS"
  _ORT_SPIN="${ORT_SPIN:-0}"
  _BLAS_THREADS="$CPU_THREADS"
fi
# VIVE mapping mode. 1 = direct-VIVE RELATIVE/clutch (pure S0); 0 = ABSOLUTE
# planner targets (S1). Resolve this BEFORE the MODE case because the gate default
# must not leak from the direct-S0 bring-up into an existing S1 launch.
VIVE_RELATIVE_RAW="${HOLOSOMA_VIVE_RELATIVE:-1}"
case "${VIVE_RELATIVE_RAW,,}" in
  1|true|yes|on) VIVE_RELATIVE=1 ;;
  0|false|no|off) VIVE_RELATIVE=0 ;;
  *) echo "[3_run_policy_vive_force] bad HOLOSOMA_VIVE_RELATIVE='$VIVE_RELATIVE_RAW' (want 1|0|true|false|yes|no|on|off)"; exit 2 ;;
esac
# Do the per-tick slew caps (HOLOSOMA_VIVE_MAX_POS_STEP / _ORN_STEP, pinned below) cover the 5pt
# FEET? Default 0 = NO: the two foot slots pass through un-slewed and the caps cover only the
# three upper points (hands + torso). The caps turn a far hand target into a glide; a GAIT is a
# far foot target by construction — FUJI's walk skill (scripts/hl_serving_5pt.py, the walk-with-box
# feet) demands 19-46 mm/tick against the 5 mm cap, so a slewed foot does not walk slowly, it
# crawls toward a target that keeps running away. SLEW_FEET=1 restores the whole-vector pool
# (every preset before 2026-09-07). Only the 5pt family has feet; other MODEs ignore it.
SLEW_FEET_RAW="${SLEW_FEET:-0}"
case "${SLEW_FEET_RAW,,}" in
  1|true|yes|on) SLEW_FEET=1 ;;
  0|false|no|off) SLEW_FEET=0 ;;
  *) echo "[3_run_policy_vive_force] bad SLEW_FEET='$SLEW_FEET_RAW' (want 1|0|true|false|yes|no|on|off)"; exit 2 ;;
esac
# Does a 5pt wire frame's OPTIONAL `ref_joints` (29 reference joint targets + velocities BY NAME —
# FUJI's walk skill sends the gait clip's own legs while it plays) override the policy's
# command_lower_body obs term? Default 1 = yes. That 24-D term is the reference motion's leg joints
# and on the live path it is read from the FROZEN idle clip — so under a walk the legs were told
# "stand" while the two foot keypoints said "swing", and the robot stepped in place. With the clip's
# legs on the wire the policy sees exactly what a baked walk clip feeds it in mjswan (where it walks).
# 5pt only, armed only, never slewed; a frame without the field is byte-identical. REF_JOINTS=0 ignores
# the field (an A/B on the robot with FUJI untouched). Reported on RecordTap as `ref_joints_live`.
REF_JOINTS_RAW="${REF_JOINTS:-1}"
case "${REF_JOINTS_RAW,,}" in
  1|true|yes|on) REF_JOINTS=1 ;;
  0|false|no|off) REF_JOINTS=0 ;;
  *) echo "[3_run_policy_vive_force] bad REF_JOINTS='$REF_JOINTS_RAW' (want 1|0|true|false|yes|no|on|off)"; exit 2 ;;
esac
# Per-MODE preset + checkpoint + clip. 2pt keeps the exact old defaults (byte-identical
# resolved argv); 3pt points at the 3pt-hybrid preset + the n4exfyvp servo-on ckpt;
# 3pt-global-vel points at the walk-enabled 183D preset + the bqlzkwok servo-on ckpt
# (canonical run, still training — bump the model number as later ckpts land). Of these
# checkpoints only the 5pt one is released (see the README); the others are runs of the
# paper's development, so those MODEs need ONNX=<your checkpoint>. ONNX /
# CLIP env overrides still win in any mode. All ONNX are I/O-adaptive: either the
# browser (actor_obs→action) or the deploy (obs+time_step→actions) export works.
case "$MODE" in
  2pt)
    PRESET="g1-29dof-holosoma-2pt-hybrid-wbt"
    ONNX="${ONNX:-logs/WBT-Holosoma/31wei7by/model_29999.onnx}"   # final servo-on ckpt (run finished)
    ;;
  3pt)
    PRESET="g1-29dof-holosoma-3pt-hybrid-wbt"
    ONNX="${ONNX:-logs/WBT-Holosoma/n4exfyvp/model_06500.onnx}"   # 3pt-hybrid servo-on ckpt (run n4exfyvp)
    ;;
  3pt-global-vel)
    PRESET="g1-29dof-holosoma-3pt-hybrid-global-vel-wbt"
    ONNX="${ONNX:-logs/WBT-Holosoma/bqlzkwok/model_02500.onnx}"   # 3pt-hybrid-global-vel servo-on ckpt (run bqlzkwok)
    echo "[3_run_policy_vive_force] ⚠️  INTERIM checkpoint — run bqlzkwok is still training (model_02500)."
    ;;
  3pt-unified)
    PRESET="g1-29dof-holosoma-3pt-hybrid-unified-wbt"
    # NO DEFAULT — the R2 twin has no TCP-geometry checkpoint. Both unified presets now
    # consume the TCP-unified obs (`vr_3point_pos_tcp`, hands at 0.242 m), and the old R2
    # default m3s0pw9i/model_02000 was trained at the 0.18 m VR lever, so it is RETIRED:
    # loading it here is a 6.7 cm/hand train-deploy mismatch, refused by the pre-flight
    # below and again by the runtime. The only TCP retrain to date is the R1 ANNEAL run
    # wda3h4ll → use MODE=3pt-unified-anneal. If an R2 TCP run lands, pass it explicitly.
    if [ -z "${ONNX:-}" ]; then
      echo "[3_run_policy_vive_force] MODE=3pt-unified has NO default checkpoint: the R2 twin has not been"
      echo "[3_run_policy_vive_force]     retrained on the TCP-unified geometry. The old default"
      echo "[3_run_policy_vive_force]     (m3s0pw9i/model_02000) tracked the 0.18 m VR lever and is RETIRED —"
      echo "[3_run_policy_vive_force]     this preset now feeds the 0.242 m TCP obs (6.7 cm/hand apart)."
      echo "[3_run_policy_vive_force]     Use MODE=3pt-unified-anneal (run wda3h4ll), or pass ONNX=<an R2 TCP ckpt>."
      exit 2
    fi
    echo "[3_run_policy_vive_force] UNIFIED (R2): observe-only integrator — SERVO=$SERVO is IGNORED, no Δq is injected."
    echo "[3_run_policy_vive_force] ⚠️  The POLICY still pushes from F^cmd: start with every F^cmd at 0 N."
    echo "[3_run_policy_vive_force] ⚠️  Explicit ONNX=$ONNX — verify Env/force_alpha ≈ 1 on its wandb run before"
    echo "[3_run_policy_vive_force]     ramping F^cmd past ~alpha*25 N (the unified twins ramp force-alpha, so an"
    echo "[3_run_policy_vive_force]     early ckpt has never seen a full-magnitude F^cmd)."
    ;;
  3pt-unified-anneal)
    PRESET="g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt"
    # TCP-UNIFIED retrain (run wda3h4ll, 64gpu-ft). Supersedes iss0qs71 entirely: this
    # preset's obs01 slot is now `vr_3point_pos_tcp` (hands on the 0.242 m force TCP), and
    # iss0qs71 was trained at the 0.18 m VR lever — a 6.7 cm/hand mismatch, so every
    # iss0qs71 export is now REFUSED (pre-flight below + the runtime term check), which is
    # also why the old NaN-window guard for model_27500..34500 is moot but kept.
    #
    # ⚠️ wda3h4ll COLLAPSED at step 19498 — model_19000 is the LAST GOOD export, which is why
    # the default does NOT track the newest checkpoint. From W&B history: every reward drops
    # ~100x across a single step (tcp_position 3.355@19497 → 0.983@19499 → 0.031@19503; same
    # cliff in tcp_orientation, force_track_pos_eq2, force_error_direct) and
    # Env/average_episode_length falls 499.8 → 212.6 → 26, i.e. the policy stopped standing.
    # force_alpha/servo_alpha both stayed 1 across the cliff, so the curriculum did NOT cause
    # it. The exports then go NaN exactly as iss0qs71 did: measured 376 NaNs in BOTH
    # normalizer_mean and normalizer_std, and all-NaN actions, on model_19500 / 20000 / 22500.
    # Verified GOOD on model_19000 (and 13500): 680D obs, TCP obs slot, 0 NaN, finite actions.
    # So this is the same divergence-then-NaN failure mode, and the run is DEAD past 19000.
    ONNX="${ONNX:-logs/WBT-Holosoma/wda3h4ll/model_19000.onnx}"
    if [ "$VIVE_RELATIVE" = "1" ]; then GATE_CMD_DEFAULT=1; fi
    echo "[3_run_policy_vive_force] UNIFIED (R1 anneal): observe-only FORCED by preset — SERVO=$SERVO is IGNORED."
    echo "[3_run_policy_vive_force] ⚠️  The POLICY still pushes from F^cmd: start with every F^cmd at 0 N."
    echo "[3_run_policy_vive_force] default checkpoint: wda3h4ll/model_19000 (last GOOD TCP-unified export)."
    echo "[3_run_policy_vive_force] TCP-UNIFIED: both hands are tracked at wrist_yaw +x 0.242 m (the force TCP),"
    echo "[3_run_policy_vive_force]     NOT the old 0.18 m VR lever — position and force now share ONE point."
    echo "[3_run_policy_vive_force] ⚠️  wda3h4ll COLLAPSED at step 19498 (episode_length 500→26, all rewards -100x):"
    echo "[3_run_policy_vive_force]     do NOT bump to model_19500 or later — they fall over. 19000 is the ceiling"
    echo "[3_run_policy_vive_force]     unless the run is restarted/rewound from a pre-collapse checkpoint."
    echo "[3_run_policy_vive_force] ⚠️  An R1 ckpt bakes servo_observe_only=False and its terminal α is NOT"
    echo "[3_run_policy_vive_force]     serialized. If α was still ≈1 at export the arms may UNDER-REACH in"
    echo "[3_run_policy_vive_force]     contact (they trained with servo help they do not get here). Check"
    echo "[3_run_policy_vive_force]     Env/servo_alpha ≈ 0 on the wandb run before trusting contact behavior."
    ;;
  5pt-unified-anneal|dense-unified-anneal)
    # MOTION-COMPLETION teleop (user 2026-08-27). The rig has THREE trackers; these two
    # policies want five keypoints / all 29 joint targets, so the SAME three-tracker stream
    # goes through Motion_Completion in the bridge and the extra channels are read out of
    # its 29-DoF FK — at the SAME 0.242 m TCP the hands use, so nothing downstream changes
    # frame. Terminal 1 MUST be the matching bridge mode:
    #     MODE=5     bash 1_vive_bridge.sh      # for 5pt-unified-anneal
    #     MODE=dense bash 1_vive_bridge.sh      # for dense-unified-anneal
    # A 3pt bridge against these presets streams a 9/12-wide frame that the width guard
    # drops, so the robot would hold the ready pose forever rather than mistrack — loud in
    # the log, safe on the rig, but it means "nothing moves" = "wrong bridge mode".
    # CURRENT run FIRST, superseded predecessors after it. A retrain keeps the SAME obs, so
    # the width check, the family probe and the hand-geometry guard all pass on the OLD
    # weights too — nothing downstream can tell you that you just armed a robot against an
    # abandoned run. Ordering here is the only thing that picks right, and a fallback is
    # ANNOUNCED rather than taken quietly.
    if [ "$MODE" = "5pt-unified-anneal" ]; then
      PRESET="g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt"
      # pq9shxs7 = WBT-Hybrid5ptUniAnneal-manipmvbox20p-32gpu-ft60000-r2, the CURRENT 5pt run
      # (finished 2026-09-08; export model_62999.onnx, sha256 45fef74c…, the released low
      # level — the README links it; put it at logs/WBT-Holosoma/pq9shxs7/model_62999.onnx —
      # a FINE-TUNE from iteration 60000, so its step numbers START at 60000).
      # tqnoooo6 (ft23000, the exact twin of dense's 9vstfdbk) is its SUPERSEDED predecessor;
      # na475b60/dlcxc59d older still. All stay as an announced fallback.
      _MC_RUNS=(pq9shxs7 tqnoooo6 na475b60 dlcxc59d)
    else
      PRESET="g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt"
      # 7vxmnwo6 = WBT-HybridDenseUniAnneal-manipmvbox20p-32gpu-ft21000-**r2**, the CURRENT
      # dense run (user 2026-09-01) — a second fine-tune round off the same ft21000 base, so
      # its steps also START at 21000 and it is STILL TRAINING (newest export moves; re-pull).
      # 9vstfdbk is round 1 and stays as an announced fallback; c2cqxjxj/7r54008a are older.
      # Verified byte-compatible with 9vstfdbk before pinning: obs [1,732] and the same five
      # outputs (actions/joint_pos/joint_vel/ref_pos_xyz/ref_quat_xyzw). That IS the hazard —
      # motion_command + the 26D force block in unchanged term order means NO width check,
      # family probe or hand-geometry guard can tell you that you armed the wrong run, so the
      # ORDER of this list is the only thing that picks r2 over r1.
      _MC_RUNS=(7vxmnwo6 9vstfdbk c2cqxjxj 7r54008a)
    fi
    _MC_RUN="${_MC_RUNS[0]}"
    # Both runs train FROM SCRATCH (a 184D/183D actor cannot warm-start off any 170D
    # unified checkpoint), so there is no "last good" export to pin the way wda3h4ll has:
    # take the newest local pull and say which one it is.
    _MC_PICKED=""
    if [ -z "${ONNX:-}" ]; then
      # Collect via nullglob + array, NOT `ls ... | sort -V | tail -1`. Under this script's
      # `set -o pipefail` an `ls` whose operand list contains ONE non-matching glob returns
      # 2, the whole pipeline inherits it, and `set -e` then kills the script IMMEDIATELY
      # AFTER the assignment succeeded -- exit 2 with no output at all. Which is exactly
      # what happened when one candidate dir existed and another did not: the launcher
      # died silently on a correctly-resolved checkpoint.
      for _r in "${_MC_RUNS[@]}"; do
        _MC_CANDS=()
        shopt -s nullglob
        _MC_CANDS+=("logs/WBT-Holosoma/$_r"/model_*.onnx)
        shopt -u nullglob
        if [ ${#_MC_CANDS[@]} -gt 0 ]; then
          # Version sort so model_04500 beats model_03500 (lexical would too here, but not
          # once a run passes model_10000).
          ONNX="$(printf '%s\n' "${_MC_CANDS[@]}" | sort -V | tail -1)"
          _MC_PICKED="$_r"
          break
        fi
      done
    fi
    if [ -z "${ONNX:-}" ] || [ ! -f "$ONNX" ]; then
      echo "[3_run_policy_vive_force] MODE=$MODE has no local checkpoint (looked under logs/WBT-Holosoma/ for runs: ${_MC_RUNS[*]})."
      echo "[3_run_policy_vive_force]     Put one at logs/WBT-Holosoma/$_MC_RUN/model_<step>.onnx (MODE=5pt-unified-anneal:"
      echo "[3_run_policy_vive_force]     the released model_62999.onnx, see the README),"
      echo "[3_run_policy_vive_force]     or pass it explicitly: ONNX=<path> MODE=$MODE bash $0"
      exit 2
    fi
    if [ -n "$_MC_PICKED" ] && [ "$_MC_PICKED" != "$_MC_RUN" ]; then
      echo "[3_run_policy_vive_force] ⚠️  FELL BACK to SUPERSEDED run $_MC_PICKED. The current run for"
      echo "[3_run_policy_vive_force]     MODE=$MODE is $_MC_RUN, which has no local checkpoint. Same obs layout,"
      echo "[3_run_policy_vive_force]     so NO other guard will flag this. Put its checkpoint at"
      echo "[3_run_policy_vive_force]       logs/WBT-Holosoma/$_MC_RUN/model_<step>.onnx"
      _MC_RUN="$_MC_PICKED"
    fi
    if [ "$VIVE_RELATIVE" = "1" ]; then GATE_CMD_DEFAULT=1; fi
    echo "[3_run_policy_vive_force] UNIFIED (R1 anneal): observe-only FORCED by preset — SERVO=$SERVO is IGNORED."
    echo "[3_run_policy_vive_force] ⚠️  The POLICY still pushes from F^cmd: start with every F^cmd at 0 N."
    echo "[3_run_policy_vive_force] MOTION COMPLETION is in the loop (run $_MC_RUN, checkpoint $ONNX)."
    echo "[3_run_policy_vive_force] ⚠️  EXPECT ~667 ms OF COMMAND LAG and ~1 s of no motion after arming."
    echo "[3_run_policy_vive_force]     Neither is a slow host: the completion conditions densely over a"
    echo "[3_run_policy_vive_force]     30-frame window with a 20-frame stride, so the first window cannot run"
    echo "[3_run_policy_vive_force]     until 30 frames exist and gap-free playback needs >= stride-1 = 19"
    echo "[3_run_policy_vive_force]     frames of lag. A SMALLER --mc-lag-frames does not reduce the peak lag"
    echo "[3_run_policy_vive_force]     (measured 700 ms either way) — it only makes it sawtooth. Compute is"
    echo "[3_run_policy_vive_force]     ~45 ms/window (~7% duty), so the host is not the constraint."
    echo "[3_run_policy_vive_force] ⚠️  Run $_MC_RUN trains FROM SCRATCH — an early export tracks POORLY."
    echo "[3_run_policy_vive_force]     Gantry-suspended, hand on the E-stop, and judge the PLUMBING first."
    if [ "$MODE" = "5pt-unified-anneal" ]; then
      echo "[3_run_policy_vive_force] 5pt: hands at wrist_yaw +x 0.242 m (the force TCP), head at torso +0.35 z,"
      echo "[3_run_policy_vive_force]     and BOTH FEET at the bare *_ankle_roll_link origin (no lever)."
    else
      echo "[3_run_policy_vive_force] dense: all 29 joint targets, remapped onto dof_names BY NAME (never by"
      echo "[3_run_policy_vive_force]     index). Pre-arm gate is joint-space (HOLOSOMA_VIVE_ARM_TOL_RAD)."
    fi
    ;;
  wbt)
    # BASE whole-body tracker exp:g1-wbt (W&B run ruo0o6fr, preset g1-29dof-holosoma-wbt,
    # 628D). The THIRD motion-completion mode and deliberately the simplest: no force block,
    # no servo, no contact phase — its whole tracking front-end is the 58D `command` term, so
    # the Terminal-1 bridge is the SAME dense wire the mode above uses.
    #
    # ⚠️ THIS IS THE SAME TRACKING FAMILY AS dense-unified-anneal, exactly: the dense obs is
    # THIS obs with the 26D force block inserted after `command` and nothing else changed
    # (157 + 26 = 183), term order preserved, and both checkpoints spell the front-end
    # `motion_command` in their own metadata. Both therefore need the completion, take the
    # same wire, and carry the same root limitation. `wbt` is not a weaker variant — it is
    # dense MINUS FORCE. Do not treat the two as separate capabilities.
    #
    # WHY THIS ONE FIRST ON HARDWARE: it is the only motion-completion policy whose replay
    # path is already measured end-to-end on recorded takes (100% survival on both, 3.9 deg
    # joint / 9.6 cm body). 5pt/dense are still training from scratch. So a bad rig run here
    # is a plumbing or a teleop problem, not "the policy might just be early" — which is the
    # whole reason to bring a new stack up on a known-good checkpoint.
    PRESET="g1-29dof-holosoma-wbt"
    # Same current-first candidate CHAIN as the 5pt/dense branch, even though there is only
    # one entry today. Spelled as an array rather than a scalar so a future exp:g1-wbt retrain
    # inherits the ordering + the announced-fallback behaviour by adding one word, instead of
    # someone re-pinning a scalar and silently keeping the old weights alive.
    _MC_RUNS=(ruo0o6fr)
    _MC_RUN="${_MC_RUNS[0]}"
    _MC_PICKED=""
    if [ -z "${ONNX:-}" ]; then
      # Same nullglob+array collection as the 5pt/dense branch, and for the same reason:
      # `ls`+pipefail under `set -e` exits 2 with no output when a dir is missing.
      for _r in "${_MC_RUNS[@]}"; do
        _MC_CANDS=()
        shopt -s nullglob
        _MC_CANDS+=("logs/WBT-Holosoma/$_r"/model_*.onnx)
        shopt -u nullglob
        if [ ${#_MC_CANDS[@]} -gt 0 ]; then
          ONNX="$(printf '%s\n' "${_MC_CANDS[@]}" | sort -V | tail -1)"
          _MC_PICKED="$_r"
          break
        fi
      done
    fi
    if [ -z "${ONNX:-}" ] || [ ! -f "$ONNX" ]; then
      echo "[3_run_policy_vive_force] MODE=wbt has no local checkpoint (looked under logs/WBT-Holosoma/ for runs: ${_MC_RUNS[*]})."
      echo "[3_run_policy_vive_force]     Put one at logs/WBT-Holosoma/$_MC_RUN/model_<step>.onnx,"
      echo "[3_run_policy_vive_force]     or pass it explicitly: ONNX=<path> MODE=wbt bash $0"
      exit 2
    fi
    if [ -n "$_MC_PICKED" ] && [ "$_MC_PICKED" != "$_MC_RUN" ]; then
      echo "[3_run_policy_vive_force] ⚠️  FELL BACK to SUPERSEDED run $_MC_PICKED (current: $_MC_RUN)."
      _MC_RUN="$_MC_PICKED"
    fi
    echo "[3_run_policy_vive_force] BASE TRACKER exp:g1-wbt (run $_MC_RUN, checkpoint $ONNX)."
    echo "[3_run_policy_vive_force] ⚠️  NO FORCE BLOCK in this obs (628D = command 58 + motion_ref_ori_b 6 +"
    echo "[3_run_policy_vive_force]     projected_gravity 3 + base_ang_vel 3 + joint_pos/vel/actions 29x3, x4)."
    echo "[3_run_policy_vive_force]     SERVO / FEMA / TARE_* / F^cmd / GATE_CMD are ALL INERT — there is no slot"
    echo "[3_run_policy_vive_force]     for them. Terminal 2 (F/T bridge) is optional and only feeds the 20 N stop."
    echo "[3_run_policy_vive_force]     This force row is the ONLY thing that differs from dense-unified-anneal:"
    echo "[3_run_policy_vive_force]     that obs is THIS obs + the 26D force block (157+26=183), same term order."
    echo "[3_run_policy_vive_force] ⚠️  NO ROOT CHANNEL: no motion_ref_pos_b, so HEADING and world position stay"
    echo "[3_run_policy_vive_force]     pinned to the injected clip. Your trackers steer POSTURE (29 joints) only."
    echo "[3_run_policy_vive_force]     SHARED WITH 5pt AND dense — none of the three MC presets has a root term"
    echo "[3_run_policy_vive_force]     (x_tgt_shift there is the force servo's per-hand dx, NOT a root channel)."
    echo "[3_run_policy_vive_force]     Do not expect to walk the robot around with ANY of these three presets."
    echo "[3_run_policy_vive_force] Terminal 1 MUST be the DENSE wire:  MODE=dense bash 1_vive_bridge.sh"
    echo "[3_run_policy_vive_force]     (MODE=wbt there is accepted as the same thing). A 3pt/5pt bridge streams a"
    echo "[3_run_policy_vive_force]     narrower frame that the width guard drops, so 'nothing moves' = wrong mode."
    echo "[3_run_policy_vive_force] ⚠️  EXPECT ~667 ms OF COMMAND LAG and ~1 s of no motion after arming — the"
    echo "[3_run_policy_vive_force]     completion's window/stride arithmetic, identical to 5pt/dense."
    echo "[3_run_policy_vive_force]     Pre-arm gate is joint-space (HOLOSOMA_VIVE_ARM_TOL_RAD)."
    ;;
  *) echo "[3_run_policy_vive_force] bad MODE='$MODE' (want 2pt|3pt|3pt-global-vel|3pt-unified|3pt-unified-anneal|5pt-unified-anneal|dense-unified-anneal|wbt)"; exit 2 ;;
esac
# FORCE_CMD_WIRE is normalized BEFORE GATE_CMD, because both GATE_CMD's default and
# GATE_CMD_N's default depend on it. The keys/echo half stays at its original site, after the
# ONNX refusals, so its messages can name the resolved checkpoint.
FORCE_CMD_WIRE="${FORCE_CMD_WIRE:-0}"
case "${FORCE_CMD_WIRE,,}" in
  1|true|yes|on) FORCE_CMD_WIRE=1 ;;
  0|false|no|off) FORCE_CMD_WIRE=0 ;;
  *) echo "[3_run_policy_vive_force] bad FORCE_CMD_WIRE='$FORCE_CMD_WIRE' (want 1|0|true|false)"; exit 2 ;;
esac
# Turning the wire ON turns the F_meas gate ON with it. The gate is HALF of what "below the
# threshold there is no force" means: without it, a hand carrying ZERO F^cmd still feeds its
# measured force to the actor obs and the servo, so a stray contact starts a force loop the
# planner never asked for. Since the wire is the ONLY way force reaches this robot, the two
# belong to one decision — and the MODE-scoped default below is 0 for absolute/S1 mode, so
# leaving it there meant the operator had to remember GATE_CMD=1 on exactly the run where
# forgetting it matters most. An explicit GATE_CMD=0 still wins (the fidelity A/B).
if (( FORCE_CMD_WIRE )); then
  GATE_CMD_DEFAULT=1
fi
GATE_CMD_RAW="${GATE_CMD:-$GATE_CMD_DEFAULT}"
case "${GATE_CMD_RAW,,}" in
  1|true|yes|on) GATE_CMD=1 ;;
  0|false|no|off) GATE_CMD=0 ;;
  *) echo "[3_run_policy_vive_force] bad GATE_CMD='$GATE_CMD_RAW' (want 1|0|true|false|yes|no|on|off)"; exit 2 ;;
esac
if (( FORCE_CMD_WIRE )) && (( ! GATE_CMD )); then
  echo "[3_run_policy_vive_force] ⚠️  FORCE_CMD_WIRE=1 with GATE_CMD=0: a hand with NO force command will"
  echo "[3_run_policy_vive_force]     still feed its measured force to the obs + servo, so a stray contact"
  echo "[3_run_policy_vive_force]     can start a force loop below the ${FORCE_CMD_WIRE_GATE_N:-1.5} N gate. Intentional only for a"
  echo "[3_run_policy_vive_force]     fidelity A/B — drop GATE_CMD=0 for the gated deploy."
fi
FORCE_CMD_WIRE_GATE_N="${FORCE_CMD_WIRE_GATE_N:-1.5}"
# WHICH HANDS' F/T SENSORS ARE IN THE LOOP AT ALL (both|left|right|none). Not a gate and not
# a threshold: an excluded hand's reading is zeroed where the sensor is read, so it reaches
# NOTHING — no force_meas_local, no contact_phase, no integrator/x_tgt_shift, no runaway
# valve, no hard stop, no F^cmd. Default is scoped by who is driving:
#   * absolute / S1 planner (HOLOSOMA_VIVE_RELATIVE=0) → LEFT. The HL-v2 checkpoints are all
#     layout `left_only`: the model predicts one hand, publishes a 3-wide (LEFT) f_cmd, and
#     the right arm is merely HELD at its measured pose. The right sensor is therefore not an
#     input to anything — and on this rig its RNEA wrench_sign is still SEEDED/UNVALIDATED
#     (see 2_ft_bridge.sh), so a sign error there makes its reading grow with orientation
#     instead of staying near zero. Leaving it live meant an untrusted sensor could reach the
#     actor obs (whenever GATE_CMD=0) and, with the hard stop armed, halt the robot.
#   * direct VIVE teleop (relative/clutch) → BOTH, unchanged: the operator drives two hands.
if [ "$VIVE_RELATIVE" = "0" ]; then
  FORCE_HANDS="${FORCE_HANDS:-left}"
else
  FORCE_HANDS="${FORCE_HANDS:-both}"
fi
case "${FORCE_HANDS,,}" in
  both|left|right|none) FORCE_HANDS="${FORCE_HANDS,,}" ;;
  *) echo "[3_run_policy_vive_force] bad FORCE_HANDS='$FORCE_HANDS' (want both|left|right|none)"; exit 2 ;;
esac
if [ "$FORCE_HANDS" != "both" ]; then
  echo "[3_run_policy_vive_force] F/T SENSORS IN THE LOOP: $FORCE_HANDS — the excluded hand's reading is zeroed at"
  echo "[3_run_policy_vive_force]     the source: no obs, no contact_phase, no servo, no runaway valve, no hard stop,"
  echo "[3_run_policy_vive_force]     no F^cmd. Pass FORCE_HANDS=both to put it back."
fi
# GATE_CMD_N's default depends on WHO authors F^cmd, because it is the threshold at which the
# MEASURED force is allowed back into the obs + servo, and that only makes sense at the value
# which admits a command in the first place:
#   * keyboard (wire OFF): 6 N, so F_meas re-enters only after a DELIBERATE command
#     (~4 nudges of 2 N) rather than the first key tap.
#   * HL planner (wire ON): the wire deadband, so "the planner commanded force" and "the
#     policy may see the measured force" are ONE threshold. DERIVED from
#     FORCE_CMD_WIRE_GATE_N rather than re-typed — two independently spelled copies of the
#     same 1.5 is exactly how a gate ends up half-applied.
# An explicit GATE_CMD_N= from the operator still wins over both.
if (( FORCE_CMD_WIRE )); then
  GATE_CMD_N="${GATE_CMD_N:-$FORCE_CMD_WIRE_GATE_N}"
else
  GATE_CMD_N="${GATE_CMD_N:-6}"
fi
# One validator for every force THRESHOLD: each is a safety-relevant number, and a tolerant
# parse fails silently (bash would treat 'FORCE_ESTOP_N=20N' as a string and the runtime
# would fall back to "off" on a run the operator believes is protected). Shell-side regex
# for the shape, then Python for finiteness — 'inf'/'nan' pass many regexes.
_require_nonneg_number() {  # $1 = name, $2 = value
  if [[ ! "$2" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || ! "$PY" - "$2" <<'PY_CHECK'
import math
import sys

value = float(sys.argv[1])
raise SystemExit(0 if math.isfinite(value) and value >= 0.0 else 1)
PY_CHECK
  then
    echo "[3_run_policy_vive_force] bad $1='$2' (want a finite non-negative number)"
    exit 2
  fi
}
_require_nonneg_number GATE_CMD_N "$GATE_CMD_N"
_require_nonneg_number FORCE_CMD_WIRE_GATE_N "$FORCE_CMD_WIRE_GATE_N"
_require_nonneg_number FORCE_ESTOP_N "$FORCE_ESTOP_N"
_require_nonneg_number WINDUP_WARN_N "$WINDUP_WARN_N"
# The runaway valve is DERIVED from the stop, never typed, so it stays ABOVE it and the hard
# stop remains the SINGLE behaviour at high force. It used to be a literal 35 against a 20 N
# stop, which held that invariant by luck; raising the stop to 40 would have inverted it and
# opened a 35-40 N band nothing has ever exercised on hardware. That band is not benign: in a
# unified (observe-only) preset a runaway trip HOLDS the servo integrator, and I/Δx are that
# policy's only force observation, so it would freeze part of the obs a few Newtons short of
# the stop. +15 keeps the historical spacing. With the stop OFF the valve is the only valve
# left, so it falls back to its original absolute 35 N rather than to "15 N above nothing".
FORCE_RUNAWAY_N="$(awk -v e="$FORCE_ESTOP_N" 'BEGIN { printf "%.1f", (e > 0 ? e + 15 : 35) }')"
# Compared with awk, not "$PY": this is a launch-time sanity check on two numbers, and it
# must hold even when the policy interpreter is not the thing being exercised (the launcher
# tests stub PY). awk 'BEGIN{exit !(cond)}' exits 0 exactly when the BAD condition holds.
if (( FORCE_CMD_WIRE )) && awk -v g="$FORCE_CMD_WIRE_GATE_N" -v e="$FORCE_ESTOP_N" \
     'BEGIN { exit !(e > 0 && g >= e) }'; then
  echo "[3_run_policy_vive_force] FORCE_CMD_WIRE_GATE_N=$FORCE_CMD_WIRE_GATE_N >= FORCE_ESTOP_N=$FORCE_ESTOP_N:"
  echo "[3_run_policy_vive_force]     the smallest force the planner is allowed to command would itself trip the"
  echo "[3_run_policy_vive_force]     hard stop, so force mode could never run. Lower the gate or raise the stop."
  exit 2
fi
# The reference clip the preset requires. HOLOSOMA_INIT_READY_POSE=1 (below) pins the init and the
# frozen hold to the READY pose rather than the clip's frame 0, so under the trackers or the
# planner the clip plays only when select+A starts it. Default: holosoma's own G1 demo clip, which
# ships with the submodule (and so reaches the robot with onboard/sync_to_g1.sh).
CLIP="${CLIP:-$PWD/thirdparty/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz}"
echo "[3_run_policy_vive_force] MODE=$MODE preset=$PRESET onnx=$ONNX GATE_CMD=$GATE_CMD"
# The policy's own banner states the hard stop, but not the runaway valve (which only speaks
# when it trips) — and that one is derived now, so print the resolved trio side by side. Two
# spellings, because "gate < STOP=0 < runaway" would print a false ordering on the one run
# where the operator most needs to read what is actually protecting the robot.
if awk -v e="$FORCE_ESTOP_N" 'BEGIN { exit !(e > 0) }'; then
  echo "[3_run_policy_vive_force] force thresholds: gate=${GATE_CMD_N} N < STOP=${FORCE_ESTOP_N} N" \
       "< runaway=${FORCE_RUNAWAY_N} N (derived = STOP+15; the STOP is the only high-force action)"
else
  echo "[3_run_policy_vive_force] force thresholds: gate=${GATE_CMD_N} N, ⚠️ HARD STOP OFF" \
       "(FORCE_ESTOP_N=0) — only the ${FORCE_RUNAWAY_N} N runaway valve is left, and it merely warns" \
       "and freezes the optional Jᵀ servo. The hardware E-STOP is the only abort."
fi
# The observe-only integrator's ‖I‖ alarm is print-only, so it is announced apart from the
# thresholds that ACT. Above 40·√3 ≈ 69 N (every axis at the clip) it cannot fire at all.
if awk -v w="$WINDUP_WARN_N" 'BEGIN { exit !(w > 0 && w <= 69.3) }'; then
  echo "[3_run_policy_vive_force] integrator ‖I‖ alarm at ${WINDUP_WARN_N} N (WINDUP_WARN_N; print-only, unified presets)"
else
  echo "[3_run_policy_vive_force] integrator ‖I‖ alarm OFF (WINDUP_WARN_N=${WINDUP_WARN_N}: 0, or above the" \
       "69 N the 40 N/axis clip allows) — print-only, nothing else changes; WINDUP_WARN_N=20 restores it"
fi
# Known-bad exports from iss0qs71: all 15 checkpoints in this closed interval have
# 376 NaNs in normalizer_mean + 376 in normalizer_std and produce all-NaN actions.
# Match the run directory and basename separately so W&B HTTPS URLs and harmless
# path components such as iss0qs71/./model_27500.onnx cannot bypass the rejection.
_ONNX_PATH="${ONNX%%[?#]*}"
_ONNX_MATCH_PATH="$_ONNX_PATH"
case "$_ONNX_PATH" in
  *://*)
    # W&B resolves the filename with pathlib, so trim equivalent trailing / and /.
    # Otherwise model_27500.onnx/. normalizes back to the prohibited cached file.
    while :; do
      case "$_ONNX_MATCH_PATH" in
        */.) _ONNX_MATCH_PATH="${_ONNX_MATCH_PATH%/.}" ;;
        */) _ONNX_MATCH_PATH="${_ONNX_MATCH_PATH%/}" ;;
        *) break ;;
      esac
    done
    ;;
  *) _ONNX_MATCH_PATH="$(realpath -m -- "$_ONNX_PATH")" ;;
esac
_ONNX_BASENAME="${_ONNX_MATCH_PATH##*/}"
_NOT_FORCE_QUALIFIED=0
if [[ "/$_ONNX_MATCH_PATH/" == */iss0qs71/* && "$_ONNX_BASENAME" =~ ^model_([0-9]+)[.]onnx$ ]]; then
  _ONNX_STEP=$((10#${BASH_REMATCH[1]}))
  if (( _ONNX_STEP >= 27500 && _ONNX_STEP <= 34500 )); then
    echo "[3_run_policy_vive_force] refusing known-bad iss0qs71/$_ONNX_BASENAME (NaN normalizer/actions)"
    exit 2
  fi
  if (( _ONNX_STEP == 27000 )); then _NOT_FORCE_QUALIFIED=1; fi
fi
# The TCP-unified retrain wda3h4ll has the SAME not-force-qualified property as
# iss0qs71/model_27000, so it must raise the same warning rather than inherit silence from
# a run-name-specific check: W&B reports Env/servo_alpha=1 for its ENTIRE history
# (min=max=1 through step 22643) — the ServoAuthorityCurriculum anneal never fired — so
# every export trained WITH full Jᵀ-servo help that it does NOT get at deploy. Unlike
# model_27000 it starts at force_alpha_init=1 (no ramp), so it HAS seen full-magnitude
# F^cmd; the gap is servo help, not force exposure. Marked for the whole run, not one step,
# because that is a property of the run's curriculum.
#
# POST-COLLAPSE REFUSAL. wda3h4ll collapsed at step 19498: Env/average_episode_length went
# 499.8 (@19497) → 212.6 (@19499) → 25.96 (@19509) and stayed ~26, while every reward fell
# ~100x at the same step; the exports then diverge to NaN. MEASURED on the actual files:
# model_19500 / 20000 / 22500 each carry 376 NaNs in normalizer_mean AND 376 in
# normalizer_std and emit all-NaN actions — byte-for-byte the same signature as the retired
# iss0qs71 27500..34500 window above. model_19000 is clean (0 NaN, finite actions).
#
# Refused HERE, ahead of the loader, for the same reason as the missing-file check: the ONNX
# is only opened inside setup_policy, which base.py runs AFTER _init_communication_components,
# so a NaN checkpoint takes the DDS interface up first and then dies mid-init. A step window
# also covers the (untested) possibility of a post-collapse export that is finite but still
# falls over — reward and episode_length had already cratered a step BEFORE the NaNs, so
# "finite" would not mean "safe" anywhere past 19000.
# Same matcher shape as the iss0qs71 rule so W&B URLs and /./ path noise cannot bypass it.
if [[ "/$_ONNX_MATCH_PATH/" == */wda3h4ll/* ]]; then
  _NOT_FORCE_QUALIFIED=1
  if [[ "$_ONNX_BASENAME" =~ ^model_([0-9]+)[.]onnx$ ]] && (( 10#${BASH_REMATCH[1]} >= 19500 )); then
    echo "[3_run_policy_vive_force] ⚠️  REFUSING wda3h4ll/$_ONNX_BASENAME — POST-COLLAPSE checkpoint."
    echo "[3_run_policy_vive_force]     The run collapsed at step 19498: Env/average_episode_length 499.8 → 26"
    echo "[3_run_policy_vive_force]     and every reward dropped ~100x in ONE step, then the exports went NaN"
    echo "[3_run_policy_vive_force]     (376 NaNs in each normalizer + all-NaN actions, measured on 19500/20000/22500)."
    echo "[3_run_policy_vive_force]     Use model_19000 (the default, last good export) or unset ONNX."
    echo "[3_run_policy_vive_force]     Aborting before DDS init."
    exit 2
  fi
fi
# FORCE_CMD_WIRE=1: accept a per-tick F^cmd from the :8770 WIRE (the HL-v2 online planner
# scripts/hl_serving.py streams a gated, sensor-frame force command alongside the pose).
# Default OFF → the wire's f_cmd is ignored and F^cmd stays 0 (pure position). Mutually
# exclusive with the keyboard keys (both drive the same buffer), so enabling the wire
# DISABLES the keys. FORCE_CMD_WIRE_GATE_N is the low-level hard-zero deadband (N) on the
# commanded force: it MIRRORS the planner's own gate (hl_serving --force-gate-n, default
# 1.5) so a planner bug cannot push a sub-gate command through. Only the HL planner sends
# f_cmd; the VIVE bridge never does.
# (FORCE_CMD_WIRE + the force thresholds were normalized and validated above, next to
# GATE_CMD, because GATE_CMD_N's default is derived from them.)
# Keys ON by default, but OFF when the wire drives F^cmd (mutual exclusion in holosoma_wbt).
_FORCE_CMD_KEYS=1
if (( FORCE_CMD_WIRE )); then
  _FORCE_CMD_KEYS=0
  echo "[3_run_policy_vive_force] FORCE_CMD_WIRE=1 — F^cmd taken from the :8770 wire (HL planner);"
  echo "[3_run_policy_vive_force]     keyboard F^cmd keys DISABLED. Low-level deadband ${FORCE_CMD_WIRE_GATE_N} N,"
  echo "[3_run_policy_vive_force]     F_meas re-enters obs+servo above ${GATE_CMD_N} N of F^cmd (GATE_CMD=$GATE_CMD)."
else
  echo "[3_run_policy_vive_force] FORCE_CMD_WIRE=0 — PURE POSITION: the wire's f_cmd is IGNORED, F^cmd stays 0."
fi
_FORCE_CMD_LOCAL="${HOLOSOMA_FORCE_CMD_LOCAL:-}"
if (( _NOT_FORCE_QUALIFIED )); then
  # This checkpoint exported with Env/servo_alpha=1 — it trained WITH full Jᵀ-servo help it
  # does NOT get at deploy, so it is NOT force-qualified. The keys are ENABLED (you asked to
  # drive F^cmd), but this is a loud warning, not a silent OK: any nonzero F^cmd makes the
  # POLICY push (x_tgt_shift=(F^cmd+I)/K) even in FREE SPACE, and the checkpoint may
  # UNDER-REACH in contact. Ramp in one 2 N nudge at a time.
  echo "[3_run_policy_vive_force] ⚠️  $_ONNX_BASENAME is NOT force-qualified (servo_alpha=1 at"
  echo "[3_run_policy_vive_force]     export — trained WITH full Jᵀ-servo help it does NOT get here)."
  echo "[3_run_policy_vive_force]     F^cmd keys (u/j/h/k/y/n) + HOLOSOMA_FORCE_CMD_LOCAL are ENABLED."
  echo "[3_run_policy_vive_force]     ⚠️  Any nonzero F^cmd makes the POLICY push (x_tgt_shift=(F^cmd+I)/K)"
  echo "[3_run_policy_vive_force]         even in FREE SPACE, and it may UNDER-REACH in contact. GANTRY +"
  echo "[3_run_policy_vive_force]         hardware E-STOP mandatory; ramp F^cmd one 2 N nudge at a time."
  if [ -n "$_FORCE_CMD_LOCAL" ]; then
    echo "[3_run_policy_vive_force]     ⚠️⚠️  HOLOSOMA_FORCE_CMD_LOCAL='$_FORCE_CMD_LOCAL' is set: this is applied"
    echo "[3_run_policy_vive_force]         UN-SLEWED at FULL magnitude on the FIRST START tick (NOT ramped like"
    echo "[3_run_policy_vive_force]         the keys) — the arm LUNGES at START. Unset it unless you mean it."
  fi
fi
# Fail-fast on a missing LOCAL checkpoint: run_policy would otherwise bring the DDS
# interface up FIRST (base.py inits communication before the policy) and only then die
# — leaving the robot in a half-initialised state. Remote schemes are skipped, NOT
# rejected: base.py::_resolve_model_path downloads wandb:// and https:// itself.
case "$ONNX" in
  wandb://*|https://*) ;;   # remote — base.py::_resolve_model_path downloads these
  *://*) echo "[3_run_policy_vive_force] unsupported scheme in ONNX=$ONNX (only wandb:// and https://)"; exit 2 ;;
  *)
    if [ ! -e "$ONNX" ]; then
      echo "[3_run_policy_vive_force] ⚠️  checkpoint NOT FOUND: $ONNX"
      echo "[3_run_policy_vive_force]     Put it there (MODE=5pt-unified-anneal: the released model_62999.onnx, see the"
      echo "[3_run_policy_vive_force]     README), pass ONNX=wandb://<entity>/<project>/<run>/model_XXXXX.onnx to let"
      echo "[3_run_policy_vive_force]     run_policy download it, or ONNX=/abs/path. Aborting before DDS init."
      exit 2
    fi
    # PRE-FLIGHT HAND-GEOMETRY CHECK (unified modes, LOCAL checkpoints only, ~0.7 s).
    # The runtime already refuses a mismatch — _validate_obs_term_order compares the preset's
    # actor_obs against the ONNX experiment_config — but it does so inside setup_policy,
    # which base.py runs AFTER _init_communication_components: the DDS interface is already
    # up when it raises, leaving the robot half-initialised. That is exactly the failure the
    # missing-file check above exists to avoid, so make the same guarantee for the geometry.
    #
    # Why it can't be a width check: the TCP retrain kept the obs at 170D/680D, so a 0.18 m
    # checkpoint loaded against this 0.242 m preset passes every dimension assert and only
    # the per-term NAME differs (slot 1). Silently accepting it would command each hand a
    # target 6.7 cm from where the policy trained. Reads metadata only (no external tensor
    # data, no onnxruntime session), so it is cheap even on the 784 MB motion-embedded export.
    # Non-fatal on its own failure (missing onnx module, unreadable metadata, an older export
    # with no experiment_config) — the runtime check remains the backstop.
    case "$MODE" in
      3pt-unified|3pt-unified-anneal)
        _GEOM="$("$PY" - "$ONNX" <<'PY_GEOM' 2>/dev/null || echo skip
import json, sys
import onnx
m = onnx.load(sys.argv[1], load_external_data=False)
meta = {x.key: x.value for x in m.metadata_props}
terms = (
    json.loads(meta.get("experiment_config", "{}"))
    .get("observation", {}).get("groups", {}).get("actor_obs", {}).get("terms", {})
)
if not terms:
    print("skip")
elif any("vr_3point_pos_tcp" in k for k in terms):
    print("tcp")
elif any("vr_3point_pos" in k for k in terms):
    print("lever")
else:
    print("skip")
PY_GEOM
)"
        case "$_GEOM" in
          tcp) echo "[3_run_policy_vive_force] pre-flight: checkpoint tracks the 0.242 m TCP — matches the preset. OK." ;;
          lever)
            echo "[3_run_policy_vive_force] ⚠️  REFUSING $_ONNX_BASENAME: it trained on the 0.18 m VR-tracker lever"
            echo "[3_run_policy_vive_force]     (obs term 'vr_3point_pos'), but MODE=$MODE now feeds the TCP-unified"
            echo "[3_run_policy_vive_force]     obs 'vr_3point_pos_tcp' (hands at 0.242 m). Same 680D width, so no"
            echo "[3_run_policy_vive_force]     dimension check catches it — each hand would be commanded a target"
            echo "[3_run_policy_vive_force]     6.7 cm from where the policy trained."
            echo "[3_run_policy_vive_force]     Retired: iss0qs71/* and m3s0pw9i/*. Use the TCP retrain wda3h4ll"
            echo "[3_run_policy_vive_force]     (default), or unset ONNX. Aborting before DDS init."
            exit 2
            ;;
          *) echo "[3_run_policy_vive_force] pre-flight hand-geometry check skipped (no obs metadata); runtime will verify." ;;
        esac
        ;;
      5pt-unified-anneal|dense-unified-anneal|wbt)
        # PRE-FLIGHT TRACKING-FAMILY CHECK, same "fail before DDS init" rationale as the
        # geometry check above, but the hazard is different: dense-unified is 732D and so is
        # g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt. That collision is an
        # arithmetic coincidence between two disjoint layouts, so EVERY dimension assert
        # passes and only the term names differ. Cross-loading them would feed 29 joint
        # targets into a 3-keypoint slot (or vice versa) — a scrambled obs, not a small error.
        #
        # `wbt` (the BASE tracker) is joint-space too, so the coarse probe cannot separate it
        # from dense-unified on the tracking term alone. `contact_phase` does — it is present
        # in every unified preset and in no other — so the probe reports the two as distinct
        # families. 628D vs 732D would also catch this pair on width, but naming it here
        # fails BEFORE DDS init and says which run to use instead.
        _FAM="$("$PY" - "$ONNX" <<'PY_FAM' 2>/dev/null || echo skip
import json, sys
import onnx
m = onnx.load(sys.argv[1], load_external_data=False)
meta = {x.key: x.value for x in m.metadata_props}
terms = (
    json.loads(meta.get("experiment_config", "{}"))
    .get("observation", {}).get("groups", {}).get("actor_obs", {}).get("terms", {})
)
if not terms:
    print("skip")
elif any("vr_5point_pos_tcp" in k for k in terms):
    print("5pt")
elif any("vr_3point" in k or "vr_2point" in k for k in terms):
    print("keypoint3")
elif any(k.endswith("command") or k.endswith("motion_command") or k.endswith("tml_command") for k in terms):
    # Joint-space. Split unified (has the force block's contact_phase) from the base tracker.
    print("dense" if any("contact_phase" in k for k in terms) else "dense_base")
else:
    print("skip")
PY_FAM
)"
        case "$MODE" in
          5pt-unified-anneal)   _WANT_FAM=5pt ;;
          dense-unified-anneal) _WANT_FAM=dense ;;
          wbt)                  _WANT_FAM=dense_base ;;
        esac
        case "$_FAM" in
          # EMPTY counts as skip, not as a refusal. `|| echo skip` only fires when $PY exits
          # NONZERO; a $PY that succeeds while printing nothing (a stub, a wrapper, an onnx
          # build that writes only to stderr) leaves _FAM="" and used to fall through to the
          # catch-all below — aborting the launch with "its tracking family is ''", which
          # reads as a checkpoint mismatch when it is really "the probe told us nothing".
          # Unlike the geometry check above, whose catch-all IS the skip arm, this case's
          # catch-all is the refusal, so the empty string has to be named. Applies to
          # 5pt/dense as much as to wbt.
          skip|"") echo "[3_run_policy_vive_force] pre-flight tracking-family check skipped (no obs metadata); runtime will verify." ;;
          "$_WANT_FAM") echo "[3_run_policy_vive_force] pre-flight: checkpoint tracking family is '$_FAM' — matches MODE=$MODE. OK." ;;
          *)
            echo "[3_run_policy_vive_force] ⚠️  REFUSING $_ONNX_BASENAME: its tracking family is '$_FAM', but"
            echo "[3_run_policy_vive_force]     MODE=$MODE feeds a '$_WANT_FAM' observation. The dense-unified obs is"
            echo "[3_run_policy_vive_force]     732D and so is the 3pt-global-vel-unified obs, so NO dimension check"
            echo "[3_run_policy_vive_force]     separates THOSE — the policy would be handed a scrambled vector."
            echo "[3_run_policy_vive_force]     5pt wants run tqnoooo6, dense wants 7vxmnwo6, wbt wants ruo0o6fr."
            echo "[3_run_policy_vive_force]     ('dense' vs 'dense_base' = with vs without the force block: the"
            echo "[3_run_policy_vive_force]      unified policy is 732D and carries contact_phase, the base tracker"
            echo "[3_run_policy_vive_force]      is 628D and carries none.) Aborting before DDS init."
            exit 2
            ;;
        esac
        ;;
    esac
    ;;
esac
# -----------------------------------------------------------------------------

# VIVE mapping mode. Default 1 = RELATIVE/clutch (byte-identical to before for all
# existing teleop use). Set VIVE_RELATIVE=0 for the ABSOLUTE mapping (tracker/planner
# pose IS the hand target + pre-arm pose gate) — REQUIRED when the high-level MuSe force
# PLANNER (scripts/muse_force_serving_*) drives this controller, since it publishes
# absolute pelvis-frame targets. Made env-overridable because an INLINE `VAR=val cmd`
# assignment (as this block is) OVERRIDES a prefixed env var, so VIVE_RELATIVE was
# resolved from HOLOSOMA_VIVE_RELATIVE above before selecting the gate default.
echo "[3_run_policy_vive_force] VIVE_RELATIVE=$VIVE_RELATIVE (1=direct VIVE clutch/S0; 0=absolute planner input)"
echo "[3_run_policy_vive_force] SLEW_FEET=$SLEW_FEET (0=the 5pt FEET bypass the 0.005/0.02 per-tick caps, upper 3 points still capped; 1=all five in one pool)"
echo "[3_run_policy_vive_force] REF_JOINTS=$REF_JOINTS (1=a 5pt frame's optional ref_joints drives command_lower_body while armed — the walk skill's gait legs; 0=ignore it, legs stay the frozen clip)"
echo "[3_run_policy_vive_force] CPU: onnxruntime intra_op=${_ORT_INTRA} spin=${_ORT_SPIN}, BLAS/OMP=${_BLAS_THREADS}" \
     "(inference is C++ already; this only stops the pools busy-waiting — CPU_THREADS=0 to uncap)"

# ROBOT NIC. The development workstation's G1 link is enp132s0; any other machine's (the robot's
# own Jetson included) is whatever carries the 192.168.123.0/24 address. IFACE= pins it; otherwise
# keep enp132s0 when it exists and fall back to the one interface that holds a 192.168.123.x
# address, saying so.
# A trailing --task.interface <nic> still wins (tyro last-wins).
IFACE="${IFACE:-}"
if [ -z "$IFACE" ]; then
  if ip -o link show enp132s0 >/dev/null 2>&1; then
    IFACE=enp132s0
  else
    IFACE="$(ip -o -4 addr show 2>/dev/null | awk '$4 ~ /^192\.168\.123\./ {print $2; exit}')"
    if [ -n "$IFACE" ]; then
      echo "[3_run_policy_vive_force] no enp132s0 on this box; using IFACE=$IFACE (it holds the 192.168.123.x robot address). IFACE= overrides."
    else
      IFACE=enp132s0
      echo "[3_run_policy_vive_force] ⚠️  no interface holds a 192.168.123.x address and enp132s0 does not exist — the robot link is down or unconfigured. Passing --task.interface $IFACE anyway; set IFACE= or configure the link."
    fi
  fi
fi

HOLOSOMA_VIVE_WS_URL=ws://localhost:8770 \
HOLOSOMA_VIVE_RELATIVE="$VIVE_RELATIVE" \
HOLOSOMA_VIVE_ORIENTATION=1 \
HOLOSOMA_INIT_READY_POSE=1 \
HOLOSOMA_ACTION_EMA_ALPHA=0.85 \
HOLOSOMA_VIVE_MAX_POS_STEP=0.005 \
HOLOSOMA_VIVE_MAX_ORN_STEP=0.02 \
HOLOSOMA_VIVE_SLEW_FEET="$SLEW_FEET" \
HOLOSOMA_VIVE_REF_JOINTS="$REF_JOINTS" \
HOLOSOMA_FORCE_WS_URL=ws://localhost:8771 \
HOLOSOMA_FORCE_MEAS_SIGN=1 \
HOLOSOMA_FORCE_MEAS_EMA_ALPHA="$FEMA" \
HOLOSOMA_FORCE_MEAS_GATE_ON_CMD="$GATE_CMD" \
HOLOSOMA_FORCE_MEAS_GATE_CMD_N="$GATE_CMD_N" \
HOLOSOMA_FORCE_TARE_ON_START="$TARE_N" \
HOLOSOMA_FORCE_TARE_ON_START_SECONDS="$TARE_S" \
HOLOSOMA_HYBRID_SERVO="$SERVO" \
HOLOSOMA_FORCE_CMD_LOCAL="$_FORCE_CMD_LOCAL" \
HOLOSOMA_FORCE_CMD_KEYS="$_FORCE_CMD_KEYS" \
HOLOSOMA_FORCE_CMD_WIRE="$FORCE_CMD_WIRE" \
HOLOSOMA_FORCE_CMD_WIRE_GATE_N="$FORCE_CMD_WIRE_GATE_N" \
HOLOSOMA_FORCE_CMD_KEY_STEP_N=2.0 \
HOLOSOMA_FORCE_CMD_KEY_CLAMP_N=25.0 \
HOLOSOMA_FORCE_CMD_KEY_SLEW_N=2.0 \
HOLOSOMA_FORCE_CONTACT_N=2.0 \
HOLOSOMA_FORCE_RUNAWAY_N="$FORCE_RUNAWAY_N" \
HOLOSOMA_FORCE_ESTOP_N="$FORCE_ESTOP_N" \
HOLOSOMA_UNIFIED_WINDUP_WARN_N="$WINDUP_WARN_N" \
HOLOSOMA_FORCE_HANDS="$FORCE_HANDS" \
HOLOSOMA_RECORD="$RECORD" \
HOLOSOMA_ORT_INTRA_THREADS="$_ORT_INTRA" \
HOLOSOMA_ORT_SPIN="$_ORT_SPIN" \
OPENBLAS_NUM_THREADS="$_BLAS_THREADS" \
OMP_NUM_THREADS="$_BLAS_THREADS" \
MKL_NUM_THREADS="$_BLAS_THREADS" \
NUMEXPR_NUM_THREADS="$_BLAS_THREADS" \
"$PY" -m wbt_wrappers_inference.run_policy \
  "inference:$PRESET" \
  --task.model-path "$ONNX" \
  --task.ref-motion-path "$CLIP" \
  --task.interface "$IFACE" \
  --task.fk-odometry true "$@"
