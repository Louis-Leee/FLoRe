"""THE single source of truth for the S1 high-level action/observation layout.

Why this module exists
----------------------
The pre-v2 pipeline hard-coded a fixed 32-dim action vector (29 real + 3 zero pad)
in FIVE places that could — and did — drift: the dataset slot table, the
``_REF_FRAME_SLOTS`` table in ``transforms.py``, the model's per-group metric dims,
the serving un-relativizer's spec list, and the client's 3-point splitter. Only two
of the five were pinned by a test. Every consumer now derives its slices from ONE
:class:`ActionLayout` built here, so a mode change cannot silently desynchronize
train/serve.

The v2 contract (per hand, per action step)
------------------------------------------
A hand contributes a CONTIGUOUS 16-dim block::

    [pos 3][rot6d 6][gripper 1][force 6]     = 16

and the head contributes a 9-dim block (a body reference, so no gripper and no
force sensor)::

    [pos 3][rot6d 6]                          = 9

Modes (named after which TRACKERS are used, matching the recording modes):

    left_only              L                 16 dims/step   1 F/T, 1 gripper
    right_only             R                 16 dims/step   1 F/T, 1 gripper
    both_hands             L, R              32 dims/step   1 F/T, 1 gripper (LEFT)
    both_hands_head        L, R, head        41 dims/step   1 F/T, 1 gripper (LEFT)
    both_hands_head_dual   L, R, head        41 dims/step   **2 F/T, 2 grippers**

and a FAMILY of modes whose action is not keypoints at all (see "The WHOLE-BODY modes"
below). All five share ONE 43-dim action vector and differ only in what the OBSERVATION's
``state`` carries:

    mode                     state                                       state_dim
    wholebody                [29 joints][L grip][R grip]                    31
    wholebody_geom_xy_rpy    [rel_xy 4][rpy 6][L grip][R grip]              12
    wholebody_geom_xy        [rel_xy 4][L grip][R grip]                      6
    wholebody_geom_rpy       [rpy 6][L grip][R grip]                         8
    wholebody_geom_none      [L grip][R grip]                                2

Keypoint order is always ``left, right, head`` restricted to the mode, so a
smaller mode's block layout is a strict PREFIX of a larger one that starts with
the same hand: ``left_only`` ⊂ ``both_hands`` ⊂ ``both_hands_head``. That is
deliberate: it lets a left-only checkpoint's weights be expanded into a bimanual
model by appending columns, without permuting the ones already learned.

``both_hands_head_dual`` is the two-hand-HARDWARE mode (the rig's second Bota + second
DM4310, recorded since 2026-08-30 and converted with ``--hands both_head_dual``). It is
the mode ``diffusion_umi_3tracker`` trains. Its ACTION vector is **column-identical** to
``both_hands_head``'s 41 dims — every hand block always reserved a gripper and a force
slot, so declaring the hardware moves nothing (that is exactly what "RESERVED vs DECLARED"
below is for). What changes is:

* ``measured_dims()`` grows by 7 (the right hand's gripper column + its 6 force columns),
  so the loss finally supervises them instead of masking them False;
* ``force_hist_channels`` 6 -> **12** (two 6-DoF wrenches);
* ``state_dim`` 19 -> **20** (the right hand gains its observed gripper opening).

⚠️ ``state`` is therefore NOT prefix-compatible between the two, even though the action
vector is. The right hand's gripper slot is INSERTED at index 13, which pushes the head's
``wrt_start_rot6d`` from ``[13:19]`` to ``[14:20]`` — the same widths would load and the
head's orientation would silently be read as (gripper, 5 of 6 rot6d components). Never
cross-load a ``both_hands_head`` checkpoint as ``both_hands_head_dual`` or the reverse;
``layout_mode`` is baked into every checkpoint and ``validate_model_layout`` pins it, and
the converter stamps ``attrs.hands_mode`` so the DATA refuses the swap too. Pinned by
``tests/test_layout.py::test_dual_mode_shares_the_action_layout_but_not_the_state_layout``.

⚠️ ``right_only`` is NOT prefix-compatible with ANY other mode, and must never be
expanded that way. It puts the right hand at dims [0:16] (it is that mode's first
keypoint), whereas ``both_hands`` puts the right hand at [16:32]. Expanding a
``right_only`` checkpoint by appending columns would therefore read every learned
right-hand column as the LEFT hand — and because the shapes match at every step, the
failure is completely silent. Retrain instead. (Pinned by
``tests/test_layout.py::test_right_only_is_not_prefix_compatible``.)

``legacy32`` is the frozen pre-v2 layout, kept ONLY so the already-trained
checkpoint (and the deploy path built against it) keeps loading. It is NOT a
prefix of the v2 modes and must never be mixed with them — the pad dims and the
non-contiguous per-hand slots are exactly what v2 removes.

The WHOLE-BODY modes (``wholebody``, ``wholebody_geom_*``)
---------------------------------------------------------
Every keypoint mode above predicts KEYPOINTS — Cartesian poses that a downstream stage still has to
turn into a robot configuration (the deploy path runs motion completion for exactly that,
see ``fuji/s1_policy/serving/hl_dense.py``). ``wholebody`` predicts the CONFIGURATION
itself: the 29 G1 joint angles the ``dense`` low level consumes directly, so the completion
moves from deploy time to DATA-PREPARATION time (``motion_completion/mc_to_hl_wholebody.py``
writes them into the zarr; nothing runs a model at 30 Hz to serve them)::

    [0:29]   joints    29 absolute joint angles, rad, :data:`WB_JOINT_NAMES` order
    [29]     L gripper normalized opening
    [30:36]  L force   6-DoF wrench, bota sensor frame
    [36]     R gripper
    [37:43]  R force

That is EXACTLY what ``DenseWire`` publishes plus the two things it cannot carry: it sends
``(29 joint pos, 29 joint vel)`` and derives the velocities from the positions by finite
difference, so a 29-column chunk on this build's 30 Hz grid is complete — there is **no root
channel to predict** (none of the whole-body low levels has one; they track 29 joints + the
reference body's ORIENTATION), and the grippers/wrenches go out on their own channels.

Three properties of this layout are load-bearing and each is a departure from the modes
above, not a variation of them:

* **Absolute, not anchor-relative.** ``pose_dims()`` is EMPTY, so nothing here undergoes the
  ``absolute_to_delta_ee`` transform and there is no anchor to un-relativize at serving. Joint
  angles already live in a body-fixed frame, so the OOD hazard that made v2 relativize poses
  (training in lighthouse-world metres, serving in pelvis frame) does not exist for them. The
  deploy-side ``JointClutch`` publishes ``base + (q[i] - q[0])`` — differences within one
  chunk — so an absolute chunk is what it wants, and because that subtraction is invariant to
  a constant offset it ALSO accepts an anchor-relative one unchanged. That is what makes
  :data:`WB_JOINT_TARGETS` a per-run A/B (``--hl-data.wb-joint-target delta``) rather than a
  second layout: the widths, the wire and the clutch are identical, only what the network is
  asked to represent differs. ``absolute`` is the default and the shipped behaviour.
* **``roles`` is ``("left", "right")`` but the mode still needs THREE trackers.** The roles
  own only the per-hand ``[grip 1][force 6]`` blocks; the chest is inside the 29 joints (via
  the waist chain and the completion's root conditioning) rather than being its own 9-dim
  block. So ``tracker_roles_required`` returns all three, and a consumer that scans "the roles
  whose tracker glitch corrupts this sample" must read THAT and not ``layout.roles`` — a chest
  rotation glitch corrupts the completion's root channel and therefore all 29 joint columns.
* **It is a SUPERSET of ``both_hands_head_dual`` on disk, not a different conversion.**
  ``hands_flag`` is the same ``both_head_dual``, the gripper/force targets are the very same
  ``hl_<role>_action_{gripper,force}`` arrays, and the joints arrive as ADDITIONAL ``wb_*``
  arrays. One zarr therefore trains either preset, and the 3-tracker preset over an augmented
  file is byte-identical to the same file before augmentation (pinned by
  ``tests/test_diffusion_umi_wholebody.py``).

The ``wholebody_geom_*`` family: the SAME action, a different STATE
------------------------------------------------------------------
``wholebody``'s ``state`` is the 29 joint angles the completion produced. That channel is the
measured OOD one at the robot: over the 159 valid episode starts of
``session_2026.09.02_merged3`` those joints sit ``0.2750 rad`` (mean) / ``0.6643 rad`` (worst)
from the pose the G1 holds after **A**, with **16 of 29** past the low level's 0.20 rad
pre-arm gate. ``JointClutch`` keeps that off the wire and ``--state-source command`` keeps it
out of the observation, but neither CLOSES it — the model still reads a posture space the
robot is not in.

So this family replaces the joints with what they MEAN: pelvis-frame Cartesian relations
between the two hands and the body, FK'd from the very same joints
(:mod:`fuji.s1_policy.data.wb_geom`, which owns the column order and the measured table)::

    rel_xy  4   [(p_L - p_R).x, (p_L - p_R).y, p_L.x, p_L.y]     metres, PELVIS frame
    rpy     6   extrinsic-XYZ Euler of R_L then R_R              radians, PELVIS frame

In those features the same 159 episode starts land 0.28-0.58 sigma from the ready pose on
every x/rotation channel, where the raw joints are 16-of-29 past the gate. The residue is the
two ``y`` channels at 2.3-2.5 sigma (~9 cm): the completion places the operator's hands wider
apart than the ready pose does, which is a PLACEMENT property of the export — recorded here,
not fixed here.

Four things make this a family of named MODES rather than a flag or an env pin:

* ``state_dim`` moves with the composition (31 / 12 / 8 / 6 / 2) and is what
  ``inference.common.validate_model_layout`` pins a checkpoint against. Spelling the
  composition in ``layout_mode`` — which every checkpoint records — means a crossed load is
  refused BY NAME rather than passing on a width coincidence, and it survives ``torchrun``
  handing the environment to 8 ranks (an env pin one rank missed is a disagreement about
  widths BEFORE the first collective, i.e. a hang).
* The ACTION is byte-identical across all five: same 43 columns, same joint block, same
  :data:`WB_JOINT_TARGETS` A/B, same ``dense`` low level. The 2x2 ablation over ``{rel_xy}`` x
  ``{rpy}`` is therefore a controlled experiment on the OBSERVATION alone.
* ``wholebody_geom_none`` (grippers only) is the degenerate cell and is included on purpose:
  it is the "does joint-space proprioception help at all, or is this task vision-only?"
  control the 2x2 grid needs to be interpretable.
* They need the SAME zarr. ``hands_flag`` is still ``both_head_dual`` and the joints still come
  from ``wb_state_joints``; the geometry is DERIVED at load time through the one shared
  ``KeypointFK``, so a lever or URDF correction propagates with no re-export. The cost is that
  a training box needs ``yourdfpy`` plus the two submodules that FK reads — refused loudly with
  the install hint, never silently zero-filled.

⚠️ ``wholebody`` and ``wholebody_geom_*`` are NOT cross-loadable and only ``layout_mode`` says
so: the action vector is identical, and ``wholebody_geom_xy_rpy``'s 12-dim state would load
into nothing else. Serving refuses by name already — ``run_hl_serving_wholebody.sh`` and
``run_infer_s1_wholebody.sh`` both gate on ``hl_data.mode == 'wholebody'``, so a geom
checkpoint is rejected at the launcher rather than at the robot.

The ``fivepoint_*`` family: keypoint ACTION, geometric STATE
-----------------------------------------------------------
``diffusion_umi_5pt``. The action is FIVE keypoint poses — the ``both_hands_head_dual`` triple
plus both feet, appended — and the state is the same pelvis-frame geometry the
``wholebody_geom_*`` family introduced. So it is the one family that draws from both worlds,
which is why ``is_wholebody`` (a claim about the ACTION) and :attr:`ActionLayout.
state_is_geometric` (a claim about the STATE) had to become two questions instead of one::

    [ 0:16]  L hand    [pos 3][rot6d 6][grip 1][force 6]   \\
    [16:32]  R hand    [pos 3][rot6d 6][grip 1][force 6]    | byte-identical to
    [32:41]  chest     [pos 3][rot6d 6]                    /  both_hands_head_dual
    [41:50]  L foot    [pos 3][rot6d 6]                    \\  the new pair
    [50:59]  R foot    [pos 3][rot6d 6]                    /

    mode                     state                                        state_dim
    fivepoint_geom_xy3_rpy   [rel_xy3 6][rpy 6][L grip][R grip]              14
    fivepoint_geom_xy3       [rel_xy3 6][L grip][R grip]                      8
    fivepoint_geom_rpy       [rpy 6][L grip][R grip]                          8
    fivepoint_geom_none      [L grip][R grip]                                 2

WHY THIS EXISTS, in one measurement. ``wholebody`` commands the 29 joints, so its output has to
land in a posture space the robot is in — and it does not: 0.2750 rad mean from the ready pose,
**16 of 29 joints past the low level's 0.20 rad pre-arm gate**. A 5-point wire is gated instead
on **3 of 5 keypoints at 70 mm** (the feet are the completion's inference and holosoma warns
rather than refuses on them), and a chunk decoded against the FK'd ready pose lands 0.7-3.1 mm
(hands) / 28.9-36.3 mm (chest) from that reference. Same task, same rig, same data — the gate
closes by construction on one and structurally cannot on the other. The redundant arm's null
space (0.40 rad = 23°) and the knee's 0.54 rad stand-up simply stop being commanded, because the
low level solves for them.

Three things about the FOOT pair, each load-bearing:

* **The feet are not tracked and never will be.** ``tracker_roles_required`` returns the same
  three roles as ``wholebody``, so ``layout.roles`` (5) and the tracker set (3) disagree — the
  opposite direction from ``wholebody``, where roles (2) were FEWER than trackers (3). Both
  disagreements exist for the same underlying reason and both are why nothing may read
  ``layout.roles`` to answer "whose glitch corrupts this sample".
* **The foot target is ``KeypointFK(wb_action_joints[t, k])``'s ankle pose**, written by
  ``scripts/add_foot_targets.py`` — a pure function of the joint target the ``wholebody`` preset
  already trains on, so the two families describe the SAME body and no second completion runs.
  It is PELVIS-frame with no root pose in it, which makes the channel a **stance** ("where the
  foot sits relative to the body") and not a footstep: a take where the operator walks will not
  produce a walking foot target. That is a deliberate trade and the measurement behind it is
  that the chest's net XY excursion is 37.9 mm median / 111.9 mm p90 over 167 episodes, with
  86% of episodes never leaving a 100 mm ball.
* **The per-chunk foot signal is small.** Completed feet move 1.9-4.4 mm per 30-frame window
  while the two export placements disagree by 2.3-3.2 mm on that same quantity, i.e. roughly
  half of a chunk-scale foot label is placement noise. The channel is worth predicting for the
  operator's YAW (10.7° median, 48.5° p90, which does re-place the feet), not for translation —
  read ``val/*`` per group before concluding the feet learned anything.

⚠️ NOT cross-loadable with ``both_hands_head_dual`` despite the action PREFIX, and the state is
where it breaks rather than the action: 59 vs 41 action columns will at least fail a shape
check, but nothing about a 14-dim geometric state resembles a 20-dim ``wrt_start_rot6d`` one.
``layout_mode`` is the pin, as for every other family. Serving refuses all four modes by name
today (no deploy builds this observation yet) — see ``docs/RUNBOOK_diffusion_umi_5pt.md``.

Observation layout
------------------
v2 deliberately does NOT feed absolute EE pose to the model. The action chunk is
expressed relative to the observation-time anchor, so the anchor itself carries no
information the image doesn't — and feeding it made the deployed model see
conditioning far outside its training range (training state was absolute
lighthouse-world metres; serving sent pelvis-frame). The observation is (with
``To == obs_history_len`` frames, oldest-first, spanning ``[t0 - (To-1)*Δ, t0]``):

    image           To camera frames, each (3, 224, 224)   (UMI-on-Air's 2-frame obs)
    state           To frames of, per keypoint,
                        [wrt_start_rot6d 6][gripper 1?]     (To, state_dim)
    force_history   8 causal anti-aliased wrench bins over [t0-1s, t0]
                    (8, 6 * n_force_hands)

Two borrowings from UMI-on-Air (DECISION-WRT0 / DECISION-OBS2, both default ON):

* **Two image frames** at ``t0`` and ``t0 - Δ`` (Δ ≈ 0.25 s), giving the model a
  short-horizon velocity cue the single frame lacked. Time-anchored, not
  frame-strided, because the camera grid is non-uniform on some sessions.
* **``wrt_start_rot6d`` in state** — the keypoint's current orientation relative to
  its EPISODE-START orientation. This is proprioception that is frame-INVARIANT to
  the world origin (so OOD-safe, unlike the absolute pose the pre-v2 policy fed),
  matching UMI's ``robot0_eef_rot_axis_angle_wrt_start`` (orientation only; UMI drops
  the translation there too).

``force_history`` is still a TIME SERIES with its own encoder (see
``fuji.s1_policy.model.force_encoder``) rather than being flattened into ``state``:
the temporal profile of a contact is the signal, and flattening a 2-hand history
into ``state`` would also overflow ``filled_state_dim`` and be silently truncated
by ``_pad_state``.

``wholebody``'s ``state`` is the one that is NOT ``wrt_start_rot6d``: it is the 29 joint
angles at ``t0`` plus the two observed gripper openings (31 dims)::

    [0:29]   joints    the SAME quantity, in the same units and order, the action predicts
    [29]     L gripper
    [30]     R gripper

i.e. genuine joint-space proprioception rather than an orientation channel — the "current
frame's joint value" a whole-body policy conditions on, and the same shape ``third_party/Psi0``
feeds (``pi0.py``: one ``nn.Linear(state)`` token; GR00T uses a 2-layer MLP). This build
already has both encodings, and ``diffusion_umi_wholebody`` selects the learned one with
``--diffusion.state-projection`` because 29 raw radians concatenated beside 768-dim ViT
features is a scale mismatch a single Linear absorbs and a zero-pad slot does not.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Fixed block widths. These are contract, not tunables: a change here changes
# every checkpoint's column meaning.
# ---------------------------------------------------------------------------
POS_DIM = 3
ROT6D_DIM = 6
GRIPPER_DIM = 1
FORCE_DIM = 6  # Fx Fy Fz Tx Ty Tz
HAND_BLOCK_DIM = POS_DIM + ROT6D_DIM + GRIPPER_DIM + FORCE_DIM  # 16
HEAD_BLOCK_DIM = POS_DIM + ROT6D_DIM  # 9

#: The G1's actuated joint count — the width of the ``wholebody`` mode's joint block, and of
#: the ``dense`` low level's command. Contract, not a tunable.
JOINT_DIM = 29
#: One hand's block in the ``wholebody`` layout: the SAME gripper + wrench a keypoint hand
#: block reserves, minus the pose (which the joints already express).
WB_HAND_BLOCK_DIM = GRIPPER_DIM + FORCE_DIM  # 7

#: The 29 joint angles' order, both in the ``wholebody`` action's ``[0:29]`` block and in its
#: ``state``. This is holosoma's ``g1_29dof.dof_names`` — the order ``qpos[7:36]`` /
#: ``dof_pos`` / ``motion_command_t`` all use, so the handoff to the ``dense`` low level is an
#: ASSIGNMENT and not a permutation.
#:
#: MIRRORED rather than imported, for the same reason
#: :data:`fuji.s1_policy.serving.live_robot.G1_29DOF_SDK_DOF_NAMES` mirrors it: the holosoma
#: config package is not importable from FUJI's ``.venv``, and ``data/`` importing from
#: ``serving/`` would invert the layering. The two copies are pinned equal by
#: ``tests/test_diffusion_umi_wholebody.py::test_joint_names_match_the_sdk_order`` — a silent
#: leg/arm swap still renders, so it has to be a test rather than a convention.
WB_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

#: What the ``wholebody`` action's 29 joint columns MEAN. Selected per training run by
#: ``HLDataConfig.wb_joint_target`` and baked into the checkpoint; the WIDTH is identical
#: either way, which is exactly why it has to be a named contract rather than a convention.
#:
#: ``absolute`` (default, and every checkpoint written before this knob existed)
#:     Radians in the export's own posture space (``attrs.wb_cond_mode``). Row 0 equals the
#:     ``state`` joints to 0.000e+00 by the exporter's construction.
#: ``delta``
#:     ``q[k] - q[0]``, i.e. anchor-relative in JOINT space, so row 0 is exactly zero and the
#:     chunk describes CHANGE from the observed configuration. The absolute posture is not
#:     thrown away — it stays in ``state`` (proprioception must be absolute, since a delta
#:     state relative to itself is identically zero) — it is removed from the TARGET.
#:
#: Two consequences make this a real A/B rather than a reparameterization:
#:
#: * The deploy-side ``JointClutch`` publishes ``base + (q[i] - q[0])``, which is INVARIANT to
#:   a constant offset on the chunk. So with the joint clutch armed (the default) the published
#:   wire is the same function of the model's output in both modes, and the export's
#:   posture-space gap to the robot's ready pose never reaches it either way. What changes is
#:   what the network has to represent: ``delta`` asks for the motion alone, ``absolute`` asks
#:   for the motion plus a posture offset the clutch then subtracts back out.
#: * ``--wb-clutch none`` (publish the chunk absolutely) is MEANINGLESS for ``delta`` — the
#:   values are near zero, i.e. the 29-joints-zero pose — so serving refuses that combination
#:   by name instead of collapsing the robot.
WB_JOINT_TARGETS: tuple[str, ...] = ("absolute", "delta")

#: What ONE whole-body ``state`` frame may carry, block name -> width, BEFORE the per-hand
#: gripper columns. The mode's ``state_geom`` tuple selects a subset in this order, so
#: ``state_dim`` and every slice in :meth:`ActionLayout.state_geom_slots` are derived from one
#: table. ``joints`` is read straight off ``wb_state_joints``; ``rel_xy`` / ``rpy`` are FK'd
#: from those same joints by :mod:`fuji.s1_policy.data.wb_geom`, which owns their COLUMN ORDER
#: (this table owns only the widths, and ``tests/test_wholebody_geom.py`` pins the two equal).
WB_STATE_BLOCK_DIM: dict[str, int] = {
    "joints": JOINT_DIM,
    "rel_xy": 4,
    "rel_xy3": 6,
    "rpy": 6,
}

# CONTRACT v3 (2026-08-07): NO HISTORY, 30 Hz, 50-step chunk. The DEFAULTS must mirror
# UMI_collection/processing/umi_convert_core.py::HLGrid — test_hl_contract_matches_converter
# asserts it, and hl_dataset._check_file rejects any file whose attrs disagree with the
# RESOLVED values (the converter re-pins its grid from --hl-* flags; this side re-pins the
# force-bin count from FUJI_FORCE_HIST_LEN, and the two must still agree per run).
#
# The observation is the SINGLE anchor frame t0: one camera image, one force bin, one
# gripper width. No look-back on either the image (OBS_HISTORY_LEN 2 -> 1) or the force
# (FORCE_HIST_LEN 8 -> 1). The force input is KEPT — the wiping task needs present
# contact — but it is now the single ACTION_DT-wide causal bin ending at t0, so the force
# the model sees is the same 33 ms average the first predicted step is scored against.
#
#: The build contract's bin count, and the value ``test_hl_contract_matches_converter``
#: pins against ``HLGrid``'s default. Read :data:`FORCE_HIST_LEN` instead of this — a
#: process may be pinned to another value (see :func:`_resolve_force_hist_len`).
FORCE_HIST_LEN_DEFAULT = 1

#: Env var that re-pins :data:`FORCE_HIST_LEN` for ONE process.
FORCE_HIST_LEN_ENV = "FUJI_FORCE_HIST_LEN"


def _resolve_force_hist_len() -> int:
    """Resolve the force-bin count once, at import, from the environment.

    An A/B over the force look-back (L=1 at 33 ms vs L=6 at 200 ms) needs two runs that
    disagree about this number, and editing the constant between them cannot express
    "two concurrent runs in one checkout" — the second run silently inherits the first
    run's edit, or worse, a serving process started meanwhile picks it up.

    It is an ENV VAR rather than a tyro flag on purpose. Every downstream consumer reads
    this value as a ``def``-time default (``HLDataset(glitch_pre_s=FORCE_HIST_SPAN_S)``,
    ``fit_hl_norm_stats``, ``hl_replay``, ``eval_checkpoints_physical``), and Python
    captures those at ``def`` time — the exact trap that made the converter's ``--hl-*``
    flags no-ops until its grid became one mutable object read at call time. A CLI flag
    parsed after import could therefore not reach them, whereas the environment is fixed
    BEFORE the first import, so the whole process agrees on one contract value. That
    per-process pin is the invariant those defaults already assume.

    Varying it is safe because three independent guards compare against the RESOLVED
    value rather than trusting it: ``hl_dataset._check_file`` rejects a zarr whose
    ``hl_force_hist_len`` attr disagrees, the checkpoint carries its own
    ``force_hist_len`` (so serving sizes its window from the model, not this build), and
    ``--resume-from`` does exact ``TrainConfig`` equality. A wrong pin is therefore a
    loud refusal at load time, not a silently mistrained model.
    """
    import os

    raw = os.environ.get(FORCE_HIST_LEN_ENV)
    if raw is None or not raw.strip():
        return FORCE_HIST_LEN_DEFAULT
    try:
        val = int(raw)
    except ValueError as exc:
        raise ValueError(f"{FORCE_HIST_LEN_ENV}={raw!r} is not an integer") from exc
    if val < 1:
        # 0 would mean "no force input at all", which is a LAYOUT change (the encoder and
        # the force action columns come and go with `force_hist_channels`), not a window
        # resize. Refuse rather than half-apply it.
        raise ValueError(f"{FORCE_HIST_LEN_ENV} must be >= 1, got {val}")
    return val


#: Number of causal force-history bins fed as INPUT, spanning ``FORCE_HIST_SPAN_S``.
#: 1 == t0 only. Each bin is one action step wide, so the span is ``L * ACTION_DT``.
FORCE_HIST_LEN = _resolve_force_hist_len()

#: Env var that drops the F/T sensor from EVERY mode for ONE process — the pure-position
#: ablation :func:`_resolve_force_hist_len` refuses to express as ``FUJI_FORCE_HIST_LEN=0``.
DROP_FORCE_ENV = "FUJI_DROP_FORCE"


def _resolve_drop_force() -> bool:
    """Resolve "this process has no F/T sensor" once, at import, from the environment.

    ``FUJI_FORCE_HIST_LEN=0`` is deliberately refused because shrinking the window and
    removing the sensor are different changes: with no sensor the force *encoder* is not
    built and the force *action columns* stop being supervised, so it is a change to the
    LAYOUT's declared hardware rather than to a window size. This flag is that change,
    kept separate so neither can be mistaken for the other.

    An env var for the same reason as :data:`FORCE_HIST_LEN`: every consumer reaches the
    layout through :func:`get_layout` at ``def``/call time, so a CLI flag parsed after
    import could not reach the dataset, the norm-stats fitter, or the metric groups, and
    the three would disagree about how many columns are real.

    Applied in :func:`get_layout` rather than by editing :data:`_V2_MODES`, so the
    mode -> ``--hands`` contract and the layout-vs-converter sensor agreement asserted by
    ``tests/test_hl_contract.py`` are untouched at the default. Those tests pin what the
    CONVERTER emits; this flag changes only what THIS process chooses to read.

    Safe to vary because the existing guard already catches a mismatch: a checkpoint
    records ``force_hist_channels``, and ``inference/common.validate_model_layout`` pins
    it against ``layout.force_hist_channels`` — so serving a force-free checkpoint
    without this flag (or vice versa) is a loud refusal, not a silent mis-feed.
    """
    import os

    raw = os.environ.get(DROP_FORCE_ENV)
    if raw is None or not raw.strip():
        return False
    val = raw.strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    if val in ("1", "true", "yes", "on"):
        return True
    raise ValueError(
        f"{DROP_FORCE_ENV}={raw!r} is not a boolean; use 1/0 (or true/false, yes/no, on/off)")


#: True when this process treats the rig as having NO F/T sensor: no force encoder, no
#: force token, and force action columns reserved-but-unsupervised. Pure-position ablation.
DROP_FORCE = _resolve_drop_force()

#: Env var for the FORCE-OUTPUT-ONLY ablation: the wrench leaves the OBSERVATION but the
#: force action columns stay supervised exactly as before.
DROP_FORCE_INPUT_ENV = "FUJI_DROP_FORCE_INPUT"


def _resolve_drop_force_input() -> bool:
    """Resolve "this process does not CONDITION on force" once, at import.

    Three switches touch force and they are deliberately separate, because they answer
    different questions and a run that confuses two of them is not comparable to anything:

    ==========================  =========  ==========  ====================================
    switch                      input      output      what it asks
    ==========================  =========  ==========  ====================================
    (default)                   6 ch       supervised  the shipped model
    ``FUJI_FORCE_HIST_LEN=L``   L bins     supervised  how much force HISTORY helps
    ``FUJI_DROP_FORCE_INPUT``   **none**   supervised  is force useful as an INPUT, or is it
                                                       predictable from vision + state?
    ``FUJI_DROP_FORCE``         none       masked      pure position: no F/T on the rig
    ==========================  =========  ==========  ====================================

    An env var rather than a CLI flag for the same reason as :data:`FORCE_HIST_LEN` and
    :data:`DROP_FORCE`: every consumer reaches the layout through :func:`get_layout` at
    ``def``/call time, so a flag parsed after import could not reach the model builder, and
    under ``torchrun`` every rank must resolve the same widths BEFORE the first collective.

    Unlike :data:`DROP_FORCE` this touches only :attr:`ActionLayout.force_input`, so
    ``force_roles`` — and with it ``measured_dims()``, the action mask, the metric groups and
    the dataset's emitted ``force_history`` width — is untouched. That is what makes the
    ablation a controlled one: the SAME norm-stats file, the SAME anchor set, and per-group
    validation metrics (including ``val/force``) directly comparable to the force-ful run.
    """
    import os

    raw = os.environ.get(DROP_FORCE_INPUT_ENV)
    if raw is None or not raw.strip():
        return False
    val = raw.strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    if val in ("1", "true", "yes", "on"):
        return True
    raise ValueError(
        f"{DROP_FORCE_INPUT_ENV}={raw!r} is not a boolean; use 1/0 (or true/false, yes/no, on/off)")


#: True when the model conditions on image + state only while still PREDICTING the force
#: columns. Implied by :data:`DROP_FORCE` (no sensor means no input either), so the two
#: compose rather than contradict.
DROP_FORCE_INPUT = _resolve_drop_force_input()
#: The build contract's step count, and the value ``test_window_geometry_constants_agree``
#: pins against ``HLGrid``'s default. Read :data:`ACTION_HORIZON` instead of this — a
#: process may be pinned to a shorter chunk (see :func:`_resolve_action_horizon`).
ACTION_HORIZON_DEFAULT = 50

#: Env var that re-pins :data:`ACTION_HORIZON` for ONE process.
ACTION_HORIZON_ENV = "FUJI_ACTION_HORIZON"


def _resolve_action_horizon() -> int:
    """Resolve the predicted step count once, at import, from the environment.

    An A/B over the chunk length (50 steps = 1.67 s vs 30 = 1.0 s) needs two runs that
    disagree about this number, and editing the constant between them cannot express "two
    concurrent runs in one checkout" — the second run silently inherits the first run's
    edit, or a serving process started meanwhile picks it up. Same reasoning, and the same
    mechanism, as :func:`_resolve_force_hist_len`; read its docstring for why this is an
    env var rather than a tyro flag (every consumer reads the value as a ``def``-time
    default, which a flag parsed after import cannot reach).

    **This re-pins the COUNT, never the RATE.** ``ACTION_HZ`` stays 30 Hz, so ``ACTION_DT``
    is untouched and step k still lands at ``t0 + k*ACTION_DT``: a shorter horizon is a
    strict PREFIX of a longer one on the same grid, not a resampling of it. That is what
    lets one zarr serve both runs — ``HLDataset`` truncates the stored chunk instead of
    needing a re-convert, and ``_check_file`` accepts ``stored >= wanted`` only while the
    stored file's own ``span/horizon`` equals this build's ``ACTION_DT`` to 1e-9.

    The pin may go ABOVE :data:`ACTION_HORIZON_DEFAULT` (H=100 / H=200 at 30 Hz = 3.33 s /
    6.67 s), and this used to be refused here. It is allowed now because the refusal was only
    ever about the ERROR MESSAGE, not about soundness: a longer horizon needs a dataset that
    stores it, which is exactly what ``_check_file`` asserts per file — with the file's name,
    its stored horizon, and both remedies (``--hl-action-horizon`` or lowering this pin). The
    default stays 50 so no existing dataset, checkpoint or launcher moves, and the prefix
    relation still means ONE 200-step conversion serves H=50/100/200 unchanged.

    ⚠️ A long horizon is not free of data consequences even when it loads: on
    ``session_2026.09.02_merged3`` (median episode 177 frames = 5.9 s) only 55.5% of anchors
    have a full 100-step future inside their episode and 23.5% a full 200-step one. The rest
    are TAIL-PADDED and masked, which is correct but means the late steps of a 200-step chunk
    are supervised by a quarter of the anchors. Read ``val/*`` per group before concluding a
    long chunk did not help.

    Safe to vary because nothing trusts the pin — the same three guards that make the
    force pin safe compare against the RESOLVED value: ``hl_dataset._check_file`` refuses a
    zarr whose grid cannot supply this many steps at this dt, the checkpoint records its
    own ``action_horizon`` and ``inference/common.validate_model_layout`` pins it
    unconditionally (so serving a 30-step model under an unpinned build is a loud refusal,
    not a silently truncated chunk), and ``--resume-from`` does exact ``TrainConfig``
    equality.
    """
    import os

    raw = os.environ.get(ACTION_HORIZON_ENV)
    if raw is None or not raw.strip():
        return ACTION_HORIZON_DEFAULT
    try:
        val = int(raw)
    except ValueError as exc:
        raise ValueError(f"{ACTION_HORIZON_ENV}={raw!r} is not an integer") from exc
    if val < 1:
        raise ValueError(f"{ACTION_HORIZON_ENV} must be >= 1, got {val}")
    return val


#: Number of predicted steps, spanning ``ACTION_SPAN_S`` at ``ACTION_HZ``.
ACTION_HORIZON = _resolve_action_horizon()
#: The uniform grid every model-facing quantity lives on. Now equal to the camera publish
#: rate (``CAMERA_PUBLISH_HZ=30``): measured camera dt 33.341 ms vs ACTION_DT 33.333 ms,
#: a 0.024% difference, so the obs grid and the action grid COINCIDE and "the recorded
#: sample nearest this step time" is exact without interpolation. Windows are still
#: defined in SECONDS and resampled, never as a count of camera frames — a dropped frame
#: or a session recorded at another rate must not silently rescale the window.
ACTION_HZ = 30.0
ACTION_DT = 1.0 / ACTION_HZ  # 0.03333 s
#: Spans are DERIVED from (count, rate) so a span literal can never drift from the rate
#: it implies (that drift is exactly how the v2 numbers became inconsistent).
#: 1.66667 s at the default H=50; moves with the pin (H=30 -> 1.0 s).
ACTION_SPAN_S = ACTION_HORIZON * ACTION_DT
#: 0.03333 s at the default L=1; moves with the pin (L=6 -> 0.2 s).
FORCE_HIST_SPAN_S = FORCE_HIST_LEN * ACTION_DT

#: Observation history (DECISION-OBS2, UMI-on-Air parity). The model sees this many
#: camera frames + state frames, oldest-first, at ``t0 - k*OBS_HISTORY_DT_S`` for
#: k=To-1..0. To == 2 was UMI's 2-frame obs; **To == 1 (v3) is the single-frame contract**
#: — the observation is t0 alone, matching FORCE_HIST_LEN == 1. With To == 1 the spacing
#: below is unused (there is no second frame to space), but it is kept as the value a
#: To > 1 run would use rather than deleted, so raising To stays a one-line change.
OBS_HISTORY_LEN = 1
#: Wall-clock spacing between consecutive obs frames. TIME-anchored (nearest camera
#: frame at ``t0 - k*dt``), not a fixed frame stride, because the camera grid is
#: non-uniform on some sessions — a fixed stride would be a variable time offset there.
OBS_HISTORY_DT_S = 0.25

#: Canonical keypoint order. Every mode's keypoint tuple is a subsequence of this.
#:
#: The two FEET are APPENDED, never interleaved, and that is the same structural promise
#: holosoma's ``VR_5POINT_BODIES`` makes on the wire: slots ``[0:3]`` of a 5-point vector are
#: byte-identically the 3-point vector, so a ``both_hands_head_dual`` checkpoint's action
#: columns are a strict PREFIX of a ``fivepoint_*`` one's. (They are still not
#: cross-loadable — ``validate_model_layout`` pins ``layout_mode`` — but the prefix is what
#: lets one wire builder serve both and one reader compare them.)
KEYPOINT_ORDER = ("left", "right", "head", "left_foot", "right_foot")

#: Keypoint roles that carry a POSE ONLY — no gripper, no force. The chest has no jaw and no
#: F/T; a foot has neither either, and unlike a hand there is no plausible future hardware
#: that would want those columns reserved (holosoma's ``VR_5POINT_FEET_BODIES`` carries not
#: even a LEVER: "a nonzero sole offset would price a point nothing else knows"). Declared as
#: a set so :func:`_build_v2` branches on the PROPERTY rather than on the string ``"head"``,
#: which is what let the feet reuse that path unchanged.
POSE_ONLY_ROLES: frozenset[str] = frozenset({"head", "left_foot", "right_foot"})

#: The two FOOT roles, in :data:`KEYPOINT_ORDER` order. The 5-point family's extra pair.
FOOT_ROLES: tuple[str, ...] = ("left_foot", "right_foot")

#: role -> zarr key triple (pos, rotvec, gripper_or_None) under ``/data``. The
#: converter's LEFT-canonical routing already writes robot0 = the PRIMARY hand and
#: stamps ``robot0_primary_role``; :func:`zarr_keys_for` resolves that indirection.
_HEAD_ZARR_KEYS = ("head_pos", "head_rot_axis_angle", None)

#: The FOOT roles' ANCHOR keys (pose at t0, camera grid) — ``FK(wb_state_joints[t])``.
#:
#: Every foot array is spelled ``wb_*`` and not ``hl_*``, deliberately: in a UMI zarr the
#: ``hl_`` prefix means "the converter's HL windowing wrote this", and nothing about a foot came
#: out of the converter. These are written by ``scripts/add_foot_targets.py`` from the
#: COMPLETION's joints, which is the same provenance every other ``wb_*`` array carries. The
#: cost of the honest name is one branch in ``HLDataset._resolve_keys``; the cost of the
#: convenient one would be a file that lies about where its labels came from.
#:
#: Both are PELVIS-frame: ``KeypointFK`` returns every body in the URDF's pelvis frame, and the
#: foot pose is a pure function of the joints with no root pose in it, so this channel commands
#: a STANCE ("where the foot sits relative to the body") and not a footstep. See
#: ``docs/RUNBOOK_diffusion_umi_5pt.md``.
_FOOT_ZARR_KEYS: dict[str, tuple[str, str, None]] = {
    "left_foot": ("wb_left_foot_pos", "wb_left_foot_rot_axis_angle", None),
    "right_foot": ("wb_right_foot_pos", "wb_right_foot_rot_axis_angle", None),
}

#: The FOOT roles' ACTION-CHUNK keys: ``(pose (T,H,6) = [pos 3][rotvec 3], valid (T,H))``.
#: Same ``(T,H,6)`` shape and same pos-then-axis-angle order as ``hl_<role>_action_pose``, so
#: ``HLDataset.__getitem__``'s per-role loop reads a foot exactly as it reads the chest.
FOOT_ACTION_KEYS: dict[str, tuple[str, str]] = {
    "left_foot": ("wb_left_foot_action_pose", "wb_left_foot_action_valid"),
    "right_foot": ("wb_right_foot_action_pose", "wb_right_foot_action_valid"),
}


# NOTE ON HASHABILITY: ``frozen=True`` generates ``__hash__`` from the fields, and two
# of them are ``slice`` objects — which only became hashable in Python 3.12. Hashing a
# KeypointSlots (e.g. ``{lay.slots(r) for r in roles}``) therefore works on 3.12+ and
# raises ``TypeError: unhashable type: 'slice'`` below it. The repo runs 3.13 and
# pyproject's floor was raised to 3.12 to match; EQUALITY (what the prefix checks and the
# tests use) works on every version regardless.
@dataclass(frozen=True)
class KeypointSlots:
    """Where one keypoint's quantities live inside an action step.

    ``gripper`` / ``force`` are None for a keypoint that has neither (the head).
    ``block`` is the contiguous span the keypoint owns; it is stored rather than
    inferred from which of gripper/force happen to be present, so a future
    gripper-without-force keypoint can't silently mis-report its own width.
    ``legacy32`` blocks are NOT contiguous, so ``block`` is None there.
    """

    name: str
    pos: slice
    rot6d: slice
    gripper: int | None
    force: slice | None
    block: slice | None = None

    @property
    def is_hand(self) -> bool:
        return self.name in ("left", "right")


@dataclass(frozen=True)
class ActionLayout:
    """Immutable description of one mode's action vector + observation widths."""

    # RESERVED vs DECLARED — the distinction that keeps checkpoints stable:
    #
    #   * The COLUMN layout (``keypoints`` / ``action_dim`` / ``*_dims()``) reserves a
    #     gripper and a force block for EVERY hand, whether or not that hand currently
    #     has the hardware. So bolting an F/T sensor onto the right hand later adds a
    #     data source without moving any column a checkpoint already learned.
    #   * ``force_roles`` / ``gripper_roles`` declare which hands PHYSICALLY have the
    #     sensor/gripper. They size the model's INPUTS (``force_history``, ``state``)
    #     and tell the dataset which columns carry a real measurement; a reserved but
    #     undeclared column stays zero and masked-False, so the loss never supervises
    #     it toward a fabricated constant.
    #
    # Read ``*_dims()`` when you mean columns; read ``*_roles`` when you mean hardware.
    mode: str
    keypoints: tuple[KeypointSlots, ...]
    action_dim: int
    #: Hands that PHYSICALLY carry an F/T sensor, in keypoint order. Sizes
    #: ``force_history``. Distinct from :meth:`force_dims` (reserved columns).
    force_roles: tuple[str, ...]
    #: Hands that PHYSICALLY carry a gripper, in keypoint order. Sizes ``state``.
    #: Distinct from :meth:`gripper_dims` (reserved columns).
    gripper_roles: tuple[str, ...]
    #: True for the frozen pre-v2 32-dim layout (padded, non-contiguous blocks).
    legacy: bool = False
    #: Does this process feed the measured wrench IN as an observation? False is the
    #: FORCE-OUTPUT-ONLY ablation (``FUJI_DROP_FORCE_INPUT``): the model predicts the force
    #: columns exactly as before but conditions on image + state alone. Distinct from
    #: ``force_roles=()`` (``FUJI_DROP_FORCE``), which removes force from BOTH sides.
    #:
    #: Deliberately a field and not just a module flag, so a test can build the ablated
    #: layout directly and so a layout object always describes itself. Defaulted True, and
    #: last, so every existing keyword construction and equality comparison is unchanged.
    force_input: bool = True
    #: The 29 absolute JOINT-ANGLE columns, for the ``wholebody`` mode; ``None`` for every
    #: keypoint mode. Defaulted None and last so no existing construction or equality
    #: comparison changes. Read :attr:`is_wholebody` for the branch, :meth:`joint_dims` for
    #: the columns.
    joints: slice | None = None
    #: Which :data:`WB_STATE_BLOCK_DIM` blocks ONE ``state`` frame carries, in order, before
    #: the per-hand gripper columns. ``()`` for every keypoint mode (their state is
    #: ``wrt_start_rot6d`` per keypoint). ``("joints",)`` is the original ``wholebody``;
    #: ``("rel_xy", "rpy")`` and its subsets are the ``wholebody_geom_*`` family. Defaulted
    #: and last so no existing construction or equality comparison changes.
    state_geom: tuple[str, ...] = ()
    #: Whether ONE state frame has the GEOMETRIC SHAPE ``[blocks][per-hand gripper]`` at all,
    #: as opposed to v2's ``[wrt_start_rot6d][gripper]`` per keypoint.
    #:
    #: Separate from :attr:`state_geom` being non-empty because ``()`` is ambiguous and the two
    #: readings differ by 30 columns: ``fivepoint_geom_none`` selects ZERO geometric blocks and
    #: its state is the two jaws (2), while ``both_hands_head_dual`` has no geometric state and
    #: its state is five rot6d plus the jaws (32). Read :attr:`state_is_geometric`, never this
    #: field directly, and note that :meth:`__post_init__` makes disagreement with
    #: :attr:`is_wholebody` impossible rather than trusting the builders to keep them in step.
    state_geometric: bool = False

    def __post_init__(self) -> None:
        # A whole-body action ALWAYS has a geometric state (joint space is one), so a builder
        # that set `joints` and forgot this flag would silently produce a 30-column-too-wide
        # state that still trains. Checked here rather than in each builder: this is the one
        # place every construction passes through, including `dataclasses.replace`.
        if self.joints is not None and not self.state_geometric:
            raise ValueError(
                f"layout {self.mode!r} has a joint action block but state_geometric=False; a "
                "whole-body state is joint space or its reduction, never wrt_start_rot6d")
        if self.state_geom and not self.state_geometric:
            raise ValueError(
                f"layout {self.mode!r} selects geometric state blocks {self.state_geom} but "
                "state_geometric=False, so state_dim would be computed the v2 way and every "
                "block would be written outside the vector it was sized for")

    @property
    def is_wholebody(self) -> bool:
        """True for the mode whose action is a robot CONFIGURATION, not keypoint poses.

        The one branch consumers need: a wholebody layout has no pose columns
        (``pose_dims() == ()``), no anchor-relative transform, no head keypoint, and a
        joint-space ``state``. Derived from the joint block's presence rather than compared
        against the mode STRING, so a second whole-body mode (a different joint count, a
        one-armed variant) gets every branch for free.

        ⚠️ This is a claim about the ACTION only. It used to double as "the state is
        geometric", which was true while ``wholebody*`` were the only modes with a
        ``state_geom`` — read :attr:`state_is_geometric` for that question. The
        ``fivepoint_*`` family is the counterexample the split exists for: keypoint POSES as
        the action, a pelvis-frame GEOMETRIC reduction as the observation."""
        return self.joints is not None

    @property
    def state_is_geometric(self) -> bool:
        """True when ONE state frame is ``[geometric blocks][per-hand gripper]`` rather than
        the v2 ``[wrt_start_rot6d][gripper]`` per keypoint.

        The two questions :attr:`is_wholebody` used to answer at once. Every whole-body mode is
        geometric (``wholebody_geom_none``'s state is grippers ALONE, which is still the
        geometric shape with zero blocks selected — hence the ``or``, not a ``bool(state_geom)``
        test), and so is the ``fivepoint_*`` family, whose action is nonetheless keypoint poses.

        No mode mixes the two: a state that carried both ``wrt_start_rot6d`` and a pelvis-frame
        geometry would be describing the same orientations twice in two conventions, one of
        which (rot6d, world-anchored to the episode start) is the very channel the geometry was
        introduced to replace."""
        return self.state_geometric

    # -- lookups -----------------------------------------------------------
    def slots(self, role: str) -> KeypointSlots:
        for kp in self.keypoints:
            if kp.name == role:
                return kp
        raise KeyError(f"layout {self.mode!r} has no keypoint {role!r} "
                       f"(present: {[k.name for k in self.keypoints]})")

    def has(self, role: str) -> bool:
        return any(kp.name == role for kp in self.keypoints)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(kp.name for kp in self.keypoints)

    @property
    def state_dim(self) -> int:
        """Width of ONE frame of the observation ``state`` vector.

        v2 (DECISION-WRT0, UMI-on-Air parity) carries, per keypoint in keypoint order,
        a contiguous block of ``[wrt_start_rot6d 6][gripper 1?]``:

        * ``wrt_start_rot6d`` (every keypoint): the keypoint's CURRENT orientation
          expressed relative to its EPISODE-START orientation (``R_start^T R_now`` as
          rot6d). Translation is deliberately dropped (position lives in the image + the
          anchor-relative action), exactly as UMI's ``_eef_rot_axis_angle_wrt_start``.
          It is frame-INVARIANT to the world origin, so unlike absolute pose it stays
          in-distribution at serving.
        * ``gripper`` (gripper roles only): the observed opening at the frame, [0, 1].

        Prefix-compatible with the action layout: ``left_only``[7] ⊂ ``both_hands``[13]
        ⊂ ``both_hands_head``[19], so a smaller-mode checkpoint's state columns are a
        strict prefix of a larger one. The observation carries ``obs_history_len``
        frames of this, so the model sees ``(To, state_dim)``.

        ⚠️ ``both_hands_head_dual``[20] is the ONE mode that breaks that chain, because
        declaring a SECOND gripper INSERTS a column mid-vector rather than appending one:
        the head's ``wrt_start_rot6d`` moves from ``[13:19]`` to ``[14:20]``. The action
        layout is unaffected (41 dims either way — the columns were already reserved), so
        the two modes' state vectors are the same WIDTH class but not the same MEANING.
        Only ``layout_mode`` distinguishes them, which is why it is baked into every
        checkpoint and pinned by ``validate_model_layout``.

        ⚠️ No whole-body mode carries ``wrt_start_rot6d`` at all. ``wholebody``[31] is
        ``[joints 29][L grip 1][R grip 1]``, joint-space proprioception, and the
        ``wholebody_geom_*`` family swaps that 29-wide block for a 0/4/6/10-wide GEOMETRIC one
        (see the module docstring) — 31 / 12 / 8 / 6 / 2. None of the five is comparable to any
        keypoint mode's state, nor to each other, and the same ``layout_mode`` pin is what
        refuses every swap.

        ⚠️ Nor is the ``fivepoint_*`` family comparable to any of them. Its ACTION is a
        prefix-extension of ``both_hands_head_dual``'s (41 -> 59), but its STATE is the
        geometric shape — 14 for ``fivepoint_geom_xy3_rpy`` — so those two share a keypoint
        action layout and NOT ONE state column. Same pin, same refusal."""
        if self.state_is_geometric:
            return (sum(WB_STATE_BLOCK_DIM[b] for b in self.state_geom)
                    + sum(GRIPPER_DIM for kp in self.keypoints
                          if kp.name in self.gripper_roles))
        return sum(ROT6D_DIM + (GRIPPER_DIM if kp.name in self.gripper_roles else 0)
                   for kp in self.keypoints)

    def state_geom_slots(self) -> dict[str, slice]:
        """block name -> its slice into ONE whole-body state frame; ``{}`` for keypoint modes.

        The blocks are laid out contiguously in :attr:`state_geom` order and BEFORE the
        grippers, so a consumer that wants the whole geometric part can take
        ``slice(0, state_dim - n_grippers)``. Absent blocks have NO entry rather than a 0-width
        slice: a consumer asking for ``slots["rpy"]`` on a mode without it raises ``KeyError``
        on the name it expected instead of silently writing nothing (the same reason
        :meth:`state_slots` omits ``wrt_start_rot6d`` for these modes).
        """
        out: dict[str, slice] = {}
        cur = 0
        for block in self.state_geom:
            width = WB_STATE_BLOCK_DIM[block]
            out[block] = slice(cur, cur + width)
            cur += width
        return out

    @property
    def joint_state_slot(self) -> slice | None:
        """Where the 29 JOINT angles sit in one state frame; None when the mode has no such
        block (every keypoint mode, and the whole ``wholebody_geom_*`` family).

        A property rather than an entry in :meth:`state_slots` because it belongs to no
        keypoint: keying it under a role would put the whole body's configuration inside one
        hand's block, and every existing consumer iterates ``state_slots()`` per role.
        """
        return self.state_geom_slots().get("joints")

    def state_slots(self) -> dict[str, dict]:
        """role -> ``{"wrt_start_rot6d": slice, "gripper": int?}`` into ONE state frame.

        Contiguous per-keypoint blocks in keypoint order, matching :attr:`state_dim`.
        The dataset fills these; the model consumes the flat vector. Kept here (not
        re-derived in the dataset) so state and action layouts can never desynchronize.

        For every mode whose state :attr:`state_is_geometric` — every ``wholebody*`` AND every
        ``fivepoint_*`` — the ``wrt_start_rot6d`` key is ABSENT (their state is joint space or
        its geometric reduction; see :meth:`state_geom_slots`) and the per-hand blocks
        are the gripper alone, laid out AFTER the geometric blocks. Omitting the key rather
        than emitting an empty slice is deliberate: a consumer that has not been taught these
        modes raises ``KeyError`` on the name it expected instead of silently writing a 0-width
        orientation.
        """
        out: dict[str, dict] = {}
        cur = 0
        if self.state_is_geometric:
            cur = sum(WB_STATE_BLOCK_DIM[b] for b in self.state_geom)
        for kp in self.keypoints:
            entry: dict = {}
            if not self.state_is_geometric:
                entry["wrt_start_rot6d"] = slice(cur, cur + ROT6D_DIM)
                cur += ROT6D_DIM
            if kp.name in self.gripper_roles:
                entry["gripper"] = cur
                cur += GRIPPER_DIM
            out[kp.name] = entry
        return out

    @property
    def force_hist_channels(self) -> int:
        """Channel width of ``force_history`` AS THE DATASET EMITS IT.

        Shape is ``(FORCE_HIST_LEN, this)``. Driven by declared hardware alone, so it is
        unaffected by :attr:`force_input` — the force-output-only ablation changes what the
        MODEL consumes, not what the recording contains. Keeping the dataset side fixed is
        what lets that ablation reuse the force-ful run's norm-stats file byte for byte and
        train on a bit-identical anchor set (see :attr:`force_input_channels`).
        """
        return len(self.force_roles) * FORCE_DIM

    @property
    def force_input_channels(self) -> int:
        """Channel width of the force window the MODEL is built to consume — 0 when force is
        input-masked.

        This is the value ``TrainConfig._post_init_hl`` pushes into the model config and that
        ``inference.common.validate_model_layout`` pins a checkpoint against, so a model built
        without a force encoder can never be handed a process that thinks it has one (or the
        reverse) without a loud refusal.

        Read THIS when sizing a model input; read :attr:`force_hist_channels` when sizing a
        dataset array or fitting normalization statistics. At the default they are equal, so
        every existing call site keeps its meaning.
        """
        return self.force_hist_channels if self.force_input else 0

    # -- derived index sets ------------------------------------------------
    def pose_dims(self) -> tuple[int, ...]:
        """Dims that undergo the anchor-relative SE(3) transform (pos + rot6d).

        Everything NOT in here (gripper, force, pad) is frame-invariant and must
        pass through un-relativized — both in the dataset and in serving's
        un-relativizer. Callers should derive their column sets from this rather
        than re-deriving "the first 9 columns", which is what silently left the
        right hand and head unsmoothed in the pre-v2 action smoother.

        EMPTY for ``wholebody``: joint angles are already body-fixed, so that mode's action is
        absolute and nothing in it is relativized. ⚠️ A consumer that reduces over these dims
        must handle the empty set explicitly — ``mask[..., []].all(axis=-1)`` is ``True``
        everywhere and a norm over 0 columns is 0.0, so the naive shape reports a PERFECT
        position error for a mode that has no position at all.
        """
        out: list[int] = []
        for kp in self.keypoints:
            out.extend(range(kp.pos.start, kp.pos.stop))
            out.extend(range(kp.rot6d.start, kp.rot6d.stop))
        return tuple(sorted(out))

    def joint_dims(self) -> tuple[int, ...]:
        """The 29 absolute JOINT-ANGLE columns (``wholebody`` only; ``()`` elsewhere).

        The whole-body counterpart of :meth:`pose_dims`: these are the columns the ``dense``
        low level publishes, and the ones a metric must report in DEGREES rather than as a
        Cartesian distance."""
        if self.joints is None:
            return ()
        return tuple(range(self.joints.start, self.joints.stop))

    def force_dims(self) -> tuple[int, ...]:
        """All RESERVED force columns, including hands with no physical sensor.

        Use this for column bookkeeping (metric groups, mask coverage). For "which
        hands actually measure force", use :attr:`force_roles`.
        """
        out: list[int] = []
        for kp in self.keypoints:
            if kp.force is not None:
                out.extend(range(kp.force.start, kp.force.stop))
        return tuple(sorted(out))

    def gripper_dims(self) -> tuple[int, ...]:
        """All RESERVED gripper columns. See :meth:`force_dims` on reserved vs declared."""
        return tuple(sorted(kp.gripper for kp in self.keypoints if kp.gripper is not None))

    def measured_dims(self) -> tuple[int, ...]:
        """Columns backed by real hardware on THIS rig: poses, plus gripper/force only
        for the declared roles. The complement (within ``action_dim``) is the set of
        reserved-but-unmeasured columns that must stay masked-False."""
        out = list(self.pose_dims()) + list(self.joint_dims())
        for kp in self.keypoints:
            if kp.gripper is not None and kp.name in self.gripper_roles:
                out.append(kp.gripper)
            if kp.force is not None and kp.name in self.force_roles:
                out.extend(range(kp.force.start, kp.force.stop))
        return tuple(sorted(out))

    def unmeasured_dims(self) -> tuple[int, ...]:
        """Reserved columns with no hardware behind them (masked-False, never trained)."""
        measured = set(self.measured_dims())
        return tuple(d for d in range(self.action_dim) if d not in measured)

    def pad_dims(self) -> tuple[int, ...]:
        """Dims belonging to no keypoint (only nonempty for ``legacy32``)."""
        claimed = (set(self.pose_dims()) | set(self.joint_dims())
                   | set(self.force_dims()) | set(self.gripper_dims()))
        return tuple(d for d in range(self.action_dim) if d not in claimed)

    def metric_group_dims(self) -> dict[str, tuple[int, ...]]:
        """Index sets for per-group training METRICS (logged, never optimized).

        Two overlapping partitions of the same real dims: by keypoint
        (left/right/head) and by quantity (pos/rot/grip/force). Groups that are
        empty in this mode are omitted entirely, so a left-only run logs no
        ``flow_loss/right``. Pad dims belong to no group.

        ``wholebody`` adds a ``joints`` group and has no ``pos``/``rot``, so the SAME
        omit-empty rule keeps the log honest: a whole-body run logs ``joints`` / ``grip`` /
        ``force`` and no position group it could not have learned.
        """
        groups: dict[str, tuple[int, ...]] = {}
        for kp in self.keypoints:
            dims = sorted(
                list(range(kp.pos.start, kp.pos.stop))
                + list(range(kp.rot6d.start, kp.rot6d.stop))
                + ([kp.gripper] if kp.gripper is not None else [])
                + (list(range(kp.force.start, kp.force.stop)) if kp.force is not None else [])
            )
            groups[kp.name] = tuple(dims)
        for name, dims in (
            ("pos", tuple(d for kp in self.keypoints for d in range(kp.pos.start, kp.pos.stop))),
            ("rot", tuple(d for kp in self.keypoints for d in range(kp.rot6d.start, kp.rot6d.stop))),
            ("joints", self.joint_dims()),
            ("grip", self.gripper_dims()),
            ("force", self.force_dims()),
        ):
            if dims:
                groups[name] = tuple(sorted(dims))
        return groups

    def zarr_keys(self, primary_role: str) -> dict[str, tuple[str, str, str | None]]:
        """role -> (pos_key, rotvec_key, gripper_key_or_None) for this layout.

        Resolves the converter's robot0/robot1 indirection: ``robot0_*`` is the
        PRIMARY hand named by the zarr's ``robot0_primary_role`` attr, ``robot1_*``
        is the other hand. Only roles present in this layout are returned, so a
        left-only dataset over a 3pt zarr simply never touches robot1/head.
        """
        return {role: keys for role, keys in zarr_keys_for(primary_role).items()
                if self.has(role)}

    def force_zarr_prefix(self, primary_role: str, role: str) -> str:
        """Zarr array-name prefix carrying ``role``'s F/T (``robot0`` or ``robot1``)."""
        if role not in self.force_roles:
            raise KeyError(f"layout {self.mode!r} has no force for role {role!r}")
        return "robot0" if role == primary_role else "robot1"


def zarr_keys_for(primary_role: str) -> dict[str, tuple[str, str, str | None]]:
    """Full role -> zarr-key-triple map under a given ``robot0_primary_role``.

    ``primary_role`` comes from the zarr root attr; the converter stamps 'left' for
    every modern (1pt/2pt/3pt) session. Legacy zarrs predate the attr and stored
    their single hand as the RIGHT hand, so callers default to 'right' there.
    """
    if primary_role not in ("left", "right"):
        raise ValueError(f"primary_role must be 'left' or 'right', got {primary_role!r}")
    other = "right" if primary_role == "left" else "left"
    return {
        primary_role: ("robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width"),
        other: ("robot1_eef_pos", "robot1_eef_rot_axis_angle", "robot1_gripper_width"),
        "head": _HEAD_ZARR_KEYS,
        # Present for EVERY primary_role, because a mode that does not declare the foot roles
        # never looks them up and a mode that does needs them regardless of which hand is
        # robot0. A dataset without the arrays fails in `_check_file` BY NAME instead of here.
        **_FOOT_ZARR_KEYS,
    }


# ---------------------------------------------------------------------------
# Mode construction
# ---------------------------------------------------------------------------
def _build_v2(mode: str, roles: tuple[str, ...], force_roles: tuple[str, ...],
              gripper_roles: tuple[str, ...], *, force_input: bool = True) -> ActionLayout:
    """Lay out ``roles`` as contiguous blocks in the order GIVEN.

    No sort is performed: ``roles`` comes from :data:`_V2_MODES`, where every tuple is
    already written in :data:`KEYPOINT_ORDER`. Sorting here would silently permute a
    checkpoint's columns if a mode were ever declared out of order, so the invariant is
    kept where it is declared and asserted by
    ``tests/test_layout.py::test_mode_role_tuples_are_in_keypoint_order``.

    ``force_roles`` / ``gripper_roles`` declare which roles have the PHYSICAL sensor
    or gripper. Every hand block still RESERVES both columns (so adding hardware later
    moves no checkpoint column) — the declared sets only control which columns carry a
    real measurement and are therefore supervised.
    """
    slots: list[KeypointSlots] = []
    cur = 0
    for role in roles:
        if role in POSE_ONLY_ROLES:
            # The chest AND both feet: 9 columns of pose, nothing reserved. Branching on the
            # PROPERTY (:data:`POSE_ONLY_ROLES`) rather than on the literal ``"head"`` is what
            # let the 5-point family reuse this path with no new block arithmetic.
            slots.append(KeypointSlots(
                name=role,
                pos=slice(cur, cur + POS_DIM),
                rot6d=slice(cur + POS_DIM, cur + POS_DIM + ROT6D_DIM),
                gripper=None,
                force=None,
                block=slice(cur, cur + HEAD_BLOCK_DIM),
            ))
            cur += HEAD_BLOCK_DIM
            continue
        # A hand always reserves its gripper AND force columns, even if a given
        # dataset lacks the stream: the per-sample validity mask (not the layout)
        # decides whether the loss supervises them, so the column meaning is
        # stable across datasets and a force-less take can share a checkpoint.
        pos = slice(cur, cur + POS_DIM)
        rot = slice(cur + POS_DIM, cur + POS_DIM + ROT6D_DIM)
        grip = cur + POS_DIM + ROT6D_DIM
        force = slice(grip + GRIPPER_DIM, grip + GRIPPER_DIM + FORCE_DIM)
        slots.append(KeypointSlots(name=role, pos=pos, rot6d=rot, gripper=grip, force=force,
                                   block=slice(cur, cur + HAND_BLOCK_DIM)))
        cur += HAND_BLOCK_DIM
    return ActionLayout(
        mode=mode,
        keypoints=tuple(slots),
        action_dim=cur,
        force_roles=tuple(r for r in roles if r in force_roles),
        gripper_roles=tuple(r for r in roles if r in gripper_roles),
        force_input=force_input,
    )


def _build_wholebody(mode: str, *, state_geom: tuple[str, ...] = ("joints",),
                     force_input: bool = True,
                     force_roles: tuple[str, ...] = ("left", "right")) -> ActionLayout:
    """Lay out the 29 joint angles, then one ``[grip 1][force 6]`` block per hand.

    ``state_geom`` names the OBSERVATION's blocks and touches no action column — that is what
    makes the ``wholebody_geom_*`` family a state-only variation of one 43-dim action vector.

    Joints FIRST and the hands after, rather than interleaved, for the same reason the
    keypoint modes keep a hand's quantities contiguous: the block a downstream stage consumes
    whole should be sliceable with one ``[:JOINT_DIM]``. ``DenseWire`` publishes exactly that
    slice, and the two hands' grippers/wrenches leave on other channels entirely.

    Every hand still RESERVES both a gripper and a force column, so this mode inherits the
    reserved-vs-declared discipline unchanged — a rig that later loses one Bota keeps every
    column's meaning and only ``force_roles`` shrinks.
    """
    unknown = [b for b in state_geom if b not in WB_STATE_BLOCK_DIM]
    if unknown:
        raise ValueError(f"unknown whole-body state block(s) {unknown}; "
                         f"expected a subset of {sorted(WB_STATE_BLOCK_DIM)}")
    if len(set(state_geom)) != len(state_geom):
        # A duplicate would widen `state_dim` while `state_geom_slots` returned one slice per
        # NAME, so the second copy would be reserved-but-never-written — invisible in shapes.
        raise ValueError(f"state_geom has duplicates: {state_geom}")
    slots: list[KeypointSlots] = []
    cur = JOINT_DIM
    for role in ("left", "right"):
        grip = cur
        force = slice(cur + GRIPPER_DIM, cur + GRIPPER_DIM + FORCE_DIM)
        slots.append(KeypointSlots(
            name=role,
            # EMPTY pose slices, not None: `pose_dims()` then reports () by construction
            # instead of every consumer needing an `is None` branch, and the one place where
            # "no columns" is dangerous (a metric reducing over them) is called out in
            # `pose_dims`' own docstring and handled in metrics.py.
            pos=slice(cur, cur),
            rot6d=slice(cur, cur),
            gripper=grip,
            force=force,
            block=slice(cur, cur + WB_HAND_BLOCK_DIM),
        ))
        cur += WB_HAND_BLOCK_DIM
    return ActionLayout(
        mode=mode,
        keypoints=tuple(slots),
        action_dim=cur,
        force_roles=tuple(r for r in ("left", "right") if r in force_roles),
        gripper_roles=("left", "right"),
        force_input=force_input,
        joints=slice(0, JOINT_DIM),
        state_geom=tuple(state_geom),
        state_geometric=True,
    )


def _build_legacy32() -> ActionLayout:
    """The frozen pre-v2 layout: 29 real + 3 pad, no force, non-contiguous blocks.

    Byte-identical to the slot table in ``umi_dataset.py`` so the already-trained
    checkpoint keeps its column meaning. Kept read-only; never extend it.
    """
    return ActionLayout(
        mode="legacy32",
        keypoints=(
            KeypointSlots("left", slice(0, 3), slice(3, 9), 9, None),
            KeypointSlots("right", slice(10, 13), slice(13, 19), 19, None),
            KeypointSlots("head", slice(20, 23), slice(23, 29), None, None),
        ),
        action_dim=32,
        force_roles=(),
        gripper_roles=("left", "right"),
        legacy=True,
    )


#: mode -> (roles, roles_with_an_FT_sensor, roles_with_a_gripper).
#:
#: The FOUR ORIGINAL MODES ASSUME ONE Bota F/T AND ONE GRIPPER, BOTH ON THE LEFT
#: (PRIMARY) HAND — the rig as it was through 2026-08-23, and the hardware every shipped
#: HL-v2 checkpoint was trained against. So in those bimanual modes only 'left' declares
#: force and gripper; the right hand's reserved columns stay zero and masked-False.
#:
#: The rig HAS had a second Bota + a second DM4310 since 2026-08-25, and
#: ``session_2026.08.30_14.11.49`` is the first session recorded with both. Rather than
#: widen ``both_hands_head`` in place — which would change its ``state_dim`` and
#: ``force_hist_channels`` and so invalidate every checkpoint trained on it — the dual
#: hardware is a NEW mode, ``both_hands_head_dual``. Adding a mode moves no column in any
#: existing one, which is the property the reserved-vs-declared split buys.
#:
#: ``right_only`` declares both on 'right' on the assumption that a right-only session
#: means the hardware was MOVED to the right hand (there is no second sensor to use).
#: If you instead record right-hand poses with the left-mounted hardware, build that
#: mode with empty force/gripper tuples rather than mislabelling the source.
_V2_MODES: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    "left_only": (("left",), ("left",), ("left",)),
    "right_only": (("right",), ("right",), ("right",)),
    "both_hands": (("left", "right"), ("left",), ("left",)),
    "both_hands_head": (("left", "right", "head"), ("left",), ("left",)),
    "both_hands_head_dual": (("left", "right", "head"),
                             ("left", "right"), ("left", "right")),
}

#: The WHOLE-BODY modes: action = a robot CONFIGURATION, not keypoint poses. Kept in their own
#: table rather than as a fifth entry in :data:`_V2_MODES`, because that table's value tuple is
#: ``(keypoint roles, force roles, gripper roles)`` and a whole-body mode's ``roles`` are only
#: the two hand BLOCKS — putting it there would make ``tracker_roles_required`` return two
#: roles for a mode that needs three trackers, which is exactly the mistake that drops the
#: chest puck's glitch mask.
#:
#: value = ``(the TRACKERS the source recording must carry, the state's geometric blocks)``.
#: The trackers are all three for every entry (the completion conditions on both hands AND the
#: chest), which is also what ``hands_flag`` maps to ``--hands both_head_dual``. The second
#: element is the ONLY difference between the five modes — see "The ``wholebody_geom_*``
#: family" in the module docstring for why the composition is spelled in the mode NAME rather
#: than set by a flag or an env pin.
_WB_TRACKERS: tuple[str, ...] = ("left", "right", "head")
_WB_MODES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # The shipped mode: 29 joint angles as proprioception.        state_dim 31
    "wholebody": (_WB_TRACKERS, ("joints",)),
    # The geometric family. Same 43-dim action, same zarr, same low level; the state is a
    # pelvis-frame reduction of the SAME joints, measured to be far closer to the robot's own
    # ready pose (module docstring). The 2x2 ablation grid, most informative first:
    "wholebody_geom_xy_rpy": (_WB_TRACKERS, ("rel_xy", "rpy")),   # state_dim 12
    "wholebody_geom_xy": (_WB_TRACKERS, ("rel_xy",)),             # state_dim  6
    "wholebody_geom_rpy": (_WB_TRACKERS, ("rpy",)),               # state_dim  8
    "wholebody_geom_none": (_WB_TRACKERS, ()),                    # state_dim  2 (grippers)
}

#: The FIVE-POINT modes: keypoint POSES as the action (like ``_V2_MODES``), a pelvis-frame
#: GEOMETRY as the observation (like ``wholebody_geom_*``). Its own table for the same reason
#: ``_WB_MODES`` has one — the value tuple carries a different thing — and because this is the
#: one family where "what the action is" and "what the state is" come from different worlds.
#:
#: value = ``(the state's geometric blocks,)``. The keypoint roles are always all FIVE, the
#: force/gripper roles always both hands, and the TRACKERS always the same three: the two feet
#: are not tracked by anything and never will be (three pucks, no foot hardware), so they are
#: absent from :func:`tracker_roles_required` while being present in the ACTION — which is the
#: whole reason the foot columns need their own provenance note.
#:
#: ⚠️ THE FEET COME FROM ``scripts/add_foot_targets.py``, NOT FROM THE CONVERTER: they are
#: ``KeypointFK(wb_action_joints[t, k])``'s ankle poses, i.e. a pure function of the joint
#: target the ``wholebody`` preset already trains on. So a ``fivepoint_*`` dataset is a
#: ``wholebody`` dataset plus four arrays, ``attrs.hands_mode`` still reads ``both_head_dual``,
#: and ``attrs.wb_foot_targets`` is the discriminator ``HLDataset`` checks for this family.
_5PT_ROLES: tuple[str, ...] = ("left", "right", "head", "left_foot", "right_foot")
_5PT_MODES: dict[str, tuple[tuple[str, ...]]] = {
    # The mode this family was built for: the three pairwise hand/chest xy differences plus
    # both hands' pelvis-frame RPY.                                        state_dim 14
    "fivepoint_geom_xy3_rpy": (("rel_xy3", "rpy"),),
    # The two one-block controls, for the same reason the wholebody grid has them: they say
    # WHICH half of the observation is carrying the run.
    "fivepoint_geom_xy3": (("rel_xy3",),),                              # state_dim  8
    "fivepoint_geom_rpy": (("rpy",),),                                  # state_dim  8
    # No proprioception at all beyond the two jaws — the floor every other cell is measured
    # against, and the only honest way to ask "does the geometry help".  state_dim  2
    "fivepoint_geom_none": ((),),
}

MODES = tuple(_V2_MODES) + tuple(_WB_MODES) + tuple(_5PT_MODES) + ("legacy32",)
DEFAULT_MODE = "left_only"

#: layout mode -> the converter's ``--hands`` value. The two vocabularies differ (the
#: layout names describe the ACTION VECTOR, the converter's describe which TRACKERS to
#: read), so the mapping is declared here rather than derived by string surgery — a
#: ``.replace()`` chain silently produces a plausible-but-wrong flag the moment a mode is
#: renamed. Pinned against the converter by
#: tests/test_hl_contract.py::test_hands_modes_match_layout_modes.
HANDS_FLAG_BY_MODE: dict[str, str] = {
    "left_only": "left",
    "right_only": "right",
    "both_hands": "both",
    "both_hands_head": "both_head",
    # Same three pucks as `both_head`, but the converter must ALSO fold the right hand's
    # Bota + DM4310 into the HL windows. A distinct flag (rather than a boolean on
    # `both_head`) is what makes `attrs.hands_mode` a sufficient gate: the two produce
    # different key SETS, so a dual file loaded as `both_hands_head` — which would drop
    # half the sensors AND mis-index the head's state block — is refused by name.
    "both_hands_head_dual": "both_head_dual",
    # SAME conversion as the dual mode, deliberately. A wholebody dataset IS a
    # `both_head_dual` zarr with the `wb_*` joint arrays added by
    # `motion_completion/mc_to_hl_wholebody.py`, so its `attrs.hands_mode` still reads
    # `both_head_dual` and the SAME file trains either preset. What separates the two is the
    # presence of `attrs.wb_windows` (checked by HLDataset for this mode only) and
    # `layout_mode` in the checkpoint — never a different --hands value, because a second
    # conversion of the same session would be a second copy of every camera frame.
    "wholebody": "both_head_dual",
    # The geom family reads the SAME arrays — it only reduces `wb_state_joints` to a
    # pelvis-frame geometry at LOAD time — so one file trains all five whole-body modes and
    # none of them is a new conversion. Built from `_WB_MODES` so a mode added there without a
    # flag here is impossible; `test_hl_contract` re-checks the mapping against the converter.
    **{mode: "both_head_dual" for mode in _WB_MODES if mode != "wholebody"},
    # The 5-point family reads the SAME conversion too: its extra foot arrays are FK'd from
    # `wb_action_joints` by `scripts/add_foot_targets.py` AFTER the completion, so there is
    # still exactly one `--hands` value and still no second copy of any camera frame.
    **{mode: "both_head_dual" for mode in _5PT_MODES},
}


def hands_flag(mode: str) -> str:
    """The ``convert_to_umi.py --hands`` value that produces ``mode``'s arrays."""
    try:
        return HANDS_FLAG_BY_MODE[mode]
    except KeyError:
        raise ValueError(f"no --hands flag for layout mode {mode!r}; "
                         f"expected one of {sorted(HANDS_FLAG_BY_MODE)}") from None


def get_layout(mode: str = DEFAULT_MODE) -> ActionLayout:
    """Build the :class:`ActionLayout` for ``mode``.

    Raises ``ValueError`` with the valid set on an unknown mode — this is the one
    validation gate every config/CLI should call, so a typo can never fall through
    to a silently mis-sliced action vector.
    """
    if mode == "legacy32":
        return _build_legacy32()
    if mode in _WB_MODES:
        # The same two force ablations apply unchanged: DROP_FORCE empties the DECLARED sensor
        # set (moving no column, so the 43-dim vector and its joint block keep their meaning)
        # and DROP_FORCE_INPUT removes only the conditioning. Routed through the same flags
        # rather than refused, so a pure-position whole-body ablation is one env var and stays
        # metric-for-metric comparable with the force-ful run.
        return _build_wholebody(
            mode,
            state_geom=_WB_MODES[mode][1],
            force_roles=() if DROP_FORCE else ("left", "right"),
            force_input=not (DROP_FORCE_INPUT or DROP_FORCE),
        )
    if mode in _5PT_MODES:
        # Keypoint action, geometric state. Built through `_build_v2` — the SAME function every
        # v2 mode uses, because the action layout genuinely is a v2 one with two pose-only roles
        # appended — and then the geometric state is attached. `dataclasses.replace` rather than
        # a widened `_build_v2` signature: the action arithmetic must not gain a parameter it
        # would then be possible to pass inconsistently, and `state_geom` touches no column.
        layout = _build_v2(
            mode, _5PT_ROLES,
            () if DROP_FORCE else ("left", "right"), ("left", "right"),
            force_input=not (DROP_FORCE_INPUT or DROP_FORCE),
        )
        return dataclasses.replace(layout, state_geom=_5PT_MODES[mode][0],
                                   state_geometric=True)
    spec = _V2_MODES.get(mode)
    if spec is None:
        raise ValueError(f"unknown layout mode {mode!r}; expected one of {MODES}")
    roles, force_roles, gripper_roles = spec
    if DROP_FORCE:
        # Reserved columns and therefore action_dim are UNCHANGED (see _build_v2): only the
        # DECLARED sensor set empties, which is what makes the force columns unsupervised and
        # drops force_history to 0 channels. So a pure-position checkpoint keeps every other
        # column's meaning and stays comparable, metric for metric, with a force-ful run.
        force_roles = ()
    # force_input is the ONLY thing this flag changes: force_roles stays intact, so the action
    # mask still supervises the force columns and the dataset still emits a 6-channel window.
    # DROP_FORCE already emptied force_roles above, which makes force_input_channels 0 anyway
    # -- passing False here too keeps the two flags composable instead of order-dependent.
    return _build_v2(mode, roles, force_roles, gripper_roles,
                     force_input=not (DROP_FORCE_INPUT or DROP_FORCE))


def tracker_roles_required(mode: str) -> tuple[str, ...]:
    """Which TRACKERS a mode needs present in the source recording.

    A mode can only be built from a session that recorded at least these roles;
    ``both_hands`` cannot be extracted from a 1pt session. The converter uses this
    to fail loudly instead of silently degrading to fewer keypoints.

    ⚠️ This is NOT ``layout.roles`` for ``wholebody``, and the difference is load-bearing:
    that mode's roles are the two hand BLOCKS (grip + wrench) while its 29 joint columns come
    from a completion conditioned on both hands AND the chest. Any consumer asking "which
    tracker's glitch corrupts this sample" must read THIS — ``HLDataset._glitch_window_keep``
    does — because a chest rotation excursion poisons the completion's root channel and hence
    all 29 supervised joint columns, while ``layout.roles`` would never look at the chest.
    For every keypoint mode the two are equal, so nothing else changes.

    ⚠️ The ``fivepoint_*`` family breaks that equality the OTHER way: ``layout.roles`` is five
    but only THREE are tracked. The two feet are the completion's inference (nothing on this rig
    measures a foot), so a "which tracker's glitch corrupts this sample" question must not ask
    about them — and equally, a glitch on any of the three DOES corrupt the feet, because they
    were FK'd from joints the completion produced off exactly those three points.
    """
    if mode == "legacy32":
        return ("left",)
    if mode in _WB_MODES:
        return _WB_MODES[mode][0]
    if mode in _5PT_MODES:
        return _WB_TRACKERS
    if mode not in _V2_MODES:
        raise ValueError(f"unknown layout mode {mode!r}; expected one of {MODES}")
    return _V2_MODES[mode][0]
