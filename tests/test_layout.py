"""Contract tests for the single action-layout registry (``fuji.s1_policy.data.layout``).

These pin the column meaning of every trained checkpoint. A failure here means a
model's learned columns no longer mean what the dataset writes into them, which is
silent at train time and catastrophic at deploy time — so treat any change that
needs these edited as a checkpoint-breaking change.
"""

from __future__ import annotations

import pytest

from fuji.s1_policy.data.layout import (
    ACTION_DT,
    ACTION_HORIZON,
    ACTION_HORIZON_DEFAULT,
    ACTION_HORIZON_ENV,
    ACTION_HZ,
    ACTION_SPAN_S,
    DROP_FORCE_ENV,
    DROP_FORCE_INPUT_ENV,
    FORCE_DIM,
    FORCE_HIST_LEN,
    FORCE_HIST_LEN_DEFAULT,
    FORCE_HIST_LEN_ENV,
    FORCE_HIST_SPAN_S,
    GRIPPER_DIM,
    HAND_BLOCK_DIM,
    HEAD_BLOCK_DIM,
    MODES,
    OBS_HISTORY_LEN,
    POS_DIM,
    ROT6D_DIM,
    _resolve_action_horizon,
    _resolve_drop_force,
    _resolve_drop_force_input,
    _resolve_force_hist_len,
    get_layout,
    tracker_roles_required,
    zarr_keys_for,
)


def test_block_widths_are_the_documented_contract():
    """Contract v3 (2026-08-07): NO history, 30 Hz, 50-step chunk.

    Deliberately LITERAL — this test's job is to fail when the contract moves, so the
    move is paired with the matching converter change. The mirror lives in
    UMI_collection/processing/umi_convert_core.py::HLGrid and is asserted equal by
    test_hl_dataset.py::test_hl_contract_matches_converter."""
    assert (POS_DIM, ROT6D_DIM, GRIPPER_DIM, FORCE_DIM) == (3, 6, 1, 6)
    assert HAND_BLOCK_DIM == 16      # the per-hand ACTION block (pos3+rot6+grip1+force6)
    assert HEAD_BLOCK_DIM == 9
    # 30 Hz == CAMERA_PUBLISH_HZ, so the obs grid and the action grid coincide. The RATE is
    # literal; the step COUNT is asserted on the DEFAULT because FUJI_ACTION_HORIZON may pin
    # this process to a shorter chunk (an H=50 vs H=30 A/B) — same treatment as
    # FORCE_HIST_LEN below, and for the same reason.
    assert ACTION_HORIZON_DEFAULT == 50
    assert ACTION_HZ == 30.0
    assert ACTION_DT == pytest.approx(1.0 / 30.0)
    # The derivation holds under ANY pin: the span is (resolved count) * dt.
    assert ACTION_SPAN_S == pytest.approx(ACTION_HORIZON * ACTION_DT)
    # No look-back: the observation is the single anchor frame t0. The one force bin is
    # exactly one action step wide. Asserted on the DEFAULT, not on the resolved value —
    # FUJI_FORCE_HIST_LEN may pin this process to another L (an L=1 vs L=6 A/B), and this
    # test's job is to pin the CONTRACT that HLGrid mirrors, not to forbid the pin.
    assert FORCE_HIST_LEN_DEFAULT == 1
    # The derivation holds under ANY pin: bins are one action step wide, always.
    assert FORCE_HIST_SPAN_S == pytest.approx(FORCE_HIST_LEN * ACTION_DT)
    assert OBS_HISTORY_LEN == 1


@pytest.mark.parametrize(
    ("env", "expect"),
    [(None, FORCE_HIST_LEN_DEFAULT), ("", FORCE_HIST_LEN_DEFAULT), ("1", 1), ("6", 6), (" 8 ", 8)],
)
def test_force_hist_len_pin_resolves(monkeypatch, env, expect):
    """``FUJI_FORCE_HIST_LEN`` re-pins the bin count for ONE process."""
    if env is None:
        monkeypatch.delenv(FORCE_HIST_LEN_ENV, raising=False)
    else:
        monkeypatch.setenv(FORCE_HIST_LEN_ENV, env)
    assert _resolve_force_hist_len() == expect


@pytest.mark.parametrize("env", ["0", "-1", "x", "1.5"])
def test_force_hist_len_pin_refuses_nonsense(monkeypatch, env):
    """A bad pin must be a loud refusal at import, not a silently mistrained model.

    ``0`` is refused specifically: it reads as "no force input", but dropping force is a
    LAYOUT change (``force_hist_channels`` gates the encoder AND the force action columns),
    so honouring it here would half-apply it.
    """
    monkeypatch.setenv(FORCE_HIST_LEN_ENV, env)
    with pytest.raises(ValueError, match=FORCE_HIST_LEN_ENV):
        _resolve_force_hist_len()


@pytest.mark.parametrize(
    ("env", "expect"),
    [(None, ACTION_HORIZON_DEFAULT), ("", ACTION_HORIZON_DEFAULT), ("50", 50), ("30", 30), (" 16 ", 16)],
)
def test_action_horizon_pin_resolves(monkeypatch, env, expect):
    """``FUJI_ACTION_HORIZON`` re-pins the chunk length for ONE process (H=50 vs H=30 A/B)."""
    if env is None:
        monkeypatch.delenv(ACTION_HORIZON_ENV, raising=False)
    else:
        monkeypatch.setenv(ACTION_HORIZON_ENV, env)
    assert _resolve_action_horizon() == expect


@pytest.mark.parametrize("env", ["0", "-1", "x", "1.5"])
def test_action_horizon_pin_refuses_nonsense(monkeypatch, env):
    """A bad pin is a loud refusal at import, not a silently mis-shaped chunk."""
    monkeypatch.setenv(ACTION_HORIZON_ENV, env)
    with pytest.raises(ValueError, match=ACTION_HORIZON_ENV):
        _resolve_action_horizon()


@pytest.mark.parametrize("env,expect", [("51", 51), ("100", 100), ("200", 200)])
def test_action_horizon_pin_may_exceed_the_default(monkeypatch, env, expect):
    """A pin ABOVE the default is ALLOWED, and the dataset is what refuses an unservable one.

    This direction used to raise here. It does not any more, and the reason is worth stating
    because the old refusal read as a soundness claim: the pin works by taking a PREFIX of what
    a dataset stores, so a longer horizon needs a dataset that stores it — which is exactly
    what ``HLDataset._check_file`` asserts, per file, naming the file, its stored horizon and
    both remedies. Refusing here instead only moved the message earlier and made the
    ``H=100``/``H=200`` runs impossible without editing a constant that every other dataset,
    checkpoint and launcher reads.

    :data:`ACTION_HORIZON_DEFAULT` deliberately stays 50 so nothing else moves, and the prefix
    relation still means ONE 200-step conversion serves H=50/100/200.
    """
    monkeypatch.setenv(ACTION_HORIZON_ENV, env)
    assert _resolve_action_horizon() == expect
    assert expect > ACTION_HORIZON_DEFAULT


@pytest.mark.parametrize("env,expect", [
    (None, False), ("", False), ("0", False), ("false", False), ("no", False), ("off", False),
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True), (" 1 ", True),
])
def test_drop_force_pin_parses(monkeypatch, env, expect):
    """The pure-position pin that ``FUJI_FORCE_HIST_LEN=0`` deliberately refuses to be."""
    if env is None:
        monkeypatch.delenv(DROP_FORCE_ENV, raising=False)
    else:
        monkeypatch.setenv(DROP_FORCE_ENV, env)
    assert _resolve_drop_force() is expect


@pytest.mark.parametrize("env", ["2", "maybe", "-1", "None"])
def test_drop_force_pin_refuses_nonsense(monkeypatch, env):
    """Silently reading an unrecognized value as False would train a force-FUL run under a
    flag the operator believes disabled force — the ablation would compare a run to itself."""
    monkeypatch.setenv(DROP_FORCE_ENV, env)
    with pytest.raises(ValueError, match=DROP_FORCE_ENV):
        _resolve_drop_force()


def test_drop_force_empties_declared_force_but_moves_no_column():
    """The pure-position ablation must be a change to DECLARED hardware only.

    Reserved columns are what keep a checkpoint's column meaning stable, so dropping the
    sensor may empty ``force_roles`` (which is what un-supervises the force columns and
    takes ``force_hist_channels`` to 0, so no encoder and no force token are built) while
    leaving ``action_dim``, ``state_dim``, the reserved force columns, and the gripper
    declaration untouched. Otherwise the ablation would not be comparable, metric for
    metric, against the force-ful run it exists to be compared against.
    """
    from fuji.s1_policy.data import layout as layout_mod

    for mode in (m for m in MODES if m != "legacy32"):
        full = get_layout(mode)
        # get_layout reads the module-level flag at CALL time, so patching it is enough.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(layout_mod, "DROP_FORCE", True)
            bare = layout_mod.get_layout(mode)
        assert bare.force_roles == (), mode
        assert bare.force_hist_channels == 0, mode
        assert full.force_hist_channels == len(full.force_roles) * FORCE_DIM, mode
        # Nothing else may move.
        assert bare.action_dim == full.action_dim, mode
        assert bare.state_dim == full.state_dim, mode
        assert bare.gripper_roles == full.gripper_roles, mode
        assert bare.force_dims() == full.force_dims(), mode
        assert bare.pose_dims() == full.pose_dims(), mode
        assert bare.gripper_dims() == full.gripper_dims(), mode
        assert bare.keypoints == full.keypoints, mode


@pytest.mark.parametrize("env,expect", [
    (None, False), ("", False), ("0", False), ("false", False), ("no", False), ("off", False),
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True), (" 1 ", True),
])
def test_drop_force_input_pin_parses(monkeypatch, env, expect):
    """The force-OUTPUT-ONLY pin: no wrench in the observation, force columns still trained."""
    if env is None:
        monkeypatch.delenv(DROP_FORCE_INPUT_ENV, raising=False)
    else:
        monkeypatch.setenv(DROP_FORCE_INPUT_ENV, env)
    assert _resolve_drop_force_input() is expect


@pytest.mark.parametrize("env", ["2", "maybe", "-1", "None"])
def test_drop_force_input_pin_refuses_nonsense(monkeypatch, env):
    """Reading an unrecognized value as False would train a force-CONDITIONED run under a flag
    the operator believes removed the input — the ablation would compare a run to itself."""
    monkeypatch.setenv(DROP_FORCE_INPUT_ENV, env)
    with pytest.raises(ValueError, match=DROP_FORCE_INPUT_ENV):
        _resolve_drop_force_input()


def test_drop_force_input_removes_the_input_and_keeps_the_output_supervised():
    """The whole point of the ablation, and the one thing that distinguishes it from
    ``FUJI_DROP_FORCE``.

    Asks "is the measured wrench useful as an INPUT, or is it already predictable from vision
    plus state?" — which is only answerable if the force COLUMNS stay supervised, so the
    force-ful run's ``val/force`` is directly comparable. So exactly one thing may change:
    ``force_input_channels`` (what the model is built to consume). ``force_roles``,
    ``measured_dims`` (the action mask), the per-group metric dims and
    ``force_hist_channels`` (the width the DATASET emits, and therefore what the norm-stats
    file describes) must all be untouched — that last one is what lets this run reuse the
    force-ful run's stats file byte for byte and train on a bit-identical anchor set.
    """
    from fuji.s1_policy.data import layout as layout_mod

    for mode in (m for m in MODES if m != "legacy32"):
        full = get_layout(mode)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(layout_mod, "DROP_FORCE_INPUT", True)
            ablated = layout_mod.get_layout(mode)

        # The ONE change: the model consumes no force window.
        assert ablated.force_input is False, mode
        assert ablated.force_input_channels == 0, mode
        assert full.force_input_channels == full.force_hist_channels, mode

        # Everything the OUTPUT side depends on is identical.
        assert ablated.force_roles == full.force_roles, mode
        assert ablated.measured_dims() == full.measured_dims(), mode
        assert ablated.unmeasured_dims() == full.unmeasured_dims(), mode
        assert ablated.metric_group_dims() == full.metric_group_dims(), mode
        # And so is the dataset's own force width, which is why the norm stats carry over.
        assert ablated.force_hist_channels == full.force_hist_channels, mode
        assert ablated.action_dim == full.action_dim, mode
        assert ablated.state_dim == full.state_dim, mode
        assert ablated.keypoints == full.keypoints, mode

        # On a rig that declares an F/T sensor the ablation must actually bite. Checked per
        # DECLARED role, not over force_dims(): a bimanual mode RESERVES 12 force columns
        # while this rig measures only the left 6, so the whole reserved block is supervised
        # in no configuration (see layout.py "RESERVED vs DECLARED").
        if full.force_roles:
            assert full.force_input_channels > 0, mode
            for role in full.force_roles:
                block = ablated.slots(role).force
                assert block is not None, (mode, role)
                supervised = set(ablated.measured_dims())
                assert set(range(block.start, block.stop)) <= supervised, (
                    f"{mode}/{role}: the force columns stopped being supervised, which makes "
                    "this the PURE-POSITION ablation instead")


def test_the_two_force_ablations_compose_rather_than_contradict():
    """``FUJI_DROP_FORCE`` implies no force input, and setting both is still pure position.

    They are different experiments, but a rig with no sensor cannot condition on one. Without
    this, ``DROP_FORCE`` alone would leave ``force_input=True`` with 0 channels behind it —
    coherent by accident today (``force_input_channels`` multiplies through an empty
    ``force_roles``) and a trap the moment either property is computed differently.
    """
    from fuji.s1_policy.data import layout as layout_mod

    for drop, drop_input in ((True, False), (True, True), (False, True)):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(layout_mod, "DROP_FORCE", drop)
            mp.setattr(layout_mod, "DROP_FORCE_INPUT", drop_input)
            lay = layout_mod.get_layout("left_only")
        assert lay.force_input is False, (drop, drop_input)
        assert lay.force_input_channels == 0, (drop, drop_input)
        # Only DROP_FORCE un-supervises the columns; DROP_FORCE_INPUT alone must not.
        # left_only reserves exactly one force block, so force_dims() IS the left block here.
        supervised = set(lay.force_dims()) <= set(lay.measured_dims())
        assert supervised == (not drop), (drop, drop_input)


@pytest.mark.parametrize(
    ("mode", "action_dim", "roles"),
    [
        ("left_only", 16, ("left",)),
        ("right_only", 16, ("right",)),
        ("both_hands", 32, ("left", "right")),
        ("both_hands_head", 41, ("left", "right", "head")),
        ("both_hands_head_dual", 41, ("left", "right", "head")),
        ("legacy32", 32, ("left", "right", "head")),
    ],
)
def test_mode_dims_and_roles(mode, action_dim, roles):
    lay = get_layout(mode)
    assert lay.action_dim == action_dim
    assert lay.roles == roles


def test_left_only_exact_slices():
    """The first deliverable's layout, spelled out. Any drift breaks the ckpt."""
    lay = get_layout("left_only")
    left = lay.slots("left")
    assert (left.pos.start, left.pos.stop) == (0, 3)
    assert (left.rot6d.start, left.rot6d.stop) == (3, 9)
    assert left.gripper == 9
    assert (left.force.start, left.force.stop) == (10, 16)
    # State (DECISION-WRT0): per keypoint [wrt_start_rot6d 6][gripper 1?]. Left has both.
    assert lay.state_dim == 7               # wrt_start_rot6d(6) + gripper(1)
    assert lay.state_slots() == {"left": {"wrt_start_rot6d": slice(0, 6), "gripper": 6}}
    assert lay.force_hist_channels == 6     # one 6-DoF wrench
    assert not lay.has("right") and not lay.has("head")


def test_v2_modes_partition_every_dim_with_no_pad():
    """v2 has NO pad dims: every column is claimed by exactly one keypoint quantity."""
    for mode in ("left_only", "right_only", "both_hands", "both_hands_head"):
        lay = get_layout(mode)
        claimed = list(lay.pose_dims()) + list(lay.force_dims()) + list(lay.gripper_dims())
        assert sorted(claimed) == list(range(lay.action_dim)), mode
        assert len(claimed) == len(set(claimed)), f"{mode}: a dim is double-claimed"
        assert lay.pad_dims() == (), mode


def test_keypoint_blocks_are_contiguous_and_ordered():
    for mode in ("left_only", "right_only", "both_hands", "both_hands_head"):
        lay = get_layout(mode)
        cursor = 0
        for kp in lay.keypoints:
            assert kp.block is not None, f"{mode}/{kp.name} has no block"
            assert kp.block.start == cursor, f"{mode}/{kp.name} not contiguous"
            # Every dim the keypoint claims must fall inside its own block.
            owned = (list(range(kp.pos.start, kp.pos.stop))
                     + list(range(kp.rot6d.start, kp.rot6d.stop))
                     + ([kp.gripper] if kp.gripper is not None else [])
                     + (list(range(kp.force.start, kp.force.stop)) if kp.force else []))
            assert all(kp.block.start <= d < kp.block.stop for d in owned), f"{mode}/{kp.name}"
            assert len(owned) == kp.block.stop - kp.block.start, f"{mode}/{kp.name} has a hole"
            cursor = kp.block.stop
        assert cursor == lay.action_dim


def test_legacy32_blocks_are_not_claimed_contiguous():
    """legacy32 slots are interleaved with pad; ``block`` must be None, not a lie."""
    lay = get_layout("legacy32")
    assert all(kp.block is None for kp in lay.keypoints)


def test_smaller_mode_is_a_strict_prefix_of_larger():
    """Prefix property: appending a hand appends COLUMNS and never permutes old ones,
    so a left-only checkpoint can be expanded into a bimanual one."""
    left = get_layout("left_only")
    both = get_layout("both_hands")
    head = get_layout("both_hands_head")
    for bigger in (both, head):
        assert bigger.slots("left") == left.slots("left")
    assert head.slots("right") == both.slots("right")


def test_pose_dims_exclude_gripper_and_force():
    """Only pos+rot6d may go through the anchor-relative SE(3) transform; a gripper
    opening and a wrench are frame-invariant scalars/vectors that must pass through."""
    lay = get_layout("both_hands_head")
    pose = set(lay.pose_dims())
    assert pose.isdisjoint(lay.force_dims())
    assert pose.isdisjoint(lay.gripper_dims())
    # Every hand contributes 9 pose dims, the head 9 -> 27.
    assert len(pose) == 27


def test_metric_groups_cover_exactly_the_real_dims():
    for mode in MODES:
        lay = get_layout(mode)
        groups = lay.metric_group_dims()
        by_keypoint: set[int] = set()
        for role in lay.roles:
            by_keypoint |= set(groups[role])
        by_quantity: set[int] = set()
        for name in ("pos", "rot", "joints", "grip", "force"):
            by_quantity |= set(groups.get(name, ()))
        # The two partitions must describe the SAME dim set. `joints` is the one group that
        # belongs to NO keypoint — it is the whole body, and putting it under a hand would
        # claim one arm owns the legs — so it joins the keypoint side explicitly here rather
        # than being quietly absent from one of the two partitions.
        assert by_keypoint | set(groups.get("joints", ())) == by_quantity, mode
        # ...and together with the pad dims cover the whole vector.
        assert by_quantity | set(lay.pad_dims()) == set(range(lay.action_dim)), mode


def test_metric_groups_omit_empty_groups():
    """A left-only run must not log a constant-zero flow_loss/right."""
    groups = get_layout("left_only").metric_group_dims()
    assert set(groups) == {"left", "pos", "rot", "grip", "force"}
    assert "right" not in groups and "head" not in groups


def test_legacy32_is_frozen_and_padded():
    """The pre-v2 layout must keep its exact historical column meaning."""
    lay = get_layout("legacy32")
    assert lay.legacy is True
    assert lay.action_dim == 32
    assert (lay.slots("left").pos, lay.slots("left").rot6d, lay.slots("left").gripper) == (
        slice(0, 3), slice(3, 9), 9)
    assert (lay.slots("right").pos, lay.slots("right").rot6d, lay.slots("right").gripper) == (
        slice(10, 13), slice(13, 19), 19)
    assert (lay.slots("head").pos, lay.slots("head").rot6d, lay.slots("head").gripper) == (
        slice(20, 23), slice(23, 29), None)
    assert lay.pad_dims() == (29, 30, 31)
    assert lay.force_dims() == ()          # legacy carries no force
    assert lay.force_hist_channels == 0


def test_legacy32_matches_the_historical_dataset_slot_table():
    """Cross-check against the live pre-v2 dataset module, which the deployed
    checkpoint was trained with. If these ever disagree, the old ckpt is mis-sliced."""
    from fuji.s1_policy.data.umi_dataset import _HEAD, _LEFT, _RIGHT, ACTION_DIM

    lay = get_layout("legacy32")
    assert lay.action_dim == ACTION_DIM
    for role, (pos_sl, rot_sl, grip_i) in (("left", _LEFT), ("right", _RIGHT), ("head", _HEAD)):
        kp = lay.slots(role)
        assert (kp.pos, kp.rot6d, kp.gripper) == (pos_sl, rot_sl, grip_i), role


def test_legacy32_matches_the_transforms_ref_frame_slots():
    from fuji.s1_policy.data.transforms import _REF_FRAME_SLOTS

    lay = get_layout("legacy32")
    for role, (pos_sl, rot_sl) in _REF_FRAME_SLOTS.items():
        kp = lay.slots(role)
        assert (kp.pos, kp.rot6d) == (pos_sl, rot_sl), role


def test_reserved_vs_measured_columns():
    """The distinction that keeps checkpoints stable: columns are RESERVED for every
    hand, but only DECLARED roles have hardware. On this rig (one Bota + one gripper,
    both left) the right hand's gripper/force columns are reserved yet unmeasured."""
    lay = get_layout("both_hands")
    assert lay.gripper_dims() == (9, 25)            # both reserved
    assert lay.gripper_roles == ("left",)           # only left is real
    assert lay.unmeasured_dims() == (25, 26, 27, 28, 29, 30, 31)
    # ...i.e. exactly the right hand's gripper + force block.
    right = lay.slots("right")
    assert set(lay.unmeasured_dims()) == {right.gripper} | set(
        range(right.force.start, right.force.stop))
    # measured + unmeasured partition the vector, with no overlap.
    assert set(lay.measured_dims()) | set(lay.unmeasured_dims()) == set(range(lay.action_dim))
    assert set(lay.measured_dims()).isdisjoint(lay.unmeasured_dims())
    # A single-hand mode has nothing unmeasured.
    assert get_layout("left_only").unmeasured_dims() == ()
    # The head has neither gripper nor force reserved, so it contributes nothing.
    head = get_layout("both_hands_head").slots("head")
    assert head.gripper is None and head.force is None


def test_observation_widths_follow_hardware_not_columns():
    """force_history is sized by DECLARED hardware (one 6-DoF sensor here). state
    (DECISION-WRT0) carries wrt_start_rot6d for EVERY keypoint plus a gripper only for
    gripper roles: left_only 6+1=7; both_hands 7+6=13; both_hands_head 7+6+6=19."""
    assert get_layout("left_only").state_dim == 7
    assert get_layout("both_hands").state_dim == 13
    assert get_layout("both_hands_head").state_dim == 19
    for mode in ("left_only", "both_hands", "both_hands_head"):
        lay = get_layout(mode)
        assert lay.force_hist_channels == 6, mode    # one physical F/T sensor
        assert len(lay.force_dims()) >= 6            # but >=6 columns are reserved
        # state_slots partitions state_dim contiguously in keypoint order.
        assert sum((6 + (1 if kp.name in lay.gripper_roles else 0))
                   for kp in lay.keypoints) == lay.state_dim


def test_dual_mode_shares_the_action_layout_but_not_the_state_layout():
    """``both_hands_head_dual`` is ``both_hands_head`` with the hardware DECLARED.

    Two properties, and the asymmetry between them is the whole trap:

    * the ACTION vector is column-identical — 41 dims, every keypoint's slices equal — so
      declaring a second F/T + gripper moved nothing a checkpoint learned. That is what
      "RESERVED vs DECLARED" buys, and it is why the modes can share a dataset's
      ``hl_*_action_pose`` arrays byte for byte.
    * the STATE vector is NOT a prefix extension. The right hand's gripper is INSERTED at
      index 13, pushing the head's wrt_start_rot6d from [13:19] to [14:20]. Both are
      plausible 19/20-wide vectors, so a cross-loaded checkpoint would read the head's
      orientation as (gripper, 5 rot6d components) with no shape error anywhere.
    """
    base = get_layout("both_hands_head")
    dual = get_layout("both_hands_head_dual")

    assert dual.action_dim == base.action_dim == 41
    assert dual.roles == base.roles
    for role in base.roles:
        assert dual.slots(role) == base.slots(role), role
    assert dual.pose_dims() == base.pose_dims()
    assert dual.force_dims() == base.force_dims()
    assert dual.gripper_dims() == base.gripper_dims()

    # Declared hardware: both hands now measure force AND gripper.
    assert dual.force_roles == ("left", "right")
    assert dual.gripper_roles == ("left", "right")
    assert dual.force_hist_channels == 12
    assert dual.force_input_channels == 12
    # ...so every reserved column is finally supervised: 41 columns, nothing masked off.
    assert dual.unmeasured_dims() == ()
    # while the single-sensor mode leaves the right hand's 7 columns unsupervised.
    assert base.unmeasured_dims() == (25, 26, 27, 28, 29, 30, 31)

    # The state INSERTION, spelled out.
    assert base.state_dim == 19 and dual.state_dim == 20
    # left is [rot6d 0:6][grip 6] in both, so the right hand starts at 7 either way...
    assert base.state_slots()["left"] == {"wrt_start_rot6d": slice(0, 6), "gripper": 6}
    assert dual.state_slots()["left"] == base.state_slots()["left"]
    assert base.state_slots()["right"] == {"wrt_start_rot6d": slice(7, 13)}
    # ...and the dual mode INSERTS the right gripper at 13, pushing the head up by one.
    assert dual.state_slots()["right"] == {"wrt_start_rot6d": slice(7, 13), "gripper": 13}
    assert base.state_slots()["head"]["wrt_start_rot6d"] == slice(13, 19)
    assert dual.state_slots()["head"]["wrt_start_rot6d"] == slice(14, 20)
    # Not a prefix in either direction, unlike left_only ⊂ both_hands ⊂ both_hands_head.
    assert base.state_slots()["head"] != dual.state_slots()["head"]


def test_dual_mode_metric_groups_cover_both_hands_force_and_grip():
    """A dual run must LOG the right hand's force/grip, or its whole contribution is
    invisible on the dashboard even though the loss is optimizing it."""
    groups = get_layout("both_hands_head_dual").metric_group_dims()
    assert set(groups) == {"left", "right", "head", "pos", "rot", "grip", "force"}
    assert groups["grip"] == (9, 25)                    # BOTH grippers
    assert groups["force"] == tuple(range(10, 16)) + tuple(range(26, 32))
    assert groups["right"] == tuple(range(16, 32))       # the full right hand block


def test_force_roles_reflect_the_single_left_hand_sensor():
    """The rig has ONE Bota + ONE gripper, both on the primary (left) hand. Force
    COLUMNS are still reserved in every hand block so adding a right sensor later
    moves no checkpoint column."""
    assert get_layout("left_only").force_roles == ("left",)
    assert get_layout("both_hands").force_roles == ("left",)
    assert get_layout("both_hands_head").force_roles == ("left",)
    assert get_layout("right_only").force_roles == ("right",)
    # Reserved regardless: the right hand block owns force columns in both_hands.
    assert get_layout("both_hands").slots("right").force == slice(26, 32)


def test_unknown_mode_raises_with_the_valid_set():
    with pytest.raises(ValueError, match="unknown layout mode"):
        get_layout("left")           # a plausible typo for left_only
    with pytest.raises(ValueError, match="unknown layout mode"):
        tracker_roles_required("bimanual")


def test_tracker_roles_required():
    assert tracker_roles_required("left_only") == ("left",)
    assert tracker_roles_required("both_hands") == ("left", "right")
    assert tracker_roles_required("both_hands_head") == ("left", "right", "head")


def test_zarr_keys_resolve_primary_role_indirection():
    """robot0_* is the PRIMARY hand; robot1_* the other. Left-canonical vs legacy."""
    left_primary = zarr_keys_for("left")
    assert left_primary["left"][0] == "robot0_eef_pos"
    assert left_primary["right"][0] == "robot1_eef_pos"
    right_primary = zarr_keys_for("right")
    assert right_primary["right"][0] == "robot0_eef_pos"
    assert right_primary["left"][0] == "robot1_eef_pos"
    assert left_primary["head"] == ("head_pos", "head_rot_axis_angle", None)
    with pytest.raises(ValueError):
        zarr_keys_for("head")


def test_layout_zarr_keys_restricted_to_mode():
    """A left-only dataset over a 3pt zarr must never touch robot1/head keys."""
    keys = get_layout("left_only").zarr_keys("left")
    assert set(keys) == {"left"}
    assert keys["left"] == ("robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width")
    both = get_layout("both_hands").zarr_keys("left")
    assert set(both) == {"left", "right"}


def test_force_zarr_prefix():
    lay = get_layout("both_hands")
    assert lay.force_zarr_prefix("left", "left") == "robot0"
    # A right-primary (legacy-style) session puts the left hand in robot1.
    assert lay.force_zarr_prefix("right", "left") == "robot1"
    with pytest.raises(KeyError):
        lay.force_zarr_prefix("left", "right")   # no right F/T sensor on this rig


def test_slots_lookup_error_names_the_present_keypoints():
    with pytest.raises(KeyError, match="present"):
        get_layout("left_only").slots("right")



def test_mode_role_tuples_are_in_keypoint_order():
    """_build_v2 lays roles out in the order GIVEN — it does not sort. So the invariant
    has to hold where the modes are DECLARED, or a mode written out of order would
    silently permute a checkpoint's columns."""
    from fuji.s1_policy.data.layout import KEYPOINT_ORDER, _V2_MODES

    rank = {r: i for i, r in enumerate(KEYPOINT_ORDER)}
    for mode, (roles, force_roles, gripper_roles) in _V2_MODES.items():
        ranks = [rank[r] for r in roles]
        assert ranks == sorted(ranks), f"{mode}: roles {roles} not in KEYPOINT_ORDER"
        # The declared hardware sets must also be subsets of the mode's roles.
        assert set(force_roles) <= set(roles), mode
        assert set(gripper_roles) <= set(roles), mode
    # ...and the built layout's keypoints come out in that same order.
    for mode in _V2_MODES:
        lay = get_layout(mode)
        assert [rank[r] for r in lay.roles] == sorted(rank[r] for r in lay.roles), mode


def test_right_only_is_not_prefix_compatible():
    """REGRESSION on a DOCSTRING claim. The prefix affordance ("expand a checkpoint by
    appending columns") holds only for the left-starting chain. right_only puts the right
    hand at [0:16] because it is that mode's first keypoint, while both_hands puts it at
    [16:32] — so expanding a right_only checkpoint would read every learned right-hand
    column as the LEFT hand, and the shapes match at every step, so the failure is
    silent. The docstring now says so; this pins it."""
    right = get_layout("right_only")
    both = get_layout("both_hands")
    assert right.slots("right") != both.slots("right")
    assert right.slots("right").pos == slice(0, 3)
    assert both.slots("right").pos == slice(16, 19)
    # It is not a prefix of ANY other mode.
    for other in ("left_only", "both_hands", "both_hands_head"):
        lay = get_layout(other)
        assert not lay.has("right") or lay.slots("right") != right.slots("right"), other
    # ...and right_only's block occupies the same columns as left_only's LEFT block,
    # which is exactly why a mode tag must be baked into the checkpoint.
    assert get_layout("left_only").slots("left").pos == right.slots("right").pos


def test_keypoint_slots_equality_works_and_hashing_needs_py312():
    """Equality is what the prefix checks use and works everywhere. Hashing depends on
    `slice` being hashable (Python 3.12+), which is why pyproject's floor is 3.12."""
    import sys

    a = get_layout("left_only").slots("left")
    b = get_layout("both_hands").slots("left")
    assert a == b
    if sys.version_info >= (3, 12):
        assert hash(a) == hash(b)
    else:  # pragma: no cover - the repo runs 3.13
        with pytest.raises(TypeError):
            hash(a)
