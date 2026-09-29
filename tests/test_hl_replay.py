"""Contract tests for :mod:`fuji.s1_policy.serving.hl_replay` — the recorded-episode
action source that lets the online planner drive the rig with a human demonstration.

The properties pinned here are the ones that decide whether a replayed chunk MOVES THE ARM
CORRECTLY:

  * the recorded chunk is byte-faithful to what the training pipeline builds as
    ``action_chunk`` (so replay reproduces the policy's TARGET, not a re-derivation);
  * it is ANCHOR-RELATIVE, so the recorded lighthouse world frame cancels and the demo
    replays from wherever the arm actually is (the clutch semantic the planner relies on);
  * the gripper column is NORMALIZED, not raw metres (the stored array is metres — feeding
    it through unconverted would command a nearly-shut jaw on a fully-open frame);
  * validity truncation is the LEADING run, so a chunk can never drip the converter's
    end-of-episode padding as if it were real motion;
  * the cursor advances on the ACTION clock, so a ~15 Hz camera grid cannot desynchronize
    from 16 Hz execution.
"""

from __future__ import annotations

import numpy as np
import pytest

from fuji.s1_policy.data.layout import ACTION_DT, ACTION_HORIZON, get_layout
from fuji.s1_policy.data.transforms import (
    GRIPPER_WIDTH_REF_M,
    delta_ee_to_absolute,
    rotvec_to_rot6d,
)
from fuji.utils.rot6d import rot6d_to_rotmat, rotmat_to_rot6d

zarr = pytest.importorskip("zarr")

from tests.test_hl_dataset import _write_hl_zarr  # noqa: E402  (shared schema-faithful builder)


def _source(path, mode="left_only"):
    from fuji.s1_policy.serving.hl_replay import RecordedEpisodeSource

    return RecordedEpisodeSource(path, get_layout(mode))


@pytest.fixture()
def left_zarr(tmp_path):
    return _write_hl_zarr(tmp_path / "replay_left.zarr.zip")


# ---------------------------------------------------------------------------
# Episode geometry
# ---------------------------------------------------------------------------
def test_episode_bounds_partition_the_file_without_gaps_or_overlap(left_zarr):
    src = _source(left_zarr)
    spans = [src.episode_bounds(e) for e in range(src.n_episodes)]
    assert spans[0][0] == 0
    assert spans[-1][1] == src.n_frames
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert prev_end == next_start, "episodes must tile the file exactly"
    assert sum(src.episode_lengths()) == src.n_frames


def test_out_of_range_episode_and_frame_raise(left_zarr):
    src = _source(left_zarr)
    with pytest.raises(IndexError):
        src.episode_bounds(src.n_episodes)
    with pytest.raises(IndexError):
        src.chunk_at(0, src.episode_lengths()[0])  # one past the end
    with pytest.raises(IndexError):
        src.chunk_at(0, -1)


def test_missing_hl_arrays_fail_loudly_naming_the_layout(tmp_path):
    """A file without the pre-baked HL windows must refuse, not replay a silent zero chunk."""
    path = _write_hl_zarr(tmp_path / "no_hl.zarr.zip", omit_keys=("hl_left_action_pose",))
    with pytest.raises(ValueError, match="missing HL action arrays"):
        _source(path)


# ---------------------------------------------------------------------------
# The chunk itself
# ---------------------------------------------------------------------------
def test_relativization_reference_is_the_chunks_own_step0(tmp_path):
    """``chunk_at`` relativizes against ``pose[frame][0]``, NOT the camera-grid anchor array.

    On a faithful file those are identical (DECISION-T0), which means the default fixture
    cannot tell the two implementations apart. So perturb the camera-grid pose away from the
    chunk's step 0 and assert step 0 still relativizes to an EXACT zero delta — pinning that
    the chunk is self-referential and therefore cannot disagree with itself by a frame."""
    path = _write_hl_zarr(tmp_path / "phase.zarr.zip", n_frames=40, n_episodes=1)
    store = zarr.ZipStore(str(path), mode="a")
    root = zarr.group(store=store)
    shifted = np.asarray(root["data"]["robot0_eef_pos"][:], dtype=np.float32) + 0.05
    root["data"]["robot0_eef_pos"][:] = shifted
    store.close()

    src = _source(path)
    slots = get_layout("left_only").slots("left")
    chunk = src.chunk_at(0, 4)
    assert np.allclose(chunk.actions[0, slots.pos], 0.0, atol=1e-12), (
        "step 0 must be a zero delta against the CHUNK's own step 0, independent of the "
        "camera-grid anchor array"
    )


def test_step0_is_the_exact_identity_delta(left_zarr):
    """DECISION-T0: step 0 of the recorded chunk IS the frame's own pose, so relativizing
    against it must give exactly zero translation and an identity rotation. This is what
    makes a replayed chunk start AT the arm instead of jumping."""
    src = _source(left_zarr)
    layout = get_layout("left_only")
    slots = layout.slots("left")
    chunk = src.chunk_at(0, 3)
    assert np.allclose(chunk.actions[0, slots.pos], 0.0, atol=1e-12)
    identity = rot6d_to_rotmat(chunk.actions[0, slots.rot6d])
    assert np.allclose(identity, np.eye(3), atol=1e-7)


def test_chunk_matches_the_training_action_chunk(left_zarr):
    """The replayed chunk must equal what ``HLDataset`` feeds the model as its TARGET.

    This is the whole justification for replay being comparable to the policy: the model is
    trained to emit exactly this vector, so executing it is executing a perfect prediction.
    Compares EVERY channel (pose, gripper, force) — the gripper column is what caught the
    metres-vs-normalized bug, since a pose-only comparison passes right through it.

    This EXACT equality is also what pins the body-frame relativization in place. A world-frame
    encode (tried, measured, reverted — see ``hl_replay._body_vs_world_transport``) breaks it by
    up to 0.033 m in the position column, i.e. replay would stop being the policy's target and
    the demo-vs-policy A/B would no longer be apples-to-apples."""
    from fuji.s1_policy.data.hl_dataset import HLDataset

    src = _source(left_zarr)
    layout = get_layout("left_only")
    slots = layout.slots("left")
    ds = HLDataset(
        left_zarr, mode="left_only", image_key="camera0_rgb", image_mean=0.5, image_std=0.5,
        drop_invalid_force_history=False, drop_glitch_windows=True,
        # To=1 is the v3 contract AND what RecordedEpisodeSource assumes, so the anchor
        # sets being compared are built on the same observation shape. (To=2 would also
        # need glitch_pre_s >= 0.25 s, i.e. a DIFFERENT window than replay's.)
        obs_history_len=1, require_episode_ends=True,
    )
    try:
        compared = 0
        for dataset_index in range(0, len(ds), 3):
            _fi, frame = ds._index[dataset_index]
            episode = int(np.searchsorted(src.ends, frame, side="right"))
            start, _end = src.episode_bounds(episode)
            target = ds[dataset_index]["action_chunk"].numpy().astype(np.float64)
            chunk = src.chunk_at(episode, frame - start)
            k = chunk.n_valid
            for name, columns in (
                ("pos", np.arange(slots.pos.start, slots.pos.stop)),
                ("rot6d", np.arange(slots.rot6d.start, slots.rot6d.stop)),
                ("gripper", np.array([slots.gripper])),
                ("force", np.arange(slots.force.start, slots.force.stop)),
            ):
                assert np.allclose(
                    chunk.actions[:k][:, columns], target[:k][:, columns], atol=1e-6
                ), f"replay {name} disagrees with the training action_chunk at index {dataset_index}"
            compared += 1
        assert compared > 3
    finally:
        ds.close()


def test_gripper_column_is_normalized_not_raw_metres(left_zarr):
    """The stored ``hl_*_action_gripper`` is ABSOLUTE METRES; the action vector wants a
    fraction of ``GRIPPER_WIDTH_REF_M``. The fixture records 0.0475 m == exactly half open,
    so a passthrough bug would show up here as 0.0475 instead of 0.5 — i.e. a jaw commanded
    ~5% open when the demonstration had it half open."""
    src = _source(left_zarr)
    slots = get_layout("left_only").slots("left")
    chunk = src.chunk_at(0, 2)
    assert np.allclose(chunk.actions[:, slots.gripper], 0.5, atol=1e-6)
    # And the decode path recovers the recorded metres.
    assert np.isclose(chunk.actions[0, slots.gripper] * GRIPPER_WIDTH_REF_M, 0.0475, atol=1e-6)


def test_force_passes_through_in_physical_units(left_zarr):
    """Force is neither relativized nor rescaled — the low level's F^cmd gate reads Newtons."""
    src = _source(left_zarr)
    slots = get_layout("left_only").slots("left")
    root = zarr.open(left_zarr, "r")
    frame = 5
    stored = np.asarray(root["data"]["hl_left_action_force"][frame], dtype=np.float64)
    chunk = src.chunk_at(0, frame)
    assert np.allclose(chunk.actions[:, slots.force], stored[: chunk.n_valid], atol=1e-6)


# ---------------------------------------------------------------------------
# Anchor invariance — the property that makes replay frame-safe
# ---------------------------------------------------------------------------
def test_decoding_from_any_anchor_reproduces_the_same_motion(left_zarr):
    """The recorded WORLD frame must not leak into the executed motion.

    Decoding one recorded chunk against two very different live anchors must yield the SAME
    rigid motion (identical step-to-step displacements and identical relative rotations),
    just placed elsewhere. That is exactly why replay needs no lighthouse->pelvis alignment:
    the world frame cancels. Also pins that step 0 lands ON the anchor, so arming and
    executing cannot jump the arm."""
    from fuji.s1_policy.serving.hl_online import decode_chunk

    src = _source(left_zarr)
    layout = get_layout("left_only")
    chunk = src.chunk_at(0, 4)

    def decode(position, rotvec):
        anchor_pos = np.asarray(position, dtype=np.float64).reshape(1, 3)
        anchor_rot = rotvec_to_rot6d(np.asarray(rotvec, dtype=np.float64)).reshape(1, 6)
        result = decode_chunk(
            chunk.actions, layout, anchor_pos, anchor_rot, force_enabled=False, gate_n=0.0
        )
        return anchor_pos[0], result.role("left")

    a_pos, a = decode([0.3, 0.2, 0.9], [0.0, 0.0, 0.0])
    b_pos, b = decode([-0.1, 0.5, 1.2], [0.4, -0.3, 2.1])

    assert np.allclose(a.position[0], a_pos, atol=1e-12), "step 0 must land on the anchor"
    assert np.allclose(b.position[0], b_pos, atol=1e-12)
    # Same shape of motion: per-step displacement lengths are frame-invariant.
    assert np.allclose(
        np.linalg.norm(np.diff(a.position, axis=0), axis=1),
        np.linalg.norm(np.diff(b.position, axis=0), axis=1),
        atol=1e-12,
    )
    # Same orientation change: R_k R_0^T is conjugation-invariant in magnitude.
    for chunk_result in (a, b):
        assert chunk_result.rot6d.shape[0] == chunk.n_valid
    rot_a = rot6d_to_rotmat(a.rot6d)
    rot_b = rot6d_to_rotmat(b.rot6d)
    rel_a = np.einsum("sij,kj->sik", rot_a, rot_a[0])
    rel_b = np.einsum("sij,kj->sik", rot_b, rot_b[0])
    angle_a = np.arccos(np.clip((np.trace(rel_a, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0))
    angle_b = np.arccos(np.clip((np.trace(rel_b, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0))
    assert np.allclose(angle_a, angle_b, atol=1e-9)
    # The gripper is frame-invariant and must be untouched by the anchor.
    assert np.allclose(a.gripper_m, b.gripper_m, atol=1e-12)


# ---------------------------------------------------------------------------
# Validity truncation
# ---------------------------------------------------------------------------
def test_validity_truncates_to_the_leading_run(tmp_path):
    """A chunk is executed IN ORDER, so a validity hole makes every later step unusable —
    even one whose own flag is True. Otherwise the drip would jump across the gap."""
    path = _write_hl_zarr(tmp_path / "holes.zarr.zip", n_frames=40, n_episodes=1)
    store = zarr.ZipStore(str(path), mode="a")
    root = zarr.group(store=store)
    valid = np.asarray(root["data"]["hl_left_action_valid"][:], dtype=np.float32)
    valid[7, 5:] = 0.0          # a clean tail truncation
    valid[8, 3] = 0.0           # a HOLE with valid steps after it
    valid[9, 0] = 0.0           # not even step 0 -> unusable frame
    root["data"]["hl_left_action_valid"][:] = valid
    store.close()

    src = _source(path)
    assert src.chunk_at(0, 7).n_valid == 5
    assert src.chunk_at(0, 8).n_valid == 3, "must stop at the hole, not count later True flags"
    assert src.chunk_at(0, 6).n_valid == ACTION_HORIZON
    with pytest.raises(ValueError, match="no valid recorded action step"):
        src.chunk_at(0, 9)


def test_actions_are_truncated_to_n_valid(left_zarr):
    """``actions`` itself is trimmed, so a caller physically cannot drip padded steps."""
    path_src = _source(left_zarr)
    chunk = path_src.chunk_at(0, 1)
    assert chunk.actions.shape[0] == chunk.n_valid


# ---------------------------------------------------------------------------
# Unplayable (step-0-masked) frames — real in clean94: 8 of 94 episodes start
# masked and 5 more have mid-episode holes, so these must be SKIPPABLE, not fatal.
# ---------------------------------------------------------------------------
@pytest.fixture()
def masked_zarr(tmp_path):
    """A zarr whose episode 0 has a masked FIRST frame and a masked mid-episode frame.

    Built with the glitch filter DISABLED (``glitch_pre_s``/``glitch_post_s`` = 0 at open
    time) in the tests below where the point is step-0 maskedness alone — a validity hole is
    itself a bridged-glitch signal, so with the real window it also contaminates the
    following second, which is correct but would conflate the two exclusions."""
    path = _write_hl_zarr(tmp_path / "masked.zarr.zip", n_frames=40, n_episodes=2)
    store = zarr.ZipStore(str(path), mode="a")
    root = zarr.group(store=store)
    valid = np.asarray(root["data"]["hl_left_action_valid"][:], dtype=np.float32)
    valid[0, 0] = 0.0        # episode 0 frame 0 unusable (the clean94 8-episode case)
    valid[6, 0] = 0.0        # a mid-episode hole (the sequential-abort case)
    valid[7, 0] = 0.0        # ...two frames wide, so the skip must scan forward
    root["data"]["hl_left_action_valid"][:] = valid
    store.close()
    return path


def _no_glitch_source(path):
    """A source with the glitch WINDOW disabled, isolating the step-0-masked exclusion."""
    from fuji.s1_policy.serving.hl_replay import RecordedEpisodeSource

    return RecordedEpisodeSource(
        path, get_layout("left_only"), glitch_pre_s=0.0, glitch_post_s=0.0
    )


def test_frame_is_usable_reports_step0_maskedness(masked_zarr):
    src = _no_glitch_source(masked_zarr)
    assert not src.frame_is_usable(0, 0)
    assert not src.frame_is_usable(0, 6)
    assert src.frame_is_usable(0, 1)
    assert not src.frame_is_usable(0, -1)                       # out of range is not usable
    assert not src.frame_is_usable(0, src.episode_lengths()[0])


def test_first_usable_frame_skips_a_masked_episode_start(masked_zarr):
    """Without this the operator picks an episode and Predict dies with a raw ValueError."""
    src = _no_glitch_source(masked_zarr)
    assert src.first_usable_frame(0) == 1
    assert src.first_usable_frame(1) == 0  # a normal episode is unaffected


def test_next_usable_frame_scans_over_a_multi_frame_hole(masked_zarr):
    """A sequential playthrough must step OVER a mid-episode hole, not abort the rest."""
    src = _no_glitch_source(masked_zarr)
    assert src.next_usable_frame(0, 6) == 8, "must skip both masked frames 6 and 7"
    assert src.next_usable_frame(0, 5) == 5, "an already-usable frame is returned unchanged"


def test_a_masked_frame_also_contaminates_its_window_by_default(masked_zarr):
    """With the REAL glitch window (the default), a validity hole is a bridged-glitch
    signal, so the frames it contaminates are excluded too — not just the hole itself.
    This is the behavior that keeps replay to the training-clean anchors."""
    src = _source(masked_zarr)          # default window
    lenient = _no_glitch_source(masked_zarr)
    assert lenient.frame_is_usable(0, 1)
    assert not src.frame_is_usable(0, 1), "frame 1's window still overlaps frame 0's hole"
    assert src.first_usable_frame(0) > lenient.first_usable_frame(0)


def test_next_usable_frame_returns_none_when_nothing_is_playable(tmp_path):
    """An entirely masked episode must report 'no playable frame', not raise."""
    path = _write_hl_zarr(tmp_path / "all_masked.zarr.zip", n_frames=40, n_episodes=2)
    store = zarr.ZipStore(str(path), mode="a")
    root = zarr.group(store=store)
    valid = np.asarray(root["data"]["hl_left_action_valid"][:], dtype=np.float32)
    a, b = 0, 20
    valid[a:b, 0] = 0.0
    root["data"]["hl_left_action_valid"][:] = valid
    store.close()
    src = _source(path)
    assert src.first_usable_frame(0) is None
    assert src.next_usable_frame(0, 0) is None
    assert src.first_usable_frame(1) == 0  # the other episode still plays


# ---------------------------------------------------------------------------
# Cursor advance
# ---------------------------------------------------------------------------
def test_advance_tracks_the_action_clock_not_the_frame_count(tmp_path):
    """Advancing by n FRAMES would replay the episode at the wrong speed, so the cursor
    advances by the TIME the executed steps consumed and lands on the nearest camera frame.

    Needs a camera grid that DIFFERS from the action grid to be meaningful, and on the v3
    contract the two deliberately coincide (ACTION_HZ == CAMERA_PUBLISH_HZ). So this test
    builds its own off-grid fixture rather than using ``left_zarr``: the time-vs-count
    distinction still has to hold for a session recorded at another rate, or for one with
    dropped frames, which is exactly why the cursor is time-based in the first place."""
    path = _write_hl_zarr(tmp_path / "offgrid.zarr.zip", n_frames=4 * ACTION_HORIZON,
                          n_episodes=1)
    # Re-stamp the camera grid at a rate that is NOT the action rate. 0.6x is chosen so
    # that for EVERY exec count below, the nearest-in-time frame differs from the frame
    # COUNT — at 0.7x, exec=4 happens to land on frame 4 and the time-vs-count distinction
    # this test exists to prove becomes untestable at that k.
    cam_dt = 0.6 * ACTION_DT
    store = zarr.ZipStore(str(path), mode="a")
    root = zarr.group(store=store)
    n = int(root["data"]["episode_timestamps"].shape[0])
    root["data"]["episode_timestamps"][:] = np.arange(n) * cam_dt
    store.close()

    src = _source(path)
    times = src.episode_times(0)
    dt_cam = float(np.median(np.diff(times)))
    assert not np.isclose(dt_cam, ACTION_DT), "fixture must have a camera grid != action grid"

    start = 0
    for n_steps in (4, 8, 16):
        landed = src.advance_frame(0, start, n_steps)
        # Advance is by the demo time the executed steps actually consumed — (k-1)*dt, since
        # step 0 is the zero-delta anchor (see the dedicated fencepost test below) — and lands
        # on the nearest camera frame, NOT on a frame offset by the step count.
        want_t = times[start] + (n_steps - 1) * ACTION_DT
        best = int(np.argmin(np.abs(times - want_t)))
        assert landed == best, f"exec={n_steps}: landed {landed}, nearest-in-time is {best}"
        # The distinguishing property: a frame-count advance would land on `start + n_steps`.
        if n_steps > 1:
            assert landed != start + n_steps or times[landed] == want_t


def test_advance_always_makes_progress_and_clamps_at_the_end(left_zarr):
    """No stall (an infinite sequential loop) and no run past the episode boundary."""
    src = _source(left_zarr)
    n = src.episode_lengths()[0]
    cursor = 0
    for _ in range(500):
        nxt = src.advance_frame(0, cursor, 4)
        assert nxt > cursor or nxt == n - 1, "cursor must advance or be pinned at the end"
        cursor = nxt
        if cursor == n - 1:
            break
    assert cursor == n - 1
    assert src.advance_frame(0, n - 1, 16) == n - 1  # already at the end: stays
    with pytest.raises(ValueError):
        src.advance_frame(0, 0, 0)


def test_advance_never_crosses_into_the_next_episode(left_zarr):
    """A replay must stop at its own demonstration's end; bleeding into the next episode
    would splice two unrelated motions into one continuous arm trajectory."""
    src = _source(left_zarr)
    for episode in range(src.n_episodes):
        n = src.episode_lengths()[episode]
        start, end = src.episode_bounds(episode)
        assert src.advance_frame(episode, n - 1, 16) == n - 1
        # The real assertion: the returned LOCAL frame maps to an absolute frame that is
        # still inside THIS episode (a `<= n-1` check cannot fail, since advance clamps).
        landed = src.advance_frame(episode, max(n - 2, 0), 16)
        assert start <= start + landed < end


def test_advance_measures_from_the_last_EXECUTED_step_not_the_step_count(left_zarr):
    """Executing ``k`` steps advances the demo by ``(k-1)*ACTION_DT``, not ``k*ACTION_DT``.

    Step 0 is the anchor's own pose (a zero delta) and commands no motion, so the arm's
    realized demo time after a k-step chunk is ``t0 + (k-1)*dt`` — which is also where the
    executor commits its anchor. Advancing by ``k`` leaves a one-step gap that is never
    commanded and SPATIALLY TRUNCATES the replay (measured on clean94: only ~70% of the
    demonstration's path length at the default 4 steps/chunk). This pins the execution
    contract, not the implementation's own formula."""
    src = _source(left_zarr)
    times = src.episode_times(0)
    for n_steps in (2, 4, 8, 16):
        landed = src.advance_frame(0, 0, n_steps)
        realized_t = times[0] + (n_steps - 1) * ACTION_DT  # where the ARM actually got to
        nearest = int(np.argmin(np.abs(times - realized_t)))
        assert landed == nearest, (
            f"exec={n_steps}: landed on frame {landed} (t={times[landed]:.4f}), but the arm "
            f"only reached t={realized_t:.4f}, whose nearest frame is {nearest}"
        )


def test_single_step_execution_still_makes_progress(left_zarr):
    """A 1-step chunk advances the demo by 0 s, so only the min-progress clamp moves it.
    Without that clamp a 1-step execution would stall the replay forever."""
    src = _source(left_zarr)
    assert src.advance_frame(0, 0, 1) == 1


# ---------------------------------------------------------------------------
# Whole-episode OPEN-LOOP plan (episode_plan) — decode the demo ONCE, drip straight.
# The property that matters: it must decode to the SAME curve the per-chunk path would,
# so the only difference from the sequential replay is the ABSENCE of a per-chunk
# re-anchor (the accumulation source), not a different trajectory.
# ---------------------------------------------------------------------------
def test_episode_plan_spans_the_whole_episode_from_the_start_frame(left_zarr):
    """A clean episode plans from its start frame all the way to the end, untruncated."""
    src = _source(left_zarr)
    n = src.episode_lengths()[0]
    plan = src.episode_plan(0, 0)
    assert plan.n_frames == n
    assert plan.actions.shape == (n, get_layout("left_only").action_dim)
    assert plan.times_s.shape == (n,)
    assert not plan.truncated
    # Step 0 is the anchor frame's own pose ⇒ an exact zero delta (same DECISION-T0 property
    # chunk_at has), so an open-loop drip starts exactly on the seeded anchor with no jump.
    s = get_layout("left_only").slots("left")
    assert np.allclose(plan.actions[0, s.pos], 0.0, atol=1e-12)
    assert np.allclose(rot6d_to_rotmat(plan.actions[0, s.rot6d]), np.eye(3), atol=1e-7)


def test_episode_plan_decodes_to_the_same_curve_as_the_drawn_path(left_zarr):
    """The plan, decoded ONCE against one anchor, IS the green ``episode_path_relative`` curve.

    This is the load-bearing equivalence: the open-loop command is a single rigid placement of
    the demonstration, identical to what the drawn path shows — so the executed-vs-recorded
    sheet compares the arm against the exact curve it was commanded, with no per-chunk
    re-anchor in between. Checked against a ROTATED anchor (an identity anchor cannot tell a
    body-frame transport from a world one — same reason as the drawn-path test)."""
    from fuji.s1_policy.serving.hl_online import decode_chunk

    src = _source(left_zarr)
    layout = get_layout("left_only")
    start = 2
    anchor_pos = np.array([[0.31, -0.2, 0.88]])
    anchor_rot = rotmat_to_rot6d(
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # +90° about +z
    ).reshape(1, 6)

    plan = src.episode_plan(0, start)
    decoded = decode_chunk(
        plan.actions, layout, anchor_pos, anchor_rot, force_enabled=False, gate_n=0.0
    ).role("left")

    # The drawn path relative to the SAME start frame, transported by the SAME anchor.
    rel_pos, rel_rot6d = src.episode_path_relative(0, start, "left")
    drawn_pos, drawn_rot6d = delta_ee_to_absolute(
        rel_pos[start:], rel_rot6d[start:], anchor_pos[0], anchor_rot[0]
    )
    assert np.allclose(decoded.position, drawn_pos, atol=1e-12)
    assert np.allclose(decoded.rot6d, drawn_rot6d, atol=1e-10)
    # And it starts exactly on the anchor (the single clutch), so no snap on the first step.
    assert np.allclose(decoded.position[0], anchor_pos[0], atol=1e-12)


def test_episode_plan_pose_columns_are_the_camera_grid_demonstration(left_zarr):
    """The plan's pose columns ARE ``episode_path_relative`` from the start frame — the raw
    camera-grid demonstration, NOT the per-frame action-grid future.

    This is the choice that makes the open-loop error honest: the per-frame
    ``hl_*_action_pose[f]`` is the converter's uniform-resample of the future, which drifts from
    the raw camera poses over a 50-step window (measured up to ~23 mm on the 2026.08.10 set), so
    concatenating action chunks would command a curve that differs from the drawn/reference
    demonstration by that resample artefact. Reading the camera grid keeps the command, the
    green line, and the comparison reference the SAME curve, so the sheet reports servo error
    only. Pinned to exact equality (they are the same array, relativized identically)."""
    src = _source(left_zarr)
    layout = get_layout("left_only")
    s = layout.slots("left")
    start = 3
    plan = src.episode_plan(0, start)
    rel_pos, rel_rot6d = src.episode_path_relative(0, start, "left")
    assert np.array_equal(plan.actions[:, s.pos], rel_pos[start:])
    assert np.array_equal(plan.actions[:, s.rot6d], rel_rot6d[start:])


def test_episode_plan_carries_the_recorded_gripper_and_force(left_zarr):
    """The plan's gripper column is a NORMALIZED fraction and force is raw physical — the same
    conventions ``chunk_at`` uses, per-frame (step 0 of each commanded frame)."""
    src = _source(left_zarr)
    layout = get_layout("left_only")
    s = layout.slots("left")
    plan = src.episode_plan(0, 0)
    # Fixture stores 0.0475 m jaw everywhere; normalized by 0.095 m ⇒ 0.5.
    assert np.allclose(plan.actions[:, s.gripper], 0.0475 / GRIPPER_WIDTH_REF_M, atol=1e-6)
    # Force step 0 per frame matches chunk_at's step-0 force at the same frame (physical units).
    for k in (0, 5, 9):
        assert np.allclose(
            plan.actions[k, s.force], src.chunk_at(0, k).actions[0, s.force], atol=1e-9
        )


def test_episode_plan_truncates_at_a_glitch_hole_rather_than_seeking_over_it(tmp_path):
    """An open-loop plan CANNOT skip a masked frame (the frames between would be a commanded
    teleport across the tracker excursion), so it stops at the last clean frame and flags it.

    This is the deliberate difference from the sequential loop, which seeks past the hole with
    ``next_usable_frame``. Built with the glitch window disabled to isolate step-0 maskedness."""
    path = _write_hl_zarr(tmp_path / "hole.zarr.zip", n_frames=60, n_episodes=1)
    store = zarr.ZipStore(str(path), mode="a")
    root = zarr.group(store=store)
    valid = np.asarray(root["data"]["hl_left_action_valid"][:], dtype=np.float32)
    valid[10, 0] = 0.0  # a masked frame mid-episode ⇒ the plan must stop at frame 9
    root["data"]["hl_left_action_valid"][:] = valid
    store.close()

    src = _no_glitch_source(path)
    plan = src.episode_plan(0, 0)
    assert plan.truncated
    # The plan stops at the first UNUSABLE frame, which is the exact ``frame_is_usable`` gate the
    # sequential loop skips on — not necessarily frame 10 itself: a frame whose action window
    # reaches into frame 10 is already unusable (even with the glitch window disabled, a masked
    # step inside the horizon fails step-0-less validity), so the run may end a frame or two
    # earlier. Deriving the boundary from the gate pins "stops at the hole, does not seek over
    # it" without hardcoding the horizon overlap.
    expected = 0
    while src.frame_is_usable(0, expected):
        expected += 1
    assert expected <= 10, "the masked frame 10 must bound the usable run"
    assert plan.n_frames == expected == plan.actions.shape[0]
    # The decisive property: it did NOT seek past the hole to the frames after 10.
    assert plan.n_frames < src.episode_lengths()[0]


def test_episode_plan_refuses_an_unusable_start_frame(masked_zarr):
    """Planning from a masked/glitch start frame is a caller error (seek first), not a silent
    empty plan — mirrors ``chunk_at`` refusing a fully-masked frame."""
    src = _no_glitch_source(masked_zarr)
    with pytest.raises(ValueError, match="not usable"):
        src.episode_plan(0, 0)  # frame 0 is masked in this fixture
    # ...but the first usable frame plans fine.
    first = src.first_usable_frame(0)
    assert first is not None and src.episode_plan(0, first).n_frames >= 1


def test_episode_plan_out_of_range_start_raises(left_zarr):
    src = _source(left_zarr)
    with pytest.raises(IndexError):
        src.episode_plan(0, src.episode_lengths()[0])
    with pytest.raises(IndexError):
        src.episode_plan(0, -1)


# ---------------------------------------------------------------------------
# The drawn path
# ---------------------------------------------------------------------------
def test_episode_path_is_relative_to_the_named_frame_and_meets_the_chunk(left_zarr):
    """The green recorded path and the executed chunk must be transported by the SAME
    anchor, so they cannot disagree about the frame (the bug that made the offline viewer
    read tilted). Pinned by requiring the path's reference row to coincide exactly with the
    decoded chunk's step 0."""
    from fuji.s1_policy.serving.hl_online import decode_chunk

    src = _source(left_zarr)
    layout = get_layout("left_only")
    ref = 5
    rel_pos, rel_rot6d = src.episode_path_relative(0, ref, "left")
    assert np.allclose(rel_pos[ref], 0.0, atol=1e-12)
    assert np.allclose(rot6d_to_rotmat(rel_rot6d[ref]), np.eye(3), atol=1e-7)
    assert rel_pos.shape[0] == src.episode_lengths()[0]

    # A ROTATED anchor, deliberately: with an identity anchor rotation a body-frame and a
    # world-frame transport are indistinguishable, so an identity-anchor test cannot police
    # this contract at all (see test_drawn_path_uses_the_same_transport_as_the_command).
    anchor_pos = np.array([[0.31, -0.2, 0.88]])
    anchor_rot = rotmat_to_rot6d(
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # +90° about +z
    ).reshape(1, 6)
    path, _ = delta_ee_to_absolute(rel_pos, rel_rot6d, anchor_pos[0], anchor_rot[0])
    chunk = src.chunk_at(0, ref)
    decoded = decode_chunk(
        chunk.actions, layout, anchor_pos, anchor_rot, force_enabled=False, gate_n=0.0
    ).role("left")
    assert np.allclose(decoded.position[0], path[ref], atol=1e-12)
    assert np.allclose(path[ref], anchor_pos[0], atol=1e-12)


def test_drawn_path_uses_the_same_transport_as_the_command(left_zarr):
    """The GREEN drawn path and the COMMANDED chunk must be the same curve in the same frame.

    THE CONTRACT THIS PINS, and why it needs a ROTATED anchor. Replaying a demo re-anchored onto
    the robot's wrist rigidly REORIENTS it by ``R_wrist_robot @ R_wrist_handheld^T`` — measured
    106.9° mean (98.5–119.0) on clean94, because the G1's ready wrist is 44° nose-down while the
    operator held the tool level. That reorientation is CORRECT and deliberate (see
    ``hl_replay._body_vs_world_transport``: it is what preserves the tool's attack angle, keeps
    the lighthouse world cancelling, and keeps the chunk equal to the policy's training target).

    What must never happen is the two DISAGREEING — e.g. "fixing" the drawn path to a
    world-frame transport so the picture looks like the recording while the arm still executes
    the body-frame motion. The green line is the operator's datum for judging the motion, so a
    prettier line over different motion is worse than an honestly reoriented one. Both are
    checked here against a strongly rotated anchor, where the two conventions differ most."""
    from fuji.s1_policy.serving.hl_online import decode_chunk

    src = _source(left_zarr)
    layout = get_layout("left_only")

    for name, rmat in (
        ("+90° yaw", np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])),
        ("180° roll", np.diag([1.0, -1.0, -1.0])),
    ):
        anchor_pos = np.array([[0.10, 0.21, 0.74]])
        anchor_rot = rotmat_to_rot6d(rmat).reshape(1, 6)
        ref = 3
        rel_pos, rel_rot6d = src.episode_path_relative(0, ref, "left")
        drawn, _ = delta_ee_to_absolute(rel_pos, rel_rot6d, anchor_pos[0], anchor_rot[0])
        chunk = src.chunk_at(0, ref)
        decoded = decode_chunk(
            chunk.actions, layout, anchor_pos, anchor_rot, force_enabled=False, gate_n=0.0
        ).role("left")

        # Step 0 of the command IS the drawn path's reference vertex IS the measured anchor.
        assert np.allclose(drawn[ref], anchor_pos[0], atol=1e-12), f"{name}: path left the anchor"
        assert np.allclose(decoded.position[0], drawn[ref], atol=1e-12), (
            f"{name}: the drawn path and the commanded chunk disagree at the cursor — the green "
            f"line would be showing a motion the arm is not making"
        )
        # And the whole overlapping span agrees, not just step 0 (the action grid is 16 Hz and
        # the drawn path is the ~15 Hz camera grid, so compare the shared reference-relative
        # displacement direction rather than index-for-index).
        assert np.allclose(
            np.linalg.norm(decoded.position[0] - anchor_pos[0]), 0.0, atol=1e-12
        ), f"{name}: commanded step 0 is not on the measured anchor — the arm would snap"


def test_unknown_role_and_bad_ref_frame_raise(left_zarr):
    src = _source(left_zarr)
    with pytest.raises(KeyError):
        src.episode_path_relative(0, 0, "right")
    with pytest.raises(IndexError):
        src.episode_path_relative(0, src.episode_lengths()[0], "left")


def test_image_at_returns_the_recorded_frame(left_zarr):
    """The operator's view during the demonstration, undecorated (not the model input)."""
    src = _source(left_zarr)
    root = zarr.open(left_zarr, "r")
    start, _end = src.episode_bounds(1)
    got = src.image_at(1, 2)
    assert got.dtype == np.uint8 and got.ndim == 3 and got.shape[2] == 3
    assert np.array_equal(got, np.asarray(root["data"]["camera0_rgb"][start + 2]))
    with pytest.raises(IndexError):
        src.image_at(1, src.episode_lengths()[1])


def test_legacy_layout_is_refused(left_zarr):
    from fuji.s1_policy.serving.hl_replay import RecordedEpisodeSource

    with pytest.raises(ValueError, match="legacy32"):
        RecordedEpisodeSource(left_zarr, get_layout("legacy32"))


# ---------------------------------------------------------------------------
# Glitch windows — the frames a tracker excursion contaminates. Real in clean94:
# without this filter, pressing Play on episode 27 commands a 153 mm step inside
# one 62.5 ms wire tick (2.45 m/s) — the arm is told to teleport.
# ---------------------------------------------------------------------------
def test_glitch_window_frames_are_not_usable(tmp_path):
    """A frame whose CHUNK contains a tracker glitch must be excluded even when its own
    step 0 is unmasked — masking protects the glitch's centre, not its neighbours."""
    n_frames = 4 * ACTION_HORIZON   # room for a real "far away" frame on both sides
    glitch_at = 2 * ACTION_HORIZON
    path = _write_hl_zarr(tmp_path / "glitch.zarr.zip", n_frames=n_frames, n_episodes=1)
    store = zarr.ZipStore(str(path), mode="a")
    root = zarr.group(store=store)
    glitch = np.zeros((n_frames, 1), dtype=np.float32)
    glitch[glitch_at] = 1.0  # a single flagged tracker excursion, mid-episode
    root["data"].create_dataset("robot0_pose_glitch", data=glitch, overwrite=True)
    store.close()

    src = _source(path)
    assert not src.frame_is_usable(0, glitch_at), "the glitch frame itself must be excluded"
    # Its neighbours are contaminated too: their windows overlap the excursion.
    assert not src.frame_is_usable(0, glitch_at - 1)
    assert not src.frame_is_usable(0, glitch_at + 1)
    # ...and far-away frames are unaffected (the filter is a window, not a whole-episode veto).
    assert src.frame_is_usable(0, 0)
    assert src.frame_is_usable(0, n_frames - 1)
    # next_usable_frame walks clear of the whole contaminated span.
    landed = src.next_usable_frame(0, glitch_at - 1)
    assert landed is not None and landed > glitch_at + 1


def test_glitch_exclusion_matches_the_training_filter(tmp_path):
    """Replay must offer EXACTLY the anchors ``HLDataset`` trains on — otherwise the
    demo-vs-policy comparison runs the policy on frames it never saw."""
    from fuji.s1_policy.data.hl_dataset import HLDataset

    path = _write_hl_zarr(tmp_path / "match.zarr.zip", n_frames=60, n_episodes=2)
    store = zarr.ZipStore(str(path), mode="a")
    root = zarr.group(store=store)
    glitch = np.zeros((60, 1), dtype=np.float32)
    glitch[[12, 44]] = 1.0
    root["data"].create_dataset("robot0_pose_glitch", data=glitch, overwrite=True)
    valid = np.asarray(root["data"]["hl_left_action_valid"][:], dtype=np.float32)
    valid[20, 5:] = 0.0  # a bridged glitch: no pose_glitch flag, only a validity hole
    root["data"]["hl_left_action_valid"][:] = valid
    store.close()

    src = _source(path)
    ds = HLDataset(
        path, mode="left_only", image_key="camera0_rgb", image_mean=0.5, image_std=0.5,
        drop_invalid_force_history=False, drop_glitch_windows=True,
        # To=1 is the v3 contract AND what RecordedEpisodeSource assumes, so the anchor
        # sets being compared are built on the same observation shape. (To=2 would also
        # need glitch_pre_s >= 0.25 s, i.e. a DIFFERENT window than replay's.)
        obs_history_len=1, require_episode_ends=True,
    )
    try:
        trained = {int(frame) for _fi, frame in ds._index}
        usable = {
            start + lf
            for episode in range(src.n_episodes)
            for start, _e in [src.episode_bounds(episode)]
            for lf in range(src.episode_lengths()[episode])
            if src.frame_is_usable(episode, lf)
        }
        assert usable - trained == set(), (
            "replay would execute frames the training filter excluded: "
            f"{sorted(usable - trained)}"
        )
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Provenance gates — --replay-zarr takes a free-form path, and the datasets
# directory holds sessions that look interchangeable but are not.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"hl_action_first_step": "t0_plus_dt"}, "t0-inclusive"),
        ({"hl_action_first_step": None}, "t0-inclusive"),
        ({"gripper_width_open_m": 0.085}, "full-open gripper width"),
        ({"hl_windows": False}, "no HL-v2 windows"),
        ({"hl_pose_frame": "relative"}, "hl_pose_frame"),
        ({"hl_force_frame": "aligned_world"}, "hl_force_frame"),
        ({"hl_action_hz": 10.0}, "hl_action_hz"),
        ({"hl_action_horizon": 8}, "hl_action_horizon"),
    ],
)
def test_contract_mismatch_is_refused(tmp_path, override, match):
    """Each of these is INVISIBLE in the array values but changes what the numbers mean.

    ``hl_action_first_step`` is the load-bearing one: a pre-v2 file excluded t0 with
    identical horizon/span attrs, so replaying it puts step 0 off the live anchor — an
    instantaneous commanded jump on the first dripped step of every chunk."""
    name = "bad_" + "_".join(str(k) for k in override) + ".zarr.zip"
    path = _write_hl_zarr(tmp_path / name, attrs_override=override)
    with pytest.raises(ValueError, match=match):
        _source(path)


def test_a_faithful_file_is_accepted(left_zarr):
    """The gates must not reject the schema the converter actually emits."""
    src = _source(left_zarr)
    assert src.n_episodes >= 1
    assert src.chunk_at(0, 0).n_valid >= 1


# ---------------------------------------------------------------------------
# HORIZON: `stored >= wanted`, mirroring HLDataset._check_file. Serving an H=30
# checkpoint (FUJI_ACTION_HORIZON=30) must still replay the 50-step session it was
# TRAINED on — i.e. the file its own hl_data.zarr_paths points at.
# ---------------------------------------------------------------------------
def test_longer_stored_horizon_replays_as_its_prefix(tmp_path):
    """A wider file yields the SAME chunk an exactly-sized one does.

    Asserted on values, not shape: the truncation must take the FIRST rows, because any
    other slice re-phases the whole chunk against the execution grid while every shape
    still matches — and a re-phased chunk is dripped to a real arm."""
    wide = _write_hl_zarr(tmp_path / "wide.zarr.zip", store_horizon=ACTION_HORIZON + 17)
    exact = _write_hl_zarr(tmp_path / "exact.zarr.zip")
    c_w, c_e = _source(wide).chunk_at(0, 3), _source(exact).chunk_at(0, 3)
    assert c_w.actions.shape[0] <= ACTION_HORIZON
    assert c_w.n_valid == c_e.n_valid
    np.testing.assert_array_equal(c_w.actions, c_e.actions)


def test_shorter_stored_horizon_is_refused_for_replay(tmp_path):
    """The prefix relation runs ONE way: fewer stored steps cannot be extended."""
    p = _write_hl_zarr(tmp_path / "short.zarr.zip", store_horizon=ACTION_HORIZON - 5)
    with pytest.raises(ValueError, match="at least"):
        _source(p)


def test_replay_prefix_requires_an_identical_step_dt(tmp_path):
    """A longer horizon at a DIFFERENT dt is a rescaled grid, not a superset.

    ``hl_action_hz`` is stamped correctly here, so only the file's own span/horizon
    betrays the 1.5x-stretched grid — exactly the case a count-only check would accept
    and then drip mis-timed against the 30 Hz execution grid."""
    wide = ACTION_HORIZON + 10
    p = _write_hl_zarr(
        tmp_path / "slowgrid.zarr.zip",
        store_horizon=wide,
        attrs_override={"hl_action_span_s": wide * ACTION_DT * 1.5},
    )
    with pytest.raises(ValueError, match="step dt"):
        _source(p)


def test_glitch_keep_mask_survives_a_wider_stored_chunk(tmp_path):
    """The glitch filter ANDs the stored validity column against an ACTION_HORIZON-wide
    grid, so an un-truncated read raises a broadcast error rather than mis-filtering.
    The mask must also AGREE with the exactly-sized file's, since the extra stored steps
    lie beyond the window this build executes."""
    wide = _source(_write_hl_zarr(tmp_path / "wide.zarr.zip", store_horizon=ACTION_HORIZON + 17))
    exact = _source(_write_hl_zarr(tmp_path / "exact.zarr.zip"))
    got = [wide.frame_is_usable(0, i) for i in range(wide.episode_lengths()[0])]
    want = [exact.frame_is_usable(0, i) for i in range(exact.episode_lengths()[0])]
    assert got == want
    assert any(want), "fixture produced no usable frame — the comparison would be vacuous"


# ---------------------------------------------------------------------------
# MODEL-FREE replay: the planner half with no policy loaded (run_hl_replay_data.sh).
# ---------------------------------------------------------------------------
def test_replay_only_model_carries_the_layout_and_no_policy():
    """``replay_only_model`` is the "no checkpoint, no GPU" planner core.

    Replaying recorded data needs the LAYOUT (to slice the action vector) and the anchor
    bookkeeping, but no policy — loading one would pin a GPU and several GB of VRAM to run a
    data playback. It must still be the SAME dataclass the real loader returns, so
    ``scripts/hl_serving.py`` keeps one code path for the wire/gate/drip/clutch."""
    from fuji.s1_policy.serving.hl_online import LoadedHLModel, replay_only_model

    loaded = replay_only_model("left_only")
    assert isinstance(loaded, LoadedHLModel)
    assert loaded.model is None  # THE point: no torch module was built
    assert loaded.layout.mode == "left_only"
    assert loaded.layout.roles == ("left",)
    assert "left" in loaded.layout.force_roles  # the planner's LEFT-canonical guard passes
    assert loaded.norm_stats == {}  # nothing normalizes without a model
    assert loaded.device.type == "cpu"  # never touches CUDA
    assert loaded.algorithm == "replay"  # not 'flow'/'diffusion': there is no sampler


def test_replay_only_planner_refuses_the_model_path_loudly():
    """A model-free planner must REFUSE ``infer_chunk`` with a readable error.

    Without this the ``None`` model surfaces as an opaque ``AttributeError`` from inside the
    batch build, which reads like a torch bug rather than "you launched without a checkpoint"."""
    from fuji.s1_policy.serving.hl_online import LivePlanner, replay_only_model

    planner = LivePlanner(replay_only_model("left_only"))
    with pytest.raises(RuntimeError, match="no policy"):
        planner.infer_chunk(seed=0, force_enabled=False, gate_n=0.0)


def test_replay_only_planner_still_decodes_a_recorded_chunk(left_zarr):
    """The whole point: with NO policy, a recorded chunk still decodes against the anchor.

    This is the model-free path end to end — seed an anchor, read a chunk off the zarr, decode
    it with the SAME ``decode_chunk`` the model path uses — proving the missing model costs
    nothing but the model itself."""
    from fuji.s1_policy.serving.hl_online import (
        LivePlanner,
        decode_chunk,
        replay_only_model,
    )

    loaded = replay_only_model("left_only")
    planner = LivePlanner(loaded)
    anchor_pos = np.array([[0.3, 0.2, 0.9]])
    anchor_rot6d = np.array([rotmat_to_rot6d(np.eye(3))])
    planner.seed_anchor(anchor_pos, anchor_rot6d)

    src = _source(left_zarr)
    chunk = src.chunk_at(0, 0)
    got_pos, got_rot = planner.snapshot_anchor()
    np.testing.assert_allclose(got_pos, anchor_pos)
    result = decode_chunk(
        chunk.actions, loaded.layout, got_pos, got_rot, force_enabled=False, gate_n=0.0
    )
    left = result.role("left")
    assert left.position.shape == (chunk.n_valid, 3)
    # Step 0 is the exact zero delta (DECISION-T0), so it lands ON the anchor: that is what
    # makes replay start from wherever the arm already is instead of jumping.
    np.testing.assert_allclose(left.position[0], anchor_pos[0], atol=1e-12)
    assert np.isfinite(left.position).all()


# ---------------------------------------------------------------------------
# PLAYBACK CONTINUITY: the render cache must not stall the arm between chunks.
# ---------------------------------------------------------------------------
def test_render_cache_carries_the_chunk_length_independently_of_the_preview_mesh():
    """REGRESSION (operator: "play episode 还是一顿一顿" — moves, stops, moves).

    ``_solve_and_cache`` used to run the mink ``ThreePointIK`` ACTION_HORIZON times, TWICE per
    executed chunk (preview + executed). Measured **6.14 ms per solve**, so 50 steps x 2 =
    **614 ms of arm-idle time** against 533 ms of actual motion at exec-steps=16 — a **46% duty
    cycle**. The arm genuinely stopped between chunks, and NOT because of the low level, the
    slew limiter, or the action grid: the planner was solving a preview mesh that, on a RECORD=1
    rig, the render loop discards on every tick (the live joints win) and that OFFLINE is
    deliberately frozen. All 100 solves were thrown away in both modes anyone runs.

    The fix makes ``cfgs`` OPTIONAL (None unless the fallback mesh is really rendered), which
    moves a trap: the render loop keyed the yellow path, the gripper/force charts AND the
    animation cursor off ``cfgs.shape[0]``/``cfgs is not None``, so skipping the IK would have
    silently blanked all of them. The cache therefore carries ``n_steps`` separately. This test
    pins that contract: with cfgs ABSENT, the chunk length and every per-step render array must
    still be present and indexable across the whole horizon.

    Kept in the replay suite because the sequential Play loop is what exposed it; the numbers
    above are why it is worth a test rather than a comment."""
    S = ACTION_HORIZON

    # The two collaborating shapes, exactly as hl_serving builds/consumes them.
    def cache(*, cfgs_present: bool) -> dict:
        return {
            "cfgs": (np.zeros((S, 20)) if cfgs_present else None),
            "wrist_pos": np.zeros((S, 3)), "wrist_wxyz": np.zeros((S, 4)),
            "path": np.arange(3 * S, dtype=float).reshape(S, 3),
            "force_pred": np.zeros((S, 6)), "grip_m": np.full(S, 0.05),
            "dirty": True, "n_chunks": 1, "n_steps": S,
        }

    for cfgs_present in (False, True):
        latest = cache(cfgs_present=cfgs_present)
        # The render loop's dirty branch: gated on n_steps, NEVER on cfgs.
        assert latest["n_steps"] == S
        assert latest["path"].shape[0] >= 2, "yellow path must survive a skipped IK solve"
        assert latest["grip_m"].shape[0] >= 2, "gripper chart must survive a skipped IK solve"
        assert latest["force_pred"].shape[0] >= 2, "force chart must survive a skipped IK solve"
        # ...and the animation cursor must walk the FULL horizon with every per-step array
        # indexable — the mesh cfgs being the only optional one.
        for i in range(latest["n_steps"]):
            assert latest["wrist_pos"][i].shape == (3,)
            assert latest["wrist_wxyz"][i].shape == (4,)
            assert np.isfinite(latest["force_pred"][min(i, S - 1), 0:3]).all()
            if latest["cfgs"] is not None:
                assert latest["cfgs"][i].shape == (20,)


def test_the_fallback_mesh_predicate_mirrors_the_render_loops_condition():
    """``_fallback_mesh_needed`` gates ACTION_HORIZON IK solves, so it must agree with the
    render loop's ``if not live_posed and not args.offline`` for every case — a disagreement
    either re-introduces the 614 ms stall (too eager) or blanks the RECORD-off preview mesh
    (too lazy).

    The predicate is a closure over ``args``/``subs``/``live_mesh_ok`` inside ``_run_viser``, so
    the TRUTH TABLE is pinned here rather than the closure itself. Encoded deliberately: a
    STALE RecordTap must read as NEEDED, because "low level up but RECORD=1 off" is exactly the
    mode the fallback mesh exists for, and a stale frame is indistinguishable from it."""
    def needed(*, offline: bool, live_mesh_ok: bool, frame_age_s: float | None) -> bool:
        if offline:
            return False
        if not live_mesh_ok:
            return True
        if frame_age_s is None or frame_age_s > 1.0:
            return True
        return False

    # OFFLINE: mesh frozen at the ready pose → never solve (this is also the mode where the
    # 3 triads must stay put, so an animated preview would be actively wrong).
    assert needed(offline=True, live_mesh_ok=True, frame_age_s=0.01) is False
    assert needed(offline=True, live_mesh_ok=False, frame_age_s=None) is False
    # RECORD=1 rig with fresh joints: the live mesh overrides cfgs on EVERY tick → never solve.
    assert needed(offline=False, live_mesh_ok=True, frame_age_s=0.01) is False
    # The three degraded cases the fallback mesh was written for → SOLVE.
    assert needed(offline=False, live_mesh_ok=True, frame_age_s=None) is True   # RECORD off
    assert needed(offline=False, live_mesh_ok=True, frame_age_s=5.0) is True    # went stale
    assert needed(offline=False, live_mesh_ok=False, frame_age_s=0.01) is True  # URDF mismatch


def test_the_recorded_gripper_is_exposed_in_metres_on_the_camera_grid(left_zarr):
    """``episode_gripper_m`` feeds the comparison sheet's jaw panel, which is read against the
    physical 43-94 mm the operator sees on the rig — so it must be the stored METRES, NOT the
    normalized fraction ``chunk_at`` puts in the action vector. Returning the fraction here
    would draw a 0.05-1.0 curve labelled "mm" and make a fully-open jaw look nearly shut, the
    same unit confusion the action-column check already guards.

    Camera grid (one entry per frame), matching ``episode_path_relative``, so the sheet's jaw
    row shares the clock its position rows are plotted on."""
    src = _source(left_zarr)
    root = zarr.open(str(left_zarr), "r")
    for ep in range(src.n_episodes):
        a, b = src.episode_bounds(ep)
        got = src.episode_gripper_m(ep, "left")
        assert got is not None and got.shape == (b - a,), "one entry per camera frame"
        assert np.allclose(got, np.asarray(root["data/robot0_gripper_width"][a:b]).reshape(-1)), \
            "the stored metres, unconverted"
        assert got.shape[0] == src.episode_times(ep).shape[0], "same clock as the drawn path"


def test_a_role_without_a_gripper_column_reports_none_rather_than_raising(left_zarr):
    """The head role has no jaw, and a caller asking for one must get a clean None so the sheet
    simply omits the panel — a KeyError here would surface as a failed SAVE on a rig."""
    src = _source(left_zarr)
    assert src.episode_gripper_m(0, "head") is None


# ---------------------------------------------------------------------------
# 2026-09-19: what the FIVE-POINT page needed from the source — per-step validity, a second
# camera view, the O(1) usability cache, and the operator-facing helpers
# ---------------------------------------------------------------------------
def _decoded(rc_actions, mode="left_only", h=None):
    from fuji.s1_policy.serving.hl_online import decode_chunk

    lay = get_layout(mode)
    n = len(lay.roles)
    anchor_pos = np.zeros((n, 3))
    anchor_rot6d = np.tile(rotmat_to_rot6d(np.eye(3)), (n, 1))
    return decode_chunk(np.asarray(rc_actions, dtype=np.float64), lay, anchor_pos, anchor_rot6d,
                        force_enabled=True, gate_n=0.0)


def test_chunk_at_carries_per_step_force_and_gripper_validity_as_the_file_has_it(left_zarr):
    src = _source(left_zarr)
    rc = src.chunk_at(0, 3)
    assert set(rc.force_step_valid) == {"left"} and set(rc.gripper_step_valid) == {"left"}
    assert rc.force_step_valid["left"].shape == (rc.n_valid,) and rc.force_step_valid["left"].dtype == bool
    assert rc.gripper_step_valid["left"].shape == (rc.n_valid,)
    fv = np.asarray(zarr.open(left_zarr, "r")["data"]["hl_left_action_force_valid"][3], dtype=np.float32)
    np.testing.assert_array_equal(rc.force_step_valid["left"], fv[: rc.n_valid] > 0.5)
    assert rc.force_valid and rc.gripper_valid          # step 0's bit, as before


def test_episode_plan_carries_each_commanded_frames_own_step0_validity(tmp_path):
    path = _write_hl_zarr(tmp_path / "plan_valid.zarr.zip", n_frames=60, n_episodes=1)
    store = zarr.ZipStore(str(path), mode="a")
    root = zarr.group(store=store)
    fv = np.asarray(root["data"]["hl_left_action_force_valid"][:], dtype=np.float32)
    fv[4, 0] = 0.0                                       # frame 4's wrench unmeasured (pre-tare)
    root["data"]["hl_left_action_force_valid"][:] = fv
    gv = np.asarray(root["data"]["hl_left_action_gripper_valid"][:], dtype=np.float32)
    gv[2, 0] = 0.0                                       # frame 2's jaw unmeasured
    root["data"]["hl_left_action_gripper_valid"][:] = gv
    store.close()
    src = _no_glitch_source(path)
    plan = src.episode_plan(0, 0)
    assert plan.force_step_valid["left"].shape == (plan.n_frames,)
    assert not plan.force_step_valid["left"][4] and plan.force_step_valid["left"][[0, 1, 2, 3, 5]].all()
    assert not plan.gripper_step_valid["left"][2] and plan.gripper_step_valid["left"][[0, 1, 3]].all()
    assert plan.force_valid and plan.gripper_valid       # frame 0's bits


def test_apply_step_validity_is_the_identity_when_everything_was_measured(left_zarr):
    from fuji.s1_policy.serving.hl_replay import apply_step_validity

    src = _source(left_zarr)
    rc = src.chunk_at(0, 2)
    res = _decoded(rc.actions)
    out, note = apply_step_validity(res, rc.force_step_valid, rc.gripper_step_valid)
    assert out is res, "a fully-measured chunk must come back as the SAME object"
    assert note == {"force": {}, "gripper": {}, "gripper_dropped": []}
    same, _ = apply_step_validity(res, None, None)        # a model chunk passes nothing
    assert same is res


def test_apply_step_validity_zeroes_unmeasured_wrench_rows_and_holds_the_jaw(left_zarr):
    from fuji.s1_policy.serving.hl_replay import apply_step_validity

    src = _source(left_zarr)
    rc = src.chunk_at(0, 2)
    res = _decoded(rc.actions)
    n = rc.n_valid
    fsv = {"left": np.ones(n, dtype=bool)}
    fsv["left"][[0, 5]] = False                          # pre-tare rows
    gsv = {"left": np.ones(n, dtype=bool)}
    gsv["left"][[0, 1, 7]] = False                       # dead encoder at the start and once later
    grip_before = res.role("left").gripper_m.copy()
    out, note = apply_step_validity(res, fsv, gsv)
    assert out is not res
    lc = out.role("left")
    assert np.all(lc.wrench_pred[[0, 5]] == 0.0) and np.all(lc.wrench_cmd[[0, 5]] == 0.0)
    assert np.all(lc.wrench_pred[1:5] != 0.0), "measured rows untouched"
    np.testing.assert_allclose(lc.position, res.role("left").position)   # the pose is never touched
    # the jaw: rows 0,1 take the first MEASURED width (row 2); row 7 holds row 6
    assert lc.gripper_m[0] == grip_before[2] and lc.gripper_m[1] == grip_before[2]
    assert lc.gripper_m[7] == grip_before[6]
    assert np.isfinite(lc.gripper_m).all(), "never NaN — the ghost and the drip both read it"
    assert note["force"] == {"left": 2} and note["gripper"] == {"left": 3} and note["gripper_dropped"] == []


def test_apply_step_validity_drops_a_jaw_that_was_never_measured_in_the_chunk(left_zarr):
    from fuji.s1_policy.serving.hl_replay import apply_step_validity

    src = _source(left_zarr)
    rc = src.chunk_at(0, 2)
    res = _decoded(rc.actions)
    gsv = {"left": np.zeros(rc.n_valid, dtype=bool)}
    out, note = apply_step_validity(res, {}, gsv)
    assert out.role("left").gripper_m is None, "no measured width at all -> 'no jaw on this role' to every consumer"
    assert note["gripper_dropped"] == ["left"] and note["gripper"] == {}
    assert out.role("left").wrench_pred is not None      # the force channel is independent


def test_image_at_takes_a_view_key_and_refuses_one_the_file_does_not_carry(tmp_path):
    path = _write_hl_zarr(tmp_path / "dual_cam.zarr.zip", n_cameras=2)
    src = _source(path)
    assert src.image_keys == ("camera0_rgb", "camera1_rgb")
    root = zarr.open(path, "r")
    a, _b = src.episode_bounds(1)
    np.testing.assert_array_equal(src.image_at(1, 3), root["data"]["camera0_rgb"][a + 3])   # the default view
    np.testing.assert_array_equal(src.image_at(1, 3, "camera1_rgb"), root["data"]["camera1_rgb"][a + 3])
    assert not np.array_equal(src.image_at(1, 3), src.image_at(1, 3, "camera1_rgb"))
    with pytest.raises(KeyError, match="camera2_rgb"):
        src.image_at(1, 3, "camera2_rgb")


def test_recorded_pose_is_the_camera_grid_pose_and_first_usable_frames_matches_per_episode(masked_zarr):
    src = _no_glitch_source(masked_zarr)
    root = zarr.open(masked_zarr, "r")
    a, _b = src.episode_bounds(1)
    pos, rot6d = src.recorded_pose(1, 2, "left")
    np.testing.assert_allclose(pos, root["data"]["robot0_eef_pos"][a + 2], atol=1e-6)
    np.testing.assert_allclose(rot6d, rotvec_to_rot6d(np.asarray(root["data"]["robot0_eef_rot_axis_angle"][a + 2],
                                                                   dtype=np.float64)), atol=1e-6)
    assert src.first_usable_frames() == [src.first_usable_frame(i) for i in range(src.n_episodes)] == [1, 0]


def test_the_cached_usability_agrees_with_a_per_frame_read_of_the_file(masked_zarr):
    """`frame_is_usable` now reads two cached whole-column bits; it must answer exactly what the
    per-frame read used to (step 0 of every role real AND outside the glitch window)."""
    for src in (_no_glitch_source(masked_zarr), _source(masked_zarr)):
        root = zarr.open(masked_zarr, "r")
        valid = np.asarray(root["data"]["hl_left_action_valid"][:], dtype=np.float32)
        keep = src._glitch_keep_mask()
        for ep in range(src.n_episodes):
            a, b = src.episode_bounds(ep)
            for lf in range(b - a):
                want = bool(valid[a + lf, 0] > 0.5) and bool(keep[a + lf])
                assert src.frame_is_usable(ep, lf) == want, (ep, lf)


def test_episode_options_label_the_masked_starts_and_pad_the_index(masked_zarr):
    from fuji.s1_policy.serving.hl_replay import episode_options

    src = _no_glitch_source(masked_zarr)
    opts = episode_options(src)
    assert len(opts) == src.n_episodes == 2
    assert opts[0].startswith("ep 0  (") and "⚠ starts at 1" in opts[0]
    assert opts[1].startswith("ep 1  (") and "⚠" not in opts[1]
    assert episode_options(src, [None, 0])[0].endswith("⚠ no playable frame")
    # a wide file pads the index so the labels sort and search as numbers

    class _Wide:
        n_episodes = 1364

        @staticmethod
        def episode_lengths():
            return [10] * 1364

        @staticmethod
        def first_usable_frames():
            return [0] * 1364

    wide = episode_options(_Wide())
    assert wide[7].startswith("ep 0007  (10 fr)") and wide[1200].startswith("ep 1200")


def test_speed_report_shows_the_step0_overhead_shrinking_with_the_chunk(left_zarr):
    """On a camera grid that IS the action grid the only speed term left is the k-1 overhead —
    every chunk re-commands its step 0 — so exec=8 reads about -12% and exec=16 about -6%, and
    a bigger chunk is always closer to the human's tempo."""
    from fuji.s1_policy.serving.hl_replay import speed_report

    src = _source(left_zarr)
    txt = speed_report(src, 0, 1.0 / ACTION_DT, (8, 16))
    assert "exec=8" in txt and "exec=16" in txt and "wall-clock speed" in txt and "30.00 Hz" in txt
    pcts = {p.split(":")[0].strip(): int(p.split(":")[1].strip().rstrip("%"))
            for p in txt.split("(")[1].split(";")[0].split(",")}
    assert -20 <= pcts["exec=8"] <= -8, txt
    assert pcts["exec=8"] <= pcts["exec=16"] <= 0, txt
    assert speed_report(src, 0, 1.0 / ACTION_DT, ()) .endswith("(; + = faster)")


def test_transport_rotation_is_the_angle_between_recorded_hand_and_anchor():
    from fuji.s1_policy.serving.hl_replay import transport_rotation_deg

    ident = rotmat_to_rot6d(np.eye(3))
    assert transport_rotation_deg(ident, ident) == pytest.approx(0.0)
    c, s = np.cos(np.deg2rad(90.0)), np.sin(np.deg2rad(90.0))
    rz90 = rotmat_to_rot6d(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))
    assert transport_rotation_deg(ident, rz90) == pytest.approx(90.0)
    assert transport_rotation_deg(rz90, ident) == pytest.approx(90.0)    # symmetric in magnitude
    # the reorientation the deploy anchor gives a level handheld hand is not small (clean94: ~107°)
    rot = rot6d_to_rotmat(rz90)
    assert transport_rotation_deg(rotmat_to_rot6d(rot @ rot), ident) == pytest.approx(180.0)
