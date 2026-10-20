"""Contract tests for the DEPLOY RECORDER + its reader
(:mod:`flore.s1_policy.serving.hl_record`, :mod:`flore.s1_policy.serving.hl_take`).

What is pinned here is what decides whether a recorded deploy take can be TRUSTED as evidence
about a real-robot run — i.e. whether "re-infer this take" answers a question about the
checkpoint rather than about the recorder:

  * the observation ROUND-TRIPS. A stored conditioning image re-normalizes to the byte-exact
    tensor the model was fed, so re-inference on a recorded prediction is the same input, not
    a lookalike. This is the single load-bearing property of the whole feature.
  * takes are keyed on EXECUTION, per the operator's ask: a Predict that is never Run leaves
    no take, and a Run does — while retried Predicts are COUNTED so the take says how many
    attempts preceded it.
  * a take is committed ATOMICALLY (``.tmp`` → rename), so a killed flush can lose the newest
    take but can never leave a partial one that a reader would read as complete.
  * every stream shares ONE take-relative clock, because the whole comparison
    (measured vs commanded vs plan) is a time alignment.
  * the reader REFUSES an unknown schema and a wrong column count instead of silently reading
    a renamed column as a different physical quantity.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from flore.s1_policy.data.layout import get_layout
from flore.s1_policy.serving.hl_online import (
    ObsSnapshot,
    denormalize_image_hwc,
    normalize_image_chw,
    resize_center_crop_224,
)
from flore.s1_policy.serving.hl_record import (
    CMD_COLS,
    EXEC_COLS,
    RecordedPrediction,
    TakeRecorder,
)
from flore.s1_policy.serving.hl_take import DeployTake, find_takes
from flore.utils.rot6d import rotmat_to_rot6d

_IDENT6 = rotmat_to_rot6d(np.eye(3))
_LAYOUT = get_layout("left_only")


def _obs(rng, *, to: int = 1, fh: int = 1) -> tuple[ObsSnapshot, np.ndarray]:
    """One synthetic observation + the uint8 image it was built from."""
    img = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    images = np.stack([normalize_image_chw(img, 0.5, 0.5) for _ in range(to)])
    return (
        ObsSnapshot(
            images=images,
            state=rng.normal(size=(to, _LAYOUT.state_dim)),
            state_valid=np.ones((to, _LAYOUT.state_dim), dtype=bool),
            force_history=rng.normal(size=(fh, _LAYOUT.force_hist_channels)),
            force_history_valid=True,
        ),
        img,
    )


def _pred(rng, *, seed: int = 7, n_exec: int = 4, source: str = "model") -> RecordedPrediction:
    obs, _ = _obs(rng)
    return RecordedPrediction(
        obs=obs,
        anchor_pos=np.zeros((1, 3)),
        anchor_rot6d=_IDENT6[None],
        prediction=rng.normal(size=(50, _LAYOUT.action_dim)),
        seed=seed,
        num_steps=16,
        t_predict=0.0,
        t_exec_start=0.1,
        n_exec=n_exec,
        source=source,
    )


def _write_one_take(root: Path, rng, *, n_ticks: int = 6, jpeg: bytes | None = None) -> Path:
    rec = TakeRecorder(root, {"algo": "diffusion"}, layout_mode="left_only")
    rec.arm("unit test", now=1000.0)
    rec.on_prediction(_pred(rng))
    for k in range(n_ticks):
        t = 0.1 + k * (1.0 / 30.0)
        rec.on_executed(
            t, tcp_pos=np.array([0.4 + 0.01 * k, 0.1, 0.9]), tcp_rot6d=_IDENT6,
            dof_pos=np.full(29, 0.01 * k), jaw_m=0.09, vive_armed=True,
            f_meas=np.zeros((2, 3)), chunk_index=0, step_index=k,
        )
        rec.on_commanded(
            t, wire_pos9=np.arange(9, dtype=float), wire_orn12=np.tile([0.0, 0, 0, 1.0], 3),
            grip_cmd_m=0.05, f_cmd=None, chunk_index=0, step_index=k,
        )
        rec.on_force(t, np.arange(_LAYOUT.force_hist_channels, dtype=float))
    if jpeg is not None:
        rec.on_camera(0.1, jpeg, seq=1)
    path = rec.close(block=True)
    assert path is not None
    return Path(path)


# ---------------------------------------------------------------------------
# The load-bearing property: the recorded observation IS the fed observation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("mean", "std"), [(0.5, 0.5), (0.485, 0.229)])
def test_conditioning_image_round_trips_bit_exactly(mean, std):
    """A stored obs image must re-normalize to the EXACT tensor the model was fed.

    This is what makes "re-infer the recorded prediction" a statement about the checkpoint. If
    the affine round-trip lost a least-significant bit, every re-inference would differ from
    its deploy chunk by an amount nobody could distinguish from a real model change."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    chw = normalize_image_chw(img, mean, std)
    recovered = denormalize_image_hwc(chw, mean, std)
    assert np.array_equal(recovered, img)
    # and the re-encode of the recovered uint8 is the same tensor, not merely a close one
    assert np.array_equal(normalize_image_chw(recovered, mean, std), chw)


def test_stored_obs_reloads_into_the_same_arrays(tmp_path):
    """State / force / validity survive the write unchanged, and the images come back as the
    uint8 crop — so the reader can rebuild the batch without the recorder having to store
    checkpoint-specific normalization."""
    rng = np.random.default_rng(1)
    obs, img = _obs(rng)
    rec = TakeRecorder(tmp_path, layout_mode="left_only")
    rec.arm("obs", now=0.0)
    rec.on_prediction(RecordedPrediction(
        obs=obs, anchor_pos=np.zeros((1, 3)), anchor_rot6d=_IDENT6[None],
        prediction=rng.normal(size=(50, _LAYOUT.action_dim)), seed=3, num_steps=16,
        t_predict=0.0, t_exec_start=0.0, n_exec=1, source="model"))
    rec.on_executed(0.0, tcp_pos=np.zeros(3), tcp_rot6d=_IDENT6, dof_pos=np.zeros(29),
                    jaw_m=0.09, vive_armed=True, f_meas=None)
    take = DeployTake(rec.close(block=True))

    stored = take.predictions[0]
    np.testing.assert_allclose(stored.state, obs.state)
    np.testing.assert_allclose(stored.force_history, obs.force_history)
    assert stored.force_history_valid is True
    assert len(stored.images) == obs.images.shape[0]
    assert stored.images[0].shape == img.shape
    assert stored.images[0].dtype == np.uint8
    # THE PIXELS ARE EXACT, not merely close. The conditioning image is stored losslessly
    # precisely so a re-inference feeds the model the same tensor; a JPEG round trip here
    # measured 4.4 mm of chunk difference on the 0808 checkpoint, i.e. it would forge a model
    # difference out of the storage format. See TakeRecorder._write_predictions.
    assert np.array_equal(stored.images[0], img)


def test_the_reloaded_obs_re_encodes_to_the_identical_input_tensor(tmp_path):
    """The end-to-end claim of the whole feature: obs -> disk -> obs is the SAME float tensor,
    so re-inferring a recorded prediction with the same checkpoint and seed must reproduce the
    deploy chunk exactly. Checked against a stand-in for the loaded model so it needs no GPU."""
    rng = np.random.default_rng(13)
    obs, _ = _obs(rng)
    rec = TakeRecorder(tmp_path, layout_mode="left_only", image_mean=0.5, image_std=0.5)
    rec.arm("exactness", now=0.0)
    rec.on_prediction(RecordedPrediction(
        obs=obs, anchor_pos=np.zeros((1, 3)), anchor_rot6d=_IDENT6[None],
        prediction=rng.normal(size=(50, _LAYOUT.action_dim)), seed=11, num_steps=16,
        t_predict=0.0, t_exec_start=0.0, n_exec=8, source="model"))
    rec.on_executed(0.0, tcp_pos=np.zeros(3), tcp_rot6d=_IDENT6, dof_pos=None, jaw_m=None,
                    vive_armed=True, f_meas=None)
    take = DeployTake(rec.close(block=True))

    class _FakeLoaded:
        obs_history_len = 1
        force_hist_len = 1
        layout = _LAYOUT

    rebuilt = take.obs_from_prediction(take.predictions[0], _FakeLoaded())
    assert np.array_equal(rebuilt.images, obs.images)
    np.testing.assert_allclose(rebuilt.state, obs.state)
    np.testing.assert_allclose(rebuilt.force_history, obs.force_history)
    assert rebuilt.force_history_valid == obs.force_history_valid


# ---------------------------------------------------------------------------
# Take boundaries: keyed on EXECUTION, not on Predict
# ---------------------------------------------------------------------------
def test_a_take_with_no_rows_is_not_written(tmp_path):
    """Arming then stopping without executing (homing, rehearsing, an offline page) must leave
    NO take: a directory full of empty takes makes the real ones hard to find."""
    rec = TakeRecorder(tmp_path, layout_mode="left_only")
    rec.arm("armed but nothing ran", now=0.0)
    assert rec.close(block=True) is None
    assert find_takes(tmp_path) == []


def test_a_plan_that_never_moved_the_arm_is_still_kept(tmp_path):
    """The inverse case: a prediction WITH no measured motion is real evidence ("the policy
    planned this and the arm did not move"), so it is written rather than discarded."""
    rng = np.random.default_rng(2)
    rec = TakeRecorder(tmp_path, layout_mode="left_only")
    rec.arm("plan only", now=0.0)
    rec.on_prediction(_pred(rng))
    take = DeployTake(rec.close(block=True))
    assert len(take.predictions) == 1
    assert len(take) == 0


def test_retried_predicts_are_counted_not_stored(tmp_path):
    """The operator's rule is "按run的时候为标准": only the EXECUTED plan is stored, but the
    number of superseded attempts is kept, because "I re-predicted 5 times before this one
    looked right" is information about the run."""
    rng = np.random.default_rng(3)
    rec = TakeRecorder(tmp_path, layout_mode="left_only")
    rec.arm("retries", now=0.0)
    for _ in range(3):
        rec.note_discarded_predict()
    rec.on_prediction(_pred(rng))
    rec.on_executed(0.0, tcp_pos=np.zeros(3), tcp_rot6d=_IDENT6, dof_pos=None, jaw_m=None,
                    vive_armed=True, f_meas=None)
    take = DeployTake(rec.close(block=True))
    assert take.meta["n_predicts_discarded"] == 3
    assert len(take.predictions) == 1
    assert take.predictions[0].index == 0


def test_ingest_before_arm_and_after_close_is_dropped(tmp_path):
    """Every ``on_*`` is called unconditionally from the planner's hot loops, so they MUST
    no-op outside a take — otherwise rows from an un-recorded stretch leak into the next take
    and its clock starts before its own t=0."""
    rng = np.random.default_rng(4)
    rec = TakeRecorder(tmp_path, layout_mode="left_only")
    rec.on_executed(0.0, tcp_pos=np.zeros(3), tcp_rot6d=_IDENT6, dof_pos=None, jaw_m=None,
                    vive_armed=False, f_meas=None)
    rec.on_force(0.0, np.zeros(_LAYOUT.force_hist_channels))
    rec.on_prediction(_pred(rng))
    assert rec.counts()["executed"] == 0
    assert rec.close(block=True) is None


def test_takes_are_numbered_in_execution_order(tmp_path):
    rng = np.random.default_rng(5)
    first = _write_one_take(tmp_path, rng, n_ticks=3)
    second = _write_one_take(tmp_path, rng, n_ticks=3)
    assert first.name == "take_000"
    assert second.name == "take_001"
    assert [p.name for p in find_takes(tmp_path)] == ["take_000", "take_001"]


# ---------------------------------------------------------------------------
# Durability + the reader's refusals
# ---------------------------------------------------------------------------
def test_a_leftover_tmp_directory_is_not_read_as_a_take(tmp_path):
    """A ``.tmp`` is a flush that was killed mid-write. The atomic rename means it is never
    mistaken for the real take, and the scan must skip it — reading one would present a
    partial take as complete."""
    rng = np.random.default_rng(6)
    real = _write_one_take(tmp_path, rng, n_ticks=3)
    shutil.copytree(real, tmp_path / "take_009.tmp")
    assert [p.name for p in find_takes(tmp_path)] == [real.name]


def test_reader_refuses_an_unknown_schema(tmp_path):
    rng = np.random.default_rng(7)
    take = _write_one_take(tmp_path, rng, n_ticks=3)
    meta = json.loads((take / "take_meta.json").read_text())
    meta["schema"] = "hl_deploy_v99"
    (take / "take_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="schema"):
        DeployTake(take)


def test_reader_refuses_a_wrong_column_count(tmp_path):
    """A silently renamed/reordered column would be read as a different physical quantity —
    the same class of trap the dataset's attr gates exist for."""
    rng = np.random.default_rng(8)
    take = _write_one_take(tmp_path, rng, n_ticks=3)
    np.save(take / "executed.npy", np.zeros((4, len(EXEC_COLS) - 1)))
    with pytest.raises(ValueError, match="columns"):
        DeployTake(take)


def test_missing_take_meta_is_a_clear_error(tmp_path):
    (tmp_path / "not_a_take").mkdir()
    with pytest.raises(FileNotFoundError, match="not a deploy take"):
        DeployTake(tmp_path / "not_a_take")


# ---------------------------------------------------------------------------
# One clock, and the columns mean what their names say
# ---------------------------------------------------------------------------
def test_all_streams_share_one_take_relative_clock(tmp_path):
    """The measured, commanded and force streams are differenced against each other, so they
    must be on ONE clock whose zero is the take's arm moment. Two clocks would misalign the
    comparison in a way that looks like tracking error."""
    rng = np.random.default_rng(9)
    take = DeployTake(_write_one_take(tmp_path, rng, n_ticks=5))
    t_exec = take.executed[:, EXEC_COLS.index("t")]
    t_cmd = take.commanded[:, CMD_COLS.index("t")]
    t_force = take.force_meas[:, 0]
    np.testing.assert_allclose(t_exec, t_cmd)
    np.testing.assert_allclose(t_exec, t_force)
    assert t_exec[0] >= 0.0
    np.testing.assert_allclose(np.diff(t_exec), 1.0 / 30.0, atol=1e-9)


def test_every_recorder_call_site_stamps_through_elapsed():
    """The test above can only prove the recorder STORES the clock it is handed — the fixture
    hands it a relative one. What actually broke was the PRODUCTION CALLER: ``on_force`` and
    ``on_camera`` in ``scripts/hl_serving.py`` passed raw ``time.monotonic()`` while
    ``on_executed``/``on_commanded`` passed ``recorder.elapsed()``, putting those two streams
    ~10^6 s away from the others. Every cross-stream lookup then resolved to sample 0: the
    replay viewer's camera froze on the first frame and "what did the sensor feel here"
    answered with the first sample forever.

    So this is a SOURCE-level check, for the same reason the reach/GIL wiring tests are: the
    seam is a call site, and no unit test of the recorder can see it.
    """
    src = (Path(__file__).resolve().parents[1] / "scripts" / "hl_serving.py").read_text()
    tree = ast.parse(src)
    timed = {"on_executed", "on_commanded", "on_force", "on_camera"}
    seen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr in timed
                and isinstance(fn.value, ast.Name) and fn.value.id == "recorder"):
            continue
        seen.add(fn.attr)
        assert node.args, f"recorder.{fn.attr} called with no timestamp"
        stamp = ast.unparse(node.args[0])
        assert "recorder.elapsed" in stamp, (
            f"recorder.{fn.attr} is stamped with {stamp!r}, not recorder.elapsed(...) — that "
            "puts this stream on a different epoch from the others (see this test's docstring)")
    assert seen == timed, f"call sites not found for {sorted(timed - seen)}"


def test_legacy_absolute_clock_is_rebased_on_read(tmp_path):
    """Takes recorded before the fix above are REPAIRED on read, so they stay viewable.

    Only the zero was wrong, so re-zeroing recovers the alignment; the reader names what it
    touched in ``legacy_clock_streams`` because the offset is recovered by aligning first
    samples and is therefore exact only to their arrival difference.
    """
    rng = np.random.default_rng(11)
    path = _write_one_take(tmp_path, rng, n_ticks=6, jpeg=b"\xff\xd8\xff\xd9")
    good = DeployTake(path)
    assert good.legacy_clock_streams == (), "a correctly-recorded take must not be touched"
    ref_force_t = good.force_meas[:, 0].copy()

    # Corrupt exactly the way the old recorder did: shift those two streams onto the raw
    # monotonic epoch, leaving executed/commanded take-relative.
    for name in ("force_meas.npy", "camera/frame_index.npy"):
        arr = np.load(path / name)
        arr[:, 0] += 951_626.0
        np.save(path / name, arr)

    legacy = DeployTake(path)
    assert set(legacy.legacy_clock_streams) == {"force_meas", "camera_index"}
    # Rebased back onto the executed clock, to within the streams' first-arrival difference.
    np.testing.assert_allclose(legacy.force_meas[:, 0], ref_force_t, atol=1e-6)
    ft_t, _ = legacy.measured_wrench_sensor()
    assert ft_t.min() >= 0.0 and ft_t.max() <= legacy.times.max() + 1.0
    # ...and the lookup that was silently pinned to sample 0 now tracks the cursor.
    assert legacy.camera_index_for_time(float(legacy.times[-1])) is not None
    assert "legacy_clock_streams" in legacy.summary()


def test_force_and_jaw_streams_carry_their_documented_quantities(tmp_path):
    """The accessors the replay viewer draws from: each must return ITS OWN quantity, because
    a take holds three different force vectors (raw sensor / servo pelvis / published command)
    and drawing one while believing it is another is the whole class of bug here."""
    rng = np.random.default_rng(12)
    take = DeployTake(_write_one_take(tmp_path, rng, n_ticks=5))
    ft_t, ft_w = take.measured_wrench_sensor()
    assert ft_w.shape == (5, _LAYOUT.force_hist_channels)
    np.testing.assert_allclose(ft_w[0], np.arange(_LAYOUT.force_hist_channels))
    np.testing.assert_allclose(ft_t, take.times)

    sv_t, sv_f = take.servo_force_pelvis("left")
    assert sv_f.shape == (5, 3)
    np.testing.assert_allclose(sv_f, 0.0)          # the fixture records a zero f_meas
    with pytest.raises(KeyError):
        take.servo_force_pelvis("head")

    # f_cmd was None at every tick, so the published force is all-NaN/zero — i.e. provably
    # "no force left this process", which is what a pure-position run must look like.
    _, pub = take.commanded_force_sensor()
    assert pub.shape == (5, 3)

    cj_t, cj = take.commanded_jaw()
    np.testing.assert_allclose(cj, 0.05)
    np.testing.assert_allclose(take.measured_jaw(), 0.09)

    pred = take.predictions[0]
    wren = take.predicted_wrench(pred)
    assert wren is not None and wren.shape == (50, _LAYOUT.force_hist_channels)
    jaw_pred = take.predicted_jaw(pred)
    assert jaw_pred is not None and jaw_pred.shape == (50,)
    # The recorder stamped chunk 0 / step k on tick k.
    assert take.executed_step_at(3) == (0, 3)


def test_measured_columns_read_back_as_the_named_quantities(tmp_path):
    rng = np.random.default_rng(10)
    take = DeployTake(_write_one_take(tmp_path, rng, n_ticks=4))
    pos, rot = take.measured_tcp()
    np.testing.assert_allclose(pos[:, 0], [0.4, 0.41, 0.42, 0.43])
    np.testing.assert_allclose(rot[0], _IDENT6)
    np.testing.assert_allclose(take.measured_jaw(), 0.09)
    t_dof, dof = take.measured_dof()
    assert dof.shape == (4, 29)
    np.testing.assert_allclose(dof[2], 0.02)


def test_non_finite_and_wrong_width_rows_are_dropped(tmp_path):
    """A bad FK or a short readback must not poison the arrays with NaN or a ragged row — the
    same contract ``ExecutionTrace.add`` has, for the same reason."""
    rec = TakeRecorder(tmp_path, layout_mode="left_only")
    rec.arm("bad rows", now=0.0)
    rec.on_executed(0.0, tcp_pos=np.array([np.nan, 0, 0]), tcp_rot6d=_IDENT6, dof_pos=None,
                    jaw_m=None, vive_armed=True, f_meas=None)
    rec.on_executed(0.1, tcp_pos=np.zeros(2), tcp_rot6d=_IDENT6, dof_pos=None, jaw_m=None,
                    vive_armed=True, f_meas=None)
    # a good row, but with a SHORT dof vector: the pose is kept, the joints are not
    rec.on_executed(0.2, tcp_pos=np.zeros(3), tcp_rot6d=_IDENT6, dof_pos=np.zeros(7),
                    jaw_m=None, vive_armed=True, f_meas=None)
    take = DeployTake(rec.close(block=True))
    assert len(take) == 1
    assert take.dof.shape[0] == 0
    assert np.isfinite(take.executed).all() or np.isnan(
        take.executed[:, EXEC_COLS.index("jaw_m")]).all()


def test_camera_frames_dedupe_on_the_bridge_sequence(tmp_path):
    """The ingest loop polls faster than the camera publishes, so without the seq dedupe a
    20 s take would store thousands of duplicate JPEGs of the same capture."""
    rec = TakeRecorder(tmp_path, layout_mode="left_only")
    rec.arm("camera", now=0.0)
    for _ in range(5):
        rec.on_camera(0.0, b"\xff\xd8fake", seq=4)  # same capture, polled 5x
    rec.on_camera(0.1, b"\xff\xd8fake2", seq=5)
    rec.on_executed(0.0, tcp_pos=np.zeros(3), tcp_rot6d=_IDENT6, dof_pos=None, jaw_m=None,
                    vive_armed=True, f_meas=None)
    take = DeployTake(rec.close(block=True))
    assert take.meta["n_frames"]["camera"] == 2
    assert take.camera_index.shape == (2, 2)


# ---------------------------------------------------------------------------
# Re-decoding a stored plan back into the measured stream's own frame
# ---------------------------------------------------------------------------
def test_a_stored_chunk_decodes_against_its_own_anchor(tmp_path):
    """The point of storing the anchor: the plan can be put back into the SAME pelvis-frame
    TCP the measured stream holds, which is what makes measured-vs-planned a tracking error
    rather than a frame difference. Step 0 of an anchor-relative chunk sits AT the anchor."""
    anchor = np.array([[0.44, 0.15, 0.95]])
    rec = TakeRecorder(tmp_path, layout_mode="left_only")
    rec.arm("decode", now=0.0)
    chunk = np.zeros((50, _LAYOUT.action_dim))
    slots = _LAYOUT.slots("left")
    chunk[:, slots.rot6d] = _IDENT6  # identity rotation at every step
    chunk[:, slots.pos] = 0.0  # zero delta => every step sits on the anchor
    rec.on_prediction(RecordedPrediction(
        obs=None, anchor_pos=anchor, anchor_rot6d=_IDENT6[None], prediction=chunk,
        seed=1, num_steps=16, t_predict=0.0, t_exec_start=0.0, n_exec=50, source="model"))
    rec.on_executed(0.0, tcp_pos=anchor[0], tcp_rot6d=_IDENT6, dof_pos=None, jaw_m=None,
                    vive_armed=True, f_meas=None)
    take = DeployTake(rec.close(block=True))
    pos = take.decoded_chunk_positions(take.predictions[0])
    np.testing.assert_allclose(pos, np.broadcast_to(anchor, (50, 3)), atol=1e-12)


def test_a_replay_sourced_prediction_reports_no_observation(tmp_path):
    """A REPLAYED chunk had no model inference, so it has no observation. The reader must say
    so rather than fabricate one — a re-inference on invented input would look like a
    comparison and be nothing of the kind."""
    rng = np.random.default_rng(12)
    rec = TakeRecorder(tmp_path, layout_mode="left_only")
    rec.arm("replay", now=0.0)
    rec.on_prediction(RecordedPrediction(
        obs=None, anchor_pos=np.zeros((1, 3)), anchor_rot6d=_IDENT6[None],
        prediction=rng.normal(size=(50, _LAYOUT.action_dim)), seed=-1, num_steps=0,
        t_predict=0.0, t_exec_start=0.0, n_exec=16, source="replay",
        replay_episode=4, replay_frame=17))
    rec.on_executed(0.0, tcp_pos=np.zeros(3), tcp_rot6d=_IDENT6, dof_pos=None, jaw_m=None,
                    vive_armed=True, f_meas=None)
    take = DeployTake(rec.close(block=True))
    stored = take.predictions[0]
    assert stored.source == "replay"
    assert stored.replay_episode == 4 and stored.replay_frame == 17
    assert stored.state is None and stored.images == ()

    class _FakeLoaded:  # only the fields obs_from_prediction reads
        obs_history_len = 1
        force_hist_len = 1
        layout = _LAYOUT

    with pytest.raises(ValueError, match="no recorded observation"):
        take.obs_from_prediction(stored, _FakeLoaded())


# ---------------------------------------------------------------------------
# The training crop, shared with the live ingest
# ---------------------------------------------------------------------------
def test_the_crop_is_aspect_preserving_and_matches_the_converter_geometry():
    """A re-inference must crop a recorded camera frame exactly as the live ingest did, or the
    comparison measures the crop instead of the checkpoint. Pinned against the converter's
    integer geometry (ceil-scale, floor-div crop)."""
    wide = np.zeros((720, 1280, 3), dtype=np.uint8)
    wide[:, 640] = 255  # a vertical line down the exact centre
    out = resize_center_crop_224(wide)
    assert out.shape == (224, 224, 3)
    # the centre line stays centred (a stretch or an off-centre crop would move it)
    col = int(np.argmax(out.astype(np.float64).sum(axis=(0, 2))))
    assert abs(col - 112) <= 1
    tall = np.zeros((1280, 720, 3), dtype=np.uint8)
    assert resize_center_crop_224(tall).shape == (224, 224, 3)


def test_the_observation_cap_is_bounded_and_reported(tmp_path, monkeypatch):
    """The ACT ensembled cadence records ONE prediction PER STEP, so an unbounded take would
    grow ~600 KiB of RAM per step (measured) — ~360 MB for a 20 s rollout inside a live control
    process. Past the cap the observation is dropped but the PLAN is kept, and the count is
    written to the take so a truncated take can never look complete."""
    import flore.s1_policy.serving.hl_record as hr

    monkeypatch.setattr(hr, "MAX_FULL_PREDICTIONS", 2)
    rng = np.random.default_rng(21)
    rec = hr.TakeRecorder(tmp_path, layout_mode="left_only")
    rec.arm("capped", now=0.0)
    for _ in range(5):
        rec.on_prediction(_pred(rng))
    rec.on_executed(0.0, tcp_pos=np.zeros(3), tcp_rot6d=_IDENT6, dof_pos=None, jaw_m=None,
                    vive_armed=True, f_meas=None)
    take = DeployTake(rec.close(block=True))
    assert len(take.predictions) == 5           # every executed plan is still recorded
    assert take.meta["n_obs_dropped"] == 3      # …and the drop is reported, not silent
    assert take.meta["max_full_predictions"] == 2
    assert take.predictions[0].state is not None
    assert take.predictions[4].state is None    # chunk/anchor/seed survive; obs does not
    assert take.predictions[4].chunk.shape == (50, _LAYOUT.action_dim)
