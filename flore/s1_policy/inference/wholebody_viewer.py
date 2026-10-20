"""The offline page for ``diffusion_umi_wholebody``: two G1 bodies, one of them violet.

The wholebody twin of :mod:`flore.s1_policy.inference.viewer`, and a separate module because the
two pages have no drawn object in common. That page's vocabulary is TRAJECTORIES — a blue target
polyline, an orange prediction, gripper clusters at a keypoint, force arrows on a path, a
recorded episode swept forward. None of it exists here: a wholebody chunk has no position and no
rotation columns at all (``layout.pose_dims()`` is empty), so there is no path to draw. What it
has is a POSTURE per step, and the only honest picture of a posture is the robot in it.

    🟪  the VIOLET body  = the policy's 29 predicted joint angles at the selected action step
    ⬜  the pale body    = the dataset's own (masked) target joints at that same step
        the two charts   = per-joint error in degrees, plus the grippers and both wrenches

The violet is ``hl_serving._GHOST_RGBA``, deliberately: across this repo violet already means
"what is about to happen / what was predicted" (the played-back chunk in ``hl_serving``, the
completion ghost in ``hl_serving_mc``). A wholebody rollout is the same claim in a page that has
nothing else to say it with.

WHY THE FLOOR IS WHERE IT IS
============================
A wholebody chunk carries NO ROOT — and that is complete rather than missing (``DenseWire``
publishes 29 positions + 29 finite-differenced velocities, so 29 columns IS the command; the
exporter writes ``wb_root_pose`` as provenance and nothing reads it). So the body's height above
the grid cannot come from the policy. It is FK'd instead: :meth:`ready_pose.KeypointFK.pelvis_z`
plants the lower sole of the TARGET's step-0 pose on ``z=0``, and that ONE height is held for the
whole chunk and shared by both bodies.

Both halves of that are choices with a visible consequence, so they are spelled out:

* **Held, not re-grounded per frame.** Re-grounding every step would keep the feet nailed to the
  grid and make a predicted CROUCH invisible — the floor would follow the robot down. Held, a
  crouch reads as the body sinking, which is what it is.
* **Shared, not per-body.** A prediction that stands 4 cm taller than the demonstration then
  visibly floats above it instead of being silently re-planted. That difference is a real
  disagreement about leg extension and this page's job is to show it.

WHAT THIS PAGE DOES NOT CLAIM
=============================
It is not a deploy preview. The absolute joints are in the dataset's ``cond_mode="home"`` posture
space, whereas the robot holds a 38 deg crouch after **A** (measured 0.514 rad apart at the knee)
— at deploy that gap is absorbed by ``hl_wholebody.WholeBodyWire``'s joint clutch, which
publishes ``base + (q[i] - q[0])`` and never the absolute pose. Here the absolute pose is exactly
what you want to see, because it is what the model was trained on and what the target is.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flore.s1_policy.data.layout import WB_JOINT_NAMES
from flore.s1_policy.inference import ready_pose as RP
from flore.s1_policy.inference.viewer import (
    _add_scalar_chart,
    _add_vector_chart,
    _checked_gui_integer,
    _checked_port,
    _checked_sample_index,
    _format_metric,
    _require_port_available,
    _set_scalar_chart,
    _set_vector_chart,
)
from flore.s1_policy.inference.wholebody import WholeBodyRollout, wholebody_summary
from flore.s1_policy.serving.live_robot import (
    JAW_JOINT_OPEN_M,
    build_sdk_to_urdf_index,
    jaw_joint_from_width_m,
    remap_dof_to_urdf,
)

#: The PREDICTION's colour: translucent violet, byte-identical to ``hl_serving._GHOST_RGBA``.
#: Not imported from that script — it is a CLI entry point that installs sys.path entries and
#: parses UMI submodule imports at module scope, and this package must stay importable without
#: any of that. Pinned equal by a test instead.
PREDICTION_RGBA = (0.62, 0.36, 0.95, 0.45)
#: The TARGET's colour: pale desaturated blue, dimmer than the violet so the prediction reads as
#: the foreground. Alpha lower too — when the two postures agree the violet must win the overlap,
#: because the prediction is the thing under test.
TARGET_RGBA = (0.55, 0.62, 0.72, 0.30)

#: How many worst joints get their own chart / table row. Three because that is what
#: ``_add_vector_chart`` draws (it is an x/y/z chart repurposed), and six in the table because a
#: 29-joint diagnosis is usually "one limb", which is 3-7 joints.
_WORST_CHART = 3
_WORST_TABLE = 6

_PANEL_MD = """\
**`diffusion_umi_wholebody`** — the policy predicts the **29 G1 joint angles** the `dense` low
level consumes, so there is no trajectory to draw and no motion completion to run.

| | what it is |
|---|---|
| 🟪 **violet body** | the PREDICTED 29 joints at the selected action step |
| ⬜ **pale body** | the dataset's own TARGET joints at that step (masked steps hold) |
| **charts** | per-joint error in degrees, both grippers, both wrenches |

The body has **no root channel** — that is complete, not missing (the wire is 29 positions +
29 finite-differenced velocities). The floor height is FK'd from the TARGET's step 0 and HELD, so
a predicted crouch shows as the body sinking and a taller prediction visibly floats.

⚠️ These are ABSOLUTE joints in the dataset's `cond_mode=home` posture space. Deploy publishes
`base + (q[i] - q[0])` through a joint clutch, never this absolute pose — see
`docs/RUNBOOK_hl_serving_wholebody.md`.
"""


@dataclass(frozen=True)
class _Request:
    generation: int
    sample_index: int
    seed: int
    num_steps: int


def _find_yam_urdf() -> str | None:
    """The rig's ``g1_29dof_yam.urdf``, or None on a checkout without the UMI submodule.

    Same walk-up as :func:`mc_humanoid._find_umi_replay_dir`. None is a real state — a fresh
    clone has no submodules — and it disables only the two BODIES, leaving every chart and
    metric intact, because those are the numbers and the bodies are the picture.
    """
    root = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = root / "third_party" / "UMI_collection" / "robot_assets" / "g1_yam" \
            / "g1_29dof_yam.urdf"
        if candidate.is_file():
            return str(candidate)
        if root.parent == root:
            break
        root = root.parent
    return None


def _jaw_pair(widths: dict, index: int) -> tuple[float, float]:
    """``(left, right)`` finger-joint values for one step, OPEN where there is no width.

    OPEN, never 0.0: 0.0 is the CLOSED end of this joint's travel, so the naive default draws a
    hand that is holding something. Same rule (and same helper) as the live planners' ghosts."""
    out = []
    for role in ("left", "right"):
        w = widths.get(role)
        out.append(JAW_JOINT_OPEN_M if w is None or index >= w.shape[0]
                   else jaw_joint_from_width_m(float(w[index])))
    return out[0], out[1]


def _worst_joints(result: WholeBodyRollout, count: int) -> list[tuple[int, float]]:
    """``[(joint index, mean |error| deg)]``, worst first, over SUPERVISED steps only.

    ``nanmean`` over :meth:`WholeBodyRollout.joint_error_deg`, whose masked entries are NaN
    precisely so an unsupervised tail cannot be averaged in as a perfect 0 — the same trap
    ``wholebody_metrics`` guards one level up."""
    err = result.joint_error_deg()
    with np.errstate(invalid="ignore"):
        per_joint = np.nanmean(np.where(np.isfinite(err), err, np.nan), axis=0)
    order = [i for i in np.argsort(-np.nan_to_num(per_joint, nan=-1.0)) if np.isfinite(per_joint[i])]
    return [(int(i), float(per_joint[i])) for i in order[:count]]


def _hold_last_valid(joints: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``(H, 29)`` target joints with UNSUPERVISED steps holding the last supervised one.

    An unsupervised step's stored value is whatever the zero-fill left there, i.e. 29 zeros — a
    real, legal, and completely wrong posture (the G1 standing bolt upright). Drawing it makes the
    pale body SNAP straight at the first masked step, which reads as a demonstration that did
    something violent rather than as one that ended. Holding is the honest render; the charts
    still show the gap as absent, because they mask.
    """
    q = np.asarray(joints, dtype=np.float64).copy()
    ok = np.asarray(valid, dtype=bool).all(axis=-1)
    if not ok.any():
        return q
    last = int(np.flatnonzero(ok)[0])
    for i in range(q.shape[0]):
        if ok[i]:
            last = i
        else:
            q[i] = q[last]
    return q


def run_wholebody_viewer(
    session,
    *,
    host: str = "127.0.0.1",
    port: int = 8088,
    sample_index: int = 0,
    seed: int = 0,
    urdf_path: str | None = None,
) -> None:
    """Serve the wholebody inference page. Blocks until interrupted."""
    import viser

    port = _checked_port(port)
    _require_port_available(host, port)
    sample_count = len(session.source)
    sample_index = _checked_sample_index(sample_index, sample_count)

    urdf_file = urdf_path or _find_yam_urdf()
    #: Bound only when the URDF loaded; ``_pose`` returns before reading it otherwise.
    remap = None
    server = viser.ViserServer(host=host, port=port,
                              label="FLoRe wholebody inference (29 joints)")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", plane="xy", shadow_opacity=0.0, infinite_grid=True,
                          cell_size=0.25)
    gui = server.gui
    gui.add_markdown(_PANEL_MD)

    bodies: dict = {}
    urdf_note = ""
    if urdf_file is None:
        urdf_note = ("the yam URDF is not checked out, so no body is drawn — every chart and "
                     "metric below is unaffected (git submodule update --init "
                     "third_party/UMI_collection)")
    else:
        try:
            from viser.extras import ViserUrdf  # noqa: PLC0415

            for name, rgba in (("target", TARGET_RGBA), ("prediction", PREDICTION_RGBA)):
                # Prediction added LAST so it wins the depth tie when the two postures agree.
                root = server.scene.add_frame(f"/{name}", show_axes=False,
                                              position=(0.0, 0.0, 0.0))
                body = ViserUrdf(server, urdf_or_path=Path(urdf_file),
                                 root_node_name=f"/{name}", mesh_color_override=rgba)
                bodies[name] = {"root": root, "urdf": body}
            joint_names = list(bodies["prediction"]["urdf"].get_actuated_joint_names())
            remap = build_sdk_to_urdf_index(joint_names)
            if remap.missing_sdf:
                urdf_note = (f"URDF is missing SDK joints {remap.missing_sdf} — the bodies are "
                             "disabled rather than drawn with a limb in the wrong place")
                bodies = {}
        except Exception as exc:  # noqa: BLE001 - the page must serve without viz extras
            urdf_note = f"could not load the drawn bodies ({exc!r}); charts are unaffected"
            bodies = {}

    # The FK that grounds the scene. Optional for the same reason the meshes are: it needs
    # yourdfpy. Without it both bodies sit with their PELVIS at the origin, which is still a
    # correct relative picture — only the floor is then unregistered, and the note says so.
    fk, fk_why = (None, "not requested") if not bodies else RP.KeypointFK.load(urdf_path=urdf_file)

    with gui.add_folder("Sample"):
        g_sample = gui.add_number("Sample index", initial_value=sample_index, min=0,
                                 max=sample_count - 1, step=1)
        g_seed = gui.add_number("Seed", initial_value=int(seed), step=1)
        g_steps = gui.add_number("Sampler steps", initial_value=int(session.num_steps), min=1,
                                step=1,
                                max=(session.max_num_steps if session.max_num_steps else None))
        b_run = gui.add_button("Re-run rollout")
        g_id = gui.add_text("identity", initial_value="…", disabled=True)
        g_status = gui.add_text("status", initial_value="starting …", disabled=True)
        if urdf_note:
            gui.add_text("bodies", initial_value=urdf_note[:110], disabled=True)

    with gui.add_folder("Playback"):
        g_step = gui.add_slider("Action step", min=0, max=1, step=1, initial_value=0)
        c_play = gui.add_checkbox("Play", initial_value=False)
        g_speed = gui.add_slider("Speed x", min=0.25, max=4.0, step=0.25, initial_value=1.0)
        c_target = gui.add_checkbox("Show the target body (pale)", initial_value=True)
        c_pred = gui.add_checkbox("Show the prediction (violet)", initial_value=True)
        g_step_note = gui.add_text("step", initial_value="…", disabled=True)

    with gui.add_folder("Metrics"):
        metric_units = {"joint_deg": "deg", "gripper_mm": "mm", "force_n": "N",
                        "torque_nm": "N·m"}
        metric_text = {key: gui.add_text(key, initial_value="-", disabled=True)
                       for key in metric_units}
        g_worst = gui.add_markdown("_no rollout yet_")

    image_handles = []
    with gui.add_folder("Conditioning images"):
        # CAMERA-MAJOR, matching HLDataset.__getitem__'s own `for key in image_keys for hf in
        # hist_frames`, so a swapped pair of bridges is visible rather than inferred.
        history_len = session.source.dataset.obs_history_len
        for camera in session.source.image_keys:
            for index in range(history_len):
                suffix = " (t0)" if index == history_len - 1 else ""
                image_handles.append(gui.add_image(
                    np.zeros((16, 16, 3), dtype=np.uint8),
                    label=f"{camera} obs {index}{suffix}", format="png"))

    charts: dict = {}
    with gui.add_folder("Joint charts"):
        # Solid = target, dashed = prediction, in DEGREES. The chart helper is the keypoint
        # page's, reused rather than reimplemented: it preserves masked GAPS (a uPlot
        # Float64Array does not), which is the whole reason that page hand-rolls SVG.
        charts["worst"] = _add_vector_chart(gui, "worst 3 joints (deg)", ("j0", "j1", "j2"))
        # target = the constant 0 an ideal rollout would sit on, prediction = the real mean
        # |error|. Titled so that reading is unambiguous.
        charts["mean_err"] = _add_scalar_chart(gui, "mean |joint error| (0 = perfect)", "deg")

    with gui.add_folder("Hand charts"):
        for role in session.source.layout.gripper_roles:
            charts[f"grip:{role}"] = _add_scalar_chart(gui, f"{role} gripper", "m")
        for role in session.source.layout.force_roles:
            charts[f"force:{role}"] = _add_vector_chart(
                gui, f"{role} force (sensor frame, N)", ("Fx", "Fy", "Fz"))

    state: dict = {"result": None, "target_q": None, "pelvis_z": 0.0, "gen": 0,
                   # True while the PLAY LOOP is writing the step slider, so its own write is
                   # not mistaken for an operator drag. Written and read on the render loop
                   # only; the rollout worker never touches it.
                   "programmatic": False}
    requests: queue.Queue = queue.Queue()
    lock = threading.Lock()

    def _apply(result: WholeBodyRollout) -> None:
        """Install one rollout: charts, metrics, images, and the held floor height."""
        target_q = _hold_last_valid(result.target_joints, result.joint_valid)
        pelvis_z = 0.0
        if fk is not None:
            try:
                pelvis_z = float(fk.pelvis_z(target_q[0]))
            except Exception:  # noqa: BLE001 - a bad pose must not lose the rollout
                pelvis_z = 0.0
        with lock:
            state["result"] = result
            state["target_q"] = target_q
            state["pelvis_z"] = pelvis_z

        ref = result.sample
        g_id.value = (f"#{ref.dataset_index} ep{ref.global_episode_id}+{ref.episode_frame} "
                      f"{ref.split} · {result.algorithm} · {result.num_steps} step(s)")
        for key, handle in metric_text.items():
            handle.value = _format_metric(float(result.metrics.get(key, np.nan)),
                                          metric_units[key])
        for handle, frame in zip(image_handles, result.conditioning_images, strict=False):
            handle.image = frame

        times = np.asarray(result.action_times_s[:result.horizon], dtype=np.float64)
        worst = _worst_joints(result, _WORST_CHART)
        if worst:
            idx = [i for i, _ in worst]
            charts["worst"]._inference_labels = tuple(
                # The full SDK name is 24-28 characters and three of them do not fit a legend, so
                # the invariant "left_"/"right_" prefix and the "_joint" suffix are dropped. The
                # TABLE below carries the full names.
                WB_JOINT_NAMES[i].replace("_joint", "").replace("left_", "L.")
                .replace("right_", "R.") for i in idx)
            _set_vector_chart(charts["worst"], times,
                              np.degrees(result.target_joints[:, idx]),
                              np.degrees(result.predicted_joints[:, idx]),
                              result.joint_valid[:, idx])
            rows = "\n".join(
                f"| `{WB_JOINT_NAMES[i]}` | {deg:.2f}° |"
                for i, deg in _worst_joints(result, _WORST_TABLE))
            g_worst.content = ("| joint | mean &#124;error&#124; |\n|---|---|\n" + rows)
        err = result.joint_error_deg()
        with np.errstate(invalid="ignore"):
            mean_err = np.nanmean(err, axis=1)
        supervised = result.joint_valid.all(axis=-1)
        _set_scalar_chart(charts["mean_err"], times, np.zeros_like(mean_err),
                          np.where(supervised, mean_err, np.nan), supervised)

        for role, pred in result.predicted_gripper_m.items():
            _set_scalar_chart(charts[f"grip:{role}"], times,
                              result.target_gripper_m[role], pred,
                              result.gripper_valid[role])
        for role, pred in result.predicted_wrench.items():
            _set_vector_chart(charts[f"force:{role}"], times,
                              result.target_wrench[role][:, 0:3], pred[:, 0:3],
                              result.wrench_valid[role][:, 0:3])

        g_step.max = max(result.horizon - 1, 1)
        g_step.value = 0
        summary = wholebody_summary(result)
        g_status.value = (f"joint {summary['metrics']['joint_deg']:.2f}° · "
                          f"worst {summary['worst_joint']} "
                          f"{summary['worst_joint_deg']:.2f}° · "
                          f"{summary['supervised_steps']}/{result.horizon} supervised")

    def _pose(index: int) -> None:
        """Put both bodies at action step ``index``."""
        with lock:
            result, target_q, pelvis_z = state["result"], state["target_q"], state["pelvis_z"]
        if result is None:
            return
        index = int(np.clip(index, 0, result.horizon - 1))
        supervised = bool(result.joint_valid[index].all())
        g_step_note.value = (f"step {index}/{result.horizon - 1} · "
                             f"t={result.action_times_s[index]:.3f}s · "
                             f"{'supervised' if supervised else 'UNSUPERVISED (target holds)'}")
        if not bodies:
            return
        for name, q, widths in (
            ("prediction", result.predicted_joints[index], result.predicted_gripper_m),
            ("target", target_q[index], result.target_gripper_m),
        ):
            left_jaw, right_jaw = _jaw_pair(widths, index)
            bodies[name]["urdf"].update_cfg(
                remap_dof_to_urdf(q, remap, left_jaw_m=left_jaw, right_jaw_m=right_jaw))
            bodies[name]["root"].position = (0.0, 0.0, pelvis_z)

    def _worker() -> None:
        while True:
            request: _Request = requests.get()
            with lock:
                if request.generation != state["gen"]:
                    continue      # superseded by a newer request; drop it
            try:
                result = session.infer(request.sample_index, seed=request.seed,
                                       num_steps=request.num_steps)
            except Exception as exc:  # noqa: BLE001 - a bad sample must not kill the page
                g_status.value = f"rollout failed: {type(exc).__name__}: {exc}"
                continue
            with lock:
                if request.generation != state["gen"]:
                    continue
            try:
                _apply(result)
                _pose(0)
            except Exception as exc:  # noqa: BLE001
                g_status.value = f"draw failed: {type(exc).__name__}: {exc}"

    def _enqueue() -> None:
        try:
            index = _checked_sample_index(
                _checked_gui_integer(g_sample.value, "sample index"), sample_count)
            request_seed = _checked_gui_integer(g_seed.value, "seed")
            steps = _checked_gui_integer(g_steps.value, "sampler steps")
        except (ValueError, IndexError) as exc:
            g_status.value = str(exc)
            return
        with lock:
            state["gen"] += 1
            generation = state["gen"]
        g_status.value = "sampling …"
        requests.put(_Request(generation=generation, sample_index=index, seed=request_seed,
                              num_steps=steps))

    threading.Thread(target=_worker, daemon=True, name="wb-rollout").start()

    @b_run.on_click
    def _(_event) -> None:
        _enqueue()

    for handle in (g_sample, g_seed, g_steps):
        handle.on_update(lambda _event: _enqueue())

    @g_step.on_update
    def _(_event) -> None:
        # Dragging the slider stops playback: a play head that fights the operator's drag is the
        # one interaction bug every scrubbing UI has, and the checkbox is right there.
        #
        # ⚠️ The PLAY LOOP also writes this slider, and viser fires `on_update` for a
        # server-side write exactly as it does for a user drag — indistinguishable from here. So
        # the loop marks its own writes and this skips the auto-stop for them; without the flag
        # Play un-checked itself on its first tick, which reads as "Play does nothing".
        if state["programmatic"]:
            return
        if c_play.value:
            c_play.value = False
        _pose(int(g_step.value))

    @c_pred.on_update
    def _(_event) -> None:
        if bodies:
            bodies["prediction"]["root"].visible = bool(c_pred.value)

    @c_target.on_update
    def _(_event) -> None:
        if bodies:
            bodies["target"]["root"].visible = bool(c_target.value)

    if fk is None and bodies:
        g_status.value = f"floor unregistered: {fk_why}"
    _enqueue()

    # The play head advances on the render loop rather than a thread: the whole per-step cost is
    # two update_cfg calls (no IK, unlike hl_serving's ghost, because the joints ARE the answer),
    # so there is nothing to offload and a second thread would only race the slider.
    next_tick = time.monotonic()
    while True:
        try:
            now = time.monotonic()
            if c_play.value and now >= next_tick:
                with lock:
                    result = state["result"]
                if result is not None:
                    dt = float(result.action_times_s[1] - result.action_times_s[0]) \
                        if result.horizon > 1 else 0.033
                    next_tick = now + max(dt / max(float(g_speed.value), 0.01), 0.001)
                    nxt = (int(g_step.value) + 1) % result.horizon
                    state["programmatic"] = True
                    try:
                        g_step.value = nxt
                    finally:
                        state["programmatic"] = False
                    _pose(nxt)
            time.sleep(0.005)
        except KeyboardInterrupt:
            return


__all__ = ["PREDICTION_RGBA", "TARGET_RGBA", "run_wholebody_viewer"]
