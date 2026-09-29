#!/usr/bin/env python3
"""Offline REPLAY + RE-INFERENCE for a DEPLOY TAKE recorded by ``scripts/hl_serving.py --record``.

A recorded deploy run is another dataset, so this is its viewer — the deploy-side sibling of
``scripts/infer_s1_{flow,diffusion,act}.py``, which does the same two things for a TRAINING
sample. Two halves, and they are deliberately the same page:

  1. **REPLAY** — walk the take and see what happened: the measured LEFT TCP path (red, FK'd
     from the recorded 29-DOF joints), the plan that was executed at each moment (yellow,
     re-decoded from the stored chunk against its own stored anchor), the wire that was
     actually published (cyan), the robot itself posed at the recorded joints, and the camera
     frame + the model's own conditioning image for the selected instant.

  2. **RE-INFERENCE** — pick ANY frame and ask a checkpoint what it predicts there (orange).
     On a frame where the policy really ran, the stored observation is fed back verbatim, so
     the SAME checkpoint + seed reproduces the deploy chunk and any difference is a real
     difference. On any other frame the observation is rebuilt from the readback (see
     :meth:`~fuji.s1_policy.serving.hl_take.DeployTake.obs_at`, which states exactly what that
     can and cannot reconstruct).

WHY THIS IS NOT ``run_hl_replay_data.sh``. That drives the ROBOT from a recorded UMI
demonstration; this one touches no hardware and opens no bridge — it reads a directory and
draws it. The comparison it serves is "the policy on the rig vs the policy now", which needs
no robot and must be runnable on a workstation with the rig powered down.

Runs headless with ``--self-test`` (prints JSON, imports no viser), which is what CI and a
quick "is this take usable?" check use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_FUJI_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_UMI = os.path.join(_FUJI_ROOT, "third_party", "UMI_collection")
for _p in (
    os.path.join(_UMI, "third_party", "humi_ik"),
    os.path.join(_UMI, "viz"),
    os.path.join(_UMI, "replay"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _FUJI_ROOT not in sys.path:
    sys.path.insert(0, _FUJI_ROOT)

from fuji.s1_policy.data.layout import ACTION_DT, ACTION_HZ  # noqa: E402
from fuji.s1_policy.inference.display import TRIAD_DRAW_SIZES, triad_offsets_from_tcp  # noqa: E402
from fuji.s1_policy.serving.hl_take import DeployTake, find_takes  # noqa: E402
from fuji.s1_policy.serving.hl_trace import rotation_geodesic_deg  # noqa: E402
from fuji.utils.rot6d import rot6d_to_rotmat, rotmat_to_quat_xyzw, rotmat_to_rot6d  # noqa: E402

_TRACE_RED = (235, 60, 60)        # MEASURED path — what the arm actually did
_PLAN_YELLOW = (255, 232, 20)     # the chunk that WAS executed (re-decoded)
_REINFER_ORANGE = (255, 138, 30)  # a NEW prediction from the checkpoint under test
_WIRE_CYAN = (60, 200, 235)       # the wire that was published
_TCP_OFFSET = np.array([0.242, 0.0, 0.0], dtype=np.float64)  # wrist→UMI-TCP (URDF lever)

# The two wrench arrows, SAME RGB pair as the online planner (hl_serving._WRENCH_*_RGB) and the
# offline inference viewer's prediction orange, so a force drawn on any of the three pages is
# the same colour for the same quantity.
_WRENCH_CMD_RGB = (245, 145, 35)    # F^cmd — the PREDICTED wrench, negated
_WRENCH_MEAS_RGB = (0, 200, 220)    # F_meas — what the sensor actually felt
#: Window (s) of measured F/T drawn around the cursor in the wall-time charts. The take is
#: scrubbed, not streamed, so this is a window on a fully-known array rather than a ring buffer.
_FT_WINDOW_S = 10.0


def _log(msg: str) -> None:
    print(f"[reinfer] {msg}", file=sys.stderr, flush=True)


def _rot6d_to_wxyz(rot6d: np.ndarray) -> np.ndarray:
    q = rotmat_to_quat_xyzw(rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64)))
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64)


class ReinferSession:
    """A loaded take + an OPTIONAL checkpoint to re-infer with.

    The checkpoint is optional on purpose: replaying a take needs no model and no GPU (the
    same reason ``hl_serving.py`` supports a model-free ``--replay-zarr``), so a take can be
    inspected on a machine that has neither.
    """

    def __init__(self, take_dir: str | Path, *, checkpoint: str | None = None,
                 algo: str | None = None, device: str = "cuda", num_steps: int | None = None):
        self.take = DeployTake(take_dir)
        self.loaded = None
        self.checkpoint = checkpoint
        if checkpoint:
            if not algo:
                raise SystemExit("--checkpoint needs --algo (flow | diffusion | act)")
            from fuji.s1_policy.serving.hl_online import load_hl_model

            _log(f"loading {algo} checkpoint {checkpoint} on {device} …")
            self.loaded = load_hl_model(checkpoint, algo, device=device, num_steps=num_steps)
            # The take stores its own layout; a checkpoint with a DIFFERENT one slices the
            # action vector at other offsets, so every comparison would be meaningless. Refuse
            # loudly rather than draw two trajectories that mean different things.
            take_mode = str(self.take.meta.get("layout_mode", "left_only"))
            if self.loaded.layout.mode != take_mode:
                raise SystemExit(
                    f"checkpoint layout {self.loaded.layout.mode!r} != the take's "
                    f"{take_mode!r} — their action columns mean different things.")
            _log(f"ready: algo={self.loaded.algorithm} To={self.loaded.obs_history_len} "
                 f"steps={self.loaded.num_steps} force_hist={self.loaded.force_hist_len}")

    # -- re-inference ------------------------------------------------------
    def reinfer_prediction(self, index: int, *, seed: int | None = None) -> dict:
        """Re-run the checkpoint on RECORDED prediction ``index``'s own observation.

        The exact-reproduction path: same observation, same seed by default, so a matching
        checkpoint must return the deploy chunk. Returns the new chunk, the stored one, and
        their per-step physical difference.
        """
        if self.loaded is None:
            raise SystemExit("re-inference needs --checkpoint/--algo")
        from fuji.s1_policy.serving.hl_online import batch_from_obs, sample_chunk

        preds = self.take.predictions
        if not preds:
            raise SystemExit(f"{self.take.path} recorded no predictions to re-run")
        i = int(np.clip(index, 0, len(preds) - 1))
        pred = preds[i]
        obs = self.take.obs_from_prediction(pred, self.loaded)
        use_seed = pred.seed if seed is None else int(seed)
        batch = batch_from_obs(self.loaded, obs)
        chunk = sample_chunk(self.loaded, batch, seed=use_seed)
        return {
            "kind": "recorded_prediction",
            "index": i,
            "seed": int(use_seed),
            "seed_matches_deploy": bool(use_seed == pred.seed),
            "chunk": chunk,
            "stored_chunk": pred.chunk,
            "anchor_pos": pred.anchor_pos,
            "anchor_rot6d": pred.anchor_rot6d,
            "diff": self._chunk_diff(chunk, pred.chunk),
        }

    def reinfer_frame(self, frame: int, *, seed: int = 0) -> dict:
        """Re-run the checkpoint at measured TICK ``frame``, from a rebuilt observation."""
        if self.loaded is None:
            raise SystemExit("re-inference needs --checkpoint/--algo")
        from fuji.s1_policy.serving.hl_online import batch_from_obs, sample_chunk

        obs, anchor_pos, anchor_rot6d = self.take.obs_at(frame, self.loaded)
        batch = batch_from_obs(self.loaded, obs)
        chunk = sample_chunk(self.loaded, batch, seed=int(seed))
        return {
            "kind": "rebuilt_frame",
            "frame": int(frame),
            "seed": int(seed),
            "chunk": chunk,
            "stored_chunk": None,
            "anchor_pos": anchor_pos,
            "anchor_rot6d": anchor_rot6d,
            "force_history_valid": bool(obs.force_history_valid),
            "diff": None,
        }

    def _chunk_diff(self, new: np.ndarray, old: np.ndarray) -> dict:
        """PHYSICAL difference between two chunks of the same layout.

        Reported in the same units + definitions as the offline viewer's Metrics panel (cm,
        geodesic degrees, mm, N), so "how far apart are these two models" reads the same
        everywhere in the repo instead of inventing a third scale."""
        layout = self.loaded.layout
        if new.shape != old.shape:
            return {"shape_mismatch": [list(new.shape), list(old.shape)]}
        out: dict = {"identical": bool(np.array_equal(new, old))}
        for role in layout.roles:
            slots = layout.slots(role)
            pos = np.linalg.norm(new[:, slots.pos] - old[:, slots.pos], axis=-1) * 100.0
            rot = rotation_geodesic_deg(new[:, slots.rot6d], old[:, slots.rot6d])
            entry = {
                "position_cm_mean": float(np.mean(pos)),
                "position_cm_max": float(np.max(pos)),
                "rotation_deg_mean": float(np.mean(rot)),
                "rotation_deg_max": float(np.max(rot)),
            }
            if slots.gripper is not None:
                from fuji.s1_policy.data.transforms import GRIPPER_WIDTH_REF_M

                g = np.abs(new[:, slots.gripper] - old[:, slots.gripper]) * GRIPPER_WIDTH_REF_M
                entry["gripper_mm_mean"] = float(np.mean(g) * 1000.0)
            if slots.force is not None:
                f = np.linalg.norm(
                    new[:, slots.force.start:slots.force.start + 3]
                    - old[:, slots.force.start:slots.force.start + 3], axis=-1)
                entry["force_n_mean"] = float(np.mean(f))
            out[role] = entry
        return out

    def decode_positions(self, chunk: np.ndarray, anchor_pos, anchor_rot6d,
                         role: str = "left") -> np.ndarray:
        """A chunk → absolute pelvis-frame TCP positions against a given anchor."""
        from fuji.s1_policy.data.transforms import delta_ee_to_absolute

        layout = (self.loaded.layout if self.loaded is not None
                  else __import__("fuji.s1_policy.data.layout", fromlist=["get_layout"])
                  .get_layout(str(self.take.meta.get("layout_mode", "left_only"))))
        ri = layout.roles.index(role)
        slots = layout.slots(role)
        pos, _ = delta_ee_to_absolute(
            chunk[:, slots.pos], chunk[:, slots.rot6d],
            np.asarray(anchor_pos)[ri], np.asarray(anchor_rot6d)[ri])
        return pos

    # -- replay statistics -------------------------------------------------
    def tracking_stats(self) -> dict:
        """Measured-vs-PLANNED tracking error, per executed prediction.

        The plan is re-decoded against its own stored anchor into the SAME pelvis-frame TCP the
        measured stream holds, so the difference is servo/IK tracking error and nothing else —
        which is why this compares against the decoded chunk rather than against the published
        wire (the wire is the LEFT CONTACT point, a different point on the same body).
        """
        take = self.take
        if not len(take):
            return {"n_measured": 0, "predictions": []}
        t = take.times
        m_pos, m_rot = take.measured_tcp()
        out = []
        for pred in take.predictions:
            if not np.isfinite(pred.t_exec_start_s) or pred.chunk.size == 0:
                continue
            plan_pos = take.decoded_chunk_positions(pred)
            n_exec = max(1, min(int(pred.n_exec), plan_pos.shape[0]))
            # The measured rows this plan owns, by the chunk index the recorder stamped — not
            # by a time window, so an interrupted or slow chunk is still attributed correctly.
            rows = np.flatnonzero(
                take.executed[:, take._exec_col["chunk_index"]] == pred.index)  # noqa: SLF001
            if rows.size < 2:
                continue
            # Put the plan on the measured clock: step k was commanded at t_exec_start + k*dt.
            plan_t = float(pred.t_exec_start_s) + np.arange(n_exec) * ACTION_DT
            ref_pos = np.stack([np.interp(t[rows], plan_t, plan_pos[:n_exec, k])
                                for k in range(3)], axis=1)
            d = np.linalg.norm(m_pos[rows] - ref_pos, axis=1)
            out.append({
                "index": pred.index,
                "source": pred.source,
                "n_exec": int(pred.n_exec),
                "n_measured": int(rows.size),
                "pos_rms_cm": float(np.sqrt(np.mean(d ** 2)) * 100.0),
                "pos_max_cm": float(np.max(d) * 100.0),
                "path_measured_cm": float(
                    np.sum(np.linalg.norm(np.diff(m_pos[rows], axis=0), axis=1)) * 100.0),
                "path_planned_cm": float(
                    np.sum(np.linalg.norm(np.diff(plan_pos[:n_exec], axis=0), axis=1)) * 100.0),
            })
        return {"n_measured": len(take), "predictions": out}

    def summary(self) -> dict:
        s = self.take.summary()
        s["tracking"] = self.tracking_stats()
        if self.loaded is not None:
            s["checkpoint"] = {
                "path": str(self.checkpoint),
                "algo": self.loaded.algorithm,
                "layout": self.loaded.layout.mode,
                "num_steps": int(self.loaded.num_steps),
            }
        return s

    def close(self) -> None:
        pass


def _self_test(session: ReinferSession, *, reinfer_index: int | None, seed: int | None) -> int:
    """Headless: load, summarize, and (with a checkpoint) re-infer one prediction."""
    payload = session.summary()
    if session.loaded is not None and session.take.predictions:
        i = 0 if reinfer_index is None else int(reinfer_index)
        res = session.reinfer_prediction(i, seed=seed)
        payload["reinference"] = {
            "kind": res["kind"],
            "index": res["index"],
            "seed": res["seed"],
            "seed_matches_deploy": res["seed_matches_deploy"],
            "chunk_shape": list(np.asarray(res["chunk"]).shape),
            "finite": bool(np.isfinite(res["chunk"]).all()),
            "diff_vs_deploy": res["diff"],
        }
    print(json.dumps(payload, indent=2, default=float))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("take", help="a recorded take_NNN directory, or a session directory "
                                "containing them (then --take-index picks one)")
    p.add_argument("--take-index", type=int, default=0,
                   help="which take to open when TAKE is a session directory (default 0)")
    p.add_argument("--checkpoint", default=None,
                   help="HL-v2 checkpoint to RE-INFER with. Omit to replay the take only "
                        "(no model, no GPU needed).")
    p.add_argument("--algo", choices=["flow", "diffusion", "act"], default=None,
                   help="policy family of --checkpoint (required with it)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-steps", type=int, default=None,
                   help="sampler steps override (default from the checkpoint)")
    p.add_argument("--frame", type=int, default=0, help="initial measured frame to select")
    p.add_argument("--reinfer-index", type=int, default=None,
                   help="which RECORDED prediction --self-test re-runs (default 0)")
    p.add_argument("--seed", type=int, default=None,
                   help="sampling seed; default REUSES each recorded prediction's own seed, "
                        "which is what makes a same-checkpoint re-inference reproduce the "
                        "deploy chunk exactly")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8089,
                   help="viser port (default 8089 — clear of the offline viewers' 8086/8087 "
                        "and the online planner's 8088, so all four can run at once)")
    p.add_argument("--yam-urdf", default=os.path.join(
        _UMI, "robot_assets", "g1_yam", "g1_29dof_yam.urdf"))
    p.add_argument("--self-test", action="store_true",
                   help="load, summarize + re-infer one prediction headlessly, print JSON, and "
                        "exit without importing viser")
    args = p.parse_args(argv)

    takes = find_takes(args.take)
    if not takes:
        raise SystemExit(f"no take_NNN directories under {args.take}")
    if not 0 <= args.take_index < len(takes):
        raise SystemExit(f"--take-index {args.take_index} outside [0, {len(takes) - 1}] "
                         f"({len(takes)} take(s) in {args.take})")
    take_dir = takes[args.take_index]
    _log(f"opening {take_dir}  ({len(takes)} take(s) available)")

    session = ReinferSession(take_dir, checkpoint=args.checkpoint, algo=args.algo,
                             device=args.device, num_steps=args.num_steps)
    try:
        if args.self_test:
            return _self_test(session, reinfer_index=args.reinfer_index, seed=args.seed)
        _run_viewer(session, args)
        return 0
    finally:
        session.close()


def _make_step_chart(gui, uplot, rc, title: str, labels):
    """A 3-line chart on the CHUNK-STEP axis (not time) — the twin of ``hl_serving``'s.

    Same shape and axis colours as ``rc.make_xyz_chart`` so the predicted and measured charts
    read alike, but the x axis is a step index, which is why it is not that function.
    """
    ac = rc.AXIS_COLORS
    x0 = np.arange(2, dtype=np.float64)
    zero = np.zeros(2)
    return gui.add_uplot(
        data=(x0, zero, zero, zero),
        series=(uplot.Series(label="step"),
                uplot.Series(label=labels[0], stroke=ac[0], width=1.5),
                uplot.Series(label=labels[1], stroke=ac[1], width=1.5),
                uplot.Series(label=labels[2], stroke=ac[2], width=1.5)),
        scales={"x": uplot.Scale(time=False)}, title=title, aspect=2.2, height=120)


def _set_step_chart(chart, xyz: np.ndarray) -> None:
    if xyz.shape[0] < 2:
        return
    chart.data = (np.arange(xyz.shape[0], dtype=np.float64),
                  xyz[:, 0], xyz[:, 1], xyz[:, 2])


def _run_viewer(session: ReinferSession, args) -> None:
    """The viser page: replay the take, re-infer on demand."""
    import viser
    from viser import uplot
    from viser.extras import ViserUrdf

    import replay_common as rc  # third_party/UMI_collection/replay (on sys.path above)
    from fuji.s1_policy.serving.live_robot import (
        JAW_JOINT_OPEN_M,
        build_sdk_to_urdf_index,
        jaw_joint_from_width_m,
        remap_dof_to_urdf,
    )
    from fuji.utils.force_frame import (
        CMD_FROM_MEAS_NOTE,
        TCP_FROM_SENSOR_NOTE,
        cmd_from_meas_wrench,
    )

    take = session.take
    if not len(take):
        raise SystemExit(f"{take.path} has no measured frames to replay (was the low level "
                         "running with RECORD=1 when this take was made?)")

    server = viser.ViserServer(host=args.host, port=args.port, label="FUJI deploy-take replay")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", plane="xy", shadow_opacity=0.0, infinite_grid=True,
                          cell_size=0.25)
    # The robot is drawn at the PELVIS origin, which is the frame every recorded pose is in —
    # so no display offset is applied anywhere in this viewer and a drawn point IS its stored
    # value. (The online planner adds its mink base_xyz; the recorder subtracts it before
    # storing, precisely so this viewer needs no such constant.)
    server.scene.add_frame("/robot", show_axes=False)
    urdf = ViserUrdf(server, urdf_or_path=Path(args.yam_urdf), root_node_name="/robot")
    remap = build_sdk_to_urdf_index(list(urdf.get_actuated_joint_names()))
    dof_t, dof_q = take.measured_dof()
    m_pos, m_rot = take.measured_tcp()
    times = take.times
    jaw = take.measured_jaw()
    # The force + jaw streams. Each carries its own frame/sign (see the accessors' docstrings);
    # nothing here re-expresses them — that happens at exactly one place per consumer, either
    # WrenchArrows (arrows) or cmd_from_meas_wrench (the F^cmd charts).
    ft_t, ft_wrench = take.measured_wrench_sensor()          # sensor frame, MEASURED sign
    servo_t, servo_f = take.servo_force_pelvis("left")       # PELVIS frame, MEASURED sign
    pub_t, pub_fcmd = take.commanded_force_sensor()          # sensor frame, ALREADY the command
    cmd_jaw_t, cmd_jaw = take.commanded_jaw()
    has_ft = ft_wrench.shape[0] > 1

    # TWO WRENCH ARROWS ON THE TCP, the same pair the online planner draws:
    #   /wrench       F^cmd  — the model's PREDICTED wrench, NEGATED (`as_command=True`)
    #   /wrench_meas  F_meas — what the Bota felt, drawn AS MEASURED (`as_command=False`)
    # Both must be TOP-LEVEL nodes: they are fed WORLD-frame origins and viser parents by path
    # prefix, so hanging them under /follow would offset them by the follow frame's own position
    # and draw the arrow at ~2x the wrist. (Same trap documented in hl_serving.)
    #
    # ANCHOR: both ride the MEASURED TCP. On the rig the planner splits them (F^cmd on the ghost
    # that previews the plan, F_meas on the live robot); here there is one robot and it is the
    # measured one, so a single anchor is the honest choice — and it is what makes "opposite and
    # equal length" readable, since a shared origin removes the position gap from the comparison.
    # The frame is `m_rot`, which is URDF-frame (`measured_tcp`), i.e. the real scene orientation.
    # The decoded chunk's rotation would be MODEL/puck frame and would need the Rx(180) seam, so
    # deliberately NOT used here.
    wrench_cmd = rc.WrenchArrows(server, name="/wrench", show_torque=False,
                                 force_color=_WRENCH_CMD_RGB, torque_color=(250, 205, 130))
    wrench_meas = rc.WrenchArrows(server, name="/wrench_meas", show_torque=False,
                                  as_command=False, force_color=_WRENCH_MEAS_RGB,
                                  torque_color=(140, 215, 235))

    # The tcp / tracker / wrist_yaw_link triad cluster, hung off a frame that rides the
    # MEASURED wrist — the same three triads the collection viz, the zarr replay and the
    # online planner draw, so this viewer can be compared to them triad-for-triad.
    follow = server.scene.add_frame("/follow", show_axes=False, origin_radius=0.0)
    from fuji.s1_policy.inference.display import PUCK_TO_WRIST_RELABEL

    for name, off in triad_offsets_from_tcp().items():
        length, radius, origin = TRIAD_DRAW_SIZES[name]
        rot = PUCK_TO_WRIST_RELABEL if name == "tracker" else np.eye(3)
        server.scene.add_frame(f"/follow/{name}", show_axes=True, axes_length=length,
                               axes_radius=radius, origin_radius=origin,
                               position=tuple(_TCP_OFFSET + off),
                               wxyz=tuple(_rot6d_to_wxyz(rotmat_to_rot6d(rot))))
        server.scene.add_label(f"/follow/{name}/label", name)

    gui = server.gui
    gui.configure_theme(control_layout="collapsible", control_width="large")
    with gui.add_folder("Take"):
        gui.add_text("dir", initial_value=take.path.name, disabled=True)
        gui.add_text("label", initial_value=take.label or "(none)", disabled=True)
        gui.add_text("measured", initial_value=f"{len(take)} ticks / "
                                               f"{float(take.meta.get('duration_s', 0.0)):.1f} s",
                     disabled=True)
        gui.add_text("plans executed", initial_value=str(len(take.predictions)), disabled=True)
        gui.add_text("retries discarded",
                     initial_value=str(take.meta.get("n_predicts_discarded", 0)), disabled=True)
        # Say it out loud when the reader had to repair the take's clock, because the repair is
        # exact only up to the two streams' first-arrival difference (tens of ms).
        if take.legacy_clock_streams:
            gui.add_text("⚠ clock rebased",
                         initial_value=(f"{', '.join(take.legacy_clock_streams)} were recorded on "
                                        f"the raw monotonic clock (pre-2026-08-17 take) and have "
                                        f"been re-zeroed on read; alignment is ±one 30 Hz step"),
                         disabled=True)
        gui.add_text("grid", initial_value=f"{ACTION_HZ:g} Hz x "
                                          f"{take.meta.get('grid', {}).get('action_horizon', '?')}",
                     disabled=True)
    with gui.add_folder("Model"):
        gui.add_text("checkpoint",
                     initial_value=(os.path.basename(str(session.checkpoint))
                                    if session.loaded is not None else "(none — replay only)"),
                     disabled=True)
        reinfer_btn = gui.add_button("🔮 Re-infer at this frame",
                                     disabled=session.loaded is None)
        reinfer_exact_btn = gui.add_button(
            "🎯 Re-infer THIS take's plan (exact obs)",
            disabled=(session.loaded is None or not take.predictions))
        seed_slider = gui.add_slider("seed", min=0, max=1000, step=1,
                                     initial_value=int(args.seed or 0),
                                     disabled=session.loaded is None)
        reinfer_status = gui.add_text("status", initial_value="(nothing re-inferred)",
                                      disabled=True)
    with gui.add_folder("Frame"):
        frame_slider = gui.add_slider("frame", min=0, max=max(len(take) - 1, 1), step=1,
                                      initial_value=int(np.clip(args.frame, 0, len(take) - 1)))
        play_toggle = gui.add_checkbox("▶ Play", initial_value=False)
        frame_readout = gui.add_text("t", initial_value="-", disabled=True)
        plan_readout = gui.add_text("plan", initial_value="-", disabled=True)
    with gui.add_folder("View"):
        show_measured = gui.add_checkbox("Measured path (red)", initial_value=True)
        show_plan = gui.add_checkbox("Executed plan (yellow)", initial_value=True)
        show_wire = gui.add_checkbox("Published wire (cyan)", initial_value=False)
        show_reinfer = gui.add_checkbox("Re-inference (orange)", initial_value=True)
        # The arrows, worded exactly as the planner words them so the pages read alike.
        show_wrench_cmd = gui.add_checkbox(
            "Show F^cmd arrow — ORANGE, on the TCP (from predicted F_meas)", initial_value=True)
        show_wrench_meas = gui.add_checkbox(
            "Show F_meas arrow — CYAN, on the TCP (what the sensor felt)",
            initial_value=has_ft, disabled=not has_ft)
        show_torque_arrows = gui.add_checkbox("…also torque arrows", initial_value=False)
        gui.add_text("Arrow frame", initial_value=TCP_FROM_SENSOR_NOTE, disabled=True)
        gui.add_text("Arrow sign", initial_value=CMD_FROM_MEAS_NOTE, disabled=True)
    with gui.add_folder("Images"):
        cam_img = gui.add_image(np.zeros((16, 16, 3), dtype=np.uint8), label="camera",
                                format="jpeg")
        obs_img = gui.add_image(np.zeros((16, 16, 3), dtype=np.uint8),
                                label="model obs (t0)", format="jpeg")
    # --- GRIPPER ------------------------------------------------------------------------------
    # The mesh jaw already animates (see the width→joint conversion in _select); this panel is
    # the NUMERIC half, because the interesting question about a grasp is not "is it shut" but
    # "did it shut as far as it was told". A persistent measured-above-commanded gap IS the grip
    # force: a position-only jaw converts position error into a squeeze, so on a successful grasp
    # these two curves are SUPPOSED to separate and stay separated.
    with gui.add_folder("Gripper (jaw)"):
        grip_readout = gui.add_text("width", initial_value="-", disabled=True)
        grip_pred_readout = gui.add_text("predicted @ step", initial_value="-", disabled=True)
        jaw_chart = gui.add_uplot(
            data=(np.array([0.0, 1.0]), np.zeros(2), np.zeros(2)),
            series=(uplot.Series(label="t (s)"),
                    uplot.Series(label="measured", stroke="#e03c3c", width=1.5),
                    uplot.Series(label="commanded", stroke="#3ce0c8", width=1.5)),
            scales={"x": uplot.Scale(time=False)},
            title="jaw aperture (mm) — measured vs commanded, whole take",
            aspect=2.2, height=120)
        jaw_pred_chart = gui.add_uplot(
            data=(np.array([0.0, 1.0]), np.zeros(2)),
            series=(uplot.Series(label="step"),
                    uplot.Series(label="predicted", stroke="#f59123", width=1.5)),
            scales={"x": uplot.Scale(time=False)},
            title="predicted jaw (mm) — this plan, per chunk step", aspect=2.2, height=110)

    # --- FORCE --------------------------------------------------------------------------------
    # Two X AXES on purpose, the same split the planner uses: the PREDICTED charts are per chunk
    # STEP (0…H-1 of one plan), the MEASURED ones are per SECOND of the take's own clock. Mixing
    # them onto one axis would imply a correspondence that a re-anchored chunk cadence does not
    # have.
    with gui.add_folder("Force (charts)"):
        fcmd_chart = _make_step_chart(gui, uplot, rc, "F^cmd = -predicted F_meas (N)",
                                      ("Fx", "Fy", "Fz"))
        tcmd_chart = _make_step_chart(gui, uplot, rc, "T^cmd = -predicted T_meas (Nm)",
                                      ("Tx", "Ty", "Tz"))
        # MEASURED: sensor frame, measured sign — no relabel, no negation. Same fidelity split
        # as the replay viewers and the planner (only the ARROW is re-expressed).
        fmeas_chart = rc.make_xyz_chart(gui, "F_meas force (N) — measured, sensor frame",
                                        ("Fx", "Fy", "Fz"))
        tmeas_chart = rc.make_xyz_chart(gui, "F_meas torque (Nm) — measured, sensor frame",
                                        ("Tx", "Ty", "Tz"))
        gui.add_text("F_meas charts",
                     initial_value=(f"±{_FT_WINDOW_S / 2:g} s around the cursor · raw Bota "
                                    f"({ft_wrench.shape[0]} samples) · NOT negated · NOT relabelled"
                                    if has_ft else "no force_meas.npy in this take"),
                     disabled=True)
        force_readout = gui.add_text("|F| at cursor", initial_value="-", disabled=True)
        # The published wire's force answers "did force actually leave this process?" — all-zero
        # is what a pure-position run looks like, and it is worth being able to prove.
        _pub_peak = (float(np.nanmax(np.linalg.norm(pub_fcmd, axis=1)))
                     if pub_fcmd.shape[0] else float("nan"))
        gui.add_text("published |F^cmd| peak",
                     initial_value=(f"{_pub_peak:.2f} N"
                                    + ("  (force switch was OFF / all sub-gate)"
                                       if not np.isfinite(_pub_peak) or _pub_peak < 1e-9
                                       else "  — force WAS published on the wire")),
                     disabled=True)

    with gui.add_folder("Tracking (measured vs planned)"):
        track_readout = gui.add_text("per plan", initial_value="(computing…)", disabled=True)

    # The jaw pair is STATIC (the whole take, set once): it is a fully-known array, so a window
    # would only hide the grasp you are looking for. mm, and on the take's own clock.
    if len(take) > 1:
        jaw_chart.data = (
            np.asarray(times, dtype=np.float64),
            np.asarray(jaw, dtype=np.float64) * 1000.0,
            (np.interp(times, cmd_jaw_t, cmd_jaw) * 1000.0 if cmd_jaw.shape[0] > 1
             else np.full(len(take), np.nan)),
        )

    stats = session.tracking_stats()
    if stats["predictions"]:
        track_readout.value = "  ".join(
            f"#{e['index']}: RMS {e['pos_rms_cm']:.2f} cm (max {e['pos_max_cm']:.2f}), "
            f"path {e['path_measured_cm']:.1f}/{e['path_planned_cm']:.1f} cm"
            for e in stats["predictions"])
    else:
        track_readout.value = "no plan had enough measured samples to compare"

    handles: dict[str, object] = {"measured": None, "plan": None, "wire": None, "reinfer": None}

    def _draw_line(key: str, pts: np.ndarray, color, width: float) -> None:
        h = handles.get(key)
        if h is not None:
            h.remove()
            handles[key] = None
        if pts is None or pts.shape[0] < 2:
            return
        seg = np.stack([pts[:-1], pts[1:]], axis=1).astype(np.float32)
        col = np.broadcast_to(np.array(color, np.uint8), (seg.shape[0], 2, 3)).copy()
        handles[key] = server.scene.add_line_segments(f"/{key}", points=seg, colors=col,
                                                      line_width=width)

    def _redraw_static() -> None:
        _draw_line("measured", m_pos if bool(show_measured.value) else None, _TRACE_RED, 3.0)
        if bool(show_wire.value):
            wire = take.commanded
            if wire.size:
                cols = [take._cmd_col[k] for k in ("pos_lx", "pos_ly", "pos_lz")]  # noqa: SLF001
                _draw_line("wire", wire[:, cols], _WIRE_CYAN, 2.0)
        else:
            _draw_line("wire", None, _WIRE_CYAN, 2.0)

    reinfer_cache: dict = {"pts": None}

    def _select(frame: int) -> None:
        i = int(np.clip(frame, 0, len(take) - 1))
        t = float(times[i])
        frame_readout.value = f"frame {i}/{len(take) - 1}  ·  t = {t:.2f} s"
        # Pose the robot at the RECORDED joints (nearest readback row), so the mesh is the
        # real configuration rather than an IK reconstruction of the TCP.
        if dof_q.shape[0]:
            j = int(np.argmin(np.abs(dof_t - t)))
            try:
                # THE JAW OPENS AND CLOSES. `jaw[i]` is a full APERTURE (m); the URDF finger
                # joint is a per-finger displacement with HALF that scale, so it needs
                # converting — passing the width straight through saturates the whole
                # 43.8-94.3 mm sweep at wide open (it runs past the 47.5 mm travel limit),
                # which is why the gripper looked frozen. The right jaw is pinned OPEN: takes
                # carry a LEFT jaw only, and 0.0 is this joint's SHUT end.
                urdf.update_cfg(remap_dof_to_urdf(
                    dof_q[j], remap, left_jaw_m=jaw_joint_from_width_m(jaw[i]),
                    right_jaw_m=JAW_JOINT_OPEN_M))
            except Exception as e:  # noqa: BLE001 — viz only
                _log(f"mesh pose error (ignored): {e!r}")
        R = rot6d_to_rotmat(m_rot[i])
        follow.position = tuple(m_pos[i] - R @ _TCP_OFFSET)
        follow.wxyz = tuple(_rot6d_to_wxyz(m_rot[i]))

        pred = take.prediction_for_time(t)
        if pred is not None and bool(show_plan.value):
            plan_pos = take.decoded_chunk_positions(pred)
            n = max(1, min(int(pred.n_exec), plan_pos.shape[0]))
            _draw_line("plan", plan_pos[:n], _PLAN_YELLOW, 3.0)
            plan_readout.value = (f"plan #{pred.index} ({pred.source}) seed {pred.seed}, "
                                  f"executed {pred.n_exec} of {pred.chunk.shape[0]}")
            if pred.images:
                obs_img.image = pred.images[-1]  # t0 = the newest conditioning frame
        else:
            _draw_line("plan", None, _PLAN_YELLOW, 3.0)
            if pred is None:
                plan_readout.value = "no plan was executing at this instant"
        ci = take.camera_index_for_time(t)
        if ci is not None:
            frame_rgb = take.camera_frame(ci)
            if frame_rgb is not None:
                cam_img.image = frame_rgb
        _update_force_and_jaw(i, t, pred)

    def _update_force_and_jaw(i: int, t: float, pred) -> None:
        """The gripper readouts + both wrench arrows + all four force charts, for one cursor.

        Kept out of ``_select`` only for length; it runs on every frame change and on every
        arrow/torque toggle, so it must stay allocation-light and must never raise (a viz error
        here would break scrubbing).
        """
        tcp_xyz = m_pos[i]
        tcp_wxyz = _rot6d_to_wxyz(m_rot[i])
        torque_on = bool(show_torque_arrows.value)

        # -- gripper numbers ------------------------------------------------------------------
        w_meas = float(jaw[i]) if i < jaw.shape[0] else float("nan")
        w_cmd = (float(np.interp(t, cmd_jaw_t, cmd_jaw)) if cmd_jaw.shape[0] > 1
                 else float("nan"))
        _gap_mm = (w_meas - w_cmd) * 1000.0
        grip_readout.value = (
            f"measured {w_meas * 1000:.1f} mm"
            + (f" · commanded {w_cmd * 1000:.1f} mm · Δ{_gap_mm:+.1f}"
               + (" (SQUEEZING)" if _gap_mm > 1.0 else "") if np.isfinite(w_cmd) else "")
        )

        # -- the plan's own predicted force + jaw, at the step the arm was really on -----------
        # `executed_step_at` is the recorder's own stamp, so this is the exact step rather than
        # an interpolation — and it stays correct for a chunk that was interrupted or ran slow.
        f_pred = t_pred = None
        # Built up locally and assigned ONCE at the end: `force_readout.value` is a GUI handle
        # that reads back the PREVIOUS frame's string, so appending to it in place would
        # accumulate across every scrub.
        parts: list[str] = []
        if pred is not None:
            wren = take.predicted_wrench(pred)
            jaw_pred = take.predicted_jaw(pred)
            _, step = take.executed_step_at(i)
            if jaw_pred is not None:
                jaw_pred_chart.data = (np.arange(jaw_pred.shape[0], dtype=np.float64),
                                       jaw_pred * 1000.0)
            if wren is not None and wren.shape[0]:
                k = int(np.clip(step if step >= 0 else 0, 0, wren.shape[0] - 1))
                # ONE conversion, HERE, for the charts: measured→command. The arrow does its own
                # (WrenchArrows(as_command=True)), so the raw prediction — not this — is what
                # gets handed to it; pre-negating for both would cancel invisibly.
                cmd = cmd_from_meas_wrench(wren)
                _set_step_chart(fcmd_chart, cmd[:, :3])
                _set_step_chart(tcmd_chart, cmd[:, 3:6])
                f_pred, t_pred = wren[k, :3], wren[k, 3:6]
                if jaw_pred is not None:
                    grip_pred_readout.value = (
                        f"{jaw_pred[k] * 1000:.1f} mm at step {k} of {jaw_pred.shape[0]}")
                parts.append(f"|F^cmd| {np.linalg.norm(f_pred):.2f} N (step {k})")
        if not parts:
            parts.append("|F^cmd| — (no plan at this instant)")

        # -- the arrows ------------------------------------------------------------------------
        # RAW MEASURED sensor triples go in; WrenchArrows applies sign (F^cmd only), the
        # sensor→TCP relabel and the frame rotation. Callers must not pre-apply any of them.
        # A NON-FINITE wrench must never reach WrenchArrows: it does not check, so `mag < 1e-6`
        # is False for NaN and the arrow would be drawn with NaN endpoints (viser geometry at
        # infinity). Hide instead — the same rule force_arrow_local_segment applies by returning
        # None.
        if f_pred is not None and not np.isfinite(f_pred).all():
            f_pred = t_pred = None
        wrench_cmd.show_torque = torque_on
        wrench_cmd.visible = bool(show_wrench_cmd.value) and f_pred is not None
        wrench_cmd.update(tcp_xyz, tcp_wxyz,
                          np.zeros(3) if f_pred is None else f_pred,
                          None if (t_pred is None or not torque_on) else t_pred)

        f_meas = t_meas = None
        if has_ft:
            k = int(np.argmin(np.abs(ft_t - t)))
            if np.isfinite(ft_wrench[k]).all():
                f_meas, t_meas = ft_wrench[k, :3], ft_wrench[k, 3:6]
        wrench_meas.show_torque = torque_on
        wrench_meas.visible = bool(show_wrench_meas.value) and f_meas is not None
        wrench_meas.update(tcp_xyz, tcp_wxyz,
                           np.zeros(3) if f_meas is None else f_meas,
                           None if (t_meas is None or not torque_on) else t_meas)

        # -- measured charts: a window around the cursor, on the take's own clock --------------
        if has_ft:
            lo, hi = t - _FT_WINDOW_S / 2.0, t + _FT_WINDOW_S / 2.0
            sel = np.flatnonzero((ft_t >= lo) & (ft_t <= hi))
            if sel.size > 1:
                w = ft_wrench[sel]
                ts = ft_t[sel]
                fmeas_chart.data = (ts, w[:, 0], w[:, 1], w[:, 2])
                tmeas_chart.data = (ts, w[:, 3], w[:, 4], w[:, 5])
            # PEAK over the window, not the cursor sample: the number to read against the 40 N
            # hard stop. Reported next to the servo's own pelvis-frame |F| so the whole F/T
            # chain can be cross-checked — the two are different frames but |F| is invariant,
            # so a real difference is tare / RNEA / the deploy-side EMA, never geometry.
            # nanmax over the window, and only when the window is non-empty: an empty selection
            # (or an all-NaN one) must read as "no data" rather than propagate a NaN into the
            # number the operator compares against the 40 N stop.
            _mags = np.linalg.norm(ft_wrench[sel][:, :3], axis=1) if sel.size else np.empty(0)
            _peak = (float(np.nanmax(_mags))
                     if _mags.size and np.isfinite(_mags).any() else float("nan"))
            _servo = (float(np.linalg.norm(servo_f[i]))
                      if servo_f.shape[0] > i and np.isfinite(servo_f[i]).all() else float("nan"))
            parts.append(f"|F_meas| {np.linalg.norm(f_meas):.2f} N "
                         f"(peak {_peak:.2f} over {_FT_WINDOW_S:g} s)")
            parts.append(f"servo |F| {_servo:.2f} N")
        force_readout.value = " · ".join(parts)

    def _do_reinfer(exact: bool) -> None:
        if session.loaded is None:
            return
        try:
            if exact:
                pred = take.prediction_for_time(float(times[int(frame_slider.value)]))
                idx = pred.index if pred is not None else 0
                res = session.reinfer_prediction(idx, seed=None)
                d = res["diff"] or {}
                left = d.get("left", {})
                reinfer_status.value = (
                    f"plan #{res['index']} re-run with its own seed {res['seed']}: "
                    + ("IDENTICAL to deploy" if d.get("identical") else
                       f"Δpos {left.get('position_cm_mean', float('nan')):.3f} cm mean / "
                       f"{left.get('position_cm_max', float('nan')):.3f} max, "
                       f"Δrot {left.get('rotation_deg_mean', float('nan')):.2f}°"))
            else:
                res = session.reinfer_frame(int(frame_slider.value),
                                           seed=int(seed_slider.value))
                reinfer_status.value = (
                    f"frame {res['frame']} seed {res['seed']}: rebuilt obs "
                    f"(force {'measured' if res['force_history_valid'] else 'NO DATA'}) → "
                    f"{res['chunk'].shape[0]} steps")
            pts = session.decode_positions(res["chunk"], res["anchor_pos"], res["anchor_rot6d"])
            reinfer_cache["pts"] = pts
            _draw_line("reinfer", pts if bool(show_reinfer.value) else None,
                       _REINFER_ORANGE, 3.0)
        except Exception as e:  # noqa: BLE001 — a failed re-inference must not kill the page
            import traceback

            _log("re-inference error:\n" + traceback.format_exc())
            reinfer_status.value = f"error: {e!r}"

    @frame_slider.on_update
    def _(_e) -> None:
        _select(int(frame_slider.value))

    @reinfer_btn.on_click
    def _(_e) -> None:
        _do_reinfer(exact=False)

    @reinfer_exact_btn.on_click
    def _(_e) -> None:
        _do_reinfer(exact=True)

    for _toggle in (show_measured, show_plan, show_wire):
        @_toggle.on_update
        def _(_e) -> None:
            _redraw_static()
            _select(int(frame_slider.value))

    # The arrow toggles need no line redraw — just re-run the arrow/chart update for the cursor.
    for _toggle in (show_wrench_cmd, show_wrench_meas, show_torque_arrows):
        @_toggle.on_update
        def _(_e) -> None:
            _i = int(frame_slider.value)
            _update_force_and_jaw(_i, float(times[_i]),
                                  take.prediction_for_time(float(times[_i])))

    @show_reinfer.on_update
    def _(_e) -> None:
        _draw_line("reinfer", reinfer_cache["pts"] if bool(show_reinfer.value) else None,
                   _REINFER_ORANGE, 3.0)

    _redraw_static()
    _select(int(frame_slider.value))
    print(f"[reinfer] http://{args.host}:{server.get_port()}  — replay {take.path.name}"
          + ("" if session.loaded is None else "; press Re-infer to run the checkpoint"),
          flush=True)

    # Playback on the take's OWN clock (the measured stream's timestamps), so the replay runs
    # at the speed the arm really moved rather than one frame per render tick.
    import time as _time

    last = {"t": _time.monotonic(), "i": int(frame_slider.value)}
    while True:
        if bool(play_toggle.value) and len(take) > 1:
            now = _time.monotonic()
            i = int(frame_slider.value)
            dt = float(times[min(i + 1, len(take) - 1)] - times[i])
            if now - last["t"] >= max(dt, 1e-3):
                last["t"] = now
                nxt = i + 1
                if nxt >= len(take):
                    play_toggle.value = False
                else:
                    frame_slider.value = nxt
        _time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    raise SystemExit(main())
