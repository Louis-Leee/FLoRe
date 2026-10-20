"""Live viser view of what --mode 5pt/dense is SENDING, against the pose the gate WANTS.

    T1:  MODE=5 bash force_deploy/1_vive_bridge.sh --publish-mc-qpos
    T4:  bash force_deploy/4_pose_match_viz.sh          # this module

WHAT PROBLEM THIS SOLVES
========================
ABSOLUTE teleop refuses to arm until the operator's points are within tolerance of the
robot's HELD REFERENCE, and until now NOTHING showed either one. The operator saw five
distances and no target, so "too far" was unactionable — on the rig the reference turned out
to be 0.44 m straight AHEAD of the pelvis and 0.10 m above it (forearms horizontal) while
the operator held the pucks up at chest height, 0.9 m away. This draws both bodies in one
scene and mirrors the gate's arithmetic, so matching the pose becomes a thing you can see.

WHY IT IS A SEPARATE PROCESS, AND IN A DIFFERENT INTERPRETER
===========================================================
It touches neither T1 nor T3. Both reasons are load-bearing:

* T3 runs a 50 Hz loop commanding a PHYSICAL G1, and T1 already sits above one core. A
  render loop in either is the mistake that has already cost this rig a session (a viser IK
  thread starving the CUDA launch loop, 46 -> 3543 ms). A read-only WS client cannot do that.
* **hssim has no viser.** UMI_collection's `.venv` has viser 1.0.30, the yam URDF with
  meshes, and `mc_frames` (numpy-only, so it imports either side). So the VIEWER runs there
  and the REFERENCE — which is holosoma's, and must come from the policy's own FK — is
  computed in hssim by `vive/mc_gate_reference.py` and handed over as JSON. That boundary is
  free because the reference is STATIC.

WHAT IS DRAWN
=============
Both bodies share one origin, because the wire is already PELVIS-RELATIVE and that is the
frame the gate compares in — so "overlap the two robots" is exactly "pass the gate".

* `/ref`  — the held reference, as a translucent ghost, plus its keypoints as GREEN spheres.
            The GATED points are drawn larger; the completion-inferred ones (5pt feet) are
            small and dim, because chasing those is the thing an operator must NOT do.
* `/live` — the completion of your trackers, opaque, plus its keypoints as ORANGE spheres,
            with a line to each reference twin. Line length IS the gap the gate prints.
* GUI     — per-point gap, the tolerance, and PASS/REFUSE per point, recomputed here from
            the same numbers. `mc_lag_s` is shown too: ~667 ms is structural, not a stall.

In `--mode dense` there are no keypoints, so the gap table is PER JOINT (radians) and only
the steerable joints are marked gated — mirroring the dense half of the gate.

THE BODY NEEDS `--publish-mc-qpos` ON THE BRIDGE (5pt only)
==========================================================
The 5pt wire carries the five keypoints, not the 29 joints, so without that flag this draws
the spheres and the reference ghost but no live body — and says so in the panel rather than
silently showing a T-pose. `--mode dense` already ships the joints.

RUN IT BY PATH, NOT `-m`
========================
`wbt_realtime_eval/__init__.py` imports `main`, which pulls in mjswan + holosoma — neither of
which exists in UMI's `.venv`. This module needs nothing from its own package (stdlib, numpy,
viser, and `mc_frames` by path), so `4_pose_match_viz.sh` invokes the FILE. Keep it that way:
adding a package-level import here would make the viewer unrunnable in the only env that has
a renderer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

_DEFAULT_UMI_DIR = os.path.expanduser("~/FLoRe/third_party/UMI_collection")
#: Sphere radii: the gated points must read as "aim at these", the derived ones as "ignore".
_R_GATED = 0.035
_R_DERIVED = 0.018
#: Redraw cap. The wire is 30 Hz; a viser scene does not need more than this and every
#: `update_cfg` is real work (measured elsewhere on this rig as the dominant render cost).
_DRAW_HZ = 15.0


def load_mc_frames(umi_dir: str):
    """Import `mc_frames` from the UMI_collection checkout — the ONE copy of the joint-name
    remap (`urdf_cfg_from_qpos`). Same path-import contract as `mc_teleop_source`: the module
    is numpy-only precisely so both envs can share it instead of carrying two that drift."""
    pkg = os.path.join(umi_dir, "motion_completion")
    if not os.path.isdir(pkg):
        raise FileNotFoundError(
            f"no motion_completion/ under {umi_dir} — set MC_UMI_DIR / --umi-dir to the "
            f"UMI_collection checkout"
        )
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    if "mc_frames" in sys.modules:
        return sys.modules["mc_frames"]
    spec = importlib.util.spec_from_file_location("mc_frames", os.path.join(pkg, "mc_frames.py"))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load mc_frames from {pkg}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mc_frames"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# the gate's arithmetic, as pure functions (no viser, no robot, no wire)
# ---------------------------------------------------------------------------

def keypoint_gaps(live_pos, ref_pos) -> np.ndarray:
    """Per-point L2 gap, over the points BOTH sides supply.

    Deliberately the same shape as `holosoma_wbt._vive_arm_pose_ok`: reshape to whole
    3-vectors and take the min count, so a short frame reads as fewer gaps rather than an
    index error."""
    lp = np.asarray(live_pos, dtype=np.float64).reshape(-1, 3)
    rp = np.asarray(ref_pos, dtype=np.float64).reshape(-1, 3)
    n = min(lp.shape[0], rp.shape[0])
    return np.linalg.norm(lp[:n] - rp[:n], axis=1)


def joint_gaps(live_q, live_names, ref_q, ref_names) -> tuple[np.ndarray, list[str]]:
    """Per-joint |gap| in REF name order, remapping the live vector BY NAME.

    Never index-aligned: MC's joint order and the policy's are both real orders and they
    are not the same. A joint the live frame does not carry reads as gap 0 (it is not
    evidence of a mismatch), which is the conservative direction for a display."""
    src = {n: float(v) for n, v in zip(live_names, np.asarray(live_q, dtype=np.float64).reshape(-1))}
    ref = np.asarray(ref_q, dtype=np.float64).reshape(-1)
    out = np.zeros(ref.shape[0], dtype=np.float64)
    for i, n in enumerate(ref_names[: ref.shape[0]]):
        if n in src:
            out[i] = abs(src[n] - ref[i])
    return out, list(ref_names[: ref.shape[0]])


def verdict(gaps, n_gated: int, tol: float) -> tuple[bool, list[int]]:
    """(would the gate PASS, indices of the gated entries that are over tol).

    `n_gated` entries are gated; the rest are reported but never refuse — see
    `holosoma_wbt._VR_5POINT_OPERATOR_PLACED` for why that split exists."""
    g = np.asarray(gaps, dtype=np.float64)
    k = int(max(0, min(n_gated, g.shape[0])))
    over = [i for i in range(k) if g[i] > tol]
    return (len(over) == 0), over


# ---------------------------------------------------------------------------
# the wire reader (its own thread; viser owns the main thread)
# ---------------------------------------------------------------------------

class WireReader:
    """Read-only :8770 client. Keeps only the newest frame; never blocks the renderer.

    Reconnects forever with a fixed backoff: the bridge is routinely restarted underneath
    this viewer, and a viewer that dies on the first hiccup is a viewer nobody leaves open.
    """

    def __init__(self, url: str):
        self._url = url
        self._lock = threading.Lock()
        self._frame: dict | None = None
        self._state = "connecting"
        self._n = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="wire", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> tuple[dict | None, str, int]:
        with self._lock:
            return (dict(self._frame) if self._frame else None), self._state, self._n

    def _run(self) -> None:
        import asyncio

        import websockets

        async def pump() -> None:
            while not self._stop.is_set():
                try:
                    async with websockets.connect(self._url, open_timeout=5) as ws:
                        with self._lock:
                            self._state = "connected"
                        while not self._stop.is_set():
                            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            try:
                                f = json.loads(msg)
                            except (TypeError, ValueError):
                                continue
                            with self._lock:
                                self._frame = f
                                self._n += 1
                except Exception as exc:  # noqa: BLE001 — any failure is "retry"
                    with self._lock:
                        self._state = f"reconnecting ({type(exc).__name__})"
                    await asyncio.sleep(1.0)

        asyncio.run(pump())


# ---------------------------------------------------------------------------
# the scene
# ---------------------------------------------------------------------------

def _yam_urdf(umi_dir: str) -> Path:
    """UMI's own yam URDF — the one with meshes, and the one the MC overlay draws.

    Safe against holosoma's copy: all 29 canonical joint origins are bit-identical between
    the two (pinned by `mc_frames`' own tests), so the completed 29 DoF FK to the same place
    in either. This is only the RENDER model; every number in the gap table comes from the
    policy-side FK via `mc_gate_reference`.
    """
    p = Path(umi_dir) / "robot_assets" / "g1_yam" / "g1_29dof_yam.urdf"
    if not p.is_file():
        raise FileNotFoundError(f"no yam URDF at {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8770")
    ap.add_argument("--gate-ref", required=True, help="JSON from vive/mc_gate_reference.py")
    ap.add_argument("--umi-dir", default=os.environ.get("MC_UMI_DIR") or _DEFAULT_UMI_DIR)
    ap.add_argument("--port", type=int, default=8085, help="viser port (default 8085)")
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (default 127.0.0.1). viser's OWN default is 0.0.0.0; this pins it "
        "to loopback because the page carries live telemetry from a physical robot and there "
        "is no auth in front of it. A non-loopback bind is allowed but warns.",
    )
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"[gate-viz] ⚠️  binding NON-LOOPBACK ({args.host}:{args.port}) — this page shows "
            "live robot telemetry and has no authentication in front of it.",
            file=sys.stderr,
            flush=True,
        )

    import viser
    from viser.extras import ViserUrdf

    F = load_mc_frames(args.umi_dir)
    ref = json.loads(Path(args.gate_ref).read_text())
    mode = str(ref.get("mode", "5pt"))
    ref_q = np.asarray(ref["qpos"], dtype=np.float64)
    ref_names = list(ref["dof_names"])
    ref_kp = np.asarray(ref.get("keypoints") or [], dtype=np.float64).reshape(-1, 3)
    labels = list(ref.get("point_labels") or [])
    n_gated_pts = int(ref.get("n_gated_points", 0))
    tol_m = float(ref.get("arm_tol_m", 0.07))
    tol_rad = float(ref.get("arm_tol_rad", 0.20))
    gated_joints = set(ref.get("gated_joints") or [])

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    urdf_path = _yam_urdf(args.umi_dir)

    ghost = ViserUrdf(server, urdf_or_path=urdf_path, root_node_name="/ref")
    live = ViserUrdf(server, urdf_or_path=urdf_path, root_node_name="/live")
    urdf_joints = list(ghost.get_actuated_joint_names())
    ghost.update_cfg(F.urdf_cfg_from_qpos(ref_q, ref_names, urdf_joints))
    live.update_cfg(F.urdf_cfg_from_qpos(ref_q, ref_names, urdf_joints))  # until a frame lands

    # Reference keypoints: static, so add once.
    for i in range(ref_kp.shape[0]):
        gated = i < n_gated_pts
        server.scene.add_icosphere(
            f"/ref/kp{i}",
            radius=_R_GATED if gated else _R_DERIVED,
            color=(60, 200, 90) if gated else (90, 120, 95),
            position=tuple(ref_kp[i]),
            opacity=0.85 if gated else 0.35,
        )

    gui = server.gui
    with gui.add_folder("what am I matching?"):
        gui.add_markdown(
            f"**mode** `{mode}` · **tol** "
            + (f"`{tol_m:.3f} m`" if mode != "dense" else f"`{tol_rad:.3f} rad`")
            + "\n\nGreen = the pose the gate wants. Orange = what your trackers are "
            "sending. **Overlap them and `start` will arm.**\n\nSmall dim spheres are "
            "completion-inferred (the feet): they are NOT gated, so do not chase them."
            # What the orange spheres ARE is a real question and the answer is not obvious:
            # they are neither a target you set nor an IK solution, they are FK off the
            # completed body. Saying so in the panel is cheaper than the operator guessing.
            + "\n\nThe orange spheres are **FK off Motion_Completion's 29-DoF output** "
            "(`select_keypoints`), not an IK solution and not your raw pucks — so they lag "
            "you by `mc_lag_s` and they move only as far as the completion moves. Read out "
            "in the pelvis frame **frozen at calibration**, which is why standing in the "
            "calibration pose should already put them on the green ones."
        )
        for i, lbl in enumerate(labels):
            if i < n_gated_pts:
                gui.add_markdown(
                    f"- **{lbl}** → x `{ref_kp[i][0]:+.3f}` y `{ref_kp[i][1]:+.3f}` "
                    f"z `{ref_kp[i][2]:+.3f}` (pelvis frame)"
                )
    status = gui.add_markdown("waiting for the bridge…")
    table = gui.add_markdown("")
    # `ViserUrdf` exposes only show_visual / show_collision — there is no per-tree opacity in
    # viser 1.0.30, so the ghost is a TOGGLE rather than a fake slider. The keypoint spheres
    # carry the opacity (add_icosphere does support it), which is where it matters anyway.
    show_ghost = gui.add_checkbox("show reference ghost", True)

    reader = WireReader(args.ws_url)
    reader.start()

    live_kp_handles: dict[int, object] = {}
    live_line_handles: dict[int, object] = {}
    last_draw = 0.0
    warned_no_qpos = False

    try:
        while True:
            time.sleep(max(0.0, (1.0 / _DRAW_HZ) - (time.time() - last_draw)))
            last_draw = time.time()
            frame, state, n = reader.latest()
            ghost.show_visual = bool(show_ghost.value)
            if frame is None:
                status.content = (
                    f"**{state}** · {n} frames · nothing to draw yet. The bridge publishes "
                    "NOTHING until it is CALIBRATED and every tracked puck is fresh."
                )
                continue

            lag = frame.get("mc_lag_s")
            head = (
                f"**{state}** · frame `{frame.get('mc_frame', '?')}` · "
                f"lag `{lag:.2f} s`" if isinstance(lag, (int, float)) else f"**{state}**"
            )

            if mode == "dense":
                lq = frame.get("joint_pos")
                lnames = frame.get("joint_names")
                if not lq or not lnames:
                    status.content = head + " · this frame carries no joints (wrong --mode?)"
                    continue
                gaps, names = joint_gaps(lq, lnames, ref_q, ref_names)
                mask = [i for i, nm in enumerate(names) if nm in gated_joints]
                over = [i for i in mask if gaps[i] > tol_rad]
                live.update_cfg(F.urdf_cfg_from_qpos(
                    np.asarray(lq, dtype=np.float64), list(lnames), urdf_joints))
                worst = sorted(mask, key=lambda i: -gaps[i])[:6]
                rows = "\n".join(
                    f"- {'❌' if gaps[i] > tol_rad else '✅'} `{names[i]}` "
                    f"**{gaps[i]:.3f}** rad"
                    for i in worst
                )
                status.content = head + (" · **WOULD ARM**" if not over else
                                         f" · **WOULD REFUSE** ({len(over)} joints)")
                table.content = "steerable joints, worst first:\n" + rows
                continue

            pos = frame.get("pos")
            if not pos:
                status.content = head + " · this frame carries no keypoints (wrong --mode?)"
                continue
            lp = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
            gaps = keypoint_gaps(lp, ref_kp)
            ok, over = verdict(gaps, n_gated_pts, tol_m)

            qpos = frame.get("mc_qpos")
            qnames = frame.get("mc_joint_names")
            if qpos and qnames:
                live.update_cfg(F.urdf_cfg_from_qpos(
                    np.asarray(qpos, dtype=np.float64), list(qnames), urdf_joints))
            elif not warned_no_qpos:
                warned_no_qpos = True
                print(
                    "[gate-viz] the wire has no `mc_qpos`, so the LIVE BODY is not drawn "
                    "(the orange keypoints still are). Restart the bridge with "
                    "--publish-mc-qpos to see the whole completion.",
                    flush=True,
                )

            for i in range(lp.shape[0]):
                gated = i < n_gated_pts
                col = (245, 150, 40) if gated else (140, 120, 100)
                if i in live_kp_handles:
                    live_kp_handles[i].position = tuple(lp[i])
                else:
                    live_kp_handles[i] = server.scene.add_icosphere(
                        f"/live/kp{i}",
                        radius=_R_GATED if gated else _R_DERIVED,
                        color=col,
                        position=tuple(lp[i]),
                        opacity=0.9 if gated else 0.35,
                    )
                if i < ref_kp.shape[0] and gated:
                    # RE-ADD, do not mutate. `LineSegmentsHandle` exposes only
                    # name/position/scale/visible/wxyz/remove — there is NO settable
                    # `points`, so `handle.points = seg` would silently set an unused Python
                    # attribute and the gap line would stay frozen at its FIRST position
                    # while the sphere moved. No exception, just a lying line. Re-adding
                    # under the same name replaces the node, which is the supported way.
                    live_line_handles[i] = server.scene.add_line_segments(
                        f"/live/gap{i}",
                        points=np.stack([lp[i], ref_kp[i]])[None, ...],
                        colors=(255, 60, 60),
                        line_width=3.0,
                    )

            rows = "\n".join(
                f"- {'✅' if (i >= n_gated_pts or gaps[i] <= tol_m) else '❌'} "
                f"`{labels[i] if i < len(labels) else f'pt{i}'}` **{gaps[i]:.3f}** m"
                + ("" if i < n_gated_pts else "  _(not gated)_")
                for i in range(gaps.shape[0])
            )
            status.content = head + (" · **WOULD ARM — press start**" if ok else
                                     f" · **WOULD REFUSE** ({len(over)} of "
                                     f"{n_gated_pts} gated points)")
            table.content = rows
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
