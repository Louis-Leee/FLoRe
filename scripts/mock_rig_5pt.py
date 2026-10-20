#!/usr/bin/env python3
"""Mock rig for scripts/hl_serving_5pt.py — the bridges a rig would run, minus the hardware.

    uv run scripts/mock_rig_5pt.py            # then: TASK=<task> ./scripts/run_hl_serving_5pt.sh --no-command-gripper

Serves whichever of the six planner ports is FREE (F/T :8771, cameras :8773/:8777, jaws :8784/:8785,
RecordTap :8779) and skips any a REAL bridge already holds, so it can fill in only the low level on a
box whose cameras/Botas/jaws are up (that is how the 2026-09-12 rig check ran). It also connects to
the planner's own :8770 as a mock 5pt low level: it reads every wire frame, echoes the five positions
back as RecordTap ``vr_cmd_pos`` (feet exempt from the slew mask, as SLEW_FEET=0 reports), arms itself
4 s after start, and reports ``ref_joints_live`` when a frame carries the walk skill's legs. The
RecordTap joints are the READY POSE (ready_pose.resolve_deploy_anchor), so the planner's measured
cross-check passes. Nothing here touches a robot, a Bota, a jaw or a camera.

⚠️ Stop it BEFORE starting the real low level: two RecordTap producers on :8779 is the same class of
failure as two writers on :8770. It exits with the planner (the :8770 client just retries otherwise).
"""
import asyncio
import base64
import json
import os
import socket
import sys
import time

import cv2
import numpy as np
import websockets

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flore.s1_policy.inference import ready_pose  # noqa: E402
from flore.s1_policy.serving.hl_5pt import WIRE_ROLES  # noqa: E402

anchor, _why = ready_pose.resolve_deploy_anchor(WIRE_ROLES)
if anchor is None or anchor.dof is None:
    raise SystemExit(f"cannot resolve the G1 ready pose: {_why}")
DOF = anchor.dof.tolist()
latest_pos = [list(map(float, anchor.pos[r] - np.array([0.0, 0.0, anchor.pelvis_z]))) for r in WIRE_ROLES]
state = {"wire_frames": 0, "armed_at": time.time() + 4.0, "ref_joints_frames": 0, "last_ref_joints": False}


async def broadcaster(port, hz, make):
    clients = set()

    async def handler(ws):
        clients.add(ws)
        try:
            async for _ in ws:
                pass
        finally:
            clients.discard(ws)

    async with websockets.serve(handler, "127.0.0.1", port, max_size=None):
        seq = 0
        while True:
            seq += 1
            msg = json.dumps(make(seq))
            for c in list(clients):
                try:
                    await c.send(msg)
                except Exception:  # noqa: BLE001 — a client that went away is not our problem
                    clients.discard(c)
            await asyncio.sleep(1.0 / hz)


def cam(stream):
    img = np.full((480, 640, 3), 90, np.uint8)
    cv2.putText(img, stream, (40, 240), 0, 2, (255, 255, 255), 3)
    _ok, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf.tobytes()).decode()
    return lambda seq: {"type": "frame", "seq": seq, "t": time.time(), "w": 640, "h": 480,
                        "stream": "color", "jpeg_b64": b64}


def ft(seq):
    z3 = [0.0, 0.0, 0.0]
    z6 = [0.0] * 6
    return {"type": "ft", "seq": seq, "t": time.time(), "left": z3, "right": z3, "left_pipe": z6, "right_pipe": z6}


def grip(seq):
    return {"seq": seq, "t": time.time(), "open": 4.9, "closed": 0.0, "cmd": 1.0, "actual": 4.9}


def record(seq):
    armed = time.time() > state["armed_at"]
    return {"type": "record", "seq": seq, "t": time.time(), "running": True, "servo_on": False,
            "vive_armed": armed, "dof_pos": DOF, "root_quat_wxyz": [1, 0, 0, 0],
            "f_meas": [[0, 0, 0], [0, 0, 0]], "f_meas_ungated": [[0, 0, 0], [0, 0, 0]],
            "vr_cmd_pos": latest_pos, "vr_cmd_slewed": [True, True, True, False, False],
            "ref_joints_live": bool(armed and state["last_ref_joints"])}


async def low_level():
    while True:
        try:
            async with websockets.connect("ws://127.0.0.1:8770", max_size=None) as ws:
                print("[mock-ll] connected to planner :8770", flush=True)
                async for m in ws:
                    d = json.loads(m)
                    if d.get("mode") != "5pt":
                        continue
                    p = np.asarray(d["pos"], float).reshape(5, 3)
                    for i in range(5):
                        latest_pos[i] = p[i].tolist()
                    state["wire_frames"] += 1
                    state["last_ref_joints"] = "ref_joints" in d
                    if "ref_joints" in d:
                        state["ref_joints_frames"] += 1
                    if state["wire_frames"] % 300 == 0:
                        print(f"[mock-ll] {state['wire_frames']} wire frames, ref_joints frames "
                              f"{state['ref_joints_frames']}, L hand {np.round(p[0], 4).tolist()}", flush=True)
        except Exception:  # noqa: BLE001 — the planner is not up yet, or went away; retry
            await asyncio.sleep(1.0)


def _free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


async def main():
    plan = [(8771, 100, ft), (8773, 30, cam("camera L")), (8777, 30, cam("camera R")),
            (8784, 50, grip), (8785, 50, grip), (8779, 50, record)]
    tasks = [low_level()]
    for port, hz, mk in plan:
        if _free(port):
            tasks.append(broadcaster(port, hz, mk))
            print(f"[mock] serving :{port}", flush=True)
        else:
            print(f"[mock] :{port} already served by a REAL bridge — not mocked", flush=True)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
