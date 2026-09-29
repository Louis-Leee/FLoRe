"""Tests for the FUJI client layer: action_to_3pt conversion + s0_driver.

These tests verify the load-bearing math of the S1→S0 translation and the
WebSocket server broadcast. No GPU, no network (except localhost WS).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from fuji.client.action_to_3pt import ActionTo3pt, ACTION_DIM
from fuji.utils.rot6d import rotmat_to_rot6d


# ---------------------------------------------------------------------------
# action_to_3pt
# ---------------------------------------------------------------------------


def test_dry_run_exits_before_opening_connections():
    from argparse import Namespace

    from fuji.client.main import run_client

    with pytest.raises(SystemExit, match="dry-run is disabled"):
        run_client(Namespace(dry_run=True))


def _identity_homes():
    """Home at identity: pos = [[0.18,-0.025,0],[0.18,0.025,0],[0,0,0.35]], rot = I."""
    pos = np.array([[0.18, -0.025, 0.0], [0.18, 0.025, 0.0], [0.0, 0.0, 0.35]])
    rot = np.stack([np.eye(3)] * 3)
    return pos, rot


def _absolute_action(pos_lrh, rot_lrh):
    """Build a 32D ABSOLUTE action from per-keypoint (pos[3], rotmat) inputs."""
    a = np.zeros(ACTION_DIM, dtype=np.float64)
    for (pos_sl, rot_sl), p, R in zip(
        [(slice(0, 3), slice(3, 9)), (slice(10, 13), slice(13, 19)), (slice(20, 23), slice(23, 29))],
        pos_lrh, rot_lrh,
    ):
        a[pos_sl] = p
        a[rot_sl] = rotmat_to_rot6d(R)
    return a


def test_absolute_action_splits_to_wire():
    """The client is a STATELESS splitter: an ABSOLUTE action is reformatted to
    the wire verbatim (server already un-relativized). No integration/drift.
    """
    home_pos, home_rot = _identity_homes()
    converter = ActionTo3pt(home_pos, home_rot)

    # Absolute target: move left hand, keep right/head at home.
    a = _absolute_action(
        [np.array([0.30, 0.10, 0.20]), home_pos[1], home_pos[2]],
        [np.eye(3), np.eye(3), np.eye(3)],
    )
    pos, orn = converter.update(a.astype(np.float32))

    np.testing.assert_allclose(pos[:3], [0.30, 0.10, 0.20], atol=1e-6)
    np.testing.assert_allclose(pos[3:6], home_pos[1], atol=1e-6)
    np.testing.assert_allclose(pos[6:9], home_pos[2], atol=1e-6)
    for i in range(3):
        q = orn[i * 4 : (i + 1) * 4]
        assert abs(np.dot(q, [0, 0, 0, 1])) > 0.9999  # identity xyzw


def test_no_compounding_across_steps():
    """CRITICAL (the blocker the e2e review caught): repeating the SAME absolute
    action must NOT accumulate. The client splits, it does not integrate.
    """
    from scipy.spatial.transform import Rotation

    home_pos, home_rot = _identity_homes()
    converter = ActionTo3pt(home_pos, home_rot)

    R = Rotation.from_euler("z", 40, degrees=True).as_matrix()
    a = _absolute_action(
        [np.array([0.25, 0.0, 0.1]), home_pos[1], home_pos[2]],
        [R, np.eye(3), np.eye(3)],
    )

    pos1, orn1 = converter.update(a.astype(np.float32))
    pos2, orn2 = converter.update(a.astype(np.float32))
    pos3, orn3 = converter.update(a.astype(np.float32))

    # Identical absolute action → identical wire output every time (no drift).
    np.testing.assert_allclose(pos1, pos2, atol=1e-7)
    np.testing.assert_allclose(pos2, pos3, atol=1e-7)
    np.testing.assert_allclose(orn1, orn2, atol=1e-7)
    q_expected = Rotation.from_matrix(R).as_quat()
    assert abs(np.dot(orn1[0:4], q_expected)) > 0.99999


def test_current_state_matches_wire():
    """current_state (32D obs) encodes the same pose as the wire output."""
    home_pos, home_rot = _identity_homes()
    converter = ActionTo3pt(home_pos, home_rot)

    a = _absolute_action(
        [np.array([0.05, -0.02, 0.35]), home_pos[1], home_pos[2]],
        [np.eye(3), np.eye(3), np.eye(3)],
    )
    pos, orn = converter.update(a.astype(np.float32))
    state = converter.current_state

    # State pos matches wire pos (both absolute).
    np.testing.assert_allclose(state[0:3], pos[0:3], atol=1e-6)
    np.testing.assert_allclose(state[10:13], pos[3:6], atol=1e-6)
    np.testing.assert_allclose(state[20:23], pos[6:9], atol=1e-6)


def test_from_flat_home():
    """from_flat_home constructs correctly from wire-format arrays."""
    home_pos = np.array([0.18, -0.025, 0, 0.18, 0.025, 0, 0, 0, 0.35], dtype=np.float64)
    home_orn = np.tile([0, 0, 0, 1], 3).astype(np.float64)  # 3x identity xyzw
    converter = ActionTo3pt.from_flat_home(home_pos, home_orn)
    assert converter._pos.shape == (3, 3)
    assert converter._rotmat.shape == (3, 3, 3)
    np.testing.assert_allclose(converter._pos[0], [0.18, -0.025, 0], atol=1e-10)


# ---------------------------------------------------------------------------
# s0_driver WS server
# ---------------------------------------------------------------------------


def test_s0_driver_broadcasts():
    """S0Driver starts a WS server; a client connects and receives 3pt JSON."""
    import websockets.sync.client

    from fuji.client.s0_driver import S0Driver

    driver = S0Driver(host="127.0.0.1", port=18770)  # non-default port for test isolation
    driver.FRESHNESS_TIMEOUT_S = 10.0  # keep broadcasting throughout the test (no stale-stop race)
    driver.start()
    time.sleep(0.2)

    pos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
    orn = np.arange(12, dtype=np.float32) * 0.1

    with websockets.sync.client.connect("ws://127.0.0.1:18770") as ws:
        # Send AFTER the client connects so the payload is guaranteed fresh and
        # a broadcast lands on this connection (avoids a connect/send race under
        # concurrent CPU load). Generous recv timeout swamps the 5ms interval.
        time.sleep(0.1)
        driver.send(pos, orn)
        msg_str = ws.recv(timeout=3.0)
        msg = json.loads(msg_str)
        assert msg["mode"] == "3pt"
        assert len(msg["pos"]) == 9
        assert len(msg["orn"]) == 12
        np.testing.assert_allclose(msg["pos"], pos.tolist(), atol=1e-5)
        np.testing.assert_allclose(msg["orn"], orn.tolist(), atol=1e-5)

    driver.stop()


def test_s0_driver_freshness_timeout_stops_rebroadcast():
    """Safety: after FRESHNESS_TIMEOUT_S with no send(), the driver STOPS
    re-broadcasting so holosoma's stale gate can fire (a hung control loop must
    not freeze S0 on the last target while looking alive)."""
    import websockets.sync.client

    from fuji.client.s0_driver import S0Driver

    driver = S0Driver(host="127.0.0.1", port=18771)
    driver.FRESHNESS_TIMEOUT_S = 0.3  # shortened for the test (default 0.5)
    driver.start()
    time.sleep(0.2)

    pos = np.ones(9, dtype=np.float32)
    orn = np.zeros(12, dtype=np.float32)

    with websockets.sync.client.connect("ws://127.0.0.1:18771") as ws:
        time.sleep(0.1)
        driver.send(pos, orn)
        # Fresh: at least one frame arrives promptly.
        msg = json.loads(ws.recv(timeout=2.0))
        assert msg["mode"] == "3pt"

        # Go silent well past the freshness window; broadcasts must STOP.
        # (Generous margins so the test is robust under concurrent CPU load —
        # the broadcast interval is 5ms, so a recv() timeout >> that is a
        # definitive "broadcaster is quiet" signal.)
        time.sleep(1.0)
        # Drain any in-flight frames. A recv timeout of 0.3s is 60x the 5ms
        # broadcast interval, so a timeout genuinely means the stream stopped
        # (not just a scheduling gap mid-stream).
        try:
            while True:
                ws.recv(timeout=0.3)
        except TimeoutError:
            pass
        # After draining, no NEW frame should arrive (broadcaster is quiet).
        with pytest.raises(TimeoutError):
            ws.recv(timeout=0.5)

    driver.stop()


# ---------------------------------------------------------------------------
# S1Client <-> real server integration (catches proto-import / RPC-wiring bugs)
# ---------------------------------------------------------------------------

def test_is_safe_to_send_rejects_non_finite():
    """Safety gate: NaN/Inf in any array must be rejected (robot never gets it)."""
    from fuji.client.main import is_safe_to_send

    good = np.zeros(32, dtype=np.float32)
    assert is_safe_to_send(good)
    assert is_safe_to_send(np.ones(9), np.zeros(12))

    nan = np.zeros(32, dtype=np.float32)
    nan[5] = np.nan
    assert not is_safe_to_send(nan)

    inf = np.zeros(9, dtype=np.float32)
    inf[0] = np.inf
    assert not is_safe_to_send(np.zeros(12), inf)  # any non-finite array fails


def test_s1_client_connect_and_predict_against_real_server():
    """The real S1Client must connect() + predict() against the real gRPC
    server (with a fake chunk policy). This exercises the proto-import path and
    the PredictRequest field wiring end-to-end — the parts the unit tests for
    each side individually cannot cover.
    """
    import numpy as np

    from fuji.client.s1_client import S1Client
    from fuji.s1_policy.serving.server import build_server, encode_image_jpeg, make_servicer

    # Reuse the deterministic fake from the server test suite.
    from tests.test_server_grpc import ACTION_DIM as SRV_AD
    from tests.test_server_grpc import ACTION_HORIZON as SRV_AH
    from tests.test_server_grpc import FakeChunkPolicy

    fake = FakeChunkPolicy()
    servicer = make_servicer(
        fake, rtc=False, action_dim=SRV_AD, action_horizon=SRV_AH, checkpoint="fake-ckpt"
    )
    server, port = build_server(servicer, host="127.0.0.1", port=0)
    try:
        client = S1Client(host="127.0.0.1", port=port, timeout=10.0)
        info = client.connect()
        assert info["action_dim"] == SRV_AD
        assert info["action_horizon"] == SRV_AH
        assert info["checkpoint"] == "fake-ckpt"

        state = np.zeros(SRV_AD, dtype=np.float32)
        rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        jpeg = encode_image_jpeg(rgb)
        action = client.predict(state, jpeg, prompt="test")
        assert action.shape == (SRV_AD,)

        # PredictChunk round-trip.
        chunk = client.predict_chunk(state, jpeg, prompt="test")
        assert chunk.shape == (SRV_AH, SRV_AD)

        client.reset()
        assert fake.reset_count >= 1
        client.close()
    finally:
        server.stop(grace=0)
