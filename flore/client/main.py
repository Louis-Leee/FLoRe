"""FLoRe client main loop — bridges S1 (gRPC) to S0 (WS :8770).

Control flow at ~30Hz:
1. Build obs: image (camera JPEG) + state (last_target from ActionTo3pt)
2. gRPC Predict → receive one drip-fed action (32D, delta-EE)
3. ActionTo3pt.update(action) → pos[9], orn[12] (absolute pelvis-frame)
4. S0Driver.send(pos, orn) → broadcast on WS :8770

Cold-start gate: S0 stays in ready pose until the first action arrives from S1
(the gRPC Predict blocks until the server's RTC generates its first chunk —
can be tens of seconds on model load + warmup).

obs_mode=last_target: the state fed to S1 is the LAST COMMANDED target (from
ActionTo3pt.current_state), NOT a readback from S0. This is proven in FAR-pi's
WBT-VLA wbt_vla_online_node and gives train/deploy consistency (training state
= tracker pose at frame t ≈ the "commanded" pose in a handheld demo).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

import cv2
import numpy as np

from flore.client.action_to_3pt import ActionTo3pt
from flore.client.s0_driver import S0Driver
from flore.client.s1_client import S1Client

logger = logging.getLogger(__name__)

# Default home poses from holosoma's 3pt-hybrid preset baked anchor (pelvis frame).
# These are the FK-derived ready-pose targets the controller starts at.
# Source: wbt_training/config_values/robot_config.py:72-73, confirmed in the plan.
_DEFAULT_HOME_POS = np.array(
    [
        [0.18, -0.025, 0.0],  # left wrist + offset
        [0.18, 0.025, 0.0],  # right wrist + offset
        [0.0, 0.0, 0.35],  # head/torso + z offset
    ],
    dtype=np.float64,
)

# Identity rotations as starting orientations (the policy's first action deltas
# are relative to these; at deploy the ready-pose quaternions from S0 should be
# used instead — TODO: read them from the preset's baked anchor NPZ).
_DEFAULT_HOME_ROTMAT = np.stack([np.eye(3)] * 3).astype(np.float64)


def is_safe_to_send(*arrays: np.ndarray) -> bool:
    """True only if every array is entirely finite (no NaN/Inf).

    The robot-facing gate: a non-finite action/target must NEVER reach the wire
    (holosoma validates only array length; json.dumps serializes NaN). On False
    the caller skips send() and the freshness gate holds the robot.
    """
    return all(np.all(np.isfinite(a)) for a in arrays)


def _resize_center_crop_224(img: np.ndarray, out_w: int = 224, out_h: int = 224) -> np.ndarray:
    """Aspect-preserving resize + CENTER CROP to 224×224 (cover fit).

    Train/deploy parity: the training converter crops camera frames with EXACT INTEGER
    geometry — ``scale=max(out/in_w, out/in_h)``, ``resize=ceil(in*scale)`` (NOT round),
    clamped ``>= out``, then a floor-div center crop, ALWAYS with ``cv2.INTER_AREA`` — see
    ``UMI_collection`` ``umi_convert_core.compute_crop_resize`` + ``convert_to_umi.image_resize``.
    A naive ``cv2.resize(frame, (224, 224))`` STRETCHES (distorts aspect) and uses
    INTER_LINEAR, feeding the policy an out-of-distribution crop. Mirror the converter's math
    exactly here. Channel-order agnostic (pure geometry), so it applies to a BGR frame as-is.
    """
    in_h, in_w = img.shape[:2]
    scale = max(out_w / in_w, out_h / in_h)
    resize_w = max(int(np.ceil(in_w * scale)), out_w)
    resize_h = max(int(np.ceil(in_h * scale)), out_h)
    crop_x = (resize_w - out_w) // 2
    crop_y = (resize_h - out_h) // 2
    resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    return resized[crop_y : crop_y + out_h, crop_x : crop_x + out_w]


def _blank_jpeg() -> bytes:
    """A VALID 224x224 black JPEG for camera-less runs. A hand-crafted byte stub
    would fail the server's cv2.imdecode and abort the RPC — encode a real one."""
    black = np.zeros((224, 224, 3), dtype=np.uint8)
    _, jpeg = cv2.imencode(".jpg", black, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg.tobytes()


def _capture_camera(cap: cv2.VideoCapture | None) -> bytes | None:
    """Capture one JPEG frame from the camera (or None if unavailable).

    Uses the SAME aspect-preserving resize + center-crop as the training converter
    (:func:`_resize_center_crop_224`), NOT a naive stretch — otherwise the policy sees a
    distorted crop it never trained on.
    """
    if cap is None or not cap.isOpened():
        return None
    ret, frame = cap.read()
    if not ret:
        return None
    frame_resized = _resize_center_crop_224(frame)
    _, jpeg = cv2.imencode(".jpg", frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg.tobytes()


def run_client(args: argparse.Namespace) -> None:
    """Main control loop."""
    if args.dry_run:
        raise SystemExit("--dry-run is disabled until I01 provides a non-actuating implementation")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    s1 = S1Client(host=args.s1_host, port=args.s1_port, timeout=args.timeout)
    s0 = S0Driver(host=args.s0_ws_host, port=args.s0_ws_port)
    converter = ActionTo3pt(_DEFAULT_HOME_POS, _DEFAULT_HOME_ROTMAT)

    stop = False

    def _sigterm(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sigterm)
    signal.signal(signal.SIGTERM, _sigterm)

    # Start S0 driver (WS server) BEFORE connecting to S1 — holosoma's
    # ViveLiveSource auto-reconnects, so the server should be up when S0 starts.
    s0.start()

    # Connect to S1 server.
    logger.info("Connecting to S1 server at %s:%d ...", args.s1_host, args.s1_port)
    try:
        info = s1.connect()
    except Exception as e:
        logger.error("Failed to connect to S1: %s", e)
        s0.stop()
        sys.exit(1)
    logger.info("S1 connected: %s", info)

    # Camera (optional; --no-camera for dry-run without hardware).
    cap = None
    if not args.no_camera:
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            logger.warning("Camera %s not available — using blank frames", args.camera_id)
            cap = None

    # Reset S1 at episode start.
    s1.reset()
    logger.info("Episode reset. Waiting for S0 to connect on ws://%s:%d ...", args.s0_ws_host, args.s0_ws_port)
    while s0.num_clients == 0 and not stop:
        time.sleep(0.1)
    if stop:
        s0.stop()
        s1.close()
        return
    logger.info(
        "S0 connected. Starting control loop at %.1f Hz (cold-start: first Predict may block ~60s for model warmup).",
        args.control_hz,
    )

    period = 1.0 / args.control_hz
    step = 0
    t_start = time.perf_counter()

    while not stop:
        t0 = time.perf_counter()

        # 1. Obs: image + last_target state.
        image_jpeg = _capture_camera(cap)
        if image_jpeg is None:
            image_jpeg = _blank_jpeg()  # valid black frame (server decodes it)
        state = converter.current_state  # 32D, obs_mode=last_target

        # 2. gRPC Predict (blocks until action ready; cold-start can be long).
        action = s1.predict(state, image_jpeg, prompt=args.prompt)

        # SAFETY GATE: never forward non-finite actions to the robot. A bad
        # checkpoint / fp16 overflow / near-zero rot6d (degenerate Gram-Schmidt)
        # can produce NaN/Inf; holosoma's ViveLiveSource validates only array
        # LENGTH, and json.dumps serializes NaN unchecked, so garbage would
        # reach the G1. Skip this step's send() instead — the freshness gate
        # (0.5s) then lets S0's 0.25s stale gate hold the robot safe.
        if not is_safe_to_send(action):
            logger.error("step %d: non-finite action from S1 — skipping send (robot holds)", step)
            step += 1
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
            continue

        # 3. Split the absolute action into S0's three-point wire layout.
        pos, orn = converter.update(action)

        # Defense-in-depth: the integrated target (quat conversion) must also be
        # finite before it hits the wire.
        if not is_safe_to_send(pos, orn):
            logger.error("step %d: non-finite target after integration — skipping send", step)
            step += 1
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
            continue

        # 4. Broadcast to S0.
        s0.send(pos, orn)

        step += 1
        if step % 100 == 0:
            elapsed = time.perf_counter() - t_start
            logger.info("step %d, %.1f Hz avg, pos_L=[%.3f,%.3f,%.3f]", step, step / elapsed, pos[0], pos[1], pos[2])

        # Rate limit.
        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)

    logger.info("Shutting down (step=%d)", step)
    s0.stop()
    s1.close()
    if cap is not None:
        cap.release()


def main():
    p = argparse.ArgumentParser(description="FLoRe client: bridges S1 (gRPC) to S0 (WS :8770)")
    p.add_argument("--s1-host", default="127.0.0.1", help="S1 gRPC server host")
    p.add_argument("--s1-port", type=int, default=50051, help="S1 gRPC server port")
    p.add_argument("--s0-ws-host", default="127.0.0.1", help="WS server bind host for S0")
    p.add_argument("--s0-ws-port", type=int, default=8770, help="WS server port for S0")
    p.add_argument("--control-hz", type=float, default=30.0, help="control loop rate")
    p.add_argument("--camera-id", default="0", help="camera device id or path")
    p.add_argument("--no-camera", action="store_true", help="skip camera (blank frames)")
    p.add_argument("--prompt", default="", help="language instruction for S1")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="disabled until I01 provides a non-actuating implementation",
    )
    p.add_argument("--timeout", type=float, default=600.0, help="gRPC timeout (s)")
    args = p.parse_args()
    run_client(args)


if __name__ == "__main__":
    main()
