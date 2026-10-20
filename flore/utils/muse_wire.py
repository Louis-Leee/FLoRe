"""Line protocol between the FLoRe planner and the MuSe-on-UVA model backend.

The MuSe force policy lives in the ``unified_video_action`` submodule and only runs in
ITS env (``uvaforce``: torch 2.7.1+cu128 + hydra + the MAR/VAE pretrained trees), while
viser + mink + the HL-v2 serving stack only run in FLoRe's ``.venv`` (torch 2.13+cu130).
The two are mutually exclusive, so ``--algo muse`` is a TWO-PROCESS shape: the planner
spawns the backend and speaks to it over a pipe. Same split as the older
``scripts/muse_force_serving_server.py`` — but there the backend held the whole planner,
whereas here it is a pure stateless model server and every bit of rig logic (wire mapper,
force gate, recorder, reach compensation, ghost) stays in ``scripts/hl_serving.py``.

THIS module is the one piece both sides must agree on byte-for-byte, so it is deliberately
**numpy-only**: no torch, no zarr, no ``flore.s1_policy`` imports. That is what lets the
uvaforce interpreter import it straight off FLoRe's source tree instead of carrying a second,
drifting copy of the codec (the failure mode that costs a day: a field renamed on one side
decodes as "absent" on the other, and an absent optional field is indistinguishable from a
zero one).

It lives under ``flore/utils/`` rather than next to the serving code for exactly that reason:
``flore.s1_policy.serving.__init__`` pulls in the gRPC server (grpc + torch + the data stack),
which the uvaforce env cannot satisfy, while ``flore/utils/__init__.py`` is empty. The MuSe
backend already imports ``flore.utils.rot6d`` the same way, so this follows a path the other
env is known to resolve.

Framing
-------
One message per line, ``MUSEHL<tag> <base64(npz)>``. Base64 keeps a payload with images in
it on a single line, so the reader is a plain ``readline()`` and never has to frame binary.
The npz always carries ``__meta__`` (a 0-d unicode array holding a JSON object) plus zero or
more named arrays.

``stdout`` is a PURE protocol channel on the backend side: hydra, the VAE and the MAR model
all print, so the backend redirects ``sys.stdout`` to stderr at import time and writes
protocol lines to the stashed real fd 1. Anything the frontend reads that does not start
with :data:`LINE_PREFIX` is backend chatter and is logged, not parsed.

Tags
----
``ready``  backend -> frontend, once: the resolved checkpoint's grid + force contract.
``chunk``  backend -> frontend, one per request: the predicted chunk.
``error``  backend -> frontend: this request failed; ``__meta__['message']`` says why. The
           backend stays alive (a bad anchor must not kill a bring-up).
(requests are untagged: the frontend writes a bare ``base64(npz)`` line to the backend's
stdin, because there is exactly one request kind per line and ``__meta__['kind']`` names it.)
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np

#: Prefix of every backend->frontend line. Chosen distinct from the older
#: ``MUSEVIZ`` (scripts/muse_force_*.py) so a mis-wired launcher that starts the WRONG
#: backend fails the handshake instead of half-decoding a foreign payload.
LINE_PREFIX = "MUSEHL"

#: Key of the JSON side-channel inside every npz.
META_KEY = "__meta__"

# --- request kinds ---------------------------------------------------------
#: A live observation assembled by the FLoRe planner from the rig's sensor rings.
KIND_LIVE = "live"
#: A dataset anchor: the backend builds the observation itself from the training zarr,
#: exactly as ``eval.py`` does. Offline inference only.
KIND_ZARR = "zarr"
#: Handshake re-request (used by the frontend's readiness wait; harmless to repeat).
KIND_PING = "ping"

# --- reply tags -----------------------------------------------------------
TAG_READY = "ready"
TAG_CHUNK = "chunk"
TAG_ERROR = "error"


def pack(tag: str, meta: dict | None = None, **arrays: np.ndarray) -> str:
    """Encode one protocol line (WITHOUT the trailing newline).

    ``tag`` empty => a request line (no prefix), which is what the frontend writes to the
    backend's stdin. A non-empty tag prefixes the line with :data:`LINE_PREFIX`.
    """
    payload = {META_KEY: np.array(json.dumps(meta or {}))}
    for name, value in arrays.items():
        if name == META_KEY:
            raise ValueError(f"{META_KEY!r} is reserved; pass it via the meta argument")
        payload[name] = np.asarray(value)
    buf = io.BytesIO()
    np.savez(buf, **payload)
    body = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"{LINE_PREFIX}{tag} {body}" if tag else body


def unpack(line: str) -> tuple[str, dict, dict[str, np.ndarray]]:
    """Decode one protocol line into ``(tag, meta, arrays)``.

    ``tag`` is ``""`` for a request line. Raises ``ValueError`` on anything that is not a
    well-formed message, so a caller can treat backend chatter as chatter: test with
    :func:`is_message` first rather than catching this.
    """
    text = line.strip()
    if not text:
        raise ValueError("empty protocol line")
    tag = ""
    if text.startswith(LINE_PREFIX):
        head, _, body = text.partition(" ")
        tag = head[len(LINE_PREFIX):]
        if not body:
            raise ValueError(f"protocol line {head!r} has no payload")
    else:
        body = text
    try:
        raw = base64.b64decode(body, validate=True)
    except Exception as exc:  # binascii.Error and friends
        raise ValueError(f"payload is not valid base64: {exc}") from exc
    with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
        names = list(npz.files)
        if META_KEY not in names:
            raise ValueError(f"payload has no {META_KEY!r} entry")
        meta = json.loads(str(npz[META_KEY]))
        arrays = {name: npz[name] for name in names if name != META_KEY}
    if not isinstance(meta, dict):
        raise ValueError(f"{META_KEY!r} must decode to a JSON object, got {type(meta).__name__}")
    return tag, meta, arrays


def is_message(line: str) -> bool:
    """True when ``line`` looks like a backend->frontend protocol line.

    A cheap prefix test, NOT a parse: the frontend uses it to split protocol from the
    model's own stdout noise before spending a base64+npz decode on it.
    """
    return line.strip().startswith(LINE_PREFIX)


__all__ = [
    "LINE_PREFIX",
    "META_KEY",
    "KIND_LIVE",
    "KIND_ZARR",
    "KIND_PING",
    "TAG_READY",
    "TAG_CHUNK",
    "TAG_ERROR",
    "pack",
    "unpack",
    "is_message",
]
