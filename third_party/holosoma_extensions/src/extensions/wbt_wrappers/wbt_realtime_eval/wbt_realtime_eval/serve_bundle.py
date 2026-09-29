#!/usr/bin/env python3
"""Serve a built wbt-realtime-eval (mjswan) bundle locally.

`python -m http.server` does NOT work for these bundles: the multi-threaded
WASM (onnxruntime + MuJoCo) needs `SharedArrayBuffer`, which the browser only
exposes when the page is "cross-origin isolated" — i.e. the server must send
`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy:
require-corp`. Plain http.server sends neither and is single-threaded (the
~750 MB ONNX blocks every other request), so the viewer hangs on "loading
scene". This server sends both headers and is threaded.

Usage::

    # default: serve ./ (cwd) on :8000
    python -m wbt_realtime_eval.serve_bundle

    # explicit dir + port
    python -m wbt_realtime_eval.serve_bundle <bundle-dir> --port 8000

Then (remote box) forward the port from your laptop:

    ssh -L 8000:localhost:8000 <your-ssh-to-this-box>

and open http://localhost:8000  (hard-refresh / incognito if it cached a
non-isolated state).
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import socketserver
import sys


class _COIHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + the cross-origin-isolation headers that
    SharedArrayBuffer (MuJoCo/ORT WASM threading) requires."""

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        # Avoid serving a stale, non-isolated copy from a previous run.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class _ThreadingServer(socketserver.ThreadingTCPServer):
    # Threaded so the big ONNX download doesn't block the wasm/npz fetches;
    # reuse_address so a quick restart doesn't hit "address already in use".
    allow_reuse_address = True
    daemon_threads = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="serve_bundle", description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Bundle dir to serve (the dir containing index.html). Default: cwd.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port (default 8000).")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0).")
    args = parser.parse_args(argv)

    directory = os.path.abspath(args.directory)
    if not os.path.isfile(os.path.join(directory, "index.html")):
        print(
            f"No index.html in {directory!r} — point this at a built bundle dir "
            "(the one with index.html, e.g. .../bench_ckpt_26500/realtime_eval_dist).",
            file=sys.stderr,
        )
        return 1

    handler = functools.partial(_COIHandler, directory=directory)
    with _ThreadingServer((args.host, args.port), handler) as httpd:
        print(f"Serving {directory}")
        print(f"  http://localhost:{args.port}   (Ctrl-C to stop)")
        print("  cross-origin-isolated headers ON, threaded — hard-refresh the browser.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
