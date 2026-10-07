#!/usr/bin/env python3
"""Build the interactive controller demo in ``demo/`` with the code release's own bundle builder.

It runs ``wbt_realtime_eval.setup_builder`` (what ``wbt-realtime-eval`` runs) for the 5-keypoint
controller, then adapts the static bundle to the project page:

* the client is built for the sub-path it is served from (``/FLoRe/demo/``, or
  ``MJSWAN_DEMO_BASE``). mjswan bakes ``base_path="/"`` by default, and it has to be the absolute
  path: with a relative base the client reads the first path segment as a project id;
* the camera starts closer and in front of the robot. With ``originType: AUTO`` the look-at point
  follows the pelvis, so its height is an offset from the pelvis, not from the floor;
* ``scripts/demo_embed.js`` is added, so the debug panels fold away and, inside the page
  (``?embed``), start smaller with the hybrid controls folded.

Run it with the Python environment of the release's browser-viz stack:

    python scripts/build_demo.py --model-path model_23000.onnx --motion-dir motions
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mjswan

# the package re-exports a function called `main`, which hides the module of the same name
realtime_eval = importlib.import_module("wbt_realtime_eval.main")

SITE = Path(__file__).resolve().parent.parent
PRESET = "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt"
BASE_PATH = os.environ.get("MJSWAN_DEMO_BASE", "/FLoRe/demo/")
CAMERA = {"distance": 2.6, "lookat": [0.0, 0.0, 0.05], "elevation": -12.0, "azimuth": 45.0}
TITLE = "FLoRe: interactive force-position controller"
log = logging.getLogger("build_demo")


def adapt(bundle: Path) -> None:
    config_path = bundle / "assets" / "config.json"
    config = json.loads(config_path.read_text())
    for project in config["projects"]:
        for scene in project["scenes"]:
            scene["camera"].update(CAMERA)
    config_path.write_text(json.dumps(config))

    shutil.copy(SITE / "scripts" / "demo_embed.js", bundle / "flore-embed.js")
    index = bundle / "index.html"
    html = index.read_text()
    html = html.replace("<title>mjswan</title>", f"<title>{TITLE}</title>")
    html = html.replace("</head>", '    <script src="flore-embed.js" defer></script>\n  </head>', 1)
    index.write_text(html)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", type=Path, required=True, help="the checkpoint's ONNX (stripped on the way)")
    ap.add_argument("--motion-dir", type=Path, required=True, help="holosoma clips; idle_default is added")
    ap.add_argument("--out", type=Path, default=SITE / "demo")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    init = mjswan.Builder.__init__

    def sub_path_init(self: mjswan.Builder, base_path: str = BASE_PATH, **kwargs: Any) -> None:
        init(self, base_path=base_path, **kwargs)

    mjswan.Builder.__init__ = sub_path_init
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        builder = realtime_eval.setup_builder(
            preset=PRESET,
            model_path=args.model_path.resolve(),
            motion_dir=args.motion_dir.resolve(),
            output_dir=bundle,
            scene_xml=None,
            force_cmd_hand_frame=False,
            show_control_point=True,
            align_inertia=True,
        )
        builder.build(output_dir=str(bundle))
        adapt(bundle)
        if args.out.exists():
            shutil.rmtree(args.out)
        shutil.copytree(bundle, args.out)
    log.info("demo written to %s", args.out)


if __name__ == "__main__":
    main()
