#!/usr/bin/env python3
"""Build the page's in-page demo in ``demo-x/``: the controller on the left hand, along x.

``demo/`` is the full demo, with both hands and all three axes, and the page opens it in a tab
of its own. This script leaves it alone, and with it ``build_demo.py`` and the code release's
checkout. It runs the same builder for the same checkpoint, on copies of the release's mjswan
template and command term, which ``scripts/demo_x.patch`` changes:

* the left hand's wall stands from the start of every episode, 2.5 cm ahead of the hand along
  the pelvis's x axis, instead of appearing with the first force command. The runtime publishes
  each wall's plane, and the page draws that plane, centred where the hand meets it, rather than
  seeding a copy of its own;
* each arrow carries its value along the wall's axis, in newtons, so that F^meas can be read
  against F^cmd;
* the panels say what the arrows and the switches are, without the release's debugging terms.

The built bundle then gets the adaptations of ``build_demo.py`` (sub-path, list guards, the
inertial switch off) and its policy file is cut down to two sliders, the left hand's F^cmd and
F^ext along x, and to the left hand's arrows. F^cmd stops at 20 N, as the wall cannot push back
with more than ``k_env`` x 8 cm (22.6 N). Of the clips, only the standing one is kept: the wall
stays where the hand started, as in training, and the others walk the hand into it or reach
through it, pressing with all the wall gives while nothing is commanded. The checkpoint and the
scene are the full demo's own: where a file holds the same thing as in ``demo/``, the bytes of
``demo/`` are kept, so that git stores it once.

Run it with the Python environment of the release's browser-viz stack, after ``build_demo.py``
(``demo/`` must exist):

    python scripts/build_demo_x.py --model-path model_23000.onnx --motion-dir motions
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import build_demo  # the full demo's builder: its preset, its adaptations, its scene check
import mjswan
import mjswan.builder as mjswan_builder
from mjswan._build_client import ClientBuilder

SITE = build_demo.SITE
FULL = SITE / "demo"
BASE_PATH = os.environ.get("MJSWAN_DEMO_X_BASE", "/FLoRe/demo-x/")
PATCH = SITE / "scripts" / "demo_x.patch"
TITLE = "FLoRe: the force-position controller, one hand along x"
# closer than the full demo's, and turned a little toward the left hand and its wall
CAMERA = {"distance": 2.2, "lookat": [0.2, 0.15, 0.1], "elevation": -10.0, "azimuth": 40.0}
NODE = "v25.5.0"  # the release's nodeenv; another node would be installed into it
F_CMD_MAX = 20.0  # N: the wall pushes back with at most k_env x 8 cm = 22.6 N
F_EXT_MAX = 20.0
SLIDERS = [
    {"label": "Fx", "key": "wbt_force_cmd_local__l_fx", "min": 0.0, "max": F_CMD_MAX, "default": 0.0,
     "step": 0.5, "group": "F^cmd, the command (N)", "group_color": "rgb(190, 140, 255)"},
    {"label": "Fx", "key": "wbt_force_ext_local__l_fx", "min": -F_EXT_MAX, "max": F_EXT_MAX, "default": 0.0,
     "step": 0.5, "group": "F^ext, an extra push (N)", "group_color": "rgb(255, 160, 80)"},
]
LEFT = "left_wrist_yaw_link"
MOTIONS = ("idle_default",)  # see the docstring; the full demo lists them all
log = logging.getLogger("build_demo_x")


def run(*cmd: str, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def copy_tree(source: Path, target: Path) -> None:
    """Copy a tree with its installed packages; by cloning the files where the disk can (APFS,
    btrfs, XFS), which takes seconds and no space, and by copying them elsewhere."""
    clone = {"darwin": ["cp", "-cR"], "linux": ["cp", "-R", "--reflink=auto"]}.get(sys.platform)
    if clone and subprocess.run([*clone, str(source), str(target)], capture_output=True, check=False).returncode == 0:
        return
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target, symlinks=True)


@contextlib.contextmanager
def patched_sources(root: Path) -> Iterator[None]:
    """Build from patched copies of mjswan's template and of the release's command term.

    The template is copied whole, with its installed packages and its node, so that nothing the
    build writes (the generated terms, Vite's bundled config, dist/) lands in the release's
    checkout; ``npm install`` is skipped, as the packages are already there.
    ``mjswan.builder`` finds its template beside its own ``__file__``.
    """
    package = Path(mjswan_builder.__file__).resolve().parent
    source = package / "template"
    node = subprocess.run([str(source / ".nodeenv" / "bin" / "node"), "--version"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if node != NODE:
        raise SystemExit(f"{source}/.nodeenv has node {node}, not {NODE}: build demo/ first")

    template = root / "mjswan" / "template"
    template.parent.mkdir()
    copy_tree(source, template)
    shutil.rmtree(template / "dist", ignore_errors=True)  # the full demo's last client
    (root / "mjswan" / "__init__.py").symlink_to(package / "__init__.py")  # Vite reads the version there
    (root / "wbt").mkdir()
    shutil.copy(build_demo.obs_registry._COMMAND_TS, root / "wbt" / "WbtTrackingCommand.ts")

    # all of the patch or none of it (Apple's patch(1) applies what it can, with fuzz)
    run("git", "apply", "--check", str(PATCH), cwd=root)
    run("git", "apply", str(PATCH), cwd=root)

    command_ts = build_demo.obs_registry._COMMAND_TS
    builder_file = mjswan_builder.__file__
    install = ClientBuilder.install_dependencies
    build_demo.obs_registry._COMMAND_TS = root / "wbt" / "WbtTrackingCommand.ts"
    build_demo.inertial_term_off(root)  # on the patched copy, as for demo/
    mjswan_builder.__file__ = str(template.parent / "builder.py")
    ClientBuilder.install_dependencies = lambda self, clean=False: None  # type: ignore[method-assign]
    try:
        yield
    finally:
        build_demo.obs_registry._COMMAND_TS = command_ts
        mjswan_builder.__file__ = builder_file
        ClientBuilder.install_dependencies = install  # type: ignore[method-assign]


def policy_file(bundle: Path) -> Path:
    config = json.loads((bundle / "assets" / "config.json").read_text())
    (policy,) = [p for project in config["projects"] for s in project["scenes"] for p in s["policies"]]
    return bundle / "main" / "assets" / policy["config"]


def one_axis(bundle: Path) -> None:
    """The left hand only, along x: two sliders, its arrows with their values, a standing wall."""
    path = policy_file(bundle)
    config_path = bundle / "assets" / "config.json"
    config = json.loads(config_path.read_text())
    for project in config["projects"]:
        for scene in project["scenes"]:
            scene["camera"].update(CAMERA)
            for listed in scene["policies"]:
                listed["motions"] = [m for m in listed["motions"] if m["name"] in MOTIONS]
    config_path.write_text(json.dumps(config))

    policy: dict[str, Any] = json.loads(path.read_text())
    kept = [m for m in policy["motions"] if m["name"] in MOTIONS]
    if [m["name"] for m in kept] != list(MOTIONS):
        raise SystemExit(f"{path.name}: the clips are not the release's ({[m['name'] for m in policy['motions']]})")
    for clip in policy["motions"]:
        if clip not in kept:
            (path.parent / clip["path"]).unlink()
    policy["motions"] = kept
    extras, motion = policy["extras"], policy["commands"]["motion"]
    keys = {s["key"] for s in extras["sliders"]}
    if not {s["key"] for s in SLIDERS} <= keys:
        raise SystemExit(f"{path.name}: the sliders are not the release's ({sorted(keys)})")
    extras["sliders"] = SLIDERS
    hands = [h["body_name"] for h in extras["hybrid"]["hands"]]
    if hands[0] != LEFT:
        raise SystemExit(f"{path.name}: the hybrid hands are {hands}, not the left hand first")
    # the right hand stays in the controller, which names the hands by their place in this list
    extras["hybrid"]["fixed_wall"] = {"hands": [0], "dir_b": [1.0, 0.0, 0.0]}
    arrows = [b for b in motion["force_viz_bodies"] if b["body_name"] == LEFT]
    if len(arrows) != 1:
        raise SystemExit(f"{path.name}: expected one set of arrows for {LEFT}")
    motion["force_viz_bodies"] = [{**arrows[0], "labels": True, "triad": False, "ext_drop_m": 0.05}]
    path.write_text(json.dumps(policy))

    index = bundle / "index.html"
    index.write_text(index.read_text().replace(f"<title>{build_demo.TITLE}</title>", f"<title>{TITLE}</title>", 1))


def same_npz(new: Path, old: Path) -> bool:
    return build_demo.same_archive(new, old)


def keep_full_bytes(bundle: Path) -> None:
    """Keep ``demo/``'s bytes for every file that holds the same thing there, so that git
    stores the checkpoint, the scene and the clips once. Zip archives (scene, clips) are
    rewritten with new dates on every build and are compared by their members' checksums."""
    build_demo.keep_unchanged_scenes(bundle, FULL)
    for new in bundle.rglob("*"):
        old = FULL / new.relative_to(bundle)
        if not new.is_file() or not old.is_file():
            continue
        if new.read_bytes() == old.read_bytes():
            continue
        if new.suffix == ".npz" and zipfile.is_zipfile(new) and same_npz(new, old):
            shutil.copyfile(old, new)


def check(bundle: Path) -> None:
    if not (bundle / "index.html").is_file():
        raise SystemExit("the build wrote no index.html")
    clients = list((bundle / "assets").glob("index-*.js"))
    if len(clients) != 1:
        raise SystemExit(f"expected one client in assets/, found {len(clients)}")
    stale = [p for p in bundle.rglob("*") if p.is_file() and p.suffix in {".html", ".js", ".json"}
             and "/FLoRe/demo/" in p.read_text(errors="ignore")]
    if stale:
        raise SystemExit(f"the full demo's sub-path is still in {stale}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", type=Path, required=True, help="the checkpoint's ONNX (stripped on the way)")
    ap.add_argument("--motion-dir", type=Path, required=True, help="holosoma clips; idle_default is added")
    ap.add_argument("--out", type=Path, default=SITE / "demo-x")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not FULL.is_dir():
        raise SystemExit(f"{FULL} is missing: build the full demo first")
    out = args.out.resolve()
    release = Path(mjswan_builder.__file__).resolve().parents[4]  # the release's checkout
    for kept in (FULL.resolve(), release):
        if out == kept or kept in out.parents or out in kept.parents:
            raise SystemExit(f"--out {out} would replace {kept}, which this script must leave alone")

    init = mjswan.Builder.__init__

    def sub_path_init(self: mjswan.Builder, base_path: str = BASE_PATH, **kwargs: Any) -> None:
        init(self, base_path=base_path, **kwargs)

    mjswan.Builder.__init__ = sub_path_init  # type: ignore[method-assign]
    try:
        with tempfile.TemporaryDirectory() as tmp, patched_sources(Path(tmp)):
            bundle = Path(tmp) / "bundle"
            builder = build_demo.realtime_eval.setup_builder(
                preset=build_demo.PRESET,
                model_path=args.model_path.resolve(),
                motion_dir=args.motion_dir.resolve(),
                output_dir=bundle,
                scene_xml=None,
                force_cmd_hand_frame=False,
                show_control_point=True,
                align_inertia=True,
            )
            builder.build(output_dir=str(bundle))
            check(bundle)
            build_demo.adapt(bundle)
            one_axis(bundle)
            keep_full_bytes(bundle)
            if out.exists():
                build_demo.keep_unchanged_scenes(bundle, out)
            # the new bundle is copied in full beside the old one before it takes its place
            staged = out.with_name(out.name + ".new")
            shutil.rmtree(staged, ignore_errors=True)
            shutil.copytree(bundle, staged)
            if out.exists():
                shutil.rmtree(out)
            staged.rename(out)
    finally:
        mjswan.Builder.__init__ = init  # type: ignore[method-assign]
    log.info("demo written to %s", out)


if __name__ == "__main__":
    main()
