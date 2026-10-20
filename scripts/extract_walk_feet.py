#!/usr/bin/env python3
"""Extract the two FEET of a holosoma motion NPZ into the small bundle the 5pt page ships with.

    uv run scripts/extract_walk_feet.py                       # the default source -> the bundled asset
    uv run scripts/extract_walk_feet.py --src other.npz --out /tmp/feet.npz

The extraction is ``flore.s1_policy.serving.walk_skill.load_walk_clip`` (ankle_roll_link origins in the
source clip's own per-frame PELVIS body frame, xyzw with w >= 0 — the low level's ``_compute_vr_fk``
convention, cross-checked against mjswan's bake to 0.0013 mm — PLUS, since 2026-09-07, the clip's 29
joint targets + velocities by name, sent as the 5pt wire's ``ref_joints`` so the low level's
``command_lower_body`` describes the same gait as the feet); this script only WRITES it out with
provenance (source name + sha256) via ``save_feet_clip`` and prints the clip's demand. Re-run it if the
extraction ever changes; ``tests/test_walk_skill.py`` pins the bundle equal to a fresh extraction of the
source whenever the source NPZ is present on the box.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flore.s1_policy.serving import walk_skill  # noqa: E402

DEFAULT_SRC = Path(os.path.expanduser("~/data/walk_clip")) / walk_skill.DEFAULT_CLIP


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="a bones_seed_g1_holosoma_f32 motion NPZ "
                    "(default: %(default)s)")
    ap.add_argument("--out", default=str(walk_skill.BUNDLED_CLIP),
                    help="where to write the extracted feet NPZ (default: the bundled asset, %(default)s)")
    args = ap.parse_args(argv)
    src = Path(os.path.expanduser(args.src))
    if not src.is_file():
        print(f"no such source clip: {src}", file=sys.stderr)
        return 2
    clip = walk_skill.load_walk_clip(src)
    if clip.source != str(src) or Path(clip.source).suffix != ".npz" or clip.name != src.name:
        print(f"{src} did not load as a raw holosoma clip (name {clip.name!r}); refusing to re-bundle a bundle",
              file=sys.stderr)
        return 2
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    out = walk_skill.save_feet_clip(clip, os.path.expanduser(args.out), source_sha256=sha)
    back = walk_skill.load_walk_clip(out)
    assert back.foot_pos.shape == clip.foot_pos.shape and (back.foot_pos == clip.foot_pos).all()
    assert (back.foot_quat == clip.foot_quat).all() and (back.root_yaw_deg == clip.root_yaw_deg).all()
    if clip.has_joints:
        assert back.has_joints and back.joint_names == clip.joint_names
        assert (back.joint_pos == clip.joint_pos).all() and (back.joint_vel == clip.joint_vel).all()
    else:
        print(f"  ⚠️ no joint channel in {src.name} ({clip.legs_note}) — the bundle is FEET-ONLY", file=sys.stderr)
    st = clip.stats()
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB) from {src.name} sha256 {sha[:16]}…")
    print(f"  {clip.describe()}")
    print("  " + st.describe(feet_slewed=False))
    print(f"  frame 0: L {clip.foot_pos[0, 0].round(4).tolist()}  R {clip.foot_pos[0, 1].round(4).tolist()} "
          f"(pelvis frame, m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
