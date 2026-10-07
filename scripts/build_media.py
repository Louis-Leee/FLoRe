#!/usr/bin/env python3
"""Build every clip and picture of the FLoRe project page from the paper's working folder.

The source tree (ICRA2027/) is only read.  An output that already exists is skipped, so a
re-run fills in what is missing; --force rebuilds, --only limits the run to some outputs.

    python3 scripts/build_media.py --src ~/Desktop/ICRA2027
    python3 scripts/build_media.py --src ~/Desktop/ICRA2027 --only supp hero --force

Needs Pillow and numpy, and an ffmpeg with libass (Homebrew: ffmpeg-full) for the
subtitles of the narrated video.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

log = logging.getLogger("build_media")

SITE = Path(__file__).resolve().parent.parent
VIDEOS = SITE / "static" / "videos"
IMAGES = SITE / "static" / "images"
POSTERS = IMAGES / "posters"

FFMPEG_CANDIDATES = ("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg", "ffmpeg")
# 🌿 (U+1F33F) from Noto Emoji, pinned to a tag: its image resources are Apache-2.0.
LEAF_URL = "https://raw.githubusercontent.com/googlefonts/noto-emoji/v2.047/png/512/emoji_u1f33f.png"

# One H.264 file per clip: every browser decodes it, so there is no HEVC twin to keep in step.
# x264's slow preset picks level 5.0 at 1080p through its five reference frames; four keep it
# at 4.1, which phones decode in hardware.  This ffmpeg lets the frames' unset colour
# properties override -color_* on the command line, so the tags are set on the frames.
X264 = ["-c:v", "libx264", "-preset", "slow", "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-refs", "4", "-level", "4.1", "-x264-params", "aq-mode=3"]
TAGS = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709"
MP4 = ["-movflags", "+faststart", "-map_metadata", "-1", "-map_chapters", "-1"]


@dataclass(frozen=True)
class Clip:
    name: str       # static/videos/<name>.mp4 and static/images/posters/<name>.jpg
    src: str        # relative to --src
    height: int     # output height; the width follows the source's 16:9
    crf: float
    poster_width: int = 960


# The order inside each task is the order of the anonymous Google Site.  Trial clips play
# muted on the page; their camera audio is dropped.
CLIPS = (
    *(Clip(f"wipe_board_{n}", f"supp/wipe_board/{n}.mp4", 1080, 24) for n in (4, 11, 13, 9, 10, 1, 6, 12)),
    *(Clip(f"rotate_box_{n}", f"supp/rotate_box/{n}.mp4", 1080, 24) for n in (2, 5, 3, 4)),
    *(Clip(f"push_gantry_{n}", f"supp/push_gantry/{n}.mp4", 1080, 24) for n in (3, 1, 4, 2)),
    Clip("push_gantry_matte", "video/IMG_3227_trans.mp4", 1080, 23, poster_width=1280),
    Clip("controller", "supp/2026-09-22 13-31-23.mp4", 720, 22, poster_width=1280),
)

# The hero reel: the side view of the gantry push that opens the supplementary video.
HERO_SRC, HERO_LEN, HERO_SEAM = "supp/push_gantry/IMG_3352.MOV", 16.0, 0.6

# The narrated supplementary video, rendered from the raw screen recording (no mosaics) with
# the subtitles and narration that supp/narration/render_narration.py last wrote.
SUPP_SRC = "supp/supp_submission.mov"
SUPP_SUBS = "supp/narration/build/subs.ass"
SUPP_FONTS = "supp/narration/build/fonts"
SUPP_AUDIO = "supp/narration/build/narration.wav"
SUPP_CRF = 20

# name -> (source, output width); plots are line art and stay PNG
PLOTS = {
    "force_wipe_board_a": ("supp/wipe_board/force_paper_x2.8.png", 1200),
    "force_wipe_board_b": ("supp/wipe_board/force_paper_x1.8.png", 1200),
    "force_rotate_box_a": ("supp/rotate_box/force_paper_synth_right.png", 1200),
    "force_rotate_box_b": ("supp/rotate_box/force_paper_x0.8.png", 1200),
    "force_push_gantry": ("supp/push_gantry/force_s0.10_a1.00_g0.60_t0.25_p0.35_seed0.png", 1800),
}
PIPELINE_SRC = "ppt/main.png"

# supp/device.png has an emoji over the demonstrator's face.  Underneath, in slide 32 of
# FloRe.pptx, is the cutout image154.png; template matching on the torso placed it in
# device.png at this scale and offset (mean error 3.9/255 outside the emoji).
DEVICE_SRC, DEVICE_PPTX, DEVICE_CUTOUT = "supp/device.png", "ppt/FloRe.pptx", "ppt/media/image154.png"
CUTOUT_SCALE, CUTOUT_X, CUTOUT_Y = 0.98850, -177.75, 24.75


def run(cmd: list) -> None:
    subprocess.run([str(c) for c in cmd], check=True)


def ffmpeg_bin() -> str:
    for c in FFMPEG_CANDIDATES:
        found = shutil.which(c)
        if found:
            return found
    sys.exit("ffmpeg not found (brew install ffmpeg-full)")


def poster(ff: str, video: Path, out: Path, width: int, at: float = 0.0) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    run([ff, "-v", "error", "-y", "-ss", f"{at:.3f}", "-i", video, "-frames:v", "1",
         "-vf", f"scale={width}:-2:flags=lanczos", "-q:v", "3", out])


def build_clip(ff: str, src_root: Path, c: Clip) -> None:
    out = VIDEOS / f"{c.name}.mp4"
    run([ff, "-v", "error", "-stats", "-y", "-i", src_root / c.src, "-map", "0:v:0", "-an",
         "-vf", f"scale=-2:{c.height}:flags=lanczos,{TAGS}", *X264, "-crf", f"{c.crf:g}", *MP4, out])
    poster(ff, out, POSTERS / f"{c.name}.jpg", c.poster_width)


def build_hero(ff: str, src_root: Path) -> None:
    """A loop with no visible seam: the first HERO_SEAM seconds are cut from the head and
    faded in over the tail, so the last frame leads straight into the first."""
    out = VIDEOS / "hero.mp4"
    body = HERO_LEN - HERO_SEAM
    graph = (f"[0:v]scale=1280:-2:flags=lanczos,{TAGS},split[a][b];"
             f"[a]trim=start={HERO_SEAM},setpts=PTS-STARTPTS[body];"
             f"[b]trim=end={HERO_SEAM},setpts=PTS-STARTPTS[head];"
             f"[body][head]xfade=transition=fade:duration={HERO_SEAM}:offset={body - HERO_SEAM:.3f}[v]")
    run([ff, "-v", "error", "-stats", "-y", "-t", f"{HERO_LEN}", "-i", src_root / HERO_SRC,
         "-filter_complex", graph, "-map", "[v]", "-an", *X264, "-crf", "28", *MP4, out])
    poster(ff, out, IMAGES / "hero.jpg", 1280)
    # the link preview: the hero's first frame, cropped to 1200 x 630
    with Image.open(IMAGES / "hero.jpg") as im:
        im = im.convert("RGB").resize((1200, 675), Image.LANCZOS).crop((0, 22, 1200, 652))
        im.save(IMAGES / "og.jpg", quality=86, optimize=True, progressive=True)


def filter_path(p: Path) -> str:
    """A path as an ffmpeg filter option; refuses the characters that the graph parser treats
    specially rather than escaping them through its two levels of quoting."""
    s = str(p)
    if any(ch in s for ch in "\\':,;[]="):
        raise ValueError(f"path not usable inside a filter graph: {s}")
    return s


def build_supp(ff: str, src_root: Path) -> None:
    out = VIDEOS / "supp.mp4"
    vf = f"ass={filter_path(src_root / SUPP_SUBS)}:fontsdir={filter_path(src_root / SUPP_FONTS)},{TAGS}"
    run([ff, "-v", "error", "-stats", "-y", "-i", src_root / SUPP_SRC, "-i", src_root / SUPP_AUDIO,
         "-map", "0:v:0", "-map", "1:a:0", "-vf", vf, *X264, "-crf", f"{SUPP_CRF:g}",
         "-c:a", "aac", "-b:a", "128k", "-ac", "1", "-ar", "48000", "-shortest", *MP4, out])
    # the title card once the robot has walked past it, taken from the recording itself so no
    # subtitle sits on it
    poster(ff, src_root / SUPP_SRC, POSTERS / "supp.jpg", 1280, at=18.5)


def flatten(path: Path) -> Image.Image:
    """RGBA on white: the figures are drawn for paper, and the page may be dark."""
    with Image.open(path) as im:
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        return bg.convert("RGB")


def fit(im: Image.Image, width: int) -> Image.Image:
    if im.width <= width:
        return im
    return im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)


def save_jpg(im: Image.Image, out: Path, quality: int = 90) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, quality=quality, subsampling=0, optimize=True, progressive=True)   # 4:4:4 keeps text crisp


def build_plots(src_root: Path) -> None:
    for name, (src, width) in PLOTS.items():
        im = fit(flatten(src_root / src), width)
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
            IMAGES / f"{name}.png", optimize=True)


def build_pipeline(src_root: Path) -> None:
    save_jpg(fit(flatten(src_root / PIPELINE_SRC), 2400), IMAGES / "pipeline.jpg")


def build_device(src_root: Path) -> None:
    dev = flatten(src_root / DEVICE_SRC)
    with zipfile.ZipFile(src_root / DEVICE_PPTX) as z, z.open(DEVICE_CUTOUT) as f:
        cut = Image.open(f).convert("RGBA")
    a = 1.0 / CUTOUT_SCALE
    placed = cut.transform(dev.size, Image.AFFINE, (a, 0, -CUTOUT_X * a, 0, a, -CUTOUT_Y * a),
                           resample=Image.BICUBIC)
    p = np.asarray(placed).astype(np.float32)
    alpha = p[..., 3:4] / 255.0
    person = p[..., :3] * alpha + 255.0 * (1 - alpha)
    d = np.asarray(dev).astype(np.float32)
    # the emoji is wherever the export differs from the cutout around the head
    head = np.zeros(d.shape[:2], dtype=bool)
    head[40:330, 190:540] = True
    emoji = (np.abs(d - person).sum(axis=2) > 60) & head
    m = Image.fromarray((emoji * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(2.5))
    w = np.asarray(m).astype(np.float32)[..., None] / 255.0
    out = np.clip(d * (1 - w) + person * w + 0.5, 0, 255).astype(np.uint8)
    save_jpg(Image.fromarray(out), IMAGES / "device.jpg")


def build_icons() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "leaf.png"
        urllib.request.urlretrieve(LEAF_URL, path)
        with Image.open(path) as im:
            leaf = im.convert("RGBA")
    leaf.resize((64, 64), Image.LANCZOS).save(IMAGES / "favicon.png", optimize=True)
    # iOS fills a transparent touch icon with black
    tile = Image.new("RGBA", (180, 180), (255, 255, 255, 255))
    tile.alpha_composite(leaf.resize((150, 150), Image.LANCZOS), (15, 15))
    tile.convert("RGB").save(IMAGES / "apple-touch-icon.png", optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="the ICRA2027 working folder")
    ap.add_argument("--only", nargs="*", help="output names (e.g. hero supp wipe_board_4 plots device)")
    ap.add_argument("--force", action="store_true", help="rebuild outputs that already exist")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    src_root = args.src.expanduser().resolve()
    ff = ffmpeg_bin()
    for d in (VIDEOS, IMAGES, POSTERS):
        d.mkdir(parents=True, exist_ok=True)

    # name -> (output that marks it done, builder)
    jobs = {c.name: (VIDEOS / f"{c.name}.mp4", lambda c=c: build_clip(ff, src_root, c)) for c in CLIPS}
    jobs |= {
        "hero": (VIDEOS / "hero.mp4", lambda: build_hero(ff, src_root)),
        "supp": (VIDEOS / "supp.mp4", lambda: build_supp(ff, src_root)),
        "plots": (IMAGES / "force_push_gantry.png", lambda: build_plots(src_root)),
        "pipeline": (IMAGES / "pipeline.jpg", lambda: build_pipeline(src_root)),
        "device": (IMAGES / "device.jpg", lambda: build_device(src_root)),
        "icons": (IMAGES / "favicon.png", build_icons),
    }
    unknown = set(args.only or ()) - set(jobs)
    if unknown:
        sys.exit(f"unknown outputs: {', '.join(sorted(unknown))}")
    for name, (done, build) in jobs.items():
        if args.only and name not in args.only:
            continue
        if done.exists() and not args.force:
            log.info("= %s (exists)", name)
            continue
        log.info("> %s", name)
        build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
