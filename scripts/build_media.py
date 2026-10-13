#!/usr/bin/env python3
"""Build every clip and picture of the FLoRe project page from the paper's working folder, and
the hero reel's teaser clips from this site's own teaser/, which git ignores.

The paper's working folder is only read.  An output that already exists is skipped, so a
re-run fills in what is missing; --force rebuilds, --only limits the run to some outputs.

    python3 scripts/build_media.py --src PAPER_DIR
    python3 scripts/build_media.py --src PAPER_DIR --only supp hero --force

Needs Pillow, and an ffmpeg with libass (Homebrew: ffmpeg-full) for the
subtitles of the narrated video.
"""
from __future__ import annotations

import argparse
import logging
import math
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

log = logging.getLogger("build_media")

SITE = Path(__file__).resolve().parent.parent
VIDEOS = SITE / "static" / "videos"
IMAGES = SITE / "static" / "images"
POSTERS = IMAGES / "posters"

FFMPEG_CANDIDATES = ("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg", "ffmpeg")
# 🌺 (U+1F33A) from Noto Emoji, pinned to a tag: its image resources are Apache-2.0.
ICON_URL = "https://raw.githubusercontent.com/googlefonts/noto-emoji/v2.047/png/512/emoji_u1f33a.png"

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


# The order inside each task is the order of the anonymous Google Site; each task then has one
# rollout of the position-only baseline, from its failure folder.  Every clip plays muted on the
# page, and the camera audio is dropped: it carries the lab's voices.
CLIPS = (
    *(Clip(f"wipe_board_{n}", f"supp/wipe_board/{n}.mp4", 1080, 24) for n in (4, 11, 13, 9, 10, 1, 6, 12)),
    Clip("wipe_board_baseline", "supp/wipe_board/failure/1.mp4", 1080, 24),
    *(Clip(f"rotate_box_{n}", f"supp/rotate_box/{n}.mp4", 1080, 24) for n in (2, 5, 3, 4)),
    Clip("rotate_box_baseline", "supp/rotate_box/failure/1.MP4", 1080, 24),
    *(Clip(f"push_gantry_{n}", f"supp/push_gantry/{n}.mp4", 1080, 24) for n in (3, 1, 4, 2)),
    Clip("push_gantry_baseline", "supp/push_gantry/failure/2.MP4", 1080, 24),
    Clip("push_gantry_matte", "video/IMG_3227_trans.mp4", 1080, 23, poster_width=1280),
    Clip("controller", "supp/2026-09-22 13-31-23.mp4", 720, 22, poster_width=1280),
    Clip("data_collection", "supp/data_collection.MP4", 1080, 24, poster_width=1280),
)


@dataclass(frozen=True)
class Shot:
    src: str | Path                 # relative to --src; an absolute path is taken as it is
    length: float | None = None     # seconds from its start; None for all of it
    fast_from: float | None = None  # from this second on it plays HERO_FAST times as fast


# The hero reel: one teaser clip per task, in the page's order, then the side view of the gantry
# push that opens the supplementary video.  The rotate-box clip plays faster once the camera
# sets off round the robot, from where its first swing comes to rest: the camera's circuit and
# then the drag, filmed from behind.
TEASER = SITE / "teaser"    # the raw clips, which git ignores
HERO_SHOTS = (
    Shot(TEASER / "1.MP4"),                    # wipe board
    Shot(TEASER / "2.MP4", fast_from=10.45),   # rotate box
    Shot(TEASER / "3.MP4"),                    # push gantry
    Shot("supp/push_gantry/IMG_3352.MOV", length=16.0),
)
HERO_FAST = 2.0
HERO_RATE = (30000, 1001)   # the teaser clips' rate; the side view is resampled to it
HERO_SEAM = 18              # frames of each cross-fade, 0.6 s
HERO_CRF = 28
OG_AT = 0.6                 # the link preview's frame of the side view, where the reel once began

# The narrated supplementary video, rendered from the raw screen recording (no mosaics) with
# the subtitles and narration that supp/narration/render_narration.py last wrote.  The recording's
# title slide is replaced by the clip that slide plays, rendered again with the title alone, in
# supp/push_gantry: `add_title_reveal.py --occlude --edge-offset 90 --x0 356 --no-subtitle --out
# IMG_3352_title_occluded_nosub.mp4`.  Like the slide, it holds the clip's first frame for
# SUPP_TITLE_HOLD seconds, then plays it until the recording's cut to the next slide.  The cut
# keeps its frame, so the narration and the subtitles keep their times.
SUPP_SRC = "supp/supp_submission.mov"
SUPP_TITLE = "supp/push_gantry/IMG_3352_title_occluded_nosub.mp4"
SUPP_TITLE_HOLD = 2.8
SUPP_TITLE_CUT = 597            # the recording's first frame of the next slide, at its 30 fps
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

# the data-collection photo, with the emoji that covers the demonstrator's face
DEVICE_SRC = "supp/device.png"


def run(cmd: list) -> None:
    subprocess.run([str(c) for c in cmd], check=True)


def ffmpeg_bin() -> str:
    for c in FFMPEG_CANDIDATES:
        found = shutil.which(c)
        if found:
            return found
    sys.exit("ffmpeg not found (brew install ffmpeg-full)")


def need(out: Path, force: bool) -> bool:
    """Whether an output is to be (re)built; each output of a job is checked on its own, so a
    missing poster is made again without re-encoding its clip."""
    if force or not out.exists():
        log.info("> %s", out.relative_to(SITE))
        return True
    log.info("= %s (exists)", out.relative_to(SITE))
    return False


def encode(cmd: list, out: Path) -> None:
    """Runs an ffmpeg command whose last argument is the output into a temporary file next to it,
    so that an interrupted encode never leaves a truncated clip that a later run would keep."""
    tmp = out.with_name(f"{out.stem}.part{out.suffix}")
    try:
        run([*cmd, tmp])
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)


def poster(ff: str, video: Path, out: Path, width: int, at: float = 0.0) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    encode([ff, "-v", "error", "-y", "-ss", f"{at:.3f}", "-i", video, "-frames:v", "1",
            "-vf", f"scale={width}:-2:flags=lanczos", "-q:v", "3"], out)


def build_clip(ff: str, src_root: Path, c: Clip, force: bool) -> None:
    out = VIDEOS / f"{c.name}.mp4"
    if need(out, force):
        encode([ff, "-v", "error", "-stats", "-y", "-i", src_root / c.src, "-map", "0:v:0", "-an",
                "-vf", f"scale=-2:{c.height}:flags=lanczos,{TAGS}", *X264, "-crf", f"{c.crf:g}", *MP4], out)
    if need(POSTERS / f"{c.name}.jpg", force):
        poster(ff, out, POSTERS / f"{c.name}.jpg", c.poster_width)


def video_length(ff: str, path: Path) -> float:
    """The length in seconds of a file's first video stream, read by the ffprobe beside ffmpeg.
    A stream that also belongs to a stream group (the DJI files) is listed twice."""
    probe = [Path(ff).with_name("ffprobe"), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration", "-of", "default=nw=1:nk=1", path]
    out = subprocess.run([str(c) for c in probe], check=True, capture_output=True, text=True).stdout
    return float(out.split()[0])


def hero_graph(ff: str, src_root: Path) -> tuple[list, str]:
    """The inputs and filter graph of the reel.  Each shot is cut to a whole number of frames that
    its source is sure to fill, so that every fade starts on a known frame; the shots then fade
    into one another over HERO_SEAM frames.  The loop has no visible seam: the first shot's head
    is cut and faded in over the last one's tail, so the last frame leads straight into the first."""
    num, den = HERO_RATE

    def sec(frames: int) -> str:
        return f"{frames * den / num:.6f}"

    inputs, chains, length = [], [], 0
    for i, s in enumerate(HERO_SHOTS):
        src = src_root / s.src
        used = min(s.length or math.inf, video_length(ff, src))
        speed, plays = "", used
        if s.fast_from is not None:
            t0 = s.fast_from
            speed = f"setpts='if(lt(T,{t0}),PTS,({t0}+(T-{t0})/{HERO_FAST})/TB)',"
            plays = min(used, t0) + max(0.0, used - t0) / HERO_FAST
        frames = math.floor(plays * num / den) - 1
        if frames <= 2 * HERO_SEAM:
            raise ValueError(f"{s.src}: {frames} frames, too few for two fades of {HERO_SEAM}")
        inputs += ["-t", f"{used:.3f}", "-i", src]
        chains.append(f"[{i}:v]setpts=PTS-STARTPTS,{speed}fps={num}/{den},trim=end_frame={frames},"
                      f"scale=1280:720:flags=lanczos,format=yuv420p,setsar=1[s{i}]")
        if i == 0:
            chains.append(f"[s0]split[b][h];[b]trim=start_frame={HERO_SEAM},setpts=PTS-STARTPTS[x0];"
                          f"[h]trim=end_frame={HERO_SEAM}[head]")
            length = frames - HERO_SEAM
            continue
        fade = f"xfade=transition=fade:duration={sec(HERO_SEAM)}:offset={sec(length - HERO_SEAM)}"
        chains.append(f"[x{i - 1}][s{i}]{fade}[x{i}]")
        length += frames - HERO_SEAM
    fade = f"xfade=transition=fade:duration={sec(HERO_SEAM)}:offset={sec(length - HERO_SEAM)}"
    chains.append(f"[x{len(HERO_SHOTS) - 1}][head]{fade},{TAGS}[v]")
    return inputs, ";".join(chains)


def build_hero(ff: str, src_root: Path, force: bool) -> None:
    out = VIDEOS / "hero.mp4"
    if need(out, force):
        inputs, graph = hero_graph(ff, src_root)
        encode([ff, "-v", "error", "-stats", "-y", *inputs, "-filter_complex", graph, "-map", "[v]",
                "-an", *X264, "-crf", f"{HERO_CRF:g}", *MP4], out)
    if need(IMAGES / "hero.jpg", force):
        poster(ff, out, IMAGES / "hero.jpg", 1280)
    if need(IMAGES / "og.jpg", force):
        # the link preview: the side view rather than the reel's first frame, a close-up of the
        # wiping, cropped to 1200 x 630
        with tempfile.TemporaryDirectory() as tmp:
            still = Path(tmp) / "side.png"
            poster(ff, src_root / HERO_SHOTS[-1].src, still, 1280, at=OG_AT)
            with Image.open(still) as im:
                im = im.convert("RGB").resize((1200, 675), Image.LANCZOS).crop((0, 22, 1200, 652))
                im.save(IMAGES / "og.jpg", quality=86, optimize=True, progressive=True)


def filter_path(p: Path) -> str:
    """A path as an ffmpeg filter option; refuses the characters that the graph parser treats
    specially rather than escaping them through its two levels of quoting."""
    s = str(p)
    if any(ch in s for ch in "\\':,;[]="):
        raise ValueError(f"path not usable inside a filter graph: {s}")
    return s


def build_supp(ff: str, src_root: Path, force: bool) -> None:
    out = VIDEOS / "supp.mp4"
    subs = f"ass={filter_path(src_root / SUPP_SUBS)}:fontsdir={filter_path(src_root / SUPP_FONTS)}"
    # The title clip covers the recording's frames before the cut, counted by number, so that every
    # frame keeps the time the narration was laid on (the recording's clock starts at 0.1 s).  A
    # title that ended before the cut would hold its last frame rather than uncover the old one.
    graph = (f"[1:v]tpad=start_duration={SUPP_TITLE_HOLD}:start_mode=clone,fps=30,"
             f"scale=1280:720:flags=lanczos,setsar=1[title];"
             f"[0:v][title]overlay=eof_action=repeat:enable='lt(n,{SUPP_TITLE_CUT})',{subs},{TAGS}[v]")
    if need(out, force):
        encode([ff, "-v", "error", "-stats", "-y", "-i", src_root / SUPP_SRC, "-i", src_root / SUPP_TITLE,
                "-i", src_root / SUPP_AUDIO, "-filter_complex", graph, "-map", "[v]", "-map", "2:a:0",
                *X264, "-crf", f"{SUPP_CRF:g}",
                "-c:a", "aac", "-b:a", "128k", "-ac", "1", "-ar", "48000", "-shortest", *MP4], out)
    if need(POSTERS / "supp.jpg", force):
        # the title once the robot has walked past it, between two subtitles
        poster(ff, out, POSTERS / "supp.jpg", 1280, at=18.5)


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


def build_plots(src_root: Path, force: bool) -> None:
    for name, (src, width) in PLOTS.items():
        if need(IMAGES / f"{name}.png", force):
            im = fit(flatten(src_root / src), width)
            im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
                IMAGES / f"{name}.png", optimize=True)


def build_pipeline(src_root: Path, force: bool) -> None:
    if need(IMAGES / "pipeline.jpg", force):
        save_jpg(fit(flatten(src_root / PIPELINE_SRC), 2400), IMAGES / "pipeline.jpg")


def build_device(src_root: Path, force: bool) -> None:
    if need(IMAGES / "device.jpg", force):
        save_jpg(flatten(src_root / DEVICE_SRC), IMAGES / "device.jpg")


def build_icons(force: bool) -> None:
    if not (need(IMAGES / "favicon.png", force) | need(IMAGES / "apple-touch-icon.png", force)):
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "icon.png"
        urllib.request.urlretrieve(ICON_URL, path)
        with Image.open(path) as im:
            icon = im.convert("RGBA")
    icon.resize((64, 64), Image.LANCZOS).save(IMAGES / "favicon.png", optimize=True)
    # iOS fills a transparent touch icon with black
    tile = Image.new("RGBA", (180, 180), (255, 255, 255, 255))
    tile.alpha_composite(icon.resize((150, 150), Image.LANCZOS), (15, 15))
    tile.convert("RGB").save(IMAGES / "apple-touch-icon.png", optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="the paper's working folder")
    ap.add_argument("--only", nargs="*", help="output names (e.g. hero supp wipe_board_4 plots device)")
    ap.add_argument("--force", action="store_true", help="rebuild outputs that already exist")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    src_root = args.src.expanduser().resolve()
    ff = ffmpeg_bin()
    for d in (VIDEOS, IMAGES, POSTERS):
        d.mkdir(parents=True, exist_ok=True)

    # name -> builder; each builder checks its own outputs one by one
    jobs = {c.name: (lambda f, c=c: build_clip(ff, src_root, c, f)) for c in CLIPS}
    jobs |= {
        "hero": lambda f: build_hero(ff, src_root, f),
        "supp": lambda f: build_supp(ff, src_root, f),
        "plots": lambda f: build_plots(src_root, f),
        "pipeline": lambda f: build_pipeline(src_root, f),
        "device": lambda f: build_device(src_root, f),
        "icons": build_icons,
    }
    unknown = set(args.only or ()) - set(jobs)
    if unknown:
        sys.exit(f"unknown outputs: {', '.join(sorted(unknown))}")
    for name, build in jobs.items():
        if not args.only or name in args.only:
            build(args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
