#!/usr/bin/env python3
"""Make the master of the page's data-collection clip: the demonstration as filmed, with the two
strips of tape on the table taken out and an arrow drawn round the crate as it turns.

    python3 scripts/clean_data_collection.py --src PAPER_DIR

It reads PAPER_DIR/supp/data_collection.MP4, which it never changes, and writes
PAPER_DIR/supp/data_collection_clean.mp4 (H.264 High 10 at CRF 9, every frame of the source), from
which build_media.py makes the page's clip.  Needs numpy, OpenCV and Pillow, and an ffmpeg with
libx264 (Homebrew: ffmpeg-full).

The camera stands still, but the table drifts by up to two pixels as the crate is dragged across
it, so each frame is first matched to the first one.  Under each tape the table is redrawn once,
from the frames in which nothing covers it, carrying the seams and edges that cross the tape over
it; each frame then takes that drawing, with what has changed on the bare table round the tape
since, wherever the tape itself shows.  Whatever is in front of the tape (the crate, a hand, a
cable) is kept, less the tape's colour on its edges.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

log = logging.getLogger("clean_data_collection")

SRC = "supp/data_collection.MP4"
OUT = "supp/data_collection_clean.mp4"
FFMPEG_CANDIDATES = ("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg", "ffmpeg")
W, H = 1920, 1080
RATE = (30000, 1001)
FRAMES = 289                    # the source's frames, which the constants below count on
FPS = RATE[0] / RATE[1]
CRF = 9
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)        # BT.709
# This ffmpeg lets the frames' unset colour properties override -color_* on the command line, so
# the tags are set on the frames, as build_media.py does.
TAGS = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"

ROI = (560, 680, 1140, 400)     # x, y, width, height of the part of the frame the tapes lie in


@dataclass(frozen=True)
class Tape:
    name: str
    ends: tuple[tuple[float, float], tuple[float, float]]   # its centre line, far end to near end
    band: float                 # px either side of that line it lies within
    hue: tuple[float, float, float]   # weights of R, G, B that read its colour: large on it, ~0 on the table
    floor: float                # that reading, at least once in the clip, on every pixel of it
    plain: range                # frames in which nothing covers it
    grow: int                   # px of table round it that are redrawn too: its coloured fringe
    fill_angle: float           # degrees: the table is redrawn under it along the seams and edges crossing it
    reach: float                # px: how far the changes on the bare table round it are carried in
    despill: bool = False       # whether its colour is also taken off the crate's foot beside it


# The crate covers the far end of the green tape from 5.6 s on, and that of the yellow one until
# then, so each tape's picture comes from the other part of the clip.
TAPES = (
    Tape("green", ((675.5, 798.4), (1122.6, 1022.1)), 40, (-0.5, 1.0, -0.5), 0.07,
         range(0, 150, 2), 6, -12.5, 9.0, despill=True),
    Tape("yellow", ((1113.6, 729.5), (1624.5, 870.4)), 25, (0.5, 0.5, -1.0), 0.05,
         range(190, 289, 2), 5, -12.0, 7.0),
)

# Two patches of the table that nothing covers, round the near ends of the tapes: the drift of each
# frame is the rigid motion that takes their first picture to theirs.
DRIFT = ((960, 900, 1120, 1040), (1430, 795, 1600, 880))

# The arrow: an orange ring round the crate, in perspective, that grows clockwise as the crate
# turns (clockwise seen from above, so on screen too), its far side hidden behind the crate.  It
# comes out from behind the crate's right end, runs along its front and ends pointing left.  The
# turn takes 4.95 to 5.87 s; the ring then follows the crate as it is dragged left and fades.
ARROW_RGB = (255, 153, 0)       # #ff9900, the page's orange
ARROW_IN, ARROW_GROW, ARROW_OUT = (4.78, 4.95), (4.95, 5.87), (6.45, 6.85)   # s
ARROW_TAIL, ARROW_SPAN = -48.0, 216.0   # degrees round the ring, clockwise on screen from its right
ARROW_SIZE = 0.64               # the ring's half width, as a share of the crate's width
ARROW_TILT = 0.30               # its height over its width: the camera looks down on it a little
ARROW_DROP = 0.18               # its centre below the crate's, as a share of the crate's height
ARROW_WIDTH = 15.0              # px of stroke at the ring's front; its back is narrower, its tail tapers
ARROW_SS = 3                    # supersampling of the drawing


def disc(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def smoothstep(x: np.ndarray | float) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease(x: float) -> float:
    """Cubic ease in and out, as the crate's turn starts and stops."""
    x = min(max(x, 0.0), 1.0)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ffmpeg_bin() -> str:
    for c in FFMPEG_CANDIDATES:
        found = shutil.which(c)
        if found:
            return found
    sys.exit("ffmpeg not found (brew install ffmpeg-full)")


def check_source(ff: str, src: Path) -> None:
    """Everything below is this clip's: a source of another size or rate is refused at once."""
    probe = [Path(ff).with_name("ffprobe"), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate", "-of", "default=nw=1", src]
    out = subprocess.run([str(c) for c in probe], check=True, capture_output=True, text=True).stdout
    info = dict(line.split("=", 1) for line in out.split())   # a stream listed twice says the same
    if (info.get("width"), info.get("height"), info.get("r_frame_rate")) != (str(W), str(H), f"{RATE[0]}/{RATE[1]}"):
        sys.exit(f"{src}: wants {W}x{H} at {RATE[0]}/{RATE[1]} fps, found {info}")


def frames(ff: str, src: Path):
    """The source's frames, as float RGB in 0..1, decoded at 16 bits with the BT.709 matrix."""
    cmd = [ff, "-v", "error", "-i", str(src), "-map", "0:v:0", "-vf",
           "scale=in_color_matrix=bt709:in_range=tv:flags=accurate_rnd+full_chroma_int+bicubic,format=rgb48le",
           "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"]
    size = W * H * 6
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint16).reshape(H, W, 3).astype(np.float32) / 65535
    if p.returncode:
        raise subprocess.CalledProcessError(p.returncode, cmd)


def roi_of(frame: np.ndarray) -> np.ndarray:
    x, y, w, h = ROI
    return frame[y:y + h, x:x + w]


def warp(img: np.ndarray, m: np.ndarray, inverse: bool = False) -> np.ndarray:
    flags = cv2.INTER_LINEAR | (cv2.WARP_INVERSE_MAP if inverse else 0)
    return cv2.warpAffine(img, m, (ROI[2], ROI[3]), flags=flags, borderMode=cv2.BORDER_REPLICATE)


# --- the table's drift ---

def drift(rois: np.ndarray) -> np.ndarray:
    """Per frame, the similarity (2 x 3, in ROI coordinates) taking the first frame's table to the
    frame's: the shifts of the two DRIFT patches, smoothed over five frames."""
    x0, y0 = ROI[:2]

    def patch(i: int, b: tuple[int, int, int, int]) -> np.ndarray:
        g = (rois[i, b[1] - y0:b[3] - y0, b[0] - x0:b[2] - x0].astype(np.float32) / 65535) @ LUMA
        return g - cv2.GaussianBlur(g, (0, 0), 6)

    n = len(rois)
    shifts = np.zeros((n, 2, 2))
    for j, b in enumerate(DRIFT):
        win = cv2.createHanningWindow((b[2] - b[0], b[3] - b[1]), cv2.CV_32F)
        ref = patch(0, b)
        for i in range(n):
            shifts[i, j] = cv2.phaseCorrelate(ref, patch(i, b), win)[0]
    k = np.array([1, 4, 6, 4, 1], np.float64) / 16
    for j in range(2):
        for a in range(2):
            shifts[:, j, a] = np.convolve(np.pad(shifts[:, j, a], 2, mode="edge"), k, "valid")
    p = np.array([((b[0] + b[2]) / 2 - x0, (b[1] + b[3]) / 2 - y0) for b in DRIFT])
    out = np.zeros((n, 2, 3), np.float32)
    for i in range(n):
        q = p + shifts[i]
        z = complex(*(q[1] - q[0])) / complex(*(p[1] - p[0]))    # rotation and scale
        r = np.array([[z.real, -z.imag], [z.imag, z.real]])
        out[i] = np.hstack([r, (q[0] - r @ p[0])[:, None]])
    log.info("table drift: up to %.1f px", np.abs(shifts).max())
    return out


# --- the tapes ---

class TapeModel:
    """One tape: where it lies, its picture with nothing over it, and the table drawn under it, all
    in the first frame's place; `remove` takes it out of a frame."""

    def __init__(self, spec: Tape, rois: np.ndarray, motion: np.ndarray) -> None:
        self.spec = spec
        self.hue = np.array(spec.hue, np.float32)
        w, h = ROI[2], ROI[3]
        # where it lies: what reads as its colour at least once in the clip, inside its band
        (ax, ay), (bx, by) = spec.ends
        length = np.hypot(bx - ax, by - ay)
        d = np.array([bx - ax, by - ay]) / length
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        rx, ry = xx + ROI[0] - ax, yy + ROI[1] - ay
        along, across = rx * d[0] + ry * d[1], -rx * d[1] + ry * d[0]
        band = (along > 0) & (along < length) & (np.abs(across) < spec.band)
        most = np.full((h, w), -1.0, np.float32)
        for r in rois:
            np.maximum(most, (r.astype(np.float32) / 65535) @ self.hue, out=most)
        _, lab, st, _ = cv2.connectedComponentsWithStats(((most > spec.floor) & band).astype(np.uint8), 8)
        core = lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
        self.hole = cv2.dilate(core.astype(np.uint8), disc(spec.grow)).astype(bool)
        # its picture: the median of the frames in which nothing covers it, each matched to the first
        stack = np.stack([warp(rois[i].astype(np.float32) / 65535, motion[i], inverse=True) for i in spec.plain])
        self.picture = np.median(stack, 0)
        # its colour, from inside it, carried out over the fringe round it
        pure = cv2.erode(core.astype(np.uint8), disc(2)).astype(bool)
        fringe = (cv2.dilate(self.hole.astype(np.uint8), disc(3)).astype(bool) & ~pure).astype(np.uint8) * 255
        self.colour = np.stack([cv2.inpaint((self.picture[..., c] * 65535).astype(np.uint16), fringe, 3,
                                            cv2.INPAINT_TELEA) for c in range(3)], -1).astype(np.float32) / 65535
        self.table = self._draw_table()
        self.feather = np.clip(cv2.distanceTransform(self.hole.astype(np.uint8), cv2.DIST_L2, 3) / 2, 0, 1)
        # a short line across the tape: what is in front of it is wide across it, unlike the streaks
        # the coding leaves along its edges
        self.across = np.zeros((9, 9), np.uint8)
        a = np.arctan2(d[1], d[0]) + np.pi / 2
        cv2.line(self.across, (round(4 - 4 * np.cos(a)), round(4 - 4 * np.sin(a))),
                 (round(4 + 4 * np.cos(a)), round(4 + 4 * np.sin(a))), 1, 1)
        log.info("%s tape: %d px", spec.name, int(self.hole.sum()))

    def _draw_table(self) -> np.ndarray:
        """The table under the tape: each pixel interpolated between the two points where a line at
        fill_angle through it leaves the tape, so that the seams and edges crossing it go on."""
        w, h = ROI[2], ROI[3]
        ys, xs = np.nonzero(self.hole)
        t = np.radians(self.spec.fill_angle)
        d = np.array([np.cos(t), np.sin(t)])
        vals, dists = [], []
        for sign in (1, -1):
            dist = np.zeros(len(ys), np.float32)
            done = np.zeros(len(ys), bool)
            for step in range(1, 200):
                px = np.rint(xs + sign * step * d[0]).astype(int)
                py = np.rint(ys + sign * step * d[1]).astype(int)
                inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
                on = np.zeros_like(done)
                on[inside] = self.hole[py[inside], px[inside]]
                left = ~done & ~on
                dist[left] = step
                done |= left
                if done.all():
                    break
            qx = (xs + sign * (dist + 1) * d[0]).astype(np.float32).reshape(-1, 1)
            qy = (ys + sign * (dist + 1) * d[1]).astype(np.float32).reshape(-1, 1)
            vals.append(cv2.remap(self.picture, qx, qy, cv2.INTER_LINEAR)[:, 0])
            dists.append(dist)
        near = (dists[1] / (dists[0] + dists[1]))[:, None]
        table = self.picture.copy()
        table[ys, xs] = near * vals[0] + (1 - near) * vals[1]
        return table

    def remove(self, c: np.ndarray, m: np.ndarray, out: np.ndarray) -> None:
        """Takes the tape out of `c` (the frame's ROI, matched by the similarity `m`) into `out`."""
        w, h = ROI[2], ROI[3]
        hole = cv2.dilate((warp(self.hole.astype(np.float32), m) > 0.5).astype(np.uint8), disc(1)).astype(bool)
        picture, table = warp(self.picture, m), warp(self.table, m)
        colour, feather = warp(self.colour, m), warp(self.feather, m)

        # What has changed on the bare table round the tape since its picture was taken -- the light,
        # reflections, the crate's shadow -- carried in under it; where nothing round it can be seen,
        # the table is darkened or lightened as the tape itself is.
        ring = cv2.dilate(hole.astype(np.uint8), disc(12)).astype(bool) & ~hole
        change = c - picture
        dl = np.abs(change @ LUMA)
        dc = np.abs(change - (change @ LUMA)[..., None]).max(-1)
        seen = (ring & (dl < 0.07) & (dc < 0.04)).astype(np.float32)
        num = cv2.GaussianBlur(change * seen[..., None], (0, 0), self.spec.reach)
        den = cv2.GaussianBlur(seen, (0, 0), self.spec.reach)
        full = cv2.GaussianBlur(ring.astype(np.float32), (0, 0), self.spec.reach)
        ys, xs = np.nonzero(hole)
        cc = c[ys, xs]

        # Does the pixel show the tape's own picture, lighter or darker, give or take a pixel of the
        # drift's error, or the tape's hue at a brightness it can have?  Then it is the tape or its
        # fringe.  Anything else is in front of it, if it is wide across the tape.
        best = np.full(len(ys), np.inf, np.float32)
        scale = np.ones(len(ys), np.float32)
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                p = picture[np.clip(ys + oy, 0, h - 1), np.clip(xs + ox, 0, w - 1)]
                s = (cc * p).sum(1) / np.maximum((p * p).sum(1), 1e-6)
                e = np.linalg.norm(cc - s[:, None] * p, axis=1) / (0.012 + 0.04 * np.linalg.norm(p, axis=1))
                e = np.where((s > 0.3) & (s < 1.6), e, np.inf)
                better = e < best
                best = np.where(better, e, best)
                scale = np.where(better, s, scale)
        k = colour[ys, xs]
        lc, lk = np.maximum(cc @ LUMA, 1e-4), np.maximum(k @ LUMA, 1e-4)
        tinted = ((cc @ self.hue) / lc) / np.maximum((k @ self.hue) / lk, 1e-3)
        tape_hued = (tinted > 0.6) & (lc > 0.3 * lk) & (lc < 1.6 * lk)
        front = np.zeros((h, w), np.uint8)
        front[ys, xs] = (best > 1.0) & ~tape_hued
        front = cv2.dilate(cv2.morphologyEx(front, cv2.MORPH_OPEN, self.across), disc(1))
        soft = cv2.GaussianBlur(front.astype(np.float32), (0, 0), 0.8)
        take = np.clip(1.0 - 1.5 * soft[ys, xs], 0, 1) * np.clip(feather[ys, xs], 0, 1)
        # beside what is in front, a tape-hued pixel that is not the tape's picture is likely the two
        # mixed, a blurred edge: it is unmixed below rather than replaced
        beside = cv2.dilate((soft > 0.5).astype(np.uint8), disc(4))[ys, xs].astype(bool)
        take = np.where(beside & (best > 1.0), 0.0, take)

        shade_n = np.zeros((h, w), np.float32)
        shade_d = np.zeros((h, w), np.float32)
        shade_n[ys, xs] = scale * (take > 0.5)
        shade_d[ys, xs] = take > 0.5
        shade_n, shade_d = cv2.GaussianBlur(shade_n, (0, 0), 3), cv2.GaussianBlur(shade_d, (0, 0), 3)
        dd, sd = den[ys, xs], shade_d[ys, xs]
        shade = np.where(sd > 0.05, shade_n[ys, xs] / np.maximum(sd, 1e-6), 1.0)[:, None]
        sure = np.clip(dd / np.maximum(full[ys, xs], 1e-6) / 0.4, 0, 1)[:, None]
        under = table[ys, xs]
        bare = sure * (under + num[ys, xs] / np.maximum(dd, 1e-6)[:, None]) + (1 - sure) * under * shade
        o = cc + take[:, None] * (bare - cc)

        # What is in front keeps its colour but for the tape's share in its edge pixels, read from
        # the hue against that of what is in front nearby, where no tape shows.
        hb, hk, hc = bare @ self.hue, k @ self.hue, cc @ self.hue
        clean = np.zeros((h, w), np.float32)
        clean[ys, xs] = (take < 0.5) & ((hc - hb) / np.maximum(hk - hb, 0.05) < 0.05)
        by_it = cv2.dilate(hole.astype(np.uint8), disc(8)).astype(bool) & ~hole & ((change @ LUMA) > 0.05)
        clean[by_it] = 1
        rn = cv2.GaussianBlur(c * clean[..., None], (0, 0), 4)
        rd = cv2.GaussianBlur(clean, (0, 0), 4)[ys, xs]
        ref = np.where((rd > 0.02)[:, None], rn[ys, xs] / np.maximum(rd, 1e-6)[:, None], bare)
        hr = ref @ self.hue
        share = np.clip((hc - hr) / np.maximum(hk - hr, 0.05) - 0.03, 0, 1)
        o2 = o - ((1 - take) * share)[:, None] * (k - bare)
        lo = np.minimum(np.minimum(cc, bare), ref) - 0.01
        hi = np.maximum(np.maximum(cc, bare), ref) + 0.01
        out[ys, xs] = np.clip(o2, lo, hi)

        if self.spec.despill:
            # The green tape lights the crate's foot beside it, and tints the edge where the two
            # meet: on the crate (a blue grey, never greener than the mean of its red and blue)
            # round the tape, the green above that is taken off.  The table is left as it is.
            zone = cv2.dilate(hole.astype(np.uint8), disc(14)).astype(bool)
            kept = np.ones((h, w), bool)
            kept[ys, xs] = take < 0.5
            zy, zx = np.nonzero(zone & kept)
            r, g, b = out[zy, zx, 0], out[zy, zx, 1], out[zy, zx, 2]
            crate = np.clip((b - r - 0.03) / 0.03, 0, 1)
            out[zy, zx, 1] = g - crate * np.maximum(0, g - (r + b) / 2 - 0.008)


# --- the arrow ---

def crate_mask(small: np.ndarray) -> np.ndarray:
    """The crate in a downscaled frame (uint8 RGB): the largest blob of its blue grey."""
    a = small.astype(np.float32) / 255
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = (((b - r) > 0.05) & (b > 0.22) & ((b - g) > 0.03)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, disc(1))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return np.zeros(m.shape, np.uint8)
    return (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)


def crate_track(smalls: np.ndarray, scale: int) -> np.ndarray:
    """Per frame, the crate's centre, width and height in px, smoothed over nine frames."""
    rows = []
    for s in smalls:
        m = cv2.morphologyEx(crate_mask(s), cv2.MORPH_CLOSE, disc(3))
        mo = cv2.moments(m)
        _, _, w, h = cv2.boundingRect(m)
        rows.append((mo["m10"] / mo["m00"], mo["m01"] / mo["m00"], w, h))
    track = np.array(rows, np.float64) * scale
    k = cv2.getGaussianKernel(9, 2.0)[:, 0]
    return np.stack([np.convolve(np.pad(track[:, j], 4, mode="edge"), k, "valid") for j in range(4)], 1)


def arrow_state(i: int) -> tuple[float, float]:
    """The arrow's opacity and how far it has grown, 0..1, in frame i."""
    t = i / FPS
    alpha = float(smoothstep((t - ARROW_IN[0]) / (ARROW_IN[1] - ARROW_IN[0]))
                  * (1 - smoothstep((t - ARROW_OUT[0]) / (ARROW_OUT[1] - ARROW_OUT[0]))))
    return alpha, ease((t - ARROW_GROW[0]) / (ARROW_GROW[1] - ARROW_GROW[0]))


def draw_arrow(frame: np.ndarray, i: int, track: np.ndarray) -> np.ndarray:
    """The frame with the arrow drawn on it, if it shows in frame i."""
    alpha, grown = arrow_state(i)
    if alpha <= 0.002:
        return frame
    cx, cy, bw, bh = track[i]
    a = ARROW_SIZE * bw
    b = ARROW_TILT * a
    cy += ARROW_DROP * bh
    span = 24 + (ARROW_SPAN - 24) * grown          # a little of it shows from the start
    th = np.radians(np.linspace(ARROW_TAIL, ARROW_TAIL + span, max(8, int(span * 2))))
    px, py = cx + a * np.cos(th), cy + b * np.sin(th)
    tx, ty = -a * np.sin(th), b * np.cos(th)
    tn = np.hypot(tx, ty)
    tx, ty = tx / tn, ty / tn
    u = np.linspace(0, 1, len(th))
    width = ARROW_WIDTH * (0.78 + 0.22 * np.sin(th)) * (0.3 + 0.7 * smoothstep(u / 0.3))
    head_w = head_l = width[-1] * 2.9
    # the stroke stops where the head begins
    back = np.cumsum(np.hypot(np.diff(px), np.diff(py))[::-1])[::-1]
    keep = np.concatenate([back > head_l * 0.85, [False]])
    keep[0] = True
    left = np.stack([px - ty * width / 2, py + tx * width / 2], 1)[keep]
    right = np.stack([px + ty * width / 2, py - tx * width / 2], 1)[keep]
    tip = np.array([px[-1] + tx[-1] * head_l * 0.15, py[-1] + ty[-1] * head_l * 0.15])
    base = np.array([px[-1] - tx[-1] * head_l, py[-1] - ty[-1] * head_l])
    wing = np.array([-ty[-1], tx[-1]]) * head_w / 2
    xs = np.concatenate([px, [tip[0], base[0] + wing[0], base[0] - wing[0]]])
    ys = np.concatenate([py, [tip[1], base[1] + wing[1], base[1] - wing[1]]])
    x0, x1 = int(max(0, xs.min() - 40)), int(min(W, xs.max() + 40))
    y0, y1 = int(max(0, ys.min() - 40)), int(min(H, ys.max() + 40))

    pic = Image.new("L", ((x1 - x0) * ARROW_SS, (y1 - y0) * ARROW_SS), 0)
    pen = ImageDraw.Draw(pic)

    def at(pts: np.ndarray) -> list[tuple[float, float]]:
        return [((x - x0) * ARROW_SS, (y - y0) * ARROW_SS) for x, y in pts]

    pen.polygon(at(np.concatenate([left, right[::-1]])), fill=255)
    (ex, ey), = at(np.array([[px[0], py[0]]]))
    r0 = width[0] / 2 * ARROW_SS
    pen.ellipse([ex - r0, ey - r0, ex + r0, ey + r0], fill=255)
    pen.polygon(at(np.array([base + wing, tip, base - wing])), fill=255)
    pic = pic.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    ink = np.asarray(pic, np.float32) / 255
    shadow = np.roll(np.asarray(pic.filter(ImageFilter.GaussianBlur(4)), np.float32) / 255, (3, 2), (0, 1))

    # the half of the ring behind the crate is hidden by it: its silhouette, the convex hull of its
    # blue grey, above the ring's centre line
    small = cv2.resize(frame, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    m = crate_mask((np.clip(small, 0, 1) * 255).astype(np.uint8))
    hull = np.zeros(m.shape, np.uint8)
    if m.any():
        cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(m)), 1)
    hull = cv2.GaussianBlur(cv2.resize(hull.astype(np.float32), (W, H)), (0, 0), 1.5)[y0:y1, x0:x1]
    far = np.clip((cy - np.arange(y0, y1, dtype=np.float32)[:, None]) / 6 + 0.5, 0, 1)
    ink *= (1 - far * hull) * alpha
    shadow *= (1 - far * hull) * alpha * 0.55

    out = frame.copy()
    reg = out[y0:y1, x0:x1] * (1 - shadow[..., None])
    out[y0:y1, x0:x1] = reg * (1 - ink[..., None]) + np.array(ARROW_RGB, np.float32) / 255 * ink[..., None]
    return out


# --- the run ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="the paper's working folder")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    src_root = args.src.expanduser().resolve()
    src, out = src_root / SRC, src_root / OUT
    ff = ffmpeg_bin()
    check_source(ff, src)
    started = time.time()

    # first pass: the tapes' part of every frame, and every frame at a quarter size for the crate
    x, y, w, h = ROI
    rois, smalls = [], []
    for f in frames(ff, src):
        rois.append((roi_of(f) * 65535 + 0.5).astype(np.uint16))
        smalls.append((cv2.resize(f, (W // 4, H // 4), interpolation=cv2.INTER_AREA) * 255 + 0.5).astype(np.uint8))
    if len(rois) != FRAMES:
        sys.exit(f"{src}: wants {FRAMES} frames, found {len(rois)}")
    rois, smalls = np.stack(rois), np.stack(smalls)
    log.info("%d frames read", len(rois))
    motion = drift(rois)
    tapes = [TapeModel(t, rois, motion) for t in TAPES]
    track = crate_track(smalls, 4)
    del rois, smalls

    # second pass: each frame cleaned, the arrow drawn, into the encoder
    tmp = out.with_name(f"{out.stem}.part{out.suffix}")
    num, den = RATE
    cmd = [ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{W}x{H}",
           "-r", f"{num}/{den}", "-i", "-", "-vf",
           f"scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p10le,{TAGS}",
           "-c:v", "libx264", "-profile:v", "high10", "-preset", "slow", "-crf", str(CRF),
           "-an", "-movflags", "+faststart", "-map_metadata", "-1", str(tmp)]
    try:
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as enc:
            for i, f in enumerate(frames(ff, src)):
                c = roi_of(f).copy()
                o = c.copy()
                for t in tapes:
                    t.remove(c, motion[i], o)
                f[y:y + h, x:x + w] = o
                f = draw_arrow(f, i, track)
                enc.stdin.write((np.clip(f, 0, 1) * 65535 + 0.5).astype(np.uint16).tobytes())
            enc.stdin.close()
        if enc.returncode:
            raise subprocess.CalledProcessError(enc.returncode, cmd)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    log.info("wrote %s in %.0f s", out, time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
