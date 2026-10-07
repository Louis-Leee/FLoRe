#!/usr/bin/env python3
"""Draw the Controller section's two diagrams into index.html.

    python3 scripts/draw_controller.py

The unified reference generator (where a hand's target goes) and the soft contact phase with its
integrator (one press, over time) are inline SVG between marker comments in index.html, so they
take the page's font and colours, scale with the column and follow the dark theme.  The press is
a toy 1-D run of the paper's law (Sec. III-A, Eq. 3) with all of its constants: a hand that lags
its target and a surface that stops it stand in for the robot.  It illustrates the law and is not
a recording.  Needs only the standard library.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
INDEX = SITE / "index.html"

SUP = 12           # superscripts, in viewBox units: 1:1 with CSS px in the 900 px column
SUP_RISE = 6.5     # how far a superscript sits above the baseline
TOLERANCE = 0.2    # how far, in viewBox units, a simplified curve may stray from the sampled one


# --- labels -------------------------------------------------------------------------------------
# A label is written in a small markup: *f* is an italic variable and `Δ` an upright symbol, both
# in the serif of the paper's mathematics; ^{cmd} is a superscript and _{I} a subscript, upright
# in the page's font.  SVG has no superscripts that every browser draws, so the shifts are dy
# offsets, undone on the next piece of text.

_TOKEN = re.compile(r"\*([^*]+)\*|`([^`]+)`|\^\{([^}]*)\}|_\{([^}]*)\}|([^*`^_]+)")


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rich(markup: str) -> str:
    out, shift, after_italic = [], 0.0, False
    for ital, upright, sup, sub, plain in _TOKEN.findall(markup):
        want = -SUP_RISE if sup else (4.0 if sub else 0.0)
        attrs = []
        if sup and after_italic:
            attrs.append('dx="3"')                   # clear of the slanted letter's top
        if want != shift:
            attrs.append(f'dy="{want - shift:g}"')
            shift = want
        if ital or upright:
            attrs.append(f'class="{"v" if ital else "m"}"')
        if sup or sub:
            attrs.append(f'font-size="{SUP}"')
        piece = esc(ital or upright or sup or sub or plain)
        out.append(f"<tspan {' '.join(attrs)}>{piece}</tspan>" if attrs else piece)
        after_italic = bool(ital)
    return "".join(out)


def text(x: float, y: float, markup: str, cls: str = "", anchor: str = "start") -> str:
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    c = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.1f}" y="{y:.1f}"{c}{a}>{rich(markup)}</text>'


def chip(x0: float, x1: float, y: float) -> str:
    """A patch of the page's background under a label that sits on the surface's hatching."""
    return f'<rect class="chip" x="{x0:.1f}" y="{y - 18:.1f}" width="{x1 - x0:.1f}" height="25" rx="5"/>'


def arrow(x1: float, y1: float, x2: float, y2: float, cls: str) -> str:
    return (f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'marker-end="url(#ctl-{cls})"/>')


def guide(x: float, y1: float, y2: float) -> str:
    return f'<line class="guide" x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}"/>'


def defs(*classes: str, hatch: bool = False) -> str:
    """Arrowheads, one per colour (a marker cannot take its colour from the line that uses it),
    and the surface's hatching.  The ids are the page's, so each is defined in one diagram only."""
    heads = "".join(
        f'<marker id="ctl-{c}" class="{c}" markerUnits="userSpaceOnUse" markerWidth="14" '
        f'markerHeight="12" refX="12" refY="6" orient="auto"><path d="M0,0 L14,6 L0,12 z"/></marker>'
        for c in classes
    )
    pattern = ('<pattern id="ctl-hatch" width="11" height="11" patternUnits="userSpaceOnUse" '
               'patternTransform="rotate(45)">'
               '<line class="hatch" x1="0" y1="0" x2="0" y2="11"/></pattern>' if hatch else "")
    return f"<defs>{heads}{pattern}</defs>"


def svg(height: int, label: str, parts: list[str]) -> str:
    body = "\n          ".join(parts)
    return (f'<svg class="ctl" viewBox="0 0 900 {height}" role="img" aria-label="{esc(label)}">\n'
            f"          {body}\n        </svg>")


# --- 1. the unified reference generator ---------------------------------------------------------

CM = 32.0                        # px per cm along the surface normal
F_CMD, K, SURFACE_CM = 15.0, 200.0, 5.0
I_SETTLED = 5.0                  # the integral once the press has settled, as in the toy press below


def generator() -> str:
    y = 150.0                                        # the line through reference, hand and target
    x_ref = 160.0                                    # p*, the motion's reference
    x_face = x_ref + SURFACE_CM * CM                 # the surface the hand presses on
    x_ff = x_ref + F_CMD / K * 100 * CM              # p* + f_cmd/K
    x_tgt = x_ff + I_SETTLED / K * 100 * CM          # p* + (f_cmd + I)/K
    wall_r, y_dim, y_force = 620.0, 228.0, 84.0
    p = [defs("cmd", "meas", "integ", hatch=True)]
    # the surface, with its name above it
    p.append(f'<rect class="wall" x="{x_face:.1f}" y="44" width="{wall_r - x_face:.1f}" height="200" '
             'fill="url(#ctl-hatch)"/>')
    p.append(f'<line class="face" x1="{x_face:.1f}" y1="44" x2="{x_face:.1f}" y2="244"/>')
    p.append(text((x_face + wall_r) / 2, 32, "surface", "gray", "middle"))
    for x in (x_ref, x_ff, x_tgt):                   # from the three points down to the shift
        p.append(guide(x, y + 15, y_dim + 8))
    # p*: the reference, a hollow ring
    p.append(f'<circle class="ref" cx="{x_ref:.1f}" cy="{y:.1f}" r="9"/>')
    p.append(text(x_ref, y - 24, "*p*^{*}", "", "middle"))
    p.append(text(x_ref - 22, y + 5, "reference", "gray note", "end"))
    # the hand: a gripper holding a pad whose face, at the TCP, is on the surface
    t = x_face
    p.append(
        f'<line class="hand" x1="{t - 62:.1f}" y1="{y:.1f}" x2="{t - 96:.1f}" y2="{y - 44:.1f}"/>'
        f'<rect class="hand" x="{t - 66:.1f}" y="{y - 31:.1f}" width="18" height="62" rx="4"/>'
        f'<rect class="hand" x="{t - 50:.1f}" y="{y - 31:.1f}" width="34" height="10" rx="3"/>'
        f'<rect class="hand" x="{t - 50:.1f}" y="{y + 21:.1f}" width="34" height="10" rx="3"/>'
        f'<rect class="pad" x="{t - 26:.1f}" y="{y - 19:.1f}" width="26" height="38" rx="2"/>'
        f'<circle class="tcp" cx="{t:.1f}" cy="{y:.1f}" r="4.5"/>'
    )
    p.append(text(t - 36, y + 56, "hand", "gray note", "middle"))
    # the forces at the TCP, drawn above it: the command into the surface, the reading out of it
    p.append(guide(t, y_force, y - 36))
    p.append(arrow(t, y_force, t + 92, y_force, "cmd"))
    p.append(arrow(t, y_force, t - 92, y_force, "meas"))
    p.append(chip(t + 100, t + 152, y_force + 6))
    p.append(text(t + 104, y_force + 6, "*f*^{cmd}", "cmd"))
    p.append(text(t - 102, y_force + 6, "*f*^{meas}", "meas", "end"))
    # p_tgt: the target the policy tracks, inside the surface
    s = 9.0
    p.append(f'<path class="tgt" d="M{x_tgt - s:.1f},{y - s:.1f} L{x_tgt + s:.1f},{y + s:.1f} '
             f'M{x_tgt - s:.1f},{y + s:.1f} L{x_tgt + s:.1f},{y - s:.1f}"/>')
    p.append(chip(x_tgt + 16, x_tgt + 66, y + 6))
    p.append(text(x_tgt + 20, y + 6, "*p*^{tgt}"))
    # the shift, as two arrows end to end: the command's part and the integral's
    p.append(arrow(x_ref, y_dim, x_ff, y_dim, "cmd"))
    p.append(arrow(x_ff, y_dim, x_tgt, y_dim, "integ"))
    p.append(text((x_ref + x_face) / 2, y_dim + 26, "*f*^{cmd} / *K*", "cmd", "middle"))
    p.append(chip((x_ff + x_tgt) / 2 - 27, (x_ff + x_tgt) / 2 + 27, y_dim + 26))
    p.append(text((x_ff + x_tgt) / 2, y_dim + 26, "*I* / *K*", "integ", "middle"))
    # the law, to the right
    xl = 660.0
    p.append(text(xl, 112, "*p*^{tgt} = *p*^{*} + `Δ`*x*"))
    p.append(text(xl, 146, "`Δ`*x* = (\u2009*f*^{cmd} + *I*\u2009) / *K*"))
    p.append(text(xl, 180, "*K* = 200 N/m", "gray"))
    p.append(text(xl, 222, "The policy tracks *p*^{tgt};", "gray note"))
    p.append(text(xl, 242, "the surface stops the hand.", "gray note"))
    return svg(270, "A hand presses on a surface. Its target is moved from the motion's reference "
               "p-star, past the hand and into the surface, by f-cmd over K and then by the integral I "
               "over K. The command f-cmd points into the surface and the measured force f-meas out of "
               "it.", p)


# --- 2. the soft contact phase and the integrator: one press ------------------------------------

@dataclass(frozen=True)
class Press:
    k: float = K                 # N/m, the reference generator's stiffness
    ki: float = 0.5              # 1/s
    i_max: float = 40.0          # N
    theta_on: float = 2.0        # N, contact onset
    theta_off: float = 1.5       # N, release
    tau_c: float = 0.05          # s, the contact phase's low-pass
    surface: float = SURFACE_CM / 100   # m from p* along the normal
    k_contact: float = 300.0     # N/m: force per metre of the hand's target beyond the surface
    tau_hand: float = 0.12       # s, how far the hand lags its target
    f_peak: float = F_CMD
    ramp: tuple[float, float, float, float] = (0.3, 0.8, 5.3, 5.8)   # ramp up, hold (until the
    t_end: float = 6.4                                               # integral settles), ramp down
    dt: float = 0.005            # s, the physics step the integrator runs at


@dataclass(frozen=True)
class Sample:
    t: float
    f_cmd: float
    f_meas: float                # its magnitude: the reading points out of the surface
    c: float
    integ: float
    shift: float                 # Δx, m


PRESS = Press()


def press(pp: Press = PRESS) -> list[Sample]:
    a, b, c_, d = pp.ramp

    def f_cmd(t: float) -> float:
        if t <= a or t >= d:
            return 0.0
        if t < b:
            return pp.f_peak * (t - a) / (b - a)
        if t <= c_:
            return pp.f_peak
        return pp.f_peak * (d - t) / (d - c_)

    x = integ = c = hyst = 0.0                       # x: where the hand would be, unstopped
    beta = 1.0 - math.exp(-pp.dt / pp.tau_c)
    out = []
    for n in range(int(round(pp.t_end / pp.dt)) + 1):
        t = n * pp.dt
        fc = f_cmd(t)
        shift = (fc + integ) / pp.k
        x += (shift - x) * pp.dt / pp.tau_hand
        f = pp.k_contact * max(0.0, x - pp.surface)
        hyst = 1.0 if f > pp.theta_on else (0.0 if f < pp.theta_off else hyst)
        c += beta * (hyst - c)
        err = fc - f                                 # f_cmd + f_meas along the normal
        integ = c * min(max(integ + pp.ki * err * pp.dt, -pp.i_max), pp.i_max)
        out.append(Sample(t, fc, f, c, integ, shift))
    return out


def crossing(samples: list[Sample], level: float, rising: bool) -> Sample:
    for s0, s1 in pairwise(samples):
        if (s0.f_meas <= level < s1.f_meas) if rising else (s0.f_meas >= level > s1.f_meas):
            return s1
    raise ValueError(f"the press never {'rises above' if rising else 'falls below'} {level} N")


def at(samples: list[Sample], t: float) -> Sample:
    return min(samples, key=lambda q: abs(q.t - t))


def simplify(points: list[tuple[float, float]], tol: float = TOLERANCE) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker, without recursion: the fewest of the sampled points that keep every
    one of them within tol of the line drawn through them."""
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    spans = [(0, len(points) - 1)]
    while spans:
        i0, i1 = spans.pop()
        (x0, y0), (x1, y1) = points[i0], points[i1]
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy) or 1e-12
        far, idx = 0.0, -1
        for i in range(i0 + 1, i1):
            d = abs(dy * (points[i][0] - x0) - dx * (points[i][1] - y0)) / norm
            if d > far:
                far, idx = d, i
        if far > tol:
            keep[idx] = True
            spans += [(i0, idx), (idx, i1)]
    return [q for q, k in zip(points, keep, strict=True) if k]


def polyline(points: list[tuple[float, float]]) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in simplify(points))


def timeline() -> str:
    pp = Press()
    s = press(pp)
    x0, x1 = 150.0, 872.0

    def tx(t: float) -> float:
        return x0 + (x1 - x0) * t / pp.t_end

    # three rows on one time axis: force, contact phase, target shift
    f_top, f_base, f_max = 40.0, 166.0, 18.0
    c_top, c_base = 204.0, 252.0
    d_top, d_base, d_max = 286.0, 400.0, 11.0        # cm

    def fy(f: float) -> float:
        return f_base - (f_base - f_top) * f / f_max

    def cy(c: float) -> float:
        return c_base - (c_base - c_top) * c

    def dy(cm: float) -> float:
        return d_base - (d_base - d_top) * cm / d_max

    on, off = crossing(s, pp.theta_on, True), crossing(s, pp.theta_off, False)
    p = [defs("axis")]
    # the time the hand is in contact, as the hysteresis has it
    p.append(f'<rect class="band" x="{tx(on.t):.1f}" y="24" width="{tx(off.t) - tx(on.t):.1f}" '
             'height="388"/>')
    p.append(text((tx(on.t) + tx(off.t)) / 2, 16, "in contact", "gray note", "middle"))
    p.append(text((x0 + tx(on.t)) / 2, 16, "free space", "gray note", "middle"))
    p.append(text((tx(off.t) + x1) / 2, 16, "free space", "gray note", "middle"))
    # baselines, and the time axis under all three
    for yb in (f_base, c_base, d_base):
        p.append(f'<line class="base" x1="{x0:.1f}" y1="{yb:.1f}" x2="{x1:.1f}" y2="{yb:.1f}"/>')
    p.append(f'<line class="axis" x1="{x0:.1f}" y1="420" x2="{x1 + 10:.1f}" y2="420" '
             'marker-end="url(#ctl-axis)"/>')
    p.append(text(x1 + 10, 442, "time", "gray note", "end"))
    # row 1: the command and the measured force, and where the hysteresis switches
    p.append(text(x0 - 16, (f_top + f_base) / 2 + 6, "force", "gray note", "end"))
    p.append(f'<path class="cmd dash" d="{polyline([(tx(q.t), fy(q.f_cmd)) for q in s])}"/>')
    p.append(f'<path class="meas" d="{polyline([(tx(q.t), fy(q.f_meas)) for q in s])}"/>')
    p.append(text(tx(1.05), fy(pp.f_peak) - 12, "*f*^{cmd}", "cmd", "middle"))
    late = at(s, 1.4)
    p.append(text(tx(late.t) + 10, fy(late.f_meas) + 26, "|*f*^{meas}|", "meas"))
    # the onset's label goes under the baseline, clear of the command's ramp; the release's beside it
    for ev, lvl, ly in ((on, pp.theta_on, f_base + 21), (off, pp.theta_off, fy(pp.theta_off) - 8)):
        p.append(f'<circle class="ev" cx="{tx(ev.t):.1f}" cy="{fy(lvl):.1f}" r="4.5"/>')
        p.append(text(tx(ev.t) + 9, ly, f"{lvl:.1f} N", "gray note"))
    # row 2: the contact phase
    p.append(text(x0 - 16, (c_top + c_base) / 2 - 3, "contact phase", "gray note", "end"))
    p.append(text(x0 - 16, (c_top + c_base) / 2 + 17, "*c*", "", "end"))
    p.append(text(x0 - 4, c_top + 5, "1", "gray note", "end"))
    p.append(text(x0 - 4, c_base + 5, "0", "gray note", "end"))
    p.append(f'<line class="base faint" x1="{x0:.1f}" y1="{c_top:.1f}" x2="{x1:.1f}" y2="{c_top:.1f}"/>')
    p.append(f'<path class="phase" d="{polyline([(tx(q.t), cy(q.c)) for q in s])}"/>')
    # row 3: the target shift, the command's part dashed and the integral's part shaded
    p.append(text(x0 - 16, (d_top + d_base) / 2 - 3, "target shift", "gray note", "end"))
    p.append(text(x0 - 16, (d_top + d_base) / 2 + 17, "`Δ`*x*", "", "end"))
    surf = dy(SURFACE_CM)
    p.append(f'<line class="surf" x1="{x0:.1f}" y1="{surf:.1f}" x2="{x1:.1f}" y2="{surf:.1f}"/>')
    p.append(text(x0 + 6, surf - 7, "surface", "gray note"))
    ff = [(tx(q.t), dy(q.f_cmd / pp.k * 100)) for q in s]
    tot = [(tx(q.t), dy(q.shift * 100)) for q in s]
    p.append(f'<path class="integ-area" d="{polyline(tot)} L{polyline(ff[::-1])[1:]} z"/>')
    p.append(f'<path class="cmd dash" d="{polyline(ff)}"/>')
    p.append(f'<path class="shift" d="{polyline(tot)}"/>')
    mid = at(s, 3.4)
    ff_cm, tot_cm = mid.f_cmd / pp.k * 100, mid.shift * 100
    p.append(text(tx(mid.t), dy((ff_cm + tot_cm) / 2) + 6, "*I* / *K*", "integ", "middle"))
    p.append(text(tx(mid.t), dy(ff_cm) + 21, "*f*^{cmd} / *K*", "cmd", "middle"))
    return svg(450, "One press over time. The command f-cmd ramps up, holds and ramps down; the "
               "measured force rises once the hand reaches the surface. Above 2.0 newtons the contact "
               "phase c turns on and the integral I builds up, adding to the shift f-cmd over K until "
               "the measured force matches the command; below 1.5 newtons c turns off and I returns "
               "to zero.", p)


# --- into the page ------------------------------------------------------------------------------

def inject(html: str, name: str, drawing: str) -> str:
    """Put the drawing between the diagram's markers, and refuse unless there is exactly one pair of
    them, in order, with no other diagram's marker in between: the captions around them are
    written by hand and must not be overwritten."""
    begin, end = f"<!-- begin draw_controller.py: {name} -->", f"<!-- end draw_controller.py: {name} -->"
    if html.count(begin) != 1 or html.count(end) != 1:
        raise SystemExit(f"index.html needs one begin and one end marker for the {name} diagram")
    i, j = html.index(begin) + len(begin), html.index(end)
    if j < i or "draw_controller.py:" in html[i:j]:
        raise SystemExit(f"the {name} diagram's markers in index.html are out of order or nested")
    return f"{html[:i]}\n        {drawing}\n        {html[j:]}"


def main() -> None:
    html = INDEX.read_text(encoding="utf-8")
    for name, draw in (("generator", generator), ("press", timeline)):
        html = inject(html, name, draw())
    INDEX.write_text(html, encoding="utf-8")
    print(f"drew the generator and the press into {INDEX.relative_to(SITE)}")


if __name__ == "__main__":
    main()
