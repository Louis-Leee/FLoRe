#!/usr/bin/env python3
"""Draw the Controller section's diagram of the unified reference generator into index.html.

    python3 scripts/draw_controller.py

The diagram (where a hand's target goes) is inline SVG between marker comments in index.html, so
it takes the page's font and colours, scales with the column and follows the dark theme.  The
motion's reference p*_x puts the hand on the surface and the force command moves the target into
it, along x, in the notation of the equations above it.  It is drawn to scale for the press that
the page's chart runs under it (site.js, PRESS) once that press has settled at the paper's gains:
the constants below are that press's, so change them together.  Needs only the standard library.
"""
from __future__ import annotations

import re
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
INDEX = SITE / "index.html"

SUP = 12           # superscripts, in viewBox units: 1:1 with CSS px in the 900 px column
SUP_RISE = 6.5     # how far a superscript sits above the baseline
SUB_DROP = 4.5     # and a subscript below it
SUB_X = 6.2        # the width of an italic x as a subscript (.ctl .vs), for a superscript over it


# --- labels -------------------------------------------------------------------------------------
# A label is written in a small markup: *f* is an italic variable and `Δ` an upright symbol, both
# in the serif of the paper's mathematics; ^{cmd} is a superscript and _{I} a subscript, upright
# in the page's font, and _{*x*} an italic subscript in the serif.  SVG has no superscripts that
# every browser draws, so the shifts are dy offsets, undone on the next piece of text.  A
# superscript and a subscript on one letter are stacked as in TeX: the subscript is written first,
# and the superscript steps back over it.

_TOKEN = re.compile(r"\*([^*]+)\*|`([^`]+)`|\^\{([^}]*)\}|_\{\*([^}*]+)\*\}|_\{([^}]*)\}|([^*`^_]+)")


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pieces(markup: str) -> list[tuple[str, str]]:
    """The label's pieces as (kind, text), with each superscript put after its subscript."""
    kinds = ("ital", "upright", "sup", "subv", "sub", "plain")
    out = [next((k, s) for k, s in zip(kinds, groups, strict=True) if s)
           for groups in _TOKEN.findall(markup)]
    for i in range(len(out) - 1):
        if out[i][0] == "sup" and out[i + 1][0] in ("sub", "subv"):
            out[i], out[i + 1] = out[i + 1], ("stack", out[i][1])
    return out


def rich(markup: str) -> str:
    out, shift, after_italic = [], 0.0, False
    for kind, piece in pieces(markup):
        want = -SUP_RISE if kind in ("sup", "stack") else (SUB_DROP if kind in ("sub", "subv") else 0.0)
        attrs = []
        if kind == "stack":
            attrs.append(f'dx="{3 - SUB_X:g}"')     # back over the subscript, clear of the letter's top
        elif kind == "sup" and after_italic:
            attrs.append('dx="3"')                   # clear of the slanted letter's top
        if want != shift:
            attrs.append(f'dy="{want - shift:g}"')
            shift = want
        cls = {"ital": "v", "upright": "m", "subv": "vs"}.get(kind)
        if cls:
            attrs.append(f'class="{cls}"')
        if kind in ("sup", "stack", "sub"):
            attrs.append(f'font-size="{SUP}"')
        out.append(f"<tspan {' '.join(attrs)}>{esc(piece)}</tspan>" if attrs else esc(piece))
        after_italic = kind == "ital" or (after_italic and kind in ("sub", "subv"))
    return "".join(out)


def text(x: float, y: float, markup: str, cls: str = "", anchor: str = "start") -> str:
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    c = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.1f}" y="{y:.1f}"{c}{a}>{rich(markup)}</text>'


def chip(x0: float, x1: float, y: float) -> str:
    """A patch of the page's background under a label that sits on the surface's hatching."""
    return f'<rect class="chip" x="{x0:.1f}" y="{y - 18:.1f}" width="{x1 - x0:.1f}" height="27" rx="5"/>'


def arrow(x1: float, y1: float, x2: float, y2: float, cls: str) -> str:
    return (f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'marker-end="url(#ctl-{cls})"/>')


def guide(x: float, y1: float, y2: float) -> str:
    return f'<line class="guide" x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}"/>'


def defs(*classes: str, hatch: bool = False) -> str:
    """Arrowheads, one per colour (a marker cannot take its colour from the line that uses it),
    and the surface's hatching.  The ids are the page's, so each is defined once."""
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


# --- the unified reference generator ------------------------------------------------------------

CM = 32.0                                # px per cm along the surface normal
F_CMD, K, K_CONTACT = 15.0, 200.0, 150.0     # N, N/m, N/m: the chart's press, its contact softer than K
I_SETTLED = F_CMD * (K / K_CONTACT - 1)      # its integral once the measured force balances the command


def generator() -> str:
    y = 150.0                                        # the line through the hand and its target
    x_face = 200.0                                   # the surface, where the reference puts the hand
    x_ff = x_face + F_CMD / K * 100 * CM             # p* + f_cmd/K
    x_tgt = x_ff + I_SETTLED / K * 100 * CM          # p* + (f_cmd + I)/K
    wall_r, y_dim, y_force = x_tgt + 86, 214.0, 84.0
    p = [defs("cmd", "meas", "integ", hatch=True)]
    # the surface, with its name above it
    p.append(f'<rect class="wall" x="{x_face:.1f}" y="44" width="{wall_r - x_face:.1f}" height="212" '
             'fill="url(#ctl-hatch)"/>')
    p.append(f'<line class="face" x1="{x_face:.1f}" y1="44" x2="{x_face:.1f}" y2="256"/>')
    p.append(text((x_face + wall_r) / 2, 32, "surface", "gray", "middle"))
    for x in (x_ff, x_tgt):                          # down to the shift, which starts at the face
        p.append(guide(x, y + 15, y_dim + 8))
    # the hand: a gripper holding a pad whose face, at the TCP, is on the surface
    t = x_face
    p.append(
        f'<line class="hand" x1="{t - 62:.1f}" y1="{y:.1f}" x2="{t - 96:.1f}" y2="{y - 44:.1f}"/>'
        f'<rect class="hand" x="{t - 66:.1f}" y="{y - 31:.1f}" width="18" height="62" rx="4"/>'
        f'<rect class="hand" x="{t - 50:.1f}" y="{y - 31:.1f}" width="34" height="10" rx="3"/>'
        f'<rect class="hand" x="{t - 50:.1f}" y="{y + 21:.1f}" width="34" height="10" rx="3"/>'
        f'<rect class="pad" x="{t - 26:.1f}" y="{y - 19:.1f}" width="26" height="38" rx="2"/>'
    )
    p.append(text(t - 36, y + 56, "hand", "gray note", "middle"))
    # p*: the motion's reference, a ring around the TCP, both on the surface
    p.append(f'<circle class="ref" cx="{t:.1f}" cy="{y:.1f}" r="10"/>')
    p.append(f'<circle class="tcp" cx="{t:.1f}" cy="{y:.1f}" r="4.5"/>')
    p.append(chip(t + 14, t + 44, y + 6))
    p.append(text(t + 17, y + 6, "*p*^{*}_{*x*}"))
    # the forces at the TCP, drawn above it: the command into the surface, the reading out of it
    p.append(guide(t, y_force, y - 36))
    p.append(arrow(t, y_force, t + 92, y_force, "cmd"))
    p.append(arrow(t, y_force, t - 92, y_force, "meas"))
    p.append(chip(t + 100, t + 152, y_force + 6))
    p.append(text(t + 104, y_force + 6, "*f*^{cmd}_{*x*}", "cmd"))
    p.append(text(t - 102, y_force + 6, "*f*^{meas}_{*x*}", "meas", "end"))
    # p_tgt: the target the policy tracks, inside the surface
    s = 9.0
    p.append(f'<path class="tgt" d="M{x_tgt - s:.1f},{y - s:.1f} L{x_tgt + s:.1f},{y + s:.1f} '
             f'M{x_tgt - s:.1f},{y + s:.1f} L{x_tgt + s:.1f},{y - s:.1f}"/>')
    p.append(chip(x_tgt + 16, x_tgt + 66, y + 6))
    p.append(text(x_tgt + 20, y + 6, "*p*^{tgt}_{*x*}"))
    # the shift, as two arrows end to end from the face: the command's part and the integral's
    p.append(arrow(x_face, y_dim, x_ff, y_dim, "cmd"))
    p.append(arrow(x_ff, y_dim, x_tgt, y_dim, "integ"))
    for x0, x1, half, label, cls in ((x_face, x_ff, 37, "*f*^{cmd}_{*x*} / *K*", "cmd"),
                                     (x_ff, x_tgt, 31, "*I*_{*x*} / *K*", "integ")):
        p.append(chip((x0 + x1) / 2 - half, (x0 + x1) / 2 + half, y_dim + 25))
        p.append(text((x0 + x1) / 2, y_dim + 25, label, cls, "middle"))
    # the law, to the right
    xl = wall_r + 36
    p.append(text(xl, 138, "*p*^{tgt}_{*x*} = *p*^{*}_{*x*} + `Δ`*x*"))
    p.append(text(xl, 172, "`Δ`*x* = ( *f*^{cmd}_{*x*} + *I*_{*x*} ) / *K*"))
    return svg(270, "A hand rests on a surface, at the motion's reference p-star x. Its target is moved "
               "from p-star x into the surface, by f-cmd x over K and then by the integral I x over K; "
               "the surface stops the hand at p-star x, so it presses. The command f-cmd x points into "
               "the surface and the measured force f-meas x out of it. p-tgt x equals p-star x plus "
               "delta x, and delta x equals f-cmd x plus I x, over K.", p)


# --- into the page ------------------------------------------------------------------------------

def inject(html: str, name: str, drawing: str) -> str:
    """Put the drawing between the diagram's markers, and refuse unless there is exactly one pair of
    them, in order, with no other diagram's marker in between: the text around them is written by
    hand and must not be overwritten."""
    begin, end = f"<!-- begin draw_controller.py: {name} -->", f"<!-- end draw_controller.py: {name} -->"
    if html.count(begin) != 1 or html.count(end) != 1:
        raise SystemExit(f"index.html needs one begin and one end marker for the {name} diagram")
    i, j = html.index(begin) + len(begin), html.index(end)
    if j < i or "draw_controller.py:" in html[i:j]:
        raise SystemExit(f"the {name} diagram's markers in index.html are out of order or nested")
    return f"{html[:i]}\n        {drawing}\n        {html[j:]}"


def main() -> None:
    html = INDEX.read_text(encoding="utf-8")
    if "draw_controller.py: press" in html:
        raise SystemExit("index.html still has the press diagram's markers; the chart under the "
                         "generator replaces it")
    INDEX.write_text(inject(html, "generator", generator()), encoding="utf-8")
    print(f"drew the generator into {INDEX.relative_to(SITE)}")


if __name__ == "__main__":
    main()
