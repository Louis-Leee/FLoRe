#!/usr/bin/env python3
"""Typeset the page's equations with KaTeX, once, into ``index.html``.

Each formula's TeX sits in a comment, and its rendering after it, up to an end marker:

    <!--tex:f^{\\mathrm{cmd}}_x-->...rendering...<!--/tex-->        inline
    <!--texd:p^{\\mathrm{tgt}}_x = ...-->...rendering...<!--/tex-->  display

The script renders every one of them again from its TeX, so the TeX is the source to edit; the
page then needs only KaTeX's stylesheet and fonts (``static/katex``, from the same release), and
no script. The macros below colour the forces and the integral as the controller's diagrams do,
through classes that ``style.css`` sets for either theme. It refuses a marker without its end,
and TeX holding "--", which would end its comment.

    python3 scripts/render_math.py          # needs Node (npx fetches KaTeX the first time)
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

log = logging.getLogger("render_math")

SITE = Path(__file__).resolve().parent.parent
INDEX = SITE / "index.html"
KATEX = "katex@0.19.0"  # static/katex holds this release's stylesheet and fonts
MACROS = {
    # a force: its f and its name in the colour of its arrow; a subscript stays plain
    r"\fcmd": r"\htmlClass{m-cmd}{f}^{\htmlClass{m-cmd}{\mathrm{cmd}}}",
    r"\fmeas": r"\htmlClass{m-meas}{f}^{\htmlClass{m-meas}{\mathrm{meas}}}",
    r"\fext": r"\htmlClass{m-ext}{f}^{\htmlClass{m-ext}{\mathrm{ext}}}",
    r"\fhext": r"\htmlClass{m-ext}{\hat f}^{\htmlClass{m-ext}{\mathrm{ext}}}",
    r"\Ii": r"\htmlClass{m-integ}{I}",  # the integral, green as in the diagrams
    r"\ptgt": r"p^{\mathrm{tgt}}",
}
MARKER = re.compile(r"<!--(tex|texd):(.*?)-->(.*?)<!--/tex-->", re.DOTALL)


def render(tex: str, display: bool) -> str:
    cmd = ["npx", "--yes", KATEX, "--trust"]  # trust: \htmlClass, for the colours
    for name, body in MACROS.items():
        cmd += ["-m", f"{name}:{body}"]
    if display:
        cmd.append("--display-mode")
    out = subprocess.run(cmd, input=tex, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise SystemExit(f"KaTeX failed on {tex!r}:\n{out.stderr.strip()}")
    return out.stdout.strip()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    html = INDEX.read_text()
    opened = len(re.findall(r"<!--texd?:", html))
    closed = html.count("<!--/tex-->")
    found = MARKER.findall(html)
    if not opened == closed == len(found):
        raise SystemExit(f"{INDEX.name}: {opened} formulas open, {closed} end markers, {len(found)} pairs")
    cache: dict[tuple[str, bool], str] = {}

    def typeset(m: re.Match[str]) -> str:
        kind, tex = m.group(1), m.group(2)
        if "--" in tex:
            raise SystemExit(f"TeX may not hold '--', which would end its comment: {tex!r}")
        key = (tex, kind == "texd")
        if key not in cache:
            cache[key] = render(*key)
        return f"<!--{kind}:{tex}-->{cache[key]}<!--/tex-->"

    new = MARKER.sub(typeset, html)
    if new != html:
        INDEX.write_text(new)
    log.info("%d formulas (%d distinct) typeset%s", len(found), len(cache), "" if new != html else ", unchanged")


if __name__ == "__main__":
    main()
