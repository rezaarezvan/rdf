from __future__ import annotations


def inject_css(svg: str, css_block: str) -> str:
    idx = svg.find("<svg")
    if idx == -1:
        return svg
    close = svg.find(">", idx)
    if close == -1:
        return svg
    return svg[: close + 1] + "\n" + css_block + "\n" + svg[close + 1 :]
