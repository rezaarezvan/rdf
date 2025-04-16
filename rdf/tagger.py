from __future__ import annotations

import re
from .themes import ColorTheme


def _apply_colour_classes(svg: str, theme: ColorTheme) -> str:
    """Replace literal hex colours with `.c1 … .c8` classes."""
    for i, (_, col) in enumerate(theme.base.items(), 1):
        pattern = rf'(style="[^"]*fill: ?{col}[^"]*"|fill="{col}")'
        svg = re.sub(pattern, f'class="c{i}"', svg)
    return svg


_TAG_PATTERNS: list[tuple[str, str]] = [
    (r'<g id="([^\"]*text[^\"]*)"', r'<g id="\1" class="text"'),
    (r'<g id="matplotlib\.axis[^"]*"', r'<g class="axis"'),
    (r'<g id="legend[^"]*"', r'<g class="legend"'),
    (r'<g id="xtick_[^"]*"', r'<g class="tick xtick"'),
    (r'<g id="ytick_[^"]*"', r'<g class="tick ytick"'),
    (r'<g id="ztick_[^"]*"', r'<g class="tick ztick"'),
    (r'<g id="patch_[^"]*"', r'<g class="patch"'),
    (r'<g id="(Poly3D|Line3D)[^"]*"', r'<g class="plot3d"'),
    (r'<g id="([^\"]*grid3d[^\"]*)"', r'<g id="\1" class="plot3d-grid"'),
]


def _apply_element_tags(svg: str) -> str:
    for pat, rep in _TAG_PATTERNS:
        svg = re.sub(pat, rep, svg)

    # generic grid handler
    svg = re.sub(r'<g id="(grid_[^"]*)"', r'<g id="\1" class="grid"', svg)

    # stroke="#000" → stroke="currentColor"
    svg = re.sub(r'stroke="#000000"', 'stroke="currentColor"', svg)
    svg = re.sub(
        r'(style="[^"]*?)stroke: ?#000000([^\"]*")', r"\1stroke: currentColor\2", svg
    )

    # after stroke → currentColor lines
    svg = re.sub(r'fill="#000000"', 'fill="currentColor"', svg)
    svg = re.sub(
        r'(style="[^"]*?)fill: ?#000000([^"]*")', r"\\1fill: currentColor\\2", svg
    )
    return svg


def tag_svg(svg: str, theme: ColorTheme) -> str:
    svg = _apply_colour_classes(svg, theme)
    svg = _apply_element_tags(svg)
    return svg
