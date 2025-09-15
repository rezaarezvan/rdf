from __future__ import annotations

import re
from .themes import ColorTheme


def _apply_colour_classes(svg: str, theme: ColorTheme) -> str:
    """Replace literal hex colours with CSS classes."""
    # Handle base colors (c1-c8)
    for i, (_, col) in enumerate(theme.base.items(), 1):
        # Handle both fill and stroke, with and without style attributes
        svg = re.sub(rf'fill="{re.escape(col)}"', f'class="c{i}"', svg)
        svg = re.sub(rf'stroke="{re.escape(col)}"', f'class="c{i}"', svg)
        svg = re.sub(
            rf'(style="[^"]*?)fill: ?{re.escape(col)}([^"]*")',
            rf"\1fill: var(--c{i})\2",
            svg,
        )
        svg = re.sub(
            rf'(style="[^"]*?)stroke: ?{re.escape(col)}([^"]*")',
            rf"\1stroke: var(--c{i})\2",
            svg,
        )

    # Handle theme-specific colors (black, gray, white, etc.)
    for color_name, hex_color in theme.light.items():
        if color_name not in theme.base:  # Skip base colors (already handled)
            escaped_hex = re.escape(hex_color.upper())
            # Handle both upper and lower case hex
            for hex_variant in [hex_color, hex_color.upper(), hex_color.lower()]:
                escaped = re.escape(hex_variant)
                # Direct attribute replacements
                svg = re.sub(rf'fill="{escaped}"', f'class="{color_name}"', svg)
                svg = re.sub(rf'stroke="{escaped}"', f'class="{color_name}"', svg)
                # Style attribute replacements
                svg = re.sub(
                    rf'(style="[^"]*?)fill: ?{escaped}([^"]*")',
                    rf"\1fill: var(--{color_name})\2",
                    svg,
                )
                svg = re.sub(
                    rf'(style="[^"]*?)stroke: ?{escaped}([^"]*")',
                    rf"\1stroke: var(--{color_name})\2",
                    svg,
                )

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
    return svg


def tag_svg(svg: str, theme: ColorTheme) -> str:
    svg = _apply_colour_classes(svg, theme)
    svg = _apply_element_tags(svg)
    return svg
