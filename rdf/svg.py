import re
from typing import TYPE_CHECKING

from rdf.theme import AXIS_SENTINEL, BLACK_SENTINEL, GRID_SENTINEL

if TYPE_CHECKING:
    from rdf.theme import ColorTheme

# Tag patterns for matplotlib SVG elements
TAG_PATTERNS = [
    (r'<g id="([^\"]*text[^\"]*)"', r'<g id="\1" class="text"'),
    (r'<g id="matplotlib\.axis[^"]*"', r'<g class="axis"'),
    (r'<g id="legend[^"]*"', r'<g class="legend"'),
    (r'<g id="xtick_[^"]*"', r'<g class="tick xtick"'),
    (r'<g id="ytick_[^"]*"', r'<g class="tick ytick"'),
    (r'<g id="ztick_[^"]*"', r'<g class="tick ztick"'),
    (r'<g id="patch_[^"]*"', r'<g class="patch"'),
    (r'<g id="(Poly3D|Line3D)[^"]*"', r'<g class="plot3d"'),
    (r'<g id="([^\"]*grid3d[^\"]*)"', r'<g id="\1" class="plot3d-grid"'),
    (r'<g id="(grid_[^"]*)"', r'<g id="\1" class="grid"'),
]

CSS_TEMPLATE = """
:root {{ --axis-color:{axis_light}; --grid-color:{grid_light}; {light} }}
@media (prefers-color-scheme:dark) {{
  :root {{ --axis-color:{axis_dark}; --grid-color:{grid_dark}; {dark} }}
}}
.plot3d path,.plot3d polygon {{ stroke:var(--axis-color); }}
.plot3d-grid line {{ stroke:var(--grid-color); }}
""".strip()


def _vars(pal: dict[str, str]) -> str:
    return "".join(f"--{k}:{v};" for k, v in sorted(pal.items()))


def build_css(theme: "ColorTheme") -> str:
    # Theme wins over base, so c1..c8 bake in as real colors per mode.
    return CSS_TEMPLATE.format(
        light=_vars(theme.base | theme.light),
        dark=_vars(theme.base | theme.dark),
        axis_light=theme.axis[0],
        axis_dark=theme.axis[1],
        grid_light=theme.grid[0],
        grid_dark=theme.grid[1],
    )


def inject_css(svg: str, css: str) -> str:
    assert "<svg" in svg, "invalid SVG"
    if (idx := svg.find("<svg")) != -1 and (close := svg.find(">", idx)) != -1:
        return svg[: close + 1] + f"\n<style>{css}</style>\n" + svg[close + 1 :]
    return svg


def _to_var(svg: str, hex_color: str, var_name: str) -> str:
    """Point every inline fill/stroke at `hex_color` to var(--var_name)."""
    esc = re.escape(hex_color)
    for prop in ("fill", "stroke"):
        svg = re.sub(
            rf'(style="[^"]*?){prop}:\s*{esc}([^"]*")',
            rf"\1{prop}:var(--{var_name})\2",
            svg,
            flags=re.IGNORECASE,
        )
    return svg


def _apply_colors(svg: str, theme: "ColorTheme") -> str:
    svg = _to_var(svg, AXIS_SENTINEL, "axis-color")
    svg = _to_var(svg, GRID_SENTINEL, "grid-color")
    svg = _to_var(svg, BLACK_SENTINEL, "black")
    for i, col in enumerate(theme.base.values(), 1):
        svg = _to_var(svg, col, f"c{i}")
    for name, hex_color in theme.light.items():
        if name not in theme.base:
            svg = _to_var(svg, hex_color, name)
    return svg


def _apply_tags(svg: str) -> str:
    for pat, rep in TAG_PATTERNS:
        svg = re.sub(pat, rep, svg)
    return svg


def tag(svg: str, theme: "ColorTheme") -> str:
    assert "<svg" in svg, "invalid SVG"
    return _apply_tags(_apply_colors(svg, theme))



_PATH_D = re.compile(r'(?<= d=")[^"]*(?=")')
_NUM = re.compile(r"-?\d+\.\d+")


def _round(m: re.Match[str]) -> str:
    return f"{float(m.group()):.2f}".rstrip("0").rstrip(".")


def minify(svg: str) -> str:
    """Round path coordinates to 0.01pt (sub-pixel); ~20% smaller gzipped."""
    return _PATH_D.sub(lambda m: _NUM.sub(_round, m.group()), svg)


def zoom(svg: str, k: float) -> str:
    """Declare the root <svg> k times larger than drawn, in pt."""
    end = svg.index(">", svg.index("<svg"))
    size = lambda m: f'{m[1]}="{float(m[2]) * k:g}pt"'
    return re.sub(r'\b(width|height)="([\d.]+)(?:pt)?"', size, svg[:end]) + svg[end:]


def process(svg: str, theme: "ColorTheme") -> str:
    """Full SVG processing: inject CSS + apply color classes + element tags."""
    return tag(inject_css(minify(svg), build_css(theme)), theme)
