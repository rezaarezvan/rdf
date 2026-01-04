from __future__ import annotations

import re
from typing import TYPE_CHECKING

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
:root {{ --axis-color:#000; --grid-stroke:rgba(0,0,0,0.2); --plot3d-stroke:rgba(0,0,0,0.3); --plot3d-grid:rgba(0,0,0,0.2); }}
@media (prefers-color-scheme:dark) {{ :root {{ --axis-color:#fff; --grid-stroke:rgba(255,255,255,0.2); --plot3d-stroke:rgba(255,255,255,0.3); --plot3d-grid:rgba(255,255,255,0.2); }} }}
{light}
@media (prefers-color-scheme:dark) {{ {dark} }}
text,.text {{ fill:var(--axis-color)!important; stroke:none!important; }}
.axis line,.axis path,.tick line,.tick path,path.domain,line.grid {{ stroke:var(--axis-color)!important; }}
.legend text {{ fill:var(--axis-color)!important; }}
.patch path[style*="stroke: currentColor"] {{ stroke:var(--axis-color)!important; }}
g.legend>g:first-child>path:first-child {{ fill:rgba(255,255,255,0.8)!important; stroke:rgba(0,0,0,0.1)!important; }}
@media (prefers-color-scheme:dark) {{ g.legend>g:first-child>path:first-child {{ fill:rgba(26,26,26,0.8)!important; stroke:rgba(255,255,255,0.1)!important; }} }}
.plot3d path,.plot3d polygon {{ stroke:var(--plot3d-stroke)!important; }}
.plot3d-grid line {{ stroke:var(--plot3d-grid)!important; }}
.plot3d-surface path[style*="fill:"] {{ fill-opacity:0.9!important; }}
@media (prefers-color-scheme:dark) {{ .plot3d-surface path[style*="fill:"] {{ fill-opacity:0.95!important; filter:saturate(1.2)!important; }} }}
""".strip()


def _cls(pal: dict[str, str]) -> str:
    return "".join(
        f".{k}{{fill:{v};stroke:{v};}}:root{{--{k}:{v};}}"
        for k, v in sorted(pal.items())
    )


def build_css(theme: ColorTheme) -> str:
    return CSS_TEMPLATE.format(
        light=_cls(theme.light | theme.base), dark=_cls(theme.dark | theme.base)
    )


def inject_css(svg: str, css: str) -> str:
    assert "<svg" in svg, "invalid SVG"
    if (idx := svg.find("<svg")) != -1 and (close := svg.find(">", idx)) != -1:
        return svg[: close + 1] + f"\n<style>{css}</style>\n" + svg[close + 1 :]
    return svg


def _apply_colors(svg: str, theme: ColorTheme) -> str:
    # Base colors c1-c8
    for i, (_, col) in enumerate(theme.base.items(), 1):
        esc = re.escape(col)
        svg = re.sub(rf'fill="{esc}"', f'class="c{i}"', svg)
        svg = re.sub(rf'stroke="{esc}"', f'class="c{i}"', svg)
        svg = re.sub(
            rf'(style="[^"]*?)fill:\s*{esc}([^"]*")', rf"\1fill:var(--c{i})\2", svg
        )
        svg = re.sub(
            rf'(style="[^"]*?)stroke:\s*{esc}([^"]*")', rf"\1stroke:var(--c{i})\2", svg
        )
    # Theme colors
    for name, hex_color in theme.light.items():
        if name not in theme.base:
            for hx in [hex_color, hex_color.upper(), hex_color.lower()]:
                esc = re.escape(hx)
                svg = re.sub(rf'fill="{esc}"', f'class="{name}"', svg)
                svg = re.sub(rf'stroke="{esc}"', f'class="{name}"', svg)
                svg = re.sub(
                    rf'(style="[^"]*?)fill:\s*{esc}([^"]*")',
                    rf"\1fill:var(--{name})\2",
                    svg,
                )
                svg = re.sub(
                    rf'(style="[^"]*?)stroke:\s*{esc}([^"]*")',
                    rf"\1stroke:var(--{name})\2",
                    svg,
                )
    return svg


def _apply_tags(svg: str) -> str:
    for pat, rep in TAG_PATTERNS:
        svg = re.sub(pat, rep, svg)
    return svg


def tag(svg: str, theme: ColorTheme) -> str:
    assert "<svg" in svg, "invalid SVG"
    return _apply_tags(_apply_colors(svg, theme))


def process(svg: str, theme: ColorTheme) -> str:
    """Full SVG processing: inject CSS + apply color classes + element tags."""
    return tag(inject_css(svg, build_css(theme)), theme)
