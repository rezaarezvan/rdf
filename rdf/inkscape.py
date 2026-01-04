from __future__ import annotations

import re

from pathlib import Path

from rdf.theme import ColorTheme
from rdf.svg import build_css, inject_css

# Inkscape/Sodipodi namespace patterns
INKSCAPE_PATTERNS = [
    (r"<sodipodi:namedview[^>]*>.*?</sodipodi:namedview>", ""),
    (r"<metadata[^>]*>.*?</metadata>", ""),
    (r"<inkscape:path-effect[^/>]*/?>", ""),
    (r"<inkscape:perspective[^/>]*/?>", ""),
    (r"<sodipodi:[^>]+/?>", ""),
    (r'\s+xmlns:inkscape="[^"]*"', ""),
    (r'\s+xmlns:sodipodi="[^"]*"', ""),
    (r'\s+inkscape:[^=]*="[^"]*"', ""),
    (r'\s+sodipodi:[^=]*="[^"]*"', ""),
]


def clean(svg: str, *, keep_ids: bool = False) -> str:
    """Remove Inkscape-specific metadata and namespaces."""
    for pat, rep in INKSCAPE_PATTERNS:
        svg = re.sub(pat, rep, svg, flags=re.DOTALL)
    if not keep_ids:
        svg = re.sub(r'\s+id="(path|rect|circle|ellipse|g)\d+"', "", svg)
    return svg


def _apply_colors(svg: str, theme: ColorTheme) -> str:
    """Replace hex colors with RDF CSS classes and variables."""
    hex_to_name = {}
    for name, hx in theme.base.items():
        hex_to_name[hx.upper()] = hex_to_name[hx.lower()] = name
    for name, hx in theme.light.items():
        hex_to_name[hx.upper()] = hex_to_name[hx.lower()] = name
    for hx in sorted(hex_to_name.keys(), key=len, reverse=True):
        name, esc = hex_to_name[hx], re.escape(hx)
        svg = re.sub(rf'fill="{esc}"', f'fill="var(--{name})" class="{name}"', svg)
        svg = re.sub(rf'stroke="{esc}"', f'stroke="var(--{name})" class="{name}"', svg)
        svg = re.sub(
            rf'(style="[^"]*?)fill:\s*{esc}([^"]*")', rf"\1fill:var(--{name})\2", svg
        )
        svg = re.sub(
            rf'(style="[^"]*?)stroke:\s*{esc}([^"]*")',
            rf"\1stroke:var(--{name})\2",
            svg,
        )
    return svg


def _apply_tags(svg: str) -> str:
    """Apply semantic class tags to common SVG elements."""
    svg = re.sub(r"<text\b", r'<text class="text"', svg)
    svg = re.sub(r'<g id="([^"]*legend[^"]*)"', r'<g id="\1" class="legend"', svg)
    svg = re.sub(r'<g id="([^"]*axis[^"]*)"', r'<g id="\1" class="axis"', svg)
    svg = re.sub(r'<g id="([^"]*grid[^"]*)"', r'<g id="\1" class="grid"', svg)
    return svg


def convert(
    input_path: Path,
    output_path: Path | None = None,
    *,
    theme: ColorTheme | None = None,
    clean_metadata: bool = True,
    keep_ids: bool = False,
) -> str:
    """
    Convert an Inkscape SVG to RDF format.

    Args:
      input_path: Path to input SVG
      output_path: Path for output (default: input_rdf.svg)
      theme: ColorTheme to use (default: ColorTheme.default())
      clean_metadata: Remove Inkscape metadata
      keep_ids: Keep element IDs when cleaning

    Returns:
      The converted SVG string
    """
    assert input_path.exists(), f"file not found: {input_path}"
    theme = theme or ColorTheme.default()
    svg = input_path.read_text("utf-8")
    svg = _apply_colors(svg, theme)
    svg = _apply_tags(svg)
    if clean_metadata:
        svg = clean(svg, keep_ids=keep_ids)
    svg = inject_css(svg, build_css(theme))
    out = output_path or input_path.parent / f"{input_path.stem}_rdf.svg"
    out.write_text(svg, "utf-8")
    print(f"converted: {input_path.name} -> {out.name}")
    return svg


def convert_batch(
    paths: list[Path], *, theme: ColorTheme | None = None, **kw
) -> list[str]:
    """Convert multiple Inkscape SVGs to RDF format."""
    return [convert(p, theme=theme, **kw) for p in paths]
