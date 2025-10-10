import re
import sys

from pathlib import Path
from rdf.themes import ColorTheme
from rdf.css import build_style


def apply_color_classes(svg: str, theme: ColorTheme) -> str:
    """Replace hex colors with RDF CSS classes and variables."""

    # Combine all palettes for lookup
    all_colors = {**theme.light, **theme.dark, **theme.base}

    # Build reverse lookup: hex -> name (prefer light mode as default)
    hex_to_name = {}
    for name, hex_val in theme.base.items():
        hex_to_name[hex_val.upper()] = name
        hex_to_name[hex_val.lower()] = name
    for name, hex_val in theme.light.items():
        hex_to_name[hex_val.upper()] = name
        hex_to_name[hex_val.lower()] = name

    # Sort by length (longest first) to avoid partial matches
    sorted_hex = sorted(hex_to_name.keys(), key=len, reverse=True)

    for hex_color in sorted_hex:
        class_name = hex_to_name[hex_color]
        escaped = re.escape(hex_color)

        # Replace fill="..." with class="..."
        svg = re.sub(
            rf'fill="{escaped}"',
            f'fill="var(--{class_name})" class="{class_name}"',
            svg,
        )

        # Replace stroke="..." with class="..."
        svg = re.sub(
            rf'stroke="{escaped}"',
            f'stroke="var(--{class_name})" class="{class_name}"',
            svg,
        )

        # Replace style attribute colors
        svg = re.sub(
            rf'(style="[^"]*?)fill:\s*{escaped}([^"]*")',
            rf"\1fill:var(--{class_name})\2",
            svg,
        )
        svg = re.sub(
            rf'(style="[^"]*?)stroke:\s*{escaped}([^"]*")',
            rf"\1stroke:var(--{class_name})\2",
            svg,
        )

    return svg


def apply_element_tags(svg: str) -> str:
    """Apply semantic class tags to common SVG elements."""

    # Tag text elements
    svg = re.sub(r"<text\b", r'<text class="text"', svg)

    # Tag groups with ID hints
    svg = re.sub(r'<g id="([^"]*legend[^"]*)"', r'<g id="\1" class="legend"', svg)
    svg = re.sub(r'<g id="([^"]*axis[^"]*)"', r'<g id="\1" class="axis"', svg)
    svg = re.sub(r'<g id="([^"]*grid[^"]*)"', r'<g id="\1" class="grid"', svg)

    return svg


def inject_css_style(svg: str, css: str) -> str:
    """Inject CSS into SVG."""
    style_tag = f"<style>{css}</style>"

    # Try to inject after <defs> if it exists
    if "<defs>" in svg:
        svg = svg.replace("</defs>", f"{style_tag}\n</defs>")
    # Otherwise inject after opening <svg> tag
    else:
        svg = re.sub(r"(<svg[^>]*>)", rf"\1\n{style_tag}", svg, count=1)

    return svg


def clean_inkscape_metadata(svg: str, keep_ids: bool = False) -> str:
    """Remove Inkscape-specific metadata and namespaces."""

    # Remove Inkscape/Sodipodi elements first (before removing namespaces)
    svg = re.sub(
        r"<sodipodi:namedview[^>]*>.*?</sodipodi:namedview>", "", svg, flags=re.DOTALL
    )
    svg = re.sub(r"<metadata[^>]*>.*?</metadata>", "", svg, flags=re.DOTALL)
    svg = re.sub(r"<inkscape:path-effect[^/>]*/?>", "", svg)
    svg = re.sub(r"<inkscape:perspective[^/>]*/?>", "", svg)
    svg = re.sub(r"<sodipodi:[^>]+/?>", "", svg)

    # Remove Inkscape namespace declarations
    svg = re.sub(r'\s+xmlns:inkscape="[^"]*"', "", svg)
    svg = re.sub(r'\s+xmlns:sodipodi="[^"]*"', "", svg)

    # Remove Inkscape/Sodipodi attributes
    svg = re.sub(r'\s+inkscape:[^=]*="[^"]*"', "", svg)
    svg = re.sub(r'\s+sodipodi:[^=]*="[^"]*"', "", svg)

    if not keep_ids:
        # Optionally remove auto-generated IDs (can make diffs cleaner)
        svg = re.sub(r'\s+id="(path|rect|circle|ellipse|g)\d+"', "", svg)

    return svg


def convert_inkscape_to_rdf(
    input_path: Path,
    output_path: Path | None = None,
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
    theme = theme or ColorTheme.default()
    svg = input_path.read_text(encoding="utf-8")

    # Apply transformations
    svg = apply_color_classes(svg, theme)
    svg = apply_element_tags(svg)

    if clean_metadata:
        svg = clean_inkscape_metadata(svg, keep_ids=keep_ids)

    # Inject RDF CSS
    css = build_style(theme)
    svg = inject_css_style(svg, css)

    # Save output
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_rdf.svg"

    output_path.write_text(svg, encoding="utf-8")
    print(f"✓ Converted: {input_path.name} → {output_path.name}")

    return svg


def main():
    if len(sys.argv) < 2:
        print("Usage: python inkscape_to_rdf.py <input.svg> [output.svg]")
        print("\nConverts Inkscape SVG to RDF-themed SVG with CSS classes")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    convert_inkscape_to_rdf(input_path, output_path)


if __name__ == "__main__":
    main()
