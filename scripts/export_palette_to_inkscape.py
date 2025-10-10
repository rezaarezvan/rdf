from pathlib import Path
from rdf.themes import ColorTheme


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def export_to_gpl(theme: ColorTheme, output_path: Path, mode: str = "light") -> None:
    """
    Export ColorTheme to GIMP/Inkscape palette (.gpl) format.

    Args:
        theme: The ColorTheme to export
        output_path: Where to save the .gpl file
        mode: "light", "dark", or "both"
    """
    lines = [
        "GIMP Palette",
        f"Name: RDF {mode.capitalize()}",
        "Columns: 4",
        "#",
    ]

    if mode in ("light", "both"):
        lines.append("# Light mode colors")
        palette = {**theme.light, **theme.base}
        for name, hex_color in sorted(palette.items()):
            r, g, b = hex_to_rgb(hex_color)
            lines.append(f"{r:3d} {g:3d} {b:3d}  {name}")

    if mode == "both":
        lines.append("#")
        lines.append("# Dark mode colors")

    if mode in ("dark", "both"):
        palette = {**theme.dark, **theme.base}
        for name, hex_color in sorted(palette.items()):
            r, g, b = hex_to_rgb(hex_color)
            suffix = f"_dark" if mode == "both" else ""
            lines.append(f"{r:3d} {g:3d} {b:3d}  {name}{suffix}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Palette exported to {output_path}")


def main():
    theme = ColorTheme.default()
    output_dir = Path("result/inkscape_palettes")

    # Export separate light and dark palettes
    export_to_gpl(theme, output_dir / "rdf_light.gpl", mode="light")
    export_to_gpl(theme, output_dir / "rdf_dark.gpl", mode="dark")
    export_to_gpl(theme, output_dir / "rdf_combined.gpl", mode="both")

    print("\nTo use in Inkscape:")
    print("1. Copy .gpl files to ~/.config/inkscape/palettes/ (Linux)")
    print("   or %APPDATA%\\inkscape\\palettes\\ (Windows)")
    print("2. Restart Inkscape")
    print("3. Access via Palette dropdown in bottom-right corner")


if __name__ == "__main__":
    main()
