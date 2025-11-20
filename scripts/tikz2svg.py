from __future__ import annotations

import os
import subprocess
import tempfile

from pathlib import Path
from typing import Optional

from rdf.themes import ColorTheme
from rdf.css import build_style
from rdf import svg_tools


class TikZ2SVG:
    """Convert TikZ diagrams to themed SVG files."""

    def __init__(
        self,
        theme: Optional[ColorTheme] = None,
        *,
        template_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize the TikZ to SVG converter.

        Args:
            theme: Color theme to apply (defaults to RDF theme)
            template_path: Path to custom LaTeX template (optional)
        """
        self.theme = theme or ColorTheme.default()
        self.template_path = template_path or self._default_template_path()

    def _default_template_path(self) -> Path:
        """Get the default template path."""
        script_dir = Path(__file__).resolve().parent
        template = script_dir / "tikz-template.tex"
        if not template.exists():
            template = Path("misc/tikz-template.tex")
        return template

    def _read_template(self) -> str:
        """Read the LaTeX template file."""
        if not self.template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {self.template_path}\n"
                f"Please ensure tikz-template.tex exists in misc/"
            )
        return self.template_path.read_text(encoding="utf-8")

    def _compile_to_pdf(self, tex_file: Path) -> Optional[Path]:
        """
        Compile LaTeX to PDF using pdflatex.

        Args:
            tex_file: Path to .tex file

        Returns:
            Path to generated PDF, or None if compilation failed
        """
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-shell-escape",
                "-halt-on-error",
                tex_file.name,
            ],
            cwd=tex_file.parent,
            env={
                **os.environ,
                "TEXINPUTS": f"{tex_file.parent}:",
                "max_print_line": "1000",
                "synctex": "1",
                "pdfcompresslevel": "0",
                "pdfobjcompresslevel": "0",
            },
            capture_output=True,
            text=True,
        )

        pdf_file = tex_file.with_suffix(".pdf")
        if result.returncode != 0 or not pdf_file.exists():
            print("\n=== LaTeX Compilation Error ===")
            print(result.stdout)
            if result.stderr:
                print("\nStderr:")
                print(result.stderr)
            return None

        return pdf_file

    def _convert_to_svg(self, pdf_file: Path) -> Optional[Path]:
        """
        Convert PDF to SVG using pdf2svg.

        Args:
            pdf_file: Path to PDF file

        Returns:
            Path to generated SVG, or None if conversion failed
        """
        svg_file = pdf_file.with_suffix(".svg")
        result = subprocess.run(
            ["pdf2svg", str(pdf_file), str(svg_file)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0 or not svg_file.exists():
            print("\n=== PDF to SVG Conversion Error ===")
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return None

        return svg_file

    def _apply_theming(self, svg_content: str, output_file: Path) -> str:
        """
        Apply RDF theme and dark mode support to SVG.

        Args:
            svg_content: Raw SVG content
            output_file: Output file path (used for unique ID prefixes)

        Returns:
            Themed SVG content
        """
        # Inject CSS theme
        css = build_style(self.theme)
        svg_content = svg_tools.inject_css(svg_content, f"<style>{css}</style>")

        # Add unique prefixes to prevent ID conflicts
        prefix = output_file.stem.replace(".", "_").replace("-", "_")
        svg_content = svg_content.replace('id="', f'id="{prefix}_')
        svg_content = svg_content.replace('href="#', f'href="#{prefix}_')
        svg_content = svg_content.replace("url(#", f"url(#{prefix}_")

        # Replace black colors with theme colors for dark mode compatibility
        replacements = [
            ('stroke="rgb(0%, 0%, 0%)"', 'stroke="var(--black)"'),
            ('stroke="black"', 'stroke="var(--black)"'),
            ('fill="rgb(0%, 0%, 0%)"', 'fill="var(--black)"'),
            ('fill="black"', 'fill="var(--black)"'),
            ("<g>", '<g class="glyph">'),
            ('text-anchor="middle"', 'text-anchor="middle" class="math"'),
            ('text-anchor="start"', 'text-anchor="start" class="math"'),
        ]

        for old, new in replacements:
            svg_content = svg_content.replace(old, new)

        return svg_content

    def convert(
        self, tikz_code: str, output_file: Path, *, cleanup: bool = True
    ) -> bool:
        """
        Convert TikZ code to themed SVG.

        Args:
            tikz_code: TikZ code to convert
            output_file: Path where SVG will be saved
            cleanup: Whether to remove temporary files

        Returns:
            True if conversion succeeded, False otherwise
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create LaTeX file
            template = self._read_template()
            tex_content = template.replace("% TIKZ_CONTENT_HERE", tikz_code)
            tex_file = tmp_path / "temp.tex"
            tex_file.write_text(tex_content, encoding="utf-8")

            # Compile to PDF
            pdf_file = self._compile_to_pdf(tex_file)
            if not pdf_file:
                return False

            # Convert to SVG
            svg_file = self._convert_to_svg(pdf_file)
            if not svg_file:
                return False

            # Apply theming
            svg_content = svg_file.read_text(encoding="utf-8")
            themed_svg = self._apply_theming(svg_content, output_file)

            # Save final output
            output_file.write_text(themed_svg, encoding="utf-8")
            print(f"✓ SVG saved to {output_file}")

        return True

    def convert_file(
        self, input_file: Path, output_file: Optional[Path] = None
    ) -> bool:
        """
        Convert TikZ file to themed SVG.

        Args:
            input_file: Path to .tikz file
            output_file: Output path (defaults to input with .svg extension)

        Returns:
            True if conversion succeeded, False otherwise
        """
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if output_file is None:
            output_file = input_file.with_suffix(".svg")

        tikz_code = input_file.read_text(encoding="utf-8")
        return self.convert(tikz_code, output_file)


def main() -> None:
    """CLI entry point."""
    import sys

    if len(sys.argv) != 3:
        print("Usage: python tikz2svg.py input.tikz output.svg")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    converter = TikZ2SVG()
    success = converter.convert_file(input_file, output_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
