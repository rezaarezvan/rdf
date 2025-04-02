import sys
import os
import re
import subprocess

from pathlib import Path


def process_tikz_to_svg(tikz_code: str, output_file: str):
    """Convert TikZ code to SVG with dark mode support."""
    try:
        # Get script directory for template path
        script_dir = Path(__file__).resolve().parent
        template_path = script_dir / "tikz-template.tex"

        # If template doesn't exist in script directory, try current directory
        if not template_path.exists():
            template_path = Path("tikz-template.tex")

        # Read template
        with open(template_path, "r") as f:
            template = f.read()
            print("Successfully read template")

        # Replace content
        tex_content = template.replace("% TIKZ_CONTENT_HERE", tikz_code)

        # Write temporary tex file
        with open("temp.tex", "w") as f:
            f.write(tex_content)
            print("Successfully wrote temp.tex")

        # Convert to PDF using pdflatex with high-quality settings
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-shell-escape",  # Allow external commands
                "-halt-on-error",  # Stop on first error
                "\\synctex=1",    # Enable better positioning
                "\\pdfcompresslevel=0",  # No compression for better quality
                "\\pdfobjcompresslevel=0",
                "temp.tex"
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("\nLaTeX Error Details:")
            print(result.stdout)
            print("\nError message:")
            print(result.stderr)
            return

        # Check if PDF was generated
        if not os.path.exists("temp.pdf"):
            print("PDF was not generated!")
            return
        print("Successfully generated PDF")

        # Convert PDF to SVG
        subprocess.run(["pdf2svg", "temp.pdf", "temp.svg"])
        if not os.path.exists("temp.svg"):
            print("SVG was not generated!")
            return
        print("Successfully generated SVG")

        # Read and process SVG content
        with open("temp.svg", "r") as f:
            svg_content = f.read()

        # Add light/dark mode support with explicit colors and media queries
        # This follows the approach used in rdp.py
        style_tag = """
        <style>
            /* Light mode (default) */
            .c1 { fill: #d62728 !important; stroke: #d62728 !important; }
            .c2 { fill: #2ca02c !important; stroke: #2ca02c !important; }
            .c3 { fill: #1f77b4 !important; stroke: #1f77b4 !important; }
            .c4 { fill: #9467bd !important; stroke: #9467bd !important; }
            .c5 { fill: #8c564b !important; stroke: #8c564b !important; }
            .c6 { fill: #e377c2 !important; stroke: #e377c2 !important; }
            .c7 { fill: #7f7f7f !important; stroke: #7f7f7f !important; }
            .c8 { fill: #bcbd22 !important; stroke: #bcbd22 !important; }

            /* Text and graphic elements */
            text, .text {
                fill: #000000 !important;
                stroke: none !important;
            }

            .axis line, .axis path, .tick line, .tick path, path.domain, line.grid {
                stroke: #000000 !important;
            }

            .glyph {
                fill: #000000 !important;
            }

            path {
                stroke: #000000 !important;
            }

            /* Dark mode via media query */
            @media (prefers-color-scheme: dark) {
                /* Dark theme colors */
                .c1 { fill: #FF4A98 !important; stroke: #FF4A98 !important; }
                .c2 { fill: #0AFAFA !important; stroke: #0AFAFA !important; }
                .c3 { fill: #7F83FF !important; stroke: #7F83FF !important; }
                .c4 { fill: #B4A0FF !important; stroke: #B4A0FF !important; }
                .c5 { fill: #FFB86B !important; stroke: #FFB86B !important; }
                .c6 { fill: #FF79C6 !important; stroke: #FF79C6 !important; }
                .c7 { fill: #CCCCCC !important; stroke: #CCCCCC !important; }
                .c8 { fill: #E6DB74 !important; stroke: #E6DB74 !important; }

                /* Dark theme text and graphics */
                text, .text {
                    fill: #FFFFFF !important;
                    stroke: none !important;
                }

                .axis line, .axis path, .tick line, .tick path, path.domain, line.grid {
                    stroke: #FFFFFF !important;
                }

                .glyph {
                    fill: #FFFFFF !important;
                }

                path {
                    stroke: #FFFFFF !important;
                }
            }
        </style>
        """
        # First add the class attribute
        svg_content = svg_content.replace("<svg", '<svg class="math"')

        # Then insert the style tag after the opening svg tag and all its attributes
        svg_pattern = r'(<svg[^>]*>)'
        svg_content = re.sub(svg_pattern, r'\1' + style_tag, svg_content)

        # Add unique prefixes to prevent ID conflicts when multiple SVGs are on the same page
        filename = os.path.basename(output_file)
        prefix = filename.replace(".svg", "").replace(
            ".", "_").replace("-", "_")
        svg_content = svg_content.replace('id="', f'id="{prefix}_')
        svg_content = svg_content.replace('href="#', f'href="#{prefix}_')
        svg_content = svg_content.replace('url(#', f'url(#{prefix}_')

        # Replace colors with currentColor for dark mode support
        replacements = [
            ('stroke="rgb(0%, 0%, 0%)"', 'stroke="currentColor"'),
            ('stroke="black"', 'stroke="currentColor"'),
            ('style="color: black"', 'style="color: currentColor"'),
            ('fill="rgb(0%, 0%, 0%)"', 'fill="currentColor"'),
            ('fill="black"', 'fill="currentColor"'),
            ("<g>", '<g class="glyph">'),
            ('fill-opacity="1"', 'fill-opacity="1" class="glyph"'),
            # Add math class to text elements
            ('text-anchor="middle"', 'text-anchor="middle" class="math"'),
            ('text-anchor="start"', 'text-anchor="start" class="math"'),
        ]
        for old, new in replacements:
            svg_content = svg_content.replace(old, new)

        # Save final SVG
        with open(output_file, "w") as f:
            f.write(svg_content)
            print(f"Successfully wrote SVG to {output_file}")

        # Cleanup temporary files
        cleanup_files = ["temp.tex", "temp.pdf",
                         "temp.svg", "temp.log", "temp.aux"]
        for file in cleanup_files:
            if os.path.exists(file):
                os.remove(file)

    except Exception as e:
        print(f"Error during conversion: {e}")
        return


def main():
    if len(sys.argv) != 3:
        print("Usage: python tikz2svg.py input.tikz output.svg")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        with open(input_file, "r") as f:
            tikz_code = f.read()
        process_tikz_to_svg(tikz_code, output_file)
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
