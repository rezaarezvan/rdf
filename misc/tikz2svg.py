import sys
import os
import subprocess


def process_tikz_to_svg(tikz_code: str, output_file: str):
    """Convert TikZ code to SVG with dark mode support."""
    try:
        # Read template
        with open("tikz-template.tex", "r") as f:
            template = f.read()
            print("Successfully read template")

        # Replace content
        tex_content = template.replace("% TIKZ_CONTENT_HERE", tikz_code)

        # Write temporary tex file
        with open("temp.tex", "w") as f:
            f.write(tex_content)
            print("Successfully wrote temp.tex")

        # Convert to PDF using pdflatex
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "temp.tex"],
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

        # Add dark mode support
        style_tag = """
        <style>
            svg {
                color: currentColor;
                display: block;
                margin: auto;
                width: 85%;
                height: auto;
            }

            .math text {
                fill: currentColor;
            }
        </style>
        """
        svg_content = svg_content.replace(
            "<svg", f'<svg class="text-black dark:text-white"{style_tag}'
        )

        # Add unique prefixes to prevent ID conflicts
        prefix = output_file.split("/")[-1].replace(".svg", "")
        svg_content = svg_content.replace('id="', f'id="{prefix}_')
        svg_content = svg_content.replace('href="#', f'href="#{prefix}_')
        svg_content = svg_content.replace("url(#", f"url(#{prefix}_")

        # Replace colors (excluding text and math elements)
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
        cleanup_files = ["temp.tex", "temp.pdf", "temp.svg", "temp.log", "temp.aux"]
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
