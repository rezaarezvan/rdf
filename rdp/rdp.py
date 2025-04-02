import os
import io
import re
import sys
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Callable, Dict
from dataclasses import dataclass


@dataclass
class ColorTheme:
    """Color theme configuration"""

    light_theme: Dict[str, str]
    dark_theme: Dict[str, str]
    base_colors: Dict[str, str]


class RDP:
    """Wrapper for creating plots with light/dark theme support"""

    def __init__(self):
        self.save_path = Path("result") / Path(sys.argv[0]).stem
        self.color_theme = ColorTheme(
            light_theme={
                "primary": "#d62728",
                "secondary": "#2ca02c",
                "tertiary": "#1f77b4",
                "quaternary": "#9467bd",
                "quinary": "#8c564b",
                "senary": "#e377c2",
                "septenary": "#7f7f7f",
                "octonary": "#bcbd22",
            },
            dark_theme={
                "primary": "#FF4A98",
                "secondary": "#0AFAFA",
                "tertiary": "#7F83FF",
                "quaternary": "#B4A0FF",
                "quinary": "#FFB86B",
                "senary": "#FF79C6",
                "septenary": "#CCCCCC",
                "octonary": "#E6DB74",
            },
            base_colors={
                "c1": "#FF0001",
                "c2": "#00FF02",
                "c3": "#0000FF",
                "c4": "#FF00FF",
                "c5": "#00FFFF",
                "c6": "#FFFF00",
                "c7": "#FF8000",
                "c8": "#8000FF",
            },
        )

        self.style_template = """
        <style>
            /* Light mode (default) */
            %(color_classes)s

            /* Text and graphic elements */
            text, .text {
                fill: #000000 !important;
                stroke: none !important;
            }

            .axis line, .axis path, .tick line, .tick path, path.domain, line.grid {
                stroke: #000000 !important;
            }

            .legend text {
                fill: #000000 !important;
            }

            g.legend > g:first-child > path:first-child {
                fill: rgba(255, 255, 255, 0.8) !important;
                stroke: rgba(0, 0, 0, 0.1) !important;
            }

            [stroke="#000000"], [stroke="black"] {
                stroke: #000000 !important;
            }

            /* Dark mode via media query */
            @media (prefers-color-scheme: dark) {
                /* Dark theme colors */
                .c1 { fill: %(primary-color-dark)s !important; stroke: %(primary-color-dark)s !important; }
                .c2 { fill: %(secondary-color-dark)s !important; stroke: %(secondary-color-dark)s !important; }
                .c3 { fill: %(tertiary-color-dark)s !important; stroke: %(tertiary-color-dark)s !important; }
                .c4 { fill: %(quaternary-color-dark)s !important; stroke: %(quaternary-color-dark)s !important; }
                .c5 { fill: %(quinary-color-dark)s !important; stroke: %(quinary-color-dark)s !important; }
                .c6 { fill: %(senary-color-dark)s !important; stroke: %(senary-color-dark)s !important; }
                .c7 { fill: %(septenary-color-dark)s !important; stroke: %(septenary-color-dark)s !important; }
                .c8 { fill: %(octonary-color-dark)s !important; stroke: %(octonary-color-dark)s !important; }

                /* Dark theme text and graphics */
                text, .text {
                    fill: #FFFFFF !important;
                    stroke: none !important;
                }

                .axis line, .axis path, .tick line, .tick path, path.domain, line.grid {
                    stroke: #FFFFFF !important;
                }

                .grid line {
                    stroke: rgba(255, 255, 255, 0.2) !important;
                }

                .legend text {
                    fill: #FFFFFF !important;
                }

                g.legend > g:first-child > path:first-child {
                    fill: rgba(26, 26, 26, 0.8) !important;
                    stroke: rgba(255, 255, 255, 0.1) !important;
                }

                [stroke="#000000"], [stroke="black"] {
                    stroke: #FFFFFF !important;
                }
            }
        </style>
        """

    def _get_style_defs(self) -> str:
        """Generate CSS style definitions with media query support for dark mode"""
        # Generate color classes for light mode
        color_classes = "\n".join(
            [
                f".c{i + 1} {{ fill: {
                    self.color_theme.light_theme[name]
                } !important; stroke: {
                    self.color_theme.light_theme[name]
                } !important; }}"
                for i, name in enumerate(self.color_theme.light_theme.keys())
            ]
        )

        # Create a dictionary with all replacement variables
        replacements = {
            "color_classes": color_classes,
            "primary-color-dark": self.color_theme.dark_theme["primary"],
            "secondary-color-dark": self.color_theme.dark_theme["secondary"],
            "tertiary-color-dark": self.color_theme.dark_theme["tertiary"],
            "quaternary-color-dark": self.color_theme.dark_theme["quaternary"],
            "quinary-color-dark": self.color_theme.dark_theme["quinary"],
            "senary-color-dark": self.color_theme.dark_theme["senary"],
            "septenary-color-dark": self.color_theme.dark_theme["septenary"],
            "octonary-color-dark": self.color_theme.dark_theme["octonary"],
        }

        return self.style_template % replacements

    # def create_themed_plot(self, plot_func: Callable[..., None], **plot_kwargs) -> str:
    def create_themed_plot(
        self, name: str, plot_func: Callable[..., None], **plot_kwargs
    ) -> str:
        """
        Create a plot with theme support

        Args:
            name: Name for the saved SVG file
            plot_func: Function that creates the matplotlib plot
            fig_size: Figure size tuple (width, height)
            **plot_kwargs: Additional arguments passed to the plot function

        Returns:
            Saved SVG content with light/dark theme support to self.save_path
        """
        os.makedirs(self.save_path, exist_ok=True)
        fig = plt.figure(facecolor="none")
        ax = plt.gca()
        ax.set_facecolor("none")

        plot_kwargs["color_map"] = self.color_theme.base_colors

        plot_func(ax=ax, **plot_kwargs)

        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color("#000000")

        ax.title.set_color("#000000")
        ax.xaxis.label.set_color("#000000")
        ax.yaxis.label.set_color("#000000")

        for spine in ax.spines.values():
            spine.set_color("#000000")
        ax.tick_params(colors="#000000")

        output = io.StringIO()
        plt.savefig(output, format="svg", bbox_inches="tight", transparent=True)
        plt.close()

        svg_content = output.getvalue()

        svg_content = re.sub(
            r"(<svg[^>]*>)", rf"\1{self._get_style_defs()}", svg_content
        )

        for i, (name, color) in enumerate(self.color_theme.base_colors.items(), 1):
            pattern = rf'style="fill: ?{color}[^"]*"|fill="{color}"'
            svg_content = re.sub(pattern, f'class="c{i}"', svg_content)

        text_patterns = [
            (r'<g id="([^"]*text[^"]*)"', r'<g id="\1" class="text"'),
            (r'<g id="matplotlib\.axis[^"]*"', r'<g class="axis"'),
            (r'<g id="legend[^"]*"', r'<g class="legend"'),
            (r'<g id="xtick_[^"]*"', r'<g class="tick xtick"'),
            (r'<g id="ytick_[^"]*"', r'<g class="tick ytick"'),
            (
                r'<path [^>]*class="[^"]*" d="M[^"]* L[^"]*" style="[^"]*(stroke-width: 0.8;)[^"]*"',
                r'<path class="domain" \1',
            ),
            (r'<g id="patch_[^"]*"', r'<g class="patch"'),
        ]

        for pattern, replacement in text_patterns:
            svg_content = re.sub(pattern, replacement, svg_content)

        svg_content = re.sub(
            r'<g id="(grid_[^"]*)"', r'<g id="\1" class="grid"', svg_content
        )

        stroke_patterns = [
            r'stroke="#000000"',
            r'style="[^"]*stroke: ?#000000[^"]*"',
            r'style="[^"]*stroke:#000000[^"]*"',
        ]

        for pattern in stroke_patterns:
            if "style=" in pattern:
                svg_content = re.sub(
                    r'(style="[^"]*?)stroke: ?#000000([^"]*")',
                    r"\1stroke: currentColor\2",
                    svg_content,
                )
            else:
                svg_content = re.sub(pattern, 'stroke="currentColor"', svg_content)

        with open(f"{self.save_path}/{name}.svg", "w") as f:
            f.write(svg_content)

        print(f"SVG saved to {self.save_path}/{name}.svg")
        return svg_content
