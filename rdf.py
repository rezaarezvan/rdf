import os
import io
import re
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path
from typing import Callable, Dict
from dataclasses import dataclass


@dataclass
class ColorTheme:
    """Color theme configuration"""

    light_theme: Dict[str, str]
    dark_theme: Dict[str, str]
    base_colors: Dict[str, str]


class RDF:
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

            /* 3D specific styles - light mode */
            .plot3d path {
                stroke: rgba(0, 0, 0, 0.3) !important;
            }

            .plot3d polygon {
                stroke: rgba(0, 0, 0, 0.2) !important;
            }

            /* 3D grid lines */
            .plot3d-grid line {
                stroke: rgba(0, 0, 0, 0.2) !important;
            }

            /* Preserve mesh colors */
            .plot3d-surface path[style*="fill:"] {
                fill-opacity: 0.9 !important;
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


                /* 3D specific styles - dark mode */
                .plot3d path {
                    stroke: rgba(255, 255, 255, 0.3) !important;
                }

                .plot3d polygon {
                    stroke: rgba(255, 255, 255, 0.2) !important;
                }

                /* 3D grid lines */
                .plot3d-grid line {
                    stroke: rgba(255, 255, 255, 0.2) !important;
                }

                /* Preserve mesh colors in dark mode */
                .plot3d-surface path[style*="fill:"] {
                    fill-opacity: 0.95 !important;
                    filter: saturate(1.2) !important; /* Boost colors in dark mode */
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

    def create_themed_plot(
        self,
        save_name: str,
        plot_func: Callable[..., None],
        is_3d: bool = False,
        **plot_kwargs,
    ) -> str:
        """
        Create a plot with theme support. Now supporting 3D plots.
        """
        os.makedirs(self.save_path, exist_ok=True)
        fig = plt.figure(facecolor="none")

        # Create appropriate axes based on is_3d flag
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor("none")
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            ax.xaxis.pane.set_edgecolor("gray")
            ax.yaxis.pane.set_edgecolor("gray")
            ax.zaxis.pane.set_edgecolor("gray")
            ax.xaxis.pane.set_alpha(0.3)
            ax.yaxis.pane.set_alpha(0.3)
            ax.zaxis.pane.set_alpha(0.3)
        else:
            ax = plt.gca()
            ax.set_facecolor("none")

        plot_kwargs["color_map"] = self.color_theme.base_colors

        # Generate the plot
        result = plot_func(ax=ax, **plot_kwargs)

        # Apply styling to text elements
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color("#000000")

        # Handle return type - some 3D funcs may return objects
        if result is not None and is_3d:
            # Special handling for returned 3D objects if needed
            pass

        # For 3D plots, handle z labels and adjust color map settings
        if is_3d:
            for text in ax.get_zticklabels():
                text.set_color("#000000")
            ax.zaxis.label.set_color("#000000")

            # If a colormap is used (e.g., for surface plots), ensure it has good contrast
            if hasattr(result, "set_cmap"):
                # Use a colormap that works well in both light and dark modes
                # Or plasma, inferno - all work well in dark mode
                result.set_cmap("viridis")

            # Make grid lines more visible with better contrast
            ax.grid(True, alpha=0.5, linestyle="-", linewidth=0.5)

        ax.title.set_color("#000000")
        ax.xaxis.label.set_color("#000000")
        ax.yaxis.label.set_color("#000000")

        for spine in ax.spines.values():
            spine.set_color("#000000")
        ax.tick_params(colors="#000000")

        # Save the figure
        output = io.StringIO()
        plt.savefig(output, format="svg",
                    bbox_inches="tight", transparent=True)
        plt.close()

        svg_content = output.getvalue()

        # Add our style definitions
        svg_content = re.sub(
            r"(<svg[^>]*>)", rf"\1{self._get_style_defs()}", svg_content
        )

        # Add special class for 3D objects to better target them in CSS
        if is_3d:
            # Mark 3D surface elements
            svg_content = re.sub(
                r'<g id="surface[^"]*"', r'<g class="plot3d-surface"', svg_content
            )

            # Mark 3D grid elements
            svg_content = re.sub(
                r'<g id="[^"]*grid3d[^"]*"', r'<g class="plot3d-grid"', svg_content
            )

        # Process the SVG content - need additional patterns for 3D elements
        # Standard 2D color replacements
        for i, (_, color) in enumerate(self.color_theme.base_colors.items(), 1):
            pattern = rf'style="fill: ?{color}[^"]*"|fill="{color}"'
            svg_content = re.sub(pattern, f'class="c{i}"', svg_content)

            # Add 3D specific replacements (surfaces, etc.)
            if is_3d:
                # Handle face colors in 3D plots
                pattern_3d = rf'facecolor="{color}"'
                svg_content = re.sub(pattern_3d, f'class="c{i}"', svg_content)

        # Apply text and other element classes
        text_patterns = [
            # Existing patterns
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
            # Add 3D specific patterns
            (r'<g id="ztick_[^"]*"', r'<g class="tick ztick"'),
            (r'<g id="(Poly3D|Line3D)[^"]*"', r'<g class="plot3d"'),
        ]

        for pattern, replacement in text_patterns:
            svg_content = re.sub(pattern, replacement, svg_content)

        # Handle grid lines
        svg_content = re.sub(
            r'<g id="(grid_[^"]*)"', r'<g id="\1" class="grid"', svg_content
        )

        # Handle stroke colors
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
                svg_content = re.sub(
                    pattern, 'stroke="currentColor"', svg_content)

        # Save the final SVG
        with open(f"{self.save_path}/{save_name}.svg", "w") as f:
            f.write(svg_content)

        print(f"SVG saved to {self.save_path}/{save_name}.svg")
        return svg_content

    def create_animated_plot(
        self,
        save_name: str,
        plot_func: Callable[..., None],
        animation_type: str = "draw",
        animation_duration: float = 2.0,
        animation_delay: float = 0.0,
        loop: bool = True,
        is_3d: bool = False,
        **plot_kwargs,
    ) -> str:
        """
        Create a plot with animation support.

        Args:
            save_name: Name for saved file
            plot_func: Function to generate the plot
            animation_type: Type of animation ('draw', 'fade', 'pulse')
            animation_duration: Duration of animation in seconds
            animation_delay: Delay before animation starts
            loop: Whether animation should loop infinitely
            is_3d: Whether this is a 3D plot
            **plot_kwargs: Additional arguments for plot_func
        """
        # Create the normal SVG
        svg_content = self.create_themed_plot(
            save_name=f"{save_name}_static",
            plot_func=plot_func,
            is_3d=is_3d,
            **plot_kwargs
        )

        # Process SVG to add animation
        animated_svg = self._add_animation(
            svg_content,
            animation_type,
            animation_duration,
            animation_delay,
            loop
        )

        # Save the animated version
        with open(f"{self.save_path}/{save_name}.svg", "w") as f:
            f.write(animated_svg)

        return animated_svg

    def _add_animation(
        self,
        svg_content: str,
        animation_type: str,
        duration: float,
        delay: float,
        loop: bool
    ) -> str:
        # Add animation iteration count based on loop setting
        iteration = "infinite" if loop else "1"

        # Add necessary CSS animation definitions
        animation_css = """
        <style>
            @keyframes drawLine {{
                from {{ stroke-dashoffset: var(--length); }}
                to {{ stroke-dashoffset: 0; }}
            }}

            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}

            @keyframes pulse {{
                0% {{ opacity: 0.7; }}
                50% {{ opacity: 1; }}
                100% {{ opacity: 0.7; }}
            }}

            .animated-path {{
                --length: 1000;
                stroke-dasharray: var(--length);
                stroke-dashoffset: var(--length);
                animation: drawLine {duration}s ease-in-out {iteration};
                animation-delay: {delay}s;
            }}

            .animated-fade {{
                opacity: 0;
                animation: fadeIn {duration}s ease-in-out {iteration};
                animation-delay: {delay}s;
            }}

            .animated-pulse {{
                animation: pulse {duration}s ease-in-out infinite;
                animation-delay: {delay}s;
            }}
        </style>
        """.format(duration=duration, delay=delay, iteration=iteration)

        # Insert animation CSS after SVG opening tag
        svg_content = re.sub(
            r"(<svg[^>]*>)", r"\1" + animation_css, svg_content)

        # Process paths based on animation type
        if animation_type == "draw":
            # Find all path elements
            path_pattern = r'<path[^>]*d="([^"]*)"[^>]*>'

            def process_path(match):
                path_element = match.group(0)

                # Add animated-path class
                if 'class="' in path_element:
                    path_element = path_element.replace(
                        'class="', 'class="animated-path ')
                else:
                    path_element = path_element.replace(
                        '<path', '<path class="animated-path"')

                return path_element

            svg_content = re.sub(path_pattern, process_path, svg_content)

        elif animation_type == "fade":
            # Add fade animation to all elements
            element_pattern = r'<(g|path|circle|rect|text)[^>]*>'

            def add_fade_class(match):
                element = match.group(0)
                if 'class="' in element:
                    element = element.replace(
                        'class="', 'class="animated-fade ')
                else:
                    element = element.replace(
                        '<' + match.group(1), '<' + match.group(1) + ' class="animated-fade"')
                return element

            svg_content = re.sub(element_pattern, add_fade_class, svg_content)

        elif animation_type == "pulse":
            # Add pulse animation to specific elements
            path_pattern = r'<path[^>]*d="([^"]*)"[^>]*>'

            def add_pulse_class(match):
                path_element = match.group(0)
                if 'class="' in path_element:
                    path_element = path_element.replace(
                        'class="', 'class="animated-pulse ')
                else:
                    path_element = path_element.replace(
                        '<path', '<path class="animated-pulse"')
                return path_element

            svg_content = re.sub(path_pattern, add_pulse_class, svg_content)

        # Add JavaScript to calculate actual path lengths
        svg_content = svg_content.replace('</svg>', '''
        <script type="text/javascript">
            document.addEventListener("DOMContentLoaded", function() {
                const paths = document.querySelectorAll('.animated-path');
                paths.forEach(path => {
                    if (path.getTotalLength) {
                        const length = path.getTotalLength();
                        path.style.setProperty('--length', length);
                        path.setAttribute('stroke-dasharray', length);
                        path.setAttribute('stroke-dashoffset', length);
                    }
                });
            });
        </script>
        </svg>
        ''')

        return svg_content
