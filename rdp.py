import io
import re
import matplotlib.pyplot as plt

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
        self.color_theme = ColorTheme(
            light_theme={
                'primary': '#d62728',
                'secondary': '#2ca02c',
                'tertiary': '#1f77b4',
                'quaternary': '#9467bd',
                'quinary': '#8c564b',
                'senary': '#e377c2',
                'septenary': '#7f7f7f',
                'octonary': '#bcbd22',
            },
            dark_theme={
                'primary': '#FF4A98',
                'secondary': '#0AFAFA',
                'tertiary': '#7F83FF',
                'quaternary': '#B4A0FF',
                'quinary': '#FFB86B',
                'senary': '#FF79C6',
                'septenary': '#CCCCCC',
                'octonary': '#E6DB74',
            },
            base_colors={
                'c1': '#FF0001',
                'c2': '#00FF02',
                'c3': '#0000FF',
                'c4': '#FF00FF',
                'c5': '#00FFFF',
                'c6': '#FFFF00',
                'c7': '#FF8000',
                'c8': '#8000FF',
            }
        )

        self.style_template = """
            <style>
                /* Light mode colors */
                :root {
                    %(light_theme)s
                    --legend-bg: rgba(255, 255, 255, 0.8);
                    --legend-border: rgba(0, 0, 0, 0.1);
                }
                /* Dark mode colors */
                .dark {
                    %(dark_theme)s
                    --legend-bg: rgba(26, 26, 26, 0.8);
                    --legend-border: rgba(255, 255, 255, 0.1);
                }
                /* Color classes */
                %(color_classes)s
                /* Make all text inherit color */
                text, .text {
                    fill: currentColor !important;
                    stroke: none !important;
                }
                /* Make all lines inherit color */
                .axis line, .axis path {
                    stroke: currentColor !important;
                }
                /* Special handling for legend */
                g.legend > g:first-child > path:first-child {
                    fill: var(--legend-bg) !important;
                    stroke: var(--legend-border) !important;
                    stroke-width: 1px !important;
                }
                .legend text {
                    fill: currentColor !important;
                }
                /* Make all default black strokes use currentColor */
                [stroke="#000000"] {
                    stroke: currentColor !important;
                }
            </style>
        """

    def _get_style_defs(self) -> str:
        """Generate CSS style definitions"""
        light_theme_vars = "\n".join(
            f"--{name}-color: {color};"
            for name, color in self.color_theme.light_theme.items()
        )

        dark_theme_vars = "\n".join(
            f"--{name}-color: {color};"
            for name, color in self.color_theme.dark_theme.items()
        )

        color_classes = "\n".join(
            f".c{i+1} {{ fill: var(--{name}-color) !important; }}"
            for i, name in enumerate(self.color_theme.light_theme.keys())
        )

        color_classes = "\n".join([
            f"""
            .c{i+1} {{
                fill: var(--{name}-color) !important;
                stroke: var(--{name}-color) !important;
            }}
            """
            for i, name in enumerate(self.color_theme.light_theme.keys())
        ])

        return f"< defs > {self.style_template % {'light_theme': light_theme_vars, 'dark_theme': dark_theme_vars, 'color_classes': color_classes}} < /defs >"

    def create_themed_plot(self, plot_func: Callable[..., None], **plot_kwargs) -> str:
        """
        Create a plot with theme support

        Args:
            plot_func: Function that creates the matplotlib plot
            fig_size: Figure size tuple (width, height)
            **plot_kwargs: Additional arguments passed to the plot function

        Returns:
            SVG string with theme support
        """
        fig = plt.figure(facecolor='none')
        ax = plt.gca()
        ax.set_facecolor('none')

        plot_kwargs['color_map'] = self.color_theme.base_colors

        plot_func(ax=ax, **plot_kwargs)

        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('#000000')

        ax.title.set_color('#000000')
        ax.xaxis.label.set_color('#000000')
        ax.yaxis.label.set_color('#000000')

        for spine in ax.spines.values():
            spine.set_color('#000000')
        ax.tick_params(colors='#000000')

        output = io.StringIO()
        plt.savefig(output, format='svg',
                    bbox_inches='tight', transparent=True)
        plt.close()

        svg_content = output.getvalue()

        svg_content = re.sub(
            r'(<svg[^>]*>)', rf'\1{self._get_style_defs()}', svg_content)

        for i, (name, color) in enumerate(self.color_theme.base_colors.items(), 1):
            pattern = rf'style="fill: ?{color}[^"]*"|fill="{color}"'
            svg_content = re.sub(pattern, f'class="c{i}"', svg_content)

        text_patterns = [
            (r'<g id="([^"]*text[^"]*)"', r'<g id="\1" class="text"'),
            (r'<g id="matplotlib\.axis[^"]*"', r'<g class="axis"'),
            (r'<g id="legend[^"]*"', r'<g class="legend"')
        ]

        for pattern, replacement in text_patterns:
            svg_content = re.sub(pattern, replacement, svg_content)

        stroke_patterns = [
            r'stroke="#000000"',
            r'style="[^"]*stroke: ?#000000[^"]*"',
            r'style="[^"]*stroke:#000000[^"]*"'
        ]

        for pattern in stroke_patterns:
            if 'style=' in pattern:
                svg_content = re.sub(
                    r'(style="[^"]*?)stroke: ?#000000([^"]*")',
                    r'\1stroke: currentColor\2',
                    svg_content
                )
            else:
                svg_content = re.sub(
                    pattern, 'stroke="currentColor"', svg_content)

        return svg_content
