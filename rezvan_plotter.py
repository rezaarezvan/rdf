import io
import re
import matplotlib.pyplot as plt
import numpy as np

from typing import Callable, Dict
from dataclasses import dataclass


@dataclass
class ColorTheme:
    """Color theme configuration"""
    light_theme: Dict[str, str]
    dark_theme: Dict[str, str]
    base_colors: Dict[str, str]


class ThemeablePlot:
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
                }
                /* Dark mode colors */
                .dark {
                    %(dark_theme)s
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

        return f"<defs>{self.style_template % {
            'light_theme': light_theme_vars,
            'dark_theme': dark_theme_vars,
            'color_classes': color_classes
        }}</defs>"

    def create_themed_plot(self, plot_func: Callable[..., None],
                           fig_size: tuple = (8, 6),
                           **plot_kwargs) -> str:
        """
        Create a plot with theme support

        Args:
            plot_func: Function that creates the matplotlib plot
            fig_size: Figure size tuple (width, height)
            **plot_kwargs: Additional arguments passed to the plot function

        Returns:
            SVG string with theme support
        """
        fig = plt.figure(figsize=fig_size, facecolor='none')
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


if __name__ == "__main__":
    from sklearn import datasets

    def create_line_plot(ax=None, color_map=None):
        x = np.linspace(0, 10, 100)
        colors = list(color_map.values())

        ax.plot(x, np.sin(x), color=colors[0], label='sin(x)')
        ax.plot(x, np.cos(x), color=colors[1], label='cos(x)')
        ax.plot(x, -np.sin(x), color=colors[2], label='-sin(x)')

        ax.set_title('Trigonometric Functions')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()

    def create_scatter_plot(ax=None, color_map=None):
        # Generate sample data
        np.random.seed(42)
        n_points = 50

        # Create three clusters
        colors = list(color_map.values())

        for i, (mx, my, color) in enumerate(zip(
            [0, 2, -2],  # x means
            [0, 2, -2],  # y means
            colors[:3]   # first three colors
        )):
            x = np.random.normal(mx, 0.3, n_points)
            y = np.random.normal(my, 0.3, n_points)
            ax.scatter(x, y, color=color, alpha=0.6, label=f'Cluster {i+1}')

        ax.set_title('Scatter Plot Example')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.legend()

    def create_bar_plot(ax=None, color_map=None):
        categories = ['A', 'B', 'C', 'D']
        values1 = [4, 3, 2, 1]
        values2 = [1, 2, 3, 4]

        colors = list(color_map.values())
        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width/2, values1, width, color=colors[0], label='Group 1')
        ax.bar(x + width/2, values2, width, color=colors[1], label='Group 2')

        ax.set_title('Bar Plot Example')
        ax.set_xlabel('Categories')
        ax.set_ylabel('Values')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

    def create_iris_plot(ax=None, color_map=None):
        iris = datasets.load_iris()
        X = iris.data[:, 2]  # Petal length
        y = iris.target

        colors = list(color_map.values())[:3]
        for i, color in enumerate(colors):
            mask = y == i
            ax.hist(X[mask], alpha=0.5, color=color,
                    label=iris.target_names[i])

        ax.set_title('Histogram of Petal Length')
        ax.set_xlabel('Petal Length (cm)')
        ax.set_ylabel('Frequency')
        ax.legend()

    # Create themed plot
    plotter = ThemeablePlot()
    svg_content = plotter.create_themed_plot(create_iris_plot)

    # Save the SVG
    with open('themed_iris_plot.svg', 'w') as f:
        f.write(svg_content)

    # Create all plot types
    plots = {
        'line': create_line_plot,
        'scatter': create_scatter_plot,
        'bar': create_bar_plot
    }

    # Generate each plot
    for name, plot_func in plots.items():
        svg_content = plotter.create_themed_plot(plot_func)
        with open(f'themed_{name}_plot.svg', 'w') as f:
            f.write(svg_content)
