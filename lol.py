import matplotlib.pyplot as plt
import io
from typing import Optional, Dict, Any

class DynamicSVGGenerator:
    """Generate dark/light mode compatible SVGs from matplotlib figures."""

    # Default color schemes
    DARK_COLORS = {
        'primary': '#FF4A98',    # Pinkish
        'secondary': '#0AFAFA',  # Cyan
        'tertiary': '#7F83FF',   # Light purple
        'background': '#000000',
        'text': '#FFFFFF',
        'grid': '#202020'
    }

    LIGHT_COLORS = {
        'primary': '#FF1B5E',     # Darker pink
        'secondary': '#00A0A0',   # Darker cyan
        'tertiary': '#4B4DFF',    # Darker purple
        'background': '#FFFFFF',
        'text': '#000000',
        'grid': '#E0E0E0'
    }

    def __init__(self):
        self.reset_style()

    def reset_style(self):
        """Reset matplotlib style to default."""
        plt.style.use('default')

    def _add_dynamic_styles(self, svg_content: str) -> str:
        """Add CSS styles for dark/light mode support."""
        style_tag = '''
        <style>
            :root {
                color-scheme: light dark;
            }

            svg {
                color: black;
                background: white;
            }

            @media (prefers-color-scheme: dark) {
                svg {
                    color: white;
                    background: black;
                }

                .background {
                    fill: black;
                }

                .tick text,
                .axis-label,
                .title {
                    fill: white;
                }

                .tick line,
                .domain {
                    stroke: white;
                }

                .grid line {
                    stroke: #202020;
                }
            }

            /* Ensure text remains visible in both modes */
            .text-preserve {
                fill: currentColor !important;
            }

            /* Handle transparent backgrounds properly */
            .figure {
                fill: none;
            }
        </style>
        '''

        # Insert style tag after the opening svg tag
        svg_with_style = svg_content.replace('<svg', f'<svg class="dynamic-svg"{style_tag}', 1)

        # Add classes to elements
        replacements = [
            ('class="text"', 'class="text-preserve"'),
            ('<text', '<text class="text-preserve"'),
            ('<path d="M 0 0', '<path class="background" d="M 0 0'),
            ('class="axis"', 'class="axis text-preserve"'),
            ('<g id="patch_1">', '<g id="patch_1" class="figure">'),
        ]

        for old, new in replacements:
            svg_with_style = svg_with_style.replace(old, new)

        return svg_with_style

    def setup_dark_mode(self):
        """Configure matplotlib for dark mode plotting."""
        plt.style.use('dark_background')
        plt.rcParams['grid.color'] = self.DARK_COLORS['grid']
        plt.rcParams['text.color'] = self.DARK_COLORS['text']
        plt.rcParams['axes.labelcolor'] = self.DARK_COLORS['text']
        plt.rcParams['xtick.color'] = self.DARK_COLORS['text']
        plt.rcParams['ytick.color'] = self.DARK_COLORS['text']

    def setup_light_mode(self):
        """Configure matplotlib for light mode plotting."""
        plt.style.use('default')
        plt.rcParams['grid.color'] = self.LIGHT_COLORS['grid']
        plt.rcParams['text.color'] = self.LIGHT_COLORS['text']
        plt.rcParams['axes.labelcolor'] = self.LIGHT_COLORS['text']
        plt.rcParams['xtick.color'] = self.LIGHT_COLORS['text']
        plt.rcParams['ytick.color'] = self.LIGHT_COLORS['text']

    def save_dynamic_svg(self,
                        fig: plt.Figure,
                        output_file: str,
                        fig_settings: Optional[Dict[str, Any]] = None):
        """
        Save a matplotlib figure as a dynamic SVG with dark/light mode support.

        Args:
            fig: matplotlib Figure object
            output_file: Path to save the SVG file
            fig_settings: Optional dictionary of figure settings
                         (e.g., {'bbox_inches': 'tight'})
        """
        if fig_settings is None:
            fig_settings = {'bbox_inches': 'tight'}

        # Save figure to an in-memory buffer
        svg_buffer = io.StringIO()
        fig.savefig(svg_buffer, format='svg', **fig_settings)
        svg_content = svg_buffer.getvalue()

        # Add dynamic styles
        svg_with_styles = self._add_dynamic_styles(svg_content)

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(svg_with_styles)

        # Close buffer
        svg_buffer.close()

# Example usage
if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import load_iris

    # Initialize generator
    svg_gen = DynamicSVGGenerator()

    # Load iris dataset
    X, y = load_iris(return_X_y=True)

    # Create histogram
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot with dark mode colors
    svg_gen.setup_dark_mode()
    plt.hist(X[y == 0, 2], color=svg_gen.DARK_COLORS['primary'],
             alpha=0.5, label='setosa')
    plt.hist(X[y == 1, 2], color=svg_gen.DARK_COLORS['secondary'],
             alpha=0.5, label='versicolor')
    plt.hist(X[y == 2, 2], color=svg_gen.DARK_COLORS['tertiary'],
             alpha=0.5, label='virginica')

    plt.title('Histogram of Petal Length')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Frequency')
    plt.legend()

    # Save as dynamic SVG
    svg_gen.save_dynamic_svg(fig, 'dynamic_iris_histogram.svg')

    # Clean up
    plt.close()
