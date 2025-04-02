import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.path import Path


def plot_convex_sets(ax=None, color_map=None):
    """
    Create a clean, blog-friendly visualization of convex and non-convex sets.
    Shows examples with line segments demonstrating convexity properties.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Set up the plot bounds
    ax.set_xlim(-1.5, 5.5)
    ax.set_ylim(-2, 2)

    # Draw convex hexagon
    hex_vertices = np.array([
        [0, 1], [0.866, 0.5], [0.866, -0.5],
        [0, -1], [-0.866, -0.5], [-0.866, 0.5]
    ])
    hex_path = Path(hex_vertices, closed=True)
    hex_patch = patches.PathPatch(
        hex_path, facecolor='none', edgecolor=color_map['c1'],
        linewidth=2, alpha=0.8
    )
    ax.add_patch(hex_patch)

    # Add line segment for hexagon
    ax.plot([-0.5, 0.5], [-0.5, 0.5], '--', color=color_map['c1'],
            linewidth=1.5, alpha=0.8)

    # Draw non-convex shape (curved)
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.8 + 0.3*np.sin(3*theta)
    x = 2 + r*np.cos(theta)
    y = r*np.sin(theta)
    ax.plot(x, y, color=color_map['c2'], linewidth=2, alpha=0.8)

    # Add line segment for non-convex shape
    ax.plot([1.5, 2.5], [-0.5, 0.5], '--', color=color_map['c2'],
            linewidth=1.5, alpha=0.8)

    # Draw square (convex)
    square = plt.Rectangle((3.5, -1), 1.5, 2, fill=False,
                           edgecolor=color_map['c3'], linewidth=2, alpha=0.8)
    ax.add_patch(square)

    # Add line segment for square
    ax.plot([3.7, 4.8], [-0.5, 0.5], '--', color=color_map['c3'],
            linewidth=1.5, alpha=0.8)

    # Add labels
    ax.text(0, -1.5, 'Convex', ha='center', va='center', fontsize=10)
    ax.text(2, -1.5, 'Non-convex', ha='center', va='center', fontsize=10)
    ax.text(4.25, -1.5, 'Convex', ha='center', va='center', fontsize=10)

    # Customize plot appearance
    ax.set_title('Examples of Convex and Non-convex Sets',
                 fontsize=12, pad=15)

    # Remove axes for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Set aspect ratio to equal for proper shape display
    ax.set_aspect('equal')


if __name__ == "__main__":
    from rdp import RDP

    plotter = RDP()
    svg_content = plotter.create_themed_plot(
        save_name='convex_sets', plot_func=plot_convex_sets)
