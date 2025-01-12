import numpy as np
import matplotlib.pyplot as plt


def plot_margin_principle(ax=None, color_map=None):
    """
    Create a clean, blog-friendly visualization of the maximum margin principle
    for Support Vector Machines (SVM).

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Generate synthetic data
    np.random.seed(42)  # For reproducibility

    # Class 1: Bottom-left cluster
    n1 = 20
    x1 = np.random.normal(-3, 1.5, n1)
    y1 = np.random.normal(-4, 1.5, n1)

    # Class 2: Top-right cluster
    n2 = 20
    x2 = np.random.normal(4, 1.5, n2)
    y2 = np.random.normal(5, 1.5, n2)

    # Plot data points
    ax.scatter(x1, y1, c=color_map['c1'], s=70, alpha=0.7,
               label='Class 1', edgecolor='white', linewidth=0.5, zorder=3)
    ax.scatter(x2, y2, c=color_map['c2'], s=70, alpha=0.7,
               label='Class 2', edgecolor='white', linewidth=0.5, zorder=3)

    # Create decision boundary and margins
    x_range = np.array([-10, 10])

    # Decision boundary: y = x + 0
    boundary = x_range

    # Margins: y = x ± 2.5
    margin_up = x_range + 2.5
    margin_down = x_range - 2.5

    # Plot decision boundary and margins
    ax.plot(x_range, boundary, '-', color='black', linewidth=2,
            label='Decision Boundary', zorder=2)
    ax.plot(x_range, margin_up, '--', color='gray', linewidth=1.5,
            label='Margin', zorder=2, alpha=0.7)
    ax.plot(x_range, margin_down, '--', color='gray', linewidth=1.5,
            zorder=2, alpha=0.7)

    # Add margin point annotation
    margin_point_x = -2
    margin_point_y = margin_down[1]
    ax.scatter(margin_point_x, margin_point_y, c=color_map['c1'],
               s=100, alpha=0.7, edgecolor='black', linewidth=1.5,
               zorder=4)
    ax.annotate('margin point', xy=(margin_point_x, margin_point_y),
                xytext=(margin_point_x - 1, margin_point_y - 2),
                fontsize=9, ha='right',
                arrowprops=dict(arrowstyle='->',
                                connectionstyle='arc3,rad=0.2'))

    # Customize plot appearance
    ax.set_title('Maximum Margin Principle', fontsize=12, pad=15)
    ax.set_xlabel(r'$x_1$', fontsize=10)
    ax.set_ylabel(r'$x_2$', fontsize=10)

    # Clean legend
    ax.legend(frameon=True, framealpha=0.9, loc='upper left',
              fontsize=9, bbox_to_anchor=(0.02, 0.98))

    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # Subtle grid
    ax.grid(True, alpha=0.15, linestyle='-', zorder=1)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add subtle ticks
    ax.tick_params(axis='both', which='major', labelsize=9)


if __name__ == "__main__":
    from rdp import RDP

    plotter = RDP()
    svg_content = plotter.create_themed_plot(plot_margin_principle)

    with open('margin_principle.svg', 'w') as f:
        f.write(svg_content)
