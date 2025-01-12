import numpy as np


def plot_halfspace(ax=None, color_map=None):
    """Plot a halfspace example showing how a line divides 2D space"""
    # Set up the plot bounds
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5

    # Create the dividing line
    x = np.array([x_min, x_max])
    y = -0.5 * x + 1  # Line equation: y = -0.5x + 1

    # Plot the line
    ax.plot(x, y, 'k-', linewidth=2, label='Decision Boundary')

    # Create grid points for coloring the halfspaces
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Determine points above/below line
    z = yy + 0.5 * xx - 1

    # Color the halfspaces
    ax.fill_between(
        x, y, y_max, color=color_map['c2'], alpha=0.2, label='Positive Halfspace')
    ax.fill_between(
        x, y_min, y, color=color_map['c1'], alpha=0.2, label='Negative Halfspace')

    # Customize plot appearance
    ax.set_title('Linear Halfspace Example', fontsize=12, pad=15)
    ax.set_xlabel(r'$x_1$', fontsize=10)
    ax.set_ylabel(r'$x_2$', fontsize=10)

    # Clean legend
    ax.legend(frameon=True, framealpha=0.9, loc='upper right', fontsize=9)

    # Subtle grid
    ax.grid(True, alpha=0.15, linestyle='-')

    # Set consistent limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set aspect ratio to be equal
    ax.set_aspect('equal')


if __name__ == "__main__":
    from rdp import RDP
    plotter = RDP()

    svg_content = plotter.create_themed_plot(plot_halfspace)
    with open('linear_halfspace.svg', 'w') as f:
        f.write(svg_content)
