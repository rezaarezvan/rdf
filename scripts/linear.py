import numpy as np


def plot_halfspace(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of a linear halfspace.
    Demonstrates how a line divides 2D space into two regions.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Set up the plot bounds with padding
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    padding = 0.2  # Add padding for better appearance

    # Create the dividing line
    x = np.array([x_min - padding, x_max + padding])
    y = -0.5 * x + 1  # Line equation: y = -0.5x + 1

    # Color the halfspaces first (lower z-order)
    ax.fill_between(
        x,
        y,
        y_max + padding,
        color=color_map["c2"],
        alpha=0.15,
        label="Positive Halfspace",
        zorder=1,
    )
    ax.fill_between(
        x,
        y_min - padding,
        y,
        color=color_map["c1"],
        alpha=0.15,
        label="Negative Halfspace",
        zorder=1,
    )

    # Plot the decision boundary on top
    ax.plot(x, y, color="black", linewidth=2, label="Decision Boundary", zorder=3)

    # Add subtle grid
    ax.grid(True, alpha=0.1, linestyle="-", zorder=0)

    # Customize plot appearance
    ax.set_title("Linear Halfspace Example", fontsize=12, pad=15)
    ax.set_xlabel(r"$x_1$", fontsize=10)
    ax.set_ylabel(r"$x_2$", fontsize=10)

    # Clean legend
    ax.legend(
        frameon=True,
        framealpha=0.9,
        loc="upper right",
        fontsize=9,
        bbox_to_anchor=(0.98, 0.98),
    )

    # Set axis limits with padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set aspect ratio to be equal for proper visualization
    ax.set_aspect("equal")

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)


if __name__ == "__main__":
    from rdp import RDP

    plotter = RDP()
    svg_content = plotter.create_themed_plot(
        save_name="linear_halfspace", plot_func=plot_halfspace
    )
