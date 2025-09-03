import numpy as np
import matplotlib.pyplot as plt


def plot_convex_functions(ax=None, color_map=None):
    """
    Create a clean, blog-friendly visualization of convex and non-convex functions.
    Shows a convex function with unique minimum and a non-convex function with
    multiple local minima.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Generate x values
    x = np.linspace(-2, 2, 200)

    # Plot convex function (parabola)
    y1 = x**2
    ax1.plot(x, y1, color=color_map["c1"], linewidth=2)

    # Mark minimum point for convex function
    min_x1, min_y1 = 0, 0
    ax1.plot(
        min_x1, min_y1, "o", color=color_map["c1"], markersize=8, label="Global Minimum"
    )

    # Plot non-convex function (double-well potential)
    y2 = x**4 - 2 * x**2 + 1
    ax2.plot(x, y2, color=color_map["c2"], linewidth=2)

    # Mark local and global minima for non-convex function
    min_x2 = np.array([-1, 1])
    min_y2 = min_x2**4 - 2 * min_x2**2 + 1
    ax2.plot(
        min_x2, min_y2, "o", color=color_map["c2"], markersize=8, label="Local Minima"
    )

    # Add saddle point
    saddle_x, saddle_y = 0, 1
    ax2.plot(
        saddle_x,
        saddle_y,
        "s",
        color=color_map["c3"],
        markersize=8,
        label="Saddle Point",
    )

    # Customize both subplots
    for ax, title in zip([ax1, ax2], ["Convex Function", "Non-convex Function"]):
        ax.set_title(title, fontsize=12, pad=15)
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Set limits with padding
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-0.5, 4)

        # Remove ticks, only keep arrows at ends
        ax.set_xticks([])
        ax.set_yticks([])

        # Add arrows at the end of axes
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

        # Add subtle grid
        ax.grid(True, alpha=0.15, linestyle="-")

    # Add legend to second subplot only
    ax2.legend(
        frameon=True,
        framealpha=0.9,
        loc="upper right",
        fontsize=9,
        bbox_to_anchor=(0.98, 0.98),
    )

    # Adjust layout
    plt.tight_layout(pad=2.0)
    return fig


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()
    svg_content = plotter.create_themed_plot(
        save_name="convex_functions", plot_func=plot_convex_functions
    )
