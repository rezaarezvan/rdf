import numpy as np
import matplotlib.pyplot as plt


def plot_sigmoid(ax=None, color_map=None):
    """
    Create a clean, blog-friendly plot of the sigmoid function.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Create smooth x range
    x = np.linspace(-10, 10, 200)  # Increased points for smoother curve
    y = 1 / (1 + np.exp(-x))

    # Plot sigmoid curve
    ax.plot(x, y, color=color_map["c1"], linewidth=2, label="Sigmoid", zorder=3)

    # Add horizontal lines at y=0 and y=1
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3, zorder=1)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.3, zorder=1)

    # Add vertical line at x=0
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3, zorder=1)

    # Customize plot appearance
    ax.set_title("Sigmoid Function", fontsize=12, pad=15)
    ax.set_xlabel("f(x)", fontsize=10)
    ax.set_ylabel(r"$\sigma(f(x))$", fontsize=10)

    # Clean legend
    ax.legend(frameon=True, framealpha=0.9, loc="lower right", fontsize=9)

    # Set consistent limits with slight padding
    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.1, 1.1)


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()
    svg_content = plotter.create_themed_plot(
        save_name="sigmoid", plot_func=plot_sigmoid
    )
