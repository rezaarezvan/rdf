import numpy as np
import matplotlib.pyplot as plt


def plot_example_question_regions(ax=None, color_map=None):
    """
    Create a clean visualization of shaded regions for example question,
    S = {
            g_1 = -x_1^3 + x_2 <= 0
            g_2 = x_1^5 - x_2 <= 0
            g_3 = -x_2 <= 0
    }

    Let bar(x) = [0 0]^T

    Plot the feasible region S and bar(x) on the axis.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Define the domain
    x_min, x_max = -2, 2
    y_min, y_max = -1, 2

    # Create a grid of points
    x = np.linspace(x_min, x_max, 400)
    y = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(x, y)

    # Define the inequalities
    g1 = -(X**3) + Y  # g_1 = -x_1^3 + x_2 <= 0
    g2 = X**5 - Y  # g_2 = x_1^5 - x_2 <= 0
    g3 = -Y  # g_3 = -x_2 <= 0

    # Create a mask for the feasible region S
    feasible_region = np.logical_and(np.logical_and(g1 <= 0, g2 <= 0), g3 <= 0)

    # Plot the feasible region
    ax.contourf(
        X,
        Y,
        feasible_region,
        levels=[0.5, 1],
        colors=[color_map["c8"]],
        alpha=0.5,
        zorder=1,
    )

    # Plot the boundary curves
    ax.contour(X, Y, g1, levels=[0], colors=color_map["c5"], linewidths=2.5, zorder=3)
    ax.contour(X, Y, g2, levels=[0], colors=color_map["c6"], linewidths=2.5, zorder=3)
    ax.axhline(0, color=color_map["c7"], linewidth=2.5, zorder=3)

    # Mark the point bar(x) = [0, 0]^T
    ax.scatter(
        0, 0, s=100, color=color_map["c1"], edgecolor="white", linewidth=2, zorder=5
    )

    # Add annotations
    ax.annotate(
        r"$ \bar{x} = [0, 0]^T $",
        xy=(0, 0),
        xytext=(-0.5, -0.5),
        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.annotate(
        r"$g_1(x) \leq 0$",
        xy=(1, 1),
        xytext=(1.5, 1.5),
        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        color=color_map["c5"],
    )

    ax.annotate(
        r"$g_2(x) \leq 0$",
        xy=(0.77, 0.3),
        xytext=(1.5, 0.5),
        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        color=color_map["c6"],
    )

    ax.annotate(
        r"$g_3(x) \leq 0$",
        xy=(0, 0),
        xytext=(0.5, -0.5),
        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        color=color_map["c7"],
    )

    # Customize plot appearance
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", "box")
    # Add x_1 and x_2 axes
    ax.axhline(0, color=color_map["black"], linewidth=1, alpha=0.8, zorder=2)
    ax.axvline(0, color=color_map["black"], linewidth=1, alpha=0.8, zorder=2)


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()

    # svg_content = plotter.create_themed_plot(
    #     save_name="example_question_regions", plot_func=plot_example_question_regions
    # )

    svg_content = plotter.create_themed_plot(
        save_name="example_question_regions4", plot_func=plot_example_question_regions
    )
