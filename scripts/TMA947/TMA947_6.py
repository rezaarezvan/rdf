import numpy as np
import matplotlib.pyplot as plt


def plot_example_question_regions(ax=None, color_map=None):
    """
    Create a clean visualization of shaded regions for example questions,
    Let S = {x in RR^2 | (x_1 - 1)^2 + x_2^2 <= 1 and (x_1 + 1)^2 + x_2^2 <= 1}
    Thus, g_1(x) => (x_1 - 1)^2 + x_2^2 - 1 <= 0
          g_2(x) => (x_1 + 1)^2 + x_2^2 - 1 <= 0

    Let bar(x) = [0 0]^T

    Plot the feasible region S and bar(x) on the axis.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Define the domain
    x_min, x_max = -3, 3
    y_min, y_max = -2, 2

    # Create a grid of points
    x = np.linspace(x_min, x_max, 400)
    y = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(x, y)

    # Define the inequalities
    g1 = (X - 1) ** 2 + Y**2 - 1  # Circle centered at (1,0) with radius 1
    g2 = (X + 1) ** 2 + Y**2 - 1  # Circle centered at (-1,0) with radius 1

    # Create a mask for the feasible region S
    feasible_region = np.logical_and(g1 <= 0, g2 <= 0)

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
    ax.contour(X, Y, g1, levels=[0], colors=color_map["c8"], linewidths=2.5, zorder=3)
    ax.contour(X, Y, g2, levels=[0], colors=color_map["c8"], linewidths=2.5, zorder=3)

    # Mark the point bar(x) = [0, 0]^T
    ax.scatter(
        0, 0, s=100, color=color_map["c1"], edgecolor="white", linewidth=2, zorder=5
    )

    # Add annotations
    ax.annotate(
        r"$ \bar{x} = [0, 0]^T $",
        xy=(0, 0),
        xytext=(-0.5, -1.2),
        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.annotate(
        r"$g_1(x) \leq 0$",
        xy=(2, 0),
        xytext=(2.5, 1.5),
        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.annotate(
        r"$g_2(x) \leq 0$",
        xy=(-2, 0),
        xytext=(-2.5, 1.5),
        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Customize plot appearance
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_title("Feasible Region S", fontsize=14)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.15, linestyle="-", zorder=0)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Add x_1 and x_2 axes
    ax.axhline(0, color=color_map["black"], linewidth=1, alpha=0.8, zorder=2)
    ax.axvline(0, color=color_map["black"], linewidth=1, alpha=0.8, zorder=2)


def plot_example_question_regions2(ax=None, color_map=None):
    """
    Create a clean visualization of shaded regions for example questions,

    min -x_1 + x_2
    s.t. x_1^2 + x_2^2 - 1 <= 0
         -x_2 <= 0

    Consider the points x1 = [0 0]^T, x2 = [-1 0]^T, and x3 = [1 0]^T

    Plot cost function, feasible region, and the points on the axis.

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
    g1 = X**2 + Y**2 - 1  # Circle centered at (0,0) with radius 1
    g2 = -Y  # Line y = 0

    # Create a mask for the feasible region
    feasible_region = np.logical_and(g1 <= 0, g2 <= 0)

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
    ax.contour(X, Y, g1, levels=[0], colors=color_map["c8"], linewidths=2.5, zorder=3)
    ax.axhline(0, color=color_map["c8"], linewidth=2.5, zorder=3)

    # Mark the points x1 = [0 0]^T, x2 = [-1 0]^T, and x3 = [1 0]^T
    points = {
        "x1": (0, 0),
        "x2": (-1, 0),
        "x3": (1, 0),
    }
    for label, (px, py) in points.items():
        ax.scatter(
            px,
            py,
            s=100,
            color=color_map["c1"],
            edgecolor="white",
            linewidth=2,
            zorder=5,
        )
        ax.annotate(
            rf"${label}$",
            xy=(px, py),
            xytext=(px, py + 0.5),
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Draw (negative) gradient vectors of cost function and constraints at point x1
    # grad f = [-1, 1]^T => -grad f = [1, -1]^T,
    # grad g1 = [2x1, 2x2]^T => at (0,0) = [0, 0]^T
    # grad g2 = [0, -1]^T
    ax.quiver(
        0,
        0,
        1,
        -1,
        angles="xy",
        scale_units="xy",
        scale=3,
        color=color_map["c1"],
        zorder=6,
    )
    ax.annotate(r"$-\nabla f$", xy=(0.5, -0.5), fontsize=10, color=color_map["c1"])
    ax.quiver(
        0,
        0,
        0,
        -1,
        angles="xy",
        scale_units="xy",
        scale=3,
        color=color_map["c2"],
        zorder=6,
    )
    ax.annotate(r"$\nabla g_2$", xy=(0.1, -0.5), fontsize=10, color=color_map["c2"])
    ax.quiver(
        0,
        0,
        0,
        0,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=color_map["c3"],
        zorder=6,
    )
    ax.annotate(r"$\nabla g_1$", xy=(0.1, 0.1), fontsize=10, color=color_map["c3"])

    # x2
    # grad f = [-1, 1]^T => -grad f = [1, -1]^T,
    # grad g1 = [2x1, 2x2]^T => at (-1,0) = [-2, 0]^T
    # grad g2 = [0, -1]^T
    ax.quiver(
        -1,
        0,
        1,
        -1,
        angles="xy",
        scale_units="xy",
        scale=3,
        color=color_map["c1"],
        zorder=6,
    )
    ax.annotate(r"$-\nabla f$", xy=(-0.5, -0.5), fontsize=10, color=color_map["c1"])
    ax.quiver(
        -1,
        0,
        0,
        -1,
        angles="xy",
        scale_units="xy",
        scale=3,
        color=color_map["c2"],
        zorder=6,
    )
    ax.annotate(r"$\nabla g_2$", xy=(-0.9, -0.5), fontsize=10, color=color_map["c2"])
    ax.quiver(
        -1,
        0,
        -2,
        0,
        angles="xy",
        scale_units="xy",
        scale=3,
        color=color_map["c3"],
        zorder=6,
    )
    ax.annotate(r"$\nabla g_1$", xy=(-1.3, 0.1), fontsize=10, color=color_map["c3"])

    # x3
    # grad f = [-1, 1]^T => -grad f = [1, -1]^T,
    # grad g1 = [2x1, 2x2]^T => at (1,0) = [2, 0]^T
    # grad g2 = [0, -1]^T
    ax.quiver(
        1,
        0,
        1,
        -1,
        angles="xy",
        scale_units="xy",
        scale=3,
        color=color_map["c1"],
        zorder=6,
    )
    ax.annotate(r"$-\nabla f$", xy=(1.5, -0.5), fontsize=10, color=color_map["c1"])
    ax.quiver(
        1,
        0,
        0,
        -1,
        angles="xy",
        scale_units="xy",
        scale=3,
        color=color_map["c2"],
        zorder=6,
    )
    ax.annotate(r"$\nabla g_2$", xy=(0.9, -0.5), fontsize=10, color=color_map["c2"])
    ax.quiver(
        1,
        0,
        2,
        0,
        angles="xy",
        scale_units="xy",
        scale=3,
        color=color_map["c3"],
        zorder=6,
    )
    ax.annotate(r"$\nabla g_1$", xy=(1.3, 0.1), fontsize=10, color=color_map["c3"])

    # Customize plot appearance
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.15, linestyle="-", zorder=0)
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=10)
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
        save_name="example_question_regions2", plot_func=plot_example_question_regions2
    )
