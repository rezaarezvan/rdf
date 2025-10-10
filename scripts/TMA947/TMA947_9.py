import numpy as np
import matplotlib.pyplot as plt


def plot_example_polyhedron_regions(ax=None, color_map=None):
    """
    Create a clean visualization of the feasible set of a LP (polyhedron),

    Consider the problem to,
    minimize x_2
    s.t. x_1 - x_2 <= 1
         x_1 + x_2 >= 1
         x_1, x_2 >= 0

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Define the domain
    x_min, x_max = -0.5, 3.5
    y_min, y_max = -0.5, 3.5

    # Define the constraint lines
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = np.linspace(y_min, y_max, 400)
    y1 = x_vals - 1  # From x1 - x2 <= 1  => x2 >= x1 - 1
    y2 = -x_vals + 1  # From x1 + x2 >= 1 => x2 >= -x1 + 1
    y3 = np.zeros_like(x_vals)  # x2 >= 0
    x3 = np.zeros_like(x_vals)  # x1 >= 0
    y4 = np.full_like(x_vals, y_max)  # x2 <= y_max
    x4 = np.full_like(x_vals, x_max)  # x1 <= x_max

    # Plot the constraint lines
    ax.plot(
        x_vals, y1, color="black", linewidth=2, label=r"$x_1 - x_2 \leq 1$", zorder=4
    )
    ax.plot(
        x_vals, y2, color="black", linewidth=2, label=r"$x_1 + x_2 \geq 1$", zorder=4
    )
    ax.plot(x_vals, y3, color="black", linewidth=2, label=r"$x_2 \geq 0$", zorder=4)
    ax.plot(x3, x_vals, color="black", linewidth=2, label=r"$x_1 \geq 0$", zorder=4)
    ax.plot(x_vals, y4, color="black", linewidth=1, linestyle="dashed", zorder=2)
    ax.plot(x4, x_vals, color="black", linewidth=1, linestyle="dashed", zorder=2)

    # Fill the feasible region
    ax.fill_between(
        x_vals,
        np.maximum(y1, y2, y3),
        y_max,
        where=(x_vals >= 0) & (y1 <= y_max) & (y2 <= y_max),
        color=color_map["c8"],
        alpha=0.5,
        zorder=1,
        label="Feasible Region",
    )

    # Mark the optimal point (1,0)
    ax.scatter(
        1,
        0,
        s=120,
        color="red",
        edgecolor="white",
        linewidth=2,
        zorder=7,
        label="Optimal Point (1,0)",
        marker="o",
    )
    # Customize plot appearance
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3, linestyle="-", zorder=0)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add coordinate axes
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5, zorder=2)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5, zorder=2)


def plot_example_polyhedron_regions_and_negative_gradient(ax=None, color_map=None):
    """
    Create a clean visualization of the feasible set of a LP (polyhedron),

    Consider the problem to,
    minimize x_2
    s.t. x_1 - x_2 <= 1
         x_1 + x_2 >= 1
         x_1, x_2 >= 0

    But, in this plot also add the negative gradient direction of the objective function at some point in the feasible region.

    To get the idea across that, if we just push/walk in this direction, we will eventually reach an extremal point.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Define the domain
    x_min, x_max = -0.5, 3.5
    y_min, y_max = -0.5, 3.5

    # Define the constraint lines
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = np.linspace(y_min, y_max, 400)
    y1 = x_vals - 1  # From x1 - x2 <= 1  => x2 >= x1 - 1
    y2 = -x_vals + 1  # From x1 + x2 >= 1 => x2 >= -x1 + 1
    y3 = np.zeros_like(x_vals)  # x2 >= 0
    x3 = np.zeros_like(x_vals)  # x1 >= 0
    y4 = np.full_like(x_vals, y_max)  # x2 <= y_max
    x4 = np.full_like(x_vals, x_max)  # x1 <= x_max

    # Plot the constraint lines
    ax.plot(
        x_vals, y1, color="black", linewidth=2, label=r"$x_1 - x_2 \leq 1$", zorder=4
    )
    ax.plot(
        x_vals, y2, color="black", linewidth=2, label=r"$x_1 + x_2 \geq 1$", zorder=4
    )
    ax.plot(x_vals, y3, color="black", linewidth=2, label=r"$x_2 \geq 0$", zorder=4)
    ax.plot(x3, x_vals, color="black", linewidth=2, label=r"$x_1 \geq 0$", zorder=4)
    ax.plot(x_vals, y4, color="black", linewidth=1, linestyle="dashed", zorder=2)
    ax.plot(x4, x_vals, color="black", linewidth=1, linestyle="dashed", zorder=2)

    # Fill the feasible region
    ax.fill_between(
        x_vals,
        np.maximum(y1, y2, y3),
        y_max,
        where=(x_vals >= 0) & (y1 <= y_max) & (y2 <= y_max),
        color=color_map["c8"],
        alpha=0.5,
        zorder=1,
        label="Feasible Region",
    )

    # Mark some point in the feasible region (1, 3/2)
    ax.scatter(
        1,
        1.5,
        s=100,
        color=color_map["c2"],
        edgecolor="white",
        linewidth=1.5,
        zorder=7,
        label="Feasible Point (2,1)",
        marker="o",
    )

    # dashed horizontal line from through (1,1.5) (y = 1.5)
    ax.hlines(
        1.5,
        0,
        2.5,
        colors=color_map["c2"],
        linestyles="dashed",
        linewidth=1.5,
        zorder=3,
    )

    # Plot the negative gradient direction of the objective function at (2,1)
    grad = np.array([0, 1])  # Gradient of x2 is [0, 1]
    neg_grad = -grad / np.linalg.norm(grad)  # Normalize
    start_point = np.array([1, 1.5])
    ax.quiver(
        start_point[0],
        start_point[1],
        neg_grad[0],
        neg_grad[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="black",
        alpha=0.8,
        width=0.005,
        zorder=8,
        label="Negative Gradient Direction",
    )

    # Mark the optimal point (1,0)
    ax.scatter(
        1,
        0,
        s=120,
        color="red",
        edgecolor="white",
        linewidth=2,
        zorder=7,
        label="Optimal Point (1,0)",
        marker="o",
    )
    # Customize plot appearance
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3, linestyle="-", zorder=0)
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Add coordinate axes
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5, zorder=2)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5, zorder=2)


def plot_example_polyhedron_regions2(ax=None, color_map=None):
    """
    Create a clean visualization of the feasible set of the polyhedron,

    Consider the polyhedron defined by,
    -2x_1 + x_2 <= 1
    x_1 - x_2 <= 1
    x_1, x_2 >= 0

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Define the domain
    x_min, x_max = -0.5, 3.5
    y_min, y_max = -0.5, 3.5

    # Define the constraint lines
    x_vals = np.linspace(x_min, x_max, 400)
    y1 = 2 * x_vals + 1  # From -2x1 + x2 <= 1  => x2 <= 2x1 + 1
    y2 = x_vals - 1  # From x1 - x2 <= 1  => x2 >= x1 - 1
    y3 = np.zeros_like(x_vals)  # x2 >= 0
    x3 = np.zeros_like(x_vals)  # x1 >= 0
    y4 = np.full_like(x_vals, y_max)  # x2 <= y_max
    x4 = np.full_like(x_vals, x_max)  # x1 <= x_max

    # Plot the constraint lines
    ax.plot(
        x_vals, y1, color="black", linewidth=2, label=r"$-2x_1 + x_2 \leq 1$", zorder=4
    )
    ax.plot(
        x_vals, y2, color="black", linewidth=2, label=r"$x_1 - x_2 \leq 1$", zorder=4
    )
    ax.plot(x_vals, y3, color="black", linewidth=2, label=r"$x_2 \geq 0$", zorder=4)
    ax.plot(x3, x_vals, color="black", linewidth=2, label=r"$x_1 \geq 0$", zorder=4)
    ax.plot(x_vals, y4, color="black", linewidth=1, linestyle="dashed", zorder=2)
    ax.plot(x4, x_vals, color="black", linewidth=1, linestyle="dashed", zorder=2)

    # Fill the feasible region
    ax.fill_between(
        x_vals,
        np.maximum(y2, y3),
        np.minimum(y1, y_max),
        where=(x_vals >= 0),
        color=color_map["c8"],
        alpha=0.5,
        zorder=1,
        label="Feasible Region",
    )
    # Mark the vertices of the polyhedron
    vertices = np.array([[0, 0], [0, 1], [1, 0]])
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        s=100,
        color=color_map["c1"],
        edgecolor="white",
        linewidth=1.5,
        zorder=7,
        label="Vertices",
        marker="o",
    )
    # Customize plot appearance
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3, linestyle="-", zorder=0)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Add coordinate axes
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5, zorder=2)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5, zorder=2)


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()

    # svg_content = plotter.create_themed_plot(
    #     save_name="example_polyhedron_regions", plot_func=plot_example_polyhedron_regions
    # )

    # svg_content = plotter.create_themed_plot(
    #     save_name="example_polyhedron_regions_and_negative_gradient", plot_func=plot_example_polyhedron_regions_and_negative_gradient
    # )

    svg_content = plotter.create_themed_plot(
        save_name="example_polyhedron_regions2",
        plot_func=plot_example_polyhedron_regions2,
    )
