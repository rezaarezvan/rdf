import numpy as np
import matplotlib.pyplot as plt


def plot_piecewise_function_analysis(ax=None, color_map=None):
    """
    Create a clean visualization of a piecewise function showing:
    - Jump discontinuity with filled/hollow circles
    - Non-differentiable point (sharp corner)
    - Stationary points (local max/min)
    - Clear domain boundaries

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """

    # Define the domain
    x_min, x_max = -1, 6

    # Create fine-grained x values for smooth curves
    x1 = np.linspace(x_min, 1, 200)  # First piece: x < 1
    x2 = np.linspace(1.01, 3, 200)  # Second piece: 1 < x < 3
    x3 = np.linspace(3, 4.5, 200)  # Third piece: 3 ≤ x < 4.5
    x4 = np.linspace(4.5, x_max, 200)  # Fourth piece: x ≥ 4.5

    # Define piecewise function
    # Piece 1: f(x) = -0.5x² + 2x + 1 for x < 1
    y1 = -0.5 * x1**2 + 2 * x1 + 1

    # Piece 2: f(x) = x² - 4x + 5 for 1 < x < 3 (has minimum at x=2)
    y2 = x2**2 - 4 * x2 + 5

    # Piece 3: f(x) = |x - 3| + 1 for 3 ≤ x < 4.5 (sharp corner at x=3)
    y3 = np.abs(x3 - 3) + 1

    # Piece 4: f(x) = -0.3(x-5)² + 2.5 for x ≥ 4.5 (has maximum at x=5)
    y4 = -0.3 * (x4 - 5) ** 2 + 2.5

    # Plot the piecewise function
    ax.plot(x1, y1, color=color_map["c8"], linewidth=2.5, zorder=3)
    ax.plot(x2, y2, color=color_map["c8"], linewidth=2.5, zorder=3)
    ax.vlines(x=3, ymin=1, ymax=2, color=color_map["c8"], linewidth=2.5, zorder=3)
    ax.plot(x3, y3, color=color_map["c8"], linewidth=2.5, zorder=3)
    ax.plot(x4, y4, color=color_map["c8"], linewidth=2.5, zorder=3)

    # Mark discontinuity at x = 1
    # Left limit: f(1⁻) = -0.5(1)² + 2(1) + 1 = 2.5
    # Right limit: f(1⁺) = (1)² - 4(1) + 5 = 2
    left_limit_1 = -0.5 * 1**2 + 2 * 1 + 1  # 2.5
    right_limit_1 = 1**2 - 4 * 1 + 5  # 2

    # Hollow circle at (1, 2.5) - function value included
    ax.scatter(
        1,
        right_limit_1,
        s=80,
        color=color_map["c8"],
        edgecolor=color_map["white"],
        linewidth=2,
        zorder=5,
    )

    # Filled circle at (1, 2) - function value not included
    ax.scatter(
        1,
        left_limit_1,
        s=80,
        facecolors="none",
        edgecolor=color_map["c8"],
        linewidth=2,
        zorder=5,
    )

    # Mark non-differentiable point at x = 3 (sharp corner)
    ax.scatter(
        3,
        1,
        s=80,
        color=color_map["c8"],
        edgecolor=color_map["white"],
        linewidth=2,
        zorder=5,
        marker="D",
    )

    # Mark stationary points
    # Local minimum at x = 2: f(2) = 4 - 8 + 5 = 1
    ax.scatter(
        2,
        1,
        s=80,
        color=color_map["c1"],
        edgecolor=color_map["white"],
        linewidth=2,
        zorder=5,
        marker="v",
    )

    # Local maximum at x = 5: f(5) = -0.3(0)² + 2.5 = 2.5
    ax.scatter(
        5,
        2.5,
        s=80,
        color=color_map["c1"],
        edgecolor=color_map["white"],
        linewidth=2,
        zorder=5,
        marker="^",
    )

    # Add vertical dashed lines at critical points
    critical_points = [1, 2, 5]
    for x_crit in critical_points:
        ax.axvline(
            x=x_crit,
            color=color_map["grey"],
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            zorder=1,
        )

    # Add annotations
    ax.annotate(
        "Jump\nDiscontinuity",
        xy=(1, 2.52),
        xytext=(0.2, 3.5),
        arrowprops=dict(arrowstyle="->", color=color_map["grey"], alpha=0.7),
        fontsize=9,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=color_map["white"], alpha=0.8),
    )

    ax.annotate(
        "Stationary\nPoint",
        xy=(2, 1),
        xytext=(1.2, 0.2),
        arrowprops=dict(arrowstyle="->", color=color_map["grey"], alpha=0.7),
        fontsize=9,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=color_map["white"], alpha=0.8),
    )

    ax.annotate(
        "Sharp Corner\n(Non-differentiable)",
        xy=(3, 0.97),
        xytext=(3.8, 0.3),
        arrowprops=dict(arrowstyle="->", color=color_map["grey"], alpha=0.7),
        fontsize=9,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=color_map["white"], alpha=0.8),
    )

    ax.annotate(
        "Stationary\nPoint",
        xy=(5, 2.5),
        xytext=(5.5, 3.2),
        arrowprops=dict(arrowstyle="->", color=color_map["grey"], alpha=0.7),
        fontsize=9,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=color_map["white"], alpha=0.8),
    )

    ax.annotate(
        "Boundary points",
        xy=(x_min, -0.3),
        xytext=(x_min + 1, -0.8),
        arrowprops=dict(arrowstyle="->", color=color_map["grey"], alpha=0.7),
        fontsize=9,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=color_map["white"], alpha=0.8),
    )
    ax.annotate(
        "Boundary points",
        xy=(x_max, -0.3),
        xytext=(x_max - 1, -0.8),
        arrowprops=dict(arrowstyle="->", color=color_map["grey"], alpha=0.7),
        fontsize=9,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=color_map["white"], alpha=0.8),
    )

    # Add domain boundaries
    ax.axvline(x=x_min, color=color_map["black"], linewidth=2, alpha=0.8)
    ax.axvline(x=x_max, color=color_map["black"], linewidth=2, alpha=0.8)

    # Customize plot appearance
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)

    # Set limits with padding
    ax.set_xlim(x_min - 0.2, x_max + 0.2)
    ax.set_ylim(-0.5, 4)

    # Add subtle grid
    ax.grid(True, alpha=0.15, linestyle="-", zorder=0)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=10)


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


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()
    # svg_content = plotter.create_themed_plot(
    #     save_name="piecewise_function_analysis",
    #     plot_func=plot_piecewise_function_analysis
    # )

    svg_content = plotter.create_themed_plot(
        save_name="example_question_regions", plot_func=plot_example_question_regions
    )
