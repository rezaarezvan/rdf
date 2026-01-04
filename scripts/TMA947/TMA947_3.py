import numpy as np
import matplotlib.pyplot as plt


def plot_convex_function_definition(ax=None, color_map=None):
    """
    Create a clean, blog-friendly visualization of the definition of a convex function.

    Shows f(λx₁ + (1-λ)x₂) ≤ λf(x₁) + (1-λ)f(x₂) for λ ∈ (0,1)

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Set up the domain
    x_min, x_max = -2, 4
    padding = 0.3

    # Create smooth x values for the function
    x = np.linspace(x_min, x_max, 200)

    # Define a convex function: f(x) = 0.3(x - 1)² + 0.5
    def f(x_val):
        return 0.3 * (x_val - 1) ** 2 + 0.5

    y = f(x)

    # Choose two points x₁ and x₂
    x1 = 0.0
    x2 = 2.5
    f_x1 = f(x1)
    f_x2 = f(x2)

    y_min, y_max = min(y) - 0.5, max(y) + 0.6

    ax.plot(x, y, color=color_map["c8"], linewidth=2.5, label=r"$f(x)$", zorder=3)

    # Plot the secant line between the two points
    secant_x = np.linspace(x1, x2, 100)
    secant_y = f_x1 + (secant_x - x1) * (f_x2 - f_x1) / (x2 - x1)
    ax.plot(
        secant_x,
        secant_y,
        "--",
        color=color_map["c8"],
        linewidth=2,
        alpha=0.7,
        label=r"$\lambda f(x_1) + (1-\lambda)f(x_2)$",
        zorder=2,
    )

    # Plot lines from graph to x-axis for the points
    ax.vlines(
        x1, -1, f_x1, colors=color_map["c8"], linestyles="dotted", zorder=1, linewidth=2
    )
    ax.vlines(
        x2, -1, f_x2, colors=color_map["c8"], linestyles="dotted", zorder=1, linewidth=2
    )

    # Mark the two base points
    ax.scatter(
        x1,
        f_x1,
        s=80,
        color=color_map["c8"],
        zorder=4,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.scatter(
        x2,
        f_x2,
        s=80,
        color=color_map["c8"],
        zorder=4,
        edgecolor="white",
        linewidth=1.5,
    )

    # Add labels for the points
    ax.text(x1, f_x1 + 0.15, r"$f(x_1)$", fontsize=11, ha="center", va="bottom")
    ax.text(x2, f_x2 + 0.15, r"$f(x_2)$", fontsize=11, ha="center", va="bottom")
    ax.text(x1, -0.3, r"$x_1$", fontsize=11, ha="right", va="top")
    ax.text(x2, -0.3, r"$x_2$", fontsize=11, ha="left", va="top")

    # Customize plot appearance
    ax.set_xlabel(r"$x$", fontsize=10)
    ax.set_ylabel(r"$f(x)$", fontsize=10)

    # Set axis limits with padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min, y_max)

    # Clean legend
    ax.legend(frameon=True, framealpha=0.9, loc="upper right", fontsize=9)


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()
    svg_content = plotter.create_themed_plot(
        name="convex_function_definition",
        plot_func=plot_convex_function_definition,
    )
