import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Rectangle, Polygon

from rdf import figure


@figure("sine_animated", animation="draw", duration=3.0, loop=True)
def animate_sine_wave(ax, color_map):
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, color=color_map["c1"], linewidth=2)
    ax.set_title("Animated Sine Wave", fontsize=12)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("sin(x)", fontsize=10)
    ax.grid(True, alpha=0.15)


# Multiple line animation


@figure("multiple_functions", animation="draw", duration=4.0, loop=True)
def animate_multiple_functions(ax, color_map):
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), color=color_map["c1"], linewidth=2, label="sin(x)")
    ax.plot(x, np.cos(x), color=color_map["c2"], linewidth=2, label="cos(x)")
    ax.plot(
        x,
        np.sin(x) * np.cos(x),
        color=color_map["c3"],
        linewidth=2,
        label="sin(x)cos(x)",
    )
    ax.set_title("Multiple Functions", fontsize=12)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.grid(True, alpha=0.15)
    ax.legend()


# Scatter plot animation


@figure("scatter_fade", animation="draw", duration=2.0, loop=True)
def animate_scatter(ax, color_map):
    np.random.seed(42)
    x = np.random.rand(50)
    y = np.random.rand(50)
    sizes = np.random.rand(50) * 200
    ax.scatter(x, y, s=sizes, color=color_map["c1"], alpha=0.7)
    ax.set_title("Scatter Plot", fontsize=12)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.grid(True, alpha=0.15)


# Bar chart animation


@figure("bar_chart_fade", animation="draw", duration=2.5, loop=True)
def animate_bar_chart(ax, color_map):
    categories = ["A", "B", "C", "D", "E"]
    values = [25, 40, 30, 55, 15]
    colors = [color_map[f"c{i + 1}"] for i in range(len(categories))]
    ax.bar(categories, values, color=colors)
    ax.set_title("Bar Chart", fontsize=12)
    ax.set_ylabel("Value", fontsize=10)
    ax.grid(True, alpha=0.15, axis="y")


# Shape animation


@figure("shapes_pulse", animation="draw", duration=2.0, loop=True)
def animate_shapes(ax, color_map):
    # Add various shapes
    circle = Circle((0.5, 0.5), 0.2, color=color_map["c1"], alpha=0.7)
    square = Rectangle((0.1, 0.1), 0.2, 0.2, color=color_map["c2"], alpha=0.7)
    triangle = Polygon(
        [[0.6, 0.1], [0.8, 0.1], [0.7, 0.3]], color=color_map["c3"], alpha=0.7
    )

    ax.add_patch(circle)
    ax.add_patch(square)
    ax.add_patch(triangle)

    ax.set_title("Geometric Shapes", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")


# 3D surface plot animation


@figure("surface_3d", animation="draw", duration=5.0, loop=True, is_3d=True)
def animate_3d_surface(ax, color_map):
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)

    ax.set_title("3D Surface Plot", fontsize=12)
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z", fontsize=10)
    ax.view_init(30, 45)

    return surf


@figure("beta_beta_squared", animation="draw", duration=2.0, loop=True)
def plot_beta_beta_squared(ax, color_map):
    """
    Plot a clean, blog-friendly visualization of a Brownian motion and its squared version.
    Starting from the SDE (x(t) = B(t)),
    dx(t) = dB(t)
    we can derive an SDE for phi(t) = B^2(t),
    d(B^2(t)) = 2B(t)dB(t) + dt
    """
    # Set up the plot bounds with padding
    x_min, x_max = 0, 2
    padding = 0.2  # Add padding for better appearance
    # Create the y values (Brownian motion, Euler-Maruyama)
    dt = 0.01
    t = np.arange(0, 1, dt)
    B = np.zeros_like(t)
    B[0] = 0
    for i in range(1, len(t)):
        B[i] = B[i - 1] + np.random.normal(0, np.sqrt(dt))
    # Create the squared Brownian motion
    B_squared = B**2
    y_min, y_max = (
        min(np.min(B), np.min(B_squared)) - padding,
        max(np.max(B), np.max(B_squared)) + padding,
    )
    x_max = y_max + padding
    # Plot the original function
    ax.plot(t, B, color=color_map["c8"], linewidth=2)
    # Plot the squared Brownian motion
    ax.plot(t, B_squared, color=color_map["c7"], linewidth=2)
    # Customize plot appearance
    ax.set_title(f"Brownian motion and its squared version", fontsize=12, pad=15)
    ax.set_xlabel(r"$t$", fontsize=10)
    ax.set_ylabel(r"$\beta(t), \beta^2(t)$", fontsize=10)
    # Set axis limits with padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    # Set aspect ratio to be equal for proper visualization
    ax.set_aspect("equal")
    # Legend
    ax.legend(
        ["$\\beta(t)$", "$\\beta^2(t)$"],
        frameon=True,
        framealpha=0.9,
        loc="upper right",
        fontsize=9,
        bbox_to_anchor=(0.98, 0.98),
    )


@figure("subfig_example", animation="draw", duration=3.0, loop=True)
def plot_subfig_example(ax, color_map):
    """
    Example of a subfigure with multiple plots.
    """
    # Create a grid of subplots
    # RDF will make fig so do not create fig here
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    # Plotting in each subplot
    animate_sine_wave(ax1, color_map)
    animate_multiple_functions(ax2, color_map)
    animate_scatter(ax3, color_map)
    animate_bar_chart(ax4, color_map)

    plt.tight_layout()
