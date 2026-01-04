import numpy as np
import matplotlib.pyplot as plt


def plot_margin_principle(ax=None, color_map=None):
    """
    Create a clean, blog-friendly visualization of the maximum margin principle
    for Support Vector Machines (SVM).
    """
    # Generate synthetic data
    np.random.seed(42)
    n_points = 12

    # Class 1 (bottom-right)
    x1 = np.random.uniform(-2.5, 0, n_points)
    y1 = -3 + np.random.normal(0, 0.8, n_points)
    additional_x1 = np.random.uniform(-5, -2.5, n_points // 2)
    additional_y1 = np.random.uniform(-5, -2, n_points // 2)
    x1 = np.concatenate([x1, additional_x1])
    y1 = np.concatenate([y1, additional_y1])

    # Class 2 (top-left)
    x2 = np.random.uniform(2.5, 5, n_points)
    y2 = np.random.uniform(2, 5, n_points)
    additional_x2 = np.random.uniform(5, 7.5, n_points // 2)
    additional_y2 = np.random.uniform(4, 7, n_points // 2)
    x2 = np.concatenate([x2, additional_x2])
    y2 = np.concatenate([y2, additional_y2])

    # Plot data points
    ax.scatter(
        x1,
        y1,
        c=color_map["c1"],
        s=50,
        alpha=0.7,
        label="Class 1",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.scatter(
        x2,
        y2,
        c=color_map["c2"],
        s=50,
        alpha=0.7,
        label="Class 2",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    # Create decision boundary and margins
    x_range = np.array([-10, 10])

    # Decision boundary: y = -x (negative slope)
    slope = -1
    intercept = 1
    boundary = slope * x_range + intercept

    # Margins: parallel lines
    margin_width = 2
    margin_up = slope * x_range + (intercept + margin_width * np.sqrt(1 + slope**2))
    margin_down = slope * x_range + (intercept - margin_width * np.sqrt(1 + slope**2))

    # Plot decision boundary and margins
    ax.plot(
        x_range,
        boundary,
        "-",
        color="black",
        linewidth=2,
        label="Decision Boundary",
        zorder=2,
    )
    ax.plot(
        x_range,
        margin_up,
        "--",
        color="gray",
        linewidth=1.5,
        label="Margin",
        zorder=2,
        alpha=0.7,
    )
    ax.plot(
        x_range, margin_down, "--", color="gray", linewidth=1.5, zorder=2, alpha=0.7
    )

    # Add margin point and annotation
    margin_point_x = -2.5
    margin_point_y = slope * margin_point_x + (
        intercept - margin_width * np.sqrt(1 + slope**2)
    )
    ax.scatter(
        margin_point_x,
        margin_point_y,
        c="gray",
        s=100,
        alpha=0.9,
        edgecolor="black",
        linewidth=1.5,
        zorder=4,
    )
    ax.annotate(
        "margin point",
        xy=(margin_point_x, margin_point_y),
        xytext=(margin_point_x - 1, margin_point_y - 1),
        fontsize=9,
        ha="right",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="gray"),
    )

    # Fill the regions
    ax.fill_between(x_range, margin_down, -10, color=color_map["c1"], alpha=0.1)
    ax.fill_between(x_range, margin_up, 10, color=color_map["c2"], alpha=0.1)

    # Customize plot appearance
    ax.set_title("Maximum Margin Principle", fontsize=12, pad=15)
    ax.set_xlabel(r"$x_1$", fontsize=10)
    ax.set_ylabel(r"$x_2$", fontsize=10)

    # Clean legend
    ax.legend(frameon=True, framealpha=0.9, loc="upper left", fontsize=9)

    # Set equal aspect ratio and limits
    ax.set_aspect("equal")
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-7.5, 7.5)


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()
    svg_content = plotter.create_themed_plot(
        save_name="margin_principle", plot_func=plot_margin_principle
    )
