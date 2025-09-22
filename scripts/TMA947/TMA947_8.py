import numpy as np
import matplotlib.pyplot as plt


def plot_example_question_regions(ax=None, color_map=None):
    """
    Create a clean visualization of shaded regions for example question,

    Consider the problem to,

    minimize x_1^2 + 1/100 x_2^2,
    subject to -x_1^2 - x_2^2 + 1 <= 0

    Plot the vector field of the objective function and the feasible region (the unit disk).

    Also plot the optimality points [0, +- 1]^T and [+-1, 0]^T, the latter are KKT points but not optimal.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Define the domain
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5

    # Create a finer grid for the background gradient
    x_fine = np.linspace(x_min, x_max, 200)
    y_fine = np.linspace(y_min, y_max, 200)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

    # Create a coarser grid for the vector field
    x_coarse = np.linspace(x_min, x_max, 20)
    y_coarse = np.linspace(y_min, y_max, 20)
    X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)

    # Define the objective function and its gradient
    def f(X, Y):
        return X**2 + (1/100) * Y**2

    def grad_f(X, Y):
        dfdx1 = 2 * X
        dfdx2 = (2/100) * Y
        return dfdx1, dfdx2

    def f_iso(X, Y):
        return X**2 + Y**2

    def grad_f_iso(X, Y):
        dfdx1 = 2 * X
        dfdx2 = 2 * Y
        return dfdx1, dfdx2

    # Compute the objective function values for background
    # Use isotropic version for radial appearance
    Z = f_iso(X_fine, Y_fine)

    # Plot the objective function as a color background
    ax.contourf(X_fine, Y_fine, Z, levels=50,
                cmap='RdYlGn_r', alpha=0.7, zorder=1)

    # Compute the gradient at each point in the coarse grid
    # Try the isotropic version first to match the radial pattern
    U, V = grad_f_iso(X_coarse, Y_coarse)  # This should give radial pattern

    # Uncomment the next line to use the actual anisotropic function:
    # U, V = grad_f(X_coarse, Y_coarse)  # This gives elliptical pattern

    # Normalize the gradient vectors for better visualization
    N = np.sqrt(U**2 + V**2)

    # Plot the vector field of the negative gradient (pointing inward like original)
    ax.quiver(
        X_coarse,
        Y_coarse,
        -U,
        -V,
        N,  # Color by magnitude
        cmap='viridis',
        scale=50,
        width=0.004,
        headwidth=4,
        headlength=5,
        alpha=0.8,
        zorder=3,
    )

    # Define the constraint boundary (unit circle)
    theta = np.linspace(0, 2 * np.pi, 400)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    # Plot the constraint boundary
    ax.plot(circle_x, circle_y, color='white', linewidth=3, zorder=4)
    ax.plot(circle_x, circle_y, color='black', linewidth=1.5, zorder=5)

    # Mark the optimal points [0, ±1] and KKT points [±1, 0]
    optimal_points = np.array([[0, 1], [0, -1]])
    kkt_points = np.array([[1, 0], [-1, 0]])

    ax.scatter(
        optimal_points[:, 0],
        optimal_points[:, 1],
        s=120,
        color='red',
        edgecolor='white',
        linewidth=2,
        zorder=7,
        label="Optimal Points",
        marker='o'
    )

    ax.scatter(
        kkt_points[:, 0],
        kkt_points[:, 1],
        s=120,
        color='limegreen',
        edgecolor='white',
        linewidth=2,
        zorder=7,
        label="KKT Points (Not Optimal)",
        marker='s'
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
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5, zorder=2)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.5, zorder=2)


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()

    svg_content = plotter.create_themed_plot(
        save_name="example_question_regions5", plot_func=plot_example_question_regions
    )
