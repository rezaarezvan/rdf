import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse, Path, PathPatch, Polygon, Patch


def plot_convex_nonconvex_sets(ax=None, color_map=None):
    """
    Create a clean, blog-friendly visualization of convex and non-convex sets
    showing the line segment property.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Create figure with two subplots side by side
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(1, 2, height_ratios=[1], width_ratios=[1, 1], wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # ---- Left subplot: Convex set ----

    # Create ellipse (convex set)
    ellipse = Ellipse(
        (0, 0),
        3,
        1.8,
        angle=15,
        facecolor=color_map["c8"],
        edgecolor=color_map["c8"],
        linewidth=2,
    )
    ax1.add_patch(ellipse)

    # Add two points inside the ellipse
    x1, y1 = -0.8, -0.2
    x2, y2 = 0.6, 0.3

    ax1.scatter(x1, y1, s=20, color="black", alpha=0.9, zorder=5)
    ax1.scatter(x2, y2, s=20, color="black", alpha=0.9, zorder=5)

    # Draw line segment between points
    ax1.plot(
        [x1, x2], [y1, y2], color="black", linewidth=2, zorder=4, label="Line segment"
    )

    # Add point labels
    ax1.text(x1 - 0.2, y1 - 0.2, "$x_1$", fontsize=12, ha="center")
    ax1.text(x2 + 0.2, y2 + 0.1, "$x_2$", fontsize=12, ha="center")

    # Customize first subplot
    ax1.set_title("Convex Set", fontsize=14, pad=15, fontweight="bold")
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect("equal")
    ax1.grid(False)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # ---- Right subplot: Non-convex set ----

    # Create crescent/C-shape (non-convex set)
    # Using parametric equations for a crescent
    theta_start = -2.5  # Start angle for C opening
    theta_end = 2.5  # End angle for C opening
    theta_c = np.linspace(theta_start, theta_end, 80)

    r_outer = 1.2
    x_outer_c = r_outer * np.cos(theta_c)
    y_outer_c = r_outer * np.sin(theta_c)

    r_inner = 0.7
    x_inner_c = r_inner * np.cos(theta_c)
    y_inner_c = r_inner * np.sin(theta_c)

    vertices = []
    codes = []

    for i in range(len(x_outer_c)):
        vertices.append((x_outer_c[i], y_outer_c[i]))
        codes.append(Path.LINETO if i > 0 else Path.MOVETO)

    vertices.append((x_inner_c[-1], y_inner_c[-1]))
    codes.append(Path.LINETO)

    for i in range(len(x_inner_c) - 1, -1, -1):
        vertices.append((x_inner_c[i], y_inner_c[i]))
        codes.append(Path.LINETO)

    codes.append(Path.CLOSEPOLY)
    vertices.append(vertices[0])

    path = Path(vertices, codes)
    patch = PathPatch(
        path, facecolor=color_map["c8"], edgecolor=color_map["c8"], linewidth=2
    )
    ax2.add_patch(patch)

    # Add two points in the crescent
    x1_nc, y1_nc = -0.5, 0.7
    x2_nc, y2_nc = -0.5, -0.7

    ax2.scatter(x1_nc, y1_nc, s=20, color="black", alpha=0.9, zorder=5)
    ax2.scatter(x2_nc, y2_nc, s=20, color="black", alpha=0.9, zorder=5)

    # Draw line segment between points (this will go outside the set)
    ax2.plot(
        [x1_nc, x2_nc],
        [y1_nc, y2_nc],
        color="black",
        linewidth=3,
        zorder=4,
        linestyle="--",
        alpha=0.8,
    )

    # Add point labels
    ax2.text(x1_nc - 0.2, y1_nc + 0.2, "$x_1$", fontsize=12, ha="center")
    ax2.text(x2_nc - 0.2, y2_nc - 0.2, "$x_2$", fontsize=12, ha="center")

    # Customize second subplot
    ax2.set_title("Non-convex Set", fontsize=14, pad=15, fontweight="bold")
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect("equal")
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])


def plot_unit_sphere_3d(ax=None, color_map=None):
    """
    Create a clean, blog-friendly visualization of a unit sphere in 3D.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Create 3D sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)

    # Parametric equations for unit sphere
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot surface
    ax.plot_surface(
        x,
        y,
        z,
        alpha=0.6,
        color=color_map["c8"],
        linewidth=0.5,
        edgecolors=color_map["c8"],
    )

    ax.set_box_aspect([1, 1, 1])

    # Set viewing angle
    ax.view_init(elev=20, azim=45)


def plot_circle_boundary(ax=None, color_map=None):
    """
    Create a clean, blog-friendly visualization of a circle boundary (hollow circle).

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Create circle boundary
    theta = np.linspace(0, 2 * np.pi, 200)
    radius = 1.0

    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)

    # Plot just the boundary
    ax.plot(
        x_circle,
        y_circle,
        color=color_map["c8"],
        linewidth=3,
        label="Unit Circle Boundary",
    )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")


def plot_sphere_and_circle(ax=None, color_map=None):
    """
    Create side-by-side visualization of unit sphere and circle boundary.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Create figure with two subplots
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)

    # 3D subplot for sphere
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    plot_unit_sphere_3d(ax1, color_map)

    # 2D subplot for circle
    ax2 = fig.add_subplot(gs[0, 1])
    plot_circle_boundary(ax2, color_map)


def plot_overlapping_sets(ax, color_map):
    ellipse1 = Ellipse(
        (0, 0),
        1.4,
        0.8,
        angle=0,
        facecolor=color_map["c8"],
        alpha=0.6,
        edgecolor=color_map["c8"],
        linewidth=2,
    )
    ellipse2 = Ellipse(
        (0.3, 0),
        1.4,
        0.8,
        angle=5,
        facecolor=color_map["c8"],
        alpha=0.4,
        edgecolor=color_map["c8"],
        linewidth=2,
    )

    ax.add_patch(ellipse1)
    ax.add_patch(ellipse2)

    # Show line segment stays inside
    ax.plot([-0.4, 0.4], [-0.2, 0.2], "black", linewidth=2)
    ax.scatter([-0.4, 0.4], [-0.2, 0.2], s=20, color="black", zorder=5)


def plot_disconnected_sets(ax, color_map):
    # Two separate ellipses
    ellipse1 = Ellipse(
        (-0.8, 0),
        0.8,
        0.6,
        angle=0,
        facecolor=color_map["c8"],
        alpha=0.6,
        edgecolor=color_map["c8"],
        linewidth=2,
    )
    ellipse2 = Ellipse(
        (0.8, 0),
        0.8,
        0.6,
        angle=0,
        facecolor=color_map["c8"],
        alpha=0.6,
        edgecolor=color_map["c8"],
        linewidth=2,
    )

    ax.add_patch(ellipse1)
    ax.add_patch(ellipse2)

    # Show line segment goes outside the union
    ax.plot([-0.8, 0.8], [0, 0], "black", linewidth=2, linestyle="--", alpha=0.8)
    ax.scatter([-0.8, 0.8], [0, 0], s=20, color="black", zorder=5)


def plot_sphere_halves(ax, color_map):
    """Helper function to plot two halves of a sphere in 3D."""
    # Create sphere coordinates
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot upper half
    z_upper = np.where(z >= 0, z, np.nan)
    ax.plot_surface(x, y, z_upper, alpha=0.7, color=color_map["c8"])

    # Plot lower half in different color
    z_lower = np.where(z <= 0, z, np.nan)
    ax.plot_surface(x, y, z_lower, alpha=0.7, color=color_map["c7"])

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=45)


def plot_three_convex_operations(ax=None, color_map=None):
    """
    Create three visualizations of convex set operations.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.4)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Overlapping
    ax2 = fig.add_subplot(gs[0, 1])  # Disconnected
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")  # 3D sphere halves

    # Plot each example
    plot_overlapping_sets(ax1, color_map)
    plot_disconnected_sets(ax2, color_map)
    plot_sphere_halves(ax3, color_map)

    # Set consistent styling for 2D subplots
    for ax in [ax1, ax2]:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.0)
        ax.set_aspect("equal")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])


def create_c_shape():
    """Create C-shape coordinates."""
    theta_start = -2.2
    theta_end = 2.2
    theta_c = np.linspace(theta_start, theta_end, 80)

    r_outer = 1.0
    x_outer_c = r_outer * np.cos(theta_c)
    y_outer_c = r_outer * np.sin(theta_c)

    r_inner = 0.5
    x_inner_c = r_inner * np.cos(theta_c)
    y_inner_c = r_inner * np.sin(theta_c)

    # Create vertices for C-shape
    vertices = []
    codes = []

    # Outer arc
    for i in range(len(x_outer_c)):
        vertices.append((x_outer_c[i], y_outer_c[i]))
        codes.append(Path.LINETO if i > 0 else Path.MOVETO)

    # Connect to inner arc
    vertices.append((x_inner_c[-1], y_inner_c[-1]))
    codes.append(Path.LINETO)

    # Inner arc (reversed)
    for i in range(len(x_inner_c) - 1, -1, -1):
        vertices.append((x_inner_c[i], y_inner_c[i]))
        codes.append(Path.LINETO)

    codes.append(Path.CLOSEPOLY)
    vertices.append(vertices[0])

    return Path(vertices, codes)


def plot_minimal_convex_hull(ax, color_map):
    c_path = create_c_shape()
    c_patch = PathPatch(
        c_path,
        facecolor=color_map["c8"],
        alpha=0.6,
        edgecolor=color_map["c8"],
        linewidth=2,
    )
    ax.add_patch(c_patch)

    # Fill in the "mouth" to show convex hull
    # Add the straight line segment that closes the C
    mouth_vertices = [
        (1.0 * np.cos(-2.2), 1.0 * np.sin(-2.2)),
        (1.0 * np.cos(2.2), 1.0 * np.sin(2.2)),
    ]

    ax.scatter(
        [v[0] for v in mouth_vertices],
        [v[1] for v in mouth_vertices],
        s=30,
        color="black",
        zorder=5,
    )

    # Draw the closing line prominently
    ax.plot(
        [mouth_vertices[0][0], mouth_vertices[1][0]],
        [mouth_vertices[0][1], mouth_vertices[1][1]],
        color="black",
        linewidth=2,
    )

    ax.text(0, -1.4, "Minimal Convex Set", ha="center", fontsize=12, fontweight="bold")


def plot_larger_convex_sets(ax, color_map):
    # Original C-shape
    c_path = create_c_shape()
    c_patch = PathPatch(
        c_path,
        facecolor=color_map["c8"],
        alpha=0.6,
        edgecolor=color_map["c8"],
        linewidth=2,
    )
    ax.add_patch(c_patch)

    # Larger convex sets (dotted)
    large_ellipse1 = Ellipse(
        (0, 0),
        3.2,
        2.8,
        angle=0,
        facecolor="none",
        edgecolor="gray",
        linewidth=2,
        linestyle="--",
        alpha=0.7,
    )
    ax.add_patch(large_ellipse1)

    large_ellipse2 = Ellipse(
        (0, 0),
        2.8,
        2.4,
        angle=15,
        facecolor="none",
        edgecolor="gray",
        linewidth=2,
        linestyle="--",
        alpha=0.7,
    )
    ax.add_patch(large_ellipse2)

    ax.text(0, -2, "Larger Convex Sets", ha="center", fontsize=12, fontweight="bold")


def plot_convex_combinations(ax, color_map):
    # Original C-shape
    c_path = create_c_shape()
    c_patch = PathPatch(
        c_path,
        facecolor=color_map["c8"],
        alpha=0.6,
        edgecolor=color_map["c8"],
        linewidth=2,
    )
    ax.add_patch(c_patch)

    # Show convex combinations with arrows
    arrow_pairs = [
        ((-0.3, 0.5), (0.7, 0.2)),
        ((-0.2, 0.6), (0.6, -0.4)),
    ]

    for start, end in arrow_pairs:
        ax.scatter(start[0], start[1], s=20, color="black", alpha=0.7, zorder=4)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5, alpha=0.7),
        )
        ax.scatter(end[0], end[1], s=20, color="black", alpha=0.7, zorder=4)

    ax.text(
        0, -1.4, "All Convex Combinations", ha="center", fontsize=12, fontweight="bold"
    )


def plot_convex_hull_construction(ax=None, color_map=None):
    """
    Create three visualizations of convex hull construction.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.4)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Minimal convex hull
    ax2 = fig.add_subplot(gs[0, 1])  # Larger convex sets
    ax3 = fig.add_subplot(gs[0, 2])  # Convex combinations

    # Plot each example
    plot_minimal_convex_hull(ax1, color_map)
    plot_larger_convex_sets(ax2, color_map)
    plot_convex_combinations(ax3, color_map)

    # Set consistent styling for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-2.0, 1.5)
        ax.set_aspect("equal")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_triangle_with_interior_point(ax, color_map):
    # Define extreme points (vertices of triangle)
    v1 = np.array([-1, -0.5])
    v2 = np.array([1, -0.5])
    v3 = np.array([0, 1])

    # Define v4 as interior point (convex combination)
    v4 = 0.4 * v1 + 0.3 * v2 + 0.3 * v3  # Convex combination

    # Draw triangle (polytope)
    triangle = Polygon(
        [v1, v2, v3],
        facecolor=color_map["c8"],
        alpha=0.3,
        edgecolor=color_map["c8"],
        linewidth=2,
    )
    ax.add_patch(triangle)

    # Plot extreme points
    ax.scatter(*v1, s=20, color="black", zorder=5)
    ax.scatter(*v2, s=20, color="black", zorder=5)
    ax.scatter(*v3, s=20, color="black", zorder=5)

    # Plot interior point v4
    ax.scatter(*v4, s=20, color="red", zorder=5, marker="s")

    # Add labels
    ax.text(
        v1[0] - 0.15, v1[1] - 0.15, "$v_1$", fontsize=12, ha="center", fontweight="bold"
    )
    ax.text(
        v2[0] + 0.15, v2[1] - 0.15, "$v_2$", fontsize=12, ha="center", fontweight="bold"
    )
    ax.text(v3[0], v3[1] + 0.15, "$v_3$", fontsize=12, ha="center", fontweight="bold")
    ax.text(
        v4[0] + 0.2,
        v4[1],
        "$v_4$",
        fontsize=12,
        ha="center",
        fontweight="bold",
        color="red",
    )

    # Show convex combination with dashed lines
    ax.plot(
        [v1[0], v4[0]], [v1[1], v4[1]], "gray", linestyle="--", alpha=0.6, linewidth=1
    )
    ax.plot(
        [v2[0], v4[0]], [v2[1], v4[1]], "gray", linestyle="--", alpha=0.6, linewidth=1
    )
    ax.plot(
        [v3[0], v4[0]], [v3[1], v4[1]], "gray", linestyle="--", alpha=0.6, linewidth=1
    )


def plot_triangle_extreme_only(ax, color_map):
    """Plot triangle with only extreme points."""
    from matplotlib.patches import Polygon

    # Define extreme points (vertices of triangle) - same as before
    v1 = np.array([-1, -0.5])
    v2 = np.array([1, -0.5])
    v3 = np.array([0, 1])

    # Draw triangle (same polytope)
    triangle = Polygon(
        [v1, v2, v3],
        facecolor=color_map["c8"],
        alpha=0.3,
        edgecolor=color_map["c8"],
        linewidth=2,
    )
    ax.add_patch(triangle)

    # Plot only extreme points
    ax.scatter(*v1, s=20, color="black", zorder=5)
    ax.scatter(*v2, s=20, color="black", zorder=5)
    ax.scatter(*v3, s=20, color="black", zorder=5)

    # Add labels
    ax.text(
        v1[0] - 0.15, v1[1] - 0.15, "$v_1$", fontsize=12, ha="center", fontweight="bold"
    )
    ax.text(
        v2[0] + 0.15, v2[1] - 0.15, "$v_2$", fontsize=12, ha="center", fontweight="bold"
    )
    ax.text(v3[0], v3[1] + 0.15, "$v_3$", fontsize=12, ha="center", fontweight="bold")

    # Add annotation showing these are extreme points
    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], color_map["c8"], linewidth=3, alpha=0.8)
    ax.plot([v2[0], v3[0]], [v2[1], v3[1]], color_map["c8"], linewidth=3, alpha=0.8)
    ax.plot([v3[0], v1[0]], [v3[1], v1[1]], color_map["c8"], linewidth=3, alpha=0.8)


def plot_extreme_points_comparison(ax=None, color_map=None):
    """
    Create comparison showing polytope with and without interior points.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.4)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # With interior point
    ax2 = fig.add_subplot(gs[0, 1])  # Extreme points only

    # Plot each example
    plot_triangle_with_interior_point(ax1, color_map)
    plot_triangle_extreme_only(ax2, color_map)

    # Set consistent styling for both subplots
    for ax in [ax1, ax2]:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_halfspace(ax=None, color_map=None):
    """
    Create a clean visualization of a polyhedron as intersection of half-spaces.
    Shows linear constraints Ax ≤ b with shaded infeasible regions.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Set up the plot bounds
    x_min, x_max = -1, 4
    y_min, y_max = -1, 4

    # Create a grid for plotting
    x = np.linspace(x_min, x_max, 400)
    y = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(x, y)

    # Define constraints: Ax ≤ b

    line1_x = np.linspace(x_min, x_max, 100)
    line1_y = 3 - line1_x
    constraint1 = X + Y <= 3

    ax.plot(
        line1_x,
        line1_y,
        color=color_map["c8"],
        linewidth=2,
        label="$x_1 + x_2 \\leq 3$",
    )

    infeasible1 = X + Y > 3
    ax.contourf(
        X,
        Y,
        infeasible1.astype(int),
        levels=[0.5, 1.5],
        colors=color_map["c8"],
        alpha=0.15,
    )

    # Add title and labels
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)

    # Set axis limits with padding
    padding = 0.1
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    legend_elements = [
        Patch(facecolor=color_map["c8"], alpha=0.15, label="Infeasible Region"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)


def plot_polyhedron_halfspaces(ax=None, color_map=None):
    """
    Create a clean visualization of a polyhedron as intersection of half-spaces.
    Shows linear constraints Ax ≤ b with shaded infeasible regions.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Set up the plot bounds
    x_min, x_max = -2, 6
    y_min, y_max = -2, 6

    # Create a grid for plotting
    x = np.linspace(x_min, x_max, 400)
    y = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(x, y)

    # Define constraints: Ax ≤ b
    # Let's use 3 constraints to form a nice polytope

    # Constraint 1: x + 2y <= 6 (line: y = 3 - 0.5x)
    line1_x = np.linspace(x_min, x_max, 100)
    line1_y = 3 - 0.5 * line1_x
    constraint1 = X + 2 * Y <= 6

    # Constraint 2: -2x + y <= -2 (line: y = -2 + 2x)
    line2_y = -2 + 2 * line1_x
    constraint2 = -2 * X + Y <= -2

    # Constraint 3: 0x - y <= -1 (line: y = 1)
    line3_y = np.ones_like(line1_x) * 1
    constraint3 = -Y <= -1

    # The feasible region is intersection of all constraints
    feasible = constraint1 & constraint2 & constraint3

    # Plot constraint lines
    ax.plot(line1_x, line1_y, color=color_map["c8"], linewidth=2, alpha=0.8)
    ax.plot(line1_x, line2_y, color=color_map["c8"], linewidth=2, alpha=0.8)
    ax.plot(line1_x, line3_y, color=color_map["c8"], linewidth=2, alpha=0.8)

    # Highlight the feasible region (polytope)
    ax.contourf(
        X, Y, feasible.astype(int), levels=[0.5, 1.5], colors=color_map["c8"], alpha=0.4
    )

    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)

    # Set axis limits with padding
    padding = 0.1
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    legend_elements = [
        Patch(facecolor=color_map["c8"], alpha=0.4, label="Feasible Region"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)


def plot_convex_cone(ax, color_map):
    """Plot a single convex cone."""

    # Define the cone boundaries (rays from origin)
    # Cone between angles -π/4 to π/4
    angle1 = -np.pi / 4
    angle2 = np.pi / 4

    # Create rays
    r_max = 3

    # Draw the cone boundary rays
    x1 = np.array([0, r_max * np.cos(angle1)])
    y1 = np.array([0, r_max * np.sin(angle1)])
    x2 = np.array([0, r_max * np.cos(angle2)])
    y2 = np.array([0, r_max * np.sin(angle2)])

    ax.plot(x1, y1, color="black", linewidth=2)
    ax.plot(x2, y2, color="black", linewidth=2)

    # Fill the cone region
    theta_fill = np.linspace(angle1, angle2, 100)
    x_fill = r_max * np.cos(theta_fill)
    y_fill = r_max * np.sin(theta_fill)

    # Create vertices for filled region (including origin)
    vertices = np.array([[0, 0]])
    vertices = np.vstack([vertices, np.column_stack([x_fill, y_fill])])
    vertices = np.vstack([vertices, [[0, 0]]])

    cone_region = Polygon(
        vertices,
        facecolor=color_map["c8"],
        alpha=0.3,
        edgecolor=color_map["c8"],
        linewidth=2,
    )
    ax.add_patch(cone_region)

    # Add some example points and their scaling
    example_point = np.array([1.5, 0.8])
    scaled_point = 0.6 * example_point
    scaled_point2 = 1.4 * example_point

    ax.scatter(*example_point, s=20, color="black", zorder=5)
    ax.scatter(*scaled_point, s=20, color="red", alpha=0.7, zorder=5)
    ax.scatter(*scaled_point2, s=20, color="red", alpha=0.7, zorder=5)

    # Draw scaling arrows
    ax.plot(
        [0, scaled_point[0]],
        [0, scaled_point[1]],
        "red",
        linewidth=1,
        alpha=0.7,
        linestyle="--",
    )
    ax.plot(
        [0, scaled_point2[0]],
        [0, scaled_point2[1]],
        "red",
        linewidth=1,
        alpha=0.7,
        linestyle="--",
    )

    # Labels
    ax.text(
        example_point[0] + 0.1, example_point[1] + 0.1, "$x$", fontsize=11, ha="left"
    )
    ax.text(
        scaled_point[0] - 0.2,
        scaled_point[1],
        "$0.6x$",
        fontsize=10,
        ha="right",
        color="red",
    )
    ax.text(
        scaled_point2[0] + 0.1,
        scaled_point2[1],
        "$1.4x$",
        fontsize=10,
        ha="left",
        color="red",
    )


def plot_nonconvex_cones(ax, color_map):
    """Plot two separate cones (non-convex union)."""

    # Define two separate cone regions
    r_max = 2.5

    # First cone: between angles π/6 to π/3
    angle1_start = np.pi / 6
    angle1_end = np.pi / 3

    # Second cone: between angles -π/3 to -π/6
    angle2_start = -np.pi / 3
    angle2_end = -np.pi / 6

    # Create first cone
    theta1 = np.linspace(angle1_start, angle1_end, 100)
    x1_fill = r_max * np.cos(theta1)
    y1_fill = r_max * np.sin(theta1)

    vertices1 = np.array([[0, 0]])
    vertices1 = np.vstack([vertices1, np.column_stack([x1_fill, y1_fill])])
    vertices1 = np.vstack([vertices1, [[0, 0]]])

    from matplotlib.patches import Polygon

    cone1 = Polygon(
        vertices1,
        facecolor=color_map["c8"],
        alpha=0.4,
        edgecolor=color_map["c8"],
        linewidth=2,
    )
    ax.add_patch(cone1)

    # Create second cone
    theta2 = np.linspace(angle2_start, angle2_end, 100)
    x2_fill = r_max * np.cos(theta2)
    y2_fill = r_max * np.sin(theta2)

    vertices2 = np.array([[0, 0]])
    vertices2 = np.vstack([vertices2, np.column_stack([x2_fill, y2_fill])])
    vertices2 = np.vstack([vertices2, [[0, 0]]])

    cone2 = Polygon(
        vertices2,
        facecolor=color_map["c8"],
        alpha=0.4,
        edgecolor=color_map["c8"],
        linewidth=2,
    )
    ax.add_patch(cone2)

    # Draw boundary rays
    ax.plot(
        [0, r_max * np.cos(angle1_start)],
        [0, r_max * np.sin(angle1_start)],
        "black",
        linewidth=2,
    )
    ax.plot(
        [0, r_max * np.cos(angle1_end)],
        [0, r_max * np.sin(angle1_end)],
        "black",
        linewidth=2,
    )
    ax.plot(
        [0, r_max * np.cos(angle2_start)],
        [0, r_max * np.sin(angle2_start)],
        "black",
        linewidth=2,
    )
    ax.plot(
        [0, r_max * np.cos(angle2_end)],
        [0, r_max * np.sin(angle2_end)],
        "black",
        linewidth=2,
    )

    # Show line segment between cones that goes outside
    point1 = np.array([1.5 * np.cos(np.pi / 4), 1.5 * np.sin(np.pi / 4)])  # In cone 1
    point2 = np.array([1.5 * np.cos(-np.pi / 4), 1.5 * np.sin(-np.pi / 4)])  # In cone 2

    ax.plot(
        [point1[0], point2[0]],
        [point1[1], point2[1]],
        "black",
        linewidth=2,
        linestyle="--",
        alpha=0.8,
    )
    ax.scatter(*point1, s=40, color="black", zorder=5)
    ax.scatter(*point2, s=40, color="black", zorder=5)

    ax.text(point1[0] + 0.1, point1[1], "$x_1$", fontsize=11, ha="left")
    ax.text(point2[0] + 0.1, point2[1], "$x_2$", fontsize=11, ha="left")


def plot_cone_comparison(ax=None, color_map=None):
    """
    Create comparison showing convex cone vs non-convex union of cones.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.4)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Convex cone
    ax2 = fig.add_subplot(gs[0, 1])  # Non-convex union

    # Plot each example
    plot_convex_cone(ax1, color_map)
    plot_nonconvex_cones(ax2, color_map)

    # Set consistent styling for both subplots
    for ax in [ax1, ax2]:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 2)
        ax.set_aspect("equal")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw origin
        ax.scatter(0, 0, s=20, color="black", marker="o", zorder=10)


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()
    svg_content = plotter.create_themed_plot(
        name="convex_nonconvex_sets", plot_func=plot_convex_nonconvex_sets
    )
    svg_content = plotter.create_themed_plot(
        name="sphere_and_circle", plot_func=plot_sphere_and_circle
    )
    svg_content = plotter.create_themed_plot(
        name="three_convex_operations", plot_func=plot_three_convex_operations
    )
    svg_content = plotter.create_themed_plot(
        name="convex_hull_construction", plot_func=plot_convex_hull_construction
    )
    svg_content = plotter.create_themed_plot(
        name="extreme_points_comparison", plot_func=plot_extreme_points_comparison
    )
    svg_content = plotter.create_themed_plot(name="halfspace", plot_func=plot_halfspace)
    svg_content = plotter.create_themed_plot(
        name="polyhedron_halfspaces", plot_func=plot_polyhedron_halfspaces
    )
    svg_content = plotter.create_themed_plot(
        name="cone_comparison", plot_func=plot_cone_comparison
    )
