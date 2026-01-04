import numpy as np
import matplotlib.pyplot as plt

"""
3d_sde.py

Make plots of different SDEs in 3D.
"""


def plot_3d_surface(ax=None, color_map=None):
    """
    Create a 3D surface plot with theme support.
    """
    # Create some sample data
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    # Create the plot
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0.5, antialiased=True)

    # Customize appearance
    ax.set_title("3D Surface Plot Example", fontsize=12)
    ax.set_xlabel("X Axis", fontsize=10)
    ax.set_ylabel("Y Axis", fontsize=10)
    ax.set_zlabel("Z Axis", fontsize=10)

    # Set viewing angle for better visualization
    ax.view_init(30, 45)

    return surf  # Return the surface for potential additional customization


def plot_3D_brownian_motion(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of a 3D Brownian motion.

    Consider a Brownian motion B(t) in 3D.

    We plot realizations in [0, T]
    """
    # Set up the plot bounds with padding
    padding = 0.2  # Add padding for better appearance
    # Create the x values
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    z_min, z_max = -5, 5
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    z = np.linspace(z_min, z_max, 100)
    # Create Brownian motion (Euler-Maruyama)
    dt = 0.01
    t = np.arange(0, 2 * np.pi, dt)
    Bx = np.zeros_like(t)
    By = np.zeros_like(t)
    Bz = np.zeros_like(t)
    Bx[0] = 0
    By[0] = 0
    Bz[0] = 0
    for i in range(1, len(t)):
        Bx[i] = Bx[i - 1] + np.random.normal(0, np.sqrt(dt))
        By[i] = By[i - 1] + np.random.normal(0, np.sqrt(dt))
        Bz[i] = Bz[i - 1] + np.random.normal(0, np.sqrt(dt))
    x_min, x_max = (
        min(np.min(Bx), np.min(By), np.min(Bz)) - padding,
        max(np.max(Bx), np.max(By), np.max(Bz)) + padding,
    )
    y_min, y_max = (
        min(np.min(Bx), np.min(By), np.min(Bz)) - padding,
        max(np.max(Bx), np.max(By), np.max(Bz)) + padding,
    )
    z_min, z_max = (
        min(np.min(Bx), np.min(By), np.min(Bz)) - padding,
        max(np.max(Bx), np.max(By), np.max(Bz)) + padding,
    )
    # Plot the Brownian motion
    ax.plot(Bx, By, Bz, color=color_map["c8"], linewidth=2)
    # Add subtle grid
    ax.grid(True, alpha=0.1, linestyle="-", zorder=0)
    # Customize plot appearance
    ax.set_xlabel(r"$\beta_x(t)$", fontsize=10)
    ax.set_ylabel(r"$\beta_y(t)$", fontsize=10)
    ax.set_zlabel(r"$\beta_z(t)$", fontsize=10)
    # Set axis limits with padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_zlim(z_min - padding, z_max + padding)
    # Set aspect ratio to be equal for proper visualization
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1


def plot_brownian_VS_sin(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of a Brownian motion and a sine function.

    Consider a Brownian motion B(t) and f(t) = sin(t).

    We plot realizations in [0, T]

    Sub plots on top of each other vertically stacked
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    # Set up the plot bounds with padding
    x_min, x_max = 0, 2 * np.pi
    padding = 0.2  # Add padding for better appearance
    # Create the x values
    x = np.linspace(x_min, x_max, 100)
    # Create the y values
    a = np.random.uniform(1, 2)
    phi = np.random.uniform(0, 2 * np.pi)
    y = a * np.sin(x + phi)
    # Create Brownian motion (Euler-Maruyama)
    dt = 0.01
    t = np.arange(0, 2 * np.pi, dt)
    B = np.zeros_like(t)
    B[0] = 0
    for i in range(1, len(t)):
        B[i] = B[i - 1] + np.random.normal(0, np.sqrt(dt))
    y_min, y_max = min(np.min(B), x_min) - padding, max(np.max(B), x_max) + padding
    # Plot the sine function
    ax1.plot(x, y, color=color_map["c8"], linewidth=2)
    # Plot the Brownian motion
    ax2.plot(t, B, color=color_map["c8"], linewidth=2)
    # Add subtle grid
    ax1.grid(True, alpha=0.1, linestyle="-", zorder=0)
    ax2.grid(True, alpha=0.1, linestyle="-", zorder=0)
    # Customize plot appearance
    ax1.set_xlabel(r"$t$", fontsize=10)
    ax1.set_ylabel(r"$f(t)$", fontsize=10)
    ax2.set_xlabel(r"$t$", fontsize=10)
    ax2.set_ylabel(r"$\beta(t)$", fontsize=10)
    # Set axis limits with padding
    ax1.set_xlim(x_min - padding, x_max + padding)
    ax1.set_ylim(y_min - padding, y_max + padding)
    ax2.set_xlim(x_min - padding, x_max + padding)
    ax2.set_ylim(y_min - padding, y_max + padding)
    # Set aspect ratio to be equal for proper visualization
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")


def plot_left_reimann_brownian(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of a left Riemann sum for a Brownian motion.

    Consider the Riemann sum sum_{i = 0}^{n - 1} f(t^{star}_i)(\beta(t_{i + 1}) - \beta(t_i)),
    where f(t) = t and \beta(t) is a Brownian motion.

    We also show the difference between the Right and Left Riemann sum.
    """
    # Set up the plot bounds with padding
    x_min, x_max = 0, 2
    padding = 0.2  # Add padding for better appearance
    # Create the x values
    x = np.linspace(x_min, x_max, 100)
    # Create the y values (Brownian motion, Euler-Maruyama)
    dt = 0.01
    t = np.arange(0, 2, dt)
    B = np.zeros_like(t)
    B[0] = 0
    for i in range(1, len(t)):
        B[i] = B[i - 1] + np.random.normal(0, np.sqrt(dt))
    y_min, y_max = min(np.min(B), x_min) - padding, max(np.max(B), x_max) + padding
    # Create the Left Riemann sum
    n = 10
    dx = (x_max - x_min) / n
    x_riemann = np.linspace(x_min, x_max, n + 1)
    y_riemann = x_riemann**2
    # Create the rectangles
    rects = np.zeros((n, 2))
    for i in range(n):
        rects[i, 0] = x_riemann[i]
        rects[i, 1] = y_riemann[i]
    # Plot the original function
    ax.plot(x, x, color=color_map["c8"], linewidth=2)
    # Plot the Brownian motion
    ax.plot(t, B, color=color_map["c7"], linewidth=2)
    # Add subtle grid
    ax.grid(True, alpha=0.1, linestyle="-", zorder=0)
    # Customize plot appearance
    ax.set_title(f"Left Riemann Sum, $n = 10$", fontsize=12, pad=15)
    ax.set_xlabel(r"$t$", fontsize=10)
    ax.set_ylabel(r"$\beta(t)$", fontsize=10)
    # Set axis limits with padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    # Add red points on the Brownian motion at the left Riemann sum points
    for i in range(n):
        ax.scatter(x_riemann[i], B[int(x_riemann[i] / dt)], color=color_map["c7"], s=50)


def plot_left_reimann_brownian(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of a left Riemann sum for a Brownian motion.

    Consider the Riemann sum sum_{i = 0}^{n - 1} \beta(t) (\beta(t_{i + 1}) - \beta(t_i)),
    where \beta(t) is a Brownian motion.
    """
    # Set up the plot bounds with padding
    x_min, x_max = 0, 2
    padding = 0.2  # Add padding for better appearance
    # Create the x values
    x = np.linspace(x_min, x_max, 100)
    # Create the y values (Brownian motion, Euler-Maruyama)
    dt = 0.01
    t = np.arange(0, 2, dt)
    B = np.zeros_like(t)
    B[0] = 0
    for i in range(1, len(t)):
        B[i] = B[i - 1] + np.random.normal(0, np.sqrt(dt))
    y_min, y_max = min(np.min(B), x_min) - padding, max(np.max(B), x_max) + padding
    # Create the Left Riemann sum
    n = 10
    dx = (x_max - x_min) / n
    x_riemann = np.linspace(x_min, x_max, n + 1)
    y_riemann = x_riemann**2
    # Create the rectangles
    rects = np.zeros((n, 2))
    for i in range(n):
        rects[i, 0] = x_riemann[i]
        rects[i, 1] = y_riemann[i]
    # Plot the Brownian motion
    ax.plot(t, B, color=color_map["c7"], linewidth=2)
    # Add subtle grid
    ax.grid(True, alpha=0.1, linestyle="-", zorder=0)
    # Customize plot appearance
    ax.set_title(f"Left Riemann Sum, $n = 10$", fontsize=12, pad=15)
    ax.set_xlabel(r"$t$", fontsize=10)
    ax.set_ylabel(r"$\beta(t)$", fontsize=10)
    # Set axis limits with padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)


def plot_second_order(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of the second-order sum of Brownian motion.
    Consider the second-order sum of Brownian motion:
    sum_{t_i < t}(Delta \beta)^2
    where Delta \beta = \beta(t_{i + 1}) - \beta(t_i)
    """
    n = 5
    dt = 0.01
    T = 2
    color_brownian = color_map["c8"]
    color_points = color_map["c1"]
    color_sum = color_map["c8"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Create the Brownian motion
    t = np.arange(0, T + dt, dt)
    B = np.cumsum(np.sqrt(dt) * np.random.randn(len(t)))

    # Define partition points evenly
    points = np.linspace(0, len(t) - 1, n + 1, dtype=int)

    # Second-order sum computation
    B2 = np.zeros_like(t)
    for i in range(len(points) - 1):
        idx_start, idx_end = points[i], points[i + 1]
        increment = (B[idx_end] - B[idx_start]) ** 2
        B2[idx_end:] += increment

    # Plot Brownian motion
    ax1.plot(t, B, color=color_brownian, linewidth=1)
    ax1.scatter(t[points], B[points], color=color_points, zorder=5)

    # Plot second-order sum
    ax2.step(t, B2, where="post", color=color_sum)

    # Customization
    ax1.set_title(f"Brownian motion with n={n}")
    ax1.set_ylabel(r"$\beta(t)$", fontsize=12)
    ax2.set_xlabel("t", fontsize=12)
    ax2.set_ylabel(r"$\sum_{t_i < t}(\Delta \beta)^2$", fontsize=12)

    for ax in (ax1, ax2):
        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()
    svg_content = plotter.create_themed_plot(
        name="3d_sde", plot_func=plot_3D_brownian_motion, is_3d=True
    )
    svg_content = plotter.create_themed_plot(
        name="3d_surface", plot_func=plot_3d_surface, is_3d=True
    )
