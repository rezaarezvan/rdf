import numpy as np
import matplotlib.pyplot as plt

"""
SDE.py

Make plots of different SDEs
"""


def plot_SDE_only_drift(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of an SDE with only drift (ODE).

    Suppose f(x, t) = 2, L(x, t) = 0.
    >= dx/dt = 2, assume that x(0) ~ N(0, 2).
    """
    # Set up the plot bounds with padding
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    padding = 0.2  # Add padding for better appearance

    # Create the drift line
    x = np.array([x_min - padding, x_max + padding])
    y = 2 * x  # Line equation: dx/dt = 2

    # Plot the drift line
    ax.plot(x, y, color=color_map["c8"], linewidth=2)

    # Add subtle grid
    ax.grid(True, alpha=0.1, linestyle="-", zorder=0)

    # Customize plot appearance
    ax.set_title("Sample of a SDE with only drift", fontsize=12, pad=15)
    ax.set_xlabel(r"$t$", fontsize=10)
    ax.set_ylabel(r"$x$", fontsize=10)

    # Set axis limits with padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set aspect ratio to be equal for proper visualization
    ax.set_aspect("equal")

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)


def plot_SDE_only_diffusion(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of an SDE with only diffusion (Brownian motion).

    Suppose f(x, t) = 0, L(x, t) = 2.
    >= dx(t) = 2 dB(t), assume that x(0) ~ N(0, 1).
    """
    # Set up the plot bounds with padding
    x_min, x_max = -10, 10
    y_min, y_max = -10, 10
    padding = 0.2  # Add padding for better appearance

    # Create the brownian motion
    x = np.linspace(x_min - padding, x_max + padding, 200)
    y = np.random.normal(0, 1, size=x.shape)  # Brownian motion
    y = 2 * y  # Scale by 2

    # Plot the brownian motion
    ax.plot(x, y, color=color_map["c8"], linewidth=2)

    # Add subtle grid
    ax.grid(True, alpha=0.1, linestyle="-", zorder=0)

    # Customize plot appearance
    ax.set_title("Sample of a SDE with only diffusion", fontsize=12, pad=15)
    ax.set_xlabel(r"$t$", fontsize=10)
    ax.set_ylabel(r"$x$", fontsize=10)

    # Set axis limits with padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set aspect ratio to be equal for proper visualization
    ax.set_aspect("equal")

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)


def plot_SDE_example(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of a (simple) SDE.

    To sample x(t), we can draw x(0) ~ p(x(0)) and then repeat:
        x(t + dt) = x(t) + f(x(t), t) dt + L(x(t), t)z, z ~ N(0, dt)

    Suppose dx(t) = 0.5x dt + dB(t)
    which corresponds to:
        f(x, t) = 0.5x and L(x, t) = 1
    """
    # Set up the plot bounds with padding
    x_min, x_max = 0, 2
    padding = 0.2  # Add padding for better appearance

    # Simulate the SDE
    dt = 0.01  # Time step
    t = np.arange(0, 1, dt)
    x = np.zeros_like(t)
    x[0] = np.random.normal(0, 1)  # Initial condition
    for i in range(1, len(t)):
        x[i] = x[i - 1] + 0.5 * x[i - 1] * dt + np.random.normal(0, np.sqrt(dt))

    y_min, y_max = np.min(x) + padding, np.max(x) - padding

    # Create the time array
    time = np.linspace(x_min - padding, x_max + padding, len(x))
    # Plot the SDE
    ax.plot(time, x, color=color_map["c8"], linewidth=2)

    # Add subtle grid
    ax.grid(True, alpha=0.1, linestyle="-", zorder=0)

    # Customize plot appearance
    ax.set_title("Sample of a SDE", fontsize=12, pad=15)
    ax.set_xlabel(r"$t$", fontsize=10)
    ax.set_ylabel(r"$x$", fontsize=10)

    # Set axis limits with padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set aspect ratio to be equal for proper visualization
    ax.set_aspect("equal")

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)


def plot_dynamic_SDE_example(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of a (simple) SDE.

    Suppose x(t) = [x(t), y(t)]^T represents the position and velocity of an object.

    According to the constant velocity model,

    dx(t) = [0, 1; 0, 0]x(t)dt + [0; q]dB(t),
    <=> [dx(t); dy(t)] = [v(t)dt; 0] + [0; q dB(t)].

    The position follows the velocity, whereas the velocity is driven by a Brownian motion.

    CV with x(0) = v(0) = 0, q = 0.5
    """
    # Set up the plot bounds with padding
    x_min, x_max = 0, 2
    padding = 0.2  # Add padding for better appearance

    # Simulate the SDE
    dt = 0.01  # Time step
    t = np.arange(0, 1, dt)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    x[0] = 0
    y[0] = 0
    for i in range(1, len(t)):
        x[i] = x[i - 1] + y[i - 1] * dt
        y[i] = y[i - 1] + np.random.normal(0, np.sqrt(dt))
    y_min, y_max = np.min(y) + padding, np.max(y) - padding

    # Create the time array
    time = np.linspace(x_min - padding, x_max + padding, len(x))
    # Plot the SDE
    ax.plot(time, y, color=color_map["c8"], linewidth=2, label="$v(t)$")
    ax.plot(time, x, color=color_map["c7"], linewidth=2, label="$x(t)$", linestyle="--")

    # Add subtle grid
    ax.grid(True, alpha=0.1, linestyle="-", zorder=0)

    # Customize plot appearance
    ax.set_title("Constant Velocity Model with $x(0) = v(0) = 0$", fontsize=12, pad=15)
    ax.set_xlabel(r"$t$", fontsize=10)
    ax.set_ylabel(r"$x(t), v(t)$", fontsize=10)

    # Set axis limits with padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    # Clean legend
    ax.legend(
        frameon=True,
        framealpha=0.9,
        loc="upper right",
        fontsize=9,
        bbox_to_anchor=(0.98, 0.98),
    )

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set aspect ratio to be equal for proper visualization
    ax.set_aspect("equal")

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)


def plot_Riemann_sum(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of a Riemann sum (approximation of an integral).

    f(x) = x^2, n = 10
    """
    # Set up the plot bounds with padding
    x_min, x_max = 0, 2
    y_min, y_max = 0, 2
    padding = 0.2  # Add padding for better appearance
    # Create the x values
    x = np.linspace(x_min, x_max, 100)
    # Create the y values
    y = x**2
    # Create the Riemann sum
    n = 10
    dx = (x_max - x_min) / n
    x_riemann = np.linspace(x_min, x_max, n + 1)
    y_riemann = x_riemann**2
    error = np.abs(y_riemann[:-1] - y_riemann[1:])
    error = np.sum(error) * dx
    # Create the rectangles
    rects = np.zeros((n, 2))
    for i in range(n):
        rects[i, 0] = x_riemann[i]
        rects[i, 1] = y_riemann[i]
    # Plot the original function
    ax.plot(x, y, color=color_map["c8"], linewidth=2)
    # Plot the rectangles
    for i in range(n):
        ax.add_patch(
            plt.Rectangle(
                (rects[i, 0], 0), dx, rects[i, 1], color=color_map["c7"], alpha=0.5
            )
        )
    # Add subtle grid
    ax.grid(True, alpha=0.1, linestyle="-", zorder=0)
    # Customize plot appearance
    ax.set_title(f"Riemann Sum, $n = 10$, and error = ${error}$", fontsize=12, pad=15)
    ax.set_xlabel(r"$x$", fontsize=10)
    ax.set_ylabel(r"$f(x)$", fontsize=10)
    # Set axis limits with padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Set aspect ratio to be equal for proper visualization
    ax.set_aspect("equal")
    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)


def plot_random_diff(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of a random function and its derivative at x(1).

    Consider x(t) = a sin(t + phi) where a ~ unif[1, 2] and phi ~ unif[0, 2pi].
    """
    # Set up the plot bounds with padding
    x_min, x_max = 0, 2 * np.pi
    y_min, y_max = -2, 2
    padding = 0.2  # Add padding for better appearance
    # Create the x values
    x = np.linspace(x_min, x_max, 100)
    # Create the y values
    a = np.random.uniform(1, 2)
    phi = np.random.uniform(0, 2 * np.pi)
    y = a * np.sin(x + phi)
    # Create the derivative
    y_diff = a * np.cos(x + phi)
    # Create the tangent line
    x_tangent = 1
    y_tangent = a * np.sin(x_tangent + phi)
    y_diff_tangent = a * np.cos(x_tangent + phi)
    slope = y_diff_tangent
    intercept = y_tangent - slope * x_tangent
    tangent_line = slope * (x - x_tangent) + y_tangent
    # Plot the original function
    ax.plot(x, y, color=color_map["c8"], linewidth=2)
    # Plot the tangent line
    ax.plot(x, tangent_line, color=color_map["c7"], linewidth=2, linestyle="--")
    # Add a point at x(1) and text that says \dot{x}(1) = y_diff(1)
    ax.scatter(x_tangent, y_tangent, color=color_map["c7"], s=50)
    ax.text(
        x_tangent + 0.1,
        y_tangent,
        r"$\dot{x}(1) = " + str(round(y_diff_tangent, 2)) + "$",
        fontsize=10,
        color=color_map["c7"],
    )
    # Add subtle grid
    ax.grid(True, alpha=0.1, linestyle="-", zorder=0)
    # Customize plot appearance
    # ax.set_title(f'Random Function and its Derivative at x(1)',
    #              fontsize=12, pad=15)
    ax.set_xlabel(r"$t$", fontsize=10)
    ax.set_ylabel(r"$x(t)$", fontsize=10)
    # Set axis limits with padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Set aspect ratio to be equal for proper visualization
    ax.set_aspect("equal")
    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)


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
    # Remove top and right spines
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    # Set aspect ratio to be equal for proper visualization
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    # Add subtle ticks
    ax1.tick_params(axis="both", which="major", labelsize=9)
    ax2.tick_params(axis="both", which="major", labelsize=9)


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
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)
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
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)


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
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


if __name__ == "__main__":
    from rdp import RDP

    plotter = RDP()
    # svg_content = plotter.create_themed_plot(
    #     save_name="SDE_only_drift", plot_func=plot_SDE_only_drift
    # )
    #
    # svg_content = plotter.create_themed_plot(
    #     save_name="SDE_only_diffusion", plot_func=plot_SDE_only_diffusion
    # )
    #
    # svg_content = plotter.create_themed_plot(
    #     save_name="SDE_example", plot_func=plot_SDE_example
    # )
    #
    # svg_content = plotter.create_themed_plot(
    #     save_name="dynamic_SDE_example", plot_func=plot_dynamic_SDE_example
    # )
    #
    # svg_content = plotter.create_themed_plot(
    #     save_name="Riemann_sum", plot_func=plot_Riemann_sum
    # )
    #
    # svg_content = plotter.create_themed_plot(
    #     save_name="random_diff", plot_func=plot_random_diff
    # )
    #
    # svg_content = plotter.create_themed_plot(
    #     save_name="random_riemann", plot_func=plot_random_riemann
    # )
    svg_content = plotter.create_themed_plot(
        save_name="brownian_VS_sin", plot_func=plot_brownian_VS_sin
    )
    # svg_content = plotter.create_themed_plot(
    #     save_name="left_riemann_brownian", plot_func=plot_left_reimann_brownian)
    # svg_content = plotter.create_themed_plot(
    #     save_name="second_order", plot_func=plot_second_order)
