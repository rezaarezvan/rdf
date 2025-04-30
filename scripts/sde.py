import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

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
        x[i] = x[i - 1] + 0.5 * x[i - 1] * \
            dt + np.random.normal(0, np.sqrt(dt))

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
    ax.plot(time, x, color=color_map["c7"],
            linewidth=2, label="$x(t)$", linestyle="--")

    # Add subtle grid
    ax.grid(True, alpha=0.1, linestyle="-", zorder=0)

    # Customize plot appearance
    ax.set_title("Constant Velocity Model with $x(0) = v(0) = 0$",
                 fontsize=12, pad=15)
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
    ax.set_title(f"Riemann Sum, $n = 10$, and error = ${
                 error}$", fontsize=12, pad=15)
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
    ax.plot(x, tangent_line,
            color=color_map["c7"], linewidth=2, linestyle="--")
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
    y_min, y_max = min(np.min(B), x_min) - \
        padding, max(np.max(B), x_max) + padding
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
    y_min, y_max = min(np.min(B), x_min) - \
        padding, max(np.max(B), x_max) + padding
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
        ax.scatter(x_riemann[i], B[int(x_riemann[i] / dt)],
                   color=color_map["c7"], s=50)


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
    y_min, y_max = min(np.min(B), x_min) - \
        padding, max(np.max(B), x_max) + padding
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


def plot_beta_beta_squared(ax=None, color_map=None):
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
    t = np.arange(0, 2, dt)
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
    # Add subtle grid
    ax.grid(True, alpha=0.1, linestyle="-", zorder=0)
    # Customize plot appearance
    ax.set_title(f"Brownian motion and its squared version",
                 fontsize=12, pad=15)
    ax.set_xlabel(r"$t$", fontsize=10)
    ax.set_ylabel(r"$\beta(t), \beta^2(t)$", fontsize=10)
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
    # Legend
    ax.legend(
        ["$\\beta(t)$", "$\\beta^2(t)$"],
        frameon=True,
        framealpha=0.9,
        loc="upper right",
        fontsize=9,
        bbox_to_anchor=(0.98, 0.98),
    )


def plot_beta_squared_second_order_taylor(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of a squared Brownian motion as a function of B(t),
    with it's second-order Taylor expansion, 2B_i(B(t) - B_i) + B_i^2.
    For some i, B_i is the value of Brownian motion at t_i.

    x-axis is B(t), y-axis is B^2(t).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    # Set seed for reproducibility
    np.random.seed(42078)
    # Create the y values (Brownian motion, Euler-Maruyama)
    dt = 0.01
    t = np.arange(0, 2, dt)
    B = np.zeros_like(t)
    B[0] = 0
    for i in range(1, len(t)):
        B[i] = B[i - 1] + np.random.normal(0, np.sqrt(dt))
    # Create the squared Brownian motion
    B_squared = B**2
    # Create the second-order Taylor expansion
    index_i = len(t) // 2
    B_i = B[index_i]
    B_i_squared = B_i**2
    taylor_expansion1 = 2 * B_i * (B - B_i) + B_i_squared
    taylor_expansion2 = 2 * B_i * (B - B_i)

    x_range = np.max(B) - np.min(B)
    y_range = np.max(B_squared) - np.min(B_squared)
    padding_x = 0.1 * x_range
    padding_y = 0.1 * y_range

    # Determine axis limits
    x_min, x_max = np.min(B) - padding_x, np.max(B) + padding_x
    y_min = min(0, np.min(taylor_expansion2)) - \
        padding_y  # Ensure 0 is included
    y_max = max(np.max(B_squared), np.max(taylor_expansion1)) + padding_y

    # Plot squared Brownian motion
    ax1.plot(B, B_squared, color=color_map["c8"],
             linewidth=2, label="$\\beta^2(t)$")
    ax2.plot(B, B_squared, color=color_map["c8"],
             linewidth=2, label="$\\beta^2(t)$")
    # Plot second-order Taylor expansion
    ax1.plot(
        B,
        taylor_expansion1,
        color=color_map["c7"],
        linewidth=2,
        label="$2\\beta_i(\\beta(t) - \\beta_i) + \\beta_i^2$",
        linestyle="--",
    )
    ax2.plot(
        B,
        taylor_expansion2,
        color=color_map["c7"],
        linewidth=2,
        label="$2\\beta_i(\\beta(t) - \\beta_i) + \\beta_i^2$",
        linestyle="--",
    )
    # Add red point at tangent point (B_i, B_i_squared)
    ax1.scatter(B_i, B_i_squared, color=color_map["c1"], s=30, alpha=0.8)
    ax2.scatter(B_i, B_i_squared, color=color_map["c1"], s=30, alpha=0.8)
    # Add subtle grid
    ax1.grid(True, alpha=0.1, linestyle="-", zorder=0)
    ax2.grid(True, alpha=0.1, linestyle="-", zorder=0)
    # Customize plot appearance
    fig.suptitle(
        "Squared Brownian motion and its second-order Taylor expansion",
        fontsize=12,
    )
    ax1.set_xlabel(r"$\beta(t)$", fontsize=10)
    ax1.set_ylabel(r"$\beta^2(t)$", fontsize=10)
    ax2.set_xlabel(r"$\beta(t)$", fontsize=10)
    ax2.set_ylabel(r"$\beta^2(t)$", fontsize=10)
    # Set axis limits with padding
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
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
    # Legend
    ax1.legend(
        frameon=True,
        framealpha=0.9,
        loc="upper right",
        fontsize=9,
        bbox_to_anchor=(0.98, 0.98),
    )

    ax2.legend(
        frameon=True,
        framealpha=0.9,
        loc="upper right",
        fontsize=9,
        bbox_to_anchor=(0.98, 0.98),
    )


def plot_ornstein_uhlenbeck(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of the Ornstein-Uhlenbeck process.

    The Ornstein-Uhlenbeck process is characterized by:
    dx(t) = -theta x(t) dt + sigma dB(t)

    This function plots multiple sample paths along with the mean (dotted line)
    and variance envelope (shaded region representing ±1 standard deviation).
    """
    theta = 0.7    # Mean reversion speed
    mu = 0.0       # Long-term mean
    sigma = 0.5    # Volatility
    T = 5.0        # Total time
    dt = 0.01      # Time step

    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps)

    # Analytical mean and variance of the OU process
    mean = mu * np.ones_like(t)
    variance = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * t))
    std_dev = np.sqrt(variance)

    # Generate multiple sample paths
    n_paths = 10
    paths = np.zeros((n_paths, n_steps))

    # Colormap for the sample paths
    colors = list(color_map.values())[:n_paths]

    # Initialize all paths at x(0) = 0
    paths[:, 0] = 0.0

    # Simulate sample paths using Euler-Maruyama method
    for i in range(n_paths):
        np.random.seed(42 + i)  # Different seed for each path
        for j in range(1, n_steps):
            drift = theta * (mu - paths[i, j-1]) * dt
            diffusion = sigma * np.random.normal(0, np.sqrt(dt))
            paths[i, j] = paths[i, j-1] + drift + diffusion

    # Plot the shaded variance envelope (±1 standard deviation)
    ax.fill_between(t, mean - std_dev, mean + std_dev, color='gray', alpha=0.2)

    # Plot all sample paths
    for i in range(n_paths):
        ax.plot(t, paths[i], color=colors[i %
                len(colors)], linewidth=1, alpha=0.8)

    # Plot the mean as a dashed line
    ax.plot(t, mean, color='black', linestyle='--',
            linewidth=1.5, label='Mean')

    # Add a line at the boundary of the standard deviation envelope
    ax.plot(t, mean + std_dev, color='black', linestyle=':',
            linewidth=0.8, alpha=0.5, label='±1 std dev')
    ax.plot(t, mean - std_dev, color='black',
            linestyle=':', linewidth=0.8, alpha=0.5)

    # Customize plot appearance
    ax.set_title(
        "Ornstein-Uhlenbeck Process: Sample Paths and Variance Envelope", fontsize=12, pad=15)
    ax.set_xlabel("Time $t$", fontsize=10)
    ax.set_ylabel("$X(t)$", fontsize=10)

    # Clean legend
    ax.legend(
        frameon=True,
        framealpha=0.9,
        loc="upper right",
        fontsize=9,
        bbox_to_anchor=(0.98, 0.98),
    )

    # Add subtle grid
    ax.grid(True, alpha=0.15, linestyle="-", zorder=0)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)


def plot_reverse_ornstein_uhlenbeck(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of the Reverse-Time Ornstein-Uhlenbeck process.

    In the reverse-time OU process, the time is reversed, resulting in:
    dx(t) = theta x(t) dt + sigma dB(t)   (note the positive theta)

    This visualizes the process going backwards in time, showing
    multiple sample paths with a legend.
    """
    # Process parameters
    # Mean reversion parameter (now acting as divergence parameter in reverse time)
    theta = 0.7
    mu = 0.0       # Long-term mean
    sigma = 0.5    # Volatility
    T = 5.0        # Total time
    dt = 0.01      # Time step
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps)

    # Generate multiple sample paths
    n_paths = 10
    paths = np.zeros((n_paths, n_steps))

    # Colormap for the sample paths
    colors = list(color_map.values())[:n_paths]

    # Initialize all paths at x(0) = 0
    paths[:, 0] = 0.0

    # Simulate sample paths using Score-based reverse-time Euler-Maruyama method
    # In reverse time, we have dx(t) = theta x(t) dt + sigma dB(t)
    for i in range(n_paths):
        np.random.seed(42 + i)  # Different seed for each path
        for j in range(1, n_steps):
            # Note the plus sign here (reverse-time effect)
            drift = theta * paths[i, j-1] * dt
            diffusion = sigma * np.random.normal(0, np.sqrt(dt))
            paths[i, j] = paths[i, j-1] + drift + diffusion

    # Plot all sample paths
    for i in range(n_paths):
        ax.plot(t, paths[i], color=colors[i % len(colors)], linewidth=1.2,
                label=f'Path {i+1}')

    # Customize plot appearance
    ax.set_title("Reverse-Time Ornstein-Uhlenbeck Process (Score-Based Euler-Maruyama)",
                 fontsize=12, pad=15)
    ax.set_xlabel("Time t (reversed)", fontsize=10)
    ax.set_ylabel("x(t)", fontsize=10)

    # Clean legend with better positioning
    ax.legend(
        frameon=True,
        framealpha=0.9,
        loc="upper right",
        fontsize=8,
        ncol=2,  # Arrange legend in two columns
        bbox_to_anchor=(0.98, 0.98),
    )

    # Add subtle grid
    ax.grid(True, alpha=0.15, linestyle="-", zorder=0)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)

    # Set y-limits to match the example
    ax.set_ylim(-2.5, 2.5)


def plot_denoising_diffusion(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of a denoising diffusion probabilistic model
    represented as a stochastic differential equation (SDE):

    dx = -x/(1 - t) dt + sqrt(2t/(1 - t)) dB(t)

    This simulation shows multiple paths starting from different initial points,
    with a cleaner legend implementation.
    """
    # Process parameters
    T = 1.0        # Total time (from 0 to 1)
    dt = 0.001     # Time step (smaller for accuracy near t=1)
    n_steps = int(T / dt)
    # Create non-uniform time grid with more points near t=1
    # where the SDE becomes more challenging to simulate
    t = np.linspace(0, 0.95, n_steps//2)  # First half: 0 to 0.95
    # Second half: 0.95 to 0.999
    t = np.append(t, np.linspace(0.95, 0.999, n_steps//2))

    # Select specific initial points to track
    # Reduced number for cleaner legend
    initial_points = [-2, -1, 0, 1, 2]
    n_paths = len(initial_points)
    paths = np.zeros((n_paths, len(t)))

    # Set colors from the color map
    colors = [color_map[f"c{i+1}"] for i in range(n_paths)]

    # Initialize paths with their respective starting points
    for i, x0 in enumerate(initial_points):
        paths[i, 0] = x0

    # Simulate the SDE: dx = -x/(1-t) dt + sqrt(2t/(1-t)) dB(t)
    for i in range(n_paths):
        np.random.seed(42 + i)  # Different seed for each path for diversity
        for j in range(1, len(t)):
            current_t = t[j-1]
            next_t = t[j]
            dt_actual = next_t - current_t

            # Avoid numerical issues near t=1
            if current_t > 0.998:
                drift_coef = -100  # Large negative value as approximation
                diffusion_coef = 100  # Large value as approximation
            else:
                drift_coef = -1.0 / (1.0 - current_t)
                diffusion_coef = np.sqrt(2.0 * current_t / (1.0 - current_t))

            # Euler-Maruyama step
            drift = drift_coef * paths[i, j-1] * dt_actual
            diffusion = diffusion_coef * \
                np.random.normal(0, np.sqrt(dt_actual))
            paths[i, j] = paths[i, j-1] + drift + diffusion

    # Plot the paths
    for i, x0 in enumerate(initial_points):
        ax.plot(t, paths[i], color=colors[i], linewidth=1.2,
                label=f'$x_0={x0}$')

    # Customize plot appearance
    ax.set_title("Simulation of SDE: $dx = -x/(1 - t)\\, dt + \\sqrt{2t/(1 - t)}\\, dB(t)$",
                 fontsize=12, pad=15)
    ax.set_xlabel("Time $t$", fontsize=10)
    ax.set_ylabel("$x(t)$", fontsize=10)

    # Add clean legend
    ax.legend(
        frameon=True,
        framealpha=0.9,
        loc="upper left",  # Place legend in upper left to avoid crowded paths
        fontsize=9,
        title="Initial $x_0$",
        title_fontsize=9,
    )

    # Add subtle grid
    ax.grid(True, alpha=0.15, linestyle="-", zorder=0)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)

    # Set y-limits similar to the example
    ax.set_ylim(-3, 3)

    # Set x-limits
    ax.set_xlim(0, 1)


def plot_brownian_to_point(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of a Brownian motion
    converging to a point/line $v$ at time $T$,

    dx(t) = (v - x(t)) / (T - t) dt + dB(t)
    x(T) = v
    """
    # Set up the plot bounds with padding
    x_min, x_max = 0, 2
    padding = 0.2  # Add padding for better appearance
    # Create the y values (Brownian motion, Euler-Maruyama)
    dt = 0.01
    t = np.arange(0, 2, dt)
    B = np.zeros_like(t)
    B[0] = 0
    for i in range(1, len(t)):
        B[i] = B[i - 1] + np.random.normal(0, np.sqrt(dt))
    y_min, y_max = min(np.min(B), x_min) - \
        padding, max(np.max(B), x_max) + padding
    # Create the target point/line
    v = 1
    T = 2
    # Plot the Brownian motion
    ax.plot(t, B, color=color_map["c7"], linewidth=2)
    # Plot the target point/line
    ax.plot([T], [v], "ro", markersize=8)
    ax.axhline(y=v, color=color_map["c8"], linestyle="--")
    # Add subtle grid
    ax.grid(True, alpha=0.1, linestyle="-", zorder=0)
    # Customize plot appearance
    ax.set_title(f"Brownian motion converging to a point at time T",
                 fontsize=12, pad=15)
    ax.set_xlabel(r"$t$", fontsize=10)
    ax.set_ylabel(r"$\beta(t)$", fontsize=10)
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


def plot_brownian_bridge(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of a Brownian motion
    going from t_0 to t_1 and then from t_1 to t_2.

    t_0, t_1 and t_2 should be clear dotted lines.
    """
    # Set up the plot bounds with padding
    t_min, t_max = 0, 2
    padding = 0.2  # Add padding for better appearance

    # Define key time points
    t0 = 0.0
    t1 = 1.0
    t2 = 2.0

    # Simulation parameters
    dt = 0.01  # Time step
    n_steps = int((t_max - t_min) / dt)
    t = np.linspace(t_min, t_max, n_steps)

    # Generate a standard Brownian motion first
    dB = np.random.normal(0, np.sqrt(dt), n_steps)
    B = np.cumsum(dB)
    B = np.insert(B, 0, 0)  # Start at 0
    t = np.insert(t, 0, 0)  # Add t=0

    # Calculate y-axis limits
    y_min, y_max = np.min(B) - padding, np.max(B) + padding

    # Plot the Brownian motion
    ax.plot(t, B, color=color_map["c1"], linewidth=2, label="Brownian Motion")

    # Add vertical dotted lines at key time points
    ax.axvline(x=t0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=t1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=t2, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # Add labels for the time points at the bottom of the graph
    ax.text(t0, y_min + 0.2, "$t_0$", ha='center', va='top', fontsize=12)
    ax.text(t1, y_min + 0.2, "$t_1$", ha='center', va='top', fontsize=12)
    ax.text(t2, y_min + 0.2, "$t_2$", ha='center', va='top', fontsize=12)

    # Add subtle grid
    ax.grid(True, alpha=0.15, linestyle="-", zorder=0)

    # Customize plot appearance
    ax.set_title("Brownian Motion Path", fontsize=12, pad=15)
    ax.set_xlabel("$t$", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)

    # Set axis limits with padding
    ax.set_xlim(t_min - padding, t_max + padding)
    ax.set_ylim(y_min, y_max)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)


def plot_brownian_transition_density(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of the transition density of a Brownian motion.
    P(beta(t) | beta(s))

    Show brownian to time s and the point beta(s) then the density (on the side) and possible paths to it (time s to t)

    Use KDE for the density on the side.
    """
    # Set up parameters
    s = 0.6        # Start time for transition
    t = 1.2        # End time for transition
    n_samples = 10  # Number of sample paths to show after time s

    # Create time grid
    dt = 0.01
    time_points = np.arange(0, t + dt, dt)  # Only go up to time t
    n_points = len(time_points)

    # Generate a Brownian motion path up to time s
    np.random.seed(42)  # For reproducibility
    dB = np.random.normal(0, np.sqrt(dt), n_points)
    B = np.zeros(n_points)

    for i in range(1, n_points):
        B[i] = B[i-1] + dB[i-1]

    # Find index for time s
    s_idx = np.abs(time_points - s).argmin()
    t_idx = len(time_points) - 1  # Last index corresponds to time t

    # Get value at time s
    value_at_s = B[s_idx]

    # Generate sample paths from s to t
    future_paths = np.zeros((n_samples, t_idx - s_idx + 1))
    future_paths[:, 0] = value_at_s  # All paths start at value_at_s

    # Different seeds for different paths
    for i in range(n_samples):
        np.random.seed(100 + i)
        for j in range(1, t_idx - s_idx + 1):
            future_paths[i, j] = future_paths[i, j-1] + \
                np.random.normal(0, np.sqrt(dt))

    # Calculate transition density parameters
    mean_at_t = value_at_s  # For Brownian motion, mean stays the same
    std_at_t = np.sqrt(t - s)  # Variance grows linearly with time

    # Calculate y-limits for the plot
    path_min = min(np.min(B[:s_idx+1]), np.min(future_paths))
    path_max = max(np.max(B[:s_idx+1]), np.max(future_paths))
    y_padding = 0.5 * (path_max - path_min)
    y_min = path_min - y_padding
    y_max = path_max + y_padding

    # Create density range for time t
    y_density = np.linspace(y_min, y_max, 100)
    density_at_t = norm.pdf(y_density, loc=mean_at_t, scale=std_at_t)

    # Normalize density for plotting
    max_density = np.max(density_at_t)
    density_at_t_normalized = density_at_t / \
        max_density * 0.3  # Scale for visual appeal

    # Plot realized path up to time s
    ax.plot(time_points[:s_idx+1], B[:s_idx+1], color=color_map["c1"],
            linewidth=2.5, label="Realized Path")

    # Mark position at time s
    ax.scatter(s, value_at_s, s=80, color=color_map["c1"], zorder=5,
               edgecolor='white', linewidth=1)

    # Plot future sample paths with fading transparency
    for i in range(n_samples):
        alpha = 0.3 if i < n_samples - 5 else 0.6  # Make a few paths more visible
        ax.plot(time_points[s_idx:t_idx+1], future_paths[i], color=color_map["c2"],
                linewidth=1, alpha=alpha, zorder=2)

    # Plot vertical lines at s and t
    ax.axvline(x=s, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=t, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # Add labels for s and t
    ax.text(s, y_min - 0.4, "$s$", ha='center', va='top', fontsize=12)
    ax.text(t, y_min - 0.4, "$t$", ha='center', va='top', fontsize=12)

    # Plot the Gaussian density at time t
    # First as filled area
    t_width = 0.15
    ax.fill_betweenx(y_density,
                     t * np.ones_like(y_density),
                     t + density_at_t_normalized,
                     color=color_map["c3"], alpha=0.3)

    # Then as a line
    ax.plot(t + density_at_t_normalized, y_density, color=color_map["c3"],
            linewidth=2, label="Density at $t$")

    # Add grid
    ax.grid(True, alpha=0.15, linestyle="-", zorder=0)

    # Customize plot appearance
    ax.set_title("Brownian Motion Transition Density", fontsize=12, pad=15)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)

    # Set axis limits - only up to time t plus density width
    ax.set_xlim(0, t + t_width + 0.2)  # Added small padding
    ax.set_ylim(y_min, y_max)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)

    # Add legend
    ax.legend(frameon=True, framealpha=0.9, loc="upper left", fontsize=9)


def plot_brownian_bridge_density(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of a Brownian bridge with transition density
    at an intermediate time.

    We visualize p(beta(t) | beta(s), beta(T)=v) where s < t < T.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Define time points
    s = 0.5    # Starting time
    t = 1.5    # Intermediate time for density
    T = 2.5    # Terminal time

    # Define values at endpoints
    beta_s = 0.0    # Value at time s
    beta_T = 2.0    # Target value v at time T

    # Number of sample paths to show
    n_samples = 12

    # Create time grid
    dt = 0.01
    time_points = np.arange(s, T + dt, dt)
    n_points = len(time_points)

    # Calculate Brownian bridge parameters for time t
    # For a bridge from x at time s to y at time T:
    # Mean at time t: x + (t-s)/(T-s) * (y-x)
    # Variance at time t: (t-s)(T-t)/(T-s)
    mean_at_t = beta_s + (t-s)/(T-s) * (beta_T - beta_s)
    std_at_t = np.sqrt((t-s)*(T-t)/(T-s))

    # Generate sample paths of the Brownian bridge
    bridges = np.zeros((n_samples, n_points))

    # Function to generate a Brownian bridge from (s,x) to (T,y)
    def generate_bridge(x, y, times):
        n = len(times)
        bridge = np.zeros(n)
        bridge[0] = x  # Start point

        # Generate a standard Brownian motion
        dW = np.random.normal(0, np.sqrt(np.diff(times)))
        W = np.zeros(n)
        W[1:] = np.cumsum(dW)

        # Transform to a Brownian bridge
        for i in range(1, n):
            t_i = times[i]
            # Linear interpolation term
            mu = x + (t_i - s)/(T - s) * (y - x)
            # Brownian fluctuation term with proper scaling
            if i < n - 1:  # All but the last point
                sigma = np.sqrt((t_i - s) * (T - t_i) / (T - s))
                bridge[i] = mu + sigma * (W[i] - (t_i - s)/(T - s) * W[-1])
            else:  # Last point is fixed
                bridge[i] = y

        return bridge

    # Generate multiple bridge sample paths
    for i in range(n_samples):
        np.random.seed(42 + i)  # For reproducibility but different paths
        bridges[i] = generate_bridge(beta_s, beta_T, time_points)

    # Calculate y-limits for the plot
    y_padding = 1.0
    y_min = min(np.min(bridges), beta_s, beta_T) - y_padding
    y_max = max(np.max(bridges), beta_s, beta_T) + y_padding

    # Create density range for time t
    y_density = np.linspace(y_min, y_max, 100)
    density_at_t = norm.pdf(y_density, loc=mean_at_t, scale=std_at_t)

    # Normalize density for plotting
    max_density = np.max(density_at_t)
    density_at_t_normalized = density_at_t / \
        max_density * 0.3  # Scale for visual appeal

    # Find index for time t in the time array
    t_idx = np.abs(time_points - t).argmin()

    # Plot sample paths
    for i in range(n_samples):
        alpha = 0.3 if i < n_samples - 3 else 0.6  # Make a few paths more visible
        ax.plot(time_points, bridges[i], color=color_map["c1"],
                linewidth=1, alpha=alpha, zorder=2)

    # Plot vertical lines at s, t, and T
    ax.axvline(x=s, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=t, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=T, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # Add labels for s, t, and T
    ax.text(s, y_min - 0.4, "$s$", ha='center', va='top', fontsize=12)
    ax.text(t, y_min - 0.5, "$t$", ha='center', va='top', fontsize=12)
    ax.text(T, y_min - 0.4, "$T$", ha='center', va='top', fontsize=12)

    # Mark positions at time s and T
    ax.scatter(s, beta_s, s=80, color=color_map["c1"], zorder=5,
               edgecolor='white', linewidth=1, label=f"β(s)={beta_s}")
    ax.scatter(T, beta_T, s=80, color=color_map["c2"], zorder=5,
               edgecolor='white', linewidth=1, label=f"β(T)=v={beta_T}")

    # Plot the Gaussian density at time t
    # First as filled area
    t_width = 0.3
    ax.fill_betweenx(y_density,
                     t * np.ones_like(y_density),
                     t + density_at_t_normalized,
                     color=color_map["c3"], alpha=0.3)

    # Then as a line
    ax.plot(t + density_at_t_normalized, y_density, color=color_map["c3"],
            linewidth=2, label="p(β(t)|β(s),β(T)=v)")

    # Add annotation for density
    ax.annotate(r"$p(\beta(t)|\beta(s),\beta(T)=v)$",
                xy=(t + 0.2, mean_at_t + 1.2*std_at_t),
                xytext=(t + 0.5, mean_at_t + 2*std_at_t),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3", color="black"),
                fontsize=10, ha='center', va='center')

    # Add annotation for beta(s)
    ax.annotate(r"$\beta(s)$",
                xy=(s, beta_s),
                xytext=(s - 0.3, beta_s - 0.5),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3", color="black"),
                fontsize=10, ha='right', va='center')

    # Add grid
    ax.grid(True, alpha=0.15, linestyle="-", zorder=0)

    # Customize plot appearance
    ax.set_title("Brownian Bridge with Transition Density",
                 fontsize=12, pad=15)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)

    # Set axis limits with padding
    ax.set_xlim(s - 0.2, T + t_width)
    ax.set_ylim(y_min, y_max)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)


def plot_schoenmakers_score_matching(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of the Schoenmakers et al. (2013) score matching approach.

    Left subplot: Shows both standard Brownian motion and the learned score-matched process
    that guides it toward the target distribution π (shown as dotted curve).

    Right subplot: Shows the reverse process from π back to the standard normal.
    """
    # Create figure with two subplots side by side
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Simulation parameters
    dt = 0.01  # Time step
    T = 1.0    # Terminal time
    t = np.linspace(0, T, int(T/dt) + 1)  # Time grid from 0 to T
    # Reversed time grid from T to 0
    t_reverse = np.linspace(T, 0, int(T/dt) + 1)
    n_paths = 10  # Number of sample paths to display

    # Target distribution parameters (for π)
    # Using a mixture of two Gaussians to make it visually distinct from normal
    mix_means = [-1.0, 1.0]
    mix_stds = [0.5, 0.5]
    mix_weights = [0.5, 0.5]

    # Function to sample from target distribution π
    def sample_target_dist(n_samples):
        components = np.random.choice([0, 1], size=n_samples, p=mix_weights)
        samples = np.zeros(n_samples)
        for i in range(n_samples):
            samples[i] = np.random.normal(
                mix_means[components[i]], mix_stds[components[i]])
        return samples

    # Function to evaluate target density (with safety checks)
    def target_density(x):
        density = (mix_weights[0] * norm.pdf(x, mix_means[0], mix_stds[0]) +
                   mix_weights[1] * norm.pdf(x, mix_means[1], mix_stds[1]))
        return np.maximum(density, 1e-10)  # Ensure non-zero

    # Function to compute score (gradient of log density) safely
    def compute_score(x):
        p_x = target_density(x)
        grad_p = (mix_weights[0] * norm.pdf(x, mix_means[0], mix_stds[0]) * ((mix_means[0] - x)/mix_stds[0]**2) +
                  mix_weights[1] * norm.pdf(x, mix_means[1], mix_stds[1]) * ((mix_means[1] - x)/mix_stds[1]**2))
        return grad_p / p_x

    # Generate standard Brownian motion paths
    brownian_paths = np.zeros((n_paths, len(t)))

    for i in range(n_paths):
        dW = np.random.normal(0, np.sqrt(dt), len(t)-1)
        brownian_paths[i, 0] = 0.0  # Start at origin
        for j in range(1, len(t)):
            brownian_paths[i, j] = brownian_paths[i, j-1] + dW[j-1]

    # Generate score-guided forward paths
    guided_paths = np.zeros((n_paths, len(t)))

    for i in range(n_paths):
        guided_paths[i, 0] = 0.0  # Start at origin

        # Use the same noise sequence for direct comparison
        dW = np.random.normal(0, np.sqrt(dt), len(t)-1)

        for j in range(1, len(t)):
            # In the forward process, we add a drift term to guide toward target
            current_t = t[j-1]
            x = guided_paths[i, j-1]

            # Simple time-dependent interpolation between Brownian and target distributions
            # At t=0, we're at standard normal; at t=T, we want to reach target
            if current_t < T-dt:  # Avoid numerical issues right at t=T
                # Interpolate between initial and target distribution
                weight_target = current_t / T

                # Mix score of standard normal and target
                standard_score = -x  # Score of standard normal is -x
                target_score = compute_score(x)
                effective_score = (1-weight_target) * \
                    standard_score + weight_target * target_score

                # Forward SDE with score guidance
                drift = effective_score * dt
                guided_paths[i, j] = guided_paths[i, j-1] + drift + dW[j-1]
            else:
                # At the end, just add noise
                guided_paths[i, j] = guided_paths[i, j-1] + dW[j-1]

    # Generate reverse paths (starting from target distribution)
    reverse_paths = np.zeros((n_paths, len(t_reverse)))
    end_points = sample_target_dist(n_paths)

    for i in range(n_paths):
        reverse_paths[i, 0] = end_points[i]  # Start at sampled end point

        for j in range(1, len(t_reverse)):
            current_t = T - (j-1) * dt  # Current time in reverse process
            x = reverse_paths[i, j-1]

            # Compute score
            score = compute_score(x)

            # Reverse-time SDE: dy = [f(y,t) - g(y,t)²∇ylog(p(y,t))]dt + g(y,t)dW
            drift = -score * dt
            diffusion = np.random.normal(0, np.sqrt(dt))
            reverse_paths[i, j] = reverse_paths[i, j-1] + drift + diffusion

    # Calculate limits for consistent plotting
    all_paths = np.concatenate([brownian_paths, guided_paths, reverse_paths])
    y_min = np.min(all_paths) - 0.5
    y_max = np.max(all_paths) + 0.5

    # Density visualization parameters
    density_x = np.linspace(y_min, y_max, 200)
    standard_normal_density = norm.pdf(density_x, 0, np.sqrt(T))
    target_density_values = np.array([target_density(x) for x in density_x])

    # Normalize densities for visualization
    density_scale = 0.2
    standard_normal_density = standard_normal_density / \
        np.max(standard_normal_density) * density_scale
    target_density_values = target_density_values / \
        np.max(target_density_values) * density_scale

    # Plot standard Brownian paths on left subplot
    for i in range(1, n_paths):
        ax1.plot(t, brownian_paths[i], color=color_map["c1"],
                 linewidth=1.2, alpha=0.7)

    # Add x(t) = \beta(t) to legend
    ax1.plot(t, brownian_paths[0], color=color_map["c1"],
             linewidth=1.2, alpha=0.7, label=r"$x(t) = \beta(t)$")

    # Plot guided paths on left subplot
    for i in range(1, n_paths):
        ax1.plot(t, guided_paths[i], color=color_map["c2"],
                 linewidth=1.2, alpha=0.7)

    # Add \mathcal{y}(t) to legend
    ax1.plot(t, guided_paths[0], color=color_map["c2"],
             linewidth=1.2, alpha=0.7, label=r"$\mathcal{y}(t)$")

    # Add standard normal density at T=1 to left subplot
    ax1.fill_betweenx(density_x, T, T + standard_normal_density,
                      color=color_map["c1"], alpha=0.3)
    ax1.plot(T + standard_normal_density, density_x,
             color=color_map["c1"], linewidth=2, label=r"$\mathcal{N}(0,T)$")

    # Add target density at T=1 to left subplot (dotted to show approximation)
    ax1.plot(T + target_density_values, density_x,
             color=color_map["c8"], linewidth=2, linestyle='--', label=r"$\pi$")

    # Plot reverse paths on right subplot
    for i in range(1, n_paths):
        ax2.plot(t_reverse, reverse_paths[i], color=color_map["c7"],
                 linewidth=1.2, alpha=0.7)

    # Add x(t)= y(t) to legend
    ax2.plot(t_reverse, reverse_paths[0], color=color_map["c7"],
             linewidth=1.2, alpha=0.7, label=r"$x(t) = y(t)$")

    # Add target density at t=T to right subplot
    ax2.fill_betweenx(density_x, T, T + target_density_values,
                      color=color_map["c8"], alpha=0.3)
    ax2.plot(T + target_density_values, density_x,
             color=color_map["c8"], linewidth=2, label=r"$\pi$")

    # Customize left subplot appearance
    ax1.set_title("Forward Process (Brownian Motion)", fontsize=12, pad=15)
    ax1.set_xlabel(r"$t$", fontsize=10)
    ax1.set_ylabel("Value", fontsize=10)
    ax1.set_xlim(0, T + density_scale + 0.1)
    ax1.set_ylim(y_min, y_max)
    ax1.grid(True, alpha=0.15, linestyle="-", zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.9)

    # Customize right subplot appearance
    ax2.set_title("Reverse Process (Score Matching)", fontsize=12, pad=15)
    ax2.set_xlabel(r"$t$", fontsize=10)
    ax2.set_ylabel("Value", fontsize=10)
    ax2.set_xlim(0, T + density_scale + 0.1)
    ax2.set_ylim(y_min, y_max)
    ax2.grid(True, alpha=0.15, linestyle="-", zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.9)


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()
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
    # svg_content = plotter.create_themed_plot(
    #     save_name="brownian_VS_sin", plot_func=plot_brownian_VS_sin
    # )
    # svg_content = plotter.create_themed_plot(
    #     save_name="left_riemann_brownian", plot_func=plot_left_reimann_brownian)
    # svg_content = plotter.create_themed_plot(
    #     save_name="second_order", plot_func=plot_second_order)
    # svg_content = plotter.create_themed_plot(
    #     save_name="beta_beta_squared", plot_func=plot_beta_beta_squared
    # )
    # svg_content = plotter.create_themed_plot(
    #     save_name="beta_squared_second_order_taylor",
    #     plot_func=plot_beta_squared_second_order_taylor,
    # )
    # svg_content = plotter.create_themed_plot(
    #     save_name="ornstein_uhlenbeck", plot_func=plot_ornstein_uhlenbeck
    # )
    # svg_content = plotter.create_themed_plot(
    #     save_name="reverse_ornstein_uhlenbeck", plot_func=plot_reverse_ornstein_uhlenbeck
    # )
    # svg_content = plotter.create_themed_plot(
    #     save_name="denoising_diffusion", plot_func=plot_denoising_diffusion
    # )
    # svg_content = plotter.create_themed_plot(
    #     save_name="brownian_to_point", plot_func=plot_brownian_to_point
    # )
    # svg_content = plotter.create_themed_plot(
    #     save_name="brownian_bridge", plot_func=plot_brownian_bridge
    # )
    # svg_content = plotter.create_themed_plot(
    #     save_name="brownian_transition_density",
    #     plot_func=plot_brownian_transition_density,
    # )
    # svg_content = plotter.create_themed_plot(
    #     save_name="brownian_bridge_density",
    #     plot_func=plot_brownian_bridge_density,
    # )
    svg_content = plotter.create_themed_plot(
        save_name="schoenmakers_score_matching",
        plot_func=plot_schoenmakers_score_matching,
    )
