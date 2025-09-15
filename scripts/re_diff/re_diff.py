import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path
from matplotlib import cm
from scipy.integrate import solve_ivp
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

SAVE_PATH = Path("result") / Path(sys.argv[0]).stem
STYLE_NAME = "paper_light"
plt.style.use(f"./{STYLE_NAME}.mplstyle")


def plot_diffusion_causality(ax=None, color_map=None):
    """
    Plot a figure illustrating diffusion processes and causal relationships with
    three panels: latent perturbations, generative flow, and causal structures.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # ---- Panel 1: Latent Perturbations ----
    ax1 = axes[0]
    # Generate base gaussian noise points
    # For reproducibility
    np.random.seed(42)
    n_points = 200
    base_points = np.random.randn(n_points, 2) * 0.8

    # Create perturbation in a specific direction
    perturb_dir = np.array([1.0, 0.5])
    perturb_dir = perturb_dir / np.linalg.norm(perturb_dir)

    # Highlight a few points to be perturbed
    highlight_indices = np.random.choice(n_points, 10, replace=False)
    perturbed_points = base_points.copy()
    perturbed_points[highlight_indices] += perturb_dir * 0.5

    # Plot all points
    ax1.scatter(
        base_points[:, 0],
        base_points[:, 1],
        s=10,
        alpha=0.4,
        color="gray",
        label="Latent Variables",
    )

    # Plot arrows showing perturbation
    for idx in highlight_indices:
        ax1.arrow(
            base_points[idx, 0],
            base_points[idx, 1],
            perturb_dir[0] * 0.5,
            perturb_dir[1] * 0.5,
            head_width=0.1,
            head_length=0.1,
            fc="red",
            ec="red",
            length_includes_head=True,
            alpha=0.8,
        )
        # Highlight base points
        ax1.scatter(
            base_points[idx, 0],
            base_points[idx, 1],
            s=30,
            color="blue",
            label="Base Variables" if idx == highlight_indices[0] else "",
        )

    # Highlight perturbed points
    ax1.scatter(
        perturbed_points[highlight_indices, 0],
        perturbed_points[highlight_indices, 1],
        s=30,
        color="red",
        label="Perturbed Variables",
    )

    ax1.set_title("Latent Perturbations")
    ax1.set_xlabel("$z_1$")
    ax1.set_ylabel("$z_2$")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)

    # ---- Panel 2: Generative Flow ----
    ax2 = axes[1]

    # Create a grid for vector field
    x = np.linspace(-2, 2, 12)
    y = np.linspace(-2, 2, 12)
    X, Y = np.meshgrid(x, y)

    # Define a simple flow field
    U = -X - 0.5 * Y
    V = -Y + 0.3 * X

    # Plot vector field representing generative flow
    ax2.quiver(X, Y, U, V, alpha=0.6, scale=20)

    # Generate flow trajectories
    def flow_ode(t, state):
        x, y = state
        return [-x - 0.5 * y, -y + 0.3 * x]

    # Plot flow trajectories for the perturbed points
    t_span = [0, 1.0]
    t_eval = np.linspace(0, 1.0, 50)

    base_trajectories = []
    perturbed_trajectories = []

    # Only trace a few points to avoid cluttering
    for idx in highlight_indices[:5]:
        # Base point trajectory
        sol_base = solve_ivp(flow_ode, t_span, base_points[idx], t_eval=t_eval)
        base_trajectories.append(sol_base.y)

        # Perturbed point trajectory
        sol_perturbed = solve_ivp(
            flow_ode, t_span, perturbed_points[idx], t_eval=t_eval
        )
        perturbed_trajectories.append(sol_perturbed.y)

        # Plot trajectories with gradual color change
        colors_base = cm.Blues(np.linspace(0.3, 0.8, len(t_eval)))
        colors_perturbed = cm.Reds(np.linspace(0.3, 0.8, len(t_eval)))

        for i in range(len(t_eval) - 1):
            ax2.plot(
                sol_base.y[0, i : i + 2],
                sol_base.y[1, i : i + 2],
                color=colors_base[i],
                alpha=0.7,
                linewidth=1.5,
            )
            ax2.plot(
                sol_perturbed.y[0, i : i + 2],
                sol_perturbed.y[1, i : i + 2],
                color=colors_perturbed[i],
                alpha=0.7,
                linewidth=1.5,
            )

    # Plot starting points
    for idx in highlight_indices[:5]:
        ax2.scatter(
            base_points[idx, 0], base_points[idx, 1], color="blue", s=30, zorder=10
        )
        ax2.scatter(
            perturbed_points[idx, 0],
            perturbed_points[idx, 1],
            color="red",
            s=30,
            zorder=10,
        )

    # Plot ending points
    for traj in base_trajectories:
        ax2.scatter(
            traj[0, -1],
            traj[1, -1],
            color="darkblue",
            s=50,
            marker="o",
            facecolors="none",
            linewidth=2,
            zorder=10,
        )

    for traj in perturbed_trajectories:
        ax2.scatter(
            traj[0, -1],
            traj[1, -1],
            color="darkred",
            s=50,
            marker="o",
            facecolors="none",
            linewidth=2,
            zorder=10,
        )

    ax2.set_title("Generative Flow")
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)

    # ---- Panel 3: Causal Structures ----
    ax3 = axes[2]

    # Define feature space with abstract features (e.g., for an image: shape, color, texture)
    # We'll use the endpoints of our flow trajectories as feature values

    # For this example, let's assume:
    # - Feature 1 (x-axis) affects "shape" properties
    # - Feature 2 (y-axis) affects "texture" properties
    # - Both together affect "color" properties (interaction)

    feature_names = ["Shape", "Texture", "Color"]

    # Calculate feature values for base and perturbed trajectories
    base_features = np.zeros((len(base_trajectories), 3))
    perturbed_features = np.zeros((len(perturbed_trajectories), 3))

    for i in range(len(base_trajectories)):
        # Shape feature (mainly determined by x-coordinate)
        base_features[i, 0] = base_trajectories[i][0, -1]
        perturbed_features[i, 0] = perturbed_trajectories[i][0, -1]

        # Texture feature (mainly determined by y-coordinate)
        base_features[i, 1] = base_trajectories[i][1, -1]
        perturbed_features[i, 1] = perturbed_trajectories[i][1, -1]

        # Color feature (determined by interaction of x and y)
        base_features[i, 2] = base_trajectories[i][0, -1] * base_trajectories[i][1, -1]
        perturbed_features[i, 2] = (
            perturbed_trajectories[i][0, -1] * perturbed_trajectories[i][1, -1]
        )

    # Normalize features for visualization
    feature_min = min(np.min(base_features), np.min(perturbed_features))
    feature_max = max(np.max(base_features), np.max(perturbed_features))
    base_features_norm = (base_features - feature_min) / (feature_max - feature_min)
    perturbed_features_norm = (perturbed_features - feature_min) / (
        feature_max - feature_min
    )

    # Define causal graph node positions
    nodes = {
        "z": (0.5, 0.9),
        "Shape": (0.3, 0.7),
        "Texture": (0.7, 0.7),
        "Color": (0.5, 0.5),
    }

    # Draw nodes
    ax3.scatter(
        nodes["z"][0],
        nodes["z"][1],
        s=2200,
        color="lightgrey",
        edgecolor="black",
        zorder=1,
    )
    ax3.text(
        nodes["z"][0],
        nodes["z"][1],
        "z",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    # Style for feature nodes based on feature differences
    feature_diffs = np.mean(
        np.abs(perturbed_features_norm - base_features_norm), axis=0
    )
    norm = Normalize(vmin=0, vmax=np.max(feature_diffs) * 1.2)

    for i, feature in enumerate(feature_names):
        color = cm.Reds(norm(feature_diffs[i]))
        ax3.scatter(
            nodes[feature][0],
            nodes[feature][1],
            s=2200,
            color=color,
            edgecolor="black",
            zorder=1,
        )
        ax3.text(
            nodes[feature][0],
            nodes[feature][1],
            feature,
            ha="center",
            va="center",
            fontsize=12,
        )

    # Draw edges with different thicknesses based on causal strength
    edges = [
        ("z", "Shape", feature_diffs[0]),
        ("z", "Texture", feature_diffs[1]),
        ("z", "Color", feature_diffs[2]),
        ("Shape", "Color", feature_diffs[0] * 0.3),
        ("Texture", "Color", feature_diffs[1] * 0.3),
    ]

    for start, end, strength in edges:
        # Scale arrow width by causal strength
        width = 1 + 5 * norm(strength)
        arrow = FancyArrowPatch(
            nodes[start],
            nodes[end],
            arrowstyle="-|>",
            color="black",
            connectionstyle="arc3,rad=0.1",
            linewidth=width,
            alpha=0.8,
            zorder=0,
        )
        ax3.add_patch(arrow)

    # Add a simple bar plot showing feature differences
    ax_inset = fig.add_axes([0.73, 0.15, 0.25, 0.2])
    ax_inset.bar(
        np.arange(len(feature_names)),
        feature_diffs,
        color=["lightblue", "lightgreen", "salmon"],
    )
    ax_inset.set_xticks(np.arange(len(feature_names)))
    ax_inset.set_xticklabels([f[0] for f in feature_names], fontsize=6)
    ax_inset.set_title("Feature Δ", fontsize=10)
    ax_inset.tick_params(axis="both", which="major", labelsize=6)

    ax3.set_title("Causal Structures")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Save the figure
    os.makedirs(SAVE_PATH, exist_ok=True)


def plot_deterministic_ODE(ax=None, color_map=None):
    """
    Creates a visualization demonstrating deterministic ODE trajectories.
    Shows multiple solutions from different initial conditions and how they
    evolve deterministically without intersecting.
    """
    fig = ax.figure
    fig.set_size_inches(12, 5.5)
    ax.remove()
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Time vector
    t = np.linspace(0, 5, 100)

    # Multiple initial conditions
    initial_conditions = [0.5, 1.5, 3.0, 4.5]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(initial_conditions)))

    for i, x0 in enumerate(initial_conditions):
        x = x0 * np.exp(-t)  # Solution: x(t) = x_0 * exp(-t)
        ax1.plot(t, x, label=f"$x_0 = {x0}$", color=colors[i], linewidth=2)

        # Add a small arrow to show direction of evolution
        mid_idx = 25
        ax1.annotate(
            "",
            xy=(t[mid_idx + 5], x[mid_idx + 5]),
            xytext=(t[mid_idx], x[mid_idx]),
            arrowprops=dict(arrowstyle="->", lw=2, color=colors[i]),
        )

    ax1.set_title("1D ODE: Exponential Decay", fontsize=12)
    ax1.set_xlabel("Time ($t$)")
    ax1.set_ylabel("State ($x(t)$)")
    ax1.set_ylim(0, 4.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    # Define a 2D ODE system: dx/dt = f(x,y), dy/dt = g(x,y)
    # Using a linear system: dx/dt = -0.5x + y, dy/dt = -x - 0.5y
    def ode_system(t, state):
        x, y = state
        dx_dt = -0.5 * x + y
        dy_dt = -x - 0.5 * y
        return [dx_dt, dy_dt]

    # Create grid for vector field
    x = np.linspace(-3, 3, 15)
    y = np.linspace(-3, 3, 15)
    X, Y = np.meshgrid(x, y)

    # Calculate vector field
    U = -0.5 * X + Y
    V = -X - 0.5 * Y

    # Plot vector field
    ax2.quiver(X, Y, U, V, alpha=0.6, scale=25, color="gray")

    # Generate and plot trajectories from different initial conditions
    initial_states = [(-2, 1), (0, 2), (2, 1), (2, -1), (0, -2), (-2, 0)]

    t_span = [0, 8]
    t_eval = np.linspace(0, 8, 100)

    # Create a colormap based on the initial distance from origin
    distances = [np.sqrt(x0**2 + y0**2) for x0, y0 in initial_states]
    norm = Normalize(min(distances), max(distances))
    colormap = plt.cm.viridis

    # Solve and plot each trajectory
    for i, (x0, y0) in enumerate(initial_states):
        sol = solve_ivp(ode_system, t_span, [x0, y0], t_eval=t_eval)

        # Get color based on distance from origin
        color = colormap(norm(np.sqrt(x0**2 + y0**2)))

        # Plot trajectory
        ax2.plot(sol.y[0], sol.y[1], color=color, linewidth=1.5)

        # Mark initial point
        ax2.scatter(x0, y0, color=color, s=40, zorder=10)

        # Add arrow to show direction
        mid_idx = len(sol.t) // 3
        arrow = FancyArrowPatch(
            (sol.y[0, mid_idx], sol.y[1, mid_idx]),
            (sol.y[0, mid_idx + 1], sol.y[1, mid_idx + 1]),
            arrowstyle="->",
            color=color,
            linewidth=1.5,
            shrinkA=0,
            shrinkB=0,
        )
        ax2.add_patch(arrow)

    # Add colorbar to show relationship between color and initial distance
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, shrink=0.7)
    cbar.set_label("Distance from origin", fontsize=10)

    ax2.set_title("2D ODE System: Phase Portrait", fontsize=12)
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$y$")
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")


def euler_maruyama(drift_func, diffusion_func, x0, t_span, dt, seed=None):
    """
    Euler-Maruyama method for simulating SDE: dx = f(x,t)dt + L(x,t)dW

    Parameters:
    -----------
    drift_func : function
        The drift function f(x,t)
    diffusion_func : function
        The diffusion function L(x,t)
    x0 : array_like
        Initial state
    t_span : tuple
        (t_start, t_end)
    dt : float
        Time step size
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    t : array
        Time points
    x : array
        Solution trajectory
    """
    if seed is not None:
        np.random.seed(seed)

    # Setup time grid
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt)
    t = np.linspace(t_start, t_end, n_steps + 1)

    # Initialize state array
    x = np.zeros((len(x0), n_steps + 1))
    x[:, 0] = x0

    # Simulate trajectory
    for i in range(n_steps):
        dW = np.sqrt(dt) * np.random.normal(size=len(x0))
        x[:, i + 1] = (
            x[:, i]
            + drift_func(x[:, i], t[i]) * dt
            + diffusion_func(x[:, i], t[i]) * dW
        )

    return t, x


def plot_stochastic_SDE(ax=None, color_map=None):
    """
    Creates a visualization demonstrating stochastic SDE trajectories compared to deterministic ODE solutions.
    Shows the effect of noise (diffusion term) in creating variance around the deterministic paths.
    """
    fig = ax.figure
    fig.set_size_inches(12, 5.5)
    ax.remove()
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # 1. Simple 1D SDE in the first subplot: dx = -x dt + σ dW
    # Parameters
    t_span = (0, 5)
    dt = 0.01
    x0 = 2.0
    n_trajectories = 15

    # Define drift and diffusion functions
    def drift_1d(x, t):
        return -x  # Same as the ODE: dx/dt = -x

    # Simulate trajectories with different diffusion coefficients
    diffusion_strengths = [0.2, 0.5, 0.8]
    colors = ["#2166AC", "#4393C3", "#92C5DE"]  # Blue gradient

    # Plot deterministic solution as reference
    t_det = np.linspace(t_span[0], t_span[1], 500)
    x_det = x0 * np.exp(-t_det)
    ax1.plot(t_det, x_det, "k-", linewidth=2.5, label="Deterministic (ODE)", zorder=10)

    # Generate and plot stochastic trajectories for each diffusion strength
    for i, sigma in enumerate(diffusion_strengths):

        def diffusion_1d(x, t):
            return sigma  # Constant diffusion

        # Simulate multiple trajectories
        trajectories = []
        for j in range(n_trajectories):
            t, x = euler_maruyama(
                drift_1d, diffusion_1d, [x0], t_span, dt, seed=42 + j + i * 100
            )
            trajectories.append(x[0])
            ax1.plot(t, x[0], color=colors[i], alpha=0.15, linewidth=0.8)

        # Calculate and plot mean trajectory
        mean_trajectory = np.mean(trajectories, axis=0)
        ax1.plot(
            t,
            mean_trajectory,
            color=colors[i],
            linewidth=2,
            label=f"σ = {sigma} (mean)",
            zorder=5,
        )

        # Calculate and plot confidence intervals (±2σ)
        std_trajectory = np.std(trajectories, axis=0)
        ax1.fill_between(
            t,
            mean_trajectory - 2 * std_trajectory,
            mean_trajectory + 2 * std_trajectory,
            color=colors[i],
            alpha=0.2,
        )

    ax1.set_title("1D SDE: Exponential Decay with Noise", fontsize=12)
    ax1.set_xlabel("Time ($t$)")
    ax1.set_ylabel("State ($x(t)$)")
    ax1.set_ylim(-1.5, 3.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    # 2. 2D SDE system in the second subplot
    # Parameters for 2D system
    t_span_2d = (0, 4)
    dt_2d = 0.01
    sigma_2d = 0.4

    # Define 2D drift and diffusion functions - same as in the ODE example
    def drift_2d(x, t):
        return np.array([-0.5 * x[0] + x[1], -x[0] - 0.5 * x[1]])

    def diffusion_2d(x, t):
        return np.array([sigma_2d, sigma_2d])

    n_initial = 6
    radius = 2.0
    theta = np.linspace(0, 2 * np.pi, n_initial, endpoint=False)
    initial_states = [(radius * np.cos(th), radius * np.sin(th)) for th in theta]

    colors_2d = plt.cm.hsv(np.linspace(0, 1, n_initial))

    det_trajectories = []
    for x0 in initial_states:
        t, x = euler_maruyama(drift_2d, lambda x, t: [0, 0], x0, t_span_2d, dt_2d)
        det_trajectories.append(x)
        ax2.plot(x[0], x[1], "--", color="gray", linewidth=1, alpha=0.5)

    for i, x0 in enumerate(initial_states):
        n_paths = 5
        for j in range(n_paths):
            t, x = euler_maruyama(
                drift_2d, diffusion_2d, x0, t_span_2d, dt_2d, seed=42 + i * 100 + j
            )

            points = np.array([x[0], x[1]]).T
            segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)

            lc = plt.matplotlib.collections.LineCollection(
                segments,
                colors=[colors_2d[i]] * len(segments),
                linewidths=1.0,
                alpha=np.linspace(1.0, 0.3, len(segments)),
            )
            ax2.add_collection(lc)

            if j == 0:
                ax2.scatter(x0[0], x0[1], color=colors_2d[i], s=30, zorder=10)

    ax2.set_title("2D SDE System: Stochastic Phase Portrait", fontsize=12)
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$y$")
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    # Save the figure
    os.makedirs(SAVE_PATH, exist_ok=True)


def plot_diffusion_models(ax=None, color_map=None):
    """
    Creates a visualization demonstrating forward and reverse diffusion processes.
    Shows how data is corrupted with noise in the forward process and how
    the reverse process reconstructs data from noise.
    """
    fig = plt.figure(figsize=(15, 7.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)

    # Top row: Forward diffusion (data to noise)
    ax_forward = fig.add_subplot(gs[0])
    # Bottom row: Reverse diffusion (noise to data)
    ax_reverse = fig.add_subplot(gs[1])

    # Remove axes for the top and bottom of the figure
    for ax in [ax_forward, ax_reverse]:
        ax.axis("off")

    # ------ Create synthetic data to demonstrate diffusion ------
    # Generate a simple shape (circle) as our "data"
    size = 512
    center = size / 2
    radius = size / 4

    def create_smiley(size=512, center=None, radius=None):
        """
        Creates a clean smiley face in a numpy array for diffusion model visualization

        Parameters:
        size (int): Size of the square array
        center (float): Center point (defaults to size/2)
        radius (float): Radius of the face (defaults to size/4)

        Returns:
        numpy array with the smiley face (values between 0-1)
        """
        # Set defaults if not provided
        if center is None:
            center = size / 2
        if radius is None:
            radius = size / 4

        # Initialize the array (0 = black, 1 = white/yellow)
        smiley = np.zeros((size, size))

        # Create the face (circle)
        y, x = np.ogrid[:size, :size]
        face_mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2
        smiley[face_mask] = 1

        # Create eyes (two small circles in upper half)
        eye_radius = radius / 10
        eye_y = center - radius / 3  # Position eyes in the upper part of the face

        # Left eye
        left_eye_x = center - radius / 3
        left_eye_mask = (x - left_eye_x) ** 2 + (y - eye_y) ** 2 <= eye_radius**2
        smiley[left_eye_mask] = 0

        # Right eye
        right_eye_x = center + radius / 3
        right_eye_mask = (x - right_eye_x) ** 2 + (y - eye_y) ** 2 <= eye_radius**2
        smiley[right_eye_mask] = 0

        # Create smile (arc in lower half of face)
        for i in range(size):
            for j in range(size):
                # Calculate position relative to where we want the smile
                x_rel = (j - center) / radius
                y_pos = center + radius * 0.3  # Position smile in lower part of face

                # Create the curved smile shape
                smile_y = y_pos - radius * 0.6 * (x_rel * x_rel) - 15

                # Define thickness and width of smile
                if abs(x_rel) < 0.6 and abs(i - smile_y) < radius * 0.08:
                    smiley[i, j] = 0

        # Apply slight Gaussian blur to smooth edges
        smiley = gaussian_filter(smiley, sigma=0.7)

        return smiley

    # Generate grayscale smiley
    smiley = create_smiley(size, center, radius)

    # Convert to RGB (yellowish smiley face)
    data_image = np.zeros((size, size, 3))
    data_image[:, :, 0] = smiley  # Red channel
    data_image[:, :, 1] = smiley * 0.8  # Green channel
    data_image[:, :, 2] = smiley * 0.2  # Blue channel

    # ------ Forward Diffusion Process (Data to Noise) ------
    n_steps = 5  # Number of diffusion steps to show

    # Initialize array for diffusion steps
    forward_images = np.zeros((n_steps, size, size, 3))
    forward_images[0] = data_image

    # Simulate forward diffusion by adding increasing amounts of noise
    noise_levels = np.linspace(0.1, 1.0, n_steps - 1)

    for i in range(1, n_steps):
        # Start with previous image
        noisy_image = forward_images[i - 1].copy()

        # Add noise proportional to current step
        noise = np.random.normal(0, noise_levels[i - 1], noisy_image.shape)
        noisy_image += noise

        # Apply slight blur to simulate diffusion effect
        noisy_image = gaussian_filter(noisy_image, sigma=0.5)

        # Clip values to keep in valid range
        noisy_image = np.clip(noisy_image, 0, 1)

        forward_images[i] = noisy_image

    # Set the last image to be pure noise
    forward_images[-1] = np.random.normal(0.5, 0.15, data_image.shape)
    forward_images[-1] = np.clip(forward_images[-1], 0, 1)

    # ------ Reverse Diffusion Process (Noise to Data) ------
    # Start with pure noise (similar to last step of forward process)
    reverse_images = np.zeros((n_steps, size, size, 3))
    reverse_images[0] = np.random.normal(0.5, 0.15, data_image.shape)
    reverse_images[0] = np.clip(reverse_images[0], 0, 1)

    # Simulate reverse diffusion
    # For illustration, we'll use a combination of denoising and morphing
    # toward the original image to simulate the learned denoising process

    for i in range(1, n_steps):
        # Use noise level proportional to steps remaining
        remaining_noise = noise_levels[-i]

        # Blend between noisy and original image with increasing weight to original
        # Non-linear blending for better visualization
        blend_factor = (i / (n_steps - 1)) ** 2

        # Add controlled noise proportional to current step
        noise = np.random.normal(
            0, remaining_noise * (1 - blend_factor), data_image.shape
        )

        # Combine noise with increasingly stronger "score-based" guidance toward true data
        reverse_images[i] = (1 - blend_factor) * (
            reverse_images[i - 1] + noise
        ) + blend_factor * data_image

        # Apply slight blur to simulate diffusion
        reverse_images[i] = gaussian_filter(
            reverse_images[i], sigma=max(0.1, 0.5 * (1 - blend_factor))
        )

        # Clip values
        reverse_images[i] = np.clip(reverse_images[i], 0, 1)

    # ------ Plot Forward Process ------
    # Create grid of images
    for i in range(n_steps):
        ax = fig.add_subplot(2, n_steps, i + 1)
        ax.imshow(forward_images[i])
        ax.set_xticks([])
        ax.set_yticks([])

        # Label time steps
        if i == 0:
            ax.set_title(
                "Original Data\n$t=0$",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, pad=0.2),
            )
        elif i == n_steps - 1:
            ax.set_title(
                "Pure Noise\n$t=T$",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, pad=0.2),
            )
        else:
            ax.set_title(f"$t={i}\\Delta t$", fontsize=10)

        if i < n_steps - 1:
            ax.annotate(
                "",
                xy=(1.25, 0.5),
                xytext=(1.05, 0.5),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=2, color="black"),
            )

    # Add equations and annotations
    fig.text(
        0.02,
        0.98,
        "Forward SDE:\n$dx = f(x,t)dt + g(t)dW_t$",
        fontsize=10,
        bbox=dict(
            facecolor="white", edgecolor="gray", alpha=0.8, boxstyle="round,pad=0.3"
        ),
    )

    fig.text(
        0.02,
        0.92,
        "Gradually adding\nnoise to data",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # ------ Plot Reverse Process ------
    # Create grid of images
    for i in range(n_steps):
        ax = fig.add_subplot(2, n_steps, n_steps + i + 1)
        ax.imshow(reverse_images[i])
        ax.set_xticks([])
        ax.set_yticks([])

        # Label time steps (reverse order)
        if i == 0:
            ax.set_title(
                "Pure Noise\n$t=T$",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, pad=0.2),
            )
        elif i == n_steps - 1:
            ax.set_title(
                "Generated Data\n$t=0$",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, pad=0.2),
            )
        else:
            ax.set_title(f"$t={n_steps - i - 1}\\Delta t$", fontsize=10)

        if i < n_steps - 1:
            ax.annotate(
                "",
                xy=(1.25, 0.5),
                xytext=(1.05, 0.5),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=2, color="black"),
            )

    # Add equations and annotations for reverse process
    fig.text(
        0.02,
        0.02,
        "Reverse SDE:\n$dx = [f(x,t) - g(t)^2\\nabla_x \\log p_t(x)]dt + g(t)d\\bar{W}_t$",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    fig.text(
        0.02,
        -0.05,
        "Score-based model\nguides denoising",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Add row titles
    fig.text(
        0.5,
        0.9,
        "Forward Diffusion Process (Data → Noise)",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.47,
        "Reverse Generation Process (Noise → Data)",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the figure
    os.makedirs(SAVE_PATH, exist_ok=True)


def plot_flow_matching(ax=None, color_map=None):
    """
    Creates a visualization demonstrating the Flow Matching framework.
    Shows how a velocity field transforms a simple Gaussian distribution
    into a more complex target distribution in a continuous manner.
    """
    fig = plt.figure(figsize=(12, 8))

    # Create 2 rows:
    # - Row 1: Distribution plots
    # - Row 2: Vector field plots
    gs = gridspec.GridSpec(2, 5, height_ratios=[1, 1], wspace=0.05, hspace=0.3)

    # Define the simple initial distribution (Gaussian)
    def gaussian_distribution(x, y, mean=[0, 0], cov=[[1, 0], [0, 1]]):
        return multivariate_normal.pdf(np.dstack([x, y]), mean=mean, cov=cov)

    # Define a more complex target distribution (mixture of Gaussians)
    def mixture_distribution(x, y, means, covs, weights):
        result = np.zeros_like(x)
        for mean, cov, weight in zip(means, covs, weights):
            result += weight * gaussian_distribution(x, y, mean, cov)
        return result / np.sum(weights)

    # Define grid for evaluating densities
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)

    # Define initial distribution: standard Gaussian
    initial_mean = [0, 0]
    initial_cov = [[1, 0], [0, 1]]
    initial_density = gaussian_distribution(X, Y, initial_mean, initial_cov)

    # Define target distribution: mixture of Gaussians
    target_means = [[-2, -1.5], [1.5, 2], [2, -1.5]]
    target_covs = [
        [[0.5, 0], [0, 0.5]],
        [[0.7, 0.3], [0.3, 0.7]],
        [[0.6, -0.2], [-0.2, 0.4]],
    ]
    target_weights = [0.3, 0.4, 0.3]
    target_density = mixture_distribution(
        X, Y, target_means, target_covs, target_weights
    )

    # Define intermediate distributions for t=0.25, t=0.5, and t=0.75
    time_steps = [0, 0.25, 0.5, 0.75, 1.0]
    all_densities = []

    for t in time_steps:
        if t == 0:
            all_densities.append(initial_density)
        elif t == 1.0:
            all_densities.append(target_density)
        else:
            # Interpolate means
            means_t = [
                (1 - t) * np.array(initial_mean) + t * np.array(mean)
                for mean in target_means
            ]

            # Interpolate covariances
            covs_t = []
            for i in range(len(target_covs)):
                cov_t = (1 - t) * np.array(initial_cov) + t * np.array(target_covs[i])
                # Ensure covariance matrix is positive definite
                covs_t.append(cov_t)

            # Calculate intermediate distribution as mixture
            density_t = mixture_distribution(X, Y, means_t, covs_t, target_weights)
            all_densities.append(density_t)

    # Define velocity field for visualization
    def velocity_field(x_grid, y_grid, t, initial_params, target_params):
        # Extract parameters
        initial_mean, initial_cov = initial_params
        target_means, target_covs, target_weights = target_params

        # Initialize velocities
        vx = np.zeros_like(x_grid)
        vy = np.zeros_like(y_grid)

        # For each point, calculate the weighted velocity based on the target mixture components
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):
                x, y = x_grid[i, j], y_grid[i, j]

                # Start with position relative to initial distribution
                pos = np.array([x, y])

                # Calculate influence of each mixture component
                total_velocity = np.zeros(2)
                total_weight = 0

                for k, (target_mean, target_cov, weight) in enumerate(
                    zip(target_means, target_covs, target_weights)
                ):
                    # Calculate intermediate mean for this component
                    mean_t = (1 - t) * np.array(initial_mean) + t * np.array(
                        target_mean
                    )

                    # Distance to this component's mean
                    dist = np.linalg.norm(pos - mean_t)

                    # Gaussian weighting based on distance
                    component_weight = weight * np.exp(-0.5 * dist**2)
                    total_weight += component_weight

                    # Direction toward this component's target
                    direction = np.array(target_mean) - np.array(initial_mean)
                    total_velocity += component_weight * direction

                if total_weight > 0:
                    total_velocity /= total_weight

                    # Scale by remaining time
                    velocity = total_velocity * (1 - t)

                    vx[i, j] = velocity[0]
                    vy[i, j] = velocity[1]

        return vx, vy

    # Calculate velocity fields for each time step
    initial_params = (initial_mean, initial_cov)
    target_params = (target_means, target_covs, target_weights)

    # Create a sparser grid for the velocity field visualization
    x_sparse = np.linspace(-4, 4, 20)
    y_sparse = np.linspace(-4, 4, 20)
    X_sparse, Y_sparse = np.meshgrid(x_sparse, y_sparse)

    velocity_fields = []
    for t in time_steps:  # Calculate velocity fields for all time steps
        if t < 1.0:  # No velocity field needed for t=1.0
            vx, vy = velocity_field(
                X_sparse, Y_sparse, t, initial_params, target_params
            )
            velocity_fields.append((vx, vy))
        else:
            velocity_fields.append((np.zeros_like(X_sparse), np.zeros_like(Y_sparse)))

    # Create a custom colormap for density visualization
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap",
        [(0.95, 0.95, 0.98), (0.7, 0.75, 0.9), (0.3, 0.4, 0.8), (0.1, 0.2, 0.6)],
    )

    # Calculate density normalization across all time steps
    density_max = max([np.max(d) for d in all_densities])

    # Create the top row - density distributions
    for i, t in enumerate(time_steps):
        ax = fig.add_subplot(gs[0, i])

        # Plot density contours without velocity field
        ax.contourf(
            X,
            Y,
            all_densities[i],
            levels=50,
            cmap=density_cmap,
            vmin=0,
            vmax=density_max,
        )

        ax.set_aspect("equal")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

        # Remove ticks for cleaner appearance
        ax.set_xticks([])
        ax.set_yticks([])

        # Add time step label
        ax.set_title(f"$t = {t:.2f}$", fontsize=12, pad=5)

        # Add distribution type in a box above the plot
        dist_type = ""
        if i == 0:
            dist_type = "Initial\nDistribution\n$p_0$"
        elif i == len(time_steps) - 1:
            dist_type = "Target\nDistribution\n$p_1$"
        else:
            dist_type = f"Intermediate\nDistribution\n$p_{{{t:.2f}}}$"

        # Create a text box above the plot
        ax.text(
            0,
            5.5,
            dist_type,
            fontsize=10,
            ha="center",
            bbox=dict(
                facecolor="white",
                alpha=0.8,
                edgecolor="lightgray",
                boxstyle="round,pad=0.3",
            ),
        )

    # Create the bottom row - vector fields
    for i, t in enumerate(time_steps):
        ax = fig.add_subplot(gs[1, i])

        # Add time step label
        ax.set_title(f"$t = {t:.2f}$", fontsize=12, pad=5)

        # For t=0 and t=1, just show the density
        if t == 0.0 or t == 1.0:
            ax.contourf(
                X,
                Y,
                all_densities[i],
                levels=30,
                cmap=density_cmap,
                vmin=0,
                vmax=density_max,
            )
        else:
            # For intermediate time steps, show velocity field
            vx, vy = velocity_fields[i]

            # Only draw meaningful arrows
            magnitude = np.sqrt(vx**2 + vy**2)
            if np.max(magnitude) > 1e-10:  # Avoid division by zero
                scale_factor = 1.0 / np.max(magnitude) * 2.0

                ax.quiver(
                    X_sparse,
                    Y_sparse,
                    vx,
                    vy,
                    color="red",
                    alpha=0.8,
                    scale_units="xy",
                    scale=1 / scale_factor,
                    width=0.005,
                    headwidth=3,
                    headlength=4,
                )

        ax.set_aspect("equal")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

        # Remove ticks for cleaner appearance
        ax.set_xticks([])
        ax.set_yticks([])

    # Add velocity field label in the middle of the middle plot in the top row
    velocity_label = fig.add_subplot(gs[0, 2])
    velocity_label.set_zorder(-1)
    velocity_label.axis("off")
    velocity_label.text(
        0.5,
        0,
        "Velocity Field\n$v_θ(x_t, t)$",
        fontsize=10,
        ha="center",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="lightgray",
            boxstyle="round,pad=0.3",
        ),
    )

    # Add caption under velocity_label
    caption = fig.add_subplot(gs[0, 2])
    caption.axis("off")
    caption.text(
        0.5,
        -0.2,
        "Flow Matching learns a velocity field $v_\\theta(x_t, t)$ that guides the transformation\n"
        + "from a simple initial distribution $p_0$ to a complex target distribution $p_1$.",
        ha="center",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.9),
    )

    # Add equation to the left of caption
    eq_ax = fig.add_subplot(gs[0, 0])
    eq_ax.axis("off")
    eq_ax.text(
        0.5,
        -0.15,
        r"$\min_\theta \ \mathbb{E}_{x_t,t}[\|v_\theta(x_t,t) - u(x_t,t)\|^2]$",
        fontsize=9,
        ha="center",
        va="center",
        bbox=dict(
            facecolor="white",
            alpha=0.9,
            edgecolor="lightgray",
            boxstyle="round,pad=0.5",
        ),
    )

    # Add process description to the right of the caption
    process_ax = fig.add_subplot(gs[0, 4])
    process_ax.axis("off")
    process_ax.text(
        0.5,
        -0.15,
        r"Process: $p_0(x) \Rightarrow p_t(x) \Rightarrow p_1(x)$",
        fontsize=9,
        ha="center",
        va="center",
        bbox=dict(
            facecolor="white",
            alpha=0.9,
            edgecolor="lightgray",
            boxstyle="round,pad=0.5",
        ),
    )

    # Save the figure
    os.makedirs(SAVE_PATH, exist_ok=True)


def plot_linear_nonlinear_transformations(ax=None, color_map=None):
    """
    Creates a clean visualization contrasting how perturbations propagate through
    linear vs. nonlinear transformations.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)

    # Create color maps
    latent_cmap = LinearSegmentedColormap.from_list(
        "latent_cmap", [(0.9, 0.9, 0.98), (0.7, 0.7, 0.9), (0.4, 0.4, 0.8)]
    )
    data_cmap = LinearSegmentedColormap.from_list(
        "data_cmap", [(0.98, 0.9, 0.9), (0.9, 0.7, 0.7), (0.8, 0.4, 0.4)]
    )

    # ---- Panel 1: Linear Transformation ----
    ax_linear = fig.add_subplot(gs[0])

    # Define grid for visualization
    resolution = 100
    z1 = np.linspace(-3, 3, resolution)
    z2 = np.linspace(-3, 3, resolution)
    Z1, Z2 = np.meshgrid(z1, z2)

    # Create Gaussian latent distribution
    latent_mean = np.array([0, 0])
    latent_cov = np.array([[1.0, 0], [0, 1.0]])
    latent_dist = multivariate_normal.pdf(
        np.dstack([Z1, Z2]), mean=latent_mean, cov=latent_cov
    )

    # Define linear transformation matrix (a rotation + scaling)
    theta = np.pi / 4  # 45-degree rotation
    scaling_x, scaling_y = 1.5, 0.7  # Scaling factors
    A = np.array(
        [
            [scaling_x * np.cos(theta), -scaling_y * np.sin(theta)],
            [scaling_x * np.sin(theta), scaling_y * np.cos(theta)],
        ]
    )

    # Apply linear transformation to grid points
    points = np.vstack((Z1.flatten(), Z2.flatten())).T
    transformed_points = np.dot(points, A.T)
    X1 = transformed_points[:, 0].reshape(Z1.shape)
    X2 = transformed_points[:, 1].reshape(Z2.shape)

    # Calculate transformed density
    det_J = np.abs(np.linalg.det(A))
    data_dist = latent_dist / det_J

    # --- Visualization setup ---
    # Split the axis into latent space (left) and data space (right)
    ax_linear.axvline(x=0, color="black", linestyle="-", linewidth=1.5)

    # Plot original distribution contours in latent space
    latent_levels = np.linspace(0, np.max(latent_dist) * 0.95, 15)
    ax_linear.contour(
        Z1 - 4, Z2, latent_dist, levels=latent_levels, cmap=latent_cmap, alpha=0.7
    )

    # Plot transformed distribution contours in data space
    data_levels = np.linspace(0, np.max(data_dist) * 0.95, 15)
    ax_linear.contour(
        X1 + 4, X2, data_dist, levels=data_levels, cmap=data_cmap, alpha=0.7
    )

    # Add distribution labels
    ax_linear.text(
        -4,
        3,
        r"$Z \sim \mathcal{N}(0, I_2)$",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
        ha="center",
    )
    ax_linear.text(
        4,
        3,
        r"$X = AZ \sim \mathcal{N}(0, AA^T)$",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
        ha="center",
    )

    # Add title and transformation equation
    ax_linear.set_title("Linear Transformation", fontsize=14, pad=10)
    ax_linear.text(
        0,
        6,
        r"$X = AZ$",
        fontsize=13,
        ha="center",
        bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.3"),
    )

    # ---- Visualize perturbations ----
    # Define clearer perturbations in latent space
    base_points_z = np.array([[-1.5, 1], [-1.5, -1]])
    perturbation_z = np.array([1.0, 0])  # Simpler perturbation in z1 direction
    perturbed_points_z = base_points_z + perturbation_z

    for i in range(len(base_points_z)):
        # Base points
        ax_linear.scatter(
            base_points_z[i, 0] - 4,
            base_points_z[i, 1],
            color="blue",
            s=40,
            zorder=10,
            edgecolor="black",
        )

        # Perturbation arrows
        ax_linear.arrow(
            base_points_z[i, 0] - 4,
            base_points_z[i, 1],
            perturbation_z[0],
            perturbation_z[1],
            head_width=0.2,
            head_length=0.3,
            fc="blue",
            ec="black",
            length_includes_head=True,
            linewidth=1,
            zorder=9,
        )

        # Perturbed points
        ax_linear.scatter(
            perturbed_points_z[i, 0] - 4,
            perturbed_points_z[i, 1],
            color="blue",
            s=40,
            zorder=10,
            alpha=0.6,
            edgecolor="black",
        )

    # Show corresponding points and perturbations in data space
    base_points_x = np.dot(base_points_z, A.T)
    perturbed_points_x = np.dot(perturbed_points_z, A.T)
    perturbation_x = perturbed_points_x - base_points_x

    for i in range(len(base_points_x)):
        # Base points
        ax_linear.scatter(
            base_points_x[i, 0] + 4,
            base_points_x[i, 1],
            color="red",
            s=40,
            zorder=10,
            edgecolor="black",
        )

        # Perturbation arrows
        ax_linear.arrow(
            base_points_x[i, 0] + 4,
            base_points_x[i, 1],
            perturbation_x[i, 0],
            perturbation_x[i, 1],
            head_width=0.2,
            head_length=0.3,
            fc="red",
            ec="black",
            length_includes_head=True,
            linewidth=1,
            zorder=9,
        )

        # Perturbed points
        ax_linear.scatter(
            perturbed_points_x[i, 0] + 4,
            perturbed_points_x[i, 1],
            color="red",
            s=40,
            zorder=10,
            alpha=0.6,
            edgecolor="black",
        )

    # ---- Panel 2: Nonlinear Transformation ----
    ax_nonlinear = fig.add_subplot(gs[1])

    # Define a nonlinear transformation
    def nonlinear_transform(z1, z2):
        # A nonlinear transformation with varying Jacobian
        x1 = z1 * (1 + 0.2 * z2**2)
        x2 = z2 * (1 + 0.3 * z1**2)  # Made more nonlinear for clarity
        return x1, x2

    # Apply nonlinear transformation to grid points
    X1_nonlin, X2_nonlin = nonlinear_transform(Z1, Z2)

    # Calculate Jacobian for the nonlinear transformation
    def jacobian(z1, z2):
        # Partial derivatives
        df1_dz1 = 1 + 0.2 * z2**2
        df1_dz2 = 2 * 0.2 * z1 * z2
        df2_dz1 = 2 * 0.3 * z1 * z2
        df2_dz2 = 1 + 0.3 * z1**2

        return np.array([[df1_dz1, df1_dz2], [df2_dz1, df2_dz2]])

    # --- Visualization setup ---
    # Split the axis into latent space (left) and data space (right)
    ax_nonlinear.axvline(x=0, color="black", linestyle="-", linewidth=1.5)

    # Draw simplified coordinate systems

    # Plot original distribution contours in latent space
    ax_nonlinear.contour(
        Z1 - 4, Z2, latent_dist, levels=latent_levels, cmap=latent_cmap, alpha=0.7
    )

    # For nonlinear transformations, we need to visualize the transformed grid
    # Create a sparser grid for cleaner visualization
    grid_points = 8
    grid_z1, grid_z2 = np.meshgrid(
        np.linspace(-2.5, 2.5, grid_points), np.linspace(-2.5, 2.5, grid_points)
    )
    grid_x1, grid_x2 = nonlinear_transform(grid_z1, grid_z2)

    # Plot simplified grid
    for i in range(grid_points):
        ax_nonlinear.plot(
            grid_z1[i, :] - 4, grid_z2[i, :], "b-", alpha=0.2, linewidth=1
        )
        ax_nonlinear.plot(
            grid_z1[:, i] - 4, grid_z2[:, i], "b-", alpha=0.2, linewidth=1
        )
        ax_nonlinear.plot(
            grid_x1[i, :] + 4, grid_x2[i, :], "r-", alpha=0.2, linewidth=1
        )
        ax_nonlinear.plot(
            grid_x1[:, i] + 4, grid_x2[:, i], "r-", alpha=0.2, linewidth=1
        )

    # Add distribution labels
    ax_nonlinear.text(
        -4,
        3,
        r"$Z \sim \mathcal{N}(0, I_2)$",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
        ha="center",
    )
    ax_nonlinear.text(
        4,
        3,
        r"$X = f(Z)$",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
        ha="center",
    )

    # Add title and transformation equation
    ax_nonlinear.set_title("Nonlinear Transformation", fontsize=14, pad=10)
    ax_nonlinear.text(
        0,
        6,
        r"$X = f(Z)$",
        fontsize=13,
        ha="center",
        bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.3"),
    )

    # ---- Visualize perturbations with local Jacobians ----
    # Use the same base points as in linear case for easy comparison
    base_points_z = np.array([[-1.5, 1], [-1.5, -1]])
    perturbation_z = np.array([1.0, 0])
    perturbed_points_z = base_points_z + perturbation_z

    for i in range(len(base_points_z)):
        # Base points
        ax_nonlinear.scatter(
            base_points_z[i, 0] - 4,
            base_points_z[i, 1],
            color="blue",
            s=40,
            zorder=10,
            edgecolor="black",
        )

        # Perturbation arrows
        ax_nonlinear.arrow(
            base_points_z[i, 0] - 4,
            base_points_z[i, 1],
            perturbation_z[0],
            perturbation_z[1],
            head_width=0.2,
            head_length=0.3,
            fc="blue",
            ec="black",
            length_includes_head=True,
            linewidth=1,
            zorder=9,
        )

        # Perturbed points
        ax_nonlinear.scatter(
            perturbed_points_z[i, 0] - 4,
            perturbed_points_z[i, 1],
            color="blue",
            s=40,
            zorder=10,
            alpha=0.6,
            edgecolor="black",
        )

    # Apply nonlinear transformation to base and perturbed points
    base_points_x = np.zeros_like(base_points_z)
    perturbed_points_x = np.zeros_like(perturbed_points_z)
    jacobian_predictions = np.zeros_like(perturbed_points_z)

    for i in range(len(base_points_z)):
        # Transform base point
        base_points_x[i, 0], base_points_x[i, 1] = nonlinear_transform(
            base_points_z[i, 0], base_points_z[i, 1]
        )

        # Transform perturbed point
        perturbed_points_x[i, 0], perturbed_points_x[i, 1] = nonlinear_transform(
            perturbed_points_z[i, 0], perturbed_points_z[i, 1]
        )

        # Calculate perturbation predicted by local Jacobian
        J = jacobian(base_points_z[i, 0], base_points_z[i, 1])
        jacobian_prediction = np.dot(J, perturbation_z)

        # Store Jacobian prediction
        jacobian_predictions[i] = jacobian_prediction

    # Plot transformed points and actual perturbations in data space
    for i in range(len(base_points_x)):
        # Base points
        ax_nonlinear.scatter(
            base_points_x[i, 0] + 4,
            base_points_x[i, 1],
            color="red",
            s=40,
            zorder=10,
            edgecolor="black",
        )

        # Actual perturbation arrows (solid)
        perturbation_x = perturbed_points_x[i] - base_points_x[i]
        ax_nonlinear.arrow(
            base_points_x[i, 0] + 4,
            base_points_x[i, 1],
            perturbation_x[0],
            perturbation_x[1],
            head_width=0.2,
            head_length=0.3,
            fc="red",
            ec="black",
            length_includes_head=True,
            linewidth=1,
            zorder=9,
        )

        # Local Jacobian predicted perturbation (dashed)
        ax_nonlinear.arrow(
            base_points_x[i, 0] + 4,
            base_points_x[i, 1],
            jacobian_predictions[i, 0],
            jacobian_predictions[i, 1],
            head_width=0.2,
            head_length=0.3,
            fc="green",
            ec="black",
            length_includes_head=True,
            linewidth=1,
            zorder=8,
            alpha=0.8,
        )

        # Perturbed points
        ax_nonlinear.scatter(
            perturbed_points_x[i, 0] + 4,
            perturbed_points_x[i, 1],
            color="red",
            s=40,
            zorder=10,
            alpha=0.6,
            edgecolor="black",
        )

    # Add legend for perturbations
    ax_nonlinear.plot([], [], "r-", linewidth=1, label="Actual Perturbation")
    ax_nonlinear.plot([], [], "g-", linewidth=1, label="Jacobian Prediction")
    ax_nonlinear.legend(loc="center right", fontsize=5, framealpha=0.9)

    # Set axis limits and remove ticks for cleaner appearance
    for ax in [ax_linear, ax_nonlinear]:
        ax.set_xlim(-8, 8)
        ax.set_ylim(-4, 4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.axvspan(
            -8, 0, facecolor="#f0f0ff", alpha=0.2, zorder=-10
        )  # Light blue for latent
        ax.axvspan(
            0, 8, facecolor="#fff0f0", alpha=0.2, zorder=-10
        )  # Light red for data

    # Save the figure
    os.makedirs(SAVE_PATH, exist_ok=True)


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()
    # svg_content = plotter.create_themed_plot(
    #     save_name="diffusion_causality", plot_func=plot_diffusion_causality, is_3d=False)
    # svg_content = plotter.create_themed_plot(
    #     save_name="deterministic_ODE", plot_func=plot_deterministic_ODE
    # )
    svg_content = plotter.create_themed_plot(
        save_name="stochastic_SDE", plot_func=plot_stochastic_SDE
    )
    # svg_content = plotter.create_themed_plot(
    #     save_name="diffusion_models", plot_func=plot_diffusion_models, is_3d=False)
    # svg_content = plotter.create_themed_plot(
    #     save_name="flow_matching", plot_func=plot_flow_matching, is_3d=False)
    # svg_content = plotter.create_themed_plot(
    #     save_name="linear_nonlinear_transformations",
    #     plot_func=plot_linear_nonlinear_transformations,
    #     is_3d=False,
