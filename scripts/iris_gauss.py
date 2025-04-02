import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_gaussian_conditionals(ax=None, color_map=None):
    """
    Create a clean, blog-friendly histogram of iris petal lengths with Gaussian fits.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Get petal length (feature index 2)
    petal_length = X[:, 2]

    # Calculate optimal number of bins using Freedman-Diaconis rule
    n_samples = len(petal_length)
    iqr = np.percentile(petal_length, 75) - np.percentile(petal_length, 25)
    bin_width = 2 * iqr / (n_samples ** (1 / 3))
    n_bins = int(np.ceil((petal_length.max() - petal_length.min()) / bin_width))

    # Create smooth x-axis for Gaussian curves
    x_smooth = np.linspace(petal_length.min() - 0.5, petal_length.max() + 0.5, 200)

    # Plot histograms and Gaussian fits for each species
    for i, species in enumerate(iris.target_names):
        mask = y == i
        species_data = petal_length[mask]

        # Plot normalized histogram
        counts, bins, _ = ax.hist(
            species_data,
            bins=n_bins,
            density=True,
            color=color_map[f"c{i + 1}"],
            alpha=0.3,
            label=f"{species} (data)",
            edgecolor="none",
        )

        # Fit and plot Gaussian
        mu = np.mean(species_data)
        sigma = np.std(species_data)
        gaussian = norm.pdf(x_smooth, mu, sigma)
        ax.plot(
            x_smooth,
            gaussian,
            "--",
            color=color_map[f"c{i + 1}"],
            label=f"{species} (Gaussian)",
            linewidth=2,
            alpha=0.8,
        )

    # Customize plot appearance
    ax.set_title(
        "Iris Petal Length Distribution with Gaussian Fits", fontsize=12, pad=15
    )
    ax.set_xlabel("Petal Length (cm)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)

    # Clean legend
    ax.legend(
        frameon=True, framealpha=0.9, loc="upper right", fontsize=9, ncol=2
    )  # Two columns for cleaner legend

    # Subtle grid
    ax.grid(True, alpha=0.15, linestyle="-")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set consistent limits with padding
    ax.set_xlim(petal_length.min() - 0.5, petal_length.max() + 0.5)

    # Set y-axis limit based on maximum density
    max_density = max(
        norm.pdf(mu, mu, sigma)
        for mu, sigma in [
            (np.mean(petal_length[y == i]), np.std(petal_length[y == i]))
            for i in range(3)
        ]
    )
    ax.set_ylim(0, max_density * 1.1)


if __name__ == "__main__":
    from rdp import RDP

    # Create and save the plot
    plotter = RDP()
    svg_content = plotter.create_themed_plot(
        save_name="iris_gaussian", plot_func=plot_gaussian_conditionals
    )
