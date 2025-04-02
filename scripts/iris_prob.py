import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_probabilities(ax=None, color_map=None):
    """
    Create clean, blog-friendly plots of class conditionals and posteriors for Iris dataset.
    Shows p(x|y) (class conditionals) and p(y|x) (posterior probabilities) for petal length.

    Args:
        ax: Matplotlib axis object (unused, following RDP convention)
        color_map: Dictionary of colors for consistent styling
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Load and prepare data
    iris = load_iris()
    X = iris.data
    y = iris.target
    petal_length = X[:, 2]
    classes = iris.target_names

    # Calculate priors
    class_counts = np.bincount(y)
    priors = class_counts / len(y)

    # Create smooth x-range for plotting
    x_range = np.linspace(petal_length.min() - 0.5, petal_length.max() + 0.5, 200)

    # Plot 1: Class Conditionals p(x|y)
    for i, (species, color) in enumerate(
        zip(classes, [color_map[f"c{j + 1}"] for j in range(3)])
    ):
        # Fit KDE
        mask = y == i
        kde = gaussian_kde(petal_length[mask], bw_method="silverman")

        # Plot density curve
        density = kde(x_range)
        ax1.plot(x_range, density, color=color, linewidth=2, label=species, zorder=3)
        ax1.fill_between(x_range, density, color=color, alpha=0.2, zorder=2)

    # Style first subplot
    ax1.set_title("p(x|y) - Class Conditionals", fontsize=12, pad=15)
    ax1.set_xlabel("Petal Length (cm)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)

    # Plot 2: Posteriors p(y|x)
    # Fit KDE for each class
    kdes = [gaussian_kde(petal_length[y == i], bw_method="silverman") for i in range(3)]

    # Calculate posteriors
    posteriors = np.zeros((len(x_range), 3))
    for i in range(3):
        likelihood = kdes[i](x_range)
        posteriors[:, i] = likelihood * priors[i]

    # Normalize posteriors
    posteriors /= posteriors.sum(axis=1, keepdims=True)

    # Plot posterior probabilities
    for i, (species, color) in enumerate(
        zip(classes, [color_map[f"c{j + 1}"] for j in range(3)])
    ):
        ax2.plot(
            x_range, posteriors[:, i], color=color, linewidth=2, label=species, zorder=3
        )
        ax2.fill_between(x_range, posteriors[:, i], color=color, alpha=0.2, zorder=2)

    # Style second subplot
    ax2.set_title("p(y|x) - Posterior Probabilities", fontsize=12, pad=15)
    ax2.set_xlabel("Petal Length (cm)", fontsize=10)
    ax2.set_ylabel("Probability", fontsize=10)

    # Common styling for both subplots
    for ax in [ax1, ax2]:
        # Clean legend
        ax.legend(frameon=True, framealpha=0.9, loc="upper right", fontsize=9)

        # Subtle grid
        ax.grid(True, alpha=0.15, linestyle="-", zorder=1)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Set y-axis limits
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

        # Set x-axis limits
        ax.set_xlim(petal_length.min() - 0.5, petal_length.max() + 0.5)

    # Adjust layout with more compact spacing
    plt.tight_layout(pad=1.2, w_pad=2)
    return fig


if __name__ == "__main__":
    from rdp import RDP

    plotter = RDP()

    svg_content = plotter.create_themed_plot(
        save_name="iris_probabilities", plot_func=plot_probabilities
    )
