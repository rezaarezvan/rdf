import numpy as np
from sklearn.datasets import load_iris


def plot_iris_histogram(ax=None, color_map=None):
    """
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

    # Create histograms for each species
    for i, species in enumerate(iris.target_names):
        mask = y == i
        ax.hist(
            petal_length[mask],
            bins=n_bins,
            color=color_map[f"c{i + 1}"],
            alpha=0.7,
            label=species,
            edgecolor="none",
        )

    # Customize plot appearance
    ax.set_title("Histogram of Petal Length", fontsize=12, pad=15)
    ax.set_xlabel("Petal Length (cm)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)

    # Clean legend
    ax.legend(frameon=True, framealpha=0.9, loc="upper right", fontsize=9)

    # Set consistent limits
    ax.set_xlim(0, max(petal_length) + 0.5)
    max_freq = max([len(petal_length[y == i]) for i in range(3)])
    ax.set_ylim(0, max_freq * 1.1)


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()
    svg_content = plotter.create_themed_plot(
        save_name="iris_histogram", plot_func=plot_iris_histogram
    )
