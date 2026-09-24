import numpy as np
from sklearn.datasets import load_iris

from rdf import figure


@figure("iris_histogram")
def plot_iris_histogram(ax, color_map):
    iris = load_iris()
    petal_length = iris.data[:, 2]
    y = iris.target

    iqr = np.percentile(petal_length, 75) - np.percentile(petal_length, 25)
    bin_width = 2 * iqr / (len(petal_length) ** (1 / 3))
    n_bins = int(np.ceil((petal_length.max() - petal_length.min()) / bin_width))

    for i, species in enumerate(iris.target_names):
        ax.hist(
            petal_length[y == i],
            bins=n_bins,
            color=color_map[f"c{i + 1}"],
            alpha=0.7,
            label=species.capitalize(),
            edgecolor="none",
        )

    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")

    ax.set_xlim(0, petal_length.max() + 0.5)
    ax.set_ylim(0, max(np.bincount(y)) * 1.1)
