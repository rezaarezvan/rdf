import numpy as np
from scipy.stats import gaussian_kde
from sklearn.datasets import load_iris

from rdf import figure


@figure("iris_probabilities", height=3.1)
def plot_probabilities(fig, color_map):
    ax1, ax2 = fig.subplots(1, 2)

    iris = load_iris()
    petal_length = iris.data[:, 2]
    y = iris.target
    priors = np.bincount(y) / len(y)

    lo, hi = petal_length.min() - 0.5, petal_length.max() + 0.5
    x = np.linspace(lo, hi, 200)

    kdes = [gaussian_kde(petal_length[y == i], bw_method="silverman") for i in range(3)]
    likelihoods = np.column_stack([kde(x) for kde in kdes])
    posteriors = likelihoods * priors
    posteriors /= posteriors.sum(axis=1, keepdims=True)

    for ax, curves in ((ax1, likelihoods), (ax2, posteriors)):
        for i, species in enumerate(iris.target_names):
            color = color_map[f"c{i + 1}"]
            ax.plot(x, curves[:, i], color=color, label=species.capitalize(), zorder=3)
            ax.fill_between(x, curves[:, i], color=color, alpha=0.2, zorder=2)

        ax.set_xlabel("Petal length (cm)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

    ax1.set_ylabel("$p(x \\mid y)$")
    ax2.set_ylabel("$p(y \\mid x)$")
    ax2.legend(loc="upper right")
