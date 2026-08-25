import numpy as np
from scipy.stats import norm
from sklearn.datasets import load_iris

from rdf import figure


@figure("iris_gaussian")
def plot_gaussian_conditionals(ax, color_map):
    iris = load_iris()
    petal_length = iris.data[:, 2]
    y = iris.target

    # Freedman-Diaconis
    iqr = np.percentile(petal_length, 75) - np.percentile(petal_length, 25)
    bin_width = 2 * iqr / (len(petal_length) ** (1 / 3))
    n_bins = int(np.ceil((petal_length.max() - petal_length.min()) / bin_width))

    lo, hi = petal_length.min() - 0.5, petal_length.max() + 0.5
    x_smooth = np.linspace(lo, hi, 200)
    peak = 0.0

    for i, species in enumerate(iris.target_names):
        data = petal_length[y == i]
        name = species.capitalize()
        ax.hist(
            data,
            bins=n_bins,
            density=True,
            color=color_map[f"c{i + 1}"],
            alpha=0.3,
            label=f"{name} (data)",
            edgecolor="none",
        )

        mu, sigma = np.mean(data), np.std(data)
        ax.plot(
            x_smooth,
            norm.pdf(x_smooth, mu, sigma),
            "--",
            color=color_map[f"c{i + 1}"],
            label=f"{name} (Gaussian)",
            alpha=0.9,
        )
        peak = max(peak, norm.pdf(mu, mu, sigma))

    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", ncol=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(0, peak * 1.1)
