import numpy as np
from sklearn.datasets import load_iris

from rdf import figure


@figure("iris_linear_separation")
def plot_linear_separation(ax, color_map):
    iris = load_iris()
    mask = iris.target < 2
    petal = iris.data[mask][:, 2:4]
    y = iris.target[mask]

    for i, species in enumerate(iris.target_names[:2]):
        ax.scatter(
            *petal[y == i].T,
            c=color_map[f"c{i + 1}"],
            s=70,
            alpha=0.8,
            label=species.capitalize(),
            edgecolor=color_map["white"],
            linewidth=0.5,
            zorder=3,
        )

    mu = np.array([petal[y == i].mean(axis=0) for i in (0, 1)])
    w = mu[1] - mu[0]
    b = -w @ mu.mean(axis=0)

    lo, hi = petal.min(axis=0), petal.max(axis=0)
    pad = (hi - lo) * 0.1
    lo, hi = lo - pad, hi + pad

    x_line = np.array([lo[0], hi[0]])
    ax.plot(
        x_line,
        -(w[0] * x_line + b) / w[1],
        "--",
        color=color_map["black"],
        label="Decision boundary",
        zorder=2,
    )

    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Petal width (cm)")
    ax.legend(loc="upper left")

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
