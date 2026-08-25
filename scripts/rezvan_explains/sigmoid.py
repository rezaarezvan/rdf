import numpy as np

from rdf import figure


@figure("sigmoid")
def plot_sigmoid(ax, color_map):
    x = np.linspace(-10, 10, 200)

    ax.plot(x, 1 / (1 + np.exp(-x)), color=color_map["c1"], label="Sigmoid", zorder=3)

    for y in (0, 1):
        ax.axhline(y=y, color=color_map["c7"], linestyle="--", linewidth=1, zorder=1)
    ax.axvline(x=0, color=color_map["c7"], linestyle="--", linewidth=1, zorder=1)

    ax.set_xlabel("$f(x)$")
    ax.set_ylabel(r"$\sigma(f(x))$")
    ax.legend(loc="upper left")

    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.1, 1.1)
