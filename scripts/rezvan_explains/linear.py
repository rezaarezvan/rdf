import numpy as np

from rdf import figure


@figure("linear_halfspace", height=6.1)
def plot_halfspace(ax, color_map):
    lo, hi = -5.2, 5.2

    x = np.array([lo, hi])
    y = -0.5 * x + 1

    ax.fill_between(
        x,
        y,
        hi,
        color=color_map["c2"],
        alpha=0.15,
        label="Positive halfspace",
        zorder=1,
    )
    ax.fill_between(
        x,
        lo,
        y,
        color=color_map["c1"],
        alpha=0.15,
        label="Negative halfspace",
        zorder=1,
    )
    ax.plot(x, y, color=color_map["black"], label="Decision boundary", zorder=3)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
