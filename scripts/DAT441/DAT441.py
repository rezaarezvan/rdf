import numpy as np

from rdf import figure

ARMS = [(1, 0.3), (3, 0.5), (5.5, 0.8), (8, 0.4)]


def _concentration(ax, color_map, offset):
    x = np.linspace(-2, 10, 1000)
    for i, (mu, sd) in enumerate(ARMS):
        c = color_map[("c1", "c2", "c8", "black")[i]]
        ax.plot(x, np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), color=c, label=rf"$\mu_{i + 1}$")
        ax.axvline(mu + offset, color=c, ls="--", lw=1)
    ax.set(xlim=(-2, 10), ylim=(0, 1.6), xlabel="$x$", ylabel="Density")
    ax.legend(loc="upper right")


@figure("four_arms_concentration", height=3.2)
def plot_concentration(ax, color_map):
    _concentration(ax, color_map, 0)


@figure("four_arms_concentration_mixed", height=3.2)
def plot_concentration_mixed(ax, color_map):
    _concentration(ax, color_map, 0.5)
