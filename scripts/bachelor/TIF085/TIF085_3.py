import numpy as np

from rdf import figure


def _path(ax, v, p, color, labels):
    ax.plot(v, p, color=color)
    i = len(v) // 2
    ax.annotate("", xy=(v[i + 1], p[i + 1]), xytext=(v[i - 1], p[i - 1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=14))
    for x, y, s in zip((v[0], v[-1]), (p[0], p[-1]), labels):
        ax.plot(x, y, "o", color=color, ms=4)
        ax.annotate(s, (x, y), xytext=(5, 5), textcoords="offset points", color=color)


def _axes(ax):
    ax.set(xlabel="$V$", ylabel="$P$", xticks=[], yticks=[], xlim=(0.6, 3.4), ylim=(0, 4))
    ax.grid(False)


@figure("pv_isobaric", height=3.2)
def plot_pv_isobaric(ax, color_map):
    v = np.linspace(1, 3, 101)
    _path(ax, v, np.full_like(v, 3.0), color_map["c1"], "12")
    _path(ax, v[::-1], np.full_like(v, 1.2), color_map["c8"], "34")
    _axes(ax)


@figure("pv_isochoric", height=3.2)
def plot_pv_isochoric(ax, color_map):
    p = np.linspace(1, 3.4, 101)
    _path(ax, np.full_like(p, 1.3), p, color_map["c1"], "12")
    _path(ax, np.full_like(p, 2.7), p[::-1], color_map["c8"], "34")
    _axes(ax)
