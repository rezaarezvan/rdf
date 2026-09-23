import csv
from pathlib import Path

import numpy as np

from rdf import figure

X = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
ANSCOMBE = {
    "I": (X, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    "II": (X, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    "III": (X, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    "IV": ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
}
PENGUINS = Path(__file__).with_name("penguins_size.csv")


def _scatter(ax, color_map, key, labels=True):
    x, y = ANSCOMBE[key]
    ax.scatter(x, y, color=color_map["c8"], s=18)
    ax.set(xlim=(3, 20), ylim=(2, 14))
    if labels:
        ax.set(xlabel="$x$", ylabel="$y$")


def _single(key):
    @figure(f"anscombe_{key.lower()}", height=3.2)
    def plot(ax, color_map):
        _scatter(ax, color_map, key)
    return plot


for _key in ANSCOMBE:
    _single(_key)


@figure("anscombe", height=5.2)
def plot_anscombe(fig, color_map):
    axs = fig.subplots(2, 2, sharex=True, sharey=True)
    for ax, key in zip(axs.ravel(), ANSCOMBE):
        _scatter(ax, color_map, key, labels=False)
        ax.set_title(key)
    for ax in axs[1]:
        ax.set_xlabel("$x$")
    for ax in axs[:, 0]:
        ax.set_ylabel("$y$")


@figure("palmer", height=4.2)
def plot_palmer(ax, color_map):
    rows = [r for r in csv.DictReader(PENGUINS.open()) if r["culmen_length_mm"] not in ("", "NA")]
    for species, c in (("Adelie", "c1"), ("Chinstrap", "c2"), ("Gentoo", "c8")):
        pts = np.array([(float(r["culmen_length_mm"]), float(r["culmen_depth_mm"])) for r in rows if r["species"] == species])
        ax.scatter(pts[:, 0], pts[:, 1], color=color_map[c], s=12, label=species)
    ax.set(xlabel="Culmen length (mm)", ylabel="Culmen depth (mm)")
    ax.legend(loc="lower left")
