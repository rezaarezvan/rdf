"""DAT410_4.py — figures for Part 4, Diagnostic Systems."""

import numpy as np

from rdf import figure


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


@figure("binary_test_ideal")
def plot_binary_test_ideal(ax, color_map):
    """
    An ideal binary test: the negative and positive populations are fully
    separated by the threshold T.
    """
    x = np.linspace(0, 10, 400)
    neg = gaussian(x, 3.0, 0.8)
    pos = 0.7 * gaussian(x, 7.5, 1.0)
    T = 5.25

    ax.fill_between(x, neg, color=color_map["c8"], alpha=0.6)
    ax.plot(x, neg, color=color_map["c8"], linewidth=1.5)
    ax.fill_between(x, pos, color=color_map["c1"], alpha=0.6)
    ax.plot(x, pos, color=color_map["c1"], linewidth=1.5)

    ax.axvline(T, color=color_map["black"], linewidth=1.5)
    ax.text(T + 0.15, 0.55, r"Threshold, $T$", ha="left", va="bottom", fontsize=11)
    ax.text(3.0, 0.22, r"$-$", ha="center", fontsize=14)
    ax.text(7.5, 0.12, r"$+$", ha="center", fontsize=14)

    ax.set_xlabel(r"Test statistic, $X$")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 0.62)
    ax.grid(False)


@figure("binary_test_overlap")
def plot_binary_test_overlap(ax, color_map):
    """
    A realistic binary test: the populations overlap, so any threshold yields
    false negatives (positives below T) and false positives (negatives above T).
    """
    x = np.linspace(0, 10, 400)
    neg = gaussian(x, 4.0, 1.1)
    pos = 0.65 * gaussian(x, 6.3, 1.2)
    T = 5.35

    ax.fill_between(x, neg, color=color_map["c8"], alpha=0.6)
    ax.plot(x, neg, color=color_map["c8"], linewidth=1.5)
    ax.fill_between(x, pos, color=color_map["c1"], alpha=0.6)
    ax.plot(x, pos, color=color_map["c1"], linewidth=1.5)

    # Error regions: FN = positives left of T, FP = negatives right of T.
    fn = (x <= T) & (pos > 0)
    fp = (x >= T) & (neg > 0)
    ax.fill_between(x[fn], pos[fn], color=color_map["c4"], alpha=0.9)
    ax.fill_between(x[fp], neg[fp], color=color_map["c4"], alpha=0.9)

    ax.axvline(T, color=color_map["black"], linewidth=1.5)
    ax.text(T + 0.15, 0.4, r"Threshold, $T$", ha="left", va="bottom", fontsize=11)
    ax.text(3.5, 0.18, r"$-$", ha="center", fontsize=14)
    ax.text(7.2, 0.1, r"$+$", ha="center", fontsize=14)
    ax.text(4.9, 0.025, "FN", ha="center", fontsize=9)
    ax.text(5.9, 0.025, "FP", ha="center", fontsize=9)

    ax.set_xlabel(r"Test statistic, $X$")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 0.46)
    ax.grid(False)
