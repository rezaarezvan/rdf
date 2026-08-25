import numpy as np

from rdf import figure

from _svm import C_LOWER, SLOPE, blobs, draw, legend


@figure("margin_principle", height=6.6)
def plot_margin_principle(ax, color_map):
    points = blobs()
    draw(ax, color_map, points)

    px = -2.5
    py = SLOPE * px + C_LOWER
    ax.scatter(
        px,
        py,
        c=color_map["c7"],
        s=100,
        edgecolor=color_map["black"],
        linewidth=1.2,
        zorder=4,
    )
    ax.annotate(
        "Margin point",
        xy=(px, py),
        xytext=(px - 1, py - 1),
        ha="right",
        fontsize=10,
        color=color_map["c7"],
        arrowprops=dict(arrowstyle="->", color=color_map["c7"]),
    )

    legend(ax, ncol=4)
