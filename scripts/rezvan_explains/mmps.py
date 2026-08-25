import numpy as np

from rdf import figure

from _svm import C_LOWER, C_UPPER, blobs, draw, legend

ROOT2 = np.sqrt(2)


def _violations(x, y, c, sign):
    """Points on the wrong side of their margin, and the foot of each slack.

    sign is +1 when the class belongs above the line x + y = c, -1 below.
    """
    slack = sign * (c - (x + y)) / ROOT2
    inside = slack > 0
    step = sign * slack[inside] / ROOT2
    return (
        x[inside],
        y[inside],
        x[inside] + step,
        y[inside] + step,
        slack[inside],
    )


@figure("soft_margin_principle", height=6.6)
def plot_soft_margin(ax, color_map):
    (x1, y1), (x2, y2) = blobs()

    # A handful of samples that stray across the margin.
    x1 = np.concatenate([x1, [-1, 1, -2]])
    y1 = np.concatenate([y1, [-1, 0, 1]])
    x2 = np.concatenate([x2, [0, -1.5, 1.5]])
    y2 = np.concatenate([y2, [2, 0, -1]])

    draw(ax, color_map, ((x1, y1), (x2, y2)))

    labelled = False
    widest = (0.0, None)
    for (x, y), c, sign, slot in (
        ((x1, y1), C_LOWER, -1, "c1"),
        ((x2, y2), C_UPPER, +1, "c2"),
    ):
        vx, vy, fx, fy, slack = _violations(x, y, c, sign)
        for a, b, u, v, s in zip(vx, vy, fx, fy, slack):
            ax.plot(
                [a, u],
                [b, v],
                "--",
                color=color_map["c7"],
                linewidth=0.9,
                zorder=4,
            )
            if s > widest[0]:
                widest = (s, (a, b, u, v))
        ax.scatter(
            vx,
            vy,
            s=70,
            facecolors="none",
            edgecolors=color_map[slot],
            linewidth=1.4,
            zorder=5,
            label=None if labelled else "Margin violation",
        )
        labelled = True

    if widest[1] is not None:
        a, b, u, v = widest[1]
        ax.annotate(
            r"$\xi_i$",
            xy=((a + u) / 2, (b + v) / 2),
            xytext=(10, 8),
            textcoords="offset points",
            fontsize=13,
        )

    legend(ax, ncol=3)
