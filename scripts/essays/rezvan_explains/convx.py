import numpy as np

from rdf import figure

_WOBBLE = 0.95
_SIGMA2 = 0.5
_SPREAD = 0.9


def _radius(theta, wobble):
    """Unit circle, dented inward at theta = pi by `wobble`."""
    return 1.0 - wobble * np.exp(-((np.abs(theta - np.pi)) ** 2) / _SIGMA2)


def _endpoints():
    theta = np.array([np.pi - _SPREAD, np.pi + _SPREAD])
    r = _radius(theta, _WOBBLE)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _panel(ax, color_map, wobble, title):
    theta = np.linspace(0, 2 * np.pi, 600)
    r = _radius(theta, wobble)
    x, y = r * np.cos(theta), r * np.sin(theta)
    ax.fill(x, y, color=color_map["c2"], alpha=0.15)
    ax.plot(x, y, color=color_map["c2"], zorder=2)

    a, b = _endpoints()
    t = np.linspace(0, 1, 400)[:, None]
    seg = (1 - t) * a + t * b
    edge = _radius(np.arctan2(seg[:, 1], seg[:, 0]) % (2 * np.pi), wobble)
    outside = np.hypot(*seg.T) > edge

    ax.plot(*seg[~outside].T, color=color_map["black"], linewidth=1.4, zorder=4)
    if outside.any():
        ax.plot(*seg[outside].T, color=color_map["c1"], linewidth=2.2, zorder=4)
        ax.annotate(
            "Outside the set",
            xy=seg[outside].mean(axis=0),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=10,
            color=color_map["c1"],
        )

    for point, label, offset in (
        (a, "$x^{(1)}$", (-11, 3)),
        (b, "$x^{(2)}$", (-11, -11)),
    ):
        ax.scatter(*point, s=26, color=color_map["black"], zorder=5)
        ax.annotate(label, point, textcoords="offset points", xytext=offset, ha="right")

    ax.set_title(title)
    ax.set_xlim(-1.7, 1.25)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


@figure("convex_sets", height=2.6)
def plot_convex_sets(fig, color_map):
    ax1, ax2 = fig.subplots(1, 2)
    _panel(ax1, color_map, 0.0, "Convex")
    _panel(ax2, color_map, _WOBBLE, "Non-convex")
