import numpy as np
from matplotlib.patches import Circle

from rdf import figure


def _bare(ax):
    ax.set_aspect("equal")
    ax.axis("off")


def _plus(ax, pts, color):
    for x, y in pts:
        ax.text(x, y, "+", ha="center", va="center", fontsize=9, color=color)


@figure("conductor_fields", height=2.6)
def plot_conductor_fields(fig, color_map):
    ink, field = color_map["black"], color_map["c1"]
    a1, a2, a3 = fig.subplots(1, 3)
    for ax in (a1, a2, a3):
        _bare(ax)

    a1.add_patch(Circle((0, 0), 1, fill=False, color=field, lw=2.5))
    t = np.linspace(0, 2 * np.pi, 28, endpoint=False)
    _plus(a1, zip(1.18 * np.cos(t), 1.18 * np.sin(t)), ink)
    a1.annotate("", xy=(np.cos(-0.9), np.sin(-0.9)), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ink, lw=0.8))
    a1.text(0.45, -0.25, "$a$", color=ink)
    a1.text(0, 0.35, r"$\vec{\mathbf{E}} = \vec{\mathbf{0}}$", ha="center", color=ink)
    a1.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4))

    x, y = np.meshgrid(np.linspace(-2.6, 2.6, 600), np.linspace(-2.4, 2.4, 560))
    r2 = x**2 + y**2
    psi = np.where(r2 >= 1, y * (1 + 1 / np.maximum(r2, 1e-9)), np.nan)
    levels = np.r_[-2.6, np.linspace(-1.75, 1.75, 11), 2.6]
    a2.contour(x, y, psi, levels=levels, colors=[field], linewidths=1.1, linestyles="solid")
    ys = np.linspace(-3, 3, 6001)
    for c in levels:
        for xx in (-2.1, 1.9):
            yy = ys[np.argmin(np.abs(ys * (1 + 1 / (xx**2 + ys**2)) - c))]
            a2.annotate("", xy=(xx + 0.15, yy), xytext=(xx, yy), arrowprops=dict(arrowstyle="-|>", color=field, lw=0))
    a2.add_patch(Circle((0, 0), 1, facecolor=color_map["c7"], alpha=0.35, edgecolor=ink, ls="--", lw=0.8))
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    for tt in th:
        a2.text(0.82 * np.cos(tt), 0.82 * np.sin(tt), "+" if np.cos(tt) > 0 else "$-$", ha="center", va="center", fontsize=8, color=ink)
    a2.text(0, 0, r"$\vec{\mathbf{E}} = 0$", ha="center", va="center", color=ink)
    a2.text(0, -2.55, r"$\vec{\mathbf{E}} = \vec{\mathbf{E}}_0 + \vec{\mathbf{E}}{}^{\prime}$", ha="center", va="top", color=ink)
    a2.set(xlim=(-2.6, 2.6), ylim=(-3.0, 2.4))

    s = np.linspace(0, 2 * np.pi, 400)
    rb = 1 + 0.28 * np.cos(s) + 0.12 * np.cos(2 * s)
    bx, by = 1.1 * rb * np.cos(s), 0.8 * rb * np.sin(s)
    a3.fill(bx, by, color=field, alpha=0.3, lw=0)
    for i in range(0, 400, 16):
        nx, ny = np.gradient(by)[i], -np.gradient(bx)[i]
        n = np.hypot(nx, ny)
        a3.annotate("", xy=(bx[i] + 0.75 * nx / n, by[i] + 0.75 * ny / n), xytext=(bx[i], by[i]), arrowprops=dict(arrowstyle="-|>", color=field, lw=0.9))
        a3.text(bx[i] - 0.12 * nx / n, by[i] - 0.12 * ny / n, "+", ha="center", va="center", fontsize=7, color=ink)
    a3.text(-1.2, 1.3, r"$\vec{\mathbf{E}}$", color=ink)
    a3.set(xlim=(-2.1, 2.3), ylim=(-1.9, 1.9))
