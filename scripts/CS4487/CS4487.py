import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch

from rdf import figure

Z = np.linspace(-4, 4, 400)
ACTIVATIONS = [
    ("Sigmoid", r"$\sigma(x) = 1/(1 + e^{-x})$", 1 / (1 + np.exp(-Z))),
    ("tanh", r"$\tanh(x)$", np.tanh(Z)),
    ("ReLU", r"$\max(0, x)$", np.maximum(0, Z)),
    ("Leaky ReLU", r"$\max(0.1x, x)$", np.maximum(0.1 * Z, Z)),
    ("Maxout", r"$\max(x, -0.5x + 1)$", np.maximum(Z, -0.5 * Z + 1)),
    ("ELU", r"$x$ if $x \geq 0$, else $e^x - 1$", np.where(Z >= 0, Z, np.exp(Z) - 1)),
]


@figure("activation_functions", height=4.8)
def plot_activation_functions(fig, color_map):
    axs = fig.subplots(2, 3)
    for ax, (name, formula, y) in zip(axs.ravel(), ACTIVATIONS):
        ax.plot(Z, y, color=color_map["c8"])
        ax.axhline(0, color=color_map["black"], lw=0.6)
        ax.axvline(0, color=color_map["black"], lw=0.6)
        ax.set_title(f"{name}\n{formula}", fontsize=11)
        ax.set(xlim=(-4, 4), ylim=(-1.5, 4), xticks=[-4, 0, 4], yticks=[-1, 0, 2, 4])


def _panel(ax, color_map, center, a, b, angle, e1, e2, axes=None):
    cx, cy = center
    ax.add_patch(Ellipse(center, 2 * a, 2 * b, angle=np.degrees(angle), facecolor=color_map["c3"], alpha=0.35, edgecolor="none"))
    for (vx, vy), c in ((e1, "c1"), (e2, "c2")):
        ax.add_patch(FancyArrowPatch((cx, cy), (cx + vx, cy + vy), arrowstyle="-|>", mutation_scale=12, color=color_map[c], lw=1.6))
    if axes:
        for (sx, sy), label, off in axes:
            ax.plot([cx, cx + sx], [cy, cy + sy], color=color_map["c8"], lw=1.6)
            ax.text(cx + sx + off[0], cy + sy + off[1], label, ha="center", va="center")


def _step(ax, color_map, start, end, label, offset):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color=color_map["black"], lw=1))
    ax.text((start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1], label, ha="center", va="center")


@figure("svd_operations", height=5.2)
def plot_svd_operations(ax, color_map):
    m = np.array([[1.2, 0.9], [0.3, 1.0]])
    u, s, vt = np.linalg.svd(m)
    flip = np.array([np.sign(u[0, 0]), np.sign(u[0, 0]) * np.sign(np.linalg.det(u))])
    u, vt = u * flip, vt * flip[:, None]
    e1, e2 = np.eye(2)
    theta_u = np.arctan2(u[1, 0], u[0, 0])
    tl, tr, bl, br = (0, 0), (5, 0), (0, -4.2), (5, -4.2)
    _panel(ax, color_map, tl, 1, 1, 0, e1, e2)
    _panel(ax, color_map, bl, 1, 1, 0, vt @ e1, vt @ e2)
    sv = np.diag(s)
    _panel(ax, color_map, br, s[0], s[1], 0, sv @ vt @ e1, sv @ vt @ e2,
           axes=[((s[0], 0), r"$\sigma_1$", (0.1, -0.3)), ((0, s[1]), r"$\sigma_2$", (-0.35, 0.1))])
    _panel(ax, color_map, tr, s[0], s[1], theta_u, m @ e1, m @ e2,
           axes=[(s[0] * u[:, 0], r"$\sigma_1$", (0.25, -0.2)), (s[1] * u[:, 1], r"$\sigma_2$", (-0.3, 0.15))])
    _step(ax, color_map, (1.4, 0), (3.2, 0), "$M$", (0, 0.3))
    _step(ax, color_map, (0, -1.3), (0, -2.9), r"$V^*$", (0.35, 0))
    _step(ax, color_map, (1.4, -4.2), (3.2, -4.2), r"$\Sigma$", (0, 0.3))
    _step(ax, color_map, (5, -3.3), (5, -1.6), "$U$", (0.3, 0))
    ax.text(2.5, -6.1, r"$M = U \Sigma V^*$", ha="center", fontsize=14)
    ax.set(xlim=(-1.5, 7), ylim=(-6.5, 1.8), aspect="equal")
    ax.axis("off")
