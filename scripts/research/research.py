import numpy as np
from scipy.integrate import solve_ivp

from rdf import figure

RNG = np.random.default_rng(42)
LATENT = RNG.standard_normal((160, 2)) * 0.8
PICKED = RNG.choice(len(LATENT), 5, replace=False)
DIR = np.array([1.0, 0.5]) / np.linalg.norm([1.0, 0.5])
EPS = 0.9


def _drift(t, s):
    x, y = s
    return [-x - 0.5 * y, -y + 0.3 * x]


def _flow(x0):
    return solve_ivp(_drift, (0, 1), x0, t_eval=np.linspace(0, 1, 60)).y


@figure("causality_delta", height=2.3)
def plot_causality_delta(fig, color_map):
    ax1, ax2, ax3 = fig.subplots(1, 3)
    base, pert = color_map["c8"], color_map["c1"]

    ax1.scatter(*LATENT.T, s=6, color=color_map["c7"], alpha=0.35, lw=0)
    x0 = LATENT[PICKED]
    for p in x0:
        ax1.annotate("", p + EPS * DIR, p, arrowprops=dict(arrowstyle="-|>", color=pert, lw=0.8, shrinkA=3, shrinkB=3, mutation_scale=6))
    ax1.scatter(*x0.T, s=10, color=base, zorder=3)
    ax1.scatter(*(x0 + EPS * DIR).T, s=10, color=pert, zorder=3)
    ax1.set_title(r"$\mathbf{x}_0 + \varepsilon \mathbf{z}$", fontsize=11)

    g = np.linspace(-2, 2, 10)
    X, Y = np.meshgrid(g, g)
    U, V = _drift(0, (X, Y))
    ax2.quiver(X, Y, U, V, color=color_map["c7"], alpha=0.6, scale=22, width=0.006)
    for p in x0[:4]:
        for start, c in ((p, base), (p + EPS * DIR, pert)):
            tr = _flow(start)
            ax2.plot(*tr, color=c, lw=1.4)
            ax2.scatter(*tr[:, -1], s=22, facecolors="none", edgecolors=c, lw=1.2, zorder=3)
    ax2.set_title(r"$\mathbf{f}_\theta$", fontsize=11)

    for ax in (ax1, ax2):
        ax.set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4), xticks=[], yticks=[], aspect="equal")
        for s in ax.spines.values():
            s.set_visible(False)

    nodes = {r"$U$": (0.5, 0.86), "Shape": (0.2, 0.5), "Texture": (0.8, 0.5), "Color": (0.5, 0.14)}
    edges = ((r"$U$", "Shape", 1.6), (r"$U$", "Texture", 1.6), (r"$U$", "Color", 0.7), ("Shape", "Color", 0.7), ("Texture", "Color", 0.7))
    for a, b, w in edges:
        ax3.annotate("", nodes[b], nodes[a], arrowprops=dict(arrowstyle="-|>", color=color_map["black"], lw=w, shrinkA=16, shrinkB=16, mutation_scale=8))
    for name, (x, y) in nodes.items():
        ax3.text(x, y, name, ha="center", va="center", fontsize=10,
                 bbox=dict(boxstyle="circle,pad=0.35" if name == r"$U$" else "round,pad=0.4", fc=color_map["white"], ec=color_map["black"], lw=0.8))
    ax3.set_title(r"$\mathbf{J}_{\mathbf{C}_\theta \circ \mathbf{f}_\theta}(\mathbf{x}_0)\, \mathbf{z}$", fontsize=11)
    ax3.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    ax3.axis("off")
