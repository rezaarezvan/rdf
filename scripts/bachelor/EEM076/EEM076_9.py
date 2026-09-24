import numpy as np
from matplotlib.patches import Ellipse

from _loop_field import coil_flux, contour_arrows, field_arrows
from rdf import figure


def _coil(ax, z0, color, n=5, a=1.0, lead="left"):
    for z in z0 + np.linspace(-0.24, 0.24, n):
        ax.add_patch(Ellipse((0, z), 2 * a, 0.35, fill=False, color=color, lw=2))
    top, bot = z0 + 0.24, z0 - 0.24
    for y0, y1 in ((top, top + 0.1), (bot, bot - 0.35)):
        ax.plot(np.linspace(-a, -3.2, 50), y0 + (y1 - y0) * np.sin(np.linspace(0, np.pi / 2, 50)), color=color, lw=2)


def _lines(ax, turns, z0, seeds, field, rho_max, z_rng, arrows):
    rho, z = np.meshgrid(np.linspace(-rho_max, rho_max, 600), np.linspace(*z_rng, 700))
    psi = coil_flux(rho, z, turns)
    levels = sorted(set(np.round(coil_flux(np.array(seeds), np.full(len(seeds), z0), turns), 6)))
    cs = ax.contour(rho, z, psi, levels=levels, colors=[field], linewidths=1.1, linestyles="solid")
    ax.axvline(0, color=field, lw=1.1)
    contour_arrows(ax, cs, turns, field)
    field_arrows(ax, arrows, turns, field)


@figure("mutual_inductance", height=4.6)
def plot_mutual_inductance(ax, color_map):
    field, coil, ink = color_map["c8"], color_map["c1"], color_map["black"]
    turns = np.linspace(-0.24, 0.24, 5)
    seeds = [0.12, 0.3, 0.6, 1.5, 2.0]
    arrows = [(0.0, 3.4), (0.0, -2.0)]
    _lines(ax, turns, 0, seeds, field, 3.6, (-2.6, 4.2), arrows)
    _coil(ax, 0, coil)
    _coil(ax, 2.6, coil)
    ax.text(-3.35, -0.1, "Coil 1", color=ink, ha="right", va="center")
    ax.text(-3.35, 2.5, "Coil 2", color=ink, ha="right", va="center")
    ax.text(3.0, 0, "$N_1$", color=ink, va="center")
    ax.text(1.15, 2.6, r"$N_2 \Phi_{21}$", color=ink, va="center")
    ax.annotate("", xy=(-2.1, -0.95), xytext=(-3.1, -0.95), arrowprops=dict(arrowstyle="-|>", color=color_map["c1"], lw=1.6))
    ax.text(-2.9, -1.35, "$I_1$", color=ink)
    ax.text(0.12, -2.5, r"$\vec{\mathbf{B}}_1$", color=ink)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set(xlim=(-4.6, 3.6), ylim=(-2.6, 4.2))


@figure("self_inductance", height=3.4)
def plot_self_inductance(ax, color_map):
    field, coil, ink = color_map["c8"], color_map["c1"], color_map["black"]
    turns = np.linspace(-0.24, 0.24, 5)
    seeds = [0.3, 0.6, 1.5, 2.0]
    arrows = [(0.0, 1.8), (0.0, -1.6)]
    _lines(ax, turns, 0, seeds, field, 3.6, (-2.0, 2.2), arrows)
    _coil(ax, 0, coil)
    ax.annotate("", xy=(-2.1, -0.95), xytext=(-3.1, -0.95), arrowprops=dict(arrowstyle="-|>", color=color_map["c1"], lw=1.6))
    ax.text(-3.35, -1.1, "$I$", color=ink)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set(xlim=(-3.6, 3.6), ylim=(-2.0, 2.2))
