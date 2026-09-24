import numpy as np

from _loop_field import coil_flux, field_arrows
from rdf import figure


@figure("solenoid_field", height=3.4)
def plot_solenoid_field(ax, color_map):
    field, wire, ink = color_map["c8"], color_map["c1"], color_map["black"]
    turns = np.linspace(-2, 2, 5)
    z, rho = np.meshgrid(np.linspace(-4.5, 4.5, 700), np.linspace(-3, 3, 500))
    psi = coil_flux(rho, z, turns)
    seeds = [0.3, 0.7, 1.6, 2.4]
    levels = sorted(set(np.round(coil_flux(np.array(seeds), np.zeros(len(seeds)), turns), 6)))
    ax.contour(z, rho, psi, levels=levels, colors=[field], linewidths=1.0, linestyles="solid")
    ax.axhline(0, color=field, lw=1.0)
    field_arrows(ax, [(s * sgn, 0) for s in seeds for sgn in (1, -1)] + [(0, 0)], turns, field, flip=True)
    for t in turns:
        for y, mark in ((1, "."), (-1, "x")):
            ax.plot(t, y, "o", ms=11, mfc="none", mec=ink, mew=0.8)
            ax.plot(t, y, mark, ms=9 if mark == "." else 6, color=wire, mew=1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set(xlim=(-4.5, 4.5), ylim=(-3, 3))
