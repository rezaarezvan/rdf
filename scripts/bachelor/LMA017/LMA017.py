import numpy as np

from rdf import figure


def _surface(ax, x, y, z, color, alpha=0.3):
    ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=False, linewidth=0.4,
                    edgecolor=color, rstride=4, cstride=4)


def _labels(ax):
    ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
    ax.set_box_aspect(None, zoom=0.85)


@figure("plane", is_3d=True, height=4.2)
def plot_plane(ax, color_map):
    x, y = np.meshgrid(np.linspace(-2, 4, 25), np.linspace(-2, 6, 25))
    _surface(ax, x, y, 8 - 4 * x - 2 * y, color_map["c8"])
    ax.plot([0], [4], [0], "o", color=color_map["c1"], ms=4)
    ax.text(0, 4, 3, "$(0, 4, 0)$", color=color_map["c1"])
    _labels(ax)
    ax.view_init(25, -60)


@figure("paraboloid_levels", is_3d=True, height=4.2)
def plot_paraboloid_levels(ax, color_map):
    r, t = np.meshgrid(np.linspace(0, 5, 41), np.linspace(0, 2 * np.pi, 61))
    x, y = r * np.cos(t), r * np.sin(t)
    _surface(ax, x, y, x**2 + y**2, color_map["c8"], alpha=0.15)
    s = np.linspace(0, 2 * np.pi, 200)
    for k in (5, 10, 15, 20):
        cx, cy = np.sqrt(k) * np.cos(s), np.sqrt(k) * np.sin(s)
        ax.plot(cx, cy, np.full_like(s, k), color=color_map["c1"], lw=1.2)
        ax.plot(cx, cy, np.zeros_like(s), color=color_map["c1"], lw=1)
    ax.set(zlim=(0, 25))
    _labels(ax)
    ax.view_init(25, -55)


@figure("level_curves_xy", height=4.2)
def plot_level_curves_xy(ax, color_map):
    x = np.linspace(-5, 5, 400)
    x = x[np.abs(x) > 0.05]
    for k, c in ((1, "black"), (2, "c8"), (5, "c2"), (10, "c1")):
        for sign, ls in ((1, "-"), (-1, "--")):
            for half in (x[x > 0], x[x < 0]):
                y = sign * k / half
                m = np.abs(y) <= 5
                ax.plot(half[m], y[m], color=color_map[c], ls=ls,
                        label=f"$k = {sign * k}$" if half[0] > 0 else None)
    ax.axhline(0, color=color_map["black"], lw=1)
    ax.axvline(0, color=color_map["black"], lw=1)
    ax.set(xlim=(-5, 5), ylim=(-5, 5), xlabel="$x$", ylabel="$y$", aspect="equal")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))


@figure("rotation_field", height=4.2)
def plot_rotation_field(ax, color_map):
    x, y = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    ax.quiver(x, y, -y, x, color=color_map["c8"], angles="xy", scale_units="xy", scale=4,
              width=0.006)
    ax.set(xlim=(-4, 4), ylim=(-4, 4), xlabel="$x$", ylabel="$y$", aspect="equal")


@figure("unit_disc", is_3d=True, height=4.2)
def plot_unit_disc(ax, color_map):
    u, v = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 2 * np.pi, 61))
    _surface(ax, u * np.cos(v), u * np.sin(v), np.ones_like(u), color_map["c8"])
    ax.set(zlim=(0, 2))
    _labels(ax)
    ax.view_init(25, -60)


@figure("cylinder", is_3d=True, height=4.2)
def plot_cylinder(ax, color_map):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 61), np.linspace(0, 1, 21))
    r = np.sqrt(2)
    _surface(ax, r * np.cos(u), r * np.sin(u), v, color_map["c8"])
    ax.set(zlim=(0, 1.5))
    _labels(ax)
    ax.set_box_aspect((1, 1, 0.6), zoom=0.85)
    ax.view_init(25, -60)
