import numpy as np

from rdf import figure


def _plane(ax, xlim, ylim):
    ax.set(xlim=xlim, ylim=ylim, xlabel="$x_1$", ylabel="$x_2$", aspect="equal")


def _bare(ax):
    ax.set(aspect="equal", xticks=[], yticks=[])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def _point(ax, color_map, p, label=None, c="black", offset=(6, 4), marker="o"):
    ax.plot(*p, marker, color=color_map[c], ms=5, zorder=5)
    if label:
        ax.annotate(label, p, xytext=offset, textcoords="offset points", color=color_map[c])


def _arrow(ax, color_map, p, v, c, label=None, offset=(4, -12)):
    ax.annotate("", xy=(p[0] + v[0], p[1] + v[1]), xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=color_map[c], lw=1.4, mutation_scale=12))
    if label:
        ax.annotate(label, (p[0] + v[0], p[1] + v[1]), xytext=offset, textcoords="offset points", color=color_map[c])


def _crescent(n=200):
    t = np.linspace(0.35 * np.pi, 1.65 * np.pi, n)
    outer = np.c_[np.cos(t), np.sin(t)]
    inner = np.c_[0.35 + 0.55 * np.cos(t[::-1]), 0.55 * np.sin(t[::-1])]
    return np.vstack([outer, inner])


def _ellipse(c, a, b, n=200, rot=0.0):
    t = np.linspace(0, 2 * np.pi, n)
    x, y = a * np.cos(t), b * np.sin(t)
    r = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    return (r @ np.vstack([x, y])).T + c


def _hull(points):
    pts = sorted(map(tuple, points))
    def build(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h
    lower, upper = build(pts), build(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _segment(ax, color_map, a, b, c="black", ls="-"):
    ax.plot(*np.c_[a, b], ls=ls, color=color_map[c], lw=1.4, zorder=4)
    for p in (a, b):
        ax.plot(*p, "o", color=color_map[c], ms=4, zorder=5)


@figure("convex_nonconvex_sets", height=2.8)
def plot_convex_nonconvex_sets(fig, color_map):
    ax1, ax2 = fig.subplots(1, 2)
    e = _ellipse((0, 0), 1.3, 0.8, rot=0.3)
    ax1.fill(*e.T, color=color_map["c8"], alpha=0.3, lw=0)
    ax1.plot(*e.T, color=color_map["c8"])
    _segment(ax1, color_map, (-0.8, -0.35), (0.6, 0.35))
    c = _crescent()
    ax2.fill(*c.T, color=color_map["c1"], alpha=0.3, lw=0)
    ax2.plot(*np.vstack([c, c[:1]]).T, color=color_map["c1"])
    _segment(ax2, color_map, (0.45, 0.85), (0.45, -0.85), ls="--")
    ax1.set_xlabel("Convex")
    ax2.set_xlabel("Non-convex")
    for ax in (ax1, ax2):
        _bare(ax)
        ax.xaxis.label.set_visible(True)


@figure("sphere_and_circle", height=3.0)
def plot_sphere_and_circle(fig, color_map):
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20))
    ax1.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color=color_map["c8"],
                     alpha=0.3, shade=False, linewidth=0.3, edgecolor=color_map["c8"])
    ax1.set(xticks=[], yticks=[], zticks=[])
    ax1.set_box_aspect(None, zoom=0.85)
    for axis in (ax1.xaxis, ax1.yaxis, ax1.zaxis):
        axis.pane.set_facecolor("none")
    t = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(t), np.sin(t), color=color_map["c1"], lw=1.6)
    a, b = np.array([np.cos(2.4), np.sin(2.4)]), np.array([np.cos(-0.5), np.sin(-0.5)])
    _segment(ax2, color_map, a, b, ls="--")
    _point(ax2, color_map, (a + b) / 2, r"$\frac{1}{2}(\mathbf{x}_1 + \mathbf{x}_2)$", offset=(-4, -20))
    _plane(ax2, (-1.4, 1.4), (-1.4, 1.4))


@figure("three_convex_operations", height=2.6)
def plot_three_convex_operations(fig, color_map):
    axs = fig.subplots(1, 3)
    sets = (
        [((-0.45, 0), 0.8), ((0.45, 0), 0.8)],
        [((-0.9, 0), 0.55), ((0.9, 0), 0.55)],
        [((0, 0), 1.0), ((0.25, 0.1), 0.45)],
    )
    chords = (((-0.9, 0.55), (0.9, 0.55)), ((-0.9, 0), (0.9, 0)), ((-0.6, -0.5), (0.7, 0.4)))
    for ax, discs, (a, b), convex in zip(axs, sets, chords, (False, False, True)):
        for (c, r), col in zip(discs, ("c8", "c1")):
            e = _ellipse(c, r, r)
            ax.fill(*e.T, color=color_map[col], alpha=0.3, lw=0)
            ax.plot(*e.T, color=color_map[col])
        _segment(ax, color_map, np.array(a), np.array(b), ls="-" if convex else "--")
        _bare(ax)
        ax.set(xlim=(-1.6, 1.6), ylim=(-1.2, 1.2))
        ax.set_xlabel("Convex" if convex else "Not convex")
        ax.xaxis.label.set_visible(True)


@figure("convex_hull_construction", height=3.0)
def plot_convex_hull_construction(fig, color_map):
    axs = fig.subplots(1, 3)
    rng = np.random.default_rng(4)
    pts = rng.uniform(-1, 1, (9, 2)) * [1.2, 0.9]
    h = _hull(pts)
    hc = np.vstack([h, h[:1]])
    for ax in axs:
        ax.fill(*h.T, color=color_map["c8"], alpha=0.3, lw=0)
        ax.plot(*hc.T, color=color_map["c8"], lw=1.4)
        _bare(ax)
        ax.set(xlim=(-1.7, 1.7), ylim=(-1.4, 1.4))
    for ax, label in zip(axs, ("(1)", "(2)", "(3)")):
        ax.set_xlabel(label)
        ax.xaxis.label.set_visible(True)
    for e in (_ellipse((0, 0), 1.55, 1.2), _ellipse((0.1, -0.05), 1.45, 1.3, rot=0.6), _ellipse((-0.05, 0.05), 1.65, 1.05, rot=-0.3)):
        axs[1].plot(*e.T, ls="--", color=color_map["c7"], lw=1)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            axs[2].plot(*np.c_[pts[i], pts[j]], color=color_map["c1"], lw=0.5, alpha=0.6)
    for ax in axs:
        ax.plot(*pts.T, "o", color=color_map["black"], ms=3.5, zorder=5)


@figure("extreme_points_comparison", height=2.8)
def plot_extreme_points_comparison(fig, color_map):
    ax1, ax2 = fig.subplots(1, 2)
    v = np.array([(0, 0), (2, 0), (1, 1.7)])
    for ax in (ax1, ax2):
        ax.fill(*v.T, color=color_map["c8"], alpha=0.3, lw=0)
        ax.plot(*np.vstack([v, v[:1]]).T, color=color_map["c8"], lw=1.4)
        for i, (p, off) in enumerate(zip(v, ((-14, -12), (4, -12), (-4, 6)))):
            _point(ax, color_map, p, rf"$\mathbf{{v}}_{i + 1}$", offset=off)
        _bare(ax)
        ax.set(xlim=(-0.4, 2.4), ylim=(-0.4, 2.0))
    v4 = np.array([1.0, 0.55])
    for p in v:
        ax1.plot(*np.c_[p, v4], ls=":", color=color_map["c7"], lw=1)
    _point(ax1, color_map, v4, r"$\mathbf{v}_4$", c="c1")
    ax1.set_xlabel(r"$\mathrm{conv}\{\mathbf{v}_1, \ldots, \mathbf{v}_4\}$")
    ax2.set_xlabel(r"$\mathrm{conv}\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3\}$")
    for ax in (ax1, ax2):
        ax.xaxis.label.set_visible(True)


@figure("halfspace", height=3.2)
def plot_halfspace(ax, color_map):
    x = np.linspace(-1, 4, 2)
    ax.fill_between(x, -1, 3 - x, color=color_map["c8"], alpha=0.25, lw=0)
    ax.plot(x, 3 - x, color=color_map["c8"], lw=1.6)
    _arrow(ax, color_map, (1.5, 1.5), (0.7, 0.7), "c1", r"$\mathbf{a}$", offset=(4, 2))
    ax.text(0.2, 0.6, r"$\mathbf{a}^T \mathbf{x} \leq b$", color=color_map["c8"])
    ax.text(3.2, 0.3, r"$\mathbf{a}^T \mathbf{x} = b$", color=color_map["c8"])
    _plane(ax, (-1, 4), (-1, 4))


@figure("polyhedron_halfspaces", height=3.4)
def plot_polyhedron_halfspaces(ax, color_map):
    x = np.linspace(-1, 6, 2)
    lines = ((3 - x / 2, "c8", r"$x_1 + 2x_2 = 6$", (4.4, -0.4)), (2 * x - 2, "c1", r"$-2x_1 + x_2 = -2$", (2.9, 3.6)), (np.ones_like(x), "c2", r"$x_2 = 1$", (4.8, 1.25)))
    for y, c, label, pos in lines:
        ax.plot(x, y, color=color_map[c], lw=1.4)
        ax.text(*pos, label, color=color_map[c])
    p = np.array([(1.5, 1), (4, 1), (2, 2)])
    ax.fill(*p.T, color=color_map["c8"], alpha=0.3, lw=0)
    ax.text(2.35, 1.3, "$P$")
    _plane(ax, (-1, 6), (-1, 4))


@figure("cone_comparison", height=2.8)
def plot_cone_comparison(fig, color_map):
    ax1, ax2 = fig.subplots(1, 2)
    r = 2.2
    def sector(ax, a0, a1, c):
        t = np.linspace(a0, a1, 50)
        ax.fill(np.r_[0, r * np.cos(t)], np.r_[0, r * np.sin(t)], color=color_map[c], alpha=0.3, lw=0)
        for a in (a0, a1):
            ax.plot([0, r * np.cos(a)], [0, r * np.sin(a)], color=color_map[c], lw=1.4)
    sector(ax1, np.radians(10), np.radians(55), "c8")
    d = np.array([np.cos(np.radians(32)), np.sin(np.radians(32))])
    ax1.plot(*np.c_[[0, 0], 2.1 * d], ls=":", color=color_map["c7"], lw=1)
    for s, label in ((0.55, r"$0.5\,\mathbf{x}$"), (1.1, r"$\mathbf{x}$"), (1.75, r"$1.6\,\mathbf{x}$")):
        _point(ax1, color_map, s * d, label, offset=(6, -12))
    _point(ax1, color_map, (0, 0), "$\\mathbf{0}$", offset=(-12, -2))
    ax1.set_xlabel("Convex cone")
    sector(ax2, np.radians(15), np.radians(40), "c1")
    sector(ax2, np.radians(-40), np.radians(-15), "c1")
    a = 1.5 * np.array([np.cos(np.radians(27)), np.sin(np.radians(27))])
    b = a * [1, -1]
    _segment(ax2, color_map, a, b, ls="--")
    ax2.annotate(r"$\mathbf{x}_1$", a, xytext=(6, 0), textcoords="offset points")
    ax2.annotate(r"$\mathbf{x}_2$", b, xytext=(6, -8), textcoords="offset points")
    _point(ax2, color_map, (0, 0), "$\\mathbf{0}$", offset=(-12, -2))
    ax2.set_xlabel("Non-convex cone")
    for ax in (ax1, ax2):
        _bare(ax)
        ax.set(xlim=(-0.4, 2.4), ylim=(-1.6, 2.0))
        ax.xaxis.label.set_visible(True)


@figure("convex_function_definition", height=3.4)
def plot_convex_function_definition(ax, color_map):
    f = lambda x: 0.3 * (x - 1) ** 2 + 0.5
    x = np.linspace(-2, 4, 300)
    x1, x2 = 0.0, 2.5
    ax.plot(x, f(x), color=color_map["c8"], lw=1.6, label="$f(x)$")
    ax.plot([x1, x2], [f(x1), f(x2)], "--", color=color_map["c1"], lw=1.4, label=r"$\lambda f(x_1) + (1 - \lambda) f(x_2)$")
    for p, label in ((x1, "$x_1$"), (x2, "$x_2$")):
        ax.plot([p, p], [0, f(p)], ":", color=color_map["c7"], lw=1)
        ax.plot(p, f(p), "o", color=color_map["c1"], ms=5)
        ax.annotate(label, (p, 0), xytext=(-4, -14), textcoords="offset points")
    ax.set(xlim=(-2, 4), ylim=(0, 3.4), xticks=[], xlabel="$x$", ylabel="$f(x)$")
    ax.legend(loc="upper center")


@figure("piecewise_function_analysis", height=3.6)
def plot_piecewise_function_analysis(ax, color_map):
    c = color_map["c8"]
    pieces = (
        (np.linspace(-1, 1, 200), lambda x: -0.5 * x**2 + 2 * x + 1),
        (np.linspace(1, 3, 200), lambda x: x**2 - 4 * x + 5),
        (np.linspace(3, 4.5, 200), lambda x: np.abs(x - 3) + 1),
        (np.linspace(4.5, 6, 200), lambda x: -0.3 * (x - 5) ** 2 + 2.5),
    )
    for x, f in pieces:
        ax.plot(x, f(x), color=c, lw=1.6)
    ax.plot(1, 2.5, "o", mfc="none", mec=c, ms=6, mew=1.4)
    ax.plot(1, 2, "o", color=c, ms=6)
    ax.plot(3, 1, "D", color=color_map["c2"], ms=6)
    ax.plot([2, 5], [1, 2.5], "v", color=color_map["c1"], ms=7)
    ax.plot([-1, 6], [-1.5, 2.45], "s", color=color_map["c5"], ms=6)
    for x in (-1, 6):
        ax.axvline(x, color=color_map["c7"], ls="--", lw=1)
    notes = (
        ((1, 2.5), (0.2, 3.6), "Discontinuity"),
        ((2, 1), (2.0, 0.1), "Stationary point"),
        ((3, 1), (3.9, 0.3), "Non-differentiable"),
        ((5, 2.5), (5.0, 3.4), "Stationary point"),
        ((-1, -1.5), (0.4, -1.5), "Boundary"),
    )
    for xy, txt, s in notes:
        ax.annotate(s, xy, xytext=txt, ha="center", va="center", fontsize=10,
                    arrowprops=dict(arrowstyle="-", color=color_map["c7"], lw=0.8, shrinkA=4, shrinkB=4))
    ax.set(xlim=(-1.5, 6.5), ylim=(-2, 4), xlabel="$x$", ylabel="$f(x)$")


def _disc_regions(ax, color_map):
    t = np.linspace(0, 2 * np.pi, 300)
    for cx, c, label, pos in ((1, "c8", r"$g_1(\mathbf{x}) \leq 0$", (1.2, 1.15)), (-1, "c1", r"$g_2(\mathbf{x}) \leq 0$", (-2.4, 1.15))):
        ax.fill(cx + np.cos(t), np.sin(t), color=color_map[c], alpha=0.2, lw=0)
        ax.plot(cx + np.cos(t), np.sin(t), color=color_map[c], lw=1.4)
        ax.text(*pos, label, color=color_map[c])
    _point(ax, color_map, (0, 0), r"$\bar{\mathbf{x}}$", offset=(4, -14))


@figure("example_question_regions", height=3.0)
def plot_example_question_regions(ax, color_map):
    _disc_regions(ax, color_map)
    _plane(ax, (-2.6, 2.6), (-1.4, 1.6))


def _half_disc(ax, color_map):
    t = np.linspace(0, np.pi, 200)
    ax.fill(np.cos(t), np.sin(t), color=color_map["c8"], alpha=0.3, lw=0)
    ax.plot(np.cos(t), np.sin(t), color=color_map["c8"], lw=1.4)
    ax.plot([-1, 1], [0, 0], color=color_map["c8"], lw=1.4)
    pts = {"x_1": (0, 0), "x_2": (-1, 0), "x_3": (1, 0)}
    for name, p in pts.items():
        _point(ax, color_map, p, rf"$\mathbf{{{name[0]}}}_{name[2]}$", offset=(-16, 6))
    return pts


@figure("example_question_regions2", height=3.0)
def plot_example_question_regions2(ax, color_map):
    _half_disc(ax, color_map)
    _plane(ax, (-1.8, 1.8), (-0.6, 1.4))


@figure("example_question_regions3", height=3.4)
def plot_example_question_regions3(ax, color_map):
    pts = _half_disc(ax, color_map)
    for name, p in pts.items():
        _arrow(ax, color_map, p, (0.45, -0.45), "c1", r"$-\nabla f$" if name == "x_3" else None, offset=(2, -10))
        _arrow(ax, color_map, p, (0, -0.55), "c2", r"$\nabla g_2$" if name == "x_3" else None, offset=(-30, -8))
    _arrow(ax, color_map, pts["x_2"], (-0.55, 0), "c5", r"$\nabla g_1$", offset=(-20, 4))
    _arrow(ax, color_map, pts["x_3"], (0.55, 0), "c5", r"$\nabla g_1$", offset=(-4, 4))
    _plane(ax, (-1.8, 1.8), (-0.9, 1.3))


@figure("example_question_regions4", height=3.2)
def plot_example_question_regions4(ax, color_map):
    x = np.linspace(-0.2, 1.3, 300)
    ax.fill_between(np.linspace(0, 1, 200), np.linspace(0, 1, 200) ** 5, np.linspace(0, 1, 200) ** 3, color=color_map["c8"], alpha=0.3, lw=0)
    ax.plot(x, x**3, color=color_map["c8"], lw=1.4)
    ax.plot(x, x**5, color=color_map["c1"], lw=1.4)
    ax.axhline(0, color=color_map["c2"], lw=1.4)
    ax.text(1.18, 0.85, r"$x_2 = x_1^3$", color=color_map["c8"])
    ax.text(0.62, 1.35, r"$x_2 = x_1^5$", color=color_map["c1"])
    ax.text(0.95, -0.2, r"$x_2 = 0$", color=color_map["c2"])
    _point(ax, color_map, (0, 0), r"$\bar{\mathbf{x}}$", offset=(-14, 6))
    ax.set(xlim=(-0.2, 1.4), ylim=(-0.4, 2), xlabel="$x_1$", ylabel="$x_2$")


@figure("example_question_regions5", height=3.4)
def plot_example_question_regions5(ax, color_map):
    g = np.linspace(-1.6, 1.6, 300)
    X, Y = np.meshgrid(g, g)
    ax.contour(X, Y, X**2 + Y**2 / 100, levels=[0.05, 0.2, 0.5, 1.0, 1.6, 2.4], colors=color_map["c7"], linewidths=0.6)
    t = np.linspace(0, 2 * np.pi, 300)
    ax.contourf(X, Y, X**2 + Y**2, levels=[1, 10], colors=[color_map["c8"]], alpha=0.3)
    ax.plot(np.cos(t), np.sin(t), color=color_map["c8"], lw=1.4)
    ax.plot([0, 0], [1, -1], "o", color=color_map["c1"], ms=7, label="Global optimality conditions")
    ax.plot([1, -1], [0, 0], "s", color=color_map["c2"], ms=6, label="KKT points only")
    _plane(ax, (-1.6, 1.6), (-1.6, 1.6))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)


def _lp_region(ax, color_map):
    p = np.array([(1, 0), (4, 3), (4, 4), (0, 4), (0, 1)])
    ax.fill(*p.T, color=color_map["c8"], alpha=0.3, lw=0)
    x = np.linspace(-0.5, 4, 2)
    for y, label, pos in ((x - 1, r"$x_1 - x_2 \leq 1$", (2.4, 0.8)), (1 - x, r"$x_1 + x_2 \geq 1$", (0.1, 1.6))):
        ax.plot(x, y, color=color_map["c8"], lw=1.4)
        ax.text(*pos, label, color=color_map["c8"])
    ax.axhline(0, color=color_map["black"], lw=0.8)
    ax.axvline(0, color=color_map["black"], lw=0.8)
    ax.text(1.4, 2.6, "$P$")
    _plane(ax, (-0.5, 3.5), (-0.5, 3.5))


@figure("example_polyhedron_regions", height=3.4)
def plot_example_polyhedron_regions(ax, color_map):
    _lp_region(ax, color_map)


@figure("example_polyhedron_regions_and_negative_gradient", height=3.4)
def plot_example_polyhedron_regions_and_negative_gradient(ax, color_map):
    _lp_region(ax, color_map)
    ax.axhline(1.5, color=color_map["c2"], ls="--", lw=1)
    _arrow(ax, color_map, (1.6, 2.6), (0, -1.0), "c1", r"$-\mathbf{c}$", offset=(4, 12))
    _point(ax, color_map, (1, 0), r"$\mathbf{x}^\star$", c="c1", offset=(8, -12))


@figure("example_polyhedron_regions2", height=3.4)
def plot_example_polyhedron_regions2(ax, color_map):
    p = np.array([(0, 0), (1, 0), (0, 1)])
    ax.fill(*np.array([(0, 0), (1, 0), (2, 1), (2, 2.6), (0.8, 2.6), (0, 1)]).T, color=color_map["c8"], alpha=0.3, lw=0)
    x = np.linspace(-0.5, 2, 2)
    ax.plot(x, 1 + 2 * x, color=color_map["c8"], lw=1.4)
    ax.plot(x, x - 1, color=color_map["c1"], lw=1.4)
    ax.text(0.9, 2.35, r"$-2x_1 + x_2 \leq 1$", color=color_map["c8"])
    ax.text(1.35, -0.35, r"$x_1 - x_2 \leq 1$", color=color_map["c1"])
    ax.axhline(0, color=color_map["black"], lw=0.8)
    ax.axvline(0, color=color_map["black"], lw=0.8)
    for q, off in zip(p, ((-30, -14), (6, 6), (-34, 2))):
        _point(ax, color_map, q, f"$({q[0]:g}, {q[1]:g})$", c="c1", offset=off)
    _plane(ax, (-0.6, 2), (-0.6, 2.6))
