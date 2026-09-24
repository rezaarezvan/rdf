import numpy as np

from rdf import figure


def _panel(ax, color_map, f, slot, chord_x, title):
    x = np.linspace(-2, 2, 200)
    ax.plot(x, f(x), color=color_map[slot], zorder=3)

    cx = np.array(chord_x)
    ax.plot(
        cx,
        f(cx),
        "--",
        color=color_map["black"],
        linewidth=1.2,
        label="Line segment",
        zorder=4,
    )
    ax.scatter(cx, f(cx), s=26, color=color_map["black"], zorder=5)

    ax.set_title(title)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.5, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for name, pos in (("left", "zero"), ("bottom", "zero")):
        ax.spines[name].set_position(pos)

    for marker, transform in (
        (">", ax.get_yaxis_transform()),
        ("^", ax.get_xaxis_transform()),
    ):
        xy = (1, 0) if marker == ">" else (0, 1)
        ax.plot(
            *xy,
            marker=marker,
            color=color_map["black"],
            transform=transform,
            clip_on=False,
        )


@figure("convex_functions", height=3.6)
def plot_convex_functions(fig, color_map):
    ax1, ax2 = fig.subplots(1, 2)

    _panel(ax1, color_map, lambda x: x**2, "c2", (-1.5, 1.2), "Convex")
    _panel(
        ax2,
        color_map,
        lambda x: x**4 - 2 * x**2 + 1,
        "c1",
        (-1.4, 1.4),
        "Non-convex",
    )

    ax1.scatter(0, 0, s=45, color=color_map["c2"], zorder=6, label="Global minimum")
    ax2.scatter(
        [-1, 1], [0, 0], s=45, color=color_map["c1"], zorder=6, label="Local minima"
    )

    for ax in (ax1, ax2):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0))
