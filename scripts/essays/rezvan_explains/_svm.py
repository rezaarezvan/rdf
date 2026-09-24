import numpy as np

SLOPE, INTERCEPT = -1.0, 1.0
MARGIN = 2.0
LIMITS = (-7.5, 7.5)
X_RANGE = np.array([-10.0, 10.0])

# Offsets of the two margin lines from the boundary, as x + y = c.
_OFFSET = MARGIN * np.sqrt(1 + SLOPE**2)
C_BOUNDARY = INTERCEPT
C_UPPER = INTERCEPT + _OFFSET
C_LOWER = INTERCEPT - _OFFSET


def blobs(n_points=12):
    """The two class clouds. Seeded to keep the published figures stable."""
    np.random.seed(42)

    x1 = np.concatenate(
        [
            np.random.uniform(-2.5, 0, n_points),
            np.random.uniform(-5, -2.5, n_points // 2),
        ]
    )
    y1 = np.concatenate(
        [
            -3 + np.random.normal(0, 0.8, n_points),
            np.random.uniform(-5, -2, n_points // 2),
        ]
    )
    x2 = np.concatenate(
        [
            np.random.uniform(2.5, 5, n_points),
            np.random.uniform(5, 7.5, n_points // 2),
        ]
    )
    y2 = np.concatenate(
        [
            np.random.uniform(2, 5, n_points),
            np.random.uniform(4, 7, n_points // 2),
        ]
    )
    return (x1, y1), (x2, y2)


def line(c):
    """The line x + y = c, sampled across X_RANGE."""
    return SLOPE * X_RANGE + c


def draw(ax, color_map, points):
    """Boundary, margins, shaded halfspaces and the two clouds."""
    ax.fill_between(X_RANGE, line(C_LOWER), -10, color=color_map["c1"], alpha=0.1)
    ax.fill_between(X_RANGE, line(C_UPPER), 10, color=color_map["c2"], alpha=0.1)

    ax.plot(
        X_RANGE,
        line(C_BOUNDARY),
        color=color_map["black"],
        label="Decision boundary",
        zorder=2,
    )
    ax.plot(
        X_RANGE,
        line(C_UPPER),
        "--",
        color=color_map["c7"],
        linewidth=1.2,
        label="Margin",
        zorder=2,
    )
    ax.plot(
        X_RANGE, line(C_LOWER), "--", color=color_map["c7"], linewidth=1.2, zorder=2
    )

    for i, ((x, y), name) in enumerate(zip(points, ("Class 1", "Class 2")), 1):
        ax.scatter(
            x,
            y,
            c=color_map[f"c{i}"],
            s=50,
            alpha=0.8,
            label=name,
            edgecolor=color_map["white"],
            linewidth=0.5,
            zorder=3,
        )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.set_xlim(*LIMITS)
    ax.set_ylim(*LIMITS)


def legend(ax, ncol=3):
    """The boundary runs corner to corner, so the legend goes under the axes."""
    ax.get_figure().legend(loc="outside lower center", ncol=ncol)
