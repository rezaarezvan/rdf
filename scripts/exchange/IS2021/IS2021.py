import numpy as np

from rdf import figure

CRITERIA = ["Feasibility", "Reasonability", "Value", "Integrability", "Sustainability"]
SCORES = np.array([2, 3, 5, 3, 2])


@figure("radar_chart", height=4.2)
def plot_radar_chart(ax, color_map):
    theta = np.pi / 2 - np.arange(len(CRITERIA)) * 2 * np.pi / len(CRITERIA)
    u = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    ring = np.vstack([u, u[:1]])
    for r in range(1, 6):
        ax.plot(*(r * ring).T, color=color_map["c7"], lw=0.6)
    for x, y in u:
        ax.plot([0, 5 * x], [0, 5 * y], color=color_map["c7"], lw=0.6)
    tick = np.array([np.cos(np.pi / 2 - np.pi / 5), np.sin(np.pi / 2 - np.pi / 5)])
    for r in range(1, 6):
        ax.text(*(r * tick + [0.05, 0.25]), str(r), ha="left", va="center", fontsize=10.5, color=color_map["c7"])
    pts = SCORES[:, None] * u
    ax.fill(*pts.T, color=color_map["c8"], alpha=0.15)
    ax.plot(*np.vstack([pts, pts[:1]]).T, color=color_map["c8"], lw=2)
    for (x, y), name in zip(u, CRITERIA):
        ax.text(5.4 * x, 5.4 * y + 0.15 * np.sign(y), name, ha="center" if abs(x) < 0.1 else ("left" if x > 0 else "right"), va="bottom" if y > 0 else "top")
    ax.set(xlim=(-7.5, 7.5), ylim=(-5.2, 6), aspect="equal")
    ax.axis("off")
