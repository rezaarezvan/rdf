import numpy as np

from rdf import figure


@figure("equipotential_curves", height=4.6)
def plot_equipotential_curves(ax, color_map):
    x, y = np.meshgrid(np.linspace(0, 10, 400), np.linspace(0, 10, 400))
    v = np.exp(-(((x - 4.2) / 2.2) ** 2 + ((y - 5.2) / 2.6) ** 2)) + 0.05 * np.sin(1.3 * x) * np.cos(1.1 * y)
    levels = [0.15, 0.3, 0.5, 0.7, 0.9]
    cs = ax.contour(x, y, v, levels=levels, colors=color_map["black"], linewidths=1.4)
    for lvl, lab, deg, xy in zip(levels[:3], ("$V = C_1$", "$V = C_2$", "$V = C_3$"), (0, -35, 35), ((9.4, 5.2), (9.4, 2.4), (9.4, 8.2))):
        seg = max(cs.allsegs[levels.index(lvl)], key=len)
        d = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        p = seg[np.argmax((seg - (4.2, 5.2)) @ d)]
        ax.annotate(lab, xy=p, xytext=xy, arrowprops=dict(arrowstyle="-|>", color=color_map["black"], lw=0.8), va="center")
    ax.annotate("$O$", (0, 0), xytext=(-12, -12), textcoords="offset points", annotation_clip=False)
    ax.set(xlim=(0, 11), ylim=(0, 10), xticks=[], yticks=[], xlabel="$x$", ylabel="$y$", aspect="equal")
    ax.grid(False)
