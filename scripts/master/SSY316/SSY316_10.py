import numpy as np
from scipy.stats import multivariate_normal as mvn

from rdf import figure


@figure("deterministic_approximate_inference", height=2.6)
def plot_deterministic_approximate_inference(fig, color_map):
    g = np.linspace(-4, 4, 200)
    pos = np.dstack(np.meshgrid(g, g))
    cov = [[0.5, 0], [0, 1]]
    p = mvn([-1.5, 0], cov).pdf(pos) + mvn([1.5, 0], cov).pdf(pos)
    qs = (mvn([0, 0], [[3, 0], [0, 3]]), mvn([-1.5, 0], cov), mvn([1.5, 0], cov))
    titles = (r"$\mathrm{KL}(p \,\|\, q)$", r"$\mathrm{KL}(q \,\|\, p)$", r"$\mathrm{KL}(q \,\|\, p)$")
    for ax, q, title in zip(fig.subplots(1, 3), qs, titles):
        ax.contour(g, g, p, levels=6, colors=[color_map["c4"]], linewidths=1)
        ax.contour(g, g, q.pdf(pos), levels=6, colors=[color_map["c2"]], linewidths=1)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), xticks=[], yticks=[], aspect="equal")
        ax.set_title(title, fontsize=11)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
