import numpy as np

from rdf import figure


@figure("residuals", height=3.6)
def plot_residuals(ax, color_map):
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(0, 10, 20))
    y = 1.5 + 0.6 * x + rng.normal(0, 0.9, x.size)
    b1, b0 = np.polyfit(x, y, 1)
    fit = b0 + b1 * x
    ax.vlines(x, fit, y, color=color_map["c1"], lw=1)
    ax.plot([-0.5, 10.5], [b0 - 0.5 * b1, b0 + 10.5 * b1], color=color_map["c8"], label=r"Fitted line $\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x$")
    ax.scatter(x, y, color=color_map["black"], s=14, zorder=3, label="Observations")
    ax.plot([], [], color=color_map["c1"], lw=1, label="Residuals")
    ax.set(xlim=(-0.5, 10.5), xlabel="$x$", ylabel="$y$")
    ax.legend(loc="upper left")
