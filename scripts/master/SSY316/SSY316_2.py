import numpy as np

from rdf import figure


@figure("ml_learning_example", height=3.4)
def plot_ml_learning_example(ax, color_map):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 10)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, x.size)
    t = np.linspace(0, 1, 400)
    ax.plot(t, np.sin(2 * np.pi * t), color=color_map["black"], ls="--", label=r"$\sin(2\pi x)$")
    for m, c in ((1, "c2"), (3, "c1"), (9, "c8")):
        ax.plot(t, np.polyval(np.polyfit(x, y, m), t), color=color_map[c], label=f"$M = {m}$")
    ax.scatter(x, y, color=color_map["black"], s=16, zorder=5)
    ax.set(xlim=(0, 1), ylim=(-2, 2), xlabel="$x$", ylabel="$y$")
    ax.legend(loc="lower left", ncol=2)


@figure("map_learning_example", height=2.8)
def plot_map_learning_example(fig, color_map):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 200)
    y = -0.3 + 0.5 * x + rng.normal(0, 0.2, x.size)
    log_prior = lambda W0, W1: -(W0**2 + W1**2)
    log_lik = lambda W0, W1: -0.5 * ((y[:, None, None] - W0 - W1 * x[:, None, None]) ** 2).sum(0) / 0.04
    panels = (
        ("Prior", "c8", (-1, 1), (-1, 1), log_prior),
        ("Likelihood", "c1", (-0.45, -0.15), (0.25, 0.75), log_lik),
        ("Posterior", "c2", (-0.45, -0.15), (0.25, 0.75), lambda a, b: log_prior(a, b) + log_lik(a, b)),
    )
    for ax, (title, c, xl, yl, f) in zip(fig.subplots(1, 3), panels):
        W0, W1 = np.meshgrid(np.linspace(*xl, 150), np.linspace(*yl, 150))
        p = np.exp(f(W0, W1) - f(W0, W1).max())
        levels = np.linspace(0.05, 1, 6)
        ax.contourf(W0, W1, p, levels=levels, colors=[color_map[c]], alpha=0.2)
        ax.contour(W0, W1, p, levels=levels, colors=[color_map[c]], linewidths=0.8)
        ax.plot(-0.3, 0.5, "+", color=color_map["black"], ms=8)
        ax.set(xlim=xl, ylim=yl, xlabel="$w_0$", ylabel="$w_1$")
        ax.set_title(title, fontsize=11)
