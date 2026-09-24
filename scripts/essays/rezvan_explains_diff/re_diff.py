import numpy as np

from rdf import figure

A = np.array([[-0.3, -1.0], [1.0, -0.3]])
STARTS = [(2, 0), (-2, 0), (0, 2), (0, -2)]
COLORS = ("c8", "c1", "c2", "c5")


def _paths(rng, x0, sigma, n=500, dt=0.01, drift=lambda x: -x):
    x = np.empty((n, np.size(x0)))
    x[0] = x0
    for i in range(1, n):
        x[i] = x[i - 1] + drift(x[i - 1]) * dt + sigma * np.sqrt(dt) * rng.standard_normal(np.size(x0))
    return x


def _field(ax, color_map):
    g = np.linspace(-2.5, 2.5, 15)
    X, Y = np.meshgrid(g, g)
    U, V = A @ np.stack([X.ravel(), Y.ravel()])
    ax.quiver(X, Y, U.reshape(X.shape), V.reshape(X.shape), color=color_map["c7"], alpha=0.5, angles="xy", width=0.004)
    ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), xlabel="$x_1$", ylabel="$x_2$", aspect="equal")


def _panels(fig, color_map, sigma):
    ax1, ax2 = fig.subplots(1, 2)
    rng = np.random.default_rng(3)
    t = np.linspace(0, 5, 500)
    for x0, c in zip((0.5, 1.5, 3.0, 4.5), COLORS):
        x = _paths(rng, [x0], sigma)[:, 0]
        ax1.plot(t, x, color=color_map[c], lw=1)
        if sigma:
            ax1.plot(t, x0 * np.exp(-t), color=color_map[c], ls="--", lw=0.8)
    ax1.set(xlabel="$t$", ylabel="$x(t)$", xlim=(0, 5))
    _field(ax2, color_map)
    for x0, c in zip(STARTS, COLORS):
        x = _paths(rng, x0, sigma, n=800, drift=lambda x: A @ x)
        ax2.plot(x[:, 0], x[:, 1], color=color_map[c], lw=1)
        ax2.plot(*x0, "o", color=color_map[c], ms=4)


@figure("deterministic_ODE", height=3.4)
def plot_deterministic_ode(fig, color_map):
    _panels(fig, color_map, 0)


@figure("stochastic_SDE", height=3.4)
def plot_stochastic_sde(fig, color_map):
    _panels(fig, color_map, 0.4)
