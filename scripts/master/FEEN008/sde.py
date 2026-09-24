import numpy as np
from scipy.stats import norm

from rdf import figure


def _bm(rng, n, dt):
    return np.concatenate([[0.0], np.cumsum(rng.normal(0, np.sqrt(dt), n - 1))])


def _below(ax, ncol=3, y=-0.22):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol)


@figure("SDE_only_drift", height=3.2)
def plot_SDE_only_drift(ax, color_map):
    rng = np.random.default_rng(1)
    t = np.linspace(0, 3, 2)
    for x0 in rng.normal(0, np.sqrt(2), 8):
        ax.plot(t, x0 + 2 * t, color=color_map["c8"], alpha=0.6, lw=1.2)
    ax.plot(t, 2 * t, color=color_map["black"], ls="--", lw=1.2, label=r"$\mathbb{E}[x(t)] = 2t$")
    ax.set(xlim=(0, 3), xlabel="$t$", ylabel="$x(t)$")
    ax.legend(loc="upper left")


@figure("SDE_only_diffusion", height=3.2)
def plot_SDE_only_diffusion(ax, color_map):
    rng = np.random.default_rng(2)
    dt, n = 0.005, 601
    t = np.arange(n) * dt
    for _ in range(8):
        ax.plot(t, rng.normal() + 2 * _bm(rng, n, dt), color=color_map["c8"], alpha=0.6, lw=1)
    ax.set(xlim=(0, t[-1]), xlabel="$t$", ylabel="$x(t)$")


@figure("SDE_example", height=3.2)
def plot_SDE_example(ax, color_map):
    rng = np.random.default_rng(3)
    dt, n = 0.005, 401
    t = np.arange(n) * dt
    for k in range(5):
        x = np.empty(n)
        x[0] = rng.normal()
        for i in range(1, n):
            x[i] = x[i - 1] + 0.5 * x[i - 1] * dt + rng.normal(0, np.sqrt(dt))
        ax.plot(t, x, color=color_map["c8"], alpha=1 if k == 0 else 0.35, lw=1.2 if k == 0 else 1)
    ax.set(xlim=(0, t[-1]), xlabel="$t$", ylabel="$x(t)$")


@figure("dynamic_SDE_example", height=3.2)
def plot_dynamic_SDE_example(ax, color_map):
    rng = np.random.default_rng(4)
    dt, n, q = 0.005, 1001, 0.5
    t = np.arange(n) * dt
    v = q * _bm(rng, n, dt)
    x = np.concatenate([[0.0], np.cumsum(v[:-1]) * dt])
    ax.plot(t, v, color=color_map["c8"], lw=1.2, label="$v(t)$")
    ax.plot(t, x, color=color_map["c1"], lw=1.5, ls="--", label="$x(t)$")
    ax.axhline(0, color=color_map["black"], lw=0.6)
    ax.set(xlim=(0, t[-1]), xlabel="$t$", ylabel="$x(t),\\ v(t)$")
    ax.legend(loc="upper left")


@figure("riemann_sum", height=3.4)
def plot_riemann_sum(ax, color_map):
    n = 10
    edges = np.linspace(0, 2, n + 1)
    ax.bar(edges[:-1], edges[:-1] ** 2, width=2 / n, align="edge", color=color_map["c1"], alpha=0.35,
           edgecolor=color_map["c1"], lw=0.8)
    x = np.linspace(0, 2, 200)
    ax.plot(x, x**2, color=color_map["c8"], lw=1.8, label="$f(x) = x^2$")
    ax.set(xlim=(-0.05, 2.05), xlabel="$x$", ylabel="$f(x)$")
    ax.legend(loc="upper left")


def _sine(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(1, 2), rng.uniform(0, 2 * np.pi)


@figure("random_diff", height=3.2)
def plot_random_diff(ax, color_map):
    a, phi = _sine(5)
    t = np.linspace(0, 2 * np.pi, 300)
    x1, dx1 = a * np.sin(1 + phi), a * np.cos(1 + phi)
    ax.plot(t, a * np.sin(t + phi), color=color_map["c8"], lw=1.8, label="$x(t)$")
    ax.plot(t, x1 + dx1 * (t - 1), color=color_map["c1"], ls="--", lw=1.2, label=f"slope $\\dot{{x}}(1) = {dx1:.2f}$")
    ax.plot(1, x1, "o", color=color_map["c1"], ms=5)
    ax.set(xlim=(0, 2 * np.pi), ylim=(-2.4, 2.4), xlabel="$t$", ylabel="$x(t)$")
    ax.legend(loc="upper right")


@figure("random_riemann", height=3.2)
def plot_random_riemann(ax, color_map):
    a, phi = _sine(5)
    edges = np.linspace(np.pi / 2, 3 * np.pi / 2, 9)
    ax.bar(edges[:-1], a * np.sin(edges[:-1] + phi), width=np.diff(edges), align="edge", color=color_map["c1"],
           alpha=0.35, edgecolor=color_map["c1"], lw=0.8)
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(t, a * np.sin(t + phi), color=color_map["c8"], lw=1.8)
    ax.axhline(0, color=color_map["black"], lw=0.6)
    for e in (edges[0], edges[-1]):
        ax.axvline(e, color=color_map["c7"], ls=":", lw=1)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set(xlim=(0, 2 * np.pi), ylim=(-2.4, 2.4), xlabel="$t$", ylabel="$x(t)$")


@figure("brownian_VS_sin", height=4.2)
def plot_brownian_VS_sin(fig, color_map):
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    rng = np.random.default_rng(6)
    a, phi = _sine(6)
    dt = 0.002
    t = np.arange(0, 2 * np.pi, dt)
    ax1.plot(t, a * np.sin(t + phi), color=color_map["c8"], lw=1.5)
    ax2.plot(t, _bm(rng, t.size, dt), color=color_map["c1"], lw=0.8)
    ax1.set(ylabel="$f(t)$")
    ax2.set(xlim=(0, t[-1]), xlabel="$t$", ylabel=r"$\beta(t)$")


@figure("second_order", height=4.4)
def plot_second_order(fig, color_map):
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    rng = np.random.default_rng(7)
    dt, T = 0.0005, 2
    t = np.linspace(0, T, int(T / dt) + 1)
    b = _bm(rng, t.size, dt)
    ax1.plot(t, b, color=color_map["black"], lw=0.7)
    for n, c in ((5, "c1"), (50, "c2"), (1000, "c8")):
        idx = np.linspace(0, t.size - 1, n + 1).astype(int)
        s = np.concatenate([[0.0], np.cumsum(np.diff(b[idx]) ** 2)])
        ax2.step(t[idx], s, where="post", color=color_map[c], lw=1.2, label=f"$n = {n}$")
        if n == 5:
            ax1.plot(t[idx], b[idx], "o", color=color_map[c], ms=4)
    ax2.plot(t, t, color=color_map["black"], ls="--", lw=1, label="$t$")
    ax1.set(ylabel=r"$\beta(t)$")
    ax2.set(xlim=(0, T), xlabel="$t$", ylabel=r"$\sum_{t_i < t} (\Delta \beta_i)^2$")
    ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))


@figure("beta_beta_squared", height=3.2)
def plot_beta_beta_squared(ax, color_map):
    rng = np.random.default_rng(8)
    dt = 0.002
    t = np.arange(0, 2, dt)
    b = _bm(rng, t.size, dt)
    ax.plot(t, b, color=color_map["c8"], lw=1.2, label=r"$\beta(t)$")
    ax.plot(t, b**2, color=color_map["c1"], lw=1.2, label=r"$\beta^2(t)$")
    ax.axhline(0, color=color_map["black"], lw=0.6)
    ax.set(xlim=(0, 2), xlabel="$t$")
    ax.legend(loc="upper left")


@figure("beta_squared_second_order_taylor", height=3.8)
def plot_beta_squared_second_order_taylor(fig, color_map):
    axs = fig.subplots(1, 2, sharey=True)
    rng = np.random.default_rng(42078)
    dt = 0.01
    b = _bm(rng, 200, dt)
    bi = b[100]
    grid = np.linspace(b.min(), b.max(), 200)
    lines = (
        (bi**2 + 2 * bi * (grid - bi), r"$2\beta_i(\beta - \beta_i) + \beta_i^2$"),
        (2 * bi * (grid - bi), r"$2\beta_i(\beta - \beta_i)$"),
    )
    for ax, (y, lab) in zip(axs, lines):
        ax.plot(b, b**2, color=color_map["c8"], lw=1.2, label=r"$\beta^2$")
        ax.plot(grid, y, color=color_map["c1"], ls="--", lw=1.2, label=lab)
        ax.plot(bi, bi**2, "o", color=color_map["c1"], ms=5)
        ax.set(xlabel=r"$\beta(t)$")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3))
    axs[0].set(ylabel=r"$\beta^2(t)$")


def _ou_var(t, lam, sigma):
    return sigma**2 / (2 * lam) * (1 - np.exp(-2 * lam * t))


@figure("ornstein_uhlenbeck", height=3.4)
def plot_ornstein_uhlenbeck(ax, color_map):
    rng = np.random.default_rng(9)
    lam, sigma, dt, T = 0.7, 0.5, 0.01, 5
    t = np.arange(0, T + dt, dt)
    sd = np.sqrt(_ou_var(t, lam, sigma))
    ax.fill_between(t, -sd, sd, color=color_map["c7"], alpha=0.25, lw=0, label=r"$\pm$ 1 std")
    for _ in range(10):
        x = np.zeros(t.size)
        for i in range(1, t.size):
            x[i] = x[i - 1] - lam * x[i - 1] * dt + sigma * rng.normal(0, np.sqrt(dt))
        ax.plot(t, x, color=color_map["c8"], alpha=0.55, lw=0.9)
    ax.plot(t, 0 * t, color=color_map["black"], ls="--", lw=1.2, label="Mean")
    ax.set(xlim=(0, T), xlabel="$t$", ylabel="$x(t)$")
    ax.legend(loc="lower left", ncol=2)


@figure("reverse_ornstein_uhlenbeck", height=3.4)
def plot_reverse_ornstein_uhlenbeck(ax, color_map):
    rng = np.random.default_rng(10)
    lam, sigma, dt, T = 0.7, 0.5, 0.01, 5
    t = np.arange(dt, T + dt / 2, dt)
    var = _ou_var(t, lam, sigma)
    sd = np.sqrt(var)
    ax.fill_between(t, -sd, sd, color=color_map["c7"], alpha=0.25, lw=0, label=r"$\pm$ 1 std")
    for _ in range(10):
        x = np.empty(t.size)
        x[-1] = rng.normal(0, sd[-1])
        for n in range(t.size - 1, 0, -1):
            x[n - 1] = x[n] - (-lam * x[n] + sigma**2 * x[n] / var[n]) * dt - sigma * np.sqrt(dt) * rng.normal()
        ax.plot(t, x, color=color_map["c8"], alpha=0.55, lw=0.9)
    ax.annotate("", xy=(0.4, 1.05), xytext=(1.6, 1.05), xycoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="-|>", color=color_map["black"], lw=0.8))
    ax.text(1.0, 1.08, "reverse time", ha="center", va="bottom", transform=ax.get_xaxis_transform())
    ax.set(xlim=(0, T), xlabel="$t$", ylabel="$x(t)$")
    _below(ax, ncol=1, y=-0.3)


@figure("denoising_diffusion", height=3.4)
def plot_denoising_diffusion(ax, color_map):
    rng = np.random.default_rng(11)
    t = np.linspace(0, 0.999, 1500)
    tau = t**2 / (1 - t) ** 2
    for x0, c in zip((-2, -1, 0, 1, 2), ("c1", "c5", "c2", "black", "c8")):
        w = np.concatenate([[0.0], np.cumsum(rng.normal(0, np.sqrt(np.diff(tau))))])
        ax.plot(t, (1 - t) * (x0 + w), color=color_map[c], lw=1, label=f"$x_0 = {x0}$")
    ax.set(xlim=(0, 1), ylim=(-3.2, 3.2), xlabel="$t$", ylabel="$x(t)$")
    _below(ax, ncol=5, y=-0.3)


@figure("brownian_to_point", height=3.2)
def plot_brownian_to_point(ax, color_map):
    rng = np.random.default_rng(12)
    v, T, dt = 1.0, 2.0, 0.002
    t = np.arange(0, T + dt / 2, dt)
    for k in range(6):
        x = np.zeros(t.size)
        for i in range(1, t.size):
            x[i] = x[i - 1] + (v - x[i - 1]) / (T - t[i - 1]) * dt + rng.normal(0, np.sqrt(dt))
        x[-1] = v
        ax.plot(t, x, color=color_map["c8"], alpha=1 if k == 0 else 0.4, lw=1.2 if k == 0 else 0.9)
    ax.axhline(v, color=color_map["c7"], ls="--", lw=1)
    ax.plot(T, v, "o", color=color_map["c1"], ms=6, zorder=5)
    ax.annotate("$v$", (T, v), xytext=(-12, 6), textcoords="offset points", color=color_map["c1"])
    ax.set(xlim=(0, T + 0.05), xlabel="$t$", ylabel="$x(t)$")


def _marks(ax, color_map, pts):
    for x, s in pts:
        ax.axvline(x, color=color_map["c7"], ls="--", lw=1)
    ax.set_xticks([x for x, _ in pts], [s for _, s in pts])


@figure("brownian_bridge", height=3.2)
def plot_brownian_bridge(ax, color_map):
    rng = np.random.default_rng(13)
    dt = 0.002
    t = np.arange(0, 2 + dt / 2, dt)
    ax.plot(t, _bm(rng, t.size, dt), color=color_map["c8"], lw=1.1)
    _marks(ax, color_map, ((0, "$t_0$"), (1, "$t_1$"), (2, "$t_2$")))
    ax.set(xlim=(-0.05, 2.05), xlabel="$t$", ylabel=r"$\beta(t)$")


def _side_density(ax, color_map, t, y, pdf, width):
    d = pdf / pdf.max() * width
    ax.fill_betweenx(y, t, t + d, color=color_map["c2"], alpha=0.3, lw=0)
    ax.plot(t + d, y, color=color_map["c2"], lw=1.5)


@figure("brownian_transition_density", height=3.4)
def plot_brownian_transition_density(ax, color_map):
    rng = np.random.default_rng(42)
    s, t, dt = 0.6, 1.2, 0.002
    ts = np.arange(0, s + dt / 2, dt)
    b = _bm(rng, ts.size, dt)
    tf = np.arange(s, t + dt / 2, dt)
    for _ in range(10):
        ax.plot(tf, b[-1] + _bm(rng, tf.size, dt), color=color_map["c8"], alpha=0.35, lw=0.8)
    ax.plot(ts, b, color=color_map["c1"], lw=1.5)
    ax.plot(s, b[-1], "o", color=color_map["c1"], ms=5, zorder=5)
    y = np.linspace(b[-1] - 3 * np.sqrt(t - s), b[-1] + 3 * np.sqrt(t - s), 200)
    _side_density(ax, color_map, t, y, norm.pdf(y, b[-1], np.sqrt(t - s)), 0.25)
    ax.text(t + 0.27, b[-1], r"$p(\beta(t) \mid \beta(s))$", va="center", color=color_map["c2"])
    _marks(ax, color_map, ((s, "$s$"), (t, "$t$")))
    ax.set(xlim=(0, t + 0.75), ylabel=r"$\beta$")


@figure("brownian_bridge_density", height=3.4)
def plot_brownian_bridge_density(ax, color_map):
    rng = np.random.default_rng(42)
    s, t, T, a, v, dt = 0.5, 1.5, 2.5, 0.0, 2.0, 0.002
    ts = np.arange(s, T + dt / 2, dt)
    frac = (ts - s) / (T - s)
    for _ in range(12):
        w = _bm(rng, ts.size, dt)
        ax.plot(ts, a + frac * (v - a) + w - frac * w[-1], color=color_map["c8"], alpha=0.35, lw=0.8)
    mean, sd = a + (t - s) / (T - s) * (v - a), np.sqrt((t - s) * (T - t) / (T - s))
    y = np.linspace(mean - 3 * sd, mean + 3 * sd, 200)
    _side_density(ax, color_map, t, y, norm.pdf(y, mean, sd), 0.3)
    ax.plot(s, a, "o", color=color_map["c1"], ms=5, zorder=5)
    ax.plot(T, v, "o", color=color_map["c1"], ms=5, zorder=5)
    ax.annotate(r"$\beta(s)$", (s, a), xytext=(-8, -14), textcoords="offset points", ha="right", color=color_map["c1"])
    ax.annotate(r"$\beta(T) = v$", (T, v), xytext=(6, 0), textcoords="offset points", ha="left", va="center", color=color_map["c1"])
    ax.text(t + 0.33, mean - 1.6 * sd, r"$p(\beta(t) \mid \beta(s), \beta(T) = v)$", color=color_map["c2"])
    _marks(ax, color_map, ((s, "$s$"), (t, "$t$"), (T, "$T$")))
    ax.set(xlim=(s - 0.2, T + 0.6), ylabel=r"$\beta$")


@figure("schoenmakers_score_matching", height=4.0)
def plot_schoenmakers_score_matching(fig, color_map):
    ax1, ax2 = fig.subplots(1, 2, sharey=True)
    rng = np.random.default_rng(14)
    dt, T, n = 0.005, 1.0, 10
    t = np.arange(0, T + dt / 2, dt)
    mus, sds = np.array([-1.0, 1.0]), np.array([0.5, 0.5])

    def pi(x):
        return 0.5 * norm.pdf(x[..., None], mus, sds).sum(-1)

    def score(x):
        p = 0.5 * norm.pdf(x[..., None], mus, sds)
        return (p * (mus - x[..., None]) / sds**2).sum(-1) / np.maximum(p.sum(-1), 1e-12)

    free = np.array([_bm(rng, t.size, dt) for _ in range(n)])
    guided = np.zeros((n, t.size))
    for i in range(1, t.size):
        w = t[i - 1] / T
        drift = (1 - w) * -guided[:, i - 1] + w * score(guided[:, i - 1])
        guided[:, i] = guided[:, i - 1] + drift * dt + rng.normal(0, np.sqrt(dt), n)
    comp = rng.integers(0, 2, n)
    back = np.zeros((n, t.size))
    back[:, -1] = rng.normal(mus[comp], sds[comp])
    for i in range(t.size - 1, 0, -1):
        back[:, i - 1] = back[:, i] - score(back[:, i]) * dt + rng.normal(0, np.sqrt(dt), n)
    y = np.linspace(-3.2, 3.2, 300)
    for k in range(n):
        ax1.plot(t, free[k], color=color_map["c7"], alpha=0.5, lw=0.8, label=r"$x(t) = \beta(t)$" if k == 0 else None)
        ax1.plot(t, guided[k], color=color_map["c8"], alpha=0.7, lw=0.9, label=r"$y(t)$" if k == 0 else None)
        ax2.plot(t, back[k], color=color_map["c1"], alpha=0.7, lw=0.9, label=r"$x(t) = y(t)$" if k == 0 else None)
    g = norm.pdf(y, 0, np.sqrt(T))
    p = pi(y)
    ax1.plot(T + 0.25 * g / g.max(), y, color=color_map["c7"], lw=1.5, label=r"$\mathcal{N}(0, T)$")
    ax1.plot(T + 0.25 * p / p.max(), y, color=color_map["c2"], lw=1.5, ls="--", label=r"$\pi$")
    ax2.fill_betweenx(y, T, T + 0.25 * p / p.max(), color=color_map["c2"], alpha=0.3, lw=0)
    ax2.plot(T + 0.25 * p / p.max(), y, color=color_map["c2"], lw=1.5, label=r"$\pi$")
    for ax in (ax1, ax2):
        ax.set(xlim=(0, T + 0.3), ylim=(-3.2, 3.2), xlabel="$t$")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax1.set(ylabel="$x$")
