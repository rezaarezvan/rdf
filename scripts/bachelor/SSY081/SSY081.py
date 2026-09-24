import numpy as np

from rdf import figure

T1, T2 = -1.0, 2.0


def _tri(t):
    return np.interp(t, [T1, 0, T2], [0, 1, 0], left=0, right=0)


def _signal_axes(ax, ticks=(), xlim=(-5, 5), ylim=(-0.3, 1.4), ylabel="$f(t)$"):
    ax.grid(False)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.set(xlim=xlim, ylim=ylim, yticks=[])
    ax.set_xticks([v for v, _ in ticks], [s for _, s in ticks])
    ax.plot(1, 0, ">", color=ax.spines["bottom"].get_edgecolor(), transform=ax.get_yaxis_transform(), clip_on=False, ms=4)
    ax.text(1, 0.06, "$t$", transform=ax.get_yaxis_transform(), ha="right")
    ax.text(0.08, ylim[1], ylabel, va="top")


def _panels(fig, n):
    return fig.subplots(n, 1) if n > 1 else [fig.subplots()]


@figure("time", height=3.6)
def plot_time(ax, color_map):
    t = np.linspace(0, 24, 400)
    temp = 20 - 5 * np.cos(2 * np.pi * (t - 3.5) / 24.0 * 0.92)
    k = np.arange(25)
    tk = 20 - 5 * np.cos(2 * np.pi * (k - 3.5) / 24.0 * 0.92)
    ax.plot(t, temp, color=color_map["c8"], label="Continuous-time")
    ax.vlines(k, 15, tk, color=color_map["c1"], lw=0.8)
    ax.plot(k, tk, "o", color=color_map["c1"], ms=3.5, label="Discrete-time")
    ax.set(xlim=(0, 24), ylim=(15, 26), xticks=range(0, 25, 4), xlabel="Time (h)", ylabel=r"Temperature ($^\circ$C)")
    ax.legend(loc="upper left")


@figure("amp", height=3.6)
def plot_amp(ax, color_map):
    t = np.linspace(0, 24, 400)
    temp = 20 - 5 * np.cos(2 * np.pi * (t - 3.5) / 24.0 * 0.92)
    ax.plot(t, temp, color=color_map["c8"], label="Continuous-amplitude (analog)")
    ax.step(t, np.round(temp), where="mid", color=color_map["c1"], label="Discrete-amplitude (digital)")
    ax.set(xlim=(0, 24), ylim=(14, 27), xticks=range(0, 25, 4), yticks=range(15, 27), xlabel="Time (h)", ylabel=r"Temperature ($^\circ$C)")
    ax.set_yticklabels([str(v) if v % 5 == 0 else "" for v in range(15, 27)])
    ax.legend(loc="upper left")


@figure("even", height=2.4)
def plot_even(fig, color_map):
    axs = fig.subplots(1, 2)
    t = np.linspace(-4, 4, 400)
    bump = np.exp(-t**2 / 2) + 0.35 * np.exp(-(np.abs(t) - 2) ** 2 / 0.3)
    for ax, y, lab in ((axs[0], bump, "$f_e(t)$"), (axs[1], np.sign(t) * bump, "$f_o(t)$")):
        ax.plot(t, y, color=color_map["c8"])
        _signal_axes(ax, [(-2, "$-a$"), (2, "$a$")], xlim=(-4, 4), ylim=(-1.4, 1.4), ylabel=lab)
        for a in (-2, 2):
            ax.plot([a, a], [0, y[np.argmin(np.abs(t - a))]], ls=":", color=color_map["c7"], lw=0.8)


@figure("period", height=2.4)
def plot_period(ax, color_map):
    t = np.linspace(-7, 7, 1200)
    ph = np.mod(t + 1, 4) - 1
    y = np.where(ph < 0, -0.3 * (ph + 1) ** 0.5 * np.exp(-(ph + 1) * 3) * 3, np.exp(-((ph - 1) ** 2) / 0.25))
    y[np.abs(np.diff(ph, prepend=ph[0])) > 1] = np.nan
    ax.plot(t, y, color=color_map["c8"])
    _signal_axes(ax, [], xlim=(-7, 7), ylim=(-0.8, 1.4))
    for x in (-1, 3):
        ax.plot([x, x], [-0.65, 0], ls=":", color=color_map["c7"], lw=0.8)
    ax.annotate("", xy=(3, -0.55), xytext=(-1, -0.55), arrowprops=dict(arrowstyle="<->", color=color_map["black"], lw=0.8))
    ax.text(1, -0.72, "$T_0$", ha="center", va="top")


def _ops(fig, color_map, rows, ticks, xlim=(-3, 5), ylim=(-0.3, 1.4)):
    axs = _panels(fig, len(rows))
    for ax, (y, lab, tk, c) in zip(axs, rows):
        t = np.linspace(*xlim, 800)
        ax.plot(t, y(t), color=color_map[c])
        _signal_axes(ax, tk, xlim=xlim, ylim=ylim, ylabel=lab)
    return axs


def _ymark(ax, y, s):
    ax.text(-0.3, y, s, ha="right", va="center")


TICKS = [(T1, "$T_1$"), (T2, "$T_2$")]


@figure("gain", height=3.8)
def plot_gain(fig, color_map):
    top, bottom = _ops(fig, color_map, [
        (_tri, "$f(t)$", TICKS, "c8"),
        (lambda t: 1.6 * _tri(t), r"$\phi(t) = A f(t)$", TICKS, "c1"),
    ], TICKS, ylim=(-0.3, 2))
    _ymark(top, 1, "$1$")
    _ymark(bottom, 1.6, "$A$")


@figure("DC_offset", height=3.8)
def plot_dc_offset(fig, color_map):
    top, bottom = _ops(fig, color_map, [
        (_tri, "$f(t)$", TICKS, "c8"),
        (lambda t: _tri(t) + 0.4, r"$\phi(t) = f(t) + B$", TICKS, "c1"),
    ], TICKS, ylim=(-0.3, 1.8))
    _ymark(top, 1, "$1$")
    bottom.annotate("", xy=(-2.6, 0.4), xytext=(-2.6, 0), arrowprops=dict(arrowstyle="<->", color=color_map["black"], lw=0.8))
    bottom.text(-2.7, 0.2, "$B$", ha="right", va="center")


@figure("expand_compress", height=5.2)
def plot_expand_compress(fig, color_map):
    _ops(fig, color_map, [
        (_tri, "$f(t)$", TICKS, "c8"),
        (lambda t: _tri(2 * t), r"$\phi(t) = f(2t)$", [(T1 / 2, "$T_1/2$"), (T2 / 2, "$T_2/2$")], "c1"),
        (lambda t: _tri(t / 2), r"$\phi(t) = f(t/2)$", [(2 * T1, "$2T_1$"), (2 * T2, "$2T_2$")], "c2"),
    ], TICKS, xlim=(-3, 5))


@figure("invert", height=3.8)
def plot_invert(fig, color_map):
    _ops(fig, color_map, [
        (_tri, "$f(t)$", TICKS, "c8"),
        (lambda t: _tri(-t), r"$\phi(t) = f(-t)$", [(-T2, "$-T_2$"), (-T1, "$-T_1$")], "c1"),
    ], TICKS, xlim=(-4, 4))


@figure("shift", height=5.2)
def plot_shift(fig, color_map):
    T = 1.5
    _ops(fig, color_map, [
        (_tri, "$f(t)$", TICKS, "c8"),
        (lambda t: _tri(t - T), r"$\phi(t) = f(t - T)$", [(T1 + T, "$T_1 + T$"), (T2 + T, "$T_2 + T$")], "c1"),
        (lambda t: _tri(t + T), r"$\phi(t) = f(t + T)$", [(T1 - T, "$T_1 - T$"), (T2 - T, "$T_2 - T$")], "c2"),
    ], TICKS, xlim=(-3.5, 4.5))


@figure("unit", height=2.2)
def plot_unit(ax, color_map):
    t = np.linspace(-3, 5, 800)
    ax.plot(t, np.where(t >= 0, 1.0, np.nan), color=color_map["c8"])
    ax.plot(t, np.where(t < 0, 0.0, np.nan), color=color_map["c8"])
    ax.plot(0, 1, "o", color=color_map["c8"], ms=4)
    _signal_axes(ax, [], xlim=(-3, 5), ylabel="$u(t)$")
    ax.text(-0.15, 1, "$1$", ha="right", va="center")


@figure("rect", height=2.2)
def plot_rect(ax, color_map):
    t = np.linspace(-1, 6, 1400)
    ax.plot(t, ((t >= 2) & (t < 4)).astype(float), color=color_map["c8"])
    _signal_axes(ax, [(2, "$2$"), (4, "$4$")], xlim=(-1, 6))
    ax.text(-0.1, 1, "$1$", ha="right", va="center")


@figure("dirac", height=2.2)
def plot_dirac(ax, color_map):
    ax.annotate("", xy=(0, 1.1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=color_map["c8"], lw=1.6, mutation_scale=14))
    _signal_axes(ax, [], xlim=(-3, 3), ylabel="")
    ax.text(0.15, 1.05, r"$\delta(t)$", color=color_map["c8"])


def _ecg(seed=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8, 4000)
    beats = sum(np.exp(-((t - b) ** 2) / 2e-4) - 0.35 * np.exp(-((t - b - 0.04) ** 2) / 4e-4) for b in (1.1, 3.0, 4.9, 6.8))
    wander = 0.6 * np.sin(2 * np.pi * 0.15 * t) + 0.25 * np.sin(2 * np.pi * 0.4 * t + 1)
    return t, beats + wander + 0.03 * rng.standard_normal(t.size)


def _highpass(x, n=200):
    return x - np.convolve(x, np.ones(n) / n, mode="same")


def _trace(ax, t, x, color):
    ax.plot(t, x, color=color, lw=0.9)
    ax.set(xlim=(0.2, 7.8), xlabel="$t$", xticks=[], yticks=[])
    ax.grid(False)


@figure("signal", height=2.0)
def plot_signal(ax, color_map):
    t, x = _ecg()
    _trace(ax, t, x, color_map["c8"])
    ax.set_ylabel("$x(t)$")


@figure("signal_out", height=2.0)
def plot_signal_out(ax, color_map):
    t, x = _ecg()
    _trace(ax, t, _highpass(x), color_map["c1"])
    ax.set_ylabel("$y(t)$")
