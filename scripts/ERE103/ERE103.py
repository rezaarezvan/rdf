import numpy as np
from matplotlib.ticker import NullFormatter

from rdf import figure

W = np.logspace(-1, 1, 400)


def _bode(fig, color_map, g, phase_ticks):
    mag, ph = fig.subplots(2, 1, sharex=True)
    h = g(1j * W)
    mag.loglog(W, np.abs(h), color=color_map["c8"])
    ph.semilogx(W, np.degrees(np.unwrap(np.angle(h))), color=color_map["c1"])
    mag.set(ylabel=r"$|G(j\omega)|$")
    mag.yaxis.set_minor_formatter(NullFormatter())
    lo, hi = np.log10(np.abs(h).min()), np.log10(np.abs(h).max())
    mag.set_ylim(10 ** np.floor(lo + 1e-9), 10 ** np.ceil(hi - 1e-9))
    ph.set(ylabel=r"$\arg G(j\omega)$ ($^\circ$)", xlabel=r"$\omega$ (rad/s)", xlim=(W[0], W[-1]), yticks=phase_ticks)
    mag.grid(True, which="both")
    ph.grid(True, which="both")


@figure("bode", height=4.4)
def plot_bode(fig, color_map):
    _bode(fig, color_map, lambda s: 0.2 * (s + 1) / (1.5625 * s**2 + 0.375 * s + 1), [-135, -90, -45, 0])


@figure("bode2", height=4.4)
def plot_bode2(fig, color_map):
    _bode(fig, color_map, lambda s: 1 / (s * (s + 2)), [-180, -135, -90])


@figure("bode3", height=4.4)
def plot_bode3(fig, color_map):
    _bode(fig, color_map, lambda s: s, [0, 45, 90, 135])


@figure("bode4", height=4.4)
def plot_bode4(fig, color_map):
    _bode(fig, color_map, lambda s: 1 / s, [-135, -90, -45, 0])


@figure("bode5", height=4.4)
def plot_bode5(fig, color_map):
    _bode(fig, color_map, lambda s: 1 + s / 2, [0, 45, 90])


@figure("bode6", height=4.4)
def plot_bode6(fig, color_map):
    _bode(fig, color_map, lambda s: 1 / (1 + s / 2), [-90, -45, 0])


@figure("lowpass", height=4.4)
def plot_lowpass(fig, color_map):
    _bode(fig, color_map, lambda s: 1 / (s + 1), [-90, -45, 0])


@figure("highpass", height=4.4)
def plot_highpass(fig, color_map):
    _bode(fig, color_map, lambda s: s / (s + 1), [0, 45, 90])


def _nyquist(ax, color_map, l, w, xlim, ylim):
    h = l(1j * w)
    ax.plot(h.real, h.imag, color=color_map["c8"])
    ax.plot(h.real, -h.imag, color=color_map["c8"], ls="--", lw=0.8)
    i = len(w) // 3
    ax.annotate("", xy=(h.real[i + 1], h.imag[i + 1]), xytext=(h.real[i], h.imag[i]),
                arrowprops=dict(arrowstyle="-|>", color=color_map["c8"], lw=0, mutation_scale=12))
    ax.plot(-1, 0, "+", color=color_map["c1"], ms=9, mew=1.5)
    ax.axhline(0, color=color_map["black"], lw=0.6)
    ax.axvline(0, color=color_map["black"], lw=0.6)
    ax.set(xlim=xlim, ylim=ylim, xlabel=r"$\mathrm{Re}\, L(j\omega)$", ylabel=r"$\mathrm{Im}\, L(j\omega)$", aspect="equal")


@figure("nyquist", height=4.2)
def plot_nyquist(ax, color_map):
    _nyquist(ax, color_map, lambda s: np.exp(-s) / (s * (1 + s)), np.logspace(-0.3, 1.5, 3000), (-1.6, 0.6), (-1.6, 1.6))


def _k(ax, color_map, k):
    _nyquist(ax, color_map, lambda s: k / (s * (s + 1) * (s + 2)), np.logspace(-0.4, 1.5, 3000), (-3, 1), (-2.4, 2.4))
    ax.plot(-k / 6, 0, "o", color=color_map["c2"], ms=4)


@figure("k2", height=4.2)
def plot_k2(ax, color_map):
    _k(ax, color_map, 2)


@figure("k6", height=4.2)
def plot_k6(ax, color_map):
    _k(ax, color_map, 6)


@figure("k12", height=4.2)
def plot_k12(ax, color_map):
    _k(ax, color_map, 12)
