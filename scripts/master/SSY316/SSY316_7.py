import numpy as np
from scipy.stats import norm

from rdf import figure

Z = np.linspace(-5, 5, 800)
P = 0.9 * norm.pdf(Z, -1.5, np.sqrt(0.4)) + 0.6 * norm.pdf(Z, 1.5, np.sqrt(0.7))
F = 1 / (1 + np.exp(-0.8 * Z))


def _base(ax, color_map):
    ax.plot(Z, P, color=color_map["c8"], label="$p(z)$")
    ax.plot(Z, F, color=color_map["c1"], label="$f(z)$")
    ax.set(xlim=(-5, 5), ylim=(0, 1.05), xlabel="$z$")


@figure("monte_carlo_inference", height=3.0)
def plot_monte_carlo_inference(ax, color_map):
    _base(ax, color_map)
    ax.legend(loc="upper left")


@figure("monte_carlo_inference_importance_sampling", height=3.0)
def plot_monte_carlo_inference_importance_sampling(ax, color_map):
    _base(ax, color_map)
    ax.plot(Z, norm.pdf(Z, 0, np.sqrt(2)), color=color_map["c2"], label="$q(z)$")
    ax.legend(loc="upper left")
