import numpy as np
from scipy.stats import beta, gamma, nbinom, poisson

from rdf import figure


@figure("beta_prior_posterior", height=3.2)
def plot_beta_prior_posterior(ax, color_map):
    x = np.linspace(0, 1, 400)
    ax.plot(x, beta.pdf(x, 33.4, 33.4), color=color_map["c8"], ls="--", label=r"Prior $\mathrm{Beta}(33.4, 33.4)$")
    ax.plot(x, beta.pdf(x, 44.4, 52.4), color=color_map["c1"], label=r"Posterior $\mathrm{Beta}(44.4, 52.4)$")
    ax.set(xlabel=r"$\theta$", ylabel="Density", xlim=(0, 1))
    ax.legend(loc="upper right")


@figure("poisson_gamma_conjugacy", height=3.2)
def plot_poisson_gamma_conjugacy(ax, color_map):
    x = np.linspace(0, 50, 1000)
    for (a, b), c, ls in (((20, 1), "c8", "-"), ((44, 2), "c1", "--"), ((67, 3), "c2", ":")):
        ax.plot(x, gamma.pdf(x, a, scale=1 / b), color=color_map[c], ls=ls, label=rf"$\mathrm{{Gamma}}({a}, {b})$")
    ax.set(xlabel=r"$\theta$", ylabel="Density", xlim=(0, 50))
    ax.legend(loc="upper right")


@figure("poisson_gamma_conjugacy_prediction", height=3.2)
def plot_poisson_gamma_conjugacy_prediction(ax, color_map):
    x = np.arange(0, 50)
    ax.plot(x, nbinom.pmf(x, 67, 3 / 4), "o-", color=color_map["c8"], ms=3, lw=1, label=r"Bayesian, $\mathrm{NegBin}(67, 1/4)$")
    ax.plot(x, poisson.pmf(x, 67 / 3), "s--", color=color_map["c1"], ms=3, lw=1, label=r"Frequentist, $\mathrm{Poisson}(67/3)$")
    ax.set(xlabel="$x_4$", ylabel="Probability", xlim=(0, 49))
    ax.legend(loc="upper right")
