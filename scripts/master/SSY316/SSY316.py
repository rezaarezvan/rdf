import numpy as np
from scipy.stats import beta

from rdf import figure

MU = np.linspace(0, 1, 400)


@figure("beta_distribution_examples", height=3.2)
def plot_beta_distribution(ax, color_map):
    for (a, b), c in zip(((1, 1), (0.1, 0.1), (2, 4)), ("c8", "c1", "c2")):
        ax.plot(MU, beta.pdf(MU, a, b), color=color_map[c], label=rf"$\mathrm{{Beta}}({a}, {b})$")
    ax.set(xlim=(0, 1), ylim=(0, 3), xlabel=r"$\mu$", ylabel="Density")
    ax.legend(loc="upper right", bbox_to_anchor=(0.9, 0.95))


def _prior_likelihood_posterior(fig, color_map, a, b, n=5, h=4):
    axs = fig.subplots(1, 3)
    curves = (
        (beta.pdf(MU, a, b), rf"$p(\mu) = \mathrm{{Beta}}({a}, {b})$", "c8"),
        (MU**h * (1 - MU) ** (n - h), rf"$p(\mathcal{{D}} \mid \mu) = \mu^{h}(1 - \mu)$", "c1"),
        (beta.pdf(MU, a + h, b + n - h), rf"$p(\mu \mid \mathcal{{D}}) = \mathrm{{Beta}}({a + h}, {b + n - h})$", "c2"),
    )
    for ax, (y, label, c) in zip(axs, curves):
        ax.plot(MU, y, color=color_map[c])
        ax.set(xlim=(0, 1), ylim=(0, 1.2 * y.max()), xlabel=r"$\mu$", yticks=[])
        ax.set_title(label, fontsize=10)


@figure("prior_likelihood_posterior", height=2.6)
def plot_prior_likelihood_posterior(fig, color_map):
    _prior_likelihood_posterior(fig, color_map, 1, 1)


@figure("prior_likelihood_posterior2", height=2.6)
def plot_prior_likelihood_posterior2(fig, color_map):
    _prior_likelihood_posterior(fig, color_map, 5, 2)
