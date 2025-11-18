import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import beta


def plot_beta_distribution(ax=None, color_map=None):
    """
    Create a plot of the Beta distribution for demonstration.

    Three realizations of the Beta distribution with different parameters,

    1. a = 1, b = 1
    2. a = 0.1, b = 0.1
    3. a = 2, b = 4

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    x = np.linspace(0, 1, 400)

    params = [(1, 1), (0.1, 0.1), (2, 4)]
    labels = [
        r"$\mathrm{Beta}(1, 1)$",
        r"$\mathrm{Beta}(0.1, 0.1)$",
        r"$\mathrm{Beta}(2, 4)$",
    ]

    for (a, b), label in zip(params, labels):
        y = beta.pdf(x, a, b)
        ax.plot(x, y, label=label, linewidth=2)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.legend(fontsize=10)

    ax.grid(True, alpha=0.15)

    # Remove top and right spines (SDE style)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)


def prior_likelihood_posterior_plot(ax=None, color_map=None):
    """
    Create a subfigure (1x3 grid) showing the prior, likelihood, and posterior distributions for:

        1. Prior: p(mu) = Beta(mu; a, b) with a=1, b=1
        2. Likelihood: p(D|mu) = mu^h (1 - mu)^(N-h) with N=5, h=4
        3. Posterior: p(mu|D) = Beta(mu; a', b') with a'=5, b'=2

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.4)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    x = np.linspace(0, 1, 400)

    # Prior
    a_prior, b_prior = 4, 2
    y_prior = beta.pdf(x, a_prior, b_prior)
    axes[0].plot(x, y_prior, color="blue", linewidth=2)
    axes[0].set_title("Prior Distribution")
    axes[0].set_xlabel(r"$\mu$")
    axes[0].set_ylabel("Density")

    # Likelihood
    N, h = 5, 4
    y_likelihood = x**h * (1 - x) ** (N - h)
    axes[1].plot(x, y_likelihood, color="orange", linewidth=2)
    axes[1].set_title("Likelihood Function")
    axes[1].set_xlabel(r"$\mu$")
    axes[1].set_ylabel("Likelihood")

    # Posterior
    a_post, b_post = a_prior + h, b_prior + (N - h)
    y_post = beta.pdf(x, a_post, b_post)
    axes[2].plot(x, y_post, color="green", linewidth=2)
    axes[2].set_title("Posterior Distribution")
    axes[2].set_xlabel(r"$\mu$")
    axes[2].set_ylabel("Density")

    for ax in axes:
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=9)

    plt.tight_layout()


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()
    # svg_content = plotter.create_themed_plot(
    #     save_name="beta_distribution_examples",
    #     plot_func=plot_beta_distribution
    # )

    svg_content = plotter.create_themed_plot(
        save_name="prior_likelihood_posterior2",
        plot_func=prior_likelihood_posterior_plot,
    )
