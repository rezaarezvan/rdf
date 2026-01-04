import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import beta, gamma, nbinom, poisson


def plot_beta_prior_beta_posterior(ax=None, color_map=None):
    """
    Plot Beta prior and posterior distributions for a Bernoulli process.

    Specifically,

    prior = Beta(33.4, 33.4)
    posterior = Beta(33.4 + 11, 33.4 + 19)

    plot posterior dashed.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """

    x = np.linspace(0, 1, 100)
    prior_a, prior_b = 33.4, 33.4
    posterior_a, posterior_b = 33.4 + 11, 33.4 + 19

    prior_pdf = beta.pdf(x, prior_a, prior_b)
    posterior_pdf = beta.pdf(x, posterior_a, posterior_b)

    ax.plot(
        x,
        prior_pdf,
        label="Prior Beta(33.4, 33.4)",
        color=color_map["c8"],
        linestyle="--",
        linewidth=2,
    )
    ax.plot(
        x,
        posterior_pdf,
        label="Posterior Beta(44.4, 52.4)",
        color=color_map["c7"],
        linewidth=2,
    )

    ax.set_title("Beta Prior and Posterior Distributions")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Density")
    ax.legend()


def plot_poison_gamma_conjugacy(ax=None, color_map=None):
    """
    Plot Poisson likelihood and Gamma prior/posterior distributions.

    Specifically,

    Poisson(theta) for some theta > 0.
    Observed values, x_1 = 20, x_2 = 24, x_3 = 23.

    Prior = pi(theta) propto_theta 1/theta.

    Posterior after x_1: theta | x_1 ~ Gamma(20, 1)

    Using this as the prior, we get the posterior after x_2:
    Posterior after x_2: theta | x_1, x_2 ~ Gamma(20+24, 1+1)

    Finally, after x_3:
    Posterior after x_3: theta | x_1, x_2, x_3 ~ Gamma(20+24+23, 1+1+1)

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """

    x = np.linspace(0, 50, 1000)

    # Prior: improper prior pi(theta) propto 1/theta
    prior_pdf = 1 / x
    prior_pdf[x <= 0] = 0  # Avoid division by zero

    # Posterior after x_1 = 20
    posterior1_pdf = gamma.pdf(x, a=20, scale=1 / 1)

    # Posterior after x_2 = 24
    posterior2_pdf = gamma.pdf(x, a=44, scale=1 / 2)

    # Posterior after x_3 = 23
    posterior3_pdf = gamma.pdf(x, a=67, scale=1 / 3)

    ax.plot(x, posterior1_pdf, label="Posterior after x1", color=color_map["c8"])
    ax.plot(
        x,
        posterior2_pdf,
        label="Posterior after x2",
        color=color_map["c7"],
        linestyle="--",
    )
    ax.plot(
        x,
        posterior3_pdf,
        label="Posterior after x3",
        color=color_map["c6"],
        linestyle=":",
    )

    ax.set_title("Poisson-Gamma Conjugacy")
    ax.set_xlabel("Theta")
    ax.set_ylabel("Density")
    ax.legend()


def plot_poison_gamma_conjugacy_prediction(ax=None, color_map=None):
    """
    Plot predictive distributions for Poisson-Gamma conjugacy.

    Specifically,

    Observed values, x_1 = 20, x_2 = 24, x_3 = 23.

    Prior = pi(theta) propto_theta 1/theta.

    Posterior after x_1: theta | x_1 ~ Gamma(20, 1)

    Using this as the prior, we get the posterior after x_2:
    Posterior after x_2: theta | x_1, x_2 ~ Gamma(20+24, 1+1)

    Finally, after x_3:
    Posterior after x_3: theta | x_1, x_2, x_3 ~ Gamma(20+24+23, 1+1+1)

    For this, we are trying to predict the values of x_4.
    Bayesian inference VS. frequentist inference.

    Bayesian:
        Negative-binomial(67, 3/(3+1))
    Frequentist:
        Poisson(67/3)

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    x = np.arange(0, 50)

    # Bayesian predictive distribution
    bayesian_pred_pdf = nbinom.pmf(x, n=67, p=3 / (3 + 1))

    # Frequentist predictive distribution
    frequentist_pred_pdf = poisson.pmf(x, mu=67 / 3)

    ax.plot(x, bayesian_pred_pdf, label="Bayesian Predictive", color=color_map["c8"])
    ax.plot(
        x,
        frequentist_pred_pdf,
        label="Frequentist Predictive",
        color=color_map["c7"],
        linestyle="--",
    )

    ax.set_title("Predictive Distributions: Bayesian vs Frequentist")
    ax.set_xlabel("$x_4$")
    ax.set_ylabel("Probability")
    ax.legend()


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()
    svg_content = plotter.create_themed_plot(
        name="beta_prior_posterior", plot_func=plot_beta_prior_beta_posterior
    )
    svg_content = plotter.create_themed_plot(
        name="poisson_gamma_conjugacy", plot_func=plot_poison_gamma_conjugacy
    )
    svg_content = plotter.create_themed_plot(
        name="poisson_gamma_conjugacy_prediction",
        plot_func=plot_poison_gamma_conjugacy_prediction,
    )
