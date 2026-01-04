import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal


def plot_monte_carlo_inference(ax=None, color_map=None):
    """
    Create with:
        p(z) bi modal Gaussian where the left one is higher
        f(z) looks like a logistic sigmoid

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    x = np.linspace(-5, 5, 800)

    # Bimodal density (make left peak taller)
    p_z = 0.9 * multivariate_normal.pdf(
        x, mean=-1.5, cov=0.4
    ) + 0.6 * multivariate_normal.pdf(x, mean=1.5, cov=0.7)

    # Smooth sigmoid curve
    f_z = 1 / (1 + np.exp(-0.8 * x))

    # Axis styling to mimic the reference figure
    ax.set_xlabel("z", fontsize=14)
    ax.set_ylim(-0.05, max(p_z) * 1.2)
    ax.set_xlim(min(x), max(x))
    ax.plot(x, p_z, label="p(z)", color=color_map["c8"], linewidth=2)
    ax.plot(x, f_z, label="f(z)", color=color_map["c7"], linewidth=2)
    ax.set_xlabel("z", fontsize=12)
    ax.legend(fontsize=10)


def plot_monte_carlo_inference_importance_sampling(ax=None, color_map=None):
    """
    Create with:
        p(z) bi modal Gaussian where the left one is higher
        f(z) looks like a logistic sigmoid
        q(z) Gaussian centered between the two modes of p(z)

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    x = np.linspace(-5, 5, 800)

    # Bimodal density (make left peak taller)
    p_z = 0.9 * multivariate_normal.pdf(
        x, mean=-1.5, cov=0.4
    ) + 0.6 * multivariate_normal.pdf(x, mean=1.5, cov=0.7)

    # Smooth sigmoid curve
    f_z = 1 / (1 + np.exp(-0.8 * x))

    # Importance sampling proposal distribution q(z)
    q_z = multivariate_normal.pdf(x, mean=0, cov=2.0)

    # Axis styling to mimic the reference figure
    ax.set_xlabel("z", fontsize=14)
    ax.set_ylim(-0.05, max(p_z) * 1.2)
    ax.set_xlim(min(x), max(x))
    ax.plot(x, p_z, label="p(z)", color=color_map["c8"], linewidth=2)
    ax.plot(x, f_z, label="f(z)", color=color_map["c7"], linewidth=2)
    ax.plot(x, q_z, label="q(z)", color=color_map["c2"], linewidth=2)
    ax.set_xlabel("z", fontsize=12)
    ax.legend(fontsize=10)


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()

    svg_content = plotter.create_themed_plot(
        name="monte_carlo_inference", plot_func=plot_monte_carlo_inference
    )
    svg_content = plotter.create_themed_plot(
        name="monte_carlo_inference_importance_sampling",
        plot_func=plot_monte_carlo_inference_importance_sampling,
    )
