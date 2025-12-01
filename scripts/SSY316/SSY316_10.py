import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)


def plot_deterministic_approximate_inference(ax=None, color_map=None):
    """
    Create with:
        Blue: Bimodal distribution, shaped like an 8
        Red: Single Gaussian (Omega = {N(mu, sigma2)})
            - Left: q*(x) = arg min_(q(x) in Omega) KL(p(x) || q(x))
            - Middle and Right: q*(x) = arg min_(q(x) in Omega) KL(q(x) || p(x))

        Plot in 2D (level curves)
            1. Single Gaussian over the entire bimodal distribution
            2. Single Gaussian over just the left mode
            3. Single Gaussian over just the right mode
    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.4)
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Create grid
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-4, 4, 200)
    xx, yy = np.meshgrid(x, y)
    pos = np.dstack((xx, yy))

    # Bimodal distribution (figure-8 shape)
    rv1 = multivariate_normal(mean=[-1.5, 0], cov=[[0.5, 0], [0, 1]])
    rv2 = multivariate_normal(mean=[1.5, 0], cov=[[0.5, 0], [0, 1]])
    p_xy = rv1.pdf(pos) + rv2.pdf(pos)

    # Different approximations
    approximations = [
        multivariate_normal(mean=[0, 0], cov=[[3, 0], [0, 3]]),  # Overall
        multivariate_normal(
            mean=[-1.5, 0], cov=[[0.5, 0], [0, 1]]),  # Left mode
        multivariate_normal(mean=[1.5, 0], cov=[
                            [0.5, 0], [0, 1]]),   # Right mode
    ]

    titles = [
        "$\\mathrm{KL}(p \\| q)$",
        "$\\mathrm{KL}(q \\| p)$",
        "$\\mathrm{KL}(q \\| p)$",
    ]

    for i in range(3):
        q_xy = approximations[i].pdf(pos)

        # Plot level curves
        ax[i].contour(
            xx,
            yy,
            p_xy,
            levels=8,
            colors=color_map["c8"],
            linewidths=1,
            alpha=0.6,
        )
        ax[i].contour(
            xx,
            yy,
            q_xy,
            levels=8,
            colors=color_map["c2"],
            linewidths=1,
            alpha=0.6,
        )

        ax[i].set_title(titles[i], fontsize=11)
        ax[i].set_xlim(-4, 4)
        ax[i].set_ylim(-4, 4)
        ax[i].grid(True, alpha=0.15)
        # Remove top and right spines (SDE style)
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)

        # Remove ticks and numbers from x and y axes
        ax[i].set_xticks([])
        ax[i].set_yticks([])


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()

    svg_content = plotter.create_themed_plot(
        save_name="deterministic_approximate_inference",
        plot_func=plot_deterministic_approximate_inference,
    )
