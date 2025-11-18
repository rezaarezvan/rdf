import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import beta


def plot_ml_learning_example(ax=None, color_map=None):
    """
    Create a plot demonstrating Maximum Likelihood (ML) learning example.

    p(x, y) = p(x) p(y | x)
    x ~ Uniform(0, 1)
    y | x ~ N(sin (2pi x), 0.1)

    Optimal predictor under quadratic loss: hat(y)^* := sin(2pi x)
    ML predictor: hat(y)_ML := mu(x, w_ML)

    D = {(x_i, y_i)}_{i=1}^N, N=10
    Plot baseline, sin(2pi x) dotted black
    and M = 1 (green), M = 3 (red), M = 9 (blue) ML predictors

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Generate data
    N = 10
    x_data = np.linspace(0, 1, N)
    y_data = np.sin(2 * np.pi * x_data) + np.random.normal(0, 0.1, N)

    # True function
    x_true = np.linspace(0, 1, 400)
    y_true = np.sin(2 * np.pi * x_true)
    ax.plot(x_true, y_true, "k--", label="True function", linewidth=2)

    # ML predictors for different model complexities
    model_complexities = [1, 3, 9]
    colors = ["green", "red", "blue"]
    for M, color in zip(model_complexities, colors):
        coeffs = np.polyfit(x_data, y_data, M)
        y_ml = np.polyval(coeffs, x_true)
        ax.plot(x_true, y_ml, color=color, label=f"ML Predictor (M={M})", linewidth=2)

    ax.scatter(x_data, y_data, color="black", zorder=5, label="Data points")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.legend(fontsize=10)

    ax.grid(True, alpha=0.15)

    # Remove top and right spines (SDE style)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)

    # Plot from x = [0, 1], y = [-3, 3]
    ax.set_xlim(0, 1)
    ax.set_ylim(-3, 3)


def plot_map_learning_example(ax=None, color_map=None):
    """
    Create a plot demonstrating Maximum A Posteriori (MAP) learning example.

    Problem: Fitting straight line to noisy measurements generated from,

    f(x, a) = a_0 + a_1 x, a_0 = -0.3, a_1 = 0.5

    by adding Gaussian noise N(0, 0.04).

    Model:
        y(x, w) = w_0 + w_1 x + \epsilon, epsilon ~N(0, 0.04)

    with prior:
        p(w) = N(w | (0 0)^T, alpha^{-1} I_2), alpha = 2

    Perform 200 measurements, plot prior, likelihood, and posterior over w, (subplot 1x3).

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.4)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Generate data
    np.random.seed(0)
    N = 200
    x_data = np.random.uniform(0, 1, N)
    a_0, a_1 = -0.3, 0.5
    y_data = a_0 + a_1 * x_data + np.random.normal(0, 0.04**0.5, N)
    # Prior parameters
    alpha = 2
    # Prior
    w0 = np.linspace(-1, 1, 100)
    w1 = np.linspace(-1, 1, 100)
    W0, W1 = np.meshgrid(w0, w1)
    prior = np.exp(-0.5 * alpha * (W0**2 + W1**2))
    axes[0].contourf(W0, W1, prior, levels=50, cmap="Blues")
    axes[0].set_title("Prior Distribution")
    axes[0].set_xlabel(r"$w_0$")
    axes[0].set_ylabel(r"$w_1$")
    # Likelihood
    likelihood = np.zeros_like(prior)
    for i in range(len(w0)):
        for j in range(len(w1)):
            w = np.array([w0[i], w1[j]])
            y_pred = w[0] + w[1] * x_data
            likelihood[j, i] = np.exp(-0.5 * np.sum((y_data - y_pred) ** 2) / 0.04)
    axes[1].contourf(W0, W1, likelihood, levels=50, cmap="Oranges")
    axes[1].set_title("Likelihood Function")
    axes[1].set_xlabel(r"$w_0$")
    axes[1].set_ylabel(r"$w_1$")
    # Posterior
    posterior = prior * likelihood
    axes[2].contourf(W0, W1, posterior, levels=50, cmap="Greens")
    axes[2].set_title("Posterior Distribution")
    axes[2].set_xlabel(r"$w_0$")
    axes[2].set_ylabel(r"$w_1$")

    for ax in axes:
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=9)

    plt.show()


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()

    # svg_content = plotter.create_themed_plot(
    #     save_name="ml_learning_example",
    #     plot_func=plot_ml_learning_example
    # )

    svg_content = plotter.create_themed_plot(
        save_name="map_learning_example", plot_func=plot_map_learning_example
    )
