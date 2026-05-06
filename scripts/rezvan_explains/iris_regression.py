import numpy as np
import sklearn.datasets as datasets

from scipy.stats import gaussian_kde

from rdf import figure


@figure("iris_probabilities")
def plot_probabilities(fig, color_map):
    """Plot p(x|y) and p(y|x) for Iris dataset petal length."""
    ax1, ax2 = fig.subplots(1, 2)

    X, y = datasets.load_iris(return_X_y=True)
    petal_length = X[:, 2]
    colors = list(color_map.values())[:3]
    classes = datasets.load_iris().target_names

    class_counts = np.bincount(y)
    priors = class_counts / len(y)

    x_range = np.linspace(petal_length.min(), petal_length.max(), 200)
    for i, color in enumerate(colors):
        mask = y == i
        kde = gaussian_kde(petal_length[mask])
        ax1.plot(x_range, kde(x_range), color=color, label=classes[i])
        ax1.fill_between(x_range, kde(x_range), alpha=0.2, color=color)

    ax1.set_title("p(x|y) - Class Conditionals")
    ax1.set_xlabel("Petal Length (cm)")
    ax1.set_ylabel("Density")
    ax1.legend()

    kdes = [gaussian_kde(petal_length[y == i]) for i in range(3)]
    posteriors = np.zeros((len(x_range), 3))
    for i in range(3):
        posteriors[:, i] = kdes[i](x_range) * priors[i]
    posteriors /= posteriors.sum(axis=1, keepdims=True)

    for i, color in enumerate(colors):
        ax2.plot(x_range, posteriors[:, i], color=color, label=classes[i])
        ax2.fill_between(x_range, posteriors[:, i], alpha=0.2, color=color)

    ax2.set_title("p(y|x) - Posterior Probabilities")
    ax2.set_xlabel("Petal Length (cm)")
    ax2.set_ylabel("Probability")
    ax2.legend()
    fig.tight_layout()


@figure("linear_halfspace")
def plot_linear_halfspace_example(ax):
    """Linear halfspace: f(x) = w·x + b, w=[5,2], b=3."""
    w = np.array([5, 2])
    b = 3
    x1 = np.linspace(-10, 10, 100)
    x2 = (-w[0] * x1 - b) / w[1]

    ax.plot(x1, x2, label="f(x) = 0")
    ax.fill_between(x1, x2, -10, alpha=0.2, color="blue", label="f(x) < 0")
    ax.fill_between(x1, x2, 10, alpha=0.2, color="red", label="f(x) > 0")
    ax.scatter(
        np.random.uniform(-10, 10, 10),
        np.random.uniform(-10, 10, 10),
        color="blue",
        label="class -1",
    )
    ax.scatter(
        np.random.uniform(-10, 10, 10),
        np.random.uniform(-10, 10, 10),
        color="red",
        label="class 1",
    )
    ax.set_title("Linear Halfspace Example")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()


@figure("sigmoid")
def plot_sigmoid(ax):
    """Plot the sigmoid function."""
    x = np.linspace(-10, 10, 100)
    y = 1 / (1 + np.exp(-x))
    ax.plot(x, y, label="sigmoid")
    ax.set_title("Sigmoid Function")
    ax.set_xlabel("f(x)")
    ax.set_ylabel(r"$\sigma(f(x))$")
    ax.legend()
