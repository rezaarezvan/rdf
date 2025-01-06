import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_probabilities(ax=None, color_map=None):
    """Plot p(y), p(x|y), and p(y|x) for Iris dataset petal length"""
    fig, (ax1, ax2) = plt.subplots(1, 2)

    X, y = datasets.load_iris(return_X_y=True)
    petal_length = X[:, 2]
    colors = list(color_map.values())[:3]
    classes = datasets.load_iris().target_names

    # # Plot 1: p(y) - Prior probabilities
    class_counts = np.bincount(y)
    priors = class_counts / len(y)
    # ax1.bar(range(3), priors, color=colors)
    # ax1.set_xticks(range(3))
    # ax1.set_xticklabels(classes, rotation=45)
    # ax1.set_title('p(y) - Class Priors')
    # ax1.set_ylabel('Probability')

    # Plot 2: p(x|y) - Class conditionals
    x_range = np.linspace(petal_length.min(), petal_length.max(), 200)
    for i, color in enumerate(colors):
        mask = y == i
        kde = gaussian_kde(petal_length[mask])
        ax1.plot(x_range, kde(x_range), color=color, label=classes[i])
        ax1.fill_between(x_range, kde(x_range), alpha=0.2, color=color)

    ax1.set_title('p(x|y) - Class Conditionals')
    ax1.set_xlabel('Petal Length (cm)')
    ax1.set_ylabel('Density')
    ax1.legend()

    # Plot 3: p(y|x) - Posteriors
    kdes = [gaussian_kde(petal_length[y == i]) for i in range(3)]
    posteriors = np.zeros((len(x_range), 3))

    for i in range(3):
        likelihood = kdes[i](x_range)
        posteriors[:, i] = likelihood * priors[i]

    posteriors /= posteriors.sum(axis=1, keepdims=True)

    for i, color in enumerate(colors):
        ax2.plot(x_range, posteriors[:, i], color=color, label=classes[i])
        ax2.fill_between(x_range, posteriors[:, i], alpha=0.2, color=color)

    ax2.set_title('p(y|x) - Posterior Probabilities')
    ax2.set_xlabel('Petal Length (cm)')
    ax2.set_ylabel('Probability')
    ax2.legend()

    plt.tight_layout()


def plot_linear_halfspace_example(ax=None, color_map=None):
    """
    Plot a simple linear halfspace example
    f(x) < 0 class -1
    f(x) > 0 class 1
    x_1 feature 1
    x_2 (y) feature 2
    w = [5, 2]^T
    b = 3

    Should color each half-space according to the class, plot some random points in each half space

    Put markers/text on the plot to show f(x) = 0, f(x) < 0, f(x) > 0 and text (class -1, class 1)
    Grid
    Blog/paper friendly/ready
    """
    fig, ax = plt.subplots()

    w = np.array([5, 2])
    b = 3
    x1 = np.linspace(-10, 10, 100)
    x2 = (-w[0] * x1 - b) / w[1]

    ax.plot(x1, x2, label='f(x) = 0')
    ax.fill_between(x1, x2, -10, alpha=0.2, color='blue', label='f(x) < 0')
    ax.fill_between(x1, x2, 10, alpha=0.2, color='red', label='f(x) > 0')

    ax.scatter(np.random.uniform(-10, 10, 10),
               np.random.uniform(-10, 10, 10), color='blue', label='class -1')
    ax.scatter(np.random.uniform(-10, 10, 10),
               np.random.uniform(-10, 10, 10), color='red', label='class 1')

    ax.set_title('Linear Halfspace Example')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()

    plt.tight_layout()


def plot_sigmoid(ax=None, color_map=None):
    """
    Plot the sigmoid function
    """
    fig, ax = plt.subplots()

    x = np.linspace(-10, 10, 100)
    y = 1 / (1 + np.exp(-x))

    ax.plot(x, y, label='sigmoid')

    ax.set_title('Sigmoid Function')
    ax.set_xlabel('f(x)')
    ax.set_ylabel(r'$\sigma(f(x))$')
    ax.legend()

    plt.tight_layout()

# def plot_


if __name__ == "__main__":
    from rdp import RDP
    plotter = RDP()
    # svg_content = plotter.create_themed_plot(plot_probabilities)
    # with open('iris_probabilities.svg', 'w') as f:
    #     f.write(svg_content)

    # svg_content = plotter.create_themed_plot(plot_linear_halfspace_example)
    # with open('linear_halfspace.svg', 'w') as f:
    #     f.write(svg_content)

    svg_content = plotter.create_themed_plot(plot_sigmoid)
    with open('sigmoid.svg', 'w') as f:
        f.write(svg_content)
