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


if __name__ == "__main__":
    from rdp import RDP
    plotter = RDP()

    svg_content = plotter.create_themed_plot(plot_probabilities)
    with open('iris_probabilities.svg', 'w') as f:
        f.write(svg_content)
