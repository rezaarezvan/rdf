import numpy as np
import sklearn.datasets as datasets
from rdp import RDP


def plot_gaussian_conditionals(ax=None, color_map=None):
    X, y = datasets.load_iris(return_X_y=True)
    petal_length = X[:, 2]

    colors = list(color_map.values())[:3]
    for i, color in enumerate(colors):
        mask = y == i
        ax.hist(X[mask, 2], color=color, alpha=0.5,
                label=datasets.load_iris().target_names[i], density=True)

        mu = np.mean(petal_length[mask])
        sigma = np.std(petal_length[mask])

        x = np.linspace(petal_length.min(), petal_length.max(), 200)
        gaussian = np.exp(-0.5 * ((x - mu) / sigma)**2) / \
            (sigma * np.sqrt(2 * np.pi))

        ax.plot(x, gaussian, color=color, linestyle='--')

    ax.set_title('Iris Petal Length Distribution with Gaussian MLE')
    ax.set_xlabel('Petal Length (cm)')
    ax.set_ylabel('Density')
    ax.legend()


if __name__ == "__main__":
    plotter = RDP()
    svg_content = plotter.create_themed_plot(plot_gaussian_conditionals)

    with open('iris_gaussian.svg', 'w') as f:
        f.write(svg_content)
