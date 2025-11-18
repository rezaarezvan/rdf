import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)


def plot_logistic_sigmoid(ax=None, color_map=None):
    """
    Create a plot of the logistic sigmoid function.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    x = np.linspace(-8, 8, 400)
    y = 1 / (1 + np.exp(-x))
    ax.plot(x, y, label="Logistic Sigmoid", color="blue", linewidth=2)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("$\\sigma(x)$", fontsize=12)
    ax.set_title("Logistic Sigmoid Function", fontsize=14)
    ax.legend(fontsize=10)

    ax.grid(True, alpha=0.15)

    # Remove top and right spines (SDE style)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)

    # Plot from x = [-8, 8], y = [0, 1]
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 1)


def plot_multi_class_regression_linear_and_quadratic(ax=None, color_map=None):
    """
    Create a plot demonstrating multi-class regression with linear and quadratic models.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.4)
    ax = [fig.add_subplot(gs[0, i]) for i in range(2)]
    # Generate synthetic data
    np.random.seed(0)
    N = 100
    # Class 0 (cluster around (0, 0))
    x0 = np.random.normal(0, 0.15, N)
    y0 = np.random.normal(0, 0.15, N)
    # Class 1 (clusters around (-1/2, 1/2) and (1/2, -1/2))
    x1 = np.concatenate(
        [np.random.normal(-0.5, 0.15, N // 2), np.random.normal(0.5, 0.15, N // 2)]
    )
    y1 = np.concatenate(
        [np.random.normal(0.5, 0.15, N // 2), np.random.normal(-0.5, 0.15, N // 2)]
    )
    # Class 2 (clusters around (-1/2, -1/2) and (1/2, 1/2))
    x2 = np.concatenate(
        [np.random.normal(-0.5, 0.15, N // 2), np.random.normal(0.5, 0.15, N // 2)]
    )
    y2 = np.concatenate(
        [np.random.normal(-0.5, 0.15, N // 2), np.random.normal(0.5, 0.15, N // 2)]
    )
    # Combine data
    X = np.vstack(
        [
            np.column_stack([x0, y0]),
            np.column_stack([x1, y1]),
            np.column_stack([x2, y2]),
        ]
    )
    y = np.concatenate([np.zeros(len(x0)), np.ones(len(x1)), np.full(len(x2), 2)])
    # Create meshgrid for decision boundary
    xx, yy = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # Consistent colors: red for class 0, gray for class 1, blue for class 2
    region_colors = ["#E57373", "#BDBDBD", "#90CAF9"]

    # Linear logistic regression
    clf_linear = LogisticRegression(multi_class="multinomial", max_iter=1000)
    clf_linear.fit(X, y)
    Z_linear = clf_linear.predict(X_grid).reshape(xx.shape)
    # Quadratic logistic regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    X_grid_poly = poly.transform(X_grid)
    clf_quad = LogisticRegression(multi_class="multinomial", max_iter=1000)
    clf_quad.fit(X_poly, y)
    Z_quad = clf_quad.predict(X_grid_poly).reshape(xx.shape)

    # Plot linear model with filled regions
    ax[0].contourf(
        xx, yy, Z_linear, levels=[-0.5, 0.5, 1.5, 2.5], colors=region_colors, alpha=0.8
    )
    ax[0].scatter(
        X[y == 0, 0], X[y == 0, 1], c="red", s=20, edgecolors="k", linewidth=0.5
    )
    ax[0].scatter(
        X[y == 1, 0], X[y == 1, 1], c="black", s=20, edgecolors="k", linewidth=0.5
    )
    ax[0].scatter(
        X[y == 2, 0], X[y == 2, 1], c="blue", s=20, edgecolors="k", linewidth=0.5
    )
    ax[0].set_xlim(-1, 1)
    ax[0].set_ylim(-1, 1)
    ax[0].set_title("Linear Logistic Regression", fontsize=11)
    ax[0].set_aspect("equal")

    # Plot quadratic model with filled regions
    ax[1].contourf(
        xx, yy, Z_quad, levels=[-0.5, 0.5, 1.5, 2.5], colors=region_colors, alpha=0.8
    )
    ax[1].scatter(
        X[y == 0, 0], X[y == 0, 1], c="red", s=20, edgecolors="k", linewidth=0.5
    )
    ax[1].scatter(
        X[y == 1, 0], X[y == 1, 1], c="black", s=20, edgecolors="k", linewidth=0.5
    )
    ax[1].scatter(
        X[y == 2, 0], X[y == 2, 1], c="blue", s=20, edgecolors="k", linewidth=0.5
    )
    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(-1, 1)
    ax[1].set_title("Quadratic Logistic Regression", fontsize=11)
    ax[1].set_aspect("equal")

    ax[0].grid(True, alpha=0.15)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[0].tick_params(axis="both", which="major", labelsize=9)
    ax[1].grid(True, alpha=0.15)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].tick_params(axis="both", which="major", labelsize=9)


def plot_multi_class_lda(ax=None, color_map=None):
    """
    Create a plot demonstrating multi-class classification using LDA.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
    ax = [fig.add_subplot(gs[0, i]) for i in range(2)]

    # Generate synthetic data for 3 classes
    np.random.seed(42)

    # Class 0 (green) - top
    mean0 = [2, 4]
    cov0 = [[0.5, 0.3], [0.3, 0.5]]
    X0 = np.random.multivariate_normal(mean0, cov0, 40)

    # Class 1 (red) - middle-right
    mean1 = [4, 3]
    cov1 = [[0.6, 0.2], [0.2, 0.4]]
    X1 = np.random.multivariate_normal(mean1, cov1, 40)

    # Class 2 (blue) - bottom-left
    mean2 = [0, 0]
    cov2 = [[1.2, 0.8], [0.8, 1.2]]
    X2 = np.random.multivariate_normal(mean2, cov2, 50)

    # Combine data
    X = np.vstack([X0, X1, X2])
    y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1)), np.full(len(X2), 2)])

    # Fit LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    # Create meshgrid for decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = lda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    colors = ["green", "red", "blue"]
    markers = ["o", "s", "x"]

    # Left plot: Just the data points
    for i, (color, marker) in enumerate(zip(colors, markers)):
        mask = y == i
        ax[0].scatter(
            X[mask, 0],
            X[mask, 1],
            c=color,
            marker=marker,
            s=50,
            edgecolors="k",
            linewidth=0.5,
            alpha=0.7,
        )
    ax[0].set_xlabel("x", fontsize=11)
    ax[0].set_ylabel("y", fontsize=11)
    ax[0].set_title("Data Points", fontsize=12)
    ax[0].grid(True, alpha=0.3)
    ax[0].set_xlim(x_min, x_max)
    ax[0].set_ylim(y_min, y_max)

    # Right plot: Data + Gaussian contours + decision boundaries
    # Plot decision regions with light shading
    ax[1].contourf(
        xx,
        yy,
        Z,
        levels=[-0.5, 0.5, 1.5, 2.5],
        colors=["lightgreen", "lightcoral", "lightblue"],
        alpha=0.3,
    )

    # Plot Gaussian contours for each class
    contour_colors = ["green", "red", "blue"]
    means = [mean0, mean1, mean2]
    covs = [cov0, cov1, cov2]

    for i, (mean, cov, color) in enumerate(zip(means, covs, contour_colors)):
        # Create Gaussian distribution
        rv = multivariate_normal(mean, cov)
        # Evaluate on grid
        pos = np.dstack((xx, yy))
        density = rv.pdf(pos)
        # Plot contours
        ax[1].contour(xx, yy, density, levels=8, colors=color, linewidths=1, alpha=0.6)

    # Plot data points
    for i, (color, marker) in enumerate(zip(colors, markers)):
        mask = y == i
        ax[1].scatter(
            X[mask, 0],
            X[mask, 1],
            c=color,
            marker=marker,
            s=50,
            edgecolors="k",
            linewidth=0.5,
            alpha=0.8,
        )

    ax[1].set_xlabel("x", fontsize=11)
    ax[1].set_ylabel("y", fontsize=11)
    ax[1].set_title("LDA with Gaussian Contours", fontsize=12)
    ax[1].grid(True, alpha=0.3)
    ax[1].set_xlim(x_min, x_max)
    ax[1].set_ylim(y_min, y_max)

    # Clean up spines
    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.tick_params(axis="both", which="major", labelsize=9)

    plt.tight_layout()


def plot_multi_class_qda(ax=None, color_map=None):
    """
    Create a plot demonstrating multi-class classification using QDA.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
    ax = [fig.add_subplot(gs[0, i]) for i in range(2)]

    # Generate synthetic data for 3 classes
    np.random.seed(42)

    # Class 0 (green) - top
    mean0 = [2, 4]
    cov0 = [[0.5, 0.3], [0.3, 0.5]]
    X0 = np.random.multivariate_normal(mean0, cov0, 40)

    # Class 1 (red) - middle-right
    mean1 = [4, 3]
    cov1 = [[0.6, 0.2], [0.2, 0.4]]
    X1 = np.random.multivariate_normal(mean1, cov1, 40)

    # Class 2 (blue) - bottom-left
    mean2 = [0, 0]
    cov2 = [[1.2, 0.8], [0.8, 1.2]]
    X2 = np.random.multivariate_normal(mean2, cov2, 50)

    # Combine data
    X = np.vstack([X0, X1, X2])
    y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1)), np.full(len(X2), 2)])

    # Fit QDA
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X, y)

    # Create meshgrid for decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = qda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    colors = ["green", "red", "blue"]
    markers = ["o", "s", "x"]

    # Left plot: Just the data points
    for i, (color, marker) in enumerate(zip(colors, markers)):
        mask = y == i
        ax[0].scatter(
            X[mask, 0],
            X[mask, 1],
            c=color,
            marker=marker,
            s=50,
            edgecolors="k",
            linewidth=0.5,
            alpha=0.7,
        )
    ax[0].set_xlabel("x", fontsize=11)
    ax[0].set_ylabel("y", fontsize=11)
    ax[0].set_title("Data Points", fontsize=12)
    ax[0].grid(True, alpha=0.3)
    ax[0].set_xlim(x_min, x_max)
    ax[0].set_ylim(y_min, y_max)

    # Right plot: Data + Gaussian contours + decision boundaries
    # Plot decision regions with light shading
    ax[1].contourf(
        xx,
        yy,
        Z,
        levels=[-0.5, 0.5, 1.5, 2.5],
        colors=["lightgreen", "lightcoral", "lightblue"],
        alpha=0.3,
    )

    # Plot Gaussian contours for each class
    contour_colors = ["green", "red", "blue"]
    means = [mean0, mean1, mean2]
    covs = [cov0, cov1, cov2]

    for i, (mean, cov, color) in enumerate(zip(means, covs, contour_colors)):
        # Create Gaussian distribution
        rv = multivariate_normal(mean, cov)
        # Evaluate on grid
        pos = np.dstack((xx, yy))
        density = rv.pdf(pos)
        # Plot contours
        ax[1].contour(xx, yy, density, levels=8, colors=color, linewidths=1, alpha=0.6)

    # Plot data points
    for i, (color, marker) in enumerate(zip(colors, markers)):
        mask = y == i
        ax[1].scatter(
            X[mask, 0],
            X[mask, 1],
            c=color,
            marker=marker,
            s=50,
            edgecolors="k",
            linewidth=0.5,
            alpha=0.8,
        )

    ax[1].set_xlabel("x", fontsize=11)
    ax[1].set_ylabel("y", fontsize=11)
    ax[1].set_title("QDA with Gaussian Contours", fontsize=12)
    ax[1].grid(True, alpha=0.3)
    ax[1].set_xlim(x_min, x_max)
    ax[1].set_ylim(y_min, y_max)

    # Clean up spines
    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.tick_params(axis="both", which="major", labelsize=9)

    plt.tight_layout()


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()

    svg_content = plotter.create_themed_plot(
        save_name="logistic_sigmoid", plot_func=plot_logistic_sigmoid
    )

    svg_content = plotter.create_themed_plot(
        save_name="multi_class_regression_linear_and_quadratic",
        plot_func=plot_multi_class_regression_linear_and_quadratic,
    )

    svg_content = plotter.create_themed_plot(
        save_name="multi_class_lda", plot_func=plot_multi_class_lda
    )

    svg_content = plotter.create_themed_plot(
        save_name="multi_class_qda", plot_func=plot_multi_class_qda
    )
