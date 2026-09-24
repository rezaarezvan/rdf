import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from rdf import figure

CLASSES = ("c1", "c7", "c8")
LEVELS = [-0.5, 0.5, 1.5, 2.5]


@figure("logistic_sigmoid", height=3.0)
def plot_logistic_sigmoid(ax, color_map):
    x = np.linspace(-8, 8, 400)
    ax.plot(x, 1 / (1 + np.exp(-x)), color=color_map["c8"])
    ax.set(xlim=(-8, 8), ylim=(0, 1), xlabel="$x$", ylabel=r"$\sigma(x)$")


def _regions(ax, color_map, xx, yy, z, colors, alpha=0.25):
    ax.contourf(xx, yy, z, levels=LEVELS, colors=[color_map[c] for c in colors], alpha=alpha)
    ax.contour(xx, yy, z, levels=LEVELS[1:-1], colors=[color_map["black"]], linewidths=0.6)


def _scatter(ax, color_map, X, y, colors, markers=("o", "o", "o")):
    for k, (c, m) in enumerate(zip(colors, markers)):
        ax.scatter(*X[y == k].T, color=color_map[c], marker=m, s=14)


@figure("multi_class_regression_linear_and_quadratic", height=3.2)
def plot_multi_class_regression_linear_and_quadratic(fig, color_map):
    rng = np.random.default_rng(0)
    n, s = 100, 0.15
    blob = lambda centres: np.vstack([rng.normal(c, s, (n // len(centres), 2)) for c in centres])
    X = np.vstack([blob([(0, 0)]), blob([(-0.5, 0.5), (0.5, -0.5)]), blob([(-0.5, -0.5), (0.5, 0.5)])])
    y = np.repeat([0, 1, 2], n)
    xx, yy = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    poly = PolynomialFeatures(2)
    models = (
        (LogisticRegression(max_iter=1000).fit(X, y).predict(grid), "Linear"),
        (LogisticRegression(max_iter=1000).fit(poly.fit_transform(X), y).predict(poly.transform(grid)), "Quadratic"),
    )
    for ax, (z, title) in zip(fig.subplots(1, 2), models):
        _regions(ax, color_map, xx, yy, z.reshape(xx.shape), CLASSES)
        _scatter(ax, color_map, X, y, CLASSES)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", xticks=[-1, 0, 1], yticks=[-1, 0, 1])
        ax.set_title(title, fontsize=11)


def _discriminant(fig, color_map, model):
    rng = np.random.default_rng(42)
    means = ([2, 4], [4, 3], [0, 0])
    covs = ([[0.5, 0.3], [0.3, 0.5]], [[0.6, 0.2], [0.2, 0.4]], [[1.2, 0.8], [0.8, 1.2]])
    X = np.vstack([rng.multivariate_normal(m, c, k) for m, c, k in zip(means, covs, (40, 40, 50))])
    y = np.repeat([0, 1, 2], (40, 40, 50))
    colors, markers = ("c2", "c1", "c8"), ("o", "s", "x")
    lo, hi = X.min(0) - 1, X.max(0) + 1
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
    z = model.fit(X, y).predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax1, ax2 = fig.subplots(1, 2)
    _regions(ax2, color_map, xx, yy, z, colors, alpha=0.15)
    for m, c, col in zip(means, covs, colors):
        ax2.contour(xx, yy, mvn(m, c).pdf(np.dstack((xx, yy))), levels=5, colors=[color_map[col]], linewidths=0.8)
    for ax in (ax1, ax2):
        _scatter(ax, color_map, X, y, colors, markers)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel="$x_1$", ylabel="$x_2$")


@figure("multi_class_lda", height=3.2)
def plot_multi_class_lda(fig, color_map):
    _discriminant(fig, color_map, LinearDiscriminantAnalysis())


@figure("multi_class_qda", height=3.2)
def plot_multi_class_qda(fig, color_map):
    _discriminant(fig, color_map, QuadraticDiscriminantAnalysis())
