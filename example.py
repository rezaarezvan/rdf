import numpy as np
from sklearn.datasets import load_iris


def plot_linear_separation(ax=None, color_map=None):
    """
    Create a clean, blog-friendly visualization of linearly separable Iris classes
    using petal measurements. Shows how two Iris species can be separated by a
    simple linear boundary.

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Select setosa and versicolor (first two classes)
    mask = y < 2
    X = X[mask]
    y = y[mask]

    # Get petal length and width
    petal_length = X[:, 2]
    petal_width = X[:, 3]

    # Add padding to axis limits
    x_min, x_max = petal_length.min(), petal_length.max()
    y_min, y_max = petal_width.min(), petal_width.max()
    padding = 0.1

    # Plot the two classes with larger, more visible points
    for i, species in enumerate(iris.target_names[:2]):
        mask = y == i
        ax.scatter(petal_length[mask], petal_width[mask],
                   c=color_map[f'c{i+1}'], s=70, alpha=0.7,
                   label=species, edgecolor='white', linewidth=0.5,
                   zorder=3)

    # Draw separating line with better positioning
    margin = 0.2
    x_line = np.array([1, 5])
    y_line = -0.42 * x_line - 0.35  # Adjusted for better separation

    # Plot decision boundary
    ax.plot(y_line, y_line, '--', color='black',
            linewidth=2, label='Decision Boundary',
            zorder=2)

    # Customize plot appearance
    ax.set_title('Linear Separation of Iris Classes\nusing Petal Measurements',
                 fontsize=12, pad=15)
    ax.set_xlabel('Petal Length (cm)', fontsize=10)
    ax.set_ylabel('Petal Width (cm)', fontsize=10)

    # Clean legend with better positioning
    ax.legend(frameon=True, framealpha=0.9, loc='upper left',
              fontsize=9, bbox_to_anchor=(0.02, 0.98))

    # Subtle grid
    ax.grid(True, alpha=0.15, linestyle='-', zorder=1)

    # Set consistent limits with padding
    x_margin = (x_max - x_min) * padding
    y_margin = (y_max - y_min) * padding
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add subtle ticks
    ax.tick_params(axis='both', which='major', labelsize=9)


if __name__ == "__main__":
    from rdp import RDP

    plotter = RDP()
    svg_content = plotter.create_themed_plot(plot_linear_separation)

    with open('iris_linear_separation.svg', 'w') as f:
        f.write(svg_content)
