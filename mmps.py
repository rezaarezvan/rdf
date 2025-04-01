import numpy as np
import matplotlib.pyplot as plt


def plot_soft_margin(ax=None, color_map=None):
    """
    Create a clean, blog-friendly visualization of the soft margin principle
    for Support Vector Machines (SVM), showing cases where perfect separation
    is not possible.
    """
    # Generate synthetic data
    np.random.seed(42)
    n_points = 12

    # Class 1 (bottom-right)
    # Core points
    x1 = np.random.uniform(-2.5, 0, n_points)
    y1 = -3 + np.random.normal(0, 0.8, n_points)
    additional_x1 = np.random.uniform(-5, -2.5, n_points//2)
    additional_y1 = np.random.uniform(-5, -2, n_points//2)
    x1 = np.concatenate([x1, additional_x1])
    y1 = np.concatenate([y1, additional_y1])

    # Add outliers for Class 1
    outlier_x1 = np.array([-1, 1, -2])
    outlier_y1 = np.array([-1, 0, 1])
    x1 = np.concatenate([x1, outlier_x1])
    y1 = np.concatenate([y1, outlier_y1])

    # Class 2 (top-left)
    # Core points
    x2 = np.random.uniform(2.5, 5, n_points)
    y2 = np.random.uniform(2, 5, n_points)
    additional_x2 = np.random.uniform(5, 7.5, n_points//2)
    additional_y2 = np.random.uniform(4, 7, n_points//2)
    x2 = np.concatenate([x2, additional_x2])
    y2 = np.concatenate([y2, additional_y2])

    # Add outliers for Class 2
    outlier_x2 = np.array([0, -1.5, 1.5])
    outlier_y2 = np.array([2, 0, -1])
    x2 = np.concatenate([x2, outlier_x2])
    y2 = np.concatenate([y2, outlier_y2])

    # Plot data points
    ax.scatter(x1, y1, c=color_map['c1'], s=50, alpha=0.7,
               label='Class 1', edgecolor='white', linewidth=0.5, zorder=3)
    ax.scatter(x2, y2, c=color_map['c2'], s=50, alpha=0.7,
               label='Class 2', edgecolor='white', linewidth=0.5, zorder=3)

    # Create decision boundary and margins
    x_range = np.array([-10, 10])

    # Decision boundary: y = -x (negative slope)
    slope = -1
    intercept = 1
    boundary = slope * x_range + intercept

    # Margins: parallel lines
    margin_width = 2
    margin_up = slope * x_range + \
        (intercept + margin_width * np.sqrt(1 + slope**2))
    margin_down = slope * x_range + \
        (intercept - margin_width * np.sqrt(1 + slope**2))

    # Plot decision boundary and margins
    ax.plot(x_range, boundary, '-', color='black', linewidth=2,
            label='Decision Boundary', zorder=2)
    ax.plot(x_range, margin_up, '--', color='gray', linewidth=1.5,
            label='Margin', zorder=2, alpha=0.7)
    ax.plot(x_range, margin_down, '--', color='gray', linewidth=1.5,
            zorder=2, alpha=0.7)

    # Fill the regions
    ax.fill_between(x_range, margin_down, -10,
                    color=color_map['c1'], alpha=0.1)
    ax.fill_between(x_range, margin_up, 10, color=color_map['c2'], alpha=0.1)

    # Add margin violation arrows for a few points
    violations = [
        # (x, y, dx, dy) for arrow coordinates
        (-1, -1, 0.5, 1),
        (1.5, -1, -0.5, 1),
        (-1.5, 0, 0.5, -1)
    ]

    for x, y, dx, dy in violations:
        ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->',
                                    color='red',
                                    alpha=0.6,
                                    linewidth=1))

    # Customize plot appearance
    ax.set_title('Soft Margin Case:\nNon-Separable Data', fontsize=12, pad=15)
    ax.set_xlabel(r'$x_1$', fontsize=10)
    ax.set_ylabel(r'$x_2$', fontsize=10)

    # Clean legend
    ax.legend(frameon=True, framealpha=0.9, loc='upper right',
              fontsize=9)

    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-7.5, 7.5)

    # Subtle grid
    ax.grid(True, alpha=0.15, linestyle='-', zorder=1)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add subtle ticks
    ax.tick_params(axis='both', which='major', labelsize=9)


if __name__ == "__main__":
    from rdp import RDP

    plotter = RDP()
    svg_content = plotter.create_themed_plot(plot_soft_margin)

    with open('soft_margin_principle.svg', 'w') as f:
        f.write(svg_content)
