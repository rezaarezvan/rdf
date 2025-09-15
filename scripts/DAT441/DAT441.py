import numpy as np
import matplotlib.pyplot as plt


def plot_concentration(ax=None, color_map=None):
    """
    Plot a clean, blog-friendly visualization of four arms with (different) mean
    \\mu_n and their empirical (dotted) distribution and show how concentration can differ.
    """
    # Set up the plot bounds with padding
    x_min, x_max = -2, 10
    y_min, y_max = -0.1, 2
    padding = 0.2

    # Create x values for plotting the distributions
    x = np.linspace(x_min, x_max, 1000)

    # Define four arms with different means and standard deviations (concentrations)
    arms = [
        {"mean": 1, "std": 0.3, "label": r"$\mu_1$"},
        {"mean": 3, "std": 0.5, "label": r"$\mu_2$"},
        {"mean": 5.5, "std": 0.8, "label": r"$\mu_3$"},
        {"mean": 8, "std": 0.4, "label": r"$\mu_4$"},
    ]

    # Plot each arm's probability distribution
    for i, arm in enumerate(arms):
        mean = arm["mean"]
        std = arm["std"]
        label = arm["label"]

        # Calculate Gaussian probability density
        pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

        # Plot the distribution curve
        ax.plot(x, pdf, color=color_map[f"c{i + 1}"], linewidth=2.5, label=label)

        # Add a vertical line at the mean
        ax.axvline(
            x=mean + 0.5,
            color=color_map[f"c{i + 1}"],
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
        )

        # Add mean label at the bottom
        ax.text(
            mean,
            -0.10,
            label,
            fontsize=11,
            ha="center",
            va="top",
            color=color_map[f"c{i + 1}"],
            weight="bold",
        )

    # Add subtle grid
    ax.grid(True, alpha=0.1, linestyle="-", zorder=0)

    # Customize plot appearance
    ax.set_title("Concentration of Four Arms", fontsize=12, pad=15)
    ax.set_xlabel(r"$x$", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)

    # Set axis limits with padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min, y_max + padding)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add subtle ticks
    ax.tick_params(axis="both", which="major", labelsize=9)

    # Clean legend
    ax.legend(
        frameon=True,
        framealpha=0.9,
        loc="upper right",
        fontsize=9,
        bbox_to_anchor=(0.98, 0.98),
    )


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()

    svg_content = plotter.create_themed_plot(
        save_name="four_arms_concentration", plot_func=plot_concentration
    )
