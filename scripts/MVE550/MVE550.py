import numpy as np
import matplotlib.pyplot as plt


def plot_stochastic_VS_deterministic_processes(ax=None, color_map=None):
    """
    Create a visualization comparing stochastic and deterministic processes.

    In this case we consider two examples (side-by-side):

        1. Bacterial growth (exponential, one black line for deterministic, other (gray) for stochastic)
            1.1 Consider the case dy/dx = 0.2y (deterministic)
        2. Stock price evolution (random walk for stochastic, smooth curve for deterministic)

    Args:
        ax: Matplotlib axis object to plot on
        color_map: Dictionary of colors for consistent styling
    """
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.4)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Bacterial growth
    ax2 = fig.add_subplot(gs[0, 1])  # Stock price evolution

    # Plot bacterial growth
    t = np.linspace(0, 10, 100)
    y_deterministic = np.exp(0.2 * t)
    ax1.plot(t, y_deterministic, color="black", linewidth=2, label="Deterministic")

    # Stochastic (start from same initial condition, add noise), 6 realizations in total
    np.random.seed(0)
    for _ in range(6):
        noise = np.random.normal(0, 0.1, size=t.shape)
        y_stochastic = np.exp(0.2 * t) + noise
        ax1.plot(
            t,
            y_stochastic,
            color="gray",
            linewidth=1,
            alpha=0.7,
            label="Stochastic" if _ == 0 else "",
        )

    ax1.set_title("Bacterial Growth")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Population")
    ax1.legend()

    # Stochastic stock price (random walk), only stochastic realization
    time_steps = 100
    stock_price_stochastic = [100]  # Initial stock price
    np.random.seed(1)
    for _ in range(1, time_steps):
        change = np.random.normal(0, 1)  # Random change
        new_price = stock_price_stochastic[-1] + change
        stock_price_stochastic.append(new_price)
    stock_price_stochastic = np.array(stock_price_stochastic)
    ax2.plot(
        np.arange(time_steps),
        stock_price_stochastic,
        color="black",
        linewidth=1.5,
        label="Stochastic",
    )

    ax2.set_title("Stock Price Evolution")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Stock Price")
    ax2.legend()


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()
    svg_content = plotter.create_themed_plot(
        name="stochastic_vs_deterministic_processes",
        plot_func=plot_stochastic_VS_deterministic_processes,
    )
