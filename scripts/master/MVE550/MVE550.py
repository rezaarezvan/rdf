import numpy as np

from rdf import figure


@figure("stochastic_vs_deterministic_processes", height=3.2)
def plot_stochastic_vs_deterministic_processes(fig, color_map):
    ax1, ax2 = fig.subplots(1, 2)
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 100)
    for i in range(6):
        ax1.plot(t, np.exp(0.2 * t) + rng.normal(0, 0.3, t.size), color=color_map["c1"], lw=0.8, alpha=0.6, label="Stochastic" if i == 0 else None)
    ax1.plot(t, np.exp(0.2 * t), color=color_map["c8"], lw=1.8, label="Deterministic")
    ax1.set(xlabel="Time", ylabel="Population")
    ax1.legend(loc="upper left")
    price = 100 + np.cumsum(np.r_[0, np.random.default_rng(1).normal(0, 1, 99)])
    ax2.plot(price, color=color_map["c1"], lw=1, label="Stochastic")
    ax2.set(xlabel="Time", ylabel="Stock price")
    ax2.legend(loc="lower left")
