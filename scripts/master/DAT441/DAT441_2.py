import numpy as np

from rdf import figure


def _bandit(eps, arms=10, steps=1000, runs=2000, seed=0):
    rng = np.random.default_rng(seed)
    q_true = rng.normal(0, 1, (runs, arms))
    best = q_true.argmax(1)
    q, n = np.zeros((runs, arms)), np.zeros((runs, arms))
    idx = np.arange(runs)
    reward, optimal = np.zeros(steps), np.zeros(steps)
    for t in range(steps):
        greedy = (q + 1e-9 * rng.random((runs, arms))).argmax(1)
        a = np.where(rng.random(runs) < eps, rng.integers(0, arms, runs), greedy)
        r = rng.normal(q_true[idx, a], 1)
        n[idx, a] += 1
        q[idx, a] += (r - q[idx, a]) / n[idx, a]
        reward[t], optimal[t] = r.mean(), (a == best).mean() * 100
    return reward, optimal


@figure("epsilon_greedy_comparison", height=3.4)
def plot_epsilon_greedy_comparison(fig, color_map):
    ax1, ax2 = fig.subplots(1, 2)
    steps = np.arange(1, 1001)
    for eps, c, label in ((0.1, "c8", r"$\varepsilon = 0.1$"), (0.01, "c1", r"$\varepsilon = 0.01$"), (0, "c2", r"$\varepsilon = 0$ (greedy)")):
        reward, optimal = _bandit(eps)
        ax1.plot(steps, reward, color=color_map[c], lw=1, label=label)
        ax2.plot(steps, optimal, color=color_map[c], lw=1, label=label)
    ax1.set(xlabel="Steps", ylabel="Average reward", xlim=(1, 1000))
    ax2.set(xlabel="Steps", ylabel=r"Optimal action (%)", xlim=(1, 1000), ylim=(0, 100))
    ax1.legend(loc="lower right")
