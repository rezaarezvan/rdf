import numpy as np
import matplotlib.pyplot as plt


def plot_epsilon_greedy_comparison(ax=None, color_map=None):
    """
    Plot side-by-side comparison of epsilon-greedy multi-armed bandit performance:
    Left: Average reward over time
    Right: Percentage of optimal actions over time

    Shows results for epsilon = 0.1, 0.01, and 0 (greedy)
    """
    # Create figure with two subplots side by side
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])  # Average reward
    ax2 = fig.add_subplot(gs[0, 1])  # Optimal action %

    # Simulation parameters
    n_arms = 10  # Number of bandit arms
    n_steps = 1000  # Number of time steps
    n_runs = 2000  # Number of independent runs for averaging

    # Epsilon values to compare
    epsilons = [0.1, 0.01, 0.0]
    epsilon_labels = [
        r"$\varepsilon = 0.1$",
        r"$\varepsilon = 0.01$",
        r"$\varepsilon = 0$ (greedy)",
    ]
    colors = [color_map["c8"], color_map["c7"], color_map["c6"]]

    # True action values (means) for each arm - arm 0 is optimal
    np.random.seed(42)  # For reproducible results
    true_values = np.random.normal(0, 1, n_arms)
    optimal_arm = np.argmax(true_values)

    # Storage for results
    avg_rewards = np.zeros((len(epsilons), n_steps))
    optimal_actions = np.zeros((len(epsilons), n_steps))

    # Run simulations for each epsilon value
    for eps_idx, epsilon in enumerate(epsilons):
        print(f"Running simulation for epsilon = {epsilon}")

        # Arrays to accumulate results across runs
        run_rewards = np.zeros((n_runs, n_steps))
        run_optimal = np.zeros((n_runs, n_steps))

        for run in range(n_runs):
            # Initialize for this run
            np.random.seed(run)  # Different seed for each run

            # Estimated values (Q-values) and action counts
            q_values = np.zeros(n_arms)
            action_counts = np.zeros(n_arms)

            # Track rewards and optimal actions for this run
            rewards = np.zeros(n_steps)
            is_optimal = np.zeros(n_steps)

            for step in range(n_steps):
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    # Explore: choose random action
                    action = np.random.randint(n_arms)
                else:
                    # Exploit: choose greedy action (break ties randomly)
                    max_value = np.max(q_values)
                    best_actions = np.where(q_values == max_value)[0]
                    action = np.random.choice(best_actions)

                # Get reward from chosen action (normal distribution around true value)
                reward = np.random.normal(true_values[action], 1.0)

                # Update action count and Q-value (incremental average)
                action_counts[action] += 1
                q_values[action] += (reward - q_values[action]) / action_counts[action]

                # Record results
                rewards[step] = reward
                is_optimal[step] = 1.0 if action == optimal_arm else 0.0

            # Store results for this run
            run_rewards[run] = rewards
            run_optimal[run] = is_optimal

        # Average across all runs
        avg_rewards[eps_idx] = np.mean(run_rewards, axis=0)
        optimal_actions[eps_idx] = (
            np.mean(run_optimal, axis=0) * 100
        )  # Convert to percentage

    # Plot results
    steps = np.arange(1, n_steps + 1)

    # Left subplot: Average reward
    for eps_idx, (epsilon, label, color) in enumerate(
        zip(epsilons, epsilon_labels, colors)
    ):
        ax1.plot(steps, avg_rewards[eps_idx], color=color, linewidth=2, label=label)

    ax1.set_title("Average Reward", fontsize=12, pad=15)
    ax1.set_xlabel("Steps", fontsize=10)
    ax1.set_ylabel("Average\nreward", fontsize=10)
    ax1.set_xlim(1, n_steps)
    ax1.set_ylim(0, 1.5)

    # Add legend
    ax1.legend(frameon=True, framealpha=0.9, loc="lower right", fontsize=9)

    # Add subtle grid
    ax1.grid(True, alpha=0.15, linestyle="-", zorder=0)

    # Remove top and right spines
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right subplot: Optimal action percentage
    for eps_idx, (epsilon, label, color) in enumerate(
        zip(epsilons, epsilon_labels, colors)
    ):
        ax2.plot(steps, optimal_actions[eps_idx], color=color, linewidth=2, label=label)

    ax2.set_title("Optimal Action Selection", fontsize=12, pad=15)
    ax2.set_xlabel("Steps", fontsize=10)
    ax2.set_ylabel("%\nOptimal\naction", fontsize=10)
    ax2.set_xlim(1, n_steps)
    ax2.set_ylim(0, 100)

    # Add percentage ticks
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    # Add legend
    ax2.legend(frameon=True, framealpha=0.9, loc="lower right", fontsize=9)

    # Add subtle grid
    ax2.grid(True, alpha=0.15, linestyle="-", zorder=0)


if __name__ == "__main__":
    from rdf import RDF

    plotter = RDF()

    svg_content = plotter.create_themed_plot(
        name="epsilon_greedy_comparison", plot_func=plot_epsilon_greedy_comparison
    )
