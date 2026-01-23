"""
03 - n-step TD Methods

Exploring the spectrum between TD(0) and Monte Carlo using n-step methods.

Demonstrates:
- n-step TD prediction
- Effect of n on learning speed and accuracy
- Finding optimal n
- n-step returns calculation
"""

import numpy as np
from typing import Dict, List, Tuple, Deque
from collections import defaultdict, deque
import matplotlib.pyplot as plt


class GridWorld:
    """GridWorld environment for n-step TD experiments."""

    def __init__(self, size: int = 5, gamma: float = 0.9):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma
        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left

        self.start_state = size * (size - 1)  # Bottom-left
        self.goal_state = size - 1  # Top-right

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        """Execute action."""
        if state == self.goal_state:
            return state, 0.0, True

        row, col = self._state_to_coord(state)
        action_effects = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        dr, dc = action_effects[action]
        new_row = max(0, min(self.size - 1, row + dr))
        new_col = max(0, min(self.size - 1, col + dc))
        next_state = self._coord_to_state(new_row, new_col)

        if next_state == self.goal_state:
            return next_state, 1.0, True
        else:
            return next_state, -0.01, False

    def reset(self) -> int:
        return self.start_state


def create_policy(env: GridWorld) -> np.ndarray:
    """Create a policy that moves toward the goal."""
    policy = np.zeros((env.n_states, len(env.actions)))

    for state in range(env.n_states):
        if state == env.goal_state:
            policy[state] = 1.0 / len(env.actions)
            continue

        row, col = env._state_to_coord(state)
        goal_row, goal_col = env._state_to_coord(env.goal_state)

        weights = np.ones(len(env.actions)) * 0.1

        if row > goal_row:
            weights[0] = 0.5
        elif row < goal_row:
            weights[2] = 0.5

        if col < goal_col:
            weights[1] = 0.5
        elif col > goal_col:
            weights[3] = 0.5

        policy[state] = weights / weights.sum()

    return policy


def n_step_td_prediction(env: GridWorld, policy: np.ndarray, n: int,
                         n_episodes: int = 300, alpha: float = 0.1) -> Tuple[Dict, List]:
    """
    n-step TD Prediction.

    Args:
        env: Environment
        policy: Policy to evaluate
        n: Number of steps to look ahead (1 = TD(0), inf = MC)
        n_episodes: Number of episodes
        alpha: Learning rate

    Returns:
        V: Value function
        value_history: Average values per episode
    """
    V = defaultdict(float)
    value_history = []

    for episode in range(n_episodes):
        # Store episode trajectory
        states = []
        rewards = []

        state = env.reset()
        states.append(state)
        T = float('inf')  # Terminal time
        t = 0

        while True:
            if t < T:
                # Take action
                action = np.random.choice(env.actions, p=policy[state])
                next_state, reward, done = env.step(state, action)

                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = t + 1

                state = next_state

            # Update state tau (the state being updated)
            tau = t - n + 1

            if tau >= 0:
                # Calculate n-step return
                G = 0.0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += (env.gamma ** (i - tau - 1)) * rewards[i - 1]

                # Add bootstrap value if not terminal
                if tau + n < T:
                    G += (env.gamma ** n) * V[states[tau + n]]

                # Update value
                V[states[tau]] += alpha * (G - V[states[tau]])

            t += 1

            if tau == T - 1:
                break

        # Track average value
        avg_value = np.mean([V[s] for s in range(env.n_states)])
        value_history.append(avg_value)

    return dict(V), value_history


def compare_n_values(env: GridWorld, policy: np.ndarray, n_runs: int = 10):
    """Compare different values of n."""
    print("\n" + "="*60)
    print("COMPARING DIFFERENT n VALUES")
    print("="*60)

    n_values = [1, 2, 3, 5, 10, 20, 50]
    n_episodes = 200
    alpha = 0.1

    all_histories = {n: [] for n in n_values}

    print(f"\nRunning {n_runs} trials for each n...")
    for n in n_values:
        print(f"  n = {n}")
        for run in range(n_runs):
            _, history = n_step_td_prediction(env, policy, n, n_episodes, alpha)
            all_histories[n].append(history)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('n-step TD Learning: Effect of n', fontsize=16)

    # Plot 1: All learning curves
    ax = axes[0, 0]
    for n in n_values:
        histories = np.array(all_histories[n])
        mean_hist = histories.mean(axis=0)
        ax.plot(mean_hist, label=f'n={n}', linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Value', fontsize=12)
    ax.set_title('Learning Curves for Different n', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 2: Selected n values with error bands
    ax = axes[0, 1]
    selected_n = [1, 5, 20]
    for n in selected_n:
        histories = np.array(all_histories[n])
        mean_hist = histories.mean(axis=0)
        std_hist = histories.std(axis=0)
        ax.plot(mean_hist, label=f'n={n}', linewidth=2)
        ax.fill_between(range(n_episodes),
                        mean_hist - std_hist,
                        mean_hist + std_hist,
                        alpha=0.3)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Value', fontsize=12)
    ax.set_title('Selected n Values (mean ± std)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 3: Convergence speed
    ax = axes[1, 0]
    convergence_speeds = []
    for n in n_values:
        histories = np.array(all_histories[n])
        mean_hist = histories.mean(axis=0)
        final_value = mean_hist[-20:].mean()
        threshold = final_value * 0.9

        converged_at = np.argmax(mean_hist >= threshold)
        if mean_hist[converged_at] < threshold:
            converged_at = n_episodes

        convergence_speeds.append(converged_at)

    ax.bar(range(len(n_values)), convergence_speeds, color='steelblue')
    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels([f'n={n}' for n in n_values])
    ax.set_xlabel('n value', fontsize=12)
    ax.set_ylabel('Episodes to 90% Convergence', fontsize=12)
    ax.set_title('Convergence Speed', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Final variance
    ax = axes[1, 1]
    final_variances = []
    for n in n_values:
        histories = np.array(all_histories[n])
        final_values = histories[:, -20:].mean(axis=1)
        final_variances.append(final_values.var())

    ax.bar(range(len(n_values)), final_variances, color='coral')
    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels([f'n={n}' for n in n_values])
    ax.set_xlabel('n value', fontsize=12)
    ax.set_ylabel('Variance of Final Values', fontsize=12)
    ax.set_title('Final Value Variance Across Runs', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/n_step_comparison.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved n-step comparison to n_step_comparison.png")
    plt.close()

    # Print statistics
    print("\nConvergence Speed (episodes to 90% of final value):")
    for n, speed in zip(n_values, convergence_speeds):
        print(f"  n={n:3d}: {speed:3d} episodes")

    print("\nFinal Value Variance:")
    for n, var in zip(n_values, final_variances):
        print(f"  n={n:3d}: {var:.6f}")


def visualize_n_step_returns(env: GridWorld, policy: np.ndarray):
    """Visualize how n-step returns are calculated."""
    print("\n" + "="*60)
    print("VISUALIZING n-STEP RETURNS")
    print("="*60)

    # Generate a single episode
    episode_states = []
    episode_rewards = []
    state = env.reset()
    done = False

    while not done and len(episode_states) < 20:
        episode_states.append(state)
        action = np.random.choice(env.actions, p=policy[state])
        next_state, reward, done = env.step(state, action)
        episode_rewards.append(reward)
        state = next_state

    # Calculate n-step returns for different n
    n_values = [1, 2, 5, len(episode_rewards)]
    V = defaultdict(float)  # Use zeros for simplicity

    fig, axes = plt.subplots(len(n_values), 1, figsize=(12, 10))
    fig.suptitle('n-step Returns Calculation Example', fontsize=16)

    for idx, n in enumerate(n_values):
        ax = axes[idx]
        returns = []

        for t in range(len(episode_states)):
            # Calculate n-step return from time t
            G = 0
            for k in range(min(n, len(episode_rewards) - t)):
                G += (env.gamma ** k) * episode_rewards[t + k]

            # Bootstrap if not terminal
            if t + n < len(episode_states):
                G += (env.gamma ** n) * V[episode_states[t + n]]

            returns.append(G)

        ax.plot(returns, marker='o', linewidth=2, markersize=6)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_ylabel('Return', fontsize=11)
        ax.set_title(f'n={n} ({"MC" if n == len(episode_rewards) else "TD"})',
                    fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Step', fontsize=12)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/n_step_returns.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved n-step returns visualization to n_step_returns.png")
    plt.close()


def find_optimal_n(env: GridWorld, policy: np.ndarray):
    """Find the optimal n for this problem."""
    print("\n" + "="*60)
    print("FINDING OPTIMAL n")
    print("="*60)

    n_values = list(range(1, 31))
    n_episodes = 200
    n_runs = 15
    alpha = 0.1

    performance = []

    print(f"\nTesting n from 1 to {max(n_values)}...")
    for n in n_values:
        if n % 5 == 1:
            print(f"  Testing n={n}...")

        episode_counts = []
        for run in range(n_runs):
            _, history = n_step_td_prediction(env, policy, n, n_episodes, alpha)
            # Count episodes to reach 90% of final value
            final_val = np.mean(history[-10:])
            threshold = final_val * 0.9
            converged_at = np.argmax(np.array(history) >= threshold)
            if history[converged_at] < threshold:
                converged_at = n_episodes
            episode_counts.append(converged_at)

        avg_episodes = np.mean(episode_counts)
        performance.append(avg_episodes)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(n_values, performance, marker='o', linewidth=2, markersize=6)

    optimal_n = n_values[np.argmin(performance)]
    optimal_episodes = min(performance)

    plt.axvline(x=optimal_n, color='r', linestyle='--', linewidth=2,
                label=f'Optimal n={optimal_n}')
    plt.scatter([optimal_n], [optimal_episodes], color='r', s=200, zorder=5,
                marker='*', edgecolors='darkred', linewidths=2)

    plt.xlabel('n (number of steps)', fontsize=13)
    plt.ylabel('Episodes to Convergence', fontsize=13)
    plt.title('Finding Optimal n for n-step TD', fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/optimal_n.png',
                dpi=150, bbox_inches='tight')
    print(f"\nOptimal n = {optimal_n} (converges in {optimal_episodes:.1f} episodes)")
    print("Saved optimal n plot to optimal_n.png")
    plt.close()


def main():
    """Main demonstration."""
    print("="*60)
    print("n-STEP TD METHODS DEMONSTRATION")
    print("="*60)

    # Create environment
    env = GridWorld(size=5, gamma=0.9)
    policy = create_policy(env)

    # Run experiments
    compare_n_values(env, policy, n_runs=10)
    visualize_n_step_returns(env, policy)
    find_optimal_n(env, policy)

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Insights:")
    print("  - Small n (1-3): Fast updates, higher bias")
    print("  - Medium n (5-10): Often optimal balance")
    print("  - Large n (>20): Lower bias, higher variance")
    print("  - n=∞: Equivalent to Monte Carlo")


if __name__ == "__main__":
    main()
