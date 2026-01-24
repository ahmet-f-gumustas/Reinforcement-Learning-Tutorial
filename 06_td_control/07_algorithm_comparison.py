"""
07 - Comprehensive Algorithm Comparison

Compare all TD control algorithms across multiple environments and metrics.

Demonstrates:
- SARSA, Q-Learning, Expected SARSA, Double Q-Learning
- Multiple test environments
- Performance metrics dashboard
- Statistical analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Callable
from collections import defaultdict
import matplotlib.pyplot as plt


# =============================================================================
# ENVIRONMENTS
# =============================================================================

class GridWorld:
    """Standard GridWorld environment."""

    def __init__(self, size: int = 6, gamma: float = 0.95):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma
        self.n_actions = 4
        self.actions = list(range(4))

        self.start_state = size * (size - 1)
        self.goal_state = size - 1

        self.obstacles = {
            self._coord_to_state(2, 2),
            self._coord_to_state(2, 3),
            self._coord_to_state(3, 2),
        }

        self.action_vectors = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def reset(self) -> int:
        return self.start_state

    def step(self, action: int, state: int = None) -> Tuple[int, float, bool]:
        if state is None:
            state = self.start_state

        if state == self.goal_state:
            return state, 0.0, True

        row, col = self._state_to_coord(state)
        dr, dc = self.action_vectors[action]

        new_row = max(0, min(self.size - 1, row + dr))
        new_col = max(0, min(self.size - 1, col + dc))
        next_state = self._coord_to_state(new_row, new_col)

        if next_state == self.goal_state:
            return next_state, 10.0, True
        elif next_state in self.obstacles:
            return next_state, -5.0, False
        else:
            return next_state, -0.1, False


class CliffWalking:
    """Cliff Walking environment."""

    def __init__(self):
        self.height = 4
        self.width = 12
        self.n_states = self.height * self.width
        self.gamma = 1.0
        self.n_actions = 4
        self.actions = list(range(4))

        self.start_state = 36  # Bottom-left
        self.goal_state = 47  # Bottom-right

        self.cliff = set(range(37, 47))  # Bottom row, columns 1-10
        self.action_vectors = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.state = self.start_state

    def reset(self) -> int:
        self.state = self.start_state
        return self.state

    def step(self, action: int, state: int = None) -> Tuple[int, float, bool]:
        if state is None:
            state = self.state

        row, col = state // 12, state % 12
        dr, dc = self.action_vectors[action]

        new_row = max(0, min(3, row + dr))
        new_col = max(0, min(11, col + dc))
        next_state = new_row * 12 + new_col

        if next_state in self.cliff:
            self.state = self.start_state
            return self.start_state, -100.0, False

        if next_state == self.goal_state:
            self.state = next_state
            return next_state, -1.0, True

        self.state = next_state
        return next_state, -1.0, False


class StochasticGridWorld:
    """GridWorld with stochastic transitions."""

    def __init__(self, size: int = 5, gamma: float = 0.95, slip_prob: float = 0.1):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma
        self.n_actions = 4
        self.actions = list(range(4))
        self.slip_prob = slip_prob

        self.start_state = size * (size - 1)
        self.goal_state = size - 1

        self.action_vectors = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def reset(self) -> int:
        return self.start_state

    def step(self, action: int, state: int = None) -> Tuple[int, float, bool]:
        if state is None:
            state = self.start_state

        if state == self.goal_state:
            return state, 0.0, True

        # Slip to random action
        if np.random.random() < self.slip_prob:
            action = np.random.choice(self.actions)

        row, col = self._state_to_coord(state)
        dr, dc = self.action_vectors[action]

        new_row = max(0, min(self.size - 1, row + dr))
        new_col = max(0, min(self.size - 1, col + dc))
        next_state = self._coord_to_state(new_row, new_col)

        if next_state == self.goal_state:
            return next_state, 10.0, True
        else:
            return next_state, -0.1, False


# =============================================================================
# ALGORITHMS
# =============================================================================

def epsilon_greedy(Q: Dict, state: int, n_actions: int, epsilon: float) -> int:
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    q_values = [Q.get((state, a), 0.0) for a in range(n_actions)]
    max_q = max(q_values)
    best = [a for a in range(n_actions) if q_values[a] == max_q]
    return np.random.choice(best)


def sarsa(env, n_episodes: int, alpha: float = 0.1, epsilon: float = 0.1) -> Tuple[Dict, List]:
    Q = defaultdict(float)
    rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, env.n_actions, epsilon)
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            next_state, reward, done = env.step(action, state)
            total_reward += reward

            next_action = epsilon_greedy(Q, next_state, env.n_actions, epsilon)
            Q[(state, action)] += alpha * (reward + env.gamma * Q[(next_state, next_action)] - Q[(state, action)])

            state = next_state
            action = next_action
            steps += 1

        rewards.append(total_reward)

    return dict(Q), rewards


def q_learning(env, n_episodes: int, alpha: float = 0.1, epsilon: float = 0.1) -> Tuple[Dict, List]:
    Q = defaultdict(float)
    rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            action = epsilon_greedy(Q, state, env.n_actions, epsilon)
            next_state, reward, done = env.step(action, state)
            total_reward += reward

            max_next_q = max([Q[(next_state, a)] for a in range(env.n_actions)])
            Q[(state, action)] += alpha * (reward + env.gamma * max_next_q - Q[(state, action)])

            state = next_state
            steps += 1

        rewards.append(total_reward)

    return dict(Q), rewards


def expected_sarsa(env, n_episodes: int, alpha: float = 0.1, epsilon: float = 0.1) -> Tuple[Dict, List]:
    Q = defaultdict(float)
    rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            action = epsilon_greedy(Q, state, env.n_actions, epsilon)
            next_state, reward, done = env.step(action, state)
            total_reward += reward

            # Expected value
            q_values = [Q[(next_state, a)] for a in range(env.n_actions)]
            max_q = max(q_values)
            best = [a for a in range(env.n_actions) if q_values[a] == max_q]

            expected_q = 0
            for a in range(env.n_actions):
                if a in best:
                    prob = (1 - epsilon) / len(best) + epsilon / env.n_actions
                else:
                    prob = epsilon / env.n_actions
                expected_q += prob * q_values[a]

            Q[(state, action)] += alpha * (reward + env.gamma * expected_q - Q[(state, action)])

            state = next_state
            steps += 1

        rewards.append(total_reward)

    return dict(Q), rewards


def double_q_learning(env, n_episodes: int, alpha: float = 0.1, epsilon: float = 0.1) -> Tuple[Dict, List]:
    Q1 = defaultdict(float)
    Q2 = defaultdict(float)
    rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            # Use sum for action selection
            q_sum = [Q1[(state, a)] + Q2[(state, a)] for a in range(env.n_actions)]
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                max_q = max(q_sum)
                best = [a for a in range(env.n_actions) if q_sum[a] == max_q]
                action = np.random.choice(best)

            next_state, reward, done = env.step(action, state)
            total_reward += reward

            if np.random.random() < 0.5:
                q1_values = [Q1[(next_state, a)] for a in range(env.n_actions)]
                best_action = np.argmax(q1_values)
                Q1[(state, action)] += alpha * (reward + env.gamma * Q2[(next_state, best_action)] - Q1[(state, action)])
            else:
                q2_values = [Q2[(next_state, a)] for a in range(env.n_actions)]
                best_action = np.argmax(q2_values)
                Q2[(state, action)] += alpha * (reward + env.gamma * Q1[(next_state, best_action)] - Q2[(state, action)])

            state = next_state
            steps += 1

        rewards.append(total_reward)

    return {k: Q1[k] + Q2[k] for k in set(Q1) | set(Q2)}, rewards


# =============================================================================
# COMPARISON
# =============================================================================

def run_comparison(env_name: str, env, n_episodes: int = 500, n_runs: int = 20):
    """Run all algorithms on an environment."""
    algorithms = [
        ('SARSA', sarsa),
        ('Q-Learning', q_learning),
        ('Expected SARSA', expected_sarsa),
        ('Double Q', double_q_learning),
    ]

    results = {name: [] for name, _ in algorithms}

    print(f"\n{env_name}: Running {n_runs} trials...")
    for run in range(n_runs):
        if run % 5 == 0:
            print(f"  Run {run + 1}/{n_runs}")

        for name, algo in algorithms:
            _, rewards = algo(env, n_episodes)
            results[name].append(rewards)

    return {name: np.array(r) for name, r in results.items()}


def create_comparison_dashboard(all_results: Dict):
    """Create comprehensive comparison dashboard."""
    print("\n" + "="*60)
    print("CREATING COMPARISON DASHBOARD")
    print("="*60)

    envs = list(all_results.keys())
    algorithms = list(all_results[envs[0]].keys())
    n_envs = len(envs)

    fig, axes = plt.subplots(n_envs, 3, figsize=(18, 5 * n_envs))
    if n_envs == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('TD Control Algorithms: Comprehensive Comparison', fontsize=18, fontweight='bold')

    colors = {'SARSA': 'blue', 'Q-Learning': 'orange', 'Expected SARSA': 'green', 'Double Q': 'red'}

    for env_idx, env_name in enumerate(envs):
        results = all_results[env_name]

        # Plot 1: Learning curves
        ax = axes[env_idx, 0]
        window = 20
        for algo_name, rewards in results.items():
            mean = rewards.mean(axis=0)
            smoothed = np.convolve(mean, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=algo_name, linewidth=2, color=colors[algo_name])

        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Total Reward', fontsize=11)
        ax.set_title(f'{env_name}: Learning Curves', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 2: Final performance boxplot
        ax = axes[env_idx, 1]
        final_data = []
        labels = []
        for algo_name in algorithms:
            final_rewards = results[algo_name][:, -50:].mean(axis=1)
            final_data.append(final_rewards)
            labels.append(algo_name)

        bp = ax.boxplot(final_data, labels=labels)
        ax.set_ylabel('Avg Reward (last 50)', fontsize=11)
        ax.set_title(f'{env_name}: Final Performance', fontsize=13)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Variance comparison
        ax = axes[env_idx, 2]
        variance_data = []
        for algo_name in algorithms:
            variance = results[algo_name].var(axis=0)
            smoothed_var = np.convolve(variance, np.ones(window)/window, mode='valid')
            ax.plot(smoothed_var, label=algo_name, linewidth=2, color=colors[algo_name])

        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Variance', fontsize=11)
        ax.set_title(f'{env_name}: Learning Variance', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/06_td_control/algorithm_comparison_dashboard.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved dashboard to algorithm_comparison_dashboard.png")
    plt.close()


def print_summary_statistics(all_results: Dict):
    """Print summary statistics for all experiments."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for env_name, results in all_results.items():
        print(f"\n{env_name}:")
        print("-" * 50)
        print(f"{'Algorithm':<18} {'Mean ± Std':<15} {'Best Run':<10} {'Worst Run':<10}")
        print("-" * 50)

        for algo_name, rewards in results.items():
            final = rewards[:, -50:].mean(axis=1)
            print(f"{algo_name:<18} {final.mean():>6.1f} ± {final.std():<6.1f} {final.max():>8.1f} {final.min():>10.1f}")


def create_summary_bar_chart(all_results: Dict):
    """Create summary bar chart comparing final performance."""
    envs = list(all_results.keys())
    algorithms = list(all_results[envs[0]].keys())

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))
    if len(envs) == 1:
        axes = [axes]

    colors = ['steelblue', 'coral', 'mediumseagreen', 'orchid']

    for env_idx, env_name in enumerate(envs):
        ax = axes[env_idx]
        results = all_results[env_name]

        means = []
        stds = []
        for algo in algorithms:
            final = results[algo][:, -50:].mean(axis=1)
            means.append(final.mean())
            stds.append(final.std())

        x = np.arange(len(algorithms))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black')

        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=15, ha='right')
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.set_title(env_name, fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/06_td_control/summary_comparison.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved summary chart to summary_comparison.png")
    plt.close()


def main():
    """Main comprehensive comparison."""
    print("="*60)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("="*60)

    # Create environments
    environments = [
        ('GridWorld', GridWorld(size=6)),
        ('CliffWalking', CliffWalking()),
        ('Stochastic Grid', StochasticGridWorld(size=5, slip_prob=0.2)),
    ]

    # Run comparisons
    all_results = {}
    for env_name, env in environments:
        results = run_comparison(env_name, env, n_episodes=500, n_runs=20)
        all_results[env_name] = results

    # Create visualizations
    create_comparison_dashboard(all_results)
    create_summary_bar_chart(all_results)
    print_summary_statistics(all_results)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print("\nKey Takeaways:")
    print("  - SARSA: Conservative, good for risky environments")
    print("  - Q-Learning: Learns optimal policy, may be risky during training")
    print("  - Expected SARSA: Low variance, stable learning")
    print("  - Double Q-Learning: Best for high-variance rewards")
    print("\nGenerated files:")
    print("  - algorithm_comparison_dashboard.png")
    print("  - summary_comparison.png")


if __name__ == "__main__":
    main()
