"""
05 - Random Walk Problem

The classic Random Walk problem from Sutton & Barto Chapter 6.

Demonstrates:
- 1D random walk environment
- True value computation
- TD vs MC on a simple problem
- Batch vs incremental updates
- RMS error analysis
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt


class RandomWalk:
    """
    Classic 5-state Random Walk from Sutton & Barto.

    States: [T_left, A, B, C, D, E, T_right]
    Start in C (center)
    Actions: Left or Right (equal probability)
    Left terminal: reward 0
    Right terminal: reward 1
    All other transitions: reward 0
    """

    def __init__(self, n_states: int = 5, gamma: float = 1.0):
        """
        Args:
            n_states: Number of non-terminal states
            gamma: Discount factor (usually 1.0 for episodic tasks)
        """
        self.n_states = n_states
        self.gamma = gamma

        # States: 0=left_terminal, 1-5=non-terminal, 6=right_terminal
        self.terminal_left = 0
        self.terminal_right = n_states + 1
        self.start_state = n_states // 2 + 1  # Middle state

        # Compute true values analytically
        self.true_values = self._compute_true_values()

    def _compute_true_values(self) -> Dict[int, float]:
        """
        Compute true values analytically.

        For random walk with equal probability left/right,
        V(state) = state / (n_states + 1)
        """
        true_V = {}
        for state in range(1, self.n_states + 1):
            true_V[state] = state / (self.n_states + 1)

        true_V[self.terminal_left] = 0.0
        true_V[self.terminal_right] = 1.0

        return true_V

    def reset(self) -> int:
        """Reset to start state."""
        return self.start_state

    def step(self, state: int) -> Tuple[int, float, bool]:
        """
        Take a random step (left or right with equal probability).

        Returns:
            next_state, reward, done
        """
        if state == self.terminal_left or state == self.terminal_right:
            return state, 0, True

        # Random action: -1 (left) or +1 (right)
        action = np.random.choice([-1, 1])
        next_state = state + action

        # Check for terminal states
        if next_state == self.terminal_left:
            return next_state, 0, True
        elif next_state == self.terminal_right:
            return next_state, 1, True
        else:
            return next_state, 0, False

    def get_state_name(self, state: int) -> str:
        """Get human-readable state name."""
        if state == self.terminal_left:
            return "T_L"
        elif state == self.terminal_right:
            return "T_R"
        else:
            # States 1-5 correspond to A-E
            state_names = ['A', 'B', 'C', 'D', 'E']
            if 1 <= state <= self.n_states:
                return state_names[state - 1]
        return "?"


def td_prediction_random_walk(env: RandomWalk, n_episodes: int,
                               alpha: float) -> Tuple[Dict, List]:
    """TD(0) prediction for random walk."""
    V = defaultdict(float)

    # Initialize non-terminal states to 0.5 (optional)
    for state in range(1, env.n_states + 1):
        V[state] = 0.5

    rms_errors = []

    for episode in range(n_episodes):
        state = env.reset()
        done = False

        while not done:
            next_state, reward, done = env.step(state)

            # TD(0) update
            V[state] += alpha * (reward + env.gamma * V[next_state] - V[state])

            state = next_state

        # Calculate RMS error
        rms_error = np.sqrt(np.mean([
            (V[s] - env.true_values[s])**2
            for s in range(1, env.n_states + 1)
        ]))
        rms_errors.append(rms_error)

    return dict(V), rms_errors


def mc_prediction_random_walk(env: RandomWalk, n_episodes: int,
                               alpha: float) -> Tuple[Dict, List]:
    """Monte Carlo prediction for random walk."""
    V = defaultdict(float)

    for state in range(1, env.n_states + 1):
        V[state] = 0.5

    rms_errors = []

    for episode in range(n_episodes):
        # Generate episode
        episode_data = []
        state = env.reset()
        done = False

        while not done:
            next_state, reward, done = env.step(state)
            episode_data.append((state, reward))
            state = next_state

        # Update values (every-visit MC)
        G = 0
        for t in range(len(episode_data) - 1, -1, -1):
            state, reward = episode_data[t]
            G = env.gamma * G + reward
            V[state] += alpha * (G - V[state])

        # Calculate RMS error
        rms_error = np.sqrt(np.mean([
            (V[s] - env.true_values[s])**2
            for s in range(1, env.n_states + 1)
        ]))
        rms_errors.append(rms_error)

    return dict(V), rms_errors


def compare_td_mc_random_walk(env: RandomWalk, n_runs: int = 100):
    """Compare TD and MC on random walk."""
    print("\n" + "="*60)
    print("RANDOM WALK: TD vs MC")
    print("="*60)

    n_episodes = 100
    alpha_td = 0.1
    alpha_mc = 0.01  # MC typically needs smaller alpha

    td_errors_all = []
    mc_errors_all = []

    print(f"\nRunning {n_runs} trials...")
    for run in range(n_runs):
        if run % 20 == 0:
            print(f"  Run {run + 1}/{n_runs}")

        _, td_errors = td_prediction_random_walk(env, n_episodes, alpha_td)
        _, mc_errors = mc_prediction_random_walk(env, n_episodes, alpha_mc)

        td_errors_all.append(td_errors)
        mc_errors_all.append(mc_errors)

    # Convert to arrays
    td_errors_all = np.array(td_errors_all)
    mc_errors_all = np.array(mc_errors_all)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Learning curves
    ax = axes[0]
    td_mean = td_errors_all.mean(axis=0)
    td_std = td_errors_all.std(axis=0)
    mc_mean = mc_errors_all.mean(axis=0)
    mc_std = mc_errors_all.std(axis=0)

    episodes = np.arange(n_episodes)
    ax.plot(episodes, td_mean, label=f'TD (α={alpha_td})', linewidth=2)
    ax.fill_between(episodes, td_mean - td_std, td_mean + td_std, alpha=0.3)
    ax.plot(episodes, mc_mean, label=f'MC (α={alpha_mc})', linewidth=2)
    ax.fill_between(episodes, mc_mean - mc_std, mc_mean + mc_std, alpha=0.3)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('RMS Error', fontsize=12)
    ax.set_title('Learning Curves (mean ± std)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Final values comparison
    ax = axes[1]

    # Run one more time for final values
    V_td, _ = td_prediction_random_walk(env, n_episodes, alpha_td)
    V_mc, _ = mc_prediction_random_walk(env, n_episodes, alpha_mc)

    states = list(range(1, env.n_states + 1))
    state_names = [env.get_state_name(s) for s in states]
    true_vals = [env.true_values[s] for s in states]
    td_vals = [V_td[s] for s in states]
    mc_vals = [V_mc[s] for s in states]

    x = np.arange(len(states))
    width = 0.25

    ax.bar(x - width, true_vals, width, label='True Values', color='green', alpha=0.7)
    ax.bar(x, td_vals, width, label='TD Estimates', color='blue', alpha=0.7)
    ax.bar(x + width, mc_vals, width, label='MC Estimates', color='red', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(state_names)
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Final Value Estimates', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/random_walk_comparison.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved random walk comparison to random_walk_comparison.png")
    plt.close()

    # Print statistics
    print(f"\nFinal RMS Error (last 10 episodes):")
    print(f"  TD: {td_mean[-10:].mean():.4f} ± {td_std[-10:].mean():.4f}")
    print(f"  MC: {mc_mean[-10:].mean():.4f} ± {mc_std[-10:].mean():.4f}")


def analyze_learning_rate(env: RandomWalk, n_runs: int = 50):
    """Analyze effect of learning rate."""
    print("\n" + "="*60)
    print("LEARNING RATE SENSITIVITY")
    print("="*60)

    alpha_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    n_episodes = 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # TD sensitivity
    ax = axes[0]
    for alpha in alpha_values:
        all_errors = []
        for _ in range(n_runs):
            _, errors = td_prediction_random_walk(env, n_episodes, alpha)
            all_errors.append(errors)

        mean_errors = np.array(all_errors).mean(axis=0)
        ax.plot(mean_errors, label=f'α={alpha}', linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('RMS Error', fontsize=12)
    ax.set_title('TD(0) Learning Rate Sensitivity', fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # MC sensitivity
    ax = axes[1]
    alpha_values_mc = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    for alpha in alpha_values_mc:
        all_errors = []
        for _ in range(n_runs):
            _, errors = mc_prediction_random_walk(env, n_episodes, alpha)
            all_errors.append(errors)

        mean_errors = np.array(all_errors).mean(axis=0)
        ax.plot(mean_errors, label=f'α={alpha}', linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('RMS Error', fontsize=12)
    ax.set_title('MC Learning Rate Sensitivity', fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/random_walk_alpha.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved learning rate analysis to random_walk_alpha.png")
    plt.close()


def batch_updating_comparison(env: RandomWalk):
    """Compare batch vs incremental updating."""
    print("\n" + "="*60)
    print("BATCH VS INCREMENTAL UPDATING")
    print("="*60)

    n_episodes = 100
    n_runs = 50

    # Generate training set
    training_episodes = []
    for _ in range(10):  # 10 episodes
        episode = []
        state = env.reset()
        done = False
        while not done:
            next_state, reward, done = env.step(state)
            episode.append((state, reward, next_state))
            state = next_state
        training_episodes.append(episode)

    # Incremental TD
    incremental_errors = []
    for _ in range(n_runs):
        V = defaultdict(float)
        for s in range(1, env.n_states + 1):
            V[s] = 0.5

        for _ in range(n_episodes):
            for episode in training_episodes:
                for state, reward, next_state in episode:
                    V[state] += 0.1 * (reward + env.gamma * V[next_state] - V[state])

            rms = np.sqrt(np.mean([
                (V[s] - env.true_values[s])**2
                for s in range(1, env.n_states + 1)
            ]))
            incremental_errors.append(rms)

    # Batch TD (repeatedly update until convergence)
    batch_errors = []
    for _ in range(n_runs):
        V = defaultdict(float)
        for s in range(1, env.n_states + 1):
            V[s] = 0.5

        for _ in range(n_episodes):
            # Repeat updates until convergence
            for _ in range(100):
                for episode in training_episodes:
                    for state, reward, next_state in episode:
                        V[state] += 0.1 * (reward + env.gamma * V[next_state] - V[state])

            rms = np.sqrt(np.mean([
                (V[s] - env.true_values[s])**2
                for s in range(1, env.n_states + 1)
            ]))
            batch_errors.append(rms)

    print(f"\nFinal RMS Error:")
    print(f"  Incremental: {np.mean(incremental_errors[-n_runs:]):.4f}")
    print(f"  Batch:       {np.mean(batch_errors[-n_runs:]):.4f}")


def main():
    """Main demonstration."""
    print("="*60)
    print("RANDOM WALK PROBLEM")
    print("="*60)

    # Create environment
    env = RandomWalk(n_states=5, gamma=1.0)

    # Print true values
    print("\nTrue Values:")
    for state in range(1, env.n_states + 1):
        name = env.get_state_name(state)
        value = env.true_values[state]
        print(f"  {name}: {value:.3f}")

    # Run experiments
    compare_td_mc_random_walk(env, n_runs=100)
    analyze_learning_rate(env, n_runs=50)
    batch_updating_comparison(env)

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Findings:")
    print("  - TD converges faster than MC on this problem")
    print("  - TD is less sensitive to learning rate")
    print("  - Batch updating can achieve lower error")
    print("  - True values can be computed analytically here")


if __name__ == "__main__":
    main()
