"""
03 - Expected SARSA

Expected SARSA uses the expected value over next actions instead of sampling.

Demonstrates:
- Expected SARSA algorithm
- Lower variance than SARSA
- Connection to Q-Learning
- Stability analysis
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt


class GridWorld:
    """GridWorld environment for Expected SARSA demonstration."""

    def __init__(self, size: int = 6, gamma: float = 0.95):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma

        self.actions = [0, 1, 2, 3]
        self.action_vectors = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        self.start_state = self._coord_to_state(size - 1, 0)
        self.goal_state = self._coord_to_state(0, size - 1)

        self.obstacles = {
            self._coord_to_state(2, 2),
            self._coord_to_state(2, 3),
            self._coord_to_state(3, 2),
        }

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
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

    def reset(self) -> int:
        return self.start_state


def get_epsilon_greedy_probs(Q: Dict, state: int, actions: List[int], epsilon: float) -> np.ndarray:
    """Get action probabilities under epsilon-greedy policy."""
    n_actions = len(actions)
    probs = np.ones(n_actions) * epsilon / n_actions

    q_values = [Q.get((state, a), 0.0) for a in actions]
    max_q = max(q_values)
    best_actions = [i for i, q in enumerate(q_values) if q == max_q]

    # Distribute remaining probability among best actions
    bonus = (1 - epsilon) / len(best_actions)
    for i in best_actions:
        probs[i] += bonus

    return probs


def expected_sarsa(env: GridWorld, n_episodes: int = 500, alpha: float = 0.1,
                   epsilon: float = 0.1, epsilon_decay: float = 0.995) -> Tuple[Dict, List]:
    """
    Expected SARSA: Uses expected value instead of sampled next action.

    Update: Q(S,A) ← Q(S,A) + α[R + γ Σ_a π(a|S')Q(S',a) - Q(S,A)]
    """
    Q = defaultdict(float)
    rewards_history = []
    current_epsilon = epsilon

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            # Choose action using epsilon-greedy
            probs = get_epsilon_greedy_probs(Q, state, env.actions, current_epsilon)
            action = np.random.choice(env.actions, p=probs)

            # Take action
            next_state, reward, done = env.step(state, action)
            total_reward += reward

            # Calculate expected Q-value for next state
            next_probs = get_epsilon_greedy_probs(Q, next_state, env.actions, current_epsilon)
            expected_q = sum(next_probs[i] * Q[(next_state, a)]
                           for i, a in enumerate(env.actions))

            # Expected SARSA update
            td_target = reward + env.gamma * expected_q
            td_error = td_target - Q[(state, action)]
            Q[(state, action)] += alpha * td_error

            state = next_state
            steps += 1

        rewards_history.append(total_reward)
        current_epsilon *= epsilon_decay

        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, ε: {current_epsilon:.3f}")

    return dict(Q), rewards_history


def sarsa(env: GridWorld, n_episodes: int, alpha: float, epsilon: float,
          epsilon_decay: float) -> Tuple[Dict, List]:
    """Standard SARSA for comparison."""
    Q = defaultdict(float)
    rewards_history = []
    current_epsilon = epsilon

    for episode in range(n_episodes):
        state = env.reset()
        probs = get_epsilon_greedy_probs(Q, state, env.actions, current_epsilon)
        action = np.random.choice(env.actions, p=probs)
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            next_state, reward, done = env.step(state, action)
            total_reward += reward

            next_probs = get_epsilon_greedy_probs(Q, next_state, env.actions, current_epsilon)
            next_action = np.random.choice(env.actions, p=next_probs)

            # SARSA update
            td_target = reward + env.gamma * Q[(next_state, next_action)]
            Q[(state, action)] += alpha * (td_target - Q[(state, action)])

            state = next_state
            action = next_action
            steps += 1

        rewards_history.append(total_reward)
        current_epsilon *= epsilon_decay

    return dict(Q), rewards_history


def q_learning(env: GridWorld, n_episodes: int, alpha: float, epsilon: float,
               epsilon_decay: float) -> Tuple[Dict, List]:
    """Q-Learning for comparison."""
    Q = defaultdict(float)
    rewards_history = []
    current_epsilon = epsilon

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            probs = get_epsilon_greedy_probs(Q, state, env.actions, current_epsilon)
            action = np.random.choice(env.actions, p=probs)

            next_state, reward, done = env.step(state, action)
            total_reward += reward

            # Q-Learning update
            max_next_q = max([Q[(next_state, a)] for a in env.actions])
            Q[(state, action)] += alpha * (reward + env.gamma * max_next_q - Q[(state, action)])

            state = next_state
            steps += 1

        rewards_history.append(total_reward)
        current_epsilon *= epsilon_decay

    return dict(Q), rewards_history


def compare_all_algorithms(env: GridWorld, n_runs: int = 20):
    """Compare Expected SARSA, SARSA, and Q-Learning."""
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON")
    print("="*60)

    n_episodes = 400
    alpha = 0.1
    epsilon = 0.1

    expected_sarsa_rewards = []
    sarsa_rewards = []
    q_learning_rewards = []

    print(f"\nRunning {n_runs} trials...")
    for run in range(n_runs):
        if run % 5 == 0:
            print(f"  Run {run + 1}/{n_runs}")

        _, e_rewards = expected_sarsa(env, n_episodes, alpha, epsilon, epsilon_decay=1.0)
        _, s_rewards = sarsa(env, n_episodes, alpha, epsilon, epsilon_decay=1.0)
        _, q_rewards = q_learning(env, n_episodes, alpha, epsilon, epsilon_decay=1.0)

        expected_sarsa_rewards.append(e_rewards)
        sarsa_rewards.append(s_rewards)
        q_learning_rewards.append(q_rewards)

    # Convert to arrays
    expected_sarsa_rewards = np.array(expected_sarsa_rewards)
    sarsa_rewards = np.array(sarsa_rewards)
    q_learning_rewards = np.array(q_learning_rewards)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Expected SARSA vs SARSA vs Q-Learning', fontsize=16)

    # Plot 1: Learning curves
    ax = axes[0, 0]
    window = 20

    for rewards, name in [(expected_sarsa_rewards, 'Expected SARSA'),
                          (sarsa_rewards, 'SARSA'),
                          (q_learning_rewards, 'Q-Learning')]:
        mean = rewards.mean(axis=0)
        smoothed = np.convolve(mean, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=name, linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward (smoothed)', fontsize=12)
    ax.set_title('Learning Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Variance over time
    ax = axes[0, 1]
    for rewards, name in [(expected_sarsa_rewards, 'Expected SARSA'),
                          (sarsa_rewards, 'SARSA'),
                          (q_learning_rewards, 'Q-Learning')]:
        variance = rewards.var(axis=0)
        smoothed_var = np.convolve(variance, np.ones(window)/window, mode='valid')
        ax.plot(smoothed_var, label=name, linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Variance Across Runs', fontsize=12)
    ax.set_title('Learning Variance', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 3: Final performance boxplot
    ax = axes[1, 0]
    final_expected = expected_sarsa_rewards[:, -50:].mean(axis=1)
    final_sarsa = sarsa_rewards[:, -50:].mean(axis=1)
    final_q = q_learning_rewards[:, -50:].mean(axis=1)

    ax.boxplot([final_expected, final_sarsa, final_q],
               labels=['Expected\nSARSA', 'SARSA', 'Q-Learning'])
    ax.set_ylabel('Average Reward (last 50 episodes)', fontsize=12)
    ax.set_title('Final Performance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Variance comparison bar chart
    ax = axes[1, 1]
    variances = [final_expected.var(), final_sarsa.var(), final_q.var()]
    colors = ['steelblue', 'coral', 'green']
    ax.bar(['Expected\nSARSA', 'SARSA', 'Q-Learning'], variances, color=colors)
    ax.set_ylabel('Variance of Final Performance', fontsize=12)
    ax.set_title('Stability Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/06_td_control/expected_sarsa_comparison.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved comparison to expected_sarsa_comparison.png")
    plt.close()

    # Print statistics
    print("\nFinal Performance Statistics:")
    print(f"  Expected SARSA: {final_expected.mean():.2f} ± {final_expected.std():.2f}")
    print(f"  SARSA:          {final_sarsa.mean():.2f} ± {final_sarsa.std():.2f}")
    print(f"  Q-Learning:     {final_q.mean():.2f} ± {final_q.std():.2f}")

    print("\nVariance (lower is more stable):")
    print(f"  Expected SARSA: {final_expected.var():.4f}")
    print(f"  SARSA:          {final_sarsa.var():.4f}")
    print(f"  Q-Learning:     {final_q.var():.4f}")


def demonstrate_expected_value_calculation(env: GridWorld):
    """Show how expected value is calculated."""
    print("\n" + "="*60)
    print("EXPECTED VALUE CALCULATION")
    print("="*60)

    # Create some Q-values
    Q = defaultdict(float)
    state = 10
    Q[(state, 0)] = 2.0  # Up
    Q[(state, 1)] = 5.0  # Right (best)
    Q[(state, 2)] = 1.0  # Down
    Q[(state, 3)] = 3.0  # Left

    epsilon = 0.2

    print(f"\nQ-values for state {state}:")
    for a in env.actions:
        print(f"  Action {a}: Q = {Q[(state, a)]:.1f}")

    # Get probabilities
    probs = get_epsilon_greedy_probs(Q, state, env.actions, epsilon)
    print(f"\nEpsilon-greedy probabilities (ε={epsilon}):")
    for a, p in zip(env.actions, probs):
        print(f"  Action {a}: π(a|s) = {p:.3f}")

    # Calculate expected value
    expected_q = sum(probs[i] * Q[(state, a)] for i, a in enumerate(env.actions))
    print(f"\nExpected Q-value: Σ π(a|s) * Q(s,a) = {expected_q:.3f}")

    # Compare with max (Q-Learning target)
    max_q = max([Q[(state, a)] for a in env.actions])
    print(f"Max Q-value: max_a Q(s,a) = {max_q:.1f}")

    # Compare with sampled (SARSA target)
    print(f"\nSARSA would sample one action with above probabilities")
    print(f"Expected SARSA uses the weighted sum: {expected_q:.3f}")
    print(f"Q-Learning uses the max: {max_q:.1f}")


def show_connection_to_q_learning():
    """Show that Expected SARSA with ε=0 equals Q-Learning."""
    print("\n" + "="*60)
    print("CONNECTION TO Q-LEARNING")
    print("="*60)

    print("\nWhen ε = 0 (greedy policy):")
    print("  - π(a*|s) = 1 for a* = argmax Q(s,a)")
    print("  - π(a|s) = 0 for all other a")
    print("")
    print("Expected SARSA target:")
    print("  Σ π(a|S') Q(S',a) = 1 × Q(S',a*) + 0 × ... = max_a Q(S',a)")
    print("")
    print("This is exactly the Q-Learning target!")
    print("")
    print("Therefore: Expected SARSA with ε=0 ≡ Q-Learning")


def main():
    """Main demonstration."""
    print("="*60)
    print("EXPECTED SARSA")
    print("="*60)

    # Create environment
    env = GridWorld(size=6, gamma=0.95)

    # Run Expected SARSA
    print("\nRunning Expected SARSA...")
    Q, rewards_history = expected_sarsa(env, n_episodes=500, alpha=0.1, epsilon=0.2, epsilon_decay=0.995)

    # Compare all algorithms
    compare_all_algorithms(env, n_runs=20)

    # Demonstrate expected value calculation
    demonstrate_expected_value_calculation(env)

    # Show connection to Q-Learning
    show_connection_to_q_learning()

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Points:")
    print("  - Expected SARSA uses expected value instead of sampling")
    print("  - Lower variance than SARSA (no sampling noise)")
    print("  - With ε=0, becomes equivalent to Q-Learning")
    print("  - Generally more stable learning")


if __name__ == "__main__":
    main()
