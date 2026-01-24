"""
02 - Q-Learning: Off-Policy TD Control

Q-Learning learns the optimal Q-values regardless of the policy being followed.

Demonstrates:
- Q-Learning algorithm implementation
- Off-policy learning characteristics
- Comparison with SARSA
- Convergence to optimal policy
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt


class GridWorld:
    """GridWorld environment for Q-Learning demonstration."""

    def __init__(self, size: int = 6, gamma: float = 0.95):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma

        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.action_names = {0: "↑", 1: "→", 2: "↓", 3: "←"}
        self.action_vectors = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        self.start_state = self._coord_to_state(size - 1, 0)
        self.goal_state = self._coord_to_state(0, size - 1)

        # Dangerous zone (high negative reward)
        self.danger_zone = {
            self._coord_to_state(2, 1),
            self._coord_to_state(2, 2),
            self._coord_to_state(2, 3),
            self._coord_to_state(3, 3),
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
        elif next_state in self.danger_zone:
            return next_state, -10.0, False
        else:
            return next_state, -0.1, False

    def reset(self) -> int:
        return self.start_state


def epsilon_greedy(Q: Dict, state: int, actions: List[int], epsilon: float) -> int:
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.choice(actions)
    else:
        q_values = [Q.get((state, a), 0.0) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return np.random.choice(best_actions)


def q_learning(env: GridWorld, n_episodes: int = 500, alpha: float = 0.1,
               epsilon: float = 0.1, epsilon_decay: float = 0.995) -> Tuple[Dict, List]:
    """
    Q-Learning: Off-policy TD Control.

    Args:
        env: Environment
        n_episodes: Number of episodes
        alpha: Learning rate
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay per episode

    Returns:
        Q: Learned Q-values
        rewards_history: Total reward per episode
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
            action = epsilon_greedy(Q, state, env.actions, current_epsilon)

            # Take action
            next_state, reward, done = env.step(state, action)
            total_reward += reward

            # Q-Learning update: Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
            max_next_q = max([Q[(next_state, a)] for a in env.actions])
            td_target = reward + env.gamma * max_next_q
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


def sarsa(env: GridWorld, n_episodes: int = 500, alpha: float = 0.1,
          epsilon: float = 0.1, epsilon_decay: float = 0.995) -> Tuple[Dict, List]:
    """SARSA for comparison."""
    Q = defaultdict(float)
    rewards_history = []
    current_epsilon = epsilon

    for episode in range(n_episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, env.actions, current_epsilon)
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            next_state, reward, done = env.step(state, action)
            total_reward += reward

            next_action = epsilon_greedy(Q, next_state, env.actions, current_epsilon)

            # SARSA update
            td_target = reward + env.gamma * Q[(next_state, next_action)]
            Q[(state, action)] += alpha * (td_target - Q[(state, action)])

            state = next_state
            action = next_action
            steps += 1

        rewards_history.append(total_reward)
        current_epsilon *= epsilon_decay

    return dict(Q), rewards_history


def compare_q_learning_sarsa(env: GridWorld, n_runs: int = 20):
    """Compare Q-Learning and SARSA."""
    print("\n" + "="*60)
    print("Q-LEARNING vs SARSA COMPARISON")
    print("="*60)

    n_episodes = 500
    alpha = 0.1
    epsilon = 0.1

    q_learning_rewards = []
    sarsa_rewards = []

    print(f"\nRunning {n_runs} trials...")
    for run in range(n_runs):
        if run % 5 == 0:
            print(f"  Run {run + 1}/{n_runs}")

        _, q_rewards = q_learning(env, n_episodes, alpha, epsilon, epsilon_decay=1.0)
        _, s_rewards = sarsa(env, n_episodes, alpha, epsilon, epsilon_decay=1.0)

        q_learning_rewards.append(q_rewards)
        sarsa_rewards.append(s_rewards)

    # Convert to arrays
    q_learning_rewards = np.array(q_learning_rewards)
    sarsa_rewards = np.array(sarsa_rewards)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curves
    ax = axes[0]
    window = 20

    q_mean = q_learning_rewards.mean(axis=0)
    q_std = q_learning_rewards.std(axis=0)
    s_mean = sarsa_rewards.mean(axis=0)
    s_std = sarsa_rewards.std(axis=0)

    q_smoothed = np.convolve(q_mean, np.ones(window)/window, mode='valid')
    s_smoothed = np.convolve(s_mean, np.ones(window)/window, mode='valid')

    episodes = np.arange(len(q_smoothed))
    ax.plot(episodes, q_smoothed, label='Q-Learning', linewidth=2)
    ax.plot(episodes, s_smoothed, label='SARSA', linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward (smoothed)', fontsize=12)
    ax.set_title('Learning Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Final performance
    ax = axes[1]
    q_final = q_learning_rewards[:, -50:].mean(axis=1)
    s_final = sarsa_rewards[:, -50:].mean(axis=1)

    positions = [0, 1]
    bp = ax.boxplot([q_final, s_final], positions=positions, widths=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(['Q-Learning', 'SARSA'])
    ax.set_ylabel('Average Reward (last 50 episodes)', fontsize=12)
    ax.set_title('Final Performance Distribution', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/06_td_control/q_learning_vs_sarsa.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved comparison to q_learning_vs_sarsa.png")
    plt.close()

    print(f"\nFinal Performance (last 50 episodes):")
    print(f"  Q-Learning: {q_final.mean():.2f} ± {q_final.std():.2f}")
    print(f"  SARSA:      {s_final.mean():.2f} ± {s_final.std():.2f}")


def visualize_q_learning(env: GridWorld, Q: Dict, rewards_history: List):
    """Visualize Q-Learning results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Q-Learning Results', fontsize=16)

    # Plot 1: Learning curve
    ax = axes[0, 0]
    window = 20
    smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    ax.plot(smoothed, linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward (smoothed)', fontsize=12)
    ax.set_title('Learning Curve', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Plot 2: Value function
    ax = axes[0, 1]
    value_grid = np.zeros((env.size, env.size))
    for state in range(env.n_states):
        row, col = env._state_to_coord(state)
        max_q = max([Q.get((state, a), 0.0) for a in env.actions])
        value_grid[row, col] = max_q

    im = ax.imshow(value_grid, cmap='viridis', interpolation='nearest')
    ax.set_title('Value Function (max Q)', fontsize=14)
    plt.colorbar(im, ax=ax)

    # Mark special states
    start_row, start_col = env._state_to_coord(env.start_state)
    goal_row, goal_col = env._state_to_coord(env.goal_state)
    ax.plot(start_col, start_row, 'rs', markersize=15, label='Start')
    ax.plot(goal_col, goal_row, 'g*', markersize=20, label='Goal')
    for danger in env.danger_zone:
        d_row, d_col = env._state_to_coord(danger)
        ax.plot(d_col, d_row, 'rx', markersize=15, markeredgewidth=3)
    ax.legend(fontsize=10)

    # Plot 3: Policy
    ax = axes[1, 0]
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(-0.5, env.size - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    for state in range(env.n_states):
        row, col = env._state_to_coord(state)

        if state == env.goal_state:
            ax.plot(col, row, 'g*', markersize=25)
            continue
        if state in env.danger_zone:
            ax.plot(col, row, 'rs', markersize=20, alpha=0.5)
            continue

        q_values = [Q.get((state, a), 0.0) for a in env.actions]
        best_action = env.actions[np.argmax(q_values)]

        dr, dc = env.action_vectors[best_action]
        ax.arrow(col, row, dc * 0.3, dr * 0.3,
                head_width=0.15, head_length=0.1,
                fc='blue', ec='blue')

    ax.set_title('Learned Policy (Optimal)', fontsize=14)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.grid(True, alpha=0.3)

    # Plot 4: Q-value difference between actions
    ax = axes[1, 1]
    q_range = np.zeros((env.size, env.size))
    for state in range(env.n_states):
        row, col = env._state_to_coord(state)
        q_values = [Q.get((state, a), 0.0) for a in env.actions]
        q_range[row, col] = max(q_values) - min(q_values)

    im = ax.imshow(q_range, cmap='hot', interpolation='nearest')
    ax.set_title('Q-value Range (max - min)', fontsize=14)
    plt.colorbar(im, ax=ax, label='Action Value Spread')

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/06_td_control/q_learning_results.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved results to q_learning_results.png")
    plt.close()


def demonstrate_off_policy(env: GridWorld):
    """Demonstrate off-policy nature of Q-Learning."""
    print("\n" + "="*60)
    print("OFF-POLICY DEMONSTRATION")
    print("="*60)

    # Train Q-Learning with high exploration
    print("\nTraining Q-Learning with very high exploration (ε=0.5)...")
    Q, _ = q_learning(env, n_episodes=500, epsilon=0.5, epsilon_decay=1.0)

    # Test the learned greedy policy
    print("\nTesting learned GREEDY policy (ε=0)...")
    greedy_rewards = []

    for _ in range(100):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 100:
            # Greedy action
            q_values = [Q.get((state, a), 0.0) for a in env.actions]
            action = env.actions[np.argmax(q_values)]
            state, reward, done = env.step(state, action)
            total_reward += reward
            steps += 1

        greedy_rewards.append(total_reward)

    print(f"\nGreedy policy performance: {np.mean(greedy_rewards):.2f} ± {np.std(greedy_rewards):.2f}")
    print("\nThis shows Q-Learning learned the OPTIMAL policy")
    print("even though it was trained with 50% random actions!")


def main():
    """Main demonstration."""
    print("="*60)
    print("Q-LEARNING: OFF-POLICY TD CONTROL")
    print("="*60)

    # Create environment
    env = GridWorld(size=6, gamma=0.95)

    # Run Q-Learning
    print("\nRunning Q-Learning...")
    Q, rewards_history = q_learning(env, n_episodes=500, alpha=0.1, epsilon=0.2, epsilon_decay=0.995)

    # Visualize results
    visualize_q_learning(env, Q, rewards_history)

    # Compare with SARSA
    compare_q_learning_sarsa(env, n_runs=20)

    # Demonstrate off-policy learning
    demonstrate_off_policy(env)

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Points:")
    print("  - Q-Learning is OFF-POLICY: learns optimal policy regardless of behavior")
    print("  - Uses max_a Q(S',a) - always assumes optimal next action")
    print("  - Converges to Q* (optimal Q-values)")
    print("  - More aggressive/optimistic than SARSA")


if __name__ == "__main__":
    main()
