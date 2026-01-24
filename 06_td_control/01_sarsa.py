"""
01 - SARSA: On-Policy TD Control

SARSA (State-Action-Reward-State-Action) learns Q-values while following
an epsilon-greedy policy.

Demonstrates:
- SARSA algorithm implementation
- Epsilon-greedy policy
- On-policy learning characteristics
- Policy visualization
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt


class GridWorld:
    """
    GridWorld environment for SARSA demonstration.

    Features obstacles and a goal state with various rewards.
    """

    def __init__(self, size: int = 6, gamma: float = 0.95):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma

        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        self.actions = [0, 1, 2, 3]
        self.action_names = {0: "↑", 1: "→", 2: "↓", 3: "←"}
        self.action_vectors = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        # Start and goal
        self.start_state = self._coord_to_state(size - 1, 0)
        self.goal_state = self._coord_to_state(0, size - 1)

        # Obstacles (negative reward)
        self.obstacles = {
            self._coord_to_state(1, 2),
            self._coord_to_state(2, 2),
            self._coord_to_state(3, 2),
            self._coord_to_state(3, 3),
        }

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        """Execute action and return next state, reward, done."""
        if state == self.goal_state:
            return state, 0.0, True

        row, col = self._state_to_coord(state)
        dr, dc = self.action_vectors[action]

        new_row = max(0, min(self.size - 1, row + dr))
        new_col = max(0, min(self.size - 1, col + dc))
        next_state = self._coord_to_state(new_row, new_col)

        # Rewards
        if next_state == self.goal_state:
            return next_state, 10.0, True
        elif next_state in self.obstacles:
            return next_state, -5.0, False
        else:
            return next_state, -0.1, False

    def reset(self) -> int:
        return self.start_state


def epsilon_greedy(Q: Dict, state: int, actions: List[int], epsilon: float) -> int:
    """
    Epsilon-greedy action selection.

    With probability epsilon: random action
    With probability 1-epsilon: greedy action
    """
    if np.random.random() < epsilon:
        return np.random.choice(actions)
    else:
        q_values = [Q.get((state, a), 0.0) for a in actions]
        max_q = max(q_values)
        # Break ties randomly
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return np.random.choice(best_actions)


def sarsa(env: GridWorld, n_episodes: int = 500, alpha: float = 0.1,
          epsilon: float = 0.1, epsilon_decay: float = 0.995) -> Tuple[Dict, List]:
    """
    SARSA: On-policy TD Control.

    Args:
        env: GridWorld environment
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
        action = epsilon_greedy(Q, state, env.actions, current_epsilon)

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            # Take action, observe reward and next state
            next_state, reward, done = env.step(state, action)
            total_reward += reward

            # Choose next action using epsilon-greedy (SARSA!)
            next_action = epsilon_greedy(Q, next_state, env.actions, current_epsilon)

            # SARSA update: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
            td_target = reward + env.gamma * Q[(next_state, next_action)]
            td_error = td_target - Q[(state, action)]
            Q[(state, action)] += alpha * td_error

            state = next_state
            action = next_action
            steps += 1

        rewards_history.append(total_reward)
        current_epsilon *= epsilon_decay

        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, ε: {current_epsilon:.3f}")

    return dict(Q), rewards_history


def extract_policy(Q: Dict, n_states: int, actions: List[int]) -> np.ndarray:
    """Extract greedy policy from Q-values."""
    policy = np.zeros(n_states, dtype=int)
    for state in range(n_states):
        q_values = [Q.get((state, a), 0.0) for a in actions]
        policy[state] = actions[np.argmax(q_values)]
    return policy


def visualize_results(env: GridWorld, Q: Dict, rewards_history: List):
    """Visualize Q-values, policy, and learning curve."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('SARSA Learning Results', fontsize=16)

    # Plot 1: Learning curve
    ax = axes[0, 0]
    window = 20
    smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    ax.plot(smoothed, linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward (smoothed)', fontsize=12)
    ax.set_title('Learning Curve', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Plot 2: Value function (max Q for each state)
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
    for obs in env.obstacles:
        obs_row, obs_col = env._state_to_coord(obs)
        ax.plot(obs_col, obs_row, 'kx', markersize=15, markeredgewidth=3)
    ax.legend(fontsize=10)

    # Plot 3: Policy arrows
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
        if state in env.obstacles:
            ax.plot(col, row, 'ks', markersize=20, alpha=0.5)
            continue

        # Get best action
        q_values = [Q.get((state, a), 0.0) for a in env.actions]
        best_action = env.actions[np.argmax(q_values)]

        # Draw arrow
        dr, dc = env.action_vectors[best_action]
        ax.arrow(col, row, dc * 0.3, dr * 0.3,
                head_width=0.15, head_length=0.1,
                fc='blue', ec='blue')

    ax.set_title('Learned Policy', fontsize=14)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.grid(True, alpha=0.3)

    # Plot 4: Q-value heatmap for each action
    ax = axes[1, 1]
    action_q_grids = []
    for action in env.actions:
        q_grid = np.zeros((env.size, env.size))
        for state in range(env.n_states):
            row, col = env._state_to_coord(state)
            q_grid[row, col] = Q.get((state, action), 0.0)
        action_q_grids.append(q_grid)

    # Show Q-values for "Right" action as example
    im = ax.imshow(action_q_grids[1], cmap='RdYlGn', interpolation='nearest')
    ax.set_title('Q-values for "Right" Action', fontsize=14)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/06_td_control/sarsa_results.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved results to sarsa_results.png")
    plt.close()


def compare_epsilon_values(env: GridWorld):
    """Compare SARSA with different epsilon values."""
    print("\n" + "="*60)
    print("COMPARING EPSILON VALUES")
    print("="*60)

    epsilon_values = [0.01, 0.1, 0.3, 0.5]
    n_episodes = 400
    n_runs = 10

    plt.figure(figsize=(12, 6))

    for epsilon in epsilon_values:
        print(f"\nTesting ε={epsilon}...")
        all_rewards = []

        for run in range(n_runs):
            _, rewards = sarsa(env, n_episodes, alpha=0.1, epsilon=epsilon, epsilon_decay=1.0)
            all_rewards.append(rewards)

        mean_rewards = np.array(all_rewards).mean(axis=0)
        window = 20
        smoothed = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, label=f'ε={epsilon}', linewidth=2)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward (smoothed)', fontsize=12)
    plt.title('SARSA: Effect of Epsilon', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/06_td_control/sarsa_epsilon_comparison.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved epsilon comparison to sarsa_epsilon_comparison.png")
    plt.close()


def demonstrate_on_policy_behavior(env: GridWorld):
    """Demonstrate that SARSA learns about the policy it follows."""
    print("\n" + "="*60)
    print("ON-POLICY BEHAVIOR DEMONSTRATION")
    print("="*60)

    # Train with high epsilon (lots of exploration)
    print("\nTraining with high epsilon (ε=0.3)...")
    Q_high_eps, _ = sarsa(env, n_episodes=500, epsilon=0.3, epsilon_decay=1.0)

    # Train with low epsilon (less exploration)
    print("\nTraining with low epsilon (ε=0.05)...")
    Q_low_eps, _ = sarsa(env, n_episodes=500, epsilon=0.05, epsilon_decay=1.0)

    # Compare Q-values for a state near an obstacle
    test_state = env._coord_to_state(2, 1)  # Near obstacle

    print(f"\nQ-values for state near obstacle:")
    print(f"{'Action':<10} {'High ε':<12} {'Low ε':<12}")
    print("-" * 34)
    for action in env.actions:
        q_high = Q_high_eps.get((test_state, action), 0.0)
        q_low = Q_low_eps.get((test_state, action), 0.0)
        print(f"{env.action_names[action]:<10} {q_high:<12.3f} {q_low:<12.3f}")

    print("\nNote: High epsilon SARSA has lower Q-values near obstacles")
    print("because it accounts for the exploration (mistakes) in its estimates.")


def main():
    """Main demonstration."""
    print("="*60)
    print("SARSA: ON-POLICY TD CONTROL")
    print("="*60)

    # Create environment
    env = GridWorld(size=6, gamma=0.95)

    # Run SARSA
    print("\nRunning SARSA...")
    Q, rewards_history = sarsa(env, n_episodes=500, alpha=0.1, epsilon=0.2, epsilon_decay=0.995)

    # Visualize results
    visualize_results(env, Q, rewards_history)

    # Compare epsilon values
    compare_epsilon_values(env)

    # Demonstrate on-policy behavior
    demonstrate_on_policy_behavior(env)

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Points:")
    print("  - SARSA is ON-POLICY: learns about the policy being followed")
    print("  - Uses Q(S',A') where A' is the actual next action")
    print("  - More conservative near dangerous states")
    print("  - Values include the cost of exploration")


if __name__ == "__main__":
    main()
