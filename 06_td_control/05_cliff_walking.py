"""
05 - Cliff Walking

The classic Cliff Walking problem demonstrating SARSA vs Q-Learning difference.

Demonstrates:
- Gymnasium CliffWalking environment
- SARSA learns safe path
- Q-Learning learns optimal (but risky) path
- On-policy vs off-policy in dangerous environments
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    print("Gymnasium not installed. Using custom CliffWalking environment.")


class CliffWalkingEnv:
    """
    Custom Cliff Walking environment.

    Grid layout (4x12):
    [  ][  ][  ][  ][  ][  ][  ][  ][  ][  ][  ][  ]
    [  ][  ][  ][  ][  ][  ][  ][  ][  ][  ][  ][  ]
    [  ][  ][  ][  ][  ][  ][  ][  ][  ][  ][  ][  ]
    [S ][C ][C ][C ][C ][C ][C ][C ][C ][C ][C ][G ]

    S = Start, G = Goal, C = Cliff
    Cliff: reward -100, back to start
    Goal: reward 0 (terminal)
    Step: reward -1
    """

    def __init__(self):
        self.height = 4
        self.width = 12
        self.n_states = self.height * self.width
        self.gamma = 1.0  # Undiscounted

        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        self.actions = [0, 1, 2, 3]
        self.action_vectors = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        # Start and goal
        self.start_state = self._coord_to_state(3, 0)
        self.goal_state = self._coord_to_state(3, 11)

        # Cliff (bottom row, columns 1-10)
        self.cliff = set()
        for col in range(1, 11):
            self.cliff.add(self._coord_to_state(3, col))

        self.state = self.start_state

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.width, state % self.width

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.width + col

    def reset(self) -> int:
        self.state = self.start_state
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """Take action, return (state, reward, terminated, truncated, info)."""
        row, col = self._state_to_coord(self.state)
        dr, dc = self.action_vectors[action]

        new_row = max(0, min(self.height - 1, row + dr))
        new_col = max(0, min(self.width - 1, col + dc))
        next_state = self._coord_to_state(new_row, new_col)

        # Check cliff
        if next_state in self.cliff:
            self.state = self.start_state
            return self.state, -100.0, False, False, {}

        # Check goal
        if next_state == self.goal_state:
            self.state = next_state
            return self.state, -1.0, True, False, {}

        # Normal step
        self.state = next_state
        return self.state, -1.0, False, False, {}


def epsilon_greedy(Q: Dict, state: int, actions: List[int], epsilon: float) -> int:
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.choice(actions)
    else:
        q_values = [Q.get((state, a), 0.0) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return np.random.choice(best_actions)


def sarsa_cliff_walking(env, n_episodes: int = 500, alpha: float = 0.5,
                        epsilon: float = 0.1) -> Tuple[Dict, List]:
    """SARSA on Cliff Walking."""
    Q = defaultdict(float)
    rewards_history = []

    for episode in range(n_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Gymnasium returns (state, info)
            state = state[0]

        action = epsilon_greedy(Q, state, env.actions, epsilon)
        total_reward = 0
        done = False

        while not done:
            result = env.step(action)
            if len(result) == 5:  # Gymnasium format
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done = result[:3]

            total_reward += reward

            next_action = epsilon_greedy(Q, next_state, env.actions, epsilon)

            # SARSA update
            Q[(state, action)] += alpha * (
                reward + env.gamma * Q[(next_state, next_action)] - Q[(state, action)]
            )

            state = next_state
            action = next_action

        rewards_history.append(total_reward)

    return dict(Q), rewards_history


def q_learning_cliff_walking(env, n_episodes: int = 500, alpha: float = 0.5,
                             epsilon: float = 0.1) -> Tuple[Dict, List]:
    """Q-Learning on Cliff Walking."""
    Q = defaultdict(float)
    rewards_history = []

    for episode in range(n_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0
        done = False

        while not done:
            action = epsilon_greedy(Q, state, env.actions, epsilon)

            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done = result[:3]

            total_reward += reward

            # Q-Learning update
            max_next_q = max([Q[(next_state, a)] for a in env.actions])
            Q[(state, action)] += alpha * (
                reward + env.gamma * max_next_q - Q[(state, action)]
            )

            state = next_state

        rewards_history.append(total_reward)

    return dict(Q), rewards_history


def compare_sarsa_q_learning(n_runs: int = 50):
    """Compare SARSA and Q-Learning on Cliff Walking."""
    print("\n" + "="*60)
    print("CLIFF WALKING: SARSA vs Q-LEARNING")
    print("="*60)

    # Create environment
    if HAS_GYM:
        env = gym.make('CliffWalking-v0')
        env.gamma = 1.0
        env.actions = [0, 1, 2, 3]
    else:
        env = CliffWalkingEnv()

    n_episodes = 500
    sarsa_rewards = []
    q_learning_rewards = []

    print(f"\nRunning {n_runs} trials...")
    for run in range(n_runs):
        if run % 10 == 0:
            print(f"  Run {run + 1}/{n_runs}")

        _, s_rewards = sarsa_cliff_walking(env, n_episodes)
        _, q_rewards = q_learning_cliff_walking(env, n_episodes)

        sarsa_rewards.append(s_rewards)
        q_learning_rewards.append(q_rewards)

    # Convert to arrays
    sarsa_rewards = np.array(sarsa_rewards)
    q_learning_rewards = np.array(q_learning_rewards)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cliff Walking: SARSA vs Q-Learning', fontsize=16)

    # Plot 1: Learning curves
    ax = axes[0, 0]
    window = 20

    s_mean = sarsa_rewards.mean(axis=0)
    q_mean = q_learning_rewards.mean(axis=0)

    s_smoothed = np.convolve(s_mean, np.ones(window)/window, mode='valid')
    q_smoothed = np.convolve(q_mean, np.ones(window)/window, mode='valid')

    ax.plot(s_smoothed, label='SARSA (safe path)', linewidth=2)
    ax.plot(q_smoothed, label='Q-Learning (optimal path)', linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward (smoothed)', fontsize=12)
    ax.set_title('Learning Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Reward distribution
    ax = axes[0, 1]
    s_final = sarsa_rewards[:, -100:].mean(axis=1)
    q_final = q_learning_rewards[:, -100:].mean(axis=1)

    ax.hist(s_final, bins=20, alpha=0.7, label='SARSA', color='blue')
    ax.hist(q_final, bins=20, alpha=0.7, label='Q-Learning', color='orange')
    ax.set_xlabel('Average Reward (last 100 episodes)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Reward Distribution', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Train one more time and show paths
    ax = axes[1, 0]
    Q_sarsa, _ = sarsa_cliff_walking(env, n_episodes=500)
    Q_qlearn, _ = q_learning_cliff_walking(env, n_episodes=500)

    # Create grid for visualization
    grid = np.zeros((4, 12))
    grid[3, 1:11] = -1  # Cliff

    im = ax.imshow(grid, cmap='RdYlGn', interpolation='nearest', vmin=-1, vmax=1)

    # Draw SARSA path
    sarsa_path = extract_path(env, Q_sarsa)
    sarsa_rows, sarsa_cols = zip(*[env._state_to_coord(s) if hasattr(env, '_state_to_coord')
                                   else (s // 12, s % 12) for s in sarsa_path])
    ax.plot(sarsa_cols, sarsa_rows, 'b-o', linewidth=3, markersize=8, label='SARSA path')

    # Draw Q-Learning path
    qlearn_path = extract_path(env, Q_qlearn)
    qlearn_rows, qlearn_cols = zip(*[env._state_to_coord(s) if hasattr(env, '_state_to_coord')
                                     else (s // 12, s % 12) for s in qlearn_path])
    ax.plot(np.array(qlearn_cols) + 0.1, np.array(qlearn_rows) + 0.1, 'r-s',
            linewidth=3, markersize=8, label='Q-Learning path')

    ax.set_title('Learned Paths (greedy)', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Plot 4: Statistics
    ax = axes[1, 1]
    ax.axis('off')

    stats_text = f"""
    SARSA vs Q-Learning Results:

    Average Reward (last 100 episodes):
      SARSA:      {s_final.mean():.1f} ± {s_final.std():.1f}
      Q-Learning: {q_final.mean():.1f} ± {q_final.std():.1f}

    During Training (with ε-greedy):
      SARSA gets HIGHER rewards because it
      learns to avoid the cliff edge.

    Greedy Policy:
      Q-Learning's policy is OPTIMAL (shortest path)
      SARSA's policy is SAFER (away from cliff)

    The Insight:
      - Q-Learning learns optimal policy
      - But during training with exploration,
        it falls off the cliff more often!
      - SARSA accounts for exploration in its
        value estimates, so it's more cautious.
    """
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/06_td_control/cliff_walking_comparison.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved comparison to cliff_walking_comparison.png")
    plt.close()

    print(f"\nPerformance Summary:")
    print(f"  SARSA:      {s_final.mean():.1f} ± {s_final.std():.1f}")
    print(f"  Q-Learning: {q_final.mean():.1f} ± {q_final.std():.1f}")


def extract_path(env, Q: Dict, max_steps: int = 50) -> List[int]:
    """Extract greedy path from Q-values."""
    state = env.start_state if hasattr(env, 'start_state') else 36
    goal = env.goal_state if hasattr(env, 'goal_state') else 47
    actions = env.actions if hasattr(env, 'actions') else [0, 1, 2, 3]

    path = [state]
    steps = 0

    while state != goal and steps < max_steps:
        q_values = [Q.get((state, a), 0.0) for a in actions]
        action = actions[np.argmax(q_values)]

        # Take greedy action (simplified for path extraction)
        if hasattr(env, '_state_to_coord'):
            row, col = env._state_to_coord(state)
            vectors = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
            dr, dc = vectors[action]
            new_row = max(0, min(3, row + dr))
            new_col = max(0, min(11, col + dc))
            state = env._coord_to_state(new_row, new_col)
        else:
            # Gymnasium state transition
            row, col = state // 12, state % 12
            vectors = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
            dr, dc = vectors[action]
            new_row = max(0, min(3, row + dr))
            new_col = max(0, min(11, col + dc))
            state = new_row * 12 + new_col

        path.append(state)
        steps += 1

    return path


def visualize_q_values(env, Q_sarsa: Dict, Q_qlearn: Dict):
    """Visualize Q-values for both algorithms."""
    print("\n" + "="*60)
    print("Q-VALUE VISUALIZATION")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Q-values: SARSA vs Q-Learning', fontsize=16)

    actions = [0, 1, 2, 3]  # Up, Right, Down, Left
    action_names = ['Up', 'Right', 'Down', 'Left']

    for idx, (Q, name) in enumerate([(Q_sarsa, 'SARSA'), (Q_qlearn, 'Q-Learning')]):
        # Max Q for each state
        ax = axes[idx, 0]
        max_q_grid = np.zeros((4, 12))
        for row in range(4):
            for col in range(12):
                state = row * 12 + col
                max_q = max([Q.get((state, a), 0.0) for a in actions])
                max_q_grid[row, col] = max_q

        im = ax.imshow(max_q_grid, cmap='viridis', interpolation='nearest')
        ax.set_title(f'{name}: max Q(s,a)', fontsize=14)
        plt.colorbar(im, ax=ax)

        # Best action for each state
        ax = axes[idx, 1]
        ax.set_xlim(-0.5, 11.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        arrows = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}

        for row in range(4):
            for col in range(12):
                state = row * 12 + col

                # Skip cliff
                if row == 3 and 1 <= col <= 10:
                    ax.plot(col, row, 'rx', markersize=10)
                    continue

                # Skip goal
                if row == 3 and col == 11:
                    ax.plot(col, row, 'g*', markersize=15)
                    continue

                q_values = [Q.get((state, a), 0.0) for a in actions]
                best_action = actions[np.argmax(q_values)]
                dx, dy = arrows[best_action]
                ax.arrow(col, row, dx, dy, head_width=0.15, head_length=0.1,
                        fc='blue', ec='blue')

        ax.set_title(f'{name}: Best Actions', fontsize=14)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/06_td_control/cliff_q_values.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved Q-values to cliff_q_values.png")
    plt.close()


def main():
    """Main demonstration."""
    print("="*60)
    print("CLIFF WALKING: THE CLASSIC SARSA vs Q-LEARNING EXAMPLE")
    print("="*60)

    print("""
    Grid Layout (4 rows x 12 columns):
    [  ][  ][  ][  ][  ][  ][  ][  ][  ][  ][  ][  ]
    [  ][  ][  ][  ][  ][  ][  ][  ][  ][  ][  ][  ]
    [  ][  ][  ][  ][  ][  ][  ][  ][  ][  ][  ][  ]
    [S ][C ][C ][C ][C ][C ][C ][C ][C ][C ][C ][G ]

    S = Start (bottom-left)
    G = Goal (bottom-right)
    C = Cliff (reward -100, back to start)

    Optimal path: Along the cliff edge (shortest)
    Safe path: Up, across, down (avoids cliff)
    """)

    # Run comparison
    compare_sarsa_q_learning(n_runs=50)

    # Visualize Q-values
    env = CliffWalkingEnv()
    Q_sarsa, _ = sarsa_cliff_walking(env, n_episodes=500)
    Q_qlearn, _ = q_learning_cliff_walking(env, n_episodes=500)
    visualize_q_values(env, Q_sarsa, Q_qlearn)

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Insights:")
    print("  - SARSA learns SAFE path (accounts for exploration)")
    print("  - Q-Learning learns OPTIMAL path (assumes greedy future)")
    print("  - During training with ε-greedy, SARSA performs BETTER")
    print("  - Q-Learning's final policy is optimal but training is riskier")
    print("  - Use SARSA when exploration mistakes are costly!")


if __name__ == "__main__":
    main()
