"""
05 - Linear SARSA (Semi-gradient SARSA)

Control with linear function approximation using semi-gradient SARSA.

Demonstrates:
- Q-value function approximation
- Semi-gradient SARSA algorithm
- Feature design for state-action pairs
- Policy learning with function approximation
"""

import numpy as np
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt


class SimpleGridWorld:
    """Simple GridWorld for linear SARSA demonstration."""

    def __init__(self, size: int = 5, gamma: float = 0.99):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma

        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        self.n_actions = 4
        self.actions = list(range(4))
        self.action_vectors = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        self.start_state = self._coord_to_state(size - 1, 0)
        self.goal_state = self._coord_to_state(0, size - 1)

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def reset(self) -> int:
        return self.start_state

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        if state == self.goal_state:
            return state, 0, True

        row, col = self._state_to_coord(state)
        dr, dc = self.action_vectors[action]

        new_row = max(0, min(self.size - 1, row + dr))
        new_col = max(0, min(self.size - 1, col + dc))
        next_state = self._coord_to_state(new_row, new_col)

        if next_state == self.goal_state:
            return next_state, 10.0, True
        else:
            return next_state, -0.1, False

    def state_to_features(self, state: int) -> np.ndarray:
        """Convert state to normalized features."""
        row, col = self._state_to_coord(state)
        return np.array([
            row / (self.size - 1),  # Normalized row
            col / (self.size - 1),  # Normalized col
        ])


class LinearQFunction:
    """
    Linear Q-value function approximation.

    Q(s, a) = w_a^T x(s)  (separate weights per action)
    """

    def __init__(self, n_state_features: int, n_actions: int):
        self.n_state_features = n_state_features
        self.n_actions = n_actions

        # Separate weight vector for each action
        self.weights = np.zeros((n_actions, n_state_features))

    def predict(self, state_features: np.ndarray, action: int) -> float:
        """Predict Q(s, a)."""
        return np.dot(self.weights[action], state_features)

    def predict_all(self, state_features: np.ndarray) -> np.ndarray:
        """Predict Q(s, a) for all actions."""
        return np.array([self.predict(state_features, a) for a in range(self.n_actions)])

    def update(self, state_features: np.ndarray, action: int,
               target: float, alpha: float):
        """Update weights using semi-gradient."""
        prediction = self.predict(state_features, action)
        error = target - prediction
        self.weights[action] += alpha * error * state_features


class TileCodedQFunction:
    """
    Q-function with tile coding features.

    Better generalization than simple linear features.
    """

    def __init__(self, n_tilings: int, n_tiles: int,
                 state_ranges: List[Tuple[float, float]], n_actions: int):
        self.n_tilings = n_tilings
        self.n_tiles = n_tiles
        self.state_ranges = state_ranges
        self.n_actions = n_actions
        self.n_dims = len(state_ranges)

        # Total features per action
        self.features_per_action = n_tilings * (n_tiles ** self.n_dims)
        self.total_features = self.features_per_action * n_actions

        self.weights = np.zeros(self.total_features)

        # Precompute tile widths and offsets
        self.tile_widths = [
            (r[1] - r[0]) / n_tiles for r in state_ranges
        ]
        np.random.seed(42)
        self.offsets = [
            [np.random.random() * self.tile_widths[d] for d in range(self.n_dims)]
            for _ in range(n_tilings)
        ]

    def get_active_features(self, state: np.ndarray, action: int) -> np.ndarray:
        """Get indices of active features."""
        active = []
        base_offset = action * self.features_per_action

        for tiling in range(self.n_tilings):
            tile_indices = []
            for d in range(self.n_dims):
                shifted = state[d] - self.state_ranges[d][0] - self.offsets[tiling][d]
                tile_idx = int(shifted / self.tile_widths[d])
                tile_idx = max(0, min(self.n_tiles - 1, tile_idx))
                tile_indices.append(tile_idx)

            # Convert multi-dim index to flat index
            flat_idx = 0
            for d, idx in enumerate(tile_indices):
                flat_idx = flat_idx * self.n_tiles + idx

            feature_idx = base_offset + tiling * (self.n_tiles ** self.n_dims) + flat_idx
            active.append(feature_idx)

        return np.array(active)

    def predict(self, state: np.ndarray, action: int) -> float:
        """Predict Q(s, a)."""
        active = self.get_active_features(state, action)
        return np.sum(self.weights[active])

    def predict_all(self, state: np.ndarray) -> np.ndarray:
        """Predict Q(s, a) for all actions."""
        return np.array([self.predict(state, a) for a in range(self.n_actions)])

    def update(self, state: np.ndarray, action: int, target: float, alpha: float):
        """Update weights."""
        active = self.get_active_features(state, action)
        prediction = np.sum(self.weights[active])
        error = target - prediction
        self.weights[active] += alpha * error / self.n_tilings


def epsilon_greedy(q_values: np.ndarray, epsilon: float) -> int:
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(len(q_values))
    return np.argmax(q_values)


def semi_gradient_sarsa(env, q_fn, n_episodes: int, alpha: float,
                        epsilon: float, epsilon_decay: float = 0.995) -> Tuple[List[float], List[int]]:
    """
    Semi-gradient SARSA with function approximation.

    Update: w ← w + α [R + γQ(S',A',w) - Q(S,A,w)] ∇Q(S,A,w)
    """
    rewards_history = []
    steps_history = []
    current_epsilon = epsilon

    for episode in range(n_episodes):
        state = env.reset()
        state_features = env.state_to_features(state)

        # Choose initial action
        q_values = q_fn.predict_all(state_features)
        action = epsilon_greedy(q_values, current_epsilon)

        total_reward = 0
        steps = 0

        while steps < 500:
            # Take action
            next_state, reward, done = env.step(state, action)
            total_reward += reward
            steps += 1

            if done:
                # Terminal update
                target = reward
                q_fn.update(state_features, action, target, alpha)
                break

            # Get next state features and action
            next_features = env.state_to_features(next_state)
            next_q_values = q_fn.predict_all(next_features)
            next_action = epsilon_greedy(next_q_values, current_epsilon)

            # SARSA update
            target = reward + env.gamma * q_fn.predict(next_features, next_action)
            q_fn.update(state_features, action, target, alpha)

            state = next_state
            state_features = next_features
            action = next_action

        rewards_history.append(total_reward)
        steps_history.append(steps)
        current_epsilon *= epsilon_decay

        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Steps: {steps}, ε: {current_epsilon:.3f}")

    return rewards_history, steps_history


def compare_feature_representations():
    """Compare simple linear vs tile coding."""
    print("\n" + "="*60)
    print("COMPARING FEATURE REPRESENTATIONS")
    print("="*60)

    env = SimpleGridWorld(size=5, gamma=0.99)
    n_episodes = 500
    n_runs = 10

    # Simple linear features
    simple_rewards = []
    print("\nTesting simple linear features...")
    for run in range(n_runs):
        q_fn = LinearQFunction(n_state_features=2, n_actions=4)
        rewards, _ = semi_gradient_sarsa(env, q_fn, n_episodes, alpha=0.01,
                                         epsilon=0.1, epsilon_decay=0.995)
        simple_rewards.append(rewards)

    # Tile coding features
    tile_rewards = []
    print("\nTesting tile coding features...")
    for run in range(n_runs):
        q_fn = TileCodedQFunction(
            n_tilings=8, n_tiles=4,
            state_ranges=[(0, 1), (0, 1)],
            n_actions=4
        )
        rewards, _ = semi_gradient_sarsa(env, q_fn, n_episodes, alpha=0.1,
                                         epsilon=0.1, epsilon_decay=0.995)
        tile_rewards.append(rewards)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curves
    ax = axes[0]
    window = 20

    simple_mean = np.array(simple_rewards).mean(axis=0)
    tile_mean = np.array(tile_rewards).mean(axis=0)

    simple_smooth = np.convolve(simple_mean, np.ones(window)/window, mode='valid')
    tile_smooth = np.convolve(tile_mean, np.ones(window)/window, mode='valid')

    ax.plot(simple_smooth, label='Simple Linear', linewidth=2)
    ax.plot(tile_smooth, label='Tile Coding', linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Learning Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Final performance
    ax = axes[1]
    simple_final = np.array(simple_rewards)[:, -50:].mean(axis=1)
    tile_final = np.array(tile_rewards)[:, -50:].mean(axis=1)

    ax.boxplot([simple_final, tile_final], labels=['Simple Linear', 'Tile Coding'])
    ax.set_ylabel('Final Reward (last 50 episodes)', fontsize=12)
    ax.set_title('Final Performance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/sarsa_features.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved comparison to sarsa_features.png")
    plt.close()


def visualize_learned_policy(env, q_fn):
    """Visualize the learned policy and Q-values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Q-value heatmap (max Q per state)
    ax = axes[0]
    q_grid = np.zeros((env.size, env.size))

    for state in range(env.n_states):
        row, col = env._state_to_coord(state)
        features = env.state_to_features(state)
        q_values = q_fn.predict_all(features)
        q_grid[row, col] = np.max(q_values)

    im = ax.imshow(q_grid, cmap='viridis', interpolation='nearest')
    ax.set_title('Max Q-value per State', fontsize=14)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax)

    # Mark start and goal
    start_row, start_col = env._state_to_coord(env.start_state)
    goal_row, goal_col = env._state_to_coord(env.goal_state)
    ax.plot(start_col, start_row, 'rs', markersize=15)
    ax.plot(goal_col, goal_row, 'g*', markersize=20)

    # Policy arrows
    ax = axes[1]
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(-0.5, env.size - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    arrows = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}

    for state in range(env.n_states):
        row, col = env._state_to_coord(state)

        if state == env.goal_state:
            ax.plot(col, row, 'g*', markersize=20)
            continue

        features = env.state_to_features(state)
        q_values = q_fn.predict_all(features)
        best_action = np.argmax(q_values)

        dx, dy = arrows[best_action]
        ax.arrow(col, row, dx, dy, head_width=0.15, head_length=0.1,
                fc='blue', ec='blue')

    ax.set_title('Learned Policy', fontsize=14)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/sarsa_policy.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved policy to sarsa_policy.png")
    plt.close()


def main():
    """Main demonstration."""
    print("="*60)
    print("LINEAR SARSA (SEMI-GRADIENT SARSA)")
    print("="*60)

    print("""
    Semi-gradient SARSA with Function Approximation:
    ------------------------------------------------
    For each step in episode:
        1. In state S, take action A using ε-greedy from Q(S,·,w)
        2. Observe R, S'
        3. Choose A' using ε-greedy from Q(S',·,w)
        4. Update: w ← w + α [R + γQ(S',A',w) - Q(S,A,w)] ∇Q(S,A,w)
        5. S ← S', A ← A'

    Q-value representation options:
        - Separate weights per action: Q(s,a) = w_a^T x(s)
        - State-action features: Q(s,a) = w^T x(s,a)
        - Tile coding: Binary features with good generalization
    """)

    # Compare feature representations
    compare_feature_representations()

    # Train and visualize final policy
    print("\n" + "="*60)
    print("TRAINING FINAL POLICY")
    print("="*60)

    env = SimpleGridWorld(size=5, gamma=0.99)
    q_fn = TileCodedQFunction(
        n_tilings=8, n_tiles=4,
        state_ranges=[(0, 1), (0, 1)],
        n_actions=4
    )

    rewards, steps = semi_gradient_sarsa(env, q_fn, n_episodes=500, alpha=0.1,
                                         epsilon=0.2, epsilon_decay=0.99)

    visualize_learned_policy(env, q_fn)

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Concepts:")
    print("  - Q(s,a) can be approximated with linear functions")
    print("  - Tile coding provides good generalization")
    print("  - Semi-gradient updates are stable for on-policy learning")
    print("  - Feature design is crucial for performance")


if __name__ == "__main__":
    main()
