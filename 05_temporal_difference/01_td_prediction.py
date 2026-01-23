"""
01 - TD(0) Prediction

Learn to estimate value functions using Temporal Difference learning.

Demonstrates:
- TD(0) prediction algorithm
- Policy evaluation with TD
- Incremental updates
- Convergence to true values
- Comparison with Monte Carlo
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt


class SimpleGridWorld:
    """
    Simple Grid World for TD prediction demonstration.

    5x5 grid, goal at top-right, start at bottom-left.
    """

    def __init__(self, size: int = 5, gamma: float = 0.9):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma

        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.action_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

        self.start_state = self.n_states - self.size
        self.goal_state = self.size - 1

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        """Take action, return (next_state, reward, done)."""
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
        """Reset to start state."""
        return self.start_state

    def render_values(self, V: Dict[int, float], title: str = "Value Function"):
        """Render value function as grid."""
        print(f"\n{title}:")
        print("+" + "--------+" * self.size)

        for row in range(self.size):
            line = "|"
            for col in range(self.size):
                state = self._coord_to_state(row, col)
                if state == self.goal_state:
                    cell = "  GOAL  "
                else:
                    v = V.get(state, 0.0)
                    cell = f" {v:6.3f} "
                line += cell + "|"
            print(line)
            print("+" + "--------+" * self.size)


def create_random_policy(n_states: int, n_actions: int) -> np.ndarray:
    """Create a random policy (equal probability for all actions)."""
    policy = np.ones((n_states, n_actions)) / n_actions
    return policy


def create_optimal_policy(env: SimpleGridWorld) -> np.ndarray:
    """Create a simple policy that moves toward the goal."""
    policy = np.zeros((env.n_states, len(env.actions)))

    for state in range(env.n_states):
        if state == env.goal_state:
            policy[state] = 1.0 / len(env.actions)
            continue

        row, col = env._state_to_coord(state)
        goal_row, goal_col = env._state_to_coord(env.goal_state)

        # Prefer actions that move toward goal
        if row > goal_row:
            policy[state][0] = 0.7  # Up
        elif row < goal_row:
            policy[state][2] = 0.7  # Down

        if col < goal_col:
            policy[state][1] = 0.7  # Right
        elif col > goal_col:
            policy[state][3] = 0.7  # Left

        # Normalize
        if policy[state].sum() == 0:
            policy[state] = 1.0 / len(env.actions)
        else:
            policy[state] = policy[state] / policy[state].sum()

    return policy


def td_prediction(env: SimpleGridWorld, policy: np.ndarray,
                  n_episodes: int = 500, alpha: float = 0.1) -> Tuple[Dict[int, float], List[float]]:
    """
    TD(0) Prediction: Estimate V^π using temporal difference learning.

    Args:
        env: GridWorld environment
        policy: Policy to evaluate (n_states x n_actions)
        n_episodes: Number of episodes to run
        alpha: Learning rate (step size)

    Returns:
        V: Estimated value function
        errors: RMS error per episode (if true values known)
    """
    V = defaultdict(float)
    errors = []

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            # Select action according to policy
            action = np.random.choice(env.actions, p=policy[state])

            # Take action
            next_state, reward, done = env.step(state, action)

            # TD(0) Update
            # V(S) ← V(S) + α[R + γV(S') - V(S)]
            td_target = reward + env.gamma * V[next_state]
            td_error = td_target - V[state]
            V[state] = V[state] + alpha * td_error

            state = next_state
            steps += 1

        # Track progress (optional)
        if episode % 100 == 0:
            print(f"Episode {episode}/{n_episodes}")

    return dict(V), errors


def mc_prediction(env: SimpleGridWorld, policy: np.ndarray,
                  n_episodes: int = 500, alpha: float = 0.1) -> Dict[int, float]:
    """
    Monte Carlo Prediction for comparison.

    First-visit MC with incremental updates.
    """
    V = defaultdict(float)

    for episode in range(n_episodes):
        # Generate episode
        episode_data = []
        state = env.reset()
        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            action = np.random.choice(env.actions, p=policy[state])
            next_state, reward, done = env.step(state, action)
            episode_data.append((state, reward))
            state = next_state
            steps += 1

        # Calculate returns and update (first-visit)
        visited = set()
        G = 0

        for t in range(len(episode_data) - 1, -1, -1):
            state, reward = episode_data[t]
            G = env.gamma * G + reward

            if state not in visited:
                visited.add(state)
                V[state] = V[state] + alpha * (G - V[state])

        if episode % 100 == 0:
            print(f"Episode {episode}/{n_episodes}")

    return dict(V)


def compare_td_mc(env: SimpleGridWorld, policy: np.ndarray,
                  n_episodes: int = 500, alpha: float = 0.1):
    """Compare TD(0) and MC prediction."""
    print("\n" + "="*60)
    print("Comparing TD(0) vs Monte Carlo Prediction")
    print("="*60)

    # Run TD
    print("\nRunning TD(0)...")
    V_td, _ = td_prediction(env, policy, n_episodes, alpha)

    # Run MC
    print("\nRunning Monte Carlo...")
    V_mc = mc_prediction(env, policy, n_episodes, alpha)

    # Display results
    env.render_values(V_td, "TD(0) Value Estimates")
    env.render_values(V_mc, "Monte Carlo Value Estimates")

    # Compare differences
    print("\nDifference (TD - MC):")
    diff = {s: V_td.get(s, 0) - V_mc.get(s, 0) for s in range(env.n_states)}
    env.render_values(diff, "TD minus MC")

    # Statistics
    differences = [abs(V_td.get(s, 0) - V_mc.get(s, 0)) for s in range(env.n_states)]
    print(f"\nMean absolute difference: {np.mean(differences):.4f}")
    print(f"Max absolute difference: {np.max(differences):.4f}")


def visualize_learning(env: SimpleGridWorld, policy: np.ndarray, n_runs: int = 10):
    """Visualize TD learning over multiple runs."""
    n_episodes = 300
    alpha_values = [0.01, 0.05, 0.1, 0.3]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('TD(0) Learning with Different Step Sizes', fontsize=14)

    for idx, alpha in enumerate(alpha_values):
        ax = axes[idx // 2, idx % 2]

        # Run multiple trials
        all_values = []
        for run in range(n_runs):
            V = defaultdict(float)
            value_history = []

            for episode in range(n_episodes):
                state = env.reset()
                done = False
                steps = 0

                while not done and steps < 100:
                    action = np.random.choice(env.actions, p=policy[state])
                    next_state, reward, done = env.step(state, action)

                    td_target = reward + env.gamma * V[next_state]
                    td_error = td_target - V[state]
                    V[state] = V[state] + alpha * td_error

                    state = next_state
                    steps += 1

                # Track average value
                avg_value = np.mean([V[s] for s in range(env.n_states)])
                value_history.append(avg_value)

            all_values.append(value_history)

        # Plot average and std
        all_values = np.array(all_values)
        mean_values = all_values.mean(axis=0)
        std_values = all_values.std(axis=0)

        ax.plot(mean_values, label=f'α={alpha}')
        ax.fill_between(range(n_episodes),
                        mean_values - std_values,
                        mean_values + std_values,
                        alpha=0.3)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Value')
        ax.set_title(f'Learning Rate α = {alpha}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/td_learning_curves.png', dpi=150, bbox_inches='tight')
    print("\nSaved learning curves to td_learning_curves.png")
    plt.close()


def main():
    """Main demonstration."""
    print("="*60)
    print("TD(0) PREDICTION DEMONSTRATION")
    print("="*60)

    # Create environment
    env = SimpleGridWorld(size=5, gamma=0.9)

    # Create policy
    print("\nUsing optimal policy (toward goal)...")
    policy = create_optimal_policy(env)

    # Run TD prediction
    print("\nRunning TD(0) Prediction...")
    V_td, _ = td_prediction(env, policy, n_episodes=500, alpha=0.1)

    # Display results
    env.render_values(V_td, "TD(0) Value Estimates")

    # Compare with MC
    compare_td_mc(env, policy, n_episodes=500, alpha=0.1)

    # Visualize learning
    print("\nGenerating learning curves...")
    visualize_learning(env, policy, n_runs=10)

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
