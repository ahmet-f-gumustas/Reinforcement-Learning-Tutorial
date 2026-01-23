"""
06 - Windy GridWorld

Stochastic environment with wind effects for testing TD robustness.

Demonstrates:
- Stochastic state transitions
- TD prediction under noise
- Policy evaluation in uncertain environment
- Visualization of learned values and paths
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt


class WindyGridWorld:
    """
    GridWorld with wind that pushes the agent upward.

    Wind strength varies by column, creating stochastic transitions.
    """

    def __init__(self, height: int = 7, width: int = 10, gamma: float = 0.95):
        self.height = height
        self.width = width
        self.n_states = height * width
        self.gamma = gamma

        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        self.actions = [0, 1, 2, 3]
        self.action_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

        # Wind strength for each column (pushes up)
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        # Start and goal positions
        self.start_state = self._coord_to_state(3, 0)
        self.goal_state = self._coord_to_state(3, 7)

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        """Convert state index to (row, col) coordinates."""
        return state // self.width, state % self.width

    def _coord_to_state(self, row: int, col: int) -> int:
        """Convert (row, col) to state index."""
        return row * self.width + col

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        """
        Execute action with wind effects.

        Returns:
            next_state, reward, done
        """
        if state == self.goal_state:
            return state, 0.0, True

        row, col = self._state_to_coord(state)

        # Action effect
        action_effects = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dr, dc = action_effects[action]

        # Apply action
        new_row = row + dr
        new_col = col + dc

        # Apply wind (push upward)
        wind_strength = self.wind[col]
        new_row -= wind_strength

        # Add stochasticity (wind is not perfectly consistent)
        if np.random.random() < 0.1:  # 10% chance of extra push
            new_row -= 1

        # Clip to valid range
        new_row = max(0, min(self.height - 1, new_row))
        new_col = max(0, min(self.width - 1, new_col))

        next_state = self._coord_to_state(new_row, new_col)

        # Reward
        if next_state == self.goal_state:
            return next_state, 10.0, True
        else:
            return next_state, -1.0, False

    def reset(self) -> int:
        """Reset to start state."""
        return self.start_state

    def render(self, V: Dict[int, float] = None, path: List[int] = None):
        """Render the gridworld with optional values and path."""
        grid = np.zeros((self.height, self.width))

        if V is not None:
            for state in range(self.n_states):
                row, col = self._state_to_coord(state)
                grid[row, col] = V.get(state, 0)

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot values
        im = ax.imshow(grid, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Value')

        # Mark start and goal
        start_row, start_col = self._state_to_coord(self.start_state)
        goal_row, goal_col = self._state_to_coord(self.goal_state)

        ax.plot(start_col, start_row, 'rs', markersize=15, label='Start')
        ax.plot(goal_col, goal_row, 'g*', markersize=20, label='Goal')

        # Draw wind arrows
        for col in range(self.width):
            if self.wind[col] > 0:
                for row in range(self.height):
                    ax.arrow(col, row, 0, -0.3 * self.wind[col],
                            head_width=0.15, head_length=0.1,
                            fc='white', ec='white', alpha=0.6)

        # Draw path if provided
        if path is not None:
            path_rows = []
            path_cols = []
            for state in path:
                row, col = self._state_to_coord(state)
                path_rows.append(row)
                path_cols.append(col)
            ax.plot(path_cols, path_rows, 'r-', linewidth=2, alpha=0.7, label='Path')

        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_title('Windy GridWorld', fontsize=14)
        ax.legend(fontsize=11)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.grid(True, alpha=0.3)

        return fig


def create_policy(env: WindyGridWorld) -> np.ndarray:
    """Create a policy that generally moves toward the goal."""
    policy = np.zeros((env.n_states, len(env.actions)))

    for state in range(env.n_states):
        if state == env.goal_state:
            policy[state] = 1.0 / len(env.actions)
            continue

        row, col = env._state_to_coord(state)
        goal_row, goal_col = env._state_to_coord(env.goal_state)

        # Prefer actions toward goal, accounting for wind
        weights = np.ones(len(env.actions)) * 0.05

        # Horizontal preference
        if col < goal_col:
            weights[1] = 0.5  # Right
        elif col > goal_col:
            weights[3] = 0.5  # Left

        # Vertical preference (consider wind)
        wind_strength = env.wind[col]
        if row > goal_row:
            weights[0] = 0.4  # Up
        elif row < goal_row:
            weights[2] = 0.4 - 0.1 * wind_strength  # Down (less if wind is strong)

        policy[state] = weights / weights.sum()

    return policy


def td_prediction_windy(env: WindyGridWorld, policy: np.ndarray,
                        n_episodes: int = 500, alpha: float = 0.1) -> Tuple[Dict, List]:
    """TD(0) prediction for windy gridworld."""
    V = defaultdict(float)
    value_history = []

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            # Select action
            action = np.random.choice(env.actions, p=policy[state])

            # Take action
            next_state, reward, done = env.step(state, action)

            # TD(0) update
            V[state] += alpha * (reward + env.gamma * V[next_state] - V[state])

            state = next_state
            steps += 1

        # Track average value
        avg_value = np.mean([V[s] for s in range(env.n_states)])
        value_history.append(avg_value)

        if episode % 100 == 0:
            print(f"Episode {episode}/{n_episodes}")

    return dict(V), value_history


def generate_episode_path(env: WindyGridWorld, policy: np.ndarray) -> List[int]:
    """Generate one episode and return the path taken."""
    path = []
    state = env.reset()
    done = False
    steps = 0

    while not done and steps < 200:
        path.append(state)
        action = np.random.choice(env.actions, p=policy[state])
        next_state, reward, done = env.step(state, action)
        state = next_state
        steps += 1

    path.append(state)
    return path


def visualize_learning(env: WindyGridWorld, policy: np.ndarray):
    """Visualize TD learning process."""
    print("\n" + "="*60)
    print("VISUALIZING LEARNING PROCESS")
    print("="*60)

    n_episodes = 500
    alpha = 0.1

    print(f"\nTraining for {n_episodes} episodes...")
    V, value_history = td_prediction_windy(env, policy, n_episodes, alpha)

    # Plot learning curve
    plt.figure(figsize=(12, 5))
    plt.plot(value_history, linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Value', fontsize=12)
    plt.title('TD Learning on Windy GridWorld', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/windy_learning.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved learning curve to windy_learning.png")
    plt.close()

    # Visualize final values
    fig = env.render(V)
    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/windy_values.png',
                dpi=150, bbox_inches='tight')
    print("Saved value function to windy_values.png")
    plt.close()

    # Generate and visualize a sample path
    print("\nGenerating sample episode...")
    path = generate_episode_path(env, policy)
    print(f"Episode length: {len(path)} steps")

    fig = env.render(V, path)
    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/windy_path.png',
                dpi=150, bbox_inches='tight')
    print("Saved episode path to windy_path.png")
    plt.close()


def compare_different_alphas(env: WindyGridWorld, policy: np.ndarray):
    """Compare learning with different step sizes."""
    print("\n" + "="*60)
    print("COMPARING LEARNING RATES")
    print("="*60)

    alpha_values = [0.01, 0.05, 0.1, 0.3, 0.5]
    n_episodes = 300
    n_runs = 10

    plt.figure(figsize=(12, 6))

    for alpha in alpha_values:
        print(f"  Testing α={alpha}...")
        all_histories = []

        for run in range(n_runs):
            _, history = td_prediction_windy(env, policy, n_episodes, alpha)
            all_histories.append(history)

        mean_history = np.array(all_histories).mean(axis=0)
        std_history = np.array(all_histories).std(axis=0)

        plt.plot(mean_history, label=f'α={alpha}', linewidth=2)
        plt.fill_between(range(n_episodes),
                        mean_history - std_history,
                        mean_history + std_history,
                        alpha=0.2)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Value', fontsize=12)
    plt.title('Learning Rate Comparison on Windy GridWorld', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/windy_alpha_comparison.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved alpha comparison to windy_alpha_comparison.png")
    plt.close()


def analyze_stochasticity_impact(env: WindyGridWorld, policy: np.ndarray):
    """Analyze how stochasticity affects learning."""
    print("\n" + "="*60)
    print("ANALYZING STOCHASTICITY IMPACT")
    print("="*60)

    n_episodes = 200
    n_runs = 20
    alpha = 0.1

    # Run multiple trials
    all_histories = []
    final_values = []

    print(f"Running {n_runs} trials...")
    for run in range(n_runs):
        V, history = td_prediction_windy(env, policy, n_episodes, alpha)
        all_histories.append(history)

        # Record final values for key states
        start_row, start_col = env._state_to_coord(env.start_state)
        final_values.append(V[env.start_state])

    # Plot variance over time
    all_histories = np.array(all_histories)
    mean_history = all_histories.mean(axis=0)
    std_history = all_histories.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curve with variance
    ax = axes[0]
    ax.plot(mean_history, linewidth=2, label='Mean')
    ax.fill_between(range(n_episodes),
                    mean_history - std_history,
                    mean_history + std_history,
                    alpha=0.3, label='±1 std')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Value', fontsize=12)
    ax.set_title('Learning Under Stochasticity', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Variance over time
    ax = axes[1]
    variance = all_histories.var(axis=0)
    ax.plot(variance, linewidth=2, color='red')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Variance Across Runs', fontsize=12)
    ax.set_title('Variance Evolution', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/windy_stochasticity.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved stochasticity analysis to windy_stochasticity.png")
    plt.close()

    print(f"\nFinal value at start state:")
    print(f"  Mean: {np.mean(final_values):.2f}")
    print(f"  Std:  {np.std(final_values):.2f}")


def main():
    """Main demonstration."""
    print("="*60)
    print("WINDY GRIDWORLD DEMONSTRATION")
    print("="*60)

    # Create environment
    env = WindyGridWorld(height=7, width=10, gamma=0.95)
    policy = create_policy(env)

    print(f"\nEnvironment: {env.height}x{env.width} grid")
    print(f"Wind strengths: {env.wind}")
    print(f"Start: {env._state_to_coord(env.start_state)}")
    print(f"Goal:  {env._state_to_coord(env.goal_state)}")

    # Run experiments
    visualize_learning(env, policy)
    compare_different_alphas(env, policy)
    analyze_stochasticity_impact(env, policy)

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Insights:")
    print("  - TD handles stochastic transitions well")
    print("  - Wind creates challenging dynamics")
    print("  - Larger α converges faster but with more variance")
    print("  - Policy must account for wind effects")


if __name__ == "__main__":
    main()
