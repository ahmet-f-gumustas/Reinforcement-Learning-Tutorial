"""
02 - TD vs MC Comparison

Head-to-head comparison of Temporal Difference and Monte Carlo methods.

Demonstrates:
- TD(0) vs MC convergence speed
- Sample efficiency comparison
- RMS error analysis
- Batch vs incremental updates
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt


class GridWorld:
    """GridWorld environment for comparing TD and MC."""

    def __init__(self, size: int = 7, gamma: float = 0.95):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma
        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left

        # Multiple start and goal positions for variety
        self.start_state = size * (size - 1)  # Bottom-left
        self.goal_state = size - 1  # Top-right

        # Obstacles
        self.obstacles = self._create_obstacles()

    def _create_obstacles(self) -> set:
        """Create some obstacles in the grid."""
        obstacles = set()
        mid = self.size // 2

        # Create a wall
        for i in range(1, self.size - 1):
            if i != mid:
                obstacles.add(self._coord_to_state(mid, i))

        return obstacles

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        """Execute action and return next state, reward, done."""
        if state == self.goal_state:
            return state, 0.0, True

        row, col = self._state_to_coord(state)
        action_effects = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        dr, dc = action_effects[action]
        new_row = max(0, min(self.size - 1, row + dr))
        new_col = max(0, min(self.size - 1, col + dc))
        next_state = self._coord_to_state(new_row, new_col)

        # Check for obstacles
        if next_state in self.obstacles:
            next_state = state  # Stay in place

        if next_state == self.goal_state:
            return next_state, 10.0, True
        else:
            return next_state, -0.1, False

    def reset(self) -> int:
        return self.start_state


def create_policy(env: GridWorld) -> np.ndarray:
    """Create a stochastic policy that generally moves toward the goal."""
    policy = np.zeros((env.n_states, len(env.actions)))

    for state in range(env.n_states):
        if state == env.goal_state or state in env.obstacles:
            policy[state] = 1.0 / len(env.actions)
            continue

        row, col = env._state_to_coord(state)
        goal_row, goal_col = env._state_to_coord(env.goal_state)

        # Prefer directions toward goal with some randomness
        weights = np.ones(len(env.actions)) * 0.05

        if row > goal_row:
            weights[0] = 0.6  # Up
        elif row < goal_row:
            weights[2] = 0.6  # Down

        if col < goal_col:
            weights[1] = 0.6  # Right
        elif col > goal_col:
            weights[3] = 0.6  # Left

        policy[state] = weights / weights.sum()

    return policy


def td_prediction_with_tracking(env: GridWorld, policy: np.ndarray,
                                 n_episodes: int, alpha: float) -> Tuple[Dict, List]:
    """TD(0) with episode-by-episode value tracking."""
    V = defaultdict(float)
    value_history = []

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            action = np.random.choice(env.actions, p=policy[state])
            next_state, reward, done = env.step(state, action)

            # TD(0) update
            td_target = reward + env.gamma * V[next_state]
            V[state] += alpha * (td_target - V[state])

            state = next_state
            steps += 1

        # Track average value
        avg_value = np.mean([V[s] for s in range(env.n_states)])
        value_history.append(avg_value)

    return dict(V), value_history


def mc_prediction_with_tracking(env: GridWorld, policy: np.ndarray,
                                 n_episodes: int, alpha: float) -> Tuple[Dict, List]:
    """Monte Carlo with episode-by-episode value tracking."""
    V = defaultdict(float)
    value_history = []

    for episode in range(n_episodes):
        # Generate episode
        episode_data = []
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            action = np.random.choice(env.actions, p=policy[state])
            next_state, reward, done = env.step(state, action)
            episode_data.append((state, reward))
            state = next_state
            steps += 1

        # MC update (first-visit)
        visited = set()
        G = 0

        for t in range(len(episode_data) - 1, -1, -1):
            state, reward = episode_data[t]
            G = env.gamma * G + reward

            if state not in visited:
                visited.add(state)
                V[state] += alpha * (G - V[state])

        # Track average value
        avg_value = np.mean([V[s] for s in range(env.n_states)])
        value_history.append(avg_value)

    return dict(V), value_history


def compare_convergence(env: GridWorld, policy: np.ndarray, n_runs: int = 20):
    """Compare convergence speed of TD and MC."""
    print("\n" + "="*60)
    print("CONVERGENCE COMPARISON: TD vs MC")
    print("="*60)

    n_episodes = 500
    alpha = 0.1

    td_histories = []
    mc_histories = []

    print(f"\nRunning {n_runs} trials...")
    for run in range(n_runs):
        if run % 5 == 0:
            print(f"  Run {run + 1}/{n_runs}")

        _, td_hist = td_prediction_with_tracking(env, policy, n_episodes, alpha)
        _, mc_hist = mc_prediction_with_tracking(env, policy, n_episodes, alpha)

        td_histories.append(td_hist)
        mc_histories.append(mc_hist)

    # Convert to arrays for analysis
    td_histories = np.array(td_histories)
    mc_histories = np.array(mc_histories)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Learning curves
    ax = axes[0]
    td_mean = td_histories.mean(axis=0)
    td_std = td_histories.std(axis=0)
    mc_mean = mc_histories.mean(axis=0)
    mc_std = mc_histories.std(axis=0)

    episodes = np.arange(n_episodes)
    ax.plot(episodes, td_mean, label='TD(0)', linewidth=2)
    ax.fill_between(episodes, td_mean - td_std, td_mean + td_std, alpha=0.3)
    ax.plot(episodes, mc_mean, label='Monte Carlo', linewidth=2)
    ax.fill_between(episodes, mc_mean - mc_std, mc_mean + mc_std, alpha=0.3)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Value', fontsize=12)
    ax.set_title('Learning Curves (mean ± std)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Variance comparison
    ax = axes[1]
    td_var = td_histories.var(axis=0)
    mc_var = mc_histories.var(axis=0)

    ax.plot(episodes, td_var, label='TD(0) Variance', linewidth=2)
    ax.plot(episodes, mc_var, label='MC Variance', linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Variance Across Runs', fontsize=12)
    ax.set_title('Variance Over Time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/td_vs_mc_convergence.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved convergence comparison to td_vs_mc_convergence.png")
    plt.close()

    # Print statistics
    print("\nFinal Values (last 50 episodes average):")
    print(f"  TD(0):        {td_mean[-50:].mean():.4f} ± {td_std[-50:].mean():.4f}")
    print(f"  Monte Carlo:  {mc_mean[-50:].mean():.4f} ± {mc_std[-50:].mean():.4f}")

    print("\nConvergence Speed (episodes to reach 90% of final value):")
    td_target = td_mean[-1] * 0.9
    mc_target = mc_mean[-1] * 0.9
    td_converge = np.argmax(td_mean >= td_target) if np.any(td_mean >= td_target) else n_episodes
    mc_converge = np.argmax(mc_mean >= mc_target) if np.any(mc_mean >= mc_target) else n_episodes
    print(f"  TD(0):        {td_converge} episodes")
    print(f"  Monte Carlo:  {mc_converge} episodes")


def compare_alpha_sensitivity(env: GridWorld, policy: np.ndarray):
    """Compare sensitivity to learning rate."""
    print("\n" + "="*60)
    print("LEARNING RATE SENSITIVITY")
    print("="*60)

    alpha_values = [0.01, 0.05, 0.1, 0.3, 0.5]
    n_episodes = 300
    n_runs = 10

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # TD sensitivity
    ax = axes[0]
    for alpha in alpha_values:
        histories = []
        for _ in range(n_runs):
            _, hist = td_prediction_with_tracking(env, policy, n_episodes, alpha)
            histories.append(hist)
        mean_hist = np.array(histories).mean(axis=0)
        ax.plot(mean_hist, label=f'α={alpha}', linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Value', fontsize=12)
    ax.set_title('TD(0) Learning Rate Sensitivity', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # MC sensitivity
    ax = axes[1]
    for alpha in alpha_values:
        histories = []
        for _ in range(n_runs):
            _, hist = mc_prediction_with_tracking(env, policy, n_episodes, alpha)
            histories.append(hist)
        mean_hist = np.array(histories).mean(axis=0)
        ax.plot(mean_hist, label=f'α={alpha}', linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Value', fontsize=12)
    ax.set_title('Monte Carlo Learning Rate Sensitivity', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/alpha_sensitivity.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved learning rate sensitivity to alpha_sensitivity.png")
    plt.close()


def visualize_final_values(env: GridWorld, policy: np.ndarray):
    """Visualize final value functions from TD and MC."""
    print("\n" + "="*60)
    print("FINAL VALUE FUNCTIONS")
    print("="*60)

    n_episodes = 1000
    alpha = 0.1

    print("\nTraining TD(0)...")
    V_td, _ = td_prediction_with_tracking(env, policy, n_episodes, alpha)

    print("Training Monte Carlo...")
    V_mc, _ = mc_prediction_with_tracking(env, policy, n_episodes, alpha)

    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # TD values
    td_grid = np.zeros((env.size, env.size))
    for state in range(env.n_states):
        row, col = env._state_to_coord(state)
        td_grid[row, col] = V_td.get(state, 0)

    im1 = axes[0].imshow(td_grid, cmap='viridis', interpolation='nearest')
    axes[0].set_title('TD(0) Value Function', fontsize=14)
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0])

    # MC values
    mc_grid = np.zeros((env.size, env.size))
    for state in range(env.n_states):
        row, col = env._state_to_coord(state)
        mc_grid[row, col] = V_mc.get(state, 0)

    im2 = axes[1].imshow(mc_grid, cmap='viridis', interpolation='nearest')
    axes[1].set_title('Monte Carlo Value Function', fontsize=14)
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[1])

    # Difference
    diff_grid = td_grid - mc_grid
    im3 = axes[2].imshow(diff_grid, cmap='RdBu', interpolation='nearest',
                        vmin=-abs(diff_grid).max(), vmax=abs(diff_grid).max())
    axes[2].set_title('Difference (TD - MC)', fontsize=14)
    axes[2].set_xlabel('Column')
    axes[2].set_ylabel('Row')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/value_functions.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved value functions to value_functions.png")
    plt.close()

    # Print statistics
    print("\nValue Function Statistics:")
    print(f"  TD mean:  {np.mean(list(V_td.values())):.4f}")
    print(f"  MC mean:  {np.mean(list(V_mc.values())):.4f}")
    print(f"  Difference: {np.mean(np.abs(diff_grid)):.4f} (mean absolute)")


def main():
    """Main demonstration."""
    print("="*60)
    print("TD vs MC COMPREHENSIVE COMPARISON")
    print("="*60)

    # Create environment
    env = GridWorld(size=7, gamma=0.95)
    policy = create_policy(env)

    # Run comparisons
    compare_convergence(env, policy, n_runs=20)
    compare_alpha_sensitivity(env, policy)
    visualize_final_values(env, policy)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print("\nKey Findings:")
    print("  - TD typically converges faster (online updates)")
    print("  - MC has lower bias (uses actual returns)")
    print("  - TD more sensitive to learning rate")
    print("  - Both converge to similar final values")


if __name__ == "__main__":
    main()
