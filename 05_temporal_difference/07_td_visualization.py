"""
07 - TD Visualization Tools

Comprehensive visualization tools for understanding TD learning.

Demonstrates:
- Value function evolution over time
- TD error tracking and analysis
- Bias-variance tradeoff
- Interactive parameter exploration
- Learning dynamics visualization
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches


class SimpleGridWorld:
    """Simple GridWorld for visualization."""

    def __init__(self, size: int = 5, gamma: float = 0.9):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma
        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.start_state = size * (size - 1)
        self.goal_state = size - 1

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
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
        return self.start_state


def create_policy(env: SimpleGridWorld) -> np.ndarray:
    """Create a simple policy."""
    policy = np.zeros((env.n_states, len(env.actions)))
    for state in range(env.n_states):
        if state == env.goal_state:
            policy[state] = 1.0 / len(env.actions)
            continue

        row, col = env._state_to_coord(state)
        goal_row, goal_col = env._state_to_coord(env.goal_state)
        weights = np.ones(len(env.actions)) * 0.1

        if row > goal_row:
            weights[0] = 0.5
        elif row < goal_row:
            weights[2] = 0.5
        if col < goal_col:
            weights[1] = 0.5
        elif col > goal_col:
            weights[3] = 0.5

        policy[state] = weights / weights.sum()

    return policy


def td_prediction_with_full_tracking(env: SimpleGridWorld, policy: np.ndarray,
                                     n_episodes: int, alpha: float) -> Dict:
    """TD prediction with comprehensive tracking."""
    V = defaultdict(float)

    # Tracking
    value_snapshots = []  # Value function at each episode
    td_errors = []  # TD errors over time
    episode_lengths = []
    updates_per_state = defaultdict(int)

    for episode in range(n_episodes):
        episode_td_errors = []
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 100:
            action = np.random.choice(env.actions, p=policy[state])
            next_state, reward, done = env.step(state, action)

            # Calculate and store TD error
            td_error = reward + env.gamma * V[next_state] - V[state]
            episode_td_errors.append(abs(td_error))

            # Update
            V[state] += alpha * td_error
            updates_per_state[state] += 1

            state = next_state
            steps += 1

        episode_lengths.append(steps)
        td_errors.append(np.mean(episode_td_errors) if episode_td_errors else 0)

        # Save snapshot every few episodes
        if episode % 5 == 0:
            snapshot = np.array([V[s] for s in range(env.n_states)])
            value_snapshots.append(snapshot)

    return {
        'V': dict(V),
        'value_snapshots': value_snapshots,
        'td_errors': td_errors,
        'episode_lengths': episode_lengths,
        'updates_per_state': dict(updates_per_state)
    }


def visualize_value_evolution(env: SimpleGridWorld, policy: np.ndarray):
    """Visualize how value function evolves over training."""
    print("\n" + "="*60)
    print("VALUE FUNCTION EVOLUTION")
    print("="*60)

    n_episodes = 200
    alpha = 0.1

    print(f"\nTraining for {n_episodes} episodes...")
    results = td_prediction_with_full_tracking(env, policy, n_episodes, alpha)

    snapshots = results['value_snapshots']

    # Select key episodes to visualize
    n_plots = min(6, len(snapshots))
    indices = np.linspace(0, len(snapshots) - 1, n_plots, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Value Function Evolution During TD Learning', fontsize=16)

    for idx, snapshot_idx in enumerate(indices):
        ax = axes[idx // 3, idx % 3]
        episode_num = snapshot_idx * 5

        # Reshape to grid
        values = snapshots[snapshot_idx].reshape(env.size, env.size)

        im = ax.imshow(values, cmap='viridis', interpolation='nearest')
        ax.set_title(f'Episode {episode_num}', fontsize=12)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax)

        # Mark start and goal
        start_row, start_col = env._state_to_coord(env.start_state)
        goal_row, goal_col = env._state_to_coord(env.goal_state)
        ax.plot(start_col, start_row, 'rs', markersize=10)
        ax.plot(goal_col, goal_row, 'g*', markersize=15)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/value_evolution.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved value evolution to value_evolution.png")
    plt.close()


def visualize_td_error_dynamics(env: SimpleGridWorld, policy: np.ndarray):
    """Visualize TD error over time."""
    print("\n" + "="*60)
    print("TD ERROR DYNAMICS")
    print("="*60)

    n_episodes = 300
    alpha_values = [0.05, 0.1, 0.3]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TD Error Dynamics', fontsize=16)

    # Plot 1: TD error over episodes for different alphas
    ax = axes[0, 0]
    for alpha in alpha_values:
        print(f"  Testing α={alpha}...")
        results = td_prediction_with_full_tracking(env, policy, n_episodes, alpha)
        td_errors = results['td_errors']

        # Smooth with moving average
        window = 10
        smoothed = np.convolve(td_errors, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=f'α={alpha}', linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average |TD Error|', fontsize=12)
    ax.set_title('TD Error vs Learning Rate', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Episode length over time
    ax = axes[0, 1]
    results = td_prediction_with_full_tracking(env, policy, n_episodes, 0.1)
    episode_lengths = results['episode_lengths']
    window = 20
    smoothed_lengths = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
    ax.plot(smoothed_lengths, linewidth=2, color='green')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Length (steps)', fontsize=12)
    ax.set_title('Episode Length Over Time', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Plot 3: Updates per state
    ax = axes[1, 0]
    updates = results['updates_per_state']
    update_grid = np.zeros((env.size, env.size))
    for state, count in updates.items():
        row, col = env._state_to_coord(state)
        update_grid[row, col] = count

    im = ax.imshow(update_grid, cmap='hot', interpolation='nearest')
    ax.set_title('Number of Updates Per State', fontsize=14)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax)

    # Plot 4: TD error distribution
    ax = axes[1, 1]
    td_errors = results['td_errors']
    ax.hist(td_errors, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('|TD Error|', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('TD Error Distribution', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/td_error_dynamics.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved TD error dynamics to td_error_dynamics.png")
    plt.close()


def visualize_bias_variance_tradeoff(env: SimpleGridWorld, policy: np.ndarray):
    """Visualize bias-variance tradeoff for different alpha values."""
    print("\n" + "="*60)
    print("BIAS-VARIANCE TRADEOFF")
    print("="*60)

    alpha_values = np.linspace(0.01, 0.5, 10)
    n_episodes = 150
    n_runs = 30

    biases = []
    variances = []

    # Compute "true" values with long training
    print("\nComputing reference values...")
    true_results = td_prediction_with_full_tracking(env, policy, 1000, 0.05)
    true_V = true_results['V']

    print(f"\nTesting {len(alpha_values)} alpha values...")
    for alpha in alpha_values:
        all_final_values = []

        for run in range(n_runs):
            results = td_prediction_with_full_tracking(env, policy, n_episodes, alpha)
            V = results['V']

            # Collect final values
            final_values = np.array([V[s] for s in range(env.n_states)])
            all_final_values.append(final_values)

        all_final_values = np.array(all_final_values)

        # Calculate bias and variance
        mean_values = all_final_values.mean(axis=0)
        true_values = np.array([true_V[s] for s in range(env.n_states)])

        bias = np.mean((mean_values - true_values) ** 2)
        variance = np.mean(all_final_values.var(axis=0))

        biases.append(bias)
        variances.append(variance)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bias and variance
    ax = axes[0]
    ax.plot(alpha_values, biases, 'o-', label='Bias²', linewidth=2, markersize=6)
    ax.plot(alpha_values, variances, 's-', label='Variance', linewidth=2, markersize=6)
    ax.plot(alpha_values, np.array(biases) + np.array(variances),
            '^-', label='Total Error (Bias² + Var)', linewidth=2, markersize=6)
    ax.set_xlabel('Learning Rate (α)', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Bias-Variance Tradeoff', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Optimal alpha
    total_error = np.array(biases) + np.array(variances)
    optimal_alpha = alpha_values[np.argmin(total_error)]
    ax.axvline(x=optimal_alpha, color='r', linestyle='--', linewidth=2,
               label=f'Optimal α={optimal_alpha:.3f}')

    # Log scale
    ax = axes[1]
    ax.semilogy(alpha_values, biases, 'o-', label='Bias²', linewidth=2, markersize=6)
    ax.semilogy(alpha_values, variances, 's-', label='Variance', linewidth=2, markersize=6)
    ax.semilogy(alpha_values, total_error, '^-', label='Total Error', linewidth=2, markersize=6)
    ax.set_xlabel('Learning Rate (α)', fontsize=12)
    ax.set_ylabel('Error (log scale)', fontsize=12)
    ax.set_title('Bias-Variance Tradeoff (Log Scale)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=optimal_alpha, color='r', linestyle='--', linewidth=2)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/bias_variance.png',
                dpi=150, bbox_inches='tight')
    print(f"\nOptimal α ≈ {optimal_alpha:.3f}")
    print("Saved bias-variance plot to bias_variance.png")
    plt.close()


def create_learning_comparison_dashboard(env: SimpleGridWorld, policy: np.ndarray):
    """Create a comprehensive comparison dashboard."""
    print("\n" + "="*60)
    print("LEARNING COMPARISON DASHBOARD")
    print("="*60)

    n_episodes = 200
    n_runs = 15
    alpha = 0.1

    # Collect data from multiple runs
    all_snapshots = []
    all_td_errors = []

    print(f"\nRunning {n_runs} trials...")
    for run in range(n_runs):
        results = td_prediction_with_full_tracking(env, policy, n_episodes, alpha)
        all_snapshots.append(results['value_snapshots'])
        all_td_errors.append(results['td_errors'])

    # Create dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('TD Learning Analysis Dashboard', fontsize=18, fontweight='bold')

    # Plot 1: Final value function (mean)
    ax1 = fig.add_subplot(gs[0, 0])
    final_values = np.array([snapshots[-1] for snapshots in all_snapshots]).mean(axis=0)
    value_grid = final_values.reshape(env.size, env.size)
    im1 = ax1.imshow(value_grid, cmap='viridis', interpolation='nearest')
    ax1.set_title('Final Value Function (Mean)', fontsize=12)
    plt.colorbar(im1, ax=ax1)

    # Plot 2: Value function std
    ax2 = fig.add_subplot(gs[0, 1])
    std_values = np.array([snapshots[-1] for snapshots in all_snapshots]).std(axis=0)
    std_grid = std_values.reshape(env.size, env.size)
    im2 = ax2.imshow(std_grid, cmap='Reds', interpolation='nearest')
    ax2.set_title('Value Uncertainty (Std)', fontsize=12)
    plt.colorbar(im2, ax=ax2)

    # Plot 3: TD error evolution
    ax3 = fig.add_subplot(gs[0, 2])
    td_errors_array = np.array(all_td_errors)
    mean_errors = td_errors_array.mean(axis=0)
    std_errors = td_errors_array.std(axis=0)
    ax3.plot(mean_errors, linewidth=2)
    ax3.fill_between(range(len(mean_errors)),
                     mean_errors - std_errors,
                     mean_errors + std_errors,
                     alpha=0.3)
    ax3.set_title('TD Error Over Time', fontsize=12)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('|TD Error|')
    ax3.grid(True, alpha=0.3)

    # Plot 4-6: Value evolution for specific states
    key_states = [env.start_state, env.n_states // 2, env.goal_state - env.size]
    state_names = ['Start', 'Middle', 'Near Goal']

    for idx, (state, name) in enumerate(zip(key_states, state_names)):
        ax = fig.add_subplot(gs[1, idx])

        state_values_over_time = []
        for snapshots in all_snapshots:
            values = [snapshot[state] for snapshot in snapshots]
            state_values_over_time.append(values)

        state_values = np.array(state_values_over_time)
        mean_vals = state_values.mean(axis=0)
        std_vals = state_values.std(axis=0)

        episodes = np.arange(len(mean_vals)) * 5
        ax.plot(episodes, mean_vals, linewidth=2)
        ax.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3)
        ax.set_title(f'{name} State Value', fontsize=12)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

    # Plot 7-9: Distribution analysis
    ax7 = fig.add_subplot(gs[2, :])
    final_td_errors = [errors[-10:] for errors in all_td_errors]
    flat_errors = [e for run_errors in final_td_errors for e in run_errors]
    ax7.hist(flat_errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax7.set_xlabel('|TD Error| (Final 10 Episodes)', fontsize=12)
    ax7.set_ylabel('Frequency', fontsize=12)
    ax7.set_title('TD Error Distribution at Convergence', fontsize=12)
    ax7.grid(True, alpha=0.3, axis='y')

    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/learning_dashboard.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved learning dashboard to learning_dashboard.png")
    plt.close()


def main():
    """Main demonstration."""
    print("="*60)
    print("TD VISUALIZATION TOOLS")
    print("="*60)

    # Create environment
    env = SimpleGridWorld(size=5, gamma=0.9)
    policy = create_policy(env)

    # Generate visualizations
    visualize_value_evolution(env, policy)
    visualize_td_error_dynamics(env, policy)
    visualize_bias_variance_tradeoff(env, policy)
    create_learning_comparison_dashboard(env, policy)

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print("\nGenerated visualizations:")
    print("  - value_evolution.png: How values change over training")
    print("  - td_error_dynamics.png: TD error analysis")
    print("  - bias_variance.png: Bias-variance tradeoff")
    print("  - learning_dashboard.png: Comprehensive analysis")


if __name__ == "__main__":
    main()
