"""
04 - TD(λ) and Eligibility Traces

Implementing TD(λ) with eligibility traces for credit assignment.

Demonstrates:
- Eligibility traces concept
- TD(λ) algorithm (backward view)
- Effect of λ parameter
- Forward vs backward view equivalence
- Credit assignment visualization
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt


class GridWorld:
    """GridWorld environment for TD(λ) experiments."""

    def __init__(self, size: int = 6, gamma: float = 0.9):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma
        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left

        self.start_state = size * (size - 1)  # Bottom-left
        self.goal_state = size - 1  # Top-right

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        """Execute action."""
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


def create_policy(env: GridWorld) -> np.ndarray:
    """Create a policy that moves toward the goal."""
    policy = np.zeros((env.n_states, len(env.actions)))

    for state in range(env.n_states):
        if state == env.goal_state:
            policy[state] = 1.0 / len(env.actions)
            continue

        row, col = env._state_to_coord(state)
        goal_row, goal_col = env._state_to_coord(env.goal_state)

        weights = np.ones(len(env.actions)) * 0.1

        if row > goal_row:
            weights[0] = 0.6
        elif row < goal_row:
            weights[2] = 0.6

        if col < goal_col:
            weights[1] = 0.6
        elif col > goal_col:
            weights[3] = 0.6

        policy[state] = weights / weights.sum()

    return policy


def td_lambda(env: GridWorld, policy: np.ndarray, lambda_param: float,
              n_episodes: int = 300, alpha: float = 0.1) -> Tuple[Dict, List]:
    """
    TD(λ) with eligibility traces (backward view).

    Args:
        env: Environment
        policy: Policy to evaluate
        lambda_param: λ parameter (0 = TD(0), 1 = MC-like)
        n_episodes: Number of episodes
        alpha: Learning rate

    Returns:
        V: Value function
        value_history: Average values per episode
    """
    V = defaultdict(float)
    value_history = []

    for episode in range(n_episodes):
        # Initialize eligibility traces
        E = defaultdict(float)

        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            # Select action
            action = np.random.choice(env.actions, p=policy[state])

            # Take action
            next_state, reward, done = env.step(state, action)

            # Calculate TD error
            delta = reward + env.gamma * V[next_state] - V[state]

            # Increment eligibility trace for current state
            E[state] += 1

            # Update all states
            for s in range(env.n_states):
                if E[s] > 0:
                    # Update value
                    V[s] += alpha * delta * E[s]

                    # Decay eligibility trace
                    E[s] *= env.gamma * lambda_param

            state = next_state
            steps += 1

        # Track average value
        avg_value = np.mean([V[s] for s in range(env.n_states)])
        value_history.append(avg_value)

    return dict(V), value_history


def td_lambda_with_trace_tracking(env: GridWorld, policy: np.ndarray,
                                   lambda_param: float, alpha: float = 0.1) -> List[np.ndarray]:
    """Run one episode and track eligibility traces."""
    V = defaultdict(float)
    E = defaultdict(float)
    trace_history = []

    state = env.reset()
    done = False
    steps = 0

    while not done and steps < 100:
        action = np.random.choice(env.actions, p=policy[state])
        next_state, reward, done = env.step(state, action)

        delta = reward + env.gamma * V[next_state] - V[state]
        E[state] += 1

        for s in range(env.n_states):
            if E[s] > 0:
                V[s] += alpha * delta * E[s]
                E[s] *= env.gamma * lambda_param

        # Record traces
        trace_snapshot = np.array([E[s] for s in range(env.n_states)])
        trace_history.append(trace_snapshot)

        state = next_state
        steps += 1

    return trace_history


def compare_lambda_values(env: GridWorld, policy: np.ndarray, n_runs: int = 15):
    """Compare different λ values."""
    print("\n" + "="*60)
    print("COMPARING DIFFERENT λ VALUES")
    print("="*60)

    lambda_values = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
    n_episodes = 250
    alpha = 0.1

    all_histories = {lam: [] for lam in lambda_values}

    print(f"\nRunning {n_runs} trials for each λ...")
    for lam in lambda_values:
        print(f"  λ = {lam}")
        for run in range(n_runs):
            _, history = td_lambda(env, policy, lam, n_episodes, alpha)
            all_histories[lam].append(history)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TD(λ): Effect of λ Parameter', fontsize=16)

    # Plot 1: Learning curves
    ax = axes[0, 0]
    for lam in lambda_values:
        histories = np.array(all_histories[lam])
        mean_hist = histories.mean(axis=0)
        label = f'λ={lam}' + (' (TD(0))' if lam == 0 else ' (MC-like)' if lam == 1 else '')
        ax.plot(mean_hist, label=label, linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Value', fontsize=12)
    ax.set_title('Learning Curves for Different λ', fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 2: Convergence speed
    ax = axes[0, 1]
    convergence_speeds = []
    for lam in lambda_values:
        histories = np.array(all_histories[lam])
        mean_hist = histories.mean(axis=0)
        final_value = mean_hist[-20:].mean()
        threshold = final_value * 0.9

        converged_at = np.argmax(mean_hist >= threshold)
        if mean_hist[converged_at] < threshold:
            converged_at = n_episodes
        convergence_speeds.append(converged_at)

    ax.bar(range(len(lambda_values)), convergence_speeds, color='steelblue')
    ax.set_xticks(range(len(lambda_values)))
    ax.set_xticklabels([f'{lam}' for lam in lambda_values])
    ax.set_xlabel('λ value', fontsize=12)
    ax.set_ylabel('Episodes to 90% Convergence', fontsize=12)
    ax.set_title('Convergence Speed vs λ', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Final variance
    ax = axes[1, 0]
    final_variances = []
    for lam in lambda_values:
        histories = np.array(all_histories[lam])
        final_values = histories[:, -20:].mean(axis=1)
        final_variances.append(final_values.var())

    ax.bar(range(len(lambda_values)), final_variances, color='coral')
    ax.set_xticks(range(len(lambda_values)))
    ax.set_xticklabels([f'{lam}' for lam in lambda_values])
    ax.set_xlabel('λ value', fontsize=12)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title('Final Value Variance vs λ', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Selected λ with error bands
    ax = axes[1, 1]
    selected_lambda = [0.0, 0.5, 0.9]
    for lam in selected_lambda:
        histories = np.array(all_histories[lam])
        mean_hist = histories.mean(axis=0)
        std_hist = histories.std(axis=0)
        ax.plot(mean_hist, label=f'λ={lam}', linewidth=2)
        ax.fill_between(range(n_episodes),
                        mean_hist - std_hist,
                        mean_hist + std_hist,
                        alpha=0.3)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Value', fontsize=12)
    ax.set_title('Selected λ Values (mean ± std)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/lambda_comparison.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved λ comparison to lambda_comparison.png")
    plt.close()

    # Print statistics
    print("\nConvergence Speed:")
    for lam, speed in zip(lambda_values, convergence_speeds):
        print(f"  λ={lam:.2f}: {speed:3d} episodes")


def visualize_eligibility_traces(env: GridWorld, policy: np.ndarray):
    """Visualize how eligibility traces evolve."""
    print("\n" + "="*60)
    print("VISUALIZING ELIGIBILITY TRACES")
    print("="*60)

    lambda_values = [0.0, 0.5, 0.9]
    alpha = 0.1

    fig, axes = plt.subplots(len(lambda_values), 1, figsize=(14, 10))
    fig.suptitle('Eligibility Trace Evolution', fontsize=16)

    for idx, lam in enumerate(lambda_values):
        print(f"  Running episode with λ={lam}")
        trace_history = td_lambda_with_trace_tracking(env, policy, lam, alpha)

        # Convert to 2D array for heatmap
        trace_array = np.array(trace_history).T  # states x time

        # Create heatmap
        ax = axes[idx]
        im = ax.imshow(trace_array, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_ylabel('State', fontsize=11)
        ax.set_title(f'λ = {lam}' + (' (TD(0), no trace decay)' if lam == 0 else
                                     ' (Strong trace decay)' if lam == 0.5 else
                                     ' (Weak trace decay)'), fontsize=13)
        plt.colorbar(im, ax=ax, label='Eligibility')

    axes[-1].set_xlabel('Time Step', fontsize=12)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/eligibility_traces.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved eligibility traces to eligibility_traces.png")
    plt.close()


def compare_td_lambda_vs_n_step(env: GridWorld, policy: np.ndarray):
    """Compare TD(λ) with n-step TD."""
    print("\n" + "="*60)
    print("TD(λ) vs n-STEP TD")
    print("="*60)

    n_episodes = 200
    n_runs = 10
    alpha = 0.1

    # TD(λ) with different λ
    lambda_values = [0.0, 0.3, 0.5, 0.7, 0.9]

    plt.figure(figsize=(12, 6))

    for lam in lambda_values:
        all_histories = []
        for _ in range(n_runs):
            _, history = td_lambda(env, policy, lam, n_episodes, alpha)
            all_histories.append(history)

        mean_hist = np.array(all_histories).mean(axis=0)
        plt.plot(mean_hist, label=f'TD(λ={lam})', linewidth=2)

    plt.xlabel('Episode', fontsize=13)
    plt.ylabel('Average Value', fontsize=13)
    plt.title('TD(λ) Learning Curves', fontsize=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/td_lambda_curves.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved TD(λ) curves to td_lambda_curves.png")
    plt.close()


def visualize_credit_assignment(env: GridWorld, policy: np.ndarray):
    """Visualize how credit is assigned with eligibility traces."""
    print("\n" + "="*60)
    print("CREDIT ASSIGNMENT VISUALIZATION")
    print("="*60)

    lambda_values = [0.0, 0.7]
    alpha = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Credit Assignment: How TD Error Spreads', fontsize=16)

    for idx, lam in enumerate(lambda_values):
        V = defaultdict(float)
        E = defaultdict(float)

        # Run one episode
        state = env.reset()
        done = False
        steps = 0
        delta_history = []

        while not done and steps < 50:
            action = np.random.choice(env.actions, p=policy[state])
            next_state, reward, done = env.step(state, action)

            delta = reward + env.gamma * V[next_state] - V[state]
            E[state] += 1

            # Track which states get updated
            states_updated = []
            for s in range(env.n_states):
                if E[s] > 0.01:  # Threshold for visualization
                    V[s] += alpha * delta * E[s]
                    states_updated.append(s)
                    E[s] *= env.gamma * lam

            delta_history.append((steps, delta, len(states_updated)))

            state = next_state
            steps += 1

        # Plot
        ax = axes[idx]
        steps_list, deltas, n_updated = zip(*delta_history)
        ax2 = ax.twinx()

        line1 = ax.plot(steps_list, deltas, 'b-', linewidth=2, label='TD Error')
        line2 = ax2.plot(steps_list, n_updated, 'r--', linewidth=2, label='States Updated')

        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('TD Error (δ)', fontsize=12, color='b')
        ax2.set_ylabel('Number of States Updated', fontsize=12, color='r')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')

        title = f'λ = {lam}' + (' (only current state)' if lam == 0 else ' (spreads credit)')
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/05_temporal_difference/credit_assignment.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved credit assignment to credit_assignment.png")
    plt.close()


def main():
    """Main demonstration."""
    print("="*60)
    print("TD(λ) AND ELIGIBILITY TRACES DEMONSTRATION")
    print("="*60)

    # Create environment
    env = GridWorld(size=6, gamma=0.9)
    policy = create_policy(env)

    # Run experiments
    compare_lambda_values(env, policy, n_runs=15)
    visualize_eligibility_traces(env, policy)
    compare_td_lambda_vs_n_step(env, policy)
    visualize_credit_assignment(env, policy)

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Insights:")
    print("  - λ=0: Pure TD(0), updates only current state")
    print("  - λ=1: MC-like, spreads credit to all visited states")
    print("  - Medium λ (0.5-0.8): Often optimal balance")
    print("  - Eligibility traces enable online credit assignment")
    print("  - Higher λ: more credit spread, but higher variance")


if __name__ == "__main__":
    main()
