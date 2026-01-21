"""
07 - Monte Carlo Visualization

Visualize MC learning process:
- Value function convergence
- Policy evolution
- Return distributions
- Comparison with DP
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class GridWorld:
    """Simple Grid World for visualization."""

    def __init__(self, size: int = 5, gamma: float = 0.9):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.gamma = gamma
        self.actions = [0, 1, 2, 3]

        self.goal_state = self.size - 1
        self.start_state = self.n_states - self.size

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        if state == self.goal_state:
            return state, 0.0, True

        row, col = self._state_to_coord(state)
        effects = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dr, dc = effects[action]

        new_row = max(0, min(self.size - 1, row + dr))
        new_col = max(0, min(self.size - 1, col + dc))
        next_state = self._coord_to_state(new_row, new_col)

        reward = 1.0 if next_state == self.goal_state else -0.04
        return next_state, reward, next_state == self.goal_state


def generate_episode(env: GridWorld, Q: Dict, epsilon: float) -> List[Tuple]:
    """Generate episode with epsilon-greedy policy."""
    episode = []
    state = env.start_state
    done = False

    while not done and len(episode) < 100:
        if np.random.random() < epsilon:
            action = np.random.randint(env.n_actions)
        else:
            q_values = [Q.get((state, a), 0.0) for a in range(env.n_actions)]
            action = np.argmax(q_values)

        next_state, reward, done = env.step(state, action)
        episode.append((state, action, reward))
        state = next_state

    return episode


def mc_control_with_history(env: GridWorld, n_episodes: int = 5000,
                             gamma: float = 0.9,
                             epsilon: float = 0.1,
                             record_interval: int = 100) -> Tuple[Dict, Dict]:
    """MC Control with detailed history for visualization."""
    Q = defaultdict(float)
    N = defaultdict(int)

    history = {
        'Q_snapshots': [],
        'episode_returns': [],
        'episode_lengths': [],
        'state_visit_counts': defaultdict(int),
        'return_distributions': defaultdict(list),
    }

    for ep in range(n_episodes):
        episode = generate_episode(env, Q, epsilon)

        # Record episode stats
        total_return = sum(r for _, _, r in episode)
        history['episode_returns'].append(total_return)
        history['episode_lengths'].append(len(episode))

        # MC update
        visited = set()
        G = 0

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            history['state_visit_counts'][state] += 1

            if (state, action) not in visited:
                visited.add((state, action))
                N[(state, action)] += 1
                Q[(state, action)] += (G - Q[(state, action)]) / N[(state, action)]

                # Store return for distribution analysis
                history['return_distributions'][state].append(G)

        # Save Q snapshot
        if (ep + 1) % record_interval == 0:
            history['Q_snapshots'].append((ep + 1, dict(Q)))

    return dict(Q), history


def compute_value_function(Q: Dict, n_states: int, n_actions: int = 4) -> np.ndarray:
    """Compute V(s) = max_a Q(s,a)."""
    V = np.zeros(n_states)
    for s in range(n_states):
        V[s] = max(Q.get((s, a), 0.0) for a in range(n_actions))
    return V


def value_iteration_dp(env: GridWorld, theta: float = 1e-6) -> np.ndarray:
    """DP Value Iteration for ground truth."""
    V = np.zeros(env.n_states)

    for _ in range(1000):
        V_new = np.zeros(env.n_states)
        delta = 0

        for s in range(env.n_states):
            if s == env.goal_state:
                continue

            action_values = []
            for a in env.actions:
                ns, r, _ = env.step(s, a)
                action_values.append(r + env.gamma * V[ns])

            V_new[s] = max(action_values)
            delta = max(delta, abs(V[s] - V_new[s]))

        V = V_new
        if delta < theta:
            break

    return V


def plot_value_convergence(env: GridWorld, history: Dict, V_true: np.ndarray):
    """Plot value function convergence over episodes."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Select episodes to show
    snapshots = history['Q_snapshots']
    indices = [0, len(snapshots) // 4, len(snapshots) // 2,
               3 * len(snapshots) // 4, len(snapshots) - 1]
    indices = list(set(min(i, len(snapshots) - 1) for i in indices))[:5]

    for idx, ax_idx in zip(indices, range(5)):
        ax = axes.flatten()[ax_idx]
        ep, Q = snapshots[idx]
        V = compute_value_function(Q, env.n_states)
        V_grid = V.reshape(env.size, env.size)

        im = ax.imshow(V_grid, cmap='RdYlGn', vmin=V_true.min(), vmax=V_true.max())
        ax.set_title(f'Episode {ep}')
        ax.set_xticks([])
        ax.set_yticks([])

        # Add values
        for i in range(env.size):
            for j in range(env.size):
                s = env._coord_to_state(i, j)
                if s == env.goal_state:
                    ax.text(j, i, 'G', ha='center', va='center', fontsize=8)
                else:
                    ax.text(j, i, f'{V[s]:.2f}', ha='center', va='center', fontsize=7)

    # True value function
    ax = axes.flatten()[5]
    V_true_grid = V_true.reshape(env.size, env.size)
    im = ax.imshow(V_true_grid, cmap='RdYlGn')
    ax.set_title('True V* (DP)')
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(env.size):
        for j in range(env.size):
            s = env._coord_to_state(i, j)
            if s == env.goal_state:
                ax.text(j, i, 'G', ha='center', va='center', fontsize=8)
            else:
                ax.text(j, i, f'{V_true[s]:.2f}', ha='center', va='center', fontsize=7)

    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.suptitle('Value Function Convergence (MC vs DP)', fontsize=14)
    plt.tight_layout()

    return fig


def plot_learning_curves(history: Dict):
    """Plot learning curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Episode returns
    returns = history['episode_returns']
    window = 100
    smoothed = np.convolve(returns, np.ones(window) / window, mode='valid')

    axes[0, 0].plot(returns, alpha=0.3, label='Raw')
    axes[0, 0].plot(range(window - 1, len(returns)), smoothed, 'r-', label='Smoothed')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_title('Episode Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Episode lengths
    lengths = history['episode_lengths']
    smoothed_len = np.convolve(lengths, np.ones(window) / window, mode='valid')

    axes[0, 1].plot(lengths, alpha=0.3, label='Raw')
    axes[0, 1].plot(range(window - 1, len(lengths)), smoothed_len, 'r-', label='Smoothed')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # State visit distribution
    visits = history['state_visit_counts']
    states = sorted(visits.keys())
    counts = [visits[s] for s in states]

    axes[1, 0].bar(states, counts, color='steelblue')
    axes[1, 0].set_xlabel('State')
    axes[1, 0].set_ylabel('Visit Count')
    axes[1, 0].set_title('State Visit Distribution')
    axes[1, 0].grid(True, axis='y')

    # Error over episodes
    snapshots = history['Q_snapshots']
    V_true = value_iteration_dp(GridWorld())

    errors = []
    episodes = []
    for ep, Q in snapshots:
        V = compute_value_function(Q, len(V_true))
        mae = np.mean(np.abs(V - V_true))
        errors.append(mae)
        episodes.append(ep)

    axes[1, 1].semilogy(episodes, errors, 'b-o')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Mean Absolute Error (log)')
    axes[1, 1].set_title('Convergence to True Values')
    axes[1, 1].grid(True)

    plt.tight_layout()
    return fig


def plot_return_distributions(history: Dict, env: GridWorld):
    """Plot return distributions for selected states."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Select states to analyze
    states_to_show = [
        env.start_state,
        env.start_state + 1,
        env.n_states // 2,
        env.goal_state - 1,
        env.goal_state - env.size,
        env.size,
    ]

    returns_dist = history['return_distributions']

    for ax, state in zip(axes.flatten(), states_to_show):
        returns = returns_dist.get(state, [])

        if returns:
            ax.hist(returns, bins=30, density=True, alpha=0.7, color='steelblue')
            ax.axvline(np.mean(returns), color='red', linestyle='--',
                      label=f'Mean: {np.mean(returns):.2f}')
            ax.axvline(np.median(returns), color='green', linestyle=':',
                      label=f'Median: {np.median(returns):.2f}')
            ax.legend(fontsize=8)

        row, col = env._state_to_coord(state)
        ax.set_title(f'State {state} (row={row}, col={col})')
        ax.set_xlabel('Return G')
        ax.set_ylabel('Density')

    fig.suptitle('Return Distributions at Different States', fontsize=14)
    plt.tight_layout()
    return fig


def plot_policy_evolution(env: GridWorld, history: Dict):
    """Plot policy evolution over training."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    arrows = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}

    snapshots = history['Q_snapshots']
    indices = [0, len(snapshots) // 5, 2 * len(snapshots) // 5,
               3 * len(snapshots) // 5, 4 * len(snapshots) // 5,
               len(snapshots) - 1]

    for plot_idx, snap_idx in enumerate(indices):
        if snap_idx >= len(snapshots):
            snap_idx = len(snapshots) - 1

        ep, Q = snapshots[snap_idx]
        ax = axes.flatten()[plot_idx]

        # Draw grid
        for i in range(env.size + 1):
            ax.axhline(y=i, color='black', linewidth=0.5)
            ax.axvline(x=i, color='black', linewidth=0.5)

        # Draw arrows for policy
        for state in range(env.n_states):
            row, col = env._state_to_coord(state)
            y = env.size - 1 - row + 0.5
            x = col + 0.5

            if state == env.goal_state:
                ax.add_patch(plt.Circle((x, y), 0.3, color='gold'))
                ax.text(x, y, 'G', ha='center', va='center', fontsize=10)
            else:
                q_values = [Q.get((state, a), 0.0) for a in range(env.n_actions)]
                best_action = np.argmax(q_values)
                dx, dy = arrows[best_action]
                ax.arrow(x - dx / 2, y - dy / 2, dx, dy,
                        head_width=0.1, head_length=0.05, fc='blue', ec='blue')

        ax.set_xlim(0, env.size)
        ax.set_ylim(0, env.size)
        ax.set_aspect('equal')
        ax.set_title(f'Episode {ep}')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('Policy Evolution During MC Learning', fontsize=14)
    plt.tight_layout()
    return fig


def text_visualization(env: GridWorld, Q: Dict, history: Dict, V_true: np.ndarray):
    """Text-based visualization for console."""
    print("\n" + "=" * 60)
    print("VALUE FUNCTION CONVERGENCE (TEXT)")
    print("=" * 60)

    V_mc = compute_value_function(Q, env.n_states)

    print("\nFinal MC Value Function:")
    print("+" + "--------+" * env.size)
    for row in range(env.size):
        line = "|"
        for col in range(env.size):
            s = env._coord_to_state(row, col)
            if s == env.goal_state:
                line += "  GOAL  |"
            else:
                line += f" {V_mc[s]:6.3f} |"
        print(line)
        print("+" + "--------+" * env.size)

    print("\nTrue Value Function (DP):")
    print("+" + "--------+" * env.size)
    for row in range(env.size):
        line = "|"
        for col in range(env.size):
            s = env._coord_to_state(row, col)
            if s == env.goal_state:
                line += "  GOAL  |"
            else:
                line += f" {V_true[s]:6.3f} |"
        print(line)
        print("+" + "--------+" * env.size)

    print("\nError (MC - DP):")
    print("+" + "--------+" * env.size)
    for row in range(env.size):
        line = "|"
        for col in range(env.size):
            s = env._coord_to_state(row, col)
            if s == env.goal_state:
                line += "  GOAL  |"
            else:
                err = V_mc[s] - V_true[s]
                line += f" {err:+6.3f} |"
        print(line)
        print("+" + "--------+" * env.size)

    mae = np.mean(np.abs(V_mc - V_true))
    print(f"\nMean Absolute Error: {mae:.4f}")


def main():
    print("=" * 60)
    print("MONTE CARLO VISUALIZATION")
    print("=" * 60)

    print("""
    This script visualizes the MC learning process:
    1. Value function convergence over episodes
    2. Learning curves (returns, lengths, errors)
    3. Return distributions at different states
    4. Policy evolution during training
    """)

    # Create environment
    env = GridWorld(size=5, gamma=0.9)
    print(f"\nGrid World: {env.size}x{env.size}")

    # Compute true values with DP
    print("Computing true values with DP...")
    V_true = value_iteration_dp(env)

    # Run MC with history
    print("Running MC Control (5000 episodes)...")
    Q, history = mc_control_with_history(env, n_episodes=5000, epsilon=0.1)

    # Text visualization
    text_visualization(env, Q, history, V_true)

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    fig1 = plot_value_convergence(env, history, V_true)
    fig1.savefig('/tmp/mc_value_convergence.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/mc_value_convergence.png")

    fig2 = plot_learning_curves(history)
    fig2.savefig('/tmp/mc_learning_curves.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/mc_learning_curves.png")

    fig3 = plot_return_distributions(history, env)
    fig3.savefig('/tmp/mc_return_distributions.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/mc_return_distributions.png")

    fig4 = plot_policy_evolution(env, history)
    fig4.savefig('/tmp/mc_policy_evolution.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/mc_policy_evolution.png")

    plt.close('all')

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\nTraining Statistics:")
    print(f"  Total episodes: {len(history['episode_returns'])}")
    print(f"  Final avg return (last 100): {np.mean(history['episode_returns'][-100:]):.3f}")
    print(f"  Final avg length (last 100): {np.mean(history['episode_lengths'][-100:]):.1f}")

    print(f"\nState Visit Statistics:")
    visits = history['state_visit_counts']
    print(f"  Most visited: State {max(visits, key=visits.get)} ({max(visits.values())} visits)")
    print(f"  Least visited: State {min(visits, key=visits.get)} ({min(visits.values())} visits)")

    print(f"\nConvergence:")
    V_mc = compute_value_function(Q, env.n_states)
    mae = np.mean(np.abs(V_mc - V_true))
    max_err = np.max(np.abs(V_mc - V_true))
    print(f"  Mean Absolute Error: {mae:.4f}")
    print(f"  Max Absolute Error: {max_err:.4f}")

    print("\n" + "=" * 60)
    print("VISUALIZATION FILES")
    print("=" * 60)
    print("""
    Generated files in /tmp/:
    1. mc_value_convergence.png - V(s) at different episodes
    2. mc_learning_curves.png - Returns, lengths, visits, errors
    3. mc_return_distributions.png - G distributions per state
    4. mc_policy_evolution.png - Policy changes over training

    Key Insights:
    - MC converges to true values (from DP)
    - Early episodes have high variance
    - States near goal converge faster
    - Return distributions show MC's sample-based nature
    """)


if __name__ == "__main__":
    main()
