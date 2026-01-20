"""
06 - MDP Visualization

This example provides visualization tools for MDPs:
- Value function heatmaps
- Policy arrow plots
- State transition diagrams
- Value iteration animation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class VisualizableGridWorld:
    """Grid World MDP with visualization capabilities."""

    def __init__(self, size: int = 5, gamma: float = 0.9):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma

        self.states = list(range(self.n_states))
        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.action_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

        self.start_state = self.n_states - self.size
        self.goal_state = self.size - 1
        self.obstacles = []
        self.terminal_states = [self.goal_state]

        self._build_transitions()

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def _build_transitions(self):
        """Build deterministic transitions."""
        self.P = {}
        action_effects = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        for s in self.states:
            self.P[s] = {}
            row, col = self._state_to_coord(s)

            if s in self.terminal_states:
                for a in self.actions:
                    self.P[s][a] = [(1.0, s, 0.0)]
                continue

            for a in self.actions:
                dr, dc = action_effects[a]
                new_row = max(0, min(self.size - 1, row + dr))
                new_col = max(0, min(self.size - 1, col + dc))
                new_state = self._coord_to_state(new_row, new_col)

                if new_state in self.obstacles:
                    new_state = s

                reward = 1.0 if new_state == self.goal_state else -0.04
                self.P[s][a] = [(1.0, new_state, reward)]


def value_iteration_with_history(mdp: VisualizableGridWorld,
                                  theta: float = 1e-6,
                                  max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, List]:
    """Value iteration that saves history for animation."""
    V = np.zeros(mdp.n_states)
    history = [V.copy()]

    for iteration in range(max_iter):
        delta = 0
        V_new = np.zeros(mdp.n_states)

        for s in mdp.states:
            if s in mdp.terminal_states:
                continue

            action_values = []
            for a in mdp.actions:
                q = 0
                for prob, s_prime, reward in mdp.P[s][a]:
                    q += prob * (reward + mdp.gamma * V[s_prime])
                action_values.append(q)

            V_new[s] = max(action_values)
            delta = max(delta, abs(V[s] - V_new[s]))

        V = V_new
        history.append(V.copy())

        if delta < theta:
            break

    # Extract policy
    policy = np.zeros(mdp.n_states, dtype=int)
    for s in mdp.states:
        if s in mdp.terminal_states:
            continue

        action_values = []
        for a in mdp.actions:
            q = 0
            for prob, s_prime, reward in mdp.P[s][a]:
                q += prob * (reward + mdp.gamma * V[s_prime])
            action_values.append(q)

        policy[s] = np.argmax(action_values)

    return V, policy, history


def plot_value_function(mdp: VisualizableGridWorld, V: np.ndarray,
                        title: str = "Value Function",
                        ax: Optional[plt.Axes] = None,
                        show_values: bool = True,
                        cmap: str = 'RdYlGn') -> plt.Axes:
    """
    Plot value function as a heatmap.

    Args:
        mdp: The Grid World MDP
        V: Value function array
        title: Plot title
        ax: Matplotlib axes (creates new if None)
        show_values: Whether to show numerical values
        cmap: Colormap name
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Reshape V to grid
    V_grid = V.reshape(mdp.size, mdp.size)

    # Create heatmap
    im = ax.imshow(V_grid, cmap=cmap, aspect='equal')

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Value')

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, mdp.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mdp.size, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

    # Add values and special markers
    for row in range(mdp.size):
        for col in range(mdp.size):
            state = mdp._coord_to_state(row, col)

            if state == mdp.goal_state:
                ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                           fill=True, color='gold', alpha=0.7))
                ax.text(col, row, 'GOAL', ha='center', va='center',
                        fontsize=10, fontweight='bold')
            elif state == mdp.start_state:
                ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                           fill=False, edgecolor='blue', linewidth=3))
                if show_values:
                    ax.text(col, row, f'{V[state]:.2f}', ha='center', va='center',
                            fontsize=9, color='blue', fontweight='bold')
            elif state in mdp.obstacles:
                ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                           fill=True, color='black'))
            elif show_values:
                ax.text(col, row, f'{V[state]:.2f}', ha='center', va='center',
                        fontsize=9)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    return ax


def plot_policy(mdp: VisualizableGridWorld, policy: np.ndarray,
                V: Optional[np.ndarray] = None,
                title: str = "Policy",
                ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot policy as arrows on grid.

    Args:
        mdp: The Grid World MDP
        policy: Policy array
        V: Optional value function for background coloring
        title: Plot title
        ax: Matplotlib axes (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Background with value function if provided
    if V is not None:
        V_grid = V.reshape(mdp.size, mdp.size)
        ax.imshow(V_grid, cmap='RdYlGn', alpha=0.5, aspect='equal')

    # Arrow directions: Up, Right, Down, Left
    arrow_dx = [0, 0.3, 0, -0.3]
    arrow_dy = [-0.3, 0, 0.3, 0]

    # Plot arrows
    for row in range(mdp.size):
        for col in range(mdp.size):
            state = mdp._coord_to_state(row, col)

            if state == mdp.goal_state:
                ax.add_patch(plt.Circle((col, row), 0.3, color='gold'))
                ax.text(col, row, 'G', ha='center', va='center',
                        fontsize=12, fontweight='bold')
            elif state in mdp.obstacles:
                ax.add_patch(plt.Rectangle((col - 0.4, row - 0.4), 0.8, 0.8,
                                           color='black'))
            else:
                a = policy[state]
                ax.arrow(col, row, arrow_dx[a], arrow_dy[a],
                        head_width=0.15, head_length=0.1,
                        fc='darkblue', ec='darkblue')

            # Mark start state
            if state == mdp.start_state:
                ax.add_patch(plt.Circle((col, row), 0.4, fill=False,
                                        edgecolor='blue', linewidth=2))

    # Grid settings
    ax.set_xlim(-0.5, mdp.size - 0.5)
    ax.set_ylim(mdp.size - 0.5, -0.5)
    ax.set_xticks(range(mdp.size))
    ax.set_yticks(range(mdp.size))
    ax.grid(True, linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)

    return ax


def plot_value_iteration_progress(history: List[np.ndarray],
                                   mdp: VisualizableGridWorld,
                                   iterations: List[int] = None) -> plt.Figure:
    """
    Plot value function at different iterations.

    Args:
        history: List of value functions at each iteration
        mdp: The Grid World MDP
        iterations: Which iterations to show (default: [0, 1, 5, 10, final])
    """
    if iterations is None:
        n = len(history)
        iterations = [0, 1, min(5, n - 1), min(10, n - 1), n - 1]
        iterations = sorted(list(set(iterations)))

    n_plots = len(iterations)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))

    if n_plots == 1:
        axes = [axes]

    # Find global min/max for consistent coloring
    all_values = np.concatenate(history)
    vmin, vmax = all_values.min(), all_values.max()

    for ax, iter_idx in zip(axes, iterations):
        V = history[iter_idx]
        V_grid = V.reshape(mdp.size, mdp.size)

        im = ax.imshow(V_grid, cmap='RdYlGn', vmin=vmin, vmax=vmax)

        # Add values
        for row in range(mdp.size):
            for col in range(mdp.size):
                state = mdp._coord_to_state(row, col)
                if state == mdp.goal_state:
                    ax.text(col, row, 'G', ha='center', va='center',
                            fontsize=10, fontweight='bold')
                else:
                    ax.text(col, row, f'{V[state]:.2f}', ha='center', va='center',
                            fontsize=8)

        ax.set_title(f'Iteration {iter_idx}')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes, label='Value', shrink=0.8)
    fig.suptitle('Value Iteration Progress', fontsize=14)
    plt.tight_layout()

    return fig


def plot_convergence(history: List[np.ndarray],
                     mdp: VisualizableGridWorld) -> plt.Figure:
    """
    Plot convergence metrics during value iteration.

    Args:
        history: List of value functions at each iteration
        mdp: The Grid World MDP
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    iterations = range(len(history))

    # 1. Max delta per iteration
    deltas = []
    for i in range(1, len(history)):
        delta = np.max(np.abs(history[i] - history[i - 1]))
        deltas.append(delta)

    axes[0].semilogy(range(1, len(history)), deltas, 'b-', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Max Delta (log scale)')
    axes[0].set_title('Convergence: Max Change')
    axes[0].grid(True)

    # 2. Value of start state over time
    start_values = [V[mdp.start_state] for V in history]
    axes[1].plot(iterations, start_values, 'g-', linewidth=2)
    axes[1].axhline(y=start_values[-1], color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('V(start)')
    axes[1].set_title('Value of Start State')
    axes[1].grid(True)

    # 3. Mean value over time
    mean_values = [np.mean(V) for V in history]
    axes[2].plot(iterations, mean_values, 'm-', linewidth=2)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Mean V(s)')
    axes[2].set_title('Mean Value Across States')
    axes[2].grid(True)

    plt.tight_layout()
    return fig


def plot_state_transition_diagram(mdp: VisualizableGridWorld,
                                   policy: np.ndarray,
                                   figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
    """
    Plot state transition diagram showing policy-induced transitions.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Position states in a grid
    pos = {}
    for state in mdp.states:
        row, col = mdp._state_to_coord(state)
        pos[state] = (col, mdp.size - 1 - row)  # Flip y for visual

    # Draw states as circles
    for state in mdp.states:
        x, y = pos[state]
        if state == mdp.goal_state:
            circle = plt.Circle((x, y), 0.3, color='gold', ec='black', linewidth=2)
            ax.text(x, y, 'G', ha='center', va='center', fontsize=12, fontweight='bold')
        elif state == mdp.start_state:
            circle = plt.Circle((x, y), 0.3, color='lightblue', ec='blue', linewidth=2)
            ax.text(x, y, 'S', ha='center', va='center', fontsize=10)
        elif state in mdp.obstacles:
            circle = plt.Circle((x, y), 0.3, color='black')
        else:
            circle = plt.Circle((x, y), 0.3, color='lightgray', ec='black')
            ax.text(x, y, str(state), ha='center', va='center', fontsize=8)
        ax.add_patch(circle)

    # Draw transitions under policy
    for state in mdp.states:
        if state in mdp.terminal_states or state in mdp.obstacles:
            continue

        action = policy[state]
        for prob, next_state, _ in mdp.P[state][action]:
            if prob > 0 and next_state != state:
                x1, y1 = pos[state]
                x2, y2 = pos[next_state]

                # Shorten arrow to not overlap with circles
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx ** 2 + dy ** 2)
                if dist > 0:
                    dx = dx / dist * (dist - 0.6)
                    dy = dy / dist * (dist - 0.6)

                ax.annotate('', xy=(x1 + dx + 0.3 * dx / (dist - 0.6 + 1e-6), y1 + dy + 0.3 * dy / (dist - 0.6 + 1e-6)),
                           xytext=(x1 + 0.3 * dx / (dist - 0.6 + 1e-6), y1 + 0.3 * dy / (dist - 0.6 + 1e-6)),
                           arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5))

    ax.set_xlim(-0.5, mdp.size - 0.5)
    ax.set_ylim(-0.5, mdp.size - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('State Transition Diagram (Under Optimal Policy)', fontsize=14)

    return fig


def create_comparison_plot(mdp: VisualizableGridWorld,
                            V: np.ndarray,
                            policy: np.ndarray) -> plt.Figure:
    """Create a comprehensive comparison plot."""
    fig = plt.figure(figsize=(14, 10))

    # Value function heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    plot_value_function(mdp, V, "Value Function V*", ax=ax1)

    # Policy arrows
    ax2 = fig.add_subplot(2, 2, 2)
    plot_policy(mdp, policy, V, "Optimal Policy", ax=ax2)

    # Q-values for start state
    ax3 = fig.add_subplot(2, 2, 3)
    start_q = []
    for a in mdp.actions:
        q = 0
        for prob, s_prime, reward in mdp.P[mdp.start_state][a]:
            q += prob * (reward + mdp.gamma * V[s_prime])
        start_q.append(q)

    bars = ax3.bar(['Up', 'Right', 'Down', 'Left'], start_q, color=['blue', 'green', 'red', 'orange'])
    best_action = np.argmax(start_q)
    bars[best_action].set_color('gold')
    bars[best_action].set_edgecolor('black')
    bars[best_action].set_linewidth(2)
    ax3.set_ylabel('Q(start, a)')
    ax3.set_title('Q-values at Start State')
    ax3.axhline(y=0, color='gray', linestyle='--')

    # Value along optimal path
    ax4 = fig.add_subplot(2, 2, 4)
    path = [mdp.start_state]
    state = mdp.start_state
    while state != mdp.goal_state and len(path) < 20:
        action = policy[state]
        for prob, next_state, _ in mdp.P[state][action]:
            state = next_state
            break
        path.append(state)

    path_values = [V[s] for s in path]
    ax4.plot(range(len(path)), path_values, 'bo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('V(s)')
    ax4.set_title('Value Along Optimal Path')
    ax4.set_xticks(range(len(path)))
    ax4.set_xticklabels([f's{s}' for s in path], rotation=45)
    ax4.grid(True)

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("MDP VISUALIZATION")
    print("=" * 60)

    print("""
    This script generates various visualizations for MDPs:
    1. Value function heatmaps
    2. Policy arrow plots
    3. Value iteration progress
    4. Convergence analysis
    5. State transition diagrams
    """)

    # Create MDP
    mdp = VisualizableGridWorld(size=5, gamma=0.9)
    print(f"\nCreated {mdp.size}x{mdp.size} Grid World")
    print(f"Start: {mdp.start_state}, Goal: {mdp.goal_state}")

    # Run value iteration
    print("\nRunning value iteration...")
    V, policy, history = value_iteration_with_history(mdp)
    print(f"Converged in {len(history) - 1} iterations")
    print(f"V*(start) = {V[mdp.start_state]:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Value function
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    plot_value_function(mdp, V, "Optimal Value Function V*", ax=ax1)
    fig1.savefig('/tmp/mdp_value_function.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/mdp_value_function.png")

    # 2. Policy
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    plot_policy(mdp, policy, V, "Optimal Policy", ax=ax2)
    fig2.savefig('/tmp/mdp_policy.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/mdp_policy.png")

    # 3. Value iteration progress
    fig3 = plot_value_iteration_progress(history, mdp)
    fig3.savefig('/tmp/mdp_vi_progress.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/mdp_vi_progress.png")

    # 4. Convergence analysis
    fig4 = plot_convergence(history, mdp)
    fig4.savefig('/tmp/mdp_convergence.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/mdp_convergence.png")

    # 5. State transition diagram
    fig5 = plot_state_transition_diagram(mdp, policy)
    fig5.savefig('/tmp/mdp_transitions.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/mdp_transitions.png")

    # 6. Comprehensive comparison
    fig6 = create_comparison_plot(mdp, V, policy)
    fig6.savefig('/tmp/mdp_comprehensive.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/mdp_comprehensive.png")

    plt.close('all')

    print("\n" + "=" * 60)
    print("CONSOLE OUTPUT (Text-based visualization)")
    print("=" * 60)

    # Text-based value function
    print("\nValue Function V*:")
    print("+" + "--------+" * mdp.size)
    for row in range(mdp.size):
        line = "|"
        for col in range(mdp.size):
            state = mdp._coord_to_state(row, col)
            if state == mdp.goal_state:
                line += "  GOAL  |"
            else:
                line += f" {V[state]:6.3f} |"
        print(line)
        print("+" + "--------+" * mdp.size)

    # Text-based policy
    arrows = ["^", ">", "v", "<"]
    print("\nOptimal Policy:")
    print("+" + "----+" * mdp.size)
    for row in range(mdp.size):
        line = "|"
        for col in range(mdp.size):
            state = mdp._coord_to_state(row, col)
            if state == mdp.goal_state:
                line += " G  |"
            elif state == mdp.start_state:
                line += f" {arrows[policy[state]]}S |"
            else:
                line += f" {arrows[policy[state]]}  |"
        print(line)
        print("+" + "----+" * mdp.size)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    Generated Visualizations:
    1. mdp_value_function.png - Heatmap of V*
    2. mdp_policy.png - Arrow diagram of optimal policy
    3. mdp_vi_progress.png - Value function at different iterations
    4. mdp_convergence.png - Convergence metrics
    5. mdp_transitions.png - State transition diagram
    6. mdp_comprehensive.png - Combined analysis

    All files saved to /tmp/

    To view: open the PNG files or use plt.show() in interactive mode.
    """)


if __name__ == "__main__":
    main()
