"""
07 - Convergence Visualization

Visualize how Dynamic Programming algorithms converge:
- Value function evolution over iterations
- Convergence rate analysis
- Effect of gamma on convergence
- Policy stability during learning
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class GridWorld:
    """Simple Grid World for convergence analysis."""

    def __init__(self, size: int = 5, gamma: float = 0.9):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma

        self.goal_state = self.size - 1
        self.start_state = self.n_states - self.size
        self.terminal_states = {self.goal_state}

        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.P = self._build_transitions()

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def _build_transitions(self) -> Dict:
        P = {}
        action_effects = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        for s in range(self.n_states):
            P[s] = {}
            row, col = self._state_to_coord(s)

            if s in self.terminal_states:
                for a in self.actions:
                    P[s][a] = [(1.0, s, 0.0)]
                continue

            for a in self.actions:
                dr, dc = action_effects[a]
                new_row = max(0, min(self.size - 1, row + dr))
                new_col = max(0, min(self.size - 1, col + dc))
                new_state = self._coord_to_state(new_row, new_col)
                reward = 1.0 if new_state == self.goal_state else -0.04
                P[s][a] = [(1.0, new_state, reward)]

        return P


def value_iteration_detailed(mdp: GridWorld, max_iter: int = 200) -> Dict:
    """Value iteration with detailed tracking."""
    V = np.zeros(mdp.n_states)

    history = {
        'V': [V.copy()],
        'delta': [],
        'policy': [],
        'V_start': [V[mdp.start_state]],
        'V_mean': [np.mean(V)],
    }

    for iteration in range(max_iter):
        delta = 0
        V_new = np.zeros(mdp.n_states)
        policy = np.zeros(mdp.n_states, dtype=int)

        for s in range(mdp.n_states):
            if s in mdp.terminal_states:
                continue

            action_values = []
            for a in mdp.actions:
                q = 0
                for prob, s_prime, reward in mdp.P[s][a]:
                    q += prob * (reward + mdp.gamma * V[s_prime])
                action_values.append(q)

            V_new[s] = max(action_values)
            policy[s] = np.argmax(action_values)
            delta = max(delta, abs(V[s] - V_new[s]))

        V = V_new
        history['V'].append(V.copy())
        history['delta'].append(delta)
        history['policy'].append(policy.copy())
        history['V_start'].append(V[mdp.start_state])
        history['V_mean'].append(np.mean(V))

        if delta < 1e-10:
            break

    return history


def policy_iteration_detailed(mdp: GridWorld) -> Dict:
    """Policy iteration with detailed tracking."""
    policy = np.zeros(mdp.n_states, dtype=int)
    V = np.zeros(mdp.n_states)

    history = {
        'V': [V.copy()],
        'policy': [policy.copy()],
        'policy_changes': [],
        'eval_iterations': [],
    }

    for outer_iter in range(100):
        # Policy Evaluation
        eval_iters = 0
        while True:
            eval_iters += 1
            delta = 0
            V_new = np.zeros(mdp.n_states)

            for s in range(mdp.n_states):
                if s in mdp.terminal_states:
                    continue
                a = policy[s]
                v = 0
                for prob, s_prime, reward in mdp.P[s][a]:
                    v += prob * (reward + mdp.gamma * V[s_prime])
                V_new[s] = v
                delta = max(delta, abs(V[s] - V_new[s]))

            V = V_new
            if delta < 1e-8:
                break

        history['eval_iterations'].append(eval_iters)
        history['V'].append(V.copy())

        # Policy Improvement
        old_policy = policy.copy()
        for s in range(mdp.n_states):
            if s in mdp.terminal_states:
                continue
            action_values = []
            for a in mdp.actions:
                q = sum(p * (r + mdp.gamma * V[sp]) for p, sp, r in mdp.P[s][a])
                action_values.append(q)
            policy[s] = np.argmax(action_values)

        changes = np.sum(old_policy != policy)
        history['policy_changes'].append(changes)
        history['policy'].append(policy.copy())

        if changes == 0:
            break

    return history


def plot_convergence_analysis(mdp: GridWorld, vi_history: Dict, pi_history: Dict):
    """Create comprehensive convergence analysis plots."""
    fig = plt.figure(figsize=(16, 12))

    # 1. Delta over iterations (VI)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.semilogy(vi_history['delta'], 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Max Delta (log scale)')
    ax1.set_title('Value Iteration Convergence')
    ax1.grid(True)
    ax1.axhline(y=1e-6, color='r', linestyle='--', label='Threshold (1e-6)')
    ax1.legend()

    # 2. V(start) over iterations (VI)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(vi_history['V_start'], 'g-', linewidth=2)
    ax2.axhline(y=vi_history['V_start'][-1], color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('V(start)')
    ax2.set_title('Value at Start State')
    ax2.grid(True)

    # 3. Mean V over iterations (VI)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(vi_history['V_mean'], 'm-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Mean V(s)')
    ax3.set_title('Mean Value Function')
    ax3.grid(True)

    # 4. Policy changes (PI)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.bar(range(len(pi_history['policy_changes'])), pi_history['policy_changes'], color='orange')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('States Changed')
    ax4.set_title('Policy Iteration: Policy Changes')
    ax4.set_xticks(range(len(pi_history['policy_changes'])))

    # 5. Evaluation iterations per PI step
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.bar(range(len(pi_history['eval_iterations'])), pi_history['eval_iterations'], color='teal')
    ax5.set_xlabel('PI Iteration')
    ax5.set_ylabel('Evaluation Steps')
    ax5.set_title('Policy Iteration: Evaluation Cost')
    ax5.set_xticks(range(len(pi_history['eval_iterations'])))

    # 6. Value function at different iterations
    ax6 = fig.add_subplot(2, 3, 6)
    iterations_to_show = [0, 1, 5, 10, len(vi_history['V']) - 1]
    iterations_to_show = [i for i in iterations_to_show if i < len(vi_history['V'])]

    for i in iterations_to_show:
        V = vi_history['V'][i]
        ax6.plot(range(mdp.n_states), V, label=f'Iter {i}', alpha=0.7)

    ax6.set_xlabel('State')
    ax6.set_ylabel('V(s)')
    ax6.set_title('Value Function Evolution')
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()
    return fig


def plot_gamma_analysis(sizes_gammas_results: Dict):
    """Plot effect of gamma on convergence."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Extract data
    gammas = sorted(sizes_gammas_results.keys())

    iterations = [sizes_gammas_results[g]['iterations'] for g in gammas]
    final_V_start = [sizes_gammas_results[g]['V_start'] for g in gammas]
    convergence_rates = []

    for g in gammas:
        deltas = sizes_gammas_results[g]['deltas']
        if len(deltas) > 10:
            # Estimate convergence rate from log(delta)
            log_deltas = np.log(np.array(deltas[5:min(20, len(deltas))]) + 1e-15)
            rate = (log_deltas[-1] - log_deltas[0]) / (len(log_deltas) - 1)
            convergence_rates.append(rate)
        else:
            convergence_rates.append(0)

    # 1. Iterations vs Gamma
    axes[0].plot(gammas, iterations, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Gamma')
    axes[0].set_ylabel('Iterations to Converge')
    axes[0].set_title('Convergence Speed vs Gamma')
    axes[0].grid(True)

    # 2. Final V(start) vs Gamma
    axes[1].plot(gammas, final_V_start, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Gamma')
    axes[1].set_ylabel('V*(start)')
    axes[1].set_title('Optimal Value vs Gamma')
    axes[1].grid(True)

    # 3. Convergence Rate vs Gamma
    axes[2].plot(gammas, convergence_rates, 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Gamma')
    axes[2].set_ylabel('Log Convergence Rate')
    axes[2].set_title('Convergence Rate vs Gamma')
    axes[2].grid(True)

    plt.tight_layout()
    return fig


def plot_value_function_heatmap(mdp: GridWorld, V: np.ndarray, iteration: int, ax=None):
    """Plot value function as heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    V_grid = V.reshape(mdp.size, mdp.size)
    im = ax.imshow(V_grid, cmap='RdYlGn')

    # Add values as text
    for i in range(mdp.size):
        for j in range(mdp.size):
            state = mdp._coord_to_state(i, j)
            if state == mdp.goal_state:
                ax.text(j, i, 'G', ha='center', va='center', fontsize=10, fontweight='bold')
            else:
                ax.text(j, i, f'{V[state]:.2f}', ha='center', va='center', fontsize=8)

    ax.set_title(f'Value Function (Iteration {iteration})')
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def create_convergence_animation_frames(mdp: GridWorld, history: Dict, n_frames: int = 10):
    """Create frames for convergence animation."""
    n_iters = len(history['V'])
    frame_indices = np.linspace(0, n_iters - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    vmin = min(V.min() for V in history['V'])
    vmax = max(V.max() for V in history['V'])

    for i, idx in enumerate(frame_indices):
        V = history['V'][idx]
        V_grid = V.reshape(mdp.size, mdp.size)

        axes[i].imshow(V_grid, cmap='RdYlGn', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'Iter {idx}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

        # Mark goal
        goal_row, goal_col = mdp._state_to_coord(mdp.goal_state)
        axes[i].text(goal_col, goal_row, 'G', ha='center', va='center',
                     fontsize=12, fontweight='bold', color='white')

    plt.suptitle('Value Function Convergence', fontsize=14)
    plt.tight_layout()
    return fig


def text_visualization(mdp: GridWorld, vi_history: Dict, pi_history: Dict):
    """Create text-based visualization for console output."""
    print("\n" + "=" * 60)
    print("VALUE FUNCTION EVOLUTION (TEXT)")
    print("=" * 60)

    iterations_to_show = [0, 1, 5, 10, len(vi_history['V']) - 1]
    iterations_to_show = [i for i in iterations_to_show if i < len(vi_history['V'])]

    for idx in iterations_to_show:
        V = vi_history['V'][idx]
        delta = vi_history['delta'][idx - 1] if idx > 0 else float('inf')

        print(f"\nIteration {idx} (delta = {delta:.6f}):")
        print("+" + "--------+" * mdp.size)

        for row in range(mdp.size):
            line = "|"
            for col in range(mdp.size):
                state = mdp._coord_to_state(row, col)
                if state == mdp.goal_state:
                    cell = "  GOAL  "
                else:
                    cell = f" {V[state]:6.3f} "
                line += cell + "|"
            print(line)
            print("+" + "--------+" * mdp.size)

    # Policy evolution for PI
    print("\n" + "=" * 60)
    print("POLICY EVOLUTION (POLICY ITERATION)")
    print("=" * 60)

    arrows = ["^", ">", "v", "<"]

    for idx in range(len(pi_history['policy'])):
        policy = pi_history['policy'][idx]
        changes = pi_history['policy_changes'][idx - 1] if idx > 0 else mdp.n_states

        print(f"\nPI Iteration {idx} ({changes} changes):")
        print("+" + "----+" * mdp.size)

        for row in range(mdp.size):
            line = "|"
            for col in range(mdp.size):
                state = mdp._coord_to_state(row, col)
                if state == mdp.goal_state:
                    cell = " G "
                else:
                    cell = f" {arrows[policy[state]]} "
                line += cell + "|"
            print(line)
            print("+" + "----+" * mdp.size)


def main():
    print("=" * 60)
    print("CONVERGENCE VISUALIZATION FOR DYNAMIC PROGRAMMING")
    print("=" * 60)

    # Create MDP
    mdp = GridWorld(size=5, gamma=0.9)
    print(f"\nGrid World: {mdp.size}x{mdp.size}")
    print(f"Gamma: {mdp.gamma}")
    print(f"Start: {mdp.start_state}, Goal: {mdp.goal_state}")

    # ============================================
    # 1. VALUE ITERATION ANALYSIS
    # ============================================
    print("\n" + "=" * 60)
    print("1. VALUE ITERATION CONVERGENCE")
    print("=" * 60)

    vi_history = value_iteration_detailed(mdp)
    print(f"\nConverged in {len(vi_history['delta'])} iterations")
    print(f"Final V(start) = {vi_history['V_start'][-1]:.6f}")

    print("\nDelta per iteration:")
    for i, delta in enumerate(vi_history['delta'][:15]):
        print(f"  Iter {i + 1:2d}: delta = {delta:.8f}")
    if len(vi_history['delta']) > 15:
        print(f"  ...")
        print(f"  Iter {len(vi_history['delta']):2d}: delta = {vi_history['delta'][-1]:.8f}")

    # ============================================
    # 2. POLICY ITERATION ANALYSIS
    # ============================================
    print("\n" + "=" * 60)
    print("2. POLICY ITERATION CONVERGENCE")
    print("=" * 60)

    pi_history = policy_iteration_detailed(mdp)
    print(f"\nConverged in {len(pi_history['policy_changes'])} iterations")

    print("\nPolicy changes per iteration:")
    for i, (changes, evals) in enumerate(zip(pi_history['policy_changes'],
                                              pi_history['eval_iterations'])):
        print(f"  Iter {i + 1}: {changes:2d} policy changes, {evals:3d} eval steps")

    # ============================================
    # 3. EFFECT OF GAMMA
    # ============================================
    print("\n" + "=" * 60)
    print("3. EFFECT OF DISCOUNT FACTOR (GAMMA)")
    print("=" * 60)

    gamma_results = {}
    for gamma in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        mdp_g = GridWorld(size=5, gamma=gamma)
        hist = value_iteration_detailed(mdp_g)
        gamma_results[gamma] = {
            'iterations': len(hist['delta']),
            'V_start': hist['V_start'][-1],
            'deltas': hist['delta'],
        }

    print("\n  Gamma  | Iterations | V*(start)")
    print("  " + "-" * 35)
    for gamma in sorted(gamma_results.keys()):
        r = gamma_results[gamma]
        print(f"  {gamma:.2f}   |    {r['iterations']:4d}    |  {r['V_start']:.4f}")

    print("""
    Observations:
    - Higher gamma = more iterations to converge
    - Higher gamma = higher optimal values
    - Convergence rate: ||V_{k+1} - V*|| <= gamma * ||V_k - V*||
    """)

    # ============================================
    # 4. TEXT VISUALIZATION
    # ============================================
    text_visualization(mdp, vi_history, pi_history)

    # ============================================
    # 5. GENERATE PLOTS
    # ============================================
    print("\n" + "=" * 60)
    print("5. GENERATING PLOTS")
    print("=" * 60)

    # Main convergence analysis
    fig1 = plot_convergence_analysis(mdp, vi_history, pi_history)
    fig1.savefig('/tmp/dp_convergence_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/dp_convergence_analysis.png")

    # Gamma analysis
    fig2 = plot_gamma_analysis(gamma_results)
    fig2.savefig('/tmp/dp_gamma_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/dp_gamma_analysis.png")

    # Convergence frames
    fig3 = create_convergence_animation_frames(mdp, vi_history)
    fig3.savefig('/tmp/dp_convergence_frames.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/dp_convergence_frames.png")

    plt.close('all')

    # ============================================
    # 6. THEORETICAL ANALYSIS
    # ============================================
    print("\n" + "=" * 60)
    print("6. THEORETICAL CONVERGENCE ANALYSIS")
    print("=" * 60)

    print("""
    VALUE ITERATION CONVERGENCE:

    Theorem: Value Iteration converges to V* at rate:
        ||V_{k+1} - V*|| <= gamma * ||V_k - V*||

    This means:
    - Linear convergence in infinity norm
    - Rate depends only on gamma
    - After k iterations: ||V_k - V*|| <= gamma^k * ||V_0 - V*||

    For gamma = 0.9:
    - After 10 iterations: error <= 0.9^10 = 0.349 * initial
    - After 50 iterations: error <= 0.9^50 = 0.005 * initial
    - After 100 iterations: error <= 0.9^100 = 0.00003 * initial

    POLICY ITERATION CONVERGENCE:

    - Converges in at most |A|^|S| iterations (finite policies)
    - In practice, usually much faster (often < 10 iterations)
    - Each iteration: O(|S|^2) for evaluation + O(|S|*|A|) for improvement
    """)

    # Verify theoretical rate
    print("\nVerifying theoretical convergence rate:")
    print("-" * 50)

    V_star = vi_history['V'][-1]
    for i in range(min(10, len(vi_history['V']) - 1)):
        V = vi_history['V'][i]
        error = np.max(np.abs(V - V_star))
        theoretical_bound = (mdp.gamma ** i) * np.max(np.abs(vi_history['V'][0] - V_star))

        print(f"  Iter {i}: error = {error:.6f}, bound = {theoretical_bound:.6f}, ratio = {error / (theoretical_bound + 1e-10):.4f}")

    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("""
    Key Convergence Properties:

    1. VALUE ITERATION
       - Linear convergence at rate gamma
       - Number of iterations proportional to log(1/epsilon) / log(1/gamma)
       - Smaller gamma = faster convergence but smaller values

    2. POLICY ITERATION
       - Finite convergence (at most |A|^|S| iterations)
       - Each iteration more expensive (full policy evaluation)
       - Usually fewer outer iterations than VI

    3. DISCOUNT FACTOR TRADEOFF
       - High gamma: Better long-term planning, slower convergence
       - Low gamma: Faster convergence, more myopic policy

    4. VISUALIZATIONS SHOW
       - Value propagates from goal to other states
       - Policy stabilizes before values fully converge
       - Convergence is exponential (linear on log scale)

    Files generated:
    - /tmp/dp_convergence_analysis.png
    - /tmp/dp_gamma_analysis.png
    - /tmp/dp_convergence_frames.png
    """)


if __name__ == "__main__":
    main()
