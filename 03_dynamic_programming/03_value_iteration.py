"""
03 - Value Iteration

This example demonstrates the Value Iteration algorithm:
- Single update combining evaluation and improvement
- Bellman Optimality Equation
- Comparison with Policy Iteration
"""

import numpy as np
from typing import Dict, List, Tuple
import time


class GridWorldMDP:
    """
    A configurable Grid World MDP for demonstrating value iteration.
    """

    def __init__(self, size: int = 4, gamma: float = 0.9,
                 obstacles: List[int] = None, stochastic: bool = False):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma
        self.stochastic = stochastic

        self.states = list(range(self.n_states))
        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.action_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
        self.action_arrows = {0: "^", 1: ">", 2: "v", 3: "<"}

        self.start_state = self.n_states - self.size
        self.goal_state = self.size - 1
        self.obstacles = obstacles if obstacles else [5]
        self.terminal_states = [self.goal_state]

        self.P = self._build_transitions()

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def _get_next_state(self, state: int, action: int) -> int:
        """Get next state for deterministic action."""
        action_effects = {
            0: (-1, 0),   # Up
            1: (0, 1),    # Right
            2: (1, 0),    # Down
            3: (0, -1),   # Left
        }

        row, col = self._state_to_coord(state)
        dr, dc = action_effects[action]
        new_row = row + dr
        new_col = col + dc

        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            new_state = self._coord_to_state(new_row, new_col)
            if new_state in self.obstacles:
                return state
            return new_state
        return state

    def _build_transitions(self) -> Dict:
        """Build transition probabilities (deterministic or stochastic)."""
        P = {}

        for s in self.states:
            P[s] = {}

            if s in self.terminal_states:
                for a in self.actions:
                    P[s][a] = [(1.0, s, 0.0)]
                continue

            if s in self.obstacles:
                for a in self.actions:
                    P[s][a] = [(1.0, s, -1.0)]
                continue

            for a in self.actions:
                if self.stochastic:
                    # Stochastic: 80% intended, 10% left, 10% right
                    transitions = []

                    # Intended direction
                    intended_state = self._get_next_state(s, a)
                    reward = 1.0 if intended_state == self.goal_state else -0.04
                    transitions.append((0.8, intended_state, reward))

                    # Left of intended (action - 1 mod 4)
                    left_action = (a - 1) % 4
                    left_state = self._get_next_state(s, left_action)
                    reward = 1.0 if left_state == self.goal_state else -0.04
                    transitions.append((0.1, left_state, reward))

                    # Right of intended (action + 1 mod 4)
                    right_action = (a + 1) % 4
                    right_state = self._get_next_state(s, right_action)
                    reward = 1.0 if right_state == self.goal_state else -0.04
                    transitions.append((0.1, right_state, reward))

                    P[s][a] = transitions
                else:
                    # Deterministic
                    next_state = self._get_next_state(s, a)
                    reward = 1.0 if next_state == self.goal_state else -0.04
                    P[s][a] = [(1.0, next_state, reward)]

        return P

    def is_terminal(self, state: int) -> bool:
        return state in self.terminal_states

    def render_values(self, V: np.ndarray, title: str = "Value Function"):
        """Render value function as grid."""
        print(f"\n{title}:")
        print("+" + "--------+" * self.size)

        for row in range(self.size):
            line = "|"
            for col in range(self.size):
                state = self._coord_to_state(row, col)
                if state == self.goal_state:
                    cell = "  GOAL  "
                elif state in self.obstacles:
                    cell = "   X    "
                else:
                    cell = f" {V[state]:6.3f} "
                line += cell + "|"
            print(line)
            print("+" + "--------+" * self.size)

    def render_policy(self, policy: np.ndarray, title: str = "Policy"):
        """Render policy as grid with arrows."""
        print(f"\n{title}:")
        print("+" + "----+" * self.size)

        for row in range(self.size):
            line = "|"
            for col in range(self.size):
                state = self._coord_to_state(row, col)
                if state == self.goal_state:
                    cell = " G "
                elif state in self.obstacles:
                    cell = " X "
                else:
                    cell = f" {self.action_arrows[policy[state]]} "
                line += cell + "|"
            print(line)
            print("+" + "----+" * self.size)


def value_iteration(mdp: GridWorldMDP, theta: float = 1e-6,
                    max_iter: int = 1000, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Value Iteration Algorithm.

    Uses Bellman Optimality Equation:
    V*(s) = max_a sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V*(s')]

    Returns:
        V: Optimal value function
        policy: Optimal policy
        history: List of (V, delta) at each iteration
    """
    V = np.zeros(mdp.n_states)
    history = []

    for iteration in range(max_iter):
        delta = 0
        V_new = np.zeros(mdp.n_states)

        for s in mdp.states:
            if mdp.is_terminal(s) or s in mdp.obstacles:
                V_new[s] = 0
                continue

            # Bellman Optimality: max over all actions
            action_values = []
            for a in mdp.actions:
                q = 0
                for prob, s_prime, reward in mdp.P[s][a]:
                    q += prob * (reward + mdp.gamma * V[s_prime])
                action_values.append(q)

            V_new[s] = max(action_values)
            delta = max(delta, abs(V[s] - V_new[s]))

        V = V_new
        history.append((V.copy(), delta))

        if verbose and iteration % 10 == 0:
            print(f"  Iteration {iteration}: delta = {delta:.8f}")

        if delta < theta:
            if verbose:
                print(f"  Converged in {iteration + 1} iterations")
            break

    # Extract optimal policy from V*
    policy = extract_policy(mdp, V)

    return V, policy, history


def extract_policy(mdp: GridWorldMDP, V: np.ndarray) -> np.ndarray:
    """Extract greedy policy from value function."""
    policy = np.zeros(mdp.n_states, dtype=int)

    for s in mdp.states:
        if mdp.is_terminal(s) or s in mdp.obstacles:
            continue

        action_values = []
        for a in mdp.actions:
            q = 0
            for prob, s_prime, reward in mdp.P[s][a]:
                q += prob * (reward + mdp.gamma * V[s_prime])
            action_values.append(q)

        policy[s] = np.argmax(action_values)

    return policy


def policy_iteration(mdp: GridWorldMDP, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, int]:
    """Policy Iteration for comparison."""
    # Initialize random policy
    np.random.seed(42)
    policy = np.random.randint(0, len(mdp.actions), size=mdp.n_states)

    iterations = 0

    while True:
        iterations += 1

        # Policy Evaluation
        V = np.zeros(mdp.n_states)
        for _ in range(1000):
            delta = 0
            V_new = np.zeros(mdp.n_states)
            for s in mdp.states:
                if mdp.is_terminal(s) or s in mdp.obstacles:
                    continue
                a = policy[s]
                v = 0
                for prob, s_prime, reward in mdp.P[s][a]:
                    v += prob * (reward + mdp.gamma * V[s_prime])
                V_new[s] = v
                delta = max(delta, abs(V[s] - V_new[s]))
            V = V_new
            if delta < theta:
                break

        # Policy Improvement
        old_policy = policy.copy()
        policy = extract_policy(mdp, V)

        if np.array_equal(old_policy, policy):
            break

        if iterations > 100:
            break

    return V, policy, iterations


def main():
    # ============================================
    # 1. VALUE ITERATION ON SIMPLE GRID
    # ============================================
    print("=" * 60)
    print("1. VALUE ITERATION ON SIMPLE GRID WORLD")
    print("=" * 60)

    mdp = GridWorldMDP(size=4, gamma=0.9, obstacles=[5])

    print("""
    Grid World:
    +---+---+---+---+
    | 0 | 1 | 2 | G |
    +---+---+---+---+
    | 4 | X | 6 | 7 |
    +---+---+---+---+
    | 8 | 9 |10 |11 |
    +---+---+---+---+
    | S |13 |14 |15 |
    +---+---+---+---+

    Running Value Iteration...
    """)

    V_star, optimal_policy, history = value_iteration(mdp, verbose=True)

    mdp.render_policy(optimal_policy, "Optimal Policy (Value Iteration)")
    mdp.render_values(V_star, "Optimal Value Function V*")

    # ============================================
    # 2. CONVERGENCE ANALYSIS
    # ============================================
    print("\n" + "=" * 60)
    print("2. CONVERGENCE ANALYSIS")
    print("=" * 60)

    print("\nDelta (max change) per iteration:")
    print("\n  Iteration |   Delta")
    print("  " + "-" * 25)

    for i in range(min(20, len(history))):
        _, delta = history[i]
        print(f"     {i + 1:3d}    | {delta:.8f}")

    if len(history) > 20:
        print(f"     ...    |   ...")
        _, delta = history[-1]
        print(f"     {len(history):3d}    | {delta:.8f}")

    # ============================================
    # 3. VALUE FUNCTION EVOLUTION
    # ============================================
    print("\n" + "=" * 60)
    print("3. VALUE FUNCTION EVOLUTION")
    print("=" * 60)

    print("\nV(12) [Start state] across iterations:")
    checkpoints = [0, 4, 9, 19, 29, len(history) - 1]
    checkpoints = [c for c in checkpoints if c < len(history)]

    for i in checkpoints:
        V, _ = history[i]
        print(f"  Iteration {i + 1:3d}: V(12) = {V[12]:.6f}")

    # ============================================
    # 4. COMPARISON WITH POLICY ITERATION
    # ============================================
    print("\n" + "=" * 60)
    print("4. COMPARISON: VALUE ITERATION vs POLICY ITERATION")
    print("=" * 60)

    # Time Value Iteration
    start_time = time.time()
    V_vi, policy_vi, history_vi = value_iteration(mdp)
    vi_time = time.time() - start_time
    vi_iterations = len(history_vi)

    # Time Policy Iteration
    start_time = time.time()
    V_pi, policy_pi, pi_iterations = policy_iteration(mdp)
    pi_time = time.time() - start_time

    print("\n  Method            | Iterations | Time (ms) | V*(12)")
    print("  " + "-" * 55)
    print(f"  Value Iteration   |    {vi_iterations:4d}    |  {vi_time * 1000:7.3f}  | {V_vi[12]:.4f}")
    print(f"  Policy Iteration  |    {pi_iterations:4d}    |  {pi_time * 1000:7.3f}  | {V_pi[12]:.4f}")

    print("\n  Policies match:", np.array_equal(policy_vi, policy_pi))
    print("  Values match:", np.allclose(V_vi, V_pi))

    # ============================================
    # 5. BELLMAN OPTIMALITY DEMONSTRATION
    # ============================================
    print("\n" + "=" * 60)
    print("5. BELLMAN OPTIMALITY EQUATION DEMONSTRATION")
    print("=" * 60)

    print("""
    Bellman Optimality Equation:
    V*(s) = max_a sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V*(s')]

    Let's verify for state 7 (one step from goal):
    """)

    s = 7
    print(f"  State: {s}")
    print(f"  Actions: Up(0), Right(1), Down(2), Left(3)")
    print(f"\n  Q*(s, a) for each action:")

    for a in mdp.actions:
        q = 0
        for prob, s_prime, reward in mdp.P[s][a]:
            q += prob * (reward + mdp.gamma * V_star[s_prime])
        print(f"    Q*({s}, {mdp.action_names[a]:5s}) = {q:.4f}")

    print(f"\n  V*({s}) = max_a Q*(s, a) = {V_star[s]:.4f}")
    print(f"  Optimal action: {mdp.action_names[optimal_policy[s]]}")

    # ============================================
    # 6. STOCHASTIC ENVIRONMENT
    # ============================================
    print("\n" + "=" * 60)
    print("6. VALUE ITERATION WITH STOCHASTIC TRANSITIONS")
    print("=" * 60)

    print("""
    Now with stochastic transitions:
    - 80% intended direction
    - 10% slip left
    - 10% slip right
    """)

    mdp_stoch = GridWorldMDP(size=4, gamma=0.9, obstacles=[5], stochastic=True)
    V_stoch, policy_stoch, _ = value_iteration(mdp_stoch, verbose=True)

    mdp_stoch.render_policy(policy_stoch, "Optimal Policy (Stochastic)")
    mdp_stoch.render_values(V_stoch, "V* (Stochastic)")

    print("\nComparison: Deterministic vs Stochastic")
    print(f"  Deterministic V*(12) = {V_star[12]:.4f}")
    print(f"  Stochastic V*(12)    = {V_stoch[12]:.4f}")
    print(f"\n  Note: Stochastic values are lower due to uncertainty!")

    # ============================================
    # 7. EFFECT OF GAMMA
    # ============================================
    print("\n" + "=" * 60)
    print("7. EFFECT OF DISCOUNT FACTOR (GAMMA)")
    print("=" * 60)

    print("\nV*(12) and iterations for different gamma values:")
    print("\n  Gamma  | V*(12)  | Iterations")
    print("  " + "-" * 32)

    for gamma in [0.5, 0.7, 0.9, 0.95, 0.99]:
        mdp_gamma = GridWorldMDP(size=4, gamma=gamma, obstacles=[5])
        V_gamma, _, history_gamma = value_iteration(mdp_gamma)
        print(f"  {gamma:.2f}   | {V_gamma[12]:7.4f} |    {len(history_gamma):3d}")

    print("""
    Observations:
    - Higher gamma -> higher values (future matters more)
    - Higher gamma -> more iterations to converge
    """)

    # ============================================
    # 8. LARGER GRID WORLD
    # ============================================
    print("\n" + "=" * 60)
    print("8. SCALABILITY: LARGER GRID WORLD (8x8)")
    print("=" * 60)

    # Create 8x8 grid with multiple obstacles
    obstacles_8x8 = [9, 10, 17, 18, 25, 33, 41, 46, 47, 54]
    mdp_large = GridWorldMDP(size=8, gamma=0.9, obstacles=obstacles_8x8)
    mdp_large.goal_state = 7  # Top-right

    print(f"\n8x8 Grid World with {len(obstacles_8x8)} obstacles")
    print(f"States: {mdp_large.n_states}")
    print(f"Goal: State {mdp_large.goal_state}")

    start_time = time.time()
    V_large, policy_large, history_large = value_iteration(mdp_large)
    large_time = time.time() - start_time

    print(f"\nValue Iteration completed:")
    print(f"  Iterations: {len(history_large)}")
    print(f"  Time: {large_time * 1000:.2f} ms")
    print(f"  V*(start) = {V_large[mdp_large.start_state]:.4f}")

    mdp_large.render_policy(policy_large, "Optimal Policy (8x8)")

    # ============================================
    # 9. SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("9. SUMMARY: VALUE ITERATION")
    print("=" * 60)

    print("""
    Value Iteration Algorithm:

    1. INITIALIZE
       V(s) = 0 for all s

    2. ITERATE
       For each state s:
         V(s) = max_a sum_s' P(s'|s,a) * [R + gamma * V(s')]

    3. CONVERGENCE
       Stop when max |V_new(s) - V_old(s)| < theta

    4. EXTRACT POLICY
       pi(s) = argmax_a Q(s, a)

    Key Properties:
    - Combines evaluation and improvement in single update
    - Guaranteed to converge to V*
    - More iterations than Policy Iteration
    - But each iteration is cheaper
    - Scales better to larger state spaces

    Bellman Optimality Equation:
    V*(s) = max_a E[R + gamma * V*(s') | s, a]

    This is the foundation of many RL algorithms!

    Next Week: Monte Carlo Methods (learning without a model)
    """)


if __name__ == "__main__":
    main()
