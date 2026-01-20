"""
02 - Policy Iteration

This example demonstrates the Policy Iteration algorithm:
- Alternating between Policy Evaluation and Policy Improvement
- Finding the optimal policy for a Grid World
- Visualizing the iteration process
"""

import numpy as np
from typing import Dict, List, Tuple


class GridWorldMDP:
    """
    A 4x4 Grid World MDP for demonstrating policy iteration.

    Grid Layout:
    +---+---+---+---+
    | 0 | 1 | 2 | 3 |  <- Goal at position 3
    +---+---+---+---+
    | 4 | X | 6 | 7 |  <- Obstacle at position 5
    +---+---+---+---+
    | 8 | 9 |10 |11 |
    +---+---+---+---+
    |12 |13 |14 |15 |  <- Start at position 12
    +---+---+---+---+
    """

    def __init__(self, size: int = 4, gamma: float = 0.9):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma

        self.states = list(range(self.n_states))
        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.action_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
        self.action_arrows = {0: "^", 1: ">", 2: "v", 3: "<"}

        self.start_state = self.n_states - self.size
        self.goal_state = self.size - 1
        self.obstacles = [5]
        self.terminal_states = [self.goal_state]

        self.P = self._build_transitions()

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def _build_transitions(self) -> Dict:
        """Build deterministic transition probabilities."""
        P = {}
        action_effects = {
            0: (-1, 0),   # Up
            1: (0, 1),    # Right
            2: (1, 0),    # Down
            3: (0, -1),   # Left
        }

        for s in self.states:
            P[s] = {}
            row, col = self._state_to_coord(s)

            # Terminal states
            if s in self.terminal_states:
                for a in self.actions:
                    P[s][a] = [(1.0, s, 0.0)]
                continue

            # Obstacles (shouldn't be entered, but handle anyway)
            if s in self.obstacles:
                for a in self.actions:
                    P[s][a] = [(1.0, s, -1.0)]
                continue

            for a in self.actions:
                dr, dc = action_effects[a]
                new_row = row + dr
                new_col = col + dc

                # Check boundaries
                if 0 <= new_row < self.size and 0 <= new_col < self.size:
                    new_state = self._coord_to_state(new_row, new_col)
                    # Check obstacle
                    if new_state in self.obstacles:
                        new_state = s  # Stay if hitting obstacle
                else:
                    new_state = s  # Stay if hitting wall

                # Rewards
                if new_state == self.goal_state:
                    reward = 1.0
                else:
                    reward = -0.04  # Small step penalty

                P[s][a] = [(1.0, new_state, reward)]

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


def policy_evaluation(mdp: GridWorldMDP, policy: np.ndarray,
                      theta: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
    """
    Policy Evaluation: Compute V_pi for a given policy.

    Bellman Expectation Equation:
    V_pi(s) = sum_s' P(s'|s, pi(s)) * [R(s, pi(s), s') + gamma * V_pi(s')]
    """
    V = np.zeros(mdp.n_states)

    for iteration in range(max_iter):
        delta = 0
        V_new = np.zeros(mdp.n_states)

        for s in mdp.states:
            if mdp.is_terminal(s) or s in mdp.obstacles:
                V_new[s] = 0
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

    return V


def policy_improvement(mdp: GridWorldMDP, V: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Policy Improvement: Create greedy policy with respect to V.

    pi'(s) = argmax_a sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]

    Returns:
        policy: New greedy policy
        stable: True if policy unchanged, False otherwise
    """
    policy = np.zeros(mdp.n_states, dtype=int)
    stable = True

    for s in mdp.states:
        if mdp.is_terminal(s) or s in mdp.obstacles:
            continue

        # Compute Q(s, a) for all actions
        action_values = []
        for a in mdp.actions:
            q = 0
            for prob, s_prime, reward in mdp.P[s][a]:
                q += prob * (reward + mdp.gamma * V[s_prime])
            action_values.append(q)

        # Greedy action
        best_action = np.argmax(action_values)
        policy[s] = best_action

    return policy, stable


def policy_iteration(mdp: GridWorldMDP, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Policy Iteration Algorithm.

    1. Initialize policy arbitrarily
    2. Policy Evaluation: Compute V_pi
    3. Policy Improvement: Improve policy greedily
    4. Repeat until policy stable

    Returns:
        optimal_policy: The optimal policy
        optimal_V: The optimal value function
        history: List of (policy, V) at each iteration
    """
    # Initialize with random policy
    np.random.seed(42)
    policy = np.random.randint(0, len(mdp.actions), size=mdp.n_states)

    history = []
    iteration = 0

    while True:
        iteration += 1

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"ITERATION {iteration}")
            print("=" * 60)

        # Policy Evaluation
        V = policy_evaluation(mdp, policy)

        if verbose:
            mdp.render_policy(policy, f"Policy (Iteration {iteration})")
            mdp.render_values(V, f"V_pi (Iteration {iteration})")

        # Save to history
        history.append((policy.copy(), V.copy()))

        # Policy Improvement
        old_policy = policy.copy()
        policy, _ = policy_improvement(mdp, V)

        # Check if policy changed
        policy_changed = not np.array_equal(old_policy, policy)

        if verbose:
            changes = np.sum(old_policy != policy)
            print(f"\nPolicy changes: {changes} states")

        if not policy_changed:
            if verbose:
                print("\nPolicy stable - OPTIMAL POLICY FOUND!")
            break

        if iteration > 100:
            print("Warning: Max iterations reached")
            break

    return policy, V, history


def compute_q_values(mdp: GridWorldMDP, V: np.ndarray) -> np.ndarray:
    """Compute Q(s, a) from V(s)."""
    Q = np.zeros((mdp.n_states, len(mdp.actions)))

    for s in mdp.states:
        for a in mdp.actions:
            q = 0
            for prob, s_prime, reward in mdp.P[s][a]:
                q += prob * (reward + mdp.gamma * V[s_prime])
            Q[s, a] = q

    return Q


def main():
    # ============================================
    # 1. CREATE MDP
    # ============================================
    print("=" * 60)
    print("POLICY ITERATION DEMONSTRATION")
    print("=" * 60)

    mdp = GridWorldMDP(size=4, gamma=0.9)

    print("""
    Grid World MDP with obstacle:

    +---+---+---+---+
    | 0 | 1 | 2 | G |  <- Goal
    +---+---+---+---+
    | 4 | X | 6 | 7 |  <- Obstacle
    +---+---+---+---+
    | 8 | 9 |10 |11 |
    +---+---+---+---+
    | S |13 |14 |15 |  <- Start
    +---+---+---+---+

    Rewards:
    - Reaching goal: +1.0
    - Each step: -0.04

    Actions: Up(^), Right(>), Down(v), Left(<)
    """)

    # ============================================
    # 2. RUN POLICY ITERATION
    # ============================================
    optimal_policy, optimal_V, history = policy_iteration(mdp, verbose=True)

    # ============================================
    # 3. FINAL RESULTS
    # ============================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    mdp.render_policy(optimal_policy, "Optimal Policy")
    mdp.render_values(optimal_V, "Optimal Value Function V*")

    print(f"\nTotal iterations: {len(history)}")
    print(f"V*(start) = V*(12) = {optimal_V[12]:.4f}")

    # ============================================
    # 4. Q-VALUES
    # ============================================
    print("\n" + "=" * 60)
    print("Q-VALUES FOR OPTIMAL POLICY")
    print("=" * 60)

    Q = compute_q_values(mdp, optimal_V)

    print("\nQ*(s, a) for selected states:")
    selected_states = [12, 8, 4, 0, 1, 2]

    print("\n  State |   Q(Up)   | Q(Right)  |  Q(Down)  |  Q(Left)")
    print("  " + "-" * 55)
    for s in selected_states:
        if s in mdp.obstacles:
            continue
        print(f"    {s:2d}  | {Q[s, 0]:9.4f} | {Q[s, 1]:9.4f} | {Q[s, 2]:9.4f} | {Q[s, 3]:9.4f}")

    # ============================================
    # 5. POLICY EVOLUTION
    # ============================================
    print("\n" + "=" * 60)
    print("POLICY EVOLUTION ACROSS ITERATIONS")
    print("=" * 60)

    print("\nPolicy changes at each iteration:")
    arrows = ["^", ">", "v", "<"]

    print("\nState 12 (Start):")
    for i, (pol, _) in enumerate(history):
        print(f"  Iteration {i + 1}: {arrows[pol[12]]}")

    print("\nState 8 (Below obstacle):")
    for i, (pol, _) in enumerate(history):
        print(f"  Iteration {i + 1}: {arrows[pol[8]]}")

    # ============================================
    # 6. VALUE FUNCTION EVOLUTION
    # ============================================
    print("\n" + "=" * 60)
    print("VALUE FUNCTION EVOLUTION")
    print("=" * 60)

    print("\nV(12) [Start state] across iterations:")
    for i, (_, V) in enumerate(history):
        print(f"  Iteration {i + 1}: V(12) = {V[12]:.4f}")

    # ============================================
    # 7. VERIFY OPTIMALITY
    # ============================================
    print("\n" + "=" * 60)
    print("VERIFY OPTIMALITY")
    print("=" * 60)

    print("""
    For optimal policy, at each state:
    V*(s) = max_a Q*(s, a) = Q*(s, pi*(s))

    Let's verify this:
    """)

    for s in [12, 8, 4, 0]:
        if s in mdp.obstacles:
            continue
        best_a = np.argmax(Q[s])
        policy_a = optimal_policy[s]
        V_s = optimal_V[s]
        Q_best = Q[s, best_a]

        print(f"  State {s}:")
        print(f"    pi*(s) = {arrows[policy_a]}, argmax_a Q(s,a) = {arrows[best_a]}")
        print(f"    V*(s) = {V_s:.4f}, max_a Q(s,a) = {Q_best:.4f}")
        print(f"    Match: {np.isclose(V_s, Q_best) and policy_a == best_a}")

    # ============================================
    # 8. COMPARISON WITH SUBOPTIMAL POLICY
    # ============================================
    print("\n" + "=" * 60)
    print("COMPARISON: OPTIMAL vs SUBOPTIMAL POLICY")
    print("=" * 60)

    # Create a suboptimal "always right" policy
    suboptimal_policy = np.ones(mdp.n_states, dtype=int)  # Always go right
    V_suboptimal = policy_evaluation(mdp, suboptimal_policy)

    print("\nSuboptimal Policy: Always go Right")
    mdp.render_policy(suboptimal_policy, "Suboptimal Policy")
    mdp.render_values(V_suboptimal, "V_pi (Suboptimal)")

    print("\nComparison of V(s) at start state:")
    print(f"  Optimal Policy:    V*(12) = {optimal_V[12]:.4f}")
    print(f"  Suboptimal Policy: V(12)  = {V_suboptimal[12]:.4f}")
    print(f"  Difference: {optimal_V[12] - V_suboptimal[12]:.4f}")

    # ============================================
    # 9. SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("SUMMARY: POLICY ITERATION")
    print("=" * 60)

    print("""
    Policy Iteration Algorithm:

    1. INITIALIZE
       - Start with arbitrary policy

    2. POLICY EVALUATION
       - Compute V_pi using Bellman expectation equation
       - Iterate until convergence

    3. POLICY IMPROVEMENT
       - For each state, select greedy action:
         pi'(s) = argmax_a Q(s, a)

    4. REPEAT
       - If policy changed, go to step 2
       - If policy stable, we found optimal policy!

    Key Properties:
    - Guaranteed to converge to optimal policy
    - Typically converges in few iterations
    - Each iteration: O(|S|^2 * |A|) for evaluation

    Next: Value Iteration (single-step updates)
    """)


if __name__ == "__main__":
    main()
