"""
01 - Policy Evaluation (Prediction)

This example demonstrates iterative policy evaluation:
- Computing V_pi(s) for a given policy
- Convergence of the value function
- Comparing different policies
"""

import numpy as np
from typing import Dict, List, Tuple


class GridWorldMDP:
    """
    A 4x4 Grid World MDP for demonstrating policy evaluation.

    Grid Layout:
    +---+---+---+---+
    | 0 | 1 | 2 | 3 |  <- Goal at position 3
    +---+---+---+---+
    | 4 | 5 | 6 | 7 |
    +---+---+---+---+
    | 8 | 9 |10 |11 |
    +---+---+---+---+
    |12 |13 |14 |15 |  <- Start at position 12
    +---+---+---+---+

    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    """

    def __init__(self, size: int = 4, gamma: float = 0.9):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma

        self.states = list(range(self.n_states))
        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.action_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

        self.start_state = self.n_states - self.size  # Bottom-left (12)
        self.goal_state = self.size - 1  # Top-right (3)
        self.terminal_states = [self.goal_state]

        self.P = self._build_transitions()
        self.R = self._build_rewards()

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

            # Terminal states stay in place
            if s in self.terminal_states:
                for a in self.actions:
                    P[s][a] = [(1.0, s, 0.0)]
                continue

            for a in self.actions:
                dr, dc = action_effects[a]
                new_row = row + dr
                new_col = col + dc

                # Check boundaries
                if 0 <= new_row < self.size and 0 <= new_col < self.size:
                    new_state = self._coord_to_state(new_row, new_col)
                else:
                    new_state = s  # Stay in place if hitting wall

                # Reward for reaching goal
                if new_state == self.goal_state:
                    reward = 1.0
                else:
                    reward = -0.04  # Small step penalty

                P[s][a] = [(1.0, new_state, reward)]

        return P

    def _build_rewards(self) -> Dict:
        """Build reward dictionary (for reference)."""
        R = {}
        for s in self.states:
            R[s] = {}
            for a in self.actions:
                for prob, s_prime, reward in self.P[s][a]:
                    R[s][a] = reward
        return R

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
                else:
                    cell = f" {V[state]:6.3f} "
                line += cell + "|"
            print(line)
            print("+" + "--------+" * self.size)

    def render_policy(self, policy: np.ndarray):
        """Render policy as grid with arrows."""
        print("\nPolicy:")
        print("+" + "----+" * self.size)
        arrows = ["^", ">", "v", "<"]

        for row in range(self.size):
            line = "|"
            for col in range(self.size):
                state = self._coord_to_state(row, col)
                if state == self.goal_state:
                    cell = " G "
                else:
                    cell = f" {arrows[policy[state]]} "
                line += cell + "|"
            print(line)
            print("+" + "----+" * self.size)


def policy_evaluation(mdp: GridWorldMDP, policy: np.ndarray,
                      theta: float = 1e-6, max_iter: int = 1000,
                      verbose: bool = False) -> Tuple[np.ndarray, List[float]]:
    """
    Iterative Policy Evaluation.

    Computes V_pi(s) for all states using the Bellman expectation equation:
    V_pi(s) = sum_a pi(a|s) * sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V_pi(s')]

    For deterministic policy: pi(a|s) = 1 for selected action, 0 otherwise
    """
    V = np.zeros(mdp.n_states)
    deltas = []

    for iteration in range(max_iter):
        delta = 0
        V_new = np.zeros(mdp.n_states)

        for s in mdp.states:
            if mdp.is_terminal(s):
                V_new[s] = 0
                continue

            # Get action from deterministic policy
            a = policy[s]

            # Bellman expectation equation
            v = 0
            for prob, s_prime, reward in mdp.P[s][a]:
                v += prob * (reward + mdp.gamma * V[s_prime])

            V_new[s] = v
            delta = max(delta, abs(V[s] - V_new[s]))

        V = V_new
        deltas.append(delta)

        if verbose and iteration % 10 == 0:
            print(f"  Iteration {iteration}: delta = {delta:.6f}")

        if delta < theta:
            if verbose:
                print(f"  Converged in {iteration + 1} iterations (delta = {delta:.8f})")
            break

    return V, deltas


def create_random_policy(mdp: GridWorldMDP) -> np.ndarray:
    """Create a random policy (uniform over actions)."""
    # For deterministic evaluation, we pick one action randomly per state
    np.random.seed(42)
    return np.random.randint(0, len(mdp.actions), size=mdp.n_states)


def create_right_down_policy(mdp: GridWorldMDP) -> np.ndarray:
    """Create a policy that prefers going right, then down."""
    policy = np.zeros(mdp.n_states, dtype=int)
    for s in mdp.states:
        row, col = mdp._state_to_coord(s)
        if col < mdp.size - 1:
            policy[s] = 1  # Right
        else:
            policy[s] = 2  # Down (when at right edge, go down)
    return policy


def create_optimal_like_policy(mdp: GridWorldMDP) -> np.ndarray:
    """Create a policy that moves towards the goal (top-right)."""
    policy = np.zeros(mdp.n_states, dtype=int)
    goal_row, goal_col = mdp._state_to_coord(mdp.goal_state)

    for s in mdp.states:
        row, col = mdp._state_to_coord(s)

        # Prefer moving towards goal
        if row > goal_row:
            policy[s] = 0  # Up
        elif col < goal_col:
            policy[s] = 1  # Right
        else:
            policy[s] = 0  # Up (default)

    return policy


def main():
    # ============================================
    # 1. CREATE MDP
    # ============================================
    print("=" * 60)
    print("1. CREATE GRID WORLD MDP")
    print("=" * 60)

    mdp = GridWorldMDP(size=4, gamma=0.9)

    print(f"""
    Grid World MDP:
    - Size: {mdp.size}x{mdp.size}
    - States: {mdp.n_states}
    - Actions: Up, Right, Down, Left
    - Goal: State {mdp.goal_state} (top-right)
    - Gamma: {mdp.gamma}

    Rewards:
    - Reaching goal: +1.0
    - Each step: -0.04 (encourages shorter paths)
    """)

    # Show grid layout
    print("Grid Layout (state numbers):")
    print("+" + "----+" * mdp.size)
    for row in range(mdp.size):
        line = "|"
        for col in range(mdp.size):
            state = mdp._coord_to_state(row, col)
            if state == mdp.goal_state:
                cell = " G "
            elif state == mdp.start_state:
                cell = " S "
            else:
                cell = f"{state:3d}"
            line += cell + "|"
        print(line)
        print("+" + "----+" * mdp.size)

    # ============================================
    # 2. POLICY EVALUATION: RANDOM POLICY
    # ============================================
    print("\n" + "=" * 60)
    print("2. POLICY EVALUATION: RANDOM POLICY")
    print("=" * 60)

    print("\nEvaluating random policy...")
    random_policy = create_random_policy(mdp)
    mdp.render_policy(random_policy)

    V_random, deltas_random = policy_evaluation(mdp, random_policy, verbose=True)
    mdp.render_values(V_random, "V_pi (Random Policy)")

    # ============================================
    # 3. POLICY EVALUATION: RIGHT-DOWN POLICY
    # ============================================
    print("\n" + "=" * 60)
    print("3. POLICY EVALUATION: RIGHT-DOWN POLICY")
    print("=" * 60)

    print("\nEvaluating right-down policy...")
    right_down_policy = create_right_down_policy(mdp)
    mdp.render_policy(right_down_policy)

    V_right_down, deltas_rd = policy_evaluation(mdp, right_down_policy, verbose=True)
    mdp.render_values(V_right_down, "V_pi (Right-Down Policy)")

    # ============================================
    # 4. POLICY EVALUATION: OPTIMAL-LIKE POLICY
    # ============================================
    print("\n" + "=" * 60)
    print("4. POLICY EVALUATION: TOWARDS-GOAL POLICY")
    print("=" * 60)

    print("\nEvaluating towards-goal policy...")
    optimal_policy = create_optimal_like_policy(mdp)
    mdp.render_policy(optimal_policy)

    V_optimal, deltas_opt = policy_evaluation(mdp, optimal_policy, verbose=True)
    mdp.render_values(V_optimal, "V_pi (Towards-Goal Policy)")

    # ============================================
    # 5. COMPARISON OF POLICIES
    # ============================================
    print("\n" + "=" * 60)
    print("5. COMPARISON OF POLICIES")
    print("=" * 60)

    print("\nValue at start state (state 12) for each policy:")
    print(f"  Random Policy:       V(12) = {V_random[12]:.4f}")
    print(f"  Right-Down Policy:   V(12) = {V_right_down[12]:.4f}")
    print(f"  Towards-Goal Policy: V(12) = {V_optimal[12]:.4f}")

    print("\nAverage value across all states:")
    print(f"  Random Policy:       avg(V) = {np.mean(V_random):.4f}")
    print(f"  Right-Down Policy:   avg(V) = {np.mean(V_right_down):.4f}")
    print(f"  Towards-Goal Policy: avg(V) = {np.mean(V_optimal):.4f}")

    print("""
    Observations:
    - The towards-goal policy has the highest values
    - Better policies lead to higher expected returns
    - Policy evaluation tells us HOW GOOD a policy is
    """)

    # ============================================
    # 6. CONVERGENCE ANALYSIS
    # ============================================
    print("=" * 60)
    print("6. CONVERGENCE ANALYSIS")
    print("=" * 60)

    print(f"\nConvergence (iterations to reach theta=1e-6):")
    print(f"  Random Policy:       {len(deltas_random)} iterations")
    print(f"  Right-Down Policy:   {len(deltas_rd)} iterations")
    print(f"  Towards-Goal Policy: {len(deltas_opt)} iterations")

    print("\nDelta values (first 10 iterations) for Towards-Goal Policy:")
    for i, d in enumerate(deltas_opt[:10]):
        print(f"  Iteration {i + 1}: delta = {d:.6f}")

    # ============================================
    # 7. EFFECT OF GAMMA
    # ============================================
    print("\n" + "=" * 60)
    print("7. EFFECT OF DISCOUNT FACTOR (GAMMA)")
    print("=" * 60)

    print("\nV(start) for towards-goal policy with different gamma values:")

    for gamma in [0.5, 0.7, 0.9, 0.95, 0.99]:
        mdp_gamma = GridWorldMDP(size=4, gamma=gamma)
        V_gamma, _ = policy_evaluation(mdp_gamma, optimal_policy)
        print(f"  gamma = {gamma}: V(12) = {V_gamma[12]:.4f}")

    print("""
    Observations:
    - Higher gamma -> higher values (future rewards matter more)
    - gamma = 0.99 values are much higher
    - gamma affects both values and convergence speed
    """)

    # ============================================
    # 8. BELLMAN EQUATION DEMONSTRATION
    # ============================================
    print("=" * 60)
    print("8. BELLMAN EQUATION DEMONSTRATION")
    print("=" * 60)

    print("""
    Let's verify the Bellman equation manually for state 7.

    State 7 is one step left of the goal (state 3).
    With towards-goal policy, action is 'Up' (action 0).
    """)

    s = 7
    a = optimal_policy[s]
    print(f"  State: {s}")
    print(f"  Action: {mdp.action_names[a]}")

    # Get transition
    for prob, s_prime, reward in mdp.P[s][a]:
        print(f"  Transition: P(s'={s_prime}|s={s}, a={a}) = {prob}")
        print(f"  Reward: R = {reward}")

        # Bellman equation
        V_s = prob * (reward + mdp.gamma * V_optimal[s_prime])
        print(f"\n  Bellman calculation:")
        print(f"  V({s}) = P * (R + gamma * V(s'))")
        print(f"        = {prob} * ({reward} + {mdp.gamma} * {V_optimal[s_prime]:.4f})")
        print(f"        = {prob} * ({reward + mdp.gamma * V_optimal[s_prime]:.4f})")
        print(f"        = {V_s:.4f}")
        print(f"\n  Computed V({s}) = {V_optimal[s]:.4f}")
        print(f"  Match: {np.isclose(V_s, V_optimal[s])}")

    # ============================================
    # 9. SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("9. SUMMARY")
    print("=" * 60)

    print("""
    Policy Evaluation Key Points:

    1. PURPOSE
       - Compute V_pi(s) for a given policy pi
       - Answers: "How good is this policy?"

    2. ALGORITHM
       - Iteratively apply Bellman expectation equation
       - Stop when max change < threshold

    3. CONVERGENCE
       - Guaranteed to converge for gamma < 1
       - Rate depends on gamma and MDP structure

    4. USE CASES
       - Evaluating fixed policies
       - Part of Policy Iteration algorithm
       - Comparing different policies

    Next: Policy Iteration (combining evaluation + improvement)
    """)


if __name__ == "__main__":
    main()
