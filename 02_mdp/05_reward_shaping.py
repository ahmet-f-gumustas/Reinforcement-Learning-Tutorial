"""
05 - Reward Shaping

This example demonstrates different reward structures and their effects:
- Sparse vs Dense rewards
- Reward shaping techniques
- Potential-based reward shaping
- Common reward design patterns
"""

import numpy as np
from typing import Dict, List, Tuple, Callable


class GridWorldWithRewards:
    """
    Grid World MDP with configurable reward functions.

    Demonstrates how different reward structures affect learning.
    """

    def __init__(self, size: int = 5, gamma: float = 0.99):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma

        self.states = list(range(self.n_states))
        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.action_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

        self.start_state = self.n_states - self.size  # Bottom-left
        self.goal_state = self.size - 1  # Top-right

        self._build_transitions()

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def _build_transitions(self):
        """Build deterministic transitions."""
        self.transitions = {}
        action_effects = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        for s in self.states:
            self.transitions[s] = {}
            row, col = self._state_to_coord(s)

            for a in self.actions:
                dr, dc = action_effects[a]
                new_row = max(0, min(self.size - 1, row + dr))
                new_col = max(0, min(self.size - 1, col + dc))
                self.transitions[s][a] = self._coord_to_state(new_row, new_col)

    def get_next_state(self, state: int, action: int) -> int:
        return self.transitions[state][action]

    def manhattan_distance(self, state: int) -> int:
        """Manhattan distance from state to goal."""
        row, col = self._state_to_coord(state)
        goal_row, goal_col = self._state_to_coord(self.goal_state)
        return abs(row - goal_row) + abs(col - goal_col)

    def euclidean_distance(self, state: int) -> float:
        """Euclidean distance from state to goal."""
        row, col = self._state_to_coord(state)
        goal_row, goal_col = self._state_to_coord(self.goal_state)
        return np.sqrt((row - goal_row) ** 2 + (col - goal_col) ** 2)


# ============================================
# REWARD FUNCTIONS
# ============================================

def sparse_reward(mdp: GridWorldWithRewards, state: int, action: int,
                  next_state: int) -> float:
    """
    Sparse Reward: Only reward at goal.

    Properties:
    - Simple and intuitive
    - Can be hard to learn (no guidance)
    - Credit assignment problem
    """
    if next_state == mdp.goal_state:
        return 1.0
    return 0.0


def sparse_with_penalty(mdp: GridWorldWithRewards, state: int, action: int,
                        next_state: int) -> float:
    """
    Sparse Reward with Step Penalty: Goal reward + small penalty per step.

    Properties:
    - Encourages shorter paths
    - Still relatively sparse
    - Common in practice
    """
    if next_state == mdp.goal_state:
        return 1.0
    return -0.01


def dense_manhattan(mdp: GridWorldWithRewards, state: int, action: int,
                    next_state: int) -> float:
    """
    Dense Reward: Based on Manhattan distance improvement.

    Properties:
    - Provides continuous feedback
    - Guides agent towards goal
    - Can lead to local optima in complex environments
    """
    if next_state == mdp.goal_state:
        return 10.0

    old_dist = mdp.manhattan_distance(state)
    new_dist = mdp.manhattan_distance(next_state)

    # Reward for getting closer, penalty for moving away
    return (old_dist - new_dist) * 0.1


def potential_based_shaping(mdp: GridWorldWithRewards, state: int, action: int,
                            next_state: int) -> float:
    """
    Potential-Based Reward Shaping (PBRS).

    F(s, s') = gamma * phi(s') - phi(s)

    where phi(s) is a potential function.

    Properties:
    - Theoretically sound (doesn't change optimal policy)
    - Provides dense feedback
    - Guaranteed to preserve optimal policy
    """
    # Base reward (sparse)
    base_reward = 1.0 if next_state == mdp.goal_state else 0.0

    # Potential function: negative distance to goal
    def phi(s):
        return -mdp.manhattan_distance(s)

    # Shaping reward
    shaping = mdp.gamma * phi(next_state) - phi(state)

    return base_reward + shaping


def curiosity_reward(mdp: GridWorldWithRewards, state: int, action: int,
                     next_state: int, visit_counts: Dict[int, int]) -> float:
    """
    Curiosity-Based Reward: Bonus for visiting less-explored states.

    Properties:
    - Encourages exploration
    - Helps in sparse reward settings
    - Diminishes over time
    """
    # Base reward
    base = 1.0 if next_state == mdp.goal_state else 0.0

    # Exploration bonus (inverse sqrt of visit count)
    count = visit_counts.get(next_state, 0) + 1
    exploration_bonus = 1.0 / np.sqrt(count)

    return base + 0.1 * exploration_bonus


# ============================================
# VALUE ITERATION FOR COMPARISON
# ============================================

def value_iteration(mdp: GridWorldWithRewards,
                    reward_fn: Callable,
                    theta: float = 1e-6,
                    max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray, int]:
    """Run value iteration with given reward function."""
    V = np.zeros(mdp.n_states)

    for iteration in range(max_iter):
        delta = 0
        V_new = np.zeros(mdp.n_states)

        for s in mdp.states:
            if s == mdp.goal_state:
                V_new[s] = 0
                continue

            action_values = []
            for a in mdp.actions:
                s_prime = mdp.get_next_state(s, a)
                r = reward_fn(mdp, s, a, s_prime)
                action_values.append(r + mdp.gamma * V[s_prime])

            V_new[s] = max(action_values)
            delta = max(delta, abs(V[s] - V_new[s]))

        V = V_new
        if delta < theta:
            break

    # Extract policy
    policy = np.zeros(mdp.n_states, dtype=int)
    for s in mdp.states:
        if s == mdp.goal_state:
            continue

        action_values = []
        for a in mdp.actions:
            s_prime = mdp.get_next_state(s, a)
            r = reward_fn(mdp, s, a, s_prime)
            action_values.append(r + mdp.gamma * V[s_prime])

        policy[s] = np.argmax(action_values)

    return V, policy, iteration + 1


def simulate_episode(mdp: GridWorldWithRewards, policy: np.ndarray,
                     reward_fn: Callable, max_steps: int = 100) -> Tuple[float, int]:
    """Simulate one episode following policy."""
    state = mdp.start_state
    total_reward = 0
    steps = 0

    while state != mdp.goal_state and steps < max_steps:
        action = policy[state]
        next_state = mdp.get_next_state(state, action)
        reward = reward_fn(mdp, state, action, next_state)
        total_reward += reward
        state = next_state
        steps += 1

    return total_reward, steps


def render_values(mdp: GridWorldWithRewards, V: np.ndarray, title: str):
    """Render value function as grid."""
    print(f"\n{title}:")
    print("+" + "--------+" * mdp.size)

    for row in range(mdp.size):
        line = "|"
        for col in range(mdp.size):
            state = mdp._coord_to_state(row, col)
            if state == mdp.goal_state:
                cell = "  GOAL  "
            else:
                cell = f" {V[state]:6.2f} "
            line += cell + "|"
        print(line)
        print("+" + "--------+" * mdp.size)


def render_policy(mdp: GridWorldWithRewards, policy: np.ndarray, title: str):
    """Render policy as grid."""
    arrows = ["^", ">", "v", "<"]
    print(f"\n{title}:")
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
    # ============================================
    # 1. INTRODUCTION
    # ============================================
    print("=" * 60)
    print("REWARD SHAPING IN REINFORCEMENT LEARNING")
    print("=" * 60)

    print("""
    Reward shaping is the art of designing reward functions that:
    1. Guide the agent towards desired behavior
    2. Make learning faster and more efficient
    3. Don't change the optimal policy (ideally)

    We'll compare different reward structures on a 5x5 grid world.
    """)

    mdp = GridWorldWithRewards(size=5, gamma=0.99)

    print(f"Grid World: {mdp.size}x{mdp.size}")
    print(f"Start: State {mdp.start_state} (bottom-left)")
    print(f"Goal: State {mdp.goal_state} (top-right)")
    print(f"Optimal path length: {mdp.manhattan_distance(mdp.start_state)} steps")

    # ============================================
    # 2. SPARSE REWARD
    # ============================================
    print("\n" + "=" * 60)
    print("1. SPARSE REWARD")
    print("=" * 60)

    print("""
    Sparse Reward: R(s,a,s') = 1 if s' is goal, else 0

    Pros:
    - Simple to define
    - Clear objective

    Cons:
    - Credit assignment problem
    - Slow learning (no guidance)
    """)

    V_sparse, policy_sparse, iters_sparse = value_iteration(mdp, sparse_reward)
    render_values(mdp, V_sparse, "Value Function (Sparse)")
    render_policy(mdp, policy_sparse, "Policy (Sparse)")
    print(f"\nConverged in {iters_sparse} iterations")

    reward, steps = simulate_episode(mdp, policy_sparse, sparse_reward)
    print(f"Episode: {steps} steps, total reward: {reward:.4f}")

    # ============================================
    # 3. SPARSE WITH STEP PENALTY
    # ============================================
    print("\n" + "=" * 60)
    print("2. SPARSE WITH STEP PENALTY")
    print("=" * 60)

    print("""
    R(s,a,s') = 1 if s' is goal, else -0.01

    Pros:
    - Encourages shorter paths
    - Simple modification

    Cons:
    - Still relatively sparse
    - Penalty magnitude matters
    """)

    V_penalty, policy_penalty, iters_penalty = value_iteration(mdp, sparse_with_penalty)
    render_values(mdp, V_penalty, "Value Function (Step Penalty)")
    render_policy(mdp, policy_penalty, "Policy (Step Penalty)")
    print(f"\nConverged in {iters_penalty} iterations")

    reward, steps = simulate_episode(mdp, policy_penalty, sparse_with_penalty)
    print(f"Episode: {steps} steps, total reward: {reward:.4f}")

    # ============================================
    # 4. DENSE REWARD (MANHATTAN DISTANCE)
    # ============================================
    print("\n" + "=" * 60)
    print("3. DENSE REWARD (DISTANCE-BASED)")
    print("=" * 60)

    print("""
    R(s,a,s') = 10 if s' is goal
              = 0.1 * (old_distance - new_distance) otherwise

    Pros:
    - Continuous feedback
    - Faster learning

    Cons:
    - Can create local optima
    - May not preserve optimal policy
    """)

    V_dense, policy_dense, iters_dense = value_iteration(mdp, dense_manhattan)
    render_values(mdp, V_dense, "Value Function (Dense)")
    render_policy(mdp, policy_dense, "Policy (Dense)")
    print(f"\nConverged in {iters_dense} iterations")

    reward, steps = simulate_episode(mdp, policy_dense, dense_manhattan)
    print(f"Episode: {steps} steps, total reward: {reward:.4f}")

    # ============================================
    # 5. POTENTIAL-BASED REWARD SHAPING
    # ============================================
    print("\n" + "=" * 60)
    print("4. POTENTIAL-BASED REWARD SHAPING (PBRS)")
    print("=" * 60)

    print("""
    F(s, s') = gamma * phi(s') - phi(s)

    where phi(s) = -manhattan_distance(s, goal)

    THEOREM: PBRS preserves the optimal policy!
    (Ng, Harada, Russell 1999)

    Pros:
    - Theoretically sound
    - Dense feedback without changing optimal policy

    Cons:
    - Requires defining potential function
    - Potential must be well-designed
    """)

    V_pbrs, policy_pbrs, iters_pbrs = value_iteration(mdp, potential_based_shaping)
    render_values(mdp, V_pbrs, "Value Function (PBRS)")
    render_policy(mdp, policy_pbrs, "Policy (PBRS)")
    print(f"\nConverged in {iters_pbrs} iterations")

    reward, steps = simulate_episode(mdp, policy_pbrs, potential_based_shaping)
    print(f"Episode: {steps} steps, total reward: {reward:.4f}")

    # ============================================
    # 6. COMPARISON
    # ============================================
    print("\n" + "=" * 60)
    print("COMPARISON OF REWARD FUNCTIONS")
    print("=" * 60)

    print("\n  Reward Type          | Iterations | Path Length | Same Policy?")
    print("  " + "-" * 60)

    results = [
        ("Sparse", iters_sparse, policy_sparse),
        ("Sparse + Penalty", iters_penalty, policy_penalty),
        ("Dense (Manhattan)", iters_dense, policy_dense),
        ("PBRS", iters_pbrs, policy_pbrs),
    ]

    reference_policy = policy_sparse
    for name, iters, policy in results:
        _, steps = simulate_episode(mdp, policy, sparse_reward)
        same = np.array_equal(policy, reference_policy)
        print(f"  {name:22s}|    {iters:4d}    |     {steps:3d}     |    {same}")

    # ============================================
    # 7. REWARD SHAPING PITFALLS
    # ============================================
    print("\n" + "=" * 60)
    print("REWARD SHAPING PITFALLS")
    print("=" * 60)

    print("""
    Common Mistakes:

    1. REWARD HACKING
       Agent finds unintended ways to maximize reward
       Example: Game agent pauses forever to avoid losing

    2. REWARD GAMING
       Agent exploits reward function loopholes
       Example: Cleaning robot makes mess to clean it again

    3. SPECIFICATION GAMING
       Agent achieves reward without intended behavior
       Example: Racing game drives in circles for speed bonus

    4. LOCAL OPTIMA
       Dense rewards can trap agent in suboptimal states
       Example: Agent stays near small reward, missing larger one

    5. REWARD MAGNITUDE ISSUES
       Unbalanced rewards cause unexpected behavior
       Example: Death penalty too small, agent sacrifices itself

    Best Practices:
    - Use PBRS when possible (guarantees optimal policy)
    - Test with random agents to find exploits
    - Monitor actual behavior, not just reward
    - Start sparse, add shaping carefully
    """)

    # ============================================
    # 8. PBRS THEOREM DEMONSTRATION
    # ============================================
    print("\n" + "=" * 60)
    print("PBRS THEOREM DEMONSTRATION")
    print("=" * 60)

    print("""
    Let's verify that PBRS preserves optimal policy.

    For any trajectory: s0 -> s1 -> s2 -> ... -> sT

    Sum of shaping rewards:
    F = sum_{t=0}^{T-1} [gamma * phi(s_{t+1}) - phi(s_t)]
      = gamma * phi(s_1) - phi(s_0)
        + gamma * phi(s_2) - phi(s_1)
        + ...
      = gamma^T * phi(s_T) - phi(s_0)

    This is a CONSTANT for any policy reaching the same terminal state!
    Therefore, policy ranking is unchanged.
    """)

    print("Verifying on our grid world:")
    print("-" * 50)

    # Different paths to goal
    paths = [
        [20, 15, 10, 5, 0, 1, 2, 3, 4],  # Up then right
        [20, 21, 22, 23, 24, 19, 14, 9, 4],  # Right then up
        [20, 21, 16, 11, 6, 1, 2, 3, 4],  # Diagonal-ish
    ]

    for i, path in enumerate(paths):
        shaping_sum = 0
        for t in range(len(path) - 1):
            s, s_next = path[t], path[t + 1]
            phi_s = -mdp.manhattan_distance(s)
            phi_s_next = -mdp.manhattan_distance(s_next)
            shaping_sum += mdp.gamma * phi_s_next - phi_s

        print(f"  Path {i + 1}: Sum of shaping rewards = {shaping_sum:.4f}")

    print("\n  All paths have similar shaping sums (differ only by gamma^T factor)")
    print("  Policy ranking preserved!")

    # ============================================
    # 9. SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("""
    Reward Shaping Techniques:

    1. SPARSE REWARDS
       - Simple but can be hard to learn
       - Use when behavior is clearly defined

    2. STEP PENALTIES
       - Encourages efficiency
       - Easy to implement

    3. DISTANCE-BASED (DENSE)
       - Provides continuous feedback
       - Risk of changing optimal policy

    4. POTENTIAL-BASED SHAPING (PBRS)
       - Theoretically sound
       - Preserves optimal policy
       - Recommended approach

    5. CURIOSITY/EXPLORATION BONUSES
       - Helps in sparse reward settings
       - Encourages state coverage

    Key Principle: The reward function defines what you want,
    the shaping helps the agent learn it faster.
    """)


if __name__ == "__main__":
    main()
