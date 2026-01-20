"""
06 - Asynchronous Dynamic Programming

Standard DP updates all states in each sweep.
Asynchronous DP methods update states more flexibly:

1. In-Place DP: Update values immediately
2. Prioritized Sweeping: Update states with largest expected change
3. Real-Time DP: Focus on states relevant to current trajectory

Benefits: Can be much faster, focuses computation where it matters.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
import heapq
import time


class GridWorldMDP:
    """Standard Grid World for comparison."""

    def __init__(self, size: int = 10, gamma: float = 0.9):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma

        self.states = list(range(self.n_states))
        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left

        self.goal_state = self.size - 1  # Top-right
        self.terminal_states = {self.goal_state}

        self.P = self._build_transitions()

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def _build_transitions(self) -> Dict:
        P = {}
        action_effects = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        for s in self.states:
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

                reward = 1.0 if new_state == self.goal_state else -0.01
                P[s][a] = [(1.0, new_state, reward)]

        return P

    def get_predecessors(self, state: int) -> Set[int]:
        """Get states that can transition into this state."""
        preds = set()
        for s in self.states:
            for a in self.actions:
                for prob, s_prime, _ in self.P[s][a]:
                    if s_prime == state and prob > 0:
                        preds.add(s)
        return preds


# ============================================
# SYNCHRONOUS VALUE ITERATION (BASELINE)
# ============================================

def synchronous_value_iteration(mdp: GridWorldMDP, theta: float = 1e-6) -> Tuple[np.ndarray, int, int]:
    """
    Standard synchronous value iteration.

    Updates ALL states in each sweep, using old values only.
    """
    V = np.zeros(mdp.n_states)
    iterations = 0
    total_updates = 0

    while True:
        iterations += 1
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
            total_updates += 1

        V = V_new

        if delta < theta:
            break

    return V, iterations, total_updates


# ============================================
# IN-PLACE VALUE ITERATION
# ============================================

def inplace_value_iteration(mdp: GridWorldMDP, theta: float = 1e-6) -> Tuple[np.ndarray, int, int]:
    """
    In-place value iteration.

    Updates values immediately, so later states in sweep use updated values.
    Often converges faster than synchronous.
    """
    V = np.zeros(mdp.n_states)
    iterations = 0
    total_updates = 0

    while True:
        iterations += 1
        delta = 0

        for s in mdp.states:
            if s in mdp.terminal_states:
                continue

            old_v = V[s]
            action_values = []
            for a in mdp.actions:
                q = 0
                for prob, s_prime, reward in mdp.P[s][a]:
                    q += prob * (reward + mdp.gamma * V[s_prime])  # Uses potentially updated V
                action_values.append(q)

            V[s] = max(action_values)
            delta = max(delta, abs(old_v - V[s]))
            total_updates += 1

        if delta < theta:
            break

    return V, iterations, total_updates


# ============================================
# PRIORITIZED SWEEPING
# ============================================

def prioritized_sweeping(mdp: GridWorldMDP, theta: float = 1e-6) -> Tuple[np.ndarray, int, int]:
    """
    Prioritized Sweeping.

    Maintain a priority queue of states by their Bellman error.
    Always update the state with the largest expected change.
    Propagate changes to predecessors.
    """
    V = np.zeros(mdp.n_states)
    total_updates = 0

    # Priority queue: (-priority, state) - negated for max-heap behavior
    # Priority = |V_new(s) - V(s)|
    pq = []

    # Initialize: compute priorities for all states
    for s in mdp.states:
        if s in mdp.terminal_states:
            continue

        action_values = []
        for a in mdp.actions:
            q = 0
            for prob, s_prime, reward in mdp.P[s][a]:
                q += prob * (reward + mdp.gamma * V[s_prime])
            action_values.append(q)

        v_new = max(action_values)
        priority = abs(v_new - V[s])

        if priority > theta:
            heapq.heappush(pq, (-priority, s))

    # Track which states are in queue to avoid duplicates
    in_queue = {s for _, s in pq}

    iterations = 0
    max_iterations = mdp.n_states * 100  # Safety limit

    while pq and iterations < max_iterations:
        iterations += 1

        # Pop highest priority state
        neg_priority, s = heapq.heappop(pq)
        in_queue.discard(s)

        if s in mdp.terminal_states:
            continue

        # Update state
        old_v = V[s]
        action_values = []
        for a in mdp.actions:
            q = 0
            for prob, s_prime, reward in mdp.P[s][a]:
                q += prob * (reward + mdp.gamma * V[s_prime])
            action_values.append(q)

        V[s] = max(action_values)
        total_updates += 1

        # If significant change, update predecessors' priorities
        if abs(V[s] - old_v) > theta:
            for pred in mdp.get_predecessors(s):
                if pred in mdp.terminal_states:
                    continue

                # Compute new priority for predecessor
                action_values = []
                for a in mdp.actions:
                    q = 0
                    for prob, s_prime, reward in mdp.P[pred][a]:
                        q += prob * (reward + mdp.gamma * V[s_prime])
                    action_values.append(q)

                v_new = max(action_values)
                priority = abs(v_new - V[pred])

                if priority > theta and pred not in in_queue:
                    heapq.heappush(pq, (-priority, pred))
                    in_queue.add(pred)

    return V, iterations, total_updates


# ============================================
# GAUSS-SEIDEL VALUE ITERATION
# ============================================

def gauss_seidel_value_iteration(mdp: GridWorldMDP, theta: float = 1e-6,
                                   sweep_order: str = 'forward') -> Tuple[np.ndarray, int, int]:
    """
    Gauss-Seidel Value Iteration with configurable sweep order.

    Different sweep orders can significantly affect convergence:
    - 'forward': 0, 1, 2, ..., n-1
    - 'backward': n-1, n-2, ..., 0
    - 'toward_goal': States ordered by distance to goal
    """
    V = np.zeros(mdp.n_states)
    iterations = 0
    total_updates = 0

    # Determine sweep order
    if sweep_order == 'forward':
        order = list(range(mdp.n_states))
    elif sweep_order == 'backward':
        order = list(range(mdp.n_states - 1, -1, -1))
    elif sweep_order == 'toward_goal':
        # Sort by distance to goal
        goal_row, goal_col = mdp._state_to_coord(mdp.goal_state)

        def dist_to_goal(s):
            r, c = mdp._state_to_coord(s)
            return abs(r - goal_row) + abs(c - goal_col)

        order = sorted(range(mdp.n_states), key=dist_to_goal)
    else:
        order = list(range(mdp.n_states))

    while True:
        iterations += 1
        delta = 0

        for s in order:
            if s in mdp.terminal_states:
                continue

            old_v = V[s]
            action_values = []
            for a in mdp.actions:
                q = 0
                for prob, s_prime, reward in mdp.P[s][a]:
                    q += prob * (reward + mdp.gamma * V[s_prime])
                action_values.append(q)

            V[s] = max(action_values)
            delta = max(delta, abs(old_v - V[s]))
            total_updates += 1

        if delta < theta:
            break

    return V, iterations, total_updates


# ============================================
# REAL-TIME DYNAMIC PROGRAMMING
# ============================================

def rtdp(mdp: GridWorldMDP, start_state: int, n_trials: int = 100,
         max_steps: int = 200) -> Tuple[np.ndarray, int]:
    """
    Real-Time Dynamic Programming (RTDP).

    Only update states that are visited during simulated trajectories.
    Focuses computation on relevant states.
    """
    V = np.zeros(mdp.n_states)
    total_updates = 0

    for trial in range(n_trials):
        state = start_state
        steps = 0

        while state not in mdp.terminal_states and steps < max_steps:
            steps += 1

            # Bellman update for current state
            action_values = []
            for a in mdp.actions:
                q = 0
                for prob, s_prime, reward in mdp.P[state][a]:
                    q += prob * (reward + mdp.gamma * V[s_prime])
                action_values.append(q)

            V[state] = max(action_values)
            total_updates += 1

            # Greedy action selection
            best_action = np.argmax(action_values)

            # Simulate transition
            transitions = mdp.P[state][best_action]
            probs = [t[0] for t in transitions]
            next_states = [t[1] for t in transitions]
            state = np.random.choice(next_states, p=probs)

    return V, total_updates


# ============================================
# COMPARISON
# ============================================

def extract_policy(mdp: GridWorldMDP, V: np.ndarray) -> np.ndarray:
    """Extract greedy policy from value function."""
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

    return policy


def compare_methods(size: int = 10):
    """Compare all async DP methods."""
    mdp = GridWorldMDP(size=size, gamma=0.9)
    theta = 1e-6

    print(f"\nGrid World: {size}x{size} ({mdp.n_states} states)")
    print("=" * 70)

    results = []

    # 1. Synchronous VI
    start = time.time()
    V_sync, iters_sync, updates_sync = synchronous_value_iteration(mdp, theta)
    time_sync = time.time() - start
    results.append(('Synchronous VI', iters_sync, updates_sync, time_sync, V_sync))

    # 2. In-Place VI
    start = time.time()
    V_inplace, iters_inplace, updates_inplace = inplace_value_iteration(mdp, theta)
    time_inplace = time.time() - start
    results.append(('In-Place VI', iters_inplace, updates_inplace, time_inplace, V_inplace))

    # 3. Prioritized Sweeping
    start = time.time()
    V_priority, iters_priority, updates_priority = prioritized_sweeping(mdp, theta)
    time_priority = time.time() - start
    results.append(('Prioritized Sweep', iters_priority, updates_priority, time_priority, V_priority))

    # 4. Gauss-Seidel (forward)
    start = time.time()
    V_gs_fwd, iters_gs_fwd, updates_gs_fwd = gauss_seidel_value_iteration(mdp, theta, 'forward')
    time_gs_fwd = time.time() - start
    results.append(('GS (forward)', iters_gs_fwd, updates_gs_fwd, time_gs_fwd, V_gs_fwd))

    # 5. Gauss-Seidel (toward goal)
    start = time.time()
    V_gs_goal, iters_gs_goal, updates_gs_goal = gauss_seidel_value_iteration(mdp, theta, 'toward_goal')
    time_gs_goal = time.time() - start
    results.append(('GS (toward goal)', iters_gs_goal, updates_gs_goal, time_gs_goal, V_gs_goal))

    # 6. RTDP
    start_state = mdp.n_states - mdp.size  # Bottom-left
    start = time.time()
    V_rtdp, updates_rtdp = rtdp(mdp, start_state, n_trials=200)
    time_rtdp = time.time() - start
    results.append(('RTDP (200 trials)', 200, updates_rtdp, time_rtdp, V_rtdp))

    # Print comparison table
    print("\n  Method              | Iterations |  Updates  |  Time (ms) | V(start)")
    print("  " + "-" * 68)

    for name, iters, updates, t, V in results:
        start_value = V[mdp.n_states - mdp.size]  # Bottom-left
        print(f"  {name:20s}|   {iters:6d}   |  {updates:7d}  |  {t * 1000:8.2f}  | {start_value:.4f}")

    # Verify all methods converge to same values
    print("\n  Value Function Comparison (vs Synchronous):")
    print("  " + "-" * 50)

    reference = results[0][4]  # Synchronous VI values
    for name, _, _, _, V in results[1:]:
        max_diff = np.max(np.abs(V - reference))
        matches = max_diff < 0.01
        print(f"  {name:20s}: max diff = {max_diff:.6f}, matches = {matches}")

    return results


def main():
    print("=" * 70)
    print("ASYNCHRONOUS DYNAMIC PROGRAMMING")
    print("=" * 70)

    print("""
    Standard DP (synchronous) updates ALL states in each iteration.
    Asynchronous DP methods can be more efficient:

    1. IN-PLACE VI
       - Update values immediately during sweep
       - Later states use updated values
       - Simple modification, often faster

    2. PRIORITIZED SWEEPING
       - Priority queue by Bellman error
       - Always update state with largest expected change
       - Propagate changes to predecessors
       - Can be much faster for large state spaces

    3. GAUSS-SEIDEL VI
       - In-place with specific sweep orders
       - 'Toward goal' order often helps

    4. REAL-TIME DP (RTDP)
       - Only update states visited during simulation
       - Focuses on relevant parts of state space
       - Useful when some states are rarely reached
    """)

    # Compare on different grid sizes
    for size in [5, 10, 20]:
        compare_methods(size)

    # ============================================
    # DETAILED ANALYSIS
    # ============================================
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS: 10x10 GRID")
    print("=" * 70)

    mdp = GridWorldMDP(size=10, gamma=0.9)

    # Show update pattern for prioritized sweeping
    print("\nPrioritized Sweeping Update Pattern:")
    print("-" * 50)

    V = np.zeros(mdp.n_states)
    update_order = []
    total_updates = 0
    theta = 1e-6

    # Initialize priority queue
    pq = []
    for s in mdp.states:
        if s in mdp.terminal_states:
            continue
        action_values = []
        for a in mdp.actions:
            q = sum(p * (r + mdp.gamma * V[sp]) for p, sp, r in mdp.P[s][a])
            action_values.append(q)
        priority = abs(max(action_values) - V[s])
        if priority > theta:
            heapq.heappush(pq, (-priority, s))

    in_queue = {s for _, s in pq}

    # First 20 updates
    print("\nFirst 20 updates:")
    for i in range(min(20, len(pq))):
        if not pq:
            break
        neg_priority, s = heapq.heappop(pq)
        in_queue.discard(s)

        row, col = mdp._state_to_coord(s)
        print(f"  Update {i + 1}: State {s:3d} (row={row}, col={col}), priority={-neg_priority:.6f}")

        # Update
        action_values = []
        for a in mdp.actions:
            q = sum(p * (r + mdp.gamma * V[sp]) for p, sp, r in mdp.P[s][a])
            action_values.append(q)
        V[s] = max(action_values)
        update_order.append(s)

    print("\nObservation: Prioritized sweeping updates states near goal first!")
    print("(Because they have the largest initial Bellman errors)")

    # ============================================
    # SWEEP ORDER EFFECT
    # ============================================
    print("\n" + "=" * 70)
    print("SWEEP ORDER EFFECT ON CONVERGENCE")
    print("=" * 70)

    mdp = GridWorldMDP(size=10, gamma=0.9)
    theta = 1e-6

    print("\nGauss-Seidel VI with different sweep orders:")
    print("\n  Order         | Iterations |  Updates")
    print("  " + "-" * 40)

    for order in ['forward', 'backward', 'toward_goal']:
        _, iters, updates = gauss_seidel_value_iteration(mdp, theta, order)
        print(f"  {order:14s}|   {iters:6d}   |  {updates:7d}")

    print("""
    The 'toward_goal' order converges fastest because:
    - Values propagate from goal to other states
    - Updating states near goal first provides better estimates
    - Later updates use more accurate values
    """)

    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("""
    Asynchronous DP Key Takeaways:

    1. IN-PLACE UPDATES
       - Simple change: V[s] = ... instead of V_new[s] = ...
       - Often 2x fewer iterations
       - No extra memory needed

    2. PRIORITIZED SWEEPING
       - Best for problems with localized value changes
       - Updates can be orders of magnitude fewer
       - Requires predecessor computation

    3. SWEEP ORDER MATTERS
       - Update states "closer" to reward first
       - Information flows toward start state
       - Can dramatically reduce iterations

    4. RTDP FOR SELECTIVE EXPLORATION
       - Only updates relevant states
       - Good when state space is large but reachable states limited
       - Combines DP with simulation

    5. ALL METHODS CONVERGE TO SAME VALUES
       - Async DP is still DP
       - Same optimal policy guaranteed
       - Just more efficient computation

    When to use which:
    - Small state space: Synchronous is fine
    - Large state space: Prioritized sweeping
    - Sparse reachability: RTDP
    - General speedup: In-place + smart sweep order
    """)


if __name__ == "__main__":
    main()
