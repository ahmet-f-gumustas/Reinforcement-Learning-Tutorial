"""
01 - Monte Carlo Prediction

Learn to estimate value functions from episodes of experience.

Demonstrates:
- First-visit MC prediction
- Every-visit MC prediction
- Incremental mean updates
- Convergence analysis
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class SimpleGridWorld:
    """
    Simple Grid World for MC prediction demonstration.

    5x5 grid, goal at top-right, start at bottom-left.
    """

    def __init__(self, size: int = 5, gamma: float = 0.9):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma

        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.action_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

        self.start_state = self.n_states - self.size
        self.goal_state = self.size - 1

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        """Take action, return (next_state, reward, done)."""
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

    def generate_episode(self, policy: np.ndarray, start_state: int = None,
                         max_steps: int = 100) -> List[Tuple[int, int, float]]:
        """Generate an episode following the policy."""
        if start_state is None:
            start_state = self.start_state

        episode = []
        state = start_state
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = policy[state]
            next_state, reward, done = self.step(state, action)
            episode.append((state, action, reward))
            state = next_state
            steps += 1

        return episode

    def render_values(self, V: Dict[int, float], title: str = "Value Function"):
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
                    v = V.get(state, 0.0)
                    cell = f" {v:6.3f} "
                line += cell + "|"
            print(line)
            print("+" + "--------+" * self.size)


def create_simple_policy(env: SimpleGridWorld) -> np.ndarray:
    """Create a simple policy that moves toward the goal."""
    policy = np.zeros(env.n_states, dtype=int)
    goal_row, goal_col = env._state_to_coord(env.goal_state)

    for s in range(env.n_states):
        row, col = env._state_to_coord(s)

        if row > goal_row:
            policy[s] = 0  # Up
        elif col < goal_col:
            policy[s] = 1  # Right
        else:
            policy[s] = 0  # Default Up

    return policy


def first_visit_mc_prediction(env: SimpleGridWorld, policy: np.ndarray,
                               n_episodes: int = 1000,
                               gamma: float = 0.9) -> Tuple[Dict, List]:
    """
    First-Visit MC Prediction.

    Only uses the FIRST visit to each state in an episode.
    """
    V = defaultdict(float)
    returns = defaultdict(list)
    history = []

    for episode_num in range(n_episodes):
        episode = env.generate_episode(policy)

        # Track visited states in this episode
        visited = set()

        # Calculate returns (backward from end)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            # First-visit check
            if state not in visited:
                visited.add(state)
                returns[state].append(G)
                V[state] = np.mean(returns[state])

        # Record history for convergence analysis
        if episode_num % 100 == 0:
            history.append({s: V[s] for s in V})

    return dict(V), history


def every_visit_mc_prediction(env: SimpleGridWorld, policy: np.ndarray,
                               n_episodes: int = 1000,
                               gamma: float = 0.9) -> Tuple[Dict, List]:
    """
    Every-Visit MC Prediction.

    Uses EVERY visit to each state in an episode.
    """
    V = defaultdict(float)
    returns = defaultdict(list)
    history = []

    for episode_num in range(n_episodes):
        episode = env.generate_episode(policy)

        # Calculate returns (backward from end)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            # Every visit - always append
            returns[state].append(G)
            V[state] = np.mean(returns[state])

        if episode_num % 100 == 0:
            history.append({s: V[s] for s in V})

    return dict(V), history


def incremental_mc_prediction(env: SimpleGridWorld, policy: np.ndarray,
                               n_episodes: int = 1000,
                               gamma: float = 0.9,
                               alpha: float = None) -> Tuple[Dict, List]:
    """
    Incremental MC Prediction (constant-alpha or averaging).

    V(s) = V(s) + alpha * (G - V(s))

    If alpha=None, uses 1/N(s) for exact averaging.
    """
    V = defaultdict(float)
    N = defaultdict(int)  # Visit counts
    history = []

    for episode_num in range(n_episodes):
        episode = env.generate_episode(policy)

        visited = set()
        G = 0

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            if state not in visited:
                visited.add(state)
                N[state] += 1

                if alpha is None:
                    # Exact averaging
                    step_size = 1.0 / N[state]
                else:
                    step_size = alpha

                V[state] = V[state] + step_size * (G - V[state])

        if episode_num % 100 == 0:
            history.append({s: V[s] for s in V})

    return dict(V), history


def compute_true_values_dp(env: SimpleGridWorld, policy: np.ndarray,
                            gamma: float = 0.9) -> Dict[int, float]:
    """Compute true V_pi using DP for comparison."""
    V = np.zeros(env.n_states)

    for _ in range(1000):
        V_new = np.zeros(env.n_states)
        for s in range(env.n_states):
            if s == env.goal_state:
                continue
            a = policy[s]
            s_next, r, done = env.step(s, a)
            V_new[s] = r + gamma * V[s_next] * (1 - done)
        V = V_new

    return {s: V[s] for s in range(env.n_states)}


def main():
    # ============================================
    # 1. INTRODUCTION
    # ============================================
    print("=" * 60)
    print("MONTE CARLO PREDICTION")
    print("=" * 60)

    print("""
    Monte Carlo Prediction estimates V_pi(s) by averaging
    returns observed after visiting state s.

    Key idea: V(s) = E[G_t | S_t = s] ≈ (1/N) * sum(G)

    Two variants:
    - First-Visit MC: Only count first visit per episode
    - Every-Visit MC: Count all visits
    """)

    # Create environment and policy
    env = SimpleGridWorld(size=5, gamma=0.9)
    policy = create_simple_policy(env)

    print(f"\nGrid World: {env.size}x{env.size}")
    print(f"Goal: State {env.goal_state}")
    print(f"Gamma: {env.gamma}")

    # ============================================
    # 2. FIRST-VISIT MC PREDICTION
    # ============================================
    print("\n" + "=" * 60)
    print("2. FIRST-VISIT MC PREDICTION")
    print("=" * 60)

    print("\nRunning First-Visit MC (1000 episodes)...")
    V_first, history_first = first_visit_mc_prediction(
        env, policy, n_episodes=1000, gamma=0.9)

    env.render_values(V_first, "V_pi (First-Visit MC)")

    # ============================================
    # 3. EVERY-VISIT MC PREDICTION
    # ============================================
    print("\n" + "=" * 60)
    print("3. EVERY-VISIT MC PREDICTION")
    print("=" * 60)

    print("\nRunning Every-Visit MC (1000 episodes)...")
    V_every, history_every = every_visit_mc_prediction(
        env, policy, n_episodes=1000, gamma=0.9)

    env.render_values(V_every, "V_pi (Every-Visit MC)")

    # ============================================
    # 4. COMPARE WITH TRUE VALUES (DP)
    # ============================================
    print("\n" + "=" * 60)
    print("4. COMPARISON WITH TRUE VALUES (DP)")
    print("=" * 60)

    V_true = compute_true_values_dp(env, policy, gamma=0.9)
    env.render_values(V_true, "V_pi (True - from DP)")

    # Calculate errors
    states = [s for s in range(env.n_states) if s != env.goal_state]

    error_first = np.mean([abs(V_first.get(s, 0) - V_true[s]) for s in states])
    error_every = np.mean([abs(V_every.get(s, 0) - V_true[s]) for s in states])

    print(f"\nMean Absolute Error:")
    print(f"  First-Visit MC: {error_first:.4f}")
    print(f"  Every-Visit MC: {error_every:.4f}")

    # ============================================
    # 5. INCREMENTAL VS BATCH UPDATE
    # ============================================
    print("\n" + "=" * 60)
    print("5. INCREMENTAL VS BATCH UPDATE")
    print("=" * 60)

    print("""
    Batch: Store all returns, compute mean at end
    Incremental: V(s) = V(s) + alpha * (G - V(s))

    Benefits of incremental:
    - Constant memory
    - Can use constant alpha for non-stationary problems
    """)

    # Incremental with averaging
    print("\nIncremental MC (alpha = 1/N):")
    V_incr_avg, _ = incremental_mc_prediction(
        env, policy, n_episodes=1000, gamma=0.9, alpha=None)

    error_incr = np.mean([abs(V_incr_avg.get(s, 0) - V_true[s]) for s in states])
    print(f"  MAE: {error_incr:.4f}")

    # Incremental with constant alpha
    print("\nIncremental MC (alpha = 0.1):")
    V_incr_const, _ = incremental_mc_prediction(
        env, policy, n_episodes=1000, gamma=0.9, alpha=0.1)

    error_const = np.mean([abs(V_incr_const.get(s, 0) - V_true[s]) for s in states])
    print(f"  MAE: {error_const:.4f}")

    # ============================================
    # 6. CONVERGENCE ANALYSIS
    # ============================================
    print("\n" + "=" * 60)
    print("6. CONVERGENCE ANALYSIS")
    print("=" * 60)

    print("\nV(start) convergence over episodes:")
    start = env.start_state

    print("\n  Episodes | First-Visit |  Every-Visit |    True")
    print("  " + "-" * 50)

    for i, (h_first, h_every) in enumerate(zip(history_first, history_every)):
        ep = i * 100
        v_f = h_first.get(start, 0)
        v_e = h_every.get(start, 0)
        print(f"     {ep:4d}  |   {v_f:7.4f}   |    {v_e:7.4f}   |  {V_true[start]:.4f}")

    # ============================================
    # 7. EFFECT OF NUMBER OF EPISODES
    # ============================================
    print("\n" + "=" * 60)
    print("7. EFFECT OF NUMBER OF EPISODES")
    print("=" * 60)

    print("\nMAE vs Number of Episodes:")
    print("\n  Episodes |   MAE")
    print("  " + "-" * 22)

    for n_ep in [100, 500, 1000, 5000, 10000]:
        V_test, _ = first_visit_mc_prediction(env, policy, n_episodes=n_ep)
        mae = np.mean([abs(V_test.get(s, 0) - V_true[s]) for s in states])
        print(f"    {n_ep:5d}  | {mae:.4f}")

    print("\n  More episodes -> Lower error (MC is consistent)")

    # ============================================
    # 8. RANDOM POLICY EVALUATION
    # ============================================
    print("\n" + "=" * 60)
    print("8. EVALUATING A RANDOM POLICY")
    print("=" * 60)

    # Create random policy
    random_policy = np.random.randint(0, 4, size=env.n_states)

    print("\nEvaluating random policy with MC...")
    V_random, _ = first_visit_mc_prediction(
        env, random_policy, n_episodes=5000, gamma=0.9)

    V_random_true = compute_true_values_dp(env, random_policy, gamma=0.9)

    print(f"\nV(start) - Random policy:")
    print(f"  MC estimate: {V_random.get(env.start_state, 0):.4f}")
    print(f"  True value:  {V_random_true[env.start_state]:.4f}")

    print(f"\nV(start) - Smart policy:")
    print(f"  MC estimate: {V_first.get(env.start_state, 0):.4f}")
    print(f"  True value:  {V_true[env.start_state]:.4f}")

    print("\n  Smart policy has higher values (as expected)!")

    # ============================================
    # 9. SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("9. SUMMARY")
    print("=" * 60)

    print("""
    Monte Carlo Prediction:

    1. ALGORITHM
       - Generate complete episode
       - For each state visited, compute return G
       - Average returns to estimate V(s)

    2. FIRST-VISIT VS EVERY-VISIT
       - First-visit: Unbiased, one sample per episode per state
       - Every-visit: More samples, also converges

    3. INCREMENTAL UPDATE
       V(s) = V(s) + alpha * (G - V(s))
       - Constant memory
       - alpha=1/N for averaging, constant for non-stationary

    4. CONVERGENCE
       - Guaranteed to converge to V_pi
       - More episodes = better estimate
       - Standard error decreases as 1/sqrt(N)

    5. KEY ADVANTAGE
       - No model required!
       - Learn from experience directly

    Next: MC Control (finding optimal policy)
    """)


if __name__ == "__main__":
    main()
