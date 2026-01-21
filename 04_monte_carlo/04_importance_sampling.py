"""
04 - Importance Sampling for Off-Policy MC

Learn about one policy (target) while following another (behavior).

Demonstrates:
- Ordinary Importance Sampling
- Weighted Importance Sampling
- Off-Policy MC Prediction
- Variance comparison
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class SimpleGridWorld:
    """Simple environment for off-policy learning demonstration."""

    def __init__(self, size: int = 4, gamma: float = 0.9):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.gamma = gamma

        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left

        self.start_state = self.n_states - self.size
        self.goal_state = self.size - 1

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
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


def generate_episode_behavior(env: SimpleGridWorld, behavior_policy: Dict,
                               start_state: int = None,
                               max_steps: int = 100) -> List[Tuple]:
    """Generate episode using behavior policy."""
    if start_state is None:
        start_state = env.start_state

    episode = []
    state = start_state
    done = False
    steps = 0

    while not done and steps < max_steps:
        # Sample action from behavior policy
        probs = behavior_policy.get(state, np.ones(env.n_actions) / env.n_actions)
        action = np.random.choice(env.n_actions, p=probs)

        next_state, reward, done = env.step(state, action)
        episode.append((state, action, reward))
        state = next_state
        steps += 1

    return episode


def compute_importance_ratio(episode: List[Tuple], target_policy: Dict,
                              behavior_policy: Dict, start_idx: int = 0) -> float:
    """
    Compute importance sampling ratio for episode from start_idx.

    rho = product_{t=start}^{T-1} [pi(A_t|S_t) / b(A_t|S_t)]
    """
    rho = 1.0
    for t in range(start_idx, len(episode)):
        state, action, _ = episode[t]

        # Get probabilities
        pi_prob = target_policy.get(state, np.ones(4) / 4)[action]
        b_prob = behavior_policy.get(state, np.ones(4) / 4)[action]

        if b_prob == 0:
            return 0.0  # Cannot happen under behavior policy

        rho *= pi_prob / b_prob

    return rho


def ordinary_importance_sampling(returns: List[float], ratios: List[float]) -> float:
    """
    Ordinary Importance Sampling.

    V(s) = (1/n) * sum(rho * G)

    Unbiased but can have high variance.
    """
    if len(returns) == 0:
        return 0.0

    weighted_returns = [r * g for r, g in zip(ratios, returns)]
    return np.mean(weighted_returns)


def weighted_importance_sampling(returns: List[float], ratios: List[float]) -> float:
    """
    Weighted Importance Sampling.

    V(s) = sum(rho * G) / sum(rho)

    Biased but much lower variance.
    """
    if len(returns) == 0 or sum(ratios) == 0:
        return 0.0

    numerator = sum(r * g for r, g in zip(ratios, returns))
    denominator = sum(ratios)

    return numerator / denominator


def off_policy_mc_prediction(env: SimpleGridWorld, target_policy: Dict,
                              behavior_policy: Dict, n_episodes: int = 10000,
                              gamma: float = 0.9) -> Tuple[Dict, Dict, Dict]:
    """
    Off-Policy MC Prediction using importance sampling.

    Returns both ordinary and weighted IS estimates.
    """
    # Store returns and ratios for each state
    returns_ois = defaultdict(list)
    ratios_ois = defaultdict(list)

    # For weighted IS (incremental)
    V_wis = defaultdict(float)
    C = defaultdict(float)  # Cumulative sum of weights

    for episode_num in range(n_episodes):
        episode = generate_episode_behavior(env, behavior_policy)

        G = 0
        W = 1.0

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            # Update weighted IS (incremental)
            C[state] += W
            V_wis[state] += (W / C[state]) * (G - V_wis[state])

            # Store for ordinary IS
            rho = compute_importance_ratio(episode, target_policy, behavior_policy, t)
            returns_ois[state].append(G)
            ratios_ois[state].append(rho)

            # Update weight for next state
            pi_prob = target_policy.get(state, np.ones(4) / 4)[action]
            b_prob = behavior_policy.get(state, np.ones(4) / 4)[action]

            if b_prob == 0:
                break

            W *= pi_prob / b_prob

            if W == 0:
                break

    # Compute ordinary IS estimates
    V_ois = {}
    for state in returns_ois:
        V_ois[state] = ordinary_importance_sampling(
            returns_ois[state], ratios_ois[state])

    return dict(V_ois), dict(V_wis), dict(returns_ois)


def on_policy_mc_prediction(env: SimpleGridWorld, policy: Dict,
                             n_episodes: int = 10000,
                             gamma: float = 0.9) -> Dict:
    """On-policy MC prediction for comparison."""
    V = defaultdict(float)
    N = defaultdict(int)

    for _ in range(n_episodes):
        episode = generate_episode_behavior(env, policy)

        visited = set()
        G = 0

        for t in range(len(episode) - 1, -1, -1):
            state, _, reward = episode[t]
            G = gamma * G + reward

            if state not in visited:
                visited.add(state)
                N[state] += 1
                V[state] += (G - V[state]) / N[state]

    return dict(V)


def create_deterministic_policy(env: SimpleGridWorld) -> Dict:
    """Create a deterministic policy (always go toward goal)."""
    policy = {}
    goal_row, goal_col = env._state_to_coord(env.goal_state)

    for s in range(env.n_states):
        probs = np.zeros(env.n_actions)
        row, col = env._state_to_coord(s)

        if row > goal_row:
            probs[0] = 1.0  # Up
        elif col < goal_col:
            probs[1] = 1.0  # Right
        else:
            probs[0] = 1.0  # Default: Up

        policy[s] = probs

    return policy


def create_uniform_policy(env: SimpleGridWorld) -> Dict:
    """Create a uniform random policy."""
    policy = {}
    for s in range(env.n_states):
        policy[s] = np.ones(env.n_actions) / env.n_actions
    return policy


def create_soft_policy(env: SimpleGridWorld, epsilon: float = 0.2) -> Dict:
    """Create epsilon-soft version of deterministic policy."""
    det_policy = create_deterministic_policy(env)
    soft_policy = {}

    for s in range(env.n_states):
        probs = det_policy[s].copy()
        # Add epsilon softness
        probs = (1 - epsilon) * probs + epsilon / env.n_actions
        soft_policy[s] = probs

    return soft_policy


def compute_variance(returns: List[float], ratios: List[float],
                     true_value: float) -> Tuple[float, float]:
    """Compute variance of OIS and WIS estimates."""
    if len(returns) < 2:
        return 0.0, 0.0

    # OIS variance
    weighted = [r * g for r, g in zip(ratios, returns)]
    ois_var = np.var(weighted)

    # WIS effective sample estimates
    # (Harder to compute, simplified approximation)
    wis_estimates = []
    for i in range(1, len(returns)):
        subset_returns = returns[:i + 1]
        subset_ratios = ratios[:i + 1]
        if sum(subset_ratios) > 0:
            est = sum(r * g for r, g in zip(subset_ratios, subset_returns)) / sum(subset_ratios)
            wis_estimates.append(est)

    wis_var = np.var(wis_estimates) if len(wis_estimates) > 1 else 0.0

    return ois_var, wis_var


def main():
    # ============================================
    # 1. INTRODUCTION
    # ============================================
    print("=" * 60)
    print("IMPORTANCE SAMPLING FOR OFF-POLICY MC")
    print("=" * 60)

    print("""
    Off-Policy Learning: Learn about target policy pi while
    following behavior policy b.

    Why?
    - Learn from expert demonstrations
    - Reuse data from old policies
    - Learn optimal policy while exploring

    Challenge: Returns come from b, not pi!

    Solution: Importance Sampling

    Importance Sampling Ratio:
    rho = product [pi(a|s) / b(a|s)]

    This corrects for the distribution mismatch.
    """)

    env = SimpleGridWorld(size=4, gamma=0.9)
    print(f"\nGrid World: {env.size}x{env.size}")

    # ============================================
    # 2. DEFINE POLICIES
    # ============================================
    print("\n" + "=" * 60)
    print("2. DEFINE POLICIES")
    print("=" * 60)

    # Target policy: deterministic toward goal
    target_policy = create_deterministic_policy(env)
    print("\nTarget Policy (pi): Deterministic - always toward goal")

    # Behavior policy: uniform random
    behavior_policy = create_uniform_policy(env)
    print("Behavior Policy (b): Uniform random")

    print("""
    We want to estimate V_pi (value under target policy)
    but we can only generate data using b (behavior policy).
    """)

    # ============================================
    # 3. IMPORTANCE SAMPLING RATIO EXAMPLE
    # ============================================
    print("\n" + "=" * 60)
    print("3. IMPORTANCE SAMPLING RATIO EXAMPLE")
    print("=" * 60)

    # Generate a sample episode
    episode = generate_episode_behavior(env, behavior_policy, max_steps=10)

    print(f"\nSample episode (first 5 steps):")
    for i, (s, a, r) in enumerate(episode[:5]):
        action_names = ["Up", "Right", "Down", "Left"]
        pi_prob = target_policy[s][a]
        b_prob = behavior_policy[s][a]
        ratio = pi_prob / b_prob

        print(f"\n  Step {i}: State {s}, Action {action_names[a]}")
        print(f"    pi({action_names[a]}|{s}) = {pi_prob:.2f}")
        print(f"    b({action_names[a]}|{s}) = {b_prob:.2f}")
        print(f"    Ratio = {ratio:.2f}")

    # Full episode ratio
    full_ratio = compute_importance_ratio(episode, target_policy, behavior_policy, 0)
    print(f"\n  Full episode importance ratio: {full_ratio:.4f}")

    print("""
    Note: Ratio can be very large or very small!
    - If behavior takes unlikely target actions: ratio > 1
    - If behavior takes likely target actions: ratio could be huge
    - Product over many steps can explode or vanish
    """)

    # ============================================
    # 4. OFF-POLICY MC PREDICTION
    # ============================================
    print("\n" + "=" * 60)
    print("4. OFF-POLICY MC PREDICTION")
    print("=" * 60)

    print("\nRunning off-policy MC (10,000 episodes)...")
    V_ois, V_wis, all_returns = off_policy_mc_prediction(
        env, target_policy, behavior_policy, n_episodes=10000)

    # On-policy for comparison (ground truth)
    print("Running on-policy MC with target policy...")
    V_true = on_policy_mc_prediction(env, target_policy, n_episodes=50000)

    # Compare at start state
    start = env.start_state
    print(f"\nValue estimates at start state ({start}):")
    print(f"  On-Policy (ground truth):  {V_true.get(start, 0):.4f}")
    print(f"  Off-Policy (Ordinary IS):  {V_ois.get(start, 0):.4f}")
    print(f"  Off-Policy (Weighted IS):  {V_wis.get(start, 0):.4f}")

    # ============================================
    # 5. OIS VS WIS COMPARISON
    # ============================================
    print("\n" + "=" * 60)
    print("5. ORDINARY VS WEIGHTED IMPORTANCE SAMPLING")
    print("=" * 60)

    print("""
    Ordinary IS: V = (1/n) * sum(rho * G)
    - Unbiased estimator
    - Can have VERY high variance

    Weighted IS: V = sum(rho * G) / sum(rho)
    - Biased (but bias -> 0 as n -> infinity)
    - Much lower variance
    - Generally preferred in practice
    """)

    # Compare errors
    print("\nComparison across all states:")
    print("\n  State | True V  |  OIS   |  WIS   | OIS Err | WIS Err")
    print("  " + "-" * 55)

    ois_errors = []
    wis_errors = []

    for s in sorted(V_true.keys()):
        if s == env.goal_state:
            continue

        v_true = V_true.get(s, 0)
        v_ois = V_ois.get(s, 0)
        v_wis = V_wis.get(s, 0)

        err_ois = abs(v_true - v_ois)
        err_wis = abs(v_true - v_wis)

        ois_errors.append(err_ois)
        wis_errors.append(err_wis)

        print(f"   {s:2d}   | {v_true:6.3f} | {v_ois:6.3f} | {v_wis:6.3f} | {err_ois:6.3f}  | {err_wis:6.3f}")

    print(f"\n  Mean Absolute Error:")
    print(f"    Ordinary IS: {np.mean(ois_errors):.4f}")
    print(f"    Weighted IS: {np.mean(wis_errors):.4f}")

    # ============================================
    # 6. VARIANCE ANALYSIS
    # ============================================
    print("\n" + "=" * 60)
    print("6. VARIANCE ANALYSIS")
    print("=" * 60)

    print("\nRunning multiple experiments to estimate variance...")

    n_experiments = 20
    ois_estimates = []
    wis_estimates = []

    for _ in range(n_experiments):
        v_ois, v_wis, _ = off_policy_mc_prediction(
            env, target_policy, behavior_policy, n_episodes=1000)
        ois_estimates.append(v_ois.get(start, 0))
        wis_estimates.append(v_wis.get(start, 0))

    print(f"\nV(start) estimates over {n_experiments} runs:")
    print(f"\n  Method      | Mean   | Std Dev | True Value")
    print("  " + "-" * 45)
    print(f"  Ordinary IS | {np.mean(ois_estimates):6.3f} | {np.std(ois_estimates):6.3f}  | {V_true.get(start, 0):.3f}")
    print(f"  Weighted IS | {np.mean(wis_estimates):6.3f} | {np.std(wis_estimates):6.3f}  | {V_true.get(start, 0):.3f}")

    print("""
    Observation: Weighted IS has MUCH lower variance!
    This is why it's preferred in practice despite being biased.
    """)

    # ============================================
    # 7. EFFECT OF BEHAVIOR POLICY
    # ============================================
    print("\n" + "=" * 60)
    print("7. EFFECT OF BEHAVIOR POLICY")
    print("=" * 60)

    print("\nComparing different behavior policies:")
    print("(Target is always the deterministic policy)")

    print("\n  Behavior Policy     | MAE (OIS) | MAE (WIS)")
    print("  " + "-" * 45)

    for eps, name in [(0.0, "Deterministic (same)"),
                      (0.2, "Soft (eps=0.2)"),
                      (0.5, "Soft (eps=0.5)"),
                      (1.0, "Uniform random")]:

        if eps == 0.0:
            b_policy = target_policy
        elif eps == 1.0:
            b_policy = create_uniform_policy(env)
        else:
            b_policy = create_soft_policy(env, epsilon=eps)

        v_ois, v_wis, _ = off_policy_mc_prediction(
            env, target_policy, b_policy, n_episodes=5000)

        errors_ois = [abs(V_true.get(s, 0) - v_ois.get(s, 0))
                      for s in V_true if s != env.goal_state]
        errors_wis = [abs(V_true.get(s, 0) - v_wis.get(s, 0))
                      for s in V_true if s != env.goal_state]

        print(f"  {name:20s} |   {np.mean(errors_ois):.4f}   |   {np.mean(errors_wis):.4f}")

    print("""
    Observations:
    - When b = pi (on-policy), no importance sampling needed
    - Closer behavior to target -> lower variance
    - Uniform random behavior -> highest variance
    """)

    # ============================================
    # 8. COVERAGE REQUIREMENT
    # ============================================
    print("\n" + "=" * 60)
    print("8. COVERAGE REQUIREMENT")
    print("=" * 60)

    print("""
    IMPORTANT: Behavior policy must have COVERAGE over target!

    Coverage: If pi(a|s) > 0, then b(a|s) > 0

    If behavior never takes an action that target would take,
    we can't estimate the value of that action!

    In our example:
    - Target: Deterministic (one action per state)
    - Uniform: Covers all actions (25% each)
    - Coverage is satisfied

    If target took action A, but behavior never takes A:
    - Ratio would be pi(A)/b(A) = positive/0 = undefined!
    - Algorithm would fail
    """)

    # ============================================
    # 9. SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("9. SUMMARY")
    print("=" * 60)

    print("""
    Importance Sampling for Off-Policy MC:

    1. THE PROBLEM
       - Want V_pi, but data comes from b
       - Returns under b != Returns under pi

    2. IMPORTANCE SAMPLING RATIO
       rho = product [pi(a|s) / b(a|s)]
       Corrects for distribution mismatch

    3. ORDINARY IS
       V = (1/n) * sum(rho * G)
       - Unbiased
       - High variance (can be extreme)

    4. WEIGHTED IS
       V = sum(rho * G) / sum(rho)
       - Biased (asymptotically unbiased)
       - Much lower variance
       - Preferred in practice

    5. COVERAGE REQUIREMENT
       b(a|s) > 0 wherever pi(a|s) > 0
       (Behavior must cover target's actions)

    6. PRACTICAL TIPS
       - Use weighted IS
       - Make behavior close to target if possible
       - More data helps reduce variance
       - Truncate ratios if they explode

    Applications:
    - Learning from demonstrations
    - Batch/offline RL
    - Safe exploration (learn optimal while being careful)
    """)


if __name__ == "__main__":
    main()
