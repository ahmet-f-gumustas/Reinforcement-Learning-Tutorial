"""
03 - Bellman Equations: Computing Value Functions

This example demonstrates:
- Bellman Expectation Equation
- Computing state value function V(s)
- Computing action value function Q(s,a)
- Bellman Optimality Equation
- Relationship between V and Q
"""

import numpy as np
from typing import Dict, List, Tuple


class SimpleMDP:
    """
    A simple 5-state MDP for demonstrating Bellman equations.

    States: S0 (start) -> S1 -> S2 -> S3 -> S4 (goal)

    Transitions:
    - From each state, can go Right (to next state) or Stay
    - S4 is terminal (goal state)
    """

    def __init__(self, gamma: float = 0.9):
        self.n_states = 5
        self.states = list(range(self.n_states))
        self.actions = [0, 1]  # 0: Stay, 1: Right
        self.action_names = {0: "Stay", 1: "Right"}
        self.gamma = gamma
        self.terminal_state = 4

        # Transition probabilities: P[s][a] = [(prob, next_state), ...]
        # With some stochasticity to make it interesting
        self.P = self._build_transitions()
        self.R = self._build_rewards()

    def _build_transitions(self) -> Dict:
        P = {}
        for s in self.states:
            P[s] = {}
            if s == self.terminal_state:
                # Terminal state stays in place
                for a in self.actions:
                    P[s][a] = [(1.0, s)]
            else:
                # Stay action: 90% stay, 10% move right
                P[s][0] = [(0.9, s), (0.1, s + 1)]
                # Right action: 80% move right, 20% stay
                P[s][1] = [(0.2, s), (0.8, s + 1)]
        return P

    def _build_rewards(self) -> Dict:
        # Reward for reaching terminal state
        R = {}
        for s in self.states:
            R[s] = {}
            for a in self.actions:
                R[s][a] = {}
                for prob, s_prime in self.P[s][a]:
                    if s_prime == self.terminal_state and s != self.terminal_state:
                        R[s][a][s_prime] = 10.0  # Big reward for reaching goal
                    elif s == self.terminal_state:
                        R[s][a][s_prime] = 0.0  # No reward in terminal
                    else:
                        R[s][a][s_prime] = -1.0  # Small penalty for each step
        return R


def compute_return(rewards: List[float], gamma: float) -> float:
    """Compute discounted return G = sum(gamma^t * r_t)."""
    G = 0
    for t, r in enumerate(rewards):
        G += (gamma ** t) * r
    return G


def policy_evaluation(mdp: SimpleMDP, policy: np.ndarray,
                      theta: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
    """
    Policy Evaluation using Bellman Expectation Equation.

    Computes V_pi(s) for a given policy pi.

    Bellman Expectation Equation:
    V_pi(s) = sum_a pi(a|s) * sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V_pi(s')]
    """
    V = np.zeros(mdp.n_states)

    for iteration in range(max_iter):
        V_old = V.copy()

        for s in mdp.states:
            if s == mdp.terminal_state:
                V[s] = 0
                continue

            # Get action from policy
            a = policy[s]

            # Bellman expectation equation
            v = 0
            for prob, s_prime in mdp.P[s][a]:
                r = mdp.R[s][a][s_prime]
                v += prob * (r + mdp.gamma * V_old[s_prime])

            V[s] = v

        # Check convergence
        delta = np.max(np.abs(V - V_old))
        if delta < theta:
            print(f"  Policy evaluation converged in {iteration + 1} iterations")
            break

    return V


def compute_q_from_v(mdp: SimpleMDP, V: np.ndarray) -> np.ndarray:
    """
    Compute Q(s,a) from V(s) using the relationship:

    Q(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
    """
    Q = np.zeros((mdp.n_states, len(mdp.actions)))

    for s in mdp.states:
        for a in mdp.actions:
            q = 0
            for prob, s_prime in mdp.P[s][a]:
                r = mdp.R[s][a][s_prime]
                q += prob * (r + mdp.gamma * V[s_prime])
            Q[s, a] = q

    return Q


def value_iteration(mdp: SimpleMDP, theta: float = 1e-6,
                    max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value Iteration using Bellman Optimality Equation.

    Bellman Optimality Equation:
    V*(s) = max_a sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V*(s')]
    """
    V = np.zeros(mdp.n_states)

    for iteration in range(max_iter):
        V_old = V.copy()

        for s in mdp.states:
            if s == mdp.terminal_state:
                V[s] = 0
                continue

            # Bellman optimality: take max over actions
            action_values = []
            for a in mdp.actions:
                v = 0
                for prob, s_prime in mdp.P[s][a]:
                    r = mdp.R[s][a][s_prime]
                    v += prob * (r + mdp.gamma * V_old[s_prime])
                action_values.append(v)

            V[s] = max(action_values)

        delta = np.max(np.abs(V - V_old))
        if delta < theta:
            print(f"  Value iteration converged in {iteration + 1} iterations")
            break

    # Extract optimal policy
    policy = np.zeros(mdp.n_states, dtype=int)
    for s in mdp.states:
        if s == mdp.terminal_state:
            continue

        action_values = []
        for a in mdp.actions:
            v = 0
            for prob, s_prime in mdp.P[s][a]:
                r = mdp.R[s][a][s_prime]
                v += prob * (r + mdp.gamma * V[s_prime])
            action_values.append(v)

        policy[s] = np.argmax(action_values)

    return V, policy


def main():
    # ============================================
    # 1. UNDERSTANDING RETURNS
    # ============================================
    print("=" * 60)
    print("1. UNDERSTANDING RETURNS (G)")
    print("=" * 60)

    print("""
    The Return G_t is the total discounted reward from time t:

    G_t = R_{t+1} + gamma*R_{t+2} + gamma^2*R_{t+3} + ...
        = sum_{k=0}^{inf} gamma^k * R_{t+k+1}
    """)

    # Example rewards sequence
    rewards = [-1, -1, -1, 10]  # 3 steps, then goal
    gamma = 0.9

    print(f"\nExample: Rewards = {rewards}, gamma = {gamma}")
    print("\nComputing return at each time step:")

    for t in range(len(rewards)):
        future_rewards = rewards[t:]
        G = compute_return(future_rewards, gamma)
        terms = [f"{gamma}^{k}*{r}" for k, r in enumerate(future_rewards)]
        print(f"  G_{t} = {' + '.join(terms)} = {G:.3f}")

    # ============================================
    # 2. CREATE MDP
    # ============================================
    print("\n" + "=" * 60)
    print("2. CREATE SIMPLE MDP")
    print("=" * 60)

    mdp = SimpleMDP(gamma=0.9)

    print(f"""
    Simple 5-state MDP:

    S0 --> S1 --> S2 --> S3 --> S4 (Goal)

    States: {mdp.states}
    Actions: Stay (0), Right (1)
    Gamma: {mdp.gamma}
    Terminal: S4

    Rewards:
    - Reaching S4: +10
    - Other transitions: -1

    Transitions (stochastic):
    - Stay: 90% stay, 10% move right
    - Right: 80% move right, 20% stay
    """)

    # ============================================
    # 3. BELLMAN EXPECTATION EQUATION
    # ============================================
    print("=" * 60)
    print("3. BELLMAN EXPECTATION EQUATION")
    print("=" * 60)

    print("""
    The Bellman Expectation Equation for V_pi(s):

    V_pi(s) = sum_a pi(a|s) * sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V_pi(s')]

    This tells us the value of a state under policy pi equals:
    - The expected immediate reward, plus
    - The discounted expected value of the next state
    """)

    # Evaluate a "always go right" policy
    print("\nEvaluating 'Always Right' policy:")
    right_policy = np.ones(mdp.n_states, dtype=int)  # Always action 1 (Right)
    V_right = policy_evaluation(mdp, right_policy)

    print("\n  State Values V_pi(s):")
    for s in mdp.states:
        print(f"    V({s}) = {V_right[s]:.3f}")

    # Evaluate a "always stay" policy
    print("\nEvaluating 'Always Stay' policy:")
    stay_policy = np.zeros(mdp.n_states, dtype=int)  # Always action 0 (Stay)
    V_stay = policy_evaluation(mdp, stay_policy)

    print("\n  State Values V_pi(s):")
    for s in mdp.states:
        print(f"    V({s}) = {V_stay[s]:.3f}")

    print("\n  Observation: 'Always Right' policy has higher values!")
    print("  This makes sense - going right reaches the goal faster.")

    # ============================================
    # 4. ACTION VALUE FUNCTION Q(s,a)
    # ============================================
    print("\n" + "=" * 60)
    print("4. ACTION VALUE FUNCTION Q(s,a)")
    print("=" * 60)

    print("""
    Q_pi(s,a) = Expected return starting from s, taking action a,
                then following policy pi.

    Relationship with V:
    Q_pi(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V_pi(s')]
    V_pi(s) = sum_a pi(a|s) * Q_pi(s,a)
    """)

    # Compute Q from V for the "right" policy
    Q_right = compute_q_from_v(mdp, V_right)

    print("\nQ-values for 'Always Right' policy:")
    print("\n  State |   Q(s, Stay)  |  Q(s, Right)")
    print("  " + "-" * 40)
    for s in mdp.states:
        print(f"    S{s}  |    {Q_right[s, 0]:7.3f}    |   {Q_right[s, 1]:7.3f}")

    print("\n  At each state, Q(s, Right) > Q(s, Stay)")
    print("  This confirms 'Right' is better than 'Stay'.")

    # ============================================
    # 5. BELLMAN OPTIMALITY EQUATION
    # ============================================
    print("\n" + "=" * 60)
    print("5. BELLMAN OPTIMALITY EQUATION")
    print("=" * 60)

    print("""
    The Bellman Optimality Equation for V*(s):

    V*(s) = max_a sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V*(s')]

    And for Q*(s,a):

    Q*(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * max_a' Q*(s',a')]

    The optimal value function gives the maximum possible return.
    """)

    print("\nRunning Value Iteration to find V* and optimal policy:")
    V_star, optimal_policy = value_iteration(mdp)

    print("\n  Optimal State Values V*(s):")
    for s in mdp.states:
        print(f"    V*({s}) = {V_star[s]:.3f}")

    print("\n  Optimal Policy:")
    for s in mdp.states:
        if s == mdp.terminal_state:
            print(f"    State {s}: Terminal")
        else:
            print(f"    State {s}: {mdp.action_names[optimal_policy[s]]}")

    # ============================================
    # 6. COMPARISON: V_pi vs V*
    # ============================================
    print("\n" + "=" * 60)
    print("6. COMPARISON: V_pi vs V*")
    print("=" * 60)

    print("\n  State | V(Stay) | V(Right) |   V*")
    print("  " + "-" * 45)
    for s in mdp.states:
        print(f"    S{s}  |  {V_stay[s]:6.3f}  |  {V_right[s]:6.3f}   | {V_star[s]:6.3f}")

    print("""
    Observations:
    - V* >= V_pi for any policy pi (optimal is always best)
    - V(Right) is close to V* (good policy)
    - V(Stay) is much lower (poor policy)
    """)

    # ============================================
    # 7. MANUAL BELLMAN CALCULATION
    # ============================================
    print("=" * 60)
    print("7. MANUAL BELLMAN CALCULATION EXAMPLE")
    print("=" * 60)

    print("""
    Let's manually compute V(S3) using the Bellman equation.

    For state S3 with 'Right' action:
    - 80% chance: go to S4 (terminal), get reward +10
    - 20% chance: stay at S3, get reward -1

    V_pi(S3) = 0.8 * (10 + 0.9 * V(S4)) + 0.2 * (-1 + 0.9 * V(S3))

    Since V(S4) = 0 (terminal state):
    V_pi(S3) = 0.8 * 10 + 0.2 * (-1 + 0.9 * V(S3))
    V_pi(S3) = 8 - 0.2 + 0.18 * V(S3)
    V_pi(S3) - 0.18 * V(S3) = 7.8
    0.82 * V_pi(S3) = 7.8
    V_pi(S3) = 9.51
    """)

    print(f"  Manual calculation: V(S3) = 9.51")
    print(f"  Our algorithm:      V(S3) = {V_right[3]:.2f}")

    # ============================================
    # 8. SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("8. SUMMARY: BELLMAN EQUATIONS")
    print("=" * 60)

    print("""
    Bellman Expectation Equation:
    ┌─────────────────────────────────────────────────────────────┐
    │ V_pi(s) = E_pi[R + gamma * V_pi(S') | S = s]                │
    │                                                              │
    │ Evaluates how good a state is under a specific policy       │
    └─────────────────────────────────────────────────────────────┘

    Bellman Optimality Equation:
    ┌─────────────────────────────────────────────────────────────┐
    │ V*(s) = max_a E[R + gamma * V*(S') | S = s, A = a]          │
    │                                                              │
    │ Gives the maximum possible value for each state             │
    └─────────────────────────────────────────────────────────────┘

    Key Relationships:
    - Q(s,a) -> V(s): V(s) = sum_a pi(a|s) * Q(s,a)
    - V(s) -> Q(s,a): Q(s,a) = R(s,a) + gamma * E[V(s')]
    - V* and Q* satisfy the optimality equations

    Next Steps:
    - Week 3: Dynamic Programming (Policy Iteration, Value Iteration)
    - Week 4: Monte Carlo methods (sample-based)
    - Week 5: TD Learning (combining DP and MC ideas)
    """)


if __name__ == "__main__":
    main()
