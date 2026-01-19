"""
04 - Gambler's Problem

Classic RL problem from Sutton & Barto (Example 4.3).

A gambler bets on coin flips:
- Heads (prob p_h): wins the stake
- Tails (prob 1-p_h): loses the stake
- Goal: Reach $100 starting from some capital
- Episode ends at $0 (lose) or $100 (win)

This is a great example of:
- Value Iteration on a non-grid MDP
- How probability affects optimal policy
- Interesting policy structure
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class GamblersProblem:
    """
    Gambler's Problem MDP.

    States: 0, 1, 2, ..., 100 (capital in dollars)
    Actions: 1, 2, ..., min(s, 100-s) (stake amount)
    """

    def __init__(self, goal: int = 100, p_heads: float = 0.4):
        self.goal = goal
        self.p_heads = p_heads
        self.p_tails = 1 - p_heads

        # States: 0 to goal (inclusive)
        self.states = list(range(goal + 1))
        self.n_states = goal + 1

        # Terminal states
        self.terminal_states = {0, goal}

    def get_actions(self, state: int) -> List[int]:
        """Get valid actions (stakes) for a state."""
        if state in self.terminal_states:
            return [0]
        # Can bet 1 to min(capital, amount_to_win)
        max_stake = min(state, self.goal - state)
        return list(range(1, max_stake + 1))

    def get_transitions(self, state: int, action: int) -> List[Tuple[float, int, float]]:
        """
        Get transition probabilities.

        Returns: [(probability, next_state, reward), ...]
        """
        if state in self.terminal_states:
            return [(1.0, state, 0.0)]

        transitions = []

        # Win (heads)
        next_state_win = state + action
        reward_win = 1.0 if next_state_win == self.goal else 0.0
        transitions.append((self.p_heads, next_state_win, reward_win))

        # Lose (tails)
        next_state_lose = state - action
        transitions.append((self.p_tails, next_state_lose, 0.0))

        return transitions


def value_iteration(mdp: GamblersProblem, gamma: float = 1.0,
                    theta: float = 1e-9, max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Value Iteration for Gambler's Problem.

    Note: gamma=1.0 is typical for this episodic problem.
    """
    V = np.zeros(mdp.n_states)
    V[mdp.goal] = 1.0  # Winning state has value 1
    history = [V.copy()]

    for iteration in range(max_iter):
        delta = 0
        V_new = V.copy()

        for s in mdp.states:
            if s in mdp.terminal_states:
                continue

            action_values = []
            for a in mdp.get_actions(s):
                q = 0
                for prob, s_prime, reward in mdp.get_transitions(s, a):
                    q += prob * (reward + gamma * V[s_prime])
                action_values.append(q)

            V_new[s] = max(action_values)
            delta = max(delta, abs(V[s] - V_new[s]))

        V = V_new
        history.append(V.copy())

        if delta < theta:
            print(f"Converged in {iteration + 1} iterations")
            break

    # Extract policy (break ties by choosing smallest stake)
    policy = np.zeros(mdp.n_states, dtype=int)
    for s in mdp.states:
        if s in mdp.terminal_states:
            continue

        best_value = -float('inf')
        best_action = 1

        for a in mdp.get_actions(s):
            q = 0
            for prob, s_prime, reward in mdp.get_transitions(s, a):
                q += prob * (reward + gamma * V[s_prime])

            # Prefer smaller stakes when values are equal (risk averse)
            if q > best_value + 1e-9:
                best_value = q
                best_action = a

        policy[s] = best_action

    return V, policy, history


def analyze_policy(mdp: GamblersProblem, policy: np.ndarray):
    """Analyze properties of the optimal policy."""
    print("\n" + "=" * 60)
    print("POLICY ANALYSIS")
    print("=" * 60)

    # Find interesting patterns
    print("\nStake patterns at different capitals:")
    print("-" * 40)

    checkpoints = [25, 50, 75, 12, 37, 62, 87]
    for s in sorted(checkpoints):
        if 0 < s < mdp.goal:
            max_stake = min(s, mdp.goal - s)
            stake_pct = policy[s] / max_stake * 100 if max_stake > 0 else 0
            print(f"  Capital ${s:3d}: Stake ${policy[s]:2d} ({stake_pct:5.1f}% of max)")

    # Count different stake strategies
    print("\nStake Strategy Distribution:")
    print("-" * 40)

    conservative = 0  # Stake < 25% of max
    moderate = 0  # Stake 25-75% of max
    aggressive = 0  # Stake > 75% of max

    for s in range(1, mdp.goal):
        max_stake = min(s, mdp.goal - s)
        if max_stake > 0:
            ratio = policy[s] / max_stake
            if ratio < 0.25:
                conservative += 1
            elif ratio < 0.75:
                moderate += 1
            else:
                aggressive += 1

    total = conservative + moderate + aggressive
    print(f"  Conservative (< 25%): {conservative:3d} ({conservative / total * 100:.1f}%)")
    print(f"  Moderate (25-75%):    {moderate:3d} ({moderate / total * 100:.1f}%)")
    print(f"  Aggressive (> 75%):   {aggressive:3d} ({aggressive / total * 100:.1f}%)")


def simulate_episodes(mdp: GamblersProblem, policy: np.ndarray,
                      start_capital: int = 50, n_episodes: int = 10000) -> dict:
    """Simulate episodes to verify policy performance."""
    wins = 0
    total_bets = []
    final_capitals = []

    for _ in range(n_episodes):
        capital = start_capital
        bets = 0

        while capital not in mdp.terminal_states:
            stake = policy[capital]
            bets += 1

            # Flip coin
            if np.random.random() < mdp.p_heads:
                capital += stake
            else:
                capital -= stake

        if capital == mdp.goal:
            wins += 1

        total_bets.append(bets)
        final_capitals.append(capital)

    return {
        'win_rate': wins / n_episodes,
        'avg_bets': np.mean(total_bets),
        'std_bets': np.std(total_bets),
    }


def plot_results(mdp: GamblersProblem, V: np.ndarray, policy: np.ndarray,
                 history: List[np.ndarray]):
    """Generate plots for the Gambler's Problem."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Value Function
    ax1 = axes[0, 0]
    ax1.plot(range(mdp.n_states), V, 'b-', linewidth=2)
    ax1.set_xlabel('Capital ($)')
    ax1.set_ylabel('Value (Probability of Winning)')
    ax1.set_title(f"Value Function (p_heads = {mdp.p_heads})")
    ax1.grid(True)
    ax1.set_xlim(0, mdp.goal)
    ax1.set_ylim(0, 1)

    # Mark key points
    for s in [25, 50, 75]:
        ax1.axvline(x=s, color='gray', linestyle='--', alpha=0.5)
        ax1.annotate(f'V({s})={V[s]:.3f}', xy=(s, V[s]),
                     xytext=(s + 2, V[s] + 0.05))

    # 2. Optimal Policy (Stakes)
    ax2 = axes[0, 1]
    stakes = policy[1:mdp.goal]  # Exclude terminal states
    ax2.bar(range(1, mdp.goal), stakes, color='green', alpha=0.7)
    ax2.set_xlabel('Capital ($)')
    ax2.set_ylabel('Optimal Stake ($)')
    ax2.set_title('Optimal Policy (Stake Amount)')
    ax2.set_xlim(0, mdp.goal)
    ax2.grid(True, axis='y')

    # 3. Value Function Evolution
    ax3 = axes[1, 0]
    iterations_to_show = [0, 1, 5, 10, 20, len(history) - 1]
    iterations_to_show = [i for i in iterations_to_show if i < len(history)]

    for i in iterations_to_show:
        alpha = 0.3 + 0.7 * (i / max(iterations_to_show))
        ax3.plot(range(mdp.n_states), history[i], alpha=alpha,
                 label=f'Iter {i}')

    ax3.set_xlabel('Capital ($)')
    ax3.set_ylabel('Value')
    ax3.set_title('Value Function Convergence')
    ax3.legend()
    ax3.grid(True)
    ax3.set_xlim(0, mdp.goal)

    # 4. Policy as percentage of max stake
    ax4 = axes[1, 1]
    stake_pct = []
    for s in range(1, mdp.goal):
        max_stake = min(s, mdp.goal - s)
        pct = policy[s] / max_stake * 100 if max_stake > 0 else 0
        stake_pct.append(pct)

    ax4.plot(range(1, mdp.goal), stake_pct, 'r-', linewidth=1)
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax4.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100%')
    ax4.set_xlabel('Capital ($)')
    ax4.set_ylabel('Stake as % of Maximum')
    ax4.set_title('Stake Aggressiveness')
    ax4.set_xlim(0, mdp.goal)
    ax4.set_ylim(0, 110)
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('/tmp/gamblers_problem.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to /tmp/gamblers_problem.png")

    return fig


def main():
    print("=" * 60)
    print("GAMBLER'S PROBLEM")
    print("=" * 60)

    print("""
    A gambler has the opportunity to make bets on a coin flip.

    Rules:
    - Start with some capital (1-99 dollars)
    - Each round, bet any amount from $1 to min(capital, 100-capital)
    - Heads (p = p_h): win the stake (capital += stake)
    - Tails (p = 1-p_h): lose the stake (capital -= stake)
    - Win if capital reaches $100
    - Lose if capital reaches $0

    Goal: Maximize probability of reaching $100
    """)

    # ============================================
    # CASE 1: UNFAVORABLE COIN (p_h = 0.4)
    # ============================================
    print("\n" + "=" * 60)
    print("CASE 1: UNFAVORABLE COIN (p_heads = 0.4)")
    print("=" * 60)

    mdp_unfair = GamblersProblem(goal=100, p_heads=0.4)
    print(f"\nCoin probability: P(heads) = {mdp_unfair.p_heads}")
    print("This is an unfavorable game (expected value < 0)")

    V_unfair, policy_unfair, history_unfair = value_iteration(mdp_unfair)

    print(f"\nValue function at key states:")
    for s in [25, 50, 75]:
        print(f"  V({s}) = {V_unfair[s]:.6f} (prob of winning from ${s})")

    print(f"\nOptimal stakes at key states:")
    for s in [25, 50, 75]:
        print(f"  State ${s}: Stake ${policy_unfair[s]}")

    analyze_policy(mdp_unfair, policy_unfair)

    # Simulation
    print("\nSimulating 10,000 episodes starting with $50:")
    results = simulate_episodes(mdp_unfair, policy_unfair, start_capital=50)
    print(f"  Win rate: {results['win_rate'] * 100:.2f}%")
    print(f"  Avg bets: {results['avg_bets']:.1f} +/- {results['std_bets']:.1f}")
    print(f"  (Theory: V(50) = {V_unfair[50] * 100:.2f}%)")

    # Plot
    fig1 = plot_results(mdp_unfair, V_unfair, policy_unfair, history_unfair)

    # ============================================
    # CASE 2: FAIR COIN (p_h = 0.5)
    # ============================================
    print("\n" + "=" * 60)
    print("CASE 2: FAIR COIN (p_heads = 0.5)")
    print("=" * 60)

    mdp_fair = GamblersProblem(goal=100, p_heads=0.5)
    V_fair, policy_fair, _ = value_iteration(mdp_fair)

    print(f"\nValue function at key states:")
    for s in [25, 50, 75]:
        print(f"  V({s}) = {V_fair[s]:.6f}")

    print("\nNote: With fair coin, V(s) = s/100 (linear!)")
    print("Any policy achieves the same winning probability!")

    # ============================================
    # CASE 3: FAVORABLE COIN (p_h = 0.55)
    # ============================================
    print("\n" + "=" * 60)
    print("CASE 3: FAVORABLE COIN (p_heads = 0.55)")
    print("=" * 60)

    mdp_favor = GamblersProblem(goal=100, p_heads=0.55)
    V_favor, policy_favor, _ = value_iteration(mdp_favor)

    print(f"\nValue function at key states:")
    for s in [25, 50, 75]:
        print(f"  V({s}) = {V_favor[s]:.6f}")

    print(f"\nOptimal stakes at key states:")
    for s in [25, 50, 75]:
        print(f"  State ${s}: Stake ${policy_favor[s]}")

    print("\nWith favorable coin, optimal policy is more conservative!")
    print("(Bet small to minimize variance and let the edge work)")

    # ============================================
    # COMPARISON
    # ============================================
    print("\n" + "=" * 60)
    print("COMPARISON ACROSS COIN PROBABILITIES")
    print("=" * 60)

    print("\nWinning probability from $50:")
    print(f"  p_h = 0.40: V(50) = {V_unfair[50] * 100:.2f}%")
    print(f"  p_h = 0.50: V(50) = {V_fair[50] * 100:.2f}%")
    print(f"  p_h = 0.55: V(50) = {V_favor[50] * 100:.2f}%")

    print("\nOptimal stake from $50:")
    print(f"  p_h = 0.40: ${policy_unfair[50]} (aggressive - need luck)")
    print(f"  p_h = 0.50: ${policy_fair[50]} (any stake is equivalent)")
    print(f"  p_h = 0.55: ${policy_favor[50]} (conservative - let edge work)")

    # ============================================
    # KEY INSIGHTS
    # ============================================
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)

    print("""
    1. UNFAVORABLE GAME (p_h < 0.5)
       - Optimal policy is often aggressive
       - Bet big to "get lucky" before the edge destroys you
       - Spiky policy with bets at 25, 50, 75 (powers of goal/4)

    2. FAIR GAME (p_h = 0.5)
       - Any betting strategy has the same winning probability
       - V(s) = s/100 is exactly linear
       - Policy is arbitrary (all equivalent)

    3. FAVORABLE GAME (p_h > 0.5)
       - Optimal policy is conservative
       - Bet small to minimize variance
       - Let the positive expected value accumulate

    4. THE SPIKE PATTERN
       - In unfavorable games, policy shows spikes at s = 25, 50, 75
       - These are "critical" states where one bet can reach goal
       - Reflects the need to "take your shot" when possible

    5. VALUE FUNCTION SHAPE
       - Unfavorable: Concave (worse than linear)
       - Fair: Linear V(s) = s/100
       - Favorable: Convex (better than linear)
    """)

    plt.close('all')


if __name__ == "__main__":
    main()
