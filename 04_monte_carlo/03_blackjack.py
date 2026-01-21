"""
03 - Blackjack with Monte Carlo

Classic example from Sutton & Barto Chapter 5.

Demonstrates:
- MC methods on a real game
- State space design
- Policy visualization
- Optimal strategy discovery
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict


def create_blackjack_env():
    """Create Blackjack environment."""
    return gym.make('Blackjack-v1', natural=False, sab=False)


def state_to_tuple(obs) -> Tuple[int, int, bool]:
    """Convert observation to hashable tuple."""
    return (obs[0], obs[1], obs[2])


def generate_episode_exploring_starts(env, Q: Dict,
                                       n_actions: int = 2) -> List[Tuple]:
    """Generate episode with exploring starts for Blackjack."""
    # Reset and get initial state
    obs, _ = env.reset()
    state = state_to_tuple(obs)

    episode = []
    done = False

    # Exploring starts: random first action
    action = np.random.randint(n_actions)

    while not done:
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        episode.append((state, action, reward))

        if not done:
            state = state_to_tuple(next_obs)
            # Greedy action selection after first
            q_values = [Q.get((state, a), 0.0) for a in range(n_actions)]
            action = np.argmax(q_values)

    return episode


def generate_episode_epsilon_greedy(env, Q: Dict, epsilon: float = 0.1,
                                     n_actions: int = 2) -> List[Tuple]:
    """Generate episode with epsilon-greedy policy."""
    obs, _ = env.reset()
    state = state_to_tuple(obs)

    episode = []
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            q_values = [Q.get((state, a), 0.0) for a in range(n_actions)]
            action = np.argmax(q_values)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        episode.append((state, action, reward))

        if not done:
            state = state_to_tuple(next_obs)

    return episode


def mc_control_exploring_starts(env, n_episodes: int = 500000,
                                 gamma: float = 1.0) -> Tuple[Dict, Dict]:
    """MC Control with Exploring Starts for Blackjack."""
    Q = defaultdict(float)
    returns = defaultdict(list)
    n_actions = 2  # 0: Stick, 1: Hit

    for episode_num in range(n_episodes):
        episode = generate_episode_exploring_starts(env, Q, n_actions)

        visited = set()
        G = 0

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            if (state, action) not in visited:
                visited.add((state, action))
                returns[(state, action)].append(G)
                Q[(state, action)] = np.mean(returns[(state, action)])

        if (episode_num + 1) % 100000 == 0:
            print(f"  Episode {episode_num + 1}/{n_episodes}")

    # Extract policy
    policy = {}
    for state in set(s for (s, a) in Q.keys()):
        q_values = [Q.get((state, a), 0.0) for a in range(n_actions)]
        policy[state] = np.argmax(q_values)

    return dict(Q), policy


def mc_control_epsilon_greedy(env, n_episodes: int = 500000,
                               gamma: float = 1.0, epsilon: float = 0.1) -> Tuple[Dict, Dict]:
    """On-policy MC Control with epsilon-greedy."""
    Q = defaultdict(float)
    N = defaultdict(int)
    n_actions = 2

    for episode_num in range(n_episodes):
        episode = generate_episode_epsilon_greedy(env, Q, epsilon, n_actions)

        visited = set()
        G = 0

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            if (state, action) not in visited:
                visited.add((state, action))
                N[(state, action)] += 1
                alpha = 1.0 / N[(state, action)]
                Q[(state, action)] += alpha * (G - Q[(state, action)])

        if (episode_num + 1) % 100000 == 0:
            print(f"  Episode {episode_num + 1}/{n_episodes}")

    # Extract greedy policy
    policy = {}
    for state in set(s for (s, a) in Q.keys()):
        q_values = [Q.get((state, a), 0.0) for a in range(n_actions)]
        policy[state] = np.argmax(q_values)

    return dict(Q), policy


def evaluate_policy(env, policy: Dict, n_episodes: int = 10000) -> Dict:
    """Evaluate a policy by playing many games."""
    wins = 0
    losses = 0
    draws = 0
    total_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state = state_to_tuple(obs)
            action = policy.get(state, 0)  # Default: stick
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        total_rewards.append(total_reward)
        if total_reward > 0:
            wins += 1
        elif total_reward < 0:
            losses += 1
        else:
            draws += 1

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': wins / n_episodes,
        'mean_reward': np.mean(total_rewards),
    }


def visualize_policy(policy: Dict, usable_ace: bool, title: str = "Policy"):
    """Visualize policy as a grid."""
    print(f"\n{title} (Usable Ace = {usable_ace}):")
    print("\n  Dealer Showing")
    print("     " + "  ".join([f"{i:2d}" for i in range(1, 11)]))
    print("    +" + "---+" * 10)

    action_symbols = {0: "S", 1: "H"}  # Stick, Hit

    for player_sum in range(21, 11, -1):
        line = f" {player_sum} |"
        for dealer_card in range(1, 11):
            state = (player_sum, dealer_card, usable_ace)
            action = policy.get(state, 0)
            line += f" {action_symbols[action]} |"
        print(line)
        print("    +" + "---+" * 10)

    print("\n  S = Stick, H = Hit")
    print("  (Player sum 12-21 shown, always hit below 12)")


def visualize_value_function(Q: Dict, usable_ace: bool, title: str = "V(s)"):
    """Visualize state values derived from Q."""
    print(f"\n{title} (Usable Ace = {usable_ace}):")
    print("\n  Dealer Showing")
    print("       " + "    ".join([f"{i:2d}" for i in range(1, 11)]))

    for player_sum in range(21, 11, -1):
        line = f" {player_sum} |"
        for dealer_card in range(1, 11):
            state = (player_sum, dealer_card, usable_ace)
            # V(s) = max_a Q(s,a)
            v = max(Q.get((state, 0), 0.0), Q.get((state, 1), 0.0))
            line += f" {v:5.2f}"
        print(line)


def create_fixed_policy(stick_threshold: int = 20) -> Dict:
    """Create a simple fixed policy: hit below threshold, stick at/above."""
    policy = {}
    for player_sum in range(4, 22):
        for dealer_card in range(1, 11):
            for usable_ace in [True, False]:
                state = (player_sum, dealer_card, usable_ace)
                if player_sum >= stick_threshold:
                    policy[state] = 0  # Stick
                else:
                    policy[state] = 1  # Hit
    return policy


def main():
    # ============================================
    # 1. INTRODUCTION
    # ============================================
    print("=" * 60)
    print("BLACKJACK WITH MONTE CARLO")
    print("=" * 60)

    print("""
    Blackjack (21) is a classic MC example.

    Rules:
    - Goal: Get cards summing close to 21, without going over
    - Face cards = 10, Ace = 1 or 11
    - Player actions: Hit (get card) or Stick (stop)
    - Dealer policy: Hit on <17, Stick on >=17
    - Win: Beat dealer or dealer busts
    - Lose: Go over 21 or dealer beats you
    - Draw: Same sum as dealer

    State: (player_sum, dealer_showing, usable_ace)
    - Player sum: 12-21 (always hit below 12)
    - Dealer showing: 1-10 (their visible card)
    - Usable ace: True if have ace counting as 11

    Why MC is perfect:
    - Episodes are short
    - No model of card distribution needed
    - Natural termination (win/lose/draw)
    """)

    env = create_blackjack_env()

    # ============================================
    # 2. EVALUATE SIMPLE POLICIES
    # ============================================
    print("\n" + "=" * 60)
    print("2. EVALUATE SIMPLE POLICIES")
    print("=" * 60)

    print("\nEvaluating fixed 'stick on X' policies:")
    print("\n  Stick on | Win Rate | Mean Reward")
    print("  " + "-" * 35)

    for threshold in [17, 18, 19, 20]:
        policy = create_fixed_policy(stick_threshold=threshold)
        results = evaluate_policy(env, policy, n_episodes=50000)
        print(f"     {threshold}    |  {results['win_rate'] * 100:5.2f}%  |   {results['mean_reward']:.4f}")

    # ============================================
    # 3. MC CONTROL WITH EXPLORING STARTS
    # ============================================
    print("\n" + "=" * 60)
    print("3. MC CONTROL WITH EXPLORING STARTS")
    print("=" * 60)

    print("\nRunning MC ES (500,000 episodes)...")
    Q_es, policy_es = mc_control_exploring_starts(env, n_episodes=500000)

    # Visualize optimal policy
    visualize_policy(policy_es, usable_ace=False, title="Optimal Policy (No Usable Ace)")
    visualize_policy(policy_es, usable_ace=True, title="Optimal Policy (Usable Ace)")

    # Evaluate optimal policy
    results_es = evaluate_policy(env, policy_es, n_episodes=100000)
    print(f"\nOptimal Policy Performance (MC ES):")
    print(f"  Win rate: {results_es['win_rate'] * 100:.2f}%")
    print(f"  Loss rate: {results_es['losses'] / 100000 * 100:.2f}%")
    print(f"  Draw rate: {results_es['draws'] / 100000 * 100:.2f}%")
    print(f"  Mean reward: {results_es['mean_reward']:.4f}")

    # ============================================
    # 4. MC CONTROL WITH EPSILON-GREEDY
    # ============================================
    print("\n" + "=" * 60)
    print("4. MC CONTROL WITH EPSILON-GREEDY")
    print("=" * 60)

    print("\nRunning Epsilon-Greedy MC (epsilon=0.1, 500,000 episodes)...")
    Q_eg, policy_eg = mc_control_epsilon_greedy(env, n_episodes=500000, epsilon=0.1)

    # Visualize policy
    visualize_policy(policy_eg, usable_ace=False, title="Policy (Epsilon-Greedy)")

    # Evaluate
    results_eg = evaluate_policy(env, policy_eg, n_episodes=100000)
    print(f"\nEpsilon-Greedy Policy Performance:")
    print(f"  Win rate: {results_eg['win_rate'] * 100:.2f}%")
    print(f"  Mean reward: {results_eg['mean_reward']:.4f}")

    # ============================================
    # 5. POLICY ANALYSIS
    # ============================================
    print("\n" + "=" * 60)
    print("5. POLICY ANALYSIS")
    print("=" * 60)

    print("""
    Key observations from optimal policy:

    1. WITHOUT USABLE ACE:
       - Stick on 17+ against dealer 2-6
       - Stick on 13+ against dealer 2-6 sometimes
       - Hit more aggressively against dealer 7-A

    2. WITH USABLE ACE:
       - More aggressive (can't bust easily)
       - Hit on soft 17 (A-6) against most dealer cards
       - The ace provides a safety net

    3. DEALER CARD MATTERS:
       - Dealer 2-6: Likely to bust, be more conservative
       - Dealer 7-A: Strong position, need to hit more
    """)

    # Compare specific states
    print("\nQ-values for some key states (MC ES):")

    states_to_show = [
        (20, 10, False),  # Strong hand vs dealer 10
        (16, 10, False),  # Tricky decision
        (13, 2, False),   # Against weak dealer
        (17, 1, True),    # Soft 17 vs Ace
    ]

    for state in states_to_show:
        q_stick = Q_es.get((state, 0), 0.0)
        q_hit = Q_es.get((state, 1), 0.0)
        best = "Stick" if q_stick > q_hit else "Hit"
        print(f"\n  State {state}:")
        print(f"    Q(Stick) = {q_stick:.4f}")
        print(f"    Q(Hit)   = {q_hit:.4f}")
        print(f"    Best: {best}")

    # ============================================
    # 6. EFFECT OF TRAINING EPISODES
    # ============================================
    print("\n" + "=" * 60)
    print("6. EFFECT OF TRAINING EPISODES")
    print("=" * 60)

    print("\nWin rate vs number of training episodes:")
    print("\n  Episodes   | Win Rate")
    print("  " + "-" * 25)

    for n_ep in [10000, 50000, 100000, 500000]:
        Q_test, policy_test = mc_control_exploring_starts(env, n_episodes=n_ep)
        results = evaluate_policy(env, policy_test, n_episodes=50000)
        print(f"   {n_ep:7d}  |  {results['win_rate'] * 100:.2f}%")

    print("\n  More episodes -> Better policy (up to a point)")

    # ============================================
    # 7. COMPARISON WITH BASIC STRATEGY
    # ============================================
    print("\n" + "=" * 60)
    print("7. COMPARISON: LEARNED VS BASIC STRATEGY")
    print("=" * 60)

    # Simple basic strategy
    basic_policy = create_fixed_policy(stick_threshold=17)

    results_basic = evaluate_policy(env, basic_policy, n_episodes=100000)
    results_learned = evaluate_policy(env, policy_es, n_episodes=100000)

    print("\n  Policy       | Win Rate | Mean Reward")
    print("  " + "-" * 40)
    print(f"  Stick on 17  |  {results_basic['win_rate'] * 100:5.2f}%  |   {results_basic['mean_reward']:.4f}")
    print(f"  MC Optimal   |  {results_learned['win_rate'] * 100:5.2f}%  |   {results_learned['mean_reward']:.4f}")

    improvement = (results_learned['mean_reward'] - results_basic['mean_reward']) / abs(results_basic['mean_reward']) * 100
    print(f"\n  Improvement: {improvement:.1f}% better mean reward")

    # ============================================
    # 8. SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("8. SUMMARY")
    print("=" * 60)

    print("""
    Blackjack MC Control Key Points:

    1. STATE REPRESENTATION
       - (player_sum, dealer_showing, usable_ace)
       - 200 states (10 sums x 10 dealer x 2 ace)
       - Abstracts away specific cards

    2. MC IS IDEAL FOR BLACKJACK
       - Short episodes
       - No model needed
       - Can play millions of games quickly

    3. OPTIMAL STRATEGY
       - Context-dependent (dealer card matters)
       - Usable ace changes strategy
       - Not simply "stick on 17"

    4. PERFORMANCE
       - ~42-43% win rate (optimal)
       - House edge is small but exists
       - MC finds near-optimal strategy

    5. LEARNING REQUIREMENTS
       - ~500,000 episodes for good convergence
       - Exploring starts helps coverage
       - Epsilon-greedy also works well

    Note: Real casinos use multiple decks, shuffling,
    and other rules that change optimal strategy slightly.
    """)


if __name__ == "__main__":
    main()
