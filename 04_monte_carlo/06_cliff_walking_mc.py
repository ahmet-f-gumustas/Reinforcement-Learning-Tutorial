"""
06 - CliffWalking with Monte Carlo

Solve Gymnasium's CliffWalking environment using MC methods.

Demonstrates:
- MC on a standard Gymnasium environment
- Risk-sensitive learning (cliff = -100)
- Comparison of different exploration strategies
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict


def create_env():
    """Create CliffWalking environment."""
    return gym.make('CliffWalking-v0')


def state_to_pos(state: int, width: int = 12) -> Tuple[int, int]:
    """Convert state to (row, col)."""
    return state // width, state % width


def generate_episode(env, Q: Dict, epsilon: float = 0.1,
                     max_steps: int = 500) -> List[Tuple]:
    """Generate episode with epsilon-greedy policy."""
    episode = []
    state, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        # Epsilon-greedy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = [Q.get((state, a), 0.0) for a in range(4)]
            action = np.argmax(q_values)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode.append((state, action, reward))
        state = next_state
        steps += 1

    return episode


def mc_control_epsilon_greedy(env, n_episodes: int = 10000,
                               gamma: float = 1.0,
                               epsilon: float = 0.1) -> Tuple[Dict, List]:
    """On-policy MC Control with epsilon-greedy."""
    Q = defaultdict(float)
    N = defaultdict(int)
    history = []

    episode_rewards = []

    for ep in range(n_episodes):
        episode = generate_episode(env, Q, epsilon)

        # Calculate total episode reward
        total_reward = sum(r for _, _, r in episode)
        episode_rewards.append(total_reward)

        # First-visit MC
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

        if (ep + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            history.append((ep + 1, avg_reward))

    return dict(Q), history


def mc_control_decaying_epsilon(env, n_episodes: int = 10000,
                                 gamma: float = 1.0,
                                 epsilon_start: float = 1.0,
                                 epsilon_end: float = 0.01) -> Tuple[Dict, List]:
    """MC Control with decaying epsilon."""
    Q = defaultdict(float)
    N = defaultdict(int)
    history = []

    episode_rewards = []

    for ep in range(n_episodes):
        # Linear decay
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (ep / n_episodes)

        episode = generate_episode(env, Q, epsilon)
        total_reward = sum(r for _, _, r in episode)
        episode_rewards.append(total_reward)

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

        if (ep + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            history.append((ep + 1, avg_reward, epsilon))

    return dict(Q), history


def extract_policy(Q: Dict, n_states: int = 48, n_actions: int = 4) -> np.ndarray:
    """Extract greedy policy from Q."""
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_values = [Q.get((s, a), 0.0) for a in range(n_actions)]
        policy[s] = np.argmax(q_values)
    return policy


def evaluate_policy(env, policy: np.ndarray, n_episodes: int = 100) -> Dict:
    """Evaluate deterministic policy."""
    total_rewards = []
    cliff_falls = 0

    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 200:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if reward == -100:
                cliff_falls += 1

            steps += 1

        total_rewards.append(total_reward)

    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'cliff_falls': cliff_falls,
        'min_reward': np.min(total_rewards),
        'max_reward': np.max(total_rewards),
    }


def render_policy(policy: np.ndarray, width: int = 12, height: int = 4):
    """Render policy as grid."""
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}

    print("\nOptimal Policy:")
    print("+" + "---+" * width)

    for row in range(height):
        line = "|"
        for col in range(width):
            state = row * width + col

            if row == height - 1 and col == 0:
                cell = " S "  # Start
            elif row == height - 1 and col == width - 1:
                cell = " G "  # Goal
            elif row == height - 1 and 0 < col < width - 1:
                cell = " C "  # Cliff
            else:
                cell = f" {arrows[policy[state]]} "

            line += cell + "|"
        print(line)
        print("+" + "---+" * width)

    print("\n  S=Start, G=Goal, C=Cliff")
    print("  ^=Up, >=Right, v=Down, <=Left")


def render_values(Q: Dict, width: int = 12, height: int = 4):
    """Render state values (max Q)."""
    print("\nState Values V(s) = max_a Q(s,a):")
    print("+" + "-------+" * width)

    for row in range(height):
        line = "|"
        for col in range(width):
            state = row * width + col
            v = max(Q.get((state, a), 0.0) for a in range(4))

            if row == height - 1 and col == 0:
                cell = " START "
            elif row == height - 1 and col == width - 1:
                cell = " GOAL  "
            elif row == height - 1 and 0 < col < width - 1:
                cell = " CLIFF "
            else:
                cell = f" {v:5.1f} "

            line += cell + "|"
        print(line)
        print("+" + "-------+" * width)


def main():
    # ============================================
    # 1. INTRODUCTION
    # ============================================
    print("=" * 60)
    print("CLIFFWALKING WITH MONTE CARLO")
    print("=" * 60)

    print("""
    CliffWalking Environment:

    Grid (4x12):
    . . . . . . . . . . . .
    . . . . . . . . . . . .
    . . . . . . . . . . . .
    S C C C C C C C C C C G

    S = Start (state 36)
    G = Goal (state 47)
    C = Cliff (states 37-46)

    Rewards:
    - Each step: -1
    - Falling off cliff: -100 (return to start)

    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    """)

    env = create_env()
    print(f"State space: {env.observation_space.n}")
    print(f"Action space: {env.action_space.n}")

    # ============================================
    # 2. MC CONTROL (CONSTANT EPSILON)
    # ============================================
    print("\n" + "=" * 60)
    print("2. MC CONTROL (EPSILON = 0.1)")
    print("=" * 60)

    print("\nTraining (10,000 episodes)...")
    Q_const, history_const = mc_control_epsilon_greedy(
        env, n_episodes=10000, epsilon=0.1)

    policy_const = extract_policy(Q_const)
    render_policy(policy_const)

    results_const = evaluate_policy(env, policy_const)
    print(f"\nPolicy Performance (100 test episodes):")
    print(f"  Mean reward: {results_const['mean_reward']:.1f}")
    print(f"  Cliff falls: {results_const['cliff_falls']}")

    # ============================================
    # 3. MC CONTROL (DECAYING EPSILON)
    # ============================================
    print("\n" + "=" * 60)
    print("3. MC CONTROL (DECAYING EPSILON)")
    print("=" * 60)

    print("\nTraining with decaying epsilon (1.0 -> 0.01)...")
    Q_decay, history_decay = mc_control_decaying_epsilon(
        env, n_episodes=10000, epsilon_start=1.0, epsilon_end=0.01)

    policy_decay = extract_policy(Q_decay)
    render_policy(policy_decay)

    results_decay = evaluate_policy(env, policy_decay)
    print(f"\nPolicy Performance:")
    print(f"  Mean reward: {results_decay['mean_reward']:.1f}")
    print(f"  Cliff falls: {results_decay['cliff_falls']}")

    # ============================================
    # 4. COMPARE EXPLORATION STRATEGIES
    # ============================================
    print("\n" + "=" * 60)
    print("4. COMPARE EXPLORATION STRATEGIES")
    print("=" * 60)

    print("\nTraining with different epsilon values:")
    print("\n  Epsilon | Mean Reward | Cliff Falls")
    print("  " + "-" * 38)

    for eps in [0.01, 0.05, 0.1, 0.2, 0.3]:
        Q_test, _ = mc_control_epsilon_greedy(env, n_episodes=10000, epsilon=eps)
        policy_test = extract_policy(Q_test)
        results = evaluate_policy(env, policy_test)
        print(f"    {eps:.2f}  |   {results['mean_reward']:6.1f}    |     {results['cliff_falls']}")

    print("""
    Observations:
    - Low epsilon: May not explore enough, suboptimal
    - High epsilon: Too much random exploration, falls off cliff
    - Middle ground or decaying epsilon works best
    """)

    # ============================================
    # 5. VALUE FUNCTION ANALYSIS
    # ============================================
    print("\n" + "=" * 60)
    print("5. VALUE FUNCTION ANALYSIS")
    print("=" * 60)

    render_values(Q_decay)

    print("""
    Observations:
    - States near cliff have lower values (risky)
    - States on safe path have higher values
    - Goal adjacent states are highest (about -1)
    """)

    # ============================================
    # 6. SAFE VS OPTIMAL PATH
    # ============================================
    print("\n" + "=" * 60)
    print("6. SAFE VS OPTIMAL PATH")
    print("=" * 60)

    print("""
    Interesting phenomenon in CliffWalking:

    OPTIMAL PATH (shortest):
    - Go right along the cliff edge
    - 13 steps to goal
    - Risk: One wrong step = -100 and restart

    SAFE PATH (longer):
    - Go up first, then right, then down
    - ~15+ steps to goal
    - Risk: Almost zero cliff falls

    During TRAINING with epsilon-greedy:
    - Random actions sometimes fall off cliff
    - Safe path gets higher average return
    - MC may learn "safer" suboptimal policy

    This is the EXPLORATION-EXPLOITATION tradeoff!
    """)

    # Compare greedy evaluation
    print("\nGreedy Policy Performance (no exploration):")
    results_greedy = evaluate_policy(env, policy_decay, n_episodes=1000)
    print(f"  Mean reward: {results_greedy['mean_reward']:.1f}")
    print(f"  Std reward: {results_greedy['std_reward']:.1f}")
    print(f"  Cliff falls: {results_greedy['cliff_falls']}")
    print(f"  Best episode: {results_greedy['max_reward']}")

    # ============================================
    # 7. LEARNING CURVES
    # ============================================
    print("\n" + "=" * 60)
    print("7. LEARNING CURVES")
    print("=" * 60)

    print("\nConstant Epsilon (0.1):")
    print("  Episodes | Avg Reward")
    print("  " + "-" * 25)
    for ep, avg_r in history_const:
        print(f"   {ep:5d}   |   {avg_r:6.1f}")

    print("\nDecaying Epsilon (1.0 -> 0.01):")
    print("  Episodes | Avg Reward | Epsilon")
    print("  " + "-" * 35)
    for ep, avg_r, eps in history_decay:
        print(f"   {ep:5d}   |   {avg_r:6.1f}   |  {eps:.3f}")

    # ============================================
    # 8. Q-VALUE ANALYSIS
    # ============================================
    print("\n" + "=" * 60)
    print("8. Q-VALUE ANALYSIS")
    print("=" * 60)

    # State near cliff (state 25 = row 2, col 1)
    state_near_cliff = 25
    print(f"\nQ-values at state {state_near_cliff} (near cliff, row 2, col 1):")
    actions = ["Up", "Right", "Down", "Left"]
    for a, name in enumerate(actions):
        q = Q_decay.get((state_near_cliff, a), 0.0)
        print(f"  {name:6s}: {q:7.2f}")

    # State far from cliff (state 1 = row 0, col 1)
    state_safe = 1
    print(f"\nQ-values at state {state_safe} (safe, row 0, col 1):")
    for a, name in enumerate(actions):
        q = Q_decay.get((state_safe, a), 0.0)
        print(f"  {name:6s}: {q:7.2f}")

    # ============================================
    # 9. SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("9. SUMMARY")
    print("=" * 60)

    print("""
    CliffWalking MC Key Points:

    1. RISK-SENSITIVE ENVIRONMENT
       - Large negative reward for cliff (-100)
       - Small negative for steps (-1)
       - Trade-off: fast vs safe

    2. EXPLORATION IMPACT
       - During training, exploration causes cliff falls
       - This biases MC towards safer policies
       - Different from what DP would find!

    3. MC BEHAVIOR
       - Learns from complete episodes
       - Episodes with cliff falls have very negative returns
       - Updates all visited states with these bad returns

    4. EPSILON SENSITIVITY
       - Too high: Many cliff falls during training
       - Too low: May not find good paths
       - Decaying epsilon often works best

    5. KEY INSIGHT
       - MC policy often differs from DP optimal
       - MC accounts for exploration risk
       - "Safe" path may have better EXPECTED return
         during training (even if suboptimal when greedy)

    This connects to SARSA vs Q-Learning (next week!):
    - SARSA: On-policy like MC, learns "safe" paths
    - Q-Learning: Off-policy, learns optimal path
    """)

    env.close()


if __name__ == "__main__":
    main()
