"""
02 - Monte Carlo Control

Learn to find optimal policies using Monte Carlo methods.

Demonstrates:
- Monte Carlo with Exploring Starts (MC ES)
- On-Policy MC Control with epsilon-greedy
- GLIE (Greedy in the Limit with Infinite Exploration)
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class GridWorldMDP:
    """Grid World for MC Control demonstration."""

    def __init__(self, size: int = 5, gamma: float = 0.9):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.gamma = gamma

        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.action_names = {0: "^", 1: ">", 2: "v", 3: "<"}

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

    def generate_episode(self, Q: Dict, epsilon: float = 0.1,
                         start_state: int = None, start_action: int = None,
                         max_steps: int = 100) -> List[Tuple[int, int, float]]:
        """Generate episode using epsilon-greedy policy derived from Q."""
        if start_state is None:
            start_state = self.start_state

        episode = []
        state = start_state
        done = False
        steps = 0

        # First action (for exploring starts)
        if start_action is not None:
            action = start_action
        else:
            action = self._epsilon_greedy_action(state, Q, epsilon)

        while not done and steps < max_steps:
            next_state, reward, done = self.step(state, action)
            episode.append((state, action, reward))
            state = next_state

            if not done:
                action = self._epsilon_greedy_action(state, Q, epsilon)
            steps += 1

        return episode

    def _epsilon_greedy_action(self, state: int, Q: Dict, epsilon: float) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = [Q.get((state, a), 0.0) for a in self.actions]
            return np.argmax(q_values)

    def get_greedy_policy(self, Q: Dict) -> np.ndarray:
        """Extract greedy policy from Q."""
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            q_values = [Q.get((s, a), 0.0) for a in self.actions]
            policy[s] = np.argmax(q_values)
        return policy

    def render_policy(self, policy: np.ndarray, title: str = "Policy"):
        """Render policy as grid."""
        print(f"\n{title}:")
        print("+" + "----+" * self.size)

        for row in range(self.size):
            line = "|"
            for col in range(self.size):
                state = self._coord_to_state(row, col)
                if state == self.goal_state:
                    cell = " G "
                else:
                    cell = f" {self.action_names[policy[state]]} "
                line += cell + "|"
            print(line)
            print("+" + "----+" * self.size)

    def render_q_values(self, Q: Dict, state: int):
        """Show Q-values for a specific state."""
        print(f"\nQ-values for state {state}:")
        for a in self.actions:
            print(f"  {self.action_names[a]}: {Q.get((state, a), 0.0):.4f}")


def mc_exploring_starts(env: GridWorldMDP, n_episodes: int = 5000,
                         gamma: float = 0.9) -> Tuple[Dict, List]:
    """
    Monte Carlo with Exploring Starts (MC ES).

    Ensures all state-action pairs are visited by starting
    episodes from random (state, action) pairs.
    """
    Q = defaultdict(float)
    returns = defaultdict(list)
    history = []

    for episode_num in range(n_episodes):
        # Exploring starts: random initial state and action
        start_state = np.random.randint(env.n_states)
        if start_state == env.goal_state:
            start_state = env.start_state
        start_action = np.random.randint(env.n_actions)

        episode = env.generate_episode(
            Q, epsilon=0.0,  # Greedy after first action
            start_state=start_state,
            start_action=start_action
        )

        # Track visited state-action pairs
        visited = set()
        G = 0

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            if (state, action) not in visited:
                visited.add((state, action))
                returns[(state, action)].append(G)
                Q[(state, action)] = np.mean(returns[(state, action)])

        if episode_num % 500 == 0:
            policy = env.get_greedy_policy(Q)
            history.append((episode_num, policy.copy(), dict(Q)))

    return dict(Q), history


def mc_epsilon_greedy(env: GridWorldMDP, n_episodes: int = 5000,
                       gamma: float = 0.9, epsilon: float = 0.1) -> Tuple[Dict, List]:
    """
    On-Policy MC Control with constant epsilon-greedy.

    Always explores with probability epsilon.
    """
    Q = defaultdict(float)
    returns = defaultdict(list)
    history = []

    for episode_num in range(n_episodes):
        episode = env.generate_episode(Q, epsilon=epsilon)

        visited = set()
        G = 0

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            if (state, action) not in visited:
                visited.add((state, action))
                returns[(state, action)].append(G)
                Q[(state, action)] = np.mean(returns[(state, action)])

        if episode_num % 500 == 0:
            policy = env.get_greedy_policy(Q)
            history.append((episode_num, policy.copy(), dict(Q)))

    return dict(Q), history


def mc_glie(env: GridWorldMDP, n_episodes: int = 10000,
            gamma: float = 0.9, epsilon_start: float = 1.0) -> Tuple[Dict, List]:
    """
    GLIE MC Control (Greedy in the Limit with Infinite Exploration).

    Epsilon decays over time: epsilon_k = 1/k
    This ensures:
    1. All state-action pairs visited infinitely often
    2. Policy converges to greedy (epsilon -> 0)
    """
    Q = defaultdict(float)
    N = defaultdict(int)  # Visit counts
    history = []

    for episode_num in range(1, n_episodes + 1):
        # Decaying epsilon
        epsilon = epsilon_start / episode_num

        episode = env.generate_episode(Q, epsilon=epsilon)

        visited = set()
        G = 0

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            if (state, action) not in visited:
                visited.add((state, action))
                N[(state, action)] += 1
                # Incremental update
                alpha = 1.0 / N[(state, action)]
                Q[(state, action)] += alpha * (G - Q[(state, action)])

        if episode_num % 500 == 0:
            policy = env.get_greedy_policy(Q)
            history.append((episode_num, policy.copy(), dict(Q), epsilon))

    return dict(Q), history


def evaluate_policy(env: GridWorldMDP, policy: np.ndarray,
                    n_episodes: int = 1000) -> Tuple[float, float]:
    """Evaluate a deterministic policy."""
    total_rewards = []

    for _ in range(n_episodes):
        state = env.start_state
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 100:
            action = policy[state]
            state, reward, done = env.step(state, action)
            total_reward += reward
            steps += 1

        total_rewards.append(total_reward)

    return np.mean(total_rewards), np.std(total_rewards)


def main():
    # ============================================
    # 1. INTRODUCTION
    # ============================================
    print("=" * 60)
    print("MONTE CARLO CONTROL")
    print("=" * 60)

    print("""
    MC Control finds optimal policies without a model.

    Challenge: Need to estimate Q(s,a), not just V(s)
    Why? Without a model, we can't do:
        pi(s) = argmax_a sum P(s'|s,a) * [R + gamma*V(s')]
        (We don't know P!)

    With Q-values:
        pi(s) = argmax_a Q(s,a)
        (No model needed!)

    But we need to explore all (state, action) pairs...
    """)

    env = GridWorldMDP(size=5, gamma=0.9)
    print(f"\nGrid World: {env.size}x{env.size}")

    # ============================================
    # 2. MC WITH EXPLORING STARTS
    # ============================================
    print("\n" + "=" * 60)
    print("2. MC WITH EXPLORING STARTS")
    print("=" * 60)

    print("""
    Idea: Start each episode from random (state, action)
    This ensures all pairs are visited infinitely often.
    """)

    print("\nRunning MC ES (5000 episodes)...")
    Q_es, history_es = mc_exploring_starts(env, n_episodes=5000)

    policy_es = env.get_greedy_policy(Q_es)
    env.render_policy(policy_es, "Optimal Policy (MC ES)")

    mean_reward, std_reward = evaluate_policy(env, policy_es)
    print(f"\nPolicy Performance:")
    print(f"  Mean reward: {mean_reward:.4f} +/- {std_reward:.4f}")

    # ============================================
    # 3. EPSILON-GREEDY MC CONTROL
    # ============================================
    print("\n" + "=" * 60)
    print("3. EPSILON-GREEDY MC CONTROL")
    print("=" * 60)

    print("""
    Alternative to exploring starts: epsilon-greedy policy

    pi(a|s) = 1 - epsilon + epsilon/|A|  if a = argmax Q
            = epsilon/|A|                 otherwise

    Always has some probability of exploration.
    """)

    print("\nRunning Epsilon-Greedy MC (epsilon=0.1, 5000 episodes)...")
    Q_eg, history_eg = mc_epsilon_greedy(env, n_episodes=5000, epsilon=0.1)

    policy_eg = env.get_greedy_policy(Q_eg)
    env.render_policy(policy_eg, "Policy (Epsilon-Greedy MC)")

    mean_reward, std_reward = evaluate_policy(env, policy_eg)
    print(f"\nPolicy Performance:")
    print(f"  Mean reward: {mean_reward:.4f} +/- {std_reward:.4f}")

    # ============================================
    # 4. GLIE MC CONTROL
    # ============================================
    print("\n" + "=" * 60)
    print("4. GLIE MC CONTROL")
    print("=" * 60)

    print("""
    GLIE = Greedy in the Limit with Infinite Exploration

    Requirements:
    1. All state-action pairs visited infinitely often
    2. Policy converges to greedy

    Solution: epsilon_k = 1/k (decaying exploration)

    GLIE MC Control converges to optimal Q*!
    """)

    print("\nRunning GLIE MC (10000 episodes)...")
    Q_glie, history_glie = mc_glie(env, n_episodes=10000)

    policy_glie = env.get_greedy_policy(Q_glie)
    env.render_policy(policy_glie, "Optimal Policy (GLIE MC)")

    mean_reward, std_reward = evaluate_policy(env, policy_glie)
    print(f"\nPolicy Performance:")
    print(f"  Mean reward: {mean_reward:.4f} +/- {std_reward:.4f}")

    # Show epsilon decay
    print("\nEpsilon decay in GLIE:")
    for ep, _, _, eps in history_glie[:5]:
        print(f"  Episode {ep}: epsilon = {eps:.4f}")
    print(f"  ...")
    print(f"  Episode {history_glie[-1][0]}: epsilon = {history_glie[-1][3]:.6f}")

    # ============================================
    # 5. Q-VALUE ANALYSIS
    # ============================================
    print("\n" + "=" * 60)
    print("5. Q-VALUE ANALYSIS")
    print("=" * 60)

    # Show Q-values for start state
    start = env.start_state
    print(f"\nQ-values at start state (state {start}):")

    for method_name, Q in [("MC ES", Q_es), ("Eps-Greedy", Q_eg), ("GLIE", Q_glie)]:
        print(f"\n  {method_name}:")
        for a in env.actions:
            q = Q.get((start, a), 0.0)
            print(f"    {env.action_names[a]}: {q:.4f}")

    # ============================================
    # 6. LEARNING CURVES
    # ============================================
    print("\n" + "=" * 60)
    print("6. LEARNING CURVES")
    print("=" * 60)

    print("\nPolicy performance during learning (GLIE):")
    print("\n  Episode | Mean Reward | Epsilon")
    print("  " + "-" * 40)

    for ep, policy, _, eps in history_glie:
        mean_r, _ = evaluate_policy(env, policy, n_episodes=200)
        print(f"    {ep:5d} |   {mean_r:7.4f}   | {eps:.4f}")

    # ============================================
    # 7. EFFECT OF EPSILON
    # ============================================
    print("\n" + "=" * 60)
    print("7. EFFECT OF EPSILON (CONSTANT)")
    print("=" * 60)

    print("\nComparing different epsilon values:")
    print("\n  Epsilon | Mean Reward")
    print("  " + "-" * 25)

    for eps in [0.01, 0.05, 0.1, 0.2, 0.5]:
        Q_test, _ = mc_epsilon_greedy(env, n_episodes=3000, epsilon=eps)
        policy_test = env.get_greedy_policy(Q_test)
        mean_r, _ = evaluate_policy(env, policy_test)
        print(f"    {eps:.2f}   |   {mean_r:.4f}")

    print("""
    Observations:
    - Too low epsilon: Not enough exploration, may miss optimal
    - Too high epsilon: Too much random behavior, slower learning
    - Middle values work best for constant epsilon
    - GLIE (decaying) is theoretically optimal
    """)

    # ============================================
    # 8. COMPARISON OF METHODS
    # ============================================
    print("\n" + "=" * 60)
    print("8. COMPARISON OF METHODS")
    print("=" * 60)

    mean_es, _ = evaluate_policy(env, policy_es)
    mean_eg, _ = evaluate_policy(env, policy_eg)
    mean_glie, _ = evaluate_policy(env, policy_glie)

    print("\n  Method          | Mean Reward | Converges to Optimal?")
    print("  " + "-" * 55)
    print(f"  MC ES           |   {mean_es:.4f}    | Yes (with exploring starts)")
    print(f"  Eps-Greedy      |   {mean_eg:.4f}    | No (constant exploration)")
    print(f"  GLIE            |   {mean_glie:.4f}    | Yes (epsilon -> 0)")

    # ============================================
    # 9. SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("9. SUMMARY")
    print("=" * 60)

    print("""
    Monte Carlo Control Key Points:

    1. ESTIMATE Q(s,a) NOT V(s)
       - Allows model-free policy improvement
       - pi(s) = argmax_a Q(s,a)

    2. EXPLORATION IS CRUCIAL
       - Must visit all (state, action) pairs
       - Exploring starts: Random initial (s,a)
       - Epsilon-greedy: Random actions with prob epsilon

    3. METHODS
       - MC ES: Exploring starts, greedy otherwise
       - Epsilon-greedy: Constant exploration
       - GLIE: Decaying epsilon, converges to optimal

    4. CONVERGENCE
       - MC ES: Converges to Q* (with exploring starts)
       - GLIE: Converges to Q* (epsilon -> 0)
       - Constant eps: Converges to near-optimal

    5. PRACTICAL CONSIDERATIONS
       - Exploring starts often impractical
       - Epsilon-greedy is most common
       - Decaying epsilon good for finite problems

    Next: Blackjack example (classic MC application)
    """)


if __name__ == "__main__":
    main()
