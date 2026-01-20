"""
05 - Solving FrozenLake with Dynamic Programming

Apply Policy Iteration and Value Iteration to solve
Gymnasium's FrozenLake environment.

Demonstrates:
- Extracting MDP model from Gymnasium
- Solving real environments with DP
- Comparing deterministic vs stochastic
- Evaluating learned policies
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, List


def extract_mdp(env) -> Dict:
    """Extract MDP components from Gymnasium environment."""
    return {
        'n_states': env.observation_space.n,
        'n_actions': env.action_space.n,
        'P': env.unwrapped.P,  # Transition model
    }


def policy_evaluation(mdp: Dict, policy: np.ndarray, gamma: float = 0.99,
                      theta: float = 1e-8) -> np.ndarray:
    """Evaluate a policy using iterative policy evaluation."""
    n_states = mdp['n_states']
    P = mdp['P']
    V = np.zeros(n_states)

    while True:
        delta = 0
        V_new = np.zeros(n_states)

        for s in range(n_states):
            a = policy[s]
            for prob, next_state, reward, done in P[s][a]:
                V_new[s] += prob * (reward + gamma * V[next_state] * (1 - done))

            delta = max(delta, abs(V[s] - V_new[s]))

        V = V_new
        if delta < theta:
            break

    return V


def policy_improvement(mdp: Dict, V: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """Improve policy greedily with respect to value function."""
    n_states = mdp['n_states']
    n_actions = mdp['n_actions']
    P = mdp['P']
    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        action_values = np.zeros(n_actions)

        for a in range(n_actions):
            for prob, next_state, reward, done in P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state] * (1 - done))

        policy[s] = np.argmax(action_values)

    return policy


def policy_iteration(mdp: Dict, gamma: float = 0.99) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Policy Iteration Algorithm.

    Returns:
        V: Optimal value function
        policy: Optimal policy
        iterations: Number of policy improvement steps
    """
    n_states = mdp['n_states']

    # Initialize random policy
    policy = np.random.randint(0, mdp['n_actions'], size=n_states)

    iterations = 0
    while True:
        iterations += 1

        # Policy Evaluation
        V = policy_evaluation(mdp, policy, gamma)

        # Policy Improvement
        new_policy = policy_improvement(mdp, V, gamma)

        # Check convergence
        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

        if iterations > 100:
            print("Warning: Max iterations reached")
            break

    return V, policy, iterations


def value_iteration(mdp: Dict, gamma: float = 0.99,
                    theta: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Value Iteration Algorithm.

    Returns:
        V: Optimal value function
        policy: Optimal policy
        iterations: Number of iterations
    """
    n_states = mdp['n_states']
    n_actions = mdp['n_actions']
    P = mdp['P']

    V = np.zeros(n_states)
    iterations = 0

    while True:
        iterations += 1
        delta = 0
        V_new = np.zeros(n_states)

        for s in range(n_states):
            action_values = np.zeros(n_actions)

            for a in range(n_actions):
                for prob, next_state, reward, done in P[s][a]:
                    action_values[a] += prob * (reward + gamma * V[next_state] * (1 - done))

            V_new[s] = np.max(action_values)
            delta = max(delta, abs(V[s] - V_new[s]))

        V = V_new

        if delta < theta:
            break

        if iterations > 10000:
            print("Warning: Max iterations reached")
            break

    # Extract policy
    policy = policy_improvement(mdp, V, gamma)

    return V, policy, iterations


def evaluate_policy(env, policy: np.ndarray, n_episodes: int = 10000) -> Dict:
    """Evaluate policy by running episodes."""
    successes = 0
    total_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        length = 0
        done = False

        while not done:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            length += 1

            if length > 100:
                break

        total_rewards.append(total_reward)
        episode_lengths.append(length)

        if total_reward > 0:
            successes += 1

    return {
        'success_rate': successes / n_episodes,
        'avg_reward': np.mean(total_rewards),
        'avg_length': np.mean(episode_lengths),
    }


def render_policy(policy: np.ndarray, size: int = 4, env_desc: List[str] = None):
    """Render policy as a grid with arrows."""
    arrows = ['<', 'v', '>', '^']  # Left, Down, Right, Up

    print("\nOptimal Policy:")
    print("+" + "---+" * size)

    for row in range(size):
        line = "|"
        for col in range(size):
            state = row * size + col

            if env_desc:
                cell_type = env_desc[row][col]
                if cell_type == 'H':
                    cell = " H "
                elif cell_type == 'G':
                    cell = " G "
                elif cell_type == 'S':
                    cell = f" {arrows[policy[state]]}S"
                else:
                    cell = f" {arrows[policy[state]]} "
            else:
                cell = f" {arrows[policy[state]]} "

            line += cell + "|"
        print(line)
        print("+" + "---+" * size)


def render_values(V: np.ndarray, size: int = 4, env_desc: List[str] = None):
    """Render value function as a grid."""
    print("\nValue Function:")
    print("+" + "-------+" * size)

    for row in range(size):
        line = "|"
        for col in range(size):
            state = row * size + col

            if env_desc and env_desc[row][col] == 'H':
                cell = "  HOLE "
            elif env_desc and env_desc[row][col] == 'G':
                cell = "  GOAL "
            else:
                cell = f" {V[state]:.4f}"

            line += cell + "|"
        print(line)
        print("+" + "-------+" * size)


def main():
    print("=" * 60)
    print("SOLVING FROZENLAKE WITH DYNAMIC PROGRAMMING")
    print("=" * 60)

    # ============================================
    # 1. DETERMINISTIC FROZENLAKE
    # ============================================
    print("\n" + "=" * 60)
    print("1. DETERMINISTIC FROZENLAKE (is_slippery=False)")
    print("=" * 60)

    env_det = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)
    mdp_det = extract_mdp(env_det)

    # Get environment description for rendering
    env_desc = ['SFFF', 'FHFH', 'FFFH', 'HFFG']

    print("""
    Grid (4x4):
    S F F F     S = Start
    F H F H     F = Frozen (safe)
    F F F H     H = Hole (game over)
    H F F G     G = Goal (reward=1)

    Actions: 0=Left, 1=Down, 2=Right, 3=Up
    Deterministic: Actions always succeed
    """)

    # Policy Iteration
    print("Running Policy Iteration...")
    V_pi, policy_pi, iters_pi = policy_iteration(mdp_det, gamma=0.99)
    print(f"  Converged in {iters_pi} iterations")

    # Value Iteration
    print("\nRunning Value Iteration...")
    V_vi, policy_vi, iters_vi = value_iteration(mdp_det, gamma=0.99)
    print(f"  Converged in {iters_vi} iterations")

    # Compare
    print(f"\nPolicies match: {np.array_equal(policy_pi, policy_vi)}")
    print(f"Values match: {np.allclose(V_pi, V_vi)}")

    render_policy(policy_vi, 4, env_desc)
    render_values(V_vi, 4, env_desc)

    # Evaluate
    print("\nEvaluating optimal policy (10,000 episodes):")
    results_det = evaluate_policy(env_det, policy_vi)
    print(f"  Success rate: {results_det['success_rate'] * 100:.1f}%")
    print(f"  Avg steps: {results_det['avg_length']:.1f}")

    env_det.close()

    # ============================================
    # 2. STOCHASTIC FROZENLAKE
    # ============================================
    print("\n" + "=" * 60)
    print("2. STOCHASTIC FROZENLAKE (is_slippery=True)")
    print("=" * 60)

    env_stoch = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True)
    mdp_stoch = extract_mdp(env_stoch)

    print("""
    Same grid, but now:
    - 1/3 chance: Move in intended direction
    - 1/3 chance: Slip left of intended
    - 1/3 chance: Slip right of intended

    This makes the problem MUCH harder!
    """)

    # Show transition example
    print("Example transitions from state 6, action Right:")
    for prob, next_state, reward, done in mdp_stoch['P'][6][2]:
        print(f"  P(s'={next_state}) = {prob:.4f}, done={done}")

    # Policy Iteration
    print("\nRunning Policy Iteration...")
    V_pi_s, policy_pi_s, iters_pi_s = policy_iteration(mdp_stoch, gamma=0.99)
    print(f"  Converged in {iters_pi_s} iterations")

    # Value Iteration
    print("\nRunning Value Iteration...")
    V_vi_s, policy_vi_s, iters_vi_s = value_iteration(mdp_stoch, gamma=0.99)
    print(f"  Converged in {iters_vi_s} iterations")

    render_policy(policy_vi_s, 4, env_desc)
    render_values(V_vi_s, 4, env_desc)

    # Evaluate
    print("\nEvaluating optimal policy (10,000 episodes):")
    results_stoch = evaluate_policy(env_stoch, policy_vi_s)
    print(f"  Success rate: {results_stoch['success_rate'] * 100:.1f}%")
    print(f"  Avg steps: {results_stoch['avg_length']:.1f}")

    env_stoch.close()

    # ============================================
    # 3. COMPARISON: DETERMINISTIC VS STOCHASTIC
    # ============================================
    print("\n" + "=" * 60)
    print("3. COMPARISON: DETERMINISTIC VS STOCHASTIC")
    print("=" * 60)

    print("\n                    | Deterministic | Stochastic")
    print("  " + "-" * 50)
    print(f"  PI Iterations     |      {iters_pi:3d}      |    {iters_pi_s:3d}")
    print(f"  VI Iterations     |      {iters_vi:3d}      |    {iters_vi_s:3d}")
    print(f"  V(start)          |    {V_vi[0]:.4f}    |   {V_vi_s[0]:.4f}")
    print(f"  Success Rate      |    {results_det['success_rate'] * 100:5.1f}%    |   {results_stoch['success_rate'] * 100:5.1f}%")
    print(f"  Avg Steps         |     {results_det['avg_length']:5.1f}     |   {results_stoch['avg_length']:5.1f}")

    # Policy differences
    policy_diff = np.sum(policy_vi != policy_vi_s)
    print(f"\n  Policy differences: {policy_diff} states")

    print("\nPolicy comparison (states with different actions):")
    arrows = ['<', 'v', '>', '^']
    for s in range(16):
        if policy_vi[s] != policy_vi_s[s]:
            row, col = s // 4, s % 4
            if env_desc[row][col] not in ['H', 'G']:
                print(f"  State {s}: Det={arrows[policy_vi[s]]}, Stoch={arrows[policy_vi_s[s]]}")

    # ============================================
    # 4. EFFECT OF GAMMA
    # ============================================
    print("\n" + "=" * 60)
    print("4. EFFECT OF DISCOUNT FACTOR (GAMMA)")
    print("=" * 60)

    env_test = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True)
    mdp_test = extract_mdp(env_test)

    print("\nV(start) and Success Rate for different gamma values:")
    print("\n  Gamma  |  V(start)  | VI Iters | Success Rate")
    print("  " + "-" * 48)

    for gamma in [0.5, 0.7, 0.9, 0.95, 0.99]:
        V, policy, iters = value_iteration(mdp_test, gamma=gamma)
        results = evaluate_policy(env_test, policy, n_episodes=5000)
        print(f"  {gamma:.2f}   |   {V[0]:.4f}   |   {iters:4d}   |   {results['success_rate'] * 100:5.1f}%")

    env_test.close()

    # ============================================
    # 5. 8x8 FROZENLAKE
    # ============================================
    print("\n" + "=" * 60)
    print("5. LARGER ENVIRONMENT: 8x8 FROZENLAKE")
    print("=" * 60)

    env_8x8 = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True)
    mdp_8x8 = extract_mdp(env_8x8)

    print(f"\n8x8 Grid:")
    print(f"  States: {mdp_8x8['n_states']}")
    print(f"  Actions: {mdp_8x8['n_actions']}")

    print("\nRunning Value Iteration...")
    V_8x8, policy_8x8, iters_8x8 = value_iteration(mdp_8x8, gamma=0.99)
    print(f"  Converged in {iters_8x8} iterations")
    print(f"  V(start) = {V_8x8[0]:.6f}")

    print("\nEvaluating optimal policy (10,000 episodes):")
    results_8x8 = evaluate_policy(env_8x8, policy_8x8)
    print(f"  Success rate: {results_8x8['success_rate'] * 100:.2f}%")
    print(f"  Avg steps: {results_8x8['avg_length']:.1f}")

    env_8x8.close()

    # ============================================
    # 6. RANDOM VS OPTIMAL POLICY
    # ============================================
    print("\n" + "=" * 60)
    print("6. RANDOM VS OPTIMAL POLICY")
    print("=" * 60)

    env_comp = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True)
    mdp_comp = extract_mdp(env_comp)

    # Optimal policy
    _, optimal_policy, _ = value_iteration(mdp_comp, gamma=0.99)

    # Random policy
    random_policy = np.random.randint(0, 4, size=16)

    print("\nComparing policies (10,000 episodes each):")

    results_opt = evaluate_policy(env_comp, optimal_policy)
    results_rand = evaluate_policy(env_comp, random_policy)

    print(f"\n              | Success Rate | Avg Reward | Avg Steps")
    print("  " + "-" * 50)
    print(f"  Optimal     |    {results_opt['success_rate'] * 100:5.1f}%    |   {results_opt['avg_reward']:.4f}   |   {results_opt['avg_length']:.1f}")
    print(f"  Random      |    {results_rand['success_rate'] * 100:5.1f}%    |   {results_rand['avg_reward']:.4f}   |   {results_rand['avg_length']:.1f}")

    improvement = (results_opt['success_rate'] - results_rand['success_rate']) / results_rand['success_rate'] * 100
    print(f"\n  Improvement: {improvement:.1f}%")

    env_comp.close()

    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("""
    Key Takeaways:

    1. DP SOLVES FROZENLAKE OPTIMALLY
       - Given the full MDP model (from env.P)
       - Both Policy Iteration and Value Iteration work

    2. STOCHASTICITY MATTERS
       - Deterministic: 100% success possible
       - Stochastic (slippery): ~74% success (4x4)
       - Optimal policy adapts to uncertainty

    3. DISCOUNT FACTOR IMPACT
       - Higher gamma = more iterations
       - But also better long-term planning
       - gamma=0.99 is typical choice

    4. SCALABILITY
       - 4x4 (16 states): Fast
       - 8x8 (64 states): Still manageable
       - DP scales to O(|S|^2 * |A|)

    5. OPTIMAL VS RANDOM
       - Optimal policy vastly outperforms random
       - Even in stochastic environment

    Limitation: DP requires full model knowledge (P, R).
    Next: Monte Carlo and TD methods learn without model!
    """)


if __name__ == "__main__":
    main()
