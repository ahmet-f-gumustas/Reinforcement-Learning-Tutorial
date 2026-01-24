"""
06 - Taxi Problem

Solving the Taxi-v3 environment from Gymnasium using TD control methods.

Demonstrates:
- Discrete state/action environment
- Hierarchical task (pickup + dropoff)
- Algorithm comparison on standard benchmark
- Success rate and reward tracking
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    print("Gymnasium not installed. Using simplified Taxi environment.")


class SimpleTaxiEnv:
    """
    Simplified Taxi environment for when Gymnasium is not available.

    5x5 grid with 4 pickup/dropoff locations (R, G, Y, B).
    Task: Pick up passenger and deliver to destination.
    """

    def __init__(self):
        self.grid_size = 5
        self.n_locations = 4  # R, G, Y, B
        self.n_states = self.grid_size * self.grid_size * self.n_locations * (self.n_locations + 1)
        self.gamma = 0.99

        # Actions: 0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff
        self.actions = list(range(6))
        self.n_actions = 6

        # Locations: (row, col)
        self.locations = {
            0: (0, 0),  # R
            1: (0, 4),  # G
            2: (4, 0),  # Y
            3: (4, 3),  # B
        }

        self.reset()

    def _encode_state(self, taxi_row, taxi_col, pass_loc, dest):
        return ((taxi_row * 5 + taxi_col) * 5 + pass_loc) * 4 + dest

    def _decode_state(self, state):
        dest = state % 4
        state //= 4
        pass_loc = state % 5
        state //= 5
        taxi_col = state % 5
        taxi_row = state // 5
        return taxi_row, taxi_col, pass_loc, dest

    def reset(self) -> int:
        self.taxi_row = np.random.randint(0, 5)
        self.taxi_col = np.random.randint(0, 5)
        self.pass_loc = np.random.randint(0, 4)  # At one of 4 locations
        self.dest = np.random.randint(0, 4)
        while self.dest == self.pass_loc:
            self.dest = np.random.randint(0, 4)
        return self._encode_state(self.taxi_row, self.taxi_col, self.pass_loc, self.dest)

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        reward = -1  # Default step penalty
        done = False

        if action == 0:  # South
            self.taxi_row = min(self.taxi_row + 1, 4)
        elif action == 1:  # North
            self.taxi_row = max(self.taxi_row - 1, 0)
        elif action == 2:  # East
            self.taxi_col = min(self.taxi_col + 1, 4)
        elif action == 3:  # West
            self.taxi_col = max(self.taxi_col - 1, 0)
        elif action == 4:  # Pickup
            if self.pass_loc < 4:
                loc = self.locations[self.pass_loc]
                if (self.taxi_row, self.taxi_col) == loc:
                    self.pass_loc = 4  # In taxi
                else:
                    reward = -10
            else:
                reward = -10
        elif action == 5:  # Dropoff
            if self.pass_loc == 4:  # Passenger in taxi
                loc = self.locations[self.dest]
                if (self.taxi_row, self.taxi_col) == loc:
                    reward = 20
                    done = True
                else:
                    reward = -10
            else:
                reward = -10

        state = self._encode_state(self.taxi_row, self.taxi_col, self.pass_loc, self.dest)
        return state, reward, done, False, {}


def get_env():
    """Get Taxi environment (Gymnasium or custom)."""
    if HAS_GYM:
        env = gym.make('Taxi-v3')
        env.gamma = 0.99
        env.actions = list(range(6))
        env.n_states = 500
        env.n_actions = 6
        return env
    else:
        return SimpleTaxiEnv()


def epsilon_greedy(Q: Dict, state: int, n_actions: int, epsilon: float) -> int:
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        q_values = [Q.get((state, a), 0.0) for a in range(n_actions)]
        max_q = max(q_values)
        best_actions = [a for a in range(n_actions) if q_values[a] == max_q]
        return np.random.choice(best_actions)


def sarsa_taxi(env, n_episodes: int = 2000, alpha: float = 0.1,
               epsilon: float = 0.1, epsilon_decay: float = 0.999) -> Tuple[Dict, List, List]:
    """SARSA on Taxi environment."""
    Q = defaultdict(float)
    rewards_history = []
    success_history = []
    current_epsilon = epsilon

    for episode in range(n_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        action = epsilon_greedy(Q, state, env.n_actions, current_epsilon)
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done = result[:3]

            total_reward += reward

            next_action = epsilon_greedy(Q, next_state, env.n_actions, current_epsilon)

            # SARSA update
            Q[(state, action)] += alpha * (
                reward + env.gamma * Q[(next_state, next_action)] - Q[(state, action)]
            )

            state = next_state
            action = next_action
            steps += 1

        rewards_history.append(total_reward)
        success_history.append(1 if total_reward > 0 else 0)
        current_epsilon *= epsilon_decay

        if episode % 500 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            print(f"SARSA Episode {episode}, Avg Reward: {avg_reward:.1f}, ε: {current_epsilon:.3f}")

    return dict(Q), rewards_history, success_history


def q_learning_taxi(env, n_episodes: int = 2000, alpha: float = 0.1,
                    epsilon: float = 0.1, epsilon_decay: float = 0.999) -> Tuple[Dict, List, List]:
    """Q-Learning on Taxi environment."""
    Q = defaultdict(float)
    rewards_history = []
    success_history = []
    current_epsilon = epsilon

    for episode in range(n_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            action = epsilon_greedy(Q, state, env.n_actions, current_epsilon)

            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done = result[:3]

            total_reward += reward

            # Q-Learning update
            max_next_q = max([Q[(next_state, a)] for a in range(env.n_actions)])
            Q[(state, action)] += alpha * (
                reward + env.gamma * max_next_q - Q[(state, action)]
            )

            state = next_state
            steps += 1

        rewards_history.append(total_reward)
        success_history.append(1 if total_reward > 0 else 0)
        current_epsilon *= epsilon_decay

        if episode % 500 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            print(f"Q-Learning Episode {episode}, Avg Reward: {avg_reward:.1f}, ε: {current_epsilon:.3f}")

    return dict(Q), rewards_history, success_history


def expected_sarsa_taxi(env, n_episodes: int = 2000, alpha: float = 0.1,
                        epsilon: float = 0.1, epsilon_decay: float = 0.999) -> Tuple[Dict, List, List]:
    """Expected SARSA on Taxi environment."""
    Q = defaultdict(float)
    rewards_history = []
    success_history = []
    current_epsilon = epsilon

    for episode in range(n_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            action = epsilon_greedy(Q, state, env.n_actions, current_epsilon)

            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done = result[:3]

            total_reward += reward

            # Expected SARSA update
            q_values = [Q[(next_state, a)] for a in range(env.n_actions)]
            max_q = max(q_values)
            best_actions = [a for a in range(env.n_actions) if q_values[a] == max_q]

            # Calculate expected Q
            expected_q = 0
            for a in range(env.n_actions):
                if a in best_actions:
                    prob = (1 - current_epsilon) / len(best_actions) + current_epsilon / env.n_actions
                else:
                    prob = current_epsilon / env.n_actions
                expected_q += prob * q_values[a]

            Q[(state, action)] += alpha * (reward + env.gamma * expected_q - Q[(state, action)])

            state = next_state
            steps += 1

        rewards_history.append(total_reward)
        success_history.append(1 if total_reward > 0 else 0)
        current_epsilon *= epsilon_decay

        if episode % 500 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            print(f"Expected SARSA Episode {episode}, Avg Reward: {avg_reward:.1f}, ε: {current_epsilon:.3f}")

    return dict(Q), rewards_history, success_history


def compare_algorithms(n_runs: int = 5):
    """Compare all algorithms on Taxi."""
    print("\n" + "="*60)
    print("TAXI PROBLEM: ALGORITHM COMPARISON")
    print("="*60)

    env = get_env()
    n_episodes = 2000

    sarsa_results = []
    q_results = []
    exp_sarsa_results = []

    print(f"\nRunning {n_runs} trials...")
    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")

        _, s_rewards, _ = sarsa_taxi(env, n_episodes)
        _, q_rewards, _ = q_learning_taxi(env, n_episodes)
        _, e_rewards, _ = expected_sarsa_taxi(env, n_episodes)

        sarsa_results.append(s_rewards)
        q_results.append(q_rewards)
        exp_sarsa_results.append(e_rewards)

    # Convert to arrays
    sarsa_results = np.array(sarsa_results)
    q_results = np.array(q_results)
    exp_sarsa_results = np.array(exp_sarsa_results)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Taxi Problem: Algorithm Comparison', fontsize=16)

    # Plot 1: Learning curves
    ax = axes[0, 0]
    window = 100

    for results, name, color in [(sarsa_results, 'SARSA', 'blue'),
                                  (q_results, 'Q-Learning', 'orange'),
                                  (exp_sarsa_results, 'Expected SARSA', 'green')]:
        mean = results.mean(axis=0)
        smoothed = np.convolve(mean, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=name, linewidth=2, color=color)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward (smoothed)', fontsize=12)
    ax.set_title('Learning Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Success rate over time
    ax = axes[0, 1]
    for results, name, color in [(sarsa_results, 'SARSA', 'blue'),
                                  (q_results, 'Q-Learning', 'orange'),
                                  (exp_sarsa_results, 'Expected SARSA', 'green')]:
        # Calculate success rate (reward > 0)
        success = (results > 0).astype(float).mean(axis=0)
        smoothed = np.convolve(success, np.ones(window)/window, mode='valid')
        ax.plot(smoothed * 100, label=name, linewidth=2, color=color)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate Over Time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 3: Final performance
    ax = axes[1, 0]
    final_rewards = [
        sarsa_results[:, -200:].mean(axis=1),
        q_results[:, -200:].mean(axis=1),
        exp_sarsa_results[:, -200:].mean(axis=1)
    ]
    ax.boxplot(final_rewards, labels=['SARSA', 'Q-Learning', 'Exp. SARSA'])
    ax.set_ylabel('Average Reward (last 200 episodes)', fontsize=12)
    ax.set_title('Final Performance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Convergence speed
    ax = axes[1, 1]

    def episodes_to_threshold(results, threshold=5):
        """Find first episode where avg reward > threshold."""
        mean = results.mean(axis=0)
        window = 50
        smoothed = np.convolve(mean, np.ones(window)/window, mode='valid')
        for i, val in enumerate(smoothed):
            if val > threshold:
                return i
        return len(smoothed)

    convergence_times = [
        episodes_to_threshold(sarsa_results),
        episodes_to_threshold(q_results),
        episodes_to_threshold(exp_sarsa_results)
    ]

    colors = ['blue', 'orange', 'green']
    ax.bar(['SARSA', 'Q-Learning', 'Exp. SARSA'], convergence_times, color=colors)
    ax.set_ylabel('Episodes to Avg Reward > 5', fontsize=12)
    ax.set_title('Convergence Speed', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/06_td_control/taxi_comparison.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved comparison to taxi_comparison.png")
    plt.close()

    # Print statistics
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"\nFinal Performance (last 200 episodes):")
    for name, results in [('SARSA', sarsa_results),
                          ('Q-Learning', q_results),
                          ('Expected SARSA', exp_sarsa_results)]:
        final = results[:, -200:].mean(axis=1)
        print(f"  {name:15s}: {final.mean():.1f} ± {final.std():.1f}")


def demonstrate_learned_policy(env, Q: Dict):
    """Demonstrate the learned policy."""
    print("\n" + "="*60)
    print("DEMONSTRATING LEARNED POLICY")
    print("="*60)

    total_rewards = []

    for _ in range(10):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 50:
            # Greedy action
            q_values = [Q.get((state, a), 0.0) for a in range(env.n_actions)]
            action = np.argmax(q_values)

            result = env.step(action)
            if len(result) == 5:
                state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                state, reward, done = result[:3]

            total_reward += reward
            steps += 1

        total_rewards.append(total_reward)

    print(f"\n10 Test Episodes (greedy policy):")
    print(f"  Average Reward: {np.mean(total_rewards):.1f}")
    print(f"  Success Rate: {100 * np.mean(np.array(total_rewards) > 0):.0f}%")
    print(f"  Min/Max: {min(total_rewards):.0f} / {max(total_rewards):.0f}")


def main():
    """Main demonstration."""
    print("="*60)
    print("TAXI PROBLEM")
    print("="*60)

    print("""
    The Taxi Problem:
    - 5x5 grid with 4 designated locations (R, G, Y, B)
    - Taxi starts at random position
    - Passenger waiting at one of 4 locations
    - Task: Pick up passenger, deliver to destination

    Actions:
    - 0: Move South
    - 1: Move North
    - 2: Move East
    - 3: Move West
    - 4: Pickup
    - 5: Dropoff

    Rewards:
    - Successful dropoff: +20
    - Illegal pickup/dropoff: -10
    - Each step: -1
    """)

    # Compare algorithms
    compare_algorithms(n_runs=5)

    # Train Q-Learning and demonstrate
    env = get_env()
    print("\nTraining Q-Learning for demonstration...")
    Q, _, _ = q_learning_taxi(env, n_episodes=2000)
    demonstrate_learned_policy(env, Q)

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Findings:")
    print("  - All algorithms can solve Taxi")
    print("  - Q-Learning often converges fastest")
    print("  - Expected SARSA provides stable learning")
    print("  - Success rate reaches ~100% with good training")


if __name__ == "__main__":
    main()
