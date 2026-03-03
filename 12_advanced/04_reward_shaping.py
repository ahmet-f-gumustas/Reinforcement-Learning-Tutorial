"""
04 - Reward Shaping and Exploration

This script covers reward shaping techniques and advanced
exploration strategies.

Demonstrates:
- Potential-based reward shaping (PBRS)
- Intrinsic motivation (curiosity)
- Count-based exploration
- Sparse vs dense rewards
- Reward design best practices
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict
from collections import defaultdict


# =============================================================================
# Environment with Sparse Rewards
# =============================================================================

class SparseGridWorld:
    """
    GridWorld with sparse reward: only get reward at the goal.
    This makes exploration very challenging.
    """

    def __init__(self, size: int = 8):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.goal = (size-1, size-1)
        self.reset()

    def reset(self) -> int:
        self.pos = (0, 0)
        return self._state()

    def _state(self) -> int:
        return self.pos[0] * self.size + self.pos[1]

    def step(self, action: int) -> Tuple[int, float, bool]:
        r, c = self.pos
        if action == 0: r = max(0, r-1)
        elif action == 1: c = min(self.size-1, c+1)
        elif action == 2: r = min(self.size-1, r+1)
        elif action == 3: c = max(0, c-1)
        self.pos = (r, c)
        done = self.pos == self.goal
        reward = 1.0 if done else 0.0  # SPARSE reward
        return self._state(), reward, done


# =============================================================================
# Potential-Based Reward Shaping (PBRS)
# =============================================================================

class PBRSAgent:
    """
    Q-Learning agent with Potential-Based Reward Shaping.

    PBRS adds a shaping reward F(s, s') = γΦ(s') - Φ(s)
    where Φ(s) is a potential function.

    Key theorem (Ng et al., 1999):
    PBRS preserves the optimal policy! The agent will converge to
    the same policy as without shaping, but potentially faster.
    """

    def __init__(self, n_states, n_actions, potential_fn,
                 lr=0.1, gamma=0.99, epsilon=0.3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.potential = potential_fn
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        # Shaped reward: r + γΦ(s') - Φ(s)
        shaping = self.gamma * self.potential(next_state) - self.potential(state)
        shaped_reward = reward + shaping

        td_target = shaped_reward + self.gamma * np.max(self.q_table[next_state]) * (1-int(done))
        self.q_table[state, action] += self.lr * (td_target - self.q_table[state, action])


# =============================================================================
# Count-Based Exploration
# =============================================================================

class CountBasedAgent:
    """
    Q-Learning with count-based exploration bonus.

    Adds intrinsic reward: r_i = β / sqrt(N(s))
    where N(s) is the visit count for state s.

    Rarely visited states get higher bonus → encourages exploration.
    """

    def __init__(self, n_states, n_actions, beta=1.0,
                 lr=0.1, gamma=0.99, epsilon=0.3):
        self.n_actions = n_actions
        self.beta = beta
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))
        self.visit_counts = np.zeros(n_states)

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        self.visit_counts[state] += 1

        # Intrinsic reward: exploration bonus
        intrinsic = self.beta / np.sqrt(self.visit_counts[state])
        total_reward = reward + intrinsic

        td_target = total_reward + self.gamma * np.max(self.q_table[next_state]) * (1-int(done))
        self.q_table[state, action] += self.lr * (td_target - self.q_table[state, action])


# =============================================================================
# Standard Q-Learning (Baseline)
# =============================================================================

class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=0.3):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        td_target = reward + self.gamma * np.max(self.q_table[next_state]) * (1-int(done))
        self.q_table[state, action] += self.lr * (td_target - self.q_table[state, action])


# =============================================================================
# Training
# =============================================================================

def train_agent(agent, env, num_episodes=1000, max_steps=100):
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state
            if done: break
        rewards.append(ep_reward)
    return rewards


def demonstrate_reward_shaping():
    """Explain reward shaping concepts."""
    print("=" * 60)
    print("REWARD SHAPING AND EXPLORATION")
    print("=" * 60)

    print("""
    The Sparse Reward Problem:
    Many real-world tasks have SPARSE rewards (e.g., goal-reaching).
    The agent rarely gets a reward signal → very slow learning.

    Solutions:
    ┌────────────────────────────┬──────────────────────────────┐
    │ Technique                   │ Description                  │
    ├────────────────────────────┼──────────────────────────────┤
    │ Dense reward engineering   │ Hand-craft dense rewards     │
    │ PBRS (Ng et al.)          │ Add γΦ(s')-Φ(s) shaping     │
    │ Count-based exploration   │ Bonus for visiting new states│
    │ Curiosity (ICM, RND)      │ Predict next state; high     │
    │                            │ error = novel = bonus        │
    │ Hindsight (HER)           │ Relabel failed goals         │
    │ Curriculum learning       │ Start easy, increase difficulty│
    └────────────────────────────┴──────────────────────────────┘

    Potential-Based Reward Shaping (PBRS):
    F(s, s') = γΦ(s') - Φ(s)

    THEOREM: Adding F preserves the optimal policy!
    Any Φ(s) works, but good choices accelerate learning.

    Example: Φ(s) = -distance_to_goal → encourages approaching goal
    """)


def main():
    print("\n" + "=" * 60)
    print("WEEK 12 - LESSON 4: REWARD SHAPING")
    print("=" * 60)

    np.random.seed(42)

    demonstrate_reward_shaping()

    env = SparseGridWorld(size=8)

    # Potential function: negative distance to goal
    def potential(state):
        r, c = state // env.size, state % env.size
        gr, gc = env.goal
        return -float(abs(r-gr) + abs(c-gc))

    # Train all agents
    results = {}

    print("\nTraining on 8x8 GridWorld with SPARSE reward...")

    np.random.seed(42)
    agent = QLearningAgent(env.n_states, env.n_actions)
    results['Q-Learning\n(baseline)'] = train_agent(agent, env, num_episodes=2000)
    print(f"  Q-Learning final avg: {np.mean(results['Q-Learning\\n(baseline)'][-200:]):.3f}")

    np.random.seed(42)
    agent = PBRSAgent(env.n_states, env.n_actions, potential)
    results['PBRS\n(shaped)'] = train_agent(agent, env, num_episodes=2000)
    print(f"  PBRS final avg: {np.mean(results['PBRS\\n(shaped)'][-200:]):.3f}")

    np.random.seed(42)
    agent = CountBasedAgent(env.n_states, env.n_actions, beta=0.5)
    results['Count-Based\n(exploration)'] = train_agent(agent, env, num_episodes=2000)
    print(f"  Count-Based final avg: {np.mean(results['Count-Based\\n(exploration)'][-200:]):.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['steelblue', 'coral', 'green']
    window = 100

    ax = axes[0]
    for (name, rewards), color in zip(results.items(), colors):
        sm = [np.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
        ax.plot(sm, label=name.replace('\n', ' '), color=color, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Reward')
    ax.set_title('Learning Curves (Sparse Reward)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    names = list(results.keys())
    finals = [np.mean(results[n][-200:]) for n in names]
    ax.bar([n.replace('\n', '\n') for n in names], finals,
           color=colors, edgecolor='black')
    ax.set_ylabel('Avg Reward (last 200 ep)')
    ax.set_title('Final Performance')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Reward Shaping on Sparse GridWorld', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('12_advanced/reward_shaping.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to '12_advanced/reward_shaping.png'")
    plt.close()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Sparse rewards make exploration extremely hard")
    print("2. PBRS: add γΦ(s')-Φ(s) to reward — preserves optimal policy!")
    print("3. Count-based: bonus for visiting new states")
    print("4. Curiosity: predict next state, bonus for prediction error")
    print("5. Good reward design is crucial for practical RL")
    print("\nNext: Curriculum Learning!")
    print("=" * 60)


if __name__ == "__main__":
    main()
