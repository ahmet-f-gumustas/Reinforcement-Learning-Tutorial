"""
02 - Model-Based Reinforcement Learning

This script introduces model-based RL, where the agent learns a model
of the environment dynamics and uses it for planning.

Demonstrates:
- Model-free vs model-based comparison
- Learning environment dynamics T(s'|s,a)
- Planning with a learned model (Dyna-Q)
- Model predictive control (MPC) concept
- When model-based methods excel
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict
from collections import defaultdict
import random


# =============================================================================
# Simple Environment
# =============================================================================

class SimpleGridWorld:
    """5x5 GridWorld for model-based RL demonstrations."""

    def __init__(self, size: int = 5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.goal = (4, 4)
        self.obstacles = [(1, 1), (2, 3), (3, 1)]
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

        new_pos = (r, c)
        if new_pos in self.obstacles:
            reward, done = -5.0, False
        elif new_pos == self.goal:
            reward, done = 10.0, True
            self.pos = new_pos
        else:
            reward, done = -0.1, False
            self.pos = new_pos
        return self._state(), reward, done


# =============================================================================
# Dyna-Q: Model-Based Q-Learning
# =============================================================================

class DynaQAgent:
    """
    Dyna-Q Agent (Sutton, 1991).

    Combines real experience with simulated experience from a learned model.

    Algorithm:
    1. Take action in real environment
    2. Update Q-values from real experience
    3. Update model T(s'|s,a) and R(s,a) from real experience
    4. Plan: repeat n times:
       a. Sample random previously visited (s, a)
       b. Use model to predict s', r
       c. Update Q-values from simulated experience

    The planning step (4) is the key: we get n free Q-updates per real step!
    """

    def __init__(self, n_states: int, n_actions: int,
                 lr: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 0.3, n_planning: int = 10):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning

        # Q-table
        self.q_table = np.zeros((n_states, n_actions))

        # Learned model: deterministic model of T(s,a) -> s', r
        self.model = {}
        self.visited = set()

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool):
        """Real experience update + model update + planning."""
        # 1. Q-learning update from real experience
        td_target = reward + self.gamma * np.max(self.q_table[next_state]) * (1 - int(done))
        self.q_table[state, action] += self.lr * (td_target - self.q_table[state, action])

        # 2. Update model
        self.model[(state, action)] = (next_state, reward, done)
        self.visited.add((state, action))

        # 3. Planning: simulate from model
        if len(self.visited) > 0:
            visited_list = list(self.visited)
            for _ in range(self.n_planning):
                s, a = visited_list[np.random.randint(len(visited_list))]
                ns, r, d = self.model[(s, a)]
                td_target = r + self.gamma * np.max(self.q_table[ns]) * (1 - int(d))
                self.q_table[s, a] += self.lr * (td_target - self.q_table[s, a])


class QLearningAgent:
    """Standard Q-Learning agent (no model) for comparison."""

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
        td_target = reward + self.gamma * np.max(self.q_table[next_state]) * (1 - int(done))
        self.q_table[state, action] += self.lr * (td_target - self.q_table[state, action])


# =============================================================================
# Neural Network World Model
# =============================================================================

class NeuralWorldModel(nn.Module):
    """
    Neural network that predicts next state and reward.

    Given (state, action), predicts:
    - next_state (deterministic prediction)
    - reward

    This is the core of model-based deep RL.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.state_predictor = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_size)
        )
        self.reward_predictor = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        next_state = self.state_predictor(x)
        reward = self.reward_predictor(x)
        return next_state, reward


# =============================================================================
# Training and Comparison
# =============================================================================

def train_agent(agent, env, num_episodes=300, max_steps=50):
    """Train an agent and return episode rewards."""
    rewards = []
    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
    return rewards


def demonstrate_model_based_concepts():
    """Explain model-based RL."""
    print("=" * 60)
    print("MODEL-BASED REINFORCEMENT LEARNING")
    print("=" * 60)

    print("""
    Model-Free vs Model-Based:

    Model-Free (DQN, PPO, A2C):
    ┌─────────────────────────────────────────┐
    │  Agent ──action──> Environment           │
    │         <──reward,state──                │
    │  Learn policy/value directly from data   │
    └─────────────────────────────────────────┘

    Model-Based:
    ┌─────────────────────────────────────────┐
    │  Agent ──action──> Environment           │
    │         <──reward,state──                │
    │              │                            │
    │         Learn Model                      │
    │    T(s'|s,a) and R(s,a)                  │
    │              │                            │
    │     Use model to plan/simulate           │
    │  (get "free" experience from imagination)│
    └─────────────────────────────────────────┘

    Advantages of Model-Based:
    ✓ Much more sample efficient (plan from imagination)
    ✓ Can transfer model across tasks
    ✓ Can predict future consequences

    Disadvantages:
    ✗ Model errors can compound (compounding error problem)
    ✗ Harder to learn in complex environments
    ✗ Planning is computationally expensive

    Key Algorithms:
    ┌────────────────────┬───────────────────────────┐
    │ Algorithm           │ Description               │
    ├────────────────────┼───────────────────────────┤
    │ Dyna-Q (Sutton)    │ Q-learning + tabular model│
    │ MBPO               │ Model-based policy opt     │
    │ Dreamer            │ World model in latent space│
    │ MuZero             │ Model for planning (Go)    │
    │ MPC                │ Plan with learned dynamics │
    └────────────────────┴───────────────────────────┘
    """)


def main():
    print("\n" + "=" * 60)
    print("WEEK 12 - LESSON 2: MODEL-BASED RL")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    demonstrate_model_based_concepts()

    # Compare Q-Learning vs Dyna-Q
    print("\n" + "=" * 60)
    print("COMPARISON: Q-LEARNING vs DYNA-Q")
    print("=" * 60)

    env = SimpleGridWorld()
    planning_steps_list = [0, 5, 20, 50]
    results = {}

    for n_plan in planning_steps_list:
        np.random.seed(42)
        if n_plan == 0:
            agent = QLearningAgent(env.n_states, env.n_actions)
            name = "Q-Learning (no model)"
        else:
            agent = DynaQAgent(env.n_states, env.n_actions, n_planning=n_plan)
            name = f"Dyna-Q (n={n_plan})"

        rewards = train_agent(agent, env, num_episodes=300)
        results[name] = rewards
        print(f"  {name:25s} | Final avg: {np.mean(rewards[-50:]):.2f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['steelblue', 'coral', 'green', 'purple']
    window = 30

    ax = axes[0]
    for (name, rewards), color in zip(results.items(), colors):
        sm = [np.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
        ax.plot(sm, label=name, color=color, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Reward')
    ax.set_title('Learning Curves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    names = list(results.keys())
    finals = [np.mean(results[n][-50:]) for n in names]
    ax.bar(range(len(names)), finals, color=colors[:len(names)], edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split('(')[0].strip() + '\n' + '(' + n.split('(')[1]
                         if '(' in n else n for n in names], fontsize=8)
    ax.set_ylabel('Avg Reward (last 50 ep)')
    ax.set_title('Sample Efficiency')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Model-Free vs Model-Based (Dyna-Q)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('12_advanced/model_based.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to '12_advanced/model_based.png'")
    plt.close()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Model-based RL learns environment dynamics T(s'|s,a), R(s,a)")
    print("2. Dyna-Q: plan n steps for every real step → n× more efficient")
    print("3. More planning steps = faster learning but more compute")
    print("4. Model errors can compound over long horizons")
    print("5. Model-based excels in sample-limited settings")
    print("\nNext: Inverse Reinforcement Learning!")
    print("=" * 60)


if __name__ == "__main__":
    main()
