"""
07 - Method Comparison: Actor-Critic vs Previous Methods

This script compares Actor-Critic methods with algorithms from
previous weeks: DQN (Week 8) and REINFORCE (Week 9).

Demonstrates:
- Side-by-side training comparison
- Sample efficiency analysis
- Learning stability comparison
- When to use which method
- Decision guide for algorithm selection
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict
import gymnasium as gym
from collections import deque
import random


# =============================================================================
# Algorithm Implementations
# =============================================================================

class DQNAgent:
    """DQN agent from Week 8 for comparison."""

    def __init__(self, state_size: int, action_size: int,
                 lr: float = 0.001, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, buffer_size: int = 10000,
                 batch_size: int = 64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.q_network = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_size)
        )
        self.target_network = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_size)
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)

    def select_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


class REINFORCEAgent:
    """REINFORCE agent from Week 9 for comparison."""

    def __init__(self, state_size: int, action_size: int,
                 lr: float = 0.001, gamma: float = 0.99):
        self.gamma = gamma
        self.network = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_size), nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state: np.ndarray) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.network(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def update(self):
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            loss += -log_prob * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()


class A2CAgent:
    """A2C agent from this week for comparison."""

    def __init__(self, state_size: int, action_size: int,
                 lr: float = 0.001, gamma: float = 0.99,
                 lam: float = 0.95, n_steps: int = 5):
        self.gamma = gamma
        self.lam = lam
        self.n_steps = n_steps

        self.shared = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

        params = (list(self.shared.parameters()) +
                  list(self.actor.parameters()) +
                  list(self.critic.parameters()))
        self.optimizer = optim.Adam(params, lr=lr)

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def select_action(self, state: np.ndarray) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        features = self.shared(state_tensor)
        logits = self.actor(features)
        value = self.critic(features).squeeze()

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(dist.log_prob(action))
        self.values.append(value)

        return action.item()

    def store(self, reward: float, done: bool):
        self.rewards.append(reward)
        self.dones.append(done)

    def should_update(self) -> bool:
        return len(self.rewards) >= self.n_steps

    def update(self, next_state: np.ndarray, done: bool):
        # Bootstrap value
        with torch.no_grad():
            features = self.shared(torch.FloatTensor(next_state).unsqueeze(0))
            next_value = self.critic(features).squeeze().item()

        # GAE
        advantages = []
        gae = 0
        values_list = [v.item() for v in self.values] + [next_value * (1 - int(done))]

        for t in reversed(range(len(self.rewards))):
            delta = (self.rewards[t] +
                     self.gamma * values_list[t+1] * (1 - int(self.dones[t])) -
                     values_list[t])
            gae = delta + self.gamma * self.lam * (1 - int(self.dones[t])) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages)
        values = torch.stack(self.values)
        returns = advantages + values.detach()

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs = torch.stack(self.log_probs)
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        total_loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.shared.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic.parameters()), 0.5)
        self.optimizer.step()

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()


# =============================================================================
# Training Functions
# =============================================================================

def train_dqn(env_name: str, num_episodes: int) -> List[float]:
    """Train DQN agent."""
    env = gym.make(env_name)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(state, action, reward, next_state, float(done))
            agent.update()
            episode_reward += reward
            state = next_state

        if (episode + 1) % 10 == 0:
            agent.update_target()

        rewards.append(episode_reward)

    env.close()
    return rewards


def train_reinforce(env_name: str, num_episodes: int) -> List[float]:
    """Train REINFORCE agent."""
    env = gym.make(env_name)
    agent = REINFORCEAgent(env.observation_space.shape[0], env.action_space.n)
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_reward(reward)
            episode_reward += reward
            state = next_state

        agent.update()
        rewards.append(episode_reward)

    env.close()
    return rewards


def train_a2c(env_name: str, num_episodes: int) -> List[float]:
    """Train A2C agent."""
    env = gym.make(env_name)
    agent = A2CAgent(env.observation_space.shape[0], env.action_space.n)
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(reward, done)
            episode_reward += reward

            if agent.should_update() or done:
                agent.update(next_state, done)

            state = next_state

        rewards.append(episode_reward)

    env.close()
    return rewards


# =============================================================================
# Comparison
# =============================================================================

def run_comparison():
    """Run full comparison of all three algorithms."""
    print("=" * 60)
    print("ALGORITHM COMPARISON ON CARTPOLE-V1")
    print("=" * 60)

    env_name = "CartPole-v1"
    num_episodes = 500

    # Train all agents
    results = {}

    for name, train_fn in [("DQN", train_dqn),
                             ("REINFORCE", train_reinforce),
                             ("A2C", train_a2c)]:
        print(f"\nTraining {name}...")
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        rewards = train_fn(env_name, num_episodes)
        results[name] = rewards
        avg = np.mean(rewards[-100:])
        print(f"  {name} final avg reward: {avg:.1f}")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'DQN': 'steelblue', 'REINFORCE': 'coral', 'A2C': 'green'}
    window = 50

    # Learning curves
    ax = axes[0]
    for name, rewards in results.items():
        smoothed = [np.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
        ax.plot(smoothed, label=name, color=colors[name], linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final performance
    ax = axes[1]
    names = list(results.keys())
    final_means = [np.mean(results[n][-100:]) for n in names]
    final_stds = [np.std(results[n][-100:]) for n in names]
    bars = ax.bar(names, final_means, yerr=final_stds, capsize=10,
                  color=[colors[n] for n in names])
    ax.set_ylabel('Average Reward (last 100 ep)')
    ax.set_title('Final Performance')
    ax.grid(True, alpha=0.3)

    # Reward variance over training
    ax = axes[2]
    for name, rewards in results.items():
        variance = [np.var(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
        ax.plot(variance, label=name, color=colors[name], alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward Variance')
    ax.set_title('Training Stability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('DQN vs REINFORCE vs A2C on CartPole-v1',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('10_actor_critic/method_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to '10_actor_critic/method_comparison.png'")
    plt.close()


def print_decision_guide():
    """Print algorithm selection guide."""
    print("\n" + "=" * 60)
    print("ALGORITHM SELECTION GUIDE")
    print("=" * 60)

    print("""
    ┌─────────────────┬──────────┬────────────┬───────────┐
    │ Feature          │   DQN    │ REINFORCE  │   A2C     │
    ├─────────────────┼──────────┼────────────┼───────────┤
    │ Action space    │ Discrete │ Both       │ Both      │
    │ Sample efficiency│ High    │ Low        │ Medium    │
    │ Variance        │ Low      │ High       │ Medium    │
    │ Bias            │ Some     │ None       │ Some      │
    │ On/Off-policy   │ Off      │ On         │ On        │
    │ Update frequency│ Per step │ Per episode│ Per n-step│
    │ Replay buffer   │ Yes      │ No         │ No        │
    │ Convergence     │ Unstable │ Guaranteed │ Good      │
    │ Implementation  │ Medium   │ Simple     │ Medium    │
    └─────────────────┴──────────┴────────────┴───────────┘

    When to use what:

    DQN:
    ✓ Discrete actions only
    ✓ Need sample efficiency (replay buffer)
    ✓ Can afford instability for better final performance
    ✗ Cannot handle continuous actions

    REINFORCE:
    ✓ Simple problems, educational purposes
    ✓ Need guaranteed convergence
    ✗ Slow learning, high variance
    ✗ Episode-based (cannot update mid-episode)

    A2C (Actor-Critic):
    ✓ Both discrete and continuous actions
    ✓ Good balance of variance and bias
    ✓ Online learning (update every n steps)
    ✓ Foundation for PPO (Week 11)

    General Rule:
    Start with A2C → Try PPO (Week 11) → Only use DQN if discrete-only
    """)


def main():
    print("\n" + "=" * 60)
    print("WEEK 10 - LESSON 7: METHOD COMPARISON")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    # 1. Run comparison
    run_comparison()

    # 2. Print decision guide
    print_decision_guide()

    print("\n" + "=" * 60)
    print("WEEK 10 SUMMARY")
    print("=" * 60)
    print("1. Actor-Critic combines policy gradient (actor) + value function (critic)")
    print("2. TD error serves as advantage estimate (lower variance than MC)")
    print("3. A2C: synchronous batch updates with n-step returns")
    print("4. A3C: asynchronous parallel workers (historical importance)")
    print("5. GAE: smooth bias-variance trade-off with lambda parameter")
    print("6. Actor-Critic works for both discrete and continuous actions")
    print("7. A2C is the foundation for PPO (next week!)")
    print("\nNext Week: PPO - Proximal Policy Optimization!")
    print("=" * 60)


if __name__ == "__main__":
    main()
