"""
07 - Algorithm Comparison: Policy Gradient vs Value-Based Methods

This script provides a comprehensive comparison between policy gradient methods
(REINFORCE) and value-based methods (DQN, Q-Learning).

Demonstrates:
- Side-by-side training comparison
- Sample efficiency analysis
- Continuous vs discrete action support
- When to use each method
- Pros and cons visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
from collections import deque
import random


# ============================================================
# POLICY GRADIENT (REINFORCE)
# ============================================================

class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.network(state)
        return F.softmax(logits, dim=-1)

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob


def train_reinforce(env: gym.Env, num_episodes: int = 500) -> List[float]:
    """Train REINFORCE agent."""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = PolicyNetwork(state_size, action_size, 64)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    gamma = 0.99

    rewards_history = []

    for episode in range(num_episodes):
        log_probs = []
        rewards = []

        state, _ = env.reset()
        done = False

        while not done:
            action, log_prob = policy_net.select_action(state)
            log_probs.append(log_prob)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            state = next_state

        # Compute returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        returns_tensor = torch.FloatTensor(returns)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)

        # Policy loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns_tensor):
            policy_loss.append(-log_prob * G)

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()

        rewards_history.append(sum(rewards))

    return rewards_history


# ============================================================
# VALUE-BASED (SIMPLE DQN)
# ============================================================

class DQNNetwork(nn.Module):
    """Q-Network for DQN."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


def train_dqn(env: gym.Env, num_episodes: int = 500) -> List[float]:
    """Train DQN agent."""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    q_net = DQNNetwork(state_size, action_size, 64)
    target_net = DQNNetwork(state_size, action_size, 64)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(10000)

    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 32
    target_update = 10

    rewards_history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = q_net(state_tensor)
                    action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            # Train if enough samples
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.LongTensor(actions)
                rewards_tensor = torch.FloatTensor(rewards)
                next_states_tensor = torch.FloatTensor(next_states)
                dones_tensor = torch.FloatTensor(dones)

                # Current Q values
                current_q = q_net(states_tensor).gather(1, actions_tensor.unsqueeze(1))

                # Target Q values
                with torch.no_grad():
                    max_next_q = target_net(next_states_tensor).max(1)[0]
                    target_q = rewards_tensor + gamma * max_next_q * (1 - dones_tensor)

                # Loss and optimize
                loss = F.mse_loss(current_q.squeeze(), target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards_history.append(episode_reward)

    return rewards_history


# ============================================================
# COMPARISON AND VISUALIZATION
# ============================================================

def compare_algorithms() -> None:
    """Compare REINFORCE and DQN on CartPole."""
    print("=" * 60)
    print("TRAINING COMPARISON: REINFORCE vs DQN")
    print("=" * 60)

    num_episodes = 500
    num_runs = 3

    print(f"\nRunning {num_runs} independent trials for each algorithm...")
    print(f"Episodes per trial: {num_episodes}")

    # Train REINFORCE
    print("\n1. Training REINFORCE...")
    reinforce_rewards = []
    for run in range(num_runs):
        env = gym.make('CartPole-v1')
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        random.seed(42 + run)

        rewards = train_reinforce(env, num_episodes)
        reinforce_rewards.append(rewards)
        env.close()
        print(f"   Run {run+1}/{num_runs} complete. Final avg: {np.mean(rewards[-50:]):.2f}")

    # Train DQN
    print("\n2. Training DQN...")
    dqn_rewards = []
    for run in range(num_runs):
        env = gym.make('CartPole-v1')
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        random.seed(42 + run)

        rewards = train_dqn(env, num_episodes)
        dqn_rewards.append(rewards)
        env.close()
        print(f"   Run {run+1}/{num_runs} complete. Final avg: {np.mean(rewards[-50:]):.2f}")

    # Average over runs
    reinforce_avg = np.mean(reinforce_rewards, axis=0)
    dqn_avg = np.mean(dqn_rewards, axis=0)

    # Plot comparison
    plot_comparison(reinforce_avg, dqn_avg)

    # Print statistics
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    print(f"\nREINFORCE:")
    print(f"  Final 50 episodes avg: {np.mean(reinforce_avg[-50:]):.2f}")
    print(f"  Best episode: {np.max(reinforce_avg):.2f}")
    print(f"  Episodes to solve (>195): {np.argmax(reinforce_avg > 195) if np.any(reinforce_avg > 195) else 'Not solved'}")

    print(f"\nDQN:")
    print(f"  Final 50 episodes avg: {np.mean(dqn_avg[-50:]):.2f}")
    print(f"  Best episode: {np.max(dqn_avg):.2f}")
    print(f"  Episodes to solve (>195): {np.argmax(dqn_avg > 195) if np.any(dqn_avg > 195) else 'Not solved'}")


def plot_comparison(reinforce_rewards: np.ndarray, dqn_rewards: np.ndarray) -> None:
    """Plot comparison between REINFORCE and DQN."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    window = 20

    # 1. Learning curves
    ax = axes[0, 0]
    if len(reinforce_rewards) >= window:
        smoothed_reinforce = np.convolve(reinforce_rewards, np.ones(window)/window, mode='valid')
        smoothed_dqn = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')

        ax.plot(smoothed_reinforce, label='REINFORCE', color='blue', linewidth=2)
        ax.plot(smoothed_dqn, label='DQN', color='red', linewidth=2)
        ax.axhline(y=195, color='green', linestyle='--', alpha=0.7, label='Solved')
        ax.set_xlabel('Episode')
        ax.set_ylabel(f'Smoothed Reward (window={window})')
        ax.set_title('Learning Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Cumulative rewards
    ax = axes[0, 1]
    cumsum_reinforce = np.cumsum(reinforce_rewards)
    cumsum_dqn = np.cumsum(dqn_rewards)

    ax.plot(cumsum_reinforce, label='REINFORCE', color='blue', linewidth=2)
    ax.plot(cumsum_dqn, label='DQN', color='red', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Sample Efficiency Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Reward distributions
    ax = axes[1, 0]
    ax.hist(reinforce_rewards, bins=30, alpha=0.5, label='REINFORCE', color='blue')
    ax.hist(dqn_rewards, bins=30, alpha=0.5, label='DQN', color='red')
    ax.axvline(x=195, color='green', linestyle='--', label='Solved threshold')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Method comparison table
    ax = axes[1, 1]
    ax.axis('off')

    comparison_text = """
    POLICY GRADIENT (REINFORCE):
    ✓ Direct policy optimization
    ✓ Handles continuous actions naturally
    ✓ Guaranteed convergence
    ✓ Stochastic policy (natural exploration)
    ✗ High variance (slow learning)
    ✗ On-policy (less sample efficient)
    ✗ Requires full episodes

    VALUE-BASED (DQN):
    ✓ Sample efficient (experience replay)
    ✓ Off-policy learning
    ✓ Can learn from any data
    ✓ Lower variance with bootstrapping
    ✗ Discrete actions only
    ✗ Convergence not guaranteed
    ✗ Can overestimate Q-values
    """

    ax.text(0.1, 0.95, comparison_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Policy Gradient vs Value-Based Methods', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to 'algorithm_comparison.png'")
    plt.close()


def print_decision_guide() -> None:
    """Print guide for choosing between methods."""
    print("\n" + "=" * 60)
    print("WHEN TO USE EACH METHOD")
    print("=" * 60)

    print("\nUse POLICY GRADIENT when:")
    print("  ✓ Actions are continuous (robot control, etc.)")
    print("  ✓ Stochastic policy is desired")
    print("  ✓ Policy is simpler than value function")
    print("  ✓ Guaranteed convergence is important")
    print("  ✓ You can afford on-policy learning")

    print("\nUse VALUE-BASED (DQN) when:")
    print("  ✓ Actions are discrete and finite")
    print("  ✓ Sample efficiency is critical")
    print("  ✓ Off-policy learning is beneficial")
    print("  ✓ You have access to experience replay")
    print("  ✓ Deterministic policy is acceptable")

    print("\nBest of Both Worlds:")
    print("  → Actor-Critic methods (Week 10)")
    print("     - Policy gradient for the actor")
    print("     - Value function for the critic")
    print("     - Lower variance, better sample efficiency")


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("WEEK 9 - LESSON 7: ALGORITHM COMPARISON")
    print("Policy Gradient vs Value-Based Methods")
    print("=" * 60)

    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    # Run comparison
    compare_algorithms()

    # Print decision guide
    print_decision_guide()

    # Summary
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Policy gradient: Direct policy optimization")
    print("2. Value-based: Learn Q-values, derive policy")
    print("3. Each has strengths and weaknesses")
    print("4. Choice depends on problem characteristics")
    print("5. Actor-Critic combines both approaches (Week 10)")
    print("\nCongratulations! You've completed Week 9!")
    print("=" * 60)


if __name__ == "__main__":
    main()
