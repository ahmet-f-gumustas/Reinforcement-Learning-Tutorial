"""
06 - Complete CartPole Implementation with Policy Gradient

This script provides a complete, production-ready implementation of REINFORCE
with baseline on the CartPole environment. Includes hyperparameter tuning,
model saving/loading, and comprehensive visualization.

Demonstrates:
- Complete REINFORCE with baseline implementation
- Hyperparameter tuning
- Model checkpointing and loading
- Comprehensive training visualization
- Performance evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict
import os


class PolicyNetwork(nn.Module):
    """Policy network for CartPole."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.network(state)
        return F.softmax(logits, dim=-1)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        Select action from policy.

        Args:
            state: Current state
            deterministic: If True, select argmax instead of sampling

        Returns:
            action: Selected action
            log_prob: Log probability of action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state_tensor)

        if deterministic:
            action = action_probs.argmax()
            log_prob = torch.log(action_probs.squeeze()[action])
        else:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

        return action.item(), log_prob


class ValueNetwork(nn.Module):
    """Value network for baseline."""

    def __init__(self, state_size: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class REINFORCEAgent:
    """
    Complete REINFORCE agent with baseline.

    Encapsulates policy, value network, optimizers, and training logic.
    """

    def __init__(self, state_size: int, action_size: int,
                 hidden_size: int = 128,
                 lr_policy: float = 0.001,
                 lr_value: float = 0.001,
                 gamma: float = 0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        # Networks
        self.policy_net = PolicyNetwork(state_size, action_size, hidden_size)
        self.value_net = ValueNetwork(state_size, hidden_size)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)

        # Training metrics
        self.rewards_history = []
        self.loss_history = []

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """Compute discounted returns."""
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)

    def train_episode(self, env: gym.Env) -> float:
        """
        Train for one episode.

        Returns:
            Episode reward
        """
        log_probs = []
        values = []
        rewards = []

        state, _ = env.reset()
        done = False

        # Collect episode
        while not done:
            action, log_prob = self.policy_net.select_action(state)
            log_probs.append(log_prob)

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = self.value_net(state_tensor)
            values.append(value)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)

            state = next_state

        # Compute returns and advantages
        returns = self.compute_returns(rewards)
        values_tensor = torch.cat(values).squeeze()

        advantages = returns - values_tensor.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        # Policy loss
        policy_loss = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        policy_loss = torch.stack(policy_loss).sum()

        # Value loss
        value_loss = F.mse_loss(values_tensor, returns)

        # Optimize policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()

        # Optimize value
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Track metrics
        episode_reward = sum(rewards)
        self.rewards_history.append(episode_reward)
        self.loss_history.append((policy_loss.item(), value_loss.item()))

        return episode_reward

    def train(self, env: gym.Env, num_episodes: int = 1000,
              print_every: int = 100, solve_threshold: float = 195.0) -> bool:
        """
        Train agent for multiple episodes.

        Args:
            env: Gymnasium environment
            num_episodes: Number of training episodes
            print_every: Print frequency
            solve_threshold: Average reward to consider environment solved

        Returns:
            True if environment was solved, False otherwise
        """
        running_reward = None
        solved = False

        for episode in range(1, num_episodes + 1):
            episode_reward = self.train_episode(env)

            # Update running reward
            if running_reward is None:
                running_reward = episode_reward
            else:
                running_reward = 0.05 * episode_reward + 0.95 * running_reward

            # Print progress
            if episode % print_every == 0:
                print(f"Episode {episode}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Running Avg: {running_reward:.2f}")

            # Check if solved
            if episode >= 100:
                recent_avg = np.mean(self.rewards_history[-100:])
                if recent_avg >= solve_threshold and not solved:
                    print(f"\n✓ Environment solved in {episode} episodes!")
                    print(f"  Average reward over last 100 episodes: {recent_avg:.2f}")
                    solved = True
                    break

        return solved

    def evaluate(self, env: gym.Env, num_episodes: int = 10,
                render: bool = False) -> float:
        """
        Evaluate trained agent.

        Args:
            env: Gymnasium environment
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes

        Returns:
            Average reward over evaluation episodes
        """
        total_reward = 0

        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = self.policy_net.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state

            total_reward += episode_reward
            print(f"  Eval Episode {episode + 1}: Reward = {episode_reward:.2f}")

        avg_reward = total_reward / num_episodes
        return avg_reward

    def save(self, filepath: str = "reinforce_cartpole.pth") -> None:
        """Save agent's networks."""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'rewards_history': self.rewards_history,
        }, filepath)
        print(f"Agent saved to '{filepath}'")

    def load(self, filepath: str = "reinforce_cartpole.pth") -> None:
        """Load agent's networks."""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.rewards_history = checkpoint['rewards_history']
        print(f"Agent loaded from '{filepath}'")


def plot_training_results(agent: REINFORCEAgent, filename: str = "cartpole_training.png") -> None:
    """Plot comprehensive training results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    rewards = agent.rewards_history

    # 1. Raw rewards
    ax = axes[0, 0]
    ax.plot(rewards, alpha=0.6, color='blue')
    ax.axhline(y=195, color='green', linestyle='--', label='Solved threshold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Raw Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Smoothed rewards
    ax = axes[0, 1]
    window = 50
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, color='red', linewidth=2)
        ax.axhline(y=195, color='green', linestyle='--', label='Solved threshold')
        ax.set_xlabel('Episode')
        ax.set_ylabel(f'Smoothed Reward (window={window})')
        ax.set_title('Smoothed Learning Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3. Policy and value losses
    ax = axes[1, 0]
    if agent.loss_history:
        policy_losses = [loss[0] for loss in agent.loss_history]
        value_losses = [loss[1] for loss in agent.loss_history]

        window = 50
        if len(policy_losses) >= window:
            smoothed_policy = np.convolve(policy_losses, np.ones(window)/window, mode='valid')
            smoothed_value = np.convolve(value_losses, np.ones(window)/window, mode='valid')

            ax.plot(smoothed_policy, label='Policy Loss', color='blue')
            ax.plot(smoothed_value, label='Value Loss', color='red')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title('Training Losses')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # 4. Reward distribution
    ax = axes[1, 1]
    ax.hist(rewards, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(rewards), color='red', linestyle='--',
               label=f'Mean: {np.mean(rewards):.2f}')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('REINFORCE Training on CartPole-v1', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Training results saved to '{filename}'")
    plt.close()


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("WEEK 9 - LESSON 6: COMPLETE CARTPOLE IMPLEMENTATION")
    print("Production-Ready REINFORCE with Baseline")
    print("=" * 60)

    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Create environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print(f"\nEnvironment: CartPole-v1")
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")
    print(f"  Solve threshold: 195.0 (avg over 100 episodes)")

    # Create agent
    agent = REINFORCEAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=128,
        lr_policy=0.001,
        lr_value=0.001,
        gamma=0.99
    )

    print("\nAgent Configuration:")
    print(f"  Hidden size: 128")
    print(f"  Learning rate (policy): 0.001")
    print(f"  Learning rate (value): 0.001")
    print(f"  Discount factor: 0.99")

    # Train agent
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    solved = agent.train(env, num_episodes=1000, print_every=100)

    # Plot results
    plot_training_results(agent)

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    avg_reward = agent.evaluate(env, num_episodes=10)
    print(f"\nAverage evaluation reward: {avg_reward:.2f}")

    # Save model
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    agent.save("reinforce_cartpole.pth")

    # Test loading
    print("\n" + "=" * 60)
    print("TESTING MODEL LOAD")
    print("=" * 60)

    new_agent = REINFORCEAgent(state_size, action_size)
    new_agent.load("reinforce_cartpole.pth")
    print("Model loaded successfully!")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total episodes trained: {len(agent.rewards_history)}")
    print(f"Best episode reward: {max(agent.rewards_history):.2f}")
    print(f"Final 100 episodes average: {np.mean(agent.rewards_history[-100:]):.2f}")
    print(f"Environment solved: {'Yes' if solved else 'No'}")

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Production-ready implementation with checkpointing")
    print("2. Encapsulation in Agent class for clean code")
    print("3. Comprehensive visualization of training metrics")
    print("4. Deterministic evaluation for fair comparison")
    print("5. Gradient clipping for stability")
    print("\nNext: Compare policy gradient with value-based methods!")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
