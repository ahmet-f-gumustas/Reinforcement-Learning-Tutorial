"""
05 - CartPole Actor-Critic: Complete Production Implementation

This script provides a complete, production-ready Actor-Critic agent
for CartPole-v1 with all best practices.

Demonstrates:
- Full A2C implementation with GAE
- Model saving and loading
- Hyperparameter configuration
- Training with early stopping
- Evaluation and testing
- Comprehensive visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import gymnasium as gym
import os
import json
from dataclasses import dataclass, asdict


@dataclass
class Config:
    """Training configuration."""
    env_name: str = "CartPole-v1"
    hidden_size: int = 128
    lr: float = 0.001
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    rollout_length: int = 128
    total_steps: int = 200000
    eval_interval: int = 10000
    eval_episodes: int = 10
    target_reward: float = 475.0
    seed: int = 42


class ActorCriticNet(nn.Module):
    """Production Actor-Critic network with orthogonal initialization."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.actor_head = nn.Linear(hidden_size, action_size)
        self.critic_head = nn.Linear(hidden_size, 1)

        # Orthogonal initialization (improves training)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        # Smaller init for output heads
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    def get_action(self, state: np.ndarray, deterministic: bool = False):
        """Get action with optional deterministic mode for evaluation."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.forward(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.squeeze().item()


class RolloutBuffer:
    """Buffer for collecting rollout data."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


def compute_gae(rewards, values, next_value, dones, gamma, lam):
    """Compute GAE advantages and returns."""
    advantages = []
    gae = 0
    values_ext = list(values) + [next_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t+1] * (1 - int(dones[t])) - values_ext[t]
        gae = delta + gamma * lam * (1 - int(dones[t])) * gae
        advantages.insert(0, gae)

    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(values)
    return advantages, returns


class CartPoleActorCritic:
    """Complete CartPole Actor-Critic agent."""

    def __init__(self, config: Config):
        self.config = config

        self.env = gym.make(config.env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.network = ActorCriticNet(
            self.state_size, self.action_size, config.hidden_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)
        self.buffer = RolloutBuffer()

        # Tracking
        self.episode_rewards = []
        self.eval_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []

    def collect_rollout(self, state: np.ndarray):
        """Collect rollout_length steps of experience."""
        self.buffer.clear()
        current_reward = 0

        for _ in range(self.config.rollout_length):
            action, log_prob, value = self.network.get_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.buffer.add(state, action, reward, done, log_prob, value)
            current_reward += reward

            if done:
                self.episode_rewards.append(current_reward)
                current_reward = 0
                state, _ = self.env.reset()
            else:
                state = next_state

        # Get bootstrap value
        with torch.no_grad():
            _, next_value = self.network(torch.FloatTensor(state).unsqueeze(0))
            next_value = next_value.squeeze().item()

        return state, next_value

    def update(self, next_value: float):
        """Perform A2C update with GAE."""
        advantages, returns = compute_gae(
            self.buffer.rewards, self.buffer.values, next_value,
            self.buffer.dones, self.config.gamma, self.config.lam
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_tensor = torch.FloatTensor(np.array(self.buffer.states))
        actions_tensor = torch.LongTensor(self.buffer.actions)

        logits, values = self.network(states_tensor)
        values = values.squeeze()

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions_tensor)
        entropy = dist.entropy().mean()

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns.detach())
        total_loss = (policy_loss
                      + self.config.value_coeff * value_loss
                      - self.config.entropy_coeff * entropy)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.entropies.append(entropy.item())

    def evaluate(self, n_episodes: int = 10) -> float:
        """Evaluate agent with deterministic policy."""
        eval_env = gym.make(self.config.env_name)
        rewards = []

        for _ in range(n_episodes):
            state, _ = eval_env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _, _ = self.network.get_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward

            rewards.append(episode_reward)

        eval_env.close()
        avg_reward = np.mean(rewards)
        self.eval_rewards.append(avg_reward)
        return avg_reward

    def train(self):
        """Full training loop."""
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        state, _ = self.env.reset()
        steps = 0
        best_eval = 0

        print(f"Training on {self.config.env_name}")
        print(f"Target reward: {self.config.target_reward}")
        print(f"Config: lr={self.config.lr}, γ={self.config.gamma}, "
              f"λ={self.config.lam}, rollout={self.config.rollout_length}")
        print("-" * 50)

        while steps < self.config.total_steps:
            state, next_value = self.collect_rollout(state)
            self.update(next_value)
            steps += self.config.rollout_length

            # Evaluation
            if steps % self.config.eval_interval < self.config.rollout_length:
                eval_reward = self.evaluate(self.config.eval_episodes)
                recent_avg = (np.mean(self.episode_rewards[-100:])
                              if self.episode_rewards else 0)
                print(f"Step {steps:>7d} | "
                      f"Train Avg: {recent_avg:>6.1f} | "
                      f"Eval: {eval_reward:>6.1f} | "
                      f"Episodes: {len(self.episode_rewards)}")

                if eval_reward > best_eval:
                    best_eval = eval_reward
                    self.save("best_model.pt")

                if eval_reward >= self.config.target_reward:
                    print(f"\nTarget reward {self.config.target_reward} reached!")
                    break

        self.env.close()
        print(f"\nTraining complete. Best eval reward: {best_eval:.1f}")

    def save(self, path: str):
        """Save model and config."""
        save_dir = "10_actor_critic"
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, path)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'episode_rewards': self.episode_rewards,
        }, full_path)

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, weights_only=False)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def plot_results(self):
        """Plot comprehensive training results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Episode rewards
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, alpha=0.3, color='steelblue')
        window = min(50, len(self.episode_rewards))
        if window > 0:
            smoothed = [np.mean(self.episode_rewards[max(0,i-window):i+1])
                        for i in range(len(self.episode_rewards))]
            ax.plot(smoothed, color='steelblue', linewidth=2, label=f'{window}-ep avg')
        ax.axhline(y=self.config.target_reward, color='red', linestyle='--',
                    alpha=0.5, label='Target')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Eval rewards
        ax = axes[0, 1]
        if self.eval_rewards:
            eval_steps = np.linspace(0, len(self.episode_rewards), len(self.eval_rewards))
            ax.plot(eval_steps, self.eval_rewards, 'o-', color='coral', linewidth=2)
            ax.axhline(y=self.config.target_reward, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Eval Reward')
        ax.set_title('Evaluation Rewards')
        ax.grid(True, alpha=0.3)

        # Losses
        ax = axes[1, 0]
        ax.plot(self.policy_losses, alpha=0.5, label='Policy Loss', color='steelblue')
        ax.plot(self.value_losses, alpha=0.5, label='Value Loss', color='coral')
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Entropy
        ax = axes[1, 1]
        ax.plot(self.entropies, alpha=0.7, color='green')
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Entropy')
        ax.set_title('Policy Entropy')
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'Actor-Critic on {self.config.env_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('10_actor_critic/cartpole_ac_results.png', dpi=150, bbox_inches='tight')
        print("Results saved to '10_actor_critic/cartpole_ac_results.png'")
        plt.close()


def main():
    print("\n" + "=" * 60)
    print("WEEK 10 - LESSON 5: COMPLETE CARTPOLE ACTOR-CRITIC")
    print("=" * 60)

    config = Config(
        lr=0.001,
        gamma=0.99,
        lam=0.95,
        rollout_length=128,
        total_steps=200000,
        eval_interval=10000,
        target_reward=475.0,
        seed=42
    )

    agent = CartPoleActorCritic(config)
    agent.train()
    agent.plot_results()

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_reward = agent.evaluate(n_episodes=20)
    print(f"Average reward over 20 episodes: {final_reward:.1f}")

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Orthogonal initialization improves training stability")
    print("2. GAE (λ=0.95) provides good bias-variance trade-off")
    print("3. Gradient clipping prevents training instabilities")
    print("4. Entropy bonus encourages exploration")
    print("5. Periodic evaluation with deterministic policy tracks true performance")
    print("6. Model checkpointing saves best performing model")
    print("\nNext: Actor-Critic for continuous action spaces!")
    print("=" * 60)


if __name__ == "__main__":
    main()
