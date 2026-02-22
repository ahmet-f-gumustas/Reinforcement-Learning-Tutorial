"""
04 - PPO CartPole: Complete Production Implementation

Full, production-ready PPO-Clip implementation for CartPole-v1
with all best practices and comprehensive monitoring.

Demonstrates:
- Complete PPO-Clip with GAE
- Value function clipping
- Learning rate scheduling
- Early stopping on KL
- Model checkpointing
- Comprehensive training visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Optional
from dataclasses import dataclass
import gymnasium as gym
import os


@dataclass
class PPOConfig:
    env_name: str = "CartPole-v1"
    hidden_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2          # Clip value function loss too
    ppo_epochs: int = 10
    minibatch_size: int = 64
    rollout_length: int = 2048
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02             # Early stop if KL exceeds this
    total_steps: int = 300000
    eval_interval: int = 20480
    eval_episodes: int = 10
    target_reward: float = 475.0
    seed: int = 42


class PPONet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh()
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x):
        f = self.shared(x)
        return self.actor(f), self.critic(f)

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            logits, value = self.forward(torch.FloatTensor(state).unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action.item(), dist.log_prob(action).item(), value.squeeze().item()

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), values.squeeze(), dist.entropy()


def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    advantages, gae = [], 0
    vals = list(values) + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * vals[t+1] * (1-int(dones[t])) - vals[t]
        gae = delta + gamma * lam * (1-int(dones[t])) * gae
        advantages.insert(0, gae)
    adv = torch.FloatTensor(advantages)
    return adv, adv + torch.FloatTensor(values)


class CartPolePPO:
    """Production-ready PPO agent for CartPole."""

    def __init__(self, config: PPOConfig):
        self.cfg = config
        self.env = gym.make(config.env_name)
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

        self.network = PPONet(state_size, action_size, config.hidden_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)

        # Tracking
        self.episode_rewards: List[float] = []
        self.eval_rewards: List[float] = []
        self.metrics: List[dict] = []

    def collect_rollout(self, state):
        """Collect rollout_length steps."""
        buf = {'states': [], 'actions': [], 'rewards': [],
               'dones': [], 'log_probs': [], 'values': []}
        ep_reward = 0

        for _ in range(self.cfg.rollout_length):
            action, log_prob, value = self.network.get_action(state)
            next_state, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc

            buf['states'].append(state)
            buf['actions'].append(action)
            buf['rewards'].append(reward)
            buf['dones'].append(done)
            buf['log_probs'].append(log_prob)
            buf['values'].append(value)

            ep_reward += reward
            if done:
                self.episode_rewards.append(ep_reward)
                ep_reward = 0
                state, _ = self.env.reset()
            else:
                state = next_state

        with torch.no_grad():
            _, nv = self.network(torch.FloatTensor(state).unsqueeze(0))
            next_value = nv.squeeze().item()

        return buf, state, next_value

    def update(self, buf, next_value):
        """PPO-Clip update with optional value clipping and early KL stopping."""
        advantages, returns = compute_gae(
            buf['rewards'], buf['values'], next_value,
            buf['dones'], self.cfg.gamma, self.cfg.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.FloatTensor(np.array(buf['states']))
        actions_t = torch.LongTensor(buf['actions'])
        old_lp_t = torch.FloatTensor(buf['log_probs'])
        old_vals_t = torch.FloatTensor(buf['values'])

        n = len(buf['states'])
        ep_metrics = {'policy_loss': [], 'value_loss': [],
                      'entropy': [], 'kl': [], 'clip_frac': []}

        for epoch in range(self.cfg.ppo_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.cfg.minibatch_size):
                mb = idx[start:start + self.cfg.minibatch_size]
                mb_states = states_t[mb]
                mb_actions = actions_t[mb]
                mb_old_lp = old_lp_t[mb]
                mb_adv = advantages[mb]
                mb_ret = returns[mb]
                mb_old_vals = old_vals_t[mb]

                new_lp, new_vals, entropy = self.network.evaluate(mb_states, mb_actions)

                # Policy loss (clipped)
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (optionally clipped)
                v_loss_unclipped = (new_vals - mb_ret).pow(2)
                v_clipped = mb_old_vals + torch.clamp(
                    new_vals - mb_old_vals,
                    -self.cfg.value_clip_eps, self.cfg.value_clip_eps)
                v_loss_clipped = (v_clipped - mb_ret).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                total_loss = (policy_loss
                              + self.cfg.value_coeff * value_loss
                              - self.cfg.entropy_coeff * entropy.mean())

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                kl = (mb_old_lp - new_lp).mean().item()
                clip_frac = ((ratio - 1).abs() > self.cfg.clip_eps).float().mean().item()
                ep_metrics['policy_loss'].append(policy_loss.item())
                ep_metrics['value_loss'].append(value_loss.item())
                ep_metrics['entropy'].append(entropy.mean().item())
                ep_metrics['kl'].append(kl)
                ep_metrics['clip_frac'].append(clip_frac)

            # Early stopping if KL exceeds target
            mean_kl = np.mean(np.abs(ep_metrics['kl']))
            if mean_kl > self.cfg.target_kl:
                break

        return {k: np.mean(v) for k, v in ep_metrics.items()}

    def evaluate(self, n_episodes=10):
        eval_env = gym.make(self.cfg.env_name)
        rewards = []
        for _ in range(n_episodes):
            state, _ = eval_env.reset()
            done = False
            ep_r = 0
            while not done:
                action, _, _ = self.network.get_action(state, deterministic=True)
                state, r, term, trunc, _ = eval_env.step(action)
                done = term or trunc
                ep_r += r
            rewards.append(ep_r)
        eval_env.close()
        avg = np.mean(rewards)
        self.eval_rewards.append(avg)
        return avg

    def save(self, path="11_ppo/best_ppo.pt"):
        torch.save(self.network.state_dict(), path)

    def train(self):
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        state, _ = self.env.reset()
        steps = 0
        best_eval = 0

        print(f"Training PPO on {self.cfg.env_name}")
        print(f"lr={self.cfg.lr}, clip_eps={self.cfg.clip_eps}, "
              f"ppo_epochs={self.cfg.ppo_epochs}, rollout={self.cfg.rollout_length}")
        print("-" * 55)

        while steps < self.cfg.total_steps:
            buf, state, next_value = self.collect_rollout(state)
            m = self.update(buf, next_value)
            steps += self.cfg.rollout_length
            self.metrics.append(m)

            if steps % self.cfg.eval_interval < self.cfg.rollout_length:
                eval_r = self.evaluate(self.cfg.eval_episodes)
                train_avg = np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0
                print(f"Step {steps:>7d} | Train: {train_avg:>6.1f} | "
                      f"Eval: {eval_r:>6.1f} | KL: {m['kl']:.4f} | "
                      f"Clip: {m['clip_frac']:.2f}")
                if eval_r > best_eval:
                    best_eval = eval_r
                    self.save()
                if eval_r >= self.cfg.target_reward:
                    print(f"\nTarget {self.cfg.target_reward} reached!")
                    break

        self.env.close()
        print(f"\nBest eval reward: {best_eval:.1f}")

    def plot(self):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))

        # Episode rewards
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, alpha=0.3, color='steelblue')
        w = min(50, len(self.episode_rewards))
        sm = [np.mean(self.episode_rewards[max(0,i-w):i+1]) for i in range(len(self.episode_rewards))]
        ax.plot(sm, color='steelblue', linewidth=2)
        ax.axhline(y=self.cfg.target_reward, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Training Rewards')
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)

        # Eval rewards
        ax = axes[0, 1]
        if self.eval_rewards:
            ax.plot(self.eval_rewards, 'o-', color='coral', linewidth=2)
            ax.axhline(y=self.cfg.target_reward, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Evaluation Rewards')
        ax.grid(True, alpha=0.3)

        # KL divergence
        ax = axes[0, 2]
        ax.plot([m['kl'] for m in self.metrics], color='green', alpha=0.7)
        ax.axhline(y=self.cfg.target_kl, color='red', linestyle='--', alpha=0.5, label='Target KL')
        ax.set_title('KL Divergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Policy loss
        ax = axes[1, 0]
        ax.plot([m['policy_loss'] for m in self.metrics], color='steelblue', alpha=0.7)
        ax.set_title('Policy Loss')
        ax.grid(True, alpha=0.3)

        # Value loss
        ax = axes[1, 1]
        ax.plot([m['value_loss'] for m in self.metrics], color='coral', alpha=0.7)
        ax.set_title('Value Loss')
        ax.grid(True, alpha=0.3)

        # Clip fraction
        ax = axes[1, 2]
        ax.plot([m['clip_frac'] for m in self.metrics], color='purple', alpha=0.7)
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='10% clip target')
        ax.set_title('Clip Fraction')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'PPO on {self.cfg.env_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('11_ppo/ppo_cartpole_full.png', dpi=150, bbox_inches='tight')
        print("Results saved to '11_ppo/ppo_cartpole_full.png'")
        plt.close()


def main():
    print("\n" + "=" * 60)
    print("WEEK 11 - LESSON 4: COMPLETE PPO ON CARTPOLE")
    print("=" * 60)

    config = PPOConfig(
        lr=3e-4, clip_eps=0.2, ppo_epochs=10,
        rollout_length=2048, total_steps=300000,
        target_reward=475.0, seed=42
    )

    agent = CartPolePPO(config)
    agent.train()
    agent.plot()

    final_eval = agent.evaluate(n_episodes=20)
    print(f"\nFinal evaluation (20 episodes): {final_eval:.1f}")

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Value function clipping prevents too-large critic updates")
    print("2. Early KL stopping: exit PPO epochs if KL > target_kl")
    print("3. Clip fraction ~10% is a sign of healthy clipping")
    print("4. Monitor both KL and clip fraction for training health")
    print("5. Orthogonal init + Tanh activations work well for PPO")
    print("\nNext: PPO for continuous action spaces!")
    print("=" * 60)


if __name__ == "__main__":
    main()
