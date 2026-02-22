"""
02 - PPO-Clip: Proximal Policy Optimization with Clipping

This script implements PPO-Clip, the most popular variant of PPO.
Instead of the expensive constrained optimization of TRPO,
PPO clips the importance sampling ratio to stay within [1-ε, 1+ε].

Demonstrates:
- PPO-Clip objective function
- Clipping mechanism visualization
- Multiple epochs on the same data
- PPO training on CartPole-v1
- Effect of clip epsilon hyperparameter
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, NamedTuple
import gymnasium as gym


# =============================================================================
# Network
# =============================================================================

class PPONetwork(nn.Module):
    """Shared Actor-Critic network for PPO."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.actor_head = nn.Linear(hidden_size, action_size)
        self.critic_head = nn.Linear(hidden_size, 1)

        # Orthogonal initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.shared(state)
        return self.actor_head(f), self.critic_head(f)

    def get_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.forward(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.squeeze().item()

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(), entropy


# =============================================================================
# PPO-Clip Core
# =============================================================================

def ppo_clip_loss(log_probs_new: torch.Tensor,
                  log_probs_old: torch.Tensor,
                  advantages: torch.Tensor,
                  clip_eps: float = 0.2) -> torch.Tensor:
    """
    PPO-Clip objective:

        L^CLIP(θ) = E_t[ min(r_t(θ) × A_t,  clip(r_t(θ), 1-ε, 1+ε) × A_t) ]

    where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

    Intuition:
    - When A > 0 (good action): r is clipped at 1+ε → don't increase prob too much
    - When A < 0 (bad action): r is clipped at 1-ε → don't decrease prob too much
    - The min() prevents overly optimistic updates in both directions

    Args:
        log_probs_new: log π_new(a|s)
        log_probs_old: log π_old(a|s) (stored, no gradient)
        advantages: A_t (normalized)
        clip_eps: Clipping parameter ε (typically 0.1 or 0.2)

    Returns:
        PPO-Clip loss (to be minimized, so negated)
    """
    # Importance sampling ratio
    ratios = torch.exp(log_probs_new - log_probs_old)

    # Unclipped objective
    surr1 = ratios * advantages

    # Clipped objective
    surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages

    # Take minimum (pessimistic bound)
    return -torch.min(surr1, surr2).mean()


def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """Compute GAE advantages."""
    advantages = []
    gae = 0
    vals = list(values) + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * vals[t+1] * (1 - int(dones[t])) - vals[t]
        gae = delta + gamma * lam * (1 - int(dones[t])) * gae
        advantages.insert(0, gae)
    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(values)
    return advantages, returns


# =============================================================================
# PPO-Clip Agent
# =============================================================================

class RolloutBuffer:
    """Stores a rollout of experience."""

    def __init__(self):
        self.states, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values = [], [], []

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


class PPOClipAgent:
    """
    PPO-Clip Agent.

    Key features:
    1. Collect rollout_length steps of experience
    2. Compute GAE advantages
    3. Run ppo_epochs epochs over the same data (minibatches)
    4. Clip importance ratio to [1-ε, 1+ε]
    """

    def __init__(self, state_size: int, action_size: int,
                 lr: float = 3e-4, gamma: float = 0.99,
                 lam: float = 0.95, clip_eps: float = 0.2,
                 ppo_epochs: int = 10, minibatch_size: int = 64,
                 entropy_coeff: float = 0.01, value_coeff: float = 0.5,
                 max_grad_norm: float = 0.5):

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm

        self.network = PPONetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def select_action(self, state: np.ndarray) -> int:
        action, log_prob, value = self.network.get_action(state)
        self.buffer.add(state, action, None, None, log_prob, value)
        return action

    def store_outcome(self, reward: float, done: bool):
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)

    def update(self, next_state: np.ndarray, done: bool) -> dict:
        """
        PPO update:
        1. Compute GAE advantages
        2. Run ppo_epochs of minibatch gradient steps
        3. Clip importance ratio to prevent large updates
        """
        with torch.no_grad():
            _, nv = self.network(torch.FloatTensor(next_state).unsqueeze(0))
            next_value = nv.squeeze().item()

        advantages, returns = compute_gae(
            self.buffer.rewards, self.buffer.values,
            next_value, self.buffer.dones, self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.FloatTensor(np.array(self.buffer.states))
        actions_t = torch.LongTensor(self.buffer.actions)
        old_log_probs_t = torch.FloatTensor(self.buffer.log_probs)

        # Multiple epochs over the same data
        metrics = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'approx_kl': []}
        n = len(self.buffer)

        for _ in range(self.ppo_epochs):
            # Shuffle indices for minibatch
            indices = np.random.permutation(n)
            for start in range(0, n, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]

                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                new_log_probs, values, entropy = self.network.evaluate(mb_states, mb_actions)

                # PPO-Clip loss
                policy_loss = ppo_clip_loss(
                    new_log_probs, mb_old_log_probs, mb_advantages, self.clip_eps)

                # Value loss (clipped version optional)
                value_loss = F.mse_loss(values, mb_returns.detach())

                # Total loss
                total_loss = (policy_loss
                              + self.value_coeff * value_loss
                              - self.entropy_coeff * entropy.mean())

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Approximate KL for monitoring
                approx_kl = (mb_old_log_probs - new_log_probs.detach()).mean().item()
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(entropy.mean().item())
                metrics['approx_kl'].append(approx_kl)

        self.buffer.clear()
        return {k: np.mean(v) for k, v in metrics.items()}


# =============================================================================
# Training
# =============================================================================

def train_ppo_clip(env_name: str = "CartPole-v1",
                   total_steps: int = 200000,
                   rollout_length: int = 2048,
                   ppo_epochs: int = 10,
                   clip_eps: float = 0.2,
                   lr: float = 3e-4) -> Tuple[List, dict]:
    """Train PPO-Clip agent."""
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = PPOClipAgent(
        state_size, action_size, lr=lr,
        clip_eps=clip_eps, ppo_epochs=ppo_epochs,
        rollout_length=rollout_length
    )

    episode_rewards = []
    update_metrics = []
    state, _ = env.reset()
    current_reward = 0
    steps = 0

    while steps < total_steps:
        # Collect rollout
        for _ in range(rollout_length):
            action = agent.select_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.store_outcome(reward, done)
            current_reward += reward
            steps += 1

            if done:
                episode_rewards.append(current_reward)
                current_reward = 0
                state, _ = env.reset()
            else:
                state = next_state

        # PPO update
        metrics = agent.update(state, False)
        update_metrics.append(metrics)

        if len(episode_rewards) > 0 and len(episode_rewards) % 20 == 0:
            avg = np.mean(episode_rewards[-20:])
            kl = metrics['approx_kl']
            print(f"Step {steps:>7d} | Episodes: {len(episode_rewards):>4d} | "
                  f"Avg Reward: {avg:>6.1f} | KL: {kl:.4f}")

    env.close()
    return episode_rewards, update_metrics


# =============================================================================
# Demonstrations
# =============================================================================

def visualize_clip_objective():
    """Visualize the PPO-Clip objective."""
    print("=" * 60)
    print("PPO-CLIP OBJECTIVE VISUALIZATION")
    print("=" * 60)

    print("""
    L^CLIP(θ) = E_t[ min(r_t × A_t,  clip(r_t, 1-ε, 1+ε) × A_t) ]

    where r_t = π_new(a|s) / π_old(a|s)

    The clip prevents the ratio from straying too far from 1:
    - If A > 0: want to increase π_new, but cap at (1+ε)
    - If A < 0: want to decrease π_new, but floor at (1-ε)
    """)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ratios = np.linspace(0, 2.5, 300)
    clip_eps = 0.2

    for ax, (advantage, title) in zip(axes, [(1.0, "A > 0 (Good Action)"),
                                              (-1.0, "A < 0 (Bad Action)")]):
        surr1 = ratios * advantage
        surr2 = np.clip(ratios, 1 - clip_eps, 1 + clip_eps) * advantage
        objective = np.minimum(surr1, surr2)

        ax.plot(ratios, surr1, 'b--', linewidth=2, label='r × A (unclipped)', alpha=0.8)
        ax.plot(ratios, surr2, 'g--', linewidth=2, label='clip(r, 1-ε, 1+ε) × A', alpha=0.8)
        ax.plot(ratios, objective, 'r-', linewidth=3, label='L^CLIP (min of both)')

        ax.axvline(x=1 - clip_eps, color='gray', linestyle=':', alpha=0.6)
        ax.axvline(x=1 + clip_eps, color='gray', linestyle=':', alpha=0.6, label=f'ε = {clip_eps}')
        ax.axvline(x=1.0, color='black', linestyle='-', alpha=0.3)

        ax.set_xlabel('Ratio r = π_new / π_old')
        ax.set_ylabel('Objective')
        ax.set_title(f'PPO-Clip: {title}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('11_ppo/ppo_clip_objective.png', dpi=150, bbox_inches='tight')
    print("Plot saved to '11_ppo/ppo_clip_objective.png'")
    plt.close()


def main():
    print("\n" + "=" * 60)
    print("WEEK 11 - LESSON 2: PPO-CLIP")
    print("Proximal Policy Optimization with Clipping")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Visualize clip objective
    visualize_clip_objective()

    # 2. Train PPO-Clip
    print("\n" + "=" * 60)
    print("TRAINING PPO-CLIP ON CARTPOLE-V1")
    print("=" * 60)

    rewards, metrics = train_ppo_clip(
        env_name="CartPole-v1",
        total_steps=200000,
        rollout_length=2048,
        ppo_epochs=10,
        clip_eps=0.2,
        lr=3e-4
    )

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    window = 30
    smoothed = [np.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
    axes[0,0].plot(rewards, alpha=0.3, color='steelblue')
    axes[0,0].plot(smoothed, color='steelblue', linewidth=2)
    axes[0,0].axhline(y=500, color='red', linestyle='--', alpha=0.5, label='Max')
    axes[0,0].set_title('Episode Rewards')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    update_steps = range(len(metrics))
    axes[0,1].plot([m['policy_loss'] for m in metrics], color='coral', alpha=0.7)
    axes[0,1].set_title('Policy Loss')
    axes[0,1].set_xlabel('Update')
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].plot([m['approx_kl'] for m in metrics], color='green', alpha=0.7)
    axes[1,0].axhline(y=0.02, color='red', linestyle='--', alpha=0.5, label='Target KL')
    axes[1,0].set_title('Approx KL Divergence')
    axes[1,0].set_xlabel('Update')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot([m['entropy'] for m in metrics], color='purple', alpha=0.7)
    axes[1,1].set_title('Policy Entropy')
    axes[1,1].set_xlabel('Update')
    axes[1,1].grid(True, alpha=0.3)

    plt.suptitle('PPO-Clip on CartPole-v1', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('11_ppo/ppo_clip_training.png', dpi=150, bbox_inches='tight')
    print("Plot saved to '11_ppo/ppo_clip_training.png'")
    plt.close()

    print(f"\nFinal Performance (last 50 episodes): "
          f"{np.mean(rewards[-50:]):.1f} ± {np.std(rewards[-50:]):.1f}")

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. PPO-Clip clips the importance ratio to [1-ε, 1+ε]")
    print("2. min() takes the pessimistic bound → safe update")
    print("3. Multiple epochs (ppo_epochs) reuse the same collected data")
    print("4. ε = 0.2 is the standard default clip value")
    print("5. Monitor KL divergence: if it exceeds ~0.02, reduce lr or ε")
    print("\nNext: PPO-Penalty variant with adaptive KL!")
    print("=" * 60)


if __name__ == "__main__":
    main()
