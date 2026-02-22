"""
03 - PPO-Penalty: PPO with Adaptive KL Penalty

This script implements the PPO-Penalty variant, which uses a KL divergence
penalty term instead of hard clipping. The penalty coefficient is adapted
automatically based on observed KL.

Demonstrates:
- PPO-Penalty objective with KL constraint
- Adaptive KL coefficient (beta)
- Comparison: PPO-Clip vs PPO-Penalty
- When each variant is preferred
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
import gymnasium as gym


class PPONetwork(nn.Module):
    """Shared Actor-Critic network."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh()
        )
        self.actor_head = nn.Linear(hidden_size, action_size)
        self.critic_head = nn.Linear(hidden_size, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, state):
        f = self.shared(state)
        return self.actor_head(f), self.critic_head(f)

    def get_action(self, state):
        with torch.no_grad():
            logits, value = self.forward(torch.FloatTensor(state).unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
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
    advantages = torch.FloatTensor(advantages)
    return advantages, advantages + torch.FloatTensor(values)


class PPOPenaltyAgent:
    """
    PPO-Penalty Agent.

    Objective:
        L^KL(θ) = E_t[r_t(θ) × A_t] - β × KL(π_old || π_new)

    Adaptive β:
        - If KL > 1.5 × d_target: β *= 2   (KL too large, increase penalty)
        - If KL < d_target / 1.5: β /= 2   (KL too small, decrease penalty)

    This automatically adjusts the KL penalty to keep updates in a safe range.
    """

    def __init__(self, state_size, action_size,
                 lr=3e-4, gamma=0.99, lam=0.95,
                 init_beta=1.0, target_kl=0.01,
                 ppo_epochs=10, minibatch_size=64,
                 value_coeff=0.5, max_grad_norm=0.5):
        self.gamma = gamma
        self.lam = lam
        self.beta = init_beta        # KL penalty coefficient
        self.target_kl = target_kl  # Target KL divergence
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm

        self.network = PPONetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Buffers
        self.states, self.actions = [], []
        self.rewards, self.dones = [], []
        self.log_probs, self.values = [], []
        self.betas = []  # Track beta evolution

    def select_action(self, state):
        action, log_prob, value = self.network.get_action(state)
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        return action

    def store(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def update(self, next_state, done):
        """
        PPO-Penalty update with adaptive beta.

        After each set of epochs:
        - Compute observed KL
        - Adjust beta based on target_kl
        """
        with torch.no_grad():
            _, nv = self.network(torch.FloatTensor(next_state).unsqueeze(0))
            next_value = nv.squeeze().item()

        advantages, returns = compute_gae(
            self.rewards, self.values, next_value, self.dones,
            self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.FloatTensor(np.array(self.states))
        actions_t = torch.LongTensor(self.actions)
        old_log_probs_t = torch.FloatTensor(self.log_probs)

        n = len(self.states)
        kl_list = []

        for _ in range(self.ppo_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.minibatch_size):
                mb_idx = idx[start:start + self.minibatch_size]
                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_lp = old_log_probs_t[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                new_lp, values, entropy = self.network.evaluate(mb_states, mb_actions)

                # Importance ratio
                ratios = torch.exp(new_lp - mb_old_lp)

                # Surrogate objective (unclipped)
                surr = (ratios * mb_adv).mean()

                # KL penalty
                kl = (mb_old_lp - new_lp).mean()  # Approx KL
                kl_list.append(kl.item())

                # Combined objective
                policy_loss = -(surr - self.beta * kl)
                value_loss = F.mse_loss(values, mb_ret.detach())
                total_loss = policy_loss + self.value_coeff * value_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # Adaptive beta update
        mean_kl = np.mean(np.abs(kl_list))
        if mean_kl < self.target_kl / 1.5:
            self.beta /= 2    # KL too small → reduce penalty
        elif mean_kl > self.target_kl * 1.5:
            self.beta *= 2    # KL too large → increase penalty
        self.beta = np.clip(self.beta, 1e-4, 100.0)
        self.betas.append(self.beta)

        # Clear buffers
        self.states.clear(); self.actions.clear()
        self.rewards.clear(); self.dones.clear()
        self.log_probs.clear(); self.values.clear()

        return {'kl': mean_kl, 'beta': self.beta}


def train_ppo_penalty(env_name="CartPole-v1", total_steps=200000,
                      rollout_length=2048, target_kl=0.01) -> Tuple[List, List, List]:
    env = gym.make(env_name)
    agent = PPOPenaltyAgent(
        env.observation_space.shape[0], env.action_space.n,
        target_kl=target_kl, rollout_length=rollout_length
    )

    episode_rewards, kl_history, beta_history = [], [], []
    state, _ = env.reset()
    current_reward = 0
    steps = 0

    while steps < total_steps:
        for _ in range(rollout_length):
            action = agent.select_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.store(reward, done)
            current_reward += reward
            steps += 1
            if done:
                episode_rewards.append(current_reward)
                current_reward = 0
                state, _ = env.reset()
            else:
                state = next_state

        metrics = agent.update(state, False)
        kl_history.append(metrics['kl'])
        beta_history.append(metrics['beta'])

        if len(episode_rewards) > 0 and len(episode_rewards) % 20 == 0:
            print(f"Step {steps:>7d} | Episodes: {len(episode_rewards):>4d} | "
                  f"Avg: {np.mean(episode_rewards[-20:]):>6.1f} | "
                  f"β: {metrics['beta']:.4f} | KL: {metrics['kl']:.4f}")

    env.close()
    return episode_rewards, kl_history, beta_history


def compare_clip_vs_penalty():
    """Compare PPO-Clip and PPO-Penalty."""
    print("\n" + "=" * 60)
    print("PPO-CLIP vs PPO-PENALTY")
    print("=" * 60)

    print("""
    PPO-Clip:
        L^CLIP = E_t[ min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t) ]

        Pros:
        ✓ Simple to implement
        ✓ No hyperparameter tuning for constraint
        ✓ More robust in practice
        ✓ Default choice for most applications

        Cons:
        ✗ Clipping is a heuristic (no theoretical bound on KL)
        ✗ ε must be tuned

    PPO-Penalty:
        L^KL  = E_t[ r_t A_t ] - β × KL(π_old || π_new)
        β adapts: if KL > 1.5×δ: β×=2, if KL < δ/1.5: β/=2

        Pros:
        ✓ Explicit control over KL divergence
        ✓ Adaptive (no fixed ε hyperparameter)
        ✓ Better understood theoretically

        Cons:
        ✗ More complex, slower adaptation
        ✗ Sensitive to initial beta
        ✗ In practice, performs similar to PPO-Clip

    Recommendation: Use PPO-Clip as default.
    PPO-Penalty is useful when you need explicit KL control.
    """)


def demonstrate_adaptive_beta():
    """Visualize how beta adapts during training."""
    print("=" * 60)
    print("PPO-PENALTY: ADAPTIVE BETA")
    print("=" * 60)

    print("""
    Beta Adaptation Rule:
        After each update:
        if KL < target_kl / 1.5:  beta /= 2  (updates too conservative)
        if KL > target_kl * 1.5:  beta *= 2  (updates too large)

    Example with target_kl = 0.01:
    """)

    beta = 1.0
    target_kl = 0.01
    simulated_kls = [0.02, 0.025, 0.015, 0.008, 0.005, 0.003, 0.01, 0.012]
    betas = [beta]

    print(f"  Initial β = {beta:.4f}, target KL = {target_kl}")
    for kl in simulated_kls:
        if kl < target_kl / 1.5:
            action = "decrease"
            beta /= 2
        elif kl > target_kl * 1.5:
            action = "increase"
            beta *= 2
        else:
            action = "keep"
        betas.append(beta)
        print(f"  KL = {kl:.4f} → {action:8s} β → β = {beta:.4f}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(simulated_kls, 'o-', color='coral', linewidth=2, label='Observed KL')
    plt.axhline(y=target_kl, color='gray', linestyle='--', label='Target KL')
    plt.axhline(y=target_kl * 1.5, color='red', linestyle=':', alpha=0.5, label='Upper bound')
    plt.axhline(y=target_kl / 1.5, color='green', linestyle=':', alpha=0.5, label='Lower bound')
    plt.xlabel('Update Step')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence Over Time')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(betas, 's-', color='steelblue', linewidth=2)
    plt.xlabel('Update Step')
    plt.ylabel('Beta (β)')
    plt.title('Adaptive Beta')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('11_ppo/ppo_penalty_beta.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to '11_ppo/ppo_penalty_beta.png'")
    plt.close()


def main():
    print("\n" + "=" * 60)
    print("WEEK 11 - LESSON 3: PPO-PENALTY")
    print("Proximal Policy Optimization with KL Penalty")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Compare variants
    compare_clip_vs_penalty()

    # 2. Show adaptive beta
    demonstrate_adaptive_beta()

    # 3. Train PPO-Penalty
    print("\n" + "=" * 60)
    print("TRAINING PPO-PENALTY ON CARTPOLE-V1")
    print("=" * 60)

    rewards, kl_history, beta_history = train_ppo_penalty(
        env_name="CartPole-v1", total_steps=200000, target_kl=0.01)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    window = 30
    smoothed = [np.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
    axes[0].plot(rewards, alpha=0.3, color='steelblue')
    axes[0].plot(smoothed, color='steelblue', linewidth=2)
    axes[0].axhline(y=500, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(kl_history, color='coral', alpha=0.8)
    axes[1].axhline(y=0.01, color='gray', linestyle='--', label='Target KL')
    axes[1].set_title('KL Divergence')
    axes[1].set_xlabel('Update')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(beta_history, color='green', alpha=0.8)
    axes[2].set_title('Adaptive Beta')
    axes[2].set_xlabel('Update')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('PPO-Penalty on CartPole-v1', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('11_ppo/ppo_penalty_training.png', dpi=150, bbox_inches='tight')
    print("Plot saved to '11_ppo/ppo_penalty_training.png'")
    plt.close()

    print(f"\nFinal Performance (last 50 episodes): "
          f"{np.mean(rewards[-50:]):.1f} ± {np.std(rewards[-50:]):.1f}")

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. PPO-Penalty adds KL divergence as a soft constraint (penalty)")
    print("2. Beta is adapted automatically based on observed KL vs target")
    print("3. If KL too large → increase beta, if too small → decrease beta")
    print("4. Provides explicit KL control but is more complex than PPO-Clip")
    print("5. In practice, PPO-Clip is preferred for simplicity")
    print("\nNext: Full production PPO on CartPole!")
    print("=" * 60)


if __name__ == "__main__":
    main()
