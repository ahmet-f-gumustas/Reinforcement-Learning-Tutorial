"""
05 - PPO for Continuous Actions

PPO with Gaussian policy for continuous action spaces.
Trains on Pendulum-v1.

Demonstrates:
- Gaussian actor for continuous actions
- PPO-Clip with continuous policy
- Action squashing (tanh) to bound actions
- Shared and separate std parameterization
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
import gymnasium as gym


class ContinuousPPONet(nn.Module):
    """
    PPO network for continuous actions.

    Actor: outputs mean μ(s) of Gaussian policy
    Critic: outputs value V(s)
    log_std: learnable parameter (not state-dependent for simplicity)
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh()
        )
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))
        self.critic_head = nn.Linear(hidden_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, state):
        f = self.shared(state)
        mean = self.actor_mean(f)
        std = self.actor_log_std.exp()
        value = self.critic_head(f)
        return mean, std, value

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            mean, std, value = self.forward(torch.FloatTensor(state).unsqueeze(0))
        if deterministic:
            action = mean.squeeze(0)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample().squeeze(0)
        dist = torch.distributions.Normal(mean.squeeze(0), std)
        log_prob = dist.log_prob(action).sum().item()
        return action.numpy(), log_prob, value.squeeze().item()

    def evaluate(self, states, actions):
        mean, std, values = self.forward(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, values.squeeze(), entropy


def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    advantages, gae = [], 0
    vals = list(values) + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * vals[t+1] * (1-int(dones[t])) - vals[t]
        gae = delta + gamma * lam * (1-int(dones[t])) * gae
        advantages.insert(0, gae)
    adv = torch.FloatTensor(advantages)
    return adv, adv + torch.FloatTensor(values)


def train_ppo_continuous(env_name="Pendulum-v1", total_steps=300000,
                          rollout_length=2048, ppo_epochs=10,
                          clip_eps=0.2, lr=3e-4, gamma=0.99, lam=0.95,
                          entropy_coeff=0.0, value_coeff=0.5,
                          max_grad_norm=0.5) -> Tuple[List, List]:
    """Train PPO with Gaussian policy on continuous control."""
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_high = float(env.action_space.high[0])

    network = ContinuousPPONet(state_size, action_size)
    optimizer = optim.Adam(network.parameters(), lr=lr)

    episode_rewards, std_history = [], []
    state, _ = env.reset()
    current_reward = 0
    steps = 0

    while steps < total_steps:
        # Collect rollout
        buf = {'states': [], 'actions': [], 'rewards': [],
               'dones': [], 'log_probs': [], 'values': []}

        for _ in range(rollout_length):
            action, log_prob, value = network.get_action(state)
            clipped = np.clip(action, -action_high, action_high)
            next_state, reward, term, trunc, _ = env.step(clipped)
            done = term or trunc

            buf['states'].append(state)
            buf['actions'].append(action)
            buf['rewards'].append(reward)
            buf['dones'].append(done)
            buf['log_probs'].append(log_prob)
            buf['values'].append(value)

            current_reward += reward
            steps += 1
            if done:
                episode_rewards.append(current_reward)
                current_reward = 0
                state, _ = env.reset()
            else:
                state = next_state

        # Bootstrap
        with torch.no_grad():
            _, _, nv = network(torch.FloatTensor(state).unsqueeze(0))
            next_value = nv.squeeze().item()

        advantages, returns = compute_gae(
            buf['rewards'], buf['values'], next_value,
            buf['dones'], gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.FloatTensor(np.array(buf['states']))
        actions_t = torch.FloatTensor(np.array(buf['actions']))
        old_lp_t = torch.FloatTensor(buf['log_probs'])

        n = rollout_length
        minibatch = 256

        for _ in range(ppo_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, minibatch):
                mb = idx[start:start + minibatch]
                new_lp, new_vals, entropy = network.evaluate(
                    states_t[mb], actions_t[mb])

                ratio = torch.exp(new_lp - old_lp_t[mb])
                surr1 = ratio * advantages[mb]
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_vals, returns[mb].detach())
                total_loss = (policy_loss
                              + value_coeff * value_loss
                              - entropy_coeff * entropy.mean())

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
                optimizer.step()

        std_history.append(network.actor_log_std.exp().mean().item())

        if len(episode_rewards) > 0 and len(episode_rewards) % 30 == 0:
            avg = np.mean(episode_rewards[-30:])
            std = network.actor_log_std.exp().mean().item()
            print(f"Step {steps:>7d} | Ep: {len(episode_rewards):>4d} | "
                  f"Avg: {avg:>7.1f} | Std: {std:.3f}")

    env.close()
    return episode_rewards, std_history


def demonstrate_continuous_differences():
    """Explain key differences for continuous PPO."""
    print("=" * 60)
    print("PPO FOR CONTINUOUS ACTIONS")
    print("=" * 60)

    print("""
    Key adaptations for continuous action spaces:

    1. POLICY DISTRIBUTION
       Discrete:    Categorical(softmax(logits))
       Continuous:  Normal(μ(s), σ)

    2. LOG PROBABILITY
       Discrete:    log π(a|s) = logits[a] - log Σ exp(logits)
       Continuous:  log π(a|s) = -½[(a-μ)²/σ²] - log σ - ½log(2π)
                               (sum over action dimensions)

    3. IMPORTANCE RATIO
       Same formula: r = exp(log π_new - log π_old)
       But uses sum of log probs across action dimensions

    4. ACTION CLIPPING
       env.action_space.high/low defines bounds
       clip(action, -limit, limit) before env.step()

    5. STANDARD DEVIATION
       Starts high (exploration) → decreases as policy improves
       Learnable: log_std = nn.Parameter(torch.zeros(action_size))

    6. ENTROPY BONUS
       H[N(μ,σ)] = 0.5 * log(2πe σ²)
       Typically set entropy_coeff = 0.0 for continuous control
       (Gaussian naturally provides exploration via σ)
    """)


def main():
    print("\n" + "=" * 60)
    print("WEEK 11 - LESSON 5: PPO FOR CONTINUOUS ACTIONS")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    demonstrate_continuous_differences()

    print("\n" + "=" * 60)
    print("TRAINING PPO ON PENDULUM-V1")
    print("=" * 60)
    print("Pendulum: [cos θ, sin θ, θ_dot] → torque ∈ [-2, 2]")
    print("Reward: -(θ² + 0.1 θ_dot² + 0.001 a²), optimal ≈ -200")
    print()

    rewards, stds = train_ppo_continuous(
        env_name="Pendulum-v1",
        total_steps=300000,
        rollout_length=2048,
        ppo_epochs=10,
        clip_eps=0.2,
        lr=3e-4
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(rewards, alpha=0.3, color='steelblue')
    w = min(30, len(rewards))
    sm = [np.mean(rewards[max(0,i-w):i+1]) for i in range(len(rewards))]
    ax.plot(sm, color='steelblue', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(stds, color='coral', linewidth=2)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Action Std')
    ax.set_title('Learned Standard Deviation')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if len(rewards) > 20:
        ax.hist(rewards[-100:], bins=20, color='green', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Count')
    ax.set_title('Final Reward Distribution')
    ax.grid(True, alpha=0.3)

    plt.suptitle('PPO on Pendulum-v1 (Continuous)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('11_ppo/ppo_continuous.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to '11_ppo/ppo_continuous.png'")
    plt.close()

    if rewards:
        print(f"\nFinal Performance (last 50 episodes): "
              f"{np.mean(rewards[-50:]):.1f} ± {np.std(rewards[-50:]):.1f}")

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Gaussian policy: π(a|s) = N(μ(s), σ)")
    print("2. Log prob summed over action dimensions")
    print("3. Importance ratio formula is the same as discrete case")
    print("4. Std starts high and decreases as policy improves")
    print("5. Entropy coeff often set to 0 for continuous control")
    print("\nNext: PPO on LunarLander - a harder challenge!")
    print("=" * 60)


if __name__ == "__main__":
    main()
