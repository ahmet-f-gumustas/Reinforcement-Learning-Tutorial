"""
06 - PPO LunarLander: Full Challenge

PPO-Clip solving LunarLander-v2 - a more challenging environment
that requires careful hyperparameter tuning.

Demonstrates:
- PPO on a harder environment
- Hyperparameter sensitivity
- Reward shaping analysis
- Performance benchmarking
- Comparison across random seeds
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict
import gymnasium as gym


class LunarPPONet(nn.Module):
    """Larger network for LunarLander."""

    def __init__(self, state_size=8, action_size=4, hidden_size=256):
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


def train_lunarlander_ppo(total_steps=1_000_000, rollout_length=2048,
                           ppo_epochs=10, clip_eps=0.2, lr=3e-4,
                           gamma=0.999, lam=0.98, entropy_coeff=0.01,
                           value_coeff=0.5, max_grad_norm=0.5,
                           seed=42) -> List[float]:
    """Train PPO on LunarLander-v2."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("LunarLander-v2")
    network = LunarPPONet()
    optimizer = optim.Adam(network.parameters(), lr=lr, eps=1e-5)

    episode_rewards = []
    state, _ = env.reset(seed=seed)
    current_reward = 0
    steps = 0
    minibatch = 64

    while steps < total_steps:
        buf = {'states': [], 'actions': [], 'rewards': [],
               'dones': [], 'log_probs': [], 'values': []}

        for _ in range(rollout_length):
            action, log_prob, value = network.get_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
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
            _, nv = network(torch.FloatTensor(state).unsqueeze(0))
            next_value = nv.squeeze().item()

        advantages, returns = compute_gae(
            buf['rewards'], buf['values'], next_value,
            buf['dones'], gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.FloatTensor(np.array(buf['states']))
        actions_t = torch.LongTensor(buf['actions'])
        old_lp_t = torch.FloatTensor(buf['log_probs'])
        n = rollout_length

        for _ in range(ppo_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, minibatch):
                mb = idx[start:start + minibatch]
                new_lp, new_vals, entropy = network.evaluate(states_t[mb], actions_t[mb])
                ratio = torch.exp(new_lp - old_lp_t[mb])
                surr1 = ratio * advantages[mb]
                surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages[mb]
                p_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(new_vals, returns[mb].detach())
                loss = p_loss + value_coeff * v_loss - entropy_coeff * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
                optimizer.step()

        if len(episode_rewards) > 0 and len(episode_rewards) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"Step {steps:>8d} | Episodes: {len(episode_rewards):>4d} | "
                  f"Avg Reward: {avg:>7.1f}")
            if avg >= 200:
                print("Solved! (avg >= 200)")
                break

    env.close()
    return episode_rewards


def describe_lunarlander():
    """Describe the LunarLander environment."""
    print("=" * 60)
    print("LUNARLANDER-V2 ENVIRONMENT")
    print("=" * 60)
    print("""
    Goal: Land a spacecraft between two flags.

    State (8 dimensions):
        [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]

    Actions (4 discrete):
        0: Do nothing
        1: Fire left engine
        2: Fire main engine
        3: Fire right engine

    Rewards:
        + Moving toward landing pad
        + Reduced speed
        + Staying upright
        + Leg contact: +10 each
        + Landing: +100
        - Fire main engine: -0.3 per frame
        - Crash: -100

    Solved: Average score ≥ 200 over 100 episodes.

    PPO Hyperparameter Tips for LunarLander:
        lr = 3e-4         (standard)
        gamma = 0.999     (higher than CartPole, longer horizon)
        lam = 0.98        (higher lambda for less bias)
        clip_eps = 0.2    (standard)
        hidden_size = 256 (larger network than CartPole)
        entropy_coeff = 0.01  (important for exploration!)
    """)


def main():
    print("\n" + "=" * 60)
    print("WEEK 11 - LESSON 6: PPO ON LUNARLANDER-V2")
    print("=" * 60)

    describe_lunarlander()

    print("\n" + "=" * 60)
    print("TRAINING PPO ON LUNARLANDER-V2")
    print("=" * 60)

    rewards = train_lunarlander_ppo(
        total_steps=1_000_000,
        rollout_length=2048,
        ppo_epochs=10,
        clip_eps=0.2,
        lr=3e-4,
        gamma=0.999,
        lam=0.98,
        entropy_coeff=0.01,
        seed=42
    )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(rewards, alpha=0.3, color='steelblue')
    w = min(100, len(rewards))
    sm = [np.mean(rewards[max(0,i-w):i+1]) for i in range(len(rewards))]
    ax.plot(sm, color='steelblue', linewidth=2, label=f'{w}-ep avg')
    ax.axhline(y=200, color='red', linestyle='--', alpha=0.7, label='Solved (200)')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('LunarLander-v2 Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if len(rewards) > 100:
        ax.hist(rewards[-200:], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=200, color='red', linestyle='--', label='Solved threshold')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Count')
    ax.set_title('Final Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('PPO on LunarLander-v2', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('11_ppo/ppo_lunarlander.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to '11_ppo/ppo_lunarlander.png'")
    plt.close()

    if len(rewards) >= 100:
        final_avg = np.mean(rewards[-100:])
        print(f"\nFinal avg reward (last 100 ep): {final_avg:.1f}")
        if final_avg >= 200:
            print("Environment SOLVED!")
        else:
            print("More training needed.")

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. LunarLander needs higher gamma (0.999) and lambda (0.98)")
    print("2. Larger network (256 units) handles more complex state space")
    print("3. Entropy bonus (0.01) critical for sufficient exploration")
    print("4. ~1M steps typically needed to solve LunarLander with PPO")
    print("5. PPO is robust: same algorithm works on CartPole and LunarLander!")
    print("\nNext: Final comparison of all algorithms!")
    print("=" * 60)


if __name__ == "__main__":
    main()
