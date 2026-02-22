"""
07 - Final Comparison: DQN vs A2C vs PPO

Comprehensive comparison of all major algorithms from Weeks 8-11
on CartPole-v1.

Demonstrates:
- Sample efficiency comparison
- Training stability
- Final performance
- Algorithm selection guide
- Complete curriculum summary
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict
import gymnasium as gym
from collections import deque
import random


# =============================================================================
# DQN (Week 8)
# =============================================================================

class DQNNet(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, a)
        )
    def forward(self, x): return self.net(x)


def train_dqn(env_name, num_episodes, seed=42):
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    env = gym.make(env_name)
    s, a = env.observation_space.shape[0], env.action_space.n
    q = DQNNet(s, a); tq = DQNNet(s, a)
    tq.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=1e-3)
    buf = deque(maxlen=10000)
    eps = 1.0
    rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_r = 0; done = False
        while not done:
            if np.random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = q(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
            ns, r, term, trunc, _ = env.step(action)
            done = term or trunc
            buf.append((state, action, r, ns, float(done)))
            ep_r += r; state = ns
            if len(buf) >= 64:
                batch = random.sample(buf, 64)
                st, ac, re, nst, dn = zip(*batch)
                st_t = torch.FloatTensor(np.array(st))
                ac_t = torch.LongTensor(ac)
                re_t = torch.FloatTensor(re)
                nst_t = torch.FloatTensor(np.array(nst))
                dn_t = torch.FloatTensor(dn)
                qv = q(st_t).gather(1, ac_t.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    nqv = tq(nst_t).max(1)[0]
                    targets = re_t + 0.99 * nqv * (1 - dn_t)
                loss = F.mse_loss(qv, targets)
                opt.zero_grad(); loss.backward(); opt.step()
        eps = max(0.01, eps * 0.995)
        if (ep + 1) % 10 == 0:
            tq.load_state_dict(q.state_dict())
        rewards.append(ep_r)
    env.close()
    return rewards


# =============================================================================
# A2C (Week 10)
# =============================================================================

def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    advantages, gae = [], 0
    vals = list(values) + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * vals[t+1] * (1-int(dones[t])) - vals[t]
        gae = delta + gamma * lam * (1-int(dones[t])) * gae
        advantages.insert(0, gae)
    adv = torch.FloatTensor(advantages)
    return adv, adv + torch.FloatTensor(values)


class ACNet(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(s, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, a)
        self.critic = nn.Linear(64, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    def forward(self, x):
        f = self.shared(x); return self.actor(f), self.critic(f)


def train_a2c(env_name, total_steps, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make(env_name)
    s, a = env.observation_space.shape[0], env.action_space.n
    net = ACNet(s, a)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    ep_rewards = []
    state, _ = env.reset(seed=seed)
    cur = 0; steps = 0; rollout = 128

    while steps < total_steps:
        buf = {'states':[], 'actions':[], 'rewards':[], 'dones':[], 'lps':[], 'vals':[]}
        for _ in range(rollout):
            with torch.no_grad():
                logits, value = net(torch.FloatTensor(state).unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            ns, r, term, trunc, _ = env.step(action.item())
            done = term or trunc
            buf['states'].append(state); buf['actions'].append(action.item())
            buf['rewards'].append(r); buf['dones'].append(done)
            buf['lps'].append(dist.log_prob(action).item())
            buf['vals'].append(value.squeeze().item())
            cur += r; steps += 1
            if done: ep_rewards.append(cur); cur = 0; state, _ = env.reset()
            else: state = ns
        with torch.no_grad():
            _, nv = net(torch.FloatTensor(state).unsqueeze(0))
            nval = nv.squeeze().item()
        adv, ret = compute_gae(buf['rewards'], buf['vals'], nval, buf['dones'])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        st_t = torch.FloatTensor(np.array(buf['states']))
        ac_t = torch.LongTensor(buf['actions'])
        logits, vals = net(st_t)
        d = torch.distributions.Categorical(logits=logits)
        p_loss = -(d.log_prob(ac_t) * adv.detach()).mean()
        v_loss = F.mse_loss(vals.squeeze(), ret.detach())
        loss = p_loss + 0.5 * v_loss - 0.01 * d.entropy().mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
    env.close()
    return ep_rewards


# =============================================================================
# PPO (Week 11)
# =============================================================================

class PPONet(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(s, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, a)
        self.critic = nn.Linear(64, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    def forward(self, x):
        f = self.shared(x); return self.actor(f), self.critic(f)


def train_ppo(env_name, total_steps, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make(env_name)
    s, a = env.observation_space.shape[0], env.action_space.n
    net = PPONet(s, a)
    opt = optim.Adam(net.parameters(), lr=3e-4)
    ep_rewards = []
    state, _ = env.reset(seed=seed)
    cur = 0; steps = 0; rollout = 2048; epochs = 10; mb = 64; clip = 0.2

    while steps < total_steps:
        buf = {'states':[], 'actions':[], 'rewards':[], 'dones':[], 'lps':[], 'vals':[]}
        for _ in range(rollout):
            with torch.no_grad():
                logits, value = net(torch.FloatTensor(state).unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            ns, r, term, trunc, _ = env.step(action.item())
            done = term or trunc
            buf['states'].append(state); buf['actions'].append(action.item())
            buf['rewards'].append(r); buf['dones'].append(done)
            buf['lps'].append(dist.log_prob(action).item())
            buf['vals'].append(value.squeeze().item())
            cur += r; steps += 1
            if done: ep_rewards.append(cur); cur = 0; state, _ = env.reset()
            else: state = ns
        with torch.no_grad():
            _, nv = net(torch.FloatTensor(state).unsqueeze(0))
            nval = nv.squeeze().item()
        adv, ret = compute_gae(buf['rewards'], buf['vals'], nval, buf['dones'])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        st_t = torch.FloatTensor(np.array(buf['states']))
        ac_t = torch.LongTensor(buf['actions'])
        old_lp = torch.FloatTensor(buf['lps'])
        n = rollout
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, mb):
                mid = idx[start:start+mb]
                logits, vals = net(st_t[mid])
                d = torch.distributions.Categorical(logits=logits)
                new_lp = d.log_prob(ac_t[mid])
                ratio = torch.exp(new_lp - old_lp[mid])
                s1 = ratio * adv[mid]
                s2 = torch.clamp(ratio, 1-clip, 1+clip) * adv[mid]
                p_loss = -torch.min(s1, s2).mean()
                v_loss = F.mse_loss(vals.squeeze(), ret[mid].detach())
                loss = p_loss + 0.5 * v_loss - 0.01 * d.entropy().mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
    env.close()
    return ep_rewards


# =============================================================================
# Comparison
# =============================================================================

def run_comparison(env_name="CartPole-v1", total_steps=200000):
    """Compare all algorithms."""
    print("=" * 60)
    print(f"COMPARISON ON {env_name}")
    print("=" * 60)

    results = {}

    # DQN (episode-based, convert steps to episodes estimate)
    episodes_approx = total_steps // 200
    print(f"\nTraining DQN (~{episodes_approx} episodes)...")
    results['DQN\n(Week 8)'] = train_dqn(env_name, episodes_approx)

    print(f"\nTraining A2C ({total_steps} steps)...")
    results['A2C\n(Week 10)'] = train_a2c(env_name, total_steps)

    print(f"\nTraining PPO ({total_steps} steps)...")
    results['PPO\n(Week 11)'] = train_ppo(env_name, total_steps)

    return results


def plot_comparison(results: Dict[str, List]):
    colors = {'DQN\n(Week 8)': 'steelblue',
              'A2C\n(Week 10)': 'coral',
              'PPO\n(Week 11)': 'green'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Learning curves
    ax = axes[0]
    for name, rewards in results.items():
        w = min(50, len(rewards))
        sm = [np.mean(rewards[max(0,i-w):i+1]) for i in range(len(rewards))]
        ax.plot(sm, label=name.replace('\n', ' '), color=colors[name], linewidth=2)
    ax.axhline(y=475, color='gray', linestyle='--', alpha=0.5, label='Target (475)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Reward')
    ax.set_title('Learning Curves')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Final performance
    ax = axes[1]
    names = list(results.keys())
    means = [np.mean(results[n][-100:]) for n in names]
    stds = [np.std(results[n][-100:]) for n in names]
    clrs = [colors[n] for n in names]
    bars = ax.bar([n.replace('\n', '\n') for n in names], means,
                  yerr=stds, capsize=8, color=clrs, alpha=0.8, edgecolor='black')
    ax.axhline(y=475, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Avg Reward (last 100 ep)')
    ax.set_title('Final Performance')
    ax.grid(True, alpha=0.3)

    # Episodes to reach 400 reward
    ax = axes[2]
    solve_episodes = []
    for name, rewards in results.items():
        w = 50
        solved = None
        for i in range(w, len(rewards)):
            if np.mean(rewards[i-w:i]) >= 400:
                solved = i
                break
        solve_episodes.append(solved if solved else len(rewards))
    bars = ax.bar([n.replace('\n', '\n') for n in names], solve_episodes,
                  color=clrs, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Episodes to Reach 400 Avg Reward')
    ax.set_title('Sample Efficiency')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Algorithm Comparison: DQN vs A2C vs PPO', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('11_ppo/final_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to '11_ppo/final_comparison.png'")
    plt.close()


def print_curriculum_summary():
    print("\n" + "=" * 60)
    print("REINFORCEMENT LEARNING CURRICULUM SUMMARY")
    print("=" * 60)
    print("""
    Week 1:  Introduction – Agent, Environment, Reward
    Week 2:  MDPs – Markov property, Bellman equations
    Week 3:  Dynamic Programming – Value/Policy Iteration
    Week 4:  Monte Carlo – Episode-based learning
    Week 5:  Temporal Difference – TD(0), n-step, Eligibility Traces
    Week 6:  Q-Learning & SARSA – On/off-policy tabular
    Week 7:  Function Approximation – Linear, neural network
    Week 8:  DQN – Experience replay, target network
    Week 9:  Policy Gradient – REINFORCE, baseline
    Week 10: Actor-Critic – A2C, A3C, GAE
    Week 11: PPO – Trust region, clip, penalty  ← YOU ARE HERE
    Week 12: Advanced – Multi-agent, Model-based, Inverse RL

    Algorithm Selection Guide:
    ┌──────────────────┬──────────────────────────────────────┐
    │ Scenario          │ Recommended Algorithm                │
    ├──────────────────┼──────────────────────────────────────┤
    │ Discrete actions  │ PPO > DQN (simpler)                  │
    │ Continuous ctrl   │ PPO > SAC (simpler) > DDPG           │
    │ Sample efficiency │ DQN (replay) > PPO > A2C             │
    │ Stability         │ PPO > A2C > REINFORCE                │
    │ Best default      │ PPO (works everywhere!)              │
    │ Educational       │ REINFORCE → A2C → PPO (this series)  │
    └──────────────────┴──────────────────────────────────────┘

    The PPO Paper (2017) Summary:
    "PPO attains the data efficiency and reliable performance of
    TRPO, while using only first-order optimization."
    — Schulman et al., 2017
    """)


def main():
    print("\n" + "=" * 60)
    print("WEEK 11 - LESSON 7: FINAL COMPARISON")
    print("DQN vs A2C vs PPO")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    results = run_comparison(env_name="CartPole-v1", total_steps=200000)

    print("\n--- Final Results ---")
    for name, rewards in results.items():
        n = name.replace('\n', ' ')
        print(f"{n:15s}: Final avg = {np.mean(rewards[-100:]):.1f} ± {np.std(rewards[-100:]):.1f}")

    plot_comparison(results)
    print_curriculum_summary()

    print("\n" + "=" * 60)
    print("WEEK 11 COMPLETE!")
    print("=" * 60)
    print("1. TRPO: constrained optimization, theoretically sound but complex")
    print("2. PPO-Clip: clip ratio to [1-ε, 1+ε], simple and effective")
    print("3. PPO-Penalty: KL penalty with adaptive beta")
    print("4. PPO = A2C + GAE + clipping → state-of-the-art baseline")
    print("5. PPO works for discrete and continuous, same hyperparameters")
    print("6. Default ε=0.2, lr=3e-4, γ=0.99, λ=0.95 work for most envs")
    print("\nNext: Week 12 - Advanced Topics (Multi-agent, Model-based, IRL)!")
    print("=" * 60)


if __name__ == "__main__":
    main()
