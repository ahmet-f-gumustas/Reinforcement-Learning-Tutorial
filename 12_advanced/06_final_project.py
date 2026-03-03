"""
06 - Final Project: Solve Your Own Environment

This script provides a complete PPO implementation that can be applied
to various Gymnasium environments as a final project template.

Demonstrates:
- Modular PPO that works on any Gymnasium environment
- Environment selection and configuration
- Hyperparameter sweeps
- Comprehensive evaluation and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import gymnasium as gym
import time


@dataclass
class ProjectConfig:
    """Configuration for the final project."""
    env_name: str = "CartPole-v1"
    hidden_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 10
    minibatch_size: int = 64
    rollout_length: int = 2048
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    total_steps: int = 200000
    seed: int = 42


class DiscretePolicy(nn.Module):
    """Actor-Critic for discrete actions."""

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

    def get_action(self, state, det=False):
        with torch.no_grad():
            logits, val = self.forward(torch.FloatTensor(state).unsqueeze(0))
        d = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(-1) if det else d.sample()
        return a.item(), d.log_prob(a).item(), val.squeeze().item()

    def evaluate(self, states, actions):
        logits, vals = self.forward(states)
        d = torch.distributions.Categorical(logits=logits)
        return d.log_prob(actions), vals.squeeze(), d.entropy()


class ContinuousPolicy(nn.Module):
    """Actor-Critic for continuous actions."""

    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh()
        )
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))
        self.critic = nn.Linear(hidden_size, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x):
        f = self.shared(x)
        return self.actor_mean(f), self.actor_log_std.exp(), self.critic(f)

    def get_action(self, state, det=False):
        with torch.no_grad():
            mean, std, val = self.forward(torch.FloatTensor(state).unsqueeze(0))
        d = torch.distributions.Normal(mean, std)
        a = mean.squeeze(0) if det else d.sample().squeeze(0)
        lp = d.log_prob(a).sum().item()
        return a.numpy(), lp, val.squeeze().item()

    def evaluate(self, states, actions):
        mean, std, vals = self.forward(states)
        d = torch.distributions.Normal(mean, std)
        lp = d.log_prob(actions).sum(-1)
        ent = d.entropy().sum(-1)
        return lp, vals.squeeze(), ent


def compute_gae(rewards, values, nv, dones, gamma, lam):
    adv, g = [], 0
    vals = list(values) + [nv]
    for t in reversed(range(len(rewards))):
        d = rewards[t] + gamma * vals[t+1] * (1-int(dones[t])) - vals[t]
        g = d + gamma * lam * (1-int(dones[t])) * g
        adv.insert(0, g)
    adv = torch.FloatTensor(adv)
    return adv, adv + torch.FloatTensor(values)


def solve_environment(config: ProjectConfig) -> Dict:
    """Universal PPO solver for any Gymnasium environment."""
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = gym.make(config.env_name)
    state_size = env.observation_space.shape[0]

    # Detect action space type
    is_continuous = hasattr(env.action_space, 'shape') and len(env.action_space.shape) > 0
    if is_continuous:
        action_size = env.action_space.shape[0]
        action_high = float(env.action_space.high[0])
        net = ContinuousPolicy(state_size, action_size, config.hidden_size)
    else:
        action_size = env.action_space.n
        net = DiscretePolicy(state_size, action_size, config.hidden_size)

    opt = optim.Adam(net.parameters(), lr=config.lr)

    episode_rewards = []
    state, _ = env.reset(seed=config.seed)
    cur_r = 0
    steps = 0

    print(f"Solving {config.env_name}")
    print(f"  State: {state_size}D, Action: {'continuous' if is_continuous else 'discrete'} "
          f"({action_size}), Steps: {config.total_steps}")
    print("-" * 50)
    start_time = time.time()

    while steps < config.total_steps:
        buf = {'s': [], 'a': [], 'r': [], 'd': [], 'lp': [], 'v': []}

        for _ in range(config.rollout_length):
            a, lp, v = net.get_action(state)
            if is_continuous:
                a_env = np.clip(a, -action_high, action_high)
            else:
                a_env = a
            ns, r, term, trunc, _ = env.step(a_env)
            done = term or trunc
            buf['s'].append(state); buf['a'].append(a); buf['r'].append(r)
            buf['d'].append(done); buf['lp'].append(lp); buf['v'].append(v)
            cur_r += r; steps += 1
            if done:
                episode_rewards.append(cur_r)
                cur_r = 0
                state, _ = env.reset()
            else:
                state = ns

        # Bootstrap
        with torch.no_grad():
            if is_continuous:
                _, _, nv = net(torch.FloatTensor(state).unsqueeze(0))
            else:
                _, nv = net(torch.FloatTensor(state).unsqueeze(0))
            nv = nv.squeeze().item()

        adv, ret = compute_gae(buf['r'], buf['v'], nv, buf['d'],
                                config.gamma, config.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        st = torch.FloatTensor(np.array(buf['s']))
        at = torch.FloatTensor(np.array(buf['a'])) if is_continuous else torch.LongTensor(buf['a'])
        olp = torch.FloatTensor(buf['lp'])
        n = config.rollout_length

        for _ in range(config.ppo_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, config.minibatch_size):
                mb = idx[start:start+config.minibatch_size]
                nlp, vals, ent = net.evaluate(st[mb], at[mb])
                ratio = torch.exp(nlp - olp[mb])
                s1 = ratio * adv[mb]
                s2 = torch.clamp(ratio, 1-config.clip_eps, 1+config.clip_eps) * adv[mb]
                loss = (-torch.min(s1, s2).mean()
                        + config.value_coeff * F.mse_loss(vals, ret[mb].detach())
                        - config.entropy_coeff * ent.mean())
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
                opt.step()

        if len(episode_rewards) > 0 and len(episode_rewards) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            elapsed = time.time() - start_time
            print(f"  Step {steps:>7d} | Ep: {len(episode_rewards):>4d} | "
                  f"Avg: {avg:>7.1f} | Time: {elapsed:.1f}s")

    env.close()
    elapsed = time.time() - start_time
    print(f"\nDone! Total time: {elapsed:.1f}s, Episodes: {len(episode_rewards)}")

    return {
        'rewards': episode_rewards,
        'config': config,
        'time': elapsed
    }


def suggest_environments():
    """Suggest environments for the final project."""
    print("=" * 60)
    print("FINAL PROJECT: ENVIRONMENT SUGGESTIONS")
    print("=" * 60)

    print("""
    Easy (start here):
    ┌─────────────────────┬──────────────────────────────┐
    │ CartPole-v1          │ Balance a pole (discrete)    │
    │ MountainCar-v0       │ Drive up a hill (discrete)   │
    │ Acrobot-v1           │ Swing a double pendulum      │
    └─────────────────────┴──────────────────────────────┘

    Medium:
    ┌─────────────────────┬──────────────────────────────┐
    │ LunarLander-v2       │ Land a spacecraft (discrete) │
    │ Pendulum-v1          │ Balance inverted (continuous) │
    │ BipedalWalker-v3     │ Walk a 2D robot (continuous)  │
    └─────────────────────┴──────────────────────────────┘

    Hard:
    ┌─────────────────────┬──────────────────────────────┐
    │ LunarLanderContinuous│ Land with continuous thrust  │
    │ CarRacing-v2         │ Race a car (image input)     │
    │ Ant-v4 (MuJoCo)     │ Control an ant robot         │
    └─────────────────────┴──────────────────────────────┘

    Challenge yourself:
    1. Pick an environment
    2. Apply PPO (use the solver above!)
    3. Tune hyperparameters
    4. Compare with DQN/A2C from previous weeks
    5. Visualize results
    """)


def main():
    print("\n" + "=" * 60)
    print("WEEK 12 - LESSON 6: FINAL PROJECT")
    print("Solve Your Own Environment with PPO")
    print("=" * 60)

    suggest_environments()

    # Solve CartPole as demo
    print("\n" + "=" * 60)
    print("DEMO: SOLVING CARTPOLE-V1")
    print("=" * 60)

    config = ProjectConfig(env_name="CartPole-v1", total_steps=100000)
    result = solve_environment(config)

    rewards = result['rewards']
    if rewards:
        plt.figure(figsize=(10, 4))
        plt.plot(rewards, alpha=0.3, color='steelblue')
        w = min(50, len(rewards))
        sm = [np.mean(rewards[max(0,i-w):i+1]) for i in range(len(rewards))]
        plt.plot(sm, color='steelblue', linewidth=2, label=f'{w}-ep avg')
        plt.axhline(y=475, color='red', linestyle='--', alpha=0.5, label='Target')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'PPO on {config.env_name} (Final Project Demo)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('12_advanced/final_project.png', dpi=150, bbox_inches='tight')
        print("Plot saved to '12_advanced/final_project.png'")
        plt.close()

        print(f"\nFinal avg (last 50 ep): {np.mean(rewards[-50:]):.1f}")

    print("\n" + "=" * 60)
    print("YOUR TURN!")
    print("=" * 60)
    print("Modify the config to solve a different environment:")
    print('  config = ProjectConfig(env_name="LunarLander-v2", total_steps=500000)')
    print('  result = solve_environment(config)')
    print("=" * 60)


if __name__ == "__main__":
    main()
