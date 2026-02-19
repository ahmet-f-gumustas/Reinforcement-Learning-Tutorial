"""
04 - GAE: Generalized Advantage Estimation

This script implements Generalized Advantage Estimation (GAE),
a technique that provides a smooth interpolation between
TD error (low variance, high bias) and Monte Carlo returns
(high variance, low bias).

Demonstrates:
- GAE computation with lambda parameter
- Bias-variance trade-off visualization
- Different lambda values and their effects
- A2C with GAE integration
- Comparison: GAE vs n-step returns
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
import gymnasium as gym


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for GAE experiments."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(hidden_size, action_size)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value


def compute_gae(rewards: List[float], values: List[float],
                next_value: float, dones: List[bool],
                gamma: float = 0.99, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation.

    GAE(γ, λ) = Σ_{l=0}^{T-t} (γλ)^l × δ_{t+l}

    where δ_t = r_t + γ V(s_{t+1}) - V(s_t)  (TD error)

    Special cases:
    - λ = 0:  GAE = δ_t                  (1-step TD, low variance, high bias)
    - λ = 1:  GAE = Σ γ^l δ_{t+l} = G_t - V(s_t)  (Monte Carlo, high variance, no bias)
    - 0<λ<1:  Smooth interpolation between the two

    Args:
        rewards: List of rewards from trajectory
        values: List of V(s) predictions
        next_value: V(s') for the last state
        dones: List of done flags
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        advantages: GAE advantages
        returns: Advantage + Value (targets for critic)
    """
    advantages = []
    gae = 0

    # Iterate backwards through the trajectory
    values_ext = list(values) + [next_value]

    for t in reversed(range(len(rewards))):
        # TD error at step t
        delta = rewards[t] + gamma * values_ext[t + 1] * (1 - int(dones[t])) - values_ext[t]

        # GAE accumulation: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        gae = delta + gamma * lam * (1 - int(dones[t])) * gae
        advantages.insert(0, gae)

    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(values)

    return advantages, returns


class GAEAgent:
    """A2C agent with GAE for advantage estimation."""

    def __init__(self, state_size: int, action_size: int,
                 lr: float = 0.001, gamma: float = 0.99,
                 lam: float = 0.95, entropy_coeff: float = 0.01,
                 value_coeff: float = 0.5, rollout_length: int = 128):
        self.gamma = gamma
        self.lam = lam
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.rollout_length = rollout_length

        self.network = ActorCriticNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def collect_rollout(self, env, state: np.ndarray) -> Tuple:
        """Collect a rollout of experience."""
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        for _ in range(self.rollout_length):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits, value = self.network(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(done)
            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze().item())

            state = next_state
            if done:
                state, _ = env.reset()

        # Bootstrap value for last state
        with torch.no_grad():
            _, next_value = self.network(torch.FloatTensor(state).unsqueeze(0))
            next_value = next_value.squeeze().item()

        return states, actions, rewards, dones, log_probs, values, next_value, state

    def update(self, states, actions, rewards, dones, old_log_probs, values, next_value):
        """Update using GAE advantages."""
        # Compute GAE
        advantages, returns = compute_gae(
            rewards, values, next_value, dones,
            gamma=self.gamma, lam=self.lam
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        logits, new_values = self.network(states_tensor)
        new_values = new_values.squeeze()

        # Policy loss
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions_tensor)
        policy_loss = -(new_log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = F.mse_loss(new_values, returns.detach())

        # Entropy bonus
        entropy = dist.entropy().mean()

        # Combined loss
        total_loss = (policy_loss
                      + self.value_coeff * value_loss
                      - self.entropy_coeff * entropy)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()


def train_with_gae(env_name: str = "CartPole-v1", total_steps: int = 100000,
                   lam: float = 0.95, gamma: float = 0.99,
                   lr: float = 0.001) -> List[float]:
    """Train agent with GAE on given environment."""
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = GAEAgent(state_size, action_size, lr=lr, gamma=gamma,
                     lam=lam, rollout_length=128)

    state, _ = env.reset()
    episode_rewards = []
    current_reward = 0
    steps = 0

    while steps < total_steps:
        # Collect rollout
        (states, actions, rewards, dones, log_probs,
         values, next_value, state) = agent.collect_rollout(env, state)

        # Track episode rewards
        for r, d in zip(rewards, dones):
            current_reward += r
            if d:
                episode_rewards.append(current_reward)
                current_reward = 0

        # Update
        agent.update(states, actions, rewards, dones, log_probs, values, next_value)
        steps += agent.rollout_length

    env.close()
    return episode_rewards


def demonstrate_gae_concept():
    """Explain GAE visually."""
    print("=" * 60)
    print("GENERALIZED ADVANTAGE ESTIMATION (GAE)")
    print("=" * 60)

    print("""
    GAE provides a smooth trade-off between bias and variance:

    GAE(γ, λ) = Σ_{l=0}^{∞} (γλ)^l × δ_{t+l}

    where δ_t = r_t + γV(s_{t+1}) - V(s_t)

    Different λ values:

    λ = 0:   A_t = δ_t = r_t + γV(s_{t+1}) - V(s_t)
             → Just TD error
             → LOW variance, HIGH bias

    λ = 0.5: A_t = δ_t + 0.5γδ_{t+1} + 0.25γ²δ_{t+2} + ...
             → Weighted combination
             → MEDIUM variance, MEDIUM bias

    λ = 1:   A_t = δ_t + γδ_{t+1} + γ²δ_{t+2} + ...
                  = G_t - V(s_t)
             → Full Monte Carlo advantage
             → HIGH variance, NO bias

    ┌──────────────────────────────────────────────────┐
    │  λ = 0 ◄──── λ = 0.95 (typical) ────► λ = 1    │
    │  TD(0)        Good trade-off          Monte Carlo│
    │  Low var      Medium var              High var   │
    │  High bias    Low bias                No bias    │
    └──────────────────────────────────────────────────┘

    Typical choice: λ = 0.95 or λ = 0.97
    """)


def demonstrate_gae_computation():
    """Show step-by-step GAE computation."""
    print("\n" + "=" * 60)
    print("GAE COMPUTATION EXAMPLE")
    print("=" * 60)

    rewards = [1.0, 0.0, 1.0, 0.0, 2.0]
    values = [3.0, 2.5, 3.5, 2.0, 4.0]
    next_value = 3.0
    dones = [False, False, False, False, False]
    gamma = 0.99
    lam = 0.95

    print(f"\nTrajectory:")
    print(f"  Rewards:     {rewards}")
    print(f"  Values V(s): {values}")
    print(f"  V(s_final):  {next_value}")
    print(f"  γ = {gamma}, λ = {lam}")

    # Step-by-step computation
    values_ext = values + [next_value]
    deltas = []
    for t in range(len(rewards)):
        delta = rewards[t] + gamma * values_ext[t+1] - values_ext[t]
        deltas.append(delta)

    print(f"\nTD errors δ_t = r_t + γV(s_{{t+1}}) - V(s_t):")
    for t, d in enumerate(deltas):
        print(f"  δ_{t} = {rewards[t]} + {gamma}×{values_ext[t+1]} - {values_ext[t]} = {d:.3f}")

    # Compute GAE for different lambda values
    print(f"\nGAE advantages for different λ:")
    for test_lam in [0.0, 0.5, 0.95, 1.0]:
        advantages, returns = compute_gae(
            rewards, values, next_value, dones, gamma, test_lam)
        print(f"  λ = {test_lam:.2f}: A = [{', '.join(f'{a:.3f}' for a in advantages.numpy())}]")


def compare_lambda_values():
    """Train with different lambda values and compare."""
    print("\n" + "=" * 60)
    print("COMPARING DIFFERENT λ VALUES ON CARTPOLE")
    print("=" * 60)

    lambda_values = [0.0, 0.5, 0.95, 1.0]
    results = {}

    for lam in lambda_values:
        print(f"\nTraining with λ = {lam}...")
        torch.manual_seed(42)
        np.random.seed(42)
        rewards = train_with_gae(
            env_name="CartPole-v1",
            total_steps=50000,
            lam=lam,
            lr=0.001
        )
        results[lam] = rewards
        if len(rewards) > 0:
            print(f"  Final avg reward: {np.mean(rewards[-50:]):.1f}")

    # Plot comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for lam, rewards in results.items():
        if len(rewards) > 0:
            window = min(30, len(rewards))
            smoothed = [np.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
            plt.plot(smoothed, label=f'λ = {lam}', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curves for Different λ')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    labels = []
    means = []
    stds = []
    for lam, rewards in results.items():
        if len(rewards) >= 20:
            labels.append(f'λ={lam}')
            means.append(np.mean(rewards[-20:]))
            stds.append(np.std(rewards[-20:]))
    if labels:
        plt.bar(labels, means, yerr=stds, capsize=10,
                color=['steelblue', 'coral', 'green', 'purple'][:len(labels)])
        plt.ylabel('Average Reward (last 20 episodes)')
        plt.title('Final Performance')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('10_actor_critic/gae_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to '10_actor_critic/gae_comparison.png'")
    plt.close()


def main():
    print("\n" + "=" * 60)
    print("WEEK 10 - LESSON 4: GAE (GENERALIZED ADVANTAGE ESTIMATION)")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Explain GAE
    demonstrate_gae_concept()

    # 2. Step-by-step computation
    demonstrate_gae_computation()

    # 3. Compare lambda values
    compare_lambda_values()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. GAE = exponentially weighted average of multi-step TD errors")
    print("2. λ controls the bias-variance trade-off")
    print("3. λ=0 → TD error (low variance, high bias)")
    print("4. λ=1 → Monte Carlo advantage (high variance, no bias)")
    print("5. λ=0.95 is a common default that works well in practice")
    print("6. GAE is used in PPO, TRPO, and most modern policy gradient methods")
    print("\nNext: Complete CartPole Actor-Critic implementation!")
    print("=" * 60)


if __name__ == "__main__":
    main()
