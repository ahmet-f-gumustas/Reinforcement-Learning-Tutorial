"""
06 - Continuous Action Actor-Critic

This script implements Actor-Critic for continuous action spaces
using Gaussian policies (building on Week 9's continuous actions).

Demonstrates:
- Gaussian actor for continuous actions
- Actor-Critic with continuous actions
- Training on Pendulum-v1
- Action squashing with tanh
- Learned standard deviation
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
import gymnasium as gym


class ContinuousActorCritic(nn.Module):
    """
    Actor-Critic network for continuous action spaces.

    Actor outputs: mean μ(s) and log_std for Gaussian policy
    Critic outputs: state value V(s)

    Action sampling: a ~ N(μ(s), σ(s))
    where σ = exp(log_std)
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        # Actor: outputs mean of Gaussian
        self.actor_mean = nn.Linear(hidden_size, action_size)
        # Learnable log standard deviation (not state-dependent)
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))

        # Critic: outputs state value
        self.critic_head = nn.Linear(hidden_size, 1)

        # Initialize
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, state: torch.Tensor):
        features = self.shared(state)
        action_mean = self.actor_mean(features)
        action_std = self.actor_log_std.exp()
        value = self.critic_head(features)
        return action_mean, action_std, value

    def get_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Sample action from Gaussian policy.

        Returns: action, log_prob, value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mean, std, value = self.forward(state_tensor)

        if deterministic:
            action = mean.squeeze(0)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample().squeeze(0)

        # Compute log probability
        dist = torch.distributions.Normal(mean.squeeze(0), std)
        log_prob = dist.log_prob(action).sum()

        return action.numpy(), log_prob.item(), value.squeeze().item()

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate given actions for policy update."""
        mean, std, values = self.forward(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        return log_probs, values.squeeze(), entropy


def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """Compute GAE advantages."""
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


def train_continuous_ac(env_name: str = "Pendulum-v1",
                        total_steps: int = 200000,
                        rollout_length: int = 2048,
                        lr: float = 0.0003,
                        gamma: float = 0.99,
                        lam: float = 0.95,
                        entropy_coeff: float = 0.0,
                        value_coeff: float = 0.5,
                        max_grad_norm: float = 0.5) -> Tuple[List, List]:
    """
    Train continuous action Actor-Critic on given environment.

    Key differences from discrete:
    1. Action is sampled from Normal distribution (not Categorical)
    2. Log probability computed for continuous values
    3. Action may need to be clipped/scaled to environment bounds
    """
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_high = env.action_space.high[0]

    network = ContinuousActorCritic(state_size, action_size)
    optimizer = optim.Adam(network.parameters(), lr=lr)

    episode_rewards = []
    all_std = []
    state, _ = env.reset()
    current_reward = 0
    steps = 0

    while steps < total_steps:
        # Collect rollout
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        for _ in range(rollout_length):
            action, log_prob, value = network.get_action(state)

            # Clip action to environment bounds
            clipped_action = np.clip(action, -action_high, action_high)
            next_state, reward, terminated, truncated, _ = env.step(clipped_action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)

            current_reward += reward
            if done:
                episode_rewards.append(current_reward)
                current_reward = 0
                state, _ = env.reset()
            else:
                state = next_state

        steps += rollout_length

        # Bootstrap
        with torch.no_grad():
            _, _, next_value = network(torch.FloatTensor(state).unsqueeze(0))
            next_value = next_value.squeeze().item()

        # Compute GAE
        advantages, returns = compute_gae(rewards, values, next_value, dones, gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.FloatTensor(np.array(actions))

        new_log_probs, new_values, entropy = network.evaluate_actions(
            states_tensor, actions_tensor)

        policy_loss = -(new_log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(new_values, returns.detach())
        total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
        optimizer.step()

        # Track standard deviation
        all_std.append(network.actor_log_std.exp().mean().item())

        if len(episode_rewards) % 50 == 0 and len(episode_rewards) > 0:
            avg = np.mean(episode_rewards[-50:])
            std = network.actor_log_std.exp().mean().item()
            print(f"Step {steps:>7d} | Episodes: {len(episode_rewards):>4d} | "
                  f"Avg Reward: {avg:>7.1f} | Std: {std:.3f}")

    env.close()
    return episode_rewards, all_std


def demonstrate_continuous_actor_critic():
    """Explain continuous action Actor-Critic."""
    print("=" * 60)
    print("ACTOR-CRITIC FOR CONTINUOUS ACTIONS")
    print("=" * 60)

    print("""
    Discrete Actions (CartPole):
        Actor output: π(a|s) = Categorical(softmax(logits))
        Sample: a ~ Categorical (a ∈ {0, 1, ..., n})

    Continuous Actions (Pendulum, Robot Control):
        Actor output: π(a|s) = N(μ_θ(s), σ_θ)
        Sample: a ~ Normal(μ, σ) (a ∈ ℝ)

    Architecture:
        state → Shared Layers → Actor Mean μ(s)
                              → Actor Std σ (learnable parameter)
                              → Critic V(s)

    Log probability:
        log π(a|s) = -½[(a-μ)/σ]² - log(σ) - ½log(2π)

    Key differences from discrete:
    1. Output is mean μ(s) instead of logits
    2. Standard deviation σ is learned (can be state-dependent)
    3. Actions are real-valued, may need clipping
    4. Log probability uses Gaussian formula
    """)


def main():
    print("\n" + "=" * 60)
    print("WEEK 10 - LESSON 6: CONTINUOUS ACTION ACTOR-CRITIC")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Explain continuous actions
    demonstrate_continuous_actor_critic()

    # 2. Train on Pendulum
    print("\n" + "=" * 60)
    print("TRAINING ON PENDULUM-V1")
    print("=" * 60)
    print("Pendulum: Swing up and balance an inverted pendulum")
    print("  State: [cos(θ), sin(θ), θ_dot]")
    print("  Action: torque ∈ [-2.0, 2.0]")
    print("  Reward: -(θ² + 0.1θ_dot² + 0.001a²)")
    print()

    rewards, stds = train_continuous_ac(
        env_name="Pendulum-v1",
        total_steps=200000,
        rollout_length=2048,
        lr=0.0003,
        gamma=0.99,
        lam=0.95
    )

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Rewards
    ax = axes[0]
    ax.plot(rewards, alpha=0.3, color='steelblue')
    window = min(30, len(rewards))
    if window > 0:
        smoothed = [np.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
        ax.plot(smoothed, color='steelblue', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.grid(True, alpha=0.3)

    # Standard deviation evolution
    ax = axes[1]
    ax.plot(stds, color='coral', linewidth=2)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Action Std')
    ax.set_title('Learned Standard Deviation')
    ax.grid(True, alpha=0.3)

    # Reward distribution
    ax = axes[2]
    if len(rewards) > 20:
        ax.hist(rewards[-100:], bins=20, color='green', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Count')
    ax.set_title('Final Reward Distribution')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Continuous Action Actor-Critic on Pendulum-v1',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('10_actor_critic/continuous_ac.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to '10_actor_critic/continuous_ac.png'")
    plt.close()

    if rewards:
        print(f"\nFinal Performance (last 50 episodes): "
              f"{np.mean(rewards[-50:]):.1f} ± {np.std(rewards[-50:]):.1f}")

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Gaussian policy π(a|s) = N(μ(s), σ) for continuous actions")
    print("2. Standard deviation σ is learned and typically decreases during training")
    print("3. Actions may need clipping to environment bounds")
    print("4. Same GAE and A2C framework works for continuous case")
    print("5. Lower learning rate often needed for continuous control")
    print("\nNext: Method comparison across all algorithms!")
    print("=" * 60)


if __name__ == "__main__":
    main()
