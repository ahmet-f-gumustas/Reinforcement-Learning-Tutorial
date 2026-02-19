"""
02 - A2C: Advantage Actor-Critic

This script implements the Advantage Actor-Critic (A2C) algorithm,
a synchronous version of A3C that collects batches of experience
before updating.

Demonstrates:
- A2C algorithm with n-step returns
- Advantage computation: A(s,a) = R_n - V(s)
- Entropy bonus for exploration
- Batch update mechanism
- Training on CartPole-v1
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
import gymnasium as gym


class A2CNetwork(nn.Module):
    """
    Shared Actor-Critic network for A2C.

    Uses shared feature extraction with separate actor and critic heads.
    Shared layers help both actor and critic learn useful state representations.
    """

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
        action_logits = self.actor_head(features)
        value = self.critic_head(features)
        return action_logits, value

    def get_action_and_value(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action and return action, log_prob, value."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, value = self.forward(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze()


class A2CAgent:
    """
    A2C Agent that collects n-step trajectories and performs batch updates.

    Key differences from basic Actor-Critic:
    1. Collects n steps of experience before updating (not every step)
    2. Uses n-step returns for lower variance
    3. Adds entropy bonus to encourage exploration
    4. Updates actor and critic together with combined loss
    """

    def __init__(self, state_size: int, action_size: int,
                 lr: float = 0.001, gamma: float = 0.99,
                 n_steps: int = 5, entropy_coeff: float = 0.01,
                 value_coeff: float = 0.5):
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff

        self.network = A2CNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Buffers for n-step collection
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def select_action(self, state: np.ndarray) -> int:
        """Select action and store transition data."""
        action, log_prob, value = self.network.get_action_and_value(state)
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        return action

    def store_reward(self, reward: float, done: bool):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns(self, next_state: np.ndarray, done: bool) -> torch.Tensor:
        """
        Compute n-step returns.

        R_t = r_t + γr_{t+1} + γ²r_{t+2} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})

        The bootstrap value V(s_{t+n}) provides:
        - Lower variance than full Monte Carlo returns
        - Some bias from value function approximation
        - Ability to update before episode ends
        """
        with torch.no_grad():
            _, next_value = self.network(torch.FloatTensor(next_state).unsqueeze(0))
            next_value = next_value.squeeze()

        returns = []
        R = next_value * (1 - int(done))

        for reward, d in zip(reversed(self.rewards), reversed(self.dones)):
            R = reward + self.gamma * R * (1 - int(d))
            returns.insert(0, R)

        return torch.stack(returns)

    def update(self, next_state: np.ndarray, done: bool) -> Tuple[float, float, float]:
        """
        Perform A2C update.

        Combined loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

        - policy_loss: -log π(a|s) * A(s,a)  (maximize advantage-weighted log prob)
        - value_loss: (R_n - V(s))^2          (minimize prediction error)
        - entropy: -Σ π(a|s) log π(a|s)       (maximize entropy for exploration)
        """
        returns = self.compute_returns(next_state, done)
        values = torch.stack(self.values)
        log_probs = torch.stack(self.log_probs)

        # Advantage = Returns - Value estimates
        advantages = returns - values.detach()
        # Normalize advantages for training stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss (actor)
        policy_loss = -(log_probs * advantages).mean()

        # Value loss (critic)
        value_loss = F.mse_loss(values, returns.detach())

        # Entropy bonus
        states_tensor = torch.FloatTensor(np.array(self.states))
        logits, _ = self.network(states_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy().mean()

        # Combined loss
        total_loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

        return policy_loss.item(), value_loss.item(), entropy.item()

    def should_update(self) -> bool:
        """Check if we have collected enough steps for an update."""
        return len(self.rewards) >= self.n_steps


def train_a2c(env_name: str = "CartPole-v1", num_episodes: int = 1000,
              n_steps: int = 5, lr: float = 0.001,
              gamma: float = 0.99) -> Tuple[List[float], List[float], List[float]]:
    """
    Train A2C agent on given environment.

    Returns episode rewards, policy losses, and value losses.
    """
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2CAgent(state_size, action_size, lr=lr, gamma=gamma, n_steps=n_steps)

    episode_rewards = []
    policy_losses = []
    value_losses = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_reward(reward, done)
            episode_reward += reward

            # Update every n_steps or at episode end
            if agent.should_update() or done:
                p_loss, v_loss, _ = agent.update(next_state, done)
                policy_losses.append(p_loss)
                value_losses.append(v_loss)

            state = next_state

        episode_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg:.1f}")

    env.close()
    return episode_rewards, policy_losses, value_losses


def demonstrate_n_step_returns():
    """Show how n-step returns work."""
    print("=" * 60)
    print("N-STEP RETURNS IN A2C")
    print("=" * 60)

    print("""
    A2C uses n-step returns as the target for advantage computation.

    1-step return (TD):
        R^(1) = r_t + γ V(s_{t+1})

    2-step return:
        R^(2) = r_t + γ r_{t+1} + γ² V(s_{t+2})

    n-step return:
        R^(n) = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})

    Full return (MC):
        R^(∞) = r_t + γr_{t+1} + γ²r_{t+2} + ... + γ^{T-t}r_T

    Trade-off:
    ┌──────────────────────────────────────────────────┐
    │  1-step ◄─────────────────────────► Full MC      │
    │  Low variance                    High variance   │
    │  High bias                       No bias         │
    │  Fast updates                    Slow updates    │
    └──────────────────────────────────────────────────┘

    A2C typically uses n=5, a good compromise.
    """)

    # Numerical example
    print("Example with n=3, γ=0.99:")
    rewards = [1.0, 0.5, 2.0]
    V_next = 3.0
    gamma = 0.99

    R = V_next
    for r in reversed(rewards):
        R = r + gamma * R
    print(f"  Rewards: {rewards}")
    print(f"  V(s_{{t+3}}) = {V_next}")
    print(f"  R^(3) = {rewards[0]} + {gamma}*{rewards[1]} + {gamma}²*{rewards[2]} + {gamma}³*{V_next}")
    print(f"        = {R:.4f}")


def demonstrate_entropy_bonus():
    """Show the effect of entropy regularization."""
    print("\n" + "=" * 60)
    print("ENTROPY BONUS FOR EXPLORATION")
    print("=" * 60)

    print("""
    A2C adds an entropy bonus to prevent premature convergence:

    Loss = Policy Loss + c₁ × Value Loss - c₂ × Entropy

    Entropy H(π) = -Σ_a π(a|s) log π(a|s)

    High entropy:  π = [0.25, 0.25, 0.25, 0.25]  → H = 1.386
    Low entropy:   π = [0.97, 0.01, 0.01, 0.01]  → H = 0.136

    The entropy bonus:
    - Encourages the policy to explore (not commit too early)
    - Prevents collapse to deterministic policy
    - Controlled by entropy coefficient c₂
    """)

    # Compute entropy examples
    uniform_probs = torch.FloatTensor([0.25, 0.25, 0.25, 0.25])
    peaked_probs = torch.FloatTensor([0.97, 0.01, 0.01, 0.01])
    moderate_probs = torch.FloatTensor([0.6, 0.2, 0.1, 0.1])

    for name, probs in [("Uniform", uniform_probs),
                         ("Moderate", moderate_probs),
                         ("Peaked", peaked_probs)]:
        entropy = -(probs * probs.log()).sum()
        print(f"  {name:10s}: π = {probs.numpy()} → H = {entropy:.3f}")


def main():
    print("\n" + "=" * 60)
    print("WEEK 10 - LESSON 2: A2C (ADVANTAGE ACTOR-CRITIC)")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Explain n-step returns
    demonstrate_n_step_returns()

    # 2. Explain entropy bonus
    demonstrate_entropy_bonus()

    # 3. Train A2C on CartPole
    print("\n" + "=" * 60)
    print("TRAINING A2C ON CARTPOLE-V1")
    print("=" * 60)

    rewards, p_losses, v_losses = train_a2c(
        env_name="CartPole-v1",
        num_episodes=800,
        n_steps=5,
        lr=0.001,
        gamma=0.99
    )

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Rewards
    window = 50
    smoothed = [np.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
    axes[0].plot(rewards, alpha=0.3, color='steelblue')
    axes[0].plot(smoothed, color='steelblue', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].axhline(y=500, color='red', linestyle='--', alpha=0.5, label='Max')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Policy loss
    axes[1].plot(p_losses, alpha=0.5, color='coral')
    axes[1].set_xlabel('Update Step')
    axes[1].set_ylabel('Policy Loss')
    axes[1].set_title('Policy Loss')
    axes[1].grid(True, alpha=0.3)

    # Value loss
    axes[2].plot(v_losses, alpha=0.5, color='green')
    axes[2].set_xlabel('Update Step')
    axes[2].set_ylabel('Value Loss')
    axes[2].set_title('Value Loss')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('10_actor_critic/a2c_training.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to '10_actor_critic/a2c_training.png'")
    plt.close()

    print(f"\nFinal Performance (last 100 episodes): {np.mean(rewards[-100:]):.1f} ± {np.std(rewards[-100:]):.1f}")

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. A2C collects n-step trajectories before updating (batch updates)")
    print("2. N-step returns provide a bias-variance trade-off")
    print("3. Entropy bonus prevents premature convergence")
    print("4. Combined loss: policy + value + entropy")
    print("5. Gradient clipping improves training stability")
    print("\nNext: A3C - Asynchronous Advantage Actor-Critic!")
    print("=" * 60)


if __name__ == "__main__":
    main()
