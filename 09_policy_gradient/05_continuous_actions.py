"""
05 - Continuous Action Spaces

This script demonstrates policy gradient methods for continuous action spaces.
We use a Gaussian policy that outputs mean and standard deviation for each action dimension.

Demonstrates:
- Gaussian policy for continuous actions
- Policy parameterization with μ(s) and σ(s)
- Log-likelihood computation for continuous distributions
- Training on Pendulum environment
- Comparison with discrete action methods
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple


class GaussianPolicy(nn.Module):
    """
    Gaussian policy for continuous action spaces.

    Outputs mean μ(s) and log standard deviation log_σ(s) for each action dimension.
    Actions are sampled from Normal(μ(s), σ(s)).
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()

        # Shared feature layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Mean head
        self.mean = nn.Linear(hidden_size, action_size)

        # Log std head (learned, not state-dependent initially)
        # Using log_std ensures std is always positive
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute mean and std of action distribution.

        Args:
            state: State tensor

        Returns:
            mean: Mean of action distribution
            std: Standard deviation of action distribution
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)

        # Convert log_std to std (ensures positivity)
        std = self.log_std.exp()

        return mean, std

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Select action by sampling from Gaussian distribution.

        Args:
            state: Current state

        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.forward(state_tensor)

        # Create normal distribution
        dist = torch.distributions.Normal(mean, std)

        # Sample action
        action = dist.sample()

        # Compute log probability
        log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions

        return action.detach().numpy()[0], log_prob


class ValueNetwork(nn.Module):
    """Value network for baseline."""

    def __init__(self, state_size: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """Compute discounted returns."""
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns


def reinforce_continuous(env: gym.Env,
                        policy_net: GaussianPolicy,
                        value_net: ValueNetwork,
                        policy_optimizer: optim.Optimizer,
                        value_optimizer: optim.Optimizer,
                        num_episodes: int = 500,
                        gamma: float = 0.99,
                        print_every: int = 50) -> List[float]:
    """
    REINFORCE for continuous action spaces.

    Args:
        env: Gymnasium environment with continuous actions
        policy_net: Gaussian policy network
        value_net: Value network (baseline)
        policy_optimizer: Optimizer for policy
        value_optimizer: Optimizer for value function
        num_episodes: Number of episodes
        gamma: Discount factor
        print_every: Print frequency

    Returns:
        List of episode rewards
    """
    rewards_history = []
    running_reward = None

    for episode in range(1, num_episodes + 1):
        log_probs = []
        values = []
        rewards = []

        state, _ = env.reset()
        done = False

        while not done:
            # Select action
            action, log_prob = policy_net.select_action(state)
            log_probs.append(log_prob)

            # Get value estimate
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = value_net(state_tensor)
            values.append(value)

            # Take action (clip to environment bounds)
            action_clipped = np.clip(action, env.action_space.low, env.action_space.high)
            next_state, reward, terminated, truncated, _ = env.step(action_clipped)
            done = terminated or truncated
            rewards.append(reward)

            state = next_state

        # Compute returns and advantages
        returns = compute_returns(rewards, gamma)
        returns_tensor = torch.FloatTensor(returns)
        values_tensor = torch.cat(values).squeeze()

        advantages = returns_tensor - values_tensor.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        # Policy loss
        policy_loss = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss.append(-log_prob * advantage)

        # Update policy
        policy_optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        policy_optimizer.step()

        # Value loss
        value_loss = F.mse_loss(values_tensor, returns_tensor)

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # Track rewards
        episode_reward = sum(rewards)
        rewards_history.append(episode_reward)

        if running_reward is None:
            running_reward = episode_reward
        else:
            running_reward = 0.05 * episode_reward + 0.95 * running_reward

        if episode % print_every == 0:
            print(f"Episode {episode}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Running Avg: {running_reward:.2f}")

    return rewards_history


def plot_continuous_results(rewards: List[float], env_name: str) -> None:
    """Plot learning curves for continuous control."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Raw rewards
    ax1.plot(rewards, alpha=0.6, color='blue')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Raw Episode Rewards')
    ax1.grid(True, alpha=0.3)

    # Smoothed rewards
    window = 20
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax2.plot(smoothed, color='red', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel(f'Smoothed Reward (window={window})')
        ax2.set_title('Smoothed Learning Curve')
        ax2.grid(True, alpha=0.3)

    plt.suptitle(f'REINFORCE on {env_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('continuous_actions_learning.png', dpi=150, bbox_inches='tight')
    print("Learning curve saved to 'continuous_actions_learning.png'")
    plt.close()


def demonstrate_gaussian_policy() -> None:
    """Demonstrate Gaussian policy mechanics."""
    print("=" * 60)
    print("GAUSSIAN POLICY FOR CONTINUOUS ACTIONS")
    print("=" * 60)

    print("\nDiscrete vs Continuous Actions:")
    print("\nDiscrete (e.g., CartPole):")
    print("  - Actions: {0, 1, 2, ...}")
    print("  - Policy: Categorical distribution π(a|s)")
    print("  - Output: Softmax over discrete actions")
    print("  - Sample: Categorical.sample()")

    print("\nContinuous (e.g., Pendulum, Robot Control):")
    print("  - Actions: Real values in [-∞, +∞] or bounded")
    print("  - Policy: Gaussian distribution N(μ(s), σ(s))")
    print("  - Output: Mean μ and std σ")
    print("  - Sample: Normal(μ, σ).sample()")

    print("\nGaussian Policy Parameterization:")
    print("  π(a|s) = N(a; μ_θ(s), σ_θ(s))")
    print("  - μ_θ(s): Mean, output of neural network")
    print("  - σ_θ(s): Std deviation, can be:")
    print("      • Fixed constant")
    print("      • Learned parameter (not state-dependent)")
    print("      • State-dependent (separate network head)")

    print("\nLog Probability:")
    print("  log π(a|s) = log N(a; μ, σ)")
    print("             = -½[(a-μ)/σ]² - log(σ) - ½log(2π)")

    print("\nPolicy Gradient:")
    print("  ∇_θ J = E[∇_θ log π(a|s) * A(s,a)]")
    print("  Same as discrete case, but with Gaussian log prob!")


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("WEEK 9 - LESSON 5: CONTINUOUS ACTION SPACES")
    print("=" * 60)

    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Explain Gaussian policy
    demonstrate_gaussian_policy()

    # 2. Train on Pendulum
    print("\n" + "=" * 60)
    print("TRAINING ON PENDULUM-V1")
    print("=" * 60)

    env = gym.make('Pendulum-v1')
    print(f"\nEnvironment: Pendulum-v1")
    print(f"  State space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Action range: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    policy_net = GaussianPolicy(state_size, action_size, 128)
    value_net = ValueNetwork(state_size, 128)
    policy_opt = optim.Adam(policy_net.parameters(), lr=0.0003)
    value_opt = optim.Adam(value_net.parameters(), lr=0.001)

    print("\nTraining REINFORCE with Gaussian policy...")
    print(f"Learning rates: policy=0.0003, value=0.001")
    print()

    num_episodes = 300
    rewards = reinforce_continuous(env, policy_net, value_net,
                                   policy_opt, value_opt,
                                   num_episodes=num_episodes,
                                   print_every=50)

    # Plot results
    plot_continuous_results(rewards, 'Pendulum-v1')

    # Test learned policy
    print("\n" + "=" * 60)
    print("TESTING LEARNED POLICY")
    print("=" * 60)

    test_rewards = []
    for i in range(10):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = policy_net.select_action(state)
            action_clipped = np.clip(action, env.action_space.low, env.action_space.high)
            next_state, reward, terminated, truncated, _ = env.step(action_clipped)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        test_rewards.append(episode_reward)
        print(f"  Test Episode {i+1}: Reward = {episode_reward:.2f}")

    print(f"\nAverage test reward: {np.mean(test_rewards):.2f}")

    # Show learned policy parameters
    print("\n" + "=" * 60)
    print("LEARNED POLICY PARAMETERS")
    print("=" * 60)
    print(f"Learned log_std: {policy_net.log_std.data}")
    print(f"Learned std: {policy_net.log_std.exp().data}")
    print("(Standard deviation controls exploration)")

    # 3. Summary
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Gaussian policy enables continuous action spaces")
    print("2. Policy outputs mean μ(s) and std σ(s)")
    print("3. Actions sampled from N(μ(s), σ(s))")
    print("4. Log probability used in policy gradient")
    print("5. Std controls exploration (like ε in ε-greedy)")
    print("6. Same REINFORCE algorithm, different distribution!")
    print("\nContinuous actions enable:")
    print("  ✓ Robot control (joint angles, forces)")
    print("  ✓ Autonomous driving (steering, throttle)")
    print("  ✓ Game AI (continuous movement)")
    print("\nNext: Complete CartPole implementation!")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
