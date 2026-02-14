"""
02 - REINFORCE Algorithm: Monte Carlo Policy Gradient

This script implements the REINFORCE algorithm, the foundational policy gradient method.
REINFORCE uses Monte Carlo sampling and the policy gradient theorem to directly
optimize the policy parameters.

Demonstrates:
- REINFORCE algorithm implementation
- Policy gradient theorem in action
- Episode trajectory collection and return calculation
- Training on CartPole environment
- Learning curves and policy visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
from collections import deque


class PolicyNetwork(nn.Module):
    """
    Policy network for REINFORCE algorithm.

    Architecture: state → FC → ReLU → FC → ReLU → FC → Softmax → action probabilities
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute action probabilities.

        Args:
            state: State tensor

        Returns:
            Action probabilities
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        Select action by sampling from policy and return log probability.

        Args:
            state: Current state as numpy array

        Returns:
            action: Selected action
            log_prob: Log probability of selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state_tensor)

        # Create categorical distribution and sample
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()

        # Get log probability for this action
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Compute discounted returns (G_t) for each timestep.

    For each timestep t, compute:
        G_t = R_{t+1} + γ*R_{t+2} + γ²*R_{t+3} + ...

    Args:
        rewards: List of rewards for an episode
        gamma: Discount factor

    Returns:
        List of returns for each timestep
    """
    returns = []
    G = 0

    # Compute returns in reverse order
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    return returns


def reinforce(env: gym.Env,
              policy_net: PolicyNetwork,
              optimizer: optim.Optimizer,
              num_episodes: int = 1000,
              gamma: float = 0.99,
              print_every: int = 100) -> List[float]:
    """
    REINFORCE algorithm implementation.

    The REINFORCE algorithm:
    1. Generate episode using current policy π_θ
    2. Compute returns G_t for each timestep
    3. Compute policy gradient: ∇J(θ) = E[∇log π_θ(a|s) * G_t]
    4. Update policy: θ ← θ + α * ∇J(θ)

    Args:
        env: Gymnasium environment
        policy_net: Policy network
        optimizer: PyTorch optimizer
        num_episodes: Number of training episodes
        gamma: Discount factor
        print_every: Print progress every N episodes

    Returns:
        List of episode rewards
    """
    rewards_history = []
    running_reward = None

    for episode in range(1, num_episodes + 1):
        # Lists to store episode data
        log_probs = []
        rewards = []

        # Generate episode
        state, _ = env.reset()
        done = False

        while not done:
            # Select action
            action, log_prob = policy_net.select_action(state)
            log_probs.append(log_prob)

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)

            state = next_state

        # Compute returns
        returns = compute_returns(rewards, gamma)

        # Normalize returns (optional but helps stability)
        returns_tensor = torch.FloatTensor(returns)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)

        # Compute policy loss
        # Loss = -E[log π(a|s) * G_t]
        # Negative because we want to maximize, but optimizer minimizes
        policy_loss = []
        for log_prob, G in zip(log_probs, returns_tensor):
            policy_loss.append(-log_prob * G)

        # Optimize policy
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()

        # Track rewards
        episode_reward = sum(rewards)
        rewards_history.append(episode_reward)

        # Update running reward
        if running_reward is None:
            running_reward = episode_reward
        else:
            running_reward = 0.05 * episode_reward + 0.95 * running_reward

        # Print progress
        if episode % print_every == 0:
            print(f"Episode {episode}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Running Avg: {running_reward:.2f}")

    return rewards_history


def plot_rewards(rewards: List[float], title: str = "REINFORCE Learning Curve",
                 filename: str = "reinforce_learning_curve.png") -> None:
    """
    Plot learning curve showing episode rewards.

    Args:
        rewards: List of episode rewards
        title: Plot title
        filename: File to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Raw rewards
    ax1.plot(rewards, alpha=0.6, color='blue')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Raw Episode Rewards')
    ax1.grid(True, alpha=0.3)

    # Smoothed rewards
    window = 50
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax2.plot(smoothed, color='red', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel(f'Smoothed Reward (window={window})')
        ax2.set_title('Smoothed Learning Curve')
        ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Learning curve saved to '{filename}'")
    plt.close()


def demonstrate_policy_gradient_theorem() -> None:
    """
    Demonstrate the policy gradient theorem conceptually.

    Shows how we compute gradients of the expected return with respect
    to policy parameters.
    """
    print("=" * 60)
    print("POLICY GRADIENT THEOREM")
    print("=" * 60)

    print("\nObjective: Maximize expected return")
    print("  J(θ) = E_τ~π_θ[R(τ)]")
    print("  where τ is a trajectory (s_0, a_0, r_1, s_1, a_1, r_2, ...)")

    print("\nPolicy Gradient Theorem:")
    print("  ∇_θ J(θ) = E_τ[Σ_t ∇_θ log π_θ(a_t|s_t) * G_t]")

    print("\nKey insights:")
    print("  1. We don't need to know environment dynamics")
    print("  2. We only need to sample trajectories from π_θ")
    print("  3. Gradient points towards actions with higher returns")
    print("  4. Monte Carlo: Use full return G_t (sum of future rewards)")

    print("\nREINFORCE Algorithm:")
    print("  For each episode:")
    print("    1. Sample trajectory τ ~ π_θ")
    print("    2. For each timestep t:")
    print("         Compute G_t = Σ_{k=t}^T γ^{k-t} r_k")
    print("    3. Compute gradient: ∇_θ J ≈ Σ_t ∇_θ log π_θ(a_t|s_t) * G_t")
    print("    4. Update: θ ← θ + α * ∇_θ J")

    print("\nWhy the log probability?")
    print("  - ∇_θ log π_θ(a|s) = ∇_θ π_θ(a|s) / π_θ(a|s)")
    print("  - Makes math work out in the policy gradient theorem")
    print("  - Increases probability of actions with positive returns")
    print("  - Decreases probability of actions with negative returns")


def test_trained_policy(env: gym.Env, policy_net: PolicyNetwork,
                        num_episodes: int = 10) -> float:
    """
    Test trained policy by running episodes without training.

    Args:
        env: Gymnasium environment
        policy_net: Trained policy network
        num_episodes: Number of test episodes

    Returns:
        Average reward over test episodes
    """
    total_reward = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = policy_net.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        total_reward += episode_reward
        print(f"  Test Episode {episode + 1}: Reward = {episode_reward:.2f}")

    avg_reward = total_reward / num_episodes
    return avg_reward


def main():
    """Main function to run REINFORCE demonstrations."""
    print("\n" + "=" * 60)
    print("WEEK 9 - LESSON 2: REINFORCE ALGORITHM")
    print("Monte Carlo Policy Gradient")
    print("=" * 60)

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Explain policy gradient theorem
    demonstrate_policy_gradient_theorem()

    # 2. Train REINFORCE on CartPole
    print("\n" + "=" * 60)
    print("TRAINING REINFORCE ON CARTPOLE")
    print("=" * 60)

    env = gym.make('CartPole-v1')

    # Create policy network
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy_net = PolicyNetwork(state_size, action_size, hidden_size=128)

    # Optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

    # Train
    print("\nTraining REINFORCE...")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Hidden size: 128, Learning rate: 0.001, Gamma: 0.99")
    print()

    num_episodes = 1000
    rewards = reinforce(env, policy_net, optimizer, num_episodes=num_episodes,
                       gamma=0.99, print_every=100)

    # Plot results
    plot_rewards(rewards, "REINFORCE on CartPole-v1")

    # 3. Test trained policy
    print("\n" + "=" * 60)
    print("TESTING TRAINED POLICY")
    print("=" * 60)

    test_env = gym.make('CartPole-v1')
    avg_reward = test_trained_policy(test_env, policy_net, num_episodes=10)

    print(f"\nAverage test reward: {avg_reward:.2f}")
    if avg_reward >= 195:
        print("✓ Environment solved! (average reward >= 195)")
    else:
        print(f"✗ Not quite solved. Need {195 - avg_reward:.2f} more reward.")

    # 4. Summary
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. REINFORCE is a Monte Carlo policy gradient method")
    print("2. Uses full episode returns G_t (high variance)")
    print("3. Updates policy to increase probability of good actions")
    print("4. No bootstrapping - waits for episode to complete")
    print("5. Guaranteed to converge to local optimum")
    print("6. Main problem: High variance → slow learning")
    print("\nNext: Learn how baselines reduce variance!")
    print("=" * 60)

    env.close()
    test_env.close()


if __name__ == "__main__":
    main()
