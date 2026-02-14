"""
03 - Baseline and Variance Reduction

This script demonstrates how adding a baseline reduces variance in policy gradients.
The baseline is typically the state value function V(s), and we use the advantage
A(s,a) = G_t - V(s) instead of raw returns.

Demonstrates:
- Problem of high variance in vanilla REINFORCE
- State value baseline V(s)
- Advantage estimation: A(s,a) = G_t - V(s)
- Comparison: with vs without baseline
- Variance reduction visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple


class PolicyNetwork(nn.Module):
    """Policy network that outputs action probabilities."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob


class ValueNetwork(nn.Module):
    """
    Value network that estimates V(s) - the baseline.

    This network learns to predict the expected return from each state,
    which serves as a baseline to reduce variance.
    """

    def __init__(self, state_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Output single value V(s)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute state value V(s).

        Returns:
            State value of shape (batch_size, 1)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """Compute discounted returns G_t."""
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns


def reinforce_with_baseline(env: gym.Env,
                            policy_net: PolicyNetwork,
                            value_net: ValueNetwork,
                            policy_optimizer: optim.Optimizer,
                            value_optimizer: optim.Optimizer,
                            num_episodes: int = 1000,
                            gamma: float = 0.99,
                            print_every: int = 100) -> Tuple[List[float], List[float]]:
    """
    REINFORCE algorithm with baseline.

    The baseline V(s) reduces variance without introducing bias:
    - Policy gradient: ∇J(θ) = E[∇log π(a|s) * (G_t - V(s))]
    - Advantage: A(s,a) = G_t - V(s)

    Args:
        env: Gymnasium environment
        policy_net: Policy network
        value_net: Value network (baseline)
        policy_optimizer: Optimizer for policy
        value_optimizer: Optimizer for value function
        num_episodes: Number of training episodes
        gamma: Discount factor
        print_every: Print frequency

    Returns:
        rewards_history: Episode rewards
        variance_history: Variance of advantages per episode
    """
    rewards_history = []
    variance_history = []
    running_reward = None

    for episode in range(1, num_episodes + 1):
        log_probs = []
        values = []
        rewards = []
        states_list = []

        # Generate episode
        state, _ = env.reset()
        done = False

        while not done:
            states_list.append(state)

            # Get action from policy
            action, log_prob = policy_net.select_action(state)
            log_probs.append(log_prob)

            # Get value estimate
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = value_net(state_tensor)
            values.append(value)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)

            state = next_state

        # Compute returns
        returns = compute_returns(rewards, gamma)
        returns_tensor = torch.FloatTensor(returns)

        # Compute advantages: A(s,a) = G_t - V(s)
        values_tensor = torch.cat(values).squeeze()
        advantages = returns_tensor - values_tensor.detach()

        # Normalize advantages (improves stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        # Track variance
        variance_history.append(advantages.var().item())

        # Policy loss: -E[log π(a|s) * A(s,a)]
        policy_loss = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        policy_loss = torch.stack(policy_loss).sum()

        # Value loss: MSE between predicted V(s) and actual returns G_t
        value_loss = F.mse_loss(values_tensor, returns_tensor)

        # Optimize policy
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Optimize value function
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
                  f"Running Avg: {running_reward:.2f} | "
                  f"Adv Variance: {variance_history[-1]:.4f}")

    return rewards_history, variance_history


def reinforce_without_baseline(env: gym.Env,
                               policy_net: PolicyNetwork,
                               policy_optimizer: optim.Optimizer,
                               num_episodes: int = 1000,
                               gamma: float = 0.99,
                               print_every: int = 100) -> Tuple[List[float], List[float]]:
    """
    Vanilla REINFORCE without baseline (for comparison).

    Returns:
        rewards_history: Episode rewards
        variance_history: Variance of returns per episode
    """
    rewards_history = []
    variance_history = []
    running_reward = None

    for episode in range(1, num_episodes + 1):
        log_probs = []
        rewards = []

        # Generate episode
        state, _ = env.reset()
        done = False

        while not done:
            action, log_prob = policy_net.select_action(state)
            log_probs.append(log_prob)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)

            state = next_state

        # Compute returns
        returns = compute_returns(rewards, gamma)
        returns_tensor = torch.FloatTensor(returns)

        # Normalize returns
        returns_normalized = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)

        # Track variance
        variance_history.append(returns_normalized.var().item())

        # Policy loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns_normalized):
            policy_loss.append(-log_prob * G)

        # Optimize
        policy_optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        policy_optimizer.step()

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
                  f"Running Avg: {running_reward:.2f} | "
                  f"Return Variance: {variance_history[-1]:.4f}")

    return rewards_history, variance_history


def plot_comparison(rewards_baseline: List[float],
                   rewards_no_baseline: List[float],
                   variance_baseline: List[float],
                   variance_no_baseline: List[float]) -> None:
    """
    Plot comparison between REINFORCE with and without baseline.

    Args:
        rewards_baseline: Rewards with baseline
        rewards_no_baseline: Rewards without baseline
        variance_baseline: Advantage variance with baseline
        variance_no_baseline: Return variance without baseline
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Learning curves
    ax = axes[0, 0]
    window = 50
    if len(rewards_baseline) >= window:
        smoothed_baseline = np.convolve(rewards_baseline, np.ones(window)/window, mode='valid')
        smoothed_no_baseline = np.convolve(rewards_no_baseline, np.ones(window)/window, mode='valid')

        ax.plot(smoothed_baseline, label='With Baseline', color='green', linewidth=2)
        ax.plot(smoothed_no_baseline, label='Without Baseline', color='red', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel(f'Smoothed Reward (window={window})')
        ax.set_title('Learning Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Variance comparison
    ax = axes[0, 1]
    window = 50
    if len(variance_baseline) >= window:
        var_smooth_baseline = np.convolve(variance_baseline, np.ones(window)/window, mode='valid')
        var_smooth_no_baseline = np.convolve(variance_no_baseline, np.ones(window)/window, mode='valid')

        ax.plot(var_smooth_baseline, label='Advantage Var (with baseline)', color='green', linewidth=2)
        ax.plot(var_smooth_no_baseline, label='Return Var (no baseline)', color='red', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Variance')
        ax.set_title('Variance Comparison (Lower is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3. Raw rewards histogram
    ax = axes[1, 0]
    ax.hist(rewards_baseline, bins=30, alpha=0.5, label='With Baseline', color='green')
    ax.hist(rewards_no_baseline, bins=30, alpha=0.5, label='Without Baseline', color='red')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Cumulative rewards
    ax = axes[1, 1]
    cumsum_baseline = np.cumsum(rewards_baseline)
    cumsum_no_baseline = np.cumsum(rewards_no_baseline)

    ax.plot(cumsum_baseline, label='With Baseline', color='green', linewidth=2)
    ax.plot(cumsum_no_baseline, label='Without Baseline', color='red', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('REINFORCE: With vs Without Baseline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('baseline_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison plot saved to 'baseline_comparison.png'")
    plt.close()


def main():
    """Main function to demonstrate baseline effects."""
    print("\n" + "=" * 60)
    print("WEEK 9 - LESSON 3: BASELINE AND VARIANCE REDUCTION")
    print("Improving REINFORCE with Value Function Baseline")
    print("=" * 60)

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Explain the concept
    print("\n" + "=" * 60)
    print("THE VARIANCE PROBLEM")
    print("=" * 60)
    print("\nVanilla REINFORCE suffers from HIGH VARIANCE:")
    print("  - Different episodes have very different returns")
    print("  - Gradient estimates are noisy")
    print("  - Learning is slow and unstable")
    print("\nSOLUTION: Use a baseline b(s)")
    print("  - Modify gradient: ∇J = E[∇log π(a|s) * (G_t - b(s))]")
    print("  - Baseline reduces variance WITHOUT introducing bias")
    print("  - Common choice: b(s) = V(s) (state value function)")
    print("\nAdvantage Function:")
    print("  A(s,a) = G_t - V(s)")
    print("  - Measures how much better action a is than average")
    print("  - Positive advantage → increase probability")
    print("  - Negative advantage → decrease probability")

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    num_episodes = 500

    # 1. Train WITH baseline
    print("\n" + "=" * 60)
    print("TRAINING WITH BASELINE")
    print("=" * 60)

    policy_net_baseline = PolicyNetwork(state_size, action_size, 128)
    value_net = ValueNetwork(state_size, 128)
    policy_opt_baseline = optim.Adam(policy_net_baseline.parameters(), lr=0.001)
    value_opt = optim.Adam(value_net.parameters(), lr=0.001)

    print("\nTraining REINFORCE with baseline...")
    rewards_baseline, var_baseline = reinforce_with_baseline(
        env, policy_net_baseline, value_net, policy_opt_baseline, value_opt,
        num_episodes=num_episodes, print_every=100
    )

    # 2. Train WITHOUT baseline
    print("\n" + "=" * 60)
    print("TRAINING WITHOUT BASELINE")
    print("=" * 60)

    env2 = gym.make('CartPole-v1')
    policy_net_no_baseline = PolicyNetwork(state_size, action_size, 128)
    policy_opt_no_baseline = optim.Adam(policy_net_no_baseline.parameters(), lr=0.001)

    print("\nTraining vanilla REINFORCE (no baseline)...")
    rewards_no_baseline, var_no_baseline = reinforce_without_baseline(
        env2, policy_net_no_baseline, policy_opt_no_baseline,
        num_episodes=num_episodes, print_every=100
    )

    # 3. Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    avg_reward_baseline = np.mean(rewards_baseline[-100:])
    avg_reward_no_baseline = np.mean(rewards_no_baseline[-100:])
    avg_var_baseline = np.mean(var_baseline[-100:])
    avg_var_no_baseline = np.mean(var_no_baseline[-100:])

    print(f"\nFinal 100 episodes average:")
    print(f"  With Baseline:    Reward = {avg_reward_baseline:.2f}, Variance = {avg_var_baseline:.4f}")
    print(f"  Without Baseline: Reward = {avg_reward_no_baseline:.2f}, Variance = {avg_var_no_baseline:.4f}")

    improvement = ((avg_reward_baseline - avg_reward_no_baseline) / abs(avg_reward_no_baseline)) * 100
    var_reduction = ((avg_var_no_baseline - avg_var_baseline) / avg_var_no_baseline) * 100

    print(f"\nImprovement with baseline:")
    print(f"  Reward: {improvement:+.1f}%")
    print(f"  Variance reduction: {var_reduction:.1f}%")

    # Plot comparison
    plot_comparison(rewards_baseline, rewards_no_baseline, var_baseline, var_no_baseline)

    # 4. Summary
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Baseline reduces variance without adding bias")
    print("2. Common baseline: V(s) learned with separate network")
    print("3. Advantage A(s,a) = G_t - V(s) used for policy update")
    print("4. Lower variance → faster, more stable learning")
    print("5. Baseline is critical for policy gradient methods")
    print("\nNext: Explore more variance reduction techniques!")
    print("=" * 60)

    env.close()
    env2.close()


if __name__ == "__main__":
    main()
