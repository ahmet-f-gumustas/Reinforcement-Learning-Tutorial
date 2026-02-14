"""
04 - Advanced Variance Reduction Techniques

This script explores various variance reduction techniques for policy gradient methods
beyond simple baseline subtraction.

Demonstrates:
- Rewards-to-go (causality principle)
- Different baseline types
- Generalized Advantage Estimation (GAE) preview
- Empirical comparison of techniques
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
    """Policy network."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.network(state)
        return F.softmax(logits, dim=-1)

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob


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


def compute_returns_full(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Compute full episode returns (original REINFORCE).

    G_t = R_{t+1} + γ*R_{t+2} + γ²*R_{t+3} + ... + γ^{T-t-1}*R_T

    This includes ALL future rewards, even those from actions taken later.
    """
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns


def compute_returns_to_go(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Compute rewards-to-go (causality principle).

    Same as full returns, but conceptually emphasizes that we only
    consider rewards that come AFTER action a_t. This is actually
    the same calculation, but we're being explicit about causality.

    Key insight: Past actions can't affect future rewards,
    so we shouldn't include past rewards in the gradient.
    """
    # This is actually the same as compute_returns_full
    # But we emphasize that we start from current timestep
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns


def reinforce_variance_technique(env: gym.Env,
                                 policy_net: PolicyNetwork,
                                 value_net: ValueNetwork,
                                 policy_optimizer: optim.Optimizer,
                                 value_optimizer: optim.Optimizer,
                                 num_episodes: int,
                                 technique: str = "full",
                                 gamma: float = 0.99) -> Tuple[List[float], List[float]]:
    """
    REINFORCE with different variance reduction techniques.

    Args:
        technique: "full" (full returns), "reward_to_go" (same as full),
                   "baseline" (with V(s)), "normalized" (normalize advantages)
    """
    rewards_history = []
    variance_history = []

    for episode in range(num_episodes):
        log_probs = []
        values = []
        rewards = []
        states_list = []

        state, _ = env.reset()
        done = False

        while not done:
            states_list.append(state)
            action, log_prob = policy_net.select_action(state)
            log_probs.append(log_prob)

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = value_net(state_tensor)
            values.append(value)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            state = next_state

        # Compute returns based on technique
        if technique == "full" or technique == "reward_to_go":
            returns = compute_returns_full(rewards, gamma)
            returns_tensor = torch.FloatTensor(returns)
            targets = returns_tensor

        elif technique == "baseline":
            returns = compute_returns_full(rewards, gamma)
            returns_tensor = torch.FloatTensor(returns)
            values_tensor = torch.cat(values).squeeze()
            targets = returns_tensor - values_tensor.detach()  # Advantage

        elif technique == "normalized":
            returns = compute_returns_full(rewards, gamma)
            returns_tensor = torch.FloatTensor(returns)
            values_tensor = torch.cat(values).squeeze()
            advantages = returns_tensor - values_tensor.detach()
            # Normalize advantages
            targets = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        else:
            raise ValueError(f"Unknown technique: {technique}")

        # Track variance
        variance_history.append(targets.var().item())

        # Policy loss
        policy_loss = []
        for log_prob, target in zip(log_probs, targets):
            policy_loss.append(-log_prob * target)

        # Update policy
        policy_optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)  # Gradient clipping
        policy_optimizer.step()

        # Update value function (for techniques that use it)
        if technique in ["baseline", "normalized"]:
            returns_tensor = torch.FloatTensor(returns)
            values_tensor = torch.cat(values).squeeze()
            value_loss = F.mse_loss(values_tensor, returns_tensor)

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

        # Track rewards
        rewards_history.append(sum(rewards))

    return rewards_history, variance_history


def compare_techniques() -> None:
    """Compare different variance reduction techniques."""
    print("=" * 60)
    print("COMPARING VARIANCE REDUCTION TECHNIQUES")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    num_episodes = 400
    num_runs = 3  # Multiple runs for statistical significance

    techniques = {
        "Full Returns": "full",
        "Baseline (V(s))": "baseline",
        "Normalized Advantage": "normalized"
    }

    results = {}

    for name, technique in techniques.items():
        print(f"\nTraining with {name}...")
        all_rewards = []
        all_variances = []

        for run in range(num_runs):
            # Create fresh networks
            policy_net = PolicyNetwork(state_size, action_size, 128)
            value_net = ValueNetwork(state_size, 128)
            policy_opt = optim.Adam(policy_net.parameters(), lr=0.002)
            value_opt = optim.Adam(value_net.parameters(), lr=0.002)

            # Set seed for reproducibility
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)

            # Train
            rewards, variances = reinforce_variance_technique(
                env, policy_net, value_net, policy_opt, value_opt,
                num_episodes, technique
            )

            all_rewards.append(rewards)
            all_variances.append(variances)

        # Average over runs
        avg_rewards = np.mean(all_rewards, axis=0)
        avg_variances = np.mean(all_variances, axis=0)

        results[name] = {
            'rewards': avg_rewards,
            'variances': avg_variances
        }

        print(f"  Final avg reward: {np.mean(avg_rewards[-50:]):.2f}")
        print(f"  Final avg variance: {np.mean(avg_variances[-50:]):.4f}")

    # Plot comparison
    plot_technique_comparison(results)

    env.close()


def plot_technique_comparison(results: dict) -> None:
    """Plot comparison of different techniques."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['blue', 'green', 'red', 'orange']
    window = 20

    # Learning curves
    for (name, data), color in zip(results.items(), colors):
        rewards = data['rewards']
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(smoothed, label=name, color=color, linewidth=2)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel(f'Smoothed Reward (window={window})')
    ax1.set_title('Learning Curves Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Variance curves
    for (name, data), color in zip(results.items(), colors):
        variances = data['variances']
        if len(variances) >= window:
            smoothed_var = np.convolve(variances, np.ones(window)/window, mode='valid')
            ax2.plot(smoothed_var, label=name, color=color, linewidth=2)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Variance')
    ax2.set_title('Gradient Variance (Lower is Better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Variance Reduction Techniques Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('variance_reduction_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to 'variance_reduction_comparison.png'")
    plt.close()


def demonstrate_causality_principle() -> None:
    """Demonstrate the causality principle in policy gradients."""
    print("\n" + "=" * 60)
    print("CAUSALITY PRINCIPLE")
    print("=" * 60)

    print("\nKey Insight: Policy gradient for action a_t should only")
    print("depend on rewards that come AFTER taking that action.")
    print("\nWhy?")
    print("  - Action a_t cannot affect rewards before time t")
    print("  - Including past rewards adds noise without information")
    print("  - Rewards-to-go: G_t = Σ_{k=t}^T γ^{k-t} r_k")

    print("\nExample episode:")
    rewards_example = [1, 1, 1, 10, 1, 1]
    print(f"  Rewards: {rewards_example}")
    print(f"  Big reward at t=3")

    returns_full = compute_returns_full(rewards_example, gamma=0.9)
    print("\n  Returns (rewards-to-go):")
    for t, (r, G) in enumerate(zip(rewards_example, returns_full)):
        print(f"    t={t}: r={r}, G_t={G:.2f}")

    print("\n  Action at t=0 gets credit for future big reward at t=3 ✓")
    print("  Action at t=4 doesn't get credit for past big reward ✓")
    print("  This is causality: only future rewards matter!")


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("WEEK 9 - LESSON 4: ADVANCED VARIANCE REDUCTION")
    print("=" * 60)

    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Explain causality
    demonstrate_causality_principle()

    # 2. Compare techniques
    compare_techniques()

    # 3. Summary
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Causality: Only use rewards-to-go, not past rewards")
    print("2. Baseline V(s) reduces variance significantly")
    print("3. Normalizing advantages improves stability")
    print("4. Gradient clipping prevents exploding gradients")
    print("5. Combination of techniques works best")
    print("\nBest practices:")
    print("  ✓ Use advantage A(s,a) = G_t - V(s)")
    print("  ✓ Normalize advantages")
    print("  ✓ Clip gradients")
    print("  ✓ Use rewards-to-go (not full episode return)")
    print("\nNext: Learn continuous action spaces!")
    print("=" * 60)


if __name__ == "__main__":
    main()
