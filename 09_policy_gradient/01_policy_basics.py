"""
01 - Policy Basics: Introduction to Policy-Based Methods

This script introduces the fundamental concepts of policy-based reinforcement learning,
contrasting them with value-based methods from previous weeks.

Demonstrates:
- Direct policy representation using neural networks
- Stochastic policies with softmax output
- Comparison between value-based and policy-based approaches
- Simple policy network architecture
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class SimpleGridWorld:
    """
    Simple 4x4 GridWorld for demonstrating policy-based methods.

    The agent starts at (0,0) and needs to reach the goal at (3,3).
    Obstacles are placed at (1,1) and (2,2).
    """

    def __init__(self, size: int = 4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # Up, Right, Down, Left
        self.obstacles = [(1, 1), (2, 2)]
        self.goal = (3, 3)
        self.start = (0, 0)
        self.reset()

    def reset(self) -> int:
        """Reset environment to starting state."""
        self.position = self.start
        return self._position_to_state(self.position)

    def _position_to_state(self, position: Tuple[int, int]) -> int:
        """Convert (row, col) position to state index."""
        return position[0] * self.size + position[1]

    def _state_to_position(self, state: int) -> Tuple[int, int]:
        """Convert state index to (row, col) position."""
        return (state // self.size, state % self.size)

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Execute action and return next state, reward, done.

        Args:
            action: 0=Up, 1=Right, 2=Down, 3=Left

        Returns:
            next_state: Next state index
            reward: Reward received
            done: Whether episode is complete
        """
        row, col = self.position

        # Apply action
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Right
            col = min(self.size - 1, col + 1)
        elif action == 2:  # Down
            row = min(self.size - 1, row + 1)
        elif action == 3:  # Left
            col = max(0, col - 1)

        new_position = (row, col)

        # Check obstacles
        if new_position in self.obstacles:
            reward = -10.0
            done = False
            # Stay in current position
        elif new_position == self.goal:
            reward = 10.0
            done = True
            self.position = new_position
        else:
            reward = -0.1
            done = False
            self.position = new_position

        return self._position_to_state(self.position), reward, done


class PolicyNetwork(nn.Module):
    """
    Simple policy network that outputs action probabilities.

    This network directly represents a stochastic policy π(a|s).
    Unlike value-based methods (Q-learning, DQN) which learn Q(s,a),
    this network learns to output a probability distribution over actions.

    Architecture:
        state → FC(64) → ReLU → FC(64) → ReLU → FC(n_actions) → Softmax → π(a|s)
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute action probabilities.

        Args:
            state: State tensor of shape (batch_size, state_size)

        Returns:
            Action probabilities of shape (batch_size, action_size)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        # Softmax to get valid probability distribution
        return F.softmax(logits, dim=-1)

    def select_action(self, state: np.ndarray) -> int:
        """
        Select action by sampling from the policy distribution.

        This is different from value-based methods where we select
        argmax Q(s,a). Here we sample from π(a|s).

        Args:
            state: Current state as numpy array

        Returns:
            Selected action index
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.forward(state_tensor)

        # Sample action from probability distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()


class ValueNetwork(nn.Module):
    """
    Value network for comparison (Q-learning style).

    This represents a value-based approach where we learn Q(s,a)
    and select actions greedily.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No softmax - these are Q-values

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.fc3.out_features)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(state_tensor)
        return q_values.argmax().item()


def state_to_onehot(state: int, n_states: int) -> np.ndarray:
    """Convert state index to one-hot encoded vector."""
    onehot = np.zeros(n_states)
    onehot[state] = 1.0
    return onehot


def visualize_policy(policy_net: PolicyNetwork, env: SimpleGridWorld,
                     title: str = "Learned Policy") -> None:
    """
    Visualize the learned policy as action probabilities on a grid.

    Args:
        policy_net: Trained policy network
        env: GridWorld environment
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    action_names = ['Up', 'Right', 'Down', 'Left']

    for action_idx, (ax, action_name) in enumerate(zip(axes.flat, action_names)):
        probs = np.zeros((env.size, env.size))

        for row in range(env.size):
            for col in range(env.size):
                state = row * env.size + col
                state_vec = state_to_onehot(state, env.n_states)
                state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)

                with torch.no_grad():
                    action_probs = policy_net(state_tensor)
                probs[row, col] = action_probs[0, action_idx].item()

        im = ax.imshow(probs, cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_title(f'P({action_name}|s)', fontsize=12)
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))

        # Mark special positions
        for obs in env.obstacles:
            ax.add_patch(plt.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1,
                                       fill=False, edgecolor='black', linewidth=3))
        ax.plot(env.goal[1], env.goal[0], 'g*', markersize=20)
        ax.plot(env.start[1], env.start[0], 'bo', markersize=10)

        plt.colorbar(im, ax=ax)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('policy_basics_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'policy_basics_visualization.png'")
    plt.close()


def demonstrate_stochastic_policy() -> None:
    """
    Demonstrate the stochastic nature of policy-based methods.

    Show how the same policy can take different actions in the same state,
    unlike deterministic value-based policies.
    """
    print("=" * 60)
    print("DEMONSTRATION: Stochastic Policy Behavior")
    print("=" * 60)

    env = SimpleGridWorld()
    policy_net = PolicyNetwork(env.n_states, env.n_actions)

    # Initialize with random weights
    state = env.reset()
    state_vec = state_to_onehot(state, env.n_states)

    print(f"\nStarting from state {state} (position {env.start})")
    print("Policy outputs (action probabilities):")

    state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
    with torch.no_grad():
        action_probs = policy_net(state_tensor)

    action_names = ['Up', 'Right', 'Down', 'Left']
    for i, (name, prob) in enumerate(zip(action_names, action_probs[0])):
        print(f"  {name:6s}: {prob:.3f}")

    print("\nSampling 20 actions from this policy:")
    actions_taken = []
    for _ in range(20):
        action = policy_net.select_action(state_vec)
        actions_taken.append(action_names[action])

    print("Actions:", ', '.join(actions_taken))
    print("\nKey Insight:")
    print("  - Policy-based: Samples from probability distribution π(a|s)")
    print("  - Value-based: Always takes argmax Q(s,a) (deterministic)")
    print("  - Stochasticity enables better exploration naturally")


def compare_representations() -> None:
    """
    Compare policy-based vs value-based representations.

    Shows the key architectural differences between the two approaches.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: Policy-Based vs Value-Based Methods")
    print("=" * 60)

    env = SimpleGridWorld()

    print("\nPolicy-Based Method (e.g., REINFORCE):")
    print("  - Network output: Action probabilities π(a|s)")
    print("  - Output activation: Softmax (ensures Σπ(a|s) = 1)")
    print("  - Action selection: Sample from π(a|s)")
    print("  - Learning target: Maximize expected return")

    policy_net = PolicyNetwork(env.n_states, env.n_actions)
    dummy_state = torch.randn(1, env.n_states)
    policy_output = policy_net(dummy_state)
    print(f"  - Example output: {policy_output[0].detach().numpy()}")
    print(f"  - Sum of probabilities: {policy_output.sum().item():.4f}")

    print("\nValue-Based Method (e.g., DQN):")
    print("  - Network output: Q-values Q(s,a)")
    print("  - Output activation: None (Q-values can be any real number)")
    print("  - Action selection: argmax Q(s,a) with ε-greedy")
    print("  - Learning target: Bellman equation Q* ")

    value_net = ValueNetwork(env.n_states, env.n_actions)
    value_output = value_net(dummy_state)
    print(f"  - Example output: {value_output[0].detach().numpy()}")
    print(f"  - These are Q-values, not probabilities")

    print("\nKey Differences:")
    print("  ✓ Policy methods: Learn π directly")
    print("  ✓ Value methods: Learn Q, derive π implicitly")
    print("  ✓ Policy methods: Handle continuous actions naturally")
    print("  ✓ Value methods: Better sample efficiency")
    print("  ✓ Policy methods: Guaranteed convergence")
    print("  ✓ Value methods: Can be off-policy")


def demonstrate_policy_improvement() -> None:
    """
    Show how policy can be improved through random search.

    This is a simplified demonstration - in practice we use gradient descent.
    """
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Policy Improvement")
    print("=" * 60)

    env = SimpleGridWorld()

    def evaluate_policy(policy_net: PolicyNetwork, n_episodes: int = 100) -> float:
        """Evaluate policy by running episodes."""
        total_reward = 0
        for _ in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            for _ in range(50):  # Max 50 steps
                state_vec = state_to_onehot(state, env.n_states)
                action = policy_net.select_action(state_vec)
                next_state, reward, done = env.step(action)
                episode_reward += reward
                state = next_state
                if done:
                    break
            total_reward += episode_reward
        return total_reward / n_episodes

    # Try multiple random policies
    print("\nTrying 10 random policies...")
    best_policy = None
    best_reward = float('-inf')
    rewards = []

    for i in range(10):
        policy = PolicyNetwork(env.n_states, env.n_actions)
        avg_reward = evaluate_policy(policy)
        rewards.append(avg_reward)

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_policy = policy

        print(f"  Policy {i+1}: Avg Reward = {avg_reward:6.2f}")

    print(f"\nBest policy found: Avg Reward = {best_reward:.2f}")
    print("Note: This is random search. Real algorithms use gradient descent!")

    # Visualize best policy
    visualize_policy(best_policy, env, "Best Random Policy")


def main():
    """Main function to run all demonstrations."""
    print("\n" + "=" * 60)
    print("WEEK 9 - LESSON 1: POLICY BASICS")
    print("Introduction to Policy-Based Reinforcement Learning")
    print("=" * 60)

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Demonstrate stochastic policy behavior
    demonstrate_stochastic_policy()

    # 2. Compare policy-based vs value-based representations
    compare_representations()

    # 3. Show policy improvement concept
    demonstrate_policy_improvement()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Policy-based methods learn π(a|s) directly")
    print("2. Output is a probability distribution over actions")
    print("3. Actions are sampled stochastically (enables exploration)")
    print("4. Different from value-based methods which learn Q(s,a)")
    print("5. Policy gradient will improve policies through gradient descent")
    print("\nNext: Learn the REINFORCE algorithm for policy optimization!")
    print("=" * 60)


if __name__ == "__main__":
    main()
