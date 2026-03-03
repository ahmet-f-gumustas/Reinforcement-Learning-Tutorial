"""
03 - Inverse Reinforcement Learning (IRL)

This script introduces IRL, which learns a reward function from
expert demonstrations rather than directly from a reward signal.

Demonstrates:
- IRL motivation and problem formulation
- Maximum Entropy IRL concept
- Simple feature-based reward learning
- Behavioral cloning vs IRL
- Imitation learning pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict


# =============================================================================
# Environment
# =============================================================================

class SimpleGridWorld:
    """GridWorld where reward function is unknown to the learner."""

    def __init__(self, size: int = 5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.goal = (4, 4)
        self.reset()

    def reset(self) -> int:
        self.pos = (0, 0)
        return self._state_idx()

    def _state_idx(self) -> int:
        return self.pos[0] * self.size + self.pos[1]

    def get_features(self, state: int) -> np.ndarray:
        """State features for reward function approximation."""
        r, c = state // self.size, state % self.size
        goal_r, goal_c = self.goal
        dist_to_goal = abs(r - goal_r) + abs(c - goal_c)
        return np.array([
            r / self.size,            # Normalized row
            c / self.size,            # Normalized col
            dist_to_goal / (2*self.size),  # Normalized distance to goal
            1.0 if (r, c) == self.goal else 0.0,  # At goal
        ], dtype=np.float32)

    def step(self, action: int) -> Tuple[int, float, bool]:
        r, c = self.pos
        if action == 0: r = max(0, r-1)
        elif action == 1: c = min(self.size-1, c+1)
        elif action == 2: r = min(self.size-1, r+1)
        elif action == 3: c = max(0, c-1)
        self.pos = (r, c)
        done = self.pos == self.goal
        # True reward (unknown to IRL learner)
        reward = 10.0 if done else -0.1
        return self._state_idx(), reward, done


# =============================================================================
# Expert Policy
# =============================================================================

def create_expert_policy(env: SimpleGridWorld) -> np.ndarray:
    """Create optimal expert policy using value iteration."""
    V = np.zeros(env.n_states)
    gamma = 0.99

    for _ in range(200):
        V_new = np.zeros(env.n_states)
        for s in range(env.n_states):
            r, c = s // env.size, s % env.size
            q_values = []
            for a in range(env.n_actions):
                nr, nc = r, c
                if a == 0: nr = max(0, r-1)
                elif a == 1: nc = min(env.size-1, c+1)
                elif a == 2: nr = min(env.size-1, r+1)
                elif a == 3: nc = max(0, c-1)
                ns = nr * env.size + nc
                done = (nr, nc) == env.goal
                reward = 10.0 if done else -0.1
                q_values.append(reward + gamma * V[ns] * (1 - int(done)))
            V_new[s] = max(q_values)
        V = V_new

    # Extract policy
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        r, c = s // env.size, s % env.size
        q_values = []
        for a in range(env.n_actions):
            nr, nc = r, c
            if a == 0: nr = max(0, r-1)
            elif a == 1: nc = min(env.size-1, c+1)
            elif a == 2: nr = min(env.size-1, r+1)
            elif a == 3: nc = max(0, c-1)
            ns = nr * env.size + nc
            done = (nr, nc) == env.goal
            reward = 10.0 if done else -0.1
            q_values.append(reward + gamma * V[ns] * (1 - int(done)))
        policy[s] = np.argmax(q_values)

    return policy


def collect_demonstrations(env, policy, n_episodes=50, max_steps=30):
    """Collect expert demonstrations."""
    demonstrations = []
    for _ in range(n_episodes):
        trajectory = []
        state = env.reset()
        for _ in range(max_steps):
            action = policy[state]
            next_state, _, done = env.step(action)
            trajectory.append((state, action, next_state))
            state = next_state
            if done:
                break
        demonstrations.append(trajectory)
    return demonstrations


# =============================================================================
# Feature-Based IRL
# =============================================================================

class RewardNetwork(nn.Module):
    """Learned reward function R(s) = w^T φ(s)."""

    def __init__(self, feature_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def compute_feature_expectations(demonstrations, env):
    """Compute average feature expectations from demonstrations."""
    gamma = 0.99
    feature_sum = np.zeros(4)
    total = 0

    for traj in demonstrations:
        for t, (s, a, ns) in enumerate(traj):
            features = env.get_features(s)
            feature_sum += (gamma ** t) * features
            total += 1

    return feature_sum / len(demonstrations)


def train_irl(env, demonstrations, num_iterations=200, lr=0.01):
    """
    Simple IRL: learn reward weights to match expert feature expectations.

    The idea: find reward R(s) = w^T φ(s) such that the policy that
    maximizes R behaves like the expert (matches feature expectations).
    """
    expert_features = compute_feature_expectations(demonstrations, env)
    print(f"Expert feature expectations: {expert_features}")

    # Simple approach: learn weights to make expert features have high reward
    reward_weights = np.random.randn(4) * 0.1
    feature_size = 4

    for iteration in range(num_iterations):
        # Compute reward for all states
        rewards = np.zeros(env.n_states)
        for s in range(env.n_states):
            features = env.get_features(s)
            rewards[s] = np.dot(reward_weights, features)

        # Compute policy under current reward (value iteration)
        V = np.zeros(env.n_states)
        gamma = 0.99
        for _ in range(100):
            V_new = np.zeros(env.n_states)
            for s in range(env.n_states):
                r, c = s // env.size, s % env.size
                q_values = []
                for a in range(env.n_actions):
                    nr, nc = r, c
                    if a == 0: nr = max(0, r-1)
                    elif a == 1: nc = min(env.size-1, c+1)
                    elif a == 2: nr = min(env.size-1, r+1)
                    elif a == 3: nc = max(0, c-1)
                    ns = nr * env.size + nc
                    q_values.append(rewards[ns] + gamma * V[ns])
                V_new[s] = max(q_values)
            V = V_new

        # Collect trajectories under learned policy
        policy = np.zeros(env.n_states, dtype=int)
        for s in range(env.n_states):
            r, c = s // env.size, s % env.size
            q_values = []
            for a in range(env.n_actions):
                nr, nc = r, c
                if a == 0: nr = max(0, r-1)
                elif a == 1: nc = min(env.size-1, c+1)
                elif a == 2: nr = min(env.size-1, r+1)
                elif a == 3: nc = max(0, c-1)
                ns = nr * env.size + nc
                q_values.append(rewards[ns] + gamma * V[ns])
            policy[s] = np.argmax(q_values)

        learner_demos = collect_demonstrations(env, policy, n_episodes=20)
        learner_features = compute_feature_expectations(learner_demos, env)

        # Update weights: increase reward for expert features, decrease for learner
        gradient = expert_features - learner_features
        reward_weights += lr * gradient

        if (iteration + 1) % 50 == 0:
            diff = np.linalg.norm(expert_features - learner_features)
            print(f"  Iteration {iteration+1}/{num_iterations} | "
                  f"Feature diff: {diff:.4f} | Weights: {reward_weights}")

    return reward_weights


# =============================================================================
# Demonstrations
# =============================================================================

def demonstrate_irl_concepts():
    """Explain IRL concepts."""
    print("=" * 60)
    print("INVERSE REINFORCEMENT LEARNING (IRL)")
    print("=" * 60)

    print("""
    Standard RL:
        Given:  Environment dynamics, REWARD function
        Learn:  Optimal policy

    Inverse RL:
        Given:  Expert DEMONSTRATIONS (trajectories)
        Learn:  REWARD function that explains the expert behavior

    Why IRL?
    1. Hard to specify reward functions for complex tasks
       (e.g., "drive safely" - what's the exact reward?)
    2. Experts can demonstrate desired behavior easily
    3. Learned reward generalizes better than cloning behavior

    Approaches:
    ┌──────────────────────────┬────────────────────────────┐
    │ Method                    │ Description                │
    ├──────────────────────────┼────────────────────────────┤
    │ Behavioral Cloning       │ Supervised: π(a|s) from    │
    │                          │ demos (not IRL, just copy) │
    │ Feature Matching IRL     │ Match expert's feature     │
    │                          │ expectations               │
    │ MaxEnt IRL (Ziebart)     │ Maximum entropy principle  │
    │ GAIL (Ho & Ermon)        │ GAN-style imitation        │
    │ AIRL                     │ Adversarial IRL            │
    └──────────────────────────┴────────────────────────────┘

    IRL vs Behavioral Cloning:
    - BC: copies actions → fails on unseen states (distribution shift)
    - IRL: learns WHY → generalizes to new situations
    """)


def main():
    print("\n" + "=" * 60)
    print("WEEK 12 - LESSON 3: INVERSE RL")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    demonstrate_irl_concepts()

    env = SimpleGridWorld()

    # Create expert
    print("\nCreating expert policy...")
    expert_policy = create_expert_policy(env)
    print("Collecting expert demonstrations...")
    demos = collect_demonstrations(env, expert_policy, n_episodes=50)
    print(f"Collected {len(demos)} demonstrations, avg length: "
          f"{np.mean([len(t) for t in demos]):.1f}")

    # Learn reward
    print("\nLearning reward function from demonstrations...")
    learned_weights = train_irl(env, demos, num_iterations=200)

    # Visualize learned reward
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # True reward
    true_rewards = np.zeros((env.size, env.size))
    for r in range(env.size):
        for c in range(env.size):
            true_rewards[r, c] = 10.0 if (r, c) == env.goal else -0.1

    ax = axes[0]
    im = ax.imshow(true_rewards, cmap='RdYlGn')
    ax.set_title('True Reward Function')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(env.size)); ax.set_yticks(range(env.size))

    # Learned reward
    learned_rewards = np.zeros((env.size, env.size))
    for r in range(env.size):
        for c in range(env.size):
            s = r * env.size + c
            features = env.get_features(s)
            learned_rewards[r, c] = np.dot(learned_weights, features)

    ax = axes[1]
    im = ax.imshow(learned_rewards, cmap='RdYlGn')
    ax.set_title('Learned Reward Function (IRL)')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(env.size)); ax.set_yticks(range(env.size))

    plt.suptitle('Inverse RL: Reward Recovery', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('12_advanced/inverse_rl.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to '12_advanced/inverse_rl.png'")
    plt.close()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. IRL learns REWARD from demonstrations, not policy")
    print("2. Learned reward captures the 'intent' behind expert behavior")
    print("3. Feature matching: match expert's discounted feature expectations")
    print("4. IRL generalizes better than behavioral cloning")
    print("5. Modern approaches (GAIL, AIRL) use adversarial training")
    print("\nNext: Reward Shaping!")
    print("=" * 60)


if __name__ == "__main__":
    main()
