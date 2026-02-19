"""
01 - Actor-Critic Basics: Introduction to Actor-Critic Architecture

This script introduces the Actor-Critic framework, which combines
policy-based (actor) and value-based (critic) methods.

Demonstrates:
- Actor-Critic architecture concept
- Separate actor and critic networks
- TD error as advantage estimate
- Basic Actor-Critic training on GridWorld
- Comparison with pure REINFORCE
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple


class SimpleGridWorld:
    """
    Simple 5x5 GridWorld for demonstrating actor-critic methods.

    Agent starts at (0,0), goal at (4,4).
    Obstacles at (1,1), (2,3), (3,1).
    """

    def __init__(self, size: int = 5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # Up, Right, Down, Left
        self.obstacles = [(1, 1), (2, 3), (3, 1)]
        self.goal = (size - 1, size - 1)
        self.start = (0, 0)
        self.reset()

    def reset(self) -> np.ndarray:
        self.position = self.start
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        state = np.zeros(self.n_states)
        state[self.position[0] * self.size + self.position[1]] = 1.0
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        row, col = self.position

        if action == 0:    # Up
            row = max(0, row - 1)
        elif action == 1:  # Right
            col = min(self.size - 1, col + 1)
        elif action == 2:  # Down
            row = min(self.size - 1, row + 1)
        elif action == 3:  # Left
            col = max(0, col - 1)

        new_position = (row, col)

        if new_position in self.obstacles:
            reward = -5.0
            done = False
        elif new_position == self.goal:
            reward = 10.0
            done = True
            self.position = new_position
        else:
            reward = -0.1
            done = False
            self.position = new_position

        return self._get_state(), reward, done


# =============================================================================
# Actor-Critic Architecture
# =============================================================================

class Actor(nn.Module):
    """
    Actor network: learns the policy pi(a|s).

    The actor decides WHAT ACTION to take.
    Output: probability distribution over actions.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class Critic(nn.Module):
    """
    Critic network: learns the state value function V(s).

    The critic evaluates HOW GOOD a state is.
    Output: single scalar value V(s).
    """

    def __init__(self, state_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ActorCriticCombined(nn.Module):
    """
    Combined Actor-Critic network with shared feature layers.

    Shared lower layers extract common state features.
    Separate heads output policy (actor) and value (critic).

    Architecture:
        state -> SharedFC(64) -> ReLU -> SharedFC(64) -> ReLU
                                                |
                               +----------------+----------------+
                               |                                 |
                          ActorHead(n_actions)              CriticHead(1)
                               |                                 |
                           Softmax                            Value
                           pi(a|s)                             V(s)
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        # Shared feature layers
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        # Actor head
        self.actor_head = nn.Linear(hidden_size, action_size)
        # Critic head
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        action_probs = F.softmax(self.actor_head(features), dim=-1)
        value = self.critic_head(features)
        return action_probs, value


# =============================================================================
# Training Algorithms
# =============================================================================

def train_actor_critic_separate(env, actor, critic, actor_optimizer, critic_optimizer,
                                 num_episodes: int = 500, gamma: float = 0.99,
                                 max_steps: int = 100) -> List[float]:
    """
    Train Actor-Critic with separate networks.

    The key idea:
    1. Actor selects actions based on policy pi(a|s)
    2. Critic evaluates states V(s)
    3. TD error (delta) = r + gamma*V(s') - V(s) serves as advantage
    4. Actor is updated using: -log_pi(a|s) * delta
    5. Critic is updated using: (r + gamma*V(s') - V(s))^2

    This is different from REINFORCE:
    - REINFORCE uses full episode returns G_t (Monte Carlo)
    - Actor-Critic uses TD error (bootstrapping) -> lower variance!
    """
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Actor: get action probabilities
            action_probs = actor(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Critic: get current state value
            value = critic(state_tensor)

            # Take action in environment
            next_state, reward, done = env.step(action.item())
            episode_reward += reward

            # Critic: get next state value
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            next_value = critic(next_state_tensor)

            # TD error (advantage estimate):
            # delta = r + gamma * V(s') - V(s)
            td_target = reward + gamma * next_value * (1 - int(done))
            td_error = td_target - value

            # Update Critic: minimize (td_target - V(s))^2
            critic_loss = td_error.pow(2)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Update Actor: maximize log_pi(a|s) * delta
            actor_loss = -log_prob * td_error.detach()  # detach: don't backprop through critic
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            state = next_state
            if done:
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"  Episode {episode+1}/{num_episodes} | Avg Reward: {avg:.2f}")

    return episode_rewards


def train_reinforce(env, policy_net, optimizer,
                    num_episodes: int = 500, gamma: float = 0.99,
                    max_steps: int = 100) -> List[float]:
    """
    Train with vanilla REINFORCE for comparison.
    Uses full episode returns (Monte Carlo) instead of TD error.
    """
    episode_rewards = []

    for episode in range(num_episodes):
        log_probs = []
        rewards = []
        state = env.reset()

        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy_net(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            next_state, reward, done = env.step(action.item())
            rewards.append(reward)
            state = next_state
            if done:
                break

        episode_rewards.append(sum(rewards))

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Policy gradient update
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"  Episode {episode+1}/{num_episodes} | Avg Reward: {avg:.2f}")

    return episode_rewards


# =============================================================================
# Demonstrations
# =============================================================================

def demonstrate_architecture():
    """Show the Actor-Critic architecture conceptually."""
    print("=" * 60)
    print("ACTOR-CRITIC ARCHITECTURE")
    print("=" * 60)

    print("""
    The Actor-Critic framework combines two ideas:

    ┌─────────────────────────────────────────────────────┐
    │                    Environment                       │
    │                                                     │
    │   state s_t ──────┬──────────────> reward r_t       │
    │                   │                  │               │
    │                   ▼                  ▼               │
    │              ┌─────────┐      ┌──────────┐          │
    │              │  ACTOR  │      │  CRITIC  │          │
    │              │ π(a|s)  │      │   V(s)   │          │
    │              └────┬────┘      └────┬─────┘          │
    │                   │                │                 │
    │              action a_t      TD error δ             │
    │                   │          = r + γV(s') - V(s)    │
    │                   │                │                 │
    │                   │    ┌───────────┘                 │
    │                   ▼    ▼                             │
    │              Update Actor:                           │
    │              θ += α * ∇log π(a|s) * δ               │
    └─────────────────────────────────────────────────────┘

    Actor (Policy):
    - Decides WHAT to do
    - Outputs action probabilities π(a|s)
    - Updated using TD error from critic as advantage

    Critic (Value):
    - Evaluates HOW GOOD the state is
    - Outputs state value V(s)
    - Updated using TD error (Bellman equation)

    Key Insight: TD error δ = r + γV(s') - V(s)
    - If δ > 0: outcome was BETTER than expected → increase action probability
    - If δ < 0: outcome was WORSE than expected → decrease action probability
    """)


def demonstrate_td_error():
    """Show how TD error works as an advantage estimator."""
    print("=" * 60)
    print("TD ERROR AS ADVANTAGE ESTIMATE")
    print("=" * 60)

    print("""
    Week 9 Baseline:   A(s,a) = G_t - V(s)     (Monte Carlo return)
    Actor-Critic:      A(s,a) ≈ δ = r + γV(s') - V(s)  (TD error)

    Comparison:
    ┌────────────────────────┬───────────────────────────────┐
    │     REINFORCE          │      Actor-Critic              │
    ├────────────────────────┼───────────────────────────────┤
    │ Advantage = G_t - V(s) │ Advantage ≈ r + γV(s') - V(s)│
    │ Uses full return G_t   │ Uses one-step bootstrap       │
    │ High variance          │ Lower variance                │
    │ No bias                │ Some bias (from V approx)     │
    │ Wait for episode end   │ Update every step!            │
    │ Monte Carlo estimate   │ Temporal Difference estimate  │
    └────────────────────────┴───────────────────────────────┘
    """)

    # Numerical example
    print("Numerical Example:")
    print("  State s: robot in room, V(s) = 5.0")
    print("  Action a: move right")
    print("  Reward r: +1.0")
    print("  Next state s': robot at door, V(s') = 8.0")
    print("  gamma = 0.99")
    print()

    V_s = 5.0
    r = 1.0
    V_s_next = 8.0
    gamma = 0.99

    td_error = r + gamma * V_s_next - V_s
    print(f"  TD error δ = r + γV(s') - V(s)")
    print(f"            = {r} + {gamma}×{V_s_next} - {V_s}")
    print(f"            = {td_error:.2f}")
    print(f"\n  δ > 0 means: this action was BETTER than expected!")
    print(f"  → Increase probability of 'move right' in this state")


def compare_actor_critic_vs_reinforce():
    """Train and compare Actor-Critic vs REINFORCE."""
    print("\n" + "=" * 60)
    print("COMPARISON: Actor-Critic vs REINFORCE")
    print("=" * 60)

    env = SimpleGridWorld()
    np.random.seed(42)
    torch.manual_seed(42)

    num_episodes = 500

    # Train REINFORCE
    print("\nTraining REINFORCE...")
    policy_net = Actor(env.n_states, env.n_actions)
    reinforce_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    reinforce_rewards = train_reinforce(env, policy_net, reinforce_optimizer,
                                         num_episodes=num_episodes)

    # Train Actor-Critic
    print("\nTraining Actor-Critic...")
    torch.manual_seed(42)
    actor = Actor(env.n_states, env.n_actions)
    critic = Critic(env.n_states)
    actor_opt = optim.Adam(actor.parameters(), lr=0.001)
    critic_opt = optim.Adam(critic.parameters(), lr=0.001)
    ac_rewards = train_actor_critic_separate(env, actor, critic, actor_opt, critic_opt,
                                              num_episodes=num_episodes)

    # Plot comparison
    window = 50
    reinforce_smooth = [np.mean(reinforce_rewards[max(0,i-window):i+1])
                        for i in range(len(reinforce_rewards))]
    ac_smooth = [np.mean(ac_rewards[max(0,i-window):i+1])
                 for i in range(len(ac_rewards))]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(reinforce_smooth, label='REINFORCE', alpha=0.8)
    plt.plot(ac_smooth, label='Actor-Critic', alpha=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    final_reinforce = reinforce_rewards[-100:]
    final_ac = ac_rewards[-100:]
    plt.bar(['REINFORCE', 'Actor-Critic'],
            [np.mean(final_reinforce), np.mean(final_ac)],
            yerr=[np.std(final_reinforce), np.std(final_ac)],
            capsize=10, color=['steelblue', 'coral'])
    plt.ylabel('Average Reward (last 100 episodes)')
    plt.title('Final Performance')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('10_actor_critic/actor_critic_basics.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to '10_actor_critic/actor_critic_basics.png'")
    plt.close()


def main():
    print("\n" + "=" * 60)
    print("WEEK 10 - LESSON 1: ACTOR-CRITIC BASICS")
    print("Introduction to the Actor-Critic Framework")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Show architecture
    demonstrate_architecture()

    # 2. Explain TD error as advantage
    demonstrate_td_error()

    # 3. Compare with REINFORCE
    compare_actor_critic_vs_reinforce()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Actor-Critic = Actor (policy) + Critic (value function)")
    print("2. Actor decides actions, Critic evaluates states")
    print("3. TD error δ = r + γV(s') - V(s) serves as advantage")
    print("4. Lower variance than REINFORCE (bootstrapping vs Monte Carlo)")
    print("5. Can update every step (no need to wait for episode end)")
    print("6. Slight bias from value function approximation")
    print("\nNext: A2C - Advantage Actor-Critic with batch updates!")
    print("=" * 60)


if __name__ == "__main__":
    main()
