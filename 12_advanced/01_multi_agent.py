"""
01 - Multi-Agent Reinforcement Learning (MARL)

This script introduces multi-agent RL concepts where multiple agents
learn and interact in a shared environment.

Demonstrates:
- Cooperative vs competitive vs mixed settings
- Independent learners (IQL)
- Centralized training, decentralized execution (CTDE)
- Simple predator-prey simulation
- Communication between agents
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict


# =============================================================================
# Multi-Agent Environment
# =============================================================================

class PredatorPreyGrid:
    """
    Simple Predator-Prey grid environment.

    Two predator agents try to catch a prey agent.
    Prey moves randomly. Predators learn to cooperate.

    Grid: NxN, predators get reward when both adjacent to prey.
    """

    def __init__(self, size: int = 7):
        self.size = size
        self.n_actions = 5  # Up, Right, Down, Left, Stay
        self.n_agents = 2   # Two predators
        self.reset()

    def reset(self) -> List[np.ndarray]:
        """Reset and return observations for each agent."""
        # Random positions (ensure no overlap)
        positions = set()
        while len(positions) < 3:
            positions.add((np.random.randint(self.size), np.random.randint(self.size)))
        positions = list(positions)
        self.predator_pos = [positions[0], positions[1]]
        self.prey_pos = positions[2]
        return self._get_observations()

    def _get_observations(self) -> List[np.ndarray]:
        """Each agent observes: own position + prey position + other agent position."""
        obs = []
        for i in range(self.n_agents):
            other = 1 - i
            ob = np.array([
                self.predator_pos[i][0] / self.size,
                self.predator_pos[i][1] / self.size,
                self.prey_pos[0] / self.size,
                self.prey_pos[1] / self.size,
                self.predator_pos[other][0] / self.size,
                self.predator_pos[other][1] / self.size,
            ], dtype=np.float32)
            obs.append(ob)
        return obs

    def _move(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        r, c = pos
        if action == 0: r = max(0, r - 1)       # Up
        elif action == 1: c = min(self.size-1, c + 1)  # Right
        elif action == 2: r = min(self.size-1, r + 1)  # Down
        elif action == 3: c = max(0, c - 1)      # Left
        # action == 4: Stay
        return (r, c)

    def _manhattan(self, p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool]:
        """Execute actions for all agents."""
        # Move predators
        for i in range(self.n_agents):
            self.predator_pos[i] = self._move(self.predator_pos[i], actions[i])

        # Move prey (random)
        prey_action = np.random.randint(self.n_actions)
        self.prey_pos = self._move(self.prey_pos, prey_action)

        # Check capture: both predators adjacent (manhattan distance ≤ 1) to prey
        d0 = self._manhattan(self.predator_pos[0], self.prey_pos)
        d1 = self._manhattan(self.predator_pos[1], self.prey_pos)
        caught = (d0 <= 1) and (d1 <= 1)

        # Rewards
        if caught:
            rewards = [10.0, 10.0]  # Cooperative reward
            done = True
        else:
            # Small penalty per step + shaping reward for getting closer
            rewards = []
            for i in range(self.n_agents):
                dist = self._manhattan(self.predator_pos[i], self.prey_pos)
                rewards.append(-0.1 - 0.05 * dist)
            done = False

        return self._get_observations(), rewards, done


# =============================================================================
# Independent Q-Learning (IQL) Agents
# =============================================================================

class IQLAgent:
    """
    Independent Q-Learning agent.

    Each agent learns independently, treating other agents as part of
    the environment. Simple but can fail in complex cooperative tasks.
    """

    def __init__(self, obs_size: int, n_actions: int,
                 lr: float = 0.001, gamma: float = 0.99, epsilon: float = 1.0):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.05

        self.q_network = nn.Sequential(
            nn.Linear(obs_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            q = self.q_network(torch.FloatTensor(obs).unsqueeze(0))
        return q.argmax().item()

    def update(self, obs, action, reward, next_obs, done):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0)

        q_val = self.q_network(obs_t)[0, action]
        with torch.no_grad():
            next_q = self.q_network(next_obs_t).max(1)[0]
            target = reward + self.gamma * next_q * (1 - int(done))

        loss = F.mse_loss(q_val, target.squeeze())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# =============================================================================
# Centralized Critic with Decentralized Actors (CTDE)
# =============================================================================

class CTDEAgent:
    """
    Centralized Training Decentralized Execution (CTDE).

    During training: critic sees all observations (centralized)
    During execution: each actor only sees own observation (decentralized)
    """

    def __init__(self, obs_size: int, global_obs_size: int, n_actions: int,
                 lr: float = 0.001, gamma: float = 0.99):
        self.n_actions = n_actions
        self.gamma = gamma

        # Decentralized actor (local observation only)
        self.actor = nn.Sequential(
            nn.Linear(obs_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions)
        )

        # Centralized critic (sees all observations)
        self.critic = nn.Sequential(
            nn.Linear(global_obs_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, local_obs: np.ndarray) -> Tuple[int, torch.Tensor]:
        logits = self.actor(torch.FloatTensor(local_obs).unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, local_obs, global_obs, log_prob, reward, next_global_obs, done):
        global_t = torch.FloatTensor(global_obs).unsqueeze(0)
        next_global_t = torch.FloatTensor(next_global_obs).unsqueeze(0)

        value = self.critic(global_t)
        with torch.no_grad():
            next_value = self.critic(next_global_t)
            td_target = reward + self.gamma * next_value * (1 - int(done))

        # Critic update
        critic_loss = F.mse_loss(value, td_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update using advantage
        with torch.no_grad():
            advantage = td_target - self.critic(global_t)
        actor_loss = -log_prob * advantage.squeeze()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()


# =============================================================================
# Training
# =============================================================================

def train_iql(num_episodes: int = 2000) -> List[float]:
    """Train Independent Q-Learning agents."""
    env = PredatorPreyGrid(size=7)
    agents = [IQLAgent(6, env.n_actions) for _ in range(env.n_agents)]
    episode_rewards = []

    for ep in range(num_episodes):
        obs_list = env.reset()
        total_reward = 0

        for step in range(50):
            actions = [agents[i].select_action(obs_list[i]) for i in range(env.n_agents)]
            next_obs_list, rewards, done = env.step(actions)

            for i in range(env.n_agents):
                agents[i].update(obs_list[i], actions[i], rewards[i],
                                next_obs_list[i], done)

            total_reward += sum(rewards)
            obs_list = next_obs_list
            if done:
                break

        episode_rewards.append(total_reward)
        if (ep + 1) % 500 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"  IQL Episode {ep+1}/{num_episodes} | Avg Reward: {avg:.2f}")

    return episode_rewards


def train_ctde(num_episodes: int = 2000) -> List[float]:
    """Train CTDE agents."""
    env = PredatorPreyGrid(size=7)
    obs_size = 6
    global_obs_size = obs_size * env.n_agents
    agents = [CTDEAgent(obs_size, global_obs_size, env.n_actions)
              for _ in range(env.n_agents)]
    episode_rewards = []

    for ep in range(num_episodes):
        obs_list = env.reset()
        total_reward = 0

        for step in range(50):
            global_obs = np.concatenate(obs_list)
            actions = []
            log_probs = []
            for i in range(env.n_agents):
                action, lp = agents[i].select_action(obs_list[i])
                actions.append(action)
                log_probs.append(lp)

            next_obs_list, rewards, done = env.step(actions)
            next_global_obs = np.concatenate(next_obs_list)

            for i in range(env.n_agents):
                agents[i].update(obs_list[i], global_obs, log_probs[i],
                                rewards[i], next_global_obs, done)

            total_reward += sum(rewards)
            obs_list = next_obs_list
            if done:
                break

        episode_rewards.append(total_reward)
        if (ep + 1) % 500 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"  CTDE Episode {ep+1}/{num_episodes} | Avg Reward: {avg:.2f}")

    return episode_rewards


# =============================================================================
# Demonstrations
# =============================================================================

def demonstrate_marl_concepts():
    """Explain multi-agent RL settings."""
    print("=" * 60)
    print("MULTI-AGENT REINFORCEMENT LEARNING (MARL)")
    print("=" * 60)

    print("""
    Multi-Agent Settings:

    1. COOPERATIVE (Team)
       All agents share the same goal.
       Examples: Robot swarm, team sports, warehouse robots
       Challenge: Credit assignment (who contributed to success?)

    2. COMPETITIVE (Adversarial)
       Agents have opposing goals (zero-sum game).
       Examples: Chess, Go, poker, predator vs prey
       Challenge: Non-stationarity (opponents change strategies)

    3. MIXED (General-Sum)
       Some cooperation, some competition.
       Examples: Traffic, market economy, social dilemmas
       Challenge: Both credit assignment AND non-stationarity

    Key Approaches:
    ┌─────────────────────┬────────────────────────────────┐
    │ Approach             │ Description                    │
    ├─────────────────────┼────────────────────────────────┤
    │ IQL                 │ Each agent learns independently │
    │ CTDE                │ Central critic, local actors   │
    │ QMIX               │ Factored Q-values              │
    │ MADDPG              │ Multi-agent DDPG               │
    │ MAPPO               │ Multi-agent PPO                │
    │ Communication       │ Agents learn to communicate    │
    └─────────────────────┴────────────────────────────────┘

    The non-stationarity problem:
    From agent i's perspective, the environment includes other agents.
    When other agents learn and change their policies, the environment
    appears non-stationary → violates Markov assumption!
    """)


def main():
    print("\n" + "=" * 60)
    print("WEEK 12 - LESSON 1: MULTI-AGENT RL")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    demonstrate_marl_concepts()

    # Train IQL
    print("\nTraining Independent Q-Learning (IQL)...")
    iql_rewards = train_iql(num_episodes=2000)

    # Train CTDE
    print("\nTraining CTDE (Centralized Training, Decentralized Execution)...")
    torch.manual_seed(42); np.random.seed(42)
    ctde_rewards = train_ctde(num_episodes=2000)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    window = 100

    ax = axes[0]
    for name, rewards, color in [("IQL", iql_rewards, 'steelblue'),
                                   ("CTDE", ctde_rewards, 'coral')]:
        sm = [np.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
        ax.plot(sm, label=name, color=color, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('IQL vs CTDE on Predator-Prey')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(['IQL', 'CTDE'],
           [np.mean(iql_rewards[-200:]), np.mean(ctde_rewards[-200:])],
           yerr=[np.std(iql_rewards[-200:]), np.std(ctde_rewards[-200:])],
           capsize=10, color=['steelblue', 'coral'])
    ax.set_ylabel('Avg Reward (last 200 episodes)')
    ax.set_title('Final Performance')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Multi-Agent RL: Predator-Prey', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('12_advanced/multi_agent.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to '12_advanced/multi_agent.png'")
    plt.close()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. MARL has cooperative, competitive, and mixed settings")
    print("2. IQL: simple but ignores other agents → limited cooperation")
    print("3. CTDE: centralized critic enables better coordination")
    print("4. Non-stationarity is the core challenge in MARL")
    print("5. Credit assignment: which agent contributed to team reward?")
    print("\nNext: Model-based RL!")
    print("=" * 60)


if __name__ == "__main__":
    main()
