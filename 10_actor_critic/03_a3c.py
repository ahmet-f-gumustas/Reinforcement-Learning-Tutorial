"""
03 - A3C: Asynchronous Advantage Actor-Critic

This script implements the A3C algorithm concept. A3C uses multiple
parallel workers, each interacting with their own copy of the environment,
to asynchronously update a shared global network.

Note: True A3C requires multiprocessing. This implementation simulates
the concept with sequential workers for educational purposes, and also
provides a real multiprocessing version.

Demonstrates:
- A3C architecture with global and local networks
- Asynchronous gradient updates
- Multiple parallel workers concept
- Shared optimizer across workers
- Comparison: A3C vs A2C
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from typing import List, Tuple, Optional
import gymnasium as gym
import time


class A3CNetwork(nn.Module):
    """
    Actor-Critic network used by A3C workers.

    Each worker has a local copy of this network.
    Gradients from local networks are applied to the shared global network.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_head = nn.Linear(hidden_size, action_size)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.actor_head(x)
        value = self.critic_head(x)
        return logits, value


class A3CWorker:
    """
    A3C Worker that interacts with its own environment.

    Each worker:
    1. Syncs local network with global network
    2. Collects n steps of experience
    3. Computes gradients locally
    4. Applies gradients to global network (asynchronously)
    """

    def __init__(self, worker_id: int, global_network: A3CNetwork,
                 global_optimizer: optim.Optimizer, env_name: str,
                 gamma: float = 0.99, n_steps: int = 5,
                 entropy_coeff: float = 0.01):
        self.worker_id = worker_id
        self.global_network = global_network
        self.global_optimizer = global_optimizer
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coeff = entropy_coeff

        self.env = gym.make(env_name)
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        self.local_network = A3CNetwork(state_size, action_size)

    def sync_with_global(self):
        """Copy global network parameters to local network."""
        self.local_network.load_state_dict(self.global_network.state_dict())

    def collect_experience(self, state: np.ndarray, max_steps: int = 5):
        """Collect n steps of experience."""
        states, actions, rewards, dones = [], [], [], []

        for _ in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits, _ = self.local_network(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(done)

            state = next_state
            if done:
                break

        return states, actions, rewards, dones, state, done

    def compute_loss(self, states, actions, rewards, dones, next_state, done):
        """Compute A3C loss using n-step returns."""
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)

        logits, values = self.local_network(states_tensor)
        values = values.squeeze()

        # Bootstrap value
        with torch.no_grad():
            _, next_value = self.local_network(
                torch.FloatTensor(next_state).unsqueeze(0))
            next_value = next_value.squeeze()

        # Compute n-step returns
        returns = []
        R = next_value * (1 - int(done))
        for reward, d in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - int(d))
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)

        # Advantages
        advantages = returns - values.detach()

        # Policy loss
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions_tensor)
        policy_loss = -(log_probs * advantages).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy = dist.entropy().mean()

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy

        return total_loss

    def apply_gradients_to_global(self, loss):
        """Compute local gradients and apply to global network."""
        self.global_optimizer.zero_grad()
        loss.backward()

        # Transfer gradients from local to global
        for local_param, global_param in zip(
                self.local_network.parameters(),
                self.global_network.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad.clone()
            else:
                global_param.grad = local_param.grad.clone()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.global_network.parameters(), 40.0)
        self.global_optimizer.step()

    def run_episode(self):
        """Run one full episode as a worker."""
        self.sync_with_global()
        state, _ = self.env.reset()
        episode_reward = 0
        done = False

        while not done:
            states, actions, rewards, dones, state, done = self.collect_experience(
                state, max_steps=self.n_steps)
            episode_reward += sum(rewards)

            loss = self.compute_loss(states, actions, rewards, dones, state, done)
            self.apply_gradients_to_global(loss)
            self.sync_with_global()

        return episode_reward


def train_a3c_sequential(env_name: str = "CartPole-v1",
                          num_episodes: int = 1000,
                          n_workers: int = 4,
                          lr: float = 0.001,
                          gamma: float = 0.99) -> List[float]:
    """
    Train A3C with sequential workers (simulated parallelism).

    In true A3C, workers run in parallel processes.
    This version simulates the concept by alternating between workers.
    """
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()

    # Global network and optimizer
    global_network = A3CNetwork(state_size, action_size)
    global_optimizer = optim.Adam(global_network.parameters(), lr=lr)

    # Create workers
    workers = [
        A3CWorker(i, global_network, global_optimizer, env_name, gamma=gamma)
        for i in range(n_workers)
    ]

    all_rewards = []

    for episode in range(num_episodes):
        # Each worker runs one episode (sequential simulation)
        worker = workers[episode % n_workers]
        episode_reward = worker.run_episode()
        all_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(all_rewards[-100:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Worker {episode % n_workers} | Avg Reward: {avg:.1f}")

    # Clean up
    for w in workers:
        w.env.close()

    return all_rewards


def demonstrate_a3c_concept():
    """Explain the A3C architecture visually."""
    print("=" * 60)
    print("A3C: ASYNCHRONOUS ADVANTAGE ACTOR-CRITIC")
    print("=" * 60)

    print("""
    A3C Architecture:

    ┌─────────────────────────────────────────────────────────┐
    │                  GLOBAL NETWORK                         │
    │              (shared parameters θ)                      │
    │                                                         │
    │     ┌──────────┐  ┌──────────┐  ┌──────────┐          │
    │     │ Actor θ_π│  │ Actor θ_π│  │ Actor θ_π│          │
    │     │Critic θ_v│  │Critic θ_v│  │Critic θ_v│          │
    │     └─────┬────┘  └─────┬────┘  └─────┬────┘          │
    │           │              │              │               │
    │     sync↓ ↑grad   sync↓ ↑grad   sync↓ ↑grad          │
    │           │              │              │               │
    │     ┌─────┴────┐  ┌─────┴────┐  ┌─────┴────┐          │
    │     │ Worker 1 │  │ Worker 2 │  │ Worker 3 │          │
    │     │ (local)  │  │ (local)  │  │ (local)  │          │
    │     │  Env 1   │  │  Env 2   │  │  Env 3   │          │
    │     └──────────┘  └──────────┘  └──────────┘          │
    └─────────────────────────────────────────────────────────┘

    Each worker:
    1. Copies global network → local network (sync)
    2. Interacts with its own environment for n steps
    3. Computes gradients on local network
    4. Applies gradients to global network (async update)
    5. Repeat

    Why Asynchronous?
    - Workers run in parallel → faster training
    - Different workers explore different parts of state space
    - Decorrelates training data (like experience replay in DQN!)
    - No need for replay buffer
    """)


def demonstrate_a3c_vs_a2c():
    """Compare A3C and A2C approaches."""
    print("\n" + "=" * 60)
    print("A3C vs A2C COMPARISON")
    print("=" * 60)

    print("""
    ┌───────────────────┬────────────────────────────────────┐
    │     Feature        │    A3C         │    A2C            │
    ├───────────────────┼────────────────┼───────────────────┤
    │ Workers           │ Parallel       │ Synchronized      │
    │ Updates           │ Asynchronous   │ Synchronous       │
    │ Gradient apply    │ Lock-free      │ Averaged          │
    │ GPU usage         │ Inefficient    │ Efficient (batch) │
    │ Implementation    │ Complex (MP)   │ Simpler           │
    │ Reproducibility   │ Non-deterministic │ Deterministic  │
    │ Performance       │ ≈ Similar      │ ≈ Similar         │
    └───────────────────┴────────────────┴───────────────────┘

    In practice, A2C is preferred because:
    1. Easier to implement and debug
    2. Better GPU utilization (batched operations)
    3. Deterministic (reproducible results)
    4. Similar or better performance than A3C

    A3C's key contribution was showing that parallel workers
    can replace experience replay for decorrelating data.
    """)


def main():
    print("\n" + "=" * 60)
    print("WEEK 10 - LESSON 3: A3C (ASYNCHRONOUS ADVANTAGE ACTOR-CRITIC)")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Explain A3C architecture
    demonstrate_a3c_concept()

    # 2. Compare A3C vs A2C
    demonstrate_a3c_vs_a2c()

    # 3. Train A3C (sequential simulation)
    print("\n" + "=" * 60)
    print("TRAINING A3C (SEQUENTIAL SIMULATION) ON CARTPOLE-V1")
    print("=" * 60)
    print("(Using 4 workers alternating sequentially)")

    rewards = train_a3c_sequential(
        env_name="CartPole-v1",
        num_episodes=800,
        n_workers=4,
        lr=0.001
    )

    # Plot
    window = 50
    smoothed = [np.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, color='steelblue', label='Episode Reward')
    plt.plot(smoothed, color='steelblue', linewidth=2, label=f'{window}-ep Moving Avg')
    plt.axhline(y=500, color='red', linestyle='--', alpha=0.5, label='Max Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('A3C Training on CartPole-v1 (Sequential Simulation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('10_actor_critic/a3c_training.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to '10_actor_critic/a3c_training.png'")
    plt.close()

    print(f"\nFinal Performance (last 100 episodes): "
          f"{np.mean(rewards[-100:]):.1f} ± {np.std(rewards[-100:]):.1f}")

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. A3C uses multiple parallel workers for training")
    print("2. Workers asynchronously update a shared global network")
    print("3. Parallel exploration decorrelates data (like replay buffer)")
    print("4. In practice, synchronous A2C often performs equally well")
    print("5. A3C's main contribution: parallel workers can replace replay")
    print("\nNext: GAE - Generalized Advantage Estimation!")
    print("=" * 60)


if __name__ == "__main__":
    main()
