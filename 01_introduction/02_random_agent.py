"""
02 - Random Action Agent

In this example, we will implement a simple Random Agent.
The agent selects a random action at each step - it does not learn.
This will serve as a baseline for the smarter agents we will develop later.
"""

import gymnasium as gym
import numpy as np


class RandomAgent:
    """A simple agent that selects random actions."""

    def __init__(self, action_space):
        """
        Args:
            action_space: Gymnasium action space
        """
        self.action_space = action_space

    def select_action(self, observation):
        """
        Selects a random action without looking at the observation.

        Args:
            observation: Observation from the environment (not used)

        Returns:
            action: Randomly selected action
        """
        return self.action_space.sample()

    def learn(self, observation, action, reward, next_observation, done):
        """
        Random agent does not learn.
        This method provides an interface for agents we will develop later.
        """
        pass


def run_episode(env, agent, render=False):
    """
    Runs a single episode.

    Args:
        env: Gymnasium environment
        agent: Agent that selects actions
        render: Whether to render visualization

    Returns:
        total_reward: Total reward collected during the episode
        step_count: Number of steps in the episode
    """
    observation, info = env.reset()
    total_reward = 0
    step_count = 0
    done = False

    while not done:
        # Get action from agent
        action = agent.select_action(observation)

        # Take step in environment
        next_observation, reward, terminated, truncated, info = env.step(action)

        # Call agent's learning function (empty for Random agent)
        done = terminated or truncated
        agent.learn(observation, action, reward, next_observation, done)

        # Update variables
        observation = next_observation
        total_reward += reward
        step_count += 1

    return total_reward, step_count


def evaluate_agent(env, agent, num_episodes=100):
    """
    Evaluates the agent over multiple episodes.

    Args:
        env: Gymnasium environment
        agent: Agent to evaluate
        num_episodes: Number of episodes to run

    Returns:
        results: Dictionary containing reward and step information for each episode
    """
    rewards = []
    steps = []

    for episode in range(num_episodes):
        total_reward, step_count = run_episode(env, agent)
        rewards.append(total_reward)
        steps.append(step_count)

        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Last reward: {total_reward:.1f}, "
                  f"Average: {np.mean(rewards):.2f}")

    return {
        "rewards": rewards,
        "steps": steps,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_steps": np.mean(steps),
        "max_reward": np.max(rewards),
        "min_reward": np.min(rewards)
    }


def main():
    print("=" * 60)
    print("RANDOM AGENT EVALUATION")
    print("=" * 60)

    # Create environment
    env = gym.make("CartPole-v1")

    # Create agent
    agent = RandomAgent(env.action_space)

    print(f"\nEnvironment: CartPole-v1")
    print(f"Agent: Random Agent")
    print(f"Number of episodes: 100")
    print("\nStarting evaluation...\n")

    # Evaluate agent
    results = evaluate_agent(env, agent, num_episodes=100)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Mean Reward: {results['mean_reward']:.2f} (+/- {results['std_reward']:.2f})")
    print(f"Mean Steps: {results['mean_steps']:.2f}")
    print(f"Max Reward: {results['max_reward']:.1f}")
    print(f"Min Reward: {results['min_reward']:.1f}")

    print("\n" + "-" * 60)
    print("ANALYSIS")
    print("-" * 60)
    print("""
Random agent survives approximately 20-25 steps in CartPole.
The maximum score in CartPole is 500 (episode terminates after 500 steps).

For comparison:
- Random Agent: ~20-25 average reward
- Simple rule-based: ~50-100
- Trained RL agent: ~500 (maximum)

In the following weeks, we will develop better agents!
    """)

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
