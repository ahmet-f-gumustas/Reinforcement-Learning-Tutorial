"""
03 - Exploring Different Gymnasium Environments

In this example, we will examine various Gymnasium environments.
We will understand the observation space, action space, and reward structure of each environment.
"""

import gymnasium as gym
import numpy as np


def explore_environment(env_name, num_episodes=5, verbose=True):
    """
    Explores a Gymnasium environment and prints its information.

    Args:
        env_name: Name of the environment
        num_episodes: Number of episodes to run
        verbose: Whether to print detailed information
    """
    try:
        env = gym.make(env_name)
    except Exception as e:
        print(f"Failed to load environment: {env_name}")
        print(f"Error: {e}")
        return None

    if verbose:
        print("\n" + "=" * 60)
        print(f"ENVIRONMENT: {env_name}")
        print("=" * 60)

        # Observation Space
        print(f"\n[Observation Space]")
        print(f"  Type: {type(env.observation_space).__name__}")
        print(f"  Details: {env.observation_space}")

        if hasattr(env.observation_space, 'shape'):
            print(f"  Shape: {env.observation_space.shape}")
        if hasattr(env.observation_space, 'n'):
            print(f"  n (discrete): {env.observation_space.n}")

        # Action Space
        print(f"\n[Action Space]")
        print(f"  Type: {type(env.action_space).__name__}")
        print(f"  Details: {env.action_space}")

        if hasattr(env.action_space, 'n'):
            print(f"  n (discrete): {env.action_space.n}")
        if hasattr(env.action_space, 'shape'):
            print(f"  Shape: {env.action_space.shape}")

    # Run episodes
    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    if verbose:
        print(f"\n[{num_episodes} Episode Results]")
        print(f"  Mean Reward: {np.mean(episode_rewards):.2f}")
        print(f"  Reward Std: {np.std(episode_rewards):.2f}")
        print(f"  Mean Length: {np.mean(episode_lengths):.1f}")
        print(f"  Min/Max Reward: {np.min(episode_rewards):.1f} / {np.max(episode_rewards):.1f}")

    env.close()

    return {
        "env_name": env_name,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths)
    }


def compare_environments():
    """Compares multiple environments."""

    # Environments to explore
    environments = [
        "CartPole-v1",
        "MountainCar-v0",
        "Acrobot-v1",
        "LunarLander-v3",
    ]

    print("\n" + "#" * 60)
    print("# GYMNASIUM ENVIRONMENT COMPARISON")
    print("#" * 60)

    results = []

    for env_name in environments:
        result = explore_environment(env_name, num_episodes=10)
        if result:
            results.append(result)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Environment':<20} {'Mean Reward':>12} {'Std':>10} {'Mean Length':>12}")
    print("-" * 60)

    for r in results:
        print(f"{r['env_name']:<20} {r['mean_reward']:>12.2f} {r['std_reward']:>10.2f} {r['mean_length']:>12.1f}")


def understand_cartpole():
    """Examines the CartPole environment in detail."""

    print("\n" + "#" * 60)
    print("# CARTPOLE DETAILED EXAMINATION")
    print("#" * 60)

    env = gym.make("CartPole-v1")

    print("""
    CartPole Problem:
    -----------------
    A pole is attached to a cart that moves along a track.
    Goal: Keep the pole balanced upright for as long as possible.

    Physics:
    - The pole starts at a slight angle
    - Gravity tries to pull the pole down
    - We can balance it by pushing the cart left/right
    """)

    # Watch an episode
    print("\nSample Episode:")
    print("-" * 40)

    obs, _ = env.reset(seed=42)
    print(f"Initial state:")
    print(f"  Cart position: {obs[0]:.4f}")
    print(f"  Cart velocity: {obs[1]:.4f}")
    print(f"  Pole angle: {obs[2]:.4f} rad ({np.degrees(obs[2]):.2f} degrees)")
    print(f"  Pole angular velocity: {obs[3]:.4f}")

    # Take a few steps
    print("\nFirst 5 steps:")
    for i in range(5):
        action = env.action_space.sample()
        action_name = "RIGHT" if action == 1 else "LEFT"
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"  Step {i+1}: {action_name:>5} -> Pole angle: {np.degrees(obs[2]):>7.2f} degrees, Reward: {reward}")

        if terminated or truncated:
            print("  Episode ended!")
            break

    print("""
    Episode Termination Conditions:
    -------------------------------
    1. Pole angle exceeds |12 degrees| -> FAILURE
    2. Cart position exceeds |2.4| -> FAILURE
    3. 500 steps completed -> SUCCESS

    Reward Structure:
    -----------------
    - +1 reward for each step
    - Maximum total reward: 500
    """)

    env.close()


def main():
    print("=" * 60)
    print("GYMNASIUM ENVIRONMENT EXPLORATION")
    print("=" * 60)
    print("""
    This script helps you explore different Gymnasium environments.

    Options:
    1. Explore single environment
    2. Compare environments
    3. CartPole detailed examination
    """)

    # CartPole detailed examination
    understand_cartpole()

    # Compare environments
    compare_environments()

    print("\n" + "=" * 60)
    print("EXERCISE SUGGESTIONS")
    print("=" * 60)
    print("""
    1. Call explore_environment() with different environments
    2. Increase the number of episodes for more reliable statistics
    3. Take your own notes:
       - Which environments are harder?
       - How do reward structures differ?
       - What's the difference between continuous vs discrete action spaces?
    """)


if __name__ == "__main__":
    main()
