"""
01 - Basic Usage of Gymnasium Environment

In this example, we will learn the basic usage of the Gymnasium library.
We will explore the CartPole-v1 environment.
"""

import gymnasium as gym


def main():
    # ============================================
    # 1. CREATING THE ENVIRONMENT
    # ============================================
    print("=" * 50)
    print("1. CREATING THE ENVIRONMENT")
    print("=" * 50)

    # Create CartPole environment
    # render_mode="human" for visualization (optional)
    env = gym.make("CartPole-v1")

    print(f"Environment: {env.spec.id}")
    print(f"Max episode steps: {env.spec.max_episode_steps}")

    # ============================================
    # 2. OBSERVATION SPACE
    # ============================================
    print("\n" + "=" * 50)
    print("2. OBSERVATION SPACE")
    print("=" * 50)

    print(f"Observation Space: {env.observation_space}")
    print(f"Observation Shape: {env.observation_space.shape}")
    print(f"Observation Low: {env.observation_space.low}")
    print(f"Observation High: {env.observation_space.high}")

    print("\nCartPole Observation Variables:")
    print("  [0] Cart Position: -4.8 to 4.8")
    print("  [1] Cart Velocity: -inf to inf")
    print("  [2] Pole Angle: ~-0.42 to ~0.42 rad")
    print("  [3] Pole Angular Velocity: -inf to inf")

    # ============================================
    # 3. ACTION SPACE
    # ============================================
    print("\n" + "=" * 50)
    print("3. ACTION SPACE")
    print("=" * 50)

    print(f"Action Space: {env.action_space}")
    print(f"Action Space n: {env.action_space.n}")

    print("\nCartPole Actions:")
    print("  0: Push left")
    print("  1: Push right")

    # ============================================
    # 4. RESETTING THE ENVIRONMENT
    # ============================================
    print("\n" + "=" * 50)
    print("4. RESETTING THE ENVIRONMENT")
    print("=" * 50)

    # Reset environment and get initial observation
    observation, info = env.reset(seed=42)

    print(f"Initial Observation: {observation}")
    print(f"Info: {info}")

    # ============================================
    # 5. TAKING A STEP
    # ============================================
    print("\n" + "=" * 50)
    print("5. TAKING A STEP")
    print("=" * 50)

    # Push right (action = 1)
    action = 1
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"Action: {action} (Push right)")
    print(f"New Observation: {observation}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated} (Did the episode end naturally?)")
    print(f"Truncated: {truncated} (Did the episode end due to time limit?)")
    print(f"Info: {info}")

    # ============================================
    # 6. RUNNING AN EPISODE
    # ============================================
    print("\n" + "=" * 50)
    print("6. RUNNING AN EPISODE")
    print("=" * 50)

    observation, info = env.reset(seed=42)
    total_reward = 0
    step_count = 0

    done = False
    while not done:
        # Select random action
        action = env.action_space.sample()

        # Take step
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        # Check if episode is done
        done = terminated or truncated

    print(f"Episode completed!")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward}")

    # Close environment
    env.close()
    print("\nEnvironment closed.")


if __name__ == "__main__":
    main()
