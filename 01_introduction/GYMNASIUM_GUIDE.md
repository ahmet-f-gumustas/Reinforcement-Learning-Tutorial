# Gymnasium Quick Reference Guide

A concise guide to the Gymnasium library for Reinforcement Learning.

## What is Gymnasium?

Gymnasium (formerly OpenAI Gym) is the standard Python library for developing and testing reinforcement learning algorithms. It provides:

- **Standardized environments** - Pre-built simulations for testing RL agents
- **Unified API** - Consistent interface across all environments
- **Benchmarking** - Compare algorithms on the same tasks

## Installation

```bash
pip install gymnasium

# For additional environments
pip install gymnasium[classic-control]  # CartPole, MountainCar, etc.
pip install gymnasium[box2d]            # LunarLander, BipedalWalker
pip install gymnasium[atari]            # Atari games
pip install gymnasium[mujoco]           # Physics simulations
```

## Core API

### 1. Create Environment

```python
import gymnasium as gym

env = gym.make("CartPole-v1")

# With rendering
env = gym.make("CartPole-v1", render_mode="human")
```

### 2. Reset Environment

```python
observation, info = env.reset()

# With seed for reproducibility
observation, info = env.reset(seed=42)
```

### 3. Take Action

```python
action = env.action_space.sample()  # Random action
observation, reward, terminated, truncated, info = env.step(action)
```

**Return values:**
| Value | Description |
|-------|-------------|
| `observation` | Current state of the environment |
| `reward` | Reward received for the action |
| `terminated` | True if episode ended naturally (goal/failure) |
| `truncated` | True if episode ended due to time limit |
| `info` | Additional diagnostic information |

### 4. Close Environment

```python
env.close()
```

## Complete Example

```python
import gymnasium as gym

# Create environment
env = gym.make("CartPole-v1")

# Run 5 episodes
for episode in range(5):
    observation, info = env.reset()
    total_reward = 0

    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode {episode + 1}: Reward = {total_reward}")
            break

env.close()
```

## Spaces

### Observation Space

```python
# Check observation space
print(env.observation_space)        # Box(-4.8, 4.8, (4,), float32)
print(env.observation_space.shape)  # (4,)

# Sample random observation
sample_obs = env.observation_space.sample()
```

### Action Space

```python
# Discrete actions (e.g., CartPole: 0=left, 1=right)
print(env.action_space)    # Discrete(2)
print(env.action_space.n)  # 2

# Continuous actions (e.g., Pendulum)
print(env.action_space)        # Box(-2.0, 2.0, (1,), float32)
print(env.action_space.shape)  # (1,)
```

## Common Environments

| Environment | Type | Actions | Difficulty |
|------------|------|---------|------------|
| `CartPole-v1` | Classic Control | Discrete(2) | Easy |
| `MountainCar-v0` | Classic Control | Discrete(3) | Medium |
| `Acrobot-v1` | Classic Control | Discrete(3) | Medium |
| `Pendulum-v1` | Classic Control | Continuous | Medium |
| `LunarLander-v3` | Box2D | Discrete(4) | Medium |
| `BipedalWalker-v3` | Box2D | Continuous | Hard |

## List Available Environments

```python
import gymnasium as gym

# All environments
all_envs = gym.envs.registry.keys()

# Filter by keyword
classic = [e for e in all_envs if "CartPole" in e]
```

## Wrappers

Wrappers modify environment behavior:

```python
from gymnasium.wrappers import TimeLimit, RecordVideo

# Limit episode length
env = gym.make("CartPole-v1")
env = TimeLimit(env, max_episode_steps=100)

# Record video
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(env, "videos/")
```

## Tips

1. **Always close environments** - Use `env.close()` or context managers
2. **Set seeds for reproducibility** - Use `env.reset(seed=42)`
3. **Check spaces first** - Understand observation/action dimensions before coding
4. **Start simple** - Begin with CartPole before complex environments

## Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Environment List](https://gymnasium.farama.org/environments/classic_control/)
- [API Reference](https://gymnasium.farama.org/api/env/)
