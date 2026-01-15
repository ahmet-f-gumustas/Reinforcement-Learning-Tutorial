# Week 1: Introduction to Reinforcement Learning

This week we will learn the fundamental concepts of Reinforcement Learning (RL) and build our first applications using the Gymnasium library.

## Contents

1. [What is Reinforcement Learning?](#what-is-reinforcement-learning)
2. [Core Concepts](#core-concepts)
3. [Gymnasium Library](#gymnasium-library)
4. [Code Examples](#code-examples)
5. [Exercises](#exercises)

---

## What is Reinforcement Learning?

Reinforcement Learning is a machine learning paradigm where an **agent** learns to make decisions by interacting with an **environment**, aiming to maximize cumulative **rewards** through its **actions**.

### Comparison with Other Learning Types

| Type | Data | Feedback | Example |
|------|------|----------|---------|
| **Supervised Learning** | Labeled | Direct | Image classification |
| **Unsupervised Learning** | Unlabeled | None | Clustering |
| **Reinforcement Learning** | Experience | Reward signal | Game playing |

### Real-World Examples

- Chess/Go playing AI (AlphaGo)
- Autonomous vehicles
- Robot control
- Recommendation systems
- Resource management

---

## Core Concepts

### 1. Agent
The entity that makes decisions and takes actions. Examples: Game-playing AI, robot.

### 2. Environment
The world the agent interacts with. Examples: Game world, physical environment.

### 3. State
Information representing the current situation of the environment.

```
s_t = State of the environment at time t
```

### 4. Action
The moves an agent can make.

```
a_t = Action selected by the agent at time t
```

### 5. Reward
Numerical feedback received by the agent after taking an action.

```
r_t = Reward received at time t
```

### 6. Policy
The agent's strategy for selecting actions based on states.

```
pi(a|s) = Probability of selecting action a in state s
```

### 7. Value Function
Represents the long-term value of a state or state-action pair.

```
V(s) = Expected cumulative reward starting from state s
Q(s,a) = Expected cumulative reward taking action a in state s
```

### 8. Episode
The sequence from initial state to terminal state.

---

## The RL Loop

```
+-------+      action (a_t)      +-----------+
| Agent | --------------------> | Environment|
+-------+                        +-----------+
    ^                                 |
    |   state (s_t+1), reward (r_t)   |
    +---------------------------------+
```

At each step:
1. Agent observes the current state (s_t)
2. Selects an action (a_t)
3. Environment transitions to new state (s_t+1)
4. Agent receives a reward (r_t)
5. Agent learns from this experience

---

## Gymnasium Library

Gymnasium (formerly OpenAI Gym) is a Python library that provides standard environments for testing RL algorithms.

### Installation

```bash
pip install gymnasium
```

### Basic Environments

| Environment | Description | Difficulty |
|-------------|-------------|------------|
| CartPole-v1 | Balance a pole on a cart | Easy |
| MountainCar-v0 | Drive a car up a hill | Medium |
| LunarLander-v3 | Land on the moon | Medium |
| Acrobot-v1 | Swing up a pendulum | Medium |

### Basic API

```python
import gymnasium as gym

# Create environment
env = gym.make("CartPole-v1")

# Reset environment
observation, info = env.reset()

# Take a step
action = env.action_space.sample()  # Random action
observation, reward, terminated, truncated, info = env.step(action)

# Close environment
env.close()
```

---

## Code Examples

This folder contains 3 example scripts:

### 1. `01_basic_environment.py`
Demonstrates basic usage of Gymnasium environments.

### 2. `02_random_agent.py`
Implementation of a simple agent that selects random actions.

### 3. `03_environment_exploration.py`
Code for exploring different environments.

---

## Exercises

### Exercise 1: Understanding the Environment
Run the `CartPole-v1` environment and answer these questions:
- What is the dimension of the observation space?
- How many different actions are in the action space?
- When does an episode end?

### Exercise 2: Reward Analysis
Run 10 episodes in `MountainCar-v0` and:
- Record the total reward for each episode
- What is the average reward?
- Why are we getting negative rewards?

### Exercise 3: Environment Comparison
Try at least 3 different Gymnasium environments and compare:
- Observation space types
- Action space types
- Reward structures

---

## Next Week

In Week 2, we will cover **Markov Decision Processes (MDP)**. This concept forms the mathematical foundation of all RL algorithms.

---

## Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Sutton & Barto Chapter 1](http://incompleteideas.net/book/RLbook2020.pdf)
- [David Silver Lecture 1](https://www.youtube.com/watch?v=2pWv7GOvuf0)
