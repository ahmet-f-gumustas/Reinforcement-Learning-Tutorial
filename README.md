# Reinforcement Learning Tutorial

A comprehensive tutorial series on Reinforcement Learning, from beginner to advanced level.

> **Turkce versiyon:** [README_tr.md](README_tr.md)

## Requirements

- Python 3.10+
- Gymnasium
- NumPy
- Matplotlib
- PyTorch (for advanced topics)

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install gymnasium numpy matplotlib torch
```

## Weekly Curriculum

| Week | Topic | Folder | Status |
|------|-------|--------|--------|
| 1 | Introduction to RL and Basic Concepts | [01_introduction](01_introduction/) | Completed |
| 2 | Markov Decision Processes (MDP) | [02_mdp](02_mdp/) | Completed |
| 3 | Dynamic Programming | [03_dynamic_programming](03_dynamic_programming/) | Completed |
| 4 | Monte Carlo Methods | [04_monte_carlo](04_monte_carlo/) | Completed |
| 5 | Temporal Difference (TD) Learning | [05_temporal_difference](05_temporal_difference/) | Completed |
| 6 | Q-Learning and SARSA | [06_td_control](06_td_control/) | Completed |
| 7 | Function Approximation | [07_function_approximation](07_function_approximation/) | Completed |
| 8 | Deep Q-Network (DQN) | [08_dqn](08_dqn/) | Pending |
| 9 | Policy Gradient Methods | [09_policy_gradient](09_policy_gradient/) | Pending |
| 10 | Actor-Critic Methods | [10_actor_critic](10_actor_critic/) | Pending |
| 11 | Proximal Policy Optimization (PPO) | [11_ppo](11_ppo/) | Pending |
| 12 | Advanced Topics and Project | [12_advanced](12_advanced/) | Pending |

## Topic Details

### Week 1: Introduction to RL and Basic Concepts
- Fundamentals of Reinforcement Learning
- Agent, Environment, State, Action, Reward concepts
- First steps with Gymnasium library
- CartPole and other basic environments

### Week 2: Markov Decision Processes (MDP)
- Markov property
- State transition probabilities
- Reward functions
- Bellman equations

### Week 3: Dynamic Programming
- Policy Evaluation
- Policy Improvement
- Policy Iteration
- Value Iteration

### Week 4: Monte Carlo Methods
- First-visit vs Every-visit MC
- MC Prediction
- MC Control
- Importance Sampling

### Week 5: Temporal Difference Learning
- TD(0) Prediction
- TD vs MC vs DP comparison
- n-step TD
- Eligibility Traces

### Week 6: Q-Learning and SARSA
- On-policy vs Off-policy
- SARSA algorithm
- Q-Learning algorithm
- Expected SARSA

### Week 7: Function Approximation
- Tabular vs Approximation methods
- Linear function approximation
- Feature engineering
- Gradient descent methods

### Week 8: Deep Q-Network (DQN)
- Neural network for Q-function
- Experience Replay
- Target Network
- DQN variants (Double DQN, Dueling DQN)

### Week 9: Policy Gradient Methods
- Policy-based methods
- REINFORCE algorithm
- Baseline concept
- Variance reduction techniques

### Week 10: Actor-Critic Methods
- Actor-Critic architecture
- A2C (Advantage Actor-Critic)
- A3C (Asynchronous A3C)
- GAE (Generalized Advantage Estimation)

### Week 11: Proximal Policy Optimization (PPO)
- Trust Region Policy Optimization (TRPO)
- PPO-Clip
- PPO-Penalty
- Practical implementation

### Week 12: Advanced Topics and Project
- Multi-agent RL
- Model-based RL
- Inverse RL
- Final project: Solve your own environment

## Resources

- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [David Silver RL Course](https://www.davidsilver.uk/teaching/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## License

MIT License
