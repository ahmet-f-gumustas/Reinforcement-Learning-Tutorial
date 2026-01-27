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

### Books & Courses
- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [David Silver RL Course](https://www.davidsilver.uk/teaching/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

### Awesome Lists & Tutorials
- [awesome-reinforcement-learning](https://github.com/awesomelistsio/awesome-reinforcement-learning) - Comprehensive list of RL frameworks, libraries, tools, and tutorials
- [awesome-deep-rl](https://github.com/kengz/awesome-deep-rl) - Curated list of Deep Reinforcement Learning resources
- [awesome-rl](https://github.com/aikorea/awesome-rl) - Reinforcement learning resources curated
- [Curated-Reinforcement-Learning-Resources](https://github.com/azminewasi/Curated-Reinforcement-Learning-Resources) - Courses and tutorials from various providers
- [reinforcement-learning-resources](https://github.com/datascienceid/reinforcement-learning-resources) - Video lectures, books, and libraries
- [dennybritz/reinforcement-learning](https://github.com/dennybritz/reinforcement-learning) - RL implementations with Python, OpenAI Gym, TensorFlow (Sutton's Book & David Silver's course)
- [awesome-machine-learning-robotics](https://github.com/Phylliade/awesome-machine-learning-robotics) - Machine Learning for Robotics resources

### Frameworks & Libraries
- [OpenAI Gym](https://gym.openai.com/) - Simulation environments for training agents
- [OpenAI Baselines](https://github.com/openai/baselines) - Expert implementations of deep RL algorithms
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - Reliable implementations of RL algorithms in PyTorch
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) - Scalable Reinforcement Learning library
- [Keras-RL](https://github.com/matthiasplappert/keras-rl) - Keras-compatible framework (DQN, SARSA, DDPG)
- [DeepMind Acme](https://github.com/deepmind/acme) - Research framework for RL by DeepMind
- [DeepMind DQN](https://github.com/deepmind/dqn) - Official DQN implementation from Nature paper

### Key Papers - Deep Reinforcement Learning
- [Human-level control through deep reinforcement learning](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) - Mnih et Al. (DQN)
- [Continuous control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) - Lillicrap et Al. (DDPG)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) - Schaul et Al.
- [Reinforcement learning with unsupervised auxiliary tasks](https://deepmind.com/blog/reinforcement-learning-unsupervised-auxiliary-tasks/) - Jaderberg et Al.
- [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286) - Heess et Al.
- [Deep RL that matters](https://arxiv.org/abs/1709.06560) - Henderson et Al. (Reproducibility)

### Key Papers - Policy Gradient & Theory
- [Simple statistical gradient-following algorithms for connectionist RL](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) - Williams (REINFORCE)
- [Policy Gradient Methods for RL with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) - Sutton et Al.
- [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf) - Silver et Al.
- [Reinforcement learning of motor skills with policy gradients](http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Neural-Netw-2008-21-682_4867%5b0%5d.pdf) - Peters and Schaal
- [Guided Policy Search](https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf) - Levine et Al.

## License

MIT License
