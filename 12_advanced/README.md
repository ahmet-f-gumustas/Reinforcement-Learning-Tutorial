# Week 12: Advanced Topics and Final Project

The final week covers advanced RL topics and provides a capstone project to apply everything learned throughout the course.

## Contents

1. [Multi-Agent RL](#multi-agent-rl)
2. [Model-Based RL](#model-based-rl)
3. [Inverse RL](#inverse-rl)
4. [Reward Shaping](#reward-shaping)
5. [Curriculum Learning](#curriculum-learning)
6. [Final Project](#final-project)
7. [Code Examples](#code-examples)
8. [Exercises](#exercises)

---

## Multi-Agent RL

Multiple agents learning and interacting in a shared environment.

### Settings

| Setting | Description | Examples |
|---------|-------------|----------|
| Cooperative | Shared goal | Robot swarm, team sports |
| Competitive | Opposing goals | Chess, Go, poker |
| Mixed | Some cooperation, some competition | Traffic, economy |

### Key Approaches

- **IQL (Independent Q-Learning):** Each agent learns independently
- **CTDE (Centralized Training, Decentralized Execution):** Central critic, local actors
- **QMIX:** Factored value functions for teams
- **MADDPG:** Multi-agent DDPG with centralized critics
- **MAPPO:** Multi-agent PPO (state-of-the-art for many tasks)

### Core Challenge: Non-Stationarity

From each agent's perspective, other agents are part of the environment. When they learn and change policies, the environment appears non-stationary.

---

## Model-Based RL

Learning a model of environment dynamics and using it for planning.

### Model-Free vs Model-Based

| Aspect | Model-Free | Model-Based |
|--------|------------|-------------|
| Learns | Policy/Value directly | Environment model T(s'|s,a) |
| Sample efficiency | Lower | Much higher |
| Computation | Less | More (planning) |
| Model errors | N/A | Can compound |
| Algorithms | DQN, PPO, A2C | Dyna-Q, MBPO, Dreamer, MuZero |

### Key Algorithms

- **Dyna-Q:** Q-learning + tabular model + planning steps
- **MBPO:** Model-based policy optimization with ensemble models
- **Dreamer:** World model in learned latent space
- **MuZero:** Learned model for MCTS planning (achieved superhuman Go, Chess, Atari)

---

## Inverse RL

Learning a reward function from expert demonstrations.

```
Standard RL:  Given reward → Learn policy
Inverse RL:   Given demonstrations → Learn reward → Derive policy
```

### Why IRL?

- Hard to specify reward functions for complex tasks
- Experts can demonstrate easily
- Learned reward generalizes better than cloned behavior

### Approaches

- **Feature Matching IRL:** Match expert's feature expectations
- **MaxEnt IRL (Ziebart):** Maximum entropy principle
- **GAIL (Ho & Ermon):** GAN-style imitation learning
- **AIRL:** Adversarial IRL for transferable rewards

---

## Reward Shaping

Techniques for designing better reward signals.

### Potential-Based Reward Shaping (PBRS)

```
F(s, s') = γΦ(s') - Φ(s)
```

**Theorem (Ng et al., 1999):** Adding PBRS preserves the optimal policy while potentially accelerating learning.

### Exploration Strategies

| Strategy | Mechanism |
|----------|-----------|
| ε-greedy | Random actions with probability ε |
| Count-based | Bonus for visiting new states: β/√N(s) |
| Curiosity (ICM) | Bonus for hard-to-predict transitions |
| RND | Random Network Distillation bonus |
| HER | Hindsight Experience Replay |

---

## Curriculum Learning

Training on progressively harder tasks.

### Strategy

1. Start with easy version of the task
2. Increase difficulty as agent improves
3. Skills transfer from easy → hard

### Approaches

- **Manual stages:** Hand-designed difficulty progression
- **Success-based:** Advance when success rate > threshold
- **Automatic Domain Randomization (ADR):** Randomize difficulty, adapt
- **Self-play:** Agent trains against its own past versions
- **Reverse curriculum:** Start near goal, progressively expand

---

## Final Project

Apply everything you've learned to solve a Gymnasium environment of your choice.

### Suggested Environments

| Difficulty | Environment | Action Space |
|-----------|-------------|--------------|
| Easy | CartPole-v1 | Discrete |
| Easy | Acrobot-v1 | Discrete |
| Medium | LunarLander-v2 | Discrete |
| Medium | Pendulum-v1 | Continuous |
| Hard | BipedalWalker-v3 | Continuous |
| Hard | CarRacing-v2 | Continuous |

### Project Template

`06_final_project.py` provides a universal PPO solver that works on any Gymnasium environment. Modify the config to target your chosen environment.

---

## Code Examples

### 1. `01_multi_agent.py`
Multi-agent RL with predator-prey.
- IQL (Independent Q-Learning)
- CTDE (Centralized Training, Decentralized Execution)
- Cooperative reward learning

**Run:** `python 01_multi_agent.py`

### 2. `02_model_based.py`
Model-based RL with Dyna-Q.
- Tabular world model
- Planning with learned model
- Q-Learning vs Dyna-Q comparison
- Different planning step counts

**Run:** `python 02_model_based.py`

### 3. `03_inverse_rl.py`
Inverse Reinforcement Learning.
- Expert demonstration collection
- Feature-based reward learning
- Reward function visualization
- IRL vs behavioral cloning

**Run:** `python 03_inverse_rl.py`

### 4. `04_reward_shaping.py`
Reward shaping and exploration.
- Potential-based reward shaping (PBRS)
- Count-based exploration bonus
- Sparse reward challenge
- Comparison of techniques

**Run:** `python 04_reward_shaping.py`

### 5. `05_curriculum_learning.py`
Curriculum learning.
- Progressive difficulty scaling
- With vs without curriculum comparison
- Automatic difficulty adjustment

**Run:** `python 05_curriculum_learning.py`

### 6. `06_final_project.py`
Universal PPO solver for the final project.
- Works on any Gymnasium environment
- Auto-detects discrete/continuous actions
- Configurable hyperparameters
- Full training visualization

**Run:** `python 06_final_project.py`

### 7. `07_summary.py`
Complete course summary.
- 12-week curriculum overview
- Algorithm taxonomy
- Key equations reference
- Algorithm selection guide
- Future learning directions

**Run:** `python 07_summary.py`

---

## Exercises

### Exercise 1: Multi-Agent Cooperation

Design a cooperative multi-agent task:
1. Create a 2-agent environment
2. Implement IQL and CTDE
3. Add communication between agents
4. Compare cooperation quality
5. Analyze credit assignment

### Exercise 2: Build a World Model

Implement a neural network world model:
1. Collect transitions from a Gymnasium environment
2. Train T(s'|s,a) and R(s,a) models
3. Use model for planning (generate synthetic rollouts)
4. Compare model-free vs model-based sample efficiency
5. Analyze model accuracy over prediction horizons

### Exercise 3: Learn from Demonstrations

Apply inverse RL:
1. Collect expert demos on a simple environment
2. Implement feature-matching IRL
3. Compare: behavioral cloning vs IRL
4. Test generalization to modified environments

### Exercise 4: Solve a New Environment

Your capstone project:
1. Choose an environment from the suggestions above
2. Apply PPO with GAE (use `06_final_project.py` as template)
3. Tune hyperparameters (lr, gamma, lambda, clip_eps)
4. Run 3 random seeds and report mean ± std
5. Create visualizations of training and final policy
6. Write a brief analysis: what worked, what didn't, why

---

## Algorithm Selection Guide

```
"What algorithm should I use?"

Discrete actions → PPO (default) or DQN (if sample efficiency needed)
Continuous actions → PPO (default) or SAC (if off-policy needed)
Very limited data → Model-based (Dreamer, MBPO)
Multi-agent → MAPPO or QMIX
Imitation → Behavioral Cloning → GAIL
Sparse reward → Curiosity + HER + Curriculum

When in doubt: start with PPO.
```

---

## Resources

### Advanced Papers
- [Multi-Agent Actor-Critic (Lowe et al., 2017)](https://arxiv.org/abs/1706.02275) - MADDPG
- [QMIX (Rashid et al., 2018)](https://arxiv.org/abs/1803.11485) - Factored Q
- [Dream to Control (Hafner et al., 2020)](https://arxiv.org/abs/1912.01603) - Dreamer
- [Mastering Atari with World Models (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104) - DreamerV3
- [MuZero (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265)
- [GAIL (Ho & Ermon, 2016)](https://arxiv.org/abs/1606.03476) - Generative Adversarial Imitation
- [Curiosity-driven Exploration (Pathak et al., 2017)](https://arxiv.org/abs/1705.05363) - ICM
- [Hindsight Experience Replay (Andrychowicz et al., 2017)](https://arxiv.org/abs/1707.01495) - HER

### Tutorials & Courses
- [Spinning Up in Deep RL](https://spinningup.openai.com/) - OpenAI
- [David Silver RL Course](https://www.davidsilver.uk/teaching/) - DeepMind
- [CS285: Deep RL](http://rail.eecs.berkeley.edu/deeprlcourse/) - UC Berkeley
- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/)

### Frameworks
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - Production RL
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Single-file implementations
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/) - Scalable RL
- [PettingZoo](https://pettingzoo.farama.org/) - Multi-agent environments

---

## Course Summary

### What We Learned

| Week | Topic | Key Algorithm |
|------|-------|---------------|
| 1 | Fundamentals | Agent-Environment loop |
| 2 | MDPs | Bellman equations |
| 3 | Dynamic Programming | Value/Policy Iteration |
| 4 | Monte Carlo | First-visit MC |
| 5 | Temporal Difference | TD(0), n-step TD |
| 6 | Q-Learning & SARSA | Off/On-policy tabular |
| 7 | Function Approximation | Linear, Neural |
| 8 | DQN | Experience replay, target net |
| 9 | Policy Gradient | REINFORCE |
| 10 | Actor-Critic | A2C, A3C, GAE |
| 11 | PPO | Clip, Penalty |
| 12 | Advanced | MARL, Model-based, IRL |

### Key Insight

The evolution of RL algorithms follows a clear path:
1. **Tabular** → works for small state spaces
2. **Function approximation** → scales to large/continuous states
3. **Deep learning** → handles complex observations (images, etc.)
4. **Policy gradient** → handles continuous actions
5. **Actor-Critic** → combines value and policy methods
6. **PPO** → stable, general-purpose deep RL

**PPO is the culmination of this journey** — it works on any environment, is simple to implement, and provides strong baseline performance.

---

**Congratulations on completing the 12-week RL tutorial!**
