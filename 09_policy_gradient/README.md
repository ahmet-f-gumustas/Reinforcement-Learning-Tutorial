# Week 9: Policy Gradient Methods

This week introduces policy gradient methods - a fundamental shift from value-based methods to directly optimizing policies.

## Contents

1. [From Value-Based to Policy-Based](#from-value-based-to-policy-based)
2. [Policy Representation](#policy-representation)
3. [Policy Gradient Theorem](#policy-gradient-theorem)
4. [REINFORCE Algorithm](#reinforce-algorithm)
5. [Variance Reduction](#variance-reduction)
6. [Continuous Actions](#continuous-actions)
7. [Code Examples](#code-examples)
8. [Exercises](#exercises)

---

## From Value-Based to Policy-Based

### What We've Learned So Far

In Weeks 6-8, we learned **value-based methods**:
- Q-Learning, SARSA (Week 6)
- Function Approximation (Week 7)
- Deep Q-Networks (Week 8)

All these methods learn a value function:
```
Q(s, a) → estimate of expected return

Policy derived implicitly:  π(s) = argmax_a Q(s, a)
```

### The Policy Gradient Approach

**Policy gradient methods** learn the policy **directly**:
```
π_θ(a|s) → probability of action a in state s

Parameterized by θ (neural network weights)
```

### Why Learn Policies Directly?

**Advantages:**
1. **Continuous Actions**: Natural support for continuous action spaces
2. **Stochastic Policies**: Can learn stochastic policies (e.g., rock-paper-scissors)
3. **Convergence**: Guaranteed convergence to local optimum
4. **Simple Policies**: Sometimes policy is simpler than value function

**Disadvantages:**
1. **High Variance**: Policy gradients have high variance → slow learning
2. **Sample Inefficiency**: Typically require more samples than value-based methods
3. **Local Optima**: May converge to local rather than global optimum
4. **On-Policy**: Basic version requires on-policy learning

---

## Policy Representation

### Discrete Actions

For discrete action spaces, use a **categorical distribution**:

```
Neural Network: s → logits → softmax → π(a|s)
```

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)  # π(a|s)
```

**Key points:**
- Output is a **probability distribution**: Σ_a π(a|s) = 1
- Actions are **sampled** from this distribution
- Softmax ensures valid probabilities

### Continuous Actions

For continuous action spaces, use a **Gaussian (Normal) distribution**:

```
π(a|s) = N(a; μ_θ(s), σ_θ(s))
```

```python
class GaussianPolicy(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.mean_layer = nn.Linear(state_size, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, state):
        mean = self.mean_layer(state)
        std = self.log_std.exp()
        return mean, std

    def sample_action(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action, log_prob
```

**Key points:**
- Output: mean μ(s) and standard deviation σ(s)
- Actions are real-valued, sampled from N(μ, σ)
- log_std ensures σ > 0

---

## Policy Gradient Theorem

### Objective

Maximize expected return:
```
J(θ) = E_τ~π_θ [R(τ)]
```

where τ = (s₀, a₀, r₁, s₁, a₁, r₂, ...) is a trajectory.

### The Theorem

The **policy gradient theorem** states:

```
∇_θ J(θ) = E_τ~π_θ [ Σ_t ∇_θ log π_θ(a_t|s_t) × G_t ]
```

where:
- `G_t = Σ_{k=t}^T γ^{k-t} r_k` is the return from timestep t
- `∇_θ log π_θ(a_t|s_t)` is the score function

### Intuition

**What does this mean?**

```
∇_θ J(θ) = E[ (gradient of log probability) × (how good was the action) ]
```

- **If action led to high return (G_t large):** Increase probability of that action
- **If action led to low return (G_t small):** Decrease probability of that action

### Why Log Probability?

The log is a mathematical convenience:
```
∇_θ log π_θ(a|s) = ∇_θ π_θ(a|s) / π_θ(a|s)
```

This creates the correct weighting for the gradient estimator.

---

## REINFORCE Algorithm

### Monte Carlo Policy Gradient

**REINFORCE** (Williams, 1992) is the simplest policy gradient algorithm:

```
Algorithm: REINFORCE

Initialize policy parameters θ randomly

For each episode:
    Generate episode (s₀, a₀, r₁, s₁, a₁, ..., s_T) using π_θ

    For t = 0 to T-1:
        Compute return: G_t = Σ_{k=t}^T γ^{k-t} r_k

        Update: θ ← θ + α × ∇_θ log π_θ(a_t|s_t) × G_t
```

### Implementation

```python
def reinforce(env, policy_net, optimizer, num_episodes, gamma=0.99):
    for episode in range(num_episodes):
        # Generate episode
        log_probs = []
        rewards = []

        state = env.reset()
        done = False

        while not done:
            # Sample action from policy
            action_probs = policy_net(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            log_probs.append(log_prob)

            # Take action
            next_state, reward, done = env.step(action)
            rewards.append(reward)
            state = next_state

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # Compute loss
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G  # Negative for gradient ascent

        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Properties

**Advantages:**
- ✓ Unbiased gradient estimates
- ✓ Guaranteed convergence to local optimum
- ✓ Works with any differentiable policy

**Disadvantages:**
- ✗ High variance → slow, unstable learning
- ✗ Requires full episodes (no online learning)
- ✗ Sample inefficient

---

## Variance Reduction

The main problem with vanilla REINFORCE is **high variance**. Different episodes can have very different returns, leading to noisy gradient estimates.

### Technique 1: Baseline Subtraction

Subtract a baseline `b(s)` from returns:

```
∇_θ J(θ) = E[ ∇_θ log π_θ(a_t|s_t) × (G_t - b(s_t)) ]
```

**Key insight:** Any baseline that doesn't depend on the action doesn't introduce bias!

**Common choice:** State value function V(s)

```
Advantage: A(s,a) = G_t - V(s_t)
```

This measures "how much better is action a than the average action in state s?"

### Why Baselines Help

**Without baseline:**
```
G_t could be:  [100, 105, 98, 102, ...]  (high variance)
```

**With baseline V(s) ≈ 101:**
```
A(s,a) = G_t - V(s):  [-1, +4, -3, +1, ...]  (lower variance!)
```

### Implementation with Baseline

```python
class ValueNetwork(nn.Module):
    """Learns V(s) as baseline"""
    def __init__(self, state_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output V(s)
        )

    def forward(self, state):
        return self.network(state)


# During training:
for log_prob, value, G in zip(log_probs, values, returns):
    advantage = G - value.detach()  # Don't backprop through baseline
    policy_loss += -log_prob * advantage

    value_loss += (G - value).pow(2)  # MSE to learn V(s)
```

### Technique 2: Rewards-to-Go

Only use future rewards (causality principle):

```
G_t = Σ_{k=t}^T γ^{k-t} r_k    (NOT Σ_{k=0}^T)
```

Past rewards can't be affected by current action, so don't include them!

### Technique 3: Advantage Normalization

Normalize advantages to have mean 0, std 1:

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
```

This helps training stability.

### Technique 4: Gradient Clipping

Prevent exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
```

---

## Continuous Actions

### Gaussian Policy

For continuous actions, parameterize policy as a Gaussian:

```
π_θ(a|s) = N(a; μ_θ(s), σ_θ(s))
```

**Log probability:**
```
log π_θ(a|s) = -½[(a - μ)/σ]² - log σ - ½log(2π)
```

### Parameterization Choices

**1. Fixed variance:**
```python
std = 0.5  # Constant
```

**2. Learned variance (not state-dependent):**
```python
log_std = nn.Parameter(torch.zeros(action_size))
std = log_std.exp()  # Ensures positivity
```

**3. State-dependent variance:**
```python
mean = mean_head(state)
std = std_head(state).exp()
```

### Policy Gradient for Continuous Actions

**Same REINFORCE algorithm**, just with different distribution:

```python
# Sample action
mean, std = policy_net(state)
dist = torch.distributions.Normal(mean, std)
action = dist.sample()
log_prob = dist.log_prob(action).sum()  # Sum over action dimensions

# Update (same as discrete case)
loss = -log_prob * advantage
```

### Applications

Continuous actions enable:
- **Robot control:** Joint angles, torques
- **Autonomous driving:** Steering angle, throttle
- **Game AI:** Continuous movement, camera control
- **Finance:** Portfolio allocation

---

## Code Examples

This folder contains 7 example scripts:

### 1. `01_policy_basics.py`
Introduction to policy-based methods.
- Stochastic policy representation
- Policy vs value networks
- Simple GridWorld demonstration
- Conceptual comparisons

**Run:** `python 01_policy_basics.py`

### 2. `02_reinforce.py`
Core REINFORCE algorithm.
- Monte Carlo policy gradient
- Policy gradient theorem demonstration
- Episode trajectory collection
- Training on CartPole-v1
- Learning curve visualization

**Run:** `python 02_reinforce.py`

### 3. `03_baseline.py`
Baseline and variance reduction.
- State value baseline V(s)
- Advantage estimation A(s,a) = G_t - V(s)
- With/without baseline comparison
- Variance reduction visualization
- Side-by-side training

**Run:** `python 03_baseline.py`

### 4. `04_variance_reduction.py`
Advanced variance reduction techniques.
- Rewards-to-go (causality)
- Different baseline types
- Normalized advantages
- Gradient clipping
- Empirical comparison

**Run:** `python 04_variance_reduction.py`

### 5. `05_continuous_actions.py`
Continuous action spaces.
- Gaussian policy π(a|s) = N(μ(s), σ(s))
- Mean and std parameterization
- Log-likelihood computation
- Training on Pendulum-v1
- Exploration via learned std

**Run:** `python 05_continuous_actions.py`

### 6. `06_cartpole_pg.py`
Complete CartPole implementation.
- Production-ready REINFORCE agent
- Hyperparameter tuning
- Model checkpointing and loading
- Comprehensive visualization
- Evaluation and testing

**Run:** `python 06_cartpole_pg.py`

### 7. `07_comparison.py`
Algorithm comparison.
- REINFORCE vs DQN side-by-side
- Sample efficiency analysis
- Learning curves comparison
- When to use each method
- Decision guide

**Run:** `python 07_comparison.py`

---

## Exercises

### Exercise 1: Basic REINFORCE

Implement REINFORCE from scratch on a simple environment:
1. Create policy network for discrete actions
2. Implement episode generation
3. Compute returns and policy gradient
4. Train on CartPole-v1
5. Plot learning curves

**Goal:** Understand basic policy gradient mechanics.

### Exercise 2: Baseline Comparison

Compare REINFORCE with and without baseline:
1. Train both versions on same environment
2. Track variance of gradients
3. Compare learning speed
4. Visualize advantage distributions
5. Measure final performance

**Goal:** Empirically verify variance reduction.

### Exercise 3: Continuous Control

Implement Gaussian policy for continuous actions:
1. Create policy network outputting μ and σ
2. Implement action sampling from Normal distribution
3. Compute log probabilities correctly
4. Train on Pendulum-v1 or MountainCarContinuous-v0
5. Visualize learned policy

**Goal:** Master continuous action spaces.

### Exercise 4: Hyperparameter Study

Study effect of hyperparameters:
1. Vary learning rate (0.0001, 0.001, 0.01)
2. Vary discount factor γ (0.9, 0.95, 0.99)
3. Vary network architecture (64, 128, 256 hidden units)
4. Try different optimizers (SGD, Adam, RMSprop)
5. Plot all results and analyze

**Goal:** Understand hyperparameter sensitivity.

### Exercise 5: LunarLander

Solve LunarLander-v2:
1. Implement REINFORCE with all variance reduction techniques
2. Tune hyperparameters for best performance
3. Achieve score > 200
4. Compare with DQN from Week 8
5. Analyze which method works better and why

**Goal:** Apply policy gradients to challenging environment.

---

## Key Equations

### Policy Gradient Theorem
```
∇_θ J(θ) = E_τ~π_θ [ Σ_t ∇_θ log π_θ(a_t|s_t) × G_t ]
```

### REINFORCE Update
```
θ ← θ + α × ∇_θ log π_θ(a_t|s_t) × G_t
```

### With Baseline
```
θ ← θ + α × ∇_θ log π_θ(a_t|s_t) × (G_t - b(s_t))
```

### Advantage Function
```
A(s,a) = Q(s,a) - V(s) ≈ G_t - V(s_t)
```

### Return (Discounted)
```
G_t = Σ_{k=t}^T γ^{k-t} r_{k+1}
```

### Categorical Policy (Discrete)
```
π_θ(a|s) = softmax(f_θ(s))_a
```

### Gaussian Policy (Continuous)
```
π_θ(a|s) = N(a; μ_θ(s), σ_θ(s))

log π_θ(a|s) = -½[(a-μ)/σ]² - log σ - ½log(2π)
```

---

## Practical Tips

### 1. Start Simple

- Begin with small networks (64-128 hidden units)
- Use simple environments (CartPole)
- Add complexity gradually

### 2. Debugging

Common issues:
- **No learning:** Check learning rate, ensure gradients flow
- **Unstable learning:** Add baseline, normalize advantages
- **Exploding gradients:** Use gradient clipping
- **Slow learning:** Increase learning rate, reduce variance

### 3. Hyperparameters

Good starting values:
```python
learning_rate = 0.001  # Policy gradient methods often need higher LR than DQN
gamma = 0.99
hidden_size = 128
normalize_advantages = True
gradient_clip = 1.0
```

### 4. Variance Reduction is Critical

Always use:
- ✓ Baseline (state value function)
- ✓ Advantage normalization
- ✓ Gradient clipping
- ✓ Rewards-to-go

### 5. Monitoring

Track during training:
- Episode rewards
- Gradient variance
- Value function MSE
- Policy entropy (for exploration)

---

## Comparison: Policy Gradient vs Value-Based

| Aspect | Policy Gradient | Value-Based (DQN) |
|--------|----------------|-------------------|
| **Learns** | Policy π_θ(a\|s) directly | Q-function, derive π |
| **Actions** | Discrete & continuous | Discrete only |
| **Policy** | Stochastic natural | Requires ε-greedy |
| **Convergence** | Guaranteed (local) | Not guaranteed |
| **Variance** | High | Lower (bootstrapping) |
| **Sample efficiency** | Lower | Higher (experience replay) |
| **On/off-policy** | On-policy | Off-policy |
| **Best for** | Continuous control, stochastic | Discrete actions, sample efficiency |

---

## Historical Context

### 1992: REINFORCE (Williams)
- First practical policy gradient algorithm
- Monte Carlo returns
- Foundation for modern methods

### 1999: Policy Gradient Theorem (Sutton et al.)
- Formalized mathematical theory
- Showed unbiased gradient estimator
- Enabled function approximation

### 2000s: Natural Policy Gradient
- Trust Region methods
- Better optimization geometry
- Precursor to TRPO, PPO

### 2010s: Deep Policy Gradients
- Combination with deep learning
- Continuous control breakthroughs
- AlphaGo, robot learning

---

## Next Week

In Week 10, we cover **Actor-Critic Methods**:
- Combine policy gradient with value function
- Reduce variance while maintaining benefits
- A2C (Advantage Actor-Critic)
- A3C (Asynchronous A3C)
- GAE (Generalized Advantage Estimation)

Actor-Critic combines the best of both worlds: policy gradients + value methods!

---

## Resources

### Papers

**Foundational:**
- [Simple Statistical Gradient-Following Algorithms (Williams, 1992)](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) - REINFORCE
- [Policy Gradient Methods for RL (Sutton et al., 1999)](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) - Policy Gradient Theorem

**Advanced:**
- [Deterministic Policy Gradient (Silver et al., 2014)](http://proceedings.mlr.press/v32/silver14.pdf) - DPG
- [Trust Region Policy Optimization (Schulman et al., 2015)](https://arxiv.org/abs/1502.05477) - TRPO
- [High-Dimensional Continuous Control (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438) - GAE

**Applications:**
- [Reinforcement Learning of Motor Skills (Peters & Schaal, 2008)](http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Neural-Netw-2008-21-682_4867%5b0%5d.pdf) - Robot control

### Tutorials & Courses

- [Spinning Up: Policy Gradients](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) - OpenAI
- [David Silver Lecture 7: Policy Gradient](https://www.youtube.com/watch?v=KHZVXao4qXs) - DeepMind
- [CS285 Policy Gradients](http://rail.eecs.berkeley.edu/deeprlcourse/) - UC Berkeley

### Code Implementations

- [Stable-Baselines3 PPO](https://github.com/DLR-RM/stable-baselines3) - Production implementation
- [OpenAI Baselines](https://github.com/openai/baselines) - Reference implementations
- [Spinning Up](https://github.com/openai/spinningup) - Educational implementations

---

## Summary

**What We Learned:**

1. ✓ Policy gradient methods optimize policies directly
2. ✓ Policy gradient theorem provides unbiased gradient estimator
3. ✓ REINFORCE is the basic Monte Carlo policy gradient algorithm
4. ✓ Baselines reduce variance without introducing bias
5. ✓ Continuous actions handled naturally with Gaussian policies
6. ✓ Various variance reduction techniques improve learning

**Key Insight:**

Policy gradients shift the paradigm from "estimate values, derive policy" to "optimize policy directly." This enables continuous control and guaranteed convergence, at the cost of higher variance and sample inefficiency.

**Next Step:**

Actor-Critic methods (Week 10) combine policy gradients with value function learning to get the best of both approaches!

---

**Continue to Week 10: [Actor-Critic Methods](../10_actor_critic/)** →
