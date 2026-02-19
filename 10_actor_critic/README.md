# Week 10: Actor-Critic Methods

This week introduces Actor-Critic methods - combining the best of policy gradient (Week 9) and value-based methods (Weeks 6-8) into a unified framework.

## Contents

1. [From Policy Gradient to Actor-Critic](#from-policy-gradient-to-actor-critic)
2. [Actor-Critic Architecture](#actor-critic-architecture)
3. [A2C: Advantage Actor-Critic](#a2c-advantage-actor-critic)
4. [A3C: Asynchronous Advantage Actor-Critic](#a3c-asynchronous-advantage-actor-critic)
5. [GAE: Generalized Advantage Estimation](#gae-generalized-advantage-estimation)
6. [Continuous Actions](#continuous-actions)
7. [Code Examples](#code-examples)
8. [Exercises](#exercises)

---

## From Policy Gradient to Actor-Critic

### The Problem with REINFORCE

In Week 9, we learned REINFORCE:
```
θ ← θ + α × ∇_θ log π_θ(a_t|s_t) × G_t
```

**Problem:** Using full episode return `G_t` causes **high variance**:
- Different episodes can have very different returns
- Must wait until episode end to update
- Slow, noisy learning

### The Solution: Use a Critic

Replace `G_t` with a learned value function:
```
θ ← θ + α × ∇_θ log π_θ(a_t|s_t) × δ_t

where δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
```

The TD error `δ_t` serves as an advantage estimate:
- If `δ > 0`: outcome was BETTER than expected → increase action probability
- If `δ < 0`: outcome was WORSE than expected → decrease action probability

### Why This Works

```
REINFORCE:      Advantage ≈ G_t - b(s)          (Monte Carlo, high variance)
Actor-Critic:   Advantage ≈ r + γV(s') - V(s)   (TD, lower variance)
```

| Property | REINFORCE | Actor-Critic |
|----------|-----------|--------------|
| Variance | High | Lower |
| Bias | None | Some (from V approximation) |
| Update timing | End of episode | Every step |
| Sample efficiency | Low | Better |

---

## Actor-Critic Architecture

### Two Components

```
┌─────────────────────────────────────────────────┐
│                 Environment                      │
│                                                  │
│    state s ──────┬──────────────> reward r       │
│                  │                   │            │
│                  ▼                   ▼            │
│           ┌──────────┐       ┌──────────┐       │
│           │  ACTOR   │       │  CRITIC  │       │
│           │  π(a|s)  │       │   V(s)   │       │
│           └────┬─────┘       └────┬─────┘       │
│                │                   │              │
│           action a          TD error δ           │
│                              = r + γV(s') - V(s) │
│                │    ┌──────────────┘              │
│                ▼    ▼                             │
│           Update Actor:                          │
│           θ += α × ∇log π(a|s) × δ              │
└─────────────────────────────────────────────────┘
```

**Actor (Policy):**
- Decides WHAT action to take
- Outputs: probability distribution π(a|s)
- Updated using advantage from critic

**Critic (Value Function):**
- Evaluates HOW GOOD a state is
- Outputs: scalar value V(s)
- Updated using TD error (Bellman equation)

### Network Architecture Options

**Option 1: Separate Networks**
```python
actor = Actor(state_size, action_size)     # π(a|s)
critic = Critic(state_size)                 # V(s)
```
- Separate optimizers
- More stable but less efficient

**Option 2: Shared Network**
```python
class ActorCritic(nn.Module):
    def __init__(self):
        self.shared = SharedLayers()
        self.actor_head = ActorHead()    # π(a|s)
        self.critic_head = CriticHead()  # V(s)
```
- Shared feature extraction
- More parameter efficient
- Most common in practice

---

## A2C: Advantage Actor-Critic

### Algorithm

A2C collects batches of experience (n steps) before updating:

```
Algorithm: A2C

For each iteration:
    Collect n steps of experience using current policy

    Compute n-step returns:
        R_t = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})

    Compute advantages:
        A_t = R_t - V(s_t)

    Update:
        Policy loss  = -E[log π(a|s) × A(s,a)]
        Value loss   = E[(R_t - V(s))²]
        Entropy loss = -E[H(π(·|s))]

        Total loss = Policy loss + c₁ × Value loss - c₂ × Entropy loss
```

### N-Step Returns

```
1-step:  R^(1) = r_t + γ V(s_{t+1})                    (low variance, high bias)
2-step:  R^(2) = r_t + γr_{t+1} + γ² V(s_{t+2})
n-step:  R^(n) = Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n})
∞-step:  G_t   = Σ_{k=0}^{T-t} γ^k r_{t+k}             (high variance, no bias)
```

Typical choice: n = 5 or n = 20

### Entropy Bonus

Entropy `H(π) = -Σ_a π(a|s) log π(a|s)` measures policy randomness:

- High entropy → policy explores (uniform distribution)
- Low entropy → policy exploits (deterministic)

Adding entropy bonus prevents premature convergence to suboptimal deterministic policies.

### Implementation

```python
# Collect n steps
for step in range(n_steps):
    action = actor(state)
    next_state, reward, done = env.step(action)
    buffer.add(state, action, reward, done)

# Compute advantages with GAE or n-step returns
advantages = compute_advantages(buffer, critic)

# Combined loss
policy_loss = -(log_probs * advantages).mean()
value_loss  = (returns - values).pow(2).mean()
entropy     = -(probs * log_probs).sum(-1).mean()

loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
```

---

## A3C: Asynchronous Advantage Actor-Critic

### Architecture

A3C (Mnih et al., 2016) uses **multiple parallel workers**:

```
┌──────────────────────────────────────────────────┐
│               GLOBAL NETWORK (shared θ)          │
│                                                   │
│     ┌────────┐    ┌────────┐    ┌────────┐      │
│     │Worker 1│    │Worker 2│    │Worker 3│      │
│     │ Env 1  │    │ Env 2  │    │ Env 3  │      │
│     └───┬────┘    └───┬────┘    └───┬────┘      │
│         │              │              │           │
│    sync↓↑grad    sync↓↑grad    sync↓↑grad       │
│         │              │              │           │
│         └──────────────┼──────────────┘           │
│                        │                          │
│               Global Parameters θ                │
└──────────────────────────────────────────────────┘
```

Each worker:
1. Copies global network → local network
2. Collects n steps of experience
3. Computes gradients locally
4. Applies gradients to global network asynchronously

### Why Asynchronous?

- **Decorrelation:** Different workers explore different states (like experience replay in DQN, but without a buffer!)
- **Parallelism:** Faster wall-clock training time
- **Simplicity:** No replay buffer needed

### A3C vs A2C in Practice

| Feature | A3C | A2C |
|---------|-----|-----|
| Workers | Parallel (async) | Synchronized |
| GPU usage | Inefficient | Efficient (batched) |
| Reproducibility | Non-deterministic | Deterministic |
| Implementation | Complex (multiprocessing) | Simpler |
| Performance | ≈ Same | ≈ Same |

**A2C is preferred in practice** because it's simpler, reproducible, and has similar performance. A3C's main contribution was demonstrating that parallel workers can replace experience replay.

---

## GAE: Generalized Advantage Estimation

### The Bias-Variance Dilemma

Different advantage estimates have different trade-offs:

```
A^(1) = δ_t = r + γV(s') - V(s)                         (1-step TD, high bias)
A^(2) = δ_t + γδ_{t+1}                                   (2-step)
A^(n) = δ_t + γδ_{t+1} + ... + γ^{n-1}δ_{t+n-1}        (n-step)
A^(∞) = G_t - V(s)                                       (MC, no bias)
```

### GAE Formula

GAE provides a smooth interpolation controlled by λ:

```
A^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l × δ_{t+l}
```

where `δ_t = r_t + γV(s_{t+1}) - V(s_t)`

### Special Cases

- **λ = 0:** `A = δ_t` (TD error only, low variance, high bias)
- **λ = 1:** `A = G_t - V(s)` (Monte Carlo advantage, high variance, no bias)
- **0 < λ < 1:** Smooth interpolation

```
λ = 0 ◄────── λ = 0.95 (typical) ──────► λ = 1
TD(0)          Good trade-off             Monte Carlo
Low variance   Medium variance            High variance
High bias      Low bias                   No bias
```

### Implementation

```python
def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values_extended = values + [next_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_extended[t+1] * (1-dones[t]) - values_extended[t]
        gae = delta + gamma * lam * (1-dones[t]) * gae
        advantages.insert(0, gae)

    return advantages
```

### Typical Values

- `λ = 0.95` or `λ = 0.97` work well for most environments
- GAE is used in PPO, TRPO, and most modern policy gradient methods

---

## Continuous Actions

### Gaussian Actor-Critic

For continuous action spaces, parameterize the policy as Gaussian:

```
π(a|s) = N(a; μ_θ(s), σ_θ)

where:
  μ_θ(s) = neural network output (state-dependent)
  σ_θ = learnable parameter (can be state-dependent)
```

### Architecture

```python
class ContinuousActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        self.shared = SharedLayers()
        self.actor_mean = nn.Linear(hidden, action_size)   # μ(s)
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))  # log σ
        self.critic = nn.Linear(hidden, 1)                 # V(s)
```

### Key Differences from Discrete

| Aspect | Discrete | Continuous |
|--------|----------|------------|
| Distribution | Categorical | Gaussian (Normal) |
| Output | Logits → Softmax | Mean μ(s), Std σ |
| Sampling | `Categorical(probs).sample()` | `Normal(μ, σ).sample()` |
| Log prob | `log π(a\|s)` for chosen action | `-(a-μ)²/2σ² - log σ - log√2π` |
| Clipping | N/A | Clip to action bounds |

---

## Code Examples

This folder contains 7 example scripts:

### 1. `01_actor_critic_basics.py`
Introduction to Actor-Critic architecture.
- Actor and Critic network design
- TD error as advantage estimate
- Separate vs shared network architectures
- Comparison with REINFORCE on GridWorld

**Run:** `python 01_actor_critic_basics.py`

### 2. `02_a2c.py`
A2C (Advantage Actor-Critic) implementation.
- N-step return computation
- Entropy bonus for exploration
- Batch update mechanism
- Training on CartPole-v1

**Run:** `python 02_a2c.py`

### 3. `03_a3c.py`
A3C (Asynchronous Advantage Actor-Critic) concept.
- Global and local network architecture
- Worker-based training (sequential simulation)
- Asynchronous gradient updates
- A3C vs A2C comparison

**Run:** `python 03_a3c.py`

### 4. `04_gae.py`
Generalized Advantage Estimation.
- GAE computation with lambda parameter
- Step-by-step numerical examples
- Bias-variance trade-off visualization
- Different lambda value comparison

**Run:** `python 04_gae.py`

### 5. `05_cartpole_ac.py`
Complete CartPole implementation.
- Production-ready A2C with GAE
- Orthogonal initialization
- Model saving/loading
- Comprehensive training visualization

**Run:** `python 05_cartpole_ac.py`

### 6. `06_continuous_ac.py`
Continuous action spaces.
- Gaussian policy for continuous control
- Learned standard deviation
- Training on Pendulum-v1
- Action clipping

**Run:** `python 06_continuous_ac.py`

### 7. `07_comparison.py`
Algorithm comparison.
- DQN vs REINFORCE vs A2C side-by-side
- Sample efficiency analysis
- Training stability comparison
- Algorithm selection guide

**Run:** `python 07_comparison.py`

---

## Exercises

### Exercise 1: Basic Actor-Critic

Implement a basic one-step Actor-Critic:
1. Create separate actor and critic networks
2. Use TD error as advantage: δ = r + γV(s') - V(s)
3. Train on CartPole-v1
4. Compare with shared network version
5. Track policy entropy during training

**Goal:** Understand the basic Actor-Critic mechanism.

### Exercise 2: N-Step Returns

Compare different n-step values:
1. Implement A2C with configurable n
2. Train with n = 1, 5, 20, 50, full episode
3. Plot learning curves for each
4. Measure variance of advantage estimates
5. Find optimal n for CartPole

**Goal:** Understand the bias-variance trade-off of n-step returns.

### Exercise 3: GAE Exploration

Explore GAE lambda parameter:
1. Implement GAE with configurable λ
2. Train with λ = 0, 0.5, 0.8, 0.95, 1.0
3. Plot learning curves and final performance
4. Compare variance of advantages
5. Determine best λ for your environment

**Goal:** Master the GAE bias-variance control.

### Exercise 4: Continuous Control

Apply Actor-Critic to continuous problems:
1. Implement Gaussian Actor-Critic
2. Train on Pendulum-v1
3. Try MountainCarContinuous-v0
4. Experiment with state-dependent vs fixed std
5. Visualize learned policy

**Goal:** Handle continuous action spaces with Actor-Critic.

### Exercise 5: LunarLander Challenge

Solve LunarLander-v2 with A2C:
1. Implement full A2C with GAE
2. Add entropy bonus and gradient clipping
3. Tune hyperparameters to reach score > 200
4. Compare with DQN (Week 8) and REINFORCE (Week 9)
5. Profile sample efficiency of each method

**Goal:** Apply Actor-Critic to a challenging environment.

---

## Key Equations

### TD Error (Advantage Estimate)
```
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
```

### N-Step Return
```
R^(n)_t = Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n})
```

### GAE
```
A^GAE(γ,λ)_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
```

### A2C Loss
```
L = L_policy + c₁ L_value - c₂ H(π)

L_policy = -E[log π(a|s) × A(s,a)]
L_value  = E[(R - V(s))²]
H(π)     = -E[Σ_a π(a|s) log π(a|s)]
```

### Gaussian Policy
```
π(a|s) = N(a; μ_θ(s), σ_θ)
log π(a|s) = -½[(a-μ)/σ]² - log σ - ½log(2π)
```

---

## Practical Tips

### 1. Hyperparameters

Good starting values:
```python
lr = 0.001              # Learning rate
gamma = 0.99            # Discount factor
lam = 0.95              # GAE lambda
entropy_coeff = 0.01    # Entropy bonus weight
value_coeff = 0.5       # Value loss weight
max_grad_norm = 0.5     # Gradient clipping
n_steps = 5 or 128      # Rollout length
```

### 2. Initialization

Orthogonal initialization improves training:
```python
nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
nn.init.zeros_(layer.bias)
# Smaller gain for output heads:
nn.init.orthogonal_(actor_head.weight, gain=0.01)
nn.init.orthogonal_(critic_head.weight, gain=1.0)
```

### 3. Common Issues

- **No learning:** Check advantage computation, ensure correct signs
- **Unstable training:** Reduce learning rate, add gradient clipping
- **Premature convergence:** Increase entropy coefficient
- **Value function divergence:** Reduce value_coeff, clip value loss
- **Continuous actions explode:** Clip actions to environment bounds

### 4. Debugging

Track these metrics:
- Episode rewards (smoothed)
- Policy entropy (should decrease slowly)
- Value loss (should decrease)
- Advantage mean and std
- Gradient norms

---

## Historical Context

### 1980s-90s: Origins
- Barto, Sutton & Anderson (1983): Actor-Critic for pole balancing
- Original TD(λ) actor-critic formulations

### 2016: A3C (Mnih et al.)
- Asynchronous parallel workers
- Showed parallel exploration replaces experience replay
- First competitive deep RL without replay buffer

### 2016: GAE (Schulman et al.)
- Generalized Advantage Estimation paper
- Smooth bias-variance control with λ
- Foundation for PPO and TRPO

### 2017: A2C
- Synchronous version of A3C
- Simpler, equally effective
- Better GPU utilization

### Modern Era
- Actor-Critic is the foundation for PPO, SAC, TD3, DDPG
- Nearly all modern deep RL uses actor-critic architectures

---

## Next Week

In Week 11, we cover **Proximal Policy Optimization (PPO)**:
- Trust Region Policy Optimization (TRPO)
- PPO-Clip: constraining policy updates
- PPO-Penalty: adaptive KL penalty
- Practical implementation with GAE

PPO builds directly on A2C + GAE from this week, adding a clipping mechanism for stable policy updates.

---

## Resources

### Papers

**Foundational:**
- [Asynchronous Methods for Deep RL (Mnih et al., 2016)](https://arxiv.org/abs/1602.01783) - A3C
- [High-Dimensional Continuous Control Using GAE (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438) - GAE

**Related:**
- [Proximal Policy Optimization (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) - PPO (next week)
- [Soft Actor-Critic (Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290) - SAC
- [Continuous Control with Deep RL (Lillicrap et al., 2016)](https://arxiv.org/abs/1509.02971) - DDPG

### Tutorials & Courses
- [Spinning Up: Actor-Critic](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) - OpenAI
- [David Silver Lecture 7](https://www.youtube.com/watch?v=KHZVXao4qXs) - Policy Gradient & Actor-Critic
- [CS285 Actor-Critic](http://rail.eecs.berkeley.edu/deeprlcourse/) - UC Berkeley

### Code Implementations
- [Stable-Baselines3 A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)
- [CleanRL A2C](https://github.com/vwxyzjn/cleanrl) - Single-file implementations
- [Spinning Up VPG](https://spinningup.openai.com/en/latest/algorithms/vpg.html) - Vanilla Policy Gradient

---

## Summary

**What We Learned:**

1. Actor-Critic = Actor (policy) + Critic (value function)
2. TD error δ = r + γV(s') - V(s) provides low-variance advantage
3. A2C: batch updates with n-step returns and entropy bonus
4. A3C: parallel workers for data decorrelation (historical)
5. GAE: λ-weighted average of multi-step TD errors
6. Works naturally with both discrete and continuous actions

**Key Insight:**

Actor-Critic methods combine the strengths of both approaches: the policy optimization capabilities of policy gradient methods with the low-variance bootstrapping of value-based methods. This combination is the foundation of nearly all modern deep RL algorithms.

**Next Step:**

PPO (Week 11) builds on A2C + GAE by adding a clipping mechanism that prevents destructively large policy updates, leading to more stable training.

---

**Continue to Week 11: [Proximal Policy Optimization (PPO)](../11_ppo/)** →
