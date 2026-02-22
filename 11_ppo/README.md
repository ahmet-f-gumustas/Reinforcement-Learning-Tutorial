# Week 11: Proximal Policy Optimization (PPO)

This week covers PPO - the most widely used deep RL algorithm today. PPO builds on Actor-Critic (Week 10) by adding a clipping mechanism that prevents destructively large policy updates.

## Contents

1. [Motivation: The Problem with Large Updates](#motivation)
2. [TRPO: Trust Region Policy Optimization](#trpo)
3. [PPO-Clip: The Simple Solution](#ppo-clip)
4. [PPO-Penalty: KL Divergence Approach](#ppo-penalty)
5. [Practical PPO Implementation](#practical-ppo)
6. [Continuous Actions](#continuous-actions)
7. [Code Examples](#code-examples)
8. [Exercises](#exercises)

---

## Motivation

### The Problem with A2C/REINFORCE

When we update the policy with gradient ascent:
```
θ_new = θ_old + α ∇J(θ)
```

The gradient is estimated from data collected with `π_old`. If the update is too large, `π_new` may be so different from `π_old` that performance collapses and may never recover.

**Why can't we just tune the learning rate?**
- Too small → very slow learning
- Too large → catastrophic collapse
- Optimal value varies throughout training and per environment

### The Trust Region Idea

Stay within a **trust region** around the current policy where the local approximation is reliable:

```
maximize  J_approx(θ)
subject to  KL(π_old || π_new) ≤ δ
```

---

## TRPO

### Algorithm

**Trust Region Policy Optimization** (Schulman et al., 2015):

```
maximize  L^CPI(θ) = E_t[ r_t(θ) × A_t ]
subject to  E_s[ KL(π_old(·|s) || π_new(·|s)) ] ≤ δ
```

where `r_t(θ) = π_new(a_t|s_t) / π_old(a_t|s_t)` is the importance ratio.

**Problems with TRPO:**
- Requires computing the Fisher Information Matrix
- Uses expensive conjugate gradient + line search
- Hard to implement correctly
- Doesn't work well with parameter sharing (actor/critic)

**Solution:** PPO approximates the TRPO constraint cheaply!

---

## PPO-Clip

### The Core Idea

Instead of a hard KL constraint, **clip** the importance ratio to stay near 1:

```
L^CLIP(θ) = E_t[ min(r_t(θ) × A_t,  clip(r_t(θ), 1-ε, 1+ε) × A_t) ]
```

where `ε = 0.2` is the typical default.

### How Clipping Works

When `A > 0` (good action — want to increase probability):
```
r_t rises → profit increases → but capped at (1+ε)
```
The `min()` prevents overly optimistic updates.

When `A < 0` (bad action — want to decrease probability):
```
r_t falls → loss increases → but floored at (1-ε)
```
The `min()` prevents the penalty from being too large.

### Visualization

```
         A > 0                          A < 0
L^CLIP │                        L^CLIP │
       │      ___________               │_________
       │     /                          │         \
       │    /                           │          \
───────┼───/──────────── r         ─────┼───────────\── r
       │  1-ε  1  1+ε                  1-ε  1  1+ε
```

### Why It Works

- Simple first-order optimization (just gradient descent)
- No second-order methods needed
- Naturally handles the trust region without explicit constraint
- Works with parameter sharing between actor and critic

### Full PPO Algorithm

```
Algorithm: PPO-Clip

For each iteration:
    1. Collect T timesteps using current π_θ_old
    2. Compute GAE advantages A_1, ..., A_T
    3. For K epochs:
          For each minibatch M ⊂ [1, T]:
              Compute r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
              L^CLIP = E_t[min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)]
              L^VF = E_t[(V_θ(s_t) - R_t)^2]
              L^S = E_t[H(π_θ(·|s_t))]  (entropy)
              L = L^CLIP - c₁ L^VF + c₂ L^S
              Update θ by gradient ascent on L
    4. θ_old ← θ
```

### Multiple Epochs

Unlike A2C (which uses each batch exactly once), PPO runs **K epochs** over the same collected data. The clipping prevents the policy from changing too much across epochs.

Typical value: K = 10 epochs

### Implementation

```python
def ppo_loss(log_probs_new, log_probs_old, advantages, clip_eps=0.2):
    ratio = torch.exp(log_probs_new - log_probs_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()
```

---

## PPO-Penalty

### Alternative Formulation

Instead of clipping, use a KL penalty with adaptive coefficient β:

```
L^KL(θ) = E_t[r_t(θ) × A_t] - β × KL(π_old || π_new)
```

**Adaptive β rule:**
```
if KL > 1.5 × d_target:   β ← β × 2   (too large, increase penalty)
if KL < d_target / 1.5:   β ← β / 2   (too small, decrease penalty)
```

### Comparison

| Feature | PPO-Clip | PPO-Penalty |
|---------|----------|-------------|
| Constraint | Hard clip on ratio | Soft KL penalty |
| Hyperparameter | ε (fixed) | β (adaptive) |
| Simplicity | Simpler | More complex |
| Performance | ≈ Same | ≈ Same |
| Recommendation | Default | When KL control needed |

**PPO-Clip is the standard choice** for most applications.

---

## Practical PPO Implementation

### Hyperparameters

```python
lr = 3e-4           # Learning rate (Adam optimizer)
gamma = 0.99        # Discount factor
lam = 0.95          # GAE lambda
clip_eps = 0.2      # PPO clip parameter
ppo_epochs = 10     # Number of epochs per update
minibatch_size = 64 # Minibatch size
entropy_coeff = 0.01 # Entropy bonus weight
value_coeff = 0.5   # Value loss weight
max_grad_norm = 0.5 # Gradient clipping
rollout_length = 2048 # Steps per rollout
target_kl = 0.02    # Early stop threshold
```

### Value Function Clipping (Optional)

Also clip the value function update:
```python
v_clipped = old_value + clip(new_value - old_value, -ε, +ε)
v_loss = max((new_value - target)², (v_clipped - target)²)
```

### Early KL Stopping

Stop PPO epochs early if KL exceeds target:
```python
for epoch in range(ppo_epochs):
    # ... update ...
    if approx_kl > target_kl:
        break  # Don't update too much
```

### Monitoring Health

| Metric | Healthy Range | Problem |
|--------|--------------|---------|
| KL divergence | < 0.02 | Too large → reduce lr or clip |
| Clip fraction | ~5-20% | 0% → too conservative; 50%+ → too aggressive |
| Policy entropy | Slowly decreasing | Rapid drop → premature convergence |
| Value loss | Decreasing | Diverging → reduce value_coeff |

### Network Architecture Tips

```python
# Orthogonal initialization
nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
# Smaller gain for output heads
nn.init.orthogonal_(actor_head.weight, gain=0.01)
nn.init.orthogonal_(critic_head.weight, gain=1.0)

# Tanh activations (better than ReLU for PPO)
nn.Tanh()
```

---

## Continuous Actions

### Gaussian Policy

```python
class ContinuousActor(nn.Module):
    def __init__(self):
        self.actor_mean = nn.Linear(hidden, action_size)
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))

    def get_dist(self, state):
        mean = self.actor_mean(state)
        std = self.actor_log_std.exp()
        return Normal(mean, std)
```

### Log Probability

For continuous actions, sum log prob over action dimensions:
```python
dist = Normal(mean, std)
log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dims!
```

### Importance Ratio

Same formula as discrete:
```python
ratio = exp(log_prob_new - log_prob_old)
```

---

## Code Examples

This folder contains 7 example scripts:

### 1. `01_trpo_concept.py`
TRPO background and motivation.
- Catastrophic update visualization
- KL divergence as policy distance
- Importance sampling ratio
- Why TRPO is complex

**Run:** `python 01_trpo_concept.py`

### 2. `02_ppo_clip.py`
PPO-Clip core implementation.
- Clip objective visualization
- Multiple epochs on same data
- Training on CartPole-v1
- KL monitoring

**Run:** `python 02_ppo_clip.py`

### 3. `03_ppo_penalty.py`
PPO-Penalty with adaptive KL.
- KL penalty objective
- Adaptive beta mechanism
- Clip vs Penalty comparison
- Training on CartPole-v1

**Run:** `python 03_ppo_penalty.py`

### 4. `04_ppo_cartpole.py`
Complete production PPO.
- Value function clipping
- Early KL stopping
- Comprehensive monitoring (clip fraction, KL, entropy)
- Model checkpointing

**Run:** `python 04_ppo_cartpole.py`

### 5. `05_ppo_continuous.py`
PPO for continuous actions.
- Gaussian actor implementation
- Training on Pendulum-v1
- Standard deviation evolution
- Action clipping

**Run:** `python 05_ppo_continuous.py`

### 6. `06_ppo_lunarlander.py`
PPO on LunarLander-v2.
- Harder environment requiring tuning
- Environment description
- Hyperparameter guide for LunarLander

**Run:** `python 06_ppo_lunarlander.py`

### 7. `07_comparison.py`
Final algorithm comparison.
- DQN vs A2C vs PPO side-by-side
- Sample efficiency analysis
- Complete curriculum summary
- Algorithm selection guide

**Run:** `python 07_comparison.py`

---

## Exercises

### Exercise 1: PPO-Clip from Scratch

Implement PPO-Clip without looking at the provided code:
1. Implement the clip loss function
2. Set up rollout collection
3. Run K epochs over the same data
4. Train on CartPole-v1

**Goal:** Deeply understand the PPO update mechanism.

### Exercise 2: Epsilon Sensitivity

Study the effect of clip epsilon ε:
1. Train with ε = 0.05, 0.1, 0.2, 0.3, 0.5
2. Plot learning curves and final performance
3. Track clip fraction for each ε
4. Find the optimal ε for your environment

**Goal:** Understand the role of the clip parameter.

### Exercise 3: Epochs Sensitivity

Study the effect of number of PPO epochs K:
1. Train with K = 1, 4, 10, 20, 40
2. Track KL divergence during updates
3. Compare final performance
4. Find the point where more epochs hurts

**Goal:** Understand why we use multiple epochs and the limit.

### Exercise 4: Continuous Control

Apply PPO to MountainCarContinuous-v0:
1. Implement Gaussian actor-critic
2. Use GAE with λ = 0.95
3. Tune entropy coefficient
4. Compare with fixed vs state-dependent std

**Goal:** Master PPO for continuous action spaces.

### Exercise 5: LunarLander Challenge

Solve LunarLander-v2 with PPO:
1. Use the recommended hyperparameters (γ=0.999, λ=0.98)
2. Run 3 different seeds and plot mean ± std
3. Achieve average score ≥ 200
4. Compare training time with DQN (Week 8) and A2C (Week 10)

**Goal:** Solve a challenging environment with PPO.

---

## Key Equations

### Importance Ratio
```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) = exp(log π_new - log π_old)
```

### PPO-Clip Objective
```
L^CLIP(θ) = E_t[ min(r_t A_t,  clip(r_t, 1-ε, 1+ε) A_t) ]
```

### PPO-Penalty Objective
```
L^KL(θ) = E_t[ r_t A_t ] - β × KL(π_old || π_new)
```

### Full PPO Loss
```
L = L^CLIP - c₁ × L^VF + c₂ × H(π)

L^VF = E_t[(V_θ(s_t) - R_t)²]
H(π) = -E_t[Σ_a π(a|s) log π(a|s)]
```

### GAE (from Week 10)
```
A^GAE_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

---

## Practical Tips

### Default Hyperparameters (Works for Most Environments)

```python
# General purpose
lr = 3e-4
gamma = 0.99
lam = 0.95
clip_eps = 0.2
ppo_epochs = 10
minibatch_size = 64
rollout_length = 2048
entropy_coeff = 0.01
value_coeff = 0.5
max_grad_norm = 0.5
```

### Environment-Specific Tuning

| Environment | gamma | lam | hidden | notes |
|------------|-------|-----|--------|-------|
| CartPole | 0.99 | 0.95 | 64 | Default settings |
| LunarLander | 0.999 | 0.98 | 256 | Higher γ, λ, bigger net |
| Pendulum | 0.99 | 0.95 | 64 | entropy_coeff=0.0 |
| Ant/HalfCheetah | 0.99 | 0.95 | 256 | Multiple parallel envs |

### Common Pitfalls

1. **Not normalizing advantages:** Always normalize per minibatch
2. **Wrong log prob for continuous:** Forget to sum over action dims
3. **Not clipping gradients:** Can cause instability
4. **Too few rollout steps:** Biased advantage estimates
5. **High entropy_coeff:** Prevents learning in complex environments

---

## Historical Context

### 2015: TRPO (Schulman et al.)
- First principled trust region method for deep RL
- Guaranteed monotonic improvement
- Complex second-order optimization

### 2016: GAE (Schulman et al.)
- Generalized Advantage Estimation
- Combined with TRPO for continuous control

### 2017: PPO (Schulman et al.)
- Simplified TRPO with clipping
- Same performance, 10x simpler
- Became the default algorithm in OpenAI

### 2018-present: PPO Dominance
- Used in OpenAI Five (Dota 2)
- OpenAI Gym baselines default
- Foundation of many SOTA algorithms
- Still competitive in 2025+

---

## Next Week

In Week 12, we cover **Advanced Topics**:
- Multi-agent RL
- Model-based RL
- Inverse RL
- Final project: Solve your own environment

---

## Resources

### Papers
- [Proximal Policy Optimization (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) - PPO
- [Trust Region Policy Optimization (Schulman et al., 2015)](https://arxiv.org/abs/1502.05477) - TRPO
- [High-Dimensional Continuous Control Using GAE (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438) - GAE

### Tutorials
- [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) - OpenAI
- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) - Huang et al.
- [CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) - Single-file reference

### Code
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [OpenAI Baselines PPO2](https://github.com/openai/baselines/tree/master/baselines/ppo2)

---

## Summary

**What We Learned:**

1. Large policy updates can catastrophically collapse performance
2. TRPO constrains KL divergence — theoretically sound but complex
3. PPO-Clip: `min(r A, clip(r, 1-ε, 1+ε) A)` — simple and effective
4. PPO-Penalty: adaptive KL penalty with β adjustment
5. Multiple epochs (K=10) over same data → better sample efficiency
6. PPO works for discrete and continuous, same hyperparameters
7. GAE + PPO-Clip = the de facto standard in deep RL

**Key Insight:**

PPO achieves TRPO-level safety guarantees with simple first-order optimization. The clipping mechanism is a brilliant heuristic: it prevents the policy ratio from straying too far from 1, keeping updates conservative without complex constrained optimization.

**Next Step:**

Week 12 covers advanced topics: multi-agent RL, model-based RL, and inverse RL — extending the foundations built in Weeks 1-11.

---

**Continue to Week 12: [Advanced Topics and Project](../12_advanced/)** →
