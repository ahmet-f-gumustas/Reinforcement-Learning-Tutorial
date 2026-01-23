# Week 5: Temporal Difference Learning

This week we learn Temporal Difference (TD) methods - the heart of modern reinforcement learning that combines the best of Monte Carlo and Dynamic Programming.

## Contents

1. [Introduction to TD Learning](#introduction-to-td-learning)
2. [TD(0) Prediction](#td0-prediction)
3. [TD vs MC vs DP](#td-vs-mc-vs-dp)
4. [n-step TD Methods](#n-step-td-methods)
5. [TD(λ) and Eligibility Traces](#td-and-eligibility-traces)
6. [Code Examples](#code-examples)
7. [Exercises](#exercises)

---

## Introduction to TD Learning

**Temporal Difference learning** is a central idea in reinforcement learning that learns from experience like MC, but can update estimates before the end of an episode like DP.

### Key Characteristics

1. **Model-Free**: No need for environment dynamics (like MC)
2. **Bootstrapping**: Updates based on other estimates (like DP)
3. **Online**: Can learn at every time step (unlike MC)
4. **Incomplete Sequences**: Works with non-terminating episodes

### The Big Idea

Instead of waiting until the end of an episode to know the actual return G_t (Monte Carlo), TD methods update immediately after each step using an estimate of the return.

---

## TD(0) Prediction

**TD(0)** is the simplest temporal difference method for policy evaluation.

### The Update Rule

```
V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
```

Where:
- `α`: learning rate (step size)
- `R_{t+1}`: immediate reward
- `γ`: discount factor
- `V(S_{t+1})`: bootstrap estimate (the TD part!)

### TD Error

The quantity in brackets is called the **TD error**:

```
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
```

This measures the difference between:
- **TD target**: `R_{t+1} + γV(S_{t+1})` (estimate of return)
- **Current estimate**: `V(S_t)`

### TD(0) Algorithm

```
Initialize V(s) arbitrarily for all s
Repeat (for each episode):
    Initialize S
    Repeat (for each step):
        A ← action given by policy for S
        Take action A, observe R, S'
        V(S) ← V(S) + α[R + γV(S') - V(S)]
        S ← S'
    until S is terminal
```

### Why It Works

TD(0) is a **sample-based** version of the Bellman equation:
- DP: `V(s) = E[R + γV(S')]` (uses true expectation)
- TD: `V(s) ← V(s) + α[sample of (R + γV(S')) - V(s)]`

---

## TD vs MC vs DP

Let's compare the three main approaches to policy evaluation.

### Update Targets

| Method | Target | Bootstraps? | Samples? |
|--------|--------|-------------|----------|
| **DP** | `E[R + γV(S')]` | Yes | No (uses P) |
| **MC** | `G_t (actual return)` | No | Yes |
| **TD** | `R + γV(S')` | Yes | Yes |

### Comparison Table

| Aspect | DP | MC | TD |
|--------|----|----|-----|
| Model Required | Yes | No | No |
| Bootstrapping | Yes | No | Yes |
| Episodes Required | No | Yes (complete) | No |
| Update Timing | Sweep all states | End of episode | Every step |
| Bias | Low | None (unbiased) | Medium |
| Variance | Low | High | Low |
| Convergence | Guaranteed* | Guaranteed | Guaranteed* |

*under certain conditions

### Conceptual Differences

**Monte Carlo**: "Wait and see"
- Wait until episode ends
- Use actual return G_t
- No assumptions about future

**Temporal Difference**: "Guess and update"
- Update immediately
- Use estimated return R + γV(S')
- Bootstraps from current value estimates

**Dynamic Programming**: "Use the model"
- Sweep through all states
- Use exact expectation from model
- No actual experience needed

### When to Use What

**Use TD when:**
- Episodes are very long or non-terminating
- You want online learning
- You need low variance estimates

**Use MC when:**
- Episodes are short
- You want unbiased estimates
- Environment has high stochasticity

**Use DP when:**
- You have a perfect model
- State space is small
- You can afford full sweeps

---

## n-step TD Methods

TD(0) uses a 1-step return. MC uses an infinite-step return. What about something in between?

### The n-step Return

**1-step return (TD):**
```
G_t^(1) = R_{t+1} + γV(S_{t+1})
```

**2-step return:**
```
G_t^(2) = R_{t+1} + γR_{t+2} + γ²V(S_{t+2})
```

**n-step return:**
```
G_t^(n) = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})
```

**∞-step return (MC):**
```
G_t^(∞) = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
```

### n-step TD Update

```
V(S_t) ← V(S_t) + α[G_t^(n) - V(S_t)]
```

### The n-step Spectrum

```
n=1 ──────── n=2 ──────── n=5 ──────── n=∞
TD(0)     Low variance              High variance
High bias                           Low bias (MC)
```

### Choosing n

- **Small n (1-3)**: Faster learning, more bias
- **Medium n (5-10)**: Often best in practice
- **Large n (→∞)**: Lower bias, but higher variance

The optimal n depends on:
- Episode length
- Problem complexity
- Reward structure

---

## TD(λ) and Eligibility Traces

Instead of choosing a single n, what if we average over all n-step returns?

### λ-return

The **λ-return** is a weighted average of all n-step returns:

```
G_t^λ = (1-λ) Σ_{n=1}^∞ λ^{n-1} G_t^(n)
```

Where λ ∈ [0, 1] controls the weighting:
- λ = 0: G_t^λ = G_t^(1) (pure TD)
- λ = 1: G_t^λ = G_t (pure MC)

### Forward View: TD(λ)

The forward view looks ahead at all future returns:

```
V(S_t) ← V(S_t) + α[G_t^λ - V(S_t)]
```

But this requires waiting until the end of the episode!

### Backward View: Eligibility Traces

The **backward view** gives us an online algorithm using **eligibility traces**:

```
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)    # TD error
E(s) ← γλE(s) + 1(S_t = s)               # Update trace
V(s) ← V(s) + αδ_t E(s)                  # Update all states
```

### Eligibility Trace Intuition

Think of eligibility as "credit assignment":
- States visited recently have high eligibility
- When TD error occurs, all eligible states are updated
- Traces decay over time (γλ decay rate)

### TD(λ) Algorithm

```
Initialize V(s) arbitrarily
Repeat (for each episode):
    E(s) = 0 for all s
    Initialize S
    Repeat (for each step):
        A ← action from policy
        Take action A, observe R, S'
        δ ← R + γV(S') - V(S)
        E(S) ← E(S) + 1
        For all s:
            V(s) ← V(s) + α·δ·E(s)
            E(s) ← γλE(s)
        S ← S'
    until S is terminal
```

### Why Eligibility Traces?

1. **Credit Assignment**: Spread credit to all recent states
2. **Online Learning**: No need to wait for episode end
3. **Flexibility**: Interpolate between TD and MC
4. **Faster Learning**: Often converges faster than TD(0)

---

## Random Walk Example

The classic **Random Walk** problem from Sutton & Barto is perfect for demonstrating TD.

### The Problem

```
[Terminal] ← A ← B ← C ← D ← E → [Terminal]
  (reward -1)               (reward +1)
```

- 5 non-terminal states (A, B, C, D, E)
- Start in C (center)
- Actions: Left or Right (random policy)
- Left terminal: reward -1
- Right terminal: reward +1

### True Values

With equal probability of going left/right:
```
V_true(A) = -0.8
V_true(B) = -0.6
V_true(C) =  0.0
V_true(D) = +0.6
V_true(E) = +0.8
```

### Why This Example?

- Small enough to compute true values
- TD converges faster than MC
- Easy to visualize learning curves
- Shows bias-variance tradeoff

---

## Code Examples

This folder contains 7 example scripts:

### 1. `01_td_prediction.py`
Basic TD(0) prediction for policy evaluation.
- TD(0) algorithm implementation
- Convergence to true values
- Comparison with MC prediction
- GridWorld example

### 2. `02_td_vs_mc.py`
Head-to-head comparison of TD, MC, and DP.
- Same policy, same environment
- Learning curves comparison
- RMS error analysis
- Sample efficiency

### 3. `03_n_step_td.py`
Exploring n-step TD methods.
- n-step returns implementation
- Effect of n on learning speed
- Finding optimal n
- GridWorld experiments

### 4. `04_td_lambda.py`
TD(λ) with eligibility traces.
- Forward view (offline)
- Backward view (online) with traces
- Effect of λ parameter
- Credit assignment visualization

### 5. `05_random_walk.py`
Classic Random Walk problem.
- 1D state space implementation
- True value computation
- Batch vs incremental TD
- Learning curve plots

### 6. `06_windy_gridworld.py`
Stochastic environment with wind.
- Custom windy gridworld
- Stochastic transitions
- TD prediction under noise
- Path visualization

### 7. `07_td_visualization.py`
Visualization tools for TD learning.
- Value function evolution
- TD error over time
- Eligibility trace heatmaps
- Interactive parameter exploration

---

## Exercises

### Exercise 1: TD vs MC Convergence

Implement both TD(0) and first-visit MC prediction on a GridWorld.
- Use the same random policy for both
- Track RMS error over episodes
- Which converges faster?
- How does α affect convergence?

### Exercise 2: Finding Optimal n

Experiment with n-step TD for different values of n (1, 2, 5, 10, 20).
- Plot learning curves for each n
- Which n works best?
- Does it depend on episode length?
- How does computational cost scale?

### Exercise 3: Random Walk

Implement the 5-state Random Walk:
1. Compute true values analytically
2. Run TD(0) with different α values
3. Run MC with different α values
4. Compare convergence speed and final accuracy

### Exercise 4: Eligibility Traces

Implement TD(λ) with eligibility traces:
1. Test λ = 0 (pure TD) and λ = 1 (MC-like)
2. Find the best λ for your problem
3. Visualize eligibility trace evolution
4. Compare with n-step TD

### Exercise 5: Batch Learning

Implement batch TD:
- Collect a fixed set of episodes
- Repeatedly update on this batch
- Compare with incremental TD
- Does it converge to better values?

---

## Key Equations

### TD(0) Update
```
V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
```

### TD Error
```
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
```

### n-step Return
```
G_t^(n) = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})
```

### λ-return
```
G_t^λ = (1-λ) Σ_{n=1}^∞ λ^{n-1} G_t^(n)
```

### Eligibility Trace Update
```
E_t(s) = γλE_{t-1}(s) + 1(S_t = s)
```

### TD(λ) with Traces
```
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
V(s) ← V(s) + αδ_t E_t(s)  for all s
```

---

## Next Week

In Week 6, we will cover **Q-Learning and SARSA**:
- On-policy TD control (SARSA)
- Off-policy TD control (Q-Learning)
- Expected SARSA
- Maximization bias and Double Q-Learning

These are TD control methods - using TD to find optimal policies, not just evaluate them!

---

## Resources

- [Sutton & Barto Chapter 6: Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf)
- [Sutton & Barto Chapter 7: n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf)
- [Sutton & Barto Chapter 12: Eligibility Traces](http://incompleteideas.net/book/RLbook2020.pdf)
- [David Silver Lecture 4: Model-Free Prediction](https://www.youtube.com/watch?v=PnHCvfgC_ZA)
- [Richard Sutton: The Importance of TD Learning](https://www.youtube.com/watch?v=EeMCEQa85tw)
