# Week 7: Function Approximation

This week we move beyond tabular methods to handle large or continuous state spaces using function approximation.

## Contents

1. [Why Function Approximation?](#why-function-approximation)
2. [Linear Function Approximation](#linear-function-approximation)
3. [Feature Engineering](#feature-engineering)
4. [Gradient Methods](#gradient-methods)
5. [Semi-Gradient TD](#semi-gradient-td)
6. [The Deadly Triad](#the-deadly-triad)
7. [Code Examples](#code-examples)
8. [Exercises](#exercises)

---

## Why Function Approximation?

### The Problem with Tabular Methods

Tabular methods store a value for **every state** (or state-action pair):
- GridWorld 10×10: 100 states ✓
- Chess: ~10^43 states ✗
- Robot control: Continuous states ✗

### The Solution: Generalization

Instead of storing V(s) for every s, we **approximate** it:

```
V(s) ≈ v̂(s, w)
```

Where `w` is a parameter vector (weights) that we learn.

### Benefits of Function Approximation

1. **Memory Efficiency**: Store weights instead of all values
2. **Generalization**: Similar states get similar values
3. **Continuous States**: Handle real-valued state spaces
4. **Faster Learning**: Update affects similar states

### The Tradeoff

- **Tabular**: Exact values, no generalization
- **Function Approx**: Approximate values, generalization

---

## Linear Function Approximation

The simplest form of function approximation uses a **linear combination** of features.

### The Model

```
v̂(s, w) = w^T x(s) = Σ w_i x_i(s)
```

Where:
- `x(s)` is a **feature vector** for state s
- `w` is the **weight vector** (parameters)
- Output is a weighted sum of features

### Example

State s = (position, velocity) in Mountain Car:

```
Features x(s):
  x_1 = position
  x_2 = velocity
  x_3 = position²
  x_4 = velocity²
  x_5 = position × velocity

Value: v̂(s, w) = w_1·pos + w_2·vel + w_3·pos² + w_4·vel² + w_5·pos·vel
```

### Why Linear?

1. **Simple**: Easy to understand and implement
2. **Guaranteed Convergence**: For on-policy methods
3. **Foundation**: Basis for understanding nonlinear methods
4. **Interpretable**: Can understand what features matter

### Gradient of Linear Function

```
∇_w v̂(s, w) = x(s)
```

The gradient is simply the feature vector! This makes updates very efficient.

---

## Feature Engineering

The choice of features **determines what can be learned**.

### Types of Features

#### 1. State Aggregation
Group similar states together:
```
x_i(s) = 1 if s in group i, else 0
```

#### 2. Polynomial Features
```
x(s) = [1, s, s², s³, ...]
```

#### 3. Fourier Basis
```
x_i(s) = cos(πi·s)  for i = 0, 1, 2, ...
```

#### 4. Tile Coding (Coarse Coding)
Multiple overlapping grids:
```
Each tile = binary feature (1 if state in tile)
Multiple tilings with offsets
```

#### 5. Radial Basis Functions (RBF)
```
x_i(s) = exp(-||s - c_i||² / 2σ²)
```

Gaussian bumps centered at c_i.

### Tile Coding in Detail

Tile coding is one of the most effective methods:

```
State Space:
+---+---+---+---+
|   |   | * |   |  ← State is here
+---+---+---+---+
|   |   |   |   |
+---+---+---+---+

Tiling 1:        Tiling 2 (offset):
+---+---+---+   +---+---+---+
| 1 | 2*| 3 |   |   | 1*| 2 |  ← Different boundaries
+---+---+---+   +---+---+---+

Active features: tile 2 from tiling 1, tile 1 from tiling 2
```

Benefits:
- Binary features (fast)
- Good generalization
- Easy to implement

### Feature Selection Guidelines

1. **Include constant**: x_0 = 1 (bias term)
2. **Normalize features**: Keep similar scales
3. **Start simple**: Add complexity as needed
4. **Domain knowledge**: Use meaningful features

---

## Gradient Methods

We use **gradient descent** to learn the weights.

### The Objective

Minimize Mean Squared Value Error:

```
VE(w) = Σ_s μ(s) [v_π(s) - v̂(s, w)]²
```

Where μ(s) is the distribution of states visited.

### Gradient Descent Update

```
w ← w - α ∇_w VE(w)
w ← w + α [v_π(s) - v̂(s, w)] ∇_w v̂(s, w)
```

But we don't know v_π(s)! We need to estimate it.

### Gradient Monte Carlo

Use the **actual return** G_t as target:

```
w ← w + α [G_t - v̂(S_t, w)] ∇_w v̂(S_t, w)
```

For linear approximation:
```
w ← w + α [G_t - v̂(S_t, w)] x(S_t)
```

### Properties of Gradient MC

- **Unbiased**: G_t is unbiased estimate of v_π(S_t)
- **High variance**: Full episode needed
- **Guaranteed convergence**: To local optimum (global for linear)

---

## Semi-Gradient TD

### The Idea

Use **TD target** instead of actual return:

```
w ← w + α [R + γv̂(S', w) - v̂(S, w)] ∇_w v̂(S, w)
```

### Why "Semi-Gradient"?

The full gradient would include the derivative of the target:

```
Full gradient: ∇_w [R + γv̂(S', w) - v̂(S, w)]²
             = 2[R + γv̂(S', w) - v̂(S, w)][γ∇_w v̂(S', w) - ∇_w v̂(S, w)]
```

But we **ignore** the ∇_w v̂(S', w) term:

```
Semi-gradient: 2[R + γv̂(S', w) - v̂(S, w)][-∇_w v̂(S, w)]
```

This makes training stable and is standard practice.

### Semi-Gradient TD(0) Algorithm

```
Initialize w arbitrarily
Repeat (for each episode):
    Initialize S
    Repeat (for each step):
        A ← action from policy
        Take action A, observe R, S'
        w ← w + α [R + γv̂(S', w) - v̂(S, w)] ∇_w v̂(S, w)
        S ← S'
    until S is terminal
```

### Linear Semi-Gradient TD(0)

```
w ← w + α [R + γ w^T x(S') - w^T x(S)] x(S)
w ← w + α δ x(S)
```

Where δ = R + γ w^T x(S') - w^T x(S) is the TD error.

---

## Control with Function Approximation

### Semi-Gradient SARSA

For control, we approximate **Q-values**:

```
q̂(s, a, w) ≈ Q(s, a)
```

SARSA update:
```
w ← w + α [R + γq̂(S', A', w) - q̂(S, A, w)] ∇_w q̂(S, A, w)
```

### Episodic Semi-Gradient SARSA Algorithm

```
Initialize w arbitrarily
Repeat (for each episode):
    S ← initial state
    A ← ε-greedy action from q̂(S, ·, w)
    Repeat (for each step):
        Take action A, observe R, S'
        If S' is terminal:
            w ← w + α [R - q̂(S, A, w)] ∇_w q̂(S, A, w)
            break
        A' ← ε-greedy from q̂(S', ·, w)
        w ← w + α [R + γq̂(S', A', w) - q̂(S, A, w)] ∇_w q̂(S, A, w)
        S ← S'; A ← A'
```

### Q-value Features

For linear approximation with actions:

**Option 1**: Separate weights per action
```
q̂(s, a, w) = w_a^T x(s)
```

**Option 2**: Action as feature
```
x(s, a) = [x(s) ⊗ one_hot(a)]  (concatenate or tensor product)
q̂(s, a, w) = w^T x(s, a)
```

---

## The Deadly Triad

### Three Dangerous Ingredients

When these three combine, learning can **diverge**:

1. **Function Approximation**: v̂(s, w) instead of table
2. **Bootstrapping**: Using v̂(S') in target (TD, not MC)
3. **Off-Policy Learning**: Learning about π while following b

### Why It's Dangerous

- Off-policy: Distribution mismatch between updates and target policy
- Bootstrapping: Errors compound (self-referential)
- Function approx: Updates affect multiple states

Together: Small errors can amplify and diverge!

### Examples of Divergence

**Baird's Counterexample**:
- Simple 7-state MDP
- Linear function approximation
- Off-policy TD
- Weights diverge to infinity!

### Solutions

1. **Avoid off-policy**: Use on-policy methods (SARSA)
2. **Avoid bootstrapping**: Use Monte Carlo
3. **Gradient TD methods**: True gradient (more complex)
4. **Experience replay**: Stabilize distribution
5. **Target networks**: Stabilize targets (DQN)

### Safe Combinations

| Method | Approx | Bootstrap | Off-Policy | Stable? |
|--------|--------|-----------|------------|---------|
| Tabular TD | No | Yes | Yes | ✓ |
| Linear MC | Yes | No | Yes | ✓ |
| Linear TD | Yes | Yes | No | ✓ |
| Linear TD | Yes | Yes | Yes | ✗ |
| DQN | Yes | Yes | Yes | ✓* |

*With experience replay and target networks

---

## Convergence Properties

### Linear On-Policy TD

**Converges** to a unique fixed point:

```
w_TD ≈ w* (near optimal)
```

The error is bounded:
```
VE(w_TD) ≤ 1/(1-γ) min_w VE(w)
```

### Linear Off-Policy TD

**May diverge** (deadly triad)

Solutions:
- Gradient TD (TDC, GTD2)
- Emphatic TD

### Nonlinear Function Approximation

- Even on-policy TD may not converge
- Local optima possible
- Need more sophisticated methods (DQN, etc.)

---

## Code Examples

This folder contains 7 example scripts:

### 1. `01_linear_approximation.py`
Introduction to linear function approximation.
- Weight vectors and features
- Linear value prediction
- Gradient computation
- Simple examples

### 2. `02_feature_engineering.py`
Different feature representations.
- Polynomial features
- Fourier basis
- Tile coding implementation
- RBF networks
- Feature comparison

### 3. `03_gradient_mc.py`
Gradient Monte Carlo prediction.
- Full return targets
- Convergence analysis
- Comparison with tabular MC
- Feature impact study

### 4. `04_semi_gradient_td.py`
Semi-gradient TD(0) prediction.
- TD with function approximation
- Linear TD convergence
- Comparison with gradient MC
- Online learning

### 5. `05_linear_sarsa.py`
Semi-gradient SARSA for control.
- Linear Q-value approximation
- Epsilon-greedy control
- Mountain Car example
- Learning curves

### 6. `06_mountain_car.py`
Solving Mountain Car with function approximation.
- Continuous state space
- Tile coding features
- SARSA with linear approx
- Visualization

### 7. `07_deadly_triad.py`
Demonstration of the deadly triad.
- Baird's counterexample
- Divergence visualization
- Safe vs unsafe combinations
- Solutions overview

---

## Exercises

### Exercise 1: Polynomial Features

Implement polynomial feature extraction:
1. Create features [1, s, s², ..., s^n]
2. Train linear approximation on Random Walk
3. Compare different polynomial degrees
4. Plot learned value function

### Exercise 2: Tile Coding

Implement tile coding from scratch:
1. Create overlapping tilings
2. Convert continuous state to feature vector
3. Apply to Mountain Car
4. Compare with polynomial features

### Exercise 3: Semi-Gradient TD vs MC

Compare semi-gradient TD and gradient MC:
1. Implement both on same environment
2. Track MSE over episodes
3. Which converges faster?
4. Which has lower final error?

### Exercise 4: Mountain Car

Solve Mountain Car with linear function approximation:
1. Design appropriate features
2. Implement semi-gradient SARSA
3. Tune hyperparameters (α, ε, features)
4. Visualize learned policy

### Exercise 5: Deadly Triad

Reproduce Baird's counterexample:
1. Implement the 7-state MDP
2. Run off-policy TD with linear approx
3. Observe divergence
4. Try on-policy TD (should work)

---

## Key Equations

### Linear Value Function
```
v̂(s, w) = w^T x(s) = Σ_i w_i x_i(s)
```

### Gradient of Linear Function
```
∇_w v̂(s, w) = x(s)
```

### Gradient MC Update
```
w ← w + α [G_t - v̂(S_t, w)] x(S_t)
```

### Semi-Gradient TD Update
```
w ← w + α [R + γv̂(S', w) - v̂(S, w)] x(S)
```

### Semi-Gradient SARSA Update
```
w ← w + α [R + γq̂(S', A', w) - q̂(S, A, w)] ∇_w q̂(S, A, w)
```

### Value Error
```
VE(w) = Σ_s μ(s) [v_π(s) - v̂(s, w)]²
```

---

## Next Week

In Week 8, we will cover **Deep Q-Networks (DQN)**:
- Neural network function approximation
- Experience replay
- Target networks
- Atari game playing

DQN is the breakthrough that made deep reinforcement learning work!

---

## Resources

- [Sutton & Barto Chapter 9: On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf)
- [Sutton & Barto Chapter 10: On-policy Control with Approximation](http://incompleteideas.net/book/RLbook2020.pdf)
- [Sutton & Barto Chapter 11: Off-policy Methods with Approximation](http://incompleteideas.net/book/RLbook2020.pdf)
- [David Silver Lecture 6: Value Function Approximation](https://www.youtube.com/watch?v=UoPei5o4fps)
- [Tile Coding Software](http://incompleteideas.net/tiles/tiles3.html)
