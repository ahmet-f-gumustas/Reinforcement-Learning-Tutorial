# Week 8: Deep Q-Network (DQN)

This week we introduce Deep Q-Networks, the breakthrough that enabled deep reinforcement learning to achieve human-level performance on Atari games.

## Contents

1. [From Function Approximation to Deep Learning](#from-function-approximation-to-deep-learning)
2. [The DQN Architecture](#the-dqn-architecture)
3. [Experience Replay](#experience-replay)
4. [Target Networks](#target-networks)
5. [DQN Training Algorithm](#dqn-training-algorithm)
6. [DQN Variants](#dqn-variants)
7. [Code Examples](#code-examples)
8. [Exercises](#exercises)

---

## From Function Approximation to Deep Learning

### The Leap from Linear to Deep

Last week we learned linear function approximation:

```
Q(s, a) ≈ w^T φ(s, a)
```

But linear approximation has limitations:
- Requires **manual feature engineering**
- Cannot capture complex nonlinear relationships
- Limited representational power

### Enter Neural Networks

Neural networks can **automatically learn features**:

```
Q(s, a) ≈ NN(s, a; θ)
```

Where θ represents all network weights.

### Why Deep Learning for RL?

1. **Automatic Feature Learning**: No manual feature engineering
2. **Universal Approximation**: Can approximate any function
3. **End-to-End Learning**: Learn directly from raw inputs (pixels)
4. **Scalability**: Handle high-dimensional state spaces

### The Challenge

Combining deep learning with RL is hard because:
- Data is **sequential and correlated**
- Target values are **non-stationary** (change as we learn)
- Small updates can cause **catastrophic changes**

DQN solves these with **Experience Replay** and **Target Networks**.

---

## The DQN Architecture

### Network Structure

```
Input (State) → Hidden Layers → Output (Q-values for all actions)
    [s]       →  [FC→ReLU]×n  →  [Q(s,a₁), Q(s,a₂), ..., Q(s,aₙ)]
```

### For Discrete Action Spaces

Input: State vector s
Output: Q-value for **each action**

```python
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No activation - Q-values can be any real number
```

### For Image Inputs (Atari-style)

```
[84×84×4 frames] → Conv → Conv → Conv → FC → FC → Q-values
```

```python
class AtariDQN(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_size)
```

### Action Selection

Given state s, choose action:

```python
def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)  # Explore
    else:
        q_values = network(state)
        return q_values.argmax()  # Exploit
```

---

## Experience Replay

### The Problem: Correlated Data

In supervised learning, we assume **i.i.d.** data (independent and identically distributed).

But in RL, consecutive experiences are **highly correlated**:
- s₁ → s₂ → s₃ → ...

Training on correlated data causes:
- **Overfitting** to recent experiences
- **Forgetting** earlier knowledge
- **Unstable learning**

### The Solution: Replay Buffer

Store experiences and sample **randomly**:

```
Replay Buffer: [(s₁,a₁,r₁,s₁'), (s₂,a₂,r₂,s₂'), ..., (sₙ,aₙ,rₙ,sₙ')]
                     ↓
              Random sampling
                     ↓
              Mini-batch for training
```

### Implementation

```python
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(next_states),
                np.array(dones))

    def __len__(self):
        return len(self.buffer)
```

### Benefits of Experience Replay

1. **Breaks correlation**: Random sampling decorrelates data
2. **Data efficiency**: Each experience used multiple times
3. **Stable learning**: Diverse mini-batches
4. **Better convergence**: Similar to supervised learning

### Prioritized Experience Replay (Extension)

Not all experiences are equal. Prioritize based on **TD-error**:

```
Priority(i) ∝ |δᵢ|^α  where δᵢ = r + γ max Q(s') - Q(s, a)
```

Higher TD-error = more surprising = more to learn.

---

## Target Networks

### The Moving Target Problem

In Q-learning, we update towards a **target**:

```
Target = r + γ max_a' Q(s', a'; θ)
```

But θ changes every update! The target **moves** as we train.

This causes:
- **Oscillations**
- **Divergence**
- **Unstable learning**

### The Solution: Frozen Target Network

Use a **separate network** for computing targets:

```
Policy Network: θ   (updated every step)
Target Network: θ⁻  (updated periodically)

Target = r + γ max_a' Q(s', a'; θ⁻)  # Uses frozen θ⁻
```

### Update Schedule

Two approaches:

**1. Hard Update** (Original DQN):
```python
# Every C steps
target_network.load_state_dict(policy_network.state_dict())
```

**2. Soft Update** (Smoother):
```python
# Every step
for target_param, policy_param in zip(target_net.parameters(),
                                       policy_net.parameters()):
    target_param.data.copy_(τ * policy_param.data + (1-τ) * target_param.data)
```

Where τ (tau) is typically 0.001 to 0.01.

### Benefits

1. **Stable targets**: Q-value targets don't change rapidly
2. **Reduced oscillations**: Smoother learning
3. **Better convergence**: More like supervised learning

---

## DQN Training Algorithm

### Complete Algorithm

```
Initialize:
    - Policy network Q with random weights θ
    - Target network Q⁻ with weights θ⁻ = θ
    - Replay buffer D with capacity N
    - Exploration rate ε

For each episode:
    Initialize state s

    For each step:
        # Action selection (ε-greedy)
        With probability ε: random action a
        Otherwise: a = argmax_a Q(s, a; θ)

        # Environment step
        Execute action a, observe reward r, next state s'

        # Store experience
        Store (s, a, r, s', done) in D

        # Learning step
        If |D| > batch_size:
            Sample random mini-batch from D

            For each (sⱼ, aⱼ, rⱼ, s'ⱼ, doneⱼ) in batch:
                If done:
                    yⱼ = rⱼ
                Else:
                    yⱼ = rⱼ + γ max_a' Q(s'ⱼ, a'; θ⁻)

            # Gradient descent step
            Loss = (1/batch) Σⱼ (yⱼ - Q(sⱼ, aⱼ; θ))²
            θ ← θ - α ∇_θ Loss

        # Update target network
        Every C steps: θ⁻ ← θ

        # Decay exploration
        ε ← max(ε_min, ε × decay)

        s ← s'
```

### Hyperparameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| γ (gamma) | 0.99 | Discount factor |
| ε_start | 1.0 | Initial exploration |
| ε_end | 0.01 | Final exploration |
| ε_decay | 0.995 | Exploration decay |
| α (lr) | 0.001 | Learning rate |
| batch_size | 32-64 | Mini-batch size |
| buffer_size | 100000 | Replay buffer capacity |
| C | 1000 | Target update frequency |

### Loss Function

Mean Squared Error (MSE) or Huber Loss:

```python
# MSE Loss
loss = F.mse_loss(q_values, target_q_values)

# Huber Loss (more robust to outliers)
loss = F.smooth_l1_loss(q_values, target_q_values)
```

---

## DQN Variants

### 1. Double DQN (DDQN)

**Problem**: Standard DQN **overestimates** Q-values.

Why? Using max creates positive bias:
```
E[max(Q)] ≥ max(E[Q])
```

**Solution**: Decouple action selection from evaluation:

```
Standard DQN:  y = r + γ Q(s', argmax_a Q(s', a; θ⁻); θ⁻)  # Same network
Double DQN:    y = r + γ Q(s', argmax_a Q(s', a; θ); θ⁻)   # Different networks
```

```python
# Double DQN target computation
with torch.no_grad():
    # Select action using policy network
    next_actions = policy_net(next_states).argmax(1)
    # Evaluate using target network
    next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1))
    target_q = rewards + gamma * next_q * (1 - dones)
```

### 2. Dueling DQN

**Idea**: Separate Q into two streams:

```
Q(s, a) = V(s) + A(s, a)
```

Where:
- **V(s)**: State value - "how good is this state?"
- **A(s, a)**: Advantage - "how much better is action a than average?"

```
                    → V(s) stream  →
State → Shared CNN →                 → Q(s,a) = V + (A - mean(A))
                    → A(s,a) stream →
```

```python
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        # Shared feature layer
        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)
        # Combine: Q = V + (A - mean(A))
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q
```

**Benefit**: Better learning when actions don't affect outcome much.

### 3. Rainbow DQN

Combines multiple improvements:

1. **Double DQN** - Reduces overestimation
2. **Dueling DQN** - Separates value and advantage
3. **Prioritized Experience Replay** - Sample important transitions
4. **Multi-step Learning** - n-step returns
5. **Distributional RL** - Learn value distribution
6. **Noisy Networks** - Learned exploration

Each contributes to better performance.

### Comparison

| Variant | Key Idea | Benefit |
|---------|----------|---------|
| DQN | Experience replay + target network | Stable deep RL |
| Double DQN | Decouple selection/evaluation | Reduce overestimation |
| Dueling DQN | Separate V and A | Better state evaluation |
| Prioritized ER | Sample by TD-error | Efficient learning |
| Rainbow | Combine all | State-of-the-art |

---

## Practical Tips

### 1. Hyperparameter Tuning

- Start with **default values** from papers
- **Learning rate** is most important
- Use **smaller networks** first, then scale up

### 2. Debugging DQN

Common issues:
- **Q-values exploding**: Reduce learning rate, use gradient clipping
- **No learning**: Check reward scale, ensure exploration
- **Unstable**: Increase target update frequency

### 3. Reward Shaping

- **Clip rewards** to [-1, 1] for stability
- **Normalize** rewards if highly variable
- Add **small penalties** for longer episodes (if applicable)

### 4. Network Architecture

- Start simple: 2-3 layers, 64-256 units
- ReLU activation (simple and effective)
- No activation on output layer

---

## Code Examples

This folder contains 6 example scripts:

### 1. `01_dqn_basics.py`
Introduction to DQN architecture.
- Neural network Q-function
- Forward pass and action selection
- Basic training loop
- Simple environment test

### 2. `02_experience_replay.py`
Understanding experience replay.
- Replay buffer implementation
- Correlation analysis
- With/without replay comparison
- Visualization of buffer contents

### 3. `03_target_network.py`
Target network mechanics.
- Hard vs soft updates
- Stability analysis
- Target network visualization
- Effect on learning curves

### 4. `04_double_dqn.py`
Double DQN implementation.
- Overestimation problem demonstration
- DDQN algorithm
- Comparison with standard DQN
- Q-value analysis

### 5. `05_dueling_dqn.py`
Dueling architecture.
- Value and advantage streams
- Network implementation
- When dueling helps
- Visual comparison

### 6. `06_cartpole_dqn.py`
Complete DQN on CartPole.
- Full training pipeline
- Hyperparameter settings
- Training visualization
- Model saving/loading

---

## Exercises

### Exercise 1: Basic DQN

Implement DQN from scratch:
1. Create a simple neural network
2. Implement replay buffer
3. Train on CartPole-v1
4. Plot learning curves

### Exercise 2: Hyperparameter Study

Study the effect of hyperparameters:
1. Vary learning rate (0.0001, 0.001, 0.01)
2. Vary batch size (16, 32, 64, 128)
3. Vary target update frequency (100, 1000, 10000)
4. Plot and analyze results

### Exercise 3: Double DQN

Implement and compare:
1. Standard DQN
2. Double DQN
3. Track Q-value estimates
4. Show overestimation reduction

### Exercise 4: Dueling DQN

Implement Dueling DQN:
1. Create dueling architecture
2. Compare with standard DQN
3. Visualize V(s) and A(s,a)
4. Analyze when it helps

### Exercise 5: LunarLander

Solve LunarLander-v2:
1. Implement full DQN with all tricks
2. Achieve score > 200
3. Visualize trained agent
4. Experiment with variants

---

## Key Equations

### DQN Target
```
y = r + γ max_a' Q(s', a'; θ⁻)
```

### Double DQN Target
```
y = r + γ Q(s', argmax_a Q(s', a; θ); θ⁻)
```

### Dueling Architecture
```
Q(s, a) = V(s; θ_v) + (A(s, a; θ_a) - mean_a'[A(s, a'; θ_a)])
```

### DQN Loss
```
L(θ) = E[(y - Q(s, a; θ))²]
```

### Soft Target Update
```
θ⁻ ← τθ + (1-τ)θ⁻
```

---

## Historical Context

### 2013: Playing Atari with Deep RL
- DeepMind's initial paper
- First successful deep RL
- Played 7 Atari games

### 2015: Human-Level Control (Nature)
- Refined DQN algorithm
- 49 Atari games
- Human-level on 29 games

### Key Innovations
1. **Experience Replay**: Broke correlation
2. **Target Network**: Stabilized training
3. **Frame stacking**: Temporal information
4. **Reward clipping**: Stable across games

---

## Next Week

In Week 9, we will cover **Policy Gradient Methods**:
- Direct policy optimization
- REINFORCE algorithm
- Baseline and variance reduction
- Actor-Critic introduction

Policy gradients learn the policy directly, without Q-values!

---

## Resources

- [Playing Atari with Deep RL (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)
- [Human-Level Control through Deep RL (Nature, 2015)](https://www.nature.com/articles/nature14236)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures for Deep RL](https://arxiv.org/abs/1511.06581)
- [Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298)
- [David Silver Lecture 6: Value Function Approximation](https://www.youtube.com/watch?v=UoPei5o4fps)
