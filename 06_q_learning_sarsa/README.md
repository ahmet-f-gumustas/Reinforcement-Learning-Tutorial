# Week 6: TD Control - SARSA and Q-Learning

This week we learn TD Control methods - using temporal difference learning to find optimal policies, not just evaluate them.

## Contents

1. [Introduction to TD Control](#introduction-to-td-control)
2. [SARSA: On-Policy TD Control](#sarsa-on-policy-td-control)
3. [Q-Learning: Off-Policy TD Control](#q-learning-off-policy-td-control)
4. [Expected SARSA](#expected-sarsa)
5. [Maximization Bias and Double Q-Learning](#maximization-bias-and-double-q-learning)
6. [Code Examples](#code-examples)
7. [Exercises](#exercises)

---

## Introduction to TD Control

In Week 5, we learned TD methods for **prediction** (evaluating a given policy). Now we use TD for **control** (finding the optimal policy).

### Control vs Prediction

| Aspect | Prediction | Control |
|--------|------------|---------|
| Goal | Estimate V^π or Q^π | Find π* |
| Updates | V(s) or Q(s,a) for fixed π | Q(s,a) while improving π |
| Output | Value function | Optimal policy |

### Why Q-values?

For control, we need Q-values instead of V-values:

**With V-values** (need model):
```
π(s) = argmax_a Σ P(s'|s,a)[R + γV(s')]
```

**With Q-values** (model-free):
```
π(s) = argmax_a Q(s,a)
```

Q-values tell us which action is best directly!

### The General TD Control Pattern

```
Initialize Q(s,a) for all s, a
Repeat (for each episode):
    Initialize S
    Choose A from S using policy derived from Q (e.g., ε-greedy)
    Repeat (for each step):
        Take action A, observe R, S'
        Choose A' from S' using policy derived from Q
        Update Q(S,A) using TD update
        S ← S'; A ← A'
    until S is terminal
```

---

## SARSA: On-Policy TD Control

**SARSA** (State-Action-Reward-State-Action) is an on-policy TD control method.

### The Name

SARSA uses the quintuple (S, A, R, S', A'):
- **S**: Current state
- **A**: Current action
- **R**: Reward received
- **S'**: Next state
- **A'**: Next action (actually taken!)

### SARSA Update Rule

```
Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
```

Key insight: Uses Q(S', A') where A' is the action **actually taken** under the current policy.

### SARSA Algorithm

```
Initialize Q(s,a) arbitrarily (Q(terminal,·) = 0)
Repeat (for each episode):
    Initialize S
    Choose A from S using ε-greedy from Q
    Repeat (for each step):
        Take action A, observe R, S'
        Choose A' from S' using ε-greedy from Q
        Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
        S ← S'; A ← A'
    until S is terminal
```

### Properties of SARSA

1. **On-Policy**: Learns about the policy being followed
2. **Conservative**: Accounts for exploration in learned values
3. **Safe**: Values reflect the actual behavior (including mistakes)

### When to Use SARSA

- When you care about the policy actually being executed
- When exploration is dangerous (cliff walking, robots)
- When the evaluation and target policies must match

---

## Q-Learning: Off-Policy TD Control

**Q-Learning** is the most famous TD control algorithm, developed by Watkins (1989).

### Q-Learning Update Rule

```
Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
```

Key difference from SARSA: Uses `max_a Q(S',a)` regardless of what action is actually taken next.

### Q-Learning Algorithm

```
Initialize Q(s,a) arbitrarily (Q(terminal,·) = 0)
Repeat (for each episode):
    Initialize S
    Repeat (for each step):
        Choose A from S using ε-greedy from Q
        Take action A, observe R, S'
        Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
        S ← S'
    until S is terminal
```

### Why is Q-Learning Off-Policy?

Q-Learning learns about the **greedy policy** (max_a Q) while following an **exploratory policy** (ε-greedy).

- **Behavior policy** (what we do): ε-greedy
- **Target policy** (what we learn): greedy

### Properties of Q-Learning

1. **Off-Policy**: Learns optimal policy regardless of behavior
2. **Optimistic**: Values reflect the best possible future
3. **Aggressive**: Assumes optimal actions will be taken

### Q-Learning vs SARSA Comparison

| Aspect | SARSA | Q-Learning |
|--------|-------|------------|
| Policy Type | On-policy | Off-policy |
| Update Target | Q(S', A') | max_a Q(S', a) |
| Learns | Policy being followed | Optimal policy |
| Risk Sensitivity | Risk-aware | Risk-neutral |
| Convergence | To Q^π | To Q* |

---

## Expected SARSA

**Expected SARSA** takes the expectation over all possible next actions.

### Update Rule

```
Q(S,A) ← Q(S,A) + α[R + γ Σ_a π(a|S')Q(S',a) - Q(S,A)]
```

Instead of sampling A', we use the expected value under the policy.

### Why Expected SARSA?

1. **Lower variance** than SARSA (no sampling of A')
2. **Generalizes** both SARSA and Q-Learning
3. **More stable** learning

### Special Cases

- **π = ε-greedy**: Expected SARSA
- **π = greedy**: Equivalent to Q-Learning!

### Expected SARSA Algorithm

```
Initialize Q(s,a) arbitrarily
Repeat (for each episode):
    Initialize S
    Repeat (for each step):
        Choose A from S using ε-greedy
        Take action A, observe R, S'

        # Calculate expected value
        expected_Q = Σ_a π(a|S') * Q(S',a)

        Q(S,A) ← Q(S,A) + α[R + γ * expected_Q - Q(S,A)]
        S ← S'
    until S is terminal
```

---

## Maximization Bias and Double Q-Learning

### The Problem: Maximization Bias

Q-Learning uses `max_a Q(S',a)` which introduces a **positive bias**.

Why? Because max of noisy estimates ≥ true max on average.

**Example**: Two actions, both with true value 0:
- Q estimates: Q(a1) = +0.1, Q(a2) = -0.1 (due to noise)
- max Q = +0.1 (biased upward!)
- True max = 0

### Demonstration of Bias

Consider a simple MDP where all actions lead to states with Q = 0 ± noise.
Q-Learning will overestimate the value because it always picks the max.

### Double Q-Learning Solution

Maintain **two independent Q-functions**: Q1 and Q2

**Update rule** (with 50% probability each):

Update Q1:
```
Q1(S,A) ← Q1(S,A) + α[R + γQ2(S', argmax_a Q1(S',a)) - Q1(S,A)]
```

Update Q2:
```
Q2(S,A) ← Q2(S,A) + α[R + γQ1(S', argmax_a Q2(S',a)) - Q2(S,A)]
```

### Why This Works

- Use Q1 to select action (argmax)
- Use Q2 to evaluate that action's value
- Decouples selection from evaluation
- Eliminates maximization bias

### Double Q-Learning Algorithm

```
Initialize Q1(s,a), Q2(s,a) arbitrarily
Repeat (for each episode):
    Initialize S
    Repeat (for each step):
        Choose A from S using ε-greedy from Q1 + Q2
        Take action A, observe R, S'

        With 0.5 probability:
            A* = argmax_a Q1(S',a)
            Q1(S,A) ← Q1(S,A) + α[R + γQ2(S',A*) - Q1(S,A)]
        else:
            A* = argmax_a Q2(S',a)
            Q2(S,A) ← Q2(S,A) + α[R + γQ1(S',A*) - Q2(S,A)]

        S ← S'
    until S is terminal
```

---

## Cliff Walking Example

The **Cliff Walking** problem perfectly illustrates the SARSA vs Q-Learning difference.

### The Environment

```
Start: S          Goal: G
[S][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][G]
[C][C][C][C][C][C][C][C][C][C][C][C]  ← Cliff (C)
```

- Actions: Up, Down, Left, Right
- Cliff: -100 reward and back to start
- Goal: +10 reward
- Step: -1 reward

### SARSA vs Q-Learning Behavior

**Q-Learning** learns the **optimal path** (along the cliff edge):
- Higher expected return
- But risky with ε-greedy (might fall off!)

**SARSA** learns the **safe path** (away from cliff):
- Lower expected return
- But accounts for exploration (won't fall off)

### Which is Better?

- **During learning** (with ε-greedy): SARSA gets more reward
- **After learning** (greedy only): Q-Learning's policy is optimal

---

## On-Policy vs Off-Policy Summary

### On-Policy (SARSA)

```
Target = Q(S', A') where A' ~ π(·|S')
```

- Learns about policy being executed
- "What reward will I get doing what I'm doing?"
- Values include exploration cost

### Off-Policy (Q-Learning)

```
Target = max_a Q(S', a)
```

- Learns about optimal policy
- "What reward could I get if I acted optimally?"
- Values ignore exploration

---

## Code Examples

This folder contains 7 example scripts:

### 1. `01_sarsa.py`
On-policy TD control with SARSA.
- SARSA algorithm implementation
- ε-greedy policy
- GridWorld learning
- Policy visualization

### 2. `02_q_learning.py`
Off-policy TD control with Q-Learning.
- Q-Learning algorithm
- Comparison with SARSA
- Optimal policy discovery
- Convergence analysis

### 3. `03_expected_sarsa.py`
Expected SARSA implementation.
- Expected value calculation
- Variance comparison with SARSA
- Connection to Q-Learning
- Stability analysis

### 4. `04_double_q_learning.py`
Addressing maximization bias.
- Maximization bias demonstration
- Double Q-Learning implementation
- Bias comparison plots
- MDP with stochastic rewards

### 5. `05_cliff_walking.py`
Classic cliff walking comparison.
- Gymnasium CliffWalking environment
- SARSA vs Q-Learning
- Safe vs optimal path
- Learning curves

### 6. `06_taxi_problem.py`
Taxi-v3 environment from Gymnasium.
- Discrete state/action space
- Hierarchical task
- Algorithm comparison
- Success rate tracking

### 7. `07_algorithm_comparison.py`
Comprehensive algorithm comparison.
- SARSA, Q-Learning, Expected SARSA, Double Q
- Multiple environments
- Performance metrics
- Visualization dashboard

---

## Exercises

### Exercise 1: Implement SARSA

Implement SARSA from scratch:
1. Create a simple GridWorld
2. Implement ε-greedy policy
3. Run SARSA and visualize learned Q-values
4. Compare different α and ε values

### Exercise 2: Q-Learning Exploration

Experiment with Q-Learning exploration:
1. Try different ε decay schedules
2. Compare constant vs decaying ε
3. What happens with ε = 0 from start?
4. Plot learning curves

### Exercise 3: Cliff Walking Analysis

On the Cliff Walking environment:
1. Implement both SARSA and Q-Learning
2. Run 500 episodes each
3. Plot reward per episode
4. Visualize learned paths
5. Explain the difference

### Exercise 4: Maximization Bias

Create an MDP to demonstrate maximization bias:
1. Start state with two actions (left, right)
2. Right: terminal with reward 0
3. Left: state with many actions, all reward N(−0.1, 1)
4. Run Q-Learning and Double Q-Learning
5. Plot percentage of "left" selections

### Exercise 5: Taxi Driver

Solve Taxi-v3 with all algorithms:
1. Implement SARSA, Q-Learning, Expected SARSA
2. Track episodes to solve (200+ reward)
3. Compare sample efficiency
4. Which learns fastest?

---

## Key Equations

### SARSA Update
```
Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
```

### Q-Learning Update
```
Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
```

### Expected SARSA Update
```
Q(S,A) ← Q(S,A) + α[R + γ Σ_a π(a|S')Q(S',a) - Q(S,A)]
```

### Double Q-Learning Update
```
Q1(S,A) ← Q1(S,A) + α[R + γQ2(S', argmax_a Q1(S',a)) - Q1(S,A)]
```

### ε-Greedy Policy
```
π(a|s) = 1 - ε + ε/|A|  if a = argmax Q(s,·)
       = ε/|A|          otherwise
```

---

## Next Week

In Week 7, we will cover **Function Approximation**:
- Linear function approximation
- Feature engineering
- Semi-gradient methods
- Deadly triad

Moving from tabular methods to handling large/continuous state spaces!

---

## Resources

- [Sutton & Barto Chapter 6: TD Control](http://incompleteideas.net/book/RLbook2020.pdf)
- [Watkins (1989): Q-Learning Thesis](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf)
- [David Silver Lecture 5: Model-Free Control](https://www.youtube.com/watch?v=0g4j2k_Ggc4)
- [Double Q-Learning Paper (2010)](https://papers.nips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf)
