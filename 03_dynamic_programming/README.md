# Week 3: Dynamic Programming

This week we will learn Dynamic Programming (DP) methods to solve Markov Decision Processes.

## Contents

1. [What is Dynamic Programming?](#what-is-dynamic-programming)
2. [Policy Evaluation](#policy-evaluation)
3. [Policy Improvement](#policy-improvement)
4. [Policy Iteration](#policy-iteration)
5. [Value Iteration](#value-iteration)
6. [Comparison of Methods](#comparison-of-methods)
7. [Code Examples](#code-examples)
8. [Exercises](#exercises)

---

## What is Dynamic Programming?

**Dynamic Programming (DP)** is a collection of algorithms that can be used to compute optimal policies given a perfect model of the environment as an MDP.

### Key Requirements for DP

1. **Perfect Model**: We need complete knowledge of:
   - State transition probabilities P(s'|s,a)
   - Reward function R(s,a,s')

2. **Finite MDP**: The state and action spaces must be finite

### Why "Dynamic Programming"?

The term comes from Richard Bellman (1950s):
- **Dynamic**: Problems involve sequential decisions over time
- **Programming**: Refers to optimization (not computer programming)

### Core Idea

DP uses the **Bellman equations** to iteratively compute value functions:

```
V(s) = E[R + gamma * V(s')]
```

By solving these equations, we can find optimal policies.

---

## Policy Evaluation

**Policy Evaluation** (also called **Prediction**) computes the state-value function V_pi for a given policy pi.

### The Problem

Given a policy pi, compute V_pi(s) for all states s.

### Bellman Expectation Equation

```
V_pi(s) = sum_a pi(a|s) * sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V_pi(s')]
```

### Iterative Policy Evaluation Algorithm

```
Initialize V(s) = 0 for all s
Repeat until convergence:
    For each state s:
        v = V(s)
        V(s) = sum_a pi(a|s) * sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
    delta = max(delta, |v - V(s)|)
Until delta < theta (small threshold)
```

### Convergence

- Guaranteed to converge to V_pi
- Convergence rate depends on gamma
- Typically fast for small state spaces

### Example: Grid World

For a random policy (equal probability for all actions):

```
Iteration 0:  All values = 0
Iteration 1:  Values near goal increase
Iteration 10: Values propagate through grid
Iteration 50: Convergence reached
```

---

## Policy Improvement

**Policy Improvement** creates a better policy given a value function.

### The Idea

If we know V_pi(s), we can improve the policy by acting **greedily**:

```
pi'(s) = argmax_a sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V_pi(s')]
```

### Policy Improvement Theorem

For any pair of deterministic policies pi and pi':

```
If Q_pi(s, pi'(s)) >= V_pi(s) for all s
Then V_pi'(s) >= V_pi(s) for all s
```

In other words: Acting greedily with respect to V_pi gives a policy at least as good as pi.

### Greedy Policy

The greedy policy always selects the action with highest expected value:

```python
def greedy_policy(V, mdp):
    policy = {}
    for s in mdp.states:
        action_values = []
        for a in mdp.actions:
            value = sum(p * (r + gamma * V[s'])
                       for p, s', r in mdp.transitions(s, a))
            action_values.append(value)
        policy[s] = argmax(action_values)
    return policy
```

---

## Policy Iteration

**Policy Iteration** alternates between policy evaluation and policy improvement until convergence.

### Algorithm

```
1. Initialize policy pi arbitrarily

2. Policy Evaluation:
   Compute V_pi using iterative policy evaluation

3. Policy Improvement:
   For each state s:
       pi(s) = argmax_a sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V_pi(s')]

4. If policy changed, go to step 2
   Else, return pi (optimal policy)
```

### Visualization

```
    +------------------+
    |                  |
    v                  |
  [Policy]             |
    |                  |
    | Evaluation       |
    v                  |
  [Value Function]     |
    |                  |
    | Improvement      |
    v                  |
  [Better Policy] -----+ (if changed)
    |
    | (if unchanged)
    v
  [Optimal Policy]
```

### Properties

- **Convergence**: Guaranteed in finite number of iterations
- **Iterations**: Usually converges very quickly (often < 10 iterations)
- **Complexity**: Each iteration requires O(|S|^2 * |A|) operations

### Example Trace

```
Iteration 1: Evaluate random policy -> Improve -> Better policy
Iteration 2: Evaluate new policy -> Improve -> Even better policy
Iteration 3: Evaluate -> Improve -> Policy unchanged -> OPTIMAL!
```

---

## Value Iteration

**Value Iteration** combines policy evaluation and improvement into a single update.

### Key Insight

We don't need to wait for policy evaluation to converge. One sweep of updates is enough!

### Bellman Optimality Equation

```
V*(s) = max_a sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V*(s')]
```

### Algorithm

```
Initialize V(s) = 0 for all s
Repeat until convergence:
    For each state s:
        V(s) = max_a sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
Until max change < theta

Extract policy:
For each state s:
    pi(s) = argmax_a sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
```

### Comparison with Policy Iteration

| Aspect | Policy Iteration | Value Iteration |
|--------|-----------------|-----------------|
| Updates | Full evaluation + improvement | Single max update |
| Convergence | Fewer outer iterations | More iterations |
| Per iteration | More expensive | Cheaper |
| Best for | Small state spaces | Larger state spaces |

### Convergence

- Converges to V* as iterations -> infinity
- Rate: ||V_{k+1} - V*|| <= gamma * ||V_k - V*||
- Faster when gamma is smaller

---

## Comparison of Methods

### Summary Table

| Method | Computes | Uses |
|--------|----------|------|
| Policy Evaluation | V_pi | Given policy pi |
| Policy Improvement | pi' | Given V_pi |
| Policy Iteration | V*, pi* | Alternates Eval/Improve |
| Value Iteration | V*, pi* | Direct optimization |

### When to Use What?

**Policy Iteration:**
- When you need the value function of a specific policy
- When state space is small
- When you want guaranteed convergence in few iterations

**Value Iteration:**
- When you only need the optimal policy
- When state space is moderate
- When simplicity is preferred

### Limitations of DP

1. **Requires complete model**: Must know P and R
2. **Curse of dimensionality**: Complexity grows exponentially with state dimensions
3. **Synchronous updates**: Must update all states (can be improved with async DP)

---

## Generalized Policy Iteration (GPI)

**GPI** is the general idea of interleaving policy evaluation and improvement.

```
      Evaluation
    /           \
   v             v
Policy  <--->  Value
   ^             ^
    \           /
     Improvement
```

### Key Insight

- Evaluation and improvement are competing processes
- Evaluation makes V consistent with pi
- Improvement makes pi greedy with respect to V
- They eventually converge to optimality

### Variations

1. **Full Policy Evaluation**: Complete evaluation before improvement
2. **Partial Policy Evaluation**: Few evaluation steps before improvement
3. **Value Iteration**: Single evaluation step (one sweep)

---

## Asynchronous Dynamic Programming

Standard DP updates all states in each iteration. **Async DP** allows more flexible updates.

### Methods

1. **In-place DP**: Update values immediately (no separate arrays)
2. **Prioritized Sweeping**: Update states with largest expected change
3. **Real-time DP**: Update states visited by agent

### Benefits

- Can be much faster in practice
- Focuses computation on important states
- Still guaranteed to converge (with proper conditions)

---

## Code Examples

This folder contains 3 example scripts:

### 1. `01_policy_evaluation.py`
Demonstrates iterative policy evaluation on a Grid World.
- Evaluates different policies (random, always-right, etc.)
- Shows convergence of value function
- Visualizes value function as heatmap

### 2. `02_policy_iteration.py`
Complete policy iteration implementation.
- Shows evaluation-improvement cycle
- Tracks policy changes across iterations
- Finds optimal policy for Grid World

### 3. `03_value_iteration.py`
Value iteration with detailed visualization.
- Single update rule combining evaluation and improvement
- Compares convergence with policy iteration
- Extracts optimal policy from value function

---

## Exercises

### Exercise 1: Manual Policy Evaluation

For the following 3-state MDP with gamma = 0.9:

```
States: S0, S1, S2 (S2 is terminal)
Actions: Left, Right
Transitions: Deterministic
Rewards: +1 for reaching S2, 0 otherwise
```

Compute V_pi for the policy "always go Right" by hand.

### Exercise 2: Policy Improvement

Given V = [2.0, 1.0, 0.0] for the above MDP:
1. Compute Q(s, a) for all state-action pairs
2. Determine the greedy policy
3. Is this policy optimal?

### Exercise 3: Implement Async DP

Modify the policy evaluation code to use in-place updates.
Compare convergence speed with synchronous updates.

### Exercise 4: Larger Grid World

Create a 10x10 Grid World with:
- Multiple obstacles
- Multiple goals with different rewards
- Run policy iteration and value iteration
- Compare number of iterations and computation time

### Exercise 5: Stochastic Transitions

Modify the Grid World to have stochastic transitions:
- 80% intended direction
- 10% left of intended
- 10% right of intended

How does this affect the optimal policy?

---

## Pseudocode Summary

### Policy Evaluation

```python
def policy_evaluation(pi, mdp, theta=1e-6):
    V = {s: 0 for s in mdp.states}
    while True:
        delta = 0
        for s in mdp.states:
            v = V[s]
            V[s] = sum(pi[s][a] * sum(p * (r + gamma * V[s_])
                       for p, s_, r in mdp.P[s][a])
                       for a in mdp.actions)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V
```

### Policy Iteration

```python
def policy_iteration(mdp):
    pi = random_policy(mdp)
    while True:
        V = policy_evaluation(pi, mdp)
        pi_new = greedy_policy(V, mdp)
        if pi_new == pi:
            break
        pi = pi_new
    return pi, V
```

### Value Iteration

```python
def value_iteration(mdp, theta=1e-6):
    V = {s: 0 for s in mdp.states}
    while True:
        delta = 0
        for s in mdp.states:
            v = V[s]
            V[s] = max(sum(p * (r + gamma * V[s_])
                       for p, s_, r in mdp.P[s][a])
                       for a in mdp.actions)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    pi = greedy_policy(V, mdp)
    return pi, V
```

---

## Next Week

In Week 4, we will cover **Monte Carlo Methods**:
- Learning without a model
- Episodic learning
- First-visit vs Every-visit MC
- MC Control

---

## Resources

- [Sutton & Barto Chapter 4: Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf)
- [David Silver Lecture 3: Planning by Dynamic Programming](https://www.youtube.com/watch?v=Nd1-UUMVfz4)
- [Stanford CS234: Dynamic Programming](https://web.stanford.edu/class/cs234/)
