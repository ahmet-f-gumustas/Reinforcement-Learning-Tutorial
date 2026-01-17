# Week 2: Markov Decision Processes (MDP)

This week we will learn the mathematical foundations of Reinforcement Learning: Markov Decision Processes.

## Contents

1. [What is MDP?](#what-is-mdp)
2. [Markov Property](#markov-property)
3. [MDP Components](#mdp-components)
4. [Bellman Equations](#bellman-equations)
5. [Code Examples](#code-examples)
6. [Exercises](#exercises)

---

## What is MDP?

A **Markov Decision Process (MDP)** is a mathematical framework for modeling decision-making problems. Almost all RL problems can be formalized as MDPs.

An MDP is defined by the tuple **(S, A, P, R, γ)**:
- **S**: Set of states
- **A**: Set of actions
- **P**: State transition probability function
- **R**: Reward function
- **γ** (gamma): Discount factor (0 ≤ γ ≤ 1)

---

## Markov Property

The **Markov Property** states that the future depends only on the present state, not on the history of past states.

### Mathematical Definition

```
P(S_{t+1} | S_t) = P(S_{t+1} | S_1, S_2, ..., S_t)
```

In other words: "The future is independent of the past given the present."

### Why is This Important?

1. **Simplifies computation**: We only need to track the current state
2. **Memory efficiency**: No need to store entire history
3. **Mathematical tractability**: Enables use of dynamic programming

### Example: Chess

In chess, the current board position contains all the information needed to make a decision. It doesn't matter how we arrived at this position.

```
State = Current board configuration
```

### Counter-Example: Stock Market

Stock prices might depend on historical trends, not just the current price. This violates the Markov property (though we can sometimes make it Markov by including more information in the state).

---

## MDP Components

### 1. State Space (S)

The set of all possible states in the environment.

```
S = {s_1, s_2, s_3, ..., s_n}
```

**Types:**
- **Discrete**: Finite number of states (e.g., grid world)
- **Continuous**: Infinite states (e.g., robot position)

### 2. Action Space (A)

The set of all possible actions the agent can take.

```
A = {a_1, a_2, ..., a_m}
```

**Types:**
- **Discrete**: Finite actions (e.g., left, right, up, down)
- **Continuous**: Infinite actions (e.g., steering angle)

### 3. Transition Probability (P)

The probability of transitioning from state s to state s' when taking action a.

```
P(s' | s, a) = Probability of reaching s' from s by taking action a
```

**Properties:**
- 0 ≤ P(s' | s, a) ≤ 1
- Σ P(s' | s, a) = 1 for all s, a

### 4. Reward Function (R)

The immediate reward received after transitioning from state s to s' via action a.

```
R(s, a, s') = Immediate reward
```

Or simplified:
```
R(s, a) = Expected reward for taking action a in state s
```

### 5. Discount Factor (γ)

Determines the importance of future rewards.

```
γ = 0: Only care about immediate rewards
γ = 1: Future rewards are as important as immediate rewards
γ = 0.9: Common choice, balances immediate and future
```

**Why discount?**
- Mathematical convenience (ensures convergence)
- Models uncertainty about the future
- Reflects preference for immediate rewards

---

## Return and Value Functions

### Return (G_t)

The total discounted reward from time t onwards:

```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
    = Σ γ^k * R_{t+k+1}  (k = 0 to ∞)
```

### State Value Function V(s)

Expected return starting from state s, following policy π:

```
V_π(s) = E_π[G_t | S_t = s]
       = E_π[R_{t+1} + γV_π(S_{t+1}) | S_t = s]
```

### Action Value Function Q(s, a)

Expected return starting from state s, taking action a, then following policy π:

```
Q_π(s, a) = E_π[G_t | S_t = s, A_t = a]
          = E_π[R_{t+1} + γQ_π(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
```

---

## Bellman Equations

The **Bellman Equations** express the relationship between the value of a state and the values of its successor states.

### Bellman Expectation Equation

For state value function:
```
V_π(s) = Σ_a π(a|s) * Σ_s' P(s'|s,a) * [R(s,a,s') + γV_π(s')]
```

For action value function:
```
Q_π(s,a) = Σ_s' P(s'|s,a) * [R(s,a,s') + γ * Σ_a' π(a'|s') * Q_π(s',a')]
```

### Bellman Optimality Equation

For optimal state value function:
```
V*(s) = max_a Σ_s' P(s'|s,a) * [R(s,a,s') + γV*(s')]
```

For optimal action value function:
```
Q*(s,a) = Σ_s' P(s'|s,a) * [R(s,a,s') + γ * max_a' Q*(s',a')]
```

### Relationship Between V and Q

```
V_π(s) = Σ_a π(a|s) * Q_π(s,a)
Q_π(s,a) = R(s,a) + γ * Σ_s' P(s'|s,a) * V_π(s')
```

---

## Grid World Example

A simple 4x4 grid world MDP:

```
+---+---+---+---+
| S |   |   | G |
+---+---+---+---+
|   | X |   |   |
+---+---+---+---+
|   |   |   |   |
+---+---+---+---+
|   |   |   |   |
+---+---+---+---+

S = Start state
G = Goal state (+1 reward)
X = Obstacle (cannot enter)
  = Normal state (0 reward)
```

**MDP Definition:**
- **S**: 15 valid cells (excluding obstacle)
- **A**: {up, down, left, right}
- **P**: Deterministic (action succeeds with probability 1)
- **R**: +1 for reaching goal, 0 otherwise
- **γ**: 0.9

---

## Code Examples

This folder contains 3 example scripts:

### 1. `01_markov_property.py`
Demonstrates the Markov property with practical examples.

### 2. `02_mdp_components.py`
Implementation of a simple MDP (Grid World) showing all components.

### 3. `03_bellman_equations.py`
Numerical computation of Bellman equations and value functions.

---

## Exercises

### Exercise 1: Design an MDP
Design an MDP for a simple game (e.g., Tic-Tac-Toe):
- Define the state space
- Define the action space
- Define the transition probabilities
- Define the reward function

### Exercise 2: Compute Returns
Given the following rewards and γ = 0.9:
```
R1 = 1, R2 = 2, R3 = 3, R4 = 0 (terminal)
```
Calculate G0, G1, G2, G3.

### Exercise 3: Bellman Equation
For a simple 3-state MDP, manually compute the value function using the Bellman equation.

### Exercise 4: Markov Property Check
Identify which of these scenarios satisfy the Markov property:
1. A chess game
2. A card game where some cards are hidden
3. Weather prediction based only on today's weather
4. Stock price prediction

---

## Next Week

In Week 3, we will cover **Dynamic Programming** methods to solve MDPs:
- Policy Evaluation
- Policy Iteration
- Value Iteration

---

## Resources

- [Sutton & Barto Chapter 3: Finite MDPs](http://incompleteideas.net/book/RLbook2020.pdf)
- [David Silver Lecture 2: MDPs](https://www.youtube.com/watch?v=lfHX2hHRMVQ)
- [Wikipedia: Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process)
