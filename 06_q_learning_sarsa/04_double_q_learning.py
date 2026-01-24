"""
04 - Double Q-Learning

Addressing maximization bias in Q-Learning with Double Q-Learning.

Demonstrates:
- Maximization bias problem
- Double Q-Learning algorithm
- Bias comparison
- When to use Double Q-Learning
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt


class MaximizationBiasMDP:
    """
    Simple MDP to demonstrate maximization bias.

    State A (start): Two actions - left or right
    - Right: Terminal state B with reward 0
    - Left: State C with many actions, each giving N(-0.1, 1.0) reward

    The true optimal action from A is RIGHT (expected reward 0)
    But Q-Learning tends to go LEFT due to maximization bias.
    """

    def __init__(self, n_left_actions: int = 10, gamma: float = 1.0):
        self.gamma = gamma
        self.n_left_actions = n_left_actions

        # States
        self.STATE_A = 0  # Start
        self.STATE_B = 1  # Terminal (right)
        self.STATE_C = 2  # Intermediate (left)
        self.TERMINAL = 3

        # Actions from A
        self.ACTION_LEFT = 0
        self.ACTION_RIGHT = 1

        # Actions from C (many actions, all leading to terminal)
        self.actions_from_C = list(range(n_left_actions))

    def reset(self) -> int:
        return self.STATE_A

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        """Execute action."""
        if state == self.STATE_A:
            if action == self.ACTION_RIGHT:
                return self.TERMINAL, 0.0, True  # Right: reward 0
            else:
                return self.STATE_C, 0.0, False  # Left: go to C

        elif state == self.STATE_C:
            # All actions lead to terminal with N(-0.1, 1.0) reward
            reward = np.random.normal(-0.1, 1.0)
            return self.TERMINAL, reward, True

        return self.TERMINAL, 0.0, True


def q_learning_bias_demo(env: MaximizationBiasMDP, n_episodes: int = 300,
                         alpha: float = 0.1, epsilon: float = 0.1) -> Tuple[Dict, List]:
    """Standard Q-Learning (demonstrates bias)."""
    Q = defaultdict(float)
    left_choices = []  # Track percentage of LEFT choices

    for episode in range(n_episodes):
        state = env.reset()

        while state != env.TERMINAL:
            # Choose action
            if state == env.STATE_A:
                if np.random.random() < epsilon:
                    action = np.random.choice([env.ACTION_LEFT, env.ACTION_RIGHT])
                else:
                    if Q[(env.STATE_A, env.ACTION_LEFT)] > Q[(env.STATE_A, env.ACTION_RIGHT)]:
                        action = env.ACTION_LEFT
                    else:
                        action = env.ACTION_RIGHT

            elif state == env.STATE_C:
                if np.random.random() < epsilon:
                    action = np.random.choice(env.actions_from_C)
                else:
                    q_values = [Q[(env.STATE_C, a)] for a in env.actions_from_C]
                    action = env.actions_from_C[np.argmax(q_values)]

            # Take action
            next_state, reward, done = env.step(state, action)

            # Q-Learning update
            if next_state == env.TERMINAL:
                max_next_q = 0
            elif next_state == env.STATE_C:
                max_next_q = max([Q[(env.STATE_C, a)] for a in env.actions_from_C])
            else:
                max_next_q = 0

            Q[(state, action)] += alpha * (reward + env.gamma * max_next_q - Q[(state, action)])

            state = next_state

        # Record if greedy action is LEFT
        is_left = Q[(env.STATE_A, env.ACTION_LEFT)] > Q[(env.STATE_A, env.ACTION_RIGHT)]
        left_choices.append(is_left)

    return dict(Q), left_choices


def double_q_learning(env: MaximizationBiasMDP, n_episodes: int = 300,
                      alpha: float = 0.1, epsilon: float = 0.1) -> Tuple[Dict, Dict, List]:
    """Double Q-Learning (reduces bias)."""
    Q1 = defaultdict(float)
    Q2 = defaultdict(float)
    left_choices = []

    for episode in range(n_episodes):
        state = env.reset()

        while state != env.TERMINAL:
            # Use sum of Q1 and Q2 for action selection
            if state == env.STATE_A:
                if np.random.random() < epsilon:
                    action = np.random.choice([env.ACTION_LEFT, env.ACTION_RIGHT])
                else:
                    combined_left = Q1[(env.STATE_A, env.ACTION_LEFT)] + Q2[(env.STATE_A, env.ACTION_LEFT)]
                    combined_right = Q1[(env.STATE_A, env.ACTION_RIGHT)] + Q2[(env.STATE_A, env.ACTION_RIGHT)]
                    if combined_left > combined_right:
                        action = env.ACTION_LEFT
                    else:
                        action = env.ACTION_RIGHT

            elif state == env.STATE_C:
                if np.random.random() < epsilon:
                    action = np.random.choice(env.actions_from_C)
                else:
                    q_values = [Q1[(env.STATE_C, a)] + Q2[(env.STATE_C, a)]
                               for a in env.actions_from_C]
                    action = env.actions_from_C[np.argmax(q_values)]

            # Take action
            next_state, reward, done = env.step(state, action)

            # Double Q-Learning update (randomly choose Q1 or Q2 to update)
            if np.random.random() < 0.5:
                # Update Q1 using Q2 for evaluation
                if next_state == env.TERMINAL:
                    target = reward
                elif next_state == env.STATE_C:
                    # Use Q1 to select action, Q2 to evaluate
                    q1_values = [Q1[(env.STATE_C, a)] for a in env.actions_from_C]
                    best_action = env.actions_from_C[np.argmax(q1_values)]
                    target = reward + env.gamma * Q2[(env.STATE_C, best_action)]
                else:
                    target = reward

                Q1[(state, action)] += alpha * (target - Q1[(state, action)])
            else:
                # Update Q2 using Q1 for evaluation
                if next_state == env.TERMINAL:
                    target = reward
                elif next_state == env.STATE_C:
                    q2_values = [Q2[(env.STATE_C, a)] for a in env.actions_from_C]
                    best_action = env.actions_from_C[np.argmax(q2_values)]
                    target = reward + env.gamma * Q1[(env.STATE_C, best_action)]
                else:
                    target = reward

                Q2[(state, action)] += alpha * (target - Q2[(state, action)])

            state = next_state

        # Record if greedy action is LEFT
        combined_left = Q1[(env.STATE_A, env.ACTION_LEFT)] + Q2[(env.STATE_A, env.ACTION_LEFT)]
        combined_right = Q1[(env.STATE_A, env.ACTION_RIGHT)] + Q2[(env.STATE_A, env.ACTION_RIGHT)]
        is_left = combined_left > combined_right
        left_choices.append(is_left)

    return dict(Q1), dict(Q2), left_choices


def compare_bias(n_runs: int = 100):
    """Compare maximization bias between Q-Learning and Double Q-Learning."""
    print("\n" + "="*60)
    print("MAXIMIZATION BIAS COMPARISON")
    print("="*60)

    env = MaximizationBiasMDP(n_left_actions=10)
    n_episodes = 300

    q_learning_left = []
    double_q_left = []

    print(f"\nRunning {n_runs} trials...")
    for run in range(n_runs):
        if run % 20 == 0:
            print(f"  Run {run + 1}/{n_runs}")

        _, q_left = q_learning_bias_demo(env, n_episodes)
        _, _, dq_left = double_q_learning(env, n_episodes)

        q_learning_left.append(q_left)
        double_q_left.append(dq_left)

    # Convert to arrays and calculate percentage
    q_learning_left = np.array(q_learning_left)
    double_q_left = np.array(double_q_left)

    q_left_pct = q_learning_left.mean(axis=0) * 100
    dq_left_pct = double_q_left.mean(axis=0) * 100

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Maximization Bias: Q-Learning vs Double Q-Learning', fontsize=16)

    # Plot 1: Percentage of LEFT choices over time
    ax = axes[0]
    ax.plot(q_left_pct, label='Q-Learning', linewidth=2)
    ax.plot(dq_left_pct, label='Double Q-Learning', linewidth=2)
    ax.axhline(y=5, color='r', linestyle='--', linewidth=2, label='Optimal (≈5%)')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('% LEFT Actions', fontsize=12)
    ax.set_title('Percentage of LEFT Actions Over Time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Plot 2: Final percentage
    ax = axes[1]
    final_q = q_left_pct[-50:].mean()
    final_dq = dq_left_pct[-50:].mean()
    optimal = 5  # Roughly optimal with ε=0.1

    bars = ax.bar(['Q-Learning', 'Double Q', 'Optimal'], [final_q, final_dq, optimal],
                  color=['coral', 'steelblue', 'green'])
    ax.set_ylabel('% LEFT Actions (last 50 episodes)', fontsize=12)
    ax.set_title('Final LEFT Action Percentage', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, [final_q, final_dq, optimal]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/06_td_control/maximization_bias.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved bias comparison to maximization_bias.png")
    plt.close()

    print(f"\nFinal % of LEFT choices (last 50 episodes):")
    print(f"  Q-Learning:        {final_q:.1f}%")
    print(f"  Double Q-Learning: {final_dq:.1f}%")
    print(f"  Optimal:           ~{optimal}% (due to exploration)")

    print(f"\nQ-Learning chooses LEFT {final_q/final_dq:.1f}x more than Double Q-Learning!")
    print("This demonstrates the maximization bias in standard Q-Learning.")


def explain_maximization_bias():
    """Explain why maximization bias occurs."""
    print("\n" + "="*60)
    print("WHY MAXIMIZATION BIAS OCCURS")
    print("="*60)

    print("""
The Problem:
------------
In state C, all actions give reward ~ N(-0.1, 1.0)
True expected value of each action: -0.1
True value of going LEFT from A: -0.1

But Q-Learning estimates:
Q(C, a_i) ≈ -0.1 + noise for each action

When we compute max_a Q(C, a):
- If we have 10 actions, we're taking max of 10 noisy estimates
- max of N(0, σ²) random variables is positive!
- So max Q overestimates the true max

Result:
-------
Q(A, LEFT) uses max Q(C, a) which is biased upward
Q(A, LEFT) > Q(A, RIGHT) even though RIGHT is better!

The Fix (Double Q-Learning):
---------------------------
Use Q1 to select action: a* = argmax Q1(s, a)
Use Q2 to evaluate: Q2(s, a*)

Since Q1 and Q2 are trained on different samples:
- Selection and evaluation are decoupled
- Eliminates the bias!
    """)


def visualize_q_values(env: MaximizationBiasMDP):
    """Visualize learned Q-values."""
    print("\n" + "="*60)
    print("Q-VALUE VISUALIZATION")
    print("="*60)

    n_episodes = 500

    print("\nTraining Q-Learning...")
    Q, _ = q_learning_bias_demo(env, n_episodes)

    print("Training Double Q-Learning...")
    Q1, Q2, _ = double_q_learning(env, n_episodes)

    # Plot Q-values for actions from state C
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Q-Learning Q-values for C
    ax = axes[0]
    q_values_c = [Q.get((env.STATE_C, a), 0) for a in env.actions_from_C]
    ax.bar(range(len(env.actions_from_C)), q_values_c, color='coral')
    ax.axhline(y=-0.1, color='g', linestyle='--', linewidth=2, label='True value (-0.1)')
    ax.axhline(y=max(q_values_c), color='r', linestyle=':', linewidth=2, label=f'Max Q = {max(q_values_c):.2f}')
    ax.set_xlabel('Action', fontsize=12)
    ax.set_ylabel('Q-value', fontsize=12)
    ax.set_title('Q-Learning: Q(C, a) values', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Double Q-Learning Q-values (averaged)
    ax = axes[1]
    q_values_c_double = [(Q1.get((env.STATE_C, a), 0) + Q2.get((env.STATE_C, a), 0)) / 2
                         for a in env.actions_from_C]
    ax.bar(range(len(env.actions_from_C)), q_values_c_double, color='steelblue')
    ax.axhline(y=-0.1, color='g', linestyle='--', linewidth=2, label='True value (-0.1)')
    ax.axhline(y=max(q_values_c_double), color='r', linestyle=':', linewidth=2,
               label=f'Max Q = {max(q_values_c_double):.2f}')
    ax.set_xlabel('Action', fontsize=12)
    ax.set_ylabel('Q-value (avg of Q1, Q2)', fontsize=12)
    ax.set_title('Double Q-Learning: Q(C, a) values', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/06_td_control/q_value_comparison.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved Q-value comparison to q_value_comparison.png")
    plt.close()

    # Print Q-values for state A
    print("\nQ-values for state A (start):")
    print(f"  Q-Learning:")
    print(f"    Q(A, LEFT):  {Q.get((env.STATE_A, env.ACTION_LEFT), 0):.3f}")
    print(f"    Q(A, RIGHT): {Q.get((env.STATE_A, env.ACTION_RIGHT), 0):.3f}")
    print(f"  Double Q-Learning (Q1 + Q2):")
    left_sum = Q1.get((env.STATE_A, env.ACTION_LEFT), 0) + Q2.get((env.STATE_A, env.ACTION_LEFT), 0)
    right_sum = Q1.get((env.STATE_A, env.ACTION_RIGHT), 0) + Q2.get((env.STATE_A, env.ACTION_RIGHT), 0)
    print(f"    Q(A, LEFT):  {left_sum:.3f}")
    print(f"    Q(A, RIGHT): {right_sum:.3f}")
    print(f"\nTrue values: Q(A, LEFT) = -0.1, Q(A, RIGHT) = 0")


def main():
    """Main demonstration."""
    print("="*60)
    print("DOUBLE Q-LEARNING: ADDRESSING MAXIMIZATION BIAS")
    print("="*60)

    # Create environment
    env = MaximizationBiasMDP(n_left_actions=10)

    print("\nMDP Description:")
    print("  State A: Start state")
    print("  Action RIGHT: Terminal, reward 0 (OPTIMAL)")
    print("  Action LEFT: Go to state C")
    print("  State C: 10 actions, each gives reward ~ N(-0.1, 1.0)")

    # Explain the bias
    explain_maximization_bias()

    # Compare algorithms
    compare_bias(n_runs=100)

    # Visualize Q-values
    visualize_q_values(env)

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Points:")
    print("  - Q-Learning has maximization bias: max of noisy estimates ≠ max of true values")
    print("  - This causes Q-Learning to overestimate action values")
    print("  - Double Q-Learning fixes this by decoupling selection and evaluation")
    print("  - Use Double Q-Learning when actions have high variance rewards")


if __name__ == "__main__":
    main()
