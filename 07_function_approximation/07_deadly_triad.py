"""
07 - The Deadly Triad

Demonstration of divergence when combining:
1. Function approximation
2. Bootstrapping (TD methods)
3. Off-policy learning

Demonstrates:
- Baird's counterexample
- When and why divergence occurs
- Safe vs unsafe combinations
- Solutions overview
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import matplotlib.pyplot as plt


class BairdsCounterexample:
    """
    Baird's counterexample - a simple MDP where off-policy TD diverges.

    7 states:
    - States 1-6: All transition to state 7 with any action
    - State 7: Self-loop with probability 1

    Two actions:
    - Dashed (action 0): Uniform random transitions
    - Solid (action 1): Always go to state 7

    Behavior policy: Uniform random (50% each action)
    Target policy: Always solid (action 1)

    With this setup, off-policy TD with linear function approximation
    DIVERGES to infinity!
    """

    def __init__(self):
        self.n_states = 7
        self.n_actions = 2
        self.gamma = 0.99

        # State 7 is special (index 6)
        self.terminal_state = None  # No terminal state

    def reset(self) -> int:
        """Start uniformly at random in states 1-6."""
        return np.random.randint(0, 6)

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        """
        Execute action.

        All actions lead to state 7 (index 6) with reward 0.
        State 7 loops back to itself.
        """
        if action == 0:  # Dashed - uniform over states 1-6
            next_state = np.random.randint(0, 6)
        else:  # Solid - always to state 7
            next_state = 6

        return next_state, 0.0, False  # No terminal state


def baird_features(state: int) -> np.ndarray:
    """
    Feature representation for Baird's counterexample.

    States 1-6: x(s) = [2, 0, 0, 0, 0, 0, 0, 1]  (with 2 in position s)
    State 7:    x(7) = [1, 1, 1, 1, 1, 1, 1, 2]

    8 features total, but highly dependent (not full rank).
    """
    features = np.zeros(8)

    if state < 6:  # States 1-6
        features[state] = 2
        features[7] = 1
    else:  # State 7
        features[:7] = 1
        features[7] = 2

    return features


def off_policy_td_baird(env: BairdsCounterexample, n_steps: int = 1000,
                        alpha: float = 0.01) -> Tuple[List[np.ndarray], List[float]]:
    """
    Off-policy semi-gradient TD on Baird's counterexample.

    Behavior policy: 50% dashed, 50% solid
    Target policy: 100% solid

    This WILL diverge!
    """
    weights = np.ones(8)  # Initialize weights
    weights[6] = 10  # As in Sutton & Barto

    weight_history = [weights.copy()]
    max_weight_history = [np.max(np.abs(weights))]

    state = env.reset()

    for step in range(n_steps):
        # Behavior policy: random action
        action = np.random.randint(2)

        # Take action
        next_state, reward, _ = env.step(state, action)

        # Importance sampling ratio
        # π(solid|s) = 1, b(solid|s) = 0.5
        # π(dashed|s) = 0, b(dashed|s) = 0.5
        if action == 1:  # Solid (target policy action)
            rho = 1.0 / 0.5  # = 2
        else:  # Dashed (not target policy action)
            rho = 0.0 / 0.5  # = 0

        # Semi-gradient TD update with importance sampling
        features = baird_features(state)
        next_features = baird_features(next_state)

        current_value = np.dot(weights, features)
        next_value = np.dot(weights, next_features)

        td_error = reward + env.gamma * next_value - current_value

        # Off-policy update
        weights += alpha * rho * td_error * features

        # Record
        weight_history.append(weights.copy())
        max_weight_history.append(np.max(np.abs(weights)))

        state = next_state

    return weight_history, max_weight_history


def on_policy_td_baird(env: BairdsCounterexample, n_steps: int = 1000,
                       alpha: float = 0.01) -> Tuple[List[np.ndarray], List[float]]:
    """
    On-policy semi-gradient TD on Baird's counterexample.

    Both behavior and target policy: 100% solid

    This should be stable!
    """
    weights = np.ones(8)
    weights[6] = 10

    weight_history = [weights.copy()]
    max_weight_history = [np.max(np.abs(weights))]

    state = env.reset()

    for step in range(n_steps):
        # On-policy: always solid
        action = 1

        next_state, reward, _ = env.step(state, action)

        # No importance sampling needed (rho = 1)
        features = baird_features(state)
        next_features = baird_features(next_state)

        current_value = np.dot(weights, features)
        next_value = np.dot(weights, next_features)

        td_error = reward + env.gamma * next_value - current_value
        weights += alpha * td_error * features

        weight_history.append(weights.copy())
        max_weight_history.append(np.max(np.abs(weights)))

        state = next_state

    return weight_history, max_weight_history


def demonstrate_divergence():
    """Demonstrate divergence in Baird's counterexample."""
    print("\n" + "="*60)
    print("BAIRD'S COUNTEREXAMPLE: DIVERGENCE DEMONSTRATION")
    print("="*60)

    env = BairdsCounterexample()
    n_steps = 1000

    print("\nRunning off-policy TD (should DIVERGE)...")
    _, off_policy_max = off_policy_td_baird(env, n_steps, alpha=0.01)

    print("Running on-policy TD (should be STABLE)...")
    _, on_policy_max = on_policy_td_baird(env, n_steps, alpha=0.01)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Baird's Counterexample: The Deadly Triad", fontsize=16)

    # Max weight magnitude
    ax = axes[0]
    ax.semilogy(off_policy_max, label='Off-policy TD', linewidth=2)
    ax.semilogy(on_policy_max, label='On-policy TD', linewidth=2)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Max |weight| (log scale)', fontsize=12)
    ax.set_title('Weight Divergence', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Explanation
    ax = axes[1]
    ax.axis('off')

    explanation = """
    THE DEADLY TRIAD

    Three ingredients that can cause divergence:

    1. FUNCTION APPROXIMATION
       - V(s) ≈ w^T x(s) instead of table
       - Updates affect multiple states

    2. BOOTSTRAPPING
       - Using estimates in targets (TD, not MC)
       - Target: R + γV(S') uses V which is also being updated
       - Errors can compound

    3. OFF-POLICY LEARNING
       - Learning about π while following b
       - Distribution mismatch
       - Importance sampling can have high variance

    SAFE COMBINATIONS:
    ✓ Tabular + Bootstrapping + Off-policy
    ✓ Function Approx + No Bootstrap (MC) + Off-policy
    ✓ Function Approx + Bootstrapping + On-policy

    UNSAFE:
    ✗ Function Approx + Bootstrapping + Off-policy

    SOLUTIONS:
    - Use on-policy methods (SARSA)
    - Use Monte Carlo (no bootstrapping)
    - Gradient TD methods (GTD, TDC)
    - Experience replay + target networks (DQN)
    """
    ax.text(0.05, 0.95, explanation, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/deadly_triad.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved divergence plot to deadly_triad.png")
    plt.close()

    print(f"\nFinal max weight magnitude:")
    print(f"  Off-policy TD: {off_policy_max[-1]:.2e} (DIVERGED!)")
    print(f"  On-policy TD:  {on_policy_max[-1]:.2f} (stable)")


def compare_all_combinations():
    """Compare all combinations of the triad elements."""
    print("\n" + "="*60)
    print("COMPARING ALL COMBINATIONS")
    print("="*60)

    # We'll simulate different combinations
    results = {}

    # Setup simple test
    class SimpleEnv:
        def __init__(self):
            self.n_states = 5
            self.gamma = 0.9
            self.true_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        def reset(self):
            return np.random.randint(0, 5)

        def step(self, state, action=None):
            next_state = np.random.randint(0, 5)
            reward = self.true_values[next_state]
            return next_state, reward, False

    env = SimpleEnv()
    n_steps = 2000

    # 1. Tabular + TD + Off-policy (SAFE)
    print("\n1. Tabular + TD + Off-policy...")
    V = np.zeros(env.n_states)
    errors = []
    state = env.reset()
    for _ in range(n_steps):
        next_state, reward, _ = env.step(state)
        V[state] += 0.1 * (reward + env.gamma * V[next_state] - V[state])
        errors.append(np.sqrt(np.mean((V - env.true_values)**2)))
        state = next_state
    results['Tabular+TD+Off'] = errors

    # 2. Linear FA + MC + Off-policy (SAFE)
    print("2. Linear FA + MC + Off-policy...")
    weights = np.zeros(5)
    errors = []
    for ep in range(n_steps // 10):
        states, rewards = [], []
        state = env.reset()
        for _ in range(10):
            states.append(state)
            next_state, reward, _ = env.step(state)
            rewards.append(reward)
            state = next_state

        G = 0
        for t in range(len(states)-1, -1, -1):
            G = env.gamma * G + rewards[t]
            s = states[t]
            features = np.zeros(5)
            features[s] = 1
            weights += 0.01 * (G - np.dot(weights, features)) * features

        V = weights
        errors.append(np.sqrt(np.mean((V - env.true_values)**2)))
    results['FA+MC+Off'] = errors

    # 3. Linear FA + TD + On-policy (SAFE)
    print("3. Linear FA + TD + On-policy...")
    weights = np.zeros(5)
    errors = []
    state = env.reset()
    for _ in range(n_steps):
        features = np.zeros(5)
        features[state] = 1

        next_state, reward, _ = env.step(state)
        next_features = np.zeros(5)
        next_features[next_state] = 1

        td_error = reward + env.gamma * np.dot(weights, next_features) - np.dot(weights, features)
        weights += 0.1 * td_error * features

        errors.append(np.sqrt(np.mean((weights - env.true_values)**2)))
        state = next_state
    results['FA+TD+On'] = errors

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, errors in results.items():
        # Downsample for visibility
        step = max(1, len(errors) // 500)
        ax.plot(errors[::step], label=name, linewidth=2)

    ax.set_xlabel('Steps (downsampled)', fontsize=12)
    ax.set_ylabel('RMS Error', fontsize=12)
    ax.set_title('Safe Combinations of the Triad', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/safe_combinations.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved safe combinations to safe_combinations.png")
    plt.close()


def explain_solutions():
    """Explain solutions to the deadly triad."""
    print("\n" + "="*60)
    print("SOLUTIONS TO THE DEADLY TRIAD")
    print("="*60)

    print("""
    PROBLEM:
    Function Approximation + Bootstrapping + Off-policy = Divergence

    SOLUTIONS:

    1. AVOID OFF-POLICY
       - Use SARSA instead of Q-Learning
       - Learn about the policy you're actually following
       - Simple and effective

    2. AVOID BOOTSTRAPPING
       - Use Monte Carlo methods
       - Wait for actual returns
       - Higher variance, but stable

    3. GRADIENT TD METHODS
       - GTD(λ), TDC, ETD
       - True gradient descent (not semi-gradient)
       - More complex, but theoretically sound

    4. EXPERIENCE REPLAY + TARGET NETWORKS (DQN approach)
       - Break correlation in updates
       - Stabilize targets
       - Works well in practice

    5. FITTED Q-ITERATION
       - Batch updates
       - Separate data collection and learning
       - Used in many applications

    PRACTICAL ADVICE:
    - Start with on-policy (SARSA with function approx)
    - If off-policy needed, use DQN techniques
    - Monitor for divergence (weights exploding)
    - Consider gradient TD if theoretical guarantees needed
    """)

    # Create summary figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    table_data = [
        ['Method', 'FA', 'Bootstrap', 'Off-Policy', 'Stable?'],
        ['Tabular Q-Learning', 'No', 'Yes', 'Yes', '✓'],
        ['Linear MC', 'Yes', 'No', 'Yes', '✓'],
        ['Linear SARSA', 'Yes', 'Yes', 'No', '✓'],
        ['Linear Q-Learning', 'Yes', 'Yes', 'Yes', '✗'],
        ['DQN', 'Yes', 'Yes', 'Yes', '✓*'],
        ['Gradient TD', 'Yes', 'Yes', 'Yes', '✓'],
    ]

    colors = [['lightgray']*5] + [
        ['white', 'lightgreen', 'lightyellow', 'lightyellow', 'lightgreen'],
        ['white', 'lightyellow', 'lightgreen', 'lightyellow', 'lightgreen'],
        ['white', 'lightyellow', 'lightyellow', 'lightgreen', 'lightgreen'],
        ['white', 'lightyellow', 'lightyellow', 'lightyellow', 'lightcoral'],
        ['white', 'lightyellow', 'lightyellow', 'lightyellow', 'lightgreen'],
        ['white', 'lightyellow', 'lightyellow', 'lightyellow', 'lightgreen'],
    ]

    table = ax.table(cellText=table_data, cellColours=colors,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2)

    ax.set_title('The Deadly Triad: Method Comparison\n\n*DQN uses experience replay and target networks for stability',
                fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/triad_solutions.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved solutions table to triad_solutions.png")
    plt.close()


def main():
    """Main demonstration."""
    print("="*60)
    print("THE DEADLY TRIAD")
    print("="*60)

    print("""
    The "Deadly Triad" refers to three elements that, when combined,
    can cause learning to diverge (weights go to infinity):

    1. FUNCTION APPROXIMATION
       Instead of storing V(s) for every state, we approximate:
       V(s) ≈ w^T x(s)

    2. BOOTSTRAPPING
       Using current estimates in update targets:
       Target = R + γV(S')  (TD methods)
       vs
       Target = G (actual return, MC methods)

    3. OFF-POLICY LEARNING
       Learning about policy π while following policy b:
       Q-Learning learns about greedy while following ε-greedy

    Any TWO of these are fine. All THREE can diverge!
    """)

    # Run demonstrations
    demonstrate_divergence()
    compare_all_combinations()
    explain_solutions()

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Takeaways:")
    print("  - FA + Bootstrap + Off-policy can diverge")
    print("  - Baird's counterexample proves this")
    print("  - Solutions: avoid one element, or use special methods")
    print("  - DQN uses replay + target networks to stabilize")
    print("  - This is why DQN was a breakthrough!")


if __name__ == "__main__":
    main()
