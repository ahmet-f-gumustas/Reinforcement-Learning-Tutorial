"""
01 - Linear Function Approximation

Introduction to approximating value functions with linear combinations of features.

Demonstrates:
- Feature vectors and weight vectors
- Linear value function approximation
- Gradient computation
- Simple prediction examples
"""

import numpy as np
from typing import Callable, List, Tuple
import matplotlib.pyplot as plt


class LinearValueFunction:
    """
    Linear function approximation for value functions.

    V(s) ≈ w^T x(s) = Σ w_i * x_i(s)
    """

    def __init__(self, n_features: int):
        """
        Initialize with random weights.

        Args:
            n_features: Number of features in feature vector
        """
        self.n_features = n_features
        self.weights = np.zeros(n_features)

    def predict(self, features: np.ndarray) -> float:
        """
        Predict value for given feature vector.

        Args:
            features: Feature vector x(s)

        Returns:
            Estimated value v̂(s, w)
        """
        return np.dot(self.weights, features)

    def gradient(self, features: np.ndarray) -> np.ndarray:
        """
        Compute gradient of value function w.r.t. weights.

        For linear function: ∇_w v̂(s, w) = x(s)
        """
        return features

    def update(self, features: np.ndarray, target: float, alpha: float):
        """
        Update weights using gradient descent.

        w ← w + α [target - v̂(s, w)] ∇_w v̂(s, w)
        """
        prediction = self.predict(features)
        error = target - prediction
        self.weights += alpha * error * self.gradient(features)

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()


def polynomial_features(state: float, degree: int) -> np.ndarray:
    """
    Create polynomial features: [1, s, s², s³, ..., s^degree]

    Args:
        state: Scalar state value
        degree: Maximum polynomial degree

    Returns:
        Feature vector of length degree+1
    """
    return np.array([state ** i for i in range(degree + 1)])


def fourier_features(state: float, order: int) -> np.ndarray:
    """
    Create Fourier basis features: cos(πi·s) for i = 0, 1, ..., order

    Args:
        state: State value in [0, 1]
        order: Number of Fourier terms

    Returns:
        Feature vector of length order+1
    """
    return np.array([np.cos(np.pi * i * state) for i in range(order + 1)])


def rbf_features(state: float, centers: np.ndarray, sigma: float = 0.2) -> np.ndarray:
    """
    Create Radial Basis Function features.

    x_i(s) = exp(-||s - c_i||² / 2σ²)

    Args:
        state: State value
        centers: RBF center positions
        sigma: Width of Gaussian bumps

    Returns:
        Feature vector
    """
    return np.exp(-((state - centers) ** 2) / (2 * sigma ** 2))


class RandomWalk:
    """
    Simple 1D Random Walk environment.

    States: 0, 1, 2, ..., n_states-1
    Terminal: state 0 (left, reward 0) and state n_states-1 (right, reward 1)
    Transitions: Equal probability left/right
    """

    def __init__(self, n_states: int = 7):
        self.n_states = n_states
        self.terminal_left = 0
        self.terminal_right = n_states - 1
        self.start_state = n_states // 2

        # Compute true values analytically
        self.true_values = self._compute_true_values()

    def _compute_true_values(self) -> np.ndarray:
        """Compute true values for random walk."""
        values = np.zeros(self.n_states)
        # V(s) = s / (n_states - 1) for non-terminal states
        for s in range(1, self.n_states - 1):
            values[s] = s / (self.n_states - 1)
        values[self.terminal_right] = 1.0
        return values

    def reset(self) -> int:
        return self.start_state

    def step(self, state: int) -> Tuple[int, float, bool]:
        """Take random step left or right."""
        if state == self.terminal_left or state == self.terminal_right:
            return state, 0, True

        # Random walk
        next_state = state + np.random.choice([-1, 1])

        if next_state == self.terminal_left:
            return next_state, 0, True
        elif next_state == self.terminal_right:
            return next_state, 1, True
        else:
            return next_state, 0, False

    def state_to_normalized(self, state: int) -> float:
        """Convert state to [0, 1] range."""
        return state / (self.n_states - 1)


def gradient_mc_prediction(env: RandomWalk, value_fn: LinearValueFunction,
                           feature_fn: Callable, n_episodes: int = 1000,
                           alpha: float = 0.01, gamma: float = 1.0) -> List[float]:
    """
    Gradient Monte Carlo prediction with linear function approximation.

    Args:
        env: Environment
        value_fn: Linear value function
        feature_fn: Function to extract features from state
        n_episodes: Number of episodes
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        List of RMS errors per episode
    """
    errors = []

    for episode in range(n_episodes):
        # Generate episode
        states = []
        rewards = []
        state = env.reset()

        while True:
            states.append(state)
            next_state, reward, done = env.step(state)
            rewards.append(reward)
            if done:
                break
            state = next_state

        # Compute returns and update (backward)
        G = 0
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            state = states[t]

            # Skip terminal states
            if state == env.terminal_left or state == env.terminal_right:
                continue

            # Get features
            normalized_state = env.state_to_normalized(state)
            features = feature_fn(normalized_state)

            # Update weights
            value_fn.update(features, G, alpha)

        # Compute RMS error
        rms = compute_rms_error(env, value_fn, feature_fn)
        errors.append(rms)

    return errors


def compute_rms_error(env: RandomWalk, value_fn: LinearValueFunction,
                      feature_fn: Callable) -> float:
    """Compute RMS error against true values."""
    errors = []
    for state in range(1, env.n_states - 1):
        normalized_state = env.state_to_normalized(state)
        features = feature_fn(normalized_state)
        predicted = value_fn.predict(features)
        true_value = env.true_values[state]
        errors.append((predicted - true_value) ** 2)

    return np.sqrt(np.mean(errors))


def compare_feature_types():
    """Compare different feature representations."""
    print("\n" + "="*60)
    print("COMPARING FEATURE TYPES")
    print("="*60)

    env = RandomWalk(n_states=7)
    n_episodes = 500
    n_runs = 20

    feature_configs = [
        ('Polynomial (deg=3)', lambda s: polynomial_features(s, 3)),
        ('Polynomial (deg=5)', lambda s: polynomial_features(s, 5)),
        ('Fourier (order=3)', lambda s: fourier_features(s, 3)),
        ('Fourier (order=5)', lambda s: fourier_features(s, 5)),
        ('RBF (5 centers)', lambda s: rbf_features(s, np.linspace(0, 1, 5))),
        ('RBF (10 centers)', lambda s: rbf_features(s, np.linspace(0, 1, 10))),
    ]

    plt.figure(figsize=(12, 6))

    for name, feature_fn in feature_configs:
        print(f"\nTesting {name}...")

        # Get feature dimension
        test_features = feature_fn(0.5)
        n_features = len(test_features)

        all_errors = []
        for run in range(n_runs):
            value_fn = LinearValueFunction(n_features)
            errors = gradient_mc_prediction(env, value_fn, feature_fn,
                                           n_episodes=n_episodes, alpha=0.01)
            all_errors.append(errors)

        mean_errors = np.array(all_errors).mean(axis=0)
        plt.plot(mean_errors, label=name, linewidth=2)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('RMS Error', fontsize=12)
    plt.title('Feature Type Comparison on Random Walk', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/feature_comparison.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved comparison to feature_comparison.png")
    plt.close()


def visualize_learned_values():
    """Visualize learned value function vs true values."""
    print("\n" + "="*60)
    print("VISUALIZING LEARNED VALUES")
    print("="*60)

    env = RandomWalk(n_states=21)  # More states for smoother visualization

    # Train with polynomial features
    feature_fn = lambda s: polynomial_features(s, 5)
    n_features = len(feature_fn(0.5))
    value_fn = LinearValueFunction(n_features)

    print("Training with polynomial features (degree 5)...")
    errors = gradient_mc_prediction(env, value_fn, feature_fn,
                                   n_episodes=2000, alpha=0.001)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Value function
    ax = axes[0]
    states = np.linspace(0, 1, 100)
    predicted_values = [value_fn.predict(feature_fn(s)) for s in states]

    true_states = np.arange(env.n_states) / (env.n_states - 1)
    true_values = env.true_values

    ax.plot(states, predicted_values, 'b-', linewidth=2, label='Learned')
    ax.plot(true_states[1:-1], true_values[1:-1], 'ro', markersize=8, label='True')
    ax.set_xlabel('State (normalized)', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Learned vs True Value Function', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Learning curve
    ax = axes[1]
    ax.plot(errors, linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('RMS Error', fontsize=12)
    ax.set_title('Learning Curve', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/learned_values.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved visualization to learned_values.png")
    plt.close()

    # Print weights
    print("\nLearned weights:")
    for i, w in enumerate(value_fn.get_weights()):
        print(f"  w_{i} (s^{i}): {w:.4f}")


def demonstrate_generalization():
    """Demonstrate how function approximation generalizes."""
    print("\n" + "="*60)
    print("GENERALIZATION DEMONSTRATION")
    print("="*60)

    env = RandomWalk(n_states=21)

    # Train only on subset of states
    feature_fn = lambda s: polynomial_features(s, 3)
    n_features = len(feature_fn(0.5))
    value_fn = LinearValueFunction(n_features)

    # Manual training on specific states
    training_states = [0.2, 0.4, 0.6, 0.8]  # Only 4 training points
    training_targets = [0.2, 0.4, 0.6, 0.8]  # Approximate true values

    print(f"Training on only {len(training_states)} states...")
    for _ in range(1000):
        for state, target in zip(training_states, training_targets):
            features = feature_fn(state)
            value_fn.update(features, target, alpha=0.01)

    # Evaluate on all states
    plt.figure(figsize=(10, 6))

    all_states = np.linspace(0, 1, 50)
    predicted = [value_fn.predict(feature_fn(s)) for s in all_states]
    true_values = all_states  # True value = state for random walk

    plt.plot(all_states, predicted, 'b-', linewidth=2, label='Learned (generalized)')
    plt.plot(all_states, true_values, 'g--', linewidth=2, label='True values')
    plt.scatter(training_states, training_targets, c='red', s=100, zorder=5, label='Training points')

    plt.xlabel('State (normalized)', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Generalization: Learning from Few Points', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/generalization.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved generalization demo to generalization.png")
    plt.close()

    print("\nKey insight: We trained on only 4 states,")
    print("but the function generalizes to ALL states!")


def main():
    """Main demonstration."""
    print("="*60)
    print("LINEAR FUNCTION APPROXIMATION")
    print("="*60)

    print("""
    Why Function Approximation?

    Tabular methods store V(s) for EVERY state:
    - GridWorld 10×10: 100 states (OK)
    - Chess: ~10^43 states (IMPOSSIBLE!)
    - Continuous states: Infinite states

    Solution: Approximate V(s) ≈ w^T x(s)
    - Store only weights (finite)
    - Generalize across similar states
    - Handle continuous/large state spaces
    """)

    # Run demonstrations
    compare_feature_types()
    visualize_learned_values()
    demonstrate_generalization()

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Concepts:")
    print("  - V(s) ≈ w^T x(s) = Σ w_i x_i(s)")
    print("  - Features x(s) define what can be learned")
    print("  - Gradient ∇_w v̂(s,w) = x(s) for linear")
    print("  - Generalization: similar states → similar values")


if __name__ == "__main__":
    main()
