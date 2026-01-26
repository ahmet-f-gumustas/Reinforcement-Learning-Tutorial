"""
03 - Gradient Monte Carlo Prediction

Monte Carlo prediction with function approximation using gradient descent.

Demonstrates:
- Gradient MC algorithm
- Convergence properties
- Comparison with tabular MC
- Feature impact analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Callable
from collections import defaultdict
import matplotlib.pyplot as plt


class RandomWalk:
    """Random Walk environment for MC prediction."""

    def __init__(self, n_states: int = 19):
        self.n_states = n_states
        self.terminal_left = 0
        self.terminal_right = n_states - 1
        self.start_state = n_states // 2

        # True values (analytically computed)
        self.true_values = np.array([i / (n_states - 1) for i in range(n_states)])
        self.true_values[0] = 0
        self.true_values[-1] = 1

    def reset(self) -> int:
        return self.start_state

    def step(self, state: int) -> Tuple[int, float, bool]:
        if state == self.terminal_left or state == self.terminal_right:
            return state, 0, True

        next_state = state + np.random.choice([-1, 1])

        if next_state == self.terminal_left:
            return next_state, 0, True
        elif next_state == self.terminal_right:
            return next_state, 1, True
        else:
            return next_state, 0, False

    def state_to_normalized(self, state: int) -> float:
        return state / (self.n_states - 1)


class LinearValueFunction:
    """Linear function approximation."""

    def __init__(self, n_features: int):
        self.weights = np.zeros(n_features)

    def predict(self, features: np.ndarray) -> float:
        return np.dot(self.weights, features)

    def update(self, features: np.ndarray, target: float, alpha: float):
        prediction = self.predict(features)
        error = target - prediction
        self.weights += alpha * error * features


def polynomial_features(state: float, degree: int) -> np.ndarray:
    """Polynomial features [1, s, s², ..., s^degree]."""
    return np.array([state ** i for i in range(degree + 1)])


def fourier_features(state: float, order: int) -> np.ndarray:
    """Fourier basis features."""
    return np.array([np.cos(np.pi * i * state) for i in range(order + 1)])


def state_aggregation_features(state: float, n_bins: int) -> np.ndarray:
    """State aggregation (one-hot) features."""
    features = np.zeros(n_bins)
    bin_idx = min(int(state * n_bins), n_bins - 1)
    features[bin_idx] = 1.0
    return features


def gradient_mc(env: RandomWalk, value_fn: LinearValueFunction,
                feature_fn: Callable, n_episodes: int, alpha: float,
                gamma: float = 1.0) -> Tuple[List[float], List[float]]:
    """
    Gradient Monte Carlo prediction.

    Update: w ← w + α [G - v̂(s,w)] ∇v̂(s,w)
    For linear: w ← w + α [G - w^T x(s)] x(s)
    """
    errors = []
    episode_returns = []

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

        # Compute returns and update
        G = 0
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            state = states[t]

            if state == env.terminal_left or state == env.terminal_right:
                continue

            normalized_state = env.state_to_normalized(state)
            features = feature_fn(normalized_state)
            value_fn.update(features, G, alpha)

        episode_returns.append(G)

        # Compute RMS error
        rms_error = compute_rms_error(env, value_fn, feature_fn)
        errors.append(rms_error)

    return errors, episode_returns


def tabular_mc(env: RandomWalk, n_episodes: int, alpha: float,
               gamma: float = 1.0) -> Tuple[Dict[int, float], List[float]]:
    """Tabular Monte Carlo for comparison."""
    V = defaultdict(float)
    errors = []

    for episode in range(n_episodes):
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

        G = 0
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            state = states[t]

            if state != env.terminal_left and state != env.terminal_right:
                V[state] += alpha * (G - V[state])

        # Compute RMS error
        se = 0
        count = 0
        for s in range(1, env.n_states - 1):
            se += (V[s] - env.true_values[s]) ** 2
            count += 1
        errors.append(np.sqrt(se / count) if count > 0 else 0)

    return V, errors


def compute_rms_error(env: RandomWalk, value_fn: LinearValueFunction,
                      feature_fn: Callable) -> float:
    """Compute RMS error vs true values."""
    se = 0
    count = 0
    for state in range(1, env.n_states - 1):
        normalized_state = env.state_to_normalized(state)
        features = feature_fn(normalized_state)
        predicted = value_fn.predict(features)
        true_value = env.true_values[state]
        se += (predicted - true_value) ** 2
        count += 1
    return np.sqrt(se / count) if count > 0 else 0


def compare_gradient_mc_tabular(n_runs: int = 30):
    """Compare gradient MC with tabular MC."""
    print("\n" + "="*60)
    print("GRADIENT MC vs TABULAR MC")
    print("="*60)

    env = RandomWalk(n_states=19)
    n_episodes = 500

    # Feature configurations
    feature_configs = [
        ('Polynomial (deg=5)', lambda s: polynomial_features(s, 5), 0.0001),
        ('Fourier (order=5)', lambda s: fourier_features(s, 5), 0.00005),
        ('State Agg (10 bins)', lambda s: state_aggregation_features(s, 10), 0.01),
    ]

    all_errors = {}

    # Run tabular MC
    print("\nRunning Tabular MC...")
    tabular_errors = []
    for run in range(n_runs):
        _, errors = tabular_mc(env, n_episodes, alpha=0.01)
        tabular_errors.append(errors)
    all_errors['Tabular MC'] = np.array(tabular_errors)

    # Run gradient MC with different features
    for name, feature_fn, alpha in feature_configs:
        print(f"Running {name}...")
        errors_list = []
        n_features = len(feature_fn(0.5))

        for run in range(n_runs):
            value_fn = LinearValueFunction(n_features)
            errors, _ = gradient_mc(env, value_fn, feature_fn, n_episodes, alpha)
            errors_list.append(errors)

        all_errors[name] = np.array(errors_list)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curves
    ax = axes[0]
    for name, errors in all_errors.items():
        mean = errors.mean(axis=0)
        std = errors.std(axis=0)
        ax.plot(mean, label=name, linewidth=2)
        ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('RMS Error', fontsize=12)
    ax.set_title('Learning Curves', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Final performance
    ax = axes[1]
    names = list(all_errors.keys())
    final_errors = [all_errors[name][:, -50:].mean(axis=1) for name in names]

    ax.boxplot(final_errors, labels=names)
    ax.set_ylabel('Final RMS Error', fontsize=12)
    ax.set_title('Final Performance', fontsize=14)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/gradient_mc_comparison.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved comparison to gradient_mc_comparison.png")
    plt.close()

    # Print statistics
    print("\nFinal RMS Error (last 50 episodes):")
    for name in names:
        final = all_errors[name][:, -50:].mean()
        std = all_errors[name][:, -50:].std()
        print(f"  {name:25s}: {final:.4f} ± {std:.4f}")


def visualize_learned_function(env: RandomWalk):
    """Visualize learned value function."""
    print("\n" + "="*60)
    print("VISUALIZING LEARNED VALUE FUNCTION")
    print("="*60)

    n_episodes = 2000

    feature_configs = [
        ('Polynomial (deg=3)', lambda s: polynomial_features(s, 3), 0.0001),
        ('Polynomial (deg=7)', lambda s: polynomial_features(s, 7), 0.00001),
        ('Fourier (order=5)', lambda s: fourier_features(s, 5), 0.00005),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Learned Value Functions', fontsize=16)

    states = np.linspace(0, 1, 100)
    true_states = np.arange(env.n_states) / (env.n_states - 1)

    for idx, (name, feature_fn, alpha) in enumerate(feature_configs):
        print(f"Training {name}...")
        n_features = len(feature_fn(0.5))
        value_fn = LinearValueFunction(n_features)
        gradient_mc(env, value_fn, feature_fn, n_episodes, alpha)

        # Plot
        ax = axes[idx]
        predicted = [value_fn.predict(feature_fn(s)) for s in states]

        ax.plot(states, predicted, 'b-', linewidth=2, label='Learned')
        ax.plot(true_states[1:-1], env.true_values[1:-1], 'ro', markersize=6, label='True')
        ax.set_xlabel('State', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(name, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/learned_functions_mc.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved learned functions to learned_functions_mc.png")
    plt.close()


def analyze_learning_rate(env: RandomWalk):
    """Analyze effect of learning rate."""
    print("\n" + "="*60)
    print("LEARNING RATE ANALYSIS")
    print("="*60)

    feature_fn = lambda s: fourier_features(s, 5)
    n_features = len(feature_fn(0.5))
    n_episodes = 500
    n_runs = 20

    alpha_values = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]

    plt.figure(figsize=(12, 6))

    for alpha in alpha_values:
        print(f"Testing α={alpha}...")
        all_errors = []

        for run in range(n_runs):
            value_fn = LinearValueFunction(n_features)
            errors, _ = gradient_mc(env, value_fn, feature_fn, n_episodes, alpha)
            all_errors.append(errors)

        mean_errors = np.array(all_errors).mean(axis=0)
        plt.plot(mean_errors, label=f'α={alpha}', linewidth=2)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('RMS Error', fontsize=12)
    plt.title('Effect of Learning Rate on Gradient MC', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/mc_learning_rate.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved learning rate analysis to mc_learning_rate.png")
    plt.close()


def main():
    """Main demonstration."""
    print("="*60)
    print("GRADIENT MONTE CARLO PREDICTION")
    print("="*60)

    print("""
    Gradient MC Algorithm:
    ----------------------
    For each episode:
        1. Generate full episode following policy
        2. Compute returns G_t for each state
        3. Update: w ← w + α [G_t - v̂(S_t,w)] ∇v̂(S_t,w)

    For linear function approximation:
        w ← w + α [G_t - w^T x(S_t)] x(S_t)

    Properties:
        - Unbiased (uses true returns)
        - High variance (full episode needed)
        - Guaranteed convergence for linear FA
    """)

    env = RandomWalk(n_states=19)

    # Run comparisons
    compare_gradient_mc_tabular(n_runs=30)
    visualize_learned_function(env)
    analyze_learning_rate(env)

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Findings:")
    print("  - Gradient MC converges to approximation of true values")
    print("  - Feature choice significantly affects performance")
    print("  - Learning rate needs careful tuning")
    print("  - Tabular MC can be more accurate but doesn't generalize")


if __name__ == "__main__":
    main()
