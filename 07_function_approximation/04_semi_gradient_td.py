"""
04 - Semi-Gradient TD Prediction

TD(0) prediction with function approximation using semi-gradient methods.

Demonstrates:
- Semi-gradient TD algorithm
- Online learning (no need for full episodes)
- Comparison with gradient MC
- Convergence properties
"""

import numpy as np
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt


class RandomWalk:
    """Random Walk environment."""

    def __init__(self, n_states: int = 19):
        self.n_states = n_states
        self.terminal_left = 0
        self.terminal_right = n_states - 1
        self.start_state = n_states // 2
        self.gamma = 1.0

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

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()


def polynomial_features(state: float, degree: int) -> np.ndarray:
    return np.array([state ** i for i in range(degree + 1)])


def fourier_features(state: float, order: int) -> np.ndarray:
    return np.array([np.cos(np.pi * i * state) for i in range(order + 1)])


def state_aggregation_features(state: float, n_bins: int) -> np.ndarray:
    features = np.zeros(n_bins)
    bin_idx = min(int(state * n_bins), n_bins - 1)
    features[bin_idx] = 1.0
    return features


def semi_gradient_td(env: RandomWalk, value_fn: LinearValueFunction,
                     feature_fn: Callable, n_episodes: int, alpha: float) -> List[float]:
    """
    Semi-gradient TD(0) prediction.

    Update: w ← w + α [R + γv̂(S',w) - v̂(S,w)] ∇v̂(S,w)
    For linear: w ← w + α [R + γw^T x(S') - w^T x(S)] x(S)

    Note: We ignore the gradient through v̂(S',w) - hence "semi-gradient"
    """
    errors = []

    for episode in range(n_episodes):
        state = env.reset()

        while True:
            if state == env.terminal_left or state == env.terminal_right:
                break

            # Get features for current state
            normalized_state = env.state_to_normalized(state)
            features = feature_fn(normalized_state)

            # Take step
            next_state, reward, done = env.step(state)

            # Get value of next state
            if done or next_state == env.terminal_left or next_state == env.terminal_right:
                next_value = 0 if next_state == env.terminal_left else (1 if next_state == env.terminal_right else 0)
                # For terminal states, use the actual terminal value
                if next_state == env.terminal_right:
                    next_value = 0  # The reward was already given
            else:
                normalized_next = env.state_to_normalized(next_state)
                next_features = feature_fn(normalized_next)
                next_value = value_fn.predict(next_features)

            # TD target and error
            td_target = reward + env.gamma * next_value
            current_value = value_fn.predict(features)
            td_error = td_target - current_value

            # Semi-gradient update
            value_fn.weights += alpha * td_error * features

            if done:
                break

            state = next_state

        # Compute RMS error
        rms_error = compute_rms_error(env, value_fn, feature_fn)
        errors.append(rms_error)

    return errors


def gradient_mc(env: RandomWalk, value_fn: LinearValueFunction,
                feature_fn: Callable, n_episodes: int, alpha: float) -> List[float]:
    """Gradient MC for comparison."""
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
            G = env.gamma * G + rewards[t]
            state = states[t]

            if state == env.terminal_left or state == env.terminal_right:
                continue

            normalized_state = env.state_to_normalized(state)
            features = feature_fn(normalized_state)

            prediction = value_fn.predict(features)
            error = G - prediction
            value_fn.weights += alpha * error * features

        rms_error = compute_rms_error(env, value_fn, feature_fn)
        errors.append(rms_error)

    return errors


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


def compare_td_mc(n_runs: int = 30):
    """Compare semi-gradient TD with gradient MC."""
    print("\n" + "="*60)
    print("SEMI-GRADIENT TD vs GRADIENT MC")
    print("="*60)

    env = RandomWalk(n_states=19)
    n_episodes = 500

    feature_fn = lambda s: fourier_features(s, 5)
    n_features = len(feature_fn(0.5))

    td_errors = []
    mc_errors = []

    print(f"\nRunning {n_runs} trials...")
    for run in range(n_runs):
        if run % 10 == 0:
            print(f"  Run {run + 1}/{n_runs}")

        # TD
        value_fn_td = LinearValueFunction(n_features)
        errors_td = semi_gradient_td(env, value_fn_td, feature_fn, n_episodes, alpha=0.0001)
        td_errors.append(errors_td)

        # MC
        value_fn_mc = LinearValueFunction(n_features)
        errors_mc = gradient_mc(env, value_fn_mc, feature_fn, n_episodes, alpha=0.00005)
        mc_errors.append(errors_mc)

    td_errors = np.array(td_errors)
    mc_errors = np.array(mc_errors)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curves
    ax = axes[0]
    td_mean = td_errors.mean(axis=0)
    mc_mean = mc_errors.mean(axis=0)
    td_std = td_errors.std(axis=0)
    mc_std = mc_errors.std(axis=0)

    ax.plot(td_mean, label='Semi-gradient TD', linewidth=2)
    ax.fill_between(range(len(td_mean)), td_mean - td_std, td_mean + td_std, alpha=0.2)
    ax.plot(mc_mean, label='Gradient MC', linewidth=2)
    ax.fill_between(range(len(mc_mean)), mc_mean - mc_std, mc_mean + mc_std, alpha=0.2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('RMS Error', fontsize=12)
    ax.set_title('Learning Curves (Fourier features, order 5)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Final performance boxplot
    ax = axes[1]
    td_final = td_errors[:, -50:].mean(axis=1)
    mc_final = mc_errors[:, -50:].mean(axis=1)

    ax.boxplot([td_final, mc_final], labels=['Semi-gradient TD', 'Gradient MC'])
    ax.set_ylabel('Final RMS Error', fontsize=12)
    ax.set_title('Final Performance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/td_vs_mc.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved comparison to td_vs_mc.png")
    plt.close()

    print(f"\nFinal RMS Error (last 50 episodes):")
    print(f"  Semi-gradient TD: {td_final.mean():.4f} ± {td_final.std():.4f}")
    print(f"  Gradient MC:      {mc_final.mean():.4f} ± {mc_final.std():.4f}")


def analyze_td_properties():
    """Analyze properties of semi-gradient TD."""
    print("\n" + "="*60)
    print("SEMI-GRADIENT TD PROPERTIES")
    print("="*60)

    env = RandomWalk(n_states=19)
    n_episodes = 1000

    # Different feature types
    feature_configs = [
        ('State Aggregation (5 bins)', lambda s: state_aggregation_features(s, 5), 0.1),
        ('State Aggregation (10 bins)', lambda s: state_aggregation_features(s, 10), 0.1),
        ('Polynomial (deg=3)', lambda s: polynomial_features(s, 3), 0.0001),
        ('Fourier (order=5)', lambda s: fourier_features(s, 5), 0.0001),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Semi-gradient TD with Different Features', fontsize=16)

    for idx, (name, feature_fn, alpha) in enumerate(feature_configs):
        ax = axes[idx // 2, idx % 2]
        print(f"Testing {name}...")

        n_features = len(feature_fn(0.5))
        n_runs = 20
        all_errors = []

        for run in range(n_runs):
            value_fn = LinearValueFunction(n_features)
            errors = semi_gradient_td(env, value_fn, feature_fn, n_episodes, alpha)
            all_errors.append(errors)

        mean_errors = np.array(all_errors).mean(axis=0)
        std_errors = np.array(all_errors).std(axis=0)

        ax.plot(mean_errors, linewidth=2)
        ax.fill_between(range(len(mean_errors)),
                       mean_errors - std_errors,
                       mean_errors + std_errors,
                       alpha=0.3)
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('RMS Error', fontsize=11)
        ax.set_title(f'{name}', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/td_features.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved feature analysis to td_features.png")
    plt.close()


def visualize_online_learning():
    """Visualize the online nature of TD learning."""
    print("\n" + "="*60)
    print("ONLINE LEARNING VISUALIZATION")
    print("="*60)

    env = RandomWalk(n_states=19)
    feature_fn = lambda s: fourier_features(s, 5)
    n_features = len(feature_fn(0.5))
    value_fn = LinearValueFunction(n_features)
    alpha = 0.001

    # Track value function at specific states during learning
    track_states = [5, 9, 13]  # Different positions
    value_history = {s: [] for s in track_states}

    n_steps = 5000
    state = env.reset()
    step = 0

    while step < n_steps:
        if state == env.terminal_left or state == env.terminal_right:
            state = env.reset()
            continue

        # Record values
        for s in track_states:
            ns = env.state_to_normalized(s)
            features = feature_fn(ns)
            value_history[s].append(value_fn.predict(features))

        # TD update
        normalized_state = env.state_to_normalized(state)
        features = feature_fn(normalized_state)

        next_state, reward, done = env.step(state)

        if done:
            next_value = 0
        else:
            normalized_next = env.state_to_normalized(next_state)
            next_features = feature_fn(normalized_next)
            next_value = value_fn.predict(next_features)

        td_target = reward + env.gamma * next_value
        td_error = td_target - value_fn.predict(features)
        value_fn.weights += alpha * td_error * features

        if done:
            state = env.reset()
        else:
            state = next_state

        step += 1

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Value evolution
    ax = axes[0]
    for s in track_states:
        true_val = env.true_values[s]
        ax.plot(value_history[s], label=f'State {s} (true={true_val:.2f})', linewidth=2)
        ax.axhline(y=true_val, linestyle='--', alpha=0.5)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Estimated Value', fontsize=12)
    ax.set_title('Online Value Learning', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Final learned function
    ax = axes[1]
    states = np.linspace(0, 1, 100)
    predicted = [value_fn.predict(feature_fn(s)) for s in states]

    true_states = np.arange(env.n_states) / (env.n_states - 1)

    ax.plot(states, predicted, 'b-', linewidth=2, label='Learned')
    ax.plot(true_states[1:-1], env.true_values[1:-1], 'ro', markersize=6, label='True')
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Final Learned Value Function', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/online_learning.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved online learning visualization to online_learning.png")
    plt.close()


def main():
    """Main demonstration."""
    print("="*60)
    print("SEMI-GRADIENT TD PREDICTION")
    print("="*60)

    print("""
    Semi-gradient TD(0) Algorithm:
    ------------------------------
    For each step in episode:
        1. Observe current state S, take action, get R, S'
        2. Compute TD target: R + γv̂(S', w)
        3. Update: w ← w + α [R + γv̂(S',w) - v̂(S,w)] ∇v̂(S,w)

    Why "Semi-gradient"?
        The full gradient would include ∇v̂(S',w) in the target,
        but we ignore it for stability. This works well in practice.

    Advantages over MC:
        - Online: Updates every step (no need for full episode)
        - Lower variance
        - Can work with continuing tasks

    Disadvantages:
        - Biased estimates
        - Can diverge with function approximation (deadly triad)
    """)

    # Run demonstrations
    compare_td_mc(n_runs=30)
    analyze_td_properties()
    visualize_online_learning()

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Findings:")
    print("  - TD learns faster (online updates)")
    print("  - MC has lower bias (uses true returns)")
    print("  - Feature choice affects both similarly")
    print("  - TD is the basis for many practical algorithms")


if __name__ == "__main__":
    main()
