"""
06 - Mountain Car with Function Approximation

Solving Mountain Car using linear function approximation with tile coding.

Demonstrates:
- Continuous state space handling
- Tile coding for 2D state space
- Semi-gradient SARSA on a real problem
- Learning curve analysis
"""

import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


class SimpleMountainCar:
    """
    Simple Mountain Car environment.

    State: (position, velocity)
    - Position: [-1.2, 0.5]
    - Velocity: [-0.07, 0.07]

    Actions: 0=Left, 1=None, 2=Right

    Goal: Reach position >= 0.5
    """

    def __init__(self):
        self.min_position = -1.2
        self.max_position = 0.5
        self.min_velocity = -0.07
        self.max_velocity = 0.07
        self.goal_position = 0.5

        self.n_actions = 3
        self.actions = [0, 1, 2]
        self.gamma = 1.0

        self.position = None
        self.velocity = None

    def reset(self) -> np.ndarray:
        self.position = np.random.uniform(-0.6, -0.4)
        self.velocity = 0.0
        return np.array([self.position, self.velocity])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Take action and return (next_state, reward, done)."""
        force = action - 1  # -1, 0, or 1

        # Physics
        self.velocity += 0.001 * force - 0.0025 * np.cos(3 * self.position)
        self.velocity = np.clip(self.velocity, self.min_velocity, self.max_velocity)

        self.position += self.velocity
        self.position = np.clip(self.position, self.min_position, self.max_position)

        # Boundary condition
        if self.position == self.min_position and self.velocity < 0:
            self.velocity = 0

        done = self.position >= self.goal_position
        reward = -1.0  # Reward is -1 for every step

        return np.array([self.position, self.velocity]), reward, done


def get_mountain_car_env():
    """Get Mountain Car environment."""
    if HAS_GYM:
        env = gym.make('MountainCar-v0')
        return env, True
    else:
        return SimpleMountainCar(), False


class TileCodingMountainCar:
    """
    Tile coding specifically designed for Mountain Car.
    """

    def __init__(self, n_tilings: int = 8, n_tiles: int = 8):
        self.n_tilings = n_tilings
        self.n_tiles = n_tiles
        self.n_actions = 3

        # State ranges
        self.pos_range = (-1.2, 0.5)
        self.vel_range = (-0.07, 0.07)

        # Tile widths
        self.pos_width = (self.pos_range[1] - self.pos_range[0]) / n_tiles
        self.vel_width = (self.vel_range[1] - self.vel_range[0]) / n_tiles

        # Random offsets for each tiling
        np.random.seed(42)
        self.offsets = [
            (np.random.random() * self.pos_width,
             np.random.random() * self.vel_width)
            for _ in range(n_tilings)
        ]

        # Total features
        self.features_per_tiling = n_tiles * n_tiles
        self.total_features = n_tilings * self.features_per_tiling * n_actions

        # Weights
        self.weights = np.zeros(self.total_features)

    def get_active_features(self, state: np.ndarray, action: int) -> List[int]:
        """Get indices of active features for given state-action pair."""
        position, velocity = state
        active = []

        action_offset = action * self.n_tilings * self.features_per_tiling

        for tiling_idx, (pos_offset, vel_offset) in enumerate(self.offsets):
            # Get tile indices
            pos_tile = int((position - self.pos_range[0] - pos_offset) / self.pos_width)
            vel_tile = int((velocity - self.vel_range[0] - vel_offset) / self.vel_width)

            # Clip to valid range
            pos_tile = max(0, min(self.n_tiles - 1, pos_tile))
            vel_tile = max(0, min(self.n_tiles - 1, vel_tile))

            # Compute feature index
            tiling_offset = tiling_idx * self.features_per_tiling
            feature_idx = action_offset + tiling_offset + pos_tile * self.n_tiles + vel_tile
            active.append(feature_idx)

        return active

    def predict(self, state: np.ndarray, action: int) -> float:
        """Predict Q(s, a)."""
        active = self.get_active_features(state, action)
        return np.sum(self.weights[active])

    def predict_all(self, state: np.ndarray) -> np.ndarray:
        """Predict Q(s, a) for all actions."""
        return np.array([self.predict(state, a) for a in range(self.n_actions)])

    def update(self, state: np.ndarray, action: int, target: float, alpha: float):
        """Update weights using semi-gradient."""
        active = self.get_active_features(state, action)
        prediction = np.sum(self.weights[active])
        error = target - prediction
        # Divide by n_tilings to keep step size reasonable
        for idx in active:
            self.weights[idx] += alpha * error / self.n_tilings


def epsilon_greedy(q_values: np.ndarray, epsilon: float) -> int:
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(len(q_values))
    return np.argmax(q_values)


def sarsa_mountain_car(env, q_fn: TileCodingMountainCar, n_episodes: int,
                       alpha: float = 0.5, epsilon: float = 0.0,
                       max_steps: int = 1000) -> Tuple[List[int], List[float]]:
    """
    Semi-gradient SARSA for Mountain Car.

    Returns:
        steps_per_episode: Number of steps to reach goal
        rewards_per_episode: Total reward per episode
    """
    steps_history = []
    rewards_history = []
    is_gym = hasattr(env, 'unwrapped')

    for episode in range(n_episodes):
        # Reset
        if is_gym:
            state, _ = env.reset()
        else:
            state = env.reset()

        # Choose initial action
        q_values = q_fn.predict_all(state)
        action = epsilon_greedy(q_values, epsilon)

        total_reward = 0
        steps = 0

        while steps < max_steps:
            # Take action
            if is_gym:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            else:
                next_state, reward, done = env.step(action)

            total_reward += reward
            steps += 1

            if done:
                # Terminal update
                q_fn.update(state, action, reward, alpha)
                break

            # Choose next action
            next_q_values = q_fn.predict_all(next_state)
            next_action = epsilon_greedy(next_q_values, epsilon)

            # SARSA update: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
            target = reward + q_fn.predict(next_state, next_action)
            q_fn.update(state, action, target, alpha)

            state = next_state
            action = next_action

        steps_history.append(steps)
        rewards_history.append(total_reward)

        if episode % 100 == 0:
            avg_steps = np.mean(steps_history[-100:]) if len(steps_history) >= 100 else np.mean(steps_history)
            print(f"Episode {episode}, Avg Steps: {avg_steps:.1f}")

    return steps_history, rewards_history


def analyze_hyperparameters():
    """Analyze effect of hyperparameters."""
    print("\n" + "="*60)
    print("HYPERPARAMETER ANALYSIS")
    print("="*60)

    env, _ = get_mountain_car_env()
    n_episodes = 500
    n_runs = 5

    # Test different number of tilings
    tiling_configs = [2, 4, 8, 16]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Effect of number of tilings
    ax = axes[0]
    for n_tilings in tiling_configs:
        print(f"\nTesting {n_tilings} tilings...")
        all_steps = []

        for run in range(n_runs):
            q_fn = TileCodingMountainCar(n_tilings=n_tilings, n_tiles=8)
            steps, _ = sarsa_mountain_car(env, q_fn, n_episodes, alpha=0.5/n_tilings)
            all_steps.append(steps)

        mean_steps = np.array(all_steps).mean(axis=0)
        window = 20
        smoothed = np.convolve(mean_steps, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=f'{n_tilings} tilings', linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Steps to Goal', fontsize=12)
    ax.set_title('Effect of Number of Tilings', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Effect of learning rate
    ax = axes[1]
    alpha_values = [0.1, 0.3, 0.5, 1.0]

    for alpha in alpha_values:
        print(f"\nTesting α={alpha}...")
        all_steps = []

        for run in range(n_runs):
            q_fn = TileCodingMountainCar(n_tilings=8, n_tiles=8)
            steps, _ = sarsa_mountain_car(env, q_fn, n_episodes, alpha=alpha/8)
            all_steps.append(steps)

        mean_steps = np.array(all_steps).mean(axis=0)
        smoothed = np.convolve(mean_steps, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=f'α={alpha}/n_tilings', linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Steps to Goal', fontsize=12)
    ax.set_title('Effect of Learning Rate', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/mountain_car_hyperparams.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved hyperparameter analysis to mountain_car_hyperparams.png")
    plt.close()


def visualize_value_function(q_fn: TileCodingMountainCar):
    """Visualize learned value function and policy."""
    print("\n" + "="*60)
    print("VISUALIZING VALUE FUNCTION")
    print("="*60)

    # Create grid
    positions = np.linspace(-1.2, 0.5, 50)
    velocities = np.linspace(-0.07, 0.07, 50)

    value_grid = np.zeros((len(velocities), len(positions)))
    policy_grid = np.zeros((len(velocities), len(positions)))

    for i, vel in enumerate(velocities):
        for j, pos in enumerate(positions):
            state = np.array([pos, vel])
            q_values = q_fn.predict_all(state)
            value_grid[i, j] = np.max(q_values)
            policy_grid[i, j] = np.argmax(q_values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Value function
    ax = axes[0]
    im = ax.imshow(value_grid, extent=[-1.2, 0.5, -0.07, 0.07],
                   aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Velocity', fontsize=12)
    ax.set_title('Value Function (max Q)', fontsize=14)
    plt.colorbar(im, ax=ax, label='Value')

    # Draw goal region
    ax.axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='Goal')
    ax.legend(fontsize=10)

    # Policy
    ax = axes[1]
    im = ax.imshow(policy_grid, extent=[-1.2, 0.5, -0.07, 0.07],
                   aspect='auto', origin='lower', cmap='coolwarm')
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Velocity', fontsize=12)
    ax.set_title('Policy (0=Left, 1=None, 2=Right)', fontsize=14)
    plt.colorbar(im, ax=ax, label='Action')

    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/mountain_car_value.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved value function to mountain_car_value.png")
    plt.close()


def demonstrate_learned_policy(env, q_fn: TileCodingMountainCar):
    """Run and visualize a test episode."""
    print("\n" + "="*60)
    print("DEMONSTRATING LEARNED POLICY")
    print("="*60)

    is_gym = hasattr(env, 'unwrapped')

    # Run test episode
    if is_gym:
        state, _ = env.reset()
    else:
        state = env.reset()

    positions = [state[0]]
    velocities = [state[1]]
    actions = []

    done = False
    steps = 0

    while not done and steps < 200:
        q_values = q_fn.predict_all(state)
        action = np.argmax(q_values)  # Greedy
        actions.append(action)

        if is_gym:
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        else:
            state, _, done = env.step(action)

        positions.append(state[0])
        velocities.append(state[1])
        steps += 1

    print(f"Episode completed in {steps} steps")
    print(f"Final position: {positions[-1]:.3f}")

    # Plot trajectory
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Position and velocity over time
    ax = axes[0]
    ax.plot(positions, 'b-', linewidth=2, label='Position')
    ax.plot(velocities, 'r-', linewidth=2, label='Velocity')
    ax.axhline(y=0.5, color='g', linestyle='--', linewidth=2, label='Goal')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Position and Velocity Over Time', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Phase space trajectory
    ax = axes[1]
    ax.plot(positions, velocities, 'b-', linewidth=1, alpha=0.7)
    ax.scatter(positions[0], velocities[0], c='green', s=100, zorder=5, label='Start')
    ax.scatter(positions[-1], velocities[-1], c='red', s=100, zorder=5, label='End')
    ax.axvline(x=0.5, color='g', linestyle='--', linewidth=2)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Velocity', fontsize=12)
    ax.set_title('Phase Space Trajectory', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/mountain_car_trajectory.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved trajectory to mountain_car_trajectory.png")
    plt.close()


def main():
    """Main demonstration."""
    print("="*60)
    print("MOUNTAIN CAR WITH FUNCTION APPROXIMATION")
    print("="*60)

    print("""
    Mountain Car Problem:
    ---------------------
    - Car stuck in a valley between two hills
    - Goal: Drive to the top of the right hill (position >= 0.5)
    - Challenge: Car's engine is too weak to climb directly
    - Solution: Build momentum by oscillating

    State Space (continuous):
    - Position: [-1.2, 0.5]
    - Velocity: [-0.07, 0.07]

    Actions (discrete):
    - 0: Push left
    - 1: No push
    - 2: Push right

    Reward: -1 per step (want to reach goal quickly)
    """)

    env, is_gym = get_mountain_car_env()

    # Analyze hyperparameters
    analyze_hyperparameters()

    # Train final agent
    print("\n" + "="*60)
    print("TRAINING FINAL AGENT")
    print("="*60)

    q_fn = TileCodingMountainCar(n_tilings=8, n_tiles=8)
    steps, rewards = sarsa_mountain_car(env, q_fn, n_episodes=500, alpha=0.5/8)

    # Visualizations
    visualize_value_function(q_fn)

    # Reset environment for demonstration
    if is_gym:
        env = gym.make('MountainCar-v0')
    else:
        env = SimpleMountainCar()

    demonstrate_learned_policy(env, q_fn)

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Findings:")
    print("  - Tile coding handles continuous states effectively")
    print("  - More tilings = better approximation but more features")
    print("  - Learning rate should scale with 1/n_tilings")
    print("  - Agent learns to build momentum (oscillate)")


if __name__ == "__main__":
    main()
