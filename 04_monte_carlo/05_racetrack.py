"""
05 - Racetrack Problem

Classic problem from Sutton & Barto (Exercise 5.12).

A car races around a track:
- Control velocity in 2D (+1, 0, -1 for each dimension)
- Goal: Cross finish line as fast as possible
- Crash into wall -> Back to start

Great example of:
- Larger state space
- Continuous-like control (discretized velocity)
- Sparse rewards
"""

import numpy as np
from typing import Tuple, List, Dict, Set
from collections import defaultdict


class RaceTrack:
    """
    Racetrack environment.

    Track is a 2D grid:
    - '#' = Wall
    - '.' = Track
    - 'S' = Start line
    - 'F' = Finish line

    State: (row, col, vel_row, vel_col)
    Actions: 9 possible (dv_row, dv_col) in {-1, 0, +1}^2
    """

    def __init__(self, track_name: str = 'simple'):
        self.track = self._create_track(track_name)
        self.height = len(self.track)
        self.width = len(self.track[0])

        self.start_positions = self._find_positions('S')
        self.finish_positions = self._find_positions('F')

        # Velocity limits
        self.max_velocity = 4
        self.min_velocity = 0

        # Actions: (dv_row, dv_col) combinations
        self.actions = []
        for dv_r in [-1, 0, 1]:
            for dv_c in [-1, 0, 1]:
                self.actions.append((dv_r, dv_c))

        self.n_actions = len(self.actions)

    def _create_track(self, track_name: str) -> List[str]:
        """Create track layout."""
        if track_name == 'simple':
            # Simple L-shaped track
            return [
                "################FFFF",
                "###############....#",
                "##############.....#",
                "#############......#",
                "############.......#",
                "###########........#",
                "##########.........#",
                "#########..........#",
                "########...........#",
                "#######............#",
                "######.............#",
                "#####..............#",
                "####...............#",
                "###................#",
                "##.................#",
                "#..................#",
                "#..................#",
                "#..................#",
                "#..................#",
                "SSSSSS.............#",
            ]
        elif track_name == 'big':
            # Larger track
            return [
                "#########################FFFFF",
                "########################......#",
                "#######################.......#",
                "######################........#",
                "#####################.........#",
                "####################..........#",
                "###################...........#",
                "##################............#",
                "#################.............#",
                "################..............#",
                "###############...............#",
                "##############................#",
                "#############.................#",
                "############..................#",
                "###########...................#",
                "##########....................#",
                "#########.....................#",
                "########......................#",
                "#######.......................#",
                "######........................#",
                "#####.........................#",
                "####..........................#",
                "###...........................#",
                "##............................#",
                "#.............................#",
                "#.............................#",
                "#.............................#",
                "#.............................#",
                "#.............................#",
                "SSSSSSSSS.....................#",
            ]
        else:
            # Default simple
            return self._create_track('simple')

    def _find_positions(self, char: str) -> List[Tuple[int, int]]:
        """Find all positions of a character in track."""
        positions = []
        for r in range(self.height):
            for c in range(self.width):
                if self.track[r][c] == char:
                    positions.append((r, c))
        return positions

    def _is_on_track(self, row: int, col: int) -> bool:
        """Check if position is on track (not wall)."""
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        return self.track[row][col] != '#'

    def _crossed_finish(self, row: int, col: int, new_row: int, new_col: int) -> bool:
        """Check if movement crossed finish line."""
        # Simple check: if new position is at or past finish
        for f_row, f_col in self.finish_positions:
            if new_row <= f_row and new_col >= f_col:
                # Check if path intersects finish
                if self.track[f_row][f_col] == 'F':
                    return True

        # Also check if new position is finish
        if 0 <= new_row < self.height and 0 <= new_col < self.width:
            if self.track[new_row][new_col] == 'F':
                return True

        return False

    def reset(self) -> Tuple[int, int, int, int]:
        """Reset to random start position with zero velocity."""
        start = self.start_positions[np.random.randint(len(self.start_positions))]
        return (start[0], start[1], 0, 0)

    def step(self, state: Tuple, action_idx: int,
             noise: bool = True) -> Tuple[Tuple, float, bool]:
        """
        Take action, return (next_state, reward, done).

        With noise=True, there's 10% chance velocity change fails.
        """
        row, col, vel_r, vel_c = state
        dv_r, dv_c = self.actions[action_idx]

        # Apply noise (10% chance action fails)
        if noise and np.random.random() < 0.1:
            dv_r, dv_c = 0, 0

        # Update velocity
        new_vel_r = np.clip(vel_r + dv_r, self.min_velocity, self.max_velocity)
        new_vel_c = np.clip(vel_c + dv_c, self.min_velocity, self.max_velocity)

        # Ensure velocity is not (0, 0) unless at start
        if new_vel_r == 0 and new_vel_c == 0:
            new_vel_r = vel_r
            new_vel_c = vel_c

        # Update position
        new_row = row - new_vel_r  # Negative because row 0 is top
        new_col = col + new_vel_c

        # Check finish
        if self._crossed_finish(row, col, new_row, new_col):
            return (new_row, new_col, new_vel_r, new_vel_c), 0, True

        # Check collision (off track or wall)
        if not self._is_on_track(new_row, new_col):
            # Reset to start
            new_state = self.reset()
            return new_state, -1, False

        return (new_row, new_col, new_vel_r, new_vel_c), -1, False

    def render_track(self, path: List[Tuple] = None):
        """Render track with optional path."""
        display = [list(row) for row in self.track]

        if path:
            for i, (r, c, _, _) in enumerate(path):
                if 0 <= r < self.height and 0 <= c < self.width:
                    if display[r][c] not in ['S', 'F', '#']:
                        display[r][c] = '*'

        print("\nTrack:")
        for row in display:
            print(''.join(row))


def generate_episode(env: RaceTrack, Q: Dict, epsilon: float = 0.1,
                     max_steps: int = 500) -> List[Tuple]:
    """Generate episode using epsilon-greedy policy."""
    episode = []
    state = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(env.n_actions)
        else:
            q_values = [Q.get((state, a), 0.0) for a in range(env.n_actions)]
            action = np.argmax(q_values)

        next_state, reward, done = env.step(state, action)
        episode.append((state, action, reward))
        state = next_state
        steps += 1

    return episode


def mc_control_racetrack(env: RaceTrack, n_episodes: int = 50000,
                          gamma: float = 1.0, epsilon: float = 0.1,
                          verbose: bool = True) -> Tuple[Dict, List]:
    """
    On-policy MC Control for Racetrack.
    """
    Q = defaultdict(float)
    N = defaultdict(int)
    history = []

    episode_lengths = []

    for ep in range(n_episodes):
        # Decay epsilon
        eps = max(0.01, epsilon * (1 - ep / n_episodes))

        episode = generate_episode(env, Q, epsilon=eps)
        episode_lengths.append(len(episode))

        # First-visit MC update
        visited = set()
        G = 0

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            if (state, action) not in visited:
                visited.add((state, action))
                N[(state, action)] += 1
                alpha = 1.0 / N[(state, action)]
                Q[(state, action)] += alpha * (G - Q[(state, action)])

        if verbose and (ep + 1) % 10000 == 0:
            avg_len = np.mean(episode_lengths[-1000:])
            print(f"  Episode {ep + 1}: Avg length (last 1000) = {avg_len:.1f}")
            history.append((ep + 1, avg_len, eps))

    return dict(Q), history


def evaluate_policy(env: RaceTrack, Q: Dict, n_episodes: int = 100,
                    render: bool = False) -> Dict:
    """Evaluate learned policy."""
    lengths = []
    successes = 0

    for ep in range(n_episodes):
        state = env.reset()
        path = [state]
        done = False
        steps = 0

        while not done and steps < 500:
            q_values = [Q.get((state, a), 0.0) for a in range(env.n_actions)]
            action = np.argmax(q_values)
            state, _, done = env.step(state, action, noise=False)
            path.append(state)
            steps += 1

        lengths.append(steps)
        if done:
            successes += 1

        if render and ep == 0:
            env.render_track(path)

    return {
        'avg_length': np.mean(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'success_rate': successes / n_episodes,
    }


def main():
    # ============================================
    # 1. INTRODUCTION
    # ============================================
    print("=" * 60)
    print("RACETRACK PROBLEM")
    print("=" * 60)

    print("""
    Classic RL problem from Sutton & Barto.

    A car races around a track:
    - State: (row, col, velocity_row, velocity_col)
    - Actions: Change velocity by (-1, 0, +1) in each dimension
    - Reward: -1 per step (minimize time to finish)
    - Crash: Go back to start

    Challenges:
    - Large state space (~10,000+ states)
    - Sparse reward (only -1 per step)
    - Stochastic (10% chance action fails)
    """)

    # ============================================
    # 2. CREATE SIMPLE TRACK
    # ============================================
    print("\n" + "=" * 60)
    print("2. SIMPLE TRACK")
    print("=" * 60)

    env = RaceTrack(track_name='simple')
    print(f"\nTrack size: {env.height} x {env.width}")
    print(f"Start positions: {len(env.start_positions)}")
    print(f"Finish positions: {len(env.finish_positions)}")
    print(f"Actions: {env.n_actions}")
    print(f"Max velocity: {env.max_velocity}")

    env.render_track()

    # ============================================
    # 3. TRAIN MC CONTROL
    # ============================================
    print("\n" + "=" * 60)
    print("3. TRAIN MC CONTROL")
    print("=" * 60)

    print("\nTraining (50,000 episodes)...")
    Q, history = mc_control_racetrack(env, n_episodes=50000, epsilon=0.3)

    print(f"\nLearned Q-values for {len(Q)} state-action pairs")

    # ============================================
    # 4. EVALUATE POLICY
    # ============================================
    print("\n" + "=" * 60)
    print("4. EVALUATE LEARNED POLICY")
    print("=" * 60)

    results = evaluate_policy(env, Q, n_episodes=100, render=True)

    print(f"\nPolicy Performance (100 test episodes, no noise):")
    print(f"  Average steps: {results['avg_length']:.1f}")
    print(f"  Min steps: {results['min_length']}")
    print(f"  Max steps: {results['max_length']}")
    print(f"  Success rate: {results['success_rate'] * 100:.1f}%")

    # ============================================
    # 5. ANALYZE POLICY
    # ============================================
    print("\n" + "=" * 60)
    print("5. POLICY ANALYSIS")
    print("=" * 60)

    # Show Q-values for some states
    print("\nQ-values at start position (row=19, col=2, vel=0,0):")
    start_state = (19, 2, 0, 0)
    for a_idx, (dv_r, dv_c) in enumerate(env.actions):
        q = Q.get((start_state, a_idx), 0.0)
        print(f"  Action ({dv_r:+d}, {dv_c:+d}): Q = {q:.2f}")

    best_action = np.argmax([Q.get((start_state, a), 0.0) for a in range(env.n_actions)])
    print(f"\n  Best action: {env.actions[best_action]}")

    # Show typical trajectory
    print("\nSample optimal trajectory:")
    state = env.reset()
    print(f"  Start: pos=({state[0]}, {state[1]}), vel=({state[2]}, {state[3]})")

    for step in range(10):
        q_values = [Q.get((state, a), 0.0) for a in range(env.n_actions)]
        action = np.argmax(q_values)
        dv_r, dv_c = env.actions[action]
        state, reward, done = env.step(state, action, noise=False)
        print(f"  Step {step + 1}: action=({dv_r:+d},{dv_c:+d}) -> pos=({state[0]}, {state[1]}), vel=({state[2]}, {state[3]})")
        if done:
            print("  FINISHED!")
            break

    # ============================================
    # 6. LEARNING CURVE
    # ============================================
    print("\n" + "=" * 60)
    print("6. LEARNING CURVE")
    print("=" * 60)

    print("\nAverage episode length during training:")
    print("\n  Episodes  | Avg Length | Epsilon")
    print("  " + "-" * 38)
    for ep, avg_len, eps in history:
        print(f"   {ep:6d}  |   {avg_len:6.1f}   |  {eps:.3f}")

    # ============================================
    # 7. BIGGER TRACK
    # ============================================
    print("\n" + "=" * 60)
    print("7. BIGGER TRACK")
    print("=" * 60)

    env_big = RaceTrack(track_name='big')
    print(f"\nBig track size: {env_big.height} x {env_big.width}")

    print("\nTraining on big track (30,000 episodes)...")
    Q_big, _ = mc_control_racetrack(env_big, n_episodes=30000, epsilon=0.3, verbose=False)

    results_big = evaluate_policy(env_big, Q_big, n_episodes=50)
    print(f"\nBig Track Performance:")
    print(f"  Average steps: {results_big['avg_length']:.1f}")
    print(f"  Success rate: {results_big['success_rate'] * 100:.1f}%")

    # ============================================
    # 8. SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("8. SUMMARY")
    print("=" * 60)

    print("""
    Racetrack Problem Key Points:

    1. STATE SPACE
       - (row, col, vel_row, vel_col)
       - ~10,000+ states for simple track
       - Much larger for realistic tracks

    2. ACTION SPACE
       - 9 actions: velocity changes in {-1, 0, +1}^2
       - Velocity bounded [0, max_vel]

    3. CHALLENGES
       - Large state space -> Need many episodes
       - Sparse reward -> Hard credit assignment
       - Stochasticity -> Must learn robust policy

    4. MC SOLUTION
       - On-policy MC control with epsilon-greedy
       - Decaying epsilon helps convergence
       - ~50,000 episodes for reasonable policy

    5. OPTIMAL BEHAVIOR
       - Accelerate quickly at start
       - Maintain high velocity on straights
       - Slow down for turns
       - Avoid walls!

    This is a great example of MC handling:
    - Larger state spaces
    - Continuous-like control (discretized)
    - No model needed
    """)


if __name__ == "__main__":
    main()
