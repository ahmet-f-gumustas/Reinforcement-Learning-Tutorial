"""
04 - Gymnasium MDP Environments

This example explores real MDP environments from Gymnasium:
- FrozenLake: Stochastic grid world
- Taxi: Discrete pickup/dropoff task
- CliffWalking: Risk vs reward tradeoff
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple


def explore_frozen_lake():
    """
    FrozenLake-v1: A classic stochastic grid world.

    Grid Layout (4x4):
    SFFF    S = Start
    FHFH    F = Frozen (safe)
    FFFH    H = Hole (fall, episode ends)
    HFFG    G = Goal (reward = 1)

    Actions: 0=Left, 1=Down, 2=Right, 3=Up
    Stochastic: 1/3 intended, 1/3 perpendicular each side
    """
    print("=" * 60)
    print("FROZENLAKE-V1: STOCHASTIC GRID WORLD")
    print("=" * 60)

    # Create environment
    env = gym.make('FrozenLake-v1', render_mode=None, is_slippery=True)

    print(f"""
    Environment: FrozenLake-v1

    Grid (4x4):
    S F F F     S = Start (state 0)
    F H F H     F = Frozen (safe)
    F F F H     H = Hole (terminal, reward=0)
    H F F G     G = Goal (terminal, reward=1)

    State space: {env.observation_space.n} states (0-15)
    Action space: {env.action_space.n} actions
    Actions: 0=Left, 1=Down, 2=Right, 3=Up

    Key property: STOCHASTIC transitions!
    - 1/3 chance: move in intended direction
    - 1/3 chance: move perpendicular (left)
    - 1/3 chance: move perpendicular (right)
    """)

    # Explore transition probabilities
    print("Transition Probabilities P(s'|s,a):")
    print("-" * 50)

    # State 0 (Start), Action 2 (Right)
    state = 0
    action = 2  # Right
    print(f"\nFrom state {state}, taking action Right:")
    for prob, next_state, reward, done in env.unwrapped.P[state][action]:
        print(f"  P(s'={next_state}) = {prob:.4f}, R={reward}, Done={done}")

    # State 6, all actions
    state = 6
    print(f"\nFrom state {state} (middle of grid):")
    action_names = ['Left', 'Down', 'Right', 'Up']
    for action in range(4):
        print(f"\n  Action: {action_names[action]}")
        for prob, next_state, reward, done in env.unwrapped.P[state][action]:
            status = "HOLE!" if done and reward == 0 else ("GOAL!" if done else "")
            print(f"    P(s'={next_state:2d}) = {prob:.4f} {status}")

    # Run a few episodes with random policy
    print("\n" + "-" * 50)
    print("Running 1000 episodes with random policy...")

    successes = 0
    total_steps = []

    for episode in range(1000):
        state, _ = env.reset()
        steps = 0
        done = False

        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            state = next_state

            if steps > 100:
                break

        if reward == 1.0:
            successes += 1
            total_steps.append(steps)

    print(f"\n  Success rate: {successes / 1000 * 100:.1f}%")
    if total_steps:
        print(f"  Avg steps to goal: {np.mean(total_steps):.1f}")

    print("\n  Note: Random policy performs poorly due to stochasticity!")

    env.close()
    return env


def explore_taxi():
    """
    Taxi-v3: Discrete navigation with pickup/dropoff.

    5x5 grid with 4 special locations (R, G, Y, B)
    Task: Pick up passenger, drop off at destination
    """
    print("\n" + "=" * 60)
    print("TAXI-V3: PICKUP AND DROPOFF")
    print("=" * 60)

    env = gym.make('Taxi-v3', render_mode=None)

    print(f"""
    Environment: Taxi-v3

    Grid (5x5):
    +---------+
    |R: | : :G|    R, G, Y, B = Special locations
    | : | : : |    | = Walls
    | : : : : |    : = Passable
    | | : | : |
    |Y| : |B: |
    +---------+

    State space: {env.observation_space.n} states
    (25 taxi positions x 5 passenger locations x 4 destinations)

    Action space: {env.action_space.n} actions
    0 = South, 1 = North, 2 = East, 3 = West
    4 = Pickup, 5 = Dropoff

    Rewards:
    - Each step: -1
    - Illegal pickup/dropoff: -10
    - Successful dropoff: +20
    """)

    # Decode a state
    print("State Encoding Example:")
    print("-" * 50)

    state = 328  # Example state
    taxi_row = state // 100
    remainder = state % 100
    taxi_col = remainder // 20
    remainder = remainder % 20
    pass_loc = remainder // 4
    dest_idx = remainder % 4

    locations = ['R (0,0)', 'G (0,4)', 'Y (4,0)', 'B (4,3)']
    print(f"\n  State {state} decodes to:")
    print(f"    Taxi position: ({taxi_row}, {taxi_col})")
    print(f"    Passenger: {locations[pass_loc] if pass_loc < 4 else 'In taxi'}")
    print(f"    Destination: {locations[dest_idx]}")

    # Explore transitions
    print("\nTransition Probabilities (Deterministic):")
    print("-" * 50)

    state = 328
    print(f"\nFrom state {state}:")
    action_names = ['South', 'North', 'East', 'West', 'Pickup', 'Dropoff']
    for action in range(6):
        for prob, next_state, reward, done in env.unwrapped.P[state][action]:
            status = "(SUCCESS!)" if done and reward > 0 else ""
            print(f"  {action_names[action]:8s}: s'={next_state}, R={reward:3d} {status}")

    # Run episodes
    print("\n" + "-" * 50)
    print("Running 100 episodes with random policy...")

    total_rewards = []
    for episode in range(100):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        total_rewards.append(total_reward)

    print(f"\n  Avg reward: {np.mean(total_rewards):.1f}")
    print(f"  Note: Negative because random policy makes many mistakes!")

    env.close()
    return env


def explore_cliff_walking():
    """
    CliffWalking-v0: Risk vs reward tradeoff.

    4x12 grid with cliff along bottom edge.
    Shortest path is risky (along cliff), safer path is longer.
    """
    print("\n" + "=" * 60)
    print("CLIFFWALKING-V0: RISK VS REWARD")
    print("=" * 60)

    env = gym.make('CliffWalking-v0', render_mode=None)

    print(f"""
    Environment: CliffWalking-v0

    Grid (4x12):
    . . . . . . . . . . . .
    . . . . . . . . . . . .
    . . . . . . . . . . . .
    S C C C C C C C C C C G

    S = Start (state 36)
    G = Goal (state 47)
    C = Cliff (falling = -100, back to start)
    . = Normal cell (-1 per step)

    State space: {env.observation_space.n} states
    Action space: {env.action_space.n} actions
    Actions: 0=Up, 1=Right, 2=Down, 3=Left

    Optimal policy dilemma:
    - Shortest path: Along cliff edge (risky)
    - Safest path: Go up, across, down (longer)
    """)

    # Show state layout
    print("State Layout:")
    print("-" * 50)
    print("\n    ", end="")
    for col in range(12):
        print(f"{col:4d}", end="")
    print()

    for row in range(4):
        print(f"  {row} ", end="")
        for col in range(12):
            state = row * 12 + col
            if row == 3 and 1 <= col <= 10:
                print("   C", end="")  # Cliff
            elif state == 36:
                print("   S", end="")  # Start
            elif state == 47:
                print("   G", end="")  # Goal
            else:
                print(f"{state:4d}", end="")
        print()

    # Transition examples
    print("\nTransitions (Deterministic):")
    print("-" * 50)

    action_names = ['Up', 'Right', 'Down', 'Left']

    # Near cliff
    state = 25  # One row above cliff
    print(f"\nFrom state {state} (above cliff):")
    for action in range(4):
        for prob, next_state, reward, done in env.unwrapped.P[state][action]:
            cliff = "(CLIFF!)" if reward == -100 else ""
            goal = "(GOAL!)" if next_state == 47 else ""
            print(f"  {action_names[action]:6s}: s'={next_state:2d}, R={reward:4d} {cliff}{goal}")

    # Run episodes
    print("\n" + "-" * 50)
    print("Running 100 episodes with random policy...")

    total_rewards = []
    cliff_falls = 0

    for episode in range(100):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 1000:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if reward == -100:
                cliff_falls += 1

            state = next_state
            steps += 1

        total_rewards.append(total_reward)

    print(f"\n  Avg reward: {np.mean(total_rewards):.1f}")
    print(f"  Cliff falls: {cliff_falls}")
    print("  Note: Random policy falls off cliff frequently!")

    env.close()
    return env


def compare_deterministic_stochastic():
    """Compare deterministic vs stochastic FrozenLake."""
    print("\n" + "=" * 60)
    print("COMPARISON: DETERMINISTIC VS STOCHASTIC")
    print("=" * 60)

    print("\nFrozenLake with is_slippery=False (Deterministic):")
    print("-" * 50)

    env_det = gym.make('FrozenLake-v1', is_slippery=False)

    state = 6
    action = 2  # Right
    print(f"\nFrom state {state}, action Right:")
    for prob, next_state, reward, done in env_det.unwrapped.P[state][action]:
        print(f"  P(s'={next_state}) = {prob:.4f}")

    print("\nFrozenLake with is_slippery=True (Stochastic):")
    print("-" * 50)

    env_stoch = gym.make('FrozenLake-v1', is_slippery=True)

    print(f"\nFrom state {state}, action Right:")
    for prob, next_state, reward, done in env_stoch.unwrapped.P[state][action]:
        print(f"  P(s'={next_state}) = {prob:.4f}")

    # Success rate comparison
    print("\n" + "-" * 50)
    print("Success rates with random policy (1000 episodes):")

    for name, env in [("Deterministic", env_det), ("Stochastic", env_stoch)]:
        successes = 0
        for _ in range(1000):
            state, _ = env.reset()
            done = False
            for _ in range(100):
                action = env.action_space.sample()
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if done:
                    if reward == 1.0:
                        successes += 1
                    break

        print(f"  {name:15s}: {successes / 10:.1f}%")

    env_det.close()
    env_stoch.close()


def extract_mdp_model(env) -> Dict:
    """
    Extract full MDP model from Gymnasium environment.

    Returns dict with:
    - n_states: Number of states
    - n_actions: Number of actions
    - P: Transition probabilities P[s][a] = [(prob, s', r, done), ...]
    - terminal_states: Set of terminal states
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Find terminal states
    terminal_states = set()
    for s in range(n_states):
        for a in range(n_actions):
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                if done:
                    terminal_states.add(next_state)

    return {
        'n_states': n_states,
        'n_actions': n_actions,
        'P': env.unwrapped.P,
        'terminal_states': terminal_states
    }


def main():
    print("=" * 60)
    print("GYMNASIUM MDP ENVIRONMENTS")
    print("=" * 60)

    print("""
    This example explores three classic MDP environments from Gymnasium:

    1. FrozenLake: Stochastic grid world (slippery ice)
    2. Taxi: Discrete pickup/dropoff task
    3. CliffWalking: Risk vs reward tradeoff

    All these environments expose the MDP model through env.P[state][action],
    which returns: [(probability, next_state, reward, done), ...]
    """)

    # Explore each environment
    explore_frozen_lake()
    explore_taxi()
    explore_cliff_walking()
    compare_deterministic_stochastic()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("""
    Key Takeaways:

    1. GYMNASIUM MDP ACCESS
       - env.observation_space.n: Number of states
       - env.action_space.n: Number of actions
       - env.P[s][a]: Transition model [(prob, s', r, done), ...]

    2. TYPES OF ENVIRONMENTS
       - Deterministic: P(s'|s,a) = 1 for one state
       - Stochastic: Multiple possible outcomes

    3. REWARD STRUCTURES
       - FrozenLake: Sparse (+1 at goal only)
       - Taxi: Dense (-1 per step, +20 success, -10 illegal)
       - CliffWalking: Risk penalty (-100 cliff)

    4. RANDOM POLICY PERFORMANCE
       - Usually very poor
       - Shows need for learning/planning algorithms

    Next: We'll solve these MDPs using Dynamic Programming!
    """)


if __name__ == "__main__":
    main()
