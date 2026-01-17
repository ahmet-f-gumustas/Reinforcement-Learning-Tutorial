"""
02 - MDP Components: Building a Grid World

This example implements a complete MDP (Grid World) demonstrating:
- State Space (S)
- Action Space (A)
- Transition Probabilities (P)
- Reward Function (R)
- Discount Factor (gamma)
"""

import numpy as np
from typing import Tuple, Dict, List


class GridWorldMDP:
    """
    A simple Grid World MDP implementation.

    Grid Layout (4x4):
    +---+---+---+---+
    | 0 | 1 | 2 | 3 |  <- Goal at position 3
    +---+---+---+---+
    | 4 | 5 | 6 | 7 |
    +---+---+---+---+
    | 8 | 9 |10 |11 |
    +---+---+---+---+
    |12 |13 |14 |15 |  <- Start at position 12
    +---+---+---+---+

    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    """

    def __init__(self, size: int = 4, gamma: float = 0.9):
        self.size = size
        self.n_states = size * size
        self.gamma = gamma

        # State Space
        self.states = list(range(self.n_states))

        # Action Space
        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.action_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

        # Special states
        self.start_state = self.n_states - self.size  # Bottom-left (12)
        self.goal_state = self.size - 1  # Top-right (3)
        self.obstacles = [5]  # Obstacle at position 5

        # Build transition and reward models
        self.transition_probs = self._build_transition_probs()
        self.rewards = self._build_rewards()

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        """Convert state index to (row, col) coordinates."""
        return state // self.size, state % self.size

    def _coord_to_state(self, row: int, col: int) -> int:
        """Convert (row, col) coordinates to state index."""
        return row * self.size + col

    def _build_transition_probs(self) -> Dict:
        """
        Build transition probability matrix.
        P[s][a] = [(prob, next_state), ...]

        For deterministic environment: prob = 1.0
        """
        P = {}

        # Action effects: (row_delta, col_delta)
        action_effects = {
            0: (-1, 0),   # Up
            1: (0, 1),    # Right
            2: (1, 0),    # Down
            3: (0, -1),   # Left
        }

        for s in self.states:
            P[s] = {}
            row, col = self._state_to_coord(s)

            # Terminal state (goal) - stays in place
            if s == self.goal_state:
                for a in self.actions:
                    P[s][a] = [(1.0, s)]
                continue

            # Obstacle - cannot be entered
            if s in self.obstacles:
                for a in self.actions:
                    P[s][a] = [(1.0, s)]
                continue

            for a in self.actions:
                dr, dc = action_effects[a]
                new_row = row + dr
                new_col = col + dc

                # Check boundaries
                if 0 <= new_row < self.size and 0 <= new_col < self.size:
                    new_state = self._coord_to_state(new_row, new_col)
                    # Check if new state is obstacle
                    if new_state in self.obstacles:
                        new_state = s  # Stay in place
                else:
                    new_state = s  # Stay in place if hitting wall

                # Deterministic transitions (prob = 1.0)
                P[s][a] = [(1.0, new_state)]

        return P

    def _build_rewards(self) -> Dict:
        """
        Build reward function.
        R[s][a][s'] = reward for transition (s, a, s')
        """
        R = {}

        for s in self.states:
            R[s] = {}
            for a in self.actions:
                R[s][a] = {}
                for prob, s_prime in self.transition_probs[s][a]:
                    if s_prime == self.goal_state and s != self.goal_state:
                        R[s][a][s_prime] = 1.0  # Reward for reaching goal
                    else:
                        R[s][a][s_prime] = -0.01  # Small negative reward (encourages shorter paths)

        return R

    def get_reward(self, state: int, action: int, next_state: int) -> float:
        """Get reward for a transition."""
        return self.rewards[state][action].get(next_state, 0.0)

    def get_transitions(self, state: int, action: int) -> List[Tuple[float, int]]:
        """Get list of (probability, next_state) for state-action pair."""
        return self.transition_probs[state][action]

    def is_terminal(self, state: int) -> bool:
        """Check if state is terminal."""
        return state == self.goal_state

    def render(self, values: np.ndarray = None, policy: np.ndarray = None):
        """Render the grid world."""
        print("\nGrid World:")
        print("+" + "------+" * self.size)

        for row in range(self.size):
            line = "|"
            for col in range(self.size):
                state = self._coord_to_state(row, col)

                if state == self.goal_state:
                    cell = "  G  "
                elif state == self.start_state:
                    cell = "  S  "
                elif state in self.obstacles:
                    cell = "  X  "
                elif values is not None:
                    cell = f"{values[state]:5.2f}"
                elif policy is not None:
                    arrows = ["↑", "→", "↓", "←"]
                    cell = f"  {arrows[policy[state]]}  "
                else:
                    cell = f" {state:2d}  "

                line += cell + "|"

            print(line)
            print("+" + "------+" * self.size)


def main():
    # ============================================
    # 1. CREATE THE MDP
    # ============================================
    print("=" * 60)
    print("1. CREATE THE MDP")
    print("=" * 60)

    mdp = GridWorldMDP(size=4, gamma=0.9)

    print(f"\nGrid World MDP (4x4)")
    print(f"Number of states: {mdp.n_states}")
    print(f"Number of actions: {len(mdp.actions)}")
    print(f"Discount factor (gamma): {mdp.gamma}")
    print(f"Start state: {mdp.start_state}")
    print(f"Goal state: {mdp.goal_state}")
    print(f"Obstacles: {mdp.obstacles}")

    mdp.render()

    # ============================================
    # 2. STATE SPACE
    # ============================================
    print("\n" + "=" * 60)
    print("2. STATE SPACE (S)")
    print("=" * 60)

    print(f"\nStates: {mdp.states}")
    print(f"\nState representation:")
    for s in [0, 5, 12, 3]:
        row, col = mdp._state_to_coord(s)
        state_type = ""
        if s == mdp.goal_state:
            state_type = "(Goal)"
        elif s == mdp.start_state:
            state_type = "(Start)"
        elif s in mdp.obstacles:
            state_type = "(Obstacle)"
        print(f"  State {s}: row={row}, col={col} {state_type}")

    # ============================================
    # 3. ACTION SPACE
    # ============================================
    print("\n" + "=" * 60)
    print("3. ACTION SPACE (A)")
    print("=" * 60)

    print(f"\nActions: {mdp.actions}")
    print("\nAction meanings:")
    for a, name in mdp.action_names.items():
        print(f"  Action {a}: {name}")

    # ============================================
    # 4. TRANSITION PROBABILITIES
    # ============================================
    print("\n" + "=" * 60)
    print("4. TRANSITION PROBABILITIES P(s'|s,a)")
    print("=" * 60)

    print("\nExample transitions from state 6 (middle of grid):")
    example_state = 6
    for a in mdp.actions:
        transitions = mdp.get_transitions(example_state, a)
        for prob, next_state in transitions:
            print(f"  P(s'={next_state} | s={example_state}, a={mdp.action_names[a]}) = {prob}")

    print("\nTransitions from state 12 (start):")
    example_state = 12
    for a in mdp.actions:
        transitions = mdp.get_transitions(example_state, a)
        for prob, next_state in transitions:
            print(f"  P(s'={next_state} | s={example_state}, a={mdp.action_names[a]}) = {prob}")

    print("\nNote: Hitting walls or obstacles keeps agent in same state.")

    # ============================================
    # 5. REWARD FUNCTION
    # ============================================
    print("\n" + "=" * 60)
    print("5. REWARD FUNCTION R(s,a,s')")
    print("=" * 60)

    print("\nReward structure:")
    print("  - Reaching goal state: +1.0")
    print("  - All other transitions: -0.01 (encourages shorter paths)")

    print("\nExample rewards:")

    # Transition to goal
    state_2 = 2
    action_right = 1
    transitions = mdp.get_transitions(state_2, action_right)
    for prob, next_state in transitions:
        reward = mdp.get_reward(state_2, action_right, next_state)
        print(f"  R(s=2, a=Right, s'=3) = {reward} (reaching goal!)")

    # Normal transition
    state_6 = 6
    action_up = 0
    transitions = mdp.get_transitions(state_6, action_up)
    for prob, next_state in transitions:
        reward = mdp.get_reward(state_6, action_up, next_state)
        print(f"  R(s=6, a=Up, s'={next_state}) = {reward}")

    # ============================================
    # 6. DISCOUNT FACTOR
    # ============================================
    print("\n" + "=" * 60)
    print("6. DISCOUNT FACTOR (gamma)")
    print("=" * 60)

    print(f"\nCurrent gamma: {mdp.gamma}")

    print("\nEffect of gamma on return calculation:")
    rewards = [0, 0, 0, 1]  # Example: 3 steps then goal
    print(f"Rewards sequence: {rewards}")

    for gamma in [0.0, 0.5, 0.9, 1.0]:
        G = sum(gamma**t * r for t, r in enumerate(rewards))
        print(f"  gamma={gamma}: G = {' + '.join([f'{gamma}^{t}*{r}' for t, r in enumerate(rewards)])} = {G:.3f}")

    # ============================================
    # 7. SIMULATING AN EPISODE
    # ============================================
    print("\n" + "=" * 60)
    print("7. SIMULATING AN EPISODE")
    print("=" * 60)

    np.random.seed(42)

    state = mdp.start_state
    total_reward = 0
    trajectory = [(state, None, None)]

    print(f"\nStarting from state {state}")

    max_steps = 50
    for step in range(max_steps):
        # Random policy
        action = np.random.choice(mdp.actions)

        # Get transition
        transitions = mdp.get_transitions(state, action)
        probs = [t[0] for t in transitions]
        next_states = [t[1] for t in transitions]
        next_state = np.random.choice(next_states, p=probs)

        # Get reward
        reward = mdp.get_reward(state, action, next_state)
        total_reward += reward

        print(f"  Step {step + 1}: State {state} --({mdp.action_names[action]})--> State {next_state}, Reward: {reward:.2f}")

        trajectory.append((next_state, action, reward))
        state = next_state

        if mdp.is_terminal(state):
            print(f"\n  Reached goal in {step + 1} steps!")
            break

    print(f"\nTotal reward: {total_reward:.2f}")

    # ============================================
    # 8. MDP TUPLE SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("8. MDP TUPLE SUMMARY")
    print("=" * 60)

    print("""
    This Grid World MDP is defined by:

    MDP = (S, A, P, R, gamma) where:

    S = {0, 1, 2, ..., 15}  (16 states in 4x4 grid)
    A = {Up, Right, Down, Left}
    P = Deterministic transitions (see transition matrix)
    R = +1 for reaching goal, -0.01 otherwise
    gamma = 0.9

    Key properties:
    - Finite state space
    - Finite action space
    - Known transition dynamics (model-based)
    - Satisfies Markov property
    """)

    mdp.render()


if __name__ == "__main__":
    main()
