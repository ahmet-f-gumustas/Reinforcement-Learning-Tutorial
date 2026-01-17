"""
01 - Understanding the Markov Property

This example demonstrates the Markov property:
"The future is independent of the past given the present."

We'll show examples where Markov property holds and where it doesn't.
"""

import numpy as np


def main():
    # ============================================
    # 1. WHAT IS THE MARKOV PROPERTY?
    # ============================================
    print("=" * 60)
    print("1. WHAT IS THE MARKOV PROPERTY?")
    print("=" * 60)

    print("""
    The Markov Property states that:

    P(S_{t+1} | S_t) = P(S_{t+1} | S_1, S_2, ..., S_t)

    In simple terms: The future depends only on the present state,
    not on the history of how we got there.
    """)

    # ============================================
    # 2. EXAMPLE: WEATHER (MARKOV)
    # ============================================
    print("\n" + "=" * 60)
    print("2. EXAMPLE: SIMPLE WEATHER MODEL (MARKOV)")
    print("=" * 60)

    # States: Sunny (0), Rainy (1)
    # Transition matrix: P[i,j] = P(next_state=j | current_state=i)
    weather_transition = np.array([
        [0.8, 0.2],  # Sunny -> [Sunny, Rainy]
        [0.4, 0.6],  # Rainy -> [Sunny, Rainy]
    ])

    states = ["Sunny", "Rainy"]

    print("\nWeather Transition Probabilities:")
    print("-" * 40)
    for i, state in enumerate(states):
        for j, next_state in enumerate(states):
            print(f"  P({next_state} | {state}) = {weather_transition[i, j]}")

    # Simulate weather
    print("\nSimulating 10 days of weather:")
    print("-" * 40)

    np.random.seed(42)
    current_state = 0  # Start with Sunny

    history = [current_state]
    for day in range(10):
        # Next state depends ONLY on current state (Markov property)
        probs = weather_transition[current_state]
        next_state = np.random.choice([0, 1], p=probs)

        print(f"  Day {day + 1}: {states[current_state]} -> {states[next_state]}")
        current_state = next_state
        history.append(current_state)

    print("\n  Key insight: Tomorrow's weather depends only on today,")
    print("  not on the weather from previous days.")

    # ============================================
    # 3. EXAMPLE: ROBOT POSITION (MARKOV)
    # ============================================
    print("\n" + "=" * 60)
    print("3. EXAMPLE: ROBOT ON A LINE (MARKOV)")
    print("=" * 60)

    print("""
    A robot moves on a 1D line with positions 0, 1, 2, 3, 4.
    Actions: Move Left (-1), Stay (0), Move Right (+1)

    The Markov property holds because:
    - Next position depends only on current position and action
    - History of past positions doesn't matter
    """)

    def robot_step(position, action, max_pos=4):
        """Deterministic robot movement."""
        new_position = position + action
        new_position = max(0, min(max_pos, new_position))
        return new_position

    # Simulate robot movement
    print("\nSimulating robot movement:")
    print("-" * 40)

    position = 2  # Start at position 2
    actions = [1, 1, 0, -1, -1, -1, 1]  # Sequence of actions
    action_names = {-1: "Left", 0: "Stay", 1: "Right"}

    print(f"  Start position: {position}")
    for i, action in enumerate(actions):
        new_position = robot_step(position, action)
        print(f"  Step {i + 1}: Position {position} + {action_names[action]} -> Position {new_position}")
        position = new_position

    # ============================================
    # 4. COUNTER-EXAMPLE: CARD GAME (NOT MARKOV)
    # ============================================
    print("\n" + "=" * 60)
    print("4. COUNTER-EXAMPLE: CARD GAME (NOT MARKOV)")
    print("=" * 60)

    print("""
    Consider a card game where you draw cards from a deck.

    State: The card you currently see
    Problem: The probability of drawing an Ace depends on
             how many Aces have already been drawn!

    This violates the Markov property because the future
    depends on the history of past draws.
    """)

    # Simulate to show non-Markov behavior
    np.random.seed(42)

    # Full deck: 4 Aces, 48 other cards
    deck = ["Ace"] * 4 + ["Other"] * 48
    np.random.shuffle(deck)

    print("\nDrawing cards from a deck:")
    print("-" * 40)

    drawn = []
    for i in range(10):
        card = deck[i]
        drawn.append(card)
        aces_drawn = drawn.count("Ace")
        aces_remaining = 4 - aces_drawn
        cards_remaining = 52 - len(drawn)

        if cards_remaining > 0:
            prob_ace = aces_remaining / cards_remaining
        else:
            prob_ace = 0

        print(f"  Draw {i + 1}: {card}")
        print(f"    P(next=Ace) = {aces_remaining}/{cards_remaining} = {prob_ace:.3f}")

    print("\n  Key insight: The probability changes based on history!")
    print("  This is NOT Markov unless we include the full deck state.")

    # ============================================
    # 5. MAKING NON-MARKOV PROCESSES MARKOV
    # ============================================
    print("\n" + "=" * 60)
    print("5. MAKING NON-MARKOV PROCESSES MARKOV")
    print("=" * 60)

    print("""
    We can often convert non-Markov processes to Markov by
    expanding the state representation.

    Card game solution:
    - Instead of state = current card
    - Use state = (current card, cards remaining in deck)

    Now the future depends only on this expanded state!

    This is called "state augmentation" and is a common technique.
    """)

    # ============================================
    # 6. GYMNASIUM ENVIRONMENTS AND MARKOV PROPERTY
    # ============================================
    print("\n" + "=" * 60)
    print("6. GYMNASIUM ENVIRONMENTS AND MARKOV PROPERTY")
    print("=" * 60)

    print("""
    Most Gymnasium environments are designed to be Markov:

    CartPole-v1:
    - State: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    - All information needed is in the current state
    - MARKOV: Yes

    Blackjack-v1:
    - State: (player_sum, dealer_showing, usable_ace)
    - Doesn't track which specific cards were dealt
    - But includes enough info for optimal play
    - MARKOV: Yes (by design)

    Atari Games (with frame stacking):
    - Single frame might not be Markov (can't see velocity)
    - Stack 4 frames to capture motion -> becomes Markov
    - MARKOV: Yes (with frame stacking)
    """)

    # ============================================
    # 7. WHY MARKOV PROPERTY MATTERS FOR RL
    # ============================================
    print("\n" + "=" * 60)
    print("7. WHY MARKOV PROPERTY MATTERS FOR RL")
    print("=" * 60)

    print("""
    The Markov property is crucial for RL because:

    1. SIMPLICITY
       - We only need to store the current state
       - No need to remember entire history

    2. BELLMAN EQUATIONS
       - V(s) can be computed recursively
       - Forms the basis of dynamic programming

    3. CONVERGENCE GUARANTEES
       - Many RL algorithms proven to converge for MDPs
       - Q-learning, SARSA, Policy Gradient, etc.

    4. COMPUTATIONAL EFFICIENCY
       - State space is manageable
       - Algorithms can be efficient

    If your problem is not Markov:
    - Augment the state with relevant history
    - Use recurrent neural networks (LSTM, GRU)
    - Use transformer architectures
    """)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    - Markov Property: Future depends only on present state
    - Most RL problems can be formulated as Markov
    - Non-Markov problems can be converted by state augmentation
    - This property enables efficient RL algorithms
    """)


if __name__ == "__main__":
    main()
