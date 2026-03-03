"""
07 - Course Summary: 12 Weeks of Reinforcement Learning

Complete summary of the entire RL curriculum with algorithm
taxonomy, key concepts, and future directions.

Demonstrates:
- Full algorithm taxonomy
- Evolution of RL methods
- Performance comparison overview
- Recommended next steps
"""

import numpy as np
import matplotlib.pyplot as plt


def print_course_map():
    """Print the complete course roadmap."""
    print("=" * 60)
    print("12-WEEK REINFORCEMENT LEARNING CURRICULUM")
    print("=" * 60)

    print("""
    ┌──────────────────────────────────────────────────────┐
    │                  FOUNDATIONS                          │
    │                                                      │
    │  Week 1: Introduction                                │
    │     Agent, Environment, State, Action, Reward        │
    │                    ↓                                 │
    │  Week 2: Markov Decision Processes                   │
    │     Markov property, Bellman equations                │
    │                    ↓                                 │
    │  Week 3: Dynamic Programming                         │
    │     Value Iteration, Policy Iteration                │
    ├──────────────────────────────────────────────────────┤
    │              TABULAR METHODS                         │
    │                                                      │
    │  Week 4: Monte Carlo                                 │
    │     Episode-based learning, MC prediction/control    │
    │                    ↓                                 │
    │  Week 5: Temporal Difference                         │
    │     TD(0), n-step TD, TD vs MC comparison            │
    │                    ↓                                 │
    │  Week 6: Q-Learning & SARSA                          │
    │     On-policy vs Off-policy, tabular Q-learning      │
    ├──────────────────────────────────────────────────────┤
    │             DEEP RL (VALUE-BASED)                    │
    │                                                      │
    │  Week 7: Function Approximation                      │
    │     Linear approximation, feature engineering        │
    │                    ↓                                 │
    │  Week 8: DQN                                         │
    │     Experience replay, target network, Double DQN    │
    ├──────────────────────────────────────────────────────┤
    │             DEEP RL (POLICY-BASED)                   │
    │                                                      │
    │  Week 9: Policy Gradient                             │
    │     REINFORCE, baseline, variance reduction          │
    │                    ↓                                 │
    │  Week 10: Actor-Critic                               │
    │     A2C, A3C, GAE                                    │
    │                    ↓                                 │
    │  Week 11: PPO                                        │
    │     TRPO, PPO-Clip, PPO-Penalty                      │
    ├──────────────────────────────────────────────────────┤
    │              ADVANCED TOPICS                          │
    │                                                      │
    │  Week 12: Advanced                                   │
    │     Multi-agent, Model-based, Inverse RL,            │
    │     Reward Shaping, Curriculum Learning               │
    └──────────────────────────────────────────────────────┘
    """)


def print_algorithm_taxonomy():
    """Print the complete algorithm taxonomy."""
    print("=" * 60)
    print("ALGORITHM TAXONOMY")
    print("=" * 60)

    print("""
    Reinforcement Learning
    │
    ├── Model-Free
    │   ├── Value-Based
    │   │   ├── Tabular
    │   │   │   ├── MC Prediction/Control (Week 4)
    │   │   │   ├── TD(0), TD(λ) (Week 5)
    │   │   │   ├── Q-Learning (Week 6)
    │   │   │   └── SARSA (Week 6)
    │   │   └── Deep
    │   │       ├── DQN (Week 8)
    │   │       ├── Double DQN
    │   │       ├── Dueling DQN
    │   │       └── Rainbow
    │   │
    │   ├── Policy-Based
    │   │   ├── REINFORCE (Week 9)
    │   │   └── Natural Policy Gradient
    │   │
    │   └── Actor-Critic (Combined)
    │       ├── A2C / A3C (Week 10)
    │       ├── PPO (Week 11)
    │       ├── TRPO
    │       ├── SAC (Soft Actor-Critic)
    │       ├── DDPG
    │       └── TD3
    │
    └── Model-Based (Week 12)
        ├── Dyna-Q
        ├── MBPO
        ├── Dreamer
        └── MuZero
    """)


def print_key_equations():
    """Print the most important equations from the course."""
    print("=" * 60)
    print("KEY EQUATIONS")
    print("=" * 60)

    print("""
    Bellman Equation (Week 2):
        V(s) = Σ_a π(a|s) Σ_{s'} T(s'|s,a) [R(s,a,s') + γ V(s')]

    TD Error (Week 5):
        δ_t = r_t + γ V(s_{t+1}) - V(s_t)

    Q-Learning Update (Week 6):
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]

    Policy Gradient Theorem (Week 9):
        ∇J(θ) = E[Σ_t ∇log π_θ(a_t|s_t) × A_t]

    GAE (Week 10):
        A^GAE_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

    PPO-Clip (Week 11):
        L = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)]
    """)


def print_practical_guide():
    """Print practical algorithm selection guide."""
    print("=" * 60)
    print("PRACTICAL ALGORITHM SELECTION GUIDE")
    print("=" * 60)

    print("""
    Q: "I have a new RL problem. What algorithm should I use?"

    Start here:
    ┌─────────────────────────────────────────────────────┐
    │  Is the action space discrete or continuous?         │
    │                                                      │
    │  DISCRETE:                                           │
    │    PPO (first try) → DQN (if sample efficiency      │
    │    matters) → Rainbow DQN (if you need best perf)    │
    │                                                      │
    │  CONTINUOUS:                                          │
    │    PPO (first try) → SAC (if off-policy needed)     │
    │    → TD3 (DDPG successor)                            │
    │                                                      │
    │  MULTI-AGENT:                                        │
    │    MAPPO → QMIX → MADDPG                            │
    │                                                      │
    │  SAMPLE-LIMITED:                                     │
    │    Model-based (Dreamer, MBPO) → DQN (replay)       │
    │                                                      │
    │  IMITATION:                                          │
    │    Behavioral Cloning → GAIL → IRL                  │
    └─────────────────────────────────────────────────────┘

    Default recommendation for any new problem: PPO
    - Works on discrete and continuous
    - Robust to hyperparameters
    - Simple to implement
    - Good baseline performance
    """)


def print_future_directions():
    """Print recommended next learning steps."""
    print("=" * 60)
    print("NEXT STEPS AND FUTURE DIRECTIONS")
    print("=" * 60)

    print("""
    After completing this course, explore:

    1. OFFLINE RL
       Learn from fixed datasets (no environment interaction)
       Algorithms: CQL, IQL, Decision Transformer

    2. GOAL-CONDITIONED RL
       Multi-goal policies, Hindsight Experience Replay (HER)
       Universal value functions

    3. HIERARCHICAL RL
       Options framework, skill discovery
       HAM, MAXQ, Feudal Networks

    4. META-RL
       Learning to learn, few-shot adaptation
       MAML, RL², Learn2Learn

    5. SAFE RL
       Constrained optimization, risk-sensitive policies
       CPO, RCPO, safe exploration

    6. REAL-WORLD RL
       Sim-to-real transfer, domain randomization
       Robotics applications

    7. RL + LANGUAGE
       RLHF (Reinforcement Learning from Human Feedback)
       Foundation model alignment

    Resources:
    - OpenAI Spinning Up: spinningup.openai.com
    - Stable-Baselines3: sb3 documentation
    - CleanRL: clean, single-file implementations
    - Sutton & Barto: The RL textbook (free online)
    """)


def create_summary_visualization():
    """Create a visual summary of the course."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Timeline data
    weeks = list(range(1, 13))
    topics = [
        'Introduction', 'MDP', 'Dynamic Prog',
        'Monte Carlo', 'TD Learning', 'Q-Learning/SARSA',
        'Func Approx', 'DQN',
        'Policy Grad', 'Actor-Critic', 'PPO',
        'Advanced'
    ]
    categories = [
        'Foundation', 'Foundation', 'Foundation',
        'Tabular', 'Tabular', 'Tabular',
        'Deep Value', 'Deep Value',
        'Deep Policy', 'Deep Policy', 'Deep Policy',
        'Advanced'
    ]
    colors_map = {
        'Foundation': '#3498db',
        'Tabular': '#2ecc71',
        'Deep Value': '#e74c3c',
        'Deep Policy': '#9b59b6',
        'Advanced': '#f39c12'
    }
    colors = [colors_map[c] for c in categories]

    bars = ax.barh(weeks, [1]*12, color=colors, edgecolor='white', height=0.8)

    for i, (week, topic) in enumerate(zip(weeks, topics)):
        ax.text(0.5, week, f'Week {week}: {topic}', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Foundation (1-3)'),
        Patch(facecolor='#2ecc71', label='Tabular Methods (4-6)'),
        Patch(facecolor='#e74c3c', label='Deep Value-Based (7-8)'),
        Patch(facecolor='#9b59b6', label='Deep Policy-Based (9-11)'),
        Patch(facecolor='#f39c12', label='Advanced Topics (12)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 12.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title('12-Week Reinforcement Learning Curriculum',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('12_advanced/course_summary.png', dpi=150, bbox_inches='tight')
    print("Summary saved to '12_advanced/course_summary.png'")
    plt.close()


def main():
    print("\n" + "=" * 60)
    print("WEEK 12 - LESSON 7: COURSE SUMMARY")
    print("12 Weeks of Reinforcement Learning")
    print("=" * 60)

    # 1. Course map
    print_course_map()

    # 2. Algorithm taxonomy
    print_algorithm_taxonomy()

    # 3. Key equations
    print_key_equations()

    # 4. Practical guide
    print_practical_guide()

    # 5. Future directions
    print_future_directions()

    # 6. Visualization
    create_summary_visualization()

    print("\n" + "=" * 60)
    print("CONGRATULATIONS!")
    print("=" * 60)
    print("""
    You have completed the 12-week Reinforcement Learning tutorial!

    What you've learned:
    ✓ RL fundamentals (MDPs, Bellman equations)
    ✓ Tabular methods (MC, TD, Q-Learning, SARSA)
    ✓ Deep RL value methods (DQN and variants)
    ✓ Policy gradient methods (REINFORCE)
    ✓ Actor-Critic methods (A2C, A3C, GAE)
    ✓ State-of-the-art PPO (Clip and Penalty)
    ✓ Advanced topics (MARL, Model-based, IRL)

    You now have a solid foundation to:
    - Apply RL to real-world problems
    - Read and understand RL research papers
    - Implement new algorithms from scratch
    - Choose the right algorithm for any task

    Keep learning, keep building!
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()
