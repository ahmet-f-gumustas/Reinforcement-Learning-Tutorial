"""
05 - Curriculum Learning for RL

This script demonstrates curriculum learning — training an agent on
progressively harder tasks to improve learning speed and final performance.

Demonstrates:
- Curriculum design for RL
- Progressive difficulty scaling
- Automatic curriculum with success rate
- Task decomposition
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
import gymnasium as gym


# =============================================================================
# Curriculum GridWorld
# =============================================================================

class CurriculumGridWorld:
    """GridWorld with adjustable difficulty (goal distance)."""

    def __init__(self, size: int = 10, goal_distance: int = 2):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.set_difficulty(goal_distance)
        self.reset()

    def set_difficulty(self, goal_distance: int):
        """Set goal distance from start (higher = harder)."""
        self.goal_distance = min(goal_distance, 2 * (self.size - 1))
        d = self.goal_distance
        gr = min(d, self.size - 1)
        gc = min(d - gr, self.size - 1)
        self.goal = (gr, gc)

    def reset(self) -> int:
        self.pos = (0, 0)
        return self._state()

    def _state(self) -> int:
        return self.pos[0] * self.size + self.pos[1]

    def step(self, action: int) -> Tuple[int, float, bool]:
        r, c = self.pos
        if action == 0: r = max(0, r-1)
        elif action == 1: c = min(self.size-1, c+1)
        elif action == 2: r = min(self.size-1, r+1)
        elif action == 3: c = max(0, c-1)
        self.pos = (r, c)
        done = self.pos == self.goal
        reward = 10.0 if done else -0.01
        return self._state(), reward, done


class QLAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=0.2):
        self.n_actions = n_actions
        self.lr, self.gamma, self.epsilon = lr, gamma, epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, s, a, r, ns, done):
        target = r + self.gamma * np.max(self.q_table[ns]) * (1-int(done))
        self.q_table[s, a] += self.lr * (target - self.q_table[s, a])


def train_with_curriculum(env, agent, stages, episodes_per_stage=500, max_steps=100):
    """Train with a curriculum of increasing difficulty."""
    all_rewards = []
    stage_boundaries = []

    for stage, difficulty in enumerate(stages):
        env.set_difficulty(difficulty)
        stage_rewards = []
        stage_boundaries.append(len(all_rewards))
        print(f"  Stage {stage+1}/{len(stages)} | Difficulty: {difficulty} | "
              f"Goal: {env.goal}")

        for ep in range(episodes_per_stage):
            state = env.reset()
            ep_r = 0
            for _ in range(max_steps):
                action = agent.select_action(state)
                ns, r, done = env.step(action)
                agent.update(state, action, r, ns, done)
                ep_r += r; state = ns
                if done: break
            stage_rewards.append(ep_r)
            all_rewards.append(ep_r)

        avg = np.mean(stage_rewards[-100:])
        print(f"    Avg reward: {avg:.2f}")

    return all_rewards, stage_boundaries


def train_without_curriculum(env, agent, final_difficulty, total_episodes=2500, max_steps=100):
    """Train directly on hardest task (no curriculum)."""
    env.set_difficulty(final_difficulty)
    rewards = []
    for _ in range(total_episodes):
        state = env.reset()
        ep_r = 0
        for _ in range(max_steps):
            action = agent.select_action(state)
            ns, r, done = env.step(action)
            agent.update(state, action, r, ns, done)
            ep_r += r; state = ns
            if done: break
        rewards.append(ep_r)
    return rewards


def demonstrate_curriculum():
    """Explain curriculum learning."""
    print("=" * 60)
    print("CURRICULUM LEARNING FOR RL")
    print("=" * 60)

    print("""
    Idea: Start with easy tasks, progressively increase difficulty.

    Example (Robot Navigation):
        Stage 1: Goal 1 meter away      (easy)
        Stage 2: Goal 3 meters away      (medium)
        Stage 3: Goal 5 meters away      (hard)
        Stage 4: Goal 10 meters away     (full task)

    Why It Works:
    1. Easy tasks provide dense learning signals early
    2. Skills transfer from easy → hard tasks
    3. Agent builds confidence before facing hard challenges
    4. Avoids getting stuck in sparse reward problems

    Curriculum Design Strategies:
    ┌────────────────────────┬────────────────────────────┐
    │ Strategy                │ Description                │
    ├────────────────────────┼────────────────────────────┤
    │ Manual stages          │ Hand-designed difficulty    │
    │ Success-based          │ Move to harder task when    │
    │                        │ success rate > threshold    │
    │ Automatic (ADR)        │ Randomize difficulty, adapt│
    │ Self-play              │ Agent plays against itself │
    │ Reverse curriculum     │ Start near goal, expand    │
    └────────────────────────┴────────────────────────────┘
    """)


def main():
    print("\n" + "=" * 60)
    print("WEEK 12 - LESSON 5: CURRICULUM LEARNING")
    print("=" * 60)

    np.random.seed(42)
    demonstrate_curriculum()

    env = CurriculumGridWorld(size=10)
    stages = [2, 5, 9, 14, 18]  # Increasing goal distance
    final_difficulty = 18
    episodes_per = 500
    total_episodes = episodes_per * len(stages)

    # With curriculum
    print("\nTraining WITH curriculum...")
    np.random.seed(42)
    agent_curr = QLAgent(env.n_states, env.n_actions)
    curriculum_rewards, boundaries = train_with_curriculum(
        env, agent_curr, stages, episodes_per_stage=episodes_per)

    # Without curriculum
    print(f"\nTraining WITHOUT curriculum (directly on difficulty={final_difficulty})...")
    np.random.seed(42)
    agent_no_curr = QLAgent(env.n_states, env.n_actions)
    no_curriculum_rewards = train_without_curriculum(
        env, agent_no_curr, final_difficulty, total_episodes=total_episodes)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    window = 100

    ax = axes[0]
    sm1 = [np.mean(curriculum_rewards[max(0,i-window):i+1])
           for i in range(len(curriculum_rewards))]
    sm2 = [np.mean(no_curriculum_rewards[max(0,i-window):i+1])
           for i in range(len(no_curriculum_rewards))]
    ax.plot(sm1, color='coral', linewidth=2, label='With Curriculum')
    ax.plot(sm2, color='steelblue', linewidth=2, label='Without Curriculum')
    for b in boundaries[1:]:
        ax.axvline(x=b, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Reward')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels = ['With\nCurriculum', 'Without\nCurriculum']
    finals = [np.mean(curriculum_rewards[-200:]),
              np.mean(no_curriculum_rewards[-200:])]
    stds = [np.std(curriculum_rewards[-200:]),
            np.std(no_curriculum_rewards[-200:])]
    ax.bar(labels, finals, yerr=stds, capsize=10,
           color=['coral', 'steelblue'], edgecolor='black')
    ax.set_ylabel('Avg Reward (last 200 ep)')
    ax.set_title('Final Performance')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Curriculum Learning Effect', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('12_advanced/curriculum_learning.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to '12_advanced/curriculum_learning.png'")
    plt.close()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Curriculum = start easy, gradually increase difficulty")
    print("2. Skills from easy tasks transfer to harder tasks")
    print("3. Avoids sparse reward problem in early learning")
    print("4. Success-based auto-curriculum is practical and robust")
    print("5. Critical for real-world robotics and complex environments")
    print("\nNext: Final Project!")
    print("=" * 60)


if __name__ == "__main__":
    main()
