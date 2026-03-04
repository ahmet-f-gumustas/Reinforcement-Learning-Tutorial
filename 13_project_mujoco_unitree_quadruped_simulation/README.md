# Section 13: Multi-Agent RL for Quadruped Robot Control (MuJoCo)

Advanced project applying Multi-Agent Reinforcement Learning (MARL) to Unitree Go2-style quadruped robot locomotion in MuJoCo physics simulation. The project demonstrates how a single robot control problem can be decomposed into multiple cooperating agents.

> **Turkce:** Bu proje, dort bacakli bir robotun Multi-Agent RL ile kontrolunu gosterir. Tek bir robot kontrol problemi, birden fazla isbirligi yapan agent'a ayristirilir.

## Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Approaches](#approaches)
   - [Single-Agent PPO (Baseline)](#1-single-agent-ppo-baseline)
   - [MAPPO Per-Leg Control](#2-mappo-per-leg-control)
   - [Hierarchical Control](#3-hierarchical-manager-worker-control)
   - [Multi-Robot Formation](#4-multi-robot-formation-control)
6. [Architecture Details](#architecture-details)
7. [Exercises](#exercises)
8. [Resources](#resources)

---

## Overview

### Why Multi-Agent for Quadrupeds?

A quadruped robot has a natural decomposition into 4 legs. Instead of training a single monolithic policy for all 12 joints, we can:

```
Single-Agent:   1 policy -> 12 joint torques (complex)
Multi-Agent:    4 agents -> 3 joint torques each (simpler per agent)
Hierarchical:   1 manager + 4 workers (temporal abstraction)
Multi-Robot:    N robots with communication (cooperative MARL)
```

**Benefits of multi-agent decomposition:**
- Simpler per-agent policy (3 actions vs 12)
- Natural credit assignment (each leg's contribution)
- Transferability (leg policies can generalize)
- Scalability (same approach works for different robots)

### Robot Model

Unitree Go2-inspired quadruped with 12 degrees of freedom:

```
         FL ----[TORSO]---- FR
              (0.36m x 0.094m)
         BL ----[     ]---- BR

Each leg: 3 joints
  - Hip Abduction (+/-25 deg): Lateral swing
  - Hip Flexion (+/-60 deg):   Forward-backward swing
  - Knee (-145 to -45 deg):    Knee flexion

Total: 4 legs x 3 joints = 12 actuators
Sensors: 4 foot contacts + IMU (gyro + accelerometer)
```

---

## Project Structure

```
13_project_mujoco_unitree_quadruped_simulation/
├── main.py                     # CLI entry point (train/eval/compare)
├── config.py                   # All hyperparameters and settings
├── README.md
│
├── models/
│   └── unitree_go2.xml         # MuJoCo 12-DOF robot model
│
├── envs/
│   ├── quadruped_env.py        # Single-robot Gymnasium environment
│   └── multi_robot_env.py      # Multi-robot formation environment
│
├── agents/
│   ├── networks.py             # All neural network architectures
│   ├── ppo_agent.py            # Single-agent PPO baseline
│   ├── mappo_agent.py          # MAPPO per-leg CTDE controller
│   ├── hierarchical_agent.py   # Hierarchical manager-worker
│   └── communication.py        # Inter-robot communication module
│
├── training/
│   ├── buffer.py               # Rollout buffers & GAE computation
│   ├── trainer.py              # Training loops for each approach
│   └── callbacks.py            # Logging & checkpoint callbacks
│
└── utils/
    └── visualization.py        # Plots, charts, gait analysis
```

---

## Installation

```bash
# From repository root
pip install gymnasium[mujoco] torch numpy matplotlib mujoco
```

---

## Quick Start

```bash
cd 13_project_mujoco_unitree_quadruped_simulation/

# Train single-agent PPO baseline
python main.py --mode train --agent single_ppo --timesteps 500000

# Train MAPPO (4 leg agents with centralized critic)
python main.py --mode train --agent mappo --timesteps 500000

# Train hierarchical (gait planner + leg controllers)
python main.py --mode train --agent hierarchical --timesteps 500000

# Train multi-robot formation (2 robots with communication)
python main.py --mode train --agent multi_robot --timesteps 300000

# Evaluate trained agent with rendering
python main.py --mode eval --agent mappo --render

# Compare all approaches
python main.py --mode compare --timesteps 200000
```

---

## Approaches

### 1. Single-Agent PPO (Baseline)

One policy controls all 12 joints simultaneously.

```
Observation (39) ──> [Actor Network (256x256)] ──> Action (12)
                      [Critic Network (256x256)] ──> Value (1)
```

- **Agent:** `agents/ppo_agent.py`
- **Algorithm:** PPO-Clip with GAE
- **Purpose:** Benchmark for multi-agent approaches

### 2. MAPPO Per-Leg Control

4 agents, each controlling one leg (3 joints). Centralized Training, Decentralized Execution (CTDE).

```
                    ┌─────────────────────────────┐
                    │   Centralized Critic (256)   │
                    │   Input: Global State (39)   │
                    └──────────┬──────────────────┘
                               │ Value
         ┌─────────┬──────────┼──────────┬─────────┐
         │         │          │          │         │
    ┌────┴───┐ ┌───┴────┐ ┌──┴─────┐ ┌──┴─────┐
    │FL Actor│ │FR Actor│ │BL Actor│ │BR Actor│
    │ (128)  │ │ (128)  │ │ (128)  │ │ (128)  │
    └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
        │ 3        │ 3        │ 3        │ 3
    Local Obs   Local Obs  Local Obs  Local Obs
     (18)        (18)       (18)       (18)
```

**Observation decomposition per agent (18 dims):**

| Component | Dim | Shared? |
|-----------|-----|---------|
| Torso z-position | 1 | Yes |
| Torso quaternion | 4 | Yes |
| Torso velocity | 6 | Yes |
| Leg joint angles | 3 | No |
| Leg joint velocities | 3 | No |
| Foot contact | 1 | No |

- **Agent:** `agents/mappo_agent.py`
- **Key Insight:** Advantages are computed from centralized critic (sees full state), but each actor only uses local observation

### 3. Hierarchical (Manager-Worker) Control

Two-level hierarchy with temporal abstraction:

```
                    ┌──────────────────────────┐
                    │   MANAGER (Gait Planner)  │
                    │   Decides every K=50 steps│
                    │   Input: Torso State (15) │
                    │   Output: Gait Cmd (6)    │
                    └────────────┬─────────────┘
                                 │ Gait Command
                    ┌────────────┴─────────────┐
                    │   Gait Command Encoder    │
                    │   (6) -> Embedding (8)    │
                    └────────────┬─────────────┘
                                 │
         ┌──────────┬────────────┼────────────┬──────────┐
         │          │            │            │          │
    ┌────┴────┐ ┌───┴─────┐ ┌───┴─────┐ ┌───┴─────┐
    │FL Worker│ │FR Worker│ │BL Worker│ │BR Worker│
    │ (128)   │ │ (128)   │ │ (128)   │ │ (128)   │
    │ Obs: 26 │ │ Obs: 26 │ │ Obs: 26 │ │ Obs: 26 │
    └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
         │ 3         │ 3         │ 3         │ 3
```

**Manager** selects gait parameters:
- 4 phase offsets (one per leg)
- Step frequency
- Step height

**Workers** produce joint torques conditioned on gait command.

- **Agent:** `agents/hierarchical_agent.py`
- **Key Insight:** Manager provides temporal abstraction (low frequency), workers provide precise control (high frequency)

### 4. Multi-Robot Formation Control

Multiple quadruped robots maintaining formation while moving forward, with learned inter-robot communication.

```
    Robot 0                          Robot 1
    ┌──────────┐                    ┌──────────┐
    │ Obs      │    messages (16)   │ Obs      │
    │ Encoder  │ <───────────────> │ Encoder  │
    │ (256)    │                    │ (256)    │
    └────┬─────┘                    └────┬─────┘
         │ hidden + comm                  │
    ┌────┴─────┐                    ┌────┴─────┐
    │  Actor   │                    │  Actor   │
    │  (256)   │                    │  (256)   │
    └────┬─────┘                    └────┬─────┘
         │ 12 actions                     │ 12 actions
```

**Communication protocol per step:**
1. Each robot encodes its observation
2. Messages are produced and exchanged
3. Received messages are decoded and added to hidden state
4. Actions are selected from augmented hidden state

**Reward:** Forward progress + Formation maintenance penalty

- **Agent:** `agents/communication.py`
- **Environment:** `envs/multi_robot_env.py` (programmatically generates multi-robot XML)

---

## Architecture Details

### Observation Space (39 dims)

```
[z_pos(1), quaternion(4), joint_angles(12), body_vel(6), joint_vel(12), contacts(4)]
   │            │              │                │              │            │
   └─ height    └─ orient.     └─ 4x3 joints    └─ lin+ang     └─ 12 vel   └─ 4 feet
```

### Action Space (12 dims)

```
[fl_abd, fl_hip, fl_knee, fr_abd, fr_hip, fr_knee, bl_abd, bl_hip, bl_knee, br_abd, br_hip, br_knee]
 └── Front Left ──┘   └── Front Right ──┘   └── Back Left ───┘   └── Back Right ──┘
```

### Reward Function

```
reward = forward_reward + healthy_reward - ctrl_cost - energy_cost

forward_reward = x_velocity * weight       (encourages forward movement)
healthy_reward = 1.0 if robot is upright    (encourages stability)
ctrl_cost     = weight * sum(action^2)      (penalizes large torques)
energy_cost   = weight * sum(|action * vel|) (penalizes energy waste)
```

### Key Algorithms

| Algorithm | Type | Key Idea |
|-----------|------|----------|
| PPO-Clip | Single-Agent | Clipped surrogate objective for stable updates |
| MAPPO | Multi-Agent | Multi-Agent PPO with shared centralized critic (CTDE) |
| Hierarchical RL | Multi-Agent | Manager-Worker with temporal abstraction |
| CommNet | Multi-Agent | Learned communication between agents |

---

## Exercises

### Exercise 1: Parameter Sharing

Modify `mappo_agent.py` to share weights across all 4 leg actors. Compare learning speed and final performance with independent actors.

### Exercise 2: Terrain Curriculum

Add terrain randomization to `quadruped_env.py`:
1. Start with flat terrain
2. Progress to uneven terrain (random bumps)
3. Compare with fixed flat terrain training

### Exercise 3: Adversarial Perturbations

Add random force perturbations during training:
1. Apply external forces to the torso at random intervals
2. Compare robustness of single-agent vs multi-agent policies
3. Test recovery behavior

### Exercise 4: QMIX for Multi-Robot

Replace the simple centralized critic in `multi_robot_env.py` with QMIX (factored Q-values). Compare formation quality.

### Exercise 5: Gait Emergence Analysis

Collect foot contact data from trained hierarchical controller and analyze:
1. What gait patterns emerge? (walk, trot, gallop)
2. Does the manager learn to select different gaits for different speeds?
3. Visualize gait transitions

---

## Resources

### Multi-Agent RL Papers
- [MAPPO (Yu et al., 2022)](https://arxiv.org/abs/2103.01955) - The Surprising Effectiveness of PPO in Cooperative MARL
- [MADDPG (Lowe et al., 2017)](https://arxiv.org/abs/1706.02275) - Multi-Agent Actor-Critic
- [QMIX (Rashid et al., 2018)](https://arxiv.org/abs/1803.11485) - Monotonic Value Function Factorisation
- [CommNet (Sukhbaatar et al., 2016)](https://arxiv.org/abs/1605.07736) - Learning Multiagent Communication

### Quadruped Locomotion Papers
- [Learning Agile Locomotion (Lee et al., 2020)](https://arxiv.org/abs/2011.11951) - Sim-to-real quadruped
- [Legged Locomotion with Transformers (Radosavovic et al., 2024)](https://arxiv.org/abs/2403.09052) - Robot Locomotion via Transformers
- [Walk These Ways (Margolis et al., 2023)](https://arxiv.org/abs/2212.11972) - Rapid locomotion adaptation

### Hierarchical RL Papers
- [FeUdal Networks (Vezhnevets et al., 2017)](https://arxiv.org/abs/1703.01161) - Feudal Networks for Hierarchical RL
- [Option-Critic (Bacon et al., 2017)](https://arxiv.org/abs/1609.05140) - Option-Critic Architecture
- [HIRO (Nachum et al., 2018)](https://arxiv.org/abs/1805.08296) - Data-Efficient Hierarchical RL

### Frameworks
- [MuJoCo](https://mujoco.org/) - Physics simulation
- [Gymnasium](https://gymnasium.farama.org/) - RL environments
- [PettingZoo](https://pettingzoo.farama.org/) - Multi-agent environments
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
