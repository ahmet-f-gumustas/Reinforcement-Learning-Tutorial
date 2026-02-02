"""
Bipedal Walker - Real-Time Walking Robot Simulation
====================================================
This example uses the BipedalWalker-v3 environment to train a robot to walk.

The robot has 4 joints (2 legs x 2 joints each) and must learn to walk
without falling. This is a continuous control problem - actions are not
discrete like "left/right" but continuous values like "apply 0.7 force".

Modes:
1. Human Mode: Control with keyboard (simplified)
2. AI Mode: Watch trained TD3 agent walk
3. Training Mode: Train the TD3 agent
4. Demo Mode: Quick training + AI demonstration

This example uses TD3 (Twin Delayed DDPG) algorithm, which is good for
continuous action spaces.
"""

import gymnasium as gym
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
from pathlib import Path
import copy
import time
import sys
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

# Output directory - where we save models and logs
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = Path(__file__).parent / "output" / SCRIPT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Device selection - use GPU if available
# GPU makes training much faster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# TERMINAL COLORS - Makes output more readable
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'      # Purple
    BLUE = '\033[94m'        # Blue
    CYAN = '\033[96m'        # Cyan
    GREEN = '\033[92m'       # Green
    YELLOW = '\033[93m'      # Yellow
    RED = '\033[91m'         # Red
    BOLD = '\033[1m'         # Bold
    DIM = '\033[2m'          # Dim
    RESET = '\033[0m'        # Reset to default

    @staticmethod
    def colorize(text, color):
        """Wrap text with color codes."""
        return f"{color}{text}{Colors.RESET}"


# =============================================================================
# TRAINING LOGGER - Beautiful training output
# =============================================================================

class TrainingLogger:
    """
    Handles all training output formatting.

    Features:
        - Progress bar
        - Color-coded metrics
        - Trend indicators
        - Periodic summaries
        - Time tracking
    """

    def __init__(self, total_episodes):
        self.total_episodes = total_episodes
        self.start_time = None
        self.episode_start_time = None

        # Statistics tracking
        self.rewards = []
        self.critic_losses = []
        self.actor_losses = []
        self.episode_steps = []

        # Best values
        self.best_reward = float('-inf')
        self.best_avg = float('-inf')

    def start_training(self):
        """Called at the beginning of training."""
        self.start_time = time.time()
        self._print_header()

    def _print_header(self):
        """Print training header."""
        print("\n" + "=" * 80)
        print(Colors.colorize("  TRAINING STARTED", Colors.BOLD + Colors.CYAN))
        print("=" * 80)
        print(f"  Total Episodes  : {Colors.colorize(str(self.total_episodes), Colors.YELLOW)}")
        print(f"  Device          : {Colors.colorize(str(device), Colors.GREEN)}")
        print(f"  Output          : {Colors.colorize(str(OUTPUT_DIR), Colors.DIM)}")
        print("=" * 80)
        print()

        # Column headers
        header = (
            f"{'Episode':>8} | "
            f"{'Progress':>12} | "
            f"{'Score':>8} | "
            f"{'Avg(100)':>10} | "
            f"{'Best':>8} | "
            f"{'Steps':>6} | "
            f"{'C.Loss':>8} | "
            f"{'A.Loss':>8} | "
            f"{'Time':>6}"
        )
        print(Colors.colorize(header, Colors.DIM))
        print("-" * 100)

    def start_episode(self):
        """Called at the start of each episode."""
        self.episode_start_time = time.time()

    def log_episode(self, episode, reward, steps, critic_loss, actor_loss, saved=False):
        """
        Log episode results with beautiful formatting.

        Args:
            episode: Current episode number
            reward: Total reward this episode
            steps: Number of steps this episode
            critic_loss: Average critic loss
            actor_loss: Average actor loss
            saved: Whether model was saved this episode
        """
        # Store statistics
        self.rewards.append(reward)
        self.episode_steps.append(steps)
        if critic_loss > 0:
            self.critic_losses.append(critic_loss)
        if actor_loss > 0:
            self.actor_losses.append(actor_loss)

        # Calculate metrics
        avg_reward = np.mean(self.rewards[-100:]) if self.rewards else 0
        episode_time = time.time() - self.episode_start_time

        # Update best values
        if reward > self.best_reward:
            self.best_reward = reward
        if avg_reward > self.best_avg:
            self.best_avg = avg_reward

        # Trend indicator (comparing to last 10 episodes average)
        trend = self._get_trend()

        # Progress bar
        progress = self._make_progress_bar(episode, self.total_episodes, width=12)

        # Color score based on performance
        score_str = self._color_score(reward)
        avg_str = self._color_avg(avg_reward)
        best_str = Colors.colorize(f"{self.best_reward:>8.1f}", Colors.YELLOW)

        # Format losses
        c_loss_str = f"{critic_loss:>8.2f}" if critic_loss > 0 else "    -   "
        a_loss_str = f"{actor_loss:>8.4f}" if actor_loss > 0 else "    -   "

        # Time formatting
        time_str = f"{episode_time:>5.1f}s"

        # Build output line
        save_indicator = Colors.colorize(" [SAVED]", Colors.GREEN) if saved else ""

        line = (
            f"{episode:>8} | "
            f"{progress} | "
            f"{score_str} | "
            f"{avg_str} {trend} | "
            f"{best_str} | "
            f"{steps:>6} | "
            f"{c_loss_str} | "
            f"{a_loss_str} | "
            f"{time_str}"
            f"{save_indicator}"
        )
        print(line)

        # Print summary every 50 episodes
        if episode > 0 and episode % 50 == 0:
            self._print_summary(episode)

    def _make_progress_bar(self, current, total, width=20):
        """Create a visual progress bar."""
        percent = current / total
        filled = int(width * percent)
        empty = width - filled

        # Use block characters for smooth progress
        bar = "█" * filled + "░" * empty
        percent_str = f"{percent*100:>5.1f}%"

        return Colors.colorize(bar, Colors.CYAN) + " " + percent_str

    def _get_trend(self):
        """Get trend indicator based on recent performance."""
        if len(self.rewards) < 20:
            return " "

        recent = np.mean(self.rewards[-10:])
        previous = np.mean(self.rewards[-20:-10])

        if recent > previous + 5:
            return Colors.colorize("↑", Colors.GREEN)
        elif recent < previous - 5:
            return Colors.colorize("↓", Colors.RED)
        else:
            return Colors.colorize("→", Colors.YELLOW)

    def _color_score(self, score):
        """Color the score based on performance."""
        if score >= 300:
            return Colors.colorize(f"{score:>8.1f}", Colors.GREEN + Colors.BOLD)
        elif score >= 200:
            return Colors.colorize(f"{score:>8.1f}", Colors.GREEN)
        elif score >= 100:
            return Colors.colorize(f"{score:>8.1f}", Colors.YELLOW)
        elif score >= 0:
            return Colors.colorize(f"{score:>8.1f}", Colors.DIM)
        else:
            return Colors.colorize(f"{score:>8.1f}", Colors.RED)

    def _color_avg(self, avg):
        """Color the average based on performance."""
        if avg >= 250:
            return Colors.colorize(f"{avg:>8.1f}", Colors.GREEN + Colors.BOLD)
        elif avg >= 150:
            return Colors.colorize(f"{avg:>8.1f}", Colors.GREEN)
        elif avg >= 50:
            return Colors.colorize(f"{avg:>8.1f}", Colors.YELLOW)
        elif avg >= 0:
            return Colors.colorize(f"{avg:>8.1f}", Colors.DIM)
        else:
            return Colors.colorize(f"{avg:>8.1f}", Colors.RED)

    def _print_summary(self, episode):
        """Print periodic summary statistics."""
        elapsed = time.time() - self.start_time
        eps_per_sec = episode / elapsed if elapsed > 0 else 0
        remaining = (self.total_episodes - episode) / eps_per_sec if eps_per_sec > 0 else 0

        print()
        print(Colors.colorize(f"  ┌─── Summary at Episode {episode} ───┐", Colors.CYAN))
        print(f"  │ Avg Reward (last 100): {Colors.colorize(f'{np.mean(self.rewards[-100:]):>8.1f}', Colors.YELLOW)}")
        print(f"  │ Best Single Episode : {Colors.colorize(f'{self.best_reward:>8.1f}', Colors.GREEN)}")
        print(f"  │ Best Average (100)  : {Colors.colorize(f'{self.best_avg:>8.1f}', Colors.GREEN)}")

        if self.critic_losses:
            avg_c_loss = np.mean(self.critic_losses[-100:])
            print(f"  │ Avg Critic Loss    : {avg_c_loss:>8.2f}")
        if self.actor_losses:
            avg_a_loss = np.mean(self.actor_losses[-100:])
            print(f"  │ Avg Actor Loss     : {avg_a_loss:>8.4f}")

        print(f"  │ Avg Steps/Episode  : {np.mean(self.episode_steps[-100:]):>8.1f}")
        print(f"  │ Time Elapsed       : {self._format_time(elapsed)}")
        print(f"  │ Est. Remaining     : {self._format_time(remaining)}")
        print(f"  │ Speed              : {eps_per_sec:>6.2f} ep/s")
        print(Colors.colorize(f"  └{'─' * 28}┘", Colors.CYAN))
        print()

    def _format_time(self, seconds):
        """Format seconds into readable time string."""
        if seconds < 60:
            return f"{seconds:>5.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins:>2}m {secs:02d}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours:>2}h {mins:02d}m"

    def finish_training(self):
        """Called at the end of training."""
        elapsed = time.time() - self.start_time

        print()
        print("=" * 80)
        print(Colors.colorize("  TRAINING COMPLETE!", Colors.BOLD + Colors.GREEN))
        print("=" * 80)
        print()
        print(f"  Total Episodes    : {Colors.colorize(str(self.total_episodes), Colors.YELLOW)}")
        print(f"  Total Time        : {Colors.colorize(self._format_time(elapsed), Colors.CYAN)}")
        print(f"  Average Speed     : {Colors.colorize(f'{self.total_episodes/elapsed:.2f} ep/s', Colors.CYAN)}")
        print()
        print(f"  Best Episode Score: {Colors.colorize(f'{self.best_reward:.1f}', Colors.GREEN + Colors.BOLD)}")
        print(f"  Best Avg (100)    : {Colors.colorize(f'{self.best_avg:.1f}', Colors.GREEN + Colors.BOLD)}")
        print(f"  Final Avg (100)   : {Colors.colorize(f'{np.mean(self.rewards[-100:]):.1f}', Colors.YELLOW)}")
        print()

        # Performance assessment
        if self.best_avg >= 300:
            print(Colors.colorize("  Status: SOLVED! Robot learned to walk!", Colors.GREEN + Colors.BOLD))
        elif self.best_avg >= 200:
            print(Colors.colorize("  Status: GOOD! Robot walks reasonably well.", Colors.GREEN))
        elif self.best_avg >= 100:
            print(Colors.colorize("  Status: LEARNING. More training recommended.", Colors.YELLOW))
        else:
            print(Colors.colorize("  Status: EARLY STAGE. Needs more training.", Colors.RED))

        print()
        print("=" * 80)


# =============================================================================
# ACTOR NETWORK (POLICY)
# =============================================================================

class Actor(nn.Module):
    """
    Actor Network - Decides what action to take.

    In continuous control, the actor outputs actual values, not probabilities.
    For BipedalWalker, it outputs 4 values between -1 and 1.

    Each value controls one joint:
        - action[0]: Hip joint 1 (left leg)
        - action[1]: Knee joint 1 (left leg)
        - action[2]: Hip joint 2 (right leg)
        - action[3]: Knee joint 2 (right leg)

    Architecture:
        Input (24 state features) -> Hidden (256) -> Hidden (256) -> Output (4 actions)

    The 24 state features include:
        - Hull angle and angular velocity
        - Horizontal and vertical speed
        - Joint angles and angular velocities
        - Leg ground contact
        - LIDAR readings (10 values)
    """

    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()

        # Three fully connected layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Store max_action to scale output
        self.max_action = max_action

    def forward(self, state):
        # ReLU activation for hidden layers
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        # Tanh activation for output layer
        # tanh outputs values between -1 and 1
        # We then scale by max_action
        return self.max_action * torch.tanh(self.fc3(x))


# =============================================================================
# CRITIC NETWORK (VALUE ESTIMATOR)
# =============================================================================

class Critic(nn.Module):
    """
    Critic Network - Estimates the value of state-action pairs.

    TD3 uses TWO critics (Q1 and Q2) to reduce overestimation bias.
    The minimum of the two estimates is used as the target.

    Input: state + action (concatenated)
    Output: Q-value (expected future reward)

    Why two critics?
        - Single critic tends to overestimate Q-values
        - Taking minimum of two estimates is more conservative
        - This leads to more stable training
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # First critic network (Q1)
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)

        # Second critic network (Q2)
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # Concatenate state and action as input
        sa = torch.cat([state, action], dim=1)

        # Q1 forward pass
        q1 = torch.relu(self.q1_fc1(sa))
        q1 = torch.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)

        # Q2 forward pass
        q2 = torch.relu(self.q2_fc1(sa))
        q2 = torch.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)

        return q1, q2

    def q1_only(self, state, action):
        """Get Q1 value only (used for actor update)."""
        sa = torch.cat([state, action], dim=1)
        q1 = torch.relu(self.q1_fc1(sa))
        q1 = torch.relu(self.q1_fc2(q1))
        return self.q1_fc3(q1)


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """
    Replay Buffer - Stores experiences for training.

    Same concept as DQN, but now we store continuous actions.
    Each experience: (state, action, reward, next_state, done)
    """

    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# TD3 AGENT
# =============================================================================

class TD3Agent:
    """
    TD3 (Twin Delayed DDPG) Agent - For continuous control.

    Key improvements over DDPG:
        1. Twin Critics: Use minimum of two Q-values (reduces overestimation)
        2. Delayed Policy Updates: Update actor less frequently than critic
        3. Target Policy Smoothing: Add noise to target actions

    TD3 is one of the best algorithms for continuous control tasks.
    """

    def __init__(self, state_dim, action_dim, max_action,
                 lr=3e-4, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Hyperparameters
        self.gamma = gamma          # Discount factor
        self.tau = tau              # Soft update rate for target networks
        self.policy_noise = policy_noise  # Noise added to target policy
        self.noise_clip = noise_clip      # Maximum noise value
        self.policy_delay = policy_delay  # Update actor every N critic updates

        # Actor networks (policy)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic networks (twin Q-functions)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Replay buffer
        self.memory = ReplayBuffer()
        self.batch_size = 256

        # Training counter
        self.total_steps = 0

        # Exploration noise
        self.exploration_noise = 0.1

        # Loss tracking for logging
        self.last_critic_loss = 0
        self.last_actor_loss = 0

    def choose_action(self, state, training=True):
        """
        Choose action using the actor network.

        During training, add exploration noise.
        During evaluation, use deterministic policy.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = self.actor(state_tensor).cpu().numpy()[0]

        # Add exploration noise during training
        if training:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = action + noise
            # Clip to valid action range
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Perform one training step.

        TD3 training procedure:
            1. Sample batch from replay buffer
            2. Compute target Q-value with clipped noise
            3. Update critics using MSE loss
            4. Every 'policy_delay' steps, update actor
            5. Soft update target networks
        """
        if len(self.memory) < self.batch_size:
            return 0, 0

        self.total_steps += 1

        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # =====================================================================
        # STEP 1: Compute target Q-value
        # =====================================================================
        with torch.no_grad():
            # Get next actions from target actor
            next_actions = self.actor_target(next_states)

            # Add clipped noise for target policy smoothing
            # This prevents the policy from exploiting Q-function errors
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)

            # Get Q-values from both target critics
            target_q1, target_q2 = self.critic_target(next_states, next_actions)

            # Take minimum Q-value (reduces overestimation)
            target_q = torch.min(target_q1, target_q2)

            # Bellman equation: Q = reward + gamma * Q_next
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # =====================================================================
        # STEP 2: Update critics
        # =====================================================================
        current_q1, current_q2 = self.critic(states, actions)

        # MSE loss for both critics
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.last_critic_loss = critic_loss.item()
        actor_loss = 0

        # =====================================================================
        # STEP 3: Delayed policy update
        # =====================================================================
        if self.total_steps % self.policy_delay == 0:
            # Actor loss: maximize Q-value (minimize negative Q)
            actor_loss = -self.critic.q1_only(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.last_actor_loss = actor_loss.item()

            # =====================================================================
            # STEP 4: Soft update target networks
            # =====================================================================
            # target = tau * current + (1-tau) * target
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            actor_loss = actor_loss.item()

        return critic_loss.item(), actor_loss

    def save(self, path):
        """Save model to file."""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'total_steps': self.total_steps
        }, path)

    def load(self, path):
        """Load model from file."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            self.actor.load_state_dict(checkpoint['actor'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.total_steps = checkpoint.get('total_steps', 0)
            print(f"Model loaded: {path}")
            return True
        return False


# =============================================================================
# GAME CLASS
# =============================================================================

class BipedalWalkerGame:
    """
    Main game class for BipedalWalker.

    The BipedalWalker environment:
        - Goal: Walk as far as possible without falling
        - Reward: +300 for reaching the end, -100 for falling
        - Small reward for moving forward
        - Penalty for using too much motor power
        - Episode ends when: robot falls or reaches end
        - Solved when average reward >= 300 over 100 episodes
    """

    def __init__(self):
        # Create environment with rendering
        self.env = gym.make("BipedalWalker-v3", render_mode="human")

        # State: 24 continuous values
        self.state_dim = self.env.observation_space.shape[0]
        # Actions: 4 continuous values (torques for 4 joints)
        self.action_dim = self.env.action_space.shape[0]
        # Max action value (usually 1.0)
        self.max_action = float(self.env.action_space.high[0])

        # Create TD3 agent
        self.agent = TD3Agent(self.state_dim, self.action_dim, self.max_action)
        self.model_path = OUTPUT_DIR / "td3_model.pt"

        # Game state
        self.state = None
        self.total_reward = 0
        self.episode = 0
        self.step_count = 0
        self.done = True
        self.mode = "human"

        # Statistics
        self.rewards_history = []
        self.best_reward = float('-inf')

    def reset(self):
        """Reset environment for new episode."""
        self.state, _ = self.env.reset()
        self.total_reward = 0
        self.step_count = 0
        self.done = False
        self.episode += 1
        return self.state

    def get_human_action(self, keys):
        """
        Convert keyboard to continuous action.

        This is simplified - real control would need analog input.
        We map keys to fixed torque values.
        """
        action = np.zeros(4)

        # Left leg controls (A, S)
        if keys[pygame.K_a]:
            action[0] = 1.0   # Left hip forward
            action[1] = -1.0  # Left knee extend
        if keys[pygame.K_s]:
            action[0] = -1.0  # Left hip backward
            action[1] = 1.0   # Left knee flex

        # Right leg controls (K, L)
        if keys[pygame.K_k]:
            action[2] = 1.0   # Right hip forward
            action[3] = -1.0  # Right knee extend
        if keys[pygame.K_l]:
            action[2] = -1.0  # Right hip backward
            action[3] = 1.0   # Right knee flex

        return action

    def step(self, action):
        """Execute one game step."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.total_reward += reward
        self.step_count += 1

        # If training, learn from this experience
        if self.mode == "train":
            self.agent.remember(self.state, action, reward, next_state, done)
            self.agent.train_step()

        self.state = next_state
        self.done = done

        if done:
            self.rewards_history.append(self.total_reward)
            if self.total_reward > self.best_reward:
                self.best_reward = self.total_reward

        return next_state, reward, done, info

    def run_human_mode(self):
        """
        Human control mode.

        Controls are simplified since walking requires precise coordination.
        It's very hard to control manually!
        """
        print("\n" + "=" * 60)
        print("BIPEDAL WALKER - HUMAN MODE")
        print("=" * 60)
        print("Controls (very difficult!):")
        print("  - A: Left leg forward")
        print("  - S: Left leg backward")
        print("  - K: Right leg forward")
        print("  - L: Right leg backward")
        print("  - R: Restart")
        print("  - Q / ESC: Quit")
        print("=" * 60)
        print("TIP: This is very hard to control manually!")
        print("     Try AI mode to see proper walking.")
        print("=" * 60)

        self.mode = "human"
        pygame.init()

        running = True
        clock = pygame.time.Clock()
        self.reset()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset()
                        print(f"\n[Restarted] Episode {self.episode}")

            if not self.done:
                keys = pygame.key.get_pressed()
                action = self.get_human_action(keys)
                _, reward, done, _ = self.step(action)

                if done:
                    result = "SUCCESS!" if self.total_reward > 200 else "FELL!"
                    print(f"Episode {self.episode}: {result} Score: {self.total_reward:.1f}")
            else:
                pygame.time.wait(1000)
                self.reset()

            clock.tick(30)

        self.env.close()
        pygame.quit()

    def run_ai_mode(self, episodes=5):
        """Watch trained AI walk."""
        print("\n" + "=" * 60)
        print("BIPEDAL WALKER - AI MODE")
        print("=" * 60)

        self.mode = "ai"

        if not self.agent.load(self.model_path):
            print("No trained model found!")
            print("Run training mode first.")
            return

        for ep in range(episodes):
            self.reset()
            print(f"\n[AI Walking] Episode {ep + 1}/{episodes}")

            while not self.done:
                action = self.agent.choose_action(self.state, training=False)
                self.step(action)
                pygame.time.wait(10)

            result = "SUCCESS!" if self.total_reward > 200 else "FELL!"
            print(f"Result: {result} Score: {self.total_reward:.1f}")

        avg_reward = np.mean(self.rewards_history[-episodes:])
        print(f"\nAverage Score: {avg_reward:.1f}")

        self.env.close()

    def run_training_mode(self, episodes=1000, save_every=100):
        """
        Train the TD3 agent to walk.

        TD3 typically needs more episodes than DQN because:
            1. Continuous actions are harder to learn
            2. Walking requires precise motor coordination
            3. The reward signal is sparse (mostly from falling)
        """
        self.mode = "train"

        # Switch to non-rendering for faster training
        self.env.close()
        self.env = gym.make("BipedalWalker-v3")

        # Initialize logger
        logger = TrainingLogger(episodes)
        logger.start_training()

        best_avg = float('-inf')

        for ep in range(1, episodes + 1):
            logger.start_episode()
            self.reset()

            # Collect losses for this episode
            episode_critic_losses = []
            episode_actor_losses = []

            while not self.done:
                action = self.agent.choose_action(self.state, training=True)
                self.step(action)

                # Track losses
                if self.agent.last_critic_loss > 0:
                    episode_critic_losses.append(self.agent.last_critic_loss)
                if self.agent.last_actor_loss != 0:
                    episode_actor_losses.append(abs(self.agent.last_actor_loss))

            # Calculate average losses
            avg_critic_loss = np.mean(episode_critic_losses) if episode_critic_losses else 0
            avg_actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0

            # Check if we should save
            avg_reward = np.mean(self.rewards_history[-100:]) if self.rewards_history else 0
            saved = False

            if avg_reward > best_avg and ep > 100:
                best_avg = avg_reward
                self.agent.save(self.model_path)
                saved = True

            if ep % save_every == 0:
                self.agent.save(self.model_path)
                saved = True

            # Log this episode
            logger.log_episode(
                episode=ep,
                reward=self.total_reward,
                steps=self.step_count,
                critic_loss=avg_critic_loss,
                actor_loss=avg_actor_loss,
                saved=saved
            )

        # Final save
        self.agent.save(self.model_path)

        # Print final summary
        logger.finish_training()

        self.env.close()
        return self.rewards_history

    def demo_mode(self):
        """Quick demo - short training + AI show."""
        print("\n" + "=" * 60)
        print(Colors.colorize("BIPEDAL WALKER - DEMO MODE", Colors.BOLD + Colors.CYAN))
        print("=" * 60)

        print("\nStarting training (300 episodes)...")
        print("Note: Walking takes time to learn!")
        self.run_training_mode(episodes=300, save_every=100)

        print("\nStarting AI demonstration...")
        self.env = gym.make("BipedalWalker-v3", render_mode="human")
        self.run_ai_mode(episodes=3)


# =============================================================================
# MAIN MENU
# =============================================================================

def print_menu():
    """Display main menu."""
    print("\n" + "=" * 60)
    print(Colors.colorize("BIPEDAL WALKER - MAIN MENU", Colors.BOLD + Colors.CYAN))
    print("=" * 60)
    print("1. Human Mode (Try to walk - very hard!)")
    print("2. AI Mode (Watch trained robot walk)")
    print("3. Training Mode (Train the robot)")
    print("4. Demo Mode (Quick train + watch)")
    print("5. Exit")
    print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bipedal Walker - Walking Robot Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bipedal_walker_game.py --watch          # Watch trained AI walk
  python bipedal_walker_game.py --watch -n 10    # Watch 10 episodes
  python bipedal_walker_game.py --train -n 500   # Train for 500 episodes
  python bipedal_walker_game.py --human          # Play with keyboard
  python bipedal_walker_game.py                  # Show interactive menu
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Watch the trained AI walk (loads saved model)"
    )
    mode_group.add_argument(
        "--train", "-t",
        action="store_true",
        help="Train the AI from scratch or continue training"
    )
    mode_group.add_argument(
        "--human", "-H",
        action="store_true",
        help="Play in human mode with keyboard"
    )
    mode_group.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Quick demo: short training + AI show"
    )

    # Options
    parser.add_argument(
        "-n", "--episodes",
        type=int,
        default=5,
        help="Number of episodes (default: 5 for watch, 1000 for train)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file (default: auto-detect in output folder)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Header
    print("\n" + "=" * 60)
    print(Colors.colorize("BIPEDAL WALKER - Walking Robot Simulation", Colors.BOLD + Colors.CYAN))
    print("=" * 60)
    print(f"  Output    : {Colors.colorize(str(OUTPUT_DIR), Colors.DIM)}")
    print(f"  Device    : {Colors.colorize(str(device), Colors.GREEN)}")

    # Check if model exists
    model_path = Path(args.model) if args.model else OUTPUT_DIR / "td3_model.pt"
    model_exists = model_path.exists()

    if model_exists:
        print(f"  Model     : {Colors.colorize('FOUND', Colors.GREEN)} ({model_path.name})")
    else:
        print(f"  Model     : {Colors.colorize('NOT FOUND', Colors.YELLOW)} (need to train first)")

    print()

    # Create game
    game = BipedalWalkerGame()

    # Override model path if specified
    if args.model:
        game.model_path = Path(args.model)

    # =========================================================================
    # DIRECT MODE (command line arguments)
    # =========================================================================

    if args.watch:
        # Watch mode - load and run trained model
        if not model_exists:
            print(Colors.colorize("ERROR: No trained model found!", Colors.RED))
            print(f"Train first with: python {Path(__file__).name} --train")
            return

        print(Colors.colorize("  Mode: WATCH (AI Playing)", Colors.GREEN + Colors.BOLD))
        print("=" * 60)
        game.run_ai_mode(episodes=args.episodes)
        return

    elif args.train:
        # Training mode
        episodes = args.episodes if args.episodes != 5 else 1000  # Default 1000 for training
        print(Colors.colorize(f"  Mode: TRAINING ({episodes} episodes)", Colors.YELLOW + Colors.BOLD))
        print("=" * 60)
        game.run_training_mode(episodes=episodes)
        return

    elif args.human:
        # Human mode
        print(Colors.colorize("  Mode: HUMAN (Keyboard Control)", Colors.CYAN + Colors.BOLD))
        print("=" * 60)
        game.run_human_mode()
        return

    elif args.demo:
        # Demo mode
        print(Colors.colorize("  Mode: DEMO (Quick Train + Watch)", Colors.HEADER + Colors.BOLD))
        print("=" * 60)
        game.demo_mode()
        return

    # =========================================================================
    # INTERACTIVE MENU (no arguments)
    # =========================================================================

    print("  This robot must learn to walk using 4 motors.")
    print("  It uses TD3 algorithm for continuous control.")

    while True:
        print_menu()
        try:
            choice = input("Your choice (1-5): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if choice == "1":
            game.run_human_mode()
        elif choice == "2":
            game.run_ai_mode(episodes=5)
        elif choice == "3":
            try:
                episodes = int(input("Number of episodes (default 1000): ").strip() or "1000")
            except ValueError:
                episodes = 1000
            game.run_training_mode(episodes=episodes)
        elif choice == "4":
            game.demo_mode()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")

    game.env.close()


if __name__ == "__main__":
    main()
