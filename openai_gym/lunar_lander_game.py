"""
Lunar Lander - Real-Time Game Simulation
=========================================
This example uses the LunarLander-v3 environment to create an interactive game.

Modes:
1. Human Mode: Control with keyboard
2. AI Mode: Watch trained DQN agent play
3. Training Mode: Train the DQN agent
4. Demo Mode: Quick training + AI demonstration

Controls (Human Mode):
- Left Arrow / A: Left engine
- Right Arrow / D: Right engine
- Up Arrow / W / Space: Main engine
- R: Restart episode
- Q / ESC: Quit game
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

# =============================================================================
# CONFIGURATION
# =============================================================================

# Output directory - where we save models and logs
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = Path(__file__).parent / "output" / SCRIPT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Device selection - use GPU if available, otherwise CPU
# GPU makes training much faster (10x or more)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# DEEP Q-NETWORK (DQN) MODEL
# =============================================================================

class DQN(nn.Module):
    """
    Deep Q-Network - A neural network that learns Q-values.

    Q-value = expected future reward for taking an action in a state.
    Higher Q-value = better action to take.

    Architecture:
        Input (8 features) -> Hidden (128) -> Hidden (128) -> Output (4 actions)

    The 8 input features are:
        - x position, y position
        - x velocity, y velocity
        - angle, angular velocity
        - left leg contact, right leg contact

    The 4 output actions are:
        - 0: Do nothing
        - 1: Fire left engine
        - 2: Fire main engine
        - 3: Fire right engine
    """

    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()

        # Three fully connected layers
        # fc1: input -> hidden (learns basic features)
        # fc2: hidden -> hidden (learns complex patterns)
        # fc3: hidden -> output (produces Q-values for each action)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        # ReLU activation: if x < 0, output 0; else output x
        # This adds non-linearity so network can learn complex patterns
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # No activation on output - Q-values can be any number
        return self.fc3(x)


# =============================================================================
# REPLAY BUFFER (EXPERIENCE MEMORY)
# =============================================================================

class ReplayBuffer:
    """
    Replay Buffer - Stores past experiences for training.

    Why we need this:
        1. Breaks correlation between consecutive samples
        2. Allows reusing rare experiences multiple times
        3. Makes training more stable and efficient

    Each experience is a tuple: (state, action, reward, next_state, done)
    """

    def __init__(self, capacity=100000):
        # deque = double-ended queue, automatically removes old items
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences.
        Random sampling breaks correlation between consecutive experiences.
        """
        batch = random.sample(self.buffer, batch_size)
        # Unzip the batch into separate arrays
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# DQN AGENT
# =============================================================================

class DQNAgent:
    """
    DQN Agent - Learns to play the game using Deep Q-Learning.

    Key concepts:
        - Policy Network: Makes decisions (which action to take)
        - Target Network: Provides stable Q-value targets for training
        - Epsilon-Greedy: Balance between exploration and exploitation

    Training loop:
        1. Observe state
        2. Choose action (epsilon-greedy)
        3. Take action, get reward and next state
        4. Store experience in replay buffer
        5. Sample random batch from buffer
        6. Calculate loss and update policy network
        7. Periodically copy policy network to target network
    """

    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):

        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = gamma              # Discount factor (0.99 = care about future)
        self.epsilon = epsilon          # Exploration rate (1.0 = 100% random)
        self.epsilon_decay = epsilon_decay  # How fast to reduce exploration
        self.epsilon_min = epsilon_min  # Minimum exploration rate

        # Two networks: policy (learning) and target (stable)
        # Why two networks? To avoid "chasing a moving target" problem
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Adam optimizer - adjusts learning rate automatically
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Experience replay buffer
        self.memory = ReplayBuffer()
        self.batch_size = 64

        # Update target network every N steps
        self.update_target_every = 10
        self.steps = 0

    def choose_action(self, state, training=True):
        """
        Choose an action using epsilon-greedy strategy.

        Epsilon-greedy:
            - With probability epsilon: take random action (explore)
            - With probability 1-epsilon: take best action (exploit)

        As training progresses, epsilon decreases, so agent explores less
        and exploits learned knowledge more.
        """
        # Exploration: random action
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Exploitation: best action according to policy network
        with torch.no_grad():  # No need to track gradients for inference
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()  # Return action with highest Q-value

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Perform one training step using a batch from replay buffer.

        The DQN loss function:
            Loss = (Q_predicted - Q_target)^2

        Where:
            Q_predicted = policy_net(state)[action]
            Q_target = reward + gamma * max(target_net(next_state))

        This is the Bellman equation - the foundation of Q-learning.
        """
        # Need enough samples to form a batch
        if len(self.memory) < self.batch_size:
            return 0

        # Sample random batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors and move to GPU/CPU
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Calculate current Q values from policy network
        # gather() selects the Q-value for the action that was taken
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Calculate target Q values from target network
        with torch.no_grad():  # Don't update target network here
            # max(Q(next_state)) - best possible future value
            next_q = self.target_net(next_states).max(1)[0]
            # Bellman equation: Q = reward + gamma * max(Q_next)
            # If done, there's no future reward
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Mean Squared Error loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Backpropagation - update network weights
        self.optimizer.zero_grad()  # Clear old gradients
        loss.backward()             # Calculate new gradients
        self.optimizer.step()       # Update weights

        # Decay epsilon - explore less over time
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Periodically update target network
        # This provides stable targets for training
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        """Save model to file."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved: {path}")

    def load(self, path):
        """Load model from file."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.epsilon = checkpoint.get('epsilon', 0.01)
            print(f"Model loaded: {path}")
            return True
        return False


# =============================================================================
# GAME CLASS
# =============================================================================

class LunarLanderGame:
    """
    Main game class - handles rendering, input, and game loop.

    The LunarLander environment:
        - Goal: Land the spacecraft between the two flags
        - Reward: +100 to +140 for landing, -100 for crashing
        - Fuel usage gives small negative reward
        - Each leg contact gives +10
        - Episode ends when: landed, crashed, or 1000 steps
        - Solved when average reward >= 200 over 100 episodes
    """

    # Human-readable action names
    ACTION_NAMES = ["Nothing", "Left engine", "Main engine", "Right engine"]

    def __init__(self):
        # Create the game environment with visual rendering
        self.env = gym.make("LunarLander-v3", render_mode="human")

        # State space: 8 continuous values (position, velocity, angle, etc.)
        self.state_size = self.env.observation_space.shape[0]
        # Action space: 4 discrete actions (nothing, left, main, right)
        self.action_size = self.env.action_space.n

        # Create the DQN agent
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.model_path = OUTPUT_DIR / "dqn_model.pt"

        # Game state variables
        self.state = None
        self.total_reward = 0
        self.episode = 0
        self.step_count = 0
        self.done = True
        self.mode = "human"  # "human", "ai", or "train"

        # Statistics tracking
        self.rewards_history = []
        self.best_reward = float('-inf')

    def reset(self):
        """Reset environment for a new episode."""
        self.state, _ = self.env.reset()
        self.total_reward = 0
        self.step_count = 0
        self.done = False
        self.episode += 1
        return self.state

    def get_human_action(self, keys):
        """
        Convert keyboard input to game action.

        Mapping:
            Up/W/Space -> Main engine (slow descent)
            Left/A -> Left engine (push right)
            Right/D -> Right engine (push left)
            Nothing -> Do nothing (fall)
        """
        if keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]:
            return 2  # Main engine
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            return 1  # Left engine
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            return 3  # Right engine
        return 0  # Do nothing

    def step(self, action):
        """
        Execute one game step.

        This is the core game loop:
            1. Take action in environment
            2. Get reward and new state
            3. If training, store experience and learn
            4. Update game state
        """
        # Execute action in environment
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Update statistics
        self.total_reward += reward
        self.step_count += 1

        # If training mode, learn from this experience
        if self.mode == "train":
            self.agent.remember(self.state, action, reward, next_state, done)
            self.agent.train_step()

        # Update state
        self.state = next_state
        self.done = done

        # Track episode results
        if done:
            self.rewards_history.append(self.total_reward)
            if self.total_reward > self.best_reward:
                self.best_reward = self.total_reward

        return next_state, reward, done, info

    def run_human_mode(self):
        """
        Human control mode - play with keyboard.

        Tips for landing:
            - Use main engine to slow descent
            - Use side engines to stay centered
            - Try to land softly between the flags
            - Both legs should touch the ground
        """
        print("\n" + "=" * 60)
        print("LUNAR LANDER - HUMAN MODE")
        print("=" * 60)
        print("Controls:")
        print("  - Left Arrow / A: Left engine")
        print("  - Right Arrow / D: Right engine")
        print("  - Up Arrow / W / Space: Main engine")
        print("  - R: Restart")
        print("  - Q / ESC: Quit")
        print("=" * 60)

        self.mode = "human"
        pygame.init()

        running = True
        clock = pygame.time.Clock()
        self.reset()

        while running:
            # Handle events (quit, restart, etc.)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset()
                        print(f"\n[Restarted] Episode {self.episode}")

            # If game not over, process input and step
            if not self.done:
                keys = pygame.key.get_pressed()
                action = self.get_human_action(keys)
                _, reward, done, _ = self.step(action)

                if done:
                    # Score > 200 means successful landing
                    result = "SUCCESS!" if self.total_reward > 200 else "CRASHED!"
                    print(f"Episode {self.episode}: {result} Score: {self.total_reward:.1f}")
            else:
                # Wait a bit then restart
                pygame.time.wait(1000)
                self.reset()

            # Limit to 30 FPS
            clock.tick(30)

        self.env.close()
        pygame.quit()

    def run_ai_mode(self, episodes=5):
        """
        AI control mode - watch trained agent play.

        The trained agent uses its learned Q-values to make decisions.
        No exploration (epsilon=0), only exploitation of learned knowledge.
        """
        print("\n" + "=" * 60)
        print("LUNAR LANDER - AI MODE")
        print("=" * 60)

        self.mode = "ai"

        # Load trained model
        if not self.agent.load(self.model_path):
            print("No trained model found!")
            print("Run training mode first: run_training_mode()")
            return

        # Disable exploration - use learned policy only
        self.agent.epsilon = 0

        for ep in range(episodes):
            self.reset()
            print(f"\n[AI Playing] Episode {ep + 1}/{episodes}")

            while not self.done:
                # AI chooses action based on learned Q-values
                action = self.agent.choose_action(self.state, training=False)
                self.step(action)
                pygame.time.wait(20)  # Small delay for visualization

            result = "SUCCESS!" if self.total_reward > 200 else "CRASHED!"
            print(f"Result: {result} Score: {self.total_reward:.1f}")

        # Print average score
        avg_reward = np.mean(self.rewards_history[-episodes:])
        print(f"\nAverage Score: {avg_reward:.1f}")

        self.env.close()

    def run_training_mode(self, episodes=500, save_every=50):
        """
        Training mode - train the DQN agent.

        The agent learns by:
            1. Playing many episodes
            2. Storing experiences in replay buffer
            3. Learning from random batches of experiences
            4. Gradually reducing exploration (epsilon decay)

        Training tips:
            - More episodes = better performance (usually)
            - Watch the average reward over last 100 episodes
            - Goal: average reward >= 200 (environment is "solved")
        """
        print("\n" + "=" * 60)
        print("LUNAR LANDER - TRAINING MODE")
        print("=" * 60)
        print(f"Episodes: {episodes}")
        print(f"Save every: {save_every} episodes")
        print(f"Device: {device}")
        print("=" * 60)

        self.mode = "train"

        # Switch to non-rendering mode for faster training
        self.env.close()
        self.env = gym.make("LunarLander-v3")

        best_avg = float('-inf')

        for ep in range(episodes):
            self.reset()

            # Play one episode
            while not self.done:
                action = self.agent.choose_action(self.state, training=True)
                self.step(action)

            # Calculate moving average of last 100 episodes
            avg_reward = np.mean(self.rewards_history[-100:]) if self.rewards_history else 0

            # Print progress every 10 episodes
            if (ep + 1) % 10 == 0:
                print(f"Episode {ep + 1}/{episodes} | "
                      f"Score: {self.total_reward:.1f} | "
                      f"Avg(100): {avg_reward:.1f} | "
                      f"Epsilon: {self.agent.epsilon:.3f}")

            # Save best model
            if avg_reward > best_avg and ep > 100:
                best_avg = avg_reward
                self.agent.save(self.model_path)

            # Periodic save
            if (ep + 1) % save_every == 0:
                self.agent.save(self.model_path)

        # Final save
        self.agent.save(self.model_path)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print(f"Best average: {best_avg:.1f}")
        print(f"Model saved: {self.model_path}")
        print("=" * 60)

        self.env.close()
        return self.rewards_history

    def demo_mode(self):
        """
        Demo mode - quick training followed by AI demonstration.

        Good for seeing results quickly without long training.
        """
        print("\n" + "=" * 60)
        print("LUNAR LANDER - DEMO MODE")
        print("=" * 60)

        # Quick training
        print("\nStarting quick training (200 episodes)...")
        self.run_training_mode(episodes=200, save_every=50)

        # AI demonstration
        print("\nStarting AI demonstration...")
        self.env = gym.make("LunarLander-v3", render_mode="human")
        self.run_ai_mode(episodes=3)


# =============================================================================
# MAIN MENU
# =============================================================================

def print_menu():
    """Display the main menu."""
    print("\n" + "=" * 60)
    print("LUNAR LANDER - MAIN MENU")
    print("=" * 60)
    print("1. Human Mode (Play with keyboard)")
    print("2. AI Mode (Watch trained agent)")
    print("3. Training Mode (Train the agent)")
    print("4. Demo Mode (Quick train + watch)")
    print("5. Exit")
    print("=" * 60)


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("LUNAR LANDER - Real-Time Game Simulation")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Device: {device}")

    game = LunarLanderGame()

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
                episodes = int(input("Number of episodes (default 500): ").strip() or "500")
            except ValueError:
                episodes = 500
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
