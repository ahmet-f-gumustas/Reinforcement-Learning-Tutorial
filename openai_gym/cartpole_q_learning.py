"""
CartPole Q-Learning Example
===========================
Bu örnek, Q-Learning algoritması kullanarak CartPole-v1 ortamını çözmeyi gösterir.
Sürekli durum uzayı, ayrık kutulara (bins) bölünerek Q-tablosu oluşturulur.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class CartPoleQLearning:
    def __init__(self, n_bins=20, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = gym.make("CartPole-v1")
        self.n_bins = n_bins
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Durum uzayı sınırları
        self.state_bounds = [
            (-4.8, 4.8),      # cart position
            (-4, 4),          # cart velocity
            (-0.418, 0.418),  # pole angle
            (-4, 4)           # pole angular velocity
        ]

        # Q-tablosu: (bins^4) x 2 actions
        self.q_table = np.zeros([n_bins] * 4 + [self.env.action_space.n])

    def discretize_state(self, state):
        """Sürekli durumu ayrık kutulara dönüştür."""
        discrete_state = []
        for i, val in enumerate(state):
            low, high = self.state_bounds[i]
            val = np.clip(val, low, high)
            bin_idx = int((val - low) / (high - low) * (self.n_bins - 1))
            discrete_state.append(bin_idx)
        return tuple(discrete_state)

    def choose_action(self, state):
        """Epsilon-greedy aksiyon seçimi."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, done):
        """Q-değerini güncelle."""
        current_q = self.q_table[state + (action,)]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state + (action,)] = current_q + self.lr * (target_q - current_q)

    def decay_epsilon(self):
        """Epsilon değerini azalt."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, n_episodes=1000, verbose=True):
        """Ajanı eğit."""
        rewards_history = []
        avg_rewards = []

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.discretize_state(next_state)

                self.update_q_table(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            self.decay_epsilon()
            rewards_history.append(total_reward)

            # Son 100 episode ortalaması
            avg_reward = np.mean(rewards_history[-100:])
            avg_rewards.append(avg_reward)

            if verbose and (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Avg Reward (last 100): {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f}")

        return rewards_history, avg_rewards

    def test(self, n_episodes=10, render=False):
        """Eğitilmiş ajanı test et."""
        if render:
            self.env = gym.make("CartPole-v1", render_mode="human")

        total_rewards = []

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            total_reward = 0
            done = False

            while not done:
                action = np.argmax(self.q_table[state])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = self.discretize_state(next_state)
                total_reward += reward

            total_rewards.append(total_reward)
            print(f"Test Episode {episode + 1}: Reward = {total_reward}")

        print(f"\nOrtalama Test Reward: {np.mean(total_rewards):.2f}")
        self.env.close()
        return total_rewards

    def plot_training(self, rewards, avg_rewards):
        """Eğitim grafiği çiz."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(rewards, alpha=0.6, label='Episode Reward')
        ax1.plot(avg_rewards, color='red', linewidth=2, label='Avg (100 ep)')
        ax1.axhline(y=195, color='green', linestyle='--', label='Solved (195)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(avg_rewards, color='red', linewidth=2)
        ax2.axhline(y=195, color='green', linestyle='--', label='Solved (195)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward (100 episodes)')
        ax2.set_title('Moving Average Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150)
        plt.show()


def main():
    print("=" * 60)
    print("CartPole Q-Learning Example")
    print("=" * 60)

    # Ajanı oluştur
    agent = CartPoleQLearning(
        n_bins=20,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    # Eğit
    print("\nEğitim başlıyor...\n")
    rewards, avg_rewards = agent.train(n_episodes=1000, verbose=True)

    # Grafiği çiz
    print("\nEğitim grafiği çiziliyor...")
    agent.plot_training(rewards, avg_rewards)

    # Test et
    print("\n" + "=" * 60)
    print("Test aşaması (render=False)")
    print("=" * 60)
    agent.test(n_episodes=10, render=False)

    # Görsel test (isteğe bağlı)
    # print("\nGörsel test için render=True yapabilirsiniz")
    # agent.test(n_episodes=3, render=True)


if __name__ == "__main__":
    main()
