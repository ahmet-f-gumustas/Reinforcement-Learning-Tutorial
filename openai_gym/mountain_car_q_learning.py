"""
MountainCar Q-Learning Example
==============================
Klasik MountainCar problemi: Araç, tepeye ulaşmak için momentum kazanmalı.
Doğrudan tepeye tırmanamaz, sağa-sola sallanarak ivme kazanması gerekir.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
from pathlib import Path

# Output klasörü
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = Path(__file__).parent / "output" / SCRIPT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class MountainCarAgent:
    def __init__(self, n_bins=40, learning_rate=0.2, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        self.env = gym.make("MountainCar-v0")
        self.n_bins = n_bins
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Durum uzayı: [pozisyon, hız]
        self.pos_bins = np.linspace(-1.2, 0.6, n_bins)
        self.vel_bins = np.linspace(-0.07, 0.07, n_bins)

        # Q-tablosu
        self.q_table = np.zeros((n_bins, n_bins, self.env.action_space.n))

    def discretize(self, state):
        """Sürekli durumu ayrık indekslere dönüştür."""
        pos_idx = np.digitize(state[0], self.pos_bins) - 1
        vel_idx = np.digitize(state[1], self.vel_bins) - 1
        pos_idx = np.clip(pos_idx, 0, self.n_bins - 1)
        vel_idx = np.clip(vel_idx, 0, self.n_bins - 1)
        return pos_idx, vel_idx

    def choose_action(self, state):
        """Epsilon-greedy politika."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """Q-Learning güncellemesi."""
        current_q = self.q_table[state][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (target_q - current_q)

    def train(self, n_episodes=5000, verbose=True):
        """Eğitim döngüsü."""
        rewards_history = []
        best_reward = -200
        success_count = 0
        recent_rewards = deque(maxlen=100)

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            state = self.discretize(state)
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.discretize(next_obs)

                # Hedefe ulaşınca bonus
                if next_obs[0] >= 0.5:
                    reward = 100
                    success_count += 1

                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            # Epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            rewards_history.append(total_reward)
            recent_rewards.append(total_reward)

            if total_reward > best_reward:
                best_reward = total_reward

            if verbose and (episode + 1) % 500 == 0:
                avg = np.mean(recent_rewards)
                print(f"Episode {episode + 1:5d} | "
                      f"Avg: {avg:7.1f} | "
                      f"Best: {best_reward:7.1f} | "
                      f"Success: {success_count:4d} | "
                      f"Eps: {self.epsilon:.3f}")

        return rewards_history

    def test(self, n_episodes=5, render=False):
        """Test aşaması."""
        if render:
            env = gym.make("MountainCar-v0", render_mode="human")
        else:
            env = self.env

        results = []
        for ep in range(n_episodes):
            state, _ = env.reset()
            state = self.discretize(state)
            total_reward = 0
            steps = 0
            done = False

            while not done:
                action = np.argmax(self.q_table[state])
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = self.discretize(next_obs)
                total_reward += reward
                steps += 1

            success = "Başarılı!" if next_obs[0] >= 0.5 else "Başarısız"
            print(f"Test {ep + 1}: {steps:3d} adım | Reward: {total_reward:6.1f} | {success}")
            results.append((steps, total_reward, next_obs[0] >= 0.5))

        if render:
            env.close()

        success_rate = sum(1 for r in results if r[2]) / len(results) * 100
        print(f"\nBaşarı oranı: {success_rate:.0f}%")
        return results

    def plot_results(self, rewards):
        """Eğitim sonuçlarını görselleştir."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Reward grafiği
        axes[0].plot(rewards, alpha=0.4, color='blue')
        window = 100
        avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
        axes[0].plot(avg, color='red', linewidth=2, label=f'{window}-ep avg')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Training Rewards')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Q-value heatmap (max Q her durum için)
        max_q = np.max(self.q_table, axis=2)
        im = axes[1].imshow(max_q.T, origin='lower', aspect='auto', cmap='viridis')
        axes[1].set_xlabel('Position')
        axes[1].set_ylabel('Velocity')
        axes[1].set_title('Max Q-Value Heatmap')
        plt.colorbar(im, ax=axes[1])

        # Politika haritası
        policy = np.argmax(self.q_table, axis=2)
        actions = ['Sol', 'Dur', 'Sağ']
        colors = ['red', 'gray', 'green']
        im2 = axes[2].imshow(policy.T, origin='lower', aspect='auto', cmap='RdYlGn')
        axes[2].set_xlabel('Position')
        axes[2].set_ylabel('Velocity')
        axes[2].set_title('Learned Policy (0=Sol, 1=Dur, 2=Sağ)')
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'training_results.png', dpi=150)
        print(f"Grafik kaydedildi: {OUTPUT_DIR / 'training_results.png'}")
        plt.show()


def main():
    print("=" * 60)
    print("MountainCar Q-Learning")
    print("=" * 60)
    print(f"Output klasörü: {OUTPUT_DIR}")
    print("\nHedef: Araç tepeye (pozisyon >= 0.5) ulaşmalı")
    print("Aksiyonlar: 0=Sola it, 1=Hiçbir şey, 2=Sağa it\n")

    # Log dosyası
    log_file = OUTPUT_DIR / "training_log.txt"

    agent = MountainCarAgent(
        n_bins=40,
        learning_rate=0.2,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.01
    )

    print("Eğitim başlıyor...")
    print("-" * 60)
    rewards = agent.train(n_episodes=5000, verbose=True)

    print("\n" + "=" * 60)
    print("Test Sonuçları")
    print("=" * 60)
    test_results = agent.test(n_episodes=10, render=False)

    print("\nGrafikler oluşturuluyor...")
    agent.plot_results(rewards)

    # Sonuçları kaydet
    print("\n" + "=" * 60)
    print("Sonuçlar kaydediliyor...")
    print("=" * 60)

    # Q-tablosunu kaydet
    q_table_path = OUTPUT_DIR / "q_table.npy"
    np.save(q_table_path, agent.q_table)
    print(f"Q-tablosu kaydedildi: {q_table_path}")

    # Rewards kaydet
    rewards_path = OUTPUT_DIR / "rewards.npy"
    np.save(rewards_path, np.array(rewards))
    print(f"Rewards kaydedildi: {rewards_path}")

    # Log dosyası
    success_count = sum(1 for r in test_results if r[2])
    with open(log_file, 'w') as f:
        f.write("MountainCar Q-Learning Training Log\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Hiperparametreler:\n")
        f.write(f"  - n_bins: {agent.n_bins}\n")
        f.write(f"  - learning_rate: {agent.lr}\n")
        f.write(f"  - discount_factor: {agent.gamma}\n")
        f.write(f"  - epsilon_decay: {agent.epsilon_decay}\n")
        f.write(f"  - epsilon_min: {agent.epsilon_min}\n\n")
        f.write(f"Eğitim Sonuçları:\n")
        f.write(f"  - Toplam episode: {len(rewards)}\n")
        f.write(f"  - Son 100 episode ortalaması: {np.mean(rewards[-100:]):.2f}\n")
        f.write(f"  - Maksimum reward: {max(rewards):.2f}\n\n")
        f.write(f"Test Sonuçları:\n")
        f.write(f"  - Başarı oranı: {success_count}/{len(test_results)}\n")
    print(f"Log kaydedildi: {log_file}")


if __name__ == "__main__":
    main()
