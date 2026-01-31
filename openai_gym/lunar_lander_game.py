"""
Lunar Lander - Gerçek Zamanlı Oyun Simülasyonu
===============================================
Bu örnek, LunarLander-v3 ortamını kullanarak interaktif bir oyun sunar.

Modlar:
1. İnsan Modu: Klavye ile kontrol et
2. AI Modu: Eğitilmiş DQN ajanını izle
3. Eğitim Modu: DQN ajanını eğit

Kontroller (İnsan Modu):
- Sol Ok / A: Sol motor
- Sağ Ok / D: Sağ motor
- Yukarı Ok / W / Space: Ana motor
- R: Yeniden başlat
- Q / ESC: Çıkış
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

# Output klasörü
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = Path(__file__).parent / "output" / SCRIPT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    """Deep Q-Network modeli."""

    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Deneyim tekrar tamponu."""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN tabanlı ajan."""

    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Ana ve hedef ağlar
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        self.batch_size = 64
        self.update_target_every = 10
        self.steps = 0

    def choose_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Mevcut Q değerleri
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Hedef Q değerleri
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Kayıp ve güncelleme
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon azalt
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Hedef ağı güncelle
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Model kaydedildi: {path}")

    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.epsilon = checkpoint.get('epsilon', 0.01)
            print(f"Model yüklendi: {path}")
            return True
        return False


class LunarLanderGame:
    """Gerçek zamanlı Lunar Lander oyunu."""

    ACTION_NAMES = ["Hiçbir şey", "Sol motor", "Ana motor", "Sağ motor"]

    def __init__(self):
        self.env = gym.make("LunarLander-v3", render_mode="human")
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        # DQN ajanı
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.model_path = OUTPUT_DIR / "dqn_model.pt"

        # Oyun durumu
        self.state = None
        self.total_reward = 0
        self.episode = 0
        self.step_count = 0
        self.done = True
        self.mode = "human"  # "human", "ai", "train"

        # İstatistikler
        self.rewards_history = []
        self.best_reward = float('-inf')

    def reset(self):
        """Ortamı sıfırla."""
        self.state, _ = self.env.reset()
        self.total_reward = 0
        self.step_count = 0
        self.done = False
        self.episode += 1
        return self.state

    def get_human_action(self, keys):
        """Klavye girişinden aksiyon al."""
        # 0: Hiçbir şey, 1: Sol, 2: Ana motor, 3: Sağ
        if keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]:
            return 2  # Ana motor
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            return 1  # Sol motor
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            return 3  # Sağ motor
        return 0  # Hiçbir şey

    def step(self, action):
        """Bir adım at."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.total_reward += reward
        self.step_count += 1

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
        """İnsan kontrolü ile oyna."""
        print("\n" + "=" * 60)
        print("🚀 LUNAR LANDER - İNSAN MODU")
        print("=" * 60)
        print("Kontroller:")
        print("  - Sol Ok / A: Sol motor")
        print("  - Sağ Ok / D: Sağ motor")
        print("  - Yukarı Ok / W / Space: Ana motor")
        print("  - R: Yeniden başlat")
        print("  - Q / ESC: Çıkış")
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
                        print(f"\n[Yeniden başlatıldı] Episode {self.episode}")

            if not self.done:
                keys = pygame.key.get_pressed()
                action = self.get_human_action(keys)
                _, reward, done, _ = self.step(action)

                if done:
                    result = "✅ BAŞARILI!" if self.total_reward > 200 else "❌ DÜŞTÜ!"
                    print(f"Episode {self.episode}: {result} Skor: {self.total_reward:.1f}")
            else:
                # Otomatik yeniden başlat
                pygame.time.wait(1000)
                self.reset()

            clock.tick(30)

        self.env.close()
        pygame.quit()

    def run_ai_mode(self, episodes=5):
        """Eğitilmiş AI ile oyna."""
        print("\n" + "=" * 60)
        print("🤖 LUNAR LANDER - AI MODU")
        print("=" * 60)

        self.mode = "ai"

        # Modeli yükle
        if not self.agent.load(self.model_path):
            print("⚠️  Eğitilmiş model bulunamadı!")
            print("Önce eğitim modunu çalıştırın: run_training_mode()")
            return

        self.agent.epsilon = 0  # Keşif yok, sadece sömürü

        for ep in range(episodes):
            self.reset()
            print(f"\n[AI Oynuyor] Episode {ep + 1}/{episodes}")

            while not self.done:
                action = self.agent.choose_action(self.state, training=False)
                self.step(action)
                pygame.time.wait(20)

            result = "✅ BAŞARILI!" if self.total_reward > 200 else "❌ DÜŞTÜ!"
            print(f"Sonuç: {result} Skor: {self.total_reward:.1f}")

        avg_reward = np.mean(self.rewards_history[-episodes:])
        print(f"\n📊 Ortalama Skor: {avg_reward:.1f}")

        self.env.close()

    def run_training_mode(self, episodes=500, save_every=50):
        """AI ajanını eğit."""
        print("\n" + "=" * 60)
        print("🎓 LUNAR LANDER - EĞİTİM MODU")
        print("=" * 60)
        print(f"Eğitim: {episodes} episode")
        print(f"Kayıt: Her {save_every} episode")
        print(f"Cihaz: {device}")
        print("=" * 60)

        self.mode = "train"

        # Render olmadan eğit (hızlı)
        self.env.close()
        self.env = gym.make("LunarLander-v3")

        best_avg = float('-inf')

        for ep in range(episodes):
            self.reset()

            while not self.done:
                action = self.agent.choose_action(self.state, training=True)
                self.step(action)

            # İstatistikler
            avg_reward = np.mean(self.rewards_history[-100:]) if self.rewards_history else 0

            if (ep + 1) % 10 == 0:
                print(f"Episode {ep + 1}/{episodes} | "
                      f"Skor: {self.total_reward:.1f} | "
                      f"Ort(100): {avg_reward:.1f} | "
                      f"ε: {self.agent.epsilon:.3f}")

            # En iyi modeli kaydet
            if avg_reward > best_avg and ep > 100:
                best_avg = avg_reward
                self.agent.save(self.model_path)

            # Periyodik kayıt
            if (ep + 1) % save_every == 0:
                self.agent.save(self.model_path)

        # Son kayıt
        self.agent.save(self.model_path)

        print("\n" + "=" * 60)
        print("✅ EĞİTİM TAMAMLANDI!")
        print(f"En iyi ortalama: {best_avg:.1f}")
        print(f"Model: {self.model_path}")
        print("=" * 60)

        self.env.close()
        return self.rewards_history

    def demo_mode(self):
        """Hızlı demo - eğitim + AI gösterimi."""
        print("\n" + "=" * 60)
        print("🎮 LUNAR LANDER - DEMO MODU")
        print("=" * 60)

        # Kısa eğitim
        print("\n📚 Hızlı eğitim başlıyor (200 episode)...")
        self.run_training_mode(episodes=200, save_every=50)

        # AI gösterimi
        print("\n🤖 Eğitilmiş AI gösterimi...")
        self.env = gym.make("LunarLander-v3", render_mode="human")
        self.run_ai_mode(episodes=3)


def print_menu():
    """Ana menüyü göster."""
    print("\n" + "=" * 60)
    print("🚀 LUNAR LANDER - ANA MENÜ")
    print("=" * 60)
    print("1. 👤 İnsan Modu (Klavye ile oyna)")
    print("2. 🤖 AI Modu (Eğitilmiş ajanı izle)")
    print("3. 🎓 Eğitim Modu (AI ajanını eğit)")
    print("4. 🎮 Demo Modu (Eğitim + Gösterim)")
    print("5. ❌ Çıkış")
    print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("🌙 LUNAR LANDER - Gerçek Zamanlı Oyun Simülasyonu")
    print("=" * 60)
    print(f"Output klasörü: {OUTPUT_DIR}")
    print(f"Cihaz: {device}")

    game = LunarLanderGame()

    while True:
        print_menu()
        try:
            choice = input("Seçiminiz (1-5): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÇıkılıyor...")
            break

        if choice == "1":
            game.run_human_mode()
        elif choice == "2":
            game.run_ai_mode(episodes=5)
        elif choice == "3":
            try:
                episodes = int(input("Episode sayısı (varsayılan 500): ").strip() or "500")
            except ValueError:
                episodes = 500
            game.run_training_mode(episodes=episodes)
        elif choice == "4":
            game.demo_mode()
        elif choice == "5":
            print("Güle güle! 👋")
            break
        else:
            print("Geçersiz seçim!")

    game.env.close()


if __name__ == "__main__":
    main()
