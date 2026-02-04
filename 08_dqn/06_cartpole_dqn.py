"""
CartPole DQN - Tam Implementasyon
=================================
Bu script, tum DQN tekniklerini birlestiren tam bir
CartPole cozumu sunar:
- Experience Replay
- Target Network
- Double DQN
- Dueling Architecture (opsiyonel)

CartPole-v1:
    - Durum: [konum, hiz, aci, acisal_hiz]
    - Aksiyonlar: 0 (sol), 1 (sag)
    - Odul: Her adim +1
    - Bitis: Cubuk duserse veya 500 adim
    - Cozum: 100 episode ortalamasi >= 475

Kullanim:
    python 06_cartpole_dqn.py              # Egit
    python 06_cartpole_dqn.py --watch      # Egitilmis modeli izle
    python 06_cartpole_dqn.py --dueling    # Dueling DQN ile egit
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random
from typing import List, Tuple, Optional
import gymnasium as gym
from pathlib import Path
import argparse
import time


# =============================================================================
# YAPILANDIRMA
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "cartpole_dqn_model.pt"


class Config:
    """Egitim konfigurasyonu."""
    # Ortam
    env_name = "CartPole-v1"

    # Ag
    hidden_size = 128
    use_dueling = False

    # Egitim
    episodes = 500
    batch_size = 64
    gamma = 0.99
    lr = 0.001

    # Epsilon
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    # Replay Buffer
    buffer_size = 100000
    min_buffer_size = 1000

    # Target Network
    target_update_freq = 100
    use_soft_update = False
    tau = 0.005

    # Double DQN
    use_double = True


# =============================================================================
# NEURAL NETWORK MODELLERI
# =============================================================================

class DQN(nn.Module):
    """Standart DQN agi."""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.network(x)


class DuelingDQN(nn.Module):
    """Dueling DQN agi."""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """Experience Replay Buffer."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# DQN AJAN
# =============================================================================

class DQNAgent:
    """
    Tam ozellikli DQN Ajani.

    Ozellikler:
    - Experience Replay
    - Target Network (hard veya soft update)
    - Double DQN (opsiyonel)
    - Dueling Architecture (opsiyonel)
    """

    def __init__(self, state_size: int, action_size: int, config: Config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        # Cihaz
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Aglar
        NetworkClass = DuelingDQN if config.use_dueling else DQN
        self.policy_net = NetworkClass(state_size, action_size, config.hidden_size).to(self.device)
        self.target_net = NetworkClass(state_size, action_size, config.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)

        # Replay Buffer
        self.memory = ReplayBuffer(capacity=config.buffer_size)

        # Epsilon
        self.epsilon = config.epsilon_start

        # Adim sayaci
        self.steps = 0

        # Istatistikler
        self.losses = []

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy aksiyon secimi."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Deneyimi bellege kaydet."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """Bir egitim adimi."""
        if len(self.memory) < self.config.min_buffer_size:
            return None

        # Ornekle
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)

        # Tensore donustur
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Hedef Q degerleri
        with torch.no_grad():
            if self.config.use_double:
                # Double DQN: policy ile sec, target ile degerlendir
                next_actions = self.policy_net(next_states_t).argmax(dim=1)
                next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                # Standard DQN
                next_q = self.target_net(next_states_t).max(dim=1)[0]

            targets = rewards_t + self.config.gamma * next_q * (1 - dones_t)

        # Mevcut Q degerleri
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

        # Loss
        loss = nn.SmoothL1Loss()(current_q, targets)  # Huber loss

        # Geri yayilim
        self.optimizer.zero_grad()
        loss.backward()
        # Gradyan kirpma (stabilite icin)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Adim sayacini artir
        self.steps += 1

        # Target network guncelleme
        if self.config.use_soft_update:
            self._soft_update()
        elif self.steps % self.config.target_update_freq == 0:
            self._hard_update()

        self.losses.append(loss.item())
        return loss.item()

    def _hard_update(self):
        """Hard target network guncellemesi."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _soft_update(self):
        """Soft target network guncellemesi."""
        tau = self.config.tau
        for target_param, policy_param in zip(self.target_net.parameters(),
                                               self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

    def decay_epsilon(self):
        """Epsilon'u azalt."""
        self.epsilon = max(self.config.epsilon_end,
                          self.epsilon * self.config.epsilon_decay)

    def save(self, path: Path):
        """Modeli kaydet."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'config': vars(self.config)
        }, path)
        print(f"Model kaydedildi: {path}")

    def load(self, path: Path) -> bool:
        """Modeli yukle."""
        if not path.exists():
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.config.epsilon_end)
        self.steps = checkpoint.get('steps', 0)
        print(f"Model yuklendi: {path}")
        return True


# =============================================================================
# EGITIM
# =============================================================================

def train(config: Config) -> List[float]:
    """DQN ajanini egit."""
    print("\n" + "="*70)
    print("DQN EGITIMI BASLIYOR")
    print("="*70)

    # Ortam
    env = gym.make(config.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Ajan
    agent = DQNAgent(state_size, action_size, config)

    # Konfigurasyonu yazdir
    print(f"\nKonfigurasyon:")
    print(f"  Ortam:          {config.env_name}")
    print(f"  Dueling:        {config.use_dueling}")
    print(f"  Double DQN:     {config.use_double}")
    print(f"  Soft Update:    {config.use_soft_update}")
    print(f"  Cihaz:          {agent.device}")
    print(f"  Episode:        {config.episodes}")
    print(f"  Batch Size:     {config.batch_size}")
    print(f"  Learning Rate:  {config.lr}")
    print(f"  Gamma:          {config.gamma}")
    print()

    # Egitim dongusu
    rewards_history = []
    best_avg = 0
    start_time = time.time()

    print(f"{'Episode':>8} | {'Reward':>8} | {'Avg(100)':>10} | {'Epsilon':>8} | {'Loss':>10} | {'Buffer':>8}")
    print("-"*70)

    for episode in range(1, config.episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        episode_losses = []

        while not done:
            # Aksiyon sec
            action = agent.choose_action(state, training=True)

            # Adim at
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Bellege kaydet
            agent.remember(state, action, reward, next_state, done)

            # Egit
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            total_reward += reward
            state = next_state

        # Epsilon azalt
        agent.decay_epsilon()

        # Istatistikler
        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-100:])
        avg_loss = np.mean(episode_losses) if episode_losses else 0

        # En iyi modeli kaydet
        if avg_reward > best_avg and episode > 100:
            best_avg = avg_reward
            agent.save(MODEL_PATH)

        # Ilerleme raporu
        if episode % 10 == 0:
            print(f"{episode:>8} | {total_reward:>8.0f} | {avg_reward:>10.1f} | "
                  f"{agent.epsilon:>8.4f} | {avg_loss:>10.4f} | {len(agent.memory):>8}")

        # Cozum kontrolu
        if avg_reward >= 475:
            print(f"\n{'='*70}")
            print(f"COZULDU! Episode {episode}'de ortalama odul {avg_reward:.1f}")
            print(f"{'='*70}")
            agent.save(MODEL_PATH)
            break

    # Son kayit
    agent.save(MODEL_PATH)

    # Egitim ozeti
    elapsed = time.time() - start_time
    print(f"\nEgitim Tamamlandi!")
    print(f"  Toplam Sure:    {elapsed/60:.1f} dakika")
    print(f"  Son 100 Ort.:   {np.mean(rewards_history[-100:]):.1f}")
    print(f"  En Iyi Ort.:    {best_avg:.1f}")
    print(f"  Max Episode:    {max(rewards_history):.0f}")

    env.close()

    # Gorsellestir
    plot_training(rewards_history, agent.losses)

    return rewards_history


def plot_training(rewards: List[float], losses: List[float]):
    """Egitim sonuclarini gorsellestir."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Odul grafiği
    axes[0].plot(rewards, alpha=0.6, label='Episode Odulu')
    if len(rewards) >= 100:
        moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        axes[0].plot(range(99, len(rewards)), moving_avg,
                    color='red', linewidth=2, label='100 Episode Ortalamasi')
    axes[0].axhline(y=475, color='green', linestyle='--', alpha=0.7, label='Cozum Esigi (475)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Toplam Odul')
    axes[0].set_title('Egitim Ilerlemesi')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss grafigi
    if losses:
        # Cok fazla nokta varsa, ornekle
        step = max(1, len(losses) // 1000)
        sampled_losses = losses[::step]
        axes[1].plot(range(0, len(losses), step), sampled_losses, alpha=0.6)
        axes[1].set_xlabel('Egitim Adimi')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Egitim Loss')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / '06_training_results.png', dpi=150)
    plt.show()
    print(f"Grafik kaydedildi: {SCRIPT_DIR / '06_training_results.png'}")


# =============================================================================
# IZLEME
# =============================================================================

def watch(config: Config, episodes: int = 5):
    """Egitilmis ajani izle."""
    print("\n" + "="*70)
    print("EGITILMIS AJANIN PERFORMANSI")
    print("="*70)

    # Ortam (render ile)
    env = gym.make(config.env_name, render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Ajan
    agent = DQNAgent(state_size, action_size, config)

    # Modeli yukle
    if not agent.load(MODEL_PATH):
        print(f"Model bulunamadi: {MODEL_PATH}")
        print("Once egitim yapin: python 06_cartpole_dqn.py")
        return

    # Epsilon'u sifirla (tam somuru)
    agent.epsilon = 0

    rewards = []
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {ep}: Reward = {total_reward:.0f}")

    print(f"\nOrtalama: {np.mean(rewards):.1f}")
    env.close()


# =============================================================================
# ANA PROGRAM
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CartPole DQN")
    parser.add_argument("--watch", "-w", action="store_true",
                       help="Egitilmis modeli izle")
    parser.add_argument("--episodes", "-n", type=int, default=5,
                       help="Izleme icin episode sayisi")
    parser.add_argument("--dueling", action="store_true",
                       help="Dueling DQN kullan")
    parser.add_argument("--train-episodes", type=int, default=500,
                       help="Egitim episode sayisi")
    args = parser.parse_args()

    # Konfigurasyon
    config = Config()
    config.use_dueling = args.dueling
    config.episodes = args.train_episodes

    if args.watch:
        watch(config, args.episodes)
    else:
        print("""
╔═══════════════════════════════════════════════════════════════════╗
║                     CARTPOLE DQN EGITIMI                          ║
╠═══════════════════════════════════════════════════════════════════╣
║  Bu script, DQN'in tum bilesenlerini birlestiren                  ║
║  tam bir CartPole cozumu sunar.                                   ║
║                                                                   ║
║  Ozellikler:                                                      ║
║  - Experience Replay (korelasyonu kirar)                          ║
║  - Target Network (kararli hedefler)                              ║
║  - Double DQN (asiri tahmin azaltma)                              ║
║  - Dueling Architecture (opsiyonel)                               ║
║                                                                   ║
║  Kullanim:                                                        ║
║    python 06_cartpole_dqn.py              # Egit                  ║
║    python 06_cartpole_dqn.py --watch      # Izle                  ║
║    python 06_cartpole_dqn.py --dueling    # Dueling ile egit      ║
╚═══════════════════════════════════════════════════════════════════╝
        """)
        train(config)


if __name__ == "__main__":
    main()
