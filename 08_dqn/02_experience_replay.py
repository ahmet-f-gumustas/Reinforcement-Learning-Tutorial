"""
Experience Replay (Deneyim Tekrari)
===================================
Bu script, Experience Replay mekanizmasini detayli aciklar:
1. Neden gerekli?
2. Nasil calisir?
3. Replay buffer implementasyonu
4. Korelasyon analizi ve gorsellestirme

Experience Replay Nedir?
    Ajanin deneyimlerini bir bellekte saklar ve
    rastgele ornekleyerek egitim yapar.

    Neden onemli?
    - Ardisik deneyimler cok korelasyonlu
    - Korelasyonlu veri ile egitim kararsiz
    - Rastgele ornekleme korelasyonu kirar

Kullanim:
    python 02_experience_replay.py
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


# =============================================================================
# BOLUM 1: REPLAY BUFFER IMPLEMENTASYONU
# =============================================================================

class ReplayBuffer:
    """
    Temel Replay Buffer implementasyonu.

    Deneyim Tuple'i:
        (state, action, reward, next_state, done)

    Ozellikler:
        - FIFO (First-In-First-Out): Eski deneyimler otomatik silinir
        - Rastgele ornekleme: Batch'ler rastgele secilir
        - Verimli bellek kullanimi: deque ile sabit boyut
    """

    def __init__(self, capacity: int = 100000):
        """
        Args:
            capacity: Maksimum deneyim sayisi
        """
        # deque: iki uclu kuyruk, maxlen ile otomatik eski veri silme
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """
        Yeni deneyim ekle.

        Args:
            state: Mevcut durum
            action: Alinan aksiyon
            reward: Alinan odul
            next_state: Sonraki durum
            done: Episode bitti mi?
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Rastgele batch ornekle.

        Neden rastgele?
            - Korelasyonu kirar
            - Her deneyimi esit ihtimalle kullanir
            - Daha kararli ogrenme saglar

        Args:
            batch_size: Orneklenecek deneyim sayisi

        Returns:
            (states, actions, rewards, next_states, dones) tuple'i
        """
        batch = random.sample(self.buffer, batch_size)

        # Batch'i ayri dizilere ayir
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self) -> int:
        """Buffer'daki deneyim sayisi."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Batch orneklemeye hazir mi?"""
        return len(self.buffer) >= batch_size


# =============================================================================
# BOLUM 2: GELISMIS REPLAY BUFFER
# =============================================================================

class PrioritizedReplayBuffer:
    """
    Oncelikli Replay Buffer (Prioritized Experience Replay - PER).

    Fikir:
        Tum deneyimler esit derecede onemli degil!
        TD hatasi yuksek olan deneyimlerden daha cok ogrenebiliriz.

    Oncelik:
        priority_i = |TD_error_i| + epsilon

    Ornekleme olasiligi:
        P(i) = priority_i^alpha / sum(priority^alpha)

    NOT: Bu basitleştirilmiş bir implementasyon.
    Gercek PER icin SumTree veri yapisi kullanilir.
    """

    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        """
        Args:
            capacity: Maksimum deneyim sayisi
            alpha: Onceliklendirme gucü (0=uniform, 1=tam oncelikli)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, td_error: Optional[float] = None):
        """Deneyim ekle, oncelik ata."""
        experience = (state, action, reward, next_state, done)

        # Yeni deneyimler icin maksimum oncelik (kesfedilmemis)
        max_priority = max(self.priorities) if self.priorities else 1.0
        priority = td_error if td_error is not None else max_priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """
        Oncelikli ornekleme.

        Args:
            batch_size: Orneklenecek sayı
            beta: Importance sampling katsayisi (bias duzeltme)

        Returns:
            (states, actions, rewards, next_states, dones, indices, weights)
        """
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Oncelikli indeks secimi
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Importance sampling agirliklari (bias duzeltme)
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        # Deneyimleri topla
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """TD hatalarina gore oncelikleri guncelle."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6  # Sifirdan kacin

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# BOLUM 3: KORELASYON ANALIZI
# =============================================================================

def analyze_correlation(data: List[np.ndarray], window_size: int = 100) -> np.ndarray:
    """
    Ardisik ornekler arasindaki korelasyonu hesapla.

    Korelasyon Nedir?
        Iki degisken arasindaki dogrusal iliski.
        -1 ile +1 arasinda deger alir.
        |korelasyon| > 0.5 genellikle guclu kabul edilir.

    Args:
        data: Durum listesi
        window_size: Pencere boyutu

    Returns:
        Korelasyon dizisi
    """
    if len(data) < window_size:
        return np.array([])

    correlations = []
    for i in range(len(data) - window_size):
        window = np.array(data[i:i + window_size])
        # Her boyut icin ardisik korelasyon
        corr = np.corrcoef(window[:-1].flatten(), window[1:].flatten())[0, 1]
        correlations.append(corr)

    return np.array(correlations)


def compare_sampling_methods(env_name: str = "CartPole-v1", num_samples: int = 1000):
    """
    Sirayla vs Rastgele ornekleme karsilastirmasi.

    Bu fonksiyon, replay buffer'in korelasyonu nasil
    kirdigini gosterir.
    """
    env = gym.make(env_name)

    # Deneyimleri topla
    replay_buffer = ReplayBuffer(capacity=num_samples)
    sequential_states = []

    state, _ = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.push(state, action, reward, next_state, done)
        sequential_states.append(state.copy())

        if done:
            state, _ = env.reset()
        else:
            state = next_state

    env.close()

    # Sirayla ornekleme korelasyonu
    seq_corr = analyze_correlation(sequential_states)

    # Rastgele ornekleme korelasyonu
    random_states = []
    for _ in range(num_samples - 100):
        batch = replay_buffer.sample(100)
        random_states.append(batch[0][0])  # Ilk durumu al

    rand_corr = analyze_correlation(random_states)

    return seq_corr, rand_corr


# =============================================================================
# BOLUM 4: DQN ILE KARSILASTIRMA
# =============================================================================

class SimpleDQN(nn.Module):
    """Basit DQN agi."""
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.network(x)


def train_with_replay(
    env_name: str = "CartPole-v1",
    episodes: int = 200,
    use_replay: bool = True,
    batch_size: int = 32
) -> List[float]:
    """
    Replay buffer ile/olmadan DQN egitimi.

    Args:
        env_name: Gym ortami
        episodes: Toplam episode
        use_replay: Replay buffer kullan?
        batch_size: Mini-batch boyutu

    Returns:
        Episode odulleri listesi
    """
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Ag ve optimizer
    q_network = SimpleDQN(state_size, action_size)
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)

    # Replay buffer (kullanilacaksa)
    replay_buffer = ReplayBuffer(capacity=10000) if use_replay else None

    # Parametreler
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    rewards_history = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy aksiyon
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_network(torch.FloatTensor(state))
                    action = q_values.argmax().item()

            # Adim at
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if use_replay:
                # Replay buffer'a ekle
                replay_buffer.push(state, action, reward, next_state, done)

                # Buffer hazirsa egit
                if replay_buffer.is_ready(batch_size):
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                    # Tensore donustur
                    states_t = torch.FloatTensor(states)
                    actions_t = torch.LongTensor(actions)
                    rewards_t = torch.FloatTensor(rewards)
                    next_states_t = torch.FloatTensor(next_states)
                    dones_t = torch.FloatTensor(dones)

                    # Q hedefleri
                    with torch.no_grad():
                        next_q = q_network(next_states_t).max(1)[0]
                        targets = rewards_t + gamma * next_q * (1 - dones_t)

                    # Mevcut Q degerleri
                    current_q = q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

                    # Loss ve guncelleme
                    loss = nn.MSELoss()(current_q, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            else:
                # Replay buffer olmadan - tek deneyimle egit
                state_t = torch.FloatTensor(state).unsqueeze(0)
                next_state_t = torch.FloatTensor(next_state).unsqueeze(0)

                with torch.no_grad():
                    if done:
                        target = reward
                    else:
                        next_q = q_network(next_state_t).max().item()
                        target = reward + gamma * next_q

                current_q = q_network(state_t)[0, action]
                loss = nn.MSELoss()(current_q, torch.tensor(target))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

        # Epsilon azalt
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            mode = "Replay" if use_replay else "No Replay"
            print(f"[{mode}] Episode {episode + 1}: Avg Reward = {avg_reward:.1f}")

    env.close()
    return rewards_history


# =============================================================================
# BOLUM 5: GORSELLEŞTIRME
# =============================================================================

def visualize_replay_effect():
    """
    Replay buffer'in etkisini gorsellestir.

    1. Korelasyon karsilastirmasi
    2. Ogrenme egrileri karsilastirmasi
    """
    print("="*60)
    print("EXPERIENCE REPLAY ETKISI ANALIZI")
    print("="*60)

    # -------------------------------------------------------------------------
    # 1. Korelasyon Analizi
    # -------------------------------------------------------------------------
    print("\n[1/2] Korelasyon analizi yapiliyor...")

    seq_corr, rand_corr = compare_sampling_methods()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sirayla ornekleme
    axes[0].plot(seq_corr, alpha=0.7)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_title('Sirayla Ornekleme (Replay Olmadan)')
    axes[0].set_xlabel('Ornekleme Adimi')
    axes[0].set_ylabel('Korelasyon')
    axes[0].set_ylim(-1, 1)
    avg_seq = np.mean(np.abs(seq_corr)) if len(seq_corr) > 0 else 0
    axes[0].text(0.05, 0.95, f'Ort. |Korelasyon|: {avg_seq:.3f}',
                transform=axes[0].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    # Rastgele ornekleme
    axes[1].plot(rand_corr, alpha=0.7, color='green')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_title('Rastgele Ornekleme (Replay Buffer)')
    axes[1].set_xlabel('Ornekleme Adimi')
    axes[1].set_ylabel('Korelasyon')
    axes[1].set_ylim(-1, 1)
    avg_rand = np.mean(np.abs(rand_corr)) if len(rand_corr) > 0 else 0
    axes[1].text(0.05, 0.95, f'Ort. |Korelasyon|: {avg_rand:.3f}',
                transform=axes[1].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    plt.savefig('02_correlation_comparison.png', dpi=150)
    plt.show()
    print("Kaydedildi: 02_correlation_comparison.png")

    # -------------------------------------------------------------------------
    # 2. Ogrenme Egrileri Karsilastirmasi
    # -------------------------------------------------------------------------
    print("\n[2/2] Ogrenme egrileri karsilastirmasi yapiliyor...")
    print("(Bu biraz zaman alabilir...)")

    rewards_with_replay = train_with_replay(use_replay=True, episodes=150)
    rewards_without_replay = train_with_replay(use_replay=False, episodes=150)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Hareketli ortalama
    window = 10

    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    ma_with = moving_average(rewards_with_replay, window)
    ma_without = moving_average(rewards_without_replay, window)

    ax.plot(range(window-1, len(rewards_with_replay)), ma_with,
            label='Replay Buffer ile', color='green', linewidth=2)
    ax.plot(range(window-1, len(rewards_without_replay)), ma_without,
            label='Replay Buffer olmadan', color='red', linewidth=2)

    ax.axhline(y=195, color='blue', linestyle='--', alpha=0.5, label='Cozum Esigi (195)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Toplam Odul')
    ax.set_title('Experience Replay Etkisi (CartPole-v1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('02_learning_comparison.png', dpi=150)
    plt.show()
    print("Kaydedildi: 02_learning_comparison.png")


def visualize_buffer_contents():
    """
    Replay buffer icerigini gorsellestir.

    Buffer'daki durumlarin dagilimini gosterir.
    """
    env = gym.make("CartPole-v1")
    replay_buffer = ReplayBuffer(capacity=5000)

    # Deneyimleri topla
    state, _ = env.reset()
    for _ in range(5000):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.push(state, action, reward, next_state, done)

        if done:
            state, _ = env.reset()
        else:
            state = next_state

    env.close()

    # Buffer'dan ornekle
    states, actions, rewards, next_states, dones = replay_buffer.sample(1000)

    # Gorsellestir
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # CartPole durumu: [konum, hiz, aci, acisal_hiz]
    feature_names = ['Araba Konumu', 'Araba Hizi', 'Cubuk Acisi', 'Acisal Hiz']

    for i, (ax, name) in enumerate(zip(axes.flatten(), feature_names)):
        ax.hist(states[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'{name} Dagilimi')
        ax.set_xlabel(name)
        ax.set_ylabel('Frekans')

    plt.suptitle('Replay Buffer Icerik Dagilimi (CartPole-v1)', fontsize=14)
    plt.tight_layout()
    plt.savefig('02_buffer_contents.png', dpi=150)
    plt.show()
    print("Kaydedildi: 02_buffer_contents.png")


# =============================================================================
# ANA PROGRAM
# =============================================================================

def main():
    """Ana fonksiyon."""
    print("\n" + "="*60)
    print("EXPERIENCE REPLAY (DENEYIM TEKRARI)")
    print("="*60)
    print("""
    Experience Replay, DQN'in en onemli bilesenidir.

    SORUN:
        RL'de veriler ardisik ve korelasyonlu:
        s1 -> s2 -> s3 -> s4 ...

        Korelasyonlu veri ile egitim:
        - Kararsizdır
        - Yakın zamanlı deneyimlere fazla uyum saglar
        - Eski bilgiyi unutur

    COZUM:
        Deneyimleri bellekte sakla
        Rastgele ornekleyerek egit

        Faydalar:
        1. Korelasyonu kirar
        2. Her deneyimi birden fazla kullanir (veri verimliligi)
        3. Daha kararli ogrenme
    """)

    # Gorsellestirmeleri olustur
    print("\nGorsellestirmeler olusturuluyor...\n")

    # Buffer icerigini goster
    print("[1/2] Buffer icerik analizi...")
    visualize_buffer_contents()

    # Replay etkisini goster
    print("\n[2/2] Replay etkisi analizi...")
    visualize_replay_effect()

    print("\n" + "="*60)
    print("SONUC")
    print("="*60)
    print("""
    Experience Replay'in etkileri:

    1. KORELASYON AZALDI:
       - Sirayla: Yuksek korelasyon (~0.8+)
       - Rastgele: Dusuk korelasyon (~0.1)

    2. OGRENME IYILESTI:
       - Replay ile: Daha hizli ve kararli ogrenme
       - Replay olmadan: Kararsiz, bazen ogrenmiyor

    3. VERI VERIMLILIGI:
       - Ayni deneyim birden fazla kez kullanilabilir
       - Nadir deneyimler kaybolmaz

    Bir sonraki dosyada Target Network'u ogrenecegiz!
    """)


if __name__ == "__main__":
    main()
