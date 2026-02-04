"""
Dueling DQN
===========
Bu script, Dueling DQN mimarisini detayli aciklar:
1. Value ve Advantage ayirimi
2. Dueling mimari
3. Standart DQN ile karsilastirma
4. Ne zaman fayda saglar?

Dueling DQN Nedir?
    Q-fonksiyonunu iki bileşene ayirir:
    Q(s, a) = V(s) + A(s, a)

    V(s): State Value - "Bu durum ne kadar iyi?"
    A(s, a): Advantage - "Bu aksiyon ortalamadan ne kadar iyi?"

    Avantaj:
    - Deger fonksiyonunu daha iyi ogrenebilir
    - Aksiyonlar birbirine yakin oldugunda daha etkili
    - Ayni deneyimden daha fazla ogrenir

Kullanim:
    python 05_dueling_dqn.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random
from typing import List, Tuple
import gymnasium as gym


# =============================================================================
# BOLUM 1: VALUE VE ADVANTAGE KAVRAMI
# =============================================================================

def explain_value_and_advantage():
    """
    Value ve Advantage kavramlarini acikla.
    """
    print("\n" + "="*60)
    print("VALUE VE ADVANTAGE KAVRAMLARI")
    print("="*60)
    print("""
    Q-Fonksiyonu Ayrismasi:
    -----------------------

    Geleneksel:
        Q(s, a) = Durum s'de aksiyon a'nin beklenen degeri

    Dueling:
        Q(s, a) = V(s) + A(s, a)

    V(s) - State Value (Durum Degeri):
        - Durumun "iyiligi"
        - Aksiyondan BAGIMSIZ
        - "Bu durumda olmak ne kadar iyi?"

    A(s, a) - Advantage (Avantaj):
        - Aksiyonun goreceli "iyiligi"
        - Duruma GORE degisir
        - "Bu aksiyon ortalamadan ne kadar iyi/kotu?"

    Onemli Ozellik:
        sum_a A(s, a) = 0 (veya max_a A(s, a) = 0)
        Yani avantajlarin ortalamasi/maksimumu sifir.


    ORNEK: Cliff Walking
    --------------------
    Durum: Ucurumun kenarinda

    V(durum) = -10  (tehlikeli durum, dusuk deger)

    Aksiyonlar:
        A(ileri) = +5   (ucurumdan uzaklasir, iyi)
        A(geri)  = -5   (ucuruma yaklasir, kotu)
        A(sol)   =  0   (notr)
        A(sag)   =  0   (notr)

    Q degerleri:
        Q(ileri) = -10 + 5 = -5
        Q(geri)  = -10 - 5 = -15
        Q(sol)   = -10 + 0 = -10
        Q(sag)   = -10 + 0 = -10


    NEDEN BU AYIRIM FAYDALI?
    ------------------------
    1. V(s) cok durumda AYNI (aksiyon onemli degil)
       Ornegin: Kazanilmis bir oyunda her aksiyon iyi
       -> V ogrenilirse, A'yi ogrenmek daha kolay

    2. Daha az veri ile daha iyi ogrenme
       - V bir kere ogrenilince tum aksiyonlara uygulanir
       - Ayni deneyimden daha fazla bilgi cikarilir

    3. Aksiyonlar birbirine yakin oldugunda
       - Standart DQN kucuk farklari ogrenmekte zorlanir
       - Dueling bu farklari Advantage ile yakalayabilir
    """)


# =============================================================================
# BOLUM 2: STANDART VS DUELING MIMARI
# =============================================================================

class StandardDQN(nn.Module):
    """Standart DQN mimarisi."""
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
    """
    Dueling DQN mimarisi.

    Mimari:
                         -> Value stream  -> V(s)
    State -> Shared     |
                         -> Advantage stream -> A(s,a)

    Cikti:
        Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))

    Neden mean cikarilir?
        - A'nin benzersiz olmasi icin
        - V ve A'nin ayrilabilir olmasi icin
        - Ogrenmeyi kolaylastirir
    """
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()

        # Paylasilan ozellik katmani
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        )

        # Value akisi (tek cikti)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)  # Tek deger: V(s)
        )

        # Advantage akisi (aksiyon sayisi kadar cikti)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)  # Her aksiyon icin A(s,a)
        )

    def forward(self, x):
        features = self.feature(x)

        # V(s) ve A(s, a) hesapla
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Q = V + (A - mean(A))
        # mean cikarma ile A benzersiz olur
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def get_value_and_advantage(self, x):
        """Debug/gorsellestirme icin V ve A'yi ayri dondur."""
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value, advantage


def visualize_architectures():
    """
    Standart ve Dueling mimarilerini gorsellestir.
    """
    print("\n" + "="*60)
    print("MIMARI KARSILASTIRMASI")
    print("="*60)

    state_size, action_size = 4, 2

    # Modelleri olustur
    standard = StandardDQN(state_size, action_size)
    dueling = DuelingDQN(state_size, action_size)

    # Parametre sayilari
    std_params = sum(p.numel() for p in standard.parameters())
    duel_params = sum(p.numel() for p in dueling.parameters())

    print(f"\nStandart DQN parametreleri: {std_params:,}")
    print(f"Dueling DQN parametreleri:  {duel_params:,}")

    # Ornek girdi ile test
    test_input = torch.randn(1, state_size)

    std_output = standard(test_input)
    duel_output = dueling(test_input)
    value, advantage = dueling.get_value_and_advantage(test_input)

    print(f"\nOrnek girdi: {test_input.squeeze().numpy()}")
    print(f"\nStandart DQN ciktisi (Q-degerleri):")
    print(f"  Q(s, a0) = {std_output[0, 0].item():.4f}")
    print(f"  Q(s, a1) = {std_output[0, 1].item():.4f}")

    print(f"\nDueling DQN bileşenleri:")
    print(f"  V(s) = {value.item():.4f}")
    print(f"  A(s, a0) = {advantage[0, 0].item():.4f}")
    print(f"  A(s, a1) = {advantage[0, 1].item():.4f}")
    print(f"  mean(A) = {advantage.mean().item():.4f}")

    print(f"\nDueling DQN ciktisi (Q-degerleri):")
    print(f"  Q(s, a0) = V + (A0 - mean(A)) = {duel_output[0, 0].item():.4f}")
    print(f"  Q(s, a1) = V + (A1 - mean(A)) = {duel_output[0, 1].item():.4f}")

    # Mimari diyagrami
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Standart DQN diyagrami
    axes[0].text(0.5, 0.9, 'STANDART DQN', ha='center', va='center',
                fontsize=14, fontweight='bold')
    axes[0].text(0.5, 0.7, 'State (4)', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue'))
    axes[0].annotate('', xy=(0.5, 0.6), xytext=(0.5, 0.65),
                    arrowprops=dict(arrowstyle='->', color='black'))
    axes[0].text(0.5, 0.5, 'Hidden (128)', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[0].annotate('', xy=(0.5, 0.4), xytext=(0.5, 0.45),
                    arrowprops=dict(arrowstyle='->', color='black'))
    axes[0].text(0.5, 0.3, 'Hidden (128)', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[0].annotate('', xy=(0.5, 0.2), xytext=(0.5, 0.25),
                    arrowprops=dict(arrowstyle='->', color='black'))
    axes[0].text(0.5, 0.1, 'Q(s,a) (2)', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen'))
    axes[0].axis('off')
    axes[0].set_title('Tek Akis')

    # Dueling DQN diyagrami
    axes[1].text(0.5, 0.9, 'DUELING DQN', ha='center', va='center',
                fontsize=14, fontweight='bold')
    axes[1].text(0.5, 0.7, 'State (4)', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue'))
    axes[1].annotate('', xy=(0.5, 0.6), xytext=(0.5, 0.65),
                    arrowprops=dict(arrowstyle='->', color='black'))
    axes[1].text(0.5, 0.5, 'Shared (128)', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # Value akisi
    axes[1].annotate('', xy=(0.3, 0.35), xytext=(0.45, 0.45),
                    arrowprops=dict(arrowstyle='->', color='blue'))
    axes[1].text(0.25, 0.25, 'V(s) (1)', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral'))

    # Advantage akisi
    axes[1].annotate('', xy=(0.7, 0.35), xytext=(0.55, 0.45),
                    arrowprops=dict(arrowstyle='->', color='green'))
    axes[1].text(0.75, 0.25, 'A(s,a) (2)', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral'))

    # Birlestirme
    axes[1].annotate('', xy=(0.5, 0.15), xytext=(0.3, 0.2),
                    arrowprops=dict(arrowstyle='->', color='black'))
    axes[1].annotate('', xy=(0.5, 0.15), xytext=(0.7, 0.2),
                    arrowprops=dict(arrowstyle='->', color='black'))
    axes[1].text(0.5, 0.1, 'Q = V + (A - mean(A))', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen'))
    axes[1].axis('off')
    axes[1].set_title('Iki Akis: Value + Advantage')

    plt.tight_layout()
    plt.savefig('05_architecture_comparison.png', dpi=150)
    plt.show()
    print("\nKaydedildi: 05_architecture_comparison.png")


# =============================================================================
# BOLUM 3: REPLAY BUFFER VE EGITIM
# =============================================================================

class ReplayBuffer:
    """Basit replay buffer."""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


def train_dqn(
    network_type: str = "standard",
    episodes: int = 200,
    seed: int = 42
) -> Tuple[List[float], List[float], List[float]]:
    """
    Standart veya Dueling DQN egitimi.

    Args:
        network_type: "standard" veya "dueling"
        episodes: Toplam episode sayisi
        seed: Random seed

    Returns:
        (rewards_history, value_history, advantage_history)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Ag secimi
    if network_type == "dueling":
        policy_net = DuelingDQN(state_size, action_size)
        target_net = DuelingDQN(state_size, action_size)
    else:
        policy_net = StandardDQN(state_size, action_size)
        target_net = StandardDQN(state_size, action_size)

    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(capacity=10000)

    # Parametreler
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 64
    target_update_freq = 100

    rewards_history = []
    value_history = []
    advantage_history = []

    # Test durumu
    test_state = torch.FloatTensor([0.0, 0.0, 0.0, 0.0]).unsqueeze(0)

    total_steps = 0

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.FloatTensor(state))
                    action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_steps += 1

            replay_buffer.push(state, action, reward, next_state, done)

            # Egitim
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states_t = torch.FloatTensor(states)
                actions_t = torch.LongTensor(actions)
                rewards_t = torch.FloatTensor(rewards)
                next_states_t = torch.FloatTensor(next_states)
                dones_t = torch.FloatTensor(dones)

                # Hedefler (Double DQN tarzı)
                with torch.no_grad():
                    best_actions = policy_net(next_states_t).argmax(dim=1)
                    next_q = target_net(next_states_t).gather(1, best_actions.unsqueeze(1)).squeeze()
                    targets = rewards_t + gamma * next_q * (1 - dones_t)

                # Mevcut Q degerleri
                current_q = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

                # Loss ve guncelleme
                loss = nn.MSELoss()(current_q, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Target network guncelleme
                if total_steps % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                # V ve A takibi (sadece Dueling icin)
                if network_type == "dueling":
                    with torch.no_grad():
                        v, a = policy_net.get_value_and_advantage(test_state)
                        value_history.append(v.item())
                        advantage_history.append(a.mean().item())

            state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            print(f"[{network_type.upper()}] Ep {episode + 1}: Avg = {avg_reward:.1f}")

    env.close()
    return rewards_history, value_history, advantage_history


# =============================================================================
# BOLUM 4: KARSILASTIRMA
# =============================================================================

def compare_architectures():
    """
    Standart ve Dueling DQN karsilastirmasi.
    """
    print("\n" + "="*60)
    print("STANDART VS DUELING DQN KARSILASTIRMASI")
    print("="*60)
    print("(Bu biraz zaman alabilir...)")

    # Standart DQN
    print("\n[1/2] Standart DQN egitimi...")
    rewards_std, _, _ = train_dqn(network_type="standard", episodes=200)

    # Dueling DQN
    print("\n[2/2] Dueling DQN egitimi...")
    rewards_duel, values, advantages = train_dqn(network_type="dueling", episodes=200)

    # Gorsellestir
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Ogrenme egrileri
    window = 10
    def moving_avg(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    ma_std = moving_avg(rewards_std, window)
    ma_duel = moving_avg(rewards_duel, window)

    axes[0, 0].plot(range(window-1, len(rewards_std)), ma_std,
                   label='Standard DQN', color='blue', linewidth=2)
    axes[0, 0].plot(range(window-1, len(rewards_duel)), ma_duel,
                   label='Dueling DQN', color='green', linewidth=2)
    axes[0, 0].axhline(y=195, color='red', linestyle='--', alpha=0.5, label='Cozum Esigi')
    axes[0, 0].set_title('Ogrenme Egrileri')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Odul')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Ham oduller karsilastirmasi
    axes[0, 1].plot(rewards_std, alpha=0.5, label='Standard', color='blue')
    axes[0, 1].plot(rewards_duel, alpha=0.5, label='Dueling', color='green')
    axes[0, 1].set_title('Ham Episode Odulleri')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Odul')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Value degisimi (sadece Dueling)
    if values:
        axes[1, 0].plot(values[:3000], alpha=0.7, color='purple')
        axes[1, 0].set_title('V(s) Degisimi (Dueling DQN)')
        axes[1, 0].set_xlabel('Egitim Adimi')
        axes[1, 0].set_ylabel('V(test_state)')
        axes[1, 0].grid(True, alpha=0.3)

    # Advantage degisimi (sadece Dueling)
    if advantages:
        axes[1, 1].plot(advantages[:3000], alpha=0.7, color='orange')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('mean(A(s,a)) Degisimi (Dueling DQN)')
        axes[1, 1].set_xlabel('Egitim Adimi')
        axes[1, 1].set_ylabel('mean(A(test_state))')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('05_dueling_comparison.png', dpi=150)
    plt.show()
    print("Kaydedildi: 05_dueling_comparison.png")

    # Istatistikler
    print("\n" + "-"*40)
    print("ISTATISTIKLER:")
    print("-"*40)
    print(f"Standard DQN: Son 50 ep ort. = {np.mean(rewards_std[-50:]):.1f}")
    print(f"Dueling DQN:  Son 50 ep ort. = {np.mean(rewards_duel[-50:]):.1f}")


# =============================================================================
# BOLUM 5: V VE A GORSELLESTIRMESI
# =============================================================================

def visualize_value_advantage():
    """
    Value ve Advantage fonksiyonlarini farkli durumlar icin gorsellestir.
    """
    print("\n" + "="*60)
    print("VALUE VE ADVANTAGE GORSELLESTIRMESI")
    print("="*60)

    # Egitilmis bir Dueling ag olustur
    state_size, action_size = 4, 2
    dueling_net = DuelingDQN(state_size, action_size)
    optimizer = optim.Adam(dueling_net.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer()

    # Biraz egit
    env = gym.make("CartPole-v1")
    print("Ag egitiliyor...")

    for episode in range(100):
        state, _ = env.reset()
        done = False
        epsilon = max(0.1, 1.0 - episode / 50)

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = dueling_net(torch.FloatTensor(state))
                    action = q.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)

            if len(replay_buffer) >= 64:
                states, actions, rewards, next_states, dones = replay_buffer.sample(64)
                states_t = torch.FloatTensor(states)
                actions_t = torch.LongTensor(actions)
                rewards_t = torch.FloatTensor(rewards)
                next_states_t = torch.FloatTensor(next_states)
                dones_t = torch.FloatTensor(dones)

                with torch.no_grad():
                    next_q = dueling_net(next_states_t).max(1)[0]
                    targets = rewards_t + 0.99 * next_q * (1 - dones_t)

                current_q = dueling_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
                loss = nn.MSELoss()(current_q, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

    env.close()
    print("Egitim tamamlandi!")

    # Farkli durumlar icin V ve A hesapla
    # CartPole: [konum, hiz, aci, acisal_hiz]
    pole_angles = np.linspace(-0.3, 0.3, 50)
    values = []
    advantages_left = []
    advantages_right = []
    q_left = []
    q_right = []

    for angle in pole_angles:
        state = torch.FloatTensor([[0.0, 0.0, angle, 0.0]])
        with torch.no_grad():
            v, a = dueling_net.get_value_and_advantage(state)
            q = dueling_net(state)

            values.append(v.item())
            advantages_left.append(a[0, 0].item())
            advantages_right.append(a[0, 1].item())
            q_left.append(q[0, 0].item())
            q_right.append(q[0, 1].item())

    # Gorsellestir
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # V(s) vs cubuk acisi
    axes[0].plot(np.degrees(pole_angles), values, 'b-', linewidth=2)
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title('V(s) vs Cubuk Acisi')
    axes[0].set_xlabel('Cubuk Acisi (derece)')
    axes[0].set_ylabel('V(s)')
    axes[0].grid(True, alpha=0.3)

    # A(s, a) vs cubuk acisi
    axes[1].plot(np.degrees(pole_angles), advantages_left, 'g-',
                linewidth=2, label='A(sol)')
    axes[1].plot(np.degrees(pole_angles), advantages_right, 'r-',
                linewidth=2, label='A(sag)')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title('A(s, a) vs Cubuk Acisi')
    axes[1].set_xlabel('Cubuk Acisi (derece)')
    axes[1].set_ylabel('A(s, a)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Q(s, a) vs cubuk acisi
    axes[2].plot(np.degrees(pole_angles), q_left, 'g-',
                linewidth=2, label='Q(sol)')
    axes[2].plot(np.degrees(pole_angles), q_right, 'r-',
                linewidth=2, label='Q(sag)')
    axes[2].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title('Q(s, a) = V + A')
    axes[2].set_xlabel('Cubuk Acisi (derece)')
    axes[2].set_ylabel('Q(s, a)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('05_value_advantage_visualization.png', dpi=150)
    plt.show()
    print("Kaydedildi: 05_value_advantage_visualization.png")

    print("""
    YORUM:
    ------
    - V(s): Cubuk dik (aci=0) oldugunda en yuksek
    - A(sol): Cubuk saga egik (+aci) oldugunda yuksek (duzeltme icin sola git)
    - A(sag): Cubuk sola egik (-aci) oldugunda yuksek (duzeltme icin saga git)
    - Q = V + A: Her aksiyonun toplam degeri
    """)


# =============================================================================
# ANA PROGRAM
# =============================================================================

def main():
    """Ana fonksiyon."""
    print("\n" + "="*60)
    print("DUELING DQN")
    print("="*60)

    # 1. Kavramlari acikla
    explain_value_and_advantage()

    # 2. Mimarileri gorsellestir
    visualize_architectures()

    # 3. Karsilastirma
    compare_architectures()

    # 4. V ve A gorsellestirmesi
    visualize_value_advantage()

    print("\n" + "="*60)
    print("SONUC")
    print("="*60)
    print("""
    Dueling DQN'in Avantajlari:

    1. DAHA IYI GENELLEME:
       - V(s) bir kere ogrenilince tum aksiyonlara uygulanir
       - Ayni deneyimden daha fazla bilgi

    2. AKSIYON FARKLARI:
       - Aksiyonlar birbirine yakin oldugunda daha etkili
       - A(s,a) kucuk farklari yakalayabilir

    3. "ONEMLI OLMAYAN" DURUMLAR:
       - Bazen aksiyon onemli degil (ornegin: zaten kazanilmis oyun)
       - Dueling bunu V(s) ile ifade edebilir

    NE ZAMAN KULLANMALI?
    --------------------
    - Cok sayida aksiyon varsa
    - Aksiyonlar benzer sonuclar veriyorsa
    - Durum degeri aksiyondan bagimsizsa

    RAINBOW DQN:
    ------------
    En iyi sonuclar icin:
    - Double DQN (asiri tahmin azaltma)
    - Dueling DQN (V/A ayirimi)
    - Prioritized Experience Replay
    - Multi-step Learning
    - ve daha fazlasi...

    bir arada kullanilir!
    """)


if __name__ == "__main__":
    main()
