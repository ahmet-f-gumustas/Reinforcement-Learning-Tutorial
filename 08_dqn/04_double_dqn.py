"""
Double DQN (DDQN)
=================
Bu script, Double DQN kavramini detayli aciklar:
1. Q-degeri asiri tahmin problemi
2. Double DQN cozumu
3. Standart DQN ile karsilastirma
4. Q-degeri analizi

Double DQN Nedir?
    Standart DQN, Q-degerlerini ASIRI TAHMIN eder (overestimate).
    Double DQN, aksiyon secimi ve degerlendirmesini ayirarak
    bu sorunu cözer.

    Standart DQN:
        target = r + gamma * max_a Q(s', a; theta-)

    Double DQN:
        a* = argmax_a Q(s', a; theta)     <- Policy network ile sec
        target = r + gamma * Q(s', a*; theta-)  <- Target network ile degerlendir

Kullanim:
    python 04_double_dqn.py
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
# BOLUM 1: ASIRI TAHMIN PROBLEMI
# =============================================================================

def demonstrate_overestimation():
    """
    Max operatorunun asiri tahmine nasil yol actigini goster.

    E[max(X)] >= max(E[X])

    Gurultulu tahminlerin maksimumunu almak,
    gercek maksimumdan daha yuksek deger verir.
    """
    print("\n" + "="*60)
    print("Q-DEGERI ASIRI TAHMIN PROBLEMI")
    print("="*60)

    np.random.seed(42)

    # Gercek Q degerleri (4 aksiyon)
    true_q_values = np.array([1.0, 2.0, 3.0, 2.5])
    print(f"\nGercek Q degerleri: {true_q_values}")
    print(f"Gercek maksimum: {true_q_values.max():.2f} (Aksiyon {true_q_values.argmax()})")

    # Gurultulu tahminler simulasyonu
    num_simulations = 10000
    noise_std = 1.0

    max_values = []
    selected_actions = []

    for _ in range(num_simulations):
        # Gurultulu Q tahminleri
        noisy_q = true_q_values + np.random.normal(0, noise_std, size=4)
        max_values.append(noisy_q.max())
        selected_actions.append(noisy_q.argmax())

    avg_max = np.mean(max_values)
    action_dist = np.bincount(selected_actions, minlength=4) / num_simulations

    print(f"\nOrtalama tahmin edilen maksimum: {avg_max:.2f}")
    print(f"Asiri tahmin miktari: {avg_max - true_q_values.max():.2f}")
    print(f"\nAksiyon secim dagilimi:")
    for i, prob in enumerate(action_dist):
        marker = " <-- En iyi" if i == true_q_values.argmax() else ""
        print(f"  Aksiyon {i}: {prob*100:.1f}%{marker}")

    # Gorsellestir
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(max_values, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=true_q_values.max(), color='green', linestyle='--',
                   linewidth=2, label=f'Gercek max: {true_q_values.max():.2f}')
    axes[0].axvline(x=avg_max, color='red', linestyle='--',
                   linewidth=2, label=f'Ort. tahmin: {avg_max:.2f}')
    axes[0].set_title('max(Q) Dagilimi (Gurultulu Tahminlerle)')
    axes[0].set_xlabel('max(Q) Degeri')
    axes[0].set_ylabel('Frekans')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Aksiyon dagilimi
    bars = axes[1].bar(range(4), action_dist, color=['blue', 'blue', 'green', 'blue'])
    bars[true_q_values.argmax()].set_color('green')
    axes[1].set_title('Secilen Aksiyon Dagilimi')
    axes[1].set_xlabel('Aksiyon')
    axes[1].set_ylabel('Secilme Olasiligi')
    axes[1].set_xticks(range(4))
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('04_overestimation_demo.png', dpi=150)
    plt.show()
    print("Kaydedildi: 04_overestimation_demo.png")

    print("""
    SONUC:
    ------
    max operatoru pozitif yanlilik (bias) olusturur.

    Neden?
    - Her aksiyon icin tahmin hatasi var (gurultu)
    - max() en yuksek TAHMINI secer
    - Pozitif hatali tahminler secilme sansi daha yuksek

    Sonuc:
    - Q degerleri asiri tahmin edilir
    - Suboptimal aksiyonlar secilir
    - Ogrenme yavaslar veya basarisiz olur

    Double DQN bu sorunu cozer!
    """)


# =============================================================================
# BOLUM 2: DQN SINIFLARI
# =============================================================================

class DQN(nn.Module):
    """Standart DQN agi."""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
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


# =============================================================================
# BOLUM 3: STANDART VS DOUBLE DQN
# =============================================================================

def compute_standard_dqn_target(
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    target_net: nn.Module,
    gamma: float
) -> torch.Tensor:
    """
    Standart DQN hedef hesaplama.

    target = r + gamma * max_a Q(s', a; theta-)

    PROBLEM: Ayni ag hem secim hem degerlendirme yapar!
    """
    with torch.no_grad():
        # Target network ile maksimum Q degeri
        next_q_values = target_net(next_states)
        max_next_q = next_q_values.max(dim=1)[0]
        targets = rewards + gamma * max_next_q * (1 - dones)
    return targets


def compute_double_dqn_target(
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    policy_net: nn.Module,
    target_net: nn.Module,
    gamma: float
) -> torch.Tensor:
    """
    Double DQN hedef hesaplama.

    a* = argmax_a Q(s', a; theta)      <- Policy network ile SECIM
    target = r + gamma * Q(s', a*; theta-)  <- Target network ile DEGERLENDIRME

    COZUM: Secim ve degerlendirme FARKLI aglarla!
    """
    with torch.no_grad():
        # Policy network ile en iyi aksiyonu SEC
        next_q_policy = policy_net(next_states)
        best_actions = next_q_policy.argmax(dim=1)

        # Target network ile secilen aksiyonu DEGERLENDIR
        next_q_target = target_net(next_states)
        next_q_values = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze()

        targets = rewards + gamma * next_q_values * (1 - dones)
    return targets


# =============================================================================
# BOLUM 4: EGITIM FONKSIYONLARI
# =============================================================================

def train_dqn(
    use_double: bool = False,
    episodes: int = 200,
    seed: int = 42
) -> Tuple[List[float], List[float]]:
    """
    Standart veya Double DQN egitimi.

    Args:
        use_double: Double DQN kullan?
        episodes: Toplam episode sayisi
        seed: Random seed

    Returns:
        (rewards_history, q_values_history)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Aglar
    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
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
    q_values_history = []

    # Test durumu icin Q degeri takibi
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

                # Hedef hesapla
                if use_double:
                    targets = compute_double_dqn_target(
                        rewards_t, next_states_t, dones_t,
                        policy_net, target_net, gamma
                    )
                else:
                    targets = compute_standard_dqn_target(
                        rewards_t, next_states_t, dones_t,
                        target_net, gamma
                    )

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

                # Q degeri takibi
                with torch.no_grad():
                    test_q = policy_net(test_state).max().item()
                    q_values_history.append(test_q)

            state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            mode = "Double DQN" if use_double else "Standard DQN"
            avg_q = np.mean(q_values_history[-100:]) if q_values_history else 0
            print(f"[{mode}] Ep {episode + 1}: Avg Reward = {avg_reward:.1f}, Avg Q = {avg_q:.2f}")

    env.close()
    return rewards_history, q_values_history


# =============================================================================
# BOLUM 5: KARSILASTIRMA
# =============================================================================

def compare_dqn_variants():
    """
    Standart DQN ve Double DQN karsilastirmasi.
    """
    print("\n" + "="*60)
    print("STANDART DQN VS DOUBLE DQN KARSILASTIRMASI")
    print("="*60)
    print("(Bu biraz zaman alabilir...)")

    # Standart DQN
    print("\n[1/2] Standart DQN egitimi...")
    rewards_std, q_values_std = train_dqn(use_double=False, episodes=200)

    # Double DQN
    print("\n[2/2] Double DQN egitimi...")
    rewards_double, q_values_double = train_dqn(use_double=True, episodes=200)

    # Gorsellestir
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Ogrenme egrileri
    window = 10
    def moving_avg(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    ma_std = moving_avg(rewards_std, window)
    ma_double = moving_avg(rewards_double, window)

    axes[0, 0].plot(range(window-1, len(rewards_std)), ma_std,
                   label='Standard DQN', color='blue', linewidth=2)
    axes[0, 0].plot(range(window-1, len(rewards_double)), ma_double,
                   label='Double DQN', color='green', linewidth=2)
    axes[0, 0].axhline(y=195, color='red', linestyle='--', alpha=0.5, label='Cozum Esigi')
    axes[0, 0].set_title('Ogrenme Egrileri')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Odul')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Q-deger karsilastirmasi
    axes[0, 1].plot(q_values_std[:3000], alpha=0.7, label='Standard DQN', color='blue')
    axes[0, 1].plot(q_values_double[:3000], alpha=0.7, label='Double DQN', color='green')
    axes[0, 1].set_title('Q-Degeri Degisimi (Ilk 3000 Adim)')
    axes[0, 1].set_xlabel('Egitim Adimi')
    axes[0, 1].set_ylabel('Q(test_state)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Q-deger histogramlari
    axes[1, 0].hist(q_values_std[-1000:], bins=50, alpha=0.7,
                   label='Standard DQN', color='blue')
    axes[1, 0].set_title('Son 1000 Adim Q-Degeri Dagilimi (Standard DQN)')
    axes[1, 0].set_xlabel('Q-Degeri')
    axes[1, 0].set_ylabel('Frekans')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(q_values_double[-1000:], bins=50, alpha=0.7,
                   label='Double DQN', color='green')
    axes[1, 1].set_title('Son 1000 Adim Q-Degeri Dagilimi (Double DQN)')
    axes[1, 1].set_xlabel('Q-Degeri')
    axes[1, 1].set_ylabel('Frekans')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('04_dqn_vs_double_dqn.png', dpi=150)
    plt.show()
    print("Kaydedildi: 04_dqn_vs_double_dqn.png")

    # Istatistikler
    print("\n" + "-"*40)
    print("ISTATISTIKLER:")
    print("-"*40)
    print(f"Standard DQN:")
    print(f"  Son 50 ep ort. odul: {np.mean(rewards_std[-50:]):.1f}")
    print(f"  Ort. Q degeri (son 1000): {np.mean(q_values_std[-1000:]):.2f}")
    print(f"  Q degeri std: {np.std(q_values_std[-1000:]):.2f}")

    print(f"\nDouble DQN:")
    print(f"  Son 50 ep ort. odul: {np.mean(rewards_double[-50:]):.1f}")
    print(f"  Ort. Q degeri (son 1000): {np.mean(q_values_double[-1000:]):.2f}")
    print(f"  Q degeri std: {np.std(q_values_double[-1000:]):.2f}")


# =============================================================================
# BOLUM 6: ASIRI TAHMIN ANALIZI
# =============================================================================

def analyze_overestimation():
    """
    Q-degeri asiri tahminini detayli analiz et.

    CartPole'da gercek Q degeri teorik olarak hesaplanabilir:
    - Her adimda +1 odul
    - Maksimum 500 adim
    - gamma = 0.99
    - Teorik maks Q ≈ sum(gamma^t) ≈ 100
    """
    print("\n" + "="*60)
    print("ASIRI TAHMIN ANALIZI")
    print("="*60)

    # Teorik maksimum Q degeri (CartPole icin)
    gamma = 0.99
    max_steps = 500
    theoretical_max_q = sum([gamma**t for t in range(max_steps)])
    print(f"\nTeorik maksimum Q degeri (CartPole): ~{theoretical_max_q:.1f}")

    # Her iki yontemle egit ve Q degerlerini karsilastir
    print("\n[1/2] Standard DQN ile analiz...")
    _, q_std = train_dqn(use_double=False, episodes=150)

    print("\n[2/2] Double DQN ile analiz...")
    _, q_double = train_dqn(use_double=True, episodes=150)

    # Gorsellestir
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(q_std, alpha=0.7, label='Standard DQN', color='blue')
    ax.plot(q_double, alpha=0.7, label='Double DQN', color='green')
    ax.axhline(y=theoretical_max_q, color='red', linestyle='--',
              linewidth=2, label=f'Teorik Maks Q: {theoretical_max_q:.1f}')

    ax.set_title('Q-Degeri Tahmini vs Teorik Deger')
    ax.set_xlabel('Egitim Adimi')
    ax.set_ylabel('Q-Degeri')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Asiri tahmin istatistikleri
    overest_std = np.mean([q - theoretical_max_q for q in q_std[-1000:] if q > theoretical_max_q])
    overest_double = np.mean([q - theoretical_max_q for q in q_double[-1000:] if q > theoretical_max_q])

    textstr = f'Ort. Asiri Tahmin (son 1000):\nStandard: {overest_std:.1f}\nDouble: {overest_double:.1f}'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    plt.savefig('04_overestimation_analysis.png', dpi=150)
    plt.show()
    print("Kaydedildi: 04_overestimation_analysis.png")


# =============================================================================
# ANA PROGRAM
# =============================================================================

def main():
    """Ana fonksiyon."""
    print("\n" + "="*60)
    print("DOUBLE DQN (DDQN)")
    print("="*60)
    print("""
    Double DQN, standart DQN'in asiri tahmin sorununu cozer.

    PROBLEM: Asiri Tahmin (Overestimation)
    --------------------------------------
    Standart DQN hedefi:
        target = r + gamma * max_a Q(s', a; theta-)

    max operatoru AYNI ag ile:
    1. En iyi aksiyonu SECER
    2. Bu aksiyonu DEGERLENDIRIR

    Gurultulu tahminlerde:
    - max() pozitif hatali tahminleri secme egiliminde
    - Q degerleri asiri tahmin edilir
    - Suboptimal politikalar ogrenilebilir

    COZUM: Double DQN
    -----------------
    Secim ve degerlendirmeyi AYIR:

        a* = argmax_a Q(s', a; theta)      <- POLICY net ile SEC
        target = r + gamma * Q(s', a*; theta-)  <- TARGET net ile DEGERLENDIR

    Farkli aglar kullanildigi icin:
    - Secim hatasi degerlendirmeye yansimaz
    - Asiri tahmin azalir
    - Daha iyi ogrenme
    """)

    # 1. Asiri tahmin problemini goster
    demonstrate_overestimation()

    # 2. Karsilastirma
    compare_dqn_variants()

    # 3. Detayli asiri tahmin analizi
    analyze_overestimation()

    print("\n" + "="*60)
    print("SONUC")
    print("="*60)
    print("""
    Double DQN'in Avantajlari:

    1. AZALAN ASIRI TAHMIN:
       - Q degerleri gercege daha yakin
       - Daha guvenilir deger tahminleri

    2. DAHA IYLESEN PERFORMANS:
       - Ozellikle karmasik ortamlarda
       - Daha kararli ogrenme

    3. MINIMAL EKSTRA MALIYET:
       - Sadece hedef hesaplama degisiyor
       - Zaten target network var

    IMPLEMENTASYON FARKI:
    ---------------------
    Standard DQN:
        next_q = target_net(s').max()

    Double DQN:
        best_a = policy_net(s').argmax()
        next_q = target_net(s')[best_a]

    Siradaki: Dueling DQN (V ve A ayirimi)
    """)


if __name__ == "__main__":
    main()
