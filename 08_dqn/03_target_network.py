"""
Target Network (Hedef Ag)
=========================
Bu script, Target Network kavramini detayli aciklar:
1. Hareket eden hedef problemi
2. Target network cozumu
3. Hard vs Soft guncelleme
4. Kararlilik analizi

Target Network Nedir?
    Q-learning'de hedef su sekilde hesaplanir:
    target = r + gamma * max Q(s', a')

    SORUN: Q(s', a') ayni ag ile hesaplaniyor!
    Ag her adimda guncellenince hedef de degisiyor.
    "Hareket eden hedefi kovalamak" gibi.

    COZUM: Ayri bir "hedef agi" kullan
    - Policy Network: Her adimda guncellenir
    - Target Network: Periyodik olarak kopyalanir

Kullanim:
    python 03_target_network.py
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
import copy


# =============================================================================
# BOLUM 1: HAREKET EDEN HEDEF PROBLEMI
# =============================================================================

def demonstrate_moving_target():
    """
    Hareket eden hedef problemini goster.

    Basit bir ornek: Sabit hedefe vs hareket eden hedefe yaklasma.
    """
    print("\n" + "="*60)
    print("HAREKET EDEN HEDEF PROBLEMI")
    print("="*60)

    # Senaryo 1: Sabit hedef
    x_fixed = 0.0
    target_fixed = 10.0
    lr = 0.1
    history_fixed = [x_fixed]

    for _ in range(50):
        x_fixed = x_fixed + lr * (target_fixed - x_fixed)
        history_fixed.append(x_fixed)

    # Senaryo 2: Hareket eden hedef (hedef = 2 * x)
    x_moving = 0.0
    history_moving = [x_moving]

    for _ in range(50):
        target_moving = 2 * x_moving + 5  # Hedef x'e bagli
        x_moving = x_moving + lr * (target_moving - x_moving)
        history_moving.append(x_moving)

    # Gorsellestir
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history_fixed, 'b-', linewidth=2, label='x degeri')
    axes[0].axhline(y=target_fixed, color='r', linestyle='--', label='Hedef (sabit)')
    axes[0].set_title('Sabit Hedef: Stabil Yakinasma')
    axes[0].set_xlabel('Adim')
    axes[0].set_ylabel('Deger')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_moving, 'b-', linewidth=2, label='x degeri')
    axes[1].set_title('Hareket Eden Hedef: Kararsiz')
    axes[1].set_xlabel('Adim')
    axes[1].set_ylabel('Deger')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.5, 0.95, 'Hedef = 2x + 5 (x ile birlikte degisiyor)',
                transform=axes[1].transAxes, fontsize=10,
                verticalalignment='top', ha='center')

    plt.tight_layout()
    plt.savefig('03_moving_target_problem.png', dpi=150)
    plt.show()
    print("Kaydedildi: 03_moving_target_problem.png")

    print("""
    ANALIZ:
    -------
    Sabit Hedef:
        - x hizla hedefe yaklasir
        - Kararli yakinasma

    Hareket Eden Hedef:
        - Hedef x'e bagli oldugu icin surekli degisiyor
        - x asla "doğru" degere ulasamiyor
        - Potansiyel olarak iraksaklik (divergence)

    DQN'de de ayni sorun var:
        Target = r + gamma * max Q(s', a'; theta)

    theta her guncellemede degisince hedef de degisiyor!
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
# BOLUM 3: HARD VS SOFT GUNCELLEME
# =============================================================================

def hard_update(target_net: nn.Module, policy_net: nn.Module):
    """
    Hard (Sert) Guncelleme: Tum agirliklari kopyala.

    target_net <- policy_net

    Orijinal DQN makalesinde kullanildi.
    Her C adimda bir kez cagrilir.
    """
    target_net.load_state_dict(policy_net.state_dict())


def soft_update(target_net: nn.Module, policy_net: nn.Module, tau: float = 0.005):
    """
    Soft (Yumusak) Guncelleme: Kademeli karistirma.

    target_net <- tau * policy_net + (1 - tau) * target_net

    Her adimda cagrilir.
    tau genellikle 0.001 - 0.01 arasi.

    Avantajlar:
    - Daha yumusak gecis
    - Ani degisiklik yok
    - Genellikle daha kararli
    """
    for target_param, policy_param in zip(target_net.parameters(),
                                          policy_net.parameters()):
        target_param.data.copy_(
            tau * policy_param.data + (1 - tau) * target_param.data
        )


def compare_update_methods():
    """
    Hard vs Soft guncelleme karsilastirmasi.

    Agirlik degisimlerini zaman icerisinde gorsellestir.
    """
    print("\n" + "="*60)
    print("HARD VS SOFT GUNCELLEME KARSILASTIRMASI")
    print("="*60)

    # Ornek aglar
    state_size, action_size = 4, 2
    policy_net = DQN(state_size, action_size)
    target_net_hard = DQN(state_size, action_size)
    target_net_soft = DQN(state_size, action_size)

    # Baslangicta ayni
    hard_update(target_net_hard, policy_net)
    hard_update(target_net_soft, policy_net)

    # Optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

    # Takip listleri
    policy_weights = []
    target_hard_weights = []
    target_soft_weights = []

    # Ilk agirligi takip et
    def get_first_weight(net):
        return net.network[0].weight[0, 0].item()

    hard_update_freq = 20
    tau = 0.05

    for step in range(200):
        # Rastgele gradyan adimi (simulasyon)
        fake_input = torch.randn(1, state_size)
        fake_target = torch.randn(1, action_size)
        output = policy_net(fake_input)
        loss = nn.MSELoss()(output, fake_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Hard guncelleme (her 20 adimda)
        if step % hard_update_freq == 0:
            hard_update(target_net_hard, policy_net)

        # Soft guncelleme (her adimda)
        soft_update(target_net_soft, policy_net, tau)

        # Agirliklari kaydet
        policy_weights.append(get_first_weight(policy_net))
        target_hard_weights.append(get_first_weight(target_net_hard))
        target_soft_weights.append(get_first_weight(target_net_soft))

    # Gorsellestir
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hard guncelleme
    axes[0].plot(policy_weights, 'b-', label='Policy Network', alpha=0.7)
    axes[0].plot(target_hard_weights, 'r-', label='Target Network', linewidth=2)
    axes[0].set_title(f'Hard Guncelleme (her {hard_update_freq} adimda)')
    axes[0].set_xlabel('Adim')
    axes[0].set_ylabel('Agirlik Degeri')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Soft guncelleme
    axes[1].plot(policy_weights, 'b-', label='Policy Network', alpha=0.7)
    axes[1].plot(target_soft_weights, 'g-', label='Target Network', linewidth=2)
    axes[1].set_title(f'Soft Guncelleme (tau={tau})')
    axes[1].set_xlabel('Adim')
    axes[1].set_ylabel('Agirlik Degeri')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('03_hard_vs_soft_update.png', dpi=150)
    plt.show()
    print("Kaydedildi: 03_hard_vs_soft_update.png")

    print("""
    KARSILASTIRMA:
    --------------
    Hard Guncelleme:
        - Target network ani atlamalar yapar
        - Policy degisince bir sure tutarsiz kalir
        - Orijinal DQN'de kullanildi

    Soft Guncelleme:
        - Target network yumusak gecis yapar
        - Her zaman policy'ye yakin kalir
        - DDPG, TD3, SAC gibi algoritmalarda yaygin
    """)


# =============================================================================
# BOLUM 4: TARGET NETWORK ILE/OLMADAN EGITIM
# =============================================================================

def train_dqn(
    use_target_network: bool = True,
    target_update_freq: int = 100,
    episodes: int = 200
) -> Tuple[List[float], List[float]]:
    """
    Target network ile/olmadan DQN egitimi.

    Args:
        use_target_network: Target network kullan?
        target_update_freq: Hard guncelleme frekansi
        episodes: Toplam episode sayisi

    Returns:
        (rewards_history, q_value_history)
    """
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

    rewards_history = []
    q_value_history = []  # Q degerlerini takip et
    total_steps = 0

    for episode in range(episodes):
        state, _ = env.reset()
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

                # Q hedefleri
                with torch.no_grad():
                    if use_target_network:
                        # Target network'ten sonraki Q degerleri
                        next_q = target_net(next_states_t).max(1)[0]
                    else:
                        # Policy network'ten (hareket eden hedef!)
                        next_q = policy_net(next_states_t).max(1)[0]

                    targets = rewards_t + gamma * next_q * (1 - dones_t)

                # Mevcut Q degerleri
                current_q = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

                # Q degerini kaydet
                q_value_history.append(current_q.mean().item())

                # Loss ve guncelleme
                loss = nn.MSELoss()(current_q, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Target network guncelleme
                if use_target_network and total_steps % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            mode = "Target" if use_target_network else "No Target"
            print(f"[{mode}] Episode {episode + 1}: Avg = {avg_reward:.1f}")

    env.close()
    return rewards_history, q_value_history


def compare_target_network_effect():
    """
    Target network etkisini karsilastir.
    """
    print("\n" + "="*60)
    print("TARGET NETWORK ETKISI KARSILASTIRMASI")
    print("="*60)
    print("(Bu biraz zaman alabilir...)")

    # Target network ile
    print("\n[1/2] Target network ILE egitim...")
    rewards_with, q_with = train_dqn(use_target_network=True, episodes=150)

    # Target network olmadan
    print("\n[2/2] Target network OLMADAN egitim...")
    rewards_without, q_without = train_dqn(use_target_network=False, episodes=150)

    # Gorsellestir
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ogrenme egrileri
    window = 10
    def moving_avg(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    ma_with = moving_avg(rewards_with, window)
    ma_without = moving_avg(rewards_without, window)

    axes[0].plot(range(window-1, len(rewards_with)), ma_with,
                label='Target Network ile', color='green', linewidth=2)
    axes[0].plot(range(window-1, len(rewards_without)), ma_without,
                label='Target Network olmadan', color='red', linewidth=2)
    axes[0].axhline(y=195, color='blue', linestyle='--', alpha=0.5, label='Cozum Esigi')
    axes[0].set_title('Ogrenme Egrileri')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Toplam Odul')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Q-deger degisimi
    axes[1].plot(q_with[:2000], alpha=0.7, label='Target Network ile', color='green')
    axes[1].plot(q_without[:2000], alpha=0.7, label='Target Network olmadan', color='red')
    axes[1].set_title('Q-Deger Degisimi (Ilk 2000 Adim)')
    axes[1].set_xlabel('Egitim Adimi')
    axes[1].set_ylabel('Ortalama Q-Degeri')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('03_target_network_effect.png', dpi=150)
    plt.show()
    print("Kaydedildi: 03_target_network_effect.png")


# =============================================================================
# BOLUM 5: Q-DEGER KARARLILIK ANALIZI
# =============================================================================

def analyze_q_stability():
    """
    Q-deger kararliligini analiz et.

    Target network olmadan Q-degerleri cok dalgalanabilir.
    """
    print("\n" + "="*60)
    print("Q-DEGER KARARLILIK ANALIZI")
    print("="*60)

    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Sabit test durumu
    test_state = torch.FloatTensor([0.0, 0.0, 0.0, 0.0]).unsqueeze(0)

    q_values_with_target = []
    q_values_without_target = []

    # Target network ile
    print("Target network ile analiz...")
    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer()

    state, _ = env.reset()
    for step in range(3000):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)

        if done:
            state, _ = env.reset()
        else:
            state = next_state

        if len(replay_buffer) >= 64:
            states, actions, rewards, next_states, dones = replay_buffer.sample(64)
            states_t = torch.FloatTensor(states)
            actions_t = torch.LongTensor(actions)
            rewards_t = torch.FloatTensor(rewards)
            next_states_t = torch.FloatTensor(next_states)
            dones_t = torch.FloatTensor(dones)

            with torch.no_grad():
                next_q = target_net(next_states_t).max(1)[0]
                targets = rewards_t + 0.99 * next_q * (1 - dones_t)

            current_q = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
            loss = nn.MSELoss()(current_q, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Test durumu icin Q-degeri kaydet
        with torch.no_grad():
            q = policy_net(test_state).max().item()
            q_values_with_target.append(q)

    env.close()

    # Target network olmadan
    print("Target network olmadan analiz...")
    env = gym.make("CartPole-v1")
    policy_net = DQN(state_size, action_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer()

    state, _ = env.reset()
    for step in range(3000):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)

        if done:
            state, _ = env.reset()
        else:
            state = next_state

        if len(replay_buffer) >= 64:
            states, actions, rewards, next_states, dones = replay_buffer.sample(64)
            states_t = torch.FloatTensor(states)
            actions_t = torch.LongTensor(actions)
            rewards_t = torch.FloatTensor(rewards)
            next_states_t = torch.FloatTensor(next_states)
            dones_t = torch.FloatTensor(dones)

            with torch.no_grad():
                # Ayni ag ile hedef hesapla
                next_q = policy_net(next_states_t).max(1)[0]
                targets = rewards_t + 0.99 * next_q * (1 - dones_t)

            current_q = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
            loss = nn.MSELoss()(current_q, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test durumu icin Q-degeri kaydet
        with torch.no_grad():
            q = policy_net(test_state).max().item()
            q_values_without_target.append(q)

    env.close()

    # Gorsellestir
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(q_values_with_target, alpha=0.7, color='green')
    axes[0].set_title('Target Network ile Q-Degeri Degisimi')
    axes[0].set_xlabel('Adim')
    axes[0].set_ylabel('Q(test_state)')
    axes[0].grid(True, alpha=0.3)

    # Varyans hesapla
    var_with = np.var(q_values_with_target[-1000:])
    axes[0].text(0.95, 0.95, f'Son 1000 adim varyans: {var_with:.4f}',
                transform=axes[0].transAxes, fontsize=10,
                verticalalignment='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat'))

    axes[1].plot(q_values_without_target, alpha=0.7, color='red')
    axes[1].set_title('Target Network olmadan Q-Degeri Degisimi')
    axes[1].set_xlabel('Adim')
    axes[1].set_ylabel('Q(test_state)')
    axes[1].grid(True, alpha=0.3)

    var_without = np.var(q_values_without_target[-1000:])
    axes[1].text(0.95, 0.95, f'Son 1000 adim varyans: {var_without:.4f}',
                transform=axes[1].transAxes, fontsize=10,
                verticalalignment='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    plt.savefig('03_q_value_stability.png', dpi=150)
    plt.show()
    print("Kaydedildi: 03_q_value_stability.png")


# =============================================================================
# ANA PROGRAM
# =============================================================================

def main():
    """Ana fonksiyon."""
    print("\n" + "="*60)
    print("TARGET NETWORK (HEDEF AG)")
    print("="*60)
    print("""
    Target Network, DQN'in ikinci kritik bilesenidir.

    SORUN: Hareket Eden Hedef
    -------------------------
    Q-learning hedefi: target = r + gamma * max Q(s', a')

    Eger Q(s', a') ayni ag ile hesaplaniyorsa:
    - Ag her guncellediginde hedef de degisir
    - "Hareket eden hedefi kovalamak" gibi
    - Kararsiz ve ıraksaklıga yol acabilir

    COZUM: Target Network
    ---------------------
    Iki ayri ag kullan:
    - Policy Network (θ): Her adimda guncellenir
    - Target Network (θ⁻): Periyodik olarak kopyalanir

    target = r + gamma * max Q(s', a'; θ⁻)

    Bu sayede hedef bir sure sabit kalir!

    Guncelleme Yontemleri:
    ----------------------
    1. Hard Update: Her C adimda θ⁻ <- θ
    2. Soft Update: Her adimda θ⁻ <- τ*θ + (1-τ)*θ⁻
    """)

    # 1. Hareket eden hedef problemini goster
    demonstrate_moving_target()

    # 2. Hard vs Soft guncelleme karsilastirmasi
    compare_update_methods()

    # 3. Q-deger kararlilik analizi
    analyze_q_stability()

    # 4. Ogrenme karsilastirmasi
    compare_target_network_effect()

    print("\n" + "="*60)
    print("SONUC")
    print("="*60)
    print("""
    Target Network'un Etkileri:

    1. KARARLI HEDEFLER:
       - Hedef bir sure sabit kalir
       - Gradyanlar daha anlamli

    2. AZALAN VARYANS:
       - Q-degerleri daha az dalgalanir
       - Daha guvenilir tahminler

    3. IYILESEN OGRENME:
       - Daha hizli yakinasma
       - Daha yuksek final performans

    ONEMLI NOKTALAR:
    - Orijinal DQN: Hard update (her 10000 adimda)
    - Modern algoritmalar: Soft update (her adimda, tau=0.005)
    - Guncelleme frekansi onemli bir hiperparametre

    Siradaki: Double DQN (asiri Q-degeri tahminini azaltmak)
    """)


if __name__ == "__main__":
    main()
