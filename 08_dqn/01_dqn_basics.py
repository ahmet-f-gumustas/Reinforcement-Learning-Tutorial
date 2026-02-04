"""
Deep Q-Network (DQN) Temelleri
==============================
Bu script, DQN'in temel bilesenlerini tanitir:
1. Sinir agi ile Q-fonksiyonu yaklasiklama
2. Epsilon-greedy aksiyon secimi
3. Bellman denkleminden ogrenme
4. Basit bir ortamda test

DQN Nedir?
    Geleneksel Q-learning'de Q-tablosu kullanilir:
    Q[durum][aksiyon] = deger

    Ancak buyuk/surekli durum uzaylarinda bu imkansiz!
    DQN, Q-fonksiyonunu bir sinir agi ile yaklasiklar:
    Q(s, a; theta) = NN(s)[a]

Kullanim:
    python 01_dqn_basics.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Tuple


# =============================================================================
# BOLUM 1: BASIT Q-AĞI (Q-NETWORK)
# =============================================================================

class SimpleQNetwork(nn.Module):
    """
    En basit Q-agi yapisi.

    Mimari:
        Girdi (durum) -> Gizli Katman -> Cikti (Q-degerler)

    Ornegin CartPole icin:
        4 ozellik -> 64 noron -> 2 aksiyon Q-degeri
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        """
        Args:
            state_size: Durum vektorunun boyutu
            action_size: Mumkun aksiyon sayisi
            hidden_size: Gizli katmandaki noron sayisi
        """
        super(SimpleQNetwork, self).__init__()

        # Tek gizli katmanli basit ag
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
            # NOT: Ciktida aktivasyon YOK - Q degerleri herhangi bir reel sayi olabilir
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Ileri gecis - durumdan Q-degerleri hesapla.

        Args:
            state: Durum tensoru [batch_size, state_size]

        Returns:
            Q-degerleri [batch_size, action_size]
        """
        return self.network(state)


# =============================================================================
# BOLUM 2: DAHA DERIN Q-AĞI
# =============================================================================

class DQN(nn.Module):
    """
    Standart DQN mimarisi - 2 gizli katman.

    Neden birden fazla katman?
        - Daha karmasik oruntuleri ogrenebilir
        - Her katman soyutlama seviyesini arttirir
        - Genel olarak daha iyi performans
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ileri gecis.

        ReLU aktivasyonu: max(0, x)
        - Basit ve etkili
        - Gradyan sorunlarini azaltir
        - Seyrek aktivasyon (bazilari 0)
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# =============================================================================
# BOLUM 3: EPSILON-GREEDY AKSIYON SECIMI
# =============================================================================

def epsilon_greedy_action(
    q_network: nn.Module,
    state: np.ndarray,
    epsilon: float,
    action_size: int
) -> int:
    """
    Epsilon-greedy stratejisi ile aksiyon sec.

    Epsilon-Greedy:
        - epsilon olasilikla: rastgele aksiyon (KESIF)
        - 1-epsilon olasilikla: en iyi aksiyon (SOMURU)

    Neden hem kesif hem somuru?
        - Sadece kesif: hicbir sey ogrenmez, rastgele hareket eder
        - Sadece somuru: yerel optimumlara takilir, daha iyi yolu bulamaz
        - Ikisinin dengesi: hem ogrenir hem kesfeder

    Args:
        q_network: Q-degerleri tahmin eden sinir agi
        state: Mevcut durum
        epsilon: Kesif orani (0-1 arasi)
        action_size: Mumkun aksiyon sayisi

    Returns:
        Secilen aksiyon indeksi
    """
    # Kesif: rastgele aksiyon
    if np.random.random() < epsilon:
        return np.random.randint(action_size)

    # Somuru: en yuksek Q-degerli aksiyon
    with torch.no_grad():  # Cikarim sirasinda gradyan hesaplama
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Batch boyutu ekle
        q_values = q_network(state_tensor)
        return q_values.argmax().item()


def epsilon_decay_schedule(
    episode: int,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995
) -> float:
    """
    Epsilon azaltma programi.

    Baslangiçta cok kesif yap, zamanla azalt.

    Neden epsilon azalir?
        - Basta: ortami tanimiyoruz, cok kesfetmeliyiz
        - Sonra: ogrenmis seylerden faydalanalim
        - Ama tamamen durma: yeni seyler olabilir

    Args:
        episode: Mevcut episode numarasi
        epsilon_start: Baslangic epsilon degeri
        epsilon_end: Minimum epsilon degeri
        epsilon_decay: Her episode'da carpan

    Returns:
        Guncel epsilon degeri
    """
    epsilon = epsilon_start * (epsilon_decay ** episode)
    return max(epsilon_end, epsilon)


# =============================================================================
# BOLUM 4: Q-OĞRENME GUNCELLEME
# =============================================================================

def compute_q_target(
    reward: float,
    next_state: np.ndarray,
    done: bool,
    q_network: nn.Module,
    gamma: float = 0.99
) -> float:
    """
    Bellman denklemi ile Q-hedefini hesapla.

    Bellman Denklemi:
        Q(s, a) = r + γ * max_a' Q(s', a')

    Anlami:
        Bir durumda bir aksiyonun degeri =
        anlik odul + gelecekteki en iyi aksiyonlarin indirili degeri

    Args:
        reward: Alinan odul
        next_state: Sonraki durum
        done: Episode bitti mi?
        q_network: Q-degerleri tahmin eden ag
        gamma: Indirim faktoru

    Returns:
        Q-hedef degeri
    """
    if done:
        # Terminal durumda gelecek yok
        return reward

    with torch.no_grad():
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        next_q_values = q_network(next_state_tensor)
        max_next_q = next_q_values.max().item()

    return reward + gamma * max_next_q


def train_step(
    q_network: nn.Module,
    optimizer: optim.Optimizer,
    state: np.ndarray,
    action: int,
    target: float
) -> float:
    """
    Tek bir egitim adimi gerceklestir.

    DQN Loss:
        L = (Q_hedef - Q_tahmin)^2

    Geri yayilim:
        1. Loss hesapla
        2. Gradyanlari hesapla
        3. Agirliklari guncelle

    Args:
        q_network: Egitilecek ag
        optimizer: Optimizer (Adam, SGD, vb.)
        state: Mevcut durum
        action: Alinan aksiyon
        target: Hesaplanan Q-hedef

    Returns:
        Loss degeri
    """
    # Tensore donustur
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    target_tensor = torch.FloatTensor([target])

    # Q-tahminini al
    q_values = q_network(state_tensor)
    q_value = q_values[0, action]

    # Loss hesapla (MSE)
    loss = nn.MSELoss()(q_value, target_tensor[0])

    # Geri yayilim
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# =============================================================================
# BOLUM 5: BASIT ORNEK - GRIDWORLD
# =============================================================================

class SimpleGridWorld:
    """
    Basit 4x4 grid ortami.

    Izgara:
        S . . .
        . X . .
        . . . .
        . . . G

    S: Baslangic (0,0)
    G: Hedef (3,3) - odul +10
    X: Engel (1,1) - odul -10
    .: Bos - odul -0.1 (uzun yollari cezalandir)

    Aksiyonlar: 0=yukari, 1=asagi, 2=sol, 3=sag
    """

    def __init__(self):
        self.grid_size = 4
        self.state_size = 2  # (x, y) pozisyonu
        self.action_size = 4

        self.start = (0, 0)
        self.goal = (3, 3)
        self.obstacle = (1, 1)

        self.reset()

    def reset(self) -> np.ndarray:
        """Ortami baslangic durumuna getir."""
        self.position = list(self.start)
        self.steps = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Normalize edilmis durum vektorunu dondur."""
        return np.array(self.position, dtype=np.float32) / (self.grid_size - 1)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Bir adim at.

        Args:
            action: 0=yukari, 1=asagi, 2=sol, 3=sag

        Returns:
            (next_state, reward, done)
        """
        self.steps += 1

        # Hareket yonleri
        moves = {
            0: (-1, 0),  # yukari
            1: (1, 0),   # asagi
            2: (0, -1),  # sol
            3: (0, 1)    # sag
        }

        # Yeni pozisyonu hesapla
        dx, dy = moves[action]
        new_x = max(0, min(self.grid_size - 1, self.position[0] + dx))
        new_y = max(0, min(self.grid_size - 1, self.position[1] + dy))

        self.position = [new_x, new_y]

        # Odul ve bitiş kontrolu
        if tuple(self.position) == self.goal:
            return self._get_state(), 10.0, True
        elif tuple(self.position) == self.obstacle:
            return self._get_state(), -10.0, True
        elif self.steps >= 50:  # Maksimum adim
            return self._get_state(), -1.0, True
        else:
            return self._get_state(), -0.1, False


# =============================================================================
# BOLUM 6: EGITIM DONGUSU
# =============================================================================

def train_dqn_simple(episodes: int = 500) -> List[float]:
    """
    Basit DQN egitimi (replay buffer ve target network OLMADAN).

    Bu basit versiyon DQN'in temellerini gosterir.
    Gercek uygulamalarda replay buffer ve target network SART!

    Args:
        episodes: Toplam episode sayisi

    Returns:
        Her episode'un odul listesi
    """
    # Ortam ve ag olustur
    env = SimpleGridWorld()
    q_network = SimpleQNetwork(env.state_size, env.action_size)
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)

    # Egitim parametreleri
    gamma = 0.99
    rewards_history = []

    print("="*60)
    print("BASIT DQN EGITIMI (GridWorld)")
    print("="*60)
    print(f"Durum boyutu: {env.state_size}")
    print(f"Aksiyon sayisi: {env.action_size}")
    print(f"Toplam episode: {episodes}")
    print("="*60)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        # Epsilon hesapla
        epsilon = epsilon_decay_schedule(episode)

        while not done:
            # Aksiyon sec
            action = epsilon_greedy_action(q_network, state, epsilon, env.action_size)

            # Adim at
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Q-hedefini hesapla
            target = compute_q_target(reward, next_state, done, q_network, gamma)

            # Ag'i guncelle
            train_step(q_network, optimizer, state, action, target)

            state = next_state

        rewards_history.append(total_reward)

        # Ilerleme raporu
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode + 1:4d} | "
                  f"Ort. Odul: {avg_reward:7.2f} | "
                  f"Epsilon: {epsilon:.3f}")

    print("="*60)
    print("EGITIM TAMAMLANDI!")
    print(f"Son 50 episode ortalama odul: {np.mean(rewards_history[-50:]):.2f}")
    print("="*60)

    return rewards_history


# =============================================================================
# BOLUM 7: GORSELLEŞTIRME
# =============================================================================

def visualize_q_values(q_network: nn.Module, grid_size: int = 4):
    """
    Ogrenilen Q-degerlerini gorsellestir.

    Her hucre icin en iyi aksiyonun Q-degerini gosterir.
    Oklar en iyi aksiyon yonunu gosterir.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Her pozisyon icin Q-degerlerini hesapla
    q_grid = np.zeros((grid_size, grid_size))
    policy_grid = np.zeros((grid_size, grid_size), dtype=int)

    action_arrows = {
        0: '↑',  # yukari
        1: '↓',  # asagi
        2: '←',  # sol
        3: '→'   # sag
    }

    for i in range(grid_size):
        for j in range(grid_size):
            state = np.array([i, j], dtype=np.float32) / (grid_size - 1)
            with torch.no_grad():
                q_values = q_network(torch.FloatTensor(state).unsqueeze(0))
                q_grid[i, j] = q_values.max().item()
                policy_grid[i, j] = q_values.argmax().item()

    # Q-degerleri isisi haritasi
    im = ax.imshow(q_grid, cmap='RdYlGn', interpolation='nearest')

    # Her hucreye ok ve Q-degeri ekle
    for i in range(grid_size):
        for j in range(grid_size):
            arrow = action_arrows[policy_grid[i, j]]
            text = f"{arrow}\n{q_grid[i, j]:.1f}"

            # Ozel hucreler
            if (i, j) == (0, 0):
                text = f"S\n{arrow}\n{q_grid[i, j]:.1f}"
            elif (i, j) == (3, 3):
                text = f"G\n{q_grid[i, j]:.1f}"
            elif (i, j) == (1, 1):
                text = f"X\n{q_grid[i, j]:.1f}"

            ax.text(j, i, text, ha='center', va='center',
                   fontsize=12, fontweight='bold')

    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_title('Ogrenilen Q-Degerleri ve Politika\n(S=Baslangic, G=Hedef, X=Engel)')

    plt.colorbar(im, label='Q-Degeri')
    plt.tight_layout()
    plt.savefig('01_q_values_visualization.png', dpi=150)
    plt.show()
    print("Gorsel kaydedildi: 01_q_values_visualization.png")


def plot_training_progress(rewards: List[float]):
    """Egitim ilerlemesini ciz."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ham oduller
    axes[0].plot(rewards, alpha=0.6, label='Episode Odulu')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Toplam Odul')
    axes[0].set_title('Egitim Ilerlemesi')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Hareketli ortalama
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(rewards)), moving_avg,
                    color='red', label=f'{window}-Episode Hareketli Ortalama')
        axes[1].axhline(y=9.0, color='green', linestyle='--', label='Cozum Esigi (~9)')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Ortalama Odul')
        axes[1].set_title(f'{window}-Episode Hareketli Ortalama')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('01_training_progress.png', dpi=150)
    plt.show()
    print("Gorsel kaydedildi: 01_training_progress.png")


# =============================================================================
# ANA PROGRAM
# =============================================================================

def main():
    """Ana fonksiyon."""
    print("\n" + "="*60)
    print("DEEP Q-NETWORK (DQN) TEMELLERI")
    print("="*60)
    print("""
    Bu script DQN'in temel kavramlarini gosterir:

    1. Q-Agi (Q-Network):
       - Durum -> Q-degerler donusumu
       - Her aksiyon icin bir Q-degeri

    2. Epsilon-Greedy:
       - Kesif vs Somuru dengesi
       - Zamanla daha az kesif

    3. Bellman Denklemi:
       - Q(s,a) = r + gamma * max Q(s',a')
       - Anlik odul + gelecek deger

    NOT: Bu BASIT bir DQN'dir!
    Gercek DQN icin Experience Replay ve Target Network gerekir.
    Bunlar sonraki orneklerde anlatilacak.
    """)

    # Egitimi calistir
    rewards = train_dqn_simple(episodes=500)

    # Sonuclari gorsellestir
    print("\nGorsellestirme olusturuluyor...")

    # Egitim ilerlemesi
    plot_training_progress(rewards)

    # Q-degerlerini gorsellestir
    env = SimpleGridWorld()
    q_network = SimpleQNetwork(env.state_size, env.action_size)
    # Tekrar egit (visualization icin)
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    for episode in range(500):
        state = env.reset()
        done = False
        epsilon = epsilon_decay_schedule(episode)
        while not done:
            action = epsilon_greedy_action(q_network, state, epsilon, env.action_size)
            next_state, reward, done = env.step(action)
            target = compute_q_target(reward, next_state, done, q_network)
            train_step(q_network, optimizer, state, action, target)
            state = next_state

    visualize_q_values(q_network)

    print("\n" + "="*60)
    print("SONUC")
    print("="*60)
    print("""
    Basit DQN GridWorld ortaminda ogrenmeyi basardi!

    Onemli Noktalar:
    1. Sinir agi Q-tablosu yerine gecti
    2. Epsilon-greedy kesif/somuru dengesi sagladi
    3. Bellman denklemi ogrenme hedefini belirledi

    Eksikler (sonraki orneklerde):
    - Experience Replay: Korelasyonu kirma
    - Target Network: Kararlı hedefler

    Bir sonraki dosyada Experience Replay'i ogrenecegiz!
    """)


if __name__ == "__main__":
    main()
