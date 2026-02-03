"""
Acrobot DQN - Salınan Robot Kol Simülasyonu
============================================
Bu örnek, Acrobot-v1 ortamında DQN algoritması kullanarak bir robot kolun
yukarı sallanmasını öğretir.

Acrobot Nedir?
    - İki bağlantılı bir robot kol sistemi
    - Alt bağlantı sabit bir noktaya bağlı
    - Üst bağlantı alt bağlantıya bağlı
    - Amaç: Uç noktayı belirli bir yüksekliğe ulaştırmak
    - Sadece orta ekleme tork uygulanabilir (alt eklem pasif)

Modlar:
    1. Human Mode: Klavye ile kontrol
    2. AI Mode: Eğitilmiş DQN ajanını izle
    3. Training Mode: DQN ajanını eğit
    4. Demo Mode: Hızlı eğitim + AI gösterimi

Kontroller (Human Mode):
    - Sol Ok / A: Negatif tork (-1)
    - Sağ Ok / D: Pozitif tork (+1)
    - Hiçbir şey: Tork yok (0)
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
import time
import argparse

# =============================================================================
# YAPILANDIRMA (CONFIGURATION)
# =============================================================================

# Çıktı klasörü - modeller ve loglar buraya kaydedilir
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = Path(__file__).parent / "output" / SCRIPT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cihaz seçimi - GPU varsa kullan, yoksa CPU
# GPU eğitimi çok daha hızlı yapar (10x veya daha fazla)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# TERMİNAL RENKLERİ - Çıktıyı daha okunabilir yapar
# =============================================================================

class Colors:
    """Terminal çıktısı için ANSI renk kodları."""
    HEADER = '\033[95m'      # Mor
    BLUE = '\033[94m'        # Mavi
    CYAN = '\033[96m'        # Camgöbeği
    GREEN = '\033[92m'       # Yeşil
    YELLOW = '\033[93m'      # Sarı
    RED = '\033[91m'         # Kırmızı
    BOLD = '\033[1m'         # Kalın
    DIM = '\033[2m'          # Soluk
    RESET = '\033[0m'        # Varsayılana dön

    @staticmethod
    def colorize(text, color):
        """Metni renk kodlarıyla sarmala."""
        return f"{color}{text}{Colors.RESET}"


# =============================================================================
# EĞİTİM LOGGER - Güzel eğitim çıktısı
# =============================================================================

class TrainingLogger:
    """
    Tüm eğitim çıktı formatlamasını yönetir.

    Özellikler:
        - İlerleme çubuğu
        - Renk kodlu metrikler
        - Trend göstergeleri (↑ ↓ →)
        - Periyodik özetler
        - Zaman takibi
    """

    def __init__(self, total_episodes):
        self.total_episodes = total_episodes
        self.start_time = None
        self.episode_start_time = None

        # İstatistik takibi
        self.rewards = []
        self.losses = []
        self.episode_steps = []
        self.epsilons = []

        # En iyi değerler
        self.best_reward = float('-inf')
        self.best_avg = float('-inf')

    def start_training(self):
        """Eğitim başlangıcında çağrılır."""
        self.start_time = time.time()
        self._print_header()

    def _print_header(self):
        """Eğitim başlığını yazdır."""
        print("\n" + "=" * 90)
        print(Colors.colorize("  ACROBOT DQN - EĞİTİM BAŞLADI", Colors.BOLD + Colors.CYAN))
        print("=" * 90)
        print(f"  Toplam Episode : {Colors.colorize(str(self.total_episodes), Colors.YELLOW)}")
        print(f"  Cihaz          : {Colors.colorize(str(device), Colors.GREEN)}")
        print(f"  Çıktı Klasörü  : {Colors.colorize(str(OUTPUT_DIR), Colors.DIM)}")
        print("=" * 90)
        print()

        # Sütun başlıkları
        header = (
            f"{'Episode':>8} | "
            f"{'İlerleme':>14} | "
            f"{'Skor':>8} | "
            f"{'Ort(100)':>10} | "
            f"{'En İyi':>8} | "
            f"{'Adım':>6} | "
            f"{'Loss':>10} | "
            f"{'Epsilon':>8} | "
            f"{'Süre':>6}"
        )
        print(Colors.colorize(header, Colors.DIM))
        print("-" * 105)

    def start_episode(self):
        """Her episode başlangıcında çağrılır."""
        self.episode_start_time = time.time()

    def log_episode(self, episode, reward, steps, loss, epsilon, saved=False):
        """
        Episode sonuçlarını güzel formatlı olarak logla.

        Args:
            episode: Mevcut episode numarası
            reward: Bu episode'daki toplam ödül
            steps: Bu episode'daki adım sayısı
            loss: Ortalama loss değeri
            epsilon: Mevcut keşif oranı
            saved: Model bu episode'da kaydedildi mi
        """
        # İstatistikleri sakla
        self.rewards.append(reward)
        self.episode_steps.append(steps)
        self.epsilons.append(epsilon)
        if loss > 0:
            self.losses.append(loss)

        # Metrikleri hesapla
        avg_reward = np.mean(self.rewards[-100:]) if self.rewards else 0
        episode_time = time.time() - self.episode_start_time

        # En iyi değerleri güncelle
        if reward > self.best_reward:
            self.best_reward = reward
        if avg_reward > self.best_avg:
            self.best_avg = avg_reward

        # Trend göstergesi (son 10 episode ortalamasına göre)
        trend = self._get_trend()

        # İlerleme çubuğu
        progress = self._make_progress_bar(episode, self.total_episodes, width=12)

        # Performansa göre skoru renklendir
        score_str = self._color_score(reward)
        avg_str = self._color_avg(avg_reward)
        best_str = Colors.colorize(f"{self.best_reward:>8.0f}", Colors.YELLOW)

        # Loss formatla
        loss_str = f"{loss:>10.4f}" if loss > 0 else "     -    "

        # Epsilon formatla
        eps_str = f"{epsilon:>8.4f}"

        # Süre formatla
        time_str = f"{episode_time:>5.1f}s"

        # Çıktı satırını oluştur
        save_indicator = Colors.colorize(" [SAVED]", Colors.GREEN) if saved else ""

        line = (
            f"{episode:>8} | "
            f"{progress} | "
            f"{score_str} | "
            f"{avg_str} {trend} | "
            f"{best_str} | "
            f"{steps:>6} | "
            f"{loss_str} | "
            f"{eps_str} | "
            f"{time_str}"
            f"{save_indicator}"
        )
        print(line)

        # Her 50 episode'da özet yazdır
        if episode > 0 and episode % 50 == 0:
            self._print_summary(episode)

    def _make_progress_bar(self, current, total, width=20):
        """Görsel ilerleme çubuğu oluştur."""
        percent = current / total
        filled = int(width * percent)
        empty = width - filled

        # Düzgün ilerleme için blok karakterleri kullan
        bar = "█" * filled + "░" * empty
        percent_str = f"{percent*100:>5.1f}%"

        return Colors.colorize(bar, Colors.CYAN) + " " + percent_str

    def _get_trend(self):
        """Son performansa göre trend göstergesi al."""
        if len(self.rewards) < 20:
            return " "

        recent = np.mean(self.rewards[-10:])
        previous = np.mean(self.rewards[-20:-10])

        # Acrobot'ta daha yüksek (daha az negatif) daha iyi
        if recent > previous + 5:
            return Colors.colorize("↑", Colors.GREEN)
        elif recent < previous - 5:
            return Colors.colorize("↓", Colors.RED)
        else:
            return Colors.colorize("→", Colors.YELLOW)

    def _color_score(self, score):
        """Performansa göre skoru renklendir."""
        # Acrobot'ta skor negatif, -100'e yakın çok iyi
        if score >= -100:
            return Colors.colorize(f"{score:>8.0f}", Colors.GREEN + Colors.BOLD)
        elif score >= -150:
            return Colors.colorize(f"{score:>8.0f}", Colors.GREEN)
        elif score >= -300:
            return Colors.colorize(f"{score:>8.0f}", Colors.YELLOW)
        elif score >= -400:
            return Colors.colorize(f"{score:>8.0f}", Colors.DIM)
        else:
            return Colors.colorize(f"{score:>8.0f}", Colors.RED)

    def _color_avg(self, avg):
        """Performansa göre ortalamayı renklendir."""
        if avg >= -100:
            return Colors.colorize(f"{avg:>8.1f}", Colors.GREEN + Colors.BOLD)
        elif avg >= -150:
            return Colors.colorize(f"{avg:>8.1f}", Colors.GREEN)
        elif avg >= -200:
            return Colors.colorize(f"{avg:>8.1f}", Colors.YELLOW)
        elif avg >= -300:
            return Colors.colorize(f"{avg:>8.1f}", Colors.DIM)
        else:
            return Colors.colorize(f"{avg:>8.1f}", Colors.RED)

    def _print_summary(self, episode):
        """Periyodik özet istatistikleri yazdır."""
        elapsed = time.time() - self.start_time
        eps_per_sec = episode / elapsed if elapsed > 0 else 0
        remaining = (self.total_episodes - episode) / eps_per_sec if eps_per_sec > 0 else 0

        print()
        print(Colors.colorize(f"  ┌─── Episode {episode} Özeti ───┐", Colors.CYAN))
        print(f"  │ Ort. Ödül (son 100)  : {Colors.colorize(f'{np.mean(self.rewards[-100:]):>8.1f}', Colors.YELLOW)}")
        print(f"  │ En İyi Tek Episode   : {Colors.colorize(f'{self.best_reward:>8.0f}', Colors.GREEN)}")
        print(f"  │ En İyi Ortalama (100): {Colors.colorize(f'{self.best_avg:>8.1f}', Colors.GREEN)}")

        if self.losses:
            avg_loss = np.mean(self.losses[-100:])
            print(f"  │ Ort. Loss            : {avg_loss:>8.4f}")

        print(f"  │ Ort. Adım/Episode    : {np.mean(self.episode_steps[-100:]):>8.1f}")
        print(f"  │ Geçen Süre           : {self._format_time(elapsed)}")
        print(f"  │ Tahmini Kalan        : {self._format_time(remaining)}")
        print(f"  │ Hız                  : {eps_per_sec:>6.2f} ep/s")
        print(Colors.colorize(f"  └{'─' * 26}┘", Colors.CYAN))
        print()

    def _format_time(self, seconds):
        """Saniyeyi okunabilir zaman formatına çevir."""
        if seconds < 60:
            return f"{seconds:>5.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins:>2}d {secs:02d}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours:>2}s {mins:02d}d"

    def finish_training(self):
        """Eğitim sonunda çağrılır."""
        elapsed = time.time() - self.start_time

        print()
        print("=" * 90)
        print(Colors.colorize("  EĞİTİM TAMAMLANDI!", Colors.BOLD + Colors.GREEN))
        print("=" * 90)
        print()
        print(f"  Toplam Episode    : {Colors.colorize(str(self.total_episodes), Colors.YELLOW)}")
        print(f"  Toplam Süre       : {Colors.colorize(self._format_time(elapsed), Colors.CYAN)}")
        print(f"  Ortalama Hız      : {Colors.colorize(f'{self.total_episodes/elapsed:.2f} ep/s', Colors.CYAN)}")
        print()
        print(f"  En İyi Episode    : {Colors.colorize(f'{self.best_reward:.0f}', Colors.GREEN + Colors.BOLD)}")
        print(f"  En İyi Ort. (100) : {Colors.colorize(f'{self.best_avg:.1f}', Colors.GREEN + Colors.BOLD)}")
        print(f"  Son Ort. (100)    : {Colors.colorize(f'{np.mean(self.rewards[-100:]):.1f}', Colors.YELLOW)}")
        print()

        # Performans değerlendirmesi
        if self.best_avg >= -100:
            print(Colors.colorize("  Durum: ÇÖZÜLDÜ! Robot kol mükemmel sallanıyor!", Colors.GREEN + Colors.BOLD))
        elif self.best_avg >= -150:
            print(Colors.colorize("  Durum: ÇOK İYİ! Robot kol iyi sallanıyor.", Colors.GREEN))
        elif self.best_avg >= -200:
            print(Colors.colorize("  Durum: İYİ. Daha fazla eğitim önerilir.", Colors.YELLOW))
        else:
            print(Colors.colorize("  Durum: ERKEN AŞAMA. Daha fazla eğitim gerekli.", Colors.RED))

        print()
        print("=" * 90)


# =============================================================================
# DERİN Q-AĞI (DEEP Q-NETWORK) MODELİ
# =============================================================================

class DQN(nn.Module):
    """
    Deep Q-Network - Q-değerlerini öğrenen bir sinir ağı.

    Q-değeri Nedir?
        - Bir durumda bir aksiyon almanın beklenen gelecek ödülü
        - Daha yüksek Q-değeri = daha iyi aksiyon

    Mimari:
        Girdi (6 özellik) -> Gizli (128) -> Gizli (128) -> Çıktı (3 aksiyon)

    6 Girdi Özelliği:
        - cos(theta1): İlk bağlantının açısının kosinüsü
        - sin(theta1): İlk bağlantının açısının sinüsü
        - cos(theta2): İkinci bağlantının açısının kosinüsü
        - sin(theta2): İkinci bağlantının açısının sinüsü
        - theta1_dot: İlk bağlantının açısal hızı
        - theta2_dot: İkinci bağlantının açısal hızı

    3 Çıktı Aksiyonu:
        - 0: Negatif tork uygula (-1)
        - 1: Tork uygulama (0)
        - 2: Pozitif tork uygula (+1)
    """

    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()

        # Üç tam bağlantılı katman
        # fc1: girdi -> gizli (temel özellikleri öğrenir)
        # fc2: gizli -> gizli (karmaşık örüntüleri öğrenir)
        # fc3: gizli -> çıktı (her aksiyon için Q-değerleri üretir)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        # ReLU aktivasyonu: x < 0 ise 0 çıktı ver, değilse x çıktı ver
        # Bu, ağın karmaşık örüntüleri öğrenebilmesi için doğrusal olmama ekler
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Çıktıda aktivasyon yok - Q-değerleri herhangi bir sayı olabilir
        return self.fc3(x)


# =============================================================================
# REPLAY BUFFER (DENEYİM BELLEĞİ)
# =============================================================================

class ReplayBuffer:
    """
    Replay Buffer - Eğitim için geçmiş deneyimleri saklar.

    Neden Buna İhtiyacımız Var?
        1. Ardışık örnekler arasındaki korelasyonu kırar
        2. Nadir deneyimlerin birden fazla kez kullanılmasını sağlar
        3. Eğitimi daha kararlı ve verimli yapar

    Her deneyim bir tuple: (durum, aksiyon, ödül, sonraki_durum, bitti)
    """

    def __init__(self, capacity=100000):
        # deque = çift uçlu kuyruk, eski öğeleri otomatik siler
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Belleğe yeni bir deneyim ekle."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Rastgele bir deneyim batch'i örnekle.
        Rastgele örnekleme ardışık deneyimler arasındaki korelasyonu kırar.
        """
        batch = random.sample(self.buffer, batch_size)
        # Batch'i ayrı dizilere ayır
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# DQN AJANI
# =============================================================================

class DQNAgent:
    """
    DQN Ajanı - Deep Q-Learning kullanarak oyunu öğrenir.

    Temel Kavramlar:
        - Policy Network (Politika Ağı): Kararlar verir (hangi aksiyonu alacak)
        - Target Network (Hedef Ağı): Eğitim için kararlı Q-değer hedefleri sağlar
        - Epsilon-Greedy: Keşif ve sömürü arasında denge

    Eğitim Döngüsü:
        1. Durumu gözlemle
        2. Aksiyon seç (epsilon-greedy)
        3. Aksiyonu al, ödül ve sonraki durumu al
        4. Deneyimi replay buffer'a sakla
        5. Buffer'dan rastgele batch örnekle
        6. Loss hesapla ve policy network'ü güncelle
        7. Periyodik olarak policy network'ü target network'e kopyala
    """

    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):

        self.state_size = state_size
        self.action_size = action_size

        # Hiperparametreler
        self.gamma = gamma              # İndirim faktörü (0.99 = geleceği önemse)
        self.epsilon = epsilon          # Keşif oranı (1.0 = %100 rastgele)
        self.epsilon_decay = epsilon_decay  # Keşifi ne kadar hızlı azalt
        self.epsilon_min = epsilon_min  # Minimum keşif oranı

        # İki ağ: policy (öğrenen) ve target (kararlı)
        # Neden iki ağ? "Hareket eden hedefi kovalama" sorununu önlemek için
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Adam optimizer - öğrenme oranını otomatik ayarlar
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Deneyim replay buffer'ı
        self.memory = ReplayBuffer()
        self.batch_size = 64

        # Her N adımda target network'ü güncelle
        self.update_target_every = 10
        self.steps = 0

        # Loss takibi
        self.last_loss = 0

    def choose_action(self, state, training=True):
        """
        Epsilon-greedy stratejisi kullanarak aksiyon seç.

        Epsilon-Greedy:
            - Epsilon olasılıkla: rastgele aksiyon al (keşfet)
            - 1-epsilon olasılıkla: en iyi aksiyonu al (sömür)

        Eğitim ilerledikçe epsilon azalır, böylece ajan daha az keşfeder
        ve öğrendiği bilgiyi daha çok kullanır.
        """
        # Keşif: rastgele aksiyon
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Sömürü: policy network'e göre en iyi aksiyon
        with torch.no_grad():  # Çıkarım için gradyan takibine gerek yok
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()  # En yüksek Q-değerli aksiyonu döndür

    def remember(self, state, action, reward, next_state, done):
        """Deneyimi replay buffer'a sakla."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Replay buffer'dan bir batch kullanarak bir eğitim adımı gerçekleştir.

        DQN Loss Fonksiyonu:
            Loss = (Q_tahmin - Q_hedef)^2

        Burada:
            Q_tahmin = policy_net(durum)[aksiyon]
            Q_hedef = ödül + gamma * max(target_net(sonraki_durum))

        Bu Bellman denklemidir - Q-learning'in temeli.
        """
        # Batch oluşturmak için yeterli örnek gerekli
        if len(self.memory) < self.batch_size:
            return 0

        # Bellekten rastgele batch örnekle
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Tensor'lara dönüştür ve GPU/CPU'ya taşı
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Policy network'ten mevcut Q değerlerini hesapla
        # gather() alınan aksiyon için Q-değerini seçer
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target network'ten hedef Q değerlerini hesapla
        with torch.no_grad():  # Target network'ü burada güncelleme
            # max(Q(sonraki_durum)) - en iyi olası gelecek değer
            next_q = self.target_net(next_states).max(1)[0]
            # Bellman denklemi: Q = ödül + gamma * max(Q_sonraki)
            # Bittiyse, gelecek ödül yok
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Mean Squared Error loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Geri yayılım - ağ ağırlıklarını güncelle
        self.optimizer.zero_grad()  # Eski gradyanları temizle
        loss.backward()             # Yeni gradyanları hesapla
        self.optimizer.step()       # Ağırlıkları güncelle

        # Epsilon'u azalt - zamanla daha az keşfet
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Periyodik olarak target network'ü güncelle
        # Bu, eğitim için kararlı hedefler sağlar
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.last_loss = loss.item()
        return loss.item()

    def save(self, path):
        """Modeli dosyaya kaydet."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        """Modeli dosyadan yükle."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.epsilon = checkpoint.get('epsilon', 0.01)
            print(f"Model yüklendi: {path}")
            return True
        return False


# =============================================================================
# OYUN SINIFI
# =============================================================================

class AcrobotGame:
    """
    Acrobot için ana oyun sınıfı.

    Acrobot Ortamı:
        - Amaç: Uç noktayı yatay çizginin üzerine çıkarmak
        - Ödül: Her adım için -1 (daha hızlı = daha iyi)
        - Episode sona erer: Hedef yüksekliğe ulaşılınca veya 500 adım
        - Çözüldü sayılır: 100 episode ortalaması >= -100
    """

    # İnsan tarafından okunabilir aksiyon isimleri
    ACTION_NAMES = ["Negatif Tork", "Tork Yok", "Pozitif Tork"]

    def __init__(self):
        # Görsel render ile oyun ortamını oluştur
        self.env = gym.make("Acrobot-v1", render_mode="human")

        # Durum uzayı: 6 sürekli değer
        self.state_size = self.env.observation_space.shape[0]
        # Aksiyon uzayı: 3 ayrık aksiyon
        self.action_size = self.env.action_space.n

        # DQN ajanını oluştur
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.model_path = OUTPUT_DIR / "dqn_model.pt"

        # Oyun durum değişkenleri
        self.state = None
        self.total_reward = 0
        self.episode = 0
        self.step_count = 0
        self.done = True
        self.mode = "human"  # "human", "ai", veya "train"

        # İstatistik takibi
        self.rewards_history = []
        self.best_reward = float('-inf')

    def reset(self):
        """Yeni episode için ortamı sıfırla."""
        self.state, _ = self.env.reset()
        self.total_reward = 0
        self.step_count = 0
        self.done = False
        self.episode += 1
        return self.state

    def get_human_action(self, keys):
        """
        Klavye girdisini oyun aksiyonuna dönüştür.

        Eşleme:
            Sol/A -> Negatif tork (saat yönünün tersine)
            Sağ/D -> Pozitif tork (saat yönünde)
            Hiçbiri -> Tork yok
        """
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            return 0  # Negatif tork
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            return 2  # Pozitif tork
        return 1  # Tork yok

    def step(self, action):
        """
        Bir oyun adımı yürüt.

        Ana oyun döngüsü:
            1. Ortamda aksiyonu al
            2. Ödül ve yeni durumu al
            3. Eğitim modundaysa, deneyimi sakla ve öğren
            4. Oyun durumunu güncelle
        """
        # Ortamda aksiyonu yürüt
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # İstatistikleri güncelle
        self.total_reward += reward
        self.step_count += 1

        # Eğitim modundaysa, bu deneyimden öğren
        if self.mode == "train":
            self.agent.remember(self.state, action, reward, next_state, done)
            self.agent.train_step()

        # Durumu güncelle
        self.state = next_state
        self.done = done

        # Episode sonuçlarını takip et
        if done:
            self.rewards_history.append(self.total_reward)
            if self.total_reward > self.best_reward:
                self.best_reward = self.total_reward

        return next_state, reward, done, info

    def run_human_mode(self):
        """
        İnsan kontrol modu - klavye ile oyna.

        İpuçları:
            - Momentum kazanmak için sallanmayı kullan
            - Doğru zamanda tork uygula
            - Amaç uç noktayı yatay çizginin üzerine çıkarmak
        """
        print("\n" + "=" * 60)
        print("ACROBOT - İNSAN MODU")
        print("=" * 60)
        print("Kontroller:")
        print("  - Sol Ok / A: Negatif tork (saat yönünün tersine)")
        print("  - Sağ Ok / D: Pozitif tork (saat yönünde)")
        print("  - Hiçbiri: Tork yok")
        print("  - R: Yeniden başlat")
        print("  - Q / ESC: Çıkış")
        print("=" * 60)
        print("İPUCU: Momentum kazanmak için sallanmayı kullanın!")
        print("=" * 60)

        self.mode = "human"
        pygame.init()

        running = True
        clock = pygame.time.Clock()
        self.reset()

        while running:
            # Olayları işle (çık, yeniden başlat, vb.)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset()
                        print(f"\n[Yeniden Başlatıldı] Episode {self.episode}")

            # Oyun bitmemişse, girdiyi işle ve adım at
            if not self.done:
                keys = pygame.key.get_pressed()
                action = self.get_human_action(keys)
                _, reward, done, _ = self.step(action)

                if done:
                    # Skor > -100 başarılı sayılır
                    result = "BAŞARILI!" if self.total_reward >= -100 else "BİTTİ"
                    print(f"Episode {self.episode}: {result} Skor: {self.total_reward:.0f} Adım: {self.step_count}")
            else:
                # Biraz bekle sonra yeniden başlat
                pygame.time.wait(1000)
                self.reset()

            # 30 FPS'e sınırla
            clock.tick(30)

        self.env.close()
        pygame.quit()

    def run_ai_mode(self, episodes=5):
        """
        AI kontrol modu - eğitilmiş ajanı izle.

        Eğitilmiş ajan öğrendiği Q-değerlerini kullanarak karar verir.
        Keşif yok (epsilon=0), sadece öğrenilen bilginin kullanımı.
        """
        print("\n" + "=" * 60)
        print("ACROBOT - AI MODU")
        print("=" * 60)

        self.mode = "ai"

        # Eğitilmiş modeli yükle
        if not self.agent.load(self.model_path):
            print("Eğitilmiş model bulunamadı!")
            print("Önce eğitim modunu çalıştırın.")
            return

        # Keşfi devre dışı bırak - sadece öğrenilen politikayı kullan
        self.agent.epsilon = 0

        for ep in range(episodes):
            self.reset()
            print(f"\n[AI Oynuyor] Episode {ep + 1}/{episodes}")

            while not self.done:
                # AI öğrendiği Q-değerlerine göre aksiyon seçer
                action = self.agent.choose_action(self.state, training=False)
                self.step(action)
                pygame.time.wait(20)  # Görselleştirme için küçük gecikme

            result = "BAŞARILI!" if self.total_reward >= -100 else "BİTTİ"
            print(f"Sonuç: {result} Skor: {self.total_reward:.0f} Adım: {self.step_count}")

        # Ortalama skoru yazdır
        avg_reward = np.mean(self.rewards_history[-episodes:])
        print(f"\nOrtalama Skor: {avg_reward:.1f}")

        self.env.close()

    def run_training_mode(self, episodes=500, save_every=50):
        """
        Eğitim modu - DQN ajanını eğit.

        Ajan şunları yaparak öğrenir:
            1. Çok sayıda episode oyna
            2. Deneyimleri replay buffer'a sakla
            3. Rastgele deneyim batch'lerinden öğren
            4. Keşifi kademeli olarak azalt (epsilon decay)

        Eğitim İpuçları:
            - Daha fazla episode = genellikle daha iyi performans
            - Son 100 episode ortalamasını izle
            - Hedef: ortalama ödül >= -100 (ortam "çözüldü")
        """
        self.mode = "train"

        # Daha hızlı eğitim için render'sız moda geç
        self.env.close()
        self.env = gym.make("Acrobot-v1")

        # Logger'ı başlat
        logger = TrainingLogger(episodes)
        logger.start_training()

        best_avg = float('-inf')

        for ep in range(1, episodes + 1):
            logger.start_episode()
            self.reset()

            # Bu episode için loss'ları topla
            episode_losses = []

            while not self.done:
                action = self.agent.choose_action(self.state, training=True)
                self.step(action)

                # Loss'ları takip et
                if self.agent.last_loss > 0:
                    episode_losses.append(self.agent.last_loss)

            # Ortalama loss hesapla
            avg_loss = np.mean(episode_losses) if episode_losses else 0

            # Kaydetmeli miyiz kontrol et
            avg_reward = np.mean(self.rewards_history[-100:]) if self.rewards_history else -500
            saved = False

            if avg_reward > best_avg and ep > 100:
                best_avg = avg_reward
                self.agent.save(self.model_path)
                saved = True

            if ep % save_every == 0:
                self.agent.save(self.model_path)
                saved = True

            # Bu episode'u logla
            logger.log_episode(
                episode=ep,
                reward=self.total_reward,
                steps=self.step_count,
                loss=avg_loss,
                epsilon=self.agent.epsilon,
                saved=saved
            )

        # Son kayıt
        self.agent.save(self.model_path)

        # Son özeti yazdır
        logger.finish_training()

        self.env.close()
        return self.rewards_history

    def demo_mode(self):
        """
        Demo modu - hızlı eğitim ardından AI gösterimi.

        Uzun eğitim olmadan hızlıca sonuç görmek için iyi.
        """
        print("\n" + "=" * 60)
        print(Colors.colorize("ACROBOT - DEMO MODU", Colors.BOLD + Colors.CYAN))
        print("=" * 60)

        # Hızlı eğitim
        print("\nHızlı eğitim başlıyor (300 episode)...")
        self.run_training_mode(episodes=300, save_every=100)

        # AI gösterimi
        print("\nAI gösterimi başlıyor...")
        self.env = gym.make("Acrobot-v1", render_mode="human")
        self.run_ai_mode(episodes=3)


# =============================================================================
# ANA MENÜ
# =============================================================================

def print_menu():
    """Ana menüyü göster."""
    print("\n" + "=" * 60)
    print(Colors.colorize("ACROBOT - ANA MENÜ", Colors.BOLD + Colors.CYAN))
    print("=" * 60)
    print("1. İnsan Modu (Klavye ile oyna)")
    print("2. AI Modu (Eğitilmiş ajanı izle)")
    print("3. Eğitim Modu (Ajanı eğit)")
    print("4. Demo Modu (Hızlı eğit + izle)")
    print("5. Çıkış")
    print("=" * 60)


def parse_args():
    """Komut satırı argümanlarını ayrıştır."""
    parser = argparse.ArgumentParser(
        description="Acrobot - Salınan Robot Kol Simülasyonu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python acrobot_dqn_game.py --watch          # Eğitilmiş AI'ı izle
  python acrobot_dqn_game.py --watch -n 10    # 10 episode izle
  python acrobot_dqn_game.py --train -n 500   # 500 episode eğit
  python acrobot_dqn_game.py --human          # Klavye ile oyna
  python acrobot_dqn_game.py                  # İnteraktif menü göster
        """
    )

    # Mod seçimi
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Eğitilmiş AI'ın oynamasını izle (kayıtlı modeli yükler)"
    )
    mode_group.add_argument(
        "--train", "-t",
        action="store_true",
        help="AI'ı sıfırdan eğit veya eğitime devam et"
    )
    mode_group.add_argument(
        "--human", "-H",
        action="store_true",
        help="Klavye ile insan modunda oyna"
    )
    mode_group.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Hızlı demo: kısa eğitim + AI gösterisi"
    )

    # Seçenekler
    parser.add_argument(
        "-n", "--episodes",
        type=int,
        default=5,
        help="Episode sayısı (varsayılan: izleme için 5, eğitim için 500)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model dosyası yolu (varsayılan: output klasöründe otomatik algıla)"
    )

    return parser.parse_args()


def main():
    """Ana giriş noktası."""
    args = parse_args()

    # Başlık
    print("\n" + "=" * 60)
    print(Colors.colorize("ACROBOT - Salınan Robot Kol Simülasyonu", Colors.BOLD + Colors.CYAN))
    print("=" * 60)
    print(f"  Çıktı     : {Colors.colorize(str(OUTPUT_DIR), Colors.DIM)}")
    print(f"  Cihaz     : {Colors.colorize(str(device), Colors.GREEN)}")

    # Model var mı kontrol et
    model_path = Path(args.model) if args.model else OUTPUT_DIR / "dqn_model.pt"
    model_exists = model_path.exists()

    if model_exists:
        print(f"  Model     : {Colors.colorize('BULUNDU', Colors.GREEN)} ({model_path.name})")
    else:
        print(f"  Model     : {Colors.colorize('BULUNAMADI', Colors.YELLOW)} (önce eğitim gerekli)")

    print()

    # Oyunu oluştur
    game = AcrobotGame()

    # Model yolu belirtilmişse üzerine yaz
    if args.model:
        game.model_path = Path(args.model)

    # =========================================================================
    # DİREKT MOD (komut satırı argümanları)
    # =========================================================================

    if args.watch:
        # İzleme modu - eğitilmiş modeli yükle ve çalıştır
        if not model_exists:
            print(Colors.colorize("HATA: Eğitilmiş model bulunamadı!", Colors.RED))
            print(f"Önce eğitin: python {Path(__file__).name} --train")
            return

        print(Colors.colorize("  Mod: İZLE (AI Oynuyor)", Colors.GREEN + Colors.BOLD))
        print("=" * 60)
        game.run_ai_mode(episodes=args.episodes)
        return

    elif args.train:
        # Eğitim modu
        episodes = args.episodes if args.episodes != 5 else 500  # Eğitim için varsayılan 500
        print(Colors.colorize(f"  Mod: EĞİTİM ({episodes} episode)", Colors.YELLOW + Colors.BOLD))
        print("=" * 60)
        game.run_training_mode(episodes=episodes)
        return

    elif args.human:
        # İnsan modu
        print(Colors.colorize("  Mod: İNSAN (Klavye Kontrolü)", Colors.CYAN + Colors.BOLD))
        print("=" * 60)
        game.run_human_mode()
        return

    elif args.demo:
        # Demo modu
        print(Colors.colorize("  Mod: DEMO (Hızlı Eğit + İzle)", Colors.HEADER + Colors.BOLD))
        print("=" * 60)
        game.demo_mode()
        return

    # =========================================================================
    # İNTERAKTİF MENÜ (argüman yok)
    # =========================================================================

    print("  Bu robot kol momentum kazanarak yukarı sallanmalı.")
    print("  DQN algoritması kullanılarak eğitilir.")

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
            print("Hoşça kalın!")
            break
        else:
            print("Geçersiz seçim!")

    game.env.close()


if __name__ == "__main__":
    main()
