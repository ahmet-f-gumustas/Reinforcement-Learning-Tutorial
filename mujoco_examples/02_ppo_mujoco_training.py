#!/usr/bin/env python3
"""
PPO ile MuJoCo Robotu Eğitme

Bu script Stable-Baselines3 kütüphanesi kullanarak PPO (Proximal Policy Optimization)
algoritması ile MuJoCo robotlarını eğitmeyi gösterir:
- PPO ajanı oluşturma ve eğitme
- Eğitim sürecini izleme (training curve)
- Modeli kaydetme ve yükleme
- Random agent vs eğitilmiş agent karşılaştırması
- Eğitilmiş ajanı görselleştirme

Gereksinimler:
    pip install stable-baselines3 gymnasium[mujoco] matplotlib

Kullanım:
    python 02_ppo_mujoco_training.py                         # Eğitim başlat
    python 02_ppo_mujoco_training.py --env Ant-v5            # Farklı ortam
    python 02_ppo_mujoco_training.py --timesteps 500000      # Daha uzun eğitim
    python 02_ppo_mujoco_training.py --eval                  # Kayıtlı modeli değerlendir
    python 02_ppo_mujoco_training.py --eval --render         # Eğitilmiş ajanı izle
    python 02_ppo_mujoco_training.py --compare               # Random vs Trained karşılaştır
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ============================================================
# Desteklenen MuJoCo Ortamları
# ============================================================

MUJOCO_ENVS = [
    "HalfCheetah-v5",    # 6 eklemli koşan robot
    "Ant-v5",            # 4 bacaklı karınca robot
    "Hopper-v5",         # Tek bacaklı zıplayan robot
    "Walker2d-v5",       # 2 bacaklı yürüyen robot
    "Humanoid-v5",       # 17 eklemli insansı robot
    "Swimmer-v5",        # Yüzen yılan robot
    "InvertedPendulum-v5",        # Ters sarkaç
    "InvertedDoublePendulum-v5",  # Çift ters sarkaç
]


# ============================================================
# Eğitim Callback'i - Reward Takibi
# ============================================================

class RewardLoggerCallback(BaseCallback):
    """
    Eğitim sırasında episode reward'larını kaydeden callback.
    Training curve çizmek için kullanılır.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_timestamps = []

    def _on_step(self) -> bool:
        # VecEnv'den episode bilgilerini al
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.episode_timestamps.append(self.num_timesteps)

                if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                    son_10 = self.episode_rewards[-10:]
                    print(
                        f"  Timestep: {self.num_timesteps:>8d} | "
                        f"Episode: {len(self.episode_rewards):>4d} | "
                        f"Son 10 Ort. Reward: {np.mean(son_10):>8.2f}"
                    )
        return True


# ============================================================
# Yardımcı Fonksiyonlar
# ============================================================

def get_save_dir(env_name):
    """Model ve grafiklerin kaydedileceği dizin"""
    save_dir = Path(__file__).parent / "trained_models" / env_name
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def create_env(env_name, seed=42):
    """Eğitim için vektörize ortam oluştur"""
    def make_env():
        env = gym.make(env_name)
        env.reset(seed=seed)
        return env
    return make_env


def plot_training_curve(callback, env_name, save_dir):
    """Eğitim sürecinin reward grafiğini çiz ve kaydet"""
    if len(callback.episode_rewards) == 0:
        print("Henüz kaydedilmiş episode yok, grafik oluşturulamadı.")
        return

    rewards = np.array(callback.episode_rewards)
    timesteps = np.array(callback.episode_timestamps)

    # Hareketli ortalama hesapla (window=20)
    window = min(20, len(rewards))
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        moving_avg_ts = timesteps[window - 1:]
    else:
        moving_avg = rewards
        moving_avg_ts = timesteps

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Reward Grafiği ---
    ax1 = axes[0]
    ax1.scatter(timesteps, rewards, alpha=0.15, s=8, color="steelblue", label="Episode Reward")
    ax1.plot(moving_avg_ts, moving_avg, color="crimson", linewidth=2, label=f"Hareketli Ort. ({window} ep)")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title(f"PPO Eğitim Eğrisi - {env_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Episode Uzunluğu Grafiği ---
    lengths = np.array(callback.episode_lengths)
    if window > 1:
        length_avg = np.convolve(lengths, np.ones(window) / window, mode="valid")
        length_avg_ts = timesteps[window - 1:]
    else:
        length_avg = lengths
        length_avg_ts = timesteps

    ax2 = axes[1]
    ax2.scatter(timesteps, lengths, alpha=0.15, s=8, color="forestgreen", label="Episode Uzunluğu")
    ax2.plot(length_avg_ts, length_avg, color="darkorange", linewidth=2, label=f"Hareketli Ort. ({window} ep)")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Episode Uzunluğu (adım)")
    ax2.set_title(f"Episode Uzunluğu - {env_name}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Kaydet
    plot_path = save_dir / "training_curve.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nEğitim grafiği kaydedildi: {plot_path}")

    plt.show()


def run_random_agent(env_name, num_episodes=10, seed=42):
    """Random agent ile performans ölç"""
    env = gym.make(env_name)
    rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    env.close()
    return rewards


def run_trained_agent(model, env_name, num_episodes=10, render=False, seed=42):
    """Eğitilmiş agent ile performans ölç"""
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        if render:
            print(f"  Episode {ep + 1}: Reward = {total_reward:.2f}")

    env.close()
    return rewards


# ============================================================
# Ana Fonksiyonlar
# ============================================================

def train(args):
    """PPO modelini eğit"""
    print("\n" + "=" * 60)
    print(f"  PPO EĞİTİMİ BAŞLIYOR")
    print(f"  Ortam: {args.env}")
    print(f"  Toplam Timestep: {args.timesteps:,}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    save_dir = get_save_dir(args.env)

    # Ortamı oluştur (Monitor wrapper ile episode bilgisi toplamak için)
    env = DummyVecEnv([create_env(args.env, seed=args.seed)])

    # PPO Hiperparametreleri
    # Bu değerler MuJoCo ortamları için iyi bir başlangıç noktasıdır
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,             # Discount factor
        gae_lambda=0.95,        # GAE lambda
        clip_range=0.2,         # PPO clip range
        ent_coef=0.0,           # Entropy katsayısı
        vf_coef=0.5,            # Value function katsayısı
        max_grad_norm=0.5,      # Gradient clipping
        verbose=0,
        seed=args.seed,
    )

    # Model mimarisini göster
    print(f"\nModel Mimarisi (MlpPolicy):")
    print(f"  Policy Network: {model.policy.net_arch}")
    print(f"  Observation Space: {model.observation_space.shape}")
    print(f"  Action Space: {model.action_space.shape}")
    print(f"  Toplam Parametre: {sum(p.numel() for p in model.policy.parameters()):,}")

    # Callback oluştur
    reward_callback = RewardLoggerCallback(verbose=1)

    # Eğitim
    print(f"\nEğitim başlıyor ({args.timesteps:,} timestep)...\n")
    model.learn(
        total_timesteps=args.timesteps,
        callback=reward_callback,
        progress_bar=True,
    )

    # Modeli kaydet
    model_path = save_dir / "ppo_model"
    model.save(str(model_path))
    print(f"\nModel kaydedildi: {model_path}.zip")

    # Eğitim grafiğini çiz
    plot_training_curve(reward_callback, args.env, save_dir)

    # Eğitim sonucu hızlı değerlendirme
    print("\n" + "-" * 40)
    print("EĞİTİM SONUCU DEĞERLENDİRME")
    print("-" * 40)

    eval_env = gym.make(args.env)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    eval_env.close()

    print(f"  Ortalama Reward: {mean_reward:.2f} (+/- {std_reward:.2f})")
    print(f"  Toplam Episode: {len(reward_callback.episode_rewards)}")

    env.close()
    print("\nEğitim tamamlandı!")
    return model


def evaluate(args):
    """Kayıtlı modeli değerlendir"""
    save_dir = get_save_dir(args.env)
    model_path = save_dir / "ppo_model.zip"

    if not model_path.exists():
        print(f"\nHata: Model bulunamadı: {model_path}")
        print(f"Önce eğitim yapın: python 02_ppo_mujoco_training.py --env {args.env}")
        return

    print("\n" + "=" * 60)
    print(f"  MODEL DEĞERLENDİRME")
    print(f"  Ortam: {args.env}")
    print(f"  Model: {model_path}")
    print(f"  Render: {'Evet' if args.render else 'Hayır'}")
    print("=" * 60)

    # Modeli yükle
    model = PPO.load(str(model_path))
    print(f"\nModel yüklendi: {model_path}")

    # Eğitilmiş ajanı çalıştır
    num_episodes = args.eval_episodes
    rewards = run_trained_agent(model, args.env, num_episodes=num_episodes, render=args.render, seed=args.seed)

    print(f"\n{'─' * 40}")
    print(f"  {num_episodes} Episode Sonuçları:")
    print(f"  Ortalama Reward: {np.mean(rewards):.2f} (+/- {np.std(rewards):.2f})")
    print(f"  Max Reward: {max(rewards):.2f}")
    print(f"  Min Reward: {min(rewards):.2f}")
    print(f"{'─' * 40}")


def compare(args):
    """Random agent vs eğitilmiş agent karşılaştırması"""
    save_dir = get_save_dir(args.env)
    model_path = save_dir / "ppo_model.zip"

    if not model_path.exists():
        print(f"\nHata: Model bulunamadı: {model_path}")
        print(f"Önce eğitim yapın: python 02_ppo_mujoco_training.py --env {args.env}")
        return

    print("\n" + "=" * 60)
    print(f"  RANDOM vs PPO KARŞILAŞTIRMASI")
    print(f"  Ortam: {args.env}")
    print("=" * 60)

    num_episodes = args.eval_episodes

    # Random agent
    print(f"\nRandom Agent ({num_episodes} episode)...")
    random_rewards = run_random_agent(args.env, num_episodes=num_episodes, seed=args.seed)

    # Eğitilmiş agent
    model = PPO.load(str(model_path))
    print(f"PPO Agent ({num_episodes} episode)...")
    trained_rewards = run_trained_agent(model, args.env, num_episodes=num_episodes, seed=args.seed)

    # Sonuçları göster
    print("\n" + "=" * 60)
    print(f"{'Metrik':<25} {'Random':>12} {'PPO':>12}")
    print("─" * 60)
    print(f"{'Ortalama Reward':<25} {np.mean(random_rewards):>12.2f} {np.mean(trained_rewards):>12.2f}")
    print(f"{'Std Reward':<25} {np.std(random_rewards):>12.2f} {np.std(trained_rewards):>12.2f}")
    print(f"{'Max Reward':<25} {max(random_rewards):>12.2f} {max(trained_rewards):>12.2f}")
    print(f"{'Min Reward':<25} {min(random_rewards):>12.2f} {min(trained_rewards):>12.2f}")
    print("=" * 60)

    # İyileşme oranı
    random_mean = np.mean(random_rewards)
    trained_mean = np.mean(trained_rewards)
    if random_mean != 0:
        improvement = ((trained_mean - random_mean) / abs(random_mean)) * 100
        print(f"\nİyileşme: {improvement:+.1f}%")

    # Karşılaştırma grafiği
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(num_episodes)
    width = 0.35

    bars1 = ax.bar(x - width / 2, random_rewards, width, label="Random Agent", color="lightcoral", alpha=0.8)
    bars2 = ax.bar(x + width / 2, trained_rewards, width, label="PPO Agent", color="steelblue", alpha=0.8)

    ax.axhline(y=np.mean(random_rewards), color="red", linestyle="--", alpha=0.5, label=f"Random Ort. ({np.mean(random_rewards):.0f})")
    ax.axhline(y=np.mean(trained_rewards), color="blue", linestyle="--", alpha=0.5, label=f"PPO Ort. ({np.mean(trained_rewards):.0f})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Toplam Reward")
    ax.set_title(f"Random vs PPO - {args.env}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    plot_path = save_dir / "comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Karşılaştırma grafiği kaydedildi: {plot_path}")

    plt.show()


# ============================================================
# Giriş Noktası
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PPO ile MuJoCo Robotu Eğitme",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python 02_ppo_mujoco_training.py                         # HalfCheetah eğit
  python 02_ppo_mujoco_training.py --env Hopper-v5         # Hopper eğit
  python 02_ppo_mujoco_training.py --timesteps 500000      # Daha uzun eğitim
  python 02_ppo_mujoco_training.py --eval --render         # Eğitilmiş ajanı izle
  python 02_ppo_mujoco_training.py --compare               # Random vs PPO karşılaştır
        """
    )

    # Ortam seçenekleri
    parser.add_argument("--env", type=str, default="HalfCheetah-v5",
                        choices=MUJOCO_ENVS,
                        help="Kullanılacak MuJoCo ortamı (varsayılan: HalfCheetah-v5)")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Toplam eğitim timestep sayısı (varsayılan: 100000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (varsayılan: 42)")

    # Modlar
    parser.add_argument("--eval", action="store_true",
                        help="Kayıtlı modeli değerlendir (eğitim yapma)")
    parser.add_argument("--compare", action="store_true",
                        help="Random vs eğitilmiş agent karşılaştırması")
    parser.add_argument("--render", action="store_true",
                        help="Ajanı görselleştir (--eval ile birlikte)")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Değerlendirme episode sayısı (varsayılan: 10)")

    args = parser.parse_args()

    # Mod seçimi
    if args.eval:
        evaluate(args)
    elif args.compare:
        compare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
