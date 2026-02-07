#!/usr/bin/env python3
"""
SAC ile Ozel Dort Bacakli Robot Egitimi / SAC Custom Quadruped Training

Bu script, 02_custom_quadruped_model.xml'de tanimlanan ozel dort bacakli
robotu SAC (Soft Actor-Critic) algoritmasi ile egitir. 03_custom_quadruped_env.py
dosyasindaki CustomQuadrupedEnv ortamini kullanir.

This script trains the custom quadruped robot defined in
02_custom_quadruped_model.xml using the SAC algorithm. Uses the
CustomQuadrupedEnv environment from 03_custom_quadruped_env.py.

Kullanim / Usage:
    python 04_train_custom_quadruped.py                         # Egitim baslat (300k steps)
    python 04_train_custom_quadruped.py --timesteps 500000      # Daha uzun egitim
    python 04_train_custom_quadruped.py --eval                  # Egitilmis ajani degerlendir
    python 04_train_custom_quadruped.py --eval --render         # Egitilmis ajani izle
    python 04_train_custom_quadruped.py --compare               # Random vs SAC karsilastir
"""

import argparse
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


# ============================================================
# Custom Environment Import
# Dosya adi rakamla basladigi icin importlib kullaniyoruz
# Using importlib because filename starts with a digit
# ============================================================

def _load_custom_env_class():
    """CustomQuadrupedEnv sinifini yukle / Load CustomQuadrupedEnv class"""
    env_file = Path(__file__).parent / "03_custom_quadruped_env.py"
    spec = importlib.util.spec_from_file_location("custom_quadruped_env", str(env_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CustomQuadrupedEnv


CustomQuadrupedEnv = _load_custom_env_class()


def make_custom_env(render_mode=None):
    """Ozel quadruped ortamini olustur / Create custom quadruped environment"""
    xml_path = str(Path(__file__).parent / "02_custom_quadruped_model.xml")
    env = CustomQuadrupedEnv(
        xml_file=xml_path,
        render_mode=render_mode,
    )
    return env


# ============================================================
# Egitim Callback'i - Reward Takibi
# ============================================================

class RewardLoggerCallback(BaseCallback):
    """
    Egitim sirasinda episode reward'larini kaydeden callback.
    Training curve cizmek icin kullanilir.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_timestamps = []

    def _on_step(self) -> bool:
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
# Yardimci Fonksiyonlar / Helper Functions
# ============================================================

def get_save_dir():
    """Model ve grafiklerin kaydedilecegi dizin / Save directory"""
    save_dir = Path(__file__).parent / "trained_models" / "custom_quadruped"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def plot_training_curve(callback, save_dir):
    """Egitim surecinin reward grafiğini ciz ve kaydet / Plot training curve"""
    if len(callback.episode_rewards) == 0:
        print("Henuz kaydedilmis episode yok, grafik olusturulamadi.")
        return

    rewards = np.array(callback.episode_rewards)
    timesteps = np.array(callback.episode_timestamps)

    window = min(20, len(rewards))
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        moving_avg_ts = timesteps[window - 1:]
    else:
        moving_avg = rewards
        moving_avg_ts = timesteps

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Reward Grafigi ---
    ax1 = axes[0]
    ax1.scatter(timesteps, rewards, alpha=0.15, s=8, color="steelblue",
                label="Episode Reward")
    ax1.plot(moving_avg_ts, moving_avg, color="crimson", linewidth=2,
             label=f"Hareketli Ort. ({window} ep)")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("SAC Egitim Egrisi - Custom Quadruped")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Episode Uzunlugu Grafigi ---
    lengths = np.array(callback.episode_lengths)
    if window > 1:
        length_avg = np.convolve(lengths, np.ones(window) / window, mode="valid")
        length_avg_ts = timesteps[window - 1:]
    else:
        length_avg = lengths
        length_avg_ts = timesteps

    ax2 = axes[1]
    ax2.scatter(timesteps, lengths, alpha=0.15, s=8, color="forestgreen",
                label="Episode Uzunlugu")
    ax2.plot(length_avg_ts, length_avg, color="darkorange", linewidth=2,
             label=f"Hareketli Ort. ({window} ep)")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Episode Uzunlugu (adim)")
    ax2.set_title("Episode Uzunlugu - Custom Quadruped")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = save_dir / "training_curve.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nEgitim grafigi kaydedildi: {plot_path}")
    plt.show()


def run_random_agent(num_episodes=10, seed=42):
    """Random agent ile performans olc / Measure random agent performance"""
    env = make_custom_env()
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


def run_trained_agent(model, num_episodes=10, render=False, seed=42):
    """Egitilmis agent ile performans olc / Measure trained agent performance"""
    render_mode = "human" if render else None
    env = make_custom_env(render_mode=render_mode)
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
# Ana Fonksiyonlar / Main Functions
# ============================================================

def train(args):
    """SAC modelini egit / Train SAC model"""
    print("\n" + "=" * 60)
    print(f"  SAC EGITIMI BASLIYOR / SAC TRAINING STARTING")
    print(f"  Ortam / Environment: Custom Quadruped")
    print(f"  Toplam Timestep: {args.timesteps:,}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    save_dir = get_save_dir()

    # Ortami olustur / Create environment
    env = make_custom_env()

    # Ortam bilgilerini goster / Show environment info
    print(f"\n  Observation Space: {env.observation_space.shape}")
    print(f"  Action Space: {env.action_space.shape}")
    print(f"  Action Range: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")

    # SAC Hiperparametreleri / SAC Hyperparameters
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,          # Ogrenme orani
        buffer_size=1_000_000,       # Replay buffer boyutu
        learning_starts=10_000,      # Ilk random exploration adim sayisi
        batch_size=256,              # Mini-batch boyutu
        tau=0.005,                   # Soft update katsayisi
        gamma=0.99,                  # Discount factor
        train_freq=1,                # Her adimda egitim
        gradient_steps=1,            # Her adimda 1 gradient step
        ent_coef="auto",             # Otomatik entropy katsayisi
        target_entropy="auto",       # Otomatik hedef entropy
        verbose=0,
        seed=args.seed,
        tensorboard_log=str(save_dir / "tb_logs"),
    )

    print(f"\nModel Mimarisi (MlpPolicy):")
    print(f"  Toplam Parametre: {sum(p.numel() for p in model.policy.parameters()):,}")

    # Callback olustur / Create callback
    reward_callback = RewardLoggerCallback(verbose=1)

    # Egitim / Training
    print(f"\nEgitim basliyor ({args.timesteps:,} timestep)...\n")
    model.learn(
        total_timesteps=args.timesteps,
        callback=reward_callback,
        progress_bar=True,
    )

    # Modeli kaydet / Save model
    model_path = save_dir / "sac_model"
    model.save(str(model_path))
    print(f"\nModel kaydedildi: {model_path}.zip")

    # Egitim grafigini ciz / Plot training curve
    plot_training_curve(reward_callback, save_dir)

    # Egitim sonucu hizli degerlendirme / Quick evaluation
    print("\n" + "-" * 40)
    print("EGITIM SONUCU DEGERLENDIRME / POST-TRAINING EVALUATION")
    print("-" * 40)

    eval_env = make_custom_env()
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    eval_env.close()

    print(f"  Ortalama Reward: {mean_reward:.2f} (+/- {std_reward:.2f})")
    print(f"  Toplam Episode: {len(reward_callback.episode_rewards)}")

    env.close()
    print("\nEgitim tamamlandi! / Training complete!")
    return model


def evaluate(args):
    """Kayitli modeli degerlendir / Evaluate saved model"""
    save_dir = get_save_dir()
    model_path = save_dir / "sac_model.zip"

    if not model_path.exists():
        print(f"\nHata: Model bulunamadi: {model_path}")
        print("Once egitim yapin: python 04_train_custom_quadruped.py")
        return

    print("\n" + "=" * 60)
    print(f"  MODEL DEGERLENDIRME / MODEL EVALUATION")
    print(f"  Ortam / Environment: Custom Quadruped")
    print(f"  Model: {model_path}")
    print(f"  Render: {'Evet' if args.render else 'Hayir'}")
    print("=" * 60)

    model = SAC.load(str(model_path))
    print(f"\nModel yuklendi: {model_path}")

    num_episodes = args.eval_episodes
    rewards = run_trained_agent(
        model, num_episodes=num_episodes, render=args.render, seed=args.seed
    )

    print(f"\n{'─' * 40}")
    print(f"  {num_episodes} Episode Sonuclari / Results:")
    print(f"  Ortalama Reward: {np.mean(rewards):.2f} (+/- {np.std(rewards):.2f})")
    print(f"  Max Reward: {max(rewards):.2f}")
    print(f"  Min Reward: {min(rewards):.2f}")
    print(f"{'─' * 40}")


def compare(args):
    """Random agent vs SAC karsilastirmasi / Random vs SAC comparison"""
    save_dir = get_save_dir()
    model_path = save_dir / "sac_model.zip"

    if not model_path.exists():
        print(f"\nHata: Model bulunamadi: {model_path}")
        print("Once egitim yapin: python 04_train_custom_quadruped.py")
        return

    print("\n" + "=" * 60)
    print(f"  RANDOM vs SAC KARSILASTIRMASI / RANDOM vs SAC COMPARISON")
    print(f"  Ortam / Environment: Custom Quadruped")
    print("=" * 60)

    num_episodes = args.eval_episodes

    print(f"\nRandom Agent ({num_episodes} episode)...")
    random_rewards = run_random_agent(num_episodes=num_episodes, seed=args.seed)

    model = SAC.load(str(model_path))
    print(f"SAC Agent ({num_episodes} episode)...")
    trained_rewards = run_trained_agent(
        model, num_episodes=num_episodes, seed=args.seed
    )

    # Sonuclari goster / Show results
    print("\n" + "=" * 60)
    print(f"{'Metrik / Metric':<25} {'Random':>12} {'SAC':>12}")
    print("─" * 60)
    print(f"{'Ortalama Reward':<25} {np.mean(random_rewards):>12.2f} {np.mean(trained_rewards):>12.2f}")
    print(f"{'Std Reward':<25} {np.std(random_rewards):>12.2f} {np.std(trained_rewards):>12.2f}")
    print(f"{'Max Reward':<25} {max(random_rewards):>12.2f} {max(trained_rewards):>12.2f}")
    print(f"{'Min Reward':<25} {min(random_rewards):>12.2f} {min(trained_rewards):>12.2f}")
    print("=" * 60)

    random_mean = np.mean(random_rewards)
    trained_mean = np.mean(trained_rewards)
    if random_mean != 0:
        improvement = ((trained_mean - random_mean) / abs(random_mean)) * 100
        print(f"\nIyilesme / Improvement: {improvement:+.1f}%")

    # Karsilastirma grafigi / Comparison chart
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(num_episodes)
    width = 0.35

    ax.bar(x - width / 2, random_rewards, width,
           label="Random Agent", color="lightcoral", alpha=0.8)
    ax.bar(x + width / 2, trained_rewards, width,
           label="SAC Agent", color="steelblue", alpha=0.8)

    ax.axhline(y=np.mean(random_rewards), color="red", linestyle="--", alpha=0.5,
               label=f"Random Ort. ({np.mean(random_rewards):.0f})")
    ax.axhline(y=np.mean(trained_rewards), color="blue", linestyle="--", alpha=0.5,
               label=f"SAC Ort. ({np.mean(trained_rewards):.0f})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Toplam Reward")
    ax.set_title("Random vs SAC - Custom Quadruped")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    plot_path = save_dir / "comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Karsilastirma grafigi kaydedildi: {plot_path}")
    plt.show()


# ============================================================
# Giris Noktasi / Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAC ile Ozel Dort Bacakli Robot Egitimi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler / Examples:
  python 04_train_custom_quadruped.py                         # Egitim baslat
  python 04_train_custom_quadruped.py --timesteps 500000      # Daha uzun egitim
  python 04_train_custom_quadruped.py --eval --render         # Ajani izle
  python 04_train_custom_quadruped.py --compare               # Random vs SAC karsilastir
        """
    )

    parser.add_argument("--timesteps", type=int, default=300_000,
                        help="Toplam egitim timestep sayisi (varsayilan: 300000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (varsayilan: 42)")
    parser.add_argument("--eval", action="store_true",
                        help="Kayitli modeli degerlendir (egitim yapma)")
    parser.add_argument("--compare", action="store_true",
                        help="Random vs SAC karsilastirmasi")
    parser.add_argument("--render", action="store_true",
                        help="Ajani gorsellestir (--eval ile birlikte)")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Degerlendirme episode sayisi (varsayilan: 10)")

    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    elif args.compare:
        compare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
