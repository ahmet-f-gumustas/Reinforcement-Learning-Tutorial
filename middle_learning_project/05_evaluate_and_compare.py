#!/usr/bin/env python3
"""
Model Degerlendirme ve Karsilastirma / Model Evaluation and Comparison

Bu script, Ant-v5 SAC baseline ve Custom Quadruped SAC modellerini
yukleyerek performanslarini karsilastirir. Random baseline'lar ile
birlikte kapsamli bir degerlendirme yapar.

This script loads both the Ant-v5 SAC baseline and Custom Quadruped SAC
models and compares their performance. Performs comprehensive evaluation
including random baselines.

Metrikler / Metrics:
    - Ortalama Reward / Mean Reward
    - Ortalama Hiz / Mean Velocity
    - Episode Uzunlugu / Episode Length
    - Enerji Verimliligi / Energy Efficiency

Kullanim / Usage:
    python 05_evaluate_and_compare.py                  # Karsilastirma
    python 05_evaluate_and_compare.py --episodes 20    # 20 episode ile
    python 05_evaluate_and_compare.py --render          # Gorsel izleme
"""

import argparse
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import SAC


# ============================================================
# Custom Environment Import
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
    return CustomQuadrupedEnv(xml_file=xml_path, render_mode=render_mode)


# ============================================================
# Degerlendirme Fonksiyonlari / Evaluation Functions
# ============================================================

def evaluate_model(model, env, num_episodes=10, seed=42):
    """
    Modeli degerlendir ve detayli metrikler topla.
    Evaluate model and collect detailed metrics.

    Returns:
        dict: rewards, velocities, lengths, ctrl_costs
    """
    results = {
        "rewards": [],
        "velocities": [],
        "lengths": [],
        "ctrl_costs": [],
    }

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0
        total_velocity = 0
        total_ctrl_cost = 0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_velocity += info.get("x_velocity", 0)
            total_ctrl_cost += abs(info.get("reward_ctrl", 0))
            steps += 1
            done = terminated or truncated

        results["rewards"].append(total_reward)
        results["velocities"].append(total_velocity / max(steps, 1))
        results["lengths"].append(steps)
        results["ctrl_costs"].append(total_ctrl_cost / max(steps, 1))

    return results


def evaluate_random(env, num_episodes=10, seed=42):
    """
    Random agent ile degerlendir.
    Evaluate with random agent.
    """
    results = {
        "rewards": [],
        "velocities": [],
        "lengths": [],
        "ctrl_costs": [],
    }

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0
        total_velocity = 0
        total_ctrl_cost = 0
        steps = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_velocity += info.get("x_velocity", 0)
            total_ctrl_cost += abs(info.get("reward_ctrl", 0))
            steps += 1
            done = terminated or truncated

        results["rewards"].append(total_reward)
        results["velocities"].append(total_velocity / max(steps, 1))
        results["lengths"].append(steps)
        results["ctrl_costs"].append(total_ctrl_cost / max(steps, 1))

    return results


def render_model(model, env, num_episodes=3, seed=42):
    """Modeli gorsel olarak calistir / Run model with visualization"""
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"  Episode {ep + 1}: Reward = {total_reward:.2f}")

    env.close()


# ============================================================
# Tablo ve Grafik / Table and Charts
# ============================================================

def print_comparison_table(ant_sac, quad_sac, ant_random, quad_random):
    """Karsilastirma tablosunu yazdir / Print comparison table"""
    print("\n" + "=" * 80)
    print("  SAC PERFORMANS KARSILASTIRMASI / SAC PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Metrik / Metric':<25} {'Ant-v5 SAC':>14} {'Custom SAC':>14} {'Ant Random':>14} {'Custom Rand':>14}")
    print("─" * 80)

    metrics = [
        ("Ort. Reward", "rewards", np.mean),
        ("Std Reward", "rewards", np.std),
        ("Ort. Hiz / Velocity", "velocities", np.mean),
        ("Ort. Episode Uzun.", "lengths", np.mean),
        ("Ort. Ctrl Cost", "ctrl_costs", np.mean),
    ]

    for name, key, func in metrics:
        print(
            f"{name:<25} "
            f"{func(ant_sac[key]):>14.2f} "
            f"{func(quad_sac[key]):>14.2f} "
            f"{func(ant_random[key]):>14.2f} "
            f"{func(quad_random[key]):>14.2f}"
        )

    # Enerji verimliligi / Energy efficiency
    ant_eff = np.mean(ant_sac["rewards"]) / max(np.mean(ant_sac["ctrl_costs"]), 0.01)
    quad_eff = np.mean(quad_sac["rewards"]) / max(np.mean(quad_sac["ctrl_costs"]), 0.01)
    print(f"{'Enerji Verimliligi':<25} {ant_eff:>14.2f} {quad_eff:>14.2f} {'N/A':>14} {'N/A':>14}")

    print("=" * 80)

    # Iyilesme oranlari / Improvement rates
    ant_imp = np.mean(ant_sac["rewards"]) - np.mean(ant_random["rewards"])
    quad_imp = np.mean(quad_sac["rewards"]) - np.mean(quad_random["rewards"])
    print(f"\nSAC Iyilesme (Random'a gore):")
    print(f"  Ant-v5:          {ant_imp:+.2f} reward")
    print(f"  Custom Quadruped: {quad_imp:+.2f} reward")


def plot_comparison(ant_sac, quad_sac, ant_random, quad_random, save_dir):
    """Karsilastirma grafiklerini ciz / Plot comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- 1. Ortalama Reward Bar Chart ---
    ax1 = axes[0, 0]
    labels = ["Ant-v5\nSAC", "Custom\nSAC", "Ant-v5\nRandom", "Custom\nRandom"]
    means = [
        np.mean(ant_sac["rewards"]),
        np.mean(quad_sac["rewards"]),
        np.mean(ant_random["rewards"]),
        np.mean(quad_random["rewards"]),
    ]
    stds = [
        np.std(ant_sac["rewards"]),
        np.std(quad_sac["rewards"]),
        np.std(ant_random["rewards"]),
        np.std(quad_random["rewards"]),
    ]
    colors = ["steelblue", "darkorange", "lightcoral", "lightsalmon"]

    bars = ax1.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax1.set_ylabel("Ortalama Reward")
    ax1.set_title("Reward Karsilastirmasi / Reward Comparison")
    ax1.grid(True, axis="y", alpha=0.3)

    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{mean:.0f}", ha="center", va="bottom", fontsize=9)

    # --- 2. Reward Box Plot ---
    ax2 = axes[0, 1]
    data = [ant_sac["rewards"], quad_sac["rewards"],
            ant_random["rewards"], quad_random["rewards"]]
    bp = ax2.boxplot(data, labels=["Ant SAC", "Custom SAC", "Ant Rand", "Custom Rand"],
                     patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("Episode Reward")
    ax2.set_title("Reward Dagilimi / Reward Distribution")
    ax2.grid(True, axis="y", alpha=0.3)

    # --- 3. Ortalama Hiz / Mean Velocity ---
    ax3 = axes[1, 0]
    vel_labels = ["Ant-v5 SAC", "Custom SAC", "Ant-v5 Random", "Custom Random"]
    vel_means = [
        np.mean(ant_sac["velocities"]),
        np.mean(quad_sac["velocities"]),
        np.mean(ant_random["velocities"]),
        np.mean(quad_random["velocities"]),
    ]
    ax3.bar(vel_labels, vel_means, color=colors, alpha=0.8)
    ax3.set_ylabel("Ortalama X Hizi (m/s)")
    ax3.set_title("Ileri Hiz Karsilastirmasi / Forward Velocity Comparison")
    ax3.grid(True, axis="y", alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha="right")

    # --- 4. Episode Uzunlugu / Episode Length ---
    ax4 = axes[1, 1]
    len_means = [
        np.mean(ant_sac["lengths"]),
        np.mean(quad_sac["lengths"]),
        np.mean(ant_random["lengths"]),
        np.mean(quad_random["lengths"]),
    ]
    ax4.bar(vel_labels, len_means, color=colors, alpha=0.8)
    ax4.set_ylabel("Ortalama Episode Uzunlugu (adim)")
    ax4.set_title("Hayatta Kalma Suresi / Survival Time")
    ax4.grid(True, axis="y", alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha="right")

    plt.suptitle("Ant-v5 vs Custom Quadruped - SAC Performans Karsilastirmasi",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    plot_path = save_dir / "comparison_chart.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nKarsilastirma grafigi kaydedildi: {plot_path}")
    plt.show()


# ============================================================
# Ana Fonksiyon / Main Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ant-v5 vs Custom Quadruped SAC Karsilastirmasi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler / Examples:
  python 05_evaluate_and_compare.py                  # Karsilastirma
  python 05_evaluate_and_compare.py --episodes 20    # 20 episode ile
  python 05_evaluate_and_compare.py --render          # Gorsel izleme
        """
    )

    parser.add_argument("--episodes", type=int, default=10,
                        help="Degerlendirme episode sayisi (varsayilan: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (varsayilan: 42)")
    parser.add_argument("--render", action="store_true",
                        help="Egitilmis ajanlari gorsel olarak izle")

    args = parser.parse_args()

    base_dir = Path(__file__).parent / "trained_models"
    ant_model_path = base_dir / "ant_v5" / "sac_model.zip"
    quad_model_path = base_dir / "custom_quadruped" / "sac_model.zip"

    # Model varligini kontrol et / Check if models exist
    models_found = True
    if not ant_model_path.exists():
        print(f"Uyari: Ant-v5 modeli bulunamadi: {ant_model_path}")
        print("  Egitim icin: python 01_ant_v5_baseline.py")
        models_found = False

    if not quad_model_path.exists():
        print(f"Uyari: Custom Quadruped modeli bulunamadi: {quad_model_path}")
        print("  Egitim icin: python 04_train_custom_quadruped.py")
        models_found = False

    if not models_found:
        print("\nHer iki model de egitilmis olmali. Eksik modelleri egitip tekrar deneyin.")
        return

    print("\n" + "=" * 60)
    print("  MODEL DEGERLENDIRME VE KARSILASTIRMA")
    print("  MODEL EVALUATION AND COMPARISON")
    print("=" * 60)

    # Modelleri yukle / Load models
    print("\nModeller yukleniyor / Loading models...")
    ant_model = SAC.load(str(ant_model_path))
    quad_model = SAC.load(str(quad_model_path))
    print("  Ant-v5 SAC modeli yuklendi.")
    print("  Custom Quadruped SAC modeli yuklendi.")

    # Render modu / Render mode
    if args.render:
        print("\n--- Ant-v5 SAC Izleme ---")
        ant_render_env = gym.make("Ant-v5", render_mode="human")
        render_model(ant_model, ant_render_env, num_episodes=3, seed=args.seed)

        print("\n--- Custom Quadruped SAC Izleme ---")
        quad_render_env = make_custom_env(render_mode="human")
        render_model(quad_model, quad_render_env, num_episodes=3, seed=args.seed)

    # Degerlendirme / Evaluation
    num_eps = args.episodes

    print(f"\nDegerlendirme basliyor ({num_eps} episode)...")

    print("  Ant-v5 SAC degerlendiriliyor...")
    ant_env = gym.make("Ant-v5")
    ant_sac = evaluate_model(ant_model, ant_env, num_episodes=num_eps, seed=args.seed)
    ant_env.close()

    print("  Custom Quadruped SAC degerlendiriliyor...")
    quad_env = make_custom_env()
    quad_sac = evaluate_model(quad_model, quad_env, num_episodes=num_eps, seed=args.seed)
    quad_env.close()

    print("  Ant-v5 Random degerlendiriliyor...")
    ant_rand_env = gym.make("Ant-v5")
    ant_random = evaluate_random(ant_rand_env, num_episodes=num_eps, seed=args.seed)
    ant_rand_env.close()

    print("  Custom Quadruped Random degerlendiriliyor...")
    quad_rand_env = make_custom_env()
    quad_random = evaluate_random(quad_rand_env, num_episodes=num_eps, seed=args.seed)
    quad_rand_env.close()

    # Sonuclari goster / Show results
    print_comparison_table(ant_sac, quad_sac, ant_random, quad_random)

    # Grafikleri ciz / Plot charts
    save_dir = base_dir / "comparison"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(ant_sac, quad_sac, ant_random, quad_random, save_dir)

    print("\nDegerlendirme tamamlandi! / Evaluation complete!")


if __name__ == "__main__":
    main()
