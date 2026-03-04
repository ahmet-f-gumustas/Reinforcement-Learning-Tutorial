"""
Gorsellestirme Araclari / Visualization Utilities

Egitim egrileri, karsilastirma grafikleri, radar chart ve
gait analizi icin gorsellestirme fonksiyonlari.

Visualization functions for training curves, comparison charts,
radar charts, and gait analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path


SAVE_DIR = Path(__file__).parent.parent / "plots"


def _ensure_save_dir():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Egitim Egrileri / Training Curves
# =============================================================================

def plot_training_curves(
    histories: Dict[str, "TrainingHistory"],
    window: int = 50,
    save: bool = True,
    show: bool = True,
):
    """
    Birden fazla yaklasiim icin egitim egrilerini ciz.
    Plot training curves for multiple approaches.

    Args:
        histories: {approach_name: TrainingHistory} dict'i
        window: Hareketli ortalama pencere boyutu
        save: Dosyaya kaydet
        show: Goster
    """
    _ensure_save_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "Single PPO": "steelblue",
        "MAPPO": "coral",
        "Hierarchical": "forestgreen",
        "Multi-Robot": "darkorchid",
    }

    # Sol: Reward egrileri / Left: Reward curves
    ax = axes[0]
    for name, history in histories.items():
        rewards = history.episode_rewards
        if len(rewards) < window:
            continue
        smoothed = np.convolve(
            rewards, np.ones(window) / window, mode="valid"
        )
        color = colors.get(name, None)
        ax.plot(smoothed, label=name, color=color, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Egitim Egrileri / Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sag: Episode uzunlugu / Right: Episode lengths
    ax = axes[1]
    for name, history in histories.items():
        lengths = history.episode_lengths
        if len(lengths) < window:
            continue
        smoothed = np.convolve(
            lengths, np.ones(window) / window, mode="valid"
        )
        color = colors.get(name, None)
        ax.plot(smoothed, label=name, color=color, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Uzunlugu / Length")
    ax.set_title("Episode Uzunluklari / Episode Lengths")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Multi-Agent Quadruped: Egitim Karsilastirmasi",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if save:
        path = SAVE_DIR / "training_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {path}")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# Bar Chart Karsilastirma / Bar Chart Comparison
# =============================================================================

def plot_comparison_bar(
    results: Dict[str, Dict[str, float]],
    save: bool = True,
    show: bool = True,
):
    """
    Final performans karsilastirmasi bar chart.
    Final performance comparison bar chart.

    Args:
        results: {approach_name: {"mean": float, "std": float}}
    """
    _ensure_save_dir()
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds = [results[n]["std"] for n in names]

    colors = ["steelblue", "coral", "forestgreen", "darkorchid"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        names, means, yerr=stds, capsize=10,
        color=colors[:len(names)], alpha=0.85, edgecolor="black",
    )

    ax.set_ylabel("Ortalama Episode Reward")
    ax.set_title("Yaklasim Karsilastirmasi / Approach Comparison")
    ax.grid(True, axis="y", alpha=0.3)

    # Degerleri barlarin ustune yaz
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 1,
            f"{mean:.1f}", ha="center", va="bottom", fontweight="bold",
        )

    plt.tight_layout()
    if save:
        path = SAVE_DIR / "comparison_bar.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {path}")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# Radar Chart
# =============================================================================

def plot_radar_chart(
    metrics: Dict[str, Dict[str, float]],
    save: bool = True,
    show: bool = True,
):
    """
    Cok-metrik radar grafigi.
    Multi-metric radar chart.

    Args:
        metrics: {approach_name: {"speed": f, "stability": f, "energy_eff": f, "reward": f}}
    """
    _ensure_save_dir()
    categories = ["Hiz/Speed", "Stabilite/Stability", "Enerji Verim.", "Reward"]
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Kapali cokgen

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ["steelblue", "coral", "forestgreen", "darkorchid"]

    for idx, (name, vals) in enumerate(metrics.items()):
        values = [vals.get(k, 0) for k in ["speed", "stability", "energy_eff", "reward"]]
        values += values[:1]
        color = colors[idx % len(colors)]
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title(
        "Cok-Metrik Karsilastirma / Multi-Metric Comparison",
        size=14, fontweight="bold", y=1.1,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    if save:
        path = SAVE_DIR / "radar_chart.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {path}")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# Gait Pattern Analizi / Gait Pattern Analysis
# =============================================================================

def plot_gait_pattern(
    foot_contacts: np.ndarray,
    title: str = "Gait Pattern",
    save: bool = True,
    show: bool = True,
):
    """
    Ayak temas kaliplari zaman serisi.
    Foot contact patterns time series.

    Args:
        foot_contacts: (T, 4) boyutunda ayak temas verisi
                       sutunlar: [FL, FR, BL, BR]
    """
    _ensure_save_dir()
    T = len(foot_contacts)
    leg_names = ["FL (On Sol)", "FR (On Sag)", "BL (Arka Sol)", "BR (Arka Sag)"]
    colors = ["steelblue", "coral", "forestgreen", "darkorchid"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 6), sharex=True)

    for i in range(4):
        ax = axes[i]
        contacts = foot_contacts[:, i]
        ax.fill_between(
            range(T), contacts, alpha=0.7, color=colors[i], step="mid"
        )
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel(leg_names[i], fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Havada", "Yerde"], fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)

    axes[-1].set_xlabel("Zaman Adimi / Timestep")
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = SAVE_DIR / "gait_pattern.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {path}")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# Ozet Tablosu / Summary Table
# =============================================================================

def print_summary_table(results: Dict[str, Dict[str, float]]):
    """
    Konsola formatli karsilastirma tablosu yazdir.
    Print formatted comparison table to console.

    Args:
        results: {approach_name: {"mean_reward": f, "mean_velocity": f,
                                   "energy_eff": f, "stability": f}}
    """
    print("\n" + "=" * 75)
    print(f"{'Yaklasim':<20} {'Ort. Reward':>12} {'Ort. Hiz':>12} "
          f"{'Enerji Vr.':>12} {'Stabilite':>12}")
    print("-" * 75)

    for name, vals in results.items():
        print(
            f"{name:<20} "
            f"{vals.get('mean_reward', 0):>12.2f} "
            f"{vals.get('mean_velocity', 0):>12.4f} "
            f"{vals.get('energy_eff', 0):>12.4f} "
            f"{vals.get('stability', 0):>12.4f}"
        )

    print("=" * 75)
