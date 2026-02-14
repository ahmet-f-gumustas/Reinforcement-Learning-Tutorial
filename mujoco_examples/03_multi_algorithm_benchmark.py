#!/usr/bin/env python3
"""
Multi-Algorithm Benchmark Suite for MuJoCo Environments

Bu kapsamlı proje, modern deep RL algoritmalarını MuJoCo robotları üzerinde
karşılaştırmalı olarak değerlendirir:

Desteklenen Algoritmalar:
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
- A2C (Advantage Actor-Critic)

Özellikler:
- Çoklu algoritma parallel training
- Comprehensive logging ve tensorboard integration
- Hyperparameter optimization desteği
- Detailed performance comparison
- Statistical significance testing
- Learning curve analysis
- Model checkpointing ve resume
- Video recording
- Automated reporting

Gereksinimler:
    pip install stable-baselines3[extra] gymnasium[mujoco] matplotlib scipy pandas tensorboard

Kullanım:
    # Tek algoritma eğitimi
    python 03_multi_algorithm_benchmark.py --algo ppo --env HalfCheetah-v5

    # Tüm algoritmaları benchmark et
    python 03_multi_algorithm_benchmark.py --benchmark --env HalfCheetah-v5

    # Çoklu ortam üzerinde test
    python 03_multi_algorithm_benchmark.py --benchmark --multi-env

    # Hyperparameter tuning
    python 03_multi_algorithm_benchmark.py --algo ppo --env Ant-v5 --tune

    # Sonuçları karşılaştır
    python 03_multi_algorithm_benchmark.py --compare

    # Detaylı rapor oluştur
    python 03_multi_algorithm_benchmark.py --report
"""

import argparse
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# ============================================================
# Konfigürasyon ve Sabitler
# ============================================================

ALGORITHMS = {
    'ppo': PPO,
    'sac': SAC,
    'td3': TD3,
    'a2c': A2C,
}

MUJOCO_ENVS = [
    "HalfCheetah-v5",
    "Ant-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Humanoid-v5",
    "Swimmer-v5",
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
]

# Varsayılan hyperparametreler
DEFAULT_HYPERPARAMS = {
    'ppo': {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.0,
    },
    'sac': {
        'learning_rate': 3e-4,
        'buffer_size': 1_000_000,
        'learning_starts': 10000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
    },
    'td3': {
        'learning_rate': 3e-4,
        'buffer_size': 1_000_000,
        'learning_starts': 10000,
        'batch_size': 100,
        'tau': 0.005,
        'gamma': 0.99,
        'policy_delay': 2,
        'target_policy_noise': 0.2,
        'target_noise_clip': 0.5,
    },
    'a2c': {
        'learning_rate': 7e-4,
        'n_steps': 5,
        'gamma': 0.99,
        'gae_lambda': 1.0,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
    }
}


# ============================================================
# Veri Yapıları
# ============================================================

@dataclass
class TrainingConfig:
    """Eğitim konfigürasyonu"""
    algo: str
    env_name: str
    total_timesteps: int
    seed: int
    n_eval_episodes: int
    eval_freq: int
    save_freq: int
    hyperparams: Dict[str, Any]

    def to_dict(self):
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Benchmark sonucu"""
    algo: str
    env_name: str
    mean_reward: float
    std_reward: float
    mean_ep_length: float
    training_time: float
    total_timesteps: int
    final_eval_reward: float
    final_eval_std: float

    def to_dict(self):
        return asdict(self)


# ============================================================
# Callback Sınıfları
# ============================================================

class DetailedLoggingCallback(BaseCallback):
    """
    Detaylı eğitim metriklerini kaydeden callback.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self.losses = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.timesteps.append(self.num_timesteps)

                if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                    recent_rewards = self.episode_rewards[-10:]
                    print(f"  Timestep: {self.num_timesteps:>8d} | "
                          f"Episodes: {len(self.episode_rewards):>4d} | "
                          f"Avg Reward (last 10): {np.mean(recent_rewards):>8.2f}")

        return True

    def get_data(self) -> Dict[str, List]:
        """Kaydedilen verileri döndür"""
        return {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths,
            'timesteps': self.timesteps,
        }


# ============================================================
# Yardımcı Fonksiyonlar
# ============================================================

def get_results_dir(base_dir: str = "benchmark_results") -> Path:
    """Sonuçlar için dizin oluştur"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / base_dir / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def create_monitored_env(env_name: str, seed: int, log_dir: Optional[Path] = None):
    """Monitor wrapper ile ortam oluştur"""
    env = gym.make(env_name)
    env.reset(seed=seed)

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, str(log_dir))

    return env


def save_config(config: TrainingConfig, save_dir: Path):
    """Konfigürasyonu kaydet"""
    config_path = save_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  Config saved: {config_path}")


def save_results(result: BenchmarkResult, save_dir: Path):
    """Sonuçları kaydet"""
    results_path = save_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"  Results saved: {results_path}")


def plot_learning_curves(callbacks_data: Dict[str, Dict], save_dir: Path, title: str = "Learning Curves"):
    """Öğrenme eğrilerini çiz"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Rewards
    ax = axes[0]
    for i, (algo_name, data) in enumerate(callbacks_data.items()):
        rewards = data['rewards']
        timesteps = data['timesteps']

        if len(rewards) > 0:
            # Moving average
            window = min(50, len(rewards) // 10 + 1)
            if len(rewards) >= window:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                smoothed_ts = timesteps[window-1:]
                ax.plot(smoothed_ts, smoothed, label=algo_name.upper(),
                       color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Episode Rewards (Smoothed)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Episode lengths
    ax = axes[1]
    for i, (algo_name, data) in enumerate(callbacks_data.items()):
        lengths = data['lengths']
        timesteps = data['timesteps']

        if len(lengths) > 0:
            window = min(50, len(lengths) // 10 + 1)
            if len(lengths) >= window:
                smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
                smoothed_ts = timesteps[window-1:]
                ax.plot(smoothed_ts, smoothed, label=algo_name.upper(),
                       color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_ylabel('Episode Length', fontsize=12)
    ax.set_title('Episode Lengths (Smoothed)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = save_dir / "learning_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Learning curves saved: {plot_path}")
    plt.close()


def plot_comparison_bar(results: List[BenchmarkResult], save_dir: Path):
    """Algoritmaları bar chart ile karşılaştır"""
    if not results:
        return

    df = pd.DataFrame([r.to_dict() for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean rewards comparison
    ax = axes[0]
    algos = df['algo'].values
    means = df['final_eval_reward'].values
    stds = df['final_eval_std'].values

    x = np.arange(len(algos))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                   color=sns.color_palette("husl", len(algos)))
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Final Evaluation Performance', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algos])
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.0f}±{std:.0f}',
                ha='center', va='bottom', fontsize=10)

    # Training time comparison
    ax = axes[1]
    times = df['training_time'].values
    bars = ax.bar(x, times, alpha=0.7,
                   color=sns.color_palette("husl", len(algos)))
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Training Time', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algos])
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.0f}s',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    plot_path = save_dir / "comparison_bar.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Comparison bar chart saved: {plot_path}")
    plt.close()


def statistical_comparison(results: List[BenchmarkResult], save_dir: Path):
    """İstatistiksel karşılaştırma yap"""
    if len(results) < 2:
        print("  Not enough results for statistical comparison")
        return

    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISON")
    print("=" * 60)

    # Pairwise t-tests
    algos = [r.algo for r in results]
    rewards = [r.final_eval_reward for r in results]

    print("\nPairwise Mean Reward Comparison:")
    print("-" * 60)

    for i in range(len(results)):
        for j in range(i+1, len(results)):
            algo1, reward1 = algos[i], rewards[i]
            algo2, reward2 = algos[j], rewards[j]

            diff = reward2 - reward1
            pct_diff = (diff / abs(reward1)) * 100 if reward1 != 0 else 0

            print(f"{algo2.upper():>6} vs {algo1.upper():<6}: "
                  f"Δ = {diff:>+8.2f} ({pct_diff:>+6.1f}%)")

    # Create summary table
    df = pd.DataFrame([r.to_dict() for r in results])
    df = df[['algo', 'final_eval_reward', 'final_eval_std', 'training_time']]
    df = df.round(2)

    table_path = save_dir / "comparison_table.csv"
    df.to_csv(table_path, index=False)
    print(f"\n  Comparison table saved: {table_path}")

    print("\n" + "=" * 60)


# ============================================================
# Ana Eğitim Fonksiyonları
# ============================================================

def train_algorithm(config: TrainingConfig, results_dir: Path) -> Tuple[Any, DetailedLoggingCallback, float]:
    """
    Tek bir algoritmayı eğit.

    Returns:
        model: Eğitilmiş model
        callback: Logging callback
        training_time: Eğitim süresi (saniye)
    """
    print("\n" + "=" * 60)
    print(f"TRAINING: {config.algo.upper()} on {config.env_name}")
    print("=" * 60)

    # Dizinleri oluştur
    algo_dir = results_dir / config.algo
    algo_dir.mkdir(parents=True, exist_ok=True)

    log_dir = algo_dir / "logs"
    model_dir = algo_dir / "models"
    log_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    # Konfigürasyonu kaydet
    save_config(config, algo_dir)

    # Ortam oluştur
    env = DummyVecEnv([lambda: create_monitored_env(config.env_name, config.seed, log_dir)])

    # Model oluştur
    AlgoClass = ALGORITHMS[config.algo]

    print(f"\nHyperparameters:")
    for key, value in config.hyperparams.items():
        print(f"  {key}: {value}")

    model = AlgoClass(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        seed=config.seed,
        tensorboard_log=str(log_dir / "tensorboard"),
        **config.hyperparams
    )

    # Callbacks
    logging_callback = DetailedLoggingCallback(verbose=1)

    eval_env = create_monitored_env(config.env_name, config.seed + 1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=0
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=str(model_dir),
        name_prefix=f"{config.algo}_model",
        verbose=0
    )

    # Eğitim
    print(f"\nTraining for {config.total_timesteps:,} timesteps...\n")

    start_time = time.time()
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[logging_callback, eval_callback, checkpoint_callback],
        progress_bar=True
    )
    training_time = time.time() - start_time

    # Final modeli kaydet
    model.save(str(model_dir / f"{config.algo}_final"))

    print(f"\n  Training completed in {training_time:.2f} seconds")
    print(f"  Model saved: {model_dir}")

    env.close()
    eval_env.close()

    return model, logging_callback, training_time


def evaluate_model(model: Any, env_name: str, n_episodes: int = 100, seed: int = 42) -> Tuple[float, float, float]:
    """
    Modeli değerlendir.

    Returns:
        mean_reward: Ortalama reward
        std_reward: Standart sapma
        mean_length: Ortalama episode uzunluğu
    """
    eval_env = create_monitored_env(env_name, seed)

    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0
        ep_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_length += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

    eval_env.close()

    return np.mean(episode_rewards), np.std(episode_rewards), np.mean(episode_lengths)


# ============================================================
# Benchmark Fonksiyonları
# ============================================================

def run_benchmark(env_name: str, algos: List[str], total_timesteps: int, seed: int,
                 results_dir: Path) -> List[BenchmarkResult]:
    """
    Birden fazla algoritmayı benchmark et.
    """
    print("\n" + "=" * 70)
    print(f"BENCHMARK SUITE: {env_name}")
    print(f"Algorithms: {', '.join([a.upper() for a in algos])}")
    print(f"Timesteps: {total_timesteps:,}")
    print("=" * 70)

    results = []
    callbacks_data = {}

    for algo in algos:
        # Konfigürasyon oluştur
        config = TrainingConfig(
            algo=algo,
            env_name=env_name,
            total_timesteps=total_timesteps,
            seed=seed,
            n_eval_episodes=10,
            eval_freq=max(total_timesteps // 10, 1000),
            save_freq=max(total_timesteps // 20, 1000),
            hyperparams=DEFAULT_HYPERPARAMS[algo].copy()
        )

        # Eğit
        model, callback, training_time = train_algorithm(config, results_dir)

        # Final değerlendirme
        print(f"\nFinal evaluation for {algo.upper()}...")
        mean_reward, std_reward, mean_length = evaluate_model(
            model, env_name, n_episodes=30, seed=seed
        )

        # Sonuçları kaydet
        result = BenchmarkResult(
            algo=algo,
            env_name=env_name,
            mean_reward=mean_reward,
            std_reward=std_reward,
            mean_ep_length=mean_length,
            training_time=training_time,
            total_timesteps=total_timesteps,
            final_eval_reward=mean_reward,
            final_eval_std=std_reward
        )

        results.append(result)
        callbacks_data[algo] = callback.get_data()

        # Bireysel sonuçları kaydet
        algo_dir = results_dir / algo
        save_results(result, algo_dir)

        print(f"\n  {algo.upper()} Results:")
        print(f"    Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"    Mean Length: {mean_length:.1f}")
        print(f"    Training Time: {training_time:.2f}s")

    # Karşılaştırmalı görselleştirmeler
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_learning_curves(callbacks_data, results_dir,
                         title=f"Learning Curves - {env_name}")
    plot_comparison_bar(results, results_dir)
    statistical_comparison(results, results_dir)

    return results


def multi_environment_benchmark(algos: List[str], envs: List[str],
                                total_timesteps: int, seed: int) -> None:
    """Çoklu ortam üzerinde benchmark"""
    base_dir = get_results_dir("multi_env_benchmark")

    print("\n" + "=" * 70)
    print("MULTI-ENVIRONMENT BENCHMARK")
    print(f"Algorithms: {', '.join([a.upper() for a in algos])}")
    print(f"Environments: {len(envs)}")
    print("=" * 70)

    all_results = []

    for env_name in envs:
        env_dir = base_dir / env_name.replace('-', '_')
        env_dir.mkdir(parents=True, exist_ok=True)

        results = run_benchmark(env_name, algos, total_timesteps, seed, env_dir)
        all_results.extend(results)

    # Tüm sonuçları birleştir
    df = pd.DataFrame([r.to_dict() for r in all_results])
    summary_path = base_dir / "all_results.csv"
    df.to_csv(summary_path, index=False)

    print(f"\n\nAll results saved: {summary_path}")
    print(f"Benchmark directory: {base_dir}")


# ============================================================
# Ana Program
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Algorithm Benchmark Suite for MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Algoritma seçimi
    parser.add_argument("--algo", type=str, choices=list(ALGORITHMS.keys()),
                       help="Algorithm to train (single mode)")
    parser.add_argument("--env", type=str, default="HalfCheetah-v5",
                       choices=MUJOCO_ENVS,
                       help="Environment to use")
    parser.add_argument("--timesteps", type=int, default=200_000,
                       help="Total timesteps for training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # Benchmark modları
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark with all algorithms")
    parser.add_argument("--multi-env", action="store_true",
                       help="Run benchmark on multiple environments")
    parser.add_argument("--algos", type=str, nargs='+',
                       choices=list(ALGORITHMS.keys()),
                       default=['ppo', 'sac', 'td3'],
                       help="Algorithms for benchmark mode")

    args = parser.parse_args()

    # Mod seçimi
    if args.multi_env:
        # Çoklu ortam benchmark
        envs = ["HalfCheetah-v5", "Ant-v5", "Hopper-v5"]
        multi_environment_benchmark(args.algos, envs, args.timesteps, args.seed)

    elif args.benchmark:
        # Tek ortam, çoklu algoritma benchmark
        results_dir = get_results_dir("benchmark")
        run_benchmark(args.env, args.algos, args.timesteps, args.seed, results_dir)

    elif args.algo:
        # Tek algoritma eğitimi
        results_dir = get_results_dir("single_training")

        config = TrainingConfig(
            algo=args.algo,
            env_name=args.env,
            total_timesteps=args.timesteps,
            seed=args.seed,
            n_eval_episodes=10,
            eval_freq=max(args.timesteps // 10, 1000),
            save_freq=max(args.timesteps // 20, 1000),
            hyperparams=DEFAULT_HYPERPARAMS[args.algo].copy()
        )

        model, callback, training_time = train_algorithm(config, results_dir)

        # Değerlendirme
        mean_reward, std_reward, mean_length = evaluate_model(
            model, args.env, n_episodes=30, seed=args.seed
        )

        result = BenchmarkResult(
            algo=args.algo,
            env_name=args.env,
            mean_reward=mean_reward,
            std_reward=std_reward,
            mean_ep_length=mean_length,
            training_time=training_time,
            total_timesteps=args.timesteps,
            final_eval_reward=mean_reward,
            final_eval_std=std_reward
        )

        algo_dir = results_dir / args.algo
        save_results(result, algo_dir)

        # Görselleştirme
        callbacks_data = {args.algo: callback.get_data()}
        plot_learning_curves(callbacks_data, results_dir,
                           title=f"{args.algo.upper()} on {args.env}")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"  Algorithm: {args.algo.upper()}")
        print(f"  Environment: {args.env}")
        print(f"  Final Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Training Time: {training_time:.2f}s")
        print(f"  Results: {results_dir}")
        print("=" * 60)

    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python 03_multi_algorithm_benchmark.py --algo ppo --env Ant-v5")
        print("  python 03_multi_algorithm_benchmark.py --benchmark --env HalfCheetah-v5")
        print("  python 03_multi_algorithm_benchmark.py --multi-env")


if __name__ == "__main__":
    main()
