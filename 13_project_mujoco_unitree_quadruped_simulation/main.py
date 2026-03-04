#!/usr/bin/env python3
"""
Multi-Agent Quadruped Robot Simulasyonu - Ana Giris Noktasi
Multi-Agent Quadruped Robot Simulation - Main Entry Point

Bu proje, Unitree Go2 tarzinda bir dort bacakli robotun MuJoCo
simulasyonunda Multi-Agent Reinforcement Learning ile kontrolunu gosterir.

This project demonstrates Multi-Agent Reinforcement Learning control
of a Unitree Go2-style quadruped robot in MuJoCo simulation.

Kullanim / Usage:
    # Tek-agent PPO baseline egitimi
    python main.py --mode train --agent single_ppo --timesteps 500000

    # MAPPO (per-leg CTDE) egitimi
    python main.py --mode train --agent mappo --timesteps 500000

    # Hiyerarsik (manager-worker) egitimi
    python main.py --mode train --agent hierarchical --timesteps 500000

    # Coklu robot formation egitimi
    python main.py --mode train --agent multi_robot --timesteps 300000

    # Egitilmis agent'i degerlendir
    python main.py --mode eval --agent mappo --render

    # Tum yaklasimlari karsilastir
    python main.py --mode compare --timesteps 200000
"""

import argparse
import sys
import numpy as np
import torch
from pathlib import Path

# Proje kokunu path'e ekle / Add project root to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from config import get_config, SAVE_DIR, EnvConfig, PPOConfig, MAPPOConfig, HierarchicalConfig, MultiRobotConfig
from training.trainer import SingleAgentTrainer, MAPPOTrainer, HierarchicalTrainer, MultiRobotTrainer
from training.callbacks import BestModelCallback
from utils.visualization import (
    plot_training_curves,
    plot_comparison_bar,
    plot_radar_chart,
    print_summary_table,
)


# =============================================================================
# Egitim Fonksiyonlari / Training Functions
# =============================================================================

def train_agent(agent_type: str, timesteps: int, seed: int):
    """Belirtilen agent turunu egit / Train the specified agent type."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_config, agent_config = get_config(agent_type)

    trainers = {
        "single_ppo": lambda: SingleAgentTrainer(env_config, agent_config),
        "mappo": lambda: MAPPOTrainer(env_config, agent_config),
        "hierarchical": lambda: HierarchicalTrainer(env_config, agent_config),
        "multi_robot": lambda: MultiRobotTrainer(env_config, agent_config),
    }

    trainer = trainers[agent_type]()
    history = trainer.train(total_timesteps=timesteps)
    return history, trainer


def evaluate_agent(agent_type: str, num_episodes: int, render: bool, seed: int):
    """Egitilmis agent'i degerlendir / Evaluate trained agent."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_config, agent_config = get_config(agent_type)

    trainers = {
        "single_ppo": lambda: SingleAgentTrainer(env_config, agent_config),
        "mappo": lambda: MAPPOTrainer(env_config, agent_config),
        "hierarchical": lambda: HierarchicalTrainer(env_config, agent_config),
        "multi_robot": lambda: MultiRobotTrainer(env_config, agent_config),
    }

    trainer = trainers[agent_type]()

    # Best model yukle / Load best model
    save_dir = str(SAVE_DIR / agent_type)
    state_dict = BestModelCallback.load_best(save_dir)

    if state_dict is not None:
        if agent_type == "single_ppo":
            trainer.agent.load_state_dict(state_dict)
        elif agent_type == "mappo":
            trainer.controller.load_state_dict(state_dict)
        elif agent_type == "hierarchical":
            trainer.controller.load_state_dict(state_dict)
        elif agent_type == "multi_robot":
            for i, agent in enumerate(trainer.agents):
                if f"agent_{i}" in state_dict:
                    agent.load_state_dict(state_dict[f"agent_{i}"])
    else:
        print(f"  UYARI: Best model bulunamadi ({save_dir}/best_model.pt), rastgele policy ile degerlendirme yapilacak.")

    trainer.evaluate(num_episodes=num_episodes, render=render)


def compare_all(timesteps: int, seed: int):
    """Tum yaklasimlari egit ve karsilastir / Train and compare all approaches."""
    print("\n" + "#" * 60)
    print("  TUM YAKLASIMLARIN KARSILASTIRMASI")
    print("  COMPARING ALL APPROACHES")
    print("#" * 60)

    agent_types = ["single_ppo", "mappo", "hierarchical", "multi_robot"]
    display_names = {
        "single_ppo": "Single PPO",
        "mappo": "MAPPO",
        "hierarchical": "Hierarchical",
        "multi_robot": "Multi-Robot",
    }

    histories = {}
    eval_results = {}

    for agent_type in agent_types:
        print(f"\n{'='*60}")
        print(f"  {display_names[agent_type]} egitiliyor...")
        print(f"{'='*60}")

        try:
            history, trainer = train_agent(agent_type, timesteps, seed)
            histories[display_names[agent_type]] = history

            # Degerlendirme
            if hasattr(trainer, 'evaluate'):
                mean_reward = trainer.evaluate(num_episodes=10)
            else:
                mean_reward = history.mean_reward_last_100

            last_rewards = history.episode_rewards[-100:] if history.episode_rewards else [0]
            eval_results[display_names[agent_type]] = {
                "mean": float(np.mean(last_rewards)),
                "std": float(np.std(last_rewards)),
            }

        except Exception as e:
            print(f"  HATA: {agent_type} egitimi basarisiz: {e}")
            continue

    # Gorsellestirme / Visualization
    if histories:
        print("\n" + "=" * 60)
        print("  SONUCLAR / RESULTS")
        print("=" * 60)

        try:
            plot_training_curves(histories, show=False)
        except Exception as e:
            print(f"  Training curves plot hatasi: {e}")

        if eval_results:
            try:
                plot_comparison_bar(eval_results, show=False)
            except Exception as e:
                print(f"  Comparison bar plot hatasi: {e}")

            # Basit sonuc tablosu
            summary = {}
            for name, vals in eval_results.items():
                summary[name] = {
                    "mean_reward": vals["mean"],
                    "mean_velocity": 0.0,
                    "energy_eff": 0.0,
                    "stability": 0.0,
                }
            print_summary_table(summary)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Quadruped Robot Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler / Examples:
  python main.py --mode train --agent single_ppo --timesteps 500000
  python main.py --mode train --agent mappo --timesteps 500000
  python main.py --mode train --agent hierarchical --timesteps 500000
  python main.py --mode train --agent multi_robot --timesteps 300000
  python main.py --mode eval --agent mappo --render
  python main.py --mode compare --timesteps 200000
        """,
    )

    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "eval", "compare"],
        help="Calisma modu: train, eval, compare (varsayilan: train)",
    )
    parser.add_argument(
        "--agent", type=str, default="single_ppo",
        choices=["single_ppo", "mappo", "hierarchical", "multi_robot"],
        help="Agent turu (varsayilan: single_ppo)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Toplam egitim adimi (varsayilan: 500000)",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10,
        help="Degerlendirme episode sayisi (varsayilan: 10)",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Gorsellestirme ile calistir",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (varsayilan: 42)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  MULTI-AGENT QUADRUPED SIMULATION")
    print(f"  Mod: {args.mode} | Agent: {args.agent}")
    print(f"  Timesteps: {args.timesteps:,} | Seed: {args.seed}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    if args.mode == "train":
        train_agent(args.agent, args.timesteps, args.seed)

    elif args.mode == "eval":
        evaluate_agent(args.agent, args.eval_episodes, args.render, args.seed)

    elif args.mode == "compare":
        compare_all(args.timesteps, args.seed)

    print("\nBitti! / Done!")


if __name__ == "__main__":
    main()
