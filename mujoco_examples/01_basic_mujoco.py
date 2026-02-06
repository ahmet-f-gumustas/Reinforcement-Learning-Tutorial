#!/usr/bin/env python3
"""
MuJoCo ile Temel Simülasyon Örneği

Bu script MuJoCo ortamlarının temel kullanımını gösterir:
- Ortam oluşturma
- Observation ve action space'leri anlama
- Random policy ile simülasyon
- Video kaydetme (opsiyonel)

Kullanım:
    python 01_basic_mujoco.py                    # Sadece simülasyon
    python 01_basic_mujoco.py --render           # Görselleştirme ile
    python 01_basic_mujoco.py --record           # Video kaydet
    python 01_basic_mujoco.py --env Ant-v5       # Farklı ortam
"""

import argparse
import gymnasium as gym
import numpy as np
from pathlib import Path


def get_available_envs():
    """MuJoCo ortamlarının listesi"""
    return [
        "HalfCheetah-v5",    # 6 eklemli koşan robot
        "Ant-v5",           # 4 bacaklı karınca robot
        "Hopper-v5",        # Tek bacaklı zıplayan robot
        "Walker2d-v5",      # 2 bacaklı yürüyen robot
        "Humanoid-v5",      # 17 eklemli insansı robot
        "Swimmer-v5",       # Yüzen yılan robot
        "InvertedPendulum-v5",        # Ters sarkaç
        "InvertedDoublePendulum-v5",  # Çift ters sarkaç
    ]


def print_env_info(env):
    """Ortam bilgilerini yazdır"""
    print("\n" + "=" * 50)
    print(f"Ortam: {env.spec.id}")
    print("=" * 50)

    # Observation space
    obs_space = env.observation_space
    print(f"\nObservation Space:")
    print(f"  Tip: {type(obs_space).__name__}")
    print(f"  Shape: {obs_space.shape}")
    print(f"  Low: {obs_space.low.min():.2f}")
    print(f"  High: {obs_space.high.max():.2f}")

    # Action space
    act_space = env.action_space
    print(f"\nAction Space:")
    print(f"  Tip: {type(act_space).__name__}")
    print(f"  Shape: {act_space.shape}")
    print(f"  Low: {act_space.low}")
    print(f"  High: {act_space.high}")

    print("=" * 50 + "\n")


def run_episode(env, render=False):
    """Tek episode çalıştır"""
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done:
        # Random action al
        action = env.action_space.sample()

        # Ortamda adım at
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1

        if render:
            env.render()

    return total_reward, steps


def main():
    parser = argparse.ArgumentParser(description="MuJoCo Temel Simülasyon")
    parser.add_argument("--env", type=str, default="HalfCheetah-v5",
                        choices=get_available_envs(),
                        help="Kullanılacak ortam")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episode sayısı")
    parser.add_argument("--render", action="store_true",
                        help="Simülasyonu görselleştir")
    parser.add_argument("--record", action="store_true",
                        help="Video kaydet")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Seed ayarla
    np.random.seed(args.seed)

    # Render modu belirle
    render_mode = None
    if args.render:
        render_mode = "human"
    elif args.record:
        render_mode = "rgb_array"

    # Ortamı oluştur
    env = gym.make(args.env, render_mode=render_mode)

    # Video kayıt wrapper
    if args.record:
        video_dir = Path(__file__).parent / "videos"
        video_dir.mkdir(exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda x: True,  # Her episode'u kaydet
            name_prefix=args.env
        )

    # Ortam bilgilerini göster
    print_env_info(env)

    # Episode'ları çalıştır
    print("Random policy ile simülasyon başlıyor...\n")
    rewards = []
    steps_list = []

    for ep in range(args.episodes):
        reward, steps = run_episode(env, render=args.render)
        rewards.append(reward)
        steps_list.append(steps)
        print(f"Episode {ep + 1}: Reward = {reward:.2f}, Steps = {steps}")

    # Özet istatistikler
    print("\n" + "-" * 40)
    print("ÖZET")
    print("-" * 40)
    print(f"Ortalama Reward: {np.mean(rewards):.2f} (+/- {np.std(rewards):.2f})")
    print(f"Ortalama Steps: {np.mean(steps_list):.1f}")
    print(f"Max Reward: {max(rewards):.2f}")
    print(f"Min Reward: {min(rewards):.2f}")

    if args.record:
        print(f"\nVideolar kaydedildi: {video_dir}/")

    env.close()
    print("\nSimülasyon tamamlandı!")


if __name__ == "__main__":
    main()
