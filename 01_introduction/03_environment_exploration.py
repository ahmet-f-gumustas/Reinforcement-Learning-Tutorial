"""
03 - Farkli Gymnasium Ortamlarini Kesfetme

Bu ornekte cesitli Gymnasium ortamlarini inceleyecegiz.
Her ortamin observation space, action space ve odul yapisini anlayacagiz.
"""

import gymnasium as gym
import numpy as np


def explore_environment(env_name, num_episodes=5, verbose=True):
    """
    Bir Gymnasium ortamini kesfeder ve bilgilerini yazdirir.

    Args:
        env_name: Ortam adi
        num_episodes: Calistirilacak episode sayisi
        verbose: Detayli bilgi yazdirilsin mi
    """
    try:
        env = gym.make(env_name)
    except Exception as e:
        print(f"Ortam yuklenemedi: {env_name}")
        print(f"Hata: {e}")
        return None

    if verbose:
        print("\n" + "=" * 60)
        print(f"ORTAM: {env_name}")
        print("=" * 60)

        # Observation Space
        print(f"\n[Observation Space]")
        print(f"  Tur: {type(env.observation_space).__name__}")
        print(f"  Detay: {env.observation_space}")

        if hasattr(env.observation_space, 'shape'):
            print(f"  Shape: {env.observation_space.shape}")
        if hasattr(env.observation_space, 'n'):
            print(f"  n (discrete): {env.observation_space.n}")

        # Action Space
        print(f"\n[Action Space]")
        print(f"  Tur: {type(env.action_space).__name__}")
        print(f"  Detay: {env.action_space}")

        if hasattr(env.action_space, 'n'):
            print(f"  n (discrete): {env.action_space.n}")
        if hasattr(env.action_space, 'shape'):
            print(f"  Shape: {env.action_space.shape}")

    # Episode'lari calistir
    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    if verbose:
        print(f"\n[{num_episodes} Episode Sonuclari]")
        print(f"  Ortalama Odul: {np.mean(episode_rewards):.2f}")
        print(f"  Odul Std: {np.std(episode_rewards):.2f}")
        print(f"  Ortalama Uzunluk: {np.mean(episode_lengths):.1f}")
        print(f"  Min/Max Odul: {np.min(episode_rewards):.1f} / {np.max(episode_rewards):.1f}")

    env.close()

    return {
        "env_name": env_name,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths)
    }


def compare_environments():
    """Birden fazla ortami karsilastirir."""

    # Kesfedilecek ortamlar
    environments = [
        "CartPole-v1",
        "MountainCar-v0",
        "Acrobot-v1",
        "LunarLander-v3",
    ]

    print("\n" + "#" * 60)
    print("# GYMNASIUM ORTAM KARSILASTIRMASI")
    print("#" * 60)

    results = []

    for env_name in environments:
        result = explore_environment(env_name, num_episodes=10)
        if result:
            results.append(result)

    # Ozet tablo
    print("\n" + "=" * 60)
    print("OZET KARSILASTIRMA TABLOSU")
    print("=" * 60)
    print(f"{'Ortam':<20} {'Ort. Odul':>12} {'Std':>10} {'Ort. Uzunluk':>12}")
    print("-" * 60)

    for r in results:
        print(f"{r['env_name']:<20} {r['mean_reward']:>12.2f} {r['std_reward']:>10.2f} {r['mean_length']:>12.1f}")


def understand_cartpole():
    """CartPole ortamini detayli inceler."""

    print("\n" + "#" * 60)
    print("# CARTPOLE DETAYLI INCELEME")
    print("#" * 60)

    env = gym.make("CartPole-v1")

    print("""
    CartPole Problemi:
    ------------------
    Bir araba uzerinde dikey duran bir cubuk var.
    Amac: Cubugu dusurmmeden mumkun oldugu kadar uzun sure dengede tutmak.

    Fizik:
    - Cubuk baslangicta hafif bir aciyla baslar
    - Yercekimi cubugu dusurmeye calisir
    - Arabayi saga/sola iterek dengeleyebiliriz
    """)

    # Bir episode izle
    print("\nOrnek Episode:")
    print("-" * 40)

    obs, _ = env.reset(seed=42)
    print(f"Baslangic durumu:")
    print(f"  Araba pozisyonu: {obs[0]:.4f}")
    print(f"  Araba hizi: {obs[1]:.4f}")
    print(f"  Cubuk acisi: {obs[2]:.4f} rad ({np.degrees(obs[2]):.2f} derece)")
    print(f"  Cubuk aci hizi: {obs[3]:.4f}")

    # Birkac adim at
    print("\nIlk 5 adim:")
    for i in range(5):
        action = env.action_space.sample()
        action_name = "SAGA" if action == 1 else "SOLA"
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"  Adim {i+1}: {action_name:>5} -> Cubuk acisi: {np.degrees(obs[2]):>7.2f} derece, Odul: {reward}")

        if terminated or truncated:
            print("  Episode bitti!")
            break

    print("""
    Episode Bitis Kosullari:
    ------------------------
    1. Cubuk acisi |12 derece|'yi gecerse -> BASARISIZ
    2. Araba pozisyonu |2.4|'u gecerse -> BASARISIZ
    3. 500 adim tamamlanirsa -> BASARILI

    Odul Yapisi:
    ------------
    - Her adimda +1 odul
    - Maksimum toplam odul: 500
    """)

    env.close()


def main():
    print("=" * 60)
    print("GYMNASIUM ORTAMLARI KESFI")
    print("=" * 60)
    print("""
    Bu script farkli Gymnasium ortamlarini kesfetmenizi saglar.

    Secenekler:
    1. Tekli ortam kesfet
    2. Ortamlari karsilastir
    3. CartPole detayli inceleme
    """)

    # CartPole detayli inceleme
    understand_cartpole()

    # Ortamlari karsilastir
    compare_environments()

    print("\n" + "=" * 60)
    print("ALISTIRMA ONERILERI")
    print("=" * 60)
    print("""
    1. explore_environment() fonksiyonunu farkli ortamlarla cagirin
    2. Episode sayisini artirarak daha guvenilir istatistikler elde edin
    3. Kendi gozlemlerinizi not alin:
       - Hangi ortamlar daha zor?
       - Odul yapilari nasil farklilik gosteriyor?
       - Continuous vs Discrete action space farki ne?
    """)


if __name__ == "__main__":
    main()
