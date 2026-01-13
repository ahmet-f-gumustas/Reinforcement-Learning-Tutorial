"""
01 - Gymnasium Ortaminin Temel Kullanimi

Bu ornekte Gymnasium kutuphanesinin temel kullanimini ogrenecegiz.
CartPole-v1 ortamini inceleyecegiz.
"""

import gymnasium as gym


def main():
    # ============================================
    # 1. ORTAM OLUSTURMA
    # ============================================
    print("=" * 50)
    print("1. ORTAM OLUSTURMA")
    print("=" * 50)

    # CartPole ortamini olustur
    # render_mode="human" gorsellestirme icin (opsiyonel)
    env = gym.make("CartPole-v1")

    print(f"Ortam: {env.spec.id}")
    print(f"Max episode adimi: {env.spec.max_episode_steps}")

    # ============================================
    # 2. OBSERVATION SPACE (GOZLEM UZAYI)
    # ============================================
    print("\n" + "=" * 50)
    print("2. OBSERVATION SPACE")
    print("=" * 50)

    print(f"Observation Space: {env.observation_space}")
    print(f"Observation Shape: {env.observation_space.shape}")
    print(f"Observation Low: {env.observation_space.low}")
    print(f"Observation High: {env.observation_space.high}")

    print("\nCartPole Gozlem Degiskenleri:")
    print("  [0] Cart Position (Araba Pozisyonu): -4.8 ile 4.8 arasi")
    print("  [1] Cart Velocity (Araba Hizi): -inf ile inf arasi")
    print("  [2] Pole Angle (Cubuk Acisi): ~-0.42 ile ~0.42 rad arasi")
    print("  [3] Pole Angular Velocity (Cubuk Aci Hizi): -inf ile inf arasi")

    # ============================================
    # 3. ACTION SPACE (EYLEM UZAYI)
    # ============================================
    print("\n" + "=" * 50)
    print("3. ACTION SPACE")
    print("=" * 50)

    print(f"Action Space: {env.action_space}")
    print(f"Action Space n: {env.action_space.n}")

    print("\nCartPole Eylemleri:")
    print("  0: Sola it")
    print("  1: Saga it")

    # ============================================
    # 4. ORTAMI SIFIRLAMA
    # ============================================
    print("\n" + "=" * 50)
    print("4. ORTAMI SIFIRLAMA")
    print("=" * 50)

    # Ortami sifirla ve baslangic gozlemini al
    observation, info = env.reset(seed=42)

    print(f"Baslangic Gozlemi: {observation}")
    print(f"Info: {info}")

    # ============================================
    # 5. BIR ADIM ATMA
    # ============================================
    print("\n" + "=" * 50)
    print("5. BIR ADIM ATMA")
    print("=" * 50)

    # Saga it (action = 1)
    action = 1
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"Eylem: {action} (Saga it)")
    print(f"Yeni Gozlem: {observation}")
    print(f"Odul: {reward}")
    print(f"Terminated: {terminated} (Episode dogal olarak bitti mi?)")
    print(f"Truncated: {truncated} (Episode zaman asimi ile mi bitti?)")
    print(f"Info: {info}")

    # ============================================
    # 6. BIR EPISODE CALISTIRMA
    # ============================================
    print("\n" + "=" * 50)
    print("6. BIR EPISODE CALISTIRMA")
    print("=" * 50)

    observation, info = env.reset(seed=42)
    total_reward = 0
    step_count = 0

    done = False
    while not done:
        # Rastgele eylem sec
        action = env.action_space.sample()

        # Adim at
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        # Episode bitti mi kontrol et
        done = terminated or truncated

    print(f"Episode tamamlandi!")
    print(f"Toplam adim: {step_count}")
    print(f"Toplam odul: {total_reward}")

    # Ortami kapat
    env.close()
    print("\nOrtam kapatildi.")


if __name__ == "__main__":
    main()
