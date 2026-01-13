"""
02 - Rastgele Eylem Secen Agent

Bu ornekte basit bir Random Agent implementasyonu yapacagiz.
Agent her adimda rastgele bir eylem secer - ogrenme yapmaz.
Bu, daha sonra gelistirecegimiz akilli agentler icin temel (baseline) olacak.
"""

import gymnasium as gym
import numpy as np


class RandomAgent:
    """Rastgele eylem secen basit bir agent."""

    def __init__(self, action_space):
        """
        Args:
            action_space: Gymnasium action space
        """
        self.action_space = action_space

    def select_action(self, observation):
        """
        Gozleme bakmadan rastgele eylem secer.

        Args:
            observation: Ortamdan gelen gozlem (kullanilmiyor)

        Returns:
            action: Rastgele secilmis eylem
        """
        return self.action_space.sample()

    def learn(self, observation, action, reward, next_observation, done):
        """
        Random agent ogrenme yapmaz.
        Bu metod ileride gelistirilecek agentler icin arayuz saglar.
        """
        pass


def run_episode(env, agent, render=False):
    """
    Bir episode calistirir.

    Args:
        env: Gymnasium ortami
        agent: Eylem secen agent
        render: Gorsellestirme yapilsin mi

    Returns:
        total_reward: Episode boyunca toplanan toplam odul
        step_count: Episode'daki adim sayisi
    """
    observation, info = env.reset()
    total_reward = 0
    step_count = 0
    done = False

    while not done:
        # Agent'tan eylem al
        action = agent.select_action(observation)

        # Ortamda adim at
        next_observation, reward, terminated, truncated, info = env.step(action)

        # Agent'in ogrenme fonksiyonunu cagir (Random agent icin bos)
        done = terminated or truncated
        agent.learn(observation, action, reward, next_observation, done)

        # Degiskenleri guncelle
        observation = next_observation
        total_reward += reward
        step_count += 1

    return total_reward, step_count


def evaluate_agent(env, agent, num_episodes=100):
    """
    Agent'i birden fazla episode uzerinde degerlendirir.

    Args:
        env: Gymnasium ortami
        agent: Degerlendirilecek agent
        num_episodes: Calistirilacak episode sayisi

    Returns:
        results: Her episode'un odul ve adim bilgilerini iceren dict
    """
    rewards = []
    steps = []

    for episode in range(num_episodes):
        total_reward, step_count = run_episode(env, agent)
        rewards.append(total_reward)
        steps.append(step_count)

        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Son odul: {total_reward:.1f}, "
                  f"Ortalama: {np.mean(rewards):.2f}")

    return {
        "rewards": rewards,
        "steps": steps,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_steps": np.mean(steps),
        "max_reward": np.max(rewards),
        "min_reward": np.min(rewards)
    }


def main():
    print("=" * 60)
    print("RANDOM AGENT DEGERLENDIRMESI")
    print("=" * 60)

    # Ortam olustur
    env = gym.make("CartPole-v1")

    # Agent olustur
    agent = RandomAgent(env.action_space)

    print(f"\nOrtam: CartPole-v1")
    print(f"Agent: Random Agent")
    print(f"Episode sayisi: 100")
    print("\nDegerlendirme basladi...\n")

    # Agent'i degerlendir
    results = evaluate_agent(env, agent, num_episodes=100)

    # Sonuclari yazdir
    print("\n" + "=" * 60)
    print("SONUCLAR")
    print("=" * 60)
    print(f"Ortalama Odul: {results['mean_reward']:.2f} (+/- {results['std_reward']:.2f})")
    print(f"Ortalama Adim: {results['mean_steps']:.2f}")
    print(f"Max Odul: {results['max_reward']:.1f}")
    print(f"Min Odul: {results['min_reward']:.1f}")

    print("\n" + "-" * 60)
    print("YORUM")
    print("-" * 60)
    print("""
Random agent CartPole'da ortalama 20-25 adim hayatta kalir.
CartPole'da maksimum skor 500'dur (episode 500 adim sonra kesilir).

Karsilastirma icin:
- Random Agent: ~20-25 ortalama odul
- Basit kural tabanli: ~50-100
- Egitilmis RL agent: ~500 (maksimum)

Ilerleyen haftalarda daha iyi agentler gelistirecegiz!
    """)

    # Ortami kapat
    env.close()


if __name__ == "__main__":
    main()
