# Reinforcement Learning Tutorial

Pekistirmeli Ogrenme (Reinforcement Learning) konusunda sifirdan ileri seviyeye kapsamli bir Turkce tutorial serisi.

> **English version:** [README.md](README.md)

## Gereksinimler

- Python 3.10+
- Gymnasium
- NumPy
- Matplotlib
- PyTorch (ileri konular icin)

## Kurulum

```bash
# Virtual environment olusturma
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Gereksinimleri yukleme
pip install gymnasium numpy matplotlib torch
```

## Haftalik Mufredat

| Hafta | Konu | Klasor | Durum |
|-------|------|--------|-------|
| 1 | RL'ye Giris ve Temel Kavramlar | [01_introduction](01_introduction/) | Tamamlandi |
| 2 | Markov Karar Surecleri (MDP) | [02_mdp](02_mdp/) | Tamamlandi |
| 3 | Dynamic Programming | [03_dynamic_programming](03_dynamic_programming/) | Tamamlandi |
| 4 | Monte Carlo Yontemleri | [04_monte_carlo](04_monte_carlo/) | Bekliyor |
| 5 | Temporal Difference (TD) Learning | [05_temporal_difference](05_temporal_difference/) | Bekliyor |
| 6 | Q-Learning ve SARSA | [06_q_learning_sarsa](06_q_learning_sarsa/) | Bekliyor |
| 7 | Function Approximation | [07_function_approximation](07_function_approximation/) | Bekliyor |
| 8 | Deep Q-Network (DQN) | [08_dqn](08_dqn/) | Bekliyor |
| 9 | Policy Gradient Yontemleri | [09_policy_gradient](09_policy_gradient/) | Bekliyor |
| 10 | Actor-Critic Yontemleri | [10_actor_critic](10_actor_critic/) | Bekliyor |
| 11 | Proximal Policy Optimization (PPO) | [11_ppo](11_ppo/) | Bekliyor |
| 12 | Ileri Duzey Konular ve Proje | [12_advanced](12_advanced/) | Bekliyor |

## Konu Detaylari

### Hafta 1: RL'ye Giris ve Temel Kavramlar
- Pekistirmeli Ogrenmenin temelleri
- Agent, Environment, State, Action, Reward kavramlari
- Gymnasium kutuphanesi ile ilk adimlar
- CartPole ve diger temel ortamlar

### Hafta 2: Markov Karar Surecleri (MDP)
- Markov ozelligi
- State gecis olasiliklari
- Odul fonksiyonlari
- Bellman denklemleri

### Hafta 3: Dynamic Programming
- Policy Evaluation
- Policy Improvement
- Policy Iteration
- Value Iteration

### Hafta 4: Monte Carlo Yontemleri
- First-visit vs Every-visit MC
- MC Prediction
- MC Control
- Importance Sampling

### Hafta 5: Temporal Difference Learning
- TD(0) Prediction
- TD vs MC vs DP karsilastirmasi
- n-step TD
- Eligibility Traces

### Hafta 6: Q-Learning ve SARSA
- On-policy vs Off-policy
- SARSA algoritmasi
- Q-Learning algoritmasi
- Expected SARSA

### Hafta 7: Function Approximation
- Tabular vs Approximation yontemleri
- Linear function approximation
- Feature engineering
- Gradient descent yontemleri

### Hafta 8: Deep Q-Network (DQN)
- Neural network ile Q-function
- Experience Replay
- Target Network
- DQN varyantlari (Double DQN, Dueling DQN)

### Hafta 9: Policy Gradient Yontemleri
- Policy-based methods
- REINFORCE algoritmasi
- Baseline kavrami
- Variance reduction teknikleri

### Hafta 10: Actor-Critic Yontemleri
- Actor-Critic mimarisi
- A2C (Advantage Actor-Critic)
- A3C (Asynchronous A3C)
- GAE (Generalized Advantage Estimation)

### Hafta 11: Proximal Policy Optimization (PPO)
- Trust Region Policy Optimization (TRPO)
- PPO-Clip
- PPO-Penalty
- Pratik implementasyon

### Hafta 12: Ileri Duzey Konular ve Proje
- Multi-agent RL
- Model-based RL
- Inverse RL
- Final proje: Kendi ortaminizi cozun

## Kaynaklar

- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [David Silver RL Course](https://www.davidsilver.uk/teaching/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## Lisans

MIT License
