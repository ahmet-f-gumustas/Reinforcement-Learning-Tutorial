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
| 4 | Monte Carlo Yontemleri | [04_monte_carlo](04_monte_carlo/) | Tamamlandi |
| 5 | Temporal Difference (TD) Learning | [05_temporal_difference](05_temporal_difference/) | Tamamlandi |
| 6 | Q-Learning ve SARSA | [06_td_control](06_td_control/) | Tamamlandi |
| 7 | Function Approximation | [07_function_approximation](07_function_approximation/) | Tamamlandi |
| 8 | Deep Q-Network (DQN) | [08_dqn](08_dqn/) | Tamamlandi |
| 9 | Policy Gradient Yontemleri | [09_policy_gradient](09_policy_gradient/) | Tamamlandi |
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

## Pratik Ornekler

| Klasor | Aciklama | Algoritmalar |
|--------|----------|--------------|
| [openai_gym](openai_gym/) | OpenAI Gymnasium ile interaktif oyun simulasyonlari | DQN, TD3, Q-Learning |

`openai_gym` klasoru, egitim gorsellestirmesi, CLI argumanlari ve detayli yorumlar iceren calistirmaya hazir ornekler icerir. Detaylar icin [openai_gym/README.md](openai_gym/README.md) dosyasina bakin.

## Kaynaklar

### Kitaplar ve Kurslar
- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [David Silver RL Course](https://www.davidsilver.uk/teaching/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

### Awesome Listeler ve Tutorial'lar
- [awesome-reinforcement-learning](https://github.com/awesomelistsio/awesome-reinforcement-learning) - Kapsamli RL framework, kutuphane, arac ve tutorial listesi
- [awesome-deep-rl](https://github.com/kengz/awesome-deep-rl) - Deep Reinforcement Learning kaynaklari
- [awesome-rl](https://github.com/aikorea/awesome-rl) - Derlenmis RL kaynaklari
- [Curated-Reinforcement-Learning-Resources](https://github.com/azminewasi/Curated-Reinforcement-Learning-Resources) - Cesitli platformlardan kurslar ve tutorial'lar
- [reinforcement-learning-resources](https://github.com/datascienceid/reinforcement-learning-resources) - Video dersler, kitaplar ve kutuphaneler
- [dennybritz/reinforcement-learning](https://github.com/dennybritz/reinforcement-learning) - Python, OpenAI Gym, TensorFlow ile RL implementasyonlari (Sutton'in kitabi ve David Silver'in kursu icin)
- [awesome-machine-learning-robotics](https://github.com/Phylliade/awesome-machine-learning-robotics) - Robotik icin Makine Ogrenimi kaynaklari

### Framework'ler ve Kutuphaneler
- [OpenAI Gym](https://gym.openai.com/) - Agent egitimi icin simulasyon ortamlari
- [OpenAI Baselines](https://github.com/openai/baselines) - Deep RL algoritmalarinin uzman implementasyonlari
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - PyTorch ile guvenilir RL algoritma implementasyonlari
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) - Olceklenebilir Reinforcement Learning kutuphanesi
- [Keras-RL](https://github.com/matthiasplappert/keras-rl) - Keras uyumlu framework (DQN, SARSA, DDPG)
- [DeepMind Acme](https://github.com/deepmind/acme) - DeepMind'in RL arastirma framework'u
- [DeepMind DQN](https://github.com/deepmind/dqn) - Nature makalesindeki resmi DQN implementasyonu

### Onemli Makaleler - Deep Reinforcement Learning
- [Human-level control through deep reinforcement learning](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) - Mnih et Al. (DQN)
- [Continuous control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) - Lillicrap et Al. (DDPG)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) - Schaul et Al.
- [Reinforcement learning with unsupervised auxiliary tasks](https://deepmind.com/blog/reinforcement-learning-unsupervised-auxiliary-tasks/) - Jaderberg et Al.
- [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286) - Heess et Al.
- [Deep RL that matters](https://arxiv.org/abs/1709.06560) - Henderson et Al. (Tekrarlanabilirlik)

### Onemli Makaleler - Policy Gradient ve Teori
- [Simple statistical gradient-following algorithms for connectionist RL](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) - Williams (REINFORCE)
- [Policy Gradient Methods for RL with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) - Sutton et Al.
- [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf) - Silver et Al.
- [Reinforcement learning of motor skills with policy gradients](http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Neural-Netw-2008-21-682_4867%5b0%5d.pdf) - Peters and Schaal
- [Guided Policy Search](https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf) - Levine et Al.

## Lisans

MIT License
