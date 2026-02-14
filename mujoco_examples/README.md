# MuJoCo Reinforcement Learning Examples

Bu klasör, MuJoCo fizik simülatörü kullanarak gerçek dünya robotları üzerinde modern deep RL algoritmalarını gösterir.

## İçindekiler

1. [Kurulum](#kurulum)
2. [Projeler](#projeler)
3. [Hızlı Başlangıç](#hızlı-başlangıç)
4. [Detaylı Kullanım](#detaylı-kullanım)

---

## Kurulum

### Gereksinimler

```bash
# Ana gereksinimler
pip install gymnasium[mujoco]
pip install stable-baselines3[extra]
pip install matplotlib seaborn pandas scipy

# Opsiyonel (görselleştirme için)
pip install tensorboard
```

### MuJoCo Kurulumu

MuJoCo, Gymnasium ile otomatik olarak kurulur. Ek bir şey yapmanıza gerek yok!

---

## Projeler

### 1. `01_basic_mujoco.py` - Temel MuJoCo Kullanımı

**Amaç:** MuJoCo ortamlarının temel kullanımını öğrenmek.

**Özellikler:**
- Observation ve action space'leri anlama
- Random policy ile simülasyon
- Video kaydetme
- Farklı robotları deneme

**Kullanım:**

```bash
# Basit simülasyon
python 01_basic_mujoco.py

# Görselleştirme ile
python 01_basic_mujoco.py --render

# Farklı robot
python 01_basic_mujoco.py --env Ant-v5

# Video kaydet
python 01_basic_mujoco.py --record --episodes 3
```

**Desteklenen Robotlar:**
- `HalfCheetah-v5` - 6 eklemli koşan robot
- `Ant-v5` - 4 bacaklı karınca robot
- `Hopper-v5` - Tek bacaklı zıplayan robot
- `Walker2d-v5` - 2 bacaklı yürüyen robot
- `Humanoid-v5` - 17 eklemli insansı robot
- `Swimmer-v5` - Yüzen yılan robot

---

### 2. `02_ppo_mujoco_training.py` - PPO ile Robot Eğitimi

**Amaç:** Proximal Policy Optimization (PPO) algoritması ile robot eğitmek.

**Özellikler:**
- Complete PPO implementasyonu
- Training curve visualization
- Model checkpointing
- Random vs Trained agent karşılaştırması
- Detaylı logging

**Kullanım:**

```bash
# Robot eğit (HalfCheetah, 100K timesteps)
python 02_ppo_mujoco_training.py

# Farklı robot eğit
python 02_ppo_mujoco_training.py --env Hopper-v5

# Daha uzun eğitim
python 02_ppo_mujoco_training.py --timesteps 500000

# Eğitilmiş modeli test et
python 02_ppo_mujoco_training.py --eval --render

# Random vs PPO karşılaştır
python 02_ppo_mujoco_training.py --compare
```

**Çıktılar:**
- Eğitilmiş model: `trained_models/{ENV_NAME}/ppo_model.zip`
- Training curve: `trained_models/{ENV_NAME}/training_curve.png`
- Comparison: `trained_models/{ENV_NAME}/comparison.png`

---

### 3. `03_multi_algorithm_benchmark.py` - Kapsamlı Algoritma Karşılaştırması 🆕

**Amaç:** Modern deep RL algoritmalarını kapsamlı bir şekilde benchmark etmek.

**Özellikler:**
- ✨ Çoklu algoritma desteği (PPO, SAC, TD3, A2C)
- 📊 Detaylı performance comparison
- 📈 Statistical significance testing
- 💾 Comprehensive logging ve checkpointing
- 🎯 Multi-environment benchmarking
- 📉 TensorBoard integration
- 🔬 Advanced visualization

**Desteklenen Algoritmalar:**
- **PPO** (Proximal Policy Optimization) - On-policy, güvenilir
- **SAC** (Soft Actor-Critic) - Off-policy, sample efficient
- **TD3** (Twin Delayed DDPG) - Off-policy, deterministik
- **A2C** (Advantage Actor-Critic) - On-policy, hızlı

#### Kullanım Örnekleri:

**1. Tek algoritma eğitimi:**
```bash
# PPO ile HalfCheetah eğit
python 03_multi_algorithm_benchmark.py --algo ppo --env HalfCheetah-v5 --timesteps 200000

# SAC ile Ant eğit
python 03_multi_algorithm_benchmark.py --algo sac --env Ant-v5 --timesteps 300000
```

**2. Tüm algoritmaları benchmark et:**
```bash
# HalfCheetah üzerinde PPO, SAC, TD3 karşılaştır
python 03_multi_algorithm_benchmark.py --benchmark --env HalfCheetah-v5

# Spesifik algoritmaları seç
python 03_multi_algorithm_benchmark.py --benchmark --env Hopper-v5 --algos ppo sac

# Daha uzun eğitim
python 03_multi_algorithm_benchmark.py --benchmark --env Ant-v5 --timesteps 500000
```

**3. Çoklu ortam benchmark:**
```bash
# 3 farklı ortam üzerinde tüm algoritmaları test et
python 03_multi_algorithm_benchmark.py --multi-env --timesteps 200000

# Bu HalfCheetah-v5, Ant-v5, Hopper-v5 üzerinde test yapar
```

#### Benchmark Sonuçları:

Benchmark tamamlandığında aşağıdaki dosyalar oluşturulur:

```
benchmark_results/
└── YYYYMMDD_HHMMSS/
    ├── ppo/
    │   ├── config.json              # Hyperparameters
    │   ├── results.json             # Final metrics
    │   ├── logs/                    # Training logs
    │   └── models/                  # Checkpoints
    ├── sac/
    │   └── ...
    ├── td3/
    │   └── ...
    ├── learning_curves.png          # Algorithm comparison
    ├── comparison_bar.png           # Performance bars
    └── comparison_table.csv         # Detailed metrics
```

#### Örnek Sonuçlar:

Tipik bir benchmark sonucu (HalfCheetah-v5, 200K timesteps):

| Algorithm | Mean Reward | Std | Training Time |
|-----------|-------------|-----|---------------|
| PPO       | 3245.67     | 234 | 285s          |
| SAC       | 4123.45     | 189 | 312s          |
| TD3       | 4089.23     | 201 | 298s          |
| A2C       | 2876.34     | 312 | 245s          |

**Sonuçlar:**
- SAC ve TD3 en yüksek performansı gösterir (off-policy, experience replay)
- PPO güvenilir ve stabil öğrenir
- A2C daha hızlı ama daha düşük performans

---

## Hızlı Başlangıç

### Yeni Başlayanlar İçin:

```bash
# 1. Basit bir simülasyon çalıştır
python 01_basic_mujoco.py --env Hopper-v5 --episodes 3

# 2. İlk robotunu eğit (kısa süre)
python 02_ppo_mujoco_training.py --env InvertedPendulum-v5 --timesteps 50000

# 3. Eğitilmiş robotu izle
python 02_ppo_mujoco_training.py --env InvertedPendulum-v5 --eval --render
```

### İleri Seviye:

```bash
# Kapsamlı benchmark çalıştır
python 03_multi_algorithm_benchmark.py --benchmark --env Ant-v5 --timesteps 300000

# Çoklu ortam karşılaştırma
python 03_multi_algorithm_benchmark.py --multi-env --timesteps 200000
```

---

## Detaylı Kullanım

### PPO Hyperparameter Tuning

`02_ppo_mujoco_training.py` dosyasında hyperparametreleri değiştirebilirsiniz:

```python
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,      # Öğrenme hızı
    n_steps=2048,            # Steps per rollout
    batch_size=64,           # Mini-batch size
    n_epochs=10,             # Optimization epochs
    gamma=0.99,              # Discount factor
    gae_lambda=0.95,         # GAE lambda
    clip_range=0.2,          # PPO clip range
)
```

### TensorBoard Kullanımı

Benchmark sırasında TensorBoard logları otomatik kaydedilir:

```bash
# TensorBoard'u başlat
tensorboard --logdir=benchmark_results/TIMESTAMP/ppo/logs/tensorboard

# Tarayıcıda açın: http://localhost:6006
```

### Ortam Seçimi Rehberi

| Ortam | Zorluk | Öğrenme Süresi | Önerilen Algoritma |
|-------|--------|----------------|-------------------|
| InvertedPendulum-v5 | Kolay | 5-10 dakika | PPO, A2C |
| HalfCheetah-v5 | Orta | 20-40 dakika | SAC, TD3 |
| Hopper-v5 | Orta | 30-60 dakika | PPO, SAC |
| Walker2d-v5 | Zor | 60-120 dakika | SAC, TD3 |
| Ant-v5 | Zor | 60-120 dakika | SAC, TD3 |
| Humanoid-v5 | Çok Zor | 3-6 saat | SAC |

### Performance Tips

1. **Sample Efficiency**:
   - Off-policy (SAC, TD3): Daha sample efficient
   - On-policy (PPO, A2C): Daha fazla sample gerektirir

2. **Stability**:
   - PPO: En stabil
   - SAC: Genelde stabil
   - TD3: Bazen hassas hyperparameter tuning gerektirir

3. **Speed**:
   - A2C: En hızlı (parallel environments)
   - PPO: Orta hızda
   - SAC/TD3: Replay buffer overhead

4. **Continuous Actions**:
   - Tüm algoritmalar continuous action spaces'i destekler
   - SAC genelde en iyi performansı gösterir

---

## Troubleshooting

### MuJoCo Kurulum Sorunları

```bash
# Eğer "No module named 'mujoco'" hatası alırsanız:
pip install --upgrade gymnasium[mujoco]

# macOS'ta rendering sorunları için:
export MUJOCO_GL=glfw
```

### Training Sorunları

**Problem:** Robot öğrenmiyor
- Learning rate'i düşürün (3e-4 → 1e-4)
- Daha uzun eğitin
- Farklı seed deneyin

**Problem:** Training çok yavaş
- Timesteps azaltın
- Daha basit ortam seçin (InvertedPendulum)
- GPU kullanın (eğer varsa)

**Problem:** Unstable learning
- PPO kullanın (en stabil)
- Batch size artırın
- Gradient clipping ekleyin

---

## Kaynaklar

### Akademik Makaleler

- **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **SAC**: [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
- **TD3**: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
- **MuJoCo**: [MuJoCo: A physics engine for model-based control](https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)

### Online Kaynaklar

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium MuJoCo Docs](https://gymnasium.farama.org/environments/mujoco/)
- [Spinning Up in Deep RL](https://spinningup.openai.com/)

### Video Tutorials

- [PPO Explained](https://www.youtube.com/watch?v=5P7I-xPq8u8)
- [SAC Tutorial](https://www.youtube.com/watch?v=SJG9j1VcP0w)

---

## İleriki Projeler

### Önerilen Uzantılar:

1. **Curriculum Learning**: Basit görevlerden karmaşığa
2. **Multi-Task Learning**: Tek model, birden fazla görev
3. **Transfer Learning**: Bir robottan diğerine bilgi transferi
4. **Domain Randomization**: Gerçek dünya robustluğu
5. **Imitation Learning**: İnsan demonstrasyonlarından öğrenme

---

## Katkıda Bulunma

Bu projeleri geliştirmek için:
- Yeni algoritma ekleyin
- Hyperparameter optimization ekleyin
- Visualization'ları iyileştirin
- Dokümantasyon ekleyin

---

## Lisans

MIT License - Detaylar için ana README'ye bakın.

---

**İyi Çalışmalar! 🤖🚀**

Sorularınız için: [GitHub Issues](https://github.com/your-repo/issues)
