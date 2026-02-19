# Final Proje: Dort Bacakli Robot Simulasyonu
# Final Project: Quadruped Robot Simulation

Bu proje, **SAC (Soft Actor-Critic)** algoritmasi kullanarak dort bacakli robotlarin
MuJoCo fizik simulasyonunda yurumeyi ogrenmesini gosterir. Iki farkli quadruped robot
karsilastirilir: Gymnasium'un hazir **Ant-v5** ortami ve sifirdan tasarlanmis **Custom Quadruped**.

This project demonstrates quadruped robots learning to walk in MuJoCo physics simulation
using the **SAC (Soft Actor-Critic)** algorithm. Two different quadruped robots are compared:
Gymnasium's built-in **Ant-v5** environment and a custom-designed **Custom Quadruped**.

---

## Proje Yapisi / Project Structure

| Dosya / File | Aciklama / Description |
|---|---|
| `01_ant_v5_baseline.py` | Ant-v5 ortaminda SAC baseline egitimi / SAC baseline training on Ant-v5 |
| `02_custom_quadruped_model.xml` | Ozel dort bacakli robot MuJoCo modeli / Custom quadruped MuJoCo XML model |
| `03_custom_quadruped_env.py` | Ozel Gymnasium ortami / Custom Gymnasium environment |
| `04_train_custom_quadruped.py` | Ozel robot SAC egitimi / Custom robot SAC training |
| `05_evaluate_and_compare.py` | Degerlendirme ve karsilastirma / Evaluation and comparison |

---

## SAC (Soft Actor-Critic) Algoritmasi Hakkinda

SAC, surekli aksiyon uzaylari icin tasarlanmis bir **off-policy** pekistirmeli ogrenme
algoritmasidir. Temel ozellikleri:

- **Entropy Maksimizasyonu**: Sadece odulu degil, ayni zamanda politikanin entropy'sini de
  maksimize eder. Bu, ajanin daha cesitli davranislar kesfetmesini saglar.
- **Replay Buffer**: Gecmis deneyimleri saklar ve tekrar kullanir (sample efficient).
- **Twin Q-Networks**: Iki ayri Q-network kullanarak overestimation bias'i azaltir.
- **Otomatik Entropy Ayarlama**: `ent_coef="auto"` ile entropy katsayisi otomatik ayarlanir.

### SAC vs PPO

| Ozellik | SAC | PPO |
|---|---|---|
| Tip | Off-policy | On-policy |
| Sample Efficiency | Yuksek | Dusuk |
| Replay Buffer | Var | Yok |
| Entropy | Otomatik ayarlama | Manuel katsayi |
| Surekli Aksiyon | Ideal | Iyi |

---

## Robot Modelleri / Robot Models

### Ant-v5 (Baseline)
- Gymnasium'un hazir dort bacakli karinca robotu
- Kuresel govde, yatay yayilan bacaklar
- 8 actuator (her bacakta 2 eklem)
- 105 boyutlu gozlem uzayi (contact force dahil)

### Custom Quadruped (Ozel Tasarim)
- Kopek benzeri dort bacakli robot
- Dikdortgen govde (0.4m x 0.2m x 0.1m)
- Bacaklar asagi dogru uzanir (gercekci anatomik yapi)
- 8 actuator (her bacakta hip + knee)
- 27 boyutlu gozlem uzayi

### Temel Farklar

| Ozellik | Ant-v5 | Custom Quadruped |
|---|---|---|
| Govde | Kure | Dikdortgen kutu |
| Bacak yonu | Yatay | Dikey (asagi) |
| Eklem sayisi | 8 | 8 |
| Obs. boyutu | 105 | 27 |
| Saglik z-araligi | [0.2, 1.0] | [0.15, 0.6] |

---

## Odul Fonksiyonu / Reward Function

Her iki ortam da ayni odul yapisini kullanir:

```
reward = forward_reward + healthy_reward - ctrl_cost
```

- **forward_reward**: `x_velocity * 1.0` (ileri hareket odulu)
- **healthy_reward**: `1.0` (robot ayakta oldugu surece)
- **ctrl_cost**: `0.5 * sum(action^2)` (enerji cezasi)

---

## Kullanim / Usage

### 1. Ant-v5 Baseline Egitimi

```bash
# Egitim baslat (300k timestep, ~10-15 dk)
python 01_ant_v5_baseline.py

# Daha uzun egitim
python 01_ant_v5_baseline.py --timesteps 500000

# Egitilmis ajani izle
python 01_ant_v5_baseline.py --eval --render

# Random vs SAC karsilastir
python 01_ant_v5_baseline.py --compare
```

### 2. Custom Quadruped Egitimi

```bash
# Egitim baslat
python 04_train_custom_quadruped.py

# Egitilmis ajani izle
python 04_train_custom_quadruped.py --eval --render

# Random vs SAC karsilastir
python 04_train_custom_quadruped.py --compare
```

### 3. Karsilastirma

```bash
# Her iki modeli karsilastir (her iki egitim tamamlanmis olmali)
python 05_evaluate_and_compare.py

# Daha fazla episode ile
python 05_evaluate_and_compare.py --episodes 20

# Gorsel izleme
python 05_evaluate_and_compare.py --render
```

---

## SAC Hiperparametreleri / SAC Hyperparameters

| Parametre | Deger | Aciklama |
|---|---|---|
| `learning_rate` | 3e-4 | Ogrenme orani |
| `buffer_size` | 1,000,000 | Replay buffer boyutu |
| `learning_starts` | 10,000 | Ilk random exploration |
| `batch_size` | 256 | Mini-batch boyutu |
| `tau` | 0.005 | Soft update katsayisi |
| `gamma` | 0.99 | Discount factor |
| `ent_coef` | auto | Entropy katsayisi (otomatik) |
| `target_entropy` | auto | Hedef entropy (otomatik) |

---

## Gereksinimler / Requirements

```
gymnasium[mujoco]
stable-baselines3
torch
matplotlib
numpy
```

Kurulum / Installation:
```bash
pip install gymnasium[mujoco] stable-baselines3 torch matplotlib
```

---

## TensorBoard ile Izleme

Egitim sirasinda TensorBoard loglari otomatik kaydedilir:

```bash
tensorboard --logdir final_project/trained_models/ant_v5/tb_logs
tensorboard --logdir final_project/trained_models/custom_quadruped/tb_logs
```
