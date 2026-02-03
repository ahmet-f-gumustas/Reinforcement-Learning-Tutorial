# OpenAI Gym Ornekleri

Bu klasor, OpenAI Gymnasium kutuphanesi kullanilarak olusturulmus interaktif pekistirmeli ogrenme orneklerini icerir. Her ornek, farkli bir ortam ve algoritma kombinasyonu sunarak RL kavramlarini pratik bir sekilde ogretmeyi amaclar.

## Gereksinimler

```bash
pip install gymnasium numpy torch pygame matplotlib
```

## Ornekler

### 1. CartPole - Q-Learning
**Dosya:** `cartpole_q_learning.py`

| Ozellik | Deger |
|---------|-------|
| Ortam | CartPole-v1 |
| Algoritma | Tabular Q-Learning |
| Durum Uzayi | 4 surekli deger (ayrik kutulara bolunur) |
| Aksiyon Uzayi | 2 ayrik aksiyon (sol/sag) |
| Hedef | Cubugun dengede tutulmasi |

```bash
python cartpole_q_learning.py
```

---

### 2. Mountain Car - Q-Learning
**Dosya:** `mountain_car_q_learning.py`

| Ozellik | Deger |
|---------|-------|
| Ortam | MountainCar-v0 |
| Algoritma | Tabular Q-Learning |
| Durum Uzayi | 2 surekli deger (pozisyon, hiz) |
| Aksiyon Uzayi | 3 ayrik aksiyon (sol/dur/sag) |
| Hedef | Arabanin tepeye ulasmasi |

```bash
python mountain_car_q_learning.py
```

---

### 3. Lunar Lander - DQN
**Dosya:** `lunar_lander_game.py`

| Ozellik | Deger |
|---------|-------|
| Ortam | LunarLander-v3 |
| Algoritma | Deep Q-Network (DQN) |
| Durum Uzayi | 8 surekli deger |
| Aksiyon Uzayi | 4 ayrik aksiyon |
| Hedef | Uzay aracinin guvenli inisi |

**Ozellikler:**
- Insan modu (klavye kontrolu)
- AI modu (egitilmis ajan)
- Egitim modu
- Demo modu

```bash
python lunar_lander_game.py           # Interaktif menu
python lunar_lander_game.py --train   # Egitim
python lunar_lander_game.py --watch   # AI izle
```

---

### 4. Bipedal Walker - TD3
**Dosya:** `bipedal_walker_game.py`

| Ozellik | Deger |
|---------|-------|
| Ortam | BipedalWalker-v3 |
| Algoritma | TD3 (Twin Delayed DDPG) |
| Durum Uzayi | 24 surekli deger |
| Aksiyon Uzayi | 4 surekli deger (motor torklari) |
| Hedef | Iki bacakli robotun yurumeyi ogrenmesi |

**Ozellikler:**
- Guzel terminal ciktisi (ilerleme cubugu, renkli metrikler)
- Trend gostergeleri (↑↓→)
- Periyodik ozet raporlari
- CLI argumanlari

```bash
python bipedal_walker_game.py              # Interaktif menu
python bipedal_walker_game.py --train -n 1000  # 1000 episode egit
python bipedal_walker_game.py --watch -n 5     # 5 episode izle
python bipedal_walker_game.py --demo           # Hizli demo
python bipedal_walker_game.py --human          # Klavye ile oyna
```

---

### 5. Acrobot - DQN
**Dosya:** `acrobot_dqn_game.py`

| Ozellik | Deger |
|---------|-------|
| Ortam | Acrobot-v1 |
| Algoritma | Deep Q-Network (DQN) |
| Durum Uzayi | 6 surekli deger (acilar ve hizlar) |
| Aksiyon Uzayi | 3 ayrik aksiyon (negatif/sifir/pozitif tork) |
| Hedef | Robot kolun uc noktasini yukari sallama |

**Ozellikler:**
- Detayli Turkce/Ingilizce yorum satirlari
- Guzel terminal ciktisi (TrainingLogger)
- Ilerleme cubugu ve trend gostergeleri
- CLI argumanlari
- 4 farkli mod (insan/AI/egitim/demo)

```bash
python acrobot_dqn_game.py              # Interaktif menu
python acrobot_dqn_game.py --train -n 500  # 500 episode egit
python acrobot_dqn_game.py --watch -n 5    # 5 episode izle
python acrobot_dqn_game.py --demo          # Hizli demo
python acrobot_dqn_game.py --human         # Klavye ile oyna
```

---

## Algoritma Karsilastirmasi

| Algoritma | Tip | Aksiyon Uzayi | Kullanim Alani |
|-----------|-----|---------------|----------------|
| Q-Learning | Tabular | Ayrik | Kucuk durum uzaylari |
| DQN | Deep Learning | Ayrik | Buyuk/surekli durum uzaylari |
| TD3 | Deep Learning | Surekli | Robot kontrolu, surekli aksiyonlar |

## Klasor Yapisi

```
openai_gym/
├── README.md                    # Bu dosya
├── main.py                      # Ana giris noktasi
├── cartpole_q_learning.py       # CartPole Q-Learning ornegi
├── mountain_car_q_learning.py   # Mountain Car Q-Learning ornegi
├── lunar_lander_game.py         # Lunar Lander DQN ornegi
├── bipedal_walker_game.py       # Bipedal Walker TD3 ornegi
├── acrobot_dqn_game.py          # Acrobot DQN ornegi
└── output/                      # Egitim ciktilari
    ├── lunar_lander_game/
    │   └── dqn_model.pt
    ├── bipedal_walker_game/
    │   └── td3_model.pt
    └── acrobot_dqn_game/
        └── dqn_model.pt
```

## Ortak CLI Argumanlari

Tum interaktif ornekler (lunar_lander, bipedal_walker, acrobot) ayni CLI yapisini kullanir:

| Arguman | Kisa | Aciklama |
|---------|------|----------|
| `--watch` | `-w` | Egitilmis AI'i izle |
| `--train` | `-t` | AI'i egit |
| `--human` | `-H` | Klavye ile oyna |
| `--demo` | `-d` | Hizli egitim + AI gosterisi |
| `--episodes` | `-n` | Episode sayisi |
| `--model` | | Ozel model dosyasi yolu |

## Egitim Ciktisi Ornegi

Bipedal Walker ve Acrobot ornekleri guzel formatli egitim ciktisi saglar:

```
================================================================================
  TRAINING STARTED
================================================================================
  Total Episodes  : 500
  Device          : cuda
  Output          : output/bipedal_walker_game
================================================================================

 Episode |   Progress    |   Score  |  Avg(100)  |    Best |  Steps | Time
---------------------------------------------------------------------------
       1 | ████░░░░░░░░   0.2% |   -89.2 |    -89.2 ↑ |   -89.2 |    234 |  1.2s
       2 | ████░░░░░░░░   0.4% |  -102.5 |    -95.9 → |   -89.2 |    312 |  1.5s
       ...
```

## Ipuclari

1. **GPU Kullanimi:** CUDA destekli GPU varsa egitim cok daha hizli olur
2. **Episode Sayisi:** Daha fazla episode genellikle daha iyi sonuc verir
3. **Model Kaydi:** Modeller otomatik olarak `output/` klasorune kaydedilir
4. **Devam Ettirme:** Egitimi durdurup `--train` ile devam edebilirsiniz

## Kaynaklar

- [Gymnasium Dokumantasyonu](https://gymnasium.farama.org/)
- [PyTorch Dokumantasyonu](https://pytorch.org/docs/)
- [DQN Paper](https://www.nature.com/articles/nature14236)
- [TD3 Paper](https://arxiv.org/abs/1802.09477)
