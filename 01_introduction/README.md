# Hafta 1: Pekistirmeli Ogrenmeye Giris

Bu hafta Pekistirmeli Ogrenmenin (Reinforcement Learning - RL) temel kavramlarini ogrenecek ve Gymnasium kutuphanesi ile ilk uygulamalarimizi yapacagiz.

## Icindekiler

1. [Pekistirmeli Ogrenme Nedir?](#pekistirmeli-ogrenme-nedir)
2. [Temel Kavramlar](#temel-kavramlar)
3. [Gymnasium Kutuphanesi](#gymnasium-kutuphanesi)
4. [Ornek Kodlar](#ornek-kodlar)
5. [Alistirmalar](#alistirmalar)

---

## Pekistirmeli Ogrenme Nedir?

Pekistirmeli Ogrenme, bir **ajanin** (agent) bir **ortamla** (environment) etkilesim kurarak, **odul** (reward) sinyallerini maksimize edecek sekilde **davranislari** (actions) ogrenmesini saglayan bir makine ogrenmesi paradigmasidir.

### Diger Ogrenme Turleri ile Karsilastirma

| Tur | Veri | Geri Bildirim | Ornek |
|-----|------|---------------|-------|
| **Supervised Learning** | Etiketli | Dogrudan | Resim siniflandirma |
| **Unsupervised Learning** | Etiketsiz | Yok | Kumeleme |
| **Reinforcement Learning** | Deneyim | Odul sinyali | Oyun oynama |

### Gercek Hayat Ornekleri

- Satranc/Go oynayan yapay zeka (AlphaGo)
- Otonom araclar
- Robot kontrolu
- Oneri sistemleri
- Kaynak yonetimi

---

## Temel Kavramlar

### 1. Agent (Ajan)
Kararlari alan ve eylemleri gerceklestiren varlik. Ornegin: Oyun oynayan AI, robot.

### 2. Environment (Ortam)
Ajanin etkilesim kurdugu dunya. Ornegin: Oyun dunyasi, fiziksel ortam.

### 3. State (Durum)
Ortamin o anki durumunu temsil eden bilgi.

```
s_t = Ortamin t anindaki durumu
```

### 4. Action (Eylem)
Ajanin yapabilecegi hareketler.

```
a_t = Ajanin t aninda sectigi eylem
```

### 5. Reward (Odul)
Ajanin bir eylemi gerceklestirdikten sonra aldigi sayisal geri bildirim.

```
r_t = t aninda alinan odul
```

### 6. Policy (Politika)
Ajanin durumlara gore eylem secme stratejisi.

```
pi(a|s) = s durumunda a eylemini secme olasiligi
```

### 7. Value Function (Deger Fonksiyonu)
Bir durumun veya durum-eylem ciftinin uzun vadeli degerini gosterir.

```
V(s) = s durumundan baslayarak beklenen toplam odul
Q(s,a) = s durumunda a eylemini yaparak beklenen toplam odul
```

### 8. Episode
Baslangic durumundan bitis durumuna kadar gecen surec.

---

## RL Dongusu

```
+-------+     eylem (a_t)      +-----------+
| Agent | ------------------> | Environment|
+-------+                      +-----------+
    ^                               |
    |   durum (s_t+1), odul (r_t)   |
    +-------------------------------+
```

Her adimda:
1. Agent mevcut durumu (s_t) gozlemler
2. Bir eylem (a_t) secer
3. Ortam yeni duruma (s_t+1) gecer
4. Agent bir odul (r_t) alir
5. Agent bu deneyimden ogrenir

---

## Gymnasium Kutuphanesi

Gymnasium (eski adiyla OpenAI Gym), RL algoritmalarini test etmek icin standart ortamlar sunan bir Python kutuphanesidir.

### Kurulum

```bash
pip install gymnasium
```

### Temel Ortamlar

| Ortam | Aciklama | Zorluk |
|-------|----------|--------|
| CartPole-v1 | Cubugu dengede tutma | Kolay |
| MountainCar-v0 | Arabayi tepeye cikarma | Orta |
| LunarLander-v3 | Ay'a inis yapma | Orta |
| Acrobot-v1 | Cubugu sallandirma | Orta |

### Temel API

```python
import gymnasium as gym

# Ortam olusturma
env = gym.make("CartPole-v1")

# Ortami sifirlama
observation, info = env.reset()

# Bir adim atma
action = env.action_space.sample()  # Rastgele eylem
observation, reward, terminated, truncated, info = env.step(action)

# Ortami kapatma
env.close()
```

---

## Ornek Kodlar

Bu klasorde 3 ornek kod bulunmaktadir:

### 1. `01_basic_environment.py`
Gymnasium ortaminin temel kullanimini gosterir.

### 2. `02_random_agent.py`
Rastgele eylem secen basit bir ajan implementasyonu.

### 3. `03_environment_exploration.py`
Farkli ortamlari kesfetmek icin kullanilan kod.

---

## Alistirmalar

### Alistirma 1: Ortami Tanima
`CartPole-v1` ortamini calistirin ve su sorulari cevaplayin:
- Observation space'in boyutu nedir?
- Action space'te kac farkli eylem var?
- Episode ne zaman sona eriyor?

### Alistirma 2: Odul Analizi
`MountainCar-v0` ortaminda 10 episode calistirin ve:
- Her episode'un toplam odulunu kaydedin
- Ortalama odul nedir?
- Neden negatif oduller aliyoruz?

### Alistirma 3: Ortam Karsilastirmasi
En az 3 farkli Gymnasium ortamini deneyin ve karsilastirin:
- Observation space turleri
- Action space turleri
- Odul yapilari

---

## Sonraki Hafta

Hafta 2'de **Markov Karar Surecleri (MDP)** konusunu isleyecegiz. Bu kavram tum RL algoritmalarinin matematiksel temelini olusturur.

---

## Kaynaklar

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Sutton & Barto Chapter 1](http://incompleteideas.net/book/RLbook2020.pdf)
- [David Silver Lecture 1](https://www.youtube.com/watch?v=2pWv7GOvuf0)
