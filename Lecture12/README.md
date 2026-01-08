# Lecture 12

## İçerik
Bu klasör, Reinforcement Learning (Pekiştirmeli Öğrenme) ve Q-Learning temellerini kapsar.

## Dosyalar
- **Lecture12.pdf**: Dersin teorik sunumu ve ders notları
- **lecture12.py**: Q-Learning algoritması simülasyonu

## Konu Başlıkları

---

### Reinforcement Learning Nedir?

**Temel Kavramlar:**
- **Agent**: Karar veren varlık (ajan)
- **Environment**: Agent'in içinde bulunduğu ortam
- **State (s)**: Ortamın mevcut durumu
- **Action (a)**: Agent'in yapabileceği hareket
- **Reward (r)**: Action'dan sonra alınan ödül/ceza
- **Policy (π)**: Hangi state'te hangi action'ı seç (strateji)

**Amaç:** Toplam ödülü maksimize eden policy bul

**Örnek Uygulamalar:**
- Oyun oynama (Chess, Go, Atari)
- Robotik (yürüme, tutma)
- Otonom araçlar
- Kaynak yönetimi

---

### lecture12.py - Q-Learning Simülasyonu

Bu script, basit bir zincir state sistemi için Q-Learning algoritmasını simüle eder.

#### Problem Yapısı

**4 State Zinciri:**
```
State 0 → State 1 → State 2 → State 3 (Terminal)
r=-2.0    r=4.0      r=1.0
```

- State 0'dan State 1'e geçiş: Reward = -2.0
- State 1'den State 2'ye geçiş: Reward = 4.0
- State 2'den State 3'e geçiş: Reward = 1.0
- State 3: Terminal state (bitiş)

#### Parametreler

```python
gamma = 0.5        # Discount factor (gelecek ödüllerinin değeri)
alpha = 0.3        # Learning rate (öğrenme hızı)
n = 4              # State sayısı
r_list = [-2.0, 4.0, 1.0]  # Reward'lar
epochs = 25        # Kaç iterasyon
q_original = [1, 2, 8]  # Başlangıç Q-değerleri
```

**Parametrelerin Anlamları:**

##### 1. Gamma (γ) - Discount Factor
```python
gamma = 0.5
```

**Açıklama:**
- Gelecekteki ödüllerin bugünkü değerini belirler
- γ ∈ [0, 1]
- **γ = 0**: Sadece şimdiki reward'a bak (miyop)
- **γ = 1**: Gelecekteki tüm reward'lar eşit önemli
- **γ = 0.5**: Gelecek %50 değer taşır

**Örnek:**
```
State 0: Hemen -2 reward AL, sonra State 1'e git
State 1: Hemen 4 reward AL, sonra State 2'ye git
State 2: Hemen 1 reward AL, terminal

State 0'dan başlayarak toplam beklenen reward:
Total = -2 + γ×4 + γ²×1
      = -2 + 0.5×4 + 0.25×1
      = -2 + 2 + 0.25
      = 0.25
```

##### 2. Alpha (α) - Learning Rate
```python
alpha = 0.3
```

**Açıklama:**
- Q-değerinin ne kadar hızlı güncelleneceği
- α ∈ [0, 1]
- **α = 0**: Hiç öğrenme (Q değişmez)
- **α = 1**: Tam öğrenme (eski bilgiyi unut)
- **α = 0.3**: Yavaş ve kararlı öğrenme

**Q-Update Formülü:**
```
Q_new(s) = Q_old(s) + α × [r + γ×Q(s') - Q_old(s)]
          = Q_old(s) + α × TD_error
```

**Örnek:**
```
Q_old = 5
r = 10, γ = 0.5, Q(s') = 8
TD_error = 10 + 0.5×8 - 5 = 10 + 4 - 5 = 9

α = 0.3:
Q_new = 5 + 0.3×9 = 5 + 2.7 = 7.7

α = 1.0:
Q_new = 5 + 1.0×9 = 5 + 9 = 14
```

#### True Q-Values Hesaplama

```python
true_q = np.zeros(n-1)  # [0, 0, 0] başlangıç
cur = 0

for j in range(len(true_q)-1, -1, -1):  # Geriye doğru: 2, 1, 0
    true_q[j] = r_list[j] + gamma * cur
    cur = true_q[j]
```

**Adım Adım Hesaplama:**

**Iteration 1 (j=2, State 2):**
```python
cur = 0  # Terminal state'ten sonra reward yok
true_q[2] = r_list[2] + gamma × cur
          = 1.0 + 0.5 × 0
          = 1.0
cur = 1.0
```

**Iteration 2 (j=1, State 1):**
```python
cur = 1.0
true_q[1] = r_list[1] + gamma × cur
          = 4.0 + 0.5 × 1.0
          = 4.0 + 0.5
          = 4.5
cur = 4.5
```

**Iteration 3 (j=0, State 0):**
```python
cur = 4.5
true_q[0] = r_list[0] + gamma × cur
          = -2.0 + 0.5 × 4.5
          = -2.0 + 2.25
          = 0.25
cur = 0.25
```

**Final true_q:**
```python
true_q = [0.25, 4.5, 1.0]
```

**Yorumlama:**
- **State 0**: 0.25 beklenen toplam reward
- **State 1**: 4.5 beklenen toplam reward (en iyi!)
- **State 2**: 1.0 beklenen toplam reward
- **State 3**: 0.0 (terminal, daha ödül yok)

Bu, **optimal policy** altında her state'ten elde edilecek toplam reward'dur.

#### Q-Table Başlatma

```python
q_table = np.zeros([epochs, n])  # (25, 4) matris

for j in range(n-1):
    q_table[0, j] = q_original[j]  # İlk satır: [1, 2, 8, 0]
```

**Q-Table Yapısı:**
```
         State0  State1  State2  State3
Epoch 0:   1       2       8       0
Epoch 1:   ?       ?       ?       0
Epoch 2:   ?       ?       ?       0
...
Epoch 24:  ?       ?       ?       0
```

- Her satır bir epoch
- Her sütun bir state'in Q-değeri
- State 3 (terminal) her zaman 0

#### Q-Learning Update Loop

```python
for x0 in range(1, epochs):  # Epoch 1'den 24'e kadar
    for x1 in range(n-1):    # State 0, 1, 2 için
        # TD (Temporal Difference) Error hesapla
        learned = r_list[x1] + gamma * q_table[x0-1, x1+1] - q_table[x0-1, x1]
        
        # Q-value güncelle
        q_table[x0, x1] = q_table[x0-1, x1] + alpha * learned
```

**TD Update Formülü:**
```
Q(s_t) ← Q(s_t) + α × [r_t + γ×Q(s_{t+1}) - Q(s_t)]
                      └─────────────────────────┘
                           TD Error (learned)
```

**Bileşenler:**
- **r_t**: Şimdiki reward
- **γ×Q(s_{t+1})**: Discounted gelecek değer
- **Q(s_t)**: Şimdiki tahmin
- **TD Error**: Ne kadar yanıldık?

#### Örnek Hesaplama (Epoch 1, State 0)

**Başlangıç:**
```python
x0 = 1, x1 = 0
q_table[0] = [1, 2, 8, 0]  # Önceki epoch
r_list[0] = -2.0
gamma = 0.5
alpha = 0.3
```

**Step 1: TD Error**
```python
learned = r_list[0] + gamma × q_table[0, 1] - q_table[0, 0]
        = -2.0 + 0.5 × 2 - 1
        = -2.0 + 1.0 - 1
        = -2.0
```

**Step 2: Q Update**
```python
q_table[1, 0] = q_table[0, 0] + alpha × learned
              = 1 + 0.3 × (-2.0)
              = 1 - 0.6
              = 0.4
```

**Yeni Q-Table:**
```
         State0  State1  State2  State3
Epoch 0:   1       2       8       0
Epoch 1:  0.4      ?       ?       0
```

#### Epoch 1 Tüm State'ler

**State 1:**
```python
learned = 4.0 + 0.5 × 8 - 2 = 4.0 + 4.0 - 2 = 6.0
q_table[1, 1] = 2 + 0.3 × 6.0 = 2 + 1.8 = 3.8
```

**State 2:**
```python
learned = 1.0 + 0.5 × 0 - 8 = 1.0 - 8 = -7.0
q_table[1, 2] = 8 + 0.3 × (-7.0) = 8 - 2.1 = 5.9
```

**Epoch 1 Sonuç:**
```
q_table[1] = [0.4, 3.8, 5.9, 0]
```

#### Convergence (Yakınsama)

25 epoch boyunca bu işlem tekrar eder:

**İlk Epochlar:**
```
Epoch 0:  [1.0,  2.0,  8.0,  0]
Epoch 1:  [0.4,  3.8,  5.9,  0]
Epoch 2:  [0.1,  4.2,  4.7,  0]
Epoch 3:  [0.1,  4.4,  3.7,  0]
...
```

**Son Epochlar:**
```
Epoch 22: [0.25, 4.49, 1.01, 0]
Epoch 23: [0.25, 4.50, 1.00, 0]
Epoch 24: [0.25, 4.50, 1.00, 0]
```

**true_q = [0.25, 4.5, 1.0]** → **Yakınsama başarılı!**

#### Görselleştirme

```python
fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=200)
colors = ['#2CA02c', '#FF7F0E', '#D62778']  # Yeşil, Turuncu, Pembe
markers = ['o', 'd', '^']

for j in range(n-1):
    # Q-değerlerinin evrimini çiz
    ax.plot(np.arange(epochs), q_table[:, j], 
            marker=markers[j], markersize=5, alpha=0.7,
            color=colors[j], linestyle='-', label=f'$Q$ s{j+1}')
    
    # True Q-değerini yatay çizgi olarak göster
    ax.axhline(y=true_q[j], color=colors[j], linestyle='--')

ax.set_ylabel('Q-values')
ax.set_xlabel('episode')
ax.set_title(r'$\gamma=$' + f'{gamma}' + r'$\alpha=$' + f'{alpha}')
plt.legend(loc='best')
plt.show()
```

**Grafik Yorumu:**
- **3 renkli çizgi**: State 0, 1, 2'nin Q-değerleri
- **Kesik çizgiler**: True Q-değerleri (hedef)
- **Y-ekseni**: Q-value
- **X-ekseni**: Epoch (zaman)

**Gözlemler:**
- State 0: 1 → 0.25 (aşağı iner)
- State 1: 2 → 4.5 (yukarı çıkar)
- State 2: 8 → 1.0 (keskin düşüş)
- Yaklaşık epoch 15'ten sonra stabil

## Gerekli Kütüphaneler

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
```

## Q-Learning Teorisi

### Bellman Equation

**Optimal Q-value:**
```
Q*(s, a) = E[r + γ × max Q*(s', a')]
                        a'
```

**Açıklama:**
- Q*(s, a): State s'te action a'yı seçmenin optimal değeri
- r: Immediate reward
- γ: Discount factor
- max Q*(s', a'): Sonraki state'te en iyi action'ın değeri
- E[...]: Beklenen değer (stochastic environment'ta)

### Q-Learning Update Rule

```
Q(s_t, a_t) ← Q(s_t, a_t) + α × [r_t + γ×max Q(s_{t+1}, a') - Q(s_t, a_t)]
                                              a'
```

**Bu kodda:**
- Tek action var (ileri git)
- max gerekmiyor
- Q(s_t) = r_t + γ×Q(s_{t+1})

### Exploration vs Exploitation

**ε-greedy Policy:**
```python
if random.random() < epsilon:
    action = random_action()  # Explore
else:
    action = best_action()     # Exploit
```

**Bu kodda kullanılmamış** (deterministik ortam)

### Convergence Koşulları

Q-Learning yakınsama garantisi için:
1. Her state-action çifti sonsuz kez ziyaret edilmeli
2. Learning rate zamanla azalmalı: Σα = ∞, Σα² < ∞
3. Reward'lar bounded olmalı

## Parametrelerin Etkisi

### Gamma (γ)

| Değer | Davranış | Kullanım |
|-------|----------|----------|
| γ = 0 | Sadece immediate reward | Kısa vadeli görevler |
| γ = 0.5 | Orta vade | Bu kod |
| γ = 0.9 | Uzun vade | Satranç, Go |
| γ = 0.99 | Çok uzun vade | Robotik |

**Düşük γ:** Miyop, gelecek önemli değil
**Yüksek γ:** Sabırlı, gelecek çok önemli

### Alpha (α)

| Değer | Davranış | Kullanım |
|-------|----------|----------|
| α = 0.1 | Yavaş öğrenme | Kararlı ortam |
| α = 0.3 | Orta hız | Bu kod |
| α = 0.5 | Hızlı öğrenme | Dinamik ortam |
| α = 1.0 | Anında öğrenme | Tek iterasyon |

**Düşük α:** Kararlı ama yavaş
**Yüksek α:** Hızlı ama titrek

## Gerçek Dünya Q-Learning

### Deep Q-Network (DQN)
```python
# Q-table yerine neural network
Q(s, a) ≈ Neural_Network(s, a; θ)
```

**Avantajlar:**
- Büyük state space'lerde çalışır
- Continuous state'leri işler
- Generalization yapar

### Experience Replay
```python
# Geçmiş deneyimleri sakla
memory = []
memory.append((s, a, r, s'))

# Rastgele batch ile öğren
batch = random.sample(memory, batch_size)
```

### Target Network
```python
# İki network: Online ve Target
Q_online(s, a)  # Sürekli güncelle
Q_target(s', a')  # Periyodik kopyala
```

## Reinforcement Learning Türleri

### Model-Free (Bu kod)
- Environment model bilinmez
- Deneme-yanılma ile öğren
- Q-Learning, SARSA

### Model-Based
- Environment model bilinir/öğrenilir
- Planlama yapılır
- Dynamic Programming

### On-Policy vs Off-Policy
- **On-Policy (SARSA)**: Kullandığın policy'den öğren
- **Off-Policy (Q-Learning)**: Optimal policy'den öğren (bu kod)

## Notlar

- Q-Learning, off-policy bir algoritmadır (optimal'i öğrenir)
- Convergence yavaştır, DQN daha hızlı
- Bu kod, deterministic environment için basitleştirilmiştir
- Gerçek RL'de exploration (ε-greedy) kritiktir
- True Q-values, Bellman equation'dan analitik hesaplanmıştır
- α ve γ hyperparametrelerdir, tuning gerekir
- 25 epoch, bu basit problem için yeterli ama karmaşık görevlerde milyonlarca adım gerekir
