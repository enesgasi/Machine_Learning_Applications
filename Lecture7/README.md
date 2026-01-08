# Lecture 7

## İçerik
Bu klasör, Neural Networks (Yapay Sinir Ağları) ve Deep Learning uygulamalarına odaklanır.

## Dosyalar
- **Lecture7.pdf**: Dersin teorik sunumu ve ders notları
- **application1.py**: Basit bir nöron simülasyonu
- **application2.py**: Keras ile taksi ücreti tahmini modeli
- **taxi-fares.csv**: New York taksi ücret verileri

## Konu Başlıkları

### 1. Application 1 - Basit Nöron Simülasyonu

#### Amaç
Tek bir nöronun nasıl öğrendiğini göstermek için gradient descent kullanarak weight'leri optimize eden basit bir neural network simülasyonu.

#### Kod Detayları

##### Veri Yapısı
```python
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T
```
- 4 eğitim örneği, her biri 3 input değeri içeriyor
- Her örnek için 1 output değeri var (0 veya 1)

##### Weight (Ağırlık) Başlatma
```python
random.seed(1)
synaptic_weights = 2 * random.random((3, 1)) - 1
```
- 3x1 boyutunda rastgele weight matrisi oluşturulur
- Değerler -1 ile 1 arasında, ortalama 0
- Seed(1) ile tekrarlanabilir sonuçlar sağlanır

##### Eğitim Döngüsü (100 İterasyon)
```python
for iteration in range(100):
    # Forward Propagation - Sigmoid aktivasyon fonksiyonu
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    
    # Backpropagation - Weight güncelleme
    # Gradient = input.T × (error × sigmoid_derivative)
    synaptic_weights += dot(training_set_inputs.T, 
                           (training_set_outputs - output) * output * (1 - output))
```

**Adım Adım:**
1. **Forward Pass**: Input × Weights sonucuna sigmoid fonksiyonu uygula
2. **Error Hesaplama**: Gerçek output - Tahmin edilen output
3. **Sigmoid Derivative**: output × (1 - output) - Gradyan hesabı için
4. **Weight Update**: Learning rate olmadan, doğrudan gradyan eklenir

##### Test ve Tahmin
```python
print(1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))
```
- [1, 0, 0] input değeriyle test yapılır
- Model bu input için tahmin yapar

#### Öğrenme Prensibi
- Model, input ve output arasındaki ilişkiyi öğrenir
- Her iterasyonda weight'ler, hatayı azaltacak şekilde güncellenir
- 100 iterasyon sonunda optimal weight'lere yaklaşır

---

### 2. Application 2 - Taksi Ücreti Tahmini (Deep Learning)

#### Amaç
New York'ta taksi ücretlerini tahmin etmek için Keras kullanarak derin bir neural network oluşturmak.

#### Veri Ön İşleme

##### 1. Veri Yükleme ve Filtreleme
```python
df = pd.read_csv('taxi-fares.csv')
df = df[df['passenger_count'] == 1]  # Sadece tek yolcu
df = df.drop(['key', 'passenger_count'], axis=1)
```

##### 2. Feature Engineering
```python
for i, row in df.iterrows():
    # Tarih-saat parse etme
    dt = datetime.datetime.strptime(row['pickup_datetime'], '%Y-%m-%d %H:%M:%S UTC')
    
    # Yeni özellikler çıkarma
    df.at[i, 'day_of_week'] = dt.weekday()  # 0=Pazartesi, 6=Pazar
    df.at[i, 'pickup_time'] = dt.hour        # 0-23 arası saat
    
    # Mesafe hesaplama (Haversine benzeri basitleştirilmiş)
    x = (row['dropoff_longitude'] - row['pickup_longitude']) * 54.6
    y = (row['dropoff_latitude'] - row['pickup_latitude']) * 69.0
    distance = sqrt(x**2 + y**2)
    df.at[i, 'distance'] = distance
```

**Açıklama:**
- **day_of_week**: Haftanın günü (0-6), hafta sonu/hafta içi farkını öğrenmek için
- **pickup_time**: Gün içi saat, rush hour'ları yakalamak için
- **distance**: Koordinatlardan mil cinsinden mesafe
  - 54.6 ve 69.0: Enlem/boylam derecelerini mile çevirme katsayıları
  - New York bölgesi için yaklaşık değerler

##### 3. Outlier Temizleme
```python
df = df[(df['distance'] > 1.0) & (df['distance'] < 10.0)]
df = df[(df['fare_amount'] > 0.0) & (df['fare_amount'] < 50.0)]
```
- Çok kısa (<1 mil) veya çok uzun (>10 mil) yolculuklar çıkarılır
- Anormal ücretler ($0-$50 dışı) temizlenir

#### Model Mimarisi

```python
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=3))  # Input layer
model.add(Dense(512, activation='relu'))                # Hidden layer
model.add(Dense(1))                                     # Output layer
```

**Katman Detayları:**

1. **Input Layer (Dense - 512 nöron)**
   - `input_dim=3`: 3 feature (gün, saat, mesafe)
   - `activation='relu'`: f(x) = max(0, x) - Negatif değerleri sıfırlar
   - 512 nöron: Karmaşık ilişkileri öğrenmek için yeterli kapasite

2. **Hidden Layer (Dense - 512 nöron)**
   - ReLU aktivasyon
   - İlk katmandan gelen bilgiyi daha da işler
   - Non-linearity ekler

3. **Output Layer (Dense - 1 nöron)**
   - Aktivasyon yok (linear)
   - Tek sayı çıktısı: Tahmin edilen ücret

#### Model Derleme
```python
model.compile(optimizer='adam', loss='mae', metrics=['mae'])
```

**Parametreler:**
- **optimizer='adam'**: Adaptif learning rate ile gradient descent
  - Momentum ve RMSProp'un kombinasyonu
  - Her parametre için ayrı learning rate
- **loss='mae'**: Mean Absolute Error
  - |gerçek - tahmin| ortalaması
  - Outlier'lara MSE'den daha az duyarlı
- **metrics=['mae']**: Validation sırasında MAE'yi göster

#### Model Eğitimi
```python
hist = model.fit(x, y, 
                 validation_split=0.2,  # %20 validation
                 epochs=20,              # 20 epoch
                 batch_size=100)         # 100 örnek/batch
```

**Parametreler Açıklaması:**

- **validation_split=0.2**:
  - Verinin %80'i eğitim, %20'si validation
  - Her epoch'ta her iki set de değerlendirilir
  - Overfitting kontrolü için

- **epochs=20**:
  - Tüm veri seti üzerinden 20 kez geçiş
  - Her epoch'ta ~38,000 örnek işlenir

- **batch_size=100**:
  - Her seferde 100 örnek işlenir
  - Her epoch'ta yaklaşık 380 backpropagation (38,000 / 100)
  - Mini-batch gradient descent

#### Eğitim Süreci Akışı

1. **Epoch başlar** → Veri karıştırılır
2. **Batch işleme**:
   - 100 örnek alınır (forward pass)
   - Loss hesaplanır (MAE)
   - Gradyanlar hesaplanır (backpropagation)
   - Weight'ler güncellenir (Adam optimizer)
3. **Tüm batch'ler tamamlandığında**:
   - Training MAE hesaplanır
   - Validation set'te test edilir
   - Validation MAE hesaplanır
4. **Epoch biter** → Sonraki epoch'a geç

#### Model Performansı

##### Görselleştirme
```python
err = hist.history['mae']          # Training MAE
val_err = hist.history['val_mae']  # Validation MAE

plt.plot(epochs, err, '-', label='Training MAE')
plt.plot(epochs, val_err, ':', label='Validation MAE')
```

**Ne gözlemlenir?**
- İlk epoch'larda MAE hızla düşer
- Training MAE < Validation MAE (normal)
- Eğer validation MAE artarsa → Overfitting

##### R² Score
```python
from sklearn.metrics import r2_score
print(r2_score(y, model.predict(x)))
```
- **R² = 1.0**: Mükemmel tahmin
- **R² = 0.0**: Ortalama kadar iyi
- **R² < 0.0**: Ortalamadan kötü

#### Tahmin Örnekleri

```python
# Cuma, saat 17:00, 2 mil
model.predict(np.array([[4, 17, 2.0]]))  # 4 = Cuma (0=Pazartesi)

# Cumartesi, saat 17:00, 2 mil
model.predict(np.array([[5, 17, 2]]))    # 5 = Cumartesi
```

**Beklenen Davranış:**
- Aynı mesafe, farklı gün/saat → Farklı ücret
- Rush hour (saat 17:00) → Daha yüksek ücret
- Hafta sonu → Farklı pricing pattern

## Gerekli Kütüphaneler

### Application 1
```python
from numpy import exp, array, random, dot
```

### Application 2
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import numpy as np
```

## Önemli Kavramlar

### Neural Network Terimleri
- **Nöron**: Ağırlıklı toplamı aktivasyon fonksiyonundan geçiren birim
- **Weight (Ağırlık)**: Her bağlantının gücü
- **Bias**: Her nöronun threshold'u (bu kodda yok)
- **Activation Function**: Non-linearity ekleyen fonksiyon
- **Forward Propagation**: Input → Output hesaplama
- **Backpropagation**: Gradient'leri geriye yayma

### Optimizasyon
- **Gradient Descent**: Loss'u minimize etmek için weight'leri güncelleme
- **Adam Optimizer**: Adaptif learning rate
- **Learning Rate**: Her adımda weight değişim miktarı
- **Epoch**: Tüm veri setinden bir geçiş
- **Batch**: Aynı anda işlenen örnek sayısı

### Overfitting Önleme
- **Validation Split**: Ayrı test seti
- **Early Stopping**: Val loss artarsa dur (kodda yok)
- **Regularization**: L1/L2 (kodda yok)
- **Dropout**: Rastgele nöronları kapat (kodda yok)

## Sonuçlar ve Gözlemler

- Application1, basit bir XOR benzeri problemi çözer
- Application2, 38,000+ örnekle gerçek dünya tahmini yapar
- Deep network (512-512-1) karmaşık non-linear ilişkileri öğrenir
- MAE metriği, dolar cinsinden ortalama hatayı verir
- Model, gün/saat/mesafe kombinasyonundan ücret tahmin eder
