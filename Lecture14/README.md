# Lecture 14

## İçerik
Bu klasör, Deep Learning ve Neural Networks ile gerçek dünya uygulamalarını kapsar.

## Dosyalar
- **Lecture14.py**: Keras/TensorFlow ile CO2 emisyonu tahmini (Neural Network Regression)
- **Lecture14_2.py**: Speech Recognition + NLP ile isim sınıflandırma (Entegre sistem)
- **carbon.csv**: Araç CO2 emisyon verileri
- **names.csv**: İsim veri seti

## Konu Başlıkları

---

### 1. Lecture14.py - CO2 Emisyonu Tahmini (Neural Network Regression)

Bu script, araçların hacim (volume) ve ağırlık (weight) bilgilerinden CO2 emisyonunu tahmin eder.

#### 1.1. Veri Yükleme

```python
import pandas as pd
import numpy as np

df = pd.read_csv("carbon.csv")
X = df[['Volume', 'Weight']]  # Features
y = df['CO2']                  # Target
```

**Veri Seti:**
- **Volume**: Motor hacmi (cm³)
- **Weight**: Araç ağırlığı (kg)
- **CO2**: CO2 emisyonu (g/km)

**Örnek Veriler:**
```
Volume  Weight  CO2
1000    790     99
1200    1100    120
1600    1252    150
2000    1600    190
```

**İlişki:**
- Büyük motor → Fazla CO2
- Ağır araç → Fazla CO2
- Non-linear ilişki (linear regression yetersiz)

#### 1.2. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42
)
```

**Parametreler:**
- **test_size=0.15**: %15 test, %85 train
  - Küçük veri seti için %10-20 test önerilir
  - Büyük veri seti için %20-30
- **random_state=42**: Tekrarlanabilir split
  - Aynı seed → Aynı train/test

#### 1.3. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**Neden Scaling?**
- Neural network'ler normalize veri ile daha iyi öğrenir
- Volume (1000-2000) ve Weight (700-1600) farklı skalalar
- Gradient descent daha hızlı yakınsar

**StandardScaler (Z-score normalization):**
```
X_scaled = (X - μ) / σ

μ: Ortalama
σ: Standart sapma
```

**Sonuç:** Her feature ortalaması 0, std sapması 1

**fit_transform vs transform:**
- `fit_transform(X_train)`: Mean/std'yi train'den hesapla ve uygula
- `transform(X_test)`: Train'den hesaplanan mean/std'yi test'e uygula
- **Neden?** Data leakage önlemek (test bilgisi train'e sızmasın)

#### 1.4. Model Mimarisi

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(256, activation='relu', input_shape=(2,)))  # Hidden layer 1
model.add(Dense(256, activation='relu'))                    # Hidden layer 2
model.add(Dense(1))                                         # Output layer
```

**Katman Detayları:**

##### Input Layer (ilk Dense)
```python
Dense(256, activation='relu', input_shape=(2,))
```
- **256 neurons**: Karmaşık pattern'leri öğrenmek için
- **activation='relu'**: Rectified Linear Unit
  - f(x) = max(0, x)
  - Negatif → 0, Pozitif → x
  - Non-linearity ekler
- **input_shape=(2,)**: 2 feature (Volume, Weight)

**Neden 256 neuron?**
- Küçük input (2) → Büyük hidden layer karmaşıklık yakalar
- Over-capacity iyi (underfitting'den daha iyi)
- Düşük neuron → Underfitting
- Çok yüksek → Overfitting

##### Hidden Layer 2
```python
Dense(256, activation='relu')
```
- İkinci saklı katman
- Daha derin özellik öğrenme
- İlk katmanın çıktısını işler

**Derin Network Mantığı:**
- Layer 1: Basit özellikler (doğrusal ilişkiler)
- Layer 2: Karmaşık özellikler (etkileşimler)
- Örnek: Volume×Weight etkileşimi

##### Output Layer
```python
Dense(1)
```
- **1 neuron**: Tek sayı tahmini (CO2)
- **No activation**: Linear activation (default)
  - Regression için gerekli
  - Herhangi değer çıkabilir (-∞ to ∞)

**Classification vs Regression:**
- Classification: Sigmoid/Softmax activation
- Regression: Linear activation (no activation)

#### 1.5. Model Derleme

```python
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```

**Optimizer: Adam**
- Adaptive Moment Estimation
- Learning rate'i otomatik ayarlar
- Momentum + RMSProp kombinasyonu
- En popüler optimizer
- Alternatifler: SGD, RMSProp, AdaGrad

**Loss: MSE (Mean Squared Error)**
```
MSE = (1/n) × Σ(y_true - y_pred)²
```
- Regression için standart
- Büyük hatalar daha fazla cezalandırılır (kare)
- Türevi alınabilir (gradient descent için)

**Metric: MAE (Mean Absolute Error)**
```
MAE = (1/n) × Σ|y_true - y_pred|
```
- Loss değil, sadece izleme için
- Dolar/kg/gram cinsinden hata
- Anlaşılması kolay

**MSE vs MAE:**
- MSE: Outlier'lara daha duyarlı
- MAE: Outlier'lara robust
- İkisini birlikte kullan: MSE için optimize et, MAE ile izle

#### 1.6. Model Eğitimi

```python
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=40,
    validation_data=(X_test, y_test)
)
```

**Parametreler:**

**epochs=50:**
- Tüm veri seti üzerinden 50 geçiş
- Her epoch'ta tüm train veri işlenir
- Daha fazla epoch → Daha iyi öğrenme (bir noktaya kadar)
- Çok fazla → Overfitting

**batch_size=40:**
- Her seferde 40 örnek işle
- Gradient'leri 40 örneğin ortalaması
- Mini-batch gradient descent

**Batch Size Etkileri:**
- **Küçük (1-32)**: Gürültülü gradient, yavaş ama generalize eder
- **Orta (32-128)**: Dengeli
- **Büyük (256+)**: Smooth gradient, hızlı ama hafıza yoğun

**validation_data=(X_test, y_test):**
- Her epoch sonunda test setinde değerlendir
- Overfitting'i izle
- Training loss ↓, Validation loss ↑ → Overfitting

**history Objesi:**
```python
history.history = {
    'loss': [0.45, 0.32, 0.28, ...],      # Training MSE
    'mae': [0.67, 0.56, 0.53, ...],       # Training MAE
    'val_loss': [0.48, 0.35, 0.31, ...],  # Validation MSE
    'val_mae': [0.69, 0.59, 0.56, ...]    # Validation MAE
}
```

#### 1.7. Model Değerlendirmesi

##### Training Progress Grafiği
```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()
```

**Yorumlama:**
- **İyi Durum**: İkisi de düşer, aralarında küçük fark
- **Overfitting**: Training düşer, Validation artar
- **Underfitting**: İkisi de yüksek kalır

##### R² Score
```python
from sklearn.metrics import r2_score

predictions = model.predict(X_train)
r2 = r2_score(y_train, predictions)
print("R² Score:", r2)
```

**R² (Coefficient of Determination):**
```
R² = 1 - (SS_res / SS_tot)

SS_res = Σ(y_true - y_pred)²  # Residual sum of squares
SS_tot = Σ(y_true - y_mean)²  # Total sum of squares
```

**Yorumlama:**
- **R² = 1.0**: Mükemmel tahmin
- **R² = 0.8**: Varyansın %80'i açıklanıyor (iyi)
- **R² = 0.5**: Orta
- **R² = 0.0**: Model, ortalama kadar iyi
- **R² < 0.0**: Ortalamadan kötü (çok kötü!)

#### 1.8. Tahmin (Prediction)

```python
new_car = np.array([[1370, 1650]])  # Volume=1370, Weight=1650
new_car_scaled = scaler.transform(new_car)

prediction = model.predict(new_car_scaled)
print("Predicted CO2 emission:", prediction[0][0])
```

**Adım Adım:**

1. **Input Hazırlama:**
   ```python
   [[1370, 1650]]  # 2D array (model 2D bekler)
   ```

2. **Scaling:**
   ```python
   # Train setindeki mean/std ile normalize et
   new_car_scaled = (new_car - mean_train) / std_train
   ```

3. **Prediction:**
   ```python
   # Forward propagation
   # Input → Hidden1 → Hidden2 → Output
   prediction = [[134.7]]  # Örnek sonuç
   ```

4. **Sonuç Çıkarma:**
   ```python
   prediction[0][0]  # 134.7 g/km CO2
   ```

**Yorumlama:**
- 1370 cm³, 1650 kg araç → 134.7 g/km CO2
- Gerçek değerle karşılaştır
- MAE kadar hata payı var

---

### 2. Lecture14_2.py - Speech Recognition + NLP Entegrasyonu

Bu script, mikrofondan isim alır ve cinsiyetini tahmin eder (Lecture 10 + Speech Recognition).

#### 2.1. Feature Extraction (Lecture 10'dan)

```python
def name_features(name, n):
    name = name.lower()
    return {
        f"first_{n}_letters": name[:n]
    }
```

**Değişiklik:**
- Lecture 10'da: Son n harf (`name[-n:]`)
- Lecture 14'te: İlk n harf (`name[:n]`)

**Neden İlk Harfler?**
- Farklı strateji test etmek için
- Bazı isimlerde başlangıç daha belirleyici

#### 2.2. Veri Hazırlama

```python
import pandas as pd
import random

df = pd.read_csv("names.csv")

def label_name(name):
    return "short" if len(name) < 6 else "long"

df["label"] = df["name"].apply(label_name)
```

**Yeni Etiketleme:**
- **"short"**: 6 karakterden kısa
- **"long"**: 6 karakter veya uzun

**Örnek:**
```
Name    Label
John    short  (4 < 6)
David   short  (5 < 6)
Alexander  long  (9 ≥ 6)
```

**Neden Uzunluk?**
- Cinsiyet yerine alternatif bir sınıflandırma
- Daha dengeli veri (erkek/kadın dengesizliği yok)

#### 2.3. Shuffle ve Split

```python
data = list(zip(df["name"], df["label"]))
random.shuffle(data)

split_index = int(len(data) * 0.8)
train_data = data[:split_index]
test_data = data[split_index:]
```

**İyi Pratik:**
- Shuffle: Veri sırasını boz
- Split: %80 train, %20 test

#### 2.4. Farklı N Değerleri için Model Eğitimi

```python
classifiers = {}

for n in [1, 2, 3]:
    # Feature extraction
    train_set = [(name_features(name, n), label) for name, label in train_data]
    test_set = [(name_features(name, n), label) for name, label in test_data]
    
    # Model eğitimi
    classifier = NaiveBayesClassifier.train(train_set)
    
    # Accuracy
    acc = nltk_accuracy(classifier, test_set)
    
    classifiers[n] = classifier
    print(f"Accuracy using first {n} letter(s): {acc:.2f}")
```

**3 Model:**
- **n=1**: İlk 1 harf (J, M, A, ...)
- **n=2**: İlk 2 harf (Jo, Ma, Al, ...)
- **n=3**: İlk 3 harf (Joh, Mar, Ale, ...)

**Beklenen Performans:**
- n=1: Düşük (~50-60%)
- n=2: Orta (~65-75%)
- n=3: Yüksek (~70-80%)

**classifiers Dictionary:**
```python
{
    1: <NaiveBayesClassifier n=1>,
    2: <NaiveBayesClassifier n=2>,
    3: <NaiveBayesClassifier n=3>
}
```

#### 2.5. Speech Recognition

```python
import speech_recognition as sr

use_speech = input("Use microphone? (y/n): ").lower()

if use_speech == "y":
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Say a name...")
            audio = recognizer.listen(source)
        
        spoken_name = recognizer.recognize_google(audio)
        print("Recognized name:", spoken_name)
    
    except Exception as e:
        print("Speech recognition failed, switching to text input.")
        spoken_name = input("Enter a name: ")
else:
    spoken_name = input("Enter a name: ")
```

**sr.Recognizer():**
- Speech recognition engine
- Farklı API'leri destekler (Google, Sphinx, Wit.ai)

**sr.Microphone():**
- Mikrofon stream'i aç
- Context manager (`with`) otomatik kapatır

**recognizer.listen():**
- Ses kaydı başlat
- Sessizlik algıladığında dur
- Audio objesi döner

**recognizer.recognize_google():**
- Google Speech-to-Text API kullan
- İnternet bağlantısı gerekir
- Ücretsiz (limitli)

**Hata Yönetimi:**
- Mikrofon yok → Exception
- Arka plan gürültüsü → Hata
- İnternet yok → Hata
- Fallback: Text input

**Alternatif Recognition Engines:**
```python
# Offline (ücretsiz)
recognizer.recognize_sphinx(audio)

# Online (API key gerekir)
recognizer.recognize_wit(audio, key="...")
recognizer.recognize_azure(audio, key="...")
```

#### 2.6. Sınıflandırma

```python
for n in [1, 2, 3]:
    result = classifiers[n].classify(name_features(spoken_name, n))
    print(f"Prediction using first {n} letter(s): {result}")
```

**Örnek Çıktı:**
```
Say a name...
Recognized name: Alexander

Prediction using first 1 letter(s): long
Prediction using first 2 letter(s): long
Prediction using first 3 letter(s): long
```

**Ensemble Mantığı:**
- 3 model birden tahmin yapar
- Çoğunluk oyu (majority voting) uygulanabilir
- Daha robust tahmin

**Majority Voting (Kodda yok ama eklenebilir):**
```python
predictions = [classifiers[n].classify(name_features(spoken_name, n)) for n in [1, 2, 3]]
final_prediction = max(set(predictions), key=predictions.count)
print(f"Final prediction: {final_prediction}")
```

## Gerekli Kütüphaneler

### Lecture14.py
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
```

**Kurulum:**
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

### Lecture14_2.py
```python
import speech_recognition as sr
import pandas as pd
import random
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
```

**Kurulum:**
```bash
pip install SpeechRecognition pyaudio pandas nltk
```

**PyAudio (Mikrofon için):**
```bash
# Windows
pip install pipwin
pipwin install pyaudio

# Mac
brew install portaudio
pip install pyaudio

# Linux
sudo apt-get install python3-pyaudio
```

## Neural Network Hyperparameters

| Parametre | Bu Kod | Önerilen Aralık | Etki |
|-----------|--------|-----------------|------|
| **Hidden Layers** | 2 | 1-5 | Daha fazla → Karmaşık pattern |
| **Neurons/Layer** | 256 | 32-512 | Daha fazla → Kapasiye |
| **Learning Rate** | Adam default (0.001) | 0.0001-0.01 | Yüksek → Hızlı ama kararsız |
| **Batch Size** | 40 | 16-128 | Büyük → Hızlı ama hafıza |
| **Epochs** | 50 | 10-1000 | Fazla → Overfitting |
| **Activation** | ReLU | ReLU, tanh, sigmoid | ReLU genelde en iyi |

## Overfitting Önleme Teknikleri

### 1. Dropout (Kodda yok)
```python
from tensorflow.keras.layers import Dropout

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # %50 nöronları rastgele kapat
```

### 2. Early Stopping
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.fit(..., callbacks=[early_stop])
```

### 3. Regularization
```python
from tensorflow.keras.regularizers import l2

model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
```

### 4. Batch Normalization
```python
from tensorflow.keras.layers import BatchNormalization

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
```

## Model Kaydetme ve Yükleme

```python
# Kaydetme
model.save('co2_model.h5')

# Yükleme
from tensorflow.keras.models import load_model
model = load_model('co2_model.h5')

# Tahmin
model.predict(new_data_scaled)
```

## Speech Recognition İpuçları

### Gürültü Azaltma
```python
recognizer = sr.Recognizer()
with sr.Microphone() as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)
    audio = recognizer.listen(source)
```

### Timeout
```python
audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
```

### Energy Threshold
```python
recognizer.energy_threshold = 4000  # Varsayılan: 300
```

## Notlar

- Neural network'ler scaling'e çok duyarlıdır
- Feature scaling train ve test'te aynı mean/std kullanmalı
- R² = 0.8+ regression için iyi kabul edilir
- Adam optimizer, çoğu durumda iyi çalışır
- Batch size GPU memory'ye göre ayarlanmalı
- Overfitting, küçük veri setlerinde yaygındır
- Speech recognition, arka plan gürültüsüne duyarlıdır
- Ensemble (birden fazla model), robust tahmin sağlar
- Bu kodlar eğitim amaçlıdır, production'da daha fazla tuning gerekir
