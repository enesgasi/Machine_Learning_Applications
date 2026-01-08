# Lecture 8

## İçerik
Bu klasör, Time Series Analysis (Zaman Serisi Analizi) ve Forecasting (Tahminleme) konularını kapsar.

## Dosyalar
- **Lecture8.pdf**: Dersin teorik sunumu ve ders notları
- **Lecture8_notes.md**: Zaman serisi kavramları ve preprocessing notları
- **lecture8.py**: Zaman serisi preprocessing teknikleri
- **Forecasting.py**: ARMA, ARIMA ve SARIMA modelleri ile IBM hisse senedi tahmini
- **IBM.csv**: IBM hisse senedi kapanış fiyatları veri seti

## Konu Başlıkları

### 1. Time Series Nedir?
Belirli zaman aralıklarında kaydedilen veri noktalarının sırası. Zamansal sıralama önemlidir.

#### Karakteristikler:
- **Trend**: Uzun vadeli yön (örn: şirket satışlarının yıllar içinde artması)
- **Seasonality**: Düzenli aralıklarla tekrarlanan desenler (örn: yazın dondurma satışı artışı)
- **Cyclic Patterns**: Sabit periyodu olmayan döngüler (ekonomik döngüler, salgınlar)
- **Irregular Components**: Rastgele, tahmin edilemeyen varyasyonlar

#### Türleri:
- **Univariate**: Tek değişken (örn: sadece hava sıcaklığı)
- **Multivariate**: Çok değişken (örn: sıcaklık, nem, basınç)
- **Regular**: Tutarlı aralıklar (her saat, her gün)
- **Irregular**: Tutarsız aralıklar

---

### 2. lecture8.py - Time Series Preprocessing

Bu script, zaman serisi verilerini analiz için hazırlamak üzere çeşitli preprocessing tekniklerini gösterir.

#### 2.1. Veri Oluşturma
```python
date_range = pd.date_range(start='/1/2020', periods=100, freq='D')
values = np.random.randn(100)
time_series = pd.DataFrame({'date': date_range, 'value': values})
time_series.set_index('date', inplace=True)
```
**Açıklama:**
- 100 günlük rastgele veri oluşturulur (normal dağılım)
- Tarih, index olarak kullanılır (pandas time series için kritik)

#### 2.2. Missing Values (NaN) Oluşturma
```python
time_series_with_nan = time_series.copy()
time_series_with_nan[::10] = np.nan  # Her 10. değeri NaN yap
```

#### 2.3. Forward Fill (NaN Doldurma)
```python
time_series_filled = time_series_with_nan.fillna(method='ffill')
time_series_filled[:1] = 1  # İlk değeri 1 yap
```
**Açıklama:**
- `fillna(method='ffill')`: Eksik değerleri, önceki geçerli değerle doldur
- Forward Fill: Son bilinen değeri kullan
- Zaman serilerinde mantıklı bir yaklaşım (değer ani değişmiyorsa)

#### 2.4. Outlier Removal (Aykırı Değer Temizleme)
```python
from scipy.stats import zscore
z_scores = zscore(time_series_filled)        # Z-score hesapla
abs_z_scores = np.abs(z_scores)              # Mutlak değer
filtered_entries = (abs_z_scores < 2)        # |z| < 2 olanları tut
time_series_no_outliers = time_series_filled[filtered_entries]
```
**Z-Score Mantığı:**
- Z-score = (değer - ortalama) / standart sapma
- |Z| < 2: Değer, ortalamanın ±2 standart sapması içinde
- |Z| ≥ 2: Outlier olarak kabul edilir ve çıkarılır
- %95 veri korunur (normal dağılımda)

#### 2.5. Moving Average (Smoothing)
```python
moving_avg = time_series_no_outliers.rolling(window=5).mean()
```
**Açıklama:**
- `rolling(window=5)`: 5 değerlik hareketli pencere
- `.mean()`: Her pencerenin ortalamasını al
- Gürültüyü azaltır, trend'i vurgular
- Örnek: [1,2,3,4,5,6] → window=3 → [2, 3, 4, 5]

#### 2.6. Differencing (Fark Alma)
```python
differenced_series = time_series_no_outliers.diff().dropna()
```
**Açıklama:**
- Her değerden bir önceki değeri çıkar: `value[t] - value[t-1]`
- Trend'i kaldırır, stationarity sağlar
- ARIMA modellerinde kritik önişlem
- Örnek: [1, 3, 6, 10] → [2, 3, 4]

#### 2.7. Scaling (Normalizasyon)
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_series = scaler.fit_transform(time_series_no_outliers.values)
```
**Açıklama:**
- Her değeri ortalaması 0, standart sapması 1 olan dağılıma dönüştürür
- Z-score normalization: (x - μ) / σ
- Farklı ölçeklerdeki özellikleri karşılaştırılabilir yapar

---

### 3. Forecasting.py - Time Series Forecasting Models

IBM hisse senedi kapanış fiyatlarını tahmin etmek için 3 farklı model kullanılır.

#### 3.1. Veri Yükleme ve Hazırlama
```python
quotes = pd.read_csv("IBM.csv")
data = quotes[['Date', 'Close']]
data.index = pd.to_datetime(data['Date'], format='%Y-%m-%d')
del data['Date']
```
**Açıklama:**
- Sadece Date ve Close (kapanış fiyatı) sütunları kullanılır
- Date, datetime index'e çevrilir
- Index olarak kullanıldıktan sonra Date sütunu silinir

#### 3.2. Train-Test Split
```python
train = data[data.index < pd.to_datetime("2023-04-10", format='%Y-%m-%d')]
test = data[data.index > pd.to_datetime("2023-04-10", format='%Y-%m-%d')]
```
**Açıklama:**
- 2023-04-10 öncesi: Training set
- 2023-04-10 sonrası: Test set
- Zaman serisinde random split YAPILMAZ (zamansal sıra bozulur)

#### 3.3. Model 1 - ARMA (AutoRegressive Moving Average)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

y = train['Close']
ARMAmodel = SARIMAX(y, order=(1, 0, 1))
ARMAmodel = ARMAmodel.fit()
```

**Order Parametreleri: (p, d, q)**
- **p = 1**: AutoRegressive (AR) order
  - Model, 1 önceki değere bakar
  - y(t) = c + φ₁·y(t-1)
- **d = 0**: Differencing order
  - 0 = Fark alınmaz, veri stationary kabul edilir
- **q = 1**: Moving Average (MA) order
  - Model, 1 önceki hataya bakar
  - ε(t-1) terimini içerir

**Tahmin:**
```python
y_pred = ARMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha=0.05)  # %95 güven aralığı
y_pred_df["Predictions"] = ARMAmodel.predict(start=..., end=...)
y_pred_df.index = test.index
```

**Açıklama:**
- `get_forecast()`: Tahmin objesi döner
- `conf_int(alpha=0.05)`: %95 güven aralığı (alt ve üst sınırlar)
- `predict()`: Nokta tahminleri

**RMSE (Root Mean Square Error):**
```python
arma_rmse = np.sqrt(mean_squared_error(test["Close"].values, y_pred_df["Predictions"]))
```
- √(Σ(gerçek - tahmin)² / n)
- Dolar cinsinden ortalama hata
- Düşük RMSE = İyi model

#### 3.4. Model 2 - ARIMA (AutoRegressive Integrated Moving Average)

```python
from statsmodels.tsa.arima.model import ARIMA

ARIMAmodel = ARIMA(y, order=(1, 0, 1))  # Not: d=0, ama genelde d≥1
ARIMAmodel = ARIMAmodel.fit()
```

**ARIMA vs ARMA:**
- ARIMA, ARMA'nın genişletilmiş halidir
- **d parametresi**: Differencing (entegrasyon) order'ı
- d=0 → ARIMA = ARMA
- d=1 → Bir kez fark alınır (trend kaldırılır)
- d=2 → İki kez fark alınır (nadiren kullanılır)

**Stationary Nedir?**
- İstatistiksel özellikler (ortalama, varyans) zamana göre değişmez
- ARMA, stationary veri gerektirir
- ARIMA, differencing ile non-stationary veriyi stationary yapar

**Örnek:**
- Non-stationary: Hisse senedi fiyatı (trend var)
- Stationary: Günlük fiyat değişimi (trend yok)

#### 3.5. Model 3 - SARIMA (Seasonal ARIMA)

```python
SARIMAXmodel = SARIMAX(y, order=(4, 1, 1), seasonal_order=(1, 2, 3, 6))
SARIMAXmodel = SARIMAXmodel.fit()
```

**Order Parametreleri: (p=4, d=1, q=1)**
- **p=4**: 4 önceki değere bakar (AR)
- **d=1**: 1 kez differencing
- **q=1**: 1 önceki hataya bakar (MA)

**Seasonal Order: (P=1, D=2, Q=3, s=6)**
- **P=1**: Seasonal AR order
  - 1 seasonal lag öncesine bakar
- **D=2**: Seasonal differencing
  - 2 kez seasonal differencing
- **Q=3**: Seasonal MA order
  - 3 seasonal lag hatasına bakar
- **s=6**: Seasonal period
  - Her 6 zaman biriminde tekrar eden desen
  - Örnek: Aylık veri + yıllık sezonluk → s=12

**Ne zaman kullanılır?**
- Düzenli tekrarlayan desenler varsa (mevsimsellik)
- Örnek: Haftalık satış verileri (hafta sonu etkisi → s=7)
- Örnek: Aylık sıcaklık (yıllık döngü → s=12)

**SARIMAX vs SARIMA:**
- SARIMAX: Exogenous (dış) değişkenler eklenebilir
- Bu kodda dış değişken yok, sadece SARIMA

#### 3.6. Model Karşılaştırma

```python
plt.plot(y_pred_out, color='green', label='ARMA Predictions')   # Yeşil
plt.plot(y_pred_out, color='Blue', label='ARIMA Predictions')   # Mavi
plt.plot(y_pred_out, color='Pink', label='SARIMA Predictions')  # Pembe
plt.legend()
plt.show()
```

**Hangi model daha iyi?**
- RMSE'yi karşılaştır (düşük olan)
- Görsel olarak test verisine yakınlığa bak
- Seasonal pattern varsa → SARIMA
- Trend varsa ama sezonluk yoksa → ARIMA
- Stationary + basit → ARMA

## Gerekli Kütüphaneler

```python
# Veri İşleme
import pandas as pd
import numpy as np

# Görselleştirme
import matplotlib.pyplot as plt
import seaborn as sns

# Time Series Modelleri
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# Preprocessing
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

# Metrikler
from sklearn.metrics import mean_squared_error
```

## Önemli Kavramlar

### Stationarity (Durağanlık)
- Ortalama, varyans ve otokorelasyon zamana göre sabit
- ARMA için gerekli, ARIMA otomatik sağlar
- Test: Augmented Dickey-Fuller (ADF) testi

### Autocorrelation (Otokorelasyon)
- Zaman serisinin kendisiyle gecikmeli korelasyonu
- AR modeli için kritik
- ACF plot ile görselleştirilir

### Differencing
- Non-stationary → Stationary dönüşümü
- 1. derece: y'(t) = y(t) - y(t-1)
- 2. derece: y''(t) = y'(t) - y'(t-1)

### Seasonal Decomposition
- Trend + Seasonal + Residual
- Additive: y = T + S + R
- Multiplicative: y = T × S × R

## Model Seçim Kriterleri

| Model | Ne zaman kullan? |
|-------|------------------|
| **ARMA** | Stationary veri, sezonluk yok |
| **ARIMA** | Non-stationary, trend var, sezonluk yok |
| **SARIMA** | Non-stationary, hem trend hem sezonluk var |

## Performans Metrikleri

- **RMSE**: √MSE, orjinal birimde hata
- **MAE**: \|gerçek - tahmin\|, outlier'a robust
- **MAPE**: (MAE / gerçek) × 100, yüzde hata
- **AIC/BIC**: Model karmaşıklığı cezası, düşük olanı seç

## Çıktılar

1. **Veri tanımlaması**: `describe()` ile istatistikler
2. **Train/Test split grafiği**: Siyah (train), Kırmızı (test)
3. **Tahmin grafikleri**: Yeşil (ARMA), Mavi (ARIMA), Pembe (SARIMA)
4. **RMSE değerleri**: Her model için hata metriği
5. **Preprocessing grafikleri**: Her adımın sonucu

## Notlar

- Zaman serilerinde random train-test split YAPILMAZ
- Model order'ları (p,d,q) grid search ile optimize edilebilir
- Seasonal period (s), veri frekansına bağlıdır (günlük→7, aylık→12)
- Differencing, trend'i kaldırır ama bilgi kaybı yaratır
- Moving average, gecikme (lag) yaratır
