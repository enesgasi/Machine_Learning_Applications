# Lecture 6

## İçerik
Bu klasör, Clustering (Kümeleme) algoritmalarına odaklanan ders materyallerini içerir.

## Dosyalar
- **Lecture6.pdf**: Dersin teorik sunumu ve ders notları
- **Kmeans.py**: K-Means Clustering algoritması uygulaması
- **KNN.py**: K-Nearest Neighbors (KNN) algoritması uygulaması

## Konu Başlıkları

### 1. K-Means Clustering
Unsupervised learning algoritması olan K-Means, veri noktalarını benzerliklerine göre gruplara ayırır.

#### Özellikler
- **Elbow Method**: Optimal küme sayısını belirleme
- **WCSS (Within-Cluster Sum of Squares)**: Küme içi varyans
- **K-means++ Initialization**: Daha iyi başlangıç merkez noktaları
- **Iris Dataset** ile uygulama

#### Kod Özellikleri
```python
# Elbow Method ile optimal küme sayısı
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', 
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Model Eğitimi
kmeans = KMeans(n_clusters=3, init='k-means++', 
                max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)
```

#### Parametreler
- **n_clusters**: Küme sayısı (Iris için 3)
- **init='k-means++'**: Akıllı başlangıç merkez seçimi
- **max_iter=300**: Maksimum iterasyon sayısı
- **n_init=10**: Algoritmanın 10 kez çalıştırılıp en iyi sonucun seçilmesi

### 2. K-Nearest Neighbors (KNN)
Supervised learning algoritması olan KNN, sınıflandırma ve regresyon için kullanılır.

#### KNN Classifier (Sınıflandırma)
- **Iris Dataset** ile çiçek türü sınıflandırması
- **K değeri optimizasyonu**: 1'den 20'ye kadar farklı K değerlerini test etme
- **Error Rate ve Accuracy karşılaştırması**
- **StandardScaler** ile veri normalizasyonu

#### Kod Özellikleri
```python
# Veri Normalizasyonu
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# Model Eğitimi
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

# Optimal K değerini bulma
for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
```

#### KNN Regressor
- **Uniform** ve **Distance** ağırlıklı tahminler
- Sinüs fonksiyonu tahmini örneği
- n_neighbors=5 ile uygulama

```python
# Uniform ve Distance weights karşılaştırması
for weights in ["uniform", "distance"]:
    knn = neighbors.KNeighborsRegressor(n_neighbors=5, weights=weights)
    y_pred = knn.fit(X, y).predict(T)
```

## Gerekli Kütüphaneler
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_iris
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
```

## Veri Setleri
- **Iris Dataset**: 3 farklı iris çiçeği türünün özellikleri
  - Sepal Length (Çanak yaprağı uzunluğu)
  - Sepal Width (Çanak yaprağı genişliği)
  - Petal Length (Taç yaprağı uzunluğu)
  - Petal Width (Taç yaprağı genişliği)

## Görselleştirmeler

### K-Means
- Kümelerin scatter plot ile gösterimi
- Merkez noktaların (centroids) işaretlenmesi
- Farklı renklerle kümelerin ayrımı

### KNN
- Error Rate vs K değeri grafiği
- Accuracy vs K değeri grafiği
- Optimal K değerinin belirlenmesi

## Performans Metrikleri
- **Accuracy**: Doğru tahmin oranı
- **Error Rate**: Hata oranı
- **WCSS (Inertia)**: K-Means için küme içi varyans

## Önemli Notlar
- KNN'de veri normalizasyonu (StandardScaler) önemlidir
- K-Means'te elbow method ile optimal küme sayısı belirlenir
- Train-test split %80-20 oranındadır
- KNN'de K değeri arttıkça model daha az hassas olur
